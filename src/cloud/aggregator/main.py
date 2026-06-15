import functions_framework
from cloudevents.http import CloudEvent
from google.cloud import storage, firestore
from safetensors.torch import load as st_load, save as st_save
import torch
import io
import os

# Configuration
TOTAL_CLIENTS = 3 
PROJECT_ID = os.getenv("GCP_PROJECT", "healthcare-fl-diagnostics")

@functions_framework.cloud_event
def aggregate_shards(cloud_event: CloudEvent):
    """
    Triggers on Firestore write to a shard upload record.
    Uses a transaction to ensure only one execution performs the aggregation.
    Performs memory-efficient iterative FedAvg and validates tensor integrity.
    """
    db = firestore.Client(project=PROJECT_ID)

    # Parse firestore document path from event (2nd Gen / Eventarc style)
    doc_path = None
    if isinstance(cloud_event.data, dict):
        doc_path = cloud_event.data.get("value", {}).get("name")
    
    if not doc_path:
        # Fallback for different Eventarc versions
        doc_path = cloud_event.get("document")

    if not doc_path:
        print(f"Error: Could not parse document path from cloud_event. Data: {cloud_event.data}")
        return False

    parts = doc_path.split('/')
    try:
        round_index = parts.index('rounds')
        shard_index = parts.index('shards')
        round_id = parts[round_index + 1]
        shard_id = parts[shard_index + 1]
    except (ValueError, IndexError):
        print(f"Error: Could not find round/shard in document path: {doc_path}")
        return False

    shard_ref = db.collection('rounds').document(round_id).collection('shards').document(shard_id)

    @firestore.transactional
    def prepare_aggregation(transaction, shard_ref):
        snapshot = shard_ref.get(transaction=transaction)
        status = snapshot.get("status") if snapshot.exists else None

        if status in ["AGGREGATING", "AGGREGATED"]:
            print(f"Aborting. Shard {shard_id} is already {status}.")
            return False

        uploads_ref = shard_ref.collection('uploads')
        uploaded_docs = list(uploads_ref.get(transaction=transaction))
        count = len(uploaded_docs)

        if count < TOTAL_CLIENTS:
            print(f"Waiting. {count}/{TOTAL_CLIENTS} clients uploaded for Round {round_id}, Shard {shard_id}.")
            return False

        # Atomic lock: Mark as AGGREGATING
        transaction.set(shard_ref, {
            "status": "AGGREGATING",
            "lock_acquired_at": firestore.SERVER_TIMESTAMP
        }, merge=True)
        return uploaded_docs

    # Execute Transaction
    transaction = db.transaction()
    uploaded_docs = prepare_aggregation(transaction, shard_ref)

    if not uploaded_docs:
        return False

    print(f"Lock acquired. Starting memory-efficient aggregation for {len(uploaded_docs)} shards...")

    storage_client = storage.Client(project=PROJECT_ID)
    bucket_in = storage_client.bucket(f"{PROJECT_ID}-fl-shards-in")
    bucket_out = storage_client.bucket(f"{PROJECT_ID}-fl-shards-out")

    avg_tensor_dict = None
    
    # VULN-405: Iterative accumulation to prevent OOM
    for client_doc in uploaded_docs:
        client_id = client_doc.id
        blob_name = f"round_{round_id}/client_{client_id}/shard_{shard_id}.safetensors"
        blob = bucket_in.blob(blob_name)

        if not blob.exists():
            print(f"Error: Blob {blob_name} missing. Aggregation incomplete.")
            shard_ref.update({"status": "FAILED", "error": f"Missing blob {blob_name}"})
            return False

        byte_stream = io.BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)

        try:
            # Load single shard
            raw_bytes = byte_stream.read()
            shard_dict = st_load(raw_bytes)
            
            # VULN-407: Mathematical Integrity Validation
            for key, tensor in shard_dict.items():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Error: Malformed tensor detected in {blob_name} (NaN/Inf) at {key}.")
                    shard_ref.update({"status": "FAILED", "error": f"Data poisoning detected in {client_id}"})
                    return False

            # Running average update: avg = sum(w_i) / N
            if avg_tensor_dict is None:
                avg_tensor_dict = {k: v / TOTAL_CLIENTS for k, v in shard_dict.items()}
            else:
                for k in shard_dict.keys():
                    avg_tensor_dict[k] += shard_dict[k] / TOTAL_CLIENTS
            
            del shard_dict

        except Exception as e:
            print(f"Error: Failed to process shard from {blob_name}. {e}")
            shard_ref.update({"status": "FAILED", "error": f"Process error in {client_id}"})
            return False

    # Upload global shard to the NEXT round's path
    out_stream = io.BytesIO()
    raw_out = st_save(avg_tensor_dict)
    out_stream.write(raw_out)
    out_stream.seek(0)

    next_round = int(round_id) + 1
    global_blob_name = f"round_{next_round}/global/shard_{shard_id}.safetensors"
    bucket_out.blob(global_blob_name).upload_from_file(out_stream)

    # Finalize state
    shard_ref.set({
        "status": "AGGREGATED",
        "aggregated_at": firestore.SERVER_TIMESTAMP
    }, merge=True)

    print(f"Successfully aggregated and uploaded global shard: {global_blob_name}")
    return True

