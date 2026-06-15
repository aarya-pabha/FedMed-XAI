import argparse
import os
import io
import torch
import numpy as np
from google.cloud import storage
from safetensors.numpy import save as st_save
from src.models.unet_pp import get_model

def seed_global_model(dry_run=False):
    # Use the project ID from environment or default
    project_id = os.getenv("GCP_PROJECT", "healthcare-fl-diagnostics")
    storage_client = storage.Client(project=project_id)
    bucket_out = storage_client.bucket(f"{project_id}-fl-shards-out")
    
    if dry_run:
        print("Initializing Tiny Mock Model for Dry Run...")
    else:
        print("Initializing Global Model for Production Stability Test...")
        
    model = get_model(dry_run=dry_run) 
    
    # Extract params as numpy arrays
    target = model._module if hasattr(model, "_module") else model
    parameters = [val.detach().cpu().numpy() for _, val in target.state_dict().items()]
    
    # Serialize with safetensors
    params_dict = {f"arr_{i}": arr for i, arr in enumerate(parameters)}
    raw_bytes = st_save(params_dict)
    byte_stream = io.BytesIO(raw_bytes)
    
    # Upload as Round 1 starting global model
    blob_name = f"round_1/global/shard_0.safetensors"
    bucket_out.blob(blob_name).upload_from_file(byte_stream)
    print(f"✅ Seeded initial global model to gs://{bucket_out.name}/{blob_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seed_global_model(dry_run=args.dry_run)
