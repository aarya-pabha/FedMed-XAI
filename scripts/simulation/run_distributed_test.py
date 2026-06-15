import argparse
import os
import time
import mlflow
import torch
from google.cloud import storage
from typing import List
import numpy as np
from src.federated.client import FlowerClient

def get_args():
    parser = argparse.ArgumentParser(description="Phase 5: Distributed Sequential Orchestrator")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of federated rounds to run")
    parser.add_argument("--dry-run", action="store_true", help="Use tiny model and mock data")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    return parser.parse_args()

def main():
    args = get_args()
    
    # GCP Environment setup
    project_id = os.getenv("GCP_PROJECT", "healthcare-fl-diagnostics")
    
    if not args.no_mlflow:
        tracking_uri = "https://mlflow-server-789083630799.us-central1.run.app"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Distributed_Execution_Test")
        mlflow.start_run(run_name=f"Sequential_Cloud_Test_{int(time.time())}")
        mlflow.log_params(vars(args))
        
    try:
        for round_num in range(1, args.num_rounds + 1):
            print(f"\n🚀 === STARTING ROUND {round_num} ===")
            
            # Sequentially run each client to respect VRAM (T4 16GB limit)
            for center_id in range(3):
                print(f"\n🏥 --- Hospital {center_id} Session ---")
                
                # Initialize client (loads model into VRAM)
                client = FlowerClient(
                    center_id=center_id, 
                    dry_run=args.dry_run,
                    batch_size=1
                )
                
                # 1. Pull Global Model (from GCS)
                # For Round 1, this requires the Seeder to have run.
                # For Round >1, this requires the Cloud Function to have finished.
                print(f"📥 Pulling global model for Round {round_num}...")
                global_params = client._pull_global_shard(round_num)
                
                # 2. Local Training & Shard Upload
                # fit() now handles training, Opacus DP, GCS upload, and Medplum logging.
                config = {"round": round_num, "lr": 5e-5}
                print(f"⚙️ Executing local fit and cloud synchronization...")
                _, _, metrics = client.fit(global_params, config)
                
                # MLflow Per-Client Metrics
                if not args.no_mlflow:
                    mlflow.log_metric(f"round_{round_num}_client_{center_id}_loss", metrics["loss"])
                    mlflow.log_metric(f"round_{round_num}_client_{center_id}_eps", metrics["epsilon"])
                
                print(f"✅ Hospital {center_id} complete. Loss: {metrics['loss']:.4f}, Epsilon: {metrics['epsilon']:.2f}")
                
                # 3. Clean up VRAM for next hospital
                del client
                torch.cuda.empty_cache()
                time.sleep(2) # Brief cooldown for GCP propagation
            
            print(f"\n🏁 Round {round_num} Clients finished. Cloud Aggregator should be triggering...")
            
        print("\n📥 Downloading final aggregated model from cloud...")
        final_round = args.num_rounds + 1
        storage_client = storage.Client(project=project_id)
        bucket_out = storage_client.bucket(f"{project_id}-fl-shards-out")
        final_blob_name = f"round_{final_round}/global/shard_0.safetensors"
        final_blob = bucket_out.blob(final_blob_name)
        
        # Poll for the final model
        timeout = 600
        start_time = time.time()
        while not final_blob.exists():
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timed out waiting for final model.")
            time.sleep(10)
            final_blob.reload()
            
        final_blob.download_to_file(open("final_diagnostic_model.safetensors", "wb"))
        print(f"\n✨ Distributed Training Successful! Final model saved to: final_diagnostic_model.safetensors")

    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        raise
    finally:
        if not args.no_mlflow:
            mlflow.end_run()

if __name__ == "__main__":
    main()
