import mlflow
import subprocess
import sys
import time
import os
from mlflow.tracking import MlflowClient

def test_mlflow_mp_fix():
    tracking_uri = "https://mlflow-server-789083630799.us-central1.run.app"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("TDD_MLflow_MP_Test")
    
    # Clean up
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name("TDD_MLflow_MP_Test")
    if exp:
        runs = client.search_runs(exp.experiment_id)
        for r in runs:
            client.delete_run(r.info.run_id)

    print("🚀 Starting Parent Run...")
    with mlflow.start_run(run_name="Parent_Orchestrator_Fixed") as run:
        parent_run_id = run.info.run_id
        print(f"   Parent Run ID: {parent_run_id}")
        
        # Simulating the worker launch with FIX: passing parent_run_id
        print("🚀 Launching Subprocess Worker with FIX...")
        worker_code = f"""
import mlflow
from mlflow.tracking import MlflowClient
import os

mlflow.set_tracking_uri("{tracking_uri}")
client = MlflowClient()

# FIX: Explicitly log to the parent run_id
client.log_metric("{parent_run_id}", "worker_metric_fixed", 1.0)
print("Worker: Logged metric to parent")
"""
        with open("temp_worker_fixed.py", "w") as f:
            f.write(worker_code)
            
        result = subprocess.run([sys.executable, "temp_worker_fixed.py"], capture_output=True, text=True)
        print(f"   Worker Output: {result.stdout}")
        print(f"   Worker Error: {result.stderr}")
        
        # Verify metric in parent run
        time.sleep(3) # Give server time to index
        updated_run = client.get_run(parent_run_id)
        metrics = updated_run.data.metrics
        
        print(f"   Metrics in Parent Run: {metrics}")
        
        if "worker_metric_fixed" in metrics:
            print("✅ SUCCESS: Worker metric found in Parent Run.")
            # Verify no orphaned runs were created
            all_runs = client.search_runs(exp.experiment_id)
            if len(all_runs) == 1:
                print("✅ SUCCESS: No orphaned runs created.")
                return True
            else:
                print(f"❌ FAILURE: Found {len(all_runs)} runs, expected only 1.")
                return False
        else:
            print("❌ FAILURE: Worker metric not found in Parent Run.")
            return False

if __name__ == "__main__":
    if not test_mlflow_mp_fix():
        sys.exit(1)
