import mlflow
import subprocess
import sys
import time
import os
from mlflow.tracking import MlflowClient

def test_mlflow_mp_logging():
    tracking_uri = "https://mlflow-server-789083630799.us-central1.run.app"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("TDD_MLflow_MP_Test")
    
    # Clean up previous runs for this test experiment to ensure fresh results
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name("TDD_MLflow_MP_Test")
    if exp:
        runs = client.search_runs(exp.experiment_id)
        for r in runs:
            client.delete_run(r.info.run_id)

    print("🚀 Starting Parent Run...")
    with mlflow.start_run(run_name="Parent_Orchestrator") as run:
        parent_run_id = run.info.run_id
        print(f"   Parent Run ID: {parent_run_id}")
        
        # Simulating the worker launch
        print("🚀 Launching Subprocess Worker...")
        # Current logic: worker just tries to log without knowing about parent
        worker_code = f"""
import mlflow
import os
mlflow.set_tracking_uri("{tracking_uri}")
mlflow.set_experiment("TDD_MLflow_MP_Test")
# This worker DOES NOT know about the parent run id
mlflow.log_metric("worker_metric", 1.0)
print("Worker: Logged metric")
"""
        with open("temp_worker.py", "w") as f:
            f.write(worker_code)
            
        result = subprocess.run([sys.executable, "temp_worker.py"], capture_output=True, text=True)
        print(f"   Worker Output: {result.stdout}")
        print(f"   Worker Error: {result.stderr}")
        
        # Verify metric in parent run
        time.sleep(2) # Give server time to index
        updated_run = client.get_run(parent_run_id)
        metrics = updated_run.data.metrics
        
        print(f"   Metrics in Parent Run: {metrics}")
        
        if "worker_metric" not in metrics:
            print("❌ FAILURE: Worker metric not found in Parent Run.")
            # We also check if it created a NEW run instead
            runs = client.search_runs(exp.experiment_id)
            print(f"   Total runs found: {len(runs)}")
            for r in runs:
                if r.info.run_id != parent_run_id:
                    print(f"   ⚠️ Found orphaned run: {r.info.run_id} with name {r.info.run_name}")
            
            return False
        else:
            print("✅ SUCCESS: Worker metric found in Parent Run.")
            return True

if __name__ == "__main__":
    if not test_mlflow_mp_logging():
        sys.exit(1)
