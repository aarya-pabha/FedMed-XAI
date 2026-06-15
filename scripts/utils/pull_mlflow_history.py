from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd

def pull_full_metrics():
    tracking_uri = "https://mlflow-server-789083630799.us-central1.run.app"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # 1. Get the latest run
    runs = mlflow.search_runs(experiment_names=["Distributed_Execution_Test"], order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        print("No runs found.")
        return
    
    run_id = runs.iloc[0].run_id
    print(f"Pulling full history for Run ID: {run_id}\n")
    
    # 2. Identify all metrics logged
    run = client.get_run(run_id)
    metric_keys = run.data.metrics.keys()
    
    # 3. Pull history for each metric
    all_data = {}
    for key in metric_keys:
        history = client.get_metric_history(run_id, key)
        # Store as (step, value)
        all_data[key] = [(m.step, round(m.value, 4)) for m in history]
    
    # 4. Display results clearly
    for key, values in all_data.items():
        print(f"Metric: {key}")
        print(f"  History: {values}")
        print("-" * 30)

if __name__ == "__main__":
    pull_full_metrics()
