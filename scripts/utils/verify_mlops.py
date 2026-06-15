import mlflow
from mlflow.tracking import MlflowClient

def list_runs():
    tracking_uri = "https://mlflow-server-789083630799.us-central1.run.app"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    print("Listing last 5 runs in experiment 'Federated_IXI_Simulation'...")
    try:
        exp = client.get_experiment_by_name("Federated_IXI_Simulation")
        if not exp:
            print("Experiment not found!")
            return
            
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=5)
        for run in runs:
            print(f"\nRun ID: {run.info.run_id}")
            print(f"Name: {run.data.tags.get('mlflow.runName', 'Unnamed')}")
            print(f"Status: {run.info.status}")
            print(f"Params: {len(run.data.params)}")
            print(f"Metrics: {len(run.data.metrics)}")
            if len(run.data.metrics) > 0:
                print(f"Sample Metrics: {list(run.data.metrics.keys())[:3]}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_runs()
