import mlflow
from mlflow.tracking import MlflowClient

def fetch_metrics():
    client = MlflowClient(tracking_uri="https://mlflow-server-789083630799.us-central1.run.app")
    run_id = "ca4130e9b81a40f7b7c7365ee3521150"
    
    print(f"--- MLflow Run ID: {run_id} ---")
    
    for c in range(3):
        loss_hist = client.get_metric_history(run_id, f"client_{c}_train_loss")
        eps_hist = client.get_metric_history(run_id, f"client_{c}_cumulative_epsilon")
        
        losses = [h.value for h in loss_hist]
        epsilons = [h.value for h in eps_hist]
        
        print(f"Hospital {c}:")
        print(f"  Loss History: {losses}")
        print(f"  Epsilon History: {epsilons}")
        print("-" * 30)

if __name__ == "__main__":
    fetch_metrics()
