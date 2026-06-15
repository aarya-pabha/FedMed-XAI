import mlflow
import os

def check_mlflow():
    uri = "https://mlflow-server-789083630799.us-central1.run.app"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("Distributed_Execution_Test")
    with mlflow.start_run(run_name="L4_Preflight_Check"):
        mlflow.log_param("preflight", "success")
        mlflow.log_metric("connection_verified", 1.0)
    print("SUCCESS: Logged to MLflow")

if __name__ == "__main__":
    check_mlflow()
