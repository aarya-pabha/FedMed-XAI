import mlflow
import requests
import sys
import socket

def test_mlflow():
    tracking_uri = "https://mlflow-server-789083630799.us-central1.run.app"
    print(f"1. Testing basic network reachability (timeout=10s)...")
    
    try:
        # Check if we can even hit the health endpoint/root
        response = requests.get(tracking_uri, timeout=10)
        print(f"   - Network Response: {response.status_code}")
    except requests.exceptions.Timeout:
        print("   - FAILURE: Network timeout. The server is unreachable from your network.")
        return False
    except Exception as e:
        print(f"   - FAILURE: Network error: {e}")
        return False

    print(f"2. Testing MLflow SDK connection...")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        # Force a network call with a smaller internal timeout if possible
        # Note: MLflow client doesn't expose a simple timeout for search_experiments
        exps = mlflow.search_experiments()
        print(f"   - SUCCESS! Found {len(exps)} experiments.")
        return True
    except Exception as e:
        print(f"   - FAILURE: MLflow SDK error: {e}")
        return False

if __name__ == "__main__":
    if not test_mlflow():
        sys.exit(1)
