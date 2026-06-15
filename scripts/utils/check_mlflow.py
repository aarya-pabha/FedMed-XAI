import urllib.request
import json

req = urllib.request.Request("https://mlflow-server-789083630799.us-central1.run.app/api/2.0/mlflow/experiments/search", data=b'{"max_results": 10}', headers={'Content-Type': 'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req) as f:
        print(json.dumps(json.loads(f.read().decode('utf-8')), indent=2))
except Exception as e:
    print(f"Error: {e}")
