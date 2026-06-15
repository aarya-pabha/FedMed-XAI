# src/utils/medplum_audit.py
import requests
import os

MEDPLUM_URL = os.getenv("MEDPLUM_URL", "https://medplum-server-789083630799.us-central1.run.app")

def log_metrics_to_medplum(client_id, client_secret, project_id, round_num, loss, dice, epsilon):
    """Logs federated training metrics as a FHIR Observation to Medplum."""
    # 1. Get OAuth Token
    token_url = f"{MEDPLUM_URL}/oauth2/token"
    auth_payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    
    token_resp = requests.post(token_url, data=auth_payload, verify=True)
    if token_resp.status_code != 200:
        print(f"Failed to get Medplum token: {token_resp.text}")
        return False
        
    access_token = token_resp.json().get("access_token")
    
    # 2. Create Observation
    obs_url = f"{MEDPLUM_URL}/fhir/R4/Observation"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/fhir+json"
    }
    
    components = [
        {"code": {"text": "Round"}, "valueInteger": round_num},
        {"code": {"text": "Loss"}, "valueQuantity": {"value": float(loss)}},
        {"code": {"text": "Epsilon"}, "valueQuantity": {"value": float(epsilon)}}
    ]
    if float(dice) != 0.0:
        components.append({"code": {"text": "Dice"}, "valueQuantity": {"value": float(dice)}})
    
    observation = {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "55284-4", "display": "Machine Learning Metrics"}]
        },
        "component": components
    }
    
    obs_resp = requests.post(obs_url, json=observation, headers=headers, verify=True)
    if obs_resp.status_code != 201:
        print(f"Failed to create Observation: {obs_resp.text}")
        return False
        
    return True
