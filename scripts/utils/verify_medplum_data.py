import requests
import json
import os

MEDPLUM_URL = "https://medplum-server-789083630799.us-central1.run.app"

def verify_medplum_data():
    creds_path = "medplum_client_credentials.json"
    if not os.path.exists(creds_path):
        print(f"❌ Error: {creds_path} not found.")
        return

    with open(creds_path, "r") as f:
        creds = json.load(f)
    
    client_id = creds.get("clientId")
    client_secret = creds.get("clientSecret")

    if not client_id or not client_secret:
        print("❌ Error: Missing credentials in JSON.")
        return

    print("🔐 Authenticating with Medplum...")
    token_url = f"{MEDPLUM_URL}/oauth2/token"
    auth_payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    
    token_resp = requests.post(token_url, data=auth_payload, verify=True)
    if token_resp.status_code != 200:
        print(f"❌ Auth Failed: {token_resp.text}")
        return
        
    access_token = token_resp.json().get("access_token")
    
    print("📥 Fetching recent Observations from Medplum...")
    obs_url = f"{MEDPLUM_URL}/fhir/R4/Observation?_sort=-_lastUpdated&_count=10"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/fhir+json"
    }
    
    obs_resp = requests.get(obs_url, headers=headers, verify=True)
    if obs_resp.status_code != 200:
        print(f"❌ Fetch Failed: {obs_resp.text}")
        return
        
    bundle = obs_resp.json()
    entries = bundle.get("entry", [])
    
    if not entries:
        print("⚠️ No Observations found in Medplum.")
        return
        
    print(f"✅ Found {len(entries)} recent Observations:\n")
    for entry in entries:
        obs = entry.get("resource", {})
        components = obs.get("component", [])
        
        round_num, loss, dice, epsilon = "N/A", "N/A", "N/A", "N/A"
        
        for comp in components:
            code_text = comp.get("code", {}).get("text", "")
            value = comp.get("valueQuantity", {}).get("value")
            if "Round" in code_text: round_num = value
            elif "Loss" in code_text: loss = value
            elif "Dice" in code_text: dice = value
            elif "Epsilon" in code_text: epsilon = value
            
        print(f"Observation ID: {obs.get('id')}")
        print(f"  Round: {round_num}")
        print(f"  Loss: {loss}")
        print(f"  Dice: {dice}")
        print(f"  Epsilon: {epsilon}")
        print("-" * 30)

if __name__ == "__main__":
    verify_medplum_data()
