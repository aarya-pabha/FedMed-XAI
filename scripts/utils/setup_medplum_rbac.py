import requests
import json
import os

class MedplumBootstrapper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.token = None

    def login(self, email, password):
        """Authenticates with the Medplum server."""
        url = f"{self.base_url}/auth/login"
        payload = {"email": email, "password": password}
        # VULN-302: Enforce TLS verification
        response = requests.post(url, json=payload, verify=True)
        
        if response.status_code != 200:
            raise Exception(f"Login failed: {response.text}")
            
        data = response.json()
        self.token = data.get("accessToken")
        return self.token

    def init_project(self, project_name):
        """Initializes a new FHIR project."""
        url = f"{self.base_url}/fhir/R4/Project/$init"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"name": project_name}
        # VULN-302: Enforce TLS verification
        response = requests.post(url, json=payload, headers=headers, verify=True)
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Project init failed: {response.text}")
            
        return response.json()

    def create_client_app(self, project_id, name):
        """Creates a ClientApplication for the project."""
        url = f"{self.base_url}/admin/projects/{project_id}/client"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"name": name}
        # VULN-302: Enforce TLS verification
        response = requests.post(url, json=payload, headers=headers, verify=True)
        
        if response.status_code not in [200, 201]:
            raise Exception(f"ClientApp creation failed: {response.text}")
            
        return response.json()

if __name__ == "__main__":
    # Default config for local dev/compose
    URL = os.getenv("MEDPLUM_URL", "http://localhost:8103")
    EMAIL = os.getenv("MEDPLUM_ADMIN_EMAIL", "admin@example.com")
    PASSWORD = os.getenv("MEDPLUM_ADMIN_PASSWORD", "medplum_admin")
    OUTPUT_FILE = "medplum_client_credentials.json"
    
    bootstrapper = MedplumBootstrapper(URL)
    try:
        print(f"🔐 Logging in to {URL}...")
        token = bootstrapper.login(EMAIL, PASSWORD)
        
        print("🏗️ Initializing project 'FL_Diagnostics_Study'...")
        project = bootstrapper.init_project("FL_Diagnostics_Study")
        project_id = project["id"]
        
        print(f"📱 Creating ClientApplication for federated nodes...")
        client_app = bootstrapper.create_client_app(project_id, "Federated_Client_Node")
        
        # VULN-301: Prevent plaintext secret leakage in logs. Write to restricted file instead.
        creds = {
            "projectId": project_id,
            "clientId": client_app["id"],
            "clientSecret": client_app.get("secret", "")
        }
        
        with open(OUTPUT_FILE, "w") as f:
            json.dump(creds, f, indent=4)
        
        # Set restricted permissions (Read/Write for owner only)
        if os.name != 'nt': # Linux/Unix
            os.chmod(OUTPUT_FILE, 0o600)
            
        print(f"\n✅ Bootstrapping Complete!")
        print(f"Credentials written to: {OUTPUT_FILE}")
        print("Note: Secret is HIDDEN from logs for HIPAA compliance.")
        
    except Exception as e:
        print(f"❌ Error: {e}")


