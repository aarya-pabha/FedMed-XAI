const { MedplumClient } = require("@medplum/core");

const crypto = require("crypto");
globalThis.window = { 
  crypto: crypto.webcrypto,
  btoa: btoa,
  atob: atob,
  TextDecoder: TextDecoder,
  location: { protocol: "http:", origin: "http://localhost", host: "localhost" },
  sessionStorage: (() => {
    const store = {};
    return {
      getItem: (key) => store[key] || null,
      setItem: (key, value) => { store[key] = value; },
      removeItem: (key) => { delete store[key]; }
    };
  })()
};
globalThis.location = globalThis.window.location;

const medplum = new MedplumClient({ 
  baseUrl: "https://medplum-server-789083630799.us-central1.run.app", 
  fetch: fetch 
});

async function run() {
  try {
    const login = await medplum.startLogin({ email: "admin@example.com", password: "medplum_admin" });
    console.log("Login result:", login);
    const profile = await medplum.processCode(login.code);
    console.log("Profile:", profile.id);
    
    // Now that we're logged in, let's create a project
    console.log("Creating Project...");
    const project = await medplum.post("fhir/R4/Project/$init", { name: "FL_Diagnostics_Study" });
    console.log("Project created:", project.id);
    
    // And create a ClientApplication
    console.log("Creating ClientApplication...");
    const clientApp = await medplum.post(`admin/projects/${project.id}/client`, { name: "Federated_Client_Node" });
    console.log("ClientApplication created:", clientApp.id);
    
    // Save them to medplum_client_credentials.json
    const fs = require("fs");
    const creds = {
        projectId: project.id,
        clientId: clientApp.id,
        clientSecret: clientApp.secret
    };
    fs.writeFileSync("medplum_client_credentials.json", JSON.stringify(creds, null, 2));
    console.log("Credentials saved to medplum_client_credentials.json!");

  } catch (err) {
    console.error("Error details:", err);
  }
}
run();