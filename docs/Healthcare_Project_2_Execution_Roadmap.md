# Healthcare Project 2: Master Execution & Architectural Roadmap

This document serves as the consolidated source of truth for the project's execution trajectory, incorporating the validated PRD requirements, compliance safeguards, and MLOps best practices.

## ⚠️ Architectural Analysis: Corrections to the Initial Plan

Upon architectural review of the initial "Next Steps", several critical sequencing errors and omissions were identified and fixed in this master roadmap:

1.  **Missing Compute Layer in Cloud Provisioning:** The original steps provisioned Google Cloud Storage (GCS) and Firestore but completely missed the deployment of the **Google Cloud Functions** required to execute the `GradsSharding` FedAvg mathematical aggregation.
2.  **Incorrect Sequencing of Compliance Backend:** Medplum was previously slated as the final step (Step 4). However, because the PRD mandates that *all* interactions and API calls must be routed through Medplum for immutable audit logging, the compliance backend must be provisioned *before* the federated cloud architecture is active.
3.  **Missing MLOps Tracking:** Data Version Control (DVC) was included, but **MLflow** setup was omitted from the execution steps. MLflow is mandatory for logging the Differential Privacy $\epsilon$ vs. Accuracy trade-off curve.
4.  **Missing Integration Testing (Mock Cloud):** Deploying directly to GCP serverless functions from a local Colab setup without an intermediate mocking phase risks burning through your $300 GCP budget on syntax errors or out-of-memory crashes.

---

## 🛠️ The Corrected Execution Roadmap

### Phase 1: MLOps Foundation & Data Pipeline (COMPLETED & AUDITED)
1.  **Dataset Initialization:** Pull and patch FLamby (Fed-IXI Tiny variant) for local loadability and MONAI 1.3+ compatibility.
2.  **Data Provenance (DVC):** Initialize Data Version Control with GCP remote storage.
3.  **Experiment Tracking (MLflow):** Deployed centralized MLflow Tracking Server on Google Cloud Run with PostgreSQL backend and GCS artifact store.
4.  **Security Remediation:** Scrubbed PHI from NIfTI headers, enabled GCS Data Access Audit Logs, and enforced Public Access Prevention (PAP).

### Phase 2: Model & Privacy Core (Local Simulation)
1.  **Architecture Initialization:** Implement the PyTorch DynUNet baseline (nnU-Net optimized) to remediate identified anisotropy in 3D brain segmentation.
2.  **Privacy Engine Wrapping:** Integrate the `Opacus` library. Configure the `PrivacyEngine` to enforce per-sample gradient clipping ($C$) and Gaussian noise injection ($\mathcal{N}$).
3.  **Local Federated Simulation:** Utilize Flower (FLWR) v1.28's `SuperLink` to simulate the multi-client network within a persistent Google Compute Engine Deep Learning VM. Verify that the DynUNet model converges under cumulative differential privacy constraints locally.
4.  **Regulatory Clinical EDA (QA):** Perform multi-domain audit (SNR, HD95, UMAP Site-Bias) to verify data readiness and satisfy FDA GMLP before Phase 3 deployment.

### Phase 3: Compliance & Security Orchestration (Medplum Control Plane)
1.  **Medplum API Deployment:** Deploy the Medplum compute layer via Docker Compose (utilizing live Cloud SQL and Redis provisioned in Phase 1) to bypass serverless ESM limitations.
2.  **RBAC Configuration:** Define strictly scoped Role-Based Access Control policies for your simulated "hospitals" and the central aggregator.
3.  **Audit Logging Integration:** Transition from Infrastructure-level GCS logs to Application-level Medplum audit logs for FHIR-native compliance.

### Phase 4: Serverless Cloud Infrastructure
1.  **Zero-Trust Storage:** Provision Google Cloud Storage buckets. Enforce AES-256 encryption at rest and strict IAM roles limiting access solely to the aggregation service account.
2.  **State Tracking (Firestore):** Deploy a Google Cloud Firestore instance to track the asynchronous uploads of gradient shards, solving the distributed race condition.
3.  **Cloud Functions Deployment:** Write and deploy the Python-based Google Cloud Functions. These functions will be triggered by Firestore state changes to perform the Federated Averaging (FedAvg) on individual $w_s$ shards.

### Phase 5: Distributed Integration & Execution
1.  **GradsSharding Implementation:** Write the client-side logic to chop the massive DynUNet gradient tensors into contiguous shards and upload them via HTTP PUT (TLS 1.3).
2.  **End-to-End Training:** Execute the full distributed training loop. GCE nodes train locally -> add Opacus noise -> shard gradients -> upload to GCS -> Firestore triggers Cloud Function -> FedAvg applied -> clients pull reconstructed shards.
3.  **Straggler Handling Verification:** Intentionally disconnect a training node and verify that the synchronization protocol functions correctly without hanging the cloud aggregator.

### Phase 6: Explainable AI & Final Reporting
1.  **Grad-CAM Integration:** Run the finalized global model on local test images and generate Grad-CAM spatial heatmaps to highlight segmented brain anomalies. (Note: SHAP integration for EHR data was scoped out as the focus remained on imaging data).
2.  **Trade-Off Analysis:** Extract MLflow data to plot the final Diagnostic Accuracy (Dual-Dice) against the Differential Privacy $\epsilon$ budget. Ensure degradation is $\le 5\%$.
