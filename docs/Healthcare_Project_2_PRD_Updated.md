# Product Requirements Document (PRD)
**Project Name:** Privacy-Preserving Federated Learning for Medical Image Diagnostics
**Document Owner:** [Your Name]
**Target Audience:** Engineering Hiring Managers, Data Science Leads, Risk Officers 

## 1. Executive Summary
The siloing of medical data due to stringent privacy regulations (HIPAA) prevents the centralized aggregation of the large datasets required to train highly accurate diagnostic deep learning models. This project solves this systemic issue by building a decentralized machine learning network. It allows isolated clinical nodes to collaboratively train a heavy-duty diagnostic model (U-Net++) on medical imagery without ever transferring raw patient data outside their local environments. 

This project explicitly demonstrates enterprise-grade software engineering, advanced privacy-preserving mathematics, and cost-effective cloud resource management tailored for the highly regulated healthcare sector.

## 2. Core Objectives & Success Metrics
* **Clinical Utility Goal:** Achieve high diagnostic localization accuracy for 3D brain segmentation.
    * *Metric:* Maximize Dice Coefficient and Intersection Over Union (IoU) on the FLamby test set without violating privacy bounds. (Achieved ~31.6% Dice Overlap in initial wide-model 5-round bounds).
* **Privacy & Compliance Goal:** Guarantee that no patient data can be reverse-engineered from model weight updates.
    * *Metric:* Maintain a Differential Privacy budget of $\epsilon \le 10.0$ and $\delta \le 10^{-5}$ while minimizing accuracy degradation. This trade-off must be documented programmatically via Dual-Dice evaluations.
* **Infrastructure Efficiency Goal:** Execute federated aggregations of massive gradient tensors without utilizing a persistent central server.
    * *Metric:* Zero out-of-memory (OOM) errors during aggregation and strict utilization of a serverless, pay-per-execution billing model (GCP Functions).

## 3. Technology Stack 
* **Federated Framework:** Flower (FLWR) (v1.28) transitioning to Serverless GradsSharding.
* **Dataset:** FLamby (Fed-IXI Tiny variant).
* **Model Architecture:** 32-Channel Wide U-Net++ with GroupNorm(8).
* **Privacy Engine:** Opacus (Local Differential Privacy).
* **Explainable AI (XAI):** Grad-CAM.
* **Cloud Infrastructure:** Google Cloud Storage, Google Cloud Functions, & Google Cloud Firestore (State Tracking).
* **Compute Environment:** Google Compute Engine (g2-standard-12 VM with NVIDIA L4 GPU, 24GB VRAM).
* **Compliance Backend:** Medplum (FHIR-native, HIPAA-compliant RBAC and audit logging).
* **MLOps & Provenance:** Centralized MLOps Engine (MLflow on Cloud Run + PostgreSQL + GCS) and Data Version Control (DVC).

## 4. Functional Requirements
### 4.1. Local Training Execution & MLOps (Client Nodes)
* The system must simulate multi-institutional clients (using a dedicated GCE VM with an NVIDIA L4 GPU, replacing the initial Google Colab Pro plan due to environment limitations).
* Each client must train a localized Wide U-Net++ model (32-channel base, GroupNorm(8)) on a partitioned segment of the FLamby dataset.
* The system must generate interpretability heatmaps using Grad-CAM to localize anomalies.
* **Data Provenance:** The dataset must be versioned using Data Version Control (DVC) to ensure full reproducibility.
* **Mandatory Data De-identification:** Programmatic scrubbing of NIfTI headers using `nibabel` before DVC tracking is a prerequisite for all clinical data.
* **Experiment Tracking:** The local training loops must integrate MLflow to track hyperparameter tuning, record the $\epsilon$ privacy budgets, and log Dual-Dice (Global vs. Local) performance metrics across iterations.

### 4.2. Privacy Engine Injection
* The local training loop must wrap PyTorch components in the Opacus `PrivacyEngine`.
* The system must apply per-sample gradient clipping and inject Gaussian noise before preparing the gradients for transmission.

### 4.3. Cloud Communication & GradsSharding
* Clients must partition the massive, noisy gradient vector into smaller contiguous shards to bypass serverless memory limits.
* Clients must upload shards via HTTP PUT to Google Cloud Storage.
* Clients must download averaged shards from GCP and concatenate them to reconstruct the updated global model.

### 4.4. Serverless Aggregation & Reliability
* **State Tracking:** Upon a client uploading a shard to Google Cloud Storage, a lightweight state-tracking mechanism in Google Cloud Firestore must increment the shard tally for the given round $t$ and shard index $s$.
* **Race Condition Mitigation:** The Google Cloud Function performing Federated Averaging (FedAvg) must only execute when the Firestore tally indicates that all required shards for that index have been uploaded.
* **Stragglers & Failure Handling:** The system must implement a timeout protocol for straggling clients. If a client disconnects, the central serverless orchestrator will proceed with aggregation if at least 80% of clients have successfully uploaded their shards within the defined time window, dropping the stragglers to prevent infinite hangs.

### 4.5. Compliance Orchestration (Phased Deployment)
The system utilizes a dual-layer compliance strategy to ensure HIPAA-ready data handling:
* **Infrastructure Layer (Complete):** Google Cloud Storage is configured with Data Access Audit Logs (DATA_READ/WRITE), Public Access Prevention (PAP), and Object Versioning to prevent tampering.
* **Application Layer (Scheduled):** All interactions, data access, and API calls will be routed through the Medplum backend. Medplum will generate immutable, tamper-evident audit logs for fine-grained FHIR-native compliance (scheduled for Phase 3).

### 4.6. Security & Infrastructure Safeguards
* **Encryption Standards:** All communication channels must utilize TLS 1.2+ (with a strict architectural preference for TLS 1.3 as per 2026 NIST/HIPAA modernization guidelines) for data-in-transit security.

* **Zero-Trust IAM Constraints:** Google Cloud Storage buckets must be configured as strictly private. Identity and Access Management (IAM) roles must be enforced such that *only* the specific Google Cloud Function service account possesses the read/write privileges required to access the gradient shards.

---

## 5. Technical Appendix: Mathematical Workflows

### A. Differential Privacy Guarantee
The mathematical guarantee of $(\epsilon, \delta)$-Differential Privacy ensures that for any two neighboring datasets $D$ and $D'$, and for any set of outcomes $S$ from training algorithm $\mathcal{A}$:

$$P[\mathcal{A}(D) \in S] \le e^{\epsilon} P[\mathcal{A}(D') \in S] + \delta$$

Opacus achieves this via Gaussian noise injection, scaled by the clipping threshold $C$ and variance $\sigma^2$:

$$w_{noisy} = w_{clipped} + \mathcal{N}(0, \sigma^2 C^2)$$

### B. GradsSharding Federated Averaging

Because heavy diagnostic models exceed serverless memory limits, the massive weight update vector $W_k^{(t)}$ for Client $k$ at round $t$ (based on the DynUNet architecture) is split into contiguous shards $S$:

$$W_k^{(t)} = \left[ w_{k,1}^{(t)}, w_{k,2}^{(t)}, \dots, w_{k,S}^{(t)} \right]$$

The Google Cloud Function computes the updated global shard $w_s^{(t+1)}$ using the element-wise average:

$$w_s^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{n} w_{k,s}^{(t)}$$

*(Where $n_k$ is the number of local images, and $n$ is total images across all clients).*

Clients reconstruct the full global model sequentially:

$$W^{(t+1)} = \left[ w_1^{(t+1)}, w_2^{(t+1)}, \dots, w_S^{(t+1)} \right]$$
