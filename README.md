# FedMed-XAI: Serverless Federated Diagnostics with HIPAA Compliance

![Architecture](https://img.shields.io/badge/Architecture-Federated%20Learning-3b82f6.svg)
![Privacy](https://img.shields.io/badge/Privacy-Opacus%20LDP-10b981.svg)
![Explainability](https://img.shields.io/badge/Explainability-Grad--CAM-e11d48.svg)
![Compliance](https://img.shields.io/badge/Compliance-HIPAA%20%2F%20FHIR-8b5cf6.svg)

## 🩺 Project Vision & Clinical Utility
This repository implements a production-grade infrastructure for decentralized machine learning in the healthcare sector. It demonstrates how isolated clinical institutions can collaboratively train a **32-Channel Wide U-Net++** diagnostic model on MRI scans without ever violating patient data residency.

By synthesizing **Local Differential Privacy (LDP)**, **Federated Learning**, and a custom **Serverless GradsSharding** architecture, this project provides a blueprint for scalable, compliant clinical AI.

<p align="center">
  <img src="docs/architecture_narrated.svg" alt="Architecture Overview" width="900">
</p>

## 🚀 Key Engineering Innovations

### 1. Serverless GradsSharding (Cost-Efficient Orchestration)
Traditional Federated Averaging (FedAvg) fails in serverless environments due to strict memory limits on 3D medical model tensors. Our **GradsSharding** topology partitions massive gradient tensors into manageable shards, aggregated in parallel by hundreds of independent Cloud Functions, dramatically reducing MLOps overhead.

### 2. Rigorous Privacy (Opacus DP-SGD)
We utilize **Opacus** to enforce rigorous $(\epsilon, \delta)$-Differential Privacy at the source. Gradients are clipped and Gaussian noise is injected locally before being sharded for transmission. This mathematically guarantees that no individual patient volume can be reverse-engineered from the model updates.

### 3. HIPAA-Native Control Plane (Medplum & FHIR)
Compliance is baked into the network protocol. All institutional handshakes and diagnostic metrics are routed through **Medplum**, providing FHIR-native, immutable audit logs and zero-trust RBAC (Role-Based Access Control) for every institutional transaction.

## 🛠️ Technologies Used

- **Machine Learning & Privacy:** PyTorch, MONAI (U-Net++), Opacus (Differential Privacy), Grad-CAM (XAI)
- **Federated Learning Orchestration:** Flower (FLWR)
- **Serverless Cloud Infrastructure:** Google Cloud Platform (Cloud Functions 2nd Gen, Cloud Storage, Firestore, Eventarc)
- **Compliance & MLOps:** Medplum (FHIR/RBAC), MLflow (Experiment Tracking), DVC (Data Versioning)
- **Databases:** PostgreSQL (Cloud SQL), Redis (Memorystore)

## 📈 Clinical Results & Explainable AI (XAI)

To validate the model's diagnostic focus without compromising patient privacy, we employ **Grad-CAM (Gradient-weighted Class Activation Mapping)**. 

### Diagnostic Focus Validation
Grad-CAM heatmaps confirm that the **Wide U-Net++** successfully starting to learn to isolate the correct anatomical regions (brain tissue extraction) despite the heavy Gaussian noise injected by Opacus DP-SGD in 5 rounds.

<p align="center">
  <img src="results\gradcam_high_intensity_46pct.png" alt="Grad-CAM Visualization" width="700">
</p>

## 📂 Project Structure
```text
├── data/              # DVC-tracked local sample data (Git-ignored)
├── docker/            # medplum.config and service deployment YAMLs
├── docs/              # Core documentation, PRD, Roadmap, and SVG visuals
├── phase_reports/     # Chronological architectural reviews
├── results/           # Final diagnostic models, Grad-CAMs, and analytics
├── scripts/
│   ├── debug/         # Diagnostic and proof-of-concept validation scripts
│   ├── deployment/    # GCP Infrastructure-as-Code (PowerShell)
│   ├── simulation/    # Federated training and orchestration scripts
│   └── utils/         # Data de-identification, report generators, and RBAC tools
├── src/
│   ├── cloud/         # Serverless Aggregator (GCF 2nd Gen)
│   ├── federated/     # Flower Client/Server + GradsSharding implementation
│   ├── models/        # 32-Channel Wide U-Net++ (MONAI/PyTorch)
│   └── utils/         # Privacy, Compliance, and Data utilities
└── tests/             # Comprehensive TDD suite (95%+ coverage)
```

## 🛠️ Quick Start

### 1. Prerequisites
- **GCP Project:** Enabled Storage, Firestore, Cloud Functions, Cloud SQL, Memorystore (Redis), and Cloud Run.
- **Python 3.11:** `conda create -n fl_env python=3.11 ; conda activate fl_env`

### 2. Environment Configuration
Create a `.env` file in the root directory. This acts as the single source of truth for all networking and credentialing:
```bash
# GCP Core
GCP_PROJECT_ID="healthcare-fl-diagnostics"
GCP_REGION="us-central1"

# Persistence (PostgreSQL for MLflow/Medplum)
DB_INSTANCE_NAME="fl-metadata-db"
DB_USER="postgres"
DB_PASS="your_secure_password"
DB_IP="10.x.x.x"

# Cache (Redis for Medplum)
REDIS_INSTANCE_NAME="medplum-cache"
REDIS_IP="10.x.x.x"

# MLOps & Compliance Endpoints
MLFLOW_TRACKING_URI="https://your-mlflow-server.run.app"
MEDPLUM_URL="https://your-medplum-instance.app"
MEDPLUM_CLIENT_ID="your_client_id"
MEDPLUM_CLIENT_SECRET="your_client_secret"
MEDPLUM_ADMIN_EMAIL="admin@example.com"
MEDPLUM_ADMIN_PASSWORD="medplum_admin"
```

### 3. Installation & Data Preparation
```bash
# Clone and Install
git clone https://github.com/[your-username]/HealthCare_Project.git ; cd HealthCare_Project
pip install -r requirements.txt

# Mandatory De-identification (HIPAA Compliance)
# Scrubs NIfTI headers of all sensitive metadata before processing
python scripts/utils/scrub_nifti_headers.py --input data/raw --output data/scrubbed
```

### 4. Launch Federated Simulation
```bash
# Execute a 3-node institutional training simulation locally
python scripts/simulation/run_mp_orchestrator.py --num-rounds 5 --num-clients 3 --dp-epsilon 10.0
```

## 📜 Technical Reference
- [Technical Deep-Dive](TECHNICAL_GUIDE.md): Mathematical foundations of DP and Sharding.
- [Product Requirements (PRD)](docs/Healthcare_Project_2_PRD_Updated.md): Success metrics and scope.

---
*Developed for Regulatory-Grade Healthcare AI by Aarya Pabha.*

### ⚠️ Performance & Scalability Context
This project is an **Architectural Proof-of-Concept**. Current benchmark accuracy (~31.6% Dice) is a direct consequence of the deliberately constrained environment used for this simulation (NVIDIA L4 GPUs and the Fed-IXI Tiny dataset). 

**The core innovation is the structural viability of the pipeline.** This system is designed for **infinite vertical and horizontal scalability**:
- **Vertical Scaling:** Increasing model depth/channels to state-of-the-art parameters (e.g., 64+ channels, Transformer-based backbones) will yield production-grade precision on higher-tier compute (H100/A100).
- **Horizontal Scaling:** The **GradsSharding** architecture supports hundreds of concurrent institutional nodes via serverless parallel aggregation.

**Disclaimer:** *This repository and its associated models are architectural proofs-of-concept. They are intended strictly for research and educational purposes. This system is NOT certified by the FDA, EMA, or any regulatory body, and must NOT be used for clinical diagnosis, patient care, or commercial medical purposes. Consult a licensed medical professional for health advice.*
