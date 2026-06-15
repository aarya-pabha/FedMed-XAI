#!/bin/bash
export PROJECT_DIR="/home/Aarya/healthcare_project"
cd $PROJECT_DIR

# Ensure Cloud SQL is running (Operational Protocol)
echo "📡 Activating Cloud SQL fl-metadata-db..."
gcloud sql instances patch fl-metadata-db --activation-policy=ALWAYS --project=healthcare-fl-diagnostics

# Activate Environment
source .venv_fix/bin/activate
export PYTHONPATH=$PROJECT_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cleanup Stale State
echo "🧹 Clearing local and remote cache..."
rm -rf $PROJECT_DIR/src/models/__pycache__ $PROJECT_DIR/src/federated/__pycache__ $PROJECT_DIR/src/utils/__pycache__
rm -f $PROJECT_DIR/ray_tmp/dp_state_client_*.pkl
rm -f $PROJECT_DIR/final_diagnostic_model.safetensors

# Purge Firestore (Nuke virtual docs)
echo "🔥 Nuking Firestore rounds..."
python scripts/nuke_firestore.py --force

# Purge GCS Shards (Clear stale model files)
echo "🧹 Wiping GCS shard buckets..."
gcloud storage rm --recursive gs://healthcare-fl-diagnostics-fl-shards-in/** 2>/dev/null || true
gcloud storage rm --recursive gs://healthcare-fl-diagnostics-fl-shards-out/** 2>/dev/null || true

# Seed Round 1
echo "🌱 Seeding Round 1 global model..."
python scripts/seed_round_0.py

# Launch Training
echo "🚀 Starting 5-Round Federated Training..."
python scripts/run_mp_orchestrator.py --num-rounds 5 --logical-batch-size 16
