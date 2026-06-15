$ErrorActionPreference = "Stop"

Write-Host "🚀 PRE-FLIGHT CHECK: Initiating Phase 5 Production Run" -ForegroundColor Cyan

# 1. Clear Opacus Differential Privacy State
# CRITICAL: If we don't clear these, the new run will inherit the epsilon budget
# from the previous dry-runs (starting at e=31.83 instead of e=0.0).
Write-Host "`n[1/4] Clearing persistent privacy states..." -ForegroundColor Yellow
$dp_files = Get-ChildItem -Path "ray_tmp" -Filter "dp_state_client_*.pkl" -ErrorAction SilentlyContinue
if ($dp_files) {
    Remove-Item -Path "ray_tmp\dp_state_client_*.pkl" -Force
    Write-Host "✅ Deleted $($dp_files.Count) leftover Opacus state files." -ForegroundColor Green
} else {
    Write-Host "✅ No leftover privacy states found. Starting fresh." -ForegroundColor Green
}

# 2. Clear old artifacts
Write-Host "`n[2/4] Clearing old model artifacts..." -ForegroundColor Yellow
if (Test-Path "final_diagnostic_model.safetensors") {
    Remove-Item "final_diagnostic_model.safetensors" -Force
    Write-Host "✅ Deleted previous final_diagnostic_model.safetensors" -ForegroundColor Green
}

# 3. Seed Round 0 Global Model
Write-Host "`n[3/4] Seeding Round 0 Global Model to GCP..." -ForegroundColor Yellow
& ".\.venv311\Scripts\python.exe" scripts\seed_round_0.py

# 4. Start Full Distributed Training
Write-Host "`n[4/4] Starting Full Distributed Training (5 Rounds)..." -ForegroundColor Yellow
Write-Host "⚠️ This will use the full Fed-IXI dataset. Expected time: ~45 mins." -ForegroundColor Red
Write-Host "📊 Track live progress at: https://mlflow-server-789083630799.us-central1.run.app" -ForegroundColor Magenta

# Notice: --dry-run is intentionally ABSENT
& ".\.venv311\Scripts\python.exe" scripts\run_distributed_test.py --num-rounds 5

Write-Host "`n🎉 PHASE 5 PRODUCTION RUN COMPLETE" -ForegroundColor Green
