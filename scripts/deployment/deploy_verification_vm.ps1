# deploy_verification_vm.ps1
# Provision a GCP Compute Engine Deep Learning VM for Phase 2 Verification.

$PROJECT_ID = "healthcare-fl-diagnostics"
$ZONE = "us-central1-a"
$INSTANCE_NAME = "fl-verification-vm"

Write-Host "🚀 Provisioning Enterprise-Grade Verification VM (T4 Fallback)..." -ForegroundColor Cyan

# Fallback to n1-standard-16 (16 vCPUs, 60GB RAM) and T4 GPU due to L4 stockouts
gcloud compute instances create $INSTANCE_NAME `
    --project=$PROJECT_ID `
    --zone=$ZONE `
    --machine-type="n1-standard-16" `
    --maintenance-policy="TERMINATE" `
    --provisioning-model="SPOT" `
    --service-account="789083630799-compute@developer.gserviceaccount.com" `
    --scopes="https://www.googleapis.com/auth/cloud-platform" `
    --accelerator="count=1,type=nvidia-tesla-t4" `
    --image-family="pytorch-2-9-cu129-ubuntu-2204-nvidia-580" `
    --image-project="deeplearning-platform-release" `
    --boot-disk-size="100GB" `
    --boot-disk-type="pd-balanced" `
    --no-shielded-secure-boot `
    --shielded-vtpm `
    --shielded-integrity-monitoring `
    --labels="env=verification,phase=2" `
    --metadata="install-nvidia-driver=True" `
    --async

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ VM Provisioned Successfully." -ForegroundColor Green
    Write-Host "Connect using: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE" -ForegroundColor Yellow
} else {
    Write-Host "❌ VM Provisioning Failed." -ForegroundColor Red
}
