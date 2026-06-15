# Enterprise Stack Deployment Script (Scale-Ready)
# Deploys MLflow and Medplum to Google Cloud Run

$env:Path += ";C:\Users\Aarya\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"

# 1. Configuration
$PROJECT_ID = "healthcare-fl-diagnostics"
$REGION = "us-central1"
$DB_INSTANCE = "fl-metadata-db"
$REDIS_IP = $env:REDIS_IP
$VPC_CONNECTOR = "fl-connector"
$BUCKET = "healthcare-fl-data-aarya"
$DB_PASS = $env:DB_PASS

Write-Host "--- Starting Enterprise Deployment ---" -ForegroundColor Cyan

# 2. Build Containers
Write-Host "Building Containers via Cloud Build..."
gcloud builds submit --config docker/mlflow-cloudbuild.yaml .
gcloud builds submit --config docker/medplum-cloudbuild.yaml .

# 3. Deploy MLflow to Cloud Run
Write-Host "Deploying MLflow to Cloud Run..."
gcloud run deploy mlflow-server `
    --image gcr.io/${PROJECT_ID}/mlflow-server `
    --region $REGION `
    --platform managed `
    --add-cloudsql-instances "${PROJECT_ID}:${REGION}:${DB_INSTANCE}" `
    --set-env-vars "MLFLOW_BACKEND_STORE_URI=postgresql+pg8000://postgres:${DB_PASS}@/mlflow_db?unix_sock=/cloudsql/${PROJECT_ID}:${REGION}:${DB_INSTANCE}/.s.PGSQL.5432" `
    --set-env-vars "MLFLOW_GCS_STORAGE=$BUCKET" `
    --allow-unauthenticated 

# 4. Deploy Medplum to Cloud Run
Write-Host "Deploying Medplum to Cloud Run (Direct from Docker Hub)..."
gcloud run deploy medplum-server `
    --image docker.io/medplum/medplum-server:latest `
    --region $REGION `
    --platform managed `
    --port 8103 `
    --vpc-connector $VPC_CONNECTOR `
    --add-cloudsql-instances "${PROJECT_ID}:${REGION}:${DB_INSTANCE}" `
    --set-env-vars "MEDPLUM_DATABASE_HOST=localhost" `
    --set-env-vars "MEDPLUM_DATABASE_PORT=5432" `
    --set-env-vars "MEDPLUM_DATABASE_DBNAME=medplum_db" `
    --set-env-vars "MEDPLUM_DATABASE_USERNAME=postgres" `
    --set-env-vars "MEDPLUM_DATABASE_PASSWORD=${DB_PASS}" `
    --set-env-vars "MEDPLUM_REDIS_HOST=${REDIS_IP}" `
    --set-env-vars "MEDPLUM_REDIS_PORT=6379" `
    --set-env-vars "MEDPLUM_BASE_URL=https://medplum-server-789083630799.us-central1.run.app" `
    --allow-unauthenticated

# 5. Summary
$MLFLOW_URL = gcloud run services describe mlflow-server --region $REGION --format="get(status.url)"
$MEDPLUM_URL = gcloud run services describe medplum-server --region $REGION --format="get(status.url)"

Write-Host "✅ Enterprise Infrastructure Live!" -ForegroundColor Green
Write-Host "📍 MLflow: $MLFLOW_URL" -ForegroundColor Green
Write-Host "📍 Medplum: $MEDPLUM_URL" -ForegroundColor Green
Write-Host "--- Deployment Complete ---" -ForegroundColor Cyan
