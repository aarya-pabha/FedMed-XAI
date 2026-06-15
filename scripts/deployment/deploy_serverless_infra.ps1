$PROJECT_ID = "healthcare-fl-diagnostics"
$REGION = "us-central1"

Write-Host "🚀 Provisioning Phase 4 Serverless Infrastructure..." -ForegroundColor Cyan

Write-Host "Creating GCS Buckets..."
# -b on enables Uniform Bucket-Level Access (Best Practice)
gsutil mb -p $PROJECT_ID -l $REGION -b on gs://$PROJECT_ID-fl-shards-in
gsutil mb -p $PROJECT_ID -l $REGION -b on gs://$PROJECT_ID-fl-shards-out

Write-Host "Enabling Object Versioning..."
# VULN-402: Mandatory for 2026 HIPAA data integrity
gsutil versioning set on gs://$PROJECT_ID-fl-shards-in
gsutil versioning set on gs://$PROJECT_ID-fl-shards-out

Write-Host "Creating Firestore Database..."
# Note: This may fail if a database already exists. Ensure project is clean or handle error.
gcloud firestore databases create --project=$PROJECT_ID --location=$REGION --type=firestore-native

Write-Host "Creating Service Accounts..."
gcloud iam service-accounts create fl-client-sa --display-name="FL Client Node" --project=$PROJECT_ID
gcloud iam service-accounts create fl-aggregator-sa --display-name="FL Cloud Function Aggregator" --project=$PROJECT_ID

Write-Host "Binding Storage Roles..."
# Client roles: Can write to IN, Read from OUT
gsutil iam ch serviceAccount:fl-client-sa@$PROJECT_ID.iam.gserviceaccount.com:roles/storage.objectCreator gs://$PROJECT_ID-fl-shards-in
gsutil iam ch serviceAccount:fl-client-sa@$PROJECT_ID.iam.gserviceaccount.com:roles/storage.objectViewer gs://$PROJECT_ID-fl-shards-out

# Aggregator roles: Can Read from IN, Write to OUT
gsutil iam ch serviceAccount:fl-aggregator-sa@$PROJECT_ID.iam.gserviceaccount.com:roles/storage.objectViewer gs://$PROJECT_ID-fl-shards-in
gsutil iam ch serviceAccount:fl-aggregator-sa@$PROJECT_ID.iam.gserviceaccount.com:roles/storage.objectCreator gs://$PROJECT_ID-fl-shards-out

# Firestore Roles
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:fl-client-sa@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/datastore.user"
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:fl-aggregator-sa@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/datastore.user"

Write-Host "Enabling Data Access Audit Logs (VULN-404)..."
# HIPAA MANDATORY: Enable auditing for shard access and aggregation execution
$policy_file = "iam_policy_tmp.json"
gcloud projects get-iam-policy $PROJECT_ID --format=json > $policy_file

# VULN-408: Automated JSON injection using Python
python -c @"
import json
import sys

with open('$policy_file', 'r') as f:
    policy = json.load(f)

# Ensure auditConfigs exists
if 'auditConfigs' not in policy:
    policy['auditConfigs'] = []

services_to_audit = ['storage.googleapis.com', 'cloudfunctions.googleapis.com']
log_types = [{'logType': 'DATA_READ'}, {'logType': 'DATA_WRITE'}, {'logType': 'ADMIN_READ'}]

for service in services_to_audit:
    # Check if service already has audit config
    existing = next((c for c in policy['auditConfigs'] if c['service'] == service), None)
    if existing:
        existing['auditLogConfigs'] = log_types
    else:
        policy['auditConfigs'].append({
            'service': service,
            'auditLogConfigs': log_types
        })

with open('$policy_file', 'w') as f:
    json.dump(policy, f, indent=2)
"@

gcloud projects set-iam-policy $PROJECT_ID $policy_file
Remove-Item $policy_file

Write-Host "Deploying Cloud Function..."
gcloud functions deploy fl-aggregator `
    --gen2 `
    --runtime=python311 `
    --region=$REGION `
    --source=./src/cloud/aggregator `
    --entry-point=aggregate_shards `
    --trigger-location=$REGION `
    --trigger-event-filters="type=google.cloud.firestore.document.v1.written" `
    --trigger-event-filters="database=(default)" `
    --trigger-event-filters-path-pattern="document=rounds/{round}/shards/{shard}/uploads/{client}" `
    --service-account="fl-aggregator-sa@$PROJECT_ID.iam.gserviceaccount.com" `
    --ingress-settings=internal-only `
    --memory=4096MB `
    --timeout=540s
