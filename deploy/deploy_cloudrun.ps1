# ==============================================================================
# Google Cloud Run Deployment Script for SmartCrop AI
# Project: idea-to-execution
# ==============================================================================

$PROJECT_ID = "idea-to-execution"
$REGION = "us-central1"
$SERVICE_NAME = "smartcrop-ai"
$IMAGE = "gcr.io/$PROJECT_ID/$SERVICE_NAME`:latest"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Deploying SmartCrop AI to Google Cloud Run" -ForegroundColor Cyan
Write-Host "Project ID: $PROJECT_ID | Region: $REGION" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green

# 1. Build and push image to Google Container Registry
Write-Host "`n[Step 1/2] Building container image via Google Cloud Build..." -ForegroundColor Yellow
gcloud builds submit --tag $IMAGE --project $PROJECT_ID

if ($LASTEXITCODE -ne 0) {
    Write-Host "Cloud Build failed. Please verify billing and permissions." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Deploy service to Cloud Run
Write-Host "`n[Step 2/2] Deploying service to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $SERVICE_NAME `
    --image $IMAGE `
    --platform managed `
    --region $REGION `
    --project $PROJECT_ID `
    --allow-unauthenticated `
    --memory 4Gi `
    --cpu 2 `
    --port 8501 `
    --timeout 300

Write-Host "`nDeployment complete! Service URL above." -ForegroundColor Green
