#!/usr/bin/env bash
# ==============================================================================
# Google Cloud Run Deployment Script for SmartCrop AI
# Project: idea-to-execution
# ==============================================================================

set -e

PROJECT_ID="idea-to-execution"
REGION="us-central1"
SERVICE_NAME="smartcrop-ai"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "=========================================================="
echo "Deploying SmartCrop AI to Google Cloud Run"
echo "Project ID: ${PROJECT_ID} | Region: ${REGION}"
echo "=========================================================="

echo -e "\n[Step 1/2] Building container image via Google Cloud Build..."
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}"

echo -e "\n[Step 2/2] Deploying service to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE}" \
    --platform managed \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --port 8501 \
    --timeout 300

echo -e "\nDeployment complete! Service URL above."
