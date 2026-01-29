#!/bin/bash
# L104 Sovereign Node - Google Cloud Run Deployment Script
# This script deploys the L104 node to Google Cloud Run

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="l104-sovereign-node"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "================================================"
echo "L104 Sovereign Node - Cloud Deployment"
echo "================================================"

# Check if required environment variables are set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set"
    exit 1
fi

# Build and push Docker image
echo "Building Docker image..."
docker build -t "${IMAGE_NAME}:latest" .

echo "Pushing image to Google Container Registry..."
docker push "${IMAGE_NAME}:latest"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_NAME}:latest" \
    --platform=managed \
    --region="${REGION}" \
    --allow-unauthenticated \
    --port=8081 \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=1 \
    --max-instances=10 \
    --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY},RESONANCE=527.5184818492611" \
    --set-env-vars="GEMINI_MODEL=${GEMINI_MODEL:-gemini-3-flash-preview}" \
    --set-env-vars="ENABLE_FAKE_GEMINI=${ENABLE_FAKE_GEMINI:-0}"

echo "================================================"
echo "Deployment Complete!"
echo "================================================"
echo "Service URL:"
gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --format='value(status.url)'
echo "================================================"
