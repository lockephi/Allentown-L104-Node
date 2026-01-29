#!/bin/bash
# L104 SOVEREIGN NODE - OMEGA CLOUD DEPLOYMENT
# Deploys the L104 DNA Core and all subsystems to cloud infrastructure
# INVARIANT: 527.5184818492612 | PILOT: LONDEL

set -e

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
PROJECT_ID="${GCP_PROJECT_ID:-l104-sovereign-node}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="l104-omega"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
GOD_CODE="527.5184818492612"

echo "█████████████████████████████████████████████████████████████████████"
echo "    L104 :: OMEGA CLOUD DEPLOYMENT"
echo "    Deploying DNA Core + Autonomous Agent + All Subsystems"
echo "    GOD_CODE: ${GOD_CODE}"
echo "█████████████████████████████████████████████████████████████████████"

# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[PHASE 1/5] ENVIRONMENT VALIDATION"
echo "─────────────────────────────────────────────────────────────────────"

# Check for required environment variables
REQUIRED_VARS="GEMINI_API_KEY"
MISSING_VARS=""

for var in $REQUIRED_VARS; do
    if [ -z "${!var}" ]; then
        MISSING_VARS="${MISSING_VARS} ${var}"
    fi
done

if [ -n "$MISSING_VARS" ]; then
    echo "⚠️  Missing required environment variables:${MISSING_VARS}"
    echo "    Set them before deployment or they will be prompted."

    if [ -z "$GEMINI_API_KEY" ]; then
        echo -n "Enter GEMINI_API_KEY: "
        read -s GEMINI_API_KEY
        echo ""
        export GEMINI_API_KEY
    fi
fi

echo "✓ GEMINI_API_KEY: SET"
echo "✓ PROJECT_ID: ${PROJECT_ID}"
echo "✓ REGION: ${REGION}"
echo "✓ SERVICE_NAME: ${SERVICE_NAME}"

# ═══════════════════════════════════════════════════════════════════════════════
# DOCKER BUILD
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[PHASE 2/5] BUILDING DOCKER IMAGE"
echo "─────────────────────────────────────────────────────────────────────"

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

echo "Building L104 Omega image..."
docker build -t "${IMAGE_NAME}:latest" \
    --build-arg GOD_CODE="${GOD_CODE}" \
    --label "l104.dna.signature=$(date +%s | sha256sum | head -c 32)" \
    --label "l104.version=OMEGA" \
    .

echo "✓ Docker image built: ${IMAGE_NAME}:latest"

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL TEST (OPTIONAL)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$1" == "--test" ]; then
    echo ""
    echo "[TEST MODE] Running container locally..."
    echo "─────────────────────────────────────────────────────────────────────"

    docker run -d \
        --name l104-omega-test \
        -p 8081:8081 \
        -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
        -e GOD_CODE="${GOD_CODE}" \
        "${IMAGE_NAME}:latest"

    echo "✓ Container started. Waiting for health check..."
    sleep 10

    # Health check
    if curl -s http://localhost:8081/health | grep -q "ok"; then
        echo "✓ Health check passed!"
        echo "  Access at: http://localhost:8081"
    else
        echo "⚠️  Health check failed. Checking logs..."
        docker logs l104-omega-test --tail 50
    fi

    echo ""
    echo "To stop: docker stop l104-omega-test && docker rm l104-omega-test"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# CLOUD PUSH
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[PHASE 3/5] PUSHING TO CONTAINER REGISTRY"
echo "─────────────────────────────────────────────────────────────────────"

# Check if gcloud is available
if command -v gcloud &> /dev/null; then
    echo "Authenticating with GCR..."
    gcloud auth configure-docker --quiet 2>/dev/null || true

    echo "Pushing image to Google Container Registry..."
    docker push "${IMAGE_NAME}:latest"
    echo "✓ Image pushed: ${IMAGE_NAME}:latest"
else
    echo "⚠️  gcloud CLI not found. Skipping GCR push."
    echo "   To deploy manually, push the image to your container registry."
fi

# ═══════════════════════════════════════════════════════════════════════════════
# CLOUD RUN DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[PHASE 4/5] DEPLOYING TO CLOUD RUN"
echo "─────────────────────────────────────────────────────────────────────"

if command -v gcloud &> /dev/null; then
    echo "Deploying ${SERVICE_NAME} to Cloud Run..."

    gcloud run deploy "${SERVICE_NAME}" \
        --image="${IMAGE_NAME}:latest" \
        --platform=managed \
        --region="${REGION}" \
        --allow-unauthenticated \
        --port=8081 \
        --memory=4Gi \
        --cpu=4 \
        --min-instances=1 \
        --max-instances=20 \
        --timeout=3600 \
        --concurrency=100 \
        --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY}" \
        --set-env-vars="GOD_CODE=${GOD_CODE}" \
        --set-env-vars="GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.0-flash-exp}" \
        --set-env-vars="ENABLE_DNA_CORE=1" \
        --set-env-vars="ENABLE_AUTONOMOUS_AGENT=1" \
        --set-env-vars="ENABLE_HEARTBEAT=1" \
        --set-env-vars="HEARTBEAT_INTERVAL=60" \
        --set-env-vars="L104_STAGE=OMEGA"

    echo ""
    echo "✓ Deployment complete!"

    # Get service URL
    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --region="${REGION}" \
        --format='value(status.url)')

    echo "  Service URL: ${SERVICE_URL}"
else
    echo "⚠️  gcloud CLI not found. Manual deployment required."
    echo ""
    echo "To deploy manually:"
    echo "1. Push ${IMAGE_NAME}:latest to your container registry"
    echo "2. Deploy to your cloud platform with these env vars:"
    echo "   - GEMINI_API_KEY"
    echo "   - GOD_CODE=${GOD_CODE}"
    echo "   - ENABLE_DNA_CORE=1"
    echo "   - ENABLE_AUTONOMOUS_AGENT=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# POST-DEPLOYMENT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[PHASE 5/5] POST-DEPLOYMENT VERIFICATION"
echo "─────────────────────────────────────────────────────────────────────"

if [ -n "$SERVICE_URL" ]; then
    echo "Waiting for service to be ready..."
    sleep 15

    echo "Checking health endpoint..."
    if curl -s "${SERVICE_URL}/health" | grep -q "ok"; then
        echo "✓ Health check passed!"
    else
        echo "⚠️  Health check pending. Service may still be starting."
    fi

    echo ""
    echo "Testing DNA Core activation..."
    curl -s "${SERVICE_URL}/l104/dna/status" 2>/dev/null || echo "DNA endpoint initializing..."
fi

# ═══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "█████████████████████████████████████████████████████████████████████"
echo "    L104 OMEGA DEPLOYMENT COMPLETE"
echo "█████████████████████████████████████████████████████████████████████"
echo ""
echo "    Service:        ${SERVICE_NAME}"
echo "    Region:         ${REGION}"
echo "    GOD_CODE:       ${GOD_CODE}"
echo "    DNA Core:       ENABLED"
echo "    Autonomous:     ENABLED"
echo "    Heartbeat:      60s interval"
echo ""
if [ -n "$SERVICE_URL" ]; then
    echo "    URL:            ${SERVICE_URL}"
    echo ""
    echo "    Endpoints:"
    echo "      Health:       ${SERVICE_URL}/health"
    echo "      DNA Status:   ${SERVICE_URL}/l104/dna/status"
    echo "      Synthesize:   ${SERVICE_URL}/l104/dna/synthesize"
    echo "      Agent:        ${SERVICE_URL}/l104/agent/status"
fi
echo ""
echo "█████████████████████████████████████████████████████████████████████"
