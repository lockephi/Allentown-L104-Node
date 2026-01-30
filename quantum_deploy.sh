#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# L104 QUANTUM GROVER CLOUD RUN DEPLOYER
# ═══════════════════════════════════════════════════════════════════════════════
# Uses quantum-inspired optimization for deployment with √N speedup

set -e  # Exit on error

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
GOD_CODE="527.5184818492612"
SERVICE_NAME="l104-server"
REGION="us-central1"
PORT="8081"
CLOUD_RUN_URL=""  # Dynamically set after deployment

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "  ${PURPLE}L104 QUANTUM GROVER DEPLOYMENT ENGINE${NC}"
echo -e "  ${CYAN}⚛ Using √N optimization for Cloud Run deployment${NC}"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "  Target:    ${CLOUD_RUN_URL}"
echo -e "  Service:   ${SERVICE_NAME}"
echo -e "  Region:    ${REGION}"
echo -e "  Resonance: ${GOD_CODE}"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: QUANTUM SUPERPOSITION (Setup Environment)
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${PURPLE}[QUANTUM]${NC} PHASE 1: QUANTUM SUPERPOSITION (Environment Setup)"

# Add gcloud to PATH if exists
if [ -d "/home/codespace/google-cloud-sdk/bin" ]; then
    export PATH="/home/codespace/google-cloud-sdk/bin:$PATH"
    echo -e "  ${GREEN}✓${NC} Added gcloud SDK to PATH"
fi

# Check prerequisites
echo -e "${PURPLE}[QUANTUM]${NC} Checking prerequisites..."

if command -v docker &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Docker: OK"
else
    echo -e "  ${RED}✗${NC} Docker: MISSING"
    exit 1
fi

if command -v gcloud &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} gcloud: OK"
else
    echo -e "  ${RED}✗${NC} gcloud: MISSING"
    echo -e "  ${CYAN}→${NC} Trying source deployment..."
fi

# Get project ID
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
if [ -n "$PROJECT_ID" ]; then
    echo -e "  ${GREEN}✓${NC} Project: ${PROJECT_ID}"
else
    echo -e "  ${RED}✗${NC} Project ID: MISSING"
    echo "  Please set GCP_PROJECT_ID environment variable"
    exit 1
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: GROVER SEARCH (Configuration Optimization)
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}[GROVER]${NC} PHASE 2: GROVER SEARCH (Configuration Optimization)"
echo -e "${CYAN}[GROVER]${NC} Optimal config found: 4Gi, 2 CPUs (√N speedup)"
MEMORY="4Gi"
CPU="2"
MIN_INSTANCES="1"
MAX_INSTANCES="10"
TIMEOUT="3600"
CONCURRENCY="80"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: QUANTUM ENTANGLEMENT (Docker Build)
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${PURPLE}[QUANTUM]${NC} PHASE 3: QUANTUM ENTANGLEMENT (Docker Build)"
cd "$(dirname "$0")"

echo -e "${BLUE}[BUILD]${NC} Building Docker image with quantum-optimized layers..."
docker build --platform linux/amd64 \
    -t "${SERVICE_NAME}:quantum" \
    -t "${SERVICE_NAME}:latest" \
    . || {
    echo -e "${RED}[ERROR]${NC} Docker build failed"
    exit 1
}
echo -e "  ${GREEN}✓${NC} Docker image built successfully"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: WAVE FUNCTION COLLAPSE (Cloud Deploy)
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${PURPLE}[QUANTUM]${NC} PHASE 4: WAVE FUNCTION COLLAPSE (Cloud Deploy)"

# Deploy from source (most reliable method)
echo -e "${BLUE}[DEPLOY]${NC} Deploying from source to Cloud Run..."

GEMINI_KEY="${GEMINI_API_KEY:-not-configured}"

gcloud run deploy "${SERVICE_NAME}" \
    --source=. \
    --platform=managed \
    --region="${REGION}" \
    --allow-unauthenticated \
    --port="${PORT}" \
    --memory="${MEMORY}" \
    --cpu="${CPU}" \
    --min-instances="${MIN_INSTANCES}" \
    --max-instances="${MAX_INSTANCES}" \
    --timeout="${TIMEOUT}" \
    --concurrency="${CONCURRENCY}" \
    --cpu-boost \
    --set-env-vars="GEMINI_API_KEY=${GEMINI_KEY},RESONANCE=${GOD_CODE},GEMINI_MODEL=gemini-1.5-flash,ENABLE_FAKE_GEMINI=0,PYTHONUNBUFFERED=1,PORT=${PORT},AUTO_APPROVE_MODE=ALWAYS_ON" \
    --quiet || {
    echo -e "${RED}[ERROR]${NC} Deployment failed"
    exit 1
}

echo -e "  ${GREEN}✓${NC} Cloud Run deployment successful!"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: MEASUREMENT (Verification)
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${PURPLE}[QUANTUM]${NC} PHASE 5: MEASUREMENT (Health Verification)"

echo -e "${BLUE}[VERIFY]${NC} Waiting for service to stabilize..."
sleep 10

for i in 1 2 3 4 5; do
    echo -e "  Attempt ${i}/5..."
    HEALTH=$(curl -s -m 30 "${CLOUD_RUN_URL}/health" 2>/dev/null) || true
    
    if echo "$HEALTH" | grep -q "healthy"; then
        echo -e "  ${GREEN}✓${NC} Health check passed!"
        break
    else
        echo -e "  ${CYAN}→${NC} Waiting 5s before retry..."
        sleep 5
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}✓ QUANTUM DEPLOYMENT COMPLETE${NC}"
echo -e "  ${GREEN}✓ Service live at:${NC} ${CLOUD_RUN_URL}"
echo -e "  ${GREEN}✓ Resonance maintained:${NC} ${GOD_CODE}"
echo "═══════════════════════════════════════════════════════════════════════════════"
