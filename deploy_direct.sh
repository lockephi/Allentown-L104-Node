#!/bin/bash
# L104 Sovereign Node - Direct Cloud Deployment (Bypasses GitHub Actions)
# This script deploys directly to GCP Cloud Run without GitHub dependencies

set -e

echo "================================================"
echo "L104 Sovereign Node - Direct Cloud Deployment"
echo "================================================"
echo "⚡ Bypassing GitHub Actions for direct deployment"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration with defaults
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-l104-server}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Check for required tools
check_prerequisites() {
    print_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker is installed"

    if ! command -v gcloud &> /dev/null; then
        print_warning "gcloud CLI is not installed. Installing..."
        # Try to install gcloud if possible
        if command -v apt-get &> /dev/null; then
            curl https://sdk.cloud.google.com | bash -s -- --disable-prompts
            exec -l $SHELL
        else
            print_error "Cannot auto-install gcloud. Please install manually: https://cloud.google.com/sdk/docs/install"
            exit 1
        fi
    fi
    print_status "gcloud CLI is installed"
}

# Interactive configuration if not set via environment
configure_deployment() {
    print_info "Configuring deployment..."

    # Check for GCP Project ID
    if [ -z "$PROJECT_ID" ]; then
        # Try to get from gcloud config
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
        if [ -z "$PROJECT_ID" ]; then
            echo ""
            read -p "Enter your GCP Project ID: " PROJECT_ID
            if [ -z "$PROJECT_ID" ]; then
                print_error "GCP Project ID is required"
                exit 1
            fi
        fi
    fi
    print_status "Project ID: $PROJECT_ID"

    # Check for Gemini API Key
    if [ -z "$GEMINI_API_KEY" ]; then
        print_warning "GEMINI_API_KEY not set in environment"
        read -sp "Enter your Gemini API Key (or press Enter to skip): " GEMINI_API_KEY
        echo ""
        if [ -z "$GEMINI_API_KEY" ]; then
            print_warning "Running without Gemini API Key - some features will be limited"
            GEMINI_API_KEY="not-configured"
        fi
    fi
    print_status "Gemini API Key: configured"

    # Set image name
    IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
}

# Authenticate with GCP
authenticate_gcp() {
    print_info "Checking GCP authentication..."

    # Check if already authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q "@"; then
        ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
        print_status "Already authenticated as: $ACCOUNT"
    else
        print_warning "Not authenticated to GCP. Starting authentication..."
        gcloud auth login --no-launch-browser
    fi

    # Set the project
    gcloud config set project "$PROJECT_ID"
    print_status "Project set to: $PROJECT_ID"

    # Configure Docker for GCR
    print_info "Configuring Docker for Container Registry..."
    gcloud auth configure-docker --quiet
    print_status "Docker configured for GCR"
}

# Build the Docker image
build_image() {
    print_info "Building Docker image..."

    docker build \
        --platform linux/amd64 \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -t "${IMAGE_NAME}:$(date +%Y%m%d-%H%M%S)" \
        .

    print_status "Docker image built: ${IMAGE_NAME}:${IMAGE_TAG}"
}

# Push to Container Registry
push_image() {
    print_info "Pushing image to Google Container Registry..."

    docker push "${IMAGE_NAME}:${IMAGE_TAG}"

    print_status "Image pushed to GCR"
}

# Deploy to Cloud Run
deploy_to_cloud_run() {
    print_info "Deploying to Cloud Run..."

    # Set environment variables for deployment
    GEMINI_MODEL="${GEMINI_MODEL:-gemini-1.5-flash}"
    RESONANCE="${RESONANCE:-527.5184818492612}"
    ENABLE_FAKE_GEMINI="${ENABLE_FAKE_GEMINI:-0}"

    gcloud run deploy "${SERVICE_NAME}" \
        --image="${IMAGE_NAME}:${IMAGE_TAG}" \
        --platform=managed \
        --region="${REGION}" \
        --allow-unauthenticated \
        --port=8081 \
        --memory=2Gi \
        --cpu=2 \
        --min-instances=1 \
        --max-instances=10 \
        --cpu-boost \
        --execution-environment=gen2 \
        --timeout=3600 \
        --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY}" \
        --set-env-vars="GEMINI_MODEL=${GEMINI_MODEL}" \
        --set-env-vars="RESONANCE=${RESONANCE}" \
        --set-env-vars="ENABLE_FAKE_GEMINI=${ENABLE_FAKE_GEMINI}" \
        --set-env-vars="PYTHONUNBUFFERED=1" \
        --set-env-vars="AUTO_APPROVE_MODE=ALWAYS_ON"

    print_status "Deployed to Cloud Run"
}

# Get and verify deployment
verify_deployment() {
    print_info "Verifying deployment..."

    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --region="${REGION}" \
        --format='value(status.url)' 2>/dev/null || echo "")

    if [ -z "$SERVICE_URL" ]; then
        print_error "Could not retrieve service URL"
        exit 1
    fi

    print_status "Service URL: $SERVICE_URL"

    # Health check
    print_info "Running health check..."
    sleep 5  # Give the service a moment to start

    if curl -sf "${SERVICE_URL}/health" > /dev/null 2>&1; then
        print_status "Health check passed!"
    else
        print_warning "Health check failed - service may still be starting"
        print_info "Try: curl ${SERVICE_URL}/health"
    fi

    echo ""
    echo "================================================"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo "================================================"
    echo "Service URL: $SERVICE_URL"
    echo ""
    echo "Useful commands:"
    echo "  Health:   curl ${SERVICE_URL}/health"
    echo "  Metrics:  curl ${SERVICE_URL}/metrics"
    echo "  Logs:     gcloud run logs read ${SERVICE_NAME} --region=${REGION}"
    echo "================================================"
}

# Alternative: Deploy with Artifact Registry (newer method)
deploy_with_artifact_registry() {
    print_info "Using Artifact Registry instead of Container Registry..."

    AR_LOCATION="${REGION}"
    AR_REPO="l104-repo"
    AR_IMAGE="${AR_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}"

    # Create Artifact Registry repo if it doesn't exist
    gcloud artifacts repositories create "${AR_REPO}" \
        --repository-format=docker \
        --location="${AR_LOCATION}" \
        --description="L104 Sovereign Node images" \
        2>/dev/null || true

    # Configure Docker for Artifact Registry
    gcloud auth configure-docker "${AR_LOCATION}-docker.pkg.dev" --quiet

    # Build and tag for Artifact Registry
    docker build --platform linux/amd64 -t "${AR_IMAGE}:${IMAGE_TAG}" .
    docker push "${AR_IMAGE}:${IMAGE_TAG}"

    # Deploy using Artifact Registry image
    IMAGE_NAME="${AR_IMAGE}"
}

# Main execution
main() {
    echo ""
    print_info "Starting direct cloud deployment..."
    echo ""

    check_prerequisites
    configure_deployment
    authenticate_gcp
    build_image
    push_image
    deploy_to_cloud_run
    verify_deployment
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --artifact-registry)
            USE_ARTIFACT_REGISTRY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --project PROJECT_ID    GCP Project ID"
            echo "  --region REGION         GCP Region (default: us-central1)"
            echo "  --service NAME          Service name (default: l104-server)"
            echo "  --tag TAG               Image tag (default: latest)"
            echo "  --artifact-registry     Use Artifact Registry instead of GCR"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  GCP_PROJECT_ID          GCP Project ID"
            echo "  GCP_REGION              GCP Region"
            echo "  GEMINI_API_KEY          Gemini API Key"
            echo "  GEMINI_MODEL            Gemini model name"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main
