#!/bin/bash
# L104 Sovereign Node - Local Docker Deployment (No Cloud, No GitHub)
# This script runs the L104 node locally using Docker

set -e

echo "================================================"
echo "L104 Sovereign Node - Local Docker Deployment"
echo "================================================"
echo "⚡ Running locally - no GitHub or cloud required"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SERVICE_NAME="l104-server"
PORT="${PORT:-8081}"
DATA_DIR="${DATA_DIR:-./l104_data}"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}[!]${NC} Docker is not running. Please start Docker first."
    exit 1
fi

# Stop existing container if running
if docker ps -q -f name="${SERVICE_NAME}" | grep -q .; then
    echo -e "${BLUE}[i]${NC} Stopping existing container..."
    docker stop "${SERVICE_NAME}" > /dev/null 2>&1 || true
    docker rm "${SERVICE_NAME}" > /dev/null 2>&1 || true
fi

# Build the image
echo -e "${BLUE}[i]${NC} Building Docker image..."
docker build -t "${SERVICE_NAME}:local" .

# Run the container
echo -e "${BLUE}[i]${NC} Starting L104 Sovereign Node..."
docker run -d \
    --name "${SERVICE_NAME}" \
    -p "${PORT}:8081" \
    -p 8080:8080 \
    -p 4160:4160 \
    -p 4161:4161 \
    -p 2404:2404 \
    -v "$(pwd)/${DATA_DIR}:/data" \
    -e "GEMINI_API_KEY=${GEMINI_API_KEY:-not-configured}" \
    -e "GEMINI_MODEL=${GEMINI_MODEL:-gemini-1.5-flash}" \
    -e "RESONANCE=${RESONANCE:-527.5184818492612}" \
    -e "ENABLE_FAKE_GEMINI=${ENABLE_FAKE_GEMINI:-1}" \
    -e "MEMORY_DB_PATH=/data/memory.db" \
    -e "RAMNODE_DB_PATH=/data/ramnode.db" \
    -e "SELF_BASE_URL=http://localhost:${PORT}" \
    --restart unless-stopped \
    "${SERVICE_NAME}:local"

# Wait for startup
echo -e "${BLUE}[i]${NC} Waiting for service to start..."
sleep 5

# Health check
if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}[✓]${NC} L104 Sovereign Node is running!"
else
    echo -e "${YELLOW}[!]${NC} Service may still be starting..."
fi

echo ""
echo "================================================"
echo -e "${GREEN}Local Deployment Complete!${NC}"
echo "================================================"
echo "Service URL: http://localhost:${PORT}"
echo ""
echo "Useful commands:"
echo "  Health:     curl http://localhost:${PORT}/health"
echo "  Logs:       docker logs -f ${SERVICE_NAME}"
echo "  Stop:       docker stop ${SERVICE_NAME}"
echo "  Restart:    docker restart ${SERVICE_NAME}"
echo "================================================"
