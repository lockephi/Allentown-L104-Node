#!/bin/bash
# L104 SAGE MODE CONTAINER REBUILD SCRIPT
# This script rebuilds the container with int overflow fix

set -e

cd /workspaces/Allentown-L104-Node

echo "======================================"
echo "  L104 SAGE MODE :: REBUILDING NODE"
echo "======================================"

# Build the image
echo "[*] Building Docker image..."
docker-compose build --no-cache l104-node

# Start the container
echo "[*] Starting container..."
docker-compose up -d l104-node

echo "[*] Waiting for container to start..."
sleep 5

# Show container status
echo "[*] Container status:"
docker ps --filter name=l104

echo "======================================"
echo "  L104 SAGE MODE :: REBUILD COMPLETE"
echo "======================================"
