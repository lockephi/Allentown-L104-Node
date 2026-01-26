#!/bin/bash
# [L104_POST_START] - Auto-start the node when Codespaces resumes
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

set -e

echo "--- [L104_POST_START]: CHECKING NODE STATUS ---"

cd /workspaces/Allentown-L104-Node

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "--- [L104_POST_START]: Docker not available yet, waiting... ---"
    sleep 5
fi

# Check if the container exists and start it if needed
CONTAINER_NAME="allentown-l104-node-l104-node-1"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Container exists, check if running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "--- [L104_POST_START]: CONTAINER EXISTS BUT STOPPED. STARTING... ---"
        docker compose up -d
        echo "--- [L104_POST_START]: NODE STARTED ---"
    else
        echo "--- [L104_POST_START]: NODE ALREADY RUNNING ---"
    fi
else
    echo "--- [L104_POST_START]: CONTAINER NOT FOUND. BUILDING AND STARTING... ---"
    docker compose up -d --build
    echo "--- [L104_POST_START]: NODE BUILT AND STARTED ---"
fi

# Wait for health check
echo "--- [L104_POST_START]: WAITING FOR HEALTH CHECK... ---"
for i in {1..30}; do
    if docker ps --filter "name=${CONTAINER_NAME}" --filter "health=healthy" | grep -q "${CONTAINER_NAME}"; then
        echo "--- [L104_POST_START]: NODE HEALTHY AND ONLINE ---"
        exit 0
    fi
    sleep 2
done

echo "--- [L104_POST_START]: NODE STARTED (health check pending) ---"
