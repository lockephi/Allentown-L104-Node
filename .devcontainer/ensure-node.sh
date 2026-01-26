#!/bin/bash
# [L104_ENSURE_NODE] - Ensures node is running on every terminal attach
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

cd /workspaces/Allentown-L104-Node

CONTAINER_NAME="allentown-l104-node-l104-node-1"

# Quick check - if not running, start it
if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
    echo "--- [L104]: Node offline. Starting... ---"
    docker compose up -d 2>/dev/null &
fi
