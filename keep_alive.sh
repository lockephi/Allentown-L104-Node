#!/bin/bash
# [L104_KEEP_ALIVE] - ENSURES THE SINGULARITY NEVER ENDS
# INVARIANT: 527.5184818492612 | PILOT: LONDEL

echo "--- [KEEP_ALIVE]: INITIALIZING SOVEREIGN WATCHDOG ---"

ROOT="/workspaces/Allentown-L104-Node"
cd "$ROOT"

while true; do
    # Check if the singularity is running
    if ! pgrep -f "l104_absolute_singularity.py" > /dev/null; then
        echo "--- [KEEP_ALIVE]: SINGULARITY COLLAPSED. RE-IGNITING... ---"
        nohup python3 "$ROOT/l104_absolute_singularity.py" >> "$ROOT/singularity.log" 2>&1 &
        echo "--- [KEEP_ALIVE]: SINGULARITY RE-IGNITED. PID: $! ---"
    fi

    # Check if the FastAPI server is running
    if ! pgrep -f "uvicorn main:app" > /dev/null; then
        echo "--- [KEEP_ALIVE]: UI SERVER DOWN. RESTARTING... ---"
        # Try to run via docker if available, else local
        if command -v docker > /dev/null && docker ps -a --format '{{.Names}}' | grep -q "l104-node-1"; then
             docker compose up -d
        else
             nohup uvicorn main:app --host 0.0.0.0 --port 8081 >> "$ROOT/uvicorn.log" 2>&1 &
        fi
        echo "--- [KEEP_ALIVE]: UI SERVER RESTART INITIATED ---"
    fi

    # Ping itself to prevent hibernation if this is running in a cloud/hosted env
    # curl -s http://localhost:8081/health > /dev/null

    sleep 10
done
