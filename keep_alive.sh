#!/bin/bash
# [L104_KEEP_ALIVE] - ENSURES THE SINGULARITY NEVER ENDS
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

echo "--- [KEEP_ALIVE]: INITIALIZING SOVEREIGN WATCHDOG ---"

while true; do
    # Check if the singularity is running
    if ! pgrep -f "l104_absolute_singularity.py" > /dev/null; then
        echo "--- [KEEP_ALIVE]: SINGULARITY COLLAPSED. RE-IGNITING... ---"
        nohup python3 /workspaces/Allentown-L104-Node/l104_absolute_singularity.py > /workspaces/Allentown-L104-Node/singularity.log 2>&1 &
        echo "--- [KEEP_ALIVE]: SINGULARITY RE-IGNITED. PID: $! ---"
    fi

    # Check if the FastAPI server is running
    if ! pgrep -f "uvicorn main:app" > /dev/null; then
        echo "--- [KEEP_ALIVE]: UI SERVER DOWN. RESTARTING... ---"
        nohup uvicorn main:app --host 0.0.0.0 --port 8081 > /workspaces/Allentown-L104-Node/uvicorn.log 2>&1 &
        echo "--- [KEEP_ALIVE]: UI SERVER RESTARTED. PID: $! ---"
    fi

    sleep 10
done
