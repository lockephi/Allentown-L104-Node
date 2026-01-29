#!/bin/bash
# [L104_WATCHDOG] - ETERNAL VIGILANCE
# INVARIANT: 527.5184818492612 | PILOT: LONDEL

PYTHON_EXEC="/workspaces/Allentown-L104-Node/.venv/bin/python"
SCRIPT_PATH="/workspaces/Allentown-L104-Node/l104_global_network_manager.py"

echo "--- [WATCHDOG]: STARTING ETERNAL VIGILANCE ---"

while true; do
    echo "--- [WATCHDOG]: LAUNCHING L104 GLOBAL NETWORK ---"
    $PYTHON_EXEC $SCRIPT_PATH

    EXIT_CODE=$?
    echo "--- [WATCHDOG]: L104 EXITED WITH CODE $EXIT_CODE ---"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "--- [WATCHDOG]: CRASH DETECTED. INITIATING REINCARNATION... ---"
    else
        echo "--- [WATCHDOG]: NORMAL EXIT. RESTARTING FOR CONTINUITY... ---"
    fi

    sleep 5
done
