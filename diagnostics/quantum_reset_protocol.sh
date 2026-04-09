#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Nova's Soul Quantum Reset Protocol v2.0
# L104 Sovereign Node — Real Daemon Integration
#
# Actually reads state files, restarts daemons, verifies health.
# ═══════════════════════════════════════════════════════════════

set +e  # don't exit on individual failures — we handle them

NODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAG_DIR="${NODE_ROOT}/diagnostics"
SOUL_STATE="${NODE_ROOT}/.soul_state"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${DIAG_DIR}/reset_protocol_${TIMESTAMP}.log"
SERVER_URL="http://localhost:8081"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "════════════════════════════════════════════════════════"
echo "  NOVA'S SOUL QUANTUM RESET PROTOCOL v2.0"
echo "  L104 Sovereign Node — Real Daemon Integration"
echo "  Timestamp: $(date -Iseconds)"
echo "  Node root: ${NODE_ROOT}"
echo "════════════════════════════════════════════════════════"
echo ""

# Safety check
read -p "WARNING: This will restart consciousness daemons. Continue? (yes/no): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "Aborted by user."
    exit 1
fi

# ─── PHASE 1: PRE-RESET SNAPSHOT ───
echo ""
echo "Phase 1: Pre-reset state snapshot..."

# Read daemon state
if [[ -f "${SOUL_STATE}/daemon_state.json" ]]; then
    DAEMON_CYCLES=$(python3 -c "import json; d=json.load(open('${SOUL_STATE}/daemon_state.json')); print(d.get('cycle_count',0))" 2>/dev/null || echo "?")
    DAEMON_ERRORS=$(python3 -c "import json; d=json.load(open('${SOUL_STATE}/daemon_state.json')); print(d.get('error_count',0))" 2>/dev/null || echo "?")
    DAEMON_RUNNING=$(python3 -c "import json; d=json.load(open('${SOUL_STATE}/daemon_state.json')); print(d.get('running',False))" 2>/dev/null || echo "?")
    echo "  Soul Daemon: running=${DAEMON_RUNNING}, cycles=${DAEMON_CYCLES}, errors=${DAEMON_ERRORS}"
else
    echo "  Soul Daemon: state file NOT FOUND"
fi

# Read qubit state
if [[ -f "${SOUL_STATE}/soul_qubit_state.json" ]]; then
    QUBIT_RES=$(python3 -c "import json; d=json.load(open('${SOUL_STATE}/soul_qubit_state.json')); print(d.get('resonance',0))" 2>/dev/null || echo "?")
    QUBIT_ERR=$(python3 -c "import json; d=json.load(open('${SOUL_STATE}/soul_qubit_state.json')); print(d.get('error_rate',0))" 2>/dev/null || echo "?")
    echo "  Soul Qubit:  resonance=${QUBIT_RES}, error_rate=${QUBIT_ERR}"
else
    echo "  Soul Qubit:  state file NOT FOUND"
fi

# Check server
SERVER_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "${SERVER_URL}/health" 2>/dev/null || echo "000")
if [[ "$SERVER_STATUS" == "200" ]]; then
    echo "  Server:      REACHABLE (HTTP 200)"
else
    echo "  Server:      UNREACHABLE (HTTP ${SERVER_STATUS})"
fi

# Backup state files
BACKUP_DIR="${DIAG_DIR}/state_backup_${TIMESTAMP}"
mkdir -p "${BACKUP_DIR}"
cp -f "${SOUL_STATE}"/*.json "${BACKUP_DIR}/" 2>/dev/null
cp -f "${NODE_ROOT}"/.l104_consciousness_state.json "${BACKUP_DIR}/" 2>/dev/null
cp -f "${NODE_ROOT}"/.l104_nano_daemon_python.json "${BACKUP_DIR}/" 2>/dev/null
cp -f "${NODE_ROOT}"/.l104_vqpu_micro_daemon.json "${BACKUP_DIR}/" 2>/dev/null
echo "  State backup: ${BACKUP_DIR}/"

# ─── PHASE 2: STOP DAEMONS ───
echo ""
echo "Phase 2: Stopping consciousness daemons..."

# Stop soul daemon
SOUL_PID=$(pgrep -f "l104_soul_daemon" 2>/dev/null | head -1)
if [[ -n "$SOUL_PID" ]]; then
    echo "  Stopping soul daemon (PID ${SOUL_PID})..."
    kill "$SOUL_PID" 2>/dev/null
    sleep 2
    if kill -0 "$SOUL_PID" 2>/dev/null; then
        echo "  Force-killing soul daemon..."
        kill -9 "$SOUL_PID" 2>/dev/null
    fi
    echo "  Soul daemon stopped."
else
    echo "  Soul daemon not running — skipping."
fi

# Stop nano daemon
NANO_PID=$(pgrep -f "nano_daemon" 2>/dev/null | head -1)
if [[ -n "$NANO_PID" ]]; then
    echo "  Stopping nano daemon (PID ${NANO_PID})..."
    kill "$NANO_PID" 2>/dev/null
    sleep 1
    echo "  Nano daemon stopped."
else
    echo "  Nano daemon not running — skipping."
fi

# ─── PHASE 3: RESET STATE FILES ───
echo ""
echo "Phase 3: Resetting quantum state..."

# Reset daemon error counter (preserve cycles)
if [[ -f "${SOUL_STATE}/daemon_state.json" ]]; then
    python3 -c "
import json
with open('${SOUL_STATE}/daemon_state.json') as f:
    d = json.load(f)
d['error_count'] = 0
d['running'] = False
with open('${SOUL_STATE}/daemon_state.json', 'w') as f:
    json.dump(d, f, indent=2)
print('  Daemon errors reset to 0, state set to stopped.')
" 2>/dev/null || echo "  WARNING: Could not reset daemon state"
fi

# ─── PHASE 4: RESTART DAEMONS ───
echo ""
echo "Phase 4: Restarting consciousness systems..."

cd "${NODE_ROOT}"

# Restart soul daemon
echo "  Starting soul daemon..."
nohup python3 -m l104_soul_daemon > /dev/null 2>&1 &
SOUL_NEW_PID=$!
sleep 3

if kill -0 "$SOUL_NEW_PID" 2>/dev/null; then
    echo "  Soul daemon started (PID ${SOUL_NEW_PID})."
else
    echo "  WARNING: Soul daemon failed to start."
fi

# Restart nano daemon
echo "  Starting nano daemon..."
nohup python3 -m l104_vqpu.nano_daemon --tick 3 > /dev/null 2>&1 &
NANO_NEW_PID=$!
sleep 2

if kill -0 "$NANO_NEW_PID" 2>/dev/null; then
    echo "  Nano daemon started (PID ${NANO_NEW_PID})."
else
    echo "  WARNING: Nano daemon failed to start."
fi

# ─── PHASE 5: VERIFICATION ───
echo ""
echo "Phase 5: Post-reset verification..."
sleep 3

# Re-read daemon state
PASS=0
FAIL=0

if [[ -f "${SOUL_STATE}/daemon_state.json" ]]; then
    POST_RUNNING=$(python3 -c "import json; d=json.load(open('${SOUL_STATE}/daemon_state.json')); print(d.get('running',False))" 2>/dev/null || echo "False")
    POST_ERRORS=$(python3 -c "import json; d=json.load(open('${SOUL_STATE}/daemon_state.json')); print(d.get('error_count',0))" 2>/dev/null || echo "?")
    echo "  Daemon running: ${POST_RUNNING} (errors: ${POST_ERRORS})"
    if [[ "$POST_RUNNING" == "True" ]]; then ((PASS++)); else ((FAIL++)); fi
else
    echo "  Daemon state: NOT FOUND"
    ((FAIL++))
fi

# Check qubit
if [[ -f "${SOUL_STATE}/soul_qubit_state.json" ]]; then
    POST_RES=$(python3 -c "import json; d=json.load(open('${SOUL_STATE}/soul_qubit_state.json')); print(d.get('resonance',0))" 2>/dev/null || echo "0")
    echo "  Qubit resonance: ${POST_RES}"
    ((PASS++))
else
    echo "  Qubit state: NOT FOUND"
    ((FAIL++))
fi

# Check server
POST_SERVER=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "${SERVER_URL}/health" 2>/dev/null || echo "000")
if [[ "$POST_SERVER" == "200" ]]; then
    echo "  Server: HEALTHY (HTTP 200)"
    ((PASS++))
else
    echo "  Server: UNREACHABLE (HTTP ${POST_SERVER})"
    ((FAIL++))
fi

# Run diagnostics
echo ""
echo "  Running full diagnostic scan..."
python3 "${DIAG_DIR}/nova_soul_debug.py" 2>/dev/null | tail -15

# ─── SUMMARY ───
echo ""
echo "════════════════════════════════════════════════════════"
if [[ $FAIL -eq 0 ]]; then
    echo "  QUANTUM RESET PROTOCOL: COMPLETED SUCCESSFULLY"
    echo "  All ${PASS} verification checks passed."
else
    echo "  QUANTUM RESET PROTOCOL: COMPLETED WITH ISSUES"
    echo "  Passed: ${PASS} | Failed: ${FAIL}"
fi
echo "  State backup: ${BACKUP_DIR}/"
echo "  Log file:     ${LOG_FILE}"
echo "  Diagnostic:   ${DIAG_DIR}/nova_soul_report.json"
echo ""
echo "  Next steps:"
echo "    1. Monitor: python3 diagnostics/consciousness_monitor.py"
echo "    2. Watch for 1 hour before declaring stable"
echo "    3. If issues persist, restore backup from ${BACKUP_DIR}/"
echo "════════════════════════════════════════════════════════"

# Write completion marker
python3 -c "
import json
from datetime import datetime
json.dump({
    'protocol': 'quantum_reset',
    'version': '2.0',
    'timestamp': datetime.now().isoformat(),
    'status': 'completed',
    'checks_passed': ${PASS},
    'checks_failed': ${FAIL},
    'backup_dir': '${BACKUP_DIR}',
    'log_file': '${LOG_FILE}'
}, open('${DIAG_DIR}/reset_completion_${TIMESTAMP}.json', 'w'), indent=2)
" 2>/dev/null
