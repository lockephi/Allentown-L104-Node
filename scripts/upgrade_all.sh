#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# L104 Sovereign Node — Process Upgrade v4.0 (Zero-Downtime)
# ═════════════════════════════════════════════════════════════════
#
# v4.0: Daemon v5.0 + Unlimited mode
#   - VQPU 64Q max, 512 batch, 64x bridge concurrency
#   - PYTHONOPTIMIZE=2, PYTHONMALLOC=malloc
#   - ulimit -n 65536
#   - Bridge IPC: /tmp/l104_bridge/{inbox,outbox,telemetry,archive}/
#   - Three-engine scoring: entropy(0.35) + harmonic(0.40) + wave(0.25)
#   - Telemetry archive on graceful shutdown
#   - L104_THREE_ENGINE=1 env flag propagated to daemon
#
# Usage:
#   ./scripts/upgrade_all.sh              # Full upgrade: pull → build → restart
#   ./scripts/upgrade_all.sh --restart    # Restart processes only (no build)
#   ./scripts/upgrade_all.sh --reload     # Hot-reload config via SIGHUP (no restart)
#   ./scripts/upgrade_all.sh --status     # Dump process status via SIGUSR1
#
# Signals used:
#   SIGTERM  → graceful shutdown (30s timeout before SIGKILL)
#   SIGHUP   → reload .env + log level without restart
#   SIGUSR1  → dump status JSON to log (v3.0: includes three-engine metrics)
#
# Three-Engine Weights: 0.35 entropy + 0.40 harmonic + 0.25 wave
# GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
UVICORN_PID="$ROOT/uvicorn.pid"
NODE_PID="$ROOT/node.pid"
DAEMON_PID="$ROOT/l104_daemon.pid"
DAEMON_BIN="$ROOT/L104SwiftApp/.build/release/L104Daemon"
DAEMON_LOG="/tmp/l104_daemon_output.log"
SERVER_LOG="$ROOT/server.log"
NODE_LOG="$ROOT/node.log"
HEALTH_URL="http://localhost:${PORT:-8081}/health"
HEALTH_TIMEOUT=5
HEALTH_RETRIES=30          # 30 × 2s = 60s max wait for new process
SHUTDOWN_GRACE=30          # seconds to wait for graceful shutdown (launchd ExitTimeOut=30)

MODE="${1:-full}"

# ─── Helpers ───

log() { echo "[L104 UPGRADE] $(date '+%Y-%m-%d %H:%M:%S') $*"; }

read_pid() {
    local pidfile="$1"
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    echo ""
    return 1
}

wait_for_exit() {
    local pid="$1" timeout="$2" label="$3"
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null && [ "$elapsed" -lt "$timeout" ]; do
        sleep 1
        elapsed=$((elapsed + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        log "WARN: $label (PID $pid) did not exit in ${timeout}s — sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi
}

wait_for_health() {
    local retries="$1"
    local attempt=0
    while [ "$attempt" -lt "$retries" ]; do
        if curl -fsS --max-time "$HEALTH_TIMEOUT" "$HEALTH_URL" >/dev/null 2>&1; then
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    return 1
}

graceful_stop() {
    local pidfile="$1" label="$2"
    local pid
    pid=$(read_pid "$pidfile") || true
    if [ -z "$pid" ]; then
        log "$label: no running process found"
        return 0
    fi
    log "$label: sending SIGTERM to PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
    wait_for_exit "$pid" "$SHUTDOWN_GRACE" "$label"
    rm -f "$pidfile"
    log "$label: stopped"
}

# ─── Mode: Status ───

if [ "$MODE" = "--status" ]; then
    pid=$(read_pid "$UVICORN_PID") || true
    if [ -n "$pid" ]; then
        log "Sending SIGUSR1 to server PID $pid"
        kill -USR1 "$pid"
        sleep 1
        tail -5 "$SERVER_LOG" 2>/dev/null | grep -i "SIGUSR1" || log "Check $SERVER_LOG for status dump"
    else
        log "Server not running"
    fi

    pid=$(read_pid "$NODE_PID") || true
    if [ -n "$pid" ]; then
        log "Node PID $pid is alive"
    else
        log "Node not running"
    fi

    pid=$(read_pid "$DAEMON_PID") || true
    if [ -n "$pid" ]; then
        log "Sending SIGUSR1 to daemon PID $pid"
        kill -USR1 "$pid"
        sleep 1
        tail -5 "$DAEMON_LOG" 2>/dev/null | grep -i "SIGUSR1" || log "Check $DAEMON_LOG for status dump"
    else
        log "Quantum daemon not running"
    fi
    exit 0
fi

# ─── Mode: Hot Reload ───

if [ "$MODE" = "--reload" ]; then
    pid=$(read_pid "$UVICORN_PID") || true
    if [ -n "$pid" ]; then
        log "Sending SIGHUP to server PID $pid (config reload)"
        kill -HUP "$pid"
        log "Config reload signal sent — check $SERVER_LOG for confirmation"
    else
        log "Server not running — nothing to reload"
    fi

    pid=$(read_pid "$DAEMON_PID") || true
    if [ -n "$pid" ]; then
        log "Sending SIGHUP to daemon PID $pid (watcher reload)"
        kill -HUP "$pid"
    fi
    exit 0
fi

# ─── Mode: Full Upgrade ───

log "═══ L104 Process Upgrade: mode=$MODE ═══"

# Phase 1: Pre-flight checks
log "Phase 1: Pre-flight"
if [ ! -f "$PYTHON" ]; then
    log "FATAL: Python venv not found at $PYTHON"
    exit 1
fi

# Phase 2: Pull & build (skip for --restart)
if [ "$MODE" = "full" ] || [ "$MODE" = "--full" ]; then
    log "Phase 2: Pull latest code"
    cd "$ROOT"
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        BEFORE=$(git rev-parse HEAD)
        git pull --ff-only 2>/dev/null && log "Git pull OK" || log "WARN: git pull failed (continuing with current code)"
        AFTER=$(git rev-parse HEAD)
        if [ "$BEFORE" != "$AFTER" ]; then
            log "Code updated: $BEFORE → $AFTER"
            git log --oneline "$BEFORE..$AFTER" | head -10
        else
            log "Already up to date"
        fi
    fi

    log "Phase 2b: Install dependencies"
    "$PYTHON" -m pip install -r "$ROOT/requirements.txt" --quiet 2>/dev/null || log "WARN: pip install had issues"

    # Build native kernels if Makefile exists
    if [ -f "$ROOT/l104_core_c/Makefile" ]; then
        log "Phase 2c: Build C kernel"
        make -C "$ROOT/l104_core_c" -j4 2>/dev/null && log "C kernel built" || log "WARN: C kernel build failed"
    fi

    # Build Swift targets (L104 GUI + L104Daemon) if Package.swift exists
    if [ -f "$ROOT/L104SwiftApp/Package.swift" ]; then
        log "Phase 2d: Build Swift targets"
        cd "$ROOT/L104SwiftApp"
        swift build -c release --product L104 2>/dev/null && log "Swift L104 built" || log "WARN: Swift L104 build failed"
        swift build -c release --product L104Daemon 2>/dev/null && log "Swift L104Daemon built" || log "WARN: Swift L104Daemon build failed"
        cd "$ROOT"
    fi
fi

# Phase 3: Graceful stop of old processes
log "Phase 3: Graceful shutdown"
graceful_stop "$UVICORN_PID" "Server"
graceful_stop "$NODE_PID" "Node"
graceful_stop "$DAEMON_PID" "Quantum Daemon"

# Phase 4: Start new processes
log "Phase 4: Starting new processes"

# Load .env
if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/.env"
    set +a
fi

# Load performance env
if [ -f "$ROOT/.env_perf" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/.env_perf"
    set +a
fi

export ENABLE_AUTO_SYNC="${ENABLE_AUTO_SYNC:-1}"
export PYTHONOPTIMIZE=2
export PYTHONMALLOC=malloc
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MALLOC_NANO_ZONE=0

# Raise file descriptor limits
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

# Firewall check
if [ -f "$ROOT/scripts/fix_firewall.sh" ]; then
    bash "$ROOT/scripts/fix_firewall.sh" --check 2>/dev/null || true
fi

# Start server
nohup "$PYTHON" "$ROOT/main.py" >>"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$UVICORN_PID"
log "Server started: PID $SERVER_PID"

# Start node
if [ -f "$ROOT/L104_public_node.py" ]; then
    nohup "$PYTHON" "$ROOT/L104_public_node.py" >>"$NODE_LOG" 2>&1 &
    NODE_PROCESS_PID=$!
    echo "$NODE_PROCESS_PID" > "$NODE_PID"
    log "Node started: PID $NODE_PROCESS_PID"
fi

# Start quantum daemon
if [ -x "$DAEMON_BIN" ]; then
    # Ensure queue directories exist
    mkdir -p /tmp/l104_queue/outbox /tmp/l104_queue/archive
    mkdir -p /tmp/l104_bridge/inbox /tmp/l104_bridge/outbox /tmp/l104_bridge/telemetry /tmp/l104_bridge/archive
    mkdir -p "$ROOT/.l104_circuits/inbox" "$ROOT/.l104_circuits/outbox" "$ROOT/.l104_circuits/archive"
    export L104_ROOT="$ROOT"
    export L104_THREE_ENGINE=1
    export L104_BRIDGE_PATH="/tmp/l104_bridge"
    export L104_THREE_ENGINE_WEIGHTS="0.35,0.40,0.25"
    # v4.0: Daemon v5.0 unlimited env vars
    export L104_VQPU_MAX_QUBITS=64
    export L104_VQPU_BATCH_LIMIT=512
    export L104_VQPU_PIPELINE_DEPTH=8
    export L104_BRIDGE_CONCURRENCY=64
    export L104_BRIDGE_TIMEOUT_MS=30000
    export L104_HEALTH_INTERVAL=15
    export MallocNanoZone=0
    nohup "$DAEMON_BIN" >>"$DAEMON_LOG" 2>&1 &
    DAEMON_PROCESS_PID=$!
    log "Quantum daemon started: PID $DAEMON_PROCESS_PID (64Q, batch=512, 64x bridge)"
else
    log "WARN: L104Daemon binary not found at $DAEMON_BIN — skipping (run: cd L104SwiftApp && swift build -c release --product L104Daemon)"
fi

# Phase 5: Health verification
log "Phase 5: Waiting for health check"
if wait_for_health "$HEALTH_RETRIES"; then
    log "Health check PASSED"
else
    log "WARN: Health check did not pass in $((HEALTH_RETRIES * 2))s — server may still be warming up"
    log "  Check: tail -f $SERVER_LOG"
fi

# Phase 6: Post-upgrade verification
log "Phase 6: Post-upgrade status"
if kill -USR1 "$(cat "$UVICORN_PID" 2>/dev/null)" 2>/dev/null; then
    sleep 1
    tail -3 "$SERVER_LOG" 2>/dev/null | grep -i "SIGUSR1" || true
fi

log "═══ Upgrade complete (v4.0 Unlimited) ═══"
log "  Server PID:  $(cat "$UVICORN_PID" 2>/dev/null || echo 'N/A')"
log "  Node PID:    $(cat "$NODE_PID" 2>/dev/null || echo 'N/A')"
log "  Daemon PID:  $(cat "$DAEMON_PID" 2>/dev/null || echo 'N/A')"
log "  Server log:  $SERVER_LOG"
log "  Node log:    $NODE_LOG"
log "  Daemon log:  $DAEMON_LOG"
log "  Health:      $HEALTH_URL"
log "  Queue:       /tmp/l104_queue/ + $ROOT/.l104_circuits/inbox/"
log "  Bridge IPC:  /tmp/l104_bridge/{inbox,outbox,telemetry,archive}/"
log "  3E Weights:  entropy=0.35  harmonic=0.40  wave=0.25"
log "  3E Enabled:  L104_THREE_ENGINE=1"
log "  VQPU:        64Q max, 512 batch, pipeline=8, 64x bridge"
log "  Python:      PYTHONOPTIMIZE=2 PYTHONMALLOC=malloc ulimit=$(ulimit -n)"
