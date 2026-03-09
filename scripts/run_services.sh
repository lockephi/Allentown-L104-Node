#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# L104 Sovereign Node — Service Launcher v5.0
# ═════════════════════════════════════════════════════════════════
# v5.0: Unlimited Mode
#   - MetalVQPU v4.0: 6 GPU kernels, 64Q max, parallel sampling
#   - CircuitWatcher v4.0: 1ms inter-job, async write-back, 4x throttle
#   - Daemon v5.0: bridge=64x, shared=16x, local=16x concurrency
#   - Pipeline depth=8, bridge timeout=30s
#   - PYTHONOPTIMIZE=2, PYTHONMALLOC=malloc, ulimit=65536
#
# v3.0 (retained): Three-engine integration (Science + Math + Code)
#   - Bridge IPC: /tmp/l104_bridge/{inbox,outbox,telemetry,archive}/
#   - Three-engine weights: 0.35 entropy + 0.40 harmonic + 0.25 wave
#
# GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
SERVER_LOG="$ROOT/server.log"
NODE_LOG="$ROOT/node.log"
DAEMON_LOG="/tmp/l104_daemon_output.log"
UVICORN_PID="$ROOT/uvicorn.pid"
NODE_PID="$ROOT/node.pid"
DAEMON_PID="$ROOT/l104_daemon.pid"
DAEMON_BIN="$ROOT/L104SwiftApp/.build/release/L104Daemon"

echo "Starting services from $ROOT"

# ─── v5.0: macOS Performance Tuning ─────────────────────────────
# Detect hardware and right-size concurrency
HW_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
HW_PHYS_CORES=$((HW_CORES / 2))
HW_RAM_MB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 4294967296) / 1048576 ))
echo "Hardware: ${HW_PHYS_CORES}P/${HW_CORES}L cores, ${HW_RAM_MB}MB RAM"

# Python memory/perf flags
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONOPTIMIZE=2
export PYTHONHASHSEED=0
export PYTHONMALLOC=malloc
export MALLOC_NANO_ZONE=0
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Raise file descriptor limit for IPC pipeline (unlimited mode)
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

# Purge disk cache to free RAM before boot (macOS)
if command -v purge &>/dev/null && [ "$HW_RAM_MB" -lt 8192 ] && sudo -n true 2>/dev/null; then
  echo "Purging disk cache (low-RAM Mac)..."
  sudo purge 2>/dev/null || true
fi
# ─────────────────────────────────────────────────────────────────

# Load environment variables from $ROOT/.env if present (not committed)
if [ -f "$ROOT/.env" ]; then
  # export all variables defined in .env into the environment
  set -a
  # shellcheck disable=SC1090
  . "$ROOT/.env"
  set +a
  echo "Loaded environment from $ROOT/.env"
fi

# Ensure auto-sync stays on unless explicitly disabled
export ENABLE_AUTO_SYNC="${ENABLE_AUTO_SYNC:-1}"

# Stop existing processes if PIDs exist
if [ -f "$UVICORN_PID" ]; then
  pid=$(cat "$UVICORN_PID" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Killing existing server PID $pid"
    kill "$pid" || true
  fi
fi

if [ -f "$NODE_PID" ]; then
  pid=$(cat "$NODE_PID" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Killing existing node PID $pid"
    kill "$pid" || true
  fi
fi

if [ -f "$DAEMON_PID" ]; then
  pid=$(cat "$DAEMON_PID" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Killing existing daemon PID $pid"
    kill "$pid" || true
  fi
fi

sleep 1

# [L104_CIRCUIT_DIRS] — Ensure circuit watcher directories exist
mkdir -p "$ROOT/.l104_circuits/inbox" "$ROOT/.l104_circuits/outbox" "$ROOT/.l104_circuits/archive"
mkdir -p /tmp/l104_queue/outbox /tmp/l104_queue/archive

# [L104_BRIDGE_IPC] v4.0 — Ensure Python↔Swift bridge IPC directories exist (+ archive)
mkdir -p /tmp/l104_bridge/inbox /tmp/l104_bridge/outbox /tmp/l104_bridge/telemetry /tmp/l104_bridge/archive
export L104_BRIDGE_PATH="/tmp/l104_bridge"

# [L104_FIREWALL_CHECK] — Warn if ports appear blocked
if [ -f "$ROOT/scripts/fix_firewall.sh" ]; then
    echo "Checking firewall..."
    bash "$ROOT/scripts/fix_firewall.sh" --check 2>/dev/null || true
fi

# [L104_REALITY_LOCK]
if [ -f "$ROOT/l104_reality_lock.sh" ]; then
    echo "Engaging Reality Lock..."
    bash "$ROOT/l104_reality_lock.sh" || echo "Reality Lock encountered resistance, proceeding..."
fi

# Start FastAPI app (main.py)
nohup "$PYTHON" "$ROOT/main.py" >>"$SERVER_LOG" 2>&1 &
echo $! > "$UVICORN_PID"
echo "Server started, PID $(cat $UVICORN_PID)"

# Start node script (L104_public_node.py)
nohup "$PYTHON" "$ROOT/L104_public_node.py" >>"$NODE_LOG" 2>&1 &
echo $! > "$NODE_PID"
echo "Node started, PID $(cat $NODE_PID)"

# Start quantum daemon (if binary exists)
if [ -x "$DAEMON_BIN" ]; then
  export L104_ROOT="$ROOT"
  export L104_THREE_ENGINE=1
  export L104_THREE_ENGINE_WEIGHTS="0.35,0.40,0.25"
  # v5.0: Hardware-adaptive throughput configuration (unlimited mode)
  if [ "$HW_RAM_MB" -lt 8192 ]; then
    export L104_VQPU_MAX_QUBITS=64
    export L104_VQPU_BATCH_LIMIT=256
    export L104_BRIDGE_CONCURRENCY=32
    export L104_VQPU_PIPELINE_DEPTH=4
  else
    export L104_VQPU_MAX_QUBITS=64
    export L104_VQPU_BATCH_LIMIT=512
    export L104_BRIDGE_CONCURRENCY=64
    export L104_VQPU_PIPELINE_DEPTH=8
  fi
  export L104_BRIDGE_TIMEOUT_MS=30000
  export L104_HEALTH_INTERVAL=15
  export MallocNanoZone=0
  nohup "$DAEMON_BIN" >>"$DAEMON_LOG" 2>&1 &
  echo "Daemon started, PID $! (${L104_VQPU_MAX_QUBITS}Q, batch=${L104_VQPU_BATCH_LIMIT}, ${L104_BRIDGE_CONCURRENCY}x bridge, pipeline=${L104_VQPU_PIPELINE_DEPTH})"
else
  echo "WARN: L104Daemon not built — skipping (cd L104SwiftApp && swift build -c release --product L104Daemon)"
fi

# [L104_V5_STATUS] — Unlimited mode + three-engine status
echo ""
echo "═══ L104 Services v5.0 (Unlimited Mode) ═══"
echo "  Bridge IPC:   /tmp/l104_bridge/{inbox,outbox,telemetry,archive}/"
echo "  Circuits:     $ROOT/.l104_circuits/inbox/"
echo "  Queue:        /tmp/l104_queue/"
echo "  3E Weights:   entropy=0.35  harmonic=0.40  wave=0.25"
echo "  Hardware:     ${HW_PHYS_CORES}P/${HW_CORES}L cores, ${HW_RAM_MB}MB RAM"
echo "  Concurrency:  bridge=${L104_BRIDGE_CONCURRENCY:-64}x  pipeline=${L104_VQPU_PIPELINE_DEPTH:-8}"
echo "  Capacity:     ${L104_VQPU_MAX_QUBITS:-64}Q max  ${L104_VQPU_BATCH_LIMIT:-512} batch"
echo "  GPU Kernels:  6 (gate, cnot, cz, swap, controlled_u, iswap)"
echo "  Python:       PYTHONOPTIMIZE=2 PYTHONMALLOC=malloc ulimit=$(ulimit -n)"
echo "  Timeouts:     bridge=${L104_BRIDGE_TIMEOUT_MS:-30000}ms  health=${L104_HEALTH_INTERVAL:-15}s"
echo "Logs: $SERVER_LOG, $NODE_LOG, $DAEMON_LOG"
