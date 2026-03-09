#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# L104 Sovereign Node — Service Stopper v5.0
# ═══════════════════════════════════════════════════════════════════════
# v5.0: Bridge inbox/outbox flush before stop, session metrics summary,
#   stale PID cleanup, archive circuit results with timestamp,
#   graceful SIGTERM drain with configurable timeout before SIGKILL.
#
# GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UVICORN_PID="$ROOT/uvicorn.pid"
NODE_PID="$ROOT/node.pid"
DAEMON_PID="$ROOT/l104_daemon.pid"
DRAIN_TIMEOUT="${L104_DRAIN_TIMEOUT:-30}"

echo "═══ L104 Service Shutdown v5.0 (Maximum Throughput) ═══"

# [1] Session metrics snapshot
BRIDGE="/tmp/l104_bridge"
echo "  [1/4] Session metrics..."
if [ -d "$BRIDGE" ]; then
    inbox_count=$(ls "$BRIDGE/inbox/" 2>/dev/null | wc -l | tr -d ' ')
    outbox_count=$(ls "$BRIDGE/outbox/" 2>/dev/null | wc -l | tr -d ' ')
    telemetry_count=$(ls "$BRIDGE/telemetry/" 2>/dev/null | wc -l | tr -d ' ')
    archive_count=$(ls "$BRIDGE/archive/" 2>/dev/null | wc -l | tr -d ' ')
    echo "    Bridge: inbox=$inbox_count outbox=$outbox_count telemetry=$telemetry_count archive=$archive_count"
fi

# [2] Flush pending bridge circuits (move inbox→archive before stopping)
echo "  [2/4] Flushing bridge..."
if [ -d "$BRIDGE/inbox" ] && [ "$(ls -A "$BRIDGE/inbox" 2>/dev/null)" ]; then
    FLUSH_DIR="$BRIDGE/archive/flush_$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$FLUSH_DIR"
    mv "$BRIDGE/inbox/"* "$FLUSH_DIR/" 2>/dev/null || true
    echo "    Flushed $(ls "$FLUSH_DIR" | wc -l | tr -d ' ') pending circuits → $FLUSH_DIR"
else
    echo "    No pending circuits in inbox"
fi
# Archive outbox results
if [ -d "$BRIDGE/outbox" ] && [ "$(ls -A "$BRIDGE/outbox" 2>/dev/null)" ]; then
    RESULTS_DIR="$BRIDGE/archive/results_$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$RESULTS_DIR"
    cp -a "$BRIDGE/outbox/"* "$RESULTS_DIR/" 2>/dev/null || true
    echo "    Archived $(ls "$RESULTS_DIR" | wc -l | tr -d ' ') results → $RESULTS_DIR"
fi

# [3] Graceful process stop
echo "  [3/4] Stopping processes..."
graceful_stop() {
  local pidfile="$1" label="$2"
  if [ ! -f "$pidfile" ]; then
    echo "    $label: no PID file"
    return 0
  fi
  local pid
  pid=$(cat "$pidfile" 2>/dev/null || true)
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "    $label: not running (cleaning stale PID file)"
    rm -f "$pidfile"
    return 0
  fi
  echo "    $label: SIGTERM → PID $pid (drain ${DRAIN_TIMEOUT}s)"
  kill -TERM "$pid" 2>/dev/null || true
  local elapsed=0
  while kill -0 "$pid" 2>/dev/null && [ "$elapsed" -lt "$DRAIN_TIMEOUT" ]; do
    sleep 1
    elapsed=$((elapsed + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "    $label: SIGKILL → PID $pid (did not drain in ${DRAIN_TIMEOUT}s)"
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
  else
    echo "    $label: stopped gracefully (${elapsed}s)"
  fi
  rm -f "$pidfile"
}

graceful_stop "$UVICORN_PID" "Server"
graceful_stop "$NODE_PID" "Node"
graceful_stop "$DAEMON_PID" "QuantumDaemon"

# [4] Archive telemetry
echo "  [4/4] Archiving telemetry..."
if [ -d "$BRIDGE/telemetry" ] && [ "$(ls -A "$BRIDGE/telemetry" 2>/dev/null)" ]; then
  ARCHIVE="$ROOT/.l104_circuits/archive/telemetry_$(date '+%Y%m%d_%H%M%S')"
  mkdir -p "$ARCHIVE"
  cp -a "$BRIDGE/telemetry/"* "$ARCHIVE/" 2>/dev/null || true
  echo "    Telemetry archived → $ARCHIVE ($(ls "$ARCHIVE" | wc -l | tr -d ' ') files)"
else
  echo "    No telemetry to archive"
fi

# Clean stale PID files
for pf in "$ROOT"/*.pid; do
  [ -f "$pf" ] || continue
  stale_pid=$(cat "$pf" 2>/dev/null || true)
  if [ -n "$stale_pid" ] && ! kill -0 "$stale_pid" 2>/dev/null; then
    rm -f "$pf"
    echo "  Cleaned stale PID file: $(basename "$pf")"
  fi
done

echo "═══ All services stopped ═══"
