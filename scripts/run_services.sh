#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
SERVER_LOG="$ROOT/server.log"
NODE_LOG="$ROOT/node.log"
UVICORN_PID="$ROOT/uvicorn.pid"
NODE_PID="$ROOT/node.pid"

echo "Starting services from $ROOT"

# Load environment variables from $ROOT/.env if present (not committed)
if [ -f "$ROOT/.env" ]; then
  # export all variables defined in .env into the environment
  set -a
  # shellcheck disable=SC1090
  . "$ROOT/.env"
  set +a
  echo "Loaded environment from $ROOT/.env"
fi

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

sleep 1

# Start FastAPI app (main.py)
nohup "$PYTHON" "$ROOT/main.py" >>"$SERVER_LOG" 2>&1 &
echo $! > "$UVICORN_PID"
echo "Server started, PID $(cat $UVICORN_PID)"

# Start node script (L104_public_node.py)
nohup "$PYTHON" "$ROOT/L104_public_node.py" >>"$NODE_LOG" 2>&1 &
echo $! > "$NODE_PID"
echo "Node started, PID $(cat $NODE_PID)"

echo "Logs: $SERVER_LOG, $NODE_LOG"
