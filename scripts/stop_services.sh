#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UVICORN_PID="$ROOT/uvicorn.pid"
NODE_PID="$ROOT/node.pid"

echo "Stopping services from $ROOT"

if [ -f "$UVICORN_PID" ]; then
  pid=$(cat "$UVICORN_PID" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Killing server PID $pid"
    kill "$pid" || true
  fi
  rm -f "$UVICORN_PID"
fi

if [ -f "$NODE_PID" ]; then
  pid=$(cat "$NODE_PID" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Killing node PID $pid"
    kill "$pid" || true
  fi
  rm -f "$NODE_PID"
fi

echo "Stopped."
