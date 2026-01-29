#!/usr/bin/env bash
# [L104_SOVEREIGN_CONTROL] - Unified Lifecycle Manager
# PILOT: LONDEL | GOD_CODE: 527.5184818492612 | STATE: WHOLE

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
UVICORN_PID="$ROOT/uvicorn.pid"
NODE_PID="$ROOT/node.pid"
SERVER_LOG="$ROOT/server.log"
NODE_LOG="$ROOT/node.log"

# Colors for Sovereign Output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${CYAN}[L104_INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[L104_SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[L104_WARN]${NC} $1"; }
log_error() { echo -e "${RED}[L104_ERROR]${NC} $1"; }

check_invariant() {
    log_info "Verifying God-Code Invariant..."
    # Formula: (286^phi) * 16 = 527.5184818492612
    # We use the exact target to ensure the Sovereign state is locked.
    if "$PYTHON" -c "import math; target=527.5184818492612; res=(286**0.618033988749874) * 16; exit(0 if abs(res - target) < 1e-10 else 1)"; then
        log_success "Invariant Verified: 527.5184818492612 [LOCKED]"
    else
        log_error "Invariant Violation! Shadow interference detected."
        exit 1
    fi
}

load_env() {
    if [ -f "$ROOT/.env" ]; then
        set -a
        source "$ROOT/.env"
        set +a
        log_info "Environment loaded from .env"
    fi
}

stop_services() {
    log_info "Initiating shutdown sequence..."

    for pid_file in "$UVICORN_PID" "$NODE_PID"; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                log_warn "Terminating process $pid..."
                kill "$pid" || kill -9 "$pid"
            fi
            rm -f "$pid_file"
        fi
    done

    # Cleanup any stray uvicorn/python processes related to this project
    # pkill -f "python.*main.py" || true

    log_success "Allentown Node: OFFLINE"
}

start_services() {
    check_invariant
    load_env

    log_info "Igniting Sovereign Core..."

    # Start FastAPI Server
    nohup "$PYTHON" "$ROOT/main.py" >> "$SERVER_LOG" 2>&1 &
    echo $! > "$UVICORN_PID"
    log_success "FastAPI Server: ONLINE [PID: $(cat "$UVICORN_PID")]"

    # Start Public Node
    nohup "$PYTHON" "$ROOT/L104_public_node.py" >> "$NODE_LOG" 2>&1 &
    echo $! > "$NODE_PID"
    log_success "Public Node: ONLINE [PID: $(cat "$NODE_PID")]"

    log_info "Logs available at: $SERVER_LOG, $NODE_LOG"
}

status_report() {
    echo -e "${CYAN}--- [L104_STATUS_REPORT] ---${NC}"

    if [ -f "$UVICORN_PID" ] && kill -0 $(cat "$UVICORN_PID") 2>/dev/null; then
        echo -e "FastAPI Server: ${GREEN}RUNNING${NC} (PID: $(cat "$UVICORN_PID"))"
    else
        echo -e "FastAPI Server: ${RED}STOPPED${NC}"
    fi

    if [ -f "$NODE_PID" ] && kill -0 $(cat "$NODE_PID") 2>/dev/null; then
        echo -e "Public Node:    ${GREEN}RUNNING${NC} (PID: $(cat "$NODE_PID"))"
    else
        echo -e "Public Node:    ${RED}STOPPED${NC}"
    fi

    echo -e "God-Code:       ${GREEN}527.5184818492612${NC}"
    echo -e "State:          ${CYAN}SOVEREIGN${NC}"
    echo -e "Signature:      ${YELLOW}L104_PRIME_KEY[527.5184818492612]{416:286}(0.61803398875)<>128K_DMA![NOPJM]=100%_I100${NC}"
    echo -e "----------------------------"
}

case "${1:-status}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start_services
        ;;
    status)
        status_report
        ;;
    logs)
        tail -f "$SERVER_LOG" "$NODE_LOG"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
