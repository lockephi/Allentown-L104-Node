#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# L104 Sovereign Node — macOS Persistent Service Manager v3.1
# ═══════════════════════════════════════════════════════════════════
# Manages launchd LaunchAgents that keep L104 services running
# permanently on macOS, auto-restarting on crash and surviving reboot.
#
# Services:
#   com.l104.fast-server        — FastAPI server (main.py, port 8081)
#   com.l104.node-server        — Public node (L104_public_node.py)
#   com.l104.vqpu-daemon        — Metal VQPU daemon (L104Daemon Swift binary)
#   com.l104.vqpu-micro-daemon  — VQPU micro daemon (5-15s micro-task loop)
#   com.l104.auto-update        — Auto-update watcher (git pull + rebuild)
#   com.l104.log-rotate         — Automatic log rotation (every 30m)
#   com.l104.health-watchdog    — Health watchdog (60s checks)
#   com.l104.boot-manager       — Boot manager (restart-on-boot supervisor)
#
# v3.1: Added vqpu-micro-daemon service, micro IPC directories
# v3.0: Added log-rotate service, health command, bridge diagnostics,
#   session uptime tracking, improved status with version + priority.
#
# Usage:
#   ./scripts/l104_service_ctl.sh start     # Load & start all services
#   ./scripts/l104_service_ctl.sh stop      # Unload all services
#   ./scripts/l104_service_ctl.sh restart   # Stop → Start
#   ./scripts/l104_service_ctl.sh status    # Show running state
#   ./scripts/l104_service_ctl.sh health    # Deep health check
#   ./scripts/l104_service_ctl.sh logs      # Tail all log files
#   ./scripts/l104_service_ctl.sh install   # Copy plists + load
#   ./scripts/l104_service_ctl.sh uninstall # Unload + remove plists
#
# GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$ROOT/logs"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

SERVICES=(
    "com.l104.fast-server"
    "com.l104.node-server"
    "com.l104.vqpu-daemon"
    "com.l104.vqpu-micro-daemon"
    "com.l104.auto-update"
    "com.l104.log-rotate"
    "com.l104.health-watchdog"
    "com.l104.boot-manager"
)

# Ensure IPC directories exist
ensure_ipc_dirs() {
    mkdir -p "$ROOT/.l104_circuits/inbox" "$ROOT/.l104_circuits/outbox" "$ROOT/.l104_circuits/archive"
    mkdir -p /tmp/l104_queue/outbox /tmp/l104_queue/archive
    mkdir -p /tmp/l104_bridge/inbox /tmp/l104_bridge/outbox /tmp/l104_bridge/telemetry /tmp/l104_bridge/archive
    mkdir -p /tmp/l104_bridge/micro/inbox /tmp/l104_bridge/micro/outbox
    mkdir -p /tmp/l104_bridge/micro/swift_inbox /tmp/l104_bridge/micro/swift_outbox
    mkdir -p "$LOG_DIR"
}

do_start() {
    echo "═══ L104 Service Start ═══"
    ensure_ipc_dirs
    for svc in "${SERVICES[@]}"; do
        plist="$LAUNCH_DIR/$svc.plist"
        if [ ! -f "$plist" ]; then
            echo "  ✗ $svc — plist not found (run 'install' first)"
            continue
        fi
        # Check if already loaded
        if launchctl list "$svc" &>/dev/null; then
            echo "  ● $svc — already running"
        else
            launchctl load -w "$plist"
            echo "  ▶ $svc — started"
        fi
    done
    echo ""
    do_status
}

do_stop() {
    echo "═══ L104 Service Stop ═══"
    for svc in "${SERVICES[@]}"; do
        plist="$LAUNCH_DIR/$svc.plist"
        if launchctl list "$svc" &>/dev/null; then
            launchctl unload "$plist" 2>/dev/null || true
            echo "  ■ $svc — stopped"
        else
            echo "  ○ $svc — not running"
        fi
    done
}

do_restart() {
    do_stop
    sleep 2
    do_start
}

do_status() {
    echo -e "${BOLD}═══ L104 Service Status v3.0 ═══${NC}"
    printf "  %-30s %-10s %-8s %-10s\n" "SERVICE" "STATE" "PID" "NICE"
    echo "  $(printf '─%.0s' {1..60})"
    for svc in "${SERVICES[@]}"; do
        if launchctl list "$svc" &>/dev/null; then
            pid=$(launchctl list "$svc" 2>/dev/null | grep '"PID"' | grep -oE '[0-9]+' || true)
            nice_val="-"
            if [ -n "$pid" ] && [ "$pid" != "-" ]; then
                nice_val=$(ps -o nice= -p "$pid" 2>/dev/null | tr -d ' ' || echo "-")
            fi
            if [ -z "$pid" ]; then
                pid="(idle)"
            fi
            printf "  ${GREEN}●${NC} %-28s %-10s %-8s %-10s\n" "$svc" "RUNNING" "$pid" "$nice_val"
        else
            printf "  ${RED}○${NC} %-28s %-10s %-8s %-10s\n" "$svc" "STOPPED" "-" "-"
        fi
    done
    echo ""

    # Port check
    echo -e "  ${CYAN}Ports:${NC}"
    for port_info in "8081:Fast Server" "8080:Bridge" "10400:P2P" "10401:RPC"; do
        port="${port_info%%:*}"
        label="${port_info#*:}"
        if lsof -iTCP:"$port" -sTCP:LISTEN -P &>/dev/null; then
            echo -e "    ${GREEN}●${NC} $port ($label) — LISTENING"
        else
            echo -e "    ${YELLOW}○${NC} $port ($label) — not bound"
        fi
    done

    # Daemon PID check
    if [ -f "$ROOT/l104_daemon.pid" ]; then
        dpid=$(cat "$ROOT/l104_daemon.pid" 2>/dev/null || true)
        if [ -n "$dpid" ] && kill -0 "$dpid" 2>/dev/null; then
            echo -e "    ${GREEN}●${NC} VQPU Daemon PID $dpid — ALIVE"
        else
            echo -e "    ${YELLOW}!${NC} VQPU Daemon PID file stale"
        fi
    fi

    # Bridge IPC status
    echo ""
    echo -e "  ${CYAN}Bridge IPC:${NC}"
    inbox_count=$(ls /tmp/l104_bridge/inbox/ 2>/dev/null | wc -l | tr -d ' ')
    outbox_count=$(ls /tmp/l104_bridge/outbox/ 2>/dev/null | wc -l | tr -d ' ')
    telemetry_count=$(ls /tmp/l104_bridge/telemetry/ 2>/dev/null | wc -l | tr -d ' ')
    echo "    inbox=$inbox_count  outbox=$outbox_count  telemetry=$telemetry_count"

    # Log sizes
    echo ""
    echo -e "  ${CYAN}Logs:${NC}"
    if [ -d "$LOG_DIR" ]; then
        total_log_kb=$(du -sk "$LOG_DIR" 2>/dev/null | cut -f1)
        echo "    Total: ${total_log_kb}KB in $LOG_DIR"
    fi
}

do_health() {
    echo -e "${BOLD}═══ L104 Deep Health Check v3.0 ═══${NC}"
    echo ""

    # 1. Service state
    local all_ok=true
    echo -e "  ${CYAN}[1/5] Service State${NC}"
    for svc in "${SERVICES[@]}"; do
        if launchctl list "$svc" &>/dev/null; then
            echo -e "    ${GREEN}✓${NC} $svc"
        else
            echo -e "    ${RED}✗${NC} $svc — NOT RUNNING"
            all_ok=false
        fi
    done

    # 2. Port binding
    echo -e "  ${CYAN}[2/5] Port Binding${NC}"
    if lsof -iTCP:8081 -sTCP:LISTEN -P &>/dev/null; then
        echo -e "    ${GREEN}✓${NC} FastAPI port 8081 bound"
    else
        echo -e "    ${RED}✗${NC} FastAPI port 8081 NOT bound"
        all_ok=false
    fi

    # 3. Bridge IPC health
    echo -e "  ${CYAN}[3/5] Bridge IPC${NC}"
    for d in inbox outbox telemetry archive; do
        if [ -d "/tmp/l104_bridge/$d" ]; then
            echo -e "    ${GREEN}✓${NC} /tmp/l104_bridge/$d exists"
        else
            echo -e "    ${RED}✗${NC} /tmp/l104_bridge/$d MISSING"
            all_ok=false
        fi
    done

    # 4. System resources
    echo -e "  ${CYAN}[4/5] System Resources${NC}"
    fd_soft=$(ulimit -n)
    echo "    File descriptors (soft): $fd_soft"
    ram_free=$(vm_stat 2>/dev/null | awk '/Pages free/ {printf "%.0f", $3 * 4096 / 1048576}')
    echo "    Free RAM: ${ram_free}MB"
    swap_used=$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}' || echo "unknown")
    echo "    Swap used: $swap_used"

    # 5. Log health
    echo -e "  ${CYAN}[5/5] Log Health${NC}"
    if [ -d "$LOG_DIR" ]; then
        for logfile in "$LOG_DIR"/*.log; do
            [ -f "$logfile" ] || continue
            log_kb=$(du -k "$logfile" 2>/dev/null | cut -f1)
            name=$(basename "$logfile")
            if [ "$log_kb" -gt 10240 ]; then
                echo -e "    ${YELLOW}!${NC} $name: ${log_kb}KB (>10MB — rotate recommended)"
            else
                echo -e "    ${GREEN}✓${NC} $name: ${log_kb}KB"
            fi
        done
    fi

    echo ""
    if $all_ok; then
        echo -e "  ${GREEN}${BOLD}ALL CHECKS PASSED${NC}"
    else
        echo -e "  ${RED}${BOLD}SOME CHECKS FAILED — review above${NC}"
    fi
    echo ""
}

do_logs() {
    echo "═══ L104 Service Logs (Ctrl-C to stop) ═══"
    tail -f "$LOG_DIR"/fast-server.*.log "$LOG_DIR"/node-server.*.log /tmp/l104_daemon_output.log 2>/dev/null
}

do_install() {
    echo "═══ L104 Service Install ═══"
    ensure_ipc_dirs

    # Validate binaries exist
    if [ ! -f "$ROOT/.venv/bin/python" ]; then
        echo "ERROR: Python venv not found at $ROOT/.venv/bin/python"
        exit 1
    fi
    if [ ! -x "$ROOT/L104SwiftApp/.build/release/L104Daemon" ]; then
        echo "WARNING: L104Daemon binary not found — VQPU service will fail until built"
        echo "  Build with: cd $ROOT/L104SwiftApp && swift build -c release --product L104Daemon"
    fi

    # Plists should already exist from creation, but verify
    for svc in "${SERVICES[@]}"; do
        plist="$LAUNCH_DIR/$svc.plist"
        if [ -f "$plist" ]; then
            echo "  ✓ $plist exists"
        else
            echo "  ✗ $plist missing — copy from repo or recreate"
        fi
    done

    do_start
    echo ""
    echo "Services will now persist across reboots."
    echo "Manage with: $0 {start|stop|restart|status|logs|uninstall}"
}

do_uninstall() {
    echo "═══ L104 Service Uninstall ═══"
    do_stop
    for svc in "${SERVICES[@]}"; do
        plist="$LAUNCH_DIR/$svc.plist"
        if [ -f "$plist" ]; then
            rm -f "$plist"
            echo "  ✗ Removed $plist"
        fi
    done
    echo "Services uninstalled. They will no longer start at login."
}

case "${1:-status}" in
    start)     do_start ;;
    stop)      do_stop ;;
    restart)   do_restart ;;
    status)    do_status ;;
    health)    do_health ;;
    logs)      do_logs ;;
    install)   do_install ;;
    uninstall) do_uninstall ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|health|logs|install|uninstall}"
        exit 1
        ;;
esac
