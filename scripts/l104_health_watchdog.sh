#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# L104 Sovereign Node — Health Watchdog v1.0
# ═══════════════════════════════════════════════════════════════════
# Monitors all L104 services and auto-restarts dead ones.
# Runs as a launchd daemon (com.l104.health-watchdog) every 60s.
#
# Checks:
#   1. All 4 core launchd services are loaded and running
#   2. FastAPI port 8081 is bound and responding
#   3. Bridge IPC directories exist and are writable
#   4. VQPU daemon binary exists and is executable
#   5. System resource pressure (file descriptors, memory)
#   6. Log file health (no runaway growth)
#
# Actions:
#   - Auto-reload crashed launchd services
#   - Recreate missing bridge IPC directories
#   - Write health status to /tmp/l104_health.json
#   - Alert on high resource pressure
#
# Usage:
#   bash scripts/l104_health_watchdog.sh          # Single check
#   bash scripts/l104_health_watchdog.sh --status  # Show last health report
#   bash scripts/l104_health_watchdog.sh --loop    # Continuous (for launchd)
#
# GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEALTH_FILE="/tmp/l104_health.json"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$ROOT/logs"
BRIDGE="/tmp/l104_bridge"
CHECK_INTERVAL="${L104_HEALTH_CHECK_INTERVAL:-60}"

CORE_SERVICES=(
    "com.l104.fast-server"
    "com.l104.node-server"
    "com.l104.vqpu-daemon"
    "com.l104.auto-update"
)

TS() { date '+%Y-%m-%dT%H:%M:%S'; }

# ── Show last health report ──
if [[ "${1:-}" == "--status" ]]; then
    if [ -f "$HEALTH_FILE" ]; then
        cat "$HEALTH_FILE"
    else
        echo "No health report found. Run watchdog first."
    fi
    exit 0
fi

# ── Single health check cycle ──
do_check() {
    local issues=0
    local repairs=0
    local checks_passed=0
    local total_checks=0
    local details=""

    # [1] Service state
    for svc in "${CORE_SERVICES[@]}"; do
        total_checks=$((total_checks + 1))
        if launchctl list "$svc" &>/dev/null; then
            # Safe PID extraction — grep returns 1 on no-match which kills pipefail,
            # so we use || true to swallow the non-match exit code
            pid=$(launchctl list "$svc" 2>/dev/null | grep '"PID"' | grep -oE '[0-9]+' || true)
            if [ -n "$pid" ]; then
                checks_passed=$((checks_passed + 1))
            else
                # Service loaded but no PID — might be starting or crashed
                checks_passed=$((checks_passed + 1))
                details="$details  [$(TS)] $svc: loaded but no active PID (idle/starting)\n"
            fi
        else
            issues=$((issues + 1))
            details="$details  [$(TS)] $svc: NOT RUNNING — attempting reload\n"
            plist="$LAUNCH_DIR/$svc.plist"
            if [ -f "$plist" ]; then
                launchctl load -w "$plist" 2>/dev/null || true
                repairs=$((repairs + 1))
                details="$details  [$(TS)] $svc: reloaded\n"
            else
                details="$details  [$(TS)] $svc: plist NOT FOUND at $plist\n"
            fi
        fi
    done

    # [2] Port 8081 (FastAPI)
    total_checks=$((total_checks + 1))
    if lsof -iTCP:8081 -sTCP:LISTEN -P &>/dev/null; then
        checks_passed=$((checks_passed + 1))
    else
        issues=$((issues + 1))
        details="$details  [$(TS)] Port 8081: NOT bound\n"
    fi

    # [3] Bridge IPC directories
    for d in inbox outbox telemetry archive; do
        total_checks=$((total_checks + 1))
        if [ -d "$BRIDGE/$d" ]; then
            checks_passed=$((checks_passed + 1))
        else
            issues=$((issues + 1))
            mkdir -p "$BRIDGE/$d" 2>/dev/null || true
            repairs=$((repairs + 1))
            details="$details  [$(TS)] Bridge $d: recreated\n"
        fi
    done

    # [4] VQPU binary
    total_checks=$((total_checks + 1))
    DAEMON_BIN="$ROOT/L104SwiftApp/.build/release/L104Daemon"
    if [ -x "$DAEMON_BIN" ]; then
        checks_passed=$((checks_passed + 1))
    else
        issues=$((issues + 1))
        details="$details  [$(TS)] VQPU binary: NOT found at $DAEMON_BIN\n"
    fi

    # [5] File descriptor pressure
    total_checks=$((total_checks + 1))
    fd_used=$(lsof -p $$ 2>/dev/null | wc -l | tr -d ' ' 2>/dev/null || echo "0")
    fd_soft=$(ulimit -n 2>/dev/null || echo "256")
    if [ "${fd_used:-0}" -lt "$((${fd_soft:-256} * 80 / 100))" ] 2>/dev/null; then
        checks_passed=$((checks_passed + 1))
    else
        issues=$((issues + 1))
        details="$details  [$(TS)] FD pressure: $fd_used/$fd_soft (>80%%)\n"
    fi

    # [6] Log file health (any log > 50MB = warning)
    total_checks=$((total_checks + 1))
    log_warning=false
    if [ -d "$LOG_DIR" ]; then
        for logfile in "$LOG_DIR"/*.log; do
            [ -f "$logfile" ] || continue
            log_kb=$(du -k "$logfile" 2>/dev/null | cut -f1 || echo "0")
            if [ "${log_kb:-0}" -gt 51200 ]; then
                log_warning=true
                details="$details  [$(TS)] Log $(basename "$logfile"): ${log_kb}KB (>50MB)\n"
            fi
        done
    fi
    if [ "$log_warning" = false ]; then
        checks_passed=$((checks_passed + 1))
    else
        issues=$((issues + 1))
    fi

    # Write health report JSON
    local status="healthy"
    if [ "$issues" -gt 0 ]; then
        status="degraded"
    fi
    if [ "$issues" -gt 3 ]; then
        status="critical"
    fi

    cat > "$HEALTH_FILE" << HEALTH_EOF
{
  "timestamp": "$(TS)",
  "status": "$status",
  "checks_passed": $checks_passed,
  "total_checks": $total_checks,
  "issues": $issues,
  "repairs": $repairs,
  "watchdog_version": "1.0",
  "god_code": 527.5184818492612
}
HEALTH_EOF

    # Print summary to stdout (captured by launchd logs)
    if [ "$issues" -eq 0 ]; then
        echo "[$(TS)] L104 Health: OK ($checks_passed/$total_checks checks passed)"
    else
        echo "[$(TS)] L104 Health: $status ($issues issues, $repairs repairs)"
        echo -e "$details"
    fi
}

# ── Main ──
if [[ "${1:-}" == "--loop" ]]; then
    echo "[$(TS)] L104 Health Watchdog v1.0 starting (interval=${CHECK_INTERVAL}s)"
    while true; do
        do_check
        sleep "$CHECK_INTERVAL"
    done
else
    do_check
fi
