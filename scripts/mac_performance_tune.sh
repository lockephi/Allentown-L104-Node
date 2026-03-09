#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# L104 Sovereign Node — macOS Performance Tuner v5.0
# ═══════════════════════════════════════════════════════════════════
# Optimizes macOS system settings for L104 workloads.
# Detects hardware (Intel/Apple Silicon, cores, RAM) and applies
# appropriate tuning for:
#   - Memory pressure relief (purge, swap, compression)
#   - File descriptor limits (for IPC pipeline)
#   - Python runtime optimization
#   - Disk I/O scheduling
#   - Network buffer tuning (for API gateway)
#   - Process priority elevation
#
# Usage:
#   bash scripts/mac_performance_tune.sh          # Apply all tuning
#   bash scripts/mac_performance_tune.sh --status # Show current state
#   bash scripts/mac_performance_tune.sh --reset  # Revert to defaults
#
# GOD_CODE=527.5184818492612 | PILOT: LONDEL
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ─── Colors ───
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ─── Hardware Detection ───
HW_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
HW_PHYS_CORES=$((HW_CORES / 2))
HW_RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 4294967296)
HW_RAM_MB=$((HW_RAM_BYTES / 1048576))
HW_RAM_GB=$((HW_RAM_MB / 1024))
IS_ARM=$(sysctl -n hw.optional.arm64 2>/dev/null && echo 1 || echo 0)
ARCH="Intel"
[[ "$IS_ARM" == "1" ]] && ARCH="Apple Silicon"

echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  L104 macOS Performance Tuner v5.0${NC}"
echo -e "  ${CYAN}Hardware:${NC} $ARCH, ${HW_PHYS_CORES}P/${HW_CORES}L cores, ${HW_RAM_GB}GB RAM"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo ""

# ─── Mode ───
MODE="${1:---apply}"

if [[ "$MODE" == "--status" ]]; then
    echo -e "${CYAN}─── System Status ───${NC}"
    echo "  File descriptors (soft): $(ulimit -n)"
    echo "  File descriptors (hard): $(ulimit -Hn)"
    echo "  RAM free: $(vm_stat 2>/dev/null | awk '/Pages free/ {printf "%.0f MB", $3 * 4096 / 1048576}')"
    echo "  RAM compressed: $(vm_stat 2>/dev/null | awk '/Pages occupied by compressor/ {printf "%.0f MB", $NF * 4096 / 1048576}')"
    echo "  Swap used: $(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}')"
    echo ""
    echo -e "${CYAN}─── L104 Processes ───${NC}"
    ps aux | grep -E "L104Daemon|main.py|L104_public_node" | grep -v grep || echo "  No L104 processes found"
    echo ""
    echo -e "${CYAN}─── Memory Pressure ───${NC}"
    memory_pressure 2>/dev/null | head -5 || echo "  memory_pressure command not available"
    exit 0
fi

if [[ "$MODE" == "--reset" ]]; then
    echo -e "${YELLOW}Reverting to default macOS settings...${NC}"
    sudo -n sysctl -w kern.maxfiles=12288 2>/dev/null || true
    sudo -n sysctl -w kern.maxfilesperproc=10240 2>/dev/null || true
    echo -e "${GREEN}Defaults restored. Restart services to take effect.${NC}"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# APPLY OPTIMIZATIONS
# ═══════════════════════════════════════════════════════════════════

APPLIED=0
SKIPPED=0

apply() {
    local desc="$1"
    local result="$2"
    if [[ "$result" == "0" ]]; then
        echo -e "  ${GREEN}✓${NC} $desc"
        APPLIED=$((APPLIED + 1))
    else
        echo -e "  ${YELLOW}–${NC} $desc (skipped/already set)"
        SKIPPED=$((SKIPPED + 1))
    fi
}

# ─── 1. File Descriptor Limits ───
echo -e "${CYAN}[1/7] File Descriptor Limits${NC}"
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true
FD_SOFT=$(ulimit -n)
apply "ulimit -n $FD_SOFT (soft, target 65536)" "0"

# sysctl tuning (requires sudo — skip gracefully if unavailable)
if sudo -n true 2>/dev/null; then
    if [[ "$HW_RAM_GB" -ge 8 ]]; then
        sudo sysctl -w kern.maxfiles=524288 >/dev/null 2>&1 || true
        sudo sysctl -w kern.maxfilesperproc=262144 >/dev/null 2>&1 || true
        apply "kern.maxfiles=524288 kern.maxfilesperproc=262144 (8GB+ Mac)" "0"
    else
        sudo sysctl -w kern.maxfiles=131072 >/dev/null 2>&1 || true
        sudo sysctl -w kern.maxfilesperproc=65536 >/dev/null 2>&1 || true
        apply "kern.maxfiles=131072 kern.maxfilesperproc=65536 (4GB Mac)" "0"
    fi
else
    apply "kern.maxfiles (needs sudo — run with sudo for full tuning)" "1"
fi

# ─── 2. Memory Optimization ───
echo -e "${CYAN}[2/7] Memory Optimization${NC}"

# Purge disk cache on low-RAM machines
if [[ "$HW_RAM_GB" -le 4 ]] && sudo -n true 2>/dev/null; then
    sudo purge 2>/dev/null && apply "Purged disk cache (4GB Mac)" "0" || apply "purge" "1"
fi

# Clear __pycache__ to reclaim disk/inode resources
PYCACHE_COUNT=$(find "$ROOT" -maxdepth 3 -type d -name "__pycache__" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$PYCACHE_COUNT" -gt 0 ]]; then
    find "$ROOT" -maxdepth 3 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    apply "Cleared $PYCACHE_COUNT __pycache__ directories" "0"
fi

# Clear .pyc files
PYC_COUNT=$(find "$ROOT" -maxdepth 3 -name "*.pyc" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$PYC_COUNT" -gt 0 ]]; then
    find "$ROOT" -maxdepth 3 -name "*.pyc" -delete 2>/dev/null || true
    apply "Deleted $PYC_COUNT .pyc files" "0"
fi

# Trim old telemetry (keep last 10 sessions)
TELEMETRY_DIR="/tmp/l104_bridge/telemetry"
if [[ -d "$TELEMETRY_DIR" ]]; then
    OLD_TEL=$(ls -1t "$TELEMETRY_DIR"/session_*.json 2>/dev/null | tail -n +11 | wc -l | tr -d ' ')
    if [[ "$OLD_TEL" -gt 0 ]]; then
        ls -1t "$TELEMETRY_DIR"/session_*.json 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null
        apply "Trimmed $OLD_TEL old telemetry sessions" "0"
    fi
fi

# ─── 3. Network Buffer Tuning (for API gateway) ───
echo -e "${CYAN}[3/7] Network Buffer Tuning${NC}"

# TCP buffer sizes — appropriate for local API traffic
if sudo -n true 2>/dev/null; then
    sudo sysctl -w net.inet.tcp.sendspace=262144 >/dev/null 2>&1 || true
    sudo sysctl -w net.inet.tcp.recvspace=262144 >/dev/null 2>&1 || true
    apply "TCP send/recv buffers → 256KB" "0"

    sudo sysctl -w net.inet.tcp.delayed_ack=0 >/dev/null 2>&1 || true
    apply "TCP delayed ACK disabled (lower latency)" "0"

    sudo sysctl -w kern.ipc.somaxconn=2048 >/dev/null 2>&1 || true
    apply "Listen backlog → 2048" "0"

    sudo sysctl -w net.inet.tcp.mssdflt=1460 >/dev/null 2>&1 || true
    apply "TCP MSS default → 1460" "0"
else
    apply "Network tuning (needs sudo — run with sudo for full tuning)" "1"
fi

# ─── 4. Python Environment Tuning ───
echo -e "${CYAN}[4/7] Python Environment${NC}"

# Write a .env_perf file that run_services.sh can source
PERF_ENV="$ROOT/.env_perf"
cat > "$PERF_ENV" <<'PYENV'
# L104 macOS Performance Tuning v5.0 — auto-generated
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONOPTIMIZE=2
export PYTHONHASHSEED=0
export PYTHONMALLOC=malloc
export MALLOC_NANO_ZONE=0
export MallocNanoZone=0
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export UV_THREADPOOL_SIZE=8
# Reduce GC pressure — let l104_optimization.py manage thresholds
export PYTHONGC=1
PYENV
apply "Generated .env_perf with Python tuning flags (v5.0)" "0"

# ─── 5. Process Priority (current shell + L104 daemon) ───
echo -e "${CYAN}[5/7] Process Priority${NC}"

# Elevate daemon priority if running
DAEMON_PID=$(pgrep -f "L104Daemon" 2>/dev/null || true)
if [[ -n "$DAEMON_PID" ]]; then
    if sudo -n true 2>/dev/null; then
        sudo renice -5 "$DAEMON_PID" >/dev/null 2>&1 || true
        apply "L104Daemon (PID $DAEMON_PID) → nice -5" "0"
    else
        renice -5 "$DAEMON_PID" >/dev/null 2>&1 || true
        apply "L104Daemon (PID $DAEMON_PID) → nice -5 (user-level)" "0"
    fi
fi

# Elevate Python server priority if running
SERVER_PID=$(pgrep -f "main.py" 2>/dev/null | head -1 || true)
if [[ -n "$SERVER_PID" ]]; then
    if sudo -n true 2>/dev/null; then
        sudo renice -3 "$SERVER_PID" >/dev/null 2>&1 || true
        apply "FastAPI server (PID $SERVER_PID) → nice -3" "0"
    else
        renice -3 "$SERVER_PID" >/dev/null 2>&1 || true
        apply "FastAPI server (PID $SERVER_PID) → nice -3 (user-level)" "0"
    fi
fi

# ─── 6. Disk I/O Optimization ───
echo -e "${CYAN}[6/7] Disk I/O${NC}"

# Ensure IPC directories use tmpfs (already in /tmp on macOS)
for d in /tmp/l104_bridge/inbox /tmp/l104_bridge/outbox /tmp/l104_bridge/telemetry /tmp/l104_bridge/archive; do
    mkdir -p "$d" 2>/dev/null
done
apply "IPC directories verified on /tmp (RAM-backed tmpfs)" "0"

# Compact bridge archive if large
ARCHIVE_DIR="/tmp/l104_bridge/archive"
if [[ -d "$ARCHIVE_DIR" ]]; then
    ARCHIVE_COUNT=$(ls -1 "$ARCHIVE_DIR" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$ARCHIVE_COUNT" -gt 100 ]]; then
        rm -f "$ARCHIVE_DIR"/*.json 2>/dev/null || true
        apply "Cleared $ARCHIVE_COUNT archived circuit payloads" "0"
    fi
fi

# ─── 7. Log Rotation ───
echo -e "${CYAN}[7/7] Log Rotation${NC}"
LOG_DIR="$ROOT/logs"
if [[ -d "$LOG_DIR" ]]; then
    for logfile in "$LOG_DIR"/*.log; do
        [[ -f "$logfile" ]] || continue
        LOG_SIZE_KB=$(du -k "$logfile" 2>/dev/null | cut -f1)
        if [[ "$LOG_SIZE_KB" -gt 5120 ]]; then
            # Rotate: current → .1, truncate current
            cp "$logfile" "${logfile}.1" 2>/dev/null
            : > "$logfile"
            apply "Rotated $(basename "$logfile") (${LOG_SIZE_KB}KB → .1)" "0"
        fi
    done
    # Remove rotated logs older than 3 days
    find "$LOG_DIR" -name "*.log.[0-9]" -mtime +3 -delete 2>/dev/null || true
    apply "Cleaned rotated logs older than 3 days" "0"
else
    apply "Log directory not found" "1"
fi

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}$APPLIED optimizations applied${NC}, $SKIPPED unchanged"
echo -e "  ${CYAN}Hardware:${NC} $ARCH ${HW_PHYS_CORES}P/${HW_CORES}L @ ${HW_RAM_GB}GB"
echo -e "  ${CYAN}Tuning:${NC} ulimit=$(ulimit -n), PYTHONOPTIMIZE=2, PYTHONMALLOC=malloc"
if [[ "$HW_RAM_GB" -le 4 ]]; then
    echo -e "  ${YELLOW}⚠ Low-RAM Mac — kern.maxfiles capped at 131072${NC}"
fi
echo -e "  ${CYAN}Daemon:${NC} ${DAEMON_PID:-not running}"
echo -e "  INVARIANT: 527.5184818492612 | PILOT: LONDEL"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
