#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# L104 Sovereign Node — Log Rotation v1.0
# ═══════════════════════════════════════════════════════════════════
# Rotates daemon logs to prevent unbounded growth.
# Keeps 3 rotated copies per log file. Max 5MB per log before rotation.
#
# Usage:
#   bash scripts/l104_log_rotate.sh          # Rotate if needed
#   bash scripts/l104_log_rotate.sh --force  # Force rotate all
#   bash scripts/l104_log_rotate.sh --status # Show log sizes
#
# Can be called from cron or launchd on a schedule:
#   */30 * * * * /path/to/scripts/l104_log_rotate.sh
#
# GOD_CODE=527.5184818492612 | PILOT: LONDEL
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
MAX_SIZE_KB="${L104_LOG_MAX_KB:-5120}"       # 5MB default
KEEP_ROTATED="${L104_LOG_KEEP:-3}"           # Keep 3 rotated copies
MODE="${1:---auto}"

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log() { echo -e "${CYAN}[L104-LOG]${NC} $(date '+%H:%M:%S') $*"; }

# ─── Status ───
if [[ "$MODE" == "--status" ]]; then
    echo -e "${CYAN}═══ L104 Log Status ═══${NC}"
    if [[ -d "$LOG_DIR" ]]; then
        printf "%-40s %10s %s\n" "FILE" "SIZE" "MODIFIED"
        printf "%-40s %10s %s\n" "────" "────" "────────"
        for f in "$LOG_DIR"/*.log "$LOG_DIR"/*.log.*; do
            [[ -f "$f" ]] || continue
            SIZE=$(du -h "$f" 2>/dev/null | cut -f1)
            MOD=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$f" 2>/dev/null || stat --format="%y" "$f" 2>/dev/null | cut -d. -f1)
            printf "%-40s %10s %s\n" "$(basename "$f")" "$SIZE" "$MOD"
        done
        echo ""
        TOTAL=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
        echo -e "  Total: ${GREEN}$TOTAL${NC}"
    else
        echo "  No logs directory found"
    fi
    exit 0
fi

# ─── Rotate Function ───
rotate_log() {
    local logfile="$1"
    local basename_log
    basename_log=$(basename "$logfile")

    # Shift existing rotated files: .3→delete, .2→.3, .1→.2, current→.1
    for i in $(seq $((KEEP_ROTATED - 1)) -1 1); do
        local next=$((i + 1))
        if [[ -f "${logfile}.${i}" ]]; then
            if [[ "$next" -gt "$KEEP_ROTATED" ]]; then
                rm -f "${logfile}.${i}"
            else
                mv "${logfile}.${i}" "${logfile}.${next}" 2>/dev/null || true
            fi
        fi
    done

    # Copy current → .1 and truncate
    if [[ -f "$logfile" ]] && [[ -s "$logfile" ]]; then
        cp "$logfile" "${logfile}.1" 2>/dev/null || true
        : > "$logfile"
        log "${GREEN}✓${NC} Rotated $basename_log → ${basename_log}.1"
        return 0
    fi
    return 1
}

# ─── Main ───
if [[ ! -d "$LOG_DIR" ]]; then
    log "No logs directory at $LOG_DIR"
    exit 0
fi

ROTATED=0
for logfile in "$LOG_DIR"/*.log; do
    [[ -f "$logfile" ]] || continue
    SIZE_KB=$(du -k "$logfile" 2>/dev/null | cut -f1)

    if [[ "$MODE" == "--force" ]] || [[ "$SIZE_KB" -gt "$MAX_SIZE_KB" ]]; then
        if rotate_log "$logfile"; then
            ROTATED=$((ROTATED + 1))
        fi
    fi
done

# Clean up old rotated logs (older than 7 days)
CLEANED=$(find "$LOG_DIR" -name "*.log.[0-9]" -mtime +7 -delete -print 2>/dev/null | wc -l | tr -d ' ')

if [[ "$ROTATED" -gt 0 ]] || [[ "$CLEANED" -gt 0 ]]; then
    log "Rotated: $ROTATED files, Cleaned: $CLEANED old copies"
elif [[ "$MODE" != "--auto" ]]; then
    log "No rotation needed (all logs < ${MAX_SIZE_KB}KB)"
fi
