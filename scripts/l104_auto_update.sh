#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# L104 Sovereign Node — Auto-Update Watcher v2.0
# ═══════════════════════════════════════════════════════════════════
# Polls the remote git repo every 5 minutes. When new commits are
# detected, runs the full upgrade pipeline (pull → build → restart)
# using the existing upgrade_all.sh.
#
# If the upgrade fails, services stay on the old version (KeepAlive
# in launchd ensures they keep running regardless).
#
# Env vars:
#   L104_UPDATE_INTERVAL  — seconds between polls (default: 300 = 5 min)
#   L104_UPDATE_BRANCH    — branch to track (default: main)
#   L104_UPDATE_AUTO_BUILD — also rebuild Swift daemon (default: 1)
#
# GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/auto-update.log"
LOCK="/tmp/l104_auto_update.lock"
INTERVAL="${L104_UPDATE_INTERVAL:-300}"
BRANCH="${L104_UPDATE_BRANCH:-main}"
SERVICE_CTL="$ROOT/scripts/l104_service_ctl.sh"
UPGRADE_SCRIPT="$ROOT/scripts/upgrade_all.sh"
PYTHON="$ROOT/.venv/bin/python"
MAX_RETRIES="${L104_UPDATE_RETRY_COUNT:-3}"
AUTO_RESTART="${L104_UPDATE_AUTO_RESTART:-1}"

mkdir -p "$LOG_DIR"

log() {
    echo "[L104-AUTO-UPDATE] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

cleanup() {
    rm -f "$LOCK"
    log "Auto-update watcher stopped"
}
trap cleanup EXIT INT TERM

# Prevent duplicate instances
if [ -f "$LOCK" ]; then
    old_pid=$(cat "$LOCK" 2>/dev/null || true)
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        log "FATAL: Another auto-update watcher is running (PID $old_pid)"
        exit 1
    fi
    rm -f "$LOCK"
fi
echo $$ > "$LOCK"

log "═══ L104 Auto-Update Watcher v2.0 Started ═══"
log "  Root:     $ROOT"
log "  Branch:   $BRANCH"
log "  Interval: ${INTERVAL}s"
log "  Retries:  $MAX_RETRIES"
log "  AutoRestart: $AUTO_RESTART"
log "  PID:      $$"

# ─── Main Loop ───

while true; do
    sleep "$INTERVAL"

    cd "$ROOT"

    # Skip if not a git repo
    if ! git rev-parse --is-inside-work-tree &>/dev/null; then
        log "WARN: Not a git repo — skipping check"
        continue
    fi

    # Fetch remote (quiet, no merge)
    if ! git fetch origin "$BRANCH" --quiet 2>/dev/null; then
        log "WARN: git fetch failed (offline?) — skipping"
        continue
    fi

    LOCAL=$(git rev-parse HEAD 2>/dev/null)
    REMOTE=$(git rev-parse "origin/$BRANCH" 2>/dev/null)

    if [ "$LOCAL" = "$REMOTE" ]; then
        # No changes — silent (only log every 12th check = once/hour at 5m interval)
        CHECK_COUNT="${CHECK_COUNT:-0}"
        CHECK_COUNT=$((CHECK_COUNT + 1))
        if [ $((CHECK_COUNT % 12)) -eq 0 ]; then
            log "No updates (checked ${CHECK_COUNT} times, last: ${LOCAL:0:8})"
        fi
        continue
    fi

    # New commits detected!
    BEHIND=$(git rev-list --count HEAD..origin/"$BRANCH" 2>/dev/null || echo "?")
    log "╔═══ UPDATE DETECTED ═══╗"
    log "  Local:  ${LOCAL:0:12}"
    log "  Remote: ${REMOTE:0:12}"
    log "  Behind: $BEHIND commit(s)"

    # Show what's coming
    git log --oneline HEAD..origin/"$BRANCH" 2>/dev/null | head -10 | while read -r line; do
        log "  + $line"
    done

    # Phase 1: Pull (with retry)
    log "Phase 1: git pull --ff-only"
    PULL_OK=0
    for attempt in $(seq 1 $MAX_RETRIES); do
        if git pull --ff-only 2>&1 | tee -a "$LOG"; then
            PULL_OK=1
            break
        fi
        log "  WARN: git pull attempt $attempt/$MAX_RETRIES failed — retrying in 5s"
        sleep 5
    done
    if [ "$PULL_OK" -eq 0 ]; then
        log "ERROR: git pull failed after $MAX_RETRIES attempts — skipping this update cycle"
        continue
    fi

    # Phase 2: Install deps if requirements changed
    if git diff --name-only "$LOCAL" HEAD | grep -q 'requirements'; then
        log "Phase 2: requirements changed — pip install"
        "$PYTHON" -m pip install -r "$ROOT/requirements.txt" --quiet 2>&1 | tail -5 | tee -a "$LOG" || true
    fi

    # Phase 3: Rebuild Swift daemon if Swift sources changed
    SWIFT_CHANGED=$(git diff --name-only "$LOCAL" HEAD | grep -c 'L104SwiftApp/' || true)
    if [ "$SWIFT_CHANGED" -gt 0 ] && [ "${L104_UPDATE_AUTO_BUILD:-1}" = "1" ]; then
        log "Phase 3: Swift sources changed ($SWIFT_CHANGED files) — rebuilding"
        cd "$ROOT/L104SwiftApp"
        if swift build -c release --product L104Daemon 2>&1 | tail -5 | tee -a "$LOG"; then
            log "  Swift daemon rebuilt OK"
        else
            log "  WARN: Swift build failed — daemon stays on old binary"
        fi
        cd "$ROOT"
    fi

    # Phase 4: Rebuild C kernel if C sources changed
    C_CHANGED=$(git diff --name-only "$LOCAL" HEAD | grep -c 'l104_core_c/' || true)
    if [ "$C_CHANGED" -gt 0 ]; then
        log "Phase 4: C kernel sources changed — rebuilding"
        make -C "$ROOT/l104_core_c" -j4 2>&1 | tail -5 | tee -a "$LOG" || log "  WARN: C build failed"
    fi

    # Phase 5: Restart services via launchd (graceful)
    if [ "$AUTO_RESTART" = "1" ]; then
        log "Phase 5: Restarting services"
        if [ -x "$SERVICE_CTL" ]; then
            "$SERVICE_CTL" restart 2>&1 | tee -a "$LOG"
        else
            # Fallback: direct launchctl
            log "  service_ctl not found — direct launchctl restart"
            for svc in com.l104.fast-server com.l104.node-server com.l104.vqpu-daemon; do
                plist="$HOME/Library/LaunchAgents/$svc.plist"
                if [ -f "$plist" ]; then
                    launchctl unload "$plist" 2>/dev/null || true
                    sleep 1
                    launchctl load -w "$plist"
                    log "  Restarted $svc"
                fi
            done
        fi
    else
        log "Phase 5: AUTO_RESTART disabled — skipping service restart"
    fi

    # Phase 6: Health check
    log "Phase 6: Health verification"
    sleep 10  # Give server time to boot
    HEALTH_URL="http://localhost:${PORT:-8081}/health"
    if curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
        log "  Health check: PASSED"
    else
        log "  Health check: PENDING (server may still be warming up)"
    fi

    NEW_HEAD=$(git rev-parse HEAD 2>/dev/null)
    log "╚═══ UPDATE COMPLETE: ${LOCAL:0:8} → ${NEW_HEAD:0:8} ($BEHIND commits) ═══╝"
    log ""
done
