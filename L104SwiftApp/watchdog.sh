#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  L104 SOVEREIGN INTELLECT — ASI WATCHDOG v1.0
#  Monitors Swift source, Python ASI core, and C bridge for changes.
#  Auto-rebuilds and hot-restarts the Sovereign Bridge on every save.
#  Requires: fswatch (brew install fswatch)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── CONFIGURATION ───
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
APP_NAME="L104Native"
BUILD_SCRIPT="$SCRIPT_DIR/build.sh"
EXECUTABLE="$SCRIPT_DIR/$APP_NAME.app/Contents/MacOS/$APP_NAME"

# ─── WATCH TARGETS ───
# Swift source
SWIFT_SOURCE="$SCRIPT_DIR/Sources/L104Native.swift"
# CPython bridge (C header + implementation)
BRIDGE_HEADER="$SCRIPT_DIR/Sources/cpython_bridge.h"
BRIDGE_SOURCE="$SCRIPT_DIR/Sources/cpython_bridge.c"
# Python ASI core module
PYTHON_ASI_CORE="$WORKSPACE/l104_asi_core.py"
# Kernel parameters (data dependency)
KERNEL_PARAMS="$WORKSPACE/kernel_parameters.json"

# ─── COLORS ───
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ─── BUILD MODE ───
BUILD_MODE="${1:-release}"
AUTO_LAUNCH="${2:-no}"

# ─── STATE ───
BUILD_COUNT=0
FAIL_COUNT=0
START_TIME=$(date +%s)

# ─── BANNER ───
echo -e "${PURPLE}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║   👁️  L104 SOVEREIGN INTELLECT — ASI WATCHDOG v1.0 👁️            ║"
echo "║   📡 Auto-Rebuild · Hot-Restart · CPython Bridge Monitor        ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo -e "║  Mode:      ${CYAN}$BUILD_MODE${PURPLE}                                            ║"
echo -e "║  Launcher:  ${CYAN}$([ "$AUTO_LAUNCH" = "launch" ] && echo "Auto-restart ON" || echo "Build only")${PURPLE}                                      ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── VERIFY TARGETS EXIST ───
echo -e "${CYAN}📋 Verifying watch targets...${NC}"
WATCH_FILES=()
for f in "$SWIFT_SOURCE" "$BRIDGE_HEADER" "$BRIDGE_SOURCE" "$PYTHON_ASI_CORE" "$KERNEL_PARAMS"; do
    if [ -f "$f" ]; then
        WATCH_FILES+=("$f")
        echo -e "${GREEN}  ✓ $(basename "$f")${NC} ${DIM}($(wc -l < "$f" | tr -d ' ') lines)${NC}"
    else
        echo -e "${YELLOW}  ⚠ $(basename "$f") not found — skipping${NC}"
    fi
done

if [ ${#WATCH_FILES[@]} -eq 0 ]; then
    echo -e "${RED}✗ No watch targets found. Exiting.${NC}"
    exit 1
fi

# ─── VERIFY FSWATCH ───
if ! command -v fswatch &> /dev/null; then
    echo -e "${RED}✗ fswatch not found. Install with: brew install fswatch${NC}"
    exit 1
fi

# ─── BUILD FUNCTION ───
do_build() {
    local trigger_file="$1"
    local trigger_name=$(basename "$trigger_file")
    BUILD_COUNT=$((BUILD_COUNT + 1))

    echo ""
    echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}🔄 Change detected: ${BOLD}$trigger_name${NC}"
    echo -e "${BLUE}   Build #$BUILD_COUNT at $(date '+%H:%M:%S')${NC}"
    echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Determine what changed for smarter rebuilds
    local build_flags="--$BUILD_MODE --force"

    # Run build
    local build_start=$(date +%s)
    if bash "$BUILD_SCRIPT" $build_flags 2>&1; then
        local build_end=$(date +%s)
        local elapsed=$((build_end - build_start))

        echo -e "${GREEN}${BOLD}✅ Build #$BUILD_COUNT succeeded${NC} ${DIM}(${elapsed}s)${NC}"

        # Auto-launch if configured
        if [ "$AUTO_LAUNCH" = "launch" ]; then
            echo -e "${CYAN}🚀 Restarting $APP_NAME...${NC}"
            pkill -f "$APP_NAME" 2>/dev/null || true
            sleep 0.5
            open "$SCRIPT_DIR/$APP_NAME.app" &
            echo -e "${GREEN}  ✓ $APP_NAME launched${NC}"
        fi

        # Verify Python bridge linkage
        if otool -L "$EXECUTABLE" 2>/dev/null | grep -q python; then
            local py_ver=$(otool -L "$EXECUTABLE" | grep python | grep -o 'python[0-9.]*' | head -1)
            echo -e "${GREEN}  🐍 CPython bridge: ${BOLD}$py_ver linked${NC}"
        fi

    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo -e "${RED}${BOLD}❌ Build #$BUILD_COUNT FAILED${NC} ${DIM}(fail count: $FAIL_COUNT)${NC}"
        echo -e "${YELLOW}   Fix the errors to resume auto-sync.${NC}"

        # Show last few errors from build log
        local log="$SCRIPT_DIR/.build/build.log"
        if [ -f "$log" ]; then
            echo -e "${RED}   Recent errors:${NC}"
            grep -E "error:" "$log" | tail -5 | while IFS= read -r line; do
                echo -e "${RED}     $line${NC}"
            done
        fi
    fi

    # Status line
    local uptime=$(( $(date +%s) - START_TIME ))
    local mins=$((uptime / 60))
    echo -e "${DIM}📊 Builds: $BUILD_COUNT | Fails: $FAIL_COUNT | Uptime: ${mins}m | Watching ${#WATCH_FILES[@]} files${NC}"
    echo -e "${PURPLE}👁️  Watching for changes...${NC}"
}

# ─── SIGNAL HANDLING ───
cleanup() {
    echo ""
    echo -e "${YELLOW}${BOLD}👁️  Watchdog shutting down...${NC}"
    local uptime=$(( $(date +%s) - START_TIME ))
    local mins=$((uptime / 60))
    echo -e "${DIM}   Session: $BUILD_COUNT builds, $FAIL_COUNT failures, ${mins}m uptime${NC}"
    if [ "$AUTO_LAUNCH" = "launch" ]; then
        pkill -f "$APP_NAME" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

# ─── MAIN WATCH LOOP ───
echo ""
echo -e "${PURPLE}${BOLD}👁️  Watchdog active — monitoring ${#WATCH_FILES[@]} files...${NC}"
echo -e "${DIM}   Press Ctrl+C to stop${NC}"
echo ""

# fswatch with latency to debounce rapid saves, one-event mode for cleaner output
fswatch --latency 2 -o "${WATCH_FILES[@]}" | while read -r _num; do
    # Identify which file triggered (best effort — fswatch -o just gives count)
    # We check modification times to guess
    latest_file=""
    latest_mtime=0
    for f in "${WATCH_FILES[@]}"; do
        mtime=$(stat -f %m "$f" 2>/dev/null || echo 0)
        if [ "$mtime" -gt "$latest_mtime" ]; then
            latest_mtime=$mtime
            latest_file=$f
        fi
    done

    do_build "${latest_file:-${WATCH_FILES[0]}}"
done
