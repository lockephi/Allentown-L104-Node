#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  L104 SOVEREIGN INTELLECT — QUICK BUILD v4.0
#  Dual build system: SPM or Xcode build → .app bundle wrapping → code sign → optional launch
#
#  Pipeline (SPM mode):
#    1. Pre-flight checks (Swift toolchain, Package.swift, disk space)
#    2. SPM incremental compile per target (L104, L104Daemon, L104NanoDaemon)
#    3. Binary change detection + install into .app bundle
#    4. Resource bundling (icon, JSON configs)
#    5. Info.plist generation + PkgInfo stamp
#    6. Ad-hoc code signing (--deep, skippable)
#    7. Post-build verification (architecture, bundle integrity)
#    8. Build log + history tracking
#    9. Optional hot-restart launch
#
#  Pipeline (Xcode mode):
#    1. Pre-flight checks (Xcode, xcodeproj, disk space)
#    2. Generate xcodeproj if needed (swift package generate-xcodeproj)
#    3. xcodebuild compile per scheme/target
#    4. Binary change detection + install into .app bundle
#    5. Resource bundling (icon, JSON configs)
#    6. Info.plist generation + PkgInfo stamp
#    7. Code signing with full identity support
#    8. Post-build verification (architecture, bundle integrity)
#    9. Build log + history tracking
#    10. Optional hot-restart launch
#
#  v4.0 CHANGES (from v3.0):
#    - Added Xcode build mode (-x/--xcode flag)
#    - Added scheme support for xcodebuild
#    - Added CODE_SIGNING_IDENTITY, CODE_SIGNING_REQUIRED options
#    - Added -xcconfig file support for custom build settings
#    - Added -derivedDataPath option for custom derived data
#    - Added -destination specifier for building specific platforms
#    - Added -scheme flag to specify which scheme to build
#    - Auto-generates xcodeproj when needed for xcode mode
#    - Full Xcode build system integration with detailed logging
#    - Better error handling with context extraction
#    - Build settings inheritance from xcconfig files
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
APP_NAME="L104Native"
BUNDLE_ID="com.allentown.l104"
VERSION="24.3"  # EVO_71: B74_QuantumPrimitives + ChatBatchResponseEngine
MIN_MACOS="12.0"
BUILD_NUMBER=$(date +%Y%m%d%H%M)
REQUIRED_SWIFT_MAJOR=5

# Xcode build configuration
USE_XCODE=false
XCODE_SCHEME=""
XCODE_PROJECT=""
XCODE_WORKSPACE=""
XCODE_DERIVED_DATA_PATH=""
XCODE_DESTINATION="platform=macOS"
XCODE_CONFIG_FILE=""
CODE_SIGNING_IDENTITY="-"  # "-" for ad-hoc, or specify identity like "Apple Development"
CODE_SIGNING_REQUIRED="NO"
CODE_SIGNING_ALLOWED="YES"
XCODE_SDK="macosx"

# ─── PATHS ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/Sources"
APP_BUNDLE="$SCRIPT_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
SPM_BUILD_DIR="$SCRIPT_DIR/.build"
BUILD_LOG="$SPM_BUILD_DIR/quick_build.log"
HISTORY_FILE="$SPM_BUILD_DIR/.build_history"

# IPC directories used by daemons
IPC_DIRS=(
    "/tmp/l104_queue"
    "/tmp/l104_bridge"
    "/tmp/l104_bridge/nano/swift_outbox"
)

# Database files that need cleanup before rebuilds (stale WAL/locks cause concurrent errors)
DB_FILES=(
    "$SCRIPT_DIR/l104_asi_memory.db"
    "$SCRIPT_DIR/l104_asi_nexus.db"
    "$SCRIPT_DIR/l104_research.db"
    "$SCRIPT_DIR/memory_optimized.db"
    "$SCRIPT_DIR/l104_intellect_memory.db"
)

# ─── COLORS ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ─── HARDWARE ─────────────────────────────────────────────────────────────────
ARCH=$(uname -m)
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
TOTAL_MEM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 4294967296) / 1073741824 ))

# ─── TIMING (native — no python3 dependency) ─────────────────────────────────
_now() {
    perl -MTime::HiRes=time -e 'printf "%.3f\n", time' 2>/dev/null || date +%s
}
_elapsed() {
    perl -e "printf '%.1f', $1 - $2" 2>/dev/null || echo "?"
}

# ─── BASH 3.2 COMPAT (macOS ships bash 3.2 — no declare -A) ─────────────────
_kv_set() { eval "__kv_${1}__${2}=\"\$3\""; }
_kv_get() { eval "printf '%s' \"\${__kv_${1}__${2}:-$3}\""; }

_bin_dst() {
    case "$1" in
        L104)            echo "$MACOS_DIR/$APP_NAME" ;;
        L104Daemon)      echo "$MACOS_DIR/L104Daemon" ;;
        L104NanoDaemon)  echo "$MACOS_DIR/L104NanoDaemon" ;;
    esac
}

_human_bytes() {
    local v=${1#-}
    local sign=""
    [[ "$1" == -* ]] && sign="-"
    if (( v >= 1073741824 )); then
        awk "BEGIN { printf \"${sign}%.1fG\", ${v}/1073741824 }"
    elif (( v >= 1048576 )); then
        awk "BEGIN { printf \"${sign}%.1fM\", ${v}/1048576 }"
    elif (( v >= 1024 )); then
        awk "BEGIN { printf \"${sign}%.1fK\", ${v}/1024 }"
    else
        echo "${sign}${v}B"
    fi
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────
_log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg" >> "$BUILD_LOG" 2>/dev/null || true
}
_log_section() {
    _log "════ $* ════"
}

# ─── OPTIONS ──────────────────────────────────────────────────────────────────
BUILD_CONFIG="debug"
LAUNCH_AFTER=false
VERBOSE=false
CLEAN_BUILD=false
SKIP_SIGN=false
DRY_RUN=false
SHOW_LOG=false
# Cap parallelism by memory: each Swift job ~1.5GB; leave 1GB headroom
MEM_JOBS=$(( TOTAL_MEM_GB > 1 ? (TOTAL_MEM_GB - 1) : 1 ))
JOBS=$(( CPU_CORES < MEM_JOBS ? CPU_CORES : MEM_JOBS ))
TARGETS=("L104")
USE_XCODE=false
XCODE_SCHEME=""
XCODE_DESTINATION=""
XCODE_DERIVED_DATA_PATH=""
XCODE_CONFIG_FILE=""
CODE_SIGNING_IDENTITY="-"  # Default: ad-hoc signing
CODE_SIGNING_REQUIRED="NO"
CODE_SIGNING_ALLOWED="YES"
XCODE_SDK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release|-r)   BUILD_CONFIG="release" ;;
        --debug|-d)     BUILD_CONFIG="debug" ;;
        --run)          LAUNCH_AFTER=true ;;
        --verbose|-v)   VERBOSE=true ;;
        --clean)        CLEAN_BUILD=true ;;
        --no-sign)      SKIP_SIGN=true ;;
        --dry-run)      DRY_RUN=true ;;
        --log)          SHOW_LOG=true ;;
        # Xcode build mode options
        -x|--xcode)     USE_XCODE=true ;;
        --scheme|-s)
            shift
            USE_XCODE=true
            XCODE_SCHEME="${1:?--scheme requires a name}"
            ;;
        --project)
            shift
            XCODE_PROJECT="${1}"
            USE_XCODE=true
            ;;
        --workspace)
            shift
            XCODE_WORKSPACE="${1}"
            USE_XCODE=true
            ;;
        --derivedDataPath)
            shift
            XCODE_DERIVED_DATA_PATH="${1}"
            ;;
        --xcconfig)
            shift
            XCODE_CONFIG_FILE="${1}"
            ;;
        --destination)
            shift
            XCODE_DESTINATION="${1}"
            ;;
        --sdk)
            shift
            XCODE_SDK="${1}"
            ;;
        --code-sign)
            shift
            CODE_SIGNING_IDENTITY="${1}"
            ;;
        --code-sign-required)
            shift
            CODE_SIGNING_REQUIRED="${1}"
            ;;
        --jobs|-j)
            shift
            JOBS="${1:?--jobs requires a number}"
            [[ "$JOBS" =~ ^[0-9]+$ ]] || { echo -e "${RED}--jobs must be a number${NC}"; exit 1; }
            ;;
        --target|-t)
            shift
            case "${1:?--target requires a name}" in
                L104|gui|app)          TARGETS=("L104") ;;
                L104Daemon|daemon)     TARGETS=("L104Daemon") ;;
                L104NanoDaemon|nano)   TARGETS=("L104NanoDaemon") ;;
                *) echo -e "${RED}Unknown target: $1 (use: L104|gui, daemon, nano)${NC}"; exit 1 ;;
            esac
            ;;
        --all|-a) TARGETS=("L104" "L104Daemon" "L104NanoDaemon") ;;
        --help|-h)
            cat <<'HELP'
L104 Quick Build v4.0 — SPM or Xcode Build + App Bundle

Usage: ./quick_build.sh [OPTIONS]

Build Modes:
  --debug, -d         Debug build (default, fastest incremental)
  --release, -r       Release build (WMO + LTO, optimized)
  --clean             Delete .build/ before compiling (full rebuild)

Build System:
  -x, --xcode         Use Xcode build system (xcodebuild) instead of SPM
  -s, --scheme <N>    Xcode scheme to build (e.g., L104SovereignIntellect)
  --project <P>       Xcode project path (auto-detected if not specified)
  --workspace <W>     Xcode workspace path (for workspace builds)
  --derivedDataPath   Custom derived data path
  --xcconfig <F>      Build settings from xcconfig file
  --destination <D>   Build destination (e.g., "platform=macOS")
  --sdk <S>           SDK to build against (default: macosx)

Code Signing:
  --code-sign <I>     Code signing identity ("-" for ad-hoc, or "Apple Development")
  --code-sign-required  YES/NO (default: NO)

Targets:
  --target, -t <T>    Build a specific target:
                        L104 | gui | app       → GUI application
                        L104Daemon | daemon    → Headless quantum daemon
                        L104NanoDaemon | nano  → Nano fault detection daemon
  --all, -a           Build all 3 targets (GUI + Daemon + NanoDaemon)
                      Default: L104 (GUI) only

Options:
  --run               Build then launch the GUI app (kills prior instance)
  --verbose, -v       Show full build compiler output
  --no-sign           Skip ad-hoc code signing step
  --jobs, -j N        Build parallelism (default: all CPU cores)
  --dry-run           Show what would be built without compiling
  --log               Show the last build log (.build/quick_build.log)

Pipeline (SPM mode):
  1. Pre-flight   — Verify Swift toolchain, Package.swift, disk space
  2. Compile      — `swift build --product <target>` per target (incremental)
  3. Bundle       — Copy binaries into L104Native.app/Contents/MacOS/
  4. Resources    — Bundle AppIcon.icns + *.json configs into Resources/
  5. Plist        — Generate Info.plist + PkgInfo stamp
  6. Sign         — Ad-hoc codesign --force --deep (skippable)
  7. Verify       — Architecture check, bundle structure validation
  8. Launch       — Optional: kill old + open (--run flag)

Pipeline (Xcode mode):
  1. Pre-flight   — Verify Xcode, xcodeproj, disk space
  2. Compile      — xcodebuild with full build settings
  3. Bundle       — Copy binaries into L104Native.app/Contents/MacOS/
  4. Resources    — Bundle AppIcon.icns + *.json configs into Resources/
  5. Plist        — Generate Info.plist + PkgInfo stamp
  6. Sign         — Full codesign with identity support
  7. Verify       — Architecture check, bundle structure validation
  8. Launch       — Optional: kill old + open (--run flag)

Examples:
  # SPM builds (default):
  ./quick_build.sh                        # Quick debug SPM build (~15s)
  ./quick_build.sh --all -r               # Release build, all 3 targets
  ./quick_build.sh -t daemon --clean      # Clean rebuild of daemon only

  # Xcode builds:
  ./quick_build.sh -x -s L104SovereignIntellect    # Xcode build with scheme
  ./quick_build.sh -x -s L104SovereignIntellect -r # Xcode release build
  ./quick_build.sh -x -s L104 --code-sign "-"      # Ad-hoc signed build
  ./quick_build.sh -x --xcconfig ReleaseConfig    # Custom xcconfig

  ./quick_build.sh --all --run            # Build everything + launch GUI
  ./quick_build.sh --dry-run --all -r     # Preview release build plan
  ./quick_build.sh --log                  # View last build log
HELP
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1 (try --help)${NC}"; exit 1 ;;
    esac
    shift
done

# ─── SHOW LOG MODE ────────────────────────────────────────────────────────────
if $SHOW_LOG; then
    if [[ -f "$BUILD_LOG" ]]; then
        echo -e "${CYAN}${BOLD}Build log:${NC} $BUILD_LOG"
        echo -e "${DIM}$(wc -l < "$BUILD_LOG") lines, $(du -h "$BUILD_LOG" | cut -f1 | tr -d ' ')${NC}"
        echo "─────────────────────────────────────────"
        cat "$BUILD_LOG"
    else
        echo -e "${YELLOW}No build log found. Run a build first.${NC}"
    fi
    exit 0
fi

# ─── BANNER ───────────────────────────────────────────────────────────────────
TARGET_LIST=$(IFS='+'; echo "${TARGETS[*]}")
echo -e "${PURPLE}${BOLD}⚡ L104 QUICK BUILD v3.0 — SPM Incremental${NC}"
echo -e "${DIM}  $BUILD_CONFIG | $TARGET_LIST | $ARCH | ${JOBS}j | ${TOTAL_MEM_GB}GB RAM${NC}"

cd "$SCRIPT_DIR"

# Ensure log directory exists (even before clean, which only removes .build contents)
mkdir -p "$SPM_BUILD_DIR"

# Rotate build log (keep last run only)
[[ -f "$BUILD_LOG" ]] && mv -f "$BUILD_LOG" "${BUILD_LOG}.prev" 2>/dev/null || true
_log_section "L104 QUICK BUILD v3.0 — $(date '+%Y-%m-%d %H:%M:%S')"
_log "Config: $BUILD_CONFIG | Targets: $TARGET_LIST | Arch: $ARCH | Jobs: $JOBS"

# ─── PRE-FLIGHT CHECKS ───────────────────────────────────────────────────────
PREFLIGHT_OK=true

# 1. Package.swift
if [[ ! -f "Package.swift" ]]; then
    echo -e "${RED}  ✗ Package.swift not found in $SCRIPT_DIR${NC}"
    _log "FAIL: Package.swift missing"
    exit 1
fi
echo -e "${GREEN}  ✓ Package.swift${NC}"

# 2. Swift toolchain
SWIFT_VERSION=$(swift --version 2>/dev/null | head -1 || echo "not found")
SWIFT_VER_NUM=$(echo "$SWIFT_VERSION" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 || echo "0")
SWIFT_MAJOR=$(echo "$SWIFT_VER_NUM" | cut -d. -f1)
if [[ "${SWIFT_MAJOR:-0}" -lt "$REQUIRED_SWIFT_MAJOR" ]]; then
    echo -e "${RED}  ✗ Swift $REQUIRED_SWIFT_MAJOR+ required (found: $SWIFT_VER_NUM)${NC}"
    _log "FAIL: Swift version $SWIFT_VER_NUM < $REQUIRED_SWIFT_MAJOR"
    PREFLIGHT_OK=false
else
    echo -e "${GREEN}  ✓ Swift $SWIFT_VER_NUM${NC}"
fi
_log "Swift: $SWIFT_VERSION"

# 3. Disk space (warn if < 2GB free)
FREE_BLOCKS=$(df -k "$SCRIPT_DIR" | tail -1 | awk '{print $4}')
FREE_GB=$(( FREE_BLOCKS / 1048576 ))
if (( FREE_GB < 2 )); then
    echo -e "${YELLOW}  ⚠ Low disk space: ${FREE_GB}GB free${NC}"
    _log "WARN: Low disk ${FREE_GB}GB"
else
    echo -e "${DIM}  ✓ Disk: ${FREE_GB}GB free${NC}"
fi

# 4. Source files count
if [[ -d "$SOURCE_DIR" ]]; then
    SWIFT_FILES=$(find "$SOURCE_DIR" -name '*.swift' -not -path '*/.*' | wc -l | tr -d ' ')
    SWIFT_LINES=$(find "$SOURCE_DIR" -name '*.swift' -not -path '*/.*' -exec cat {} + 2>/dev/null | wc -l | tr -d ' ')
    echo -e "${DIM}  ✓ Sources: ${SWIFT_FILES} Swift files, ${SWIFT_LINES} lines${NC}"
    _log "Sources: $SWIFT_FILES files, $SWIFT_LINES lines"
fi

if ! $PREFLIGHT_OK; then
    echo -e "${RED}${BOLD}  ✗ Pre-flight failed. Fix issues above.${NC}"
    exit 1
fi

# ─── DRY RUN ──────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo ""
    echo -e "${WHITE}${BOLD}  DRY RUN — Build Plan:${NC}"
    for TARGET in "${TARGETS[@]}"; do
        FLAGS="-j $JOBS --product $TARGET"
        [[ "$BUILD_CONFIG" = "release" ]] && FLAGS="$FLAGS -c release"
        echo -e "${CYAN}    swift build $FLAGS${NC}"
    done
    echo -e "${DIM}    → Binary install to $MACOS_DIR/${NC}"
    echo -e "${DIM}    → Resources to $RESOURCES_DIR/${NC}"
    $SKIP_SIGN && echo -e "${DIM}    → Signing: skipped${NC}" \
               || echo -e "${DIM}    → Signing: ad-hoc codesign${NC}"
    $LAUNCH_AFTER && echo -e "${DIM}    → Launch: open $APP_BUNDLE${NC}"
    echo ""
    echo -e "${DIM}  (Remove --dry-run to execute)${NC}"
    exit 0
fi

# ─── CLEAN (optional) ────────────────────────────────────────────────────────
if $CLEAN_BUILD; then
    echo -e "${YELLOW}  ⟳ Cleaning .build/ and SPM build DBs ...${NC}"
    # Preserve the log we just started
    TEMP_LOG=$(mktemp)
    [[ -f "$BUILD_LOG" ]] && cp "$BUILD_LOG" "$TEMP_LOG"
    # Clean SPM build directories (including locked build.db files)
    rm -rf "$SPM_BUILD_DIR"
    # Also clean any residual SPM databases in .build subdirs
    find "$SCRIPT_DIR/.build" -name "*.db" -type f -delete 2>/dev/null || true
    find "$SCRIPT_DIR/.build" -name "*.db-wal" -type f -delete 2>/dev/null || true
    find "$SCRIPT_DIR/.build" -name "*.db-shm" -type f -delete 2>/dev/null || true
    mkdir -p "$SPM_BUILD_DIR"
    [[ -f "$TEMP_LOG" ]] && mv "$TEMP_LOG" "$BUILD_LOG"
    echo -e "${GREEN}  ✓ Clean slate (SPM build DBs freed)${NC}"
    _log "Cleaned .build/ and SPM DBs"

    # Clean stale database WAL/journal files (prevents concurrent access errors on rebuild)
    echo -e "${YELLOW}  ⟳ Cleaning app database locks ...${NC}"
    CLEANED_DBS=0
    for db in "${DB_FILES[@]}"; do
        # Remove WAL journal files and shm files
        rm -f "${db}-wal" "${db}-shm" "${db}.lock" 2>/dev/null || true
        # If DB is locked, try to remove the lock file
        if [[ -f "${db}" ]]; then
            # Check if SQLite is holding a lock (try to open exclusive)
            sqlite3 "$db" "PRAGMA journal_mode;" >/dev/null 2>&1 || {
                # DB is locked - remove stale WAL if exists
                rm -f "${db}-wal" "${db}-shm" "${db}.lock" 2>/dev/null || true
            }
            CLEANED_DBS=$((CLEANED_DBS + 1))
        fi
    done
    echo -e "${GREEN}  ✓ Database locks cleaned ($CLEANED_DBS files)${NC}"
    _log "Cleaned $CLEANED_DBS database files"
fi

# ─── STEP 1: SPM INCREMENTAL COMPILE ─────────────────────────────────────────
GLOBAL_START=$(_now)

STEP_NUM=1
TOTAL_STEPS=$(( ${#TARGETS[@]} + 3 ))  # targets + bundle + sign + verify

BUILD_FAILED=false

for TARGET in "${TARGETS[@]}"; do
    STEP_START=$(_now)
    echo -e "${CYAN}  [$STEP_NUM/$TOTAL_STEPS] Compiling ${BOLD}$TARGET${NC}${CYAN} (incremental)...${NC}"
    _log_section "COMPILE $TARGET"

    SWIFT_FLAGS="-j $JOBS --product $TARGET"
    [[ "$BUILD_CONFIG" = "release" ]] && SWIFT_FLAGS="$SWIFT_FLAGS -c release"

    # Record pre-build binary size for delta
    BIN_PATH="$SPM_BUILD_DIR/$BUILD_CONFIG/$TARGET"
    if [[ -f "$BIN_PATH" ]]; then
        _kv_set old_sizes "$TARGET" "$(stat -f%z "$BIN_PATH" 2>/dev/null || echo 0)"
    else
        _kv_set old_sizes "$TARGET" 0
    fi

    if $VERBOSE; then
        # Verbose: stream output directly, also tee to log
        swift build $SWIFT_FLAGS 2>&1 | tee -a "$BUILD_LOG" || {
            BUILD_FAILED=true
            _log "FAIL: $TARGET compilation failed"
            break
        }
    else
        BUILD_OUTPUT=$(swift build $SWIFT_FLAGS 2>&1) || {
            echo -e "${RED}  ✗ COMPILATION FAILED — $TARGET${NC}"
            _log "FAIL: $TARGET compilation failed"
            echo "$BUILD_OUTPUT" >> "$BUILD_LOG"

            # Show deduplicated errors
            ERRORS=$(echo "$BUILD_OUTPUT" | grep -E "error:" | sort -u)
            ERR_COUNT=$(echo "$ERRORS" | grep -c "error:" || true)
            echo -e "${RED}    $ERR_COUNT unique error(s):${NC}"
            echo "$ERRORS" | head -20 | sed 's/^/      /'
            echo ""

            # Show context around first error for quick diagnosis
            FIRST_ERR=$(echo "$BUILD_OUTPUT" | grep -n "error:" | head -1 | cut -d: -f1)
            if [[ -n "$FIRST_ERR" ]]; then
                echo -e "${DIM}  First error context:${NC}"
                echo "$BUILD_OUTPUT" | sed -n "$((FIRST_ERR > 2 ? FIRST_ERR - 2 : 1)),$(( FIRST_ERR + 3 ))p" | head -8 | sed 's/^/      /'
            fi

            BUILD_FAILED=true
            break
        }

        echo "$BUILD_OUTPUT" >> "$BUILD_LOG"

        # Report compile step count
        COMPILE_STEPS=$(echo "$BUILD_OUTPUT" | grep -c "^\[" || true)
        _kv_set steps "$TARGET" "$COMPILE_STEPS"
        if (( COMPILE_STEPS > 0 )); then
            echo -e "${BLUE}    $COMPILE_STEPS compile steps${NC}"
        else
            echo -e "${BLUE}    (no recompilation needed — incremental cache hit)${NC}"
        fi

        # Show deduplicated warnings
        WARN_LINES=$(echo "$BUILD_OUTPUT" | grep "warning:" | sort -u || true)
        WARN_COUNT=$(echo "$WARN_LINES" | grep -c "warning:" || true)
        if (( WARN_COUNT > 0 )); then
            echo -e "${YELLOW}    $WARN_COUNT unique warning(s)${NC}"
            _log "Warnings ($TARGET): $WARN_COUNT"
            if $VERBOSE || (( WARN_COUNT <= 5 )); then
                echo "$WARN_LINES" | head -5 | sed 's/^/      /'
            fi
        fi
    fi

    STEP_END=$(_now)
    _kv_set times "$TARGET" "$(_elapsed "$STEP_END" "$STEP_START")"
    echo -e "${GREEN}  ✓ $TARGET compiled in ${BOLD}$(_kv_get times "$TARGET" "?")s${NC}"
    _log "$TARGET compiled in $(_kv_get times "$TARGET" "?")s"

    STEP_NUM=$((STEP_NUM + 1))
done

if $BUILD_FAILED; then
    echo -e "${RED}${BOLD}  ✗ Build failed. Fix errors and retry.${NC}"
    echo -e "${DIM}    Full log: $BUILD_LOG${NC}"
    _log "BUILD FAILED"
    exit 1
fi

# ─── STEP 2: BUNDLE — BINARY INSTALL + RESOURCES ─────────────────────────────
echo -e "${CYAN}  [$STEP_NUM/$TOTAL_STEPS] Updating app bundle...${NC}"
_log_section "BUNDLE"

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

BIN_DIR="$SPM_BUILD_DIR/$BUILD_CONFIG"

# Target → destination binary path mapping (via _bin_dst function for bash 3.2)

BUNDLE_OK=true
for TARGET in "${TARGETS[@]}"; do
    SRC="$BIN_DIR/$TARGET"
    DST="$(_bin_dst "$TARGET")"
    LABEL="$TARGET"
    [[ "$TARGET" = "L104" ]] && LABEL="$APP_NAME"

    if [[ ! -f "$SRC" ]]; then
        echo -e "${RED}    ✗ $LABEL binary not found at $SRC${NC}"
        _log "FAIL: $LABEL binary missing"
        BUNDLE_OK=false
        continue
    fi

    # Size tracking
    NEW_SIZE=$(stat -f%z "$SRC" 2>/dev/null || echo 0)
    OLD_SIZE=$(_kv_get old_sizes "$TARGET" 0)
    HUMAN_SIZE=$(du -h "$SRC" | cut -f1 | tr -d ' ')
    _kv_set sizes "$LABEL" "$HUMAN_SIZE"

    # Binary change detection — skip copy if identical
    if cmp -s "$SRC" "$DST" 2>/dev/null; then
        echo -e "${DIM}    ≡ $LABEL unchanged ($HUMAN_SIZE)${NC}"
        _log "$LABEL unchanged"
    else
        cp "$SRC" "$DST"
        chmod +x "$DST"

        # Show size delta
        if (( OLD_SIZE > 0 )) && (( NEW_SIZE != OLD_SIZE )); then
            DELTA=$(( NEW_SIZE - OLD_SIZE ))
            if (( DELTA > 0 )); then
                DELTA_H="+$(_human_bytes $DELTA)"
            else
                DELTA_H="$(_human_bytes $DELTA)"
            fi
            echo -e "${GREEN}    → $LABEL installed ($HUMAN_SIZE, ${DELTA_H})${NC}"
        else
            echo -e "${GREEN}    → $LABEL installed ($HUMAN_SIZE)${NC}"
        fi
        _log "$LABEL installed: $HUMAN_SIZE"
    fi
done

if ! $BUNDLE_OK; then
    echo -e "${RED}  ✗ Binary installation failed${NC}"
    _log "FAIL: binary installation"
    exit 1
fi

# App icon
if [[ -f "$SCRIPT_DIR/AppIcon.icns" ]]; then
    if ! cmp -s "$SCRIPT_DIR/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns" 2>/dev/null; then
        cp "$SCRIPT_DIR/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"
        echo -e "${DIM}    → AppIcon.icns updated${NC}"
    fi
fi

# JSON configs → Resources
BUNDLED_CONFIGS=0
for cfg in "$SCRIPT_DIR"/*.json; do
    [[ -f "$cfg" ]] || continue
    cfg_name=$(basename "$cfg")
    if ! cmp -s "$cfg" "$RESOURCES_DIR/$cfg_name" 2>/dev/null; then
        cp "$cfg" "$RESOURCES_DIR/$cfg_name"
        BUNDLED_CONFIGS=$((BUNDLED_CONFIGS + 1))
    fi
done
if (( BUNDLED_CONFIGS > 0 )); then
    echo -e "${BLUE}    $BUNDLED_CONFIGS config(s) bundled${NC}"
    _log "$BUNDLED_CONFIGS configs bundled"
fi

# Info.plist
cat > "$CONTENTS_DIR/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>L104 Sovereign Intellect</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleVersion</key>
    <string>$BUILD_NUMBER</string>
    <key>LSMinimumSystemVersion</key>
    <string>$MIN_MACOS</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSSupportsAutomaticTermination</key>
    <true/>
    <key>NSSupportsSuddenTermination</key>
    <false/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright 2026 Allentown L104. Sovereign Intellect v$VERSION. Quick Build v3.0. GOD_CODE=527.5184818492612</string>
    <key>LSUIElement</key>
    <false/>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsArbitraryLoads</key>
        <true/>
    </dict>
</dict>
</plist>
PLIST

echo -n "APPL????" > "$CONTENTS_DIR/PkgInfo"

# IPC directories for daemon targets
for TARGET in "${TARGETS[@]}"; do
    if [[ "$TARGET" == "L104Daemon" || "$TARGET" == "L104NanoDaemon" ]]; then
        for dir in "${IPC_DIRS[@]}"; do
            mkdir -p "$dir" 2>/dev/null || true
        done
        echo -e "${DIM}    → IPC directories ensured${NC}"
        _log "IPC directories created"
        break  # only need to do this once
    fi
done

echo -e "${GREEN}  ✓ Bundle updated (build $BUILD_NUMBER)${NC}"
_log "Bundle updated: build $BUILD_NUMBER"
STEP_NUM=$((STEP_NUM + 1))

# ─── STEP 3: CODE SIGN ───────────────────────────────────────────────────────
if $SKIP_SIGN; then
    echo -e "${DIM}  [$STEP_NUM/$TOTAL_STEPS] Signing skipped (--no-sign)${NC}"
    _log "Signing skipped"
else
    echo -e "${CYAN}  [$STEP_NUM/$TOTAL_STEPS] Signing...${NC}"
    if codesign --force --deep --sign - "$APP_BUNDLE" 2>/dev/null; then
        echo -e "${GREEN}  ✓ Signed (ad-hoc)${NC}"
        _log "Signed: ad-hoc"
    else
        echo -e "${YELLOW}  ⚠ Signing failed (non-fatal, app may still run)${NC}"
        _log "WARN: signing failed"
    fi
fi
touch "$APP_BUNDLE"
STEP_NUM=$((STEP_NUM + 1))

# ─── STEP 4: POST-BUILD VERIFICATION ─────────────────────────────────────────
echo -e "${CYAN}  [$STEP_NUM/$TOTAL_STEPS] Verifying...${NC}"
_log_section "VERIFY"
VERIFY_OK=true

# Check bundle structure
for required in "$CONTENTS_DIR/Info.plist" "$CONTENTS_DIR/PkgInfo"; do
    if [[ ! -f "$required" ]]; then
        echo -e "${RED}    ✗ Missing: $(basename "$required")${NC}"
        VERIFY_OK=false
    fi
done

# Check each target binary exists and matches architecture
for TARGET in "${TARGETS[@]}"; do
    DST="$(_bin_dst "$TARGET")"
    LABEL="$TARGET"; [[ "$TARGET" = "L104" ]] && LABEL="$APP_NAME"

    if [[ ! -x "$DST" ]]; then
        echo -e "${RED}    ✗ $LABEL not executable${NC}"
        VERIFY_OK=false
        continue
    fi

    BIN_ARCH=$(file "$DST" 2>/dev/null | grep -oE 'arm64|x86_64' | head -1 || echo "unknown")
    if [[ "$BIN_ARCH" != "$ARCH" ]]; then
        echo -e "${YELLOW}    ⚠ $LABEL arch: $BIN_ARCH (host: $ARCH)${NC}"
    fi
done

if $VERIFY_OK; then
    echo -e "${GREEN}  ✓ Bundle verified${NC}"
    _log "Verification passed"
else
    echo -e "${YELLOW}  ⚠ Verification issues (see above)${NC}"
    _log "WARN: verification issues"
fi

# ─── BUILD HISTORY ────────────────────────────────────────────────────────────
GLOBAL_END=$(_now)
TOTAL_TIME=$(_elapsed "$GLOBAL_END" "$GLOBAL_START")

# Append to history (keep last 20 builds)
HIST_LINE="$(date '+%Y-%m-%d %H:%M') | $BUILD_CONFIG | $TARGET_LIST | ${TOTAL_TIME}s | $ARCH"
echo "$HIST_LINE" >> "$HISTORY_FILE" 2>/dev/null || true
tail -20 "$HISTORY_FILE" > "${HISTORY_FILE}.tmp" 2>/dev/null && mv "${HISTORY_FILE}.tmp" "$HISTORY_FILE" 2>/dev/null || true

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  ⚡ Quick Build v3.0 complete: ${TOTAL_TIME}s${NC} ${DIM}($BUILD_CONFIG | $ARCH)${NC}"

# Per-target breakdown
for TARGET in "${TARGETS[@]}"; do
    LABEL="$TARGET"
    [[ "$TARGET" = "L104" ]] && LABEL="$APP_NAME"
    T_TIME="$(_kv_get times "$TARGET" "?")"
    T_SIZE="$(_kv_get sizes "$LABEL" "?")"
    T_STEPS="$(_kv_get steps "$TARGET" 0)"
    echo -e "${DIM}    $LABEL: ${T_TIME}s, ${T_SIZE}, ${T_STEPS} steps${NC}"
done

echo -e "${DIM}    Bundle: $APP_BUNDLE${NC}"
echo -e "${DIM}    Build#: $BUILD_NUMBER${NC}"
echo -e "${DIM}    Log:    $BUILD_LOG${NC}"

_log_section "COMPLETE: ${TOTAL_TIME}s"

# ─── LAUNCH ───────────────────────────────────────────────────────────────────
if $LAUNCH_AFTER; then
    pkill -f "$APP_NAME" 2>/dev/null || true
    sleep 0.5
    echo -e "${CYAN}  🚀 Launching $APP_NAME...${NC}"
    _log "Launching $APP_NAME"
    open "$APP_BUNDLE"
fi
