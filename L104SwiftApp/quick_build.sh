#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  L104 SOVEREIGN INTELLECT — QUICK BUILD v1.0
#  Incremental SPM build + .app bundle wrapping
#  ~15s incremental vs ~150s full rebuild on 4GB machines
#  Uses swift build (SPM) for incremental compilation, then wraps
#  the resulting binary in L104Native.app bundle with Info.plist,
#  CPython bridge, code signing, and hot-restart support.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── CONFIGURATION ───
APP_NAME="L104Native"
BUNDLE_ID="com.allentown.l104"
VERSION="24.1"
BUILD_NUMBER=$(date +%Y%m%d%H%M)
MIN_MACOS="12.0"

# ─── PATHS ───
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/Sources"
L104V2_DIR="$SOURCE_DIR/L104v2"
APP_BUNDLE="$SCRIPT_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
EXECUTABLE="$MACOS_DIR/$APP_NAME"
SPM_BUILD_DIR="$SCRIPT_DIR/.build"

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

# ─── HARDWARE ───
ARCH=$(uname -m)
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ─── OPTIONS ───
BUILD_CONFIG="debug"
LAUNCH_AFTER=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release|-r)   BUILD_CONFIG="release" ;;
        --debug|-d)     BUILD_CONFIG="debug" ;;
        --run)          LAUNCH_AFTER=true ;;
        --verbose|-v)   VERBOSE=true ;;
        --help|-h)
            echo "L104 Quick Build — Incremental SPM + App Bundle"
            echo ""
            echo "Usage: ./quick_build.sh [OPTIONS]"
            echo "  --debug, -d    Debug build (default, fastest)"
            echo "  --release, -r  Release build (optimized)"
            echo "  --run          Build and launch"
            echo "  --verbose, -v  Show compiler output"
            echo "  --help, -h     This help"
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
    shift
done

# ─── BANNER ───
echo -e "${PURPLE}${BOLD}⚡ L104 QUICK BUILD — SPM Incremental${NC} ${DIM}($BUILD_CONFIG)${NC}"

# Ensure we're in the directory containing Package.swift
cd "$SCRIPT_DIR"

# ─── STEP 1: SPM INCREMENTAL BUILD ───
BUILD_START=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)

SWIFT_BUILD_FLAGS=""
if [ "$BUILD_CONFIG" = "release" ]; then
    SWIFT_BUILD_FLAGS="-c release"
fi

echo -e "${CYAN}  [1/4] Compiling (incremental)...${NC}"

if $VERBOSE; then
    swift build $SWIFT_BUILD_FLAGS 2>&1
else
    BUILD_OUTPUT=$(swift build $SWIFT_BUILD_FLAGS 2>&1) || {
        echo -e "${RED}  ✗ COMPILATION FAILED${NC}"
        echo "$BUILD_OUTPUT" | grep -E "error:" | head -15
        exit 1
    }
    # Show compile step count
    STEPS=$(echo "$BUILD_OUTPUT" | grep -c "^\[" || true)
    if [ "$STEPS" -gt 0 ]; then
        echo -e "${BLUE}    $STEPS compile steps${NC}"
    fi
    # Show warnings
    WARNINGS=$(echo "$BUILD_OUTPUT" | grep -c "warning:" || true)
    if [ "$WARNINGS" -gt 0 ]; then
        echo -e "${YELLOW}    $WARNINGS warning(s)${NC}"
    fi
fi

BUILD_END=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)
COMPILE_TIME=$(python3 -c "print(f'{${BUILD_END} - ${BUILD_START}:.1f}')" 2>/dev/null || echo "?")
echo -e "${GREEN}  ✓ Compiled in ${BOLD}${COMPILE_TIME}s${NC}"

# ─── STEP 2: LOCATE SPM BINARY ───
if [ "$BUILD_CONFIG" = "release" ]; then
    SPM_BINARY="$SPM_BUILD_DIR/release/L104"
else
    SPM_BINARY="$SPM_BUILD_DIR/debug/L104"
fi

if [ ! -f "$SPM_BINARY" ]; then
    echo -e "${RED}  ✗ SPM binary not found at $SPM_BINARY${NC}"
    exit 1
fi

SPM_BINARY_SIZE=$(du -h "$SPM_BINARY" | cut -f1 | tr -d ' ')

# ─── STEP 3: CREATE/UPDATE APP BUNDLE ───
echo -e "${CYAN}  [2/4] Updating app bundle...${NC}"

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

# Copy binary, renaming from "L104" to "L104Native"
cp "$SPM_BINARY" "$EXECUTABLE"
chmod +x "$EXECUTABLE"

# Copy icon
if [ -f "$SCRIPT_DIR/AppIcon.icns" ]; then
    cp "$SCRIPT_DIR/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"
fi

# Create Info.plist
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
    <string>Copyright 2026 Allentown L104. Sovereign Intellect v$VERSION. Quick Build. GOD_CODE=527.5184818492612</string>
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
echo -e "${GREEN}  ✓ Bundle updated${NC}"

# ─── STEP 4: CODE SIGN ───
echo -e "${CYAN}  [3/4] Signing...${NC}"
if codesign --force --deep --sign - "$APP_BUNDLE" 2>/dev/null; then
    echo -e "${GREEN}  ✓ Signed (ad-hoc)${NC}"
else
    echo -e "${YELLOW}  ⚠ Signing skipped${NC}"
fi
touch "$APP_BUNDLE"

# ─── SUMMARY ───
TOTAL_END=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)
TOTAL_TIME=$(python3 -c "print(f'{${TOTAL_END} - ${BUILD_START}:.1f}')" 2>/dev/null || echo "?")

echo -e "${CYAN}  [4/4] Done!${NC}"
echo -e "${GREEN}${BOLD}  ⚡ Quick build: ${TOTAL_TIME}s${NC} ${DIM}(binary: $SPM_BINARY_SIZE, mode: $BUILD_CONFIG)${NC}"

# ─── LAUNCH ───
if $LAUNCH_AFTER; then
    pkill -f "$APP_NAME" 2>/dev/null || true
    sleep 0.5
    echo -e "${CYAN}  🚀 Launching...${NC}"
    open "$APP_BUNDLE"
fi
