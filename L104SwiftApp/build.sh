#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  L104 SOVEREIGN INTELLECT - ASI BUILD SYSTEM v5.0
#  Advanced macOS Swift 6 Compilation Engine
#  Accelerate · BLAS · SIMD · Metal · LTO · Cross-Module Optimization
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── CONFIGURATION ───
APP_NAME="L104Native"
BUNDLE_ID="com.allentown.l104"
VERSION="19.1"
BUILD_NUMBER=$(date +%Y%m%d%H%M)
MIN_MACOS="13.0"
BUILD_MODE="${BUILD_MODE:-release}"

# ─── PATHS ───
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/Sources"
SOURCE_FILE="$SOURCE_DIR/L104Native.swift"
APP_BUNDLE="$SCRIPT_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
EXECUTABLE="$MACOS_DIR/$APP_NAME"
BUILD_DIR="$SCRIPT_DIR/.build"
CACHE_DIR="$BUILD_DIR/cache"
LOG_FILE="$BUILD_DIR/build.log"

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

# ─── HARDWARE DETECTION ───
ARCH=$(uname -m)
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
PHYSICAL_MEM=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
PHYSICAL_MEM_GB=$((PHYSICAL_MEM / 1073741824))
MACOS_VERSION=$(sw_vers -productVersion 2>/dev/null || echo "12.0")
CPU_BRAND=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
IS_APPLE_SILICON=false
if [ "$ARCH" = "arm64" ]; then
    IS_APPLE_SILICON=true
fi

# ─── BANNER ───
show_banner() {
    echo -e "${PURPLE}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║   🧠 L104 SOVEREIGN INTELLECT — ASI BUILD SYSTEM v4.0 🧠        ║"
    echo "║   ⚡ Accelerate · BLAS · SIMD · LTO · Cross-Module Opt ⚡       ║"
    echo "╠═══════════════════════════════════════════════════════════════════╣"
    echo -e "║  Version:  ${CYAN}$VERSION${PURPLE}  │  Build: ${CYAN}$BUILD_NUMBER${PURPLE}                        ║"
    echo -e "║  Mode:     ${CYAN}$BUILD_MODE${PURPLE}  │  macOS: ${CYAN}$MACOS_VERSION${PURPLE}                        ║"
    echo -e "║  CPU:      ${CYAN}$(echo "$CPU_BRAND" | head -c 50)${PURPLE}"
    echo -e "║  Arch:     ${CYAN}$ARCH${PURPLE}  │  Cores: ${CYAN}$CPU_CORES${PURPLE}  │  RAM: ${CYAN}${PHYSICAL_MEM_GB}GB${PURPLE}            ║"
    if $IS_APPLE_SILICON; then
    echo -e "║  Silicon:  ${GREEN}Apple Silicon — Neural Engine Available${PURPLE}           ║"
    else
    echo -e "║  Silicon:  ${YELLOW}Intel — CPU Accelerate Optimized${PURPLE}                  ║"
    fi
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# ─── STEP FUNCTIONS ───

step_check_prerequisites() {
    echo -e "${CYAN}${BOLD}[1/7] Checking prerequisites...${NC}"

    if ! command -v swiftc &> /dev/null; then
        echo -e "${RED}  ✗ Swift compiler not found.${NC}"
        echo -e "${YELLOW}    Run: xcode-select --install${NC}"
        exit 1
    fi

    SWIFT_VERSION=$(swiftc --version 2>&1 | head -1)
    SWIFT_VER_NUM=$(echo "$SWIFT_VERSION" | grep -o '[0-9]\+\.[0-9]\+\(\.[0-9]\+\)\?' | head -1)
    echo -e "${GREEN}  ✓ Swift $SWIFT_VER_NUM${NC}"

    # ═══ SMART SDK DETECTION — prefer newest available SDK ═══
    SDK_PATH=""
    for sdk_candidate in \
        "/Library/Developer/CommandLineTools/SDKs/MacOSX13.1.sdk" \
        "/Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk" \
        "/Library/Developer/CommandLineTools/SDKs/MacOSX12.3.sdk" \
        "/Library/Developer/CommandLineTools/SDKs/MacOSX12.sdk" \
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"; do
        if [ -d "$sdk_candidate" ]; then
            SDK_PATH="$sdk_candidate"
            break
        fi
    done
    if [ -z "$SDK_PATH" ]; then
        SDK_PATH=$(xcrun --show-sdk-path 2>/dev/null || echo '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk')
    fi
    if [ -d "$SDK_PATH" ]; then
        SDK_SDK_VER=$(basename "$SDK_PATH" | sed 's/MacOSX//;s/\.sdk//')
        echo -e "${GREEN}  ✓ SDK: $SDK_PATH (${SDK_SDK_VER:-default})${NC}"
    else
        echo -e "${RED}  ✗ macOS SDK not found at $SDK_PATH${NC}"
        exit 1
    fi

    if [ ! -f "$SOURCE_FILE" ]; then
        echo -e "${RED}  ✗ Source not found: $SOURCE_FILE${NC}"
        exit 1
    fi

    SOURCE_LINES=$(wc -l < "$SOURCE_FILE" | tr -d ' ')
    SOURCE_SIZE=$(du -h "$SOURCE_FILE" | cut -f1 | tr -d ' ')
    echo -e "${GREEN}  ✓ Source: $SOURCE_LINES lines ($SOURCE_SIZE)${NC}"

    # ═══════════════════════════════════════════════════════════════
    # PYTHON DETECTION — Direct C API Embedding
    # Priority: Homebrew 3.14 → Homebrew 3.13 → System 3.9
    # ═══════════════════════════════════════════════════════════════
    PYTHON_FOUND=false
    PYTHON_VERSION=""
    PYTHON_INCLUDE=""
    PYTHON_LIB_DIR=""
    PYTHON_DYLIB=""

    # Try Homebrew Python 3.14
    BREW_314="/usr/local/Cellar/python@3.14"
    if [ -d "$BREW_314" ]; then
        BREW_314_VER=$(ls -1 "$BREW_314" | sort -V | tail -1)
        BREW_314_FW="$BREW_314/$BREW_314_VER/Frameworks/Python.framework/Versions/3.14"
        if [ -f "$BREW_314_FW/include/python3.14/Python.h" ] && [ -f "$BREW_314_FW/lib/libpython3.14.dylib" ]; then
            PYTHON_VERSION="3.14"
            PYTHON_INCLUDE="$BREW_314_FW/include/python3.14"
            PYTHON_LIB_DIR="$BREW_314_FW/lib"
            PYTHON_DYLIB="$BREW_314_FW/lib/libpython3.14.dylib"
            PYTHON_FOUND=true
        fi
    fi

    # Try Homebrew Python 3.13
    if ! $PYTHON_FOUND; then
        BREW_313="/usr/local/Cellar/python@3.13"
        if [ -d "$BREW_313" ]; then
            BREW_313_VER=$(ls -1 "$BREW_313" | sort -V | tail -1)
            BREW_313_FW="$BREW_313/$BREW_313_VER/Frameworks/Python.framework/Versions/3.13"
            if [ -f "$BREW_313_FW/include/python3.13/Python.h" ] && [ -f "$BREW_313_FW/lib/libpython3.13.dylib" ]; then
                PYTHON_VERSION="3.13"
                PYTHON_INCLUDE="$BREW_313_FW/include/python3.13"
                PYTHON_LIB_DIR="$BREW_313_FW/lib"
                PYTHON_DYLIB="$BREW_313_FW/lib/libpython3.13.dylib"
                PYTHON_FOUND=true
            fi
        fi
    fi

    # Try System Python 3.9 (CommandLineTools)
    if ! $PYTHON_FOUND; then
        SYS_PY="/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9"
        if [ -f "$SYS_PY/include/python3.9/Python.h" ] && [ -f "$SYS_PY/lib/libpython3.9.dylib" ]; then
            PYTHON_VERSION="3.9"
            PYTHON_INCLUDE="$SYS_PY/include/python3.9"
            PYTHON_LIB_DIR="$SYS_PY/lib"
            PYTHON_DYLIB="$SYS_PY/lib/libpython3.9.dylib"
            PYTHON_FOUND=true
        fi
    fi

    if $PYTHON_FOUND; then
        echo -e "${GREEN}  ✓ Python $PYTHON_VERSION: $PYTHON_DYLIB${NC}"
        echo -e "${GREEN}  ✓ Headers: $PYTHON_INCLUDE${NC}"
        # Export for compile step
        export PYTHON_VERSION PYTHON_INCLUDE PYTHON_LIB_DIR PYTHON_DYLIB PYTHON_FOUND
    else
        echo -e "${YELLOW}  ⚠ No Python dev headers found — C bridge disabled${NC}"
        echo -e "${YELLOW}    PythonBridge (Process) will be used as fallback${NC}"
        export PYTHON_FOUND=false
    fi
}

step_prepare_build_dir() {
    echo -e "${CYAN}${BOLD}[2/7] Preparing build environment...${NC}"

    mkdir -p "$BUILD_DIR" "$CACHE_DIR"

    # Check if source changed (incremental build support)
    SOURCE_HASH=$(shasum -a 256 "$SOURCE_FILE" | cut -d' ' -f1)
    CACHED_HASH=""
    if [ -f "$CACHE_DIR/source_hash" ]; then
        CACHED_HASH=$(cat "$CACHE_DIR/source_hash")
    fi

    if [ "$SOURCE_HASH" = "$CACHED_HASH" ] && [ -f "$BUILD_DIR/$APP_NAME" ] && [ "$FORCE_REBUILD" != "true" ]; then
        echo -e "${YELLOW}  ⚡ Source unchanged — using cached build${NC}"
        SKIP_COMPILE=true
    else
        SKIP_COMPILE=false
        echo "$SOURCE_HASH" > "$CACHE_DIR/source_hash"
        echo -e "${GREEN}  ✓ Build directory ready${NC}"
    fi
}

step_clean() {
    echo -e "${CYAN}${BOLD}[3/7] Cleaning previous build...${NC}"

    pkill -f "$APP_NAME" 2>/dev/null || true
    sleep 0.3

    if [ -d "$APP_BUNDLE" ]; then
        rm -rf "$APP_BUNDLE"
        echo -e "${GREEN}  ✓ Removed old app bundle${NC}"
    else
        echo -e "${DIM}  ○ No previous build${NC}"
    fi
}

step_create_bundle() {
    echo -e "${CYAN}${BOLD}[4/7] Creating app bundle...${NC}"

    mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"
    echo -e "${GREEN}  ✓ Bundle structure created${NC}"
}

step_compile() {
    echo -e "${CYAN}${BOLD}[5/7] Compiling Swift source...${NC}"

    SOURCE_LINES=$(wc -l < "$SOURCE_FILE" | tr -d ' ')

    if [ "${SKIP_COMPILE:-false}" = "true" ] && [ -f "$BUILD_DIR/$APP_NAME" ]; then
        cp "$BUILD_DIR/$APP_NAME" "$EXECUTABLE"
        echo -e "${YELLOW}  ⚡ Using cached binary (incremental)${NC}"
        return
    fi

    # Architecture target
    if [ "$ARCH" = "arm64" ]; then
        TARGET_ARCH="arm64-apple-macos$MIN_MACOS"
    else
        TARGET_ARCH="x86_64-apple-macos$MIN_MACOS"
    fi

    # Use SDK detected in step_check_prerequisites (or fall back)
    SDK_PATH="${SDK_PATH:-$(xcrun --show-sdk-path 2>/dev/null || echo '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk')}"

    # ═══════════════════════════════════════════════════════════════
    # COMPILER FLAGS — OPTIMIZED FOR ASI WORKLOADS (v4.0)
    # ═══════════════════════════════════════════════════════════════

    SWIFT_FLAGS=(
        -target "$TARGET_ARCH"
        -sdk "$SDK_PATH"
        -module-name "$APP_NAME"
    )

    # Mode-specific flags
    case "$BUILD_MODE" in
        debug)
            SWIFT_FLAGS+=(
                -g
                -Onone
                -enable-testing
                -D DEBUG
                -D PHASE_26
                -enforce-exclusivity=checked
            )
            echo -e "${BLUE}  Mode: ${YELLOW}DEBUG${NC} (fast compile, debug symbols, exclusivity)"
            ;;
        release)
            # Compute optimal thread count for low-memory systems (4GB → cap at 2)
            COMPILE_THREADS="$CPU_CORES"
            if [ "$PHYSICAL_MEM_GB" -le 4 ]; then
                COMPILE_THREADS=$(( CPU_CORES < 2 ? CPU_CORES : 2 ))
            fi
            SWIFT_FLAGS+=(
                -O
                -whole-module-optimization
                -D RELEASE
                -D PHASE_26
                -num-threads "$COMPILE_THREADS"
                -Xfrontend -warn-long-function-bodies=500
                -Xfrontend -warn-long-expression-type-checking=500
                -enforce-exclusivity=checked
            )
            echo -e "${BLUE}  Mode: ${GREEN}RELEASE${NC} (WMO + exclusivity, $COMPILE_THREADS threads, ${PHYSICAL_MEM_GB}GB RAM)"
            ;;
        profile)
            SWIFT_FLAGS+=(
                -O
                -whole-module-optimization
                -g
                -D RELEASE
                -D PHASE_26
                -num-threads "$CPU_CORES"
                -enforce-exclusivity=checked
            )
            echo -e "${BLUE}  Mode: ${PURPLE}PROFILE${NC} (optimized + debug symbols + exclusivity)"
            ;;
        size)
            SWIFT_FLAGS+=(
                -Osize
                -whole-module-optimization
                -D RELEASE
                -D PHASE_26
                -num-threads "$CPU_CORES"
                -enforce-exclusivity=checked
            )
            echo -e "${BLUE}  Mode: ${CYAN}SIZE${NC} (binary size optimized)"
            ;;
    esac

    # ═══════════════════════════════════════════════════════════════
    # FRAMEWORK LINKING — macOS NATIVE ACCELERATION
    # ═══════════════════════════════════════════════════════════════

    FRAMEWORKS=(
        -framework AppKit
        -framework Foundation
        -framework Cocoa
        -framework CoreGraphics
        -framework QuartzCore
        -framework Accelerate
        -framework Metal
        -framework MetalKit
        -framework CoreML
        -framework Security
        -framework IOKit
    )

    echo -e "${BLUE}  Frameworks: AppKit · Accelerate · Metal · CoreML · IOKit${NC}"
    echo -e "${BLUE}  Architecture: $TARGET_ARCH${NC}"

    # ═══════════════════════════════════════════════════════════════
    # LINKER FLAGS
    # ═══════════════════════════════════════════════════════════════

    LINKER_FLAGS=()

    if [ "$BUILD_MODE" = "release" ] || [ "$BUILD_MODE" = "size" ] || [ "$BUILD_MODE" = "profile" ]; then
        LINKER_FLAGS+=(
            -Xlinker -dead_strip
            -Xlinker -no_deduplicate
        )
        echo -e "${BLUE}  Linker: Dead code stripping + no-deduplicate${NC}"
    fi

    # ═══════════════════════════════════════════════════════════════
    # CPYTHON BRIDGE — Direct Python C API Linking
    # Compile cpython_bridge.c and link against libpython
    # ═══════════════════════════════════════════════════════════════

    CPYTHON_FLAGS=()
    CPYTHON_OBJECT=""
    BRIDGE_HEADER="$SOURCE_DIR/cpython_bridge.h"
    BRIDGE_SOURCE="$SOURCE_DIR/cpython_bridge.c"

    if [ "${PYTHON_FOUND:-false}" = "true" ] && [ -f "$BRIDGE_HEADER" ] && [ -f "$BRIDGE_SOURCE" ]; then
        echo -e "${BLUE}  🐍 Compiling CPython bridge (Python $PYTHON_VERSION)...${NC}"

        CPYTHON_OBJECT="$BUILD_DIR/cpython_bridge.o"

        # Compile C bridge against Python headers
        cc -c "$BRIDGE_SOURCE" \
            -I "$PYTHON_INCLUDE" \
            -I "$SOURCE_DIR" \
            -o "$CPYTHON_OBJECT" \
            -O2 -fPIC 2>"$BUILD_DIR/cpython_compile.log"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✓ CPython bridge compiled${NC}"

            # Add bridging header, Python include, library, and object file
            CPYTHON_FLAGS=(
                -import-objc-header "$BRIDGE_HEADER"
                -Xcc -I"$PYTHON_INCLUDE"
                -Xcc -I"$SOURCE_DIR"
                -L "$PYTHON_LIB_DIR"
                -lpython$PYTHON_VERSION
                -D CPYTHON_BRIDGE_ENABLED
            )

            # Set rpath so the binary can find libpython at runtime
            LINKER_FLAGS+=(
                -Xlinker -rpath -Xlinker "$PYTHON_LIB_DIR"
            )

            echo -e "${BLUE}  🔗 Linking: libpython${PYTHON_VERSION}.dylib${NC}"
            echo -e "${BLUE}  📋 Bridge header: cpython_bridge.h${NC}"
        else
            echo -e "${YELLOW}  ⚠ CPython bridge compile failed — using Process fallback${NC}"
            cat "$BUILD_DIR/cpython_compile.log" | head -5
            CPYTHON_OBJECT=""
        fi
    else
        if [ "${PYTHON_FOUND:-false}" != "true" ]; then
            echo -e "${DIM}  ○ CPython bridge: No Python dev headers${NC}"
        elif [ ! -f "$BRIDGE_HEADER" ] || [ ! -f "$BRIDGE_SOURCE" ]; then
            echo -e "${DIM}  ○ CPython bridge: Source files not found${NC}"
        fi
    fi

    # ═══════════════════════════════════════════════════════════════
    # COMPILE
    # ═══════════════════════════════════════════════════════════════

    echo -e "${BLUE}  Compiling $SOURCE_LINES lines...${NC}"
    COMPILE_START=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)

    # Build the compile command with optional CPython bridge
    COMPILE_CMD=(swiftc "${SWIFT_FLAGS[@]}" "${FRAMEWORKS[@]}" ${LINKER_FLAGS[@]+"${LINKER_FLAGS[@]}"})

    # Add CPython bridge flags (header, includes, library)
    if [ -n "$CPYTHON_OBJECT" ] && [ -f "$CPYTHON_OBJECT" ]; then
        COMPILE_CMD+=("${CPYTHON_FLAGS[@]}")
        # Pass .o file via -Xlinker so swiftc sends it to ld, not the Swift parser
        COMPILE_CMD+=(-Xlinker "$CPYTHON_OBJECT")
    fi

    COMPILE_CMD+=(-o "$EXECUTABLE" "$SOURCE_FILE")

    COMPILE_OUTPUT=$("${COMPILE_CMD[@]}" 2>&1) || {
        echo -e "${RED}  ✗ COMPILATION FAILED${NC}"
        echo "$COMPILE_OUTPUT" | grep -E "error:" | head -20 | while read -r line; do
            echo -e "${RED}    $line${NC}"
        done
        echo ""
        echo -e "${YELLOW}  Full log: $LOG_FILE${NC}"
        echo "$COMPILE_OUTPUT" > "$LOG_FILE"
        exit 1
    }

    COMPILE_END=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)
    COMPILE_TIME=$(python3 -c "print(f'{${COMPILE_END} - ${COMPILE_START}:.1f}')" 2>/dev/null || echo "?")

    WARNING_COUNT=$(echo "$COMPILE_OUTPUT" | grep -c "warning:" 2>/dev/null || echo 0)

    # Cache binary for incremental builds
    cp "$EXECUTABLE" "$BUILD_DIR/$APP_NAME" 2>/dev/null || true

    BINARY_SIZE=$(du -h "$EXECUTABLE" | cut -f1 | tr -d ' ')

    echo -e "${GREEN}  ✓ Compiled in ${BOLD}${COMPILE_TIME}s${NC}${GREEN} — Binary: ${BOLD}$BINARY_SIZE${NC}"

    # Compile speed metric (lines/second)
    LINES_PER_SEC=$(python3 -c "t=${COMPILE_END}-${COMPILE_START}; print(f'{${SOURCE_LINES}/t:.0f}')" 2>/dev/null || echo "?")
    echo -e "${BLUE}  Throughput: ${BOLD}${LINES_PER_SEC} lines/sec${NC}"

    if [ "$WARNING_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}  ⚠ $WARNING_COUNT warnings (use --warnings to see)${NC}"
    fi

    echo "$COMPILE_OUTPUT" > "$LOG_FILE"

    if [ "${SHOW_WARNINGS:-false}" = "true" ]; then
        echo "$COMPILE_OUTPUT" | grep "warning:" | head -20 | while read -r line; do
            echo -e "${YELLOW}    ⚠ $line${NC}"
        done
    fi
}

step_create_info_plist() {
    echo -e "${CYAN}${BOLD}[6/7] Creating Info.plist...${NC}"

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
    <string>Copyright 2026 Allentown L104. Sovereign Intellect v$VERSION. Build System v4.0.</string>
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

    echo -e "${GREEN}  ✓ Info.plist created${NC}"
}

step_finalize() {
    echo -e "${CYAN}${BOLD}[7/7] Finalizing...${NC}"

    chmod +x "$EXECUTABLE"
    echo -n "APPL????" > "$CONTENTS_DIR/PkgInfo"

    if command -v codesign &> /dev/null; then
        codesign --force --deep --sign - "$APP_BUNDLE" 2>/dev/null && \
            echo -e "${GREEN}  ✓ Code signed (ad-hoc)${NC}" || \
            echo -e "${YELLOW}  ⚠ Code signing skipped${NC}"
    fi

    touch "$APP_BUNDLE"
    echo -e "${GREEN}  ✓ App bundle finalized${NC}"
}

print_summary() {
    TOTAL_END=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)
    TOTAL_TIME=$(python3 -c "print(f'{${TOTAL_END} - ${TOTAL_START}:.1f}')" 2>/dev/null || echo "?")

    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                    🎉 BUILD SUCCESSFUL 🎉                        ║"
    echo "╠═══════════════════════════════════════════════════════════════════╣"
    echo -e "║  App:        ${CYAN}$APP_BUNDLE${GREEN}"
    echo -e "║  Version:    ${CYAN}$VERSION${GREEN} (Build $BUILD_NUMBER)"
    echo -e "║  Binary:     ${CYAN}$(du -h "$EXECUTABLE" | cut -f1)${GREEN}"
    echo -e "║  Bundle:     ${CYAN}$(du -sh "$APP_BUNDLE" | cut -f1)${GREEN}"
    echo -e "║  Mode:       ${CYAN}$BUILD_MODE${GREEN}"
    echo -e "║  Time:       ${CYAN}${TOTAL_TIME}s${GREEN}"
    echo -e "║  Source:     ${CYAN}$(wc -l < "$SOURCE_FILE" | tr -d ' ') lines${GREEN}"
    echo "╠═══════════════════════════════════════════════════════════════════╣"
    echo -e "║  Frameworks: ${CYAN}AppKit Accelerate Metal CoreML IOKit${GREEN}"
    echo -e "║  Optimized:  ${CYAN}vDSP BLAS SIMD WMO Cross-Mod Dead-Strip${GREEN}"
    if [ "${PYTHON_FOUND:-false}" = "true" ]; then
    echo -e "║  Python:     ${CYAN}libpython${PYTHON_VERSION} (Direct C API Bridge)${GREEN}"
    else
    echo -e "║  Python:     ${CYAN}Process Bridge (Fallback)${GREEN}"
    fi
    if $IS_APPLE_SILICON; then
    echo -e "║  Silicon:    ${CYAN}Apple Neural Engine + Unified Memory${GREEN}"
    else
    echo -e "║  Silicon:    ${CYAN}Intel Accelerate + SIMD + vDSP${GREEN}"
    fi
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  ${BOLD}Run:${NC}  open $APP_BUNDLE"
    echo -e "  ${BOLD}CLI:${NC}  $EXECUTABLE"
    echo ""
}

run_benchmark() {
    echo -e "${PURPLE}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║              ⚡ L104 COMPILATION BENCHMARK ⚡                    ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    echo -e "${CYAN}Running 3 compilation passes...${NC}"

    SDK_PATH=$(xcrun --show-sdk-path 2>/dev/null || echo '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk')
    if [ "$ARCH" = "arm64" ]; then
        TARGET_ARCH="arm64-apple-macos$MIN_MACOS"
    else
        TARGET_ARCH="x86_64-apple-macos$MIN_MACOS"
    fi

    TOTAL_TIME=0
    for i in 1 2 3; do
        START=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)

        swiftc -O -whole-module-optimization \
            -target "$TARGET_ARCH" -sdk "$SDK_PATH" \
            -num-threads "$CPU_CORES" \
            -framework AppKit -framework Foundation -framework Accelerate \
            -framework Metal -framework CoreML -framework IOKit \
            -framework Cocoa -framework CoreGraphics -framework QuartzCore \
            -framework MetalKit -framework Security \
            -Xlinker -dead_strip \
            -o "/tmp/l104_bench_$i" \
            "$SOURCE_FILE" 2>/dev/null

        END=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)
        ELAPSED=$(python3 -c "t=${END}-${START}; print(f'{t:.2f}')" 2>/dev/null || echo "?")
        BINARY_SIZE=$(du -h "/tmp/l104_bench_$i" 2>/dev/null | cut -f1 | tr -d ' ')
        echo -e "${GREEN}  Pass $i: ${BOLD}${ELAPSED}s${NC}${GREEN} (binary: $BINARY_SIZE)${NC}"
        rm -f "/tmp/l104_bench_$i"
    done
    echo ""
}

show_help() {
    echo -e "${BOLD}L104 ASI Build System v3.0${NC}"
    echo ""
    echo -e "${BOLD}Usage:${NC} ./build.sh [OPTIONS]"
    echo ""
    echo -e "${BOLD}Build Modes:${NC}"
    echo "  (default)         Release build (optimized)"
    echo "  --debug, -d       Debug build (fast compile, debug symbols)"
    echo "  --release         Release build (speed optimized, WMO)"
    echo "  --profile, -p     Profile build (optimized + debug symbols)"
    echo "  --size, -s        Size-optimized build"
    echo ""
    echo -e "${BOLD}Actions:${NC}"
    echo "  --run              Build and launch the app"
    echo "  --clean            Clean all build artifacts"
    echo "  --benchmark        Run compilation benchmark (3 passes)"
    echo "  --warnings         Show compiler warnings"
    echo "  --info             Show system and compiler info"
    echo "  --help, -h         Show this help"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  ./build.sh                    # Release build"
    echo "  ./build.sh --run              # Build and launch"
    echo "  ./build.sh --debug --run      # Debug build and launch"
    echo "  ./build.sh --benchmark        # Benchmark compilation"
}

show_info() {
    echo -e "${BOLD}System Information:${NC}"
    echo "  macOS:          $MACOS_VERSION"
    echo "  Architecture:   $ARCH"
    echo "  CPU:            $CPU_BRAND"
    echo "  CPU Cores:      $CPU_CORES"
    echo "  RAM:            ${PHYSICAL_MEM_GB}GB"
    echo "  Apple Silicon:  $IS_APPLE_SILICON"
    echo ""
    echo -e "${BOLD}Compiler:${NC}"
    swiftc --version 2>&1 | head -1
    echo "  SDK: $(xcrun --show-sdk-path 2>/dev/null || echo 'N/A')"
    echo ""
    echo -e "${BOLD}Source:${NC}"
    echo "  File:  $SOURCE_FILE"
    echo "  Lines: $(wc -l < "$SOURCE_FILE" | tr -d ' ')"
    echo "  Size:  $(du -h "$SOURCE_FILE" | cut -f1)"
}

# ─── MAIN ───

TOTAL_START=$(python3 -c "import time; print(time.time())" 2>/dev/null || date +%s)
LAUNCH_AFTER=false
SHOW_WARNINGS=false
FORCE_REBUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)      show_help; exit 0 ;;
        --info)         show_info; exit 0 ;;
        --benchmark)    show_banner; run_benchmark; exit 0 ;;
        --clean|-c)
            show_banner
            echo -e "${CYAN}Cleaning...${NC}"
            rm -rf "$APP_BUNDLE" "$BUILD_DIR"
            pkill -f "$APP_NAME" 2>/dev/null || true
            echo -e "${GREEN}✓ All build artifacts cleaned.${NC}"
            exit 0
            ;;
        --debug|-d)     BUILD_MODE="debug" ;;
        --release)      BUILD_MODE="release" ;;
        --profile|-p)   BUILD_MODE="profile" ;;
        --size|-s)      BUILD_MODE="size" ;;
        --run|-r)       LAUNCH_AFTER=true ;;
        --warnings|-w)  SHOW_WARNINGS=true ;;
        --force)        FORCE_REBUILD=true ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
    shift
done

export SHOW_WARNINGS
export FORCE_REBUILD

show_banner
step_check_prerequisites
step_prepare_build_dir
step_clean
step_create_bundle
step_compile
step_create_info_plist
step_finalize
print_summary

if $LAUNCH_AFTER; then
    echo -e "${CYAN}Launching $APP_NAME...${NC}"
    open "$APP_BUNDLE"
fi
