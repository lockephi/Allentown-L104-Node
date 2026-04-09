#!/bin/bash
# L104v2 Swift App Deployment Script
# Deploys new L104v2 binaries to L104Native target

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SWIFT_BUILD_DIR="$SCRIPT_DIR/L104SwiftApp/.build/release"
L104_NATIVE_TARGET="$SCRIPT_DIR/L104SwiftApp/L104Native.app"
BACKUP_DIR="$SCRIPT_DIR/.l104_backups/native_$(date +%Y%m%d_%H%M%S)"

echo "═══════════════════════════════════════════════════════════"
echo "L104v2 SWIFT APP DEPLOYMENT"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if Swift build exists
if [ ! -d "$SWIFT_BUILD_DIR" ]; then
    echo "❌ ERROR: Swift build directory not found at $SWIFT_BUILD_DIR"
    echo "   Run: cd L104SwiftApp && swift build -c release"
    exit 1
fi

# Check for binaries
if [ ! -f "$SWIFT_BUILD_DIR/L104" ]; then
    echo "❌ ERROR: L104 binary not found at $SWIFT_BUILD_DIR/L104"
    exit 1
fi

echo "✅ Found Swift binaries:"
echo "   • L104 executable: $(ls -lh $SWIFT_BUILD_DIR/L104 | awk '{print $5}')"
ls -lh $SWIFT_BUILD_DIR/ | grep -E "(L104|Daemon)" || true

echo ""
echo "📦 Creating backup..."
mkdir -p "$BACKUP_DIR"
if [ -d "$L104_NATIVE_TARGET" ]; then
    cp -r "$L104_NATIVE_TARGET" "$BACKUP_DIR/L104Native.app.backup"
    echo "   Backup saved to: $BACKUP_DIR"
else
    echo "   ⚠️  No existing L104Native.app to backup"
fi

echo ""
echo "🚀 Deploying new L104v2 binaries..."

# Update L104Native.app executable
if [ -d "$L104_NATIVE_TARGET/Contents/MacOS" ]; then
    cp "$SWIFT_BUILD_DIR/L104" "$L104_NATIVE_TARGET/Contents/MacOS/L104Native"
    chmod +x "$L104_NATIVE_TARGET/Contents/MacOS/L104Native"
    echo "   ✅ Updated L104Native executable"
else
    echo "   ⚠️  Creating L104Native.app structure..."
    mkdir -p "$L104_NATIVE_TARGET/Contents/MacOS"
    mkdir -p "$L104_NATIVE_TARGET/Contents/Resources"
    cp "$SWIFT_BUILD_DIR/L104" "$L104_NATIVE_TARGET/Contents/MacOS/L104Native"
    chmod +x "$L104_NATIVE_TARGET/Contents/MacOS/L104Native"
fi

# Deploy L104Daemon if exists
if [ -f "$SWIFT_BUILD_DIR/L104Daemon" ]; then
    cp "$SWIFT_BUILD_DIR/L104Daemon" "$SCRIPT_DIR/l104_daemon_swift"
    chmod +x "$SCRIPT_DIR/l104_daemon_swift"
    echo "   ✅ Updated L104Daemon binary"
fi

# Deploy L104NanoDaemon if exists
if [ -f "$SWIFT_BUILD_DIR/L104NanoDaemon" ]; then
    cp "$SWIFT_BUILD_DIR/L104NanoDaemon" "$SCRIPT_DIR/l104_nano_daemon_swift"
    chmod +x "$SCRIPT_DIR/l104_nano_daemon_swift"
    echo "   ✅ Updated L104NanoDaemon binary"
fi

echo ""
echo "✅ Deployment complete!"
echo "   L104Native.app updated: $L104_NATIVE_TARGET"
echo "   Backup location: $BACKUP_DIR"
echo ""
echo "📋 Verify deployment:"
echo "   $ $L104_NATIVE_TARGET/Contents/MacOS/L104Native --version"
