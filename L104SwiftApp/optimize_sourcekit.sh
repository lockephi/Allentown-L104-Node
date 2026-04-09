#!/bin/bash
# L104 SourceKit-LSP Optimization Script
# Reduces CPU by optimizing indexing and clearing caches

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$1" in
  stop)
    echo "🛑 Stopping all sourcekit processes..."
    pkill -9 sourcekit-lsp 2>/dev/null
    pkill -9 SourceKitService 2>/dev/null
    pkill -9 swift-frontend 2>/dev/null
    echo "✓ Done"
    ;;

  clean)
    echo "🧹 Cleaning sourcekit caches..."
    rm -rf "$SCRIPT_DIR/.build/index" 2>/dev/null
    rm -rf ~/Library/Developer/Xcode/DerivedData/L104SovereignIntellect-*/Index.noindex 2>/dev/null
    rm -rf ~/Library/Caches/com.apple.dt.sourcekit-lsp 2>/dev/null
    echo "✓ Done"
    ;;

  status)
    echo "📊 SourceKit processes:"
    ps aux | grep -i sourcekit | grep -v grep | awk '{printf "  PID %s: CPU %.1f%%, MEM %s\n", $2, $3, $4}'

    echo ""
    echo "📁 Cache sizes:"
    if [ -d "$SCRIPT_DIR/.build/index" ]; then
      SIZE=$(du -sh "$SCRIPT_DIR/.build/index" 2>/dev/null | cut -f1)
      echo "  Index: $SIZE"
    fi
    ;;

  optimize)
    echo "⚡ Applying optimizations..."

    # Stop processes first
    $0 stop

    # Clean caches
    $0 clean

    # Limit concurrent jobs
    export SWIFTC_MAXIMUM_CONCURRENT_JOBS=1

    # Use release mode for faster indexing
    export SWIFT_BUILD_CONFIGURATION=release

    echo "✓ Optimizations applied"
    echo ""
    echo "To start coding, open Xcode or run: swift build -c release"
    ;;

  split-large-files)
    echo "✂️  Large files that should be split:"
    find "$SCRIPT_DIR/Sources" -name "*.swift" -exec wc -l {} \; 2>/dev/null | \
      awk '$1 > 2000 {print "  " $2 ": " $1 " lines"}' | sort -t: -k2 -rn | head -10

    echo ""
    echo "Run: python3 $SCRIPT_DIR/split_mainview.py --execute"
    echo "to split H11_MainView.swift"
    ;;

  *)
    echo "L104 SourceKit-LSP Optimization"
    echo ""
    echo "Usage: $0 {stop|clean|status|optimize|split-large-files}"
    echo ""
    echo "Commands:"
    echo "  stop              Stop all sourcekit processes"
    echo "  clean             Clear all caches"
    echo "  status            Show current state"
    echo "  optimize          Apply all optimizations"
    echo "  split-large-files Identify files that should be split"
    ;;
esac
