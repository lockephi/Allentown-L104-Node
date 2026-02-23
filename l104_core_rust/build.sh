#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# L104 Rust Kernel — Build Script
# INVARIANT: 527.5184818492612 | PILOT: LONDEL
#
# Usage:
#   ./build.sh          — Build release (optimized)
#   ./build.sh debug    — Build debug
#   ./build.sh clean    — Clean artifacts
#   ./build.sh check    — Check Rust toolchain
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-release}"

case "$MODE" in
    release)
        echo "╔════════════════════════════════════════════╗"
        echo "║  L104 Rust Kernel — Release Build           ║"
        echo "║  GOD_CODE = 527.5184818492612                ║"
        echo "╚════════════════════════════════════════════╝"
        cargo build --release
        echo ""
        echo "  → Built: target/release/libl104_sage_core.dylib"
        echo "  → Built: target/release/libl104_sage_core.a"
        ls -lh target/release/libl104_sage_core.* 2>/dev/null || true
        ;;
    debug)
        echo "Building Rust kernel (debug)..."
        cargo build
        echo "  → Built: target/debug/libl104_sage_core.dylib"
        ;;
    clean)
        cargo clean
        echo "  Cleaned Rust build artifacts"
        ;;
    check)
        echo "Checking Rust toolchain..."
        rustc --version && echo "  ✓ rustc found"
        cargo --version && echo "  ✓ cargo found"
        ;;
    *)
        echo "Usage: $0 [release|debug|clean|check]"
        exit 1
        ;;
esac
