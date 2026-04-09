#!/bin/bash
# L104 SourceKit-LSP Management Script
# Reduces CPU consumption by managing indexing

case "$1" in
  stop)
    echo "Stopping sourcekit-lsp processes..."
    pkill -9 sourcekit-lsp 2>/dev/null
    pkill -9 SourceKitService 2>/dev/null
    echo "Done."
    ;;
  start)
    echo "Starting sourcekit-lsp..."
    # Don't start - let IDE start on demand
    echo "SourceKit starts on demand by IDE"
    ;;
  status)
    echo "SourceKit processes:"
    ps aux | grep -i sourcekit | grep -v grep
    ;;
  config)
    echo "Current config:"
    cat ~/.sourcekit-lsp/config.json 2>/dev/null || echo "No config found"
    ;;
  *)
    echo "Usage: $0 {stop|start|status|config}"
    echo ""
    echo "Recommendations for reducing CPU:"
    echo "  1. Stop sourcekit when not coding: $0 stop"
    echo "  2. Split large files (>2000 lines) into smaller modules"
    echo "  3. Use 'release' configuration for faster indexing"
    echo "  4. Disable background indexing in config"
    ;;
esac
