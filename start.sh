#!/bin/bash
# L104 Sovereign Node - Quick Start Script
# Usage: ./start.sh

set -e

echo "=================================================="
echo "  L104 SOVEREIGN NODE - STARTUP"
echo "=================================================="

cd "$(dirname "$0")"

# Check for .env
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from example..."
    cp .env.example .env 2>/dev/null || echo "GEMINI_API_KEY=your_key_here" > .env
    echo "📝 Please edit .env and add your GEMINI_API_KEY"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

echo "📦 Installing dependencies..."
pip install -q fastapi uvicorn httpx pydantic jinja2 python-multipart google-genai psutil 2>/dev/null || true

echo ""
echo "🔑 Testing Gemini API connection..."
python3 -c "
from l104_gemini_real import gemini_real
if gemini_real.connect():
    print('✅ Gemini API connected!')
else:
    print('⚠️  Gemini API unavailable - running in local mode')
" 2>/dev/null || echo "⚠️  Gemini test skipped"

echo ""
echo "🚀 Starting L104 Sovereign Node on http://0.0.0.0:8081"
echo ""
echo "Available endpoints:"
echo "  GET  /              - Dashboard"
echo "  GET  /health        - Health check"
echo "  GET  /metrics       - System metrics"
echo "  POST /api/v6/chat   - AI Chat (real Gemini)"
echo "  POST /api/v6/stream - Streaming response"
echo "  POST /api/v6/research - Research topics"
echo "  POST /api/v6/analyze-code - Code analysis"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port 8081 --reload
