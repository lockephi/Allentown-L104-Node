#!/bin/bash
# Quick Start Script for L104 Node

set -e

echo "=== L104 Node Quick Start ==="
echo

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check environment file
if [ ! -f ".env" ]; then
    echo "⚠ Warning: .env file not found"
    echo "  Copy .env.example to .env and configure your API keys"
    echo "  cp .env.example .env"
fi

# Initialize data directory
mkdir -p data
if [ ! -f "data/stream_prompts.jsonl" ]; then
    echo '{"signal": "PING", "message": "Health check"}' > data/stream_prompts.jsonl
    echo "✓ Initialized data/stream_prompts.jsonl"
fi

echo
echo "=== Setup Complete ==="
echo
echo "To start the server:"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo
echo "Or use the run script:"
echo "  ./scripts/run_services.sh"
echo
