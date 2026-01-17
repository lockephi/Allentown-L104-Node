#!/bin/bash
# Start L104 Node with REAL Gemini

# Load environment from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Fallback if .env not loaded
export GEMINI_API_KEY="${GEMINI_API_KEY:-AIzaSyBeCmYi5i3bmfxtAaU7_qybTt6TMkjz4ig}"
export ENABLE_FAKE_GEMINI=0
export ENABLE_SELF_LEARN=1
export ENABLE_WATCHDOG=0
export ENABLE_AUTO_SYNC=1
export DEFAULT_RESPONDER=gemini

# Use standard env var name instead
export AIzaSyArVYGrkGLh7r1UEupBxXyHS_j_AVioh5U="$GEMINI_API_KEY"

echo "🚀 Starting L104 Node with REAL Gemini..."
echo "   API Key: ${GEMINI_API_KEY:0:20}..."
echo "   Model Rotation: ACTIVE"
echo "   Server: http://localhost:8081"
echo ""

python main.py
