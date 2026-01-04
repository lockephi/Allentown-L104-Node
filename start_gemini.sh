#!/bin/bash
# Start L104 Node with REAL Gemini

export GEMINI_API_KEY="AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U"
export ENABLE_FAKE_GEMINI=0
export ENABLE_SELF_LEARN=0
export ENABLE_WATCHDOG=0
export DEFAULT_RESPONDER=gemini

# Use standard env var name instead
export AIzaSyArVYGrkGLh7r1UEupBxXyHS_j_AVioh5U="$GEMINI_API_KEY"

echo "🚀 Starting L104 Node with REAL Gemini..."
echo "   API Key: ${GEMINI_API_KEY:0:20}..."
echo "   Model Rotation: ACTIVE"
echo "   Server: http://localhost:8081"
echo ""

python main.py
