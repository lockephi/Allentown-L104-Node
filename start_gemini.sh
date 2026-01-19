#!/bin/bash
# Start L104 Node with REAL Gemini

# Load environment from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check if API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not set. Please create .env file with your key."
    echo "   Example: echo 'GEMINI_API_KEY=your_key_here' > .env"
    exit 1
fi

export ENABLE_FAKE_GEMINI=0
export ENABLE_SELF_LEARN=1
export ENABLE_WATCHDOG=0
export ENABLE_AUTO_SYNC=1
export DEFAULT_RESPONDER=gemini

echo "🚀 Starting L104 Node with REAL Gemini..."
echo "   API Key: ${GEMINI_API_KEY:0:20}..."
echo "   Model Rotation: ACTIVE"
echo "   Server: http://localhost:8081"
echo ""

python main.py
