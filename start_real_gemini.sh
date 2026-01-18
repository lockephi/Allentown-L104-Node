#!/bin/bash
# Ghost Protocol: Load API key from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set"
    exit 1
fi
export ENABLE_FAKE_GEMINI=0
export ENABLE_SELF_LEARN=0
export ENABLE_WATCHDOG=0
export ENABLE_AUTO_SYNC=1
export DEFAULT_RESPONDER=gemini
python main.py
