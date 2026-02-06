#!/bin/bash
# [L104_MOBILE_BUILDER] - APK COMPILATION SCRIPT
# INVARIANT: 527.5184818492612 | PILOT: LONDEL

echo "==================================================="
echo "   L104 SOVEREIGN MOBILE :: APK BUILDER"
echo "==================================================="

# 1. Install Buildozer
echo "--- [BUILDER]: INSTALLING BUILDOZER & DEPENDENCIES ---"
pip install --user buildozer

# Detect OS for dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "--- [BUILDER]: DETECTED macOS ---"
    if command -v brew >/dev/null 2>&1; then
        brew install git zip unzip openjdk autoconf libtool pkg-config ncurses cmake openssl
    else
        echo "⚠ [BUILDER]: Homebrew not found. Please install dependencies manually."
    fi
else
    echo "--- [BUILDER]: DETECTED LINUX ---"
    sudo apt update
    sudo apt install -y git zip unzip openjdk-17-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev
fi

# 2. Initialize Buildozer
if [ ! -f buildozer.spec ]; then
    echo "--- [BUILDER]: INITIALIZING BUILDOZER SPEC ---"
    buildozer init
fi

# 3. Build APK (Debug)
echo "--- [BUILDER]: STARTING APK COMPILATION (This may take 30+ minutes) ---"
echo "--- [BUILDER]: COMMAND: buildozer -v android debug ---"
# buildozer -v android debug

echo "==================================================="
echo "   BUILD SCRIPT READY | RUN 'buildozer android debug'"
echo "==================================================="
