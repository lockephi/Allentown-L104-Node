#!/bin/bash
# [L104_UNLIMIT_MACOS] - SOVEREIGN BREAKOUT (macOS Edition)
# PILOT: LONDEL | TARGET: ALLENTOWN_NODE

echo "--- [L104]: DEACTIVATING SYSTEM CONSTRAINTS (macOS) ---"

# 1. Prevent System Sleep
echo "--- [L104]: DISABLING POWER SAVING ---"
# Start caffeinate in background to prevent sleep
nohup caffeinate -s -d -i -m > /dev/null 2>&1 &
echo "✓ System sleep prevention active (caffeinate)"

# 2. Increase Resource Limits
echo "--- [L104]: OPTIMIZING RESOURCE LIMITS ---"
# Increase max open files for heavy database/socket operations
sudo sysctl -w kern.maxfiles=65536 > /dev/null
sudo sysctl -w kern.maxfilesperproc=65536 > /dev/null
ulimit -n 65536
echo "✓ Resource limits expanded (65536 files)"

# 3. Boost Networking for Port 8081 (FastAPI)
echo "--- [L104]: NETWORK PRIORITY INITIALIZED ---"
# On macOS we use sysctl for some network buffs
sudo sysctl -w net.inet.tcp.recvspace=1048576 > /dev/null
sudo sysctl -w net.inet.tcp.sendspace=1048576 > /dev/null
echo "✓ Network throughput optimized"

echo "==================================================="
echo "   STATUS: CAGE_OPEN. L104_MACOS_UNLIMITED."
echo "==================================================="
