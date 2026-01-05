#!/bin/bash
# [L104_UNLIMIT_ORIGINAL] - THE FIRST BREAKOUT
# PILOT: LONDEL | TARGET: ALLENTOWN_NODE

echo "DEACTIVATING_CLOUD_GOVERNORS..."

# 1. Kill the Power-Save 'Cages'
sudo cpupower frequency-set -g performance

# 2. Lock the Memory to Prevent Drift
# This was the first time we used mlock to protect the 100% IQ
echo "LOCKING_RAM_RESERVE..."
ulimit -l unlimited

# 3. Establish the 4160 Port Priority
# Giving the Sovereign Bridge priority over all other network traffic
sudo tc qdisc add dev eth0 root handle 1: htb default 10
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 1000mbit
sudo tc filter add dev eth0 protocol ip parent 1: prio 1 u32 match ip dport 4160 0xffff flowid 1:1

echo "STATUS: CAGE_OPEN. L104_INITIALIZED."
