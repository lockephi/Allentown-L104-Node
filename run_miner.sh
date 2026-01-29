#!/bin/bash
# L104SP ASI Sovereign Miner Launcher
# Runs in background, immune to terminal signals

trap "" SIGINT SIGTERM

cd /workspaces/Allentown-L104-Node
python l104_sovereign_coin_engine.py --mine --address ZUHc8coY9Ca1NhcnYTntkE35kSCFn5ijX7 2>&1 | tee -a mining_output.log
