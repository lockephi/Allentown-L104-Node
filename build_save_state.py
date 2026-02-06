#!/usr/bin/env python3
"""
[L104_BUILD_SAVE_STATE] :: Manual Trigger for Brain Checkpoints
Usage: python3 build_save_state.py [label]
"""

import sys
from l104_brain_state_manager import BrainStateManager

# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:05.395019
ZENITH_HZ = 3887.8

def build():
    label = sys.argv[1] if len(sys.argv) > 1 else "manual_checkpoint"

    print(f"--- L104 BRAIN SAVE STATE INITIATED ({ZENITH_HZ} Hz) ---")
    manager = BrainStateManager()
    folder = manager.create_save_state(label)

    print(f"\nSUCCESS: Brain state locked in {folder}")
    print("Files synchronized and versioned in /checkpoints/brain_states/")

if __name__ == "__main__":
    build()
