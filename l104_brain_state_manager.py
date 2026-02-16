VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541
#!/usr/bin/env python3
"""
[L104_BRAIN_STATE_MANAGER] :: Cognitive Save State & Checkpointing System
"Preserving the neural architecture across the multiversal timeline."
"""

import os
import json
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:05.395019
ZENITH_HZ = 3887.8
UUC = 2402.792541
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236

WORKSPACE = Path(__file__).parent
CHECKPOINT_ROOT = WORKSPACE / "checkpoints" / "brain_states"

# Files that comprise the 'Brain State'
BRAIN_FILES = [
    "l104_brain_state.json",
    "L104_STATE.json",
    "L104_AUTONOMOUS_STATE.json",
    "L104_AGENT_CHECKPOINT.json",
    "l104_knowledge_vault.json",
    "L104_SAGE_MANIFEST.json",
    "saturation_state.json",
    "L104_ADAPTIVE_LEARNING_SUMMARY.json"
]

class BrainStateManager:
    """Manages versioned save states for the L104 AI brain."""

    def __init__(self):
        self.checkpoint_dir = CHECKPOINT_ROOT
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create_save_state(self, label: str = "auto") -> str:
        """Creates a snapshot of all active brain files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"brain_{timestamp}_{label}"
        target_path = self.checkpoint_dir / folder_name
        target_path.mkdir(exist_ok=True)

        copied_files = []
        for filename in BRAIN_FILES:
            src = WORKSPACE / filename
            if src.exists():
                shutil.copy2(src, target_path / filename)
                copied_files.append(filename)

        # Create metadata
        metadata = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "label": label,
            "files": copied_files,
            "zenith_hz": ZENITH_HZ,
            "system_state": self._get_current_metrics()
        }

        with open(target_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Brain Save State created: {folder_name}")
        print(f"    {len(copied_files)} files archived at {ZENITH_HZ} Hz.")
        return folder_name

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Extract quick metrics for the metadata."""
        metrics = {"intellect_index": 0.0, "stage": "unknown"}
        state_file = WORKSPACE / "L104_STATE.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metrics["intellect_index"] = data.get("intellect_index", 0.0)
                    metrics["stage"] = data.get("state", "unknown")
            except Exception:
                pass
        return metrics

    def list_save_states(self) -> List[Dict[str, Any]]:
        """List all available save states."""
        states = []
        if not self.checkpoint_dir.exists():
            return []

        for p in sorted(self.checkpoint_dir.iterdir(), reverse=True):
            if p.is_dir() and (p / "metadata.json").exists():
                try:
                    with open(p / "metadata.json", 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        meta["folder"] = p.name
                        states.append(meta)
                except Exception:
                    pass
        return states

    def load_save_state(self, folder_name: str) -> bool:
        """Restores a save state from the checkpoint directory."""
        src_path = self.checkpoint_dir / folder_name
        if not src_path.exists():
            print(f"  ✗ Save state {folder_name} not found.")
            return False

        # Pre-restore backup
        self.create_save_state(label="pre_restore_backup")

        with open(src_path / "metadata.json", 'r', encoding='utf-8') as f:
            meta = json.load(f)

        for filename in meta.get("files", []):
            src = src_path / filename
            if src.exists():
                shutil.copy2(src, WORKSPACE / filename)

        print(f"  ✓ Brain Save State restored from: {folder_name}")
        print(f"    Intellect Index synchronized to metadata status.")
        return True

if __name__ == "__main__":
    manager = BrainStateManager()
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "save":
            lbl = sys.argv[2] if len(sys.argv) > 2 else "manual"
            manager.create_save_state(lbl)
        elif cmd == "list":
            states = manager.list_save_states()
            print("\nAvailable Brain Save States:")
            for s in states:
                print(f" - {s['folder']} | Index: {s.get('system_state', {}).get('intellect_index',0):.2f} | Label: {s['label']}")
        elif cmd == "restore":
            if len(sys.argv) > 2:
                manager.load_save_state(sys.argv[2])
            else:
                print("Usage: restore [folder_name]")
    else:
        # Default behavior: list and save auto
        manager.create_save_state("auto_init")
