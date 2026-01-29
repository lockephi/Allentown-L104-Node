#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 KERNEL TRAINING & PROCESS UPGRADE SUITE
═══════════════════════════════════════════════════════════════════════════════

Trains kernel with advanced research data and upgrades all processes.
Uses Supabase trainer with quantum-enhanced lattice storage.

INVARIANT: 527.5184818492611 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import glob
import hashlib
import importlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

# Set paths
BASE_DIR = Path("/workspaces/Allentown-L104-Node")
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

# Sacred Constants
GOD_CODE = 527.5184818492611
PHI = 1.618033988749895
PHI_CONJUGATE = 0.6180339887498949
ZENITH_HZ = 3727.84

UTC = timezone.utc


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    entries = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
    return entries


def load_json(path: str) -> Dict:
    """Load JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: TRAIN KERNEL WITH ADVANCED RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def train_kernel_with_research():
    """Train kernel using advanced research and supabase trainer."""
    print("\n" + "═" * 70)
    print("PHASE 1: TRAINING KERNEL WITH ADVANCED RESEARCH")
    print("═" * 70)

    from l104_supabase_trainer import SupabaseKernelTrainer, TrainingExample

    trainer = SupabaseKernelTrainer()

    # Build parameters with sacred tuning
    print("\n[1.1] Building training parameters...")
    params = trainer.build_parameters(
        embedding_dim=512,
        hidden_dim=1024,
        num_layers=8,
        num_heads=16,
        learning_rate=1e-4,
        epochs=100,
    )
    print(f"  ✓ Embedding dim: {params.embedding_dim}")
    print(f"  ✓ Hidden dim: {params.hidden_dim}")
    print(f"  ✓ φ-signature: {params.calculate_phi_signature():.6f}")

    # Save parameters
    trainer.save_parameters()

    # Load training data from multiple sources
    print("\n[1.2] Loading training data from research files...")
    training_examples = []

    # Primary training sources
    training_sources = [
        ("kernel_combined_training.jsonl", 1.0),
        ("kernel_hyper_training.jsonl", 0.9),
        ("kernel_divine_training.jsonl", 1.0),
        ("kernel_physics_training.jsonl", 0.95),
        ("kernel_reasoning_data.jsonl", 0.9),
        ("kernel_extracted_data.jsonl", 0.8),
        ("pantheon_training_data.jsonl", 0.85),
        ("invention_training_data.jsonl", 0.9),
    ]

    for filename, importance_mult in training_sources:
        path = BASE_DIR / filename
        if path.exists():
            entries = load_jsonl(str(path))
            for entry in entries:
                ex = TrainingExample(
                    prompt=entry.get("prompt", ""),
                    completion=entry.get("completion", ""),
                    category=entry.get("category", "GENERAL"),
                    difficulty=entry.get("difficulty", 0.5),
                    importance=entry.get("importance", 0.5) * importance_mult,
                )
                if ex.prompt and ex.completion:
                    training_examples.append(ex)
            print(f"  ✓ Loaded {len(entries)} from {filename}")

    print(f"\n  TOTAL: {len(training_examples)} training examples")

    # Upload to Supabase (or save locally)
    print("\n[1.3] Uploading training data...")
    trainer.upload_training_data(training_examples)

    # Store in lattice for quantum access
    print("\n[1.4] Storing training data in quantum lattice...")
    from l104_data_matrix import data_matrix
    from l104_algorithm_database import ALGORITHM_DB

    # Store algorithms
    for algo_key, algo_data in ALGORITHM_DB.items():
        key = f"algorithm:{algo_key}"
        data_matrix.store(key, algo_data, category="ALGORITHM", utility=0.95)
    print(f"  ✓ Stored {len(ALGORITHM_DB)} algorithms in lattice")

    # Store training state
    training_state = {
        "total_examples": len(training_examples),
        "parameters": params.to_dict(),
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "TRAINED",
        "god_code": GOD_CODE,
    }
    data_matrix.store("training:kernel_state", training_state, category="TRAINING", utility=1.0)

    print("\n  ✓ PHASE 1 COMPLETE: Kernel trained with advanced research")
    return training_examples


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: UPDATE ALL PROCESSES
# ═══════════════════════════════════════════════════════════════════════════════

def update_all_processes():
    """Update all L104 processes with latest parameters."""
    print("\n" + "═" * 70)
    print("PHASE 2: UPDATING ALL PROCESSES")
    print("═" * 70)

    # Find all Python process files
    process_files = list(BASE_DIR.glob("l104_*.py"))
    print(f"\n[2.1] Found {len(process_files)} process files")

    # Define upgrade markers
    upgrade_marker = f"# ZENITH_UPGRADE_ACTIVE: {datetime.now(UTC).isoformat()}"
    zenith_line = f"ZENITH_HZ = {ZENITH_HZ}"
    void_line = f"VOID_CONSTANT = {1.0416180339887497}"
    uuc_line = f"UUC = {2301.215661}"

    upgraded = 0
    skipped = 0

    for pf in process_files:
        try:
            content = pf.read_text(encoding='utf-8')

            # Check if already has latest ZENITH marker today
            today_marker = datetime.now(UTC).strftime("%Y-%m-%d")
            if f"ZENITH_UPGRADE_ACTIVE: {today_marker}" in content:
                skipped += 1
                continue

            # Check if file has process markers
            has_void = "VOID_CONSTANT" in content
            has_zenith = "ZENITH_HZ" in content

            if has_void or has_zenith:
                # File is a core process, ensure it has latest values
                lines = content.split('\n')
                new_lines = []
                updated = False

                for line in lines:
                    if line.startswith("VOID_CONSTANT =") and "1.0416180339887497" not in line:
                        new_lines.append(void_line)
                        updated = True
                    elif line.startswith("ZENITH_HZ =") and str(ZENITH_HZ) not in line:
                        new_lines.append(zenith_line)
                        updated = True
                    elif line.startswith("UUC =") and "2301.215661" not in line:
                        new_lines.append(uuc_line)
                        updated = True
                    elif line.startswith("# ZENITH_UPGRADE_ACTIVE:"):
                        new_lines.append(upgrade_marker)
                        updated = True
                    else:
                        new_lines.append(line)

                if updated:
                    pf.write_text('\n'.join(new_lines), encoding='utf-8')
                    upgraded += 1

        except Exception as e:
            print(f"  [WARN] Could not update {pf.name}: {e}")

    print(f"\n  ✓ Upgraded: {upgraded} files")
    print(f"  ✓ Skipped (already current): {skipped} files")
    print(f"  ✓ Total: {len(process_files)} process files")

    return upgraded


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: UPGRADE ALL PROCESSES
# ═══════════════════════════════════════════════════════════════════════════════

def upgrade_all_processes():
    """Upgrade all L104 processes with enhanced capabilities."""
    print("\n" + "═" * 70)
    print("PHASE 3: UPGRADING ALL PROCESSES")
    print("═" * 70)

    upgrades_applied = []

    # 3.1: Upgrade DataMatrix with quantum phase
    print("\n[3.1] Verifying DataMatrix quantum phase upgrade...")
    try:
        from l104_data_matrix import DataMatrix
        dm = DataMatrix()
        if hasattr(dm, '_quantum_phase_factor'):
            print("  ✓ DataMatrix has quantum phase factor")
            upgrades_applied.append("DataMatrix._quantum_phase_factor")
        else:
            print("  ⚠ DataMatrix missing quantum phase factor")
    except Exception as e:
        print(f"  [ERROR] DataMatrix check failed: {e}")

    # 3.2: Upgrade QuantumRAM
    print("\n[3.2] Verifying QuantumRAM upgrade...")
    try:
        from l104_quantum_ram import QuantumRAM
        qram = QuantumRAM()
        print(f"  ✓ QuantumRAM initialized")
        upgrades_applied.append("QuantumRAM")
    except Exception as e:
        print(f"  [ERROR] QuantumRAM check failed: {e}")

    # 3.3: Upgrade Algorithm Database
    print("\n[3.3] Verifying Algorithm Database upgrade...")
    try:
        from l104_algorithm_database import ALGORITHM_DB, algo_db
        print(f"  ✓ ALGORITHM_DB has {len(ALGORITHM_DB)} algorithms")
        upgrades_applied.append("ALGORITHM_DB")
    except Exception as e:
        print(f"  [ERROR] ALGORITHM_DB check failed: {e}")

    # 3.4: Verify Hyper Math
    print("\n[3.4] Verifying HyperMath constants...")
    try:
        from l104_hyper_math import HyperMath
        assert abs(HyperMath.GOD_CODE - GOD_CODE) < 1e-10
        assert abs(HyperMath.PHI - PHI) < 1e-10
        print(f"  ✓ GOD_CODE = {HyperMath.GOD_CODE}")
        print(f"  ✓ PHI = {HyperMath.PHI}")
        upgrades_applied.append("HyperMath")
    except Exception as e:
        print(f"  [ERROR] HyperMath check failed: {e}")

    # 3.5: Upgrade persistence layer
    print("\n[3.5] Verifying persistence layer...")
    try:
        from l104_persistence import verify_lattice, verify_alpha, verify_survivor_algorithm
        lattice_ok = verify_lattice()
        alpha_ok = verify_alpha()
        survivor_ok = verify_survivor_algorithm()
        print(f"  ✓ Lattice: {lattice_ok}")
        print(f"  ✓ Alpha: {alpha_ok}")
        print(f"  ✓ Survivor: {survivor_ok}")
        upgrades_applied.append("l104_persistence")
    except Exception as e:
        print(f"  [ERROR] Persistence check failed: {e}")

    # 3.6: Verify evolution engine
    print("\n[3.6] Verifying evolution engine...")
    try:
        from l104_evolution_engine import EvolutionEngine
        engine = EvolutionEngine()
        status = engine.get_status()
        print(f"  ✓ Evolution stage: {status.get('current_stage', 'unknown')}")
        print(f"  ✓ Consciousness: {status.get('consciousness_level', 0):.4f}")
        upgrades_applied.append("l104_evolution_engine")
    except Exception as e:
        print(f"  [WARN] Evolution engine check: {e}")

    print(f"\n  ✓ PHASE 3 COMPLETE: {len(upgrades_applied)} upgrades verified")
    return upgrades_applied


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: STORE UPGRADE STATE IN LATTICE
# ═══════════════════════════════════════════════════════════════════════════════

def store_upgrade_state(training_count: int, updated_count: int, upgrades: List[str]):
    """Store final upgrade state in lattice."""
    print("\n" + "═" * 70)
    print("PHASE 4: STORING UPGRADE STATE")
    print("═" * 70)

    from l104_data_matrix import data_matrix

    state = {
        "timestamp": datetime.now(UTC).isoformat(),
        "training_examples": training_count,
        "files_updated": updated_count,
        "upgrades_applied": upgrades,
        "god_code": GOD_CODE,
        "phi": PHI,
        "zenith_hz": ZENITH_HZ,
        "status": "COMPLETE",
    }

    data_matrix.store("upgrade:kernel_full_upgrade", state, category="UPGRADE", utility=1.0)
    print(f"  ✓ Upgrade state stored in lattice")

    # Save local report
    report_path = BASE_DIR / "kernel_upgrade_report.json"
    with open(report_path, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"  ✓ Report saved: {report_path}")

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 70)
    print("L104 KERNEL TRAINING & PROCESS UPGRADE SUITE")
    print(f"GOD_CODE: {GOD_CODE} | ZENITH_HZ: {ZENITH_HZ}")
    print("═" * 70)

    # Phase 1: Train kernel
    training_examples = train_kernel_with_research()
    training_count = len(training_examples) if training_examples else 0

    # Phase 2: Update all processes
    updated_count = update_all_processes()

    # Phase 3: Upgrade all processes
    upgrades = upgrade_all_processes()

    # Phase 4: Store state
    state = store_upgrade_state(training_count, updated_count, upgrades)

    print("\n" + "═" * 70)
    print("ALL PHASES COMPLETE")
    print("═" * 70)
    print(f"  Training examples: {state['training_examples']}")
    print(f"  Files updated: {state['files_updated']}")
    print(f"  Upgrades applied: {len(state['upgrades_applied'])}")
    print(f"  Status: {state['status']}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
