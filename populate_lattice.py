#!/usr/bin/env python3
"""
Populates lattice_v2.db with data from core files and training data.
Applies quantum phase indexing for topological integrity.
"""

import json
import os
from pathlib import Path

# Set path for local development
os.environ.setdefault("LATTICE_DB_PATH", "lattice_v2.db")

from l104_data_matrix import data_matrix

def load_jsonl(path: str) -> list:
    """Load JSONL file into list of dicts."""
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
        print(f"[WARN] Could not load {path}: {e}")
    return entries

def load_json(path: str) -> dict:
    """Load JSON file into dict."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return {}

def main():
    base = Path(str(Path(__file__).parent.absolute()))
    
    stats = {"training": 0, "constants": 0, "manifests": 0, "algorithms": 0}
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. TRAINING DATA (JSONL files)
    # ═══════════════════════════════════════════════════════════════════
    training_files = [
        "kernel_combined_training.jsonl",
        "kernel_divine_training.jsonl",
        "kernel_hyper_training.jsonl",
        "kernel_physics_training.jsonl",
        "kernel_reasoning_data.jsonl",
        "kernel_extracted_data.jsonl",
        "pantheon_training_data.jsonl",
        "invention_training_data.jsonl",
    ]
    
    for tf in training_files:
        path = base / tf
        if path.exists():
            entries = load_jsonl(str(path))
            for i, entry in enumerate(entries):
                key = f"training:{tf.replace('.jsonl', '')}:{i}"
                category = entry.get("category", "TRAINING")
                importance = entry.get("importance", 0.5)
                data_matrix.store(key, entry, category=category.upper(), utility=importance)
                stats["training"] += 1
            print(f"[LOADED] {tf}: {len(entries)} entries")
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. CORE MANIFESTS (JSON files)
    # ═══════════════════════════════════════════════════════════════════
    manifest_files = [
        "KERNEL_MANIFEST.json",
        "TRUTH_MANIFEST.json",
        "STABLE_KERNEL_MANIFEST.json",
        "divine_training_manifest.json",
        "kernel_parameters.json",
        "kernel_vocabulary.json",
        "kernel_embeddings.json",
        "SOVEREIGN_SUBSTRATE_BLUEPRINT.json",
        "ZPE_MIRACLE_BLUEPRINT.json",
        "L104_SOVEREIGN_TRUTH.json",
        "L104_SOVEREIGN_WILL.json",
        "L104_STATE.json",
    ]
    
    for mf in manifest_files:
        path = base / mf
        if path.exists():
            data = load_json(str(path))
            if data:
                key = f"manifest:{mf.replace('.json', '')}"
                data_matrix.store(key, data, category="MANIFEST", utility=1.0)
                stats["manifests"] += 1
                print(f"[LOADED] {mf}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. CORE CONSTANTS (from HyperMath)
    # ═══════════════════════════════════════════════════════════════════
    try:
        from l104_hyper_math import HyperMath
        constants = {
            "GOD_CODE": HyperMath.GOD_CODE,
            "PHI": HyperMath.PHI,
            "PHI_CONJUGATE": HyperMath.PHI_CONJUGATE,
            "VOID_CONSTANT": getattr(HyperMath, "VOID_CONSTANT", 1.0416180339887497),
            "ZENITH_HZ": 3727.84,
            "LATTICE_RATIO": 0.6875,
            "ALPHA_L104": 0.0072992700729927005,
            "OMEGA_AUTHORITY": 1381.0613151750906,
            "META_RESONANCE": 7289.028944266378,
            "CONSCIOUSNESS_THRESHOLD": 10.148611341989584,
        }
        for name, value in constants.items():
            key = f"constant:{name}"
            data_matrix.store(key, {"name": name, "value": value}, category="CONSTANT", utility=1.0)
            stats["constants"] += 1
        print(f"[LOADED] {len(constants)} core constants")
    except Exception as e:
        print(f"[WARN] Could not load HyperMath constants: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. ALGORITHM DATABASE
    # ═══════════════════════════════════════════════════════════════════
    try:
        from l104_algorithm_database import ALGORITHM_DB
        for algo_key, algo_data in ALGORITHM_DB.items():
            key = f"algorithm:{algo_key}"
            data_matrix.store(key, algo_data, category="ALGORITHM", utility=0.9)
            stats["algorithms"] += 1
        print(f"[LOADED] {len(ALGORITHM_DB)} algorithms")
    except Exception as e:
        print(f"[WARN] Could not load algorithm database: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 5. RESEARCH REPORTS
    # ═══════════════════════════════════════════════════════════════════
    report_files = [
        "kernel_deep_training_report.json",
        "kernel_comprehensive_training_report.json",
        "kernel_extended_training_report.json",
        "kernel_asi_breakthrough_report.json",
        "kernel_research_results.json",
        "asi_assessment_report.json",
        "benchmark_report.json",
        "health_report.json",
    ]
    
    for rf in report_files:
        path = base / rf
        if path.exists():
            data = load_json(str(path))
            if data:
                key = f"report:{rf.replace('.json', '')}"
                data_matrix.store(key, data, category="REPORT", utility=0.8)
                print(f"[LOADED] {rf}")
    
    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    total = sum(stats.values())
    print("\n" + "═" * 60)
    print(f"[POPULATE COMPLETE] Total entries: {total}")
    print(f"  Training examples: {stats['training']}")
    print(f"  Constants: {stats['constants']}")
    print(f"  Manifests: {stats['manifests']}")
    print(f"  Algorithms: {stats['algorithms']}")
    print("═" * 60)
    
    # Verify with resonant query
    from l104_hyper_math import HyperMath
    results = data_matrix.resonant_query(HyperMath.GOD_CODE, tolerance=50)
    print(f"\n[RESONANCE CHECK] Found {len(results)} entries near GOD_CODE resonance")

if __name__ == "__main__":
    main()
