#!/usr/bin/env python3
"""
Scan a sample of Python files across the whole workspace for errors.
"""
import sys
import time
import pathlib
import random
from typing import Dict, Any, List
from l104_three_engine_integration import get_three_engine

def scan_file(filepath: pathlib.Path, engine) -> Dict[str, Any]:
    """Scan a single Python file."""
    try:
        code = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        return {
            "file": str(filepath),
            "error": f"Failed to read file: {e}",
            "valid": False,
        }

    try:
        result = engine.cross_validate({"code": code})
    except Exception as e:
        return {
            "file": str(filepath),
            "error": f"Three-engine validation failed: {e}",
            "valid": False,
        }

    code_val = result.get("code_validation", {})
    science_val = result.get("science_validation", {})
    math_val = result.get("math_validation", {})

    return {
        "file": str(filepath),
        "valid": code_val.get("valid", False) and science_val.get("valid", False) and math_val.get("valid", False),
        "code_smells": code_val.get("smells", 0),
        "science_error": science_val.get("error"),
        "math_error": math_val.get("error"),
        "consensus": result.get("consensus", {}),
    }

def main():
    print("Loading three-engine unified...")
    te = get_three_engine()
    print("Engine loaded.")

    # Collect Python files across workspace, excluding certain directories
    workspace = pathlib.Path(".")
    exclude_dirs = {".git", "__pycache__", ".kernel_build", ".claude", ".devcontainer",
                    ".github", ".l104_circuits", ".l104_mailbox", ".l104_swift_backups",
                    ".quantum_storage", ".roo", ".soul_backups", ".soul_state",
                    ".unified_evolution", "BUILDS", "checkpoints", "config", "contracts",
                    "dashboard", "data", "demo_advanced_data", "deployment_configs",
                    "diagnostics", "docs", "elixir", "fine_tune_exports", "go", "k8s",
                    "kernel_archive", "kernel_cloud_state", "kubo", "l104_agent_system",
                    "logs", "models", "path", "quantum", "quantum_algorithms",
                    "requirements", "routers", "rust", "sage_diffusion_output",
                    "scripts", "skills", "sklearn", "sklearn_mock", "source", "src",
                    "supabase", "SwiftQuantum", "systems", "templates", "tests",
                    "training_data", "website", "wiki"}

    py_files = []
    for f in workspace.glob("**/*.py"):
        # Skip excluded directories
        if any(part in exclude_dirs for part in f.parts):
            continue
        # Skip hidden files
        if f.name.startswith("."):
            continue
        py_files.append(f)

    print(f"Total Python files found: {len(py_files)}")
    # Limit to 50 files for performance
    if len(py_files) > 50:
        print("Sampling 50 random files...")
        py_files = random.sample(py_files, 50)

    results = []
    for i, f in enumerate(py_files):
        print(f"[{i+1}/{len(py_files)}] Scanning {f}...")
        start = time.time()
        res = scan_file(f, te)
        elapsed = time.time() - start
        res["scan_time"] = elapsed
        results.append(res)
        if res.get("error"):
            print(f"  Error: {res['error']}")
        else:
            print(f"  Valid: {res['valid']}, Smells: {res.get('code_smells')}")

    # Summary
    print("\n=== WORKSPACE SCAN SUMMARY ===")
    total = len(results)
    valid = sum(1 for r in results if r.get("valid"))
    total_smells = sum(r.get("code_smells", 0) for r in results)
    errors = [r for r in results if r.get("error")]

    print(f"Total files scanned: {total}")
    print(f"Valid files: {valid}")
    print(f"Files with errors: {len(errors)}")
    print(f"Total code smells: {total_smells}")

    if errors:
        print("\n=== ERRORS ===")
        for err in errors:
            print(f"{err['file']}: {err['error']}")

    # Files with smells
    smell_files = [r for r in results if r.get("code_smells", 0) > 0]
    if smell_files:
        print("\n=== CODE SMELLS (first 20) ===")
        for r in smell_files[:20]:
            print(f"{r['file']}: {r['code_smells']} smells")

    # Save results
    import json
    output_path = pathlib.Path("workspace_scan_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()