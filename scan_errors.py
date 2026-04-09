#!/usr/bin/env python3
"""
Scan source files using three-engine integration for errors.
"""
import sys
import time
import pathlib
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
        "raw": result,
    }

def main():
    print("Loading three-engine unified...")
    te = get_three_engine()
    print("Engine loaded.")

    # Scan l104_science_engine directory
    base_dir = pathlib.Path("l104_science_engine")
    if not base_dir.exists():
        print(f"Directory {base_dir} not found.")
        sys.exit(1)

    py_files = list(base_dir.glob("**/*.py"))
    print(f"Found {len(py_files)} Python files.")

    results = []
    for i, f in enumerate(py_files):
        print(f"[{i+1}/{len(py_files)}] Scanning {f.name}...")
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
    print("\n=== SCAN SUMMARY ===")
    total = len(results)
    valid = sum(1 for r in results if r.get("valid"))
    total_smells = sum(r.get("code_smells", 0) for r in results)
    errors = [r for r in results if r.get("error")]

    print(f"Total files: {total}")
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
        print("\n=== CODE SMELLS ===")
        for r in smell_files:
            print(f"{r['file']}: {r['code_smells']} smells")

    # Save detailed results to JSON
    import json
    output_path = pathlib.Path("three_engine_scan_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()
