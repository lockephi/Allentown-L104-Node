#!/usr/bin/env python3
"""Test imports for L104 codebase"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

modules_to_test = [
    "routers",
    "l104_asi",
    "l104_quantum_engine",
    "l104_evolution_engine",
    "l104_consciousness_engine",
    "l104_math_engine",
    "l104_numerical_engine",
    "l104_neural_engine",
    "l104_intellect",
    "l104_gate_engine",
    "l104_science_engine",
    "l104_god_code_simulator",
    "l104_audio_simulation",
    "l104_vqpu",
    "l104_quantum_gate_engine",
    "l104_simulator",
    "l104_code_engine",
    "l104_api",
    "l104_server",
    "l104_interfaces",
    "l104_unification",
    "l104_ml_engine",
]

print("Testing imports for L104 codebase...")
print("=" * 60)

successful = []
failed = []

for module_name in modules_to_test:
    try:
        module_path = module_name.replace(".", "/") + "/__init__.py"
        if os.path.exists(module_path):
            __import__(module_name)
            successful.append(module_name)
            print(f"✓ {module_name}")
        else:
            print(f"⚠ {module_name} (no __init__.py)")
            failed.append(f"{module_name} - missing __init__.py")
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        failed.append(f"{module_name} - {e}")
    except Exception as e:
        print(f"✗ {module_name}: {type(e).__name__}: {e}")
        failed.append(f"{module_name} - {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print(f"Successful: {len(successful)}/{len(modules_to_test)}")
print(f"Failed: {len(failed)}/{len(modules_to_test)}")

if failed:
    print("\nFailed imports:")
    for f in failed:
        print(f"  - {f}")

# Also test some specific files
print("\n" + "=" * 60)
print("Testing specific file imports...")

specific_files = [
    "routers/ai.py",
    "l104_asi/core.py",
    "l104_quantum_engine/brain.py",
    "l104_evolution_engine/evolution.py",
]

for file_path in specific_files:
    if os.path.exists(file_path):
        try:
            # Try to import by file path
            module_name = file_path.replace("/", ".").replace(".py", "")
            __import__(module_name)
            print(f"✓ {file_path}")
        except Exception as e:
            print(f"✗ {file_path}: {type(e).__name__}: {e}")
    else:
        print(f"⚠ {file_path} (file not found)")