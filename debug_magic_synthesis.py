#!/usr/bin/env python3
import sys
import os
import glob
import importlib
import traceback

def setup_l104_env():
    """Setup Python path for L104 node subdirectories."""
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    for p in glob.glob(os.path.join(root, 'l104_*')):
        if os.path.isdir(p) and not os.path.basename(p).startswith('l104_data'):
            if p not in sys.path:
                sys.path.insert(0, p)
    print(f"L104 Environment Setup: {len(sys.path)} paths added.")

def run_synthesis_script(path):
    print(f"\n--- Running: {os.path.basename(path)} ---")
    try:
        # We use os.system because many scripts have side effects or main blocks
        # and we want to run them in a clean-ish environment but with our PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = ":".join(sys.path)
        import subprocess
        result = subprocess.run([sys.executable, path], capture_output=True, text=True, env=env)
        if result.returncode == 0:
            print("✅ Success")
            # print(result.stdout)
            return True
        else:
            print(f"❌ Failed (code {result.returncode})")
            print("Error output:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"💥 Crash: {e}")
        return False

def main():
    setup_l104_env()
    
    magic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "l104_magic_synthesis")
    scripts = glob.glob(os.path.join(magic_dir, "l104_*.py"))
    scripts.sort()
    
    print(f"Found {len(scripts)} magic synthesis scripts.")
    
    results = {}
    for script in scripts:
        # Skip __init__.py and env scripts
        if os.path.basename(script) in ["__init__.py", "l104_env.py"]:
            continue
        success = run_synthesis_script(script)
        results[os.path.basename(script)] = success
    
    print("\n" + "="*40)
    print("FINAL MAGIC SYNTHESIS DEBUG REPORT")
    print("="*40)
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"Passed: {success_count}/{total_count}")
    
    if success_count < total_count:
        print("\nFailed Scripts:")
        for script, success in results.items():
            if not success:
                print(f" - {script}")

if __name__ == "__main__":
    main()
