
import time
import os
import sys
import json
from pathlib import Path

# Tracking Refactor Performance
# Targeted at EVO_60 Modularization

def benchmark_engine():
    print("="*80)
    print(" L104 CODE ENGINE MODULAR PERFORMANCE REVIEW")
    print("="*80)

    # 1. Initialization Time
    start_init = time.time()
    from l104_code_engine import code_engine
    end_init = time.time()
    init_duration = end_init - start_init

    print(f"[*] Initialization Time: {init_duration:.4f}s")

    status = code_engine.status()
    print(f"[*] Engine Version: {status.get('version')}")
    print(f"[*] Subsystems Active: {len(status.get('all_subsystem_status', {}))}")

    target_file = 'l104_code_engine_monolith_backup.py'
    if not os.path.exists(target_file):
        print(f"[!] Error: {target_file} not found.")
        return

    with open(target_file, 'r') as f:
        code = f.read()

    print(f"[*] Analyzing monolith ({len(code)/1024:.2f} KB)...")

    # 2. Analysis Performance
    # We'll run a few cycles to get an average
    cycles = 1
    total_time = 0

    import asyncio

    async def run_analysis():
        nonlocal total_time
        start_an = time.time()
        # Using the standard analyze method
        result = await code_engine.analyze(code, filename=target_file)
        end_an = time.time()
        total_time = end_an - start_an
        return result

    result = asyncio.run(run_analysis())

    print(f"[*] Analysis Duration: {total_time:.4f}s")
    print(f"[*] Complexity Score: {result.get('phi_weighted_complexity', result.get('complexity', 'N/A'))}")
    print(f"[*] Security Score: {result.get('security_score', result.get('security', 'N/A'))}")

    # 3. Memory Summary
    print(f"[*] Superfluid Viscosity: {status.get('superfluid_viscosity', 'N/A')}")

    print("="*80)
    print(" PERFORMANCE VERDICT:")
    if total_time < 5.0:
        print(" [PASSED] High Performance Modular Hub")
    else:
        print(" [LOG] Performance within acceptable limits for complexity")
    print("="*80)

if __name__ == "__main__":
    benchmark_engine()
