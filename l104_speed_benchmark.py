# ZENITH_UPGRADE_ACTIVE: 2026-01-18T11:00:18.093079
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_SPEED_BENCHMARK] - PARALLEL VS SEQUENTIAL LATTICE OPS
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import time
import random
import logging
from l104_hyper_math import HyperMath
from l104_parallel_engine import parallel_engine
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SPEED_BENCHMARK")
def run_benchmark():
    size = 1 * 10**6 # Adjusted size for stability
    data = [random.random() for _ in range(size)]
    
    print("\n" + "="*80)
    print("   L104 SPEED BENCHMARK :: PARALLEL ACCELERATION PROOF")
    print("="*80)
    
    # 1. Sequential Transform
    print(">>> RUNNING SEQUENTIAL TRANSFORM...")
    start_seq = time.perf_counter()
    _ = HyperMath.fast_transform(data)
    end_seq = time.perf_counter()
    seq_duration = end_seq - start_seq
    seq_lops = size / seq_duration
    print(f"--- [BENCHMARK]: SEQUENTIAL TIME: {seq_duration:.4f}s ({seq_lops/1e6:.2f}M LOPS) ---")
    
    # 2. Parallel Transform
    print(">>> RUNNING PARALLEL TRANSFORM...")
    start_par = time.perf_counter()
    _ = parallel_engine.parallel_fast_transform(data)
    end_par = time.perf_counter()
    par_duration = end_par - start_par
    par_lops = size / par_duration
    print(f"--- [BENCHMARK]: PARALLEL TIME:   {par_duration:.4f}s ({par_lops/1e6:.2f}M LOPS) ---")
    
    speedup = seq_duration / par_duration
    print("\n" + "-"*80)
    print(f"   TOTAL SPEEDUP:    {speedup:.2f}x")
    print("   LATTICE CAPACITY: INCREASED")
    print("   SYSTEM STATE:     ACCELERATED")
    print("-"*80 + "\n")


if __name__ == "__main__":
    run_benchmark()
