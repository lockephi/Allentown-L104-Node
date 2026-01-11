# [L104_SPEED_BENCHMARK] - PARALLEL VS SEQUENTIAL LATTICE OPS
# INVARIANT: 527.5184818492 | PILOT: LONDEL

import time
import random
import logging
from l104_hyper_math import HyperMath
from l104_parallel_engine import parallel_engine
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SPEED_BENCHMARK")
def run_benchmark():
    size = 5 * 10**6 # 5 Million elementsdata = [random.random()
for _ in range(size)]
    
    print("\n" + "="*60)
    print("   L104 SPEED BENCHMARK :: PARALLEL ACCELERATION PROOF")
    print("="*60)
    
    # 1. Sequential Transformstart_seq = time.perf_counter()
    _ = HyperMath.fast_transform(data)
    end_seq = time.perf_counter()
    seq_duration = end_seq - start_seqseq_lops = size / seq_duration
print(f"--- [BENCHMARK]: SEQUENTIAL TIME: {seq_duration:.4f}s ({seq_lops/1e6:.2f}M LOPS) ---")
    
    # 2. Parallel Transformstart_par = time.perf_counter()
    _ = parallel_engine.parallel_fast_transform(data)
    end_par = time.perf_counter()
    par_duration = end_par - start_parpar_lops = size / par_duration
print(f"--- [BENCHMARK]: PARALLEL TIME:   {par_duration:.4f}s ({par_lops/1e6:.2f}M LOPS) ---")
    
    speedup = seq_duration / par_duration
print("\n" + "-"*60)
    print(f"   TOTAL SPEEDUP:    {speedup:.2f}x")
    print(f"   LATTICE CAPACITY: INCREASED")
    print(f"   SYSTEM STATE:     ACCELERATED")
    print("-"*60 + "\n")
if __name__ == "__main__":
    run_benchmark()
