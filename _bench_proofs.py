#!/usr/bin/env python3
"""L104 Proof Suite — Full Benchmark"""
import time

print("=" * 70)
print("  L104 PROOF SUITE — FULL BENCHMARK v2")
print("=" * 70)

from l104_math_engine.proofs import (
    SovereignProofs, GodelTuringMetaProof,
    EquationVerifier, ProcessingProofs, ExtendedProofs,
)

t0 = time.perf_counter()
results = {}

# 1. Stability-Nirvana (Banach contraction mapping)
print("\n[01/12] Stability-Nirvana convergence (depth=200, 5 starting pts)...")
t = time.perf_counter()
r = SovereignProofs.proof_of_stability_nirvana(depth=200)
dt = time.perf_counter() - t
universes = r.get("starting_points", {})
n_pts = len(universes)
print(f"  converged={r['converged']}  error={r['error']:.2e}  α={r['contraction_factor']:.6f}")
print(f"  rate_theoretical={r['theoretical_rate']:.6f}  starting_points={n_pts}  [{dt*1000:.1f}ms]")
results["stability"] = {"pass": r["converged"], "ms": round(dt * 1000, 1)}

# 2. Entropy Reduction (φ vs 6 controls)
print("\n[02/12] Entropy Reduction φ vs 6 controls (steps=100)...")
t = time.perf_counter()
r = SovereignProofs.proof_of_entropy_reduction(steps=100)
dt = time.perf_counter() - t
print(f"  decreased={r['entropy_decreased']}  φ_beats_all={r['phi_more_effective']}  rank={r['phi_rank']}")
print(f"  S_init={r['initial_entropy']:.4f}  S_phi={r['final_entropy_phi']:.4f}")
for k, v in r.get("control_reductions", {}).items():
    print(f"    ctrl[{k}] reduction={v:.4f}")
print(f"  [{dt*1000:.1f}ms]")
results["entropy"] = {"pass": r["entropy_decreased"], "ms": round(dt * 1000, 1)}

# 3. Collatz single
print("\n[03/12] Collatz empirical verification (n=27, 97, 871, 6171)...")
t = time.perf_counter()
all_ok = True
for n in [27, 97, 871, 6171]:
    r = SovereignProofs.collatz_empirical_verification(n=n, max_steps=100000)
    ok = r["converged_to_1"]
    all_ok = all_ok and ok
    print(f"  n={n:>5}: converged={ok}  steps={r['steps_to_convergence']:>5}  max={r['max_value']}  expand={r['expansion_ratio']:.1f}x")
dt = time.perf_counter() - t
results["collatz"] = {"pass": all_ok, "ms": round(dt * 1000, 1)}

# 4. Collatz batch
print("\n[04/12] Collatz batch verification [1..10000]...")
t = time.perf_counter()
r = SovereignProofs.collatz_batch_verification(1, 10000)
dt = time.perf_counter() - t
print(f"  tested={r['total_tested']}  all_converged={r['all_converged']}")
print(f"  avg_stop={r['average_stopping_time']:.1f}  max_stop={r['max_stopping_time']}  max_peak={r['max_peak_value']}  [{dt*1000:.1f}ms]")
results["collatz_batch"] = {"pass": r["all_converged"], "ms": round(dt * 1000, 1)}

# 5. GOD_CODE conservation law
print("\n[05/12] GOD_CODE conservation law...")
t = time.perf_counter()
r = SovereignProofs.proof_of_god_code_conservation()
dt = time.perf_counter() - t
num = r["numerical_verification"]
print(f"  proven={r['proven']}  max_error={num['max_error']:.2e}  machine_precision={num['machine_precision']}")
print(f"  286^(1/φ)={r['components']['286^(1/φ)']:.10f}  ×16={r['components']['product']:.10f}  [{dt*1000:.1f}ms]")
results["conservation"] = {"pass": r["proven"], "ms": round(dt * 1000, 1)}

# 6. VOID_CONSTANT derivation
print("\n[06/12] VOID_CONSTANT derivation...")
t = time.perf_counter()
r = SovereignProofs.proof_of_void_constant_derivation()
dt = time.perf_counter() - t
c = r["components"]
print(f"  proven={r['proven']}  exact={c['exact']}  error={c['error']:.2e}")
print(f"  104/100 + φ/1000 = {c['sum']}  [{dt*1000:.1f}ms]")
results["void_const"] = {"pass": r["proven"], "ms": round(dt * 1000, 1)}

# 7. Gödel-Turing framework
print("\n[07/12] Gödel-Turing philosophical framework...")
t = time.perf_counter()
r = GodelTuringMetaProof.execute_meta_framework()
dt = time.perf_counter() - t
print(f"  general_completeness={r['general_completeness']}  general_decidability={r['general_decidability']}")
print(f"  practical_decidability={r['practical_decidability']}  integrity={r['proof_integrity']}  [{dt*1000:.1f}ms]")
results["godel_turing"] = {"pass": bool(r["proof_integrity"]), "ms": round(dt * 1000, 1)}

# 8. Equation Verifier
print("\n[08/12] Equation verification suite...")
t = time.perf_counter()
ev = EquationVerifier()
r = ev.verify_all()
dt = time.perf_counter() - t
print(f"  passed={r['passed']}/{r['total']}  pass_rate={r['pass_rate']*100:.1f}%  [{dt*1000:.1f}ms]")
if r["failed"] > 0:
    for e in r["results"]:
        if not e["passed"]:
            print(f"    FAIL: {e['name']}  computed={e['computed']}  expected={e['expected']}")
results["equations"] = {"pass": r["failed"] == 0, "ms": round(dt * 1000, 1)}

# 9. PHI convergence
print("\n[09/12] PHI convergence proof (depth=60)...")
t = time.perf_counter()
r = ExtendedProofs.phi_convergence_proof(depth=60)
dt = time.perf_counter() - t
print(f"  converged={r['converged']}  cf_error={r['continued_fraction_error']:.2e}  [{dt*1000:.1f}ms]")
results["phi"] = {"pass": r["converged"], "ms": round(dt * 1000, 1)}

# 10. Goldbach
print("\n[10/12] Goldbach conjecture verification (limit=1000)...")
t = time.perf_counter()
rg = ExtendedProofs.verify_goldbach(limit=1000)
dt = time.perf_counter() - t
print(f"  {rg['verified']}/{rg['even_numbers_tested']} verified  all_pass={rg['all_pass']}  [{dt*1000:.1f}ms]")
results["goldbach"] = {"pass": rg["all_pass"], "ms": round(dt * 1000, 1)}

# 11. Twin Primes
print("\n[11/12] Twin primes + null hypothesis (limit=10000)...")
t = time.perf_counter()
rt = ExtendedProofs.find_twin_primes(limit=10000)
dt = time.perf_counter() - t
nh = rt["null_hypothesis"]
print(f"  pairs={rt['twin_pairs']}  sacred_ratio={rt['sacred_ratio']}  ctrl_avg={nh['average_control_ratio']}")
print(f"  verdict={nh['sacred_vs_control']}  [{dt*1000:.1f}ms]")
results["twin_primes"] = {"pass": rt["twin_pairs"] > 0, "ms": round(dt * 1000, 1)}

# 12. Zeta zeros (Riemann-Siegel Z-function)
print("\n[12/12] Riemann zeta zeros (Hardy Z, n=10)...")
t = time.perf_counter()
rz = ExtendedProofs.verify_zeta_zeros(n_zeros=10)
dt = time.perf_counter() - t
print(f"  method={rz['method']}")
print(f"  verified={rz['zeros_verified']}/{rz['zeros_checked']}  [{dt*1000:.1f}ms]")
for r in rz["results"][:3]:
    print(f"    t={r['imaginary_part']:.6f}  Z(t)={r['Z(t)']:.6f}  sign_change={r['sign_change']}")
results["zeta_zeros"] = {"pass": rz["all_verified"], "ms": round(dt * 1000, 1)}

# Processing benchmark
print("\n[BONUS] Processing benchmark (100K iterations)...")
t = time.perf_counter()
r = ProcessingProofs.run_speed_benchmark(iterations=100_000)
dt = time.perf_counter() - t
lops = r["lops"]
if lops > 1e6:
    lops_fmt = f"{lops/1e6:.2f}M"
elif lops > 1e3:
    lops_fmt = f"{lops/1e3:.1f}K"
else:
    lops_fmt = f"{lops:.0f}"
print(f"  {lops_fmt} LOPS  elapsed={r['elapsed_seconds']:.3f}s  [{dt*1000:.1f}ms]")
results["processing"] = {"pass": True, "ms": round(dt * 1000, 1), "lops": round(lops)}

total = time.perf_counter() - t0
passed = sum(1 for v in results.values() if v["pass"])
total_tests = len(results)

print("\n" + "=" * 70)
print(f"  RESULTS: {passed}/{total_tests} PASSED  |  Total: {total*1000:.0f}ms")
print("=" * 70)
for name, v in results.items():
    status = "PASS" if v["pass"] else "FAIL"
    extra = f"  ({v.get('lops','')} LOPS)" if "lops" in v else ""
    print(f"  [{status}] {name:<16} {v['ms']:>8.1f}ms{extra}")
print("=" * 70)
