#!/usr/bin/env python3
"""Run God Code Simulator simulations with timeout protection per sim."""

import signal
import time
import sys

from l104_god_code_simulator import god_code_simulator


def timeout_handler(signum, frame):
    raise TimeoutError("Simulation timed out")


def run_single(name, timeout_sec=20):
    """Run a single simulation with a timeout."""
    old = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = god_code_simulator.run(name)
        signal.alarm(0)
        return result
    except TimeoutError:
        signal.alarm(0)
        return None
    except Exception as e:
        signal.alarm(0)
        return e
    finally:
        signal.signal(signal.SIGALRM, old)


print("=" * 70)
print("  L104 GOD CODE SIMULATOR — Full Simulation Run")
print("=" * 70)

# Get simulation names from the catalog
all_names = god_code_simulator.catalog.list_all()
categories = sorted(god_code_simulator.catalog.categories)
print(f"  Registered: {len(all_names)} simulations across {len(categories)} categories")
print(f"  Categories: {', '.join(categories)}")

total_pass = 0
total_fail = 0
total_skip = 0
total_sims = 0

for cat in categories:
    names = god_code_simulator.catalog.list_by_category(cat)
    print(f"\n  ━━━ {cat.upper()} ({len(names)} sims) ━━━")

    for name in names:
        total_sims += 1
        sys.stdout.write(f"    {name:45s} ")
        sys.stdout.flush()

        t0 = time.time()
        r = run_single(name, timeout_sec=20)
        dt = (time.time() - t0) * 1000

        if r is None:
            total_skip += 1
            print(f"TIMEOUT ({dt:.0f}ms)")
        elif isinstance(r, Exception):
            total_fail += 1
            print(f"ERROR: {r}")
        else:
            if r.passed:
                total_pass += 1
                line = f"PASS  fidelity={r.fidelity:.4f}"
            else:
                total_fail += 1
                line = f"FAIL  fidelity={r.fidelity:.4f}"
            if r.god_code_measured > 0:
                line += f"  gc={r.god_code_measured:.4f}"
            if r.sacred_alignment > 0:
                line += f"  sacred={r.sacred_alignment:.4f}"
            if r.entanglement_entropy > 0:
                line += f"  ee={r.entanglement_entropy:.4f}"
            line += f"  ({dt:.0f}ms)"
            print(line)
            if not r.passed and r.detail:
                print(f"          -> {r.detail[:120]}")

print(f"\n{'=' * 70}")
print(f"  RESULTS: {total_sims} total | {total_pass} passed | {total_fail} failed | {total_skip} timeout")
print(f"  Pass Rate: {total_pass / max(total_sims - total_skip, 1) * 100:.1f}%")
print(f"{'=' * 70}")

# Parametric sweeps
print(f"\n{'=' * 70}")
print("  PARAMETRIC SWEEPS")
print(f"{'=' * 70}")

for sweep_name in ["dial_a", "noise", "depth"]:
    print(f"\n  --- {sweep_name.upper()} Sweep ---")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)
        if sweep_name == "dial_a":
            sweep = god_code_simulator.parametric_sweep(sweep_name, start=0, stop=8)
        else:
            sweep = god_code_simulator.parametric_sweep(sweep_name)
        signal.alarm(0)

        if isinstance(sweep, dict):
            for k, v in list(sweep.items())[:10]:
                print(f"    {k}: {v}")
        elif isinstance(sweep, list):
            for i, s in enumerate(sweep[:10]):
                if hasattr(s, "fidelity"):
                    print(f"    step {i}: fidelity={s.fidelity:.4f} passed={s.passed}")
                else:
                    print(f"    step {i}: {s}")
        else:
            print(f"    result: {sweep}")
    except TimeoutError:
        signal.alarm(0)
        print("    TIMEOUT")
    except Exception as e:
        signal.alarm(0)
        print(f"    Error: {e}")

# Adaptive optimization
print(f"\n{'=' * 70}")
print("  ADAPTIVE CIRCUIT OPTIMIZATION")
print(f"{'=' * 70}")
try:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)
    opt = god_code_simulator.adaptive_optimize(target_fidelity=0.99, nq=4, depth=4)
    signal.alarm(0)
    if isinstance(opt, dict):
        for k, v in opt.items():
            print(f"  {k}: {v}")
    else:
        print(f"  Result: {opt}")
except TimeoutError:
    signal.alarm(0)
    print("  TIMEOUT")
except Exception as e:
    signal.alarm(0)
    print(f"  Error: {e}")

print(f"\n{'=' * 70}")
print("  DONE")
print(f"{'=' * 70}")
