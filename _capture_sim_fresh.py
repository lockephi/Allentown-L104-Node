#!/usr/bin/env python3
"""Capture fresh God Code Simulator data for pipeline optimization."""
import json, time, sys
sys.path.insert(0, ".")

t0 = time.time()
from l104_god_code_simulator import god_code_simulator

# 1. Run all 23 simulations
print("Running all simulations...")
report = god_code_simulator.run_all()
print(f"  Completed {len(report)} simulations in {time.time()-t0:.1f}s")

# Extract key metrics
sim_data = {}
for name, result in report.items():
    r = result if isinstance(result, dict) else {"value": result}
    sim_data[name] = {
        "fidelity": r.get("fidelity", r.get("value", 0)),
        "sacred_score": r.get("sacred_score", 0),
        "god_code_alignment": r.get("god_code_alignment", 0),
    }

# 2. Parametric sweep — dial_a
print("Running parametric sweep (dial_a)...")
try:
    sweep = god_code_simulator.parametric_sweep("dial_a", start=0, stop=8)
    sim_data["sweep_dial_a"] = sweep if isinstance(sweep, dict) else {"values": list(sweep) if hasattr(sweep, '__iter__') else sweep}
except Exception as e:
    sim_data["sweep_dial_a"] = {"error": str(e)}

# 3. Adaptive optimization
print("Running adaptive optimization...")
try:
    opt = god_code_simulator.adaptive_optimize(target_fidelity=0.999, nq=4, depth=4)
    sim_data["adaptive_opt"] = opt if isinstance(opt, dict) else {"result": str(opt)}
except Exception as e:
    sim_data["adaptive_opt"] = {"error": str(e)}

# 4. Noise resilience
print("Running noise resilience optimization...")
try:
    noise = god_code_simulator.optimize_noise_resilience(nq=4, noise_level=0.05)
    sim_data["noise_resilience"] = noise if isinstance(noise, dict) else {"result": str(noise)}
except Exception as e:
    sim_data["noise_resilience"] = {"error": str(e)}

total = time.time() - t0
sim_data["_meta"] = {"total_time_s": total, "n_simulations": len(report)}
print(f"Total: {total:.1f}s")

# Write output
with open("_sim_data_fresh.json", "w") as f:
    json.dump(sim_data, f, indent=2, default=str)
print("Saved to _sim_data_fresh.json")

# Print summary
for name, data in list(sim_data.items())[:10]:
    if name.startswith("_"): continue
    fid = data.get("fidelity", "?")
    sac = data.get("sacred_score", "?")
    print(f"  {name}: fidelity={fid}, sacred={sac}")
