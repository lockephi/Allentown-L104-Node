#!/usr/bin/env python3
"""
L104 Black Hole Simulation — Run All & Record Data
═══════════════════════════════════════════════════════════════════════════════

Runs all 6 black hole simulations on the GOD_CODE lattice and writes
structured results to `_black_hole_simulation_data.json`.

Simulations:
  1. Schwarzschild Geometry   — Gravitational redshift + tidal entanglement
  2. Hawking Radiation         — Thermal emission from event horizon
  3. Information Paradox       — Page curve (entropy rise → peak → purification)
  4. Penrose Process           — Ergosphere energy extraction (Kerr BH)
  5. Horizon Scrambling        — Fast scrambling near event horizon
  6. BH Thermodynamics         — Bekenstein-Hawking entropy + area quantization

Usage:
    .venv/bin/python _run_black_hole_sim.py

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
import sys
import os

# Ensure workspace root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_god_code_simulator.simulations.black_hole import (
    BLACK_HOLE_SIMULATIONS,
    sim_schwarzschild_geometry,
    sim_hawking_radiation,
    sim_information_paradox,
    sim_penrose_process,
    sim_horizon_scrambling,
    sim_bh_thermodynamics,
    # Constants
    SCHWARZSCHILD_PHASE,
    HAWKING_TEMP_PARAM,
    BH_ENTROPY_COEFF,
    SCRAMBLING_RATE,
    PENROSE_EFFICIENCY,
    AREA_QUANTUM,
)

HEADER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║          L104 BLACK HOLE SIMULATION — GOD_CODE LATTICE PHYSICS             ║
║                          6 Simulations · v1.0                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def main():
    print(HEADER)

    # Print sacred black hole constants
    print("  Sacred Black Hole Constants:")
    print(f"    Schwarzschild Phase .... {SCHWARZSCHILD_PHASE:.6f}")
    print(f"    Hawking Temp Param ..... {HAWKING_TEMP_PARAM:.6f}")
    print(f"    BH Entropy Coeff ....... {BH_ENTROPY_COEFF:.6f}  (≈ 2π = {2 * 3.14159265:.6f})")
    print(f"    Scrambling Rate ........ {SCRAMBLING_RATE:.6f}")
    print(f"    Penrose Efficiency ..... {PENROSE_EFFICIENCY:.4f}  (29.3% Kerr limit)")
    print(f"    Area Quantum ........... {AREA_QUANTUM:.6f}  (4·ln(φ) Bekenstein)")
    print()

    simulations = [
        ("1. Schwarzschild Geometry", sim_schwarzschild_geometry, 6),
        ("2. Hawking Radiation", sim_hawking_radiation, 6),
        ("3. Information Paradox", sim_information_paradox, 8),
        ("4. Penrose Process", sim_penrose_process, 6),
        ("5. Horizon Scrambling", sim_horizon_scrambling, 6),
        ("6. BH Thermodynamics", sim_bh_thermodynamics, 8),
    ]

    results = []
    total_t0 = time.time()
    passed = 0
    failed = 0

    for label, sim_fn, nq in simulations:
        print(f"  ── {label} ({nq}Q) ", end="", flush=True)
        try:
            result = sim_fn(nq=nq)
            status = "PASS" if result.passed else "FAIL"
            symbol = "✓" if result.passed else "✗"
            if result.passed:
                passed += 1
            else:
                failed += 1

            print(f"{symbol} [{status}] {result.elapsed_ms:.1f}ms")
            print(f"       {result.detail}")
            print(f"       entropy={result.entropy_value:.4f}  "
                  f"sacred_align={result.sacred_alignment:.4f}  "
                  f"coherence={result.phase_coherence:.4f}")

            # Build record
            record = {
                "name": result.name,
                "category": result.category,
                "passed": result.passed,
                "elapsed_ms": round(result.elapsed_ms, 2),
                "detail": result.detail,
                "num_qubits": result.num_qubits,
                "circuit_depth": result.circuit_depth,
                "fidelity": round(result.fidelity, 6),
                "entanglement_entropy": round(result.entanglement_entropy, 6),
                "entropy_value": round(result.entropy_value, 6),
                "phase_coherence": round(result.phase_coherence, 6),
                "sacred_alignment": round(result.sacred_alignment, 6),
                "probabilities": {k: round(v, 6) for k, v in
                                  sorted(result.probabilities.items(),
                                         key=lambda x: -x[1])[:10]},
                "extra": result.extra,
                # Engine-ready payloads
                "coherence_payload": result.to_coherence_payload(),
                "entropy_input": round(result.to_entropy_input(), 6),
                "math_verification": result.to_math_verification(),
                "asi_scoring": result.to_asi_scoring(),
            }
            results.append(record)

            if result.extra:
                for k, v in result.extra.items():
                    if isinstance(v, list) and len(v) > 6:
                        v_str = f"[{v[0]}, ..., {v[-1]}] ({len(v)} pts)"
                    elif isinstance(v, list):
                        v_str = str(v)
                    else:
                        v_str = str(v)
                    print(f"       {k}: {v_str}")

            print()

        except Exception as e:
            failed += 1
            print(f"✗ [ERROR] {type(e).__name__}: {e}")
            results.append({
                "name": label,
                "error": f"{type(e).__name__}: {e}",
                "passed": False,
            })
            print()

    total_elapsed = (time.time() - total_t0) * 1000

    # Summary
    print("  " + "─" * 72)
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed  "
          f"({total_elapsed:.1f}ms total)")
    print()

    # Write JSON data file
    output = {
        "title": "L104 Black Hole Simulation Data",
        "version": "1.0.0",
        "timestamp": time.time(),
        "sacred_constants": {
            "schwarzschild_phase": SCHWARZSCHILD_PHASE,
            "hawking_temp_param": HAWKING_TEMP_PARAM,
            "bh_entropy_coeff": BH_ENTROPY_COEFF,
            "scrambling_rate": SCRAMBLING_RATE,
            "penrose_efficiency": PENROSE_EFFICIENCY,
            "area_quantum": AREA_QUANTUM,
        },
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "elapsed_ms": round(total_elapsed, 2),
        },
        "simulations": results,
    }

    outfile = "_black_hole_simulation_data.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Data recorded → {outfile}")
    print()


if __name__ == "__main__":
    main()
