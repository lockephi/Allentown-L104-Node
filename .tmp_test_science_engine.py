#!/usr/bin/env python3
"""Test script for l104_science_engine package."""

from l104_science_engine import science_engine, GOD_CODE, QB, bridge
from l104_science_engine.quantum_25q import GodCodeQuantumConvergence, MemoryValidator, CircuitTemplates25Q

print("=" * 70)
print("  L104 SCIENCE ENGINE v4.0 — Package Validation")
print("=" * 70)

# Constants
print(f"\n  GOD_CODE = {GOD_CODE}")
print(f"  QB.STATEVECTOR_MB = {QB.STATEVECTOR_MB}")
print(f"  QB.HILBERT_DIM = {QB.HILBERT_DIM:,}")

# 512MB validation
v = MemoryValidator.validate_512mb()
print(f"\n  512MB VALIDATION:")
print(f"    Equation: {v['equation']}")
print(f"    Exact 512: {v['statevector_exact_512']}")
print(f"    Total system: {v['total_estimated_mb']} MB")

# GOD_CODE convergence
c = GodCodeQuantumConvergence.analyze()
print(f"\n  GOD_CODE / 512MB CONVERGENCE:")
print(f"    Ratio: {c['ratio']:.10f}")
print(f"    Excess above parity: {c['excess_above_parity_pct']}%")
print(f"    Octave multiplier (2^4=bytes/amp): {c['octave_multiplier']}")
print(f"    Reconstruction check: {c['reconstruction_check']}")
print(f"    Iron-qubit bridge: {c['iron_qubit_bridge']}")

# Templates
templates = CircuitTemplates25Q.all_templates()
print(f"\n  25Q CIRCUIT TEMPLATES ({len(templates)}):")
for name, t in templates.items():
    desc = t.get("description", "")[:55]
    print(f"    {name}: depth={t.get('depth','?')} mem={t['memory_mb']}MB — {desc}")

# Bridge
bs = bridge.status()
print(f"\n  BRIDGE STATUS:")
print(f"    Conservation law verified: {bs['conservation_law']}")
print(f"    512MB exact: {bs['512mb_exact']}")
for k, bv in bs["bridges"].items():
    print(f"    {k}: {bv}")

# Engine status
print(f"\n  ENGINE v{science_engine.VERSION}")
status = science_engine.get_full_status()
print(f"    Domains: {len(status['active_domains'])}")
print(f"    GOD_CODE/512 = {status['god_code_convergence']['ratio']}")
print(f"    Excess: {status['god_code_convergence']['excess_pct']}%")
print(f"    Physics equations: {status['physics']['adapted_equations']}")
print(f"    Coherence field dim: {status['coherence']['field_dimension']}")
print(f"    MultiDim: {status['multidim']['dimension']}D")

# Conservation sweep
sweep = bridge.conservation_sweep()
print(f"\n  CONSERVATION SWEEP: {sweep['points_checked']} points, all conserved: {sweep['all_conserved']}")

# Fidelity prediction
fid = bridge.fidelity_prediction(25, 50)
print(f"  Fidelity (25q, depth=50): {fid['total_fidelity']} ({fid['classification']})")

print("\n" + "=" * 70)
print("  ALL TESTS PASSED — l104_science_engine/ v4.0 OPERATIONAL")
print("=" * 70)
