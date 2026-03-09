#!/usr/bin/env python3
"""Quick verification of the canonical GOD_CODE qubit."""

from l104_god_code_simulator.god_code_qubit import GOD_CODE_QUBIT, GOD_CODE_PHASE

print("═══ GOD_CODE QUBIT — Canonical Verification ═══")
print(GOD_CODE_QUBIT)
print()

# Run unitary verification
v = GOD_CODE_QUBIT.verify()
for k, val in v.items():
    if k == "decomposition":
        print("  decomposition:")
        for dk, dv in val.items():
            print(f"    {dk}: {dv}")
    elif k == "qpu":
        print(f"  qpu: job={val['job_id']}  fidelity={val['fidelity']}")
    else:
        print(f"  {k}: {val}")

print()
status = "PASS ✓" if v["PASS"] else "FAIL ✗"
print(f"OVERALL: {status}")

# Also verify imports from other packages work
print("\n═══ Cross-package imports ═══")
from l104_god_code_simulator import GOD_CODE_QUBIT as q1, GodCodeQubit
print(f"  god_code_simulator: {repr(q1)}")

from l104_god_code_simulator.quantum_primitives import GOD_CODE_GATE, GOD_CODE_QUBIT as q2
print(f"  quantum_primitives: GOD_CODE_GATE shape={GOD_CODE_GATE.shape}")
print(f"  quantum_primitives: GOD_CODE_QUBIT = {repr(q2)}")

from l104_quantum_gate_engine import GOD_CODE_QUBIT as q3
print(f"  quantum_gate_engine: {repr(q3)}")

from l104_science_engine import GOD_CODE_QUBIT as q4
print(f"  science_engine: {repr(q4)}")

# Verify phase consistency
from l104_quantum_gate_engine.constants import GOD_CODE_PHASE_ANGLE as qge_phase
from l104_god_code_simulator.constants import GOD_CODE_PHASE_ANGLE as sim_phase
from l104_science_engine.constants import GOD_CODE_PHASE as sci_phase

print("\n═══ Phase consistency ═══")
print(f"  god_code_qubit.GOD_CODE_PHASE:        {GOD_CODE_PHASE:.15f}")
print(f"  quantum_gate_engine.PHASE_ANGLE:      {qge_phase:.15f}")
print(f"  god_code_simulator.constants.PHASE:   {sim_phase:.15f}")
print(f"  science_engine.constants.GOD_CODE_PHASE: {sci_phase:.15f}")
match = (abs(GOD_CODE_PHASE - qge_phase) < 1e-12
         and abs(GOD_CODE_PHASE - sim_phase) < 1e-12
         and abs(GOD_CODE_PHASE - sci_phase) < 1e-12)
print(f"  All phases match: {match}")
print(f"\n{'═'*50}")
print(f"ALL CHECKS: {'PASS ✓' if v['PASS'] and match else 'FAIL ✗'}")
