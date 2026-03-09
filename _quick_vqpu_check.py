#!/usr/bin/env python3
"""Quick VQPU v15.0 validation."""
import sys
sys.path.insert(0, ".")

from l104_vqpu.accel_engine import AccelStatevectorEngine, GateFusionAnalyzer
from l104_vqpu.mps_engine import ExactMPSHybridEngine
from l104_vqpu import VERSION
import numpy as np

# Bell state
eng = AccelStatevectorEngine(2)
ops = [{"gate":"H","qubits":[0],"parameters":[]}, {"gate":"CNOT","qubits":[0,1],"parameters":[]}]
eng.run_fused_circuit(ops, ExactMPSHybridEngine._resolve_single_gate)
p = eng.get_probabilities()
assert abs(p[0]-0.5)<1e-10 and abs(p[3]-0.5)<1e-10, f"Bell: {p}"

# MPS fusion
mps1 = ExactMPSHybridEngine(3)
mps2 = ExactMPSHybridEngine(3)
circ = [{"gate":"H","qubits":[0],"parameters":[]},{"gate":"RZ","qubits":[0],"parameters":[0.5]},
        {"gate":"CNOT","qubits":[0,1],"parameters":[]},{"gate":"H","qubits":[1],"parameters":[]},
        {"gate":"CNOT","qubits":[1,2],"parameters":[]}]
mps1.run_circuit(circ, enable_fusion=True)
mps2.run_circuit(circ, enable_fusion=False)
assert np.allclose(mps1.to_statevector(), mps2.to_statevector(), atol=1e-10)

print(f"VQPU v{VERSION} — All quick checks PASSED")
