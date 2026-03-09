#!/usr/bin/env python3
"""Quick IBM Cloud QPU connection test."""
import os, warnings, logging
logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(override=True)

token = os.environ.get('IBMQ_TOKEN')
channel = os.environ.get('IBM_QUANTUM_CHANNEL', 'ibm_cloud')
print(f"Token: {token[:12]}...{token[-6:]}")
print(f"Channel: {channel}")

from qiskit_ibm_runtime import QiskitRuntimeService

svc = QiskitRuntimeService(channel=channel, token=token)
backends = svc.backends(simulator=False, operational=True, min_num_qubits=20)
print(f"\nCONNECTED — {len(backends)} real QPU backends:")
for b in backends:
    print(f"  - {b.name}")

# Pick least busy
least = svc.least_busy(simulator=False, operational=True, min_num_qubits=20)
print(f"\nLeast busy: {least.name}")

# Quick Bell state test
from l104_quantum_gate_engine import GateCircuit as QuantumCircuit
generate_preset_pass_manager = None  # Use l104_quantum_gate_engine.GateCompiler
from qiskit_ibm_runtime import SamplerV2

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

pm = generate_preset_pass_manager(optimization_level=1, backend=least)
isa_qc = pm.run(qc)

print(f"\nSubmitting Bell state to {least.name} (Open plan - job mode)...")
sampler = SamplerV2(mode=least)
job = sampler.run([isa_qc], shots=100)
print(f"Job ID: {job.job_id()}")
print("Waiting for result (may take a few minutes in queue)...")
result = job.result()
counts = result[0].data.meas.get_counts()
print(f"Bell state counts: {counts}")
print("\n=== REAL QPU EXECUTION SUCCESS ===")
