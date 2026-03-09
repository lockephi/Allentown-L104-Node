#!/usr/bin/env python3
"""Test L104 Quantum Runtime — activate IBM QPU from cold state."""
import os
os.environ.pop("IBM_QUANTUM_CHANNEL", None)

from dotenv import load_dotenv
load_dotenv(override=True)

print("=== L104 Quantum Runtime — IBM Cloud QPU Activation ===\n")

from l104_quantum_runtime import get_runtime
rt = get_runtime()

print("\n--- Activating IBM QPU from cold state ---")
rt.set_real_hardware(True)

status = rt.get_status()
print(f"\nRuntime Status:")
print(f"  Connected:     {status.get('connected', False)}")
print(f"  Exec Mode:     {status.get('execution_mode', 'unknown')}")
print(f"  Backend:       {status.get('default_backend', 'none')}")
print(f"  Real Hardware: {status.get('real_hardware', False)}")

backend_info = rt.get_backend_info()
print(f"\nBackend Info:")
print(f"  Name:          {backend_info.get('name', '?')}")
print(f"  Num Qubits:    {backend_info.get('num_qubits', '?')}")
print(f"  Is Real:       {backend_info.get('is_real', False)}")
print(f"  Queue:         {backend_info.get('pending_jobs', '?')}")
print(f"  Mode:          {backend_info.get('mode', '?')}")

if status.get('connected'):
    avail = rt.get_available_backends()
    print(f"\nAvailable QPU backends ({len(avail)}):")
    for b in avail:
        print(f"  - {b.get('name', '?')}: {b.get('num_qubits', '?')}q")
    print("\n=== IBM QPU CONNECTED SUCCESSFULLY ===")
else:
    print("\n=== Connection failed — check output above ===")
