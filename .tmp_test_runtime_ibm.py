#!/usr/bin/env python3
"""Test L104 Quantum Runtime with new IBM Cloud token."""
import os
os.environ.pop("IBM_QUANTUM_CHANNEL", None)  # let runtime read from .env

from dotenv import load_dotenv
load_dotenv(override=True)

print("=== L104 Quantum Runtime — IBM Cloud Token Test ===\n")

from l104_quantum_runtime import get_runtime, ExecutionMode
rt = get_runtime()
status = rt.get_status()

print(f"\nRuntime Status:")
print(f"  Connected:     {status.get('connected', False)}")
print(f"  Exec Mode:     {status.get('execution_mode', 'unknown')}")
print(f"  Backend:       {status.get('default_backend', 'none')}")
print(f"  Real Hardware: {status.get('real_hardware', False)}")
print(f"  Num Qubits:    {status.get('num_qubits', 0)}")

backend_info = status.get("backend_info", {})
if backend_info:
    print(f"  Backend Info:  {backend_info.get('name', '?')} — {backend_info.get('num_qubits', '?')}q")
    print(f"  Is Real:       {backend_info.get('is_real', False)}")
    print(f"  Queue:         {backend_info.get('pending_jobs', '?')}")

print("\n=== DONE ===")
