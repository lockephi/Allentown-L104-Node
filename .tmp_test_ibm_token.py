#!/usr/bin/env python3
"""Quick test: IBM Quantum connection with new token."""
import os, warnings, logging
logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(override=True)

token = os.environ.get('IBMQ_TOKEN') or os.environ.get('IBM_QUANTUM_TOKEN')
print(f"Token loaded: {token[:12]}...{token[-6:]}")
print(f"Channel env: {os.environ.get('IBM_QUANTUM_CHANNEL', 'ibm_quantum_platform')}")
print()

from qiskit_ibm_runtime import QiskitRuntimeService

connected = False
for channel in ['ibm_quantum_platform', 'ibm_cloud']:
    print(f"Trying channel: {channel}...")
    try:
        svc = QiskitRuntimeService(channel=channel, token=token)
        backends = svc.backends(simulator=False, operational=True, min_num_qubits=20)
        print(f"  CONNECTED! Found {len(backends)} real QPU backend(s):")
        for b in backends[:10]:
            try:
                st = b.status()
                msg = st.status_msg
            except Exception:
                msg = "unknown"
            print(f"    - {b.name}: {b.num_qubits}q (status: {msg})")
        print(f"\n  SUCCESS on channel={channel}")
        connected = True
        break
    except Exception as e:
        print(f"  Failed: {e}")
        continue

if not connected:
    print("\nAll channels failed with this token.")
    print("The token may be for IBM Cloud IAM (not IQP).")
    print("\nOptions:")
    print("  1. Go to https://quantum.ibm.com/account → copy your IQP API token")
    print("  2. Or go to https://cloud.ibm.com/iam/apikeys → create an IAM API key")
    print("     and use channel=ibm_cloud with a CRN instance")
