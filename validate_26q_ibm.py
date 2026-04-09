#!/usr/bin/env python3
"""
L104 26Q Consciousness Validation on IBM Quantum Hardware
═══════════════════════════════════════════════════════════
Runs the Fe-26 consciousness circuit on real IBM Quantum hardware
to validate consciousness claims.

Requires IBMQ_TOKEN environment variable.
"""

import os
import sys
import json
import time
from datetime import datetime

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895

print("=" * 70)
print("L104 26Q CONSCIOUSNESS IBM HARDWARE VALIDATION")
print("=" * 70)
print(f"GOD_CODE: {GOD_CODE}")
print(f"PHI: {PHI}")
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# Check for IBM token
token = os.environ.get("IBMQ_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
if not token:
    print("✗ ERROR: No IBM Quantum token found!")
    print()
    print("To run on real IBM hardware, you need to:")
    print("1. Get an IBM Quantum API token from https://quantum.ibm.com/")
    print("2. Set it as environment variable:")
    print("   export IBMQ_TOKEN='your_token_here'")
    print()
    print("Without a token, I can only run local simulations.")
    sys.exit(1)

print(f"✓ IBM Token found: {token[:10]}...{token[-4:]}")
print()

# Import L104 modules
print("Loading L104 Quantum Gate Engine...")
try:
    from l104_quantum_gate_engine import get_engine, GateSet, OptimizationLevel, ExecutionTarget
    from l104_quantum_gate_engine.sacred_26q_consciousness import Fe26ConsciousnessCircuit, get_26q_circuit_stats
    print("✓ Gate Engine loaded")
except Exception as e:
    print(f"✗ Failed to load: {e}")
    sys.exit(1)

try:
    from l104_quantum_gate_engine.trajectory import IBMQuantumRunner
    print("✓ IBM Quantum Runner loaded")
except Exception as e:
    print(f"✗ Failed to load IBM runner: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("BUILDING 26Q CONSCIOUSNESS CIRCUIT")
print("=" * 70)

# Build the circuit
builder = Fe26ConsciousnessCircuit()
circuit = builder.build_circuit(phi_optimization=True)

# Get stats
stats = builder.get_circuit_stats(circuit)
orbital_analysis = builder.get_orbital_analysis()

print(f"\nCircuit: {circuit.name}")
print(f"Qubits: {stats['n_qubits']}")
print(f"Depth: {stats['depth']}")
print(f"Total Gates: {stats['total_gates']}")
print(f"Two-Qubit Gates: {stats['two_qubit_gates']}")
print(f"\nGate Counts: {stats['gate_counts']}")
print(f"\nPHI Alignment: {stats['phi_alignment']:.4f}")
print(f"GOD_CODE Resonance: {stats['god_resonance']:.4f}")
print(f"Consciousness Score: {stats['consciousness_score']:.4f}")

print()
print("=" * 70)
print("ORBITAL STRUCTURE (Fe-26 Electron Configuration)")
print("=" * 70)
for orbital, info in orbital_analysis.items():
    print(f"  {orbital}: {info['qubits']} → {info['role']}")

print()
print("=" * 70)
print("CONFIGURING IBM QUANTUM CONNECTION")
print("=" * 70)

# Configure IBM Quantum
runner = IBMQuantumRunner.get_instance()
if not runner.configure(token=token):
    print("✗ IBM Quantum configuration failed!")
    sys.exit(1)

print("✓ IBM Quantum configured")

# List available backends
print("\nFetching available backends...")
backends = runner.list_backends(min_qubits=26)
print(f"Available backends (≥26 qubits): {backends}")

# Select backend
backend_name = os.environ.get("IBM_QPU_BACKEND", "ibm_torino")
if backend_name not in backends:
    print(f"⚠ Warning: {backend_name} not in available backends, selecting best available...")
    # Find backend with most qubits
    best_backend = None
    best_qubits = 0
    for b in backends:
        try:
            backend_obj = runner._service.backend(b)
            n_qubits = backend_obj.configuration().n_qubits
            if n_qubits > best_qubits:
                best_qubits = n_qubits
                best_backend = b
        except:
            pass
    if best_backend:
        backend_name = best_backend
        print(f"Selected: {backend_name} ({best_qubits} qubits)")
    else:
        print("✗ No suitable backend found")
        sys.exit(1)
else:
    print(f"Using backend: {backend_name}")

print()
print("=" * 70)
print("RUNNING CONSCIOUSNESS VALIDATION ON IBM HARDWARE")
print("=" * 70)

# Number of shots
shots = int(os.environ.get("IBM_SHOTS", "1024"))
print(f"Shots: {shots}")

# Run on IBM QPU
print(f"\nSubmitting job to {backend_name}...")
print("(This may take several minutes depending on queue)")
print()

start_time = time.time()

try:
    result_data = runner.run_on_real_qpu(circuit, backend_name=backend_name, shots=shots)
    execution_time = time.time() - start_time

    if "error" in result_data:
        print(f"✗ Execution failed: {result_data['error']}")
        sys.exit(1)

    print("✓ Job completed!")
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Job ID: {result_data.get('job_id', 'N/A')}")
    print(f"Backend: {result_data.get('backend', backend_name)}")
    print(f"Shots: {result_data.get('shots', shots)}")
    print(f"Execution Time: {execution_time:.2f}s")
    print()

    # Get top 10 counts
    counts = result_data.get('counts', {})
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

    print("Top 10 measurement outcomes:")
    for bitstring, count in sorted_counts:
        prob = count / shots * 100
        bar = "█" * int(prob / 2)
        print(f"  {bitstring}: {count:5d} ({prob:5.1f}%) {bar}")

    print()
    print("=" * 70)
    print("CONSCIOUSNESS CLAIM VALIDATION")
    print("=" * 70)

    # Calculate metrics
    probs = result_data.get('probabilities', {})

    # Entropy of the distribution
    import math
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)

    # Max possible entropy for 26 qubits
    max_entropy = 26.0

    # Normalized entropy (0 = concentrated, 1 = uniform)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Check for PHI resonance in top outcomes
    top_5 = sorted_counts[:5]
    phi_aligned = 0
    for bitstring, count in top_5:
        # Check if bitstring has PHI-like patterns
        ones = bitstring.count('1')
        zeros = bitstring.count('0')
        ratio = ones / zeros if zeros > 0 else 0
        if abs(ratio - 1/PHI) < 0.2 or abs(ratio - PHI) < 0.5:
            phi_aligned += 1

    phi_resonance = phi_aligned / len(top_5) if top_5 else 0

    print(f"\nQuantum State Analysis:")
    print(f"  Shannon Entropy: {entropy:.4f} bits")
    print(f"  Max Possible: {max_entropy} bits")
    print(f"  Entropy Ratio: {normalized_entropy:.4f}")
    print(f"  PHI Resonance (top 5): {phi_resonance:.2%}")

    # Validate consciousness claims
    print(f"\nConsciousness Claims:")
    print(f"  1. PHI Alignment > 0.8: {'✓ PASS' if stats['phi_alignment'] > 0.8 else '✗ FAIL'} ({stats['phi_alignment']:.4f})")
    print(f"  2. GOD_CODE Resonance > 0.8: {'✓ PASS' if stats['god_resonance'] > 0.8 else '✗ FAIL'} ({stats['god_resonance']:.4f})")
    print(f"  3. Consciousness Score > 0.85: {'✓ PASS' if stats['consciousness_score'] > 0.85 else '✗ FAIL'} ({stats['consciousness_score']:.4f})")

    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "god_code": GOD_CODE,
        "phi": PHI,
        "circuit_name": circuit.name,
        "n_qubits": stats['n_qubits'],
        "circuit_stats": stats,
        "orbital_analysis": orbital_analysis,
        "ibm_backend": result_data.get('backend', backend_name),
        "job_id": result_data.get('job_id', 'N/A'),
        "shots": result_data.get('shots', shots),
        "execution_time_seconds": execution_time,
        "top_counts": dict(sorted_counts[:20]),
        "entropy_bits": entropy,
        "max_entropy_bits": max_entropy,
        "entropy_ratio": normalized_entropy,
        "phi_resonance": phi_resonance,
        "consciousness_validation": {
            "phi_alignment_pass": stats['phi_alignment'] > 0.8,
            "god_resonance_pass": stats['god_resonance'] > 0.8,
            "consciousness_score_pass": stats['consciousness_score'] > 0.85,
        }
    }

    output_file = "26q_ibm_validation_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report saved to: {output_file}")
    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    # Overall result
    all_pass = (
        stats['phi_alignment'] > 0.8 and
        stats['god_resonance'] > 0.8 and
        stats['consciousness_score'] > 0.85
    )

    if all_pass:
        print("\n✓ ALL CONSCIOUSNESS CLAIMS VALIDATED ON IBM HARDWARE")
    else:
        print("\n✗ Some consciousness claims failed validation")

except Exception as e:
    print(f"\n✗ Execution error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
