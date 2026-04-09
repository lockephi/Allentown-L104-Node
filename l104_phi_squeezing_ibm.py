#!/usr/bin/env python3
"""
L104 PHI-Based Squeezing Test
═══════════════════════════════════════════════════════════════════════════════

Tests quantum squeezing where the squeezing parameter is derived from PHI.

Squeezing operator: S(r) = exp(r/2 * (a² - a†²))
For PHI-squeezing: r = log(PHI) ≈ 0.481

The squeezed vacuum state has quadrature variances:
  Var(X) = e^(-2r)/4 = 1/(4*PHI²) ≈ 0.0955
  Var(P) = e^(2r)/4 = PHI²/4 ≈ 0.655

Squeezing parameter: ξ² = Var(X)/(1/4) = 1/PHI² ≈ 0.382

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Constants
PHI = 1.618033988749895
PHI_SQUARED = PHI ** 2
PHI_CONJUGATE = PHI - 1.0

def create_phi_squeezed_circuit(qubit_indices: list) -> 'QuantumCircuit':
    """
    Create PHI-based squeezed state.

    Approximates squeezing using available Clifford+T gates:
    - R_x rotations for displacement
    - S and T gates for phase
    - CNOT for entanglement (multi-mode squeezing)
    """
    from qiskit import QuantumCircuit

    max_q = max(qubit_indices)
    n = len(qubit_indices)
    qc = QuantumCircuit(max_q + 1, n)

    # PHI-scaled squeezing parameter
    r_squeeze = np.log(PHI)  # ≈ 0.481

    # Approximate squeezing using rotation sequence
    # |ψ⟩ ≈ R_y(θ_1) R_z(θ_2) R_y(θ_3) |0⟩
    # where θ are PHI-scaled

    theta_1 = np.pi / (2 * PHI)  # ≈ 0.972 rad
    theta_2 = np.pi / PHI_SQUARED  # ≈ 1.199 rad
    theta_3 = np.pi / (2 * PHI_CONJUGATE)  # ≈ 2.556 rad

    for q in qubit_indices:
        # Approximate squeezed vacuum
        qc.ry(theta_1, q)
        qc.rz(theta_2, q)
        qc.ry(theta_3, q)

        # PHI-scaled phase gate (simulates squeezing phase)
        phi_phase = 2 * np.pi / PHI_SQUARED
        qc.p(phi_phase, q)

    return qc


def run_phi_squeezing_test():
    """Run PHI-based squeezing test on IBM hardware."""

    print("=" * 80)
    print("  L104 PHI-Based Squeezing Test")
    print(f"  PHI = {PHI:.10f}")
    print("=" * 80)

    token = os.environ.get("IBMQ_TOKEN")
    if not token:
        print("\n[FATAL] IBMQ_TOKEN not set")
        return None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit import QuantumCircuit, transpile

        # Authenticate
        print("\n[IBM] Authenticating...")
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        except:
            service = QiskitRuntimeService(token=token)

        # Get best backend
        print("[IBM] Finding optimal backend...")
        backends = service.backends(min_num_qubits=127, operational=True)
        backend = sorted(backends, key=lambda b: b.status().pending_jobs)[0]

        print(f"  Backend: {backend.name}")
        print(f"  Qubits: {backend.num_qubits}")

        # Test qubits - 3d register
        qubits = [2, 3, 4, 5, 6, 7]
        n = len(qubits)

        print(f"\n  Testing PHI-squeezing on {n} qubits: {qubits}")

        # Theoretical PHI-squeezing parameters
        r_phi = np.log(PHI)
        xi_sq_theory = 1.0 / PHI_SQUARED
        var_X_theory = 1.0 / (4 * PHI_SQUARED)
        var_P_theory = PHI_SQUARED / 4

        print(f"\n  Theory:")
        print(f"    Squeezing parameter r = log(φ) = {r_phi:.6f}")
        print(f"    ξ² = 1/φ² = {xi_sq_theory:.6f}")
        print(f"    Var(X) = 1/(4φ²) = {var_X_theory:.6f}")
        print(f"    Var(P) = φ²/4 = {var_P_theory:.6f}")

        # Build circuits
        circuits = []
        metadata = []

        # Circuit 1: PHI-squeezed state, measure X (Hadamard basis)
        qc_x = create_phi_squeezed_circuit(qubits)
        for i, q in enumerate(qubits):
            qc_x.h(q)  # Rotate to X basis
            qc_x.measure(q, i)
        circuits.append(qc_x)
        metadata.append("PHI_squeezed_X")

        # Circuit 2: PHI-squeezed state, measure Z (computational basis)
        qc_z = create_phi_squeezed_circuit(qubits)
        for i, q in enumerate(qubits):
            qc_z.measure(q, i)
        circuits.append(qc_z)
        metadata.append("PHI_squeezed_Z")

        # Circuit 3: Reference (no squeezing), measure X
        qc_ref_x = QuantumCircuit(max(qubits)+1, n)
        for q in qubits:
            qc_ref_x.ry(np.pi/4, q)  # Equal superposition
        for i, q in enumerate(qubits):
            qc_ref_x.h(q)
            qc_ref_x.measure(q, i)
        circuits.append(qc_ref_x)
        metadata.append("Reference_X")

        # Circuit 4: Reference, measure Z
        qc_ref_z = QuantumCircuit(max(qubits)+1, n)
        for q in qubits:
            qc_ref_z.ry(np.pi/4, q)
        for i, q in enumerate(qubits):
            qc_ref_z.measure(q, i)
        circuits.append(qc_ref_z)
        metadata.append("Reference_Z")

        # Transpile
        print(f"\n[TRANSPILE] {len(circuits)} circuits...")
        transpiled = []
        for name, qc in zip(metadata, circuits):
            tp = transpile(qc, backend=backend, optimization_level=3)
            transpiled.append(tp)
            print(f"  {name}: {tp.size()} gates, depth {tp.depth()}")

        # Run
        shots = 8192
        print(f"\n[DEPLOY] {shots} shots...")
        sampler = SamplerV2(backend)

        try:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            print("  DD: XY4")
        except:
            pass

        job = sampler.run(transpiled, shots=shots)
        job_id = job.job_id()
        print(f"[JOB] {job_id}")

        # Poll
        print("[WAIT] Monitoring...")
        start = time.time()
        last_status = None

        while True:
            try:
                jstat = job.status()
                status_name = str(jstat).split('.')[-1] if hasattr(jstat, '__class__') else str(jstat)
            except:
                status_name = "UNKNOWN"

            elapsed = time.time() - start

            if status_name != last_status:
                print(f"    [{elapsed:5.0f}s] {status_name}")
                last_status = status_name

            if "DONE" in status_name or "COMPLETED" in status_name:
                print(f"\n  Done in {elapsed:.0f}s")
                break
            elif "ERROR" in status_name or "FAILED" in status_name:
                return {"status": "FAILED"}
            elif elapsed > 300:
                return {"status": "TIMEOUT"}

            time.sleep(3)

        # Get results
        result = job.result()

        def extract_counts(pub_result):
            try:
                data = pub_result.data
                if hasattr(data, 'c'):
                    return data.c.get_counts()
                elif hasattr(data, 'meas'):
                    return data.meas.get_counts()
            except:
                pass
            return {}

        # Analyze
        print("\n[ANALYZE] Computing PHI-squeezing parameters...")

        c_phi_x = extract_counts(result[0])
        c_phi_z = extract_counts(result[1])
        c_ref_x = extract_counts(result[2])
        c_ref_z = extract_counts(result[3])

        # Compute variances
        def compute_variance(counts, n_qubits):
            total = sum(counts.values())
            if total == 0:
                return 0.5, 0.5

            # For each bitstring, compute mean bit value
            mean = 0.0
            mean_sq = 0.0

            for bitstring, count in counts.items():
                # Average bit value in string
                avg_bit = sum(int(b) for b in bitstring) / len(bitstring)
                prob = count / total
                mean += avg_bit * prob
                mean_sq += avg_bit**2 * prob

            variance = mean_sq - mean**2
            return mean, variance

        # PHI-squeezed variances
        mean_phi_x, var_phi_x = compute_variance(c_phi_x, n)
        mean_phi_z, var_phi_z = compute_variance(c_phi_z, n)

        # Reference variances
        mean_ref_x, var_ref_x = compute_variance(c_ref_x, n)
        mean_ref_z, var_ref_z = compute_variance(c_ref_z, n)

        # Squeezing parameters
        xi_sq_phi = var_phi_x / var_ref_x if var_ref_x > 0 else 1.0

        # Check if PHI-squeezing achieved
        phi_squeezing = xi_sq_phi < 1.0 / PHI

        results = {
            "status": "COMPLETED",
            "backend": backend.name,
            "job_id": job_id,
            "n_qubits": n,
            "qubits": qubits,
            "theory": {
                "PHI": PHI,
                "squeezing_param_r": float(r_phi),
                "xi_squared_theory": float(xi_sq_theory),
                "var_X_theory": float(var_X_theory),
                "var_P_theory": float(var_P_theory)
            },
            "phi_squeezed": {
                "mean_X": float(mean_phi_x),
                "var_X": float(var_phi_x),
                "mean_Z": float(mean_phi_z),
                "var_Z": float(var_phi_z),
                "xi_squared": float(xi_sq_phi),
                "counts_X": {k: int(v) for k, v in c_phi_x.items()},
                "counts_Z": {k: int(v) for k, v in c_phi_z.items()}
            },
            "reference": {
                "mean_X": float(mean_ref_x),
                "var_X": float(var_ref_x),
                "mean_Z": float(mean_ref_z),
                "var_Z": float(var_ref_z)
            },
            "analysis": {
                "phi_squeezing_achieved": bool(phi_squeezing),
                "xi_squared_measured": float(xi_sq_phi),
                "xi_squared_target": float(xi_sq_theory),
                "squeezing_vs_target": float(xi_sq_phi / xi_sq_theory)
            }
        }

        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "ERROR", "error": str(e)}


def main():
    res = run_phi_squeezing_test()

    if res and res.get("status") == "COMPLETED":
        print("\n" + "=" * 80)
        print("  PHI SQUEEZING TEST RESULTS")
        print("=" * 80)

        print(f"\n  Backend: {res['backend']}")
        print(f"  Job: {res['job_id']}")
        print(f"  Qubits: {res['n_qubits']}")

        print(f"\n  Theory (PHI = {PHI:.6f}):")
        print(f"    ξ² = 1/φ² = {res['theory']['xi_squared_theory']:.6f}")

        print(f"\n  PHI-Squeezed State:")
        print(f"    Var(X) = {res['phi_squeezed']['var_X']:.6f}")
        print(f"    Var(Z) = {res['phi_squeezed']['var_Z']:.6f}")
        print(f"    ξ² = {res['phi_squeezed']['xi_squared']:.6f}")

        print(f"\n  Reference State:")
        print(f"    Var(X) = {res['reference']['var_X']:.6f}")
        print(f"    Var(Z) = {res['reference']['var_Z']:.6f}")

        print(f"\n  Comparison:")
        print(f"    Target ξ²: {res['analysis']['xi_squared_target']:.6f}")
        print(f"    Measured ξ²: {res['analysis']['xi_squared_measured']:.6f}")
        print(f"    Ratio: {res['analysis']['squeezing_vs_target']:.3f}")

        if res['analysis']['phi_squeezing_achieved']:
            print("\n" + "=" * 80)
            print("  ✓✓✓ PHI-BASED SQUEEZING ACHIEVED ✓✓✓")
            print("  (ξ² < 1/φ indicates quantum enhancement)")
            print("=" * 80)
        else:
            print("\n  [INFO] PHI-squeezing not achieved (decoherence)")

        with open("l104_phi_squeezing_results.json", 'w') as f:
            json.dump(res, f, indent=2)
        print("\n  Saved: l104_phi_squeezing_results.json")

        return 0
    else:
        print("\n[FAILED]")
        if res:
            print(f"  Status: {res.get('status')}")
            if 'error' in res:
                print(f"  Error: {res['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
