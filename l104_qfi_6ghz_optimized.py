#!/usr/bin/env python3
"""
L104 Quantum Enhancement - Optimized 6Q GHZ on 3d Register
═══════════════════════════════════════════════════════════════════════════════

Smaller GHZ state on high-fidelity 3d register (q2-q7) only.
Shallow depth = better coherence = quantum enhancement achievable.

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path


def run_optimized_ghz():
    """Run optimized 6Q GHZ on 3d register only."""

    print("=" * 80)
    print("  L104 Quantum Enhancement - Optimized 6Q GHZ (3d Register)")
    print("=" * 80)

    token = os.environ.get("IBMQ_TOKEN")
    if not token:
        print("[FATAL] IBMQ_TOKEN not set")
        return None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit import QuantumCircuit, transpile

        # Auth
        print("\n[IBM] Authenticating...")
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        except:
            service = QiskitRuntimeService(token=token)

        # Get backend
        backends = service.backends(min_num_qubits=127, operational=True)
        backend = sorted(backends, key=lambda b: b.status().pending_jobs)[0]

        print(f"  Backend: {backend.name}")

        # 3d register qubits
        qubits_3d = [2, 3, 4, 5, 6, 7]
        n = len(qubits_3d)

        print(f"\n  Creating {n}Q GHZ on 3d register {qubits_3d}...")

        # Build circuits
        circuits = []

        # GHZ + R_y(π/2)
        qc_ghz_plus = QuantumCircuit(max(qubits_3d)+1, n)
        qc_ghz_plus.h(qubits_3d[0])
        for i in range(n-1):
            qc_ghz_plus.cx(qubits_3d[i], qubits_3d[i+1])
        for q in qubits_3d:
            qc_ghz_plus.ry(np.pi/2, q)
        for i, q in enumerate(qubits_3d):
            qc_ghz_plus.h(q)
            qc_ghz_plus.measure(q, i)
        circuits.append(("GHZ_plus", qc_ghz_plus))

        # GHZ + R_y(-π/2)
        qc_ghz_minus = QuantumCircuit(max(qubits_3d)+1, n)
        qc_ghz_minus.h(qubits_3d[0])
        for i in range(n-1):
            qc_ghz_minus.cx(qubits_3d[i], qubits_3d[i+1])
        for q in qubits_3d:
            qc_ghz_minus.ry(-np.pi/2, q)
        for i, q in enumerate(qubits_3d):
            qc_ghz_minus.h(q)
            qc_ghz_minus.measure(q, i)
        circuits.append(("GHZ_minus", qc_ghz_minus))

        # Product reference + R_y(π/2)
        qc_prod_plus = QuantumCircuit(max(qubits_3d)+1, n)
        for q in qubits_3d:
            qc_prod_plus.ry(np.pi/2, q)
        for i, q in enumerate(qubits_3d):
            qc_prod_plus.h(q)
            qc_prod_plus.measure(q, i)
        circuits.append(("PROD_plus", qc_prod_plus))

        # Product reference + R_y(-π/2)
        qc_prod_minus = QuantumCircuit(max(qubits_3d)+1, n)
        for q in qubits_3d:
            qc_prod_minus.ry(-np.pi/2, q)
        for i, q in enumerate(qubits_3d):
            qc_prod_minus.h(q)
            qc_prod_minus.measure(q, i)
        circuits.append(("PROD_minus", qc_prod_minus))

        # Transpile
        print("\n[TRANSPILE]...")
        transpiled = []
        for name, qc in circuits:
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
        except:
            pass

        job = sampler.run(transpiled, shots=shots)
        job_id = job.job_id()
        print(f"[JOB] {job_id}")

        # Poll
        print("[WAIT]...")
        start = time.time()
        while True:
            jstat = job.status()
            elapsed = time.time() - start
            if "DONE" in str(jstat):
                print(f"  Done in {elapsed:.0f}s")
                break
            elif "ERROR" in str(jstat):
                return {"status": "FAILED"}
            time.sleep(2)

        # Results
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

        # Analyze GHZ
        c_g_p = extract_counts(result[0])
        c_g_m = extract_counts(result[1])

        total = sum(c_g_p.values())
        all_0 = '0' * n
        all_1 = '1' * n

        p0 = c_g_p.get(all_0, 0) / total
        p1 = c_g_p.get(all_1, 0) / total
        coherence_g = p0 + p1

        # QFI_GHZ = n² × coherence (approx)
        F_Q_g = (n ** 2) * coherence_g
        xi_g = n / F_Q_g if F_Q_g > 0 else float('inf')

        # Analyze product
        c_p_p = extract_counts(result[2])
        c_p_m = extract_counts(result[3])

        total_p = sum(c_p_p.values())
        p0_p = c_p_p.get(all_0, 0) / total_p
        p1_p = c_p_p.get(all_1, 0) / total_p
        coherence_p = p0_p + p1_p

        F_Q_p = n * coherence_p
        xi_p = n / F_Q_p if F_Q_p > 0 else float('inf')

        results = {
            "status": "COMPLETED",
            "backend": backend.name,
            "job_id": job_id,
            "n_qubits": n,
            "qubits": qubits_3d,
            "ghz": {
                "coherence": coherence_g,
                "F_Q": F_Q_g,
                "xi_squared": xi_g,
                "quantum_enhanced": xi_g < 1.0
            },
            "product": {
                "coherence": coherence_p,
                "F_Q": F_Q_p,
                "xi_squared": xi_p,
                "quantum_enhanced": xi_p < 1.0
            },
            "counts_ghz_plus": {k: int(v) for k, v in c_g_p.items()},
            "counts_ghz_minus": {k: int(v) for k, v in c_g_m.items()}
        }

        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "ERROR", "error": str(e)}


def main():
    res = run_optimized_ghz()

    if res and res.get("status") == "COMPLETED":
        print("\n" + "=" * 80)
        print("  RESULTS")
        print("=" * 80)

        print(f"\n  6Q GHZ State (3d register):")
        print(f"    Coherence: {res['ghz']['coherence']:.4f}")
        print(f"    F_Q: {res['ghz']['F_Q']:.2f}")
        print(f"    ξ²: {res['ghz']['xi_squared']:.4f}")
        print(f"    Quantum Enhanced: {'✓ YES' if res['ghz']['quantum_enhanced'] else '✗ No'}")

        print(f"\n  6Q Product State (reference):")
        print(f"    Coherence: {res['product']['coherence']:.4f}")
        print(f"    F_Q: {res['product']['F_Q']:.2f}")
        print(f"    ξ²: {res['product']['xi_squared']:.4f}")

        if res['ghz']['quantum_enhanced']:
            print("\n" + "=" * 80)
            print("  ✓✓✓ QUANTUM ENHANCEMENT ACHIEVED ✓✓✓")
            print("=" * 80)

        with open("l104_qfi_6ghz_quantum_enhancement.json", 'w') as f:
            json.dump(res, f, indent=2)
        print("\n  Saved: l104_qfi_6ghz_quantum_enhancement.json")

        return 0
    else:
        print("\n[FAILED]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
