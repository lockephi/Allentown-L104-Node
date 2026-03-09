"""
L104 God Code Simulator — QPU Runner v1.0.0
═══════════════════════════════════════════════════════════════════════════════

Run GOD_CODE verification circuits on real IBM Quantum hardware.
Importable module with structured results — replaces _tmp_qpu_verification.py.

Usage:
    from l104_god_code_simulator.qpu_runner import QPUVerificationRunner

    runner = QPUVerificationRunner()
    result = runner.run()                     # Full verification (4 circuits)
    result = runner.run(circuits=["bell_state", "1q_god_code"])  # Subset
    result = runner.run(backend="ibm_fez")    # Override backend

Requires:
    - qiskit-ibm-runtime >= 0.45
    - IBMQ_TOKEN or IBM_QUANTUM_TOKEN env var
    - IBM_QUANTUM_CHANNEL env var (default: ibm_cloud)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .constants import GOD_CODE, PHI


# ═══════════════════════════════════════════════════════════════════════════════
#  CIRCUIT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_verification_circuits() -> Dict[str, Dict[str, Any]]:
    """Build the 4 standard GOD_CODE verification circuits."""
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("qiskit is required: pip install qiskit qiskit-ibm-runtime")

    circuits: Dict[str, Dict[str, Any]] = {}

    # Circuit 1: Bell State (2Q) — deterministic entanglement check
    qc1 = QuantumCircuit(2, name="bell_state")
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.measure_all()
    circuits["bell_state"] = {
        "qc": qc1,
        "ideal": {"00": 0.5, "11": 0.5},
        "desc": "Bell state |Φ+⟩ — expect 50/50 on |00⟩,|11⟩",
    }

    # Circuit 2: 1Q GOD_CODE Phase Gate — sacred alignment
    theta_god = 2.0 * np.pi * (GOD_CODE / 1000.0)
    qc2 = QuantumCircuit(1, name="1q_god_code")
    qc2.h(0)
    qc2.rz(theta_god, 0)
    qc2.h(0)
    qc2.measure_all()
    p1_ideal = np.sin(theta_god / 2) ** 2
    circuits["1q_god_code"] = {
        "qc": qc2,
        "ideal": {"0": 1.0 - p1_ideal, "1": p1_ideal},
        "desc": f"GOD_CODE phase gate θ={theta_god:.4f} rad",
    }

    # Circuit 3: GHZ State (3Q) — multipartite entanglement
    qc3 = QuantumCircuit(3, name="ghz_3q")
    qc3.h(0)
    qc3.cx(0, 1)
    qc3.cx(1, 2)
    qc3.measure_all()
    circuits["ghz_3q"] = {
        "qc": qc3,
        "ideal": {"000": 0.5, "111": 0.5},
        "desc": "GHZ |000⟩+|111⟩ — tripartite entanglement",
    }

    # Circuit 4: PHI-rotation interference (1Q)
    theta_phi = 2.0 * np.pi / PHI
    qc4 = QuantumCircuit(1, name="phi_interference")
    qc4.h(0)
    qc4.rz(theta_phi, 0)
    qc4.h(0)
    qc4.measure_all()
    p1_phi = np.sin(theta_phi / 2) ** 2
    circuits["phi_interference"] = {
        "qc": qc4,
        "ideal": {"0": 1.0 - p1_phi, "1": p1_phi},
        "desc": f"PHI interference θ=2π/φ={theta_phi:.4f} rad",
    }

    return circuits


def _bhattacharyya_fidelity(measured: Dict[str, float], ideal: Dict[str, float]) -> float:
    """Compute Bhattacharyya fidelity between measured and ideal distributions."""
    all_states = sorted(set(list(measured.keys()) + list(ideal.keys())))
    bc = sum(np.sqrt(measured.get(s, 0) * ideal.get(s, 0)) for s in all_states)
    return float(bc ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QPUVerificationResult:
    """Structured result from a QPU verification run."""
    backend: str = ""
    job_id: str = ""
    shots: int = 4096
    elapsed_s: float = 0.0
    mean_fidelity: float = 0.0
    circuits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = ""
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "job_id": self.job_id,
            "shots": self.shots,
            "elapsed_s": round(self.elapsed_s, 2),
            "mean_fidelity": round(self.mean_fidelity, 8),
            "circuits": self.circuits,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
        }

    def to_scoring_payload(self) -> Dict[str, Any]:
        """Convert to ASI/AGI scoring-compatible payload."""
        return {
            "qpu_verified": self.success,
            "mean_fidelity": self.mean_fidelity,
            "backend": self.backend,
            "circuit_count": len(self.circuits),
            "all_pass": all(c.get("status") == "PASS" for c in self.circuits.values()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  QPU VERIFICATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

PREFERRED_BACKENDS = ["ibm_torino", "ibm_fez", "ibm_marrakesh"]


class QPUVerificationRunner:
    """
    Run GOD_CODE verification circuits on real IBM Quantum hardware.

    Encapsulates IBM Quantum connection, circuit building, transpilation,
    execution, and fidelity analysis into a clean importable interface.
    """

    def __init__(self, token: Optional[str] = None, channel: Optional[str] = None,
                 instance: Optional[str] = None):
        self._token = token or os.environ.get("IBMQ_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
        self._channel = channel or os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_cloud")
        self._instance = instance or os.environ.get("IBM_QUANTUM_INSTANCE") or None
        self._service = None

    def _connect(self):
        """Establish IBM Quantum connection."""
        if self._service is not None:
            return

        if not self._token:
            raise RuntimeError("No IBM Quantum token. Set IBMQ_TOKEN or IBM_QUANTUM_TOKEN env var.")

        from qiskit_ibm_runtime import QiskitRuntimeService
        svc_kwargs: Dict[str, Any] = {"channel": self._channel, "token": self._token}
        if self._instance:
            svc_kwargs["instance"] = self._instance
        self._service = QiskitRuntimeService(**svc_kwargs)

    def _select_backend(self, backend_name: Optional[str] = None):
        """Select the best available backend."""
        if backend_name:
            return self._service.backend(backend_name)

        backends = self._service.backends(min_num_qubits=2, operational=True)
        backend_names = [b.name for b in backends]

        for pref in PREFERRED_BACKENDS:
            if pref in backend_names:
                return self._service.backend(pref)

        return self._service.least_busy(min_num_qubits=2, operational=True)

    def run(self, circuits: Optional[List[str]] = None,
            backend: Optional[str] = None,
            shots: int = 4096,
            verbose: bool = False) -> QPUVerificationResult:
        """
        Execute QPU verification and return structured results.

        Args:
            circuits: List of circuit names to run (default: all 4).
                      Options: "bell_state", "1q_god_code", "ghz_3q", "phi_interference"
            backend: IBM backend name override (default: auto-select ibm_torino).
            shots: Shots per circuit (default: 4096).
            verbose: Print progress to stdout.

        Returns:
            QPUVerificationResult with per-circuit fidelities and counts.
        """
        from datetime import datetime, timezone

        result = QPUVerificationResult(shots=shots)
        result.timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Connect
            if verbose:
                print(f"  Connecting to IBM Quantum (channel={self._channel})...")
            self._connect()

            # Select backend
            hw = self._select_backend(backend)
            result.backend = hw.name
            if verbose:
                print(f"  Backend: {hw.name} ({hw.num_qubits}Q)")

            # Build circuits
            all_circuits = _build_verification_circuits()
            if circuits:
                selected = {k: v for k, v in all_circuits.items() if k in circuits}
            else:
                selected = all_circuits

            if not selected:
                result.error = f"No matching circuits. Available: {list(all_circuits.keys())}"
                return result

            # Transpile
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            pm = generate_preset_pass_manager(backend=hw, optimization_level=2)
            transpiled = []
            circuit_names = []
            for name, c in selected.items():
                transpiled.append(pm.run(c["qc"]))
                circuit_names.append(name)
                if verbose:
                    print(f"  Transpiled {name}: depth={transpiled[-1].depth()}")

            # Execute via Batch mode
            from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch
            if verbose:
                print(f"  Submitting {len(transpiled)} circuits ({shots} shots each)...")

            t0 = time.time()
            with Batch(backend=hw) as batch:
                sampler = Sampler(mode=batch)
                job = sampler.run(transpiled, shots=shots)
                result.job_id = job.job_id()
                if verbose:
                    print(f"  Job ID: {result.job_id} — waiting...")
                qpu_result = job.result()
            result.elapsed_s = time.time() - t0

            # Analyze
            all_fidelities = []
            for i, name in enumerate(circuit_names):
                pub_result = qpu_result[i]
                counts = pub_result.data.meas.get_counts()
                total = sum(counts.values())
                probs = {k: v / total for k, v in sorted(counts.items(), key=lambda x: -x[1])}
                ideal = selected[name]["ideal"]
                fidelity = _bhattacharyya_fidelity(probs, ideal)
                all_fidelities.append(fidelity)
                status = "PASS" if fidelity > 0.85 else "WARN" if fidelity > 0.70 else "FAIL"

                result.circuits[name] = {
                    "counts": counts,
                    "probabilities": dict(list(probs.items())[:8]),
                    "fidelity": round(fidelity, 10),
                    "status": status,
                    "description": selected[name]["desc"],
                }

                if verbose:
                    print(f"  {status}  {name}: fidelity={fidelity:.6f}")

            result.mean_fidelity = float(np.mean(all_fidelities))
            result.success = True

            if verbose:
                print(f"  Mean fidelity: {result.mean_fidelity:.6f}")
                print(f"  Elapsed: {result.elapsed_s:.1f}s")

        except Exception as e:
            result.error = str(e)
            result.success = False

        return result


__all__ = [
    "QPUVerificationRunner",
    "QPUVerificationResult",
    "PREFERRED_BACKENDS",
]
