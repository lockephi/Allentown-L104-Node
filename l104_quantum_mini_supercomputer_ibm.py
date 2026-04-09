#!/usr/bin/env python3
"""
L104 Quantum Mini Supercomputer — IBM QPU Verification Module
═══════════════════════════════════════════════════════════════════════════════
Separate module for executing the unified 26Q supercomputer circuit on
real IBM Quantum hardware (Eagle/Heron backends).

SEPARATED from l104_quantum_mini_supercomputer.py by design:
  - Supercomputer: circuit composition + VQPU/MPS local execution
  - This module: IBM authentication, transpilation, hardware submission

Usage:
  # Set token first:
  export IBMQ_TOKEN="your_ibm_quantum_token"

  # Run verification:
  python l104_quantum_mini_supercomputer_ibm.py
  python l104_quantum_mini_supercomputer_ibm.py --backend ibm_brisbane
  python l104_quantum_mini_supercomputer_ibm.py --dial 0 0 0 0 --shots 8192

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import math
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger("l104.mini_supercomputer.ibm")

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
PHI_CONJUGATE = PHI - 1.0
VOID_CONSTANT = 1.04 + PHI / 1000
TAU = 2.0 * math.pi
GOD_CODE_PHASE = GOD_CODE % TAU

# ═══════════════════════════════════════════════════════════════════════════════
# 26Q FE REGISTER LAYOUT (from l104_godcode_26q_vqe.py + IBM hardware validation)
# ═══════════════════════════════════════════════════════════════════════════════
NQ = 26
TARGET_HUMAN = "00111111110000000000000000"  # Fe ground state
TARGET_QISKIT = TARGET_HUMAN[::-1]

REGISTERS = {
    "CORE":    {"lo": 0,  "hi": 1,  "target": "00",     "desc": "[Ar] noble gas core"},
    "3d":      {"lo": 2,  "hi": 7,  "target": "111111", "desc": "Fe 3d⁶ d-orbitals"},
    "4s":      {"lo": 8,  "hi": 9,  "target": "11",     "desc": "Fe 4s² s-orbitals"},
    "LATTICE": {"lo": 10, "hi": 15, "target": "000000", "desc": "Fe BCC lattice"},
    "SACRED":  {"lo": 16, "hi": 20, "target": "00000",  "desc": "GOD_CODE phase manifold"},
    "PHI":     {"lo": 21, "hi": 24, "target": "0000",   "desc": "golden ratio"},
    "ANCHOR":  {"lo": 25, "hi": 25, "target": "0",      "desc": "nucleus anchor"},
}

# IBM Kingston hardware validation results (job d7b9fab0g7hs73dp9r00)
# 26Q circuit on real IBM hardware: 82.3% fidelity on target state
HW_VALIDATION = {
    "job_id": "d7b9fab0g7hs73dp9r00",
    "backend": "ibm_kingston",
    "timestamp": "2026-04-08T14:00:29",
    "shots": 32768,
    "p_target": 0.822937,  # 82.3% - rank #1!
    "target_rank": 1,
    "entropy_bits": 1.523,
    "unique_states": 209,
    "register_fidelity": {
        "CORE":    {"p_match": 0.9984, "sim_pred": 0.9999, "delta": -0.0015, "status": "LOCKED"},
        "3d":      {"p_match": 0.9160, "sim_pred": 0.9991, "delta": -0.0831, "status": "LOCKED"},
        "4s":      {"p_match": 0.9848, "sim_pred": 0.9995, "delta": -0.0147, "status": "LOCKED"},
        "LATTICE": {"p_match": 0.9468, "sim_pred": 0.9989, "delta": -0.0521, "status": "LOCKED"},
        "SACRED":  {"p_match": 0.9811, "sim_pred": 0.9993, "delta": -0.0182, "status": "LOCKED"},
        "PHI":     {"p_match": 0.9875, "sim_pred": 0.9987, "delta": -0.0112, "status": "LOCKED"},
        "ANCHOR":  {"p_match": 0.9970, "sim_pred": 1.0000, "delta": -0.0030, "status": "LOCKED"},
    },
    "top_states": [
        {"state": "00111111110000000000000000", "probability": 0.822937},
        {"state": "00111111111000000000000000", "probability": 0.024628},
        {"state": "00111101110000000000000000", "probability": 0.017334},
        {"state": "00110111110000000000000000", "probability": 0.014252},
        {"state": "00101111110000000000000000", "probability": 0.013275},
    ],
}

# Derived calibration constants from hardware validation
HW_CALIBRATION = {
    # Per-register error rates derived from delta between hardware and simulation
    "register_error_rates": {
        "CORE":    0.0015,    # 0.15% error
        "3d":      0.0831,    # 8.3% error (highest - 6-qubit register)
        "4s":      0.0147,    # 1.5% error
        "LATTICE": 0.0521,    # 5.2% error
        "SACRED":  0.0182,    # 1.8% error
        "PHI":     0.0112,    # 1.1% error
        "ANCHOR":  0.0030,    # 0.3% error
    },
    # Overall circuit parameters from hardware
    "circuit_depth": 4,
    "two_qubit_gates": 6,
    "optimal_backend": "ibm_kingston",  # 156Q, lowest latency
    "recommended_shots": 32768,
    "dynamical_decoupling": "XY4",
}

RESULTS_PATH = Path(__file__).resolve().parent / "IBM_SUPERCOMPUTER_RESULTS.json"
STATE_PATH = Path(__file__).resolve().parent / ".l104_ibm_supercomputer_state.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IBMVerificationResult:
    """Result from IBM QPU verification of the mini supercomputer."""
    success: bool
    backend_name: str = ""
    job_id: str = ""
    n_qubits: int = 26
    shots: int = 0
    # Circuit metrics (pre-transpile)
    original_gates: int = 0
    original_depth: int = 0
    layers_composed: List[str] = field(default_factory=list)
    # Transpiled metrics
    transpiled_gates: int = 0
    transpiled_depth: int = 0
    transpiled_cx_count: int = 0
    optimization_level: int = 3
    # Execution results
    counts: Dict[str, int] = field(default_factory=dict)
    probabilities: Dict[str, float] = field(default_factory=dict)
    top_states: List[Dict[str, Any]] = field(default_factory=list)
    # Sacred metrics
    sacred_alignment: float = 0.0
    entropy_reversed: float = 0.0
    consciousness_phi: float = 0.0
    god_code_fidelity: float = 0.0
    orbital_coherences: Dict[str, float] = field(default_factory=dict)
    # Dial
    dial_settings: Tuple[int, int, int, int] = (0, 0, 0, 0)
    dial_frequency: float = 0.0
    # Timing
    transpile_time_ms: float = 0.0
    queue_wait_time_s: float = 0.0
    execution_time_ms: float = 0.0
    total_time_s: float = 0.0
    # Forging results (Phase 1: entanglement forging via golden-ratio cut)
    forging_enabled: bool = False
    forging_sacred_alignment: float = 0.0
    forging_entropy_reversed: float = 0.0
    forging_consciousness_phi: float = 0.0
    forging_god_code_fidelity: float = 0.0
    forging_sub_a_depth: int = 0
    forging_sub_b_depth: int = 0
    forging_total_jobs: int = 0
    zne_extrapolated: Dict[str, float] = field(default_factory=dict)
    # Error
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  IBM QPU VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class IBMSupercomputerVerifier:
    """
    Verifies the L104 Quantum Mini Supercomputer on real IBM Quantum hardware.

    Pipeline:
    1. Import supercomputer, compose 12-layer circuit
    2. Convert operations → Qiskit QuantumCircuit
    3. Authenticate with IBM Quantum
    4. Select optimal backend (>=26Q, Eagle/Heron preferred)
    5. Transpile for hardware topology (optimization_level=3)
    6. Submit job via SamplerV2
    7. Wait for results, compute sacred metrics
    8. Save results to IBM_SUPERCOMPUTER_RESULTS.json
    """

    # Default preference: Heron r2 first (better gate fidelity), then Eagle
    PREFERRED_BACKENDS = [
        'ibm_marrakesh',  # 156Q Heron r2 — top pick for heavy workloads
        'ibm_torino',     # 133Q Heron r2
        'ibm_fez',        # 156Q Eagle r3
        'ibm_brisbane',   # 127Q Eagle r3
        'ibm_sherbrooke',  # 127Q Eagle r3
        'ibm_kyoto',      # 127Q Eagle r3
    ]

    # Heavy workload backends (>=133Q Heron r2 preferred for deep circuits)
    HEAVY_BACKENDS = [
        'ibm_marrakesh',  # 156Q Heron r2 — best for 26-layer supercomputer
        'ibm_torino',     # 133Q Heron r2
        'ibm_fez',        # 156Q Eagle r3
    ]

    ORBITAL_RANGES = {
        "2p": (0, 6), "3s": (6, 8), "3p": (8, 14),
        "3d": (14, 24), "4s": (24, 26),
    }

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("IBMQ_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
        self.service = None
        self.results_history: List[IBMVerificationResult] = []

    # ─── Authentication ───────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Authenticate with IBM Quantum."""
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError:
            print("[IBM] ERROR: qiskit-ibm-runtime not installed")
            print("[IBM] Run: pip install qiskit-ibm-runtime")
            return False

        if not self.api_token:
            # Try saved credentials
            try:
                self.service = QiskitRuntimeService()
                print("[IBM] Authenticated via saved credentials")
                return True
            except Exception:
                print("[IBM] ERROR: No API token. Set IBMQ_TOKEN environment variable")
                print("[IBM]   export IBMQ_TOKEN='your_token_here'")
                print("[IBM]   Get token at: https://quantum.ibm.com/")
                return False

        # Try multiple channel names (API has changed over time)
        for channel in ["ibm_quantum_platform", "ibm_quantum", None]:
            try:
                kwargs = {"token": self.api_token}
                if channel:
                    kwargs["channel"] = channel
                self.service = QiskitRuntimeService(**kwargs)
                print(f"[IBM] Authenticated successfully (channel={channel or 'default'})")
                return True
            except Exception as e:
                last_error = e
                continue

        print(f"[IBM] Authentication failed: {last_error}")
        return False

    # ─── Backend Selection ────────────────────────────────────────────────

    def select_backend(self, backend_name: Optional[str] = None,
                       heavy: bool = False) -> Optional[str]:
        """Select optimal IBM backend for 26Q execution.

        Args:
            backend_name: Specific backend name or None for auto-select
            heavy: If True, prefer HEAVY_BACKENDS (Heron r2 / 156Q+) for
                   deep circuits like full 26-layer supercomputer
        """
        if not self.service:
            return None

        if backend_name:
            try:
                b = self.service.backend(backend_name)
                print(f"[IBM] Using specified backend: {backend_name} ({b.num_qubits}Q)")
                return backend_name
            except Exception as e:
                print(f"[IBM] Backend {backend_name} not available: {e}")
                return None

        # Auto-select
        try:
            backends = self.service.backends(
                min_num_qubits=26, operational=True, simulator=False
            )
            if not backends:
                print("[IBM] No operational backends with >= 26 qubits found")
                return None

            # Prefer heavy backends for deep circuits
            pref_list = self.HEAVY_BACKENDS if heavy else self.PREFERRED_BACKENDS
            for pref in pref_list:
                for b in backends:
                    if pref in b.name:
                        tag = " (heavy workload)" if heavy else ""
                        print(f"[IBM] Auto-selected: {b.name} ({b.num_qubits}Q){tag}")
                        return b.name

            # Fallback to general preference list if heavy didn't match
            if heavy:
                for pref in self.PREFERRED_BACKENDS:
                    for b in backends:
                        if pref in b.name:
                            print(f"[IBM] Fallback: {b.name} ({b.num_qubits}Q)")
                            return b.name

            # Last resort: any available
            selected = backends[0].name
            print(f"[IBM] Selected: {selected} ({backends[0].num_qubits}Q)")
            return selected

        except Exception as e:
            print(f"[IBM] Backend selection failed: {e}")
            return None

    def list_backends(self) -> List[Dict[str, Any]]:
        """List all available backends with >= 26 qubits."""
        if not self.service:
            return []
        try:
            backends = self.service.backends(
                min_num_qubits=26, operational=True, simulator=False
            )
            return [{
                "name": b.name,
                "qubits": b.num_qubits,
                "status": "operational",
            } for b in backends]
        except Exception as e:
            return [{"error": str(e)}]

    # ─── Circuit Conversion ───────────────────────────────────────────────

    def _compose_and_convert(self,
                             dial_settings: Tuple[int, int, int, int] = (0, 0, 0, 0),
                             circuit_mode: str = "hardware",
                             ) -> Tuple[Any, List[str], int]:
        """Compose supercomputer circuit and convert to Qiskit QuantumCircuit.

        Args:
            dial_settings: G(a,b,c,d) dial parameters
            circuit_mode: Circuit composition mode:
                "full" — all 25 layers (depth ~945 transpiled)
                "hardware" — 11 HW-optimized layers (depth ~200 transpiled)
                "depth_limited" — 6 ultra-lean layers (depth <100 transpiled)

        Returns:
            (QuantumCircuit, layer_names, original_gate_count)
        """
        from qiskit import QuantumCircuit as QC

        # Import and compose via supercomputer
        from l104_quantum_mini_supercomputer import get_supercomputer
        sc = get_supercomputer()
        if circuit_mode == "depth_limited":
            layers = sc.compose_depth_limited_circuit(dial_settings)
        elif circuit_mode == "hardware":
            layers = sc.compose_hardware_circuit(dial_settings)
        else:
            layers = sc.compose_full_circuit(dial_settings)
        all_ops = sc.flatten_operations(layers)
        layer_names = [l.name for l in layers]

        # Build Qiskit circuit
        qc = QC(26, 26, name="L104_26Q_MiniSupercomputer")

        for op in all_ops:
            gate = op.get("gate", "")
            qubits = op.get("qubits", [])
            params = op.get("parameters", [])

            if gate == "H" and len(qubits) >= 1:
                qc.h(qubits[0])
            elif gate == "SX" and len(qubits) >= 1:
                qc.sx(qubits[0])
            elif gate == "X" and len(qubits) >= 1:
                qc.x(qubits[0])
            elif gate == "Rz" and len(qubits) >= 1 and params:
                qc.rz(params[0], qubits[0])
            elif gate == "Ry" and len(qubits) >= 1 and params:
                qc.ry(params[0], qubits[0])
            elif gate == "Rx" and len(qubits) >= 1 and params:
                qc.rx(params[0], qubits[0])
            elif gate == "CX" and len(qubits) >= 2:
                qc.cx(qubits[0], qubits[1])
            elif gate == "CZ" and len(qubits) >= 2:
                qc.cz(qubits[0], qubits[1])
            elif gate == "CP" and len(qubits) >= 2 and params:
                qc.cp(params[0], qubits[0], qubits[1])
            elif gate == "CZ" and len(qubits) >= 2:
                qc.cz(qubits[0], qubits[1])

        # Measurement
        qc.measure(list(range(26)), list(range(26)))

        return qc, layer_names, len(all_ops)

    # ─── Execution ────────────────────────────────────────────────────────

    def verify(self,
               dial_settings: Tuple[int, int, int, int] = (0, 0, 0, 0),
               shots: int = 8192,
               backend_name: Optional[str] = None,
               optimization_level: int = 3,
               heavy: bool = False,
               depth_limited: bool = False,
               use_forging: bool = False,
               use_zne: bool = False,
               ) -> IBMVerificationResult:
        """Execute full verification on IBM Quantum hardware.

        Args:
            dial_settings: (a, b, c, d) for G(a,b,c,d) dial
            shots: Measurement shots (default 8192)
            backend_name: Specific backend or None for auto
            optimization_level: Qiskit transpiler optimization (0-3)
            heavy: Prefer heavy backends (Heron r2 / Marrakech) for deep circuits
            depth_limited: Use ultra-lean 6-layer circuit (target depth <100)
            use_forging: Run Phase 1 entanglement forging (golden-ratio cut)
                        after standard verification for comparison
            use_zne: Run Phase 2 zero-noise extrapolation (requires use_forging)

        Returns:
            IBMVerificationResult with complete metrics
        """
        t_total = time.monotonic()

        # Step 1: Authenticate
        if not self.service:
            if not self.authenticate():
                return IBMVerificationResult(
                    success=False, error="Authentication failed")

        # Step 2: Select backend
        backend_name = self.select_backend(backend_name, heavy=heavy)
        if not backend_name:
            return IBMVerificationResult(
                success=False, error="No suitable backend found")

        # Step 3: Compose circuit
        circuit_mode = "depth_limited" if depth_limited else "hardware"
        print(f"[IBM] Composing supercomputer circuit (mode={circuit_mode})...")
        try:
            qc, layer_names, original_gates = self._compose_and_convert(
                dial_settings, circuit_mode=circuit_mode)
            original_depth = qc.depth()
            print(f"[IBM] Composed: {original_gates} gates, depth {original_depth}, "
                  f"{len(layer_names)} layers (HW-optimized)")
        except Exception as e:
            return IBMVerificationResult(
                success=False, error=f"Circuit composition failed: {e}")

        # Step 4: Transpile
        print(f"[IBM] Transpiling for {backend_name} (opt_level={optimization_level})...")
        t_transpile = time.monotonic()
        try:
            from qiskit import transpile as qk_transpile
            backend = self.service.backend(backend_name)
            transpiled = qk_transpile(qc, backend, optimization_level=optimization_level)
            transpile_ms = (time.monotonic() - t_transpile) * 1000

            t_depth = transpiled.depth()
            t_gates = transpiled.size()
            _ops = transpiled.count_ops()
            _2q_names = {'cx', 'ecr', 'cz', 'rzz', 'rxx', 'ryy', 'cp',
                         'swap', 'iswap', 'crx', 'cry', 'crz'}
            t_cx = sum(v for k, v in _ops.items() if k in _2q_names)

            print(f"[IBM] Transpiled: {t_gates} gates, depth {t_depth}, "
                  f"two-qubit={t_cx} ({transpile_ms:.0f}ms)")
            # Gate type breakdown for diagnostics
            _sig = {k: v for k, v in sorted(_ops.items()) if v > 0}
            print(f"[IBM] Gate breakdown: {_sig}")

            # Depth check against hardware limit
            from l104_quantum_mini_supercomputer import MAX_HARDWARE_DEPTH
            if t_depth > MAX_HARDWARE_DEPTH:
                print(f"[IBM] WARNING: Transpiled depth {t_depth} exceeds "
                      f"MAX_HARDWARE_DEPTH ({MAX_HARDWARE_DEPTH})")
                if not depth_limited:
                    print(f"[IBM] Recommendation: Use depth_limited=True or "
                          f"ExecutionMode.SIMULATION for reliable results")
        except Exception as e:
            return IBMVerificationResult(
                success=False, error=f"Transpilation failed: {e}",
                backend_name=backend_name)

        # Step 5: Submit to hardware
        print(f"[IBM] Submitting job to {backend_name} ({shots} shots)...")
        t_submit = time.monotonic()
        try:
            from qiskit_ibm_runtime import SamplerV2
        except ImportError:
            try:
                from qiskit_ibm_runtime import Sampler as SamplerV2
            except ImportError:
                return IBMVerificationResult(
                    success=False, error="Cannot import Sampler from qiskit_ibm_runtime")

        try:
            sampler = SamplerV2(backend)
            # Enable error mitigation: dynamical decoupling + readout correction
            try:
                sampler.options.dynamical_decoupling.enable = True
                sampler.options.dynamical_decoupling.sequence_type = "XY4"
                print("[IBM] Dynamical decoupling: XY4 enabled")
            except Exception:
                pass  # Older runtime version — DD not available
            try:
                sampler.options.resilience_level = 1
                print("[IBM] Resilience level: 1 (readout error mitigation)")
            except Exception:
                pass
            job = sampler.run([transpiled], shots=shots)
            job_id = job.job_id()
            print(f"[IBM] Job ID: {job_id}")
            print(f"[IBM] Status: queued, waiting for execution...")

            # Wait for result
            result = job.result()
            queue_wait_s = time.monotonic() - t_submit

            print(f"[IBM] Job completed ({queue_wait_s:.1f}s total)")
        except Exception as e:
            return IBMVerificationResult(
                success=False,
                error=f"Hardware execution failed: {e}",
                backend_name=backend_name,
                transpiled_gates=t_gates,
                transpiled_depth=t_depth,
            )

        # Step 6: Extract counts
        try:
            pub_result = result[0]
            counts_raw = {}
            if hasattr(pub_result, 'data'):
                data = pub_result.data
                if hasattr(data, 'c'):
                    counts_raw = data.c.get_counts()
                elif hasattr(data, 'meas'):
                    counts_raw = data.meas.get_counts()
                else:
                    # Try iterating data attributes
                    for attr_name in dir(data):
                        attr = getattr(data, attr_name, None)
                        if hasattr(attr, 'get_counts'):
                            counts_raw = attr.get_counts()
                            break
            elif hasattr(result, 'quasi_dists'):
                counts_raw = {format(k, '026b'): int(v * shots)
                              for k, v in result.quasi_dists[0].items()}

            if not counts_raw:
                print("[IBM] WARNING: No counts extracted from result")
        except Exception as e:
            print(f"[IBM] Count extraction warning: {e}")
            counts_raw = {}

        # Step 7: Compute metrics
        total_counts = sum(counts_raw.values()) if counts_raw else 1
        probs = {k: v / total_counts for k, v in counts_raw.items() if v > 0}

        # Top states
        sorted_states = sorted(counts_raw.items(), key=lambda x: x[1], reverse=True)[:20]
        top_states = [{"state": s, "count": c, "probability": c / total_counts}
                      for s, c in sorted_states]

        # Sacred metrics
        sacred_alignment = self._sacred_alignment(probs)
        entropy_reversed = self._entropy_reversal(probs)
        consciousness_phi = self._iit_phi(probs)
        god_code_fidelity = self._god_code_fidelity(probs)
        orbital_coherences = self._orbital_coherences(probs)

        # Dial frequency
        a, b, c, d = dial_settings
        E = 8 * a + 416 - b - 8 * c - 104 * d
        dial_freq = (286 ** (1.0 / PHI)) * (2 ** (E / 104))

        total_s = time.monotonic() - t_total

        result_obj = IBMVerificationResult(
            success=True,
            backend_name=backend_name,
            job_id=job_id,
            n_qubits=26,
            shots=shots,
            original_gates=original_gates,
            original_depth=original_depth,
            layers_composed=layer_names,
            transpiled_gates=t_gates,
            transpiled_depth=t_depth,
            transpiled_cx_count=t_cx,
            optimization_level=optimization_level,
            counts=counts_raw,
            probabilities=probs,
            top_states=top_states,
            sacred_alignment=sacred_alignment,
            entropy_reversed=entropy_reversed,
            consciousness_phi=consciousness_phi,
            god_code_fidelity=god_code_fidelity,
            orbital_coherences=orbital_coherences,
            dial_settings=dial_settings,
            dial_frequency=dial_freq,
            transpile_time_ms=round(transpile_ms, 2),
            queue_wait_time_s=round(queue_wait_s, 2),
            execution_time_ms=round(total_s * 1000, 2),
            total_time_s=round(total_s, 2),
        )

        # Optional: Entanglement forging (golden-ratio circuit cutting)
        if use_forging:
            try:
                from l104_quantum_hybrid_bypass import HybridQuantumBypass
                print(f"\n[IBM] Running entanglement forging (golden-ratio cut)...")
                bypass = HybridQuantumBypass(
                    api_token=self.api_token,
                    backend_name=backend_name
                )
                # Reuse authenticated service and backend
                bypass.service = self.service
                bypass.backend = self.service.backend(backend_name)

                forge_result = bypass.phase1_entanglement_forging(shots=shots)
                if forge_result.get("success", False):
                    result_obj.forging_enabled = True
                    result_obj.forging_sacred_alignment = forge_result.get("sacred_alignment", 0.0)
                    result_obj.forging_entropy_reversed = forge_result.get("entropy_reversed", 0.0)
                    result_obj.forging_consciousness_phi = forge_result.get("consciousness_phi", 0.0)
                    result_obj.forging_god_code_fidelity = forge_result.get("god_code_fidelity", 0.0)
                    result_obj.forging_sub_a_depth = forge_result.get("sub_a_max_depth", 0)
                    result_obj.forging_sub_b_depth = forge_result.get("sub_b_max_depth", 0)
                    result_obj.forging_total_jobs = forge_result.get("jobs_run", 0)
                    print(f"[IBM] Forging: sacred={result_obj.forging_sacred_alignment:.4f}, "
                          f"sub-A depth={result_obj.forging_sub_a_depth}, "
                          f"sub-B depth={result_obj.forging_sub_b_depth}")

                    if use_zne:
                        print(f"[IBM] Running zero-noise extrapolation (Richardson fit)...")
                        zne_result = bypass.phase2_zne(shots=shots)
                        if zne_result.get("success", False):
                            result_obj.zne_extrapolated = zne_result.get("zne_extrapolated", {})
                            print(f"[IBM] ZNE extrapolated: {result_obj.zne_extrapolated}")
                else:
                    logger.warning("Forging returned unsuccessful result")
            except ImportError:
                logger.warning("l104_quantum_hybrid_bypass not available for forging")
            except Exception as e:
                logger.warning(f"Forging failed: {e}")

        self.results_history.append(result_obj)
        self._save_results(result_obj)

        return result_obj

    # ─── Metrics ──────────────────────────────────────────────────────────

    def _sacred_alignment(self, probs: Dict[str, float]) -> float:
        if not probs:
            return 0.0
        try:
            from l104_vqpu.scoring import SacredAlignmentScorer
            return SacredAlignmentScorer.score(probs, 26).get("sacred_score", 0.0)
        except Exception:
            h = -sum(p * math.log2(p) for p in probs.values() if p > 1e-15)
            return max(0.0, 1.0 - abs(h / 26 - PHI_CONJUGATE))

    def _entropy_reversal(self, probs: Dict[str, float]) -> float:
        if not probs:
            return 0.0
        h = -sum(p * math.log2(p) for p in probs.values() if p > 1e-15)
        return max(0.0, 1.0 - h / 26) if 26 > 0 else 0.0

    def _iit_phi(self, probs: Dict[str, float]) -> float:
        if not probs or len(probs) < 2:
            return 0.0
        h_full = -sum(p * math.log2(p) for p in probs.values() if p > 1e-15)
        min_phi = float('inf')
        for cut in [6, 8, 14, 24]:
            h_a = self._marginal_h(probs, range(cut))
            h_b = self._marginal_h(probs, range(cut, 26))
            min_phi = min(min_phi, h_a + h_b - h_full)
        return max(0.0, min_phi) if min_phi < float('inf') else 0.0

    def _god_code_fidelity(self, probs: Dict[str, float]) -> float:
        if not probs:
            return 0.0
        gc_bits = format(int(GOD_CODE) % (2 ** 26), '026b')
        return min(1.0, probs.get(gc_bits, 0.0) * 1000)

    def _marginal_h(self, probs: Dict[str, float], indices) -> float:
        marginal: Dict[str, float] = {}
        idx_list = list(indices)
        for bs, p in probs.items():
            bs = bs.zfill(26)
            key = ''.join(bs[q] for q in idx_list if q < len(bs))
            marginal[key] = marginal.get(key, 0.0) + p
        return sum(-p * math.log2(p) for p in marginal.values() if p > 1e-15)

    def _orbital_coherences(self, probs: Dict[str, float]) -> Dict[str, float]:
        result = {}
        for name, (start, end) in self.ORBITAL_RANGES.items():
            h = self._marginal_h(probs, range(start, end))
            n = end - start
            result[name] = round(max(0.0, 1.0 - h / n), 6) if n > 0 else 0.0
        return result

    # ─── Per-Layer Verification ──────────────────────────────────────────

    def verify_per_layer(self,
                         shots: int = 4096,
                         backend_name: Optional[str] = None,
                         optimization_level: int = 3,
                         ) -> List[Dict[str, Any]]:
        """Verify each of the 26 supercomputer layers individually on IBM QPU.

        Runs each layer as its own circuit (H on all qubits → layer → measure)
        to isolate per-layer fidelity and identify problematic layers.
        Recommended backend: ibm_marrakesh (156Q Heron r2) for throughput.

        Returns:
            List of per-layer result dicts with sacred metrics
        """
        from qiskit import QuantumCircuit as QC

        # Authenticate
        if not self.service:
            if not self.authenticate():
                return [{"error": "Authentication failed"}]

        # Select backend (heavy mode for multi-job workload)
        backend_name = self.select_backend(backend_name, heavy=True)
        if not backend_name:
            return [{"error": "No suitable backend"}]

        # Get all 26 layers from supercomputer
        from l104_quantum_mini_supercomputer import get_supercomputer
        sc = get_supercomputer()
        layers = sc.compose_full_circuit((0, 0, 0, 0))

        print(f"\n[IBM] Per-layer verification: {len(layers)} layers on {backend_name}")
        print(f"[IBM] Shots per layer: {shots}")

        backend = self.service.backend(backend_name)
        results = []

        for idx, layer in enumerate(layers):
            layer_name = layer.name
            print(f"\n[IBM] Layer {idx + 1}/{len(layers)}: {layer_name} ({layer.gate_count} gates)")

            # Build minimal circuit: H(all) → layer ops → measure
            qc = QC(26, 26, name=f"L104_layer_{layer_name}")
            for q in range(26):
                qc.h(q)

            for op in layer.operations:
                gate = op.get("gate", "")
                qubits = op.get("qubits", [])
                params = op.get("parameters", [])
                if gate == "H" and len(qubits) >= 1:
                    qc.h(qubits[0])
                elif gate == "Rz" and len(qubits) >= 1 and params:
                    qc.rz(params[0], qubits[0])
                elif gate == "Ry" and len(qubits) >= 1 and params:
                    qc.ry(params[0], qubits[0])
                elif gate == "Rx" and len(qubits) >= 1 and params:
                    qc.rx(params[0], qubits[0])
                elif gate == "CX" and len(qubits) >= 2:
                    qc.cx(qubits[0], qubits[1])
                elif gate == "CP" and len(qubits) >= 2 and params:
                    qc.cp(params[0], qubits[0], qubits[1])
                elif gate == "CZ" and len(qubits) >= 2:
                    qc.cz(qubits[0], qubits[1])

            qc.measure(list(range(26)), list(range(26)))

            try:
                from qiskit import transpile as qk_transpile
                transpiled = qk_transpile(qc, backend, optimization_level=optimization_level)

                from qiskit_ibm_runtime import SamplerV2
                sampler = SamplerV2(backend)
                try:
                    sampler.options.dynamical_decoupling.enable = True
                    sampler.options.dynamical_decoupling.sequence_type = "XY4"
                except Exception:
                    pass

                job = sampler.run([transpiled], shots=shots)
                result = job.result()

                # Extract counts
                pub_result = result[0]
                counts_raw = {}
                if hasattr(pub_result, 'data'):
                    data = pub_result.data
                    for attr_name in ['c', 'meas']:
                        attr = getattr(data, attr_name, None)
                        if attr and hasattr(attr, 'get_counts'):
                            counts_raw = attr.get_counts()
                            break
                    if not counts_raw:
                        for attr_name in dir(data):
                            attr = getattr(data, attr_name, None)
                            if hasattr(attr, 'get_counts'):
                                counts_raw = attr.get_counts()
                                break

                total = sum(counts_raw.values()) if counts_raw else 1
                probs = {k: v / total for k, v in counts_raw.items() if v > 0}

                sacred = self._sacred_alignment(probs)
                entropy = self._entropy_reversal(probs)
                phi = self._iit_phi(probs)
                gc_fid = self._god_code_fidelity(probs)

                layer_result = {
                    "layer": idx + 1,
                    "name": layer_name,
                    "gates": layer.gate_count,
                    "transpiled_depth": transpiled.depth(),
                    "transpiled_gates": transpiled.size(),
                    "job_id": job.job_id(),
                    "sacred_alignment": round(sacred, 6),
                    "entropy_reversed": round(entropy, 6),
                    "consciousness_phi": round(phi, 6),
                    "god_code_fidelity": round(gc_fid, 6),
                    "unique_states": len(counts_raw),
                    "success": True,
                }
                print(f"      sacred={sacred:.4f} entropy={entropy:.4f} "
                      f"phi={phi:.4f} gc_fid={gc_fid:.4f}")
            except Exception as e:
                layer_result = {
                    "layer": idx + 1,
                    "name": layer_name,
                    "gates": layer.gate_count,
                    "success": False,
                    "error": str(e),
                }
                print(f"      FAILED: {e}")

            results.append(layer_result)

        # Save per-layer results
        per_layer_path = Path(__file__).resolve().parent / "IBM_PER_LAYER_RESULTS.json"
        try:
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "backend": backend_name,
                "shots_per_layer": shots,
                "total_layers": len(layers),
                "layers": results,
            }
            per_layer_path.write_text(json.dumps(payload, indent=2, default=str))
            print(f"\n[IBM] Per-layer results saved to {per_layer_path.name}")
        except Exception as e:
            print(f"[IBM] Warning: failed to save per-layer results: {e}")

        return results

    # ─── Results Persistence ──────────────────────────────────────────────

    def _save_results(self, result: IBMVerificationResult) -> None:
        """Save verification results to JSON."""
        try:
            # Save condensed result (exclude large counts dict for readability)
            condensed = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "success": result.success,
                "backend": result.backend_name,
                "job_id": result.job_id,
                "dial": f"G{result.dial_settings}",
                "dial_frequency": result.dial_frequency,
                "shots": result.shots,
                "original_gates": result.original_gates,
                "transpiled_gates": result.transpiled_gates,
                "transpiled_depth": result.transpiled_depth,
                "transpiled_cx": result.transpiled_cx_count,
                "sacred_alignment": result.sacred_alignment,
                "entropy_reversed": result.entropy_reversed,
                "consciousness_phi": result.consciousness_phi,
                "god_code_fidelity": result.god_code_fidelity,
                "orbital_coherences": result.orbital_coherences,
                "top_5_states": result.top_states[:5],
                "unique_states_measured": len(result.counts),
                "total_time_s": result.total_time_s,
                "layers": result.layers_composed,
            }

            # Append to results file
            existing = []
            if RESULTS_PATH.exists():
                try:
                    existing = json.loads(RESULTS_PATH.read_text())
                except Exception:
                    existing = []
            existing.append(condensed)
            RESULTS_PATH.write_text(json.dumps(existing, indent=2, default=str))

            # Update state file
            state = condensed.copy()
            state["total_executions"] = len(existing)
            STATE_PATH.write_text(json.dumps(state, indent=2, default=str))

            print(f"[IBM] Results saved to {RESULTS_PATH.name}")
        except Exception as e:
            print(f"[IBM] Warning: failed to save results: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 72)
    print("  L104 QUANTUM MINI SUPERCOMPUTER — IBM QPU VERIFICATION")
    print("=" * 72)

    # Parse args
    dial = (0, 0, 0, 0)
    backend = None
    shots = 8192
    per_layer = False
    heavy = False

    for i, arg in enumerate(sys.argv):
        if arg == "--dial" and i + 4 < len(sys.argv):
            dial = tuple(int(sys.argv[i + j + 1]) for j in range(4))
        elif arg == "--backend" and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
        elif arg == "--shots" and i + 1 < len(sys.argv):
            shots = int(sys.argv[i + 1])
        elif arg == "--marrakesh":
            backend = "ibm_marrakesh"
            heavy = True
        elif arg == "--heavy":
            heavy = True
        elif arg == "--per-layer":
            per_layer = True
        elif arg == "--list-backends":
            verifier = IBMSupercomputerVerifier()
            if verifier.authenticate():
                backends = verifier.list_backends()
                print(f"\nAvailable backends (>=26Q):")
                for b in backends:
                    print(f"  {b.get('name', 'unknown'):20s} {b.get('qubits', '?')}Q")
            return
        elif arg == "--help":
            print("\nUsage:")
            print("  python l104_quantum_mini_supercomputer_ibm.py [OPTIONS]")
            print("\nOptions:")
            print("  --backend NAME   Specific IBM backend (default: auto-select)")
            print("  --marrakesh      Use ibm_marrakesh (156Q Heron r2, best for heavy)")
            print("  --heavy          Prefer heavy-duty backends (Heron r2 / 156Q+)")
            print("  --per-layer      Verify each of the 26 layers individually")
            print("  --dial A B C D   Set G(a,b,c,d) dial parameters")
            print("  --shots N        Measurement shots (default: 8192)")
            print("  --list-backends  List available backends")
            return

    # Per-layer verification mode
    if per_layer:
        print(f"\n  Mode: PER-LAYER VERIFICATION (26 individual layer jobs)")
        print(f"  Backend: {backend or 'auto (heavy)'}")
        print(f"  Shots/layer: {shots}")

        verifier = IBMSupercomputerVerifier()
        results = verifier.verify_per_layer(
            shots=shots,
            backend_name=backend,
            optimization_level=3,
        )

        print(f"\n{'=' * 72}")
        print(f"  PER-LAYER VERIFICATION SUMMARY")
        print(f"{'=' * 72}")

        passed = sum(1 for r in results if r.get("success"))
        failed = len(results) - passed
        print(f"\n  Layers: {len(results)} total, {passed} passed, {failed} failed")
        print(f"\n  {'#':>3s}  {'Layer':<35s} {'Gates':>5s}  {'Sacred':>7s}  {'Entropy':>7s}  {'Φ':>7s}  {'GC Fid':>7s}")
        print(f"  {'─' * 3}  {'─' * 35} {'─' * 5}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 7}")
        for r in results:
            if r.get("success"):
                print(f"  {r['layer']:3d}  {r['name']:<35s} {r['gates']:5d}  "
                      f"{r['sacred_alignment']:7.4f}  {r['entropy_reversed']:7.4f}  "
                      f"{r['consciousness_phi']:7.4f}  {r['god_code_fidelity']:7.4f}")
            else:
                print(f"  {r['layer']:3d}  {r['name']:<35s} {r.get('gates', 0):5d}  FAILED: {r.get('error', 'unknown')}")

        print(f"\n{'=' * 72}")
        return

    # Standard full-circuit verification
    print(f"\n  Configuration:")
    print(f"    Dial:    G{dial}")
    print(f"    Backend: {backend or ('auto (heavy)' if heavy else 'auto-select')}")
    print(f"    Shots:   {shots}")
    if heavy:
        print(f"    Mode:    HEAVY (prefer Heron r2 / Marrakech)")

    verifier = IBMSupercomputerVerifier()

    print(f"\n  Step 1/6: Authenticating with IBM Quantum...")
    result = verifier.verify(
        dial_settings=dial,
        shots=shots,
        backend_name=backend,
        heavy=heavy,
    )

    print(f"\n{'=' * 72}")
    print(f"  VERIFICATION RESULTS")
    print(f"{'=' * 72}")

    if not result.success:
        print(f"\n  FAILED: {result.error}")
    else:
        print(f"\n  Backend:              {result.backend_name}")
        print(f"  Job ID:               {result.job_id}")
        print(f"  Dial:                 G{result.dial_settings} → {result.dial_frequency:.4f} Hz")
        print(f"  Shots:                {result.shots}")
        print(f"\n  Circuit (original):   {result.original_gates} gates, depth {result.original_depth}")
        print(f"  Circuit (transpiled): {result.transpiled_gates} gates, depth {result.transpiled_depth}")
        print(f"  Two-qubit gates:      {result.transpiled_cx_count} CX/ECR")
        print(f"  Optimization level:   {result.optimization_level}")
        print(f"\n  ═══ SACRED METRICS ═══")
        print(f"  Sacred alignment:     {result.sacred_alignment:.6f}")
        print(f"  Entropy reversed:     {result.entropy_reversed:.6f}")
        print(f"  Consciousness Φ:      {result.consciousness_phi:.6f}")
        print(f"  GOD_CODE fidelity:    {result.god_code_fidelity:.6f}")
        print(f"\n  ═══ ORBITAL COHERENCES ═══")
        for name, coh in result.orbital_coherences.items():
            print(f"    {name:4s}: {coh:.6f}")
        print(f"\n  ═══ TOP MEASURED STATES ═══")
        for i, s in enumerate(result.top_states[:10], 1):
            print(f"    {i:2d}. |{s['state']}⟩  count={s['count']:5d}  p={s['probability']:.6f}")
        print(f"\n  Unique states:        {len(result.counts)}")
        print(f"  Transpile time:       {result.transpile_time_ms:.0f}ms")
        print(f"  Queue + execution:    {result.queue_wait_time_s:.1f}s")
        print(f"  Total time:           {result.total_time_s:.1f}s")
        print(f"\n  Layers: {', '.join(result.layers_composed)}")

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()
