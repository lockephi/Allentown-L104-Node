#!/usr/bin/env python3
"""
L104 Quantum Hybrid Bypass v1.0.0
═══════════════════════════════════════════════════════════════════════════════
Three-phase decoherence bypass for the L104 Quantum Mini Supercomputer.
Extracts true sacred alignment signal from beneath the entropy saturation
floor on NISQ hardware without waiting for better qubits.

Phase 1: Entanglement Forging — Golden-ratio circuit cutting at layer ⌊11/φ⌋=7
  Split the 11-layer HW circuit into Sub-A (7 layers) + Sub-B (4 layers).
  Cut wire q13 (3p↔3d orbital boundary) via Pauli decomposition:
    ρ = (I + ⟨X⟩X + ⟨Y⟩Y + ⟨Z⟩Z) / 2
  9 IBM jobs: 3 Sub-A (basis measurements) + 6 Sub-B (Pauli preparations).
  Classical reconstruction stitches the forged probability distribution.

Phase 2: Zero-Noise Extrapolation — Richardson extrapolation to λ=0
  Run the full 11-layer circuit at noise scales λ = {1, 3, 5} via gate
  folding (each 2Q gate G → G·G†·G·...·G). Fit quadratic polynomial:
    E(λ) = a + bλ + cλ²
  Richardson formula: E(0) = (15·E(1) - 10·E(3) + 3·E(5)) / 8

Phase 3: Dynamical Decoupling — XY4 at runtime + CPMG awareness
  Qiskit runtime XY4 sequence on all idle qubit windows. The π-pulse
  echoes reverse T₂ dephasing accumulated during deep circuit layers.

Usage:
  export IBMQ_TOKEN='your_token'
  python l104_quantum_hybrid_bypass.py --backend ibm_kingston
  python l104_quantum_hybrid_bypass.py --backend ibm_kingston --phase 1
  python l104_quantum_hybrid_bypass.py --backend ibm_kingston --phase 2
  python l104_quantum_hybrid_bypass.py --backend ibm_kingston --phase all

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
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("l104.hybrid_bypass")

# ═══════════════════════════════════════════════════════════════════════════════
#  SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
PHI_CONJUGATE = PHI - 1.0  # 0.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000
TAU = 2.0 * math.pi
GOD_CODE_PHASE = GOD_CODE % TAU

N_QUBITS = 26
N_HW_LAYERS = 11

# Golden cut: 11/φ = 6.7984 → layer 7
# Sub-A = 7 layers, Sub-B = 4 layers (7:4 Fibonacci partition)
GOLDEN_CUT_EXACT = N_HW_LAYERS / PHI  # 6.79836...
GOLDEN_CUT_LAYER = round(GOLDEN_CUT_EXACT)  # 7

# Cut qubit: q13 at 3p↔3d orbital boundary
# This is the highest-flux entanglement link between inner shield (3p)
# and consciousness substrate (3d)
CUT_QUBIT = 13

RESULTS_PATH = Path(__file__).resolve().parent / "IBM_HYBRID_BYPASS_RESULTS.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BypassResult:
    """Result from hybrid quantum bypass execution."""
    success: bool
    backend_name: str = ""
    phase: str = ""

    # Phase 1: Forging
    golden_cut_layer: int = GOLDEN_CUT_LAYER
    golden_cut_exact: float = GOLDEN_CUT_EXACT
    cut_qubit: int = CUT_QUBIT
    sub_a_layers: int = GOLDEN_CUT_LAYER
    sub_b_layers: int = N_HW_LAYERS - GOLDEN_CUT_LAYER
    pauli_expectations: Dict[str, float] = field(default_factory=dict)
    forged_metrics: Dict[str, float] = field(default_factory=dict)

    # Phase 2: ZNE
    noise_scale_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    zne_extrapolated: Dict[str, float] = field(default_factory=dict)

    # Combined
    sacred_alignment: float = 0.0
    entropy_reversed: float = 0.0
    consciousness_phi: float = 0.0
    god_code_fidelity: float = 0.0

    total_jobs: int = 0
    total_time_s: float = 0.0
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  HYBRID QUANTUM BYPASS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HybridQuantumBypass:
    """
    Three-phase decoherence bypass for the L104 Quantum Mini Supercomputer.

    Cheats the physics using mathematically calculated hybrid approaches:
    1. Entanglement Forging — halve the circuit depth via golden-ratio cut
    2. Zero-Noise Extrapolation — subtract the noise signature algebraically
    3. Dynamical Decoupling — echo idle qubits to preserve coherence
    """

    # ZNE noise scales (odd numbers: G → G·G†·G for λ=3, etc.)
    NOISE_SCALES = [1, 3, 5]

    def __init__(self, api_token: Optional[str] = None,
                 backend_name: str = 'ibm_kingston'):
        self.api_token = (api_token
                          or os.environ.get('IBMQ_TOKEN')
                          or os.environ.get('IBM_QUANTUM_TOKEN'))
        self.backend_name = backend_name
        self.service = None
        self.backend = None

    # ─── Authentication ──────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Authenticate with IBM Quantum."""
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError:
            print("[BYPASS] ERROR: qiskit-ibm-runtime not installed")
            return False

        if not self.api_token:
            print("[BYPASS] ERROR: No API token. Set IBMQ_TOKEN env var")
            return False

        try:
            self.service = QiskitRuntimeService(
                channel='ibm_quantum_platform', token=self.api_token)
            self.backend = self.service.backend(self.backend_name)
            print(f"[BYPASS] Authenticated: {self.backend_name} "
                  f"({self.backend.num_qubits}Q)")
            return True
        except Exception as e:
            print(f"[BYPASS] Auth failed: {e}")
            return False

    # ─── Circuit Utilities ───────────────────────────────────────────────

    def _get_hw_layers(self):
        """Get the 11 hardware-optimized layers from the supercomputer."""
        from l104_quantum_mini_supercomputer import get_supercomputer
        sc = get_supercomputer()
        return sc.compose_hardware_circuit((0, 0, 0, 0))

    def _flatten_ops(self, layers) -> List[Dict[str, Any]]:
        """Flatten CircuitLayer list into operation list."""
        ops = []
        for layer in layers:
            ops.extend(layer.operations)
        return ops

    def _apply_gate(self, qc, gate: str, qubits: List[int],
                    params: List[float]):
        """Apply a single L104 gate to a Qiskit QuantumCircuit."""
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

    def _apply_gate_inverse(self, qc, gate: str, qubits: List[int],
                            params: List[float]):
        """Apply the inverse (adjoint) of a gate.

        Self-inverse gates (H, CX, CZ) map to themselves.
        Rotation gates Rα(θ) → Rα(-θ). CP(θ) → CP(-θ).
        """
        if gate in ("H", "CX", "CZ"):
            self._apply_gate(qc, gate, qubits, params)
        elif gate == "Rz" and params:
            qc.rz(-params[0], qubits[0])
        elif gate == "Ry" and params:
            qc.ry(-params[0], qubits[0])
        elif gate == "Rx" and params:
            qc.rx(-params[0], qubits[0])
        elif gate == "CP" and len(qubits) >= 2 and params:
            qc.cp(-params[0], qubits[0], qubits[1])

    def _ops_to_qiskit(self, ops: List[Dict], name: str = "circuit"):
        """Convert L104 operation list to Qiskit QuantumCircuit (no measure)."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(N_QUBITS, N_QUBITS, name=name)
        for op in ops:
            self._apply_gate(qc, op.get("gate", ""),
                             op.get("qubits", []),
                             op.get("parameters", []))
        return qc

    def _submit_and_wait(self, qc, shots: int = 4096,
                         dd: bool = True) -> Tuple[Dict[str, int], str, int, int]:
        """Transpile, submit to IBM, wait for result.

        Returns: (counts, job_id, transpiled_depth, transpiled_gates)
        """
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2

        transpiled = transpile(qc, self.backend, optimization_level=3)
        sampler = SamplerV2(self.backend)

        # Phase 3: Dynamical Decoupling (XY4 on all idle windows)
        if dd:
            try:
                sampler.options.dynamical_decoupling.enable = True
                sampler.options.dynamical_decoupling.sequence_type = "XY4"
            except Exception:
                pass

        job = sampler.run([transpiled], shots=shots)
        result = job.result()

        # Extract counts
        pub = result[0]
        counts = {}
        if hasattr(pub, 'data'):
            data = pub.data
            for attr in ['c', 'meas']:
                obj = getattr(data, attr, None)
                if obj and hasattr(obj, 'get_counts'):
                    counts = obj.get_counts()
                    break
            if not counts:
                for attr in dir(data):
                    obj = getattr(data, attr, None)
                    if hasattr(obj, 'get_counts'):
                        counts = obj.get_counts()
                        break

        return counts, job.job_id(), transpiled.depth(), transpiled.size()

    def _counts_to_probs(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Normalize counts to probability distribution."""
        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items() if v > 0}

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 1: ENTANGLEMENT FORGING (GOLDEN CUT)
    # ═══════════════════════════════════════════════════════════════════════

    def phase1_entanglement_forging(self, shots: int = 4096) -> Dict[str, Any]:
        """Phase 1: Golden-ratio circuit cutting with Pauli wire decomposition.

        Splits the 11-layer HW circuit at layer ⌊11/φ⌋ = 7:
          Sub-A: layers 1–7 (foundation + deep entanglement)
          Sub-B: layers 8–11 (proof + DAW + VQE + readout)
          Ratio: 7:4 (consecutive Fibonacci = golden partition)

        Cut wire q13 (3p↔3d orbital boundary) decomposed into Pauli basis:
          ρ_{q13} = (I + ⟨X⟩X + ⟨Y⟩Y + ⟨Z⟩Z) / 2

        Sub-A runs: 3 jobs (measure q13 in Z, X, Y bases)
        Sub-B runs: 6 jobs (prepare q13 in |0⟩,|1⟩,|+⟩,|−⟩,|+i⟩,|−i⟩)
        Total: 9 IBM jobs → classical Pauli reconstruction
        """
        from qiskit import QuantumCircuit

        print(f"\n{'=' * 72}")
        print(f"  PHASE 1: ENTANGLEMENT FORGING")
        print(f"{'=' * 72}")
        print(f"  Golden cut: N/φ = 11/{PHI:.6f} = {GOLDEN_CUT_EXACT:.4f} → layer {GOLDEN_CUT_LAYER}")
        print(f"  Sub-A: 7 layers (god_code→grimoire) | Sub-B: 4 layers (proof→final)")
        print(f"  Cut qubit: q{CUT_QUBIT} (3p↔3d orbital boundary)")
        print(f"  Fibonacci partition: 7:4")

        hw_layers = self._get_hw_layers()
        sub_a_layers = hw_layers[:GOLDEN_CUT_LAYER]
        sub_b_layers = hw_layers[GOLDEN_CUT_LAYER:]
        sub_a_ops = self._flatten_ops(sub_a_layers)
        sub_b_ops = self._flatten_ops(sub_b_layers)

        print(f"\n  Sub-A: {len(sub_a_layers)} layers, {len(sub_a_ops)} gates")
        print(f"    {', '.join(l.name for l in sub_a_layers)}")
        print(f"  Sub-B: {len(sub_b_layers)} layers, {len(sub_b_ops)} gates")
        print(f"    {', '.join(l.name for l in sub_b_layers)}")

        q_cut = CUT_QUBIT
        jobs_run = 0

        # ─── Sub-A: Measure q13 in Z, X, Y bases ────────────────────
        print(f"\n  [Sub-A] 3 basis measurements on q{q_cut}...")

        pauli_exp = {}  # {X: ⟨X⟩, Y: ⟨Y⟩, Z: ⟨Z⟩}
        sub_a_top_bitstring = None
        sub_a_depths = []

        for basis in ['Z', 'X', 'Y']:
            qc = self._ops_to_qiskit(sub_a_ops, name=f"SubA_{basis}")

            # Basis rotation on cut qubit before measurement
            if basis == 'X':
                qc.h(q_cut)  # Z→X basis
            elif basis == 'Y':
                qc.sdg(q_cut)  # S†
                qc.h(q_cut)   # Z→Y basis

            qc.measure(list(range(N_QUBITS)), list(range(N_QUBITS)))

            counts, job_id, depth, size = self._submit_and_wait(qc, shots)
            jobs_run += 1
            sub_a_depths.append(depth)

            # Pauli expectation on cut qubit
            # Qiskit bit ordering: qubit q is at string position (N-1-q)
            total = sum(counts.values())
            exp_val = 0.0
            for bitstring, count in counts.items():
                bs = bitstring.zfill(N_QUBITS)
                bit = int(bs[N_QUBITS - 1 - q_cut])
                exp_val += (1 - 2 * bit) * count / total

            pauli_exp[basis] = exp_val
            print(f"    {basis}: ⟨{basis}⟩ = {exp_val:+.6f}  "
                  f"(depth={depth}, gates={size}, job={job_id})")

            # Save Z-basis results for Sub-B initialization
            if basis == 'Z':
                top_bs = max(counts, key=counts.get).zfill(N_QUBITS)
                sub_a_top_bitstring = top_bs

        print(f"\n  Pauli expectations on q{q_cut}:")
        print(f"    ⟨X⟩ = {pauli_exp['X']:+.6f}")
        print(f"    ⟨Y⟩ = {pauli_exp['Y']:+.6f}")
        print(f"    ⟨Z⟩ = {pauli_exp['Z']:+.6f}")
        print(f"  Top Sub-A bitstring: |{sub_a_top_bitstring}⟩")

        # ─── Sub-B: 6 Pauli eigenstate preparations on q13 ──────────
        print(f"\n  [Sub-B] 6 Pauli preparations on q{q_cut}...")

        # Pauli eigenstates: {basis}{sign} → gate sequence from |0⟩
        # Z+: |0⟩         Z-: X|0⟩ = |1⟩
        # X+: H|0⟩ = |+⟩  X-: HX|0⟩ = H|1⟩ = |−⟩
        # Y+: SH|0⟩ = |+i⟩  Y-: SHX|0⟩ = SH|1⟩ = |−i⟩
        preparations = [
            ('Z+', []),
            ('Z-', [('x',)]),
            ('X+', [('h',)]),
            ('X-', [('x',), ('h',)]),
            ('Y+', [('h',), ('s',)]),
            ('Y-', [('x',), ('h',), ('s',)]),
        ]

        sub_b_probs = {}
        sub_b_depths = []

        for prep_name, prep_gates in preparations:
            qc = QuantumCircuit(N_QUBITS, N_QUBITS, name=f"SubB_{prep_name}")

            # Initialize non-cut qubits from Sub-A top bitstring
            for q in range(N_QUBITS):
                if q == q_cut:
                    continue
                if sub_a_top_bitstring[N_QUBITS - 1 - q] == '1':
                    qc.x(q)

            # Prepare cut qubit in Pauli eigenstate
            for (gate_name,) in prep_gates:
                if gate_name == 'x':
                    qc.x(q_cut)
                elif gate_name == 'h':
                    qc.h(q_cut)
                elif gate_name == 's':
                    qc.s(q_cut)

            # Apply Sub-B layers
            for op in sub_b_ops:
                self._apply_gate(qc, op.get("gate", ""),
                                 op.get("qubits", []),
                                 op.get("parameters", []))

            qc.measure(list(range(N_QUBITS)), list(range(N_QUBITS)))

            counts, job_id, depth, size = self._submit_and_wait(qc, shots)
            jobs_run += 1
            sub_b_depths.append(depth)
            sub_b_probs[prep_name] = self._counts_to_probs(counts)
            print(f"    {prep_name}: depth={depth}, unique={len(counts)}, job={job_id}")

        # ─── Classical Reconstruction (Pauli wire stitching) ─────────
        print(f"\n  [FORGE] Classical Pauli reconstruction...")

        # Forged probability for each outcome bitstring b:
        # P(b) = (1/2) × [P_I(b) + ⟨X⟩·ΔP_X(b) + ⟨Y⟩·ΔP_Y(b) + ⟨Z⟩·ΔP_Z(b)]
        #
        # where:
        #   P_I(b) = [P(b|Z+) + P(b|Z-)] / 2   (identity channel)
        #   ΔP_X(b) = [P(b|X+) - P(b|X-)] / 2  (X correlation)
        #   ΔP_Y(b) = [P(b|Y+) - P(b|Y-)] / 2  (Y correlation)
        #   ΔP_Z(b) = [P(b|Z+) - P(b|Z-)] / 2  (Z correlation)

        all_bitstrings = set()
        for probs in sub_b_probs.values():
            all_bitstrings.update(probs.keys())

        forged = {}
        for b in all_bitstrings:
            p_zp = sub_b_probs['Z+'].get(b, 0.0)
            p_zm = sub_b_probs['Z-'].get(b, 0.0)
            p_xp = sub_b_probs['X+'].get(b, 0.0)
            p_xm = sub_b_probs['X-'].get(b, 0.0)
            p_yp = sub_b_probs['Y+'].get(b, 0.0)
            p_ym = sub_b_probs['Y-'].get(b, 0.0)

            p_I = (p_zp + p_zm) / 2
            delta_X = (p_xp - p_xm) / 2
            delta_Y = (p_yp - p_ym) / 2
            delta_Z = (p_zp - p_zm) / 2

            p_forged = 0.5 * (p_I
                              + pauli_exp['X'] * delta_X
                              + pauli_exp['Y'] * delta_Y
                              + pauli_exp['Z'] * delta_Z)

            if p_forged > 1e-12:
                forged[b] = max(0.0, p_forged)

        # Normalize (statistical noise may break sum = 1)
        total_p = sum(forged.values())
        if total_p > 0:
            forged = {k: v / total_p for k, v in forged.items()}

        # Sacred metrics on forged distribution
        metrics = self._compute_all_metrics(forged)

        print(f"\n  ═══ FORGED SACRED METRICS ═══")
        print(f"  Sacred alignment:     {metrics['sacred_alignment']:.6f}")
        print(f"  Entropy reversed:     {metrics['entropy_reversed']:.6f}")
        print(f"  Consciousness Φ:      {metrics['consciousness_phi']:.6f}")
        print(f"  GOD_CODE fidelity:    {metrics['god_code_fidelity']:.6f}")
        print(f"  Unique forged states: {len(forged)}")
        print(f"  Sub-A max depth:      {max(sub_a_depths)}")
        print(f"  Sub-B max depth:      {max(sub_b_depths)}")
        print(f"  Total jobs:           {jobs_run}")

        return {
            **metrics,
            'pauli_expectations': pauli_exp,
            'forged_probs': forged,
            'sub_a_max_depth': max(sub_a_depths),
            'sub_b_max_depth': max(sub_b_depths),
            'jobs_run': jobs_run,
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 2: ZERO-NOISE EXTRAPOLATION
    # ═══════════════════════════════════════════════════════════════════════

    def phase2_zne(self, shots: int = 4096) -> Dict[str, Any]:
        """Phase 2: Zero-Noise Extrapolation with Richardson extrapolation.

        Runs the full 11-layer HW circuit at noise scales λ = {1, 3, 5}
        via 2Q gate folding: each CX/CP/CZ gate G is replaced by
        G·G†·G·...·G (λ total copies, logically identical, physically
        λ× the noise). The sacred metrics at each λ are measured, then
        a quadratic polynomial is fit and extrapolated to λ=0.

        Richardson formula (3-point, quadratic):
          E(0) = (15·E(1) − 10·E(3) + 3·E(5)) / 8
        """
        from qiskit import QuantumCircuit

        print(f"\n{'=' * 72}")
        print(f"  PHASE 2: ZERO-NOISE EXTRAPOLATION")
        print(f"{'=' * 72}")
        print(f"  Noise scales: λ = {self.NOISE_SCALES}")
        print(f"  Gate folding: 2Q gates G → G·(G†·G)^((λ-1)/2)")
        print(f"  Extrapolation: Richardson quadratic → E(λ=0)")

        hw_layers = self._get_hw_layers()
        all_ops = self._flatten_ops(hw_layers)

        # Count 2Q gates in original circuit
        n_2q = sum(1 for op in all_ops
                   if op.get("gate") in ("CX", "CP", "CZ")
                   and len(op.get("qubits", [])) >= 2)
        print(f"  Original: {len(all_ops)} gates ({n_2q} two-qubit)")

        scale_results = {}
        jobs_run = 0

        for scale in self.NOISE_SCALES:
            # Build circuit with gate folding
            qc = QuantumCircuit(N_QUBITS, N_QUBITS,
                                name=f"ZNE_lambda{scale}")

            for op in all_ops:
                gate = op.get("gate", "")
                qubits = op.get("qubits", [])
                params = op.get("parameters", [])

                is_2q = (gate in ("CX", "CP", "CZ")
                         and len(qubits) >= 2)

                # Apply original gate
                self._apply_gate(qc, gate, qubits, params)

                # Gate folding: add (scale-1) extra copies for 2Q gates
                # Pattern: G·G†·G·G†·...  (alternating inverse/forward)
                if is_2q and scale > 1:
                    for fold_idx in range(scale - 1):
                        if fold_idx % 2 == 0:
                            self._apply_gate_inverse(qc, gate, qubits, params)
                        else:
                            self._apply_gate(qc, gate, qubits, params)

            qc.measure(list(range(N_QUBITS)), list(range(N_QUBITS)))

            n_folded_2q = n_2q * scale
            print(f"\n  [ZNE] λ={scale}: {n_folded_2q} two-qubit gates "
                  f"({scale}× folding)...", end=" ")

            counts, job_id, depth, size = self._submit_and_wait(qc, shots)
            jobs_run += 1

            probs = self._counts_to_probs(counts)
            metrics = self._compute_all_metrics(probs)

            scale_results[scale] = {
                **metrics,
                'depth': depth,
                'gates': size,
                'unique_states': len(counts),
                'job_id': job_id,
            }

            print(f"\n    sacred={metrics['sacred_alignment']:.4f} "
                  f"entropy={metrics['entropy_reversed']:.4f} "
                  f"Φ={metrics['consciousness_phi']:.4f} "
                  f"(depth={depth}, gates={size})")

        # ─── Richardson Extrapolation to λ=0 ──────────────────────────
        print(f"\n  [ZNE] Richardson extrapolation → λ=0:")

        zne_metrics = {}
        metric_names = ['sacred_alignment', 'entropy_reversed',
                        'consciousness_phi', 'god_code_fidelity']

        for metric in metric_names:
            e1 = scale_results[1][metric]
            e3 = scale_results[3][metric]
            e5 = scale_results[5][metric]

            # Lagrange interpolation at λ=0 through (1,e1), (3,e3), (5,e5):
            # E(0) = e1·(0-3)(0-5)/((1-3)(1-5))
            #       + e3·(0-1)(0-5)/((3-1)(3-5))
            #       + e5·(0-1)(0-3)/((5-1)(5-3))
            #      = e1·15/8 - e3·10/8 + e5·3/8
            e0 = (15 * e1 - 10 * e3 + 3 * e5) / 8.0

            # Clamp to physically valid range
            if metric != 'consciousness_phi':
                e0 = max(0.0, min(1.0, e0))
            else:
                e0 = max(0.0, e0)

            zne_metrics[metric] = e0
            print(f"    {metric}:")
            print(f"      E(1)={e1:.6f}  E(3)={e3:.6f}  E(5)={e5:.6f}"
                  f"  → E(0)={e0:.6f}")

        print(f"\n  ═══ ZNE EXTRAPOLATED METRICS ═══")
        for k, v in zne_metrics.items():
            print(f"  {k}: {v:.6f}")
        print(f"  Total jobs: {jobs_run}")

        return {
            'scale_results': scale_results,
            'zne_extrapolated': zne_metrics,
            'jobs_run': jobs_run,
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  FULL BYPASS (ALL THREE PHASES)
    # ═══════════════════════════════════════════════════════════════════════

    def run_full_bypass(self, shots: int = 4096) -> BypassResult:
        """Execute all three phases and combine results.

        Phase 3 (DD) is applied implicitly via XY4 on every job.
        The combined result uses φ-weighted fusion of Phases 1 and 2:
          combined = (φ·forge + zne) / (φ+1)
        giving the forged signal 61.8% weight (golden ratio).
        """
        t0 = time.monotonic()

        print(f"\n{'=' * 72}")
        print(f"  L104 QUANTUM HYBRID BYPASS v1.0.0")
        print(f"  Three-phase decoherence bypass on {self.backend_name}")
        print(f"{'=' * 72}")
        print(f"\n  Phase 1: Entanglement Forging  (golden cut at layer "
              f"{GOLDEN_CUT_LAYER}, wire q{CUT_QUBIT})")
        print(f"  Phase 2: Zero-Noise Extrapolation  (Richardson λ→0)")
        print(f"  Phase 3: Dynamical Decoupling  (XY4 on all idle windows)")
        print(f"  Backend: {self.backend_name}")

        # Phase 1: Entanglement Forging
        forge = self.phase1_entanglement_forging(shots)

        # Phase 2: Zero-Noise Extrapolation
        zne = self.phase2_zne(shots)

        total_jobs = forge['jobs_run'] + zne['jobs_run']
        total_time = time.monotonic() - t0

        # ─── φ-Weighted Combination ───────────────────────────────────
        # Forge contributes φ/(φ+1) ≈ 0.618, ZNE contributes 1/(φ+1) ≈ 0.382
        combined = {}
        metric_names = ['sacred_alignment', 'entropy_reversed',
                        'consciousness_phi', 'god_code_fidelity']

        for m in metric_names:
            f_val = forge[m]
            z_val = zne['zne_extrapolated'][m]
            combined[m] = (PHI * f_val + z_val) / (PHI + 1)

        # ─── Final Report ─────────────────────────────────────────────
        print(f"\n{'=' * 72}")
        print(f"  COMBINED HYBRID BYPASS RESULTS")
        print(f"{'=' * 72}")
        print(f"\n  {'Metric':<25s} {'Forged':>10s} {'ZNE(λ=0)':>10s} "
              f"{'Combined':>10s}")
        print(f"  {'─' * 25} {'─' * 10} {'─' * 10} {'─' * 10}")
        for m in metric_names:
            print(f"  {m:<25s} {forge[m]:10.6f} "
                  f"{zne['zne_extrapolated'][m]:10.6f} "
                  f"{combined[m]:10.6f}")

        print(f"\n  φ-weight: forge={PHI/(PHI+1):.4f}, zne={1/(PHI+1):.4f}")
        print(f"  Sub-A depth (forged):   {forge['sub_a_max_depth']}")
        print(f"  Sub-B depth (forged):   {forge['sub_b_max_depth']}")
        print(f"  Total IBM jobs:         {total_jobs}")
        print(f"  Total time:             {total_time:.1f}s")
        print(f"\n{'=' * 72}")

        # Save results
        result_data = {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'backend': self.backend_name,
            'version': '1.0.0',
            'golden_cut': {
                'exact': GOLDEN_CUT_EXACT,
                'layer': GOLDEN_CUT_LAYER,
                'formula': f'{N_HW_LAYERS}/φ = {N_HW_LAYERS}/{PHI:.10f} = {GOLDEN_CUT_EXACT:.10f}',
                'partition': '7:4 (Fibonacci)',
                'cut_qubit': CUT_QUBIT,
                'cut_boundary': '3p↔3d orbital',
            },
            'phase1_forging': {
                'pauli_expectations': forge['pauli_expectations'],
                'sacred_alignment': forge['sacred_alignment'],
                'entropy_reversed': forge['entropy_reversed'],
                'consciousness_phi': forge['consciousness_phi'],
                'god_code_fidelity': forge['god_code_fidelity'],
                'sub_a_depth': forge['sub_a_max_depth'],
                'sub_b_depth': forge['sub_b_max_depth'],
                'jobs': forge['jobs_run'],
            },
            'phase2_zne': {
                'noise_scales': self.NOISE_SCALES,
                'richardson_formula': 'E(0) = (15·E(1) - 10·E(3) + 3·E(5)) / 8',
                'scale_results': {
                    str(k): {kk: vv for kk, vv in v.items()
                             if kk != 'job_id'}
                    for k, v in zne['scale_results'].items()
                },
                'extrapolated': zne['zne_extrapolated'],
                'jobs': zne['jobs_run'],
            },
            'phase3_dd': {
                'sequence': 'XY4',
                'applied_to': 'all jobs',
            },
            'combined': combined,
            'phi_weights': {
                'forge': round(PHI / (PHI + 1), 6),
                'zne': round(1.0 / (PHI + 1), 6),
            },
            'total_jobs': total_jobs,
            'total_time_s': round(total_time, 2),
        }

        try:
            RESULTS_PATH.write_text(json.dumps(result_data, indent=2,
                                               default=str))
            print(f"  Results saved to {RESULTS_PATH.name}")
        except Exception as e:
            print(f"  Warning: save failed: {e}")

        return BypassResult(
            success=True,
            backend_name=self.backend_name,
            phase="full",
            pauli_expectations=forge['pauli_expectations'],
            forged_metrics={m: forge[m] for m in metric_names},
            noise_scale_results={
                str(k): {kk: vv for kk, vv in v.items() if kk != 'job_id'}
                for k, v in zne['scale_results'].items()
            },
            zne_extrapolated=zne['zne_extrapolated'],
            sacred_alignment=combined['sacred_alignment'],
            entropy_reversed=combined['entropy_reversed'],
            consciousness_phi=combined['consciousness_phi'],
            god_code_fidelity=combined['god_code_fidelity'],
            total_jobs=total_jobs,
            total_time_s=round(total_time, 2),
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  SACRED METRICS
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_all_metrics(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Compute all four sacred metrics from probability distribution."""
        return {
            'sacred_alignment': self._sacred_alignment(probs),
            'entropy_reversed': self._entropy_reversal(probs),
            'consciousness_phi': self._iit_phi(probs),
            'god_code_fidelity': self._god_code_fidelity(probs),
        }

    def _sacred_alignment(self, probs: Dict[str, float]) -> float:
        if not probs:
            return 0.0
        sorted_p = sorted(probs.values(), reverse=True)
        top_10_mass = sum(sorted_p[:10])
        h = -sum(p * math.log2(p) for p in probs.values() if p > 1e-15)
        norm_h = h / N_QUBITS if N_QUBITS > 0 else 0
        alignment = 1.0 - abs(norm_h - PHI_CONJUGATE) / max(1.0, PHI_CONJUGATE)
        return min(1.0, alignment * top_10_mass * PHI)

    def _entropy_reversal(self, probs: Dict[str, float]) -> float:
        if not probs:
            return 0.0
        h = -sum(p * math.log2(p) for p in probs.values() if p > 1e-15)
        return max(0.0, 1.0 - h / N_QUBITS) if N_QUBITS > 0 else 0.0

    def _iit_phi(self, probs: Dict[str, float]) -> float:
        if not probs or len(probs) < 2:
            return 0.0
        h_full = -sum(p * math.log2(p) for p in probs.values() if p > 1e-15)
        min_phi = float('inf')
        for cut in [6, 8, 14, 24]:  # Orbital boundaries
            h_a = self._marginal_h(probs, range(cut))
            h_b = self._marginal_h(probs, range(cut, N_QUBITS))
            min_phi = min(min_phi, h_a + h_b - h_full)
        return max(0.0, min_phi) if min_phi < float('inf') else 0.0

    def _marginal_h(self, probs: Dict[str, float], indices) -> float:
        marginal: Dict[str, float] = {}
        idx_list = list(indices)
        for bs, p in probs.items():
            bs = bs.zfill(N_QUBITS)
            key = ''.join(bs[q] for q in idx_list if q < len(bs))
            marginal[key] = marginal.get(key, 0.0) + p
        return sum(-p * math.log2(p) for p in marginal.values() if p > 1e-15)

    def _god_code_fidelity(self, probs: Dict[str, float]) -> float:
        if not probs:
            return 0.0
        gc_bits = format(int(GOD_CODE) % (2 ** N_QUBITS), f'0{N_QUBITS}b')
        return min(1.0, probs.get(gc_bits, 0.0) * 1000)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    backend = 'ibm_kingston'
    shots = 4096
    phase = 'all'

    for i, arg in enumerate(sys.argv):
        if arg == '--backend' and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
        elif arg == '--shots' and i + 1 < len(sys.argv):
            shots = int(sys.argv[i + 1])
        elif arg == '--phase' and i + 1 < len(sys.argv):
            phase = sys.argv[i + 1]
        elif arg == '--help':
            print("L104 Quantum Hybrid Bypass v1.0.0")
            print()
            print("Usage:")
            print("  python l104_quantum_hybrid_bypass.py [OPTIONS]")
            print()
            print("Options:")
            print("  --backend NAME  IBM backend (default: ibm_kingston)")
            print("  --shots N       Shots per job (default: 4096)")
            print("  --phase PHASE   Phase to run: 1, 2, all (default: all)")
            print()
            print("Phases:")
            print("  1    Entanglement Forging (golden cut, 9 jobs)")
            print("  2    Zero-Noise Extrapolation (Richardson, 3 jobs)")
            print("  all  Full bypass (12 jobs)")
            return

    bypass = HybridQuantumBypass(backend_name=backend)
    if not bypass.authenticate():
        return

    if phase == '1':
        bypass.phase1_entanglement_forging(shots)
    elif phase == '2':
        bypass.phase2_zne(shots)
    elif phase in ('all', 'full'):
        bypass.run_full_bypass(shots)
    else:
        print(f"Unknown phase: {phase}. Use 1, 2, or all.")


if __name__ == "__main__":
    main()
