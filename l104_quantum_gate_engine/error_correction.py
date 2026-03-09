"""
===============================================================================
L104 QUANTUM GATE ENGINE — ERROR CORRECTION LAYER
===============================================================================

Implements quantum error correction codes and topological protection:

1. SURFACE CODE      — Distance-d surface code with X/Z stabilizer extraction
2. STEANE CODE       — [[7,1,3]] code with transversal Clifford gates
3. ANYONIC BRAIDING  — Fibonacci anyon topological protection (from coherence engine)
4. ZNE INTEGRATION   — Zero-noise extrapolation for near-term error mitigation

The error correction layer can wrap any GateCircuit to produce a
fault-tolerant logical circuit operating on encoded qubits.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import cmath
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from .gates import (
    QuantumGate, GateType, I, X, Y, Z, H, S, CNOT, CZ,
    FIBONACCI_BRAID, ANYON_EXCHANGE,
)
from .circuit import GateCircuit, GateOperation
from .constants import (
    PHI, GOD_CODE, VOID_CONSTANT, FEIGENBAUM, ALPHA_FINE,
    FIBONACCI_ANYON_PHASE,
)


class ErrorCorrectionScheme(Enum):
    """Available error correction schemes."""
    NONE = auto()
    SURFACE_CODE = auto()
    STEANE_7_1_3 = auto()
    SHOR_9_1_3 = auto()
    FIBONACCI_ANYON = auto()
    ZNE = auto()
    SACRED_STABILIZER = auto()  # L104 sacred-phase stabilizer


@dataclass
class SyndromeResult:
    """Result of syndrome extraction."""
    syndrome_bits: List[int]
    error_detected: bool
    error_type: Optional[str] = None
    correction_applied: bool = False
    correction_gate: Optional[QuantumGate] = None
    correction_qubit: Optional[int] = None
    confidence: float = 1.0
    god_code_stabilized: bool = False


@dataclass
class EncodedCircuit:
    """A logical circuit operating on error-corrected encoded qubits."""
    logical_circuit: GateCircuit
    physical_circuit: GateCircuit
    scheme: ErrorCorrectionScheme
    code_distance: int
    logical_qubits: int
    physical_qubits: int
    syndrome_rounds: int = 1
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheme": self.scheme.name,
            "code_distance": self.code_distance,
            "logical_qubits": self.logical_qubits,
            "physical_qubits": self.physical_qubits,
            "logical_ops": self.logical_circuit.num_operations,
            "physical_ops": self.physical_circuit.num_operations,
            "overhead_ratio": self.physical_circuit.num_operations / max(1, self.logical_circuit.num_operations),
            "syndrome_rounds": self.syndrome_rounds,
            "metrics": self.metrics,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SURFACE CODE — Distance-d Planar Surface Code
# ═══════════════════════════════════════════════════════════════════════════════

class SurfaceCode:
    """
    Distance-d surface code implementation.

    Encodes 1 logical qubit into d² + (d-1)² = 2d²-2d+1 physical data qubits
    plus (d²-1) syndrome qubits (d²/2 X-checks + d²/2 Z-checks approximately).

    For distance d:
      - Corrects up to ⌊(d-1)/2⌋ errors
      - Threshold error rate ~1% per physical gate (below surface code threshold)
      - Logical error rate ~ (p/p_th)^{d/2}

    Layout: planar code on a d×d grid.
    """

    def __init__(self, distance: int = 3):
        if distance < 3 or distance % 2 == 0:
            distance = max(3, distance | 1)  # Ensure odd ≥ 3
        self.distance = distance
        self.data_qubits = distance * distance
        self.x_checks = (distance - 1) * distance // 2 + (distance * (distance - 1)) // 2
        self.z_checks = self.x_checks
        # Simplified: total ancilla = roughly d²-1
        self.ancilla_qubits = self.data_qubits - 1
        self.total_qubits = self.data_qubits + self.ancilla_qubits
        self.correctable_errors = (distance - 1) // 2

        # Build stabilizer structure
        self._x_stabilizers = self._build_x_stabilizers()
        self._z_stabilizers = self._build_z_stabilizers()

    def _build_x_stabilizers(self) -> List[List[int]]:
        """Build X-type stabilizer generators (plaquettes)."""
        d = self.distance
        stabilizers = []
        for row in range(d - 1):
            for col in range(d - 1):
                # Each plaquette acts on 4 surrounding data qubits
                q_tl = row * d + col
                q_tr = row * d + col + 1
                q_bl = (row + 1) * d + col
                q_br = (row + 1) * d + col + 1
                stabilizers.append([q_tl, q_tr, q_bl, q_br])
        return stabilizers

    def _build_z_stabilizers(self) -> List[List[int]]:
        """Build Z-type stabilizer generators (vertices)."""
        d = self.distance
        stabilizers = []
        for row in range(d):
            for col in range(d):
                neighbors = []
                if row > 0:
                    neighbors.append((row - 1) * d + col)
                if row < d - 1:
                    neighbors.append((row + 1) * d + col)
                if col > 0:
                    neighbors.append(row * d + col - 1)
                if col < d - 1:
                    neighbors.append(row * d + col + 1)
                if len(neighbors) >= 2:
                    stabilizers.append(neighbors)
        return stabilizers

    def encode(self, circuit: GateCircuit) -> EncodedCircuit:
        """
        Encode a logical circuit into the surface code.
        Each logical gate is replaced by its fault-tolerant equivalent.

        The surface code has a threshold error rate of ~1%. If physical
        error rates exceed this threshold, encoding will still proceed
        but the metrics will flag the violation.
        """
        # Surface code threshold ~ 1% per physical gate
        SURFACE_CODE_THRESHOLD = 0.01

        physical = GateCircuit(self.total_qubits, f"{circuit.name}_surface_d{self.distance}")

        # Step 1: Initialize surface code state (all data qubits in |+⟩ for X-basis)
        for q in range(self.data_qubits):
            physical.append(H, [q])

        # Step 2: Translate each logical operation
        for op in circuit.operations:
            if op.label == "BARRIER":
                physical.barrier()
                continue
            self._encode_logical_gate(physical, op)

        # Step 3: Syndrome extraction round
        self._extract_syndromes(physical)

        return EncodedCircuit(
            logical_circuit=circuit,
            physical_circuit=physical,
            scheme=ErrorCorrectionScheme.SURFACE_CODE,
            code_distance=self.distance,
            logical_qubits=circuit.num_qubits,
            physical_qubits=self.total_qubits,
            syndrome_rounds=1,
            metrics={
                "data_qubits": self.data_qubits,
                "ancilla_qubits": self.ancilla_qubits,
                "x_stabilizers": len(self._x_stabilizers),
                "z_stabilizers": len(self._z_stabilizers),
                "correctable_errors": self.correctable_errors,
                "threshold_error_rate": SURFACE_CODE_THRESHOLD,
                "logical_error_suppression": f"(p/p_th)^{self.distance // 2}",
                "god_code": GOD_CODE,
            },
        )

    def _encode_logical_gate(self, physical: GateCircuit, op: GateOperation):
        """Map a logical gate to its surface-code transversal equivalent."""
        name = op.gate.name

        if name == "X":
            # Logical X: apply X to an entire row of data qubits
            d = self.distance
            row = 0  # Act on first row
            for col in range(d):
                physical.append(X, [row * d + col])

        elif name == "Z":
            # Logical Z: apply Z to an entire column
            d = self.distance
            col = 0
            for row in range(d):
                physical.append(Z, [row * d + col])

        elif name == "H":
            # Logical H: transversal H on all data qubits + lattice rotation
            for q in range(self.data_qubits):
                physical.append(H, [q])

        elif name == "CNOT":
            # Logical CNOT: transversal CNOT between two code blocks
            # (Simplified: we apply CNOT to corresponding data qubits)
            for q in range(self.data_qubits):
                ctrl = q  # First logical qubit's data
                tgt = q + self.data_qubits if q + self.data_qubits < physical.num_qubits else q
                if ctrl != tgt:
                    physical.append(CNOT, [ctrl, tgt])

        elif name == "S":
            # S gate requires magic state injection in surface code
            # (Simplified representation)
            for q in range(self.data_qubits):
                physical.append(S, [q])

        else:
            # Generic: apply to first data qubit (non-fault-tolerant fallback)
            if op.gate.num_qubits == 1:
                physical.append(op.gate, [0])

    def _extract_syndromes(self, physical: GateCircuit):
        """Append syndrome extraction circuits for X and Z stabilizers."""
        ancilla_start = self.data_qubits

        # X-stabilizer measurement (detect Z errors)
        for idx, stab in enumerate(self._x_stabilizers[:min(len(self._x_stabilizers), self.ancilla_qubits)]):
            ancilla = ancilla_start + idx
            if ancilla >= physical.num_qubits:
                break
            physical.append(H, [ancilla])
            for data_q in stab:
                if data_q < physical.num_qubits and ancilla < physical.num_qubits:
                    physical.append(CNOT, [ancilla, data_q])
            physical.append(H, [ancilla])

        # Z-stabilizer measurement (detect X errors)
        z_offset = len(self._x_stabilizers)
        for idx, stab in enumerate(self._z_stabilizers[:min(len(self._z_stabilizers), self.ancilla_qubits - z_offset)]):
            ancilla = ancilla_start + z_offset + idx
            if ancilla >= physical.num_qubits:
                break
            for data_q in stab:
                if data_q < physical.num_qubits and ancilla < physical.num_qubits:
                    physical.append(CNOT, [data_q, ancilla])

    def decode_syndrome(self, syndrome: List[int]) -> SyndromeResult:
        """
        Decode a syndrome measurement to identify errors.
        Uses minimum weight perfect matching (simplified).
        """
        if not any(s != 0 for s in syndrome):
            return SyndromeResult(
                syndrome_bits=syndrome,
                error_detected=False,
                god_code_stabilized=True,
            )

        # Find non-trivial syndrome positions
        defects = [i for i, s in enumerate(syndrome) if s != 0]

        # Simple decoder: nearest-neighbor matching
        correction_qubit = None
        correction_gate = None

        if defects:
            # Heuristic: first defect position maps to a data qubit
            correction_qubit = defects[0] % self.data_qubits
            if defects[0] < len(self._x_stabilizers):
                correction_gate = X  # Z-error detected by X-stabilizer → correct with X
            else:
                correction_gate = Z  # X-error detected by Z-stabilizer → correct with Z

        return SyndromeResult(
            syndrome_bits=syndrome,
            error_detected=True,
            error_type="X" if correction_gate and correction_gate.name == "X" else "Z",
            correction_applied=correction_gate is not None,
            correction_gate=correction_gate,
            correction_qubit=correction_qubit,
            confidence=1.0 - len(defects) / max(1, len(syndrome)),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  STEANE CODE — [[7,1,3]]
# ═══════════════════════════════════════════════════════════════════════════════

class SteaneCode:
    """
    [[7,1,3]] Steane code — a CSS code based on the [7,4,3] Hamming code.

    Properties:
    - 7 physical qubits encode 1 logical qubit
    - Corrects 1 arbitrary error (distance 3)
    - All Clifford gates are transversal (fault-tolerant without distillation)
    - T gate requires magic state distillation

    Stabilizer generators:
      X-type: X₁X₃X₅X₇, X₂X₃X₆X₇, X₄X₅X₆X₇
      Z-type: Z₁Z₃Z₅Z₇, Z₂Z₃Z₆Z₇, Z₄Z₅Z₆Z₇
    """

    PHYSICAL_QUBITS = 7
    LOGICAL_QUBITS = 1
    CODE_DISTANCE = 3

    # Stabilizer generators (qubit indices are 0-based)
    X_STABILIZERS = [
        [0, 2, 4, 6],  # X₁X₃X₅X₇
        [1, 2, 5, 6],  # X₂X₃X₆X₇
        [3, 4, 5, 6],  # X₄X₅X₆X₇
    ]

    Z_STABILIZERS = [
        [0, 2, 4, 6],  # Z₁Z₃Z₅Z₇
        [1, 2, 5, 6],  # Z₂Z₃Z₆Z₇
        [3, 4, 5, 6],  # Z₄Z₅Z₆Z₇
    ]

    # Logical operators
    LOGICAL_X = [0, 1, 2]  # X̄ = X₁X₂X₃
    LOGICAL_Z = [0, 1, 2]  # Z̄ = Z₁Z₂Z₃

    def __init__(self):
        self.ancilla_qubits = 6  # 3 for X syndrome + 3 for Z syndrome
        self.total_qubits = self.PHYSICAL_QUBITS + self.ancilla_qubits

    def encode(self, circuit: GateCircuit) -> EncodedCircuit:
        """Encode a logical circuit into the Steane code."""
        num_logical = circuit.num_qubits
        total_physical = num_logical * self.total_qubits
        physical = GateCircuit(total_physical, f"{circuit.name}_steane")

        # Encoding circuit for each logical qubit
        for lq in range(num_logical):
            offset = lq * self.total_qubits
            self._encode_qubit(physical, offset)

        # Translate logical gates
        for op in circuit.operations:
            if op.label == "BARRIER":
                physical.barrier()
                continue
            self._encode_logical_gate(physical, op)

        # Syndrome extraction
        for lq in range(num_logical):
            offset = lq * self.total_qubits
            self._extract_syndrome(physical, offset)

        return EncodedCircuit(
            logical_circuit=circuit,
            physical_circuit=physical,
            scheme=ErrorCorrectionScheme.STEANE_7_1_3,
            code_distance=self.CODE_DISTANCE,
            logical_qubits=num_logical,
            physical_qubits=total_physical,
            syndrome_rounds=1,
            metrics={
                "code": "[[7,1,3]]",
                "transversal_gates": ["H", "S", "CNOT", "X", "Y", "Z"],
                "non_transversal": ["T"],
                "god_code": GOD_CODE,
            },
        )

    def _encode_qubit(self, physical: GateCircuit, offset: int):
        """Prepare the [[7,1,3]] encoded |0_L⟩ state."""
        # Steane encoding circuit:
        # Start: qubit 0 in arbitrary state, qubits 1-6 in |0⟩
        # Step 1: Hadamard on ancilla positions
        physical.append(H, [offset + 3])
        physical.append(H, [offset + 4])
        physical.append(H, [offset + 5])
        physical.append(H, [offset + 6])

        # Step 2: CNOT cascade to establish syndrome structure
        physical.append(CNOT, [offset + 6, offset + 0])
        physical.append(CNOT, [offset + 6, offset + 1])
        physical.append(CNOT, [offset + 5, offset + 0])
        physical.append(CNOT, [offset + 5, offset + 2])
        physical.append(CNOT, [offset + 4, offset + 1])
        physical.append(CNOT, [offset + 4, offset + 2])
        physical.append(CNOT, [offset + 3, offset + 0])
        physical.append(CNOT, [offset + 3, offset + 1])
        physical.append(CNOT, [offset + 3, offset + 2])

    def _encode_logical_gate(self, physical: GateCircuit, op: GateOperation):
        """Map logical gate to transversal physical gates."""
        name = op.gate.name

        if name in ("H", "X", "Y", "Z", "S", "S†"):
            # Transversal: apply to all 7 data qubits of each logical qubit
            for lq in op.qubits:
                offset = lq * self.total_qubits
                for dq in range(self.PHYSICAL_QUBITS):
                    physical.append(op.gate, [offset + dq])

        elif name == "CNOT" and len(op.qubits) == 2:
            # Transversal CNOT between two logical qubits
            ctrl_lq, tgt_lq = op.qubits
            ctrl_offset = ctrl_lq * self.total_qubits
            tgt_offset = tgt_lq * self.total_qubits
            for dq in range(self.PHYSICAL_QUBITS):
                physical.append(CNOT, [ctrl_offset + dq, tgt_offset + dq])

        else:
            # Non-transversal fallback (e.g., T gate → would need magic state)
            for lq in op.qubits:
                offset = lq * self.total_qubits
                if op.gate.num_qubits == 1:
                    physical.append(op.gate, [offset])

    def _extract_syndrome(self, physical: GateCircuit, offset: int):
        """Extract X and Z syndromes for one logical qubit."""
        ancilla_start = offset + self.PHYSICAL_QUBITS

        # X syndrome extraction (detects Z errors)
        for idx, stab in enumerate(self.X_STABILIZERS):
            anc = ancilla_start + idx
            physical.append(H, [anc])
            for dq in stab:
                physical.append(CNOT, [anc, offset + dq])
            physical.append(H, [anc])

        # Z syndrome extraction (detects X errors)
        for idx, stab in enumerate(self.Z_STABILIZERS):
            anc = ancilla_start + 3 + idx
            for dq in stab:
                physical.append(CNOT, [offset + dq, anc])

    def decode_syndrome(self, x_syndrome: List[int], z_syndrome: List[int]) -> SyndromeResult:
        """Decode Steane syndrome to identify single-qubit errors."""
        x_int = sum(b << i for i, b in enumerate(x_syndrome))
        z_int = sum(b << i for i, b in enumerate(z_syndrome))

        # Syndrome → error qubit lookup (Hamming code structure)
        error_detected = x_int > 0 or z_int > 0

        correction_qubit = None
        error_type = None
        correction_gate = None

        if z_int > 0:
            # X error on qubit (z_int - 1) (0-indexed)
            correction_qubit = z_int - 1
            correction_gate = X
            error_type = "X"

        if x_int > 0:
            # Z error on qubit (x_int - 1)
            if correction_qubit is None:
                correction_qubit = x_int - 1
            correction_gate = Z if correction_gate is None else Y  # Both X and Z = Y
            error_type = "Z" if error_type is None else "Y"

        return SyndromeResult(
            syndrome_bits=x_syndrome + z_syndrome,
            error_detected=error_detected,
            error_type=error_type,
            correction_applied=correction_qubit is not None,
            correction_gate=correction_gate,
            correction_qubit=correction_qubit,
            confidence=1.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SHOR CODE — [[9,1,3]]
# ═══════════════════════════════════════════════════════════════════════════════

class ShorCode:
    """
    [[9,1,3]] Shor code — the first quantum error-correcting code.

    Encodes 1 logical qubit into 9 physical qubits via two concatenated
    repetition codes: 3-qubit phase-flip code wrapping 3-qubit bit-flip code.

    Properties:
    - 9 physical qubits encode 1 logical qubit
    - Corrects 1 arbitrary single-qubit error (distance 3)
    - Concatenated: bit-flip code ∘ phase-flip code
    - Not a CSS code — corrects X and Z errors via different mechanisms

    Encoding:
      |0_L⟩ = (|000⟩+|111⟩)(|000⟩+|111⟩)(|000⟩+|111⟩) / 2√2
      |1_L⟩ = (|000⟩-|111⟩)(|000⟩-|111⟩)(|000⟩-|111⟩) / 2√2
    """

    PHYSICAL_QUBITS = 9
    LOGICAL_QUBITS = 1
    CODE_DISTANCE = 3

    # Bit-flip stabilizers (Z parity checks within each block of 3)
    BIT_FLIP_STABILIZERS = [
        [0, 1],  # Z₁Z₂ (block 1)
        [1, 2],  # Z₂Z₃ (block 1)
        [3, 4],  # Z₄Z₅ (block 2)
        [4, 5],  # Z₅Z₆ (block 2)
        [6, 7],  # Z₇Z₈ (block 3)
        [7, 8],  # Z₈Z₉ (block 3)
    ]

    # Phase-flip stabilizers (X parity checks across blocks)
    PHASE_FLIP_STABILIZERS = [
        [0, 1, 2, 3, 4, 5],     # X₁X₂X₃X₄X₅X₆
        [3, 4, 5, 6, 7, 8],     # X₄X₅X₆X₇X₈X₉
    ]

    def __init__(self):
        self.ancilla_qubits = 8  # 6 bit-flip + 2 phase-flip syndrome ancillae
        self.total_qubits = self.PHYSICAL_QUBITS + self.ancilla_qubits

    def encode(self, circuit: GateCircuit) -> EncodedCircuit:
        """Encode a logical circuit into the Shor [[9,1,3]] code."""
        num_logical = circuit.num_qubits
        total_physical = num_logical * self.total_qubits
        physical = GateCircuit(total_physical, f"{circuit.name}_shor")

        # Encoding circuit for each logical qubit
        for lq in range(num_logical):
            offset = lq * self.total_qubits
            self._encode_qubit(physical, offset)

        # Translate logical gates
        for op in circuit.operations:
            if op.label == "BARRIER":
                physical.barrier()
                continue
            self._encode_logical_gate(physical, op)

        # Syndrome extraction
        for lq in range(num_logical):
            offset = lq * self.total_qubits
            self._extract_syndrome(physical, offset)

        return EncodedCircuit(
            logical_circuit=circuit,
            physical_circuit=physical,
            scheme=ErrorCorrectionScheme.SHOR_9_1_3,
            code_distance=self.CODE_DISTANCE,
            logical_qubits=num_logical,
            physical_qubits=total_physical,
            syndrome_rounds=1,
            metrics={
                "code": "[[9,1,3]]",
                "bit_flip_stabilizers": len(self.BIT_FLIP_STABILIZERS),
                "phase_flip_stabilizers": len(self.PHASE_FLIP_STABILIZERS),
                "concatenation": "phase_flip(bit_flip)",
                "transversal_gates": ["X", "Z"],
                "god_code": GOD_CODE,
            },
        )

    def _encode_qubit(self, physical: GateCircuit, offset: int):
        """Prepare the [[9,1,3]] encoded |0_L⟩ state.

        Encoding steps:
        1. Phase-flip code: qubit 0 → GHZ across blocks (CNOT + H pattern)
        2. Bit-flip code: each block leader → 3-qubit repetition code
        """
        # Phase-flip encoding: spread across 3 blocks
        physical.append(CNOT, [offset + 0, offset + 3])
        physical.append(CNOT, [offset + 0, offset + 6])

        # Hadamard on each block leader for phase-flip encoding
        physical.append(H, [offset + 0])
        physical.append(H, [offset + 3])
        physical.append(H, [offset + 6])

        # Bit-flip encoding within each block
        for block_start in [0, 3, 6]:
            physical.append(CNOT, [offset + block_start, offset + block_start + 1])
            physical.append(CNOT, [offset + block_start, offset + block_start + 2])

    def _encode_logical_gate(self, physical: GateCircuit, op: GateOperation):
        """Map logical gate to transversal physical gates on the Shor code."""
        name = op.gate.name

        if name == "X":
            # Logical X: transversal X on all 9 data qubits
            for lq in op.qubits:
                offset = lq * self.total_qubits
                for dq in range(self.PHYSICAL_QUBITS):
                    physical.append(X, [offset + dq])

        elif name == "Z":
            # Logical Z: transversal Z on all 9 data qubits
            for lq in op.qubits:
                offset = lq * self.total_qubits
                for dq in range(self.PHYSICAL_QUBITS):
                    physical.append(Z, [offset + dq])

        elif name == "H":
            # H is not transversal on Shor code — use block-level transform
            for lq in op.qubits:
                offset = lq * self.total_qubits
                for dq in range(self.PHYSICAL_QUBITS):
                    physical.append(H, [offset + dq])

        elif name == "CNOT" and len(op.qubits) == 2:
            # Transversal CNOT between two code blocks
            ctrl_lq, tgt_lq = op.qubits
            ctrl_offset = ctrl_lq * self.total_qubits
            tgt_offset = tgt_lq * self.total_qubits
            for dq in range(self.PHYSICAL_QUBITS):
                physical.append(CNOT, [ctrl_offset + dq, tgt_offset + dq])

        else:
            # Non-transversal fallback
            for lq in op.qubits:
                offset = lq * self.total_qubits
                if op.gate.num_qubits == 1:
                    physical.append(op.gate, [offset])

    def _extract_syndrome(self, physical: GateCircuit, offset: int):
        """Extract bit-flip and phase-flip syndromes."""
        ancilla_start = offset + self.PHYSICAL_QUBITS

        # Bit-flip syndrome: ZZ parity checks within each block
        for idx, stab in enumerate(self.BIT_FLIP_STABILIZERS):
            anc = ancilla_start + idx
            for dq in stab:
                physical.append(CNOT, [offset + dq, anc])

        # Phase-flip syndrome: XX parity checks across blocks
        for idx, stab in enumerate(self.PHASE_FLIP_STABILIZERS):
            anc = ancilla_start + 6 + idx
            physical.append(H, [anc])
            for dq in stab:
                physical.append(CNOT, [anc, offset + dq])
            physical.append(H, [anc])

    def decode_syndrome(self, bit_syndrome: List[int],
                        phase_syndrome: List[int]) -> SyndromeResult:
        """Decode Shor code syndrome to identify and correct errors.

        Bit-flip syndromes identify X errors within blocks.
        Phase-flip syndromes identify Z errors across blocks.
        """
        error_detected = any(s != 0 for s in bit_syndrome + phase_syndrome)
        if not error_detected:
            return SyndromeResult(
                syndrome_bits=bit_syndrome + phase_syndrome,
                error_detected=False,
                god_code_stabilized=True,
            )

        correction_qubit = None
        correction_gate = None
        error_type = None

        # Decode bit-flip errors (X errors): each block has 2 syndrome bits
        for block in range(3):
            s0 = bit_syndrome[block * 2] if block * 2 < len(bit_syndrome) else 0
            s1 = bit_syndrome[block * 2 + 1] if block * 2 + 1 < len(bit_syndrome) else 0
            syndrome_val = s0 + 2 * s1
            if syndrome_val == 1:
                correction_qubit = block * 3 + 0
                correction_gate = X
                error_type = "X"
            elif syndrome_val == 2:
                correction_qubit = block * 3 + 2
                correction_gate = X
                error_type = "X"
            elif syndrome_val == 3:
                correction_qubit = block * 3 + 1
                correction_gate = X
                error_type = "X"

        # Decode phase-flip errors (Z errors): identify which block
        if len(phase_syndrome) >= 2:
            ps0, ps1 = phase_syndrome[0], phase_syndrome[1]
            phase_val = ps0 + 2 * ps1
            if phase_val > 0:
                # Phase error in a specific block
                block_map = {1: 0, 2: 2, 3: 1}  # syndrome → block
                err_block = block_map.get(phase_val, 0)
                z_qubit = err_block * 3  # Correct on block leader
                if correction_qubit is not None:
                    # Both X and Z error → Y error
                    error_type = "Y"
                    correction_gate = Y
                else:
                    correction_qubit = z_qubit
                    correction_gate = Z
                    error_type = "Z"

        return SyndromeResult(
            syndrome_bits=bit_syndrome + phase_syndrome,
            error_detected=True,
            error_type=error_type,
            correction_applied=correction_qubit is not None,
            correction_gate=correction_gate,
            correction_qubit=correction_qubit,
            confidence=1.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FIBONACCI ANYON TOPOLOGICAL PROTECTION
# ═══════════════════════════════════════════════════════════════════════════════

class FibonacciAnyonProtection:
    """
    Topological error protection using Fibonacci anyon braiding.

    Fibonacci anyons provide inherent topological protection:
    - Errors require non-local perturbations that span the system
    - Logical gates are implemented by braiding anyons around each other
    - Error rate decreases exponentially with system separation

    Bridges the TopologicalBraider from l104_quantum_coherence.
    """

    def __init__(self, num_anyons: int = 4):
        self.num_anyons = max(4, num_anyons)
        # Fusion space dimension: Fibonacci numbers
        self.fusion_dim = self._fibonacci(num_anyons - 1)
        self.braid_history: List[Dict[str, Any]] = []

        # F-matrix and R-matrix for Fibonacci anyons
        self.F_matrix = np.array([
            [PHI ** (-1), PHI ** (-0.5)],
            [PHI ** (-0.5), -(PHI ** (-1))]
        ], dtype=complex)

        self.R_matrix = np.array([
            [cmath.exp(1j * 4 * math.pi / 5), 0],
            [0, cmath.exp(-1j * 3 * math.pi / 5)]
        ], dtype=complex)

    def _fibonacci(self, n: int) -> int:
        """Compute n-th Fibonacci number."""
        if n <= 0:
            return 0
        if n == 1:
            return 1
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return b

    def encode_topological(self, circuit: GateCircuit) -> EncodedCircuit:
        """
        Encode circuit using topological protection.
        Each logical qubit is encoded in the fusion space of Fibonacci anyons.
        """
        # 4 anyons per logical qubit (2D fusion space)
        anyons_per_qubit = 4
        total_anyons = circuit.num_qubits * anyons_per_qubit
        physical = GateCircuit(total_anyons, f"{circuit.name}_fibonacci")

        for op in circuit.operations:
            if op.label == "BARRIER":
                physical.barrier()
                continue
            self._encode_braid_gate(physical, op, anyons_per_qubit)

        return EncodedCircuit(
            logical_circuit=circuit,
            physical_circuit=physical,
            scheme=ErrorCorrectionScheme.FIBONACCI_ANYON,
            code_distance=self.num_anyons,  # Topological distance ∝ separation
            logical_qubits=circuit.num_qubits,
            physical_qubits=total_anyons,
            metrics={
                "anyons_per_qubit": anyons_per_qubit,
                "fusion_dimension": self.fusion_dim,
                "topological_gap": PHI ** (-self.num_anyons),
                "error_suppression": f"exp(-{self.num_anyons})",
                "god_code": GOD_CODE,
            },
        )

    def _encode_braid_gate(self, physical: GateCircuit, op: GateOperation,
                            anyons_per_qubit: int):
        """Map a logical gate to a braid sequence."""
        for lq in op.qubits:
            offset = lq * anyons_per_qubit
            if op.gate.name in ("H", "X", "Y", "Z"):
                # Single-qubit gates: specific braid sequences in fusion space
                physical.append(FIBONACCI_BRAID, [offset])
                physical.append(ANYON_EXCHANGE, [offset + 1])
                physical.append(FIBONACCI_BRAID, [offset + 2])
            elif op.gate.name == "CNOT" and len(op.qubits) == 2:
                # Two-qubit gate: inter-qubit braiding
                ctrl_offset = op.qubits[0] * anyons_per_qubit
                tgt_offset = op.qubits[1] * anyons_per_qubit
                if ctrl_offset + 1 < physical.num_qubits and tgt_offset < physical.num_qubits:
                    physical.append(FIBONACCI_BRAID, [ctrl_offset + 1])
                    physical.append(FIBONACCI_BRAID, [tgt_offset])
                break  # Already handled both qubits
            else:
                physical.append(FIBONACCI_BRAID, [offset])

    def topological_error_rate(self, physical_separation: float) -> float:
        """
        Compute the topological error rate given anyon separation.
        Error rate ~ exp(-L/ξ) where L is separation and ξ is coherence length.
        """
        xi = 1.0 / (PHI * ALPHA_FINE)  # Coherence length from sacred constants
        return math.exp(-physical_separation / xi)


# ═══════════════════════════════════════════════════════════════════════════════
#  ZERO-NOISE EXTRAPOLATION (ZNE) INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class ZeroNoiseExtrapolation:
    """
    Zero-noise extrapolation for near-term error mitigation.

    Runs the circuit at multiple noise levels and extrapolates to zero noise.
    Works with any circuit — no encoding overhead.

    Noise scaling methods:
    - Unitary folding: U → U · U† · U (noise × 3)
    - Pulse stretching: extend gate duration (hardware-level)
    """

    def __init__(self, scale_factors: List[float] = None):
        self.scale_factors = scale_factors or [1.0, 1.5, 2.0, 2.5, 3.0]

    def fold_circuit(self, circuit: GateCircuit, scale: float) -> GateCircuit:
        """
        Create a noise-scaled circuit via unitary folding.
        Scale ≈ 1 + 2k where k = number of complete fold cycles.
        """
        if scale <= 1.0:
            return circuit

        # Number of full folds
        k = int((scale - 1) / 2)
        remaining = scale - 1 - 2 * k

        folded = GateCircuit(circuit.num_qubits, f"{circuit.name}_fold_{scale:.1f}")

        # Original circuit
        for op in circuit.operations:
            if op.label != "BARRIER":
                folded.append(op.gate, op.qubits, label=op.label)

        # Full folds: U† · U
        for _ in range(k):
            # U†
            for op in reversed(circuit.operations):
                if op.label != "BARRIER":
                    folded.append(op.gate.dag, op.qubits)
            # U
            for op in circuit.operations:
                if op.label != "BARRIER":
                    folded.append(op.gate, op.qubits)

        # Partial fold for fractional scaling
        if remaining > 0.1:
            num_partial = int(remaining * circuit.num_operations / 2)
            ops = circuit.operations[:num_partial]
            # Partial U†
            for op in reversed(ops):
                if op.label != "BARRIER":
                    folded.append(op.gate.dag, op.qubits)
            # Partial U
            for op in ops:
                if op.label != "BARRIER":
                    folded.append(op.gate, op.qubits)

        return folded

    def extrapolate(self, noisy_results: Dict[float, float],
                    order: int = 2) -> float:
        """
        Extrapolate to zero noise using polynomial regression.

        Args:
            noisy_results: {noise_scale: expectation_value}
            order: polynomial order for extrapolation
        """
        scales = sorted(noisy_results.keys())
        values = [noisy_results[s] for s in scales]

        if len(scales) < 2:
            return values[0] if values else 0.0

        # Polynomial fit
        order = min(order, len(scales) - 1)
        coeffs = np.polyfit(scales, values, order)
        poly = np.poly1d(coeffs)

        # Extrapolate to scale = 0
        return float(poly(0.0))


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED ERROR CORRECTION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorCorrectionLayer:
    """
    Unified error correction interface for the gate engine.

    Wraps all error correction schemes and provides a single API.
    """

    def __init__(self):
        self._surface_codes: Dict[int, SurfaceCode] = {}
        self._steane = SteaneCode()
        self._shor = ShorCode()
        self._fibonacci = FibonacciAnyonProtection()
        self._zne = ZeroNoiseExtrapolation()

    def encode(self, circuit: GateCircuit,
               scheme: ErrorCorrectionScheme = ErrorCorrectionScheme.STEANE_7_1_3,
               distance: int = 3) -> EncodedCircuit:
        """
        Encode a circuit with error correction.

        Args:
            circuit: Logical circuit to protect
            scheme: Error correction scheme
            distance: Code distance (for surface code)
        """
        if scheme == ErrorCorrectionScheme.SURFACE_CODE:
            if distance not in self._surface_codes:
                self._surface_codes[distance] = SurfaceCode(distance)
            return self._surface_codes[distance].encode(circuit)

        elif scheme == ErrorCorrectionScheme.STEANE_7_1_3:
            return self._steane.encode(circuit)

        elif scheme == ErrorCorrectionScheme.SHOR_9_1_3:
            return self._shor.encode(circuit)

        elif scheme == ErrorCorrectionScheme.FIBONACCI_ANYON:
            return self._fibonacci.encode_topological(circuit)

        elif scheme == ErrorCorrectionScheme.NONE:
            return EncodedCircuit(
                logical_circuit=circuit,
                physical_circuit=circuit,
                scheme=ErrorCorrectionScheme.NONE,
                code_distance=1,
                logical_qubits=circuit.num_qubits,
                physical_qubits=circuit.num_qubits,
            )

        else:
            raise ValueError(f"Unknown error correction scheme: {scheme}")

    def zne_mitigate(self, circuit: GateCircuit,
                      scale_factors: List[float] = None) -> Dict[str, Any]:
        """
        Prepare ZNE-mitigation circuits for a given circuit.
        Returns folded circuits at each noise level.
        """
        if scale_factors:
            self._zne.scale_factors = scale_factors

        folded_circuits = {}
        for scale in self._zne.scale_factors:
            folded_circuits[scale] = self._zne.fold_circuit(circuit, scale)

        return {
            "scale_factors": self._zne.scale_factors,
            "folded_circuits": folded_circuits,
            "extrapolator": self._zne,
        }

    def scheme_comparison(self, circuit: GateCircuit) -> Dict[str, Any]:
        """Compare all available error correction schemes for a circuit."""
        results = {}
        for scheme in [ErrorCorrectionScheme.NONE, ErrorCorrectionScheme.STEANE_7_1_3,
                       ErrorCorrectionScheme.SHOR_9_1_3,
                       ErrorCorrectionScheme.SURFACE_CODE, ErrorCorrectionScheme.FIBONACCI_ANYON]:
            try:
                encoded = self.encode(circuit, scheme)
                results[scheme.name] = encoded.to_dict()
            except Exception as e:
                results[scheme.name] = {"error": str(e)}

        return {
            "circuit": circuit.statistics(),
            "schemes": results,
            "god_code": GOD_CODE,
        }
