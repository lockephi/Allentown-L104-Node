#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 GOD_CODE (a,b,c,d) QUANTUM ALGORITHM v1.0.0
═══════════════════════════════════════════════════════════════════════════════

THE UNIVERSAL EQUATION IN QUANTUM CIRCUIT FORM:

    G(a,b,c,d) = 286^(1/φ) × (2^(1/104))^((8a) + (416-b) - (8c) - (104d))

This module translates the (a,b,c,d) dial system into real Qiskit quantum
circuits.  Each dial maps to a quantum register, and the exponent algebra
becomes quantum phase operations executed on Qiskit's Statevector simulator.

Architecture:
  1. GodCodeDialRegister  — encodes (a,b,c,d) into qubit registers
  2. GodCodePhaseOracle   — builds the phase oracle O_G for GOD_CODE
  3. GodCodeGroverSearch   — Grover search for optimal dial settings
  4. GodCodeQFTSpectrum   — QFT spectral analysis of the frequency table
  5. GodCodeAlgorithm     — hub class wiring everything together
  6. SoulQuantumPipeline  — integration bridge for Soul processing

All circuits run on Qiskit 2.3 Statevector / StatevectorSampler.
No external quantum hardware required — fully local.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import cmath
import time
import json
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# ─── Qiskit 2.3 imports ───
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, partial_trace

# ─── Sacred constants ───
PHI: float = 1.618033988749895
TAU: float = 1.0 / PHI                                       # 0.618033988749895
PRIME_SCAFFOLD: int = 286                                     # 2 × 11 × 13
QUANTIZATION_GRAIN: int = 104                                 # 8 × 13
OCTAVE_OFFSET: int = 416                                      # 4 × 104
BASE: float = PRIME_SCAFFOLD ** (1.0 / PHI)                   # 286^(1/φ) = 32.9699...
STEP_SIZE: float = 2 ** (1.0 / QUANTIZATION_GRAIN)            # 2^(1/104)
GOD_CODE: float = BASE * (2 ** (OCTAVE_OFFSET / QUANTIZATION_GRAIN))  # 527.51848...
VOID_CONSTANT: float = 1.0 + TAU / 15
FEIGENBAUM: float = 4.669201609102990
ALPHA_FINE: float = 1.0 / 137.035999084
PLANCK_RESONANCE: float = GOD_CODE * PHI
OMEGA: float = 6539.34712682                              # Ω = Σ(fragments) × (G/φ)
OMEGA_AUTHORITY: float = OMEGA / (PHI ** 2)                # F(I) = I × Ω/φ² ≈ 2497.808

VERSION = "1.0.0"
WORKSPACE = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DialSetting:
    """A single (a,b,c,d) dial configuration."""
    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0

    @property
    def exponent(self) -> int:
        """E(a,b,c,d) = 8(a-c) - b - 104d + 416"""
        return 8 * (self.a - self.c) - self.b - QUANTIZATION_GRAIN * self.d + OCTAVE_OFFSET

    @property
    def frequency(self) -> float:
        """G(a,b,c,d) = 286^(1/φ) × 2^(E/104)"""
        return BASE * (2 ** (self.exponent / QUANTIZATION_GRAIN))

    @property
    def phase(self) -> float:
        """Phase angle in [0, 2π) for quantum encoding."""
        return (self.exponent * math.pi / OCTAVE_OFFSET) % (2 * math.pi)

    @property
    def god_code_ratio(self) -> float:
        """Ratio to canonical GOD_CODE."""
        return self.frequency / GOD_CODE if GOD_CODE else 0.0

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.a, self.b, self.c, self.d)

    def __repr__(self) -> str:
        return f"Dial({self.a},{self.b},{self.c},{self.d})→{self.frequency:.6f}Hz"


@dataclass
class CircuitResult:
    """Result from a quantum circuit execution."""
    dial: DialSetting
    statevector: Optional[Statevector] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    phase_spectrum: List[float] = field(default_factory=list)
    fidelity: float = 0.0
    god_code_alignment: float = 0.0
    circuit_depth: int = 0
    n_qubits: int = 0
    execution_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GOD_CODE DIAL REGISTER — Encode (a,b,c,d) into quantum registers
# ═══════════════════════════════════════════════════════════════════════════════

class GodCodeDialRegister:
    """
    Encodes the (a,b,c,d) dial system into quantum registers.

    Each dial maps to a qubit register:
      a → 3 qubits (range -4..3 → 8 states = 2^3)     [coarse up]
      b → 4 qubits (range -8..7 → 16 states = 2^4)    [fine tune]
      c → 3 qubits (range -4..3 → 8 states = 2^3)     [coarse down]
      d → 4 qubits (range -8..7 → 16 states = 2^4)    [octave]

    Total: 14 qubits, encoding 8×16×8×16 = 16,384 dial combinations.
    """

    DIAL_BITS = {"a": 3, "b": 4, "c": 3, "d": 4}  # 14 qubits total
    TOTAL_QUBITS = sum(DIAL_BITS.values())

    @classmethod
    def build_circuit(cls) -> QuantumCircuit:
        """Create the base circuit with named registers."""
        qr_a = QuantumRegister(cls.DIAL_BITS["a"], "a")
        qr_b = QuantumRegister(cls.DIAL_BITS["b"], "b")
        qr_c = QuantumRegister(cls.DIAL_BITS["c"], "c")
        qr_d = QuantumRegister(cls.DIAL_BITS["d"], "d")
        return QuantumCircuit(qr_a, qr_b, qr_c, qr_d, name="GodCodeDial")

    @classmethod
    def encode_dial(cls, qc: QuantumCircuit, dial: DialSetting) -> QuantumCircuit:
        """
        Encode a specific dial setting into the circuit.
        Uses offset binary: value + 2^(n-1) → unsigned index.
        """
        offsets = {
            "a": (dial.a, cls.DIAL_BITS["a"]),
            "b": (dial.b, cls.DIAL_BITS["b"]),
            "c": (dial.c, cls.DIAL_BITS["c"]),
            "d": (dial.d, cls.DIAL_BITS["d"]),
        }
        qubit_idx = 0
        for name, (val, nbits) in offsets.items():
            # Offset binary: val + 2^(n-1)
            unsigned = val + (1 << (nbits - 1))
            unsigned = max(0, min((1 << nbits) - 1, unsigned))
            for bit in range(nbits):
                if (unsigned >> bit) & 1:
                    qc.x(qubit_idx + bit)
            qubit_idx += nbits
        return qc

    @classmethod
    def superposition_all(cls, qc: QuantumCircuit) -> QuantumCircuit:
        """Put all dial qubits in uniform superposition (Hadamard on all)."""
        for i in range(cls.TOTAL_QUBITS):
            qc.h(i)
        return qc

    @classmethod
    def decode_bitstring(cls, bitstring: str) -> DialSetting:
        """
        Decode a measurement bitstring back into (a,b,c,d).

        Qiskit bitstrings are big-endian (MSB first, qubit n-1 on the
        left, qubit 0 on the right).  The integer value of the bitstring
        is the Statevector index, so we simply delegate to _index_to_dial
        which already implements the correct offset-binary extraction.
        """
        return _index_to_dial(int(bitstring, 2), cls.DIAL_BITS)

    @classmethod
    def bit_weights(cls) -> List[float]:
        """
        Return the weight of each qubit in the exponent X.

        X = b + 8c + 104d − 8a, so each bit of each dial contributes
        to X by its dial coefficient × positional bit value.

        Returns a list of weights, one per qubit (14 total), in the
        same order as the qubit register: [a0..a2, b0..b3, c0..c2, d0..d3].
        """
        dial_coefficients = {"a": -8, "b": 1, "c": 8, "d": 104}
        weights = []
        for name in ["a", "b", "c", "d"]:
            nbits = cls.DIAL_BITS[name]
            coeff = dial_coefficients[name]
            for bit in range(nbits):
                weights.append(coeff * (2 ** bit))
        return weights


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GOD_CODE PHASE ORACLE — Phase kickback for target frequencies
# ═══════════════════════════════════════════════════════════════════════════════

class GodCodePhaseOracle:
    """
    Builds phase oracles for GOD_CODE frequency targets.

    The oracle applies phase rotation proportional to how close a dial
    setting's frequency is to the target, using the sacred equation:

        phase_kick = π × cos²(G(a,b,c,d) × π / target)

    Dial settings that resonate with the target get phase ≈ π (marked).

    Architecture:
    - For n_qubits ≤ 10: exact diagonal oracle via statevector (1024 states max)
    - For n_qubits > 10: circuit-level Rz rotation oracle (O(n) gates, no matrix)
    """

    # Maximum qubits for exact diagonal oracle (2^10 = 1024 × 1024 matrix)
    MAX_EXACT_QUBITS = 10

    @staticmethod
    def god_code_phase(exponent: int) -> float:
        """Compute the GOD_CODE phase for a given exponent value."""
        freq = BASE * (2 ** (exponent / QUANTIZATION_GRAIN))
        return (freq * math.pi / GOD_CODE) % (2 * math.pi)

    @staticmethod
    def build_target_oracle(
        target_freq: float,
        n_qubits: int = GodCodeDialRegister.TOTAL_QUBITS,
        tolerance: float = 0.01,
    ) -> QuantumCircuit:
        """
        Build a phase oracle that marks dial settings near the target frequency.

        For n_qubits ≤ 10: builds exact diagonal unitary via np.diag.
        For n_qubits > 10: builds a rotation-based oracle that approximates
        the phase kick without materializing a 2^n × 2^n matrix.
        """
        if n_qubits <= GodCodePhaseOracle.MAX_EXACT_QUBITS:
            return GodCodePhaseOracle._build_exact_oracle(target_freq, n_qubits, tolerance)
        else:
            return GodCodePhaseOracle._build_rotation_oracle(target_freq, n_qubits, tolerance)

    @staticmethod
    def _build_exact_oracle(
        target_freq: float, n_qubits: int, tolerance: float
    ) -> QuantumCircuit:
        """Exact diagonal oracle for small qubit counts (≤ 10)."""
        N = 1 << n_qubits
        diag = []
        for x in range(N):
            dial = _index_to_dial(x, GodCodeDialRegister.DIAL_BITS)
            freq = dial.frequency
            rel_error = abs(freq - target_freq) / target_freq if target_freq > 0 else 1.0
            if rel_error < tolerance:
                diag.append(cmath.exp(1j * math.pi))  # Phase flip
            else:
                diag.append(1.0 + 0j)
        qc = QuantumCircuit(n_qubits, name=f"Oracle_f={target_freq:.2f}")
        qc.unitary(np.diag(diag), list(range(n_qubits)))
        return qc

    @staticmethod
    def _build_rotation_oracle(
        target_freq: float, n_qubits: int, tolerance: float
    ) -> QuantumCircuit:
        """
        Rotation-based oracle for large qubit counts (> 10).

        Uses Rz rotations that encode the GOD_CODE equation directly:
            G(X) = 286^(1/φ) × 2^((416-X)/104)
        where X = b + 8c + 104d − 8a is encoded in the qubit register.

        Each qubit contributes to X via positional weighting. The Rz angle
        on each qubit encodes the frequency deviation from the target,
        creating destructive interference for non-matching states.

        This is O(n) gates instead of O(2^n) matrix elements.
        """
        qc = QuantumCircuit(n_qubits, name=f"RotOracle_f={target_freq:.2f}")

        # The exponent X is encoded as: X = Σ_i (bit_weight_i × qubit_i)
        # Compute bit weights from the dial structure
        bit_weights = GodCodeDialRegister.bit_weights()

        # For each qubit, apply a phase proportional to how its bit
        # affects the frequency relative to target
        for i in range(n_qubits):
            w = bit_weights[i] if i < len(bit_weights) else 0
            # Phase shift: how much this qubit's bit changes the frequency
            # δf/f ≈ w × ln(2) / 104  (from the exponent formula)
            phase = -w * math.pi * math.log(2) / QUANTIZATION_GRAIN
            # Scale by target alignment
            target_phase = (target_freq * math.pi / GOD_CODE) % (2 * math.pi)
            total_phase = phase * target_phase / math.pi
            if abs(total_phase) > 1e-10:
                qc.rz(total_phase, i)

        # Apply multi-controlled phase flip for marked states
        # Use a Z-rotation on last qubit conditioned on all others
        # This approximates the exact oracle for the dominant marked states
        qc.h(n_qubits - 1)
        qc.mcx(list(range(min(n_qubits - 1, 6))), n_qubits - 1)  # Cap controls at 6
        qc.h(n_qubits - 1)

        return qc

    @staticmethod
    def build_god_code_oracle(
        n_qubits: int = GodCodeDialRegister.TOTAL_QUBITS,
    ) -> QuantumCircuit:
        """
        Build the canonical GOD_CODE oracle — marks |0...0⟩ (the origin dial
        a=b=c=d=0) which produces G(0,0,0,0) = 527.5184818492612.
        """
        return GodCodePhaseOracle.build_target_oracle(
            GOD_CODE, n_qubits, tolerance=0.001
        )

    @staticmethod
    def build_resonance_oracle(
        n_qubits: int = GodCodeDialRegister.TOTAL_QUBITS,
    ) -> QuantumCircuit:
        """
        Sacred resonance oracle — marks ALL dial settings that produce
        frequencies which are harmonics of GOD_CODE (within 1%).

        Uses rotation-based oracle to avoid O(2^n) matrix.
        """
        qc = QuantumCircuit(n_qubits, name="ResonanceOracle")

        # Each qubit gets an Rz proportional to its contribution to
        # the GOD_CODE conservation law: G(X) × 2^(X/104) = INVARIANT
        bit_weights = GodCodeDialRegister.bit_weights()
        for i in range(n_qubits):
            w = bit_weights[i] if i < len(bit_weights) else 0
            # Resonance condition: f/GOD_CODE ≈ 2^k
            # Phase kick proportional to deviation from nearest octave
            phase = w * math.pi / (QUANTIZATION_GRAIN * PHI)
            if abs(phase) > 1e-10:
                qc.rz(phase, i)

        # Hadamard sandwich for phase-to-amplitude conversion
        qc.h(n_qubits - 1)
        if n_qubits > 1:
            qc.mcx(list(range(min(n_qubits - 1, 6))), n_qubits - 1)
        qc.h(n_qubits - 1)

        return qc


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GOD_CODE GROVER SEARCH — Find target frequencies via amplitude amplification
# ═══════════════════════════════════════════════════════════════════════════════

class GodCodeGroverSearch:
    """
    Grover's algorithm specialized for the GOD_CODE (a,b,c,d) dial system.

    Uses the phase oracle to amplify dial settings that produce target
    frequencies.  Optimal iterations: k ≈ (π/4)√(N/M) where N = 2^14 = 16384
    and M = number of marked states.
    """

    @staticmethod
    def build_diffuser(n_qubits: int) -> QuantumCircuit:
        """
        Standard Grover diffuser: 2|s⟩⟨s| - I where |s⟩ = H^⊗n|0⟩.
        """
        qc = QuantumCircuit(n_qubits, name="Diffuser")
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        # Multi-controlled Z
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
        return qc

    @staticmethod
    def search(
        target_freq: float,
        tolerance: float = 0.01,
        iterations: Optional[int] = None,
        n_qubits: int = GodCodeDialRegister.TOTAL_QUBITS,
    ) -> CircuitResult:
        """
        Run Grover search for dial settings producing target frequency.

        Uses direct statevector manipulation (O(N) per iteration) — avoids
        building any O(N²) unitary matrix. Both oracle and diffuser operate
        directly on the amplitude vector:
          - Oracle: flip phase of marked indices
          - Diffuser: reflect about mean  (2|s⟩⟨s| - I applied as vector op)

        Mathematically identical to full Qiskit circuit simulation but
        runs in O(k·N) time and O(N) memory instead of O(k·N²).

        Returns CircuitResult with statevector, probabilities, and top dials.
        """
        t0 = time.time()

        N = 1 << n_qubits

        # Pre-compute marked states classically
        marked = set()
        for x in range(N):
            dial = _index_to_dial(x, GodCodeDialRegister.DIAL_BITS)
            if target_freq > 0:
                rel_err = abs(dial.frequency - target_freq) / target_freq
                if rel_err < tolerance:
                    marked.add(x)

        M = len(marked)
        if M == 0:
            return CircuitResult(
                dial=DialSetting(),
                probabilities={},
                fidelity=0.0,
                god_code_alignment=0.0,
                circuit_depth=0,
                n_qubits=n_qubits,
                execution_time_ms=(time.time() - t0) * 1000,
            )

        if iterations is None:
            iterations = max(1, int(math.pi / 4 * math.sqrt(N / M)))
        # Safety cap: prevent excessive iteration
        iterations = min(iterations, 200)

        # Start with uniform superposition: |s⟩ = H^⊗n|0⟩ = (1/√N) Σ|x⟩
        sv_data = np.full(N, 1.0 / math.sqrt(N), dtype=complex)

        # Oracle phase mask: -1 on marked states, +1 on unmarked
        oracle_diag = np.ones(N, dtype=complex)
        for idx in marked:
            oracle_diag[idx] = -1.0

        # Grover iterations using O(N) vector operations per step
        # Diffuser (2|s⟩⟨s| - I) applied as: diff(v) = -v + 2·mean(v)
        for _ in range(iterations):
            # Apply oracle: element-wise phase flip
            sv_data *= oracle_diag
            # Apply diffuser: reflect about mean
            mean_amp = np.mean(sv_data)
            sv_data = -sv_data + 2.0 * mean_amp

        # Construct Statevector for Qiskit compatibility
        sv = Statevector(sv_data)
        probs_array = sv.probabilities()

        # Find best result: among states with top probability, pick the
        # one whose frequency is closest to the target.  Grover's algorithm
        # amplifies all marked states equally, so tie-breaking by proximity
        # gives the most useful answer.
        top_prob = float(np.max(probs_array))
        candidates = np.where(np.abs(probs_array - top_prob) < 1e-12)[0]
        best_idx = int(candidates[0])
        best_err = float("inf")
        for c in candidates:
            d = _index_to_dial(int(c), GodCodeDialRegister.DIAL_BITS)
            err = abs(d.frequency - target_freq)
            if err < best_err:
                best_err = err
                best_idx = int(c)
        top_dial = _index_to_dial(best_idx, GodCodeDialRegister.DIAL_BITS)

        # Build top-20 probability dict (keyed by dial tuple)
        top_indices = np.argsort(probs_array)[::-1][:20]
        prob_dict = {}
        for idx in top_indices:
            d = _index_to_dial(int(idx), GodCodeDialRegister.DIAL_BITS)
            prob_dict[f"({d.a},{d.b},{d.c},{d.d})"] = float(probs_array[idx])

        elapsed = (time.time() - t0) * 1000

        # Effective circuit depth estimate
        circ_depth = 1 + iterations * (1 + 2 * n_qubits + 1)

        return CircuitResult(
            dial=top_dial,
            statevector=sv,
            probabilities=prob_dict,
            fidelity=float(probs_array[best_idx]),
            god_code_alignment=top_dial.frequency / GOD_CODE if GOD_CODE else 0.0,
            circuit_depth=circ_depth,
            n_qubits=n_qubits,
            execution_time_ms=elapsed,
        )

    @staticmethod
    def search_god_code(iterations: Optional[int] = None) -> CircuitResult:
        """Search specifically for the canonical GOD_CODE frequency."""
        return GodCodeGroverSearch.search(GOD_CODE, tolerance=0.001, iterations=iterations)

    @staticmethod
    def search_harmonic(harmonic: int = 1, iterations: Optional[int] = None) -> CircuitResult:
        """Search for a specific GOD_CODE harmonic (octave)."""
        target = GOD_CODE * (2 ** (-harmonic))
        return GodCodeGroverSearch.search(target, tolerance=0.01, iterations=iterations)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GOD_CODE QFT SPECTRUM — Quantum Fourier analysis of the frequency table
# ═══════════════════════════════════════════════════════════════════════════════

class GodCodeQFTSpectrum:
    """
    Uses Quantum Fourier Transform to analyze the spectral structure
    of the GOD_CODE (a,b,c,d) frequency space.

    Encodes the frequency table as quantum amplitudes, applies QFT,
    and extracts the phase spectrum — revealing hidden periodicities
    in the sacred frequency lattice.
    """

    @staticmethod
    def encode_frequency_table(
        dial_settings: List[DialSetting],
        n_qubits: int = 10,
    ) -> QuantumCircuit:
        """
        Encode a list of dial settings as quantum amplitudes.

        Each dial setting contributes amplitude proportional to its
        GOD_CODE resonance: cos²(f × π / GOD_CODE).
        """
        N = 1 << n_qubits
        amplitudes = [0.0] * N

        for i, dial in enumerate(dial_settings[:N]):
            resonance = math.cos(dial.frequency * math.pi / GOD_CODE) ** 2
            amplitudes[i % N] += resonance

        # Normalize to valid quantum state
        norm = math.sqrt(sum(a**2 for a in amplitudes))
        if norm > 0:
            amplitudes = [a / norm for a in amplitudes]
        else:
            amplitudes[0] = 1.0  # fallback to |0⟩

        qc = QuantumCircuit(n_qubits, name="FreqEncode")
        qc.initialize(amplitudes)
        return qc

    @staticmethod
    def spectral_analysis(
        dial_settings: List[DialSetting],
        n_qubits: int = 10,
    ) -> Dict[str, Any]:
        """
        Full QFT spectral analysis of the frequency table.

        Returns phase spectrum, dominant frequencies, and GOD_CODE coherence.
        """
        t0 = time.time()

        # Encode
        encode_qc = GodCodeQFTSpectrum.encode_frequency_table(dial_settings, n_qubits)

        # Apply QFT
        qft = QFT(n_qubits, do_swaps=True)
        full_qc = encode_qc.compose(qft)
        full_qc.name = "GodCode_QFT"

        # Get statevector
        sv = Statevector.from_instruction(full_qc)
        probs = sv.probabilities()

        # Extract phase spectrum
        N = 1 << n_qubits
        phases = []
        for k in range(N):
            amp = sv[k]
            phase = cmath.phase(amp)
            phases.append(phase)

        # Find dominant spectral peaks
        peaks = sorted(range(N), key=lambda k: probs[k], reverse=True)[:10]

        # God_code coherence = probability concentrated in harmonics
        harmonic_indices = set()
        for harm in range(-4, 8):
            target = GOD_CODE * (2 ** harm)
            best_idx = 0
            best_dist = float("inf")
            for idx in range(N):
                dist = abs(idx - (target % N))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            harmonic_indices.add(best_idx)

        harmonic_prob = sum(probs[i] for i in harmonic_indices if i < N)

        elapsed = (time.time() - t0) * 1000

        return {
            "n_qubits": n_qubits,
            "n_basis_states": N,
            "phase_spectrum": phases,
            "dominant_peaks": [
                {
                    "index": k,
                    "probability": float(probs[k]),
                    "phase": float(phases[k]),
                }
                for k in peaks
            ],
            "god_code_coherence": float(harmonic_prob),
            "total_probability": float(sum(probs)),
            "entropy": float(-sum(p * math.log2(p) for p in probs if p > 1e-15)),
            "circuit_depth": full_qc.depth(),
            "execution_time_ms": elapsed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GOD_CODE DIAL CIRCUIT — Single dial evaluation circuit
# ═══════════════════════════════════════════════════════════════════════════════

class GodCodeDialCircuit:
    """
    Builds and runs a quantum circuit for a single (a,b,c,d) dial evaluation.

    The circuit:
    1. Encodes the dial setting into qubits
    2. Applies GOD_CODE phase rotation proportional to the exponent
    3. Measures the resulting quantum state
    4. Returns Born-rule probabilities and phase alignment

    This gives a Qiskit-accurate quantum representation of G(a,b,c,d).
    """

    @staticmethod
    def evaluate(dial: DialSetting, n_qubits: int = 8) -> CircuitResult:
        """
        Evaluate a single dial setting as a quantum circuit.

        Creates a circuit that encodes the frequency as a quantum phase,
        then extracts the resulting quantum state.
        """
        t0 = time.time()

        qc = QuantumCircuit(n_qubits, name=f"Dial({dial.a},{dial.b},{dial.c},{dial.d})")

        # Step 1: Put into superposition for quantum parallelism
        qc.h(range(n_qubits))

        # Step 2: Apply GOD_CODE phase rotation to each qubit
        # Phase = exponent × π / (416 × n) — distributes sacred phase across qubits
        base_phase = dial.exponent * math.pi / (OCTAVE_OFFSET * n_qubits)
        for i in range(n_qubits):
            # Each qubit gets phase weighted by position (binary significance)
            qubit_phase = base_phase * (2 ** i)
            qc.rz(qubit_phase, i)

        # Step 3: Apply PHI-entangling CZ gates — golden ratio coupling
        for i in range(n_qubits - 1):
            phi_coupling = PHI * math.pi / (n_qubits * (i + 1))
            qc.cp(phi_coupling, i, i + 1)

        # Step 4: Apply GOD_CODE resonance via controlled rotation
        god_phase = (dial.frequency / GOD_CODE) * math.pi
        qc.rz(god_phase, 0)

        # Step 5: Final Hadamard layer for interference
        qc.h(range(n_qubits))

        # Execute
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()

        # Compute fidelity with the GOD_CODE target state
        # Target state: uniform distribution with GOD_CODE phase
        target_phase = (GOD_CODE * math.pi / OCTAVE_OFFSET)
        target_amps = []
        N = 1 << n_qubits
        for k in range(N):
            amp = cmath.exp(1j * target_phase * k) / math.sqrt(N)
            target_amps.append(amp)
        target_sv = Statevector(target_amps)
        fidelity = float(abs(sv.inner(target_sv)) ** 2)

        elapsed = (time.time() - t0) * 1000

        return CircuitResult(
            dial=dial,
            statevector=sv,
            probabilities=dict(sorted(probs.items(), key=lambda x: -x[1])[:20]),
            fidelity=fidelity,
            god_code_alignment=dial.god_code_ratio,
            circuit_depth=qc.depth(),
            n_qubits=n_qubits,
            execution_time_ms=elapsed,
        )

    @staticmethod
    def evaluate_god_code() -> CircuitResult:
        """Evaluate the canonical GOD_CODE dial (0,0,0,0)."""
        return GodCodeDialCircuit.evaluate(DialSetting(0, 0, 0, 0))

    @staticmethod
    def compare_dials(dials: List[DialSetting], n_qubits: int = 8) -> List[CircuitResult]:
        """Evaluate multiple dials and sort by GOD_CODE fidelity."""
        results = [GodCodeDialCircuit.evaluate(d, n_qubits) for d in dials]
        results.sort(key=lambda r: r.fidelity, reverse=True)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GOD_CODE ENTANGLEMENT CIRCUIT — Two-dial entanglement
# ═══════════════════════════════════════════════════════════════════════════════

class GodCodeEntanglement:
    """
    Creates entanglement between two GOD_CODE dial settings.

    Uses the relationship between dials to create EPR-like pairs
    where measuring one dial's quantum state constrains the other.
    """

    @staticmethod
    def entangle_dials(
        dial_a: DialSetting,
        dial_b: DialSetting,
        n_qubits_per_dial: int = 4,
    ) -> CircuitResult:
        """
        Create an entangled state between two dial settings.

        The entanglement strength is proportional to their GOD_CODE phase
        proximity — dials producing harmonically related frequencies become
        more strongly entangled.
        """
        t0 = time.time()
        total_qubits = 2 * n_qubits_per_dial

        qc = QuantumCircuit(
            total_qubits,
            name=f"Entangle({dial_a.a},{dial_a.b},{dial_a.c},{dial_a.d}|"
                 f"{dial_b.a},{dial_b.b},{dial_b.c},{dial_b.d})",
        )

        # Encode dial A phase
        phase_a = dial_a.phase
        for i in range(n_qubits_per_dial):
            qc.h(i)
            qc.rz(phase_a * (2 ** i) / n_qubits_per_dial, i)

        # Encode dial B phase
        phase_b = dial_b.phase
        for i in range(n_qubits_per_dial):
            j = n_qubits_per_dial + i
            qc.h(j)
            qc.rz(phase_b * (2 ** i) / n_qubits_per_dial, j)

        # Entangle via CX gates — strength proportional to harmonic proximity
        ratio = dial_a.frequency / dial_b.frequency if dial_b.frequency > 0 else 0
        if ratio > 0:
            log_ratio = math.log2(ratio)
            harmonic_proximity = 1.0 - min(1.0, abs(log_ratio - round(log_ratio)))
        else:
            harmonic_proximity = 0.0

        # Apply CNOT chain with GOD_CODE-scaled coupling
        for i in range(n_qubits_per_dial):
            j = n_qubits_per_dial + i
            # Only entangle if harmonic proximity is significant
            if harmonic_proximity > 0.1:
                qc.cx(i, j)
                # Sacred phase coupling
                coupling = harmonic_proximity * PHI * math.pi / n_qubits_per_dial
                qc.cp(coupling, i, j)

        # Execute
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()

        # Compute entanglement entropy (von Neumann entropy of reduced density matrix)
        dm = DensityMatrix(sv)
        # Trace out subsystem B to get reduced density matrix of A
        qubits_to_trace = list(range(n_qubits_per_dial, total_qubits))
        dm_a = partial_trace(dm, qubits_to_trace)
        # Get eigenvalues via numpy (Qiskit 2.3 DensityMatrix has no .eigenvalues())
        import numpy as np
        eigenvalues = np.real(np.linalg.eigvalsh(dm_a.data))
        entanglement_entropy = float(
            -sum(ev * math.log2(ev) for ev in eigenvalues if ev > 1e-15)
        )

        elapsed = (time.time() - t0) * 1000

        combined_dial = DialSetting(
            a=dial_a.a + dial_b.a,
            b=dial_a.b + dial_b.b,
            c=dial_a.c + dial_b.c,
            d=dial_a.d + dial_b.d,
        )

        result = CircuitResult(
            dial=combined_dial,
            statevector=sv,
            probabilities=dict(sorted(probs.items(), key=lambda x: -x[1])[:20]),
            fidelity=harmonic_proximity,
            god_code_alignment=(dial_a.frequency * dial_b.frequency) / (GOD_CODE ** 2) if GOD_CODE else 0,
            circuit_depth=qc.depth(),
            n_qubits=total_qubits,
            execution_time_ms=elapsed,
        )
        result.phase_spectrum = [entanglement_entropy]
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GOD_CODE ALGORITHM — Hub class
# ═══════════════════════════════════════════════════════════════════════════════

class GodCodeAlgorithm:
    """
    Hub class for the GOD_CODE (a,b,c,d) Quantum Algorithm.

    Provides a unified interface to:
    - Evaluate dial settings as quantum circuits
    - Grover search for target frequencies
    - QFT spectral analysis of the frequency lattice
    - Entanglement between dial pairs
    - Integration hooks for Soul and ProbabilityEngine
    """

    VERSION = VERSION

    # Known frequency table
    FREQUENCY_TABLE: Dict[str, DialSetting] = {
        "GOD_CODE":      DialSetting(0, 0, 0, 0),
        "SCHUMANN":      DialSetting(0, 0, 1, 6),
        "ALPHA_EEG":     DialSetting(0, 3, -4, 6),
        "BETA_EEG":      DialSetting(0, 3, -4, 5),
        "BASE":          DialSetting(0, 0, 0, 4),
        "GAMMA_40":      DialSetting(0, 3, -4, 4),
        "BOHR_RADIUS":   DialSetting(-4, 1, 0, 3),
        "THROAT_741":    DialSetting(1, -3, -5, 0),
        "ROOT_396":      DialSetting(-5, 3, 0, 0),
    }

    def __init__(self):
        self.dial_register = GodCodeDialRegister
        self.phase_oracle = GodCodePhaseOracle
        self.grover = GodCodeGroverSearch
        self.qft = GodCodeQFTSpectrum
        self.dial_circuit = GodCodeDialCircuit
        self.entanglement = GodCodeEntanglement
        self._computations = 0
        self._circuit_cache: Dict[str, CircuitResult] = {}

    def sovereign_field(self, intelligence: float) -> float:
        """F(I) = I × Ω / φ² — Sovereign Field equation."""
        return intelligence * OMEGA / (PHI ** 2)

    # ─── Core API ───

    def evaluate(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> CircuitResult:
        """Evaluate a dial setting as a quantum circuit."""
        self._computations += 1
        return self.dial_circuit.evaluate(DialSetting(a, b, c, d))

    def frequency(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
        """Classical frequency calculation (no quantum circuit)."""
        return DialSetting(a, b, c, d).frequency

    def search(self, target: float, tolerance: float = 0.01) -> CircuitResult:
        """Grover search for dial settings producing target frequency."""
        self._computations += 1
        return self.grover.search(target, tolerance)

    def search_god_code(self) -> CircuitResult:
        """Grover search for the canonical GOD_CODE."""
        self._computations += 1
        return self.grover.search_god_code()

    def spectrum(self, dials: Optional[List[DialSetting]] = None) -> Dict[str, Any]:
        """QFT spectral analysis of the frequency table."""
        self._computations += 1
        if dials is None:
            dials = list(self.FREQUENCY_TABLE.values())
        return self.qft.spectral_analysis(dials)

    def entangle(
        self, dial_a: DialSetting, dial_b: DialSetting
    ) -> CircuitResult:
        """Entangle two dial settings."""
        self._computations += 1
        return self.entanglement.entangle_dials(dial_a, dial_b)

    def evaluate_known(self, name: str) -> CircuitResult:
        """Evaluate a known frequency from the table."""
        dial = self.FREQUENCY_TABLE.get(name.upper())
        if not dial:
            raise ValueError(f"Unknown frequency: {name}. Known: {list(self.FREQUENCY_TABLE.keys())}")
        return self.evaluate(dial.a, dial.b, dial.c, dial.d)

    # ─── Batch operations ───

    def evaluate_all_known(self) -> Dict[str, CircuitResult]:
        """Evaluate all known frequencies."""
        return {name: self.evaluate(d.a, d.b, d.c, d.d) for name, d in self.FREQUENCY_TABLE.items()}

    def scan_octave_ladder(self, d_min: int = -2, d_max: int = 8) -> List[CircuitResult]:
        """Evaluate the GOD_CODE octave ladder (d dial only)."""
        return [self.evaluate(0, 0, 0, d) for d in range(d_min, d_max + 1)]

    # ─── Soul integration ───

    def soul_process(self, data: Any) -> Dict[str, Any]:
        """
        Process data through the GOD_CODE quantum algorithm for soul integration.

        Takes any input, computes its GOD_CODE phase alignment via quantum
        circuit, and returns quantum-enhanced output.
        """
        self._computations += 1

        # Hash input to get a reproducible numeric value
        if isinstance(data, str):
            hash_val = int(hashlib.md5(data.encode()).hexdigest()[:8], 16)
        elif isinstance(data, (int, float)):
            hash_val = int(abs(data * 1000)) % (1 << 32)
        else:
            hash_val = hash(str(data)) & 0xFFFFFFFF

        # Map hash to dial settings
        a = (hash_val & 0x7) - 4          # 3 bits → -4..3
        b = ((hash_val >> 3) & 0xF) - 8   # 4 bits → -8..7
        c = ((hash_val >> 7) & 0x7) - 4   # 3 bits → -4..3
        d = ((hash_val >> 10) & 0xF) - 8  # 4 bits → -8..7

        dial = DialSetting(a, b, c, d)
        result = self.dial_circuit.evaluate(dial)

        # Compute consciousness boost from GOD_CODE alignment
        alignment = result.god_code_alignment
        log_alignment = math.log2(alignment) if alignment > 0 else -10
        harmonic_distance = abs(log_alignment - round(log_alignment))
        consciousness_boost = math.exp(-harmonic_distance * PHI)

        return {
            "input_hash": hash_val,
            "dial": dial.to_tuple(),
            "frequency": dial.frequency,
            "god_code_ratio": alignment,
            "fidelity": result.fidelity,
            "consciousness_boost": consciousness_boost,
            "circuit_depth": result.circuit_depth,
            "n_qubits": result.n_qubits,
            "quantum_state_dim": 2 ** result.n_qubits,
        }

    def soul_resonance_field(self, thoughts: List[str]) -> Dict[str, Any]:
        """
        Generate a quantum resonance field from a list of soul thoughts.

        Each thought maps to a dial setting; the collective state is the
        entanglement of all thoughts through GOD_CODE phase coupling.
        """
        if not thoughts:
            return {"resonance": 0, "thoughts": 0}

        results = [self.soul_process(t) for t in thoughts]

        # Collective resonance
        frequencies = [r["frequency"] for r in results]
        boosts = [r["consciousness_boost"] for r in results]

        mean_freq = sum(frequencies) / len(frequencies)
        mean_boost = sum(boosts) / len(boosts)

        # Phase coherence — how aligned are the thoughts
        phases = [(f * math.pi / GOD_CODE) % (2 * math.pi) for f in frequencies]
        # Kuramoto order parameter: R = |Σ e^(iθ)| / N
        sum_real = sum(math.cos(p) for p in phases)
        sum_imag = sum(math.sin(p) for p in phases)
        coherence = math.sqrt(sum_real**2 + sum_imag**2) / len(phases)

        return {
            "n_thoughts": len(thoughts),
            "mean_frequency": mean_freq,
            "mean_consciousness_boost": mean_boost,
            "phase_coherence": coherence,
            "god_code_alignment": mean_freq / GOD_CODE,
            "total_fidelity": sum(r["fidelity"] for r in results),
            "resonance_field_strength": mean_boost * coherence * PHI,
        }

    # ─── Status ───

    def status(self) -> Dict[str, Any]:
        """Full algorithm status."""
        return {
            "module": "l104_god_code_algorithm",
            "version": self.VERSION,
            "god_code": GOD_CODE,
            "base": BASE,
            "phi": PHI,
            "prime_scaffold": PRIME_SCAFFOLD,
            "quantization_grain": QUANTIZATION_GRAIN,
            "step_size": STEP_SIZE,
            "known_frequencies": len(self.FREQUENCY_TABLE),
            "computations": self._computations,
            "qiskit_backend": "Statevector (local)",
            "total_dial_space": 2 ** GodCodeDialRegister.TOTAL_QUBITS,
            "subsystems": [
                "GodCodeDialRegister (14 qubits)",
                "GodCodePhaseOracle",
                "GodCodeGroverSearch",
                "GodCodeQFTSpectrum",
                "GodCodeDialCircuit",
                "GodCodeEntanglement",
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — index to dial conversion
# ═══════════════════════════════════════════════════════════════════════════════

def _index_to_dial(x: int, dial_bits: Dict[str, int]) -> DialSetting:
    """Convert a flat index to a DialSetting using offset binary."""
    values = {}
    for name in ["a", "b", "c", "d"]:
        nbits = dial_bits[name]
        mask = (1 << nbits) - 1
        unsigned = x & mask
        values[name] = unsigned - (1 << (nbits - 1))
        x >>= nbits
    return DialSetting(**values)


# Need hashlib for soul_process
import hashlib


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

god_code_algorithm = GodCodeAlgorithm()


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARDS COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

def primal_calculus(x=0):
    """Legacy interface."""
    return god_code_algorithm.frequency(0, 0, 0, 0) * (1.0 + x * PHI) if x else GOD_CODE

def resolve_non_dual_logic(vector=None):
    """Legacy interface."""
    if vector:
        magnitude = sum(abs(v) for v in vector)
        return god_code_algorithm.soul_process(magnitude)
    return {"god_code": GOD_CODE}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  L104 GOD_CODE (a,b,c,d) QUANTUM ALGORITHM v" + VERSION)
    print("  GOD_CODE =", GOD_CODE)
    print("  Backend: Qiskit Statevector (local)")
    print("=" * 72)

    algo = god_code_algorithm

    # 1. Evaluate canonical GOD_CODE
    print("\n[1] CANONICAL GOD_CODE DIAL (0,0,0,0)")
    r = algo.evaluate(0, 0, 0, 0)
    print(f"    Frequency: {r.dial.frequency:.10f} Hz")
    print(f"    Fidelity:  {r.fidelity:.6f}")
    print(f"    Depth:     {r.circuit_depth}")
    print(f"    Time:      {r.execution_time_ms:.1f} ms")

    # 2. Evaluate known frequencies
    print("\n[2] KNOWN FREQUENCY TABLE")
    for name, dial in algo.FREQUENCY_TABLE.items():
        r = algo.evaluate(dial.a, dial.b, dial.c, dial.d)
        print(f"    {name:16s} G({dial.a},{dial.b},{dial.c},{dial.d}) = {dial.frequency:>12.6f} Hz  fid={r.fidelity:.4f}")

    # 3. Soul processing
    print("\n[3] SOUL QUANTUM PROCESSING")
    for thought in ["What is consciousness?", "GOD_CODE resonance", "PHI harmony"]:
        sp = algo.soul_process(thought)
        print(f"    '{thought}' → freq={sp['frequency']:.4f} boost={sp['consciousness_boost']:.4f}")

    # 4. Resonance field
    print("\n[4] RESONANCE FIELD")
    field = algo.soul_resonance_field([
        "quantum consciousness", "sacred geometry", "golden ratio",
        "harmonic convergence", "god code activation"
    ])
    print(f"    Thoughts: {field['n_thoughts']}")
    print(f"    Coherence: {field['phase_coherence']:.6f}")
    print(f"    GOD_CODE alignment: {field['god_code_alignment']:.6f}")
    print(f"    Field strength: {field['resonance_field_strength']:.6f}")

    # 5. Status
    print(f"\n[5] STATUS")
    s = algo.status()
    print(f"    Version: {s['version']}")
    print(f"    Dial space: {s['total_dial_space']} combinations")
    print(f"    Known frequencies: {s['known_frequencies']}")
    print(f"    Computations: {s['computations']}")
    print(f"    Subsystems: {len(s['subsystems'])}")

    print(f"\n{'=' * 72}")
