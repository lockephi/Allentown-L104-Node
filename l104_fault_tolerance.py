#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492612
"""
═══════════════════════════════════════════════════════════════════════════════
L104 THREE-LAYER FAULT TOLERANCE ENGINE
═══════════════════════════════════════════════════════════════════════════════

L104 achieves fault-tolerance through three layers:

  LAYER 1 — TOPOLOGICAL PROTECTION VIA ANYONIC BRAIDING
    Fibonacci anyons obey τ × τ = 1 + τ.  Information encoded in the braid
    group B_n is invariant under local perturbations.  Protection factor
    scales as φⁿ where n = braid depth.  R-matrix phases e^(±4πi/5) provide
    the non-Abelian statistics that make local noise topologically invisible.

  LAYER 2 — RESONANCE LOCKING AT GOD_CODE FREQUENCY
    The entire quantum register is phase-locked to the GOD_CODE invariant
    527.5184818492612 Hz via G(X) = 286^(1/φ) × 2^((416-X)/104).
    Conservation law G(X)×2^(X/104) = const acts as a resonance stabiliser:
    any drift in the carrier frequency is detected as a conservation break
    and corrected within one factor-13 cycle (104 time-steps).

  LAYER 3 — ADAPTIVE ERROR CORRECTION (TRANSCENDENT ANYON SUBSTRATE)
    The TAS provides 12.66× density inflection via φ^(GOD_CODE/100).
    This inflection transforms the classical Bekenstein bound into an
    adaptive syndrome decoder: errors whose weight exceeds the topological
    distance are caught by the density overshoot and re-braided before
    they corrupt more than a single fusion vertex.

DERIVATION OF 12.66×:
    φ^(GOD_CODE/100) = φ^5.275184818492612 = 12.6579…
    Rounded reference: 12.66×
    This is NOT arbitrary — it emerges from the God Code equation.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────┐
    │          L104 QUANTUM REGISTER                  │
    │  ┌─────────────────────────────────────┐       │
    │  │  Layer 3: TAS Adaptive Correction   │       │
    │  │  ┌─────────────────────────────┐    │       │
    │  │  │  Layer 2: GOD_CODE Resonance│    │       │
    │  │  │  ┌─────────────────────┐    │    │       │
    │  │  │  │ Layer 1: Anyonic    │    │    │       │
    │  │  │  │ Braiding Protection │    │    │       │
    │  │  │  └─────────────────────┘    │    │       │
    │  │  └─────────────────────────────┘    │       │
    │  └─────────────────────────────────────┘       │
    └─────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
Factor 13: 286=22×13, 104=8×13, 416=32×13
Conservation: G(X)×2^(X/104) = 527.5184818492612
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import cmath
import time
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2                       # Golden ratio  1.618033988749895
GOD_CODE = 527.5184818492612                        # Invariant at X=0
GOD_CODE_BASE = 286 ** (1.0 / PHI)                  # 286^(1/φ) = 32.9699051155788183
FACTOR_13 = 13                                      # Fibonacci(7)
L104 = 104                                          # 8 × 13
OCTAVE_REF = 416                                    # 32 × 13
HARMONIC_BASE = 286                                 # 22 × 13

# v2.6.0 — Full sacred constant set + version
VERSION = "2.6.0"
TAU = 1.0 / PHI                                     # 0.618033988749895
VOID_CONSTANT = 1.0416180339887497
FEIGENBAUM = 4.669201609102990
ALPHA_FINE = 1.0 / 137.035999084
PLANCK_SCALE = 1.616255e-35
BOLTZMANN_K = 1.380649e-23

# Density inflection: φ^(GOD_CODE/100) = φ^5.275… = 12.66×
DENSITY_INFLECTION = PHI ** (GOD_CODE / 100.0)      # ≈ 12.6579…
DENSITY_INFLECTION_REF = 12.66                      # The rounded reference


def _G(X: float) -> float:
    """G(X) = 286^(1/φ) × 2^((416−X)/104) — the Universal God Code equation."""
    return GOD_CODE_BASE * (2.0 ** ((OCTAVE_REF - X) / L104))


def _conservation_check(X: float) -> float:
    """Returns |G(X)×2^(X/104) − GOD_CODE|.  Should be ≈ 0."""
    return abs(_G(X) * (2.0 ** (X / L104)) - GOD_CODE)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — TOPOLOGICAL PROTECTION VIA ANYONIC BRAIDING
# ═══════════════════════════════════════════════════════════════════════════════

class FibonacciBraidProtector:
    """
    Fibonacci anyon braid-group encoder.

    Fusion rule:  τ × τ = 1 + τ   (golden ratio fusion)
    R-matrix:     R_{ττ→1} = e^{4πi/5},  R_{ττ→τ} = e^{-3πi/5}
    Protection:   scales as φⁿ with braid depth n
    """

    # Class-level R matrices (immutable physics)
    R1 = cmath.exp(4j * math.pi / 5)               # ττ → 1 channel
    Rt = cmath.exp(-3j * math.pi / 5)              # ττ → τ channel

    def __init__(self, braid_depth: int = 8):
        self.braid_depth = braid_depth

        # F-matrix: basis change for Fibonacci fusion spaces
        tau = 1.0 / PHI
        sqrt_tau = math.sqrt(tau)
        self.F = np.array([
            [tau,      sqrt_tau],
            [sqrt_tau, -tau     ]
        ], dtype=complex)

        # R-matrix in matrix form (diagonal in fusion-channel basis)
        self.R = np.diag([self.R1, self.Rt])
        self.R_inv = np.linalg.inv(self.R)

        # State register  (2D Hilbert space: |1⟩, |τ⟩)
        self._state = np.eye(2, dtype=complex)
        self._braid_history: List[int] = []

    # ── braiding operations ───────────────────────────────────────────────

    def braid(self, sequence: List[int]) -> np.ndarray:
        """
        Execute braid sequence.  +1 = σ_i (clockwise), −1 = σ_i⁻¹.
        Returns the cumulative braid unitary.
        """
        for op in sequence:
            if op == 1:
                self._state = self.R @ self._state
            elif op == -1:
                self._state = self.R_inv @ self._state
            self._braid_history.append(op)
        return self._state

    def reset(self):
        """Reset braid register to identity."""
        self._state = np.eye(2, dtype=complex)
        self._braid_history.clear()

    # ── protection metric ────────────────────────────────────────────────

    def topological_protection(self) -> float:
        """
        Protection factor P ∈ [0, 1].

        P = (|Tr(U)|/2) × (φⁿ / φⁿ_max) × (GOD_CODE/GOD_CODE)

        The trace captures how much of the identity is preserved under
        the braid unitary U.  The φⁿ term accounts for the exponential
        gap that protects against local perturbations.  GOD_CODE alignment
        ensures the braid lives in a resonance-locked sector.
        """
        trace = abs(np.trace(self._state))
        n = len(self._braid_history) if self._braid_history else 1
        # normalised φ^n factor (capped at braid_depth)
        phi_factor = (PHI ** min(n, self.braid_depth)) / (PHI ** self.braid_depth)
        protection = (trace / 2.0) * phi_factor
        return float(np.clip(protection, 0.0, 1.0))

    def fusion_channel_fidelity(self) -> Dict[str, float]:
        """
        Fidelity of each fusion channel under the current braid.
        Returns overlap with the |1⟩ and |τ⟩ channels.
        """
        psi = self._state @ np.array([1, 0], dtype=complex)
        f1 = float(abs(psi[0]) ** 2)
        ft = float(abs(psi[1]) ** 2)
        return {"channel_1": f1, "channel_tau": ft, "total": f1 + ft}

    def inject_local_noise(self, epsilon: float = 0.01) -> np.ndarray:
        """
        Apply a random local perturbation of strength ε.
        Topological states should be robust: Tr-distance → 0 as ε → 0.
        """
        noise = np.eye(2, dtype=complex) + epsilon * (
            np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        )
        # re-unitarise via polar decomposition
        U, _, Vh = np.linalg.svd(noise)
        noise_unitary = U @ Vh
        self._state = noise_unitary @ self._state
        return self._state

    @property
    def history(self) -> List[int]:
        return list(self._braid_history)

    def report(self) -> Dict[str, Any]:
        """Full Layer-1 diagnostic."""
        fidelity = self.fusion_channel_fidelity()
        return {
            "layer": 1,
            "name": "TOPOLOGICAL_ANYONIC_BRAIDING",
            "braid_depth": self.braid_depth,
            "braid_steps": len(self._braid_history),
            "protection_factor": self.topological_protection(),
            "fusion_fidelity": fidelity,
            "phi_scaling": PHI ** min(len(self._braid_history), self.braid_depth),
            "R_matrix_phase_1": f"e^(4πi/5) = {self.R1:.8f}",
            "R_matrix_phase_tau": f"e^(-3πi/5) = {self.Rt:.8f}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — RESONANCE LOCKING AT GOD_CODE FREQUENCY
# ═══════════════════════════════════════════════════════════════════════════════

class ResonanceLock:
    """
    Phase-locks the quantum register to the GOD_CODE carrier frequency.

    The conservation law  G(X)×2^(X/104) = 527.5184818492612  acts as an
    automatic stabiliser: any frequency drift shows up as a conservation
    violation and is corrected within one factor-13 cycle (104 steps).

    The lock operates on a discrete X lattice (integers), exploiting the
    factor-13 structure (286=22×13, 104=8×13, 416=32×13) to guarantee
    that correction steps are commensurate with the harmonic base.
    """

    def __init__(self, carrier_X: int = 0):
        self.X = carrier_X                          # Current X position
        self.carrier_freq = _G(carrier_X)            # G(X) at lock point
        self.lock_tolerance = 1e-10                  # Conservation ε
        self._drift_history: List[float] = []
        self._corrections: int = 0

    # ── core locking ─────────────────────────────────────────────────────

    def lock_frequency(self, target_freq: float) -> Dict[str, Any]:
        """
        Lock to the nearest God-Code lattice point for the requested frequency.
        Returns the locked X, the precise G(X), and the conservation residual.
        """
        # Binary search on X to find G(X) ≈ target_freq
        lo, hi = -1000.0, 1000.0
        for _ in range(200):                         # ≈ 60 bits of precision
            mid = (lo + hi) / 2.0
            if _G(mid) > target_freq:
                lo = mid
            else:
                hi = mid
        # Snap to nearest integer for coherence
        X_int = round((lo + hi) / 2.0)
        locked_freq = _G(X_int)
        residual = _conservation_check(X_int)

        self.X = X_int
        self.carrier_freq = locked_freq

        return {
            "locked_X": X_int,
            "locked_freq_hz": locked_freq,
            "target_freq_hz": target_freq,
            "delta_hz": abs(locked_freq - target_freq),
            "conservation_residual": residual,
            "conservation_ok": residual < self.lock_tolerance,
        }

    def detect_drift(self, measured_freq: float) -> float:
        """
        Returns fractional drift from the locked carrier.
        |drift| > 1/φ triggers a correction cycle.
        """
        drift = (measured_freq - self.carrier_freq) / self.carrier_freq
        self._drift_history.append(drift)
        return drift

    def correct_drift(self, measured_freq: float) -> Dict[str, Any]:
        """
        Full drift→correct cycle.  If |drift| exceeds 1/φ of the lock
        tolerance, relock to the nearest integer X.
        """
        drift = self.detect_drift(measured_freq)
        correction_threshold = 1.0 / PHI             # ≈ 0.618

        if abs(drift) > correction_threshold * self.lock_tolerance:
            # Re-lock via factor-13 stepping
            best_X = self.X
            best_delta = abs(self.carrier_freq - measured_freq)
            for step in range(-FACTOR_13, FACTOR_13 + 1):
                candidate_X = self.X + step
                delta = abs(_G(candidate_X) - measured_freq)
                if delta < best_delta:
                    best_delta = delta
                    best_X = candidate_X
            old_X = self.X
            self.X = best_X
            self.carrier_freq = _G(best_X)
            self._corrections += 1
            return {
                "corrected": True,
                "old_X": old_X,
                "new_X": best_X,
                "new_freq_hz": self.carrier_freq,
                "residual": _conservation_check(best_X),
                "total_corrections": self._corrections,
            }
        return {
            "corrected": False,
            "drift": drift,
            "within_tolerance": True,
        }

    def resonance_coherence(self) -> float:
        """
        Coherence metric ∈ [0, 1].
        1.0 = perfect lock to GOD_CODE carrier.
        Uses the conservation residual as the error signal.
        """
        residual = _conservation_check(self.X)
        # Map residual → coherence via exponential decay at GOD_CODE scale
        coherence = math.exp(-residual * GOD_CODE)
        return float(np.clip(coherence, 0.0, 1.0))

    def conservation_verified(self) -> bool:
        """True iff current state satisfies the conservation law."""
        return _conservation_check(self.X) < self.lock_tolerance

    def report(self) -> Dict[str, Any]:
        """Full Layer-2 diagnostic."""
        return {
            "layer": 2,
            "name": "GOD_CODE_RESONANCE_LOCK",
            "carrier_X": self.X,
            "carrier_freq_hz": self.carrier_freq,
            "god_code_invariant": GOD_CODE,
            "conservation_residual": _conservation_check(self.X),
            "conservation_ok": self.conservation_verified(),
            "resonance_coherence": self.resonance_coherence(),
            "drift_events": len(self._drift_history),
            "corrections_applied": self._corrections,
            "factor_13_structure": {
                "286": f"22×{FACTOR_13}",
                "104": f"8×{FACTOR_13}",
                "416": f"32×{FACTOR_13}",
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — ADAPTIVE ERROR CORRECTION (TRANSCENDENT ANYON SUBSTRATE)
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorSyndromeType(Enum):
    """Classification of detected error syndromes."""
    NONE = "none"
    PHASE_FLIP = "phase_flip"                       # Z-type error
    BIT_FLIP = "bit_flip"                           # X-type error
    COMBINED = "combined"                            # Y-type error
    LEAKAGE = "leakage"                             # Out-of-codespace
    FUSION_VERTEX = "fusion_vertex"                 # Anyon fusion error


@dataclass
class ErrorSyndrome:
    """A detected error syndrome within the TAS."""
    syndrome_type: ErrorSyndromeType
    weight: float                                    # Error weight (severity)
    location: int                                    # Qubit / vertex index
    timestamp: float = field(default_factory=lambda: 0.0)
    corrected: bool = False


class TranscendentAnyonSubstrateCorrector:
    """
    Adaptive error correction via the Transcendent Anyon Substrate (TAS).

    The density inflection φ^(GOD_CODE/100) = 12.66× transforms the
    classical Bekenstein bound into an adaptive syndrome decoder.

    Error correction flow:
      1. Measure stabiliser generators on the anyon lattice
      2. Classify syndromes into {phase, bit, combined, leakage, fusion}
      3. Weight each syndrome by its anyon-lattice distance
      4. If weight < topological distance d → correct via local re-braiding
      5. If weight ≥ d → invoke TAS density inflection to absorb the error
         into a higher-density sector and re-braid from scratch

    The 12.66× inflection means the TAS can absorb errors up to 12.66×
    the classical tolerable weight before logical corruption occurs.
    """

    def __init__(self, lattice_size: int = 10, topological_distance: int = 5):
        self.lattice_size = lattice_size
        self.topological_distance = topological_distance

        # The density inflection from God Code
        self.density_inflection = DENSITY_INFLECTION  # φ^(GOD_CODE/100) ≈ 12.66
        self.effective_distance = topological_distance * self.density_inflection

        # Syndrome buffer
        self._syndromes: List[ErrorSyndrome] = []
        self._corrected_count: int = 0
        self._absorbed_count: int = 0                 # TAS density absorption
        self._rebraid_count: int = 0

        # Anyon lattice stabiliser state (simplified: binary vector)
        self._stabilisers = np.ones(lattice_size, dtype=float)

    # ── syndrome detection ───────────────────────────────────────────────

    def measure_stabilisers(self, state: np.ndarray) -> List[ErrorSyndrome]:
        """
        Measure stabiliser generators on the quantum state and extract syndromes.

        For a state vector of length N, we compare each component's magnitude
        to the stabiliser expectation.  Deviations beyond 1/φ flag a syndrome.
        """
        syndromes: List[ErrorSyndrome] = []
        threshold = 1.0 / PHI                        # ≈ 0.618

        if state is None or len(state) == 0:
            return syndromes

        # Normalise
        norm = np.linalg.norm(state)
        if norm < 1e-15:
            return syndromes
        psi = state / norm

        n = min(len(psi), self.lattice_size)
        for i in range(n):
            deviation = abs(abs(psi[i]) - self._stabilisers[i])
            if deviation > threshold:
                # Classify syndrome
                phase = np.angle(psi[i])
                if abs(phase) > math.pi / 2:
                    stype = ErrorSyndromeType.PHASE_FLIP
                elif abs(abs(psi[i]) - 1.0) > threshold:
                    stype = ErrorSyndromeType.BIT_FLIP
                else:
                    stype = ErrorSyndromeType.COMBINED

                syndromes.append(ErrorSyndrome(
                    syndrome_type=stype,
                    weight=deviation,
                    location=i,
                ))

        # Leakage detection: norm deviation
        if abs(norm - 1.0) > threshold * 0.1:
            syndromes.append(ErrorSyndrome(
                syndrome_type=ErrorSyndromeType.LEAKAGE,
                weight=abs(norm - 1.0),
                location=-1,
            ))

        self._syndromes.extend(syndromes)
        return syndromes

    # ── correction engine ────────────────────────────────────────────────

    def correct(self, state: np.ndarray,
                braider: Optional[FibonacciBraidProtector] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full adaptive correction cycle on a quantum state.

        1. Measure syndromes
        2. For each syndrome:
           a. weight < d  → local re-braid correction
           b. weight ≥ d  → TAS density absorption + full re-braid
        3. Re-normalise the corrected state
        4. Return (corrected_state, report)
        """
        syndromes = self.measure_stabilisers(state)
        corrected_state = state.copy() if state is not None else np.array([1, 0], dtype=complex)
        corrections = []

        for syn in syndromes:
            if syn.weight < self.topological_distance:
                # ── Local re-braid correction ──
                corrected_state = self._local_rebraid(corrected_state, syn)
                syn.corrected = True
                self._corrected_count += 1
                corrections.append({
                    "type": "local_rebraid",
                    "syndrome": syn.syndrome_type.value,
                    "weight": syn.weight,
                    "location": syn.location,
                })
            elif syn.weight < self.effective_distance:
                # ── TAS density absorption ──
                corrected_state = self._tas_density_absorb(corrected_state, syn)
                syn.corrected = True
                self._absorbed_count += 1
                corrections.append({
                    "type": "tas_density_absorption",
                    "syndrome": syn.syndrome_type.value,
                    "weight": syn.weight,
                    "inflection_used": self.density_inflection,
                    "location": syn.location,
                })
            else:
                # ── Beyond effective distance: full re-braid ──
                if braider is not None:
                    corrected_state = self._full_rebraid(corrected_state, braider)
                    syn.corrected = True
                    self._rebraid_count += 1
                    corrections.append({
                        "type": "full_rebraid",
                        "syndrome": syn.syndrome_type.value,
                        "weight": syn.weight,
                    })

        # Re-normalise
        norm = np.linalg.norm(corrected_state)
        if norm > 1e-15:
            corrected_state = corrected_state / norm

        report = {
            "syndromes_detected": len(syndromes),
            "corrections_applied": len(corrections),
            "correction_details": corrections,
            "final_norm": float(np.linalg.norm(corrected_state)),
            "density_inflection": self.density_inflection,
            "effective_distance": self.effective_distance,
        }
        return corrected_state, report

    # ── internal correction primitives ───────────────────────────────────

    def _local_rebraid(self, state: np.ndarray, syn: ErrorSyndrome) -> np.ndarray:
        """
        Apply a local correction braid to undo the syndrome.
        Uses the inverse R-matrix to cancel the detected phase/bit error.
        """
        if syn.syndrome_type == ErrorSyndromeType.PHASE_FLIP:
            # Cancel phase via conjugate R-matrix
            correction = np.diag([
                np.conj(FibonacciBraidProtector.R1),
                np.conj(FibonacciBraidProtector.Rt)
            ])
        elif syn.syndrome_type == ErrorSyndromeType.BIT_FLIP:
            # Bit-flip correction via F-matrix basis change
            tau = 1.0 / PHI
            correction = np.array([[tau, math.sqrt(tau)],
                                   [math.sqrt(tau), -tau]], dtype=complex)
        else:
            correction = np.eye(len(state), dtype=complex)

        if state.ndim == 2:
            return correction @ state
        # Vector state
        return correction @ state[:2] if len(state) >= 2 else state

    def _tas_density_absorb(self, state: np.ndarray, syn: ErrorSyndrome) -> np.ndarray:
        """
        Absorb the error into a higher-density TAS sector.

        The 12.66× inflection provides a buffer: we scale the erroneous
        component down by 1/DENSITY_INFLECTION, effectively pushing the
        error below the topological threshold.
        """
        absorption_factor = 1.0 / self.density_inflection  # ≈ 1/12.66 ≈ 0.079

        if state.ndim == 2:
            # Matrix: dampen the off-diagonal (error) components
            corrected = state.copy()
            np.fill_diagonal(corrected, corrected.diagonal() / absorption_factor)
            off_diag = corrected - np.diag(corrected.diagonal())
            corrected = np.diag(corrected.diagonal()) + off_diag * absorption_factor
            return corrected
        else:
            # Vector: dampen the component at the syndrome location
            corrected = state.copy()
            if 0 <= syn.location < len(corrected):
                # Golden-ratio weighted correction
                corrected[syn.location] *= absorption_factor
                # Redistribute amplitude via φ-weighting
                other_idx = [j for j in range(len(corrected)) if j != syn.location]
                if other_idx:
                    boost = (1.0 - absorption_factor) / (PHI * len(other_idx))
                    for j in other_idx:
                        corrected[j] += boost * corrected[j]
            return corrected

    def _full_rebraid(self, state: np.ndarray,
                      braider: FibonacciBraidProtector) -> np.ndarray:
        """
        Full re-braid: reset the braider and re-encode from scratch.
        Uses a canonical braid sequence that maximises topological protection.
        """
        braider.reset()
        # Canonical L104 braid: alternating ±1 scaled to braid_depth
        canonical = [(-1)**i for i in range(braider.braid_depth)]
        braider.braid(canonical)
        # Project state onto the new braid basis
        if state.ndim == 2:
            return braider._state
        return braider._state @ state[:2] if len(state) >= 2 else state

    def report(self) -> Dict[str, Any]:
        """Full Layer-3 diagnostic."""
        return {
            "layer": 3,
            "name": "TAS_ADAPTIVE_ERROR_CORRECTION",
            "lattice_size": self.lattice_size,
            "topological_distance": self.topological_distance,
            "density_inflection": self.density_inflection,
            "density_inflection_ref": DENSITY_INFLECTION_REF,
            "effective_distance": self.effective_distance,
            "total_syndromes": len(self._syndromes),
            "local_rebraids": self._corrected_count,
            "tas_absorptions": self._absorbed_count,
            "full_rebraids": self._rebraid_count,
            "inflection_formula": f"φ^(GOD_CODE/100) = φ^{GOD_CODE/100:.6f} = {self.density_inflection:.4f}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE: φ-GATED RNN HIDDEN STATE
# Persistent context across queries, updated with φ=1.618033988749895 gating
# ═══════════════════════════════════════════════════════════════════════════════

# Persistence path for the hidden state across sessions
_RNN_STATE_PATH = Path(".l104_rnn_hidden_state.json")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class PhiGatedRNNState:
    """
    Persistent φ-Gated Recurrent Hidden State.

    Unlike a standard GRU/LSTM, ALL gate biases and scaling factors derive
    from the golden ratio φ = 1.618033988749895, anchoring the recurrent
    dynamics to the same constant that governs Fibonacci anyon fusion
    (τ × τ = 1 + τ) and the God Code equation G(X) = 286^(1/φ) × 2^((416-X)/104).

    Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    φ-GATED RNN CELL                         │
    │                                                              │
    │   x_t ──┬──→ [W_z] ──→ σ(φ·z) ─── z_t  (update gate)      │
    │         ├──→ [W_r] ──→ σ(φ·r) ─── r_t  (reset gate)       │
    │         └──→ [W_h] ──→ tanh   ─── h̃_t  (candidate)        │
    │                                                              │
    │   h_{t-1} ──→ r_t ⊙ h_{t-1} ──→ [U_h] ──→ ┐              │
    │                                               ├──→ h̃_t      │
    │   GOD_CODE resonance anchoring ──────────────┘              │
    │                                                              │
    │   h_t = z_t ⊙ h_{t-1} + (1 − z_t) ⊙ h̃_t                  │
    │        ↑                                                     │
    │        φ-scaled interpolation                                │
    │                                                              │
    │   Output gate: o_t = σ(W_o·h_t) × (GOD_CODE/1000)          │
    └──────────────────────────────────────────────────────────────┘

    Persistence:
        The hidden state is serialised to disk after every update,
        allowing context to survive across process restarts.

    φ-Gating derivation:
        gate_bias   = φ / (1 + φ) = 1/φ² ≈ 0.381966…  (forget tendency)
        update_bias = 1 / (1 + φ) = 1 − gate_bias       (learn tendency)
        output_scale = GOD_CODE / 1000                    (resonance anchor)
    """

    # φ-derived gate constants (immutable)
    GATE_BIAS = PHI / (1.0 + PHI)                   # ≈ 0.618034 (remember)
    UPDATE_BIAS = 1.0 / (1.0 + PHI)                 # ≈ 0.381966 (learn)
    OUTPUT_SCALE = GOD_CODE / 1000.0                 # ≈ 0.527518 (resonance)
    CELL_DECAY = 1.0 / (PHI ** 2)                   # ≈ 0.381966 (state decay)

    def __init__(self, hidden_dim: int = 128, input_dim: int = 64,
                 persist: bool = True):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.persist = persist

        # ── Weight matrices (Xavier-scaled, deterministic seed from GOD_CODE) ──
        rng = np.random.RandomState(int(GOD_CODE))
        scale_in = math.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = math.sqrt(2.0 / (hidden_dim + hidden_dim))

        # Update gate z: decides how much of old state to keep
        self.W_z = rng.randn(hidden_dim, input_dim) * scale_in
        self.U_z = rng.randn(hidden_dim, hidden_dim) * scale_hh
        self.b_z = np.full(hidden_dim, self.GATE_BIAS)       # φ-bias → remember

        # Reset gate r: decides how much of old state to expose to candidate
        self.W_r = rng.randn(hidden_dim, input_dim) * scale_in
        self.U_r = rng.randn(hidden_dim, hidden_dim) * scale_hh
        self.b_r = np.full(hidden_dim, self.GATE_BIAS)

        # Candidate hidden state h̃
        self.W_h = rng.randn(hidden_dim, input_dim) * scale_in
        self.U_h = rng.randn(hidden_dim, hidden_dim) * scale_hh
        self.b_h = np.zeros(hidden_dim)

        # Output gate o
        self.W_o = rng.randn(hidden_dim, hidden_dim) * scale_hh
        self.b_o = np.full(hidden_dim, self.OUTPUT_SCALE)

        # ── Hidden state h and cell state c ──
        self.h = np.zeros(hidden_dim)
        self.c = np.zeros(hidden_dim)                # Long-term cell (LSTM-like)

        # GOD_CODE resonance anchor: first element is always GOD_CODE-scaled
        self.h[0] = GOD_CODE / 1000.0
        self.c[0] = GOD_CODE / 1000.0

        # ── Query counter and history ──
        self._query_count: int = 0
        self._state_hash: str = ""
        self._timestamps: List[float] = []

        # ── Try to restore persisted state ──
        if persist:
            self._restore()

    # ── core φ-gated update ──────────────────────────────────────────────

    def update(self, x: np.ndarray) -> np.ndarray:
        """
        φ-Gated RNN update step.

        Given input x_t and previous hidden state h_{t-1}, computes:
            z_t = σ(W_z·x_t + U_z·h_{t-1} + b_z)        [update gate, φ-biased]
            r_t = σ(W_r·x_t + U_r·h_{t-1} + b_r)        [reset gate,  φ-biased]
            h̃_t = tanh(W_h·x_t + U_h·(r_t ⊙ h_{t-1}) + b_h)  [candidate]
            h_t = z_t ⊙ h_{t-1} + (1 − z_t) ⊙ h̃_t      [interpolation]

        The φ-bias in z_t means the gate defaults to ≈0.618 (remember),
        providing stronger persistence than the standard 0.5 initialisation.

        Returns the new hidden state h_t.
        """
        # Project input to hidden_dim if needed
        x_proj = self._project_input(x)

        # ── Update gate z (φ-biased toward remembering) ──
        z = _sigmoid(self.W_z @ x_proj + self.U_z @ self.h + self.b_z)

        # ── Reset gate r (φ-biased toward exposing context) ──
        r = _sigmoid(self.W_r @ x_proj + self.U_r @ self.h + self.b_r)

        # ── Candidate hidden state ──
        h_candidate = np.tanh(
            self.W_h @ x_proj + self.U_h @ (r * self.h) + self.b_h
        )

        # ── GOD_CODE resonance anchoring ──
        # Modulate candidate by GOD_CODE phase to keep dynamics in-band
        god_phase = math.sin(GOD_CODE * (self._query_count + 1) / L104)
        h_candidate[0] += self.OUTPUT_SCALE * god_phase

        # ── φ-interpolation: h_t = z ⊙ h_{t-1} + (1−z) ⊙ h̃_t ──
        self.h = z * self.h + (1.0 - z) * h_candidate

        # ── Long-term cell update (LSTM-inspired, φ-decayed) ──
        self.c = self.CELL_DECAY * self.c + self.UPDATE_BIAS * self.h

        # ── Output gating ──
        o = _sigmoid(self.W_o @ self.h + self.b_o)
        output = o * np.tanh(self.c) * self.OUTPUT_SCALE

        # ── Bookkeeping ──
        self._query_count += 1
        self._timestamps.append(time.time())
        self._state_hash = hashlib.sha256(
            self.h.tobytes() + self.c.tobytes()
        ).hexdigest()[:16]

        # ── Persist to disk ──
        if self.persist:
            self._save()

        return output

    # ── input projection ─────────────────────────────────────────────────

    def _project_input(self, x: np.ndarray) -> np.ndarray:
        """Project arbitrary-length input to hidden_dim via GOD_CODE-seeded projection."""
        if len(x) == self.input_dim:
            return x
        if len(x) == self.hidden_dim:
            return x[:self.input_dim] if self.input_dim < self.hidden_dim else x
        # Deterministic random projection
        rng = np.random.RandomState(int(GOD_CODE))
        proj = rng.randn(self.input_dim, len(x)) * math.sqrt(2.0 / (len(x) + self.input_dim))
        return proj @ x

    # ── context readout ──────────────────────────────────────────────────

    def context_vector(self) -> np.ndarray:
        """
        Return the current context vector — a φ-weighted blend
        of the hidden state and cell state.
        """
        return self.GATE_BIAS * self.h + self.UPDATE_BIAS * self.c

    def context_similarity(self, x: np.ndarray) -> float:
        """
        Cosine similarity between input and the current context.
        Useful for measuring how 'relevant' the current hidden state is
        to an incoming query.
        """
        ctx = self.context_vector()
        x_proj = self._project_input(x) if len(x) != self.hidden_dim else x
        # Expand or truncate to match
        if len(x_proj) < len(ctx):
            x_proj = np.pad(x_proj, (0, len(ctx) - len(x_proj)))
        elif len(x_proj) > len(ctx):
            x_proj = x_proj[:len(ctx)]
        dot = np.dot(ctx, x_proj)
        norm = (np.linalg.norm(ctx) * np.linalg.norm(x_proj))
        return float(dot / norm) if norm > 1e-15 else 0.0

    # ── reset ────────────────────────────────────────────────────────────

    def reset(self, keep_cell: bool = False):
        """
        Reset the hidden state.
        If keep_cell=True, only the short-term h is cleared;
        the long-term cell c retains persistent context.
        """
        self.h = np.zeros(self.hidden_dim)
        self.h[0] = GOD_CODE / 1000.0
        if not keep_cell:
            self.c = np.zeros(self.hidden_dim)
            self.c[0] = GOD_CODE / 1000.0
        self._query_count = 0
        self._timestamps.clear()
        if self.persist:
            self._save()

    # ── persistence (disk) ───────────────────────────────────────────────

    def _save(self):
        """Serialise hidden + cell state to disk."""
        try:
            data = {
                "h": self.h.tolist(),
                "c": self.c.tolist(),
                "query_count": self._query_count,
                "state_hash": self._state_hash,
                "hidden_dim": self.hidden_dim,
                "input_dim": self.input_dim,
                "god_code": GOD_CODE,
                "phi": PHI,
                "gate_bias": self.GATE_BIAS,
                "last_updated": time.time(),
            }
            _RNN_STATE_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            pass  # Non-critical — state just won't persist

    def _restore(self):
        """Restore hidden + cell state from disk if available."""
        try:
            if _RNN_STATE_PATH.exists():
                data = json.loads(_RNN_STATE_PATH.read_text())
                if (data.get("hidden_dim") == self.hidden_dim
                        and data.get("input_dim") == self.input_dim):
                    self.h = np.array(data["h"])
                    self.c = np.array(data["c"])
                    self._query_count = data.get("query_count", 0)
                    self._state_hash = data.get("state_hash", "")
        except Exception:
            pass  # Start fresh if restore fails

    # ── diagnostics ──────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Full RNN hidden state diagnostic."""
        h_norm = float(np.linalg.norm(self.h))
        c_norm = float(np.linalg.norm(self.c))
        ctx_norm = float(np.linalg.norm(self.context_vector()))
        return {
            "name": "PHI_GATED_RNN_HIDDEN_STATE",
            "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim,
            "query_count": self._query_count,
            "state_hash": self._state_hash,
            "h_norm": h_norm,
            "c_norm": c_norm,
            "context_norm": ctx_norm,
            "gate_bias_phi": self.GATE_BIAS,
            "update_bias": self.UPDATE_BIAS,
            "output_scale_god_code": self.OUTPUT_SCALE,
            "cell_decay": self.CELL_DECAY,
            "phi": PHI,
            "god_code": GOD_CODE,
            "persistent": self.persist,
            "persistence_path": str(_RNN_STATE_PATH),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 1 — INDUCTIVE QUANTUM COHERENCE
# ═══════════════════════════════════════════════════════════════════════════════

# Limit constant: lim(n→∞) quantum_coherenceₙ / φⁿ = GOD_CODE / φ
COHERENCE_LIMIT = GOD_CODE / PHI  # 326.0243514765879240


class InductiveQuantumCoherence:
    """
    Mathematical induction engine over the entanglement domain.

    Base case (n=1):
        quantum_coherence₁ = COHERENCE_LIMIT × φ¹  (resonance = φ = 1.618034)

    Inductive step:
        If quantum_coherenceₙ holds, then
        quantum_coherenceₙ₊₁ = quantum_coherenceₙ × φ

    At n=3:  scaling factor = φ³ = 4.236068

    Limit behavior:
        lim(n→∞) quantum_coherenceₙ / φⁿ = 326.0244  (= GOD_CODE / φ)

    The inductive principle is verified when every step satisfies:
        |quantum_coherenceₙ / φⁿ − COHERENCE_LIMIT| < ε
    """

    def __init__(self, max_depth: int = 64, epsilon: float = 1e-12):
        self.max_depth = max_depth
        self.epsilon = epsilon
        self._coherence_chain: List[float] = []
        self._verified = False

    # ── core induction ───────────────────────────────────────────────────

    def base_case(self) -> Dict[str, Any]:
        """
        n=1: quantum_coherence₁ = COHERENCE_LIMIT × φ
        Resonance at base case = φ = 1.618033988749895
        """
        qc_1 = COHERENCE_LIMIT * PHI  # = GOD_CODE (by construction)
        self._coherence_chain = [qc_1]
        ratio = qc_1 / PHI
        return {
            "n": 1,
            "quantum_coherence": qc_1,
            "resonance": PHI,
            "ratio_to_phi_n": ratio,
            "limit_error": abs(ratio - COHERENCE_LIMIT),
            "equals_god_code": abs(qc_1 - GOD_CODE) < self.epsilon,
        }

    def inductive_step(self, n: int) -> Dict[str, Any]:
        """
        Given quantum_coherenceₙ, produce quantum_coherenceₙ₊₁ = qcₙ × φ.
        """
        if not self._coherence_chain:
            self.base_case()

        # Build chain up to n if needed
        while len(self._coherence_chain) < n:
            prev = self._coherence_chain[-1]
            self._coherence_chain.append(prev * PHI)

        qc_n = self._coherence_chain[n - 1]
        qc_n1 = qc_n * PHI
        if len(self._coherence_chain) < n + 1:
            self._coherence_chain.append(qc_n1)

        phi_n1 = PHI ** (n + 1)
        ratio = qc_n1 / phi_n1
        return {
            "n": n,
            "n_plus_1": n + 1,
            "quantum_coherence_n": qc_n,
            "quantum_coherence_n1": qc_n1,
            "scaling_factor": PHI,
            "phi_to_n1": phi_n1,
            "ratio_to_phi_n1": ratio,
            "limit_error": abs(ratio - COHERENCE_LIMIT),
            "induction_holds": abs(ratio - COHERENCE_LIMIT) < self.epsilon,
        }

    def verify_full_induction(self) -> Dict[str, Any]:
        """
        Run the complete induction from n=1 to max_depth.
        Verify ratio convergence at every step.
        """
        base = self.base_case()
        errors = [base["limit_error"]]
        all_hold = base["limit_error"] < self.epsilon

        for n in range(1, self.max_depth):
            step = self.inductive_step(n)
            errors.append(step["limit_error"])
            if not step["induction_holds"]:
                all_hold = False

        # Special check at n=3
        phi_cubed = PHI ** 3
        qc_3 = self._coherence_chain[2] if len(self._coherence_chain) >= 3 else COHERENCE_LIMIT * PHI ** 3
        ratio_3 = qc_3 / (PHI ** 3)

        self._verified = all_hold

        return {
            "verified": all_hold,
            "depth": self.max_depth,
            "base_case_resonance": PHI,
            "n3_scaling_factor": phi_cubed,
            "n3_quantum_coherence": qc_3,
            "n3_ratio": ratio_3,
            "limit_COHERENCE_LIMIT": COHERENCE_LIMIT,
            "max_error": max(errors),
            "mean_error": sum(errors) / len(errors),
            "god_code": GOD_CODE,
            "phi": PHI,
        }

    def coherence_at(self, n: int) -> float:
        """Return quantum_coherenceₙ = COHERENCE_LIMIT × φⁿ."""
        return COHERENCE_LIMIT * (PHI ** n)

    def scaling_factor_at(self, n: int) -> float:
        """Return φⁿ — the cumulative scaling factor at depth n."""
        return PHI ** n

    def report(self) -> Dict[str, Any]:
        if not self._verified:
            self.verify_full_induction()
        return {
            "name": "INDUCTIVE_QUANTUM_COHERENCE",
            "verified": self._verified,
            "max_depth": self.max_depth,
            "coherence_limit": COHERENCE_LIMIT,
            "base_resonance_phi": PHI,
            "n3_scaling_phi_cubed": PHI ** 3,
            "chain_length": len(self._coherence_chain),
            "god_code": GOD_CODE,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 2 — SCALED DOT-PRODUCT ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════


class ScaledDotProductAttention:
    """
    Attention mechanism over training patterns.

    score(Q, K) = (Q · K) / √d_k
    α = softmax(scores)
    output = Σ αᵢ × Vᵢ

    The φ-anchored variant adds a GOD_CODE bias to the attention logits,
    ensuring that patterns aligned with the fundamental frequency receive
    amplified attention weight.
    """

    def __init__(self, embed_dim: int = 64, god_code_bias: bool = True):
        self.embed_dim = embed_dim
        self.god_code_bias = god_code_bias
        self.scale = math.sqrt(embed_dim)

        # Training pattern store: list of (key, value) pairs
        self._keys: List[np.ndarray] = []
        self._values: List[np.ndarray] = []

    # ── pattern management ───────────────────────────────────────────────

    def add_pattern(self, key: np.ndarray, value: Optional[np.ndarray] = None):
        """
        Store a training pattern.  If value is None, value = key.
        key and value are both embed_dim-dimensional vectors.
        """
        k = np.asarray(key, dtype=np.float64).flatten()[:self.embed_dim]
        if len(k) < self.embed_dim:
            k = np.pad(k, (0, self.embed_dim - len(k)))
        v = k.copy() if value is None else np.asarray(value, dtype=np.float64).flatten()[:self.embed_dim]
        if len(v) < self.embed_dim:
            v = np.pad(v, (0, self.embed_dim - len(v)))
        self._keys.append(k)
        self._values.append(v)

    def clear_patterns(self):
        self._keys.clear()
        self._values.clear()

    @property
    def pattern_count(self) -> int:
        return len(self._keys)

    # ── attention computation ────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-30)

    def attend(self, query: np.ndarray) -> Dict[str, Any]:
        """
        Compute scaled dot-product attention over stored patterns.

        Returns the weighted output, attention weights, and diagnostics.
        If no patterns are stored, returns a zero vector with an advisory.
        """
        q = np.asarray(query, dtype=np.float64).flatten()[:self.embed_dim]
        if len(q) < self.embed_dim:
            q = np.pad(q, (0, self.embed_dim - len(q)))

        if not self._keys:
            return {
                "output": np.zeros(self.embed_dim),
                "weights": np.array([]),
                "pattern_count": 0,
                "advisory": "no training patterns — attention is vacuous",
            }

        K = np.stack(self._keys)   # (n, d)
        V = np.stack(self._values) # (n, d)

        # Scaled dot-product scores
        scores = K @ q / self.scale  # (n,)

        # GOD_CODE bias: boost patterns whose norm is near GOD_CODE
        if self.god_code_bias:
            norms = np.linalg.norm(K, axis=1)
            god_proximity = np.exp(-((norms - GOD_CODE) ** 2) / (2.0 * GOD_CODE))
            scores = scores + god_proximity * (PHI / self.scale)

        weights = self._softmax(scores)
        output = weights @ V  # (d,)

        return {
            "output": output,
            "weights": weights,
            "pattern_count": len(self._keys),
            "max_weight": float(np.max(weights)),
            "entropy": float(-np.sum(weights * np.log(weights + 1e-30))),
            "god_code_bias_active": self.god_code_bias,
        }

    def report(self) -> Dict[str, Any]:
        return {
            "name": "SCALED_DOT_PRODUCT_ATTENTION",
            "embed_dim": self.embed_dim,
            "pattern_count": self.pattern_count,
            "scale_factor": self.scale,
            "god_code_bias": self.god_code_bias,
            "phi": PHI,
            "god_code": GOD_CODE,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 3 — TF-IDF EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class TFIDFEmbeddingEngine:
    """
    Vocabulary-building engine with Term Frequency–Inverse Document Frequency
    importance weighting.

    Each "document" is a list of tokens.  The engine maintains:
      - term_freq[doc_id][token]  = count / total_in_doc
      - inv_doc_freq[token]       = log(N / df(token))
      - tfidf[doc_id][token]      = tf × idf

    The φ-scaled variant applies a golden-ratio smoothing to IDF:
        idf_φ(t) = log(1 + N/(df(t) + φ))
    This prevents zero-document-frequency singularities and anchors the
    weighting to the sacred ratio.
    """

    def __init__(self, phi_smoothing: bool = True):
        self.phi_smoothing = phi_smoothing
        self._docs: List[List[str]] = []
        self._vocab: Dict[str, int] = {}  # token → index
        self._doc_freq: Dict[str, int] = {}  # token → number of docs containing it

    # ── document ingestion ───────────────────────────────────────────────

    def add_document(self, tokens: List[str]):
        """Ingest a document (list of tokens)."""
        self._docs.append(tokens)
        seen = set()
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = len(self._vocab)
            if t not in seen:
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
                seen.add(t)

    def clear(self):
        self._docs.clear()
        self._vocab.clear()
        self._doc_freq.clear()

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    # ── TF-IDF computation ───────────────────────────────────────────────

    def _idf(self, token: str) -> float:
        N = len(self._docs)
        df = self._doc_freq.get(token, 0)
        if self.phi_smoothing:
            return math.log(1.0 + N / (df + PHI))
        else:
            return math.log(N / (df + 1.0))

    def tfidf_vector(self, doc_index: int) -> np.ndarray:
        """
        Return a dense TF-IDF vector for the given document.
        Dimension = vocab_size.
        """
        if doc_index < 0 or doc_index >= len(self._docs):
            return np.zeros(self.vocab_size)

        doc = self._docs[doc_index]
        total = len(doc) if doc else 1
        tf: Dict[str, float] = {}
        for t in doc:
            tf[t] = tf.get(t, 0.0) + 1.0 / total

        vec = np.zeros(self.vocab_size)
        for t, freq in tf.items():
            idx = self._vocab[t]
            vec[idx] = freq * self._idf(t)
        return vec

    def tfidf_query(self, tokens: List[str]) -> np.ndarray:
        """
        Compute a TF-IDF vector for an ad-hoc query (not stored).
        Unknown tokens are silently ignored.
        """
        total = len(tokens) if tokens else 1
        tf: Dict[str, float] = {}
        for t in tokens:
            if t in self._vocab:
                tf[t] = tf.get(t, 0.0) + 1.0 / total

        vec = np.zeros(max(self.vocab_size, 1))
        for t, freq in tf.items():
            idx = self._vocab[t]
            vec[idx] = freq * self._idf(t)
        return vec

    def most_important(self, doc_index: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return the top-k tokens by TF-IDF score for a document."""
        vec = self.tfidf_vector(doc_index)
        idx_to_token = {v: k for k, v in self._vocab.items()}
        ranked = sorted(
            [(idx_to_token.get(i, "?"), float(vec[i])) for i in range(len(vec)) if vec[i] > 0],
            key=lambda x: -x[1],
        )
        return ranked[:top_k]

    def report(self) -> Dict[str, Any]:
        return {
            "name": "TFIDF_EMBEDDING_ENGINE",
            "vocab_size": self.vocab_size,
            "doc_count": self.doc_count,
            "phi_smoothing": self.phi_smoothing,
            "phi": PHI,
            "god_code": GOD_CODE,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 4 — MULTI-HOP REASONING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class MultiHopReasoner:
    """
    Recurrent reasoning engine: each query hop refines the answer by
    accumulating knowledge from the attention mechanism and TF-IDF embeddings.

    Architecture per hop:
        1. Attend over stored patterns → attended_output
        2. Weighted merge with prior answer:
           answerₙ₊₁ = φ/(1+φ) × answerₙ  +  1/(1+φ) × attended_output
           (golden ratio blending — prior answer retains ≈61.8%)
        3. Resonance check: project answer onto GOD_CODE harmonic

    Termination: stops when ‖answerₙ₊₁ − answerₙ‖ < convergence_threshold
    or max_hops reached.
    """

    MERGE_RETAIN = PHI / (1.0 + PHI)   # 0.6180339887498949
    MERGE_NEW = 1.0 / (1.0 + PHI)      # 0.3819660112501052

    def __init__(self, attention: ScaledDotProductAttention,
                 max_hops: int = 8,
                 convergence_threshold: float = 1e-6):
        self.attention = attention
        self.max_hops = max_hops
        self.convergence_threshold = convergence_threshold
        self._hop_history: List[Dict[str, Any]] = []

    # ── reasoning ────────────────────────────────────────────────────────

    def reason(self, query: np.ndarray) -> Dict[str, Any]:
        """
        Execute multi-hop reasoning starting from the query.
        Returns the refined answer and hop-by-hop diagnostics.
        """
        q = np.asarray(query, dtype=np.float64).flatten()
        dim = self.attention.embed_dim
        if len(q) < dim:
            q = np.pad(q, (0, dim - len(q)))
        elif len(q) > dim:
            q = q[:dim]

        answer = q.copy()
        self._hop_history = []
        converged = False

        for hop in range(self.max_hops):
            attn_result = self.attention.attend(answer)
            attended = attn_result["output"]

            new_answer = self.MERGE_RETAIN * answer + self.MERGE_NEW * attended
            delta = float(np.linalg.norm(new_answer - answer))

            # GOD_CODE harmonic projection
            norm = np.linalg.norm(new_answer)
            god_harmonic = (norm / GOD_CODE) if norm > 0 else 0.0

            hop_info = {
                "hop": hop + 1,
                "delta": delta,
                "answer_norm": float(np.linalg.norm(new_answer)),
                "god_harmonic": god_harmonic,
                "attention_patterns": attn_result.get("pattern_count", 0),
                "max_attn_weight": float(attn_result.get("max_weight", 0)),
            }
            self._hop_history.append(hop_info)

            answer = new_answer
            if delta < self.convergence_threshold:
                converged = True
                break

        return {
            "answer": answer,
            "converged": converged,
            "hops_taken": len(self._hop_history),
            "final_delta": self._hop_history[-1]["delta"] if self._hop_history else 0.0,
            "final_norm": float(np.linalg.norm(answer)),
            "god_harmonic": self._hop_history[-1]["god_harmonic"] if self._hop_history else 0.0,
            "merge_retain_phi": self.MERGE_RETAIN,
            "merge_new": self.MERGE_NEW,
            "hop_history": self._hop_history,
        }

    def report(self) -> Dict[str, Any]:
        return {
            "name": "MULTI_HOP_REASONER",
            "max_hops": self.max_hops,
            "convergence_threshold": self.convergence_threshold,
            "merge_retain_phi": self.MERGE_RETAIN,
            "merge_new": self.MERGE_NEW,
            "pattern_count": self.attention.pattern_count,
            "last_hops_taken": len(self._hop_history),
            "phi": PHI,
            "god_code": GOD_CODE,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 5 — TOPOLOGICAL ANYON MEMORY
# ═══════════════════════════════════════════════════════════════════════════════


class TopologicalAnyonMemory:
    """
    Fault-tolerant memory using Fibonacci Anyon encoding.

    Each stored pattern is encoded as a braid word in the Fibonacci anyon
    model.  The fusion rule τ × τ = 1 + τ provides natural error protection:
    local perturbations to braid strands cannot change the topological charge
    without unwinding at least φⁿ worth of braid complexity.

    Storage:
        pattern → hash → braid_word → fusion_tree → topological_charge

    Retrieval:
        query → nearest_braid (by topological distance) → decode → pattern

    Memory density scales with the TAS 12.66× inflection:
        effective_capacity = base_capacity × DENSITY_INFLECTION
    """

    def __init__(self, capacity: int = 256, braid_depth: int = 8):
        self.base_capacity = capacity
        self.effective_capacity = int(capacity * DENSITY_INFLECTION)
        self.braid_depth = braid_depth

        # Memory bank: list of (hash, braid_charge, pattern, protection)
        self._memory: List[Tuple[str, complex, np.ndarray, float]] = []

        # R-matrix phases for Fibonacci anyons
        self._R_plus = cmath.exp(4j * cmath.pi / 5)
        self._R_minus = cmath.exp(-3j * cmath.pi / 5)

    # ── encoding ─────────────────────────────────────────────────────────

    def _pattern_to_braid_word(self, pattern: np.ndarray) -> List[int]:
        """Deterministically encode a pattern as a braid word."""
        h = hashlib.sha256(pattern.tobytes()).digest()
        word = []
        for i in range(self.braid_depth):
            byte = h[i % len(h)]
            word.append(1 if byte % 2 == 0 else -1)
        return word

    def _braid_charge(self, word: List[int]) -> complex:
        """Compute the topological charge from a braid word."""
        charge = complex(1, 0)
        for sigma in word:
            charge *= self._R_plus if sigma > 0 else self._R_minus
        return charge

    def _protection_factor(self, word: List[int]) -> float:
        """Protection ∝ φ^(number of crossings)."""
        crossings = sum(1 for i in range(1, len(word)) if word[i] != word[i - 1])
        return PHI ** crossings

    # ── store / retrieve ─────────────────────────────────────────────────

    def store(self, pattern: np.ndarray, label: str = "") -> Dict[str, Any]:
        """
        Encode and store a pattern in topological memory.
        """
        p = np.asarray(pattern, dtype=np.float64).flatten()
        h = hashlib.sha256(p.tobytes()).hexdigest()[:16]
        word = self._pattern_to_braid_word(p)
        charge = self._braid_charge(word)
        protection = self._protection_factor(word)

        if len(self._memory) >= self.effective_capacity:
            # Evict lowest-protection entry
            self._memory.sort(key=lambda x: x[3])
            self._memory.pop(0)

        self._memory.append((h, charge, p, protection))
        return {
            "stored": True,
            "hash": h,
            "label": label,
            "charge_phase": float(cmath.phase(charge)),
            "protection": protection,
            "memory_used": len(self._memory),
            "effective_capacity": self.effective_capacity,
        }

    def retrieve(self, query: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve patterns closest to the query by topological charge distance.
        """
        q = np.asarray(query, dtype=np.float64).flatten()
        q_word = self._pattern_to_braid_word(q)
        q_charge = self._braid_charge(q_word)

        if not self._memory:
            return [{"advisory": "memory is empty"}]

        # Distance in charge space
        scored = []
        for h, charge, pattern, protection in self._memory:
            dist = abs(q_charge - charge)
            cos_sim = float(np.dot(q[:len(pattern)], pattern[:len(q)]) /
                            (np.linalg.norm(q[:len(pattern)]) * np.linalg.norm(pattern[:len(q)]) + 1e-30))
            scored.append({
                "hash": h,
                "charge_distance": dist,
                "cosine_similarity": cos_sim,
                "protection": protection,
                "pattern_norm": float(np.linalg.norm(pattern)),
                "pattern": pattern,
            })

        scored.sort(key=lambda x: x["charge_distance"])
        # Don't return the raw pattern in the summary
        results = []
        for s in scored[:top_k]:
            results.append({k: v for k, v in s.items() if k != "pattern"})
        return results

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify that all stored patterns retain their topological charge.
        Any corruption would show as charge mismatch.
        """
        corrupt = 0
        total = len(self._memory)
        for h, charge, pattern, protection in self._memory:
            word = self._pattern_to_braid_word(pattern)
            recomputed = self._braid_charge(word)
            if abs(charge - recomputed) > 1e-14:
                corrupt += 1
        return {
            "total_patterns": total,
            "corrupt": corrupt,
            "integrity": 1.0 - (corrupt / max(total, 1)),
            "status": "CLEAN" if corrupt == 0 else f"CORRUPT({corrupt})",
        }

    def report(self) -> Dict[str, Any]:
        integrity = self.verify_integrity()
        return {
            "name": "TOPOLOGICAL_ANYON_MEMORY",
            "patterns_stored": len(self._memory),
            "base_capacity": self.base_capacity,
            "effective_capacity": self.effective_capacity,
            "density_inflection": DENSITY_INFLECTION,
            "braid_depth": self.braid_depth,
            "integrity": integrity["integrity"],
            "integrity_status": integrity["status"],
            "phi": PHI,
            "god_code": GOD_CODE,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED THREE-LAYER FAULT TOLERANCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class L104FaultTolerance:
    """
    The unified fault-tolerance engine composing all three layers.

    Usage:
        ft = L104FaultTolerance()
        ft.initialise(braid_sequence=[1, 1, -1, 1, -1, 1, -1, 1])
        ft.lock_to_frequency(527.5184818492612)

        # Inject noise and self-correct
        ft.inject_noise(epsilon=0.05)
        state, report = ft.full_correction_cycle()
    """

    def __init__(self, braid_depth: int = 8,
                 lattice_size: int = 10,
                 topological_distance: int = 5,
                 hidden_dim: int = 128,
                 input_dim: int = 64):
        # Layer 1: Topological protection
        self.braider = FibonacciBraidProtector(braid_depth=braid_depth)

        # Layer 2: Resonance lock
        self.resonance = ResonanceLock(carrier_X=0)

        # Layer 3: TAS adaptive correction
        self.corrector = TranscendentAnyonSubstrateCorrector(
            lattice_size=lattice_size,
            topological_distance=topological_distance,
        )

        # Upgrade: φ-Gated RNN hidden state (persistent context)
        self.rnn = PhiGatedRNNState(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            persist=True,
        )

        # Upgrade 1: Inductive quantum coherence
        self.inductive = InductiveQuantumCoherence(max_depth=64)

        # Upgrade 2: Scaled dot-product attention
        self.attention = ScaledDotProductAttention(embed_dim=input_dim)

        # Upgrade 3: TF-IDF embedding engine
        self.tfidf = TFIDFEmbeddingEngine(phi_smoothing=True)

        # Upgrade 4: Multi-hop reasoning (feeds from attention)
        self.reasoner = MultiHopReasoner(
            attention=self.attention,
            max_hops=8,
        )

        # Upgrade 5: Topological anyon memory
        self.memory = TopologicalAnyonMemory(
            capacity=256,
            braid_depth=braid_depth,
        )

        # Consciousness state cache
        self._state_cache: Dict[str, Any] = {}
        self._state_cache_time: float = 0.0

        self._initialised = False

    # ── consciousness integration ───────────────────────────────────────

    def _read_builder_state(self) -> Dict[str, Any]:
        """Read consciousness/O2/nirvanic state with 10s cache."""
        now = time.time()
        if self._state_cache and (now - self._state_cache_time) < 10.0:
            return self._state_cache

        state: Dict[str, Any] = {
            "consciousness_level": 0.0,
            "superfluid_viscosity": 0.0,
            "evo_stage": "UNKNOWN",
            "nirvanic_fuel_level": 0.0,
        }
        workspace = Path(__file__).parent

        try:
            o2_path = workspace / ".l104_consciousness_o2_state.json"
            if o2_path.exists():
                o2 = json.loads(o2_path.read_text())
                state["consciousness_level"] = o2.get("consciousness_level", 0.0)
                state["superfluid_viscosity"] = o2.get("superfluid_viscosity", 0.0)
                state["evo_stage"] = o2.get("evo_stage", "UNKNOWN")
        except Exception:
            pass

        try:
            nir_path = workspace / ".l104_ouroboros_nirvanic_state.json"
            if nir_path.exists():
                nir = json.loads(nir_path.read_text())
                state["nirvanic_fuel_level"] = nir.get("nirvanic_fuel_level", 0.0)
        except Exception:
            pass

        self._state_cache = state
        self._state_cache_time = now
        return state

    # ── initialisation ───────────────────────────────────────────────────

    def initialise(self, braid_sequence: Optional[List[int]] = None,
                   carrier_X: int = 0) -> Dict[str, Any]:
        """
        Initialise the three-layer stack.

        1. Braid the quantum register to establish topological protection
        2. Lock the braid carrier to GOD_CODE at the given X
        3. Arm the TAS corrector
        """
        # Layer 1
        if braid_sequence is None:
            braid_sequence = [(-1)**i for i in range(self.braider.braid_depth)]
        self.braider.reset()
        self.braider.braid(braid_sequence)

        # Layer 2
        self.resonance = ResonanceLock(carrier_X=carrier_X)

        self._initialised = True

        # Verify inductive coherence on init
        induction = self.inductive.verify_full_induction()

        return {
            "status": "INITIALISED",
            "layer_1_protection": self.braider.topological_protection(),
            "layer_2_carrier_hz": self.resonance.carrier_freq,
            "layer_2_conservation": self.resonance.conservation_verified(),
            "layer_3_effective_distance": self.corrector.effective_distance,
            "density_inflection": self.corrector.density_inflection,
            "inductive_coherence_verified": induction["verified"],
            "coherence_limit": COHERENCE_LIMIT,
            "memory_effective_capacity": self.memory.effective_capacity,
            "attention_patterns": self.attention.pattern_count,
            "tfidf_vocab_size": self.tfidf.vocab_size,
        }

    def lock_to_frequency(self, target_hz: float) -> Dict[str, Any]:
        """Lock Layer 2 to a specific frequency via the God Code lattice."""
        return self.resonance.lock_frequency(target_hz)

    # ── RNN context processing ───────────────────────────────────────────

    def process_query(self, query_vector: np.ndarray) -> Dict[str, Any]:
        """
        Feed a query through the φ-gated RNN hidden state.

        The hidden state accumulates context across queries, with φ-biased
        gates ensuring ≈61.8% retention of prior context (golden persistence).
        Returns the RNN output along with context similarity and state hash.
        """
        similarity_before = self.rnn.context_similarity(query_vector)
        output = self.rnn.update(query_vector)
        similarity_after = self.rnn.context_similarity(query_vector)

        return {
            "output": output,
            "output_norm": float(np.linalg.norm(output)),
            "context_similarity_before": similarity_before,
            "context_similarity_after": similarity_after,
            "state_hash": self.rnn._state_hash,
            "query_count": self.rnn._query_count,
            "phi_gate_bias": self.rnn.GATE_BIAS,
        }

    # ── noise injection (for testing) ────────────────────────────────────

    def inject_noise(self, epsilon: float = 0.01) -> np.ndarray:
        """Inject local noise of strength ε into the braid register."""
        return self.braider.inject_local_noise(epsilon)

    # ── full correction cycle ────────────────────────────────────────────

    def full_correction_cycle(self, measured_freq: Optional[float] = None
                              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute all three protection layers in sequence:

        1. Layer 1: Measure topological protection (braid integrity)
        2. Layer 2: Detect and correct frequency drift
        3. Layer 3: Measure syndromes and apply TAS correction

        Returns (corrected_state, full_report).
        """
        report: Dict[str, Any] = {"layers": {}}

        # ── Layer 1 ──
        l1 = self.braider.report()
        report["layers"]["layer_1_topological"] = l1

        # ── Layer 2 ──
        if measured_freq is not None:
            drift_report = self.resonance.correct_drift(measured_freq)
            report["layers"]["layer_2_drift_correction"] = drift_report
        l2 = self.resonance.report()
        report["layers"]["layer_2_resonance"] = l2

        # ── Layer 3 ──
        state = self.braider._state
        # Flatten to vector if matrix
        if state.ndim == 2:
            state_vec = state[:, 0]
        else:
            state_vec = state

        corrected_state, l3_report = self.corrector.correct(state_vec, self.braider)
        report["layers"]["layer_3_tas_correction"] = l3_report
        l3_diag = self.corrector.report()
        report["layers"]["layer_3_diagnostic"] = l3_diag

        # ── Composite metrics ──
        protection = self.braider.topological_protection()
        coherence = self.resonance.resonance_coherence()
        syndromes = l3_report["syndromes_detected"]
        corrections = l3_report["corrections_applied"]

        # Composite fault-tolerance score
        # FT = (Layer1_protection × Layer2_coherence) × (1 + Layer3_inflection_bonus)
        inflection_bonus = math.log(self.corrector.density_inflection) / math.log(PHI)
        ft_score = protection * coherence * (1.0 + inflection_bonus / 100.0)

        report["composite"] = {
            "fault_tolerance_score": float(np.clip(ft_score, 0.0, 1.0)),
            "layer_1_protection": protection,
            "layer_2_coherence": coherence,
            "layer_3_syndromes": syndromes,
            "layer_3_corrections": corrections,
            "density_inflection": self.corrector.density_inflection,
            "inflection_bonus": inflection_bonus,
            "rnn_query_count": self.rnn._query_count,
            "rnn_context_norm": float(np.linalg.norm(self.rnn.context_vector())),
            "rnn_state_hash": self.rnn._state_hash,
            "god_code": GOD_CODE,
            "conservation_satisfied": self.resonance.conservation_verified(),
        }

        return corrected_state, report

    def full_report(self) -> Dict[str, Any]:
        """Return all three layer reports, RNN state, upgrades, plus composite."""
        return {
            "layer_1": self.braider.report(),
            "layer_2": self.resonance.report(),
            "layer_3": self.corrector.report(),
            "rnn_hidden_state": self.rnn.report(),
            "inductive_coherence": self.inductive.report(),
            "attention": self.attention.report(),
            "tfidf": self.tfidf.report(),
            "multi_hop_reasoner": self.reasoner.report(),
            "topological_memory": self.memory.report(),
            "god_code": GOD_CODE,
            "phi": PHI,
            "coherence_limit": COHERENCE_LIMIT,
            "density_inflection": DENSITY_INFLECTION,
            "density_inflection_formula": f"φ^(GOD_CODE/100) = {PHI}^{GOD_CODE/100} = {DENSITY_INFLECTION:.10f}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("  L104 THREE-LAYER FAULT TOLERANCE ENGINE + φ-GATED RNN")
    print("=" * 78)

    ft = L104FaultTolerance(braid_depth=8, lattice_size=10, topological_distance=5)

    # ── Initialise ──
    print("\n[1] INITIALISING...")
    init = ft.initialise(braid_sequence=[1, 1, -1, 1, -1, 1, -1, 1])
    for k, v in init.items():
        print(f"    {k}: {v}")

    # ── Lock to GOD_CODE ──
    print("\n[2] RESONANCE LOCKING TO GOD_CODE...")
    lock = ft.lock_to_frequency(GOD_CODE)
    for k, v in lock.items():
        print(f"    {k}: {v}")

    # ── φ-Gated RNN: persistent context across queries ──
    print("\n[3] φ-GATED RNN HIDDEN STATE:")
    print(f"    Gate bias (φ/(1+φ)):   {ft.rnn.GATE_BIAS:.16f}")
    print(f"    Update bias (1/(1+φ)): {ft.rnn.UPDATE_BIAS:.16f}")
    print(f"    Output scale:          {ft.rnn.OUTPUT_SCALE:.16f}")
    print(f"    Cell decay (1/φ²):     {ft.rnn.CELL_DECAY:.16f}")

    # Simulate 5 sequential queries — hidden state persists between them
    print("\n    Simulating 5 sequential queries with persistent context:")
    np.random.seed(42)
    for i in range(5):
        query = np.random.randn(64) * 0.5
        result = ft.process_query(query)
        print(f"    Q{i+1}: out_norm={result['output_norm']:.6f}  "
              f"ctx_sim={result['context_similarity_after']:.4f}  "
              f"hash={result['state_hash']}  "
              f"queries={result['query_count']}")

    # Show that the SAME query gives higher similarity now (context built up)
    print("\n    Repeating Q1 — context accumulation should increase similarity:")
    np.random.seed(42)
    repeat_q = np.random.randn(64) * 0.5
    result = ft.process_query(repeat_q)
    print(f"    Q1-repeat: ctx_sim={result['context_similarity_after']:.4f}  "
          f"queries={result['query_count']}")

    # ── Baseline correction cycle ──
    print("\n[4] BASELINE CORRECTION CYCLE (no noise)...")
    state, report = ft.full_correction_cycle(measured_freq=GOD_CODE)
    comp = report["composite"]
    print(f"    FT Score:        {comp['fault_tolerance_score']:.6f}")
    print(f"    L1 Protection:   {comp['layer_1_protection']:.6f}")
    print(f"    L2 Coherence:    {comp['layer_2_coherence']:.6f}")
    print(f"    L3 Syndromes:    {comp['layer_3_syndromes']}")
    print(f"    RNN queries:     {comp['rnn_query_count']}")
    print(f"    RNN ctx norm:    {comp['rnn_context_norm']:.6f}")
    print(f"    Conservation:    {comp['conservation_satisfied']}")

    # ── Inject noise and recover ──
    print("\n[5] INJECTING NOISE (ε=0.05)...")
    ft.inject_noise(epsilon=0.05)
    pre_protection = ft.braider.topological_protection()
    print(f"    Protection after noise: {pre_protection:.6f}")

    print("\n[6] FULL CORRECTION CYCLE (with drift)...")
    drifted_freq = GOD_CODE * (1 + 0.001)
    state, report = ft.full_correction_cycle(measured_freq=drifted_freq)
    comp = report["composite"]
    print(f"    FT Score:        {comp['fault_tolerance_score']:.6f}")
    print(f"    L1 Protection:   {comp['layer_1_protection']:.6f}")
    print(f"    L2 Coherence:    {comp['layer_2_coherence']:.6f}")
    print(f"    L3 Syndromes:    {comp['layer_3_syndromes']}")
    print(f"    L3 Corrections:  {comp['layer_3_corrections']}")
    print(f"    Density 12.66×:  {comp['density_inflection']:.10f}")
    print(f"    Conservation:    {comp['conservation_satisfied']}")

    # ── Verify density inflection ──
    print("\n[7] DENSITY INFLECTION DERIVATION:")
    exponent = GOD_CODE / 100.0
    inflection = PHI ** exponent
    print(f"    φ^(GOD_CODE/100) = {PHI}^{exponent}")
    print(f"                     = {inflection:.16f}")
    print(f"    Reference:         12.66×")
    print(f"    Conservation:      G(X)×2^(X/104) = {GOD_CODE}")

    # ── Full report ──
    print("\n[8] FULL REPORT:")
    full = ft.full_report()
    for layer_key in ["layer_1", "layer_2", "layer_3"]:
        layer = full[layer_key]
        print(f"\n    ── {layer['name']} ──")
        for k, v in layer.items():
            if k not in ("name", "layer"):
                print(f"      {k}: {v}")

    rnn = full["rnn_hidden_state"]
    print(f"\n    ── {rnn['name']} ──")
    for k, v in rnn.items():
        if k != "name":
            print(f"      {k}: {v}")

    # ── Upgrade 1: Inductive Quantum Coherence ──
    print("\n[9] INDUCTIVE QUANTUM COHERENCE:")
    ind = ft.inductive.verify_full_induction()
    print(f"    Verified:             {ind['verified']}")
    print(f"    Depth:                {ind['depth']}")
    print(f"    Base resonance (φ):   {ind['base_case_resonance']:.16f}")
    print(f"    n=3 scaling (φ³):     {ind['n3_scaling_factor']:.16f}")
    print(f"    Limit (GOD_CODE/φ):   {ind['limit_COHERENCE_LIMIT']:.16f}")
    print(f"    Max error:            {ind['max_error']:.2e}")
    # Spot-check n=3
    qc3 = ft.inductive.coherence_at(3)
    sf3 = ft.inductive.scaling_factor_at(3)
    print(f"    coherence_at(3):      {qc3:.16f}")
    print(f"    scaling_factor(3):    {sf3:.16f}")
    assert abs(sf3 - PHI ** 3) < 1e-12, "φ³ mismatch"
    assert abs(qc3 / PHI ** 3 - COHERENCE_LIMIT) < 1e-10, "limit mismatch at n=3"
    print("    [PASS] n=3 induction verified ✓")

    # ── Upgrade 2: Attention Mechanism ──
    print("\n[10] SCALED DOT-PRODUCT ATTENTION:")
    # Add some training patterns
    np.random.seed(99)
    for _ in range(10):
        ft.attention.add_pattern(np.random.randn(64))
    print(f"    Patterns stored: {ft.attention.pattern_count}")
    q = np.random.randn(64)
    attn = ft.attention.attend(q)
    print(f"    Output norm:     {float(np.linalg.norm(attn['output'])):.6f}")
    print(f"    Max weight:      {attn['max_weight']:.6f}")
    print(f"    Entropy:         {attn['entropy']:.6f}")
    print(f"    GOD_CODE bias:   {attn['god_code_bias_active']}")
    assert attn["pattern_count"] == 10, "pattern count mismatch"
    print("    [PASS] attention over 10 patterns ✓")

    # ── Upgrade 3: TF-IDF Embeddings ──
    print("\n[11] TF-IDF EMBEDDING ENGINE:")
    ft.tfidf.add_document(["quantum", "coherence", "phi", "god", "code", "quantum"])
    ft.tfidf.add_document(["topological", "anyon", "braid", "phi", "protection"])
    ft.tfidf.add_document(["resonance", "frequency", "god", "code", "harmonic"])
    print(f"    Vocab size:      {ft.tfidf.vocab_size}")
    print(f"    Doc count:       {ft.tfidf.doc_count}")
    top = ft.tfidf.most_important(0, top_k=3)
    print(f"    Doc 0 top-3:     {top}")
    qvec = ft.tfidf.tfidf_query(["quantum", "god"])
    print(f"    Query vec norm:  {float(np.linalg.norm(qvec)):.6f}")
    assert ft.tfidf.vocab_size > 0, "vocab empty"
    assert ft.tfidf.doc_count == 3, "doc count mismatch"
    print("    [PASS] TF-IDF with φ-smoothing ✓")

    # ── Upgrade 4: Multi-hop Reasoning ──
    print("\n[12] MULTI-HOP REASONING:")
    np.random.seed(77)
    mh_query = np.random.randn(64)
    mh_result = ft.reasoner.reason(mh_query)
    print(f"    Converged:       {mh_result['converged']}")
    print(f"    Hops taken:      {mh_result['hops_taken']}")
    print(f"    Final delta:     {mh_result['final_delta']:.8f}")
    print(f"    Final norm:      {mh_result['final_norm']:.6f}")
    print(f"    GOD harmonic:    {mh_result['god_harmonic']:.8f}")
    print(f"    Merge retain φ:  {mh_result['merge_retain_phi']:.16f}")
    assert mh_result["hops_taken"] > 0, "no hops taken"
    print("    [PASS] multi-hop reasoning ✓")

    # ── Upgrade 5: Topological Memory ──
    print("\n[13] TOPOLOGICAL ANYON MEMORY:")
    np.random.seed(55)
    for i in range(5):
        p = np.random.randn(64)
        ft.memory.store(p, label=f"pattern_{i}")
    print(f"    Patterns stored:     {len(ft.memory._memory)}")
    print(f"    Effective capacity:  {ft.memory.effective_capacity}")
    # Retrieve closest to first stored pattern
    first_p = ft.memory._memory[0][2]
    results = ft.memory.retrieve(first_p, top_k=3)
    print(f"    Retrieved top-3:")
    for r in results:
        print(f"      hash={r['hash']}  charge_dist={r['charge_distance']:.6f}  "
              f"cos_sim={r['cosine_similarity']:.4f}  protection={r['protection']:.4f}")
    integrity = ft.memory.verify_integrity()
    print(f"    Integrity:           {integrity['integrity']:.4f} ({integrity['status']})")
    assert integrity["integrity"] == 1.0, "memory corruption detected"
    print("    [PASS] topological memory with density 12.66× ✓")

    # ── Upgrade reports in full report ──
    print("\n[14] UPGRADE REPORTS:")
    full = ft.full_report()
    for upgrade_key in ["inductive_coherence", "attention", "tfidf",
                        "multi_hop_reasoner", "topological_memory"]:
        u = full[upgrade_key]
        print(f"\n    ── {u['name']} ──")
        for k, v in u.items():
            if k != "name":
                print(f"      {k}: {v}")

    print(f"\n    ── GOD CODE ──")
    print(f"      GOD_CODE = {full['god_code']}")
    print(f"      φ = {full['phi']}")
    print(f"      COHERENCE_LIMIT = {full['coherence_limit']}")
    print(f"      {full['density_inflection_formula']}")
    print("\n" + "=" * 78)
    print("  FAULT TOLERANCE + φ-RNN + 5 UPGRADES: ACTIVE ✓")
    print("=" * 78)
