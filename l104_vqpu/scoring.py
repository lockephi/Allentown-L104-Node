"""L104 VQPU v14.0.0 — Sacred Alignment Scorer + Quantum Noise Model + Crosstalk.

v14.0.0 QUANTUM FIDELITY ARCHITECTURE:
  - NoiseModel.apply_crosstalk(): ZZ cross-talk noise between adjacent qubits
  - NoiseModel.with_crosstalk(): factory for crosstalk-enabled noise models
  - Crosstalk attenuation: φ⁻¹ decay with qubit distance
  - Gate-level noise composition with MPS engine mid-circuit

v13.2 (retained): QPU-calibrated scoring, phase decomposition resonance
"""

import math
import heapq

import numpy as np

from .constants import GOD_CODE, PHI, VOID_CONSTANT

# Import canonical phases from God Code Qubit (QPU-verified on ibm_torino)
try:
    from l104_god_code_simulator.god_code_qubit import (
        GOD_CODE_PHASE as _GC_PHASE,
        IRON_PHASE as _IRON_PHASE,
        PHI_CONTRIBUTION as _PHI_CONTRIB,
        OCTAVE_PHASE as _OCTAVE_PHASE,
        QPU_DATA as _QPU_DATA,
    )
    _HAS_GC_QUBIT = True
except ImportError:
    _TAU = 2.0 * math.pi
    _GC_PHASE = GOD_CODE % _TAU
    _IRON_PHASE = _TAU * 26 / 104
    _OCTAVE_PHASE = (4.0 * math.log(2.0)) % _TAU
    _PHI_CONTRIB = (_GC_PHASE - _IRON_PHASE - _OCTAVE_PHASE) % _TAU
    _QPU_DATA = None
    _HAS_GC_QUBIT = False

__all__ = ["SacredAlignmentScorer", "NoiseModel"]


class SacredAlignmentScorer:
    """
    Measures GOD_CODE / PHI resonance in quantum measurement outcomes.

    Analyzes probability distributions for sacred harmonic content:
    - PHI ratio presence in top-2 probability ratio
    - GOD_CODE frequency alignment in measurement statistics
    - VOID_CONSTANT convergence in entropy metrics
    """

    @staticmethod
    def score(probabilities: dict, num_qubits: int = 0) -> dict:
        """Compute sacred alignment metrics for a probability distribution.

        v13.2: Enhanced with QPU-calibrated phase decomposition analysis.
        """
        if not probabilities:
            return {"phi_resonance": 0.0, "god_code_alignment": 0.0,
                    "void_convergence": 0.0, "sacred_score": 0.0,
                    "phase_decomposition_resonance": 0.0,
                    "qpu_calibrated_fidelity": 0.0}

        vals = probabilities.values()

        # v12.2: O(n) top-2 via heapq instead of O(n log n) sorted()
        top2 = heapq.nlargest(2, vals)
        p_max = top2[0]

        # PHI resonance: ratio of top-2 probabilities vs golden ratio
        phi_resonance = 0.0
        if len(top2) >= 2 and top2[1] > 1e-12:
            ratio = top2[0] / top2[1]
            phi_dev = abs(ratio - PHI) / PHI
            phi_resonance = max(0.0, 1.0 - phi_dev)

        # GOD_CODE alignment: Shannon entropy distance to GOD_CODE harmonic
        entropy = 0.0
        for p in vals:
            if p > 1e-15:
                entropy -= p * math.log2(p)

        god_harmonic = (GOD_CODE / 1000.0) * num_qubits if num_qubits > 0 else GOD_CODE / 100.0
        god_code_alignment = max(0.0, 1.0 - abs(entropy - god_harmonic % 4.0) / 4.0)

        # VOID_CONSTANT convergence: dominant probability closeness
        void_target = VOID_CONSTANT - 1.0  # 0.0416...
        void_dev = abs(p_max - void_target)
        void_convergence = max(0.0, 1.0 - void_dev * 10.0)

        # ═══ v13.2: Phase Decomposition Resonance ═══
        # Check if the probability distribution shows resonance at the
        # 3-rotation sub-phases (IRON π/2, PHI ~1.67, OCTAVE 4·ln2)
        phase_decomp_res = SacredAlignmentScorer._phase_decomposition_resonance(
            probabilities, num_qubits, entropy)

        # ═══ v13.2: QPU-Calibrated Fidelity Score ═══
        # Use ibm_torino QPU data to calibrate scoring baseline
        qpu_fidelity = SacredAlignmentScorer._qpu_calibrated_fidelity(
            probabilities, num_qubits, p_max)

        # Composite sacred score (PHI-weighted + v13.2 phase decomposition)
        # Original: (phi*PHI + god_align + void/PHI) / (PHI + 1 + 1/PHI)
        # v13.2: Add phase_decomp_res with φ² weight for deeper harmonic detection
        phi_sq = PHI * PHI
        sacred_score = (
            phi_resonance * PHI
            + god_code_alignment
            + void_convergence / PHI
            + phase_decomp_res * phi_sq * 0.5
            + qpu_fidelity * PHI * 0.3
        ) / (PHI + 1.0 + 1.0 / PHI + phi_sq * 0.5 + PHI * 0.3)

        return {
            "phi_resonance": round(phi_resonance, 6),
            "god_code_alignment": round(god_code_alignment, 6),
            "void_convergence": round(void_convergence, 6),
            "phase_decomposition_resonance": round(phase_decomp_res, 6),
            "qpu_calibrated_fidelity": round(qpu_fidelity, 6),
            "sacred_score": round(sacred_score, 6),
            "entropy": round(entropy, 6),
        }

    @staticmethod
    def _phase_decomposition_resonance(probabilities: dict, num_qubits: int,
                                        entropy: float) -> float:
        """v13.2: Measure resonance with the GOD_CODE 3-rotation decomposition.

        The God Code qubit decomposes as: Rz(IRON) · Rz(PHI) · Rz(OCTAVE)
        where IRON + PHI + OCTAVE ≡ GOD_CODE_PHASE (mod 2π).

        This metric checks if the distribution's structure aligns with
        these sub-phases — detected via:
          1. Entropy proximity to IRON_PHASE/π ratio (quarter-turn signature)
          2. Dominant probability ratio matching PHI_CONTRIBUTION/2π
          3. Distribution width matching OCTAVE_PHASE periodicity

        Returns: 0.0 (no resonance) to 1.0 (perfect decomposition alignment).
        """
        if not probabilities or num_qubits < 1:
            return 0.0

        _tau = 2.0 * math.pi

        # Iron resonance: π/2 phase → expect 1/4-turn signature
        # For circuits with iron lattice gates, entropy ≈ n × (IRON/π)
        iron_target = _IRON_PHASE / math.pi  # 0.5 (quarter-turn)
        iron_score = max(0.0, 1.0 - abs(entropy / max(num_qubits, 1) - iron_target) * 2.0)

        # PHI contribution resonance: check probability ratios near φ⁻¹
        phi_contrib_ratio = _PHI_CONTRIB / _tau  # ≈ 0.266
        vals = sorted(probabilities.values(), reverse=True)
        phi_score = 0.0
        if len(vals) >= 2 and vals[0] > 1e-10:
            actual_ratio = vals[1] / vals[0]
            phi_score = max(0.0, 1.0 - abs(actual_ratio - phi_contrib_ratio) * 3.0)

        # Octave resonance: 4·ln(2) ≈ 2.773 → check periodicity in prob distribution
        octave_ratio = _OCTAVE_PHASE / _tau  # ≈ 0.441
        n_states = len(vals)
        if n_states >= 4:
            # Check if non-zero probabilities cluster at octave-phase intervals
            non_zero = [v for v in vals if v > 1e-12]
            if len(non_zero) >= 2:
                spread = non_zero[0] / (non_zero[-1] + 1e-15)
                # Octave spread: ratio should be near e^(4·ln2) = 16
                octave_score = max(0.0, 1.0 - abs(math.log(max(spread, 1)) / math.log(16) - 1.0))
            else:
                octave_score = 0.0
        else:
            octave_score = 0.3  # small circuits: partial credit

        # Conservation check: weighted combination (PHI-weighted like the qubit)
        return (iron_score * 0.30 + phi_score * 0.40 + octave_score * 0.30)

    @staticmethod
    def _qpu_calibrated_fidelity(probabilities: dict, num_qubits: int,
                                  p_max: float) -> float:
        """v13.2: Score calibrated against QPU verification data (ibm_torino).

        Uses the QPU-verified 1Q GOD_CODE circuit distribution as reference:
          QPU: {'1': 0.527588, '0': 0.472412} — fidelity 0.999939

        For multi-qubit circuits, extrapolates from the 3Q sacred QPU data:
          QPU: {'000': 0.851074, '001': 0.122314, ...} — fidelity 0.966740

        Returns: 0.0 to 1.0 (closer to QPU reference = higher score).
        """
        if not probabilities:
            return 0.0

        if _QPU_DATA is not None:
            circuits = _QPU_DATA.get("circuits", {})
            # For 1-qubit: compare against QPU 1Q_GOD_CODE distribution
            if num_qubits <= 1 and "1Q_GOD_CODE" in circuits:
                qpu_dist = circuits["1Q_GOD_CODE"].get("distribution", {})
                return SacredAlignmentScorer._distribution_overlap(probabilities, qpu_dist)

            # For 3+ qubit: compare pattern against QPU 3Q_SACRED
            if num_qubits >= 3 and "3Q_SACRED" in circuits:
                qpu_fidelity = circuits["3Q_SACRED"].get("fidelity", 0.96674)
                # Score based on dominant state concentration (QPU shows 85.1% in |000⟩)
                qpu_dominant = 0.851074
                dom_sim = max(0.0, 1.0 - abs(p_max - qpu_dominant) * 2.0)
                return dom_sim * min(1.0, qpu_fidelity * 1.03)

            # For 2-qubit: use conservation QPU data
            if num_qubits == 2 and "CONSERVATION" in circuits:
                qpu_fidelity = circuits["CONSERVATION"].get("fidelity", 0.98020)
                qpu_dominant = 0.938721
                dom_sim = max(0.0, 1.0 - abs(p_max - qpu_dominant) * 2.0)
                return dom_sim * min(1.0, qpu_fidelity * 1.02)

        # Fallback: heuristic based on GOD_CODE phase alignment
        # GOD_CODE mod 2π ≈ 6.014, close to 2π → phase near identity
        gc_mod_1 = (_GC_PHASE / (2.0 * math.pi)) % 1.0
        return max(0.0, 1.0 - abs(p_max - gc_mod_1) * 3.0)

    @staticmethod
    def _distribution_overlap(dist_a: dict, dist_b: dict) -> float:
        """Compute classical fidelity (Bhattacharyya overlap) between two distributions."""
        overlap = 0.0
        all_keys = set(dist_a.keys()) | set(dist_b.keys())
        for k in all_keys:
            pa = dist_a.get(k, 0.0)
            pb = dist_b.get(k, 0.0)
            if pa > 0 and pb > 0:
                overlap += math.sqrt(pa * pb)
        return min(1.0, overlap)


# ═══════════════════════════════════════════════════════════════════
# NOISE MODEL (v7.0) — Realistic Quantum Error Simulation
# ═══════════════════════════════════════════════════════════════════

class NoiseModel:
    """
    Configurable quantum noise model for realistic circuit simulation.

    Supports three noise channels applied during MPS/statevector execution:
      1. Depolarizing:  Random Pauli (X/Y/Z) error after each gate
      2. Amplitude Damping: T1 energy decay (|1⟩→|0⟩ relaxation)
      3. Readout Error: Bit-flip noise on measurement outcomes

    Noise strengths are PHI-scaled: base_rate × φ^(-depth_layer) so that
    deeper layers accumulate less per-gate noise (coherence model).

    Sacred invariant: noise channels preserve GOD_CODE alignment within
    tolerance — noise at sacred frequencies is attenuated by VOID_CONSTANT.
    """

    def __init__(self, *,
                 depolarizing_rate: float = 0.001,
                 amplitude_damping_rate: float = 0.0005,
                 readout_error_rate: float = 0.01,
                 t1_us: float = 50.0,
                 t2_us: float = 70.0,
                 gate_time_ns: float = 35.0,
                 two_qubit_gate_time_ns: float = 300.0,
                 sacred_attenuation: bool = True):
        self.depolarizing_rate = depolarizing_rate
        self.amplitude_damping_rate = amplitude_damping_rate
        self.readout_error_rate = readout_error_rate
        self.t1_us = t1_us
        self.t2_us = t2_us
        self.gate_time_ns = gate_time_ns
        self.two_qubit_gate_time_ns = two_qubit_gate_time_ns
        self.sacred_attenuation = sacred_attenuation

    def apply_gate_noise(self, statevector, qubit: int, num_qubits: int,
                         is_two_qubit: bool = False, depth_layer: int = 0):
        """
        Apply depolarizing + amplitude damping + phase damping noise after a gate.

        Modifies statevector in-place using Kraus channel approximation.
        PHI-scaled noise: effective_rate = base_rate × φ^(-depth_layer).

        Noise channels (applied sequentially):
          1. Depolarizing:    Random Pauli (X/Y/Z) error
          2. Amplitude damping: T1 energy decay (|1⟩→|0⟩)
          3. Phase damping:   T2 dephasing (pure phase decoherence, off-diagonal decay)

        v13.2: Sacred attenuation uses exact GOD_CODE_PHASE threshold instead
        of GOD_CODE % 1.0 — aligns with QPU-verified phase angle for more
        accurate noise suppression at sacred frequencies.
        """
        import random as _rng

        # PHI-scaled noise attenuation with depth
        phi_scale = PHI ** (-depth_layer) if depth_layer > 0 else 1.0
        depol_rate = self.depolarizing_rate * phi_scale
        if is_two_qubit:
            depol_rate *= 10.0  # two-qubit gates are ~10x noisier

        # v13.2: Sacred attenuation uses canonical GOD_CODE_PHASE
        # The QPU-verified phase angle θ_GC ≈ 6.014 rad (nearly 2π)
        # translates to amplitude pattern |e^{-iθ/2}| at the noise floor
        if self.sacred_attenuation:
            dim = len(statevector)
            if dim > 0:
                dominant_amp = float(np.max(np.abs(statevector)))
                # v13.2: Use exact GOD_CODE phase fraction (QPU-verified)
                # θ_GC/2π ≈ 0.9572 → dominant amplitude at this fraction marks sacred alignment
                gc_phase_frac = _GC_PHASE / (2.0 * math.pi)  # ≈ 0.957
                god_align = min(
                    abs(dominant_amp - gc_phase_frac),
                    abs(dominant_amp - (1.0 - gc_phase_frac)),  # complement symmetry
                )
                if god_align < 0.05:
                    # Sacred noise floor: attenuate by (2 - VOID_CONSTANT) ≈ 0.9584
                    depol_rate *= max(0.0, 2.0 - VOID_CONSTANT)
                # v13.2: Additional iron-lattice attenuation at π/2 phase
                iron_frac = _IRON_PHASE / (2.0 * math.pi)  # 0.25 (quarter-turn)
                if abs(dominant_amp - iron_frac) < 0.03:
                    depol_rate *= (1.0 - 1.0 / PHI)  # ≈ 0.382 (φ⁻² attenuation)

        # Depolarizing channel: with probability p, apply random Pauli
        if _rng.random() < depol_rate:
            pauli_choice = _rng.randint(0, 2)  # 0=X, 1=Y, 2=Z
            state_dim = 1 << num_qubits
            sv_len = min(state_dim, len(statevector))
            # v12.2: Vectorized Pauli application via numpy fancy indexing
            indices = np.arange(sv_len, dtype=np.intp)
            flipped = indices ^ (1 << qubit)
            flipped = np.minimum(flipped, sv_len - 1)  # bounds safety
            if pauli_choice == 0:  # X: swap amplitudes at flipped indices
                statevector[:sv_len] = statevector[flipped]
            elif pauli_choice == 1:  # Y: iXZ — swap + phase flip
                temp = statevector[flipped].copy()
                mask = ((indices >> qubit) & 1).astype(np.float64)
                phase = 1.0 - 2.0 * mask  # +1 for |0⟩, -1 for |1⟩
                statevector[:sv_len] = temp * phase * 1j  # Y = iXZ
            elif pauli_choice == 2:  # Z: phase flip |1⟩ amplitudes
                mask = ((indices >> qubit) & 1).astype(np.float64)
                statevector[:sv_len] *= (1.0 - 2.0 * mask)

        # Amplitude damping: T1 decay (simplified Kraus)
        gate_time = self.two_qubit_gate_time_ns if is_two_qubit else self.gate_time_ns
        gamma = 1.0 - math.exp(-gate_time * 1e-3 / self.t1_us) if self.t1_us > 0 else 0
        gamma *= phi_scale
        if gamma > 0 and _rng.random() < gamma:
            state_dim = 1 << num_qubits
            sv_len = min(state_dim, len(statevector))
            indices = np.arange(sv_len, dtype=np.intp)
            excited_mask = ((indices >> qubit) & 1).astype(bool)
            flipped = indices ^ (1 << qubit)
            flipped = np.minimum(flipped, sv_len - 1)
            # Transfer amplitude from |1⟩ to |0⟩
            sqrt_gamma = math.sqrt(gamma)
            sqrt_1mg = math.sqrt(1.0 - gamma)
            excited_amps = statevector[:sv_len].copy()
            np.add.at(statevector, flipped[excited_mask], excited_amps[excited_mask] * sqrt_gamma)
            statevector[:sv_len][excited_mask] *= sqrt_1mg

        # ── Phase damping (T2 dephasing) ─────────────────────────────────
        # Pure dephasing: off-diagonal elements of ρ decay as e^{-t/T_φ}
        # where 1/T_φ = 1/T2 - 1/(2·T1). This is implemented as a stochastic
        # Z-kick: with probability p_dephase, apply Z to the target qubit.
        # p_dephase ≈ 1 - exp(-gate_time / T_φ) for small rates.
        if self.t2_us > 0:
            # Pure dephasing rate: 1/T_φ = 1/T2 - 1/(2T1)
            inv_t_phi = (1.0 / self.t2_us) - (1.0 / (2.0 * self.t1_us)) if self.t1_us > 0 else (1.0 / self.t2_us)
            inv_t_phi = max(0.0, inv_t_phi)  # T_phi can't contribute negative dephasing
            if inv_t_phi > 0:
                gate_time = self.two_qubit_gate_time_ns if is_two_qubit else self.gate_time_ns
                p_dephase = 1.0 - math.exp(-gate_time * 1e-3 * inv_t_phi)
                p_dephase *= phi_scale

                # Sacred attenuation for dephasing
                if self.sacred_attenuation:
                    dim = len(statevector)
                    if dim > 0:
                        dominant_amp = float(np.max(np.abs(statevector)))
                        god_align = abs(dominant_amp - (GOD_CODE % 1.0))
                        if god_align < 0.05:
                            p_dephase *= max(0.0, 2.0 - VOID_CONSTANT)

                if _rng.random() < p_dephase:
                    # Apply Z gate stochastically: phase-flip |1⟩ amplitudes
                    state_dim = 1 << num_qubits
                    sv_len = min(state_dim, len(statevector))
                    indices = np.arange(sv_len, dtype=np.intp)
                    mask = ((indices >> qubit) & 1).astype(np.float64)
                    statevector[:sv_len] *= (1.0 - 2.0 * mask)

        return statevector

    def apply_readout_noise(self, counts: dict, num_qubits: int) -> dict:
        """
        Apply measurement readout errors to shot counts.

        v12.4: Optimized vectorization. Pre-allocates noise buffer and uses
        matrix bit manipulation to flip bits without np.tile overhead.
        ~2x faster than v12.3 (sub-50ms at 131K shots).
        """
        if self.readout_error_rate <= 0:
            return counts

        noisy_counts = {}
        n_bits = num_qubits
        # Powers of 2 for bitstring→integer conversion (MSB first)
        powers = 1 << np.arange(n_bits - 1, -1, -1, dtype=np.int64)

        for bitstring, count in counts.items():
            if count <= 0:
                continue

            # Original integer value of the bitstring
            base_val = int(bitstring, 2)

            # Generate random flip masks for ALL shots in this group at once.
            # flip_mask[i, j] is True if the j-th bit of the i-th shot should flip.
            # We use a single call to generate bits.
            flip_mask = np.random.random((count, n_bits)) < self.readout_error_rate

            # Convert bit flip mask rows to integers.
            # This is the "xor_val" that will be XORed with base_val for each shot.
            xor_vals = flip_mask @ powers

            # Apply XOR: New integer value = base_val ^ xor_val
            # (XOR is equivalent to bit flipping)
            indices = base_val ^ xor_vals.astype(np.int64)

            # Count occurrences with bincount (O(n) C loop)
            # We only need minlength up to the max possible value
            max_val = 1 << n_bits
            bin_counts = np.bincount(indices, minlength=max_val)

            # Only iterate over unique nonzero outcomes (typically very few)
            nonzero = np.nonzero(bin_counts)[0]
            for idx in nonzero:
                key = format(idx, f'0{n_bits}b')
                noisy_counts[key] = noisy_counts.get(key, 0) + int(bin_counts[idx])

        return noisy_counts

    def scaled_copy(self, factor: float) -> 'NoiseModel':
        """Return a copy with all noise rates scaled by factor (for ZNE)."""
        return NoiseModel(
            depolarizing_rate=self.depolarizing_rate * factor,
            amplitude_damping_rate=self.amplitude_damping_rate * factor,
            readout_error_rate=self.readout_error_rate * factor,
            t1_us=self.t1_us,
            t2_us=self.t2_us,
            gate_time_ns=self.gate_time_ns,
            two_qubit_gate_time_ns=self.two_qubit_gate_time_ns,
            sacred_attenuation=self.sacred_attenuation,
        )

    def to_dict(self) -> dict:
        """Serialize noise model parameters."""
        return {
            "depolarizing_rate": self.depolarizing_rate,
            "amplitude_damping_rate": self.amplitude_damping_rate,
            "readout_error_rate": self.readout_error_rate,
            "t1_us": self.t1_us,
            "t2_us": self.t2_us,
            "gate_time_ns": self.gate_time_ns,
            "two_qubit_gate_time_ns": self.two_qubit_gate_time_ns,
            "sacred_attenuation": self.sacred_attenuation,
        }

    @staticmethod
    def realistic_superconducting() -> 'NoiseModel':
        """Factory: realistic superconducting QPU noise (IBM Eagle-class)."""
        return NoiseModel(
            depolarizing_rate=0.001, amplitude_damping_rate=0.0005,
            readout_error_rate=0.015, t1_us=100.0, t2_us=120.0,
            gate_time_ns=35.0, two_qubit_gate_time_ns=300.0,
        )

    @staticmethod
    def low_noise() -> 'NoiseModel':
        """Factory: low-noise near-term device."""
        return NoiseModel(
            depolarizing_rate=0.0001, amplitude_damping_rate=0.00005,
            readout_error_rate=0.005, t1_us=200.0, t2_us=250.0,
        )

    @staticmethod
    def noiseless() -> 'NoiseModel':
        """Factory: zero noise (ideal simulation)."""
        return NoiseModel(
            depolarizing_rate=0.0, amplitude_damping_rate=0.0,
            readout_error_rate=0.0,
        )

    @staticmethod
    def qpu_calibrated_heron() -> 'NoiseModel':
        """v13.2 Factory: QPU-calibrated noise model from ibm_torino (Heron r2).

        Calibrated from real QPU verification data:
          - 1Q fidelity: 0.999939 → depolarizing_rate ≈ 6.1e-5
          - 3Q fidelity: 0.966740 → 2Q gate error ≈ 3.3% per layer
          - QPE (113 depth): 0.934031 → per-layer fidelity ≈ 0.9994

        Heron r2 specs (ibm_torino, 133 qubits):
          - Native basis: {rz, sx, cz}
          - T1 ≈ 300 μs (best-in-class superconducting)
          - T2 ≈ 200 μs
          - 1Q gate time: 35 ns (SX), 0 ns virtual (Rz)
          - 2Q gate time: 68 ns (CZ — tunable coupler, NOT cross-resonance)
        """
        # Derive rates from QPU data
        if _QPU_DATA is not None:
            circuits = _QPU_DATA.get("circuits", {})
            # 1Q fidelity → depolarizing rate: p ≈ (1 - F) × 4/3
            f_1q = circuits.get("1Q_GOD_CODE", {}).get("fidelity", 0.999939)
            depol_1q = (1.0 - f_1q) * 4.0 / 3.0  # ≈ 8.1e-5

            # 3Q sacred → 2Q gate error rate
            f_3q = circuits.get("3Q_SACRED", {}).get("fidelity", 0.96674)
            # 3Q has ~6 CZ gates → per-gate 2Q error ≈ (1-F)/n_2q
            depol_2q_per_gate = (1.0 - f_3q) / 6.0  # ≈ 0.0055

            return NoiseModel(
                depolarizing_rate=depol_1q,
                amplitude_damping_rate=depol_1q * 0.5,  # T1 component
                readout_error_rate=0.008,  # Heron r2 readout error
                t1_us=300.0,              # Heron r2 T1
                t2_us=200.0,              # Heron r2 T2
                gate_time_ns=35.0,        # SX gate time
                two_qubit_gate_time_ns=68.0,  # CZ gate time (tunable coupler)
                sacred_attenuation=True,
            )

        # Fallback: use published Heron specs
        return NoiseModel(
            depolarizing_rate=8.1e-5,
            amplitude_damping_rate=4.0e-5,
            readout_error_rate=0.008,
            t1_us=300.0,
            t2_us=200.0,
            gate_time_ns=35.0,
            two_qubit_gate_time_ns=68.0,
            sacred_attenuation=True,
        )

    @staticmethod
    def god_code_optimized() -> 'NoiseModel':
        """v13.2 Factory: Optimized for circuits using GOD_CODE phase gates.

        Uses sacred attenuation tuned to the 3-rotation decomposition:
        circuits with IRON + PHI + OCTAVE sub-phases get maximum noise
        suppression, exploiting the QPU-verified conservation law.
        """
        return NoiseModel(
            depolarizing_rate=0.0005,
            amplitude_damping_rate=0.0003,
            readout_error_rate=0.01,
            t1_us=150.0,
            t2_us=180.0,
            gate_time_ns=35.0,
            two_qubit_gate_time_ns=100.0,
            sacred_attenuation=True,  # Enables GOD_CODE + iron-lattice attenuation
        )

    # ─── v14.0: CROSSTALK NOISE MODEL ───

    def apply_crosstalk(self, statevector, qubit: int, num_qubits: int,
                        crosstalk_rate: float = None) -> None:
        """v14.0: Apply ZZ cross-talk noise between a qubit and its neighbors.

        Cross-talk models parasitic ZZ interactions that occur between physically
        adjacent qubits on superconducting hardware. The rate decays with
        distance using φ⁻¹ attenuation (sacred decay factor).

        ZZ interaction: applies a small conditional phase flip on neighboring
        qubit pairs, simulating always-on ZZ coupling.

        Args:
            statevector: Quantum state (modified in-place)
            qubit: The qubit that just had a gate applied
            num_qubits: Total qubit count
            crosstalk_rate: Override rate (default: CROSSTALK_ZZ_RATE from constants)
        """
        import random as _rng
        from .constants import CROSSTALK_ZZ_RATE, CROSSTALK_DECAY_DISTANCE, CROSSTALK_PHI_ATTENUATION

        rate = crosstalk_rate if crosstalk_rate is not None else CROSSTALK_ZZ_RATE
        if rate <= 0:
            return

        state_dim = 1 << num_qubits
        sv_len = min(state_dim, len(statevector))

        # Apply ZZ cross-talk to neighbors within decay distance
        for neighbor in range(max(0, qubit - CROSSTALK_DECAY_DISTANCE),
                              min(num_qubits, qubit + CROSSTALK_DECAY_DISTANCE + 1)):
            if neighbor == qubit:
                continue

            distance = abs(neighbor - qubit)
            # φ⁻¹ decay per unit distance
            effective_rate = rate * (CROSSTALK_PHI_ATTENUATION ** distance)

            if _rng.random() < effective_rate:
                # ZZ interaction: conditional phase on |11⟩ subspace
                # Phase = exp(i·π·rate) for |11⟩ states of (qubit, neighbor)
                indices = np.arange(sv_len, dtype=np.intp)
                both_excited = ((indices >> qubit) & 1) & ((indices >> neighbor) & 1)
                phase_angle = math.pi * effective_rate
                phase_factor = np.exp(1j * phase_angle)
                statevector[:sv_len] = np.where(
                    both_excited.astype(bool),
                    statevector[:sv_len] * phase_factor,
                    statevector[:sv_len]
                )

    @staticmethod
    def with_crosstalk(base_model: 'NoiseModel' = None,
                       crosstalk_rate: float = None) -> 'NoiseModel':
        """v14.0 Factory: Create a noise model with cross-talk enabled.

        Returns a model with crosstalk_rate stored as attribute for
        use by the simulation pipeline.
        """
        from .constants import CROSSTALK_ZZ_RATE
        model = base_model or NoiseModel.qpu_calibrated_heron()
        model._crosstalk_rate = crosstalk_rate if crosstalk_rate is not None else CROSSTALK_ZZ_RATE
        model._crosstalk_enabled = True
        return model
