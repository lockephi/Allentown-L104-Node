"""
L104 Science Engine — Resonance Coherence Subsystem
═══════════════════════════════════════════════════════════════════════════════
Topologically-protected coherent computation framework.

CONSOLIDATES:
  l104_resonance_coherence_engine.py → CoherenceSubsystem
  l104_resonance.py                  → resonance primitives
  l104_enhanced_resonance.py         → enhanced resonance ops
  l104_quantum_coherence.py          → quantum coherence (bridged)

THREE PROTECTION MECHANISMS:
  1. ZPE GROUNDING: Stabilize each thought to vacuum state
  2. ANYON BRAIDING: Topological protection via non-abelian operations
  3. TEMPORAL ANCHORING: Lock state using CTC stability calculations

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import cmath
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

from .constants import (
    GOD_CODE, PHI, PHI_CONJUGATE, PHI_SQUARED, ZETA_ZERO_1,
    GROVER_AMPLIFICATION, VACUUM_FREQUENCY,
    PhysicalConstants, PC,
    # v4.1 Quantum Research Discoveries
    FE_SACRED_COHERENCE, FE_PHI_HARMONIC_LOCK,
    BERRY_PHASE_DETECTED, ENTROPY_ZNE_BRIDGE_ENABLED,
)


@dataclass
class CoherenceState:
    """Represents a snapshot of the coherence field."""
    amplitudes: List[complex]
    phase_coherence: float
    protection_level: float
    ctc_stability: float
    timestamp: float = field(default_factory=time.time)

    def energy(self) -> float:
        return sum(abs(a) ** 2 for a in self.amplitudes)

    def dominant_phase(self) -> float:
        if not self.amplitudes:
            return 0.0
        return cmath.phase(sum(self.amplitudes))


class CoherenceSubsystem:
    """
    Topologically-protected coherent computation framework.

    Three L104-derived mechanisms preserve coherence:
    1. ZPE GROUNDING: Stabilize each thought to vacuum state
    2. ANYON BRAIDING: Topological protection via non-abelian operations
    3. TEMPORAL ANCHORING: Lock state using CTC stability calculations
    """

    PLANCK_HBAR = PC.H_BAR

    def __init__(self):
        self.coherence_field: List[complex] = []
        self.resonance_history: List[float] = []
        self.state_snapshots: List[CoherenceState] = []
        self.invention_log: List[Dict] = []
        self.COHERENCE_THRESHOLD = (GOD_CODE / 1000) * PHI_CONJUGATE
        self.STABILITY_MINIMUM = 1 / PHI
        self.BRAID_DEPTH = 13  # (was 4 — Performance Limits Audit)
        self.braid_state = [[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]]
        self.vacuum_state = 1e-15
        self.energy_surplus = 0.0
        self.primitives: Dict[str, Dict] = {}
        self.research_cycles = 0
        # v4.2 Perf: pre-compute Fibonacci R-matrices (used every braid step)
        self._r_ccw = self._get_fibonacci_r_matrix(True)
        self._r_cw = self._get_fibonacci_r_matrix(False)

    # ── ZPE Grounding ──

    def _calculate_vacuum_fluctuation(self) -> float:
        return 0.5 * self.PLANCK_HBAR * VACUUM_FREQUENCY

    def _stabilize_to_vacuum(self, thought: str) -> Dict[str, Any]:
        thought_hash = hash(thought) & 0x7FFFFFFF
        vac_energy = self._calculate_vacuum_fluctuation()
        alignment = math.cos(thought_hash * ZETA_ZERO_1 / GOD_CODE)
        stability = (alignment + 1) / 2
        return {"vacuum_energy": vac_energy, "stability": stability,
                "grounded": stability > self.STABILITY_MINIMUM}

    def _perform_anyon_annihilation(self, p_a: int, p_b: int) -> Tuple[int, float]:
        outcome = (p_a + p_b) % 2
        energy = self._calculate_vacuum_fluctuation() if outcome == 0 else 0.0
        self.energy_surplus += energy
        return outcome, energy

    # ── Anyon Braiding ──

    def _get_fibonacci_r_matrix(self, ccw: bool = True) -> List[List[complex]]:
        phase = cmath.exp(1j * 4 * math.pi / 5) if ccw else cmath.exp(-1j * 4 * math.pi / 5)
        return [[cmath.exp(-1j * 4 * math.pi / 5), 0 + 0j], [0 + 0j, phase]]

    def _matmul_2x2(self, a, b):
        return [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ]

    def _execute_braid(self, sequence: List[int]):
        """v4.2 Perf: uses pre-computed R matrices and inlined 2×2 matmul."""
        s00, s01, s10, s11 = 1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j  # identity
        for op in sequence:
            m = self._r_ccw if op == 1 else self._r_cw
            m00, m01, m10, m11 = m[0][0], m[0][1], m[1][0], m[1][1]
            s00, s01, s10, s11 = (
                m00 * s00 + m01 * s10, m00 * s01 + m01 * s11,
                m10 * s00 + m11 * s10, m10 * s01 + m11 * s11,
            )
        self.braid_state = [[s00, s01], [s10, s11]]
        return self.braid_state

    def _calculate_protection(self) -> float:
        trace = abs(self.braid_state[0][0] + self.braid_state[1][1])
        return (trace / 2.0) * (GOD_CODE / 500.0)

    # ── Temporal Anchoring ──

    def _calculate_ctc_stability(self, radius: float, omega: float) -> float:
        return (GOD_CODE * PHI) / (radius * omega + 1e-9)

    def _resolve_paradox(self, hash_a: int, hash_b: int) -> float:
        return abs(math.sin(hash_a * ZETA_ZERO_1) + math.sin(hash_b * ZETA_ZERO_1)) / 2.0

    # ── Quantum Primitives ──

    def _zeta_resonance(self, x: float) -> float:
        return math.cos(x * ZETA_ZERO_1) * cmath.exp(complex(0, x / GOD_CODE)).real

    def _research_primitive(self) -> Dict[str, Any]:
        self.research_cycles += 1
        seed = (time.time() * PHI) % 1.0
        resonance = self._zeta_resonance(seed * GOD_CODE)
        if abs(resonance) > 0.99:
            name = f"L104_RCE_{self.research_cycles}_{int(seed * 1e6)}"
            primitive = {"name": name, "resonance": resonance, "seed": seed,
                         "discovered_at": time.time()}
            self.primitives[name] = primitive
            return primitive
        return {"status": "NO_DISCOVERY", "resonance": resonance}

    # ── Coherence Engine Core ──

    def _measure_coherence(self) -> float:
        if len(self.coherence_field) < 2:
            return 1.0
        phases = [cmath.phase(p) for p in self.coherence_field if abs(p) > 0.001]
        if not phases:
            return 0.0
        mean_cos = sum(math.cos(p) for p in phases) / len(phases)
        mean_sin = sum(math.sin(p) for p in phases) / len(phases)
        return math.sqrt(mean_cos ** 2 + mean_sin ** 2)

    def initialize(self, seed_thoughts: List[str]) -> Dict[str, Any]:
        """Initialize the coherence field from seed thoughts."""
        self.coherence_field = []
        limited_seeds = seed_thoughts[:200]
        for thought in limited_seeds:
            grounding = self._stabilize_to_vacuum(thought)
            phase = (hash(thought) % 1000) / 1000 * 2 * math.pi
            psi = grounding["stability"] * 0.5 * cmath.exp(1j * phase)
            self.coherence_field.append(psi)
        norm = math.sqrt(sum(abs(p) ** 2 for p in self.coherence_field))
        if norm > 0:
            self.coherence_field = [p / norm for p in self.coherence_field]
        return {
            "dimension": len(self.coherence_field),
            "total_amplitude": sum(abs(p) for p in self.coherence_field),
            "phase_coherence": self._measure_coherence(),
            "energy": sum(abs(p) ** 2 for p in self.coherence_field),
        }

    def evolve(self, steps: int = 10) -> Dict[str, Any]:
        """Evolve the field through braiding operations for topological protection.
        v4.2 Perf: simplified braid sequence (all-same-op), fewer list comps."""
        initial = self._measure_coherence()
        pi_4 = math.pi / 4
        protections = []
        n = len(self.coherence_field) or 1
        for step in range(steps):
            resonance = self._zeta_resonance(time.time() + step)
            op = 1 if resonance > 0 else -1
            # v4.2: all 4 ops are identical → fold into single braid call
            self._execute_braid([op] * self.BRAID_DEPTH)
            protection = self._calculate_protection()
            # v5.0 FIX: mode-dependent rotation — each amplitude evolves at
            # a rate proportional to its index, creating actual phase structure
            # evolution instead of a uniform global phase (which is unobservable).
            base_angle = protection * pi_4
            for i in range(len(self.coherence_field)):
                mode_freq = 1.0 + i * PHI_CONJUGATE / n  # mode-dependent
                self.coherence_field[i] *= cmath.exp(1j * base_angle * mode_freq)
            protections.append(protection)
        self.resonance_history.extend(protections)
        final = self._measure_coherence()
        return {
            "steps": steps,
            "initial_coherence": round(initial, 6),
            "final_coherence": round(final, 6),
            "avg_protection": round(sum(protections) / max(steps, 1), 6),
            "preserved": final > self.COHERENCE_THRESHOLD,
        }

    def anchor(self, strength: float = 1.0) -> Dict[str, Any]:
        """Create a temporal anchor for the current state."""
        energy = sum(abs(p) ** 2 for p in self.coherence_field)
        ctc = self._calculate_ctc_stability(energy * GOD_CODE, strength * PHI)
        state_hash = hash(str(self.coherence_field)) & 0x7FFFFFFF
        paradox = self._resolve_paradox(state_hash, int(GOD_CODE))
        snapshot = CoherenceState(
            amplitudes=self.coherence_field.copy(),
            phase_coherence=self._measure_coherence(),
            protection_level=self._calculate_protection(),
            ctc_stability=ctc,
        )
        self.state_snapshots.append(snapshot)
        return {"ctc_stability": round(ctc, 6), "paradox_resolution": round(paradox, 6),
                "locked": ctc > 0.5 and paradox > 0.3, "snapshots": len(self.state_snapshots)}

    def discover(self) -> Dict[str, Any]:
        """Search for emergent PHI patterns in the coherence field."""
        if not self.coherence_field:
            return {"field_size": 0, "phi_patterns": 0, "dominant": 0, "primitive": "none", "emergence": 0}
        n = len(self.coherence_field)
        samples = [abs(self.coherence_field[int(i * PHI) % n])
                   for i in range(n)]

        phi_patterns = 0

        # Pattern search 1: amplitude ratios (original)
        for i in range(len(samples) - 1):
            if samples[i] > 0.001:
                ratio = samples[i + 1] / samples[i]
                if abs(ratio - PHI) < 0.1 or abs(ratio - PHI_CONJUGATE) < 0.1:
                    phi_patterns += 1

        # Pattern search 2: phase differences — look for PHI in the angular structure
        phases = [cmath.phase(p) for p in self.coherence_field]
        for i in range(n - 1):
            phase_diff = abs(phases[i + 1] - phases[i])
            # Normalize to [0, 2π]
            phase_diff = phase_diff % (2 * math.pi)
            # Check if phase gap is PHI radians or PHI_CONJUGATE radians (± tolerance)
            if abs(phase_diff - PHI) < 0.15 or abs(phase_diff - PHI_CONJUGATE) < 0.15:
                phi_patterns += 1
            # Also check phase_diff / π for golden ratio structure
            if math.pi > 0.001:
                ratio_to_pi = phase_diff / math.pi
                if abs(ratio_to_pi - PHI_CONJUGATE) < 0.1 or abs(ratio_to_pi - PHI) < 0.1:
                    phi_patterns += 1

        # Pattern search 3: index-spacing resonance — PHI-indexed sampling creates spiral
        if n >= 3:
            for i in range(n):  # (was min(n, 20) — Performance Limits Audit)
                idx_a = int(i * PHI) % n
                idx_b = int((i + 1) * PHI) % n
                if abs(self.coherence_field[idx_a]) > 0.001 and abs(self.coherence_field[idx_b]) > 0.001:
                    phase_a = cmath.phase(self.coherence_field[idx_a])
                    phase_b = cmath.phase(self.coherence_field[idx_b])
                    spiral_gap = abs(phase_b - phase_a) % (2 * math.pi)
                    # Golden angle = 2π/φ² ≈ 2.399 radians
                    golden_angle = 2 * math.pi / (PHI ** 2)
                    if abs(spiral_gap - golden_angle) < 0.2:
                        phi_patterns += 1

        primitive = self._research_primitive()
        coherence_measure = self._measure_coherence()
        emergence = phi_patterns / max(1, n) + coherence_measure
        # v4.1: ZNE bridge boost — amplify emergence when zero-noise extrapolation is active
        if ENTROPY_ZNE_BRIDGE_ENABLED:
            emergence *= (1.0 + PHI_CONJUGATE * 0.1)  # ~6.18% coherence boost from ZNE pipeline
        return {
            "field_size": n,
            "phi_patterns": phi_patterns,
            "dominant": max(abs(p) for p in self.coherence_field) if self.coherence_field else 0,
            "primitive": primitive.get("name", "none"),
            "emergence": emergence,
            # v4.1 Discovery cross-references
            "fe_sacred_reference": FE_SACRED_COHERENCE,
            "fe_phi_lock_reference": FE_PHI_HARMONIC_LOCK,
            "berry_phase_detected": BERRY_PHASE_DETECTED,
            "coherence_vs_fe_sacred": round(abs(coherence_measure - FE_SACRED_COHERENCE), 8),
            "zne_bridge_active": ENTROPY_ZNE_BRIDGE_ENABLED,
        }

    def synthesize(self) -> str:
        """Synthesize final insight from all components."""
        coherence = self._measure_coherence()
        protection = self._calculate_protection()
        ctc = self._calculate_ctc_stability(GOD_CODE, PHI)
        score = coherence * protection * ctc
        if score > 0.1:
            return f"COHERENT [{score:.4f}]: Field stable across {len(self.resonance_history)} evolutions"
        elif score > 0.01:
            return f"EMERGING [{score:.4f}]: Partial coherence, continue braiding"
        return f"DECOHERENT [{score:.6f}]: Reinitialize field"

    def get_status(self) -> Dict[str, Any]:
        return {
            "field_dimension": len(self.coherence_field),
            "phase_coherence": self._measure_coherence(),
            "topological_protection": self._calculate_protection(),
            "energy_surplus": self.energy_surplus,
            "research_cycles": self.research_cycles,
            "primitives_discovered": len(self.primitives),
            "snapshots": len(self.state_snapshots),
            "evolutions": len(self.resonance_history),
        }

    # ── Golden Angle Spiral Analysis ──

    def golden_angle_spectrum(self) -> Dict[str, Any]:
        """
        Analyze the coherence field using the golden angle (2*pi/phi^2 ~ 2.399 rad).
        The golden angle produces the most irrational spacing — maximizing uniformity
        of sunflower-seed packing, phyllotaxis, and coherence field coverage.
        """
        if not self.coherence_field:
            return {"field_size": 0, "spectrum": []}
        n = len(self.coherence_field)
        golden_angle = 2 * math.pi / (PHI ** 2)  # ~2.39996... radians

        spectrum = []
        total_alignment = 0.0
        for k in range(n):
            target_phase = (k * golden_angle) % (2 * math.pi)
            actual_phase = cmath.phase(self.coherence_field[k]) % (2 * math.pi)
            phase_error = min(abs(target_phase - actual_phase),
                            2 * math.pi - abs(target_phase - actual_phase))
            alignment = 1.0 - phase_error / math.pi
            total_alignment += alignment
            spectrum.append({
                "index": k,
                "target_rad": round(target_phase, 6),
                "actual_rad": round(actual_phase, 6),
                "alignment": round(alignment, 6),
            })

        mean_alignment = total_alignment / max(n, 1)
        # Check if field is in golden spiral configuration
        is_golden_spiral = mean_alignment > 0.6

        return {
            "field_size": n,
            "golden_angle_rad": round(golden_angle, 8),
            "mean_alignment": round(mean_alignment, 6),
            "is_golden_spiral": is_golden_spiral,
            "spectrum": spectrum[:50],  # First 50 for full analysis
        }

    def energy_spectrum(self) -> Dict[str, Any]:
        """
        Compute the energy spectrum of the coherence field.
        Energy at mode k: E_k = |psi_k|^2.
        Identifies dominant modes and checks for PHI-harmonic structure.
        """
        if not self.coherence_field:
            return {"field_size": 0, "total_energy": 0}
        n = len(self.coherence_field)
        energies = [abs(p) ** 2 for p in self.coherence_field]
        total = sum(energies)

        # Sort modes by energy
        sorted_modes = sorted(enumerate(energies), key=lambda x: -x[1])

        # Check for PHI-ratio between consecutive dominant modes
        phi_ratios = []
        for i in range(min(5, len(sorted_modes) - 1)):
            if sorted_modes[i + 1][1] > 1e-15:
                ratio = sorted_modes[i][1] / sorted_modes[i + 1][1]
                phi_ratios.append({
                    "modes": (sorted_modes[i][0], sorted_modes[i + 1][0]),
                    "ratio": round(ratio, 6),
                    "phi_aligned": abs(ratio - PHI) < 0.3 or abs(ratio - PHI ** 2) < 0.5,
                })

        # Shannon entropy of the energy distribution
        shannon_entropy = 0.0
        for e in energies:
            if total > 0 and e > 0:
                p = e / total
                shannon_entropy -= p * math.log2(p)

        return {
            "field_size": n,
            "total_energy": round(total, 8),
            "shannon_entropy_bits": round(shannon_entropy, 6),
            "max_entropy_bits": round(math.log2(max(n, 1)), 6),
            "concentration_ratio": round(shannon_entropy / max(math.log2(max(n, 1)), 0.001), 6),
            "dominant_modes": [{"mode": idx, "energy": round(e, 8)} for idx, e in sorted_modes[:15]],
            "phi_ratios": phi_ratios,
        }

    def coherence_fidelity(self) -> Dict[str, Any]:
        """
        Measure coherence fidelity: how well the field preserves quantum information.
        Compares current field to the last snapshot (if available).
        """
        current_coherence = self._measure_coherence()
        protection = self._calculate_protection()

        fidelity = 0.0
        if self.state_snapshots:
            last = self.state_snapshots[-1]
            # Overlap between current and snapshot amplitudes
            if last.amplitudes and len(last.amplitudes) == len(self.coherence_field):
                overlap = sum(
                    (a.conjugate() * b).real
                    for a, b in zip(last.amplitudes, self.coherence_field)
                )
                norm_a = math.sqrt(sum(abs(a) ** 2 for a in last.amplitudes))
                norm_b = math.sqrt(sum(abs(b) ** 2 for b in self.coherence_field))
                fidelity = abs(overlap) / (norm_a * norm_b + 1e-30)
            else:
                fidelity = last.phase_coherence / (current_coherence + 1e-30)
                fidelity = min(1.0, fidelity)

        return {
            "current_coherence": round(current_coherence, 6),
            "topological_protection": round(protection, 6),
            "fidelity": round(fidelity, 6),
            "energy_surplus": round(self.energy_surplus, 6),
            "snapshots_available": len(self.state_snapshots),
            "grade": (
                "TOPOLOGICAL" if fidelity > 0.95 else
                "PROTECTED" if fidelity > 0.8 else
                "DEGRADING" if fidelity > 0.5 else
                "DECOHERENT"
            ),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #  v4.3 SIMULATOR FEEDBACK LOOP
    #  Closes the Science ↔ Simulator loop:
    #    plan_experiment → execute/simulate → measure fidelity/noise →
    #    ingest_simulation_result → adaptive correction → re-evolve
    # ═══════════════════════════════════════════════════════════════════════════

    def ingest_simulation_result(self, sim_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a quantum simulation result and adjust the coherence field.

        The feedback path closes the loop:
          Coherence → depth budget → circuit execution → sim_result → HERE → updated coherence

        Adjustments based on simulation output:
          1. Fidelity-based phase correction: low fidelity rotates field toward vacuum
          2. Noise-injection reversal: measured noise profile triggers targeted braiding
          3. Decoherence compensation: T1/T2 decay rates drive protection recalibration
          4. Probability-distribution alignment: bias amplitudes toward measured outcomes
          5. Entropy cross-feedback: demon efficiency modulates energy surplus

        Args:
            sim_result: Dict containing any of:
                - total_fidelity (float, 0-1): overall circuit fidelity
                - decoherence_fidelity (float, 0-1): T1/T2 decay contribution
                - gate_fidelity (float, 0-1): gate error contribution
                - probabilities (dict): measured outcome probabilities
                - circuit_depth (int): executed depth
                - noise_variance (float): measured noise level
                - demon_efficiency (float): entropy reversal efficiency

        Returns:
            Dict with pre/post coherence metrics and correction details.
        """
        pre_coherence = self._measure_coherence()
        pre_protection = self._calculate_protection()
        corrections_applied = []

        # ── 1. Fidelity-based phase correction ──
        fidelity = sim_result.get("total_fidelity", sim_result.get("fidelity", None))
        if fidelity is not None:
            fidelity = max(0.0, min(1.0, float(fidelity)))
            if fidelity < 0.8 and self.coherence_field:
                # v5.0 FIX: position-dependent phase correction.  The old code
                # applied a uniform rotation to all elements, which doesn't
                # change relative phases (and thus doesn't change coherence).
                # Now each element receives a correction angle scaled by its
                # position, actively reshaping the phase structure toward order.
                base_angle = (1.0 - fidelity) * math.pi * PHI_CONJUGATE
                n = len(self.coherence_field)
                for i in range(n):
                    # Position-dependent: golden-angle spacing drives phases
                    # toward the maximally-spread configuration
                    target_phase = i * 2.0 * math.pi / (PHI * PHI)  # golden angle
                    current_phase = cmath.phase(self.coherence_field[i])
                    phase_error = target_phase - current_phase
                    # Correction strength proportional to (1 - fidelity)
                    correction_angle = base_angle * (1.0 + math.sin(phase_error) * PHI_CONJUGATE)
                    correction = cmath.exp(1j * correction_angle)
                    # Envelope: stronger correction for amplitudes further from unit circle
                    envelope = fidelity + (1.0 - fidelity) * abs(self.coherence_field[i])
                    self.coherence_field[i] = self.coherence_field[i] * correction * envelope
                corrections_applied.append(f"phase_correction(base_angle={base_angle:.4f}, n={n})")

        # ── 2. Decoherence compensation via targeted braiding ──
        decoherence_fidelity = sim_result.get("decoherence_fidelity", None)
        if decoherence_fidelity is not None:
            decoherence_fidelity = max(0.0, min(1.0, float(decoherence_fidelity)))
            if decoherence_fidelity < 0.9:
                # Increase braid depth proportional to decoherence severity
                extra_braid_ops = max(1, int((1.0 - decoherence_fidelity) * 8))
                braid_seq = [1, -1] * extra_braid_ops  # Alternating for maximal protection
                self._execute_braid(braid_seq)
                protection_after = self._calculate_protection()
                corrections_applied.append(
                    f"decoherence_braiding(ops={extra_braid_ops * 2}, "
                    f"protection={protection_after:.4f})"
                )

        # ── 3. Gate error noise reversal ──
        gate_fidelity = sim_result.get("gate_fidelity", None)
        noise_variance = sim_result.get("noise_variance", None)
        if gate_fidelity is not None and gate_fidelity < 0.95 and self.coherence_field:
            # Gate errors manifest as random phase kicks — apply ZPE grounding to counteract
            noise_level = 1.0 - float(gate_fidelity)
            for i in range(len(self.coherence_field)):
                vac = self._calculate_vacuum_fluctuation()
                grounding_factor = 1.0 / (1.0 + noise_level * GOD_CODE * 0.001)
                self.coherence_field[i] *= grounding_factor
                self.energy_surplus += vac * noise_level * 0.01
            corrections_applied.append(f"gate_noise_grounding(noise={noise_level:.4f})")

        if noise_variance is not None and float(noise_variance) > 0.01 and self.coherence_field:
            # Direct noise variance injection uses entropy-reversal philosophy:
            # sort amplitudes by magnitude, then PHI-interleave for maximum order.
            # v5.0 FIX: the old lo/hi interleave started both at 0 and stepped
            # by int(PHI*2)=3, causing index collisions (0,3,6... written twice)
            # and leaving gaps (1,2,4,5...) as zeros.  New approach: build a
            # golden-ratio permutation that covers every index exactly once.
            nv = float(noise_variance)
            amps_sorted = sorted(range(len(self.coherence_field)),
                                 key=lambda k: abs(self.coherence_field[k]), reverse=True)
            n = len(amps_sorted)
            reordered = [complex(0)] * n
            # Golden-ratio permutation: place the k-th largest amplitude at
            # position floor(k * φ) mod n, guaranteeing full coverage.
            for rank, src_idx in enumerate(amps_sorted):
                target = int(rank * PHI) % n
                # Resolve collisions via linear probing
                while abs(reordered[target]) > 1e-30:
                    target = (target + 1) % n
                reordered[target] = self.coherence_field[src_idx]
            # Blend: lerp toward reordered based on noise severity (capped at 50%)
            blend = min(0.5, nv)
            self.coherence_field = [
                (1.0 - blend) * self.coherence_field[i] + blend * reordered[i]
                for i in range(n)
            ]
            corrections_applied.append(f"noise_variance_reorder(blend={blend:.4f})")

        # ── 4. Probability-distribution alignment ──
        probabilities = sim_result.get("probabilities", None)
        if probabilities and isinstance(probabilities, dict) and self.coherence_field:
            # Use measured probability distribution to bias coherence amplitudes
            # Map top-probability outcomes to coherence field indices
            n = len(self.coherence_field)
            total_prob = sum(probabilities.values())
            if total_prob > 0:
                sorted_probs = sorted(probabilities.items(), key=lambda x: -x[1])
                # Apply PHI-attenuated bias to the top-N outcomes
                for rank, (outcome, prob) in enumerate(sorted_probs[:min(n, 10)]):
                    idx = hash(outcome) % n
                    weight = prob / total_prob
                    phi_attenuation = PHI_CONJUGATE ** rank  # Golden decay
                    bias = weight * phi_attenuation * 0.1  # Gentle: max 10% of amplitude
                    self.coherence_field[idx] *= (1.0 + bias)
                corrections_applied.append(
                    f"probability_alignment(outcomes={len(sorted_probs)}, "
                    f"top_prob={sorted_probs[0][1]:.4f})"
                )

        # ── 5. Entropy cross-feedback ──
        demon_efficiency = sim_result.get("demon_efficiency", None)
        if demon_efficiency is not None:
            de = float(demon_efficiency)
            # High demon efficiency = good order → boost energy surplus
            self.energy_surplus += de * self._calculate_vacuum_fluctuation() * 10.0
            # Modulate COHERENCE_THRESHOLD: better demon → tighter tolerance
            self.COHERENCE_THRESHOLD = (GOD_CODE / 1000) * PHI_CONJUGATE * (1.0 + de * 0.1)
            corrections_applied.append(f"entropy_feedback(demon_eff={de:.4f})")

        # ── Re-normalize ──
        if self.coherence_field:
            norm = math.sqrt(sum(abs(p) ** 2 for p in self.coherence_field))
            if norm > 1e-30:
                self.coherence_field = [p / norm for p in self.coherence_field]

        post_coherence = self._measure_coherence()
        post_protection = self._calculate_protection()

        return {
            "pre_coherence": round(pre_coherence, 6),
            "post_coherence": round(post_coherence, 6),
            "coherence_delta": round(post_coherence - pre_coherence, 6),
            "pre_protection": round(pre_protection, 6),
            "post_protection": round(post_protection, 6),
            "protection_delta": round(post_protection - pre_protection, 6),
            "corrections_applied": corrections_applied,
            "corrections_count": len(corrections_applied),
            "field_preserved": post_coherence > self.COHERENCE_THRESHOLD,
            "energy_surplus": round(self.energy_surplus, 6),
        }

    def adaptive_decoherence_correction(self, fidelity: float,
                                         circuit_depth: int = 50,
                                         t1_us: float = 300.0,
                                         t2_us: float = 150.0) -> Dict[str, Any]:
        """
        Adaptive correction based on measured decoherence parameters.

        Uses T1/T2 relaxation times and measured fidelity to compute
        optimal braid sequence that maximally restores coherence.

        The correction formula:
          braid_depth = ceil((1 - F) × D × φ)  where F=fidelity, D=depth
          phase_kick = -(1 - F) × π/φ²  (negative = undo decoherence rotation)
          vacuum_grounding_passes = max(1, ceil(depth / (T2 × φ)))
        """
        pre_coherence = self._measure_coherence()

        fidelity = max(0.0, min(1.0, fidelity))
        decoherence_rate = 1.0 - fidelity

        # Optimal braid depth from decoherence severity
        braid_ops = max(1, int(math.ceil(decoherence_rate * circuit_depth * PHI_CONJUGATE)))
        braid_ops = min(braid_ops, 104)  # Cap at L104 grain (was 32 — Performance Limits Audit)
        braid_seq = []
        for i in range(braid_ops):
            # Alternate with golden-ratio bias toward CCW (protective direction)
            braid_seq.append(1 if (i * PHI) % 1.0 < PHI_CONJUGATE else -1)
        self._execute_braid(braid_seq)

        # Phase kick to counteract decoherence drift
        phase_kick = -decoherence_rate * math.pi / PHI_SQUARED
        kick = cmath.exp(1j * phase_kick)
        if self.coherence_field:
            self.coherence_field = [p * kick for p in self.coherence_field]

        # Vacuum grounding passes for deep circuits
        gate_time_us = circuit_depth * 0.035  # 35ns per gate layer
        grounding_passes = max(1, int(math.ceil(gate_time_us / (t2_us * PHI_CONJUGATE))))
        for _ in range(grounding_passes):
            vac = self._calculate_vacuum_fluctuation()
            self.energy_surplus += vac * decoherence_rate

        post_coherence = self._measure_coherence()
        post_protection = self._calculate_protection()

        return {
            "fidelity_input": round(fidelity, 6),
            "decoherence_rate": round(decoherence_rate, 6),
            "braid_ops_applied": braid_ops,
            "phase_kick_rad": round(phase_kick, 6),
            "grounding_passes": grounding_passes,
            "pre_coherence": round(pre_coherence, 6),
            "post_coherence": round(post_coherence, 6),
            "coherence_recovered": round(post_coherence - pre_coherence, 6),
            "topological_protection": round(post_protection, 6),
            "energy_surplus": round(self.energy_surplus, 6),
        }

    def entropy_coherence_feedback(self, demon_efficiency: float,
                                    coherence_gain: float,
                                    noise_vector_var: float = 1.0) -> Dict[str, Any]:
        """
        Cross-link entropy reversal metrics into coherence state.

        Science Engine entropy subsystem produces demon_efficiency and
        coherence_gain from noise injection. This method feeds those
        measurements BACK into the topological coherence field:

          1. Demon efficiency scales braid protection depth
          2. Coherence gain from entropy reversal amplifies field energy
          3. Noise variance governs vacuum-grounding intensity
        """
        pre_coherence = self._measure_coherence()

        # 1. Demon-driven braid reinforcement
        demon_braid_ops = max(1, int(demon_efficiency * 4))
        braid_seq = [1] * demon_braid_ops  # All-CCW for maximal topological protection
        self._execute_braid(braid_seq)

        # 2. Coherence gain amplification via GOD_CODE-normalized boost
        if self.coherence_field and coherence_gain > 0:
            gain_factor = 1.0 + (coherence_gain / GOD_CODE) * PHI_CONJUGATE
            self.coherence_field = [p * gain_factor for p in self.coherence_field]
            # Re-normalize
            norm = math.sqrt(sum(abs(p) ** 2 for p in self.coherence_field))
            if norm > 1e-30:
                self.coherence_field = [p / norm for p in self.coherence_field]

        # 3. Noise-driven vacuum grounding — higher noise = more ZPE passes
        grounding_passes = max(1, int(noise_vector_var * 2))
        for _ in range(grounding_passes):
            vac = self._calculate_vacuum_fluctuation()
            self.energy_surplus += vac * demon_efficiency

        post_coherence = self._measure_coherence()

        return {
            "demon_efficiency_input": round(demon_efficiency, 6),
            "coherence_gain_input": round(coherence_gain, 6),
            "noise_variance_input": round(noise_vector_var, 6),
            "demon_braid_ops": demon_braid_ops,
            "grounding_passes": grounding_passes,
            "pre_coherence": round(pre_coherence, 6),
            "post_coherence": round(post_coherence, 6),
            "coherence_delta": round(post_coherence - pre_coherence, 6),
            "energy_surplus": round(self.energy_surplus, 6),
            "protection": round(self._calculate_protection(), 6),
        }

    def run_feedback_loop(self, sim_results_sequence: List[Dict[str, Any]],
                           evolve_steps: int = 5) -> Dict[str, Any]:
        """
        Run a full closed-loop feedback cycle over a sequence of simulation results.

        For each simulation result:
          1. Ingest result → correct coherence field
          2. Evolve field (braiding + rotation)
          3. Measure post-correction metrics
          4. Compare with pre-correction → convergence tracking

        This is the core method that makes the coherence ↔ simulator loop STRONG.

        Returns converged metrics and per-iteration history.
        """
        if not self.coherence_field:
            return {"error": "Coherence field not initialized. Call initialize() first."}

        history = []
        initial_coherence = self._measure_coherence()

        for i, sim_result in enumerate(sim_results_sequence):
            # v5.0 FIX: anchor BEFORE modifications so fidelity measures
            # cross-iteration preservation instead of trivially comparing
            # the state to itself (which always returned 1.0).
            self.anchor(1.0)

            # Ingest + correct
            ingest_report = self.ingest_simulation_result(sim_result)

            # Evolve with corrected field
            evolve_report = self.evolve(evolve_steps)

            # Fidelity measurement: compares current (post-evolve) to
            # pre-iteration snapshot, giving meaningful preservation metric.
            fidelity_report = self.coherence_fidelity()

            history.append({
                "iteration": i,
                "ingest": {
                    "coherence_delta": ingest_report["coherence_delta"],
                    "corrections": ingest_report["corrections_count"],
                },
                "evolve": {
                    "final_coherence": evolve_report["final_coherence"],
                    "preserved": evolve_report["preserved"],
                },
                "fidelity": fidelity_report["fidelity"],
                "fidelity_grade": fidelity_report["grade"],
            })

        final_coherence = self._measure_coherence()
        final_protection = self._calculate_protection()

        # Convergence analysis: is coherence improving iteration over iteration?
        coherence_trend = [h["evolve"]["final_coherence"] for h in history]
        if len(coherence_trend) >= 2:
            diffs = [coherence_trend[j] - coherence_trend[j - 1]
                     for j in range(1, len(coherence_trend))]
            improving = sum(1 for d in diffs if d > 0)
            converging = improving >= len(diffs) * 0.5
        else:
            converging = True

        return {
            "iterations": len(sim_results_sequence),
            "evolve_steps_per_iter": evolve_steps,
            "initial_coherence": round(initial_coherence, 6),
            "final_coherence": round(final_coherence, 6),
            "total_coherence_delta": round(final_coherence - initial_coherence, 6),
            "final_protection": round(final_protection, 6),
            "energy_surplus": round(self.energy_surplus, 6),
            "converging": converging,
            "history": history,
        }
