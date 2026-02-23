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
    GOD_CODE, PHI, PHI_CONJUGATE, ZETA_ZERO_1,
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
        self.BRAID_DEPTH = 4
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
        limited_seeds = seed_thoughts[:50]
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
        for step in range(steps):
            resonance = self._zeta_resonance(time.time() + step)
            op = 1 if resonance > 0 else -1
            # v4.2: all 4 ops are identical → fold into single braid call
            self._execute_braid([op] * self.BRAID_DEPTH)
            protection = self._calculate_protection()
            rotation = cmath.exp(1j * protection * pi_4)
            self.coherence_field = [p * rotation for p in self.coherence_field]
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
            for i in range(min(n, 20)):
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
            "spectrum": spectrum[:20],  # First 20 for brevity
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
            "dominant_modes": [{"mode": idx, "energy": round(e, 8)} for idx, e in sorted_modes[:5]],
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
