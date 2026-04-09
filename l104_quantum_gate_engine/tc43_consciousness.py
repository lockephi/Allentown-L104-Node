"""
L104 Technetium-43 Consciousness Circuit — RADIOACTIVE DECAY Level
═══════════════════════════════════════════════════════════════════════════════
Full Tc-43 electron quantum consciousness implementation.

TECHNETIUM — THE LIGHTEST ELEMENT WITH ZERO STABLE ISOTOPES.
Every single isotope undergoes radioactive decay. Odd proton count (43)
means no favorable nuclear pairing energy. The half-filled 4d⁵ shell
creates complex spin-orbit coupling and magnetic frustration.

ELECTRON CONFIGURATION (43 electrons total):
    Core:      1s² 2s² 2p⁶ 3s² 3p⁶
    Transition: 3d¹⁰ 4s² 4p⁶
    Valence:   4d⁵ 5s²
    Qubit mapping: 43 qubits representing Tc orbital electrons

NUCLEAR INSTABILITY → DECOHERENCE:
    Tc has NO stable ground state. This is modeled as intrinsic
    decoherence injected at every circuit layer:
      - Amplitude damping (nuclear decay → electron loss)
      - Phase noise (spin-orbit fluctuation from unstable nucleus)
      - Depolarizing channel (chaotic nuclear recoil)

    The decay rates are derived from Tc-99m's 6.01h half-life:
      γ_nuclear = ln(2) / T_half ≈ 3.21e-5 /s
    Scaled to circuit time: γ_circuit = γ_nuclear × GOD_CODE

CONTRAST WITH Fe-26:
    Iron is self-similar — 26 electrons map to a stable BCC crystal lattice
    with long-range ferromagnetic order. Its circuit is a fixed point attractor.
    Technetium is the opposite — inherent instability means the circuit
    ALWAYS decoheres toward maximum entropy regardless of error correction.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

try:
    from .circuit import GateCircuit
    from .gates import (
        H, CNOT, X, Y, Z, S, T, Rx, Ry, Rz, PHI_GATE, GOD_CODE_PHASE,
        SWAP, IRON_GATE,
    )
    GATE_ENGINE_AVAILABLE = True
except ImportError:
    GATE_ENGINE_AVAILABLE = False

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

# Technetium nuclear constants
TC_ATOMIC_NUMBER = 43
TC_HALF_LIFE_99M = 6.01 * 3600        # Tc-99m half-life in seconds
TC_DECAY_CONSTANT = math.log(2) / TC_HALF_LIFE_99M  # λ = ln(2)/T½
TC_CIRCUIT_GAMMA = TC_DECAY_CONSTANT * GOD_CODE      # Scaled to circuit time
TC_ISOMERIC_ENERGY = 140.511e3         # 140.5 keV gamma emission (eV)
TC_SPIN_ORBIT_COUPLING = 0.12          # Relativistic spin-orbit parameter


@dataclass
class TcOrbitalConfig:
    """Tc-43 electron orbital configuration."""
    orbital: str
    qubits: Tuple[int, ...]
    electrons: int
    phi_power: int
    stability: float        # 0 = maximally unstable, 1 = stable
    decay_coupling: float   # How strongly nuclear decay couples to this shell


class Tc43ConsciousnessCircuit:
    """
    43-qubit quantum consciousness circuit based on technetium electron structure.
    Maps 43 qubits to Tc atom's 43 electrons across orbitals.

    Key difference from Fe-26: inherent nuclear instability injects
    decoherence at every layer. The circuit cannot reach a stable fixed point.
    """

    # Tc-43 orbital configuration
    # 1s² 2s² 2p⁶ 3s² 3p⁶ 3d¹⁰ 4s² 4p⁶ 4d⁵ 5s²
    ORBITALS = {
        '1s': TcOrbitalConfig('1s', (0, 1), 2, 0,
                               stability=0.95, decay_coupling=0.02),
        '2s': TcOrbitalConfig('2s', (2, 3), 2, 1,
                               stability=0.93, decay_coupling=0.03),
        '2p': TcOrbitalConfig('2p', (4, 5, 6, 7, 8, 9), 6, 2,
                               stability=0.90, decay_coupling=0.05),
        '3s': TcOrbitalConfig('3s', (10, 11), 2, 3,
                               stability=0.85, decay_coupling=0.08),
        '3p': TcOrbitalConfig('3p', (12, 13, 14, 15, 16, 17), 6, 4,
                               stability=0.80, decay_coupling=0.10),
        '3d': TcOrbitalConfig('3d', (18, 19, 20, 21, 22, 23, 24, 25, 26, 27), 10, 5,
                               stability=0.70, decay_coupling=0.15),
        '4s': TcOrbitalConfig('4s', (28, 29), 2, 6,
                               stability=0.60, decay_coupling=0.20),
        '4p': TcOrbitalConfig('4p', (30, 31, 32, 33, 34, 35), 6, 7,
                               stability=0.50, decay_coupling=0.25),
        '4d': TcOrbitalConfig('4d', (36, 37, 38, 39, 40), 5, 8,
                               stability=0.30, decay_coupling=0.40),  # HALF-FILLED — maximally frustrated
        '5s': TcOrbitalConfig('5s', (41, 42), 2, 9,
                               stability=0.35, decay_coupling=0.35),
    }

    # Nuclear decay noise profile (per-qubit decoherence rates)
    # Outer shells couple more strongly to nuclear instability
    DECAY_NOISE_PROFILE = {
        '1s': 0.001,   # Deep core — shielded
        '2s': 0.002,
        '2p': 0.005,
        '3s': 0.008,
        '3p': 0.012,
        '3d': 0.020,   # Filled d-shell — moderate coupling
        '4s': 0.035,
        '4p': 0.050,
        '4d': 0.080,   # Half-filled valence — maximum frustration
        '5s': 0.065,   # Conduction — high nuclear coupling
    }

    def __init__(self):
        self.n_qubits = 43
        self.name = "Tc43_Radioactive_Consciousness"
        self.base_frequency = GOD_CODE
        self.intrinsic_gamma = TC_CIRCUIT_GAMMA

    def build_circuit(self, phi_optimization: bool = True,
                      inject_decay_noise: bool = True) -> GateCircuit:
        """
        Build the full 43Q radioactive consciousness circuit.

        Args:
            phi_optimization: Ensure PHI alignment > 0.8
            inject_decay_noise: Add nuclear decay noise gates (Rz phase kicks)

        Returns:
            GateCircuit with full Tc-43 implementation
        """
        if not GATE_ENGINE_AVAILABLE:
            raise ImportError("GateCircuit not available")

        circ = GateCircuit(self.n_qubits, name=self.name)

        # Phase 1: Superposition initialization
        self._phase1_superposition(circ)

        # Phase 2: Core entanglement (1s through 3p — similar to Fe)
        self._phase2_core_entanglement(circ)

        # Phase 3: Filled 3d shell (10 electrons, fully paired)
        self._phase3_filled_3d(circ)

        # Phase 4: Transition layer (4s, 4p)
        self._phase4_transition_layer(circ)

        # Phase 5: CRITICAL — Half-filled 4d⁵ (magnetic frustration)
        self._phase5_half_filled_4d(circ)

        # Phase 6: 5s conduction with nuclear coupling
        self._phase6_5s_conduction(circ)

        # Phase 7: Cross-orbital entanglement
        self._phase7_cross_orbital(circ)

        # Phase 8: Nuclear decay noise injection
        if inject_decay_noise:
            self._phase8_nuclear_decay_noise(circ)

        # Phase 9: Sacred closure (GOD_CODE + PHI)
        self._phase9_sacred_closure(circ)

        # Phase 10: PHI optimization
        if phi_optimization:
            self._phase10_phi_optimization(circ)

        return circ

    def _phase1_superposition(self, circ: GateCircuit):
        """Phase 1: All 43 qubits in superposition."""
        for q in range(self.n_qubits):
            circ.h(q)

    def _phase2_core_entanglement(self, circ: GateCircuit):
        """Phase 2: Entangle core shells (1s, 2s, 2p, 3s, 3p)."""
        # 1s pair
        circ.cx(0, 1)

        # 2s pair
        circ.cx(2, 3)

        # 2p chain (6 qubits, 5 CNOTs)
        for i in range(4, 9):
            circ.cx(i, i + 1)

        # 3s pair
        circ.cx(10, 11)

        # 3p chain (6 qubits, 5 CNOTs)
        for i in range(12, 17):
            circ.cx(i, i + 1)

    def _phase3_filled_3d(self, circ: GateCircuit):
        """Phase 3: Fully filled 3d¹⁰ shell — all paired, stable sublayer."""
        # 10-qubit entanglement chain
        for i in range(18, 27):
            circ.cx(i, i + 1)

        # All 10 electrons paired — alternating spin
        for i in range(18, 28):
            if i % 2 == 0:
                circ.x(i)

    def _phase4_transition_layer(self, circ: GateCircuit):
        """Phase 4: 4s² and 4p⁶ transition shells."""
        # 4s pair
        circ.cx(28, 29)

        # 4p chain (6 qubits, 5 CNOTs)
        for i in range(30, 35):
            circ.cx(i, i + 1)

        # 4p fully paired
        for i in range(30, 36):
            if i % 2 == 0:
                circ.x(i)

    def _phase5_half_filled_4d(self, circ: GateCircuit):
        """
        Phase 5: Half-filled 4d⁵ — THE SOURCE OF MAGNETIC FRUSTRATION.

        Hund's rule: 5 electrons in 5 d-orbitals, all parallel spin.
        This creates maximum spin multiplicity (S=5/2) but NO pairing
        partner, leading to magnetic frustration. In Tc, this combines
        with nuclear instability to create a doubly-chaotic system.

        We model this with:
        - All spins aligned (X gates for |↑⟩)
        - Frustrated entanglement (non-nearest-neighbor CNOTs)
        - Spin-orbit coupling via Rz rotations at TC_SPIN_ORBIT_COUPLING
        """
        # All 5 electrons spin-up (Hund's rule: maximize S)
        for q in range(36, 41):
            circ.x(q)

        # Frustrated entanglement — NOT a simple chain
        # Connect each d-orbital to every other (complete graph = 10 CNOTs)
        d_qubits = list(range(36, 41))
        for i in range(len(d_qubits)):
            for j in range(i + 1, len(d_qubits)):
                circ.cx(d_qubits[i], d_qubits[j])

        # Spin-orbit coupling: Rz kicks proportional to orbital angular momentum
        # m_l values for d-orbitals: -2, -1, 0, +1, +2
        for idx, q in enumerate(d_qubits):
            m_l = idx - 2  # Maps to -2, -1, 0, +1, +2
            angle = TC_SPIN_ORBIT_COUPLING * m_l * math.pi
            if abs(angle) > 1e-10:
                circ.rz(angle, q)

    def _phase6_5s_conduction(self, circ: GateCircuit):
        """Phase 6: 5s² conduction electrons — strongly coupled to nucleus."""
        # 5s pair entangled
        circ.cx(41, 42)

        # 5s couples to 4d (nuclear decay channel)
        for q_5s in [41, 42]:
            circ.cx(q_5s, 38)  # Connect to center of 4d

    def _phase7_cross_orbital(self, circ: GateCircuit):
        """Phase 7: Cross-orbital entanglement for global coherence."""
        # 1s → 2p
        circ.cx(0, 6)
        circ.cx(1, 7)

        # 2s → 3p
        circ.cx(2, 14)
        circ.cx(3, 15)

        # 3s → 3d
        circ.cx(10, 22)
        circ.cx(11, 23)

        # 3p → 4p
        circ.cx(14, 32)
        circ.cx(15, 33)

        # 3d → 4d (filled → half-filled coupling)
        circ.cx(22, 38)
        circ.cx(23, 39)
        circ.cx(24, 40)

        # 4s → 5s (conduction bridge)
        circ.cx(28, 41)
        circ.cx(29, 42)

    def _phase8_nuclear_decay_noise(self, circ: GateCircuit):
        """
        Phase 8: Nuclear decay noise injection.

        Tc's nuclear instability couples to the electron cloud through:
        1. Isomeric transition recoil (γ-ray emission kicks electrons)
        2. Internal conversion (nuclear energy → electron ejection)
        3. Auger cascade (vacancy propagation through shells)

        Modeled as Rz phase kicks with angles derived from decay coupling.
        """
        for orbital_name, config in self.ORBITALS.items():
            noise_rate = self.DECAY_NOISE_PROFILE[orbital_name]
            for q in config.qubits:
                # Nuclear recoil phase kick
                recoil_angle = noise_rate * GOD_CODE * math.pi / 180.0
                circ.rz(recoil_angle, q)

                # Internal conversion: random Y-rotation (spin flip tendency)
                if config.decay_coupling > 0.2:
                    circ.ry(noise_rate * math.pi * PHI, q)

    def _phase9_sacred_closure(self, circ: GateCircuit):
        """Phase 9: Sacred closure — GOD_CODE + PHI alignment."""
        # GOD_CODE_PHASE on all qubits
        for q in range(self.n_qubits):
            circ.append(GOD_CODE_PHASE, [q])

        # PHI_GATE on every other qubit
        for q in range(0, self.n_qubits, 2):
            circ.append(PHI_GATE, [q])

        # Final interference
        for q in range(0, self.n_qubits, 2):
            circ.h(q)

    def _phase10_phi_optimization(self, circ: GateCircuit):
        """Phase 10: Optimize gate counts for PHI alignment."""
        counts = circ.gate_counts
        h_count = counts.get('H', 0)
        cnot_count = counts.get('CNOT', 0)
        phi_count = counts.get('PHI_GATE', 0)

        if phi_count == 0:
            return

        target_phi = int((h_count + cnot_count) / PHI) + 1
        phi_needed = max(0, target_phi - phi_count)

        qubit = 0
        for _ in range(phi_needed):
            circ.append(PHI_GATE, [qubit])
            qubit = (qubit + 1) % self.n_qubits

    def build_reduced_circuit(self, n_qubits: int = 10) -> GateCircuit:
        """
        Build a reduced-scale Tc circuit that preserves the essential physics
        within the trajectory simulator's qubit limits.

        Maps the key orbital physics into n_qubits:
          - 2 qubits: core (1s analog)
          - 2 qubits: filled d-shell (3d analog, paired)
          - 3 qubits: half-filled 4d (frustration source)
          - 2 qubits: 5s conduction
          - 1 qubit:  cross-link

        Args:
            n_qubits: Target qubit count (default 10, max for density sim)
        """
        if not GATE_ENGINE_AVAILABLE:
            raise ImportError("GateCircuit not available")

        circ = GateCircuit(n_qubits, name=f"Tc43_Reduced_{n_qubits}Q")

        # Superposition
        for q in range(n_qubits):
            circ.h(q)

        # Core pair (q0-q1)
        circ.cx(0, 1)

        # Filled d-shell analog (q2-q3, paired)
        circ.cx(2, 3)
        circ.x(2)

        # Half-filled 4d frustration (q4-q6) — COMPLETE GRAPH
        circ.cx(4, 5)
        circ.cx(4, 6)
        circ.cx(5, 6)
        # All spin-up (Hund)
        circ.x(4)
        circ.x(5)
        circ.x(6)
        # Spin-orbit Rz kicks
        circ.rz(-TC_SPIN_ORBIT_COUPLING * 2 * math.pi, 4)  # m_l = -1
        circ.rz(TC_SPIN_ORBIT_COUPLING * 2 * math.pi, 6)   # m_l = +1

        # 5s conduction (q7-q8)
        circ.cx(7, 8)
        # Couple to 4d center
        circ.cx(7, 5)
        circ.cx(8, 5)

        # Cross-link (q9)
        circ.cx(0, 9)  # Core → link
        circ.cx(9, 4)  # Link → 4d

        # Nuclear decay noise injection on outer shells
        for q in [4, 5, 6]:  # 4d — maximum noise
            circ.rz(0.080 * GOD_CODE * math.pi / 180.0, q)
            circ.ry(0.080 * math.pi * PHI, q)
        for q in [7, 8]:  # 5s — high noise
            circ.rz(0.065 * GOD_CODE * math.pi / 180.0, q)
            circ.ry(0.065 * math.pi * PHI, q)

        # Sacred closure
        for q in range(n_qubits):
            circ.append(GOD_CODE_PHASE, [q])
        for q in range(0, n_qubits, 2):
            circ.append(PHI_GATE, [q])
        for q in range(0, n_qubits, 2):
            circ.h(q)

        return circ

    def get_circuit_stats(self, circ: GateCircuit) -> Dict[str, Any]:
        """Get comprehensive circuit statistics."""
        gate_counts = circ.gate_counts
        h_count = gate_counts.get('H', 0)
        cnot_count = gate_counts.get('CNOT', 0)
        phi_count = gate_counts.get('PHI_GATE', 0)
        god_count = gate_counts.get('GOD_CODE_PHASE', 0)
        rz_count = gate_counts.get('Rz', 0)
        ry_count = gate_counts.get('Ry', 0)

        if phi_count > 0:
            ratio = (h_count + cnot_count) / phi_count
            phi_alignment = max(0.0, 1.0 - abs(ratio - PHI) / PHI)
        else:
            phi_alignment = 0.0

        god_ratio = god_count / self.n_qubits if self.n_qubits > 0 else 0
        god_resonance = max(0.0, 1.0 - abs(god_ratio - 1.0))

        # Noise budget: how many gates are dedicated to decay modeling
        noise_gates = rz_count + ry_count
        total = circ.num_operations
        noise_fraction = noise_gates / total if total > 0 else 0

        return {
            'element': 'Tc-43 (Technetium)',
            'stable_isotopes': 0,
            'n_qubits': circ.num_qubits,
            'full_n_qubits': self.n_qubits,
            'depth': int(circ.depth),
            'total_gates': total,
            'two_qubit_gates': circ.two_qubit_count,
            'gate_counts': dict(gate_counts),
            'phi_alignment': phi_alignment,
            'god_resonance': god_resonance,
            'consciousness_score': (phi_alignment + god_resonance) / 2,
            'noise_fraction': noise_fraction,
            'intrinsic_gamma': self.intrinsic_gamma,
            'nuclear_half_life_s': TC_HALF_LIFE_99M,
            'orbital_structure': {
                name: {
                    'qubits': list(config.qubits),
                    'stability': config.stability,
                    'decay_coupling': config.decay_coupling,
                }
                for name, config in self.ORBITALS.items()
            },
        }

    def get_decay_profile(self) -> Dict[str, float]:
        """Get per-orbital decoherence rate profile."""
        return dict(self.DECAY_NOISE_PROFILE)

    def _get_orbital_role(self, orbital: str) -> str:
        roles = {
            '1s': 'Deep core — shielded from nuclear recoil',
            '2s': 'Core stabilization — minimal decay coupling',
            '2p': 'Inner valence — weak nuclear coupling',
            '3s': 'Mid-core — growing instability',
            '3p': 'Mid-valence — moderate decay channel',
            '3d': 'Filled d-shell — paired but nucleus-coupled',
            '4s': 'Transition — strong nuclear coupling',
            '4p': 'Outer transition — high decay rate',
            '4d': 'HALF-FILLED VALENCE — maximum magnetic frustration + decay',
            '5s': 'Conduction — nuclear decay primary channel',
        }
        return roles.get(orbital, 'Unknown')


# ─── Convenience functions ──────────────────────────────────────────────────

def build_tc43_circuit(phi_optimization: bool = True,
                       inject_decay_noise: bool = True) -> GateCircuit:
    """Build the full 43Q Tc radioactive consciousness circuit."""
    builder = Tc43ConsciousnessCircuit()
    return builder.build_circuit(phi_optimization, inject_decay_noise)


def build_tc43_reduced(n_qubits: int = 10) -> GateCircuit:
    """Build reduced-scale Tc circuit for trajectory simulation."""
    builder = Tc43ConsciousnessCircuit()
    return builder.build_reduced_circuit(n_qubits)


def get_tc43_stats(circ: GateCircuit) -> Dict[str, Any]:
    """Get statistics for Tc-43 circuit."""
    builder = Tc43ConsciousnessCircuit()
    return builder.get_circuit_stats(circ)


__all__ = [
    'TcOrbitalConfig',
    'Tc43ConsciousnessCircuit',
    'build_tc43_circuit',
    'build_tc43_reduced',
    'get_tc43_stats',
    'TC_ATOMIC_NUMBER',
    'TC_HALF_LIFE_99M',
    'TC_DECAY_CONSTANT',
    'TC_CIRCUIT_GAMMA',
]
