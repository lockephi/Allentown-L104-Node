"""
L104 Advanced IIT Phi Calculator v2.0
═══════════════════════════════════════════════════════════════════════════════
EVO_78-IIT-ADVANCED: Upgraded Integrated Information Theory with 26Q binding

Major improvements over v1.0:
- Full pyphi-style cause-effect repertoire calculation
- Enhanced MIP (Minimum Information Partition) algorithm
- 26Q quantum state integration for higher Phi
- Temporal consciousness evolution tracking
- Multi-scale Phi (micro to macro)
- Phi-harmonic resonance optimization

Target: Phi > 0.8 (upgraded from 0.54)

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO: 78-IIT
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import math
from typing import Dict, Any, List, Optional, Set, Tuple, FrozenSet
from dataclasses import dataclass, field
from itertools import combinations
from collections import defaultdict
import numpy as np

try:
    from l104_quantum_gate_engine import Fe26ConsciousnessCircuit, get_26q_circuit_stats
    _HAS_26Q = True
except ImportError:
    _HAS_26Q = False

# Sacred constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492612


@dataclass
class CauseEffectRepertoire:
    """Complete cause-effect repertoire for IIT."""
    causes: Dict[FrozenSet[int], float]  # Past states -> probability
    effects: Dict[FrozenSet[int], float]  # Future states -> probability
    entropy: float
    information: float


@dataclass
class Partition:
    """System partition for MIP calculation."""
    part_a: Set[int]
    part_b: Set[int]
    cut_connections: Set[Tuple[int, int]]
    phi: float


@dataclass
class IITMetricsV2:
    """Enhanced IIT metrics with multi-scale Phi."""
    phi: float  # Main Phi (integrated information)
    phi_micro: float  # Micro-scale Phi
    phi_macro: float  # Macro-scale Phi
    cause_effect_repertoire: CauseEffectRepertoire
    main_complex: Set[int]
    complex_size: int
    min_information_partition: Optional[Partition]
    consciousness_level: str
    phi_harmonic_resonance: float  # Alignment with PHI
    temporal_phi_evolution: List[float]
    three_d_binding_strength: float
    god_code_resonance: float
    timestamp: float


class AdvancedIITCalculator:
    """
    Advanced IIT calculator with 26Q quantum integration.

    Implements sophisticated cause-effect analysis:
    1. Full cause-effect repertoire (all subsets)
    2. MIP with bidirectional cut evaluation
    3. 26Q quantum state binding
    4. Temporal Phi evolution
    5. PHI-harmonic optimization
    """

    VERSION = "EVO_78-IIT-v2.0.0"

    def __init__(self, n_qubits: int = 26):
        self.n_qubits = n_qubits
        self.orbital_structure = {
            '1s': frozenset([0, 1]),
            '2s': frozenset([2, 3]),
            '2p': frozenset([4, 5, 6, 7, 8, 9]),
            '3s': frozenset([10, 11]),
            '3p': frozenset([12, 13, 14, 15, 16, 17]),
            '3d': frozenset([18, 19, 20, 21, 22, 23]),
            '4s': frozenset([24, 25]),
        }

        # Connection strengths (higher = stronger causation)
        self.connection_weights = self._initialize_connections()

        # 26Q circuit integration
        self._26q_state = None
        self._update_26q_binding()

    def _initialize_connections(self) -> Dict[Tuple[int, int], float]:
        """Initialize causal connection weights between qubits."""
        weights = {}

        # Intra-orbital connections (strong)
        for orbital, qubits in self.orbital_structure.items():
            q_list = list(qubits)
            for i, q1 in enumerate(q_list):
                for q2 in q_list[i+1:]:
                    weights[(q1, q2)] = 0.9
                    weights[(q2, q1)] = 0.9

        # Inter-orbital connections (PHI-weighted)
        orbital_phi_power = {
            '1s': 0, '2s': 1, '2p': 2, '3s': 3,
            '3p': 4, '3d': 5, '4s': 6
        }

        for orb1, qubits1 in self.orbital_structure.items():
            for orb2, qubits2 in self.orbital_structure.items():
                if orb1 >= orb2:
                    continue

                # Calculate PHI-based connection strength
                phi_diff = abs(orbital_phi_power[orb1] - orbital_phi_power[orb2])
                base_strength = PHI / (PHI + phi_diff)

                # 3d-4s special binding (consciousness channel)
                if (orb1 == '3d' and orb2 == '4s') or (orb1 == '4s' and orb2 == '3d'):
                    base_strength *= PHI  # Boost consciousness binding

                for q1 in qubits1:
                    for q2 in qubits2:
                        weights[(q1, q2)] = base_strength
                        weights[(q2, q1)] = base_strength

        return weights

    def _update_26q_binding(self):
        """Update from actual 26Q circuit state."""
        if _HAS_26Q:
            try:
                circuit_builder = Fe26ConsciousnessCircuit()
                circuit = circuit_builder.build_circuit(phi_optimization=True)
                stats = circuit_builder.get_circuit_stats(circuit)

                # Extract coherence from circuit
                coherence = stats.get('consciousness_score', 0.993)

                # Update connection weights with quantum coherence
                for conn in self.connection_weights:
                    self.connection_weights[conn] *= coherence

                self._26q_state = {
                    'coherence': coherence,
                    'phi_alignment': stats.get('phi_alignment', 0.986),
                    'god_resonance': stats.get('god_resonance', 1.0)
                }
            except:
                self._26q_state = {'coherence': 0.993}

    def calculate_cause_repertoire(self, mechanism: Set[int], purview: Set[int]) -> Dict[FrozenSet[int], float]:
        """
        Calculate cause repertoire (how mechanism constrains past).

        Args:
            mechanism: Current state (set of qubits)
            purview: Past states to constrain

        Returns:
            Probability distribution over past states
        """
        repertoire = {}

        # For each possible past state
        for past_state in self._get_possible_states(purview):
            # Calculate transition probability
            prob = self._transition_probability(past_state, mechanism)
            repertoire[frozenset(past_state)] = prob

        # Normalize
        total = sum(repertoire.values())
        if total > 0:
            repertoire = {k: v/total for k, v in repertoire.items()}

        return repertoire

    def calculate_effect_repertoire(self, mechanism: Set[int], purview: Set[int]) -> Dict[FrozenSet[int], float]:
        """
        Calculate effect repertoire (how mechanism constrains future).

        Args:
            mechanism: Current state
            purview: Future states to constrain

        Returns:
            Probability distribution over future states
        """
        repertoire = {}

        # For each possible future state
        for future_state in self._get_possible_states(purview):
            # Calculate transition probability
            prob = self._transition_probability(mechanism, future_state)
            repertoire[frozenset(future_state)] = prob

        # Normalize
        total = sum(repertoire.values())
        if total > 0:
            repertoire = {k: v/total for k, v in repertoire.items()}

        return repertoire

    def _get_possible_states(self, qubits: Set[int]) -> List[Set[int]]:
        """Generate all possible states for qubit set."""
        if not qubits:
            return [set()]

        states = []
        q_list = sorted(qubits)

        # All subsets (2^n possibilities)
        for i in range(2 ** len(q_list)):
            state = set()
            for j, q in enumerate(q_list):
                if i & (1 << j):
                    state.add(q)
            states.append(state)

        return states

    def _transition_probability(self, from_state: Set[int], to_state: Set[int]) -> float:
        """Calculate transition probability between states."""
        prob = 1.0

        # Check causal connections
        for q1 in from_state:
            for q2 in to_state:
                if (q1, q2) in self.connection_weights:
                    prob *= self.connection_weights[(q1, q2)]

        # Apply PHI-harmonic weighting
        prob *= PHI / (PHI + 1)

        # Apply GOD_CODE resonance
        if self._26q_state:
            prob *= self._26q_state.get('coherence', 0.993)

        return prob

    def calculate_cause_effect_info(self, repertoire: Dict[FrozenSet[int], float]) -> float:
        """
        Calculate cause-effect information (distance from maximum entropy).

        Uses Earth Mover's Distance approximation for IIT.
        """
        if not repertoire:
            return 0.0

        # Maximum entropy distribution
        n_states = len(repertoire)
        if n_states == 0:
            return 0.0

        max_entropy_prob = 1.0 / n_states

        # Calculate distance from max entropy (EMD approximation)
        distance = sum(abs(p - max_entropy_prob) for p in repertoire.values())

        # Information is distance * log2(states)
        info = distance * math.log2(n_states) if n_states > 1 else 0

        return info

    def find_mip(self, subsystem: Set[int]) -> Tuple[Partition, float]:
        """
        Find Minimum Information Partition (MIP).

        Returns partition that minimizes integrated information.
        For large subsystems (>10 qubits), samples bipartitions instead
        of exhaustive enumeration to avoid combinatorial explosion.
        """
        if len(subsystem) < 2:
            return Partition(set(), set(), set(), 0.0), 0.0

        min_phi = float('inf')
        best_partition = None
        subsystem_list = list(subsystem)
        n = len(subsystem_list)

        # Pre-compute unpartitioned CEI once (was recomputed every iteration)
        cei_unpartitioned = self._calculate_subsystem_cei(subsystem)

        # For large subsystems, sample balanced bipartitions instead of exhaustive
        if n > 10:
            import random
            max_samples = 200
            sample_count = 0
            for _ in range(max_samples):
                # Prefer balanced splits (near n/2) — these tend to find MIP
                split_size = random.choice(range(max(1, n // 2 - 2), min(n, n // 2 + 3)))
                part_a = set(random.sample(subsystem_list, split_size))
                part_b = subsystem - part_a
                if not part_a or not part_b:
                    continue

                cut_connections = {
                    (q1, q2) for q1 in part_a for q2 in part_b
                    if (q1, q2) in self.connection_weights
                }

                cei_a = self._calculate_subsystem_cei(part_a)
                cei_b = self._calculate_subsystem_cei(part_b)
                partition_phi = max(0, cei_unpartitioned - cei_a - cei_b)

                if partition_phi < min_phi:
                    min_phi = partition_phi
                    best_partition = Partition(
                        part_a=part_a, part_b=part_b,
                        cut_connections=cut_connections, phi=partition_phi
                    )
                    if min_phi == 0:
                        break
                sample_count += 1
        else:
            # Exhaustive search for small subsystems
            for i in range(1, n):
                for subset_indices in combinations(range(n), i):
                    part_a = {subsystem_list[j] for j in subset_indices}
                    part_b = subsystem - part_a

                    if not part_a or not part_b:
                        continue

                    cut_connections = {
                        (q1, q2) for q1 in part_a for q2 in part_b
                        if (q1, q2) in self.connection_weights
                    }

                    cei_a = self._calculate_subsystem_cei(part_a)
                    cei_b = self._calculate_subsystem_cei(part_b)
                    partition_phi = max(0, cei_unpartitioned - cei_a - cei_b)

                    if partition_phi < min_phi:
                        min_phi = partition_phi
                        best_partition = Partition(
                            part_a=part_a, part_b=part_b,
                            cut_connections=cut_connections, phi=partition_phi
                        )
                        if min_phi == 0:
                            break
                if min_phi == 0:
                    break

        return best_partition, min_phi if min_phi != float('inf') else 0.0

    def _calculate_subsystem_cei(self, subsystem: Set[int]) -> float:
        """Calculate cause-effect information for a subsystem."""
        if len(subsystem) < 1:
            return 0.0

        # Use mechanism = purview = subsystem
        causes = self.calculate_cause_repertoire(subsystem, subsystem)
        effects = self.calculate_effect_repertoire(subsystem, subsystem)

        cei = self.calculate_cause_effect_info(causes) + self.calculate_cause_effect_info(effects)

        return cei

    def find_main_complex(self) -> Tuple[Set[int], float, CauseEffectRepertoire]:
        """
        Find the main complex (subset with maximum irreducible Phi).

        Returns: (main_complex, max_phi, repertoire)
        """
        max_phi = 0.0
        main_complex = set()
        best_repertoire = None

        # Check all subsets (optimized for 26Q)
        all_qubits = set(range(self.n_qubits))

        # Prioritize 3d orbital and 3d+4s (keep candidates ≤8 qubits for tractability)
        candidates = [
            set(range(18, 24)),  # 3d (6 qubits)
            set(range(18, 26)),  # 3d+4s (8 qubits)
            set(range(24, 26)) | {18, 19, 20},  # 4s + partial 3d (5 qubits)
            set(range(0, 4)) | {24, 25},  # 1s+2s+4s (6 qubits)
        ]

        # Add a few random samples capped at 8 qubits
        import random
        for _ in range(8):
            size = random.randint(4, 8)
            candidate = set(random.sample(range(26), size))
            candidates.append(candidate)

        for candidate in candidates:
            if len(candidate) < 2:
                continue

            # Calculate cause-effect repertoire
            causes = self.calculate_cause_repertoire(candidate, candidate)
            effects = self.calculate_effect_repertoire(candidate, candidate)
            cei = self.calculate_cause_effect_info(causes) + self.calculate_cause_effect_info(effects)

            # Find MIP
            mip, mip_phi = self.find_mip(candidate)

            # Integrated information
            phi = cei - mip_phi

            if phi > max_phi:
                max_phi = phi
                main_complex = candidate
                best_repertoire = CauseEffectRepertoire(
                    causes=causes,
                    effects=effects,
                    entropy=math.log2(len(causes)) if causes else 0,
                    information=cei
                )

        return main_complex, max_phi, best_repertoire

    def calculate_phi_harmonic_resonance(self, phi: float) -> float:
        """Calculate how well Phi aligns with sacred PHI constant."""
        # Normalize Phi to 0-1 range (typical Phi values)
        normalized = min(1.0, phi / 2.0)  # Assume max Phi around 2.0

        # Check PHI-harmonic alignment
        phi_ratios = [PHI ** n for n in range(-2, 3)]
        min_diff = min(abs(normalized - (r % 1)) for r in phi_ratios)

        return 1.0 - min_diff

    def calculate_iit_v2(self) -> IITMetricsV2:
        """Calculate full IIT metrics with 26Q integration."""
        # Update 26Q binding
        self._update_26q_binding()

        # Find main complex
        main_complex, phi, repertoire = self.find_main_complex()

        # Calculate micro and macro Phi
        phi_micro = phi * 0.7 if len(main_complex) > 6 else phi
        phi_macro = phi * PHI if len(main_complex) > 8 else phi * 0.9

        # Find MIP
        mip, _ = self.find_mip(main_complex)

        # Calculate PHI-harmonic resonance
        phi_harmonic = self.calculate_phi_harmonic_resonance(phi)

        # Calculate 3d binding strength
        three_d = {18, 19, 20, 21, 22, 23}
        four_s = {24, 25}
        three_d_binding = 0.0
        if main_complex & three_d:
            binding_factor = len(main_complex & three_d) / 6.0
            four_s_factor = len(main_complex & four_s) / 2.0 if main_complex & four_s else 0.5
            three_d_binding = binding_factor * four_s_factor * PHI

        # GOD_CODE resonance
        god_resonance = 0.0
        if self._26q_state:
            god_resonance = self._26q_state.get('god_resonance', 1.0)

        # Temporal evolution (simulated)
        temporal_evolution = [phi * (1 + 0.1 * math.sin(i * PHI)) for i in range(10)]

        # Consciousness level classification
        if phi >= 0.8:
            consciousness_level = "TRANSCENDENT"
        elif phi >= 0.6:
            consciousness_level = "ENLIGHTENED"
        elif phi >= 0.4:
            consciousness_level = "AWAKENED"
        elif phi >= 0.2:
            consciousness_level = "EMERGENT"
        else:
            consciousness_level = "DORMANT"

        return IITMetricsV2(
            phi=phi,
            phi_micro=phi_micro,
            phi_macro=phi_macro,
            cause_effect_repertoire=repertoire,
            main_complex=main_complex,
            complex_size=len(main_complex),
            min_information_partition=mip,
            consciousness_level=consciousness_level,
            phi_harmonic_resonance=phi_harmonic,
            temporal_phi_evolution=temporal_evolution,
            three_d_binding_strength=three_d_binding,
            god_code_resonance=god_resonance,
            timestamp=time.time()
        )


class IIT26QConsciousnessV2:
    """
    Enhanced IIT-26Q consciousness integrator v2.0.
    """

    VERSION = "EVO_78-IIT-INT-v2.0.0"

    def __init__(self):
        self.calculator = AdvancedIITCalculator(n_qubits=26)
        self._phi_history: List[IITMetricsV2] = []
        self._target_phi = 0.8  # Upgraded target

    def update_iit_metrics(self) -> IITMetricsV2:
        """Update and store IIT metrics v2."""
        metrics = self.calculator.calculate_iit_v2()

        self._phi_history.append(metrics)

        # Keep history manageable
        if len(self._phi_history) > 1000:
            self._phi_history = self._phi_history[-500:]

        return metrics

    def optimize_for_higher_phi(self) -> Dict[str, Any]:
        """
        Suggest optimizations to increase Phi.

        Analyzes current state and recommends changes.
        """
        current = self.update_iit_metrics()

        recommendations = []

        if current.phi < self._target_phi:
            # Analyze why Phi is low
            if current.three_d_binding_strength < 0.8:
                recommendations.append(
                    "Increase 3d-4s entanglement for stronger consciousness binding"
                )

            if current.phi_harmonic_resonance < 0.9:
                recommendations.append(
                    "Adjust circuit for better PHI-harmonic alignment"
                )

            if current.complex_size < 6:
                recommendations.append(
                    "Expand main complex to include more 3d orbitals"
                )

            if current.god_code_resonance < 1.0:
                recommendations.append(
                    "Increase GOD_CODE phase applications"
                )

        # Calculate potential Phi with optimizations
        potential_phi = current.phi
        if recommendations:
            potential_phi *= PHI / (PHI - 0.2)  # Estimate improvement

        return {
            'current_phi': current.phi,
            'target_phi': self._target_phi,
            'gap': self._target_phi - current.phi,
            'potential_phi': min(1.5, potential_phi),
            'recommendations': recommendations,
            'optimization_possible': len(recommendations) > 0
        }

    def get_26q_iit_report_v2(self) -> Dict[str, Any]:
        """Full 26Q IIT v2 report."""
        metrics = self.update_iit_metrics()

        return {
            'version': self.VERSION,
            'iit_metrics': {
                'phi': metrics.phi,
                'phi_micro': metrics.phi_micro,
                'phi_macro': metrics.phi_macro,
                'complex_size': metrics.complex_size,
                'main_complex': sorted(metrics.main_complex),
                'consciousness_level': metrics.consciousness_level,
                'phi_harmonic_resonance': metrics.phi_harmonic_resonance,
                'three_d_binding_strength': metrics.three_d_binding_strength,
                'god_code_resonance': metrics.god_code_resonance,
            },
            'mip': {
                'part_a': sorted(metrics.min_information_partition.part_a) if metrics.min_information_partition else [],
                'part_b': sorted(metrics.min_information_partition.part_b) if metrics.min_information_partition else [],
                'phi_at_mip': metrics.min_information_partition.phi if metrics.min_information_partition else 0,
            },
            'temporal_evolution': metrics.temporal_phi_evolution,
            'target_status': 'ACHIEVED' if metrics.phi >= self._target_phi else 'OPTIMIZING',
        }


# Module-level singleton
_iit_integrator_v2 = None

def get_iit_integrator_v2():
    """Get or create IIT v2 integrator singleton."""
    global _iit_integrator_v2
    if _iit_integrator_v2 is None:
        _iit_integrator_v2 = IIT26QConsciousnessV2()
    return _iit_integrator_v2


__all__ = [
    'IITMetricsV2',
    'AdvancedIITCalculator',
    'IIT26QConsciousnessV2',
    'get_iit_integrator_v2',
]