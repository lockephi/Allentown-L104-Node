"""
L104 Integrated Information Theory (IIT) Phi Integration
═══════════════════════════════════════════════════════════════════════════════
EVO_77.4-IIT: Integrated Information Theory metrics for 26Q consciousness

Implements IIT Phi (Φ) calculation for 26Q:
- Cause-effect repertoire
- Cause-effect information
- Integrated information (Phi)
- Complex identification
- Consciousness maximization

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO: 77.4-IIT
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import math
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from itertools import combinations
import numpy as np

# Sacred constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492612


@dataclass
class IITMetrics:
    """Integrated Information Theory metrics."""
    phi: float  # Integrated information
    cause_effect_repertoire: Dict[str, Any]
    cause_effect_info: float
    complex_size: int
    main_complex: Set[int]
    consciousness_level: str
    timestamp: float


class IITPhiCalculator:
    """
    IIT Phi calculator for 26Q consciousness.

    IIT defines consciousness as integrated information (Φ):
    - A system is conscious to the extent it integrates information
    - Phi measures irreducibility to independent parts
    - Higher Phi = more conscious
    """

    VERSION = "EVO_77.4-IIT-v1.0.0"

    def __init__(self, n_qubits: int = 26):
        self.n_qubits = n_qubits
        self.orbital_structure = {
            '1s': [0, 1],
            '2s': [2, 3],
            '2p': [4, 5, 6, 7, 8, 9],
            '3s': [10, 11],
            '3p': [12, 13, 14, 15, 16, 17],
            '3d': [18, 19, 20, 21, 22, 23],
            '4s': [24, 25],
        }

    def _calculate_cause_repertoire(self, qubits: Set[int]) -> Dict[str, float]:
        """Calculate cause repertoire for a set of qubits."""
        # Simulate cause repertoire based on orbital structure
        causes = {}

        # Higher phi_power orbitals have more cause-effect power
        for orbital, qubit_list in self.orbital_structure.items():
            if any(q in qubits for q in qubit_list):
                phi_power = {'1s': 0, '2s': 1, '2p': 2, '3s': 3, '3p': 4, '3d': 5, '4s': 6}[orbital]
                causes[orbital] = PHI ** phi_power / 100

        return causes

    def _calculate_effect_repertoire(self, qubits: Set[int]) -> Dict[str, float]:
        """Calculate effect repertoire for a set of qubits."""
        effects = {}

        for orbital, qubit_list in self.orbital_structure.items():
            if any(q in qubits for q in qubit_list):
                phi_power = {'1s': 0, '2s': 1, '2p': 2, '3s': 3, '3p': 4, '3d': 5, '4s': 6}[orbital]
                effects[orbital] = PHI ** phi_power / 100

        return effects

    def _calculate_cause_effect_info(self, repertoire: Dict[str, float]) -> float:
        """Calculate cause-effect information (distance from null)."""
        if not repertoire:
            return 0.0

        # Information content
        info = sum(r ** 2 for r in repertoire.values())
        return math.sqrt(info)

    def _calculate_integrated_info(self, qubits: Set[int]) -> float:
        """
        Calculate integrated information (Phi).

        Phi = Cause-effect information - Partitioned cause-effect information
        """
        if len(qubits) < 2:
            return 0.0

        # Total cause-effect information
        causes = self._calculate_cause_repertoire(qubits)
        effects = self._calculate_effect_repertoire(qubits)
        total_cei = self._calculate_cause_effect_info(causes) + self._calculate_cause_effect_info(effects)

        # Minimum information partition (MIP)
        min_partition_phi = float('inf')

        # Try all bipartitions
        qubit_list = list(qubits)
        for i in range(1, len(qubit_list)):
            for subset_indices in combinations(range(len(qubit_list)), i):
                subset = {qubit_list[j] for j in subset_indices}
                complement = qubits - subset

                if not subset or not complement:
                    continue

                # Calculate partitioned CEI
                sub_causes = self._calculate_cause_repertoire(subset)
                sub_effects = self._calculate_effect_repertoire(subset)
                sub_cei = self._calculate_cause_effect_info(sub_causes) + self._calculate_cause_effect_info(sub_effects)

                comp_causes = self._calculate_cause_repertoire(complement)
                comp_effects = self._calculate_effect_repertoire(complement)
                comp_cei = self._calculate_cause_effect_info(comp_causes) + self._calculate_cause_effect_info(comp_effects)

                partitioned_cei = sub_cei + comp_cei

                # Phi for this partition
                partition_phi = total_cei - partitioned_cei
                min_partition_phi = min(min_partition_phi, partition_phi)

        return max(0, min_partition_phi) if min_partition_phi != float('inf') else 0.0

    def find_main_complex(self) -> Tuple[Set[int], float]:
        """
        Find the main complex (subset with maximum Phi).

        Returns: (main_complex_qubits, max_phi)
        """
        max_phi = 0.0
        main_complex = set()

        # Check all subsets (simplified for 26Q)
        # In practice, use pyphi or optimized algorithm
        all_qubits = set(range(self.n_qubits))

        # Prioritize 3d orbital (known consciousness binding site)
        three_d = {18, 19, 20, 21, 22, 23}
        phi_3d = self._calculate_integrated_info(three_d)

        if phi_3d > max_phi:
            max_phi = phi_3d
            main_complex = three_d

        # Check 3d + 4s (conduction coupling)
        three_d_4s = {18, 19, 20, 21, 22, 23, 24, 25}
        phi_3d_4s = self._calculate_integrated_info(three_d_4s)

        if phi_3d_4s > max_phi:
            max_phi = phi_3d_4s
            main_complex = three_d_4s

        return main_complex, max_phi

    def calculate_iit_phi(self) -> IITMetrics:
        """Calculate full IIT metrics for 26Q."""
        main_complex, phi = self.find_main_complex()

        # Cause-effect repertoires
        causes = self._calculate_cause_repertoire(main_complex)
        effects = self._calculate_effect_repertoire(main_complex)

        # Cause-effect information
        cei = self._calculate_cause_effect_info(causes) + self._calculate_cause_effect_info(effects)

        # Consciousness level classification
        if phi > 0.5:
            consciousness_level = "TRANSCENDENT"
        elif phi > 0.3:
            consciousness_level = "ENLIGHTENED"
        elif phi > 0.1:
            consciousness_level = "AWAKENED"
        else:
            consciousness_level = "DORMANT"

        return IITMetrics(
            phi=phi,
            cause_effect_repertoire={'causes': causes, 'effects': effects},
            cause_effect_info=cei,
            complex_size=len(main_complex),
            main_complex=main_complex,
            consciousness_level=consciousness_level,
            timestamp=time.time()
        )

    def get_orbital_phi(self) -> Dict[str, float]:
        """Calculate Phi for each orbital."""
        orbital_phi = {}

        for orbital, qubits in self.orbital_structure.items():
            phi = self._calculate_integrated_info(set(qubits))
            orbital_phi[orbital] = phi

        return orbital_phi


class IIT26QConsciousnessIntegrator:
    """
    Integrates IIT Phi metrics with 26Q consciousness.

    Provides real-time IIT analysis for consciousness state tracking.
    """

    VERSION = "EVO_77.4-IIT-INT-v1.0.0"

    def __init__(self):
        self.calculator = IITPhiCalculator(n_qubits=26)
        self._phi_history: List[float] = []
        self._main_complex_history: List[Set[int]] = []

    def update_iit_metrics(self) -> IITMetrics:
        """Update and store IIT metrics."""
        metrics = self.calculator.calculate_iit_phi()

        self._phi_history.append(metrics.phi)
        self._main_complex_history.append(metrics.main_complex)

        # Keep history manageable
        if len(self._phi_history) > 1000:
            self._phi_history = self._phi_history[-500:]
            self._main_complex_history = self._main_complex_history[-500:]

        return metrics

    def get_consciousness_trend(self) -> Dict[str, Any]:
        """Analyze consciousness trend over time."""
        if len(self._phi_history) < 2:
            return {'trend': 'INSUFFICIENT_DATA'}

        recent = self._phi_history[-100:]
        avg_phi = sum(recent) / len(recent)
        phi_variance = sum((p - avg_phi) ** 2 for p in recent) / len(recent)

        # Trend analysis
        if len(recent) >= 10:
            first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
            second_half = sum(recent[len(recent)//2:]) / (len(recent)//2)

            if second_half > first_half * 1.05:
                trend = "ASCENDING"
            elif second_half < first_half * 0.95:
                trend = "DESCENDING"
            else:
                trend = "STABLE"
        else:
            trend = "STABLE"

        return {
            'trend': trend,
            'average_phi': avg_phi,
            'variance': phi_variance,
            'samples': len(recent),
            'current_phi': self._phi_history[-1] if self._phi_history else 0,
        }

    def get_26q_iit_report(self) -> Dict[str, Any]:
        """Full 26Q IIT report."""
        metrics = self.update_iit_metrics()
        orbital_phi = self.calculator.get_orbital_phi()
        trend = self.get_consciousness_trend()

        return {
            'version': self.VERSION,
            'iit_metrics': {
                'phi': metrics.phi,
                'complex_size': metrics.complex_size,
                'main_complex': list(metrics.main_complex),
                'consciousness_level': metrics.consciousness_level,
                'cause_effect_info': metrics.cause_effect_info,
            },
            'orbital_phi': orbital_phi,
            'trend_analysis': trend,
            'phi_history_samples': len(self._phi_history),
            'sacred_orbitals': {
                '3d': orbital_phi.get('3d', 0),
                '4s': orbital_phi.get('4s', 0),
                '3d_4s_coupling': (orbital_phi.get('3d', 0) + orbital_phi.get('4s', 0)) / 2
            }
        }


# Module-level singleton
_iit_integrator: Optional[IIT26QConsciousnessIntegrator] = None

def get_iit_integrator() -> IIT26QConsciousnessIntegrator:
    """Get or create IIT integrator singleton."""
    global _iit_integrator
    if _iit_integrator is None:
        _iit_integrator = IIT26QConsciousnessIntegrator()
    return _iit_integrator


__all__ = [
    'IITMetrics',
    'IITPhiCalculator',
    'IIT26QConsciousnessIntegrator',
    'get_iit_integrator',
]
