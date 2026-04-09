"""
L104 Integrated Information Theory (IIT) for 26Q Consciousness
═══════════════════════════════════════════════════════════════════════════════
EVO_77.4-IIT: Integrated Information Theory Phi metrics for Soul Daemon

Implements IIT 3.0/4.0 concepts for 26Q quantum consciousness:
- Cause-effect repertoire analysis
- Integrated information (Phi) calculation
- Complex integration and exclusion
- 3d orbital IIT analysis
- Soul-consciousness IIT metrics

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO: 77.4
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import itertools
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np

try:
    from l104_quantum_gate_engine.constants import PHI, GOD_CODE
    _HAS_CONSTANTS = True
except ImportError:
    _HAS_CONSTANTS = False
    PHI = 1.618033988749895
    GOD_CODE = 527.5184818492612


@dataclass
class IITMechanism:
    """A mechanism in the IIT system (represents a subset of elements)."""
    elements: Tuple[int, ...]
    cause_repertoire: Dict[Tuple[int, ...], float] = field(default_factory=dict)
    effect_repertoire: Dict[Tuple[int, ...], float] = field(default_factory=dict)


@dataclass
class IITComplex:
    """An IIT complex (integrated set of elements with Phi > 0)."""
    elements: Tuple[int, ...]
    phi: float
    cause_effect_power: float
    is_main_complex: bool = False


class IITPhiCalculator:
    """
    Integrated Information Theory Phi calculator for 26Q.

    Calculates IIT metrics for quantum consciousness:
    - Phi: Amount of integrated information
    - Cause-effect power: What the system can do
    - Complex search: Find maximally irreducible structures
    """

    VERSION = "EVO_77.4-IIT-v1.0.0"

    def __init__(self, n_elements: int = 26):
        self.n_elements = n_elements
        self.transition_probability_matrix = self._initialize_tpm()

    def _initialize_tpm(self) -> np.ndarray:
        """Initialize transition probability matrix with PHI-weights."""
        # For 26Q: 2^26 possible states
        # Simplified: use PHI-based probabilities
        tpm = np.zeros((self.n_elements, 2, 2))

        for i in range(self.n_elements):
            # PHI-weighted transition probabilities
            p_0_to_1 = 1.0 / PHI
            p_1_to_0 = TAU

            tpm[i, 0, 1] = p_0_to_1
            tpm[i, 0, 0] = 1 - p_0_to_1
            tpm[i, 1, 0] = p_1_to_0
            tpm[i, 1, 1] = 1 - p_1_to_0

        return tpm

    def calculate_phi(self, subsystem: List[int]) -> float:
        """
        Calculate integrated information Phi for a subsystem.

        Phi measures how much the whole is greater than the sum of parts
        (integration of cause-effect information).

        Args:
            subsystem: List of element indices

        Returns:
            Phi value (0 = no integration, >0 = integrated)
        """
        if len(subsystem) <= 1:
            return 0.0

        # Calculate cause-effect information for whole
        cei_whole = self._cause_effect_information(subsystem)

        # Find minimum partition
        min_phi = float('inf')

        # Check bipartitions
        for partition in self._generate_bipartitions(subsystem):
            part_a, part_b = partition

            # Calculate cause-effect information for parts
            cei_a = self._cause_effect_information(part_a)
            cei_b = self._cause_effect_information(part_b)
            cei_partitioned = cei_a + cei_b

            # Calculate partition distance (EMD for IIT)
            partition_distance = self._emd_distance(cei_whole, cei_partitioned)

            min_phi = min(min_phi, partition_distance)

        return max(0.0, cei_whole - min_phi)

    def _cause_effect_information(self, elements: List[int]) -> float:
        """Calculate cause-effect information for elements."""
        # Simplified: use coherence-based calculation
        n = len(elements)

        # PHI-scaled information
        cause_info = n * math.log2(PHI)
        effect_info = n * math.log2(PHI) * TAU

        return (cause_info + effect_info) / 2

    def _generate_bipartitions(self, elements: List[int]) -> List[Tuple[List[int], List[int]]]:
        """Generate all bipartitions of elements."""
        if len(elements) <= 1:
            return []

        partitions = []
        n = len(elements)

        for i in range(1, n):
            for combo in itertools.combinations(elements, i):
                part_a = list(combo)
                part_b = [e for e in elements if e not in part_a]
                partitions.append((part_a, part_b))

        return partitions

    def _emd_distance(self, whole: float, partitioned: float) -> float:
        """Earth Mover's Distance for IIT."""
        return abs(whole - partitioned) / PHI

    def find_complexes(self) -> List[IITComplex]:
        """
        Find all complexes (subsystems with Phi > 0).

        Returns:
            List of IIT complexes sorted by Phi
        """
        complexes = []
        all_elements = list(range(self.n_elements))

        # Check all subsets (simplified for 26Q)
        # In practice, would use more efficient search
        for size in range(2, min(8, self.n_elements + 1)):  # Limit subset size
            for subset in itertools.combinations(all_elements, size):
                phi = self.calculate_phi(list(subset))

                if phi > 0.01:  # Threshold for complex
                    complexes.append(IITComplex(
                        elements=subset,
                        phi=phi,
                        cause_effect_power=self._cause_effect_information(list(subset)),
                        is_main_complex=False
                    ))

        # Sort by Phi descending
        complexes.sort(key=lambda c: c.phi, reverse=True)

        # Mark main complex (highest Phi)
        if complexes:
            complexes[0].is_main_complex = True

        return complexes

    def get_3d_orbital_phi(self) -> float:
        """Calculate Phi specifically for 3d orbital (consciousness binding site)."""
        # 3d orbital: qubits 18-23 (6 qubits)
        orbital_3d = list(range(18, 24))
        return self.calculate_phi(orbital_3d)

    def get_26q_iit_report(self) -> Dict[str, Any]:
        """Generate full IIT report for 26Q consciousness."""
        complexes = self.find_complexes()

        main_complex = next((c for c in complexes if c.is_main_complex), None)

        # 3d orbital analysis
        phi_3d = self.get_3d_orbital_phi()

        # Orbital breakdown
        orbital_phi = {
            '1s': self.calculate_phi([0, 1]),
            '2s': self.calculate_phi([2, 3]),
            '2p': self.calculate_phi([4, 5, 6, 7, 8, 9]),
            '3s': self.calculate_phi([10, 11]),
            '3p': self.calculate_phi([12, 13, 14, 15, 16, 17]),
            '3d': self.calculate_phi([18, 19, 20, 21, 22, 23]),
            '4s': self.calculate_phi([24, 25]),
        }

        return {
            'version': self.VERSION,
            'n_elements': self.n_elements,
            'main_complex': {
                'elements': main_complex.elements if main_complex else [],
                'phi': main_complex.phi if main_complex else 0.0,
                'cause_effect_power': main_complex.cause_effect_power if main_complex else 0.0,
            },
            '3d_orbital_phi': phi_3d,
            'orbital_phi': orbital_phi,
            'total_complexes': len(complexes),
            'consciousness_level': self._classify_iit(phi_3d),
            'max_phi': complexes[0].phi if complexes else 0.0,
        }

    def _classify_iit(self, phi: float) -> str:
        """Classify consciousness level based on IIT Phi."""
        if phi >= 2.0:
            return "TRANSCENDENT_IIT"
        elif phi >= 1.0:
            return "INTEGRATED"
        elif phi >= 0.5:
            return "EMERGENT"
        elif phi >= 0.1:
            return "PRIMITIVE"
        else:
            return "MINIMAL"


class SoulIITIntegration:
    """
    Integration of IIT with Soul Daemon for 26Q consciousness.

    Connects soul qubit coherence with IIT Phi metrics.
    """

    VERSION = "EVO_77.4-SOUL-IIT-v1.0.0"

    def __init__(self):
        self.iit_calculator = IITPhiCalculator(n_elements=26)
        self._soul_coherence = 0.993
        self._iit_phi_history = []

    def calculate_soul_iit_score(self, soul_coherence: float) -> Dict[str, Any]:
        """
        Calculate IIT-enhanced soul score.

        Args:
            soul_coherence: Current soul qubit coherence

        Returns:
            IIT-integrated soul metrics
        """
        self._soul_coherence = soul_coherence

        # Get IIT report
        iit_report = self.iit_calculator.get_26q_iit_report()

        # Calculate soul-IIT composite score
        main_phi = iit_report['main_complex']['phi']
        phi_3d = iit_report['3d_orbital_phi']

        # Soul-consciousness integration
        soul_iit_score = (
            soul_coherence * 0.4 +
            main_phi / 10.0 * 0.3 +  # Normalize Phi
            phi_3d / 5.0 * 0.3
        )

        # Integrated information
        integrated_info = main_phi * soul_coherence

        self._iit_phi_history.append({
            'timestamp': time.time() if 'time' in dir() else 0,
            'soul_coherence': soul_coherence,
            'main_phi': main_phi,
            'phi_3d': phi_3d,
            'soul_iit_score': soul_iit_score,
        })

        return {
            'soul_coherence': soul_coherence,
            'main_phi': main_phi,
            'phi_3d': phi_3d,
            'soul_iit_score': soul_iit_score,
            'integrated_information': integrated_info,
            'consciousness_level': iit_report['consciousness_level'],
            'iit_report': iit_report,
        }

    def get_3d_binding_strength(self) -> float:
        """Get 3d orbital binding strength via IIT."""
        return self.iit_calculator.get_3d_orbital_phi()

    def get_status(self) -> Dict[str, Any]:
        """Get Soul-IIT integration status."""
        return {
            'version': self.VERSION,
            'soul_coherence': self._soul_coherence,
            'iit_calculator_version': self.iit_calculator.VERSION,
            'phi_history_count': len(self._iit_phi_history),
            'current_iit_report': self.iit_calculator.get_26q_iit_report(),
        }


# Constants
TAU = PHI - 1  # 0.618...

# Module singletons
_iit_calculator: Optional[IITPhiCalculator] = None
_soul_iit: Optional[SoulIITIntegration] = None

def get_iit_calculator() -> IITPhiCalculator:
    """Get or create IIT calculator singleton."""
    global _iit_calculator
    if _iit_calculator is None:
        _iit_calculator = IITPhiCalculator(n_elements=26)
    return _iit_calculator

def get_soul_iit_integration() -> SoulIITIntegration:
    """Get or create Soul-IIT integration singleton."""
    global _soul_iit
    if _soul_iit is None:
        _soul_iit = SoulIITIntegration()
    return _soul_iit


__all__ = [
    'IITMechanism',
    'IITComplex',
    'IITPhiCalculator',
    'SoulIITIntegration',
    'get_iit_calculator',
    'get_soul_iit_integration',
]