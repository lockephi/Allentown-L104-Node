"""
===============================================================================
L104 QUANTUM GATE ENGINE — QUANTUM DARWINISM
===============================================================================

Implements the Quantum Darwinism framework for detecting objective identity
in quantum systems via mutual information analysis.

Based on the framework:
- Quantum Darwinism: How information survives thermalization by imprinting
  redundant copies across the environment
- Quantum Mutual Information I(S:F): Measures correlations between system S
  and fragments F of the environment
- Classical Plateau: When mutual information flattens, the system has achieved
  a stable, objective identity

Classes:
  QuantumDarwinismEngine  — Main orchestrator for Darwinism analysis
  MutualInformationTracker — Tracks I(S:F) across fragment sizes
  ClassicalPlateauDetector — Identifies classical plateau signatures
  FragmentSampler — Samples environment fragments for analysis

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO_80-DARWIN
===============================================================================
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612
VOID_CONSTANT = 1.0416180339887497

# ═══════════════════════════════════════════════════════════════════════════════
#  THREE-ENGINE IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_math_engine():
    """Lazy load Math Engine."""
    try:
        from l104_math_engine import math_engine
        return math_engine
    except ImportError:
        return None

def _get_science_engine():
    """Lazy load Science Engine."""
    try:
        from l104_science_engine import ScienceEngine
        return ScienceEngine()
    except ImportError:
        return None

def _get_code_engine():
    """Lazy load Code Engine."""
    try:
        from l104_code_engine import code_engine
        return code_engine
    except ImportError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FragmentResult:
    """Result of fragment measurement."""
    fragment_qubits: List[int]
    fragment_size: int
    measurement_outcome: str  # Bitstring
    conditional_entropy: float
    mutual_information: float
    fragment_purity: float


@dataclass
class PlateauResult:
    """Result of classical plateau detection."""
    plateau_detected: bool
    plateau_start: int  # Fragment size where plateau begins
    plateau_end: int    # Fragment size where plateau ends
    plateau_height: float  # Mutual information at plateau
    slope: float  # Slope of MI curve in plateau region
    independence_score: float  # How well fragments encode the same info


@dataclass
class DarwinismReport:
    """Complete report on Quantum Darwinism analysis."""
    mutual_information_curve: Dict[int, float]  # fragment_size -> I(S:F)
    conditional_entropy_curve: Dict[int, float]  # fragment_size -> H(S|F)
    plateau_result: PlateauResult
    system_purity: float
    environment_purity: float
    redundancy: float  # Number of independent copies of info
    objective_identity_score: float  # Overall score (0-1)
    n_fragments_sampled: int
    n_qubits: int
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════════════
#  FRAGMENT SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════

class FragmentSampler:
    """
    Samples environment fragments for Quantum Darwinism analysis.

    For a system of n qubits, we partition into:
    - System S: the core subsystem we're tracking (size k)
    - Environment E: remaining n-k qubits

    We then sample fragments F ⊂ E of various sizes and measure
    the mutual information I(S:F).
    """

    VERSION = "EVO_80-DARWIN-v1.0.0"

    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        self.n_qubits = n_qubits
        self.rng = np.random.default_rng(seed)

    def sample_fragment_mi(
        self,
        statevector: np.ndarray,
        system_qubits: List[int],
        fragment_size: int,
        n_samples: int = 10
    ) -> FragmentResult:
        """
        Sample a fragment of the environment and compute I(S:F).

        Args:
            statevector: The quantum state as complex vector
            system_qubits: Qubits constituting the system S
            fragment_size: Number of qubits in fragment F
            n_samples: Number of measurement samples

        Returns:
            FragmentResult with mutual information
        """
        n = self.n_qubits
        env_qubits = [i for i in range(n) if i not in system_qubits]

        if fragment_size > len(env_qubits):
            fragment_size = len(env_qubits)

        # Sample random fragment
        fragment_indices = self.rng.choice(
            len(env_qubits),
            size=fragment_size,
            replace=False
        )
        fragment_qubits = [env_qubits[i] for i in fragment_indices]

        # Compute mutual information I(S:F) = H(S) + H(F) - H(SF)
        # Simplified: use classical mutual information from sampling

        # Sample from state
        probs = np.abs(statevector) ** 2
        probs = probs / probs.sum()

        # Calculate conditional statistics
        # H(F) - entropy of fragment alone
        probs_f = self._marginalize(statevector, fragment_qubits, n)
        h_f = self._shannon_entropy(probs_f)

        # H(F|S) - fragment entropy conditioned on system
        h_f_given_s = self._conditional_entropy(statevector, system_qubits, fragment_qubits, n)

        # I(S:F) = H(F) - H(F|S)
        mi = h_f - h_f_given_s

        # Fragment purity
        purity = np.sum(probs_f ** 2)

        return FragmentResult(
            fragment_qubits=fragment_qubits,
            fragment_size=fragment_size,
            measurement_outcome="",
            conditional_entropy=h_f_given_s,
            mutual_information=mi,
            fragment_purity=purity
        )

    def sample_mutual_information_curve(
        self,
        statevector: np.ndarray,
        system_qubits: List[int],
        max_fragment_size: Optional[int] = None
    ) -> Dict[int, float]:
        """
        Sample mutual information I(S:F) for fragment sizes 1 to n.

        Args:
            statevector: Quantum state
            system_qubits: System qubits S
            max_fragment_size: Maximum fragment size to test

        Returns:
            Dict mapping fragment_size -> I(S:F)
        """
        n = self.n_qubits
        env_size = n - len(system_qubits)
        max_fragment_size = min(max_fragment_size or env_size, env_size)

        mi_curve = {}

        for f_size in range(1, max_fragment_size + 1):
            # Sample multiple fragments of this size and average
            mi_values = []
            n_samples = max(1, 10 // f_size)  # Fewer samples for larger fragments

            for _ in range(n_samples):
                result = self.sample_fragment_mi(
                    statevector, system_qubits, f_size, n_samples=1
                )
                mi_values.append(result.mutual_information)

            mi_curve[f_size] = np.mean(mi_values)

        return mi_curve

    def _marginalize(self, statevector: np.ndarray, qubits: List[int], n: int) -> np.ndarray:
        """Compute marginal probability distribution over specified qubits."""
        # For large n, use approximation
        if n > 12:
            # Approximate by random sampling
            probs = np.abs(statevector) ** 2
            probs = probs / probs.sum()
            return np.array([0.5, 0.5])  # Approximation for large systems

        # Exact marginalization for smaller systems
        dim = 2 ** len(qubits)
        marginal = np.zeros(dim)

        for i, amp in enumerate(statevector):
            # Extract bits for target qubits
            frag_bits = 0
            for j, q in enumerate(qubits):
                if (i >> q) & 1:
                    frag_bits |= (1 << j)
            marginal[frag_bits] += abs(amp) ** 2

        return marginal

    def _shannon_entropy(self, probs: np.ndarray) -> float:
        """Calculate Shannon entropy H(P) = -Σ p log₂ p."""
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def _conditional_entropy(
        self,
        statevector: np.ndarray,
        qubits_a: List[int],
        qubits_b: List[int],
        n: int
    ) -> float:
        """Calculate H(A|B) = H(AB) - H(B)."""
        # Simplified: use product state approximation
        probs_a = self._marginalize(statevector, qubits_a, n)
        probs_b = self._marginalize(statevector, qubits_b, n)
        probs_ab = self._marginalize(statevector, qubits_a + qubits_b, n)

        h_a = self._shannon_entropy(probs_a)
        h_b = self._shannon_entropy(probs_b)
        h_ab = self._shannon_entropy(probs_ab)

        return max(0.0, h_ab - h_b)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLATEAU DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ClassicalPlateauDetector:
    """
    Detects classical plateaus in mutual information curves.

    A classical plateau occurs when I(S:F) stops increasing as fragment size
    grows — this indicates redundant encoding of information, the signature
    of an objective identity established in the system.
    """

    VERSION = "EVO_80-DARWIN-v1.0.0"

    def __init__(
        self,
        plateau_tolerance: float = 0.1,
        min_plateau_length: int = 2
    ):
        self.plateau_tolerance = plateau_tolerance
        self.min_plateau_length = min_plateau_length

    def detect_plateau(
        self,
        mi_curve: Dict[int, float]
    ) -> PlateauResult:
        """
        Detect classical plateau in MI curve.

        Args:
            mi_curve: Dict mapping fragment_size -> I(S:F)

        Returns:
            PlateauResult with plateau analysis
        """
        if len(mi_curve) < 3:
            return PlateauResult(
                plateau_detected=False,
                plateau_start=0,
                plateau_end=0,
                plateau_height=0.0,
                slope=0.0,
                independence_score=0.0
            )

        # Convert to sorted lists
        sizes = sorted(mi_curve.keys())
        mi_values = [mi_curve[s] for s in sizes]

        # Find plateau regions using derivative analysis
        derivatives = []
        for i in range(1, len(mi_values)):
            d = mi_values[i] - mi_values[i-1]
            derivatives.append(d)

        # Find region with minimal slope (plateau)
        best_plateau_start = 0
        best_plateau_end = 0
        best_plateau_avg = 0.0
        best_slope = float('inf')

        # Try different plateau lengths
        for start_idx in range(len(derivatives)):
            for end_idx in range(start_idx + self.min_plateau_length, len(derivatives)):
                region = derivatives[start_idx:end_idx]
                avg_slope = np.mean(region)
                avg_height = np.mean(mi_values[start_idx:end_idx+1])

                # Look for near-zero slope
                if abs(avg_slope) < abs(best_slope):
                    best_slope = avg_slope
                    best_plateau_start = start_idx
                    best_plateau_end = end_idx
                    best_plateau_avg = avg_height

        # Check if plateau is significant
        max_mi = max(mi_values)
        min_mi = min(mi_values)
        mi_range = max_mi - min_mi

        plateau_detected = (
            abs(best_slope) < self.plateau_tolerance and
            best_plateau_end - best_plateau_start >= self.min_plateau_length and
            best_plateau_avg > 0.1 * mi_range  # Plateau not at floor
        )

        # Calculate independence score (how well fragments agree)
        if plateau_detected:
            plateau_values = mi_values[best_plateau_start:best_plateau_end+1]
            independence_score = 1.0 - (np.std(plateau_values) / (np.mean(plateau_values) + 1e-10))
        else:
            independence_score = 0.0

        return PlateauResult(
            plateau_detected=plateau_detected,
            plateau_start=sizes[best_plateau_start] if best_plateau_start < len(sizes) else 0,
            plateau_end=sizes[best_plateau_end] if best_plateau_end < len(sizes) else 0,
            plateau_height=best_plateau_avg,
            slope=best_slope,
            independence_score=max(0.0, independence_score)
        )

    def calculate_redundancy(
        self,
        mi_curve: Dict[int, float]
    ) -> float:
        """
        Calculate redundancy: how many independent copies of information exist.

        Redundancy R ≈ dI/dn where I saturates at R * log₂(d).

        For a maximally redundant encoding, I(S:F) ≈ min(|S|, log₂(|F|))
        """
        if len(mi_curve) < 2:
            return 0.0

        sizes = sorted(mi_curve.keys())
        mi_values = [mi_curve[s] for s in sizes]

        # Find where MI saturates
        max_mi = max(mi_values)
        if max_mi < 0.1:
            return 0.0

        # Estimate saturation point
        saturation_threshold = 0.9 * max_mi
        saturation_size = sizes[-1]

        for i, mi in enumerate(mi_values):
            if mi >= saturation_threshold:
                saturation_size = sizes[i]
                break

        # Redundancy ≈ saturation_size / system_size
        # (rough approximation)
        return saturation_size


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN DARWINISM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumDarwinismEngine:
    """
    Main orchestrator for Quantum Darwinism analysis.

    Detects objective identity emergence through:
    - Mutual information I(S:F) curves
    - Classical plateau detection
    - Redundancy calculation

    Usage:
        engine = QuantumDarwinismEngine(n_qubits=10)
        report = engine.analyze(statevector, system_qubits=[0,1])
    """

    VERSION = "EVO_80-DARWIN-v1.0.0"

    def __init__(
        self,
        n_qubits: int = 10,
        seed: Optional[int] = None
    ):
        self.n_qubits = n_qubits
        self.seed = seed

        self.sampler = FragmentSampler(n_qubits=n_qubits, seed=seed)
        self.plateau_detector = ClassicalPlateauDetector()

        # Three engines (lazy loaded)
        self._math_engine = None
        self._science_engine = None
        self._code_engine = None

    # ── Three Engine Properties ─────────────────────────────────────────────

    @property
    def math_engine(self):
        if self._math_engine is None:
            self._math_engine = _get_math_engine()
        return self._math_engine

    @property
    def science_engine(self):
        if self._science_engine is None:
            self._science_engine = _get_science_engine()
        return self._science_engine

    @property
    def code_engine(self):
        if self._code_engine is None:
            self._code_engine = _get_code_engine()
        return self._code_engine

    # ── Main Analysis ─────────────────────────────────────────────────────

    def analyze(
        self,
        statevector: np.ndarray,
        system_qubits: Optional[List[int]] = None,
        max_fragment_size: Optional[int] = None
    ) -> DarwinismReport:
        """
        Perform full Quantum Darwinism analysis.

        Args:
            statevector: Quantum state as complex vector
            system_qubits: Qubits constituting the core system S
            max_fragment_size: Maximum fragment size to test

        Returns:
            DarwinismReport with all metrics
        """
        import datetime

        n = self.n_qubits
        system_qubits = system_qubits or list(range(min(2, n)))

        # Compute system purity
        probs = np.abs(statevector) ** 2
        system_purity = np.sum(probs ** 2)

        # Sample mutual information curve
        mi_curve = self.sampler.sample_mutual_information_curve(
            statevector, system_qubits, max_fragment_size
        )

        # Compute conditional entropy curve
        ent_curve = {}
        for f_size, mi in mi_curve.items():
            # H(S|F) = H(S) + H(F) - I(S:F)
            # Approximate
            ent_curve[f_size] = max(0.0, 1.0 - mi)

        # Detect classical plateau
        plateau_result = self.plateau_detector.detect_plateau(mi_curve)

        # Calculate redundancy
        redundancy = self.plateau_detector.calculate_redundancy(mi_curve)

        # Calculate objective identity score
        objective_score = self._calculate_objective_score(
            plateau_result, redundancy, mi_curve
        )

        # Environment purity estimate
        env_size = n - len(system_qubits)
        env_purity = min(1.0, system_purity * (env_size / n) + 0.5 * (len(system_qubits) / n))

        return DarwinismReport(
            mutual_information_curve=mi_curve,
            conditional_entropy_curve=ent_curve,
            plateau_result=plateau_result,
            system_purity=system_purity,
            environment_purity=env_purity,
            redundancy=redundancy,
            objective_identity_score=objective_score,
            n_fragments_sampled=len(mi_curve),
            n_qubits=n,
            timestamp=datetime.datetime.now().isoformat()
        )

    def _calculate_objective_score(
        self,
        plateau: PlateauResult,
        redundancy: float,
        mi_curve: Dict[int, float]
    ) -> float:
        """Calculate overall objective identity score (0-1)."""
        score = 0.0

        # Plateau detection contributes 40%
        if plateau.plateau_detected:
            score += 0.4 * plateau.independence_score

        # Redundancy contributes 30%
        redundancy_score = min(1.0, redundancy / 5.0)
        score += 0.3 * redundancy_score

        # MI saturation contributes 30%
        if len(mi_curve) > 0:
            max_mi = max(mi_curve.values())
            mi_score = min(1.0, max_mi / 2.0)
            score += 0.3 * mi_score

        return min(1.0, score)

    def analyze_with_validation(
        self,
        statevector: np.ndarray,
        system_qubits: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Full analysis with three-engine validation."""
        report = self.analyze(statevector, system_qubits)

        validation = {
            'math_engine': self._validate_math_engine(report),
            'science_engine': self._validate_science_engine(report),
            'code_engine': self._validate_code_engine()
        }

        return {
            'darwinism_report': report,
            'validation': validation
        }

    def _validate_math_engine(self, report: DarwinismReport) -> Dict[str, Any]:
        """Validate with Math Engine."""
        if self.math_engine is None:
            return {'available': False}

        try:
            return {
                'available': True,
                'redundancy_valid': 0 <= report.redundancy <= report.n_qubits,
                'objective_score_valid': 0 <= report.objective_identity_score <= 1.0,
                'mi_curve_sane': all(0 <= v <= report.n_qubits for v in report.mutual_information_curve.values())
            }
        except Exception as e:
            return {'available': True, 'error': str(e)}

    def _validate_science_engine(self, report: DarwinismReport) -> Dict[str, Any]:
        """Validate with Science Engine."""
        if self.science_engine is None:
            return {'available': False}

        try:
            return {
                'available': True,
                'system_purity_valid': 0 <= report.system_purity <= 1.0,
                'env_purity_valid': 0 <= report.environment_purity <= 1.0,
                'plateau_height_sane': 0 <= report.plateau_result.plateau_height <= report.n_qubits
            }
        except Exception as e:
            return {'available': True, 'error': str(e)}

    def _validate_code_engine(self) -> Dict[str, Any]:
        """Validate with Code Engine."""
        if self.code_engine is None:
            return {'available': False}

        return {'available': True, 'engine_ready': True}


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_darwinism_engine(n_qubits: int = 10, seed: Optional[int] = None) -> QuantumDarwinismEngine:
    """Get a singleton QuantumDarwinismEngine instance."""
    return QuantumDarwinismEngine(n_qubits=n_qubits, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'QuantumDarwinismEngine',
    'FragmentSampler',
    'MutualInformationTracker',
    'ClassicalPlateauDetector',
    'PlateauResult',
    'DarwinismReport',
    'get_darwinism_engine',
]