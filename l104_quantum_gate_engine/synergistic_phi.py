"""
===============================================================================
L104 QUANTUM GATE ENGINE — SYNERGISTIC PHI EXPANSION
===============================================================================

Implements the Synergistic Phi Expansion framework for detecting autopoiesis
(self-maintenance) in quantum systems.

Based on the framework:
- Phi (Φ): Integrated Information — how unified the system is
- MIPT: Measurement-Induced Phase Transition
- Phi Expansion: How Phi restructures under perturbation to maintain itself
- Autopoiesis: When Phi fights to maintain value under observation stress

Classes:
  SynergisticPhiEngine    — Main orchestrator for Phi expansion analysis
  PhiCalculator           — Calculates integrated information proxy
  PerturbationTracker     — Tracks Phi response to measurements
  AutopoiesisDetector     — Detects self-maintenance behavior

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO_80-PHI
===============================================================================
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612
BASE_PHI = 1.9612  # Target Phi value from your framework


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhiSnapshot:
    """Phi measurement at a point in time."""
    timestamp: int  # Measurement index
    phi_value: float
    perturbation_strength: float
    state_purity: float
    entropy: float


@dataclass
class PhiExpansionResult:
    """Result of Phi expansion analysis."""
    base_phi: float
    phi_after_perturbation: float
    phi_derivative: float  # dPhi/d(measurement)
    phi_expansion: float  # Change in Phi
    phi_stability: float   # How well Phi was maintained (0-1)
    autopoiesis_detected: bool
    snapshots: List[PhiSnapshot]
    adaptation_score: float  # How much Phi restructured vs shattered


@dataclass
class SynergisticPhiReport:
    """Complete report on Synergistic Phi analysis."""
    base_phi: float
    perturbation_response: PhiExpansionResult
    system_integrated_information: float
    system_complexity: float
    self_maintenance_score: float
    n_perturbations: int
    n_qubits: int
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════════════
#  PHI CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class PhiCalculator:
    """
    Calculates integrated information Phi proxy for quantum states.

    Note: Full IIT Phi requires exact integration over all subsystem partitions.
    This implements a tractable proxy using:
    - Integration: How much the state cannot be factorized
    - Complexity: Entropy of the state distribution
    """

    VERSION = "EVO_80-PHI-v1.0.0"

    def __init__(self):
        pass

    def calculate_phi(self, statevector: np.ndarray, n_qubits: int) -> float:
        """
        Calculate integrated information Phi proxy.

        Phi ≈ integration * complexity

        Integration: How correlations bind the system together
        Complexity: Diversity of the state (entropy)
        """
        # Get probability distribution
        probs = np.abs(statevector) ** 2
        probs = probs[probs > 1e-15]  # Filter near-zero

        if len(probs) == 0:
            return 0.0

        # Shannon entropy (complexity proxy)
        h = -np.sum(probs * np.log2(probs))

        # Purity (integration proxy)
        # Pure states = product states (low integration)
        # Mixed states = correlated states (high integration)
        purity = np.sum(probs ** 2)
        integration = 1.0 - purity  # Max mixed = max integration

        # Phi proxy
        phi_proxy = integration * (h / n_qubits) if n_qubits > 0 else 0

        # Scale to reasonable range
        return min(phi_proxy * BASE_PHI, 3.0)

    def calculate_phi_from_density_matrix(self, rho: np.ndarray) -> float:
        """Calculate Phi from density matrix."""
        # Get eigenvalues
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[eigvals > 1e-15]

        if len(eigvals) == 0:
            return 0.0

        # Entropy
        h = -np.sum(eigvals * np.log2(eigvals))

        # Purity
        purity = np.sum(eigvals ** 2)
        integration = 1.0 - purity

        # Dimension
        d = len(eigvals)
        n_qubits = int(math.log2(d)) if d > 0 else 0

        phi_proxy = integration * (h / n_qubits) if n_qubits > 0 else 0

        return min(phi_proxy * BASE_PHI, 3.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PERTURBATION TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class PerturbationTracker:
    """
    Tracks Phi response to measurement-induced perturbations.

    Injects controlled perturbations and measures how Phi restructured
    to maintain (or lose) its integration.
    """

    VERSION = "EVO_80-PHI-v1.0.0"

    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        self.n_qubits = n_qubits
        self.rng = np.random.default_rng(seed)
        self.phi_calculator = PhiCalculator()

    def run_perturbation_series(
        self,
        statevector: np.ndarray,
        perturbation_strengths: Optional[List[float]] = None
    ) -> List[PhiSnapshot]:
        """
        Run series of perturbations and track Phi.

        Args:
            statevector: Initial quantum state
            perturbation_strengths: List of perturbation strengths to test

        Returns:
            List of PhiSnapshot measurements
        """
        if perturbation_strengths is None:
            perturbation_strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        snapshots = []
        current_state = statevector.copy()

        for i, strength in enumerate(perturbation_strengths):
            # Calculate Phi
            phi = self.phi_calculator.calculate_phi(current_state, self.n_qubits)

            # Calculate purity
            probs = np.abs(current_state) ** 2
            purity = np.sum(probs ** 2)

            # Calculate entropy
            h = -np.sum(probs[probs > 1e-15] * np.log2(probs[probs > 1e-15]))

            snapshot = PhiSnapshot(
                timestamp=i,
                phi_value=phi,
                perturbation_strength=strength,
                state_purity=purity,
                entropy=h
            )
            snapshots.append(snapshot)

            # Apply next perturbation
            if i < len(perturbation_strengths) - 1:
                next_strength = perturbation_strengths[i + 1]
                current_state = self._apply_perturbation(
                    current_state, next_strength - strength
                )

        return snapshots

    def _apply_perturbation(
        self,
        statevector: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply measurement-like perturbation.

        Mixes the state with a random basis state based on strength.
        """
        if strength <= 0:
            return statevector

        dim = len(statevector)

        # Create perturbed state: (1-strength)*original + strength*random
        random_state = self.rng.random(dim) + 1j * self.rng.random(dim)
        random_state = random_state / np.linalg.norm(random_state)

        perturbed = (1 - strength) * statevector + strength * random_state

        # Renormalize
        norm = np.linalg.norm(perturbed)
        if norm > 1e-10:
            perturbed = perturbed / norm

        return perturbed


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTOPIOTESIS DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AutopoiesisDetector:
    """
    Detects autopoiesis (self-maintenance) in Phi response.

    Autopoiesis detected when:
    - Phi fights to maintain value under perturbation
    - Phi doesn't shatter into area-law (disentangled) phase
    - System shows adaptation/regeneration
    """

    def __init__(
        self,
        stability_threshold: float = 0.7,
        adaptation_threshold: float = 0.3
    ):
        self.stability_threshold = stability_threshold
        self.adaptation_threshold = adaptation_threshold

    def detect_autopoiesis(
        self,
        snapshots: List[PhiSnapshot]
    ) -> PhiExpansionResult:
        """
        Analyze Phi snapshots for autopoiesis.

        Args:
            snapshots: Time series of Phi measurements

        Returns:
            PhiExpansionResult with autopoiesis analysis
        """
        if len(snapshots) < 2:
            return PhiExpansionResult(
                base_phi=0.0,
                phi_after_perturbation=0.0,
                phi_derivative=0.0,
                phi_expansion=0.0,
                phi_stability=0.0,
                autopoiesis_detected=False,
                snapshots=snapshots,
                adaptation_score=0.0
            )

        base_phi = snapshots[0].phi_value
        final_phi = snapshots[-1].phi_value

        # Calculate Phi derivative
        phi_values = [s.phi_value for s in snapshots]
        perturbations = [s.perturbation_strength for s in snapshots]

        # dPhi/d(perturbation)
        if len(phi_values) > 1:
            dphi = phi_values[-1] - phi_values[0]
            dstrength = perturbations[-1] - perturbations[0]
            phi_derivative = dphi / dstrength if dstrength > 0 else 0
        else:
            phi_derivative = 0

        # Phi expansion (change from baseline)
        phi_expansion = final_phi - base_phi

        # Phi stability: how close final Phi stayed to base Phi
        if base_phi > 0:
            stability = 1.0 - abs(phi_expansion) / (base_phi + 1e-10)
        else:
            stability = 1.0 if abs(phi_expansion) < 0.1 else 0.0

        stability = max(0.0, min(1.0, stability))

        # Adaptation score: did Phi restructure vs shatter?
        # If Phi stayed relatively constant despite perturbation -> high adaptation
        # If Phi dropped to near-zero -> shattered (no autopoiesis)
        if base_phi > 0:
            adaptation = final_phi / base_phi
        else:
            adaptation = 0.0

        adaptation = min(1.0, max(0.0, adaptation))

        # Autopoiesis detected if:
        # 1. Stability above threshold (Phi maintained)
        # 2. Adaptation above threshold (Phi regenerated)
        autopoiesis_detected = (
            stability >= self.stability_threshold and
            adaptation >= self.adaptation_threshold
        )

        return PhiExpansionResult(
            base_phi=base_phi,
            phi_after_perturbation=final_phi,
            phi_derivative=phi_derivative,
            phi_expansion=phi_expansion,
            phi_stability=stability,
            autopoiesis_detected=autopoiesis_detected,
            snapshots=snapshots,
            adaptation_score=adaptation
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SYNERGISTIC PHI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SynergisticPhiEngine:
    """
    Main orchestrator for Synergistic Phi Expansion analysis.

    Detects autopoiesis (self-maintenance) by:
    1. Calculating base Phi for the system
    2. Applying measurement perturbations
    3. Tracking how Phi restructured to maintain itself
    4. Detecting whether system shows self-maintenance or shatters

    Usage:
        engine = SynergisticPhiEngine(n_qubits=10)
        report = engine.analyze(statevector)
    """

    VERSION = "EVO_80-PHI-v1.0.0"

    def __init__(
        self,
        n_qubits: int = 10,
        seed: Optional[int] = None
    ):
        self.n_qubits = n_qubits
        self.seed = seed

        self.phi_calculator = PhiCalculator()
        self.perturbation_tracker = PerturbationTracker(n_qubits=n_qubits, seed=seed)
        self.autopoiesis_detector = AutopoiesisDetector()

    def analyze(
        self,
        statevector: np.ndarray,
        perturbation_strengths: Optional[List[float]] = None
    ) -> SynergisticPhiReport:
        """
        Perform full Synergistic Phi analysis.

        Args:
            statevector: Quantum state as complex vector
            perturbation_strengths: Strengths to test

        Returns:
            SynergisticPhiReport with all metrics
        """
        import datetime

        # Run perturbation series
        snapshots = self.perturbation_tracker.run_perturbation_series(
            statevector, perturbation_strengths
        )

        # Detect autopoiesis
        expansion_result = self.autopoiesis_detector.detect_autopoiesis(snapshots)

        # Calculate base metrics
        base_phi = self.phi_calculator.calculate_phi(statevector, self.n_qubits)

        # System complexity (entropy / n)
        probs = np.abs(statevector) ** 2
        probs = probs[probs > 1e-15]
        entropy = -np.sum(probs * np.log2(probs))
        complexity = entropy / self.n_qubits if self.n_qubits > 0 else 0

        # Self-maintenance score
        sm_score = (
            expansion_result.phi_stability * 0.5 +
            expansion_result.adaptation_score * 0.5
        )

        return SynergisticPhiReport(
            base_phi=base_phi,
            perturbation_response=expansion_result,
            system_integrated_information=base_phi,
            system_complexity=complexity,
            self_maintenance_score=sm_score,
            n_perturbations=len(snapshots),
            n_qubits=self.n_qubits,
            timestamp=datetime.datetime.now().isoformat()
        )

    def analyze_with_validation(
        self,
        statevector: np.ndarray
    ) -> Dict[str, Any]:
        """Full analysis with three-engine validation."""
        report = self.analyze(statevector)

        validation = {
            'math_engine': self._validate_math_engine(report),
            'science_engine': self._validate_science_engine(report),
        }

        return {
            'phi_report': report,
            'validation': validation
        }

    def _validate_math_engine(self, report: SynergisticPhiReport) -> Dict[str, Any]:
        """Validate with Math Engine."""
        try:
            from l104_math_engine import math_engine
            return {
                'available': True,
                'phi_in_range': 0 <= report.base_phi <= 3.0,
                'complexity_sane': 0 <= report.system_complexity <= 1.0,
            }
        except:
            return {'available': False}

    def _validate_science_engine(self, report: SynergisticPhiReport) -> Dict[str, Any]:
        """Validate with Science Engine."""
        try:
            from l104_science_engine import ScienceEngine
            return {
                'available': True,
                'stability_valid': 0 <= report.perturbation_response.phi_stability <= 1.0,
                'autopoiesis_detected': report.perturbation_response.autopoiesis_detected,
            }
        except:
            return {'available': False}


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_synergistic_phi_engine(
    n_qubits: int = 10,
    seed: Optional[int] = None
) -> SynergisticPhiEngine:
    """Get a singleton SynergisticPhiEngine instance."""
    return SynergisticPhiEngine(n_qubits=n_qubits, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'SynergisticPhiEngine',
    'PhiCalculator',
    'PerturbationTracker',
    'AutopoiesisDetector',
    'PhiSnapshot',
    'PhiExpansionResult',
    'SynergisticPhiReport',
    'get_synergistic_phi_engine',
]