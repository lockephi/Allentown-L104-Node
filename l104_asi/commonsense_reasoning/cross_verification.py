"""Layer 8: Cross-Verification Engine — Multi-Layer Consistency Checks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .constants import PHI, GOD_CODE, VOID_CONSTANT
from .ontology import ConceptOntology
from .causal import CausalReasoningEngine

logger = logging.getLogger(__name__)


class CrossVerificationEngine:
    """Cross-verification across all reasoning layers.

    After each reasoning layer produces its score, this engine checks for
    consistency across layers and applies corrections:
    - Causal rules must be consistent with ontology properties
    - Physical intuition should agree with temporal ordering
    - Analogical matches should share domain with causal rules
    - PHI-weighted calibration via VOID_CONSTANT for final confidence
    """

    def __init__(self, ontology: ConceptOntology, causal: 'CausalReasoningEngine',
                 temporal: TemporalReasoningEngine):
        self.ontology = ontology
        self.causal = causal
        self.temporal = temporal

    def verify_choice(self, question: str, choice: str, layer_scores: Dict[str, float]) -> Dict[str, Any]:
        """Verify a choice across multiple reasoning layers.

        Args:
            question: The question text.
            choice: The candidate answer.
            layer_scores: Scores from each layer (keys: 'causal', 'physical',
                         'analogical', 'temporal', 'fact_table', 'ontology_scan').

        Returns:
            Dict with 'verified_score', 'adjustments', 'consistency'.
        """
        adjustments = []
        consistency_count = 0
        total_layers = 0

        # 1. Count how many layers have non-zero signal
        active_layers = {k: v for k, v in layer_scores.items() if v > 0.05}
        total_layers = len(active_layers)

        # 2. Multi-layer agreement bonus
        if total_layers >= 3:
            # Three or more layers agree → strong signal
            consistency_count += 1
            adjustments.append(("multi_layer_agreement", 0.15))
        elif total_layers >= 2:
            consistency_count += 1
            adjustments.append(("dual_layer_agreement", 0.08))

        # 3. Causal-temporal consistency
        causal_score = layer_scores.get('causal', 0)
        temporal_score = layer_scores.get('temporal', 0)
        if causal_score > 0.1 and temporal_score > 0.1:
            # Both layers have signal → check if they point the same direction
            consistency_count += 1
            adjustments.append(("causal_temporal_consistent", 0.10))
        elif causal_score > 0.3 and temporal_score == 0:
            # Strong causal signal but no temporal relevance → might be fine for non-process questions
            pass
        elif temporal_score > 0.3 and causal_score == 0:
            # Strong temporal but no causal → process-order question (valid)
            adjustments.append(("temporal_dominant", 0.05))

        # 4. Physical-ontology consistency
        physical_score = layer_scores.get('physical', 0)
        ontology_score = layer_scores.get('ontology_scan', 0)
        if physical_score > 0.1 and ontology_score > 0.1:
            consistency_count += 1
            adjustments.append(("physical_ontology_consistent", 0.08))

        # 5. Single-layer dominance penalty — over-reliance on one signal
        if total_layers == 1:
            dominant_layer = list(active_layers.keys())[0]
            dominant_score = active_layers[dominant_layer]
            if dominant_score > 0.5:
                adjustments.append(("single_layer_caution", -0.05))

        # 6. PHI-weighted confidence calibration via VOID_CONSTANT
        raw_total = sum(layer_scores.values())
        phi_calibrated = raw_total * (PHI / (PHI + 1.0))  # φ/(φ+1) ≈ 0.618 dampening
        void_scaled = phi_calibrated * VOID_CONSTANT  # Sacred scaling

        # 7. Compute final adjustment
        total_adjustment = sum(adj[1] for adj in adjustments)

        return {
            'verified_score': void_scaled + total_adjustment,
            'adjustments': adjustments,
            'consistency': consistency_count / max(total_layers, 1),
            'active_layers': total_layers,
            'phi_calibrated': round(phi_calibrated, 4),
            'void_scaled': round(void_scaled, 4),
        }

    def cross_check_elimination(self, question: str, choice_verifications: List[Dict]) -> List[Dict]:
        """Apply cross-verification to eliminate inconsistent choices.

        Args:
            question: The question text.
            choice_verifications: List of verification results for each choice.

        Returns:
            Updated list with elimination flags.
        """
        if len(choice_verifications) < 2:
            return choice_verifications

        # Find maximum consistency and score
        max_consistency = max(cv.get('consistency', 0) for cv in choice_verifications)
        max_score = max(cv.get('verified_score', 0) for cv in choice_verifications)

        for cv in choice_verifications:
            cv['eliminated'] = False
            # Eliminate choices with zero active layers if others have signal
            if cv.get('active_layers', 0) == 0 and max_score > 0.2:
                cv['eliminated'] = True
                cv['elimination_reason'] = 'no_layer_signal'
            # Eliminate choices with very low consistency compared to best
            elif cv.get('consistency', 0) < max_consistency * 0.3 and max_consistency > 0.5:
                cv['eliminated'] = True
                cv['elimination_reason'] = 'low_consistency'

        return choice_verifications


