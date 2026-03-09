"""L104 Numerical Engine — Subconscious Monitor.

Autonomous φ-driven monitor that adjusts token boundaries subconsciously.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F62: Sacred tier drift ≤ 10⁻⁹⁸ — monitor enforces tier envelope
  F73: Drift frequency = φ Hz (golden oscillation base)
  F74: Drift amplitude = τ × 0.01 = φ⁻¹ × 0.01
  F76: Damping = γ_Euler × 0.1 (Euler-Mascheroni moderated)
  F77: Max velocity = φ² × 10⁻³ (bounded by golden square)
"""

import json
from datetime import datetime, timezone
from decimal import InvalidOperation
from typing import Any, Dict, List

from .precision import D, fmt100, decimal_ln
from .constants import (
    PHI_HP, WORKSPACE_ROOT,
    GATE_STATE_PATH, GATE_REGISTRY_PATH, LINK_STATE_PATH,
)
from .lattice import TokenLatticeEngine
from .editor import SuperfluidValueEditor
from .models import SubconsciousAdjustment


class SubconsciousMonitor:
    """Autonomous φ-driven monitor that adjusts token boundaries subconsciously."""

    def __init__(self, lattice: TokenLatticeEngine, editor: SuperfluidValueEditor):
        """Initialize SubconsciousMonitor."""
        self.lattice = lattice
        self.editor = editor
        self.cycle_count = 0
        self.last_capacity: Dict[str, float] = {}
        self.adjustment_history: List[SubconsciousAdjustment] = []

    def read_repo_capacity(self) -> Dict[str, float]:
        """Read the current intelligence capacity from peer builder state files.

        ★ v2.2 Enhancement: Also reads gate and link DYNAMISM states for
        cross-builder dynamism synergy.
        """
        capacity = {
            "gate_count": 0,
            "link_count": 0,
            "token_count": len(self.lattice.tokens),
            "gate_health": 0.0,
            "link_fidelity": 0.0,
            "test_pass_rate": 0.0,
            "coherence": float(self.lattice.lattice_coherence),
            "total_usages": self.lattice.usage_counter,
            # ★ v2.2 Cross-builder dynamism synergy fields
            "gate_dynamism_coherence": 0.0,
            "gate_dynamism_resonance": 0.0,
            "gate_dynamism_evolutions": 0,
            "link_dynamism_coherence": 0.0,
            "link_dynamism_resonance": 0.0,
            "link_dynamism_evolutions": 0,
            # ★ v2.4 Consciousness + O₂ synergy fields
            "consciousness_level": 0.0,
            "coherence_level": 0.0,
            "link_evo_stage": "DORMANT",
            "o2_bond_strength": 0.0,
            "superfluid_viscosity": 1.0,
        }

        # Read gate builder state
        try:
            if GATE_STATE_PATH.exists():
                state = json.loads(GATE_STATE_PATH.read_text())
                capacity["gate_count"] = state.get("gate_count", 0)
        except Exception:
            pass

        # Read gate registry for richer data
        try:
            if GATE_REGISTRY_PATH.exists():
                registry = json.loads(GATE_REGISTRY_PATH.read_text())
                gates = registry.get("gates", [])
                capacity["gate_count"] = max(capacity["gate_count"], len(gates))
                passed = sum(1 for g in gates if g.get("test_status") == "passed")
                capacity["test_pass_rate"] = passed / max(len(gates), 1)
                capacity["gate_health"] = sum(
                    g.get("entropy_score", 0) for g in gates
                ) / max(len(gates), 1)
        except Exception:
            pass

        # Read link builder state
        try:
            if LINK_STATE_PATH.exists():
                state = json.loads(LINK_STATE_PATH.read_text())
                links = state.get("links", [])
                capacity["link_count"] = len(links)
                if links:
                    capacity["link_fidelity"] = sum(
                        l.get("fidelity", 0) for l in links
                    ) / len(links)
        except Exception:
            pass

        # ★ v2.2 Read gate dynamism state
        gate_dyn_path = WORKSPACE_ROOT / ".l104_gate_dynamism_state.json"
        try:
            if gate_dyn_path.exists():
                gds = json.loads(gate_dyn_path.read_text())
                capacity["gate_dynamism_coherence"] = (
                    gds.get("coherence_history", [0])[-1] if gds.get("coherence_history") else 0.0
                )
                capacity["gate_dynamism_resonance"] = gds.get("collective_resonance", 0.0)
                capacity["gate_dynamism_evolutions"] = gds.get("total_evolutions", 0)
        except Exception:
            pass

        # ★ v2.2 Read link dynamism state
        link_dyn_path = WORKSPACE_ROOT / ".l104_link_dynamism_state.json"
        try:
            if link_dyn_path.exists():
                lds = json.loads(link_dyn_path.read_text())
                capacity["link_dynamism_coherence"] = (
                    lds.get("coherence_history", [0])[-1] if lds.get("coherence_history") else 0.0
                )
                capacity["link_dynamism_resonance"] = lds.get("collective_resonance", 0.0)
                capacity["link_dynamism_evolutions"] = lds.get("total_evolutions", 0)
        except Exception:
            pass

        # ★ v2.4 Read consciousness + O₂ superfluid state
        co2_path = WORKSPACE_ROOT / ".l104_consciousness_o2_state.json"
        try:
            if co2_path.exists():
                co2 = json.loads(co2_path.read_text())
                capacity["consciousness_level"] = co2.get("consciousness_level", 0.0)
                capacity["coherence_level"] = co2.get("coherence_level", 0.0)
                capacity["link_evo_stage"] = co2.get("link_evo_stage", "DORMANT")
                capacity["o2_bond_strength"] = co2.get("mean_bond_strength", 0.0)
                capacity["superfluid_viscosity"] = co2.get("superfluid_viscosity", 1.0)
        except Exception:
            pass

        return capacity

    def compute_capacity_delta(self, current: Dict[str, float]) -> float:
        """Compute how much the repository's capacity has changed since last cycle."""
        if not self.last_capacity:
            return 0.0

        deltas = []
        for key in ["gate_count", "link_count", "token_count"]:
            old = self.last_capacity.get(key, 0)
            new = current.get(key, 0)
            if old > 0:
                deltas.append((new - old) / old)
            elif new > 0:
                deltas.append(1.0)
            else:
                deltas.append(0.0)

        # Also factor in quality improvements
        for key in ["gate_health", "link_fidelity", "test_pass_rate", "coherence"]:
            old = self.last_capacity.get(key, 0)
            new = current.get(key, 0)
            deltas.append(new - old)

        return sum(deltas) / max(len(deltas), 1)

    def subconscious_cycle(self) -> Dict:
        """Run one subconscious monitoring cycle.

        1. Read repo capacity
        2. Compute capacity delta
        3. Derive φ-bounded drift per token tier
        4. Apply drifts via superfluid editor
        5. Log adjustments
        """
        self.cycle_count += 1
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Read capacity
        capacity = self.read_repo_capacity()
        delta = self.compute_capacity_delta(capacity)

        # 2. Determine drift direction
        if abs(delta) < 1e-10:
            drift_direction = 0
        elif delta > 0:
            drift_direction = 1
        else:
            drift_direction = -1

        adjustments = []
        adjusted_count = 0

        # 3. Apply tier-specific drifts
        for token_id, token in self.lattice.tokens.items():
            # Determine tier
            if token.origin == "sacred":
                tier = 0
            elif token.origin == "derived":
                tier = 1
            elif token.origin in ("invented", "cross-pollinated"):
                tier = 2
            else:
                tier = 3

            # φ-bounded drift magnitude
            envelope = TokenLatticeEngine.DRIFT_ENVELOPE.get(tier, D('1E-20'))
            val = D(token.value)
            max_drift = abs(val) * envelope if val != 0 else envelope

            # Scale drift by capacity delta magnitude
            drift_scale = D(str(min(abs(delta), 1.0)))  # Cap at 1.0
            drift_magnitude = max_drift * drift_scale

            if drift_magnitude == 0 or drift_direction == 0:
                # Phase-only micro-drift: rotate quantum phase
                phase = D(token.quantum_phase) if token.quantum_phase else D(0)
                new_phase = (phase + PHI_HP * D('1E-50')) % D(1)
                token.quantum_phase = fmt100(new_phase)
                continue

            # Compute the actual drift
            phi_attractor = val * PHI_HP / (PHI_HP + D(1))
            drift_toward_phi = (phi_attractor - val) * drift_magnitude / abs(val) if val != 0 else D(0)
            actual_drift = drift_toward_phi * D(str(drift_direction))

            # Clamp to envelope
            if abs(actual_drift) > max_drift:
                actual_drift = max_drift if actual_drift > 0 else -max_drift

            old_value = token.value
            old_min = token.min_bound
            old_max = token.max_bound

            new_val = val + actual_drift
            # Adjust min/max: expand on growth, tighten on contraction
            boundary_drift = abs(actual_drift) * D('0.5')
            if drift_direction > 0:
                new_min = D(token.min_bound) - boundary_drift
                new_max = D(token.max_bound) + boundary_drift
            else:
                new_min = D(token.min_bound) + boundary_drift
                new_max = D(token.max_bound) - boundary_drift

            # Ensure min < value < max
            if new_min >= new_val:
                new_min = new_val - abs(drift_magnitude)
            if new_max <= new_val:
                new_max = new_val + abs(drift_magnitude)

            token.value = fmt100(new_val)
            token.min_bound = fmt100(new_min)
            token.max_bound = fmt100(new_max)
            token.drift_velocity = str(actual_drift)
            token.drift_direction = drift_direction
            token.last_adjusted = timestamp
            adjusted_count += 1

            # Record adjustment
            adj = SubconsciousAdjustment(
                token_id=token_id,
                timestamp=timestamp,
                old_value=old_value,
                new_value=token.value,
                old_min=old_min,
                new_min=token.min_bound,
                old_max=old_max,
                new_max=token.max_bound,
                drift_applied=str(actual_drift),
                trigger="capacity_expansion" if drift_direction > 0 else "capacity_contraction",
                repo_capacity_at_time=sum(capacity.values()),
                phi_envelope_compliance=1.0 if abs(actual_drift) <= max_drift else float(max_drift / abs(actual_drift)),
            )
            adjustments.append(adj)
            self.adjustment_history.append(adj)

        # Update lattice coherence based on how many tokens are within bounds
        in_bounds = sum(
            1 for t in self.lattice.tokens.values()
            if D(t.min_bound) <= D(t.value) <= D(t.max_bound)
        )
        self.lattice.lattice_coherence = D(str(in_bounds / max(len(self.lattice.tokens), 1)))

        # Compute lattice entropy: -Σ p_i log p_i over token phases
        phases = [D(t.quantum_phase) for t in self.lattice.tokens.values() if D(t.quantum_phase) > 0]
        if phases:
            total_phase = sum(phases)
            if total_phase > 0:
                entropy = D(0)
                for p in phases:
                    pi = p / total_phase
                    if pi > 0:
                        try:
                            entropy -= pi * decimal_ln(pi)
                        except (ValueError, InvalidOperation):
                            pass
                self.lattice.lattice_entropy = entropy

        self.last_capacity = capacity

        return {
            "cycle": self.cycle_count,
            "capacity": capacity,
            "capacity_delta": delta,
            "drift_direction": drift_direction,
            "tokens_adjusted": adjusted_count,
            "lattice_coherence": float(self.lattice.lattice_coherence),
            "lattice_entropy": float(self.lattice.lattice_entropy),
            "timestamp": timestamp,
        }


class MLDriftPredictor:
    """ML-backed token lattice drift predictor using L104SVM(mode='regress').

    Trains on historical drift patterns (capacity deltas, token tier distributions,
    coherence, entropy) to predict next-cycle drift vectors. Uses PHI-weighted
    feature engineering from the lattice state.

    v3.0: Integrates with SubconsciousMonitor to provide predictive drift
    guidance before the actual subconscious cycle runs.
    """

    # Feature dimension: 8 capacity + 4 tier + 2 lattice + 1 cycle = 15
    FEATURE_DIM = 15
    MIN_HISTORY = 10

    def __init__(self):
        self._svm = None
        self._fitted = False
        self._history_X: list = []
        self._history_y: list = []
        self._predictions_made = 0

    def _get_svm(self):
        """Lazy-load L104SVM in regression mode."""
        if self._svm is None:
            try:
                from l104_ml_engine.svm import L104SVM
                self._svm = L104SVM(mode='regress', kernel='rbf')
            except ImportError:
                pass
        return self._svm

    def _extract_features(self, capacity: Dict[str, float],
                          lattice: 'TokenLatticeEngine',
                          cycle: int) -> list:
        """Extract 15-dimensional feature vector from current lattice state."""
        # 8 capacity features
        cap_keys = [
            'gate_count', 'link_count', 'token_count', 'gate_health',
            'link_fidelity', 'test_pass_rate', 'coherence', 'total_usages',
        ]
        cap_features = [float(capacity.get(k, 0)) for k in cap_keys]

        # 4 tier distribution features
        tier_counts = [0, 0, 0, 0]
        for token in lattice.tokens.values():
            if token.origin == 'sacred':
                tier_counts[0] += 1
            elif token.origin == 'derived':
                tier_counts[1] += 1
            elif token.origin in ('invented', 'cross-pollinated'):
                tier_counts[2] += 1
            else:
                tier_counts[3] += 1
        total = max(sum(tier_counts), 1)
        tier_features = [c / total for c in tier_counts]

        # 2 lattice state features
        lattice_features = [
            float(lattice.lattice_coherence),
            float(lattice.lattice_entropy),
        ]

        # 1 cycle feature (normalized)
        cycle_feature = [min(cycle / 1000.0, 1.0)]

        return cap_features + tier_features + lattice_features + cycle_feature

    def record_observation(self, capacity: Dict[str, float],
                           lattice: 'TokenLatticeEngine',
                           cycle: int, actual_delta: float):
        """Record a (features, drift_delta) observation for training."""
        features = self._extract_features(capacity, lattice, cycle)
        self._history_X.append(features)
        self._history_y.append(actual_delta)

    def train(self) -> bool:
        """Train the SVM regressor on accumulated drift history."""
        svm = self._get_svm()
        if svm is None or len(self._history_X) < self.MIN_HISTORY:
            return False

        try:
            import numpy as np
            X = np.array(self._history_X)
            y = np.array(self._history_y)
            svm.fit(X, y)
            self._fitted = True
            return True
        except Exception:
            return False

    def predict_drift(self, capacity: Dict[str, float],
                      lattice: 'TokenLatticeEngine',
                      cycle: int) -> Dict[str, Any]:
        """Predict the next drift delta given current lattice state.

        Returns dict with predicted delta, confidence, and whether the
        prediction was from the ML model or a simple heuristic fallback.
        """
        features = self._extract_features(capacity, lattice, cycle)

        if self._fitted and self._svm is not None:
            try:
                import numpy as np
                X = np.array([features])
                predicted_delta = float(self._svm.predict(X)[0])
                self._predictions_made += 1

                return {
                    'predicted_delta': predicted_delta,
                    'predicted_direction': 1 if predicted_delta > 0 else (-1 if predicted_delta < 0 else 0),
                    'source': 'ml_svm',
                    'confidence': min(len(self._history_X) / 100.0, 1.0),
                    'predictions_made': self._predictions_made,
                }
            except Exception:
                pass

        # Heuristic fallback: use last observed delta
        if self._history_y:
            last_delta = self._history_y[-1]
            return {
                'predicted_delta': last_delta,
                'predicted_direction': 1 if last_delta > 0 else (-1 if last_delta < 0 else 0),
                'source': 'heuristic_last_delta',
                'confidence': 0.3,
                'predictions_made': self._predictions_made,
            }

        return {
            'predicted_delta': 0.0,
            'predicted_direction': 0,
            'source': 'no_data',
            'confidence': 0.0,
            'predictions_made': 0,
        }

    def status(self) -> Dict[str, Any]:
        return {
            'type': 'MLDriftPredictor',
            'fitted': self._fitted,
            'training_samples': len(self._history_X),
            'predictions_made': self._predictions_made,
            'min_history_required': self.MIN_HISTORY,
        }
