"""
===============================================================================
L104 ML ENGINE — ENSEMBLE CLASSIFIERS v1.0.0
===============================================================================

Sacred-tuned ensemble classifiers wrapping scikit-learn with GOD_CODE-derived
hyperparameters and PHI-weighted voting.

Classes:
  L104RandomForest       — 104-tree forest with PHI-normalized importances
  L104GradientBoosting   — Sacred learning rate gradient boosting
  L104AdaBoost           — VOID_CONSTANT-scaled adaptive boosting
  L104EnsembleClassifier — Meta-ensemble combining all classifiers + SVM

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Iterator
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from sklearn.preprocessing import StandardScaler

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, VOID_CONSTANT,
    RF_N_ESTIMATORS_SACRED, RF_MAX_DEPTH_SACRED,
    RF_MIN_SAMPLES_SPLIT_SACRED, RF_MIN_SAMPLES_LEAF_SACRED,
    GB_N_ESTIMATORS_SACRED, GB_LEARNING_RATE_SACRED,
    GB_MAX_DEPTH_SACRED, GB_SUBSAMPLE_SACRED,
    ADABOOST_N_ESTIMATORS_SACRED, ADABOOST_LEARNING_RATE_SACRED,
    ENSEMBLE_WEIGHT_DECAY,
)


class L104RandomForest:
    """Random Forest with 104 estimators and PHI-weighted feature importances.

    Sacred tuning:
      n_estimators    = 104  (L104 sacred number)
      max_depth       = 12   (int(PHI * 8))
      min_samples_split = 4  (int(PHI * 3))
      min_samples_leaf  = 2  (Fibonacci(3))
    """

    def __init__(
        self,
        mode: str = 'classify',
        n_estimators: int = RF_N_ESTIMATORS_SACRED,
        max_depth: int = RF_MAX_DEPTH_SACRED,
        **kwargs,
    ):
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._fitted = False

        base_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=kwargs.pop('min_samples_split', RF_MIN_SAMPLES_SPLIT_SACRED),
            min_samples_leaf=kwargs.pop('min_samples_leaf', RF_MIN_SAMPLES_LEAF_SACRED),
            random_state=104,
            n_jobs=-1,
        )
        base_params.update(kwargs)

        if mode == 'classify':
            self._model = RandomForestClassifier(**base_params)
        elif mode == 'regress':
            self._model = RandomForestRegressor(**base_params)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'classify' or 'regress'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'L104RandomForest':
        """Fit the random forest."""
        self._model.fit(np.atleast_2d(X), y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels or regression values."""
        return self._model.predict(np.atleast_2d(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.mode != 'classify':
            raise RuntimeError("predict_proba only available for mode='classify'")
        return self._model.predict_proba(np.atleast_2d(X))

    def feature_importance_sacred(self) -> Dict[str, Any]:
        """Return PHI-normalized feature importances.

        Raw importances are scaled by PHI so that the most important
        feature has importance = PHI (1.618...) and others are proportional.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        raw = self._model.feature_importances_
        max_imp = raw.max() if raw.max() > 0 else 1.0
        phi_normalized = (raw / max_imp) * PHI
        return {
            'raw': raw.tolist(),
            'phi_normalized': phi_normalized.tolist(),
            'n_features': len(raw),
            'most_important_idx': int(np.argmax(raw)),
        }

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy or R²."""
        return self._model.score(np.atleast_2d(X), y)


class L104GradientBoosting:
    """Gradient Boosting with sacred learning rate and PHI shrinkage.

    Sacred tuning:
      n_estimators  = 104  (L104 sacred number)
      learning_rate = 1/(PHI * 104) ≈ 0.00594
      max_depth     = 4    (int(PHI * 3))
      subsample     = 0.618... (PHI_CONJUGATE)
    """

    def __init__(
        self,
        mode: str = 'classify',
        n_estimators: int = GB_N_ESTIMATORS_SACRED,
        learning_rate: float = GB_LEARNING_RATE_SACRED,
        **kwargs,
    ):
        self.mode = mode
        self._fitted = False

        base_params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=kwargs.pop('max_depth', GB_MAX_DEPTH_SACRED),
            subsample=kwargs.pop('subsample', GB_SUBSAMPLE_SACRED),
            random_state=104,
        )
        base_params.update(kwargs)

        if mode == 'classify':
            self._model = GradientBoostingClassifier(**base_params)
        elif mode == 'regress':
            self._model = GradientBoostingRegressor(**base_params)
        else:
            raise ValueError(f"Unknown mode '{mode}'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'L104GradientBoosting':
        """Fit the gradient boosting model."""
        self._model.fit(np.atleast_2d(X), y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict."""
        return self._model.predict(np.atleast_2d(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.mode != 'classify':
            raise RuntimeError("predict_proba only available for mode='classify'")
        return self._model.predict_proba(np.atleast_2d(X))

    def staged_predict(self, X: np.ndarray) -> Iterator:
        """Staged prediction for early stopping analysis."""
        return self._model.staged_predict(np.atleast_2d(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy or R²."""
        return self._model.score(np.atleast_2d(X), y)


class L104AdaBoost:
    """AdaBoost with VOID_CONSTANT-normalized sample weights.

    Sacred tuning:
      n_estimators  = 52   (104 / 2)
      learning_rate = 1.0416... (VOID_CONSTANT)
    """

    def __init__(
        self,
        mode: str = 'classify',
        n_estimators: int = ADABOOST_N_ESTIMATORS_SACRED,
        learning_rate: float = ADABOOST_LEARNING_RATE_SACRED,
        **kwargs,
    ):
        self.mode = mode
        self._fitted = False

        base_params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=104,
        )
        base_params.update(kwargs)

        if mode == 'classify':
            self._model = AdaBoostClassifier(**base_params)
        elif mode == 'regress':
            self._model = AdaBoostRegressor(**base_params)
        else:
            raise ValueError(f"Unknown mode '{mode}'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'L104AdaBoost':
        """Fit the AdaBoost model."""
        self._model.fit(np.atleast_2d(X), y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict."""
        return self._model.predict(np.atleast_2d(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy or R²."""
        return self._model.score(np.atleast_2d(X), y)


class L104EnsembleClassifier:
    """Meta-ensemble: RF + GradientBoosting + AdaBoost + SVM with PHI-weighted voting.

    Combines all L104 classifiers into a single ensemble using
    PHI-conjugate decay weights (τ = 0.618...^i). Optionally includes
    a quantum-enhanced classifier.

    Parameters
    ----------
    include_svm : bool
        Include L104SVM in the ensemble (default: True)
    include_quantum : bool
        Include quantum classifier (default: False, requires quantum gate engine)
    mode : str
        'classify' or 'regress'
    """

    def __init__(
        self,
        include_svm: bool = True,
        include_quantum: bool = False,
        mode: str = 'classify',
    ):
        self.mode = mode
        self.include_svm = include_svm
        self.include_quantum = include_quantum
        self._scaler = StandardScaler()
        self._fitted = False

        # Build component models
        self._models: List[Tuple[str, Any]] = [
            ('random_forest', L104RandomForest(mode=mode)),
            ('gradient_boosting', L104GradientBoosting(mode=mode)),
            ('adaboost', L104AdaBoost(mode=mode)),
        ]

        if include_svm:
            from .svm import L104SVM
            self._models.append(('svm', L104SVM(mode=mode, kernel='phi_kernel')))

        # Quantum classifier added lazily in fit() if requested
        self._weights: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'L104EnsembleClassifier':
        """Fit all models in the ensemble."""
        X = np.atleast_2d(X).astype(np.float64)
        X_scaled = self._scaler.fit_transform(X)

        if self.include_quantum and not any(n == 'quantum' for n, _ in self._models):
            try:
                from .quantum_classifiers import VariationalQuantumClassifier
                n_features = X.shape[1]
                n_qubits = min(n_features, 8)
                vqc = VariationalQuantumClassifier(n_qubits=n_qubits)
                self._models.append(('quantum', vqc))
            except ImportError:
                pass

        for name, model in self._models:
            if name == 'svm':
                model.fit(X_scaled, y)
            elif name == 'quantum':
                model.fit(X_scaled[:, :model.n_qubits] if X_scaled.shape[1] > model.n_qubits
                          else X_scaled, y)
            else:
                model.fit(X_scaled, y)

        # PHI-conjugate decay weights
        n = len(self._models)
        raw_weights = np.array([ENSEMBLE_WEIGHT_DECAY ** i for i in range(n)])
        self._weights = raw_weights / raw_weights.sum()
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict via PHI-weighted voting."""
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        X = np.atleast_2d(X).astype(np.float64)
        X_scaled = self._scaler.transform(X)

        predictions = []
        for name, model in self._models:
            if name == 'quantum':
                x_in = (X_scaled[:, :model.n_qubits]
                        if X_scaled.shape[1] > model.n_qubits else X_scaled)
                predictions.append(model.predict(x_in))
            else:
                predictions.append(model.predict(X_scaled))

        predictions = np.array(predictions)

        if self.mode == 'classify':
            n_samples = predictions.shape[1]
            result = np.empty(n_samples, dtype=predictions.dtype)
            for i in range(n_samples):
                votes = {}
                for j, w in enumerate(self._weights):
                    pred = predictions[j, i]
                    votes[pred] = votes.get(pred, 0.0) + w
                result[i] = max(votes, key=votes.get)
            return result
        else:
            return np.average(predictions, axis=0, weights=self._weights)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via weighted averaging of per-model probas."""
        if self.mode != 'classify':
            raise RuntimeError("predict_proba only available for mode='classify'")
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted.")

        X = np.atleast_2d(X).astype(np.float64)
        X_scaled = self._scaler.transform(X)

        probas = []
        weights_used = []
        for idx, (name, model) in enumerate(self._models):
            try:
                if name == 'quantum':
                    x_in = (X_scaled[:, :model.n_qubits]
                            if X_scaled.shape[1] > model.n_qubits else X_scaled)
                    probas.append(model.predict_proba(x_in))
                else:
                    probas.append(model.predict_proba(X_scaled))
                weights_used.append(self._weights[idx])
            except (RuntimeError, AttributeError):
                continue

        if not probas:
            raise RuntimeError("No model supports predict_proba")

        weights_arr = np.array(weights_used)
        weights_arr /= weights_arr.sum()
        result = sum(w * p for w, p in zip(weights_arr, probas))
        return result

    def confidence_score(self, X: np.ndarray) -> np.ndarray:
        """Per-sample PHI-weighted confidence (max proba)."""
        probas = self.predict_proba(X)
        return probas.max(axis=1)

    def sacred_alignment(self, X: np.ndarray, y: np.ndarray) -> float:
        """GOD_CODE resonance of ensemble predictions.

        Measures how well the ensemble accuracy aligns with GOD_CODE harmonics.
        """
        preds = self.predict(X)
        if self.mode == 'classify':
            acc = np.mean(preds == y)
        else:
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            acc = 1.0 - ss_res / max(ss_tot, 1e-12)

        god_code_product = acc * GOD_CODE
        nearest_harmonic = round(god_code_product / PHI) * PHI
        alignment = 1.0 - abs(god_code_product - nearest_harmonic) / GOD_CODE
        return max(0.0, min(1.0, alignment))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return ensemble accuracy or R²."""
        preds = self.predict(X)
        if self.mode == 'classify':
            return float(np.mean(preds == y))
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / max(ss_tot, 1e-12))

    def status(self) -> Dict[str, Any]:
        """Return ensemble status."""
        return {
            'models': [name for name, _ in self._models],
            'mode': self.mode,
            'fitted': self._fitted,
            'weights': self._weights.tolist() if self._fitted else [],
            'include_svm': self.include_svm,
            'include_quantum': self.include_quantum,
        }
