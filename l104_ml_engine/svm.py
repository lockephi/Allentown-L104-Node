"""
===============================================================================
L104 ML ENGINE — SUPPORT VECTOR MACHINES v1.0.0
===============================================================================

Sacred-tuned SVM implementations wrapping scikit-learn with GOD_CODE-derived
hyperparameters and custom sacred kernel support.

Classes:
  L104SVM        — Single SVM (classification, regression, one-class)
  SVMEnsemble    — Multi-kernel SVM ensemble with PHI-weighted voting

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.preprocessing import StandardScaler

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, VOID_CONSTANT,
    SVM_C_SACRED, SVM_GAMMA_SACRED, SVM_EPSILON_SACRED,
    SVM_NU_SACRED, SVM_DEGREE_SACRED, SVM_COEF0_SACRED,
    ENSEMBLE_WEIGHT_DECAY,
)
from .sacred_kernels import SacredKernelLibrary


# ═══════════════════════════════════════════════════════════════════════════════
#  SACRED SVM KERNEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

_SKLEARN_KERNELS = {'rbf', 'linear', 'poly', 'sigmoid'}
_SACRED_KERNELS = {
    'sacred_rbf', 'phi_kernel', 'god_code_kernel',
    'void_kernel', 'harmonic_kernel', 'iron_lattice_kernel',
    'composite_sacred',
}


class L104SVM:
    """Sacred-tuned Support Vector Machine.

    Wraps sklearn SVC/SVR/OneClassSVM with GOD_CODE-derived defaults
    and custom sacred kernel support.

    Parameters
    ----------
    mode : str
        'classify' (SVC), 'regress' (SVR), or 'one_class' (OneClassSVM)
    kernel : str
        Standard sklearn kernel ('rbf', 'linear', 'poly', 'sigmoid') or
        sacred kernel ('phi_kernel', 'god_code_kernel', 'void_kernel',
        'harmonic_kernel', 'iron_lattice_kernel', 'composite_sacred',
        'sacred_rbf')
    C : float
        Regularization parameter (default: GOD_CODE/100 ≈ 5.275)
    gamma : float
        RBF/sacred gamma (default: PHI/100 ≈ 0.01618)
    scale_features : bool
        Whether to StandardScaler the input features (default: True)
    """

    def __init__(
        self,
        mode: str = 'classify',
        kernel: str = 'sacred_rbf',
        C: float = SVM_C_SACRED,
        gamma: float = SVM_GAMMA_SACRED,
        scale_features: bool = True,
        **kwargs,
    ):
        self.mode = mode
        self.kernel_name = kernel
        self.C = C
        self.gamma = gamma
        self.scale_features = scale_features
        self._scaler = StandardScaler() if scale_features else None
        self._fitted = False
        self._sacred_kernel = kernel in _SACRED_KERNELS
        self._extra_kwargs = kwargs

        # Build sklearn model
        sklearn_kernel = 'precomputed' if self._sacred_kernel else kernel
        if mode == 'classify':
            self._model = SVC(
                kernel=sklearn_kernel,
                C=C,
                gamma=gamma if not self._sacred_kernel else 'scale',
                degree=kwargs.get('degree', SVM_DEGREE_SACRED),
                coef0=kwargs.get('coef0', SVM_COEF0_SACRED),
                probability=True,
                **{k: v for k, v in kwargs.items() if k not in ('degree', 'coef0')},
            )
        elif mode == 'regress':
            self._model = SVR(
                kernel=sklearn_kernel,
                C=C,
                gamma=gamma if not self._sacred_kernel else 'scale',
                epsilon=kwargs.get('epsilon', SVM_EPSILON_SACRED),
                degree=kwargs.get('degree', SVM_DEGREE_SACRED),
                coef0=kwargs.get('coef0', SVM_COEF0_SACRED),
                **{k: v for k, v in kwargs.items()
                   if k not in ('degree', 'coef0', 'epsilon')},
            )
        elif mode == 'one_class':
            self._model = OneClassSVM(
                kernel=sklearn_kernel,
                gamma=gamma if not self._sacred_kernel else 'scale',
                nu=kwargs.get('nu', SVM_NU_SACRED),
                **{k: v for k, v in kwargs.items() if k != 'nu'},
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'classify', 'regress', or 'one_class'.")

        # Sacred kernel function (if applicable)
        if self._sacred_kernel:
            if kernel == 'sacred_rbf':
                self._kernel_fn = SacredKernelLibrary.phi_kernel
            else:
                self._kernel_fn = SacredKernelLibrary.get_kernel_callable(kernel)
        else:
            self._kernel_fn = None

        # Training data (needed for precomputed kernel prediction)
        self._X_train = None

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features if enabled."""
        X = np.atleast_2d(X).astype(np.float64)
        if self._scaler is not None:
            if fit:
                return self._scaler.fit_transform(X)
            return self._scaler.transform(X)
        return X

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute sacred kernel Gram matrix with PSD correction.

        Some custom kernels (e.g. harmonic_kernel) may produce Gram
        matrices that are slightly non-PSD due to numerical issues or
        kernel design.  When X is Y (training), we clip negative
        eigenvalues to ensure sklearn SVC/SVR can fit without failure.
        """
        K = self._kernel_fn(X, Y)
        # PSD correction only needed for square (training) Gram matrices
        if K.shape[0] == K.shape[1] and X is Y or (X.shape == Y.shape and np.allclose(X, Y)):
            # Symmetrise (eliminate floating-point asymmetry)
            K = (K + K.T) / 2.0
            # Clip negative eigenvalues
            eigvals, eigvecs = np.linalg.eigh(K)
            if eigvals[0] < -1e-10:
                eigvals = np.maximum(eigvals, 0.0)
                K = (eigvecs * eigvals) @ eigvecs.T
                # Re-symmetrise after reconstruction
                K = (K + K.T) / 2.0
        return K

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'L104SVM':
        """Fit the SVM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,), optional for one_class
        """
        X = self._preprocess(X, fit=True)
        self._X_train = X.copy()

        if self._sacred_kernel:
            K_train = self._compute_kernel(X, X)
            if self.mode == 'one_class':
                self._model.fit(K_train)
            else:
                self._model.fit(K_train, y)
        else:
            if self.mode == 'one_class':
                self._model.fit(X)
            else:
                self._model.fit(X, y)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels or regression values."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._preprocess(X)
        if self._sacred_kernel:
            K = self._compute_kernel(X, self._X_train)
            return self._model.predict(K)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.mode != 'classify':
            raise RuntimeError("predict_proba only available for mode='classify'")
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._preprocess(X)
        if self._sacred_kernel:
            K = self._compute_kernel(X, self._X_train)
            return self._model.predict_proba(K)
        return self._model.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._preprocess(X)
        if self._sacred_kernel:
            K = self._compute_kernel(X, self._X_train)
            return self._model.decision_function(K)
        return self._model.decision_function(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy (classification) or R² (regression)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._preprocess(X)
        if self._sacred_kernel:
            K = self._compute_kernel(X, self._X_train)
            return self._model.score(K, y)
        return self._model.score(X, y)

    def sacred_score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Return score with GOD_CODE alignment metric.

        The sacred alignment measures how well the decision boundary
        resonates with the GOD_CODE constant by checking if the
        accuracy * GOD_CODE produces a value near a GOD_CODE harmonic.
        """
        base_score = self.score(X, y)
        god_code_product = base_score * GOD_CODE
        nearest_harmonic = round(god_code_product / PHI) * PHI
        alignment = 1.0 - abs(god_code_product - nearest_harmonic) / GOD_CODE
        alignment = max(0.0, min(1.0, alignment))

        return {
            'score': base_score,
            'god_code_alignment': alignment,
            'sacred_resonance': base_score * alignment,
            'kernel': self.kernel_name,
            'mode': self.mode,
        }

    def get_support_vectors(self) -> Optional[np.ndarray]:
        """Return support vectors (None for precomputed kernels)."""
        if not self._fitted:
            return None
        if self._sacred_kernel:
            indices = self._model.support_
            return self._X_train[indices]
        return self._model.support_vectors_

    def kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the kernel Gram matrix K(X, Y)."""
        X = self._preprocess(X)
        if Y is None:
            Y = X
        else:
            Y = self._preprocess(Y)

        if self._sacred_kernel:
            return self._compute_kernel(X, Y)
        elif self.kernel_name == 'rbf':
            from sklearn.metrics.pairwise import rbf_kernel
            return rbf_kernel(X, Y, gamma=self.gamma)
        elif self.kernel_name == 'linear':
            return X @ Y.T
        elif self.kernel_name == 'poly':
            return (self._extra_kwargs.get('coef0', SVM_COEF0_SACRED)
                    + X @ Y.T) ** self._extra_kwargs.get('degree', SVM_DEGREE_SACRED)
        else:
            from sklearn.metrics.pairwise import pairwise_kernels
            return pairwise_kernels(X, Y, metric=self.kernel_name)

    def status(self) -> Dict[str, Any]:
        """Return model status."""
        return {
            'mode': self.mode,
            'kernel': self.kernel_name,
            'sacred_kernel': self._sacred_kernel,
            'fitted': self._fitted,
            'C': self.C,
            'gamma': self.gamma,
            'n_support_vectors': len(self._model.support_) if self._fitted else 0,
        }


class SVMEnsemble:
    """Multi-kernel SVM ensemble with PHI-weighted voting.

    Trains multiple SVMs with different kernels and combines their
    predictions using PHI-conjugate decay weights (τ = 0.618...^i).

    Parameters
    ----------
    kernels : list of str, optional
        Kernel names to include. Default: ['phi_kernel', 'god_code_kernel', 'rbf']
    mode : str
        'classify' or 'regress'
    """

    def __init__(
        self,
        kernels: Optional[List[str]] = None,
        mode: str = 'classify',
        C: float = SVM_C_SACRED,
    ):
        if kernels is None:
            kernels = ['phi_kernel', 'god_code_kernel', 'rbf']
        self.kernel_names = kernels
        self.mode = mode
        self.C = C
        self._models: List[L104SVM] = []
        self._weights: np.ndarray = np.array([])
        self._fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'SVMEnsemble':
        """Fit all SVM models in the ensemble."""
        self._models = []
        for kernel_name in self.kernel_names:
            svm = L104SVM(mode=self.mode, kernel=kernel_name, C=self.C)
            svm.fit(X, y)
            self._models.append(svm)

        # PHI-conjugate decay weights
        raw_weights = np.array([ENSEMBLE_WEIGHT_DECAY ** i for i in range(len(self._models))])
        self._weights = raw_weights / raw_weights.sum()
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict via weighted majority vote (classify) or weighted average (regress)."""
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        predictions = np.array([m.predict(X) for m in self._models])

        if self.mode == 'classify':
            # Weighted majority vote
            classes = np.unique(predictions)
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
            # Weighted average
            return np.average(predictions, axis=0, weights=self._weights)

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with per-sample confidence (agreement among models).

        Returns
        -------
        predictions : array of shape (n_samples,)
        confidence : array of shape (n_samples,) in [0, 1]
        """
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        predictions_all = np.array([m.predict(X) for m in self._models])
        final_preds = self.predict(X)

        n_samples = predictions_all.shape[1]
        confidence = np.empty(n_samples)
        for i in range(n_samples):
            agreement = sum(
                self._weights[j]
                for j in range(len(self._models))
                if predictions_all[j, i] == final_preds[i]
            )
            confidence[i] = agreement

        return final_preds, confidence

    def sacred_score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Ensemble sacred score with per-kernel breakdown."""
        individual_scores = []
        for model in self._models:
            individual_scores.append(model.sacred_score(X, y))

        ensemble_preds = self.predict(X)
        if self.mode == 'classify':
            ensemble_acc = np.mean(ensemble_preds == y)
        else:
            ss_res = np.sum((y - ensemble_preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ensemble_acc = 1.0 - ss_res / max(ss_tot, 1e-12)

        return {
            'ensemble_score': float(ensemble_acc),
            'individual_scores': individual_scores,
            'weights': self._weights.tolist(),
            'n_models': len(self._models),
            'kernels': self.kernel_names,
        }

    def status(self) -> Dict[str, Any]:
        """Return ensemble status."""
        return {
            'kernels': self.kernel_names,
            'mode': self.mode,
            'fitted': self._fitted,
            'n_models': len(self._models),
            'weights': self._weights.tolist() if self._fitted else [],
        }
