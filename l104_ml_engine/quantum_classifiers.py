"""
===============================================================================
L104 ML ENGINE — QUANTUM CLASSIFIERS v1.0.0
===============================================================================

Quantum-enhanced classification circuits using variational quantum algorithms
from l104_quantum_gate_engine.

Classes:
  VariationalQuantumClassifier — VQC: data encoding + variational ansatz + classification
  QuantumNearestNeighbor       — Quantum kernel k-NN (k=3)
  QuantumEnsembleClassifier    — Hybrid quantum-classical ensemble

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, VOID_CONSTANT,
    QUANTUM_SVM_DEFAULT_QUBITS, QUANTUM_SVM_MAX_QUBITS,
    VQC_DEFAULT_DEPTH, SACRED_LEARNING_RATE, VQC_DEFAULT_SHOTS,
    ENSEMBLE_WEIGHT_DECAY,
)


class VariationalQuantumClassifier:
    """Variational Quantum Classifier (VQC).

    Pipeline:
      1. Encode classical features into quantum state via feature map
      2. Apply variational ansatz with trainable parameters
      3. Measure output qubits to determine class label
      4. Optimize parameters via QNNTrainer (gradient-based)

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default: 4)
    ansatz : str
        Ansatz type: 'vqc', 'sacred', 'hardware_efficient' (default: 'vqc')
    depth : int
        Circuit depth (default: 3)
    n_classes : int
        Number of output classes (default: 2)
    """

    def __init__(
        self,
        n_qubits: int = QUANTUM_SVM_DEFAULT_QUBITS,
        ansatz: str = 'vqc',
        depth: int = VQC_DEFAULT_DEPTH,
        n_classes: int = 2,
    ):
        self.n_qubits = min(n_qubits, QUANTUM_SVM_MAX_QUBITS)
        self.ansatz_type = ansatz
        self.depth = depth
        self.n_classes = n_classes
        self._circuit = None
        self._optimal_params: Optional[np.ndarray] = None
        self._fitted = False
        self._classes: Optional[np.ndarray] = None
        self._training_history: List[float] = []

    def _build_circuit(self):
        """Build the variational circuit."""
        from l104_quantum_gate_engine.quantum_ml import AnsatzLibrary
        if self.ansatz_type == 'vqc':
            return AnsatzLibrary.vqc_classifier(
                self.n_qubits, depth=self.depth, n_classes=self.n_classes
            )
        elif self.ansatz_type == 'sacred':
            return AnsatzLibrary.sacred_ansatz(self.n_qubits, depth=self.depth)
        elif self.ansatz_type == 'hardware_efficient':
            return AnsatzLibrary.hardware_efficient(self.n_qubits, depth=self.depth)
        else:
            return AnsatzLibrary.vqc_classifier(
                self.n_qubits, depth=self.depth, n_classes=self.n_classes
            )

    def _encode_and_evaluate(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Encode features + evaluate circuit -> probabilities."""
        circuit = self._circuit
        # Merge feature encoding with variational parameters
        n_params = circuit.num_parameters
        n_feat = len(x)
        # First n_feat parameters are data-encoded, rest are trainable
        full_params = params.copy()
        for i in range(min(n_feat, n_params)):
            full_params[i] = params[i] + x[i] * PHI  # PHI-scaled encoding

        sv = circuit.statevector(full_params)
        probs = np.abs(sv) ** 2

        # Extract class probabilities from first ceil(log2(n_classes)) qubits
        n_output = max(1, int(np.ceil(np.log2(max(self.n_classes, 2)))))
        n_output = min(n_output, self.n_qubits)

        class_probs = np.zeros(self.n_classes)
        n_states = 2 ** self.n_qubits
        for state_idx in range(min(n_states, len(probs))):
            # Extract output qubit bits
            class_idx = state_idx % self.n_classes
            class_probs[class_idx] += probs[state_idx]

        # Normalize
        total = class_probs.sum()
        if total > 0:
            class_probs /= total
        return class_probs

    def fit(self, X: np.ndarray, y: np.ndarray,
            max_iterations: int = 50) -> Dict[str, Any]:
        """Train the VQC.

        Uses a simple gradient-free optimization (coordinate descent)
        to minimize classification loss.

        Returns training metrics.
        """
        X = np.atleast_2d(X).astype(np.float64)
        self._classes = np.unique(y)
        n_classes_actual = len(self._classes)
        if n_classes_actual > self.n_classes:
            self.n_classes = n_classes_actual

        # Map labels to 0..n_classes-1
        label_map = {c: i for i, c in enumerate(self._classes)}
        y_mapped = np.array([label_map[yi] for yi in y])

        # Build circuit
        self._circuit = self._build_circuit()
        n_params = self._circuit.num_parameters

        # Initialize parameters near zero with PHI-scaled perturbation
        rng = np.random.default_rng(104)
        params = rng.normal(0, SACRED_LEARNING_RATE, size=n_params)

        # Simple optimization loop (coordinate descent + random perturbation)
        best_loss = float('inf')
        best_params = params.copy()
        lr = 0.1
        self._training_history = []

        for iteration in range(max_iterations):
            # Compute loss
            loss = 0.0
            for i in range(len(X)):
                x_trunc = X[i, :self.n_qubits] if X.shape[1] > self.n_qubits else X[i]
                probs = self._encode_and_evaluate(x_trunc, params)
                target = y_mapped[i]
                loss -= np.log(max(probs[target], 1e-10))
            loss /= len(X)
            self._training_history.append(float(loss))

            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()

            # Stochastic perturbation update (SPSA-inspired)
            delta = rng.choice([-1, 1], size=n_params).astype(float)
            params_plus = params + lr * delta
            params_minus = params - lr * delta

            loss_plus = 0.0
            loss_minus = 0.0
            for i in range(len(X)):
                x_trunc = X[i, :self.n_qubits] if X.shape[1] > self.n_qubits else X[i]
                p_plus = self._encode_and_evaluate(x_trunc, params_plus)
                p_minus = self._encode_and_evaluate(x_trunc, params_minus)
                target = y_mapped[i]
                loss_plus -= np.log(max(p_plus[target], 1e-10))
                loss_minus -= np.log(max(p_minus[target], 1e-10))
            loss_plus /= len(X)
            loss_minus /= len(X)

            grad_estimate = (loss_plus - loss_minus) / (2 * lr * delta + 1e-10)
            params -= lr * grad_estimate

            # Decay learning rate
            lr *= (1.0 - PHI_CONJUGATE / max_iterations)

        self._optimal_params = best_params
        self._fitted = True

        # Compute final accuracy
        preds = self._predict_internal(X, y_mapped)
        accuracy = float(np.mean(preds == y_mapped))

        return {
            'final_loss': float(best_loss),
            'accuracy': accuracy,
            'n_parameters': n_params,
            'n_iterations': max_iterations,
            'ansatz': self.ansatz_type,
            'n_qubits': self.n_qubits,
        }

    def _predict_internal(self, X: np.ndarray, y_mapped=None) -> np.ndarray:
        """Predict mapped class indices."""
        results = []
        for i in range(len(X)):
            x_trunc = X[i, :self.n_qubits] if X.shape[1] > self.n_qubits else X[i]
            probs = self._encode_and_evaluate(x_trunc, self._optimal_params)
            results.append(np.argmax(probs))
        return np.array(results)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        X = np.atleast_2d(X).astype(np.float64)
        mapped = self._predict_internal(X)
        return self._classes[mapped]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        X = np.atleast_2d(X).astype(np.float64)
        probas = []
        for i in range(len(X)):
            x_trunc = X[i, :self.n_qubits] if X.shape[1] > self.n_qubits else X[i]
            probs = self._encode_and_evaluate(x_trunc, self._optimal_params)
            probas.append(probs)
        return np.array(probas)

    def status(self) -> Dict[str, Any]:
        return {
            'n_qubits': self.n_qubits,
            'ansatz': self.ansatz_type,
            'depth': self.depth,
            'n_classes': self.n_classes,
            'fitted': self._fitted,
            'n_parameters': self._circuit.num_parameters if self._circuit else 0,
            'training_steps': len(self._training_history),
        }


class QuantumNearestNeighbor:
    """Quantum kernel k-NN classifier.

    Uses quantum kernel distance (1 - K(x_i, x_j)) as the distance metric
    for k-nearest neighbor classification.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for feature encoding (default: 4)
    k : int
        Number of nearest neighbors (default: 3, Fibonacci(4))
    kernel_type : str
        Quantum kernel type (default: 'sacred')
    """

    def __init__(
        self,
        n_qubits: int = QUANTUM_SVM_DEFAULT_QUBITS,
        k: int = 3,
        kernel_type: str = 'sacred',
    ):
        self.n_qubits = min(n_qubits, QUANTUM_SVM_MAX_QUBITS)
        self.k = k
        self.kernel_type = kernel_type
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._qkernel = None
        self._fitted = False

    def _get_kernel(self):
        """Lazy-load quantum kernel."""
        if self._qkernel is None:
            from l104_quantum_gate_engine.quantum_ml import QuantumKernel, KernelType
            kt_map = {
                'sacred': KernelType.SACRED_KERNEL,
                'phi_encoded': KernelType.PHI_ENCODED,
                'god_code_phase': KernelType.GOD_CODE_PHASE,
                'zz': KernelType.ZZ_FEATURE_MAP,
            }
            kt = kt_map.get(self.kernel_type, KernelType.SACRED_KERNEL)
            self._qkernel = QuantumKernel(self.n_qubits, kernel_type=kt)
        return self._qkernel

    def _truncate(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float64)
        if X.shape[1] > self.n_qubits:
            return X[:, :self.n_qubits]
        elif X.shape[1] < self.n_qubits:
            return np.hstack([X, np.zeros((X.shape[0], self.n_qubits - X.shape[1]))])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumNearestNeighbor':
        """Store training data."""
        self._X_train = self._truncate(X)
        self._y_train = np.asarray(y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict via quantum kernel k-NN."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        X = self._truncate(X)
        qk = self._get_kernel()

        # Compute kernel matrix (similarity) between test and train
        K = qk.compute_kernel(X, self._X_train).kernel_matrix
        # Convert to distance: d = 1 - K
        D = 1.0 - K

        predictions = []
        for i in range(len(X)):
            # Find k nearest neighbors
            nearest_indices = np.argsort(D[i])[:self.k]
            nearest_labels = self._y_train[nearest_indices]
            # Weighted vote by similarity (1/distance)
            votes = {}
            for j, idx in enumerate(nearest_indices):
                label = nearest_labels[j]
                weight = 1.0 / (D[i, idx] + 1e-10)
                votes[label] = votes.get(label, 0.0) + weight
            predictions.append(max(votes, key=votes.get))

        return np.array(predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))


class QuantumEnsembleClassifier:
    """Hybrid quantum-classical ensemble classifier.

    Combines:
      - VariationalQuantumClassifier (VQC)
      - QuantumSVM (from quantum_svm module)
      - QuantumNearestNeighbor

    with PHI-weighted voting.

    Parameters
    ----------
    n_qubits : int
        Shared qubit count for all quantum classifiers (default: 4)
    """

    def __init__(self, n_qubits: int = QUANTUM_SVM_DEFAULT_QUBITS):
        self.n_qubits = min(n_qubits, QUANTUM_SVM_MAX_QUBITS)
        self._vqc = VariationalQuantumClassifier(n_qubits=self.n_qubits, ansatz='vqc')
        self._qnn = QuantumNearestNeighbor(n_qubits=self.n_qubits, k=3)
        self._qsvm = None  # Lazy load
        self._fitted = False
        self._weights = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumEnsembleClassifier':
        """Fit all quantum classifiers."""
        X = np.atleast_2d(X).astype(np.float64)

        models = []

        # VQC
        self._vqc.fit(X, y, max_iterations=30)
        models.append(('vqc', self._vqc))

        # QNN
        self._qnn.fit(X, y)
        models.append(('qnn', self._qnn))

        # QuantumSVM (lazy)
        try:
            from .quantum_svm import QuantumSVM
            self._qsvm = QuantumSVM(n_qubits=self.n_qubits, kernel_type='sacred')
            self._qsvm.fit(X, y)
            models.append(('qsvm', self._qsvm))
        except Exception:
            pass

        self._models = models

        # PHI-weighted
        n = len(models)
        raw = np.array([ENSEMBLE_WEIGHT_DECAY ** i for i in range(n)])
        self._weights = raw / raw.sum()
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict via PHI-weighted voting."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        X = np.atleast_2d(X).astype(np.float64)

        all_preds = []
        for name, model in self._models:
            all_preds.append(model.predict(X))

        all_preds = np.array(all_preds)
        n_samples = all_preds.shape[1]
        result = np.empty(n_samples, dtype=all_preds.dtype)

        for i in range(n_samples):
            votes = {}
            for j, w in enumerate(self._weights):
                pred = all_preds[j, i]
                votes[pred] = votes.get(pred, 0.0) + w
            result[i] = max(votes, key=votes.get)
        return result

    def hybrid_confidence(self, X: np.ndarray) -> np.ndarray:
        """Per-sample confidence (agreement among quantum models)."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        X = np.atleast_2d(X).astype(np.float64)
        all_preds = [m.predict(X) for _, m in self._models]
        final = self.predict(X)

        confidence = np.zeros(len(X))
        for i in range(len(X)):
            agreement = sum(
                self._weights[j] for j in range(len(self._models))
                if all_preds[j][i] == final[i]
            )
            confidence[i] = agreement
        return confidence

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def status(self) -> Dict[str, Any]:
        return {
            'n_qubits': self.n_qubits,
            'fitted': self._fitted,
            'models': [name for name, _ in self._models] if self._fitted else [],
            'weights': self._weights.tolist() if self._fitted else [],
        }
