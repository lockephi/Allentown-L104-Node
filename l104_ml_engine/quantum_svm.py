"""
===============================================================================
L104 ML ENGINE — QUANTUM SVM v1.0.0
===============================================================================

Quantum-enhanced Support Vector Machines using quantum kernel circuits from
l104_quantum_gate_engine. Bridges quantum feature maps to classical sklearn SVC.

Classes:
  QuantumSVM           — Quantum kernel → precomputed SVC pipeline
  QuantumSVMFeatureMap — Sacred feature map builders for quantum encoding

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from .constants import (
    PHI, GOD_CODE, VOID_CONSTANT,
    QUANTUM_SVM_DEFAULT_QUBITS, QUANTUM_SVM_MAX_QUBITS,
    SVM_C_SACRED,
)


class QuantumSVMFeatureMap:
    """Sacred feature map builders for quantum SVM encoding.

    Provides static methods that return ParameterisedCircuit instances
    from l104_quantum_gate_engine for use as quantum feature maps.
    """

    @staticmethod
    def phi_encoded_map(n_qubits: int, depth: int = 2):
        """PHI-scaled feature encoding circuit.

        Returns a ParameterisedCircuit from AnsatzLibrary.phi_encoded_map().
        """
        from l104_quantum_gate_engine.quantum_ml import AnsatzLibrary
        return AnsatzLibrary.phi_encoded_map(n_qubits, depth=depth)

    @staticmethod
    def god_code_phase_map(n_qubits: int, depth: int = 2):
        """GOD_CODE phase-shifted encoding circuit.

        Returns a ParameterisedCircuit from AnsatzLibrary.god_code_phase_map().
        """
        from l104_quantum_gate_engine.quantum_ml import AnsatzLibrary
        return AnsatzLibrary.god_code_phase_map(n_qubits, depth=depth)

    @staticmethod
    def svm_feature_encoder(n_qubits: int, depth: int = 2):
        """Full SVM feature encoder circuit.

        Returns a ParameterisedCircuit with PHI + VOID + GOD_CODE encoding.
        """
        from l104_quantum_gate_engine.quantum_ml import AnsatzLibrary
        return AnsatzLibrary.svm_feature_encoder(n_qubits, depth=depth)

    @staticmethod
    def sacred_map(n_qubits: int, depth: int = 1):
        """Sacred ansatz feature map (GOD_CODE phase injection)."""
        from l104_quantum_gate_engine.quantum_ml import AnsatzLibrary
        return AnsatzLibrary.sacred_ansatz(n_qubits, depth=depth)


# Map of feature map names to KernelType enum values
_KERNEL_TYPE_MAP = {
    'sacred': 'SACRED_KERNEL',
    'phi_encoded': 'PHI_ENCODED',
    'god_code_phase': 'GOD_CODE_PHASE',
    'iron_lattice': 'IRON_LATTICE',
    'harmonic': 'HARMONIC_FOURIER',
    'zz': 'ZZ_FEATURE_MAP',
    'iqp': 'IQP',
}


class QuantumSVM:
    """Quantum-enhanced SVM using quantum kernel circuits.

    Computes quantum kernel matrices via l104_quantum_gate_engine's
    QuantumKernel, then feeds them into sklearn SVC(kernel='precomputed')
    for classical SVM training.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for quantum feature encoding (default: 4)
    kernel_type : str
        Quantum kernel type: 'sacred', 'phi_encoded', 'god_code_phase',
        'iron_lattice', 'harmonic', 'zz', 'iqp' (default: 'sacred')
    C : float
        SVM regularization parameter (default: GOD_CODE/100 ≈ 5.275)
    feature_map : optional
        Custom ParameterisedCircuit for feature encoding
    """

    def __init__(
        self,
        n_qubits: int = QUANTUM_SVM_DEFAULT_QUBITS,
        kernel_type: str = 'sacred',
        C: float = SVM_C_SACRED,
        feature_map=None,
    ):
        self.n_qubits = min(n_qubits, QUANTUM_SVM_MAX_QUBITS)
        self.kernel_type_name = kernel_type
        self.C = C
        self._custom_feature_map = feature_map
        self._qkernel = None
        self._svc = None
        self._X_train: Optional[np.ndarray] = None
        self._fitted = False

    def _get_quantum_kernel(self):
        """Lazy-load the quantum kernel."""
        if self._qkernel is None:
            from l104_quantum_gate_engine.quantum_ml import QuantumKernel, KernelType

            if self._custom_feature_map is not None:
                self._qkernel = QuantumKernel(
                    self.n_qubits, feature_map=self._custom_feature_map,
                )
            else:
                kt_name = _KERNEL_TYPE_MAP.get(self.kernel_type_name, 'SACRED_KERNEL')
                kt = KernelType[kt_name]
                self._qkernel = QuantumKernel(self.n_qubits, kernel_type=kt)

        return self._qkernel

    def _truncate_features(self, X: np.ndarray) -> np.ndarray:
        """Truncate or pad features to match qubit count."""
        X = np.atleast_2d(X).astype(np.float64)
        n_feat = X.shape[1]
        if n_feat > self.n_qubits:
            return X[:, :self.n_qubits]
        elif n_feat < self.n_qubits:
            pad = np.zeros((X.shape[0], self.n_qubits - n_feat))
            return np.hstack([X, pad])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumSVM':
        """Fit the quantum SVM.

        1. Truncate/pad features to n_qubits dimensions
        2. Compute quantum kernel matrix K(X_train, X_train)
        3. Fit sklearn SVC(kernel='precomputed') on K
        """
        from sklearn.svm import SVC

        X = self._truncate_features(X)
        self._X_train = X.copy()

        qk = self._get_quantum_kernel()
        kr = qk.compute_kernel(X)
        K_train = kr.kernel_matrix

        self._svc = SVC(kernel='precomputed', C=self.C, probability=True)
        self._svc.fit(K_train, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        X = self._truncate_features(X)
        qk = self._get_quantum_kernel()
        K_test = qk.compute_kernel(X, self._X_train).kernel_matrix
        return self._svc.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        X = self._truncate_features(X)
        qk = self._get_quantum_kernel()
        K_test = qk.compute_kernel(X, self._X_train).kernel_matrix
        return self._svc.predict_proba(K_test)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Test accuracy."""
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def quantum_kernel_matrix(self, X: np.ndarray,
                               Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the quantum kernel matrix K(X, Y)."""
        X = self._truncate_features(X)
        if Y is not None:
            Y = self._truncate_features(Y)
        qk = self._get_quantum_kernel()
        return qk.compute_kernel(X, Y).kernel_matrix

    def sacred_alignment_score(self) -> float:
        """Sacred alignment of the trained quantum kernel.

        Measures how well the kernel alignment correlates with
        GOD_CODE harmonics.
        """
        if not self._fitted or self._X_train is None:
            return 0.0
        qk = self._get_quantum_kernel()
        K = qk.compute_kernel(self._X_train).kernel_matrix
        trace = np.trace(K)
        n = K.shape[0]
        mean_diag = trace / n
        alignment = 1.0 - abs(mean_diag - PHI) / PHI
        return max(0.0, min(1.0, alignment))

    def status(self) -> Dict[str, Any]:
        """Return model status."""
        return {
            'n_qubits': self.n_qubits,
            'kernel_type': self.kernel_type_name,
            'C': self.C,
            'fitted': self._fitted,
            'n_support_vectors': len(self._svc.support_) if self._fitted else 0,
            'sacred_alignment': self.sacred_alignment_score(),
        }
