"""
===============================================================================
L104 ML ENGINE v2.0.0 — SACRED MACHINE LEARNING + CROSS-ENGINE INTEGRATION
===============================================================================

Comprehensive ML engine for the L104 Sovereign Node. Provides classical SVMs,
ensemble classifiers, clustering, quantum-enhanced classification, and
cross-engine knowledge synthesis — all tuned to GOD_CODE sacred constants.

ARCHITECTURE:
  MLEngine (singleton orchestrator)
    ├── L104SVM / SVMEnsemble        — Support Vector Machines (sacred kernels)
    ├── L104RandomForest             — 104-tree sacred forest
    ├── L104GradientBoosting         — GOD_CODE learning rate boosting
    ├── L104AdaBoost                 — VOID_CONSTANT adaptive boosting
    ├── L104EnsembleClassifier       — PHI-weighted meta-ensemble
    ├── L104KMeans / L104DBSCAN      — Sacred clustering (PHI-spiral init)
    ├── L104SpectralClustering       — PHI-affinity spectral decomposition
    ├── SacredKernelLibrary          — 6 custom SVM kernels
    ├── QuantumSVM                   — Quantum kernel → sklearn SVC
    ├── QuantumSVMFeatureMap         — Sacred quantum feature maps
    ├── VariationalQuantumClassifier — VQC circuit classifier
    ├── QuantumNearestNeighbor       — Quantum kernel k-NN
    ├── QuantumEnsembleClassifier    — Hybrid quantum-classical ensemble
    ├── CrossEngineFeatureExtractor  — 50-feature cross-engine extraction
    └── KnowledgeSynthesizer         — ML-powered knowledge fusion

SACRED CONSTANTS:
  SVM_C = GOD_CODE/100 ≈ 5.275 | SVM_γ = PHI/100 ≈ 0.01618
  RF trees = 104 | KMeans K = 13 (Fib(7)) | GBM lr = 1/(PHI*104) ≈ 0.00594

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

__version__ = "2.0.0"
__author__ = "L104 Sovereign Node"

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

from .constants import (
    ML_ENGINE_VERSION,
    PHI, GOD_CODE, VOID_CONSTANT, OMEGA,
    SVM_C_SACRED, SVM_GAMMA_SACRED,
    RF_N_ESTIMATORS_SACRED, GB_LEARNING_RATE_SACRED,
    KMEANS_K_SACRED, GOLDEN_ANGLE_RAD,
    QUANTUM_SVM_DEFAULT_QUBITS, VQC_DEFAULT_DEPTH,
    SACRED_LEARNING_RATE,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  SACRED KERNELS
# ═══════════════════════════════════════════════════════════════════════════════

from .sacred_kernels import SacredKernelLibrary

# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSICAL ML
# ═══════════════════════════════════════════════════════════════════════════════

from .svm import L104SVM, SVMEnsemble
from .classifiers import (
    L104RandomForest,
    L104GradientBoosting,
    L104AdaBoost,
    L104EnsembleClassifier,
)
from .clustering import L104KMeans, L104DBSCAN, L104SpectralClustering

# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM ML
# ═══════════════════════════════════════════════════════════════════════════════

from .quantum_svm import QuantumSVM, QuantumSVMFeatureMap
from .quantum_classifiers import (
    VariationalQuantumClassifier,
    QuantumNearestNeighbor,
    QuantumEnsembleClassifier,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

from .knowledge_synthesis import CrossEngineFeatureExtractor, KnowledgeSynthesizer

# ═══════════════════════════════════════════════════════════════════════════════
#  CROSS-ENGINE INTEGRATION (v2.0)
# ═══════════════════════════════════════════════════════════════════════════════

from .cross_engine import (
    VQPUKernelAccelerator,
    QDAFeatureEnricher,
    EnrichedFeatureExtractor,
    MLCrossEngineHub,
    ml_cross_engine_hub,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  ML ENGINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

from typing import Dict, Any, Optional


class MLEngine:
    """L104 ML Engine orchestrator — singleton hub for all ML capabilities.

    Provides lazy-loaded access to all ML subsystems:
      - Classical: SVM, Random Forest, Gradient Boosting, AdaBoost, ensemble
      - Clustering: KMeans, DBSCAN, Spectral
      - Quantum: QuantumSVM, VQC, QNN, quantum ensemble
      - Synthesis: Cross-engine feature extraction, knowledge fusion

    Usage:
        from l104_ml_engine import ml_engine
        ml_engine.svm.fit(X, y)
        ml_engine.classifier.fit(X, y)
        ml_engine.status()
    """

    def __init__(self):
        # Classical ML (eager init — lightweight)
        self.svm = L104SVM()
        self.svm_ensemble = SVMEnsemble()
        self.random_forest = L104RandomForest()
        self.gradient_boosting = L104GradientBoosting()
        self.adaboost = L104AdaBoost()
        self.classifier = L104EnsembleClassifier()

        # Clustering (eager init — lightweight)
        self.kmeans = L104KMeans()
        self.dbscan = L104DBSCAN()
        self.spectral = L104SpectralClustering()

        # Sacred kernels
        self.sacred_kernels = SacredKernelLibrary()

        # Quantum ML (lazy — requires quantum gate engine)
        self._quantum_svm: Optional[QuantumSVM] = None
        self._quantum_classifier: Optional[VariationalQuantumClassifier] = None
        self._quantum_ensemble: Optional[QuantumEnsembleClassifier] = None

        # Knowledge synthesis (lazy — requires all engines)
        self._knowledge_synthesizer: Optional[KnowledgeSynthesizer] = None
        self._feature_extractor: Optional[CrossEngineFeatureExtractor] = None

        # ══════ Cross-engine integration (lazy-loaded) ══════
        self._vqpu_bridge = None
        self._quantum_data_analyzer = None
        self._cross_engine_hub: Optional[MLCrossEngineHub] = None

    def get_cross_engine_hub(self) -> MLCrossEngineHub:
        """Get or create the cross-engine integration hub (v2.0)."""
        if self._cross_engine_hub is None:
            self._cross_engine_hub = MLCrossEngineHub()
        return self._cross_engine_hub

    def cross_engine_analysis(self, source: str = "") -> Dict[str, Any]:
        """Run full cross-engine ML analysis using all available L104 engines.

        v2.0: Pipelines through VQPU, QDA, Science, Math, Code, and Quantum engines.
        """
        return self.get_cross_engine_hub().full_cross_engine_analysis(source)

    def get_quantum_svm(self, n_qubits: int = 4) -> QuantumSVM:
        """Get or create the quantum SVM."""
        if self._quantum_svm is None or self._quantum_svm.n_qubits != n_qubits:
            self._quantum_svm = QuantumSVM(n_qubits=n_qubits)
        return self._quantum_svm

    def get_quantum_classifier(self, n_qubits: int = 4) -> VariationalQuantumClassifier:
        """Get or create the variational quantum classifier."""
        if self._quantum_classifier is None or self._quantum_classifier.n_qubits != n_qubits:
            self._quantum_classifier = VariationalQuantumClassifier(n_qubits=n_qubits)
        return self._quantum_classifier

    def get_quantum_ensemble(self, n_qubits: int = 4) -> QuantumEnsembleClassifier:
        """Get or create the quantum ensemble classifier."""
        if self._quantum_ensemble is None or self._quantum_ensemble.n_qubits != n_qubits:
            self._quantum_ensemble = QuantumEnsembleClassifier(n_qubits=n_qubits)
        return self._quantum_ensemble

    def get_knowledge_synthesizer(self) -> KnowledgeSynthesizer:
        """Get or create the knowledge synthesizer."""
        if self._knowledge_synthesizer is None:
            self._knowledge_synthesizer = KnowledgeSynthesizer()
        return self._knowledge_synthesizer

    def get_feature_extractor(self) -> CrossEngineFeatureExtractor:
        """Get or create the cross-engine feature extractor."""
        if self._feature_extractor is None:
            self._feature_extractor = CrossEngineFeatureExtractor()
        return self._feature_extractor

    def get_vqpu_bridge(self):
        """Get or create the VQPU bridge for Metal GPU quantum execution."""
        if self._vqpu_bridge is None:
            try:
                from l104_vqpu import get_bridge
                self._vqpu_bridge = get_bridge()
            except Exception:
                pass
        return self._vqpu_bridge

    def get_quantum_data_analyzer(self):
        """Get or create the quantum data analyzer."""
        if self._quantum_data_analyzer is None:
            try:
                from l104_quantum_data_analyzer import QuantumDataAnalyzer
                self._quantum_data_analyzer = QuantumDataAnalyzer()
            except Exception:
                pass
        return self._quantum_data_analyzer

    def status(self) -> Dict[str, Any]:
        """Return full ML engine status."""
        return {
            'version': ML_ENGINE_VERSION,
            'classical': {
                'svm': self.svm.status(),
                'random_forest': {'fitted': self.random_forest._fitted},
                'gradient_boosting': {'fitted': self.gradient_boosting._fitted},
                'adaboost': {'fitted': self.adaboost._fitted},
                'ensemble': self.classifier.status(),
            },
            'clustering': {
                'kmeans': self.kmeans.status(),
                'dbscan': self.dbscan.status(),
                'spectral': self.spectral.status(),
            },
            'quantum': {
                'quantum_svm_loaded': self._quantum_svm is not None,
                'quantum_classifier_loaded': self._quantum_classifier is not None,
                'quantum_ensemble_loaded': self._quantum_ensemble is not None,
            },
            'synthesis': {
                'synthesizer_loaded': self._knowledge_synthesizer is not None,
                'feature_extractor_loaded': self._feature_extractor is not None,
                'vqpu_bridge_loaded': self._vqpu_bridge is not None,
                'quantum_data_analyzer_loaded': self._quantum_data_analyzer is not None,
                'cross_engine_hub_loaded': self._cross_engine_hub is not None,
            },
            'sacred_constants': {
                'SVM_C': SVM_C_SACRED,
                'SVM_gamma': SVM_GAMMA_SACRED,
                'RF_trees': RF_N_ESTIMATORS_SACRED,
                'KMeans_K': KMEANS_K_SACRED,
                'GBM_lr': GB_LEARNING_RATE_SACRED,
                'GOD_CODE': GOD_CODE,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_ml_engine: Optional[MLEngine] = None


def get_engine() -> MLEngine:
    """Get or create the singleton MLEngine."""
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = MLEngine()
    return _ml_engine


ml_engine = get_engine()
