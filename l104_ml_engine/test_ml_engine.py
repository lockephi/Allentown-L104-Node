"""
===============================================================================
L104 ML ENGINE — UNIT TESTS v1.0.0
===============================================================================
Tests for all l104_ml_engine modules: SVM, classifiers, clustering,
sacred kernels, quantum SVM, quantum classifiers, knowledge synthesis.

Run: python -m pytest l104_ml_engine/test_ml_engine.py -v
===============================================================================
"""

import numpy as np
import pytest

# Sacred constants for verification
PHI = 1.618033988749895
GOD_CODE = 527.5184818492612
VOID_CONSTANT = 1.0416180339887497


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_constants_consistency():
    """ML constants are correctly derived from sacred constants."""
    from l104_ml_engine.constants import (
        SVM_C_SACRED, SVM_GAMMA_SACRED, RF_N_ESTIMATORS_SACRED,
        KMEANS_K_SACRED, GB_LEARNING_RATE_SACRED, GOLDEN_ANGLE_RAD,
    )
    assert abs(SVM_C_SACRED - GOD_CODE / 100.0) < 1e-10
    assert abs(SVM_GAMMA_SACRED - PHI / 100.0) < 1e-10
    assert RF_N_ESTIMATORS_SACRED == 104
    assert KMEANS_K_SACRED == 13
    assert abs(GB_LEARNING_RATE_SACRED - 1.0 / (PHI * 104)) < 1e-10
    assert abs(GOLDEN_ANGLE_RAD - 2.399963229728653) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
#  SACRED KERNEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_sacred_kernels_shape_and_rbf_psd():
    """Sacred kernels produce correct-shaped Gram matrices; RBF-based ones are PSD."""
    from l104_ml_engine.sacred_kernels import SacredKernelLibrary

    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 4))

    for name in ['phi', 'god_code', 'void', 'harmonic', 'iron', 'composite']:
        kernel_fn = SacredKernelLibrary.get_kernel_callable(name)
        K = kernel_fn(X, X)
        assert K.shape == (20, 20), f"Kernel {name} wrong shape"
        assert np.all(np.isfinite(K)), f"Kernel {name} has non-finite values"

    # RBF-based kernels (phi, void, iron) are guaranteed PSD
    for name in ['phi', 'void', 'iron']:
        kernel_fn = SacredKernelLibrary.get_kernel_callable(name)
        K = kernel_fn(X, X)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > -1e-6), f"Kernel {name} not PSD: min eigenvalue = {eigenvalues.min()}"


def test_sacred_kernel_list():
    """Kernel listing returns expected kernels."""
    from l104_ml_engine.sacred_kernels import SacredKernelLibrary
    kernels = SacredKernelLibrary.list_kernels()
    assert len(kernels) == 6
    assert 'phi' in kernels
    assert 'composite' in kernels


# ═══════════════════════════════════════════════════════════════════════════════
#  SVM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_binary_data(n=50):
    """Generate simple binary classification data."""
    rng = np.random.default_rng(104)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_svm_classify_basic():
    """L104SVM classification with sacred RBF kernel."""
    from l104_ml_engine.svm import L104SVM
    X, y = _make_binary_data(60)
    svm = L104SVM(mode='classify', kernel='phi_kernel')
    svm.fit(X[:40], y[:40])
    preds = svm.predict(X[40:])
    acc = np.mean(preds == y[40:])
    assert acc > 0.5, f"SVM accuracy too low: {acc}"


def test_svm_regress():
    """L104SVM regression mode."""
    from l104_ml_engine.svm import L104SVM
    rng = np.random.default_rng(104)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + rng.normal(0, 0.1, 50)
    svm = L104SVM(mode='regress', kernel='rbf')
    svm.fit(X[:35], y[:35])
    preds = svm.predict(X[35:])
    assert preds.shape == (15,)


def test_svm_sacred_score():
    """Sacred score returns expected keys."""
    from l104_ml_engine.svm import L104SVM
    X, y = _make_binary_data(40)
    svm = L104SVM(mode='classify', kernel='phi_kernel')
    svm.fit(X, y)
    result = svm.sacred_score(X, y)
    assert 'score' in result
    assert 'god_code_alignment' in result
    assert 'sacred_resonance' in result
    assert 0 <= result['god_code_alignment'] <= 1


def test_svm_ensemble():
    """SVMEnsemble produces predictions."""
    from l104_ml_engine.svm import SVMEnsemble
    X, y = _make_binary_data(60)
    ensemble = SVMEnsemble(kernels=['phi_kernel', 'rbf'])
    ensemble.fit(X[:40], y[:40])
    preds = ensemble.predict(X[40:])
    assert len(preds) == 20
    preds2, conf = ensemble.predict_with_confidence(X[40:])
    assert len(conf) == 20
    assert np.all(conf >= 0) and np.all(conf <= 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSIFIER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_random_forest_sacred():
    """L104RandomForest with 104 estimators."""
    from l104_ml_engine.classifiers import L104RandomForest
    X, y = _make_binary_data(60)
    rf = L104RandomForest()
    rf.fit(X[:40], y[:40])
    assert rf._model.n_estimators == 104
    importances = rf.feature_importance_sacred()
    assert abs(sum(importances['raw']) - 1.0) < 0.01


def test_gradient_boosting_sacred():
    """L104GradientBoosting with sacred learning rate."""
    from l104_ml_engine.classifiers import L104GradientBoosting
    X, y = _make_binary_data(60)
    gb = L104GradientBoosting()
    gb.fit(X[:40], y[:40])
    acc = gb.score(X[40:], y[40:])
    assert acc > 0.4


def test_ensemble_classifier():
    """L104EnsembleClassifier meta-ensemble."""
    from l104_ml_engine.classifiers import L104EnsembleClassifier
    X, y = _make_binary_data(80)
    ec = L104EnsembleClassifier(include_svm=True, include_quantum=False)
    ec.fit(X[:60], y[:60])
    preds = ec.predict(X[60:])
    assert len(preds) == 20
    alignment = ec.sacred_alignment(X[60:], y[60:])
    assert 0 <= alignment <= 1


# ═══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_kmeans_phi_spiral():
    """L104KMeans with PHI-spiral centroid initialization."""
    from l104_ml_engine.clustering import L104KMeans
    rng = np.random.default_rng(104)
    X = rng.normal(size=(100, 4))
    km = L104KMeans(n_clusters=5, init='phi_spiral')
    km.fit(X)
    labels = km.predict(X)
    assert len(set(labels)) > 1
    centers = km.cluster_centers()
    assert centers.shape == (5, 4)


def test_dbscan_void():
    """L104DBSCAN with VOID_CONSTANT eps."""
    from l104_ml_engine.clustering import L104DBSCAN
    rng = np.random.default_rng(104)
    X = np.vstack([rng.normal(0, 0.3, (30, 3)), rng.normal(3, 0.3, (30, 3))])
    db = L104DBSCAN()
    db.fit(X)
    assert db.n_clusters >= 1


def test_spectral_phi_affinity():
    """L104SpectralClustering with PHI affinity."""
    from l104_ml_engine.clustering import L104SpectralClustering
    rng = np.random.default_rng(104)
    X = rng.normal(size=(30, 3))
    sc = L104SpectralClustering(n_clusters=3)
    sc.fit(X)
    labels = sc.predict(X)
    assert len(labels) == 30


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM SVM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_quantum_svm_fit_predict():
    """QuantumSVM trains and predicts with quantum kernel."""
    from l104_ml_engine.quantum_svm import QuantumSVM
    X, y = _make_binary_data(30)
    qsvm = QuantumSVM(n_qubits=3, kernel_type='sacred')
    qsvm.fit(X[:20], y[:20])
    preds = qsvm.predict(X[20:])
    assert len(preds) == 10
    status = qsvm.status()
    assert status['fitted'] is True
    assert status['n_qubits'] == 3


def test_quantum_svm_kernel_matrix():
    """QuantumSVM kernel matrix has correct shape."""
    from l104_ml_engine.quantum_svm import QuantumSVM
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 3))
    qsvm = QuantumSVM(n_qubits=3)
    K = qsvm.quantum_kernel_matrix(X)
    assert K.shape == (10, 10)
    # Diagonal should be ~1 (self-similarity)
    assert np.all(np.diag(K) > 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE SYNTHESIS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_feature_extractor_dimensions():
    """CrossEngineFeatureExtractor produces 50-dimensional features."""
    from l104_ml_engine.knowledge_synthesis import CrossEngineFeatureExtractor
    fe = CrossEngineFeatureExtractor()
    features = fe.extract_all("print('hello')")
    assert len(features) == 50
    names = fe.feature_names()
    assert len(names) == 50


def test_knowledge_synthesizer():
    """KnowledgeSynthesizer produces valid synthesis report."""
    from l104_ml_engine.knowledge_synthesis import KnowledgeSynthesizer
    ks = KnowledgeSynthesizer()
    result = ks.synthesize("test query")
    assert 'coherence_score' in result
    assert 'availability_score' in result
    assert 'sacred_alignment' in result
    assert 0 <= result['coherence_score'] <= 1


# ═══════════════════════════════════════════════════════════════════════════════
#  ML ENGINE SINGLETON TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_ml_engine_singleton():
    """get_engine() returns same instance."""
    from l104_ml_engine import get_engine
    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2


def test_ml_engine_status():
    """MLEngine.status() returns expected structure."""
    from l104_ml_engine import ml_engine
    status = ml_engine.status()
    assert 'version' in status
    assert status['version'] == '1.0.0'
    assert 'classical' in status
    assert 'clustering' in status
    assert 'quantum' in status
    assert 'synthesis' in status
    assert 'sacred_constants' in status
    assert abs(status['sacred_constants']['GOD_CODE'] - GOD_CODE) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
