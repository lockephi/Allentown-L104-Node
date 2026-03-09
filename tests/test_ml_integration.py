"""
===============================================================================
L104 ML ENGINE — INTEGRATION TESTS v1.0.0
===============================================================================
Tests ML engine integration hooks across ASI, pipeline, consciousness,
optimization, code engine, precognition, and numerical engine subsystems.

Run: python -m pytest tests/test_ml_integration.py -v
===============================================================================
"""

import numpy as np
import pytest

# Sacred constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492612


# ═══════════════════════════════════════════════════════════════════════════════
#  ASI ML SCORING INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_asi_ml_scoring_dimensions():
    """ASI compute_asi_score returns a valid float with ML dimensions factored in."""
    from l104_asi.core import ASICore
    core = ASICore()
    result = core.compute_asi_score()
    # compute_asi_score returns a float
    assert isinstance(result, (int, float))
    assert 0.0 <= result <= 10000.0  # Score is non-negative


def test_asi_ml_engine_lazy_load():
    """ASI core lazily loads ml_engine without import errors."""
    from l104_asi.core import ASICore
    core = ASICore()
    scores = core._compute_ml_engine_scores()
    assert isinstance(scores, dict)
    assert len(scores) == 3
    for key in ('ml_svm_coherence', 'ml_classifier_confidence', 'ml_knowledge_synthesis'):
        assert key in scores
        assert 0.0 <= scores[key] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ML ROUTER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_ml_pipeline_router_init():
    """MLPipelineRouter initializes with base router."""
    from l104_asi.pipeline import MLPipelineRouter
    router = MLPipelineRouter()
    status = router.get_status()
    assert status['type'] == 'MLPipelineRouter'
    assert status['ml_fitted'] is False
    assert status['training_samples'] == 0


def test_ml_pipeline_router_fallback():
    """MLPipelineRouter falls back to base TF-IDF before training."""
    from l104_asi.pipeline import MLPipelineRouter
    router = MLPipelineRouter()
    routes = router.route("calculate the god_code alignment")
    assert len(routes) > 0
    # Should have valid (name, score) tuples
    for name, score in routes:
        assert isinstance(name, str)
        assert isinstance(score, (int, float))


def test_ml_pipeline_router_training():
    """MLPipelineRouter trains after sufficient experiences."""
    from l104_asi.pipeline import MLPipelineRouter
    router = MLPipelineRouter()

    # Record experiences across 3 subsystems
    queries = {
        'direct_solution': ["calculate phi", "compute god_code", "solve math", "what is 2+2",
                            "compute fibonacci", "calculate entropy", "math problem",
                            "solve equation", "phi value", "god_code constant"],
        'asi_harness': ["analyze code", "optimize function", "debug class",
                        "refactor method", "code review", "fix bug",
                        "program error", "code quality", "function test", "compile code"],
        'knowledge': ["explain consciousness", "what is quantum", "describe entropy",
                      "how does gravity work", "knowledge about physics",
                      "tell me about fibonacci", "describe golden ratio",
                      "what is thermodynamics", "explain relativity", "history of science"],
    }

    for subsystem, qs in queries.items():
        for q in qs:
            router.record_experience(q, subsystem, success=True)

    success = router.train()
    assert success is True
    assert router._ml_fitted is True

    # Now routes should use ML model
    routes = router.route("calculate the value of phi squared")
    assert len(routes) > 0


def test_ml_pipeline_router_feedback():
    """MLPipelineRouter propagates feedback to base router."""
    from l104_asi.pipeline import MLPipelineRouter
    router = MLPipelineRouter()
    router.feedback('direct_solution', ['phi', 'calculate'], True, 0.9)
    assert router._base_router._feedback_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSCIOUSNESS ML TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_consciousness_ml_tests_registered():
    """Consciousness verifier has ML test functions registered."""
    from l104_asi.consciousness import ConsciousnessVerifier
    assert 'ml_consciousness_classifier' in ConsciousnessVerifier.TESTS
    assert 'ml_qualia_clustering' in ConsciousnessVerifier.TESTS
    assert len(ConsciousnessVerifier.TESTS) >= 18


def test_consciousness_ml_classifier():
    """ML consciousness classifier produces valid score."""
    from l104_asi.consciousness import ConsciousnessVerifier
    verifier = ConsciousnessVerifier()
    verifier.run_all_tests()
    # ml_consciousness_classifier and ml_qualia_clustering should be in results
    if 'ml_consciousness_classifier' in verifier.test_results:
        score = verifier.test_results['ml_consciousness_classifier']
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  REASONING QUANTUM SVM SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def test_quantum_svm_solver_callable():
    """QuantumSVMSolver is callable and returns expected dict structure."""
    from l104_asi.reasoning import QuantumSVMSolver
    solver = QuantumSVMSolver()
    result = solver({'query': 'test reasoning problem', 'type': 'classification'})
    assert 'solution' in result
    assert 'confidence' in result
    assert 0.0 <= result['confidence'] <= 1.0


def test_quantum_svm_solver_features():
    """QuantumSVMSolver extracts features from input."""
    from l104_asi.reasoning import QuantumSVMSolver
    solver = QuantumSVMSolver()
    features = solver._extract_features("test quantum consciousness sacred phi resonance")
    assert len(features) == 8
    assert all(isinstance(f, float) for f in features)


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZATION ML SURROGATE
# ═══════════════════════════════════════════════════════════════════════════════

def test_optimization_ml_surrogate():
    """optimize_with_ml_surrogate runs and returns valid result."""
    from l104_intellect.optimization import L104DynamicOptimizationEngine

    def simple_objective(x):
        return sum(xi ** 2 for xi in x)

    engine = L104DynamicOptimizationEngine()
    result = engine.optimize_with_ml_surrogate(
        objective_fn=simple_objective,
        bounds=[(-2, 2)] * 3,
        n_initial=15,
        max_iter=3,
    )
    assert 'best_x' in result
    assert 'best_y' in result
    assert 'n_evaluations' in result
    assert len(result['best_x']) == 3


# ═══════════════════════════════════════════════════════════════════════════════
#  CODE ENGINE ML CLASSIFY
# ═══════════════════════════════════════════════════════════════════════════════

def test_code_engine_ml_classify():
    """Code engine ml_code_quality_classify returns valid dict."""
    from l104_code_engine.quantum import QuantumCodeIntelligenceCore
    core = QuantumCodeIntelligenceCore()
    result = core.ml_code_quality_classify("def hello():\n    return 'world'")
    assert 'quality_score' in result
    assert 'god_code_alignment' in result
    assert 'features' in result
    assert 0.0 <= result['quality_score'] <= 1.0
    assert 0.0 <= result['god_code_alignment'] <= 1.0


def test_code_engine_ml_method_exists():
    """ml_code_quality_classify method exists on QuantumCodeIntelligenceCore."""
    from l104_code_engine.quantum import QuantumCodeIntelligenceCore
    core = QuantumCodeIntelligenceCore()
    assert hasattr(core, 'ml_code_quality_classify')
    assert callable(core.ml_code_quality_classify)


# ═══════════════════════════════════════════════════════════════════════════════
#  PRECOGNITION ML WEIGHT LEARNER
# ═══════════════════════════════════════════════════════════════════════════════

def test_precognition_ml_weight_learner():
    """MLEnsembleWeightLearner initializes and produces default weights."""
    from l104_data_precognition import MLEnsembleWeightLearner
    learner = MLEnsembleWeightLearner()
    weights = learner.predict_weights([1.0, 2.0, 3.0])
    assert isinstance(weights, dict)


def test_precognition_ml_training():
    """MLEnsembleWeightLearner can train on synthetic predictions."""
    from l104_data_precognition import MLEnsembleWeightLearner
    learner = MLEnsembleWeightLearner()

    algo_names = ['ema', 'arima', 'spectral', 'wavelet', 'bayesian', 'fourier', 'quantum']
    rng = np.random.default_rng(104)
    for _ in range(20):
        preds = {name: float(rng.normal(0, 1)) for name in algo_names}
        actual = float(rng.normal(0, 1))
        learner.record_prediction(preds, actual)

    weights = learner.learn_weights()
    assert isinstance(weights, dict)
    if weights:
        total = sum(weights.values())
        assert total > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  NUMERICAL ENGINE ML DRIFT PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

def test_ml_drift_predictor_init():
    """MLDriftPredictor initializes correctly."""
    from l104_numerical_engine.monitor import MLDriftPredictor
    predictor = MLDriftPredictor()
    status = predictor.status()
    assert status['type'] == 'MLDriftPredictor'
    assert status['fitted'] is False
    assert status['training_samples'] == 0


def test_ml_drift_predictor_no_data():
    """MLDriftPredictor returns no_data prediction when untrained."""
    from l104_numerical_engine.monitor import MLDriftPredictor
    from l104_numerical_engine.lattice import TokenLatticeEngine
    predictor = MLDriftPredictor()
    lattice = TokenLatticeEngine()
    result = predictor.predict_drift({}, lattice, 0)
    assert result['source'] == 'no_data'
    assert result['predicted_delta'] == 0.0


def test_ml_drift_predictor_heuristic():
    """MLDriftPredictor falls back to heuristic with insufficient data."""
    from l104_numerical_engine.monitor import MLDriftPredictor
    from l104_numerical_engine.lattice import TokenLatticeEngine
    predictor = MLDriftPredictor()
    lattice = TokenLatticeEngine()

    # Record a few observations (less than MIN_HISTORY)
    for i in range(5):
        predictor.record_observation(
            {'gate_count': i, 'link_count': i, 'token_count': 10},
            lattice, i, 0.01 * i
        )

    result = predictor.predict_drift({}, lattice, 5)
    assert result['source'] == 'heuristic_last_delta'
    assert result['confidence'] == 0.3


def test_ml_drift_predictor_training():
    """MLDriftPredictor trains and predicts after sufficient history."""
    from l104_numerical_engine.monitor import MLDriftPredictor
    from l104_numerical_engine.lattice import TokenLatticeEngine
    predictor = MLDriftPredictor()
    lattice = TokenLatticeEngine()

    rng = np.random.default_rng(104)
    for i in range(20):
        capacity = {
            'gate_count': float(10 + i), 'link_count': float(5 + i),
            'token_count': float(len(lattice.tokens)), 'gate_health': 0.8,
            'link_fidelity': 0.9, 'test_pass_rate': 0.95,
            'coherence': 0.99, 'total_usages': float(100 + i * 10),
        }
        delta = float(rng.normal(0.01, 0.005))
        predictor.record_observation(capacity, lattice, i, delta)

    success = predictor.train()
    assert success is True

    result = predictor.predict_drift(
        {'gate_count': 30, 'link_count': 25, 'token_count': 50,
         'gate_health': 0.8, 'link_fidelity': 0.9, 'test_pass_rate': 0.95,
         'coherence': 0.99, 'total_usages': 300},
        lattice, 20
    )
    assert result['source'] == 'ml_svm'
    assert result['confidence'] > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM GATE ENGINE EXTENSIONS
# ═══════════════════════════════════════════════════════════════════════════════

def test_quantum_ml_new_ansatz_types():
    """QuantumML has new VQC_CLASSIFIER and SVM_FEATURE_ENCODER ansatz types."""
    from l104_quantum_gate_engine.quantum_ml import AnsatzType
    assert hasattr(AnsatzType, 'VQC_CLASSIFIER')
    assert hasattr(AnsatzType, 'SVM_FEATURE_ENCODER')


def test_quantum_ml_new_kernel_types():
    """QuantumML has new kernel types for ML integration."""
    from l104_quantum_gate_engine.quantum_ml import KernelType
    assert hasattr(KernelType, 'PHI_ENCODED')
    assert hasattr(KernelType, 'GOD_CODE_PHASE')
    assert hasattr(KernelType, 'IRON_LATTICE')
    assert hasattr(KernelType, 'HARMONIC_FOURIER')


def test_quantum_svm_trainer_exists():
    """QuantumSVMTrainer class is available in quantum_ml."""
    from l104_quantum_gate_engine.quantum_ml import QuantumSVMTrainer
    trainer = QuantumSVMTrainer(num_qubits=3)
    assert trainer.num_qubits == 3


def test_svm_encoding_gate():
    """SVM_ENCODING_GATE produces valid gate."""
    from l104_quantum_gate_engine.gates import SVM_ENCODING_GATE
    gate = SVM_ENCODING_GATE(0.5)
    assert gate is not None
    assert gate.num_qubits == 1


def test_classifier_measurement_gate():
    """CLASSIFIER_MEASUREMENT_GATE produces valid gate."""
    from l104_quantum_gate_engine.gates import CLASSIFIER_MEASUREMENT_GATE
    gate = CLASSIFIER_MEASUREMENT_GATE(4)
    assert gate is not None
    assert gate.num_qubits == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  CROSS-ENGINE IMPORT SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_ml_engine_import_smoke():
    """All ML engine public exports are importable."""
    from l104_ml_engine import (
        ml_engine, get_engine, MLEngine,
        L104SVM, SVMEnsemble,
        L104RandomForest, L104GradientBoosting, L104EnsembleClassifier,
        L104KMeans, L104DBSCAN, L104SpectralClustering,
        QuantumSVM,
        VariationalQuantumClassifier, QuantumNearestNeighbor, QuantumEnsembleClassifier,
        CrossEngineFeatureExtractor, KnowledgeSynthesizer,
    )
    assert ml_engine is not None
    assert get_engine() is ml_engine


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
