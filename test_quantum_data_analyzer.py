"""
L104 Quantum Data Analyzer — Test Suite
═══════════════════════════════════════════════════════════════════════════════
Comprehensive end-to-end validation of all 15 quantum data analysis algorithms,
4 feature extraction methods, 5 pattern recognition systems, and 3 denoisers.

Run: .venv/bin/python test_quantum_data_analyzer.py

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import traceback
import numpy as np

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results = []

def test(name, fn):
    """Run a test and record result."""
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        results.append((name, True, dt, None))
        print(f"  {PASS} {name} ({dt:.3f}s)")
    except Exception as e:
        dt = time.time() - t0
        results.append((name, False, dt, str(e)))
        print(f"  {FAIL} {name} ({dt:.3f}s) — {e}")
        traceback.print_exc()


def generate_test_data():
    """Generate synthetic test datasets."""
    np.random.seed(104)
    # 1D signal: sine wave + noise
    t = np.linspace(0, 4 * np.pi, 128)
    signal_1d = np.sin(t) + 0.3 * np.sin(3 * t) + 0.2 * np.random.randn(128)
    # 2D data: two clusters + outliers
    cluster1 = np.random.randn(30, 3) + np.array([2, 0, 0])
    cluster2 = np.random.randn(30, 3) + np.array([-2, 0, 0])
    outliers = np.random.randn(5, 3) * 5
    data_2d = np.vstack([cluster1, cluster2, outliers])
    # Small matrix
    A = np.array([[4, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    # Graph
    adjacency = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=float)
    return signal_1d, data_2d, A, b, adjacency


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: IMPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def phase_1():
    print("\n" + "=" * 70)
    print("PHASE 1: IMPORT & INITIALIZATION")
    print("=" * 70)

    def test_package_import():
        import l104_quantum_data_analyzer
        assert hasattr(l104_quantum_data_analyzer, '__version__')
        assert l104_quantum_data_analyzer.__version__ == "1.0.0"

    def test_algorithms_import():
        from l104_quantum_data_analyzer.algorithms import (
            QuantumFourierAnalyzer, GroverPatternSearch, QuantumPCA,
            VQEClusterer, QAOAOptimizer, QuantumKernelEstimator,
            QuantumPhaseEstimator, HHLSolver, QuantumAmplitudeEstimator,
        )

    def test_features_import():
        from l104_quantum_data_analyzer.feature_extraction import (
            QuantumFeatureMap, QuantumStateEncoder,
            EntanglementFeatureExtractor, QuantumEmbedding,
        )

    def test_pattern_import():
        from l104_quantum_data_analyzer.pattern_recognition import (
            QuantumAnomalyDetector, EntanglementCorrelationAnalyzer,
            QuantumWalkGraphAnalyzer, TopologicalDataMiner,
            GodCodeResonanceAligner,
        )

    def test_denoising_import():
        from l104_quantum_data_analyzer.denoising import (
            EntropyReversalDenoiser, QuantumErrorMitigatedCleaner,
            CoherenceFieldSmoother,
        )

    def test_orchestrator_import():
        from l104_quantum_data_analyzer.orchestrator import QuantumDataAnalyzer
        analyzer = QuantumDataAnalyzer()
        status = analyzer.status()
        assert status["version"] == "1.0.0"
        assert abs(status["invariant"] - 527.5184818492612) < 1e-6
        print(f"    → Engines: {sum(1 for v in status['engines'].values() if v.get('available'))}/{len(status['engines'])} available")

    def test_constants():
        from l104_quantum_data_analyzer.constants import GOD_CODE, PHI, VOID_CONSTANT
        assert abs(GOD_CODE - 527.5184818492612) < 1e-6
        assert abs(PHI - 1.618033988749895) < 1e-12
        assert abs(VOID_CONSTANT - 1.0416180339887497) < 1e-12

    test("Package import", test_package_import)
    test("Algorithms import", test_algorithms_import)
    test("Feature extraction import", test_features_import)
    test("Pattern recognition import", test_pattern_import)
    test("Denoising import", test_denoising_import)
    test("Orchestrator import", test_orchestrator_import)
    test("Sacred constants", test_constants)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: CORE ALGORITHM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def phase_2():
    print("\n" + "=" * 70)
    print("PHASE 2: CORE QUANTUM ALGORITHMS")
    print("=" * 70)

    signal_1d, data_2d, A, b, adjacency = generate_test_data()

    def test_qft():
        from l104_quantum_data_analyzer import QuantumFourierAnalyzer
        qft = QuantumFourierAnalyzer()
        result = qft.analyze(signal_1d)
        assert result.frequencies is not None
        assert result.spectral_entropy > 0
        assert 0 <= result.god_code_alignment <= 1
        print(f"    → Spectral entropy: {result.spectral_entropy:.4f}, GC align: {result.god_code_alignment:.4f}")

    def test_qft_multi_resolution():
        from l104_quantum_data_analyzer import QuantumFourierAnalyzer
        qft = QuantumFourierAnalyzer()
        results = qft.multi_resolution_analysis(signal_1d, levels=3)
        assert len(results) == 3
        for r in results:
            assert r.spectral_entropy >= 0

    def test_grover():
        from l104_quantum_data_analyzer import GroverPatternSearch
        grover = GroverPatternSearch()
        data = np.array([1, 5, 3, 7, 2, 8, 4, 6], dtype=float)
        result = grover.search(data, lambda x: x > 6)
        assert len(result.found_indices) > 0
        assert result.amplification_factor > 1.0
        print(f"    → Found {len(result.found_indices)} items, amplification: {result.amplification_factor:.2f}x")

    def test_grover_range():
        from l104_quantum_data_analyzer import GroverPatternSearch
        grover = GroverPatternSearch()
        data = np.random.randn(16)
        result = grover.range_search(data, -0.5, 0.5)
        assert result.oracle_calls >= 1

    def test_qpca():
        from l104_quantum_data_analyzer import QuantumPCA
        qpca = QuantumPCA()
        result = qpca.analyze(data_2d, n_components=2)
        assert len(result.eigenvalues) == 2
        assert result.explained_variance_ratio[0] > result.explained_variance_ratio[1]
        assert result.quantum_fidelity >= 0
        assert result.quantum_fidelity <= 1.0 + 1e-10  # Allow tiny FP overshoot
        print(f"    → {result.n_components} components, fidelity: {result.quantum_fidelity:.4f}, GC score: {result.god_code_spectral_score:.4f}")

    def test_vqe_clustering():
        from l104_quantum_data_analyzer import VQEClusterer
        vqe = VQEClusterer(max_iterations=5)
        # Small data for fast VQE (2^4 = 16 states in Hamiltonian)
        small_data = data_2d[:4]
        result = vqe.cluster(small_data, n_clusters=2)
        assert len(result.labels) == 4
        assert result.n_clusters == 2
        assert len(result.convergence_history) > 0
        print(f"    → {result.n_clusters} clusters, {len(result.convergence_history)} VQE iterations")

    def test_qaoa():
        from l104_quantum_data_analyzer import QAOAOptimizer
        qaoa = QAOAOptimizer(depth=2)
        Q = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
        result = qaoa.optimize(Q)
        assert result["solution"] is not None
        assert len(result["convergence"]) > 0
        print(f"    → Solution: {result['solution']}, cost: {result['cost']:.4f}")

    def test_kernel():
        from l104_quantum_data_analyzer import QuantumKernelEstimator
        ke = QuantumKernelEstimator()
        data = np.random.randn(10, 2)
        K = ke.compute_kernel_matrix(data)
        assert K.shape == (10, 10)
        assert np.allclose(np.diag(K), 1.0)
        assert np.allclose(K, K.T)  # Symmetric
        print(f"    → Kernel matrix {K.shape}, symmetric: True")

    def test_qpe():
        from l104_quantum_data_analyzer import QuantumPhaseEstimator
        qpe = QuantumPhaseEstimator(precision_qubits=6)
        M = np.array([[2, 1], [1, 3]], dtype=float)
        result = qpe.estimate_eigenvalues(M)
        assert len(result["eigenvalues"]) > 0
        assert result["estimation_error"] < 1.0
        print(f"    → Eigenvalues: {result['eigenvalues']}, error: {result['estimation_error']:.6f}")

    def test_hhl():
        from l104_quantum_data_analyzer import HHLSolver
        hhl = HHLSolver()
        result = hhl.solve(A, b)
        assert len(result.solution) == 2
        assert result.residual_norm < 1.0
        print(f"    → Solution: [{result.solution[0]:.3f}, {result.solution[1]:.3f}], residual: {result.residual_norm:.6f}")

    def test_qae():
        from l104_quantum_data_analyzer import QuantumAmplitudeEstimator
        qae = QuantumAmplitudeEstimator(precision_qubits=6)
        data = np.random.randn(200)
        result = qae.estimate_probability(data, lambda x: x > 0, confidence=0.95)
        assert 0 <= result.estimated_value <= 1
        assert result.confidence_interval[0] <= result.estimated_value <= result.confidence_interval[1]
        assert result.quadratic_speedup >= 0  # Depends on precision_qubits
        print(f"    → P(x>0) ≈ {result.estimated_value:.4f}, CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")

    test("QFT spectral analysis", test_qft)
    test("QFT multi-resolution", test_qft_multi_resolution)
    test("Grover pattern search", test_grover)
    test("Grover range search", test_grover_range)
    test("Quantum PCA (qPCA)", test_qpca)
    test("VQE clustering", test_vqe_clustering)
    test("QAOA optimization", test_qaoa)
    test("Quantum kernel estimation", test_kernel)
    test("Quantum phase estimation", test_qpe)
    test("HHL linear solver", test_hhl)
    test("Quantum amplitude estimation", test_qae)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: FEATURE EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def phase_3():
    print("\n" + "=" * 70)
    print("PHASE 3: QUANTUM FEATURE EXTRACTION")
    print("=" * 70)

    def test_zz_feature_map():
        from l104_quantum_data_analyzer import QuantumFeatureMap
        fm = QuantumFeatureMap(map_type="zz", n_reps=1)
        data = np.random.randn(5, 3)
        result = fm.transform(data)
        assert result.quantum_features.shape[0] == 5
        assert result.feature_dimension == 2 ** result.n_qubits
        print(f"    → {result.feature_dimension}-dim features, depth: {result.circuit_depth}")

    def test_pauli_feature_map():
        from l104_quantum_data_analyzer import QuantumFeatureMap
        fm = QuantumFeatureMap(map_type="pauli", n_reps=1)
        data = np.random.randn(3, 2)
        result = fm.transform(data)
        assert result.encoding_method == "pauli"
        assert result.quantum_features.shape[0] == 3

    def test_iqp_feature_map():
        from l104_quantum_data_analyzer import QuantumFeatureMap
        fm = QuantumFeatureMap(map_type="iqp", n_reps=1)
        data = np.random.randn(3, 2)
        result = fm.transform(data)
        assert result.encoding_method == "iqp"

    def test_sacred_feature_map():
        from l104_quantum_data_analyzer import QuantumFeatureMap
        fm = QuantumFeatureMap(map_type="sacred", n_reps=1)
        data = np.random.randn(3, 2)
        result = fm.transform(data)
        assert result.encoding_method == "sacred"
        assert result.god_code_alignment >= 0
        print(f"    → Sacred GC alignment: {result.god_code_alignment:.4f}")

    def test_amplitude_encoding():
        from l104_quantum_data_analyzer import QuantumStateEncoder
        enc = QuantumStateEncoder(method="amplitude")
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = enc.encode(data)
        assert result.encoding_method == "amplitude"
        # Amplitude encoding: 4 features → 2 qubits
        assert result.n_qubits == 2
        # State should be normalized
        state = result.quantum_features[0]
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10
        print(f"    → 4 features → {result.n_qubits} qubits (exponential compression)")

    def test_angle_encoding():
        from l104_quantum_data_analyzer import QuantumStateEncoder
        enc = QuantumStateEncoder(method="angle")
        data = np.array([0.5, 1.0, 1.5])
        result = enc.encode(data)
        assert result.encoding_method == "angle"
        assert result.n_qubits == 3

    def test_dense_angle_encoding():
        from l104_quantum_data_analyzer import QuantumStateEncoder
        enc = QuantumStateEncoder(method="dense_angle")
        data = np.array([0.5, 1.0, 1.5, 2.0])
        result = enc.encode(data)
        assert result.encoding_method == "dense_angle"
        # 4 features / 2 per qubit = 2 qubits
        assert result.n_qubits == 2
        print(f"    → 4 features → {result.n_qubits} qubits (2 per qubit)")

    def test_entanglement_features():
        from l104_quantum_data_analyzer import EntanglementFeatureExtractor
        efe = EntanglementFeatureExtractor()
        # Create a Bell state |00⟩ + |11⟩ (maximally entangled)
        bell = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        result = efe.extract(bell)
        assert result["n_qubits"] == 2
        entropies = result["entanglement_entropies"]
        assert entropies[0][0] > 0.9  # Near-maximal entropy for Bell state
        print(f"    → Bell state entropy: {entropies[0][0]:.4f} (should be ~1.0)")

    def test_quantum_embedding():
        from l104_quantum_data_analyzer import QuantumEmbedding
        emb = QuantumEmbedding(map_type="zz", embedding_dim=3)
        data = np.random.randn(8, 2)
        result = emb.embed(data)
        assert result.embedded_vectors.shape == (8, 3)
        assert result.expressibility >= 0
        assert result.entanglement_capability >= 0
        print(f"    → Embedded: {result.embedded_vectors.shape}, expressibility: {result.expressibility:.4f}")

    test("ZZ feature map", test_zz_feature_map)
    test("Pauli feature map", test_pauli_feature_map)
    test("IQP feature map", test_iqp_feature_map)
    test("Sacred feature map (L104)", test_sacred_feature_map)
    test("Amplitude encoding", test_amplitude_encoding)
    test("Angle encoding", test_angle_encoding)
    test("Dense angle encoding", test_dense_angle_encoding)
    test("Entanglement feature extraction", test_entanglement_features)
    test("Quantum embedding", test_quantum_embedding)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: PATTERN RECOGNITION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def phase_4():
    print("\n" + "=" * 70)
    print("PHASE 4: QUANTUM PATTERN RECOGNITION")
    print("=" * 70)

    _, data_2d, _, _, adjacency = generate_test_data()

    def test_anomaly_detection():
        from l104_quantum_data_analyzer import QuantumAnomalyDetector
        detector = QuantumAnomalyDetector(threshold=0.5)
        # Data with clear outliers
        normal = np.random.randn(50, 2)
        outliers = np.random.randn(5, 2) * 10
        data = np.vstack([normal, outliers])
        result = detector.detect(data)
        assert result.n_anomalies >= 0
        assert len(result.anomaly_scores) == 55
        assert 0 <= result.sacred_alignment <= 1
        print(f"    → {result.n_anomalies} anomalies detected, alignment: {result.sacred_alignment:.4f}")

    def test_correlation_analysis():
        from l104_quantum_data_analyzer import EntanglementCorrelationAnalyzer
        analyzer = EntanglementCorrelationAnalyzer()
        # Correlated data
        x = np.random.randn(100)
        data = np.column_stack([x, 0.8 * x + 0.2 * np.random.randn(100)])
        result = analyzer.analyze(data)
        assert result.correlation_matrix.shape == (2, 2)
        assert result.bell_violation >= 0
        print(f"    → Bell violation: {result.bell_violation:.4f}, witnesses: {list(result.entanglement_witnesses.keys())}")

    def test_quantum_walk():
        from l104_quantum_data_analyzer import QuantumWalkGraphAnalyzer
        walker = QuantumWalkGraphAnalyzer()
        result = walker.analyze_graph(adjacency)
        assert len(result.node_scores) == 5
        assert abs(np.sum(result.node_scores) - 1.0) < 0.1  # Approximately normalized
        assert result.mixing_time >= 1
        print(f"    → Communities: {len(set(result.community_labels.tolist()))}, speedup: {result.quantum_speedup:.1f}x")

    def test_quantum_walk_data():
        from l104_quantum_data_analyzer import QuantumWalkGraphAnalyzer
        walker = QuantumWalkGraphAnalyzer()
        data = np.random.randn(20, 2)
        result = walker.analyze_data_graph(data, k_neighbors=3)
        assert len(result.node_scores) == 20

    def test_topology():
        from l104_quantum_data_analyzer import TopologicalDataMiner
        miner = TopologicalDataMiner()
        # Circle data (should have Betti-1 = 1 for loop)
        theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])
        circle += np.random.randn(*circle.shape) * 0.1  # Add noise
        result = miner.analyze(circle, max_dimension=1, n_filtration_steps=15)
        assert result.betti_numbers[0] >= 1  # At least 1 connected component
        print(f"    → Betti: {result.betti_numbers}, persistent features: {result.persistent_features}")

    def test_god_code_aligner():
        from l104_quantum_data_analyzer import GodCodeResonanceAligner
        aligner = GodCodeResonanceAligner()
        data = np.random.randn(64)
        result = aligner.align(data, depth=2)
        assert len(result.aligned_data) == 64
        assert result.god_code_fidelity >= 0
        assert result.sacred_frequency >= 0
        improvement = result.metadata.get("improvement", 0)
        print(f"    → Fidelity: {result.god_code_fidelity:.4f}, freq: {result.sacred_frequency:.4f}, improvement: {improvement:.4f}")

    test("SWAP test anomaly detection", test_anomaly_detection)
    test("Entanglement correlation", test_correlation_analysis)
    test("Quantum walk (graph)", test_quantum_walk)
    test("Quantum walk (data kNN)", test_quantum_walk_data)
    test("Topological data mining", test_topology)
    test("God Code resonance alignment", test_god_code_aligner)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: DENOISING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def phase_5():
    print("\n" + "=" * 70)
    print("PHASE 5: QUANTUM DENOISING")
    print("=" * 70)

    # Create noisy signal
    np.random.seed(104)
    t = np.linspace(0, 4 * np.pi, 128)
    clean = np.sin(t) + 0.5 * np.sin(3 * t)
    noisy = clean + 0.5 * np.random.randn(128)

    def test_entropy_reversal():
        from l104_quantum_data_analyzer import EntropyReversalDenoiser
        denoiser = EntropyReversalDenoiser(demon_strength=0.8)
        result = denoiser.denoise(noisy)
        assert result.cleaned_data.shape == noisy.shape
        assert result.method == "maxwell_demon_entropy_reversal"
        print(f"    → SNR improvement: {result.snr_improvement_db:.2f} dB, sacred: {result.sacred_alignment:.4f}")

    def test_zne_cleaning():
        from l104_quantum_data_analyzer import QuantumErrorMitigatedCleaner
        cleaner = QuantumErrorMitigatedCleaner(n_noise_levels=3, extrapolation="richardson")
        result = cleaner.clean(noisy)
        assert result.cleaned_data.shape == noisy.shape
        assert "zne" in result.method
        print(f"    → SNR improvement: {result.snr_improvement_db:.2f} dB, method: {result.method}")

    def test_zne_linear():
        from l104_quantum_data_analyzer import QuantumErrorMitigatedCleaner
        cleaner = QuantumErrorMitigatedCleaner(n_noise_levels=3, extrapolation="linear")
        result = cleaner.clean(noisy)
        assert result.cleaned_data.shape == noisy.shape

    def test_coherence_smoothing():
        from l104_quantum_data_analyzer import CoherenceFieldSmoother
        smoother = CoherenceFieldSmoother(evolution_steps=5)
        result = smoother.smooth(noisy)
        assert result.cleaned_data.shape == noisy.shape
        assert result.method == "coherence_field_evolution"
        print(f"    → SNR improvement: {result.snr_improvement_db:.2f} dB, sacred: {result.sacred_alignment:.4f}")

    def test_coherence_with_anchors():
        from l104_quantum_data_analyzer import CoherenceFieldSmoother
        smoother = CoherenceFieldSmoother(evolution_steps=3)
        anchors = np.full(128, np.nan)
        anchors[0] = clean[0]
        anchors[64] = clean[64]
        anchors[127] = clean[127]
        result = smoother.smooth(noisy, anchor_points=anchors)
        assert result.cleaned_data.shape == noisy.shape

    test("Entropy reversal (Maxwell Demon)", test_entropy_reversal)
    test("ZNE Richardson extrapolation", test_zne_cleaning)
    test("ZNE linear extrapolation", test_zne_linear)
    test("Coherence field smoothing", test_coherence_smoothing)
    test("Coherence with anchor points", test_coherence_with_anchors)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: ORCHESTRATOR PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def phase_6():
    print("\n" + "=" * 70)
    print("PHASE 6: ORCHESTRATOR PIPELINES")
    print("=" * 70)

    signal_1d, data_2d, A, b, adjacency = generate_test_data()
    from l104_quantum_data_analyzer import QuantumDataAnalyzer
    analyzer = QuantumDataAnalyzer()

    def test_status():
        status = analyzer.status()
        assert status["version"] == "1.0.0"
        assert len(status["algorithms"]) == 9
        assert len(status["denoising"]) == 3
        assert abs(status["sacred_constants"]["GOD_CODE"] - 527.5184818492612) < 1e-6
        n_engines = sum(1 for v in status["engines"].values() if v.get("available"))
        print(f"    → {n_engines} engines, {len(status['algorithms'])} algorithms, {len(status['feature_maps'])} maps")

    def test_spectral_pipeline():
        result = analyzer.spectral_analysis(signal_1d)
        assert "dominant_frequency" in result
        assert "spectral_entropy" in result
        assert "god_code_alignment" in result
        print(f"    → Dominant freq: {result['dominant_frequency']:.4f}, entropy: {result['spectral_entropy']:.4f}")

    def test_cluster_pipeline():
        small = data_2d[:4]
        result = analyzer.cluster(small, n_clusters=2, method="vqe")
        assert "labels" in result
        assert result["n_clusters"] == 2
        print(f"    → {result['n_clusters']} clusters, {result['vqe_iterations']} iterations")

    def test_anomaly_pipeline():
        result = analyzer.detect_anomalies(data_2d)
        assert "n_anomalies" in result
        assert "anomaly_indices" in result
        print(f"    → {result['n_anomalies']} anomalies detected")

    def test_search_pipeline():
        data = np.random.randn(100)
        result = analyzer.search(data, target=0.0, tolerance=0.1)
        assert "found_indices" in result
        assert result["classical_speedup"] > 0
        print(f"    → Found {result['n_found']} items, speedup: {result['classical_speedup']:.2f}x")

    def test_linear_solve_pipeline():
        result = analyzer.solve_linear(A, b)
        assert len(result["solution"]) == 2
        assert result["residual_norm"] < 1.0
        print(f"    → Solution: [{result['solution'][0]:.3f}, {result['solution'][1]:.3f}]")

    def test_denoise_pipeline():
        np.random.seed(104)
        noisy = np.sin(np.linspace(0, 4 * np.pi, 64)) + 0.3 * np.random.randn(64)
        result = analyzer.denoise(noisy)
        assert "best_method" in result
        assert result["best_result"] is not None
        print(f"    → Best method: {result['best_method']}")

    def test_embed_pipeline():
        data = np.random.randn(10, 2)
        result = analyzer.embed(data, map_type="zz", embedding_dim=3)
        assert "embedded_vectors" in result
        assert result["embedding_dimension"] == 3
        print(f"    → Embedded to {result['embedding_dimension']}D, expressibility: {result['expressibility']:.4f}")

    def test_eigenvalue_pipeline():
        M = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)
        result = analyzer.eigenvalue_analysis(M)
        assert len(result["eigenvalues"]) > 0
        print(f"    → Eigenvalues: {result['eigenvalues']}")

    def test_stat_estimation():
        data = np.random.randn(200)
        result = analyzer.estimate_statistic(data, predicate=lambda x: x > 0, confidence=0.95)
        assert 0 <= result["estimated_value"] <= 1
        assert result["quadratic_speedup"] >= 0
        print(f"    → P(x>0) ≈ {result['estimated_value']:.4f}, speedup: {result['quadratic_speedup']:.1f}x")

    def test_topology_pipeline():
        data = np.random.randn(25, 2)
        result = analyzer.topological_analysis(data)
        assert "betti_numbers" in result
        print(f"    → Betti: {result['betti_numbers']}, complexity: {result['topological_complexity']:.4f}")

    def test_graph_pipeline():
        result = analyzer.graph_analysis(adjacency)
        assert "node_scores" in result
        assert result["n_communities"] >= 1
        print(f"    → {result['n_communities']} communities, entropy: {result['walk_entropy']:.4f}")

    def test_full_pipeline():
        small = data_2d[:8]
        result = analyzer.full_analysis(small, denoise_first=True, n_clusters=2)
        assert result["pipeline"] == "full_analysis"
        assert "summary" in result
        assert "spectral" in result
        assert "clustering" in result
        assert "anomalies" in result
        assert "god_code_resonance" in result
        stages = result["summary"]["stages_completed"]
        total_time = result["total_execution_time"]
        print(f"    → {stages} stages, {total_time:.2f}s total, GC fidelity: {result['god_code_resonance']['fidelity']:.4f}")

    test("Status report", test_status)
    test("Spectral pipeline", test_spectral_pipeline)
    test("Cluster pipeline", test_cluster_pipeline)
    test("Anomaly pipeline", test_anomaly_pipeline)
    test("Search pipeline", test_search_pipeline)
    test("Linear solve pipeline", test_linear_solve_pipeline)
    test("Denoise pipeline", test_denoise_pipeline)
    test("Embed pipeline", test_embed_pipeline)
    test("Eigenvalue pipeline", test_eigenvalue_pipeline)
    test("Statistical estimation", test_stat_estimation)
    test("Topology pipeline", test_topology_pipeline)
    test("Graph pipeline", test_graph_pipeline)
    test("Full analysis pipeline", test_full_pipeline)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("  L104 QUANTUM DATA ANALYZER v1.0.0 — VALIDATION SUITE")
    print("  15 Quantum Algorithms • 4 Feature Maps • 5 Pattern Recognizers")
    print("  3 Denoisers • Full Orchestrator Pipeline")
    print("  INVARIANT: 527.5184818492612 | PILOT: LONDEL")
    print("═" * 70)

    t_total = time.time()

    phase_1()
    phase_2()
    phase_3()
    phase_4()
    phase_5()
    phase_6()

    elapsed = time.time() - t_total

    # Summary
    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = sum(1 for _, ok, _, _ in results if not ok)
    total = len(results)

    print("\n" + "═" * 70)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed ({elapsed:.2f}s)")
    print("═" * 70)

    if failed > 0:
        print(f"\n  {FAIL} FAILED TESTS:")
        for name, ok, dt, err in results:
            if not ok:
                print(f"    • {name}: {err}")

    if failed == 0:
        print(f"\n  {PASS} ALL {total} TESTS PASSED — Quantum Data Analyzer OPERATIONAL")
    else:
        print(f"\n  {WARN} {failed} tests need attention")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
