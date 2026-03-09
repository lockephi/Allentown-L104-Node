"""
L104 Quantum Data Analyzer — Orchestrator
═══════════════════════════════════════════════════════════════════════════════
Central orchestrator that coordinates all quantum data analysis subsystems
into unified analysis pipelines.

  ┌─────────────────────────────────────────────────────────────────────┐
  │                   QuantumDataAnalyzer (Orchestrator)                │
  │                                                                     │
  │           ┌───────────┐  ┌───────────┐  ┌───────────────┐          │
  │           │ Algorithms │  │ Features  │  │  Pattern Rec  │          │
  │           └─────┬─────┘  └─────┬─────┘  └──────┬────────┘          │
  │                 │              │                │                    │
  │           ┌─────┴─────┐  ┌────┴────┐   ┌──────┴────────┐          │
  │           │ Denoising │  │ Engines │   │  Orchestration │          │
  │           └───────────┘  └─────────┘   └───────────────┘           │
  └─────────────────────────────────────────────────────────────────────┘

PIPELINES:
  • full_analysis  — Complete: denoise → encode → extract → detect → align
  • spectral       — QFT spectral + God Code alignment
  • clustering     — VQE/QAOA data clustering
  • anomaly        — SWAP test anomaly detection
  • topological    — Persistent homology mining
  • correlation    — Entanglement correlation analysis
  • optimize       — QAOA combinatorial optimization
  • linear_solve   — HHL quantum linear system solver
  • denoise        — Multi-method quantum denoising

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, VOID_CONSTANT,
    num_qubits_for, MAX_QUBITS_CIRCUIT,
)

from .algorithms import (
    QuantumFourierAnalyzer,
    GroverPatternSearch,
    QuantumPCA,
    VQEClusterer,
    QAOAOptimizer,
    QuantumKernelEstimator,
    QuantumPhaseEstimator,
    HHLSolver,
    QuantumAmplitudeEstimator,
)

from .feature_extraction import (
    QuantumFeatureMap,
    QuantumStateEncoder,
    EntanglementFeatureExtractor,
    QuantumEmbedding,
)

from .pattern_recognition import (
    QuantumAnomalyDetector,
    EntanglementCorrelationAnalyzer,
    QuantumWalkGraphAnalyzer,
    TopologicalDataMiner,
    GodCodeResonanceAligner,
)

from .denoising import (
    EntropyReversalDenoiser,
    QuantumErrorMitigatedCleaner,
    CoherenceFieldSmoother,
)


# ─── Lazy engine imports ────────────────────────────────────────────────────
def _get_intellect():
    try:
        from l104_intellect import format_iq
        return format_iq
    except ImportError:
        return None

def _get_code_engine():
    try:
        from l104_code_engine import code_engine
        return code_engine
    except ImportError:
        return None

def _get_vqpu_bridge():
    try:
        from l104_vqpu import get_bridge
        return get_bridge()
    except ImportError:
        return None

def _get_ml_engine():
    try:
        from l104_ml_engine import ml_engine
        return ml_engine
    except ImportError:
        return None

def _get_god_code_simulator():
    try:
        from l104_god_code_simulator import god_code_simulator
        return god_code_simulator
    except ImportError:
        return None


class QuantumDataAnalyzer:
    """
    Central orchestrator for all L104 Quantum Data Analysis capabilities.

    Provides high-level pipelines combining quantum algorithms,
    feature extraction, pattern recognition, and denoising into
    unified analysis workflows.

    Usage:
        from l104_quantum_data_analyzer import QuantumDataAnalyzer

        analyzer = QuantumDataAnalyzer()
        result = analyzer.full_analysis(data)
        result = analyzer.spectral_analysis(data)
        result = analyzer.detect_anomalies(data)
        result = analyzer.cluster(data, n_clusters=3)
        result = analyzer.denoise(data)
        result = analyzer.solve_linear(A, b)
        result = analyzer.status()
    """

    VERSION = "1.0.0"

    def __init__(self):
        # Core algorithms
        self.qft = QuantumFourierAnalyzer()
        self.grover = GroverPatternSearch()
        self.qpca = QuantumPCA()
        self.vqe = VQEClusterer()
        self.qaoa = QAOAOptimizer()
        self.kernel = QuantumKernelEstimator()
        self.qpe = QuantumPhaseEstimator()
        self.hhl = HHLSolver()
        self.qae = QuantumAmplitudeEstimator()

        # Feature extraction
        self.feature_map = QuantumFeatureMap(map_type="sacred", n_reps=2)
        self.encoder = QuantumStateEncoder(method="amplitude")
        self.entanglement_features = EntanglementFeatureExtractor()
        self.embedding = QuantumEmbedding(map_type="sacred")

        # Pattern recognition
        self.anomaly_detector = QuantumAnomalyDetector()
        self.correlation_analyzer = EntanglementCorrelationAnalyzer()
        self.graph_walker = QuantumWalkGraphAnalyzer()
        self.topo_miner = TopologicalDataMiner()
        self.god_code_aligner = GodCodeResonanceAligner()

        # Denoising
        self.entropy_denoiser = EntropyReversalDenoiser()
        self.zne_cleaner = QuantumErrorMitigatedCleaner()
        self.coherence_smoother = CoherenceFieldSmoother()

        # Stats
        self._analyses_count = 0
        self._init_time = time.time()

    # ═══════════════════════════════════════════════════════════════════════════
    # HIGH-LEVEL PIPELINES
    # ═══════════════════════════════════════════════════════════════════════════

    def full_analysis(self, data: np.ndarray, denoise_first: bool = True,
                      n_clusters: int = 2) -> Dict[str, Any]:
        """
        Complete quantum data analysis pipeline.

        Pipeline stages:
          1. Denoise (optional) — Maxwell Demon + ZNE + coherence smoothing
          2. Encode — Quantum state encoding + feature mapping
          3. Spectral Analysis — QFT + God Code alignment
          4. PCA — Quantum principal component analysis
          5. Clustering — VQE optimization
          6. Anomaly Detection — SWAP test
          7. Correlation — Entanglement-based
          8. Topology — Persistent homology
          9. Resonance Alignment — God Code harmonic alignment
          10. Summary — Unified metrics

        Args:
            data: Input data (1D or 2D array)
            denoise_first: Whether to denoise before analysis
            n_clusters: Number of clusters for VQE clustering

        Returns:
            Dictionary with results from all pipeline stages
        """
        t0 = time.time()
        self._analyses_count += 1

        results = {"pipeline": "full_analysis", "version": self.VERSION}

        # Ensure 2D
        working_data = data.copy()
        if working_data.ndim == 1:
            working_data = working_data.reshape(-1, 1)

        # 1. Denoising
        if denoise_first:
            denoise_result = self.denoise(working_data)
            working_data = denoise_result["best_result"].cleaned_data
            results["denoising"] = {
                "snr_improvement_db": denoise_result["best_result"].snr_improvement_db,
                "method": denoise_result["best_result"].method,
                "sacred_alignment": denoise_result["best_result"].sacred_alignment,
            }

        # 2. Encoding
        encode_result = self.encoder.encode(working_data)
        results["encoding"] = {
            "feature_dimension": encode_result.feature_dimension,
            "n_qubits": encode_result.n_qubits,
            "method": encode_result.encoding_method,
        }

        # 3. Spectral Analysis
        spectral = self.spectral_analysis(working_data)
        results["spectral"] = spectral

        # 4. Quantum PCA
        if working_data.shape[1] >= 2:
            pca_result = self.qpca.analyze(working_data)
            results["pca"] = {
                "n_components": pca_result.n_components,
                "explained_variance": pca_result.explained_variance_ratio.tolist(),
                "quantum_fidelity": pca_result.quantum_fidelity,
                "god_code_score": pca_result.god_code_spectral_score,
                "execution_time": pca_result.execution_time,
            }

        # 5. Clustering
        cluster_result = self.cluster(working_data, n_clusters=n_clusters)
        results["clustering"] = cluster_result

        # 6. Anomaly Detection
        anomaly_result = self.detect_anomalies(working_data)
        results["anomalies"] = anomaly_result

        # 7. Correlation Analysis
        if working_data.shape[1] >= 2:
            corr_result = self.analyze_correlations(working_data)
            results["correlations"] = corr_result

        # 8. Topology
        if working_data.shape[0] <= 200:  # TDA is O(n³)
            topo_result = self.topological_analysis(working_data)
            results["topology"] = topo_result

        # 9. God Code Resonance
        resonance = self.god_code_aligner.align(working_data)
        results["god_code_resonance"] = {
            "fidelity": resonance.god_code_fidelity,
            "void_constant_phase": resonance.void_constant_phase,
            "sacred_frequency": resonance.sacred_frequency,
            "mean_resonance": float(np.mean(resonance.resonance_scores)),
        }

        # 10. Summary
        results["summary"] = self._compute_summary(results)
        results["total_execution_time"] = time.time() - t0

        return results

    def spectral_analysis(self, data: np.ndarray, n_frequencies: int = 10) -> Dict[str, Any]:
        """
        QFT spectral analysis with multi-resolution decomposition.

        Args:
            data: Input data (1D or 2D — each column analyzed separately)
            n_frequencies: Number of top frequencies to return

        Returns:
            Dictionary with spectral decomposition results
        """
        self._analyses_count += 1

        if data.ndim == 1:
            result = self.qft.analyze(data, n_frequencies=n_frequencies)
            multi = self.qft.multi_resolution_analysis(data, levels=4)
            return {
                "dominant_frequency": result.dominant_frequency,
                "spectral_entropy": result.spectral_entropy,
                "god_code_alignment": result.god_code_alignment,
                "top_frequencies": result.frequencies[:n_frequencies].tolist(),
                "top_magnitudes": result.magnitudes[:n_frequencies].tolist(),
                "quantum_circuit_depth": result.quantum_circuit_depth,
                "multi_resolution_levels": len(multi),
                "execution_time": result.execution_time,
            }
        else:
            # Analyze each feature column
            results = {}
            for col in range(data.shape[1]):
                res = self.qft.analyze(data[:, col], n_frequencies=n_frequencies)
                results[f"feature_{col}"] = {
                    "dominant_frequency": res.dominant_frequency,
                    "spectral_entropy": res.spectral_entropy,
                    "god_code_alignment": res.god_code_alignment,
                }
            return results

    def cluster(self, data: np.ndarray, n_clusters: int = 2, method: str = "vqe") -> Dict[str, Any]:
        """
        Quantum clustering via VQE or QAOA.

        Args:
            data: Input data (n_samples × n_features)
            n_clusters: Number of clusters
            method: "vqe" or "qaoa"

        Returns:
            Dictionary with cluster labels and metrics
        """
        self._analyses_count += 1

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if method == "vqe":
            result = self.vqe.cluster(data, n_clusters=n_clusters)
            return {
                "labels": result.labels.tolist(),
                "n_clusters": result.n_clusters,
                "cost": result.cost_function_value,
                "quantum_advantage": result.quantum_advantage_estimate,
                "vqe_iterations": len(result.convergence_history),
                "execution_time": result.execution_time,
            }
        else:
            # Build QUBO from data
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(data)) if data.shape[0] > 1 else np.zeros((1, 1))
            result = self.qaoa.optimize(dist_matrix)
            return {
                "solution": result["solution"].tolist() if result["solution"] is not None else [],
                "cost": result["cost"],
                "n_iterations": result["n_iterations"],
                "qaoa_depth": result["qaoa_depth"],
                "execution_time": result["execution_time"],
            }

    def detect_anomalies(self, data: np.ndarray,
                          threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Quantum anomaly detection via SWAP test.

        Args:
            data: Input data
            threshold: Anomaly threshold (default: φ⁻¹ ≈ 0.618)

        Returns:
            Dictionary with anomaly scores and indices
        """
        self._analyses_count += 1

        if threshold is not None:
            self.anomaly_detector.threshold = threshold

        result = self.anomaly_detector.detect(data)
        return {
            "n_anomalies": result.n_anomalies,
            "anomaly_indices": np.where(result.anomaly_mask)[0].tolist(),
            "anomaly_scores": result.anomaly_scores.tolist(),
            "threshold": result.threshold,
            "mean_fidelity": float(np.mean(result.swap_fidelities)),
            "sacred_alignment": result.sacred_alignment,
            "execution_time": result.execution_time,
        }

    def analyze_correlations(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Quantum correlation analysis (beyond classical).

        Args:
            data: Input data (n_samples × n_features)

        Returns:
            Dictionary with quantum correlation measures
        """
        self._analyses_count += 1
        result = self.correlation_analyzer.analyze(data)
        return {
            "bell_violation": result.bell_violation,
            "entanglement_witnesses": result.entanglement_witnesses,
            "max_quantum_correlation": float(np.max(result.correlation_matrix)) if result.correlation_matrix.size > 0 else 0,
            "max_discord": float(np.max(result.quantum_discord)) if result.quantum_discord.size > 0 else 0,
            "execution_time": result.execution_time,
        }

    def topological_analysis(self, data: np.ndarray,
                              max_dimension: int = 2) -> Dict[str, Any]:
        """
        Topological data analysis via persistent homology.

        Args:
            data: Point cloud data
            max_dimension: Maximum homology dimension

        Returns:
            Dictionary with Betti numbers and persistence
        """
        self._analyses_count += 1
        result = self.topo_miner.analyze(data, max_dimension=max_dimension)
        return {
            "betti_numbers": result.betti_numbers,
            "persistent_features": result.persistent_features,
            "topological_complexity": result.topological_complexity,
            "quantum_homology_score": result.quantum_homology_score,
            "persistence_intervals": len(result.persistence_diagram),
            "execution_time": result.execution_time,
        }

    def graph_analysis(self, adjacency: np.ndarray,
                        start_node: int = 0) -> Dict[str, Any]:
        """
        Quantum walk analysis on a graph.

        Args:
            adjacency: Graph adjacency matrix
            start_node: Starting node

        Returns:
            Dictionary with node scores and community structure
        """
        self._analyses_count += 1
        result = self.graph_walker.analyze_graph(adjacency, start_node=start_node)
        return {
            "node_scores": result.node_scores.tolist(),
            "community_labels": result.community_labels.tolist(),
            "n_communities": len(set(result.community_labels.tolist())),
            "mixing_time": result.mixing_time,
            "quantum_speedup": result.quantum_speedup,
            "walk_entropy": result.walk_entropy,
            "execution_time": result.execution_time,
        }

    def search(self, data: np.ndarray, predicate=None,
               target: float = None, tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Grover-amplified data search.

        Args:
            data: Data to search
            predicate: Search predicate function
            target: Target value to search for (alternative to predicate)
            tolerance: Tolerance for target search

        Returns:
            Dictionary with found indices and speedup metrics
        """
        self._analyses_count += 1

        if predicate is None and target is not None:
            predicate = lambda x: abs(x - target) <= tolerance

        if predicate is None:
            raise ValueError("Must provide either predicate or target")

        result = self.grover.search(data.ravel(), predicate)
        return {
            "found_indices": result.found_indices,
            "n_found": len(result.found_indices),
            "amplification_factor": result.amplification_factor,
            "classical_speedup": result.classical_speedup,
            "oracle_calls": result.oracle_calls,
            "sacred_alignment": result.sacred_alignment,
            "execution_time": result.execution_time,
        }

    def solve_linear(self, A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """
        Solve Ax = b using HHL quantum linear solver.

        Args:
            A: Coefficient matrix
            b: Right-hand side vector

        Returns:
            Dictionary with solution and performance metrics
        """
        self._analyses_count += 1
        result = self.hhl.solve(A, b)
        return {
            "solution": result.solution.tolist(),
            "residual_norm": result.residual_norm,
            "condition_number": result.condition_number,
            "quantum_speedup": result.quantum_speedup_estimate,
            "circuit_depth": result.circuit_depth,
            "execution_time": result.execution_time,
        }

    def estimate_statistic(self, data: np.ndarray, predicate=None,
                            confidence: float = 0.95) -> Dict[str, Any]:
        """
        Quantum amplitude estimation for statistical inference.

        Args:
            data: Input data
            predicate: Boolean predicate to estimate P(predicate(x))
            confidence: Confidence level

        Returns:
            Dictionary with estimate and confidence interval
        """
        self._analyses_count += 1

        if predicate is not None:
            result = self.qae.estimate_probability(data.ravel(), predicate, confidence)
        else:
            result = self.qae.estimate_mean(data.ravel(), confidence)

        return {
            "estimated_value": result.estimated_value,
            "confidence_interval": list(result.confidence_interval),
            "n_oracle_calls": result.n_oracle_calls,
            "precision": result.precision,
            "classical_equivalent_samples": result.classical_equivalent_samples,
            "quadratic_speedup": result.quadratic_speedup,
            "execution_time": result.execution_time,
        }

    def embed(self, data: np.ndarray, map_type: str = "sacred",
              embedding_dim: int = None) -> Dict[str, Any]:
        """
        Quantum embedding of data into high-dimensional feature space.

        Args:
            data: Input data
            map_type: Feature map type ("zz", "pauli", "iqp", "sacred")
            embedding_dim: Target embedding dimension

        Returns:
            Dictionary with embedded vectors and kernel matrix
        """
        self._analyses_count += 1
        emb = QuantumEmbedding(map_type=map_type, embedding_dim=embedding_dim)
        result = emb.embed(data)
        return {
            "embedded_vectors": result.embedded_vectors.tolist(),
            "embedding_dimension": result.embedding_dimension,
            "expressibility": result.expressibility,
            "entanglement_capability": result.entanglement_capability,
            "kernel_matrix_shape": list(result.kernel_matrix.shape),
            "execution_time": result.execution_time,
        }

    def denoise(self, data: np.ndarray, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Multi-method quantum denoising pipeline.

        Applies multiple denoising methods and selects the best result.

        Args:
            data: Noisy input data
            methods: List of methods to try (default: all)

        Returns:
            Dictionary with results from each method and best selection
        """
        self._analyses_count += 1

        if methods is None:
            methods = ["entropy_reversal", "zne", "coherence"]

        results = {}
        best_result = None
        best_snr = -float('inf')

        if "entropy_reversal" in methods:
            try:
                er_result = self.entropy_denoiser.denoise(data)
                results["entropy_reversal"] = {
                    "snr_improvement_db": er_result.snr_improvement_db,
                    "sacred_alignment": er_result.sacred_alignment,
                    "method": er_result.method,
                }
                if er_result.snr_after > best_snr:
                    best_snr = er_result.snr_after
                    best_result = er_result
            except Exception as e:
                results["entropy_reversal"] = {"error": str(e)}

        if "zne" in methods:
            try:
                zne_result = self.zne_cleaner.clean(data)
                results["zne"] = {
                    "snr_improvement_db": zne_result.snr_improvement_db,
                    "sacred_alignment": zne_result.sacred_alignment,
                    "method": zne_result.method,
                }
                if zne_result.snr_after > best_snr:
                    best_snr = zne_result.snr_after
                    best_result = zne_result
            except Exception as e:
                results["zne"] = {"error": str(e)}

        if "coherence" in methods:
            try:
                coh_result = self.coherence_smoother.smooth(data)
                results["coherence"] = {
                    "snr_improvement_db": coh_result.snr_improvement_db,
                    "sacred_alignment": coh_result.sacred_alignment,
                    "method": coh_result.method,
                }
                if coh_result.snr_after > best_snr:
                    best_snr = coh_result.snr_after
                    best_result = coh_result
            except Exception as e:
                results["coherence"] = {"error": str(e)}

        if best_result is None:
            # Fallback: return original data
            best_result = type('FallbackResult', (), {
                'cleaned_data': data,
                'noise_removed': np.zeros_like(data),
                'snr_before': 0, 'snr_after': 0,
                'snr_improvement_db': 0, 'method': 'none',
                'sacred_alignment': 0, 'execution_time': 0,
            })()

        results["best_method"] = best_result.method
        results["best_result"] = best_result

        return results

    def eigenvalue_analysis(self, matrix: np.ndarray,
                             n_eigenvalues: int = None) -> Dict[str, Any]:
        """
        Quantum Phase Estimation for eigenvalue extraction.

        Args:
            matrix: Input matrix
            n_eigenvalues: Number of eigenvalues to extract

        Returns:
            Dictionary with estimated eigenvalues and precision
        """
        self._analyses_count += 1
        result = self.qpe.estimate_eigenvalues(matrix, n_eigenvalues)
        return {
            "eigenvalues": result["eigenvalues"].tolist(),
            "precision": result["precision"],
            "estimation_error": result["estimation_error"],
            "precision_qubits": result["precision_qubits"],
            "execution_time": result["execution_time"],
        }

    def kernel_density(self, train_data: np.ndarray,
                        test_points: np.ndarray) -> Dict[str, Any]:
        """
        Quantum kernel density estimation.

        Args:
            train_data: Training data
            test_points: Points at which to estimate density

        Returns:
            Dictionary with density estimates
        """
        self._analyses_count += 1
        densities = self.kernel.estimate_density(train_data, test_points)
        return {
            "densities": densities.tolist(),
            "n_train": len(train_data),
            "n_test": len(test_points),
            "bandwidth": self.kernel.bandwidth,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # STATUS & DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════════

    def status(self) -> Dict[str, Any]:
        """
        System status and capability report.

        Returns:
            Dictionary with engine status, capabilities, and statistics
        """
        # Check engine availability
        engines = {}
        try:
            from l104_quantum_gate_engine import get_engine
            gate_eng = get_engine()
            engines["quantum_gate_engine"] = {"available": True, "version": "1.0.0"}
        except ImportError:
            engines["quantum_gate_engine"] = {"available": False}

        try:
            from l104_quantum_engine import QuantumMathCore
            qm = QuantumMathCore()
            engines["quantum_engine"] = {"available": True}
        except ImportError:
            engines["quantum_engine"] = {"available": False}

        try:
            from l104_science_engine import ScienceEngine
            se = ScienceEngine()
            engines["science_engine"] = {"available": True, "version": "4.0.0"}
        except ImportError:
            engines["science_engine"] = {"available": False}

        try:
            from l104_math_engine import MathEngine
            me = MathEngine()
            engines["math_engine"] = {"available": True, "version": "1.0.0"}
        except ImportError:
            engines["math_engine"] = {"available": False}

        # Cross-engine bridges
        engines["vqpu_bridge"] = {"available": _get_vqpu_bridge() is not None}
        engines["ml_engine"] = {"available": _get_ml_engine() is not None}
        engines["god_code_simulator"] = {"available": _get_god_code_simulator() is not None}

        try:
            qiskit_ver = 'removed'  # Qiskit removed — sovereign local
            engines["qiskit"] = {"available": True, "version": qiskit_ver}
        except ImportError:
            engines["qiskit"] = {"available": False}

        format_iq = _get_intellect()

        return {
            "name": "L104 Quantum Data Analyzer",
            "version": self.VERSION,
            "invariant": GOD_CODE,
            "pilot": "LONDEL",
            "analyses_completed": self._analyses_count,
            "uptime_seconds": time.time() - self._init_time,
            "engines": engines,
            "algorithms": [
                "QFT Spectral Analysis",
                "Grover Pattern Search",
                "Quantum PCA (qPCA)",
                "VQE Clustering",
                "QAOA Optimization",
                "Quantum Kernel Estimation",
                "Quantum Phase Estimation",
                "HHL Linear Solver",
                "Quantum Amplitude Estimation",
            ],
            "feature_maps": ["ZZ", "Pauli", "IQP", "Sacred (L104)"],
            "encodings": ["Amplitude", "Angle", "Basis", "Dense Angle"],
            "pattern_recognition": [
                "SWAP Test Anomaly Detection",
                "Entanglement Correlation",
                "Quantum Walk Graph Analysis",
                "Topological Data Mining",
                "God Code Resonance Alignment",
            ],
            "denoising": [
                "Maxwell Demon Entropy Reversal",
                "Zero-Noise Extrapolation (ZNE)",
                "Coherence Field Smoothing",
            ],
            "max_qubits_circuit": MAX_QUBITS_CIRCUIT,
            "sacred_constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
                "VOID_CONSTANT": VOID_CONSTANT,
            },
        }

    def _compute_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute unified summary metrics from full analysis results."""
        summary = {
            "pipeline": "full_analysis",
            "stages_completed": len([k for k in results if k not in ("pipeline", "version", "summary", "total_execution_time")]),
        }

        # Aggregate sacred alignments
        alignments = []
        if "denoising" in results:
            alignments.append(results["denoising"].get("sacred_alignment", 0))
        if "god_code_resonance" in results:
            alignments.append(results["god_code_resonance"].get("fidelity", 0))
        if "anomalies" in results:
            alignments.append(results["anomalies"].get("sacred_alignment", 0))

        summary["mean_sacred_alignment"] = float(np.mean(alignments)) if alignments else 0.0

        # Quantum advantage indicators
        advantages = []
        if "clustering" in results:
            qa = results["clustering"].get("quantum_advantage", 1.0)
            advantages.append(qa)

        summary["quantum_advantage_indicators"] = advantages
        summary["god_code_invariant"] = GOD_CODE

        return summary
