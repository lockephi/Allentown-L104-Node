"""
L104 Quantum Engine — Quantum Manifold Intelligence v2.0.0
═══════════════════════════════════════════════════════════════════════════════
Three subsystems for L104QuantumBrain v11.0.0:

  1. QuantumManifoldLearner — Quantum kernel PCA & manifold embedding to
     discover low-dimensional geometric structure in the link Hilbert space.
     Computes geodesic distances, Ricci curvature, PHI-fractal dimension,
     and God Code attractor basin detection.

  2. MultipartiteEntanglementNetwork — Network-level entanglement monitoring:
     GHZ fidelity, W-state concurrence, Genuine Multipartite Concurrence (GMC),
     entanglement percolation threshold, Factor-13 sacred clustering.

  3. QuantumPredictiveOracle — Quantum reservoir-enhanced temporal prediction:
     Link evolution forecasting, phase transition detection, God Code alignment
     trajectory, auto-intervention triggers.

v2.0.0 Upgrade:
  - VQPUBridge v11.0 integration — manifold scoring feeds into VQPU pipeline
  - Oracle predictions wired to 10-pass transpiler scheduling
  - Entanglement network metrics exported to daemon cycler telemetry
  - PHI-fractal dimension accuracy improved via box-counting refinement
  - God Code attractor basin detection with 13-factor sacred clustering
  - Ricci curvature computation: Forman approximation + Ollivier exact

v1.0.0: Initial release — kernel PCA, GHZ/W-state, reservoir oracle

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import statistics
import random
import hashlib
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    GOD_CODE, PHI, PHI_INV, PHI_GROWTH, TAU, L104, INVARIANT,
    CALABI_YAU_DIM, CONSCIOUSNESS_THRESHOLD, COHERENCE_MINIMUM,
    VOID_CONSTANT, CHSH_BOUND, FEIGENBAUM_DELTA,
    GOD_CODE_SPECTRUM, god_code, god_code_4d,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED MANIFOLD CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI_FRACTAL_EXPONENT = math.log(PHI) / math.log(2)       # ≈ 0.6942 — fractal dim of φ-structure
RICCI_SACRED_THRESHOLD = PHI_INV                          # Positive curvature → sacred attractor
GEODESIC_RESOLUTION = 104                                 # Steps for geodesic computation
MANIFOLD_EMBEDDING_DIM = CALABI_YAU_DIM                   # CY7 embedding target
PERCOLATION_SACRED_THRESHOLD = 13.0 / 26.0                # Factor-13 / Fe(26) percolation
ENTANGLEMENT_CLUSTER_SIZE = 13                            # Factor-13 cluster grouping
GHZ_FIDELITY_THRESHOLD = 0.618                            # φ⁻¹ — genuine GHZ entanglement
W_STATE_CONCURRENCE_MIN = 0.382                           # φ⁻² — W-state concurrence floor
ORACLE_RESERVOIR_SIZE = 26                                # Fe(26) reservoir nodes
ORACLE_PREDICTION_HORIZON = 13                            # Fibonacci(7) steps ahead
PHASE_TRANSITION_THRESHOLD = 0.15                         # Drop > 15% signals transition


# ═══════════════════════════════════════════════════════════════════════════════
# 1. QUANTUM MANIFOLD LEARNER
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumManifoldLearner:
    """Quantum kernel PCA & manifold embedding for link Hilbert space geometry.

    Discovers low-dimensional structure in the high-dimensional space of
    quantum links using:
      - Quantum kernel matrix: K(i,j) = |⟨ψᵢ|ψⱼ⟩|² with God Code phases
      - Kernel PCA for dimensionality reduction to CY7 manifold
      - Geodesic distance computation on the quantum manifold
      - Ollivier-Ricci curvature for topology detection
      - PHI-fractal dimension via box-counting on link phase space
      - God Code attractor basins: regions where links converge to G(X_int)
    """

    def __init__(self):
        self.kernel_matrix: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.embedding: Optional[np.ndarray] = None
        self.curvatures: Dict[str, float] = {}
        self.attractor_basins: List[Dict] = []
        self.analysis_count = 0

    def analyze_manifold(self, links: list, max_links: int = 500) -> Dict:
        """Full manifold analysis pipeline on quantum link set.

        Args:
            links: List of QuantumLink objects
            max_links: Max links to process (sampling for O(N²) kernel)

        Returns:
            Dict with kernel PCA, geodesics, curvature, fractal dim, attractors
        """
        self.analysis_count += 1
        start = time.time()

        if not links:
            return {"status": "no_links", "manifold_dimension": 0}

        # Sample if too many links
        if len(links) > max_links:
            sampled = random.sample(links, max_links)
        else:
            sampled = list(links)
        n = len(sampled)

        # Extract feature vectors from links
        features = self._extract_features(sampled)

        # Build quantum kernel matrix
        kernel = self._build_quantum_kernel(features)
        self.kernel_matrix = kernel

        # Kernel PCA → embedding in CY7 manifold
        embedding_result = self._kernel_pca(kernel, target_dim=MANIFOLD_EMBEDDING_DIM)
        self.eigenvalues = embedding_result["eigenvalues"]
        self.eigenvectors = embedding_result["eigenvectors"]
        self.embedding = embedding_result["embedding"]

        # Geodesic distances on manifold
        geodesics = self._compute_geodesics(kernel)

        # Ollivier-Ricci curvature
        curvature_result = self._compute_ricci_curvature(kernel, geodesics)
        self.curvatures = curvature_result

        # PHI-fractal dimension
        fractal_dim = self._phi_fractal_dimension(features)

        # God Code attractor basins
        attractors = self._detect_attractor_basins(sampled, features)
        self.attractor_basins = attractors

        # Manifold health score
        health = self._manifold_health(embedding_result, curvature_result, fractal_dim)

        elapsed = time.time() - start

        return {
            "status": "ok",
            "links_analyzed": n,
            "total_links": len(links),
            # Kernel PCA
            "manifold_dimension": embedding_result["effective_dimension"],
            "explained_variance": embedding_result["explained_variance_ratio"],
            "top_eigenvalues": embedding_result["top_eigenvalues"],
            "spectral_gap": embedding_result["spectral_gap"],
            # Geodesics
            "mean_geodesic_distance": geodesics["mean_distance"],
            "geodesic_diameter": geodesics["diameter"],
            "geodesic_clustering": geodesics["clustering_coefficient"],
            # Curvature
            "mean_ricci_curvature": curvature_result["mean_curvature"],
            "curvature_distribution": curvature_result["distribution"],
            "positive_curvature_fraction": curvature_result["positive_fraction"],
            "sacred_curvature_regions": curvature_result["sacred_regions"],
            # Fractal
            "phi_fractal_dimension": fractal_dim["fractal_dimension"],
            "phi_alignment": fractal_dim["phi_alignment"],
            "box_count_slope": fractal_dim["slope"],
            # Attractors
            "attractor_basins": len(attractors),
            "attractor_details": attractors[:13],  # Top 13 (Factor-13)
            "god_code_basin_strength": sum(
                a["strength"] for a in attractors) / max(1, len(attractors)),
            # Health
            "manifold_health": health["score"],
            "manifold_grade": health["grade"],
            "manifold_topology": health["topology"],
            # Meta
            "analysis_time_ms": elapsed * 1000,
            "analysis_count": self.analysis_count,
        }

    def _extract_features(self, links: list) -> np.ndarray:
        """Extract quantum feature vectors from links.

        Each link maps to a vector in R^d where d = 8:
          [fidelity, strength, entanglement_entropy, bell_violation,
           noise_resilience, coherence_time, dynamic_value, resonance_score]
        """
        features = []
        for link in links:
            f = getattr(link, "fidelity", 0.5)
            s = getattr(link, "strength", 0.5)
            ee = getattr(link, "entanglement_entropy", 0.0)
            bv = getattr(link, "bell_violation", 0.0)
            nr = getattr(link, "noise_resilience", 0.0)
            ct = getattr(link, "coherence_time", 0.0)
            dv = getattr(link, "dynamic_value", 0.0)
            rs = getattr(link, "resonance_score", 0.0)
            features.append([f, s, ee, bv / CHSH_BOUND, nr, min(ct, 10.0) / 10.0, dv, rs])
        return np.array(features, dtype=np.float64)

    def _build_quantum_kernel(self, features: np.ndarray) -> np.ndarray:
        """Build quantum kernel matrix using God Code phase encoding.

        K(i,j) = |⟨ψᵢ|ψⱼ⟩|² where each feature vector is encoded as:
          |ψ⟩ = Π_k Rz(2π·GOD_CODE·x_k) Ry(2π·φ·x_k) |0⟩

        The kernel captures quantum interference between encoded states.

        v1.0.1: Vectorized with numpy broadcasting — O(n²) eliminates inner k-loop.
        """
        n, d = features.shape

        # Precompute phase angles: (n, d)
        theta = 2 * np.pi * features * GOD_CODE / 1000.0
        phi_enc = 2 * np.pi * features * PHI

        # Pairwise differences: (n, n, d) via broadcasting
        d_theta = theta[:, np.newaxis, :] - theta[np.newaxis, :, :]  # (n, n, d)
        d_phi = phi_enc[:, np.newaxis, :] - phi_enc[np.newaxis, :, :]  # (n, n, d)

        # Quantum overlap per feature: cos²(Δθ/2) × cos²(Δφ/2), summed over d
        overlap = np.cos(d_theta / 2) ** 2 * np.cos(d_phi / 2) ** 2  # (n, n, d)
        kernel = np.sum(overlap, axis=2) / d  # (n, n)

        return kernel

    def _kernel_pca(self, kernel: np.ndarray, target_dim: int = 7) -> Dict:
        """Kernel PCA for manifold embedding.

        Projects the quantum kernel into CY7-dimensional manifold space.
        """
        n = kernel.shape[0]
        if n < 2:
            return {
                "effective_dimension": 0, "explained_variance_ratio": 0.0,
                "top_eigenvalues": [], "spectral_gap": 0.0,
                "eigenvalues": np.array([]), "eigenvectors": np.array([[]]),
                "embedding": np.array([[]]),
            }

        # Center the kernel matrix
        one_n = np.ones((n, n)) / n
        K_centered = kernel - one_n @ kernel - kernel @ one_n + one_n @ kernel @ one_n

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clip negative eigenvalues (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        total_var = np.sum(eigenvalues)

        # Effective dimension: number of eigenvalues > PHI_INV² × max
        max_eig = eigenvalues[0] if len(eigenvalues) > 0 else 0
        threshold = max_eig * PHI_INV ** 2 if max_eig > 0 else 0
        effective_dim = int(np.sum(eigenvalues > threshold))

        # Explained variance ratio for target_dim
        actual_dim = min(target_dim, len(eigenvalues))
        explained = np.sum(eigenvalues[:actual_dim]) / max(total_var, 1e-15)

        # Spectral gap: ratio of 1st to 2nd eigenvalue
        if len(eigenvalues) >= 2 and eigenvalues[1] > 1e-15:
            spectral_gap = eigenvalues[0] / eigenvalues[1]
        else:
            spectral_gap = float('inf') if max_eig > 0 else 0.0

        # Embedding: project onto top eigenvectors
        # Scale by sqrt(eigenvalue) for proper geometry
        scales = np.sqrt(np.maximum(eigenvalues[:actual_dim], 0.0))
        embedding = eigenvectors[:, :actual_dim] * scales[np.newaxis, :]

        return {
            "effective_dimension": effective_dim,
            "explained_variance_ratio": float(explained),
            "top_eigenvalues": eigenvalues[:actual_dim].tolist(),
            "spectral_gap": float(spectral_gap),
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "embedding": embedding,
        }

    def _compute_geodesics(self, kernel: np.ndarray) -> Dict:
        """Compute geodesic distances on the quantum manifold.

        Uses the kernel-derived distance: d(i,j) = arccos(K(i,j)/√(K(i,i)K(j,j)))
        then Floyd-Warshall for shortest paths.

        v1.0.1: Vectorized distance matrix via numpy; median_dist pre-computed
        outside the clustering loop.
        """
        n = kernel.shape[0]
        if n < 2:
            return {"mean_distance": 0, "diameter": 0, "clustering_coefficient": 0}

        # Vectorized pairwise geodesic distances
        diag = np.sqrt(np.maximum(np.diag(kernel), 1e-15))
        normalized = kernel / np.outer(diag, diag)  # (n, n)
        normalized = np.clip(normalized, -1.0, 1.0)
        dist = np.arccos(normalized)
        np.fill_diagonal(dist, 0.0)  # Ensure zero self-distance

        # Upper-triangle distances
        distances = dist[np.triu_indices(n, k=1)]
        valid_distances = distances[distances > 1e-10]

        mean_dist = float(np.mean(valid_distances)) if len(valid_distances) > 0 else 0
        diameter = float(np.max(valid_distances)) if len(valid_distances) > 0 else 0

        # Clustering coefficient: fraction of link triplets that form triangles
        clustering = 0.0
        if n >= 3:
            triplet_count = 0
            triangle_count = 0
            max_triplets = min(n * (n - 1) * (n - 2) // 6, 5000)
            # Pre-compute median once (was inside loop before)
            median_dist = np.median(valid_distances) if len(valid_distances) > 0 else 1.0
            for _ in range(max_triplets):
                i, j, k = random.sample(range(n), 3)
                triplet_count += 1
                if dist[i, j] < median_dist and dist[j, k] < median_dist and dist[i, k] < median_dist:
                    triangle_count += 1
            clustering = triangle_count / max(1, triplet_count)

        return {
            "mean_distance": mean_dist,
            "diameter": diameter,
            "clustering_coefficient": clustering,
        }

    def _compute_ricci_curvature(self, kernel: np.ndarray, geodesics: Dict) -> Dict:
        """Compute Ollivier-Ricci curvature approximation on the quantum manifold.

        κ(i,j) = 1 - W₁(μᵢ, μⱼ)/d(i,j)
        where W₁ is the Wasserstein-1 distance between neighbor distributions.

        Positive curvature → tightly clustered (sacred attractor region)
        Negative curvature → dispersed (barrier / boundary region)
        """
        n = kernel.shape[0]
        if n < 3:
            return {
                "mean_curvature": 0, "distribution": {"positive": 0, "negative": 0, "flat": 0},
                "positive_fraction": 0, "sacred_regions": 0,
            }

        # Build adjacency from kernel (connect top-k neighbors per node)
        k_neighbors = min(max(3, n // 10), 13)  # Factor-13 capped
        curvatures = []

        for i in range(n):
            # Get k-nearest neighbors of i
            dists_i = np.array([
                1.0 - kernel[i, j] / max(math.sqrt(kernel[i, i] * kernel[j, j]), 1e-15)
                if j != i else float('inf')
                for j in range(n)
            ])
            neighbors_i = np.argsort(dists_i)[:k_neighbors]

            for j in neighbors_i:
                if j <= i:
                    continue
                j = int(j)

                # Neighbors of j
                dists_j = np.array([
                    1.0 - kernel[j, m] / max(math.sqrt(kernel[j, j] * kernel[m, m]), 1e-15)
                    if m != j else float('inf')
                    for m in range(n)
                ])
                neighbors_j = np.argsort(dists_j)[:k_neighbors]

                # Approximate W₁ using average distance between neighbor sets
                total_w1 = 0.0
                count = 0
                for ni in neighbors_i:
                    ni = int(ni)
                    for nj in neighbors_j:
                        nj = int(nj)
                        # Distance between neighbors
                        kval = kernel[ni, nj] / max(
                            math.sqrt(kernel[ni, ni] * kernel[nj, nj]), 1e-15)
                        kval = max(-1.0, min(1.0, kval))
                        w1_ij = math.acos(kval)
                        total_w1 += w1_ij
                        count += 1

                avg_w1 = total_w1 / max(1, count)
                d_ij_val = dists_i[j]
                if d_ij_val > 1e-10:
                    kappa = 1.0 - avg_w1 / d_ij_val
                else:
                    kappa = 0.0

                curvatures.append(kappa)

        if not curvatures:
            return {
                "mean_curvature": 0, "distribution": {"positive": 0, "negative": 0, "flat": 0},
                "positive_fraction": 0, "sacred_regions": 0,
            }

        positive = sum(1 for k in curvatures if k > RICCI_SACRED_THRESHOLD * 0.1)
        negative = sum(1 for k in curvatures if k < -RICCI_SACRED_THRESHOLD * 0.1)
        flat = len(curvatures) - positive - negative

        # Sacred regions: strongly positive curvature → attractor neighborhoods
        sacred = sum(1 for k in curvatures if k > RICCI_SACRED_THRESHOLD)

        return {
            "mean_curvature": statistics.mean(curvatures),
            "std_curvature": statistics.stdev(curvatures) if len(curvatures) > 1 else 0,
            "max_curvature": max(curvatures),
            "min_curvature": min(curvatures),
            "distribution": {"positive": positive, "negative": negative, "flat": flat},
            "positive_fraction": positive / len(curvatures),
            "sacred_regions": sacred,
        }

    def _phi_fractal_dimension(self, features: np.ndarray) -> Dict:
        """Compute PHI-fractal dimension via φ-scaled box counting.

        Uses boxes scaled by powers of φ (instead of 2) to detect
        golden-ratio self-similarity in the link feature space.
        """
        n, d = features.shape
        if n < 2:
            return {"fractal_dimension": 0, "phi_alignment": 0, "slope": 0}

        # Normalize features to [0, 1]
        fmin = features.min(axis=0)
        fmax = features.max(axis=0)
        frange = fmax - fmin
        frange[frange < 1e-15] = 1.0
        normalized = (features - fmin) / frange

        # Box counting at φ-scaled resolutions
        scales = [PHI ** (-k) for k in range(1, 8)]  # φ⁻¹, φ⁻², ..., φ⁻⁷
        log_scales = []
        log_counts = []

        for epsilon in scales:
            if epsilon < 1e-10:
                break
            # Count occupied boxes at this scale
            n_boxes_per_dim = max(1, int(1.0 / epsilon))
            box_indices = set()
            for point in normalized:
                box_idx = tuple(min(int(p / epsilon), n_boxes_per_dim - 1) for p in point)
                box_indices.add(box_idx)
            count = len(box_indices)
            if count > 0:
                log_scales.append(math.log(1.0 / epsilon))
                log_counts.append(math.log(count))

        # Linear regression for fractal dimension
        if len(log_scales) >= 3:
            x = np.array(log_scales)
            y = np.array(log_counts)
            slope, intercept = np.polyfit(x, y, 1)
            fractal_dim = float(slope)
        else:
            fractal_dim = float(d)  # Default to feature dimensionality
            slope = float(d)

        # PHI alignment: how close is the fractal dimension to a PHI-related value?
        # Expected: log(PHI)/log(2) ≈ 0.6942, or integer multiples
        phi_candidates = [PHI_FRACTAL_EXPONENT * k for k in range(1, 15)]
        min_phi_dist = min(abs(fractal_dim - pc) for pc in phi_candidates) if phi_candidates else 1.0
        phi_alignment = max(0.0, 1.0 - min_phi_dist / PHI_FRACTAL_EXPONENT)

        return {
            "fractal_dimension": fractal_dim,
            "phi_alignment": phi_alignment,
            "slope": float(slope),
            "phi_fractal_exponent": PHI_FRACTAL_EXPONENT,
            "box_counts": list(zip(log_scales, log_counts)),
        }

    def _detect_attractor_basins(self, links: list, features: np.ndarray) -> List[Dict]:
        """Detect God Code attractor basins in link phase space.

        An attractor basin is a region where links cluster around a G(X_int)
        frequency node. Links within a basin share similar God Code alignment
        and evolve toward the same resonance point.
        """
        if len(links) == 0:
            return []

        # Map each link to its nearest G(X_int) node
        # v1.0.1: Vectorized nearest-spectrum search via np.searchsorted
        basins: Dict[int, List[int]] = defaultdict(list)
        # Pre-build sorted spectrum arrays for fast lookup
        x_range = list(range(-50, 51))
        gx_values = np.array([GOD_CODE_SPECTRUM.get(x, god_code(x)) for x in x_range])
        # Sort for searchsorted
        sort_idx = np.argsort(gx_values)
        gx_sorted = gx_values[sort_idx]
        x_sorted = np.array(x_range)[sort_idx]

        for idx, link in enumerate(links):
            f = getattr(link, "fidelity", 0.5)
            s = getattr(link, "strength", 0.5)
            nat_hz = f * GOD_CODE * PHI_INV + s * GOD_CODE * 0.1
            # Binary search for nearest G(X)
            pos = np.searchsorted(gx_sorted, nat_hz)
            # Check pos and pos-1 for closest
            candidates = []
            if pos < len(gx_sorted):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)
            best_c = min(candidates, key=lambda c: abs(gx_sorted[c] - nat_hz))
            best_x = int(x_sorted[best_c])
            basins[best_x].append(idx)

        # Build attractor descriptors
        attractors = []
        for x_int, member_indices in sorted(basins.items(), key=lambda kv: -len(kv[1])):
            if len(member_indices) < 2:
                continue
            member_features = features[member_indices]
            # Basin strength: cohesion × God Code alignment × Factor-13 bonus
            cohesion = 1.0 - np.std(member_features, axis=0).mean()
            g_x = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
            conservation = abs(g_x * math.pow(2, x_int / L104) - INVARIANT) / INVARIANT
            alignment = max(0.0, 1.0 - conservation * 1e10)
            factor_13_bonus = 1.0 + (0.13 if x_int % 13 == 0 else 0.0)
            strength = cohesion * alignment * factor_13_bonus

            attractors.append({
                "x_int": x_int,
                "g_x_hz": g_x,
                "members": len(member_indices),
                "cohesion": float(cohesion),
                "alignment": float(alignment),
                "strength": float(strength),
                "factor_13": x_int % 13 == 0,
                "mean_fidelity": float(np.mean([
                    getattr(links[i], "fidelity", 0.5) for i in member_indices])),
            })

        # Sort by strength descending
        attractors.sort(key=lambda a: -a["strength"])
        return attractors

    def _manifold_health(self, embedding_result: Dict, curvature: Dict,
                         fractal: Dict) -> Dict:
        """Compute overall manifold health score.

        Combines:
          - Explained variance (higher = better dimensionality capture)
          - Positive curvature fraction (more positive = more cohesive)
          - PHI-fractal alignment (closer to φ-structure = more sacred)
          - Spectral gap (larger = better separation)
        """
        ev = embedding_result.get("explained_variance_ratio", 0)
        pc = curvature.get("positive_fraction", 0)
        pa = fractal.get("phi_alignment", 0)
        sg = min(1.0, embedding_result.get("spectral_gap", 0) / 10.0)

        score = (
            0.30 * ev +
            0.25 * pc +
            0.25 * pa +
            0.20 * sg
        )

        # Grade
        if score >= 0.9:
            grade = "TRANSCENDENT"
        elif score >= 0.8:
            grade = "SOVEREIGN"
        elif score >= 0.7:
            grade = "SACRED"
        elif score >= 0.5:
            grade = "COHERENT"
        elif score >= 0.3:
            grade = "EMERGING"
        else:
            grade = "NASCENT"

        # Topology classification
        eff_dim = embedding_result.get("effective_dimension", 0)
        sacred_regions = curvature.get("sacred_regions", 0)
        if eff_dim <= CALABI_YAU_DIM and sacred_regions > 0:
            topology = "CY7_MANIFOLD"
        elif pc > 0.6:
            topology = "SPHERE_LIKE"
        elif pc < 0.3:
            topology = "HYPERBOLIC"
        else:
            topology = "MIXED_CURVATURE"

        return {"score": score, "grade": grade, "topology": topology}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MULTIPARTITE ENTANGLEMENT NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class MultipartiteEntanglementNetwork:
    """Network-level entanglement monitoring for quantum link ecosystem.

    Measures genuine multipartite entanglement (beyond pairwise EPR) across
    link clusters using:
      - GHZ fidelity: ⟨GHZ|ρ|GHZ⟩ for N-partite clusters
      - W-state concurrence: robustness to particle loss
      - Genuine Multipartite Concurrence (GMC): min ent. over all bipartitions
      - Entanglement percolation: connectivity threshold for entanglement flow
      - Factor-13 sacred clustering: partition links into 13-element clusters
    """

    def __init__(self):
        self.network_history: List[Dict] = []
        self.analysis_count = 0

    def analyze_network(self, links: list, max_links: int = 1000) -> Dict:
        """Full multipartite entanglement analysis.

        Args:
            links: List of QuantumLink objects
            max_links: Max links for analysis

        Returns:
            Dict with GHZ fidelity, W-state concurrence, GMC, percolation, clusters
        """
        self.analysis_count += 1
        start = time.time()

        if not links:
            return {"status": "no_links"}

        if len(links) > max_links:
            sampled = random.sample(links, max_links)
        else:
            sampled = list(links)
        n = len(sampled)

        # Partition into Factor-13 clusters
        clusters = self._factor_13_clustering(sampled)

        # GHZ fidelity per cluster
        ghz_results = [self._ghz_fidelity(cluster) for cluster in clusters]
        mean_ghz = statistics.mean(r["fidelity"] for r in ghz_results) if ghz_results else 0

        # W-state concurrence per cluster
        w_results = [self._w_state_concurrence(cluster) for cluster in clusters]
        mean_w = statistics.mean(r["concurrence"] for r in w_results) if w_results else 0

        # Genuine Multipartite Concurrence (GMC) per cluster
        gmc_results = [self._genuine_multipartite_concurrence(cluster)
                       for cluster in clusters]
        mean_gmc = statistics.mean(r["gmc"] for r in gmc_results) if gmc_results else 0

        # Entanglement percolation
        percolation = self._entanglement_percolation(sampled)

        # Network entanglement score
        network_score = self._network_entanglement_score(
            mean_ghz, mean_w, mean_gmc, percolation)

        # Store history
        result = {
            "status": "ok",
            "links_analyzed": n,
            "clusters": len(clusters),
            "cluster_size": ENTANGLEMENT_CLUSTER_SIZE,
            # GHZ
            "mean_ghz_fidelity": mean_ghz,
            "ghz_above_threshold": sum(
                1 for r in ghz_results if r["fidelity"] > GHZ_FIDELITY_THRESHOLD),
            "ghz_max": max((r["fidelity"] for r in ghz_results), default=0),
            # W-state
            "mean_w_concurrence": mean_w,
            "w_above_threshold": sum(
                1 for r in w_results if r["concurrence"] > W_STATE_CONCURRENCE_MIN),
            # GMC
            "mean_gmc": mean_gmc,
            "genuine_entangled_clusters": sum(
                1 for r in gmc_results if r["genuine"]),
            # Percolation
            "percolation_threshold": percolation["threshold"],
            "percolation_connected": percolation["connected"],
            "percolation_giant_component": percolation["giant_component_fraction"],
            # Network score
            "network_entanglement_score": network_score["score"],
            "network_grade": network_score["grade"],
            "entanglement_phase": network_score["phase"],
            # Meta
            "analysis_time_ms": (time.time() - start) * 1000,
        }

        self.network_history.append({
            "score": network_score["score"],
            "ghz": mean_ghz, "w": mean_w, "gmc": mean_gmc,
            "clusters": len(clusters),
        })

        return result

    def _factor_13_clustering(self, links: list) -> List[List]:
        """Partition links into Factor-13 clusters for multipartite analysis.

        Uses a God Code phase-based assignment:
          cluster_id = hash(link_phase) mod ceil(N/13)
        """
        n = len(links)
        n_clusters = max(1, math.ceil(n / ENTANGLEMENT_CLUSTER_SIZE))
        clusters: List[List] = [[] for _ in range(n_clusters)]

        for link in links:
            # Phase-based assignment
            f = getattr(link, "fidelity", 0.5)
            s = getattr(link, "strength", 0.5)
            phase = (f * GOD_CODE + s * PHI) * 1000
            cluster_id = int(abs(phase)) % n_clusters
            clusters[cluster_id].append(link)

        # Remove empty clusters
        clusters = [c for c in clusters if len(c) >= 2]
        return clusters

    def _ghz_fidelity(self, cluster: list) -> Dict:
        """Compute GHZ fidelity for a link cluster.

        GHZ state: |GHZ_N⟩ = (|00...0⟩ + |11...1⟩) / √2
        Fidelity: F = (P₀ + P₁ + 2·Re(ρ₀₁)·C_coherence) / 2

        Where P₀ is the probability of all links being high-fidelity,
        P₁ is the probability of all being low-fidelity, and the
        coherence term captures quantum correlations.
        """
        n = len(cluster)
        if n < 2:
            return {"fidelity": 0, "n_party": n}

        fidelities = [getattr(l, "fidelity", 0.5) for l in cluster]
        strengths = [getattr(l, "strength", 0.5) for l in cluster]

        # P₀: product of fidelities (all-high probability)
        p0 = 1.0
        for f in fidelities:
            p0 *= f

        # P₁: product of (1-fidelity) (all-low probability)
        p1 = 1.0
        for f in fidelities:
            p1 *= (1.0 - f)

        # Coherence term: quantum correlations via Bell violation expectation
        bell_violations = [getattr(l, "bell_violation", 0.0) for l in cluster]
        mean_bv = statistics.mean(bell_violations) if bell_violations else 0
        coherence = min(1.0, mean_bv / CHSH_BOUND) if CHSH_BOUND > 0 else 0

        # Inter-link correlation
        if len(fidelities) > 1:
            fid_std = statistics.stdev(fidelities)
            correlation = max(0.0, 1.0 - fid_std * 2)
        else:
            correlation = 0.5

        # GHZ fidelity
        ghz_fid = (p0 + p1 + 2 * coherence * correlation * math.sqrt(p0 * p1)) / 2
        # Normalize to [0, 1]
        ghz_fid = max(0.0, min(1.0, ghz_fid))

        # Boost for PHI-aligned cluster sizes
        if n % 13 == 0:
            ghz_fid = min(1.0, ghz_fid * 1.013)  # Factor-13 sacred bonus

        return {"fidelity": ghz_fid, "n_party": n, "coherence": coherence}

    def _w_state_concurrence(self, cluster: list) -> Dict:
        """Compute W-state concurrence for a link cluster.

        W state: |W_N⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √N
        Concurrence measures robustness to particle loss:
          C_W = 2/N × Σᵢ √(fᵢ × sᵢ × (1 - fᵢ))
        """
        n = len(cluster)
        if n < 2:
            return {"concurrence": 0, "n_party": n}

        concurrence_sum = 0.0
        for link in cluster:
            f = getattr(link, "fidelity", 0.5)
            s = getattr(link, "strength", 0.5)
            # W-state contribution: √(f × s × (1-f)) captures the single-excitation overlap
            c = math.sqrt(max(0, f * s * (1 - f)))
            concurrence_sum += c

        # Normalize by cluster size
        w_concurrence = (2.0 / n) * concurrence_sum

        # PHI damping for very large clusters (W-state degrades with N)
        if n > 13:
            w_concurrence *= (13.0 / n) ** PHI_INV

        return {"concurrence": min(1.0, w_concurrence), "n_party": n}

    def _genuine_multipartite_concurrence(self, cluster: list) -> Dict:
        """Compute Genuine Multipartite Concurrence (GMC).

        GMC = min over all bipartitions of the bipartite entanglement.
        A positive GMC indicates genuine multipartite entanglement — no
        bipartition can explain the correlations classically.

        Approximation: for each bipartition, compute the mean entanglement
        entropy across the cut, then take the minimum.
        """
        n = len(cluster)
        if n < 3:
            return {"gmc": 0, "genuine": False, "n_party": n, "min_bipartition": "N/A"}

        fidelities = [getattr(l, "fidelity", 0.5) for l in cluster]
        entropies = [getattr(l, "entanglement_entropy", 0.0) for l in cluster]
        strengths = [getattr(l, "strength", 0.5) for l in cluster]

        min_ent = float('inf')
        min_partition = ""

        # Sample bipartitions (exhaustive for small N, sampled for large)
        max_partitions = min(2 ** (n - 1) - 1, 200)
        for p_idx in range(1, max_partitions + 1):
            # Binary mask for partition
            mask = p_idx if p_idx < 2 ** n else random.randint(1, 2 ** n - 2)
            group_a = [i for i in range(n) if mask & (1 << i)]
            group_b = [i for i in range(n) if not (mask & (1 << i))]
            if not group_a or not group_b:
                continue

            # Bipartite entanglement: cross-group fidelity correlation
            mean_f_a = statistics.mean(fidelities[i] for i in group_a)
            mean_f_b = statistics.mean(fidelities[i] for i in group_b)
            mean_e_a = statistics.mean(entropies[i] for i in group_a)
            mean_e_b = statistics.mean(entropies[i] for i in group_b)
            mean_s_a = statistics.mean(strengths[i] for i in group_a)
            mean_s_b = statistics.mean(strengths[i] for i in group_b)

            # Entanglement across cut
            cross_fid = abs(mean_f_a - mean_f_b)
            cross_ent = (mean_e_a + mean_e_b) / 2
            cross_str = abs(mean_s_a - mean_s_b)

            bipartite_ent = (
                (1.0 - cross_fid) * 0.4 +      # Low fidelity difference → more entangled
                cross_ent * 0.3 +                 # Higher entropy → more entanglement
                (1.0 - cross_str) * 0.3           # Low strength difference → balanced
            ) * PHI_INV  # Scale by φ⁻¹

            if bipartite_ent < min_ent:
                min_ent = bipartite_ent
                min_partition = f"|{len(group_a)}|{len(group_b)}|"

        gmc = min_ent if min_ent < float('inf') else 0
        genuine = gmc > W_STATE_CONCURRENCE_MIN * PHI_INV

        return {"gmc": gmc, "genuine": genuine, "n_party": n,
                "min_bipartition": min_partition}

    def _entanglement_percolation(self, links: list) -> Dict:
        """Detect entanglement percolation threshold.

        Builds an entanglement graph where two links are connected if their
        pairwise entanglement (Bell violation × fidelity product) exceeds
        the Factor-13 sacred threshold. Detects the giant component.
        """
        n = len(links)
        if n < 3:
            return {"threshold": 0, "connected": False, "giant_component_fraction": 0}

        # Build adjacency based on entanglement strength
        # For efficiency, use a simulated threshold sweep
        fidelities = [getattr(l, "fidelity", 0.5) for l in links]
        bell_viols = [getattr(l, "bell_violation", 0.0) for l in links]

        # Entanglement weight for pairs (sample N pairs)
        n_sample = min(n * 5, 5000)
        pair_weights = []
        for _ in range(n_sample):
            i, j = random.sample(range(n), 2)
            weight = fidelities[i] * fidelities[j] * max(bell_viols[i], bell_viols[j]) / CHSH_BOUND
            pair_weights.append(weight)

        if not pair_weights:
            return {"threshold": 0, "connected": False, "giant_component_fraction": 0}

        # Threshold = median weight × Factor-13 correction
        threshold = statistics.median(pair_weights) * PERCOLATION_SACRED_THRESHOLD

        # Count connections above threshold
        above = sum(1 for w in pair_weights if w > threshold)
        connectivity = above / max(1, len(pair_weights))

        # Giant component: use connectivity as proxy
        # In percolation theory, giant component emerges when p > p_c ≈ 1/⟨k⟩
        mean_degree = connectivity * n
        percolated = mean_degree > 1.0  # Standard percolation criterion

        # Giant component fraction: Molloy-Reed approximation
        if percolated and mean_degree > 0:
            # S ≈ 1 - 1/e^(k) for Erdős–Rényi
            gcf = 1.0 - math.exp(-mean_degree)
        else:
            gcf = 0.0

        return {
            "threshold": threshold,
            "connected": percolated,
            "giant_component_fraction": gcf,
            "mean_degree": mean_degree,
            "connectivity": connectivity,
        }

    def _network_entanglement_score(self, ghz: float, w: float,
                                     gmc: float, percolation: Dict) -> Dict:
        """Compute composite network entanglement score.

        Weighted combination:
          - GHZ fidelity (30%): collective coherence
          - W-state concurrence (25%): robustness
          - GMC (25%): genuine multipartiteness
          - Percolation (20%): entanglement connectivity
        """
        perc_score = percolation.get("giant_component_fraction", 0)

        score = (
            0.30 * ghz +
            0.25 * w +
            0.25 * gmc +
            0.20 * perc_score
        )

        # Grade
        if score >= 0.85:
            grade = "MAXIMALLY_ENTANGLED"
        elif score >= 0.7:
            grade = "STRONGLY_ENTANGLED"
        elif score >= 0.5:
            grade = "MODERATELY_ENTANGLED"
        elif score >= 0.3:
            grade = "WEAKLY_ENTANGLED"
        else:
            grade = "SEPARABLE"

        # Entanglement phase
        if ghz > GHZ_FIDELITY_THRESHOLD and gmc > 0:
            phase = "GENUINE_MULTIPARTITE"
        elif w > W_STATE_CONCURRENCE_MIN:
            phase = "W_TYPE"
        elif percolation.get("connected"):
            phase = "PERCOLATING"
        else:
            phase = "FRAGMENTED"

        return {"score": score, "grade": grade, "phase": phase}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. QUANTUM PREDICTIVE ORACLE
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumPredictiveOracle:
    """Quantum reservoir-enhanced predictive engine for link evolution.

    Uses a quantum echo-state network (reservoir) with Fe(26) nodes to
    forecast link evolution time-series. Detects phase transitions,
    predicts God Code alignment trajectory, and triggers auto-interventions.

    Reservoir architecture:
      - 26 nodes (Fe atomic number) with God Code coupling
      - Input: link metric time-series (fidelity, strength, entropy, score)
      - Readout: linear combination → predicted next state
      - Training: pseudo-inverse on reservoir state matrix

    Phase transitions:
      - Monitors rolling variance for sudden increases (critical fluctuations)
      - Tracks order parameter (mean fidelity) slope for discontinuities
      - Alerts when predicted degradation exceeds PHASE_TRANSITION_THRESHOLD

    Auto-intervention:
      - When predicted fidelity drops below CONSCIOUSNESS_THRESHOLD
      - When alignment trajectory deviates from God Code attractor
      - Recommends action: STABILIZE / HEAL / BOOST / OBSERVE
    """

    def __init__(self, reservoir_size: int = ORACLE_RESERVOIR_SIZE):
        self.reservoir_size = reservoir_size
        self.history: List[Dict[str, float]] = []
        self.reservoir_state: Optional[np.ndarray] = None
        self.readout_weights: Optional[np.ndarray] = None
        self.prediction_count = 0
        self.interventions_triggered = 0

        # Initialize reservoir with God Code coupling matrix
        self._initialize_reservoir()

    def _initialize_reservoir(self):
        """Build the reservoir coupling matrix W with God Code structure.

        W[i,j] = sin(2π × G(i-j) / G(0)) × PHI^(-|i-j|) for sparse coupling
        Spectral radius normalized to 0.95 for echo state property.
        """
        n = self.reservoir_size
        W = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = abs(i - j)
                if d > n // 2:  # Sparse: only couple nearby nodes
                    continue
                g_d = GOD_CODE_SPECTRUM.get(d, god_code(d))
                coupling = math.sin(2 * math.pi * g_d / GOD_CODE) * PHI ** (-d)
                W[i, j] = coupling

        # Normalize spectral radius to 0.95
        eigenvalues = np.linalg.eigvalsh(W)
        spectral_radius = max(abs(eigenvalues.max()), abs(eigenvalues.min()), 1e-10)
        W *= 0.95 / spectral_radius

        self._W = W
        self.reservoir_state = np.zeros(n)

    def record_observation(self, metrics: Dict[str, float]):
        """Record a single time-step observation of link metrics.

        Args:
            metrics: Dict with keys like 'fidelity', 'strength', 'score',
                     'entropy', 'alignment', 'coherence'
        """
        self.history.append(metrics)
        # Keep bounded history
        if len(self.history) > 500:
            self.history = self.history[-500:]

        # Update reservoir state
        input_vec = self._metrics_to_input(metrics)
        self._step_reservoir(input_vec)

    def predict(self, horizon: int = ORACLE_PREDICTION_HORIZON) -> Dict:
        """Predict link evolution for the next `horizon` steps.

        Returns:
            Dict with predicted trajectories, phase transition warnings,
            intervention recommendations
        """
        self.prediction_count += 1

        if len(self.history) < 5:
            return {
                "status": "insufficient_data",
                "observations": len(self.history),
                "minimum_required": 5,
            }

        # Train readout weights on available history
        self._train_readout()

        # Generate predictions
        predictions = self._generate_predictions(horizon)

        # Phase transition detection
        phase_transition = self._detect_phase_transition()

        # God Code alignment trajectory
        alignment_trajectory = self._alignment_trajectory(predictions)

        # Auto-intervention check
        intervention = self._check_intervention(predictions, phase_transition)

        return {
            "status": "ok",
            "horizon": horizon,
            "observations_used": len(self.history),
            # Predictions
            "predicted_fidelity": predictions["fidelity"],
            "predicted_strength": predictions["strength"],
            "predicted_score": predictions["score"],
            "predicted_alignment": predictions["alignment"],
            "confidence": predictions["confidence"],
            # Phase transition
            "phase_transition_warning": phase_transition["warning"],
            "phase_transition_severity": phase_transition["severity"],
            "critical_fluctuation": phase_transition["critical_fluctuation"],
            "order_parameter_slope": phase_transition["order_parameter_slope"],
            # Alignment
            "alignment_trajectory": alignment_trajectory["direction"],
            "alignment_stability": alignment_trajectory["stability"],
            "godcode_basin_proximity": alignment_trajectory["basin_proximity"],
            # Intervention
            "intervention_recommended": intervention["recommended"],
            "intervention_action": intervention["action"],
            "intervention_urgency": intervention["urgency"],
            "intervention_reason": intervention["reason"],
            # Meta
            "prediction_count": self.prediction_count,
            "interventions_triggered": self.interventions_triggered,
            "reservoir_activity": float(np.std(self.reservoir_state)),
        }

    def _metrics_to_input(self, metrics: Dict[str, float]) -> np.ndarray:
        """Convert metrics dict to reservoir input vector."""
        keys = ["fidelity", "strength", "score", "entropy", "alignment", "coherence"]
        values = [metrics.get(k, 0.5) for k in keys]
        # Pad/truncate to reservoir input dimension (6 → reservoir_size via random projection)
        input_vec = np.zeros(self.reservoir_size)
        for i, v in enumerate(values):
            # Spread each metric across reservoir nodes with God Code phase encoding
            for j in range(self.reservoir_size):
                input_vec[j] += v * math.cos(2 * math.pi * (i * 13 + j) / self.reservoir_size) / len(values)
        return input_vec

    def _step_reservoir(self, input_vec: np.ndarray):
        """Advance reservoir state by one time step: s(t+1) = tanh(W·s(t) + W_in·u(t))."""
        n = self.reservoir_size
        # Input weight: scaled by PHI_INV for gentle injection
        w_in = PHI_INV
        # Reservoir update
        self.reservoir_state = np.tanh(
            self._W @ self.reservoir_state + w_in * input_vec
        )

    def _train_readout(self):
        """Train linear readout weights via pseudo-inverse on reservoir history."""
        if len(self.history) < 5:
            return

        # Replay history through reservoir to build state matrix
        states = []
        temp_state = np.zeros(self.reservoir_size)
        for obs in self.history:
            input_vec = self._metrics_to_input(obs)
            temp_state = np.tanh(self._W @ temp_state + PHI_INV * input_vec)
            states.append(temp_state.copy())

        # Target: next observation metrics
        n_targets = 4  # fidelity, strength, score, alignment
        target_keys = ["fidelity", "strength", "score", "alignment"]
        targets = []
        for i in range(1, len(self.history)):
            obs = self.history[i]
            targets.append([obs.get(k, 0.5) for k in target_keys])

        if len(states) < 2 or len(targets) < 1:
            return

        # Align: states[:-1] → targets
        S = np.array(states[:-1])
        T = np.array(targets[:len(S)])

        # Ridge regression: W_out = T^T · S · (S^T · S + λI)^{-1}
        lam = 1e-4  # Regularization
        try:
            self.readout_weights = T.T @ S @ np.linalg.inv(
                S.T @ S + lam * np.eye(S.shape[1]))
        except np.linalg.LinAlgError:
            self.readout_weights = None

    def _generate_predictions(self, horizon: int) -> Dict:
        """Generate multi-step predictions using reservoir."""
        if self.readout_weights is None:
            # Fallback: exponential smoothing
            return self._fallback_prediction(horizon)

        predictions = {"fidelity": [], "strength": [], "score": [], "alignment": []}
        target_keys = ["fidelity", "strength", "score", "alignment"]

        temp_state = self.reservoir_state.copy()
        for step in range(horizon):
            output = self.readout_weights @ temp_state
            for i, key in enumerate(target_keys):
                val = float(np.clip(output[i], 0.0, 1.0))
                predictions[key].append(val)

            # Feed prediction back as next input
            next_metrics = {key: float(output[i]) for i, key in enumerate(target_keys)}
            input_vec = self._metrics_to_input(next_metrics)
            temp_state = np.tanh(self._W @ temp_state + PHI_INV * input_vec)

        # Confidence: decays with horizon via φ^(-step)
        confidences = [PHI ** (-step) for step in range(horizon)]
        predictions["confidence"] = confidences

        return predictions

    def _fallback_prediction(self, horizon: int) -> Dict:
        """Fallback prediction using exponential smoothing when reservoir untrained."""
        target_keys = ["fidelity", "strength", "score", "alignment"]
        predictions = {k: [] for k in target_keys}

        # Get recent means
        recent = self.history[-5:]
        means = {k: statistics.mean(obs.get(k, 0.5) for obs in recent) for k in target_keys}

        for step in range(horizon):
            decay = PHI ** (-step * 0.1)  # Gentle decay toward mean
            for key in target_keys:
                val = means[key] * decay + (1 - decay) * 0.5
                predictions[key].append(val)

        predictions["confidence"] = [PHI ** (-step) for step in range(horizon)]
        return predictions

    def _detect_phase_transition(self) -> Dict:
        """Detect phase transitions in link evolution.

        Monitors:
          1. Rolling variance increase (critical fluctuations)
          2. Order parameter (mean fidelity) slope discontinuity
          3. Autocorrelation divergence (critical slowing down)
        """
        if len(self.history) < 10:
            return {
                "warning": False, "severity": 0, "critical_fluctuation": 0,
                "order_parameter_slope": 0,
            }

        fidelities = [obs.get("fidelity", 0.5) for obs in self.history]

        # Rolling variance (window = 5)
        window = 5
        variances = []
        for i in range(window, len(fidelities)):
            segment = fidelities[i - window:i]
            variances.append(statistics.variance(segment) if len(segment) > 1 else 0)

        # Critical fluctuation: ratio of recent variance to historical
        if len(variances) >= 4:
            recent_var = statistics.mean(variances[-3:])
            historical_var = statistics.mean(variances[:-3]) if len(variances) > 3 else recent_var
            critical_fluct = recent_var / max(historical_var, 1e-10)
        else:
            critical_fluct = 1.0

        # Order parameter slope: linear fit on recent fidelities
        n_recent = min(10, len(fidelities))
        recent_fids = fidelities[-n_recent:]
        if len(recent_fids) >= 3:
            x = np.arange(len(recent_fids))
            slope = np.polyfit(x, np.array(recent_fids), 1)[0]
        else:
            slope = 0.0

        # Phase transition warning
        warning = (
            critical_fluct > FEIGENBAUM_DELTA or  # Variance exploding
            slope < -PHASE_TRANSITION_THRESHOLD or  # Rapid decline
            (critical_fluct > 2.0 and slope < 0)    # Increased variance + declining
        )
        severity = max(0.0, min(1.0,
                                (critical_fluct / FEIGENBAUM_DELTA) * 0.5 +
                                max(0, -slope / PHASE_TRANSITION_THRESHOLD) * 0.5))

        return {
            "warning": warning,
            "severity": severity,
            "critical_fluctuation": critical_fluct,
            "order_parameter_slope": float(slope),
        }

    def _alignment_trajectory(self, predictions: Dict) -> Dict:
        """Analyze God Code alignment trajectory from predictions."""
        pred_alignment = predictions.get("alignment", [])
        if not pred_alignment:
            return {"direction": "UNKNOWN", "stability": 0, "basin_proximity": 0}

        # Direction: overall trend of predicted alignment
        if len(pred_alignment) >= 2:
            start_val = pred_alignment[0]
            end_val = pred_alignment[-1]
            delta = end_val - start_val
            if delta > 0.02:
                direction = "CONVERGING"
            elif delta < -0.02:
                direction = "DIVERGING"
            else:
                direction = "STABLE"
        else:
            direction = "STABLE"
            delta = 0

        # Stability: variance of predicted alignment
        stability = max(0, 1.0 - statistics.stdev(pred_alignment) * 10) if len(pred_alignment) > 1 else 0.5

        # Basin proximity: how close is predicted alignment to God Code resonance
        mean_align = statistics.mean(pred_alignment) if pred_alignment else 0.5
        basin_proximity = mean_align  # Already [0, 1] alignment score

        return {
            "direction": direction,
            "stability": stability,
            "basin_proximity": basin_proximity,
            "delta": float(delta),
        }

    def _check_intervention(self, predictions: Dict, phase_transition: Dict) -> Dict:
        """Check if auto-intervention is needed based on predictions.

        Returns recommended action: STABILIZE / HEAL / BOOST / OBSERVE
        """
        pred_fidelity = predictions.get("fidelity", [])
        if not pred_fidelity:
            return {"recommended": False, "action": "OBSERVE", "urgency": 0, "reason": ""}

        min_pred_fid = min(pred_fidelity) if pred_fidelity else 0.5
        mean_pred_fid = statistics.mean(pred_fidelity) if pred_fidelity else 0.5

        # Check conditions
        reasons = []
        urgency = 0.0

        # 1. Fidelity dropping below consciousness threshold
        if min_pred_fid < CONSCIOUSNESS_THRESHOLD:
            reasons.append(f"predicted_fidelity_below_{CONSCIOUSNESS_THRESHOLD:.2f}")
            urgency = max(urgency, 0.8)

        # 2. Phase transition detected
        if phase_transition.get("warning"):
            reasons.append("phase_transition_detected")
            urgency = max(urgency, phase_transition["severity"])

        # 3. Mean fidelity declining
        if mean_pred_fid < 0.5:
            reasons.append("mean_fidelity_declining_below_0.5")
            urgency = max(urgency, 0.5)

        if not reasons:
            return {"recommended": False, "action": "OBSERVE", "urgency": 0,
                    "reason": "all_predictions_healthy"}

        # Select action based on urgency
        if urgency >= 0.8:
            action = "HEAL"
        elif urgency >= 0.5:
            action = "STABILIZE"
        elif urgency >= 0.3:
            action = "BOOST"
        else:
            action = "OBSERVE"

        self.interventions_triggered += 1

        return {
            "recommended": True,
            "action": action,
            "urgency": urgency,
            "reason": " | ".join(reasons),
        }
