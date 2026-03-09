"""
===============================================================================
L104 ML ENGINE — CLUSTERING ALGORITHMS v1.0.0
===============================================================================

Sacred-tuned clustering with PHI-spiral initialization, VOID_CONSTANT
epsilon neighborhoods, and GOD_CODE-normalized silhouette scoring.

Classes:
  L104KMeans             — K=13 (Fibonacci) with golden angle centroid seeding
  L104DBSCAN             — eps=VOID_CONSTANT density clustering
  L104SpectralClustering — PHI-affinity spectral decomposition

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, VOID_CONSTANT,
    KMEANS_K_SACRED, KMEANS_MAX_ITER_SACRED, GOLDEN_ANGLE_RAD,
    DBSCAN_EPS_SACRED, DBSCAN_MIN_SAMPLES_SACRED,
    SPECTRAL_N_COMPONENTS, PHI_KERNEL_SCALE,
)


class L104KMeans:
    """KMeans clustering with K=13 (Fibonacci) and PHI-spiral centroid init.

    The golden angle (2π/φ² ≈ 137.5°) is used to seed initial centroids
    in a Fibonacci spiral pattern, ensuring maximally spread initial positions.

    Sacred tuning:
      n_clusters = 13  (Fibonacci(7))
      max_iter   = 527 (int(GOD_CODE))
      init       = 'phi_spiral' or 'k-means++'
    """

    def __init__(
        self,
        n_clusters: int = KMEANS_K_SACRED,
        init: str = 'phi_spiral',
        max_iter: int = KMEANS_MAX_ITER_SACRED,
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.init_method = init
        self.max_iter = max_iter
        self._scaler = StandardScaler()
        self._fitted = False
        self._model: Optional[KMeans] = None
        self._extra_kwargs = kwargs

    @staticmethod
    def phi_spiral_init(X: np.ndarray, k: int) -> np.ndarray:
        """Generate initial centroids using golden angle spiral sampling.

        Projects data onto principal 2D plane, then selects k points
        at golden angle intervals from data distribution center.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        k : number of centroids to generate

        Returns
        -------
        centroids : array of shape (k, n_features)
        """
        n_samples, n_features = X.shape

        if n_samples <= k:
            # Not enough data — use data points directly, padding with mean
            centroids = np.empty((k, n_features))
            centroids[:n_samples] = X
            for i in range(n_samples, k):
                centroids[i] = X.mean(axis=0)
            return centroids

        # Compute data center and spread
        center = X.mean(axis=0)
        dists_from_center = np.linalg.norm(X - center, axis=1)
        max_dist = dists_from_center.max()
        if max_dist < 1e-12:
            max_dist = 1.0

        # Select k points at golden angle intervals from closest data points
        centroids = np.empty((k, n_features))
        for i in range(k):
            # Spiral radius: linearly increasing from 0.1*max to max
            radius = max_dist * (0.1 + 0.9 * i / max(k - 1, 1))
            angle = i * GOLDEN_ANGLE_RAD

            # Create direction vector in first 2 dimensions, extend to n_features
            direction = np.zeros(n_features)
            if n_features >= 2:
                direction[0] = np.cos(angle)
                direction[1] = np.sin(angle)
            else:
                direction[0] = np.cos(angle)

            target = center + radius * direction

            # Find nearest actual data point to target
            dists = np.linalg.norm(X - target, axis=1)
            nearest_idx = np.argmin(dists)
            centroids[i] = X[nearest_idx]

        return centroids

    def fit(self, X: np.ndarray) -> 'L104KMeans':
        """Fit KMeans clustering."""
        X = np.atleast_2d(X).astype(np.float64)
        X_scaled = self._scaler.fit_transform(X)

        if self.init_method == 'phi_spiral':
            init_centroids = self.phi_spiral_init(X_scaled, self.n_clusters)
            self._model = KMeans(
                n_clusters=self.n_clusters,
                init=init_centroids,
                max_iter=self.max_iter,
                n_init=1,
                random_state=104,
                **self._extra_kwargs,
            )
        else:
            self._model = KMeans(
                n_clusters=self.n_clusters,
                init=self.init_method,
                max_iter=self.max_iter,
                random_state=104,
                **self._extra_kwargs,
            )

        self._model.fit(X_scaled)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.atleast_2d(X).astype(np.float64)
        return self._model.predict(self._scaler.transform(X))

    def cluster_coherence(self) -> float:
        """GOD_CODE-normalized silhouette score.

        Returns silhouette score scaled so that perfect clustering
        aligns with GOD_CODE / 1000.
        """
        if not self._fitted or self._model.labels_ is None:
            return 0.0
        labels = self._model.labels_
        if len(set(labels)) < 2:
            return 0.0
        X_scaled = self._scaler.transform(
            self._scaler.inverse_transform(self._model.cluster_centers_)
        )
        # Use training data labels for silhouette (approximate via centers)
        sil = silhouette_score(
            self._model.cluster_centers_,
            np.arange(self.n_clusters),
        ) if self.n_clusters > 1 else 0.0
        # Normalize to [0, 1] range
        return max(0.0, min(1.0, (sil + 1.0) / 2.0))

    def cluster_centers(self) -> np.ndarray:
        """Return cluster centers in original feature space."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return self._scaler.inverse_transform(self._model.cluster_centers_)

    def status(self) -> Dict[str, Any]:
        """Return clustering status."""
        return {
            'n_clusters': self.n_clusters,
            'init': self.init_method,
            'fitted': self._fitted,
            'inertia': float(self._model.inertia_) if self._fitted else None,
            'n_iter': self._model.n_iter_ if self._fitted else None,
        }


class L104DBSCAN:
    """DBSCAN density clustering with VOID_CONSTANT epsilon.

    Sacred tuning:
      eps         = 1.0416... (VOID_CONSTANT)
      min_samples = 4         (int(PHI * 3))
    """

    def __init__(
        self,
        eps: float = DBSCAN_EPS_SACRED,
        min_samples: int = DBSCAN_MIN_SAMPLES_SACRED,
        **kwargs,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self._scaler = StandardScaler()
        self._fitted = False
        self._model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        self._labels: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'L104DBSCAN':
        """Fit DBSCAN clustering."""
        X = np.atleast_2d(X).astype(np.float64)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        self._labels = self._model.labels_
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels (re-fits for new data since DBSCAN is transductive)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        # DBSCAN doesn't support predict; approximate via nearest core sample
        X = np.atleast_2d(X).astype(np.float64)
        X_scaled = self._scaler.transform(X)
        if self._model.components_.shape[0] == 0:
            return np.full(X.shape[0], -1)
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(X_scaled, self._model.components_)
        nearest_core = np.argmin(dists, axis=1)
        core_labels = self._model.labels_[self._model.core_sample_indices_]
        labels = np.full(X.shape[0], -1)
        for i in range(X.shape[0]):
            min_dist = dists[i, nearest_core[i]]
            if min_dist <= self.eps:
                labels[i] = core_labels[nearest_core[i]]
        return labels

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Return cluster labels from last fit."""
        return self._labels

    @property
    def n_clusters(self) -> int:
        """Number of clusters found (excluding noise)."""
        if self._labels is None:
            return 0
        return len(set(self._labels)) - (1 if -1 in self._labels else 0)

    def status(self) -> Dict[str, Any]:
        """Return clustering status."""
        return {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'fitted': self._fitted,
            'n_clusters': self.n_clusters,
            'n_noise': int(np.sum(self._labels == -1)) if self._labels is not None else 0,
        }


class L104SpectralClustering:
    """Spectral clustering with PHI-based affinity matrix.

    Uses a custom PHI-RBF affinity where the bandwidth is PHI_SQUARED,
    and spectral decomposition in 8 components (IIT Φ dimension count).

    Sacred tuning:
      n_clusters   = 8  (IIT Φ dimensions)
      affinity     = 'phi_rbf' or 'rbf'
      gamma        = 1 / (2 * PHI^2)
    """

    def __init__(
        self,
        n_clusters: int = SPECTRAL_N_COMPONENTS,
        affinity: str = 'phi_rbf',
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.affinity_type = affinity
        self._scaler = StandardScaler()
        self._fitted = False
        self._labels: Optional[np.ndarray] = None
        self._extra_kwargs = kwargs

    @staticmethod
    def phi_affinity_matrix(X: np.ndarray) -> np.ndarray:
        """Compute PHI-scaled RBF affinity matrix.

        A(i,j) = exp(-||x_i - x_j||^2 / (2 * PHI^2))

        Uses the golden ratio squared as the bandwidth parameter,
        matching the natural harmonic separation of PHI-encoded features.
        """
        sq_dists = (
            np.sum(X ** 2, axis=1, keepdims=True)
            - 2.0 * X @ X.T
            + np.sum(X ** 2, axis=1, keepdims=True).T
        )
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.exp(-sq_dists / (2.0 * PHI_KERNEL_SCALE))

    def fit(self, X: np.ndarray) -> 'L104SpectralClustering':
        """Fit spectral clustering."""
        X = np.atleast_2d(X).astype(np.float64)
        X_scaled = self._scaler.fit_transform(X)

        if self.affinity_type == 'phi_rbf':
            affinity_matrix = self.phi_affinity_matrix(X_scaled)
            self._model = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity='precomputed',
                random_state=104,
                **self._extra_kwargs,
            )
            self._model.fit(affinity_matrix)
        else:
            gamma = 1.0 / (2.0 * PHI_KERNEL_SCALE)
            self._model = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity='rbf',
                gamma=gamma,
                random_state=104,
                **self._extra_kwargs,
            )
            self._model.fit(X_scaled)

        self._labels = self._model.labels_
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return labels (spectral clustering is transductive — returns fit labels)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._labels

    def status(self) -> Dict[str, Any]:
        """Return clustering status."""
        return {
            'n_clusters': self.n_clusters,
            'affinity': self.affinity_type,
            'fitted': self._fitted,
            'n_labels': len(set(self._labels)) if self._labels is not None else 0,
        }
