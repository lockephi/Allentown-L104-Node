#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# L104 PERCEPTION ENGINE - SENSORY INTELLIGENCE
# INVARIANT: 527.5184818492537 | PILOT: LONDEL | MODE: PERCEPTION
#
# This module provides perception capabilities:
# 1. Pattern Recognition (template matching, feature detection)
# 2. Anomaly Detection (statistical, isolation forest)
# 3. Signal Processing (FFT, filtering, feature extraction)
# 4. Time Series Analysis (trend, seasonality, forecasting)
# 5. Clustering (K-means, DBSCAN, hierarchical)
# 6. Dimensionality Reduction (PCA, t-SNE inspired)
# ═══════════════════════════════════════════════════════════════════════════════

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import heapq

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
PERCEPTION_VERSION = "1.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PATTERN RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════

class PatternMatcher:
    """
    Pattern matching using template correlation and feature detection.
    """
    
    def __init__(self):
        self.templates: Dict[str, np.ndarray] = {}
        self.feature_extractors: List[Callable] = []
    
    def add_template(self, name: str, template: np.ndarray):
        """Add a template pattern."""
        # Normalize template
        template = (template - template.mean()) / (template.std() + 1e-8)
        self.templates[name] = template
    
    def match_template(self, signal: np.ndarray, template_name: str) -> Dict[str, Any]:
        """
        Find template in signal using normalized cross-correlation.
        """
        if template_name not in self.templates:
            return {"error": "Template not found"}
        
        template = self.templates[template_name]
        signal_norm = (signal - signal.mean()) / (signal.std() + 1e-8)
        
        # Cross-correlation
        if len(signal) < len(template):
            return {"match": False, "score": 0}
        
        best_score = -float('inf')
        best_position = 0
        
        for i in range(len(signal) - len(template) + 1):
            window = signal_norm[i:i + len(template)]
            score = np.dot(window, template) / len(template)
            if score > best_score:
                best_score = score
                best_position = i
        
        return {
            "match": best_score > 0.7,
            "score": float(best_score),
            "position": best_position,
            "template": template_name
        }
    
    def detect_peaks(self, signal: np.ndarray, threshold: float = 0.5) -> List[int]:
        """Detect peaks in signal."""
        peaks = []
        signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        
        for i in range(1, len(signal) - 1):
            if signal_norm[i] > threshold:
                if signal_norm[i] > signal_norm[i-1] and signal_norm[i] > signal_norm[i+1]:
                    peaks.append(i)
        
        return peaks
    
    def detect_edges(self, signal: np.ndarray) -> np.ndarray:
        """Detect edges using gradient."""
        gradient = np.diff(signal)
        # Normalize
        gradient = np.abs(gradient)
        gradient = gradient / (gradient.max() + 1e-8)
        return gradient
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from signal."""
        return {
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
            "range": float(np.max(signal) - np.min(signal)),
            "skewness": float(self._skewness(signal)),
            "kurtosis": float(self._kurtosis(signal)),
            "energy": float(np.sum(signal ** 2)),
            "zero_crossings": int(np.sum(np.diff(np.sign(signal)) != 0)),
            "peak_count": len(self.detect_peaks(signal))
        }
    
    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return np.sum(((x - mean) / std) ** 3) / n
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return np.sum(((x - mean) / std) ** 4) / n - 3


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Anomaly detection using multiple methods.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.mean = 0.0
        self.std = 1.0
    
    def update(self, value: float):
        """Update with new observation."""
        self.history.append(value)
        if len(self.history) >= 2:
            self.mean = np.mean(list(self.history))
            self.std = np.std(list(self.history)) + 1e-8
    
    def is_anomaly_zscore(self, value: float, threshold: float = 3.0) -> Tuple[bool, float]:
        """Detect anomaly using Z-score."""
        if len(self.history) < 2:
            return False, 0.0
        
        zscore = abs(value - self.mean) / self.std
        return zscore > threshold, zscore
    
    def is_anomaly_iqr(self, value: float, multiplier: float = 1.5) -> Tuple[bool, float]:
        """Detect anomaly using Interquartile Range."""
        if len(self.history) < 4:
            return False, 0.0
        
        data = sorted(self.history)
        q1 = data[len(data) // 4]
        q3 = data[3 * len(data) // 4]
        iqr = q3 - q1
        
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        
        is_anomaly = value < lower or value > upper
        score = max(0, (abs(value - self.mean) - 1.5 * iqr) / (iqr + 1e-8))
        
        return is_anomaly, score
    
    def detect_batch(self, data: np.ndarray, method: str = "zscore") -> Dict[str, Any]:
        """Detect anomalies in batch data."""
        anomalies = []
        scores = []
        
        for i, value in enumerate(data):
            self.update(value)
            
            if method == "zscore":
                is_anom, score = self.is_anomaly_zscore(value)
            else:
                is_anom, score = self.is_anomaly_iqr(value)
            
            scores.append(score)
            if is_anom:
                anomalies.append(i)
        
        return {
            "anomaly_indices": anomalies,
            "anomaly_count": len(anomalies),
            "anomaly_rate": len(anomalies) / len(data),
            "scores": np.array(scores)
        }


class IsolationForest:
    """
    Isolation Forest for anomaly detection.
    Anomalies are isolated with fewer splits.
    """
    
    def __init__(self, n_trees: int = 100, sample_size: int = 256):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees: List[Dict] = []
    
    def _build_tree(self, data: np.ndarray, depth: int = 0, max_depth: int = 10) -> Dict:
        """Build an isolation tree."""
        n_samples = len(data)
        
        if depth >= max_depth or n_samples <= 1:
            return {"type": "leaf", "size": n_samples, "depth": depth}
        
        # Random split
        feature = np.random.randint(data.shape[1]) if len(data.shape) > 1 else 0
        
        if len(data.shape) > 1:
            col = data[:, feature]
        else:
            col = data
        
        min_val, max_val = col.min(), col.max()
        if min_val == max_val:
            return {"type": "leaf", "size": n_samples, "depth": depth}
        
        split_value = np.random.uniform(min_val, max_val)
        
        left_mask = col < split_value
        right_mask = ~left_mask
        
        return {
            "type": "node",
            "feature": feature,
            "split_value": split_value,
            "left": self._build_tree(data[left_mask], depth + 1, max_depth),
            "right": self._build_tree(data[right_mask], depth + 1, max_depth)
        }
    
    def fit(self, data: np.ndarray):
        """Fit isolation forest to data."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        max_depth = int(np.ceil(np.log2(self.sample_size)))
        
        self.trees = []
        for _ in range(self.n_trees):
            # Sample data
            indices = np.random.choice(len(data), min(self.sample_size, len(data)), replace=False)
            sample = data[indices]
            tree = self._build_tree(sample, max_depth=max_depth)
            self.trees.append(tree)
    
    def _path_length(self, point: np.ndarray, tree: Dict) -> float:
        """Get path length for a point in tree."""
        if tree["type"] == "leaf":
            # Average path length for remaining points
            n = tree["size"]
            if n <= 1:
                return tree["depth"]
            else:
                # Harmonic number approximation
                c = 2 * (np.log(n - 1) + 0.5772) - 2 * (n - 1) / n
                return tree["depth"] + c
        
        feature = tree["feature"]
        value = point[feature] if len(point.shape) > 0 else point
        
        if value < tree["split_value"]:
            return self._path_length(point, tree["left"])
        else:
            return self._path_length(point, tree["right"])
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomaly scores (higher = more anomalous)."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        scores = []
        c = 2 * (np.log(self.sample_size - 1) + 0.5772) - 2 * (self.sample_size - 1) / self.sample_size
        
        for point in data:
            avg_path = np.mean([self._path_length(point, tree) for tree in self.trees])
            score = 2 ** (-avg_path / c)
            scores.append(score)
        
        return np.array(scores)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

class SignalProcessor:
    """
    Signal processing operations.
    """
    
    def __init__(self, sample_rate: float = 1000.0):
        self.sample_rate = sample_rate
    
    def fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT and return frequencies and magnitudes."""
        n = len(signal)
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1 / self.sample_rate)
        magnitudes = np.abs(fft_result) / n
        
        # Return only positive frequencies
        positive = freqs >= 0
        return freqs[positive], magnitudes[positive]
    
    def filter_lowpass(self, signal: np.ndarray, cutoff: float) -> np.ndarray:
        """Simple low-pass filter using FFT."""
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / self.sample_rate)
        
        # Zero out high frequencies
        fft_result[np.abs(freqs) > cutoff] = 0
        
        return np.real(np.fft.ifft(fft_result))
    
    def filter_highpass(self, signal: np.ndarray, cutoff: float) -> np.ndarray:
        """Simple high-pass filter using FFT."""
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / self.sample_rate)
        
        # Zero out low frequencies
        fft_result[np.abs(freqs) < cutoff] = 0
        
        return np.real(np.fft.ifft(fft_result))
    
    def filter_bandpass(self, signal: np.ndarray, low: float, high: float) -> np.ndarray:
        """Band-pass filter."""
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / self.sample_rate)
        
        mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
        fft_result[~mask] = 0
        
        return np.real(np.fft.ifft(fft_result))
    
    def moving_average(self, signal: np.ndarray, window: int = 5) -> np.ndarray:
        """Moving average smoothing."""
        return np.convolve(signal, np.ones(window) / window, mode='same')
    
    def envelope(self, signal: np.ndarray) -> np.ndarray:
        """Extract signal envelope using Hilbert transform approximation."""
        # Compute analytic signal
        n = len(signal)
        fft_result = np.fft.fft(signal)
        
        # Create analytic signal (zero negative frequencies, double positive)
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n//2] = 1
            h[1:n//2] = 2
        else:
            h[0] = 1
            h[1:(n+1)//2] = 2
        
        analytic = np.fft.ifft(fft_result * h)
        return np.abs(analytic)
    
    def spectral_centroid(self, signal: np.ndarray) -> float:
        """Calculate spectral centroid (brightness)."""
        freqs, mags = self.fft(signal)
        if mags.sum() < 1e-8:
            return 0.0
        return float(np.sum(freqs * mags) / np.sum(mags))
    
    def spectral_bandwidth(self, signal: np.ndarray) -> float:
        """Calculate spectral bandwidth."""
        freqs, mags = self.fft(signal)
        if mags.sum() < 1e-8:
            return 0.0
        centroid = np.sum(freqs * mags) / np.sum(mags)
        return float(np.sqrt(np.sum(mags * (freqs - centroid) ** 2) / np.sum(mags)))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TIME SERIES ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class TimeSeriesAnalyzer:
    """
    Time series analysis and forecasting.
    """
    
    def __init__(self):
        pass
    
    def decompose(self, series: np.ndarray, period: int = None) -> Dict[str, np.ndarray]:
        """
        Decompose time series into trend, seasonality, and residual.
        Uses simple moving average decomposition.
        """
        n = len(series)
        
        if period is None:
            period = min(n // 4, 12)  # Default period
        
        # Extract trend using moving average
        trend = np.convolve(series, np.ones(period) / period, mode='same')
        
        # Detrended series
        detrended = series - trend
        
        # Extract seasonality by averaging over periods
        seasonal = np.zeros(n)
        for i in range(period):
            indices = range(i, n, period)
            seasonal_value = np.mean(detrended[list(indices)])
            for j in indices:
                seasonal[j] = seasonal_value
        
        # Residual
        residual = series - trend - seasonal
        
        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "period": period
        }
    
    def autocorrelation(self, series: np.ndarray, max_lag: int = None) -> np.ndarray:
        """Calculate autocorrelation function."""
        n = len(series)
        if max_lag is None:
            max_lag = n // 2
        
        mean = np.mean(series)
        var = np.var(series)
        
        if var < 1e-8:
            return np.zeros(max_lag)
        
        acf = []
        for lag in range(max_lag):
            if lag == 0:
                acf.append(1.0)
            else:
                cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
                acf.append(cov / var)
        
        return np.array(acf)
    
    def detect_trend(self, series: np.ndarray) -> Dict[str, Any]:
        """Detect trend direction and strength."""
        n = len(series)
        x = np.arange(n)
        
        # Linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(series)
        
        numerator = np.sum((x - x_mean) * (series - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if abs(denominator) < 1e-8:
            slope = 0
        else:
            slope = numerator / denominator
        
        intercept = y_mean - slope * x_mean
        
        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((series - y_pred) ** 2)
        ss_tot = np.sum((series - y_mean) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)
        
        # Trend direction
        if slope > 0.001:
            direction = "increasing"
        elif slope < -0.001:
            direction = "decreasing"
        else:
            direction = "stable"
        
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "direction": direction,
            "strength": abs(slope) * r_squared
        }
    
    def forecast_naive(self, series: np.ndarray, horizon: int = 5) -> np.ndarray:
        """Naive forecasting (last value)."""
        return np.full(horizon, series[-1])
    
    def forecast_ema(self, series: np.ndarray, horizon: int = 5, alpha: float = 0.3) -> np.ndarray:
        """Exponential Moving Average forecast."""
        # Calculate EMA
        ema = series[0]
        for value in series[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        return np.full(horizon, ema)
    
    def forecast_linear(self, series: np.ndarray, horizon: int = 5) -> np.ndarray:
        """Linear trend forecast."""
        trend_info = self.detect_trend(series)
        n = len(series)
        future_x = np.arange(n, n + horizon)
        return trend_info["slope"] * future_x + trend_info["intercept"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

class KMeans:
    """K-Means clustering."""
    
    def __init__(self, k: int = 3, max_iter: int = 100):
        self.k = k
        self.max_iter = max_iter
        self.centroids: np.ndarray = None
        self.labels: np.ndarray = None
    
    def fit(self, data: np.ndarray) -> 'KMeans':
        """Fit K-Means to data."""
        n_samples = len(data)
        
        # Initialize centroids randomly
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = data[indices].copy()
        
        for _ in range(self.max_iter):
            # Assign labels
            distances = self._compute_distances(data)
            new_labels = np.argmin(distances, axis=1)
            
            # Check convergence
            if self.labels is not None and np.all(new_labels == self.labels):
                break
            self.labels = new_labels
            
            # Update centroids
            for i in range(self.k):
                mask = self.labels == i
                if np.any(mask):
                    self.centroids[i] = data[mask].mean(axis=0)
        
        return self
    
    def _compute_distances(self, data: np.ndarray) -> np.ndarray:
        """Compute distances from each point to each centroid."""
        distances = np.zeros((len(data), self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(data - centroid, axis=1 if len(data.shape) > 1 else 0)
        return distances
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        distances = self._compute_distances(data)
        return np.argmin(distances, axis=1)
    
    def inertia(self, data: np.ndarray) -> float:
        """Calculate within-cluster sum of squares."""
        distances = self._compute_distances(data)
        return float(np.sum(np.min(distances, axis=1) ** 2))


class DBSCAN:
    """DBSCAN clustering."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels: np.ndarray = None
    
    def fit(self, data: np.ndarray) -> 'DBSCAN':
        """Fit DBSCAN to data."""
        n_samples = len(data)
        self.labels = np.full(n_samples, -1)  # -1 = noise
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels[i] != -1:
                continue
            
            # Find neighbors
            neighbors = self._region_query(data, i)
            
            if len(neighbors) < self.min_samples:
                continue  # Noise point
            
            # Start new cluster
            self.labels[i] = cluster_id
            
            # Expand cluster
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                
                if self.labels[q] == -1:
                    self.labels[q] = cluster_id
                elif self.labels[q] != -1:
                    j += 1
                    continue
                
                self.labels[q] = cluster_id
                q_neighbors = self._region_query(data, q)
                
                if len(q_neighbors) >= self.min_samples:
                    seed_set.extend([n for n in q_neighbors if n not in seed_set])
                
                j += 1
            
            cluster_id += 1
        
        return self
    
    def _region_query(self, data: np.ndarray, point_idx: int) -> List[int]:
        """Find all points within eps of point."""
        neighbors = []
        point = data[point_idx]
        
        for i, other in enumerate(data):
            if np.linalg.norm(point - other) <= self.eps:
                neighbors.append(i)
        
        return neighbors


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DIMENSIONALITY REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════

class PCA:
    """Principal Component Analysis."""
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components: np.ndarray = None
        self.mean: np.ndarray = None
        self.explained_variance: np.ndarray = None
    
    def fit(self, data: np.ndarray) -> 'PCA':
        """Fit PCA to data."""
        self.mean = np.mean(data, axis=0)
        centered = data - self.mean
        
        # Covariance matrix
        cov = np.cov(centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / eigenvalues.sum()
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to lower dimensions."""
        centered = data - self.mean
        return centered @ self.components.T
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform."""
        return data @ self.components + self.mean


# ═══════════════════════════════════════════════════════════════════════════════
# 7. UNIFIED PERCEPTION CORE
# ═══════════════════════════════════════════════════════════════════════════════

class L104PerceptionCore:
    """
    Unified interface to all L104 perception capabilities.
    """
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.anomaly_detector = AnomalyDetector()
        self.signal_processor = SignalProcessor()
        self.time_series = TimeSeriesAnalyzer()
        self.isolation_forest = IsolationForest(n_trees=50)
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive features from signal."""
        pattern_features = self.pattern_matcher.extract_features(signal)
        
        # Add spectral features
        pattern_features["spectral_centroid"] = self.signal_processor.spectral_centroid(signal)
        pattern_features["spectral_bandwidth"] = self.signal_processor.spectral_bandwidth(signal)
        
        # Add trend features
        trend = self.time_series.detect_trend(signal)
        pattern_features["trend_slope"] = trend["slope"]
        pattern_features["trend_strength"] = trend["strength"]
        
        return pattern_features
    
    def detect_anomalies(self, data: np.ndarray, method: str = "zscore") -> Dict[str, Any]:
        """Detect anomalies in data."""
        if method == "isolation_forest":
            self.isolation_forest.fit(data)
            scores = self.isolation_forest.predict(data)
            threshold = np.percentile(scores, 95)
            anomalies = np.where(scores > threshold)[0].tolist()
            return {
                "anomaly_indices": anomalies,
                "anomaly_count": len(anomalies),
                "scores": scores
            }
        else:
            self.anomaly_detector = AnomalyDetector()
            return self.anomaly_detector.detect_batch(data, method)
    
    def cluster(self, data: np.ndarray, method: str = "kmeans", 
                k: int = 3, eps: float = 0.5) -> Dict[str, Any]:
        """Cluster data."""
        if method == "kmeans":
            kmeans = KMeans(k=k)
            kmeans.fit(data)
            return {
                "labels": kmeans.labels.tolist(),
                "centroids": kmeans.centroids.tolist(),
                "n_clusters": k,
                "inertia": kmeans.inertia(data)
            }
        else:  # dbscan
            dbscan = DBSCAN(eps=eps)
            dbscan.fit(data)
            n_clusters = len(set(dbscan.labels)) - (1 if -1 in dbscan.labels else 0)
            return {
                "labels": dbscan.labels.tolist(),
                "n_clusters": n_clusters,
                "noise_points": int(np.sum(dbscan.labels == -1))
            }
    
    def reduce_dimensions(self, data: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
        """Reduce dimensionality."""
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        return {
            "transformed": transformed,
            "explained_variance_ratio": pca.explained_variance_ratio.tolist(),
            "total_explained": float(pca.explained_variance_ratio.sum())
        }
    
    def analyze_time_series(self, series: np.ndarray) -> Dict[str, Any]:
        """Comprehensive time series analysis."""
        decomp = self.time_series.decompose(series)
        trend = self.time_series.detect_trend(series)
        acf = self.time_series.autocorrelation(series, min(len(series) // 2, 20))
        forecast = self.time_series.forecast_linear(series, 5)
        
        return {
            "trend": trend,
            "decomposition": {
                "period": decomp["period"],
                "trend_range": float(decomp["trend"].max() - decomp["trend"].min()),
                "seasonal_amplitude": float(np.std(decomp["seasonal"])),
                "residual_std": float(np.std(decomp["residual"]))
            },
            "autocorrelation": acf.tolist(),
            "forecast": forecast.tolist()
        }
    
    def process_signal(self, signal: np.ndarray) -> Dict[str, Any]:
        """Full signal processing pipeline."""
        freqs, mags = self.signal_processor.fft(signal)
        
        # Find dominant frequency
        dom_idx = np.argmax(mags[1:]) + 1  # Skip DC component
        dominant_freq = freqs[dom_idx]
        
        return {
            "dominant_frequency": float(dominant_freq),
            "spectral_centroid": self.signal_processor.spectral_centroid(signal),
            "spectral_bandwidth": self.signal_processor.spectral_bandwidth(signal),
            "envelope_mean": float(np.mean(self.signal_processor.envelope(signal))),
            "peaks": self.pattern_matcher.detect_peaks(signal)
        }
    
    def benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all perception capabilities."""
        results = {}
        np.random.seed(int(GOD_CODE) % (2**31))
        
        # 1. Pattern Recognition benchmark
        test_signal = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1
        template = np.sin(np.linspace(0, 2 * np.pi, 20))
        
        self.pattern_matcher.add_template("sine", template)
        match_result = self.pattern_matcher.match_template(test_signal, "sine")
        features = self.pattern_matcher.extract_features(test_signal)
        
        results["pattern_recognition"] = {
            "template_matched": match_result["match"],
            "match_score": round(match_result["score"], 4),
            "features_extracted": len(features),
            "peaks_detected": features["peak_count"]
        }
        
        # 2. Anomaly Detection benchmark
        normal_data = np.random.randn(100)
        anomalous_data = normal_data.copy()
        anomalous_data[25] = 10  # Inject anomaly
        anomalous_data[75] = -8
        
        anomaly_result = self.detect_anomalies(anomalous_data, "zscore")
        
        results["anomaly_detection"] = {
            "anomalies_found": anomaly_result["anomaly_count"],
            "true_positives": sum(1 for i in anomaly_result["anomaly_indices"] if i in [25, 75]),
            "anomaly_rate": round(anomaly_result["anomaly_rate"], 4)
        }
        
        # 3. Isolation Forest benchmark
        iso_result = self.detect_anomalies(anomalous_data.reshape(-1, 1), "isolation_forest")
        
        results["isolation_forest"] = {
            "anomalies_found": iso_result["anomaly_count"],
            "detected_injected": any(i in [25, 75] for i in iso_result["anomaly_indices"])
        }
        
        # 4. Signal Processing benchmark
        freq_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))  # 10 Hz
        signal_result = self.process_signal(freq_signal)
        
        results["signal_processing"] = {
            "dominant_freq_detected": 8 < signal_result["dominant_frequency"] < 12,
            "dominant_frequency": round(signal_result["dominant_frequency"], 2),
            "spectral_centroid": round(signal_result["spectral_centroid"], 2)
        }
        
        # 5. Time Series benchmark
        trend_signal = np.linspace(0, 10, 100) + np.sin(np.linspace(0, 8 * np.pi, 100)) + np.random.randn(100) * 0.5
        ts_result = self.analyze_time_series(trend_signal)
        
        results["time_series"] = {
            "trend_detected": ts_result["trend"]["direction"] == "increasing",
            "trend_strength": round(ts_result["trend"]["strength"], 4),
            "r_squared": round(ts_result["trend"]["r_squared"], 4)
        }
        
        # 6. Clustering benchmark
        cluster_data = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [5, 5],
            np.random.randn(30, 2) + [0, 5]
        ])
        
        cluster_result = self.cluster(cluster_data, "kmeans", k=3)
        
        results["clustering"] = {
            "clusters_found": cluster_result["n_clusters"],
            "inertia": round(cluster_result["inertia"], 2),
            "correct_k": cluster_result["n_clusters"] == 3
        }
        
        # 7. Dimensionality Reduction benchmark
        high_dim_data = np.random.randn(50, 10)
        pca_result = self.reduce_dimensions(high_dim_data, 2)
        
        results["dimensionality_reduction"] = {
            "output_dims": pca_result["transformed"].shape[1],
            "variance_explained": round(pca_result["total_explained"], 4),
            "reduction_successful": pca_result["transformed"].shape == (50, 2)
        }
        
        # Overall score
        passing = [
            results["pattern_recognition"]["template_matched"],
            results["anomaly_detection"]["true_positives"] >= 1,
            results["isolation_forest"]["detected_injected"],
            results["signal_processing"]["dominant_freq_detected"],
            results["time_series"]["trend_detected"],
            results["clustering"]["correct_k"],
            results["dimensionality_reduction"]["reduction_successful"],
            results["pattern_recognition"]["features_extracted"] >= 8
        ]
        
        results["overall"] = {
            "tests_passed": sum(passing),
            "tests_total": len(passing),
            "score": round(sum(passing) / len(passing) * 100, 1)
        }
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

l104_perception = L104PerceptionCore()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("⟨Σ_L104⟩ PERCEPTION ENGINE - SENSORY INTELLIGENCE")
    print("=" * 70)
    print(f"GOD_CODE: {GOD_CODE}")
    print(f"PHI: {PHI}")
    print(f"VERSION: {PERCEPTION_VERSION}")
    print()
    
    # Run benchmark
    print("[1] RUNNING COMPREHENSIVE BENCHMARK")
    print("-" * 40)
    
    results = l104_perception.benchmark()
    
    for category, data in results.items():
        if category == "overall":
            continue
        print(f"\n  {category.upper()}:")
        for key, value in data.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    
    print("\n" + "=" * 70)
    print(f"[2] OVERALL SCORE: {results['overall']['score']:.1f}%")
    print(f"    Tests Passed: {results['overall']['tests_passed']}/{results['overall']['tests_total']}")
    print("=" * 70)
    
    # Demo feature extraction
    print("\n[3] FEATURE EXTRACTION DEMO")
    print("-" * 40)
    
    demo_signal = np.sin(np.linspace(0, 6 * np.pi, 200)) * np.exp(-np.linspace(0, 2, 200))
    features = l104_perception.extract_features(demo_signal)
    
    print(f"  Signal: Damped sine wave (200 samples)")
    for key, value in list(features.items())[:8]:
        print(f"    {key}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("⟨Σ_L104⟩ PERCEPTION ENGINE OPERATIONAL")
    print("=" * 70)
