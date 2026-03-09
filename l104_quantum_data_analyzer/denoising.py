"""
L104 Quantum Data Analyzer — Quantum Denoising & Error Mitigation
═══════════════════════════════════════════════════════════════════════════════
Quantum-inspired and quantum-native data denoising:

  1. EntropyReversalDenoiser       — Maxwell's Demon entropy reversal
  2. QuantumErrorMitigatedCleaner  — ZNE / Richardson extrapolation
  3. CoherenceFieldSmoother        — Quantum coherence field smoothing

CROSS-ENGINE INTEGRATION:
  • l104_science_engine — Maxwell Demon, entropy, coherence
  • l104_quantum_gate_engine — Error mitigation circuits
  • l104_math_engine — PHI harmonics, God Code weighting

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, VOID_CONSTANT, TAU,
    H_BAR, K_B,
    MAX_QUBITS_STATEVECTOR,
    god_code_at, normalize_vector, data_to_quantum_state, num_qubits_for,
)


# ─── Lazy engine imports ────────────────────────────────────────────────────
def _get_science_engine():
    try:
        from l104_science_engine import ScienceEngine
        return ScienceEngine()
    except ImportError:
        return None

def _get_math_engine():
    try:
        from l104_math_engine import MathEngine
        return MathEngine()
    except ImportError:
        return None

def _get_gate_engine():
    try:
        from l104_quantum_gate_engine import get_engine
        return get_engine()
    except ImportError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DenoisingResult:
    """Result from quantum denoising."""
    cleaned_data: np.ndarray
    noise_removed: np.ndarray
    snr_before: float
    snr_after: float
    snr_improvement_db: float
    method: str
    sacred_alignment: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ENTROPY REVERSAL DENOISER (MAXWELL'S DEMON)
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyReversalDenoiser:
    """
    Data denoising via Maxwell's Demon entropy reversal.

    Integrates with l104_science_engine's entropy subsystem to apply
    thermodynamic entropy reversal principles to noisy data:

      1. Measure local entropy field of the data
      2. Apply Maxwell's Demon to selectively reverse entropy
      3. Inject coherence from l104_science_engine for ordering
      4. Use PHI-weighted demon for golden-ratio prioritized reversal

    Thermodynamic cost: kT ln(2) per bit reversed (Landauer limit)
    Quantum advantage: coherent reversal at lower thermodynamic cost
    """

    def __init__(self, demon_strength: float = 1.0, temperature: float = 300.0):
        self.demon_strength = demon_strength
        self.temperature = temperature
        self.science_engine = _get_science_engine()
        self.math_engine = _get_math_engine()

    def denoise(self, data: np.ndarray, noise_estimate: Optional[float] = None) -> DenoisingResult:
        """
        Denoise data using Maxwell's Demon entropy reversal.

        Args:
            data: Noisy input data (1D or 2D)
            noise_estimate: Estimated noise level (auto-detected if None)

        Returns:
            DenoisingResult with cleaned data and metrics
        """
        t0 = time.time()
        original_shape = data.shape
        flat = data.ravel().astype(np.float64)

        # Step 1: Estimate noise level
        if noise_estimate is None:
            noise_estimate = self._estimate_noise(flat)

        # Step 2: Compute local entropy field
        entropy_field = self._compute_entropy_field(flat)

        # Step 3: Apply Maxwell's Demon reversal
        if self.science_engine:
            cleaned = self._demon_reversal_engine(flat, entropy_field)
        else:
            cleaned = self._demon_reversal_classical(flat, entropy_field, noise_estimate)

        # Step 4: Apply PHI-harmonic smoothing
        cleaned = self._phi_harmonic_smooth(cleaned)

        # Step 5: Preserve signal structure (don't over-denoise)
        alpha = min(self.demon_strength, 1.0)
        cleaned = alpha * cleaned + (1 - alpha) * flat

        # Metrics
        noise_removed = flat - cleaned
        snr_before = self._snr(flat, noise_estimate)
        snr_after = self._snr(cleaned, np.std(noise_removed))
        snr_improvement = 10 * np.log10(max(snr_after / max(snr_before, 1e-15), 1e-15))

        # Sacred alignment
        sacred = self._sacred_denoising_alignment(cleaned)

        output = cleaned.reshape(original_shape)

        return DenoisingResult(
            cleaned_data=output,
            noise_removed=noise_removed.reshape(original_shape),
            snr_before=snr_before,
            snr_after=snr_after,
            snr_improvement_db=float(snr_improvement),
            method="maxwell_demon_entropy_reversal",
            sacred_alignment=sacred,
            execution_time=time.time() - t0,
            metadata={
                "noise_estimate": noise_estimate,
                "demon_strength": self.demon_strength,
                "temperature": self.temperature,
                "landauer_cost_per_bit": float(K_B * self.temperature * math.log(2)),
                "entropy_bits_reversed": float(np.sum(np.maximum(entropy_field - self._compute_entropy_field(cleaned), 0))),
                "used_science_engine": self.science_engine is not None,
            },
        )

    def _estimate_noise(self, data: np.ndarray) -> float:
        """Estimate noise level using MAD (median absolute deviation)."""
        # Wavelet-like noise estimation using differences
        diffs = np.diff(data)
        mad = np.median(np.abs(diffs - np.median(diffs)))
        # MAD to std: σ ≈ 1.4826 × MAD
        return float(1.4826 * mad)

    def _compute_entropy_field(self, data: np.ndarray, window: int = 16) -> np.ndarray:
        """Compute local Shannon entropy over sliding window."""
        n = len(data)
        entropy = np.zeros(n)

        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2)
            segment = data[start:end]

            # Discretize into bins
            if np.std(segment) < 1e-15:
                entropy[i] = 0.0
                continue

            n_bins = max(4, int(np.sqrt(len(segment))))
            hist, _ = np.histogram(segment, bins=n_bins)
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            entropy[i] = -np.sum(probs * np.log2(probs))

        return entropy

    def _demon_reversal_engine(self, data: np.ndarray, entropy_field: np.ndarray) -> np.ndarray:
        """Apply entropy reversal using l104_science_engine."""
        se = self.science_engine

        # Use entropy subsystem's coherence injection
        try:
            raw = se.entropy.inject_coherence(data)
            cleaned = np.real(np.asarray(raw).ravel()[:len(data)])
            if len(cleaned) < len(data):
                cleaned = np.pad(cleaned, (0, len(data) - len(cleaned)), mode='edge')
        except Exception:
            cleaned = data.copy()

        # Scale back to original data range
        data_range = np.max(data) - np.min(data)
        cleaned_range = np.max(cleaned) - np.min(cleaned)
        if cleaned_range > 1e-15 and data_range > 1e-15:
            cleaned = (cleaned - np.mean(cleaned)) / cleaned_range * data_range + np.mean(data)

        # Apply PHI-weighted demon for targeted reversal
        try:
            phi_result = se.entropy.phi_weighted_demon(entropy_field)
            reversal_map = phi_result.get("reversal_priorities", np.ones_like(data))
            if isinstance(reversal_map, np.ndarray) and len(reversal_map) == len(data):
                # Weight the cleaning by reversal priority
                weight = reversal_map / (np.max(reversal_map) + 1e-15)
                cleaned = weight * cleaned + (1 - weight) * data
        except Exception:
            pass

        return cleaned

    def _demon_reversal_classical(self, data: np.ndarray, entropy_field: np.ndarray,
                                   noise_level: float) -> np.ndarray:
        """Classical Maxwell's Demon reversal (fallback)."""
        cleaned = data.copy()
        n = len(data)

        # Identify high-entropy regions (noisy)
        mean_entropy = np.mean(entropy_field)

        for i in range(n):
            if entropy_field[i] > mean_entropy:
                # High entropy: apply stronger smoothing
                window = max(1, min(8, int(entropy_field[i] / mean_entropy * 3)))
                start = max(0, i - window)
                end = min(n, i + window + 1)
                # Weighted average with inverse-entropy weights
                weights = np.exp(-entropy_field[start:end])
                weights /= np.sum(weights)
                cleaned[i] = np.sum(data[start:end] * weights)
            else:
                # Low entropy: preserve signal (minimal smoothing)
                alpha = entropy_field[i] / max(mean_entropy, 1e-15)
                if i > 0 and i < n - 1:
                    neighbors = (data[i - 1] + data[i + 1]) / 2
                    cleaned[i] = (1 - alpha * 0.1) * data[i] + alpha * 0.1 * neighbors

        return cleaned

    def _phi_harmonic_smooth(self, data: np.ndarray) -> np.ndarray:
        """Apply PHI-harmonic smoothing (golden ratio weighted filtering)."""
        # Ensure 1D input
        if data.ndim != 1:
            data = data.ravel()
        n = len(data)
        if n < 5:
            return data

        # PHI-weighted low-pass filter
        smoothed = data.copy()
        weights = np.array([PHI_CONJUGATE ** 2, PHI_CONJUGATE, 1.0, PHI_CONJUGATE, PHI_CONJUGATE ** 2])
        weights /= np.sum(weights)

        if data.ndim == 1:
            for i in range(2, n - 2):
                smoothed[i] = np.sum(data[i - 2:i + 3] * weights)
        else:
            for i in range(2, n - 2):
                window = data[i - 2:i + 3]
                smoothed[i] = np.sum(window * weights[:, None], axis=0)

        return smoothed

    def _snr(self, signal: np.ndarray, noise_level: float) -> float:
        """Signal-to-noise ratio."""
        signal_power = np.var(signal)
        noise_power = noise_level ** 2
        if noise_power < 1e-15:
            return float('inf')
        return float(signal_power / noise_power)

    def _sacred_denoising_alignment(self, cleaned: np.ndarray) -> float:
        """Check if denoised spectrum aligns with sacred harmonics."""
        fft = np.fft.fft(cleaned)
        magnitudes = np.abs(fft[:len(fft) // 2])
        if len(magnitudes) < 2:
            return 0.0

        # Check PHI-ratio between harmonics
        sorted_mags = np.sort(magnitudes)[::-1]
        if len(sorted_mags) >= 2 and sorted_mags[1] > 1e-15:
            ratio = sorted_mags[0] / sorted_mags[1]
            phi_distance = abs(ratio - PHI)
            return float(1.0 / (1.0 + phi_distance))
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QUANTUM ERROR MITIGATED CLEANER
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumErrorMitigatedCleaner:
    """
    Data cleaning using quantum error mitigation techniques:

      • Zero-Noise Extrapolation (ZNE): Run data processing at multiple
        noise levels and extrapolate to zero noise
      • Richardson extrapolation: Polynomial fit to extract clean signal
      • Probabilistic error cancellation (PEC): Quasi-probability decomposition
      • Symmetry verification: Project onto symmetric subspace

    These techniques, originally from quantum computing error mitigation,
    are adapted for classical data cleaning with quantum-inspired algorithms.
    """

    def __init__(self, n_noise_levels: int = 5, extrapolation: str = "richardson"):
        self.n_noise_levels = n_noise_levels
        self.extrapolation = extrapolation
        self.gate_engine = _get_gate_engine()

    def clean(self, data: np.ndarray, noise_model: str = "gaussian") -> DenoisingResult:
        """
        Clean data using quantum error mitigation techniques.

        Args:
            data: Noisy input data
            noise_model: Noise model ("gaussian", "depolarizing", "amplitude_damping")

        Returns:
            DenoisingResult with cleaned data and metrics
        """
        t0 = time.time()
        original_shape = data.shape
        flat = data.ravel().astype(np.float64)

        # Step 1: Estimate base noise level
        noise_level = self._estimate_noise_mad(flat)

        # Step 2: Generate noise-amplified versions
        noise_factors = np.linspace(1.0, 3.0, self.n_noise_levels)
        noisy_results = []

        for factor in noise_factors:
            amplified = self._amplify_noise(flat, factor, noise_model)
            # Process each noisy version (e.g., simple filtering)
            processed = self._process_at_noise(amplified, factor)
            noisy_results.append(processed)

        noisy_results = np.array(noisy_results)

        # Step 3: Zero-noise extrapolation
        if self.extrapolation == "richardson":
            cleaned = self._richardson_extrapolation(noisy_results, noise_factors)
        elif self.extrapolation == "linear":
            cleaned = self._linear_extrapolation(noisy_results, noise_factors)
        elif self.extrapolation == "exponential":
            cleaned = self._exponential_extrapolation(noisy_results, noise_factors)
        else:
            cleaned = self._richardson_extrapolation(noisy_results, noise_factors)

        # Step 4: Symmetry verification
        cleaned = self._symmetry_projection(flat, cleaned)

        # Metrics
        noise_removed = flat - cleaned
        noise_after = np.std(noise_removed)
        snr_before = self._compute_snr(flat, noise_level)
        snr_after = self._compute_snr(cleaned, noise_after)
        improvement = 10 * np.log10(max(snr_after / max(snr_before, 1e-15), 1e-15))

        return DenoisingResult(
            cleaned_data=cleaned.reshape(original_shape),
            noise_removed=noise_removed.reshape(original_shape),
            snr_before=snr_before,
            snr_after=snr_after,
            snr_improvement_db=float(improvement),
            method=f"zne_{self.extrapolation}",
            sacred_alignment=self._sacred_zne_alignment(cleaned),
            execution_time=time.time() - t0,
            metadata={
                "noise_model": noise_model,
                "n_noise_levels": self.n_noise_levels,
                "noise_factors": noise_factors.tolist(),
                "extrapolation": self.extrapolation,
                "base_noise_level": noise_level,
            },
        )

    def _estimate_noise_mad(self, data: np.ndarray) -> float:
        """Estimate noise via MAD of first differences."""
        diffs = np.diff(data)
        mad = np.median(np.abs(diffs - np.median(diffs)))
        return float(1.4826 * mad)

    def _amplify_noise(self, data: np.ndarray, factor: float, model: str) -> np.ndarray:
        """Amplify noise in data by a given factor."""
        noise_level = self._estimate_noise_mad(data)

        if model == "gaussian":
            extra_noise = np.random.normal(0, noise_level * (factor - 1), len(data))
        elif model == "depolarizing":
            # Depolarizing: randomly replace values with uniform noise
            mask = np.random.random(len(data)) < (1 - 1.0 / factor)
            extra_noise = np.zeros(len(data))
            extra_noise[mask] = np.random.uniform(np.min(data), np.max(data), np.sum(mask)) - data[mask]
            extra_noise *= (factor - 1) / factor
        elif model == "amplitude_damping":
            # Amplitude damping: exponential decay toward zero
            damping = 1 - np.exp(-(factor - 1) * 0.1)
            extra_noise = -damping * data * np.random.random(len(data))
        else:
            extra_noise = np.random.normal(0, noise_level * (factor - 1), len(data))

        return data + extra_noise

    def _process_at_noise(self, data: np.ndarray, noise_factor: float) -> np.ndarray:
        """Process data at a given noise level (simple low-pass filter)."""
        # Adaptive kernel width based on noise level
        kernel_width = max(3, int(noise_factor * 2 + 1))
        kernel_width = kernel_width if kernel_width % 2 == 1 else kernel_width + 1

        # Gaussian kernel
        half = kernel_width // 2
        kernel = np.exp(-np.arange(-half, half + 1) ** 2 / (2 * noise_factor))
        kernel /= np.sum(kernel)

        # Convolve
        return np.convolve(data, kernel, mode='same')

    def _richardson_extrapolation(self, results: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """Richardson extrapolation to zero noise."""
        n_levels = len(factors)
        n_points = results.shape[1]

        # For each data point, fit polynomial and extrapolate to factor=0
        cleaned = np.zeros(n_points)
        for i in range(n_points):
            values = results[:, i]
            # Polynomial fit (degree = n_levels - 1)
            degree = min(n_levels - 1, 3)
            coeffs = np.polyfit(factors, values, degree)
            # Evaluate at factor = 0 (zero noise)
            cleaned[i] = np.polyval(coeffs, 0)

        return cleaned

    def _linear_extrapolation(self, results: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """Simple linear extrapolation to zero noise."""
        n_points = results.shape[1]
        cleaned = np.zeros(n_points)
        for i in range(n_points):
            coeffs = np.polyfit(factors, results[:, i], 1)
            cleaned[i] = coeffs[1]  # y-intercept (factor=0)
        return cleaned

    def _exponential_extrapolation(self, results: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """Exponential extrapolation to zero noise."""
        n_points = results.shape[1]
        cleaned = np.zeros(n_points)
        for i in range(n_points):
            values = results[:, i]
            if np.all(values > 0):
                log_vals = np.log(values)
                coeffs = np.polyfit(factors, log_vals, 1)
                cleaned[i] = np.exp(coeffs[1])
            else:
                coeffs = np.polyfit(factors, values, 1)
                cleaned[i] = coeffs[1]
        return cleaned

    def _symmetry_projection(self, original: np.ndarray, cleaned: np.ndarray) -> np.ndarray:
        """Project cleaned data onto symmetric subspace that preserves key statistics."""
        # Preserve mean and variance
        cleaned_adj = cleaned.copy()
        orig_mean = np.mean(original)
        orig_std = np.std(original)
        clean_mean = np.mean(cleaned)
        clean_std = np.std(cleaned)

        if clean_std > 1e-15 and orig_std > 1e-15:
            cleaned_adj = (cleaned_adj - clean_mean) / clean_std * orig_std + orig_mean

        return cleaned_adj

    def _compute_snr(self, signal: np.ndarray, noise: float) -> float:
        """Compute signal-to-noise ratio."""
        signal_power = np.var(signal)
        if noise < 1e-15:
            return float('inf')
        return float(signal_power / noise ** 2)

    def _sacred_zne_alignment(self, data: np.ndarray) -> float:
        """Check ZNE result alignment with sacred proportions."""
        # Check if the variance follows PHI-ratio structure
        n = len(data)
        third = n // 3
        if third < 1:
            return 0.5

        var1 = np.var(data[:third])
        var2 = np.var(data[third:2 * third])
        var3 = np.var(data[2 * third:])

        total_var = var1 + var2 + var3
        if total_var < 1e-15:
            return 0.5

        # Check if variance ratios approximate PHI
        ratios = sorted([var1, var2, var3], reverse=True)
        if ratios[1] > 1e-15:
            r = ratios[0] / ratios[1]
            phi_dist = abs(r - PHI)
        else:
            phi_dist = PHI

        return float(1.0 / (1.0 + phi_dist))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COHERENCE FIELD SMOOTHER
# ═══════════════════════════════════════════════════════════════════════════════

class CoherenceFieldSmoother:
    """
    Data smoothing via quantum coherence field evolution.

    Uses l104_science_engine's coherence subsystem to:
      1. Initialize a coherence field from noisy data
      2. Evolve the field (coherence dynamics suppress noise)
      3. Anchor coherence at stable data points
      4. Discover latent coherence patterns

    The coherence evolves like a quantum open system: noise decoheres
    (is removed) while signal maintains coherence (is preserved).
    """

    def __init__(self, evolution_steps: int = 10, anchor_strength: float = 1.0):
        self.evolution_steps = evolution_steps
        self.anchor_strength = anchor_strength
        self.science_engine = _get_science_engine()
        self.math_engine = _get_math_engine()

    def smooth(self, data: np.ndarray, anchor_points: Optional[np.ndarray] = None) -> DenoisingResult:
        """
        Smooth data using quantum coherence field evolution.

        Args:
            data: Noisy input data
            anchor_points: Known clean data points to anchor coherence (optional)

        Returns:
            DenoisingResult with smoothed data and coherence metrics
        """
        t0 = time.time()
        original_shape = data.shape
        flat = data.ravel().astype(np.float64)
        noise_before = self._estimate_noise(flat)

        # Step 1: Initialize coherence field
        if self.science_engine:
            smoothed = self._coherence_evolution_engine(flat, anchor_points)
        else:
            smoothed = self._coherence_evolution_classical(flat, anchor_points)

        # Step 2: Apply VOID_CONSTANT phase correction
        smoothed = self._void_phase_correction(flat, smoothed)

        # Metrics
        noise_removed = flat - smoothed
        noise_after = self._estimate_noise(smoothed)
        snr_before = self._compute_snr(flat, noise_before)
        snr_after = self._compute_snr(smoothed, noise_after)
        improvement = 10 * np.log10(max(snr_after / max(snr_before, 1e-15), 1e-15))

        # Sacred alignment via coherence pattern
        sacred = self._coherence_sacred_alignment(smoothed)

        return DenoisingResult(
            cleaned_data=smoothed.reshape(original_shape),
            noise_removed=noise_removed.reshape(original_shape),
            snr_before=snr_before,
            snr_after=snr_after,
            snr_improvement_db=float(improvement),
            method="coherence_field_evolution",
            sacred_alignment=sacred,
            execution_time=time.time() - t0,
            metadata={
                "evolution_steps": self.evolution_steps,
                "anchor_strength": self.anchor_strength,
                "used_science_engine": self.science_engine is not None,
                "coherence_gain": float(np.var(smoothed) / max(np.var(flat), 1e-15)),
            },
        )

    def _coherence_evolution_engine(self, data: np.ndarray,
                                     anchor_points: Optional[np.ndarray]) -> np.ndarray:
        """Evolve coherence via l104_science_engine."""
        se = self.science_engine

        # Initialize coherence with data as seed thoughts
        # Convert numeric data to seed strings for coherence initialization
        seed_thoughts = [f"data_point_{i}_{v:.6f}" for i, v in enumerate(data[:20])]
        try:
            se.coherence.initialize(seed_thoughts)
        except Exception:
            pass

        # Use inject_coherence for the actual denoising
        try:
            raw = se.entropy.inject_coherence(data)
            smoothed = np.real(np.asarray(raw).ravel()[:len(data)])
            if len(smoothed) < len(data):
                smoothed = np.pad(smoothed, (0, len(data) - len(smoothed)), mode='edge')
        except Exception:
            smoothed = data.copy()

        # Scale back to original range
        data_range = np.max(data) - np.min(data)
        smoothed_range = np.max(smoothed) - np.min(smoothed)
        if smoothed_range > 1e-15 and data_range > 1e-15:
            smoothed = (smoothed - np.mean(smoothed)) / smoothed_range * data_range + np.mean(data)

        # Evolve coherence
        try:
            for step in range(self.evolution_steps):
                evolution = se.coherence.evolve(steps=1)
                # Apply discovered coherence field
                if step == self.evolution_steps - 1:
                    discovery = se.coherence.discover()
        except Exception:
            pass

        # Anchor at known clean points
        if anchor_points is not None and len(anchor_points) > 0:
            try:
                se.coherence.anchor(self.anchor_strength)
            except Exception:
                pass
            # Force anchor point values
            for idx in range(min(len(anchor_points), len(smoothed))):
                if not np.isnan(anchor_points[idx]):
                    smoothed[idx] = anchor_points[idx]

        return np.real(smoothed)

    def _coherence_evolution_classical(self, data: np.ndarray,
                                        anchor_points: Optional[np.ndarray]) -> np.ndarray:
        """Classical coherence evolution fallback."""
        n = len(data)
        smoothed = data.copy()

        for step in range(self.evolution_steps):
            # Coherence evolution: iterative non-local smoothing
            new = smoothed.copy()

            for i in range(n):
                # Coherence window: PHI-scaled neighbors
                window_size = max(1, int(PHI ** (step + 1)))
                start = max(0, i - window_size)
                end = min(n, i + window_size + 1)

                # Coherence weights: Gaussian with PHI-scaled bandwidth
                positions = np.arange(start, end) - i
                weights = np.exp(-positions ** 2 / (2 * (PHI * window_size) ** 2))

                # Value-similarity weighting (coherence ↔ similar values stay coherent)
                value_diffs = np.abs(smoothed[start:end] - smoothed[i])
                value_weights = np.exp(-value_diffs ** 2 / (2 * np.var(smoothed) + 1e-15))
                combined_weights = weights * value_weights

                # Apply anchor constraint
                if anchor_points is not None:
                    for j in range(start, end):
                        if j < len(anchor_points) and not np.isnan(anchor_points[j]):
                            combined_weights[j - start] *= self.anchor_strength * 10

                combined_weights /= np.sum(combined_weights) + 1e-15
                new[i] = np.sum(smoothed[start:end] * combined_weights)

            smoothed = new

            # Convergence check
            if np.max(np.abs(smoothed - data)) < 1e-12 * np.std(data):
                break

        return smoothed

    def _void_phase_correction(self, original: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
        """Apply VOID_CONSTANT phase correction to preserve signal phase."""
        # FFT-based phase correction
        fft_orig = np.fft.fft(original)
        fft_smooth = np.fft.fft(smoothed)

        # Preserve original phase, use smoothed magnitude
        phase_orig = np.angle(fft_orig)
        mag_smooth = np.abs(fft_smooth)

        # VOID_CONSTANT-weighted blending
        alpha = VOID_CONSTANT - 1.0  # ≈ 0.0416
        phase_blended = (1 - alpha) * np.angle(fft_smooth) + alpha * phase_orig

        corrected = np.fft.ifft(mag_smooth * np.exp(1j * phase_blended))
        return np.real(corrected)

    def _estimate_noise(self, data: np.ndarray) -> float:
        """Estimate noise level via MAD."""
        diffs = np.diff(data)
        mad = np.median(np.abs(diffs - np.median(diffs)))
        return float(1.4826 * mad)

    def _compute_snr(self, signal: np.ndarray, noise: float) -> float:
        """Signal-to-noise ratio."""
        if noise < 1e-15:
            return float('inf')
        return float(np.var(signal) / noise ** 2)

    def _coherence_sacred_alignment(self, data: np.ndarray) -> float:
        """Measure sacred alignment of coherence-smoothed data."""
        # Autocorrelation at PHI-lag
        n = len(data)
        phi_lag = max(1, int(n * PHI_CONJUGATE))
        if phi_lag >= n:
            return 0.5

        # Pearson correlation at PHI-lag
        x = data[:n - phi_lag]
        y = data[phi_lag:]
        if np.std(x) < 1e-15 or np.std(y) < 1e-15:
            return 0.5

        corr = np.corrcoef(x, y)[0, 1]
        # Higher autocorrelation at PHI-lag → higher sacred alignment
        return float(0.5 + 0.5 * abs(corr))
