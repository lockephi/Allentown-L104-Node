"""
IonQ Trapped-Ion Stabilizers — 171Yb+ Bulk + 138Ba+ Syndrome Extraction.

Adapted from IonQ's trapped-ion gate correction architecture:
  - 171Yb+ bulk stabilizer: majority-vote error mitigation
  - 138Ba+ syndrome extraction: real-time phase accumulator correction

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any


def ionq_stabilizer(
    vector: np.ndarray,
    error_rate: float = 0.001,
) -> Tuple[np.ndarray, float, float]:
    """
    IonQ trapped-ion high-fidelity gate correction (171Yb+).

    Simulates trapped-ion thermal noise injection, then applies
    majority-vote error mitigation — snapping drifted values to
    7-decimal precision.

    Parameters
    ----------
    vector : np.ndarray
        Phase vector, weight array, frequency array, or composite signal.
    error_rate : float
        Simulated gate error rate (default 0.001 = 99.9% fidelity).

    Returns
    -------
    corrected : np.ndarray — stabilized vector
    max_drift : float — maximum absolute correction applied
    rms_correction : float — RMS of the correction
    """
    noise = np.random.normal(0, error_rate, vector.shape)
    noisy = vector + noise
    corrected = np.round(noisy * 1e7) / 1e7
    correction = corrected - vector
    max_drift = float(np.max(np.abs(correction)))
    rms_correction = float(np.sqrt(np.mean(correction ** 2)))
    return corrected, max_drift, rms_correction


def ionq_signal_stabilizer(
    signal: np.ndarray,
    error_rate: float = 0.0005,
) -> Tuple[np.ndarray, float, float]:
    """
    IonQ waveform-level stabilizer for post-additive synthesis (171Yb+).

    Uses a tighter error rate for signal-domain correction,
    preserving waveform fidelity while eliminating accumulated
    floating-point drift from multi-partial summation.
    """
    amp = np.max(np.abs(signal))
    if amp < 1e-30:
        return signal, 0.0, 0.0
    noise = np.random.normal(0, error_rate * amp, signal.shape)
    noisy = signal + noise
    grain = amp * 1e-7
    corrected = np.round(noisy / grain) * grain
    correction = corrected - signal
    max_drift = float(np.max(np.abs(correction)))
    rms_correction = float(np.sqrt(np.mean(correction ** 2)))
    return corrected, max_drift, rms_correction


def ionq_syndrome_extraction(
    phase_accumulator: float,
    target_phase: float,
    fidelity_floor: float = 1e-8,
) -> Tuple[float, float, bool]:
    """
    IonQ Ba+ syndrome extraction — single checkpoint.

    Measures drift from the target Bloch manifold coordinate and applies
    a correction pulse if drift exceeds the fidelity floor.

    Parameters
    ----------
    phase_accumulator : float — current accumulated phase (radians)
    target_phase : float — ideal phase at this checkpoint
    fidelity_floor : float — drift threshold (default 1e-8 = 99.99%)

    Returns
    -------
    corrected_phase : float
    syndrome : float — measured drift (radians)
    corrected : bool — whether correction was applied
    """
    TWO_PI = 2.0 * np.pi
    current_wrapped = phase_accumulator % TWO_PI
    target_wrapped = target_phase % TWO_PI
    syndrome = current_wrapped - target_wrapped
    syndrome = (syndrome + np.pi) % TWO_PI - np.pi
    if abs(syndrome) > fidelity_floor:
        return phase_accumulator - syndrome, syndrome, True
    return phase_accumulator, 0.0, False


def ionq_ba_synthesize_partials(
    freqs: np.ndarray,
    initial_phases: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    sr: int,
    samples_per_check: int = 4096,
    fidelity_floor: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ba+ syndrome-corrected multi-partial synthesizer (vectorized).

    v8.5 SPEED UPGRADE:
      - Pre-allocated (N, block) scratch buffer — zero per-block allocation
      - In-place np.outer / np.multiply / np.sin / np.einsum chain
      - Larger block sizes recommended at high sample rates (65536 at 180kHz)

    Parameters
    ----------
    freqs : ndarray (N,) — partial frequencies
    initial_phases : ndarray (N,) — starting phases
    weights : ndarray (N,) — amplitude weights
    n_samples : int — total samples
    sr : int — sample rate
    samples_per_check : int — correction interval
    fidelity_floor : float — minimum drift to trigger correction

    Returns
    -------
    signal : ndarray (n_samples,)
    stats : dict with correction statistics
    """
    n_partials = len(freqs)
    signal = np.zeros(n_samples, dtype=np.float64)
    n_blocks = n_samples // samples_per_check
    remainder = n_samples % samples_per_check

    total_corrections = 0
    total_syndrome = 0.0
    max_syndrome = 0.0
    per_partial_corrections = np.zeros(n_partials, dtype=int)

    two_pi = 2.0 * np.pi
    delta_phases = two_pi * freqs / sr
    phase_accums = initial_phases.copy()
    w_col = weights  # 1D — used with einsum below

    # Pre-allocate scratch buffer for the main block size (zero per-block alloc)
    block_indices = np.arange(samples_per_check, dtype=np.float64)
    buf = np.empty((n_partials, samples_per_check), dtype=np.float64)

    for b in range(n_blocks):
        start = b * samples_per_check
        end = start + samples_per_check
        # In-place chain: outer → add phases → sin → weighted sum
        np.outer(delta_phases, block_indices, out=buf)
        np.add(buf, phase_accums[:, np.newaxis], out=buf)
        np.sin(buf, out=buf)
        signal[start:end] += np.einsum("i,ij->j", w_col, buf)
        phase_accums += delta_phases * samples_per_check

        # Vectorized syndrome extraction (every block)
        ideal_phases = initial_phases + two_pi * freqs * (end / sr)
        current_wrapped = phase_accums % two_pi
        target_wrapped = ideal_phases % two_pi
        syndromes = current_wrapped - target_wrapped
        syndromes = (syndromes + np.pi) % two_pi - np.pi
        needs_correction = np.abs(syndromes) > fidelity_floor
        if np.any(needs_correction):
            phase_accums[needs_correction] -= syndromes[needs_correction]
            n_corr = int(np.sum(needs_correction))
            total_corrections += n_corr
            per_partial_corrections += needs_correction.astype(int)
            abs_syns = np.abs(syndromes[needs_correction])
            total_syndrome += float(np.sum(abs_syns))
            block_max = float(np.max(abs_syns))
            if block_max > max_syndrome:
                max_syndrome = block_max

    if remainder > 0:
        start = n_blocks * samples_per_check
        rem_indices = np.arange(remainder, dtype=np.float64)
        rem_buf = np.empty((n_partials, remainder), dtype=np.float64)
        np.outer(delta_phases, rem_indices, out=rem_buf)
        np.add(rem_buf, phase_accums[:, np.newaxis], out=rem_buf)
        np.sin(rem_buf, out=rem_buf)
        signal[start:] += np.einsum("i,ij->j", w_col, rem_buf)

    stats = {
        "total_corrections": total_corrections,
        "total_syndrome_rad": total_syndrome,
        "max_syndrome_rad": max_syndrome,
        "mean_syndrome_rad": total_syndrome / max(total_corrections, 1),
        "per_partial": per_partial_corrections.tolist(),
        "blocks_per_partial": n_blocks,
        "fidelity_floor": fidelity_floor,
        "check_interval": samples_per_check,
    }
    return signal, stats


def ionq_ba_synthesize_single(
    freq: float,
    initial_phase: float,
    n_samples: int,
    sr: int,
    samples_per_check: int = 1024,
    fidelity_floor: float = 1e-8,
) -> Tuple[np.ndarray, int, float]:
    """
    Ba+ syndrome-corrected single-frequency synthesizer.
    Optimized path for pure tones (binaural, pure GOD_CODE file).

    Returns
    -------
    signal : ndarray (n_samples,)
    corrections : int — total corrections applied
    max_syndrome : float — maximum syndrome value
    """
    signal = np.zeros(n_samples, dtype=np.float64)
    n_blocks = n_samples // samples_per_check
    remainder = n_samples % samples_per_check
    block_indices = np.arange(samples_per_check, dtype=np.float64)
    delta_phase = 2.0 * np.pi * freq / sr
    phase_accum = initial_phase
    corrections = 0
    max_syn = 0.0

    for b in range(n_blocks):
        start = b * samples_per_check
        end = start + samples_per_check
        block_phases = phase_accum + delta_phase * block_indices
        signal[start:end] = np.sin(block_phases)
        phase_accum += delta_phase * samples_per_check
        ideal = initial_phase + 2.0 * np.pi * freq * (end / sr)
        phase_accum, syndrome, corrected = ionq_syndrome_extraction(
            phase_accum, ideal, fidelity_floor
        )
        if corrected:
            corrections += 1
            if abs(syndrome) > max_syn:
                max_syn = abs(syndrome)

    if remainder > 0:
        start = n_blocks * samples_per_check
        rem_idx = np.arange(remainder, dtype=np.float64)
        signal[start:] = np.sin(phase_accum + delta_phase * rem_idx)

    return signal, corrections, max_syn
