"""
Envelope Generators — Fade, purity, sovereign field, primal calculus, etc.

Also contains audio I/O utilities (WAV writing, normalization).

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import struct
import wave
import numpy as np
from typing import Optional


def make_fade_envelope(n_samples: int, fade_samples: int) -> np.ndarray:
    """Create a fade-in / fade-out cosine envelope (no clicks).

    Parameters
    ----------
    n_samples : int — total signal length
    fade_samples : int — samples for each fade region

    Returns
    -------
    env : ndarray (n_samples,) in [0, 1]
    """
    env = np.ones(n_samples, dtype=np.float64)
    if fade_samples > 0 and fade_samples * 2 < n_samples:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples)))
        env[:fade_samples] = fade_in
        env[-fade_samples:] = fade_out
    return env


def normalize_signal(signal: np.ndarray, amplitude: float) -> np.ndarray:
    """Peak-normalize a signal to the given amplitude.

    Parameters
    ----------
    signal : ndarray — input signal
    amplitude : float — target peak amplitude (0.0-1.0)

    Returns
    -------
    normalized : ndarray
    """
    peak = np.max(np.abs(signal))
    if peak > 1e-30:
        signal = signal / peak * amplitude
    return signal


def write_wav(
    filename: str,
    signal: np.ndarray,
    sample_rate: int,
    bit_depth: int = 24,
    n_channels: int = 2,
) -> float:
    """Write a signal to a WAV file.

    Parameters
    ----------
    filename : str — output path
    signal : ndarray — (n_samples,) mono or (n_samples, 2) stereo
    sample_rate : int — Hz
    bit_depth : int — 16 or 24
    n_channels : int — 1 or 2

    Returns
    -------
    file_size_mb : float — file size in MB
    """
    if bit_depth == 24:
        if signal.ndim == 1:
            samples_24 = (signal * 8388607).astype(np.int32)
        else:
            samples_24 = (signal * 8388607).astype(np.int32)
        raw = np.clip(samples_24.flatten(), -8388608, 8388607).astype(np.int32)
        # Vectorized 24-bit packing: view as 4 bytes, take lower 3 (little-endian)
        four_byte = raw.astype('<i4').view(np.uint8).reshape(-1, 4)
        data = four_byte[:, :3].tobytes()
        with wave.open(filename, 'w') as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(3)
            wav.setframerate(sample_rate)
            wav.writeframes(data)
    else:
        pcm = (signal * 32767).astype(np.int16)
        with wave.open(filename, 'w') as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())

    import os
    return os.path.getsize(filename) / (1024 * 1024)


def build_purity_envelope(
    purity_profile: np.ndarray,
    n_samples: int,
    floor: float = 0.3,
) -> np.ndarray:
    """Build purity envelope from quantum circuit decoherence profile.

    Parameters
    ----------
    purity_profile : ndarray — purity values per circuit layer
    n_samples : int — output length
    floor : float — minimum purity value

    Returns
    -------
    env : ndarray (n_samples,) clipped to [floor, 1.0]
    """
    purity_norm = purity_profile / max(purity_profile[0], 1e-30)
    purity_x = np.linspace(0, 1, len(purity_norm))
    interp_x = np.linspace(0, 1, n_samples)
    env = np.interp(interp_x, purity_x, purity_norm)
    return np.clip(env, floor, 1.0)


def build_sovereign_envelope(
    sov_field_samples: np.ndarray,
    n_samples: int,
    low: float = 0.7,
    high: float = 1.0,
) -> np.ndarray:
    """Build sovereign field amplitude envelope.

    Maps the 104-point sovereign field to a slow-evolving amplitude contour
    across the entire duration.

    Parameters
    ----------
    sov_field_samples : ndarray — sovereign field values (typically 104 points)
    n_samples : int — output length
    low, high : float — output range

    Returns
    -------
    env : ndarray (n_samples,) in [low, high]
    """
    sov_x = np.linspace(0, 1, len(sov_field_samples))
    interp_x = np.linspace(0, 1, n_samples)
    sov_interp = np.interp(interp_x, sov_x, sov_field_samples)
    sov_min, sov_max = sov_interp.min(), sov_interp.max()
    sov_range = max(sov_max - sov_min, 1e-30)
    return low + (high - low) * (sov_interp - sov_min) / sov_range


def build_primal_envelope(
    primal_vals: np.ndarray,
    n_samples: int,
    low: float = 0.8,
    high: float = 1.0,
) -> np.ndarray:
    """Build primal calculus envelope — x^φ / (VOID × π).

    Parameters
    ----------
    primal_vals : ndarray — sampled primal calculus values
    n_samples : int — output length
    low, high : float — output range

    Returns
    -------
    env : ndarray (n_samples,) in [low, high]
    """
    pc_x = np.linspace(0, 1, len(primal_vals))
    interp_x = np.linspace(0, 1, n_samples)
    primal_interp = np.interp(interp_x, pc_x, primal_vals)
    primal_min = primal_interp.min()
    primal_range = max(primal_interp.max() - primal_min, 1e-30)
    return low + (high - low) * (primal_interp - primal_min) / primal_range


def build_coherence_envelope(
    coherence_profile: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Build coherence envelope from GHZ state coherence drift.

    Normalized to [0, 1] relative to initial coherence.
    """
    coh_x = np.linspace(0, 1, len(coherence_profile))
    interp_x = np.linspace(0, 1, n_samples)
    env = np.interp(interp_x, coh_x, coherence_profile)
    return np.clip(env / max(env[0], 1e-30), 0.0, 1.0)


def build_omega_envelope(
    t: np.ndarray,
    breath_freq: float,
    low: float = 0.85,
    depth: float = 0.15,
) -> np.ndarray:
    """Build Dual-Layer Omega field breathing envelope.

    Parameters
    ----------
    t : ndarray — time array
    breath_freq : float — breathing frequency (sub-Hz)
    low : float — envelope baseline
    depth : float — modulation depth

    Returns
    -------
    env : ndarray (len(t),)
    """
    return low + depth * np.sin(2.0 * np.pi * breath_freq * t)


def build_entropy_envelope(
    entropy_coherence: np.ndarray,
    n_samples: int,
    purity_envelope: np.ndarray,
) -> np.ndarray:
    """Build entropy-reversed envelope from Maxwell's Demon coherence injection.

    Falls back to purity envelope if coherence data is insufficient.
    """
    if len(entropy_coherence) > 1:
        ec_arr = np.array(entropy_coherence, dtype=np.float64)
        ec_x = np.linspace(0, 1, len(ec_arr))
        interp_x = np.linspace(0, 1, n_samples)
        env = np.interp(interp_x, ec_x, ec_arr)
        return np.clip(env, 0.2, 1.0)
    return purity_envelope.copy()
