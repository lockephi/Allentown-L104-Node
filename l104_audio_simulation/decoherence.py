"""
Decoherence Audio — RealisticNoiseEngine-based quantum noise audio.

Generates GOD_CODE audio with realistic quantum decoherence noise from
an IBM Eagle noise profile (T1/T2, gate depolarization, readout, ZZ crosstalk).

Ported from _gen_decoherence_audio.py.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, Any, Optional

from .constants import GOD_CODE, PHI, DEFAULT_SAMPLE_RATE, DEFAULT_DURATION
from .envelopes import write_wav


def generate_decoherence_audio(
    output_file: str = "god_code_decoherence_5min.wav",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration: float = DEFAULT_DURATION,
    bit_depth: int = 24,
    n_quantum_samples: int = 1000,
    freq_wobble_hz: float = 0.2,
    amplitude_mod_range: float = 0.1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generate GOD_CODE audio with realistic quantum decoherence noise.

    Parameters
    ----------
    output_file : str
        Output WAV filename.
    sample_rate : int
        Audio sample rate (Hz).
    duration : float
        Audio duration (seconds).
    bit_depth : int
        Bit depth (16 or 24).
    n_quantum_samples : int
        Number of quantum Bell-state measurements for modulation envelope.
    freq_wobble_hz : float
        Maximum frequency wobble (±Hz) from quantum deviation.
    amplitude_mod_range : float
        Amplitude modulation range (±) around 0.95 baseline.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with keys: file, frequency, sample_rate, bit_depth, duration,
    quantum_samples, mean_deviation, std_deviation, elapsed_s
    """
    from l104_quantum_gate_engine import get_engine
    from l104_quantum_gate_engine.trajectory import RealisticNoiseEngine, NoiseProfile

    engine = get_engine()
    noise_engine = RealisticNoiseEngine(NoiseProfile.ibm_eagle())

    if verbose:
        print(f"  Noise profile: IBM Eagle — "
              f"T1={noise_engine.profile.t1_us}μs  "
              f"T2={noise_engine.profile.t2_us}μs  "
              f"1Q_err={noise_engine.profile.single_gate_error:.2e}")

    # Phase 1: Quantum noise sampling
    t0 = time.time()
    quantum_mods = []
    for i in range(n_quantum_samples):
        circ = engine.bell_pair()
        result = noise_engine.realistic_simulate(circ, shots=64, with_crosstalk=True)
        probs = result["probabilities"]
        p00 = probs.get("00", 0.5)
        quantum_mods.append(p00 - 0.5)
    quantum_mods = np.array(quantum_mods)

    if verbose:
        print(f"  Quantum sampling: {n_quantum_samples} Bell-state measurements "
              f"({time.time() - t0:.1f}s)")
        print(f"    mean_dev={np.mean(quantum_mods):.6f}  "
              f"std_dev={np.std(quantum_mods):.6f}")

    # Phase 2: Audio generation
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float64)

    # Interpolate quantum modulations across audio duration
    q_indices = np.linspace(0, len(quantum_mods) - 1, n_samples)
    q_envelope = np.interp(q_indices, np.arange(len(quantum_mods)), quantum_mods)

    # Amplitude modulation (±range around 0.95)
    amp_mod = 0.95 + q_envelope * amplitude_mod_range

    # Frequency modulation (±wobble Hz)
    freq_mod = GOD_CODE + q_envelope * freq_wobble_hz
    phase_acc = np.cumsum(2.0 * np.pi * freq_mod / sample_rate)
    modulated = np.sin(phase_acc)

    # Combine
    samples = modulated * amp_mod * 0.9

    if verbose:
        print(f"  Audio generated: {n_samples:,} samples")

    # Write WAV
    mb = write_wav(output_file, samples, sample_rate, bit_depth, channels=1)

    elapsed = time.time() - t0
    if verbose:
        print(f"  Written: {output_file} ({mb:.1f} MB) in {elapsed:.1f}s")

    return {
        "file": output_file,
        "frequency": GOD_CODE,
        "sample_rate": sample_rate,
        "bit_depth": bit_depth,
        "duration": duration,
        "quantum_samples": n_quantum_samples,
        "mean_deviation": float(np.mean(quantum_mods)),
        "std_deviation": float(np.std(quantum_mods)),
        "elapsed_s": elapsed,
        "size_mb": mb,
    }
