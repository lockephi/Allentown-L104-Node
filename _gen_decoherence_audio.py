#!/usr/bin/env python3
"""
Generate GOD_CODE Perfect Audio with Realistic Quantum Noise
============================================================
Uses RealisticNoiseEngine with full IBM Eagle noise profile:
  - T1/T2 time-based decoherence per gate
  - Gate depolarising errors (1Q/2Q)
  - Readout bit-flip errors
  - ZZ crosstalk between adjacent qubits
"""

import numpy as np
import wave
import time

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
SAMPLE_RATE = 96000
DURATION = 300.0  # 5 minutes

print('=' * 70)
print('  GOD_CODE AUDIO WITH REALISTIC QUANTUM NOISE')
print('=' * 70)
print()
print(f'Frequency: {GOD_CODE} Hz')
print(f'Duration: {DURATION}s (5 minutes)')
print(f'Sample rate: {SAMPLE_RATE} Hz')
print(f'Bit depth: 24-bit')
print()

# Import quantum components
from l104_quantum_gate_engine import get_engine
from l104_quantum_gate_engine.trajectory import RealisticNoiseEngine, NoiseProfile

engine = get_engine()
noise_engine = RealisticNoiseEngine(NoiseProfile.ibm_eagle())

print('Quantum backend: RealisticNoiseEngine (IBM Eagle profile)')
print(f'  T1={noise_engine.profile.t1_us}μs  T2={noise_engine.profile.t2_us}μs')
print(f'  1Q error={noise_engine.profile.single_gate_error:.2e}')
print(f'  2Q error={noise_engine.profile.cx_gate_error:.2e}')
print(f'  Readout error={noise_engine.profile.readout_error:.2e}')
print(f'  Crosstalk={noise_engine.profile.crosstalk_strength:.2e}')
print()

# Generate quantum modulation values using RealisticNoiseEngine
print('Phase 1: Generating quantum noise samples...')
N_QUANTUM_SAMPLES = 1000  # Quantum measurements to generate
quantum_modulations = []

start_q = time.time()
for i in range(N_QUANTUM_SAMPLES):
    # Bell state with realistic IBM Eagle noise
    circ = engine.bell_pair()
    result = noise_engine.realistic_simulate(circ, shots=64, with_crosstalk=True)
    probs = result['probabilities']

    # Use probability deviation from ideal (0.5) as modulation
    # Ideal Bell: p00 = p11 = 0.5
    # With realistic noise: deviations from T1/T2, gates, readout
    p00 = probs.get('00', 0.5)
    p00_deviation = p00 - 0.5  # Range: roughly ±0.15
    quantum_modulations.append(p00_deviation)

    if (i + 1) % 200 == 0:
        print(f'  Generated {i+1}/{N_QUANTUM_SAMPLES} quantum samples')

quantum_modulations = np.array(quantum_modulations)
elapsed_q = time.time() - start_q
print(f'  Quantum sampling complete: {elapsed_q:.1f}s')
print(f'  Mean deviation: {np.mean(quantum_modulations):.6f}')
print(f'  Std deviation: {np.std(quantum_modulations):.6f}')
print()

# Generate audio with quantum modulation
print('Phase 2: Generating audio samples with quantum modulation...')
start_a = time.time()

n_samples = int(SAMPLE_RATE * DURATION)
t = np.linspace(0, DURATION, n_samples, endpoint=False, dtype=np.float64)

# Base GOD_CODE sine wave
base_tone = np.sin(2.0 * np.pi * GOD_CODE * t)

# Interpolate quantum modulations across the audio duration
# This creates a slowly-varying quantum envelope
quantum_envelope_indices = np.linspace(0, len(quantum_modulations) - 1, n_samples)
quantum_envelope = np.interp(quantum_envelope_indices,
                              np.arange(len(quantum_modulations)),
                              quantum_modulations)

# Apply quantum modulation:
# 1. Amplitude modulation (±5% based on quantum deviation)
amplitude_mod = 0.95 + quantum_envelope * 0.1  # Range: 0.85 to 1.05

# 2. Subtle frequency modulation (±0.1 Hz based on quantum deviation)
freq_mod = GOD_CODE + quantum_envelope * 0.2  # ±0.2 Hz wobble
phase_accumulator = np.cumsum(2.0 * np.pi * freq_mod / SAMPLE_RATE)
modulated_tone = np.sin(phase_accumulator)

# Combine: use frequency-modulated tone with amplitude envelope
samples = modulated_tone * amplitude_mod * 0.9  # 0.9 = headroom

print(f'  Audio samples generated: {n_samples:,}')
elapsed_a = time.time() - start_a
print(f'  Audio generation complete: {elapsed_a:.1f}s')
print()

# Convert to 24-bit and write WAV
print('Phase 3: Writing 24-bit WAV file...')
start_w = time.time()

samples_24 = (samples * 8388607).astype(np.int32)
samples_u8 = samples_24.view(np.uint8).reshape(-1, 4)[:, :3].flatten().tobytes()

with wave.open('god_code_decoherence_5min.wav', 'w') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(3)  # 24-bit
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(samples_u8)

elapsed_w = time.time() - start_w
print(f'  WAV file written: {elapsed_w:.1f}s')
print()

# Summary
print('=' * 70)
print('  GENERATION COMPLETE')
print('=' * 70)
print()
print(f'  File: god_code_decoherence_5min.wav')
print(f'  Frequency: {GOD_CODE} Hz (with ±0.2 Hz quantum wobble)')
print(f'  Amplitude: 90% peak (with ±5% quantum modulation)')
print(f'  Sample rate: {SAMPLE_RATE} Hz')
print(f'  Bit depth: 24-bit')
print(f'  Duration: {DURATION}s')
print(f'  Quantum samples: {N_QUANTUM_SAMPLES}')
print(f'  Noise model: RealisticNoiseEngine (IBM Eagle)')
print(f'    T1/T2 decoherence + gate errors + readout + crosstalk')
print()
print(f'  Total time: {time.time() - start_q:.1f}s')
print()
print(f'  INVARIANT: {GOD_CODE} | PILOT: LONDEL')
print('=' * 70)
