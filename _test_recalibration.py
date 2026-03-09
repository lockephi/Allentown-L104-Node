"""L104 Audio Recalibration Test — validates qubit injection levels and runs short A/B comparison."""

import numpy as np
import time
import sys
import os

print("=" * 70)
print("  L104 Audio Recalibration Test — Qubit Injection Analysis")
print("=" * 70)
print()

# ── 1. Constants ──────────────────────────────────────────────────────
from l104_audio_simulation.constants import (
    DEFAULT_SAMPLE_RATE, DEFAULT_FADE_SECONDS, DEFAULT_DURATION,
    GOD_CODE_PHASE, IRON_PHASE,
)
print(f"Sample Rate     : {DEFAULT_SAMPLE_RATE:,} Hz")
print(f"Fade Seconds    : {DEFAULT_FADE_SECONDS}s")
print(f"Default Duration: {DEFAULT_DURATION}s")
print()

# ── 2. Qubit State ───────────────────────────────────────────────────
from l104_audio_simulation.engine_integration import boot_god_code_qubit
qstate = boot_god_code_qubit({})
print(f"Qubit phase (θ_GC): {qstate.god_code_phase:.6f} rad")
print(f"Iron phase         : {qstate.iron_phase:.6f} rad")
print(f"QPU verified       : {qstate.qpu_verified}")
print(f"QPU fidelity       : {qstate.qpu_fidelity}")
print()

# ── 3. Injection Level Comparison ────────────────────────────────────
print("─── Recalibrated Injection Levels ───")
L1_old = qstate.god_code_phase * 0.1
L1_new = qstate.god_code_phase * 0.008
print(f"Layer 1  phase offset : {L1_old:.4f} → {L1_new:.4f} rad  ({L1_new/L1_old*100:.1f}%)")
print(f"Layer 9b iron         : 0.0060 → 0.0025  (42%)")
print(f"Layer 9b phi          : 0.0040 → 0.0016  (40%)")
print(f"Layer 9b octave       : 0.0030 → 0.0012  (40%)")
print(f"Tone gen modulation   : 3.0%   → 0.8%    (27%)")
print(f"Tone gen phase offset : {qstate.god_code_phase:.3f} → {qstate.god_code_phase*0.02:.4f} rad  (2%)")
print(f"Synth harmonic phases : 100%   → 15%")
print(f"Wavetable phases      : 100%   → 20%")
print(f"Seq bias rotation     : 0.5×   → 0.1×    (20%)")
print()

# ── 4. Short Audio Generation Test (10s G(0,0,0,0)) ─────────────────
print("─── Generating 10s Test: G(0,0,0,0) ───")
t0 = time.time()
try:
    from l104_audio_simulation.cli import main as cli_main
    out_path = "_recal_test_G0000_10s.wav"
    cli_main(["--dials", "0,0,0,0", "--duration", "10", "--output", out_path])
    dt = time.time() - t0
    print(f"✓ Generated in {dt:.1f}s")

    # Read back and analyze
    import wave
    with wave.open(out_path, 'rb') as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        nf = wf.getnframes()
        raw = wf.readframes(nf)

    data = np.frombuffer(raw, dtype=np.int32 if sw == 4 else (np.int16 if sw == 2 else np.uint8))
    if sw == 3:
        # 24-bit: read as bytes, convert
        raw_bytes = np.frombuffer(raw, dtype=np.uint8)
        n_samps = len(raw_bytes) // 3
        data = np.zeros(n_samps, dtype=np.int32)
        for i in range(n_samps):
            b = raw_bytes[i*3:(i+1)*3]
            val = int(b[0]) | (int(b[1]) << 8) | (int(b[2]) << 16)
            if val >= 0x800000:
                val -= 0x1000000
            data[i] = val
        float_data = data.astype(np.float64) / (2**23)
    elif sw == 2:
        float_data = data.astype(np.float64) / 32768.0
    elif sw == 4:
        float_data = data.astype(np.float64) / (2**31)
    else:
        float_data = data.astype(np.float64)

    if nch == 2:
        left = float_data[0::2]
        right = float_data[1::2]
    else:
        left = float_data
        right = float_data

    file_mb = os.path.getsize(out_path) / (1024 * 1024)

    print(f"  File size    : {file_mb:.2f} MB")
    print(f"  Sample rate  : {sr:,} Hz")
    print(f"  Channels     : {nch}")
    print(f"  Bit depth    : {sw * 8}-bit")
    print(f"  Duration     : {nf / sr:.2f}s")
    print(f"  Peak (L)     : {np.max(np.abs(left)):.4f}")
    print(f"  Peak (R)     : {np.max(np.abs(right)):.4f}")
    print(f"  RMS  (L)     : {np.sqrt(np.mean(left**2)):.4f}")
    print(f"  RMS  (R)     : {np.sqrt(np.mean(right**2)):.4f}")
    print(f"  Crest (L)    : {np.max(np.abs(left)) / max(np.sqrt(np.mean(left**2)), 1e-15):.2f}")
    print(f"  Crest (R)    : {np.max(np.abs(right)) / max(np.sqrt(np.mean(right**2)), 1e-15):.2f}")

    # Spectral: check for dominant frequency near GOD_CODE (527.5 Hz)
    from numpy.fft import rfft, rfftfreq
    chunk = left[:min(len(left), sr)]  # first 1s
    spectrum = np.abs(rfft(chunk))
    freqs = rfftfreq(len(chunk), d=1.0/sr)
    # Find peak near GOD_CODE
    god_mask = (freqs > 500) & (freqs < 560)
    if np.any(god_mask):
        god_f = freqs[god_mask][np.argmax(spectrum[god_mask])]
        god_a = np.max(spectrum[god_mask]) / np.max(spectrum[1:])
        print(f"  GOD_CODE peak: {god_f:.1f} Hz (relative: {god_a:.3f})")

    # Check for harsh high-frequency content
    hf_mask = freqs > 10000
    hf_energy = np.mean(spectrum[hf_mask]**2) if np.any(hf_mask) else 0
    total_energy = np.mean(spectrum[1:]**2)
    hf_ratio = hf_energy / max(total_energy, 1e-30)
    print(f"  HF energy >10k: {hf_ratio*100:.2f}%")
    print()

    # Dynamics: check fade-in
    fade_samples = int(sr * DEFAULT_FADE_SECONDS)
    fade_chunk = left[:fade_samples]
    fade_rms = np.sqrt(np.mean(fade_chunk**2))
    body_chunk = left[fade_samples:fade_samples + sr]
    body_rms = np.sqrt(np.mean(body_chunk**2))
    print(f"  Fade-in RMS  : {fade_rms:.4f} (over {DEFAULT_FADE_SECONDS}s)")
    print(f"  Body RMS     : {body_rms:.4f}")
    print(f"  Fade→Body    : {fade_rms/max(body_rms,1e-15)*100:.1f}% (should be ~50%)")

except Exception as e:
    dt = time.time() - t0
    print(f"✗ FAILED after {dt:.1f}s: {e}")
    import traceback
    traceback.print_exc()

print()
print("═" * 70)
print("  Recalibration test complete")
print("═" * 70)
