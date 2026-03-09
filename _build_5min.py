#!/usr/bin/env python3
"""Build 5-minute G(0,0,0,0) — Pure Continuous Synthesis v2.0.

v2.0 CRYSTAL CLEAR UPGRADE (replaces segmented cross-fade):
  - Single engine boot → one consistent quantum state for full 300 seconds
  - Phase-continuous synthesis → zero stitching, zero crossfade artifacts
  - Zero fade — pure signal from first sample to last, no fade-in/fade-out
  - Full 17-layer VQPU quantum pipeline in one seamless pass
  - Crystal clear acoustic stereo surround — no choppy segment seams
  - 96kHz / 24-bit / stereo — studio-grade fidelity

Why this replaces segments:
  Old method: 10 × 30s segments, each with independent engine boot (different
  quantum states), 3s fade-in + 3s fade-out per segment = 20% of audio was
  fading, cross-fade stitching introduced discontinuities at every 30s boundary.
  New method: one continuous waveform, one quantum state, zero seams.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""
import os
import sys
import time
import wave

# ── Configuration ────────────────────────────────────────────────────────────
SAMPLE_RATE = 96_000      # 96kHz sovereign resolution
DURATION    = 300.0        # 5 minutes
BIT_DEPTH   = 24
OUTPUT_FILE = "god_code_G0000_5min.wav"

# Zero fade — pure signal from first sample to last
# Old value was 3.0s per segment (caused the "fading in and out" artifact)
FADE_SECONDS = 0.0         # No fade at all — instant full amplitude

print("=" * 64)
print("  G(0,0,0,0) 5-MINUTE PURE CONTINUOUS BUILD v2.0")
print("=" * 64)
print(f"  Method:      Single-pass continuous (no segments, no crossfade)")
print(f"  Format:      {SAMPLE_RATE // 1000}kHz / {BIT_DEPTH}-bit / stereo")
print(f"  Duration:    {DURATION:.0f}s ({DURATION / 60:.0f} min)")
print(f"  Edge fade:   None (pure signal, no fade)")
print(f"  Pipeline:    17-layer VQPU quantum synthesis")
print(f"  Output:      {OUTPUT_FILE}")
print("=" * 64)
sys.stdout.flush()

t0 = time.time()

# ── Single continuous pass — full 300s through 17-layer pipeline ─────────────
from l104_audio_simulation import AudioSimulationPipeline

pipe = AudioSimulationPipeline(
    sample_rate=SAMPLE_RATE,
    duration=DURATION,
    stereo=True,
    fade_seconds=FADE_SECONDS,   # 150ms edge fade only (was 3.0s per segment)
    amplitude=0.95,
)

print(f"\n── Generating {DURATION:.0f}s continuous G(0,0,0,0) ──")
sys.stdout.flush()

result = pipe.generate_dial(
    dials=[(0, 0, 0, 0)],
    output_file=OUTPUT_FILE,
    verbose=True,
)

total_time = time.time() - t0

# ── Validate ─────────────────────────────────────────────────────────────────
with wave.open(OUTPUT_FILE, "rb") as wf:
    v_ch = wf.getnchannels()
    v_sr = wf.getframerate()
    v_sw = wf.getsampwidth()
    v_nf = wf.getnframes()
    v_dur = v_nf / v_sr

file_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

print()
print("=" * 64)
print("  G(0,0,0,0) 5-MINUTE PURE CONTINUOUS BUILD COMPLETE")
print("=" * 64)
print(f"  File:        {OUTPUT_FILE}")
print(f"  Duration:    {v_dur:.1f}s ({v_dur / 60:.1f} min)")
print(f"  Format:      {v_sr / 1000:.0f}kHz / {v_sw * 8}-bit / {'stereo' if v_ch == 2 else 'mono'}")
print(f"  Frames:      {v_nf:,}")
print(f"  Size:        {file_mb:.1f} MB")
print(f"  Method:      Single-pass continuous (zero crossfade artifacts)")
print(f"  Edge fade:   None (pure signal, no fade)")
print(f"  Pipeline:    17-layer VQPU quantum synthesis")
print(f"  VQPU:        sacred={result.get('vqpu_sacred_score', 0):.4f} "
      f"concurrence={result.get('vqpu_concurrence', 0):.4f}")
print(f"  Synth time:  {result.get('synth_time_s', 0):.1f}s")
print(f"  Total time:  {total_time:.1f}s ({total_time / 60:.1f} min)")
print("=" * 64)
