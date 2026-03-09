#!/usr/bin/env python3
"""Quick test: VQPU-linked 16-layer pipeline — 10s G(0,0,0,0) at 180kHz."""

import time
import numpy as np

t0 = time.time()
from l104_audio_simulation.pipeline import AudioSimulationPipeline

suite = AudioSimulationPipeline(
    sample_rate=180000,
    duration=10.0,
    bit_depth=24,
    stereo=True,
    fade_seconds=0.5,
    amplitude=0.95,
)

print("Pipeline status:", suite.status())
print()

result = suite.generate_dial(
    dials=[(0, 0, 0, 0)],
    output_file="test_vqpu_link_10s.wav",
    verbose=True,
)

total = time.time() - t0
print()
print("=== VQPU-LINKED TEST RESULTS ===")
for k, v in result.items():
    print(f"  {k}: {v}")
print(f"  wall_time: {total:.2f}s")

# Quick audio analysis
import struct, wave
with wave.open("test_vqpu_link_10s.wav", "rb") as wf:
    sr = wf.getframerate()
    ch = wf.getnchannels()
    frames = wf.getnframes()
    sw = wf.getsampwidth()
    dur_s = frames / sr
    print(f"\n=== WAV ANALYSIS ===")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Channels: {ch}")
    print(f"  Bit depth: {sw * 8}")
    print(f"  Duration: {dur_s:.2f}s")
    print(f"  Frames: {frames:,}")

# Read a chunk for spectral check
data = np.fromfile("test_vqpu_link_10s.wav", dtype=np.int8)
file_mb = len(data) / (1024 * 1024)
print(f"  File size: {file_mb:.2f} MB")
print("\nDone.")
