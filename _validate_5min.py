#!/usr/bin/env python3
"""Validate 5-minute WAV file."""
import wave, os
import numpy as np

f = 'god_code_G0000_5min.wav'
with wave.open(f, 'rb') as wf:
    ch = wf.getnchannels()
    sr = wf.getframerate()
    sw = wf.getsampwidth()
    nf = wf.getnframes()
    dur = nf / sr

    wf.setpos(nf // 2)
    raw = wf.readframes(min(96000, nf - nf // 2))
    raw_bytes = np.frombuffer(raw, dtype=np.uint8)
    triplets = raw_bytes.reshape(-1, 3)
    sign_byte = np.where(triplets[:, 2] >= 128, np.uint8(0xFF), np.uint8(0x00))
    four_bytes = np.column_stack([triplets, sign_byte])
    samples = four_bytes.view(np.int32).flatten().astype(np.float64) / 8388607.0
    L = samples[0::2]
    R = samples[1::2]

mb = os.path.getsize(f) / (1024 * 1024)
print(f'File:       {f}')
print(f'Size:       {mb:.1f} MB')
print(f'Duration:   {dur:.1f}s ({dur/60:.1f} min)')
print(f'Format:     {ch}ch / {sw*8}-bit / {sr}Hz')
print(f'Frames:     {nf:,}')
print(f'Mid-sample: L peak={np.max(np.abs(L)):.4f} RMS={np.sqrt(np.mean(L**2)):.4f}')
print(f'            R peak={np.max(np.abs(R)):.4f} RMS={np.sqrt(np.mean(R**2)):.4f}')
valid = nf > 28_000_000 and dur > 299
print(f'Status:     {"VALID" if valid else "INVALID"}')
