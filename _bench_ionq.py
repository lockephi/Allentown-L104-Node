"""Compare IonQ in-place vs original approach."""
import numpy as np, time

n_p = 13; n_s = 1_800_000; sr = 180_000
freqs = np.array([527.518 * (i+1) for i in range(n_p)])
phases = np.zeros(n_p)
weights = np.ones(n_p) / n_p

# V2.4.0 (in-place + einsum)
from l104_audio_simulation.ionq_stabilizer import ionq_ba_synthesize_partials
t0 = time.time()
for _ in range(3):
    ionq_ba_synthesize_partials(freqs, phases, weights, n_s, sr, samples_per_check=16384)
t_new = (time.time() - t0) / 3
print(f"IonQ v2.4.0 (in-place+einsum): {t_new:.3f}s")

# Original approach (allocating temp arrays)
two_pi = 2.0 * np.pi
delta_phases = two_pi * freqs / sr
block_indices = np.arange(16384, dtype=np.float64)
w_col = weights[:, np.newaxis]
n_blocks = n_s // 16384

t0 = time.time()
for _ in range(3):
    signal = np.zeros(n_s, dtype=np.float64)
    phase_accums = phases.copy()
    for b in range(n_blocks):
        start = b * 16384
        end = start + 16384
        bp = phase_accums[:, np.newaxis] + delta_phases[:, np.newaxis] * block_indices[np.newaxis, :]
        signal[start:end] += np.sum(w_col * np.sin(bp), axis=0)
        phase_accums += delta_phases * 16384
t_old = (time.time() - t0) / 3
print(f"IonQ original (temp arrays):   {t_old:.3f}s")
print(f"Speedup: {t_old/t_new:.2f}x")
