"""Diagnose v2.4.0 regression — isolate which change caused slowdown."""
import time
import numpy as np

print("="*60)
print("V2.4.0 REGRESSION DIAGNOSTIC")
print("="*60)

# Test 1: IonQ in-place performance
print("\n[1] IonQ Ba+ synthesis benchmark (in-place vs baseline)...")
from l104_audio_simulation.ionq_stabilizer import ionq_ba_synthesize_partials

n_partials = 13
n_samples = 1_800_000
sr = 180_000
freqs = np.array([527.518 * (i + 1) for i in range(n_partials)])
phases = np.zeros(n_partials)
weights = np.ones(n_partials) / n_partials

# Block size 65536
t0 = time.time()
sig1, _ = ionq_ba_synthesize_partials(freqs, phases, weights, n_samples, sr, samples_per_check=65536)
t_65k = time.time() - t0
print(f"  65536 blocks: {t_65k:.3f}s")

# Block size 16384
t0 = time.time()
sig2, _ = ionq_ba_synthesize_partials(freqs, phases, weights, n_samples, sr, samples_per_check=16384)
t_16k = time.time() - t0
print(f"  16384 blocks: {t_16k:.3f}s")

# Block size 4096
t0 = time.time()
sig3, _ = ionq_ba_synthesize_partials(freqs, phases, weights, n_samples, sr, samples_per_check=4096)
t_4k = time.time() - t0
print(f"  4096 blocks:  {t_4k:.3f}s")

# Max diff
diff = np.max(np.abs(sig1 - sig2))
print(f"  Max diff (65k vs 16k): {diff:.2e}")

# Test 2: vectorized_additive with ndarray vs closure
print("\n[2] vectorized_additive: ndarray vs closure phase_mod...")
from l104_audio_simulation.synthesis import vectorized_additive

t_arr = np.linspace(0, 10.0, n_samples, endpoint=False, dtype=np.float64)

# Pre-computed ndarray (1, n_samples)
mod_1d = np.sin(0.1 * t_arr)
mod_arr = mod_1d[np.newaxis, :]  # (1, n_samples)

# Closure
def _mod_closure(c0, c1):
    return np.broadcast_to(mod_1d[c0:c1][np.newaxis, :], (n_partials, c1 - c0))

t0 = time.time()
s1 = vectorized_additive(freqs, phases, weights, t_arr, n_samples, phase_mod=mod_arr)
t_arr_pm = time.time() - t0
print(f"  ndarray:  {t_arr_pm:.3f}s")

t0 = time.time()
s2 = vectorized_additive(freqs, phases, weights, t_arr, n_samples, phase_mod=_mod_closure)
t_cls_pm = time.time() - t0
print(f"  closure:  {t_cls_pm:.3f}s")

t0 = time.time()
s3 = vectorized_additive(freqs, phases, weights, t_arr, n_samples, phase_mod=None)
t_none = time.time() - t0
print(f"  no mod:   {t_none:.3f}s")

# Test 3: _aa_mod thread-local scratch overhead
print("\n[3] AA mod: thread-local scratch vs original closure...")
import threading
aa_sweep_rates = np.random.uniform(0.01, 0.1, n_partials)
aa_phases_arr = np.random.uniform(0, 2*np.pi, n_partials)
aa_mod_depths = np.random.uniform(0.05, 0.15, n_partials)

# Thread-local version
_aa_tls = threading.local()
_aa_n = n_partials
_aa_two_pi = 2.0 * np.pi

def _aa_mod_tls(c0, c1):
    chunk = c1 - c0
    if not hasattr(_aa_tls, 'buf') or _aa_tls.buf.shape != (_aa_n, chunk):
        _aa_tls.buf = np.empty((_aa_n, chunk), dtype=np.float64)
    buf = _aa_tls.buf
    np.outer(aa_sweep_rates, t_arr[c0:c1], out=buf)
    np.multiply(buf, _aa_two_pi, out=buf)
    np.add(buf, aa_phases_arr[:, np.newaxis], out=buf)
    np.sin(buf, out=buf)
    np.multiply(buf, aa_mod_depths[:, np.newaxis], out=buf)
    return buf

# Original version
def _aa_mod_orig(c0, c1):
    tc = t_arr[c0:c1]
    return aa_mod_depths[:, np.newaxis] * np.sin(
        _aa_two_pi * aa_sweep_rates[:, np.newaxis] * tc[np.newaxis, :]
        + aa_phases_arr[:, np.newaxis]
    )

t0 = time.time()
s_tls = vectorized_additive(freqs, phases, weights, t_arr, n_samples, phase_mod=_aa_mod_tls)
t_tls = time.time() - t0
print(f"  thread-local: {t_tls:.3f}s")

t0 = time.time()
s_orig = vectorized_additive(freqs, phases, weights, t_arr, n_samples, phase_mod=_aa_mod_orig)
t_orig_aa = time.time() - t0
print(f"  original:     {t_orig_aa:.3f}s")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"  IonQ 65K blocks:  {t_65k:.3f}s (vs 16K: {t_16k:.3f}s, 4K: {t_4k:.3f}s)")
print(f"  Gate ndarray:     {t_arr_pm:.3f}s (vs closure: {t_cls_pm:.3f}s)")
print(f"  AA thread-local:  {t_tls:.3f}s (vs original: {t_orig_aa:.3f}s)")
print(f"{'='*60}")
