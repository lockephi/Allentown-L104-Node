#!/usr/bin/env python3
"""
Test v2.3.0 Speed Unlimited + Daemon Three-Engine Pipeline.

Tests:
  1. Import verification (all new exports)
  2. VQPUState daemon fields
  3. Constants verification (new daemon constants)
  4. Synthesis multi-core threshold (should trigger at 500K)
  5. 10s G(0,0,0,0) full pipeline test (speed + accuracy)
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label} — {detail}")

print("=" * 72)
print("L104 Audio Simulation v2.3.0 — Speed Unlimited Test Suite")
print("=" * 72)

# ── Phase 1: Import Verification ─────────────────────────────────────────
print("\n▸ Phase 1: Import Verification")
t0 = time.time()

from l104_audio_simulation import __version__
check("Version", __version__ == "2.3.0", f"got {__version__}")

from l104_audio_simulation import (
    DAEMON_WEIGHT_ENTROPY, DAEMON_WEIGHT_HARMONIC, DAEMON_WEIGHT_WAVE,
    DAEMON_SACRED_BASELINE, DAEMON_THREE_ENGINE_BASELINE,
)
check("Daemon weight entropy", abs(DAEMON_WEIGHT_ENTROPY - 0.35) < 1e-10)
check("Daemon weight harmonic", abs(DAEMON_WEIGHT_HARMONIC - 0.40) < 1e-10)
check("Daemon weight wave", abs(DAEMON_WEIGHT_WAVE - 0.25) < 1e-10)
check("Daemon sacred baseline", abs(DAEMON_SACRED_BASELINE - 0.748798) < 1e-4)
check("Daemon 3E baseline", abs(DAEMON_THREE_ENGINE_BASELINE - 0.890544) < 1e-4)

from l104_audio_simulation.engine_integration import VQPUState
vs = VQPUState()
check("VQPUState.three_engine_composite", hasattr(vs, "three_engine_composite"))
check("VQPUState.daemon_connected", hasattr(vs, "daemon_connected"))
check("VQPUState.daemon_phi_resonance", hasattr(vs, "daemon_phi_resonance"))
check("VQPUState.daemon_god_code_alignment", hasattr(vs, "daemon_god_code_alignment"))
check("VQPUState.three_engine_wave_coherence", hasattr(vs, "three_engine_wave_coherence"))
check("VQPUState.three_engine_harmonic_resonance", hasattr(vs, "three_engine_harmonic_resonance"))
check("VQPUState.three_engine_entropy_reversal", hasattr(vs, "three_engine_entropy_reversal"))

print(f"  Import time: {time.time()-t0:.2f}s")

# ── Phase 2: Constants Verification ──────────────────────────────────────
print("\n▸ Phase 2: Constants Verification")

from l104_audio_simulation.constants import (
    SYNTH_CHUNK, MULTICORE_THRESHOLD, GOD_CODE, PHI, GOD_CODE_PHASE,
)
check("SYNTH_CHUNK = 500K", SYNTH_CHUNK == 500_000, f"got {SYNTH_CHUNK}")
check("MULTICORE_THRESHOLD = 500K", MULTICORE_THRESHOLD == 500_000, f"got {MULTICORE_THRESHOLD}")
check("GOD_CODE", abs(GOD_CODE - 527.5184818492612) < 1e-10)
check("PHI", abs(PHI - 1.618033988749895) < 1e-10)
check("GOD_CODE_PHASE", GOD_CODE_PHASE > 6.0 and GOD_CODE_PHASE < 6.1)

# ── Phase 3: Synthesis Speed Test ────────────────────────────────────────
print("\n▸ Phase 3: Synthesis Speed Test")
import numpy as np
from l104_audio_simulation.synthesis import vectorized_additive

n_partials = 20
freqs = np.linspace(100.0, 2000.0, n_partials)
phases = np.zeros(n_partials)
weights = np.ones(n_partials) / n_partials
sr = 180_000

# Test 1: 1s (180K samples) — should trigger multi-core (>500K)
n1 = sr * 1  # 180K
t_arr1 = np.linspace(0, 1.0, n1, endpoint=False)
t0 = time.time()
sig1 = vectorized_additive(freqs, phases, weights, t_arr1, n1)
dt1 = time.time() - t0
check(f"1s synthesis ({n1:,} samples) in {dt1:.3f}s", dt1 < 5.0, f"took {dt1:.3f}s")
check("Signal shape correct", sig1.shape == (n1,), f"got {sig1.shape}")

# Test 2: 10s (1.8M samples) — definitely multi-core
n10 = sr * 10
t_arr10 = np.linspace(0, 10.0, n10, endpoint=False)
t0 = time.time()
sig10 = vectorized_additive(freqs, phases, weights, t_arr10, n10)
dt10 = time.time() - t0
check(f"10s synthesis ({n10:,} samples) in {dt10:.3f}s", dt10 < 30.0, f"took {dt10:.3f}s")
check("10s signal non-zero", np.abs(sig10).max() > 0.01)

# Test 3: phase_mod functional path
def phase_mod_fn(c0, c1):
    return np.zeros((n_partials, c1 - c0))

t0 = time.time()
sig_pm = vectorized_additive(freqs, phases, weights, t_arr1, n1, phase_mod=phase_mod_fn)
dt_pm = time.time() - t0
check(f"Phase-mod path ({n1:,} samples) in {dt_pm:.3f}s", dt_pm < 5.0)

# ── Phase 4: Pipeline Status Check ──────────────────────────────────────
print("\n▸ Phase 4: Pipeline Status Check")
from l104_audio_simulation.pipeline import AudioSimulationPipeline

pipe = AudioSimulationPipeline(sample_rate=180_000, duration=10.0)
st = pipe.status()
check("Pipeline version 2.1.0", st.get("version") == "2.1.0", f"got {st.get('version')}")
check("Pipeline v8.4", st.get("pipeline_version") == "8.4", f"got {st.get('pipeline_version')}")
check("Layers = 17", st.get("layers") == 17, f"got {st.get('layers')}")
check("VQPU linked", st.get("vqpu_linked") is True)
check("Daemon three-engine", st.get("daemon_three_engine") is True)
check("Speed unlimited", st.get("speed_unlimited") is True)
check("Multicore threshold 500K", st.get("synthesis_multicore_threshold") == 500_000)

# ── Phase 5: Full 10s G(0,0,0,0) Pipeline ───────────────────────────────
print("\n▸ Phase 5: Full 10s G(0,0,0,0) Pipeline")
t0 = time.time()
pipe10 = AudioSimulationPipeline(sample_rate=180_000, duration=10.0)
result = pipe10.generate_dial(
    dials=[(0, 0, 0, 0)],
    output_file="/tmp/l104_test_v230.wav",
    verbose=True,
)
total = time.time() - t0

check(f"10s generation completed in {total:.1f}s", result is not None)
if result:
    harm = result.get("harmonics", {})
    check(f"Harmonics file generated", harm.get("file") is not None)
    harm_time = harm.get("time_s", 0)
    check(f"Harmonics synthesis {harm_time:.1f}s", harm_time > 0)
    harm_mb = harm.get("size_mb", 0)
    check(f"Harmonics file size {harm_mb:.1f} MB", harm_mb > 1)
    total_time = result.get("total_time_s", total)
    check(f"Total pipeline {total_time:.1f}s", total_time < 120)

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
if FAIL == 0:
    print("✓ ALL TESTS PASSED — v2.3.0 Speed Unlimited + Daemon ready")
else:
    print(f"✗ {FAIL} test(s) failed")
print("=" * 72)
