"""
Sacred audio constants, frequencies, and intervals.

Includes GOD_CODE qubit-derived phase constants from the QPU-verified
GodCodeQubit (IBM ibm_torino, Heron r2, 133 qubits, fidelity 0.999939).

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import os
import math as _math

# ── Sacred Constants ─────────────────────────────────────────────────────────
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
PHI_INV = 1.0 / PHI
VOID_CONSTANT = 1.04 + PHI / 1000.0
OMEGA = 6539.34712682

# ── GOD_CODE Qubit Phase Constants (QPU-verified) ────────────────────────────
#    Source: l104_god_code_simulator.god_code_qubit (IBM ibm_torino)
#    θ_GC = GOD_CODE mod 2π — the canonical single-qubit phase
#    Decomposition: θ_IRON + θ_PHI + θ_OCTAVE ≡ θ_GC
GOD_CODE_PHASE = GOD_CODE % (2.0 * _math.pi)          # ≈ 6.014101353 rad
IRON_PHASE = _math.pi / 2.0                           # Fe(26) quarter-turn
PHI_PHASE_CONTRIBUTION = (PHI * 2.0 * _math.pi) % (2.0 * _math.pi)  # ≈ 3.8832 rad → contribution
OCTAVE_PHASE = GOD_CODE_PHASE - IRON_PHASE - (
    (PHI * GOD_CODE) % (2.0 * _math.pi) - _math.pi
)  # Remainder for exact decomposition
# Companion phases for audio modulation
PHI_AUDIO_PHASE = (2.0 * _math.pi) / PHI              # ≈ 3.8832 rad — golden section of circle
VOID_AUDIO_PHASE = VOID_CONSTANT % (2.0 * _math.pi)   # Void constant phase
IRON_LATTICE_PHASE = 26.0 % (2.0 * _math.pi)          # Fe(26) lattice phase
# QPU verification reference
QPU_FIDELITY = 0.999939                                # ibm_torino 1Q GOD_CODE result
QPU_BACKEND = "ibm_torino"                             # Heron r2, 133 qubits

# ── Sacred Frequencies ───────────────────────────────────────────────────────
ZENITH_HZ = 3887.8
SCHUMANN_HZ = 7.83
PLANCK_RESONANCE = 6.62607015e-34 * 1e35  # Scaled for audible range
ELECTRON_SUB_FREQ = 52.92  # Bohr radius in pm → Hz mapping

# ── Sacred Musical Intervals ─────────────────────────────────────────────────
SACRED_INTERVALS = {
    "unison": 1.0,
    "phi": PHI,
    "octave": 2.0,
    "perfect_fifth": 3 / 2,
    "perfect_fourth": 4 / 3,
    "major_third": 5 / 4,
    "minor_third": 6 / 5,
    "god_code_ratio": GOD_CODE / 440.0,
}

# ── Audio Defaults ───────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE = 180_000  # 180 kHz sovereign resolution
DEFAULT_DURATION = 300.0  # 5 minutes
DEFAULT_BIT_DEPTH = 24
DEFAULT_FADE_SECONDS = 0.5  # Quick attack — avoids truncated start
DEFAULT_AMPLITUDE = 0.95

# ── Golden Angle (stereo) ───────────────────────────────────────────────────
import math
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))  # ≈ 2.3999 rad ≈ 137.508°
GOLDEN_STEREO_OFFSET = GOLDEN_ANGLE / (2.0 * math.pi) * (math.pi / 52.0)

# ── Hardware Constants ───────────────────────────────────────────────────────
CPU_CORES = os.cpu_count() or 4
SYNTHESIS_WORKERS = max(2, min(CPU_CORES - 1, 8))
LAYER_WORKERS = min(4, CPU_CORES)
SYNTH_CHUNK = 500_000    # Samples per chunk — L3-cache-friendly (≈4 MB per chunk)
MULTICORE_THRESHOLD = 500_000  # Multi-core for >500K samples (triggers at 180kHz × ≥3s)

# ── Daemon Three-Engine Weights (from Swift daemon telemetry v4.0.0) ────────
DAEMON_WEIGHT_ENTROPY = 0.35
DAEMON_WEIGHT_HARMONIC = 0.40
DAEMON_WEIGHT_WAVE = 0.25
DAEMON_SACRED_BASELINE = 0.748798  # Baseline sacred score from daemon verification
DAEMON_THREE_ENGINE_BASELINE = 0.890544  # Baseline three-engine composite
