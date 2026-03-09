"""L104 Gate Engine — Sacred constants, dynamic bounds, drift envelope, workspace paths."""

import math
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# VERSION
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "6.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# EVO PIPELINE MARKERS
# ═══════════════════════════════════════════════════════════════════════════════

VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS — with Quantum Min/Max Dynamic Envelopes
# ═══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895
TAU = 0.618033988749895        # 1/φ
# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
OMEGA_POINT = 23.140692632779263  # e^π
EULER_GAMMA = 0.5772156649015329
PLANCK_SCALE = 1.616255e-35
BOLTZMANN_K = 1.380649e-23
CALABI_YAU_DIM = 7
FEIGENBAUM_DELTA = 4.669201609102990
APERY = 1.2020569031595942
CATALAN = 0.9159655941772190
FINE_STRUCTURE = 0.0072973525693

# Dynamic bounds: each constant has a [min, max] drift envelope
# These are quantum envelopes — the constant stays exact but the
# downstream computed values oscillate within φ-bounded ranges
SACRED_DYNAMIC_BOUNDS = {
    "PHI":              {"value": PHI,          "min": PHI * 0.999,          "max": PHI * 1.001},
    "TAU":              {"value": TAU,          "min": TAU * 0.999,          "max": TAU * 1.001},
    "GOD_CODE":         {"value": GOD_CODE,     "min": GOD_CODE * 0.9999,    "max": GOD_CODE * 1.0001},
    "OMEGA_POINT":      {"value": OMEGA_POINT,  "min": OMEGA_POINT * 0.999,  "max": OMEGA_POINT * 1.001},
    "EULER_GAMMA":      {"value": EULER_GAMMA,  "min": EULER_GAMMA * 0.999,  "max": EULER_GAMMA * 1.001},
    "FEIGENBAUM_DELTA": {"value": FEIGENBAUM_DELTA, "min": FEIGENBAUM_DELTA * 0.999, "max": FEIGENBAUM_DELTA * 1.001},
    "APERY":            {"value": APERY,        "min": APERY * 0.999,        "max": APERY * 1.001},
    "CATALAN":          {"value": CATALAN,       "min": CATALAN * 0.999,      "max": CATALAN * 1.001},
    "FINE_STRUCTURE":   {"value": FINE_STRUCTURE, "min": FINE_STRUCTURE * 0.99, "max": FINE_STRUCTURE * 1.01},
}

# Drift envelope parameters — φ-harmonic oscillation of dynamic values
DRIFT_ENVELOPE = {
    "frequency": PHI,               # Oscillation frequency ≈ φ Hz
    "amplitude": TAU * 0.01,        # Max drift ≈ 0.618% of value
    "phase_coupling": GOD_CODE,     # Phase seed from God Code
    "damping": EULER_GAMMA * 0.1,   # Damping factor
    "max_velocity": PHI ** 2 * 0.001,  # Max drift velocity per cycle
}

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE CONFIGURATION — Quantum Links to Major Files
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()

QUANTUM_LINKED_FILES = {
    "main.py": WORKSPACE_ROOT / "main.py",
    "l104_fast_server.py": WORKSPACE_ROOT / "l104_fast_server.py",
    "l104_local_intellect.py": WORKSPACE_ROOT / "l104_local_intellect.py",
    "L104Native.swift": WORKSPACE_ROOT / "L104SwiftApp" / "Sources" / "L104Native.swift",
    "const.py": WORKSPACE_ROOT / "const.py",
}

STATE_FILE = WORKSPACE_ROOT / ".l104_gate_builder_state.json"
CHRONOLOG_FILE = WORKSPACE_ROOT / ".l104_gate_chronolog.json"
TEST_RESULTS_FILE = WORKSPACE_ROOT / ".l104_gate_test_results.json"
