"""
L104 Gate Engine — Decomposed Logic Gate Builder Package
========================================================

Version: 6.0.0
Source:   Decomposed from l104_logic_gate_builder.py (4,475 lines → 17 modules)

Public API
----------
Constants:
    PHI, TAU, GOD_CODE, OMEGA_POINT, EULER_GAMMA, CALABI_YAU_DIM,
    FEIGENBAUM_DELTA, APERY, CATALAN, FINE_STRUCTURE, PLANCK_SCALE,
    BOLTZMANN_K, VERSION

Models:
    LogicGate, GateLink, ChronologEntry

Functions:
    sage_logic_gate, quantum_logic_gate, entangle_values,
    higher_dimensional_dissipation

Classes:
    HyperASILogicGateEnvironment  — Master orchestrator
    GateDynamismEngine            — Quantum min/max dynamism
    GateValueEvolver              — φ-harmonic value evolution
    OuroborosSageNirvanicEngine   — Ouroboros entropy fuel engine
    QuantumGateComputationEngine  — Quantum computation algorithms
    ConsciousnessO2GateEngine     — Consciousness O₂ modulation
    InterBuilderFeedbackBus       — Inter-builder message bus
    GateResearchEngine            — Advanced research engine
    StochasticGateResearchLab     — Stochastic gate R&D lab
    GateTestGenerator             — Automated test generation
    GateChronolizer               — Chronological gate tracking
    QuantumLinkManager            — Quantum link management
    PythonGateAnalyzer            — Python AST gate analysis
    SwiftGateAnalyzer             — Swift regex gate analysis
    JavaScriptGateAnalyzer        — JS regex gate analysis
    GateLinkAnalyzer              — Cross-file link analysis

CLI:
    python -m l104_gate_engine [command]
"""

__version__ = "6.0.0"

# ─── Constants ───────────────────────────────────────────────────
from .constants import (
    VERSION,
    PHI, TAU, GOD_CODE, OMEGA_POINT, EULER_GAMMA,
    CALABI_YAU_DIM, FEIGENBAUM_DELTA, APERY, CATALAN,
    FINE_STRUCTURE, PLANCK_SCALE, BOLTZMANN_K,
    SACRED_DYNAMIC_BOUNDS, DRIFT_ENVELOPE,
    WORKSPACE_ROOT, QUANTUM_LINKED_FILES,
    STATE_FILE, CHRONOLOG_FILE, TEST_RESULTS_FILE,
)

# ─── Models ──────────────────────────────────────────────────────
from .models import LogicGate, GateLink, ChronologEntry

# ─── Standalone Functions ────────────────────────────────────────
from .gate_functions import (
    sage_logic_gate,
    quantum_logic_gate,
    entangle_values,
    higher_dimensional_dissipation,
)

# ─── Analyzers ───────────────────────────────────────────────────
from .analyzers import (
    PythonGateAnalyzer,
    SwiftGateAnalyzer,
    JavaScriptGateAnalyzer,
    GateLinkAnalyzer,
)

# ─── Dynamism ────────────────────────────────────────────────────
from .dynamism import GateDynamismEngine, GateValueEvolver

# ─── Cross-System Engines ────────────────────────────────────────
from .nirvanic import OuroborosSageNirvanicEngine
from .quantum_computation import QuantumGateComputationEngine
from .consciousness import ConsciousnessO2GateEngine
from .feedback_bus import InterBuilderFeedbackBus

# ─── Research & Infrastructure ───────────────────────────────────
from .research import GateResearchEngine, StochasticGateResearchLab
from .test_generator import GateTestGenerator
from .chronolizer import GateChronolizer
from .link_manager import QuantumLinkManager

# ─── Orchestrator ────────────────────────────────────────────────
from .orchestrator import HyperASILogicGateEnvironment, main

__all__ = [
    # Constants
    "VERSION", "PHI", "TAU", "GOD_CODE", "OMEGA_POINT", "EULER_GAMMA",
    "CALABI_YAU_DIM", "FEIGENBAUM_DELTA", "APERY", "CATALAN",
    "FINE_STRUCTURE", "PLANCK_SCALE", "BOLTZMANN_K",
    "SACRED_DYNAMIC_BOUNDS", "DRIFT_ENVELOPE",
    "WORKSPACE_ROOT", "QUANTUM_LINKED_FILES",
    "STATE_FILE", "CHRONOLOG_FILE", "TEST_RESULTS_FILE",
    # Models
    "LogicGate", "GateLink", "ChronologEntry",
    # Functions
    "sage_logic_gate", "quantum_logic_gate",
    "entangle_values", "higher_dimensional_dissipation",
    # Analyzers
    "PythonGateAnalyzer", "SwiftGateAnalyzer",
    "JavaScriptGateAnalyzer", "GateLinkAnalyzer",
    # Dynamism
    "GateDynamismEngine", "GateValueEvolver",
    # Cross-System Engines
    "OuroborosSageNirvanicEngine",
    "QuantumGateComputationEngine",
    "ConsciousnessO2GateEngine",
    "InterBuilderFeedbackBus",
    # Research & Infrastructure
    "GateResearchEngine", "StochasticGateResearchLab",
    "GateTestGenerator", "GateChronolizer", "QuantumLinkManager",
    # Orchestrator
    "HyperASILogicGateEnvironment", "main",
]
