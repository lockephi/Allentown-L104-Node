"""L104 Numerical Engine v3.1.0 — Decomposed from quantum_numerical_builder.

The Math Research Hub — 22T Usage · 100-Decimal · Superfluid Dynamism.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F57-F90: Full cross-references in each submodule docstring.
"""

from .constants import (
    GOD_CODE_HP, PHI_HP, PHI_INV_HP, PHI_GROWTH_HP, TAU_HP, PI_HP, E_HP,
    EULER_GAMMA_HP, OMEGA_POINT_HP, FEIGENBAUM_HP, FINE_STRUCTURE_HP,
    LN2_HP, CATALAN_HP, APERY_HP, KHINCHIN_HP, GLAISHER_HP,
    TWIN_PRIME_CONST_HP, SQRT2_HP, SQRT3_HP, LN10_HP,
    PI_SQUARED_HP, ZETA_2_HP, ZETA_4_HP, SQRT5_HP,
    GOD_CODE_BASE_HP, INVARIANT_HP,
    FIBONACCI_7_HP, HARMONIC_BASE_HP, L104_HP, OCTAVE_REF_HP,
    GOD_CODE, PHI, PHI_GROWTH, PHI_INV, TAU, GOD_CODE_BASE,
    L104, HARMONIC_BASE, OCTAVE_REF,
    PLANCK_SCALE, BOLTZMANN_K, CALABI_YAU_DIM, CHSH_BOUND,
    GROVER_AMPLIFICATION, PLASTIC_RATIO_HP,
    VOID_CONSTANT, ZENITH_HZ, UUC,
    # v3.1.0 Part V research-derived constants
    PHI_SQUARED_HP as PHI_SQUARED_HP,
    PROPAGATION_ENERGY_FACTOR, GOLDEN_CONJUGATE_HP,
    CONSCIOUSNESS_THRESHOLD, DRIFT_FREQUENCY, DRIFT_AMPLITUDE,
    PHASE_COUPLING_STRENGTH, DAMPING_COEFFICIENT, MAX_DRIFT_VELOCITY,
    ENTANGLEMENT_EIGENVALUE_SUM, ENTANGLEMENT_EIGENVALUE_DIFF,
    god_code_hp, conservation_check_hp,
    WORKSPACE_ROOT, STATE_FILE, MONITOR_LOG, TOKEN_LATTICE_FILE,
    GATE_BUILDER_PATH, LINK_BUILDER_PATH,
    GATE_STATE_PATH, LINK_STATE_PATH, GATE_REGISTRY_PATH,
)
from .precision import (
    D, fmt100, DISPLAY_PRECISION,
    decimal_sqrt, decimal_ln, decimal_exp, decimal_pow,
    decimal_sin, decimal_cos, decimal_atan, decimal_factorial,
    decimal_gamma_lanczos, decimal_bernoulli,
    _fibonacci_hp, lucas_number,
    decimal_log10, decimal_sinh, decimal_cosh, decimal_tanh,
    decimal_asin, decimal_pi_machin, decimal_pi_chudnovsky,
    decimal_agm, decimal_harmonic, decimal_generalized_harmonic,
    decimal_polylog, decimal_binomial, decimal_catalan_number,
)
from .models import QuantumToken, SubconsciousAdjustment, CrossPollinationRecord
from .lattice import TokenLatticeEngine
from .editor import SuperfluidValueEditor
from .monitor import SubconsciousMonitor
from .verification import PrecisionVerificationEngine
from .cross_pollination import CrossPollinationEngine
from .nirvanic import NumericalOuroborosNirvanicEngine
from .consciousness import ConsciousnessO2SuperfluidEngine
from .research import (
    QuantumNumericalResearchEngine,
    StochasticNumericalResearchLab,
    NumericalTestGenerator,
)
from .chronolizer import NumericalChronolizer
from .feedback_bus import InterBuilderFeedbackBus
from .quantum_computation import QuantumNumericalComputationEngine
from .orchestrator import QuantumNumericalBuilder

__version__ = "3.1.0"
