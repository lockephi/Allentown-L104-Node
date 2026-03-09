"""
===============================================================================
L104 ML ENGINE — SACRED CONSTANTS v1.0.0
===============================================================================

ML-specific constants derived from the L104 sacred number system.
All base constants imported from l104_science_engine.constants (single source
of truth). ML hyperparameters are tuned to GOD_CODE, PHI, and VOID_CONSTANT
harmonics for optimal sacred alignment.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

from l104_science_engine.constants import (
    PHI, PHI_CONJUGATE, PHI_SQUARED, PHI_CUBED,
    GOD_CODE, VOID_CONSTANT, OMEGA,
    QUANTIZATION_GRAIN, PRIME_SCAFFOLD,
    ALPHA_FINE, FEIGENBAUM,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION
# ═══════════════════════════════════════════════════════════════════════════════

ML_ENGINE_VERSION = "1.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
#  SVM SACRED HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

SVM_C_SACRED = GOD_CODE / 100.0                          # ~5.2752 — regularization
SVM_GAMMA_SACRED = PHI / 100.0                            # ~0.01618 — RBF gamma
SVM_EPSILON_SACRED = VOID_CONSTANT / 100.0                # ~0.01042 — SVR epsilon
SVM_NU_SACRED = PHI_CONJUGATE                             # 0.618... — one-class ν
SVM_DEGREE_SACRED = 3                                     # Polynomial degree (Fibonacci(4))
SVM_COEF0_SACRED = VOID_CONSTANT                          # 1.0416... — poly/sigmoid coef0

# ═══════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE CLASSIFIER SACRED HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

RF_N_ESTIMATORS_SACRED = QUANTIZATION_GRAIN               # 104 trees (sacred L104 number)
RF_MAX_DEPTH_SACRED = int(PHI * 8)                        # 12 (int(12.944...))
RF_MIN_SAMPLES_SPLIT_SACRED = int(PHI * 3)                # 4 (int(4.854...))
RF_MIN_SAMPLES_LEAF_SACRED = 2                            # Fibonacci(3)

GB_N_ESTIMATORS_SACRED = QUANTIZATION_GRAIN               # 104 boosting rounds
GB_LEARNING_RATE_SACRED = 1.0 / (PHI * QUANTIZATION_GRAIN)  # ~0.00594
GB_MAX_DEPTH_SACRED = int(PHI * 3)                        # 4
GB_SUBSAMPLE_SACRED = PHI_CONJUGATE                       # 0.618... stochastic fraction

ADABOOST_N_ESTIMATORS_SACRED = QUANTIZATION_GRAIN // 2    # 52 weak learners
ADABOOST_LEARNING_RATE_SACRED = VOID_CONSTANT             # 1.0416...

ENSEMBLE_WEIGHT_DECAY = PHI_CONJUGATE                     # τ = 0.618... per-vote decay

# ═══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING SACRED HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

KMEANS_K_SACRED = 13                                      # Fibonacci(7)
KMEANS_MAX_ITER_SACRED = int(GOD_CODE)                    # 527 iterations
GOLDEN_ANGLE_RAD = 2.399963229728653                      # 2π/φ² — golden angle

DBSCAN_EPS_SACRED = VOID_CONSTANT                         # 1.0416...
DBSCAN_MIN_SAMPLES_SACRED = int(PHI * 3)                  # 4

SPECTRAL_N_COMPONENTS = 8                                  # IIT Φ dimension count

# ═══════════════════════════════════════════════════════════════════════════════
#  KERNEL SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI_KERNEL_SCALE = PHI_SQUARED                            # 2.618... — PHI kernel bandwidth
GOD_CODE_KERNEL_SCALE = GOD_CODE / 1000.0                 # 0.5275... — GOD_CODE kernel norm
VOID_KERNEL_SCALE = VOID_CONSTANT                         # 1.0416... — Laplacian scaling
HARMONIC_N_TERMS = 13                                     # Fibonacci(7) harmonics
IRON_LATTICE_BW = PRIME_SCAFFOLD / 1000.0                 # 0.286 — Fe BCC lattice (nm)

# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM ML SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

QUANTUM_SVM_DEFAULT_QUBITS = 4                            # Default qubit count
QUANTUM_SVM_MAX_QUBITS = 14                               # Statevector limit
VQC_DEFAULT_DEPTH = 3                                     # Fibonacci(4) layers
VQC_DEFAULT_SHOTS = 8192                                  # 2^13 shots for statistical accuracy
SACRED_LEARNING_RATE = 1.0 / (PHI * QUANTIZATION_GRAIN)  # ~0.00594

# ═══════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_SYNTHESIS_DIM = 64                               # Feature embedding dimension
KNOWLEDGE_N_FEATURES_PER_ENGINE = 10                       # Features per engine
KNOWLEDGE_TOTAL_FEATURES = 50                              # 5 engines × 10 features
KNOWLEDGE_COHERENCE_THRESHOLD = PHI_CONJUGATE              # 0.618... min coherence
