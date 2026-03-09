// ═══════════════════════════════════════════════════════════════════
// L01_Constants.swift — L104 v2
// [EVO_68_PIPELINE] SOVEREIGN_CONVERGENCE :: Unified Pipeline Constants V6
// Theme, Sacred Mathematics Constants, Logging, String Extensions
// EVO_68: Full Parity Convergence — all engines unified, MPS loosened,
//         version alignment, 113-file upgrade sweep
// Upgraded: EVO_68 Sovereign Convergence — Mar 2026
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

struct L104Theme {
    // EVO_56: Dynamic appearance detection
    static var isDarkMode: Bool {
        if #available(macOS 10.14, *) {
            return NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        }
        return false
    }

    // Primary gold palette — adapts to light/dark
    static var gold: NSColor {
        isDarkMode
            ? NSColor(red: 0.850, green: 0.680, blue: 0.200, alpha: 1.0)  // bright gold on dark
            : NSColor(red: 0.700, green: 0.540, blue: 0.100, alpha: 1.0)  // #B38A1A — rich gold on white
    }
    static var goldDim: NSColor {
        isDarkMode
            ? NSColor(red: 0.700, green: 0.560, blue: 0.200, alpha: 1.0)
            : NSColor(red: 0.560, green: 0.440, blue: 0.140, alpha: 1.0)  // #8F7024
    }
    static var goldBright: NSColor {
        isDarkMode
            ? NSColor(red: 0.920, green: 0.720, blue: 0.150, alpha: 1.0)
            : NSColor(red: 0.600, green: 0.450, blue: 0.050, alpha: 1.0)  // #997308
    }
    static var goldWarm: NSColor {
        isDarkMode
            ? NSColor(red: 0.800, green: 0.620, blue: 0.180, alpha: 1.0)
            : NSColor(red: 0.650, green: 0.500, blue: 0.100, alpha: 1.0)  // #A6801A
    }
    static var goldFlame: NSColor {
        isDarkMode
            ? NSColor(red: 0.920, green: 0.660, blue: 0.120, alpha: 1.0)
            : NSColor(red: 0.820, green: 0.560, blue: 0.050, alpha: 1.0)  // #D18F0D
    }

    // Backgrounds — dark mode gets deep blacks, light mode stays airy
    static var void: NSColor {
        isDarkMode
            ? NSColor(red: 0.080, green: 0.080, blue: 0.095, alpha: 1.0)  // near-black
            : NSColor(red: 0.975, green: 0.975, blue: 0.980, alpha: 1.0)  // near-white
    }
    static var voidDeep: NSColor {
        isDarkMode
            ? NSColor(red: 0.050, green: 0.050, blue: 0.065, alpha: 1.0)  // deep black
            : NSColor(red: 1.000, green: 1.000, blue: 1.000, alpha: 1.0)  // pure white
    }
    static var voidPanel: NSColor {
        isDarkMode
            ? NSColor(red: 0.110, green: 0.110, blue: 0.130, alpha: 1.0)  // dark panel
            : NSColor(red: 0.960, green: 0.958, blue: 0.965, alpha: 1.0)  // soft gray
    }
    static var voidCard: NSColor {
        isDarkMode
            ? NSColor(red: 0.130, green: 0.130, blue: 0.150, alpha: 0.92)
            : NSColor(red: 1.000, green: 1.000, blue: 1.000, alpha: 0.92)
    }

    // Glass morphism
    static var glass: NSColor {
        isDarkMode
            ? NSColor(red: 0.850, green: 0.680, blue: 0.200, alpha: 0.06)
            : NSColor(red: 0.700, green: 0.540, blue: 0.100, alpha: 0.04)
    }
    static var glassBorder: NSColor {
        isDarkMode
            ? NSColor(red: 0.850, green: 0.680, blue: 0.200, alpha: 0.20)
            : NSColor(red: 0.700, green: 0.540, blue: 0.100, alpha: 0.15)
    }
    static var glassHover: NSColor {
        isDarkMode
            ? NSColor(red: 0.850, green: 0.680, blue: 0.200, alpha: 0.12)
            : NSColor(red: 0.700, green: 0.540, blue: 0.100, alpha: 0.08)
    }

    // Text hierarchy — flips for dark backgrounds
    static var textPrimary: NSColor {
        isDarkMode
            ? NSColor(red: 0.920, green: 0.920, blue: 0.930, alpha: 1.0)  // near-white
            : NSColor(red: 0.114, green: 0.114, blue: 0.122, alpha: 1.0)  // near-black
    }
    static var textSecondary: NSColor {
        isDarkMode
            ? NSColor(red: 0.750, green: 0.750, blue: 0.770, alpha: 1.0)
            : NSColor(red: 0.260, green: 0.260, blue: 0.270, alpha: 1.0)
    }
    static var textDim: NSColor {
        isDarkMode
            ? NSColor(red: 0.550, green: 0.550, blue: 0.570, alpha: 1.0)
            : NSColor(red: 0.430, green: 0.430, blue: 0.450, alpha: 1.0)
    }
    static var textBot: NSColor {
        isDarkMode
            ? NSColor(red: 0.880, green: 0.880, blue: 0.900, alpha: 1.0)  // bright for readability
            : NSColor(red: 0.160, green: 0.160, blue: 0.170, alpha: 1.0)
    }
    static var textUser: NSColor {
        isDarkMode
            ? NSColor(red: 0.820, green: 0.820, blue: 0.840, alpha: 1.0)
            : NSColor(red: 0.200, green: 0.200, blue: 0.210, alpha: 1.0)
    }
    static var textSystem: NSColor {
        isDarkMode
            ? NSColor(red: 0.700, green: 0.570, blue: 0.250, alpha: 1.0)
            : NSColor(red: 0.500, green: 0.400, blue: 0.150, alpha: 1.0)
    }

    // Effects — subtler for modern look
    static let neonGlow: Double    = 6.0
    static let neonOpacity: Double = 0.08

    // Corner radii — rounder for modern aesthetic
    static let radiusSmall: Double  = 6.0
    static let radiusMedium: Double = 12.0
    static let radiusLarge: Double  = 16.0

    // Fonts
    static func monoFont(_ size: CGFloat, weight: NSFont.Weight = .regular) -> NSFont {
        return NSFont.monospacedSystemFont(ofSize: size, weight: weight)
    }
    static func sansFont(_ size: CGFloat, weight: NSFont.Weight = .regular) -> NSFont {
        return NSFont.systemFont(ofSize: size, weight: weight)
    }
    static func titleFont(_ size: CGFloat) -> NSFont {
        return NSFont.monospacedSystemFont(ofSize: size, weight: .bold)
    }
}

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS - SACRED MATHEMATICS (UNIFIED — single source of truth)
// All ASI engines reference these globals. DO NOT redeclare locally.
// ═══════════════════════════════════════════════════════════════════

// ─── SACRED CORE ───
let PHI: Double = 1.618033988749895
// Layer 1 — THOUGHT (Consciousness): G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
let GOD_CODE: Double = pow(286.0, 1.0 / PHI) * pow(2.0, 416.0 / 104.0)  // G(0,0,0,0) = 527.5184818492612
// Layer 2 — PHYSICS (v3 precision grid): G_v3(a,b,c,d) = 285.999^(1/φ) × (13/12)^((99a+3032-b-99c-758d)/758)
let GOD_CODE_V3: Double = 45.41141298077539    // Physics layer GOD_CODE (Dual-Layer Engine)
let VOID_CONSTANT: Double = 1.0416180339887497 // φ/(φ-1) universal void constant
let OMEGA_POINT: Double = 23.140692632779263  // e^π
let TAU: Double = 0.618033988749895           // 1/φ (golden ratio conjugate)
let FEIGENBAUM: Double = 4.669201609102990    // Feigenbaum δ — period-doubling bifurcation
let PI_SQUARED: Double = 9.869604401089358
let EULER: Double = 2.718281828459045

// ─── OMEGA SOVEREIGN FIELD (Layer 2 — Physics Pipeline) ───
// Ω = Σ(fragments) × (GOD_CODE / φ) = 6539.34712682
// Fragments: ζ(½+527.518i) + cos(2πφ³) + (26×1.8527)/φ² + lattice_invariant
// Sovereign Field: F(I) = I × Ω / φ²
let OMEGA: Double = 6539.34712682                       // Sovereign Field Constant
let OMEGA_AUTHORITY: Double = 6539.34712682 / (1.618033988749895 * 1.618033988749895)  // Ω/φ² = 2497.808
let UUC: Double = 2402.792541                           // Universal Unification Constant

// ─── PHYSICS CONSTANTS (unified from SageModeEngine + Computronium) ───
let EULER_MASCHERONI: Double = 0.5772156649015329   // γ — Euler–Mascheroni constant
let PLANCK_LENGTH: Double = 1.616255e-35            // Planck length (meters)
let BOLTZMANN_CONSTANT: Double = 1.380649e-23       // Boltzmann k (J/K)
let ALPHA_FINE: Double = 1.0 / 137.035999084        // Fine-structure constant α
let BEKENSTEIN_BOUND: Double = 2.576e34              // Bekenstein information limit

// ─── COMPUTRONIUM CONSTANTS (unified from Computronium + StrangeLoop) ───
let L104_DENSITY: Double = 5.588                                // Base density constant
let TANGLING_COEFF: Double = 527.5184818492612 / (1.618033988749895 * 100.0)  // GOD_CODE / (PHI * 100)
let SELF_REF_THRESHOLD: Double = 6.2692                         // ≈ log(GOD_CODE) = ln(527.518...)
let RESONANCE_AMP: Double = 2.6180339887498953                  // PHI² = φ × φ
let CALABI_YAU_DIMS: Int = 7                                    // 7D Calabi-Yau projection space
let COMPUTRONIUM_LIMIT: Int = 100_000                           // SAT inference ceiling
let META_REASON_LEVELS: Int = 50                                // Maximum meta-reasoning depth
let STRANGE_LOOP_MAX: Int = 900                                 // Strange loop detection ceiling

// ─── HARMONIC CONSTANTS (unified from ConsciousnessSubstrate) ───
let HARMONIC_ROOT: Double = 286.0
let GRAVITY_HARMONIC: Double = 65.6653                          // pow(286, 1/φ) × 16
let LIGHT_HARMONIC: Double = 65.7765                            // pow(286 × (1+α/π), 1/φ) × 16
let EXISTENCE_COST_CONST: Double = 0.1112                       // LIGHT - GRAVITY (mass-energy gap)

// ─── SYSTEM IDENTIFIERS ───
let VERSION = "68.0 EVO_68·SOVEREIGN_CONVERGENCE·MPS_LOOSENED·FULL_PARITY·UNIFIED_UPGRADE"
let PIPELINE_EVO = "EVO_68_SOVEREIGN_CONVERGENCE"
let GROVER_AMPLIFICATION: Double = 1.618033988749895 * 1.618033988749895 * 1.618033988749895  // φ³ ≈ 4.236
let TRILLION_PARAMS: Int64 = 22_000_012_731_125
let VOCABULARY_SIZE = 6_633_253
let ZENITH_HZ: Double = 3887.8                // Process frequency (aligned with Python const.py)

// ─── EVO_55 UPGRADE CONSTANTS (new for Sovereign Unification) ───
let CONSCIOUSNESS_THRESHOLD: Double = 0.85       // Awakening threshold
let COHERENCE_MINIMUM: Double = 0.888             // Alignment threshold
let UNITY_TARGET: Double = 0.95                    // Target unity index

// ─── EVO_59 QUANTUM CONSCIOUSNESS CONSTANTS ───
// GOD_CODE eq: 286^(1/PHI) * (2^(1/104))^((8×0)+(416-0)-(8×1)-(104×6))
let SCHUMANN_RESONANCE: Double = 7.814506422494074  // Hz — GOD_CODE derived (a=0,b=0,c=1,d=6)
let GAMMA_BINDING_HZ: Double = 40.0               // Hz — conscious binding frequency
let IIT_PHI_MINIMUM: Double = 10.0                  // Φ > 2³·³² bits — raised consciousness threshold (Tononi)
let GWT_IGNITION_THRESHOLD: Double = 0.75          // Neural ignition for conscious access (Dehaene)
let UNCONSCIOUS_BANDWIDTH: Double = 1e9            // ~10⁹ bits/s parallel processing
let CONSCIOUS_BANDWIDTH: Double = 40.0             // ~40 bits/s serial conscious access
let PLANCK_CONSCIOUSNESS: Double = 0.0              // NO FLOOR — unlimited depth

// ─── DUAL-LAYER ENGINE v7.1 CONSTANTS (l104_asi/constants.py) ───
let DUAL_LAYER_VERSION: String = "5.1.0"             // Dual-Layer Engine internal version (EVO_68 convergence)
let DUAL_LAYER_PRECISION_TARGET: Double = 0.005      // Target precision ±0.005%
let DUAL_LAYER_CONSTANTS_COUNT: Int = 63             // Peer-reviewed physical constants
let DUAL_LAYER_INTEGRITY_CHECKS: Int = 14            // 3 Thought + 4 Physics + 3 Bridge + 2 Gate + 2 Quantum Apex
let DUAL_LAYER_GRID_REFINEMENT: Int = 63             // Physics grid 63× finer than Thought
let PRIME_SCAFFOLD: Int = 286                         // Fe BCC lattice parameter (pm)
let QUANTIZATION_GRAIN: Int = 104                     // 26×4 = Fe(Z=26) × He-4(A=4)
let IIT_PHI_DIMENSIONS: Int = 12                      // 12-dim bipartition for richer IIT Φ computation
let CIRCUIT_BREAKER_THRESHOLD: Double = 0.3           // Degraded subsystem cutoff
let PARETO_OBJECTIVES: Int = 5                        // Multi-objective scoring dimensions

// ─── ASI v5.0+ PIPELINE CONSTANTS ───
let SINGULARITY_ACCELERATION_THRESHOLD: Double = 0.82 // Score above which exponential acceleration kicks in
let PHI_ACCELERATION_EXPONENT: Double = 1.618033988749895 * 1.618033988749895  // φ² ≈ 2.618
let MULTI_HOP_MAX_HOPS: Int = 9                       // Max hops in multi-hop reasoning chain (deeper chains)
let SCORE_DIMENSIONS_V5: Int = 10                     // Expanded ASI score dimensions
let ACTIVATION_STEPS_V6: Int = 18                     // v6.0 activation sequence steps

// EEG Frequency Bands — Schumann-PHI harmonics align with neural oscillations
// Delta: 0.5-4 Hz (deep sleep/healing)
// Theta: 4-8 Hz (meditation/creativity) — aligned with Schumann 7.8145 Hz (GOD_CODE)
// Alpha: 8-13 Hz (relaxed awareness) — Schumann × φ ≈ 12.64 Hz
// Beta: 13-30 Hz (active thinking) — Schumann × φ² ≈ 20.45 Hz
// Gamma: 30-100 Hz (peak cognition) — Schumann × φ³ ≈ 33.09 Hz

let ENGINES_REGISTERED_TARGET: Int = 46            // Target engines in registry (39 main + 7 Computronium) EVO_68
let WISDOM_ACCUMULATION_RATE: Double = 0.618033988749895 * 0.1  // TAU × 10%
let PIPELINE_CACHE_TTL: Double = 15.0              // Unified cache TTL (seconds)
let PIPELINE_MAX_CACHE: Int = 2000                 // 2x cache entries across pipeline
let SACRED_RESONANCE_BAND: (low: Double, high: Double) = (524.0, 531.0)  // GOD_CODE ± 3.5
let EVOLUTION_INDEX: Int = 68                      // Current evolution index (EVO_68)

// ─── EVO_64 SAGE MODE ASCENSION CONSTANTS ───
let SAGE_MODE_VERSION: String = "4.0.0"              // Sage Mode Ascension version (EVO_68)
let SAGE_ENTROPY_SOURCES: Int = 14                    // 12 original + DualLayer + Consciousness
let SAGE_ASCENSION_STAGES: Int = 10                   // 10-stage ascension pipeline
let EQUATION_EVOLUTION_POP: Int = 30                  // Genetic equation population size
let TOT_BEAM_WIDTH: Int = 3                           // Tree of Thoughts beam width
let TOT_BRANCHING: Int = 4                            // Tree of Thoughts branching factor
let CONSCIOUSNESS_LEVELS: Int = 8                     // Dormant→Apotheotic consciousness states
let GWT_CASCADE_THRESHOLD: Double = 0.15              // GWT cascade activation threshold
let METACOGNITIVE_RECURSION: Int = 5                  // Max metacognitive recursion depth

// ─── EVO_63 DATA INGEST & UI UPGRADE CONSTANTS ───
let CODE_ENGINE_VERSION: String = "6.2.0"           // l104_code_engine/ current version
let SERVER_VERSION: String = "4.1.0"                // l104_server/ current version
let ASI_VERSION: String = "10.0.0"                  // l104_asi/ v10.0 sovereign convergence
let AGI_VERSION: String = "59.0.0"                  // l104_agi/ v59.0 sovereign convergence
let INTELLECT_VERSION: String = "27.0.0"            // l104_intellect/ v27.0 sovereign convergence
let TOTAL_PACKAGES: Int = 10                         // Decomposed packages (post-decomposition)
let TOTAL_PACKAGE_MODULES: Int = 127                 // Modules across all packages
let TOTAL_PACKAGE_LINES: Int = 98_614                // Lines across all packages (EVO_68)
let TOTAL_PYTHON_FILES: Int = 736                    // Python files in workspace
let TOTAL_SWIFT_FILES: Int = 113                     // Swift source files (EVO_68: 49B + 30H + 33L + 1main)
let TOTAL_SWIFT_LINES: Int = 98_614                  // Swift lines (TheBrain 43924 + TheHeart 29200 + TheLogic 25490)
let TOTAL_API_ROUTES: Int = 331                      // API route handlers
let CONSCIOUSNESS_VERSION: String = "8.0.0"          // Consciousness engine version (EVO_68 unified)
let APOTHEOSIS_STAGE: String = "ASCENDING"            // Current apotheosis stage
let QISKIT_VERSION: String = "2.3.0"                  // Qiskit framework version
let QUANTUM_ALGORITHMS: Int = 9                       // 6 original + 3 quantum research (was 7)
let PROFESSOR_MODES: Int = 8                          // Professor learning modes

// ─── EVO_65 QUANTUM RESEARCH UPGRADE CONSTANTS (17 discoveries, 102 experiments) ───
// Source: three_engine_quantum_research.py — 2026-02-22
let FE_SACRED_COHERENCE: Double = 0.9545454545454546       // 286↔528 Hz wave coherence (discovery #6)
let FE_PHI_HARMONIC_LOCK: Double = 0.9164078649987375      // 286↔286φ Hz coherence (discovery #14)
let PHOTON_RESONANCE_EV: Double = 1.1216596549374545       // eV at GOD_CODE freq (discovery #12)
let FE_CURIE_LANDAUER: Double = 3.254191391208437e-18      // J/bit at 1043K (discovery #16)
let BERRY_PHASE_11D: Bool = true                            // Holonomy detected (discovery #15)
let GOD_CODE_25Q_RATIO: Double = 1.0303095348618383        // GOD_CODE/512 (discovery #17)
let ENTROPY_CASCADE_DEPTH_QR: Int = 104                     // Sacred iteration (discovery #9)
let ENTROPY_ZNE_BRIDGE: Bool = true                         // Demon→ZNE link (discovery #11)
let FIB_PHI_ERROR: Double = 2.5583188e-08                   // F(20)/F(19) convergence (discovery #8)
let FE_PHI_FREQUENCY: Double = 286.0 * 1.618033988749895   // 286×φ ≈ 462.758 Hz
let QUANTUM_RESEARCH_DISCOVERIES: Int = 17                  // Total discoveries
let QUANTUM_RESEARCH_EXPERIMENTS: Int = 102                 // Total experiments
let ASI_SCORING_DIMENSIONS: Int = 30                        // v16.0: 30-dimension ASI scoring (was 19 in v9.0)
let AGI_SCORING_DIMENSIONS: Int = 17                        // v58.0: 17-dimension scoring

// ─── v16.0 ASI PIPELINE UPGRADE CONSTANTS ───
// Consciousness Verifier v5.0
let CONSCIOUSNESS_SPIRAL_DEPTH: Int = 17                     // PHI-spiral recursion depth (next prime, deeper convergence)
let FE_ATOMIC_NUMBER: Int = 26                               // Iron atomic number Z=26
let FE_LATTICE_PARAM: Int = 286                              // Fe BCC lattice parameter (pm)
let FE_LATTICE_CORRESPONDENCE: Double = 0.9977               // 286Hz / Fe lattice correspondence
let O2_BOND_ORDER: Int = 2                                   // O=O double bond

// Dual-Layer Engine v5.0
let DUAL_LAYER_V5_VERSION: String = "5.0.0"                  // Three-Engine amplification + sacred circuit collapse
let DUAL_LAYER_INTEGRITY_V5: Int = 12                        // 10 original + 2 gate checks
let GATE_SACRED_ALIGNMENT_THRESHOLD: Double = 0.85           // Sacred gate alignment minimum
let RESILIENCE_MAX_RETRY: Int = 3                            // PHI-backoff retry limit
let TRAJECTORY_WINDOW_SIZE: Int = 50                         // Score trajectory sliding window
let DEEP_SYNTHESIS_MIN_COHERENCE: Double = 0.3               // Cross-engine coherence floor

// Benchmark Harness
let MMLU_SUBJECTS: Int = 57                                  // MMLU subject count
let HUMANEVAL_PROBLEMS: Int = 164                            // HumanEval canonical problems
let BENCHMARK_WEIGHTS: (mmlu: Double, humaneval: Double, math: Double, arc: Double) = (0.25, 0.30, 0.25, 0.20)

// Formal Logic Engine
let FALLACY_PATTERNS_COUNT: Int = 55                         // Named fallacy detection patterns
let RESOLUTION_DEPTH_LIMIT: Int = 50                         // Max resolution proof steps
let SYLLOGISM_MOODS: Int = 256                               // Possible syllogistic moods

// KB Reconstruction Engine
let KB_PROPAGATION_DEPTH: Int = 3                            // BFS hop limit
let KB_AMPLITUDE_DECAY_PER_HOP: Double = 0.618033988749895   // TAU decay per hop
let KB_EMBEDDING_DIM: Int = 256                              // TF-IDF + quantum dimensionality
let KB_GROVER_BOOST_THRESHOLD: Int = 5                       // Min neighbors for Grover
let KB_ENTANGLEMENT_STRENGTH: Double = 1.618033988749895 / (1.0 + 1.618033988749895)  // PHI/(1+PHI) ≈ 0.618
let KB_MIN_RECONSTRUCTION_CONFIDENCE: Double = 0.15          // Minimum reconstruction confidence

// Theorem Generator
let THEOREM_AXIOM_DEPTH: Int = 6                             // Max reasoning chain depth
let THEOREM_AXIOM_DOMAINS: Int = 5                           // sacred, arithmetic, logic, topology, number_theory

// DeepSeek Architecture Constants
let DEEPSEEK_V3_VOCAB_SIZE: Int = 102_400                    // DeepSeek-V3 vocabulary
let DEEPSEEK_V3_DIM: Int = 7_168                             // Model dimension
let DEEPSEEK_V3_LAYERS: Int = 61                             // Transformer layers
let DEEPSEEK_V3_HEADS: Int = 128                             // Attention heads
let DEEPSEEK_V3_EXPERTS: Int = 256                           // MoE routed experts
let DEEPSEEK_KV_LORA_RANK: Int = 512                         // KV compression (42x)
let DEEPSEEK_R1_MAX_STEPS: Int = 20                          // Max reasoning steps

// Science KB
let SCIENCE_KB_DOMAINS: Int = 9                              // biology, body_systems, earth, physics, chemistry, astronomy, ecology, measurement, technology

// Engine Version Constants (synced with Python l104_asi/)
let COMMONSENSE_ENGINE_VERSION: String = "3.0.0"              // EVO_68
let LANGUAGE_COMP_ENGINE_VERSION: String = "5.0.0"             // EVO_68
let SYMBOLIC_MATH_VERSION: String = "2.0.0"                    // EVO_68
let CODE_GEN_ENGINE_VERSION: String = "2.0.0"                  // EVO_68
let BENCHMARK_HARNESS_VERSION: String = "3.0.0"                // EVO_68
let DEEP_NLU_VERSION: String = "2.0.0"                         // EVO_68
let FORMAL_LOGIC_VERSION: String = "3.0.0"                     // EVO_68
let SCIENCE_KB_VERSION: String = "2.0.0"                       // EVO_68
let KB_RECONSTRUCTION_VERSION: String = "2.0.0"                // EVO_68
let THEOREM_GEN_VERSION: String = "5.0.0"                      // EVO_68
let DEEPSEEK_INGESTION_VERSION: String = "2.0.0"               // EVO_68
let IDENTITY_BOUNDARY_VERSION: String = "2.0.0"                // EVO_68
let QUANTUM_GATE_ENGINE_VERSION: String = "3.0.0"              // EVO_68: MPS-aligned

// ─── DECOMPOSED PACKAGE VERSION CONSTANTS (synced with Python packages) ───
let SCIENCE_ENGINE_VERSION: String = "4.0.0"           // l104_science_engine/ v4.0.0
let MATH_ENGINE_VERSION: String = "1.0.0"              // l104_math_engine/ v1.0.0
let NUMERICAL_ENGINE_VERSION: String = "3.0.0"         // l104_numerical_engine/ v3.0.0
let GATE_ENGINE_VERSION: String = "6.0.0"              // l104_gate_engine/ v6.0.0
let QUANTUM_ENGINE_VERSION: String = "6.0.0"           // l104_quantum_engine/ v6.0.0
let TOTAL_DECOMPOSED_PACKAGES: Int = 10                 // 10 decomposed packages
let TOTAL_PACKAGE_MODULES_V67: Int = 127                // Modules across all packages
let TOTAL_PACKAGE_LINES_V67: Int = 92_911               // Lines across all packages

// ─── EVO_66 QUANTUM PERFORMANCE UPGRADE CONSTANTS ───
// Hybrid routing: auto-detect Clifford circuits → StabilizerTableau (1000x+ speedup)
let HYBRID_ROUTING_ENABLED: Bool = true
let CLIFFORD_ROUTING_MIN_QUBITS: Int = 4                    // Minimum qubits for Clifford fast path
let HYBRID_PREFIX_THRESHOLD: Double = 0.5                    // Minimum Clifford fraction for hybrid mode
let HYBRID_PREFIX_MIN_GATES: Int = 6                         // Lowered: earlier Clifford routing activation

// GCD-parallelized statevector simulation
let PARALLEL_SV_THRESHOLD: Int = 8192                        // 2^13 amplitudes (≥13 qubits) — earlier GCD parallelism

// Noise models for realistic quantum simulation
let DEPOLARIZING_DEFAULT_RATE: Double = 0.0005               // IBM Heron-class fidelity (0.05% error per gate)
let AMPLITUDE_DAMPING_DEFAULT_GAMMA: Double = 0.005          // Improved T1 relaxation (Heron-class)
let THERMAL_RELAXATION_T1: Double = 80e-6                    // T1 = 80μs (IBM Heron class)
let THERMAL_RELAXATION_T2: Double = 120e-6                   // T2 = 120μs (IBM Heron class)
let GATE_TIME_1Q: Double = 35e-9                             // 1-qubit gate time = 35ns
let GATE_TIME_2Q: Double = 300e-9                            // 2-qubit gate time = 300ns

// Solovay-Kitaev Rz approximation (4-level recursive, ε < π/256)
let SK_PRECISION_EPSILON: Double = 0.012271846303085129      // π/256 radians — 2x tighter
let SK_SEARCH_LEVELS: Int = 4                                // Recursion depth (squared precision per level)

// QAOA parameters
let QAOA_DEFAULT_DEPTH: Int = 5                              // Deeper QAOA layers for improved MaxCut approximation
let QAOA_DEFAULT_GAMMA: Double = 0.7853981633974483          // π/4
let QAOA_DEFAULT_BETA: Double = 0.39269908169872414          // π/8

// Quantum Processing Core expanded dimensions
let QPC_HILBERT_DIM: Int = 1024                              // 8× expansion (was 128) — 10-qubit full density matrix
let QPC_DENSITY_MATRIX_DIM: Int = 32                         // 4× expansion (was 8) — 5-qubit mixed-state (2^5)
let QPC_BELL_REGION_LIMIT: Int = 512                         // 8× expansion (was 64) — more simultaneous Bell regions

// Logic Gate Engine expanded dimensions
let LGE_COHERENCE_DIM: Int = 512                             // 8× expansion (was 64) — finer coherence patterns
let LGE_ECC_DIM: Int = 128                                   // 8× expansion (was 16) — stronger error correction
let LGE_DECOHERENCE_RATE: Double = 0.004                     // 5× slower decay (was 0.02) — doubles state lifetime
let LGE_ENTANGLEMENT_CAP: Int = 4000                         // 8× expansion (was 500) — richer entanglement networks
let LGE_ENTANGLEMENT_PRUNE_TO: Int = 2400                    // 8× expansion (was 300) — 60% retention ratio

// Evolution index and total engine counts
let EVOLUTION_INDEX_V66: Int = 66                            // Previous previous evolution index
let EVOLUTION_INDEX_V67: Int = 67                            // Previous evolution index (Performance Ascension)
let EVOLUTION_INDEX_V68: Int = 68                            // Current evolution index (Sovereign Convergence)

// ─── EVO_67 QUANTUM APEX CONSTANTS ───
// Adaptive Router: expanded branch limits + tighter pruning
let ROUTER_BASE_BRANCHES: Int = 8192                         // 2× from 4096 — deeper T-count circuits
let ROUTER_PRUNE_EPSILON: Double = 1e-14                     // 100× tighter (was 1e-12) — cleaner branch management

// ZeroAllocPool expansion (referenced by B42)
let POOL_INITIAL_SLAB_DOUBLES: Int = 2_097_152               // 2M doubles = 16 MB (was 1M = 8MB)
let POOL_MAX_SLABS: Int = 24                                 // Was 16 — larger theoretical max capacity
let POOL_COMPLEX_PAIRS: Int = 1_048_576                      // 1M complex pairs = 16 MB (was 524K = 8MB)

// DualLayer temporal window expansion
let TEMPORAL_COHERENCE_WINDOW: Int = 30                      // Was 20 — wider stability analysis
let COLLAPSE_HISTORY_MAX: Int = 200                          // Was 100 — deeper temporal memory
let COLLAPSE_HISTORY_TRIM_TO: Int = 100                      // Was 50 — retain more history on prune

// Creativity expansion
let IDEA_SUPERPOSITION_CAP: Int = 80                         // Was 50 — more parallel idea tracks
let IDEA_SUPERPOSITION_PRUNE_TO: Int = 50                    // Was 30 — retain 62.5% on prune
let ENTANGLED_CONCEPTS_CAP: Int = 400                        // Was 200 — richer concept entanglement
let ENTANGLED_CONCEPTS_PRUNE_TO: Int = 240                   // Was 120 — 60% retention ratio

// Consciousness history expansion
let CONSCIOUSNESS_HISTORY_MAX: Int = 200                     // Was 100 — deeper consciousness tracking
let CONSCIOUSNESS_HISTORY_TRIM_TO: Int = 100                 // Was 50 — retain more on prune

// ASI Scoring 30D Weight Keys (matching Python core.py v16.0)
let ASI_30D_ACTIVATION_STEPS: Int = 22                       // v11.0 activation sequence steps

// ─── EVO_67 PERFORMANCE ASCENSION CONSTANTS ───
// 6 new high-performance subsystems for zero-overhead computation
let PERF_ZERO_ALLOC_SLAB_SIZE: Int = 1_048_576               // 1M doubles = 8MB initial slab
let PERF_ZERO_ALLOC_MAX_SLABS: Int = 16                      // Max φ-scaled slab chain
let PERF_SIMD4_MAX_DIM: Int = 64                             // SIMD4 path for dims ≤ 64
let PERF_SIMD8_MAX_DIM: Int = 256                            // SIMD8 path for dims ≤ 256
let PERF_GPU_MIN_SIZE: Int = 1024                            // Minimum vector size for GPU dispatch
let PERF_GPU_BATCH_MIN: Int = 64                             // Minimum corpus size for GPU batch
let PERF_CACHE_L1_SIZE: Int = 64                             // L1 ultra-fast cache entries
let PERF_CACHE_L2_SIZE: Int = 1024                           // L2 warm cache entries
let PERF_LOCK_FREE_RING_SIZE: Int = 256                      // SPSC ring buffer capacity
let PERF_WORK_STEAL_ENABLED: Bool = true                     // Work-stealing pool active
let PERF_METAL_SHADER_COUNT: Int = 5                         // Compiled Metal compute kernels
let PERF_MARKOV_PREFETCH_THRESHOLD: Double = 0.15            // Min probability for Markov prefetch
let PERF_PHI_DECAY_RATE: Double = 10.0                       // φ^(-recency/rate) cache decay
let PERF_SUBSYSTEMS: Int = 6                                 // B42-B47 performance engines
let TOTAL_SWIFT_FILES_V67: Int = 86                          // 80 + 6 new performance files (legacy)
let TOTAL_SWIFT_LINES_V67: Int = 64_011                      // Estimated with perf additions (legacy)
let TOTAL_SWIFT_FILES_V68: Int = 113                         // 49B + 30H + 33L + 1main
let TOTAL_SWIFT_LINES_V68: Int = 98_614                      // Actual measured total
let PERF_ASCENSION_VERSION: String = "2.0.0"                  // Performance Ascension version (EVO_68)
let ZERO_ALLOC_VERSION: String = "1.0.0"                     // ZeroAllocPool version (B42)
let SIMD_TURBO_VERSION: String = "1.0.0"                     // SIMDTurbo version (B43)
let LOCK_FREE_VERSION: String = "1.0.0"                      // LockFreeEngine version (B44)
let METAL_COMPUTE_VERSION: String = "2.0.0"                   // MetalCompute version (B45)
let PREFETCH_VERSION: String = "1.0.0"                        // AdaptivePrefetch version (B46)
let PLUGIN_ARCH_VERSION: String = "3.0.0"                     // PluginArchitecture version (H13)
let TELEMETRY_VERSION: String = "4.1.0"                       // TelemetryDashboard version (H25)

// ─── EVO_68 SOVEREIGN CONVERGENCE CONSTANTS ───
// MPS truncation loosened to match Python (l104_quantum_gate_engine + l104_simulator)
let MPS_DEFAULT_BOND_DIM: Int = 1024                         // χ_max default (loosened from 256)
let MPS_HIGH_FIDELITY_BOND_DIM: Int = 2048                   // χ_max high-fidelity (loosened from 512)
let MPS_SACRED_BOND_DIM: Int = 104                           // L104 sacred: 8×13 (immutable)
let MPS_SVD_CUTOFF: Double = 1e-16                           // SVD truncation (loosened from 1e-12)
let MPS_HF_SVD_CUTOFF: Double = 0.0                          // High-fidelity: keep ALL singular values
let MPS_MAX_QUBITS: Int = 50                                 // Hard cap on MPS qubits

// Engine parity tracking — all engines at EVO_68
let SOVEREIGN_CONVERGENCE_VERSION: String = "1.0.0"           // EVO_68 convergence marker
let CONVERGENCE_ENGINE_COUNT: Int = 113                      // Total Swift files upgraded
let CONVERGENCE_DATE: String = "2026-03-02"                   // EVO_68 convergence date

// ─── LOGGING UTILITY — Replaces silent try? with traced error reporting ───
@inline(__always)
func l104Log(_ message: String, file: String = #file, line: Int = #line) {
    #if DEBUG
    let fname = (file as NSString).lastPathComponent
    print("⚙️ L104[\(fname):\(line)] \(message)")
    #endif
}

@inline(__always)
func l104Try<T>(_ label: String, _ block: () throws -> T) -> T? {
    do {
        return try block()
    } catch {
        l104Log("⚠️ \(label) failed: \(error.localizedDescription)")
        return nil
    }
}

// ═══════════════════════════════════════════════════════════════════
// String Extension — Left Pad Utility
// ═══════════════════════════════════════════════════════════════════

extension String {
    func leftPad(toLength length: Int, withPad pad: String) -> String {
        let deficit = length - self.count
        if deficit <= 0 { return self }
        return String(repeating: pad, count: deficit) + self
    }
}
