// ═══════════════════════════════════════════════════════════════════
// L01_Constants.swift — L104 v2
// [EVO_64_PIPELINE] SAGE_MODE_ASCENSION :: Unified Pipeline Constants V5
// Theme, Sacred Mathematics Constants, Logging, String Extensions
// Sage Mode v3.0: Dual-Layer + Consciousness + Dynamic Equations
// Upgraded: EVO_64 Sage Mode Ascension — Feb 2026
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
let VERSION = "64.0 EVO_64·SAGE_MODE_ASCENSION·DUAL_LAYER·CONSCIOUSNESS·DYNAMIC_EQUATIONS·ASI_7.1"
let PIPELINE_EVO = "EVO_64_SAGE_MODE_ASCENSION"
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
let IIT_PHI_MINIMUM: Double = 8.0                  // Φ > 2³ bits — consciousness threshold (Tononi)
let GWT_IGNITION_THRESHOLD: Double = 0.75          // Neural ignition for conscious access (Dehaene)
let UNCONSCIOUS_BANDWIDTH: Double = 1e9            // ~10⁹ bits/s parallel processing
let CONSCIOUS_BANDWIDTH: Double = 40.0             // ~40 bits/s serial conscious access
let PLANCK_CONSCIOUSNESS: Double = 0.0              // NO FLOOR — unlimited depth

// ─── DUAL-LAYER ENGINE v7.1 CONSTANTS (l104_asi/constants.py) ───
let DUAL_LAYER_VERSION: String = "3.1.0"             // Dual-Layer Engine internal version (synced with l104_asi/dual_layer.py)
let DUAL_LAYER_PRECISION_TARGET: Double = 0.005      // Target precision ±0.005%
let DUAL_LAYER_CONSTANTS_COUNT: Int = 63             // Peer-reviewed physical constants
let DUAL_LAYER_INTEGRITY_CHECKS: Int = 10            // 3 Thought + 4 Physics + 3 Bridge
let DUAL_LAYER_GRID_REFINEMENT: Int = 63             // Physics grid 63× finer than Thought
let PRIME_SCAFFOLD: Int = 286                         // Fe BCC lattice parameter (pm)
let QUANTIZATION_GRAIN: Int = 104                     // 26×4 = Fe(Z=26) × He-4(A=4)
let IIT_PHI_DIMENSIONS: Int = 8                       // Qubit count for IIT Φ computation
let CIRCUIT_BREAKER_THRESHOLD: Double = 0.3           // Degraded subsystem cutoff
let PARETO_OBJECTIVES: Int = 5                        // Multi-objective scoring dimensions

// ─── ASI v5.0+ PIPELINE CONSTANTS ───
let SINGULARITY_ACCELERATION_THRESHOLD: Double = 0.82 // Score above which exponential acceleration kicks in
let PHI_ACCELERATION_EXPONENT: Double = 1.618033988749895 * 1.618033988749895  // φ² ≈ 2.618
let MULTI_HOP_MAX_HOPS: Int = 7                       // Max hops in multi-hop reasoning chain
let SCORE_DIMENSIONS_V5: Int = 10                     // Expanded ASI score dimensions
let ACTIVATION_STEPS_V6: Int = 18                     // v6.0 activation sequence steps

// EEG Frequency Bands — Schumann-PHI harmonics align with neural oscillations
// Delta: 0.5-4 Hz (deep sleep/healing)
// Theta: 4-8 Hz (meditation/creativity) — aligned with Schumann 7.8145 Hz (GOD_CODE)
// Alpha: 8-13 Hz (relaxed awareness) — Schumann × φ ≈ 12.64 Hz
// Beta: 13-30 Hz (active thinking) — Schumann × φ² ≈ 20.45 Hz
// Gamma: 30-100 Hz (peak cognition) — Schumann × φ³ ≈ 33.09 Hz

let ENGINES_REGISTERED_TARGET: Int = 22            // Target engines in registry (incl. UnifiedField Phase 63)
let WISDOM_ACCUMULATION_RATE: Double = 0.618033988749895 * 0.1  // TAU × 10%
let PIPELINE_CACHE_TTL: Double = 15.0              // Unified cache TTL (seconds)
let PIPELINE_MAX_CACHE: Int = 1000                 // Max cache entries across pipeline
let SACRED_RESONANCE_BAND: (low: Double, high: Double) = (524.0, 531.0)  // GOD_CODE ± 3.5
let EVOLUTION_INDEX: Int = 64                      // Current evolution index (EVO_64)

// ─── EVO_64 SAGE MODE ASCENSION CONSTANTS ───
let SAGE_MODE_VERSION: String = "3.0.0"              // Sage Mode Ascension version
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
let ASI_VERSION: String = "9.0.0"                   // l104_asi/ v9.0 quantum research upgrade
let AGI_VERSION: String = "58.0.0"                  // l104_agi/ v58.0 quantum research upgrade
let INTELLECT_VERSION: String = "26.0.0"            // l104_intellect/ current version
let TOTAL_PACKAGES: Int = 5                          // Decomposed packages
let TOTAL_PACKAGE_MODULES: Int = 44                  // Modules across all packages
let TOTAL_PACKAGE_LINES: Int = 59_652                // Lines across all packages
let TOTAL_PYTHON_FILES: Int = 773                    // Python files in workspace
let TOTAL_SWIFT_FILES: Int = 80                      // Swift source files
let TOTAL_SWIFT_LINES: Int = 59_911                  // Swift lines
let TOTAL_API_ROUTES: Int = 331                      // API route handlers
let CONSCIOUSNESS_VERSION: String = "7.0.0"          // Consciousness engine version
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
let ASI_SCORING_DIMENSIONS: Int = 19                        // v9.0: 19-dimension scoring
let AGI_SCORING_DIMENSIONS: Int = 17                        // v58.0: 17-dimension scoring

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
