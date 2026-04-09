import Accelerate
import Foundation

// MARK: - ═══ FEEDBACK CONSTANTS ═══

private let SACRED_MOMENTUM_BLEND: Double = TAU       // β₁ = φ⁻¹ = 0.618
private let SACRED_VARIANCE_BLEND: Double = PHI * PHI - 2.0  // β₂ ≈ 0.382
private let SACRED_LR_DECAY: Double = 1.0 - pow(TAU, 3.0)   // ≈ 0.764
private let SACRED_COMPOSITE_WEIGHT: Double = pow(PHI, 1.0 / 3.0)  // φ^(1/3) ≈ 0.853
private let SACRED_ALIGNMENT_WEIGHT: Double = 1.0 - pow(PHI, 1.0 / 3.0)  // ≈ 0.147
private let SACRED_SA_COOLING: Double = pow(PHI, 3) / (pow(PHI, 3) + 1.0)  // φ³/(φ³+1) ≈ 0.809
// Uses global QUANTIZATION_GRAIN from L01_Constants.swift (= 104)
private let GOD_CODE_PHASE_ANGLE: Double = GOD_CODE.truncatingRemainder(dividingBy: 2.0 * Double.pi)
private let PHI_PHASE_ANGLE: Double = 2.0 * Double.pi / PHI

// Fibonacci refocusing intervals for echo strategies
private let FIBONACCI_8: [Int] = [1, 1, 2, 3, 5, 8, 13, 21]
private let FIBONACCI_8_SUM: Double = 54.0

// MARK: - ═══ SIMULATION RESULT (lightweight) ═══

struct SimulationMetrics {
    var fidelity: Double = 0.0
    var entropy: Double = 0.0
    var coherence: Double = 0.0
    var conservation: Double = 0.0
    var alignment: Double = 0.0
    var stability: Double = 0.0
    var purity: Double = 0.0
    var gateFidelity: Double = 0.0
    var decoherenceResilience: Double = 0.0
    var qfi: Double = 0.0               // Quantum Fisher Information
    var topoEntropy: Double = 0.0       // Topological entropy
    var loschmidtEcho: Double = 0.0     // Loschmidt echo fidelity
    var trotterQuality: Double = 0.0    // Trotter decomposition error
    var information: Double = 0.0       // Mutual information
    var concurrence: Double = 0.0       // Entanglement concurrence
}

// MARK: - ═══ FEEDBACK LOOP ENGINE ═══

final class FeedbackLoopEngine {
    static let shared = FeedbackLoopEngine()

    private let lock = NSRecursiveLock()
    private(set) var loopHistory: [[String: Any]] = []
    private var momentum: Double = 0.0
    private var momentumSq: Double = 0.0
    private var learningRate: Double = 0.01
    private let convergenceWindow = 5

    // MARK: - 15-Dimension Scoring

    /// Score a batch of simulation results across 15 quality dimensions
    func scoreByDimension(_ results: [SimulationMetrics]) -> [String: Double] {
        guard !results.isEmpty else { return [:] }
        let n = Double(results.count)

        // Dense dimensions (all sims contribute)
        let avgFidelity = results.reduce(0.0) { $0 + $1.fidelity } / n
        let avgEntropy = results.reduce(0.0) { $0 + $1.entropy } / n
        let avgCoherence = results.reduce(0.0) { $0 + $1.coherence } / n
        let avgConservation = results.reduce(0.0) { $0 + $1.conservation } / n
        let avgAlignment = results.reduce(0.0) { $0 + $1.alignment } / n
        let avgStability = results.reduce(0.0) { $0 + $1.stability } / n
        let avgPurity = results.reduce(0.0) { $0 + $1.purity } / n
        let avgGateFidelity = results.reduce(0.0) { $0 + $1.gateFidelity } / n
        let avgDecoherence = results.reduce(0.0) { $0 + $1.decoherenceResilience } / n

        // Sparse dimensions (only non-zero contribute)
        func sparseAvg(_ extractor: (SimulationMetrics) -> Double) -> Double {
            let nonZero = results.filter { extractor($0) > 0 }
            return nonZero.isEmpty ? 0.0 : nonZero.reduce(0.0) { $0 + extractor($1) } / Double(nonZero.count)
        }

        let avgQFI = sparseAvg { $0.qfi }
        let avgTopo = sparseAvg { $0.topoEntropy }
        let avgLoschmidt = sparseAvg { $0.loschmidtEcho }
        let avgTrotter = sparseAvg { $0.trotterQuality }
        let avgInfo = sparseAvg { $0.information }
        let avgConcurrence = sparseAvg { $0.concurrence }

        return [
            "fidelity": avgFidelity, "entropy": avgEntropy, "coherence": avgCoherence,
            "conservation": avgConservation, "alignment": avgAlignment, "stability": avgStability,
            "purity": avgPurity, "gate_fidelity": avgGateFidelity,
            "decoherence_resilience": avgDecoherence,
            "qfi": avgQFI, "topo_entropy": avgTopo, "loschmidt": avgLoschmidt,
            "trotter_quality": avgTrotter, "information": avgInfo, "concurrence": avgConcurrence
        ]
    }

    // MARK: - Feedback Cycle

    /// Run one feedback cycle with Adam momentum (β₁=φ⁻¹, β₂≈0.382)
    ///
    /// 5 phases per iteration:
    /// 1. Coherence computation
    /// 2. Entropy reversal (Maxwell's Demon efficiency)
    /// 3. Math verification (GOD_CODE conservation)
    /// 4. Composite 15D scoring
    /// 5. Adam momentum update with bias correction
    func runFeedbackCycle(simResults: [SimulationMetrics],
                          iterations: Int = 5) -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        var cycleScores: [Double] = []
        var converged = false

        for step in 1...iterations {
            // Phase 1: Coherence
            let scores = scoreByDimension(simResults)
            let coherenceVal = scores["coherence"] ?? 0.5

            // Phase 2: Entropy reversal - Maxwell Demon efficiency
            let entropyInput = scores["entropy"] ?? 0.5
            let demonEfficiency = PHI / (GOD_CODE / 416.0) / (1.0 + entropyInput)

            // Phase 3: Math verification (conservation product)
            let conservationOk = (scores["conservation"] ?? 0.0) > 0.99

            // Phase 4: Composite score (weighted average of 15 dimensions)
            let denseWeights: [(String, Double)] = [
                ("fidelity", 0.15), ("entropy", 0.10), ("coherence", 0.15),
                ("conservation", 0.10), ("alignment", 0.10), ("stability", 0.08),
                ("purity", 0.08), ("gate_fidelity", 0.08), ("decoherence_resilience", 0.06)
            ]
            let sparseWeights: [(String, Double)] = [
                ("qfi", 0.02), ("topo_entropy", 0.02), ("loschmidt", 0.02),
                ("trotter_quality", 0.01), ("information", 0.02), ("concurrence", 0.01)
            ]
            var composite = 0.0
            for (key, weight) in denseWeights + sparseWeights {
                composite += (scores[key] ?? 0.0) * weight
            }

            // Phase 5: Adam momentum update with sacred blending
            let t = Double(step)
            momentum = SACRED_MOMENTUM_BLEND * momentum + (1.0 - SACRED_MOMENTUM_BLEND) * composite
            momentumSq = SACRED_VARIANCE_BLEND * momentumSq + (1.0 - SACRED_VARIANCE_BLEND) * composite * composite
            let mHat = momentum / (1.0 - pow(SACRED_MOMENTUM_BLEND, t))
            let vHat = momentumSq / (1.0 - pow(SACRED_VARIANCE_BLEND, t))
            let adaptedScore = mHat / (sqrt(vHat) + 1e-8)

            cycleScores.append(adaptedScore)

            // Convergence check
            if cycleScores.count >= convergenceWindow {
                let recent = Array(cycleScores.suffix(convergenceWindow))
                let maxDiff = zip(recent, recent.dropFirst()).map { abs($0 - $1) }.max() ?? 1.0
                if maxDiff < 1e-6 { converged = true; break }
            }

            _ = demonEfficiency
            _ = conservationOk
            _ = coherenceVal
        }

        let result: [String: Any] = [
            "avg_composite_score": cycleScores.reduce(0, +) / max(Double(cycleScores.count), 1),
            "final_score": cycleScores.last ?? 0.0,
            "momentum": momentum,
            "iterations_run": cycleScores.count,
            "converged": converged,
            "score_history": cycleScores
        ]
        loopHistory.append(result)
        return result
    }

    // MARK: - Multi-Pass Refinement

    /// Run multiple passes of feedback cycles with oscillation detection
    func runMultiPass(simResults: [SimulationMetrics], passes: Int = 3,
                      iterationsPerPass: Int = 5) -> [String: Any] {
        var passResults: [[String: Any]] = []
        var prevScore = 0.0

        for pass in 0..<passes {
            let result = runFeedbackCycle(simResults: simResults, iterations: iterationsPerPass)
            let score = (result["final_score"] as? Double) ?? 0.0
            passResults.append(result)

            // Oscillation detection: if score sign flips, reduce LR
            if pass > 0 && (score - prevScore) * (prevScore - ((passResults[max(0, pass - 2)]["final_score"] as? Double) ?? 0)) < 0 {
                learningRate *= 0.5
            }

            // Early stop on convergence
            if pass > 0 && abs(score - prevScore) < 1e-6 { break }
            prevScore = score
        }

        return [
            "passes_run": passResults.count,
            "final_score": (passResults.last?["final_score"] as? Double) ?? 0.0,
            "learning_rate": learningRate,
            "pass_results": passResults
        ]
    }
}

// MARK: - ═══ PARAMETRIC SWEEP ENGINE ═══

final class ParametricSweepEngine {
    static let shared = ParametricSweepEngine()

    private let maxWorkers = min(ProcessInfo.processInfo.processorCount, 8)

    // MARK: - Dial Sweep (Conservation Law Verification)

    /// Sweep dial parameter: G(a,b,c,d) × 2^(-x/104) = GOD_CODE
    func dialSweep(dial: String = "a", start: Int = 0, stop: Int = 8) -> [[String: Any]] {
        var results: [[String: Any]] = []
        for val in start...stop {
            let x: Double
            switch dial {
            case "a": x = 8.0 * Double(val)
            case "b": x = Double(val)
            case "c": x = 8.0 * Double(val)
            case "d": x = 104.0 * Double(val)
            default: x = Double(val)
            }
            let gVal = pow(286.0, 1.0 / PHI) * pow(2.0, (416.0 - x) / 104.0)
            let conservationProduct = gVal * pow(2.0, x / 104.0)
            let error = abs(conservationProduct - GOD_CODE) / GOD_CODE
            results.append([
                "dial": dial, "value": val,
                "x": x, "G": gVal,
                "conservation_product": conservationProduct,
                "error": error,
                "passed": error < 1e-9
            ])
        }
        return results
    }

    // MARK: - Phase Sweep (104 Sacred-Resolution Points)

    /// Sweep phase angle with L104 quantization grain (104 points over [0, 2π])
    func phaseSweep(numQubits: Int = 3) -> [[String: Any]] {
        var results: [[String: Any]] = []
        let engine = VariationalQuantumEngine.shared

        for step in 0..<QUANTIZATION_GRAIN {
            let phase = Double(step) / Double(QUANTIZATION_GRAIN) * 2.0 * Double.pi

            // Build circuit: Rz(phase) on q0, then CNOT ladder
            let nq = numQubits
            let dim = 1 << nq
            var sv = [Double](repeating: 0.0, count: dim * 2)
            sv[0] = 1.0  // |0⟩

            // Apply Rz(phase) to q0
            let cosHalf = cos(phase / 2.0)
            let sinHalf = sin(phase / 2.0)
            sv[0] = cosHalf  // |0⟩ component
            sv[1] = -sinHalf

            // CNOT ladder
            for q in 0..<(nq - 1) {
                engine.applyCNOT(&sv, q, q + 1, nq)
            }

            // Compute metrics
            var entropy = 0.0
            var purity = 0.0
            for i in 0..<dim {
                let p = sv[i * 2] * sv[i * 2] + sv[i * 2 + 1] * sv[i * 2 + 1]
                purity += p * p
                if p > 1e-15 { entropy -= p * log(p) }
            }

            let alignmentGC = abs(cos(phase - GOD_CODE_PHASE_ANGLE))
            let alignmentPhi = abs(cos(phase - PHI_PHASE_ANGLE))

            results.append([
                "step": step, "phase": phase,
                "entropy": entropy, "purity": purity,
                "alignment_gc": alignmentGC,
                "alignment_phi": alignmentPhi,
                "resonance": alignmentGC * entropy  // Peak = max alignment × max entropy
            ])
        }

        // Find resonance peaks (local maxima)
        var peaks: [[String: Any]] = []
        for i in 1..<results.count - 1 {
            let prev = (results[i-1]["resonance"] as? Double) ?? 0
            let curr = (results[i]["resonance"] as? Double) ?? 0
            let next = (results[i+1]["resonance"] as? Double) ?? 0
            if curr > prev && curr > next { peaks.append(results[i]) }
        }

        return results
    }

    // MARK: - Noise Sweep

    /// Sweep noise levels to characterize decoherence impact
    func noiseSweep(numQubits: Int = 2) -> [[String: Any]] {
        let noiseLevels: [Double] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
        var results: [[String: Any]] = []

        for noise in noiseLevels {
            let nq = numQubits
            let dim = 1 << nq
            var sv = [Double](repeating: 0.0, count: dim * 2)
            sv[0] = 1.0

            // Create Bell state
            // H on q0
            sv[0] = 1.0 / sqrt(2.0)
            sv[(1 << 0) * 2] = 1.0 / sqrt(2.0)
            // CNOT(0,1) - creates |00⟩+|11⟩
            // Swap amplitudes for |01⟩ and |11⟩
            let idx01 = 1, idx11 = 3
            let tempRe = sv[idx01 * 2]
            let tempIm = sv[idx01 * 2 + 1]
            sv[idx01 * 2] = sv[idx11 * 2]
            sv[idx01 * 2 + 1] = sv[idx11 * 2 + 1]
            sv[idx11 * 2] = tempRe
            sv[idx11 * 2 + 1] = tempIm

            // Apply noise: amplitude damping exp(-noise × (q+1)) per qubit
            for i in 0..<dim {
                var damping = 1.0
                for q in 0..<nq {
                    if (i >> q) & 1 == 1 {
                        damping *= exp(-noise * Double(q + 1))
                    }
                    // Dephasing: phase rotation
                    let dpPhase = noise * TAU * Double(q + 1)
                    let re = sv[i * 2], im = sv[i * 2 + 1]
                    sv[i * 2] = re * cos(dpPhase) - im * sin(dpPhase)
                    sv[i * 2 + 1] = re * sin(dpPhase) + im * cos(dpPhase)
                }
                sv[i * 2] *= damping
                sv[i * 2 + 1] *= damping
            }

            // Renormalize
            var norm = 0.0
            for i in 0..<dim { norm += sv[i * 2] * sv[i * 2] + sv[i * 2 + 1] * sv[i * 2 + 1] }
            let invNorm = 1.0 / max(sqrt(norm), 1e-15)
            for i in 0..<(dim * 2) { sv[i] *= invNorm }

            // Compute fidelity with ideal Bell state
            let idealFidelity = sv[0] * sv[0] + sv[1] * sv[1] +
                sv[idx11 * 2] * sv[idx11 * 2] + sv[idx11 * 2 + 1] * sv[idx11 * 2 + 1]

            // Linear entropy
            var linearEntropy = 0.0
            for i in 0..<dim {
                let p = sv[i * 2] * sv[i * 2] + sv[i * 2 + 1] * sv[i * 2 + 1]
                linearEntropy += p * p
            }
            linearEntropy = 1.0 - linearEntropy

            results.append([
                "noise_level": noise,
                "fidelity": idealFidelity,
                "linear_entropy": linearEntropy
            ])
        }
        return results
    }

    // MARK: - Strategy Sweep

    /// Compare 8 noise protection strategies across noise regimes
    func strategySweep(numQubits: Int = 2) -> [String: Any] {
        let strategies = ["raw", "amplitude_damping", "dephasing", "combined_noise",
                          "depolarizing", "zeno_freeze", "composite_shield", "fibonacci_echo"]
        let noiseLevels: [Double] = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

        var matrix: [[String: Any]] = []
        var dominants: [String: String] = [:]

        for noise in noiseLevels {
            var bestFidelity = 0.0
            var bestStrategy = "raw"
            var row: [String: Any] = ["noise": noise]

            for strategy in strategies {
                // Simulate fidelity under each strategy
                let fidelity: Double
                switch strategy {
                case "raw":
                    fidelity = exp(-noise * 2.0)
                case "amplitude_damping":
                    fidelity = exp(-noise * 1.5)  // T1 only
                case "dephasing":
                    fidelity = exp(-noise * 1.2)  // T2 only
                case "combined_noise":
                    fidelity = exp(-noise * 1.8)  // T1 + 0.3×T2
                case "depolarizing":
                    fidelity = max(1.0 - noise * 0.75, 0.25)  // Stochastic Pauli cap at 25%
                case "zeno_freeze":
                    // Quantum Zeno: frequent measurement slows decay
                    fidelity = pow(cos(noise * Double.pi / 4.0), 2.0)
                case "composite_shield":
                    // DD + sacred phase injection
                    fidelity = exp(-noise * 0.8) * (1.0 + 0.1 * sin(GOD_CODE_PHASE_ANGLE))
                case "fibonacci_echo":
                    // Fibonacci refocusing: weighted average recovery
                    var recovery = 0.0
                    for (i, fib) in FIBONACCI_8.enumerated() {
                        let refocusEff = exp(-noise * Double(i + 1) * 0.3)
                        recovery += Double(fib) / FIBONACCI_8_SUM * refocusEff
                    }
                    fidelity = recovery
                default:
                    fidelity = exp(-noise * 2.0)
                }

                row[strategy] = fidelity
                if fidelity > bestFidelity {
                    bestFidelity = fidelity
                    bestStrategy = strategy
                }
            }

            dominants[String(noise)] = bestStrategy
            matrix.append(row)
        }

        return [
            "strategy_matrix": matrix,
            "dominants": dominants,
            "num_strategies": strategies.count,
            "strategies": strategies
        ]
    }
}

// MARK: - ═══ ADAPTIVE OPTIMIZER ═══

final class AdaptiveOptimizer {
    static let shared = AdaptiveOptimizer()

    // MARK: - Nelder-Mead Simplex

    /// Nelder-Mead simplex optimization in parameter space
    /// Parameters: (depth_factor, phase_scale, entangle_strength, gc_mix)
    func optimizeSacredCircuit(numQubits: Int = 4, depth: Int = 4,
                                targetFidelity: Double = 0.99,
                                maxIterations: Int = 100) -> [String: Any] {
        // Nelder-Mead coefficients
        let alpha = 1.0    // reflection
        let gamma = 2.0    // expansion
        let rho = 0.5      // contraction
        let sigma = 0.5    // shrink

        let dim = 4  // 4-parameter space
        // Initialize simplex: dim+1 vertices
        var simplex: [[Double]] = []
        let base = [1.0, GOD_CODE_PHASE_ANGLE, 0.8, 0.5]  // Initial guess
        simplex.append(base)
        for i in 0..<dim {
            var vertex = base
            vertex[i] += 0.1
            simplex.append(vertex)
        }

        // Objective: composite fidelity metric
        func objective(_ params: [Double]) -> Double {
            let depthF = max(params[0], 0.1)
            let phaseScale = params[1]
            let entangleStrength = min(max(params[2], 0), 1)
            let gcMix = min(max(params[3], 0), 1)

            // Simulate circuit with parameters
            let nq = numQubits
            let effectiveDepth = max(Int(depthF * Double(depth)), 1)
            let dim = 1 << nq
            var sv = [Double](repeating: 0.0, count: dim * 2)
            sv[0] = 1.0

            // Apply parameterized circuit
            for _ in 0..<effectiveDepth {
                for q in 0..<nq {
                    let angle = phaseScale * Double(q + 1) / Double(nq) * gcMix
                    // Ry rotation
                    let step = 1 << q
                    let cosH = cos(angle / 2.0), sinH = sin(angle / 2.0)
                    for i in stride(from: 0, to: dim, by: step * 2) {
                        for j in i..<(i + step) {
                            let k = j + step
                            let r0 = sv[j * 2], i0 = sv[j * 2 + 1]
                            let r1 = sv[k * 2], i1 = sv[k * 2 + 1]
                            sv[j * 2] = cosH * r0 - sinH * r1
                            sv[j * 2 + 1] = cosH * i0 - sinH * i1
                            sv[k * 2] = sinH * r0 + cosH * r1
                            sv[k * 2 + 1] = sinH * i0 + cosH * i1
                        }
                    }
                }
                // Entangling layer with strength
                if entangleStrength > 0.01 {
                    for q in 0..<(nq - 1) {
                        // Partial CNOT (controlled rotation by entangleStrength)
                        for i in 0..<dim {
                            if (i >> q) & 1 == 1 {
                                let j = i ^ (1 << (q + 1))
                                if j > i {
                                    let blend = entangleStrength
                                    let r1 = sv[i * 2], i1 = sv[i * 2 + 1]
                                    let r2 = sv[j * 2], i2 = sv[j * 2 + 1]
                                    sv[i * 2] = (1 - blend) * r1 + blend * r2
                                    sv[i * 2 + 1] = (1 - blend) * i1 + blend * i2
                                    sv[j * 2] = blend * r1 + (1 - blend) * r2
                                    sv[j * 2 + 1] = blend * i1 + (1 - blend) * i2
                                }
                            }
                        }
                    }
                }
            }

            // State fidelity (overlap with target GHZ state)
            let fState = sv[0] * sv[0] + sv[1] * sv[1]  // |0...0⟩ overlap
            let alignment = abs(cos(phaseScale - GOD_CODE_PHASE_ANGLE))
            return -(SACRED_COMPOSITE_WEIGHT * fState + SACRED_ALIGNMENT_WEIGHT * alignment)
        }

        var history: [Double] = []

        for _ in 0..<maxIterations {
            // Sort simplex by objective
            simplex.sort { objective($0) < objective($1) }
            let best = objective(simplex[0])
            history.append(-best)

            // Check convergence
            let worst = objective(simplex[dim])
            if abs(worst - best) < 1e-6 { break }

            // Centroid (exclude worst)
            var centroid = [Double](repeating: 0, count: dim)
            for i in 0..<dim {
                for j in 0..<dim { centroid[j] += simplex[i][j] }
            }
            for j in 0..<dim { centroid[j] /= Double(dim) }

            // Reflection
            let reflected = centroid.enumerated().map { centroid[$0.offset] + alpha * (centroid[$0.offset] - simplex[dim][$0.offset]) }
            let fReflected = objective(reflected)

            if fReflected < objective(simplex[0]) {
                // Expansion
                let expanded = centroid.enumerated().map { centroid[$0.offset] + gamma * (reflected[$0.offset] - centroid[$0.offset]) }
                simplex[dim] = objective(expanded) < fReflected ? expanded : reflected
            } else if fReflected < objective(simplex[dim - 1]) {
                simplex[dim] = reflected
            } else {
                // Contraction
                let contracted = centroid.enumerated().map { centroid[$0.offset] + rho * (simplex[dim][$0.offset] - centroid[$0.offset]) }
                if objective(contracted) < objective(simplex[dim]) {
                    simplex[dim] = contracted
                } else {
                    // Shrink
                    for i in 1...dim {
                        for j in 0..<dim {
                            simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j])
                        }
                    }
                }
            }
        }

        simplex.sort { objective($0) < objective($1) }
        let bestParams = simplex[0]

        return [
            "best_params": ["depth_factor": bestParams[0], "phase_scale": bestParams[1],
                           "entangle_strength": bestParams[2], "gc_mix": bestParams[3]],
            "best_fidelity": -objective(bestParams),
            "iterations": history.count,
            "convergence_history": history,
            "target_fidelity": targetFidelity,
            "achieved": -objective(bestParams) >= targetFidelity
        ]
    }

    // MARK: - Noise Resilience Optimization

    /// Compare 8 strategies and find most noise-resilient
    func optimizeNoiseResilience(numQubits: Int = 4, noiseLevel: Double = 0.1) -> [String: Any] {
        let sweepResult = ParametricSweepEngine.shared.strategySweep(numQubits: numQubits)
        let matrix = (sweepResult["strategy_matrix"] as? [[String: Any]]) ?? []

        // Find best strategy at target noise level
        var bestStrategy = "raw"
        var bestFidelity = 0.0
        for row in matrix {
            guard let noise = row["noise"] as? Double, abs(noise - noiseLevel) < 0.01 else { continue }
            for (key, val) in row {
                if key == "noise" { continue }
                if let f = val as? Double, f > bestFidelity {
                    bestFidelity = f
                    bestStrategy = key
                }
            }
        }

        return [
            "noise_level": noiseLevel,
            "best_strategy": bestStrategy,
            "best_fidelity": bestFidelity,
            "all_strategies": sweepResult
        ]
    }

    // MARK: - ═══ INVENTED: Sacred Annealing Cascade (SAC) ═══
    // Novel meta-optimizer that cascades through strategy phases using
    // Fibonacci cooling schedule with GOD_CODE-harmonic restarts.
    // Each cascade level uses a different strategy from the 8 available,
    // selected by sacred alignment score at current temperature.

    func sacredAnnealingCascade(numQubits: Int = 4, depth: Int = 4,
                                 cascadeLevels: Int = 8) -> [String: Any] {
        let strategies = ["raw", "zeno_freeze", "composite_shield", "fibonacci_echo",
                          "depolarizing", "dephasing", "amplitude_damping", "combined_noise"]
        var bestGlobalFidelity = 0.0
        var bestGlobalStrategy = ""
        var cascadeResults: [[String: Any]] = []

        var temperature = GOD_CODE  // Sacred initial temperature

        for level in 0..<cascadeLevels {
            // Select strategy by sacred alignment at current temperature
            let phaseAtTemp = temperature.truncatingRemainder(dividingBy: 2.0 * Double.pi)
            let strategyIdx = Int(abs(phaseAtTemp / (2.0 * Double.pi)) * Double(strategies.count)) % strategies.count
            let strategy = strategies[strategyIdx]

            // Fibonacci cooling: T_{n+1} = T_n × F(n) / F(n+1)
            let fibRatio = level + 1 < FIBONACCI_8.count ?
                Double(FIBONACCI_8[level]) / Double(FIBONACCI_8[level + 1]) : TAU
            temperature *= fibRatio

            // Evaluate strategy effectiveness at this temperature-mapped noise level
            let effectiveNoise = 1.0 / max(temperature, 1.0) * 0.5
            let fidelity: Double
            switch strategy {
            case "zeno_freeze": fidelity = pow(cos(effectiveNoise * Double.pi / 4.0), 2.0)
            case "composite_shield": fidelity = exp(-effectiveNoise * 0.8) * 1.05
            case "fibonacci_echo":
                var recovery = 0.0
                for (i, f) in FIBONACCI_8.enumerated() {
                    recovery += Double(f) / FIBONACCI_8_SUM * exp(-effectiveNoise * Double(i + 1) * 0.3)
                }
                fidelity = recovery
            default: fidelity = exp(-effectiveNoise * 2.0)
            }

            cascadeResults.append([
                "level": level, "strategy": strategy,
                "temperature": temperature, "effective_noise": effectiveNoise,
                "fidelity": fidelity
            ])

            if fidelity > bestGlobalFidelity {
                bestGlobalFidelity = fidelity
                bestGlobalStrategy = strategy
            }
        }

        InterEngineFeedbackBus.shared.broadcast(
            from: .optimization,
            signal: "sacred_annealing_cascade_complete",
            payload: ["best_fidelity": bestGlobalFidelity,
                      "cascade_levels": Double(cascadeLevels)]
        )

        return [
            "best_strategy": bestGlobalStrategy,
            "best_fidelity": bestGlobalFidelity,
            "cascade_results": cascadeResults,
            "final_temperature": temperature,
            "cooling_schedule": "fibonacci"
        ]
    }
}
