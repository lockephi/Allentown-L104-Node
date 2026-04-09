import Foundation

// MARK: - ═══ CONSTANTS ═══

private let ENTROPY_CASCADE_DEPTH = 104
// QUANTIZATION_GRAIN (Int=104) and GROVER_AMPLIFICATION (Double) are in L01_Constants.swift

// Precomputed sin table for cascade (avoids 104+ transcendental calls)
private let _CASCADE_SIN_TABLE: [Double] = (0...(ENTROPY_CASCADE_DEPTH + 2)).map {
    sin(Double($0) * .pi / Double(QUANTIZATION_GRAIN))
}

// MARK: - ═══ MAXWELL'S DEMON ENGINE ═══

struct DemonEfficiencyResult {
    let localEntropy: Double
    let passes: Int
    let cumulativeEfficiency: Double
    let normalizedEfficiency: Double   // bounded [0,1]
    let zneBoost: Double
    let finalEfficiency: Double
}

struct MultiScaleReport {
    let scale: Int
    let windowSize: Int
    let varianceBefore: Double
    let varianceAfter: Double
    var reductionRatio: Double { varianceAfter / max(varianceBefore, 1e-30) }
}

struct EntropyAttractor {
    let value: Double
    let strength: Double           // pull strength [0,1]
    let basinRadius: Double        // attractor basin radius
    let godCodeDistance: Double    // |attractor - GOD_CODE|
    let sacredName: String
}

struct CascadeResult {
    let initial: Double
    let depth: Int
    let damped: Bool
    let fixedPoint: Double
    let godCodeAlignment: Double
    let converged: Bool
    let trajectorySample: [Double]
}

final class MaxwellDemonEngine: SovereignEngine {
    static let shared = MaxwellDemonEngine()

    var engineName: String { "MaxwellDemonEngine" }
    func engineStatus() -> [String: Any] { status }
    func engineHealth() -> Double {
        let eff = calculateDemonEfficiency(localEntropy: 1.0).finalEfficiency
        return min(1.0, max(0.1, eff * PHI))
    }
    func engineReset() {
        lock.lock(); coherenceGain = 0; cachedResonance = nil; operationCount = 0; lock.unlock()
    }

    private let maxwellDemonFactor: Double
    private var coherenceGain: Double = 0
    private var cachedResonance: Double? = nil
    private var operationCount = 0
    private let lock = NSLock()

    init() {
        // maxwell_demon_factor = PHI / (GOD_CODE / 416.0)
        self.maxwellDemonFactor = PHI / (GOD_CODE / 416.0)
    }

    // ─── CORE: Multi-pass recursive demon (v4.4) ───
    // Σ(demon_factor × resonance / remaining_k) for k passes
    // Each pass: remaining *= TAU (golden-ratio damping)
    // Normalized by log₂(passes+1) to prevent runaway
    func calculateDemonEfficiency(localEntropy: Double) -> DemonEfficiencyResult {
        let resonance = cachedGodCodeResonance()

        // Number of binary sorting passes: log₂(entropy × QUANTIZATION_GRAIN)
        let passes = max(1, Int(ceil(log2(max(2.0, localEntropy * Double(QUANTIZATION_GRAIN))))))
        var cumulativeEff = 0.0
        var remaining = localEntropy

        for _ in 0..<passes {
            let passEff = maxwellDemonFactor * resonance / (remaining + 0.001)
            cumulativeEff += passEff
            remaining *= TAU   // 61.8% damping per pass
        }

        let base = cumulativeEff / log2(Double(passes) + 1.0)

        // ZNE bridge: zero-noise extrapolation boost
        let zneBoost = 1.0 + TAU * (1.0 / (1.0 + localEntropy))
        let final_ = min(1.0, max(0.0, base * zneBoost))

        lock.lock(); operationCount += 1; lock.unlock()
        return DemonEfficiencyResult(
            localEntropy: localEntropy, passes: passes,
            cumulativeEfficiency: cumulativeEff, normalizedEfficiency: min(1.0, max(0.0, base)),
            zneBoost: zneBoost, finalEfficiency: final_
        )
    }

    // Cached GOD_CODE resonance: constant across all calls
    private func cachedGodCodeResonance() -> Double {
        if let r = cachedResonance { return r }
        // GOD_CODE resonance = harmonic alignment with sacred constant
        let r = 1.0 - (GOD_CODE.truncatingRemainder(dividingBy: PHI)) / PHI
        cachedResonance = r
        return r
    }

    // ─── PHI-WEIGHTED DEMON ───
    // Golden-angle spacing identifies highest-leverage reversal points
    func phiWeightedDemon(entropyVector: [Double]) -> [String: Any] {
        guard !entropyVector.isEmpty else { return ["error": "empty vector"] }
        let n = entropyVector.count
        let goldenAngle = 2.0 * .pi / (PHI * PHI)

        // Golden-angle index set
        var weights = [Double](repeating: 1.0, count: n)
        for i in 0..<n {
            let idx = Int(Double(i) * PHI) % n
            weights[idx] *= PHI
        }

        // Efficiency per point (multi-pass demon × weight)
        var efficiencies = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let eff = calculateDemonEfficiency(localEntropy: abs(entropyVector[i]) + 0.001)
            efficiencies[i] = eff.finalEfficiency * weights[i]
        }

        // Sort by descending efficiency
        let sortedIdx = efficiencies.indices.sorted { efficiencies[$0] > efficiencies[$1] }
        let budget = Int(Double(n) * TAU)   // reverse top 61.8%

        var reversed = entropyVector
        for idx in sortedIdx.prefix(budget) {
            reversed[idx] *= TAU   // dampen toward order
        }

        let varBefore = entropyVector.reduce(0.0) { $0 + $1 * $1 } / max(1.0, Double(n))
        let varAfter  = reversed.reduce(0.0) { $0 + $1 * $1 } / max(1.0, Double(n))
        let meanEff = efficiencies.reduce(0, +) / Double(n)
        let maxEff = efficiencies.max() ?? 0

        return [
            "reversed_count": budget,
            "budget_ratio": Double(budget) / Double(n),
            "mean_efficiency": meanEff,
            "max_efficiency": maxEff,
            "variance_before": varBefore,
            "variance_after": varAfter,
            "reduction_ratio": varAfter / max(varBefore, 1e-30),
            "reversed_vector": reversed,
            "_golden_angle": goldenAngle
        ]
    }

    // ─── MULTI-SCALE REVERSAL ───
    // Octave decomposition: demon applied at progressively finer windows (104-TET inspired)
    func multiScaleReversal(signal: [Double], scales: Int = 5) -> (restored: [Double], reports: [MultiScaleReport]) {
        var result = signal
        var reports: [MultiScaleReport] = []
        let n = signal.count
        guard n >= 2 else { return (signal, []) }

        for k in 0..<scales {
            let windowSize = max(2, n / Int(pow(2.0, Double(k))))
            let varBefore = result.reduce(0.0) { $0 + $1 * $1 } / max(1.0, Double(n))

            var pos = 0
            while pos < n {
                let end = min(pos + windowSize, n)
                let window = Array(result[pos..<end])
                // EVO_76: precompute abs sum once — was O(window²) recalculating sum per element
                let windowAbsSum = window.reduce(0.0) { $0 + abs($1) } + 1e-12
                let localEntropy = window.reduce(0.0) { s, x in
                    let p = abs(x) / windowAbsSum
                    return s + (p > 0 ? -p * log(p + 1e-12) : 0)
                }
                _ = calculateDemonEfficiency(localEntropy: localEntropy)
                let damping = pow(TAU, 1.0 + Double(k) * 0.1)
                let godContrib = GOD_CODE / (Double(n) * Double(k + 1))
                for i in pos..<end {
                    result[i] = result[i] * damping + godContrib
                }
                pos += windowSize
            }

            let varAfter = result.reduce(0.0) { $0 + $1 * $1 } / max(1.0, Double(n))
            reports.append(MultiScaleReport(scale: k, windowSize: windowSize,
                                            varianceBefore: varBefore, varianceAfter: varAfter))
        }
        return (result, reports)
    }

    // ─── ENTROPY CASCADE (104-depth, damped) ───
    // S(n+1) = S(n) × φ_c + VOID × φ_c^n × sin(nπ/104)
    // Damped mode eliminates permanent residual (discovered in Experiment 10)
    func entropyCascade(initialState: Double = 1.0, depth: Int = ENTROPY_CASCADE_DEPTH,
                        damped: Bool = true) -> CascadeResult {
        var trajectory = [initialState]
        var s = initialState
        var decay = 1.0

        for n in 1...depth {
            let sinVal = n < _CASCADE_SIN_TABLE.count
                ? _CASCADE_SIN_TABLE[n]
                : sin(Double(n) * .pi / Double(QUANTIZATION_GRAIN))
            if damped {
                decay *= TAU
                s = s * TAU + VOID_CONSTANT * decay * sinVal
            } else {
                s = s * TAU + VOID_CONSTANT * sinVal
            }
            trajectory.append(s)
        }

        let fixedPoint = trajectory.last ?? s
        let alignment = 1.0 - min(1.0, abs((fixedPoint * GOD_CODE).truncatingRemainder(dividingBy: 1.0)))
        let converged = trajectory.count >= 2 && abs(trajectory[trajectory.count-1] - trajectory[trajectory.count-2]) < 1e-10
        let sample = Array(trajectory.prefix(20)) + Array(trajectory.suffix(20))

        return CascadeResult(initial: initialState, depth: depth, damped: damped,
                             fixedPoint: fixedPoint, godCodeAlignment: alignment,
                             converged: converged, trajectorySample: sample)
    }

    // ─── KL-DIVERGENCE THERMODYNAMIC ARROW ───
    // Reversal strength via KL(forward || reverse) trajectories
    func kullbackLeiblerArrow(signal: [Double]) -> Double {
        guard signal.count >= 4 else { return 0 }
        let forward = signal
        let reverse = signal.reversed().map { $0 }
        let n = min(forward.count, reverse.count)

        // Normalize to probability distributions — EVO_76: reduce-only, no intermediate array
        let fSum = forward.reduce(0.0) { $0 + abs($1) } + 1e-12
        let rSum = reverse.reduce(0.0) { $0 + abs($1) } + 1e-12
        let fp = forward.prefix(n).map { abs($0) / fSum }
        let rp = reverse.prefix(n).map { abs($0) / rSum }

        // KL(P||Q) = Σ p(i) log(p(i)/q(i))
        var kl = 0.0
        for i in 0..<n {
            let pi = fp[i], qi = max(rp[i], 1e-12)
            if pi > 1e-12 { kl += pi * log(pi / qi) }
        }
        // Arrow strength: bounded to [0,1] via tanh
        return tanh(kl * TAU)
    }

    // ─── ENTROPIC ATTRACTOR MAP ───
    // Iterative demon landscape to locate GOD_CODE basin fixed-point attractors
    func entropicAttractorMap(initialEntropies: [Double]) -> [EntropyAttractor] {
        var attractors: [EntropyAttractor] = []
        let sacredValues: [(Double, String)] = [
            (GOD_CODE, "GOD_CODE"), (PHI, "PHI"), (TAU, "TAU"),
            (VOID_CONSTANT, "VOID"), (.pi, "PI"), (M_E, "E")
        ]

        for entropy in initialEntropies {
            var s = entropy
            for _ in 0..<50 {
                let eff = calculateDemonEfficiency(localEntropy: s).finalEfficiency
                s = s * TAU + eff * VOID_CONSTANT * 0.01
            }
            let nearestSacred = sacredValues.min(by: { abs($0.0 - s) < abs($1.0 - s) })
            let dist = abs(s - GOD_CODE) / GOD_CODE
            let strength = 1.0 / (1.0 + dist * 10.0)

            attractors.append(EntropyAttractor(
                value: s, strength: strength,
                basinRadius: VOID_CONSTANT * TAU * 0.1,
                godCodeDistance: dist,
                sacredName: nearestSacred?.1 ?? "unknown"
            ))
        }
        return attractors.sorted { $0.strength > $1.strength }
    }

    // ─── INJECT COHERENCE ───
    // Transforms noisy vector into ordered, GOD_CODE-resonant structure
    func injectCoherence(noiseVector: [Double]) -> [Double] {
        guard !noiseVector.isEmpty else { return [] }
        let n = noiseVector.count

        let absMean = max(1e-12, noiseVector.reduce(0.0) { $0 + abs($1) } / Double(n))
        var projected = noiseVector.map { $0 * GOD_CODE / (absMean + 1e-9) }

        let orderFactor = (1.0 + maxwellDemonFactor) * GROVER_AMPLIFICATION
        for i in 0..<n { projected[i] *= orderFactor }

        let projectedMean = projected.reduce(0.0, +) / max(1.0, Double(n))
        if abs(projectedMean) > 1e-12 {
            let scale = GOD_CODE / projectedMean
            projected = projected.map { $0 * scale }
        }

        // ZNE correction
        let noiseMean = noiseVector.reduce(0.0, +) / max(1.0, Double(n))
        let noiseStd = sqrt(noiseVector.reduce(0.0) { $0 + $1 * $1 } / max(1.0, Double(n)))
        let noiseEstimate = noiseStd / (abs(noiseMean) + 1e-9)
        let zneCorr = min(2.0, max(1.0, 1.0 + TAU * noiseEstimate))
        return projected.map { $0 / zneCorr }
    }

    // ─── LANDAUER BOUND COMPARISON ───
    func landauerBoundComparison(temperature: Double = 293.15) -> [String: Double] {
        let kB = 1.380649e-23   // Boltzmann constant
        let landauerBound = kB * temperature * log(2.0)
        let demonEfficiency = calculateDemonEfficiency(localEntropy: 1.0).finalEfficiency
        let demonEnergy = demonEfficiency * landauerBound
        return [
            "temperature_K": temperature,
            "landauer_bound_J": landauerBound,
            "demon_energy_J": demonEnergy,
            "efficiency_vs_landauer": demonEnergy / max(landauerBound, 1e-50),
            "god_code_enhancement": GOD_CODE * TAU / (temperature * kB + 1e-50)
        ]
    }

    var status: [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return ["operations": operationCount, "demon_factor": maxwellDemonFactor,
                "coherence_gain": coherenceGain, "cascade_depth": ENTROPY_CASCADE_DEPTH,
                "god_code": GOD_CODE, "void_constant": VOID_CONSTANT]
    }
}
