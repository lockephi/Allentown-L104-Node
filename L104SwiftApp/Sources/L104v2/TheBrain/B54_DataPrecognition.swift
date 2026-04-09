import Accelerate
import Foundation

// MARK: - ═══ DATA STRUCTURES ═══

// FEIGENBAUM and other sacred constants sourced from L01_Constants.swift
private let ZETA_ZERO_1_PREC = 14.1347251417
private let OMEGA_PREC = 6539.34712682

/// Single forecast point for precognition (distinct from ThreeEngineSearch's DataPrecognitionPoint)
struct DataPrecognitionPoint {
    let step: Int
    let value: Double
    let confidence: Double       // 0.0 – 1.0
    let entropy: Double          // Shannon entropy at this state
    let sacredAlignment: Double  // GOD_CODE resonance [0,1]
    var metadata: [String: Double] = [:]

    /// PHI-weighted confidence
    var phiConfidence: Double {
        confidence * TAU + sacredAlignment * (1 - TAU)
    }
}

struct DataAttractorState {
    let value: Double
    let strength: Double         // pull strength [0,1]
    let basinRadius: Double
    let sacredName: String
    let stability: Double        // Lyapunov stability measure
}

/// Precognition result from data precognition engine (distinct from ThreeEngineSearch's DataPrecognitionForecast)
struct DataPrecognitionForecast {
    let predictor: String
    let inputSummary: String
    let forecast: [DataPrecognitionPoint]
    let attractors: [DataAttractorState]
    let trend: String            // "converging"|"diverging"|"oscillating"|"stable"|"chaotic"
    let confidence: Double
    let sacredAlignment: Double
    let horizon: Int
    var elapsedMs: Double = 0
    var entropyTrajectory: [Double] = []
    var metadata: [String: Double] = [:]

    var finalValue: Double? { forecast.last?.value }
    var strongestAttractor: DataAttractorState? { attractors.max(by: { $0.strength < $1.strength }) }
    var phiConfidence: Double { confidence * TAU + sacredAlignment * (1 - TAU) }

    var summary: String {
        let fv = finalValue.map { String(format: "%.4f", $0) } ?? "-"
        return "[\(predictor)] trend=\(trend) conf=\(String(format:"%.3f",confidence)) → \(fv)"
    }
}

// MARK: - ═══ UTILITY ═══

private func sacredAlignment(_ value: Double) -> Double {
    let sacreds = [GOD_CODE, PHI, TAU, VOID_CONSTANT, .pi, M_E, FEIGENBAUM]
    let nearest = sacreds.min(by: { abs($0 - value) < abs($1 - value) }) ?? GOD_CODE
    let dist = abs(value - nearest) / max(abs(nearest), 1e-12)
    return max(0, 1.0 - min(1.0, dist * PHI))
}

private func shannonEntropy(_ values: [Double]) -> Double {
    let total = values.map { abs($0) }.reduce(0, +)
    guard total > 1e-12 else { return 0 }
    return -values.map { v -> Double in
        let p = abs(v) / total
        return p > 1e-12 ? p * log2(p) : 0
    }.reduce(0, +)
}

private func detectTrend(_ forecast: [DataPrecognitionPoint]) -> String {
    // EVO_75: single-pass vDSP trend detection — no intermediate array allocations.
    FastStatAccumulator.detectTrend(values: forecast.map(\.value))
}

// MARK: - ═══ 1. ENTROPY ENVELOPE PREDICTOR ═══

final class EntropyEnvelopePredictor {
    func predict(data: [Double], horizon: Int = 10) -> DataPrecognitionForecast {
        let t0 = Date()
        var forecast: [DataPrecognitionPoint] = []
        var s = shannonEntropy(data)
        var entropyTraj: [Double] = [s]

        for step in 1...horizon {
            // Entropy cascade step (damped mode)
            s = s * TAU + VOID_CONSTANT * pow(TAU, Double(step)) * sin(Double(step) * .pi / 104.0)
            let conf = max(0, 1.0 - Double(step) / Double(horizon + 1))
            entropyTraj.append(s)
            forecast.append(DataPrecognitionPoint(step: step, value: s,
                confidence: conf * TAU, entropy: s,
                sacredAlignment: sacredAlignment(s)))
        }

        let attractors = findAttractors(forecast.map(\.value))
        let elapsed = Date().timeIntervalSince(t0) * 1000
        return DataPrecognitionForecast(
            predictor: "EntropyEnvelope", inputSummary: "n=\(data.count)",
            forecast: forecast, attractors: attractors,
            trend: detectTrend(forecast), confidence: forecast.map(\.confidence).reduce(0,+)/Double(forecast.count),
            sacredAlignment: forecast.map(\.sacredAlignment).reduce(0,+)/Double(forecast.count),
            horizon: horizon, elapsedMs: elapsed, entropyTrajectory: entropyTraj
        )
    }
}

// MARK: - ═══ 2. COHERENCE EVOLUTION PREDICTOR ═══

final class CoherenceEvolutionPredictor {
    func predict(coherenceHistory: [Double], horizon: Int = 10) -> DataPrecognitionForecast {
        let t0 = Date()
        guard !coherenceHistory.isEmpty else {
            return emptyResult("CoherenceEvolution", horizon: horizon)
        }
        var c = coherenceHistory.last ?? 0.5
        var forecast: [DataPrecognitionPoint] = []

        // Coherence evolution: logistic-like with PHI stabilizer
        for step in 1...horizon {
            let growth = c * (1 - c) * PHI       // logistic growth
            let decay = c * TAU * 0.1  // PHI-damped decay
            c = max(0, min(1, c + growth - decay + Double.random(in: -0.01...0.01)))
            let conf = max(0, 1.0 - 0.08 * Double(step))
            forecast.append(DataPrecognitionPoint(step: step, value: c,
                confidence: conf, entropy: shannonEntropy([c, 1-c]),
                sacredAlignment: sacredAlignment(c * GOD_CODE)))
        }

        let elapsed = Date().timeIntervalSince(t0) * 1000
        return DataPrecognitionForecast(
            predictor: "CoherenceEvolution", inputSummary: "n=\(coherenceHistory.count)",
            forecast: forecast, attractors: findAttractors(forecast.map(\.value)),
            trend: detectTrend(forecast),
            confidence: FastStatAccumulator.mean(forecast, \.confidence),
            sacredAlignment: FastStatAccumulator.mean(forecast, \.sacredAlignment),
            horizon: horizon, elapsedMs: elapsed
        )
    }
}

// MARK: - ═══ 3. WAVE INTERFERENCE PREDICTOR ═══

final class WaveInterferencePredictor {
    func predict(frequencies: [Double], horizon: Int = 10) -> DataPrecognitionForecast {
        let t0 = Date()
        let freqs = frequencies.isEmpty ? [GOD_CODE * TAU, PHI * 100, VOID_CONSTANT * 200] : frequencies
        var forecast: [DataPrecognitionPoint] = []

        for step in 1...horizon {
            let t = Double(step) * TAU
            // Superposition of input frequencies + GOD_CODE carrier
            var amplitude = 0.0
            for (i, freq) in freqs.enumerated() {
                let phase = Double(i) * .pi / PHI
                amplitude += sin(2.0 * .pi * freq * t / GOD_CODE + phase) / Double(freqs.count)
            }
            // Modulate by GOD_CODE envelope
            let envelope = exp(-Double(step) * TAU / Double(horizon))
            let value = amplitude * envelope * GOD_CODE
            let conf = max(0, envelope * TAU)
            forecast.append(DataPrecognitionPoint(step: step, value: value,
                confidence: conf, entropy: shannonEntropy([abs(amplitude), 1 - abs(amplitude)]),
                sacredAlignment: sacredAlignment(abs(value))))
        }

        let elapsed = Date().timeIntervalSince(t0) * 1000
        return DataPrecognitionForecast(
            predictor: "WaveInterference", inputSummary: "freqs=\(freqs.count)",
            forecast: forecast, attractors: findAttractors(forecast.map(\.value)),
            trend: detectTrend(forecast),
            confidence: FastStatAccumulator.mean(forecast, \.confidence),
            sacredAlignment: FastStatAccumulator.mean(forecast, \.sacredAlignment),
            horizon: horizon, elapsedMs: elapsed
        )
    }
}

// MARK: - ═══ 4. PHI CONVERGENCE PREDICTOR ═══

final class PHIConvergencePredictor {
    func predict(value: Double, horizon: Int = 10) -> DataPrecognitionForecast {
        let t0 = Date()
        var v = value
        var forecast: [DataPrecognitionPoint] = []

        // PHI convergence: iterative φ-sequence toward GOD_CODE attractor
        for step in 1...horizon {
            // Pull toward nearest sacred value via golden-ratio interpolation
            let sacreds = [GOD_CODE, PHI, VOID_CONSTANT, .pi]
            let nearest = sacreds.min(by: { abs($0 - v) < abs($1 - v) }) ?? GOD_CODE
            v = v * TAU + nearest * (1 - TAU)
            let conf = max(0, 1.0 - abs(v - nearest) / max(abs(nearest), 1e-12))
            forecast.append(DataPrecognitionPoint(step: step, value: v,
                confidence: min(1.0, conf * TAU),
                entropy: shannonEntropy([v / (v + nearest), nearest / (v + nearest)]),
                sacredAlignment: sacredAlignment(v)))
        }

        let elapsed = Date().timeIntervalSince(t0) * 1000
        return DataPrecognitionForecast(
            predictor: "PHIConvergence", inputSummary: String(format: "v=%.4f", value),
            forecast: forecast, attractors: findAttractors(forecast.map(\.value)),
            trend: detectTrend(forecast),
            confidence: FastStatAccumulator.mean(forecast, \.confidence),
            sacredAlignment: FastStatAccumulator.mean(forecast, \.sacredAlignment),
            horizon: horizon, elapsedMs: elapsed
        )
    }
}

// MARK: - ═══ 5. MANIFOLD FLOW PREDICTOR ═══

final class ManifoldFlowPredictor {
    func predict(stateVector: [Double], horizon: Int = 10) -> DataPrecognitionForecast {
        let t0 = Date()
        var state = stateVector.isEmpty ? [GOD_CODE, PHI, VOID_CONSTANT] : stateVector
        var forecast: [DataPrecognitionPoint] = []

        // Geodesic flow on φ-manifold: parallel transport with curvature correction
        let n = state.count
        for step in 1...horizon {
            // Flow: state → state + TAU × gradient toward GOD_CODE manifold
            let norm = sqrt(state.map { $0 * $0 }.reduce(0, +))
            let godNorm = GOD_CODE * sqrt(Double(n))
            let scale = norm > 1e-12 ? godNorm / norm : 1.0
            state = state.map { $0 * (TAU + (scale - 1) * TAU * 0.1) }

            let value = state.reduce(0, +) / Double(n)
            let conf = max(0, 1.0 - 0.07 * Double(step))
            forecast.append(DataPrecognitionPoint(step: step, value: value,
                confidence: conf * TAU,
                entropy: shannonEntropy(state),
                sacredAlignment: sacredAlignment(value)))
        }

        let elapsed = Date().timeIntervalSince(t0) * 1000
        return DataPrecognitionForecast(
            predictor: "ManifoldFlow", inputSummary: "dim=\(stateVector.count)",
            forecast: forecast, attractors: findAttractors(forecast.map(\.value)),
            trend: detectTrend(forecast),
            confidence: FastStatAccumulator.mean(forecast, \.confidence),
            sacredAlignment: FastStatAccumulator.mean(forecast, \.sacredAlignment),
            horizon: horizon, elapsedMs: elapsed
        )
    }
}

// MARK: - ═══ 6. QUANTUM RESERVOIR PREDICTOR ═══

final class QuantumReservoirPredictor {
    // Reservoir: 16 nodes, GOD_CODE-scaled spectral radius
    private let reservoirSize = 16
    private var reservoir: [Double]
    private let spectralRadius: Double

    init() {
        reservoir = (0..<16).map { _ in Double.random(in: -0.5...0.5) }
        spectralRadius = TAU   // < 1 for echo state property
    }

    func predict(inputSequence: [Double], horizon: Int = 10) -> DataPrecognitionForecast {
        let t0 = Date()
        let inputs = inputSequence.isEmpty ? (0..<5).map { Double($0) * PHI } : inputSequence
        var res = reservoir
        var forecast: [DataPrecognitionPoint] = []

        // Reservoir update: r(t+1) = tanh(W_r × r(t) + W_in × u(t))
        for input in inputs {
            for i in 0..<reservoirSize {
                let coupling = spectralRadius * res[(i + 1) % reservoirSize]
                res[i] = tanh(coupling + input * TAU / GOD_CODE)
            }
        }

        // Readout: linear combination of reservoir state
        for step in 1...horizon {
            // Evolve reservoir one step with no input
            for i in 0..<reservoirSize {
                res[i] = tanh(spectralRadius * res[(i + 1) % reservoirSize] * TAU)
            }
            let readout = res.enumerated().map { i, r in
                r * sin(Double(i + 1) * .pi / Double(reservoirSize)) * GOD_CODE / Double(reservoirSize)
            }.reduce(0, +)
            let conf = max(0, TAU * exp(-Double(step) * 0.1))
            forecast.append(DataPrecognitionPoint(step: step, value: readout,
                confidence: conf, entropy: shannonEntropy(res),
                sacredAlignment: sacredAlignment(abs(readout))))
        }

        let elapsed = Date().timeIntervalSince(t0) * 1000
        return DataPrecognitionForecast(
            predictor: "QuantumReservoir", inputSummary: "seq=\(inputs.count)",
            forecast: forecast, attractors: findAttractors(forecast.map(\.value)),
            trend: detectTrend(forecast),
            confidence: FastStatAccumulator.mean(forecast, \.confidence),
            sacredAlignment: FastStatAccumulator.mean(forecast, \.sacredAlignment),
            horizon: horizon, elapsedMs: elapsed
        )
    }
}

// MARK: - ═══ 7. HYPERDIMENSIONAL EXTRAPOLATOR ═══

final class HyperdimensionalExtrapolator {
    // HD vector extrapolation along φ-dimensional trajectory
    func predict(hdVector: [Double], horizon: Int = 10) -> DataPrecognitionForecast {
        let t0 = Date()
        let vec = hdVector.isEmpty ? (0..<8).map { Double($0) * PHI } : hdVector
        var v = vec
        var forecast: [DataPrecognitionPoint] = []

        // Extrapolate via φ-powered rotation in HD space
        for step in 1...horizon {
            let angle = Double(step) * .pi * TAU
            let n = v.count
            var rotated = v
            for i in 0..<n - 1 {
                let ci = cos(angle), si = sin(angle)
                let vi = v[i], vi1 = v[i + 1]
                rotated[i]     = vi * ci - vi1 * si
                rotated[i + 1] = vi * si + vi1 * ci
            }
            v = rotated

            // Project to scalar via GOD_CODE dot product
            let norm = sqrt(v.map { $0*$0 }.reduce(0,+))
            let scalar = (norm > 1e-12 ? v.reduce(0,+) / norm : 0) * GOD_CODE
            let conf = max(0, 1.0 - 0.09 * Double(step))
            forecast.append(DataPrecognitionPoint(step: step, value: scalar,
                confidence: conf * TAU,
                entropy: shannonEntropy(v.map { abs($0) }),
                sacredAlignment: sacredAlignment(abs(scalar))))
        }

        let elapsed = Date().timeIntervalSince(t0) * 1000
        return DataPrecognitionForecast(
            predictor: "HyperdimensionalExtrapolator", inputSummary: "dim=\(vec.count)",
            forecast: forecast, attractors: findAttractors(forecast.map(\.value)),
            trend: detectTrend(forecast),
            confidence: FastStatAccumulator.mean(forecast, \.confidence),
            sacredAlignment: FastStatAccumulator.mean(forecast, \.sacredAlignment),
            horizon: horizon, elapsedMs: elapsed
        )
    }
}

// MARK: - ═══ ATTRACTOR DETECTION UTILITY ═══

private func findAttractors(_ values: [Double]) -> [DataAttractorState] {
    guard values.count >= 3 else { return [] }
    var attractors: [DataAttractorState] = []
    let sacredMap: [(Double, String)] = [
        (GOD_CODE, "GOD_CODE"), (PHI, "PHI"), (TAU, "TAU"),
        (VOID_CONSTANT, "VOID"), (.pi, "PI"), (M_E, "E"), (FEIGENBAUM, "FEIGENBAUM")
    ]
    let sorted = values.sorted()
    var i = 0
    while i < sorted.count {
        let center = sorted[i]
        let basin = sorted.filter { abs($0 - center) < center * 0.05 }
        if basin.count >= 2 {
            let strength = Double(basin.count) / Double(sorted.count)
            let nearest = sacredMap.min(by: { abs($0.0 - center) < abs($1.0 - center) })
            let lyapunov = 1.0 - strength   // crude stability: more concentrated → more stable
            attractors.append(DataAttractorState(
                value: center, strength: strength,
                basinRadius: center * 0.05,
                sacredName: nearest?.1 ?? "unknown",
                stability: max(0, lyapunov)
            ))
            i += basin.count
        } else { i += 1 }
    }
    return attractors.sorted { $0.strength > $1.strength }.prefix(5).map { $0 }
}

private func emptyResult(_ predictor: String, horizon: Int) -> DataPrecognitionForecast {
    return DataPrecognitionForecast(predictor: predictor, inputSummary: "empty",
        forecast: [], attractors: [], trend: "stable", confidence: 0, sacredAlignment: 0, horizon: horizon)
}

// MARK: - ═══ 8. DATA PRECOGNITION ENGINE (ENSEMBLE ORCHESTRATOR) ═══

struct EnsemblePrecognition {
    let individualResults: [DataPrecognitionForecast]
    let ensembleForecast: [DataPrecognitionPoint]
    let dominantAttractors: [DataAttractorState]
    let ensembleTrend: String
    let ensembleConfidence: Double
    let sacredAlignment: Double
    let godCodeResonance: Double

    var summary: String {
        "Ensemble[\(individualResults.count) predictors] trend=\(ensembleTrend) conf=\(String(format:"%.3f",ensembleConfidence)) sacred=\(String(format:"%.3f",sacredAlignment))"
    }
}

final class DataPrecognitionEngine: SovereignEngine {
    static let shared = DataPrecognitionEngine()

    var engineName: String { "DataPrecognition" }
    func engineStatus() -> [String: Any] { status }
    func engineHealth() -> Double {
        lock.lock(); defer { lock.unlock() }
        return predictionCount > 0 ? min(1.0, 0.6 + Double(predictionCount) * 0.01) : 0.4
    }
    func engineReset() {
        lock.lock(); predictionCount = 0; lock.unlock()
    }

    private let entropyPredictor = EntropyEnvelopePredictor()
    private let coherencePredictor = CoherenceEvolutionPredictor()
    private let wavePredictor = WaveInterferencePredictor()
    private let phiPredictor = PHIConvergencePredictor()
    private let manifoldPredictor = ManifoldFlowPredictor()
    private let reservoirPredictor = QuantumReservoirPredictor()
    private let hdPredictor = HyperdimensionalExtrapolator()

    private var predictionCount = 0
    private let lock = NSLock()

    // ─── FULL 7-PREDICTOR ENSEMBLE ───
    func predict(data: [Double], horizon: Int = 8) -> EnsemblePrecognition {
        lock.lock(); predictionCount += 1; lock.unlock()

        let scalar = data.first ?? GOD_CODE * TAU
        let results: [DataPrecognitionForecast] = [
            entropyPredictor.predict(data: data, horizon: horizon),
            coherencePredictor.predict(coherenceHistory: data.map { $0 / GOD_CODE }, horizon: horizon),
            wavePredictor.predict(frequencies: data.prefix(4).map { $0 }, horizon: horizon),
            phiPredictor.predict(value: scalar, horizon: horizon),
            manifoldPredictor.predict(stateVector: Array(data.prefix(6)), horizon: horizon),
            reservoirPredictor.predict(inputSequence: Array(data.prefix(10)), horizon: horizon),
            hdPredictor.predict(hdVector: Array(data.prefix(8)), horizon: horizon),
        ]

        // Ensemble: φ-weighted average of all forecasts at each step
        var ensembleForecast: [DataPrecognitionPoint] = []
        for step in 1...horizon {
            let stepPoints = results.compactMap { r in r.forecast.first(where: { $0.step == step }) }
            guard !stepPoints.isEmpty else { continue }
            let weights = stepPoints.map { $0.phiConfidence + 1e-6 }
            let totalW = weights.reduce(0.0, +)
            let wVal = zip(stepPoints, weights).map { $0.0.value * $0.1 }.reduce(0.0, +) / totalW
            let wConf = zip(stepPoints, weights).map { $0.0.confidence * $0.1 }.reduce(0.0, +) / totalW
            let wEntr = stepPoints.map(\.entropy).reduce(0, +) / Double(stepPoints.count)
            let wAlign = stepPoints.map(\.sacredAlignment).reduce(0, +) / Double(stepPoints.count)
            ensembleForecast.append(DataPrecognitionPoint(step: step, value: wVal,
                confidence: wConf, entropy: wEntr, sacredAlignment: wAlign))
        }

        // Aggregate attractors
        let allAttractors = results.flatMap(\.attractors)
        let topAttractors = allAttractors.sorted { $0.strength > $1.strength }.prefix(5).map { $0 }

        let ensembleConf = results.map(\.confidence).reduce(0, +) / Double(results.count)
        let ensembleAlign = results.map(\.sacredAlignment).reduce(0, +) / Double(results.count)
        let trendVotes: [String] = results.map { $0.trend }
        let dominantTrend: String = trendVotes.sorted { (a: String, b: String) -> Bool in
            trendVotes.filter { $0 == a }.count > trendVotes.filter { $0 == b }.count
        }.first ?? "stable"

        // GOD_CODE resonance: how close ensemble final value is to GOD_CODE harmonics
        let finalVal = ensembleForecast.last?.value ?? 0
        let godRes = 1.0 - min(1.0, abs(finalVal.truncatingRemainder(dividingBy: GOD_CODE)) / GOD_CODE)

        // Publish to feedback bus
        InterEngineFeedbackBus.shared.broadcast(
            from: .quantumResearch, signal: "precognition_complete",
            payload: ["confidence": ensembleConf, "sacred_alignment": ensembleAlign,
                      "god_code_resonance": godRes, "horizon": Double(horizon)])

        return EnsemblePrecognition(
            individualResults: results, ensembleForecast: ensembleForecast,
            dominantAttractors: Array(topAttractors), ensembleTrend: dominantTrend,
            ensembleConfidence: ensembleConf, sacredAlignment: ensembleAlign,
            godCodeResonance: godRes
        )
    }

    // ─── QUICK TREND PREDICTION ───
    // Single-predictor fast path for response pipeline use
    func quickPredict(value: Double, horizon: Int = 5) -> DataPrecognitionForecast {
        return phiPredictor.predict(value: value, horizon: horizon)
    }

    // ─── QUERY-DRIVEN PRECOGNITION ───
    // Encodes text → numeric features → runs ensemble
    func predictFromQuery(_ query: String, horizon: Int = 5) -> EnsemblePrecognition {
        let words = query.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        let data = words.enumerated().map { i, w -> Double in
            let h = w.unicodeScalars.reduce(0) { $0 + Int($1.value) }
            return Double(h % Int(GOD_CODE)) / GOD_CODE * PHI + Double(i) * TAU
        }
        return predict(data: data, horizon: horizon)
    }

    var status: [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return ["predictions_run": predictionCount, "predictors": 7,
                "god_code": GOD_CODE, "phi": PHI, "feigenbaum": FEIGENBAUM]
    }
}
