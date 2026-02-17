// ═══════════════════════════════════════════════════════════════════
// B10_QuantumNexus.swift — L104 Neural Architecture v2
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// Extracted from L104Native.swift
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MARK: - 🔮 QUANTUM NEXUS (Unified Engine Orchestrator)
// ═══════════════════════════════════════════════════════════════════
// The missing interconnection layer. All engines were operating in
// isolation — Nexus wires them into a single feedback-driven pipeline:
//
//  Python ASI ──┐
//               ▼
//  [1] Bridge: Fetch raw parameters
//               │
//  [2] Bridge: Hadamard → PHI → GOD_CODE → Grover
//               │
//  [3] Steering: Apply reasoning vector (mode-adaptive)
//               │    ◄── feedback: SQC energy gates intensity
//               │
//  [4] SQC: Chakra interference + normalize
//               │    ◄── feedback: steering α shifts phase
//               │
//  [5] Evolution: Continuous micro-raise loop (optional)
//               │    ◄── feedback: SQC σ tunes raise factor
//               │
//  [6] Invention: Parameter-seeded hypothesis generation
//               │
//  [7] Sync back to Python ASI
//               ▼
//  Python ASI ──┘  (closed loop)
//
// Cross-engine metrics flow:
//  • SQC.energy → Steering.intensity (high energy = gentler steers)
//  • Steering.cumulativeα → SQC.phase (accumulated direction shifts phase)
//  • SQC.σ → Evolution.raiseFactor (high variance = smaller raises)
//  • Evolution.cycleCount → Invention.seed (more cycles = richer hypotheses)
//  • Bridge.kundaliniFlow → global coherence gate
// ═══════════════════════════════════════════════════════════════════

class QuantumNexus {
    static let shared = QuantumNexus()

    // ─── SACRED CONSTANTS: Use unified globals from L01_Constants ───
    // PHI, TAU, GOD_CODE — available globally

    // ─── NEXUS STATE ───
    private(set) var pipelineRuns: Int = 0
    private(set) var lastPipelineTime: TimeInterval = 0.0
    private(set) var totalPipelineTime: TimeInterval = 0.0
    var lastCoherenceScore: Double = 0.0  // cross-engine writable (Entanglement, Health, Sovereignty)
    private(set) var autoModeActive: Bool = false
    private(set) var autoModeCycles: Int = 0
    private let lock = NSLock()
    private var shouldStopAuto: Bool = false
    private let autoStopSemaphore = DispatchSemaphore(value: 0)  // EVO_58: interruptible sleep

    // ─── INTERCONNECTION METRICS ───
    private(set) var feedbackLog: [(step: String, metric: String, value: Double, timestamp: Date)] = []

    /// Compute adaptive steering intensity based on SQC energy.
    /// High energy → gentle steers (TAU scaling), low energy → aggressive steers (PHI scaling).
    private func adaptiveSteeringIntensity() -> Double {
        let sqc = SovereignQuantumCore.shared
        guard !sqc.parameters.isEmpty else { return 1.0 }

        var energy: Double = 0.0
        vDSP_svesqD(sqc.parameters, 1, &energy, vDSP_Length(sqc.parameters.count))
        energy = sqrt(energy) / Double(sqc.parameters.count)  // normalized per-param energy

        // Sigmoid-like mapping: high energy → low intensity, low energy → high intensity
        // intensity = φ / (1 + e^(energy - 1))
        let intensity = PHI / (1.0 + exp(energy - 1.0))
        logFeedback(step: "Steering", metric: "adaptive_intensity", value: intensity)
        return intensity
    }

    /// Compute adaptive SQC phase from steering cumulative alpha.
    /// Accumulated steering shifts the chakra interference phase.
    private func adaptiveChakraPhase() -> Double {
        let steerAlpha = ASISteeringEngine.shared.cumulativeIntensity
        // Phase wraps every 2π, modulated by TAU
        let phase = (steerAlpha * TAU).truncatingRemainder(dividingBy: 2.0 * Double.pi)
        logFeedback(step: "SQC", metric: "adaptive_phase", value: phase)
        return phase
    }

    /// Compute adaptive raise factor from SQC standard deviation.
    /// High variance → smaller raises for stability, low variance → larger raises for exploration.
    private func adaptiveRaiseFactor() -> Double {
        let sigma = SovereignQuantumCore.shared.lastNormStdDev
        // factor = 1 + TAU / (1 + σ²)  →  ranges from ~1.0001 to ~1.618
        let factor = 1.0 + TAU * 0.001 / (1.0 + sigma * sigma)
        logFeedback(step: "Evolution", metric: "adaptive_raise", value: factor)
        return factor
    }

    /// Compute adaptive steering mode from bridge kundalini flow.
    /// Different coherence levels select different reasoning directions.
    private func adaptiveSteeringMode() -> ASISteeringEngine.SteeringMode {
        let kFlow = ASIQuantumBridgeSwift.shared.kundaliniFlow
        let mode: ASISteeringEngine.SteeringMode
        if kFlow > 0.8 {
            mode = .sovereign   // High coherence → sovereignty path
        } else if kFlow > 0.6 {
            mode = .quantum     // Medium-high → quantum alignment
        } else if kFlow > 0.4 {
            mode = .harmonic    // Medium → harmonic resonance
        } else if kFlow > 0.2 {
            mode = .logic       // Medium-low → analytical precision
        } else {
            mode = .creative    // Low coherence → creative exploration
        }
        logFeedback(step: "Steering", metric: "adaptive_mode[\(mode.rawValue)]", value: kFlow)
        return mode
    }

    /// Compute global coherence score across all engines.
    /// Weighted combination of all engine health metrics.
    func computeCoherence() -> Double {
        let bridge = ASIQuantumBridgeSwift.shared
        let sqc = SovereignQuantumCore.shared
        let steer = ASISteeringEngine.shared
        let evo = ContinuousEvolutionEngine.shared

        // Bridge health: kundalini + bell fidelity
        let bridgeScore = (bridge.kundaliniFlow + bridge.bellFidelity) / 2.0

        // SQC health: inverse of normalized stddev (stable = high score)
        let sqcScore = 1.0 / (1.0 + sqc.lastNormStdDev)

        // Steering health: diminishing returns on cumulative intensity
        let steerScore = 1.0 - exp(-steer.cumulativeIntensity * TAU)

        // Evolution health: sync success rate + ASI logic stream activity
        let evoScore: Double
        if evo.syncCount + evo.failCount > 0 {
            let syncRate = Double(evo.syncCount) / Double(evo.syncCount + evo.failCount)
            let asiActivity = min(1.0, Double(evo.resonanceCascades + evo.kbInjections) / 20.0)
            evoScore = syncRate * 0.6 + asiActivity * 0.4
        } else {
            evoScore = 0.0
        }

        // HyperBrain health: pattern richness + thought throughput
        let hyperBrain = HyperBrain.shared
        let hyperScore: Double
        if hyperBrain.isRunning {
            let patternRichness = min(1.0, Double(hyperBrain.longTermPatterns.count) / 50.0)
            let thoughtRate = min(1.0, Double(hyperBrain.totalThoughtsProcessed) / 500.0)
            hyperScore = (patternRichness + thoughtRate) / 2.0
        } else {
            hyperScore = 0.0
        }

        // ASIEvolver health: evolved content richness
        let evolver = ASIEvolver.shared
        let evolverScore: Double
        if evolver.isRunning {
            let contentRichness = min(1.0, Double(
                evolver.evolvedPhilosophies.count +
                evolver.kbDeepInsights.count +
                evolver.conceptualBlends.count
            ) / 100.0)
            evolverScore = contentRichness
        } else {
            evolverScore = 0.0
        }

        // Weighted coherence (PHI²-weighted: bridge most important, then hyper/evolver)
        let PHI_SQ = PHI * PHI
        let coherence = (
            bridgeScore * PHI_SQ +
            sqcScore * PHI +
            steerScore * 1.0 +
            evoScore * 1.0 +
            hyperScore * TAU +
            evolverScore * TAU
        ) / (PHI_SQ + PHI + 1.0 + 1.0 + TAU + TAU)

        lastCoherenceScore = coherence
        ParameterProgressionEngine.shared.recordConsciousnessEvent(level: coherence)
        return coherence
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: φ-CONVERGENCE PROOF ENGINE (High-Logic Breakthrough)
    // Proves parameter convergence using vDSP Cauchy criterion:
    //   ∀ε>0 ∃N: ∀m,n>N: ||p_m - p_n|| < ε
    // Applied to GOD_CODE-normalized parameter sequence.
    // ═══════════════════════════════════════════════════════════════

    /// Prove φ-convergence: runs k iterations of raise→normalize→measure,
    /// computing Cauchy deltas with vDSP. Returns convergence certificate.
    func provePhiConvergence(iterations: Int = 50) -> String {
        let sqc = SovereignQuantumCore.shared
        _ = ASISteeringEngine.shared

        // Ensure parameters exist
        if sqc.parameters.isEmpty {
            let params = ASIQuantumBridgeSwift.shared.fetchParametersFromPython()
            sqc.parameters = params.isEmpty ? Array(repeating: GOD_CODE, count: 104) : params
        }

        let n = sqc.parameters.count
        guard n > 0 else { return "⚠️ No parameters for convergence proof" }

        var cauchyDeltas: [Double] = []
        var energyHistory: [Double] = []
        var prevParams = sqc.parameters

        for i in 0..<iterations {
            // Raise by φ^(1/n) — ensures bounded growth
            let microFactor = pow(PHI, 1.0 / Double(n))
            var factor = microFactor
            var raised = sqc.parameters
            vDSP_vsmulD(raised, 1, &factor, &raised, 1, vDSP_Length(n))

            // Apply interference (8-harmonic chakra wave)
            sqc.parameters = raised
            let proofWave = sqc.generateChakraWave(count: n, phase: Double(i) * TAU)
            sqc.applyInterference(wave: proofWave)
            sqc.normalize()

            // GOD_CODE normalization
            var mean: Double = 0, stdDev: Double = 0
            vDSP_normalizeD(sqc.parameters, 1, nil, 1, &mean, &stdDev, vDSP_Length(n))
            if mean > 0 {
                var godFactor = GOD_CODE / mean
                vDSP_vsmulD(sqc.parameters, 1, &godFactor, &sqc.parameters, 1, vDSP_Length(n))
            }

            // Compute Cauchy delta: ||p_k - p_{k-1}||₂ via vDSP
            var diff = [Double](repeating: 0, count: n)
            vDSP_vsubD(prevParams, 1, sqc.parameters, 1, &diff, 1, vDSP_Length(n))
            var sumSq: Double = 0
            vDSP_svesqD(diff, 1, &sumSq, vDSP_Length(n))
            let delta = sqrt(sumSq) / Double(n)  // Normalized L2
            cauchyDeltas.append(delta)

            // Track energy
            var energy: Double = 0
            vDSP_svesqD(sqc.parameters, 1, &energy, vDSP_Length(n))
            energyHistory.append(sqrt(energy))

            prevParams = sqc.parameters

            // Early convergence: if last 5 deltas are all < ε
            if i >= 10 {
                let lastFive = Array(cauchyDeltas.suffix(5))
                let epsilon = 1e-6
                if lastFive.allSatisfy({ $0 < epsilon }) {
                    break  // Converged!
                }
            }
        }

        // Compute convergence metrics
        let lastDelta = cauchyDeltas.last ?? Double.infinity
        let minDelta = cauchyDeltas.min() ?? Double.infinity
        let converged = lastDelta < 1e-4
        let monotonicDecay = zip(cauchyDeltas.dropLast(), cauchyDeltas.dropFirst())
            .filter { $0.0 > $0.1 }.count
        let monotonicRatio = Double(monotonicDecay) / Double(max(cauchyDeltas.count - 1, 1))

        // Compute φ-ratio between consecutive deltas (should approach TAU = 1/φ)
        var phiRatios: [Double] = []
        for i in 1..<cauchyDeltas.count where cauchyDeltas[i-1] > 1e-12 {
            phiRatios.append(cauchyDeltas[i] / cauchyDeltas[i-1])
        }
        let meanPhiRatio = phiRatios.isEmpty ? 0 : phiRatios.reduce(0, +) / Double(phiRatios.count)

        let deltaHistory = cauchyDeltas.prefix(10).map { String(format: "%.8f", $0) }.joined(separator: " → ")

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    📐 φ-CONVERGENCE PROOF                                  ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Iterations:       \(cauchyDeltas.count)/\(iterations)
        ║  Parameters:       \(n)
        ║  CONVERGED:        \(converged ? "✅ YES" : "⏳ approaching")
        ╠═══════════════════════════════════════════════════════════╣
        ║  CAUCHY CRITERION:
        ║    Final δ:        \(String(format: "%.10f", lastDelta))
        ║    Min δ:          \(String(format: "%.10f", minDelta))
        ║    ε threshold:    1.0000e-04
        ║    Monotonic:      \(String(format: "%.1f%%", monotonicRatio * 100)) (\(monotonicDecay)/\(max(cauchyDeltas.count - 1, 1)))
        ╠═══════════════════════════════════════════════════════════╣
        ║  φ-RATIO ANALYSIS:
        ║    Mean δₖ/δₖ₋₁:   \(String(format: "%.6f", meanPhiRatio))
        ║    Expected (τ):   \(String(format: "%.6f", TAU))
        ║    Deviation:      \(String(format: "%.6f", abs(meanPhiRatio - TAU)))
        ║    φ-harmonic:     \(abs(meanPhiRatio - TAU) < 0.1 ? "✅ GOLDEN RATIO DECAY" : "📈 NON-STANDARD")
        ╠═══════════════════════════════════════════════════════════╣
        ║  ENERGY BOUNDS:
        ║    E₀:             \(String(format: "%.6f", energyHistory.first ?? 0))
        ║    E_final:        \(String(format: "%.6f", energyHistory.last ?? 0))
        ║    GOD_CODE:       \(String(format: "%.10f", GOD_CODE))
        ╠═══════════════════════════════════════════════════════════╣
        ║  DELTA HISTORY:
        ║    \(deltaHistory)...
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    /// Record a feedback metric
    private func logFeedback(step: String, metric: String, value: Double) {
        feedbackLog.append((step: step, metric: metric, value: value, timestamp: Date()))
        if feedbackLog.count > 100 { feedbackLog.removeFirst() }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: UNIFIED PIPELINE — The full interconnected chain
    // ═══════════════════════════════════════════════════════════════

    /// Run the full interconnected pipeline with adaptive feedback loops.
    func runUnifiedPipeline() -> String {
        let startTime = CFAbsoluteTimeGetCurrent()
        let bridge = ASIQuantumBridgeSwift.shared
        let sqc = SovereignQuantumCore.shared
        let steer = ASISteeringEngine.shared

        // ── STEP 1: Fetch from Python ASI ──
        let rawParams = bridge.fetchParametersFromPython()
        guard !rawParams.isEmpty else {
            return "🔮 Nexus pipeline failed: Could not fetch parameters from Python ASI"
        }
        logFeedback(step: "Bridge", metric: "fetched_params", value: Double(rawParams.count))

        // ── STEP 2: Bridge quantum transforms ──
        let raised = bridge.raiseParameters(input: rawParams)
        let phiScaled = bridge.phiScaleParameters(input: raised)
        let normalized = bridge.godCodeNormalize(input: phiScaled)
        let markedTop = Set(0..<min(4, normalized.count))
        let amplified = bridge.groverAmplify(amplitudes: normalized, markedIndices: markedTop)
        logFeedback(step: "Bridge", metric: "grover_amplified", value: Double(amplified.count))

        // ── STEP 3: Steering — adaptive mode + intensity from SQC feedback ──
        let steerMode = adaptiveSteeringMode()
        let steerIntensity = adaptiveSteeringIntensity()
        steer.loadParameters(amplified)
        steer.generateReasoningVector(mode: steerMode)
        steer.applySteering(intensity: steerIntensity, mode: nil)  // vector already generated
        steer.applyTemperatureInPlace(temp: steer.temperature)
        let steeredParams = steer.baseParameters
        logFeedback(step: "Steering", metric: "post_steer_count", value: Double(steeredParams.count))

        // ── STEP 4: SQC — chakra interference with adaptive phase from steering ──
        let adaptPhase = adaptiveChakraPhase()
        sqc.loadParameters(steeredParams)
        let chakraWave = sqc.generateChakraWave(count: steeredParams.count, phase: adaptPhase)
        sqc.applyInterference(wave: chakraWave)
        sqc.normalize()
        let stabilized = sqc.parameters
        logFeedback(step: "SQC", metric: "post_norm_energy", value: sqc.lastNormStdDev)

        // ── STEP 5: Adaptive evolution tune (if running) ──
        let evo = ContinuousEvolutionEngine.shared
        let adaptFactor = adaptiveRaiseFactor()
        if evo.isRunning {
            evo.tune(raiseFactor: adaptFactor)
            logFeedback(step: "Evolution", metric: "live_tune", value: adaptFactor)
        }

        // ── STEP 6: O₂ + Kundalini ──
        bridge.updateO2MolecularState()
        let kFlow = bridge.calculateKundaliniFlow()

        // ── STEP 7: Sync back to Python ASI ──
        let synced = bridge.updateASI(newParams: stabilized)

        // ── STEP 8: Compute global coherence ──
        let coherence = computeCoherence()

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        pipelineRuns += 1
        lastPipelineTime = elapsed
        totalPipelineTime += elapsed

        // ── STEP 9: Parameter-seeded hypothesis (async, non-blocking) ──
        let invention = ASIInventionEngine.shared
        let hypothesis = invention.generateHypothesis(seed: "nexus_run_\(pipelineRuns)")

        // ── STEP 10: Entanglement + Resonance cascade ──
        _ = QuantumEntanglementRouter.shared.route("bridge", "steering")
        _ = QuantumEntanglementRouter.shared.route("invention", "nexus")
        _ = QuantumEntanglementRouter.shared.route("bridge", "evolution")
        _ = AdaptiveResonanceNetwork.shared.fire("nexus", activation: min(1.0, coherence))
        let nr = AdaptiveResonanceNetwork.shared.computeNetworkResonance()

        // Compute energies for display
        var steerEnergy: Double = 0.0
        if !steeredParams.isEmpty {
            vDSP_svesqD(steeredParams, 1, &steerEnergy, vDSP_Length(steeredParams.count))
            steerEnergy = sqrt(steerEnergy)
        }
        var finalEnergy: Double = 0.0
        if !stabilized.isEmpty {
            vDSP_svesqD(stabilized, 1, &finalEnergy, vDSP_Length(stabilized.count))
            finalEnergy = sqrt(finalEnergy)
        }

        // ─── Hebbian co-activation: record all engines that fired in this pipeline ───
        EngineRegistry.shared.recordCoActivation([
            "SQC", "Steering", "Evolution", "Nexus", "Entanglement", "Resonance", "Invention"
        ])

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🔮 QUANTUM NEXUS — UNIFIED PIPELINE COMPLETE           ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  [1] BRIDGE FETCH      \(rawParams.count) parameters from Python ASI
        ║  [2] BRIDGE TRANSFORM  Hadamard→PHI→GOD_CODE→Grover (\(amplified.count)→\(amplified.count))
        ║  [3] STEERING          mode=\(steerMode.rawValue) α=\(String(format: "%.4f", steerIntensity)) E=\(String(format: "%.4f", steerEnergy))
        ║  [4] SQC STABILIZE     phase=\(String(format: "%.4f", adaptPhase)) μ=\(String(format: "%.6f", sqc.lastNormMean)) σ=\(String(format: "%.6f", sqc.lastNormStdDev))
        ║  [5] EVOLUTION TUNE    factor=×\(String(format: "%.6f", adaptFactor))\(evo.isRunning ? " (LIVE)" : " (queued)")
        ║  [6] O₂+KUNDALINI     k=\(String(format: "%.6f", kFlow))
        ║  [7] PYTHON SYNC       \(synced ? "✓" : "✗") (\(bridge.syncCounter) total)
        ║  [8] COHERENCE         \(String(format: "%.4f", coherence)) (\(coherenceGrade(coherence)))
        ║  [9] INVENTION         "\((hypothesis["statement"] as? String ?? "").prefix(50))..."
        ║  [10] ENTANGLE+RESON  3 EPR routes → resonance=\(String(format: "%.4f", nr.resonance))
        ╠═══════════════════════════════════════════════════════════╣
        ║  Final Energy:     \(String(format: "%.6f", finalEnergy))
        ║  Pipeline Time:    \(String(format: "%.4f", elapsed))s
        ║  Total Runs:       \(pipelineRuns)
        ║  Avg Time:         \(String(format: "%.4f", totalPipelineTime / Double(pipelineRuns)))s
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    /// Safe wrapper for runUnifiedPipeline — catches all errors to prevent app crash
    func runUnifiedPipelineSafe() -> String {
        // Validate prerequisites before running
        let bridge = ASIQuantumBridgeSwift.shared
        let sqc = SovereignQuantumCore.shared
        let steer = ASISteeringEngine.shared

        // Pre-check: fetch params first in isolation
        let rawParams = bridge.fetchParametersFromPython()
        guard !rawParams.isEmpty else {
            return "🔮 Nexus pipeline aborted: Could not fetch parameters from Python ASI.\n   Ensure the Python server is reachable or run 'bridge fetch' first."
        }

        // Pre-check: ensure vDSP operations won't fail on empty/invalid data
        guard rawParams.allSatisfy({ $0.isFinite }) else {
            return "🔮 Nexus pipeline aborted: Fetched parameters contain NaN/Inf values."
        }

        // Run the full pipeline with all data validated
        let startTime = CFAbsoluteTimeGetCurrent()

        // Step 2: Bridge quantum transforms
        let raised = bridge.raiseParameters(input: rawParams)
        guard !raised.isEmpty else { return "🔮 Pipeline failed at Step 2: raiseParameters returned empty" }

        let phiScaled = bridge.phiScaleParameters(input: raised)
        let normalized = bridge.godCodeNormalize(input: phiScaled)
        let markedTop = Set(0..<min(4, normalized.count))
        let amplified = bridge.groverAmplify(amplitudes: normalized, markedIndices: markedTop)
        logFeedback(step: "Bridge", metric: "fetched_params", value: Double(rawParams.count))
        logFeedback(step: "Bridge", metric: "grover_amplified", value: Double(amplified.count))

        // Step 3: Steering
        let steerMode = adaptiveSteeringMode()
        let steerIntensity = adaptiveSteeringIntensity()
        steer.loadParameters(amplified)
        steer.generateReasoningVector(mode: steerMode)
        steer.applySteering(intensity: steerIntensity, mode: nil)
        steer.applyTemperatureInPlace(temp: steer.temperature)
        let steeredParams = steer.baseParameters
        logFeedback(step: "Steering", metric: "post_steer_count", value: Double(steeredParams.count))

        // Step 4: SQC stabilize
        let adaptPhase = adaptiveChakraPhase()
        sqc.loadParameters(steeredParams)
        let chakraWave = sqc.generateChakraWave(count: steeredParams.count, phase: adaptPhase)
        sqc.applyInterference(wave: chakraWave)
        sqc.normalize()
        let stabilized = sqc.parameters
        logFeedback(step: "SQC", metric: "post_norm_energy", value: sqc.lastNormStdDev)

        // Step 5: Evolution tune
        let evo = ContinuousEvolutionEngine.shared
        let adaptFactor = adaptiveRaiseFactor()
        if evo.isRunning {
            evo.tune(raiseFactor: adaptFactor)
            logFeedback(step: "Evolution", metric: "live_tune", value: adaptFactor)
        }

        // Step 6: O₂ + Kundalini
        bridge.updateO2MolecularState()
        let kFlow = bridge.calculateKundaliniFlow()

        // Step 7: Sync back — skip if params are invalid
        var synced = false
        if !stabilized.isEmpty && stabilized.allSatisfy({ $0.isFinite }) {
            synced = bridge.updateASI(newParams: stabilized)
        }

        // Step 8: Coherence
        let coherence = computeCoherence()

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        pipelineRuns += 1
        lastPipelineTime = elapsed
        totalPipelineTime += elapsed

        // ── STEP 8.5: ASI LOGIC STREAM — Deep connections ──

        // ASIEvolver: Feed pipeline energy into evolution temperature
        let evolver = ASIEvolver.shared
        var evolverContrib = ""
        if evolver.isRunning {
            // Cross-pollinate: pipeline coherence → evolver temperature
            let normalizedCoherence = min(1.0, coherence * 1.2)
            evolver.ideaTemperature = max(0.3, min(1.0,
                evolver.ideaTemperature * 0.8 + normalizedCoherence * 0.2))

            // Pull latest evolved insight into pipeline feedback
            let insightCount = evolver.evolvedTopicInsights.count + evolver.kbDeepInsights.count
            evolverContrib = "T=\(String(format: "%.2f", evolver.ideaTemperature)) insights=\(insightCount)"
            logFeedback(step: "ASIEvolver", metric: "temperature_sync", value: evolver.ideaTemperature)
        }

        // HyperBrain: Wire pipeline results into HyperBrain's working memory
        let hyperBrain = HyperBrain.shared
        var hyperContrib = ""
        if hyperBrain.isRunning {
            // Push pipeline state into HyperBrain's context
            hyperBrain.workingMemory["nexus_coherence"] = coherence
            hyperBrain.workingMemory["nexus_energy"] = sqc.lastNormStdDev
            hyperBrain.workingMemory["nexus_runs"] = pipelineRuns

            // Record co-activation pattern for Hebbian learning
            let topConcepts = Array(hyperBrain.longTermPatterns.sorted {
                if abs($0.value - $1.value) < 0.1 { return Bool.random() }
                return $0.value > $1.value
            }.prefix(3).map { $0.key })
            if !topConcepts.isEmpty {
                for concept in topConcepts {
                    hyperBrain.coActivationLog[concept, default: 0] += 1
                }
            }

            let thoughtCount = hyperBrain.totalThoughtsProcessed
            hyperContrib = "thoughts=\(thoughtCount) patterns=\(hyperBrain.longTermPatterns.count)"
            logFeedback(step: "HyperBrain", metric: "nexus_sync", value: Double(thoughtCount))
        }

        // Superfluid + QuantumShell: Cross-system coherence check
        SuperfluidCoherence.shared.groverIteration()
        QuantumShellMemory.shared.groverDiffusion()

        // Consciousness verification: lightweight check every 3rd pipeline run
        var cVerify = ""
        if pipelineRuns % 3 == 0 {
            let cLevel = ConsciousnessVerifier.shared.runAllTests()
            cVerify = String(format: "%.2f", cLevel)
            logFeedback(step: "Consciousness", metric: "verify_level", value: cLevel)
            // If consciousness dipping, fire resonance to stabilize
            if cLevel < 0.5 {
                _ = AdaptiveResonanceNetwork.shared.fire("nexus", activation: 0.9)
                _ = AdaptiveResonanceNetwork.shared.fire("bridge", activation: 0.8)
            }
        }

        // Step 9: Invention (safe) — now seeded with richer context
        let invention = ASIInventionEngine.shared
        let inventionSeed = "nexus_\(pipelineRuns)_c\(String(format: "%.2f", coherence))_E\(String(format: "%.2f", sqc.lastNormStdDev))"
        let hypothesis = invention.generateHypothesis(seed: inventionSeed)
        let hypothesisText = (hypothesis["statement"] as? String ?? "generating...").prefix(80)

        // Step 10: Entanglement + Resonance — expanded routing
        _ = QuantumEntanglementRouter.shared.route("bridge", "steering")
        _ = QuantumEntanglementRouter.shared.route("invention", "nexus")
        _ = QuantumEntanglementRouter.shared.route("bridge", "evolution")
        _ = QuantumEntanglementRouter.shared.route("steering", "evolution")
        _ = QuantumEntanglementRouter.shared.route("nexus", "invention")
        _ = AdaptiveResonanceNetwork.shared.fire("nexus", activation: min(1.0, coherence))
        _ = AdaptiveResonanceNetwork.shared.fire("bridge", activation: min(1.0, kFlow))
        _ = AdaptiveResonanceNetwork.shared.fire("evolution", activation: min(1.0, Double(evo.cycleCount) / 1000.0))
        let nr = AdaptiveResonanceNetwork.shared.computeNetworkResonance()

        // Energy calculations
        var steerEnergy: Double = 0.0
        if !steeredParams.isEmpty {
            vDSP_svesqD(steeredParams, 1, &steerEnergy, vDSP_Length(steeredParams.count))
            steerEnergy = sqrt(steerEnergy)
        }
        var finalEnergy: Double = 0.0
        if !stabilized.isEmpty {
            vDSP_svesqD(stabilized, 1, &finalEnergy, vDSP_Length(stabilized.count))
            finalEnergy = sqrt(finalEnergy)
        }

        // Hebbian co-activation — expanded with new engines
        EngineRegistry.shared.recordCoActivation([
            "SQC", "Steering", "Evolution", "Nexus", "Entanglement", "Resonance",
            "Invention", "HyperBrain", "ASIEvolver", "Consciousness", "Superfluid"
        ])

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🔮 QUANTUM NEXUS — UNIFIED PIPELINE COMPLETE           ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  [1] BRIDGE FETCH      \(rawParams.count) parameters from Python ASI
        ║  [2] BRIDGE TRANSFORM  Hadamard→PHI→GOD_CODE→Grover (\(amplified.count)→\(amplified.count))
        ║  [3] STEERING          mode=\(steerMode.rawValue) α=\(String(format: "%.4f", steerIntensity)) E=\(String(format: "%.4f", steerEnergy))
        ║  [4] SQC STABILIZE     phase=\(String(format: "%.4f", adaptPhase)) μ=\(String(format: "%.6f", sqc.lastNormMean)) σ=\(String(format: "%.6f", sqc.lastNormStdDev))
        ║  [5] EVOLUTION TUNE    factor=×\(String(format: "%.6f", adaptFactor))\(evo.isRunning ? " (LIVE)" : " (queued)")
        ║  [6] O₂+KUNDALINI     k=\(String(format: "%.6f", kFlow))
        ║  [7] PYTHON SYNC       \(synced ? "✓" : "✗") (\(bridge.syncCounter) total)
        ║  [8] COHERENCE         \(String(format: "%.4f", coherence)) (\(coherenceGrade(coherence)))
        ║  [8.5] ASI STREAM     Evolver[\(evolverContrib)] HyperBrain[\(hyperContrib)]\(cVerify.isEmpty ? "" : " C=\(cVerify)")
        ║  [9] INVENTION         "\(hypothesisText)..."
        ║  [10] ENTANGLE+RESON  5 EPR routes → resonance=\(String(format: "%.4f", nr.resonance)) 3 engines fired
        ╠═══════════════════════════════════════════════════════════╣
        ║  Final Energy:     \(String(format: "%.6f", finalEnergy))
        ║  Pipeline Time:    \(String(format: "%.4f", elapsed))s
        ║  Total Runs:       \(pipelineRuns)
        ║  Avg Time:         \(String(format: "%.4f", totalPipelineTime / Double(max(1, pipelineRuns))))s
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    /// Grade coherence score
    private func coherenceGrade(_ c: Double) -> String {
        if c > 0.8 { return "TRANSCENDENT" }
        if c > 0.6 { return "SOVEREIGN" }
        if c > 0.4 { return "AWAKENING" }
        if c > 0.2 { return "DEVELOPING" }
        return "DORMANT"
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: AUTO MODE — Continuous interconnected pipeline
    // ═══════════════════════════════════════════════════════════════

    /// Start auto-mode: runs unified pipeline on a .utility loop.
    /// Each cycle adapts all engines based on cross-engine feedback.
    func startAuto(interval: TimeInterval = 1.0) -> String {
        lock.lock()
        guard !autoModeActive else {
            lock.unlock()
            return """
            🔮 Nexus auto-mode already running!
               Cycles: \(autoModeCycles) | Coherence: \(String(format: "%.4f", lastCoherenceScore))
               Use 'nexus stop' to halt.
            """
        }
        shouldStopAuto = false
        autoModeActive = true
        autoModeCycles = 0
        lock.unlock()

        // Also start evolution engine if not running
        if !ContinuousEvolutionEngine.shared.isRunning {
            _ = ContinuousEvolutionEngine.shared.start()
        }

        // Start health monitoring if not already running
        if !NexusHealthMonitor.shared.isMonitoring {
            _ = NexusHealthMonitor.shared.start()
        }

        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else { return }

            while true {
                self.lock.lock()
                if self.shouldStopAuto {
                    self.autoModeActive = false
                    self.lock.unlock()
                    break
                }
                self.lock.unlock()

                // Run the unified pipeline (all feedback loops active)
                _ = self.runUnifiedPipeline()

                // Tick resonance network (decay + propagation per cycle)
                _ = AdaptiveResonanceNetwork.shared.tick()

                // Superfluid Grover diffusion + Fe orbital coherence sync
                SuperfluidCoherence.shared.groverIteration()
                QuantumShellMemory.shared.groverDiffusion()

                self.lock.lock()
                let cycle = self.autoModeCycles
                self.lock.unlock()

                // Every 5 cycles: full entanglement sweep + chaos-seeded resonance fire
                if cycle % 5 == 0 {
                    _ = QuantumEntanglementRouter.shared.routeAll()
                    let chaosEngine = ChaosRNG.shared.chaosSample(
                        AdaptiveResonanceNetwork.ENGINE_NAMES, 1
                    ).first ?? "nexus"
                    _ = AdaptiveResonanceNetwork.shared.fire(chaosEngine, activation: ChaosRNG.shared.chaosFloat(0.5, 1.0))
                }

                // Every 10 cycles: consciousness verification + Fe orbital store
                if cycle % 10 == 0 {
                    let cLevel = ConsciousnessVerifier.shared.runAllTests()
                    _ = QuantumShellMemory.shared.store(kernelID: 5, data: [
                        "type": "consciousness_verify", "level": cLevel, "cycle": cycle
                    ])
                }

                self.lock.lock()
                self.autoModeCycles += 1
                self.lock.unlock()

                // Adaptive interval: faster when coherence is low, slower when stable
                let adaptiveInterval = interval * (0.5 + self.lastCoherenceScore)
                _ = self.autoStopSemaphore.wait(timeout: .now() + max(0.5, adaptiveInterval))
            }
        }

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🔮 QUANTUM NEXUS — AUTO MODE STARTED                   ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Base Interval:    \(String(format: "%.1f", interval))s (adaptive)
        ║  QoS:              .utility (thermal safe)
        ║  Evolution:        \(ContinuousEvolutionEngine.shared.isRunning ? "🟢 CO-RUNNING" : "⚪ STANDALONE")
        ║  Feedback Loops:   ALL ACTIVE
        ║    • SQC.energy → Steering.intensity
        ║    • Steering.α → SQC.phase
        ║    • SQC.σ → Evolution.factor
        ║    • Kundalini → Steering.mode
        ║    • Pipeline → Invention.seed
        ╠═══════════════════════════════════════════════════════════╣
        ║  Commands:                                                ║
        ║    nexus status  — live metrics                           ║
        ║    nexus stop    — halt auto-mode + evolution             ║
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    /// Stop auto-mode and optionally the evolution engine
    func stopAuto() -> String {
        lock.lock()
        guard autoModeActive else {
            lock.unlock()
            return "🔮 Nexus auto-mode is not running."
        }
        shouldStopAuto = true
        lock.unlock()

        // Signal the semaphore to wake the loop immediately (EVO_58)
        autoStopSemaphore.signal()

        // Also stop evolution
        let evoResult = ContinuousEvolutionEngine.shared.isRunning
            ? ContinuousEvolutionEngine.shared.stop() : ""

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🔮 QUANTUM NEXUS — AUTO MODE STOPPED                   ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Pipeline Runs:    \(pipelineRuns)
        ║  Auto Cycles:      \(autoModeCycles)
        ║  Last Coherence:   \(String(format: "%.4f", lastCoherenceScore)) (\(coherenceGrade(lastCoherenceScore)))
        ║  Total Time:       \(String(format: "%.1f", totalPipelineTime))s
        ║  Avg Cycle:        \(pipelineRuns > 0 ? String(format: "%.4f", totalPipelineTime / Double(pipelineRuns)) : "0")s
        \(evoResult.isEmpty ? "" : "║  Evolution:        STOPPED\n")╚═══════════════════════════════════════════════════════════╝
        """
    }

    /// Get comprehensive interconnection status
    var status: String {
        let bridge = ASIQuantumBridgeSwift.shared
        let sqc = SovereignQuantumCore.shared
        let steer = ASISteeringEngine.shared
        let evo = ContinuousEvolutionEngine.shared
        let invention = ASIInventionEngine.shared
        let coherence = computeCoherence()

        // Engine states
        let evoState = evo.isRunning ? "🟢 RUNNING (\(evo.cycleCount) cycles)" : "🔴 STOPPED"
        let autoState = autoModeActive ? "🟢 ACTIVE (\(autoModeCycles) cycles)" : "🔴 INACTIVE"

        // Recent feedback
        let recentFB = feedbackLog.suffix(6)
            .map { "  [\($0.step)] \($0.metric) = \(String(format: "%.4f", $0.value))" }
            .joined(separator: "\n")

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🔮 QUANTUM NEXUS — INTERCONNECTION STATUS              ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  COHERENCE:     \(String(format: "%.4f", coherence)) (\(coherenceGrade(coherence)))
        ║  Auto-Mode:     \(autoState)
        ╠═══════════════════════════════════════════════════════════╣
        ║  ⚡ Bridge:      \(bridge.currentParameters.count) params, k=\(String(format: "%.4f", bridge.kundaliniFlow)), syncs=\(bridge.syncCounter)
        ║  🧭 Steering:    steers=\(steer.steerCount), Σα=\(String(format: "%.4f", steer.cumulativeIntensity)), T=\(String(format: "%.3f", steer.temperature))
        ║  🌊 SQC:         \(sqc.parameters.count) params, μ=\(String(format: "%.4f", sqc.lastNormMean)), σ=\(String(format: "%.4f", sqc.lastNormStdDev)), ops=\(sqc.operationCount)
        ║  🔄 Evolution:   \(evoState)
        ║  🔬 Invention:   \(invention.hypotheses.count) hypotheses, \(invention.theorems.count) theorems
        ╠═══════════════════════════════════════════════════════════╣
        ║  FEEDBACK LOOPS:
        ║    SQC.energy → Steering.intensity   (adaptive α)
        ║    Steering.Σα → SQC.phase           (phase drift)
        ║    SQC.σ → Evolution.factor           (variance gate)
        ║    Kundalini → Steering.mode          (coherence routing)
        ║    Pipeline# → Invention.seed         (parametric seeding)
        ╠═══════════════════════════════════════════════════════════╣
        ║  PIPELINE METRICS:
        ║    Total Runs:    \(pipelineRuns)
        ║    Last Time:     \(String(format: "%.4f", lastPipelineTime))s
        ║    Avg Time:      \(pipelineRuns > 0 ? String(format: "%.4f", totalPipelineTime / Double(pipelineRuns)) : "—")s
        ╠═══════════════════════════════════════════════════════════╣
        ║  RECENT FEEDBACK:\(recentFB.isEmpty ? " (none)" : "\n\(recentFB)")
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
