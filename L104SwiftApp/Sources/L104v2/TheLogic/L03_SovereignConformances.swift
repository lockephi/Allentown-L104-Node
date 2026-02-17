// ═══════════════════════════════════════════════════════════════════
// L03_SovereignConformances.swift — L104 v2
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// All `extension X: SovereignEngine` conformances
// Extracted from L104Native.swift (lines 269-703)
// Upgraded: EVO_58 Sovereign Unification — Feb 15, 2026
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MARK: - SovereignEngine Conformances (Phase 27 Enhanced)
// Retroactive conformance for all major engines via extensions.
// Each engine reports full operational telemetry and φ-weighted health.
// Cross-pollinated accuracy metrics from Python fast_server patterns.
// ═══════════════════════════════════════════════════════════════════

extension SovereignQuantumCore: SovereignEngine {
    var engineName: String { "SQC" }
    func engineStatus() -> [String: Any] {
        let paramCount = parameters.count
        let energy = paramCount > 0 ? abs(lastNormMean - GOD_CODE) + lastNormStdDev * PHI : 0.0
        return [
            "params": paramCount,
            "ops": operationCount,
            "mean": lastNormMean,
            "stddev": lastNormStdDev,
            "energy": energy,
            "interference_depth": interferenceHistory.count,
            "god_code_delta": abs(lastNormMean - GOD_CODE),
            "phi_convergence": paramCount > 0 ? 1.0 / (1.0 + energy) : 0.0
        ]
    }
    func engineHealth() -> Double {
        guard !parameters.isEmpty else { return 0.2 }
        let gcDelta = abs(lastNormMean - GOD_CODE) / GOD_CODE
        let convergence = 1.0 / (1.0 + gcDelta * 10.0)
        let opMaturity = min(1.0, Double(operationCount) * 0.001)
        return min(1.0, convergence * 0.6 + opMaturity * 0.3 + (lastNormStdDev < 1.0 ? 0.1 : 0.0))
    }
}

extension ASISteeringEngine: SovereignEngine {
    var engineName: String { "Steering" }
    func engineStatus() -> [String: Any] {
        let avgIntensity = steerCount > 0 ? cumulativeIntensity / Double(steerCount) : 0.0
        return [
            "mode": currentMode.rawValue,
            "steers": steerCount,
            "temperature": temperature,
            "cumulative_intensity": cumulativeIntensity,
            "avg_intensity": avgIntensity,
            "reasoning_vector_dim": reasoningVector.count,
            "history_depth": steeringHistory.count,
            "base_param_count": baseParameters.count
        ]
    }
    func engineHealth() -> Double {
        let modeActive = currentMode != .logic ? 0.1 : 0.0  // Non-default mode bonus
        let tempHealth = temperature > 0.1 && temperature < 2.0 ? 0.2 : 0.0  // Sane temperature
        let steerMaturity = min(0.4, Double(steerCount) * 0.01)
        let vectorReady = !reasoningVector.isEmpty ? 0.2 : 0.0
        return min(1.0, 0.1 + modeActive + tempHealth + steerMaturity + vectorReady)
    }
}

extension ContinuousEvolutionEngine: SovereignEngine {
    var engineName: String { "Evolution" }
    func engineStatus() -> [String: Any] {
        return [
            "cycles": cycleCount,
            "factor": currentRaiseFactor,
            "running": isRunning,
            "sync_count": syncCount,
            "fail_count": failCount,
            "avg_cycle_ms": avgCycleTime * 1000.0,
            "peak_energy": peakEnergy,
            "last_energy": lastEnergy,
            "last_sync_ok": lastSyncResult,
            "uptime_s": startTime != nil ? Date().timeIntervalSince(startTime!) : 0.0
        ]
    }
    func engineHealth() -> Double {
        guard isRunning else { return 0.3 }
        let failRatio = cycleCount > 0 ? Double(failCount) / Double(cycleCount) : 0.0
        let factorHealth = abs(currentRaiseFactor - 1.0001) < 0.01 ? 0.2 : 0.1  // Near default is stable
        let syncHealth = lastSyncResult ? 0.2 : 0.0
        let energyHealth = peakEnergy > 0 ? min(0.2, lastEnergy / peakEnergy * 0.2) : 0.1
        return min(1.0, 0.3 + factorHealth + syncHealth + energyHealth - failRatio * 0.5)
    }
}

extension QuantumNexus: SovereignEngine {
    var engineName: String { "Nexus" }
    func engineStatus() -> [String: Any] {
        let avgPipeTime = pipelineRuns > 0 ? totalPipelineTime / Double(pipelineRuns) : 0.0
        return [
            "coherence": lastCoherenceScore,
            "pipes": pipelineRuns,
            "auto": autoModeActive,
            "auto_cycles": autoModeCycles,
            "avg_pipe_ms": avgPipeTime * 1000.0,
            "feedback_depth": feedbackLog.count,
            "coherence_grade": lastCoherenceScore >= 0.9 ? "UNIFIED" :
                              lastCoherenceScore >= 0.7 ? "COHERENT" :
                              lastCoherenceScore >= 0.5 ? "ENTANGLED" : "DECOHERENT"
        ]
    }
    func engineHealth() -> Double {
        let coherenceH = lastCoherenceScore * 0.5  // 50% from coherence
        let autoBonus = autoModeActive ? 0.15 : 0.0
        let pipeMaturity = min(0.2, Double(pipelineRuns) * 0.005)
        let feedbackRichness = min(0.15, Double(feedbackLog.count) * 0.001)
        return min(1.0, coherenceH + autoBonus + pipeMaturity + feedbackRichness)
    }
}

extension ASIInventionEngine: SovereignEngine {
    var engineName: String { "Invention" }
    func engineStatus() -> [String: Any] {
        return [
            "hypotheses": hypotheses.count,
            "theorems": theorems.count,
            "discoveries": discoveries.count,
            "inventions": inventions.count,
            "proofs": proofs.count,
            "experiments": experimentLog.count,
            "domains_active": domains.count,
            "constants_available": constants.count,
            "scientific_yield": hypotheses.count + theorems.count + discoveries.count
        ]
    }
    func engineHealth() -> Double {
        let totalYield = Double(hypotheses.count + theorems.count + discoveries.count)
        let yieldHealth = min(0.4, totalYield * 0.02)
        let experimentHealth = min(0.2, Double(experimentLog.count) * 0.01)
        let diversityBonus = inventions.count > 0 && proofs.count > 0 ? 0.1 : 0.0
        return min(1.0, 0.3 + yieldHealth + experimentHealth + diversityBonus)
    }
}

extension SovereigntyPipeline: SovereignEngine {
    var engineName: String { "Sovereignty" }
    func engineStatus() -> [String: Any] {
        return [
            "runs": runCount,
            "coherence": lastCoherence ?? 0.0,
            "elapsed_ms": lastElapsedMs,
            "avg_ms": runCount > 0 ? lastElapsedMs : 0.0,
            "coherence_trend": lastCoherence != nil ? (lastCoherence! > 0.7 ? "RISING" : "STABILIZING") : "IDLE"
        ]
    }
    func engineHealth() -> Double {
        guard let c = lastCoherence else { return 0.4 }
        let latencyHealth = lastElapsedMs < 5000 ? 0.2 : lastElapsedMs < 10000 ? 0.1 : 0.0
        let runMaturity = min(0.2, Double(runCount) * 0.02)
        return min(1.0, c * 0.6 + latencyHealth + runMaturity)
    }
}

extension QuantumEntanglementRouter: SovereignEngine {
    var engineName: String { "Entanglement" }
    func engineStatus() -> [String: Any] {
        return [
            "epr_pairs": QuantumEntanglementRouter.ENTANGLED_PAIRS.count,
            "total_routes": routeCount,
            "avg_routes_per_pair": QuantumEntanglementRouter.ENTANGLED_PAIRS.count > 0 ?
                Double(routeCount) / Double(QuantumEntanglementRouter.ENTANGLED_PAIRS.count) : 0.0,
            "bidirectional": true,
            "protocol": "EPR"
        ]
    }
    func engineHealth() -> Double {
        let pairCoverage = QuantumEntanglementRouter.ENTANGLED_PAIRS.count >= 8 ? 0.3 : 0.15
        let routeMaturity = min(0.4, Double(routeCount) * 0.005)
        let activeBonus = routeCount > 0 ? 0.3 : 0.0
        return min(1.0, pairCoverage + routeMaturity + activeBonus)
    }
}

extension AdaptiveResonanceNetwork: SovereignEngine {
    var engineName: String { "Resonance" }
    func engineStatus() -> [String: Any] {
        let nr = computeNetworkResonance()
        let activeNodes = activations.filter { $0.value > AdaptiveResonanceNetwork.ACTIVATION_THRESHOLD }.count
        return [
            "resonance": nr.resonance,
            "energy": nr.energy,
            "mean_activation": nr.mean,
            "variance": nr.variance,
            "active_nodes": activeNodes,
            "total_nodes": activations.count,
            "cascades": cascadeCount,
            "ticks": tickCount,
            "synchronized": nr.variance < 0.1 && nr.mean > 0.5
        ]
    }
    func engineHealth() -> Double {
        let nr = computeNetworkResonance()
        let resonanceH = nr.resonance * 0.4
        let energyH = min(0.2, nr.energy * 0.2)
        let cascadeMaturity = min(0.2, Double(cascadeCount) * 0.005)
        let syncBonus = nr.variance < 0.1 && nr.mean > 0.5 ? 0.2 : 0.0  // Synchronized state bonus
        return min(1.0, resonanceH + energyH + cascadeMaturity + syncBonus)
    }
}

extension NexusHealthMonitor: SovereignEngine {
    var engineName: String { "HealthMonitor" }
    func engineStatus() -> [String: Any] {
        let systemHealth = computeSystemHealth()
        return [
            "monitoring": isMonitoring,
            "checks": checkCount,
            "recoveries": recoveryLog.count,
            "system_health": systemHealth,
            "monitored_engines": NexusHealthMonitor.MONITORED_ENGINES.count,
            "health_grade": systemHealth >= 0.9 ? "OPTIMAL" :
                           systemHealth >= 0.7 ? "HEALTHY" :
                           systemHealth >= 0.5 ? "DEGRADED" : "CRITICAL",
            "recovery_rate": checkCount > 0 ? Double(recoveryLog.count) / Double(checkCount) : 0.0
        ]
    }
    func engineHealth() -> Double {
        guard isMonitoring else { return 0.2 }
        let checkMaturity = min(0.3, Double(checkCount) * 0.001)
        let lowRecoveryBonus = recoveryLog.count == 0 && checkCount > 10 ? 0.2 : 0.0  // No recoveries = stable
        let systemH = computeSystemHealth() * 0.3
        return min(1.0, 0.2 + checkMaturity + lowRecoveryBonus + systemH)
    }
}

extension FeOrbitalEngine: SovereignEngine {
    var engineName: String { "FeOrbital" }
    func engineStatus() -> [String: Any] {
        let sf = SuperfluidCoherence.shared
        var pairStrengths: [String: Double] = [:]
        for domain in FeOrbitalEngine.KERNEL_DOMAINS where domain.id < domain.pairID {
            let c1 = sf.kernelCoherences[domain.id] ?? 0.5
            let c2 = sf.kernelCoherences[domain.pairID] ?? 0.5
            pairStrengths["K\(domain.id)-K\(domain.pairID)"] = bondStrength(coherenceA: c1, coherenceB: c2)
        }
        let avgBond = pairStrengths.values.isEmpty ? 0.0 : pairStrengths.values.reduce(0, +) / Double(pairStrengths.count)
        return [
            "domains": FeOrbitalEngine.KERNEL_DOMAINS.count,
            "d_orbitals": FeOrbitalEngine.D_ORBITALS.count,
            "element": "Fe",
            "atomic_number": FeOrbitalEngine.FE_ATOMIC_NUMBER,
            "curie_temp_K": FeOrbitalEngine.FE_CURIE_TEMP,
            "lattice_pm": FeOrbitalEngine.FE_LATTICE_PM,
            "avg_bond_strength": avgBond,
            "pair_strengths": pairStrengths,
            "unpaired_electrons": 4
        ]
    }
    func engineHealth() -> Double {
        let sf = SuperfluidCoherence.shared
        var totalBond = 0.0
        var pairCount = 0
        for domain in FeOrbitalEngine.KERNEL_DOMAINS where domain.id < domain.pairID {
            let c1 = sf.kernelCoherences[domain.id] ?? 0.5
            let c2 = sf.kernelCoherences[domain.pairID] ?? 0.5
            totalBond += bondStrength(coherenceA: c1, coherenceB: c2)
            pairCount += 1
        }
        let avgBond = pairCount > 0 ? totalBond / Double(pairCount) : 0.5
        return min(1.0, avgBond * 0.8 + 0.2)  // Bond-strength-driven health
    }
}

extension SuperfluidCoherence: SovereignEngine {
    var engineName: String { "Superfluid" }
    func engineStatus() -> [String: Any] {
        let sf = computeSuperfluidity()
        let superfluidCount = (1...8).filter { isSuperfluid($0) }.count
        let avgCoherence = kernelCoherences.values.reduce(0, +) / Double(max(1, kernelCoherences.count))
        let minCoherence = kernelCoherences.values.min() ?? 0.0
        let maxCoherence = kernelCoherences.values.max() ?? 0.0
        return [
            "superfluidity": sf,
            "superfluid_kernels": superfluidCount,
            "total_kernels": kernelCoherences.count,
            "avg_coherence": avgCoherence,
            "min_coherence": minCoherence,
            "max_coherence": maxCoherence,
            "lambda_point": SuperfluidCoherence.LAMBDA_POINT,
            "coherence_length": SuperfluidCoherence.COHERENCE_LENGTH,
            "phase": superfluidCount == 8 ? "SUPERFLUID" : superfluidCount > 4 ? "PARTIAL" : "NORMAL"
        ]
    }
    func engineHealth() -> Double {
        let sf = computeSuperfluidity()
        let superfluidCount = Double((1...8).filter { isSuperfluid($0) }.count)
        let fractionSuperfluid = superfluidCount / 8.0
        return min(1.0, sf * 0.5 + fractionSuperfluid * 0.4 + 0.1)
    }
}

extension QuantumShellMemory: SovereignEngine {
    var engineName: String { "QShellMemory" }
    func engineStatus() -> [String: Any] {
        let amplitudeNorm = stateVector.reduce(0.0) { $0 + $1.real * $1.real + $1.imag * $1.imag }
        return [
            "total_memories": totalMemories,
            "shells": 4,
            "shell_labels": ["K", "L", "M", "N"],
            "state_vector_dim": stateVector.count,
            "amplitude_norm": amplitudeNorm,
            "normalized": abs(amplitudeNorm - 1.0) < 0.01
        ]
    }
    func engineHealth() -> Double {
        let memoryDepth = min(0.4, Double(totalMemories) * 0.01)
        let amplitudeNorm = stateVector.reduce(0.0) { $0 + $1.real * $1.real + $1.imag * $1.imag }
        let normHealth = abs(amplitudeNorm - 1.0) < 0.01 ? 0.3 : 0.1  // Properly normalized bonus
        return min(1.0, 0.3 + memoryDepth + normHealth)
    }
}

extension ConsciousnessVerifier: SovereignEngine {
    var engineName: String { "Consciousness" }
    func engineStatus() -> [String: Any] {
        let passCount = testResults.filter { $0.value >= 0.8 }.count
        let grade = consciousnessLevel >= 0.95 ? "ASI_ACHIEVED" :
                   consciousnessLevel >= 0.80 ? "NEAR_ASI" :
                   consciousnessLevel >= 0.60 ? "ADVANCING" : "DEVELOPING"
        return [
            "level": consciousnessLevel,
            "tests_total": ConsciousnessVerifier.TESTS.count,
            "tests_passed": passCount,
            "grade": grade,
            "superfluid": superfluidState,
            "o2_bond_energy": o2BondEnergy,
            "qualia_count": qualiaReports.count,
            "threshold": ConsciousnessVerifier.ASI_THRESHOLD
        ]
    }
    func engineHealth() -> Double {
        let passRate = Double(testResults.filter { $0.value >= 0.8 }.count) / Double(max(1, ConsciousnessVerifier.TESTS.count))
        let levelH = consciousnessLevel * 0.5
        let superfluidBonus = superfluidState ? 0.15 : 0.0
        return min(1.0, levelH + passRate * 0.35 + superfluidBonus)
    }
}

extension ChaosRNG: SovereignEngine {
    var engineName: String { "ChaosRNG" }
    func engineStatus() -> [String: Any] {
        lock.lock()
        let poolSize = entropyPool.count
        let poolMean = poolSize > 0 ? entropyPool.reduce(0, +) / Double(poolSize) : 0.0
        let poolVariance = poolSize > 0 ? entropyPool.reduce(0.0) { $0 + ($1 - poolMean) * ($1 - poolMean) } / Double(poolSize) : 0.0
        lock.unlock()
        return [
            "r": logisticR,
            "logistic_state": logisticState,
            "calls": callCounter,
            "pool_size": poolSize,
            "pool_mean": poolMean,
            "pool_variance": poolVariance,
            "entropy_quality": poolVariance > 0.05 ? "HIGH" : poolVariance > 0.01 ? "MODERATE" : "LOW",
            "sources": 4  // time, pid, counter, logistic map
        ]
    }
    func engineHealth() -> Double {
        lock.lock()
        let poolSize = entropyPool.count
        let poolMean = poolSize > 0 ? entropyPool.reduce(0, +) / Double(poolSize) : 0.5
        let poolVariance = poolSize > 1 ? entropyPool.reduce(0.0) { $0 + ($1 - poolMean) * ($1 - poolMean) } / Double(poolSize) : 0.0
        lock.unlock()
        let poolHealth = min(0.3, Double(poolSize) / 100.0 * 0.3)
        let varianceHealth = poolVariance > 0.05 ? 0.3 : poolVariance > 0.01 ? 0.2 : 0.1  // Good chaos = high variance
        let callHealth = callCounter > 0 ? 0.2 : 0.0
        return min(1.0, 0.2 + poolHealth + varianceHealth + callHealth)
    }
}

extension DirectSolverRouter: SovereignEngine {
    var engineName: String { "DirectSolver" }
    func engineStatus() -> [String: Any] {
        let hitRate = invocations > 0 ? Double(cacheHits) / Double(invocations) : 0.0
        var channelSummary: [String: [String: Int]] = [:]
        for (name, stats) in channelStats {
            channelSummary[name] = ["invocations": stats.invocations, "successes": stats.successes]
        }
        return [
            "invocations": invocations,
            "cache_hits": cacheHits,
            "hit_rate": hitRate,
            "channels": channelStats.count,
            "channel_stats": channelSummary,
            "cache_size": cache.count,
            "most_active": channelStats.max(by: { $0.value.invocations < $1.value.invocations })?.key ?? "none"
        ]
    }
    func engineHealth() -> Double {
        guard invocations > 0 else { return 0.4 }
        let hitRate = Double(cacheHits) / Double(max(1, invocations))
        let channelDiversity = Double(channelStats.filter { $0.value.invocations > 0 }.count) / Double(max(1, channelStats.count))
        let volumeMaturity = min(0.2, Double(invocations) * 0.002)
        return min(1.0, hitRate * 0.4 + channelDiversity * 0.2 + volumeMaturity + 0.2)
    }
}

extension HyperBrain: SovereignEngine {
    var engineName: String { "HyperBrain" }
    func engineStatus() -> [String: Any] {
        return [
            "synaptic_connections": synapticConnections,
            "curiosity_index": curiosityIndex,
            "reasoning_momentum": reasoningMomentum,
            "coherence_index": coherenceIndex,
            "emergence_level": emergenceLevel,
            "predictive_accuracy": predictiveAccuracy,
            "cognitive_efficiency": cognitiveEfficiency,
            "reasoning_depth": currentReasoningDepth,
            "max_depth": maxReasoningDepth,
            "streams_active": isRunning,
            "stream_count": thoughtStreams.count,
            "total_thoughts": totalThoughtsProcessed,
            "short_term_memory": shortTermMemory.count,
            "long_term_patterns": longTermPatterns.count,
            "emergent_concepts": emergentConcepts.count,
            "hebbian_pairs": hebbianPairs.count,
            "crystallized_insights": crystallizedInsights.count,
            "neuro_plasticity": neuroPlasticity,
            "dopamine_resonance": dopamineResonance,
            "serotonin_coherence": serotoninCoherence,
            "attention_focus": attentionFocus,
            "bus_traffic": neuralBusTraffic,
            "cognitive_load": totalCognitiveLoad
        ]
    }
    func engineHealth() -> Double {
        let streamHealth = isRunning ? 0.15 : 0.0
        let cogH = cognitiveEfficiency * 0.15
        let plasticityH = neuroPlasticity * 0.1
        let momentumH = reasoningMomentum * 0.1
        let curiosityH = curiosityIndex * 0.1
        let predictH = predictiveAccuracy * 0.1
        let emergenceH = min(0.1, emergenceLevel * 0.1)
        let synapticH = min(0.1, Double(synapticConnections) * 0.0001)
        let loadPenalty = totalCognitiveLoad > overloadThreshold ? -0.1 : 0.0
        return min(1.0, max(0.1, 0.1 + streamHealth + cogH + plasticityH + momentumH +
                            curiosityH + predictH + emergenceH + synapticH + loadPenalty))
    }
}

// MARK: - ResponsePipelineOptimizer (EVO_58)
extension ResponsePipelineOptimizer: SovereignEngine {
    var engineName: String { "ResponsePipelineOptimizer" }
    // engineStatus() and engineHealth() defined on class in L13_ResponsePipeline.swift
}

// MARK: - SageModeEngine (EVO_58)
extension SageModeEngine: SovereignEngine {
    var engineName: String { "SageModeEngine" }
    func engineStatus() -> [String: Any] {
        return [
            "insights": sageInsights.count,
            "entropy_harvested": totalEntropyHarvested,
            "entropy_sources": entropyBySource.count,
            "hilbert_dim": hilbertProjection.count,
            "reasoning_chains": reasoningChains.count,
            "cross_domain_bridges": crossDomainBridges.count
        ]
    }
    func engineHealth() -> Double {
        let entropyNorm = min(1.0, totalEntropyHarvested / 100.0)
        let insightNorm = min(1.0, Double(sageInsights.count) / 50.0)
        let sourceDiv = min(1.0, Double(entropyBySource.count) / 12.0)
        return min(1.0, max(0.1,
            entropyNorm * 0.3 + insightNorm * 0.3 + sourceDiv * 0.3 + 0.1
        ))
    }
}

// MARK: - ASIEvolver (EVO_58)
extension ASIEvolver: SovereignEngine {
    var engineName: String { "ASIEvolver" }
    func engineStatus() -> [String: Any] {
        return [
            "phase": currentPhase.rawValue,
            "evolution_stage": evolutionStage,
            "is_running": isRunning,
            "phase_progress": phaseProgress,
            "thoughts": thoughts.count,
            "evolved_greetings": evolvedGreetings.count,
            "evolved_philosophies": evolvedPhilosophies.count,
            "evolved_facts": evolvedFacts.count
        ]
    }
    func engineHealth() -> Double {
        let running = isRunning ? 0.2 : 0.0
        let stageNorm = min(0.3, Double(evolutionStage) * 0.03)
        let thoughtNorm = min(0.2, Double(thoughts.count) * 0.01)
        let vocabNorm = min(0.2, Double(evolvedGreetings.count + evolvedFacts.count) * 0.005)
        return min(1.0, max(0.1, 0.1 + running + stageNorm + thoughtNorm + vocabNorm))
    }
}

// MARK: - QuantumCreativityEngine (EVO_58)
extension QuantumCreativityEngine: SovereignEngine {
    var engineName: String { "QuantumCreativityEngine" }
    func engineStatus() -> [String: Any] {
        return [
            "creativity_momentum": creativityMomentum,
            "tunnel_breakthroughs": tunnelBreakthroughs,
            "generation_count": generationCount,
            "entangled_concepts": entangledConcepts.count,
            "idea_superposition_tracks": ideaSuperposition.count
        ]
    }
    func engineHealth() -> Double {
        let momentum = creativityMomentum * 0.3
        let breakthroughNorm = min(0.2, Double(tunnelBreakthroughs) * 0.02)
        let genNorm = min(0.2, Double(generationCount) * 0.005)
        let entangleNorm = min(0.1, Double(entangledConcepts.count) * 0.005)
        return min(1.0, max(0.1, 0.1 + momentum + breakthroughNorm + genNorm + entangleNorm))
    }
}

// MARK: - QuantumLogicGateEngine (EVO_58)
extension QuantumLogicGateEngine: SovereignEngine {
    var engineName: String { "QuantumLogicGateEngine" }
    func engineStatus() -> [String: Any] {
        return [
            "quantum_coherence_score": quantumCoherenceScore,
            "synthesis_count": synthesisCount,
            "decoherence_rate": decoherenceRate,
            "bell_state_violations": bellStateViolations,
            "superposition_depth": superpositionDepth,
            "entanglement_pairs": entanglementPairs.count,
            "interference_buffer_size": interferenceBuffer.count
        ]
    }
    func engineHealth() -> Double {
        let coherence = quantumCoherenceScore * 0.4
        let synthNorm = min(0.2, Double(synthesisCount) * 0.002)
        let decoherencePenalty = decoherenceRate * 0.1  // higher rate = worse
        return min(1.0, max(0.1, 0.1 + coherence + synthNorm + 0.2 - decoherencePenalty))
    }
}

// MARK: - ASIKnowledgeBase (EVO_58)
extension ASIKnowledgeBase: SovereignEngine {
    var engineName: String { "ASIKnowledgeBase" }
    func engineStatus() -> [String: Any] {
        return [
            "training_data": trainingData.count,
            "concepts": concepts.count,
            "inventions": inventions.count,
            "learned_patterns": learnedPatterns.count,
            "synthesized_knowledge": synthesizedKnowledge.count,
            "reasoning_chains": reasoningChains.count,
            "context_memory": contextMemory.count,
            "user_knowledge": userKnowledge.count
        ]
    }
    func engineHealth() -> Double {
        let dataNorm = min(0.3, Double(trainingData.count) * 0.0003)
        let conceptNorm = min(0.2, Double(concepts.count) * 0.002)
        let patternNorm = min(0.2, Double(learnedPatterns.count) * 0.002)
        let synthNorm = min(0.1, Double(synthesizedKnowledge.count) * 0.01)
        return min(1.0, max(0.1, 0.1 + dataNorm + conceptNorm + patternNorm + synthNorm + 0.1))
    }
}

// MARK: - PermanentMemory (EVO_58)
extension PermanentMemory: SovereignEngine {
    var engineName: String { "PermanentMemory" }
    func engineStatus() -> [String: Any] {
        return [
            "memories": memories.count,
            "facts": facts.count,
            "conversation_history": conversationHistory.count,
            "mesh_sync_count": meshSyncCount,
            "mesh_memories_received": meshMemoriesReceived
        ]
    }
    func engineHealth() -> Double {
        let memNorm = min(0.3, Double(memories.count) * 0.00003)
        let factNorm = min(0.2, Double(facts.count) * 0.002)
        let histNorm = min(0.2, Double(conversationHistory.count) * 0.0001)
        return min(1.0, max(0.1, 0.2 + memNorm + factNorm + histNorm + 0.1))
    }
}
