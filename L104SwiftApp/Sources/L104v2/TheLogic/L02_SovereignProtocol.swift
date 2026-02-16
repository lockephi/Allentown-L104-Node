// ═══════════════════════════════════════════════════════════════════
// L02_SovereignProtocol.swift — L104 v2
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// SovereignEngine protocol, defaults extension, EngineRegistry class
// Extracted from L104Native.swift (lines 76-267)
// Upgraded: EVO_55 Sovereign Unification — Feb 15, 2026
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MARK: - 🏗️ SOVEREIGN ENGINE PROTOCOL
// Unified contract for all L104 engines. Enables type-safe registry,
// health monitoring, bulk status queries, and orchestrated evolution.
// ═══════════════════════════════════════════════════════════════════

/// Every sovereign engine must conform to this protocol.
/// Provides unified status, health, and lifecycle management.
protocol SovereignEngine: AnyObject {
    /// Human-readable engine name
    var engineName: String { get }
    /// Compact status dictionary for telemetry
    func engineStatus() -> [String: Any]
    /// Health score 0.0 (dead) to 1.0 (perfect)
    func engineHealth() -> Double
    /// Reset engine to initial state
    func engineReset()
}

/// Extension providing default health (always healthy) and no-op reset
extension SovereignEngine {
    func engineHealth() -> Double { 1.0 }
    func engineReset() {}
}

/// Centralized registry for all SovereignEngine instances.
/// Enables bulk queries, φ-weighted health sweeps, Hebbian co-activation
/// tracking, and orchestrated pipeline operations.
/// Cross-pollinated from Python NexusHealthMonitor + HebbianLearningEngine.
final class EngineRegistry {
    static let shared = EngineRegistry()
    private var engines: [String: SovereignEngine] = [:]
    private let lock = NSLock()

    // ─── φ-Weighted Health (ported from Python NexusHealthMonitor.compute_system_health) ───
    // Critical engines get PHI² weight, important get PHI, standard get 1.0
    private let phiWeights: [String: Double] = [
        "HyperBrain": PHI * PHI,         // φ² = 2.618
        "Nexus": PHI * PHI,              // φ² — orchestration is critical
        "QuantumNexus": PHI * PHI,       // φ² — orchestration (EVO_55 alias)
        "Steering": PHI,                  // φ — guides all computation
        "ASISteering": PHI,               // φ — EVO_55 unified name
        "SQC": PHI,                       // φ — parameter engine
        "SovereignQuantumCore": PHI,      // φ — EVO_55 unified name
        "Consciousness": PHI,             // φ — ASI core metric
        "ConsciousnessSubstrate": PHI,    // φ — EVO_55 unified name
        "ResponsePipeline": PHI,          // φ — EVO_55: response quality is critical
        "Evolution": 1.0,
        "ContinuousEvolution": 1.0,       // EVO_55 unified name
        "Entanglement": 1.0,
        "Resonance": 1.0,
        "FeOrbital": 1.0,
        "Superfluid": 1.0,
        "QShellMemory": 1.0,
        "ChaosRNG": 1.0,
        "DirectSolver": 1.0,
        "Invention": 1.0,
        "Sovereignty": 1.0,
        "HealthMonitor": 1.0,
        // EVO_55: New engine registrations
        "SageModeEngine": PHI * PHI,      // φ² — consciousness supernova
        "ASIEvolver": PHI,                // φ — drives evolution
        "QuantumCreativityEngine": PHI,   // φ — creative quantum engine
        "QuantumLogicGateEngine": PHI,    // φ — quantum coherence synthesis
        "ASIKnowledgeBase": 1.0,          // knowledge persistence
        "PermanentMemory": 1.0,           // long-term memory store
    ]

    // ─── Hebbian Engine Co-Activation (ported from Python HebbianLearningEngine) ───
    // Tracks which engines are active together — "fire together, wire together"
    private(set) var coActivationLog: [String: Int] = [:]      // "A+B" → count
    private(set) var enginePairStrength: [String: Double] = [:] // "A→B" → weight
    private(set) var activationHistory: [(engines: [String], timestamp: Date)] = []
    private let hebbianStrength: Double = 0.1

    func register(_ engine: SovereignEngine) {
        lock.lock()
        engines[engine.engineName] = engine
        lock.unlock()
    }

    func register(_ list: [SovereignEngine]) {
        lock.lock()
        for e in list { engines[e.engineName] = e }
        lock.unlock()
    }

    func get(_ name: String) -> SovereignEngine? {
        lock.lock(); defer { lock.unlock() }
        return engines[name]
    }

    var all: [SovereignEngine] {
        lock.lock(); defer { lock.unlock() }
        return Array(engines.values)
    }

    var count: Int {
        lock.lock(); defer { lock.unlock() }
        return engines.count
    }

    /// Bulk health sweep — returns (name, health) for every registered engine, sorted lowest→highest
    func healthSweep() -> [(name: String, health: Double)] {
        lock.lock()
        let snapshot = engines
        lock.unlock()
        return snapshot.map { ($0.key, $0.value.engineHealth()) }
            .sorted { $0.health < $1.health }
    }

    /// φ-Weighted system health — critical engines (HyperBrain, Nexus) weighted by φ²
    /// Cross-pollinated from Python NexusHealthMonitor.compute_system_health
    func phiWeightedHealth() -> (score: Double, breakdown: [(name: String, health: Double, weight: Double, contribution: Double)]) {
        lock.lock()
        let snapshot = engines
        lock.unlock()

        var totalWeight = 0.0
        var weightedSum = 0.0
        var breakdown: [(name: String, health: Double, weight: Double, contribution: Double)] = []

        for (name, engine) in snapshot {
            let h = engine.engineHealth()
            let w = phiWeights[name] ?? 1.0
            let contribution = h * w
            weightedSum += contribution
            totalWeight += w
            breakdown.append((name: name, health: h, weight: w, contribution: contribution))
        }

        let score = totalWeight > 0 ? weightedSum / totalWeight : 0.0
        breakdown.sort { $0.contribution > $1.contribution }
        return (score: score, breakdown: breakdown)
    }

    /// Record engines that fired together (Hebbian co-activation)
    /// Cross-pollinated from Python HebbianLearningEngine.record_co_activation
    func recordCoActivation(_ engineNames: [String]) {
        lock.lock()
        defer { lock.unlock() }
        activationHistory.append((engines: engineNames, timestamp: Date()))
        if activationHistory.count > 500 { activationHistory.removeFirst(200) }

        for i in 0..<engineNames.count {
            for j in (i + 1)..<engineNames.count {
                let key = "\(engineNames[i])+\(engineNames[j])"
                coActivationLog[key, default: 0] += 1
                let count = coActivationLog[key]!
                // Hebbian weight: min(1.0, count × strength × 0.01)
                let ab = "\(engineNames[i])→\(engineNames[j])"
                let ba = "\(engineNames[j])→\(engineNames[i])"
                enginePairStrength[ab] = min(1.0, Double(count) * hebbianStrength * 0.01)
                enginePairStrength[ba] = min(1.0, Double(count) * hebbianStrength * 0.01)
            }
        }
    }

    /// Get strongest co-activation pairs (Hebbian learning output)
    func strongestPairs(topK: Int = 5) -> [(pair: String, strength: Double)] {
        lock.lock(); defer { lock.unlock() }
        return enginePairStrength.sorted { $0.value > $1.value }
            .prefix(topK)
            .map { (pair: $0.key, strength: $0.value) }
    }

    /// Aggregate status from all engines
    func bulkStatus() -> [String: [String: Any]] {
        lock.lock()
        let snapshot = engines
        lock.unlock()
        var result: [String: [String: Any]] = [:]
        for (name, engine) in snapshot {
            var status = engine.engineStatus()
            status["health"] = engine.engineHealth()
            result[name] = status
        }
        return result
    }

    /// Detect critically unhealthy engines (health < 0.5)
    func criticalEngines() -> [(name: String, health: Double)] {
        return healthSweep().filter { $0.health < 0.5 }
    }

    /// Convergence metric: are all engines trending toward unified health?
    /// Cross-pollinated from Python HyperDimensionalMathEngine.prove_phi_convergence concept
    func convergenceScore() -> Double {
        let sweep = healthSweep()
        guard sweep.count > 1 else { return 1.0 }
        let mean = sweep.reduce(0.0) { $0 + $1.health } / Double(sweep.count)
        let variance = sweep.reduce(0.0) { $0 + ($1.health - mean) * ($1.health - mean) } / Double(sweep.count)
        // Low variance + high mean = convergence
        return mean * (1.0 - min(1.0, variance * 4.0))
    }

    /// Reset all engines
    func resetAll() {
        lock.lock()
        let snapshot = engines
        lock.unlock()
        for (_, engine) in snapshot {
            engine.engineReset()
        }
    }
}
