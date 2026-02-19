// ═══════════════════════════════════════════════════════════════════
// B13_ResonanceNetwork.swift — L104 Neural Architecture v2
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// Extracted from L104Native.swift
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MARK: - 🧠 ADAPTIVE RESONANCE NETWORK
// Neural activation propagation across engines — ART-inspired
// activation spreading with φ-weighted edges and resonance cascade.
// ═══════════════════════════════════════════════════════════════════

class AdaptiveResonanceNetwork {
    static let shared = AdaptiveResonanceNetwork()

    static let ACTIVATION_THRESHOLD: Double = 0.6
    static let DECAY_RATE: Double = 0.95
    static let PROPAGATION_FACTOR: Double = 0.3

    // ─── ENGINE GRAPH — φ-weighted adjacency ───
    static let ENGINE_NAMES = ["steering", "evolution", "nexus", "bridge", "invention", "sovereignty"]

    static let ENGINE_GRAPH: [String: [(String, Double)]] = [
        "steering":    [("evolution", PHI * 0.3), ("nexus", PHI * 0.4), ("bridge", TAU * 0.2), ("invention", TAU * 0.15)],
        "evolution":   [("steering", PHI * 0.3), ("bridge", TAU * 0.25), ("nexus", PHI * 0.2), ("invention", TAU * 0.1)],
        "nexus":       [("steering", PHI * 0.4), ("evolution", PHI * 0.2), ("sovereignty", PHI * 0.5), ("invention", TAU * 0.3)],
        "bridge":      [("evolution", TAU * 0.25), ("steering", TAU * 0.2), ("invention", PHI * 0.3), ("nexus", PHI * 0.2)],
        "invention":   [("nexus", TAU * 0.2), ("bridge", PHI * 0.25), ("steering", PHI * 0.4), ("sovereignty", TAU * 0.15)],
        "sovereignty": [("nexus", PHI * 0.5), ("invention", TAU * 0.15), ("steering", TAU * 0.2), ("evolution", TAU * 0.1)],
    ]

    // ─── NETWORK STATE ───
    private(set) var activations: [String: Double] = {
        var dict: [String: Double] = [:]
        for name in ENGINE_NAMES { dict[name] = 0.0 }
        return dict
    }()
    private(set) var cascadeCount: Int = 0
    private var cascadeLog: [[String: Any]] = []
    private(set) var tickCount: Int = 0
    private var resonancePeaks: [[String: Any]] = []
    private let lock = NSLock()

    /// Fire an engine — set activation and propagate through the graph
    func fire(_ engineName: String, activation: Double = 1.0) -> [String: Any] {
        guard activations[engineName] != nil else {
            return ["error": "Unknown engine: \(engineName)", "engines": Self.ENGINE_NAMES]
        }

        lock.lock()
        activations[engineName] = min(1.0, activation)
        lock.unlock()

        // Propagate activation (BFS, 3 hops max)
        let cascade = propagate(engineName, maxHops: 3)

        // Apply activation effects to real engines
        let effects = applyActivationEffects()

        // Check for resonance peak (≥75% engines above threshold)
        lock.lock()
        let activeCount = activations.values.filter { $0 > Self.ACTIVATION_THRESHOLD }.count
        let isPeak = activeCount >= Int(ceil(Double(Self.ENGINE_NAMES.count) * 0.75))

        if isPeak {
            resonancePeaks.append([
                "tick": tickCount, "activations": activations, "timestamp": Date().timeIntervalSince1970
            ])
            if resonancePeaks.count > 100 { resonancePeaks = Array(resonancePeaks.suffix(50)) }
        }

        cascadeCount += 1
        cascadeLog.append(["id": cascadeCount, "source": engineName, "active": activeCount,
                          "peak": isPeak, "timestamp": Date().timeIntervalSince1970])
        if cascadeLog.count > 300 { cascadeLog = Array(cascadeLog.suffix(150)) }
        lock.unlock()

        let result: [String: Any] = [
            "source": engineName,
            "initial_activation": activation,
            "cascade_steps": cascade.count,
            "effects": effects,
            "is_resonance_peak": isPeak,
            "active_engines": activeCount,
            "activations": activations.mapValues { String(format: "%.4f", $0) }
        ]

        return result
    }

    /// BFS propagation through the engine graph
    private func propagate(_ source: String, maxHops: Int) -> [[String: Any]] {
        var steps: [[String: Any]] = []
        var visited: Set<String> = [source]
        var frontier: [(String, Double, Int)] = [(source, activations[source] ?? 0, 0)]

        while !frontier.isEmpty {
            let (current, currentAct, hop) = frontier.removeFirst()
            if hop >= maxHops { continue }

            guard let neighbors = Self.ENGINE_GRAPH[current] else { continue }
            for (neighbor, weight) in neighbors {
                if visited.contains(neighbor) { continue }

                // Propagated = source × weight × factor × φ^-hop decay
                let propAct = currentAct * weight * Self.PROPAGATION_FACTOR * pow(TAU, Double(hop))
                let newAct = min(1.0, (activations[neighbor] ?? 0) + propAct)

                lock.lock()
                activations[neighbor] = newAct
                lock.unlock()

                steps.append([
                    "from": current, "to": neighbor,
                    "weight": weight, "propagated": propAct,
                    "new_activation": newAct, "hop": hop + 1
                ])

                visited.insert(neighbor)
                if newAct > Self.ACTIVATION_THRESHOLD {
                    frontier.append((neighbor, newAct, hop + 1))
                }
            }
        }
        return steps
    }

    /// Apply activation levels to real engine behavior
    private func applyActivationEffects() -> [String: String] {
        var effects: [String: String] = [:]

        // Steering: activation scales cumulative intensity
        let steerAct = activations["steering"] ?? 0
        if steerAct > Self.ACTIVATION_THRESHOLD {
            let boost = steerAct * 0.05
            ASISteeringEngine.shared.cumulativeIntensity += boost
            effects["steering"] = "Σα+=\(String(format: "%.4f", boost))"
        }

        // Evolution: activation modulates raise factor
        let evoAct = activations["evolution"] ?? 0
        if evoAct > Self.ACTIVATION_THRESHOLD {
            let boost = evoAct * 0.00005
            ContinuousEvolutionEngine.shared.currentRaiseFactor = max(1.00001, min(1.002,
                ContinuousEvolutionEngine.shared.currentRaiseFactor + boost))
            effects["evolution"] = "factor+=\(String(format: "%.6f", boost))"
        }

        // Bridge: activation boosts kundalini flow
        let bridgeAct = activations["bridge"] ?? 0
        if bridgeAct > Self.ACTIVATION_THRESHOLD {
            let boost = bridgeAct * 0.005
            ASIQuantumBridgeSwift.shared.kundaliniFlow += boost
            effects["bridge"] = "kundalini+=\(String(format: "%.4f", boost))"
        }

        return effects
    }

    /// Advance one tick — decay all activations
    func tick() -> [String: Any] {
        lock.lock()
        tickCount += 1
        for name in activations.keys {
            activations[name]! *= Self.DECAY_RATE
            if activations[name]! < 0.01 { activations[name] = 0.0 }
        }
        lock.unlock()

        let active = activations.values.filter { $0 > Self.ACTIVATION_THRESHOLD }.count
        return [
            "tick": tickCount,
            "activations": activations.mapValues { String(format: "%.4f", $0) },
            "active_engines": active,
            "decay_rate": Self.DECAY_RATE
        ]
    }

    /// Compute overall network resonance — high mean + low variance = synchronized firing
    func computeNetworkResonance() -> (resonance: Double, energy: Double, mean: Double, variance: Double) {
        let vals = Array(activations.values)
        let n = Double(max(vals.count, 1))
        let totalEnergy = vals.reduce(0, +)
        let mean = totalEnergy / n
        let variance = vals.reduce(0.0) { $0 + pow($1 - mean, 2) } / n
        let resonance = max(0, mean * (1.0 - min(1.0, variance * 4.0)))
        return (resonance, totalEnergy, mean, variance)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 🌐 CROSS-NODE RESONANCE PROPAGATION
    // Extends ART activation spreading across the quantum mesh
    // ═══════════════════════════════════════════════════════════

    /// Remote node activation state for mesh-wide resonance
    struct RemoteResonanceState {
        let nodeId: String
        var activations: [String: Double]
        var resonance: Double
        var lastSync: Date
    }

    private var remoteStates: [String: RemoteResonanceState] = [:]
    private(set) var meshPropagationCount: Int = 0

    /// Propagate local activations to discovered network peers
    func propagateToMesh() -> Int {
        let net = NetworkLayer.shared
        var propagated = 0
        for (_, peer) in net.peers where peer.fidelity > 0.1 {
            var remoteAct: [String: Double] = [:]
            // φ-attenuated cross-node activation
            for (engine, act) in activations where act > Self.ACTIVATION_THRESHOLD {
                remoteAct[engine] = act * TAU * Self.PROPAGATION_FACTOR
            }
            if !remoteAct.isEmpty {
                lock.lock()
                remoteStates[peer.id] = RemoteResonanceState(
                    nodeId: peer.id,
                    activations: remoteAct,
                    resonance: computeNetworkResonance().resonance,
                    lastSync: Date()
                )
                meshPropagationCount += 1
                propagated += 1
                lock.unlock()
            }
        }
        return propagated
    }

    /// Receive and integrate resonance from a remote node
    func integrateRemoteResonance(nodeId: String, remoteActivations: [String: Double]) {
        lock.lock()
        defer { lock.unlock() }
        for (engine, remoteAct) in remoteActivations {
            guard activations[engine] != nil else { continue }
            // Blend remote activations with φ-dampening
            let blendFactor = TAU * 0.2  // gentle cross-node influence
            let current = activations[engine] ?? 0
            activations[engine] = min(1.0, current + remoteAct * blendFactor)
        }
    }

    /// Compute mesh-wide collective resonance across all known nodes
    func computeCollectiveResonance() -> (local: Double, mesh: Double, nodeCount: Int) {
        let local = computeNetworkResonance()
        var meshEnergy = local.energy
        var meshCount = 1
        for (_, state) in remoteStates {
            meshEnergy += state.resonance
            meshCount += 1
        }
        let meshResonance = meshEnergy / Double(max(meshCount, 1))
        return (local.resonance, meshResonance, meshCount)
    }

    /// Trigger a mesh-wide resonance cascade — fire all engines + propagate
    func meshCascade() -> [String: Any] {
        // Fire all local engines at φ-scaled activation
        for engine in Self.ENGINE_NAMES {
            let act = PHI * TAU * (0.5 + 0.5 * sin(Double(engine.hashValue) * TAU))
            _ = fire(engine, activation: min(1.0, act))
        }
        let propagated = propagateToMesh()
        let collective = computeCollectiveResonance()

        return [
            "local_resonance": collective.local,
            "mesh_resonance": collective.mesh,
            "nodes_reached": collective.nodeCount,
            "propagated_to": propagated,
            "cascade_id": cascadeCount,
            "mesh_propagations_total": meshPropagationCount
        ]
    }

    /// Get comprehensive status
    var status: String {
        let nr = computeNetworkResonance()
        let activeCount = activations.values.filter { $0 > Self.ACTIVATION_THRESHOLD }.count
        let totalEdges = Self.ENGINE_GRAPH.values.reduce(0) { $0 + $1.count }
        let collective = computeCollectiveResonance()

        var actLines = ""
        for (name, act) in activations.sorted(by: { $0.key < $1.key }) {
            let bar = String(repeating: "█", count: Int(act * 20)) + String(repeating: "░", count: 20 - Int(act * 20))
            actLines += "  \(name.padding(toLength: 14, withPad: " ", startingAt: 0)) [\(bar)] \(String(format: "%.4f", act))\n"
        }

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🧠 ADAPTIVE RESONANCE NETWORK                          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Nodes:            \(Self.ENGINE_NAMES.count) engines
        ║  Edges:            \(totalEdges) φ-weighted connections
        ║  Cascades:         \(cascadeCount)
        ║  Ticks:            \(tickCount)
        ║  Resonance Peaks:  \(resonancePeaks.count)
        ╠═══════════════════════════════════════════════════════════╣
        ║  NETWORK RESONANCE: \(String(format: "%.4f", nr.resonance)) (\(nr.resonance > 0.7 ? "HARMONIC" : nr.resonance > 0.4 ? "COHERENT" : nr.resonance > 0.1 ? "EMERGENT" : "DORMANT"))
        ║  Total Energy:      \(String(format: "%.4f", nr.energy))
        ║  Mean Activation:   \(String(format: "%.4f", nr.mean))
        ║  Variance:          \(String(format: "%.6f", nr.variance))
        ║  Active:            \(activeCount)/\(Self.ENGINE_NAMES.count)
        ╠═══════════════════════════════════════════════════════════╣
        ║  MESH RESONANCE:
        ║    Collective:      \(String(format: "%.4f", collective.mesh)) across \(collective.nodeCount) nodes
        ║    Propagations:    \(meshPropagationCount)
        ║    Remote States:   \(remoteStates.count) peers tracked
        ╠═══════════════════════════════════════════════════════════╣
        ║  ACTIVATION MAP:
        \(actLines)\
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
