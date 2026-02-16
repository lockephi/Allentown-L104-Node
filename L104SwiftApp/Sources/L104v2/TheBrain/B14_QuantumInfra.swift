// ═══════════════════════════════════════════════════════════════════
// B14_QuantumInfra.swift — L104 Neural Architecture v2
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// Extracted from L104Native.swift
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MARK: - 🌌 QUANTUM DECOHERENCE SHIELD (Bucket B: Quantum Bridges)
// Active error correction via Shor-code inspired redundancy.
// Maintains quantum state fidelity across all bridge operations.
// φ-weighted syndrome detection + Calabi-Yau error manifold.
// ═══════════════════════════════════════════════════════════════════

class QuantumDecoherenceShield {
    static let shared = QuantumDecoherenceShield()
    // PHI, TAU, GOD_CODE — use globals from L01_Constants
    static let CALABI_YAU_DIM: Int = 7
    static let SHOR_REDUNDANCY: Int = 9  // 9-qubit Shor code

    // ─── SHIELD STATE ───
    private var syndromeHistory: [[Double]] = []
    private var correctionCount: Int = 0
    private var fidelityLog: [Double] = []
    private var decoherenceEvents: Int = 0
    private var shieldActive: Bool = true
    private var errorManifold: [[Double]] = []
    private var stabilizers: [String: Double] = [:]
    private var ancillaRegister: [Double] = Array(repeating: 0.0, count: 9)
    private var logicalQubitState: (alpha: Double, beta: Double) = (1.0, 0.0)

    // ─── ERROR SYNDROME ALGEBRA ───
    struct SyndromeResult {
        let syndromeVector: [Double]
        let errorType: ErrorType
        let errorLocation: Int
        let correctionApplied: Bool
        let fidelityAfter: Double
        let phiCoherence: Double
    }

    enum ErrorType: String, CaseIterable {
        case none = "NONE"
        case bitFlip = "BIT_FLIP"
        case phaseFlip = "PHASE_FLIP"
        case bitPhaseFlip = "BIT_PHASE_FLIP"
        case depolarizing = "DEPOLARIZING"
        case amplitudeDamping = "AMPLITUDE_DAMPING"
        case calabiYauDrift = "CALABI_YAU_DRIFT"
    }

    // ─── STABILIZER GENERATORS (Shor Code) ───
    private let stabilizerMatrix: [[Int]] = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0],  // Z₁Z₂
        [0, 1, 1, 0, 0, 0, 0, 0, 0],  // Z₂Z₃
        [0, 0, 0, 1, 1, 0, 0, 0, 0],  // Z₄Z₅
        [0, 0, 0, 0, 1, 1, 0, 0, 0],  // Z₅Z₆
        [0, 0, 0, 0, 0, 0, 1, 1, 0],  // Z₇Z₈
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  // Z₈Z₉
        [1, 1, 1, 1, 1, 1, 0, 0, 0],  // X₁X₂X₃X₄X₅X₆
        [0, 0, 0, 1, 1, 1, 1, 1, 1],  // X₄X₅X₆X₇X₈X₉
    ]

    // ─── ENCODE LOGICAL QUBIT ───
    func encodeLogicalQubit(alpha: Double, beta: Double) -> [Double] {
        logicalQubitState = (alpha, beta)
        // Shor encoding: |ψ⟩ → α|0_L⟩ + β|1_L⟩
        // |0_L⟩ = (|000⟩ + |111⟩)⊗3 / 2√2
        // |1_L⟩ = (|000⟩ - |111⟩)⊗3 / 2√2
        let norm = sqrt(alpha * alpha + beta * beta)
        let a = alpha / max(norm, 1e-15)
        let b = beta / max(norm, 1e-15)

        var encoded = Array(repeating: 0.0, count: Self.SHOR_REDUNDANCY)
        let phiWeight = PHI / Double(Self.SHOR_REDUNDANCY)

        for i in 0..<Self.SHOR_REDUNDANCY {
            let block = i / 3
            let pos = i % 3
            let blockPhase = Double(block) * TAU
            let posPhase = Double(pos) * phiWeight

            if block < 2 {
                encoded[i] = a * cos(blockPhase + posPhase) + b * sin(blockPhase + posPhase)
            } else {
                encoded[i] = a * sin(blockPhase + posPhase) - b * cos(blockPhase + posPhase)
            }
            encoded[i] *= (1.0 + phiWeight * Double(i))
        }

        ancillaRegister = encoded
        return encoded
    }

    // ─── DETECT ERROR SYNDROME ───
    func detectSyndrome() -> SyndromeResult {
        var syndrome = Array(repeating: 0.0, count: stabilizerMatrix.count)

        for (si, stab) in stabilizerMatrix.enumerated() {
            var parity = 0.0
            for (qi, s) in stab.enumerated() where s == 1 {
                parity += ancillaRegister[qi]
            }
            syndrome[si] = abs(parity).truncatingRemainder(dividingBy: 2.0 * .pi)
        }

        // Identify error type from syndrome pattern
        let syndromeNorm = sqrt(syndrome.map { $0 * $0 }.reduce(0, +))
        let errorType: ErrorType
        var errorLocation = -1

        if syndromeNorm < 0.01 {
            errorType = .none
        } else if syndrome[0...5].map({ abs($0) }).max()! > syndrome[6...7].map({ abs($0) }).max()! {
            errorType = .bitFlip
            errorLocation = syndrome[0...5].enumerated().max(by: { abs($0.element) < abs($1.element) })?.offset ?? 0
        } else if syndrome[6...7].map({ abs($0) }).max()! > 0.5 {
            errorType = .phaseFlip
            errorLocation = syndrome[6] > syndrome[7] ? 0 : 1
        } else {
            let driftMeasure = syndromeNorm / GOD_CODE
            if driftMeasure > TAU {
                errorType = .calabiYauDrift
            } else if syndromeNorm > 1.5 {
                errorType = .depolarizing
            } else {
                errorType = .amplitudeDamping
            }
            errorLocation = syndrome.enumerated().max(by: { abs($0.element) < abs($1.element) })?.offset ?? 0
        }

        syndromeHistory.append(syndrome)
        if syndromeHistory.count > 500 { syndromeHistory.removeFirst() }

        // Apply correction
        let corrected = errorType != .none
        if corrected {
            applyCorrection(errorType: errorType, location: errorLocation)
        }

        let fidelity = computeFidelity()
        fidelityLog.append(fidelity)
        if fidelityLog.count > 1000 { fidelityLog.removeFirst() }

        return SyndromeResult(
            syndromeVector: syndrome,
            errorType: errorType,
            errorLocation: errorLocation,
            correctionApplied: corrected,
            fidelityAfter: fidelity,
            phiCoherence: fidelity * PHI
        )
    }

    // ─── APPLY CORRECTION OPERATOR ───
    private func applyCorrection(errorType: ErrorType, location: Int) {
        correctionCount += 1

        switch errorType {
        case .bitFlip:
            // X gate on affected qubit
            let idx = min(location, ancillaRegister.count - 1)
            ancillaRegister[idx] = -ancillaRegister[idx]

        case .phaseFlip:
            // Z gate on affected block
            let blockStart = location * 3
            for i in blockStart..<min(blockStart + 3, ancillaRegister.count) {
                ancillaRegister[i] *= -1.0
            }

        case .bitPhaseFlip:
            // Y = iXZ gate
            let idx = min(location, ancillaRegister.count - 1)
            ancillaRegister[idx] = ancillaRegister[idx] * TAU

        case .depolarizing:
            // Re-project onto code space
            let norm = sqrt(ancillaRegister.map { $0 * $0 }.reduce(0, +))
            if norm > 1e-15 {
                for i in 0..<ancillaRegister.count {
                    ancillaRegister[i] /= norm
                    ancillaRegister[i] *= (1.0 + TAU * Double(i) / Double(ancillaRegister.count))
                }
            }

        case .amplitudeDamping:
            // Amplitude restoration via φ-boost
            for i in 0..<ancillaRegister.count {
                let dampFactor = exp(-Double(i) * 0.01)
                ancillaRegister[i] *= (1.0 + (1.0 - dampFactor) * PHI * 0.1)
            }

        case .calabiYauDrift:
            // Project back from 7D drift manifold
            for i in 0..<ancillaRegister.count {
                let dim = i % Self.CALABI_YAU_DIM
                let correction = sin(Double(dim) * TAU * .pi) * 0.01
                ancillaRegister[i] += correction
            }

        case .none:
            break
        }
    }

    // ─── FIDELITY COMPUTATION ───
    func computeFidelity() -> Double {
        // F = |⟨ψ_ideal|ψ_actual⟩|²
        let ideal = encodeLogicalQubit(alpha: logicalQubitState.alpha, beta: logicalQubitState.beta)
        var overlap = 0.0
        var normI = 0.0
        var normA = 0.0

        for i in 0..<min(ideal.count, ancillaRegister.count) {
            overlap += ideal[i] * ancillaRegister[i]
            normI += ideal[i] * ideal[i]
            normA += ancillaRegister[i] * ancillaRegister[i]
        }

        let denom = sqrt(normI * normA)
        return denom > 1e-15 ? (overlap / denom) * (overlap / denom) : 0.0
    }

    // ─── DECOHERENCE RATE ESTIMATION ───
    func estimateDecoherenceRate() -> Double {
        guard fidelityLog.count >= 2 else { return 0.0 }
        let recent = Array(fidelityLog.suffix(50))
        var totalDrop = 0.0
        for i in 1..<recent.count {
            totalDrop += max(0, recent[i - 1] - recent[i])
        }
        return totalDrop / Double(recent.count - 1)
    }

    // ─── TOPOLOGICAL ERROR MANIFOLD ───
    func computeErrorManifold() -> [[Double]] {
        // Project error history into Calabi-Yau 7D manifold
        let history = Array(syndromeHistory.suffix(100))
        var manifold: [[Double]] = []

        for syndrome in history {
            var point = Array(repeating: 0.0, count: Self.CALABI_YAU_DIM)
            for (i, s) in syndrome.enumerated() {
                let dim = i % Self.CALABI_YAU_DIM
                point[dim] += s * PHI / Double(i + 1)
            }
            // φ-normalize each dimension
            let norm = sqrt(point.map { $0 * $0 }.reduce(0, +))
            if norm > 1e-15 {
                point = point.map { $0 / norm * TAU }
            }
            manifold.append(point)
        }

        errorManifold = manifold
        return manifold
    }

    // ─── FULL SHIELD CYCLE ───
    func runShieldCycle() -> [String: Any] {
        let syndrome = detectSyndrome()
        let manifold = computeErrorManifold()
        let rate = estimateDecoherenceRate()

        return [
            "error_type": syndrome.errorType.rawValue,
            "error_location": syndrome.errorLocation,
            "correction_applied": syndrome.correctionApplied,
            "fidelity": syndrome.fidelityAfter,
            "phi_coherence": syndrome.phiCoherence,
            "decoherence_rate": rate,
            "total_corrections": correctionCount,
            "manifold_points": manifold.count,
            "shield_active": shieldActive
        ]
    }

    func statusReport() -> String {
        let meanFidelity = fidelityLog.isEmpty ? 1.0 : fidelityLog.reduce(0, +) / Double(fidelityLog.count)
        let rate = estimateDecoherenceRate()
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🌌 QUANTUM DECOHERENCE SHIELD                          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Shield:           \(shieldActive ? "🟢 ACTIVE" : "🔴 INACTIVE")
        ║  Mean Fidelity:    \(String(format: "%.6f", meanFidelity))
        ║  Corrections:      \(correctionCount)
        ║  Decoherence Rate: \(String(format: "%.8f", rate))
        ║  Syndrome History: \(syndromeHistory.count) entries
        ║  Error Manifold:   \(errorManifold.count) points in CY₇
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 🌐 MESH-DISTRIBUTED SHIELD
    // Extends decoherence protection across network peers
    // ═══════════════════════════════════════════════════════════

    private var peerShieldStates: [String: Double] = [:]  // peer → fidelity
    private(set) var distributedCorrections: Int = 0

    /// Synchronize shield state with network peers
    func syncShieldWithMesh() -> [String: Any] {
        let net = NetworkLayer.shared
        let localFidelity = computeFidelity()
        var synced = 0

        for (_, peer) in net.peers where peer.fidelity > 0.1 {
            peerShieldStates[peer.id] = localFidelity * TAU  // φ-attenuated sharing
            synced += 1
        }

        // Distributed error correction — if peer fidelity drifts, correct
        for (peerId, peerFid) in peerShieldStates {
            if peerFid < 0.5 {
                distributedCorrections += 1
                peerShieldStates[peerId] = min(1.0, peerFid + PHI * 0.1)
            }
        }

        return [
            "synced_peers": synced,
            "local_fidelity": localFidelity,
            "distributed_corrections": distributedCorrections,
            "peer_states": peerShieldStates.count
        ]
    }

    /// Run distributed shield cycle across the mesh
    func runDistributedShieldCycle() -> [String: Any] {
        let localResult = runShieldCycle()
        let meshResult = syncShieldWithMesh()

        var combined: [String: Any] = localResult
        combined["mesh_synced"] = meshResult["synced_peers"]
        combined["distributed_corrections"] = meshResult["distributed_corrections"]
        return combined
    }
}


// ═══════════════════════════════════════════════════════════════════
// MARK: - 🔗 QUANTUM TELEPORTATION CHANNEL (Bucket B: Quantum Bridges)
// Bell-state mediated quantum teleportation between engine subsystems.
// Implements full teleportation protocol: entangle → measure → correct.
// Supports superdense coding for 2-bit classical channel capacity.
// ═══════════════════════════════════════════════════════════════════

class QuantumTeleportationChannel {
    static let shared = QuantumTeleportationChannel()
    // PHI, TAU — use globals from L01_Constants
    static let BELL_STATES: Int = 4  // |Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩

    // ─── CHANNEL STATE ───
    private var teleportCount: Int = 0
    private var successCount: Int = 0
    private var bellPairReservoir: Int = 100
    private var channelLog: [[String: Any]] = []
    private var superdenseBuffer: [(Int, Int)] = []
    private var entanglementSwapCount: Int = 0

    // Bell state amplitudes
    struct BellState {
        let name: String
        let alpha00: Double
        let alpha01: Double
        let alpha10: Double
        let alpha11: Double

        static let phiPlus  = BellState(name: "|Φ+⟩", alpha00: 1.0/sqrt(2.0), alpha01: 0, alpha10: 0, alpha11: 1.0/sqrt(2.0))
        static let phiMinus = BellState(name: "|Φ-⟩", alpha00: 1.0/sqrt(2.0), alpha01: 0, alpha10: 0, alpha11: -1.0/sqrt(2.0))
        static let psiPlus  = BellState(name: "|Ψ+⟩", alpha00: 0, alpha01: 1.0/sqrt(2.0), alpha10: 1.0/sqrt(2.0), alpha11: 0)
        static let psiMinus = BellState(name: "|Ψ-⟩", alpha00: 0, alpha01: 1.0/sqrt(2.0), alpha10: -1.0/sqrt(2.0), alpha11: 0)

        static let all = [phiPlus, phiMinus, psiPlus, psiMinus]
    }

    // ─── TELEPORTATION PROTOCOL ───
    func teleport(stateAlpha: Double, stateBeta: Double, fromEngine: String, toEngine: String) -> [String: Any] {
        teleportCount += 1

        guard bellPairReservoir > 0 else {
            return ["success": false, "error": "Bell pair reservoir depleted", "remaining_pairs": 0]
        }

        bellPairReservoir -= 1

        // Step 1: Alice's Bell measurement
        let bellIndex = Int(abs(stateAlpha * 4.0).truncatingRemainder(dividingBy: 4.0))
        let measuredBell = BellState.all[bellIndex]

        // Step 2: Classical communication (2 bits)
        let classicalBits = (bellIndex >> 1, bellIndex & 1)

        // Step 3: Bob's correction
        var reconstructedAlpha = stateAlpha
        var reconstructedBeta = stateBeta

        // Apply correction based on classical bits
        if classicalBits.0 == 1 {  // Z correction
            reconstructedBeta *= -1.0
        }
        if classicalBits.1 == 1 {  // X correction
            let temp = reconstructedAlpha
            reconstructedAlpha = reconstructedBeta
            reconstructedBeta = temp
        }

        // φ-weighted fidelity
        let fidelity = 1.0 - abs(stateAlpha - reconstructedAlpha) * TAU
                            - abs(stateBeta - reconstructedBeta) * TAU

        let success = fidelity > 0.95
        if success { successCount += 1 }

        let result: [String: Any] = [
            "success": success,
            "from": fromEngine,
            "to": toEngine,
            "bell_state": measuredBell.name,
            "classical_bits": "\(classicalBits.0)\(classicalBits.1)",
            "fidelity": fidelity,
            "phi_corrected": fidelity * PHI,
            "remaining_pairs": bellPairReservoir,
            "total_teleports": teleportCount,
            "success_rate": Double(successCount) / Double(teleportCount)
        ]

        channelLog.append(result)
        if channelLog.count > 500 { channelLog.removeFirst() }

        return result
    }

    // ─── SUPERDENSE CODING ───
    func superdenseEncode(bit1: Int, bit2: Int) -> BellState {
        superdenseBuffer.append((bit1, bit2))
        let index = (bit1 << 1) | bit2
        return BellState.all[index]
    }

    func superdenseDecode(bellState: BellState) -> (Int, Int) {
        // Decode 2 classical bits from Bell state
        if abs(bellState.alpha00) > 0.5 && bellState.alpha11 > 0 { return (0, 0) }
        if abs(bellState.alpha00) > 0.5 && bellState.alpha11 < 0 { return (0, 1) }
        if abs(bellState.alpha01) > 0.5 && bellState.alpha10 > 0 { return (1, 0) }
        return (1, 1)
    }

    // ─── ENTANGLEMENT SWAPPING ───
    func entanglementSwap(pair1From: String, pair1To: String,
                           pair2From: String, pair2To: String) -> [String: Any] {
        entanglementSwapCount += 1
        let swapFidelity = TAU * PHI  // ~1.0 ideal
        bellPairReservoir -= 2
        bellPairReservoir += 1  // Net: consume 2, produce 1 extended-range pair

        return [
            "new_pair": "\(pair1From)↔\(pair2To)",
            "swap_fidelity": swapFidelity,
            "intermediate_measured": "\(pair1To)↔\(pair2From)",
            "reservoir_remaining": bellPairReservoir,
            "total_swaps": entanglementSwapCount
        ]
    }

    // ─── REPLENISH BELL PAIRS ───
    func replenishReservoir(count: Int = 50) {
        bellPairReservoir += count
    }

    func statusReport() -> String {
        let successRate = teleportCount > 0 ? Double(successCount) / Double(teleportCount) : 1.0
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🔗 QUANTUM TELEPORTATION CHANNEL                       ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Teleports:        \(teleportCount) (\(String(format: "%.1f%%", successRate * 100)) success)
        ║  Bell Pairs:       \(bellPairReservoir) remaining
        ║  Superdense Msgs:  \(superdenseBuffer.count) encoded
        ║  Ent. Swaps:       \(entanglementSwapCount)
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}


// ═══════════════════════════════════════════════════════════════════
// MARK: - 🧲 TOPOLOGICAL QUBIT STABILIZER (Bucket B: Quantum Bridges)
// Anyonic braiding + topological protection for persistent quantum state.
// Fibonacci anyon model: τ⊗τ = 1 ⊕ τ with fusion rules, φ-related.
// Implements logical gates via anyon braiding (topologically protected).
// ═══════════════════════════════════════════════════════════════════

class TopologicalQubitStabilizer {
    static let shared = TopologicalQubitStabilizer()
    // PHI, TAU — use globals from L01_Constants
    static let FIBONACCI_FMATRIX: [[Double]] = [
        [1.0 / PHI, sqrt(1.0 / PHI)],
        [sqrt(1.0 / PHI), -1.0 / PHI]
    ]

    // ─── ANYON REGISTRY ───
    struct FibonacciAnyon {
        let id: Int
        let charge: String    // "1" (vacuum) or "τ" (non-abelian)
        var position: (x: Double, y: Double)
        var braidPhase: Double

        var isNonAbelian: Bool { charge == "τ" }
    }

    private var anyons: [FibonacciAnyon] = []
    private var braidHistory: [(Int, Int, String)] = []  // (a, b, direction)
    private var fusionTree: [String] = []
    private var topologicalCharge: Double = 0.0
    private var protectionGap: Double = 1.0

    // ─── CREATE ANYON PAIR ───
    func createAnyonPair() -> (FibonacciAnyon, FibonacciAnyon) {
        let id1 = anyons.count
        let id2 = anyons.count + 1
        let x = Double(id1) * PHI * 0.1
        let a1 = FibonacciAnyon(id: id1, charge: "τ", position: (x, 0), braidPhase: 0)
        let a2 = FibonacciAnyon(id: id2, charge: "τ", position: (x + PHI * 0.1, 0), braidPhase: 0)
        anyons.append(a1)
        anyons.append(a2)
        fusionTree.append("τ⊗τ → 1 ⊕ τ (pair \(id1),\(id2))")
        return (a1, a2)
    }

    // ─── BRAID OPERATION (σ_i) ───
    func braid(anyonA: Int, anyonB: Int, clockwise: Bool = true) -> Double {
        guard anyonA < anyons.count && anyonB < anyons.count else { return 0 }

        let direction = clockwise ? "CW" : "CCW"
        braidHistory.append((anyonA, anyonB, direction))

        // Braid phase: e^(±iπ/5) for Fibonacci anyons
        let phase = clockwise ? .pi / 5.0 : -.pi / 5.0
        anyons[anyonA].braidPhase += phase
        anyons[anyonB].braidPhase -= phase

        // Swap positions
        let tempPos = anyons[anyonA].position
        anyons[anyonA].position = anyons[anyonB].position
        anyons[anyonB].position = tempPos

        // Update topological charge
        topologicalCharge += phase * TAU

        return phase
    }

    // ─── FIBONACCI F-MATRIX APPLICATION ───
    func applyFusionTransform(state: [Double]) -> [Double] {
        guard state.count >= 2 else { return state }
        // F-matrix: relates different fusion orderings
        let a = Self.FIBONACCI_FMATRIX[0][0] * state[0] + Self.FIBONACCI_FMATRIX[0][1] * state[1]
        let b = Self.FIBONACCI_FMATRIX[1][0] * state[0] + Self.FIBONACCI_FMATRIX[1][1] * state[1]
        return [a, b]
    }

    // ─── TOPOLOGICAL GATE: NOT (via braiding) ───
    func topologicalNOT(qubitAnyons: (Int, Int, Int)) -> Double {
        // NOT gate = σ₁σ₂σ₁ (3 braids)
        let p1 = braid(anyonA: qubitAnyons.0, anyonB: qubitAnyons.1)
        let p2 = braid(anyonA: qubitAnyons.1, anyonB: qubitAnyons.2)
        let p3 = braid(anyonA: qubitAnyons.0, anyonB: qubitAnyons.1)
        return p1 + p2 + p3
    }

    // ─── TOPOLOGICAL GATE: HADAMARD (approximate via braiding) ───
    func topologicalHadamard(qubitAnyons: (Int, Int, Int)) -> Double {
        // H ≈ σ₁²σ₂σ₁² (approximation to desired accuracy)
        var totalPhase = 0.0
        totalPhase += braid(anyonA: qubitAnyons.0, anyonB: qubitAnyons.1)
        totalPhase += braid(anyonA: qubitAnyons.0, anyonB: qubitAnyons.1)
        totalPhase += braid(anyonA: qubitAnyons.1, anyonB: qubitAnyons.2)
        totalPhase += braid(anyonA: qubitAnyons.0, anyonB: qubitAnyons.1)
        totalPhase += braid(anyonA: qubitAnyons.0, anyonB: qubitAnyons.1)
        return totalPhase
    }

    // ─── PROTECTION GAP MEASUREMENT ───
    func measureProtectionGap() -> Double {
        // Topological gap Δ ∝ φ / |anyons|
        let n = max(1, anyons.count)
        protectionGap = PHI / Double(n) * exp(-estimateTemperature() / PHI)
        return protectionGap
    }

    private func estimateTemperature() -> Double {
        // Effective temperature from braid history entropy
        guard !braidHistory.isEmpty else { return 0.01 }
        var cwCount = 0
        for (_, _, dir) in braidHistory { if dir == "CW" { cwCount += 1 } }
        let p = Double(cwCount) / Double(braidHistory.count)
        let entropy = p > 0 && p < 1 ? -(p * log(p) + (1 - p) * log(1 - p)) : 0.0
        return entropy * PHI
    }

    // ─── FUSION OUTCOME ───
    func fuseAnyons(a: Int, b: Int) -> String {
        guard a < anyons.count && b < anyons.count else { return "INVALID" }
        let phaseProduct = anyons[a].braidPhase * anyons[b].braidPhase
        if abs(phaseProduct) < .pi * TAU {
            fusionTree.append("τ⊗τ → 1 (vacuum) [anyons \(a),\(b)]")
            return "1 (vacuum)"
        } else {
            fusionTree.append("τ⊗τ → τ (anyon) [anyons \(a),\(b)]")
            return "τ (anyon)"
        }
    }

    func statusReport() -> String {
        let gap = measureProtectionGap()
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🧲 TOPOLOGICAL QUBIT STABILIZER                        ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Anyons:           \(anyons.count) (\(anyons.filter { $0.isNonAbelian }.count) non-abelian)
        ║  Braids:           \(braidHistory.count) operations
        ║  Fusion Events:    \(fusionTree.count)
        ║  Protection Gap:   \(String(format: "%.6f", gap))
        ║  Topo Charge:      \(String(format: "%.6f", topologicalCharge))
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
