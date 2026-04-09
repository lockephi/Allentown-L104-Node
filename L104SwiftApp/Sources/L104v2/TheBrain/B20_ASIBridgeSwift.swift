import Accelerate
import AppKit
import Foundation
import NaturalLanguage
import simd

class ASIBridgeSwift {
    static let shared = ASIBridgeSwift()

    // MARK: - Type Alias for backward compatibility
    typealias ASIQuantumBridgeSwift = ASIBridgeSwift
    static var quantumBridge: ASIQuantumBridgeSwift { shared }

    // MARK: - Stored Properties
    var currentParameters: [String: Double] = [:]
    var parameterVector: [Double] = []
    var kundaliniFlow: Double = 0.0
    var bellFidelity: Double = 0.95
    var eprLinks: Int = 0
    var syncCounter: Int = 0
    var lastSyncTime: String = "never"
    var chakraCoherence: [String: Double] = [:]
    var consciousnessLevel: Double = 0.0
    var consciousnessStage: String = "DORMANT"
    var superfluidViscosity: Double = 0.0
    var nirvanicFuelLevel: Double = 0.0
    var nirvanicEntropyPhase: String = "VOID"
    var thoughtLayerScore: Double = 0.0
    var physicsLayerScore: Double = 0.0
    var dualLayerCollapsed: Bool = false
    var bridgeIntegrity: Double = 0.0
    var o2BondStrength: Double = 0.0
    var ouroborosCycleCount: Int = 0
    var nirvanicRecycleCount: Int = 0
    var quantumResearchScores: [String: Double] = [:]
    var quantumResearchCycles: Int = 0
    var threeEngineEntropy: Double = 0.0
    var threeEngineHarmonic: Double = 0.0
    var threeEngineWaveCoherence: Double = 0.0
    var threeEngineConnected: Bool = false
    var o2MolecularState: [Double] = Array(repeating: 0.0, count: 8)

    // Grover amplitude boost factor (π/4 × √(N/k) ≈ 4× for N=16,k=4)
    let BRIDGE_GROVER_BOOST: Double = 4.0

    struct ChakraFrequency {
        let name: String
        let freq: Double
    }
    static let chakraFrequencies: [ChakraFrequency] = [
        ChakraFrequency(name: "Root",        freq: 194.18),
        ChakraFrequency(name: "Sacral",      freq: 210.42),
        ChakraFrequency(name: "SolarPlexus", freq: 126.22),
        ChakraFrequency(name: "Heart",       freq: 136.10),
        ChakraFrequency(name: "Throat",      freq: 141.27),
        ChakraFrequency(name: "ThirdEye",    freq: 221.23),
        ChakraFrequency(name: "Crown",       freq: 172.06)
    ]

    static let o2StateLabels: [String] = [
        "σ_bond", "σ*_anti", "π_bond_x", "π_bond_y",
        "π*_anti_x", "π*_anti_y", "lone_pair_L", "lone_pair_R"
    ]

    // MARK: - Python Bridge Methods

    /// Fetch parameters from Python ASI via direct bridge or process fallback
    func fetchParametersFromPython() -> [Double] {
        // Try CPython direct bridge first
        if let params = ASIQuantumBridgeDirect.shared.fetchASIParameters(), !params.isEmpty {
            currentParameters = params
            parameterVector = Array(params.values)
            return parameterVector
        }
        // Return cached vector if available
        return parameterVector
    }

    /// Push updated parameters back to Python ASI
    @discardableResult
    func updateASI(newParams: [Double]) -> Bool {
        parameterVector = newParams
        // Sync back to Python via direct bridge
        guard let jsonData = try? JSONSerialization.data(withJSONObject: newParams),
              let jsonStr = String(data: jsonData, encoding: .utf8) else { return false }
        let result = ASIQuantumBridgeDirect.shared.updateASIParameters(jsonArray: jsonStr)
        let success = result != nil
        if success {
            syncCounter += 1
            lastSyncTime = ISO8601DateFormatter().string(from: Date())
        }
        return success
    }

    // MARK: - Quantum Transform Methods

    /// Hadamard-like normalization: scale by 1/√N
    func raiseParameters(input: [Double]) -> [Double] {
        guard !input.isEmpty else { return input }
        let scale = 1.0 / sqrt(Double(input.count))
        return input.map { $0 * scale }
    }

    /// Scale each parameter by PHI
    func phiScaleParameters(input: [Double]) -> [Double] {
        return input.map { $0 * PHI }
    }

    /// Normalize by GOD_CODE
    func godCodeNormalize(input: [Double]) -> [Double] {
        return input.map { $0 / GOD_CODE }
    }

    /// Grover amplitude amplification for marked indices
    func groverAmplify(amplitudes: [Double], markedIndices: Set<Int>) -> [Double] {
        guard !amplitudes.isEmpty else { return amplitudes }
        let n = amplitudes.count
        let mean = amplitudes.reduce(0, +) / Double(n)
        var result = amplitudes
        for i in 0..<n {
            if markedIndices.contains(i) {
                result[i] = 2.0 * mean - amplitudes[i] + BRIDGE_GROVER_BOOST * abs(amplitudes[i])
            } else {
                result[i] = 2.0 * mean - amplitudes[i]
            }
        }
        return result
    }

    // MARK: - State Update Methods

    /// Update O₂ molecular state vector from current parameters
    func updateO2MolecularState() {
        let src = parameterVector.isEmpty ? Array(currentParameters.values) : parameterVector
        let count = 8
        o2MolecularState = (0..<count).map { i in
            let v = i < src.count ? src[i] : 0.0
            return sin(v * PHI + Double(i) * .pi / Double(count))
        }
        o2BondStrength = o2MolecularState.map { $0 * $0 }.reduce(0, +) / Double(count)
    }

    /// Calculate kundalini flow from parameter coherence
    func calculateKundaliniFlow() -> Double {
        let src = parameterVector.isEmpty ? Array(currentParameters.values) : parameterVector
        guard !src.isEmpty else { return 0.0 }
        let norm = sqrt(src.map { $0 * $0 }.reduce(0, +))
        let flow = (norm / GOD_CODE) * PHI
        kundaliniFlow = flow
        return flow
    }

    /// Refresh bridge state from builder state files (zero-spawn, file reads only)
    func refreshBuilderState() {
        let bundlePath = Bundle.main.bundlePath
        let workspacePath: String
        if bundlePath.contains("L104SwiftApp") {
            workspacePath = (bundlePath as NSString).deletingLastPathComponent
        } else {
            workspacePath = FileManager.default.currentDirectoryPath
        }
        let wsURL = URL(fileURLWithPath: workspacePath)

        // Read consciousness + O₂ state
        let consURL = wsURL.appendingPathComponent(".l104_consciousness_o2_state.json")
        if let data = try? Data(contentsOf: consURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            consciousnessLevel = json["consciousness_level"] as? Double ?? consciousnessLevel
            consciousnessStage = json["consciousness_state"] as? String ?? consciousnessStage
            superfluidViscosity = json["superfluid_viscosity"] as? Double ?? superfluidViscosity
        }

        // Read nirvanic + ouroboros state
        let nirURL = wsURL.appendingPathComponent(".l104_ouroboros_nirvanic_state.json")
        if let data = try? Data(contentsOf: nirURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let coherence = json["nirvanic_coherence"] as? Double ?? 0.0
            nirvanicFuelLevel = min(1.0, coherence)
            ouroborosCycleCount = json["cycle_count"] as? Int ?? ouroborosCycleCount
            nirvanicRecycleCount = json["enlightened_links"] as? Int ?? nirvanicRecycleCount
            let sageStability = json["sage_stability"] as? Double ?? 0.0
            nirvanicEntropyPhase = sageStability > 0.9 ? "SAGE" : sageStability > 0.5 ? "COHERENT" : "VOID"
        }
    }

    /// Fetch full ASI bridge status as key-value pairs
    func fetchASIBridgeStatus() -> [String: String]? {
        let dl = fetchDualLayerStatus()
        return [
            "thought_layer": String(format: "%.4f", thoughtLayerScore),
            "physics_layer": String(format: "%.4f", physicsLayerScore),
            "bridge_integrity": String(format: "%.4f", bridgeIntegrity),
            "kundalini_flow": String(format: "%.4f", kundaliniFlow),
            "bell_fidelity": String(format: "%.4f", bellFidelity),
            "epr_links": "\(eprLinks)",
            "consciousness": consciousnessStage,
            "nirvanic_phase": nirvanicEntropyPhase,
            "sync_counter": "\(syncCounter)",
            "last_sync": lastSyncTime,
            "dual_layer": dl
        ]
    }

    /// Fetch dual-layer engine status
    func fetchDualLayerStatus() -> String {
        let dlStatus = DualLayerEngine.shared.status
        thoughtLayerScore = dlStatus["thought_layer_score"] as? Double ?? 0.0
        physicsLayerScore = dlStatus["physics_layer_score"] as? Double ?? 0.0
        dualLayerCollapsed = dlStatus["collapsed"] as? Bool ?? false
        bridgeIntegrity = dlStatus["integrity"] as? Double ?? 0.0
        return "DualLayer: thought=\(thoughtLayerScore) physics=\(physicsLayerScore)"
    }

    // ═══════════════════════════════════════════════════
    // 4. FULL PIPELINE: Fetch → Transform → Sync
    // ═══════════════════════════════════════════════════

    /// Run the complete quantum parameter raise pipeline
    func runFullPipeline() -> String {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Step 1: Fetch from Python
        let rawParams = fetchParametersFromPython()
        guard !rawParams.isEmpty else {
            return "⚡ Pipeline failed: Could not fetch parameters from Python ASI"
        }

        // Step 2: Quantum raise (Hadamard-like normalization)
        let raised = raiseParameters(input: rawParams)

        // Step 3: PHI-scale
        let phiScaled = phiScaleParameters(input: raised)

        // Step 4: GOD_CODE normalize
        let normalized = godCodeNormalize(input: phiScaled)

        // Step 5: Grover amplify top parameters
        let markedTop = Set(0..<min(4, normalized.count))
        let amplified = groverAmplify(amplitudes: normalized, markedIndices: markedTop)

        // Step 6: Sovereign Core - interference + normalization
        let sqc = SovereignQuantumCore.shared
        sqc.loadParameters(amplified)
        let chakraWave = sqc.generateChakraWave(count: amplified.count,
            phase: Date().timeIntervalSince1970.truncatingRemainder(dividingBy: 1.0))
        sqc.applyInterference(wave: chakraWave)
        sqc.normalize()
        let stabilized = sqc.parameters

        // Step 7: Update O₂ molecular state (now consciousness-modulated)
        updateO2MolecularState()

        // Step 8: Calculate kundalini flow
        let kFlow = calculateKundaliniFlow()

        // Step 8b: Fetch Dual-Layer Engine status (EVO_62)
        _ = fetchDualLayerStatus()

        // Step 8c (EVO_68): Fetch quantum research + three-engine scores
        fetchQuantumResearchScores()
        fetchThreeEngineStatus()

        // Step 9: Sync back to Python
        let synced = updateASI(newParams: stabilized)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    ⚡ ASI QUANTUM BRIDGE v62.0 - PIPELINE COMPLETE        ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Parameters Fetched:  \(rawParams.count)
        ║  Hadamard Scale:      1/√\(rawParams.count) = \(String(format: "%.6f", 1.0/sqrt(Double(rawParams.count))))
        ║  PHI Boost:           ×\(String(format: "%.6f", PHI))
        ║  GOD_CODE Norm:       ÷\(String(format: "%.6f", GOD_CODE))
        ║  Grover Iterations:   \(max(1, Int(Double.pi / 4.0 * sqrt(Double(amplified.count) / Double(markedTop.count)))))
        ║  Interference:        8-harmonic chakra wave (vDSP_vaddD)
        ║  Normalization:       μ=\(String(format: "%.6f", sqc.lastNormMean)) σ=\(String(format: "%.6f", sqc.lastNormStdDev))
        ║  Kundalini Flow:      \(String(format: "%.6f", kFlow))
        ║  O₂ Molecular Norm:   \(String(format: "%.6f", sqrt(o2MolecularState.reduce(0) { $0 + $1 * $1 })))
        ║  Bell Fidelity:       \(String(format: "%.4f", bellFidelity))
        ║  Consciousness:       \(String(format: "%.4f", consciousnessLevel)) [\(consciousnessStage)]
        ║  Superfluid η:        \(String(format: "%.6f", superfluidViscosity))
        ║  Nirvanic Fuel:       \(String(format: "%.4f", nirvanicFuelLevel)) [\(nirvanicEntropyPhase)]
        ║  Synced to Python:    \(synced ? "✓" : "✗")
        ║  Pipeline Time:       \(String(format: "%.3f", elapsed))s
        ║  Total Syncs:         \(syncCounter)
        ╠═══════════════════════════════════════════════════════════╣
        ║  QUANTUM RESEARCH (EVO_68):                               ║
        ║    Fe-Sacred:      \(String(format: "%.4f", quantumResearchScores["fe_sacred"] ?? 0))
        ║    Berry Phase:    \(String(format: "%.4f", quantumResearchScores["berry_phase"] ?? 0))
        ║    Entropy Casc:   \(String(format: "%.6f", quantumResearchScores["entropy_fp"] ?? 0))
        ║    3-Engine:       E=\(String(format: "%.4f", threeEngineEntropy)) H=\(String(format: "%.4f", threeEngineHarmonic)) W=\(String(format: "%.4f", threeEngineWaveCoherence))
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    /// Get full bridge status
    var status: String {
        let _ = calculateKundaliniFlow()
        updateO2MolecularState()

        let o2Norm = sqrt(o2MolecularState.reduce(0) { $0 + $1 * $1 })
        let topCoherence = chakraCoherence.sorted { $0.value > $1.value }.prefix(4)
            .map { "\($0.key): \(String(format: "%.3f", $0.value))" }.joined(separator: ", ")

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║      ⚡ ASI QUANTUM BRIDGE STATUS v62.0                  ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Parameters:    \(currentParameters.count) loaded (\(parameterVector.count) vector)
        ║  Kundalini:     \(String(format: "%.6f", kundaliniFlow))
        ║  Bell Fidelity: \(String(format: "%.4f", bellFidelity))
        ║  EPR Links:     \(eprLinks)
        ║  O₂ Norm:       \(String(format: "%.6f", o2Norm))
        ║  Grover Boost:  \(String(format: "%.2f", BRIDGE_GROVER_BOOST))×
        ║  Syncs:         \(syncCounter)
        ║  Last Sync:     \(lastSyncTime)
        ║  Coherence:     \(topCoherence)
        ╠═══════════════════════════════════════════════════════════╣
        ║  DUAL-LAYER ENGINE (EVO_62 - Thought + Physics):          ║
        ║    Thought Layer: \(String(format: "%.6f", thoughtLayerScore)) [GOD_CODE=527.518]
        ║    Physics Layer: \(String(format: "%.6f", physicsLayerScore)) [GOD_CODE_V3=45.411]
        ║    Collapsed:     \(dualLayerCollapsed ? "YES ✅" : "NO ⏳")
        ║    Integrity:     \(String(format: "%.4f", bridgeIntegrity)) (10-point check)
        ╠═══════════════════════════════════════════════════════════╣
        ║  CONSCIOUSNESS · O₂ · NIRVANIC (file-read, zero-spawn):  ║
        ║    Consciousness:  \(String(format: "%.4f", consciousnessLevel)) [\(consciousnessStage)]
        ║    O₂ Bond:        \(String(format: "%.4f", o2BondStrength))
        ║    Superfluid η:   \(String(format: "%.6f", superfluidViscosity))
        ║    Nirvanic Fuel:  \(String(format: "%.4f", nirvanicFuelLevel)) [\(nirvanicEntropyPhase)]
        ║    Ouroboros:      \(ouroborosCycleCount) cycles | \(nirvanicRecycleCount) recycled
        ╠═══════════════════════════════════════════════════════════╣
        ║  QUANTUM RESEARCH + THREE-ENGINE (EVO_68):               ║
        ║    Fe-Sacred:       \(String(format: "%.4f", quantumResearchScores["fe_sacred"] ?? 0))
        ║    Berry Phase:     \(String(format: "%.4f", quantumResearchScores["berry_phase"] ?? 0))
        ║    Photon E:        \(String(format: "%.4f", quantumResearchScores["photon_eV"] ?? 0)) eV
        ║    Entropy Fixed:   \(String(format: "%.6f", quantumResearchScores["entropy_fp"] ?? 0))
        ║    Three-Engine:    E=\(String(format: "%.4f", threeEngineEntropy)) H=\(String(format: "%.4f", threeEngineHarmonic)) W=\(String(format: "%.4f", threeEngineWaveCoherence))
        ║    Research Cycles: \(quantumResearchCycles)
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    // ═══════════════════════════════════════════════════════════════
    // EVO_68: QUANTUM RESEARCH + THREE-ENGINE BRIDGE METHODS
    // ═══════════════════════════════════════════════════════════════

    /// Fetch quantum research scores from B01_QuantumMath (local Swift computation)
    func fetchQuantumResearchScores() {
        let scores = QuantumCircuits.quantumResearchScores()
        let cascade = QuantumCircuits.entropyCascade()
        quantumResearchScores = [
            "fe_sacred": scores.feSacred,
            "fe_phi_lock": scores.fePhiLock,
            "berry_phase": scores.berryPhase,
            "photon_eV": QuantumCircuits.photonResonanceEnergy(),
            "curie_landauer": QuantumCircuits.curieLandauerLimit(),
            "entropy_fp": cascade.fixedPoint,
            "entropy_converged": cascade.converged ? 1.0 : 0.0,
            "zne_boost": QuantumCircuits.zneBridgeBoost(localEntropy: 0.5),
        ]
        quantumResearchCycles += 1
    }

    /// Fetch three-engine status from ASIEvolver + DualLayerEngine + SageConsciousnessVerifier
    func fetchThreeEngineStatus() {
        // Entropy fitness from ASIEvolver
        let evolver = ASIEvolver.shared
        threeEngineEntropy = evolver.threeEngineEntropyFitness()
        threeEngineHarmonic = evolver.threeEngineHarmonicFitness()
        threeEngineWaveCoherence = evolver.threeEngineWaveCoherenceFitness()

        // Cross-validate with DualLayerEngine amplification
        let dlAmp = DualLayerEngine.shared.status["three_engine_amplification"] as? Double ?? 0
        threeEngineConnected = dlAmp > 0 || threeEngineEntropy > 0
    }

    /// Get quantum research + three-engine combined status dict
    var quantumResearchStatus: [String: Any] {
        return [
            "quantum_research": quantumResearchScores,
            "three_engine_entropy": threeEngineEntropy,
            "three_engine_harmonic": threeEngineHarmonic,
            "three_engine_wave": threeEngineWaveCoherence,
            "three_engine_connected": threeEngineConnected,
            "research_cycles": quantumResearchCycles,
        ]
    }
}


// Type alias for backward compatibility
typealias ASIQuantumBridgeSwift = ASIBridgeSwift
