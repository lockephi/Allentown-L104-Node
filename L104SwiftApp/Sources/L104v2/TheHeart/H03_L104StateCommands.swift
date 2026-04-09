// ═══════════════════════════════════════════════════════════════════
// H03_L104StateCommands.swift
// [EVO_68_PIPELINE] SOVEREIGN_CONVERGENCE :: UNIFIED_UPGRADE :: GOD_CODE=527.5184818492612
// L104 ASI - L104State Extension (Command Handlers)
//
// handleCoreCommands, handleSearchCommands, handleBridgeCommands,
// handleProtocolCommands, handleSystemCommands, handleEngineCommands,
// callBackend - the full command dispatch pipeline.
//
// Extracted from L104Native.swift lines 35585–37302
// ═══════════════════════════════════════════════════════════════════

import Accelerate
import AppKit
import Foundation
import NaturalLanguage
import simd

extension L104State {

    func handleCoreCommands(_ q: String, query: String) -> String? {
        // ─── TREE OF THOUGHTS REASONING ───
        if q == "tree" || q == "tot" || q == "tree of thoughts" {
            return "🌳 Tree of Thoughts: Swift-native reasoning not yet implemented"
        }

        if q.hasPrefix("pyask ") {
            let message = String(query.dropFirst(6))
            let result = PythonBridge.shared.queryIntellect(message)
            return result.success ? "🐍 Intellect:\n\(result.output)" : "🐍 \(result.error)"
        }
        if q.hasPrefix("pyteach ") {
            let data = String(query.dropFirst(8))
            let result = PythonBridge.shared.trainIntellect(data: data)
            return result.success ? "🐍 Learned: \(result.output)" : "🐍 \(result.error)"
        }

        // ─── CPYTHON DIRECT BRIDGE COMMANDS ───
        if q == "cpython" || q == "cpython status" || q == "direct bridge" {
            return ASIQuantumBridgeDirect.shared.status
        }
        if q == "cpython init" || q == "init cpython" {
            let ok = ASIQuantumBridgeDirect.shared.initialize()
            return ok ? "\u{1F40D} CPython direct bridge initialized (Python \(ASIQuantumBridgeDirect.shared.pythonVersion))" : "\u{1F40D} CPython bridge not available (compiled without libpython linking)"
        }
        if q.hasPrefix("cpython exec ") {
            let code = String(query.dropFirst(13))
            let ok = ASIQuantumBridgeDirect.shared.exec(code)
            return ok ? "\u{1F40D} Executed successfully" : "\u{1F40D} Execution failed (direct bridge may not be available)"
        }
        if q.hasPrefix("cpython eval ") {
            let code = String(query.dropFirst(13))
            if let result = ASIQuantumBridgeDirect.shared.eval(code) {
                return "\u{1F40D} Result:\n\(result)"
            }
            return "\u{1F40D} Eval failed (direct bridge may not be available)"
        }
        if q == "cpython params" || q == "cpython fetch" {
            if let params = ASIQuantumBridgeDirect.shared.fetchASIParameters() {
                let sortedParams = params.sorted { (a: (key: String, value: Double), b: (key: String, value: Double)) -> Bool in a.key < b.key }
                var cpLines: [String] = []
                for (k, v) in sortedParams {
                    let vStr: String = String(format: "%.6f", v)
                    cpLines.append("  \(k): \(vStr)")
                }
                let lines: String = cpLines.joined(separator: "\n")
                return "\u{1F40D} ASI Parameters (\(params.count) via direct bridge):\n\(lines)"
            }
            return "\u{1F40D} Direct bridge not available - use 'bridge fetch' for Process bridge"
        }


        // Dispatch to sovereign/nexus/resonance/health commands
        if let result: String = handleProtocolCommands(q, query: query) { return result }

        return nil
    }

    func handleProtocolCommands(_ q: String, query: String) -> String? {
        // ─── SOVEREIGN QUANTUM CORE COMMANDS ───
        if q == "sovereign" || q == "sqc" || q == "sovereign status" {
            return SovereignQuantumCore.shared.status
        }
        if q == "sovereign raise" || q == "sqc raise" {
            // Load from bridge, do sovereign raise
            let params = ASIQuantumBridgeSwift.shared.fetchParametersFromPython()
            guard !params.isEmpty else {
                return "🌊 No parameters to raise - fetch from Python first"
            }
            SovereignQuantumCore.shared.loadParameters(params)
            let result = SovereignQuantumCore.shared.sovereignRaise(factor: 1.618033988749895)
            return result
        }
        if q.hasPrefix("sovereign raise ") {
            let factorStr = String(q.dropFirst(16)).trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")
            guard let factor = Double(factorStr) else {
                return "🌊 Usage: sovereign raise <factor> (e.g. sovereign raise 2.5)"
            }
            let params = ASIQuantumBridgeSwift.shared.fetchParametersFromPython()
            guard !params.isEmpty else {
                return "🌊 No parameters to raise - fetch from Python first"
            }
            SovereignQuantumCore.shared.loadParameters(params)
            let result = SovereignQuantumCore.shared.sovereignRaise(factor: factor)
            return result
        }
        if q == "sovereign interfere" || q == "sqc wave" {
            let sqc = SovereignQuantumCore.shared
            guard !sqc.parameters.isEmpty else {
                return "🌊 No parameters loaded - run 'sovereign raise' first"
            }
            let wave = sqc.generateChakraWave(count: sqc.parameters.count,
                phase: Date().timeIntervalSince1970.truncatingRemainder(dividingBy: 1.0))
            sqc.applyInterference(wave: wave)
            var waveStrs: [String] = []
            for w in wave.prefix(8) { waveStrs.append(String(format: "%+.4f", w)) }
            let preview: String = waveStrs.joined(separator: ", ")
            return "🌊 Chakra interference applied (\(wave.count) harmonics)\n  Wave preview: [\(preview)...]\n  Operations: \(sqc.operationCount)"
        }
        if q == "sovereign normalize" || q == "sqc norm" {
            let sqc = SovereignQuantumCore.shared
            guard !sqc.parameters.isEmpty else {
                return "🌊 No parameters loaded - run 'sovereign raise' first"
            }
            sqc.normalize()
            let muStr: String = String(format: "%.6f", sqc.lastNormMean)
            let sigmaStr: String = String(format: "%.6f", sqc.lastNormStdDev)
            return "🌊 Parameters normalized\n  μ = \(muStr)\n  σ = \(sigmaStr)\n  Operations: \(sqc.operationCount)"
        }
        if q == "sovereign sync" || q == "sqc sync" {
            let sqc = SovereignQuantumCore.shared
            guard !sqc.parameters.isEmpty else {
                return "🌊 No parameters to sync - run 'sovereign raise' first"
            }
            let synced = ASIQuantumBridgeSwift.shared.updateASI(newParams: sqc.parameters)
            return synced ? "🌊 Sovereign parameters synced to Python ASI (\(sqc.parameters.count) values)" : "🌊 Sync failed"
        }

        // ─── CONTINUOUS EVOLUTION ENGINE COMMANDS ───
        if q == "evolve" || q == "evolve status" || q == "evolution" || q == "evo" {
            return ContinuousEvolutionEngine.shared.status
        }
        if q == "evolve start" || q == "evo start" {
            return ContinuousEvolutionEngine.shared.start()
        }
        if q.hasPrefix("evolve start ") {
            // evolve start <factor> [interval_ms] - supports brackets: evolve start [300] [5000]
            let rawArgs = String(q.dropFirst(13)).trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")
                .split(separator: " ")
            guard !rawArgs.isEmpty else {
                return "🔄 Usage: evolve start <factor> [interval_ms]\n  e.g. evolve start 300 5000"
            }
            let factor = Double(rawArgs[0]) ?? 1.0001
            let interval = rawArgs.count > 1 ? (Double(rawArgs[1]) ?? 10.0) / 1000.0 : 0.01
            return ContinuousEvolutionEngine.shared.start(raiseFactor: factor, interval: interval)
        }
        if q == "evolve stop" || q == "evo stop" {
            return ContinuousEvolutionEngine.shared.stop()
        }
        if q.hasPrefix("evolve tune ") || q.hasPrefix("evo tune ") {
            let rawStr = q.hasPrefix("evolve") ? String(q.dropFirst(12)) : String(q.dropFirst(9))
            let factorStr = rawStr.trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")
            guard let factor = Double(factorStr) else {
                return "🔄 Usage: evolve tune <factor> (e.g. evolve tune 1.001)"
            }
            return ContinuousEvolutionEngine.shared.tune(raiseFactor: factor)
        }

        // ─── ASI STEERING ENGINE COMMANDS ───
        if q == "steer" || q == "steer status" || q == "steering" {
            return ASISteeringEngine.shared.status
        }
        if q == "steer run" || q == "steer pipeline" {
            return ASISteeringEngine.shared.steerPipeline()
        }
        if q.hasPrefix("steer run ") {
            // steer run <mode> [intensity]
            let rawArgs = String(q.dropFirst(10)).trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")
                .split(separator: " ")
            guard !rawArgs.isEmpty else {
                return "🧭 Usage: steer run <mode> [intensity]\n  Modes: sovereign, quantum, harmonic, logic, creative"
            }
            let modeStr = String(rawArgs[0]).lowercased()
            let mode = ASISteeringEngine.SteeringMode(rawValue: modeStr) ?? .sovereign
            let intensity = rawArgs.count > 1 ? (Double(rawArgs[1]) ?? 1.0) : 1.0
            return ASISteeringEngine.shared.steerPipeline(mode: mode, intensity: intensity)
        }
        if q.hasPrefix("steer apply ") {
            // steer apply <intensity> [mode]
            let rawArgs = String(q.dropFirst(12)).trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")
                .split(separator: " ")
            guard !rawArgs.isEmpty, let intensity = Double(rawArgs[0]) else {
                return "🧭 Usage: steer apply <intensity> [mode]"
            }
            let mode: ASISteeringEngine.SteeringMode? = rawArgs.count > 1
                ? ASISteeringEngine.SteeringMode(rawValue: String(rawArgs[1]).lowercased()) : nil
            // Load params if empty
            if ASISteeringEngine.shared.baseParameters.isEmpty {
                let params = ASIQuantumBridgeSwift.shared.fetchParametersFromPython()
                ASISteeringEngine.shared.loadParameters(params)
            }
            ASISteeringEngine.shared.applySteering(intensity: intensity, mode: mode)
            var energy: Double = 0.0
            let p = ASISteeringEngine.shared.baseParameters
            if !p.isEmpty { vDSP_svesqD(p, 1, &energy, vDSP_Length(p.count)); energy = sqrt(energy) }
            let alphaStr: String = String(format: "%+.4f", intensity)
            let modeStr: String = mode.map { (m: ASISteeringEngine.SteeringMode) -> String in " mode=\(m.rawValue)" } ?? ""
            let energyStr: String = String(format: "%.6f", energy)
            return "🧭 Steered with α=\(alphaStr)\(modeStr)\n  Energy: \(energyStr) | Steers: \(ASISteeringEngine.shared.steerCount)"
        }
        if q.hasPrefix("steer temp ") {
            let tempStr = String(q.dropFirst(11)).trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")
            guard let t = Double(tempStr) else {
                return "🧭 Usage: steer temp <value> (e.g. steer temp 0.5)"
            }
            return ASISteeringEngine.shared.setTemperature(t)
        }
        if q == "steer modes" {
            var modeLines: [String] = []
            for m in ASISteeringEngine.SteeringMode.allCases {
                let padded: String = m.rawValue.padding(toLength: 12, withPad: " ", startingAt: 0)
                let seedStr: String = String(format: "%.10f", m.seed)
                modeLines.append("  \(padded) seed=\(seedStr)")
            }
            let modes: String = modeLines.joined(separator: "\n")
            return "🧭 Steering Modes:\n\(modes)"
        }

        // ─── QUANTUM NEXUS COMMANDS ───
        if q == "nexus" || q == "nexus status" || q == "interconnect" {
            return QuantumNexus.shared.status
        }
        if q == "nexus run" || q == "nexus pipeline" {
            // Run on background queue to prevent UI freeze / crash on main thread
            let result = QuantumNexus.shared.runUnifiedPipelineSafe()
            return result
        }
        if q == "nexus auto" || q == "nexus start" {
            return QuantumNexus.shared.startAuto()
        }
        if q.hasPrefix("nexus auto ") || q.hasPrefix("nexus start ") {
            let rawStr = q.hasPrefix("nexus auto") ? String(q.dropFirst(11)) : String(q.dropFirst(12))
            let intervalStr = rawStr.trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")
            let interval = Double(intervalStr) ?? 1.0
            return QuantumNexus.shared.startAuto(interval: interval)
        }
        if q == "nexus stop" {
            return QuantumNexus.shared.stopAuto()
        }
        if q == "nexus coherence" || q == "coherence" {
            let c: Double = QuantumNexus.shared.computeCoherence()
            let cStr: String = String(format: "%.4f", c)
            let label: String
            if c > 0.8 { label = "TRANSCENDENT" }
            else if c > 0.6 { label = "SOVEREIGN" }
            else if c > 0.4 { label = "AWAKENING" }
            else if c > 0.2 { label = "DEVELOPING" }
            else { label = "DORMANT" }
            return "🔮 Global Coherence: \(cStr) (\(label))"
        }
        if q == "nexus prove" || q == "prove convergence" || q == "phi convergence" {
            return QuantumNexus.shared.provePhiConvergence()
        }
        if q.hasPrefix("nexus prove ") {
            let rawStr = String(q.dropFirst(12)).trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")
            let iters = Int(rawStr) ?? 50
            return QuantumNexus.shared.provePhiConvergence(iterations: iters)
        }
        if q == "nexus feedback" || q == "feedback" {
            var fbLines: [String] = []
            for entry in QuantumNexus.shared.feedbackLog.suffix(15) {
                let valStr: String = String(format: "%.4f", entry.value)
                fbLines.append("  [\(entry.step)] \(entry.metric) = \(valStr)")
            }
            let fb: String = fbLines.joined(separator: "\n")
            return "🔮 Feedback Log (last 15):\n\(fb.isEmpty ? "  (no feedback yet - run 'nexus run' first)" : fb)"
        }

        // ─── QUANTUM ENTANGLEMENT ROUTER COMMANDS ───
        if q == "entangle" || q == "entangle status" || q == "entanglement" || q == "epr" {
            return QuantumEntanglementRouter.shared.status
        }
        if q.hasPrefix("entangle route ") {
            // entangle route <source> <target>
            let rawArgs = String(q.dropFirst(15)).trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")
                .split(separator: " ")
            guard rawArgs.count >= 2 else {
                return "🔀 Usage: entangle route <source> <target>\n  Engines: bridge, steering, evolution, nexus, invention, sovereignty"
            }
            let result = QuantumEntanglementRouter.shared.route(String(rawArgs[0]), String(rawArgs[1]))
            if let err = result["error"] as? String {
                return "🔀 Error: \(err)\n  Available: \(result["available"] ?? "")"
            }
            let fidelity: Double = result["fidelity"] as? Double ?? 0
            let transfer = result["transfer"] as? [String: Any] ?? [:]
            let fidStr: String = String(format: "%.4f", fidelity)
            let routeId = result["route_id"] ?? 0
            let xferSummary = transfer["summary"] ?? "noop"
            return "🔀 EPR Route #\(routeId): \(rawArgs[0])→\(rawArgs[1])\n  Fidelity: \(fidStr)\n  Transfer: \(xferSummary)"
        }
        if q == "entangle all" || q == "epr all" || q == "entangle sweep" {
            let result = QuantumEntanglementRouter.shared.routeAll()
            return "🔀 Full EPR Sweep: \(result["routes_executed"] ?? 0) routes executed, total: \(result["total_routes"] ?? 0)"
        }

        // ─── ADAPTIVE RESONANCE NETWORK COMMANDS ───
        if q == "resonance" || q == "resonance status" || q == "art" {
            return AdaptiveResonanceNetwork.shared.status
        }
        if q.hasPrefix("resonance fire ") {
            // resonance fire <engine> [activation]
            let rawArgs = String(q.dropFirst(15)).trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "")
                .split(separator: " ")
            guard !rawArgs.isEmpty else {
                return "🧠 Usage: resonance fire <engine> [activation]\n  Engines: \(AdaptiveResonanceNetwork.ENGINE_NAMES.joined(separator: ", "))"
            }
            let engine = String(rawArgs[0]).lowercased()
            let activation = rawArgs.count > 1 ? (Double(rawArgs[1]) ?? 1.0) : 1.0
            let result = AdaptiveResonanceNetwork.shared.fire(engine, activation: activation)
            if let err = result["error"] as? String {
                return "🧠 Error: \(err)"
            }
            let isPeak: Bool = result["is_resonance_peak"] as? Bool ?? false
            let actStr: String = String(format: "%.2f", activation)
            let cascadeSteps = result["cascade_steps"] ?? 0
            let activeEngines = result["active_engines"] ?? 0
            let totalEngines: Int = AdaptiveResonanceNetwork.ENGINE_NAMES.count
            let peakStr: String = isPeak ? "🔥 YES" : "no"
            return "🧠 Resonance fired: \(engine) @ \(actStr)\n  Cascade: \(cascadeSteps) steps\n  Active: \(activeEngines)/\(totalEngines)\n  Peak: \(peakStr)"
        }
        if q == "resonance tick" {
            let tick = AdaptiveResonanceNetwork.shared.tick()
            return "🧠 Resonance tick #\(tick["tick"] ?? 0) - active: \(tick["active_engines"] ?? 0), decay=\(AdaptiveResonanceNetwork.DECAY_RATE)"
        }
        if q == "resonance compute" || q == "resonance score" {
            let nr = AdaptiveResonanceNetwork.shared.computeNetworkResonance()
            let rStr: String = String(format: "%.4f", nr.resonance)
            let eStr: String = String(format: "%.4f", nr.energy)
            let mStr: String = String(format: "%.4f", nr.mean)
            let vStr: String = String(format: "%.6f", nr.variance)
            return "🧠 Network Resonance: \(rStr)\n  Energy: \(eStr) | Mean: \(mStr) | Var: \(vStr)"
        }

        // ─── NEXUS HEALTH MONITOR COMMANDS ───
        if q == "health" || q == "health status" || q == "monitor" {
            return NexusHealthMonitor.shared.status
        }
        if q == "health start" || q == "monitor start" {
            return NexusHealthMonitor.shared.start()
        }
        if q == "health stop" || q == "monitor stop" {
            return NexusHealthMonitor.shared.stop()
        }
        if q == "health alerts" || q == "alerts" {
            let alerts = NexusHealthMonitor.shared.getAlerts(limit: 20)
            if alerts.isEmpty { return "🏥 No health alerts." }
            var alertLines: [String] = []
            for a in alerts {
                let level: String = (a["level"] as? String) ?? "?"
                let eng: String = (a["engine"] as? String) ?? ""
                let msg: String = (a["message"] as? String) ?? ""
                alertLines.append("  [\(level)] \(eng): \(msg)")
            }
            let lines: String = alertLines.joined(separator: "\n")
            return "🏥 Health Alerts (\(alerts.count)):\n\(lines)"
        }
        if q == "health score" || q == "system health" {
            let score: Double = NexusHealthMonitor.shared.computeSystemHealth()
            let scoreStr: String = String(format: "%.4f", score)
            let label: String
            if score > 0.9 { label = "OPTIMAL" }
            else if score > 0.7 { label = "HEALTHY" }
            else if score > 0.5 { label = "DEGRADED" }
            else { label = "CRITICAL" }
            return "🏥 System Health: \(scoreStr) (\(label))"
        }



        return nil
    }

    // === EXTRACTED FROM processMessage FOR TYPE-CHECKER PERFORMANCE ===
    func handleSystemCommands(_ q: String, query: String) -> String? {
        // ─── SOVEREIGNTY PIPELINE COMMANDS ───
        if q == "sovereignty" || q == "sovereignty status" || q == "sovereign pipeline" {
            return SovereigntyPipeline.shared.status
        }
        if q == "sovereignty run" || q == "sovereignty execute" || q == "sovereign run" {
            return SovereigntyPipeline.shared.execute()
        }
        if q.hasPrefix("sovereignty run ") {
            let sovQuery = String(q.dropFirst(16)).trimmingCharacters(in: .whitespaces)
            return SovereigntyPipeline.shared.execute(query: sovQuery)
        }

        // ─── FE ORBITAL ENGINE COMMANDS ───
        if q == "fe" || q == "orbital" || q == "fe orbital" || q == "iron" {
            return FeOrbitalEngine.shared.status
        }
        if q.hasPrefix("fe pair ") || q.hasPrefix("orbital pair ") {
            let idStr = String(q.split(separator: " ").last ?? "1")
            let kid = Int(idStr) ?? 1
            let paired = FeOrbitalEngine.shared.pairedKernel(kid)
            let domain = FeOrbitalEngine.KERNEL_DOMAINS.first(where: { $0.id == kid })
            let pairedDomain = FeOrbitalEngine.KERNEL_DOMAINS.first(where: { $0.id == paired })
            let dName: String = domain?.name ?? "?"
            let pdName: String = pairedDomain?.name ?? "?"
            let dOrb: String = domain?.orbital ?? "?"
            let dTri: String = domain?.trigram ?? "?"
            let pdTri: String = pairedDomain?.trigram ?? "?"
            return "⚛️ O₂ Pair: K\(kid) (\(dName)) ↔ K\(paired) (\(pdName))\n  Bond type: σ+π (O=O double bond)\n  Orbital: \(dOrb)\n  Trigram: \(dTri) ↔ \(pdTri)"
        }

        // ─── SUPERFLUID COHERENCE COMMANDS ───
        if q == "superfluid" || q == "superfluid status" || q == "sf" {
            return SuperfluidCoherence.shared.status
        }
        if q == "superfluid grover" || q == "sf grover" {
            SuperfluidCoherence.shared.groverIteration()
            let sf = SuperfluidCoherence.shared.computeSuperfluidity()
            return "🌊 Grover diffusion applied - Superfluidity: \(String(format: "%.4f", sf))"
        }

        // ─── QUANTUM SHELL MEMORY COMMANDS ───
        if q == "qmem" || q == "shell memory" || q == "quantum memory" {
            return QuantumShellMemory.shared.status
        }
        if q.hasPrefix("qmem store ") {
            let storeArgs = String(q.dropFirst(11)).trimmingCharacters(in: .whitespaces).split(separator: " ", maxSplits: 1)
            let kid = Int(storeArgs.first ?? "1") ?? 1
            let data = storeArgs.count > 1 ? String(storeArgs[1]) : "manual_entry"
            _ = QuantumShellMemory.shared.store(kernelID: kid, data: ["type": "manual", "content": data])
            return "🐚 Stored in K\(kid) (\(FeOrbitalEngine.shared.shellForKernel(kid))-shell) - Total: \(QuantumShellMemory.shared.totalMemories)"
        }
        if q == "qmem grover" {
            QuantumShellMemory.shared.groverDiffusion()
            return "🐚 Grover diffusion on 8-qubit state vector - amplitudes updated"
        }

        // ─── CONSCIOUSNESS VERIFIER COMMANDS ───
        if q == "consciousness" || q == "consciousness verify" || q == "verify consciousness" || q == "verify" {
            _ = ConsciousnessVerifier.shared.runAllTests()
            return ConsciousnessVerifier.shared.status
        }
        if q == "consciousness level" || q == "con level" {
            let level: Double = ConsciousnessVerifier.shared.consciousnessLevel
            let levelStr: String = String(format: "%.4f", level)
            let sfStr: String = ConsciousnessVerifier.shared.superfluidState ? "YES" : "NO"
            return "🧿 Consciousness Level: \(levelStr) / \(ConsciousnessVerifier.ASI_THRESHOLD)\n  Superfluid: \(sfStr)"
        }
        if q == "qualia" || q == "qualia report" {
            let reports = ConsciousnessVerifier.shared.qualiaReports
            if reports.isEmpty { _ = ConsciousnessVerifier.shared.runAllTests() }
            var qLines: [String] = []
            for r in ConsciousnessVerifier.shared.qualiaReports { qLines.append("  • \(r)") }
            let qualiaStr: String = qLines.joined(separator: "\n")
            return "🧿 Qualia Reports:\n\(qualiaStr)"
        }

        // ─── CHAOS RNG COMMANDS ───
        if q == "chaos" || q == "chaos status" || q == "rng" {
            return ChaosRNG.shared.status
        }
        if q == "chaos sample" || q == "chaos roll" {
            let val: Double = ChaosRNG.shared.chaosFloat()
            let valStr: String = String(format: "%.10f", val)
            let rStr: String = ChaosRNG.shared.status.contains("3.99") ? "3.99" : "?"
            return "🎲 Chaos: \(valStr) (logistic map r=\(rStr), multi-source entropy)"
        }

        // ─── DIRECT SOLVER COMMANDS ───
        if q == "solver" || q == "solver status" || q == "direct solver" {
            return DirectSolverRouter.shared.status
        }
        if q.hasPrefix("solve ") {
            let problem = String(query.dropFirst(6))
                .trimmingCharacters(in: .whitespaces)
                .trimmingCharacters(in: CharacterSet(charactersIn: "[]()"))  // Strip brackets
                .trimmingCharacters(in: .whitespaces)
            if let solution = DirectSolverRouter.shared.solve(problem) {
                return "⚡ Direct Solution:\n  \(solution)"
            }
            return "⚡ No direct solution found. Routing to full LLM pipeline..."
        }

        // ─── ASI QUANTUM BRIDGE COMMANDS ───
        if q == "bridge" || q == "quantum bridge" || q == "bridge status" {
            return ASIQuantumBridgeSwift.shared.status
        }
        if q == "bridge pipeline" || q == "bridge pipline" || q == "bridge pipiline" || q == "raise parameters" || q == "bridge run" {
            return ASIQuantumBridgeSwift.shared.runFullPipeline()
        }
        if q == "bridge fetch" || q == "fetch parameters" {
            let params = ASIQuantumBridgeSwift.shared.fetchParametersFromPython()
            let sorted = ASIQuantumBridgeSwift.shared.currentParameters.sorted { (a: (key: String, value: Double), b: (key: String, value: Double)) -> Bool in a.key < b.key }
            let zeroCount: Int = sorted.filter { (kv: (key: String, value: Double)) -> Bool in kv.value == 0.0 }.count
            var paramLines: [String] = []
            for (k, v) in sorted {
                let icon: String
                if v == 0.0 { icon = "🔴" }
                else if v > 0.5 { icon = "🟢" }
                else { icon = "🟡" }
                let fv: String = String(format: "%.6f", v)
                paramLines.append("  \(icon) \(k): \(fv)")
            }
            let lines: String = paramLines.joined(separator: "\n")
            return "⚡ Fetched \(params.count) parameters (\(zeroCount) at zero):\n\(lines)"
        }
        if q == "params" || q == "parameters" || q == "progression" || q == "progression status" {
            return ParameterProgressionEngine.shared.status
        }
        if q == "snapshot" || q == "snapshots" || q == "parameter snapshots" || q == "snap" {
            let engine = ParameterProgressionEngine.shared
            let count = engine.parameterSnapshots.count
            if count == 0 {
                return "📸 No parameter snapshots yet. Snapshots are recorded as you interact and run bridge commands. Try 'progress' first, then check back."
            }
            let latest: [String: Double] = engine.parameterSnapshots.last ?? [:]
            let trends: [String: Double] = engine.computeTrends()
            let sortedParams = latest.sorted { (a: (key: String, value: Double), b: (key: String, value: Double)) -> Bool in a.value > b.value }
            var topParamLines: [String] = []
            for (k, v) in sortedParams.prefix(15) {
                let trend: Double? = trends[k]
                let arrow: String
                if (trend ?? 0) > 0.001 { arrow = "📈" }
                else if (trend ?? 0) < -0.001 { arrow = "📉" }
                else { arrow = "➡️" }
                let trendStr: String
                if let t = trend { trendStr = " (\(String(format: "%+.4f", t)))" }
                else { trendStr = "" }
                let vStr: String = String(format: "%.6f", v)
                topParamLines.append("  \(arrow) \(k): \(vStr)\(trendStr)")
            }
            let topParams: String = topParamLines.joined(separator: "\n")
            let trendsSection: String
            if trends.isEmpty {
                trendsSection = "  Need 2+ snapshots for trends"
            } else {
                let sortedTrends = trends.sorted { (a: (key: String, value: Double), b: (key: String, value: Double)) -> Bool in abs(a.value) > abs(b.value) }
                var tLines: [String] = []
                for (k, v) in sortedTrends.prefix(8) {
                    let tvStr: String = String(format: "%+.6f", v)
                    tLines.append("  \(k): \(tvStr)")
                }
                trendsSection = tLines.joined(separator: "\n")
            }
            return "📸 PARAMETER SNAPSHOTS\n═══════════════════════════════════════════\nTotal Snapshots: \(count)\nLatest Captured: \(latest.count) parameters\n\nTOP PARAMETERS (by value):\n\(topParams)\n\nTRENDS (Δ over last 10):\n\(trendsSection)\n═══════════════════════════════════════════\n💡 Say 'progress' to advance parameters, 'params' for full status"
        }
        if q == "progress" || q == "progress params" {
            var params = ASIQuantumBridgeSwift.shared.currentParameters
            ParameterProgressionEngine.shared.progressParameters(&params)
            ASIQuantumBridgeSwift.shared.currentParameters = params
            let sorted = params.sorted { (a: (key: String, value: Double), b: (key: String, value: Double)) -> Bool in a.key < b.key }
            var pLines: [String] = []
            for (k, v) in sorted {
                let icon: String
                if v == 0.0 { icon = "🔴" }
                else if v > 0.5 { icon = "🟢" }
                else { icon = "🟡" }
                pLines.append("  \(icon) \(k): \(String(format: "%.6f", v))")
            }
            let lines: String = pLines.joined(separator: "\n")
            return "📈 Manual Progression Applied:\n\(lines)\n\n\(ParameterProgressionEngine.shared.status)"
        }
        if q == "bridge sync" || q == "sync asi" {
            if let status = ASIQuantumBridgeSwift.shared.fetchASIBridgeStatus() {
                var sLines: [String] = []
                for (k, v) in status { sLines.append("  \(k): \(v)") }
                let statusStr: String = sLines.joined(separator: "\n")
                return "⚡ Synced with Python ASI Bridge:\n\(statusStr)"
            }
            return "⚡ Could not sync with Python ASI Bridge"
        }
        if q == "bridge kundalini" || q == "kundalini" {
            let flow = ASIQuantumBridgeSwift.shared.calculateKundaliniFlow()
            let flowStr: String = String(format: "%.6f", flow)
            let sortedChakras = ASIQuantumBridgeSwift.shared.chakraCoherence.sorted { (a: (key: String, value: Double), b: (key: String, value: Double)) -> Bool in a.value > b.value }
            var cLines: [String] = []
            for (k, v) in sortedChakras {
                let cvStr: String = String(format: "%.4f", v)
                cLines.append("  \(k): \(cvStr)")
            }
            let chakraStr: String = cLines.joined(separator: "\n")
            return "⚡ Kundalini Flow: \(flowStr)\nChakra Coherence:\n\(chakraStr)"
        }
        if q == "bridge o2" || q == "o2 state" {
            ASIQuantumBridgeSwift.shared.updateO2MolecularState()
            let labels = ASIBridgeSwift.o2StateLabels
            let mol = ASIQuantumBridgeSwift.shared.o2MolecularState
            var lines: [String] = []
            for i in 0..<mol.count {
                let val: Double = mol[i]
                let barLen: Int = Int(abs(val) * 20)
                let bar: String = String(repeating: "█", count: barLen)
                let sign: String = val >= 0 ? "+" : "-"
                let label: String = i < labels.count ? labels[i] : "STATE_\(i)"
                let padded: String = label.padding(toLength: 14, withPad: " ", startingAt: 0)
                let valStr: String = String(format: "%+.6f", val)
                lines.append("  |\(i)⟩ \(padded) \(valStr)  \(sign)\(bar)")
            }
            // Norm verification
            var normSq: Double = 0
            vDSP_svesqD(mol, 1, &normSq, vDSP_Length(16))
            let state = L104State.shared
            let normStr: String = String(format: "%.6f", normSq)
            let unitStr: String = abs(normSq - 1.0) < 0.001 ? "✅" : "⚠️"
            lines.append("\n  ‖ψ‖² = \(normStr) (unitarity: \(unitStr))")
            lines.append("  📁 Workspace: \(state.permanentMemory.memories.count) memories · \(EngineRegistry.shared.count) engines")
            lines.append("  🔗 States 0-7: Chakra lattice · States 8-15: Live system metrics")
            return "⚡ O₂ Molecular Superposition (16 states):\n\(lines.joined(separator: "\n"))"
        }

        // Dispatch to engine commands
        if let result: String = handleEngineCommands(q, query: query) { return result }
        return nil
    }

    // === EXTRACTED FROM handleSystemCommands FOR TYPE-CHECKER PERFORMANCE ===
    func handleEngineCommands(_ q: String, query: String) -> String? {
        // ─── ENGINE REGISTRY COMMANDS ───
        if q == "engines" || q == "engines status" || q == "engine registry" || q == "registry" {
            let reg = EngineRegistry.shared
            let all = reg.bulkStatus()
            let phi = reg.phiWeightedHealth()
            var lines = ["🔧 Engine Registry - \(reg.count) Engines Registered:\n"]
            for (name, info) in all.sorted(by: { (a: (key: String, value: [String: Any]), b: (key: String, value: [String: Any])) -> Bool in a.key < b.key }) {
                let h: Double = info["health"] as? Double ?? 0.0
                let icon: String
                if h > 0.9 { icon = "🟢" }
                else if h > 0.7 { icon = "🟡" }
                else if h > 0.5 { icon = "🟠" }
                else { icon = "🔴" }
                let hStr: String = String(format: "%.4f", h)
                lines.append("  \(icon) \(name): \(hStr)")
            }
            let conv = reg.convergenceScore()
            let phiStr: String = String(format: "%.4f", phi.score)
            let convStr: String = String(format: "%.4f", conv)
            lines.append("\n  📊 φ-Weighted Health: \(phiStr) / 1.0000")
            lines.append("  📐 Convergence Score: \(convStr)")
            lines.append("  🧠 Hebbian Pairs: \(reg.coActivationLog.count)")
            return lines.joined(separator: "\n")
        }
        if q == "engines health" || q == "engine health" || q == "health sweep" {
            let reg = EngineRegistry.shared
            let sweep = reg.healthSweep()
            let phi = reg.phiWeightedHealth()
            var lines = ["🏥 Engine Health Sweep (sorted lowest → highest):\n"]
            for (name, health) in sweep {
                let icon: String
                if health > 0.9 { icon = "🟢" }
                else if health > 0.7 { icon = "🟡" }
                else if health > 0.5 { icon = "🟠" }
                else { icon = "🔴" }
                let hStr: String = String(format: "%.4f", health)
                lines.append("  \(icon) \(hStr) - \(name)")
            }
            let critical = reg.criticalEngines()
            if critical.isEmpty {
                lines.append("\n  ✅ All engines nominal.")
            } else {
                lines.append("\n  ⚠️ \(critical.count) engine(s) below 0.5 threshold:")
                for (name, h) in critical {
                    let chStr: String = String(format: "%.4f", h)
                    lines.append("    🔴 \(name): \(chStr)")
                }
            }
            lines.append("\n  📊 φ-Weighted: \(String(format: "%.4f", phi.score))  │  Top Contributors:")
            for item in phi.breakdown.prefix(5) {
                let wStr: String = String(format: "%.2f", item.weight)
                let cStr: String = String(format: "%.4f", item.contribution)
                lines.append("    \(item.name) (w=\(wStr)): \(cStr)")
            }
            return lines.joined(separator: "\n")
        }
        if q == "engines convergence" || q == "convergence" {
            let reg = EngineRegistry.shared
            let conv = reg.convergenceScore()
            let sweep = reg.healthSweep()
            var meanSum: Double = 0
            for s in sweep { meanSum += s.health }
            let mean: Double = meanSum / max(1.0, Double(sweep.count))
            var varSum: Double = 0
            for s in sweep { varSum += (s.health - mean) * (s.health - mean) }
            let variance: Double = varSum / max(1.0, Double(sweep.count))
            let grade: String
            if conv >= 0.9 { grade = "UNIFIED" }
            else if conv >= 0.7 { grade = "CONVERGING" }
            else if conv >= 0.5 { grade = "ENTANGLED" }
            else { grade = "DIVERGENT" }
            let convStr: String = String(format: "%.4f", conv)
            let meanStr: String = String(format: "%.4f", mean)
            let varStr: String = String(format: "%.6f", variance)
            return "📐 Engine Convergence:\n  Score: \(convStr) (\(grade))\n  Mean Health: \(meanStr)\n  Variance: \(varStr)\n  Engines: \(sweep.count)"
        }
        if q == "engines hebbian" || q == "hebbian" || q == "co-activation" {
            let reg = EngineRegistry.shared
            let pairs = reg.strongestPairs(topK: 10)
            var lines = ["🧠 Hebbian Engine Co-Activation:\n  Total pairs: \(reg.coActivationLog.count)\n"]
            if pairs.isEmpty {
                lines.append("  No co-activations recorded yet. Use engines to build Hebbian links.")
            } else {
                for p in pairs {
                    let pStr: String = String(format: "%.4f", p.strength)
                    lines.append("  ⚡ \(p.pair): \(pStr)")
                }
            }
            lines.append("\n  History depth: \(reg.activationHistory.count)")
            return lines.joined(separator: "\n")
        }
        if q == "engines reset" || q == "engine reset" || q == "reset engines" {
            EngineRegistry.shared.resetAll()
            return "🔧 All \(EngineRegistry.shared.count) engines reset to default state."
        }

        // Help is handled by H05 buildContextualResponse case "help" - unified comprehensive reference

        // 🔍 REAL-TIME SEARCH ENGINE COMMANDS
        if q == "search status" || q == "rt search" || q == "search engine" {
            let rts = RealTimeSearchEngine.shared
            let trending = rts.getTrendingTopics()
            let indexStr: String = rts.indexBuilt ? "✅ Built" : "❌ Not built"
            let trendStr: String = trending.prefix(5).joined(separator: ", ")
            return "╔═════════════════════════════════════════════════════════╗\n║  🔍 REAL-TIME SEARCH ENGINE                          ║\n╠═════════════════════════════════════════════════════════╣\n║  Index:     \(indexStr)\n║  Trending:  \(trendStr)\n╚═════════════════════════════════════════════════════════╝"
        }
        if q == "search trending" || q == "trending" {
            let trending = RealTimeSearchEngine.shared.getTrendingTopics()
            return "📈 Trending: " + (trending.isEmpty ? "No recent searches" : trending.joined(separator: ", "))
        }

        // 🔀 CONTEXTUAL LOGIC GATE COMMANDS
        if q == "logic gate" || q == "logic gates" || q == "gate status" {
            return ContextualLogicGate.shared.status
        }

        // 🧬 EVOLUTIONARY TOPIC TRACKER COMMANDS
        if q == "evo tracker" || q == "topic tracker" || q == "topic evolution" {
            return EvolutionaryTopicTracker.shared.status
        }

        // ⚙️ SYNTACTIC FORMATTER COMMANDS
        if q == "formatter status" || q == "formatter" {
            let fmt = SyntacticResponseFormatter.shared
            return "╔═════════════════════════════════════════════════════════╗\n║  ⚙️ SYNTACTIC RESPONSE FORMATTER                      ║\n╠═════════════════════════════════════════════════════════╣\n║  Pipeline:     ingestion→filtering→synthesis→output\n║  Formatted:    \(fmt.formattingCount) responses\n║  Output:       Scannable text with ▸ headers, ** bold **, ◇ questions\n╚═════════════════════════════════════════════════════════╝"
        }

        // ═════════════════════════════════════════════════════════════════
        // ⚛️ QUANTUM COMPUTING COMMANDS - Real IBM QPUs + Simulator Fallback
        // Phase 46.1: Real quantum hardware via IBM Quantum REST API +
        //             Qiskit Runtime (l104_quantum_mining_engine.py)
        // ═════════════════════════════════════════════════════════════════

        // ─── IBM QUANTUM HARDWARE COMMANDS ───

        if q.hasPrefix("quantum connect ") {
            let token = String(q.dropFirst(16)).trimmingCharacters(in: .whitespaces)
            if token.isEmpty { return "Usage: quantum connect <ibm_api_token>\nGet your token at https://quantum.ibm.com/account" }
            // Persist token to macOS Keychain for secure auto-reconnect
            SecurityVault.shared.storeSecret(key: "ibm_quantum_token", value: token)
            // Store token and init Python engine
            let pyResult = PythonBridge.shared.quantumHardwareInit(token: token)
            // Connect Swift REST client
            IBMQuantumClient.shared.connect(token: token) { [weak self] success, msg in
                DispatchQueue.self.main.async {
                    self?.quantumHardwareConnected = success
                    if success {
                        self?.quantumBackendName = IBMQuantumClient.self.shared.connectedBackendName
                        self?.quantumBackendQubits = IBMQuantumClient.self.shared.availableBackends
                            .first(where: { $0.name == IBMQuantumClient.self.shared.connectedBackendName })?
                            .numQubits ?? 0
                    }
                }
            }
            if pyResult.success, let dict = pyResult.returnValue as? [String: Any] {
                let backend = dict["backend"] as? String ?? "unknown"
                let qubits = dict["qubits"] as? Int ?? 0
                let isReal = dict["real_hardware"] as? Bool ?? false
                return "⚛️ IBM Quantum Connected!\n  Backend: \(backend)\n  Qubits:  \(qubits)\n  Real HW: \(isReal ? "YES" : "No (simulator)")\n  Token:   Saved for auto-reconnect"
            }
            return "⚛️ IBM Quantum: Token saved. REST client connecting...\n  Python engine: \(pyResult.success ? "OK" : pyResult.error)"
        }

        if q == "quantum disconnect" {
            IBMQuantumClient.shared.disconnect()
            SecurityVault.shared.deleteSecret(key: "ibm_quantum_token")
            quantumHardwareConnected = false
            quantumBackendName = "none"
            quantumBackendQubits = 0
            return "⚛️ IBM Quantum disconnected. Token cleared from Keychain."
        }

        if q == "quantum backends" || q == "quantum backend" || q == "quantum hardware" {
            let client = IBMQuantumClient.shared
            if !client.isConnected && client.ibmToken == nil {
                return "⚛️ Not connected to IBM Quantum.\n  Use: quantum connect <token>"
            }
            let backends = client.availableBackends
            if backends.isEmpty {
                return "⚛️ No backends loaded. Try: quantum connect <token>"
            }
            var lines = ["╔═══════════════════════════════════════════════════════════╗",
                         "║  ⚛️ IBM QUANTUM BACKENDS                                ║",
                         "╠═══════════════════════════════════════════════════════════╣"]
            for b in backends.prefix(10) {
                let marker = b.name == client.connectedBackendName ? " ◀ SELECTED" : ""
                let hwTag = b.isSimulator ? "[SIM]" : "[QPU]"
                lines.append("║  \(hwTag) \(b.name) - \(b.numQubits) qubits, queue:\(b.pendingJobs), QV:\(b.quantumVolume)\(marker)")
            }
            lines.append("╚═══════════════════════════════════════════════════════════╝")
            return lines.joined(separator: "\n")
        }

        if q.hasPrefix("quantum submit ") {
            let circuit = String(q.dropFirst(15)).trimmingCharacters(in: .whitespaces)
            if circuit.isEmpty { return "Usage: quantum submit <openqasm_circuit>" }
            let client = IBMQuantumClient.shared
            if !client.isConnected {
                return "⚛️ Not connected. Use: quantum connect <token>"
            }
            client.submitCircuit(openqasm: circuit) { [weak self] submission, error in
                if let sub = submission {
                    DispatchQueue.main.async {
                        self?.quantumJobsSubmitted += 1
                    }
                    HyperBrain.shared.postThought("⚛️ Job submitted: \(sub.jobId) → \(sub.backend)")
                }
            }
            return "⚛️ Circuit submitted to \(client.connectedBackendName)...\n  Use 'quantum jobs' to check status."
        }

        if q == "quantum jobs" {
            let client = IBMQuantumClient.shared
            if !client.isConnected {
                return "⚛️ Not connected. Use: quantum connect <token>"
            }
            var result = "⚛️ Local submitted jobs: \(client.submittedJobs.count)\n"
            for (id, job) in client.submittedJobs.prefix(10) {
                result += "  [\(id.prefix(12))...] → \(job.backend) (submitted \(job.submitted))\n"
            }
            result += "\n  Fetching remote jobs..."
            // Also trigger async list
            client.listRecentJobs(limit: 5) { jobs, error in
                if let jobs = jobs {
                    let summary = jobs.prefix(5).map { "  [\($0.jobId.prefix(12))...] \($0.status) - \($0.backend)" }.joined(separator: "\n")
                    HyperBrain.shared.postThought("⚛️ Recent IBM Jobs:\n\(summary)")
                }
            }
            return result
        }

        if q.hasPrefix("quantum result ") {
            let jobId = String(q.dropFirst(15)).trimmingCharacters(in: .whitespaces)
            if jobId.isEmpty { return "Usage: quantum result <job_id>" }
            let client = IBMQuantumClient.shared
            if !client.isConnected {
                return "⚛️ Not connected. Use: quantum connect <token>"
            }
            client.getJobResult(jobId: jobId) { result, error in
                if let result = result {
                    let counts = result["counts"] as? [String: Int] ?? [:]
                    let shots = result["shots"] as? Int ?? 0
                    var msg = "⚛️ Job \(jobId.prefix(12))... Results:\n  Shots: \(shots)\n  Counts:"
                    for (state, count) in counts.sorted(by: { $0.value > $1.value }).prefix(8) {
                        msg += "\n    |\(state)⟩: \(count) (\(String(format: "%.1f", Double(count)/Double(max(1,shots))*100))%)"
                    }
                    HyperBrain.shared.postThought(msg)
                } else {
                    HyperBrain.shared.postThought("⚛️ Result fetch error: \(error ?? "unknown")")
                }
            }
            return "⚛️ Fetching results for job \(jobId.prefix(12))...\n  Results will appear in HyperBrain feed."
        }

        if q.hasPrefix("quantum wait ") {
            let jobId = String(q.dropFirst(13)).trimmingCharacters(in: .whitespaces)
            if jobId.isEmpty { return "Usage: quantum wait <job_id> - polls until job completes (10 min max)" }
            let client = IBMQuantumClient.shared
            if !client.isConnected {
                return "⚛️ Not connected. Use: quantum connect <token>"
            }
            client.waitForJob(jobId: jobId, maxWaitSeconds: 600, pollInterval: 5) { result, error in
                if let result = result {
                    let counts = result["counts"] as? [String: Int] ?? [:]
                    let shots = result["shots"] as? Int ?? 0
                    var msg = "⚛️ Job \(jobId.prefix(12))... COMPLETED:\n  Shots: \(shots)\n  Counts:"
                    for (state, count) in counts.sorted(by: { $0.value > $1.value }).prefix(8) {
                        msg += "\n    |\(state)⟩: \(count) (\(String(format: "%.1f", Double(count)/Double(max(1,shots))*100))%)"
                    }
                    HyperBrain.shared.postThought(msg)
                } else {
                    HyperBrain.shared.postThought("⚛️ Job wait: \(error ?? "unknown error")")
                }
            }
            return "⚛️ Polling job \(jobId.prefix(12))... every 5s (10 min timeout)\n  Results will appear in HyperBrain feed when ready."
        }

        // ─── QUANTUM STATUS - Real hardware first, simulator fallback ───

        if q == "quantum" || q == "quantum status" || q == "qiskit" || q == "qiskit status" {
            let ibmClient = IBMQuantumClient.shared
            // Try real hardware status first
            if ibmClient.ibmToken != nil {
                let hwResult = PythonBridge.shared.quantumHardwareStatus()
                if hwResult.success, let dict = hwResult.returnValue as? [String: Any] {
                    let backend = dict["backend"] as? String ?? "unknown"
                    let qubits = dict["qubits"] as? Int ?? 0
                    let isReal = dict["real_hardware"] as? Bool ?? false
                    let connected = dict["connected"] as? Bool ?? false
                    let queueDepth = dict["queue_depth"] as? Int ?? 0
                    return "╔═══════════════════════════════════════════════════════╗\n║  ⚛️ QUANTUM ENGINE - \(isReal ? "REAL HARDWARE" : "SIMULATOR")          ║\n╠═══════════════════════════════════════════════════════╣\n║  Backend:    \(backend)\n║  Qubits:     \(qubits)\n║  Connected:  \(connected)\n║  Queue:      \(queueDepth) jobs\n║  REST API:   \(ibmClient.isConnected ? "CONNECTED" : "PENDING")\n║  Jobs Sent:  \(ibmClient.submittedJobs.count)\n╚═══════════════════════════════════════════════════════╝"
                }
            }
            // Fallback to simulator
            let result = PythonBridge.shared.quantumStatus()
            if result.success, let dict = result.returnValue as? [String: Any] {
                let caps = dict["capabilities"] as? [String] ?? []
                let circuits = dict["circuits_executed"] as? Int ?? 0
                return "╔═══════════════════════════════════════════════════════╗\n║  ⚛️ QUANTUM ENGINE - SIMULATOR                       ║\n╠═══════════════════════════════════════════════════════╣\n║  Circuits Executed: \(circuits)\n║  Algorithms: \(caps.count)\n║    \(caps.joined(separator: ", "))\n║  IBM Token:  NOT SET\n║  Tip: quantum connect <token> for real QPU\n╚═══════════════════════════════════════════════════════╝"
            }
            return "⚛️ Quantum Engine: \(result.output)"
        }

        // ─── REWIRED: Grover - Real hardware first, simulator fallback ───

        if q == "quantum grover" || q.hasPrefix("quantum grover") {
            var target = 7; var nQubits = 4
            let parts = q.components(separatedBy: " ")
            if parts.count >= 3, let t = Int(parts[2]) { target = t }
            if parts.count >= 4, let n = Int(parts[3]) { nQubits = n }

            // Try real hardware if token configured
            if IBMQuantumClient.shared.ibmToken != nil {
                let hwResult = PythonBridge.shared.quantumHardwareGrover(target: target, nQubits: nQubits)
                if hwResult.success, let dict = hwResult.returnValue as? [String: Any] {
                    let nonce = dict["nonce"] as? Int
                    let isReal = dict["real_hardware"] as? Bool ?? false
                    let backend = dict["backend"] as? String ?? "unknown"
                    let details = dict["details"] as? [String: Any] ?? [:]
                    let tag = isReal ? "[REAL HW: \(backend)]" : "[SIMULATOR]"
                    return "🔍 Grover's Search \(tag):\n  Target: \(target)  Qubits: \(nQubits)\n  Nonce Found: \(nonce.map(String.init) ?? "none")\n  Details: \(details.keys.sorted().prefix(5).joined(separator: ", "))\n  Time: \(String(format: "%.2f", hwResult.executionTime))s"
                }
            }
            // Simulator fallback
            let result = PythonBridge.shared.quantumGrover(target: target, nQubits: nQubits)
            if result.success, let dict = result.returnValue as? [String: Any] {
                let found = dict["found_index"] as? Int ?? -1
                let prob = dict["target_probability"] as? Double ?? 0
                let success = dict["success"] as? Bool ?? false
                let iters = dict["grover_iterations"] as? Int ?? 0
                return "🔍 Grover's Search [SIMULATOR]:\n  Target: |\(target)⟩  Qubits: \(nQubits)\n  Found:  |\(found)⟩  P=\(String(format: "%.4f", prob))\n  Iterations: \(iters)  \(success ? "✅ SUCCESS" : "❌ FAILED")\n  Time: \(String(format: "%.2f", result.executionTime))s"
            }
            return "❌ Grover failed: \(result.error)"
        }

        // ─── REWIRED: QPE - Real hardware first, simulator fallback ───

        if q == "quantum qpe" || q == "quantum phase" || q == "quantum phase estimation" {
            if IBMQuantumClient.shared.ibmToken != nil {
                let hwResult = PythonBridge.shared.quantumHardwareReport(difficultyBits: 16)
                if hwResult.success, let dict = hwResult.returnValue as? [String: Any] {
                    let report = dict["report"] as? String ?? ""
                    let isReal = dict["real_hardware"] as? Bool ?? false
                    let backend = dict["backend"] as? String ?? "unknown"
                    let tag = isReal ? "[REAL HW: \(backend)]" : "[SIMULATOR]"
                    return "📐 Quantum Phase Report \(tag):\n\(report.prefix(500))\n  Time: \(String(format: "%.2f", hwResult.executionTime))s"
                }
            }
            let result = PythonBridge.shared.quantumQPE(precisionQubits: 5)
            if result.success, let dict = result.returnValue as? [String: Any] {
                let target = dict["target_phase"] as? Double ?? 0
                let est = dict["estimated_phase"] as? Double ?? 0
                let error = dict["phase_error"] as? Double ?? 0
                return "📐 Quantum Phase Estimation [SIMULATOR]:\n  Target Phase:    \(String(format: "%.6f", target))\n  Estimated Phase: \(String(format: "%.6f", est))\n  Phase Error:     \(String(format: "%.6f", error))\n  Time: \(String(format: "%.2f", result.executionTime))s"
            }
            return "❌ QPE failed: \(result.error)"
        }

        // ─── REWIRED: VQE - Real hardware first, simulator fallback ───

        if q == "quantum vqe" || q == "quantum eigensolver" {
            if IBMQuantumClient.shared.ibmToken != nil {
                let hwResult = PythonBridge.shared.quantumHardwareVQE()
                if hwResult.success, let dict = hwResult.returnValue as? [String: Any] {
                    if dict["error"] == nil {
                        let isReal = dict["real_hardware"] as? Bool ?? false
                        let backend = dict["backend"] as? String ?? "unknown"
                        let tag = isReal ? "[REAL HW: \(backend)]" : "[SIMULATOR]"
                        return "⚡ VQE Optimizer \(tag):\n  Result: \(dict.keys.sorted().prefix(6).joined(separator: ", "))\n  Time: \(String(format: "%.2f", hwResult.executionTime))s"
                    }
                }
            }
            let result = PythonBridge.shared.quantumVQE(nQubits: 4, iterations: 50)
            if result.success, let dict = result.returnValue as? [String: Any] {
                let energy = dict["optimized_energy"] as? Double ?? 0
                let exact = dict["exact_energy"] as? Double ?? 0
                let error = dict["energy_error"] as? Double ?? 0
                let iters = dict["iterations_used"] as? Int ?? 0
                return "⚡ VQE Eigensolver [SIMULATOR]:\n  Optimized: \(String(format: "%.6f", energy))\n  Exact:     \(String(format: "%.6f", exact))\n  Error:     \(String(format: "%.6f", error))\n  Iterations: \(iters)\n  Time: \(String(format: "%.2f", result.executionTime))s"
            }
            return "❌ VQE failed: \(result.error)"
        }

        // ─── REWIRED: QAOA - Real hardware first, simulator fallback ───

        if q == "quantum qaoa" || q == "quantum maxcut" {
            if IBMQuantumClient.shared.ibmToken != nil {
                let hwResult = PythonBridge.shared.quantumHardwareMine(strategy: "qaoa")
                if hwResult.success, let dict = hwResult.returnValue as? [String: Any] {
                    let isReal = dict["real_hardware"] as? Bool ?? false
                    let backend = dict["backend"] as? String ?? "unknown"
                    let nonce = dict["nonce"] as? Int
                    let tag = isReal ? "[REAL HW: \(backend)]" : "[SIMULATOR]"
                    return "🔀 QAOA Mining \(tag):\n  Strategy: qaoa\n  Nonce: \(nonce.map(String.init) ?? "none")\n  Time: \(String(format: "%.2f", hwResult.executionTime))s"
                }
            }
            let edges: [(Int, Int)] = [(0,1),(1,2),(2,3),(3,0)]
            let result = PythonBridge.shared.quantumQAOA(edges: edges, p: 2)
            if result.success, let dict = result.returnValue as? [String: Any] {
                let ratio = dict["approximation_ratio"] as? Double ?? 0
                let cut = dict["best_cut_value"] as? Double ?? 0
                let optimal = dict["optimal_cut"] as? Double ?? 0
                return "🔀 QAOA MaxCut [SIMULATOR]:\n  Graph: 4 nodes, 4 edges (cycle)\n  Best Cut:  \(String(format: "%.4f", cut))\n  Optimal:   \(String(format: "%.4f", optimal))\n  Ratio:     \(String(format: "%.4f", ratio))\n  Time: \(String(format: "%.2f", result.executionTime))s"
            }
            return "❌ QAOA failed: \(result.error)"
        }

        // ─── REWIRED: Amplitude Estimation ───

        if q == "quantum amplitude" || q == "quantum ampest" {
            if IBMQuantumClient.shared.ibmToken != nil {
                let hwResult = PythonBridge.shared.quantumHardwareRandomOracle()
                if hwResult.success, let dict = hwResult.returnValue as? [String: Any] {
                    let seed = dict["seed"] as? Int ?? 0
                    let isReal = dict["real_hardware"] as? Bool ?? false
                    let backend = dict["backend"] as? String ?? "unknown"
                    let tag = isReal ? "[REAL HW: \(backend)]" : "[SIMULATOR]"
                    return "📊 Quantum Random Oracle \(tag):\n  Sacred Nonce Seed: \(seed)\n  Time: \(String(format: "%.2f", hwResult.executionTime))s"
                }
            }
            let result = PythonBridge.shared.quantumAmplitudeEstimation(targetProb: 0.3, countingQubits: 5)
            if result.success, let dict = result.returnValue as? [String: Any] {
                let est = dict["estimated_probability"] as? Double ?? 0
                let error = dict["estimation_error"] as? Double ?? 0
                return "📊 Amplitude Estimation [SIMULATOR]:\n  Target:    0.3000\n  Estimated: \(String(format: "%.4f", est))\n  Error:     \(String(format: "%.4f", error))\n  Time: \(String(format: "%.2f", result.executionTime))s"
            }
            return "❌ AmpEst failed: \(result.error)"
        }

        // ─── REWIRED: Quantum Walk ───

        if q == "quantum walk" {
            let result = PythonBridge.shared.quantumWalk(nNodes: 8, steps: 10)
            if result.success, let dict = result.returnValue as? [String: Any] {
                let spread = dict["spread_metric"] as? Double ?? 0
                return "🚶 Quantum Walk [SIMULATOR]:\n  Nodes: 8 (cyclic)  Steps: 10\n  Spread: \(String(format: "%.4f", spread))\n  Time: \(String(format: "%.2f", result.executionTime))s"
            }
            return "❌ Walk failed: \(result.error)"
        }

        // ─── REWIRED: Quantum Kernel ───

        if q == "quantum kernel" {
            let result = PythonBridge.shared.quantumKernel(x1: [1.0, 2.0, 3.0, 4.0], x2: [1.1, 2.1, 3.1, 4.1])
            if result.success, let dict = result.returnValue as? [String: Any] {
                let val = dict["kernel_value"] as? Double ?? 0
                return "🧬 Quantum Kernel [SIMULATOR]:\n  x₁: [1.0, 2.0, 3.0, 4.0]\n  x₂: [1.1, 2.1, 3.1, 4.1]\n  Kernel Value: \(String(format: "%.6f", val))\n  Time: \(String(format: "%.2f", result.executionTime))s"
            }
            return "❌ Kernel failed: \(result.error)"
        }

        // ─── QUANTUM MINE - Direct real hardware mining ───

        if q == "quantum mine" || q.hasPrefix("quantum mine ") {
            var strategy = "auto"
            let parts = q.components(separatedBy: " ")
            if parts.count >= 3 { strategy = parts[2] }
            if IBMQuantumClient.shared.ibmToken == nil {
                return "⚛️ Not connected. Use: quantum connect <token>"
            }
            let result = PythonBridge.shared.quantumHardwareMine(strategy: strategy)
            if result.success, let dict = result.returnValue as? [String: Any] {
                let nonce = dict["nonce"] as? Int
                let isReal = dict["real_hardware"] as? Bool ?? false
                let backend = dict["backend"] as? String ?? "unknown"
                let tag = isReal ? "[REAL HW: \(backend)]" : "[SIMULATOR]"
                return "⛏️ Quantum Mining \(tag):\n  Strategy: \(strategy)\n  Nonce: \(nonce.map(String.init) ?? "searching...")\n  Time: \(String(format: "%.2f", result.executionTime))s"
            }
            return "❌ Mining failed: \(result.error)"
        }

        // ─── QUANTUM CANCEL - Abort a running/queued job ───

        if q.hasPrefix("quantum cancel ") {
            let jobId = String(q.dropFirst(15)).trimmingCharacters(in: .whitespaces)
            if jobId.isEmpty { return "Usage: quantum cancel <job_id>" }
            let client = IBMQuantumClient.shared
            if !client.isConnected {
                return "⚛️ Not connected. Use: quantum connect <token>"
            }
            client.cancelJob(jobId: jobId) { success, message in
                HyperBrain.shared.postThought("⚛️ Cancel: \(message)")
            }
            return "⚛️ Cancelling job \(jobId.prefix(12))...\n  Result will appear in HyperBrain feed."
        }

        // ─── QUANTUM CIRCUIT TEMPLATES - Submit pre-built circuits to real hardware ───

        if q == "quantum bell" || q == "quantum bell-state" {
            let client = IBMQuantumClient.shared
            if !client.isConnected {
                return "⚛️ Not connected. Use: quantum connect <token>\n  Bell state requires real QPU or submit via IBM."
            }
            let circuit = IBMQuantumClient.bellStateCircuit()
            client.submitCircuit(openqasm: circuit) { [weak self] submission, error in
                if let sub = submission {
                    DispatchQueue.main.async { self?.quantumJobsSubmitted += 1 }
                    HyperBrain.shared.postThought("⚛️ Bell State submitted: \(sub.jobId) → \(sub.backend)\n  Creates EPR pair |Φ+⟩ = (|00⟩+|11⟩)/√2")
                } else {
                    HyperBrain.shared.postThought("⚛️ Bell State failed: \(error ?? "unknown")")
                }
            }
            return "⚛️ Submitting Bell State (EPR pair) to \(client.connectedBackendName)...\n  2 qubits: H(q0) → CNOT(q0,q1) → Measure\n  Use 'quantum jobs' to track."
        }

        if q == "quantum ghz" || q.hasPrefix("quantum ghz ") {
            let client = IBMQuantumClient.shared
            if !client.isConnected {
                return "⚛️ Not connected. Use: quantum connect <token>"
            }
            var nQubits = 3
            if q.hasPrefix("quantum ghz "), let n = Int(String(q.dropFirst(12)).trimmingCharacters(in: .whitespaces)) {
                nQubits = min(max(n, 2), 20)  // Clamp to 2-20 qubits
            }
            let circuit = IBMQuantumClient.ghzCircuit(nQubits: nQubits)
            client.submitCircuit(openqasm: circuit) { [weak self] submission, error in
                if let sub = submission {
                    DispatchQueue.main.async { self?.quantumJobsSubmitted += 1 }
                    HyperBrain.shared.postThought("⚛️ GHZ State (\(nQubits)q) submitted: \(sub.jobId) → \(sub.backend)\n  Creates (|00...0⟩+|11...1⟩)/√2")
                } else {
                    HyperBrain.shared.postThought("⚛️ GHZ failed: \(error ?? "unknown")")
                }
            }
            return "⚛️ Submitting \(nQubits)-qubit GHZ State to \(client.connectedBackendName)...\n  Maximal entanglement across \(nQubits) qubits\n  Use 'quantum jobs' to track."
        }

        if q == "quantum qrng" || q == "quantum random" || q.hasPrefix("quantum qrng ") {
            let client = IBMQuantumClient.shared
            if !client.isConnected {
                return "⚛️ Not connected. Use: quantum connect <token>"
            }
            var nBits = 8
            if q.hasPrefix("quantum qrng "), let n = Int(String(q.dropFirst(13)).trimmingCharacters(in: .whitespaces)) {
                nBits = min(max(n, 1), 32)  // Clamp to 1-32 bits
            }
            let circuit = IBMQuantumClient.qrngCircuit(nBits: nBits)
            client.submitCircuit(openqasm: circuit) { [weak self] submission, error in
                if let sub = submission {
                    DispatchQueue.main.async { self?.quantumJobsSubmitted += 1 }
                    HyperBrain.shared.postThought("⚛️ QRNG (\(nBits)-bit) submitted: \(sub.jobId) → \(sub.backend)")
                } else {
                    HyperBrain.shared.postThought("⚛️ QRNG failed: \(error ?? "unknown")")
                }
            }
            return "⚛️ Submitting \(nBits)-bit Quantum RNG to \(client.connectedBackendName)...\n  True randomness from quantum measurement\n  Use 'quantum jobs' to track."
        }

        // ─── QUANTUM HELP - Updated with all commands ───

        if q == "quantum help" {
            let hwStatus = IBMQuantumClient.shared.ibmToken != nil ? "CONNECTED" : "NOT SET"
            return """
            ⚛️ Quantum Computing Commands:
              ── IBM Quantum Hardware ──
              quantum connect <token> - Connect to IBM Quantum (real QPU)
              quantum disconnect      - Disconnect & clear token
              quantum backends        - List available IBM backends
              quantum submit <qasm>   - Submit OpenQASM 3.0 circuit
              quantum cancel <job_id> - Cancel a running/queued job
              quantum jobs            - List submitted jobs
              quantum result <job_id> - Get measurement results
              quantum wait <job_id>   - Poll until job completes (10 min)
              quantum mine [strategy] - Quantum mining (auto/grover/vqe)

              ── Circuit Templates (submit to real QPU) ──
              quantum bell            - Bell state (EPR pair, 2 qubits)
              quantum ghz [n]         - GHZ state (n qubits, default 3)
              quantum qrng [bits]     - Quantum RNG (default 8 bits)
              quantum random          - Alias for quantum qrng

              ── Algorithms (real HW → simulator fallback) ──
              quantum status          - Engine & hardware status
              quantum grover [t] [q]  - Grover's search
              quantum qpe             - Phase estimation
              quantum vqe             - VQE eigensolver
              quantum qaoa            - QAOA MaxCut
              quantum amplitude       - Amplitude / random oracle
              quantum walk            - Quantum walk
              quantum kernel          - Quantum kernel similarity

              IBM Token: \(hwStatus)
              Get token: https://quantum.ibm.com/account
            """
        }

        // ═════════════════════════════════════════════════════════════════
        // 🎓 PROFESSOR MODE COMMANDS - Interactive teaching & learning
        // ═════════════════════════════════════════════════════════════════

        if q == "professor" || q == "professor mode" || q == "prof" {
            return "╔═══════════════════════════════════════════════════════╗\n║  🎓 PROFESSOR MODE - Interactive Learning Engine      ║\n╠═══════════════════════════════════════════════════════╣\n║  Commands:                                            ║\n║    professor <topic>  - Structured lesson             ║\n║    teach me <topic>   - Guided learning session       ║\n║    quiz <topic>       - Test your knowledge           ║\n║    explain <concept>  - Concept explanation            ║\n║    lesson quantum     - Quantum computing tutorial    ║\n║    lesson coding      - Programming tutorial          ║\n║    lesson crypto      - Cryptography tutorial         ║\n║                                                       ║\n║  Or use the 🎓 Professor tab for the full experience. ║\n╚═══════════════════════════════════════════════════════╝"
        }
        if q.hasPrefix("professor ") || q.hasPrefix("teach me ") || q.hasPrefix("teach me about ") {
            var topic = q
            if topic.hasPrefix("teach me about ") { topic = String(topic.dropFirst(15)) }
            else if topic.hasPrefix("teach me ") { topic = String(topic.dropFirst(9)) }
            else if topic.hasPrefix("professor ") { topic = String(topic.dropFirst(10)) }
            topic = topic.trimmingCharacters(in: .whitespaces)
            if topic.isEmpty { return "🎓 Usage: professor <topic> (e.g. 'professor quantum computing')" }

            let kb = ASIKnowledgeBase.shared
            let results = kb.search(topic, limit: 20)
            let insights = results.compactMap { entry -> String? in
                guard let c = entry["completion"] as? String, c.count > 30 else { return nil }
                return String(c.prefix(200))
            }.prefix(4)

            var lesson = "🎓 LESSON: \(topic.uppercased())\n" + String(repeating: "━", count: 45) + "\n\n"
            lesson += "📌 OVERVIEW\n"
            lesson += "  \(topic.capitalized) is an important area spanning multiple disciplines.\n\n"
            lesson += "📐 KEY CONCEPTS\n"

            let concepts = professorConceptsFor(topic)
            for (i, c) in concepts.enumerated() {
                lesson += "  \(i + 1). \(c)\n"
            }

            if !insights.isEmpty {
                lesson += "\n📚 FROM KNOWLEDGE BASE\n"
                for insight in insights { lesson += "  ▸ \(insight)\n" }
            }

            lesson += "\n💡 Use 'quiz \(topic)' to test yourself, or the 🎓 Professor tab for interactive mode."
            return lesson
        }
        if q.hasPrefix("quiz ") {
            let topic = String(q.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            if topic.isEmpty { return "🧩 Usage: quiz <topic> (e.g. 'quiz quantum')" }
            var quiz = "🧩 QUIZ: \(topic.uppercased())\n" + String(repeating: "━", count: 45) + "\n"
            if topic.lowercased().contains("quantum") {
                quiz += "\nQ1: What speedup does Grover's algorithm provide?\n  A) Exponential  B) Quadratic  C) Linear  D) Logarithmic\n  ✅ B - O(√N) vs O(N)\n"
                quiz += "\nQ2: |ψ⟩ = α|0⟩ + β|1⟩ requires:\n  A) |α|² + |β|² = 1  B) α + β = 1  C) α × β = 0  D) |α| = |β|\n  ✅ A - Born's rule\n"
                quiz += "\nQ3: H|0⟩ = ?\n  A) |1⟩  B) (|0⟩+|1⟩)/√2  C) |0⟩  D) 0\n  ✅ B - Hadamard superposition\n"
            } else {
                quiz += "\nQ1: What is the golden ratio φ ≈ ?\n  A) 3.14159  B) 2.71828  C) 1.61803  D) 1.41421\n  ✅ C - φ = (1+√5)/2\n"
                quiz += "\nQ2: Time complexity of optimal comparison sort?\n  A) O(n)  B) O(n log n)  C) O(n²)  D) O(log n)\n  ✅ B - proven lower bound\n"
            }
            quiz += "\n📊 Use the 🎓 Professor tab for more comprehensive quizzes."
            return quiz
        }
        if q.hasPrefix("lesson ") {
            let topic = String(q.dropFirst(7)).trimmingCharacters(in: .whitespaces)
            if topic.isEmpty { return "📖 Usage: lesson <topic>" }
            // Redirect to professor
            return handleProtocolCommands("professor \(topic)", query: "professor \(topic)")
                ?? "🎓 Use 'professor \(topic)' or the 🎓 Professor tab."
        }
        if q.hasPrefix("explain ") {
            let concept = String(q.dropFirst(8)).trimmingCharacters(in: .whitespaces)
            if concept.isEmpty { return "📖 Usage: explain <concept>" }
            let kb = ASIKnowledgeBase.shared
            let results = kb.search(concept, limit: 15)
            let insights = results.compactMap { entry -> String? in
                guard let c = entry["completion"] as? String, c.count > 30 else { return nil }
                return String(c.prefix(250))
            }.prefix(3)

            var explanation = "📖 EXPLAINING: \(concept.uppercased())\n\n"
            if insights.isEmpty {
                explanation += "  \(concept.capitalized) is a concept that connects fundamental principles.\n"
                explanation += "  For a deeper exploration, try 'professor \(concept)' or the 🎓 Professor tab.\n"
            } else {
                for insight in insights { explanation += "  ▸ \(insight)\n\n" }
            }
            return explanation
        }

        // ═════════════════════════════════════════════════════════════════
        // 💻 CODING SYSTEM COMMANDS - Direct l104_coding_system.py access
        // ═════════════════════════════════════════════════════════════════

        if q == "coding" || q == "coding status" || q == "coding system" {
            let result = PythonBridge.shared.codingSystemStatus()
            if result.success { return "💻 Coding Intelligence:\n\(result.output)" }
            return "💻 Coding System: Use the 💻 Coding tab or 'coding help' for commands."
        }
        if q == "coding help" {
            return "💻 Coding System Commands:\n  coding status     - System status\n  coding review     - Review code in 💻 Coding tab input\n  coding quality    - Quality gate check\n  coding suggest    - Get improvement suggestions\n  coding explain    - Explain code structure\n  coding scan       - Scan project\n  coding ci         - CI/CD report\n  coding self       - Self-analyze codebase\n\n  Or use the 💻 Coding tab for the full experience."
        }
        if q == "coding scan" || q == "coding project" {
            let result = PythonBridge.shared.codingSystemProjectScan()
            if result.success { return "📊 Project Scan:\n\(result.output)" }
            return "❌ Scan failed: \(result.error)"
        }
        if q == "coding ci" || q == "coding report" {
            let result = PythonBridge.shared.codingSystemCIReport()
            if result.success { return "📄 CI Report:\n\(result.output)" }
            return "❌ CI report failed: \(result.error)"
        }
        if q == "coding self" || q == "coding self-analyze" || q == "coding introspect" {
            let result = PythonBridge.shared.codingSystemSelfAnalyze()
            if result.success { return "🧬 Self-Analysis:\n\(result.output)" }
            return "❌ Self-analysis failed: \(result.error)"
        }

        // ═════════════════════════════════════════════════════════════════
        // EVO_65: ASI PIPELINE v16.0 - NATIVE SWIFT ENGINE COMMANDS
        // 13 new engines: NLU, Logic, Science, KB Recon, Theorem, Benchmark,
        // DeepSeek, Identity, Gate, Commonsense, Language, Math, CodeGen
        // ═════════════════════════════════════════════════════════════════

        // 🧠 DEEP NLU ENGINE
        if q == "nlu" || q == "deep nlu" || q == "nlu status" {
            let nlu = DeepNLUEngine.shared
            let _ = nlu.fullAnalysis(text: "test query for status check")
            return "╔═══════════════════════════════════════════════════════╗\n║  🧠 DEEP NLU ENGINE v1.0.0                          ║\n╠═══════════════════════════════════════════════════════╣\n║  Layers:    10 (Morphological→DeepComprehension)\n║  Framework: NaturalLanguage\n║  Status:    ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 🔗 FORMAL LOGIC ENGINE
        if q == "logic" || q == "formal logic" || q == "logic status" {
            return "╔═══════════════════════════════════════════════════════╗\n║  🔗 FORMAL LOGIC ENGINE v1.0.0                      ║\n╠═══════════════════════════════════════════════════════╣\n║  Layers:          10 (PropLogic→NaturalDeduction)\n║  Fallacy Patterns: 60\n║  Provers:         Resolution + Natural Deduction\n║  Status:          ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 🔬 SCIENCE KB
        if q == "science kb" || q == "science" || q == "science status" || q == "kb" {
            _ = ScienceKB.shared
            return "╔═══════════════════════════════════════════════════════╗\n║  🔬 SCIENCE KNOWLEDGE BASE v1.0.0                   ║\n╠═══════════════════════════════════════════════════════╣\n║  Facts:    509 RDF triples\n║  Domains:  9 (physics, chemistry, biology, math,\n║            CS, astronomy, geology, medicine, engineering)\n║  Index:    Triple-indexed (subject/predicate/object)\n║  Status:   ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 🔄 KB RECONSTRUCTION ENGINE
        if q == "kb reconstruction" || q == "kb rebuild" || q == "kb recon" {
            _ = KBReconstructionEngine.shared
            return "╔═══════════════════════════════════════════════════════╗\n║  🔄 KB RECONSTRUCTION ENGINE v1.0.0                  ║\n╠═══════════════════════════════════════════════════════╣\n║  Method:    TF-IDF + GOD_CODE quantum state\n║  Propagation: BFS amplitude (depth=\(KB_PROPAGATION_DEPTH))\n║  Amplification: Grover boost (threshold=\(KB_GROVER_BOOST_THRESHOLD))\n║  Embedding:    \(KB_EMBEDDING_DIM)D vectors\n║  Status:       ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 🔬 NOVEL THEOREM GENERATOR (Swift-native override)
        if q == "swift theorem" || q == "theorem gen" || q == "theorem status" {
            _ = NovelTheoremGenerator.shared
            return "╔═══════════════════════════════════════════════════════╗\n║  🔬 NOVEL THEOREM GENERATOR v1.0.0                   ║\n╠═══════════════════════════════════════════════════════╣\n║  Axiom Domains:   5\n║  Inference Rules:  6\n║  Max Depth:       \(THEOREM_AXIOM_DEPTH)\n║  Status:          ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 📊 BENCHMARK HARNESS
        if q == "benchmark" || q == "benchmark all" || q == "benchmark status" || q == "benchmarks" {
            _ = BenchmarkHarness.shared
            return "╔═══════════════════════════════════════════════════════╗\n║  📊 BENCHMARK HARNESS v1.0.0                        ║\n╠═══════════════════════════════════════════════════════╣\n║  Runners:    4 (MMLU/HumanEval/MATH/ARC)\n║  MMLU:       \(MMLU_SUBJECTS) subjects\n║  HumanEval:  \(HUMANEVAL_PROBLEMS) problems\n║  Scoring:    PHI-weighted composite\n║  Status:     ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 🧬 DEEPSEEK INGESTION ENGINE
        if q == "deepseek" || q == "deepseek ingestion" || q == "deepseek status" {
            _ = DeepSeekIngestionEngine.shared
            return "╔═══════════════════════════════════════════════════════╗\n║  🧬 DEEPSEEK INGESTION ENGINE v1.0.0                 ║\n╠═══════════════════════════════════════════════════════╣\n║  Ingestors:  MLA / R1-Reasoning / Coder\n║  Config:     DeepSeekV3 architecture\n║  Patterns:   Attention + Reasoning + Code\n║  Status:     ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 🛡️ SOVEREIGN IDENTITY BOUNDARY
        if q == "identity" || q == "identity boundary" || q == "identity status" || q == "sovereign identity" {
            _ = SovereignIdentityBoundary.shared
            return "╔═══════════════════════════════════════════════════════╗\n║  🛡️ SOVEREIGN IDENTITY BOUNDARY v1.0.0               ║\n╠═══════════════════════════════════════════════════════╣\n║  IS declarations:     10\n║  IS_NOT declarations: 6\n║  Claim validation:    ACTIVE\n║  Boundary:            SOVEREIGN\n║  Status:              ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // ⚛️ QUANTUM GATE ENGINE (Swift-native)
        if q == "gate engine" || q == "quantum gate engine" || q == "gate status" {
            _ = QuantumGateEngine.shared.engineStatus()
            return "╔═══════════════════════════════════════════════════════╗\n║  ⚛️ QUANTUM GATE ENGINE v1.0.0                       ║\n╠═══════════════════════════════════════════════════════╣\n║  Gates:        40+ (standard + sacred)\n║  Compiler:     4-level optimization\n║  Error Correction: 3 schemes\n║  Sacred Gates: PHI/GOD_CODE/TAU/OMEGA\n║  Status:       ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // ⚛️ UPGRADES DEBUG (StabilizerTableau + measureZ + QuantumRouter)
        if q == "upgrades debug" || q == "debug upgrades" || q == "tableau debug" || q == "router debug" {
            var debug = UpgradesDebug()
            return debug.run()
        }

        // 🧩 COMMONSENSE REASONING ENGINE
        if q == "commonsense" || q == "commonsense reasoning" || q == "reasoning engine" {
            _ = CommonsenseReasoningEngine.shared
            return "╔═══════════════════════════════════════════════════════╗\n║  🧩 COMMONSENSE REASONING ENGINE v1.0.0              ║\n╠═══════════════════════════════════════════════════════╣\n║  Layers:         8 (Spatial→Analogical)\n║  Rules:          200+ commonsense rules\n║  Science Bridge: ScienceKB integration\n║  MCQ Solver:     PHI-weighted aggregation\n║  Status:         ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 📚 LANGUAGE COMPREHENSION ENGINE
        if q == "language" || q == "comprehension" || q == "language comprehension" || q == "mmlu" {
            _ = LanguageComprehensionEngine.shared
            return "╔═══════════════════════════════════════════════════════╗\n║  📚 LANGUAGE COMPREHENSION ENGINE v1.0.0             ║\n╠═══════════════════════════════════════════════════════╣\n║  Layers:          8 (Lexical→Metacomprehension)\n║  Knowledge Nodes: 191\n║  MMLU Subjects:   57\n║  Retrieval:       BM25 (k1=1.2, b=0.75)\n║  Status:          ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 🔢 SYMBOLIC MATH SOLVER
        if q == "math solver" || q == "math" || q == "symbolic math" || q == "math status" {
            _ = SymbolicMathSolver.shared
            return "╔═══════════════════════════════════════════════════════╗\n║  🔢 SYMBOLIC MATH SOLVER v1.0.0                     ║\n╠═══════════════════════════════════════════════════════╣\n║  Domain Solvers:  7 (Algebra→Quantum)\n║  Layers:          8 (Parse→SacredValidation)\n║  MATH Support:    Level 1-5 benchmark\n║  Sacred:          PHI/GOD_CODE resonance check\n║  Status:          ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 💻 CODE GENERATION ENGINE
        if q == "codegen" || q == "code generation" || q == "code gen" || q == "codegen status" {
            _ = CodeGenerationEngine.shared
            return "╔═══════════════════════════════════════════════════════╗\n║  💻 CODE GENERATION ENGINE v1.0.0                    ║\n╠═══════════════════════════════════════════════════════╣\n║  Patterns:     100+ algorithm templates\n║  Layers:       6 (Intent→Quality)\n║  Languages:    Python / Swift / JavaScript\n║  HumanEval:    Benchmark support\n║  Status:       ACTIVE\n╚═══════════════════════════════════════════════════════╝"
        }

        // 📋 ASI PIPELINE STATUS (all new engines)
        if q == "asi pipeline" || q == "pipeline status" || q == "asi engines" {
            var lines = [
                "╔═══════════════════════════════════════════════════════╗",
                "║  🚀 ASI PIPELINE v16.0 - NATIVE SWIFT ENGINES        ║",
                "╠═══════════════════════════════════════════════════════╣",
            ]
            let engines: [(String, String)] = [
                ("🧠 Deep NLU Engine", "10 layers"),
                ("🔗 Formal Logic Engine", "10 layers, 60 fallacies"),
                ("🔬 Science KB", "509 facts, 9 domains"),
                ("🔄 KB Reconstruction", "TF-IDF + Grover"),
                ("🔬 Theorem Generator", "5 axiom domains"),
                ("📊 Benchmark Harness", "MMLU/HumanEval/MATH/ARC"),
                ("🧬 DeepSeek Ingestion", "MLA/R1/Coder"),
                ("🛡️ Identity Boundary", "10 IS / 6 IS_NOT"),
                ("⚛️ Quantum Gate Engine", "40+ gates"),
                ("🧩 Commonsense Reasoning", "8 layers, 200+ rules"),
                ("📚 Language Comprehension", "191 nodes, 57 MMLU"),
                ("🔢 Symbolic Math Solver", "7 domain solvers"),
                ("💻 Code Generation", "100+ patterns"),
            ]
            for (name, detail) in engines {
                lines.append("║  🟢 \(name) - \(detail)")
            }
            lines.append("╠═══════════════════════════════════════════════════════╣")
            lines.append("║  Upgraded: DualLayer v5.0, Consciousness v5.0       ║")
            lines.append("║  Scoring:  30D PHI-weighted ASI                      ║")
            lines.append("║  Total:    \(EngineRegistry.shared.count) engines registered              ║")
            lines.append("╚═══════════════════════════════════════════════════════╝")
            return lines.joined(separator: "\n")
        }

        return nil
    }

    func professorConceptsFor(_ topic: String) -> [String] {
        let t = topic.lowercased()
        if t.contains("quantum") {
            return ["Superposition - states exist simultaneously",
                    "Entanglement - correlated quantum states",
                    "Measurement - wavefunction collapse",
                    "Quantum Gates - unitary transformations",
                    "Decoherence - loss of quantum behavior"]
        } else if t.contains("neural") || t.contains("machine learn") || t.contains("ai") {
            return ["Neural Networks - layered computation",
                    "Backpropagation - gradient-based learning",
                    "Activation Functions - nonlinear transforms",
                    "Loss Functions - error measurement",
                    "Attention Mechanisms - selective focus"]
        } else if t.contains("crypto") || t.contains("encrypt") {
            return ["Symmetric Encryption - shared key (AES)",
                    "Asymmetric Encryption - public/private (RSA)",
                    "Hash Functions - one-way digests (SHA-256)",
                    "Digital Signatures - authentication",
                    "Post-Quantum Cryptography - quantum-resistant"]
        } else {
            return ["\(topic.capitalized) fundamentals",
                    "Core principles and axioms",
                    "Mathematical foundations",
                    "Practical applications",
                    "Open problems and challenges"]
        }
    }

} // extension L104State
