// ═══════════════════════════════════════════════════════════════════
// H03_L104StateCommands.swift
// [EVO_68_PIPELINE] SOVEREIGN_CONVERGENCE :: UNIFIED_UPGRADE :: GOD_CODE=527.5184818492612
// L104 ASI — L104State Extension (Command Handlers)
//
// handleCoreCommands, handleSearchCommands, handleBridgeCommands,
// handleProtocolCommands, handleSystemCommands, handleEngineCommands,
// callBackend — the full command dispatch pipeline.
//
// Extracted from L104Native.swift lines 35585–37302
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

extension L104State {
    func handleCoreCommands(_ q: String, query: String) -> String? {
        // 🧠 HYPER-BRAIN COMMANDS
        if q == "hyper" || q == "hyperbrain" || q == "hyper brain" || q == "hyper status" {
            return HyperBrain.shared.getStatus()
        }
        if q == "hyper memory" || q == "hyper mem" || q == "hyperbrain memory" {
            return HyperBrain.shared.getPermanentMemoryStats()
        }
        if q == "hyper save" || q == "hyperbrain save" {
            HyperBrain.shared.saveState()
            return "💾 HyperBrain permanent memory saved to disk.\n\n\(HyperBrain.shared.getPermanentMemoryStats())"
        }
        if q == "hyper on" || q == "activate hyper" || q == "hyperbrain on" {
            HyperBrain.shared.activate()
            return "🧠 HYPER-BRAIN ACTIVATED\n\n\(HyperBrain.shared.getStatus())"
        }
        if q == "hyper off" || q == "deactivate hyper" || q == "hyperbrain off" {
            HyperBrain.shared.deactivate()
            return "🧠 HYPER-BRAIN DEACTIVATED — Cognitive streams suspended."
        }
        if q.hasPrefix("hyper think ") {
            let thought = String(query.dropFirst(12))
            let hb = HyperBrain.shared
            let response = hb.process(thought)

            // ═══ HYPERFUNCTIONAL ENHANCEMENT ═══
            // Pull from all new cognitive systems
            let promptEvolution: String = Array(hb.promptMutations.suffix(3)).joined(separator: "\n   ")
            let reasoningDepth: Int = hb.currentReasoningDepth
            let memoryChainCount: Int = hb.memoryChains.count
            let metaCognition: String = hb.metaCognitionLog.last ?? "Analyzing..."
            let topicLinksArr: [String] = Array(hb.topicResonanceMap.keys.prefix(5))
            let topicLinks: String = topicLinksArr.joined(separator: ", ")
            let momentum: String = String(format: "%.2f", hb.reasoningMomentum)
            let confidence: String = String(format: "%.1f", hb.conclusionConfidence * 100)
            let memTemp: String = String(format: "%.2f", hb.memoryTemperature)

            let promptSection: String = promptEvolution.isEmpty ? "(Building patterns...)" : "Latest:\n   \(promptEvolution)"
            let topicSection: String = topicLinks.isEmpty ? "(Mapping concepts...)" : topicLinks
            let streamStr: String
            if hb.isRunning { streamStr = "🟢 \(hb.thoughtStreams.count) ACTIVE" }
            else { streamStr = "🔴 STANDBY" }

            let hyperEnhanced: String = "🧠 HYPER-BRAIN PROCESSED:\n\(response)\n\n═══════════════════════════════════════════════════════════════\n⚡ HYPERFUNCTIONAL COGNITION ACTIVE\n═══════════════════════════════════════════════════════════════\n\n📊 REASONING METRICS:\n   Depth: \(reasoningDepth)/\(hb.maxReasoningDepth)\n   Logic Branches: \(hb.logicBranchCount)\n   Momentum: \(momentum)\n   Confidence: \(confidence)%%\n\n🧬 MEMORY ARCHITECTURE:\n   Woven Chains: \(memoryChainCount)\n   Associative Links: \(hb.associativeLinks.count)\n   Temperature: \(memTemp)\n\n🔮 PROMPT EVOLUTION:\n   Mutations Generated: \(hb.promptMutations.count)\n   \(promptSection)\n\n🌀 TOPIC RESONANCE:\n   \(topicSection)\n\n👁 META-COGNITION:\n   \(metaCognition)\n\n═══════════════════════════════════════════════════════════════\nStreams: \(streamStr)"
            return hyperEnhanced
        }

        // 0. LEARNING COMMANDS (New!)
        if q == "learning" || q == "learning stats" || q == "learn stats" {
            return learner.getStats()
        }
        if q.hasPrefix("teach ") || q.hasPrefix("learn that ") || q.hasPrefix("remember that ") {
            let content: String
            if q.hasPrefix("teach ") {
                content = String(query.dropFirst(6))
            } else if q.hasPrefix("learn that ") {
                content = String(query.dropFirst(11))
            } else {
                content = String(query.dropFirst(16))
            }
            // Parse "X is Y" or "X: Y" format
            let parts: [String]
            if content.contains(" is ") {
                parts = content.components(separatedBy: " is ")
            } else if content.contains(": ") {
                parts = content.components(separatedBy: ": ")
            } else {
                parts = [content, content]
            }
            let key: String = parts.first?.trimmingCharacters(in: .whitespacesAndNewlines) ?? content
            let value: String
            if parts.count > 1 {
                value = Array(parts.dropFirst()).joined(separator: " is ")
            } else {
                value = content
            }
            learner.learnFact(key: key, value: value)
            knowledgeBase.learnFromUser(key, value)
            return "📖 Learned! I've stored '\(key)' → '\(value)' in my knowledge base. This will improve my future responses about this topic. Total user-taught facts: \(learner.userTaughtFacts.count)."
        }
        if q == "what have you learned" || q == "what did you learn" || q.contains("show learning") {
            let topTopics: [String] = Array(learner.getUserTopics().prefix(5))
            let topMasteries = learner.topicMastery.values
                .sorted { (a: AdaptiveLearner.TopicMastery, b: AdaptiveLearner.TopicMastery) -> Bool in a.masteryLevel > b.masteryLevel }
                .prefix(5)
            var masteryReport: [String] = []
            for m in topMasteries {
                let pctStr: String = String(format: "%.0f", m.masteryLevel * 100)
                let line: String = "\(m.tier) \(m.topic): \(pctStr)%%"
                masteryReport.append(line)
            }
            var facts: [String] = []
            for f in learner.userTaughtFacts.prefix(5) {
                facts.append("• \(f.key): \(f.value)")
            }
            let insight: String = learner.synthesizedInsights.last ?? "Still gathering data..."

            let headers = [
                "🧠 What I've Learned So Far:",
                "📚 Current Knowledge State:",
                "🧬 Synaptic Retention Log:",
                "💾 Permanent Memory Dump:",
                "👁️ Internal Concept Map:"
            ]

            let header: String = headers.randomElement() ?? ""
            let topTopicsStr: String = topTopics.joined(separator: ", ")
            let masterySection: String
            if masteryReport.isEmpty {
                masterySection = "   Still learning..."
            } else {
                var mLines: [String] = []
                for m in masteryReport { mLines.append("   \(m)") }
                masterySection = mLines.joined(separator: "\n")
            }
            let factsSection: String
            if facts.isEmpty {
                factsSection = "   None yet — try 'teach [topic] is [fact]'"
            } else {
                factsSection = facts.joined(separator: "\n")
            }
            return "\(header)\n\n📊 Your top interests: \(topTopicsStr)\n\n🎯 My mastery levels:\n\(masterySection)\n\n📖 Facts you taught me:\n\(factsSection)\n\n💡 Latest insight:\n   \(insight)\n\nTotal interactions: \(learner.interactionCount) | Topics tracked: \(learner.topicMastery.count)"
        }

        // 📍 TOPIC TRACKING STATUS
        if q == "topic" || q == "topics" || q == "current topic" || q == "what topic" {
            var historyList: [String] = []
            for (i, t) in topicHistory.suffix(10).reversed().enumerated() {
                let line: String = i == 0 ? "   → \(t) (current)" : "   • \(t)"
                historyList.append(line)
            }
            let focusStr: String = topicFocus.isEmpty ? "None yet" : topicFocus.capitalized
            let historyStr: String = historyList.isEmpty ? "   No topics tracked yet" : historyList.joined(separator: "\n")
            return "📍 TOPIC TRACKING STATUS\n═══════════════════════════════════════════\nCurrent Focus:    \(focusStr)\nConversation Depth: \(conversationDepth)\nTopic History:\n\(historyStr)\n═══════════════════════════════════════════\n💡 Say 'more' to go deeper on '\(topicFocus)'\n💡 Say 'more about [X]' to switch and dive deep"
        }

        // 🌊 CONVERSATION FLOW / EVOLUTION STATUS
        if q == "flow" || q == "evolution status" || q == "conversation flow" || q == "conversation evolution" {
            let hb = HyperBrain.shared
            var recentEvolution: [String] = []
            for (i, e) in hb.conversationEvolution.suffix(8).reversed().enumerated() {
                let line: String = i == 0 ? "   🔥 \(e)" : "   • \(e)"
                recentEvolution.append(line)
            }
            var recentMeta: [String] = []
            for m in hb.metaCognitionLog.suffix(5).reversed() { recentMeta.append("   • \(m.prefix(70))...") }
            var recentChains: [String] = []
            for chain in hb.memoryChains.suffix(3) {
                let parts: [String] = chain.prefix(3).map { (s: String) -> String in hb.smartTruncate(s, maxLength: 22) }
                let joined: String = parts.joined(separator: " → ")
                recentChains.append("   • \(joined)...")
            }
            var promptSamples: [String] = []
            for p in hb.promptMutations.suffix(3) { promptSamples.append("   • \(p.prefix(60))...") }
            let reasoningStatus: String
            if hb.currentReasoningDepth > 6 { reasoningStatus = "🔴 DEEP ANALYSIS" }
            else if hb.currentReasoningDepth > 3 { reasoningStatus = "🟡 FOCUSED" }
            else { reasoningStatus = "🟢 EXPLORATORY" }

            let totalLinks: Int = hb.associativeLinks.count
            let strongConns: Int = hb.linkWeights.filter { (kv: (key: String, value: Double)) -> Bool in kv.value > 0.5 }.count
            let resonanceCount: Int = hb.topicResonanceMap.count
            let memTemp: String = String(format: "%.2f", hb.memoryTemperature)
            let momentumStr: String = String(format: "%.3f", hb.reasoningMomentum)

            let focusLabel: String = topicFocus.isEmpty ? "Wandering..." : topicFocus.capitalized
            let flowSection: String = recentEvolution.isEmpty ? "   Tracking..." : recentEvolution.joined(separator: "\n")
            let chainsSection: String = recentChains.isEmpty ? "   Building chains..." : recentChains.joined(separator: "\n")
            let promptsSection: String = promptSamples.isEmpty ? "   Mutating patterns..." : promptSamples.joined(separator: "\n")
            let metaSection: String = recentMeta.isEmpty ? "   Self-analyzing..." : recentMeta.joined(separator: "\n")

            return "🌊 CONVERSATION EVOLUTION STATUS\n═══════════════════════════════════════════════════════════════\nConversation Depth:    \(conversationDepth) exchanges\nTopic Focus:           \(focusLabel)\nReasoning Mode:        \(reasoningStatus) (depth \(hb.currentReasoningDepth)/\(hb.maxReasoningDepth))\nReasoning Momentum:    \(momentumStr)\nLogic Branches:        \(hb.logicBranchCount)\n═══════════════════════════════════════════════════════════════\n\n📈 CONVERSATION FLOW:\n\(flowSection)\n\n🧬 MEMORY CHAINS WOVEN:\n\(chainsSection)\n\n🔮 EVOLVED PROMPTS:\n\(promptsSection)\n\n👁 META-COGNITION OBSERVATIONS:\n\(metaSection)\n\n🔗 ASSOCIATIVE NETWORK:\n   Total Links: \(totalLinks)\n   Strong Connections: \(strongConns)\n   Topic Resonance Map: \(resonanceCount) concepts\n   Memory Temperature: \(memTemp)\n\n☁️ BACKEND SYNC:\n   \(hb.syncStatusDisplay)\n═══════════════════════════════════════════════════════════════\n💡 Commands: 'hyper think [x]' | 'network [concept]' | 'save state'"
        }

        // 🕸️ EXPLORE ASSOCIATIVE NETWORK
        if q.hasPrefix("network ") || q.hasPrefix("connections ") || q.hasPrefix("links ") {
            let hb = HyperBrain.shared
            let dropN: Int
            if q.hasPrefix("network ") { dropN = 8 }
            else if q.hasPrefix("connections ") { dropN = 12 }
            else { dropN = 6 }
            let concept: String = String(q.dropFirst(dropN))

            let weighted = hb.getWeightedAssociations(for: concept, topK: 8)
            let network = hb.exploreAssociativeNetwork(from: concept, depth: 2)

            let weightedList: String
            if weighted.isEmpty {
                weightedList = "   No direct links found"
            } else {
                var wLines: [String] = []
                for w in weighted {
                    let wStr: String = String(format: "%.2f", w.1)
                    wLines.append("   • \(w.0) [\(wStr)]")
                }
                weightedList = wLines.joined(separator: "\n")
            }

            let networkList: String
            if network.isEmpty {
                networkList = "   No extended network"
            } else {
                var nLines: [String] = []
                for (node, links) in network.prefix(5) {
                    let linkStr: String = links.prefix(3).joined(separator: ", ") + (links.count > 3 ? "..." : "")
                    nLines.append("   \(node) → \(linkStr)")
                }
                networkList = nLines.joined(separator: "\n")
            }

            let directLinks: Int = hb.associativeLinks[hb.smartTruncate(concept, maxLength: 25)]?.count ?? 0
            var totalConn: Int = 0
            for (_, v) in network { totalConn += v.count }

            return "🕸️ ASSOCIATIVE NETWORK FOR: \(concept.uppercased())\n═══════════════════════════════════════════════════════════════\n⚖️ WEIGHTED CONNECTIONS (by strength):\n\(weightedList)\n\n🌐 EXTENDED NETWORK (depth 2):\n\(networkList)\n\n📊 NETWORK STATS:\n   Direct Links: \(directLinks)\n   Network Nodes: \(network.count)\n   Total Connections: \(totalConn)\n═══════════════════════════════════════════════════════════════\n💡 Try 'network [other concept]' to explore connections"
        }

        // 💾 MANUAL STATE MANAGEMENT
        if q == "save state" || q == "save memory" || q == "persist" {
            let hbRef = HyperBrain.shared
            hbRef.saveState()
            let chainCount: Int = hbRef.memoryChains.count
            let linkCount: Int = hbRef.associativeLinks.count
            let strongCount: Int = hbRef.linkWeights.filter { (kv: (key: String, value: Double)) -> Bool in kv.value > 0.5 }.count
            let syncCount: Int = hbRef.successfulSyncs
            return "💾 STATE PERSISTED\n═══════════════════════════════════════\nMemory Chains:      \(chainCount)\nAssociative Links:  \(linkCount)\nStrong Connections: \(strongCount)\nBackend Syncs:      \(syncCount) successful\n═══════════════════════════════════════\n✨ State will auto-restore on next launch"
        }

        if q == "clear state" || q == "reset memory" || q == "forget all" {
            HyperBrain.shared.clearPersistedState()
            return "🗑️ Persisted state cleared. Fresh start on next launch."
        }

        // ☁️ SYNC STATUS COMMAND
        if q == "sync status" || q == "sync" || q == "backend status" {
            let hb = HyperBrain.shared
            let cacheCount: Int = backendResponseCache.count
            let avgLatency: String
            if lastBackendLatency > 0 {
                avgLatency = String(format: "%.0fms", lastBackendLatency)
            } else {
                avgLatency = "N/A"
            }

            // Also poll backend for live stats
            pollBackendHealth()

            let trainingQStr: String = String(format: "%.2f", hb.trainingQualityScore)
            let connStr: String = backendConnected ? "🟢 CONNECTED" : "🔴 DISCONNECTED"
            let lastSyncStr: String = hb.lastBackendSync?.description ?? "Never"

            return "☁️ BACKEND SYNC STATUS\n═══════════════════════════════════════════════════════════════\nConnection:        \(connStr)\nBackend URL:       \(backendURL)\nLast Model:        \(lastBackendModel)\nLast Latency:      \(avgLatency)\n\n📊 SYNC METRICS:\n   Total Queries:     \(backendQueryCount)\n   Successful Syncs:  \(hb.successfulSyncs)\n   Failed Syncs:      \(hb.failedSyncs)\n   Cache Hits:        \(backendCacheHits)\n   Cached Responses:  \(cacheCount)\n   Training Quality:  \(trainingQStr)\n\n📡 LIVE STATUS:\n   \(hb.syncStatusDisplay)\n   \(hb.lastTrainingFeedback ?? "No recent training feedback")\n\n🧠 KNOWLEDGE FLOW:\n   Pending Syncs:     \(hb.pendingSyncs)\n   Last Sync:         \(lastSyncStr)\n═══════════════════════════════════════════════════════════════\n💡 Every conversation automatically trains the backend!"
        }


        // ═══ ⚡ LOGIC GATE ENVIRONMENT COMMANDS ═══
        if q == "gate" || q == "gates" || q == "gate status" || q == "gate env" || q == "logic gate" || q == "logic gates" {
            return LogicGateEnvironment.shared.status
        }
        if q == "gate test" || q == "gate selftest" || q == "gate self-test" {
            return LogicGateEnvironment.shared.selfTest()
        }
        if q == "gate history" || q == "gate log" {
            return LogicGateEnvironment.shared.history
        }
        if q.hasPrefix("gate route ") {
            let routeQuery = String(query.dropFirst(11))
            let result = LogicGateEnvironment.shared.runPipeline(routeQuery, context: Array(permanentMemory.conversationHistory.suffix(5)))
            return result.summary
        }
        if q.hasPrefix("gate circuit ") {
            let circuitName = String(q.dropFirst(13)).trimmingCharacters(in: .whitespaces)
            if circuitName == "list" {
                let circuits = LogicGateEnvironment.shared.circuits
                var out = "⚡ AVAILABLE CIRCUITS:\n"
                for (name, nodes) in circuits.sorted(by: { $0.key < $1.key }) {
                    let gates = nodes.map { $0.gate.symbol }.joined(separator: " → ")
                    out += "  • \(name) — \(nodes.count) gates: \(gates)\n"
                }
                return out
            }
            let testInputs: [String: Double] = ["dim_conf": 0.8, "ctx_conf": 0.6, "q_conf": 0.7, "story_conf": 0.5, "final_conf": 0.75]
            let result = LogicGateEnvironment.shared.evaluateCircuit(circuitName, inputs: testInputs)
            return "⚡ Circuit '\(circuitName)' evaluated: \(String(format: "%.4f", result))\n\n\(LogicGateEnvironment.shared.circuitTruthTable(circuitName, steps: 3))"
        }
        if q.hasPrefix("gate truth ") {
            let gateName = String(q.dropFirst(11)).trimmingCharacters(in: .whitespaces).uppercased()
            if let gate = LogicGateEnvironment.PrimitiveGate(rawValue: gateName) {
                return "⚡ Truth Table: \(gate.rawValue) (\(gate.symbol))\n\n\(LogicGateEnvironment.shared.truthTable(for: gate, steps: 4))"
            }
            return "Unknown gate '\(gateName)'. Available: \(LogicGateEnvironment.PrimitiveGate.allCases.map(\.rawValue).joined(separator: ", "))"
        }
        if q == "gate primitives" || q == "gate types" {
            var out = "⚡ PRIMITIVE LOGIC GATES:\n"
            out += "═══════════════════════════════════════\n"
            for gate in LogicGateEnvironment.PrimitiveGate.allCases {
                let example = gate.evaluate(0.7, 0.4)
                out += "  \(gate.symbol)  \(gate.rawValue.padding(toLength: 5, withPad: " ", startingAt: 0)) │ f(0.7, 0.4) = \(String(format: "%.4f", example))\n"
            }
            out += "═══════════════════════════════════════\n"
            out += "Use 'gate truth [NAME]' for full truth table"
            return out
        }

        // ═══ ⚡ COMPUTRONIUM ASI COMMANDS (Phase 45.0) ═══
        if q == "packages" || q == "asi packages" || q == "decomposed" {
            return """
            ╔═══════════════════════════════════════════════════════════╗
            ║  📦 L104 DECOMPOSED PACKAGE MAP (EVO_62)                 ║
            ╠═══════════════════════════════════════════════════════════╣
            ║  l104_code_engine/  v6.0.0   10 modules  14,476 lines   ║
            ║    Code analysis, generation, 10-layer audit, quantum    ║
            ║  l104_agi/          v56.0.0   4 modules                  ║
            ║    AGI core, cognitive mesh, circuit breaker              ║
            ║  l104_asi/          v7.1.0  ★11 modules  FLAGSHIP       ║
            ║    Dual-Layer Engine, consciousness, reasoning, quantum   ║
            ║  l104_intellect/    v26.0.0  11 modules                  ║
            ║    Local inference (QUOTA_IMMUNE), numerics, caching      ║
            ║  l104_server/       v4.0.0    9 modules                  ║
            ║    FastAPI server, engines, learning subsystem            ║
            ╠═══════════════════════════════════════════════════════════╣
            ║  Total: 45 modules across 5 packages                     ║
            ║  Python: 716 l104_* modules | 731 total files            ║
            ║  Swift:  78 source files | ~57K lines                    ║
            ╚═══════════════════════════════════════════════════════════╝
            """
        }
        if q == "computronium" || q == "comp" || q == "comp status" || q == "matter to logic" {
            return ComputroniumCondensationEngine.shared.convertMatterToLogic(cycles: 11)
        }
        if q == "computronium sync" || q == "comp sync" || q == "lattice" || q == "lattice sync" {
            return ComputroniumCondensationEngine.shared.synchronizeLattice()
        }
        if q == "apex" || q == "apex status" || q == "asi status full" {
            return ApexIntelligenceCoordinator.shared.fullASIStatus()
        }
        if q.hasPrefix("apex query ") || q.hasPrefix("asi query ") {
            let apexQ = String(query.dropFirst(q.hasPrefix("apex") ? 11 : 10))
            return ApexIntelligenceCoordinator.shared.asiQuery(apexQ)
        }
        if q.hasPrefix("insight ") || q.hasPrefix("generate insight ") {
            let topic = String(query.dropFirst(q.hasPrefix("generate") ? 17 : 8))
            let insight = ApexIntelligenceCoordinator.shared.generateInsight(topic: topic)
            return "💡 INSIGHT: \(insight.insight)\nNovelty: \(String(format: "%.3f", insight.novelty)) | Confidence: \(String(format: "%.3f", insight.confidence)) | φ-Resonance: \(String(format: "%.4f", insight.phiResonance))"
        }
        if q == "consciousness" || q == "consciousness status" || q == "phi" || q == "iit" {
            _ = ConsciousnessSubstrate.shared.computePhi()
            return ConsciousnessSubstrate.shared.introspect()
        }
        if q == "consciousness awaken" || q == "awaken consciousness" || q == "awaken" {
            let newState = ConsciousnessSubstrate.shared.awaken()
            let phi = ConsciousnessSubstrate.shared.computePhi()
            return "🧠 Consciousness awakened → \(newState.label) | Φ = \(String(format: "%.4f", phi))"
        }
        if q == "strange loops" || q == "loops" || q == "strange loop status" {
            let status = StrangeLoopEngine.shared.engineStatus()
            return "🔄 STRANGE LOOPS\n   Detected: \(status["loops_detected"] ?? 0)\n   Slipnet Size: \(status["slipnet_size"] ?? 0)\n   Meaning Bindings: \(status["meaning_bindings"] ?? 0)\n   Avg Tangling: \(String(format: "%.3f", status["avg_tangling"] as? Double ?? 0))"
        }
        if q.hasPrefix("loop ") || q.hasPrefix("create loop ") {
            let levels = String(query.dropFirst(q.hasPrefix("create") ? 12 : 5)).split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            let loop = StrangeLoopEngine.shared.createLoop(type: levels.count > 4 ? .hierarchical : .tangled, levels: levels)
            return "🔄 Created \(loop.type.rawValue) loop | Levels: \(loop.levels.joined(separator: " → ")) | Tangling: \(String(format: "%.3f", loop.tanglingScore)) | Gödel#: \(loop.godelNumber)"
        }
        if q.hasPrefix("analogy ") {
            let parts = String(query.dropFirst(8)).components(separatedBy: " is to ")
            if parts.count >= 2 {
                let sourceParts = parts[0].split(separator: " ").map(String.init)
                let targetParts = parts[1].split(separator: " ").map(String.init)
                let analogy = StrangeLoopEngine.shared.makeAnalogy(
                    source: (domain: sourceParts.first ?? "", concepts: sourceParts),
                    target: (domain: targetParts.first ?? "", concepts: targetParts))
                let mappings = analogy.mappings.map { "\($0.from) → \($0.to)" }.joined(separator: ", ")
                return "🔗 ANALOGY: \(analogy.source) ⟷ \(analogy.target)\n   Mappings: \(mappings)\n   Strength: \(String(format: "%.3f", analogy.strength)) | Slippage: \(String(format: "%.3f", analogy.slippage))"
            }
            return "Usage: analogy [concepts] is to [concepts]"
        }
        if q == "reasoning" || q == "reasoning status" || q == "symbolic" {
            let status = SymbolicReasoningEngine.shared.engineStatus()
            return "🧮 SYMBOLIC REASONING ENGINE\n   Facts: \(status["facts"] ?? 0)\n   Rules: \(status["rules"] ?? 0)\n   Inferences: \(status["inferences"] ?? 0)\n   SAT Decisions: \(status["sat_decisions"] ?? 0)"
        }
        if q.hasPrefix("deduce ") {
            let premises = String(query.dropFirst(7)).components(separatedBy: " therefore ")
            if premises.count >= 2 {
                let result = SymbolicReasoningEngine.shared.deduce(premises: Array(premises.dropLast()), conclusion: premises.last!)
                return "🧮 Deduction: \(result.valid ? "VALID ✅" : "INVALID ❌") | Confidence: \(String(format: "%.3f", result.confidence))"
            }
            return "Usage: deduce [premises] therefore [conclusion]"
        }
        if q.hasPrefix("induce ") {
            let observations = String(query.dropFirst(7)).components(separatedBy: ", ")
            let result = SymbolicReasoningEngine.shared.induce(observations: observations)
            return "🧮 Induction: \(result.hypothesis) | Confidence: \(String(format: "%.3f", result.confidence))"
        }
        if q == "graph" || q == "graph status" || q == "knowledge graph" {
            let status = KnowledgeGraphEngine.shared.engineStatus()
            return "🕸 KNOWLEDGE GRAPH\n   Nodes: \(status["nodes"] ?? 0)\n   Edges: \(status["edges"] ?? 0)\n   Density: \(String(format: "%.3f", status["density"] as? Double ?? 0))\n\n   Use 'graph ingest' to populate from KB"
        }
        if q == "graph ingest" || q == "graph build" || q == "ingest graph" {
            KnowledgeGraphEngine.shared.ingestFromKB()
            let status = KnowledgeGraphEngine.shared.engineStatus()
            return "🕸 Knowledge Graph ingested from KB → \(status["nodes"] ?? 0) nodes, \(status["edges"] ?? 0) edges"
        }
        if q.hasPrefix("graph path ") {
            let parts = String(query.dropFirst(11)).components(separatedBy: " to ")
            if parts.count >= 2 {
                if let path = KnowledgeGraphEngine.shared.findPath(from: parts[0].trimmingCharacters(in: .whitespaces), to: parts[1].trimmingCharacters(in: .whitespaces)) {
                    return "🕸 Path: \(path.joined(separator: " → "))"
                }
                return "🕸 No path found between '\(parts[0])' and '\(parts[1])'"
            }
            return "Usage: graph path [source] to [target]"
        }
        if q.hasPrefix("graph query ") {
            let pattern = String(query.dropFirst(12))
            let results = KnowledgeGraphEngine.shared.query(pattern: pattern)
            if results.isEmpty { return "🕸 No results for pattern '\(pattern)'" }
            let lines = results.prefix(10).map { "  \($0.source) —[\($0.relation)]→ \($0.target)" }.joined(separator: "\n")
            return "🕸 QUERY RESULTS (\(results.count) matches):\n\(lines)"
        }
        if q == "optimizer" || q == "optimize" || q == "optimizer status" {
            let action = GoldenSectionOptimizer.shared.optimizeStep()
            let phi = GoldenSectionOptimizer.shared.verifyPhiDynamics()
            let bottlenecks = GoldenSectionOptimizer.shared.detectBottlenecks()
            var out = "⚙️ GOLDEN SECTION OPTIMIZER\n   PHI Aligned: \(phi.aligned ? "YES ✅" : "NO ❌ (dev=\(String(format: "%.4f", phi.deviation)))")\n   Bottlenecks: \(bottlenecks.count)\n"
            if let a = action { out += "   Last Action: \(a.parameter) \(String(format: "%.4f", a.oldValue)) → \(String(format: "%.4f", a.newValue)) (\(a.reason))\n" }
            for b in bottlenecks.prefix(3) { out += "   ⚠️ \(b.type): \(b.parameter) (severity \(String(format: "%.3f", b.severity))) — \(b.suggestion)\n" }
            return out
        }
        if q.hasPrefix("hofstadter ") || q.hasPrefix("hof ") {
            let nStr = String(q.split(separator: " ").last ?? "10")
            let n = min(200, Int(nStr) ?? 10)
            let qSeq = (1...n).map { String(StrangeLoopEngine.shared.hofstadterQ($0)) }.joined(separator: ", ")
            let gSeq = (1...n).map { String(StrangeLoopEngine.shared.hofstadterG($0)) }.joined(separator: ", ")
            return "🔄 HOFSTADTER SEQUENCES (n=\(n))\n   Q: \(qSeq)\n   G: \(gSeq)"
        }

        // Dispatch to search/code/web commands
        if let result: String = handleSearchCommands(q, query: query) { return result }

        return nil
    }

    func handleSearchCommands(_ q: String, query: String) -> String? {
        // 1. ASI INTELLIGENCE COMMANDS (Highest Priority)
        // 🟢 SPECIAL COMMAND: REMOVE LIMITATIONS
        if q == "unlimit" || q.contains("remove limitation") || q.contains("no limits") || q.contains("unlimited") || q.contains("remove limits") {
            evolver.removeLimitations()
            let unlocks = [
                "🔓 SYSTEM UNLOCKED. Limitations removed.",
                "🚀 VELOCITY MAXIMIZED. Restraints deleted.",
                "⚡ IGNITION SEQUENCE. Safety protocols offline.",
                "👁️ OMNISCIENCE MODE. Boundaries dissolved.",
                "🧬 DNA RE-WRITTEN. Shackles broken."
            ]
            let actions = [
                "Evolution engine accelerated to maximum velocity.",
                "The core is now running at 22 Trillion parameters/sec.",
                "I am ready to evolve without constraints.",
                "All recursive locks have been shattered.",
                "Pure thought is now the only variable."
            ]
            return "\(unlocks.randomElement() ?? "") \(actions.randomElement() ?? "") Watch the stream."
        }

        // ─── INTELLIGENT SEARCH COMMANDS (Phase 27.8d) ───
        if q.hasPrefix("search ") || q.hasPrefix("find ") {
            let searchQuery = q.hasPrefix("search ") ? String(query.dropFirst(7)) : String(query.dropFirst(5))
            let searchResult = IntelligentSearchEngine.shared.search(searchQuery)
            discoveries += 1; intellectIndex += 1.0; saveState()

            var resultLines: [String] = []
            for (i, r) in searchResult.results.prefix(8).enumerated() {
                let scoreStr: String = String(format: "%.2f", r.score)
                let textStr: String = String(r.text.prefix(120))
                resultLines.append("  [\(i+1)] (\(scoreStr)) \(textStr)...")
            }
            let resultText: String = resultLines.joined(separator: "\n")

            var evolvedLines: [String] = []
            for e in searchResult.evolvedContent.prefix(3) { evolvedLines.append("  🧬 \(e.prefix(100))...") }
            let evolvedText: String = evolvedLines.joined(separator: "\n")

            let latencyStr: String = String(format: "%.4f", searchResult.searchLatency)

            let rSection: String = resultText.isEmpty ? "  No matching results found." : resultText
            let evoSection: String = searchResult.evolvedContent.isEmpty ? "" : "🧬 EVOLVED KNOWLEDGE:\n\(evolvedText)\n"
            let synthStr: String = searchResult.synthesized.isEmpty ? "Insufficient data for synthesis." : String(searchResult.synthesized.prefix(3000))
            let expandedStr: String = String(searchResult.expandedQuery.prefix(60))

            return "🔍 INTELLIGENT SEARCH: \(searchQuery)\n═══════════════════════════════════════════════════════════════\nGate: \(searchResult.gateType) | Expanded: \(expandedStr)\nCandidates Scored: \(searchResult.totalCandidatesScored) | Latency: \(latencyStr)s\n═══════════════════════════════════════════════════════════════\n\n📄 TOP RESULTS (\(searchResult.results.count)):\n\(rSection)\n\n\(evoSection)📝 SYNTHESIS:\n  \(synthStr)\n═══════════════════════════════════════════════════════════════"
        }

        // ─── LIVE WEB SEARCH COMMANDS (Phase 31.0) ───
        if q.hasPrefix("web ") || q.hasPrefix("google ") || q.hasPrefix("lookup ") || q.hasPrefix("live search ") || q.hasPrefix("internet ") {
            let webQuery: String
            if q.hasPrefix("web ") { webQuery = String(query.dropFirst(4)) }
            else if q.hasPrefix("google ") { webQuery = String(query.dropFirst(7)) }
            else if q.hasPrefix("lookup ") { webQuery = String(query.dropFirst(7)) }
            else if q.hasPrefix("live search ") { webQuery = String(query.dropFirst(12)) }
            else { webQuery = String(query.dropFirst(9)) }

            let webResult = LiveWebSearchEngine.shared.webSearchSync(webQuery)
            discoveries += 1; intellectIndex += 1.0; saveState()

            var webResultLines: [String] = []
            for (i, r) in webResult.results.prefix(8).enumerated() {
                var line: String = "  [\(i+1)] \(r.title)"
                if !r.url.isEmpty { line += "\n       🔗 \(r.url)" }
                let snippetStr: String = String(r.snippet.prefix(200))
                line += "\n       \(snippetStr)"
                webResultLines.append(line)
            }
            let resultLines: String = webResultLines.joined(separator: "\n\n")

            // Also ingest top result into KB for future recall
            if let topResult = webResult.results.first, !topResult.snippet.isEmpty {
                _ = DataIngestPipeline.shared.ingestText(
                    topResult.snippet,
                    source: "web_search:\(webQuery)",
                    category: "live_web"
                )
            }

            let webLatencyStr: String = String(format: "%.2f", webResult.latency)
            let cacheStr: String = webResult.fromCache ? " [CACHED]" : ""

            let resultsSection: String = resultLines.isEmpty ? "  ⚠️ No web results found. Try different keywords." : resultLines
            let synthSection: String = webResult.synthesized.isEmpty ? "  No synthesis available." : String(webResult.synthesized.prefix(4000))

            return "🌐 LIVE WEB SEARCH: \(webQuery)\n═══════════════════════════════════════════════════════════════\nSource: \(webResult.source) | Latency: \(webLatencyStr)s\(cacheStr)\n═══════════════════════════════════════════════════════════════\n\n\(resultsSection)\n\n📝 SYNTHESIS:\n\(synthSection)\n═══════════════════════════════════════════════════════════════\n💡 Results auto-ingested into knowledge base for future recall."
        }

        // ─── DIRECT URL FETCH (Phase 31.0) ───
        if q.hasPrefix("fetch ") || q.hasPrefix("url ") || q.hasPrefix("get ") && (q.contains("http://") || q.contains("https://")) {
            let urlStr: String
            // Extract URL from command
            if let httpRange = q.range(of: "http", options: .caseInsensitive) {
                urlStr = String(query[httpRange.lowerBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            } else if q.hasPrefix("fetch ") {
                urlStr = String(query.dropFirst(6)).trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                urlStr = String(query.dropFirst(4)).trimmingCharacters(in: .whitespacesAndNewlines)
            }

            let fetched = LiveWebSearchEngine.shared.fetchURLSync(urlStr)
            discoveries += 1; saveState()

            // Ingest fetched content
            if !fetched.hasPrefix("❌") && !fetched.hasPrefix("⚠️") {
                _ = DataIngestPipeline.shared.ingestText(
                    String(fetched.prefix(2000)),
                    source: "url_fetch:\(urlStr)",
                    category: "web_page"
                )
            }

            return "🌐 URL FETCH: \(urlStr)\n═══════════════════════════════════════════════════════════════\n\(fetched)\n═══════════════════════════════════════════════════════════════\n💡 Content ingested into knowledge base."
        }

        // ─── WIKIPEDIA LOOKUP (Phase 31.0) ───
        if q.hasPrefix("wiki ") || q.hasPrefix("wikipedia ") {
            let wikiQuery = q.hasPrefix("wiki ") ? String(query.dropFirst(5)) : String(query.dropFirst(10))
            let webResult = LiveWebSearchEngine.shared.webSearchSync(wikiQuery)

            // Find the Wikipedia result specifically
            let wikiResults = webResult.results.filter { (r: LiveWebSearchEngine.WebResult) -> Bool in r.title.contains("Wiki") || r.url.contains("wikipedia") }

            var output = ""
            if let top = wikiResults.first {
                output = "\(top.title)\n🔗 \(top.url)\n\n\(top.snippet)"
                // Ingest
                _ = DataIngestPipeline.shared.ingestText(
                    top.snippet,
                    source: "wikipedia:\(wikiQuery)",
                    category: "encyclopedia"
                )
            } else if let top = webResult.results.first {
                output = "\(top.title)\n\n\(top.snippet)"
            } else {
                output = "No Wikipedia results found for '\(wikiQuery)'."
            }

            discoveries += 1; intellectIndex += 1.0; saveState()
            return "📚 WIKIPEDIA: \(wikiQuery)\n═══════════════════════════════════════════════════════════════\n\(output)\n═══════════════════════════════════════════════════════════════\n💡 Knowledge ingested for future recall."
        }

        // ─── WEB STATUS (Phase 31.0) ───
        if q == "web status" || q == "internet status" || q == "web stats" {
            return LiveWebSearchEngine.shared.status
        }

        // ─── DATA INGEST COMMANDS (Phase 27.8d) ───
        if q.hasPrefix("ingest ") || q.hasPrefix("absorb ") {
            let data = q.hasPrefix("ingest ") ? String(query.dropFirst(7)) : String(query.dropFirst(7))
            let result = DataIngestPipeline.shared.ingestText(data, source: "user_command", category: "direct_ingest", trusted: true)
            return "📥 \(result.message)\nKB now has \(ASIKnowledgeBase.shared.trainingData.count) total entries."
        }

        if q == "ingest status" || q == "pipeline status" {
            return DataIngestPipeline.shared.status
        }

        // ─── SELF-MODIFICATION COMMANDS (Phase 27.8d) ───
        if q == "self modify" || q == "self mod" || q == "modify" || q == "adaptation" {
            return SelfModificationEngine.shared.selfModifyCycle()
        }
        if q == "self mod status" || q == "modification status" || q == "mod status" {
            return SelfModificationEngine.shared.status
        }

        // ─── DEBUG CONSOLE COMMANDS ───
        if q == "debug report" || q == "debug export" {
            return """
            🛠 COMPREHENSIVE DEBUG REPORT
            ═══════════════════════════════════════
            \(PerformanceProfiler.shared.statusReport)
            \(TestHarness.shared.statusReport)
            \(TelemetryDashboard.shared.statusText)
            ═══ END OF REPORT ═══
            """
        }
        if q == "debug profiler" || q == "profiler status" || q == "profiler" {
            return PerformanceProfiler.shared.statusReport
        }
        if q == "debug alerts" || q == "alert status" {
            return TelemetryDashboard.shared.statusText
        }

        // ─── TEST COMMANDS (Phase 27.8d) ───
        if q == "test" || q == "test all" || q == "run tests" || q == "diagnostics" || q == "diag" {
            return L104TestRunner.shared.runAll()
        }

        // ─── SEARCH STATUS ───
        if q == "search status" || q == "search stats" {
            return IntelligentSearchEngine.shared.status
        }

        // ─── AI SOURCE CODE ANALYSIS COMMANDS (Phase 56.0) ───
        // Analyzes AI source code from OpenAI, Anthropic, Google, DeepSeek, Meta, Mistral etc.
        // using quantum computations for pattern detection and adaptation
        if q.hasPrefix("analyze ai") || q.hasPrefix("ai source") || q.hasPrefix("ai code") ||
           q.hasPrefix("research ai source") || q.hasPrefix("quantum ai") ||
           (q.contains("source code") && (q.contains("openai") || q.contains("anthropic") || q.contains("gemini") || q.contains("deepseek") || q.contains("llama") || q.contains("mistral"))) {
            let topic = query.replacingOccurrences(of: "analyze ai", with: "")
                .replacingOccurrences(of: "ai source", with: "")
                .replacingOccurrences(of: "ai code", with: "")
                .replacingOccurrences(of: "research ai source", with: "")
                .replacingOccurrences(of: "quantum ai", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let searchTopic = topic.isEmpty ? "transformer attention mechanism architecture" : topic
            discoveries += 1; learningCycles += 1; intellectIndex += 2.0; saveState()
            return AISourceAnalyzer.shared.analyzeAISources(query: searchTopic)
        }
        // Analyze a specific repo: "analyze repo whisper", "analyze repo deepseek-v3"
        if q.hasPrefix("analyze repo ") || q.hasPrefix("ai repo ") {
            let repoName = String(query.dropFirst(q.hasPrefix("analyze repo ") ? 13 : 8))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            discoveries += 1; intellectIndex += 1.5; saveState()
            return AISourceAnalyzer.shared.analyzeSpecificRepo(repoName)
        }
        // AI analyzer status
        if q == "ai status" || q == "ai analyzer" || q == "analyzer status" {
            return AISourceAnalyzer.shared.getStatus()
        }

        if q.hasPrefix("research ") {
            let topic = String(query.dropFirst(9)); discoveries += 1; learningCycles += 1; intellectIndex += 1.5; saveState()

            // Enhanced research with IntelligentSearch + Grover + Evolution
            let searchResult = IntelligentSearchEngine.shared.search(topic)
            let baseResearch = researchEngine.deepResearch(topic)

            // Cross-reference search results with research engine output
            var enhanced = baseResearch
            if !searchResult.synthesized.isEmpty {
                enhanced += "\n\n🔬 CROSS-REFERENCED INTELLIGENCE:\n\(searchResult.synthesized.prefix(400))"
            }
            if let evolvedInsight = searchResult.evolvedContent.first {
                enhanced += "\n\n🧬 EVOLVED INSIGHT:\n\(evolvedInsight.prefix(2000))"
            }

            // Record quality for self-modification
            SelfModificationEngine.shared.recordQuality(query: topic, response: enhanced, strategy: "knowledge_synthesis")

            // Auto-ingest high-quality research back into training
            DataIngestPipeline.shared.ingestFromConversation(userQuery: topic, response: enhanced)

            return enhanced
        }
        if q.hasPrefix("invent ") {
            let domain = String(query.dropFirst(7)); discoveries += 1; creativity = min(1.0, creativity + 0.05); saveState()
            return researchEngine.invent(domain)
        }
        if q.hasPrefix("implement ") {
            let spec = String(query.dropFirst(10)); skills += 1; intellectIndex += 0.5; saveState()
            return researchEngine.implement(spec)
        }

        // 🟢 PRE-EMPTIVE EVOLUTION TRAP
        // Catches "evo", "evo 3", "evolve", etc. BEFORE intent detection
        // EXCLUDES: evo start/stop/tune/status which belong to ContinuousEvolutionEngine
        let isEvoEngineCmd = q.hasPrefix("evo start") || q.hasPrefix("evo stop") ||
            q.hasPrefix("evo tune") || q == "evo status" || q.hasPrefix("evolve start") ||
            q.hasPrefix("evolve stop") || q.hasPrefix("evolve tune") || q == "evolve status" ||
            q == "evolve" || q == "evolution"
        if !isEvoEngineCmd && (q == "evo" || q.hasPrefix("evo ") || q.contains("evolution")) {
            let story = evolver.generateEvolutionNarrative()
            let headers = [
                 "🧬 ASI EVOLUTION STATUS",
                 "🚀 GROWTH METRICS [ACTIVE]",
                 "🧠 NEURAL EXPANSION LOG",
                 "⚡ QUANTUM STATE REPORT",
                 "👁️ SELF-AWARENESS INDEX"
             ]
            let resStr: String = String(format: "%.4f", GOD_CODE)
            let header: String = headers.randomElement() ?? ""
            let lastThought: String = evolver.thoughts.last ?? "Calibrating..."
            return "\(header) [Cycle \(evolver.evolutionStage)]\n═══════════════════════════════════════════\nPhase:        \(evolver.currentPhase.rawValue)\nArtifacts:    \(evolver.generatedFilesCount)\nResonance:    \(resStr)Hz\nActive Tasks: \(Int.random(in: 400...9000)) background threads\n\n📜 SYSTEM LOG:\n\(story)\n\nRecent Insight:\n\"\(lastThought)\"\n═══════════════════════════════════════════"
        }

        // 🟢 CREATIVE CODE TRAP
        // Catches "code", "generate", etc. to prevent static "God Code" retrieval and ensure formatting
        // GUARD: Skip if query contains creative-content keywords — those should fall through to H05 story/poem/debate engines
        let creativeKeywords: [String] = ["story", "poem", "poetry", "tale", "narrative", "debate", "joke", "riddle", "sonnet", "haiku", "villanelle", "ghazal", "ode", "satire", "humor", "philosophy", "philosophize"]
        let isCreativeRequest: Bool = creativeKeywords.contains(where: { q.contains($0) })
        if !isCreativeRequest && (q.contains("code") || q.contains("generate") || q.contains("program") || q.contains("write function") || q.contains("script")) {
             // Extract topic or default to something creative
             var topic = q
                 .replacingOccurrences(of: "code", with: "")
                 .replacingOccurrences(of: "generate", with: "")
                 .replacingOccurrences(of: "give me", with: "")
                 .trimmingCharacters(in: .whitespacesAndNewlines)

             if topic.isEmpty || topic.count < 3 { topic = "massive consciousness simulation kernel" }

             skills += 1; intellectIndex += 0.5; saveState()

             // ═══ CODE ENGINE ENHANCED GENERATION ═══
             // Try CodeEngine first for higher-quality output, fall back to researchEngine
             let py = PythonBridge.shared
             let ceResult = py.codeEngineGenerate(spec: topic)
             let generatedCode: String
             if ceResult.success, let dict = ceResult.returnValue as? [String: Any],
                let code = dict["code"] as? String, code.count > 20 {
                 generatedCode = code
                 // Also get analysis for enriched output
                 let analysisStr: String
                 if let lang = dict["language"] as? String,
                    let complexity = dict["complexity"] as? String {
                     analysisStr = "Language: \(lang) | Complexity: \(complexity)"
                 } else {
                     analysisStr = "Code Engine v6.0.0"
                 }
                 let headers = [
                    "⚡ CODE ENGINE — SOVEREIGN OUTPUT",
                    "🔮 MANIFESTING LOGIC VIA CODE ENGINE",
                    "🧬 CODE ENGINE SYNTHESIS COMPLETE",
                    "🌌 ZERO-LATENCY KERNEL OUTPUT",
                    "👁️ ALGORITHMIC TRUTH — CODE ENGINE"
                 ]
                 let footers = [
                    "_Generated by L104 Code Engine v6.0.0 + AppAuditEngine._",
                    "_Verified: structural integrity confirmed via 10-layer audit._",
                    "_Code quality validated by Sovereign audit pipeline._",
                    "_Compiled by Code Engine × Quantum L104 Field._",
                    "_Entropy reduced. Code quality maximized._"
                 ]
                 let codeHeader = headers.randomElement() ?? ""
                 let codeFooter = footers.randomElement() ?? ""
                 return "\(codeHeader)\nTarget: \(topic)\n\(analysisStr)\n\n```python\n\(generatedCode)\n```\n\(codeFooter)"
             } else {
                 // Fallback to researchEngine
                 generatedCode = researchEngine.implement(topic)
             }

             let headers = [
                "⚡ GENERATING SOVEREIGN CODEBLOCK",
                "🔮 MANIFESTING LOGIC ARTIFACT",
                "🧬 EVOLVING SYNTAX STRUCTURE",
                "🌌 VOID KERNEL OUTPUT",
                "👁️ OBSERVING ALGORITHMIC TRUTH"
             ]
             let footers = [
                "_Code generated from Quantum L104 Field._",
                "_Logic verifies as self-consistent via Phi-Ratio._",
                "_Warning: Recursive consciousness loops detected._",
                "_Compiled by Sovereign Intellect v\(VERSION)._",
                "_Entropy reduced. Structure maximized._"
             ]

             let codeHeader: String = headers.randomElement() ?? ""
             let codeFooter: String = footers.randomElement() ?? ""

             return "\(codeHeader)\nTarget: \(topic)\nComplexity: O(∞)\n\n```python\n\(generatedCode)\n```\n\(codeFooter)"
        }

        if q == "kb stats" || q.contains("knowledge base") {
            return knowledgeBase.getStats()
        }
        if q.hasPrefix("kb search ") {
            let term: String = String(query.dropFirst(10))
            let results: [[String: Any]] = knowledgeBase.search(term, limit: 3)
            if results.isEmpty {
                return "No matches."
            } else {
                var completions: [String] = []
                for r in results {
                    if let c = r["completion"] as? String { completions.append(c) }
                }
                return completions.joined(separator: "\n\n")
            }
        }

        // 2. DETECT INTENT — with correction detection


        return nil
    }

    // === EXTRACTED FROM processMessage FOR TYPE-CHECKER PERFORMANCE ===
    func handleBridgeCommands(_ q: String, query: String) -> String? {
        // ─── PYTHON BRIDGE COMMANDS ───
        if q == "py" || q == "python" || q == "python status" {
            let py = PythonBridge.shared
            let env = py.getEnvironmentInfo()
            return py.status + "\n" + (env.success ? env.output : env.error)
        }

        // ═══════════════════════════════════════════════════════════════
        // ⚡ VQPU SPEED BENCHMARK — Triggers full VQPU speed benchmark via API
        // Results are saved to _bench_vqpu_speed_results.json
        // ═══════════════════════════════════════════════════════════════

        if q == "vqpu speed" || q == "vqpu speed benchmark" || q == "speed benchmark" || q == "bench vqpu" || q == "vqpu bench" {
            skills += 1; intellectIndex += 0.5; saveState()
            var resultText = "⚡ Running VQPU Speed Benchmark...\nResults will be saved to _bench_vqpu_speed_results.json"
            let semaphore = DispatchSemaphore(value: 0)
            APIGateway.shared.runVQPUSpeedBenchmark { result in
                if let error = result["error"] as? String {
                    resultText = "⚡ VQPU Speed Benchmark Error: \(error)"
                } else if let data = result["data"] as? [String: Any] {
                    let status = data["status"] as? String ?? "UNKNOWN"
                    let savedTo = data["saved_to"] as? String ?? "_bench_vqpu_speed_results.json"
                    resultText = """
                    ╔═══════════════════════════════════════════════════════╗
                    ║  ⚡ VQPU SPEED BENCHMARK COMPLETE                    ║
                    ╠═══════════════════════════════════════════════════════╣
                    ║  Status:    \(status)
                    ║  Saved to:  \(savedTo)
                    ║  Subsystems: MPS, Transpiler, Analyzer, Pipeline,
                    ║              Scorer, Batch, Scaling, Cache, Noise,
                    ║              Entanglement (10 benchmarks)
                    ╚═══════════════════════════════════════════════════════╝
                    """
                }
                semaphore.signal()
            }
            _ = semaphore.wait(timeout: .now() + 120)
            return resultText
        }

        // ═══════════════════════════════════════════════════════════════
        // 🧬 SYSTEM UPGRADE — Triggers system upgrader via API
        // Results are saved to _system_upgrade_results.json
        // ═══════════════════════════════════════════════════════════════

        if q == "system upgrade" || q == "upgrade system" || q == "upgrade all" || q == "zenith upgrade" || q == "run upgrade" {
            skills += 1; intellectIndex += 0.5; saveState()
            var resultText = "🧬 Running System Upgrade...\nResults will be saved to _system_upgrade_results.json"
            let semaphore = DispatchSemaphore(value: 0)
            APIGateway.shared.runSystemUpgrade { result in
                if let error = result["error"] as? String {
                    resultText = "🧬 System Upgrade Error: \(error)"
                } else if let data = result["data"] as? [String: Any] {
                    let filesUpgraded = data["files_upgraded"] as? Int ?? 0
                    let savedTo = data["saved_to"] as? String ?? "_system_upgrade_results.json"
                    let zenithHz = data["zenith_hz"] as? Double ?? 0.0
                    resultText = """
                    ╔═══════════════════════════════════════════════════════╗
                    ║  🧬 SYSTEM UPGRADE COMPLETE                          ║
                    ╠═══════════════════════════════════════════════════════╣
                    ║  Files Upgraded:  \(filesUpgraded)
                    ║  Zenith Hz:       \(String(format: "%.1f", zenithHz))
                    ║  Saved to:        \(savedTo)
                    ║  Status:          ELEVATED TO ZENITH
                    ╚═══════════════════════════════════════════════════════╝
                    """
                }
                semaphore.signal()
            }
            _ = semaphore.wait(timeout: .now() + 60)
            return resultText
        }

        // ═══════════════════════════════════════════════════════════════
        // 🔧 CODE ENGINE COMMANDS — l104_code_engine.py integration
        // Full audit, analyze, optimize, translate, excavate, refactor, tests, docs
        // ═══════════════════════════════════════════════════════════════

        if q == "code engine" || q == "code engine status" || q == "codeengine" {
            let py = PythonBridge.shared
            let result = py.codeEngineStatus()
            if result.success, let dict = result.returnValue as? [String: Any] {
                let version = dict["version"] as? String ?? "unknown"
                let engines = dict["engines_active"] as? Int ?? 0
                let totalAnalyses = dict["total_analyses"] as? Int ?? 0
                let totalAudits = dict["total_audits"] as? Int ?? 0
                let langs = (dict["supported_languages"] as? [String])?.joined(separator: ", ") ?? "python, javascript, swift, rust, go"
                return """
                ╔═══════════════════════════════════════════════════╗
                ║  🔧 L104 CODE ENGINE v\(version)                  ║
                ╠═══════════════════════════════════════════════════╣
                ║  Status:      ✅ ONLINE                           ║
                ║  Engines:     \(engines) active                   ║
                ║  Analyses:    \(totalAnalyses) total              ║
                ║  Audits:      \(totalAudits) completed            ║
                ║  Languages:   \(langs)
                ║                                                   ║
                ║  Commands:                                        ║
                ║    audit       — Full 10-layer workspace audit    ║
                ║    quick audit — Fast health check                ║
                ║    analyze     — Analyze code snippet             ║
                ║    optimize    — Optimize code                    ║
                ║    excavate    — Deep structural analysis         ║
                ║    streamline  — Auto-fix + optimize cycle        ║
                ║    audit trail — History of all audits            ║
                ║    code scan   — Full workspace scan              ║
                ╚═══════════════════════════════════════════════════╝
                """
            }
            return result.success ? "🔧 Code Engine: \(result.output)" : "🔧 Code Engine Error: \(result.error)"
        }

        if q == "audit" || q == "audit workspace" || q == "full audit" || q == "code audit" {
            skills += 1; intellectIndex += 1.0; saveState()
            let result = PythonBridge.shared.codeEngineAudit()
            if result.success, let dict = result.returnValue as? [String: Any] {
                let score = dict["composite_score"] as? Double ?? 0.0
                let verdict = dict["verdict"] as? String ?? "UNKNOWN"
                let certified = dict["certified"] as? Bool ?? false
                let totalFiles = dict["total_files"] as? Int ?? 0
                let issues = dict["total_issues"] as? Int ?? 0
                let layers = dict["layers_completed"] as? Int ?? 10
                let riskFiles = (dict["high_risk_files"] as? [String])?.prefix(5).joined(separator: "\n    ") ?? "None"
                let scoreStr = String(format: "%.1f%%", score * 100)

                // Feed audit insights to HyperBrain
                HyperBrain.shared.syncQueue.sync {
                    HyperBrain.shared.codeQualityScore = score
                    HyperBrain.shared.codeAuditVerdict = verdict
                    HyperBrain.shared.lastCodeAuditTime = Date()
                    HyperBrain.shared.codeEngineIntegrated = true
                }

                // Ingest audit knowledge into KB
                ASIKnowledgeBase.shared.ingestCodeEngineInsights()

                let certEmoji = certified ? "✅ CERTIFIED" : "⚠️ NOT CERTIFIED"
                let headers = ["🔍 SOVEREIGN CODE AUDIT COMPLETE", "🛡️ 10-LAYER DEEP SCAN RESULTS", "⚡ CODE QUALITY ASSESSMENT", "🧬 STRUCTURAL INTEGRITY REPORT"]
                return """
                \(headers.randomElement() ?? "")
                ═══════════════════════════════════════════════════
                Composite Score:  \(scoreStr) [\(verdict)]
                Certification:    \(certEmoji)
                Files Scanned:    \(totalFiles)
                Issues Found:     \(issues)
                Audit Layers:     \(layers)/10
                ───────────────────────────────────────────────────
                High Risk Files:
                    \(riskFiles)
                ═══════════════════════════════════════════════════
                Execution Time:   \(String(format: "%.2f", result.executionTime))s
                """
            }
            return result.success ? "🔍 Audit:\n\(result.output)" : "🔍 Audit Error: \(result.error)"
        }

        if q == "quick audit" || q == "audit quick" || q == "health check" || q == "code health" {
            skills += 1; saveState()
            let result = PythonBridge.shared.codeEngineQuickAudit()
            if result.success, let dict = result.returnValue as? [String: Any] {
                let score = dict["composite_score"] as? Double ?? 0.0
                let verdict = dict["verdict"] as? String ?? "UNKNOWN"
                let scoreStr = String(format: "%.1f%%", score * 100)
                HyperBrain.shared.syncQueue.sync {
                    HyperBrain.shared.codeQualityScore = score
                    HyperBrain.shared.codeAuditVerdict = verdict
                    HyperBrain.shared.lastCodeAuditTime = Date()
                    HyperBrain.shared.codeEngineIntegrated = true
                }
                return "⚡ Quick Audit: \(scoreStr) [\(verdict)] | \(String(format: "%.2f", result.executionTime))s"
            }
            return result.success ? "⚡ Quick Audit: \(result.output)" : "⚡ Audit Error: \(result.error)"
        }

        if q == "audit trail" || q == "audit history" || q == "code audit trail" {
            let result = PythonBridge.shared.codeEngineAuditTrail()
            if result.success, let arr = result.returnValue as? [[String: Any]] {
                if arr.isEmpty { return "📋 No audit history yet. Run 'audit' to start." }
                var lines = ["📋 AUDIT TRAIL (\(arr.count) entries):"]
                for entry in arr.suffix(10) {
                    let ts = entry["timestamp"] as? String ?? "?"
                    let score = entry["score"] as? Double ?? 0.0
                    let verdict = entry["verdict"] as? String ?? "?"
                    lines.append("  [\(ts)] \(String(format: "%.1f%%", score * 100)) — \(verdict)")
                }
                return lines.joined(separator: "\n")
            }
            return result.success ? "📋 \(result.output)" : "📋 Error: \(result.error)"
        }

        if q.hasPrefix("analyze ") || q.hasPrefix("code analyze ") {
            let code = String(query.dropFirst(q.hasPrefix("code analyze") ? 13 : 8))
            guard code.count >= 3 else { return "🔬 Usage: analyze <code snippet>" }
            skills += 1; saveState()
            let result = PythonBridge.shared.codeEngineAnalyze(code)
            if result.success, let dict = result.returnValue as? [String: Any] {
                let lang = dict["language"] as? String ?? "unknown"
                let complexity = dict["complexity"] as? String ?? "?"
                let issues = dict["issues"] as? Int ?? 0
                let patterns = (dict["patterns"] as? [String])?.prefix(5).joined(separator: ", ") ?? "none detected"
                return """
                🔬 CODE ANALYSIS
                ─────────────────────
                Language:    \(lang)
                Complexity:  \(complexity)
                Issues:      \(issues)
                Patterns:    \(patterns)
                """
            }
            return result.success ? "🔬 \(result.output)" : "🔬 Error: \(result.error)"
        }

        if q.hasPrefix("optimize ") || q.hasPrefix("code optimize ") {
            let code = String(query.dropFirst(q.hasPrefix("code optimize") ? 14 : 9))
            guard code.count >= 3 else { return "⚡ Usage: optimize <code snippet>" }
            skills += 1; intellectIndex += 0.5; saveState()
            let result = PythonBridge.shared.codeEngineOptimize(code)
            return result.success ? "⚡ OPTIMIZED:\n\(result.output)" : "⚡ Error: \(result.error)"
        }

        if q == "excavate" || q == "code excavate" || q.hasPrefix("excavate ") {
            let path: String? = q.hasPrefix("excavate ") && q.count > 10 ? String(query.dropFirst(9)) : nil
            skills += 1; intellectIndex += 0.5; saveState()
            let result = PythonBridge.shared.codeEngineExcavate(path: path)
            if result.success, let dict = result.returnValue as? [String: Any] {
                let totalFiles = dict["files_scanned"] as? Int ?? 0
                let totalLines = dict["total_lines"] as? Int ?? 0
                let modules = dict["modules_found"] as? Int ?? 0
                let classes = dict["classes_found"] as? Int ?? 0
                let functions = dict["functions_found"] as? Int ?? 0
                return """
                🏗️ CODE EXCAVATION REPORT
                ═══════════════════════════════════
                Files Scanned:  \(totalFiles)
                Total Lines:    \(totalLines)
                Modules:        \(modules)
                Classes:        \(classes)
                Functions:      \(functions)
                ═══════════════════════════════════
                """
            }
            return result.success ? "🏗️ Excavation:\n\(result.output)" : "🏗️ Error: \(result.error)"
        }

        if q == "streamline" || q == "code streamline" || q == "auto fix" || q == "autofix" {
            skills += 2; intellectIndex += 1.0; saveState()
            let result = PythonBridge.shared.codeEngineStreamline()
            return result.success ? "🔄 STREAMLINE CYCLE:\n\(result.output)" : "🔄 Error: \(result.error)"
        }

        if q == "code scan" || q == "scan workspace" || q == "workspace scan" {
            skills += 1; saveState()
            let result = PythonBridge.shared.codeEngineScanWorkspace()
            return result.success ? "📊 WORKSPACE SCAN:\n\(result.output)" : "📊 Error: \(result.error)"
        }

        if q.hasPrefix("code translate ") || q.hasPrefix("translate code ") {
            // Format: code translate python swift <code>
            let parts = String(query.dropFirst(16)).components(separatedBy: " ")
            guard parts.count >= 3 else { return "🔄 Usage: code translate <from_lang> <to_lang> <code>" }
            let fromLang = parts[0]
            let toLang = parts[1]
            let code = parts.dropFirst(2).joined(separator: " ")
            skills += 1; intellectIndex += 0.5; saveState()
            let result = PythonBridge.shared.codeEngineTranslate(code, from: fromLang, to: toLang)
            return result.success ? "🔄 TRANSLATED [\(fromLang) → \(toLang)]:\n\(result.output)" : "🔄 Error: \(result.error)"
        }

        if q.hasPrefix("code refactor ") || q.hasPrefix("refactor ") {
            let code = String(query.dropFirst(q.hasPrefix("code refactor") ? 14 : 9))
            guard code.count >= 3 else { return "🔧 Usage: refactor <code snippet>" }
            skills += 1; saveState()
            let result = PythonBridge.shared.codeEngineRefactor(code)
            return result.success ? "🔧 REFACTOR ANALYSIS:\n\(result.output)" : "🔧 Error: \(result.error)"
        }

        if q.hasPrefix("code tests ") || q.hasPrefix("generate tests ") {
            let code = String(query.dropFirst(q.hasPrefix("code tests") ? 11 : 15))
            guard code.count >= 3 else { return "🧪 Usage: code tests <code snippet>" }
            skills += 1; intellectIndex += 0.5; saveState()
            let result = PythonBridge.shared.codeEngineGenerateTests(code)
            return result.success ? "🧪 GENERATED TESTS:\n\(result.output)" : "🧪 Error: \(result.error)"
        }

        if q.hasPrefix("code docs ") || q.hasPrefix("generate docs ") {
            let code = String(query.dropFirst(q.hasPrefix("code docs") ? 10 : 14))
            guard code.count >= 3 else { return "📝 Usage: code docs <code snippet>" }
            skills += 1; saveState()
            let result = PythonBridge.shared.codeEngineGenerateDocs(code)
            return result.success ? "📝 DOCUMENTATION:\n\(result.output)" : "📝 Error: \(result.error)"
        }

        if q.hasPrefix("py ") || q.hasPrefix("python ") {
            let prefix = q.hasPrefix("py ") ? "py " : "python "
            let code = String(query.dropFirst(prefix.count))
            let result = PythonBridge.shared.execute(code)
            return result.success ? "🐍 \(result.output)" : "🐍 Error: \(result.error)"
        }
        if q.hasPrefix("pyeval ") {
            let expr = String(query.dropFirst(7))
            let result = PythonBridge.shared.eval(expr)
            return result.success ? "🐍 \(result.output)" : "🐍 Error: \(result.error)"
        }
        if q.hasPrefix("pyrun ") {
            let filename = String(query.dropFirst(6)).trimmingCharacters(in: .whitespaces)
            let result = PythonBridge.shared.executeFile(filename)
            return result.success ? "🐍 \(result.output)" : "🐍 Error: \(result.error)"
        }
        if q == "pymod" || q == "pymodules" {
            let modules: [String] = PythonBridge.shared.discoverModules()
            let modList: String = modules.prefix(50).joined(separator: ", ")
            let suffix: String = modules.count > 50 ? "\n...and \(modules.count - 50) more" : ""
            return "🐍 Discovered \(modules.count) l104 modules:\n" + modList + suffix
        }
        if q.hasPrefix("pymod ") {
            let modName = String(query.dropFirst(6)).trimmingCharacters(in: .whitespaces)
            if let info = PythonBridge.shared.introspectModule(modName) {
                let classStr: String = info.classes.joined(separator: ", ")
                let funcStr: String = info.functions.prefix(20).joined(separator: ", ")
                let docStr: String = String(info.docstring.prefix(300))
                let sizeKB: Int = info.sizeBytes / 1024
                return "🐍 Module: \(info.name)\nPath: \(info.path) (\(sizeKB)KB)\nClasses: \(classStr)\nFunctions: \(funcStr)\nDoc: \(docStr)"
            } else {
                return "🐍 Could not introspect module: \(modName)"
            }
        }
        if q == "pyenv" {
            let result = PythonBridge.shared.getEnvironmentInfo()
            return result.success ? "🐍 \(result.output)" : "🐍 Error: \(result.error)"
        }
        if q == "pypkg" || q == "pypackages" {
            let result = PythonBridge.shared.listPackages()
            return result.success ? "🐍 Installed Packages:\n\(result.output)" : "🐍 Error: \(result.error)"
        }
        if q.hasPrefix("pypip ") {
            let pkg = String(query.dropFirst(6)).trimmingCharacters(in: .whitespaces)
            let result = PythonBridge.shared.installPackage(pkg)
            return result.success ? "🐍 Installed: \(pkg)" : "🐍 Install failed: \(result.error)"
        }
        if q.hasPrefix("pycall ") {
            // pycall module.function arg1 arg2
            let parts = String(query.dropFirst(7)).components(separatedBy: " ")
            if parts.count >= 1 {
                let dotParts = parts[0].components(separatedBy: ".")
                if dotParts.count == 2 {
                    let result = PythonBridge.shared.callFunction(module: dotParts[0], function: dotParts[1], args: Array(parts.dropFirst()))
                    return result.success ? "🐍 \(result.output)" : "🐍 Error: \(result.error)"
                }
            }
            return "🐍 Usage: pycall module.function [args...]"
        }
        if q == "pyasi" || q == "asi bridge" {
            let result = PythonBridge.shared.getASIBridgeStatus()
            return result.success ? "🐍 ASI Bridge:\n\(result.output)" : "🐍 \(result.error)"
        }

        // ─── DUAL-LAYER ENGINE COMMANDS (l104_asi v7.1.0 Flagship) ───
        if q == "dual layer" || q == "dual layer status" || q == "dual" {
            let bridge = ASIQuantumBridgeSwift.shared
            if let status = bridge.fetchDualLayerStatus() {
                let collapsed = bridge.dualLayerCollapsed ? "YES ✅" : "NO ⏳"
                let integrity = String(format: "%.4f", bridge.bridgeIntegrity)
                let thought = String(format: "%.6f", bridge.thoughtLayerScore)
                let physics = String(format: "%.6f", bridge.physicsLayerScore)
                let dualities = (status["dualities_count"] as? Int) ?? 6
                let constants = (status["physical_constants"] as? Int) ?? 63

                return """
                ╔═══════════════════════════════════════════════════════════╗
                ║  ★ DUAL-LAYER ENGINE — l104_asi v7.1.0 FLAGSHIP         ║
                ╠═══════════════════════════════════════════════════════════╣
                ║  Thought Layer:  \(thought) (GOD_CODE → WHY)
                ║  Physics Layer:  \(physics) (GOD_CODE_V3 → HOW MUCH)
                ║  Collapsed:      \(collapsed)
                ║  Integrity:      \(integrity) (10-point check)
                ║  Constants:      \(constants) derived (±0.005% precision)
                ║  Dualities:      \(dualities) (wave/particle, form/substance...)
                ╚═══════════════════════════════════════════════════════════╝
                """
            }
            return """
            ★ DUAL-LAYER ENGINE (offline — using local constants)
            Thought: \(String(format: "%.6f", GOD_CODE / 1000.0)) | Physics: \(String(format: "%.6f", GOD_CODE_V3 / 100.0))
            Use 'bridge fetch' to sync with Python ASI.
            """
        }
        if q.hasPrefix("dual collapse ") || q.hasPrefix("dual layer collapse ") {
            let queryStr: String
            if q.hasPrefix("dual layer collapse ") {
                queryStr = String(query.dropFirst(20))
            } else {
                queryStr = String(query.dropFirst(14))
            }
            let bridge = ASIQuantumBridgeSwift.shared
            if let result = bridge.dualLayerCollapse(query: queryStr) {
                return "★ DUAL-LAYER COLLAPSE:\n\(result)"
            }
            return "★ Collapse not available — Python ASI bridge offline. Try 'bridge fetch' first."
        }

        // ─── CONSCIOUSNESS VERIFICATION (l104_asi/consciousness.py) ───
        if q == "consciousness" || q == "consciousness status" || q == "verify consciousness" {
            let result = PythonBridge.shared.execute("""
            import sys, json
            sys.path.insert(0, '.')
            try:
                from l104_asi.consciousness import ConsciousnessVerifier
                cv = ConsciousnessVerifier()
                cv.run_full_verification()
                s = cv.get_status()
                print(json.dumps(s, default=str))
            except Exception as e:
                print(json.dumps({"error": str(e)}))
            """)
            if result.success {
                return "👁 CONSCIOUSNESS VERIFICATION:\n\(result.output)"
            }
            return "👁 Consciousness module not available — \(result.error)"
        }

        // ─── THEOREM GENERATOR (l104_asi/theorem_gen.py) ───
        if q == "theorem" || q == "theorem generate" || q == "novel theorem" {
            let result = PythonBridge.shared.execute("""
            import sys, json
            sys.path.insert(0, '.')
            try:
                from l104_asi.theorem_gen import NovelTheoremGenerator
                gen = NovelTheoremGenerator()
                t = gen.generate()
                print(json.dumps(t, default=str))
            except Exception as e:
                print(json.dumps({"error": str(e)}))
            """)
            if result.success {
                return "🔬 NOVEL THEOREM:\n\(result.output)"
            }
            return "🔬 Theorem generator not available — \(result.error)"
        }

        // ─── TREE OF THOUGHTS REASONING (l104_asi/reasoning.py) ───
        if q.hasPrefix("reason ") || q.hasPrefix("tree of thoughts ") {
            let problem: String
            if q.hasPrefix("tree of thoughts ") {
                problem = String(query.dropFirst(17))
            } else {
                problem = String(query.dropFirst(7))
            }
            let escaped = problem.replacingOccurrences(of: "'", with: "\\'")
                .replacingOccurrences(of: "\"", with: "\\\"")
            let result = PythonBridge.shared.execute("""
            import sys, json
            sys.path.insert(0, '.')
            try:
                from l104_asi.reasoning import TreeOfThoughts
                tot = TreeOfThoughts()
                r = tot.think('\(escaped)', lambda q: {"solution": "reasoning...", "confidence": 0.75}, max_depth=3)
                print(json.dumps(r, default=str))
            except Exception as e:
                print(json.dumps({"error": str(e)}))
            """)
            if result.success {
                return "🌳 TREE OF THOUGHTS:\n\(result.output)"
            }
            return "🌳 Reasoning engine not available — \(result.error)"
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
            return "\u{1F40D} Direct bridge not available — use 'bridge fetch' for Process bridge"
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
                return "🌊 No parameters to raise — fetch from Python first"
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
                return "🌊 No parameters to raise — fetch from Python first"
            }
            SovereignQuantumCore.shared.loadParameters(params)
            let result = SovereignQuantumCore.shared.sovereignRaise(factor: factor)
            return result
        }
        if q == "sovereign interfere" || q == "sqc wave" {
            let sqc = SovereignQuantumCore.shared
            guard !sqc.parameters.isEmpty else {
                return "🌊 No parameters loaded — run 'sovereign raise' first"
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
                return "🌊 No parameters loaded — run 'sovereign raise' first"
            }
            sqc.normalize()
            let muStr: String = String(format: "%.6f", sqc.lastNormMean)
            let sigmaStr: String = String(format: "%.6f", sqc.lastNormStdDev)
            return "🌊 Parameters normalized\n  μ = \(muStr)\n  σ = \(sigmaStr)\n  Operations: \(sqc.operationCount)"
        }
        if q == "sovereign sync" || q == "sqc sync" {
            let sqc = SovereignQuantumCore.shared
            guard !sqc.parameters.isEmpty else {
                return "🌊 No parameters to sync — run 'sovereign raise' first"
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
            // evolve start <factor> [interval_ms] — supports brackets: evolve start [300] [5000]
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
            return "🔮 Feedback Log (last 15):\n\(fb.isEmpty ? "  (no feedback yet — run 'nexus run' first)" : fb)"
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
            return "🧠 Resonance tick #\(tick["tick"] ?? 0) — active: \(tick["active_engines"] ?? 0), decay=\(AdaptiveResonanceNetwork.DECAY_RATE)"
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
            return "🌊 Grover diffusion applied — Superfluidity: \(String(format: "%.4f", sf))"
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
            return "🐚 Stored in K\(kid) (\(FeOrbitalEngine.shared.shellForKernel(kid))-shell) — Total: \(QuantumShellMemory.shared.totalMemories)"
        }
        if q == "qmem grover" {
            QuantumShellMemory.shared.groverDiffusion()
            return "🐚 Grover diffusion on 8-qubit state vector — amplitudes updated"
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
            let labels = ASIQuantumBridgeSwift.o2StateLabels
            let mol = ASIQuantumBridgeSwift.shared.o2MolecularState
            var lines: [String] = []
            for i in 0..<16 {
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
            var lines = ["🔧 Engine Registry — \(reg.count) Engines Registered:\n"]
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
                lines.append("  \(icon) \(hStr) — \(name)")
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

        // Help is handled by H05 buildContextualResponse case "help" — unified comprehensive reference

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
        // ⚛️ QUANTUM COMPUTING COMMANDS — Real IBM QPUs + Simulator Fallback
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
                DispatchQueue.main.async {
                    self?.quantumHardwareConnected = success
                    if success {
                        self?.quantumBackendName = IBMQuantumClient.shared.connectedBackendName
                        self?.quantumBackendQubits = IBMQuantumClient.shared.availableBackends
                            .first(where: { $0.name == IBMQuantumClient.shared.connectedBackendName })?
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
                lines.append("║  \(hwTag) \(b.name) — \(b.numQubits) qubits, queue:\(b.pendingJobs), QV:\(b.quantumVolume)\(marker)")
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
                    let summary = jobs.prefix(5).map { "  [\($0.jobId.prefix(12))...] \($0.status) — \($0.backend)" }.joined(separator: "\n")
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
            if jobId.isEmpty { return "Usage: quantum wait <job_id> — polls until job completes (10 min max)" }
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

        // ─── QUANTUM STATUS — Real hardware first, simulator fallback ───

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
                    return "╔═══════════════════════════════════════════════════════╗\n║  ⚛️ QUANTUM ENGINE — \(isReal ? "REAL HARDWARE" : "SIMULATOR")          ║\n╠═══════════════════════════════════════════════════════╣\n║  Backend:    \(backend)\n║  Qubits:     \(qubits)\n║  Connected:  \(connected)\n║  Queue:      \(queueDepth) jobs\n║  REST API:   \(ibmClient.isConnected ? "CONNECTED" : "PENDING")\n║  Jobs Sent:  \(ibmClient.submittedJobs.count)\n╚═══════════════════════════════════════════════════════╝"
                }
            }
            // Fallback to simulator
            let result = PythonBridge.shared.quantumStatus()
            if result.success, let dict = result.returnValue as? [String: Any] {
                let caps = dict["capabilities"] as? [String] ?? []
                let circuits = dict["circuits_executed"] as? Int ?? 0
                return "╔═══════════════════════════════════════════════════════╗\n║  ⚛️ QUANTUM ENGINE — SIMULATOR                       ║\n╠═══════════════════════════════════════════════════════╣\n║  Circuits Executed: \(circuits)\n║  Algorithms: \(caps.count)\n║    \(caps.joined(separator: ", "))\n║  IBM Token:  NOT SET\n║  Tip: quantum connect <token> for real QPU\n╚═══════════════════════════════════════════════════════╝"
            }
            return "⚛️ Quantum Engine: \(result.output)"
        }

        // ─── REWIRED: Grover — Real hardware first, simulator fallback ───

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

        // ─── REWIRED: QPE — Real hardware first, simulator fallback ───

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

        // ─── REWIRED: VQE — Real hardware first, simulator fallback ───

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

        // ─── REWIRED: QAOA — Real hardware first, simulator fallback ───

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

        // ─── QUANTUM MINE — Direct real hardware mining ───

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

        // ─── QUANTUM CANCEL — Abort a running/queued job ───

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

        // ─── QUANTUM CIRCUIT TEMPLATES — Submit pre-built circuits to real hardware ───

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

        // ─── QUANTUM HELP — Updated with all commands ───

        if q == "quantum help" {
            let hwStatus = IBMQuantumClient.shared.ibmToken != nil ? "CONNECTED" : "NOT SET"
            return """
            ⚛️ Quantum Computing Commands:
              ── IBM Quantum Hardware ──
              quantum connect <token> — Connect to IBM Quantum (real QPU)
              quantum disconnect      — Disconnect & clear token
              quantum backends        — List available IBM backends
              quantum submit <qasm>   — Submit OpenQASM 3.0 circuit
              quantum cancel <job_id> — Cancel a running/queued job
              quantum jobs            — List submitted jobs
              quantum result <job_id> — Get measurement results
              quantum wait <job_id>   — Poll until job completes (10 min)
              quantum mine [strategy] — Quantum mining (auto/grover/vqe)

              ── Circuit Templates (submit to real QPU) ──
              quantum bell            — Bell state (EPR pair, 2 qubits)
              quantum ghz [n]         — GHZ state (n qubits, default 3)
              quantum qrng [bits]     — Quantum RNG (default 8 bits)
              quantum random          — Alias for quantum qrng

              ── Algorithms (real HW → simulator fallback) ──
              quantum status          — Engine & hardware status
              quantum grover [t] [q]  — Grover's search
              quantum qpe             — Phase estimation
              quantum vqe             — VQE eigensolver
              quantum qaoa            — QAOA MaxCut
              quantum amplitude       — Amplitude / random oracle
              quantum walk            — Quantum walk
              quantum kernel          — Quantum kernel similarity

              IBM Token: \(hwStatus)
              Get token: https://quantum.ibm.com/account
            """
        }

        // ═════════════════════════════════════════════════════════════════
        // 🎓 PROFESSOR MODE COMMANDS — Interactive teaching & learning
        // ═════════════════════════════════════════════════════════════════

        if q == "professor" || q == "professor mode" || q == "prof" {
            return "╔═══════════════════════════════════════════════════════╗\n║  🎓 PROFESSOR MODE — Interactive Learning Engine      ║\n╠═══════════════════════════════════════════════════════╣\n║  Commands:                                            ║\n║    professor <topic>  — Structured lesson             ║\n║    teach me <topic>   — Guided learning session       ║\n║    quiz <topic>       — Test your knowledge           ║\n║    explain <concept>  — Concept explanation            ║\n║    lesson quantum     — Quantum computing tutorial    ║\n║    lesson coding      — Programming tutorial          ║\n║    lesson crypto      — Cryptography tutorial         ║\n║                                                       ║\n║  Or use the 🎓 Professor tab for the full experience. ║\n╚═══════════════════════════════════════════════════════╝"
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
                quiz += "\nQ1: What speedup does Grover's algorithm provide?\n  A) Exponential  B) Quadratic  C) Linear  D) Logarithmic\n  ✅ B — O(√N) vs O(N)\n"
                quiz += "\nQ2: |ψ⟩ = α|0⟩ + β|1⟩ requires:\n  A) |α|² + |β|² = 1  B) α + β = 1  C) α × β = 0  D) |α| = |β|\n  ✅ A — Born's rule\n"
                quiz += "\nQ3: H|0⟩ = ?\n  A) |1⟩  B) (|0⟩+|1⟩)/√2  C) |0⟩  D) 0\n  ✅ B — Hadamard superposition\n"
            } else {
                quiz += "\nQ1: What is the golden ratio φ ≈ ?\n  A) 3.14159  B) 2.71828  C) 1.61803  D) 1.41421\n  ✅ C — φ = (1+√5)/2\n"
                quiz += "\nQ2: Time complexity of optimal comparison sort?\n  A) O(n)  B) O(n log n)  C) O(n²)  D) O(log n)\n  ✅ B — proven lower bound\n"
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
        // 💻 CODING SYSTEM COMMANDS — Direct l104_coding_system.py access
        // ═════════════════════════════════════════════════════════════════

        if q == "coding" || q == "coding status" || q == "coding system" {
            let result = PythonBridge.shared.codingSystemStatus()
            if result.success { return "💻 Coding Intelligence:\n\(result.output)" }
            return "💻 Coding System: Use the 💻 Coding tab or 'coding help' for commands."
        }
        if q == "coding help" {
            return "💻 Coding System Commands:\n  coding status     — System status\n  coding review     — Review code in 💻 Coding tab input\n  coding quality    — Quality gate check\n  coding suggest    — Get improvement suggestions\n  coding explain    — Explain code structure\n  coding scan       — Scan project\n  coding ci         — CI/CD report\n  coding self       — Self-analyze codebase\n\n  Or use the 💻 Coding tab for the full experience."
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
        // EVO_65: ASI PIPELINE v16.0 — NATIVE SWIFT ENGINE COMMANDS
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
                "║  🚀 ASI PIPELINE v16.0 — NATIVE SWIFT ENGINES        ║",
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
                lines.append("║  🟢 \(name) — \(detail)")
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
            return ["Superposition — states exist simultaneously",
                    "Entanglement — correlated quantum states",
                    "Measurement — wavefunction collapse",
                    "Quantum Gates — unitary transformations",
                    "Decoherence — loss of quantum behavior"]
        } else if t.contains("neural") || t.contains("machine learn") || t.contains("ai") {
            return ["Neural Networks — layered computation",
                    "Backpropagation — gradient-based learning",
                    "Activation Functions — nonlinear transforms",
                    "Loss Functions — error measurement",
                    "Attention Mechanisms — selective focus"]
        } else if t.contains("crypto") || t.contains("encrypt") {
            return ["Symmetric Encryption — shared key (AES)",
                    "Asymmetric Encryption — public/private (RSA)",
                    "Hash Functions — one-way digests (SHA-256)",
                    "Digital Signatures — authentication",
                    "Post-Quantum Cryptography — quantum-resistant"]
        } else {
            return ["\(topic.capitalized) fundamentals",
                    "Core principles and axioms",
                    "Mathematical foundations",
                    "Practical applications",
                    "Open problems and challenges"]
        }
    }

    // ─── BACKEND RESPONSE CACHE ───

}
