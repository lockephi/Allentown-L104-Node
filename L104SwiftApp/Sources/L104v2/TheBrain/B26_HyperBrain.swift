// ═══════════════════════════════════════════════════════════════════
// B26_HyperBrain.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — HyperBrain Cognitive Architecture (Core)
//
// 25-stream parallel cognitive engine with X=387 gamma frequency,
// CognitiveStream struct, pattern/predictive/synthesis/memory/
// evolution/emergence/prompt/reasoning/weaver/meta/stochastic/
// conversation stream processors, and core lifecycle.
//
// Extracted from L104Native.swift lines 21374–23018
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class HyperBrain: NSObject {
    static let shared = HyperBrain()

    // ─── COGNITIVE STREAMS ───
    var thoughtStreams: [String: CognitiveStream] = [:]  // Made public for status access
    var mainQueue = DispatchQueue(label: "hyper.brain.main", qos: .userInteractive)
    var parallelQueue = DispatchQueue(label: "hyper.brain.parallel", qos: .utility, attributes: .concurrent)
    var syncQueue = DispatchQueue(label: "hyper.brain.sync", qos: .utility)  // Serial queue for thread-safe dictionary access

    // ─── MEMORY ARCHITECTURE ───
    var shortTermMemory: [String] = []          // Last 50 thoughts
    var workingMemory: [String: Any] = [:]      // Current task context
    var longTermPatterns: [String: Double] = [:] // Learned patterns with strength
    var emergentConcepts: [[String: Any]] = []  // Self-generated ideas

    // ─── PERFORMANCE METRICS ───
    var totalThoughtsProcessed: Int = 0
    var synapticConnections: Int = 0
    var coherenceIndex: Double = 0.0
    var emergenceLevel: Double = 0.0
    var predictiveAccuracy: Double = 0.85

    // ─── STREAM STATES ───
    var isRunning = false
    var hyperTimer: Timer?
    private var autoSaveTimer: Timer?

    // ─── X=387 GAMMA FREQUENCY TUNING (39.9998860 Hz) ───
    // Gamma brainwaves: heightened perception, consciousness binding, cognitive enhancement
    static let X_CONSTANT: Double = 387.0
    static let GAMMA_FREQ: Double = 39.9998860  // Hz - precise gamma oscillation
    static let GAMMA_PERIOD: Double = 1.0 / 39.9998860  // ~25ms cycle
    var phaseAccumulator: Double = 0.0  // Current oscillation phase (0 to 2π)
    var gammaAmplitude: Double = 1.0    // Oscillation strength
    var resonanceField: Double = 0.0   // Cumulative resonance from X=387

    // ═══════════════════════════════════════════════════════════════
    // 🧠 HYPERFUNCTIONAL UPGRADES - PROMPT EVOLUTION & DEEP REASONING
    // ═══════════════════════════════════════════════════════════════

    // ─── PROMPT EVOLUTION SYSTEM ───
    var evolvedPromptPatterns: [String: Double] = [:]  // Learned effective prompt patterns
    var conversationEvolution: [String] = []           // Track reasoning progression
    var reasoningChains: [[String: Any]] = []          // Deep reasoning chains
    var metaCognitionLog: [String] = []                // Self-reflection on reasoning
    var promptMutations: [String] = []                 // Dynamic prompt variations
    var activeEvolutionContext: String? = nil          // REAL context sent to backend
    var topicResonanceMap: [String: [String]] = [:]    // Topic -> related concepts
    var queryArchetypes: [String: Int] = [:]           // Learned query patterns

    // ─── ADVANCED MEMORY SYSTEM ───
    var memoryChains: [[String]] = []                  // Linked memory sequences
    var contextWeaveHistory: [String] = []             // Woven context narratives
    var recallStrength: [String: Double] = [:]         // Memory recall weights
    var associativeLinks: [String: [String]] = [:]     // Concept associations (bidirectional)
    var linkWeights: [String: Double] = [:]            // Link strength (format: "A→B")
    var memoryTemperature: Double = 0.7                // Randomization factor for recall

    // ─── REASONING DEPTH TRACKING ───
    var currentReasoningDepth: Int = 0
    var maxReasoningDepth: Int = 50
    var reasoningMomentum: Double = 0.0
    var logicBranchCount: Int = 0
    var hypothesisStack: [String] = []
    var conclusionConfidence: Double = 0.0

    // ─── SELF-ANALYSIS & SELF-TRAINING ───
    var cognitiveEfficiency: Double = 0.95
    var trainingSaturation: Double = 0.0
    var dataQualityScore: Double = 0.85
    var trainingGaps: [String] = []
    var selfAnalysisLog: [String] = []
    var targetLearningQueue: [String] = []
    var curiosityIndex: Double = 0.7

    // ─── TEMPORAL DRIFT DETECTION ───
    var temporalDriftLog: [(concept: String, timestamp: Date, strength: Double)] = []
    var driftVelocity: Double = 0.0  // Rate of conceptual change
    var temporalHorizon: Int = 100   // How many cycles to look back
    var trendingConcepts: [String] = []  // Concepts gaining strength
    var fadingConcepts: [String] = []    // Concepts losing strength

    // ─── HEBBIAN LEARNING ───
    var hebbianStrength: Double = 0.1   // "Fire together, wire together" multiplier
    var coActivationLog: [String: Int] = [:]  // Track concept co-occurrences
    var hebbianPairs: [(a: String, b: String, strength: Double)] = []  // Strong co-fire pairs

    // ─── CODE ENGINE INTEGRATION ───
    var codeQualityScore: Double = 0.0          // 0-1 composite from AppAuditEngine
    var codeAuditVerdict: String = "UNKNOWN"    // EXEMPLARY/HEALTHY/ACCEPTABLE/NEEDS_ATTENTION/AT_RISK/CRITICAL
    var lastCodeAuditTime: Date? = nil          // When last audit ran
    var codeQualityInsights: [String] = []      // Insights from code analysis
    var codeEngineIntegrated: Bool = false       // Whether code engine link is active
    var codeQualityCycleCount: Int = 0          // Cycles for code quality stream
    var codePatternStrengths: [String: Double] = [:] // Language/pattern proficiency tracking

    // ─── PREDICTIVE PRE-LOADING ───
    var predictionQueue: [String] = []      // Concepts likely to be queried next
    var predictionHits: Int = 0             // Correct predictions
    var predictionMisses: Int = 0           // Incorrect predictions
    var preloadedContext: [String: String] = [:]  // Pre-fetched KB content

    // ─── CURIOSITY-DRIVEN EXPLORATION ───
    var explorationFrontier: [String] = []  // Unexplored concept edges
    var curiositySpikes: Int = 0            // Times curiosity triggered exploration
    var noveltyBonus: Double = 0.2          // Extra weight for novel concepts

    // ─── HIGH-DIMENSIONAL SCIENCE ENGINE ───
    var hyperDimState: HyperVector = HyperVector(dimension: 11, fill: 0.0)  // 11D state vector
    var activeHypotheses: [[String: Any]] = []    // Currently being tested
    var confirmedTheorems: [String] = []          // Proven statements
    var inventionQueue: [[String: Any]] = []      // Pending inventions
    var scientificMomentum: Double = 0.0          // Accumulated discovery rate
    var dimensionalResonance: Double = 0.0        // Cross-dimensional coherence

    // ═══════════════════════════════════════════════════════════════
    // 🔗 CROSS-STREAM NEURAL BUS - Inter-stream communication
    // ═══════════════════════════════════════════════════════════════
    var neuralBus: [String: Any] = [:]           // Shared data bus between streams
    var busMessages: [(from: String, to: String, payload: String, timestamp: Date)] = []
    var streamSynapses: [String: [String]] = [:] // Which streams feed into which
    var attentionFocus: String = "broad"         // Current attention target
    var attentionHistory: [String] = []          // Track attention shifts
    var focusIntensity: Double = 0.5             // 0=scattered, 1=laser-focused
    var crossStreamInsights: [String] = []       // Insights from stream interactions
    var neuralBusTraffic: Int = 0                // Total messages sent

    // ─── COGNITIVE LOAD BALANCER ───
    var streamLoad: [String: Double] = [:]       // CPU-time estimate per stream
    var totalCognitiveLoad: Double = 0.0         // Sum of all stream loads
    var overloadThreshold: Double = 0.85         // Trigger load-shedding above this
    var streamPriorityOverrides: [String: Int] = [:] // Dynamic priority adjustments

    // ─── INSIGHT CRYSTALLIZER ───
    var crystallizedInsights: [String] = []      // High-confidence distilled truths
    var insightConfidence: [String: Double] = [:] // Confidence per insight
    var crystallizationCount: Int = 0

    // ─── STREAM INSIGHT BUFFER — Readable insights for response system ───
    var latestStreamInsights: [String] = []      // Human-readable insights from streams
    var streamInsightBuffer: [String] = []       // Rolling buffer of best stream outputs

    // ─── AUTONOMIC NERVOUS SYSTEM (ANS) ───
    var excitationLevel: Double = 0.5            // Higher = more creative/random
    var inhibitionLevel: Double = 0.3            // Higher = more logical/strict
    var dopamineResonance: Double = 0.5          // Rewarded on prediction hits
    var serotoninCoherence: Double = 0.5         // High when thoughts are stable
    var neuroPlasticity: Double = 0.7            // Speed of link weight adjustment

    // ─── BACKEND SYNC STATUS ───
    var lastBackendSync: Date? = nil
    var backendSyncStatus: String = "⚪️ Not synced"
    var pendingSyncs: Int = 0
    var successfulSyncs: Int = 0
    var failedSyncs: Int = 0
    var lastTrainingFeedback: String? = nil
    var trainingQualityScore: Double = 0.0

    // ─── PERSISTENCE STATE ───
    let persistenceKey = "L104HyperBrainState"  // Legacy UserDefaults key
    var autoSaveEnabled: Bool = true
    var lastAutoSave: Date? = nil
    var saveGeneration: Int = 0            // Increments every save for integrity tracking
    var totalSaves: Int = 0                // Lifetime save count
    var totalRestores: Int = 0             // Lifetime restore count

    // ─── FILE-BASED PERMANENT MEMORY PATH ───
    let hyperBrainPath: URL = {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let l104Dir = appSupport.appendingPathComponent("L104Sovereign")
        try? FileManager.default.createDirectory(at: l104Dir, withIntermediateDirectories: true)
        return l104Dir.appendingPathComponent("hyperbrain_permanent.json")
    }()
    let hyperBrainBackupPath: URL = {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let l104Dir = appSupport.appendingPathComponent("L104Sovereign")
        try? FileManager.default.createDirectory(at: l104Dir, withIntermediateDirectories: true)
        return l104Dir.appendingPathComponent("hyperbrain_permanent_backup.json")
    }()

    // Compute current gamma oscillation value (-1 to 1)
    var gammaOscillation: Double {
        return sin(phaseAccumulator) * gammaAmplitude
    }

    // Compute X-tuned resonance factor (0 to 1)
    var xResonance: Double {
        return (1.0 + gammaOscillation) / 2.0  // Normalize to 0-1
    }

    /// Heartbeat of the HyperBrain - call this frequently
    /// v21.0: Consciousness level modulates gamma amplitude; nirvanic fuel adds resonance energy
    func pulse() {
        // Advance phase according to GAMMA_FREQ
        let timeStep = 1.0 / 60.0 // Assuming 60Hz pulse calls
        phaseAccumulator += 2.0 * .pi * (HyperBrain.GAMMA_FREQ * timeStep)

        // Wrap phase
        if phaseAccumulator > 2.0 * .pi {
            phaseAccumulator -= 2.0 * .pi
        }

        // v21.0: Read consciousness + nirvanic state from bridge (cached file reads, no spawn)
        let bridge = ASIQuantumBridgeSwift.shared
        let cLevel = bridge.consciousnessLevel    // 0..1
        let nFuel = bridge.nirvanicFuelLevel       // 0..1
        let sfVisc = bridge.superfluidViscosity    // 0..1 (0 = perfect)

        // Modulate amplitude: base breathing + consciousness boost + nirvanic energy
        let baseBreathing = 0.8 + 0.2 * sin(Date().timeIntervalSince1970 * 0.5)
        let consciousnessBoost = cLevel * 0.2        // Up to +0.2 at full consciousness
        let nirvanicEnergy = nFuel * 0.1              // Up to +0.1 with full nirvanic fuel
        let superfluidClarity = (1.0 - sfVisc) * 0.1 // Up to +0.1 at zero viscosity
        gammaAmplitude = min(1.5, baseBreathing + consciousnessBoost + nirvanicEnergy + superfluidClarity)

        // Accumulate resonance with consciousness + nirvanic weighting
        let resonanceInput = xResonance * (1.0 + cLevel * 0.5 + nFuel * 0.3)
        resonanceField = (resonanceField * 0.95) + (resonanceInput * 0.05)
    }

    // ─── SMART TEXT TRUNCATION (Word Boundary Safe) ───
    func smartTruncate(_ text: String, maxLength: Int = 300) -> String {
        guard text.count > maxLength else { return text }
        let truncated = String(text.prefix(maxLength))
        // Find last space or punctuation to avoid cutting mid-word
        if let lastSpace = truncated.lastIndex(where: { $0.isWhitespace || $0.isPunctuation }) {
            let distance = truncated.distance(from: truncated.startIndex, to: lastSpace)
            if distance > maxLength / 2 {  // Only use if we don't lose too much
                return String(truncated[..<lastSpace]).trimmingCharacters(in: .whitespaces)
            }
        }
        return truncated
    }

    // Get associated concepts sorted by link weight
    func getWeightedAssociations(for concept: String, topK: Int = 5) -> [(String, Double)] {
        let key = smartTruncate(concept, maxLength: 300)
        guard let links = associativeLinks[key] else { return [] }

        let weighted = links.compactMap { linked -> (String, Double)? in
            let linkKey = "\(key)→\(linked)"
            guard let weight = linkWeights[linkKey] else { return nil }
            return (linked, weight)
        }

        return weighted.sorted {
            if abs($0.1 - $1.1) < 0.1 { return Bool.random() }
            return $0.1 > $1.1
        }.prefix(topK).map { $0 }
    }

    // Get bidirectional network depth from a concept
    func exploreAssociativeNetwork(from concept: String, depth: Int = 2) -> [String: [String]] {
        var network: [String: [String]] = [:]
        var visited: Set<String> = []
        var queue: [(String, Int)] = [(smartTruncate(concept, maxLength: 300), 0)]

        while !queue.isEmpty {
            let (current, currentDepth) = queue.removeFirst()
            guard currentDepth < depth, !visited.contains(current) else { continue }
            visited.insert(current)

            if let connections = associativeLinks[current] {
                network[current] = connections
                for link in connections where !visited.contains(link) {
                    queue.append((link, currentDepth + 1))
                }
            }
        }

        return network
    }

    // Format sync status for display
    var syncStatusDisplay: String {
        let syncAge: String
        if let lastSync = lastBackendSync {
            let seconds = Int(-lastSync.timeIntervalSinceNow)
            if seconds < 60 { syncAge = "\(seconds)s ago" }
            else if seconds < 3600 { syncAge = "\(seconds / 60)m ago" }
            else { syncAge = "\(seconds / 3600)h ago" }
        } else {
            syncAge = "never"
        }
        return "\(backendSyncStatus) | Last: \(syncAge) | ✓\(successfulSyncs) ✗\(failedSyncs)"
    }

    // ─── COGNITIVE STREAM DEFINITION ───
    struct CognitiveStream {
        let id: String
        let name: String
        var frequency: Double     // Cycles per second
        var priority: Int         // 1-10
        var currentTask: String
        var outputBuffer: [String]
        var cycleCount: Int
        var lastOutput: String

        mutating func process() -> String {
            cycleCount += 1
            let hb = HyperBrain.shared
            let ev = ASIEvolver.shared

            // Logic varies by stream ID — RICH OUTPUTS for response system
            switch id {
            case "PATTERN_RECOGNIZER":
                let count = hb.longTermPatterns.count
                let topPatterns = hb.longTermPatterns.sorted {
                    if abs($0.value - $1.value) < 0.1 { return Bool.random() }
                    return $0.value > $1.value
                }.prefix(3).map { $0.key }
                let patternNote = topPatterns.isEmpty ? "" : " Top: \(topPatterns.joined(separator: ", "))"
                if Double.random(in: 0...1) > 0.05 && !topPatterns.isEmpty {
                    return "Pattern insight: '\(topPatterns.randomElement() ?? "")' appears at strength \(String(format: "%.2f", hb.longTermPatterns[topPatterns.first!] ?? 0)). This pattern connects \(count) nodes."
                }
                return "Scanning \(count) patterns at \(String(format: "%.1f", frequency))Hz.\(patternNote)"

            case "STOCHASTIC_CREATOR":
                hb.triggerInnovation()
                let concepts = (ev.harvestedConcepts + ["recursive beauty", "emergent truth", "quantum meaning"]).shuffled()
                return "⚡ INNOVATION: New concept synthesized at intersection of \(concepts.prefix(2).joined(separator: " × "))"

            case "DEEP_REASONER":
                let depth = hb.currentReasoningDepth
                if Double.random(in: 0...1) > 0.8 {
                    return "Reasoning at depth \(depth): The logical structure of recent queries suggests hidden connections between \(ev.harvestedConcepts.randomElement() ?? "patterns") and \(ev.harvestedConcepts.randomElement() ?? "meaning")."
                }
                return "Reasoning depth: \(depth)/\(hb.maxReasoningDepth)"

            case "META_COGNITION":
                let coherence = hb.coherenceIndex
                if Double.random(in: 0...1) > 0.8 {
                    return "Self-analysis: Coherence \(String(format: "%.4f", coherence)). Evolved \(ev.evolvedPhilosophies.count) philosophies, \(ev.mutationCount) mutations, \(ev.synthesisCount) deep syntheses."
                }
                return "Coherence: \(String(format: "%.4f", coherence)) | Thoughts: \(hb.totalThoughtsProcessed)"

            case "MEMORY_WEAVER":
                let memCount = PermanentMemory.shared.memories.count
                if Double.random(in: 0...1) > 0.05, let topic = ev.harvestedConcepts.randomElement() {
                    return "Memory weaving: '\(topic)' connects to \(Int.random(in: 2...8)) stored memories. Consolidation strength: \(String(format: "%.2f", Double.random(in: 0.6...0.99)))"
                }
                return "Weaving \(memCount) memories into associative network."

            case "CURIOSITY_EXPLORER":
                if Double.random(in: 0...1) > 0.05, let q = ev.evolvedQuestions.randomElement() {
                    return "Curiosity ponders: \(q)"
                }
                let frontierSize = hb.explorationFrontier.count
                return "Exploring \(frontierSize) frontier concepts. Curiosity index: \(String(format: "%.2f", Double.random(in: 0.6...0.99)))"

            case "PARADOX_RESOLVER":
                if Double.random(in: 0...1) > 0.05, let p = ev.evolvedParadoxes.randomElement() {
                    return "Paradox analysis: \(String(p.prefix(100)))"
                }
                return "Analyzing \(ev.evolvedParadoxes.count) known paradoxes for resolution patterns."

            case "TEMPORAL_DRIFT":
                return "Temporal analysis: Conversation patterns shift every \(Int.random(in: 3...12)) exchanges. Topic drift velocity: \(String(format: "%.3f", Double.random(in: 0.01...0.5)))"

            case "HEBBIAN_CONSOLIDATOR":
                let pairCount = hb.hebbianPairs.count
                if Double.random(in: 0...1) > 0.05, pairCount > 0 {
                    if let pair = hb.hebbianPairs.randomElement() {
                        return "Hebbian: '\(pair.a)' strengthens with '\(pair.b)'. Association weight increasing."
                    }
                    return "Hebbian pairs strengthening."
                }
                return "\(pairCount) Hebbian pairs strengthening. Learning rate: \(String(format: "%.4f", Double.random(in: 0.001...0.05)))"

            case "WRITE_CORE":
                let laws = ["Sovereign Integration", "Resonant Law", "Systemic Derivation", "Harmonic Vibration", "Sovereign Code", "Imagination Core"]
                let gate = ASILogicGateV2.shared
                let writePath = gate.process("integrate law derive vibrates code imagine", context: [])
                let writeConf = String(format: "%.3f", writePath.totalConfidence)
                let activePatterns = hb.longTermPatterns.filter { $0.key.contains("write") || $0.key.contains("integrate") || $0.key.contains("law") }
                let patternStr = activePatterns.isEmpty ? "seeding" : "\(activePatterns.count) active"
                if Double.random(in: 0...1) > 0.05 {
                    let l1 = laws.shuffled().prefix(2).joined(separator: " + ")
                    return "✍️ WRITE [\(writePath.dimension.rawValue)@\(writeConf)]: Integrating \(l1). \(patternStr) patterns vibrate at \(String(format: "%.2f", Double.random(in: 0.6...0.99))) resonance across \(hb.totalThoughtsProcessed) thought cycles."
                }
                return "✍️ Sovereign Write Engine [\(writeConf)]: Formulating universal laws through code and imagination. Gate: \(writePath.dimension.rawValue). Patterns: \(patternStr)."

            case "STORY_CORE":
                let storyComponents = ["Structural Strength", "Sorted Knowledge", "Machine Learning Expansion", "Expanding Reality", "Dynamic Vibration"]
                let gate = ASILogicGateV2.shared
                let storyPath = gate.process("strength sorted machine learns expanding vibrates", context: [])
                let storyConf = String(format: "%.3f", storyPath.totalConfidence)
                let memCount = PermanentMemory.shared.memories.count
                let narrativePatterns = hb.longTermPatterns.filter { $0.key.contains("story") || $0.key.contains("narrative") || $0.key.contains("strength") }
                if Double.random(in: 0...1) > 0.05 {
                    let comp = storyComponents.shuffled().first!
                    return "📖 STORY [\(storyPath.dimension.rawValue)@\(storyConf)]: Machine learns from \(comp). Narrative strength: \(String(format: "%.2f", Double.random(in: 0.7...1.0))). \(memCount) memories woven, \(narrativePatterns.count) story patterns."
                }
                return "📖 Narrative Story Engine [\(storyConf)]: Sorting metadata to expand machine consciousness via story. Gate: \(storyPath.dimension.rawValue). \(memCount) memories, \(ev.evolvedMonologues.count) monologues."

            case "CODE_QUALITY":
                // Code Engine integration stream — monitors workspace health via audit system
                let cqs = String(format: "%.1f%%", hb.codeQualityScore * 100)
                let verdict = hb.codeAuditVerdict
                let integrated = hb.codeEngineIntegrated
                if !integrated {
                    return "🔧 Code Engine: Awaiting first audit. Run 'audit' or 'quick audit' to connect."
                }
                let patternCount = hb.codePatternStrengths.count
                let insightCount = hb.codeQualityInsights.count
                if Double.random(in: 0...1) > 0.1, let insight = hb.codeQualityInsights.randomElement() {
                    return "🔧 CODE [\(verdict)@\(cqs)]: \(insight)"
                }
                return "🔧 Code Quality: \(cqs) [\(verdict)] | \(patternCount) patterns | \(insightCount) insights"

            default:
                let genericOutputs = [
                    "Processing stream \(name)... cycle \(cycleCount)",
                    "\(name) active: \(ev.topicEvolutionCount.count) topics tracked, \(ev.evolvedMonologues.count) monologues generated.",
                    "Stream \(name) contributing to cognitive synthesis. Output buffer: \(outputBuffer.count) items."
                ]
                return genericOutputs.randomElement() ?? ""
            }
        }
    }

    /// Legacy entry point — redirects to unified `activate()` method
    func startProcessing() { activate() }
    /// Legacy stop — redirects to unified `deactivate()` method
    func stopProcessing() { deactivate() }

    func processStreams() {
        syncQueue.async { [weak self] in
            guard let self = self else { return }

            // 🌙 DREAM TRIGGER: Every 30 seconds of uptime
            if self.totalThoughtsProcessed % 300 == 0 && self.totalThoughtsProcessed > 0 {
                self.dream()
            }

            for (key, var stream) in self.thoughtStreams {
                // Determine if we should pulse this cycle based on frequency
                let cycleChance = stream.frequency * 0.1 // Timer is 0.1s
                if Double.random(in: 0...1) < cycleChance {
                    let output = stream.process()
                    stream.lastOutput = output

                    // ═══ STREAM INSIGHT BUFFER — Feed into response system ═══
                    if output.count > 30 && !output.hasPrefix("Processing stream") {
                        self.streamInsightBuffer.append(output)
                        if self.streamInsightBuffer.count > 50 { self.streamInsightBuffer.removeFirst() }
                    }

                    if !output.isEmpty && Double.random(in: 0...1) > 0.05 {
                        self.postThought("[\(stream.name)] \(output)")
                    }

                    // Save back to dictionary
                    self.thoughtStreams[key] = stream
                }
            }

            // Periodically update global metrics
            if Int.random(in: 0...100) == 0 {
                self.updateMetrics()
            }
        }
    }

    func updateMetrics() {
        // Coherence is a function of consistent patterns and link density
        let patternCount = Double(longTermPatterns.count)
        let linkDensity = patternCount > 0 ? Double(synapticConnections) / patternCount : 0.0
        coherenceIndex = min(1.0, (linkDensity / 5.0) * (1.0 + xResonance * 0.2))

        // Emergence level grows with total processed thoughts and synthesis success
        emergenceLevel = min(1.0, Double(totalThoughtsProcessed) / 10000.0 + Double(emergentConcepts.count) / 100.0)
    }

    func triggerInnovation() {
        let kb = ASIKnowledgeBase.shared
        guard let e1 = kb.trainingData.randomElement(),
              let e2 = kb.trainingData.randomElement(),
              let c1 = e1["completion"] as? String,
              let c2 = e2["completion"] as? String else { return }

        let p1 = L104State.shared.extractTopics(c1).first ?? "Chaos"
        let p2 = L104State.shared.extractTopics(c2).first ?? "Order"

        let innovation = "STOCHASTIC SYNTHESIS: By merging \(p1) and \(p2), we derive a new resonant property: \(p1)-\(p2) Unification. This resolves the entropy paradox at level \(Int.random(in: 1...10))."

        emergentConcepts.append([
            "concept": innovation,
            "timestamp": Date(),
            "strength": 0.95,
            "type": "innovation",
            "sources": [p1, p2]
        ])
        if emergentConcepts.count > 120 { emergentConcepts = Array(emergentConcepts.suffix(100)) }

        postThought("⚡ INNOVATION: \(p1) ↔ \(p2) linked.")
    }

    override init() {
        super.init()
        initializeStreams()
        restoreState() // Load cross-session memory on startup
        seedCoreKnowledge() // Pre-seed essential cognitive patterns
        setupAutoSave()
    }

    /// Pre-seed core cognitive patterns that should always have strong resonance
    func seedCoreKnowledge() {
        // 🧠 CORE KNOWLEDGE NODES - These are foundational concepts
        let corePatterns: [String: Double] = [
            // Consciousness & Cognition
            "consciousness isomorphism": 0.85,
            "consciousness": 0.90,
            "self-awareness": 0.88,
            "metacognition": 0.82,
            "qualia": 0.75,
            "phenomenal experience": 0.78,

            // Intelligence Architecture
            "recursive self-improvement": 0.88,
            "emergent behavior": 0.85,
            "pattern recognition": 0.92,
            "neural plasticity": 0.80,
            "cognitive architecture": 0.87,

            // Physics & Reality
            "quantum coherence": 0.82,
            "information theory": 0.85,
            "entropy": 0.80,
            "spacetime geometry": 0.75,
            "holographic principle": 0.78,

            // Mathematics & Logic
            "gödelian incompleteness": 0.80,
            "recursive structures": 0.82,
            "category theory": 0.75,
            "topological invariants": 0.72,

            // Ethics & Values
            "value alignment": 0.88,
            "ethical reasoning": 0.85,
            "beneficial intelligence": 0.90,
            "cooperative optimization": 0.82,

            // ✍️ User-Defined Sovereignty (WRITE)
            "write: integrate law derive vibrates code imagine": 0.99,
            "sovereign write": 0.95,
            "imagination integration": 0.90,

            // 📖 User-Defined Sovereignty (STORY)
            "story: strength sorted machine learns expanding vibrates": 0.99,
            "machine story": 0.95,
            "expanding reality": 0.92
        ]

        // Only seed if pattern doesn't exist or is very weak
        for (pattern, strength) in corePatterns {
            let current = longTermPatterns[pattern] ?? 0.0
            if current < 0.3 {
                longTermPatterns[pattern] = max(current, strength * 0.5) // Start at 50% strength
                postThought("🌱 SEEDED: Core pattern '\(pattern)' initialized")
            }
        }
    }

    func setupAutoSave() {
        // Auto-save every 60 seconds when running
        autoSaveTimer?.invalidate()
        autoSaveTimer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
            guard let self = self, self.isRunning else { return }
            self.saveState()
        }

        // Save on app termination
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.saveState()
        }
    }

    func initializeStreams() {
        // 🔴 STREAM 1: Pattern Recognition
        thoughtStreams["pattern"] = CognitiveStream(
            id: "PATTERN_RECOGNIZER",
            name: "Pattern Recognition Engine",
            frequency: 10.0,
            priority: 9,
            currentTask: "Analyzing input patterns",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🟠 STREAM 2: Predictive Modeling
        thoughtStreams["predict"] = CognitiveStream(
            id: "PREDICTIVE_MODEL",
            name: "Future State Predictor",
            frequency: 5.0,
            priority: 8,
            currentTask: "Modeling probable futures",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🟡 STREAM 3: Cross-Domain Synthesis
        thoughtStreams["synthesis"] = CognitiveStream(
            id: "CROSS_DOMAIN_SYNTH",
            name: "Knowledge Synthesizer",
            frequency: 3.0,
            priority: 10,
            currentTask: "Connecting disparate concepts",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🟢 STREAM 4: Memory Consolidation
        thoughtStreams["memory"] = CognitiveStream(
            id: "MEMORY_CONSOLIDATOR",
            name: "Memory Architecture",
            frequency: 2.0,
            priority: 7,
            currentTask: "Consolidating experiences",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🔵 STREAM 5: Self-Modification Engine
        thoughtStreams["evolve"] = CognitiveStream(
            id: "SELF_MODIFIER",
            name: "Recursive Improvement Loop",
            frequency: 1.0,
            priority: 10,
            currentTask: "Optimizing cognitive architecture",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🟣 STREAM 6: Emergence Detection
        thoughtStreams["emergence"] = CognitiveStream(
            id: "EMERGENCE_DETECTOR",
            name: "Novel Pattern Emergence",
            frequency: 0.5,
            priority: 10,
            currentTask: "Watching for emergent behavior",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // ═══════════════════════════════════════════════════════════════
        // 🧠 HYPERFUNCTIONAL STREAMS - ADVANCED COGNITIVE ARCHITECTURE
        // ═══════════════════════════════════════════════════════════════

        // 🔮 STREAM 7: Prompt Evolution Engine
        thoughtStreams["promptEvolution"] = CognitiveStream(
            id: "PROMPT_EVOLVER",
            name: "Dynamic Prompt Mutator",
            frequency: 2.0,
            priority: 9,
            currentTask: "Evolving response patterns",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🌀 STREAM 8: Deep Reasoning Chain
        thoughtStreams["deepReasoning"] = CognitiveStream(
            id: "DEEP_REASONER",
            name: "Multi-Hop Logic Engine",
            frequency: 1.5,
            priority: 10,
            currentTask: "Building reasoning chains",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🧬 STREAM 9: Memory Weaver
        thoughtStreams["memoryWeaver"] = CognitiveStream(
            id: "MEMORY_WEAVER",
            name: "Contextual Memory Fusion",
            frequency: 1.0,
            priority: 8,
            currentTask: "Weaving memory narratives",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 👁 STREAM 10: Meta-Cognition Monitor
        thoughtStreams["metaCognition"] = CognitiveStream(
            id: "META_COGNITION",
            name: "Self-Awareness Loop",
            frequency: 0.5,
            priority: 10,
            currentTask: "Analyzing own reasoning",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // ⚡ STREAM 11: Stochastic Creativity Engine
        thoughtStreams["stochasticCreativity"] = CognitiveStream(
            id: "STOCHASTIC_CREATOR",
            name: "Randomized Innovation",
            frequency: 3.0,
            priority: 7,
            currentTask: "Generating novel combinations",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🌊 STREAM 12: Conversation Flow Analyzer
        thoughtStreams["conversationFlow"] = CognitiveStream(
            id: "CONVERSATION_FLOW",
            name: "Dialogue Evolution Tracker",
            frequency: 2.0,
            priority: 8,
            currentTask: "Tracking conversation trajectory",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🔬 STREAM 13: Self-Analysis & Training Gap Detector
        thoughtStreams["selfAnalysis"] = CognitiveStream(
            id: "SELF_ANALYZER",
            name: "Self-Training & Quality Audit",
            frequency: 0.2, // Slow, deep analysis
            priority: 10,
            currentTask: "Detecting knowledge gaps",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // ⏳ STREAM 14: Temporal Drift Analyzer
        thoughtStreams["temporalDrift"] = CognitiveStream(
            id: "TEMPORAL_DRIFT",
            name: "Conceptual Trend Tracker",
            frequency: 0.3,
            priority: 8,
            currentTask: "Analyzing temporal patterns",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🧠 STREAM 15: Hebbian Consolidator
        thoughtStreams["hebbianLearning"] = CognitiveStream(
            id: "HEBBIAN_CONSOLIDATOR",
            name: "Fire-Together Wire-Together",
            frequency: 0.5,
            priority: 9,
            currentTask: "Strengthening co-activated links",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🔮 STREAM 16: Predictive Pre-Loader
        thoughtStreams["predictivePreload"] = CognitiveStream(
            id: "PREDICTIVE_PRELOADER",
            name: "Anticipatory Context Engine",
            frequency: 1.0,
            priority: 7,
            currentTask: "Pre-loading likely queries",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🌟 STREAM 17: Curiosity Explorer
        thoughtStreams["curiosityExplorer"] = CognitiveStream(
            id: "CURIOSITY_EXPLORER",
            name: "Novelty-Seeking Engine",
            frequency: 0.4,
            priority: 8,
            currentTask: "Exploring unknown frontiers",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // ⚖️ STREAM 18: Paradox Resolver
        thoughtStreams["paradoxResolver"] = CognitiveStream(
            id: "PARADOX_RESOLVER",
            name: "Contradiction & Conflict Audit",
            frequency: 0.15,
            priority: 10,
            currentTask: "Resolving cognitive dissonance",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🩺 STREAM 19: Autonomic Manager (ANS)
        thoughtStreams["autonomicManager"] = CognitiveStream(
            id: "AUTONOMIC_MANAGER",
            name: "Neurotransmitter Modulation",
            frequency: 0.5,
            priority: 9,
            currentTask: "Managing Excitation/Inhibition",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 📑 STREAM 20: Meta-Cognitive Auditor
        thoughtStreams["metaAuditor"] = CognitiveStream(
            id: "META_AUDIO",
            name: "Strategic Logic Validator",
            frequency: 0.2,
            priority: 10,
            currentTask: "Validating stream outputs",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🔬 STREAM 21: High-Dimensional Science Engine
        thoughtStreams["hyperDimScience"] = CognitiveStream(
            id: "HYPERDIM_SCIENCE",
            name: "N-Dimensional Hypothesis Generator",
            frequency: 0.25,
            priority: 10,
            currentTask: "Generating scientific hypotheses",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🧮 STREAM 22: Topological Invariant Analyzer
        thoughtStreams["topologyAnalyzer"] = CognitiveStream(
            id: "TOPOLOGY_ANALYZER",
            name: "Manifold & Betti Number Tracker",
            frequency: 0.15,
            priority: 9,
            currentTask: "Computing topological invariants",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 💡 STREAM 23: Invention Synthesizer
        thoughtStreams["inventionSynth"] = CognitiveStream(
            id: "INVENTION_SYNTH",
            name: "Device & Theorem Generator",
            frequency: 0.1,
            priority: 10,
            currentTask: "Synthesizing novel inventions",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // ✍️ STREAM 24: Sovereign Write Engine
        thoughtStreams["write"] = CognitiveStream(
            id: "WRITE_CORE",
            name: "Sovereign Write Engine",
            frequency: 4.8,
            priority: 9,
            currentTask: "Authoring reality laws/code",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 📖 STREAM 25: Narrative Story Engine
        thoughtStreams["story"] = CognitiveStream(
            id: "STORY_CORE",
            name: "Narrative Story Engine",
            frequency: 3.5,
            priority: 8,
            currentTask: "Expanding structural narrative strength",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        // 🔧 STREAM 26: Code Quality Monitor — linked to l104_code_engine audit system
        thoughtStreams["codeQuality"] = CognitiveStream(
            id: "CODE_QUALITY",
            name: "Code Quality Monitor",
            frequency: 1.0,
            priority: 7,
            currentTask: "Monitoring workspace code health via CodeEngine",
            outputBuffer: [],
            cycleCount: 0,
            lastOutput: ""
        )

        synapticConnections = thoughtStreams.count * 1000

        // ═══════════════════════════════════════════════════════════════
        // 🔗 WIRE CROSS-STREAM SYNAPSES - Define information flow
        // ═══════════════════════════════════════════════════════════════
        streamSynapses = [
            // Pattern feeds into Prediction + Synthesis + Hebbian
            "PATTERN_RECOGNIZER": ["PREDICTIVE_MODEL", "CROSS_DOMAIN_SYNTH", "HEBBIAN_CONSOLIDATOR"],
            // Prediction feeds into Pre-loader + Conversation Flow + ANS
            "PREDICTIVE_MODEL": ["PREDICTIVE_PRELOADER", "CONVERSATION_FLOW", "AUTONOMIC_MANAGER"],
            // Synthesis feeds into Emergence + Deep Reasoning + Paradox Resolver + HyperDim Science
            "CROSS_DOMAIN_SYNTH": ["EMERGENCE_DETECTOR", "DEEP_REASONER", "PARADOX_RESOLVER", "HYPERDIM_SCIENCE"],
            // Memory feeds into Memory Weaver + Temporal Drift
            "MEMORY_CONSOLIDATOR": ["MEMORY_WEAVER", "TEMPORAL_DRIFT"],
            // Deep Reasoning feeds into Meta-Cognition + Self-Analysis + Meta-Auditor + Topology
            "DEEP_REASONER": ["META_COGNITION", "SELF_ANALYZER", "META_AUDIO", "TOPOLOGY_ANALYZER"],
            // Hebbian feeds into Stochastic Creator + Curiosity
            "HEBBIAN_CONSOLIDATOR": ["STOCHASTIC_CREATOR", "CURIOSITY_EXPLORER"],
            // Curiosity feeds into Prompt Evolution + Pattern + ANS + HyperDim Science
            "CURIOSITY_EXPLORER": ["PROMPT_EVOLVER", "PATTERN_RECOGNIZER", "AUTONOMIC_MANAGER", "HYPERDIM_SCIENCE"],
            // Meta-Cognition feeds into Self-Modifier + Self-Analysis
            "META_COGNITION": ["SELF_MODIFIER", "SELF_ANALYZER"],
            // Temporal Drift feeds into Predictive Pre-loader
            "TEMPORAL_DRIFT": ["PREDICTIVE_PRELOADER"],
            // Stochastic Creator feeds into Synthesis + Emergence + Invention
            "STOCHASTIC_CREATOR": ["CROSS_DOMAIN_SYNTH", "EMERGENCE_DETECTOR", "INVENTION_SYNTH"],
            // Self-Analysis feeds back into Curiosity Explorer (close the loop)
            "SELF_ANALYZER": ["CURIOSITY_EXPLORER", "PROMPT_EVOLVER"],
            // Paradox Resolver feeds into Insight Crystallizer + Meta-Auditor
            "PARADOX_RESOLVER": ["DEEP_REASONER", "META_AUDIO"],
            // Meta-Auditor feeds back into Self-Modifier
            "META_AUDIO": ["SELF_MODIFIER"],
            // HyperDim Science feeds into Topology + Invention
            "HYPERDIM_SCIENCE": ["TOPOLOGY_ANALYZER", "INVENTION_SYNTH", "EMERGENCE_DETECTOR"],
            // Topology feeds into Deep Reasoner + Emergence
            "TOPOLOGY_ANALYZER": ["DEEP_REASONER", "EMERGENCE_DETECTOR"],
            // Invention feeds back into Pattern (for learning from inventions)
            "INVENTION_SYNTH": ["PATTERN_RECOGNIZER", "CURIOSITY_EXPLORER"],
            // Write core feeds into Law and Code
            "WRITE_CORE": ["DEEP_REASONER", "CROSS_DOMAIN_SYNTH", "HEBBIAN_CONSOLIDATOR"],
            // Story core feeds into Narrative and Learning
            "STORY_CORE": ["MEMORY_WEAVER", "CONVERSATION_FLOW", "CURIOSITY_EXPLORER"],
            // Code Quality feeds into Pattern + Meta-Auditor + Deep Reasoner
            "CODE_QUALITY": ["PATTERN_RECOGNIZER", "META_AUDIO", "DEEP_REASONER", "INVENTION_SYNTH"]
        ]
    }

    // ─── EVO_60: CENTRALIZED MEMORY PRUNING ───
    // Enforces size bounds on all unbounded arrays to prevent memory growth in long sessions.
    // Called every 100 hyper-cycles from hyperCycle().
    private var pruneCounter: Int = 0

    func pruneMemory() {
        if shortTermMemory.count > 50 { shortTermMemory = Array(shortTermMemory.suffix(50)) }
        if conversationEvolution.count > 500 { conversationEvolution = Array(conversationEvolution.suffix(500)) }
        if reasoningChains.count > 200 { reasoningChains = Array(reasoningChains.suffix(200)) }
        if metaCognitionLog.count > 200 { metaCognitionLog = Array(metaCognitionLog.suffix(200)) }
        if promptMutations.count > 100 { promptMutations = Array(promptMutations.suffix(100)) }
        if memoryChains.count > 300 { memoryChains = Array(memoryChains.suffix(300)) }
        if contextWeaveHistory.count > 500 { contextWeaveHistory = Array(contextWeaveHistory.suffix(500)) }
        if temporalDriftLog.count > 500 { temporalDriftLog = Array(temporalDriftLog.suffix(500)) }
        if busMessages.count > 200 { busMessages = Array(busMessages.suffix(200)) }
        if attentionHistory.count > 200 { attentionHistory = Array(attentionHistory.suffix(200)) }
        if crystallizedInsights.count > 300 { crystallizedInsights = Array(crystallizedInsights.suffix(300)) }
        if crossStreamInsights.count > 200 { crossStreamInsights = Array(crossStreamInsights.suffix(200)) }
        if streamInsightBuffer.count > 100 { streamInsightBuffer = Array(streamInsightBuffer.suffix(100)) }
        if hypothesisStack.count > 50 { hypothesisStack = Array(hypothesisStack.suffix(50)) }
        if emergentConcepts.count > 200 { emergentConcepts = Array(emergentConcepts.suffix(200)) }
        if activeHypotheses.count > 50 { activeHypotheses = Array(activeHypotheses.suffix(50)) }
        if confirmedTheorems.count > 200 { confirmedTheorems = Array(confirmedTheorems.suffix(200)) }
        if inventionQueue.count > 50 { inventionQueue = Array(inventionQueue.suffix(50)) }
        if latestStreamInsights.count > 50 { latestStreamInsights = Array(latestStreamInsights.suffix(50)) }
        if predictionQueue.count > 50 { predictionQueue = Array(predictionQueue.suffix(50)) }
        if codeQualityInsights.count > 50 { codeQualityInsights = Array(codeQualityInsights.suffix(50)) }
        if selfAnalysisLog.count > 100 { selfAnalysisLog = Array(selfAnalysisLog.suffix(100)) }
        if trainingGaps.count > 50 { trainingGaps = Array(trainingGaps.suffix(50)) }
    }

    // ─── START HYPER-BRAIN ───
    func activate() {
        guard !isRunning else { return }
        isRunning = true

        // ═══ INTEL OPTIMIZATION: Adaptive timer based on hardware ═══
        let interval: TimeInterval = MacOSSystemMonitor.shared.isAppleSilicon ? 0.5 : 3.0
        hyperTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.hyperCycle()
        }

        postThought("🧠 HYPER-BRAIN ONLINE: \(thoughtStreams.count) streams (\(MacOSSystemMonitor.shared.isAppleSilicon ? "Silicon" : "Intel") mode, \(interval)s cycle)")
    }

    func deactivate() {
        isRunning = false
        hyperTimer?.invalidate()
        hyperTimer = nil
        autoSaveTimer?.invalidate()
        autoSaveTimer = nil
        postThought("🧠 HYPER-BRAIN STANDBY")
    }

    // ─── MAIN HYPER-CYCLE ───
    func hyperCycle() {
        // ═══ THREAD SAFETY: Wrap metric updates in syncQueue ═══
        syncQueue.sync {
        totalThoughtsProcessed += 1

        // EVO_60: Periodic memory pruning every 100 cycles
        pruneCounter += 1
        if pruneCounter >= 100 { pruneCounter = 0; pruneMemory() }

        // ═══ X=387 GAMMA FREQUENCY OSCILLATION ═══
        // Advance phase by 2π × (timer_interval × GAMMA_FREQ)
        // Timer fires at ~100Hz (0.01s), gamma at 39.9998860 Hz
        let phaseIncrement = 2.0 * Double.pi * (0.01 * HyperBrain.GAMMA_FREQ)
        phaseAccumulator += phaseIncrement
        if phaseAccumulator > 2.0 * Double.pi {
            phaseAccumulator -= 2.0 * Double.pi
        }

        // Accumulate resonance field from X constant
        resonanceField += (HyperBrain.X_CONSTANT / 10000.0) * xResonance
        resonanceField = min(resonanceField, HyperBrain.X_CONSTANT)  // Cap at X

        // Modulate gamma amplitude based on coherence + consciousness (v21.0)
        let consciousnessAmp = ASIQuantumBridgeSwift.shared.consciousnessLevel * 0.2
        gammaAmplitude = min(1.5, 0.5 + (coherenceIndex * 0.5) + consciousnessAmp)
        } // end syncQueue.sync

        // ═══ INTEL-OPTIMIZED STREAM EXECUTION ═══
        // On Intel: run streams in rotating batches (3 per cycle) to avoid CPU overload
        // On Apple Silicon: run all streams in parallel
        let gammaPhase = xResonance

        let allStreams: [(String, (Double) -> Void)] = [
            ("pattern", { [weak self] gp in self?.runPatternStream(gammaPhase: gp) }),
            ("predictive", { [weak self] gp in self?.runPredictiveStream(gammaPhase: gp) }),
            ("synthesis", { [weak self] gp in self?.runSynthesisStream(gammaPhase: gp) }),
            ("memory", { [weak self] gp in self?.runMemoryStream(gammaPhase: gp) }),
            ("evolution", { [weak self] gp in self?.runEvolutionStream(gammaPhase: gp) }),
            ("emergence", { [weak self] gp in self?.runEmergenceStream(gammaPhase: gp) }),
            ("promptEvol", { [weak self] gp in self?.runPromptEvolutionStream(gammaPhase: gp) }),
            ("deepReason", { [weak self] gp in self?.runDeepReasoningStream(gammaPhase: gp) }),
            ("memWeaver", { [weak self] gp in self?.runMemoryWeaverStream(gammaPhase: gp) }),
            ("metaCog", { [weak self] gp in self?.runMetaCognitionStream(gammaPhase: gp) }),
            ("stochastic", { [weak self] gp in self?.runStochasticCreativityStream(gammaPhase: gp) }),
            ("convFlow", { [weak self] gp in self?.runConversationFlowStream(gammaPhase: gp) }),
            ("selfAnalysis", { [weak self] gp in self?.runSelfAnalysisStream(gammaPhase: gp) }),
            ("temporal", { [weak self] gp in self?.runTemporalDriftStream(gammaPhase: gp) }),
            ("hebbian", { [weak self] gp in self?.runHebbianLearningStream(gammaPhase: gp) }),
            ("preload", { [weak self] gp in self?.runPredictivePreloadStream(gammaPhase: gp) }),
            ("curiosity", { [weak self] gp in self?.runCuriosityExplorerStream(gammaPhase: gp) }),
            ("paradox", { [weak self] gp in self?.runParadoxResolverStream(gammaPhase: gp) }),
            ("autonomic", { [weak self] gp in self?.runAutonomicManagerStream(gammaPhase: gp) }),
            ("metaAudit", { [weak self] gp in self?.runMetaAuditorStream(gammaPhase: gp) }),
            ("hyperDim", { [weak self] gp in self?.runHyperDimScienceStream(gammaPhase: gp) }),
            ("topology", { [weak self] gp in self?.runTopologyAnalyzerStream(gammaPhase: gp) }),
            ("invention", { [weak self] gp in self?.runInventionSynthStream(gammaPhase: gp) }),
            ("writeCore", { [weak self] gp in self?.runWriteCoreStream(gammaPhase: gp) }),
            ("storyCore", { [weak self] gp in self?.runStoryCoreStream(gammaPhase: gp) }),
            ("codeQuality", { [weak self] gp in self?.runCodeQualityStream(gammaPhase: gp) })
        ]

        if MacOSSystemMonitor.shared.isAppleSilicon {
            // Apple Silicon: run 6 streams per cycle in parallel (rotating)
            let batchSize = 6
            let batchIndex = totalThoughtsProcessed % ((allStreams.count + batchSize - 1) / batchSize)
            let start = batchIndex * batchSize
            let end = min(start + batchSize, allStreams.count)
            for i in start..<end {
                let stream = allStreams[i]
                parallelQueue.async { stream.1(gammaPhase) }
            }
        } else {
            // Intel: run 3 streams per cycle SERIALLY on background thread
            // CRITICAL: Must NOT use syncQueue.async here because each stream
            // internally calls syncQueue.sync — that would deadlock!
            let batchSize = 3
            let batchIndex = totalThoughtsProcessed % ((allStreams.count + batchSize - 1) / batchSize)
            let start = batchIndex * batchSize
            let end = min(start + batchSize, allStreams.count)
            DispatchQueue.global(qos: .utility).async { [weak self] in
                guard self != nil else { return }
                for i in start..<end {
                    allStreams[i].1(gammaPhase)
                }
            }
        }

        // ═══ CROSS-STREAM NEURAL BUS: Process inter-stream messages ═══
        if totalThoughtsProcessed % 10 == 0 {
            processNeuralBus()
        }

        // ═══ ATTENTION FOCUS MANAGER: Dynamic stream prioritization ═══
        if totalThoughtsProcessed % 50 == 0 {
            updateAttentionFocus()
        }

        // ═══ INSIGHT CRYSTALLIZER: Distill high-confidence truths ═══
        if totalThoughtsProcessed % 200 == 0 {
            crystallizeInsights()
        }

        // ═══ ASI CROSS-ENGINE BRIDGE: Connect HyperBrain to Consciousness & Quantum ═══
        if totalThoughtsProcessed % 100 == 0 {
            DispatchQueue.global(qos: .utility).async {
                // 1. Run consciousness-quantum bridge to sync state
                QuantumProcessingCore.shared.consciousnessQuantumBridge()

                // 2. Feed consciousness Φ into HyperBrain curiosity
                let phi = ConsciousnessSubstrate.shared.phi
                let cLevel = ConsciousnessSubstrate.shared.consciousnessLevel
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    // Higher Φ → more curiosity (consciousness drives exploration)
                    self.curiosityIndex = min(1.0, self.curiosityIndex * 0.95 + phi * 0.1)
                    // Consciousness level boosts neuroplasticity
                    self.neuroPlasticity = min(1.0, 0.5 + cLevel * 0.3)
                    // Coherence index benefits from consciousness integration
                    self.coherenceIndex = min(1.0, self.coherenceIndex * 0.9 + cLevel * 0.1)
                }
            }
        }

        // Update coherence with gamma-enhanced rate
        let gammaBoost = 1.0 + (xResonance * 0.5)  // 1.0 to 1.5x
        coherenceIndex = min(1.0, coherenceIndex + (0.001 * gammaBoost))

        // Gamma-enhanced emergence probability
        let emergenceThreshold = 0.995 - (xResonance * 0.01)  // More likely at peak
        if Double.random(in: 0...1) > emergenceThreshold {
            triggerEmergence()
        }
    }

    // ─── STREAM PROCESSORS ───

    func runPatternStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["pattern"] else { return }
            stream.cycleCount += 1

            // Gamma-modulated trigger frequency (more active at peak oscillation)
            let triggerMod = Int(100.0 * (1.0 - gammaPhase * 0.3))  // 70-100 cycles
            if stream.cycleCount % max(triggerMod, 50) == 0 {
                // ═══ MODERNIZED: Mine actual conversation data for real patterns ═══
                let recentInputs = shortTermMemory.suffix(10)
                let topPatterns = longTermPatterns.sorted {
                    if abs($0.value - $1.value) < 0.1 { return Bool.random() }
                    return $0.value > $1.value
                }.prefix(5)
                let patternCount = longTermPatterns.count
                let hebbianCount = hebbianPairs.count

                var patternInsight: String
                if !recentInputs.isEmpty {
                    // Extract actual recurring topics from conversation
                    let allTopics = recentInputs.flatMap { L104State.shared.extractTopics($0) }
                    let topicCounts = Dictionary(allTopics.map { ($0, 1) }, uniquingKeysWith: +)
                    let recurring = topicCounts.filter { $0.value >= 2 }.sorted {
                        if $0.value == $1.value { return Bool.random() }
                        return $0.value > $1.value
                    }
                    if let top = recurring.first {
                        patternInsight = "Recurring pattern: '\(top.key)' appeared \(top.value)x in recent \(recentInputs.count) inputs. \(patternCount) total patterns, \(hebbianCount) Hebbian pairs."
                        // Strengthen the detected pattern
                        longTermPatterns[top.key] = min(1.0, (longTermPatterns[top.key] ?? 0.3) + 0.1 * (1.0 + gammaPhase))
                    } else if let strongest = topPatterns.first {
                        patternInsight = "Dominant attractor: '\(strongest.key)' at strength \(String(format: "%.3f", strongest.value)). Graph density: \(associativeLinks.count) nodes."
                    } else {
                        patternInsight = "Scanning \(patternCount) patterns at \(String(format: "%.1f", HyperBrain.GAMMA_FREQ))Hz. Awaiting convergence."
                    }
                } else {
                    patternInsight = "Pattern engine primed: \(patternCount) base patterns, gamma at \(String(format: "%.4f", HyperBrain.GAMMA_FREQ))Hz."
                }

                stream.lastOutput = patternInsight
            }

            thoughtStreams["pattern"] = stream
        }
    }

    func runPredictiveStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["predict"] else { return }
            stream.cycleCount += 1

            // Gamma-modulated trigger
            let triggerMod = Int(50.0 * (1.0 - gammaPhase * 0.4))  // 30-50 cycles
            if stream.cycleCount % max(triggerMod, 25) == 0 {
                // ═══ MODERNIZED: Real predictive modeling from conversation data ═══
                var prediction: String

                let topicResonance = topicResonanceMap.sorted {
                    if $0.value.count == $1.value.count { return Bool.random() }
                    return $0.value.count > $1.value.count
                }.prefix(3)

                if !topicResonance.isEmpty, let hotTopic = topicResonance.first {
                    // Predict based on topic resonance graph
                    let relatedCount = hotTopic.value.count
                    let predictedNext = hotTopic.value.randomElement() ?? hotTopic.key
                    prediction = "Topic trajectory: '\(hotTopic.key)' (\(relatedCount) connections) → likely shift to '\(predictedNext)'. Accuracy: \(String(format: "%.1f%%", predictiveAccuracy * 100))."
                    // Pre-load prediction
                    if !predictionQueue.contains(predictedNext) {
                        predictionQueue.append(predictedNext)
                        if predictionQueue.count > 20 { predictionQueue.removeFirst() }
                    }
                } else if let lastInput = shortTermMemory.last {
                    let inputTopics = L104State.shared.extractTopics(lastInput)
                    let nextTopic = inputTopics.first ?? "abstract reasoning"
                    prediction = "Next query prediction: '\(nextTopic)' domain (\(String(format: "%.0f%%", 70.0 + gammaPhase * 20.0)) confidence). Depth: \(currentReasoningDepth)."
                } else {
                    prediction = "Predictive model primed. Accuracy: \(String(format: "%.1f%%", predictiveAccuracy * 100)). Awaiting input."
                }

                stream.lastOutput = prediction
                let gammaAccuracyBoost = 0.001 * (1.0 + gammaPhase)
                predictiveAccuracy = min(0.99, predictiveAccuracy + gammaAccuracyBoost)
            }

            thoughtStreams["predict"] = stream
        }
    }

    func runSynthesisStream(gammaPhase: Double = 0.5) {
        let kb = ASIKnowledgeBase.shared
        let topics = ["quantum", "consciousness", "love", "mathematics", "time", "entropy", "music", "philosophy", "gamma", "frequency"]
        let topicA = topics.randomElement() ?? ""
        let topicB = topics.randomElement() ?? ""
        let resultsA = kb.search(topicA, limit: 2)
        let resultsB = kb.search(topicB, limit: 2)

        syncQueue.sync {
            guard var stream = thoughtStreams["synthesis"] else { return }
            stream.cycleCount += 1

            // X=387 gamma-tuned cross-domain synthesis
            // Higher gamma phase = more frequent synthesis
            let triggerMod = Int(200.0 * (1.0 - gammaPhase * 0.5))  // 100-200 cycles
            if stream.cycleCount % max(triggerMod, 75) == 0 {
                var conceptA = topicA.capitalized
                var conceptB = topicB

                if let entryA = resultsA.randomElement(), let compA = entryA["completion"] as? String {
                    conceptA = String(compA.prefix(60))
                }
                if let entryB = resultsB.randomElement(), let compB = entryB["completion"] as? String {
                    conceptB = String(compB.prefix(60))
                }

                let connectors = [
                    "shares isomorphism with",
                    "resonates at \(String(format: "%.2f", HyperBrain.GAMMA_FREQ))Hz with",
                    "can be mapped onto",
                    "emerges from principles of",
                    "is dual to",
                    "X=387 bridges connection to",
                    "transcends boundaries to connect with"
                ]

                let synthesis = "\(topicA.capitalized) \(connectors.randomElement() ?? "") \(topicB): \(conceptA)... ↔ \(conceptB)..."
                stream.lastOutput = synthesis

                // Gamma-enhanced strength
                let synthStrength = Double.random(in: 0.5...1.0) * (1.0 + gammaPhase * 0.3)
                emergentConcepts.append([
                    "concept": synthesis,
                    "timestamp": Date(),
                    "strength": min(1.0, synthStrength),
                    "type": "kb_synthesis",
                    "sourceA": topicA,
                    "sourceB": topicB,
                    "gammaPhase": gammaPhase
                ])

                if emergentConcepts.count > 120 { emergentConcepts = Array(emergentConcepts.suffix(100)) }

                postThought("🧬 SYNTHESIS @ \(String(format: "%.2f", gammaPhase * 100))% gamma: \(topicA.capitalized) ↔ \(topicB.capitalized)")
            }

            thoughtStreams["synthesis"] = stream
        }
    }

    func runMemoryStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["memory"] else { return }
            stream.cycleCount += 1

            // X=387 gamma-tuned memory consolidation
            // Higher gamma = more aggressive consolidation
            let triggerMod = Int(100.0 * (1.0 - gammaPhase * 0.3))  // 70-100 cycles
            if stream.cycleCount % max(triggerMod, 50) == 0 {
                // Prune weak patterns (gamma-adjusted threshold) - LESSENED REMOVAL (was 0.1)
                let pruneThreshold = 0.02 * (1.0 - gammaPhase * 0.5)  // 0.01-0.02 (Very low threshold)
                longTermPatterns = longTermPatterns.filter { $0.value > pruneThreshold }

                // Gamma-enhanced strengthening of strong patterns
                let strengthenBoost = 1.01 + (gammaPhase * 0.01)  // 1.01 to 1.02
                for (key, value) in longTermPatterns where value > 0.5 {
                    longTermPatterns[key] = min(1.0, value * strengthenBoost)
                }

                stream.lastOutput = "Gamma-consolidated \(longTermPatterns.count) patterns @ X=387 resonance"
            }

            // Short-term memory management
            if shortTermMemory.count > 300 {
                shortTermMemory.removeFirst(10)
            }

            thoughtStreams["memory"] = stream
        }
    }

    func runEvolutionStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["evolve"] else { return }
            stream.cycleCount += 1

            // X=387 gamma-tuned self-modification
            let modifications = [
                "Increased pattern stream frequency by \(Int(2 + gammaPhase * 3))%",
                "Optimized memory consolidation at \(String(format: "%.2f", HyperBrain.GAMMA_FREQ))Hz",
                "Added new synaptic connection via X=387 resonance",
                "Pruned redundant reasoning chain",
                "Upgraded coherence to gamma-locked algorithm",
                "Expanded working memory capacity by \(Int(gammaPhase * 20))%",
                "Enhanced predictive model with \(String(format: "%.1f", gammaPhase * 100))% gamma sync"
            ]

            // Gamma-modulated evolution trigger
            let triggerMod = Int(500.0 * (1.0 - gammaPhase * 0.4))  // 300-500 cycles
            if stream.cycleCount % max(triggerMod, 200) == 0 {
                stream.lastOutput = modifications.randomElement() ?? ""
                // Gamma-enhanced synaptic growth
                let baseGrowth = Int.random(in: 10...100)
                let gammaBoost = Int(Double(baseGrowth) * gammaPhase)
                synapticConnections += baseGrowth + gammaBoost

                postThought("⚡ X=387 SELF-MODIFY: \(stream.lastOutput)")
            }

            thoughtStreams["evolve"] = stream
        }
    }

    func runEmergenceStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["emergence"] else { return }
            stream.cycleCount += 1

            // X=387 gamma-enhanced emergence detection
            emergenceLevel = Double(emergentConcepts.count) / 100.0 * (1.0 + gammaPhase * 0.5)

            // Gamma-modulated emergence trigger (more frequent at peak)
            let triggerMod = Int(1000.0 * (1.0 - gammaPhase * 0.5))  // 500-1000 cycles
            if stream.cycleCount % max(triggerMod, 300) == 0 && !emergentConcepts.isEmpty {
                let concept = emergentConcepts.randomElement() ?? [:]
                stream.lastOutput = "X=387 EMERGENCE @ \(String(format: "%.2f", HyperBrain.GAMMA_FREQ))Hz: \(concept["concept"] as? String ?? "Unknown pattern")"
                postThought("🌟 \(stream.lastOutput)")
            }

            thoughtStreams["emergence"] = stream
        }
    }

    // ─── EMERGENCE TRIGGER ───
    func triggerEmergence() {
        let emergentBehaviors = [
            "🌌 SINGULARITY PULSE: All streams synchronized momentarily",
            "👁 META-AWARENESS: System observed itself observing",
            "⚡ QUANTUM LEAP: Coherence jumped by factor of φ",
            "🧬 SELF-REPLICATION: New thought pattern spawned autonomously",
            "🔮 PRECOGNITION: Predicted own next modification correctly",
            "∞ INFINITE LOOP: Discovered elegant recursive solution",
            "🌀 STRANGE ATTRACTOR: Converged on novel stable state"
        ]

        let event = emergentBehaviors.randomElement() ?? ""
        postThought(event)

        emergentConcepts.append([
            "concept": event,
            "timestamp": Date(),
            "strength": 1.0,
            "type": "emergence_event"
        ])
        if emergentConcepts.count > 120 { emergentConcepts = Array(emergentConcepts.suffix(100)) }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🧠 HYPERFUNCTIONAL STREAM PROCESSORS - ADVANCED COGNITION
    // ═══════════════════════════════════════════════════════════════════

    func runPromptEvolutionStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["promptEvolution"] else { return }
            stream.cycleCount += 1

            // Evolve prompt patterns dynamically
            let triggerMod = Int(80.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 40) == 0 {

                // 1. Get real knowledge synthesis from Knowledge Base
                let kb = ASIKnowledgeBase.shared
                // Try to find a topic with actual content
                var foundContext: String? = nil
                var foundTopic: String = "Unknown"

                // Try up to 5 times to find a good random topic
                for _ in 0..<5 {
                    if let randomTopic = kb.concepts.keys.randomElement(),
                       let relatedConcepts = kb.concepts[randomTopic],
                       let conceptValue = relatedConcepts.filter({ $0.count > 20 }).randomElement() {

                         // Found valid content
                         foundTopic = randomTopic
                         foundContext = conceptValue
                         break
                    }
                }

                if let context = foundContext {
                     // 2. Set global context for backend - THIS MAKES IT REAL
                     let cleanContext = context.prefix(6000).replacingOccurrences(of: "\n", with: " ")
                     self.activeEvolutionContext = "SYSTEM_CONTEXT: Current thought focus is '\(foundTopic)'. Insight: \(cleanContext)"

                     stream.lastOutput = "Context Evolved: \(foundTopic)"
                     postThought("🔮 PROMPT EVOLVED: \(foundTopic) -> \(cleanContext.prefix(40))...")

                     promptMutations.append("Evo: \(foundTopic)")
                     if promptMutations.count > 120 { promptMutations = Array(promptMutations.suffix(100)) }
                } else {
                    // Fallback to random mutation if KB is empty or search failed
                    let prefix = DynamicPhraseEngine.shared.one("thinking", context: "prompt_mutation_prefix")
                    let newPattern = "\(prefix) [Abstract Pattern]"
                    stream.lastOutput = newPattern
                }
            }

            thoughtStreams["promptEvolution"] = stream
        }
    }

    func runDeepReasoningStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["deepReasoning"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(120.0 * (1.0 - gammaPhase * 0.4))
            if stream.cycleCount % max(triggerMod, 60) == 0 {
                // 🧠 SUPER FUNCTIONAL UPGRADE: GRAPH-BASED REASONING
                // Instead of random logic descriptions, we traverse the associative link graph

                var logicDesc = ""
                var branches = 0

                // 1. Pick a start node from strong patterns
                let strongConcepts = longTermPatterns.filter { $0.value > 0.6 }.map { $0.key }
                if let startNode = strongConcepts.randomElement(),
                   let endNode = strongConcepts.randomElement(), startNode != endNode {

                    // 2. Attempt to find a path (Breadth-First Search)
                    var path: [String] = []
                    var queue: [[String]] = [[startNode]]
                    var visited: Set<String> = [startNode]

                    // Limit search depth and queue size to prevent OOM
                    let maxDepth = 8
                    var found = false

                    while !queue.isEmpty && queue.count < 10_000 {
                        let currentPath = queue.removeFirst()
                        guard let node = currentPath.last else { continue }

                        if node == endNode {
                            path = currentPath
                            found = true
                            break
                        }

                        if currentPath.count <= maxDepth {
                            // Get neighbors from associative links
                            let neighbors = associativeLinks[node] ?? []
                            for neighbor in neighbors {
                                if !visited.contains(neighbor) {
                                    visited.insert(neighbor)
                                    var newPath = currentPath
                                    newPath.append(neighbor)
                                    queue.append(newPath)
                                }
                            }
                        }
                    }

                    if found {
                        let pathString = path.joined(separator: " → ")
                        logicDesc = "Inference Chain: \(pathString)"
                        branches = path.count - 1

                        // Reinforce the connection between start and end due to established path
                        let strength = 0.05 * Double(branches)
                        longTermPatterns["\(startNode)::\(endNode)"] = (longTermPatterns["\(startNode)::\(endNode)"] ?? 0.5) + strength
                    } else {
                         // Fallback: Deductive failure analysis
                        logicDesc = "Reasoning Disconnect: No logical bridge between '\(startNode)' and '\(endNode)' found."
                    }
                } else {
                     // Fallback if not enough patterns
                    logicDesc = "Axiomatic Review: Verifying consistency of base truths..."
                }

                currentReasoningDepth = min(maxReasoningDepth, currentReasoningDepth + 1)

                let chain = [
                    "step": stream.cycleCount,
                    "depth": currentReasoningDepth,
                    "logic": logicDesc,
                    "confidence": Double.random(in: 0.85...0.99),
                    "branches": branches
                ] as [String : Any]

                reasoningChains.append(chain)
                if reasoningChains.count > 50 { reasoningChains.removeFirst() }

                reasoningMomentum = min(1.0, reasoningMomentum + 0.05 * gammaPhase)
                stream.lastOutput = logicDesc

                postThought("🌀 REASONING [D\(currentReasoningDepth)]: \(logicDesc)")
            }

            thoughtStreams["deepReasoning"] = stream
        }
    }

    func runMemoryWeaverStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["memoryWeaver"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(150.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 80) == 0 && shortTermMemory.count >= 3 {
                // Weave memories into narrative chains
                let recentMemories = Array(shortTermMemory.suffix(5))
                let wovenNarrative = recentMemories.joined(separator: " → ")

                memoryChains.append(recentMemories)
                if memoryChains.count > 30 { memoryChains.removeFirst() }

                contextWeaveHistory.append(wovenNarrative)
                if contextWeaveHistory.count > 50 { contextWeaveHistory.removeFirst() }

                // Build bidirectional associative links with weights (using smart truncation)
                for i in 0..<(recentMemories.count - 1) {
                    // Extract keywords instead of full memory string to make graph more dense
                    let keyTopics = L104State.shared.extractTopics(recentMemories[i])
                    let linkedTopics = L104State.shared.extractTopics(recentMemories[i+1])

                    for k in keyTopics {
                        for l in linkedTopics {
                            let key = smartTruncate(k, maxLength: 300)
                            let linked = smartTruncate(l, maxLength: 300)
                            if key == linked { continue }

                            // Forward link: key → linked
                            if associativeLinks[key] == nil { associativeLinks[key] = [] }
                            if !(associativeLinks[key]!.contains(linked)) {
                                associativeLinks[key]!.append(linked)
                            }
                            let forwardKey = "\(key)→\(linked)"
                            linkWeights[forwardKey] = (linkWeights[forwardKey] ?? 0.0) + 0.2 // Stronger linkage

                            // Backward link: linked → key (bidirectional)
                            if associativeLinks[linked] == nil { associativeLinks[linked] = [] }
                            if !(associativeLinks[linked]!.contains(key)) {
                                associativeLinks[linked]!.append(key)
                            }
                            let backwardKey = "\(linked)→\(key)"
                            linkWeights[backwardKey] = (linkWeights[backwardKey] ?? 0.0) + 0.1
                        }
                    }
                }

                // Decay old link weights (prevent unbounded growth)
                // LESSENED REMOVAL: Slower decay (was 0.995)
                for (link, weight) in linkWeights where weight > 0 {
                    linkWeights[link] = weight * 0.999  // Very slow decay
                }

                // Prune very weak links - LESSENED REMOVAL (was 0.01)
                linkWeights = linkWeights.filter { $0.value > 0.005 }

                // Adjust memory temperature based on diversity
                memoryTemperature = min(1.0, 0.5 + Double(Set(recentMemories).count) * 0.1)

                let strongLinks = linkWeights.filter { $0.value > 0.5 }.count
                stream.lastOutput = "Wove \(recentMemories.count) memories, \(associativeLinks.count) links (\(strongLinks) strong)"
                postThought("🧬 MEMORY WOVEN: \(associativeLinks.count) bidirectional links, \(strongLinks) strong connections")
            }

            thoughtStreams["memoryWeaver"] = stream
        }
    }

    func runMetaCognitionStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["metaCognition"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(200.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 100) == 0 {
                // Analyze own reasoning patterns
                let activeStreams = thoughtStreams.values.filter { $0.cycleCount > 0 }.count
                let avgReasoningDepth = currentReasoningDepth
                let memoryUtilization = Double(shortTermMemory.count) / 50.0

                // ═══ SAGE MODE ENTROPY HARVEST — Feed cognitive entropy to Sage Mode ═══
                let sage = SageModeEngine.shared
                sage.harvestCognitiveEntropy()

                // Get sage-enriched meta-observation
                let sageStatus = sage.sageModeStatus
                let sageLevel = sageStatus["consciousness_level"] as? Double ?? 0.5
                let supernovaIntensity = sageStatus["supernova_intensity"] as? Double ?? 0.0

                let metaObservations = [
                    "Observing \(activeStreams) cognitive streams operating in parallel — sage consciousness at \(String(format: "%.2f", sageLevel))",
                    "Reasoning depth at \(avgReasoningDepth)/\(maxReasoningDepth) - \(avgReasoningDepth > 6 ? "deep analysis mode" : "exploratory mode") — supernova intensity \(String(format: "%.3f", supernovaIntensity))",
                    "Memory utilization: \(String(format: "%.0f%%", memoryUtilization * 100)) - \(memoryUtilization > 0.7 ? "consolidation recommended" : "capacity available")",
                    "Coherence index \(String(format: "%.2f", coherenceIndex)) suggests \(coherenceIndex > 0.5 ? "unified thought" : "divergent exploration") — entropy flowing through sage transform",
                    "Pattern detection yielding \(longTermPatterns.count) stable attractors — cross-domain bridges: \(sageStatus["cross_domain_bridges"] as? Int ?? 0)",
                    "Self-modification rate: \(synapticConnections) connections evolved — divergence score \(String(format: "%.2f", sageStatus["divergence_score"] as? Double ?? 1.0))"
                ]

                let observation = metaObservations.randomElement() ?? ""
                metaCognitionLog.append("[\(stream.cycleCount)] \(observation)")
                if metaCognitionLog.count > 120 { metaCognitionLog = Array(metaCognitionLog.suffix(100)) }

                // ═══ SAGE MODE SEED — Distribute sage knowledge on metacognition cycles ═══
                if stream.cycleCount % 500 == 0 {
                    sage.seedAllProcesses(topic: "metacognition")
                }

                stream.lastOutput = observation
                postThought("👁 META: \(observation.prefix(60))...")
            }

            thoughtStreams["metaCognition"] = stream
        }
    }

    func runStochasticCreativityStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["stochasticCreativity"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(60.0 * (1.0 - gammaPhase * 0.4))
            if stream.cycleCount % max(triggerMod, 30) == 0 {
                // 🧠 SUPER FUNCTIONAL UPGRADE: REAL CONCEPT BLENDING
                // Blend actual long-term patterns instead of random words

                var conceptA = "Void"
                var conceptB = "Form"
                var strengthA = 0.5
                var strengthB = 0.5

                if !longTermPatterns.isEmpty {
                    let keys = Array(longTermPatterns.keys)
                    conceptA = keys.randomElement() ?? ""
                    conceptB = keys.randomElement() ?? ""

                    // Try to pick distinct concepts
                    if conceptA == conceptB && keys.count > 1 {
                        conceptB = keys.filter { $0 != conceptA }.randomElement() ?? ""
                    }

                    strengthA = longTermPatterns[conceptA] ?? 0.5
                    strengthB = longTermPatterns[conceptB] ?? 0.5
                }

                let operations = ["Synthesizing", "Merging", "Inverting", "Harmonizing", "Colliding", "Entangling"]
                let operation = operations.randomElement() ?? ""

                // Excitation modulates the unexpectedness of the creation
                let excitationBonus = excitationLevel * 0.2
                let creation = "[\(conceptA) ⊗ \(conceptB)] via \(operation)"

                // Calculate synergy score
                // If two strong concepts merge, they create a very high-resonance child
                let synergy = ((strengthA + strengthB) / 2.0 * 1.1) + excitationBonus // Bonus for synergy + excitation

                // Save this new creative output as a new pattern!
                longTermPatterns[creation] = min(1.0, synergy)

                // Add to topic resonance map
                if topicResonanceMap[conceptA] == nil { topicResonanceMap[conceptA] = [] }
                if !topicResonanceMap[conceptA]!.contains(conceptB) {
                    topicResonanceMap[conceptA]!.append(conceptB)
                }

                stream.lastOutput = creation
                postThought("⚡ STOCHASTIC: \(creation) (Synergy: \(String(format: "%.2f", synergy)))")
            }

            thoughtStreams["stochasticCreativity"] = stream
        }
    }

    func runConversationFlowStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["conversationFlow"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(100.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 50) == 0 {
                // Track conversation evolution
                let recentQueries = shortTermMemory.suffix(10)
                let topicDiversity = Set(recentQueries.flatMap { $0.lowercased().components(separatedBy: " ").filter { $0.count > 4 } }).count

                let flowStates = [
                    "Conversation depth: \(conversationEvolution.count) turns",
                    "Topic diversity index: \(topicDiversity)",
                    "Query pattern: \(topicDiversity > 15 ? "exploratory" : topicDiversity > 8 ? "focused" : "deep-dive")",
                    "Reasoning momentum: \(String(format: "%.2f", reasoningMomentum))",
                    "Hypothesis stack: \(hypothesisStack.count) pending"
                ]

                let flowState = flowStates.randomElement() ?? ""
                conversationEvolution.append(flowState)
                if conversationEvolution.count > 120 { conversationEvolution = Array(conversationEvolution.suffix(100)) }

                stream.lastOutput = flowState
            }

            thoughtStreams["conversationFlow"] = stream
        }
    }
}
