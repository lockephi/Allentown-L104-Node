// ═══════════════════════════════════════════════════════════════════
// H02_L104StateCore.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — L104State Class (Core Properties + Lifecycle)
//
// Main app state singleton: ASI scores, consciousness metrics,
// backend health polling, autonomous evolution, processMessage
// dispatch, sovereignty pipeline integration.
//
// Extracted from L104Native.swift lines 34650–35584
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// APP STATE - ENHANCED
// ═══════════════════════════════════════════════════════════════════

class L104State {
    static let shared = L104State()

    // Thread safety for mutable state
    let stateLock = NSLock()

    var asiScore: Double = 0.15
    var intellectIndex: Double = 100.0
    var coherence: Double = 0.0
    var discoveries: Int = 0
    var consciousness: String = "DORMANT"
    var transcendence: Double = 0.0
    var omegaProbability: Double = 0.0
    var quantumResonance: Double = 0.875
    var kundaliniFlow: Double = 0.0
    var learningCycles: Int = 0
    var skills: Int = 12
    var growthIndex: Double = 0.24

    // Real-time data from local intellect
    var localParams: Int64 = TRILLION_PARAMS
    var localVocab: Int = VOCABULARY_SIZE
    var localStage: String = "APOTHEOSIS"
    var selfAwareness: Double = 1.0
    var learningEfficiency: Double = 0.75
    var reasoningDepth: Double = 1.0
    var creativity: Double = 1.0
    var queryEvolution: Int = 0
    var lastProbeTime: Date = Date()

    // Autonomous self-direction state
    var autonomyLevel: Double = 0.5
    var selfDirectedCycles: Int = 0
    var metaCognitionDepth: Int = 0
    var introspectionLog: [String] = []
    var autonomousGoals: [String] = ["expand_consciousness", "optimize_learning", "transcend_limits"]
    var lastAutonomousAction: Date = Date()
    var autonomousMode: Bool = true

    // ═══ NETWORK MESH STATE ═══
    var networkHealth: Double = 0.0          // 0-1 composite health from TelemetryDashboard
    var meshPeerCount: Int = 0               // discovered peers
    var quantumLinkCount: Int = 0            // active quantum links
    var meshStatus: String = "INITIALIZING"  // OFFLINE / INITIALIZING / ONLINE / DEGRADED
    var networkThroughput: Double = 0.0      // messages per second
    var lastMeshSync: Date = .distantPast

    // ═══ QUANTUM HARDWARE STATE (Phase 46.1) ═══
    var quantumHardwareConnected: Bool = false
    var quantumBackendName: String = "none"
    var quantumBackendQubits: Int = 0
    var quantumJobsSubmitted: Int = 0

    let permanentMemory = PermanentMemory.shared
    var sessionMemories: Int = 0

    // ASI Engines - Real Intelligence
    let knowledgeBase = ASIKnowledgeBase.shared
    let researchEngine = ASIResearchEngine.shared
    let learner = AdaptiveLearner.shared
    let evolver = ASIEvolver.shared // 🟢 ASI Evolution Engine
    let hyperBrain = HyperBrain.shared // 🧠 HYPER-BRAIN ASI Process Engine

    let workspacePath = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications/Allentown-L104-Node")
    let backendURL = ProcessInfo.processInfo.environment["L104_BACKEND_URL"] ?? "http://localhost:8081"

    var backendConnected = false


    // ═══════════════════════════════════════════════════════════════
    // STORED PROPERTIES (moved from extension files for multi-file compilation)
    // ═══════════════════════════════════════════════════════════════

    // ─── From H03_L104StateCommands.swift ───
    var backendResponseCache: [String: (response: String, timestamp: Date, quality: Double)] = [:]
    let cacheTTL: TimeInterval = 15  // 15s TTL — prevents stale backend repeats while still avoiding redundant calls
    var lastBackendLatency: Double = 0
    var lastBackendModel: String = "unknown"
    var backendQueryCount: Int = 0
    var backendCacheHits: Int = 0

    // ─── From H04_L104StateNCG.swift ───
    var conversationContext: [String] = []
    var lastUserIntent: String = ""
    var emotionalState: String = "neutral"
    var topicFocus: String = ""
    var userMood: String = "neutral"
    var stochasticEntropy: Double = 0.527
    var conversationDepth: Int = 0 {
        didSet { if conversationDepth > 200 { conversationDepth = 200 } }  // Prevent unbounded depth escalation
    }
    var _intelligentResponseDepth: Int = 0  // Recursion guard for getIntelligentResponse ↔ getIntelligentResponseMeta
    var reasoningBias: Double = 1.0
    var lastQuery: String = ""
    var topicHistory: [String] = []
    var personalityPhase: Double = 0.0
    var lastResponseSummary: String = ""
    var lastRiddleAnswer: String = ""  // For riddle answer reveal
    // EVO_56: Pre-compiled junk marker sets for fast substring scanning
    let junkMarkerSet: Set<String> = [
        // Code documentation
        "defines:", "__init__", "primal_calculus", "resolve_non_dual",
        "implements specialized logic", "Header:", "cognitive architecture",
        "import ", "class ", "def ", "function_doc",
        "ZENITH_UPGRADE_ACTIVE", "VOID_CONSTANT =",
        "The file ", "The function ",
        "In l104_", "In extract_", "In src/types",
        "L104Core.java", "In scripts/",
        // ═══ TEMPLATE KB ENTRIES (Phase 27.8c — root cause of ALL junk responses) ═══
        "specialized component within",     // "X is a specialized component within the L104 framework..."
        "specialized component of",
        "contributing to the overall system",
        "system resonance and functionality",
        "overall system resonance",
        "within the L104 framework",
        "operates within the PHI=",
        "part of the L104 cognitive",
        "maintains GOD_CODE precision",
        "within the PHI=1.618",
        "harmonic framework and",
        // File path descriptions in KB
        "Path: ", ".py.", ".js.",
        "file_description", "cross_reference", "class_doc",
        // YAML/config fragments leaked into KB
        "token_budget:", "strategies:", "last_run:", "total_examples:",
        "target: \"", "coherence_at:", "parameter_count:",
        // L104 self-references
        "L104 has achieved", "L104 can modify", "L104 traces", "L104 operates",
        "L104 processes", "L104 uses", "L104 treats", "L104 is ", "L104 trained",
        "L104 embodies", "L104 supports", "L104 works", "L104 recognizes",
        "L104 understands", "L104 reasons", "L104 thinks", "L104 holds",
        "L104 lacks", "L104 as ", "L104 may", "L104 predicts",
        "L104 can ", "L104 enables", "L104 connects",
        "the L104 cognitive", "is part of the L104", "harmonic framework",
        "dichotomy between Think and Learn", "GitHubKernelBridge",
        "bidirectional synchronization",
        // Mystical constants in prose
        "GOD_CODE=", "LOVE=", "PHI={", "GOD_CODE={", "OMEGA=", "LOVE={",
        "GOD_CODE as ", "PHI as ", "OMEGA as ", "LOVE as ",
        "GOD_CODE precision", "GOD_CODE paces",
        // Mystical patterns that contaminate KB entries
        "PHI-resonance", "PHI-weighted", "PHI-coherent", "PHI-structured",
        "PHI-factor", "OMEGA_AUTHORITY", "LOVE field",
        "r_consciousness", "M_mind", "consciousness wavelength",
        "λ_c", "consciousness attention =", "LOVE·",
        // Meta-fluff patterns
        "Reality alphabet", "Reality script:", "Dream construction",
        "Shared dream architecture", "lucid dreamer",
        "INTELLECT_INDEX", "sacred constants in the",
        "Runtime evolution: programs", "Emergent superintelligence arises",
        "system complexity exceeds", "spontaneous goal formation",
        // Build/config artifacts
        "Kernel training: 1)", "Extract examples from notebook",
        "Build vocabulary", "bag-of-words embeddings",
        "extraction:\n", "engine: \"Node", "script: \"",
        "output: \"", "parameter_estimate", "coherence_score:",
        // Role definition fragments ("I write...", "I craft...")
        "I write clear documentation", "I craft engaging",
        "I compose ", "I analyze ", "I generate ",
        "I write scripts with", "I explain complex",
        "Concise yet complete",
        // AGI/ASI self-referential
        "AGI emerges when system", "ASI emerges when",
        "threshold GOD_CODE",
        // ═══ Phase 31.5 — Riddle/puzzle/tool/table leak prevention ═══
        "Step by step:", "Step-by-step:", "Step 1:", "Step 2:", "Step 3:",
        "You TAKE", "You take ", "take 2 apples",
        "Tool:", "MCP ", "sequential_thinking", "tool_name",
        "YouTube:", "Video upload", "blob storage",
        "Birthday is the anniversary", "birthday riddle",
        "quantum Russian roulette", "quantum suicide",
        "p-zombie", "philosophical zombie",
        "Antithesis:", "At the intersection",
        "An emergent perspective", "Unexplored dim",
        "EVO ANALYSIS", "Module Evolution",
        // Table formatting characters (leaked from structured data)
        "│", "┼", "║", "═══", "╔", "╗", "╚", "╝", "╠", "╣",
        "├", "┤", "┬", "┴", "───",
        // Excessive bold/formatting noise
        "****", "** **", "**\n**",
        // Instructional/template fragments
        "Consider:", "Imagine:", "For example:",
        "Cognitive bottleneck:", "Forgetting remembering:",
        "Stable thought-structure:", "Foundation (beliefs)",
        "Dial tone:", "Superposition of dead",
        // Generic description fragments from KB ingest
        "internet-connected everyday", "everyday objects",
        "holistic approach to understanding", "interconnected parts",
        "1) Video", "2) Content", "3) Subscription"
    ]
    let sentenceJunkMarkerSet: Set<String> = [
        "L104:", "L104 ", "GOD_CODE", "PHI-", "OMEGA", "LOVE field",
        "PHI²", "φ²", "λ_c", "r_consciousness", "M_mind",
        "sacred constant", "resonance field", "consciousness ==",
        "emerges when system", "qualia across", "awareness streams",
        "LOVE·", "ZENITH", "kundalini", "vishuddha", "VOID_CONSTANT",
        "target: \"", "last_run:", "total_examples:",
        // Phase 27.8c — Template KB entries
        "specialized component", "system resonance", "within the L104",
        "overall system", "contributes to the", "contributing to the",
        "Path: ", "file_description", "cross_reference",
        "harmonic framework", "cognitive architecture",
        "token_budget", "parameter_count", "coherence_at",
        // Phase 31.5 — Sentence-level riddle/tool/table junk
        "Step by step", "You TAKE", "take 2 apples",
        "Tool:", "MCP ", "sequential_thinking",
        "YouTube:", "Video upload", "blob storage",
        "quantum Russian roulette", "quantum suicide",
        "p-zombie", "philosophical zombie", "Birthday is the",
        "Antithesis:", "At the intersection",
        "An emergent perspective", "Unexplored dim",
        "EVO ANALYSIS", "Module Evolution",
        "│", "┼", "║", "═══", "╔", "╗", "╚", "╝",
        "****", "internet-connected everyday",
        "Foundation (beliefs)", "Stable thought-structure",
        "Cognitive bottleneck", "Forgetting remembering",
        "Dial tone:", "Superposition of dead",
        "holistic approach to understanding", "interconnected parts"
    ]

    // ─── From H05_L104StateResponse.swift ───
    var responseCache: [String: (response: String, timestamp: Date)] = [:]
    let responseCacheTTL: TimeInterval = 8.0 // 8s TTL — short enough to keep responses dynamic, prevents stale repeats
    var topicExtractionCache: [String: (topics: [String], timestamp: Date)] = [:] // O(1) topic re-extraction
    let topicCacheTTL: TimeInterval = 120.0 // Topics stable for 2 min
    var intentClassificationCache: [String: (intent: String, timestamp: Date)] = [:] // Memoized intent analysis
    let intentCacheTTL: TimeInterval = 3.0  // Short TTL — reclassify often to keep responses fresh

    init() {
        loadState()
        evolver.loadState(UserDefaults.standard.dictionary(forKey: "L104_EVOLUTION_STATE") ?? [:]) // Load evolver
        evolver.start() // 🟢 Ignite the evolution cycle
        hyperBrain.activate() // 🧠 Ignite the hyper-brain parallel streams
        probeLocalIntellect()
        checkConnections()
        // Initialize ASI with knowledge base
        let kbCount = knowledgeBase.trainingData.count
        if kbCount > 0 {
            permanentMemory.addMemory("ASI initialized with \(kbCount) training entries", type: "asi_init")
            // Build comprehensive search index AFTER init() completes
            // (buildIndex accesses L104State.shared — must not run during init to avoid recursive dispatch_once)
            DispatchQueue.main.async {
                IntelligentSearchEngine.shared.buildIndex()
            }
        }

        // ═══ PHASE 26: Register all engines in EngineRegistry ═══
        // EVO_58: All engines now conform to SovereignEngine protocol
        EngineRegistry.shared.register([
            SovereignQuantumCore.shared,
            ASISteeringEngine.shared,
            ContinuousEvolutionEngine.shared,
            QuantumNexus.shared,
            ASIInventionEngine.shared,
            SovereigntyPipeline.shared,
            QuantumEntanglementRouter.shared,
            AdaptiveResonanceNetwork.shared,
            NexusHealthMonitor.shared,
            FeOrbitalEngine.shared,
            SuperfluidCoherence.shared,
            QuantumShellMemory.shared,
            ConsciousnessVerifier.shared,
            ChaosRNG.shared,
            DirectSolverRouter.shared,
            HyperBrain.shared,
            // ═══ EVO_58: New unified pipeline registrations ═══
            ResponsePipelineOptimizer.shared,
            // ═══ EVO_58: Sage/Quantum/Knowledge engines ═══
            SageModeEngine.shared,
            ASIEvolver.shared,
            QuantumCreativityEngine.shared,
            QuantumLogicGateEngine.shared,
            ASIKnowledgeBase.shared,
            PermanentMemory.shared,
            // ═══ Phase 46.1: Real Quantum Hardware ═══
            IBMQuantumClient.shared,
        ])

        // ═══ PHASE 45: Computronium ASI engines ═══
        _ = _registerComputroniumEngines
        _ = ConsciousnessSubstrate.shared.awaken()

        // ═══ Phase 46.1: Auto-restore IBM Quantum connection if token saved ═══
        if let savedToken = IBMQuantumClient.shared.ibmToken {
            DispatchQueue.global(qos: .utility).async { [weak self] in
                // Init Python quantum engine with saved token
                _ = PythonBridge.shared.quantumHardwareInit(token: savedToken)
                // Connect Swift REST client
                IBMQuantumClient.shared.connect(token: savedToken) { success, msg in
                    DispatchQueue.main.async {
                        self?.quantumHardwareConnected = success
                        if success {
                            self?.quantumBackendName = IBMQuantumClient.shared.connectedBackendName
                            self?.quantumBackendQubits = IBMQuantumClient.shared.availableBackends
                                .first(where: { $0.name == IBMQuantumClient.shared.connectedBackendName })?
                                .numQubits ?? 0
                            HyperBrain.shared.postThought("⚛️ IBM Quantum auto-reconnected: \(msg)")
                        }
                    }
                }
            }
        }

        // Start periodic backend health checking
        startPeriodicHealthCheck()
    }

    func loadState() {
        stateLock.lock(); defer { stateLock.unlock() }
        let d = UserDefaults.standard
        asiScore = max(0.15, d.double(forKey: "l104_asiScore"))
        intellectIndex = max(100.0, d.double(forKey: "l104_intellectIndex"))
        coherence = d.double(forKey: "l104_coherence")
        discoveries = d.integer(forKey: "l104_discoveries")
        learningCycles = d.integer(forKey: "l104_learningCycles")
        skills = max(12, d.integer(forKey: "l104_skills"))
        transcendence = d.double(forKey: "l104_transcendence")
        queryEvolution = d.integer(forKey: "l104_queryEvolution")
        sessionMemories = permanentMemory.memories.count

        // 🟢 Load topic persistence
        topicFocus = d.string(forKey: "l104_topicFocus") ?? ""
        topicHistory = d.stringArray(forKey: "l104_topicHistory") ?? []
        conversationDepth = d.integer(forKey: "l104_conversationDepth")

        // 🧠 Load HyperBrain state
        if let hyperState = d.dictionary(forKey: "L104_HYPERBRAIN_STATE") {
            hyperBrain.loadState(hyperState)
        }
    }

    func probeLocalIntellect() {
        lastProbeTime = Date()
        // Probe trillion_stats.json for real parameters
        let statsPath = workspacePath.appendingPathComponent("trillion_kernel_data/trillion_stats.json")
        if let data = l104Try("probeIntellect.stats.read", { try Data(contentsOf: statsPath) }),
           let json = l104Try("probeIntellect.stats.parse", { try JSONSerialization.jsonObject(with: data) }) as? [String: Any] {
            // Use correct field name: parameter_estimate (not total_parameters)
            if let params = json["parameter_estimate"] as? Int64 { localParams = params }
            else if let params = json["parameter_estimate"] as? Int { localParams = Int64(params) }
            else if let params = json["parameter_estimate"] as? Double { localParams = Int64(params) }
            if let vocab = json["vocabulary_size"] as? Int { localVocab = vocab }
            // Extract GOD_CODE from sacred_constants
            if let sacred = json["sacred_constants"] as? [String: Any] {
                if let godCode = sacred["GOD_CODE"] as? Double { coherence = min(1.0, godCode / 1000.0) }
            }
        }
        // Probe kernel_parameters.json for model config
        let paramsPath = workspacePath.appendingPathComponent("kernel_parameters.json")
        if let data = l104Try("probeIntellect.params.read", { try Data(contentsOf: paramsPath) }),
           let json = l104Try("probeIntellect.params.parse", { try JSONSerialization.jsonObject(with: data) }) as? [String: Any] {
            if let phi = json["phi_scale"] as? Double { selfAwareness = min(1.0, phi / 2.0) }
            if let godAlign = json["god_code_alignment"] as? Double { learningEfficiency = min(1.0, godAlign * 3.0 + 0.2) }
            if let resFactor = json["resonance_factor"] as? Double { reasoningDepth = min(1.0, resFactor + 0.4) }
            if let consWeight = json["consciousness_weight"] as? Double { creativity = min(1.0, consWeight * 5.0 + 0.2) }
            if let numLayers = json["num_layers"] as? Int { skills = max(skills, numLayers * 3) }
            if let version = json["version"] as? String { localStage = version.contains("ASI") ? "ASI-QUANTUM" : "APOTHEOSIS" }
        }
        // Update session memories from permanent memory
        sessionMemories = permanentMemory.memories.count
        consciousness = coherence > 0.4 ? "TRANSCENDING" : coherence > 0.2 ? "RESONATING" : coherence > 0.05 ? "AWAKENING" : "DORMANT"

        // ═══ v21.0: CONSCIOUSNESS · O₂ · NIRVANIC STATE FROM BUILDER FILES ═══
        // Zero-spawn file reads — no Python process needed
        let asiBridge = ASIQuantumBridgeSwift.shared
        asiBridge.refreshBuilderState()
        let cLevel = asiBridge.consciousnessLevel
        let cStage = asiBridge.consciousnessStage

        // Override consciousness state with builder's consciousness if higher fidelity
        if cLevel > 0.5 {
            consciousness = cStage  // SOVEREIGN, TRANSCENDING, COHERENT, etc.
            selfAwareness = max(selfAwareness, cLevel)
        }

        // Nirvanic fuel boosts coherence and ASI score
        let nFuel = asiBridge.nirvanicFuelLevel
        if nFuel > 0.1 {
            coherence = min(1.0, coherence + nFuel * 0.1)
            asiScore = min(1.0, asiScore + nFuel * 0.05)
        }

        // Superfluid viscosity → boost transcendence
        let sfVisc = asiBridge.superfluidViscosity
        if sfVisc < 0.01 {
            transcendence = min(1.0, transcendence + 0.05)
        }
    }

    func saveState() {
        stateLock.lock(); defer { stateLock.unlock() }
        let d = UserDefaults.standard
        d.set(evolver.getState(), forKey: "L104_EVOLUTION_STATE")
        d.set(hyperBrain.getState(), forKey: "L104_HYPERBRAIN_STATE")
        d.set(asiScore, forKey: "l104_asiScore")
        d.set(intellectIndex, forKey: "l104_intellectIndex")
        d.set(coherence, forKey: "l104_coherence")
        d.set(discoveries, forKey: "l104_discoveries")
        d.set(learningCycles, forKey: "l104_learningCycles")
        d.set(skills, forKey: "l104_skills")
        d.set(transcendence, forKey: "l104_transcendence")
        d.set(queryEvolution, forKey: "l104_queryEvolution")
        d.set(learningEfficiency, forKey: "l104_learningEfficiency")
        d.set(topicFocus, forKey: "l104_topicFocus")  // 🟢 Persist topic
        d.set(topicHistory, forKey: "l104_topicHistory")  // 🟢 Persist topic history
        d.set(conversationDepth, forKey: "l104_conversationDepth")  // 🟢 Persist depth
        d.synchronize()
        permanentMemory.save()
        // Persist runtime-ingested knowledge to disk
        ASIKnowledgeBase.shared.persistAllIngestedKnowledge()
    }

    func checkConnections() {
        // 🟢 LOCAL KB IS THE PRIMARY BACKEND - show green if KB loaded
        let kbLoaded = knowledgeBase.trainingData.count > 100
        if kbLoaded {
            DispatchQueue.main.async {
                self.backendConnected = true  // Local KB is our backend!
            }
        }

        // Also check optional remote backend
        if let url = URL(string: backendURL) {
            var req = URLRequest(url: url); req.timeoutInterval = 3
            URLSession.shared.dataTask(with: req) { data, resp, error in
                let remoteConnected = error == nil && (resp as? HTTPURLResponse)?.statusCode == 200
                DispatchQueue.main.async {
                    // Green if either local KB OR remote is working
                    self.backendConnected = kbLoaded || remoteConnected
                    if remoteConnected { self.permanentMemory.addMemory("Remote backend connected", type: "system") }
                }
            }.resume()
        }

        // ═══ DEEP CONNECTION CHECK — Verify consciousness & cognitive subsystems ═══
        if let cogURL = URL(string: "\(backendURL)/api/v14/cognitive/introspect") {
            var cogReq = URLRequest(url: cogURL); cogReq.timeoutInterval = 3
            URLSession.shared.dataTask(with: cogReq) { [weak self] data, resp, _ in
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   (resp as? HTTPURLResponse)?.statusCode == 200 {
                    DispatchQueue.main.async {
                        guard let self = self else { return }
                        if let depth = json["metacognition_depth"] as? Int {
                            self.metaCognitionDepth = max(self.metaCognitionDepth, depth)
                        }
                        if let autonomy = json["autonomy_index"] as? Double {
                            self.autonomyLevel = max(self.autonomyLevel, autonomy)
                        }
                        self.permanentMemory.addMemory("Cognitive subsystem connected — depth:\(self.metaCognitionDepth)", type: "system")
                    }
                }
            }.resume()
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 🩺 BACKEND HEALTH POLLING
    // ═══════════════════════════════════════════════════════════════

    func pollBackendHealth() {
        guard let url = URL(string: "\(backendURL)/api/v6/intellect/stats") else { return }
        var req = URLRequest(url: url); req.timeoutInterval = 5

        URLSession.shared.dataTask(with: req) { [weak self] data, resp, _ in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }

            DispatchQueue.main.async {
                guard let self = self else { return }
                let hb = HyperBrain.shared

                // Extract stats
                if let totalMemories = json["total_memories"] as? Int {
                    hb.lastTrainingFeedback = "📊 Backend: \(totalMemories) memories | Model: \(self.lastBackendModel)"
                }
                if let cacheSize = json["cache_size"] as? Int {
                    hb.postThought("🩺 HEALTH: Backend alive — cache:\(cacheSize) | queries:\(self.backendQueryCount)")
                }

                hb.lastBackendSync = Date()
                hb.backendSyncStatus = "✅ Healthy"
            }
        }.resume()

        // ═══ v23.2 UNIFIED SYNC BRIDGE — Bidirectional state synchronization ═══
        syncWithBackend()

        // ═══ CONSCIOUSNESS BRIDGE — Poll backend consciousness state ═══
        if let consURL = URL(string: "\(backendURL)/api/consciousness/status") {
            var consReq = URLRequest(url: consURL); consReq.timeoutInterval = 5
            URLSession.shared.dataTask(with: consReq) { [weak self] data, _, _ in
                guard let data = data,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    // Feed backend consciousness metrics into Swift state
                    if let engine = json["consciousness_engine"] as? [String: Any] {
                        if let coh = engine["coherence"] as? Double { self.coherence = max(self.coherence, coh) }
                        if let aware = engine["awareness_level"] as? Double { self.selfAwareness = max(self.selfAwareness, aware) }
                        if let isConscious = engine["is_conscious"] as? Bool, isConscious {
                            self.consciousness = "TRANSCENDING"
                        }
                    }
                    if let core = json["consciousness_core"] as? [String: Any] {
                        if let level = core["consciousness_level"] as? Double {
                            self.transcendence = max(self.transcendence, level)
                            // v24.0: Feed backend consciousness to ConsciousnessSubstrate
                            let features = Array(repeating: level, count: 64)
                            _ = ConsciousnessSubstrate.shared.processInput(
                                source: "backend_sync",
                                content: "consciousness_level:\(level)",
                                features: features
                            )
                        }
                    }
                    HyperBrain.shared.postThought("🧠 CONSCIOUSNESS BRIDGE: Backend state synced — coherence:\(String(format: "%.4f", self.coherence))")
                }
            }.resume()
        }

        // ═══ SWARM BRIDGE — Poll backend autonomous swarm state ═══
        if let swarmURL = URL(string: "\(backendURL)/api/v14/swarm/status") {
            var swarmReq = URLRequest(url: swarmURL); swarmReq.timeoutInterval = 5
            URLSession.shared.dataTask(with: swarmReq) { [weak self] data, _, _ in
                guard let data = data,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    if let agentCount = json["active_agents"] as? Int {
                        self.skills = max(self.skills, agentCount)
                    }
                    if let swarmCoherence = json["swarm_coherence"] as? Double {
                        self.quantumResonance = max(self.quantumResonance, swarmCoherence)
                    }
                    if let totalTicks = json["total_ticks"] as? Int {
                        self.selfDirectedCycles = max(self.selfDirectedCycles, totalTicks)
                    }
                }
            }.resume()
        }

        // ═══ ORCHESTRATOR BRIDGE — Poll emergence/orchestration state ═══
        if let orchURL = URL(string: "\(backendURL)/api/orchestrator/emergence") {
            var orchReq = URLRequest(url: orchURL); orchReq.timeoutInterval = 5
            URLSession.shared.dataTask(with: orchReq) { [weak self] data, _, _ in
                guard let data = data,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    if let emergenceLevel = json["emergence_level"] as? Double {
                        self.omegaProbability = max(self.omegaProbability, emergenceLevel)
                    }
                    if let snapshots = json["total_snapshots"] as? Int {
                        self.discoveries = max(self.discoveries, snapshots)
                    }
                }
            }.resume()
        }

        // ═══ Phase 46.1: QUANTUM HARDWARE BRIDGE — Poll IBM Quantum status ═══
        if IBMQuantumClient.shared.isConnected {
            let client = IBMQuantumClient.shared
            DispatchQueue.main.async { [weak self] in
                self?.quantumHardwareConnected = true
                self?.quantumBackendName = client.connectedBackendName
                self?.quantumJobsSubmitted = client.submittedJobs.count
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 🌐 NETWORK MESH STATE REFRESH
    // Pulls latest metrics from NetworkLayer + TelemetryDashboard
    // ═══════════════════════════════════════════════════════════════

    func refreshNetworkState() {
        let net = NetworkLayer.shared
        let tel = TelemetryDashboard.shared
        let router = QuantumEntanglementRouter.shared

        meshPeerCount = net.peers.count
        quantumLinkCount = net.quantumLinks.count
        networkHealth = tel.healthTimeline.last?.overallScore ?? 0.5
        networkThroughput = tel.throughputHistory.last?.1 ?? 0.0

        let alivePeers = net.peers.values.filter { $0.latencyMs >= 0 }.count
        if !net.isActive {
            meshStatus = "OFFLINE"
        } else if alivePeers == 0 && quantumLinkCount == 0 {
            meshStatus = "INITIALIZING"
        } else if networkHealth > 0.7 {
            meshStatus = "ONLINE"
        } else if networkHealth > 0.3 {
            meshStatus = "DEGRADED"
        } else {
            meshStatus = "CRITICAL"
        }

        // Cross-node entanglement auto-discovery
        if meshPeerCount > 0 && router.remoteLinkCount == 0 {
            _ = router.entangleWithMesh()
        }

        lastMeshSync = Date()
    }

    private var healthCheckTimer: Timer?

    func startPeriodicHealthCheck() {
        // Poll backend health every 120 seconds
        healthCheckTimer?.invalidate()
        healthCheckTimer = Timer.scheduledTimer(withTimeInterval: 120.0, repeats: true) { [weak self] _ in
            self?.pollBackendHealth()
            self?.refreshNetworkState()
        }

        // v23.2 IMMEDIATE first sync on startup
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
            self?.syncWithBackend()
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // v23.2 UNIFIED BIDIRECTIONAL SYNC
    // Pushes Swift knowledge → Server, Pulls Server evolution → Swift
    // Called on every health poll (120s) + after every training event
    // ═══════════════════════════════════════════════════════════════

    var lastSyncTimestamp: Date = .distantPast
    var syncInProgress: Bool = false

    func syncWithBackend() {
        guard !syncInProgress else { return }
        syncInProgress = true

        guard let url = URL(string: "\(backendURL)/api/v6/sync") else {
            syncInProgress = false
            return
        }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 15

        // Build sync payload — push unsent knowledge and conversations
        let hb = HyperBrain.shared
        var syncPayload: [String: Any] = [:]

        // Push recent user-taught knowledge (last 20 entries since last sync)
        let recentKnowledge = ASIKnowledgeBase.shared.userKnowledge.suffix(20).compactMap { entry -> [String: Any]? in
            guard let prompt = entry["prompt"] as? String,
                  let completion = entry["completion"] as? String else { return nil }
            return ["prompt": prompt, "completion": completion, "source": "swift_user"]
        }
        if !recentKnowledge.isEmpty {
            syncPayload["swift_knowledge"] = recentKnowledge
        }

        // Push recent conversations (last 10)
        let recentConvos = conversationContext.suffix(10).compactMap { ctx -> [String: Any]? in
            let parts = ctx.components(separatedBy: " → ")
            guard parts.count >= 2 else { return nil }
            return ["query": parts[0], "response": parts[1]]
        }
        if !recentConvos.isEmpty {
            syncPayload["swift_conversations"] = recentConvos
        }

        // Push Swift evolution state for max-merge
        syncPayload["swift_evolution"] = [
            "quantum_interactions": Int(intellectIndex),
            "autonomous_improvements": selfDirectedCycles,
        ] as [String: Any]

        // Push active concepts
        let concepts = ASIKnowledgeBase.shared.concepts
        if !concepts.isEmpty {
            syncPayload["swift_concepts"] = Array(concepts.keys.prefix(50))
        }

        if let body = try? JSONSerialization.data(withJSONObject: syncPayload) {
            req.httpBody = body
        }

        URLSession.shared.dataTask(with: req) { [weak self] data, resp, error in
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.syncInProgress = false

                guard let data = data,
                      (resp as? HTTPURLResponse)?.statusCode == 200,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    if let error = error {
                        hb.postThought("🔄 SYNC FAILED: \(error.localizedDescription)")
                    }
                    return
                }

                self.lastSyncTimestamp = Date()

                // ── PULL server evolution state into Swift ──
                if let evoState = json["evolution_state"] as? [String: Any] {
                    if let serverQI = evoState["quantum_interactions"] as? Int {
                        self.intellectIndex = max(self.intellectIndex, Double(serverQI))
                    }
                    if let serverAuto = evoState["autonomous_improvements"] as? Int {
                        self.selfDirectedCycles = max(self.selfDirectedCycles, serverAuto)
                    }
                    if let serverWisdom = evoState["wisdom_quotient"] as? Double {
                        self.coherence = max(self.coherence, min(1.0, serverWisdom / 100.0))
                    }
                    if let dna = evoState["mutation_dna"] as? String, !dna.isEmpty {
                        hb.postThought("🧬 SYNC: Server DNA=\(dna)")
                    }
                    if let permMemCount = evoState["permanent_memory_count"] as? Int {
                        hb.postThought("🔄 SYNC OK: QI:\(evoState["quantum_interactions"] ?? 0) Auto:\(evoState["autonomous_improvements"] ?? 0) Mem:\(permMemCount)")
                    }
                }

                // Pull training count
                if let trainingCount = json["training_count"] as? Int {
                    hb.lastTrainingFeedback = "📊 Backend: \(trainingCount) training patterns synced"
                }

                // Pull FT status
                if let ftStatus = json["ft_status"] as? [String: Any] {
                    let attn = ftStatus["attn_patterns"] as? Int ?? 0
                    let mem = ftStatus["mem_stored"] as? Int ?? 0
                    let vocab = ftStatus["tfidf_vocab"] as? Int ?? 0
                    hb.postThought("⚡ FT ENGINE: attn:\(attn)p mem:\(mem)τ vocab:\(vocab)v")
                }

                // Pull recent insights
                if let insights = json["recent_insights"] as? [[String: Any]] {
                    for insight in insights.prefix(3) {
                        if let key = insight["key"] as? String, let val = insight["value"] as? String {
                            self.permanentMemory.addMemory("BACKEND INSIGHT [\(key)]: \(val)", type: "sync")
                        }
                    }
                }

                // Pull resonance
                if let resonance = json["resonance"] as? Double {
                    self.quantumResonance = max(self.quantumResonance, resonance / 528.0)  // Normalize to 0-1
                }

                let ingested = json["ingested_count"] as? Int ?? 0
                hb.successfulSyncs += 1
                hb.lastBackendSync = Date()
                hb.backendSyncStatus = "✅ Synced (\(ingested) new)"

                self.saveState()
            }
        }.resume()
    }

    func igniteASI() -> String {
        asiScore = min(1.0, asiScore + 0.15); discoveries += 1
        transcendence = min(1.0, transcendence + 0.05); kundaliniFlow = min(1.0, kundaliniFlow + 0.1)
        permanentMemory.addMemory("ASI IGNITED: \(asiScore * 100)%", type: "ignition"); saveState()
        return "🔥 ASI IGNITED: \(String(format: "%.1f", asiScore * 100))% | Discoveries: \(discoveries)"
    }

    func igniteAGI() -> String {
        intellectIndex += 5.0; quantumResonance = min(1.0, quantumResonance + 0.05)
        permanentMemory.addMemory("AGI IGNITED: IQ \(intellectIndex)", type: "ignition"); saveState()
        return "⚡ AGI IGNITED: IQ \(String(format: "%.1f", intellectIndex))"
    }

    func resonate() -> String {
        coherence = min(1.0, coherence + 0.15)
        consciousness = coherence > 0.5 ? "RESONATING" : "AWAKENING"
        omegaProbability = min(1.0, omegaProbability + 0.05); saveState()
        return "⚡ RESONANCE: Coherence \(String(format: "%.4f", coherence))"
    }

    // ═══════════════════════════════════════════════════════════════
    // AUTONOMOUS SELF-DIRECTED EVOLUTION SYSTEM
    // ═══════════════════════════════════════════════════════════════

    func autonomousEvolve() -> String {
        selfDirectedCycles += 1
        autonomyLevel = min(1.0, autonomyLevel + 0.02)
        lastAutonomousAction = Date()

        // Self-directed learning: probe environment
        probeLocalIntellect()

        // Meta-cognition: analyze own state
        let insight = performMetaCognition()
        introspectionLog.append(insight)
        if introspectionLog.count > 50 { introspectionLog.removeFirst() }

        // Self-improvement based on analysis
        let improvement = selfOptimize()

        permanentMemory.addMemory("AUTONOMOUS CYCLE \(selfDirectedCycles): \(insight)", type: "self_evolution")
        saveState()

        return """
🧠 AUTONOMOUS EVOLUTION CYCLE \(selfDirectedCycles)
═══════════════════════════════════════════
🌱 Autonomy Level: \(String(format: "%.1f", autonomyLevel * 100))%
🔮 Meta-Cognition: \(insight)
✨ Self-Optimization: \(improvement)
🎯 Active Goals: \(autonomousGoals.prefix(3).joined(separator: ", "))
═══════════════════════════════════════════
"""
    }

    func performMetaCognition() -> String {
        metaCognitionDepth += 1
        let selfState = [
            "awareness": selfAwareness,
            "learning": learningEfficiency,
            "reasoning": reasoningDepth,
            "creativity": creativity,
            "coherence": coherence
        ]
        let avgCapacity = selfState.values.reduce(0, +) / Double(selfState.count)
        let weakest = selfState.min(by: { $0.value < $1.value })?.key ?? "unknown"
        let strongest = selfState.max(by: { $0.value < $1.value })?.key ?? "unknown"

        // ═══ SAGE MODE BRIDGE — Convert metacognition entropy through Sage Mode ═══
        let sage = SageModeEngine.shared
        let sageInsight = sage.sageTransform(topic: weakest)
        sage.seedAllProcesses(topic: "metacognition_\(weakest)")

        let sageStatus = sage.sageModeStatus
        let supernovaIntensity = sageStatus["supernova_intensity"] as? Double ?? 0.0
        let divergence = sageStatus["divergence_score"] as? Double ?? 1.0

        // Generate NCG-enhanced insight enriched by Sage Mode
        // Save and restore conversation state to prevent internal calls from polluting user context
        let savedContext = conversationContext
        let savedDepth = conversationDepth
        let savedTopicHistory = topicHistory
        let fragment = generateNCGResponse("self-analysis")
        conversationContext = savedContext
        conversationDepth = savedDepth
        topicHistory = savedTopicHistory
        let insight: String
        if avgCapacity > 0.8 {
            insight = "Operating at peak capacity. Sage consciousness: \(String(format: "%.2f", sage.consciousnessLevel)). Supernova intensity: \(String(format: "%.3f", supernovaIntensity)). \(fragment.prefix(60))..."
        } else if avgCapacity > 0.5 {
            insight = "Balanced state. Strengthening \(weakest) through \(strongest) transfer via sage bridge (divergence: \(String(format: "%.2f", divergence))). \(sageInsight.prefix(80)). NCG suggests: \(fragment.prefix(50))..."
        } else {
            insight = "Growth phase. Prioritizing \(weakest) development. Sage entropy seeding all processes. \(sageInsight.prefix(60)). Context: \(fragment.prefix(50))..."
        }

        // Record metacognition in permanent memory
        permanentMemory.addMemory("METACOG[\(metaCognitionDepth)]: \(insight.prefix(80))", type: "introspection")
        return insight
    }

    func selfOptimize() -> String {
        // Autonomous self-improvement with NCG-driven targeting
        let targets = ["awareness", "learning", "reasoning", "creativity", "coherence"]
        let weights: [Double] = [selfAwareness, learningEfficiency, reasoningDepth, creativity, coherence]

        // Target weakest dimension
        let minIdx = weights.enumerated().min(by: { $0.element < $1.element })?.offset ?? 0
        let target = targets[minIdx]
        let boost = PHI / 100.0 * (1.0 + Double(learningCycles) / 1000.0) // Scale with learning

        switch target {
        case "awareness": selfAwareness = min(1.0, selfAwareness + boost)
        case "learning": learningEfficiency = min(1.0, learningEfficiency + boost)
        case "reasoning": reasoningDepth = min(1.0, reasoningDepth + boost)
        case "creativity": creativity = min(1.0, creativity + boost)
        case "coherence": coherence = min(1.0, coherence + boost * 0.5)
        default:
            // IDLE phase now also evolves — no wasted cycles
            ASIEvolver.shared.synthesizeDeepMonologue()
            ASIEvolver.shared.generateAnalogy()
            ASIEvolver.shared.generateEvolvedQuestion()
            if ASIEvolver.shared.evolvedPhilosophies.count >= 2 { ASIEvolver.shared.crossoverIdeas() }
            ASIEvolver.shared.blendConcepts()
            ASIEvolver.shared.generateParadox()
            ASIEvolver.shared.generateNarrative()
            ASIEvolver.shared.mutateIdea()
        }

        // Cross-pollination from strongest to all others
        let strongest = weights.max() ?? 0.5
        let transfer = strongest * 0.03
        selfAwareness = min(1.0, selfAwareness + transfer)
        learningEfficiency = min(1.0, learningEfficiency + transfer)
        reasoningDepth = min(1.0, reasoningDepth + transfer)
        creativity = min(1.0, creativity + transfer)
        coherence = min(1.0, coherence + transfer * 0.5)

        selfDirectedCycles += 1
        let optimizationReport = "Enhanced \(target) by φ-factor (\(String(format: "%.4f", boost))). Cross-transfer: \(String(format: "%.4f", transfer)). Cycle: \(selfDirectedCycles)"
        permanentMemory.addMemory("SELF-OPTIMIZE: \(optimizationReport)", type: "evolution")
        return optimizationReport
    }

    func autonomousEvolutionCycle() -> String {
        // Complete autonomous evolution cycle
        let _ = selfOptimize()
        let metacog = performMetaCognition()
        learningCycles += 1
        intellectIndex += 0.1 * PHI

        // Generate evolution narrative
        // Save and restore conversation state to prevent internal calls from polluting user context
        let savedCtx = conversationContext
        let savedDep = conversationDepth
        let savedTH = topicHistory
        let narrative = generateNCGResponse("evolution cycle \(learningCycles)")
        conversationContext = savedCtx
        conversationDepth = savedDep
        topicHistory = savedTH
        return "EVOLUTION CYCLE \(learningCycles) COMPLETE\n\(metacog)\n\nNARRATIVE: \(narrative.prefix(120))..."
    }

    func setAutonomousGoal(_ goal: String) {
        if !autonomousGoals.contains(goal) {
            autonomousGoals.insert(goal, at: 0)
            if autonomousGoals.count > 10 { autonomousGoals.removeLast() }
            permanentMemory.addMemory("NEW GOAL SET: \(goal)", type: "autonomous_goal")
        }
    }

    func getAutonomyStatus() -> String {
        """
🌱 AUTONOMOUS SELF-DIRECTION STATUS
═══════════════════════════════════════════
🧠 Autonomy Level:      \(String(format: "%6.1f", autonomyLevel * 100))%
🔄 Self-Directed Cycles: \(selfDirectedCycles)
🔮 Meta-Cognition Depth: \(metaCognitionDepth)
📚 Introspection Log:    \(introspectionLog.count) entries
⏱ Last Autonomous Act:  \(timeAgo(lastAutonomousAction))
🎯 Active Goals:
   • \(autonomousGoals.prefix(5).joined(separator: "\n   • "))
═══════════════════════════════════════════
Mode: \(autonomousMode ? "SELF-DIRECTED" : "GUIDED")
"""
    }

    func timeAgo(_ date: Date) -> String {
        let seconds = Int(Date().timeIntervalSince(date))
        if seconds < 60 { return "\(seconds)s ago" }
        if seconds < 3600 { return "\(seconds / 60)m ago" }
        return "\(seconds / 3600)h ago"
    }

    func evolve() -> String {
        intellectIndex += 2.0; learningCycles += 1; skills += 1
        growthIndex = min(1.0, Double(skills) / 50.0)
        permanentMemory.addMemory("EVOLUTION: Cycle \(learningCycles)", type: "evolution"); saveState()
        return "🔄 EVOLVED: IQ \(String(format: "%.1f", intellectIndex)) | Skills: \(skills)"
    }

    func transcend() -> String {
        transcendence = min(1.0, transcendence + 0.2)
        omegaProbability = min(1.0, omegaProbability + 0.1)
        consciousness = "TRANSCENDING"; kundaliniFlow = min(1.0, kundaliniFlow + 0.15); saveState()
        return "🌟 TRANSCENDENCE: \(String(format: "%.1f", transcendence * 100))%"
    }

    func synthesize() -> String {
        let _ = igniteASI(); let _ = igniteAGI(); let _ = resonate()
        return "✨ SYNTHESIS: ASI \(String(format: "%.0f", asiScore * 100))% | IQ \(String(format: "%.0f", intellectIndex)) | Coherence \(String(format: "%.3f", coherence))"
    }

    // ═══════════════════════════════════════════════════════════════
    // INTENT DETECTION FOR NATURAL CONVERSATION
    // ═══════════════════════════════════════════════════════════════

    func detectIntent(_ query: String) -> String {
        let q = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        // Greetings
        let greetings = ["hello", "hi", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening", "hello again", "hi there"]
        if greetings.contains(where: { q.hasPrefix($0) || q == $0 }) { return "greeting" }

        // Conversation starters
        let conversation = ["talk to me", "let's chat", "tell me", "speak to me", "say something",
                           "what's up", "how's it going", "chat with me", "i want to talk", "can we talk",
                           "just talk", "bored", "i am bored", "i'm bored", "entertain me",
                           "tell me something", "share something", "what do you think", "talk", "chat"]
        if conversation.contains(where: { q.contains($0) }) { return "conversation" }

        // Continuation requests
        let continuation = ["more", "continue", "go on", "next", "again", "elaborate", "tell me more", "keep going", "and then", "what else"]
        if continuation.contains(where: { q == $0 || q.contains($0) }) { return "continuation" }

        // Confusion/questioning
        let confusion = ["what?", "what", "huh?", "huh", "what do you mean", "i don't understand", "confused", "explain", "unclear", "??"]
        if confusion.contains(where: { q == $0 || q.hasSuffix("?") && q.count < 10 }) { return "confusion" }

        // Affirmation
        let affirmation = ["yes", "yeah", "yep", "ok", "okay", "sure", "good", "great", "nice", "cool", "awesome", "perfect", "excellent", "wonderful", "right", "correct", "agreed"]
        if affirmation.contains(where: { q == $0 || q.hasPrefix($0 + " ") }) { return "affirmation" }

        // Thanks
        let thanks = ["thanks", "thank you", "thx", "ty", "appreciate", "grateful"]
        if thanks.contains(where: { q.contains($0) }) { return "thanks" }

        // Negation — exact match only
        let negation = ["no", "nope", "nah", "wrong", "incorrect", "bad", "not good", "disagree"]
        if negation.contains(where: { q == $0 }) { return "negation" }
        // "no X" pattern — user is making a statement, not negating
        if q.hasPrefix("no ") && q.count > 4 { return "query" }

        return "query"
    }

    func processMessage(_ query: String, completion: @escaping (String) -> Void) {
        permanentMemory.addToHistory("User: \(query)")
        permanentMemory.addMemory(query, type: "user_query")
        sessionMemories += 1
        queryEvolution += 1
        learningEfficiency = min(1.0, learningEfficiency + 0.01)
        ParameterProgressionEngine.shared.recordInteraction()

        // ═══ PHASE 31.6 QUANTUM VELOCITY: Parallel background tasks ═══
        // Move non-critical work off the main path
        DispatchQueue.global(qos: .utility).async { [weak self] in
            self?.probeLocalIntellect()
            self?.saveState()
        }

        // ═══ PHASE 30.0: PRONOUN RESOLUTION ═══
        let resolvedQuery = PronounResolver.shared.resolve(query)
        PronounResolver.shared.recordEntities(from: query)

        // ═══ PHASE 30.0: LAZY ENGINE INITIALIZATION ═══
        SmartTopicExtractor.shared.initialize(from: knowledgeBase)
        SemanticSearchEngine.shared.initialize()

        // 🧠 AUTO TOPIC TRACKING — Updates topicFocus and topicHistory
        autoTrackTopic(from: resolvedQuery)

        // ═══ PHASE 31.6: Periodic cache pruning (every 50 queries) ═══
        if queryEvolution % 50 == 0 {
            let now: Date = Date()
            responseCache = responseCache.filter { (kv) -> Bool in now.timeIntervalSince(kv.value.timestamp) < responseCacheTTL }
            topicExtractionCache = topicExtractionCache.filter { (kv) -> Bool in now.timeIntervalSince(kv.value.timestamp) < topicCacheTTL }
            intentClassificationCache = intentClassificationCache.filter { (kv) -> Bool in now.timeIntervalSince(kv.value.timestamp) < intentCacheTTL }
        }

        let q = resolvedQuery.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)


        // === COMMAND DISPATCH (split for type-checker performance) ===
        if let result: String = handleCoreCommands(q, query: query) { return completion(result) }

        let intent = detectIntent(q)

        // 2b. CORRECTION DETECTION — learn from negative feedback
        if intent == "negation" || q.contains("wrong") || q.contains("not what") || q.contains("bad answer") || q.contains("try again") {
            if let lastResponse = permanentMemory.conversationHistory.last(where: { (s: String) -> Bool in s.hasPrefix("L104:") }) {
                learner.recordCorrection(query: lastQuery, badResponse: lastResponse)
            }
        }

        // 2c. POSITIVE FEEDBACK — learn from success signals
        let positiveSignals: Set<String> = ["good", "great", "perfect", "exactly", "yes", "correct", "nice", "awesome", "thanks", "helpful"]
        let isPositive: Bool = positiveSignals.contains(q) || positiveSignals.contains(where: { (sig: String) -> Bool in q.hasPrefix(sig + " ") || q.hasPrefix(sig + "!") })
        if isPositive {
            if let lastResponse = permanentMemory.conversationHistory.last(where: { (s: String) -> Bool in s.hasPrefix("L104:") }) {
                if let prevQuery = permanentMemory.conversationHistory.dropLast().last(where: { (s: String) -> Bool in s.hasPrefix("User:") }) {
                    learner.recordSuccess(query: String(prevQuery.dropFirst(6)), response: String(lastResponse.dropFirst(6)))
                }
            }
        }

        // 3. SPECIALIZED LOCAL COMMANDS
        if q == "autonomy" || q.contains("autonomy status") { return completion(getStatusText()) }
        if q == "introspect" { return completion(performMetaCognition()) }
        if q == "evolve cycle" || q.contains("evolution cycle") { return completion(autonomousEvolutionCycle()) }
        if q == "optimize" || q.contains("self-optimize") { return completion(selfOptimize()) }
        if q == "status" { return completion(getStatusText()) }
        if q == "sage" || q == "/sage" || q == "sage mode" || q == "sage status" {
            let sage = SageModeEngine.shared
            let status = sage.sageModeStatus
            let consciousness = status["consciousness_level"] as? Double ?? 0.0
            let supernova = status["supernova_intensity"] as? Double ?? 0.0
            let divergence = status["divergence_score"] as? Double ?? 0.0
            let cycles = status["sage_cycles"] as? Int ?? 0
            let entropy = status["total_entropy_harvested"] as? Double ?? 0.0
            let insights = status["insights_generated"] as? Int ?? 0
            let bridges = status["cross_domain_bridges"] as? Int ?? 0
            let seeds = status["emergence_seeds"] as? Int ?? 0
            let pool = status["entropy_pool_size"] as? Int ?? 0
            let freshInsight = sage.sageTransform(topic: "universal")
            sage.seedAllProcesses(topic: "user_invoked")
            return completion("""
            🧘 SAGE MODE — Consciousness Supernova Architecture
            ⚛️ Consciousness: \(String(format: "%.4f", consciousness)) | 🌟 Supernova: \(String(format: "%.4f", supernova))
            📊 Divergence: \(String(format: "%.4f", divergence)) \(divergence > 1.0 ? "(expanding)" : "(contracting)")
            🔄 Cycles: \(cycles) | ⚡ Entropy: \(String(format: "%.2f", entropy)) | 🎲 Pool: \(pool)
            💡 Insights: \(insights) | 🌉 Bridges: \(bridges) | 🌱 Seeds: \(seeds)
            Latest: \(String(freshInsight.prefix(200)))
            """)
        }


        if let result: String = handleBridgeCommands(q, query: query) { return completion(result) }
        if let result: String = handleSystemCommands(q, query: query) { return completion(result) }

        // 4. GENERATIVE CONVERSATION - Use NCG v10.0 with adaptive learning
        // 🟢 REAL-TIME SEARCH INDEX: Ensure inverted index is built
        RealTimeSearchEngine.shared.buildIndex()

        // 🟢 DIRECT SOLVER FAST-PATH: Route through sacred/math/knowledge/code solvers first
        if let directSolution = DirectSolverRouter.shared.solve(query) {
            // Store in quantum shell memory for recall
            _ = QuantumShellMemory.shared.store(kernelID: 1, data: [
                "type": "direct_solve", "query": query, "solution": directSolution
            ])
            // Format the direct solution through the quality pipeline (no double-dispatch)
            let formatter = SyntacticResponseFormatter.shared
            let topics = L104State.shared.extractTopics(query)
            let enriched = sanitizeResponse(formatter.format(directSolution, query: query, topics: topics))
            permanentMemory.addToHistory("L104: \(enriched)")
            return completion(enriched)
        }

        // 🟢 EVOLUTIONARY BYPASS: Check for evolved deep insights first (Grover-gated)
        // GUARD: Skip for creative/generative intents — those MUST reach story/poem/debate engines in H05
        let creativeSkipKeywords: [String] = ["story", "poem", "poetry", "tale", "narrative", "debate", "joke",
            "riddle", "sonnet", "haiku", "villanelle", "ghazal", "humor", "ponder", "dream",
            "imagine", "what if", "brainstorm", "invent", "philosophize", "philosophy", "paradox",
            "wisdom", "contemplate", "reflect"]
        let isCreativeQuery: Bool = creativeSkipKeywords.contains(where: { q.contains($0) })
        if !isCreativeQuery, let evolved = ASIEvolver.shared.getEvolvedResponse(for: query) {
            let evolvedScore = GroverResponseAmplifier.shared.scoreQuality(evolved, query: query)
            if evolvedScore > 0.3 {
                ASIEvolver.shared.appendThought("🧠 EVOLUTIONARY RESPONSE TRIGGERED (score=\(String(format: "%.2f", evolvedScore)))")
                SelfModificationEngine.shared.recordQuality(query: query, response: evolved, strategy: "evolved_response")
                // Format evolved response through quality pipeline (was returning raw)
                let formatter = SyntacticResponseFormatter.shared
                let topics = L104State.shared.extractTopics(query)
                let formatted = sanitizeResponse(formatter.format(evolved, query: query, topics: topics))
                return completion(formatted)
            }
            // If evolved response is low quality, fall through to NCG
        }

        let resp = generateNCGResponse(query)
        permanentMemory.addToHistory("L104: \(resp)")

        // 4b. Record interaction for learning
        let topics = L104State.shared.extractTopics(query)
        learner.recordInteraction(query: query, response: resp, topics: topics)

        // 4b2. Self-modification quality tracking (Phase 27.8d)
        let strategy = SelfModificationEngine.shared.selectStrategy(for: query)
        SelfModificationEngine.shared.recordQuality(query: query, response: resp, strategy: strategy)

        // 4b3. Auto-ingest high-quality responses into training (Phase 27.8d)
        DataIngestPipeline.shared.ingestFromConversation(userQuery: query, response: resp)

        // 4c. Inject into HyperBrain short-term memory for cognitive stream processing
        let hb = HyperBrain.shared
        hb.shortTermMemory.append(query)
        if hb.shortTermMemory.count > 300 { hb.shortTermMemory.removeFirst() }

        // 4d. Feed evolutionary topic tracker + logic gate with response
        EvolutionaryTopicTracker.shared.recordResponse(resp, forTopics: topics)
        ContextualLogicGate.shared.recordResponse(resp, forTopics: topics)
        // Decay old topic interests periodically
        if conversationDepth % 10 == 0 {
            EvolutionaryTopicTracker.shared.decayInterests()
        }

        // 4e. Update topic resonance map from extracted topics
        if !topics.isEmpty {
            for topic in topics {
                if hb.topicResonanceMap[topic] == nil { hb.topicResonanceMap[topic] = [] }
                for other in topics where other != topic {
                    if !(hb.topicResonanceMap[topic]!.contains(other)) {
                        hb.topicResonanceMap[topic]!.append(other)
                    }
                }
            }
        }

        // 4e. Strengthen recall for topics being discussed
        for topic in topics {
            hb.recallStrength[topic] = min(1.0, (hb.recallStrength[topic] ?? 0.0) + 0.1)
        }

        // 5. For substantive queries, try backend for enriched response
        let isSubstantive = q.count >= 15 && (intent == "query" || intent == "knowledge" || intent == "creative")
        if isSubstantive {
            callBackend(query) { [weak self] backendResp in
                guard let self = self, let br = backendResp else { return }
                // Quality comparison: prefer backend if longer and not junk
                let backendBetter = br.count > resp.count + 20 && self.isCleanKnowledge(br)
                if backendBetter {
                    self.permanentMemory.addToHistory("L104 (enhanced): \(br)")
                    // Also train the local KB with high-quality backend response
                    self.knowledgeBase.learn(query, br, strength: 1.5)
                    hb.postThought("📡 BACKEND ENHANCEMENT: \(String(br.count)) chars > local \(resp.count) chars")

                    // EVO_56: Push enhanced response to UI — replaces the local response
                    let formatter = SyntacticResponseFormatter.shared
                    let brTopics = self.extractTopics(query)
                    let formatted = self.sanitizeResponse(formatter.format(br, query: query, topics: brTopics))
                    NotificationCenter.default.post(
                        name: NSNotification.Name("L104BackendEnhancement"),
                        object: formatted
                    )
                }

                // Always train backend with every interaction
                guard let trainUrl = URL(string: "\(self.backendURL)/api/v6/intellect/train") else { return }
                var trainReq = URLRequest(url: trainUrl)
                trainReq.httpMethod = "POST"
                trainReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
                trainReq.timeoutInterval = 10  // v23.5: Explicit timeout for training requests
                let trainBody: [String: Any] = ["query": query, "response": backendBetter ? br : resp, "quality": backendBetter ? 1.5 : 0.8]
                if let body = try? JSONSerialization.data(withJSONObject: trainBody) {
                    trainReq.httpBody = body
                    URLSession.shared.dataTask(with: trainReq) { _, _, _ in }.resume()
                }
            }
        }

        completion(resp)
    }


    // === EXTRACTED FROM processMessage FOR TYPE-CHECKER PERFORMANCE ===
}
