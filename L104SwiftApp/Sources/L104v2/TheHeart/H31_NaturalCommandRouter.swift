import AppKit
import Foundation

// MARK: - Intent

enum CommandIntent {
    case engineStatus(engine: String)
    case engineRun(engine: String, args: String)
    case engineDebug(engine: String)
    case goalSubmit(goal: String)
    case goalChain(goals: [String])
    case conditionalDispatch(condition: String, action: String)
    case cognitiveState
    case serverStatus
    case systemReport
    case knowledgeSearch(query: String)
    case researchQuery(topic: String)
    case intellectQuery(query: String)
    case deepSeekQuery(prompt: String)
    case agentDeploy(task: String)
    case codeRead(target: String)
    case codeModify(target: String, change: String)
    case buildRun
    case sqaVerify(text: String)
    case helpList
    case none
}

// MARK: - NaturalCommandRouter

/// Maps free-text queries to live engine calls.
/// Sits in processMessage() after handleSystemCommands(), before DirectSolverRouter.
/// If it recognises the intent it returns a real engine result string.
/// If not, it returns nil and the normal pipeline continues.
final class NaturalCommandRouter {
    static let shared = NaturalCommandRouter()
    private let modifier = SelfModifier.shared
    private let reader   = CodeReader.shared

    // ── Engine alias table - maps common names to canonical keys ──
    private let engineAliases: [String: String] = [
        // Cognitive loop
        "cognitive loop": "cognitive", "cognitive":    "cognitive", "cl":       "cognitive",
        "cog loop":       "cognitive", "feedback loop":"cognitive",
        // SQA
        "sqa":            "sqa",       "verifier":     "sqa",       "quantum annealing": "sqa",
        "system 2":       "sqa",       "sqa verifier": "sqa",       "annealer": "sqa",
        // Quantum AI Daemon
        "daemon":         "daemon",    "quantum daemon":"daemon",   "qai daemon":"daemon",
        "qaidaemon":      "daemon",    "ai daemon":    "daemon",
        // Autonomous Agent
        "agent":          "agent",     "autonomous agent":"agent",  "task agent":"agent",
        // Hyper Brain
        "hyperbrain":     "hyper",     "hyper brain":  "hyper",     "hyper":    "hyper",
        // Fidelity
        "fidelity":       "fidelity",  "sacred constants":"fidelity","constants":"fidelity",
        // Harmony
        "harmony":        "harmony",   "cross-engine": "harmony",   "harmonizer":"harmony",
        // API Gateway
        "api gateway":    "gateway",   "gateway":      "gateway",   "endpoints":"gateway",
        // Network
        "network":        "network",   "mesh":         "network",   "peers":    "network",
        // Evolution
        "evolution":      "evolution", "evolver":      "evolution", "evo":      "evolution",
        // Server
        "server":         "server",    "fast server":  "server",    "backend":  "server",
        // Performance
        "performance":    "perf",      "profiler":     "perf",      "benchmark":"perf",
        // Telemetry
        "telemetry":      "telemetry", "dashboard":    "telemetry",
        // Security
        "security":       "security",  "vault":        "security",
        // Emotion
        "emotion":        "emotion",   "emotional core":"emotion",  "feelings": "emotion",
        // Memory
        "memory":         "memory",    "permanent memory":"memory",
        // Knowledge / Research
        "knowledge base": "kb",        "kb":           "kb",       "knowledge": "kb",
        "research":       "research",  "researcher":   "research",
        // Intellect / ASI
        "intellect":      "intellect", "local intellect":"intellect",
        "asi":            "asi",        "asi core":     "asi",
        // Sage
        "sage":           "sage",       "sage mode":    "sage",
        // Computronium
        "computronium":   "computronium",
    ]

    // ── Intent detection verbs ──
    private let statusVerbs  = ["status", "show", "display", "what is", "how is", "check",
                                 "report", "info", "information", "state", "health", "tell me about",
                                 "what's", "whats", "describe", "summary", "summarize"]
    private let runVerbs     = ["run", "execute", "start", "trigger", "launch", "activate",
                                 "initiate", "fire", "go", "begin", "kick off", "force"]
    private let debugVerbs   = ["debug", "diagnose", "why is", "fix", "troubleshoot",
                                 "what's wrong", "what is wrong", "investigate", "analyse", "analyze",
                                 "inspect", "test", "self-test", "selftest", "verify"]
    private let modifyVerbs  = ["modify", "edit", "change", "update", "add", "remove", "refactor",
                                 "improve", "rewrite", "patch", "alter", "adjust"]
    private let readVerbs    = ["read", "open", "view", "show me", "display", "print", "cat",
                                 "what does", "explain", "show file", "load"]
    private let goalVerbs    = ["do ", "task:", "agent:", "goal:", "submit goal", "submit task",
                                 "run task", "execute goal", "perform"]
    private let buildVerbs   = ["build", "compile", "swift build", "rebuild", "recompile"]
    private let verifyVerbs  = ["verify", "sqa verify", "check logic", "validate logic",
                                 "quantum verify", "run sqa on", "verify logic of"]
    private let searchVerbs  = ["search", "search kb", "search knowledge", "look up",
                                 "find in kb", "what do you know about", "do you know about",
                                 "knowledge search", "kb search", "query kb"]
    private let researchVerbs = ["research", "deep research", "investigate", "deep dive",
                                  "study", "explore topic", "research topic"]
    private let intellectVerbs = ["ask intellect", "query intellect", "intellect:", "ask asi",
                                   "query asi", "asi:", "ask python", "run intellect",
                                   "intellect query", "local intellect:"]
    private let deepSeekVerbs  = ["ask deepseek", "deepseek:", "ds:", "ask ds",
                                   "query deepseek", "deepseek ask", "send to deepseek",
                                   "deepseek chat", "deepseek reasoner"]
    private let deployVerbs    = ["deploy agent", "deploy task", "deploy", "openclaw",
                                   "api deploy", "agent deploy", "send to mesh"]
    private let reportVerbs    = ["full report", "system report", "all status", "full status",
                                   "system dashboard", "dashboard", "all engines",
                                   "full system", "report", "overview", "summary report"]
    private let helpVerbs      = ["help", "commands", "what can you do", "command list",
                                   "available commands", "how do i", "show commands",
                                   "list commands", "what commands", "capabilities"]

    func route(_ q: String, query: String) -> String? {
        let lower = q.lowercased().trimmingCharacters(in: .whitespaces)
        let intent = detectIntent(lower, original: query)
        switch intent {
        case .none:
            return nil
        case .engineStatus(let e):
            return engineStatus(e)
        case .engineRun(let e, let args):
            return engineRun(e, args: args)
        case .engineDebug(let e):
            return engineDebug(e)
        case .goalSubmit(let goal):
            let id = AutonomousAgent.shared.submitGoal(goal, priority: .normal)
            return "Goal submitted to AutonomousAgent. Task ID: \(id)\nGoal: \(goal)\nUse 'agent status' to track progress."
        case .goalChain(let goals):
            return submitGoalChain(goals)
        case .conditionalDispatch(let condition, let action):
            return handleConditional(condition: condition, action: action)
        case .systemReport:
            return SystemOrchestrator.shared.fullReport()
        case .knowledgeSearch(let query):
            return kbSearch(query)
        case .researchQuery(let topic):
            return researchQuery(topic)
        case .intellectQuery(let query):
            return intellectQuery(query)
        case .deepSeekQuery(let prompt):
            return deepSeekQuery(prompt)
        case .agentDeploy(let task):
            return agentDeploy(task)
        case .helpList:
            return helpText()
        case .cognitiveState:
            return CognitiveLoop.shared.statusReport + "\n\nNext goal: \(CognitiveLoop.shared.buildNextGoal())"
        case .serverStatus:
            return checkServerStatus()
        case .codeRead(let target):
            return reader.read(target)
        case .codeModify(let target, let change):
            return modifier.modify(target: target, change: change)
        case .buildRun:
            return runBuild()
        case .sqaVerify(let text):
            let verdict = SQAVerifier.shared.verify(goal: query, result: text)
            return "SQA Verification Result:\n\(verdict.summary)"
        }
    }

    // MARK: - Intent Detection

    private func detectIntent(_ q: String, original: String) -> CommandIntent {
        // Help
        if helpVerbs.contains(where: { q == $0 || q.hasPrefix($0 + " ") }) {
            return .helpList
        }

        // Build
        if buildVerbs.contains(where: { q.hasPrefix($0) || q == $0 }) {
            return .buildRun
        }

        // System-wide report
        if reportVerbs.contains(where: { q == $0 || q.hasPrefix($0) }) {
            return .systemReport
        }

        // Goal chain - "first X, then Y, then Z" / "do X and then Y"
        if q.hasPrefix("first ") || (q.contains(" then ") && q.contains(",")) {
            let goals = parseGoalChain(original)
            if goals.count > 1 { return .goalChain(goals: goals) }
        }

        // Conditional dispatch - "if X then Y" / "if X is low fix it"
        if q.hasPrefix("if "), let (cond, action) = parseConditional(q) {
            return .conditionalDispatch(condition: cond, action: action)
        }

        // Research engine — checked before KB search to prevent "research" matching "search" substring
        for verb in researchVerbs {
            if q.hasPrefix(verb + " ") || q.hasPrefix(verb + ":") {
                let rest = extractAfter(verb, in: original)
                if !rest.isEmpty { return .researchQuery(topic: rest) }
            }
        }

        // Knowledge base search
        for verb in searchVerbs {
            if q.hasPrefix(verb) || q.contains(verb) {
                let rest = extractAfter(verb, in: original)
                if !rest.isEmpty { return .knowledgeSearch(query: rest) }
            }
        }

        // Intellect / ASI Python query
        for verb in intellectVerbs {
            if q.hasPrefix(verb) {
                let rest = extractAfter(verb, in: original)
                if !rest.isEmpty { return .intellectQuery(query: rest) }
            }
        }

        // DeepSeek chat query
        for verb in deepSeekVerbs {
            if q.hasPrefix(verb) {
                let rest = extractAfter(verb, in: original)
                if !rest.isEmpty { return .deepSeekQuery(prompt: rest) }
            }
        }

        // Agent deploy via API gateway
        for verb in deployVerbs {
            if q.hasPrefix(verb + " ") || q.hasPrefix(verb + ":") {
                let rest = extractAfter(verb, in: original)
                if !rest.isEmpty { return .agentDeploy(task: rest) }
            }
        }

        // Goal submission
        for verb in goalVerbs {
            if q.hasPrefix(verb) {
                let goal = String(original.dropFirst(verb.count)).trimmingCharacters(in: .whitespaces)
                if !goal.isEmpty { return .goalSubmit(goal: goal) }
            }
        }

        // SQA verify
        for verb in verifyVerbs {
            if q.hasPrefix(verb) {
                let text = String(original.dropFirst(verb.count)).trimmingCharacters(in: .whitespaces)
                return .sqaVerify(text: text.isEmpty ? original : text)
            }
        }

        // Cognitive state
        if q.contains("cognitive state") || q.contains("next goal") || q.contains("focus vector") ||
           q.contains("cognitive loop state") || q == "cl state" || q == "loop state" {
            return .cognitiveState
        }

        // Server status
        if q.contains("server status") || q.contains("is the server") || q.contains("server running") ||
           q.contains("backend status") || q.contains("fast server") {
            return .serverStatus
        }

        // Code modify - "modify [target] to [change]" / "edit [file]: [change]"
        for verb in modifyVerbs {
            if q.hasPrefix(verb + " ") {
                let rest = String(original.dropFirst(verb.count + 1))
                if let (target, change) = parseModifyTarget(rest) {
                    return .codeModify(target: target, change: change)
                }
            }
        }

        // Code read - "read B62" / "show me B62_CognitiveLoop"
        for verb in readVerbs {
            if q.hasPrefix(verb + " ") {
                let rest = String(original.dropFirst(verb.count + 1)).trimmingCharacters(in: .whitespaces)
                if looksLikeFile(rest) {
                    return .codeRead(target: rest)
                }
            }
        }

        // Engine-targeted intents - detect verb + engine name
        if let (verb, engine) = extractVerbEngine(q) {
            if debugVerbs.contains(verb)  { return .engineDebug(engine: engine) }
            if runVerbs.contains(verb)    { return .engineRun(engine: engine, args: "") }
            if statusVerbs.contains(verb) { return .engineStatus(engine: engine) }
        }

        // Bare engine name with no verb → status
        if let canonical = resolveEngine(q) {
            return .engineStatus(engine: canonical)
        }

        return .none
    }

    private func extractAfter(_ verb: String, in text: String) -> String {
        let lower = text.lowercased()
        if let range = lower.range(of: verb) {
            let after = String(text[range.upperBound...])
                .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(.init(charactersIn: ":--")))
            return after
        }
        return ""
    }

    private func parseGoalChain(_ text: String) -> [String] {
        // "first X, then Y, then Z" or "do X, then do Y"
        var goals: [String] = []
        let cleaned = text
            .replacingOccurrences(of: "first ", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "then ", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "after that ", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "finally ", with: "", options: .caseInsensitive)
        goals = cleaned.components(separatedBy: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.count > 3 }
        return goals
    }

    private func parseConditional(_ q: String) -> (String, String)? {
        // "if X then Y" / "if X is Y do Z"
        let separators = [" then ", " do ", ": "]
        for sep in separators {
            if let range = q.range(of: sep) {
                let cond   = String(q[q.index(q.startIndex, offsetBy: 3)..<range.lowerBound])
                let action = String(q[range.upperBound...])
                if !cond.isEmpty && !action.isEmpty { return (cond, action) }
            }
        }
        return nil
    }

    private func extractVerbEngine(_ q: String) -> (String, String)? {
        // Try "verb engine" and "engine verb"
        for verb in (statusVerbs + runVerbs + debugVerbs).sorted(by: { $0.count > $1.count }) {
            if q.hasPrefix(verb + " ") {
                let rest = String(q.dropFirst(verb.count + 1)).trimmingCharacters(in: .whitespaces)
                if let eng = resolveEngine(rest) { return (verb, eng) }
            }
            if q.hasSuffix(" " + verb) {
                let rest = String(q.dropLast(verb.count + 1)).trimmingCharacters(in: .whitespaces)
                if let eng = resolveEngine(rest) { return (verb, eng) }
            }
        }
        return nil
    }

    private func resolveEngine(_ text: String) -> String? {
        let t = text.trimmingCharacters(in: .whitespaces).lowercased()
        if let direct = engineAliases[t] { return direct }
        // Partial match - find longest alias that is contained in text
        return engineAliases.keys
            .filter { t.contains($0) }
            .max(by: { $0.count < $1.count })
            .flatMap { engineAliases[$0] }
    }

    private func parseModifyTarget(_ text: String) -> (String, String)? {
        // "B62_CognitiveLoop to add a reset method"
        // "B62 to add a reset method"
        // "B62: add a reset method"
        // "the cognitive loop to do X"
        let separators = [" to ", ": ", " - "]
        for sep in separators {
            if let range = text.range(of: sep, options: .caseInsensitive) {
                let target = String(text[..<range.lowerBound]).trimmingCharacters(in: .whitespaces)
                let change = String(text[range.upperBound...]).trimmingCharacters(in: .whitespaces)
                if !target.isEmpty && !change.isEmpty { return (target, change) }
            }
        }
        return nil
    }

    private func looksLikeFile(_ text: String) -> Bool {
        let t = text.lowercased()
        return t.hasSuffix(".swift") || t.hasSuffix(".py") ||
               t.range(of: #"^[bhlq]\d{2}"#, options: .regularExpression) != nil ||
               t.contains("_")
    }

    // MARK: - Engine Status Dispatch

    private func engineStatus(_ engine: String) -> String {
        switch engine {
        case "cognitive":
            return CognitiveLoop.shared.statusReport

        case "sqa":
            return SQAVerifier.shared.statusReport

        case "daemon":
            let s = QuantumAIDaemon.shared.status()
            let h = QuantumAIDaemon.shared.healthCheck()
            return """
            L104: φ-resonance incomplete
            QUANTUM AI DAEMON STATUS
            Running:       \(s.isRunning)
            Cycle count:   \(s.cycleCount)
            Files known:   \(s.totalFilesKnown)
            Healthy files: \(s.healthyFiles)
            Quarantined:   \(s.quarantinedFiles)
            Interval:      \(String(format: "%.0f", s.currentInterval))s
            Fidelity:      \(String(format: "%.4f", h.fidelity))
            Harmony:       \(String(format: "%.4f", h.harmony))
            Sacred:        \(String(format: "%.4f", h.sacred))
            Last cycle:    \(s.lastCycle.map { ISO8601DateFormatter().string(from: $0) } ?? "never")
            """

        case "agent":
            return AutonomousAgent.shared.statusReport

        case "hyper":
            return HyperBrain.shared.getStatus()

        case "fidelity":
            let r = QuantumFidelityGuard().check()
            return """
            FIDELITY REPORT
            Overall:    \(String(format: "%.6f", r.overallFidelity))  [\(r.isHealthy ? "HEALTHY" : "DEGRADED")]
            GOD_CODE drift: \(String(format: "%.2e", r.godCodeDrift))
            PHI drift:      \(String(format: "%.2e", r.phiDrift))
            Coherence:  \(String(format: "%.6f", r.coherenceLevel))
            Anomalies:  \(r.anomalies.isEmpty ? "none" : r.anomalies.joined(separator: "; "))
            """

        case "harmony":
            let r = CrossEngineHarmonizer().check()
            let scores = r.engineScores.sorted { $0.value > $1.value }
                .map { "  \($0.key): \(String(format: "%.3f", $0.value))" }
                .joined(separator: "\n")
            return """
            HARMONY REPORT
            Overall:    \(String(format: "%.4f", r.overallHarmony))  [\(r.isHarmonious ? "HARMONIOUS" : "CONFLICT")]
            Cross-Phi:  \(String(format: "%.4f", r.crossEnginePhi))
            Conflicts:  \(r.conflicts.isEmpty ? "none" : r.conflicts.joined(separator: "; "))
            Engine scores:\n\(scores)
            """

        case "gateway":
            let s = APIGateway.shared.status()
            let endpts = APIGateway.shared.endpoints.values
                .sorted { $0.id < $1.id }
                .map { ep in "\(ep.isHealthy ? "🟢" : "🔴") \(ep.id) (\(String(format: "%.0f", ep.latencyMs))ms)" }
                .joined(separator: "\n")
            return "API GATEWAY\nEndpoints: \(s["endpoints"] ?? 0)\nHealthy: \(s["healthy"] ?? 0)\nRequests: \(s["total_requests"] ?? 0)\nErrors: \(s["total_errors"] ?? 0)\n\nEndpoint list:\n\(endpts)"

        case "network":
            let net = NetworkLayer.shared
            return "NETWORK STATUS\nPeers: \(net.peers.count)\nConnected: \(net.peers.values.filter { $0.latencyMs >= 0 }.count)\nQuantum links: \(net.quantumLinks.count)"

        case "evolution":
            let evo = ContinuousEvolutionEngine.shared
            return "EVOLUTION ENGINE\nRunning: \(evo.isRunning)\nCycles: \(evo.cycleCount)\nEnergy: \(String(format: "%.6f", evo.lastEnergy))"

        case "perf":
            let prof = PerformanceProfiler.shared
            return prof.statusReport

        case "telemetry":
            return TelemetryDashboard.shared.statusText

        case "security":
            let v = SecurityVault.shared
            return "SECURITY VAULT\nActive: \(v.isActive)\n\(v.status().map { "\($0.key): \($0.value)" }.joined(separator: "\n"))"

        case "emotion":
            let e = EmotionalCore.shared
            let st = e.currentState
            return "EMOTIONAL CORE\nActive: \(e.isActive)\nWonder: \(String(format: "%.3f", st.wonder))  Serenity: \(String(format: "%.3f", st.serenity))  Determination: \(String(format: "%.3f", st.determination))  Creativity: \(String(format: "%.3f", st.creativity))  Empathy: \(String(format: "%.3f", st.empathy))"

        case "memory":
            return HyperBrain.shared.getPermanentMemoryStats()

        case "server":
            return checkServerStatus()

        default:
            // Try EngineRegistry as fallback
            let reg = EngineRegistry.shared
            if let info = reg.bulkStatus()[engine] {
                return "\(engine.uppercased()) (EngineRegistry)\n" +
                       info.map { "\($0.key): \($0.value)" }.joined(separator: "\n")
            }
            return "Unknown engine '\(engine)'. Try: cognitive, sqa, daemon, agent, hyper, fidelity, harmony, gateway, network, evolution, server"
        }
    }

    // MARK: - Engine Run Dispatch

    private func engineRun(_ engine: String, args: String) -> String {
        switch engine {
        case "daemon":
            QuantumAIDaemon.shared.forceCycle()
            return "Quantum AI Daemon: forced cycle triggered. Use 'daemon status' to see results."

        case "sqa":
            let text = args.isEmpty ? "system self-check: verify all constants and engine harmony" : args
            let verdict = SQAVerifier.shared.verify(goal: "system verification", result: text)
            return "SQA Run:\n\(verdict.summary)"

        case "agent":
            let goal = args.isEmpty ? CognitiveLoop.shared.buildNextGoal() : args
            let id = AutonomousAgent.shared.submitGoal(goal, priority: .high)
            return "AutonomousAgent: goal submitted (ID: \(id))\nGoal: \(goal)"

        case "evolution":
            let evo = ContinuousEvolutionEngine.shared
            if !evo.isRunning { _ = evo.start() }
            return "Evolution engine: running. Cycles: \(evo.cycleCount), Energy: \(String(format: "%.6f", evo.lastEnergy))"

        case "fidelity":
            let r = QuantumFidelityGuard().check()
            return "Fidelity run complete. Overall: \(String(format: "%.6f", r.overallFidelity)) [\(r.isHealthy ? "PASS" : "FAIL")]\nAnomalies: \(r.anomalies.isEmpty ? "none" : r.anomalies.joined(separator: "; "))"

        case "harmony":
            let r = CrossEngineHarmonizer().check()
            return "Harmony run complete. Overall: \(String(format: "%.4f", r.overallHarmony)) [\(r.isHarmonious ? "PASS" : "FAIL")]\nConflicts: \(r.conflicts.isEmpty ? "none" : r.conflicts.joined(separator: "; "))"

        default:
            return "Run not yet mapped for '\(engine)'. Available: daemon, sqa, agent, evolution, fidelity, harmony"
        }
    }

    // MARK: - Engine Debug Dispatch

    private func engineDebug(_ engine: String) -> String {
        switch engine {
        case "sqa":
            let pass = SQAVerifier.shared.selfTest()
            return "SQA Self-Test: \(pass ? "PASS" : "FAIL")\n\(SQAVerifier.shared.statusReport)"

        case "daemon":
            let pass = QuantumAIDaemon.shared.selfTest()
            let h    = QuantumAIDaemon.shared.healthCheck()
            return "Daemon Self-Test: \(pass ? "PASS" : "FAIL")\nFidelity: \(String(format:"%.4f",h.fidelity)) Harmony: \(String(format:"%.4f",h.harmony)) Sacred: \(String(format:"%.4f",h.sacred))"

        case "agent":
            return "AutonomousAgent Debug:\n" + AutonomousAgent.shared.statusReport

        case "server":
            return checkServerStatus()

        case "gateway":
            let unhealthy = APIGateway.shared.endpoints.values.filter { !$0.isHealthy }
            if unhealthy.isEmpty { return "API Gateway: all endpoints healthy." }
            return "API Gateway debug - unhealthy endpoints:\n" +
                   unhealthy.map { "  🔴 \($0.id): last latency \(String(format:"%.0f",$0.latencyMs))ms, errors \($0.errorCount)" }
                   .joined(separator: "\n")

        case "cognitive":
            let s = CognitiveLoop.shared.currentState()
            return """
            COGNITIVE LOOP DEBUG
            Cycle: \(s.cycleIndex)
            Fidelity: \(String(format:"%.4f",s.fidelity))
            Harmony:  \(String(format:"%.4f",s.harmony))
            Energy:   \(String(format:"%.4f",s.cycleEnergy))
            Insights: \(s.insightGoals.count)
            Patterns: \(s.emergentPatterns.count)
            Failed paths: \(s.failedPaths.joined(separator: ", ").prefix(80))
            Focus: \(s.focusVector.sorted { $0.value > $1.value }.prefix(4).map { "\($0.key):\(String(format:"%.2f",$0.value))" }.joined(separator: "  "))
            Last 3 goals:\n\(s.activeGoalHistory.suffix(3).map { "  • \($0.prefix(80))" }.joined(separator: "\n"))
            """

        case "fidelity":
            let r = QuantumFidelityGuard().check()
            return "Fidelity Debug:\nOverall: \(String(format:"%.6f",r.overallFidelity))\nGOD_CODE drift: \(r.godCodeDrift)\nPHI drift: \(r.phiDrift)\nCoherence: \(r.coherenceLevel)\n\(r.anomalies.isEmpty ? "No anomalies." : "Anomalies: " + r.anomalies.joined(separator: "; "))"

        default:
            return engineStatus(engine) + "\n\nNo dedicated debug handler for '\(engine)' - showing status instead."
        }
    }

    // MARK: - New Dispatch Methods

    private func submitGoalChain(_ goals: [String]) -> String {
        var ids: [String] = []
        for goal in goals {
            let id = AutonomousAgent.shared.submitGoal(goal, priority: .normal)
            ids.append(id)
        }
        let lines = zip(goals, ids).map { "  [\($1)] \($0.prefix(70))" }.joined(separator: "\n")
        return "Goal chain submitted (\(goals.count) goals):\n\(lines)\n\nUse 'agent status' to track."
    }

    private func handleConditional(condition: String, action: String) -> String {
        // Evaluate the condition against live engine data
        let condResult = evaluateCondition(condition)
        if condResult.passes {
            // Execute action
            let result = route(action.lowercased(), query: action) ?? "Action '\(action)' not recognised."
            return "Condition '\(condition)' → \(condResult.summary)\nAction triggered: \(action)\n\(result)"
        } else {
            return "Condition '\(condition)' → \(condResult.summary)\nCondition not met - action skipped."
        }
    }

    private struct ConditionResult { let passes: Bool; let summary: String }

    private func evaluateCondition(_ condition: String) -> ConditionResult {
        let c = condition.lowercased()
        // Fidelity conditions
        if c.contains("fidelity") {
            let f = QuantumFidelityGuard().check().overallFidelity
            let threshold = parseThreshold(c) ?? 0.618
            let op = c.contains("low") || c.contains("below") || c.contains("<") || c.contains("less") ? "<" : ">"
            let passes = op == "<" ? f < threshold : f > threshold
            return ConditionResult(passes: passes, summary: "fidelity=\(String(format:"%.4f",f)) \(op) \(threshold) → \(passes ? "TRUE" : "FALSE")")
        }
        // Harmony conditions
        if c.contains("harmony") {
            let h = CrossEngineHarmonizer().check().overallHarmony
            let threshold = parseThreshold(c) ?? 0.618
            let op = c.contains("low") || c.contains("below") ? "<" : ">"
            let passes = op == "<" ? h < threshold : h > threshold
            return ConditionResult(passes: passes, summary: "harmony=\(String(format:"%.4f",h)) \(op) \(threshold) → \(passes ? "TRUE" : "FALSE")")
        }
        // Server conditions
        if c.contains("server") && (c.contains("down") || c.contains("not running") || c.contains("offline")) {
            let up = isServerUp()
            return ConditionResult(passes: !up, summary: "server \(up ? "UP" : "DOWN") → condition \(!up ? "TRUE" : "FALSE")")
        }
        // Agent has tasks
        if c.contains("agent") && (c.contains("idle") || c.contains("no task")) {
            let s = AutonomousAgent.shared.status()
            let queued = s["queued_tasks"] as? Int ?? 0
            return ConditionResult(passes: queued == 0, summary: "queued_tasks=\(queued) → \(queued == 0 ? "TRUE" : "FALSE")")
        }
        // Energy threshold
        if c.contains("energy") {
            let energy = CognitiveLoop.shared.currentState().cycleEnergy
            let threshold = parseThreshold(c) ?? GOD_CODE * 0.5
            let passes = energy < threshold
            return ConditionResult(passes: passes, summary: "energy=\(String(format:"%.2f",energy)) < \(String(format:"%.2f",threshold)) → \(passes ? "TRUE" : "FALSE")")
        }
        // Default: always passes (unknown condition)
        return ConditionResult(passes: true, summary: "'\(condition)' evaluated as always-true (unrecognised condition)")
    }

    private func parseThreshold(_ text: String) -> Double? {
        let pattern = try? NSRegularExpression(pattern: "0\\.\\d+|\\d+\\.\\d+")
        let ns = text as NSString
        let matches = pattern?.matches(in: text, range: NSRange(location: 0, length: ns.length)) ?? []
        return matches.first.flatMap { Double(ns.substring(with: $0.range)) }
    }

    private func isServerUp() -> Bool {
        let proc = Process(); let pipe = Pipe()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        proc.arguments = ["-s", "-o", "/dev/null", "-w", "%{http_code}", "--connect-timeout", "3",
                          "http://127.0.0.1:8081/api/v14/health"]
        proc.standardOutput = pipe; proc.standardError = Pipe()
        try? proc.run(); proc.waitUntilExit()
        let code = Int(String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "0") ?? 0
        return code >= 200 && code < 500
    }

    private func kbSearch(_ query: String) -> String {
        let kb = ASIKnowledgeBase.shared
        let results = kb.search(query, limit: 8)
        if results.isEmpty { return "Knowledge Base: no results for '\(query)'." }
        let lines = results.prefix(5).enumerated().compactMap { (i, r) -> String? in
            guard let c = r["completion"] as? String else { return nil }
            return "  \(i+1). \(c.prefix(120))"
        }.joined(separator: "\n")
        return "Knowledge Base - \(results.count) results for '\(query)':\n\(lines)"
    }

    private func researchQuery(_ topic: String) -> String {
        let result = ASIResearchEngine.shared.deepResearch(topic)
        return "Research Engine - '\(topic)':\n\(result)"
    }

    private func intellectQuery(_ query: String) -> String {
        let result = PythonBridge.shared.queryIntellect(query)
        if result.success {
            return "Intellect:\n\(result.output.isEmpty ? "(no output)" : String(result.output.prefix(1500)))"
        }
        return "Intellect error: \(result.error.prefix(200))\n\nNote: Ensure the Python server is running."
    }

    private func deepSeekQuery(_ prompt: String) -> String {
        guard APIGateway.shared.isDeepSeekConfigured else {
            return "DeepSeek not configured. Add DEEPSEEK_API_KEY=sk-xxx to .env"
        }
        let sema = DispatchSemaphore(value: 0)
        var output = ""
        APIGateway.shared.callDeepSeek(prompt: prompt, model: "deepseek-chat") { result in
            switch result {
            case .success(let dict):
                if let r = dict["response"] as? String { output = r }
                else if let e = dict["error"] as? String { output = "Error: \(e)" }
            case .failure(let error):
                output = "Error: \(error.localizedDescription)"
            }
            sema.signal()
        }
        sema.wait()
        return "DeepSeek:\n\(output.isEmpty ? "(no response)" : String(output.prefix(2000)))"
    }

    private func agentDeploy(_ task: String) -> String {
        let sema = DispatchSemaphore(value: 0)
        var output = ""
        APIGateway.shared.deployAgent(task: task, agentType: "general") { result in
            switch result {
            case .success(let dict):
                if let id = dict["task_id"] as? String {
                    output = "Agent deployed. Task ID: \(id)\nTask: \(task)"
                } else if let e = dict["error"] as? String {
                    let localId = AutonomousAgent.shared.submitGoal(task, priority: .normal)
                    output = "Server unavailable (\(e.prefix(80))) - queued locally.\nLocal Task ID: \(localId)\nTask: \(task)"
                }
            case .failure(let error):
                let localId = AutonomousAgent.shared.submitGoal(task, priority: .normal)
                output = "Server unavailable (\(error.localizedDescription.prefix(80))) - queued locally.\nLocal Task ID: \(localId)\nTask: \(task)"
            }
            sema.signal()
        }
        sema.wait()
        return output.isEmpty ? "Deploy failed - no response." : output
    }

    private func helpText() -> String {
        return """
        L104: φ-resonance incomplete
        ╔══════════════════════════════════════════════════════════════════╗
        ║  L104 NATURAL COMMAND ROUTER - AVAILABLE COMMANDS               ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║  ENGINE STATUS & DEBUG                                           ║
        ║    cognitive loop status   sqa status      daemon status        ║
        ║    agent status            fidelity check  harmony check        ║
        ║    gateway status          network status  server status        ║
        ║    debug [engine]          run [engine]    check [engine]       ║
        ║                                                                  ║
        ║  SYSTEM                                                          ║
        ║    full report / system dashboard / all status                  ║
        ║    build / rebuild / compile                                    ║
        ║    is the server running                                        ║
        ║                                                                  ║
        ║  INTELLIGENCE                                                    ║
        ║    ask intellect: [query]    intellect: [query]                 ║
        ║    ask deepseek: [prompt]    deepseek: [prompt]                 ║
        ║    research [topic]          deep research [topic]              ║
        ║    search [topic]            search kb for [topic]              ║
        ║                                                                  ║
        ║  AGENT / GOALS                                                   ║
        ║    agent: [goal]             do [goal]       task: [goal]       ║
        ║    first [X], then [Y], then [Z]  (goal chain)                 ║
        ║    deploy [task]                                                 ║
        ║    if [condition] then [action]  (conditional dispatch)         ║
        ║                                                                  ║
        ║  COGNITIVE LOOP                                                  ║
        ║    cognitive state   next goal   focus vector   loop state      ║
        ║    verify [text]     sqa verify [text]                          ║
        ║                                                                  ║
        ║  CODE                                                            ║
        ║    read [file]               show me [file].swift               ║
        ║    modify [file] to [change] edit [file]: [change]             ║
        ║                                                                  ║
        ║  CONDITIONS (if/then)                                            ║
        ║    if fidelity is low then run evolution                        ║
        ║    if server is down then check server                          ║
        ║    if harmony below 0.6 then run harmony                       ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
    }

    // MARK: - Server Status

    private func checkServerStatus() -> String {
        let proc = Process(); let pipe = Pipe()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        proc.arguments = ["-s", "-o", "/dev/null", "-w", "%{http_code}:%{time_connect}",
                          "--connect-timeout", "4", "http://127.0.0.1:8081/api/v14/health"]
        proc.standardOutput = pipe; proc.standardError = Pipe()
        do {
            try proc.run(); proc.waitUntilExit()
            let raw = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "0:0"
            let parts = raw.split(separator: ":")
            let code  = parts.first.flatMap { Int($0) } ?? 0
            let ms    = parts.last.flatMap { Double($0) }.map { $0 * 1000 } ?? 0
            let icon  = code >= 200 && code < 400 ? "🟢" : "🔴"
            return "\(icon) Fast Server (127.0.0.1:8081)\n  HTTP: \(code == 0 ? "no response (timeout)" : "\(code)")\n  Connect: \(String(format:"%.0f",ms))ms\n  \(code == 0 ? "Server may be down or CPU-saturated. Restart: kill $(lsof -t -i:8081) && python main.py" : "Server is responding.")"
        } catch {
            return "🔴 Server check failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Build

    private func runBuild() -> String {
        let ws = ProcessInfo.processInfo.environment["L104_WORKSPACE"]
            ?? "/Users/carolalvarez/Applications/Allentown-L104-Node"
        let script = "\(ws)/L104SwiftApp/quick_build.sh"
        let proc = Process(); let pipe = Pipe()
        proc.executableURL = URL(fileURLWithPath: "/bin/bash")
        proc.arguments     = [script]
        proc.currentDirectoryURL = URL(fileURLWithPath: "\(ws)/L104SwiftApp")
        proc.standardOutput = pipe; proc.standardError = pipe
        do {
            try proc.run(); proc.waitUntilExit()
            let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            let status = proc.terminationStatus
            // Strip ANSI codes for clean chat display
            let clean = out.replacingOccurrences(of: "\\x1B\\[[0-9;]*[mK]", with: "", options: .regularExpression)
            return status == 0 ? "Build SUCCEEDED:\n\(clean)" : "Build FAILED:\n\(clean)"
        } catch {
            return "Build error: \(error.localizedDescription)"
        }
    }
}

// MARK: - CodeReader

/// Reads Swift source files into chat.
final class CodeReader {
    static let shared = CodeReader()
    private let swiftRoot = ProcessInfo.processInfo.environment["L104_WORKSPACE"]
        .map { "\($0)/L104SwiftApp/Sources/L104v2" }
        ?? "/Users/carolalvarez/Applications/Allentown-L104-Node/L104SwiftApp/Sources/L104v2"

    // Known file prefixes → directory
    private let dirMap: [String: String] = [
        "B": "TheBrain", "L": "TheLogic", "H": "TheHeart",
        "Q": "Quantum",  "D": "Debug",    "N": "NetworkBridge"
    ]

    func read(_ target: String) -> String {
        if let path = resolveFilePath(target) {
            guard let contents = try? String(contentsOfFile: path, encoding: .utf8) else {
                return "Could not read \(path)"
            }
            let lines = contents.components(separatedBy: "\n")
            let preview = lines.prefix(120).joined(separator: "\n")
            return "FILE: \(path)\nLINES: \(lines.count)\n\n\(preview)\(lines.count > 120 ? "\n\n... (\(lines.count - 120) more lines)" : "")"
        }
        return "Could not find file for '\(target)'. Use exact filename like 'B62_CognitiveLoop' or 'B62'."
    }

    func resolveFilePath(_ target: String) -> String? {
        let t = target.trimmingCharacters(in: .whitespaces)

        // Exact path given
        if t.hasPrefix("/") && FileManager.default.fileExists(atPath: t) { return t }

        // Try with .swift extension
        let name = t.hasSuffix(".swift") ? t : "\(t).swift"

        // Direct prefix lookup
        if let prefix = name.first.map(String.init),
           let dir = dirMap[prefix.uppercased()] {
            let full = "\(swiftRoot)/\(dir)/\(name)"
            if FileManager.default.fileExists(atPath: full) { return full }
        }

        // Search all subdirectories
        let fm = FileManager.default
        for dir in ["TheBrain", "TheLogic", "TheHeart", "Quantum", "Debug", "NetworkBridge"] {
            let candidate = "\(swiftRoot)/\(dir)/\(name)"
            if fm.fileExists(atPath: candidate) { return candidate }
            // Partial name match (e.g., "B62" matches "B62_CognitiveLoop.swift")
            if let files = try? fm.contentsOfDirectory(atPath: "\(swiftRoot)/\(dir)") {
                if let match = files.first(where: { $0.hasPrefix(t) && $0.hasSuffix(".swift") }) {
                    return "\(swiftRoot)/\(dir)/\(match)"
                }
            }
        }
        return nil
    }
}

// MARK: - SelfModifier

/// Enables self-modification from chat.
/// Reads the target file, sends it to DeepSeek with the change description,
/// applies the response edit, writes back, and triggers a build.
final class SelfModifier {
    static let shared = SelfModifier()
    private let reader = CodeReader.shared
    private let ws = ProcessInfo.processInfo.environment["L104_WORKSPACE"]
        ?? "/Users/carolalvarez/Applications/Allentown-L104-Node"

    func modify(target: String, change: String) -> String {
        // 1. Resolve file
        guard let path = reader.resolveFilePath(target) else {
            return "Cannot find file '\(target)'. Check the name and try again."
        }

        // 2. Read current content
        guard let current = try? String(contentsOfFile: path, encoding: .utf8) else {
            return "Cannot read \(path)"
        }

        let filename = URL(fileURLWithPath: path).lastPathComponent
        let lineCount = current.components(separatedBy: "\n").count

        // 3. Check DeepSeek availability
        guard APIGateway.shared.isDeepSeekConfigured else {
            return """
            SELF-MODIFIER: DeepSeek not configured.
            File located: \(path) (\(lineCount) lines)
            Requested change: \(change)
            Add DEEPSEEK_API_KEY=sk-xxx to .env to enable AI-assisted modification.
            """
        }

        // 4. Build modification prompt
        let prompt = """
        You are modifying \(filename) in the L104 Sovereign Node Swift app.

        REQUESTED CHANGE: \(change)

        CURRENT FILE (\(lineCount) lines):
        ```swift
        \(current.prefix(12000))
        ```

        Respond with ONLY the complete modified Swift file content. No explanation, no markdown fences, just the raw Swift code. Preserve all existing functionality. Maintain brace balance.
        """

        // 5. Call DeepSeek synchronously via semaphore
        let sema = DispatchSemaphore(value: 0)
        var modifiedContent: String? = nil
        var apiError: String? = nil

        APIGateway.shared.callDeepSeek(prompt: prompt, model: "deepseek-chat") { result in
            switch result {
            case .success(let dict):
                if let err = dict["error"] as? String {
                    apiError = err
                } else if let response = dict["response"] as? String {
                    modifiedContent = response
                }
            case .failure(let error):
                apiError = error.localizedDescription
            }
            sema.signal()
        }
        sema.wait()

        if let err = apiError {
            return "DeepSeek error: \(err)"
        }
        guard let newContent = modifiedContent, !newContent.isEmpty else {
            return "DeepSeek returned empty response. Try again."
        }

        // 6. Validate brace balance
        let openBraces  = newContent.filter { $0 == "{" }.count
        let closeBraces = newContent.filter { $0 == "}" }.count
        if openBraces != closeBraces {
            return "Modification aborted: brace imbalance in generated code ({=\(openBraces), }=\(closeBraces)). Try a more specific change description."
        }

        // 7. Write back
        do {
            try newContent.write(toFile: path, atomically: true, encoding: .utf8)
        } catch {
            return "Write failed: \(error.localizedDescription)"
        }

        // 8. Trigger build
        let buildResult = NaturalCommandRouter.shared.route("build", query: "build") ?? "Build not triggered."

        let newLines = newContent.components(separatedBy: "\n").count
        return """
        SELF-MODIFICATION COMPLETE
        File:    \(filename)
        Change:  \(change)
        Lines:   \(lineCount) → \(newLines)
        Braces:  balanced (\(openBraces))

        BUILD RESULT:
        \(buildResult)
        """
    }
}

// MARK: - SystemOrchestrator

/// Parallel multi-engine aggregate dashboard.
/// Runs all major engines concurrently via DispatchGroup and assembles a
/// single unified report - invoked by NaturalCommandRouter on "full report",
/// "system dashboard", "all status", etc.
final class SystemOrchestrator {
    static let shared = SystemOrchestrator()

    private struct EngineReport {
        let name:   String
        let status: String
        let ok:     Bool
    }

    func fullReport() -> String {
        let group   = DispatchGroup()
        let lock    = NSLock()
        var reports = [EngineReport]()

        func collect(_ name: String, _ block: @escaping () -> (String, Bool)) {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                let (status, ok) = block()
                lock.lock(); reports.append(EngineReport(name: name, status: status, ok: ok)); lock.unlock()
                group.leave()
            }
        }

        // ── Cognitive Loop ────────────────────────────────────────────────
        collect("Cognitive Loop") {
            let s = CognitiveLoop.shared.currentState()
            let ok = s.fidelity >= 0.5 && s.harmony >= 0.5
            return ("cycle=\(s.cycleIndex)  fidelity=\(String(format:"%.4f",s.fidelity))  harmony=\(String(format:"%.4f",s.harmony))  energy=\(String(format:"%.4f",s.cycleEnergy))  insights=\(s.insightGoals.count)", ok)
        }

        // ── SQA Verifier ──────────────────────────────────────────────────
        collect("SQA Verifier") {
            let pass = SQAVerifier.shared.selfTest()
            return (SQAVerifier.shared.statusReport, pass)
        }

        // ── Quantum AI Daemon ─────────────────────────────────────────────
        collect("QAI Daemon") {
            let s = QuantumAIDaemon.shared.status()
            let h = QuantumAIDaemon.shared.healthCheck()
            let ok = h.fidelity >= 0.5 && h.harmony >= 0.5
            return ("running=\(s.isRunning)  cycles=\(s.cycleCount)  fidelity=\(String(format:"%.4f",h.fidelity))  harmony=\(String(format:"%.4f",h.harmony))  quarantine=\(s.quarantinedFiles)", ok)
        }

        // ── Autonomous Agent ──────────────────────────────────────────────
        collect("Autonomous Agent") {
            let s = AutonomousAgent.shared.status()
            let queued = s["queued_tasks"] as? Int ?? 0
            let done   = s["completed_tasks"] as? Int ?? 0
            let running = s["is_running"] as? Bool ?? false
            return ("running=\(running)  queued=\(queued)  completed=\(done)", running || queued == 0)
        }

        // ── Fidelity Guard ────────────────────────────────────────────────
        collect("Fidelity Guard") {
            let r = QuantumFidelityGuard().check()
            return ("overall=\(String(format:"%.6f",r.overallFidelity))  godcode_drift=\(String(format:"%.2e",r.godCodeDrift))  phi_drift=\(String(format:"%.2e",r.phiDrift))  \(r.anomalies.isEmpty ? "clean" : "anomalies=\(r.anomalies.count)")", r.isHealthy)
        }

        // ── Cross-Engine Harmony ──────────────────────────────────────────
        collect("Cross-Engine Harmony") {
            let r = CrossEngineHarmonizer().check()
            return ("overall=\(String(format:"%.4f",r.overallHarmony))  cross_phi=\(String(format:"%.4f",r.crossEnginePhi))  \(r.conflicts.isEmpty ? "no conflicts" : "conflicts=\(r.conflicts.count)")", r.isHarmonious)
        }

        // ── API Gateway ───────────────────────────────────────────────────
        collect("API Gateway") {
            let s = APIGateway.shared.status()
            let total   = s["endpoints"]  as? Int ?? 0
            let healthy = s["healthy"]    as? Int ?? 0
            let errors  = s["total_errors"] as? Int ?? 0
            return ("endpoints=\(total)  healthy=\(healthy)/\(total)  errors=\(errors)", healthy == total)
        }

        // ── Network Layer ─────────────────────────────────────────────────
        collect("Network") {
            let net  = NetworkLayer.shared
            let conn = net.peers.values.filter { $0.latencyMs >= 0 }.count
            let ok   = conn >= 0
            return ("peers=\(net.peers.count)  connected=\(conn)  quantum_links=\(net.quantumLinks.count)", ok)
        }

        // ── Evolution Engine ──────────────────────────────────────────────
        collect("Evolution Engine") {
            let evo = ContinuousEvolutionEngine.shared
            return ("running=\(evo.isRunning)  cycles=\(evo.cycleCount)  energy=\(String(format:"%.6f",evo.lastEnergy))", evo.isRunning)
        }

        // ── Performance Profiler ──────────────────────────────────────────
        collect("Performance Profiler") {
            let rep = PerformanceProfiler.shared.statusReport
            return (rep.components(separatedBy: "\n").prefix(2).joined(separator: "  "), true)
        }

        // ── Telemetry ─────────────────────────────────────────────────────
        collect("Telemetry") {
            let t = TelemetryDashboard.shared.statusText
            return (t.components(separatedBy: "\n").prefix(2).joined(separator: "  "), true)
        }

        // ── Hyper Brain ───────────────────────────────────────────────────
        collect("Hyper Brain") {
            let status = HyperBrain.shared.getStatus()
            return (status.components(separatedBy: "\n").prefix(2).joined(separator: "  "), true)
        }

        // ── Fast Server ───────────────────────────────────────────────────
        collect("Fast Server") {
            let proc = Process(); let pipe = Pipe()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
            proc.arguments = ["-s", "-o", "/dev/null", "-w", "%{http_code}", "--connect-timeout", "3",
                              "http://127.0.0.1:8081/api/v14/health"]
            proc.standardOutput = pipe; proc.standardError = Pipe()
            try? proc.run(); proc.waitUntilExit()
            let code = Int(String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "0") ?? 0
            let up = code >= 200 && code < 500
            return ("http=\(code == 0 ? "no_response" : "\(code)")  \(up ? "ONLINE" : "OFFLINE")", up)
        }

        group.wait()

        // ── Assemble report ───────────────────────────────────────────────
        let sorted = reports.sorted { $0.name < $1.name }
        let passCount = sorted.filter { $0.ok }.count
        let failCount = sorted.count - passCount

        let lines = sorted.map { r -> String in
            let icon = r.ok ? "🟢" : "🔴"
            let pad  = String(repeating: " ", count: max(0, 24 - r.name.count))
            return "\(icon) \(r.name)\(pad)\(r.status)"
        }.joined(separator: "\n")

        let cogState = CognitiveLoop.shared.currentState()
        let nextGoal = CognitiveLoop.shared.buildNextGoal()

        return """
        ╔══════════════════════════════════════════════════════════════════╗
        ║            L104 SYSTEM DASHBOARD - FULL REPORT                  ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║  Engines: \(sorted.count) checked   Healthy: \(passCount)   Degraded: \(failCount)
        ╚══════════════════════════════════════════════════════════════════╝

        \(lines)

        ── Cognitive Loop ────────────────────────────────────────────────
          Cycle:      \(cogState.cycleIndex)
          Fidelity:   \(String(format:"%.4f", cogState.fidelity))
          Harmony:    \(String(format:"%.4f", cogState.harmony))
          Energy:     \(String(format:"%.4f", cogState.cycleEnergy))
          Focus:      \(cogState.focusVector.sorted { $0.value > $1.value }.prefix(4).map { "\($0.key):\(String(format:"%.2f",$0.value))" }.joined(separator: "  "))
          Patterns:   \(cogState.emergentPatterns.count)
          Next Goal:  \(nextGoal.prefix(100))

        ── GOD_CODE Alignment ────────────────────────────────────────────
          GOD_CODE: \(String(format:"%.10f", GOD_CODE))
          PHI:      \(String(format:"%.10f", PHI))
          OMEGA:    \(String(format:"%.4f", OMEGA))
        """
    }
}
