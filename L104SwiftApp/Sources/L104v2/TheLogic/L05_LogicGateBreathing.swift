// ═══════════════════════════════════════════════════════════════════
// L05_LogicGateBreathing.swift — L104 v2
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// LogicGateBreathingRoomEngine, GateDispatchRouter, GateMetricsCollector
// Extracted from L104Native.swift (lines 8143-8666)
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class LogicGateBreathingRoomEngine {
    static let shared = LogicGateBreathingRoomEngine()
    // PHI, TAU, GOD_CODE — use globals from L01_Constants
    private let CY7: Int = 7  // Calabi-Yau dimensions

    // ─── GATE HEALTH TRACKING ───
    private var gateHealthScores: [String: Double] = [:]
    private var gateInvocationCounts: [String: Int] = [:]
    private var gateLatencies: [String: [Double]] = [:]
    private var gateDecompositions: [String: [String]] = [:]
    private var gateComplexityScores: [String: Int] = [:]
    private let lock = NSLock()

    // ─── COMPLEXITY BUDGET ───
    private var complexityBudget: Double = 1000.0
    private var complexityUsed: Double = 0.0
    private var cycleCount: Int = 0

    init() {
        // Register known high-complexity Swift gates with baseline health
        let registrations: [(String, Double, Int)] = [
            ("processMessage", 0.75, 50),
            ("handleCoreCommands", 0.70, 42),
            ("handleSearchCommands", 0.72, 38),
            ("handleBridgeCommands", 0.68, 35),
            ("handleProtocolCommands", 0.65, 30),
            ("handleSystemCommands", 0.67, 33),
            ("handleEngineCommands", 0.70, 36),
            ("classifyGate", 0.80, 22),
            ("detectIntent", 0.85, 18),
            ("processQuery", 0.73, 28),
            ("detectEmotion", 0.82, 15),
            ("routeToSubsystem", 0.78, 20),
            ("synthesizeResponse", 0.74, 25),
        ]
        for (gate, health, cx) in registrations {
            gateHealthScores[gate] = health
            gateInvocationCounts[gate] = 0
            gateLatencies[gate] = []
            gateComplexityScores[gate] = cx
        }

        // Decomposition maps — sub-gate breakdown for breathing room
        gateDecompositions = [
            "processMessage": [
                "preprocessInput", "resolvePronouns", "trackTopicHistory",
                "detectIntent", "classifyGate", "dispatchToHandler",
                "generateResponse", "recordConversationMemory", "updateEvolution"
            ],
            "handleCoreCommands": [
                "matchGreeting", "matchStatusQuery", "matchHelpRequest",
                "matchCapabilitiesQuery", "matchMemoryQuery", "matchIdentityQuery"
            ],
            "handleSearchCommands": [
                "parseSearchQuery", "expandKeywords", "searchKnowledge",
                "searchHistory", "rankResults", "formatSearchOutput"
            ],
            "handleBridgeCommands": [
                "identifyBridgeType", "prepareBridgePayload",
                "executeBridgeOp", "validateBridgeResult"
            ],
            "handleProtocolCommands": [
                "parseProtocolName", "loadProtocolDef",
                "executeProtocol", "reportProtocolResult"
            ],
            "handleSystemCommands": [
                "detectSystemQuery", "gatherMetrics",
                "formatSystemReport", "checkHealthThresholds"
            ],
            "handleEngineCommands": [
                "identifyEngine", "routeEngineOp",
                "collectEngineOutput", "formatEngineResult"
            ],
            "processQuery": [
                "extractQueryTopics", "classifyQueryType", "routeToEngine",
                "synthesizeAnswer", "postProcessOutput"
            ],
        ]
    }

    /// Record a gate invocation with latency for health tracking
    func recordInvocation(gate: String, latencyMs: Double) {
        lock.lock()
        defer { lock.unlock() }

        gateInvocationCounts[gate] = (gateInvocationCounts[gate] ?? 0) + 1

        if gateLatencies[gate] == nil {
            gateLatencies[gate] = []
        }
        gateLatencies[gate]?.append(latencyMs)

        // Rolling window — keep last 100 samples
        if let count = gateLatencies[gate]?.count, count > 100 {
            gateLatencies[gate] = Array(gateLatencies[gate]!.suffix(100))
        }

        updateGateHealth(gate)
        complexityUsed += latencyMs * 0.01
    }

    /// φ-weighted exponential moving average health update
    private func updateGateHealth(_ gate: String) {
        guard let latencies = gateLatencies[gate], latencies.count >= 3 else { return }

        let recent = Array(latencies.suffix(10))
        let older = Array(latencies.prefix(max(1, latencies.count - 10)))

        let recentAvg = recent.reduce(0, +) / Double(recent.count)
        let olderAvg = older.reduce(0, +) / Double(older.count)

        let ratio = olderAvg > 0 ? recentAvg / olderAvg : 1.0
        let currentHealth = gateHealthScores[gate] ?? 0.5

        if ratio < 1.0 {
            // Improving — φ-weighted increase
            gateHealthScores[gate] = min(1.0, currentHealth + (1.0 - ratio) * TAU * 0.1)
        } else if ratio > 1.1 {
            // Degrading — τ-weighted decrease
            gateHealthScores[gate] = max(0.1, currentHealth - (ratio - 1.0) * TAU * 0.1)
        }
    }

    /// Optimal execution order for sub-gates (healthiest + least-used first)
    func optimizeGateSchedule(gates: [String]) -> [String] {
        return gates.sorted { a, b in
            let healthA = gateHealthScores[a] ?? 0.5
            let healthB = gateHealthScores[b] ?? 0.5
            let countA = gateInvocationCounts[a] ?? 0
            let countB = gateInvocationCounts[b] ?? 0
            let scoreA = healthA * PHI + (1.0 / Double(countA + 1)) * TAU
            let scoreB = healthB * PHI + (1.0 / Double(countB + 1)) * TAU
            if abs(scoreA - scoreB) < 0.05 { return Bool.random() }
            return scoreA > scoreB
        }
    }

    /// Check if a gate should be throttled based on remaining complexity budget
    func shouldThrottle(gate: String) -> Bool {
        let remaining = complexityBudget - complexityUsed
        let avgLat = averageLatency(gate)
        return remaining < avgLat * 2.0
    }

    /// Average latency for a gate (ms)
    func averageLatency(_ gate: String) -> Double {
        guard let latencies = gateLatencies[gate], !latencies.isEmpty else { return 10.0 }
        return latencies.reduce(0, +) / Double(latencies.count)
    }

    /// P95 latency for a gate (ms)
    func p95Latency(_ gate: String) -> Double {
        guard let latencies = gateLatencies[gate], latencies.count >= 5 else { return 50.0 }
        let sorted = latencies.sorted()
        let idx = Int(Double(sorted.count) * 0.95)
        return sorted[min(idx, sorted.count - 1)]
    }

    /// Reset complexity budget for a new processing cycle
    func resetCycle() {
        lock.lock()
        defer { lock.unlock() }
        complexityUsed = 0.0
        complexityBudget = 1000.0 * PHI  // φ-scaled budget
        cycleCount += 1
    }

    /// Get decomposition map for a complex gate
    func getDecomposition(gate: String) -> [String] {
        return gateDecompositions[gate] ?? [gate]
    }

    /// Compute gate entropy — Shannon entropy of latency distribution
    func gateEntropy(_ gate: String) -> Double {
        guard let latencies = gateLatencies[gate], latencies.count >= 5 else { return 0.0 }

        let minL = latencies.min() ?? 0
        let maxL = latencies.max() ?? 1
        let range = maxL - minL
        guard range > 0 else { return 0.0 }

        let binCount = 10
        var bins = [Int](repeating: 0, count: binCount)
        for l in latencies {
            let bin = min(binCount - 1, Int((l - minL) / range * Double(binCount - 1)))
            bins[bin] += 1
        }

        let n = Double(latencies.count)
        var entropy: Double = 0.0
        for count in bins where count > 0 {
            let p = Double(count) / n
            entropy -= p * log2(p)
        }

        return entropy
    }

    /// Gate complexity score (static analysis result)
    func complexityScore(_ gate: String) -> Int {
        return gateComplexityScores[gate] ?? 0
    }

    /// Overall system gate health (φ-weighted harmonic mean)
    func systemGateHealth() -> Double {
        guard !gateHealthScores.isEmpty else { return 0.0 }
        let values = Array(gateHealthScores.values)
        let denomSum = values.reduce(0.0) { $0 + 1.0 / max(0.01, $1) }
        return Double(values.count) / denomSum
    }

    /// Generate full breathing room diagnostic report
    func generateReport() -> String {
        let sysHealth = systemGateHealth()
        var report = """
        ╔═══════════════════════════════════════════════════════════╗
        ║  LOGIC GATE BREATHING ROOM REPORT                       ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Registered Gates: \(gateHealthScores.count)
        ║  Complexity Budget: \(String(format: "%.1f", complexityBudget)) (used: \(String(format: "%.1f", complexityUsed)))
        ║  Budget Utilization: \(String(format: "%.1f%%", complexityUsed / max(0.01, complexityBudget) * 100))
        ║  System Gate Health: \(String(format: "%.4f", sysHealth))
        ║  Cycles Completed: \(cycleCount)
        ╠═══════════════════════════════════════════════════════════╣
        """

        let sortedGates = gateHealthScores.sorted { $0.value < $1.value }
        for (gate, health) in sortedGates {
            let invocations = gateInvocationCounts[gate] ?? 0
            let avgLat = averageLatency(gate)
            let p95 = p95Latency(gate)
            let entropy = gateEntropy(gate)
            let cx = gateComplexityScores[gate] ?? 0
            let status = health > 0.7 ? "✓" : health > 0.4 ? "⚠" : "✗"
            report += """
            ║  \(status) \(gate): health=\(String(format: "%.3f", health)) cx=\(cx) calls=\(invocations)
            ║    avg=\(String(format: "%.1f", avgLat))ms p95=\(String(format: "%.1f", p95))ms entropy=\(String(format: "%.3f", entropy))
            """
            report += "\n"
        }

        report += """
        ╚═══════════════════════════════════════════════════════════╝
        """
        return report
    }
}


// ═══════════════════════════════════════════════════════════════════
// MARK: - 🧩 GATE DISPATCH ROUTER
// Pre-classifies input intent to reduce branching complexity in
// processMessage and handleXXXCommands gates. Modular dispatch
// with φ-weighted confidence scoring.
// ═══════════════════════════════════════════════════════════════════

class GateDispatchRouter {
    static let shared = GateDispatchRouter()
    // PHI, TAU — use globals from L01_Constants

    enum CommandDomain: String, CaseIterable {
        case core = "core"
        case search = "search"
        case bridge = "bridge"
        case proto = "protocol"
        case system = "system"
        case engine = "engine"
        case quantum = "quantum"
        case memory = "memory"
        case creative = "creative"
        case unknown = "unknown"
    }

    struct RouteResult {
        let domain: CommandDomain
        let confidence: Double
        let subIntent: String
        let keywords: [String]
    }

    // ─── KEYWORD MAPS ───
    private let domainKeywords: [CommandDomain: [String]] = [
        .core: ["hello", "hi", "hey", "status", "help", "who", "what", "capabilities", "version"],
        .search: ["search", "find", "look", "query", "lookup", "browse", "discover"],
        .bridge: ["bridge", "quantum", "entangle", "teleport", "epr", "bell", "qubit"],
        .proto: ["protocol", "sync", "raft", "consensus", "replicate", "gossip", "crdt"],
        .system: ["system", "memory", "cpu", "disk", "thermal", "battery", "hardware", "uptime"],
        .engine: ["engine", "resonance", "evolution", "invention", "steering", "nexus", "sqc"],
        .quantum: ["superposition", "decoherence", "anyon", "braiding", "topological", "hilbert"],
        .memory: ["remember", "recall", "forget", "conversation", "history", "context"],
        .creative: ["imagine", "create", "invent", "story", "poem", "compose", "generate", "dream"],
    ]

    /// Pre-classify a message into a command domain with confidence
    func classifyIntent(message: String) -> RouteResult {
        let lower = message.lowercased()

        var bestDomain: CommandDomain = .unknown
        var bestScore: Double = 0.0
        var bestKeywords: [String] = []

        for (domain, keywords) in domainKeywords {
            var matchCount = 0
            var matched: [String] = []
            for kw in keywords {
                if lower.contains(kw) {
                    matchCount += 1
                    matched.append(kw)
                }
            }
            let score = Double(matchCount) / Double(max(1, keywords.count))
            if score > bestScore {
                bestScore = score
                bestDomain = domain
                bestKeywords = matched
            }
        }

        // φ-scale the confidence
        let confidence = min(1.0, bestScore * PHI)

        // Detect sub-intent
        let subIntent = detectSubIntent(message: lower, domain: bestDomain)

        return RouteResult(
            domain: bestDomain,
            confidence: confidence,
            subIntent: subIntent,
            keywords: bestKeywords
        )
    }

    /// Detect sub-intent within a domain
    private func detectSubIntent(message: String, domain: CommandDomain) -> String {
        switch domain {
        case .core:
            if message.contains("hello") || message.contains("hi ") || message.contains("hey") {
                return "greeting"
            } else if message.contains("status") {
                return "status_query"
            } else if message.contains("help") {
                return "help_request"
            } else if message.contains("who") || message.contains("what") {
                return "identity_query"
            }
            return "general"
        case .search:
            if message.contains("context") || message.contains("recent") {
                return "context_search"
            }
            return "knowledge_search"
        case .bridge:
            if message.contains("teleport") { return "teleportation" }
            if message.contains("entangle") { return "entanglement" }
            if message.contains("decohere") { return "decoherence_shield" }
            return "bridge_general"
        case .system:
            if message.contains("memory") { return "memory_check" }
            if message.contains("thermal") || message.contains("temp") { return "thermal_check" }
            if message.contains("cpu") { return "cpu_check" }
            return "system_general"
        case .engine:
            if message.contains("resonance") { return "resonance_engine" }
            if message.contains("evolution") { return "evolution_engine" }
            if message.contains("invention") { return "invention_engine" }
            return "engine_general"
        default:
            return "unclassified"
        }
    }

    /// Batch-classify multiple messages and return routing table
    func batchClassify(messages: [String]) -> [RouteResult] {
        return messages.map { classifyIntent(message: $0) }
    }

    /// Get the handler name for a domain (maps to handleXXXCommands)
    func handlerName(for domain: CommandDomain) -> String {
        switch domain {
        case .core: return "handleCoreCommands"
        case .search: return "handleSearchCommands"
        case .bridge: return "handleBridgeCommands"
        case .proto: return "handleProtocolCommands"
        case .system: return "handleSystemCommands"
        case .engine: return "handleEngineCommands"
        case .quantum: return "handleBridgeCommands"
        case .memory: return "handleCoreCommands"
        case .creative: return "handleCoreCommands"
        case .unknown: return "handleCoreCommands"
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// MARK: - 📊 GATE METRICS COLLECTOR
// Centralizes metrics gathering that was duplicated across multiple
// high-complexity gates. Single source of truth for live system
// metrics with φ-weighted caching.
// ═══════════════════════════════════════════════════════════════════

class GateMetricsCollector {
    static let shared = GateMetricsCollector()
    // PHI, TAU — use globals from L01_Constants

    private var metricsCache: [String: Any] = [:]
    private var cacheTimestamp: Double = 0
    private let cacheTTL: Double = 2.0  // seconds
    private let lock = NSLock()

    struct LiveMetrics {
        let uptime: Double
        let memoryUsedMB: Int
        let totalMemoryMB: Int
        let memoryPressure: Double
        let cpuLoad: Double
        let thermalState: String
        let threadCount: Int
        let gateHealthAvg: Double
        let responseLatencyAvg: Double
        let conversationCount: Int
    }

    /// Collect current system metrics with caching
    func collectMetrics() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        let now = Date().timeIntervalSince1970
        if now - cacheTimestamp < cacheTTL && !metricsCache.isEmpty {
            return metricsCache
        }

        let info = ProcessInfo.processInfo
        let physMem = info.physicalMemory
        let totalMB = Int(physMem / (1024 * 1024))
        let activeCPU = info.activeProcessorCount

        var metrics: [String: Any] = [
            "uptime_seconds": info.systemUptime,
            "total_memory_mb": totalMB,
            "active_cpus": activeCPU,
            "processor_count": info.processorCount,
            "os_version": info.operatingSystemVersionString,
            "host_name": info.hostName,
            "phi_coefficient": PHI,
            "tau_coefficient": TAU,
            "gate_health": LogicGateBreathingRoomEngine.shared.systemGateHealth(),
            "collection_timestamp": now,
        ]

        // Thermal state heuristic
        let thermal: String
        if activeCPU >= info.processorCount {
            thermal = "nominal"
        } else if activeCPU >= info.processorCount / 2 {
            thermal = "fair"
        } else {
            thermal = "serious"
        }
        metrics["thermal_state"] = thermal

        // Memory pressure estimate
        let estimatedUsed = totalMB / 2  // Conservative estimate
        let pressure = Double(estimatedUsed) / Double(max(1, totalMB))
        metrics["memory_pressure"] = pressure
        metrics["memory_used_mb"] = estimatedUsed

        metricsCache = metrics
        cacheTimestamp = now
        return metrics
    }

    /// Collect only the essential metrics for gate decisions
    func collectEssentialMetrics() -> (memPressure: Double, thermal: String, cpuLoad: Double) {
        let m = collectMetrics()
        return (
            memPressure: m["memory_pressure"] as? Double ?? 0.5,
            thermal: m["thermal_state"] as? String ?? "nominal",
            cpuLoad: Double(m["active_cpus"] as? Int ?? 1) / Double(max(1, m["processor_count"] as? Int ?? 1))
        )
    }

    /// Format metrics for display in gate reports
    func formatMetricsReport() -> String {
        let m = collectMetrics()
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║  GATE METRICS SNAPSHOT                                   ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Uptime: \(String(format: "%.0f", m["uptime_seconds"] as? Double ?? 0))s
        ║  Memory: \(m["memory_used_mb"] ?? 0)MB / \(m["total_memory_mb"] ?? 0)MB
        ║  Pressure: \(String(format: "%.1f%%", (m["memory_pressure"] as? Double ?? 0) * 100))
        ║  CPUs: \(m["active_cpus"] ?? 0) / \(m["processor_count"] ?? 0)
        ║  Thermal: \(m["thermal_state"] ?? "unknown")
        ║  Gate Health: \(String(format: "%.4f", m["gate_health"] as? Double ?? 0))
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    /// Invalidate cache (call after major state changes)
    func invalidateCache() {
        lock.lock()
        defer { lock.unlock() }
        metricsCache = [:]
        cacheTimestamp = 0
    }
}
