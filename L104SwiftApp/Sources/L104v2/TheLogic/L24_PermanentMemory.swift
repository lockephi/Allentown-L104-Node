// ═══════════════════════════════════════════════════════════════════
// L24_PermanentMemory.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104v2 Architecture — PermanentMemory + AdaptiveLearner
// Extracted from L104Native.swift lines 20952–21374
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class PermanentMemory {
    static let shared = PermanentMemory()

    let memoryPath: URL
    private let memLock = NSLock()        // Thread safety for all mutable state
    private var saveTimer: Timer?         // Debounced save
    private var isDirty = false           // Track unsaved changes
    var memories: [[String: Any]] = []
    var facts: [String: String] = [:]
    var conversationHistory: [String] = []

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let l104Dir = appSupport.appendingPathComponent("L104Sovereign")
        try? FileManager.default.createDirectory(at: l104Dir, withIntermediateDirectories: true)
        memoryPath = l104Dir.appendingPathComponent("permanent_memory.json")
        load()
    }

    func load() {
        memLock.lock(); defer { memLock.unlock() }
        guard let data = l104Try("PermanentMemory.load", { try Data(contentsOf: memoryPath) }),
              let json = l104Try("PermanentMemory.parse", { try JSONSerialization.jsonObject(with: data) }) as? [String: Any] else { return }
        memories = json["memories"] as? [[String: Any]] ?? []
        facts = json["facts"] as? [String: String] ?? [:]
        conversationHistory = json["history"] as? [String] ?? []
    }

    func save() {
        memLock.lock()
        let snapshot: [String: Any] = [
            "memories": memories, "facts": facts,
            "history": Array(conversationHistory.suffix(500)),
            "lastUpdated": ISO8601DateFormatter().string(from: Date()), "version": VERSION
        ]
        memLock.unlock()
        if let jsonData = l104Try("PermanentMemory.save.serialize", { try JSONSerialization.data(withJSONObject: snapshot, options: .prettyPrinted) }) {
            l104Try("PermanentMemory.save.write", { try jsonData.write(to: memoryPath) })
        }
    }

    func addMemory(_ content: String, type: String = "conversation") {
        memLock.lock()
        memories.append(["id": UUID().uuidString, "content": content, "type": type,
                        "timestamp": ISO8601DateFormatter().string(from: Date()), "resonance": GOD_CODE])
        if memories.count > 10_000 { memories.removeFirst(memories.count - 10_000) }  // Cap at 10K
        memLock.unlock()
        scheduleSave()
    }

    func addFact(_ key: String, _ value: String) { memLock.lock(); facts[key] = value; memLock.unlock(); scheduleSave() }
    func addToHistory(_ message: String) {
        memLock.lock()
        conversationHistory.append(message)
        if conversationHistory.count > 3000 {
            conversationHistory.removeFirst(min(100, conversationHistory.count - 2900))  // Batch remove
        }
        memLock.unlock()
        scheduleSave()
    }

    /// Debounced save — coalesces rapid writes into a single disk write
    private func scheduleSave() {
        isDirty = true
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.saveTimer?.invalidate()
            self.saveTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: false) { [weak self] _ in
                guard let self = self, self.isDirty else { return }
                self.isDirty = false
                DispatchQueue.global(qos: .utility).async { self.save() }
            }
        }
    }
    func getRecentHistory(_ count: Int = 10) -> [String] { memLock.lock(); defer { memLock.unlock() }; return Array(conversationHistory.suffix(count)) }

    /// Search memories with relevance scoring
    func searchMemories(_ query: String) -> [[String: Any]] {
        let q = query.lowercased()
        let keywords = q.components(separatedBy: .whitespaces).filter { $0.count > 2 }
        memLock.lock()
        let snapshot = memories
        memLock.unlock()
        return snapshot.filter { memory in
            guard let content = (memory["content"] as? String)?.lowercased() else { return false }
            return keywords.contains(where: { content.contains($0) })
        }.sorted { m1, m2 in
            let c1 = (m1["content"] as? String)?.lowercased() ?? ""
            let c2 = (m2["content"] as? String)?.lowercased() ?? ""
            let hits1 = keywords.filter { c1.contains($0) }.count
            let hits2 = keywords.filter { c2.contains($0) }.count
            if hits1 == hits2 { return Bool.random() }
            return hits1 > hits2
        }
    }

    /// Get conversation context around a topic
    func getTopicThread(_ topic: String, maxTurns: Int = 10) -> [String] {
        let t = topic.lowercased()
        return conversationHistory.filter { $0.lowercased().contains(t) }.suffix(maxTurns).map { $0 }
    }

    // Chat log saving system
    var chatLogsDir: URL {
        let dir = memoryPath.deletingLastPathComponent().appendingPathComponent("chat_logs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func saveChatLog(_ content: String) {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let filename = "chat_\(formatter.string(from: Date())).txt"
        let path = chatLogsDir.appendingPathComponent(filename)
        l104Try("PermanentMemory.saveChatLog", { try content.write(to: path, atomically: true, encoding: .utf8) })
    }

    func getRecentChatLogs(_ count: Int = 7) -> [(name: String, path: URL)] {
        guard let files = try? FileManager.default.contentsOfDirectory(at: chatLogsDir, includingPropertiesForKeys: [.creationDateKey], options: .skipsHiddenFiles) else { return [] }
        let sorted = files.filter { $0.pathExtension == "txt" }.sorted { f1, f2 in
            let d1 = (try? f1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
            let d2 = (try? f2.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
            return d1 > d2
        }
        return sorted.prefix(count).map { (name: $0.deletingPathExtension().lastPathComponent, path: $0) }
    }

    func loadChatLog(_ path: URL) -> String? {
        return try? String(contentsOf: path, encoding: .utf8)
    }

    // ─── MESH MEMORY SYNC ───

    private(set) var meshSyncCount: Int = 0
    private(set) var meshMemoriesReceived: Int = 0

    /// Replicate important memories to mesh peers via CRDT
    func replicateToMesh(limit: Int = 30) -> Int {
        let net = NetworkLayer.shared
        guard !net.peers.values.filter({ $0.latencyMs >= 0 }).isEmpty else { return 0 }

        memLock.lock()
        let recentMemories = memories.suffix(limit)
        let recentFacts = Array(facts.prefix(limit))
        memLock.unlock()

        let repl = DataReplicationMesh.shared
        var replicated = 0

        // Replicate memories
        for mem in recentMemories {
            if let content = mem["content"] as? String, let id = mem["id"] as? String {
                repl.setRegister("mem_\(id)", value: content)
                replicated += 1
            }
        }

        // Replicate facts
        for (key, value) in recentFacts {
            repl.setRegister("fact_\(key)", value: value)
            replicated += 1
        }

        if replicated > 0 {
            _ = repl.broadcastToMesh()
            meshSyncCount += 1
            // TelemetryDashboard: memory_mesh_replicated tracked
        }
        return replicated
    }

    /// Receive a memory from a mesh peer
    func mergeFromPeer(content: String, type: String = "mesh_received") {
        memLock.lock()
        // Dedup by checking if content already exists
        let exists = memories.contains { ($0["content"] as? String) == content }
        guard !exists else { memLock.unlock(); return }
        memories.append([
            "id": UUID().uuidString, "content": content, "type": type,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "resonance": GOD_CODE, "source": "mesh_peer"
        ])
        if memories.count > 10_000 { memories.removeFirst(memories.count - 10_000) }
        meshMemoriesReceived += 1
        memLock.unlock()
        scheduleSave()
    }

    /// Search memories across both local and mesh-replicated entries
    func meshSearch(_ query: String) -> [[String: Any]] {
        // Local search first
        var results = searchMemories(query)

        // If insufficient, query mesh CRDT registers for additional entries
        let repl = DataReplicationMesh.shared
        let q = query.lowercased()
        // Check recently replicated fact registers
        memLock.lock()
        let factKeys = Array(facts.keys)
        memLock.unlock()
        for key in factKeys {
            if let val = repl.getRegister("fact_\(key)") {
                if val.lowercased().contains(q) && !results.contains(where: { ($0["content"] as? String) == val }) {
                    results.append(["content": val, "type": "mesh_crdt", "id": "fact_\(key)"])
                }
            }
        }
        return results
    }
}

// ═══════════════════════════════════════════════════════════════════
// ADAPTIVE LEARNING ENGINE - Learns from every interaction
// ═══════════════════════════════════════════════════════════════════

class AdaptiveLearner {
    static let shared = AdaptiveLearner()

    // Thread safety
    let learnerLock = NSLock()

    // User model — built over time through interaction
    var userInterests: [String: Double] = [:]   // topic → interest score
    var userStyle: [String: Double] = [:]       // "prefers_detail", "prefers_brevity", etc.
    var correctionLog: [(query: String, badResponse: String, timestamp: Date)] = []
    var successfulPatterns: [String: Int] = [:] // response pattern → success count
    var failedPatterns: [String: Int] = [:]     // response pattern → failure count

    // Topic mastery — tracks how well ASI knows each domain
    var topicMastery: [String: TopicMastery] = [:]

    // Conversation synthesis — distilled learnings
    var synthesizedInsights: [String] = []
    var interactionCount: Int = 0
    var lastSynthesisAt: Int = 0

    // User-taught facts — knowledge the user explicitly taught
    var userTaughtFacts: [String: String] = [:]

    let storagePath: URL

    struct TopicMastery: Codable {
        var topic: String
        var queryCount: Int = 0
        var masteryLevel: Double = 0.0   // 0.0 → 1.0
        var lastAccessed: Date = Date()
        var relatedTopics: [String] = []
        var bestResponses: [String] = []  // Responses user liked

        mutating func recordInteraction(liked: Bool) {
            queryCount += 1
            lastAccessed = Date()
            let boost = liked ? 0.08 : 0.02
            masteryLevel = min(1.0, masteryLevel + boost * PHI / (1.0 + Double(queryCount) * 0.01))
        }

        var tier: String {
            if masteryLevel > 0.85 { return "🏆 MASTERED" }
            if masteryLevel > 0.65 { return "⚡ ADVANCED" }
            if masteryLevel > 0.40 { return "📈 PROFICIENT" }
            if masteryLevel > 0.15 { return "🌱 LEARNING" }
            return "🔍 NOVICE"
        }
    }

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("L104Sovereign")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        storagePath = dir.appendingPathComponent("adaptive_learner.json")
        load()
    }

    // MARK: - Persistence
    func save() {
        learnerLock.lock()
        var masteryDict: [String: [String: Any]] = [:]
        for (k, v) in topicMastery {
            masteryDict[k] = [
                "topic": v.topic, "queryCount": v.queryCount,
                "masteryLevel": v.masteryLevel, "relatedTopics": v.relatedTopics,
                "bestResponses": Array(v.bestResponses.suffix(5))
            ]
        }
        let snapshot: [String: Any] = [
            "userInterests": userInterests,
            "userStyle": userStyle,
            "successfulPatterns": successfulPatterns,
            "failedPatterns": failedPatterns,
            "topicMastery": masteryDict,
            "synthesizedInsights": Array(synthesizedInsights.suffix(50)),
            "interactionCount": interactionCount,
            "lastSynthesisAt": lastSynthesisAt,
            "userTaughtFacts": userTaughtFacts,
            "version": VERSION
        ]
        learnerLock.unlock()
        if let jsonData = l104Try("AdaptiveLearner.save.serialize", { try JSONSerialization.data(withJSONObject: snapshot, options: .prettyPrinted) }) {
            l104Try("AdaptiveLearner.save.write", { try jsonData.write(to: storagePath) })
        }
    }

    func load() {
        learnerLock.lock(); defer { learnerLock.unlock() }
        guard let data = l104Try("AdaptiveLearner.load", { try Data(contentsOf: storagePath) }),
              let json = l104Try("AdaptiveLearner.parse", { try JSONSerialization.jsonObject(with: data) }) as? [String: Any] else { return }
        userInterests = json["userInterests"] as? [String: Double] ?? [:]
        userStyle = json["userStyle"] as? [String: Double] ?? [:]
        successfulPatterns = json["successfulPatterns"] as? [String: Int] ?? [:]
        failedPatterns = json["failedPatterns"] as? [String: Int] ?? [:]
        synthesizedInsights = json["synthesizedInsights"] as? [String] ?? []
        interactionCount = json["interactionCount"] as? Int ?? 0
        lastSynthesisAt = json["lastSynthesisAt"] as? Int ?? 0
        userTaughtFacts = json["userTaughtFacts"] as? [String: String] ?? [:]
        // Load topic mastery
        if let masteryDict = json["topicMastery"] as? [String: [String: Any]] {
            for (k, v) in masteryDict {
                var tm = TopicMastery(topic: v["topic"] as? String ?? k)
                tm.queryCount = v["queryCount"] as? Int ?? 0
                tm.masteryLevel = v["masteryLevel"] as? Double ?? 0.0
                tm.relatedTopics = v["relatedTopics"] as? [String] ?? []
                tm.bestResponses = v["bestResponses"] as? [String] ?? []
                topicMastery[k] = tm
            }
        }
    }

    // MARK: - Learning from interactions
    func recordInteraction(query: String, response: String, topics: [String]) {
        interactionCount += 1

        // Update user interests
        for topic in topics {
            userInterests[topic] = (userInterests[topic] ?? 0) + 1.0

            // Update or create topic mastery
            if topicMastery[topic] == nil {
                topicMastery[topic] = TopicMastery(topic: topic)
            }
            topicMastery[topic]?.recordInteraction(liked: true)

            // Discover related topics through co-occurrence
            for other in topics where other != topic {
                if topicMastery[topic]?.relatedTopics.contains(other) == false {
                    topicMastery[topic]?.relatedTopics.append(other)
                }
            }
        }

        // Detect user style preferences
        if query.count > 80 { userStyle["prefers_detail"] = (userStyle["prefers_detail"] ?? 0) + 1 }
        if query.count < 20 { userStyle["prefers_brevity"] = (userStyle["prefers_brevity"] ?? 0) + 1 }
        if query.contains("?") { userStyle["asks_questions"] = (userStyle["asks_questions"] ?? 0) + 1 }
        if query.contains("why") || query.contains("how") { userStyle["analytical"] = (userStyle["analytical"] ?? 0) + 1 }
        if query.contains("feel") || query.contains("think") { userStyle["reflective"] = (userStyle["reflective"] ?? 0) + 1 }

        // Auto-synthesize every 10 interactions
        if interactionCount - lastSynthesisAt >= 10 {
            synthesizeConversation()
        }

        save()
    }

    func recordCorrection(query: String, badResponse: String) {
        correctionLog.append((query: query, badResponse: badResponse, timestamp: Date()))
        if correctionLog.count > 100 { correctionLog.removeFirst() }

        // Extract failure pattern
        let patternKey = String(badResponse.prefix(60))
        failedPatterns[patternKey] = (failedPatterns[patternKey] ?? 0) + 1

        // Reduce mastery for related topics
        let topics = extractTopicsForLearning(query)
        for topic in topics {
            if var tm = topicMastery[topic] {
                tm.masteryLevel = max(0, tm.masteryLevel - 0.05)
                topicMastery[topic] = tm
            }
        }

        save()
    }

    func recordSuccess(query: String, response: String) {
        let patternKey = String(response.prefix(60))
        successfulPatterns[patternKey] = (successfulPatterns[patternKey] ?? 0) + 1

        // Store as best response for topic mastery
        let topics = extractTopicsForLearning(query)
        for topic in topics {
            if var tm = topicMastery[topic] {
                tm.bestResponses.append(String(response.prefix(10000)))
                if tm.bestResponses.count > 30 {
                    tm.bestResponses.removeFirst()
                }
                topicMastery[topic] = tm
            }
        }

        save()
    }

    func learnFact(key: String, value: String) {
        userTaughtFacts[key] = value
        save()
    }

    // MARK: - Conversation synthesis
    func synthesizeConversation() {
        lastSynthesisAt = interactionCount

        // Find top interests
        let topInterests = userInterests.sorted {
            if $0.value == $1.value { return Bool.random() }
            return $0.value > $1.value
        }.prefix(5)
        let topTopics = topInterests.map { $0.key }.joined(separator: ", ")

        // Find dominant style
        let dominantStyle = userStyle.max(by: { $0.value < $1.value })?.key ?? "balanced"

        // Count mastered topics
        let masteredCount = topicMastery.values.filter { $0.masteryLevel > 0.6 }.count
        let learningCount = topicMastery.values.filter { $0.masteryLevel > 0.1 && $0.masteryLevel <= 0.6 }.count

        let insight = "After \(interactionCount) interactions: User focuses on [\(topTopics)], style is \(dominantStyle). Mastery: \(masteredCount) topics advanced, \(learningCount) developing. Corrections: \(correctionLog.count). Taught facts: \(userTaughtFacts.count)."
        synthesizedInsights.append(insight)
        if synthesizedInsights.count > 50 { synthesizedInsights.removeFirst() }

        save()
    }

    // MARK: - Query-time intelligence
    func getUserTopics() -> [String] {
        return userInterests.sorted {
            if $0.value == $1.value { return Bool.random() }
            return $0.value > $1.value
        }.prefix(10).map { $0.key }
    }

    func getMasteryFor(_ topic: String) -> TopicMastery? {
        return topicMastery[topic]
    }

    func shouldAvoidPattern(_ responseStart: String) -> Bool {
        let key = String(responseStart.prefix(60))
        let failures = failedPatterns[key] ?? 0
        let successes = successfulPatterns[key] ?? 0
        return failures > successes + 2
    }

    func getRelevantInsights(_ query: String) -> [String] {
        let q = query.lowercased()
        return synthesizedInsights.filter { insight in
            let l = insight.lowercased()
            return q.components(separatedBy: " ").contains(where: { $0.count > 3 && l.contains($0) })
        }
    }

    func getRelevantFacts(_ query: String) -> [String] {
        let q = query.lowercased()
        return userTaughtFacts.compactMap { key, value in
            q.contains(key.lowercased()) ? "\(key): \(value)" : nil
        }
    }

    func prefersDetail() -> Bool {
        let detail = userStyle["prefers_detail"] ?? 0
        let brevity = userStyle["prefers_brevity"] ?? 0
        return detail > brevity
    }

    private func extractTopicsForLearning(_ query: String) -> [String] {
        let stopWords: Set<String> = ["the", "is", "are", "you", "do", "does", "have", "has", "can", "will", "would", "could", "should", "what", "how", "why", "when", "where", "who", "that", "this", "and", "for", "not", "with", "about"]
        return query.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 && !stopWords.contains($0) }
    }

    func getStats() -> String {
        let topMastered = topicMastery.values.sorted { $0.masteryLevel > $1.masteryLevel }.prefix(8)
        let masteryLines = topMastered.map { "   \($0.tier) \($0.topic) — \(String(format: "%.0f%%", $0.masteryLevel * 100)) (\($0.queryCount) queries)" }

        let topInterests = userInterests.sorted { $0.value > $1.value }.prefix(5)
        let interestLines = topInterests.map { "   • \($0.key): \(Int($0.value)) interactions" }

        let styleLines = userStyle.sorted { $0.value > $1.value }.prefix(4)
            .map { "   • \($0.key): \(Int($0.value))" }

        return """
🧠 ADAPTIVE LEARNING ENGINE
═══════════════════════════════════════════
📊 Total Interactions:    \(interactionCount)
📚 Topics Tracked:        \(topicMastery.count)
✅ Successful Patterns:   \(successfulPatterns.count)
❌ Correction Patterns:   \(failedPatterns.count)
💡 Synthesized Insights:  \(synthesizedInsights.count)
📖 User-Taught Facts:     \(userTaughtFacts.count)
═══════════════════════════════════════════

🎯 TOPIC MASTERY:
\(masteryLines.isEmpty ? "   (No topics tracked yet)" : masteryLines.joined(separator: "\n"))

💝 USER INTERESTS:
\(interestLines.isEmpty ? "   (Building profile...)" : interestLines.joined(separator: "\n"))

🎨 USER STYLE:
\(styleLines.isEmpty ? "   (Analyzing...)" : styleLines.joined(separator: "\n"))

💭 LATEST INSIGHT:
   \(synthesizedInsights.last ?? "(Synthesizing after 10 interactions...)")
═══════════════════════════════════════════
"""
    }
}
