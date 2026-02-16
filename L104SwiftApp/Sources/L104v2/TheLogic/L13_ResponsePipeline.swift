// ═══════════════════════════════════════════════════════════════════
// L13_ResponsePipeline.swift
// L104v2 — EVO_55 Pipeline-Integrated Response System V2
//   ResponsePipelineOptimizer (upgraded: adaptive TTL, φ-decay eviction)
//   ResponseConfidenceEngine (upgraded: multi-signal fusion)
//   ResponsePlanner (upgraded: deeper plan templates)
//
// Response pipeline: caching, quality scoring, confidence reporting,
// and multi-turn response planning. Streams through EVO_55 unified pipeline.
// Upgraded: Feb 15, 2026 — Sovereign Unification
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class ResponsePipelineOptimizer {
    static let shared = ResponsePipelineOptimizer()
    // PHI, TAU — use globals from L01_Constants

    private var responseCache: [String: (response: String, timestamp: Double, quality: Double)] = [:]
    private let maxCacheSize = PIPELINE_MAX_CACHE   // EVO_55: use unified constant (1000)
    private let cacheTTL: Double = PIPELINE_CACHE_TTL  // EVO_55: use unified constant (15s)
    private let lock = NSLock()
    private var cacheHits: Int = 0
    private var cacheMisses: Int = 0
    private var totalEvictions: Int = 0
    private var adaptiveTTLMultiplier: Double = 1.0  // EVO_55: adapts based on hit rate

    /// Check for cached response (similarity-based lookup)
    func getCachedResponse(query: String) -> String? {
        lock.lock()
        defer { lock.unlock() }

        let now = Date().timeIntervalSince1970
        let key = normalizeQuery(query)
        let effectiveTTL = cacheTTL * adaptiveTTLMultiplier

        if let cached = responseCache[key], now - cached.timestamp < effectiveTTL {
            cacheHits += 1
            adaptTTL()  // EVO_55: boost TTL when hit rate is high
            return cached.response
        }
        cacheMisses += 1
        adaptTTL()
        return nil
    }

    /// Cache a response with quality score for φ-decay eviction
    func cacheResponse(query: String, response: String) {
        lock.lock()
        defer { lock.unlock() }

        let quality = _scoreResponseUnlocked(response, query: query)

        if responseCache.count >= maxCacheSize {
            // EVO_55: φ-decay eviction — evict lowest-quality oldest entries
            let now = Date().timeIntervalSince1970
            let scored = responseCache.map { (key: $0.key, age: now - $0.value.timestamp, quality: $0.value.quality) }
            let sorted = scored.sorted { a, b in
                // Lower quality × older = more evictable
                let scoreA = a.quality * pow(TAU, a.age / cacheTTL)  // φ-decay
                let scoreB = b.quality * pow(TAU, b.age / cacheTTL)
                return scoreA < scoreB
            }
            let toRemove = sorted.prefix(maxCacheSize / 4)
            for item in toRemove {
                responseCache.removeValue(forKey: item.key)
            }
            totalEvictions += toRemove.count
        }

        let key = normalizeQuery(query)
        responseCache[key] = (response: response, timestamp: Date().timeIntervalSince1970, quality: quality)
    }

    /// Normalize a query for cache lookup
    private func normalizeQuery(_ query: String) -> String {
        return query.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
    }

    /// EVO_55: Adaptive TTL — boost cache duration when hit rate is high
    private func adaptTTL() {
        let total = cacheHits + cacheMisses
        guard total > 20 else { return }  // need enough samples
        let hitRate = Double(cacheHits) / Double(total)
        // High hit rate → extend TTL (up to 2x); low hit rate → shrink
        adaptiveTTLMultiplier = max(0.5, min(2.0, 0.5 + hitRate * PHI))
    }

    /// Score a response on multiple quality dimensions (thread-safe wrapper)
    func scoreResponse(_ response: String, query: String) -> Double {
        lock.lock()
        defer { lock.unlock() }
        return _scoreResponseUnlocked(response, query: query)
    }

    /// Internal scoring — call only while holding lock
    private func _scoreResponseUnlocked(_ response: String, query: String) -> Double {
        var score: Double = 0.5

        // Length appropriateness
        let wordCount = response.split(separator: " ").count
        if wordCount >= 10 && wordCount <= 300 {
            score += 0.1
        } else if wordCount < 5 {
            score -= 0.15
        }

        // Relevance — keyword overlap
        let queryWords = Set(query.lowercased().split(separator: " ").map(String.init))
        let responseWords = Set(response.lowercased().split(separator: " ").map(String.init))
        let overlap = Double(queryWords.intersection(responseWords).count)
        let relevance = overlap / Double(max(1, queryWords.count))
        score += relevance * TAU * 0.3

        // Formatting quality
        if response.contains("\n") { score += 0.02 }

        // Coherence — sentence count vs word count ratio
        let sentences = response.split(separator: ".").count
        let coherenceRatio = Double(sentences) / Double(max(1, wordCount))
        if coherenceRatio > 0.05 && coherenceRatio < 0.3 {
            score += 0.05
        }

        return min(1.0, max(0.0, score * PHI))
    }

    /// Select best response from candidates using φ-weighted scoring
    func selectBestResponse(candidates: [(String, Double)], query: String) -> String {
        guard !candidates.isEmpty else { return "I need more context to answer that." }

        var best: (response: String, score: Double) = ("", -1.0)
        for (response, baseScore) in candidates {
            let qualityScore = scoreResponse(response, query: query)
            let combined = baseScore * PHI + qualityScore * TAU
            if combined > best.score {
                best = (response, combined)
            }
        }

        return best.response
    }

    /// Cache statistics (EVO_55: includes eviction + adaptive TTL data)
    func cacheStats() -> (size: Int, maxSize: Int, hitRate: Double) {
        let total = cacheHits + cacheMisses
        let rate = total > 0 ? Double(cacheHits) / Double(total) : 0.0
        return (responseCache.count, maxCacheSize, rate)
    }

    func engineStatus() -> [String: Any] {
        let (size, maxSize, hitRate) = cacheStats()
        return [
            "cache_size": size,
            "max_size": maxSize,
            "hit_rate": hitRate,
            "evictions": totalEvictions,
            "adaptive_ttl_multiplier": adaptiveTTLMultiplier,
            "effective_ttl": cacheTTL * adaptiveTTLMultiplier,
            "mesh_hits": meshCacheHits,
            "mesh_broadcasts": meshCacheBroadcasts
        ]
    }

    /// EVO_55: φ-weighted health — cache utilization, hit rate, eviction pressure
    func engineHealth() -> Double {
        let total = cacheHits + cacheMisses
        let hitRate = total > 0 ? Double(cacheHits) / Double(total) : 0.0
        let cacheUtilization = Double(responseCache.count) / Double(maxCacheSize)
        let evictionPressure = total > 0 ? 1.0 - min(1.0, Double(totalEvictions) / Double(max(1, total))) : 1.0
        return min(1.0, max(0.1,
            hitRate * 0.3 +
            cacheUtilization * 0.2 +
            evictionPressure * 0.2 +
            adaptiveTTLMultiplier * 0.2 +
            0.1  // base health
        ))
    }

    // ═══ MESH-DISTRIBUTED RESPONSE ROUTING ═══

    private var meshCacheHits: Int = 0
    private var meshCacheBroadcasts: Int = 0

    /// Check mesh CRDT for cached response from other peers
    func getMeshCachedResponse(query: String) -> String? {
        let net = NetworkLayer.shared
        guard net.isActive && !net.peers.isEmpty else { return nil }

        let key = "resp_\(fnvHash(normalizeQuery(query)))"
        let repl = DataReplicationMesh.shared
        if let cached = repl.getRegister(key), !cached.isEmpty {
            meshCacheHits += 1
            return cached
        }
        return nil
    }

    /// Broadcast a high-quality response to mesh for peer caching
    func broadcastResponseToMesh(query: String, response: String) {
        let net = NetworkLayer.shared
        guard net.isActive && !net.quantumLinks.isEmpty else { return }

        // Only broadcast high-quality responses
        let score = scoreResponse(response, query: query)
        guard score > 0.6 else { return }

        let key = "resp_\(fnvHash(normalizeQuery(query)))"
        let repl = DataReplicationMesh.shared
        // Truncate to reasonable mesh size
        let truncated = String(response.prefix(500))
        repl.setRegister(key, value: truncated)
        _ = repl.broadcastToMesh()
        meshCacheBroadcasts += 1

        TelemetryDashboard.shared.record(metric: "response_mesh_broadcast", value: 1.0)
    }

    /// Route query to best peer (lowest latency, quantum-linked preferred)
    func routeQueryToBestPeer(_ query: String) -> String? {
        let net = NetworkLayer.shared
        guard net.isActive else { return nil }

        // Find best quantum-linked peer
        var bestPeer: NetworkLayer.Peer? = nil
        var bestLatency: Double = Double.infinity

        for (_, peer) in net.peers where peer.latencyMs >= 0 {
            if peer.isQuantumLinked && peer.latencyMs < bestLatency {
                bestPeer = peer
                bestLatency = peer.latencyMs
            }
        }

        guard let peer = bestPeer else { return nil }

        // Check if peer has cached this response in CRDT
        let key = "resp_\(fnvHash(normalizeQuery(query)))"
        let repl = DataReplicationMesh.shared
        if let peerResponse = repl.getRegister(key) {
            TelemetryDashboard.shared.record(metric: "response_mesh_route", value: 1.0)
            return "[\(peer.name)] \(peerResponse)"
        }

        return nil
    }

    /// FNV-1a hash for cache keys
    private func fnvHash(_ text: String) -> UInt64 {
        var hash: UInt64 = 14695981039346656037
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        return hash
    }

    /// Extended cache stats including mesh
    var meshStats: [String: Any] {
        let (size, maxSize, hitRate) = cacheStats()
        return [
            "cache_size": size,
            "max_size": maxSize,
            "hit_rate": hitRate,
            "mesh_hits": meshCacheHits,
            "mesh_broadcasts": meshCacheBroadcasts
        ]
    }
}


// ═══════════════════════════════════════════════════════════════════
// RESPONSE CONFIDENCE ENGINE — Multi-level confidence scoring
// ═══════════════════════════════════════════════════════════════════

class ResponseConfidenceEngine {
    static let shared = ResponseConfidenceEngine()

    struct ConfidenceReport {
        let level: ConfidenceLevel
        let score: Double              // 0.0 - 1.0
        let kbMatchQuality: Double     // Best fragment relevance
        let sourceCount: Int           // Number of corroborating sources
        let sourceType: SourceType     // Where the info came from
        let categories: [String]       // KB categories that contributed
        let reasoning: String          // Why this confidence level

        enum ConfidenceLevel: String {
            case verified = "Verified"       // Multiple high-quality sources
            case high = "High"               // Strong single source or computation
            case moderate = "Moderate"        // Partial match or evolved content
            case speculative = "Speculative" // Template/synthesis, low match
            case uncertain = "Uncertain"     // Fallback, no real data
        }

        enum SourceType: String {
            case computation = "Direct Computation"
            case knowledgeBase = "Knowledge Base"
            case userTaught = "User-Taught"
            case evolved = "Cross-Domain Synthesis"
            case reasoning = "Logical Reasoning"
            case template = "Template Response"
        }

        var indicator: String {
            switch level {
            case .verified:    return "🟢 Verified"
            case .high:        return "🔵 High Confidence"
            case .moderate:    return "🟡 Moderate Confidence"
            case .speculative: return "🟠 Speculative"
            case .uncertain:   return "🔴 Low Confidence"
            }
        }

        var footer: String {
            var parts: [String] = []
            parts.append("[\(indicator) · \(sourceType.rawValue)]")
            if sourceCount > 1 { parts.append("Corroborated by \(sourceCount) sources") }
            if !categories.isEmpty { parts.append("Domains: \(categories.prefix(3).joined(separator: ", "))") }
            return parts.joined(separator: " · ")
        }
    }

    func score(
        kbFragments: [(text: String, relevance: Double, category: String)],
        isComputed: Bool = false,
        isUserTaught: Bool = false,
        isEvolved: Bool = false,
        queryKeywordHits: Int = 0,
        totalQueryKeywords: Int = 1
    ) -> ConfidenceReport {
        // Computation results are inherently high confidence
        if isComputed {
            return ConfidenceReport(
                level: .verified, score: 0.95, kbMatchQuality: 1.0,
                sourceCount: 1, sourceType: .computation,
                categories: ["computation"], reasoning: "Direct mathematical/scientific computation"
            )
        }

        if isUserTaught {
            return ConfidenceReport(
                level: .high, score: 0.85, kbMatchQuality: 0.9,
                sourceCount: 1, sourceType: .userTaught,
                categories: ["user-taught"], reasoning: "From information you provided"
            )
        }

        guard !kbFragments.isEmpty else {
            return ConfidenceReport(
                level: isEvolved ? .speculative : .uncertain,
                score: isEvolved ? 0.35 : 0.15,
                kbMatchQuality: 0.0, sourceCount: 0,
                sourceType: isEvolved ? .evolved : .template,
                categories: [], reasoning: isEvolved ? "Synthesized from cross-domain analysis" : "No direct knowledge match"
            )
        }

        let bestRelevance = kbFragments.map(\.relevance).max() ?? 0
        let avgRelevance = kbFragments.map(\.relevance).reduce(0, +) / Double(kbFragments.count)
        let categories = Array(Set(kbFragments.map(\.category)))
        let keywordCoverage = totalQueryKeywords > 0 ? Double(queryKeywordHits) / Double(totalQueryKeywords) : 0

        let rawScore = (bestRelevance * 0.35) + (avgRelevance * 0.25) +
                       (min(1.0, Double(kbFragments.count) / 5.0) * 0.2) +
                       (keywordCoverage * 0.2)

        let level: ConfidenceReport.ConfidenceLevel
        switch rawScore {
        case 0.75...: level = .verified
        case 0.55..<0.75: level = .high
        case 0.35..<0.55: level = .moderate
        case 0.15..<0.35: level = .speculative
        default: level = .uncertain
        }

        return ConfidenceReport(
            level: level, score: min(1.0, rawScore),
            kbMatchQuality: bestRelevance, sourceCount: kbFragments.count,
            sourceType: .knowledgeBase, categories: categories,
            reasoning: "Based on \(kbFragments.count) knowledge entries with \(String(format: "%.0f%%", keywordCoverage * 100)) keyword coverage"
        )
    }
}


// ═══════════════════════════════════════════════════════════════════
// MULTI-TURN RESPONSE PLANNER — Structured exploration with plan tracking
// ═══════════════════════════════════════════════════════════════════

class ResponsePlanner {
    static let shared = ResponsePlanner()

    struct Plan {
        let topic: String
        let sections: [PlanSection]
        var currentIndex: Int = 0
        let createdAt: Date = Date()

        struct PlanSection {
            let title: String
            let prompt: String     // What to search/generate for
            let depth: String      // "overview", "detailed", "expert"
        }

        var isComplete: Bool { currentIndex >= sections.count }
        var currentSection: PlanSection? {
            guard currentIndex < sections.count else { return nil }
            return sections[currentIndex]
        }

        var overview: String {
            let items = sections.enumerated().map { idx, sec in
                let marker = idx == currentIndex ? "→" : (idx < currentIndex ? "✓" : "○")
                return "\(marker) \(idx + 1). \(sec.title)"
            }.joined(separator: "\n")
            return "📋 **Exploration Plan: \(topic)**\n\(items)"
        }
    }

    private var activePlan: Plan?
    private var planHistory: [Plan] = []

    /// Determine if a query warrants a multi-turn plan
    func shouldPlan(_ query: String) -> Bool {
        let q = query.lowercased()
        let complexMarkers = ["explain", "teach me", "deep dive", "everything about",
                              "comprehensive", "thorough", "full analysis", "break down",
                              "walk me through", "guide me through", "all about"]
        let topicWords = query.components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 3 }
        return complexMarkers.contains(where: { q.contains($0) }) || topicWords.count >= 4
    }

    /// Create a structured exploration plan for a complex topic
    func createPlan(for topic: String, query: String) -> Plan {
        let sections: [Plan.PlanSection]
        let q = query.lowercased()

        // Detect domain-specific plan templates
        if q.contains("history") || q.contains("evolution of") || q.contains("how did") {
            sections = [
                .init(title: "Origins & Early Development", prompt: "origins history early development of \(topic)", depth: "detailed"),
                .init(title: "Key Milestones & Breakthroughs", prompt: "major milestones breakthroughs discoveries in \(topic)", depth: "detailed"),
                .init(title: "Modern State & Current Understanding", prompt: "current state modern understanding of \(topic)", depth: "detailed"),
                .init(title: "Future Directions & Open Questions", prompt: "future directions open questions in \(topic)", depth: "expert"),
            ]
        } else if q.contains("how does") || q.contains("mechanism") || q.contains("how works") {
            sections = [
                .init(title: "Core Mechanism", prompt: "core mechanism fundamental principle of \(topic)", depth: "detailed"),
                .init(title: "Key Components & Interactions", prompt: "components parts interactions in \(topic)", depth: "detailed"),
                .init(title: "Real-World Applications", prompt: "applications examples uses of \(topic)", depth: "detailed"),
                .init(title: "Edge Cases & Limitations", prompt: "limitations edge cases challenges of \(topic)", depth: "expert"),
            ]
        } else {
            // General comprehensive plan
            sections = [
                .init(title: "Definition & Core Concepts", prompt: "what is definition core concepts of \(topic)", depth: "standard"),
                .init(title: "Deep Analysis", prompt: "deep analysis detailed explanation of \(topic)", depth: "detailed"),
                .init(title: "Connections & Implications", prompt: "connections implications relationships of \(topic)", depth: "detailed"),
                .init(title: "Critical Perspectives & Open Questions", prompt: "critical analysis debate open questions about \(topic)", depth: "expert"),
                .init(title: "Synthesis & Key Takeaways", prompt: "synthesis summary key insights about \(topic)", depth: "expert"),
            ]
        }

        let plan = Plan(topic: topic, sections: sections)
        activePlan = plan
        return plan
    }

    /// Advance to next section in active plan
    func advancePlan() -> Plan.PlanSection? {
        guard var plan = activePlan else { return nil }
        plan.currentIndex += 1
        if plan.isComplete {
            planHistory.append(plan)
            activePlan = nil
            return nil
        }
        activePlan = plan
        return plan.currentSection
    }

    var hasActivePlan: Bool { activePlan != nil && !(activePlan?.isComplete ?? true) }
    var currentPlan: Plan? { activePlan }

    func clearPlan() {
        if let plan = activePlan { planHistory.append(plan) }
        activePlan = nil
    }
}
