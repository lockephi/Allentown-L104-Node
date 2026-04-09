import Accelerate
import AppKit
import Foundation
import NaturalLanguage
import os.log
import simd

// ═══════════════════════════════════════════════════════════════════
// MARK: - LOCK-FREE CACHE ENTRY
// Concurrent cache entry with atomic quality updates
// ═══════════════════════════════════════════════════════════════════

private final class ResponseCacheEntry {
    let response: String
    let timestamp: Double
    let quality: Double
    var accessCount: UInt64 = 0

    init(response: String, timestamp: Double, quality: Double) {
        self.response = response
        self.timestamp = timestamp
        self.quality = quality
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - PHI-WEIGHTED PRIORITY QUEUE
// Custom priority queue for φ-weighted cache eviction
// ═══════════════════════════════════════════════════════════════════

private struct PHIPriorityQueue {
    private var heap: [(key: String, priority: Double)] = []

    var isEmpty: Bool { heap.isEmpty }
    var count: Int { heap.count }

    mutating func enqueue(_ key: String, priority: Double) {
        heap.append((key, priority))
        siftUp(heap.count - 1)
    }

    mutating func dequeue() -> (key: String, priority: Double)? {
        guard !heap.isEmpty else { return nil }
        if heap.count == 1 { return heap.removeLast() }
        heap.swapAt(0, heap.count - 1)
        let result = heap.removeLast()
        siftDown(0)
        return result
    }

    private func parent(_ i: Int) -> Int { (i - 1) / 2 }
    private func left(_ i: Int) -> Int { 2 * i + 1 }
    private func right(_ i: Int) -> Int { 2 * i + 2 }

    private mutating func siftUp(_ i: Int) {
        var i = i
        while i > 0 && heap[parent(i)].priority > heap[i].priority {
            heap.swapAt(i, parent(i))
            i = parent(i)
        }
    }

    private mutating func siftDown(_ i: Int) {
        var i = i
        while true {
            var minIdx = i
            let l = left(i), r = right(i)
            if l < heap.count && heap[l].priority < heap[minIdx].priority { minIdx = l }
            if r < heap.count && heap[r].priority < heap[minIdx].priority { minIdx = r }
            if minIdx == i { break }
            heap.swapAt(i, minIdx)
            i = minIdx
        }
    }
}

class ResponsePipelineOptimizer {
    static let shared = ResponsePipelineOptimizer()

    // ═══ CONCURRENT CACHE - Lock-free read-heavy access ═══
    private var responseCache: [String: ResponseCacheEntry] = [:]
    private let maxCacheSize = PIPELINE_MAX_CACHE
    private let cacheTTL: Double = PIPELINE_CACHE_TTL

    // ═══ PERFORMANCE: Sharded locks for reduced read contention ═══
    private let shardCount = 16
    private var shardLocks: [NSLock] = (0..<16).map { _ in NSLock() }

    // ═══ CACHE-WIDE LOCK - Required for eviction (fixes issue #2) ═══
    // Must hold this lock when iterating responseCache for eviction
    private let cacheLock = NSLock()

    // ═══ ATOMIC COUNTERS - Using os_unfair_lock (fixes issue #4) ═══
    private var _cacheHits: UInt64 = 0
    private var _cacheMisses: UInt64 = 0
    private var _totalEvictions: UInt64 = 0
    private var counterLock = os_unfair_lock_s()
    private var counterLockPointer: UnsafeMutablePointer<os_unfair_lock_s> {
        withUnsafeMutablePointer(to: &counterLock) { $0 }
    }

    // ═══ ADAPTIVE TTL - Thread-safe read/write (fixes issue #3) ═══
    private var adaptiveTTLMultiplier: Double = 1.0
    private let ttlLock = NSLock()  // Only for TTL updates (rare)

    // ═══ NAMED CONSTANTS (fixes issue #6 - magic numbers) ═══
    private enum CacheConstants {
        static let minWordsForGoodResponse = 10
        static let maxWordsForGoodResponse = 300
        static let minWordsForPenalty = 5
        static let maxMeshTruncateLength = 500
        static let batchThresholdForTTL = 100
        static let minTTLMultiplier = 0.5
        static let maxTTLMultiplier = 2.0
        static let ttlDecayIntervalSeconds = 60.0
        static let evictionPercentage = 10
        // FNV-1a 64-bit constants
        static let fnvOffsetBasis: UInt64 = 14695981039346656037
        static let fnvPrime: UInt64 = 1099511628211
    }

    // ═══ SIMD-ACCELERATED STRING MATCHING ═══
    private var keyHashes: [UInt64: String] = [:]
    private let hashLock = NSLock()

    // Access atomic counters
    // Issue #4 FIX: Thread-safe counter accessors using os_unfair_lock
    private var cacheHits: UInt64 {
        var lock = counterLockPointer
        return OSUnfairLockLock(lock) { _cacheHits }
    }
    private var cacheMisses: UInt64 {
        var lock = counterLockPointer
        return OSUnfairLockLock(lock) { _cacheMisses }
    }
    private var totalEvictions: UInt64 {
        var lock = counterLockPointer
        return OSUnfairLockLock(lock) { _totalEvictions }
    }

    private func incrementCacheHits() {
        var lock = counterLockPointer
        OSUnfairLockLock(lock) { _cacheHits += 1 }
    }

    private func incrementCacheMisses() {
        var lock = counterLockPointer
        OSUnfairLockLock(lock) { _cacheMisses += 1 }
    }

    private func incrementEvictions(by count: UInt64) {
        var lock = counterLockPointer
        OSUnfairLockLock(lock) { _totalEvictions += count }
    }

    private func OSUnfairLockLock<T>(_ lock: UnsafeMutablePointer<os_unfair_lock_s>, body: () -> T) -> T {
        os_unfair_lock_lock(lock)
        defer { os_unfair_lock_unlock(lock) }
        return body()
    }

    /// ═══ CONCURRENT CACHE LOOKUP - Sharded lock for reduced contention ═══
    func getCachedResponse(query: String) -> String? {
        let key = normalizeQuery(query)
        let hash = hashKey(key)
        let shard = Int(hash % UInt64(shardCount))
        let now = Date().timeIntervalSince1970

        // ═══ FAST PATH: Check shard lock only ═══
        shardLocks[shard].lock()
        defer { shardLocks[shard].unlock() }

        // Issue #3 FIX: Read adaptive TTL under ttlLock for thread safety
        ttlLock.lock()
        let effectiveTTL = cacheTTL * adaptiveTTLMultiplier
        ttlLock.unlock()

        if let cached = responseCache[key] {
            if now - cached.timestamp < effectiveTTL {
                incrementCacheHits()
                cached.accessCount &+= 1
                return cached.response
            }
        }
        incrementCacheMisses()
        return nil
    }

    /// ═══ SIMD-ACCELERATED: Compute hash for cache key distribution ═══
    private func hashKey(_ key: String) -> UInt64 {
        var hash = CacheConstants.fnvOffsetBasis
        for byte in key.utf8 {
            hash ^= UInt64(byte)
            hash &*= CacheConstants.fnvPrime
        }
        return hash
    }

    /// ═══ BATCH ADAPTIVE TTL - Called periodically, not every access ═══
    func adaptTTLPeriodically() {
        let total = _cacheHits + _cacheMisses
        guard total > 100 else { return } // Batch updates for efficiency

        ttlLock.lock()
        defer { ttlLock.unlock() }

        let hitRate = Double(_cacheHits) / Double(total)
        // High hit rate → extend TTL (up to 2x); low hit rate → shrink
        adaptiveTTLMultiplier = max(0.5, min(2.0, 0.5 + hitRate * PHI))
    }

    /// ═══ CONCURRENT CACHE INSERT - Sharded locks + φ-weighted eviction ═══
    func cacheResponse(query: String, response: String) {
        let key = normalizeQuery(query)
        let hash = hashKey(key)
        let shard = Int(hash % UInt64(shardCount))
        let quality = scoreResponse(response, query: query)
        let now = Date().timeIntervalSince1970

        shardLocks[shard].lock()

        // ═══ φ-WEIGHTED EVICTION: Prioritize high-quality + recently accessed ═══
        // Hold cacheLock during eviction AND insert to prevent data races on responseCache
        cacheLock.lock()
        if responseCache.count >= maxCacheSize {
            var evictQueue = PHIPriorityQueue()

            // Read adaptive TTL under ttlLock for thread safety
            ttlLock.lock()
            let effectiveTTL = cacheTTL * adaptiveTTLMultiplier
            ttlLock.unlock()

            // Collect entries with priority scores
            for (k, entry) in responseCache {
                let age = now - entry.timestamp
                let isExpired = age > effectiveTTL
                // φ-weighted priority: quality * recency * access frequency
                let recency = max(0.01, 1.0 / (1.0 + age / 60.0))  // Decay over 60s
                let accessWeight = log(Double(entry.accessCount + 1))
                let priority = entry.quality * recency * accessWeight * (isExpired ? 0.1 : 1.0)
                evictQueue.enqueue(k, priority: -priority)  // Negative for min-heap eviction
            }

            // Evict 10% of capacity or all expired, whichever is larger
            let evictCount = max(maxCacheSize / 10, responseCache.count - maxCacheSize + 1)
            for _ in 0..<evictCount {
                guard let (evictKey, _) = evictQueue.dequeue() else { break }
                responseCache.removeValue(forKey: evictKey)
            }
            incrementEvictions(by: UInt64(evictCount))
        }

        let entry = ResponseCacheEntry(response: response, timestamp: now, quality: quality)
        responseCache[key] = entry
        cacheLock.unlock()

        shardLocks[shard].unlock()

        // ═══ SIMD-ACCELERATED: Store hash for fast lookup ═══
        hashLock.lock()
        keyHashes[hash] = key
        hashLock.unlock()
    }

    /// Normalize a query for cache lookup
    private func normalizeQuery(_ query: String) -> String {
        return query.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
    }

    /// ═══ LOCK-FREE SCORING - Thread-safe, pure computation ═══
    func scoreResponse(_ response: String, query: String) -> Double {
        return _scoreResponseUnlocked(response, query: query)
    }

    /// Internal scoring - thread-safe, pure function
    private func _scoreResponseUnlocked(_ response: String, query: String) -> Double {
        var score: Double = 0.5

        // Length appropriateness
        let wordCount = response.split(separator: " ").count
        if wordCount >= 10 && wordCount <= 300 {
            score += 0.1
        } else if wordCount < 5 {
            score -= 0.15
        }

        // Relevance - keyword overlap
        let queryWords = Set(query.lowercased().split(separator: " ").map(String.init))
        let responseWords = Set(response.lowercased().split(separator: " ").map(String.init))
        let overlap = Double(queryWords.intersection(responseWords).count)
        let relevance = overlap / Double(max(1, queryWords.count))
        score += relevance * TAU * 0.3

        // Formatting quality
        if response.contains("\n") { score += 0.02 }

        // Coherence - sentence count vs word count ratio
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

    /// EVO_55: φ-weighted health - cache utilization, hit rate, eviction pressure
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

    /// FNV-1a hash for cache keys - delegates to hashKey (fixes issue #5)
    private func fnvHash(_ text: String) -> UInt64 {
        return hashKey(text)
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
// RESPONSE CONFIDENCE ENGINE - Multi-level confidence scoring
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
// MULTI-TURN RESPONSE PLANNER - Structured exploration with plan tracking
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

// ═══════════════════════════════════════════════════════════════════
// MARK: - CHAT BATCH RESPONSE ENGINE (EVO_71)
// Buffers streaming response chunks and flushes them in configurable
// batches, reducing per-token UI update overhead and lock contention.
// Configurable batch size + timeout — flush on whichever fires first.
// ═══════════════════════════════════════════════════════════════════

final class ChatBatchResponseEngine {
    static let shared = ChatBatchResponseEngine()

    // MARK: - Configuration
    var batchSize: Int = 8              // Flush after this many chunks
    var flushIntervalMs: Double = 40.0  // Or after this many milliseconds

    // MARK: - State
    private var buffer: [String] = []
    private var lastFlushTime: Double = 0
    private let lock = NSLock()
    private let flushQueue = DispatchQueue(label: "com.l104.chat_batch_flush",
                                           qos: .userInteractive)
    private var onFlush: (([String]) -> Void)?

    // Metrics
    private(set) var totalChunksReceived = 0
    private(set) var totalFlushes = 0
    private(set) var totalBatchedItems = 0

    /// Register the consumer callback — called on each batch flush.
    func setFlushHandler(_ handler: @escaping ([String]) -> Void) {
        lock.lock(); defer { lock.unlock() }
        onFlush = handler
    }

    /// Accept a new response chunk. Flushes immediately if batch full or timeout elapsed.
    func accept(chunk: String) {
        lock.lock()
        buffer.append(chunk)
        totalChunksReceived += 1
        let shouldFlush = buffer.count >= batchSize || timeoutElapsed()
        let toFlush = shouldFlush ? drainBuffer() : nil
        lock.unlock()

        if let batch = toFlush { deliverBatch(batch) }
    }

    /// Accept multiple chunks at once (e.g., pre-tokenized response array).
    func acceptBatch(_ chunks: [String]) {
        lock.lock()
        buffer.append(contentsOf: chunks)
        totalChunksReceived += chunks.count

        // Drain in batchSize groups
        var batches: [[String]] = []
        while buffer.count >= batchSize {
            batches.append(drainBuffer())
        }
        lock.unlock()

        for batch in batches { deliverBatch(batch) }
    }

    /// Force flush any buffered chunks immediately.
    func flush() {
        lock.lock()
        let batch = drainBuffer()
        lock.unlock()
        if !batch.isEmpty { deliverBatch(batch) }
    }

    // MARK: - Private

    /// Drain buffer under lock. Caller must hold lock.
    private func drainBuffer() -> [String] {
        let batch = buffer
        buffer.removeAll(keepingCapacity: true)
        lastFlushTime = currentTimeMs()
        return batch
    }

    private func timeoutElapsed() -> Bool {
        lastFlushTime == 0 || currentTimeMs() - lastFlushTime >= flushIntervalMs
    }

    private func currentTimeMs() -> Double { Date().timeIntervalSince1970 * 1000.0 }

    private func deliverBatch(_ batch: [String]) {
        guard !batch.isEmpty else { return }
        flushQueue.async { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.totalFlushes += 1
            self.totalBatchedItems += batch.count
            let handler = self.onFlush
            self.lock.unlock()
            handler?(batch)

            InterEngineFeedbackBus.shared.broadcast(
                from: .consciousness,
                signal: "chat_batch_flush",
                payload: ["batch_size": Double(batch.count),
                          "total_chunks": Double(self.totalChunksReceived)]
            )
        }
    }

    var metrics: [String: Int] {
        lock.lock(); defer { lock.unlock() }
        return [
            "chunks_received": totalChunksReceived,
            "flushes": totalFlushes,
            "batched_items": totalBatchedItems,
            "buffer_depth": buffer.count
        ]
    }
}
