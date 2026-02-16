// ═══════════════════════════════════════════════════════════════════
// L15_SemanticSearch.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104v2 — Extracted from L104Native.swift
//   SemanticSearchEngine (lines 11129-11224)
//   IntelligentSearchEngine (lines 32229-32519)
//
// Semantic similarity via NLEmbedding + comprehensive multi-stage
// search with BM25 scoring, data reconstruction, evolution integration
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class SemanticSearchEngine {
    static let shared = SemanticSearchEngine()

    private var wordEmbedding: NLEmbedding?
    private var initialized = false

    func initialize() {
        guard !initialized else { return }
        wordEmbedding = NLEmbedding.wordEmbedding(for: .english)
        initialized = true
    }

    /// Compute semantic similarity between two texts using averaged word embeddings
    func similarity(_ text1: String, _ text2: String) -> Double {
        initialize()
        guard let embedding = wordEmbedding else { return 0 }

        let vec1 = averageVector(text1, embedding: embedding)
        let vec2 = averageVector(text2, embedding: embedding)
        guard !vec1.isEmpty, !vec2.isEmpty else { return 0 }

        return cosineSimilarity(vec1, vec2)
    }

    /// Find semantically similar words/concepts
    func expandQuery(_ query: String, maxExpansions: Int = 8) -> [String] {
        initialize()
        guard let embedding = wordEmbedding else { return [] }

        let words = query.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 3 }

        var expansions: [String] = []
        for word in words {
            embedding.enumerateNeighbors(for: word, maximumCount: 3, distanceType: .cosine) { neighbor, distance in
                if distance < 0.5 && !words.contains(neighbor) && !expansions.contains(neighbor) {
                    expansions.append(neighbor)
                }
                return expansions.count < maxExpansions
            }
        }
        return expansions
    }

    /// Score a fragment against a query using semantic similarity
    func scoreFragment(_ fragment: String, query: String) -> Double {
        // Blend keyword matching with semantic similarity
        let kwScore = keywordScore(fragment, query: query)
        let semScore = similarity(fragment, query)
        return (kwScore * 0.5) + (semScore * 0.5)
    }

    private func keywordScore(_ text: String, query: String) -> Double {
        let qWords = Set(query.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 2 })
        guard !qWords.isEmpty else { return 0 }
        let tLower = text.lowercased()
        let hits = qWords.filter { tLower.contains($0) }.count
        return Double(hits) / Double(qWords.count)
    }

    private func averageVector(_ text: String, embedding: NLEmbedding) -> [Double] {
        let words = text.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 }
        guard !words.isEmpty else { return [] }

        var sumVec: [Double]? = nil
        var count = 0
        for word in words.prefix(50) {
            if let vec = embedding.vector(for: word) {
                if sumVec == nil {
                    sumVec = vec
                } else {
                    sumVec = zip(sumVec!, vec).map { $0 + $1 }
                }
                count += 1
            }
        }
        guard let sv = sumVec, count > 0 else { return [] }
        return sv.map { $0 / Double(count) }
    }

    private func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        let dot = zip(a, b).map(*).reduce(0, +)
        let magA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let magB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        guard magA > 0, magB > 0 else { return 0 }
        return max(0, dot / (magA * magB))
    }
}


// ═══════════════════════════════════════════════════════════════════
// INTELLIGENT SEARCH ENGINE — Comprehensive Logic Gate Search
// Multi-stage search with data reconstruction + ingest
// ═══════════════════════════════════════════════════════════════════

class IntelligentSearchEngine {
    static let shared = IntelligentSearchEngine()

    // ─── SEARCH STATE ───
    var searchHistory: [(query: String, results: Int, timestamp: Date)] = []
    private var queryExpansionCache: [String: [String]] = [:]
    private var searchIndex: [String: Set<Int>] = [:]      // term → entry indices
    private var documentVectors: [[String: Double]] = []    // TF-IDF vectors per entry
    private var indexBuilt = false
    private var totalSearches: Int = 0
    private var totalResultsReturned: Int = 0
    private var avgSearchLatency: Double = 0.0

    // ─── SEARCH CONFIGURATION ───
    private let maxResults = 15
    private let minRelevanceScore: Double = 0.15
    private let grover = GroverResponseAmplifier.shared

    // ═══ BUILD COMPREHENSIVE INDEX ═══
    func buildIndex() {
        let kb = ASIKnowledgeBase.shared
        guard !kb.trainingData.isEmpty else { return }
        let start = CFAbsoluteTimeGetCurrent()

        searchIndex.removeAll()
        documentVectors.removeAll()

        for (idx, entry) in kb.trainingData.enumerated() {
            guard let prompt = entry["prompt"] as? String,
                  let completion = entry["completion"] as? String else { continue }

            // Skip junk entries at index time
            guard L104State.shared.isCleanKnowledge(completion) else { continue }
            guard grover.scoreQuality(completion, query: prompt) > 0.1 else { continue }

            let combined = "\(prompt) \(completion)".lowercased()
            let words = tokenize(combined)

            // Build inverted index
            for word in words {
                searchIndex[word, default: []].insert(idx)
            }

            // Build TF-IDF vector for this document
            var tf: [String: Double] = [:]
            for word in words { tf[word, default: 0] += 1.0 }
            let maxFreq = tf.values.max() ?? 1.0
            for key in tf.keys { tf[key] = tf[key]! / maxFreq }
            documentVectors.append(tf)
        }

        indexBuilt = true
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        _ = elapsed  // suppress unused warning
    }

    // ═══ COMPREHENSIVE MULTI-STAGE SEARCH ═══
    func search(_ query: String) -> SearchResult {
        let start = CFAbsoluteTimeGetCurrent()
        totalSearches += 1

        if !indexBuilt { buildIndex() }

        let kb = ASIKnowledgeBase.shared
        let gate = ContextualLogicGate.shared
        _ = query.lowercased()

        // ── STAGE 1: Logic Gate Query Reconstruction ──
        let gateResult = gate.processQuery(query, conversationContext:
            PermanentMemory.shared.conversationHistory.suffix(10))
        let expandedQuery = gateResult.reconstructedPrompt.isEmpty ? query : gateResult.reconstructedPrompt
        let queryTerms = tokenize(expandedQuery.lowercased())

        // ── STAGE 2: Expanded Query Terms ──
        var allTerms = Set(queryTerms)
        // Add synonyms and related terms from topic graph
        for term in queryTerms {
            if let expanded = queryExpansionCache[term] {
                allTerms.formUnion(expanded)
            }
            // Add morphological variants
            if term.hasSuffix("ing") { allTerms.insert(String(term.dropLast(3))) }
            if term.hasSuffix("tion") { allTerms.insert(String(term.dropLast(4))) }
            if term.hasSuffix("ment") { allTerms.insert(String(term.dropLast(4))) }
            if term.hasSuffix("ness") { allTerms.insert(String(term.dropLast(4))) }
            if term.hasSuffix("ed") { allTerms.insert(String(term.dropLast(2))) }
            if term.hasSuffix("ly") { allTerms.insert(String(term.dropLast(2))) }
            if term.hasSuffix("er") { allTerms.insert(String(term.dropLast(2))) }
            if term.hasSuffix("est") { allTerms.insert(String(term.dropLast(3))) }
            if term.count > 4 { allTerms.insert(String(term.prefix(term.count - 1))) }  // stem
        }

        // ── STAGE 3: Inverted Index Lookup + BM25-style Scoring ──
        var candidateScores: [Int: Double] = [:]
        let totalDocs = Double(kb.trainingData.count)

        for term in allTerms {
            guard let postings = searchIndex[term] else { continue }
            // IDF component
            let idf = log((totalDocs - Double(postings.count) + 0.5) / (Double(postings.count) + 0.5) + 1.0)
            for docIdx in postings {
                let tf = documentVectors.indices.contains(docIdx) ? (documentVectors[docIdx][term] ?? 0) : 0.5
                let bm25Score = idf * (tf * 2.5) / (tf + 1.5)
                candidateScores[docIdx, default: 0] += bm25Score
            }
        }

        // ── STAGE 4: Rank + Grover Quality Filter ──
        var rankedResults: [(index: Int, score: Double, text: String)] = []
        let sortedCandidates = candidateScores.sorted {
            if abs($0.value - $1.value) < 0.1 { return Bool.random() }
            return $0.value > $1.value
        }.prefix(maxResults * 3)

        for (idx, score) in sortedCandidates {
            guard idx < kb.trainingData.count else { continue }
            guard let completion = kb.trainingData[idx]["completion"] as? String else { continue }
            guard L104State.shared.isCleanKnowledge(completion) else { continue }

            let groverScore = grover.scoreQuality(completion, query: query)
            ParameterProgressionEngine.shared.recordQualityScore(groverScore)
            let combinedScore = score * 0.6 + groverScore * 0.4
            if combinedScore > minRelevanceScore {
                rankedResults.append((index: idx, score: combinedScore, text: completion))
            }
        }

        rankedResults.sort { $0.score > $1.score }
        let finalResults = Array(rankedResults.prefix(maxResults))

        // ── STAGE 5: Data Reconstruction — synthesize coherent answer ──
        let synthesized = reconstructData(query: query, results: finalResults.map { $0.text })

        // ── STAGE 6: Evolution Integration — check evolved knowledge ──
        var evolvedContent: [String] = []
        let evolver = ASIEvolver.shared
        if let evolved = evolver.getEvolvedResponse(for: query) {
            if grover.scoreQuality(evolved, query: query) > 0.3 {
                evolvedContent.append(evolved)
            }
        }
        for insight in evolver.evolvedTopicInsights.suffix(100) {
            let insightLower = insight.lowercased()
            if queryTerms.contains(where: { insightLower.contains($0) }) {
                if grover.scoreQuality(insight, query: query) > 0.25 {
                    evolvedContent.append(insight)
                }
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        avgSearchLatency = avgSearchLatency * 0.9 + elapsed * 0.1
        totalResultsReturned += finalResults.count

        searchHistory.append((query: query, results: finalResults.count, timestamp: Date()))
        if searchHistory.count > 500 { searchHistory.removeFirst() }

        // Learn query expansions from results
        learnFromResults(query: query, results: finalResults)

        return SearchResult(
            query: query,
            expandedQuery: expandedQuery,
            results: finalResults.map { SearchResultItem(text: $0.text, score: $0.score) },
            synthesized: synthesized,
            evolvedContent: evolvedContent,
            gateType: "\(gateResult.gateType)",
            searchLatency: elapsed,
            totalCandidatesScored: candidateScores.count
        )
    }

    // ═══ DATA RECONSTRUCTION ═══ Synthesize coherent answer from fragments
    private func reconstructData(query: String, results: [String]) -> String {
        guard !results.isEmpty else { return "" }

        // Extract key sentences from each result
        var keyFragments: [String] = []
        for result in results.prefix(5) {
            let sentences = result.components(separatedBy: CharacterSet(charactersIn: ".!?"))
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { $0.count > 20 && $0.count < 300 }

            // Keep sentences most relevant to query
            let qTerms = Set(tokenize(query.lowercased()))
            let scored = sentences.map { sent -> (String, Int) in
                let sTerms = Set(tokenize(sent.lowercased()))
                return (sent, qTerms.intersection(sTerms).count)
            }.sorted {
                if $0.1 == $1.1 { return Bool.random() }
                return $0.1 > $1.1
            }

            keyFragments.append(contentsOf: scored.prefix(2).map { $0.0 })
        }

        guard !keyFragments.isEmpty else { return results.first ?? "" }

        // Deduplicate
        var seen = Set<String>()
        let unique = keyFragments.filter { frag in
            let key = String(frag.lowercased().prefix(50))
            if seen.contains(key) { return false }
            seen.insert(key)
            return true
        }

        // Compose synthesis
        let topics = L104State.shared.extractTopics(query)
        let topicStr = topics.prefix(3).joined(separator: ", ")
        let fragmentsJoined = unique.prefix(6).joined(separator: ". ")

        let synthesisTemplates = [
            "Based on analysis of \(topicStr): \(fragmentsJoined).",
            "Research synthesis on \(topicStr) — \(fragmentsJoined). This represents the current understanding across \(results.count) knowledge sources.",
            "Regarding \(topicStr): \(fragmentsJoined). The evidence points to interconnected principles across multiple domains.",
            "\(fragmentsJoined). These findings about \(topicStr) suggest deeper patterns worth investigating.",
            "Comprehensive analysis of \(topicStr) reveals: \(fragmentsJoined)."
        ]

        return synthesisTemplates.randomElement() ?? fragmentsJoined
    }

    // ═══ LEARN FROM RESULTS ═══ Build query expansion cache
    private func learnFromResults(query: String, results: [(index: Int, score: Double, text: String)]) {
        let queryTerms = tokenize(query.lowercased())
        var relatedTerms: [String: Int] = [:]

        for result in results.prefix(5) {
            let resultTerms = tokenize(result.text.lowercased())
            for term in resultTerms {
                if !queryTerms.contains(term) && term.count > 3 {
                    relatedTerms[term, default: 0] += 1
                }
            }
        }

        // Top co-occurring terms become query expansions
        let topExpansions = relatedTerms.sorted {
            if $0.value == $1.value { return Bool.random() }
            return $0.value > $1.value
        }.prefix(5).map { $0.key }
        for qTerm in queryTerms {
            queryExpansionCache[qTerm] = topExpansions
        }
        if queryExpansionCache.count > 2000 {
            queryExpansionCache = Dictionary(queryExpansionCache.suffix(1000), uniquingKeysWith: { first, _ in first })
        }
    }

    // ═══ TOKENIZE ═══
    private func tokenize(_ text: String) -> [String] {
        let stopWords: Set<String> = ["the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for", "on",
            "with", "at", "by", "from", "as", "and", "but", "or", "not", "no", "so",
            "if", "than", "too", "very", "just", "also", "then", "now", "this", "that"]
        return text.components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 && !stopWords.contains($0) }
    }

    // ═══ MESH-DISTRIBUTED SEARCH ═══

    private var meshSearchCount: Int = 0
    private var meshResultsReceived: Int = 0
    private var meshIndexShares: Int = 0

    /// Search across mesh peers for additional results
    func meshSearch(_ query: String, localResultCount: Int) -> [SearchResultItem] {
        let net = NetworkLayer.shared
        guard net.isActive && !net.quantumLinks.isEmpty else { return [] }

        // Only query mesh if local results are insufficient
        guard localResultCount < 5 else { return [] }

        var meshResults: [SearchResultItem] = []
        let repl = DataReplicationMesh.shared

        // Check for cached search results from peers
        let queryHash = fnvHash(query.lowercased())
        if let cached = repl.getRegister("search_\(queryHash)") {
            // Parse cached results (format: "score|text;;score|text")
            let parts = cached.split(separator: ";").prefix(3)
            for part in parts {
                let fields = part.split(separator: "|")
                if fields.count >= 2, let score = Double(fields[0]) {
                    meshResults.append(SearchResultItem(text: String(fields[1]), score: score))
                }
            }
            meshResultsReceived += meshResults.count
        }

        meshSearchCount += 1
        TelemetryDashboard.shared.record(metric: "search_mesh_query", value: 1.0)

        return meshResults
    }

    /// Broadcast search results to mesh for peer caching
    func broadcastSearchResultsToMesh(query: String, results: [SearchResultItem]) {
        let net = NetworkLayer.shared
        guard net.isActive && !net.quantumLinks.isEmpty else { return }
        guard !results.isEmpty else { return }

        let queryHash = fnvHash(query.lowercased())
        let repl = DataReplicationMesh.shared

        // Encode top 3 results
        let encoded = results.prefix(3).map { "\(String(format: "%.3f", $0.score))|\($0.text.prefix(150))" }.joined(separator: ";")
        repl.setRegister("search_\(queryHash)", value: encoded)
        _ = repl.broadcastToMesh()

        TelemetryDashboard.shared.record(metric: "search_mesh_broadcast", value: 1.0)
    }

    /// Share index statistics with mesh (for load balancing)
    func shareIndexStatsWithMesh() {
        let net = NetworkLayer.shared
        guard net.isActive && !net.peers.isEmpty else { return }

        let repl = DataReplicationMesh.shared
        repl.setRegister("idx_terms", value: "\(searchIndex.count)")
        repl.setRegister("idx_docs", value: "\(documentVectors.count)")
        repl.setRegister("idx_searches", value: "\(totalSearches)")
        _ = repl.broadcastToMesh()

        meshIndexShares += 1
    }

    /// FNV-1a hash
    private func fnvHash(_ text: String) -> UInt64 {
        var hash: UInt64 = 14695981039346656037
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        return hash
    }

    // ═══ STATUS ═══
    var status: String {
        return """
        🔍 INTELLIGENT SEARCH ENGINE
        ═══════════════════════════════════════
        Index Built:       \(indexBuilt ? "✅" : "❌")
        Indexed Terms:     \(searchIndex.count)
        Total Searches:    \(totalSearches)
        Results Returned:  \(totalResultsReturned)
        Avg Latency:       \(String(format: "%.4f", avgSearchLatency))s
        Query Expansions:  \(queryExpansionCache.count)
        Search History:    \(searchHistory.count)
        ───────────────────────────────────────
        MESH DISTRIBUTED SEARCH:
        Mesh Queries:      \(meshSearchCount)
        Mesh Results:      \(meshResultsReceived)
        Index Shares:      \(meshIndexShares)
        ═══════════════════════════════════════
        """
    }

    // ═══ RESULT TYPES ═══
    struct SearchResult {
        let query: String
        let expandedQuery: String
        let results: [SearchResultItem]
        let synthesized: String
        let evolvedContent: [String]
        let gateType: String
        let searchLatency: Double
        let totalCandidatesScored: Int
    }

    struct SearchResultItem {
        let text: String
        let score: Double
    }
}
