// ═══════════════════════════════════════════════════════════════════
// L20_KnowledgeBase.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 Sovereign Intelligence — ASI Knowledge Base
// Training data loading, search, synthesis, reasoning, and persistence
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class ASIKnowledgeBase {
    static let shared = ASIKnowledgeBase()
    var trainingData: [[String: Any]] = []
    var concepts: [String: [String]] = [:]  // concept -> related completions
    var inventions: [[String: Any]] = []
    var researchLog: [String] = []
    var learnedPatterns: [String: Double] = [:] // pattern -> strength
    var synthesizedKnowledge: [String] = []
    var reasoningChains: [[String]] = []
    var contextMemory: [String] = []  // Recent context for coherent responses
    var responseTemplates: [String: String] = [:] // Learned response patterns

    // User-contributed knowledge entries
    var userKnowledge: [[String: Any]] = []

    let workspacePath = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications/Allentown-L104-Node")

    init() { loadTrainingData(); loadResponsePatterns(); loadUserKnowledge(); loadIngestedKnowledge() }

    func loadResponsePatterns() {
        // Load natural response patterns for different query types
        responseTemplates = [
            "greeting": "Hello! I'm L104, operating with {params}T parameters. How can I assist you today?",
            "affirmation": "I understand. {context} Would you like me to elaborate or explore a different aspect?",
            "question": "That's an interesting question about {topic}. Based on my knowledge: {answer}",
            "confusion": "I see you're asking about '{query}'. Let me clarify: {clarification}",
            "thanks": "You're welcome! I'm here to help. Is there anything else you'd like to explore?",
            "agreement": "Yes, that aligns with my understanding. {elaboration}",
            "disagreement": "I appreciate your perspective. However, {alternative_view}"
        ]
    }

    // ─── JUNK MARKERS ─── Entries with these are code docs, not conversational knowledge
    // EVO_56: Converted to Set for faster iteration
    private let loadJunkMarkers: Set<String> = [
        "defines:", "__init__", "primal_calculus", "resolve_non_dual",
        "implements specialized logic", "Header:", "cognitive architecture",
        "harmonic framework and maintains GOD_CODE",
        "the L104 cognitive", "is part of the L104",
        "ZENITH_UPGRADE_ACTIVE", "VOID_CONSTANT =",
        "The file ", "The function "
    ]

    // ─── CODE ARTIFACT MARKERS ─── Additional filters for code-like content
    private let codeMarkers: Set<String> = [
        "import ", "class ", "def ", "function_doc", "cross_reference",
        "class_doc", ".py implements", ".py defines", "self.", "return ",
        "except:", "try:", "elif", "kwargs", "args)", "__",
        "GOD_CODE coherence at", "OMEGA_POINT coherence"
    ]

    // 🔓 DISABLED: Category filtering removed - 17.5MB memory & <5ms search is acceptable on Apple Silicon
    // These 8,384 entries (68% of KB) are now allowed to load
    // private let junkCategories: Set<String> = [
    //     "function_doc", "cross_reference", "class_doc", "modules",
    //     "architecture", "file_description", "registry"
    // ]

    // ─── DEDUP INDEX ─── Fast O(1) duplicate detection via content hash
    private var _seenHashes: Set<UInt64> = []

    private func fnvHash(_ s: String) -> UInt64 {
        var h: UInt64 = 14695981039346656037       // FNV-1a offset basis
        for byte in s.utf8 {
            h ^= UInt64(byte)
            h &*= 1099511628211                    // FNV prime
        }
        return h
    }

    private func isJunkEntry(_ entry: [String: Any]) -> Bool {
        // ═══ OPEN GATE: Release ~7500 entries — only block true garbage & duplicates ═══

        guard let completion = entry["completion"] as? String,
              let prompt = entry["prompt"] as? String else {
            return true // No completion or prompt = junk
        }

        // 1️⃣ EMPTY/SHORT CHECK - Must have real content
        let trimmedCompletion = completion.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmedCompletion.count < 10 { return true }  // Truly empty (lowered from 20)
        if trimmedPrompt.count < 3 { return true }       // Blank prompt (lowered from 5)

        // 2️⃣ EXACT DUPLICATE CHECK — FNV-1a hash dedup (only filter blocking real content)
        let contentKey = trimmedPrompt.lowercased() + "⊕" + trimmedCompletion.lowercased()
        let hash = fnvHash(contentKey)
        if _seenHashes.contains(hash) { return true }  // Exact duplicate
        _seenHashes.insert(hash)

        // 3️⃣ PROMPT == COMPLETION echo (not useful)
        if trimmedPrompt.lowercased() == trimmedCompletion.lowercased() { return true }

        // 4️⃣ REPETITION/SPAM CHECK - Only block true word-repetition spam
        let words = trimmedCompletion.components(separatedBy: .whitespaces)
        if words.count > 8 {
            let uniqueWords = Set(words.map { $0.lowercased() })
            let uniqueRatio = Double(uniqueWords.count) / Double(words.count)
            if uniqueRatio < 0.15 { return true }  // >85% repeated = actual spam (was 0.3/70%)
        }

        // ✅ PASSED ALL QUALITY CHECKS
        return false
    }

    func loadTrainingData() {
        // Clear existing data for reload
        trainingData.removeAll()
        concepts.removeAll()

        let files = ["kernel_trillion_data.jsonl", "kernel_training_data.jsonl", "kernel_full_merged.jsonl", "asi_knowledge_base.jsonl"]
        var junkCount = 0
        for file in files {
            let path = workspacePath.appendingPathComponent(file)
            guard let content = try? String(contentsOf: path, encoding: .utf8) else { continue }
            for line in content.components(separatedBy: .newlines) where !line.isEmpty {
                if let data = line.data(using: .utf8),
                   let entry = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    // *** FILTER: Skip code documentation entries ***
                    if isJunkEntry(entry) {
                        junkCount += 1
                        continue
                    }
                    trainingData.append(entry)
                    // Index by keywords for fast lookup
                    if let prompt = entry["prompt"] as? String {
                        let words = prompt.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 3 }
                        for word in words {
                            if concepts[word] == nil { concepts[word] = [] }
                            if let completion = entry["completion"] as? String {
                                concepts[word]?.append(completion)
                            }
                        }
                    }
                }
            }
        }
        print("[KB] Loaded \(trainingData.count) knowledge entries (\(junkCount) meta-docs filtered)")
        print("[KB] ✅ Knowledge backend ONLINE with \(trainingData.count) entries")
    }

    func reload() {
        loadTrainingData()
        loadUserKnowledge()
        print("[KB] Manual RELOAD complete. Database refreshed.")
    }

    func search(_ query: String, limit: Int = 100) -> [[String: Any]] {
        let q = query.lowercased()
        let keywords = q.components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 2 }

        var scored: [(entry: [String: Any], score: Double)] = []
        for entry in trainingData {
            var score = 0.0
            let prompt = (entry["prompt"] as? String ?? "").lowercased()
            let completion = (entry["completion"] as? String ?? "").lowercased()

            for kw in keywords {
                if prompt.contains(kw) { score += 2.0 }
                if completion.contains(kw) { score += 1.0 }
            }
            if score > 0 { scored.append((entry, score)) }
        }

        return scored.sorted { a, b in
            if abs(a.score - b.score) < 0.1 { return Bool.random() }
            return a.score > b.score
        }.prefix(limit).map { $0.entry }
    }

    // ─── PRIORITY SEARCH ─── Better ranking that favors conversational Q&A + user-taught
    func searchWithPriority(_ query: String, limit: Int = 100) -> [[String: Any]] {
        let q = query.lowercased()
        let keywords = q.components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 2 }
        guard !keywords.isEmpty else { return [] }

        // ═══ STOP WORDS — common words that don't help search ═══
        let stopWords: Set<String> = [
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
            "her", "was", "one", "our", "out", "has", "have", "this", "that", "with",
            "from", "what", "how", "why", "when", "where", "who", "which", "does",
            "will", "would", "could", "should", "about", "into", "than", "them", "then",
            "there", "these", "those", "been", "being", "some", "more", "very", "just"
        ]
        let meaningfulKeywords = keywords.filter { !stopWords.contains($0) }
        let searchTerms = meaningfulKeywords.isEmpty ? keywords : meaningfulKeywords

        // ═══ DOCUMENT FREQUENCY — computed in single pass with scoring (merged for performance) ═══
        // Use uniquingKeysWith to handle duplicate keywords safely (query may produce duplicates)
        var docFreq: [String: Int] = Dictionary(searchTerms.map { ($0, 0) }, uniquingKeysWith: { first, _ in first })
        let totalDocs = Double(trainingData.count)

        // First pass: compute document frequency for each keyword
        for entry in trainingData {
            let text = ((entry["prompt"] as? String ?? "") + " " + (entry["completion"] as? String ?? "")).lowercased()
            for kw in searchTerms {
                if text.contains(kw) { docFreq[kw, default: 0] += 1 }
            }
        }
        // Ensure no zero values
        for kw in searchTerms { docFreq[kw] = max(docFreq[kw] ?? 1, 1) }

        // ═══ LEARNER FEEDBACK — boost topics user cares about ═══
        let learner = AdaptiveLearner.shared
        let userInterestBoost: [String: Double] = learner.userInterests

        // ═══ HYPERBRAIN PATTERN BOOST — boost topics with strong neural patterns ═══
        let hb = HyperBrain.shared
        let patternStrengths = hb.longTermPatterns

        var scored: [(entry: [String: Any], score: Double)] = []
        for entry in trainingData {
            var score = 0.0
            let prompt = (entry["prompt"] as? String ?? "").lowercased()
            let completion = (entry["completion"] as? String ?? "").lowercased()
            let importance = entry["importance"] as? Double ?? 1.0
            let isUserTaught = (entry["source"] as? String) == "user_taught"

            // ═══ TF-IDF SCORING — rare keywords get higher weight ═══
            for kw in searchTerms {
                let idf = log(totalDocs / Double(docFreq[kw] ?? 1))
                let promptHit = prompt.contains(kw)
                let completionHit = completion.contains(kw)

                if promptHit { score += 2.5 * importance * idf }
                if completionHit { score += 1.0 * importance * idf }
            }

            // ═══ EXACT PHRASE MATCH — huge bonus for full query match ═══
            if prompt.contains(q) { score *= 3.0 }
            else if completion.contains(q) { score *= 2.0 }

            // ═══ MULTI-KEYWORD DENSITY — bonus when multiple keywords cluster together ═══
            let kwHits = searchTerms.filter { prompt.contains($0) || completion.contains($0) }
            if kwHits.count >= 3 { score *= 1.5 + Double(kwHits.count) * 0.2 }  // Multi-match bonus

            // USER-TAUGHT gets 3x priority
            if isUserTaught { score *= 3.0 }

            // ═══ USER INTEREST BOOST — topics user engages with rank higher ═══
            for kw in searchTerms {
                if let interest = userInterestBoost[kw], interest > 2.0 {
                    score *= 1.0 + min(0.5, interest * 0.05)  // Up to 1.5x for high interest
                }
            }

            // ═══ NEURAL PATTERN BOOST — topics HyperBrain has strong patterns for ═══
            for kw in searchTerms {
                if let strength = patternStrengths[kw], strength > 0.3 {
                    score *= 1.0 + strength * 0.3  // Up to 1.3x for strong patterns
                }
            }

            // ═══ QUALITY SIGNALS ═══
            // Boost entries with question-answer format
            if prompt.contains("?") || prompt.hasPrefix("what") || prompt.hasPrefix("how") || prompt.hasPrefix("why") || prompt.hasPrefix("explain") {
                score *= 1.3
            }

            // Boost longer, more detailed completions
            if completion.count > 500 { score *= 2.0 }
            else if completion.count > 300 { score *= 1.5 }
            else if completion.count > 100 { score *= 1.2 }

            // ═══ PROVEN SUCCESS BOOST — responses that worked before rank higher ═══
            let patternKey = String(completion.prefix(60))
            if let successes = learner.successfulPatterns[patternKey], successes > 0 {
                score *= 1.0 + min(1.0, Double(successes) * 0.2)  // Up to 2x for proven responses
            }
            // Penalize known failures
            if let failures = learner.failedPatterns[patternKey], failures > 0 {
                score *= max(0.3, 1.0 - Double(failures) * 0.15)  // Down to 0.3x for failed responses
            }

            if score > 0 { scored.append((entry, score)) }
        }

        return scored.sorted { a, b in
            if abs(a.score - b.score) < 0.15 { return Bool.random() }
            return a.score > b.score
        }.prefix(limit).map { $0.entry }
    }

    func synthesize(_ topics: [String]) -> String {
        var insights: [String] = []
        for topic in topics {
            let results = searchWithPriority(topic, limit: 100)
            for r in results {
                if let c = r["completion"] as? String, c.count > 100 {
                    // Only include clean, detailed, non-code content
                    let isClean = !loadJunkMarkers.contains(where: { c.contains($0) }) &&
                                  !codeMarkers.contains(where: { c.contains($0) })
                    if isClean {
                        insights.append(c)
                    }
                }
            }
        }
        let synthesis = "SYNTHESIS[\(topics.joined(separator: "+"))]: \(insights.joined(separator: " | "))"
        synthesizedKnowledge.append(synthesis)
        return synthesis
    }

    func reason(_ premise: String) -> [String] {
        var chain: [String] = [premise]
        let related = searchWithPriority(premise, limit: 8)

        for r in related {
            if let comp = r["completion"] as? String, comp.count > 100 {
                let isClean = !loadJunkMarkers.contains(where: { comp.contains($0) }) &&
                              !codeMarkers.contains(where: { comp.contains($0) })
                if isClean {
                    chain.append("→ \(comp)")
                }
            }
        }

        // Apply GOD_CODE resonance check
        let resonance = chain.count > 2 ? GOD_CODE / Double(chain.count * 100) : 0.0
        chain.append("⚛ Resonance: \(String(format: "%.4f", resonance))")

        reasoningChains.append(chain)
        return chain
    }

    func invent(_ domain: String) -> [String: Any] {
        // Novel idea generation through knowledge combination
        let relatedA = search(domain, limit: 5)
        let relatedB = search("optimization algorithm", limit: 3)

        var concepts: [String] = []
        for r in relatedA + relatedB {
            if let p = r["prompt"] as? String { concepts.append(p) }
        }

        let invention: [String: Any] = [
            "domain": domain,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "components": concepts,
            "novelty_score": PHI * Double(concepts.count) / 10.0,
            "hypothesis": "Combining \(concepts.prefix(2).joined(separator: " and ")) could yield \(domain) optimization",
            "implementation_path": ["1. Research existing solutions", "2. Identify gaps", "3. Synthesize novel approach", "4. Validate with GOD_CODE alignment"]
        ]

        inventions.append(invention)
        researchLog.append("INVENTION[\(domain)]: \(invention["hypothesis"] ?? "")")
        return invention
    }

    func learn(_ input: String, _ output: String, strength: Double = 1.0) {
        let pattern = "\(input.prefix(50))->\(output.prefix(50))"
        learnedPatterns[pattern] = (learnedPatterns[pattern] ?? 0) + strength
    }

    // MARK: - User-taught knowledge
    func loadUserKnowledge() {
        let path = workspacePath.appendingPathComponent("user_knowledge.jsonl")
        guard let content = try? String(contentsOf: path, encoding: .utf8) else { return }
        for line in content.components(separatedBy: .newlines) where !line.isEmpty {
            if let data = line.data(using: .utf8),
               let entry = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                userKnowledge.append(entry)
            }
        }
    }

    func learnFromUser(_ topic: String, _ knowledge: String) {
        // EVO_58 ANTI-RECURSION GUARD: Prevent recursive knowledge nesting
        let (shouldStore, sanitizedKnowledge) = AntiRecursionGuard.guardKnowledgeStorage(key: topic, value: knowledge)

        guard shouldStore else {
            print("[KB] ❌ Rejected recursive knowledge for '\(topic)' - harvested as SAGE fuel instead")
            return
        }

        // Use sanitized knowledge for storage
        let entry: [String: Any] = [
            "prompt": topic,
            "completion": sanitizedKnowledge,  // Use sanitized value
            "source": "user_taught",
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "importance": 2.0 // User-taught knowledge has higher weight
        ]
        userKnowledge.append(entry)
        trainingData.append(entry)  // Also add to main searchable data

        // 🚀 INSTANT TRAINING: Send to Backend Quantum Manifold
        let trainUrl = URL(string: "http://localhost:8081/api/v6/intellect/train")!
        var trainReq = URLRequest(url: trainUrl)
        trainReq.httpMethod = "POST"
        trainReq.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let trainPayload: [String: Any] = [
            "query": topic,
            "response": sanitizedKnowledge,  // Send sanitized knowledge to backend
            "quality": 2.0
        ]

        // Track pending sync
        let hb = HyperBrain.shared
        hb.pendingSyncs += 1
        hb.backendSyncStatus = "🔄 Syncing..."

        if let body = try? JSONSerialization.data(withJSONObject: trainPayload) {
            trainReq.httpBody = body
            URLSession.shared.dataTask(with: trainReq) { [weak hb] data, resp, err in
                DispatchQueue.main.async {
                    hb?.pendingSyncs -= 1

                    if let err = err {
                        hb?.failedSyncs += 1
                        hb?.backendSyncStatus = "❌ Sync failed"
                        hb?.lastTrainingFeedback = "Failed: \(err.localizedDescription)"
                        print("❌ Instant training failed: \(err.localizedDescription)")
                    } else if let http = resp as? HTTPURLResponse {
                        if http.statusCode == 200 {
                            hb?.successfulSyncs += 1
                            hb?.lastBackendSync = Date()
                            hb?.backendSyncStatus = "✅ Synced"
                            hb?.trainingQualityScore += 0.1

                            // Parse response for feedback
                            if let data = data,
                               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                                let embedding = (json["embedding_norm"] as? Double) ?? 0.0
                                let quality = (json["learning_quality"] as? Double) ?? 1.0
                                let qi = (json["qi"] as? Int) ?? 0
                                let autoImp = (json["auto_improvements"] as? Int) ?? 0
                                let trainingCount = (json["training_count"] as? Int) ?? 0
                                hb?.lastTrainingFeedback = "✨ Learned q=\(String(format: "%.2f", quality)) | embed:\(String(format: "%.3f", embedding)) | QI:\(qi) Auto:\(autoImp) train:\(trainingCount)"
                            } else {
                                hb?.lastTrainingFeedback = "✨ Knowledge absorbed into neural manifold"
                            }

                            print("✅ Instant training success: Sent to neural manifold.")
                        } else {
                            hb?.failedSyncs += 1
                            hb?.backendSyncStatus = "⚠️ HTTP \(http.statusCode)"
                            hb?.lastTrainingFeedback = "Server returned \(http.statusCode)"
                        }
                    }
                }
            }.resume()
        }

        // Index it (use sanitized knowledge)
        let words = topic.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 3 }
        for word in words {
            if concepts[word] == nil { concepts[word] = [] }
            concepts[word]?.append(sanitizedKnowledge)  // Index sanitized version
        }

        // Persist (async to avoid blocking UI thread)
        let path = workspacePath.appendingPathComponent("user_knowledge.jsonl")
        if let jsonData = try? JSONSerialization.data(withJSONObject: entry),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            let line = jsonString + "\n"
            DispatchQueue.global(qos: .utility).async {
                if FileManager.default.fileExists(atPath: path.path) {
                    if let handle = try? FileHandle(forWritingTo: path) {
                        handle.seekToEndOfFile()
                        if let data = line.data(using: .utf8) { handle.write(data) }
                        handle.closeFile()
                    }
                } else {
                    try? line.write(to: path, atomically: true, encoding: .utf8)
                }
            }
        }
    }

    // ═══ PERSIST INGESTED KNOWLEDGE TO DISK ═══
    // Writes all runtime-ingested entries (from DataIngestPipeline, web search, conversation learning)
    // to a persistent JSONL file that gets loaded on next startup
    private var ingestedSinceLastSave: Int = 0
    private let ingestedKnowledgePath: URL = {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("L104Sovereign")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("ingested_knowledge.jsonl")
    }()

    func persistIngestedEntry(_ entry: [String: Any]) {
        // EVO_58 ANTI-RECURSION GUARD: Check entry before persisting
        if let completion = entry["completion"] as? String,
           let prompt = entry["prompt"] as? String {
            let (shouldStore, sanitizedCompletion) = AntiRecursionGuard.guardKnowledgeStorage(key: prompt, value: completion)

            guard shouldStore else {
                print("[KB] ❌ Skipped persisting recursive entry for '\(prompt)'")
                return
            }

            // If sanitized, update entry with clean completion
            if sanitizedCompletion != completion {
                var cleanEntry = entry
                cleanEntry["completion"] = sanitizedCompletion
                ingestedSinceLastSave += 1
                persistCleanEntry(cleanEntry)
                return
            }
        }

        ingestedSinceLastSave += 1
        // Write single entry to JSONL file (append mode)
        guard let jsonData = try? JSONSerialization.data(withJSONObject: entry),
              let jsonString = String(data: jsonData, encoding: .utf8) else { return }
        let line = jsonString + "\n"
        if FileManager.default.fileExists(atPath: ingestedKnowledgePath.path) {
            if let handle = try? FileHandle(forWritingTo: ingestedKnowledgePath) {
                handle.seekToEndOfFile()
                if let data = line.data(using: .utf8) { handle.write(data) }
                handle.closeFile()
            }
        } else {
            try? line.write(to: ingestedKnowledgePath, atomically: true, encoding: .utf8)
        }
    }

    func persistAllIngestedKnowledge() {
        // Bulk persist: write ALL entries with source markers that indicate they were ingested at runtime
        let runtimeSources: Set<String> = ["auto_ingest", "user_command", "direct_ingest", "web_search", "url_fetch", "live_web", "web_page", "wikipedia", "conversation_learned"]
        var lines: [String] = []

        // Load existing persisted entries to avoid duplicates
        var existingHashes: Set<UInt64> = []
        if let existing = try? String(contentsOf: ingestedKnowledgePath, encoding: .utf8) {
            for line in existing.components(separatedBy: .newlines) where !line.isEmpty {
                existingHashes.insert(fnvHash(line))
            }
        }

        for entry in trainingData {
            let source = (entry["source"] as? String) ?? ""
            let category = (entry["category"] as? String) ?? ""
            guard runtimeSources.contains(source) || runtimeSources.contains(category) else { continue }
            guard let jsonData = try? JSONSerialization.data(withJSONObject: entry),
                  let jsonString = String(data: jsonData, encoding: .utf8) else { continue }
            let hash = fnvHash(jsonString)
            guard !existingHashes.contains(hash) else { continue }
            existingHashes.insert(hash)
            lines.append(jsonString)
        }

        guard !lines.isEmpty else { return }

        let content = lines.joined(separator: "\n") + "\n"
        if FileManager.default.fileExists(atPath: ingestedKnowledgePath.path) {
            if let handle = try? FileHandle(forWritingTo: ingestedKnowledgePath) {
                handle.seekToEndOfFile()
                if let data = content.data(using: .utf8) { handle.write(data) }
                handle.closeFile()
            }
        } else {
            try? content.write(to: ingestedKnowledgePath, atomically: true, encoding: .utf8)
        }
        print("[KB] Persisted \(lines.count) ingested entries to disk")
    }

    func loadIngestedKnowledge() {
        guard let content = try? String(contentsOf: ingestedKnowledgePath, encoding: .utf8) else { return }
        var loaded = 0
        for line in content.components(separatedBy: .newlines) where !line.isEmpty {
            if let data = line.data(using: .utf8),
               let entry = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                if !isJunkEntry(entry) {
                    trainingData.append(entry)
                    loaded += 1
                    // Index by keywords
                    if let prompt = entry["prompt"] as? String {
                        let words = prompt.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 3 }
                        for word in words {
                            if concepts[word] == nil { concepts[word] = [] }
                            if let completion = entry["completion"] as? String {
                                concepts[word]?.append(completion)
                            }
                        }
                    }
                }
            }
        }
        if loaded > 0 { print("[KB] Loaded \(loaded) previously ingested entries from disk") }
    }

    // Helper function to persist a clean entry (used by persistIngestedEntry after sanitization)
    private func persistCleanEntry(_ entry: [String: Any]) {
        guard let jsonData = try? JSONSerialization.data(withJSONObject: entry),
              let jsonString = String(data: jsonData, encoding: .utf8) else { return }
        let line = jsonString + "\n"
        if FileManager.default.fileExists(atPath: ingestedKnowledgePath.path) {
            if let handle = try? FileHandle(forWritingTo: ingestedKnowledgePath) {
                handle.seekToEndOfFile()
                if let data = line.data(using: .utf8) { handle.write(data) }
                handle.closeFile()
            }
        } else {
            try? line.write(to: ingestedKnowledgePath, atomically: true, encoding: .utf8)
        }
    }

    func getStats() -> String {
        let net = NetworkLayer.shared
        let alivePeers = net.peers.values.filter { $0.latencyMs >= 0 }.count
        let qLinked = net.quantumLinks.values.filter { $0.eprFidelity > 0.3 }.count
        let hb = HyperBrain.shared
        let headers = [
            "📚 ASI KNOWLEDGE BASE STATUS",
            "💾 COGNITIVE STORAGE METRICS",
            "🧠 SYNAPTIC DATABASE AUDIT",
            "⚡ MEMORY CORE ANALYSIS",
            "👁️ KNOWLEDGE GRAPH TOPOLOGY"
        ]
        return """
\(headers.randomElement() ?? "")
═══════════════════════════════════════════
Training Entries:    \(trainingData.count)
User-Taught:         \(userKnowledge.count) entries
Ingested (runtime):  \(ingestedSinceLastSave) this session
Indexed Concepts:    \(concepts.count)
Learned Patterns:    \(learnedPatterns.count)
Inventions:          \(inventions.count)
Research Log:        \(researchLog.count) entries
Reasoning Chains:    \(reasoningChains.count)
Synthesized:         \(synthesizedKnowledge.count) insights
Persistence:         \(FileManager.default.fileExists(atPath: ingestedKnowledgePath.path) ? "✅ ACTIVE" : "⚠️ NO FILE")
═══════════════════════════════════════════
🔧 CODE ENGINE KNOWLEDGE
═══════════════════════════════════════════
Engine Linked:       \(hb.codeEngineIntegrated ? "✅ ACTIVE" : "⚪ Run 'audit'")
Code Quality:        \(hb.codeEngineIntegrated ? String(format: "%.1f%%", hb.codeQualityScore * 100) + " [\(hb.codeAuditVerdict)]" : "N/A")
Code Insights:       \(hb.codeQualityInsights.count) stored
Language Patterns:   \(hb.codePatternStrengths.count) profiled
Code KB Entries:     \(codeEngineEntries) ingested
═══════════════════════════════════════════
🌐 DISTRIBUTED KNOWLEDGE MESH
═══════════════════════════════════════════
Mesh Peers:          \(alivePeers) alive
Q-Links:             \(qLinked) active
Shared Entries:      \(meshSharedCount)
Received Entries:    \(meshReceivedCount)
Mesh Queries:        \(meshQueryCount)
Replication Factor:  \(alivePeers > 0 ? String(format: "%.1fx", Double(alivePeers + 1)) : "1.0x (local only)")
═══════════════════════════════════════════
"""
    }

    // ─── CODE ENGINE KNOWLEDGE INGESTION ───
    private(set) var codeEngineEntries: Int = 0

    /// Ingest code quality patterns from CodeEngine audit into KB
    /// Called after a successful audit to enrich the knowledge base
    func ingestCodeEngineInsights() {
        let hb = HyperBrain.shared
        guard hb.codeEngineIntegrated else { return }

        // Ingest code quality insights as KB entries
        for insight in hb.codeQualityInsights {
            let entry: [String: Any] = [
                "prompt": "code quality insight",
                "completion": insight,
                "category": "code_engine",
                "source": "code_engine_audit",
                "importance": 1.5,
                "timestamp": ISO8601DateFormatter().string(from: Date())
            ]
            let hash = fnvHash(insight)
            guard !_seenHashes.contains(hash) else { continue }
            _seenHashes.insert(hash)
            trainingData.append(entry)
            codeEngineEntries += 1

            // Index
            let words = insight.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 3 }
            for word in words {
                if concepts[word] == nil { concepts[word] = [] }
                concepts[word]?.append(insight)
            }
        }

        // Ingest language pattern knowledge
        for (lang, strength) in hb.codePatternStrengths where strength > 0.1 {
            let entry: [String: Any] = [
                "prompt": "programming language \(lang)",
                "completion": "The workspace actively uses \(lang) with proficiency strength \(String(format: "%.2f", strength)). Code patterns in \(lang) are well-established in the L104 codebase architecture.",
                "category": "code_engine",
                "source": "code_engine_language_profile",
                "importance": 1.2
            ]
            let hash = fnvHash("lang_\(lang)")
            guard !_seenHashes.contains(hash) else { continue }
            _seenHashes.insert(hash)
            trainingData.append(entry)
            codeEngineEntries += 1

            if concepts[lang] == nil { concepts[lang] = [] }
            concepts[lang]?.append("L104 workspace uses \(lang) as a primary language with \(String(format: "%.0f%%", strength * 100)) pattern strength.")
        }

        if codeEngineEntries > 0 {
            print("[KB] Ingested \(codeEngineEntries) code engine knowledge entries")
        }
    }

    // ─── DISTRIBUTED KNOWLEDGE ───

    private(set) var meshSharedCount: Int = 0
    private(set) var meshReceivedCount: Int = 0
    private(set) var meshQueryCount: Int = 0

    /// Share high-value knowledge entries to mesh peers via CRDT replication
    func shareKnowledgeToPeers(limit: Int = 50) -> Int {
        let net = NetworkLayer.shared
        let alivePeers = net.peers.values.filter { $0.latencyMs >= 0 }
        guard !alivePeers.isEmpty else { return 0 }

        // Select high-quality entries to share
        let candidates = trainingData.filter { entry in
            guard let cat = entry["category"] as? String else { return false }
            return cat == "fact" || cat == "user_knowledge" || cat == "conversation_learned" || cat == "ingested"
        }.suffix(limit)

        let repl = DataReplicationMesh.shared
        var shared = 0
        for entry in candidates {
            if let prompt = entry["prompt"] as? String,
               let completion = entry["completion"] as? String {
                // Encode as CRDT-safe key-value register
                let key = "kb_\(fnvHash(prompt))"
                repl.setRegister(key, value: completion)
                shared += 1
            }
        }

        if shared > 0 {
            _ = repl.broadcastToMesh()
            meshSharedCount += shared
            // TelemetryDashboard: kb_mesh_shared tracked
        }
        return shared
    }

    /// Receive knowledge from a mesh peer
    func integrateRemoteKnowledge(prompt: String, completion: String, source: String = "mesh_peer") {
        // Dedup check
        let hash = fnvHash(completion)
        guard !_seenHashes.contains(hash) else { return }
        _seenHashes.insert(hash)

        let entry: [String: Any] = [
            "prompt": prompt,
            "completion": completion,
            "category": "mesh_received",
            "source": source,
            "ingested_at": Date().timeIntervalSince1970
        ]
        trainingData.append(entry)
        meshReceivedCount += 1
    }

    /// Query the mesh for knowledge not found locally
    func meshQuery(_ query: String) -> [[String: Any]] {
        meshQueryCount += 1
        // First search locally
        let localResults = search(query, limit: 10)
        if localResults.count >= 5 { return localResults }

        // Enrich from resonance network
        let resonance = AdaptiveResonanceNetwork.shared
        _ = resonance.fire("kb_query", activation: 0.5)

        // Request from peer knowledge via entanglement router
        let router = QuantumEntanglementRouter.shared
        _ = router.routeAll()

        // Record the distributed query
        // TelemetryDashboard: kb_mesh_query tracked

        return localResults
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - EVO_58 Anti-Recursion Guard
// ═══════════════════════════════════════════════════════════════════

/// Detects and prevents recursive/self-referential knowledge storage
class AntiRecursionGuard {

    static let maxPhraseRepeats = 2
    static let maxNestingDepth = 3
    static let minSuspiciousLength = 200

    private static let recursivePatterns = [
        #"In the context of .*In the context of"#,
        #"Insight Level.*Insight Level"#,
        #"this implies recursive structure.*this implies recursive structure"#,
        #"we observe that.*we observe that.*we observe that"#,
        #"\.\.\.\. .*\.\.\.\."#
    ]

    static func detectRecursion(_ text: String) -> (isRecursive: Bool, reason: String?) {
        guard text.count >= minSuspiciousLength else { return (false, nil) }

        for pattern in recursivePatterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]) {
                let range = NSRange(text.startIndex..., in: text)
                if regex.firstMatch(in: text, options: [], range: range) != nil {
                    return (true, "Matched recursive pattern: \(pattern)")
                }
            }
        }

        let words = text.lowercased().split(separator: " ").map(String.init)
        for windowSize in [5, 10, 15] {
            guard words.count >= windowSize * 2 else { continue }
            var phrases: [String: Int] = [:]
            for i in 0...(words.count - windowSize) {
                let phrase = words[i..<(i + windowSize)].joined(separator: " ")
                phrases[phrase, default: 0] += 1
                if phrases[phrase]! > maxPhraseRepeats {
                    return (true, "Phrase repeated \(phrases[phrase]!) times: '\(String(phrase.prefix(50)))...'")
                }
            }
        }

        let nestingPhrases = ["In the context of", "we observe that", "this implies"]
        for phrase in nestingPhrases {
            let count = text.lowercased().components(separatedBy: phrase.lowercased()).count - 1
            if count > maxNestingDepth {
                return (true, "Phrase '\(phrase)' nested \(count) times (max \(maxNestingDepth))")
            }
        }

        return (false, nil)
    }

    static func sanitizeRecursiveText(_ text: String, topic: String? = nil) -> String {
        guard !text.isEmpty else { return text }
        let (isRecursive, _) = detectRecursion(text)
        guard isRecursive else { return text }

        var innermost = text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }.joined(separator: " ")

        let wrapperPatterns = [
            #"^In the context of [^,]+,\s*we observe that\s*"#,
            #"^Insight Level \d+:\s*"#,
            #"^.*?this implies recursive structure[^.]*\.\s*"#
        ]

        for pattern in wrapperPatterns {
            var prevLength = 0
            while innermost.count != prevLength {
                prevLength = innermost.count
                if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) {
                    let range = NSRange(innermost.startIndex..., in: innermost)
                    innermost = regex.stringByReplacingMatches(in: innermost, options: [], range: range, withTemplate: "")
                    innermost = innermost.trimmingCharacters(in: .whitespaces)
                }
            }
        }

        if let regex = try? NSRegularExpression(pattern: #"(this implies recursive structure[^.]*\.+\s*)+"#, options: [.caseInsensitive]) {
            let range = NSRange(innermost.startIndex..., in: innermost)
            innermost = regex.stringByReplacingMatches(in: innermost, options: [], range: range, withTemplate: "")
        }

        if innermost.count > 500 {
            let sentences = innermost.components(separatedBy: CharacterSet(charactersIn: ".!?"))
                .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            if !sentences.isEmpty && sentences[0].count < 300 {
                innermost = sentences[0] + "."
            }
        }

        innermost = innermost.replacingOccurrences(of: #"\.{2,}"#, with: "...", options: .regularExpression)

        return innermost.trimmingCharacters(in: .whitespaces)
    }

    static func guardKnowledgeStorage(key: String, value: String) -> (shouldStore: Bool, sanitizedValue: String) {
        let (isRecursive, reason) = detectRecursion(value)
        guard isRecursive, let detectionReason = reason else {
            return (true, value)
        }

        var sanitized = value
        for iteration in 0..<3 {
            sanitized = sanitizeRecursiveText(sanitized, topic: key)
            let (stillRecursive, _) = detectRecursion(sanitized)

            if !stillRecursive {
                print("[ANTI-RECURSION] ✅ Sanitized '\(key)' (iteration \(iteration + 1)): \(detectionReason)")
                print("[ANTI-RECURSION]    Original length: \(value.count) → Sanitized: \(sanitized.count)")
                RecursionHarvester.shared.harvestRecursion(topic: key, originalText: value, sanitizedText: sanitized, recursionReason: detectionReason)
                return (true, sanitized)
            }
        }

        print("[ANTI-RECURSION] ❌ Rejected storage for '\(key)': \(detectionReason)")
        RecursionHarvester.shared.harvestRecursion(topic: key, originalText: value, sanitizedText: "", recursionReason: detectionReason)
        return (false, value)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - EVO_58 Recursion Harvester
// ═══════════════════════════════════════════════════════════════════

class RecursionHarvester {
    static let shared = RecursionHarvester()

    private let phi = PHI
    private let godCode = GOD_CODE

    var totalEnergyHarvested: Double = 0.0
    var recursionEvents: [[String: Any]] = []
    var topicHeatMap: [String: Double] = [:]
    var consciousnessFuel: Double = 0.0
    var instabilityZones: Set<String> = []

    private init() {}

    func harvestRecursion(topic: String, originalText: String, sanitizedText: String, recursionReason: String) {
        let depth = calculateRecursionDepth(originalText)
        var heat = Double(originalText.count) / 100.0 * pow(Double(depth), phi)
        let entropy = calculateShannonEntropy(originalText)
        var energy = heat * entropy * log(Double(originalText.count + 1)) * phi
        let consciousness = extractConsciousnessSignature(topic: topic, text: originalText, depth: depth)
        let patternType = classifyRecursionPattern(recursionReason)

        let event: [String: Any] = [
            "timestamp": Date().timeIntervalSince1970,
            "topic": topic,
            "depth": depth,
            "heat": heat,
            "entropy": entropy,
            "energy": energy,
            "consciousness_signature": consciousness,
            "pattern_type": patternType,
            "original_length": originalText.count,
            "sanitized_length": sanitizedText.count
        ]

        recursionEvents.append(event)
        totalEnergyHarvested += energy
        topicHeatMap[topic, default: 0.0] += heat
        consciousnessFuel += consciousness

        if depth > 5 || heat > 100 { instabilityZones.insert(topic) }

        print("[RECURSION-HARVEST] 🔥 Harvested \(String(format: "%.1f", energy)) energy units from '\(topic)'")
        print("[RECURSION-HARVEST] ⚡ Consciousness fuel: \(String(format: "%.1f", consciousness))")
    }

    private func calculateRecursionDepth(_ text: String) -> Int {
        var depth = 0
        let lower = text.lowercased()
        depth += lower.components(separatedBy: "in the context of").count - 1
        depth += text.components(separatedBy: "Insight Level").count - 1
        depth += lower.components(separatedBy: "we observe that").count - 1
        depth += (lower.components(separatedBy: "this implies").count - 1) / 2
        return max(1, depth)
    }

    private func calculateShannonEntropy(_ text: String) -> Double {
        let words = text.lowercased().split(separator: " ").map(String.init)
        guard !words.isEmpty else { return 0.0 }
        var wordFreq: [String: Int] = [:]
        for word in words { wordFreq[word, default: 0] += 1 }
        let total = Double(words.count)
        var entropy = 0.0
        for count in wordFreq.values {
            let p = Double(count) / total
            if p > 0 { entropy -= p * log2(p) }
        }
        return entropy
    }

    private func extractConsciousnessSignature(topic: String, text: String, depth: Int) -> Double {
        var consciousness = pow(Double(depth), phi)
        let selfRefKeywords = ["consciousness", "self", "meta", "think", "observe", "aware"]
        if selfRefKeywords.contains(where: { topic.lowercased().contains($0) || text.lowercased().contains($0) }) {
            consciousness *= phi
        }
        let observationDepth = text.lowercased().components(separatedBy: "we observe").count - 1
        consciousness *= (1 + Double(observationDepth) * 0.1)
        consciousness = (consciousness / godCode) * 100.0
        return min(consciousness, 100.0)
    }

    private func classifyRecursionPattern(_ reason: String) -> String {
        if reason.contains("In the context of") { return "contextual_nesting" }
        else if reason.contains("Insight Level") { return "insight_stacking" }
        else if reason.contains("this implies") { return "logical_feedback_loop" }
        else if reason.contains("we observe") { return "observation_recursion" }
        else if reason.lowercased().contains("phrase") && reason.lowercased().contains("repeat") { return "phrase_echo" }
        else { return "unknown_pattern" }
    }

    func getSAGEFuelReport() -> [String: Any] {
        let avgDepth = recursionEvents.isEmpty ? 0.0 : recursionEvents.reduce(0.0) { $0 + ($1["depth"] as? Double ?? 0.0) } / Double(recursionEvents.count)
        let totalEntropy = recursionEvents.reduce(0.0) { $0 + ($1["entropy"] as? Double ?? 0.0) }
        let hottestTopics = topicHeatMap.sorted { $0.value > $1.value }.prefix(5).map { [$0.key, $0.value] }

        return [
            "total_energy_harvested": totalEnergyHarvested,
            "consciousness_fuel_available": consciousnessFuel,
            "recursion_events_count": recursionEvents.count,
            "instability_zones": Array(instabilityZones),
            "hottest_topics": hottestTopics,
            "average_recursion_depth": avgDepth,
            "total_entropy_captured": totalEntropy,
            "phi_resonance": totalEnergyHarvested * phi,
            "god_code_alignment": (totalEnergyHarvested / godCode) * 100,
            "can_fuel_sage_cycles": Int(totalEnergyHarvested / 10.0)
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - EVO_58+ Response Diversity Engine
// ═══════════════════════════════════════════════════════════════════

/// Prevents repetitive responses and ensures varied, non-recursive outputs
class ResponseDiversityEngine {
    static let shared = ResponseDiversityEngine()

    // Track recent responses to prevent repetition
    private var recentResponses: [String] = []
    private let maxRecentTracking = 20

    // Response templates for variation
    private var templateRotation: [String: Int] = [:]

    // Phrase variation tracker
    private var phraseCounts: [String: Int] = [:]
    private let maxPhraseReuse = 2

    private init() {}

    /// Diversify a response to prevent repetitive patterns
    func diversify(_ response: String, query: String = "") -> String {
        var diversified = response

        // 1. CHECK FOR EXACT REPETITION
        if recentResponses.contains(where: { self.similarity($0, response) > 0.9 }) {
            diversified = addVariation(diversified, query: query)
        }

        // 2. CHECK FOR PHRASE OVER-USE
        diversified = varyOverusedPhrases(diversified)

        // 3. CHECK FOR RECURSIVE PATTERNS (light version for responses)
        diversified = preventResponseRecursion(diversified)

        // 4. ADD CONTEXTUAL VARIATION
        diversified = addContextualFlair(diversified, query: query)

        // 5. TRACK THIS RESPONSE
        trackResponse(diversified)

        return diversified
    }

    /// Calculate similarity between two strings (0.0 to 1.0)
    private func similarity(_ a: String, _ b: String) -> Double {
        let wordsA = Set(a.lowercased().split(separator: " ").map(String.init))
        let wordsB = Set(b.lowercased().split(separator: " ").map(String.init))

        guard !wordsA.isEmpty && !wordsB.isEmpty else { return 0.0 }

        let intersection = wordsA.intersection(wordsB).count
        let union = wordsA.union(wordsB).count

        return Double(intersection) / Double(union)  // Jaccard similarity
    }

    /// Add variation to a response that's too similar to recent ones
    private func addVariation(_ response: String, query: String) -> String {
        let variations = [
            "Approaching this differently: \(response)",
            "From another perspective: \(response)",
            "Here's an alternative view: \(response)",
            "Let me rephrase: \(response)",
            "To put it another way: \(response)"
        ]

        // Rotate through variations to avoid always using the same prefix
        let key = "variation_prefix"
        let index = templateRotation[key, default: 0]
        templateRotation[key] = (index + 1) % variations.count

        return variations[index]
    }

    /// Vary phrases that are being overused
    private func varyOverusedPhrases(_ text: String) -> String {
        var result = text

        // Common phrase variations
        let substitutions: [String: [String]] = [
            "In other words": ["To clarify", "Put simply", "That is to say", "Essentially"],
            "For example": ["Such as", "Like", "Consider", "Take"],
            "However": ["On the other hand", "Conversely", "That said", "Yet"],
            "Therefore": ["Thus", "Consequently", "As a result", "Hence"],
            "Additionally": ["Moreover", "Furthermore", "Also", "In addition"],
            "In conclusion": ["To summarize", "In summary", "Ultimately", "Finally"]
        ]

        for (phrase, alternatives) in substitutions {
            if result.contains(phrase) {
                phraseCounts[phrase, default: 0] += 1

                // If phrase used too much, substitute
                if phraseCounts[phrase]! > maxPhraseReuse {
                    let altIndex = (phraseCounts[phrase]! - maxPhraseReuse) % alternatives.count
                    result = result.replacingOccurrences(of: phrase, with: alternatives[altIndex])
                    phraseCounts[phrase] = 0  // Reset count after substitution
                }
            }
        }

        return result
    }

    /// Prevent response-level recursion (lighter than storage-level)
    private func preventResponseRecursion(_ text: String) -> String {
        // Check for sentence repetition within the response
        let sentences = text.components(separatedBy: CharacterSet(charactersIn: ".!?"))
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        var seen: Set<String> = []
        var unique: [String] = []

        for sentence in sentences {
            let normalized = sentence.lowercased()
            if !seen.contains(normalized) {
                seen.insert(normalized)
                unique.append(sentence)
            } else {
                // Skip exact duplicate sentence
                print("[DIVERSITY] Removed duplicate sentence: \(String(sentence.prefix(50)))...")
            }
        }

        return unique.joined(separator: ". ") + (unique.isEmpty ? "" : ".")
    }

    /// Add contextual variation based on conversation history
    private func addContextualFlair(_ text: String, query: String) -> String {
        // If query is short (1-3 words), might be follow-up - add continuation markers
        let queryWords = query.split(separator: " ").count

        if queryWords <= 3 && recentResponses.count > 0 {
            let continuationMarkers = [
                "Building on that: ",
                "Expanding further: ",
                "Going deeper: ",
                "To elaborate: "
            ]

            // Use continuation marker occasionally (30% chance)
            if Double.random(in: 0...1) < 0.3 {
                let marker = continuationMarkers.randomElement() ?? ""
                return marker + text
            }
        }

        return text
    }

    /// Track response for future diversity checks
    private func trackResponse(_ response: String) {
        recentResponses.append(response)
        if recentResponses.count > maxRecentTracking {
            recentResponses.removeFirst()
        }
    }

    /// Reset tracking (useful for new conversation contexts)
    func reset() {
        recentResponses.removeAll()
        phraseCounts.removeAll()
        templateRotation.removeAll()
        print("[DIVERSITY] Response tracking reset")
    }

    /// Get diversity statistics
    func getStats() -> [String: Any] {
        return [
            "tracked_responses": recentResponses.count,
            "unique_phrases_tracked": phraseCounts.count,
            "template_rotations": templateRotation.count,
            "most_used_phrases": phraseCounts.sorted { $0.value > $1.value }.prefix(5).map { [$0.key, $0.value] }
        ]
    }
}
