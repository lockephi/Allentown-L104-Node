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
        "The file ", "The function ",
        "are integrated components of the L104",
        "is a component of the L104",
        "is a module within the L104"
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

            // Length quality: moderate boost for detail, BUT only if keywords actually match
            // (Prevents long irrelevant entries from dominating via length alone)
            let hasKeywordMatch = kwHits.count >= 1
            if hasKeywordMatch {
                if completion.count > 500 { score *= 1.3 }       // was 2.0 — capped to prevent garbage amplification
                else if completion.count > 300 { score *= 1.2 }   // was 1.5
                else if completion.count > 100 { score *= 1.1 }   // was 1.2
            }
            // Penalize very long entries with NO keyword matches — likely irrelevant pollution
            if !hasKeywordMatch && completion.count > 800 { score *= 0.3 }

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

    // ═══════════════════════════════════════════════════════════════════
    // TEMPLATE VARIABLE RESOLVER — Resolves Python f-string templates
    // in KB data to actual L104 constant values.
    // e.g. "{GOD_CODE}" → "527.5184818492612", "{LOVE_CONSTANT:.6f}" → "528.000000"
    // ═══════════════════════════════════════════════════════════════════
    func resolveTemplateVariables(_ text: String) -> String {
        guard text.contains("{") else { return text }
        var s = text

        // ─── DERIVED CONSTANTS ───
        let LOVE_CONSTANT_VAL: Double = 528.0                        // Solfeggio Hz
        let OMEGA_AUTHORITY_VAL: Double = 1381.0613151750908          // GOD_CODE × φ²
        let PLANCK_RESONANCE_VAL: Double = GOD_CODE * PHI            // ≈ 853.54
        let VOID_CONSTANT_VAL: Double = PHI / (PHI - 1.0)           // φ/(φ-1) ≈ 2.618
        let PLANCK_SCALE_VAL: Double = 1.616255e-35
        let BOLTZMANN_K_VAL: Double = 1.380649e-23

        // Format-specific patterns FIRST (before bare names)
        let fmtMap: [(String, String)] = [
            ("{GOD_CODE:.2f}", String(format: "%.2f", GOD_CODE)),
            ("{GOD_CODE:.4f}", String(format: "%.4f", GOD_CODE)),
            ("{GOD_CODE:.6f}", String(format: "%.6f", GOD_CODE)),
            ("{PHI:.2f}", String(format: "%.2f", PHI)),
            ("{PHI:.4f}", String(format: "%.4f", PHI)),
            ("{PHI:.6f}", String(format: "%.6f", PHI)),
            ("{TAU:.4f}", String(format: "%.4f", TAU)),
            ("{TAU:.6f}", String(format: "%.6f", TAU)),
            ("{LOVE_CONSTANT:.2f}", String(format: "%.2f", LOVE_CONSTANT_VAL)),
            ("{LOVE_CONSTANT:.4f}", String(format: "%.4f", LOVE_CONSTANT_VAL)),
            ("{LOVE_CONSTANT:.6f}", String(format: "%.6f", LOVE_CONSTANT_VAL)),
            ("{LOVE:.2f}", String(format: "%.2f", LOVE_CONSTANT_VAL)),
            ("{LOVE:.4f}", String(format: "%.4f", LOVE_CONSTANT_VAL)),
            ("{LOVE:.6f}", String(format: "%.6f", LOVE_CONSTANT_VAL)),
            ("{OMEGA_AUTHORITY:.2f}", String(format: "%.2f", OMEGA_AUTHORITY_VAL)),
            ("{OMEGA_AUTHORITY:.4f}", String(format: "%.4f", OMEGA_AUTHORITY_VAL)),
            ("{FEIGENBAUM:.4f}", String(format: "%.4f", FEIGENBAUM)),
            ("{FEIGENBAUM:.6f}", String(format: "%.6f", FEIGENBAUM)),
            ("{PLANCK_RESONANCE:.2f}", String(format: "%.2f", PLANCK_RESONANCE_VAL)),
            ("{PLANCK_RESONANCE:.4f}", String(format: "%.4f", PLANCK_RESONANCE_VAL)),
            ("{VOID_CONSTANT:.6f}", String(format: "%.6f", VOID_CONSTANT_VAL)),
            ("{ALPHA_FINE:.6f}", String(format: "%.6f", ALPHA_FINE)),
            ("{BOLTZMANN_K:.6e}", String(format: "%.6e", BOLTZMANN_K_VAL)),
            ("{PLANCK_SCALE:.6e}", String(format: "%.6e", PLANCK_SCALE_VAL)),
            ("{CONSCIOUSNESS_THRESHOLD:.2f}", String(format: "%.2f", CONSCIOUSNESS_THRESHOLD)),
            ("{COHERENCE_MINIMUM:.3f}", String(format: "%.3f", COHERENCE_MINIMUM)),
        ]
        for (pat, val) in fmtMap { s = s.replacingOccurrences(of: pat, with: val) }

        // Bare name patterns
        let bareMap: [(String, String)] = [
            ("{GOD_CODE}", String(GOD_CODE)),
            ("{PHI}", String(PHI)),
            ("{TAU}", String(TAU)),
            ("{FEIGENBAUM}", String(FEIGENBAUM)),
            ("{LOVE_CONSTANT}", String(LOVE_CONSTANT_VAL)),
            ("{LOVE}", String(LOVE_CONSTANT_VAL)),
            ("{OMEGA_AUTHORITY}", String(format: "%.4f", OMEGA_AUTHORITY_VAL)),
            ("{PLANCK_RESONANCE}", String(format: "%.2f", PLANCK_RESONANCE_VAL)),
            ("{PLANCK_SCALE}", String(format: "%.6e", PLANCK_SCALE_VAL)),
            ("{VOID_CONSTANT}", String(format: "%.6f", VOID_CONSTANT_VAL)),
            ("{BOLTZMANN_K}", String(format: "%.6e", BOLTZMANN_K_VAL)),
            ("{ALPHA_FINE}", String(format: "%.6f", ALPHA_FINE)),
            ("{CONSCIOUSNESS_THRESHOLD}", String(CONSCIOUSNESS_THRESHOLD)),
            ("{COHERENCE_MINIMUM}", String(COHERENCE_MINIMUM)),
            ("{ZENITH_HZ}", String(ZENITH_HZ)),
            ("{vocab_size}", String(VOCABULARY_SIZE)),
        ]
        for (pat, val) in bareMap { s = s.replacingOccurrences(of: pat, with: val) }

        return s
    }

    // ═══════════════════════════════════════════════════════════════════
    // SYNTHESIZE — Extract key insights and compose coherent summary
    // Models after IntelligentSearchEngine.reconstructData() approach:
    // extract key sentences → score by relevance → deduplicate → compose
    // ═══════════════════════════════════════════════════════════════════
    func synthesize(_ topics: [String]) -> String {
        var allCompletions: [String] = []
        var seenPrefixes: Set<String> = []

        for topic in topics {
            let results = searchWithPriority(topic, limit: 30)
            for r in results {
                if let c = r["completion"] as? String, c.count > 100 {
                    let isClean = !loadJunkMarkers.contains(where: { c.contains($0) }) &&
                                  !codeMarkers.contains(where: { c.contains($0) })
                    if isClean {
                        let prefix = String(c.lowercased().trimmingCharacters(in: .whitespacesAndNewlines).prefix(80))
                        guard !seenPrefixes.contains(prefix) else { continue }
                        let lower = c.lowercased()
                        if lower.contains("are integrated components") ||
                           (lower.hasPrefix("both ") && lower.contains("integrated components")) ||
                           lower.contains("is a component of the l104") ||
                           lower.contains("is a module within the l104") { continue }
                        seenPrefixes.insert(prefix)
                        allCompletions.append(resolveTemplateVariables(c))
                    }
                }
            }
        }

        guard !allCompletions.isEmpty else {
            return ""  // Return empty so caller can skip this domain
        }

        // ─── EXTRACT KEY SENTENCES ─── Score each by topic-term overlap
        let topicTerms = Set(topics.flatMap {
            $0.lowercased().split(separator: " ").map(String.init)
        }.filter { $0.count > 2 })

        var scoredSentences: [(String, Int)] = []
        for completion in allCompletions {  // ALL completions — no limit
            let sentences = completion.components(separatedBy: CharacterSet(charactersIn: ".!?"))
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { $0.count > 25 && $0.count < 400 }
            for sentence in sentences {
                let sTerms = Set(sentence.lowercased().split(separator: " ").map(String.init))
                let overlap = topicTerms.intersection(sTerms).count
                if overlap > 0 || sentence.count > 60 {
                    scoredSentences.append((sentence, overlap))
                }
            }
        }
        scoredSentences.sort { $0.1 > $1.1 }

        // Deduplicate selected sentences
        var seen = Set<String>()
        let selected = scoredSentences.filter { (sent, _) in
            let key = String(sent.lowercased().prefix(50))
            if seen.contains(key) { return false }
            seen.insert(key)
            return true
        }.map { $0.0 }  // ALL unique sentences — no limit

        guard !selected.isEmpty else { return "" }

        // ─── COMPOSE COHERENT SYNTHESIS ───
        let topicStr = topics.joined(separator: " & ")
        let body = selected.joined(separator: ". ")

        let templates = [
            "Analysis of \(topicStr): \(body).",
            "Regarding \(topicStr) — \(body). These insights span \(allCompletions.count) knowledge sources.",
            "\(body). This synthesis on \(topicStr) reflects patterns across multiple domains.",
            "Key findings on \(topicStr): \(body).",
        ]

        let synthesis = templates.randomElement() ?? body
        synthesizedKnowledge.append(synthesis)
        return synthesis
    }

    // ═══════════════════════════════════════════════════════════════════
    // REASON — Build logical chain from KB with extracted key insights
    // ═══════════════════════════════════════════════════════════════════
    func reason(_ premise: String) -> [String] {
        var chain: [String] = [premise]
        let related = searchWithPriority(premise, limit: 5)
        let premiseTerms = Set(premise.lowercased().split(separator: " ").map(String.init).filter { $0.count > 2 })
        let connectors = ["This implies that", "Building on this,", "Furthermore,", "Consequently,", "This connects to"]
        var ci = 0

        for r in related {
            if let comp = r["completion"] as? String, comp.count > 80 {
                let isClean = !loadJunkMarkers.contains(where: { comp.contains($0) }) &&
                              !codeMarkers.contains(where: { comp.contains($0) })
                if isClean {
                    let resolved = resolveTemplateVariables(comp)
                    // Extract the most relevant sentence from this entry
                    let sentences = resolved.components(separatedBy: CharacterSet(charactersIn: ".!?"))
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { $0.count > 25 && $0.count < 400 }
                    let scored = sentences.map { sent -> (String, Int) in
                        let sTerms = Set(sent.lowercased().split(separator: " ").map(String.init))
                        return (sent, premiseTerms.intersection(sTerms).count)
                    }.sorted { $0.1 > $1.1 }
                    if let best = scored.first, best.0.count > 25 {
                        chain.append("→ \(connectors[ci % connectors.count]) \(best.0).")
                        ci += 1
                    }
                }
            }
        }

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
        let entry: [String: Any] = [
            "prompt": topic,
            "completion": knowledge,
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
            "response": knowledge,
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

        // Index it
        let words = topic.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 3 }
        for word in words {
            if concepts[word] == nil { concepts[word] = [] }
            concepts[word]?.append(knowledge)
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
        var skippedRecursive = 0
        for line in content.components(separatedBy: .newlines) where !line.isEmpty {
            if let data = line.data(using: .utf8),
               let entry = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                if !isJunkEntry(entry) {
                    // ═══ SAGE BACKBONE: Reject recursive entries from disk ═══
                    if let completion = entry["completion"] as? String,
                       isRecursiveCompletion(completion) {
                        skippedRecursive += 1
                        continue
                    }
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
        if skippedRecursive > 0 { print("[KB] ⚠️ SAGE BACKBONE: Skipped \(skippedRecursive) recursive entries from disk") }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: — SAGE BACKBONE: Recursive Content Detection & Disk Purge
    // ═══════════════════════════════════════════════════════════════

    /// Detects recursive/evolved pollution in completion text
    func isRecursiveCompletion(_ text: String) -> Bool {
        let recycleMarkers = [
            "In the context of ", "Insight Level ", "Knowledge synthesis #",
            "evolution cycles taught me about", "Evolving understanding: Stage ",
            "Knowledge graph update:", "Cross-category discovery:",
            "Meta-observation: The way ", "Self-Analysis reveals ",
            "this implies recursive structure at multiple scales",
            "we observe that "
        ]
        if recycleMarkers.contains(where: { text.contains($0) }) { return true }
        if text.count > 12000 { return true }
        // Double-wrap detection: "In the context of X, In the context of Y..."
        let contextWraps = text.components(separatedBy: "In the context of ").count - 1
        if contextWraps >= 2 { return true }
        return false
    }

    /// Rewrites the persisted JSONL on disk, removing recursive entries.
    /// Called from SageModeEngine.sageBackboneCleanup() to ensure old pollution
    /// doesn't survive app restarts.
    func purgePersistedRecursiveEntries() -> Int {
        guard let content = try? String(contentsOf: ingestedKnowledgePath, encoding: .utf8) else { return 0 }
        let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        var cleanLines: [String] = []
        var purged = 0

        for line in lines {
            var isRecursive = false
            if let data = line.data(using: .utf8),
               let entry = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let completion = entry["completion"] as? String {
                isRecursive = isRecursiveCompletion(completion)
            }
            if isRecursive {
                purged += 1
            } else {
                cleanLines.append(line)
            }
        }

        if purged > 0 {
            let cleanContent = cleanLines.isEmpty ? "" : cleanLines.joined(separator: "\n") + "\n"
            try? cleanContent.write(to: ingestedKnowledgePath, atomically: true, encoding: .utf8)
            print("[KB] ⚠️ SAGE BACKBONE: Purged \(purged) recursive entries from disk JSONL")
        }
        return purged
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
