// ═══════════════════════════════════════════════════════════════════
// L16_NLPEngines.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104v2 — Extracted from L104Native.swift (lines 11331-11526)
//
// SMART TOPIC EXTRACTOR — NLTagger-powered noun phrase extraction
// with concept dictionary and alias resolution
// PRONOUN RESOLVER — Context-aware coreference resolution with
// NLTagger POS analysis
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class SmartTopicExtractor {
    static let shared = SmartTopicExtractor()

    private var knownConcepts: Set<String> = []  // Pre-seeded from KB prompts
    private var conceptAliases: [String: String] = [:]  // "ML" → "machine learning"
    private var initialized = false

    func initialize(from kb: ASIKnowledgeBase) {
        guard !initialized else { return }
        // Build concept dictionary from KB prompts
        for entry in kb.trainingData {
            if let prompt = entry["prompt"] as? String {
                let clean = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
                if clean.count > 3 && clean.count < 60 {
                    knownConcepts.insert(clean)
                }
                // Extract 2-3 word phrases as known concepts
                let words = clean.components(separatedBy: .whitespaces)
                if words.count >= 2 && words.count <= 4 {
                    knownConcepts.insert(words.joined(separator: " "))
                }
            }
        }
        // Common aliases
        conceptAliases = [
            "ml": "machine learning", "ai": "artificial intelligence",
            "qm": "quantum mechanics", "gr": "general relativity",
            "ode": "ordinary differential equation", "pde": "partial differential equation",
            "nn": "neural network", "dna": "deoxyribonucleic acid",
            "rna": "ribonucleic acid", "cpu": "central processing unit",
            "gpu": "graphics processing unit", "nlp": "natural language processing",
            "cv": "computer vision", "rl": "reinforcement learning",
        ]
        initialized = true
    }

    /// Extract topics using NLTagger noun phrases + known concept matching
    func extractTopics(_ query: String) -> [String] {
        let q = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        // 1. Check for known multi-word concepts (longest match first)
        var matchedConcepts: [String] = []
        let sortedConcepts = knownConcepts.sorted { $0.count > $1.count }
        for concept in sortedConcepts.prefix(2000) {
            if q.contains(concept) && concept.count > 4 {
                matchedConcepts.append(concept)
                if matchedConcepts.count >= 5 { break }
            }
        }

        // 2. Expand aliases
        for (alias, full) in conceptAliases {
            if q.components(separatedBy: .whitespaces).contains(alias) {
                matchedConcepts.append(full)
            }
        }

        // 3. NLTagger noun extraction
        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = query
        var nounPhrases: [String] = []
        var currentPhrase: [String] = []

        tagger.enumerateTags(in: query.startIndex..<query.endIndex, unit: .word, scheme: .lexicalClass) { tag, range in
            let word = String(query[range]).lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            if let tag = tag {
                if tag == .noun || tag == .adjective {
                    currentPhrase.append(word)
                } else {
                    if !currentPhrase.isEmpty {
                        let phrase = currentPhrase.joined(separator: " ")
                        if phrase.count > 2 {
                            nounPhrases.append(phrase)
                        }
                        currentPhrase = []
                    }
                }
            }
            return true
        }
        // Flush remaining phrase
        if !currentPhrase.isEmpty {
            let phrase = currentPhrase.joined(separator: " ")
            if phrase.count > 2 { nounPhrases.append(phrase) }
        }

        // 4. Merge and deduplicate: known concepts first, then noun phrases
        var seen = Set<String>()
        var result: [String] = []
        for topic in matchedConcepts + nounPhrases {
            let t = topic.trimmingCharacters(in: .whitespacesAndNewlines)
            guard t.count > 2, !seen.contains(t) else { continue }
            seen.insert(t)
            result.append(t)
        }

        // 5. Resonance sort
        return result.sorted { t1, t2 in
            let r1 = HyperBrain.shared.longTermPatterns[t1] ?? 0.0
            let r2 = HyperBrain.shared.longTermPatterns[t2] ?? 0.0
            if abs(r1 - r2) < 0.1 { return Bool.random() }
            if r1 != r2 { return r1 > r2 }
            return t1.count > t2.count
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// PRONOUN RESOLVER — Context-aware coreference resolution
// with NLTagger POS analysis
// ═══════════════════════════════════════════════════════════════════

class PronounResolver {
    static let shared = PronounResolver()

    private var entityHistory: [(entity: String, timestamp: Date, type: EntityType)] = []

    enum EntityType {
        case singular   // it, this, that
        case plural     // they, those, these
        case person     // he, she, they (person)
        case place      // there
    }

    /// Record entities mentioned in a message for future pronoun resolution
    func recordEntities(from message: String) {
        let extractor = SmartTopicExtractor.shared
        let topics = extractor.extractTopics(message)
        let now = Date()
        for topic in topics {
            entityHistory.append((entity: topic, timestamp: now, type: classifyEntity(topic)))
        }
        // Keep last 50 entities
        if entityHistory.count > 50 {
            entityHistory = Array(entityHistory.suffix(50))
        }
    }

    /// Resolve pronouns in a query using entity history
    func resolve(_ query: String) -> String {
        let q = query.lowercased()
        let pronouns: [(pattern: String, type: EntityType)] = [
            ("it", .singular), ("this", .singular), ("that", .singular),
            ("they", .plural), ("those", .plural), ("these", .plural),
            ("there", .place),
        ]

        var resolved = query
        for (pronoun, type) in pronouns {
            // Check if the query contains a pronoun in a referential position
            let patterns = [
                "about \(pronoun)", "is \(pronoun)", "does \(pronoun)", "did \(pronoun)",
                "was \(pronoun)", "of \(pronoun)", "with \(pronoun)", "for \(pronoun)",
                "\(pronoun)?", "\(pronoun) work", "\(pronoun) mean", "what \(pronoun)",
                "why \(pronoun)", "how \(pronoun)", "explain \(pronoun)", "more \(pronoun)",
            ]

            for pattern in patterns {
                if q.contains(pattern) {
                    if let entity = findBestEntity(type: type) {
                        resolved = resolved.replacingOccurrences(
                            of: pattern, with: pattern.replacingOccurrences(of: pronoun, with: entity),
                            options: .caseInsensitive
                        )
                        break
                    }
                }
            }
        }
        return resolved
    }

    private func findBestEntity(type: EntityType) -> String? {
        // Recency-weighted: most recent entity of matching type
        let candidates = entityHistory.reversed()
        for entry in candidates {
            if entry.type == type || type == .singular {
                return entry.entity
            }
        }
        // Fallback: just the most recent entity
        return entityHistory.last?.entity
    }

    private func classifyEntity(_ entity: String) -> EntityType {
        let words = entity.components(separatedBy: .whitespaces)
        if words.count > 1 || entity.hasSuffix("s") { return .plural }
        return .singular
    }
}
