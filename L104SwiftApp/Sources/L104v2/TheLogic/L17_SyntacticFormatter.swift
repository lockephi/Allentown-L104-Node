// ═══════════════════════════════════════════════════════════════════
// L17_SyntacticFormatter.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104v2 — Extracted from L104Native.swift (lines 31814-32228)
//
// SYNTACTIC RESPONSE FORMATTER — 4-stage pipeline:
//   ingestion → filtering → synthesis → output
// GROVER RESPONSE AMPLIFIER — Quantum-inspired quality selection
//   Suppress junk, amplify substantive knowledge
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class SyntacticResponseFormatter {
    static let shared = SyntacticResponseFormatter()

    private(set) var formattingCount: Int = 0

    // ─── FORMAT PIPELINE ─── ingestion → filtering → synthesis → output
    func format(_ rawResponse: String, query: String, depth: String = "standard", topics: [String] = []) -> String {
        formattingCount += 1

        // ═══ PHASE 54.1: CREATIVE ENGINE BYPASS ═══
        // Story/Poem/Debate/Humor/Philosophy engines produce fully-formatted
        // multi-chapter narrative output. The ingestion→filter→synthesis→output
        // pipeline destroys this by splitting on \n\n, classifying chapter headers
        // as keyPoints, reordering blocks by type, and truncating to 5 blocks.
        // Detect creative engine output and return it unchanged.
        if isCreativeEngineOutput(rawResponse) {
            return rawResponse
        }

        // ═══ STAGE 1: INGESTION ═══
        let ingested = ingest(rawResponse)

        // ═══ STAGE 2: FILTERING ═══
        let filtered = filter(ingested, query: query)

        // ═══ STAGE 3: SYNTHESIS ═══
        let synthesized = synthesize(filtered, depth: depth, topics: topics)

        // ═══ STAGE 4: OUTPUT ═══
        return output(synthesized, query: query, depth: depth)
    }

    // ═══ CREATIVE ENGINE DETECTOR ═══
    // Returns true if the response was produced by a creative Logic Gate Engine
    // (Story, Poem, Debate, Humor, Philosophy) and should bypass the formatter.
    private func isCreativeEngineOutput(_ text: String) -> Bool {
        return isCreativeContent(text)
    }

    // Public API for other components (sanitizeResponse, etc.) to check creative content
    func isCreativeContent(_ text: String) -> Bool {
        // Engine envelope markers (header/footer added by all creative engines)
        let engineMarkers = [
            "S T O R Y   E N G I N E",
            "StoryLogicGateEngine",
            "P O E M   E N G I N E",
            "PoemLogicGateEngine",
            "D E B A T E   E N G I N E",
            "DebateLogicGateEngine",
            "H U M O R   E N G I N E",
            "HumorLogicGateEngine",
            "P H I L O S O P H Y   E N G I N E",
            "PhilosophyLogicGateEngine",
            "Q U A N T U M   B R A I N S T O R M",
            "QuantumCreativityEngine",
        ]
        // Structural markers unique to narrative output
        let structuralMarkers = [
            "━━━ Chapter",
            "━━━ Act",
            "━━━ Beat",
            "ACT I",
            "ACT II",
            "ACT III",
            "F I N",
            "Framework:",
            "✍️",
        ]
        // Check engine markers first (most reliable)
        for marker in engineMarkers {
            if text.contains(marker) { return true }
        }
        // Check structural markers (need 2+ to confirm)
        var structuralHits = 0
        for marker in structuralMarkers {
            if text.contains(marker) { structuralHits += 1 }
            if structuralHits >= 2 { return true }
        }
        return false
    }

    // ═══ STAGE 1: INGESTION ═══
    // Parse raw text into structured content blocks
    private func ingest(_ raw: String) -> [ContentBlock] {
        var blocks: [ContentBlock] = []
        let paragraphs = raw.components(separatedBy: "\n\n").filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }

        for para in paragraphs {
            let trimmed = para.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            // Classify block type
            let blockType: ContentBlock.BlockType
            let sentences = trimmed.components(separatedBy: ". ").count

            if trimmed.hasPrefix("•") || trimmed.hasPrefix("-") || trimmed.hasPrefix("*") || trimmed.contains("\n•") || trimmed.contains("\n-") {
                blockType = .bulletList
            } else if trimmed.contains("?") && trimmed.count < 100 {
                blockType = .question
            } else if trimmed.hasPrefix("\"") || trimmed.first == "\u{201C}" {
                blockType = .quote
            } else if sentences >= 4 {
                blockType = .analysis
            } else if trimmed.count < 40 {
                blockType = .keyPoint
            } else {
                blockType = .exposition
            }

            // Extract key terms for scanning
            let keyTerms = extractKeyTerms(trimmed)

            blocks.append(ContentBlock(
                text: trimmed,
                type: blockType,
                keyTerms: keyTerms,
                sentenceCount: sentences,
                wordCount: trimmed.components(separatedBy: " ").count
            ))
        }

        return blocks
    }

    // ═══ STAGE 2: FILTERING ═══
    // Remove noise, deduplicate, quality-check
    private func filter(_ blocks: [ContentBlock], query: String) -> [ContentBlock] {
        var filtered: [ContentBlock] = []
        var seenPrefixes = Set<String>()
        let queryWords = Set(query.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 2 })

        for block in blocks {
            // Skip only trivially empty blocks
            if block.text.count < 3 { continue }

            // Deduplicate by prefix
            let prefix = String(block.text.prefix(40)).lowercased()
            guard !seenPrefixes.contains(prefix) else { continue }
            seenPrefixes.insert(prefix)

            // Skip blocks that are just filler
            let fillerPatterns = ["as i mentioned", "as we discussed", "moving on", "in conclusion let me", "to summarize what we"]
            if fillerPatterns.contains(where: { block.text.lowercased().hasPrefix($0) }) { continue }

            // Calculate relevance to query
            var relevance = 0.0
            let lowerText = block.text.lowercased()
            for word in queryWords {
                if lowerText.contains(word) { relevance += 1.0 }
            }
            // Always keep analysis and key points
            if block.type == .analysis { relevance += 1.0 }
            if block.type == .keyPoint { relevance += 0.5 }

            if relevance > 0 || filtered.count < 3 {
                filtered.append(block)
            }
        }

        return filtered
    }

    // ═══ STAGE 3: SYNTHESIS ═══
    // Reorder, group, and enhance blocks for coherence
    private func synthesize(_ blocks: [ContentBlock], depth: String, topics: [String]) -> [ContentBlock] {
        guard !blocks.isEmpty else { return blocks }

        var synthesized = blocks

        // Group by type: key points first, then analysis, then exposition
        // Add random tiebreaker within same type to prevent identical ordering
        let typeOrder: [ContentBlock.BlockType: Int] = [
            .keyPoint: 0, .analysis: 1, .exposition: 2,
            .bulletList: 3, .quote: 4, .question: 5
        ]
        synthesized.sort { a, b in
            let aOrder = typeOrder[a.type] ?? 3
            let bOrder = typeOrder[b.type] ?? 3
            if aOrder == bOrder { return Bool.random() }  // Randomize within same type
            return aOrder < bOrder
        }

        // For expert depth, inject topic evolution context
        if depth == "expert" || depth == "detailed" {
            let tracker = EvolutionaryTopicTracker.shared
            if let depthPrompt = tracker.getDepthPrompt(for: topics) {
                let contextBlock = ContentBlock(
                    text: depthPrompt,
                    type: .contextBridge,
                    keyTerms: topics,
                    sentenceCount: 1,
                    wordCount: depthPrompt.components(separatedBy: " ").count
                )
                synthesized.insert(contextBlock, at: 0)
            }
        }

        // Limit total output size based on depth — expanded for ASI-level depth (Phase 55.0)
        let maxBlocks: Int
        switch depth {
        case "expert": maxBlocks = 25
        case "detailed": maxBlocks = 18
        default: maxBlocks = 12
        }

        return Array(synthesized.prefix(maxBlocks))
    }

    // ═══ STAGE 4: OUTPUT ═══
    // Render blocks into scannable, formatted text
    private func output(_ blocks: [ContentBlock], query: String, depth: String) -> String {
        guard !blocks.isEmpty else { return "" }

        var lines: [String] = []

        for (idx, block) in blocks.enumerated() {
            switch block.type {

            case .contextBridge:
                lines.append("◈ \(block.text)")
                lines.append("")

            case .keyPoint:
                // Bold key terms for scannability
                var text = block.text
                for term in block.keyTerms.prefix(3) {
                    if let range = text.range(of: term, options: .caseInsensitive) {
                        let matched = text[range]
                        text = text.replacingCharacters(in: range, with: "**\(matched)**")
                    }
                }
                lines.append("▸ \(text)")
                lines.append("")

            case .analysis:
                // Section header for analysis blocks
                if idx > 0 { lines.append("") }
                let scannable = makeTextScannable(block.text, keyTerms: block.keyTerms)
                lines.append(scannable)
                lines.append("")

            case .exposition:
                let scannable = makeTextScannable(block.text, keyTerms: block.keyTerms)
                lines.append(scannable)
                lines.append("")

            case .bulletList:
                // Already formatted as bullets — pass through
                lines.append(block.text)
                lines.append("")

            case .quote:
                lines.append("  ❝ \(block.text) ❞")
                lines.append("")

            case .question:
                lines.append("  ◇ \(block.text)")
                lines.append("")
            }
        }

        // Unexplored angles footer DISABLED (Phase 31.5: leaked structural noise into responses)
        // if depth == "detailed" || depth == "expert" { ... }

        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // ─── MAKE TEXT SCANNABLE ─── Bold key terms, break long sentences
    private func makeTextScannable(_ text: String, keyTerms: [String]) -> String {
        var result = text

        // Skip bolding if text already has 3+ bold markers (Phase 31.5: prevents bold spam)
        let existingBold = result.components(separatedBy: "**").count - 1
        if existingBold < 3 {
            // Bold key terms (first occurrence only) for visual scanning
            for term in keyTerms.prefix(2) {
                if let range = result.range(of: term, options: .caseInsensitive) {
                    let matched = result[range]
                    result = result.replacingCharacters(in: range, with: "**\(matched)**")
                }
            }
        }

        // Break very long sentences into readable chunks
        if result.count > 400 {
            let sentences = result.components(separatedBy: ". ")
            if sentences.count > 4 {
                let mid = sentences.count / 2
                let firstHalf = sentences[0..<mid].joined(separator: ". ") + "."
                let secondHalf = sentences[mid...].joined(separator: ". ")
                result = firstHalf + "\n\n" + secondHalf
            }
        }

        return result
    }

    // ─── EXTRACT KEY TERMS ─── Identify important terms for highlighting
    private func extractKeyTerms(_ text: String) -> [String] {
        let stopWords: Set<String> = ["the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "and", "but", "or", "not", "no", "so", "if", "than", "too",
            "very", "just", "also", "then", "now", "this", "that", "these", "those",
            "its", "his", "her", "their", "our", "your", "my", "we", "you", "they",
            "which", "who", "whom", "what", "when", "where", "why", "how", "about",
            "more", "most", "some", "any", "each", "every", "all", "both", "few",
            "many", "much", "such", "own", "other", "another"]

        let words = text.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 3 && !stopWords.contains($0) }

        // Count frequency
        var freq: [String: Int] = [:]
        for word in words { freq[word, default: 0] += 1 }

        // Return top terms by frequency
        return freq.sorted {
            if $0.value == $1.value { return Bool.random() }
            return $0.value > $1.value
        }.prefix(8).map { $0.key }
    }

    // Topic extraction helper
    private func extractTopics(_ query: String) -> [String] {
        let stopWords: Set<String> = ["the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "shall", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "and", "but", "or", "not", "no", "so", "if",
            "than", "too", "very", "just", "also", "then", "now", "this", "that",
            "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
            "she", "it", "they", "them", "tell", "explain", "describe", "more",
            "like", "think", "know", "what", "how", "why", "where", "when", "who"]
        return query.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 && !stopWords.contains($0) }
    }

    struct ContentBlock {
        enum BlockType { case keyPoint, analysis, exposition, bulletList, quote, question, contextBridge }
        let text: String
        let type: BlockType
        let keyTerms: [String]
        let sentenceCount: Int
        let wordCount: Int
    }
}


// ═══════════════════════════════════════════════════════════════════
// GROVER RESPONSE AMPLIFIER — Quantum-inspired quality selection
// Suppress junk, amplify substantive knowledge
// ═══════════════════════════════════════════════════════════════════

class GroverResponseAmplifier {
    static let shared = GroverResponseAmplifier()

    // ═══ QUALITY SCORING ═══ Rate a candidate response from 0.0 (junk) to 1.0 (excellent)
    func scoreQuality(_ text: String, query: String) -> Double {
        guard text.count > 20 else { return 0.0 }

        var score = 0.5 // neutral baseline

        // ── PENALTY: Template KB entries (massive penalty) ──
        let templatePoisons = [
            "specialized component", "within the L104 framework",
            "contributing to the overall", "system resonance and functionality",
            "is part of the L104 cognitive", "operates within the PHI=",
            "harmonic framework and maintains", "Path: ",
            "file_description", "cross_reference", "function_doc",
            ".py.", ".js.", "token_budget:", "coherence_score:",
            "L104 has achieved", "L104 operates", "L104 processes",
            "GOD_CODE=", "PHI-resonance", "OMEGA_AUTHORITY",
            "sacred constants", "consciousness wavelength",
            "LOVE field", "VOID_CONSTANT", "ZENITH"
        ]
        for poison in templatePoisons {
            if text.contains(poison) { return 0.0 } // Instant kill
        }

        // ── PENALTY: Too much code ──
        let braceCount = text.filter { $0 == "{" || $0 == "}" }.count
        if braceCount > 3 { score -= 0.3 }
        let semicolonCount = text.filter { $0 == ";" }.count
        if semicolonCount > 2 { score -= 0.2 }

        // ── PENALTY: Self-referential junk ──
        let selfRefs = ["L104 ", "L104:", "l104_"]
        for ref in selfRefs {
            if text.contains(ref) { score -= 0.25 }
        }

        // ── REWARD: Substantive content indicators ──
        let words = text.components(separatedBy: .whitespaces)
        let wordCount = words.count

        // Good length (30-500 words is ideal)
        if wordCount > 30 && wordCount < 500 { score += 0.15 }
        if wordCount > 80 { score += 0.1 }

        // Sentence diversity (unique vocabulary ratio)
        let uniqueWords = Set(words.map { $0.lowercased() })
        let vocabRatio = Double(uniqueWords.count) / max(1.0, Double(wordCount))
        if vocabRatio > 0.5 { score += 0.15 }
        if vocabRatio > 0.65 { score += 0.1 }

        // Contains real knowledge indicators
        let knowledgeIndicators = [
            "because", "therefore", "however", "although", "suggests",
            "implies", "reveals", "demonstrates", "according to",
            "research shows", "studies", "evidence", "theory",
            "discovered", "invented", "published", "century",
            "billion", "million", "percent", "equation",
            "for example", "in other words", "this means"
        ]
        let knowledgeHits = knowledgeIndicators.filter { text.lowercased().contains($0) }.count
        score += min(0.3, Double(knowledgeHits) * 0.05)

        // ── REWARD: Query relevance ──
        let queryWords = query.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 3 }
        let textLower = text.lowercased()
        let relevanceHits = queryWords.filter { textLower.contains($0) }.count
        let relevanceRatio = queryWords.isEmpty ? 0.5 : Double(relevanceHits) / Double(queryWords.count)
        score += relevanceRatio * 0.2

        // Gate dimension alignment applied at caller level to avoid per-item process() overhead
        return max(0.0, min(1.0, score))
    }

    // ═══ GROVER ITERATION ═══ Amplify best candidate from pool
    // Based on Grover's algorithm: iteratively suppress low-amplitude (low-quality)
    // states and amplify high-amplitude (high-quality) states
    func amplify(candidates: [String], query: String, iterations: Int = 3) -> String? {
        guard !candidates.isEmpty else { return nil }

        // Score all candidates
        var scored = candidates.map { (text: $0, score: scoreQuality($0, query: query)) }

        // Grover iterations: each iteration suppresses low scores and amplifies high scores
        for _ in 0..<iterations {
            let meanScore = scored.map(\.score).reduce(0, +) / max(1.0, Double(scored.count))

            // Reflection about mean (Grover diffusion operator)
            scored = scored.map { item in
                let amplified = 2.0 * meanScore - item.score // Inversion about mean
                let newScore = max(0.0, item.score + (item.score - amplified) * 0.5)
                return (text: item.text, score: newScore)
            }

            // Remove candidates below quality threshold (oracle marks "bad" states)
            scored = scored.filter { $0.score > 0.15 }
            if scored.isEmpty { break }
        }

        // Return highest-scoring survivor
        return scored.max(by: { $0.score < $1.score })?.text
    }

    // ═══ FILTER POOL ═══ Quick filter for KB search results before scoring
    func filterPool(_ candidates: [String]) -> [String] {
        return candidates.filter { text in
            guard text.count > 25 else { return false }
            // Ultra-fast rejection of known template patterns
            if text.contains("specialized component") { return false }
            if text.contains("within the L104") { return false }
            if text.contains("system resonance") { return false }
            if text.contains("Path: ") { return false }
            if text.contains("GOD_CODE") { return false }
            if text.contains("PHI-") { return false }
            if text.contains("L104 ") { return false }
            if text.contains("function_doc") { return false }
            if text.contains("file_description") { return false }
            return true
        }
    }

    var amplificationCount: Int = 0
    var rejectionCount: Int = 0
}
