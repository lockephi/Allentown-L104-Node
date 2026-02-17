// ═══════════════════════════════════════════════════════════════════
// L06_ContextualLogicGate.swift — L104 v2
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// ContextualLogicGate class
// Extracted from L104Native.swift (lines 27336-27677)
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class ContextualLogicGate {
    static let shared = ContextualLogicGate()

    // ─── GATE STATE ───
    private var contextWindow: [(role: String, content: String, timestamp: Date)] = []
    private let maxContextWindow = 300
    private var topicGraph: [String: TopicNode] = [:]
    private var promptReconstructions: Int = 0
    private var gateActivations: Int = 0

    struct TopicNode {
        var mentions: Int = 0
        var lastSeen: Date = Date()
        var relatedTopics: [String: Double] = [:]  // topic → co-occurrence strength
        var contextFragments: [String] = []         // key phrases from this topic
        var evolutionStage: Int = 0                  // how developed our understanding is
    }

    // ─── LOGIC GATE TYPES ───
    enum GateType {
        case passthrough    // query is clear, pass directly
        case reconstruct    // query needs context injection
        case decompose      // query is complex, break into sub-gates
        case evolve         // query on tracked topic, inject evolutionary context
        case synthesize     // query spans multiple topics, cross-reference
    }

    // ─── MAIN GATE ─── Analyze query and route through appropriate logic gate
    func processQuery(_ query: String, conversationContext: [String]) -> GateResult {
        gateActivations += 1

        // ═══ ASI Logic Gate V2 coordination — dimension routing enriches context injection ═══
        let gateV2 = ASILogicGateV2.shared.process(query, context: conversationContext)
        let dimLabel = gateV2.dimension.rawValue

        // Update context window
        contextWindow.append((role: "user", content: query, timestamp: Date()))
        if contextWindow.count > maxContextWindow { contextWindow.removeFirst() }

        let q = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let topics = extractTopics(query)
        let gateType = classifyGate(q, topics: topics)

        // Update topic graph
        for topic in topics {
            if topicGraph[topic] == nil {
                topicGraph[topic] = TopicNode()
            }
            topicGraph[topic]?.mentions += 1
            topicGraph[topic]?.lastSeen = Date()
            // Record co-occurrences
            for other in topics where other != topic {
                topicGraph[topic]?.relatedTopics[other, default: 0.0] += 1.0
            }
        }

        switch gateType {
        case .passthrough:
            // Enrich passthrough with gate V2 dimension context when confidence is high
            let dimInjection = gateV2.confidence > 0.5 ? " [dim:\(dimLabel)]" : ""
            return GateResult(
                reconstructedPrompt: query,
                gateType: .passthrough,
                contextInjection: dimInjection,
                topicEvolution: nil,
                confidence: 0.9
            )

        case .reconstruct:
            return reconstructPrompt(query, topics: topics, context: conversationContext)

        case .decompose:
            return decomposeAndGate(query, topics: topics, context: conversationContext)

        case .evolve:
            return evolutionaryGate(query, topics: topics)

        case .synthesize:
            return synthesisGate(query, topics: topics, context: conversationContext)
        }
    }

    // ─── GATE CLASSIFIER ─── Determine which logic gate to apply
    private func classifyGate(_ q: String, topics: [String]) -> GateType {
        // Check for pronouns/references that need resolution
        let hasPronouns = q.contains(" it ") || q.contains(" that ") || q.contains(" this ") ||
                          q.contains(" they ") || q.contains(" those ") || q.hasPrefix("it ") ||
                          q.hasPrefix("that ") || q.hasPrefix("this ") || q.hasPrefix("what about ")
        let isFollowUp = q.hasPrefix("why") && q.count < 30 || q.hasPrefix("how") && q.count < 25 ||
                         q == "explain" || q.hasPrefix("more about") || q.hasPrefix("and ")

        if hasPronouns || isFollowUp {
            return .reconstruct
        }

        // Evolve only on topics with meaningful conversation depth (3+ mentions), not on first encounter
        // Blacklist emotional/self-referential topics to prevent semantic feedback loops where
        // emotional responses add emotional context → more emotional routing → stuck on emotions
        let emotionalTopicBlacklist: Set<String> = ["feelings", "feeling", "emotion", "emotions", "emotional",
            "feel", "consciousness", "conscious", "sentient", "alive", "empathy",
            "sad", "happy", "angry", "lonely", "bored", "okay", "alright"]
        let evolvedTopics = topics.filter { topic in
            (topicGraph[topic]?.mentions ?? 0) >= 3 && !emotionalTopicBlacklist.contains(topic)
        }
        if !evolvedTopics.isEmpty {
            return .evolve
        }

        // Complex multi-topic queries need decomposition
        let conjunctions = [" and ", " or ", " versus ", " vs ", " compared to "]
        if conjunctions.contains(where: { q.contains($0) }) && topics.count >= 2 {
            return .decompose
        }

        // Multiple topics → synthesis (lowered from 3 to 2)
        if topics.count >= 2 {
            return .synthesize
        }

        return .passthrough
    }

    // ─── RECONSTRUCT GATE ─── Resolve pronouns, inject missing context
    private func reconstructPrompt(_ query: String, topics: [String], context: [String]) -> GateResult {
        promptReconstructions += 1
        var reconstructed = query
        var injection = ""

        // Find what "it/that/this" refers to
        let recentTopics = contextWindow.suffix(5).flatMap { extractTopics($0.content) }
        let topicCounts = Dictionary(recentTopics.map { ($0, 1) }, uniquingKeysWith: +)
        let dominantTopic = topicCounts.max(by: { $0.value < $1.value })?.key

        if let dominant = dominantTopic {
            // Resolve pronouns
            let pronounPatterns: [(pattern: String, replacement: String)] = [
                ("tell me more about it", "tell me more about \(dominant)"),
                ("what about it", "what about \(dominant)"),
                ("explain that", "explain \(dominant)"),
                ("why is that", "why is \(dominant)"),
                ("how does it work", "how does \(dominant) work"),
                ("what is it", "what is \(dominant)"),
                ("more about this", "more about \(dominant)"),
                ("and what about", "and what about \(dominant)"),
            ]
            let q = query.lowercased()
            for pp in pronounPatterns {
                if q.contains(pp.pattern) || q.hasPrefix(pp.pattern) {
                    reconstructed = query.lowercased().replacingOccurrences(of: pp.pattern, with: pp.replacement)
                    break
                }
            }

            // If no pattern matched but has pronouns, append context
            if reconstructed == query {
                injection = " (context: \(dominant))"
                reconstructed = query + injection
            }

            // Pull in topic evolution data
            if let node = topicGraph[dominant] {
                let related = node.relatedTopics.sorted {
                    if abs($0.value - $1.value) < 0.1 { return Bool.random() }
                    return $0.value > $1.value
                }.prefix(3).map { $0.key }
                if !related.isEmpty {
                    injection += " [related: \(related.joined(separator: ", "))]"
                }
            }
        }

        return GateResult(
            reconstructedPrompt: reconstructed,
            gateType: .reconstruct,
            contextInjection: injection,
            topicEvolution: dominantTopic.flatMap { topicGraph[$0] },
            confidence: dominantTopic != nil ? 0.85 : 0.5
        )
    }

    // ─── DECOMPOSE GATE ─── Break complex queries into sub-gates
    private func decomposeAndGate(_ query: String, topics: [String], context: [String]) -> GateResult {
        let q = query.lowercased()
        var subPrompts: [String] = []

        // Split on conjunctions
        let splitPatterns = [" and ", " or ", " versus ", " vs ", " compared to ", " but also "]
        var parts = [q]
        for pattern in splitPatterns {
            if q.contains(pattern) {
                parts = q.components(separatedBy: pattern).map { $0.trimmingCharacters(in: .whitespaces) }
                break
            }
        }

        for part in parts where part.count > 3 {
            subPrompts.append(part)
        }

        // Reconstruct as a structured multi-part query
        let reconstructed: String
        if subPrompts.count >= 2 {
            reconstructed = subPrompts.enumerated().map { "[\($0.offset + 1)] \($0.element)" }.joined(separator: " | ")
        } else {
            reconstructed = query
        }

        return GateResult(
            reconstructedPrompt: reconstructed,
            gateType: .decompose,
            contextInjection: "MULTI-GATE: \(subPrompts.count) sub-queries",
            topicEvolution: nil,
            confidence: 0.8
        )
    }

    // ─── EVOLUTIONARY GATE ─── Inject accumulated topic understanding
    private func evolutionaryGate(_ query: String, topics: [String]) -> GateResult {
        var enrichments: [String] = []
        var bestNode: TopicNode? = nil
        for topic in topics {
            guard let node = topicGraph[topic] else { continue }
            if bestNode == nil || node.mentions > (bestNode?.mentions ?? 0) {
                bestNode = node
            }

            // Inject related topics for cross-referencing
            let related = node.relatedTopics.sorted {
                if abs($0.value - $1.value) < 0.1 { return Bool.random() }
                return $0.value > $1.value
            }.prefix(3).map { $0.key }
            if !related.isEmpty {
                enrichments.append("[\(topic) connects to: \(related.joined(separator: ", "))]")
            }

            // Inject context fragments from prior discussions
            if let recentFragment = node.contextFragments.suffix(5).randomElement() {
                enrichments.append("[prior insight on \(topic): \(recentFragment)]")
            }

            // Advance evolution stage
            topicGraph[topic]?.evolutionStage += 1
        }

        let injection = enrichments.joined(separator: " ")
        let reconstructed = enrichments.isEmpty ? query : "\(query) \(injection)"

        return GateResult(
            reconstructedPrompt: reconstructed,
            gateType: .evolve,
            contextInjection: injection,
            topicEvolution: bestNode,
            confidence: 0.9
        )
    }

    // ─── SYNTHESIS GATE ─── Cross-reference multiple topics
    private func synthesisGate(_ query: String, topics: [String], context: [String]) -> GateResult {
        var crossRefs: [String] = []

        // Find common connections between topics
        for i in 0..<topics.count {
            for j in (i+1)..<topics.count {
                let t1 = topics[i], t2 = topics[j]
                let node1 = topicGraph[t1]
                let node2 = topicGraph[t2]

                // Check if they're already linked
                let strength = (node1?.relatedTopics[t2] ?? 0) + (node2?.relatedTopics[t1] ?? 0)
                if strength > 0 {
                    crossRefs.append("[\(t1)↔\(t2) strength:\(String(format: "%.1f", strength))]")
                }

                // Find bridge topics
                let related1 = Set((node1?.relatedTopics.keys).map(Array.init) ?? [])
                let related2 = Set((node2?.relatedTopics.keys).map(Array.init) ?? [])
                let bridges = related1.intersection(related2)
                if let firstBridge = bridges.randomElement() {
                    crossRefs.append("[bridge: \(t1)→\(firstBridge)→\(t2)]")
                }
            }
        }

        let injection = crossRefs.isEmpty ? "" : crossRefs.joined(separator: " ")
        let reconstructed = crossRefs.isEmpty ? query : "\(query) \(injection)"

        return GateResult(
            reconstructedPrompt: reconstructed,
            gateType: .synthesize,
            contextInjection: injection,
            topicEvolution: nil,
            confidence: 0.85
        )
    }

    // ─── RECORD RESPONSE CONTEXT ─── Feed response back into topic graph
    func recordResponse(_ response: String, forTopics topics: [String]) {
        contextWindow.append((role: "assistant", content: String(response.prefix(2000)), timestamp: Date()))
        if contextWindow.count > maxContextWindow { contextWindow.removeFirst() }

        // Extract key phrases from response for future context injection
        let sentences = response.components(separatedBy: ". ").filter { $0.count > 20 }
        for topic in topics {
            guard topicGraph[topic] != nil else { continue }
            if let keySentence = sentences.first(where: { $0.lowercased().contains(topic) }) {
                topicGraph[topic]?.contextFragments.append(String(keySentence.prefix(1500)))
                if (topicGraph[topic]?.contextFragments.count ?? 0) > 50 {
                    topicGraph[topic]?.contextFragments.removeFirst()
                }
            }
        }
    }

    // Topic extraction (mirrors L104State method)
    private func extractTopics(_ query: String) -> [String] {
        let stopWords: Set<String> = ["the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "shall", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "about", "between", "after",
            "before", "above", "below", "and", "but", "or", "not", "no", "so", "if",
            "than", "too", "very", "just", "also", "then", "now", "here", "there",
            "when", "where", "why", "how", "what", "which", "who", "whom", "whose",
            "this", "that", "these", "those", "i", "me", "my", "we", "our", "you",
            "your", "he", "she", "it", "they", "them", "its", "his", "her", "their",
            "tell", "explain", "describe", "more", "like", "think", "know"]
        let words = query.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 && !stopWords.contains($0) }
        return Array(Set(words))
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🔀 CONTEXTUAL LOGIC GATE                                 ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Activations:        \(gateActivations)
        ║  Reconstructions:    \(promptReconstructions)
        ║  Topics Tracked:     \(topicGraph.count)
        ║  Context Window:     \(contextWindow.count)/\(maxContextWindow)
        ║  Top Topics:
        \(topicGraph.sorted { $0.value.mentions > $1.value.mentions }.prefix(5).map { "║    • \($0.key): \($0.value.mentions) mentions (stage \($0.value.evolutionStage))" }.joined(separator: "\n"))
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    struct GateResult {
        let reconstructedPrompt: String
        let gateType: GateType
        let contextInjection: String
        let topicEvolution: TopicNode?
        let confidence: Double
    }
}
