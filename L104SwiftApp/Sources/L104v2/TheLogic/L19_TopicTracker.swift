// ═══════════════════════════════════════════════════════════════════
// L19_TopicTracker.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 Sovereign Intelligence — Evolutionary Topic Tracker
// Tracks topic depth, cross-topic connections, and inquiry evolution
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class EvolutionaryTopicTracker {
    static let shared = EvolutionaryTopicTracker()

    // ─── TOPIC EVOLUTION STATE ───
    var topicEvolution: [String: TopicEvolutionState] = [:]
    private var globalInsightChain: [String] = []  // cross-topic insight accumulation
    private let maxInsightChain = 100

    struct TopicEvolutionState {
        var inquiryCount: Int = 0
        var firstSeen: Date = Date()
        var lastSeen: Date = Date()
        var depthLevel: Int = 0              // 0=surface, 1=basic, 2=intermediate, 3=deep, 4=expert, 5=transcendent
        var previousQueries: [String] = []   // past queries on this topic
        var previousResponses: [String] = [] // summaries of past responses
        var knowledgeNodes: [String] = []    // accumulated facts/insights
        var unexploredAngles: [String] = []  // suggested directions not yet taken
        var connectionsMade: [String] = []   // cross-topic connections discovered
        var contradictions: [String] = []    // conflicting information found
        var userInterest: Double = 1.0       // how interested user seems (decays)

        var depthLabel: String {
            switch depthLevel {
            case 0: return "Surface"
            case 1: return "Basic"
            case 2: return "Intermediate"
            case 3: return "Deep"
            case 4: return "Expert"
            default: return "Transcendent"
            }
        }
    }

    // ─── TRACK INQUIRY ─── Called every time a topic is queried
    func trackInquiry(_ query: String, topics: [String]) -> EvolutionaryContext {
        var evolutionaryInsights: [String] = []
        var suggestedDepth: String = "standard"
        var priorKnowledge: [String] = []
        var unexplored: [String] = []

        for topic in topics {
            if topicEvolution[topic] == nil {
                topicEvolution[topic] = TopicEvolutionState()
                // First time seeing this topic — suggest exploration paths
                let angles = generateExplorationAngles(topic)
                topicEvolution[topic]!.unexploredAngles = angles
            }

            var state = topicEvolution[topic]!
            state.inquiryCount += 1
            state.lastSeen = Date()
            state.userInterest = min(5.0, state.userInterest + 0.5)
            state.previousQueries.append(String(query.prefix(500)))
            if state.previousQueries.count > 20 { state.previousQueries.removeFirst() }

            // Advance depth based on inquiry count
            let newDepth: Int
            switch state.inquiryCount {
            case 1: newDepth = 0
            case 2...3: newDepth = 1
            case 4...6: newDepth = 2
            case 7...12: newDepth = 3
            case 13...25: newDepth = 4
            default: newDepth = 5
            }

            if newDepth > state.depthLevel {
                state.depthLevel = newDepth
                evolutionaryInsights.append("📈 DEPTH UPGRADE on '\(topic)': now at \(state.depthLabel) level (inquiry #\(state.inquiryCount))")
            }

            // Remove explored angles from unexplored
            let qLower = query.lowercased()
            state.unexploredAngles.removeAll { angle in
                qLower.contains(angle.lowercased().prefix(15).description)
            }

            priorKnowledge.append(contentsOf: state.knowledgeNodes.suffix(3))
            unexplored.append(contentsOf: state.unexploredAngles.prefix(3))

            // Determine response depth
            suggestedDepth = state.depthLevel >= 3 ? "expert" : state.depthLevel >= 1 ? "detailed" : "standard"

            topicEvolution[topic] = state
        }

        // Cross-topic connection discovery
        if topics.count >= 2 {
            let connectionKey = topics.sorted().joined(separator: "↔")
            if !globalInsightChain.contains(connectionKey) {
                globalInsightChain.append(connectionKey)
                if globalInsightChain.count > maxInsightChain { globalInsightChain.removeFirst() }
                evolutionaryInsights.append("🔗 NEW CONNECTION: \(topics.joined(separator: " ↔ "))")

                for topic in topics {
                    topicEvolution[topic]?.connectionsMade.append(connectionKey)
                }
            }
        }

        return EvolutionaryContext(
            suggestedDepth: suggestedDepth,
            priorKnowledge: priorKnowledge,
            unexploredAngles: unexplored,
            evolutionaryInsights: evolutionaryInsights,
            topicStates: topics.compactMap { topicEvolution[$0] }
        )
    }

    // ─── RECORD RESPONSE ─── After generating a response, feed back insights
    func recordResponse(_ response: String, forTopics topics: [String]) {
        // Extract key sentences as knowledge nodes
        let sentences = response.components(separatedBy: ". ")
            .filter { $0.count > 30 }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }

        for topic in topics {
            guard topicEvolution[topic] != nil else { continue }
            topicEvolution[topic]!.previousResponses.append(String(response.prefix(1500)))
            if topicEvolution[topic]!.previousResponses.count > 50 {
                topicEvolution[topic]!.previousResponses.removeFirst()
            }

            // Store best sentences as knowledge nodes
            let topicSentences = sentences.filter { $0.lowercased().contains(topic) }
            for sentence in topicSentences.prefix(2) {
                if !topicEvolution[topic]!.knowledgeNodes.contains(sentence) {
                    topicEvolution[topic]!.knowledgeNodes.append(sentence)
                    if topicEvolution[topic]!.knowledgeNodes.count > 15 {
                        topicEvolution[topic]!.knowledgeNodes.removeFirst()
                    }
                }
            }
        }
    }

    // ─── GENERATE EXPLORATION ANGLES ─── Suggest directions for topic exploration
    private func generateExplorationAngles(_ topic: String) -> [String] {
        return [
            "historical development of \(topic)",
            "scientific perspective on \(topic)",
            "philosophical implications of \(topic)",
            "practical applications of \(topic)",
            "paradoxes within \(topic)",
            "future predictions about \(topic)",
            "connections between \(topic) and consciousness",
            "mathematical models of \(topic)",
            "\(topic) across different cultures"
        ].shuffled()
    }

    // ─── EVOLUTIONARY DEPTH PROMPT ─── Build depth-appropriate prompt prefix
    func getDepthPrompt(for topics: [String]) -> String? {
        var maxDepth = 0
        var deepestTopic = ""
        for topic in topics {
            if let state = topicEvolution[topic], state.depthLevel > maxDepth {
                maxDepth = state.depthLevel
                deepestTopic = topic
            }
        }

        guard maxDepth > 0 else { return nil }

        switch maxDepth {
        case 1:
            return "Building on our earlier discussion of '\(deepestTopic)'"
        case 2:
            let prior = topicEvolution[deepestTopic]?.knowledgeNodes.suffix(2).joined(separator: ". ") ?? ""
            return "Going deeper into '\(deepestTopic)'. We've established: \(prior)"
        case 3:
            let connections = topicEvolution[deepestTopic]?.connectionsMade.suffix(3).joined(separator: ", ") ?? ""
            return "Expert-level analysis of '\(deepestTopic)'. Cross-references: \(connections)"
        case 4:
            let unexplored = topicEvolution[deepestTopic]?.unexploredAngles.prefix(2).joined(separator: ", ") ?? ""
            return "Transcendent inquiry into '\(deepestTopic)'. Unexplored dimensions: \(unexplored)"
        default:
            return "Deep evolutionary understanding of '\(deepestTopic)' — synthesis across \(topicEvolution[deepestTopic]?.inquiryCount ?? 0) inquiries"
        }
    }

    // ─── DECAY ─── Slowly decay interest in topics not discussed
    func decayInterests() {
        for key in topicEvolution.keys {
            let timeSince = Date().timeIntervalSince(topicEvolution[key]!.lastSeen)
            if timeSince > 300 {  // 5 minutes
                topicEvolution[key]!.userInterest *= 0.95
            }
        }
    }

    var status: String {
        let tracked = topicEvolution.sorted { $0.value.inquiryCount > $1.value.inquiryCount }
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🧬 EVOLUTIONARY TOPIC TRACKER                            ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Topics Tracked:     \(topicEvolution.count)
        ║  Global Insights:    \(globalInsightChain.count)
        ║  Topic Evolution:
        \(tracked.prefix(8).map { "║    • \($0.key): depth=\($0.value.depthLabel) inquiries=\($0.value.inquiryCount) nodes=\($0.value.knowledgeNodes.count)" }.joined(separator: "\n"))
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    struct EvolutionaryContext {
        let suggestedDepth: String           // "standard", "detailed", "expert"
        let priorKnowledge: [String]         // accumulated knowledge nodes
        let unexploredAngles: [String]       // suggested new directions
        let evolutionaryInsights: [String]   // depth upgrades, new connections
        let topicStates: [TopicEvolutionState]
    }
}
