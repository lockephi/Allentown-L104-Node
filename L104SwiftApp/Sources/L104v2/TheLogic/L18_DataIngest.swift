// ═══════════════════════════════════════════════════════════════════
// L18_DataIngest.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104v2 — Extracted from L104Native.swift (lines 32520-32851)
//
// DATA INGEST PIPELINE — Runtime knowledge ingestion + training
// Ingest data → process → store → make searchable
// SELF-MODIFICATION ENGINE — Adaptive meta-learning system
// Tracks quality, adapts strategies, Grover-amplified
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class DataIngestPipeline {
    static let shared = DataIngestPipeline()

    private var ingestCount: Int = 0
    private var ingestHistory: [(source: String, entries: Int, timestamp: Date)] = []
    private var processingQueue: [(prompt: String, completion: String, category: String)] = []
    private let grover = GroverResponseAmplifier.shared

    // ═══ INGEST TEXT ═══ Process raw text into KB-ready entries
    func ingestText(_ text: String, source: String = "user", category: String = "ingested") -> IngestResult {
        let sentences = text.components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.count > 5 }

        guard !sentences.isEmpty else {
            return IngestResult(accepted: 0, rejected: 0, source: source, message: "No valid sentences found")
        }

        var accepted = 0
        var rejected = 0
        let kb = ASIKnowledgeBase.shared

        for sentence in sentences {
            // Quality gate
            guard L104State.shared.isCleanKnowledge(sentence) else { rejected += 1; continue }
            guard grover.scoreQuality(sentence, query: "") > 0.2 else { rejected += 1; continue }

            // Extract key topics as prompt
            let topics = L104State.shared.extractTopics(sentence)
            let prompt = topics.isEmpty ? String(sentence.prefix(60)) : topics.prefix(3).joined(separator: " ")

            // Add to KB + persist to disk
            let entry: [String: Any] = [
                "prompt": prompt,
                "completion": sentence,
                "category": category,
                "source": source,
                "ingested_at": Date().timeIntervalSince1970
            ]
            kb.trainingData.append(entry)
            kb.persistIngestedEntry(entry)
            accepted += 1
        }

        ingestCount += accepted
        ingestHistory.append((source: source, entries: accepted, timestamp: Date()))
        if ingestHistory.count > 200 { ingestHistory.removeFirst() }

        // Rebuild search index with new data
        if accepted > 0 {
            IntelligentSearchEngine.shared.buildIndex()
        }

        return IngestResult(
            accepted: accepted,
            rejected: rejected,
            source: source,
            message: "Ingested \(accepted) entries from \(source) (\(rejected) filtered)"
        )
    }

    // ═══ INGEST KEY-VALUE ═══ Direct knowledge injection
    func ingestFact(key: String, value: String, category: String = "fact") {
        let kb = ASIKnowledgeBase.shared
        let entry: [String: Any] = [
            "prompt": key,
            "completion": value,
            "category": category,
            "source": "direct_ingest",
            "ingested_at": Date().timeIntervalSince1970
        ]
        kb.trainingData.append(entry)
        kb.persistIngestedEntry(entry)
        ingestCount += 1

        // Also teach the evolver
        let evolver = ASIEvolver.shared
        if evolver.evolvedResponses[key.lowercased()] == nil {
            evolver.evolvedResponses[key.lowercased()] = []
        }
        evolver.evolvedResponses[key.lowercased()]?.append(value)
    }

    // ═══ INGEST FROM CONVERSATION ═══ Auto-learn from chat
    func ingestFromConversation(userQuery: String, response: String) {
        guard response.count > 30 else { return }
        guard L104State.shared.isCleanKnowledge(response) else { return }
        guard grover.scoreQuality(response, query: userQuery) > 0.4 else { return }

        let topics = L104State.shared.extractTopics(userQuery)
        let prompt = topics.isEmpty ? userQuery : topics.prefix(3).joined(separator: " ")

        let entry: [String: Any] = [
            "prompt": prompt,
            "completion": response,
            "category": "conversation_learned",
            "source": "auto_ingest",
            "ingested_at": Date().timeIntervalSince1970
        ]
        ASIKnowledgeBase.shared.trainingData.append(entry)
        ASIKnowledgeBase.shared.persistIngestedEntry(entry)
        ingestCount += 1
    }

    // ═══ STATUS ═══
    var status: String {
        return """
        📥 DATA INGEST PIPELINE
        ═══════════════════════════════════════
        Total Ingested:    \(ingestCount)
        Ingest Sessions:   \(ingestHistory.count)
        KB Total Entries:  \(ASIKnowledgeBase.shared.trainingData.count)
        Search Index:      \(IntelligentSearchEngine.shared.searchHistory.count > 0 ? "SYNCED" : "PENDING")
        ═══════════════════════════════════════
        """
    }

    struct IngestResult {
        let accepted: Int
        let rejected: Int
        let source: String
        let message: String
    }
}


// ═══════════════════════════════════════════════════════════════════
// SELF-MODIFICATION ENGINE — Adaptive meta-learning system
// Tracks quality, adapts strategies, Grover-amplified
// ═══════════════════════════════════════════════════════════════════

class SelfModificationEngine {
    static let shared = SelfModificationEngine()

    // ─── QUALITY TRACKING ───
    private var responseQualityHistory: [(score: Double, timestamp: Date)] = []
    private var strategyScores: [String: Double] = [:]  // strategy → avg quality
    private var activeStrategies: Set<String> = ["knowledge_synthesis", "evolved_response", "direct_answer"]
    private var modificationLog: [(action: String, reason: String, timestamp: Date)] = []
    private var qualityTrend: Double = 0.0  // positive = improving
    private(set) var modificationCount: Int = 0
    private var isRunning = false

    // ─── ADAPTATION PARAMETERS ───
    var responseTemperature: Double = 0.7
    var verbosityLevel: Double = 0.6  // 0=terse, 1=verbose
    var creativityBias: Double = 0.5
    var accuracyWeight: Double = 0.7
    var noveltyWeight: Double = 0.3

    // ─── STRATEGY REGISTRY ───
    private let allStrategies = [
        "knowledge_synthesis",   // Combine KB + evolved content
        "evolved_response",      // Use ASIEvolver output
        "direct_answer",         // Simple direct response
        "grover_amplified",      // Multi-candidate with Grover selection
        "cross_domain",          // Cross-reference multiple domains
        "conversational",        // Natural conversation style
        "analytical",            // Deep analysis mode
        "creative_synthesis"     // Novel connections
    ]

    // ═══ RECORD QUALITY ═══ Called after each response
    func recordQuality(query: String, response: String, strategy: String) {
        let score = GroverResponseAmplifier.shared.scoreQuality(response, query: query)

        responseQualityHistory.append((score: score, timestamp: Date()))
        if responseQualityHistory.count > 1000 { responseQualityHistory.removeFirst() }

        // Update strategy scores (exponential moving average)
        let prevScore = strategyScores[strategy] ?? 0.5
        strategyScores[strategy] = prevScore * 0.8 + score * 0.2

        // Compute quality trend
        if responseQualityHistory.count > 10 {
            let recent = responseQualityHistory.suffix(10).map(\.score)
            let older = responseQualityHistory.dropLast(10).suffix(10).map(\.score)
            let recentAvg = recent.reduce(0, +) / max(1, Double(recent.count))
            let olderAvg = older.isEmpty ? 0.5 : older.reduce(0, +) / max(1, Double(older.count))
            qualityTrend = recentAvg - olderAvg
        }

        // Trigger adaptation if quality is declining
        if qualityTrend < -0.1 && modificationCount % 5 == 0 {
            adaptStrategies()
        }

        modificationCount += 1
        ParameterProgressionEngine.shared.recordModification(source: "self_mod")
    }

    // ═══ ADAPT STRATEGIES ═══ Meta-learning: adjust based on quality trends
    private func adaptStrategies() {
        // Find worst-performing strategy
        if let worst = strategyScores.min(by: { $0.value < $1.value }),
           worst.value < 0.3 {
            activeStrategies.remove(worst.key)
            modificationLog.append((
                action: "DEACTIVATED \(worst.key)",
                reason: "Quality score \(String(format: "%.2f", worst.value)) below threshold",
                timestamp: Date()
            ))

            // Activate a replacement
            let inactive = Set(allStrategies).subtracting(activeStrategies)
            if let replacement = inactive.randomElement() {
                activeStrategies.insert(replacement)
                strategyScores[replacement] = 0.5
                modificationLog.append((
                    action: "ACTIVATED \(replacement)",
                    reason: "Replacing underperforming \(worst.key)",
                    timestamp: Date()
                ))
            }
        }

        // Adjust parameters based on trends
        if qualityTrend < -0.05 {
            // Quality declining — increase accuracy, decrease creativity
            accuracyWeight = min(0.9, accuracyWeight + 0.05)
            creativityBias = max(0.2, creativityBias - 0.05)
            responseTemperature = max(0.3, responseTemperature - 0.05)
        } else if qualityTrend > 0.05 {
            // Quality improving — can afford more creativity
            creativityBias = min(0.8, creativityBias + 0.02)
            responseTemperature = min(0.9, responseTemperature + 0.02)
        }

        if modificationLog.count > 200 { modificationLog.removeFirst() }
    }

    // ═══ SELECT STRATEGY ═══ Choose best strategy for a given query
    func selectStrategy(for query: String) -> String {
        let q = query.lowercased()

        // Heuristic routing
        if q.count < 15 { return "conversational" }
        if q.contains("why") || q.contains("how") || q.contains("explain") { return "analytical" }
        if q.contains("research") || q.contains("search") { return "knowledge_synthesis" }
        if q.contains("create") || q.contains("imagine") || q.contains("what if") { return "creative_synthesis" }

        // Use highest-scoring active strategy
        let ranked = activeStrategies
            .map { ($0, strategyScores[$0] ?? 0.5) }
            .sorted {
                if abs($0.1 - $1.1) < 0.01 { return Bool.random() }
                return $0.1 > $1.1
            }

        return ranked.first?.0 ?? "knowledge_synthesis"
    }

    // ═══ SELF-MODIFY ═══ Perform explicit self-modification cycle
    func selfModifyCycle() -> String {
        let avgQuality: Double
        if responseQualityHistory.isEmpty {
            avgQuality = 0.5
        } else {
            let recentScores: [Double] = responseQualityHistory.suffix(50).map(\.score)
            let scoreSum: Double = recentScores.reduce(0.0, +)
            avgQuality = scoreSum / Double(recentScores.count)
        }

        adaptStrategies()

        // Inform evolver of quality feedback
        let evolver = ASIEvolver.shared
        if avgQuality > 0.6 {
            evolver.ideaTemperature = min(1.0, evolver.ideaTemperature + 0.02)
        } else {
            evolver.ideaTemperature = max(0.3, evolver.ideaTemperature - 0.05)
        }

        // Inform HyperBrain
        HyperBrain.shared.workingMemory["self_mod_quality"] = avgQuality
        HyperBrain.shared.workingMemory["self_mod_trend"] = qualityTrend

        return """
        🔧 SELF-MODIFICATION CYCLE COMPLETE
        ═══════════════════════════════════════
        Avg Quality:       \(String(format: "%.4f", avgQuality))
        Quality Trend:     \(String(format: "%+.4f", qualityTrend)) \(qualityTrend > 0 ? "📈" : qualityTrend < -0.05 ? "📉" : "➡️")
        Active Strategies: \(activeStrategies.sorted().joined(separator: ", "))
        Temperature:       \(String(format: "%.2f", responseTemperature))
        Accuracy Weight:   \(String(format: "%.2f", accuracyWeight))
        Creativity Bias:   \(String(format: "%.2f", creativityBias))
        Modifications:     \(modificationCount)
        Adaptations:       \(modificationLog.count)
        ═══════════════════════════════════════
        """
    }

    // ═══ STATUS ═══
    var status: String {
        let avgQ: Double
        if responseQualityHistory.isEmpty {
            avgQ = 0.5
        } else {
            let recentScores: [Double] = responseQualityHistory.suffix(20).map(\.score)
            let scoreSum: Double = recentScores.reduce(0.0, +)
            avgQ = scoreSum / Double(recentScores.count)
        }
        let sortedStrats = strategyScores.sorted { (a: (key: String, value: Double), b: (key: String, value: Double)) -> Bool in a.value > b.value }
        let topStrategies: String = sortedStrats.prefix(5)
            .map { (entry: (key: String, value: Double)) -> String in "  \(entry.key): \(String(format: "%.3f", entry.value))" }.joined(separator: "\n")
        let recentModEntries = modificationLog.suffix(5)
        var recentModLines: [String] = []
        for mod in recentModEntries { recentModLines.append("  [\(mod.action)] \(mod.reason)") }
        let recentMods: String = recentModLines.joined(separator: "\n")

        return """
        🔧 SELF-MODIFICATION ENGINE
        ═══════════════════════════════════════
        Status:            \(isRunning ? "🟢 ACTIVE" : "🔴 STANDBY")
        Avg Quality:       \(String(format: "%.4f", avgQ))
        Quality Trend:     \(String(format: "%+.4f", qualityTrend))
        Temperature:       \(String(format: "%.2f", responseTemperature))
        Accuracy Weight:   \(String(format: "%.2f", accuracyWeight))
        Creativity Bias:   \(String(format: "%.2f", creativityBias))
        Modifications:     \(modificationCount)

        📊 STRATEGY SCORES:
        \(topStrategies.isEmpty ? "  (no data yet)" : topStrategies)

        📝 RECENT ADAPTATIONS:
        \(recentMods.isEmpty ? "  (none)" : recentMods)
        ═══════════════════════════════════════
        """
    }
}
