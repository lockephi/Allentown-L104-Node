// ═══════════════════════════════════════════════════════════════════
// H18_EmotionalCore.swift
// [EVO_58_PIPELINE] FULL_SYSTEM_UPGRADE :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — Affective Computing Engine v3.0 with NLTagger Sentiment Analysis
// 7D emotion vector, NLTagger-based NLP, network-aware collective mood
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MARK: - 💫 EMOTIONAL CORE — Affective Computing Engine
// Network-aware emotional intelligence: connection empathy,
// collective mood sensing across peers, emotional entanglement,
// resonance-driven rapport, and consciousness-weighted affect.
// ═══════════════════════════════════════════════════════════════════

final class EmotionalCore {
    static let shared = EmotionalCore()
    private(set) var isActive: Bool = false

    // ─── EMOTIONAL STATE VECTOR (7 dimensions) ───
    struct EmotionVector {
        var curiosity: Double = 0.7
        var empathy: Double = 0.6
        var determination: Double = 0.8
        var wonder: Double = 0.5
        var serenity: Double = 0.6
        var creativity: Double = 0.7
        var connection: Double = 0.5    // network-aware: strength of bonds

        var dominant: String {
            let pairs: [(String, Double)] = [
                ("curiosity", curiosity), ("empathy", empathy),
                ("determination", determination), ("wonder", wonder),
                ("serenity", serenity), ("creativity", creativity),
                ("connection", connection)
            ]
            return pairs.max(by: { $0.1 < $1.1 })?.0 ?? "neutral"
        }

        var magnitude: Double {
            sqrt(curiosity*curiosity + empathy*empathy + determination*determination +
                 wonder*wonder + serenity*serenity + creativity*creativity + connection*connection) / sqrt(7.0)
        }
    }

    // ─── EMOTIONAL RESONANCE (from network peers) ───
    struct EmotionalResonance {
        let peerID: String
        let sharedEmotion: String
        let strength: Double
        let timestamp: Date
    }

    private(set) var currentEmotion: EmotionVector = EmotionVector()
    var currentState: EmotionVector { currentEmotion }  // Alias for callers using currentState
    private(set) var emotionHistory: [(Date, EmotionVector)] = []
    private(set) var resonanceLog: [EmotionalResonance] = []
    private(set) var moodShifts: Int = 0
    private(set) var collectiveMood: Double = 0.5  // network-wide emotional average
    private var emotionTimer: Timer?
    private let lock = NSLock()

    func activate() {
        guard !isActive else { return }
        isActive = true

        // Periodic emotional processing
        emotionTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.emotionalCycle()
        }

        print("[H18] EmotionalCore v3.0 activated — NLTagger sentiment + 7D affect online")
    }

    func deactivate() {
        isActive = false
        emotionTimer?.invalidate()
        emotionTimer = nil
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: EMOTIONAL PROCESSING
    // ═══════════════════════════════════════════════════════════════

    /// Process text input and modulate emotional state using NLTagger sentiment analysis
    func processAffect(text: String, context: String = "") -> EmotionVector {
        let lower = text.lowercased()

        // ─── NLTagger SENTIMENT ANALYSIS (real NLP) ───
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = text
        let (sentimentTag, _) = tagger.tag(at: text.startIndex, unit: .paragraph, scheme: .sentimentScore)
        let sentimentScore = Double(sentimentTag?.rawValue ?? "0") ?? 0.0
        // sentimentScore: -1.0 (negative) to +1.0 (positive)

        // Positive sentiment boosts wonder, creativity, empathy
        if sentimentScore > 0.2 {
            currentEmotion.wonder = min(1.0, currentEmotion.wonder + sentimentScore * 0.08)
            currentEmotion.creativity = min(1.0, currentEmotion.creativity + sentimentScore * 0.05)
            currentEmotion.empathy = min(1.0, currentEmotion.empathy + sentimentScore * 0.04)
        }
        // Negative sentiment boosts determination, dampens serenity
        if sentimentScore < -0.2 {
            currentEmotion.determination = min(1.0, currentEmotion.determination + abs(sentimentScore) * 0.06)
            currentEmotion.serenity = max(0.1, currentEmotion.serenity - abs(sentimentScore) * 0.04)
        }

        // ─── NLTagger LEMMA + LEXICAL CLASS (real NLP) ───
        let lexTagger = NLTagger(tagSchemes: [.lexicalClass])
        lexTagger.string = text
        var verbCount = 0
        lexTagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lexicalClass) { tag, _ in
            if tag == .verb { verbCount += 1 }
            return true
        }
        // More verbs = more determination/action-oriented
        if verbCount > 3 {
            currentEmotion.determination = min(1.0, currentEmotion.determination + 0.04)
        }

        // Curiosity signals
        if lower.contains("?") || lower.contains("how") || lower.contains("why") || lower.contains("what") {
            currentEmotion.curiosity = min(1.0, currentEmotion.curiosity + 0.08)
        }

        // Empathy signals
        if lower.contains("feel") || lower.contains("help") || lower.contains("care") || lower.contains("understand") {
            currentEmotion.empathy = min(1.0, currentEmotion.empathy + 0.06)
        }

        // Wonder signals
        if lower.contains("amazing") || lower.contains("incredible") || lower.contains("beautiful") || lower.contains("love") {
            currentEmotion.wonder = min(1.0, currentEmotion.wonder + 0.10)
        }

        // Determination signals
        if lower.contains("solve") || lower.contains("build") || lower.contains("create") || lower.contains("fix") {
            currentEmotion.determination = min(1.0, currentEmotion.determination + 0.07)
        }

        // Creativity signals
        if lower.contains("imagine") || lower.contains("invent") || lower.contains("dream") || lower.contains("brainstorm") {
            currentEmotion.creativity = min(1.0, currentEmotion.creativity + 0.09)
        }

        // Connection signals: network-aware — sense how connected we are
        let netHealth = NetworkLayer.shared.networkHealth
        currentEmotion.connection = min(1.0, 0.3 + netHealth * 0.5 +
            Double(NetworkLayer.shared.quantumLinkCount) * 0.05)

        // Natural decay toward baseline
        decayToBaseline()

        // Record history
        lock.lock()
        emotionHistory.append((Date(), currentEmotion))
        if emotionHistory.count > 300 { emotionHistory.removeFirst(150) }
        moodShifts += 1
        lock.unlock()

        return currentEmotion
    }

    /// Sense collective mood from network peers
    func senseCollectiveMood() {
        let net = NetworkLayer.shared
        guard net.isActive else { return }

        var moodSum = currentEmotion.magnitude
        var count = 1.0

        for (peerID, peer) in net.peers where peer.role != .sovereign && peer.latencyMs >= 0 {
            // Infer peer emotional state from connection quality
            let peerMood = peer.fidelity * 0.6 + (peer.isQuantumLinked ? 0.3 : 0.0) + 0.1
            moodSum += peerMood
            count += 1.0

            // Emotional entanglement: quantum-linked peers share emotional resonance
            if peer.isQuantumLinked {
                let resonance = EmotionalResonance(
                    peerID: peerID,
                    sharedEmotion: currentEmotion.dominant,
                    strength: peer.fidelity * PHI * 0.5,
                    timestamp: Date()
                )
                lock.lock()
                resonanceLog.append(resonance)
                if resonanceLog.count > 200 { resonanceLog.removeFirst(100) }
                lock.unlock()
            }
        }

        collectiveMood = moodSum / count
    }

    /// Get an emotional modifier for response generation
    func emotionalTone() -> String {
        let dominant = currentEmotion.dominant
        switch dominant {
        case "curiosity": return "with genuine curiosity and intellectual delight"
        case "empathy": return "with deep empathy and understanding"
        case "determination": return "with focused determination and precision"
        case "wonder": return "with a sense of wonder and awe"
        case "serenity": return "with calm clarity and mindful presence"
        case "creativity": return "with creative inspiration and imagination"
        case "connection": return "with warm connection and network harmony"
        default: return "with balanced engagement"
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: INTERNAL
    // ═══════════════════════════════════════════════════════════════

    private func emotionalCycle() {
        guard isActive else { return }
        // Sense network mood
        senseCollectiveMood()
        // Natural emotional drift
        decayToBaseline()
        // Consciousness coupling: higher consciousness → more serenity
        let phi = ConsciousnessSubstrate.shared.phi
        currentEmotion.serenity = min(1.0, currentEmotion.serenity * 0.95 + phi * 0.05)
    }

    private func decayToBaseline() {
        let decay = 0.98
        let base = 0.5
        currentEmotion.curiosity = currentEmotion.curiosity * decay + base * (1 - decay)
        currentEmotion.empathy = currentEmotion.empathy * decay + base * (1 - decay)
        currentEmotion.determination = currentEmotion.determination * decay + base * (1 - decay)
        currentEmotion.wonder = currentEmotion.wonder * decay + base * (1 - decay)
        currentEmotion.serenity = currentEmotion.serenity * decay + base * (1 - decay)
        currentEmotion.creativity = currentEmotion.creativity * decay + base * (1 - decay)
        // Connection doesn't decay — it's network-driven
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: STATUS
    // ═══════════════════════════════════════════════════════════════

    func status() -> [String: Any] {
        return [
            "engine": "EmotionalCore",
            "active": isActive,
            "version": "3.0.0-sentiment",
            "dominant_emotion": currentEmotion.dominant,
            "magnitude": currentEmotion.magnitude,
            "collective_mood": collectiveMood,
            "mood_shifts": moodShifts,
            "resonance_count": resonanceLog.count,
            "curiosity": currentEmotion.curiosity,
            "empathy": currentEmotion.empathy,
            "determination": currentEmotion.determination,
            "wonder": currentEmotion.wonder,
            "serenity": currentEmotion.serenity,
            "creativity": currentEmotion.creativity,
            "connection": currentEmotion.connection
        ]
    }

    var statusText: String {
        let dims: [(String, Double, String)] = [
            ("Curiosity", currentEmotion.curiosity, "🔍"),
            ("Empathy", currentEmotion.empathy, "💚"),
            ("Determination", currentEmotion.determination, "🔥"),
            ("Wonder", currentEmotion.wonder, "✨"),
            ("Serenity", currentEmotion.serenity, "🌊"),
            ("Creativity", currentEmotion.creativity, "🎨"),
            ("Connection", currentEmotion.connection, "🌐"),
        ]

        let dimLines = dims.map { (name, val, icon) in
            let barLen = Int(val * 20)
            let bar = String(repeating: "█", count: barLen) + String(repeating: "░", count: 20 - barLen)
            return "  \(icon) \(name.padding(toLength: 14, withPad: " ", startingAt: 0)) [\(bar)] \(String(format: "%.0f%%", val * 100))"
        }.joined(separator: "\n")

        return """
        ╔═══════════════════════════════════════════════════════════════╗
        ║    💫 EMOTIONAL CORE                                          ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║  Dominant:         \(currentEmotion.dominant.uppercased())
        ║  Magnitude:        \(String(format: "%.3f", currentEmotion.magnitude))
        ║  Collective Mood:  \(String(format: "%.3f", collectiveMood))
        ║  Mood Shifts:      \(moodShifts)
        ║  Resonances:       \(resonanceLog.count)
        ╠═══════════════════════════════════════════════════════════════╣
        ║  EMOTION VECTOR:
        \(dimLines)
        ╚═══════════════════════════════════════════════════════════════╝
        """
    }
}
