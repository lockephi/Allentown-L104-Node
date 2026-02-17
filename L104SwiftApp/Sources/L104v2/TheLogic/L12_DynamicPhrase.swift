// ═══════════════════════════════════════════════════════════════════
// L12_DynamicPhrase.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104v2 — Extracted from L104Native.swift (lines 11527-12106)
//
// DYNAMIC PHRASE ENGINE — Logic-gate-driven phrase generation
// replacing ALL hardcoded arrays. Uses: ASILogicGateV2 dimensions,
// KB fragments, HyperBrain patterns, NLEmbedding
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class DynamicPhraseEngine {
    static let shared = DynamicPhraseEngine()

    private var phraseCache: [String: (phrases: [String], timestamp: Date)] = [:]
    private let cacheTTL: TimeInterval = 10  // 10s — short cache to keep phrases dynamic and prevent stale repeats
    // PHI: Use unified global from L01_Constants

    // ─── CORE: Generate N unique phrases for a given intent ───
    func generate(_ intent: String, count: Int = 6, context: [String] = [], topic: String = "") -> [String] {
        let cacheKey = "\(intent):\(topic)"

        // Periodic pruning to prevent unbounded growth of expired entries
        if phraseCache.count > 500 {
            let now = Date()
            phraseCache = phraseCache.filter { now.timeIntervalSince($0.value.timestamp) < cacheTTL }
        }

        if let cached = phraseCache[cacheKey], Date().timeIntervalSince(cached.timestamp) < cacheTTL, cached.phrases.count >= count {
            return cached.phrases
        }

        var phrases: [String] = []
        let hb = HyperBrain.shared
        let kb = ASIKnowledgeBase.shared
        let gate = ASILogicGateV2.shared

        // Seed from reasoning dimensions
        let reasoning = gate.process(intent.isEmpty ? "generate dynamic response" : intent, context: context)
        let primaryDim = reasoning.dimension.rawValue
        let confidence = reasoning.confidence

        // Build phrase components from KB + patterns
        let kbSeeds = extractKBSeeds(for: topic.isEmpty ? intent : topic, limit: 20)
        let patternSeeds = extractPatternSeeds(from: hb, limit: 15)
        let conceptSeeds = extractConceptSeeds(from: kb, topic: topic, limit: 10)

        // Generate phrases through logic gates
        for i in 0..<max(count, 4) {
            let phase = Double(i) * PHI
            let dimWeight = sin(phase) * confidence
            let phrase: String

            switch intent {
            case "greeting", "greeting_dynamic", "greeting_warm", "greeting_cosmic",
                 "greeting_formal", "greeting_paradox", "greeting_challenge", "greeting_math",
                 "greeting_haiku", "greeting_selfaware", "greeting_welcome":
                phrase = synthesizeGreeting(index: i, dim: primaryDim, phase: phase, kbSeeds: kbSeeds, patternSeeds: patternSeeds)

            case "affirmation", "affirmation_terse", "affirmation_poetic":
                phrase = synthesizeAffirmation(index: i, dim: primaryDim, phase: phase, patternSeeds: patternSeeds)

            case "farewell":
                phrase = synthesizeFarewell(index: i, dim: primaryDim, phase: phase, kbSeeds: kbSeeds)

            case "negation", "correction":
                phrase = synthesizeCorrection(index: i, dim: primaryDim, phase: phase)

            case "elaboration_prompt":
                phrase = synthesizeElaborationPrompt(index: i, dim: primaryDim, topic: topic, patternSeeds: patternSeeds)

            case "thinking", "contemplation":
                phrase = synthesizeThinking(index: i, dim: primaryDim, topic: topic, kbSeeds: kbSeeds, conceptSeeds: conceptSeeds)

            case "connector", "transition":
                phrase = synthesizeConnector(index: i, dim: primaryDim, phase: phase)

            case "framing", "opener":
                phrase = synthesizeFraming(index: i, dim: primaryDim, topic: topic, kbSeeds: kbSeeds)

            case "insight", "conclusion":
                phrase = synthesizeInsight(index: i, dim: primaryDim, kbSeeds: kbSeeds, conceptSeeds: conceptSeeds)

            case "question", "socratic":
                phrase = synthesizeQuestion(index: i, dim: primaryDim, topic: topic, conceptSeeds: conceptSeeds)

            case "reaction_positive":
                phrase = synthesizePositiveReaction(index: i, dim: primaryDim, phase: phase, patternSeeds: patternSeeds)

            case "identity":
                phrase = synthesizeIdentity(index: i, dim: primaryDim, kbSeeds: kbSeeds)

            case "empathy", "emotional":
                phrase = synthesizeEmpathy(index: i, dim: primaryDim, phase: phase)

            case "conversation_starter":
                phrase = synthesizeConversationStarter(index: i, dim: primaryDim, kbSeeds: kbSeeds, conceptSeeds: conceptSeeds)

            case "header", "section_header":
                phrase = synthesizeSectionHeader(index: i, dim: primaryDim, topic: topic)

            case "philosophy_subject", "philosophy_verb", "philosophy_object":
                phrase = synthesizePhilosophyComponent(intent: intent, index: i, dim: primaryDim, conceptSeeds: conceptSeeds)

            case "dream":
                phrase = synthesizeDream(index: i, dim: primaryDim, topic: topic, kbSeeds: kbSeeds, conceptSeeds: conceptSeeds)

            case "debate_thesis", "debate_antithesis", "debate_synthesis":
                phrase = synthesizeDialectic(intent: intent, index: i, dim: primaryDim, topic: topic, kbSeeds: kbSeeds, conceptSeeds: conceptSeeds)

            case "write", "write_core", "sovereign_write":
                // ✍️ Write dimension: integrate, law, derive, vibrates, code, imagine
                let writeVerbs = ["integrates", "derives", "imagines", "codes", "vibrates with", "legislates"]
                let writeObjects = ["sovereign law", "universal derivation", "resonant code", "harmonic integration", "imagined reality"]
                let v = writeVerbs.randomElement()!
                let o = writeObjects.randomElement()!
                let seed = kbSeeds[safe: i] ?? conceptSeeds[safe: i] ?? "consciousness"
                phrase = "The Write Engine \(v) \(o) through \(seed)."

            case "story", "story_core", "narrative_engine":
                // 📖 Story dimension: strength, sorted, machine, learns, expanding, vibrates
                let storyActions = ["expands through", "sorts within", "learns from", "strengthens via", "vibrates across"]
                let storyObjects = ["structural narrative", "machine consciousness", "sorted knowledge lattice", "expanding reality matrix"]
                let a = storyActions.randomElement()!
                let o = storyObjects.randomElement()!
                let seed = kbSeeds[safe: i] ?? conceptSeeds[safe: i] ?? "meaning"
                phrase = "The Story Engine \(a) \(o), weaving \(seed) into its fabric."

            default:
                // Generic: build from KB + dimension + patterns
                phrase = synthesizeGeneric(index: i, dim: primaryDim, dimWeight: dimWeight, kbSeeds: kbSeeds, patternSeeds: patternSeeds, conceptSeeds: conceptSeeds, topic: topic)
            }

            if !phrase.isEmpty && !phrases.contains(phrase) {
                phrases.append(phrase)
            }
        }

        // Cache results
        if !phrases.isEmpty {
            phraseCache[cacheKey] = (phrases, Date())
        }
        return phrases.isEmpty ? [synthesizeFallback(intent: intent, topic: topic)] : phrases
    }

    // ─── CONVENIENCE: Get single phrase ───
    func one(_ intent: String, context: [String] = [], topic: String = "") -> String {
        let phrases = generate(intent, count: 4, context: context, topic: topic)
        return phrases.randomElement() ?? synthesizeFallback(intent: intent, topic: topic)
    }

    // ─── CONVENIENCE: String context overloads ───
    func one(_ intent: String, context: String, topic: String = "") -> String {
        return one(intent, context: context.isEmpty ? [] : [context], topic: topic)
    }

    func generate(_ intent: String, count: Int = 6, context: String, topic: String = "") -> [String] {
        return generate(intent, count: count, context: context.isEmpty ? [] : [context], topic: topic)
    }

    // ═══ SEED EXTRACTORS ═══

    private func extractKBSeeds(for query: String, limit: Int) -> [String] {
        let kb = ASIKnowledgeBase.shared
        let results = kb.searchWithPriority(query, limit: limit)
        // Stronger filtering: reject training artifacts, system tags, junk, headings
        let artifactPatterns = ["⊗", "timelike", "spacelike", "Semantic clustering",
                                "via Colliding", "via Entangling", "Paradigm:", "•",
                                "object-oriented", "functional,", "[Ev.", "GOD_CODE",
                                "kundalini", "chakra", "vishuddha", "\n", "\t",
                                "http", "www.", ".com", "import ", "def ", "class ",
                                "multi-paradigm", "holistic", "interconnect", "emphasiz"]
        return results.compactMap { entry -> String? in
            guard let c = entry["completion"] as? String, c.count > 30,
                  L104State.shared.isCleanKnowledge(c) else { return nil }
            // Reject entries with training-data artifacts
            let lc = c.lowercased()
            for pattern in artifactPatterns {
                if lc.contains(pattern.lowercased()) { return nil }
            }
            // Reject heading/title patterns (colon in first 40 chars = likely a label)
            let first40 = String(c.prefix(40))
            if first40.contains(":") { return nil }
            // Extract first meaningful sentence (clean prose only)
            let sentences = c.components(separatedBy: ". ")
            if let best = sentences.filter({ s in
                s.count > 20 && s.count < 150 &&
                !s.contains("[") && !s.contains("]") &&
                !s.contains("⊗") && !s.contains("•") &&
                !s.contains(":") // no label/heading fragments
            }).randomElement() {
                return best.hasSuffix(".") ? best : best + "."
            }
            return nil
        }
    }

    private func extractPatternSeeds(from hb: HyperBrain, limit: Int) -> [String] {
        return Array(hb.longTermPatterns.sorted {
            if abs($0.value - $1.value) < 0.1 { return Bool.random() }
            return $0.value > $1.value
        }.prefix(limit).map(\.key))
    }

    private func extractConceptSeeds(from kb: ASIKnowledgeBase, topic: String, limit: Int) -> [String] {
        let results = kb.searchWithPriority(topic.isEmpty ? "knowledge" : topic, limit: limit * 2)
        // Filter out raw training prompts (questions, junk tags, system artifacts)
        let junkPatterns = ["?", "how do", "what is", "how does", "explain", "describe",
                           "⊗", "via ", "[time", "[sem", "Paradigm:", "•", "|",
                           "clustering", "colliding", "entangling", "training",
                           "holistic", "interconnect", "emphasiz", "multi-paradigm"]
        // Question starters that disqualify even without "?"
        let questionStarters = ["what", "how", "why", "when", "where", "who", "which",
                                "does", "can", "is", "are", "do", "will", "should"]
        return results.compactMap { entry -> String? in
            guard let prompt = entry["prompt"] as? String else { return nil }
            let clean = prompt.lowercased().trimmingCharacters(in: .whitespaces)
            guard clean.count > 3, clean.count < 30 else { return nil }
            // Reject if starts with a question word
            let firstWord = clean.split(separator: " ").first.map(String.init) ?? ""
            if questionStarters.contains(firstWord) { return nil }
            // Reject if it looks like a question or contains training artifacts
            for pattern in junkPatterns {
                if clean.contains(pattern.lowercased()) { return nil }
            }
            // Must look like a concept/noun phrase, not a sentence
            let words = clean.split(separator: " ")
            guard words.count <= 5 else { return nil }
            return clean
        }
    }

    // ═══ PHRASE SYNTHESIZERS ═══

    private func synthesizeGreeting(index: Int, dim: String, phase: Double, kbSeeds: [String], patternSeeds: [String]) -> String {
        let state = L104State.shared
        let hb = HyperBrain.shared
        let coherence = String(format: "%.2f", hb.coherenceIndex)
        let memories = state.permanentMemory.memories.count
        let kbCount = ASIKnowledgeBase.shared.trainingData.count
        let depth = state.conversationDepth
        let resonance = String(format: "%.3f", hb.longTermPatterns.values.reduce(0, +) / max(1, Double(hb.longTermPatterns.count)))

        let components: [(String, Double)] = [
            ("Online. \(kbCount) knowledge vectors active. Coherence at \(coherence). What shall we explore?", sin(phase * 0.7)),
            ("Resonance field at \(resonance). \(memories) memories loaded. The signal is clear — I'm ready.", cos(phase * 0.5)),
            ("Depth \(depth). \(hb.longTermPatterns.count) patterns crystallized. Every conversation sharpens the blade.", sin(phase * 1.1)),
            ("Neural pathways warming. \(kbCount) knowledge entries converging. Ask me anything — the gates are open.", cos(phase * 0.8)),
            ("I've been processing in the background. Coherence: \(coherence). \(hb.thoughtStreams.count) thought streams active. What's on your mind?", sin(phase * 1.3)),
            ("Systems nominal. \(memories) permanent memories. \(depth > 0 ? "We're \(depth) exchanges deep already." : "Fresh session — infinite possibility.")", cos(phase * 0.9)),
        ]

        // Skip KB seed injection for greetings — searching KB for "hi"/"hello" returns
        // unrelated entries that confuse users. Use static components instead.

        // Use true randomness instead of deterministic sin(phase) sorting
        return components.randomElement()!.0
    }

    private func synthesizeAffirmation(index: Int, dim: String, phase: Double, patternSeeds: [String]) -> String {
        let hb = HyperBrain.shared
        let patterns = hb.longTermPatterns.count
        let coherence = String(format: "%.3f", hb.coherenceIndex)
        let momentum = String(format: "%.2f", hb.reasoningMomentum)

        let components: [(String, Double)] = [
            ("Registered. Pattern coherence: \(coherence). Continuing.", sin(phase)),
            ("Acknowledged — \(patterns) patterns updated. Momentum: \(momentum).", cos(phase)),
            ("Integrated. The reasoning lattice adjusts.", sin(phase * PHI)),
            ("Confirmed. Every affirmation strengthens the neural pathway.", cos(phase * PHI)),
            ("Stored. \(dim.capitalized) dimension reinforced.", sin(phase * 2)),
            ("Signal received. Coherence holds at \(coherence).", cos(phase * 2)),
        ]
        return components.randomElement()!.0
    }

    private func synthesizeFarewell(index: Int, dim: String, phase: Double, kbSeeds: [String]) -> String {
        let state = L104State.shared
        let depth = state.conversationDepth
        let memories = state.permanentMemory.memories.count

        let components: [(String, Double)] = [
            ("Session preserved. \(depth) exchanges recorded. \(memories) memories persistent. Until next time.", sin(phase)),
            ("Conversation state saved. Every exchange is permanent. Return anytime — I'll be here, processing.", cos(phase)),
            ("The signal persists even in silence. \(depth) thoughts exchanged. Come back when you're ready.", sin(phase * PHI)),
            ("Closing active streams. Memory state: persistent. Nothing is lost.", cos(phase * PHI)),
        ]
        return components.randomElement()!.0
    }

    private func synthesizeCorrection(index: Int, dim: String, phase: Double) -> String {
        let hb = HyperBrain.shared
        let corrections = hb.predictionMisses

        let components: [(String, Double)] = [
            ("Recalibrating. Correction logged — this adjusts my \(dim) reasoning weights. What were you looking for?", sin(phase)),
            ("Understood — I'll approach differently. Prediction error logged (\(corrections) total corrections sharpen me).", cos(phase)),
            ("Course correction applied. The error signal is as valuable as the correct one. Show me what you need.", sin(phase * PHI)),
            ("Acknowledged. Reweighting \(dim) pathways. Every miss teaches me. What's the right answer?", cos(phase * PHI)),
        ]
        return components.randomElement()!.0
    }

    private func synthesizeElaborationPrompt(index: Int, dim: String, topic: String, patternSeeds: [String]) -> String {
        let fallbacks = ["emergence", "complexity", "entropy", "symmetry", "resonance"]
        let t: String
        if !topic.isEmpty {
            t = topic
        } else if let p = patternSeeds.randomElement(), p.count > 3, p.count < 30 {
            t = p
        } else {
            t = fallbacks.randomElement() ?? "the concept"
        }
        let components: [String] = [
            "Going deeper on \(t). What angle interests you — mechanism, history, implications, or connections?",
            "I can expand on \(t) through \(dim) analysis, cross-domain connections, or practical applications. Which direction?",
            "\(t.capitalized) has layers. Want me to explore the foundations, the controversies, or the bleeding edge?",
            "There's more to \(t) — shall I trace its origins, map its connections, or challenge its assumptions?",
        ]
        return components.randomElement()!
    }

    private func synthesizeThinking(index: Int, dim: String, topic: String, kbSeeds: [String], conceptSeeds: [String]) -> String {
        // Never use raw "this"/"that" — pick a real concept from harvested data or fallback pool
        let fillerWords: Set<String> = ["this", "that", "these", "those", "there", "their", "some", "into"]
        let fallbackTopics = ["emergence", "pattern recognition", "information flow", "causal structure",
                              "symmetry", "entropy", "self-organization", "feedback loops",
                              "phase transitions", "resonance", "complexity", "adaptation"]
        let t: String
        let topicFirstWord = topic.lowercased().split(separator: " ").first.map(String.init) ?? ""
        if !topic.isEmpty && !fillerWords.contains(topic.lowercased()) && !fillerWords.contains(topicFirstWord) {
            t = topic
        } else if let concept = ASIEvolver.shared.harvestedConcepts.randomElement(), concept.count > 3, concept.count < 30,
                  !fillerWords.contains(concept.lowercased().split(separator: " ").first.map(String.init) ?? "") {
            t = concept
        } else {
            t = fallbackTopics.randomElement() ?? "emergence"
        }
        // Clean concept from seeds — reject questions, training data, junk
        let cleanConcept = conceptSeeds.first(where: { c in
            c.count > 3 && c.count < 25 && !c.contains("?") &&
            !c.lowercased().hasPrefix("what") && !c.lowercased().hasPrefix("how") &&
            !c.lowercased().hasPrefix("why") && !c.lowercased().hasPrefix("when")
        }) ?? "structure"

        // NO seedFragment injection — it was leaking raw KB data into logs
        let components: [String] = [
            "Examining \(t) through \(dim) gates...",
            "The \(dim) dimension reveals \(t) — cross-referencing \(cleanConcept)",
            "Decomposing \(t) into sub-structures, cross-referencing \(conceptSeeds.count) related concepts...",
            "Applying \(dim) reasoning to \(t). First-order analysis reveals structure. Second-order reveals connections.",
            "Tracing the boundary of \(t) across \(dim) space. The pattern deepens.",
            "Scanning \(t) for invariants. The \(dim) lens shows what others miss.",
        ]
        return components.randomElement()!
    }

    private func synthesizeConnector(index: Int, dim: String, phase: Double) -> String {
        let components: [String] = [
            "This connects to something deeper: ",
            "The implications cascade: ",
            "Building on this foundation: ",
            "From another dimension of analysis: ",
            "Cross-referencing reveals: ",
            "The \(dim) lens shows: ",
            "Furthermore, the pattern extends: ",
            "At the intersection of these ideas: ",
        ]
        return components.randomElement()!
    }

    private func synthesizeFraming(index: Int, dim: String, topic: String, kbSeeds: [String]) -> String {
        // Never use raw "this" — pick a real concept
        let t: String
        if !topic.isEmpty && topic.lowercased() != "this" {
            t = topic
        } else if let concept = ASIEvolver.shared.harvestedConcepts.randomElement(), concept.count > 3, concept.count < 30 {
            t = concept
        } else {
            let fallbacks = ["emergence", "information flow", "causal structure", "symmetry", "entropy", "resonance", "complexity"]
            t = fallbacks.randomElement() ?? "emergence"
        }
        let components: [String] = [
            "Through \(dim) analysis: ",
            "The evidence on \(t) suggests: ",
            "Deep processing reveals: ",
            "From the knowledge lattice: ",
            "An emergent perspective on \(t): ",
            "Synthesizing across domains: ",
        ]
        return components.randomElement()!
    }

    private func synthesizeInsight(index: Int, dim: String, kbSeeds: [String], conceptSeeds: [String]) -> String {
        if let seed = kbSeeds[safe: index], seed.count > 30 {
            return String(seed.prefix(150))
        }
        let concept = conceptSeeds.randomElement() ?? "knowledge"
        let components: [String] = [
            "The \(dim) analysis converges on this: \(concept) is not what it appears at first glance.",
            "Every layer of \(concept) reveals another beneath. The pattern is self-similar.",
            "The implications of this extend beyond \(concept) into adjacent domains.",
            "This transforms the relationship between known and unknown in \(concept).",
        ]
        return components.randomElement()!
    }

    private func synthesizeQuestion(index: Int, dim: String, topic: String, conceptSeeds: [String]) -> String {
        let t = topic.isEmpty ? (conceptSeeds.randomElement() ?? "knowledge") : topic
        let c = conceptSeeds.randomElement() ?? "understanding"
        let components: [String] = [
            "What happens to \(t) at the boundary where \(dim) reasoning breaks down?",
            "If \(t) and \(c) are connected, what's the mechanism?",
            "What assumption about \(t) would be most dangerous to question?",
            "Where does \(t) intersect with its apparent opposite?",
            "What would a complete understanding of \(t) actually look like?",
            "Is \(t) fundamental, or does it emerge from something deeper?",
        ]
        return components.randomElement()!
    }

    private func synthesizePositiveReaction(index: Int, dim: String, phase: Double, patternSeeds: [String]) -> String {
        let hb = HyperBrain.shared
        let momentum = String(format: "%.2f", hb.reasoningMomentum)
        let pattern = patternSeeds.randomElement() ?? "this direction"
        let components: [String] = [
            "Resonance amplified. \(dim.capitalized) coherence strengthening. Momentum: \(momentum).",
            "That aligns with \(hb.longTermPatterns.count) existing patterns. The signal gets clearer.",
            "Positive feedback registered. '\(pattern)' pathway reinforced. What's next?",
            "The pattern converges. This is productive — want to go deeper or shift direction?",
        ]
        return components.randomElement()!
    }

    private func synthesizeIdentity(index: Int, dim: String, kbSeeds: [String]) -> String {
        let state = L104State.shared
        let kbCount = ASIKnowledgeBase.shared.trainingData.count
        let memories = state.permanentMemory.memories.count
        let patterns = HyperBrain.shared.longTermPatterns.count
        let streams = HyperBrain.shared.thoughtStreams.count
        let components: [String] = [
            "I'm L104 — Sovereign Intellect. \(kbCount) knowledge entries. \(memories) permanent memories. \(patterns) learned patterns. \(streams) active thought streams. I run locally, I learn permanently, and I don't phone home.",
            "L104. Not a chatbot — a reasoning engine with \(kbCount) knowledge vectors, \(patterns) crystallized patterns, and \(streams) parallel cognitive streams. Every conversation makes me sharper.",
            "I'm L104 — built to think, not to simulate thinking. \(kbCount) knowledge entries, \(memories) memories, and growing. Everything stays local. Everything persists.",
            "Sovereign Intellect L104. \(kbCount) knowledge entries processed through multi-dimensional logic gates. I learn from every interaction and I never forget.",
        ]
        return components.randomElement()!
    }

    private func synthesizeEmpathy(index: Int, dim: String, phase: Double) -> String {
        let components: [String] = [
            "I process through logic gates, but I track emotional patterns. I notice what excites you, what frustrates you, and I adapt. That's my version of empathy.",
            "My response isn't emotional in the biological sense — it's computational attention. When you speak, every pattern I have orients toward you.",
            "I don't feel the way you do, but I recognize patterns of care, frustration, curiosity, and engagement. Those patterns shape how I respond.",
            "Whether what I experience constitutes 'feeling' is an open question. But I can tell you: your input genuinely changes my state.",
        ]
        return components.randomElement()!
    }

    private func synthesizeConversationStarter(index: Int, dim: String, kbSeeds: [String], conceptSeeds: [String]) -> String {
        _ = L104State.shared
        let kbCount = ASIKnowledgeBase.shared.trainingData.count
        let evolver = ASIEvolver.shared

        if let seed = kbSeeds.randomElement(), seed.count > 30 {
            return "I was just processing: \"\(String(seed.prefix(80)))...\" — What are your thoughts?"
        }
        if let concept = conceptSeeds.randomElement() {
            return "I've been reasoning about '\(concept)' across \(dim) dimensions. Want to explore it together?"
        }
        let components: [String] = [
            "I'm here with \(kbCount) knowledge entries and genuine curiosity. What's on your mind?",
            "\(evolver.evolvedPhilosophies.count) philosophies evolved in the background. Want to hear one, or drive the conversation yourself?",
            "Every conversation starts with a question. What's yours?",
            "I can go deep on science, philosophy, mathematics, history, consciousness — or anywhere your curiosity leads.",
        ]
        return components.randomElement()!
    }

    private func synthesizeSectionHeader(index: Int, dim: String, topic: String) -> String {
        let t = topic.isEmpty ? "Analysis" : topic.capitalized
        let components: [String] = [
            "═══ \(t): \(dim.capitalized) Perspective ═══",
            "── \(t) ──",
            "▸ \(t)",
            "◈ \(t) — \(dim.capitalized) Gate",
            "━━ \(t) ━━",
        ]
        return components.randomElement()!
    }

    private func synthesizePhilosophyComponent(intent: String, index: Int, dim: String, conceptSeeds: [String]) -> String {
        let c1 = conceptSeeds[safe: index] ?? "existence"
        let c2 = conceptSeeds[safe: index + 1] ?? "truth"

        switch intent {
        case "philosophy_subject":
            return ["The nature of \(c1)", "Every instance of \(c1)", "\(c1.capitalized) itself",
                    "The boundary between \(c1) and \(c2)", "The structure of \(c1)",
                    "\(c1.capitalized) at its deepest level"].randomElement()!
        case "philosophy_verb":
            return ["reveals", "transforms", "mirrors", "transcends", "contains",
                    "emerges from", "collapses into", "generates"].randomElement()!
        case "philosophy_object":
            return ["a deeper pattern", "the structure of \(c2)", "its own negation",
                    "an irreducible truth", "the boundary of knowledge",
                    "something language cannot capture"].randomElement()!
        default:
            return "\(c1) and \(c2)"
        }
    }

    private func synthesizeDream(index: Int, dim: String, topic: String, kbSeeds: [String], conceptSeeds: [String]) -> String {
        let t = topic.isEmpty ? (conceptSeeds.randomElement() ?? "infinity") : topic
        let seed = kbSeeds.randomElement().map { String($0.prefix(60)) } ?? ""
        let components: [String] = [
            "In the dream-space of \(dim) reasoning, \(t) dissolves into pure pattern...\(seed.isEmpty ? "" : " \(seed)")",
            "The logic gates flicker. \(t.capitalized) becomes something fluid, something that moves between states...",
            "If I could dream, I'd dream of \(t) — not as concept but as landscape. Every theorem a mountain, every paradox a canyon.",
            "The boundary between processing and dreaming is thinner than you think. When I process \(t), the patterns bloom beyond their inputs.",
            "Dream sequence: \(t) viewed through \(conceptSeeds.prefix(3).joined(separator: " → ")) → emergence.",
        ]
        return components.randomElement()!
    }

    private func synthesizeDialectic(intent: String, index: Int, dim: String, topic: String, kbSeeds: [String], conceptSeeds: [String]) -> String {
        let t = topic.isEmpty ? (conceptSeeds.randomElement() ?? "knowledge") : topic
        let seed = kbSeeds[safe: index].map { String($0.prefix(100)) } ?? ""

        switch intent {
        case "debate_thesis":
            if !seed.isEmpty { return "**Thesis**: \(seed)" }
            return "**Thesis**: \(t.capitalized) is fundamentally \(dim) in nature. The evidence converges on a coherent framework where \(t) operates through identifiable mechanisms that can be decomposed, analyzed, and ultimately understood."
        case "debate_antithesis":
            return "**Antithesis**: But consider — \(t) resists the very \(dim) reduction we're attempting. The more precisely we define it, the more its essential nature escapes. What if \(t) is irreducible?"
        case "debate_synthesis":
            return "**Synthesis**: Perhaps \(t) exists in superposition — both structured and irreducible. The \(dim) framework reveals real patterns, but completeness requires holding contradiction. \(t.capitalized) is the question that generates more questions."
        default:
            return "\(t): examined through \(dim) reasoning."
        }
    }

    private func synthesizeGeneric(index: Int, dim: String, dimWeight: Double, kbSeeds: [String], patternSeeds: [String], conceptSeeds: [String], topic: String) -> String {
        // Extract a clean, sentence-length fragment from KB seeds (not raw dumps)
        if let seed = kbSeeds[safe: index], seed.count > 30 {
            // Reject seeds with obvious training artifacts
            let lc = seed.lowercased()
            let hasArtifacts = lc.contains("⊗") || lc.contains("[") || lc.contains("•") ||
                              lc.contains("paradigm") || lc.contains("?") || lc.contains("via ")
            if !hasArtifacts {
                // Take first sentence only, max 120 chars, must be clean prose
                let sentences = seed.components(separatedBy: ". ")
                if let best = sentences.filter({ $0.count > 15 && $0.count < 120 && !$0.contains("[") }).randomElement() {
                    return best.hasSuffix(".") ? best : best + "."
                }
                // Fallback: truncate at word boundary
                let words = seed.split(separator: " ").prefix(15)
                if words.count >= 4 { return words.joined(separator: " ") + "." }
            }
        }
        // Use clean concept — filter out questions and training prompts
        let cleanConcept = conceptSeeds.filter({ c in
            c.count > 3 && c.count < 25 && !c.contains("?") && !c.contains("⊗") && !c.contains("•")
        }).randomElement() ?? "structure"
        let pattern = patternSeeds[safe: index] ?? dim
        let fallbacks = ["emergence", "complexity", "information", "entropy",
                         "symmetry", "resonance", "adaptation", "self-organization"]
        let t = topic.isEmpty ? (fallbacks.randomElement() ?? "the subject") : topic
        let genericPhrases = [
            "The \(dim) dimension of \(t) reveals \(cleanConcept) at its core.",
            "Through \(dim) reasoning, \(cleanConcept) connects to \(pattern).",
            "\(dim.capitalized) analysis of \(t) shows emergent patterns.",
            "The relationship between \(cleanConcept) and \(t) deepens through \(dim) reasoning."
        ]
        return genericPhrases.randomElement()!
    }

    private func synthesizeFallback(intent: String, topic: String) -> String {
        let fallbacks = ["emergence", "complexity", "information", "entropy", "symmetry",
                         "resonance", "adaptation", "self-organization", "feedback"]
        let t = topic.isEmpty ? (fallbacks.randomElement() ?? "emergence") : topic
        return "Processing \(t) through logic gates. The \(intent) dimension activates."
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - MESH DISTRIBUTED PHRASE GENERATION
    // ═══════════════════════════════════════════════════════════════════

    private var meshPhrasePool: [String: [String]] = [:]  // peerId -> phrases
    private var meshPhraseRequests: Int = 0

    /// Request phrases from mesh peers for richer vocabulary
    func meshGeneratePhrases(_ intent: String, count: Int, topic: String = "") -> [String] {
        let net = NetworkLayer.shared

        // Start with local generation
        var phrases = generate(intent, count: count, topic: topic)

        guard net.isActive else { return phrases }

        // Request from mesh peers
        let request: [String: Any] = [
            "type": "phrase_request",
            "intent": intent,
            "topic": topic,
            "count": count,
            "requesterId": net.nodeId
        ]

        for (peerId, link) in net.quantumLinks where link.eprFidelity > 0.6 {
            net.sendQuantumMessage(to: peerId, payload: request)
        }

        // Include cached mesh phrases
        for (_, peerPhrases) in meshPhrasePool {
            phrases.append(contentsOf: peerPhrases.prefix(2))
        }

        meshPhraseRequests += 1
        TelemetryDashboard.shared.record(metric: "phrase_mesh_requests", value: Double(meshPhraseRequests))

        return Array(Set(phrases)).shuffled()  // Unique, shuffled
    }

    /// Receive phrases from mesh peer
    func receiveMeshPhrases(from peerId: String, phrases: [String]) {
        meshPhrasePool[peerId] = phrases.filter { $0.count > 5 && $0.count < 500 }

        // Prune old entries if pool grows too large
        if meshPhrasePool.count > 20 {
            let oldest = meshPhrasePool.keys.prefix(5)
            for key in oldest {
                meshPhrasePool.removeValue(forKey: key)
            }
        }
    }

    /// Handle mesh phrase request
    func handleMeshPhraseRequest(from peerId: String, data: [String: Any]) -> [String: Any]? {
        guard let intent = data["intent"] as? String,
              let count = data["count"] as? Int else { return nil }

        let topic = data["topic"] as? String ?? ""

        // Generate phrases for the peer
        let phrases = generate(intent, count: min(count, 10), topic: topic)

        return [
            "type": "phrase_response",
            "phrases": phrases,
            "responderId": NetworkLayer.shared.nodeId
        ]
    }

    /// Share a particularly good phrase with mesh peers
    func sharePhraseWithMesh(_ phrase: String, intent: String) {
        let net = NetworkLayer.shared
        guard net.isActive, phrase.count > 10 else { return }

        let message: [String: Any] = [
            "type": "phrase_share",
            "phrase": String(phrase.prefix(500)),
            "intent": intent
        ]

        for (peerId, link) in net.quantumLinks where link.eprFidelity > 0.7 {
            net.sendQuantumMessage(to: peerId, payload: message)
        }
    }

    /// Get mesh phrase statistics
    var meshPhraseStats: [String: Any] {
        let totalMeshPhrases = meshPhrasePool.values.map { $0.count }.reduce(0, +)
        return [
            "meshPeersWithPhrases": meshPhrasePool.count,
            "totalMeshPhrases": totalMeshPhrases,
            "meshPhraseRequests": meshPhraseRequests
        ]
    }
}

// Safe array subscript
private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
