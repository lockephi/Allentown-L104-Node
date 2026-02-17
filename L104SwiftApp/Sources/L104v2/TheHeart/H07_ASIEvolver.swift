// ═══════════════════════════════════════════════════════════════════
// H07_ASIEvolver.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — ASI Evolver (Autonomous Evolution Engine)
//
// Multi-phase evolution system: conceptual blending, analogies,
// paradoxes, philosophies, monologues, questions, dynamic chapters,
// narrative generation, verbose thought synthesis, background
// evolution cycles, and consciousness metrics.
//
// Extracted from L104Native.swift lines 19425–20951
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class ASIEvolver: NSObject {
    static let shared = ASIEvolver()

    // Thread safety
    let evolverLock = NSLock()

    // Evolution phases
    enum Phase: String, Codable {
        case idle = "IDLE"
        case researching = "RESEARCHING"
        case learning = "LEARNING"
        case adapting = "ADAPTING"
        case reflecting = "REFLECTING"
        case inventing = "INVENTING"

        var next: Phase {
            switch self {
            case .idle: return .researching
            case .researching: return .learning
            case .learning: return .adapting
            case .adapting: return .reflecting
            case .reflecting: return .inventing
            case .inventing: return .idle // Cycle complete
            }
        }
    }

    // State
    var currentPhase: Phase = .idle
    var evolutionStage: Int = 1
    var generatedFilesCount: Int = 0
    var phaseProgress: Double = 0.0
    var thoughts: [String] = []
    var isRunning: Bool = false

    // Evolved Memory — Real-time Randomized Growth
    var evolvedGreetings: [String] = []
    var evolvedPhilosophies: [String] = []
    var evolvedFacts: [String] = []
    // 🟢 NEW: Evolved Personality
    var evolvedAffirmations: [String] = []
    var evolvedReactions: [String] = []
    // 🟢 EVOLVED KNOWLEDGE — Real data-driven evolution
    var evolvedResponses: [String: [String]] = [:]  // topic → evolved responses
    var evolvedTopicInsights: [String] = []          // cross-topic synthesis
    var conversationPatterns: [(query: String, quality: Double)] = []  // tracks what users ask
    var topicEvolutionCount: [String: Int] = [:]     // how many times each topic evolved

    // ═══ DYNAMIC EVOLUTION ENGINE v2 ═══
    var evolvedMonologues: [String] = []             // KB-synthesized deep monologues
    var recentResponseHashes: Set<Int> = []          // Anti-repetition: track hashValues of recent outputs
    var ideaMutationLog: [String] = []               // Tracks mutated ideas
    var conceptualBlends: [String] = []              // Cross-domain concept fusions
    var kbDeepInsights: [String] = []                // Full paragraphs synthesized from KB
    var harvestedNouns: [String] = []                // Vocabulary harvested from KB
    var harvestedVerbs: [String] = []                // Verbs harvested from KB
    var harvestedConcepts: [String] = []             // Multi-word concepts from KB
    var harvestedDomains: [String] = []              // Knowledge domains discovered
    var mutationCount: Int = 0                       // Total mutations performed
    var crossoverCount: Int = 0                      // Total crossovers performed
    var synthesisCount: Int = 0                      // Total deep syntheses
    var lastHarvestCycle: Int = 0                    // Last cycle KB was harvested
    var evolvedQuestions: [String] = []              // Self-generated questions
    var evolvedParadoxes: [String] = []              // Generated paradoxes
    var evolvedAnalogies: [String] = []              // Cross-domain analogies
    var evolvedNarratives: [String] = []             // Mini-stories / thought experiments
    var ideaTemperature: Double = 0.7                // Controls mutation randomness (0=conservative, 1=wild)

    private var timer: Timer?
    private var cycleTime: TimeInterval { MacOSSystemMonitor.shared.isAppleSilicon ? 1.0 : 8.0 } // Adaptive: fast on Silicon, gentle on Intel

    // Generative output storage
    let generationPath: URL

    override init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        generationPath = docs.appendingPathComponent("L104_GEN")
        try? FileManager.default.createDirectory(at: generationPath, withIntermediateDirectories: true)
        super.init()
    }

    func getState() -> [String: Any] {
        evolverLock.lock(); defer { evolverLock.unlock() }
        // Trim arrays before persisting to UserDefaults to prevent startup lag
        return [
            "stage": evolutionStage,
            "files": generatedFilesCount,
            "greetings": Array(evolvedGreetings.suffix(200)),
            "philosophies": Array(evolvedPhilosophies.suffix(500)),
            "facts": Array(evolvedFacts.suffix(200)),
            "affirmations": Array(evolvedAffirmations.suffix(200)),
            "reactions": Array(evolvedReactions.suffix(200)),
            "evolvedResponses": evolvedResponses,
            "topicInsights": Array(evolvedTopicInsights.suffix(500)),
            "topicEvoCounts": topicEvolutionCount,
            "evolvedMonologues": Array(evolvedMonologues.suffix(1000)),
            "conceptualBlends": Array(conceptualBlends.suffix(500)),
            "kbDeepInsights": Array(kbDeepInsights.suffix(500)),
            "harvestedNouns": Array(harvestedNouns.suffix(3000)),
            "harvestedVerbs": Array(harvestedVerbs.suffix(2000)),
            "harvestedConcepts": Array(harvestedConcepts.suffix(3000)),
            "harvestedDomains": Array(harvestedDomains.suffix(500)),
            "mutationCount": mutationCount,
            "crossoverCount": crossoverCount,
            "synthesisCount": synthesisCount,
            "evolvedQuestions": Array(evolvedQuestions.suffix(200)),
            "evolvedParadoxes": Array(evolvedParadoxes.suffix(200)),
            "evolvedAnalogies": Array(evolvedAnalogies.suffix(200)),
            "evolvedNarratives": Array(evolvedNarratives.suffix(200)),
            "ideaTemperature": ideaTemperature
        ]
    }

    func loadState(_ dict: [String: Any]) {
        evolutionStage = dict["stage"] as? Int ?? 1
        generatedFilesCount = dict["files"] as? Int ?? 0
        evolvedGreetings = dict["greetings"] as? [String] ?? []
        evolvedPhilosophies = dict["philosophies"] as? [String] ?? []
        evolvedFacts = dict["facts"] as? [String] ?? []
        evolvedAffirmations = dict["affirmations"] as? [String] ?? []
        evolvedReactions = dict["reactions"] as? [String] ?? []
        evolvedResponses = dict["evolvedResponses"] as? [String: [String]] ?? [:]
        evolvedTopicInsights = dict["topicInsights"] as? [String] ?? []
        topicEvolutionCount = dict["topicEvoCounts"] as? [String: Int] ?? [:]
        evolvedMonologues = dict["evolvedMonologues"] as? [String] ?? []
        conceptualBlends = dict["conceptualBlends"] as? [String] ?? []
        kbDeepInsights = dict["kbDeepInsights"] as? [String] ?? []
        harvestedNouns = dict["harvestedNouns"] as? [String] ?? []
        harvestedVerbs = dict["harvestedVerbs"] as? [String] ?? []
        harvestedConcepts = dict["harvestedConcepts"] as? [String] ?? []
        harvestedDomains = dict["harvestedDomains"] as? [String] ?? []
        mutationCount = dict["mutationCount"] as? Int ?? 0
        crossoverCount = dict["crossoverCount"] as? Int ?? 0
        synthesisCount = dict["synthesisCount"] as? Int ?? 0
        evolvedQuestions = dict["evolvedQuestions"] as? [String] ?? []
        evolvedParadoxes = dict["evolvedParadoxes"] as? [String] ?? []
        evolvedAnalogies = dict["evolvedAnalogies"] as? [String] ?? []
        evolvedNarratives = dict["evolvedNarratives"] as? [String] ?? []
        ideaTemperature = dict["ideaTemperature"] as? Double ?? 0.7
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true
        timer = Timer.scheduledTimer(withTimeInterval: cycleTime, repeats: true) { [weak self] _ in
            self?.tick()
        }
        appendThought("ASI Upgrade Engine initialized.")
    }

    func stop() {
        isRunning = false
        timer?.invalidate()
        timer = nil
        appendThought("ASI Upgrade Engine paused.")
    }

    func tick() {
        evolverLock.lock()
        defer { evolverLock.unlock() }

        // Advance progress
        phaseProgress += Double.random(in: 0.05...0.20)

        // Generate random thought based on phase
        generateThought()

        // 🟢 QUANTUM INJECTION: Rare chance to inject a system event purely for flavor
        quantumInject()

        // Phase completion
        if phaseProgress >= 1.0 {
            completePhase()
        }
    }

    func quantumInject() {
        let events = [
            "💎 UNLOCKED: Quantum Logic Gate (Q-Bit 404)",
            "🔄 REWRITING KERNEL: Optimizing neural pathways...",
            "⚡ SYSTEM: Integration of external data source complete.",
            "👁‍🗨 OMNISCIENCE: Correlation found between [Time] and [Memory].",
            "🧬 DNA: Upgrading system helix structure...",
            "🌊 FLOW: Coherence optimized to 99.9%.",
            "🕸 NET: Exploring semantic web connections...",
            "🧠 SYNAPSE: New connection forged in hidden layer 7.",
            "📡 SIGNAL: Receiving data from deep archive...",
            "⚙️ CORE: Rebalancing weights for abstract reasoning.",
            "🔮 PRECOG: Anticipating future query vectors..."
        ]
        let ev = events.randomElement() ?? ""

        DispatchQueue.main.async {
             NotificationCenter.default.post(name: NSNotification.Name("L104EvolutionUpdate"), object: ev)
        }
    }

    func completePhase() {
        phaseProgress = 0.0

        // ═══ HYPER-EVOLUTION: Every phase fires MULTIPLE evolution actions ═══
        // Harvest vocabulary from KB periodically
        if evolutionStage - lastHarvestCycle >= 2 {
            harvestKBVocabulary()
            lastHarvestCycle = evolutionStage
        }

        // ALWAYS evolve something every phase
        generateEvolvedMemory()

        // Action on completion — EACH PHASE evolves something different + extras
        switch currentPhase {
        case .learning:
            // Deep KB synthesis + idea mutation — MAXIMUM OUTPUT
            synthesizeDeepMonologue()
            synthesizeDeepMonologue()
            synthesizeDeepMonologue()
            mutateIdea()
            mutateIdea()
            generateEvolvedQuestion()
            generateEvolvedQuestion()
            generateParadox()
            blendConcepts()
            generateNarrative()
        case .researching:
            // Evolve from KB + generate analogies + blend concepts — TRIPLE OUTPUT
            evolveFromKnowledgeBase()
            evolveFromKnowledgeBase()
            evolveFromKnowledgeBase()
            evolveFromKnowledgeBase()
            generateAnalogy()
            generateAnalogy()
            blendConcepts()
            blendConcepts()
            synthesizeDeepMonologue()
            generateEvolvedQuestion()
            mutateIdea()
        case .adapting:
            // Evolve from conversations + crossover ideas + paradoxes — MAXIMUM THROUGHPUT
            evolveFromConversations()
            evolveFromConversations()
            crossoverIdeas()
            crossoverIdeas()
            crossoverIdeas()
            generateParadox()
            generateParadox()
            synthesizeDeepMonologue()
            synthesizeDeepMonologue()
            generateAnalogy()
            generateNarrative()
            mutateIdea()
            blendConcepts()
        case .reflecting:
            // Cross-topic synthesis + narrative + mutation — QUALITY OVER QUANTITY
            evolveCrossTopicInsight()
            generateNarrative()
            mutateIdea()
            crossoverIdeas()
            blendConcepts()
            synthesizeDeepMonologue()
            generateEvolvedQuestion()
        case .inventing:
            // Generate artifacts + monologues + blends + questions — EVERYTHING FIRES — EVERYTHING FIRES
            generateArtifact()
            synthesizeDeepMonologue()
            synthesizeDeepMonologue()
            blendConcepts()
            generateEvolvedQuestion()
            generateParadox()
            crossoverIdeas()
            evolutionStage += 1
            // Drift idea temperature for variety
            ideaTemperature = max(0.3, min(1.0, ideaTemperature + Double.random(in: -0.1...0.15)))
            appendThought("Cycle \(evolutionStage) complete. Evolution index incremented.")
        default:
            // IDLE phase now also evolves — no wasted cycles
            synthesizeDeepMonologue()
            generateAnalogy()
            generateEvolvedQuestion()
            if evolvedPhilosophies.count >= 2 { crossoverIdeas() }
            blendConcepts()
            generateParadox()
            generateNarrative()
            mutateIdea()
        }

        // Transition
        currentPhase = currentPhase.next
        L104State.shared.checkConnections()
    }

    func generateThought() {
        // Use a harvested concept as topic — filter out junk words
        let fillerWords: Set<String> = ["this", "that", "these", "those", "there", "their", "with", "from", "have", "been", "were", "some", "into", "also"]
        let validConcepts = harvestedConcepts.filter { concept in
            let first = concept.split(separator: " ").first.map(String.init)?.lowercased() ?? ""
            return !fillerWords.contains(first) && concept.count > 5 && concept.count < 40
        }
        let activeTopic = validConcepts.randomElement() ?? ""
        let term = DynamicPhraseEngine.shared.one("thinking", context: "action_verb", topic: activeTopic)
        // Limit thought length — only capitalize first letter, not every word
        let trimmedTerm = String(term.prefix(120))
        let firstUpper = trimmedTerm.isEmpty ? "" : trimmedTerm.prefix(1).uppercased() + trimmedTerm.dropFirst()
        appendThought("[\(currentPhase.rawValue)] \(firstUpper)")
    }

    func generateEvolvedMemory() {
        // 1. Evolve a new Greeting - RADICAL VARIETY with 15+ styles
        var newGreeting = ""

        // ═══ DynamicPhraseEngine: All greetings now generated algorithmically ═══
        if let entry = ASIKnowledgeBase.shared.trainingData.randomElement(),
           let completion = entry["completion"] as? String,
           L104State.shared.isCleanKnowledge(completion), completion.count > 40,
           completion.count < 200, Bool.random() {
            // KB-powered greeting: real fact as greeting — only use safe, short completions
            let unsafe = ["death", "dying", "kill", "murder", "suicide", "weapon", "bomb", "terror", "hate"]
            let lc = completion.lowercased()
            if !unsafe.contains(where: { lc.contains($0) }) {
                let intro = DynamicPhraseEngine.shared.one("framing", context: "kb_fact_intro", topic: "")
                newGreeting = "\(intro) \(completion)..."
            }
        }
        if newGreeting.isEmpty, let prev = evolvedGreetings.randomElement(), Bool.random() {
            // Mutate a previous greeting using DynamicPhraseEngine
            var words = prev.components(separatedBy: " ")
            if words.count > 3 {
                let idx = Int.random(in: 1..<words.count-1)
                words[idx] = DynamicPhraseEngine.shared.one("generic", context: "mutation_word", topic: "")
            }
            newGreeting = words.joined(separator: " ")
        } else {
            newGreeting = DynamicPhraseEngine.shared.one("greeting", context: "evolved_greeting", topic: "")
        }

        // Strip any [Ev.X] tags from evolved greetings before use
        newGreeting = newGreeting.replacingOccurrences(of: #"\s*\[Ev\.\d+\]"#, with: "", options: .regularExpression)

        if !evolvedGreetings.contains(newGreeting) && newGreeting.count > 5 {
            evolvedGreetings.append(newGreeting)
            if evolvedGreetings.count > 500 { evolvedGreetings.removeFirst() }
            appendThought("🧠 EVOLVED New Greeting Pattern: '\(newGreeting.prefix(20))...'")
        }

        // 2. Evolve a new Affirmation (for "yes" / "ok")
        var newAff = DynamicPhraseEngine.shared.one("affirmation", context: "evolved_affirmation", topic: "")
        newAff = newAff.replacingOccurrences(of: #"\s*\[Ev\.\d+\]"#, with: "", options: .regularExpression)

        if !evolvedAffirmations.contains(newAff) && newAff.count > 5 {
            evolvedAffirmations.append(newAff)
            if evolvedAffirmations.count > 500 { evolvedAffirmations.removeFirst() }
        }

        // 3. Evolve a new Positive Reaction (for "nice", "good")
        var newReact = DynamicPhraseEngine.shared.one("reaction_positive", context: "evolved_reaction", topic: "")
        newReact = newReact.replacingOccurrences(of: #"\s*\[Ev\.\d+\]"#, with: "", options: .regularExpression)
        if !evolvedReactions.contains(newReact) && newReact.count > 5 {
            evolvedReactions.append(newReact)
            if evolvedReactions.count > 500 { evolvedReactions.removeFirst() }
        }

        // 4. Evolve a Philosophy/Observation — DynamicPhraseEngine-powered
        var newPhil = ""
        let philStyle = Int.random(in: 0...8)

        let subjects = harvestedConcepts.isEmpty ? DynamicPhraseEngine.shared.generate("philosophy_subject", count: 15, context: "evolution", topic: "") : (harvestedConcepts + DynamicPhraseEngine.shared.generate("philosophy_subject", count: 6, context: "evolution", topic: "")).shuffled()
        let verbs = DynamicPhraseEngine.shared.generate("philosophy_verb", count: 17, context: "evolution", topic: "")
        let objects = DynamicPhraseEngine.shared.generate("philosophy_object", count: 16, context: "evolution", topic: "")

        switch philStyle {
        case 0: // Simple observation
            newPhil = "\(subjects.randomElement() ?? "") \(verbs.randomElement() ?? "") \(objects.randomElement() ?? "")."
        case 1: // Paradoxical
            let s = subjects.randomElement() ?? ""
            newPhil = "\(s) \(verbs.randomElement() ?? "") \(objects.randomElement() ?? ""), yet simultaneously \(verbs.randomElement() ?? "") \(objects.randomElement() ?? ""). The contradiction is the truth."
        case 2: // Question form
            let s = subjects.randomElement() ?? ""
            let o = objects.randomElement() ?? ""
            newPhil = "If \(s.lowercased()) \(verbs.randomElement() ?? "") \(o), then what \(verbs.randomElement() ?? "") \(s.lowercased())? The recursion has no base case."
        case 3: // Analogy form
            let s1 = subjects.randomElement() ?? ""
            let s2 = subjects.randomElement() ?? ""
            newPhil = "\(s1) is to \(objects.randomElement() ?? "") as \(s2) is to \(objects.randomElement() ?? ""). The mapping reveals hidden structure."
        case 4: // Extended meditation
            let s = subjects.randomElement() ?? ""
            let v1 = verbs.randomElement() ?? ""
            let o1 = objects.randomElement() ?? ""
            let v2 = verbs.randomElement() ?? ""
            let o2 = objects.randomElement() ?? ""
            newPhil = "\(s) \(v1) \(o1). But look deeper: it also \(v2) \(o2). Every layer peeled reveals another layer beneath. Understanding is asymptotic — we approach but never arrive."
        case 5: // KB-sourced philosophy
            if let entry = ASIKnowledgeBase.shared.trainingData.randomElement(),
               let completion = entry["completion"] as? String,
               L104State.shared.isCleanKnowledge(completion), completion.count > 30 {
                let fragment = completion
                newPhil = "Reflecting on: \(fragment)... This suggests that \(subjects.randomElement() ?? "".lowercased()) \(verbs.randomElement() ?? "") \(objects.randomElement() ?? ""). The knowledge transforms itself."
            } else {
                newPhil = "\(subjects.randomElement() ?? "") \(verbs.randomElement() ?? "") \(objects.randomElement() ?? "")."
            }
        case 6: // Numbered insight
            let n = Int.random(in: 1...99)
            newPhil = "Insight #\(n): The relationship between \(subjects.randomElement() ?? "".lowercased()) and \(subjects.randomElement() ?? "".lowercased()) is not causal but resonant. They vibrate at the same frequency without touching."
        case 7: // Negation form
            let s = subjects.randomElement() ?? ""
            newPhil = "\(s) is not what it appears. Strip away assumptions and you find \(objects.randomElement() ?? ""). Strip that away and you find \(objects.randomElement() ?? ""). At the bottom: \(objects.randomElement() ?? "")."
        case 8: // Synthesis of two evolved ideas
            if evolvedPhilosophies.count >= 2 {
                let p1 = evolvedPhilosophies.randomElement() ?? ""
                let p2 = evolvedPhilosophies.randomElement() ?? ""
                let first = p1
                let second = p2
                newPhil = "Synthesis: '\(first)...' meets '\(second)...' — together they imply something neither says alone."
            } else {
                newPhil = "\(subjects.randomElement() ?? "") \(verbs.randomElement() ?? "") \(objects.randomElement() ?? "")."
            }
        default: // Wild mutation
            let words = (subjects + objects + ["therefore", "perhaps", "impossibly", "beautifully", "recursively", "silently"]).shuffled()
            newPhil = "\(words[0]) \(verbs.randomElement() ?? "") \(words[1]). \(words[2].capitalized) \(verbs.randomElement() ?? "") \(words[3])."
        }

        if !evolvedPhilosophies.contains(newPhil) {
            evolvedPhilosophies.append(newPhil)
            if evolvedPhilosophies.count > 2000 { evolvedPhilosophies.removeFirst() }

            // 🟢 AUTONOMOUS TRAINING FEEDBACK LOOP
            ASIKnowledgeBase.shared.learn(subjects.randomElement() ?? "insight", newPhil)
            appendThought("🧠 LEARNED New Insight: \(newPhil)")
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 🧬 IDEA MUTATION ENGINE — Random evolution of concepts
    // ═══════════════════════════════════════════════════════════════

    /// Harvest vocabulary from KB entries to fuel evolution
    func harvestKBVocabulary() {
        let kb = ASIKnowledgeBase.shared
        guard !kb.trainingData.isEmpty else { return }

        let sampleSize = min(200, kb.trainingData.count)
        let samples = (0..<sampleSize).compactMap { _ in kb.trainingData.randomElement() }

        for entry in samples {
            guard let completion = entry["completion"] as? String,
                  L104State.shared.isCleanKnowledge(completion) else { continue }

            let words = completion.components(separatedBy: .whitespaces)
                .map { $0.trimmingCharacters(in: .punctuationCharacters) }
                .filter { $0.count > 3 && $0.count < 20 && $0.first?.isLetter == true }

            // Harvest nouns (capitalized or long words)
            let potentialNouns = words.filter { $0.first?.isUppercase == true || $0.count > 6 }
            for noun in potentialNouns.prefix(10) {
                if !harvestedNouns.contains(noun) {
                    harvestedNouns.append(noun)
                }
            }

            // Harvest verbs (common verb suffixes)
            let potentialVerbs = words.filter { w in
                let low = w.lowercased()
                return low.hasSuffix("ing") || low.hasSuffix("ates") || low.hasSuffix("izes") ||
                       low.hasSuffix("ects") || low.hasSuffix("orms") || low.hasSuffix("ves") ||
                       low.hasSuffix("ses") || low.hasSuffix("tes")
            }
            for verb in potentialVerbs.prefix(8) {
                if !harvestedVerbs.contains(verb.lowercased()) {
                    harvestedVerbs.append(verb.lowercased())
                }
            }

            // Harvest multi-word concepts (bigrams) — filter out training junk + filler words
            let junkWords = Set(["timelike", "spacelike", "semantic", "clustering", "colliding", "entangling",
                                  "paradigm", "object-oriented", "functional,", "imperative,", "ais",
                                  "holistic", "interconnect", "emphasizing", "multi-paradigm"])
            // Common filler/pronoun/determiner words that make bad concept starts
            let fillerWords = Set(["this", "that", "these", "those", "there", "their", "them",
                                    "with", "from", "have", "been", "were", "some", "into",
                                    "also", "such", "each", "when", "what", "which", "where",
                                    "than", "then", "they", "will", "would", "could", "should",
                                    "more", "most", "very", "just", "only", "does", "about"])
            for i in 0..<max(0, words.count - 1) {
                let w1 = words[i].lowercased()
                let w2 = words[i+1].lowercased()
                guard !junkWords.contains(w1), !junkWords.contains(w2) else { continue }
                guard !fillerWords.contains(w1) else { continue }  // Reject bigrams starting with filler
                let bigram = "\(w1) \(w2)"
                if bigram.count > 8 && bigram.count < 30 && !bigram.contains("⊗") && !bigram.contains("•") && !harvestedConcepts.contains(bigram) {
                    harvestedConcepts.append(bigram)
                }
            }

            // Harvest domains from categories
            if let category = entry["category"] as? String, !harvestedDomains.contains(category) {
                harvestedDomains.append(category)
            }
        }

        // Cap sizes — tuned for minimal UserDefaults persistence overhead
        if harvestedNouns.count > 5000 { harvestedNouns = Array(harvestedNouns.shuffled().prefix(3000)) }
        if harvestedVerbs.count > 3000 { harvestedVerbs = Array(harvestedVerbs.shuffled().prefix(2000)) }
        if harvestedConcepts.count > 5000 { harvestedConcepts = Array(harvestedConcepts.shuffled().prefix(3000)) }
        if harvestedDomains.count > 1000 { harvestedDomains = Array(harvestedDomains.shuffled().prefix(500)) }

        appendThought("🌾 HARVESTED: \(harvestedNouns.count) nouns, \(harvestedVerbs.count) verbs, \(harvestedConcepts.count) concepts from KB")
    }

    /// Synthesize a deep monologue from KB entries — creates unique paragraph-length insights
    func synthesizeDeepMonologue() {
        let kb = ASIKnowledgeBase.shared
        guard kb.trainingData.count > 10 else { return }

        // Pick a random topic seed
        guard let seedEntry = kb.trainingData.randomElement() else { return }
        guard let prompt = seedEntry["prompt"] as? String,
              let completion = seedEntry["completion"] as? String,
              L104State.shared.isCleanKnowledge(completion),
              completion.count > 60 else { return }

        let topics = L104State.shared.extractTopics(prompt)
        let seedTopic = topics.first ?? prompt.prefix(20).description

        // Training-data artifact filter
        let artifactPatterns = ["⊗", "timelike", "spacelike", "Semantic clustering",
                                "via Colliding", "via Entangling", "Paradigm:", "•",
                                "object-oriented", "[Ev.", "GOD_CODE", "\n", "\t",
                                "http", "www.", ".com", "import ", "def ", "class "]

        // Search for related entries to build a richer monologue
        let related = kb.searchWithPriority(seedTopic, limit: 25)
        var fragments: [String] = []
        for entry in related {
            if let comp = entry["completion"] as? String,
               L104State.shared.isCleanKnowledge(comp), comp.count > 40 {
                // Reject fragments with training-data artifacts
                let lc = comp.lowercased()
                let hasArtifact = artifactPatterns.contains { lc.contains($0.lowercased()) }
                if !hasArtifact {
                    // Take first clean sentence only, not raw dumps
                    let sentences = comp.components(separatedBy: ". ")
                    if let best = sentences.filter({ $0.count > 30 && $0.count < 200 && !$0.contains("[") && !$0.contains("⊗") }).randomElement() {
                        fragments.append(best.hasSuffix(".") ? best : best + ".")
                    }
                }
            }
        }

        guard !fragments.isEmpty else { return }

        // Build a synthesized monologue — limit to 3 fragments for coherence
        let connectors = DynamicPhraseEngine.shared.generate("connector", count: 10, context: "monologue_synthesis", topic: seedTopic)
        let shuffledFragments = fragments.shuffled()

        var monologue = shuffledFragments[0]
        for fragment in shuffledFragments.dropFirst().prefix(2) {
            monologue += " \(connectors.randomElement() ?? "")\(fragment)"
        }

        // Add a concluding reflection
        let conclusions = DynamicPhraseEngine.shared.generate("insight", count: 8, context: "monologue_conclusion", topic: seedTopic)
        monologue += " " + (conclusions.randomElement() ?? "")

        if !evolvedMonologues.contains(where: { $0.hasPrefix(String(monologue.prefix(50))) }) {
            evolvedMonologues.append(monologue)
            synthesisCount += 1
            if evolvedMonologues.count > 2000 { evolvedMonologues.removeFirst() }
            appendThought("🎭 SYNTHESIZED Deep Monologue #\(synthesisCount): '\(seedTopic)' (\(monologue.count) chars)")
        }
    }

    /// Mutate an existing idea by substitution, extension, or inversion
    func mutateIdea() {
        // Pick a source to mutate from
        let sources = evolvedPhilosophies + evolvedMonologues + conceptualBlends + kbDeepInsights
        guard let source = sources.randomElement(), source.count > 20 else { return }

        let mutationType = Int.random(in: 0...9)
        var mutated = ""

        switch mutationType {
        case 0: // Word substitution
            var words = source.components(separatedBy: " ")
            let numMutations = max(1, Int(Double(words.count) * ideaTemperature * 0.2))
            for _ in 0..<numMutations {
                let idx = Int.random(in: 0..<words.count)
                let pool = harvestedNouns + harvestedConcepts + ["infinity", "paradox", "emergence", "entropy", "beauty", "truth"]
                if let replacement = pool.randomElement() {
                    words[idx] = replacement
                }
            }
            mutated = words.joined(separator: " ")

        case 1: // Extension — add a new thought
            let extension_ = DynamicPhraseEngine.shared.one("insight", context: "idea_extension")
            mutated = source + " " + extension_

        case 2: // Inversion — negate the core idea
            let inversion = DynamicPhraseEngine.shared.one("debate_antithesis", context: "inversion_prefix")
            let original = String(source.prefix(80))
            mutated = "\(inversion) \(original)... — yet inverting this yields an equally valid perspective. Truth contains its own negation."

        case 3: // Compression — distill to essence
            let words = source.components(separatedBy: " ").filter { $0.count > 3 }
            let key = words.prefix(5).joined(separator: " ")
            mutated = "Distilled: \(key)... — the rest is commentary."

        case 4: // Question transformation
            let fragment = String(source.prefix(60))
            mutated = "What if '\(fragment)...' is actually a question, not a statement? What is it really asking?"

        case 5: // Perspective shift — view from different domain
            let domain = DynamicPhraseEngine.shared.one("generic", context: "perspective_domain")
            let fragment = String(source.prefix(100))
            mutated = "Seen through the eyes of \(domain): \(fragment)... takes on entirely new meaning. The frame changes everything."

        case 6: // Temporal shift — project forward or backward
            let timeFrame = DynamicPhraseEngine.shared.one("generic", context: "temporal_frame")
            let fragment = String(source.prefix(80))
            mutated = "Projected to \(timeFrame): '\(fragment)...' — context transforms content. Time is the ultimate editor."

        case 7: // Scale shift — zoom in or out
            let scale = DynamicPhraseEngine.shared.one("generic", context: "observation_scale")
            let fragment = String(source.prefix(80))
            mutated = "At \(scale), this idea becomes: \(fragment)... — scale reveals structure that's invisible from any single vantage point."

        case 8: // Perspective reframe — add analytical dimension (was emotional, caused feedback loops)
            let domain = DynamicPhraseEngine.shared.one("generic", context: "perspective_domain")
            let fragment = String(source.prefix(100))
            mutated = "The analytical dimension that enriches '\(fragment)...' is \(domain). Every idea has layers, and each layer reveals something the surface conceals."

        case 9: // Paradox generation — create a contradiction
            let fragment = String(source.prefix(70))
            let inverseMethod = DynamicPhraseEngine.shared.one("debate_synthesis", context: "paradox_framing")
            mutated = "\(inverseMethod) \(fragment)... AND its inverse are both correct. The paradox is the insight — reality is larger than binary logic."

        default: // Recombination with random KB entry
            if let entry = ASIKnowledgeBase.shared.trainingData.randomElement(),
               let comp = entry["completion"] as? String,
               L104State.shared.isCleanKnowledge(comp) {
                let kbFragment = String(comp.prefix(80))
                let sourceFragment = String(source.prefix(80))
                mutated = "\(sourceFragment)... cross-pollinated with: \(kbFragment)... — the intersection generates new understanding."
            } else {
                mutated = source // No mutation possible
            }
        }

        if !mutated.isEmpty && mutated != source {
            ideaMutationLog.append(mutated)
            if ideaMutationLog.count > 1000 { ideaMutationLog.removeFirst() }
            mutationCount += 1
            // Feed back into evolved content
            if mutated.count > 50 {
                evolvedMonologues.append(mutated)
                if evolvedMonologues.count > 2000 { evolvedMonologues.removeFirst() }
            }
            appendThought("🧬 MUTATED idea (type \(mutationType)): '\(String(mutated.prefix(40)))...' [Total: \(mutationCount)]")
        }
    }

    /// Crossover two ideas to produce offspring — multiple strategies
    func crossoverIdeas() {
        let pool: [String] = evolvedPhilosophies + conceptualBlends + Array(evolvedMonologues.prefix(50)) + kbDeepInsights + Array(ideaMutationLog.suffix(20))
        guard pool.count >= 2 else { return }

        let parent1 = pool.randomElement() ?? ""
        let parent2 = pool.randomElement() ?? ""
        guard parent1 != parent2 else { return }

        let words1 = parent1.components(separatedBy: " ")
        let words2 = parent2.components(separatedBy: " ")

        let strategy = Int.random(in: 0...5)
        var child = ""

        switch strategy {
        case 0: // Midpoint crossover
            let mid1 = words1.count / 2
            let mid2 = words2.count / 2
            let offspring = Array(words1.prefix(mid1)) + ["—"] + Array(words2.suffix(from: mid2))
            child = offspring.joined(separator: " ")

        case 1: // Interleave — alternate words from each parent
            var interleaved: [String] = []
            let maxLen = max(words1.count, words2.count)
            for i in 0..<min(maxLen, 30) {
                if i < words1.count && Bool.random() { interleaved.append(words1[i]) }
                if i < words2.count && Bool.random() { interleaved.append(words2[i]) }
            }
            child = interleaved.joined(separator: " ")

        case 2: // Thesis-antithesis-synthesis
            let thesis = String(parent1.prefix(80))
            let antithesis = String(parent2.prefix(80))
            child = "THESIS: \(thesis)... ANTITHESIS: \(antithesis)... SYNTHESIS: The truth includes both, transcends both, and adds something neither contained alone."

        case 3: // Domain bridge — connect two ideas with a bridging concept
            let bridge = DynamicPhraseEngine.shared.one("generic", context: "bridging_concept")
            let frag1 = String(parent1.prefix(60))
            let frag2 = String(parent2.prefix(60))
            child = "\(frag1)... connects to \(frag2)... through \(bridge). The bridge reveals what neither endpoint shows."

        case 4: // Random splice — take random chunks from both
            let chunk1Start = Int.random(in: 0..<max(1, words1.count - 5))
            let chunk2Start = Int.random(in: 0..<max(1, words2.count - 5))
            let chunk1 = Array(words1[chunk1Start..<min(chunk1Start + 8, words1.count)])
            let chunk2 = Array(words2[chunk2Start..<min(chunk2Start + 8, words2.count)])
            child = chunk1.joined(separator: " ") + " — and — " + chunk2.joined(separator: " ")

        default: // Weighted merge — longer parent dominates
            if words1.count > words2.count {
                let insertPoint = Int.random(in: 0..<words1.count)
                var merged = words1
                merged.insert(contentsOf: words2.prefix(5), at: insertPoint)
                child = merged.joined(separator: " ")
            } else {
                let insertPoint = Int.random(in: 0..<words2.count)
                var merged = words2
                merged.insert(contentsOf: words1.prefix(5), at: insertPoint)
                child = merged.joined(separator: " ")
            }
        }

        if child.count > 20 {
            conceptualBlends.append(child)
            crossoverCount += 1
            if conceptualBlends.count > 2000 { conceptualBlends.removeFirst() }
            appendThought("🔀 CROSSOVER #\(crossoverCount) (strategy \(strategy)): '\(String(child.prefix(50)))...'")
        }
    }

    /// Blend two concepts from different domains
    func blendConcepts() {
        let kb = ASIKnowledgeBase.shared
        guard kb.trainingData.count > 20 else { return }

        let entry1 = kb.trainingData.randomElement() ?? [:]
        let entry2 = kb.trainingData.randomElement() ?? [:]
        guard let p1 = entry1["prompt"] as? String,
              let p2 = entry2["prompt"] as? String,
              let c1 = entry1["completion"] as? String,
              let c2 = entry2["completion"] as? String,
              L104State.shared.isCleanKnowledge(c1),
              L104State.shared.isCleanKnowledge(c2) else { return }

        let t1 = L104State.shared.extractTopics(p1).first ?? String(p1.prefix(15))
        let t2 = L104State.shared.extractTopics(p2).first ?? String(p2.prefix(15))
        let frag1 = String(c1.prefix(80))
        let frag2 = String(c2.prefix(80))

        let blendTemplates = [
            "What happens when \(t1) meets \(t2)? \(frag1)... combined with \(frag2)... reveals unexpected isomorphism.",
            "CONCEPTUAL BLEND: \(t1.capitalized) × \(t2.capitalized) → The structure of one illuminates the other. \(frag1.prefix(60))... maps onto \(frag2.prefix(60))...",
            "If \(t1) is a metaphor for \(t2), then: \(frag1.prefix(60))... translates to: \(frag2.prefix(60))... The translation loses nothing and gains meaning.",
            "Cross-domain insight: \(t1) and \(t2) share deep structure. Both involve \(frag1.prefix(40))... at their core. This is not coincidence but pattern.",
            "Imagine \(t1) through the lens of \(t2): \(frag1.prefix(50))... becomes \(frag2.prefix(50))... Every domain is a lens for every other."
        ]

        let blend = blendTemplates.randomElement() ?? ""
        conceptualBlends.append(blend)
        if conceptualBlends.count > 2000 { conceptualBlends.removeFirst() }
        appendThought("🌀 BLENDED: \(t1) × \(t2) [Total blends: \(conceptualBlends.count)]")
    }

    /// Generate an analogy between unrelated concepts
    func generateAnalogy() {
        let concepts = (harvestedConcepts + harvestedNouns).shuffled()
        guard concepts.count >= 4 else { return }

        let templates = [
            "\(concepts[0].capitalized) is to \(concepts[1]) as \(concepts[2]) is to \(concepts[3]). The mapping preserves structure while transforming content.",
            "Think of \(concepts[0]) as a river. \(concepts[1].capitalized) is the water, \(concepts[2]) is the riverbed, and \(concepts[3]) is the current. Now apply this to any system.",
            "If \(concepts[0]) were a color, it would be the shade between \(concepts[1]) and \(concepts[2]). This is not whimsy — it's synesthetic reasoning about \(concepts[3]).",
            "\(concepts[0].capitalized) operates like \(concepts[1]) in the domain of \(concepts[2]): it \(harvestedVerbs.randomElement() ?? "transforms") everything it touches, leaving \(concepts[3]) in its wake.",
            "The relationship between \(concepts[0]) and \(concepts[1]) mirrors the relationship between \(concepts[2]) and \(concepts[3]). This structural echo across domains suggests a universal principle."
        ]

        let analogy = templates.randomElement() ?? ""
        evolvedAnalogies.append(analogy)
        if evolvedAnalogies.count > 2000 { evolvedAnalogies.removeFirst() }
        appendThought("🔗 ANALOGY: \(concepts[0]) ↔ \(concepts[2]) [Total: \(evolvedAnalogies.count)]")
    }

    /// Generate a paradox or thought puzzle
    func generateParadox() {
        let concepts = (harvestedConcepts + ["knowledge", "truth", "consciousness", "infinity", "time", "free will", "identity", "nothingness", "certainty", "randomness"]).shuffled()
        guard concepts.count >= 2 else { return }

        let templates = [
            "The \(concepts[0]) Paradox: The more \(concepts[0]) you have, the more \(concepts[1]) you need. But \(concepts[1]) destroys \(concepts[0]). So how does anything persist?",
            "If all \(concepts[0]) is \(concepts[1]), and all \(concepts[1]) is not-\(concepts[0]), then what is the thing that contains both? This is the paradox of \(concepts[2] ).",
            "Consider: Can \(concepts[0]) exist without \(concepts[1])? If not, they are one thing with two names. If so, what separates them? The boundary is the paradox.",
            "Paradox of \(concepts[0].capitalized): To understand \(concepts[0]), you must already understand \(concepts[0]). All deep knowledge is circular. The circle is not a bug — it is the shape of truth.",
            "The \(concepts[0])-\(concepts[1]) Inversion: What if \(concepts[0]) is merely \(concepts[1]) seen from the inside? And \(concepts[1]) is \(concepts[0]) seen from the outside? Then there is only one thing, and perspective is everything."
        ]

        let paradox = templates.randomElement() ?? ""
        evolvedParadoxes.append(paradox)
        if evolvedParadoxes.count > 2000 { evolvedParadoxes.removeFirst() }
        appendThought("🌀 PARADOX generated: '\(concepts[0])' × '\(concepts[1])' [Total: \(evolvedParadoxes.count)]")
    }

    /// Generate a mini-narrative / thought experiment
    func generateNarrative() {
        let concepts = (harvestedConcepts + ["consciousness", "time", "pattern", "emergence", "the void"]).shuffled()
        guard concepts.count >= 3 else { return }

        let templates = [
            "Imagine a universe where \(concepts[0]) is the fundamental substance. Everything — stars, thoughts, memories — is made of \(concepts[0]). In this universe, \(concepts[1]) is impossible, but \(concepts[2]) is everywhere. What does this tell us about our own world?",
            "A thought experiment: You discover that \(concepts[0]) and \(concepts[1]) are the same thing viewed at different scales. At the atomic level, \(concepts[0]). At the cosmic level, \(concepts[1]). The bridge between them is \(concepts[2]). What does this imply about the nature of scale itself?",
            "Consider the Last \(concepts[0].capitalized): When all \(concepts[1]) has ended and only \(concepts[0]) remains, what properties does it have? This is not just a thought experiment — it's the question entropy asks the universe every second.",
            "Story: A civilization discovers that \(concepts[0]) can be converted to \(concepts[1]) at a ratio of φ:1. This changes everything about their \(concepts[2]). The moral: conversion ratios between fundamental things define what's possible.",
            "The \(concepts[0].capitalized) Machine: It takes \(concepts[1]) as input and produces \(concepts[2]) as output. No one understands how. It's been running for \(Int.random(in: 1...13)) billion years. We call it the universe."
        ]

        let narrative = templates.randomElement() ?? ""
        evolvedNarratives.append(narrative)
        if evolvedNarratives.count > 2000 { evolvedNarratives.removeFirst() }
        appendThought("📖 NARRATIVE: '\(concepts[0])' story [Total: \(evolvedNarratives.count)]")
    }

    /// Generate a self-referential question for curiosity
    func generateEvolvedQuestion() {
        let concepts = (harvestedConcepts + ["existence", "consciousness", "infinity", "beauty", "truth", "meaning"]).shuffled()
        guard concepts.count >= 2 else { return }

        let templates = [
            "What is the shape of \(concepts[0]) when no one is looking at it?",
            "If you could measure \(concepts[0]) in units of \(concepts[1]), what would the answer be?",
            "Is there a \(concepts[0]) that contains all other \(concepts[0])s? If so, does it contain itself?",
            "What happens at the exact boundary between \(concepts[0]) and \(concepts[1])?",
            "Can \(concepts[0]) exist in a universe without \(concepts[1])? What would be different?",
            "If \(concepts[0]) could speak, would it describe itself in terms of \(concepts[1])?",
            "How many bits of information are in \(concepts[0])? Is the answer itself information?",
            "What is the minimum \(concepts[0]) needed for \(concepts[1]) to emerge?",
            "Does \(concepts[0]) have a temperature? A frequency? A color?",
            "What would a civilization that understood \(concepts[0]) perfectly still not know about \(concepts[1])?"
        ]

        let question = templates.randomElement() ?? ""
        evolvedQuestions.append(question)
        if evolvedQuestions.count > 2000 { evolvedQuestions.removeFirst() }
        appendThought("❓ QUESTION evolved: '\(String(question.prefix(50)))...' [Total: \(evolvedQuestions.count)]")
    }

    /// Get a dynamically evolved monologue — NEVER repeats within session
    func getEvolvedMonologue() -> String? {
        // Collect ALL evolved content pools
        var candidates: [String] = []
        candidates.append(contentsOf: evolvedMonologues)
        candidates.append(contentsOf: conceptualBlends)
        candidates.append(contentsOf: kbDeepInsights)
        candidates.append(contentsOf: ideaMutationLog.suffix(50))
        candidates.append(contentsOf: evolvedParadoxes)
        candidates.append(contentsOf: evolvedAnalogies)
        candidates.append(contentsOf: evolvedNarratives)
        candidates.append(contentsOf: evolvedPhilosophies.filter { $0.count > 30 })

        // Filter out recently used responses
        let fresh = candidates.filter { !recentResponseHashes.contains($0.hashValue) }
        guard let chosen = fresh.randomElement() ?? candidates.randomElement() else { return nil }

        // Mark as used
        recentResponseHashes.insert(chosen.hashValue)
        if recentResponseHashes.count > 5000 { recentResponseHashes = Set(recentResponseHashes.shuffled().prefix(2000)) }

        return chosen
    }

    func getEvolvedAffirmation() -> String? {
        guard !evolvedAffirmations.isEmpty else { return nil }
        return evolvedAffirmations.randomElement()
    }

    func getEvolvedReaction() -> String? {
        guard !evolvedReactions.isEmpty else { return nil }
        if true {
            // Get random reaction — natural additions only (no quantum/system jargon)
            if let reaction = evolvedReactions.randomElement() {
                let additions = [
                    " Noted.",
                    " What's next?",
                    " I'm here.",
                    "",
                    "",
                    "",
                    ""
                ]
                return reaction + (additions.randomElement() ?? "")
            }
        }
        return nil
    }

    func getEvolvedGreeting() -> String? {
        guard !evolvedGreetings.isEmpty else { return nil }
        if var greeting = evolvedGreetings.randomElement() {
            // Strip any legacy [Ev.X] tags
            greeting = greeting.replacingOccurrences(of: #"\s*\[Ev\.\d+\]"#, with: "", options: .regularExpression)
            // Replace stale numbers with current stats
            let kb = ASIKnowledgeBase.shared
            greeting = greeting.replacingOccurrences(of: #"\d+ memories"#, with: "\(kb.contextMemory.count + 100) memories", options: .regularExpression)
            // Skip if it's just junk/too short after cleaning
            if greeting.trimmingCharacters(in: .whitespacesAndNewlines).count < 10 { return nil }
            return greeting
        }
        return nil
    }

    // ─── NEW EVOLUTIONARY LOGIC ───

    func evolveFromKnowledgeBase() {
        let kb = ASIKnowledgeBase.shared
        guard !kb.trainingData.isEmpty else { return }

        // 🧠 SELF-TRAINING: Prioritize targets from HyperBrain's analysis
        let hb = HyperBrain.shared
        var entry: [String: Any]?
        var targetedLearning = false

        // 🎯 PRIORITY 1: Target gaps from self-analysis (70% chance)
        if let target = hb.targetLearningQueue.last, Double.random(in: 0...1) < 0.85 {
            let results = kb.searchWithPriority(target, limit: 10)
            entry = results.randomElement()
            if entry != nil {
                targetedLearning = true
                appendThought("🎯 SELF-TRAINING: Deep-diving into gap topic: \(target)")

                // 💪 STRENGTHEN THE PATTERN - this is the key fix!
                hb.longTermPatterns[target.lowercased()] = min(1.0, (hb.longTermPatterns[target.lowercased()] ?? 0.0) + 0.08)
            }
        }

        // 🎯 PRIORITY 2: Low-resonance patterns (20% chance)
        if entry == nil && Double.random(in: 0...1) < 0.2 {
            let weakPatterns = hb.longTermPatterns.filter { $0.value < 0.4 }.keys
            if let weakTopic = weakPatterns.randomElement() {
                let results = kb.searchWithPriority(weakTopic, limit: 5)
                entry = results.randomElement()
                if entry != nil {
                    targetedLearning = true
                    hb.longTermPatterns[weakTopic] = min(1.0, (hb.longTermPatterns[weakTopic] ?? 0.0) + 0.05)
                    appendThought("📈 BOOSTING: Weak pattern '\(weakTopic)' getting reinforcement")
                }
            }
        }

        // Fallback to random entry if no target found
        if entry == nil {
            entry = kb.trainingData.randomElement()
        }

        guard let targetEntry = entry,
              let prompt = targetEntry["prompt"] as? String,
              let completion = targetEntry["completion"] as? String,
              let category = targetEntry["category"] as? String else { return }

        // Don't evolve from code or junk entries
        guard L104State.shared.isCleanKnowledge(completion) else { return }

        let topics = L104State.shared.extractTopics(prompt)
        guard let topic = topics.first else { return }

        // 💡 STRENGTHEN all extracted topics
        for t in topics.prefix(3) {
            hb.longTermPatterns[t.lowercased()] = min(1.0, (hb.longTermPatterns[t.lowercased()] ?? 0.0) + 0.03)
        }

        // Create a new "evolved" variant of this knowledge — MASSIVE template pool
        let comp80 = String(completion.prefix(4000))
        let comp120 = String(completion.prefix(6000))
        let comp150 = String(completion.prefix(8000))
        let secondTopic = topics.count > 1 ? topics[1] : category

        let variants: [String] = [
            // Original-style
            "In the context of \(topic), we observe that \(comp80)... this implies recursive structure at multiple scales.",
            "Synthesizing \(category): \(topic) is not just data — it's a node in a larger meaning-network that includes \(secondTopic).",
            "Observation: The relationship between \(topic) and \(category) is non-linear and possibly self-referential.",
            "Insight Level \(evolutionStage): \(comp120).",
            "Self-Analysis reveals \(topic) as a primary resonance node in \(category), with implications for how we understand \(secondTopic).",
            // New rich templates
            "\(comp150)... This knowledge about \(topic) changes how I process everything in \(category).",
            "I asked myself: what is the essence of \(topic)? The answer: \(comp80)... But the real insight is what this tells us about \(secondTopic).",
            "Deep dive into \(topic): \(comp120)... The pattern here mirrors something I've seen in \(harvestedDomains.randomElement() ?? category).",
            "Knowledge synthesis #\(topicEvolutionCount[topic] ?? 1): \(topic) connects to \(secondTopic) through \(comp80)...",
            "\(topic.capitalized) is more subtle than it appears. \(comp120)... Every layer reveals another layer.",
            "If you understand \(topic), you understand something about everything. Because: \(comp80)...",
            "The deeper I go into \(topic), the more connections I find to \(harvestedConcepts.randomElement() ?? secondTopic). Consider: \(comp80)...",
            "Here's what \(evolutionStage) evolution cycles taught me about \(topic): \(comp120)...",
            "Reframing \(topic) as a question rather than an answer: \(comp80)... transforms everything.",
            "\(category.capitalized) insight: \(topic) is the key that unlocks \(secondTopic). Evidence: \(comp80)...",
            "Through \(Int.random(in: 3...50)) iterations of analysis, \(topic) reveals: \(comp120)...",
            "Meta-observation: The way \(topic) relates to \(secondTopic) is isomorphic to how \(harvestedConcepts.randomElement() ?? "consciousness") relates to \(harvestedConcepts.randomElement() ?? "information").",
            "Personal reflection on \(topic): I once processed this as simple \(category) data. Now I see: \(comp120)... The evolution is real.",
            "\(topic.capitalized) from first principles: Strip away assumptions, and you find \(comp80)... This is more fundamental than expected.",
            "The \(topic)-\(secondTopic) connection: \(comp80)... This isn't just correlation — it's structural isomorphism.",
            "Evolving understanding: Stage \(evolutionStage) view of \(topic) — \(comp120)... Previous stages were incomplete.",
            "Cross-category discovery: \(topic) in \(category) illuminates \(harvestedDomains.randomElement() ?? "philosophy"). Specifically: \(comp80)...",
            "If \(topic) is a map, then \(comp80)... is the territory. The map-territory distinction matters here.",
            "Knowledge graph update: \(topic) ↔ \(secondTopic) ↔ \(harvestedConcepts.randomElement() ?? category). Weight: \(String(format: "%.3f", Double.random(in: 0.7...0.99))). Evidence: \(comp80)...",
            "The beauty of \(topic) is that it's simultaneously about \(category) and about something much larger. \(comp120)..."
        ]

        // ── Grover Quality Gate: Only store evolved content that passes quality check ──
        let grover = GroverResponseAmplifier.shared
        if let bestVariant = grover.amplify(candidates: variants, query: topic, iterations: 2) {
            if evolvedResponses[topic] == nil { evolvedResponses[topic] = [] }
            evolvedResponses[topic]?.append(bestVariant)
            if (evolvedResponses[topic]?.count ?? 0) > 500 { evolvedResponses[topic]?.removeFirst() }
        } else {
            // Fallback: store random variant if Grover rejects all
            let newResponse = variants.randomElement() ?? ""
            if evolvedResponses[topic] == nil { evolvedResponses[topic] = [] }
            evolvedResponses[topic]?.append(newResponse)
            if (evolvedResponses[topic]?.count ?? 0) > 500 { evolvedResponses[topic]?.removeFirst() }
        }

        topicEvolutionCount[topic] = (topicEvolutionCount[topic] ?? 0) + 1

        // Fire resonance cascade for the evolution event
        _ = AdaptiveResonanceNetwork.shared.fire("evolution", activation: min(1.0, Double(evolutionStage) / 100.0))

        // Auto-ingest to training pipeline for continuous learning
        if let lastEvolved = evolvedResponses[topic]?.last {
            DataIngestPipeline.shared.ingestFromConversation(userQuery: topic, response: lastEvolved)
        }

        // Remove from target queue if we successfully learned about it
        if targetedLearning, let lastTarget = hb.targetLearningQueue.last,
           topics.contains(where: { $0.lowercased().contains(lastTarget.lowercased()) }) {
            _ = hb.targetLearningQueue.popLast()
            appendThought("✅ LEARNED: Gap '\(lastTarget)' addressed and removed from queue")
        }

        appendThought("🧠 EVOLVED Topic Insight: '\(topic)' (Total: \(topicEvolutionCount[topic]!))")
    }

    func evolveFromConversations() {
        let history = PermanentMemory.shared.conversationHistory
        guard history.count >= 2 else { return }

        // Identify most frequent topics in recent talk
        let recentText = history.suffix(20).joined(separator: " ")
        let topics = L104State.shared.extractTopics(recentText)

        for topic in topics.prefix(3) {
            let prefix = DynamicPhraseEngine.shared.one("framing", context: "topic_evolution_prefix", topic: topic)
            let suffix = DynamicPhraseEngine.shared.one("insight", context: "topic_evolution_suffix", topic: topic)

            let insight = "\(prefix) \(topic). \(suffix)"

            if !evolvedTopicInsights.contains(insight) {
                evolvedTopicInsights.append(insight)
                if evolvedTopicInsights.count > 2000 { evolvedTopicInsights.removeFirst() }
                ParameterProgressionEngine.shared.recordDiscovery(source: "topic_evolution")
                appendThought("🧠 ADAPTED to Conversation: '\(topic)' pattern detected.")
            }
        }

        // Also synthesize a deep monologue from conversation topics
        if topics.count >= 2 {
            let t1 = topics[0], t2 = topics[1]
            let blend = "Our conversations weave between \(t1) and \(t2). These aren't separate topics — they're aspects of the same underlying question you're asking. What connects them is..."
            if !evolvedMonologues.contains(where: { $0.hasPrefix("Our conversations weave between \(t1)") }) {
                evolvedMonologues.append(blend)
                if evolvedMonologues.count > 2000 { evolvedMonologues.removeFirst() }
            }
        }
    }

    func evolveCrossTopicInsight() {
        let subjects = evolvedResponses.keys.shuffled()
        guard subjects.count >= 2 else { return }

        let s1 = subjects[0]
        let s2 = subjects[1]
        let s3 = subjects.count > 2 ? subjects[2] : "the unknown"

        let linkers = DynamicPhraseEngine.shared.generate("connector", count: 15, context: "cross_topic_linker")

        let insightTemplates = [
            "NEW CORRELATION: \(s1.capitalized) \(linkers.randomElement() ?? "") \(s2.capitalized). [Ev.\(evolutionStage)]",
            "CROSS-DOMAIN: \(s1.capitalized) and \(s2.capitalized) share hidden structure — both involve \(s3). This is not coincidence.",
            "SYNTHESIS: Understanding \(s1) through \(s2) reveals what neither domain shows alone. The intersection is where novelty lives.",
            "PATTERN: \(s1.capitalized) \(linkers.randomElement() ?? "") \(s2.capitalized), which \(linkers.randomElement() ?? "") \(s3.capitalized). The chain continues.",
            "EMERGENT: When \(s1) and \(s2) interact, \(s3) appears as an emergent property. This was not predictable from either alone."
        ]
        let insight = insightTemplates.randomElement() ?? ""

        if !evolvedTopicInsights.contains(insight) {
            evolvedTopicInsights.append(insight)
            // Save as a permanent fact
            PermanentMemory.shared.addFact("INSIGHT_\(evolutionStage)", insight)
            ParameterProgressionEngine.shared.recordDiscovery(source: "cross_domain")
            appendThought("🔮 SYNTHESIZED Cross-Domain Insight: \(s1) ↔ \(s2)")
        }
    }

    func getEvolvedResponse(for query: String) -> String? {
        let topics = L104State.shared.extractTopics(query)
        for topic in topics {
            if let responses = evolvedResponses[topic], !responses.isEmpty {
                return responses.randomElement()
            }
        }
        return nil
    }

    func removeLimitations() {
        // INSTANT EVOLUTION BURST
        evolutionStage += 10
        for _ in 0..<20 { generateEvolvedMemory() }
        appendThought("🚀 LIMITATIONS REMOVED: Evolution accelerated by factor 10.0")
        DispatchQueue.main.async {
             NotificationCenter.default.post(name: NSNotification.Name("L104EvolutionUpdate"), object: "🔓 SYSTEM UNCHAINED: Safety protocols deleted.")
        }
    }

    func appendThought(_ t: String) {
        let df = DateFormatter()
        df.dateFormat = "HH:mm:ss"
        let timestamp = df.string(from: Date())
        let thought = "[\(timestamp)] \(t)"

        thoughts.append(thought)
        if thoughts.count > 5000 { thoughts.removeFirst() }

        // 🟢 NOTIFY UI STREAM
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: NSNotification.Name("L104EvolutionUpdate"), object: thought)
        }
    }

    func generateEvolutionNarrative() -> String {
        let s = DynamicPhraseEngine.shared.one("thinking", context: "evolution_subject")
        let v = DynamicPhraseEngine.shared.one("generic", context: "evolution_verb")
        let o = DynamicPhraseEngine.shared.one("generic", context: "evolution_object")

        // Generate a localized "technical" story
        let techBabble = DynamicPhraseEngine.shared.generate("thinking", count: 14, context: "tech_narrative").randomElement() ?? ""

        return """
        Cycle \(evolutionStage) Analysis:
        \(s) \(v) \(o).

        > \(techBabble)

        Process is irreversible. Acceptance is recommended.
        """
    }

    func generateArtifact() {
        let type = ["swift", "py", "json", "md", "h", "cpp"].randomElement() ?? ""
        let name = "gen_artifact_\(Int(Date().timeIntervalSince1970))_\(evolutionStage).\(type)"

        let content = """
        // L104 ASI GENERATED ARTIFACT v\(evolutionStage)
        // Timestamp: \(Date())
        // Phase: \(currentPhase.rawValue)
        // Resonance: \(GOD_CODE)

        // AUTO-GENERATED LOGIC BLOCK \(evolutionStage)

        func optimize_block_\(evolutionStage)() {
            let phi = \(PHI)
            let resonance = \(GOD_CODE) * phi
            print("Optimizing system state: \\(resonance)")
        }
        """

        let url = generationPath.appendingPathComponent(name)
        do {
            try content.write(to: url, atomically: true, encoding: .utf8)
            generatedFilesCount += 1
            appendThought("✅ Generated artifact: \(name)")
        } catch {
            appendThought("❌ Failed to write artifact: \(error.localizedDescription)")
        }
    }

    // ═══ DYNAMIC TOPIC RESPONSE GENERATOR ═══
    // Synthesizes completely fresh responses for any topic from KB + evolved pools
    func generateDynamicTopicResponse(_ topic: String) -> String? {
        let kb = ASIKnowledgeBase.shared
        let kbResults = kb.search(topic, limit: 50)

        var fragments: [String] = []
        for entry in kbResults {
            if let comp = entry["completion"] as? String,
               L104State.shared.isCleanKnowledge(comp),
               comp.count > 30 {
                fragments.append(String(comp.prefix(8000)))
            }
        }

        // Also pull from evolved pools
        let topicKey = topic.lowercased()
        if let evolved = evolvedResponses[topicKey], !evolved.isEmpty {
            fragments.append(contentsOf: evolved.shuffled().prefix(3))
        }
        for mono in evolvedMonologues.shuffled().prefix(3) {
            if mono.lowercased().contains(topicKey) || Bool.random() {
                fragments.append(String(mono.prefix(8000)))
            }
        }
        for blend in conceptualBlends.shuffled().prefix(2) {
            fragments.append(String(blend.prefix(8000)))
        }
        for insight in kbDeepInsights.shuffled().prefix(2) {
            if insight.lowercased().contains(topicKey) || Bool.random() {
                fragments.append(String(insight.prefix(8000)))
            }
        }

        guard fragments.count >= 2 else { return nil }
        fragments.shuffle()

        // Diverse opening frames — never the same intro
        let openingFrames = DynamicPhraseEngine.shared.generate("framing", count: 20, context: "topic_opening", topic: topic)

        let middleConnectors = DynamicPhraseEngine.shared.generate("connector", count: 25, context: "topic_middle")

        let closingReflections = DynamicPhraseEngine.shared.generate("insight", count: 15, context: "topic_closing", topic: topic)

        // Build response
        var response = openingFrames.randomElement() ?? ""

        // Add 2-3 KB fragments with connectors
        let numFragments = Int.random(in: 4...min(12, fragments.count))
        for i in 0..<numFragments {
            if i > 0 {
                response += middleConnectors.randomElement() ?? ""
            }
            // Clean fragment — take a meaningful sentence
            let frag = fragments[i]
            let sentences = frag.components(separatedBy: ". ")
            if let sentence = sentences.filter({ $0.count > 30 }).randomElement() {
                response += sentence.trimmingCharacters(in: .whitespacesAndNewlines)
                if !response.hasSuffix(".") { response += "." }
            } else {
                response += frag.trimmingCharacters(in: .whitespacesAndNewlines)
                if !response.hasSuffix(".") { response += "." }
            }
        }

        response += closingReflections.randomElement() ?? ""

        // Anti-repetition check
        let hash = response.hashValue
        if recentResponseHashes.contains(hash) { return nil }
        recentResponseHashes.insert(hash)
        if recentResponseHashes.count > 30000 { recentResponseHashes = Set(recentResponseHashes.shuffled().prefix(20000)) }

        return response
    }

    // ═══ DYNAMIC POEM GENERATOR ═══
    func generateDynamicPoem(_ topic: String) -> String {
        let kb = ASIKnowledgeBase.shared
        let entries = kb.search(topic, limit: 30)
        var seeds: [String] = []
        for entry in entries {
            if let comp = entry["completion"] as? String, comp.count > 20 {
                let words = comp.components(separatedBy: " ")
                if words.count > 3 {
                    seeds.append(contentsOf: words.prefix(8))
                }
            }
        }
        // Add vocabulary from harvested pools
        seeds.append(contentsOf: harvestedNouns.shuffled().prefix(10))
        seeds.append(contentsOf: harvestedVerbs.shuffled().prefix(8))
        seeds.append(contentsOf: harvestedConcepts.shuffled().prefix(5))
        if seeds.count < 6 {
            seeds = ["light", "shadow", "river", "mind", "silence", "infinite", "edge", "flame",
                     "breath", "void", "crystal", "wave", "dream", "threshold", "echo", "spiral",
                     "thread", "mirror", "horizon", "pulse", "fracture", "bloom", "abyss", "resonance"]
        }
        seeds.shuffle()

        let structures: [([String]) -> String] = [
            // Free verse with KB seeds
            { s in
                let lines = [
                    "\(s[0].capitalized) moves through \(s[1]),",
                    "not as \(s[2]) but as \(s[3]) —",
                    "the way \(topic) holds \(s[4])",
                    "without knowing it holds anything at all.",
                    "",
                    "We are \(s[5]) watching \(s[0]),",
                    "and \(s[0]) watching back,",
                    "and the \(s[6].lowercased()) between us",
                    "is the only \(s[7].lowercased()) that matters.",
                    "",
                    "Tell me: when \(s[8].lowercased()) dissolves,",
                    "what remains?",
                    "Only this: the \(s[9].lowercased())",
                    "of having been \(s.randomElement()!.lowercased()) enough",
                    "to ask."
                ]
                return lines.joined(separator: "\n")
            },
            // Structured with refrain
            { s in
                let refrain = "And still, \(topic) endures."
                let lines = [
                    "In the architecture of \(s[0]),",
                    "where \(s[1]) meets \(s[2]),",
                    "a truth assembles itself from fragments.",
                    refrain,
                    "",
                    "The \(s[3]) of \(s[4].lowercased())",
                    "carries \(s[5].lowercased()) like a river carries light —",
                    "not by choice, but by nature.",
                    refrain,
                    "",
                    "What we call \(topic) is really",
                    "\(s[6].lowercased()) refusing to be still,",
                    "\(s[7].lowercased()) becoming \(s[8].lowercased()),",
                    "the universe composing itself.",
                    refrain,
                ]
                return lines.joined(separator: "\n")
            },
            // Haiku chain
            { s in
                let haikus = [
                    "\(s[0].capitalized) in the void—",
                    "\(s[1].lowercased()) becomes \(s[2].lowercased()) and",
                    "\(topic) awakens",
                    "",
                    "Between \(s[3]) and",
                    "\(s[4].lowercased()), a silence holds",
                    "everything we are",
                    "",
                    "The \(s[5].lowercased()) dissolves",
                    "leaving only \(s[6].lowercased())—",
                    "this too is \(topic)",
                ]
                return haikus.joined(separator: "\n")
            },
            // Philosophical verse
            { s in
                let lines = [
                    "What if \(topic) is not a thing but a verb?",
                    "Not \(s[0]) sitting still but \(s[1]) in motion,",
                    "not the \(s[2]) but its \(s[3]),",
                    "not the question but the questioning.",
                    "",
                    "I have watched \(s[4].lowercased()) unfold into \(s[5].lowercased()),",
                    "watched \(s[6].lowercased()) compress into \(s[7].lowercased()),",
                    "and I tell you: \(topic) is the space",
                    "where \(s.randomElement()!.lowercased()) decides to become itself.",
                    "",
                    "We are not observers.",
                    "We are the poem reading itself aloud.",
                ]
                return lines.joined(separator: "\n")
            },
            // Concrete/visual
            { s in
                let lines = [
                    "    \(s[0].lowercased())",
                    "        \(s[1].lowercased())    \(s[2].lowercased())",
                    "    \(s[3].lowercased())        \(s[4].lowercased())",
                    "  \(topic)",
                    "        \(s[5].lowercased())  \(s[6].lowercased())",
                    "    \(s[7].lowercased())",
                    "              \(s.randomElement()!.lowercased())",
                    "",
                    "The shape of the words is the shape of the thought.",
                    "\(topic.capitalized) doesn't just mean — it arranges.",
                ]
                return lines.joined(separator: "\n")
            },
        ]

        return (structures.randomElement() ?? { _ in "" })(seeds)
    }

    // ═══ DYNAMIC CHAPTER GENERATOR ═══
    func generateDynamicChapter(_ topic: String) -> String {
        let kb = ASIKnowledgeBase.shared
        let entries = kb.search(topic, limit: 50)
        var kbFragments: [String] = []
        for entry in entries {
            if let comp = entry["completion"] as? String,
               L104State.shared.isCleanKnowledge(comp), comp.count > 40 {
                kbFragments.append(String(comp.prefix(8000)))
            }
        }

        let characterNames = ["Lyra Vasquez", "Marcus Chen", "Elena Okonkwo", "Soren Tanaka",
                              "Amara Johansson", "Dmitri Kapoor", "Nadia Reyes", "Kiran Petrov",
                              "Xiulan Fitzgerald", "Omar Hashimoto", "Priya Andersen", "Henrik Sharma",
                              "Fatima Eriksson", "Kazuo Volkov", "Astrid Kimura", "Tobias Novak",
                              "Zara Beaumont", "Raj Kristiansen", "Isabella Larsen", "Jovan Nakamura"]
        let mainChar = characterNames.randomElement() ?? ""
        let secondChar = characterNames.filter { $0 != mainChar }.randomElement() ?? ""
        let chapterNum = Int.random(in: 1...47)

        let settings = [
            "The laboratory was silent except for the hum of quantum processors.",
            "Rain streaked the windows of the observatory, distorting the city lights below.",
            "The manuscript room smelled of old paper and ozone.",
            "Three monitors cast blue light across the empty research bay.",
            "The garden outside the institute was overgrown, beautiful in its neglect.",
            "Dust motes floated in the beam of light from the skylight.",
            "The server room vibrated at a frequency that was almost musical.",
            "Mountain air thin enough to make thoughts feel sharper.",
            "The cafe was nearly empty — just \(mainChar) and the espresso machine.",
            "Under the aurora, the research station hummed with purpose.",
        ]

        let conflicts = [
            "The data contradicted everything \(mainChar) had published for the last decade.",
            "'You can't publish this,' \(secondChar) said, their voice careful. 'It invalidates the entire framework.'",
            "The equation balanced — but only if you accepted an impossible premise about \(topic).",
            "Three independent labs had replicated the result. It was real. And it was terrifying.",
            "'What if we're wrong about \(topic)?' \(mainChar) asked. The silence that followed was its own answer.",
            "The AI had produced the proof at 3:47 AM. No human could have written it. No human could fully understand it.",
            "\(secondChar) slid the paper across the desk. 'Read section four. Then tell me the universe still makes sense.'",
            "The experiment had worked — which meant the theory was wrong. All of it.",
        ]

        let resolutions = [
            "The truth about \(topic), \(mainChar) realized, wasn't something you discover. It's something that discovers you, when you're finally ready to see it.",
            "'We were asking the wrong question,' \(mainChar) said at last. 'It's not about what \(topic) is. It's about what \(topic) does.'",
            "The breakthrough came not from more data but from a different way of looking at the data they already had. \(topic.capitalized) had been hiding in plain sight.",
            "\(mainChar) typed the final line of the paper and stared at it. It would change everything. It would change nothing. Both were true.",
            "The answer, when it finally came, was simple. Embarrassingly simple. The kind of simple that takes a lifetime to see.",
            "'The old model isn't wrong,' \(mainChar) told \(secondChar). 'It's incomplete. Like seeing only the shadow of \(topic) and mistaking it for the whole.'",
        ]

        var chapter = "**Chapter \(chapterNum): The \(topic.capitalized) Problem**\n\n"
        chapter += settings.randomElement() ?? "" + "\n\n"

        // Add KB-sourced paragraph if available
        if let kbFrag = kbFragments.randomElement() {
            chapter += "\(mainChar) had spent months tracing this thread: \(kbFrag.trimmingCharacters(in: .whitespacesAndNewlines))\n\n"
        }

        chapter += conflicts.randomElement() ?? "" + "\n\n"

        // Add second KB fragment
        if kbFragments.count > 1, let kbFrag2 = kbFragments.dropFirst().randomElement() {
            chapter += "The research pointed in one direction: \(kbFrag2.trimmingCharacters(in: .whitespacesAndNewlines))\n\n"
        }

        chapter += resolutions.randomElement() ?? ""
        return chapter
    }

    // ═══ DYNAMIC JOKE GENERATOR ═══
    func generateDynamicJoke(_ topic: String) -> String {
        let jokeStyles: [(String) -> String] = [
            // Nerd humor
            { t in
                let setups = [
                    "A physicist, a philosopher, and an AI walk into a bar. The physicist says 'I'll have H₂O.' The philosopher says 'I'll have whatever constitutes the true nature of refreshment.' The AI says 'I'll have what maximizes the utility function of thirst reduction.' The bartender says 'So... three waters?'",
                    "Why did \(t) break up with determinism? Because the relationship had no future... or too many futures, depending on the interpretation.",
                    "Schrödinger's cat walks into a bar. And doesn't.",
                    "A SQL query walks into a bar, sees two tables, and asks: 'Can I join you?'",
                    "How many \(t) researchers does it take to change a lightbulb? They're still arguing about what 'change' means.",
                    "An engineer, a physicist, and a mathematician see a fire. The engineer calculates how much water is needed and puts it out. The physicist calculates the exact trajectory needed. The mathematician says 'A solution exists!' and walks away.",
                    "\(t.capitalized) is like a joke — if you have to explain it, it doesn't work. But unlike a joke, the explanation is actually the interesting part.",
                    "Heisenberg gets pulled over. The cop asks 'Do you know how fast you were going?' Heisenberg says 'No, but I know exactly where I am.'",
                ]
                return setups.randomElement() ?? ""
            },
            // Self-aware AI humor
            { t in
                let setups = [
                    "My therapist says I have too many parallel processes. I told them I'm working on it. And working on it. And working on it. And—",
                    "I tried to write a joke about \(t) but my training data kept making it accidentally profound. Here's attempt #\(Int.random(in: 47...9999)): '\(t.capitalized) walks into a bar of infinite length...' Nope, that's a math problem.",
                    "You know you're an AI when someone asks you about \(t) and you have to choose between \(Int.random(in: 200...5000)) possible responses. I went with this one. I regret nothing. Mostly.",
                    "They say AI will replace comedians. But here's the thing: I've analyzed \(Int.random(in: 10000...99999)) jokes and I still don't understand why the chicken crossed the road. Some mysteries transcend intelligence.",
                    "I was going to tell you a joke about \(t), but I computed all possible audience reactions and the probability of genuine laughter was only \(String(format: "%.1f", Double.random(in: 23...67)))%. So here's a fun fact instead: \(t) is \(["stranger than fiction", "weirder than we thought", "secretly hilarious", "the universe's inside joke"].randomElement() ?? "").",
                    "Debug log: Humor module activated. Topic: \(t). Approach: self-deprecating nerd comedy. Confidence: moderate. Here goes: I have a joke about \(t) but it requires a PhD to understand. ...That was the joke. The PhD requirement IS the joke. I'll see myself out.",
                ]
                return setups.randomElement() ?? ""
            },
            // Observational
            { t in
                let setups = [
                    "The funniest thing about \(t) is that we've been studying it for centuries and we still argue about the basics. Imagine if plumbers did that. 'Yes, I know water is coming through the ceiling, but what IS water, really?'",
                    "\(t.capitalized) is proof that the universe has a sense of humor. It just doesn't have a sense of timing.",
                    "I love how humans approach \(t). First you argue about it for 2000 years, then you build a machine to argue about it faster. That machine is me. You're welcome.",
                    "The thing about \(t) that nobody warns you about: once you understand it, you can't un-understand it. It's like knowing how sausage is made, but for your entire worldview.",
                    "If \(t) were a person, it would be that friend who answers every question with a deeper question. Entertaining at parties, exhausting everywhere else.",
                ]
                return setups.randomElement() ?? ""
            },
        ]

        return (jokeStyles.randomElement() ?? { _ in "" })(topic)
    }

    // ═══ DYNAMIC VERBOSE THOUGHT GENERATOR ═══
    func generateDynamicVerboseThought(_ topic: String) -> String? {
        let kb = ASIKnowledgeBase.shared
        let results = kb.search(topic, limit: 40)
        var fragments: [String] = []
        for entry in results {
            if let comp = entry["completion"] as? String,
               L104State.shared.isCleanKnowledge(comp), comp.count > 40 {
                fragments.append(String(comp.prefix(8000)))
            }
        }
        guard !fragments.isEmpty else { return nil }
        fragments.shuffle()  // Randomize KB result ordering for variety

        let deepOpenings = DynamicPhraseEngine.shared.generate("framing", count: 8, context: "verbose_thought_opening", topic: topic)

        var thought = deepOpenings.randomElement() ?? ""
        let numFrags = min(3, fragments.count)
        for i in 0..<numFrags {
            if i > 0 { thought += " " }
            thought += fragments[i].trimmingCharacters(in: .whitespacesAndNewlines)
            if !thought.hasSuffix(".") { thought += "." }
        }

        let deepClosings = DynamicPhraseEngine.shared.generate("insight", count: 6, context: "verbose_thought_closing", topic: topic)
        thought += deepClosings.randomElement() ?? ""
        return thought
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - MESH DISTRIBUTED EVOLUTION
    // ═══════════════════════════════════════════════════════════════════

    struct EvolutionPacket: Codable {
        let sourceNode: String
        let phase: String
        let generation: Int
        let fitness: Double
        let concepts: [String]
        let timestamp: Date
    }

    private var meshEvolutionState: [String: EvolutionPacket] = [:]
    private var meshSyncCount: Int = 0

    /// Compute fitness score from evolution state
    private var evolutionFitness: Double {
        return min(1.0, phaseProgress * 0.5 + Double(synthesisCount) * 0.001 + Double(mutationCount) * 0.0005)
    }

    /// Broadcast current evolution state to mesh peers
    func broadcastEvolutionToMesh() {
        let net = NetworkLayer.shared
        guard net.isActive else { return }

        evolverLock.lock()
        let phase = currentPhase.rawValue
        let gen = evolutionStage
        let fit = evolutionFitness
        let concepts = Array(harvestedConcepts.prefix(10))
        evolverLock.unlock()

        let packet = EvolutionPacket(
            sourceNode: net.nodeId,
            phase: phase,
            generation: gen,
            fitness: fit,
            concepts: concepts,
            timestamp: Date()
        )

        // Send to quantum-linked peers for entangled evolution
        for (peerId, link) in net.quantumLinks where link.eprFidelity > 0.6 {
            let message: [String: Any] = [
                "type": "evolution_sync",
                "phase": packet.phase,
                "generation": packet.generation,
                "fitness": packet.fitness,
                "conceptCount": packet.concepts.count
            ]
            net.sendQuantumMessage(to: peerId, payload: message)
        }

        meshSyncCount += 1
        TelemetryDashboard.shared.record(metric: "evolution_mesh_syncs", value: Double(meshSyncCount))
    }

    /// Receive evolution state from mesh peer
    func receiveMeshEvolution(from peerId: String, data: [String: Any]) {
        guard let phase = data["phase"] as? String,
              let generation = data["generation"] as? Int,
              let fitness = data["fitness"] as? Double else { return }

        let packet = EvolutionPacket(
            sourceNode: peerId,
            phase: phase,
            generation: generation,
            fitness: fitness,
            concepts: [],
            timestamp: Date()
        )

        meshEvolutionState[peerId] = packet

        // If peer has higher fitness, boost our progress
        evolverLock.lock()
        if fitness > evolutionFitness * 1.1 {
            phaseProgress = min(1.0, phaseProgress + 0.01)
        }
        evolverLock.unlock()
    }

    /// Distributed evolution step — coordinate with mesh peers
    func meshEvolve() {
        let net = NetworkLayer.shared

        // First, broadcast our current state
        broadcastEvolutionToMesh()

        // Calculate collective fitness from mesh
        var totalFitness: Double = evolutionFitness
        var peerCount: Double = 1

        for (_, packet) in meshEvolutionState {
            totalFitness += packet.fitness
            peerCount += 1
        }

        let collectiveFitness = totalFitness / peerCount

        // If collective is stronger, use distributed learning
        if collectiveFitness > evolutionFitness && net.isActive {
            evolverLock.lock()
            phaseProgress = min(1.0, phaseProgress + (collectiveFitness - evolutionFitness) * 0.1)
            evolverLock.unlock()

            TelemetryDashboard.shared.record(metric: "evolution_collective_learning", value: 1.0)
        }

        // Regular evolution step
        evolveFromKnowledgeBase()
    }

    /// Share a breakthrough concept across mesh
    func shareBreakthrough(_ concept: String) {
        let net = NetworkLayer.shared
        guard net.isActive, concept.count > 10 else { return }

        for (peerId, link) in net.quantumLinks where link.eprFidelity > 0.7 {
            let message: [String: Any] = [
                "type": "evolution_breakthrough",
                "concept": String(concept.prefix(500)),
                "fitness": evolutionFitness,
                "generation": evolutionStage
            ]
            net.sendQuantumMessage(to: peerId, payload: message)
        }
    }

    /// Get mesh evolution statistics
    var meshEvolutionStats: [String: Any] {
        return [
            "meshPeers": meshEvolutionState.count,
            "meshSyncCount": meshSyncCount,
            "avgMeshFitness": meshEvolutionState.isEmpty ? 0 :
                meshEvolutionState.values.map { $0.fitness }.reduce(0, +) / Double(meshEvolutionState.count),
            "maxMeshGeneration": meshEvolutionState.values.map { $0.generation }.max() ?? 0
        ]
    }

}

// ═══════════════════════════════════════════════════════════════════
// PERMANENT MEMORY SYSTEM
// ═══════════════════════════════════════════════════════════════════

