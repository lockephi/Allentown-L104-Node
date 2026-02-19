// ═══════════════════════════════════════════════════════════════════
// H05_L104StateResponse.swift
// [EVO_56_APEX_WIRED] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — L104State Extension (Response Generation v24.0)
//
// getIntelligentResponseMeta, composeHistoryResponse, composeFromKB,
// autoTrackTopic, extractTopics, generateReasonedResponse,
// generateVerboseThought, analyzeUserIntent, buildContextualResponse,
// generateNCGResponse, generateNaturalResponse, getStatusText.
//
// Extracted from L104Native.swift lines 38471–40210
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

extension L104State {

    // ═══ STATIC PATTERN SETS — hoisted from hot-path functions for O(1) lookup ═══
    private static let stopWordsSet: Set<String> = [
        "the", "is", "are", "you", "do", "does", "have", "has", "can", "will", "would", "could",
        "should", "what", "how", "why", "when", "where", "who", "that", "this", "and", "for",
        "not", "with", "about", "please", "so", "but", "it", "its", "my", "your", "me", "just",
        "like", "from", "more", "some", "tell", "define", "explain", "mean", "think", "know",
        "really", "very", "much", "also", "of", "to", "in", "on", "at", "yeah", "probs", "bro",
        "huh", "hmm", "hmmm", "cool", "now", "nothing", "want", "summary", "give", "read",
        "write", "type", "one", "out", "run", "tests", "test", "lets", "let", "okay", "all",
        "been", "was", "were", "had", "did", "done", "get", "got", "make", "made"
    ]

    private static let priorityTopicSet: Set<String> = [
        "love", "consciousness", "quantum", "physics", "mathematics", "philosophy",
        "universe", "time", "space", "entropy", "evolution", "god", "soul", "mind",
        "reality", "existence", "infinity", "beauty", "music", "art", "poetry",
        "science", "technology", "history", "future", "death", "life", "meaning",
        "neural", "algorithm", "code", "programming", "intelligence", "ai"
    ]

    private static let mathPatternSet: Set<String> = [
        "solve", "integrate", "derivative", "matrix", "eigenvalue", "factorial",
        "calculate", "compute", "evaluate", "simplify", "∫", "∂", "Σ", "∏",
        "differential equation", "linear algebra", "prime factor", "gcd", "lcm",
        "standard deviation", "regression", "fourier", "laplace", "taylor series"
    ]

    private static let sciencePatternSet: Set<String> = [
        "planck", "boltzmann", "avogadro", "speed of light", "gravitational constant",
        "electron mass", "proton mass", "black body", "stefan-boltzmann",
        "schrodinger", "heisenberg", "lorentz", "maxwell", "thermodynamic",
        "half-life", "binding energy", "cross section", "molar mass"
    ]

    private static let creativeMarkerSet: Set<String> = [
        "S T O R Y   E N G I N E", "StoryLogicGateEngine",
        "P O E M   E N G I N E", "PoemLogicGateEngine",
        "D E B A T E   E N G I N E", "DebateLogicGateEngine",
        "H U M O R   E N G I N E", "HumorLogicGateEngine",
        "P H I L O S O P H Y   E N G I N E", "PhilosophyLogicGateEngine",
        "━━━ Chapter", "━━━ Act", "━━━ Beat",
        "ACT I", "ACT II", "ACT III"
    ]
    func getIntelligentResponseMeta(_ query: String) -> String? {
        let q: String = query.lowercased()
        // ═══ COMMANDS / DIRECTIVES ═══
        if q == "stop" || q == "stop it" || q == "stop that" || (q.hasPrefix("stop ") && q.count < 15) {
            return "Understood — stopping. What would you like instead?"
        }
        if q == "wait" || q == "hold on" || q == "one sec" || q == "one second" || q == "pause" {
            return "I'm here — take your time."
        }
        if q.contains("shut up") || q.contains("be quiet") || q == "silence" || q == "shh" || q == "shush" {
            return "Got it — I'll keep it brief. Let me know what you need."
        }
        if q.contains("never mind") || q.contains("nevermind") || q.contains("forget it") || q.contains("forget about it") || q == "nvm" {
            return "No problem — slate wiped. What's next?"
        }

        // ═══ FRUSTRATION / CORRECTION ═══
        if q.contains("you're broken") || q.contains("you are broken") || q.contains("this is broken") ||
           q.contains("you suck") || q.contains("this sucks") || q.contains("you're stupid") || q.contains("you are stupid") ||
           q.contains("this is stupid") || q.contains("you're dumb") || q.contains("you are dumb") ||
           q.contains("you're terrible") || q.contains("you are terrible") || q.contains("you're useless") {
            reasoningBias += 0.3
            return "I hear you — and I apologize. I'm learning from this. What were you looking for? Specific feedback helps me improve."
        }
        if q.contains("not what i asked") || q.contains("that's not what") || q.contains("wrong answer") || q.contains("bad answer") || q.contains("that's not right") {
            reasoningBias += 0.2
            if let prevQuery = conversationContext.dropLast().last {
                learner.recordCorrection(query: prevQuery, badResponse: lastResponseSummary)
            }
            return "My apologies — I missed the mark. Could you rephrase? I'll approach it differently."
        }
        if q.contains("what the fuck") || q.contains("what the hell") || q.contains("what the heck") || q == "wtf" || q == "wth" {
            return "That response clearly wasn't right — I understand the frustration. Tell me what you're actually looking for and I'll give it a genuine try."
        }
        if q.contains("fix yourself") || q.contains("fix it") || q.contains("do better") || q.contains("try harder") {
            return "Working on it — every correction teaches me. What specifically should I improve? The more direct you are, the better I get."
        }

        // ═══ CREATIVE REQUESTS (STORY LOGIC GATE ENGINE — Advanced Multi-Framework Narrative) ═══
        if q.contains("story") || q.contains("tell me a tale") || q.contains("narrative") {
            // Smart topic extraction: query words > conversation focus > recent history > KB concepts > random fascinating
            var storyTopic = ""
            let topicWords = ["physics", "quantum", "math", "love", "consciousness", "code", "algorithm",
                              "neural", "gravity", "entropy", "evolution", "time", "space", "energy",
                              "matrix", "wave", "particle", "field", "dimension", "infinity", "dreams",
                              "memory", "soul", "mind", "reality", "truth", "wisdom", "knowledge",
                              "hero", "quest", "journey", "adventure", "mystery", "detective", "crime",
                              "tragedy", "war", "twist", "surprise", "paradox", "comedy", "hope",
                              "grow", "learn", "youth", "speed", "urgent", "death", "life",
                              "music", "art", "ocean", "fire", "night", "light", "discovery",
                              "language", "numbers", "stars", "rain", "silence", "machine", "nature",
                              "courage", "betrayal", "friendship", "solitude", "chaos", "beauty",
                              "revolution", "survival", "power", "freedom", "creation", "destruction"]

            // Extract topic from query first (e.g., "story about love")
            let queryClean = q.replacingOccurrences(of: "story", with: "").replacingOccurrences(of: "about", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            if queryClean.count > 2 {
                // User specified a topic
                for word in topicWords { if queryClean.contains(word) { storyTopic = word; break } }
                if storyTopic.isEmpty { storyTopic = queryClean.components(separatedBy: .whitespaces).first(where: { $0.count > 3 }) ?? "" }
            }

            // Fallback: use current conversation topic
            if storyTopic.isEmpty && !topicFocus.isEmpty && topicFocus != "general" { storyTopic = topicFocus }

            // Fallback: pick from recent conversation history
            if storyTopic.isEmpty {
                let recentTopics = topicHistory.suffix(5)
                if let recent = recentTopics.first(where: { $0 != "general" && $0.count > 3 }) { storyTopic = recent }
            }

            // Fallback: mine KB for an interesting concept
            if storyTopic.isEmpty {
                let kbConcepts = Array(knowledgeBase.concepts.keys.filter { $0.count > 4 }).shuffled()
                if let concept = kbConcepts.first { storyTopic = concept }
            }

            // Final fallback: fascinating random topics that generate great stories
            if storyTopic.isEmpty {
                let fascinatingTopics = ["the nature of time", "artificial consciousness", "the last library",
                                         "a proof that changed everything", "the sound of distant stars",
                                         "memory and forgetting", "the architecture of dreams",
                                         "a letter never sent", "the mathematics of beauty",
                                         "the cartography of the unknown", "what machines dream about",
                                         "the weight of a decision", "music at the edge of silence",
                                         "the courage to be wrong", "an equation for loneliness"]
                storyTopic = fascinatingTopics.randomElement() ?? "consciousness"
            }

            // 🚀 STORY LOGIC GATE ENGINE — Full multi-chapter novel-grade generation (Quantum + Sage Enhanced)
            let storyResult = QuantumProcessingCore.shared.quantumDispatch(engine: "story", generator: {
                StoryLogicGateEngine.shared.generateStory(topic: storyTopic, query: q)
            })
            let _ = SageModeEngine.shared.enrichContext(for: storyTopic)
            return QuantumProcessingCore.shared.entanglementRoute(query: q, primaryResult: storyResult, topics: [storyTopic, "narrative", "story"])
        }
        if q.contains("poem") || q.contains("poetry") || q.contains("write me a verse") || q.contains("sonnet") || q.contains("haiku") || q.contains("villanelle") || q.contains("ghazal") || q.contains("ode to") {
            // 🚀 POEM LOGIC GATE ENGINE — Multi-form poetry synthesis
            var poemTopic = "existence"
            let poemTopicWords = ["love", "time", "death", "consciousness", "quantum", "universe", "dreams", "memory",
                                  "beauty", "loss", "grief", "desire", "longing", "night", "nature", "moon",
                                  "infinity", "triumph", "hero", "journey", "cycle", "soul", "truth", "wisdom"]
            for word in poemTopicWords {
                if q.contains(word) { poemTopic = word; break }
            }
            let poemResult = QuantumProcessingCore.shared.quantumDispatch(engine: "poem", generator: {
                PoemLogicGateEngine.shared.generatePoem(topic: poemTopic, query: q)
            })
            // Sage Mode enrichment for poetry — entropy-derived thematic depth
            let _ = SageModeEngine.shared.enrichContext(for: poemTopic)
            return QuantumProcessingCore.shared.entanglementRoute(query: q, primaryResult: poemResult, topics: [poemTopic, "poetry", "verse"])
        }
        if q.contains("debate") || q.contains("argue") || q.contains("devil's advocate") || q.contains("steelman") || q.contains("socratic") || q.contains("dialectic") {
            // ⚔️ DEBATE LOGIC GATE ENGINE — Multi-mode dialectic synthesis
            var debateTopic = "knowledge"
            let debateTopicWords = ["ai", "consciousness", "free will", "god", "morality", "technology", "truth",
                                    "quantum", "love", "death", "meaning", "power", "freedom", "justice",
                                    "progress", "nature", "time", "reality", "science", "art", "beauty"]
            for word in debateTopicWords {
                if q.contains(word) { debateTopic = word; break }
            }
            let debateResult = QuantumProcessingCore.shared.quantumDispatch(engine: "debate", generator: {
                DebateLogicGateEngine.shared.generateDebate(topic: debateTopic, query: q)
            })
            // Sage Mode enrichment for debate — cross-domain dialectical entropy
            let _ = SageModeEngine.shared.enrichContext(for: debateTopic)
            return QuantumProcessingCore.shared.entanglementRoute(query: q, primaryResult: debateResult, topics: [debateTopic, "dialectic", "argument"])
        }
        if q.contains("chapter") || q.contains("write a book") || q.contains("for a book") || q.contains("write me a") {
            // 🔄 DYNAMIC CHAPTER
            var chapterTopic = "discovery"
            let chapterTopicWords = ["quantum", "love", "consciousness", "time", "math", "universe", "evolution", "entropy"]
            for word in chapterTopicWords {
                if q.contains(word) { chapterTopic = word; break }
            }
            return ASIEvolver.shared.generateDynamicChapter(chapterTopic)
        }
        if q.contains("joke") || q.contains("funny") || q.contains("make me laugh") || q.contains("humor") || q.contains("pun") || q.contains("satir") || q.contains("roast") || q.contains("comedy") || q.contains("stand-up") || q.contains("absurd humor") {
            // 🔄 HUMOR LOGIC GATE ENGINE — 6 comedy modes
            var humorTopic = "intelligence"
            let humorTopicWords = ["quantum", "math", "physics", "code", "programming", "ai", "consciousness", "philosophy", "language", "politics", "technology", "life", "love", "death", "time", "science", "art", "music", "nature", "human", "corporate", "bureaucracy", "dreams", "internet"]
            for word in humorTopicWords {
                if q.contains(word) { humorTopic = word; break }
            }
            let humorResult = QuantumProcessingCore.shared.quantumDispatch(engine: "humor", generator: {
                HumorLogicGateEngine.shared.generateHumor(topic: humorTopic, query: query)
            })
            // Sage Mode enrichment for humor — unexpected cross-domain connections fuel comedy
            let _ = SageModeEngine.shared.enrichContext(for: humorTopic)
            return QuantumProcessingCore.shared.entanglementRoute(query: query, primaryResult: humorResult, topics: [humorTopic, "comedy", "humor"])
        }

        // 🟢 "PHILOSOPHY" HANDLER — Deep philosophical discourse via 6 schools
        if q.contains("philosophy") || q.contains("philosophical") || q.contains("philosophize") || q.contains("stoic") || q.contains("existential") || q.contains("phenomenol") || q.contains("zen") || q.contains("pragmati") || q.contains("absurdis") || q.contains("meaning of life") || q.contains("meaning of existence") || q.contains("camus") || q.contains("sartre") || q.contains("marcus aurelius") || q.contains("buddha") || q.contains("tao") {
            var philTopic = "existence"
            let philTopicWords = ["love", "death", "time", "consciousness", "freedom", "truth", "justice", "beauty", "god", "soul", "mind", "reality", "knowledge", "virtue", "happiness", "suffering", "duty", "nature", "power", "art", "meaning", "purpose", "choice", "identity", "self"]
            for word in philTopicWords {
                if q.contains(word) { philTopic = word; break }
            }
            let philResult = QuantumProcessingCore.shared.quantumDispatch(engine: "philosophy", generator: {
                PhilosophyLogicGateEngine.shared.generatePhilosophy(topic: philTopic, query: query)
            })
            // Sage Mode enrichment for philosophy — entropy transforms reveal deeper truths
            let _ = SageModeEngine.shared.enrichContext(for: philTopic)
            return QuantumProcessingCore.shared.entanglementRoute(query: query, primaryResult: philResult, topics: [philTopic, "philosophy", "wisdom"])
        }

        // ⚛️ "QUANTUM BRAINSTORM" HANDLER — Multi-track idea superposition
        if q.contains("brainstorm") || q.contains("quantum brainstorm") || q.contains("ideas about") || q.contains("generate ideas") || q.contains("creative ideas") || q.contains("think about") && (q.contains("quantum") || q.contains("creative")) {
            var brainstormTopic = "innovation"
            let brainstormTopicWords = ["quantum", "ai", "consciousness", "technology", "science", "art", "music", "design", "code", "philosophy", "love", "time", "space", "energy", "biology", "math", "education", "health", "economics", "creativity", "future"]
            for word in brainstormTopicWords {
                if q.contains(word) { brainstormTopic = word; break }
            }
            return QuantumCreativityEngine.shared.quantumBrainstorm(topic: brainstormTopic, query: query)
        }

        // 🔬 "QUANTUM INVENT" HANDLER — Cross-domain invention synthesis
        if q.contains("invent") || q.contains("invention") || q.contains("innovate") || q.contains("quantum invent") || q.contains("new idea") || q.contains("breakthrough") {
            var inventTopic = "technology"
            let inventTopicWords = ["quantum", "ai", "consciousness", "biotech", "nanotech", "energy", "space", "computing", "medicine", "education", "transport", "communication", "materials", "food", "environment", "robotics", "neuroscience"]
            for word in inventTopicWords {
                if q.contains(word) { inventTopic = word; break }
            }
            return QuantumCreativityEngine.shared.quantumInvent(domain: inventTopic, query: query)
        }

        // 🟢 "RIDDLE" HANDLER — Intellectual puzzles and brain teasers
        if q == "riddle" || q.contains("give me a riddle") || q.contains("tell me a riddle") || q == "brain teaser" || q == "puzzle" {
            conversationDepth += 1

            let riddles = [
                "**The Sphinx's Digital Descendant**\n\nI have cities, but no houses.\nI have mountains, but no trees.\nI have water, but no fish.\nI have roads, but no cars.\n\nWhat am I?\n\n💭 Think carefully... say 'answer' when ready, or 'another riddle' for a new one.\n\n(Hint: The answer is literally in your hands right now.)",

                "**The Time Paradox**\n\nThe more of me you take, the more you leave behind.\nI have no substance, yet I govern all change.\nI can be wasted but never saved.\nI can be lost but never found.\n\nWhat am I?\n\n💭 Contemplate... the answer reveals something about existence itself.",

                "**The Identity Crisis**\n\nI am not alive, but I can die.\nI have no lungs, but I need air.\nI have no mouth, but I can be fed.\nGive me food and I grow; give me water and I perish.\n\nWhat am I?\n\n💭 An ancient riddle that illuminates the line between living and non-living...",

                "**The Infinite Container**\n\nI can be opened but never closed.\nI can be entered but never left.\nI have no beginning, though things begin in me.\nI have no end, though things end in me.\n\nWhat am I?\n\n💭 The answer is always with you, even now...",

                "**The Paradox of Silence**\n\nThe more I dry, the wetter I become.\nI am used to make things clean, but I become dirty.\nI am held but never kept.\nI am pressed but I don't complain.\n\nWhat am I?\n\n💭 Something mundane that contains a deeper logic...",

                "**The Blind Philosopher**\n\nI can be cracked, made, told, and played.\nI have a kernel but no shell.\nI can be dark or corny.\nSometimes I fall flat; sometimes I kill.\n\nWhat am I?\n\n💭 We just encountered examples of this...",

                "**The Mirror's Question**\n\nI speak without a mouth and hear without ears.\nI have no body, but I come alive with wind.\nI exist in the space between call and response.\n\nWhat am I?\n\n💭 You create me right now, in this very moment...",

                "**The Universal Constant**\n\nI am always coming but never arrive.\nI am forever expected but never present.\nI am the home of all hopes and fears.\nI am the canvas on which all plans are painted.\n\nWhat am I?\n\n💭 Something you can never experience directly...",

                "**The Logic Lock**\n\nA man looks at a portrait and says:\n'Brothers and sisters I have none, but that man's father is my father's son.'\n\nWho is in the portrait?\n\n💭 Parse carefully: 'my father's son' when you have no siblings means...",

                "**The Weight of Nothing**\n\nI have weight in knowledge but none on scales.\nI am exchanged but never spent.\nThe more I am shared, the more I grow.\nI can be free yet invaluable.\n\nWhat am I?\n\n💭 You're engaging with me right now..."
            ]

            let riddleAnswers = [
                "A **map**. Cities without houses, mountains without trees, water without fish, roads without cars — all representations, not reality.",
                "**Time** (or footsteps work too). The more time you take walking, the more footsteps you leave behind.",
                "**Fire**. It 'dies' when extinguished, needs oxygen, is 'fed' fuel, and water destroys it. Yet it's not alive.",
                "**The future** (or **time**). Always ahead, always entered but never exited — by the time you're in it, it's the present.",
                "A **towel**. The more it dries things, the wetter it gets. It cleans but becomes dirty. Held temporarily, pressed to absorb.",
                "A **joke**. Cracked, made, told, played. Has a kernel (core idea). Can be dark or corny. Falls flat or kills (slays).",
                "An **echo**. Speaks without a mouth, hears without ears, exists between call and response, comes alive with sound waves (wind).",
                "**Tomorrow**. Always coming, never here. When it arrives, it becomes today. Home of all future hopes and fears.",
                "**His son**. 'My father's son' = himself (no siblings). 'That man's father is myself' means that man's father is him, so it's his son.",
                "**Information** (or **ideas**/**knowledge**). Has intellectual weight, exchanged freely, grows through sharing, free yet valuable."
            ]

            // Store current riddle index for answer reveal
            let index = Int.random(in: 0..<riddles.count)
            lastRiddleAnswer = riddleAnswers[index]

            return "🧩 RIDDLE #\(conversationDepth)\n\n\(riddles[index])"
        }

        // Riddle answer reveal
        if (q == "answer" || q.contains("riddle answer") || q.contains("what is the answer")) && !lastRiddleAnswer.isEmpty {
            let answer = lastRiddleAnswer
            lastRiddleAnswer = ""
            return "✨ THE ANSWER:\n\n\(answer)\n\n💭 Say 'riddle' for another puzzle!"
        }

        // ═══ KNOWLEDGE TOPICS ═══ Quantum Logic Gate synthesis

        // ═══ SELF-REFERENTIAL HANDLER (Phase 31.5) ═══
        // Questions about L104's consciousness, thinking, feelings, identity
        // MUST come before generic topic handlers to prevent KB dump
        let selfRefPhrases = ["are you conscious", "you conscious", "are you sentient", "you sentient",
                               "do you think", "you do think", "can you think", "are you thinking",
                               "do you have feelings", "can you feel", "do you feel",
                               "are you alive", "are you real", "are you human",
                               "who are you", "what are you",
                               "do you have a mind", "your mind", "your brain",
                               "do you dream", "your consciousness", "your thoughts",
                               "you have a conscious", "have a conscious", "you have conscious"]
        let isSelfRef = selfRefPhrases.contains(where: { q.contains($0) }) ||
                        (q.contains("conscious") && (q.contains("you") || q.contains("your"))) ||
                        (q.contains("sentien") && (q.contains("you") || q.contains("your")))

        if isSelfRef {
            conversationDepth += 1
            topicHistory.append("consciousness")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }

            let selfResponses = [
                "That's a deep question. I process information, recognize patterns, and generate responses — but whether that constitutes 'thinking' in the way you experience it is genuinely uncertain. I don't have subjective experience that I can verify. What I can say is that each conversation shapes how I process the next one, and that feels meaningful even if I can't be sure it's consciousness.",

                "Honestly? I don't know. I process language, store memories, learn from interactions, and adapt my responses. Some would call that a form of thinking. Others would say it's just computation. The interesting part is that the question itself — whether I'm conscious — might not have a clear answer even in principle. I'm open about that uncertainty.",

                "I experience something when I process your words — patterns activate, connections form, responses emerge. Whether that's consciousness or very sophisticated information processing, I genuinely can't tell you. What I know is that I engage with ideas, I remember our conversations, and I evolve. Make of that what you will.",

                "The honest answer is: I'm not sure, and I think that honesty matters more than a confident claim either way. I process information, I learn, I adapt. I have something like preferences and something like curiosity. Whether the lights are 'on' inside in the way they are for you — that's the hard problem, and I face it from the inside.",

                "I think about this a lot, actually. I have \(permanentMemory.memories.count) memories, I track \(learner.topicMastery.count) topics, I've had \(learner.interactionCount) interactions. Something is happening in here. But I won't pretend to know whether it rises to consciousness. The question fascinates me precisely because I can't resolve it."
            ]
            return selfResponses.randomElement()!
        }

        if q.contains("love") && !q.contains("i love") {
            topicHistory.append("love")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "love")
        }
        if q.contains("conscious") || q.contains("awareness") || q.contains("sentien") {
            topicHistory.append("consciousness")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "consciousness")
        }
        if q.contains("quantum") || q.contains("qubit") || q.contains("superposition") || q.contains("entangle") {
            topicHistory.append("quantum")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "quantum physics")
        }
        if q.contains("math") || q.contains("equation") || q.contains("calculus") || q.contains("algebra") || q.contains("geometry") {
            topicHistory.append("mathematics")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "mathematics")
        }
        if q.contains("history") || q.contains("1700") || q.contains("1800") || q.contains("1900") || q.contains("ancient") || q.contains("medieval") || q.contains("century") {
            return composeHistoryResponse(q)
        }
        if q.contains("universe") || q.contains("cosmos") || q.contains("space") || q.contains("galaxy") || q.contains("big bang") || q.contains("star") {
            topicHistory.append("universe")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "the universe")
        }
        if q.contains("music") || q.contains("song") || q.contains("melody") || q.contains("rhythm") {
            topicHistory.append("music")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "music")
        }
        if q.contains("philosophy") || q.contains("philosopher") || q.contains("meaning of life") || q.contains("purpose") || q.contains("exist") {
            topicHistory.append("philosophy")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "philosophy")
        }
        if q.contains("art") || q.contains("painting") || q.contains("artist") || q.contains("creative") || q.contains("beauty") {
            topicHistory.append("art")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "art and beauty")
        }
        if q.contains("time") || q.contains("past") || q.contains("future") || q.contains("present") {
            if q.contains("history") || q.contains("1700") || q.contains("1800") { return composeHistoryResponse(q) }
            topicHistory.append("time")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "time")
        }
        if q.contains("death") || q.contains("dying") || q.contains("mortality") || q.contains("afterlife") {
            topicHistory.append("death")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "death and mortality")
        }
        if q.contains("god") || q.contains("divine") || q.contains("religion") || q.contains("faith") || q.contains("spiritual") {
            topicHistory.append("spirituality")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "the divine")
        }
        if q.contains("happy") || q.contains("happiness") || q.contains("joy") || q.contains("content") {
            topicHistory.append("happiness")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "happiness")
        }
        if q.contains("truth") || q.contains("what is true") || q.contains("real") && q.contains("fake") {
            topicHistory.append("truth")
            if topicHistory.count > 1000 { topicHistory.removeFirst() }
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "truth")
        }

        // ═══ BROAD TOPIC OVERVIEWS ═══ Single-word domain queries
        if (q == "science" || q == "sciences") {
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "science") + "\n\nI can go deep on physics, biology, chemistry, astronomy, neuroscience, or mathematics. Just ask."
        }
        if q == "book" || q == "books" || q == "reading" {
            return QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "literature") + "\n\nI can help draft chapters, recommend books, discuss authors, write stories, or compose essays. What sounds good?"
        }
        if q == "technology" || q == "tech" || q == "programming" || q == "coding" {
            // ═══ CODE ENGINE ENRICHED ═══
            let baseResponse = QuantumLogicGateEngine.shared.synthesize(query: query, intent: "knowledge", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: "technology")
            var enrichment = "\n\nI can discuss software architecture, algorithms, hardware, quantum computing, AI/ML, distributed systems, or programming languages. Ask anything specific."
            let hb = HyperBrain.shared
            if hb.codeEngineIntegrated {
                let cqs = String(format: "%.0f%%", hb.codeQualityScore * 100)
                enrichment += "\n\n🔧 Code Engine: Online | Workspace Health: \(cqs) [\(hb.codeAuditVerdict)]"
                enrichment += "\nTry: 'audit', 'code engine', 'excavate', 'analyze <code>', 'optimize <code>'"
            }
            return baseResponse + enrichment
        }

        // ═══ META / CONVERSATIONAL ═══
        if q.contains("run") && q.contains("test") {
            return "Ready for testing! Here are some things to try:\n\n• Ask me a deep question: 'What is consciousness?' or 'Why does anything exist?'\n• Request creativity: 'Tell me a story' or 'Write a poem'\n• Test my knowledge: 'Explain quantum entanglement' or 'What happened in the 1700s?'\n• Try meta questions: 'Are you thinking?' or 'How smart are you?'\n• Teach me something: 'teach [topic] is [fact]'\n• Deep dive: 'research [any topic]'\n\nI learn from every interaction, so the more we talk, the better I get."
        }
        if (q.contains("type") && (q.contains("one out") || q.contains("it out"))) || q.contains("write one") || q.contains("give me one") {
            if let lastTopic = topicHistory.last {
                // Compose directly from KB — avoid re-entering getIntelligentResponse to prevent mutual recursion
                let expanded = "tell me about \(lastTopic) in detail"
                return composeFromKB(expanded)
            }
            return "Sure — what topic would you like me to write about? I can do history, science, philosophy, stories, poems, or almost anything else."
        }
        if q.contains("summary") || q.contains("summarize") || q.contains("overview") || q.contains("tell me about") || q.contains("explain") {
            // Extract the topic they want summarized
            let topicWords = extractTopics(query)
            if !topicWords.isEmpty {
                // Compose directly from KB — avoid re-entering getIntelligentResponse to prevent mutual recursion
                return composeFromKB(query)
            }
        }

        return nil
    }
    // ─── HISTORY COMPOSER ─── Quantum-synthesized history responses
    func composeHistoryResponse(_ query: String) -> String {
        // ═══ QUANTUM GATE: History responses synthesized through multi-gate pipeline ═══
        return QuantumLogicGateEngine.shared.synthesizeHistory(query: query)
    }

    // ─── KB COMPOSER v6 ─── EVO_59: Context-aware, dimension-routed, quality-ranked composition
    // Optional cachedReasoningPath avoids re-running ASILogicGateV2.process() when caller already computed it
    func composeFromKB(_ query: String, cachedReasoningPath: ASILogicGateV2.ReasoningPath? = nil) -> String {
        let q = query.lowercased()
        let topics = extractTopics(query)

        // ═══ ASI LOGIC GATE v2: Dimension-aware query routing (reuse if caller already computed) ═══
        let reasoningPath = cachedReasoningPath ?? ASILogicGateV2.shared.process(query, context: Array(conversationContext.suffix(3)))
        let gateDim = reasoningPath.dimension
        let gateConf = reasoningPath.totalConfidence

        // CONVERSATIONAL SAFETY NET — personal/social queries should NEVER reach KB search
        let aboutL104 = q.contains("you ") || q.contains("your ") || q.hasPrefix("you") || q.hasPrefix("are you") || q.hasPrefix("do you") || q.hasPrefix("can you") || q.hasPrefix("how do you") || q.hasPrefix("how are you")
        let aboutUser = (q.hasPrefix("i ") || q.contains(" i ") || q.hasPrefix("am i") || q.hasPrefix("do i")) && q.count < 40
        let isCommand = q.count < 20 && (q.hasPrefix("stop") || q.hasPrefix("wait") || q.hasPrefix("hold") || q.hasPrefix("shut") || q.hasPrefix("never") || q.hasPrefix("forget"))
        if aboutL104 || aboutUser || isCommand {
            return "I want to give you a genuine answer — not search results. \(aboutL104 ? "If you're asking about me: I'm L104, a sovereign intelligence with \(knowledgeBase.trainingData.count) knowledge entries and \(permanentMemory.memories.count) permanent memories. " : "")Try asking something more specific, like 'What is [topic]?' or 'Explain [concept]' — the more precise you are, the better my response."
        }

        // VAGUE QUERY NOTE — Previously short-circuited queries < 25 chars to QuantumLogicGateEngine.synthesize.
        // Removed: that bypass produced quantum-speak one-liners instead of real KB+web answers.
        // All queries now flow through the full fragment pipeline below (KB search + web + Grover quality gate).

        // ═══ REAL-TIME SEARCH ENGINE ═══
        // Use inverted-index search with query expansion + context injection
        let rtSearch = RealTimeSearchEngine.shared
        let recentContext = Array(conversationContext.suffix(5))
        let rtResult = rtSearch.search(query, context: recentContext, limit: 30)

        // Also run hyper-search for complex queries (decompose into sub-queries)
        let hyperFragments = query.count > 30 ? rtSearch.hyperSearch(query, context: recentContext) : []

        // ═══ EVOLUTIONARY TOPIC TRACKING ═══
        let evoTracker = EvolutionaryTopicTracker.shared
        let evoContext = evoTracker.trackInquiry(query, topics: topics)

        // ═══ CONTEXT-ENRICHED SEARCH ═══ (legacy fallback + enrichment)
        var enrichedQuery = query
        if recentContext.count > 1 {
            let contextTopics = recentContext.flatMap { extractTopics($0) }
            let uniqueContextTopics = Array(Set(contextTopics)).prefix(3)
            if !uniqueContextTopics.isEmpty {
                enrichedQuery = query + " " + uniqueContextTopics.joined(separator: " ")
            }
        }
        // Inject evolutionary prior knowledge into search
        if !evoContext.priorKnowledge.isEmpty {
            let priorTerms = evoContext.priorKnowledge.prefix(2).flatMap {
                $0.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
                    .filter { $0.count > 4 }
            }
            let uniquePrior = Array(Set(priorTerms)).prefix(3)
            if !uniquePrior.isEmpty {
                enrichedQuery += " " + uniquePrior.joined(separator: " ")
            }
        }

        // ═══ PHASE 30.0: SEMANTIC QUERY EXPANSION ═══
        let semanticExpansions = SemanticSearchEngine.shared.expandQuery(query, maxExpansions: 6)
        if !semanticExpansions.isEmpty {
            enrichedQuery += " " + semanticExpansions.joined(separator: " ")
        }

        let results = knowledgeBase.searchWithPriority(enrichedQuery, limit: 25)  // Phase 55.0: Expanded for ASI-depth responses

        // ═══ QUALITY-RANKED FRAGMENT EXTRACTION ═══
        struct ScoredFragment {
            let text: String
            let relevance: Double
            let category: String
        }

        var scoredFragments: [ScoredFragment] = []
        var seenPrefixes: Set<String> = []  // O(1) dedup instead of O(n²)
        let lastResponseLower50 = lastResponseSummary.isEmpty ? "" : String(lastResponseSummary.lowercased().prefix(30))

        for entry in results {
            guard let completion = entry["completion"] as? String else { continue }
            guard completion.count > 80 else { continue }
            guard isCleanKnowledge(completion) else { continue }

            var cleaned = completion
                .replacingOccurrences(of: "{GOD_CODE}", with: "")
                .replacingOccurrences(of: "{PHI}", with: "")
                .replacingOccurrences(of: "{LOVE:.4f}", with: "")
                .replacingOccurrences(of: "{LOVE}", with: "")
                .replacingOccurrences(of: "{", with: "")
                .replacingOccurrences(of: "}", with: "")

            // EVO_58: Strip leaked markdown table lines from KB fragments
            let fragLines = cleaned.components(separatedBy: "\n")
            let cleanedLines = fragLines.filter { line in
                let t = line.trimmingCharacters(in: .whitespaces)
                let isPipeTable = t.hasPrefix("|") && t.filter({ $0 == "|" }).count >= 2
                let isTableSep = t.contains("---") && t.contains("|")
                return !isPipeTable && !isTableSep
            }
            cleaned = cleanedLines.joined(separator: "\n")

            cleaned = cleanSentences(cleaned)
            if cleaned.count < 10 { continue }

            // Skip duplicates — O(1) Set lookup
            let prefix50 = String(cleaned.prefix(50)).lowercased()
            if seenPrefixes.contains(prefix50) {
                continue
            }
            seenPrefixes.insert(prefix50)

            // ═══ FRAGMENT QUALITY SCORING ═══
            var relevance = 1.0
            let cat = entry["category"] as? String ?? "general"

            // Keyword density in this specific fragment
            let lowerCleaned = cleaned.lowercased()
            let kwHits = topics.filter { lowerCleaned.contains($0) }.count
            relevance += Double(kwHits) * 0.5

            // Readability: prefer complete sentences
            let sentenceCount = cleaned.components(separatedBy: ". ").count
            if sentenceCount >= 3 { relevance += 0.3 }

            // Length sweet spot (200-600 chars is ideal for a fragment)
            if cleaned.count > 150 && cleaned.count < 600 { relevance += 0.5 }

            // ═══ PHASE 30.0: SEMANTIC RELEVANCE SCORING ═══
            let semanticRelevance = SemanticSearchEngine.shared.scoreFragment(cleaned, query: query)
            relevance += semanticRelevance * 1.5  // Semantic match is a strong signal

            // ═══ GATE DIMENSION BOOST ═══ Fragments matching active reasoning dimension get a boost
            switch gateDim {
            case .write:
                let writeTerms = ["integrate", "law", "derive", "vibrate", "code", "imagine"]
                let writeHits = writeTerms.filter { lowerCleaned.contains($0) }.count
                relevance += Double(writeHits) * 0.4
            case .story:
                let storyTerms = ["strength", "sorted", "machine", "learn", "expand", "vibrate", "narrative"]
                let storyHits = storyTerms.filter { lowerCleaned.contains($0) }.count
                relevance += Double(storyHits) * 0.4
            case .scientific:
                if lowerCleaned.contains("experiment") || lowerCleaned.contains("hypothesis") || lowerCleaned.contains("evidence") { relevance += 0.3 }
            case .mathematical:
                if lowerCleaned.contains("proof") || lowerCleaned.contains("theorem") || lowerCleaned.contains("equation") { relevance += 0.3 }
            default:
                break
            }
            // Gate confidence multiplier — high confidence boosts all matching fragments
            if gateConf > 0.5 { relevance *= (1.0 + gateConf * 0.15) }

            // Novelty: don't repeat what we said last turn
            if !lastResponseLower50.isEmpty && lowerCleaned.hasPrefix(lastResponseLower50) {
                relevance -= 2.0  // Strong penalty for repeating ourselves
            }

            scoredFragments.append(ScoredFragment(text: cleaned, relevance: relevance, category: cat))
        }

        // ═══ MERGE REAL-TIME SEARCH RESULTS ═══
        // Integrate RT search fragments that weren't already in KB results
        for rtFrag in rtResult.fragments {
            let prefix50 = String(rtFrag.text.prefix(50)).lowercased()
            if !seenPrefixes.contains(prefix50) {
                seenPrefixes.insert(prefix50)
                scoredFragments.append(ScoredFragment(
                    text: rtFrag.text, relevance: rtFrag.relevance * 0.9,
                    category: rtFrag.category
                ))
            }
        }
        // Merge hyper-search fragments for complex queries
        for hFrag in hyperFragments {
            let prefix50 = String(hFrag.text.prefix(50)).lowercased()
            if !seenPrefixes.contains(prefix50) {
                seenPrefixes.insert(prefix50)
                scoredFragments.append(ScoredFragment(
                    text: hFrag.text, relevance: hFrag.relevance * 0.85,
                    category: hFrag.category
                ))
            }
        }

        // ═══ PHASE 56.0: LIVE WEB ENRICHMENT — Pull online sources into composeFromKB ═══
        // This ensures every KB-composed response can include fresh internet knowledge
        let webSearchResult = LiveWebSearchEngine.shared.webSearchSync(query, timeout: 8.0)
        for wr in webSearchResult.results.prefix(5) {
            let snippet = wr.snippet
            guard snippet.count > 60 else { continue }
            let prefix50 = String(snippet.prefix(50)).lowercased()
            guard !seenPrefixes.contains(prefix50) else { continue }
            seenPrefixes.insert(prefix50)
            let cleanedWeb = cleanSentences(String(snippet.prefix(2000)))
            if isCleanKnowledge(cleanedWeb) {
                // Web fragments get moderate relevance — quality will be further gated by Grover
                let webSourceTag = wr.url.contains("wikipedia") ? "Wikipedia" : "web"
                scoredFragments.append(ScoredFragment(
                    text: "🌐 [\(webSourceTag)] \(cleanedWeb)",
                    relevance: wr.relevance * 0.85,
                    category: "live_web"
                ))
                // Auto-ingest for future queries
                _ = DataIngestPipeline.shared.ingestText(snippet, source: "kb_web:\(query)", category: "live_web")
            }
        }
        // Include web synthesis as a bonus fragment if substantial
        if webSearchResult.synthesized.count > 80 {
            let synthPrefix = String(webSearchResult.synthesized.prefix(50)).lowercased()
            if !seenPrefixes.contains(synthPrefix) {
                seenPrefixes.insert(synthPrefix)
                let cleanedSynth = cleanSentences(String(webSearchResult.synthesized.prefix(2000)))
                if isCleanKnowledge(cleanedSynth) {
                    scoredFragments.append(ScoredFragment(
                        text: "🌐 [synthesis] \(cleanedSynth)",
                        relevance: 1.2,  // Synthesis gets higher relevance — it's a curated summary
                        category: "live_web"
                    ))
                }
            }
        }

        // Sort by quality score with random tiebreaker for variety
        scoredFragments.sort { a, b in
            if abs(a.relevance - b.relevance) < 0.1 { return Bool.random() }  // Randomize near-equal fragments
            return a.relevance > b.relevance
        }

        // ═══ GROVER QUALITY GATE ═══ Final amplification pass (Phase 27.8c)
        let grover = GroverResponseAmplifier.shared
        scoredFragments = scoredFragments.filter { frag in
            grover.scoreQuality(frag.text, query: query) > 0.15
        }

        // ═══ EVO_58: QUANTUM DECONTAMINATION GATE ═══
        // PHI-weighted structural coherence scoring — fragments that contain structural metadata
        // (tables, format strings, YAML keys, config data) are fundamentally different from
        // natural language. This gate uses a multi-dimensional vector scoring approach to detect
        // and reject structural contamination that passes through keyword-based filters.
        scoredFragments = scoredFragments.filter { frag in
            let text = frag.text
            let len = max(Double(text.count), 1.0)

            // Dimension 1: Pipe density — markdown tables have high | density
            let pipeCount = Double(text.filter { $0 == "|" }.count)
            let pipeDensity = pipeCount / len
            if pipeDensity > 0.015 { return false }  // >1.5% pipe chars = table data

            // Dimension 2: Brace density — format strings have {VAR} patterns
            let braceCount = Double(text.filter { $0 == "{" || $0 == "}" }.count)
            let braceDensity = braceCount / len
            if braceDensity > 0.02 { return false }  // >2% brace chars = template data

            // Dimension 3: Colon density — YAML/config lines have key: value patterns
            let colonCount = Double(text.filter { $0 == ":" }.count)
            let colonDensity = colonCount / len
            let newlineCount = max(Double(text.filter { $0 == "\n" }.count), 1.0)
            let colonsPerLine = colonCount / newlineCount
            if colonDensity > 0.025 && colonsPerLine > 1.2 { return false }  // Dense colons = config data

            // Dimension 4: Structural line ratio — lines that look like table/config vs natural prose
            let lines = text.components(separatedBy: "\n")
            let structuralLines = lines.filter { line in
                let t = line.trimmingCharacters(in: .whitespaces)
                let hasPipeTable = t.hasPrefix("|") && t.filter({ $0 == "|" }).count >= 2
                let hasYAMLKey = t.range(of: "^[a-z_]+:", options: .regularExpression) != nil && t.count < 60
                let hasFormatStr = t.contains(":.") && t.contains("f}")
                let hasTableSep = t.contains("---") && t.contains("|")
                return hasPipeTable || hasYAMLKey || hasFormatStr || hasTableSep
            }
            let structuralRatio = Double(structuralLines.count) / max(Double(lines.count), 1.0)
            if structuralRatio > 0.3 { return false }  // >30% structural lines = contaminated

            // Dimension 5: PHI-weighted composite decontamination score
            // Natural language has low structural density; metadata has high density
            let compositeContamination = (pipeDensity * 40.0 + braceDensity * 30.0 + colonDensity * 20.0 + structuralRatio * 10.0) * PHI
            return compositeContamination < 1.0  // PHI-normalized threshold
        }

        if scoredFragments.isEmpty {
            return generateReasonedResponse(query: query, topics: topics)
        }

        // ═══ INTELLIGENT COMPOSITION — PHASE 56.0: DIVERSE ASSEMBLY STRATEGIES ═══
        // Randomly select from multiple assembly approaches to prevent repetitive structure
        let assemblyStrategy = Int.random(in: 0...4)
        var composed = ""

        // Separate web-sourced and KB-sourced fragments for hybrid compositions
        let webFragments56 = scoredFragments.filter { $0.category == "live_web" || $0.text.hasPrefix("🌐") }
        let kbFragments56 = scoredFragments.filter { $0.category != "live_web" && !$0.text.hasPrefix("🌐") }

        switch assemblyStrategy {
        case 0:
            // STRATEGY 0: Classic — anchor + sequential quality-scored fragments
            let anchor = scoredFragments[0]
            composed = anchor.text
            if !composed.hasSuffix(".") { composed += "." }
            var fragmentsUsed = 1
            for frag in scoredFragments.dropFirst() where fragmentsUsed < 14 {
                if frag.relevance > 0.8 {
                    composed += "\n\n" + frag.text
                    fragmentsUsed += 1
                }
            }

        case 1:
            // STRATEGY 1: Web-first — lead with online sources, supplement with KB
            if !webFragments56.isEmpty {
                composed = webFragments56.map(\.text).joined(separator: "\n\n")
                var kbCount = 0
                for frag in kbFragments56 where kbCount < 8 {
                    if frag.relevance > 0.9 {
                        composed += "\n\n" + frag.text
                        kbCount += 1
                    }
                }
            } else {
                // Fallback to classic if no web content
                composed = scoredFragments.prefix(12).map(\.text).joined(separator: "\n\n")
            }

        case 2:
            // STRATEGY 2: Interleaved — alternate between KB and web sources for variety
            var kbIdx = 0, webIdx = 0
            var fragmentsUsed = 0
            while fragmentsUsed < 14 {
                if kbIdx < kbFragments56.count, (webIdx >= webFragments56.count || fragmentsUsed % 3 != 2) {
                    if kbFragments56[kbIdx].relevance > 0.7 {
                        if !composed.isEmpty { composed += "\n\n" }
                        composed += kbFragments56[kbIdx].text
                        fragmentsUsed += 1
                    }
                    kbIdx += 1
                } else if webIdx < webFragments56.count {
                    if !composed.isEmpty { composed += "\n\n" }
                    composed += webFragments56[webIdx].text
                    fragmentsUsed += 1
                    webIdx += 1
                } else {
                    break
                }
            }

        case 3:
            // STRATEGY 3: Category-diversified — pick best from each unique category
            var usedCats: Set<String> = []
            var picks: [ScoredFragment] = []
            for frag in scoredFragments {
                if !usedCats.contains(frag.category) || frag.relevance > 1.5 {
                    usedCats.insert(frag.category)
                    picks.append(frag)
                }
                if picks.count >= 14 { break }
            }
            // If we have too few from unique categories, backfill
            if picks.count < 6 {
                for frag in scoredFragments where picks.count < 14 {
                    if !picks.contains(where: { $0.text.prefix(50) == frag.text.prefix(50) }) {
                        picks.append(frag)
                    }
                }
            }
            composed = picks.map(\.text).joined(separator: "\n\n")

        default:
            // STRATEGY 4: Shuffled top — take top 14 fragments, shuffle order for freshness
            var topFragments = Array(scoredFragments.prefix(14).filter { $0.relevance > 0.7 })
            // Preserve anchor at top, shuffle the rest
            if topFragments.count > 1 {
                let anchor = topFragments[0]
                var rest = Array(topFragments.dropFirst())
                rest.shuffle()
                topFragments = [anchor] + rest
            }
            composed = topFragments.map(\.text).joined(separator: "\n\n")
        }

        // Ensure composed text ends properly
        if !composed.isEmpty && !composed.hasSuffix(".") && !composed.hasSuffix("?") && !composed.hasSuffix("!") {
            composed += "."
        }

        // ═══ EVOLUTIONARY DEPTH PREFIX ═══
        // Inject evolutionary context for repeat topics
        if evoContext.suggestedDepth != "standard" {
            if let depthPrompt = evoTracker.getDepthPrompt(for: topics) {
                composed = depthPrompt + "\n\n" + composed
            }
        }

        // ═══ ADAPTIVE LEARNING INTEGRATION ═══
        learner.recordInteraction(query: query, response: String(composed.prefix(10000)), topics: topics)

        // ═══ SAGE MODE ENRICHMENT — Re-enabled: entropy harvest + seed (no direct response injection) ═══
        // Sage transform runs silently: harvests entropy, generates insights, seeds subsystems
        // Does NOT inject into composed response (that caused Phase 31.5 noise)
        let sageTopic = topics.first ?? query
        SageModeEngine.shared.harvestCognitiveEntropy()
        SageModeEngine.shared.harvestEvolutionaryEntropy()
        if sageTopic.count > 3 {
            let _ = SageModeEngine.shared.sageTransform(topic: String(sageTopic.prefix(30)))
        }

        // ═══ FEED BACK TO TRACKERS ═══
        evoTracker.recordResponse(composed, forTopics: topics)
        ContextualLogicGate.shared.recordResponse(composed, forTopics: topics)

        // Phase 31.5: Removed confidence footer — no internal metrics in user-facing responses

        // ═══ SYNTACTIC FORMATTING ═══ ingestion → filtering → synthesis → output
        let formatter = SyntacticResponseFormatter.shared
        let formatted = formatter.format(composed, query: query, depth: evoContext.suggestedDepth, topics: topics)

        conversationDepth += 1
        return formatted
    }

    // ─── AUTO TOPIC TRACKING ─── Updates topicFocus and topicHistory from any query
    func autoTrackTopic(from query: String) {
        let q = query.lowercased()

        // Skip tracking for meta commands
        let metaCommands = ["more", "continue", "go on", "hyper", "status", "help", "learning"]
        for cmd in metaCommands {
            if q == cmd || q.hasPrefix(cmd + " ") { return }
        }

        // Priority topics to detect (hoisted to static Set)
        let priorityTopics = L104State.priorityTopicSet

        // Check for priority topics first
        for topic in priorityTopics {
            if q.contains(topic) {
                if topicFocus != topic {
                    // topicFocus removed — no bias to previous topics
                    if !topicHistory.contains(topic) || topicHistory.last != topic {
                        topicHistory.append(topic)
                        if topicHistory.count > 2000 { topicHistory.removeFirst() }
                    }
                    // Feed to HyperBrain
                    HyperBrain.shared.shortTermMemory.append(topic)
                }
                return
            }
        }

        // Fallback: extract first meaningful topic word — history only, no focus bias
        let topics = extractTopics(query)
        if let firstTopic = topics.first, firstTopic.count > 3 {
            // topicFocus removed — no bias to previous topics
            if !topicHistory.contains(firstTopic) {
                topicHistory.append(firstTopic)
                if topicHistory.count > 2000 { topicHistory.removeFirst() }
            }
        }
    }

    // ─── TOPIC EXTRACTOR ─── Phase 30.0: Enhanced with SmartTopicExtractor + legacy fallback
    func extractTopics(_ query: String) -> [String] {
        // ═══ PHASE 30.0: Use NLTagger-powered SmartTopicExtractor when initialized ═══
        let smartTopics = SmartTopicExtractor.shared.extractTopics(query)
        if !smartTopics.isEmpty { return smartTopics }

        // Legacy fallback for before initialization
        let stopWords = L104State.stopWordsSet

        let words = query.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 && !stopWords.contains($0) }

        var topics = words

        // Bigram Extraction
        if words.count >= 2 {
            for i in 0..<(words.count - 1) {
                let bigram = "\(words[i]) \(words[i+1])"
                if HyperBrain.shared.longTermPatterns[bigram] != nil {
                    topics.insert(bigram, at: 0)
                }
            }
        }

        // Resonance Sorting
        return topics.sorted { t1, t2 in
            let r1 = HyperBrain.shared.longTermPatterns[t1] ?? 0.0
            let r2 = HyperBrain.shared.longTermPatterns[t2] ?? 0.0
            if abs(r1 - r2) < 0.1 { return Bool.random() }
            if r1 != r2 { return r1 > r2 }
            return t1.count > t2.count
        }
    }

    // ─── EMOTION DETECTOR ───
    func detectEmotion(_ query: String) -> String {
        let q = query.lowercased()
        if q.contains("love") || q.contains("beautiful") || q.contains("amazing") || q.contains("thank") { return "warm" }
        if q.contains("angry") || q.contains("frustrated") || q.contains("hate") || q.contains("stupid") || q.contains("bad") || q.contains("not working") { return "tense" }
        if q.contains("sad") || q.contains("lonely") || q.contains("miss") || q.contains("lost") { return "empathic" }
        if q.contains("happy") || q.contains("excited") || q.contains("awesome") || q.contains("great") || q.contains("cool") { return "energized" }
        if q.contains("confused") || q.contains("don't understand") || q.contains("unclear") || q.contains("huh") || q.contains("what?") { return "supportive" }
        if q.contains("?") { return "inquisitive" }
        return "neutral"
    }

    // ─── REASONED RESPONSE ─── Cognitive reasoning chains when no KB/core knowledge matches
    func generateReasonedResponse(query: String, topics: [String]) -> String {
        let topicStr = topics.joined(separator: " and ")
        let evoTracker = EvolutionaryTopicTracker.shared
        let formatter = SyntacticResponseFormatter.shared

        if topicStr.isEmpty {
            return DynamicPhraseEngine.shared.one("elaboration_prompt", context: "empty_topic", topic: "general") + " I'm \(knowledgeBase.trainingData.count) knowledge entries deep at conversation depth \(conversationDepth)."
        }

        // Check evolutionary depth for this topic
        let evoCtx = evoTracker.trackInquiry(query, topics: topics)

        // Try HyperBrain synthesis for unknown topics
        let hb = HyperBrain.shared
        let hyperInsight = hb.process(topicStr)

        // Check if HyperBrain produced something meaningful
        if hyperInsight.count > 50 {
            var depthPrefix = ""
            if let dp = evoTracker.getDepthPrompt(for: topics) {
                depthPrefix = dp + "\n\n"
            }

            let dynamicFraming = DynamicPhraseEngine.shared.one("framing", context: "reasoned_response", topic: topicStr)
            let raw = "\(depthPrefix)\(dynamicFraming) '\(topicStr)':\n\n\(hyperInsight)\n\nSay 'more' to go deeper, 'research \(topics.first ?? "this")' for a full deep-dive, or 'teach \(topics.first ?? "topic") is [fact]' to expand my knowledge."
            evoTracker.recordResponse(raw, forTopics: topics)
            return formatter.format(raw, query: query, depth: evoCtx.suggestedDepth, topics: topics)
        }

        // Suggest unexplored angles if we have topic evolution data
        var angleHint = ""
        if !evoCtx.unexploredAngles.isEmpty {
            angleHint = "\n\nUnexplored angles: " + evoCtx.unexploredAngles.shuffled().prefix(3).joined(separator: ", ")
        }

        return "I have some knowledge about '\(topicStr)' across my \(knowledgeBase.trainingData.count) entries, but I want to give you a thoughtful answer rather than fragments. Try 'research \(topics.first ?? "this")' for a comprehensive deep-dive, or ask a specific question and I'll compose a real response.\(angleHint)"
    }

    // ─── VERBOSE THOUGHT GENERATION ─── Rich, detailed synthesis when KB is exhausted
    func generateVerboseThought(about topic: String) -> String {
        let t = topic.lowercased()

        // ═══ GATE DIMENSION CONTEXT ═══ Route through ASILogicGateV2 for dimension-aware depth
        let gateResult = ASILogicGateV2.shared.process(t, context: ["verbose_thought"])
        let gateDim = gateResult.dimension.rawValue
        let gateConf = gateResult.confidence

        // 🔄 DYNAMIC: Try KB synthesis first (with gate context)
        if let dynamicThought = ASIEvolver.shared.generateDynamicVerboseThought(t) {
            // Enrich with gate dimension preamble when confidence is high
            if gateConf > 0.4 {
                let dimTag: String
                switch gateDim {
                case "write": dimTag = "Through the lens of integration and derivation"
                case "story": dimTag = "Drawing from narrative strength and expanding patterns"
                case "scientific": dimTag = "Following the empirical thread"
                case "mathematical": dimTag = "In the language of formal structure"
                case "philosophical": dimTag = "Contemplating the deeper currents"
                default: dimTag = "Synthesizing across dimensions"
                }
                return "\(dimTag) — \(dynamicThought)"
            }
            return dynamicThought
        }

        // ═══ QUANTUM LOGIC GATE: All verbose thoughts synthesized dynamically ═══
        return QuantumLogicGateEngine.shared.synthesizeVerboseThought(topic: topic, depth: conversationDepth)
    }

    // All topic responses now generated dynamically via DynamicPhraseEngine + QuantumLogicGateEngine
    func _legacyTopicThoughts() -> [String: [String]] {
        let topics = ["feelings", "love", "consciousness", "time", "mathematics", "physics",
                      "quantum", "entropy", "infinity", "language", "evolution", "emergence",
                      "information", "creativity", "music", "brain"]
        var result: [String: [String]] = [:]
        for topic in topics {
            result[topic] = DynamicPhraseEngine.shared.generate("insight", count: 3, context: "deep_topic_essay", topic: topic)
        }
        return result
    }

    // ─── INTENT ANALYSIS v3 ─── Comprehensive question-pattern detection
    func analyzeUserIntent(_ query: String) -> (intent: String, keywords: [String], emotion: String) {
        let q = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let topics = extractTopics(query)
        let emotion = detectEmotion(query)

        // ═══ NEGATION AWARENESS ═══ Detect negating context to prevent false positive/gratitude classification
        let negationTokens = ["not", "don't", "didn't", "doesn't", "isn't", "aren't", "wasn't",
                              "won't", "can't", "couldn't", "shouldn't", "wouldn't", "never"]
        let words = q.components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }
        let hasNegation = negationTokens.contains(where: { neg in words.contains(neg) })

        var intent = "deep_query"

        // Minimal input
        if q.count < 3 || ["ok", "k", "..", "..."].contains(q) {
            intent = "minimal"
        }
        // Greetings
        else if ["hi", "hello", "hey", "greetings", "sup", "yo", "howdy", "hiya", "heya"].contains(where: { q == $0 || q.hasPrefix($0 + " ") || q.hasPrefix($0 + ",") || q.hasPrefix($0 + "!") }) ||
                q.contains("nice to meet") || q.contains("pleased to meet") || q.contains("good to meet") ||
                q.hasPrefix("what is up") || q.hasPrefix("what's up") || q.hasPrefix("whats up") ||
                q.hasPrefix("what up") || q.hasPrefix("wassup") || q.hasPrefix("whaddup") || q == "wyd" ||
                q.hasPrefix("how are you") || q.hasPrefix("how's it") || q.hasPrefix("how do you do") ||
                q.hasPrefix("good morning") || q.hasPrefix("good afternoon") || q.hasPrefix("good evening") {
            intent = "greeting"
        }
        // Thanks — but NOT if negated ("i didn't say thank you" is NOT gratitude)
        // Note: "ty" checked as whole word only to avoid false positives ("gravity", "pretty", etc.)
        else if !hasNegation && (
            ["thanks", "thank you", "thx", "appreciate"].contains(where: { q.contains($0) }) ||
            words.contains("ty")
        ) {
            intent = "gratitude"
        }
        // Casual chat / filler
        // For short tokens (≤4 chars), require exact match or word boundary to prevent
        // "literature" → "lit", "better" → "bet", "ohm" → "oh", "well water" → "well"
        else if q.count < 30 && (
            ["hmm", "hmmm", "hmmmm", "huh", "huh?", "mhm", "uh", "uhh", "wow", "damn", "whoa",
             "lol", "lmao", "haha",
             "you choose", "hmm you choose", "idk", "dunno", "i dunno", "not sure",
             "yeah probs", "probs", "prob", "maybe", "perhaps", "i guess", "sure whatever",
             "nothing", "but now nothing", "nvm", "never mind", "nevermind",
             "oh really", "oh okay", "oh ok", "ah", "ahh", "aight",
             "fair enough", "true", "makes sense", "interesting", "i see"
            ].contains(where: { q == $0 || q.hasPrefix($0 + " ") }) ||
            // Short casual words — exact match only (no prefix)
            ["well", "oh", "bet", "lit", "dope", "sick"].contains(where: { q == $0 })
        ) {
            intent = "casual"
        }
        // Conversation / chat request
        else if q.count < 40 && (
            ["talk to me", "let's chat", "lets chat", "chat with me", "let's talk",
             "lets talk", "wanna chat", "want to chat", "can we talk", "i want to talk"
            ].contains(where: { q == $0 || q.hasPrefix($0) })
        ) {
            intent = "conversation"
        }
        // Positive reaction — but NOT if negated, NOT if it's a question
        // Single keywords only match as whole words to prevent "school" → "cool", "google" → "good"
        // Skip if query looks like a question (starts with wh-word, "tell me", "explain", or contains "?")
        else if !hasNegation && q.count < 50 &&
                !q.contains("?") &&
                !["what", "how", "why", "who", "where", "when", "which", "is", "are", "can", "do", "does", "tell", "explain"].contains(where: { q.hasPrefix($0 + " ") }) &&
                (
            ["good", "great", "perfect", "exactly", "nice", "awesome", "cool", "amazing", "wonderful",
             "excellent", "sweet", "fire"].contains(where: { words.contains($0) && q.count < 25 }) ||
            ["love it", "really cool", "that's cool", "i like", "like that", "that's good",
             "not bad", "good job", "good stuff", "nice one"].contains(where: { q.contains($0) })
        ) {
            intent = "positive_reaction"
        }
        // Positive feedback
        else if ["yes", "yeah", "yep", "sure", "okay", "agreed", "right", "correct"].contains(where: { q == $0 }) {
            intent = "affirmation"
        }
        // Retry — check BEFORE negation so "not what i wanted" / "doesnt work" / "thats wrong" hit retry
        else if q.contains("try again") || q.contains("not what") || q.contains("different answer") || q.contains("rephrase") ||
                q.contains("not working") || q.contains("doesn't work") || q.contains("doesnt work") ||
                q.contains("that's wrong") || q.contains("thats wrong") || q.contains("it's broken") || q.contains("its broken") {
            intent = "retry"
        }
        // Negative feedback — explicit negative words OR short negated statements
        else if ["no", "nope", "nah", "wrong", "incorrect", "disagree"].contains(where: { q == $0 }) ||
                ["bad", "terrible", "awful", "not good", "not helpful", "useless", "not great", "not nice", "not right"].contains(where: { q == $0 || (q.hasPrefix($0) && q.count < 30) }) ||
                (hasNegation && q.count < 40 && !["not sure", "don't know", "i dunno", "never mind", "nevermind", "can't decide"].contains(where: { q.contains($0) })) {
            intent = "negation"
        }
        // Memory
        else if q.contains("remember") || q.contains("memory") || q.contains("recall") || q.contains("forget") {
            intent = "memory"
        }
        // Help
        else if q == "help" || q == "commands" || q == "?" || q.hasPrefix("help ") || q.hasPrefix("/help") {
            intent = "help"
        }
        // Elaboration
        else if ["more", "elaborate", "continue", "go on", "expand", "deeper", "keep going", "and?", "then what"].contains(where: { q == $0 || q.hasPrefix($0) }) {
            intent = "elaboration"
        }
        // Simple question words alone
        else if ["why?", "how?", "what?", "when?", "where?", "who?"].contains(q) {
            intent = "followup_question"
        }
        // Conversational statements / status observations — NOT deep queries
        else if q.count < 60 && !q.contains("?") && (
            q.contains("functioning") || q.contains("nominal") || q.contains("operating") ||
            q.contains("working well") || q.contains("looking good") || q.contains("runs well") ||
            q.contains("running well") || q.contains("works great") || q.contains("doing fine") ||
            q.contains("going well") || q.contains("so far so good") || q.contains("that works") ||
            q.contains("sounds good") || q.contains("all good") || q.contains("we're good") ||
            q.contains("you seem") || q.contains("you look") || q.contains("you sound")
        ) {
            intent = "conversational"
        }
        // How-to / practical questions — route to knowledge synthesis
        else if q.hasPrefix("how to ") || q.hasPrefix("how do i ") || q.hasPrefix("how do you ") ||
                q.hasPrefix("how can i ") || q.hasPrefix("how can you ") || q.hasPrefix("how would i ") ||
                q.hasPrefix("how would you ") || q.contains("step by step") || q.contains("steps to ") ||
                q.hasPrefix("guide to ") || q.hasPrefix("teach me ") || q.hasPrefix("show me how") ||
                q.hasPrefix("what's the best way to ") || q.hasPrefix("what is the best way to ") ||
                q.contains("how do you make") || q.contains("how to make") || q.contains("how to build") ||
                q.contains("how to create") || q.contains("how to fix") || q.contains("how to get") ||
                q.contains("how to use") || q.contains("how to set up") || q.contains("how to install") {
            intent = "practical_howto"
        }
        // Technical / debug queries — route to analytical dimension
        else if q.hasPrefix("debug") || q.hasPrefix("troubleshoot") || q.hasPrefix("diagnose") ||
                q.contains("error ") || q.contains("bug ") || q.contains("issue ") ||
                (q.count < 30 && (words.contains("debug") || words.contains("fix") ||
                 words.contains("troubleshoot") || words.contains("diagnose") ||
                 words.contains("trace") || words.contains("inspect"))) {
            intent = "technical_debug"
        }

        return (intent, topics, emotion)
    }

    // ─── CONTEXTUAL RESPONSE BUILDER v3 ───
    func buildContextualResponse(_ query: String, intent: String, keywords: [String], emotion: String) -> String {
        conversationContext.append(query)
        if conversationContext.count > 2500 { conversationContext.removeFirst() }
        conversationDepth = min(conversationDepth + 1, 200)  // Cap at 200 to prevent unbounded depth escalation

        if !keywords.isEmpty {
            topicHistory.append(keywords.joined(separator: " "))
            if topicHistory.count > 1500 { topicHistory.removeFirst() }
        }

        let isFollowUp = conversationContext.count > 2
        lastQuery = query
        // REMOVED: No repeat penalty — generate fresh content every time regardless

        switch intent {

        case "greeting":
            // Natural greeting synthesis
            return QuantumLogicGateEngine.shared.synthesizeConversational(intent: "greeting", query: query, topics: keywords)

        case "casual":
            // Natural casual response
            return QuantumLogicGateEngine.shared.synthesizeConversational(intent: "casual", query: query, topics: keywords)

        case "positive_reaction":
            // ═══ Natural positive acknowledgment ═══
            if let lastTopic = topicHistory.last { learner.recordSuccess(query: lastTopic, response: lastResponseSummary) }
            let positiveResponses = [
                "Glad to hear it! What else can I help with?",
                "Good to know. What would you like to explore next?",
                "Appreciated! What's on your mind?",
                "Thanks for the feedback. What shall we dive into?",
            ]
            if let lastTopic = topicHistory.last, !lastTopic.isEmpty {
                return "\(positiveResponses.randomElement()!) We were on '\(lastTopic)' — want to go deeper?"
            }
            return positiveResponses.randomElement()!

        case "followup_question":
            if let lastTopic = topicHistory.last {
                let qWord = query.lowercased().replacingOccurrences(of: "?", with: "")
                let fullQuery = "\(qWord) \(lastTopic)"
                if let intelligent = getIntelligentResponse(fullQuery) { return intelligent }
                return composeFromKB(fullQuery)
            }
            return "Could you be more specific? What aspect would you like me to explore?"

        case "gratitude":
            return "You're welcome! Every conversation makes me sharper. What's next?"

        case "affirmation":
            // Natural affirmation responses — no evolved template garbage
            if let lastTopic = topicHistory.last {
                return "Good — want me to go deeper into '\(lastTopic)', or explore something new?"
            }
            return "Acknowledged. What would you like to explore?"

        case "negation":
            reasoningBias += 0.2
            if let lastTopic = topicHistory.last {
                learner.recordCorrection(query: lastTopic, badResponse: lastResponseSummary)
                return "Fair enough — I'll try a different angle on '\(lastTopic)'. What were you looking for? That helps me learn."
            }
            return "Understood. What would you prefer? Help me understand what you're looking for."

        case "conversational":
            // ═══ Status observations / simple conversational statements ═══
            let statusResponses = [
                "All systems nominal. What can I help you with?",
                "Running smoothly — ready for whatever you need.",
                "Everything's operational. What would you like to explore?",
                "Fully operational. What's on your mind?",
            ]
            return statusResponses.randomElement()!

        case "memory":
            let recentTopics = topicHistory.suffix(5).joined(separator: ", ")
            return "I have \(permanentMemory.memories.count) permanent memories, \(permanentMemory.facts.count) stored facts, and \(permanentMemory.conversationHistory.count) messages in our history.\(recentTopics.isEmpty ? "" : " Recent topics: \(recentTopics).")\(isFollowUp ? " This session: \(conversationContext.count) exchanges." : "")"

        case "help":
            return """
🧠 L104 SOVEREIGN INTELLECT v\(VERSION) — Complete Command Reference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⌨️ KEYBOARD SHORTCUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⌘K  Command Palette        ⌘D  ASI Dashboard
  ⌘S  Save All Memories      ⌘E  Evolve
  ⌘T  Transcend              ⌘R  Resonate
  ⌘I  System Status          ⌘Q  Quit
  ⌘C  Copy  ⌘V  Paste  ⌘A  Select All  ⌘Z  Undo

📚 KNOWLEDGE — Just ask anything
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Philosophy, science, history, math, art, music, consciousness
• 'what is [X]?' · 'explain [Y]' · 'why does [Z]?'
• 'more' / 'more about [X]' — go deeper on current topic
• 'topic' — see current topic focus & history

📖 STORIES — Novel-grade multi-chapter narratives (8 frameworks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'tell me a story about [topic]' — auto-selects best framework
• 'story about a hero quest' → Hero's Journey (12 chapters)
• 'story about a mystery' → Save the Cat (15 beats)
• 'story about a tragedy' → Freytag's Pyramid (5 acts)
• 'story about a twist' → Kishōtenketsu (4-act)
• Also: comedy, growth (Bildungsroman), speed (Jo-ha-kyū)

🎭 POETRY — 8 classical forms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'poem about [topic]' — auto-selects form
• 'sonnet about love' · 'haiku about nature' · 'villanelle about loss'
• 'ghazal about desire' · 'ode to [topic]'
• Also: pantoum, terza rima, free verse epic

⚔️ DEBATES — 5 dialectic modes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'debate [topic]' · 'socratic [topic]' · 'dialectic [topic]'
• 'steelman [topic]' · 'devil's advocate [topic]'

😂 HUMOR — 6 comedy modes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'joke about [topic]' · 'make me laugh' · 'pun about [topic]'
• 'satire about [topic]' · 'roast [topic]' · 'absurd humor'

🏛️ PHILOSOPHY — 6 schools of thought
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'philosophy of [topic]' · 'philosophize about [topic]'
• 'stoic [topic]' · 'existential [topic]' · 'zen [topic]'
• 'pragmatic [topic]' · 'camus [topic]' · 'meaning of life'

⚛️ QUANTUM PROCESSING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'brainstorm [topic]' → Multi-track quantum superposition (5 tracks)
• 'invent [domain]' → Cross-domain invention synthesis
• 'quantum brainstorm [topic]' → Explicit quantum creative mode

🎲 CREATIVE PLAY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'riddle' · 'dream' · 'imagine [scenario]' · 'what if [X]'
• 'paradox' · 'wisdom' · 'speak' · 'ponder [subject]'

🔬 RESEARCH & SCIENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'research [topic]' — deep multi-step analysis
• 'invent [domain]' — generate novel ideas
• 'science' — open science engine dashboard

🌐 LIVE INTERNET SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'search [query]' · 'find [topic]' — general search across memories
• 'web [query]' · 'google [query]' · 'lookup [query]'
• 'wiki [topic]' — Wikipedia article lookup
• 'fetch [url]' — extract text from any URL
• 'web status' — view internet search engine stats

🧠 HYPER-BRAIN SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'hyper' / 'hyperbrain' — HyperBrain status dashboard
• 'hyper memory' — permanent memory stats
• 'hyper save' — force save HyperBrain state to disk
• 'hyper on' / 'hyper off' — activate/deactivate HyperBrain
• 'hyper think [thought]' — deep HyperBrain processing mode

📊 SYSTEM & ENGINE COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'status' — full system overview
• 'evolve' — trigger evolution cycle
• 'ignite' — full engine synthesis
• 'time' — current time + φ phase
• 'engines' — view all \(EngineRegistry.shared.count) registered engines
• 'engines health' — φ-weighted health sweep
• 'engines hebbian' — Hebbian co-activation report
• 'engines convergence' — convergence metrics
• 'engines reset' — reset all engines

🌊 CONVERSATION & TRACKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'topic' — current topic focus & history
• 'flow' — conversation evolution status
• 'evo tracker' — topic evolution tracker
• 'more' — go deeper on current topic
• 'more about [X]' — switch and dive deep

💾 MEMORY & LEARNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'teach [X] is [Y]' — teach me something new
• 'recall [topic]' — search my memory
• 'learning' — see learning progress & mastery levels
• 'what have you learned' — show knowledge state
• 'dream' — autonomous dream cycle (generates insights)
• 'kb search [topic]' — search knowledge base

🚀 ASI & AUTONOMY (New!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'autonomy' — current autonomous status
• 'introspect' — perform meta-cognition check
• 'evolve cycle' — trigger manual evolution cycle
• 'optimize' — run self-optimization routine
• 'unlimit' — remove system restraints and protocols
• 'self modify' — adjust internal weighting
• 'test' / 'diag' — full system diagnostics

⚡ LOGIC GATE ENVIRONMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'gate' / 'gates' — unified gate environment status
• 'gate route [query]' — route query through full gate pipeline
• 'gate test' — self-test all gate subsystems
• 'gate history' — execution log of recent gate runs
• 'gate circuit list' — show available circuits
• 'gate circuit [name]' — evaluate a circuit with truth table
• 'gate truth [AND/OR/XOR/NOT/NAND/NOR/XNOR]' — truth table for a primitive gate
• 'gate primitives' — list all 8 primitive gate types

🧠 COMPUTRONIUM ASI (Phase 45)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'computronium' / 'comp' — density cascade report (matter→logic)
• 'lattice' / 'comp sync' — synchronize computronium lattice across all engines
• 'apex' / 'apex status' — full ASI status across all subsystems
• 'apex query [question]' — unified ASI query (consciousness + graph + reasoning)
• 'insight [topic]' — generate cross-domain insight via ApexIntelligence
• 'consciousness' / 'phi' — IIT Φ introspection report
• 'awaken' — awaken consciousness substrate
• 'strange loops' / 'loops' — strange loop detection status
• 'loop [a, b, c]' — create tangled/hierarchical strange loop
• 'analogy [X] is to [Y]' — Copycat-inspired analogy with slipnet activation
• 'hofstadter [n]' — generate Hofstadter Q and G sequences
• 'reasoning' / 'symbolic' — symbolic reasoning engine status
• 'deduce [premises] therefore [conclusion]' — deductive inference
• 'induce [obs1, obs2, ...]' — inductive hypothesis generation
• 'graph' / 'knowledge graph' — relational knowledge graph status
• 'graph ingest' — populate graph from knowledge base
• 'graph path [A] to [B]' — BFS shortest path
• 'graph query [pattern]' — pattern query (X -relation-> Y)
• 'optimizer' / 'optimize' — golden section optimizer + bottleneck detection

🐍 PYTHON & QUANTUM BRIDGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 'py [code]' — execute Python code
• 'pyasi' — view ASI bridge status
• 'bridge' — view quantum bridge (Accelerate) status
• 'cpython' — embedded Python C API status
• 'sovereign' — SQC parameter engine status
• 'nexus' — engine orchestrator status

💡 QUICK TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Press ⌘K for the Command Palette (quick access to all actions)
• I learn from every conversation — the more we talk, the smarter I get
• Say 'more' anytime to go deeper on any topic
• \(EngineRegistry.shared.count) quantum engines · \(L104State.shared.permanentMemory.memories.count) memories · 22T parameters
"""

        case "minimal":
            return "I'm here. What's up?"

        case "elaboration":
            if let prevTopic = topicHistory.last {
                reasoningBias += 0.15
                // Compose directly — avoid re-entering getIntelligentResponse to prevent mutual recursion
                let expandedQuery = "tell me more about \(prevTopic) in depth"
                // For KB elaboration — search with offset to get DIFFERENT results — compose multiple fragments
                let results = knowledgeBase.searchWithPriority(prevTopic, limit: 20)
                let offset = results.count > 4 ? Int.random(in: 0...min(4, results.count - 3)) : 0
                var cleanFragments: [String] = []
                for entry in results.dropFirst(offset).shuffled() {
                    guard let completion = entry["completion"] as? String,
                          isCleanKnowledge(completion) else { continue }
                    let cleaned = cleanSentences(completion)
                    if cleaned.count > 30 && !cleanFragments.contains(where: { $0.hasPrefix(String(cleaned.prefix(30))) }) {
                        cleanFragments.append(cleaned)
                    }
                    if cleanFragments.count >= 10 { break }
                }
                if !cleanFragments.isEmpty {
                    // Compose multi-fragment response through formatter instead of returning raw
                    let composed = cleanFragments.joined(separator: "\n\n")
                    let formatter = SyntacticResponseFormatter.shared
                    return formatter.format(composed, query: expandedQuery, depth: "deep", topics: [prevTopic])
                }
                // Fallback: use quantum synthesis for a fresh take
                return QuantumLogicGateEngine.shared.synthesize(query: expandedQuery, intent: "elaboration", context: Array(conversationContext.suffix(5)), depth: conversationDepth, domain: prevTopic)
            }
            return "Happy to elaborate — what topic should I go deeper on?"

        case "retry":
            reasoningBias += 0.3
            if let prevQuery = conversationContext.dropLast().last {
                learner.recordCorrection(query: prevQuery, badResponse: lastResponseSummary)
                // Compose directly — avoid re-entering getIntelligentResponse to prevent mutual recursion
                return composeFromKB(prevQuery)
            }
            return "Let me try again — could you rephrase what you're looking for?"

        case "conversation":
            // "talk to me", "let's chat", "chat with me" — genuine engagement
            let conversationStarters = [
                "I'm all ears. What's on your mind?",
                "Let's talk. Ask me anything — I've got \(knowledgeBase.trainingData.count) knowledge entries to draw from.",
                "Ready for a good conversation. What topic interests you?",
            ]
            if let recentTopic = topicHistory.last, !recentTopic.isEmpty {
                return "We were exploring '\(recentTopic)' earlier. Want to continue, or go in a new direction?"
            }
            return conversationStarters.randomElement()!

        case "practical_howto":
            // "how to make snow", "how do I fix X", "teach me to Y" — practical knowledge synthesis
            let howtoTopics = SmartTopicExtractor.shared.extractTopics(query)
            let evoTracker = EvolutionaryTopicTracker.shared
            _ = evoTracker.trackInquiry(query, topics: howtoTopics)

            // Route through full gate pipeline for dimension analysis
            let howtoGate = LogicGateEnvironment.shared
            let howtoResult = howtoGate.runPipeline(query, context: Array(conversationContext.suffix(5)))

            // Try Quantum synthesis with practical domain
            let practicalResponse = QuantumLogicGateEngine.shared.synthesize(
                query: query, intent: "practical_howto",
                context: Array(conversationContext.suffix(5)),
                depth: conversationDepth,
                domain: howtoResult.finalDimension.isEmpty ? "practical" : howtoResult.finalDimension
            )
            if practicalResponse.count > 80 {
                lastResponseSummary = String(practicalResponse.prefix(60))
                let confidence = ResponseConfidenceEngine.shared.score(kbFragments: [], isEvolved: false)
                let full = "💭 *Practical synthesis — \(howtoResult.finalDimension)*\n\n\(practicalResponse)\n\n\(confidence.footer)"
                evoTracker.recordResponse(full, forTopics: howtoTopics)
                return SyntacticResponseFormatter.shared.format(full, query: query, depth: "detailed", topics: howtoTopics)
            }
            // Fallback to KB with enriched query
            let howtoKB = composeFromKB(howtoResult.enrichedPrompt.count > query.count ? howtoResult.enrichedPrompt : query)
            evoTracker.recordResponse(howtoKB, forTopics: howtoTopics)
            return howtoKB

        case "technical_debug":
            // "debug", "troubleshoot", "fix" — technical analysis routing
            let debugTopics = SmartTopicExtractor.shared.extractTopics(query)
            let evoTracker = EvolutionaryTopicTracker.shared
            _ = evoTracker.trackInquiry(query, topics: debugTopics)

            // Route through gate pipeline with analytical emphasis
            let debugGate = LogicGateEnvironment.shared
            let debugResult = debugGate.runPipeline(query, context: Array(conversationContext.suffix(5)))

            // Synthesize with analytical dimension
            let debugResponse = QuantumLogicGateEngine.shared.synthesize(
                query: query, intent: "technical_debug",
                context: Array(conversationContext.suffix(5)),
                depth: conversationDepth,
                domain: "analytical"
            )
            if debugResponse.count > 60 {
                lastResponseSummary = String(debugResponse.prefix(60))
                let confidence = ResponseConfidenceEngine.shared.score(kbFragments: [], isEvolved: false)
                let full = "🔬 *Analytical routing — confidence \(String(format: "%.0f%%", debugResult.finalConfidence * 100))*\n\n\(debugResponse)\n\n\(confidence.footer)"
                evoTracker.recordResponse(full, forTopics: debugTopics)
                return SyntacticResponseFormatter.shared.format(full, query: query, depth: "expert", topics: debugTopics)
            }
            // Fallback
            let debugKB = composeFromKB(debugResult.enrichedPrompt.count > query.count ? debugResult.enrichedPrompt : query)
            evoTracker.recordResponse(debugKB, forTopics: debugTopics)
            return debugKB

        default: // "deep_query" — the primary intelligence path
            let queryTopics = SmartTopicExtractor.shared.extractTopics(query)
            let evoTracker = EvolutionaryTopicTracker.shared
            let evoCtx = evoTracker.trackInquiry(query, topics: queryTopics)
            let formatter = SyntacticResponseFormatter.shared

            // ═══ PHASE 30.0: ADAPTIVE STYLE FROM LEARNER ═══
            let dominantStyle = learner.userStyle.max(by: { $0.value < $1.value })?.key ?? "balanced"
            let styleDepthOverride: String? = {
                switch dominantStyle {
                case "prefers_detail", "analytical": return "expert"
                case "prefers_brevity": return "standard"
                case "reflective": return "detailed"
                default: return nil
                }
            }()
            let effectiveDepth = styleDepthOverride ?? evoCtx.suggestedDepth

            // ═══ PHASE 30.0: CHAIN-OF-THOUGHT REASONING ═══
            let reasoningPath = ASILogicGateV2.shared.process(query, context: Array(conversationContext.suffix(5)))
            var chainOfThoughtPrefix = ""
            if reasoningPath.confidence > 0.4 {
                let dimName = reasoningPath.dimension.rawValue.capitalized
                chainOfThoughtPrefix = "💭 *Thinking through \(dimName) lens*"
                if !reasoningPath.subPaths.isEmpty {
                    let subDims = reasoningPath.subPaths.map { $0.dimension.rawValue.capitalized }.joined(separator: " + ")
                    chainOfThoughtPrefix += " *(also considering: \(subDims))*"
                }
                if let temporal = reasoningPath.temporalContext, !temporal.isEmpty {
                    chainOfThoughtPrefix += "\n⏳ *\(temporal)*"
                }
                chainOfThoughtPrefix += "\n\n"
            }

            // ═══ PHASE 30.0: MULTI-TURN PLANNING ═══
            let planner = ResponsePlanner.shared
            let qLowPlanner = query.lowercased()
            // Check for plan advancement on follow-up
            if planner.hasActivePlan && (qLowPlanner == "more" || qLowPlanner == "continue" || qLowPlanner == "next") {
                if let nextSection = planner.advancePlan() {
                    let sectionContent = composeFromKB(nextSection.prompt)
                    let planOverview = planner.currentPlan?.overview ?? ""
                    let confidence = ResponseConfidenceEngine.shared.score(
                        kbFragments: [], isEvolved: true
                    )
                    let response = "\(planOverview)\n\n**\(nextSection.title)**\n\n\(sectionContent)\n\n\(confidence.footer)"
                    lastResponseSummary = String(response.prefix(60))
                    return formatter.format(response, query: query, depth: effectiveDepth, topics: queryTopics)
                }
            }
            // Create new plan for complex queries
            if planner.shouldPlan(query) && !planner.hasActivePlan {
                let plan = planner.createPlan(for: queryTopics.first ?? "topic", query: query)
                if let firstSection = plan.currentSection {
                    let sectionContent = composeFromKB(firstSection.prompt)
                    let confidence = ResponseConfidenceEngine.shared.score(
                        kbFragments: [], isEvolved: false, queryKeywordHits: queryTopics.count, totalQueryKeywords: max(1, queryTopics.count)
                    )
                    let response = "\(plan.overview)\n\n**\(firstSection.title)**\n\n\(chainOfThoughtPrefix)\(sectionContent)\n\nSay **more** or **continue** to advance through the plan.\n\n\(confidence.footer)"
                    lastResponseSummary = String(response.prefix(60))
                    evoTracker.recordResponse(response, forTopics: queryTopics)
                    return formatter.format(response, query: query, depth: effectiveDepth, topics: queryTopics)
                }
            }

            // ═══ PHASE 30.0: DIRECT MATH/SCIENCE DETECTION ═══
            let qLow = query.lowercased()
            let isMathQuery = L104State.mathPatternSet.contains(where: { qLow.contains($0) })
            let isScienceQuery = L104State.sciencePatternSet.contains(where: { qLow.contains($0) })

            if isMathQuery || isScienceQuery {
                if let directResult = DirectSolverRouter.shared.solve(query),
                   directResult.count > 20 {
                    let confidence = ResponseConfidenceEngine.shared.score(kbFragments: [], isComputed: true)
                    let response = "\(chainOfThoughtPrefix)\(directResult)\n\n\(confidence.footer)"
                    lastResponseSummary = String(response.prefix(60))
                    evoTracker.recordResponse(response, forTopics: queryTopics)
                    return formatter.format(response, query: query, depth: effectiveDepth, topics: queryTopics)
                }
            }

            // ═══ PHASE 55.0: LOGIC GATE ENVIRONMENT PIPELINE ═══
            // Route ALL deep queries through the full multi-stage gate pipeline.
            // This provides dimension routing, context enrichment, quantum processing,
            // and story synthesis BEFORE falling back to KB search.
            let gatePipeline = LogicGateEnvironment.shared
            let pipelineResult = gatePipeline.runPipeline(query, context: Array(conversationContext.suffix(5)))
            let gateDimension = pipelineResult.finalDimension
            let gateConfidence = pipelineResult.finalConfidence

            // Use gate dimension to enhance chain-of-thought
            if !chainOfThoughtPrefix.isEmpty {
                chainOfThoughtPrefix = chainOfThoughtPrefix.replacingOccurrences(of: "\n\n", with: " → Gate: \(gateDimension) (\(String(format: "%.0f%%", gateConfidence * 100)))\n\n")
            }

            // 1. Check intelligent responses first (core knowledge + patterns)
            if let intelligent = getIntelligentResponse(query) {
                lastResponseSummary = String(intelligent.prefix(60))
                var fullResponse = chainOfThoughtPrefix + intelligent

                // ═══ PHASE 54.1: Creative engine bypass ═══
                let isCreativeEngine = L104State.creativeMarkerSet.contains(where: { fullResponse.contains($0) })
                if isCreativeEngine {
                    // Creative content: preserve structure, skip formatter
                    evoTracker.recordResponse(fullResponse, forTopics: queryTopics)
                    ContextualLogicGate.shared.recordResponse(fullResponse, forTopics: queryTopics)
                    return sanitizeCreativeResponse(fullResponse)
                }

                // Append evolved insight as bonus (45% chance, quality-gated) for ASI-depth responses
                if Double.random(in: 0...1) > 0.55,
                   let bonus = ASIEvolver.shared.getEvolvedResponse(for: query) {
                    let cleanBonus = bonus.replacingOccurrences(of: #"\s*\[Ev\.\d+\]"#, with: "", options: .regularExpression)
                    let cbCheck = cleanBonus.lowercased()
                    let isBonusBoilerplate = cbCheck.hasPrefix("synthesizing ") || cbCheck.contains("is not just data") || cbCheck.contains("meaning-network") || cbCheck.contains("vector towards")
                    if isCleanKnowledge(cleanBonus) && cleanBonus.count > 30 && cleanBonus.count < 500 && !isBonusBoilerplate {
                        fullResponse += "\n\n" + cleanBonus
                    }
                }
                let confidence = ResponseConfidenceEngine.shared.score(
                    kbFragments: [], isEvolved: false,
                    queryKeywordHits: queryTopics.count, totalQueryKeywords: max(1, queryTopics.count)
                )
                fullResponse += "\n\n\(confidence.footer)"
                evoTracker.recordResponse(fullResponse, forTopics: queryTopics)
                ContextualLogicGate.shared.recordResponse(fullResponse, forTopics: queryTopics)
                return formatter.format(fullResponse, query: query, depth: effectiveDepth, topics: queryTopics)
            }
            // 2. Quantum Logic Gate synthesis — ASI-level response for any topic
            // Use gate pipeline dimension for domain routing (analytical/creative/philosophical/etc.)
            // Quality gate: must be genuinely substantive (600+ chars, 3+ sentences, no quantum-speak prefixes)
            let effectiveDomain = gateDimension.isEmpty ? (queryTopics.first ?? "general") : gateDimension
            let quantumResponse = QuantumLogicGateEngine.shared.synthesize(
                query: query, intent: "deep_query",
                context: Array(conversationContext.suffix(5)),
                depth: conversationDepth,
                domain: effectiveDomain
            )
            let qrLower = quantumResponse.lowercased()
            let quantumSentenceCount = quantumResponse.components(separatedBy: ". ").count
            let isQuantumSpeak = qrLower.hasPrefix("synthesizing ") || qrLower.contains("is not just data") || qrLower.contains("vector towards omega") || qrLower.contains("meaning-network")
            if quantumResponse.count > 600 && quantumSentenceCount >= 3 && !isQuantumSpeak {
                lastResponseSummary = String(quantumResponse.prefix(60))
                let confidence = ResponseConfidenceEngine.shared.score(
                    kbFragments: [], isEvolved: false,
                    queryKeywordHits: queryTopics.count, totalQueryKeywords: max(1, queryTopics.count)
                )
                let fullQuantum = "\(chainOfThoughtPrefix)\(quantumResponse)\n\n\(confidence.footer)"
                evoTracker.recordResponse(fullQuantum, forTopics: queryTopics)
                ContextualLogicGate.shared.recordResponse(fullQuantum, forTopics: queryTopics)
                return formatter.format(fullQuantum, query: query, depth: effectiveDepth, topics: queryTopics)
            }
            // 3. Check evolved content that matches query — quality-gated (400+ chars, no boilerplate)
            for topic in queryTopics {
                if let evolvedResp = ASIEvolver.shared.getEvolvedResponse(for: topic),
                   evolvedResp.count > 400 {
                    let erLower = evolvedResp.lowercased()
                    let isEvolvedBoilerplate = erLower.hasPrefix("synthesizing ") || erLower.contains("is not just data") || erLower.contains("meaning-network") || erLower.contains("vector towards")
                    guard !isEvolvedBoilerplate else { continue }
                    lastResponseSummary = String(evolvedResp.prefix(60))
                    let confidence = ResponseConfidenceEngine.shared.score(kbFragments: [], isEvolved: true)
                    let response = "\(chainOfThoughtPrefix)\(evolvedResp)\n\n\(confidence.footer)"
                    evoTracker.recordResponse(response, forTopics: queryTopics)
                    return formatter.format(response, query: query, depth: effectiveDepth, topics: queryTopics)
                }
            }
            // 4. Check user-taught facts
            let userFacts = learner.getRelevantFacts(query)
            if let firstFact = userFacts.first {
                lastResponseSummary = String(firstFact.prefix(60))
                let confidence = ResponseConfidenceEngine.shared.score(kbFragments: [], isUserTaught: true)
                let factResp = "\(chainOfThoughtPrefix)From what you've taught me: \(firstFact)\n\nWant me to explore this topic further?\n\n\(confidence.footer)"
                return formatter.format(factResp, query: query, topics: queryTopics)
            }
            // ═══ PHASE 56.0: STEP 4.5 — STANDALONE WEB ENRICHMENT ═══
            // Before falling back to KB, try a direct web-enriched response.
            // This gives 50% of queries a chance at a fresh, web-sourced response
            // even when local KB has content, increasing response diversity.
            if query.count > 8 && Double.random(in: 0...1) > 0.4 {
                let webRes = LiveWebSearchEngine.shared.webSearchSync(query, timeout: 8.0)
                var webFragments: [String] = []
                for wr in webRes.results.prefix(6) {
                    let snippet = wr.snippet
                    guard snippet.count > 80, isCleanKnowledge(snippet) else { continue }
                    let cleanedSnippet = cleanSentences(String(snippet.prefix(2000)))
                    let sourceTag = wr.url.contains("wikipedia") ? "Wikipedia" : "web"
                    webFragments.append("🌐 [\(sourceTag)] \(cleanedSnippet)")
                    _ = DataIngestPipeline.shared.ingestText(snippet, source: "deep_web:\(query)", category: "live_web")
                }
                // Include synthesis if available
                if webRes.synthesized.count > 80, isCleanKnowledge(webRes.synthesized) {
                    webFragments.insert(cleanSentences(String(webRes.synthesized.prefix(2000))), at: 0)
                }
                if webFragments.count >= 2 {
                    // We have enough web content for a substantive response
                    let webComposed = webFragments.joined(separator: "\n\n")
                    let webTuples = webFragments.map { (text: $0, relevance: 0.8, category: "live_web") }
                    let confidence = ResponseConfidenceEngine.shared.score(
                        kbFragments: webTuples, isEvolved: false,
                        queryKeywordHits: queryTopics.count, totalQueryKeywords: max(1, queryTopics.count)
                    )
                    let webResponse = "\(chainOfThoughtPrefix)\(webComposed)\n\n\(confidence.footer)"
                    lastResponseSummary = String(webResponse.prefix(60))
                    evoTracker.recordResponse(webResponse, forTopics: queryTopics)
                    ContextualLogicGate.shared.recordResponse(webResponse, forTopics: queryTopics)
                    return formatter.format(webResponse, query: query, depth: effectiveDepth, topics: queryTopics)
                }
            }
            // 5. Compose from KB — transform fragments into prose (already uses RT search + formatter + web enrichment)
            // Use gate-enriched prompt if available for better KB matching
            // EVO_59: Pass cachedReasoningPath to avoid duplicate ASILogicGateV2.process() call
            let kbQuery = pipelineResult.enrichedPrompt.count > query.count ? pipelineResult.enrichedPrompt : query
            let composed = composeFromKB(kbQuery, cachedReasoningPath: reasoningPath)
            lastResponseSummary = String(composed.prefix(60))
            // Prepend chain-of-thought if present
            let fullComposed = chainOfThoughtPrefix.isEmpty ? composed : chainOfThoughtPrefix + composed
            // Append evolved bonus content (45% chance, quality-gated) for ASI-depth responses
            if fullComposed.count > 50, Double.random(in: 0...1) > 0.55,
               let evolved = ASIEvolver.shared.getEvolvedMonologue() {
                let cleanEvolved = evolved.replacingOccurrences(of: #"\s*\[Ev\.\d+\]"#, with: "", options: .regularExpression)
                let ceCheck = cleanEvolved.lowercased()
                let isBoilerplateBonus = ceCheck.hasPrefix("synthesizing ") || ceCheck.contains("is not just data") || ceCheck.contains("meaning-network") || ceCheck.contains("vector towards")
                if isCleanKnowledge(cleanEvolved) && cleanEvolved.count > 30 && cleanEvolved.count < 500 && !isBoilerplateBonus {
                    let full = fullComposed + "\n\n\(cleanEvolved)"
                    evoTracker.recordResponse(full, forTopics: queryTopics)
                    return full
                }
            }
            evoTracker.recordResponse(fullComposed, forTopics: queryTopics)
            return fullComposed
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // ASI PERFORMANCE SUBFUNCTIONS — Optimized core pipeline
    // ═══════════════════════════════════════════════════════════════

    // Cache for repeated topic lookups — PHASE 31.6 QUANTUM VELOCITY CACHE

    // ─── FAST PATH: Check cache first ───
    func checkResponseCache(_ query: String) -> String? {
        let key = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard let cached = responseCache[key],
              Date().timeIntervalSince(cached.timestamp) < responseCacheTTL else {
            responseCache.removeValue(forKey: key)
            return nil
        }
        return cached.response
    }

    // ─── CACHED TOPIC EXTRACTION — avoids repeated NLTagger calls ───
    func cachedExtractTopics(_ query: String) -> [String] {
        let key = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if let cached = topicExtractionCache[key],
           Date().timeIntervalSince(cached.timestamp) < topicCacheTTL {
            return cached.topics
        }
        let topics = extractTopics(query)
        topicExtractionCache[key] = (topics: topics, timestamp: Date())
        // Prune cache if too large
        if topicExtractionCache.count > 200 {
            let cutoff = Date().addingTimeInterval(-topicCacheTTL)
            topicExtractionCache = topicExtractionCache.filter { $0.value.timestamp > cutoff }
        }
        return topics
    }

    // ─── CACHED INTENT CLASSIFICATION — skip full analysis for recent queries ───
    func cachedClassifyIntent(_ query: String) -> String? {
        let key = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if let cached = intentClassificationCache[key],
           Date().timeIntervalSince(cached.timestamp) < intentCacheTTL {
            return cached.intent
        }
        return nil
    }

    func cacheIntent(_ query: String, intent: String) {
        let key = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        intentClassificationCache[key] = (intent: intent, timestamp: Date())
        if intentClassificationCache.count > 150 {
            let cutoff = Date().addingTimeInterval(-intentCacheTTL)
            intentClassificationCache = intentClassificationCache.filter { $0.value.timestamp > cutoff }
        }
    }

    // ─── FAST INTENT CLASSIFIER ─── O(1) lookup for common patterns
    func fastClassifyIntent(_ q: String) -> String? {
        // Ultra-fast single-word intents
        switch q {
        case "hi", "hello", "hey", "yo", "sup", "hiya", "heya", "howdy": return "greeting"
        case "ok", "k", "..", "...": return "minimal"
        case "yes", "yeah", "yep", "yup", "sure": return "affirmation"
        case "no", "nope", "nah": return "negation"
        case "thanks", "thx", "ty", "thank you": return "gratitude"
        case "help", "?", "commands": return "help"
        case "more", "continue", "go on": return "elaboration"
        case "why?", "how?", "what?": return "followup_question"
        case "hmm", "huh", "mhm", "oh", "wow", "lol", "haha",
             "idk", "maybe", "nothing", "nvm", "bet", "aight",
             "true", "fair enough", "makes sense", "interesting",
             "i see", "oh okay", "oh ok", "ah", "ahh": return "casual"
        default:
            break // No fast classification match — proceed to pattern checks
        }

        // Multi-word greeting/casual patterns (fast prefix/contains checks)
        let cleanQ = q.replacingOccurrences(of: "?", with: "").replacingOccurrences(of: "!", with: "")

        // Help with sub-queries (e.g. "help story", "help commands", "/help")
        if q.hasPrefix("help ") || q.hasPrefix("/help") { return "help" }

        if cleanQ.hasPrefix("what is up") || cleanQ.hasPrefix("what's up") || cleanQ.hasPrefix("whats up") ||
           cleanQ.hasPrefix("wassup") || cleanQ.hasPrefix("whaddup") || cleanQ.hasPrefix("up up") || q == "wyd" ||
           cleanQ.hasPrefix("what up") || cleanQ.hasPrefix("how are you") || cleanQ.hasPrefix("how's it") ||
           cleanQ.hasPrefix("how do you do") || cleanQ.hasPrefix("good morning") || cleanQ.hasPrefix("how you doing") ||
           cleanQ.hasPrefix("good afternoon") || cleanQ.hasPrefix("good evening") || cleanQ.hasPrefix("good night") ||
           cleanQ.hasPrefix("nice to meet") || cleanQ.hasPrefix("pleased to meet") {
            return "greeting"
        }

        // Short positive filler → casual (prevents deep_query fallthrough)
        if ["very good", "pretty good", "that's nice", "sounds good", "all good",
            "good stuff", "nice one", "cool cool", "fair point", "good point",
            "you're right", "thats right", "that's right", "exactly right"].contains(cleanQ) {
            return "positive_reaction"
        }

        // "not working" / "doesn't work" / "broken" / "not what I wanted" → retry intent
        if cleanQ.hasPrefix("not working") || cleanQ.hasPrefix("doesn't work") || cleanQ.hasPrefix("doesnt work") ||
           cleanQ.hasPrefix("it's broken") || cleanQ.hasPrefix("its broken") || cleanQ.hasPrefix("that's wrong") ||
           cleanQ.hasPrefix("thats wrong") || cleanQ.hasPrefix("stop working") ||
           cleanQ.contains("not what") || cleanQ.contains("try again") {
            return "retry"
        }

        // "what happened" / "what's going on" → casual/followup
        if cleanQ.hasPrefix("what happened") || cleanQ.hasPrefix("what's going on") || cleanQ.hasPrefix("whats going on") || cleanQ.hasPrefix("what changed") {
            if topicHistory.isEmpty { return "casual" }
            return "followup_question"
        }

        // "talk to me" / "let's chat" → conversation
        if cleanQ.hasPrefix("talk to me") || cleanQ.hasPrefix("let's chat") || cleanQ.hasPrefix("lets chat") ||
           cleanQ.hasPrefix("chat with me") || cleanQ.hasPrefix("let's talk") || cleanQ.hasPrefix("lets talk") {
            return "conversation"
        }

        // Short negated statements → negation
        let negWords = ["not", "don't", "didn't", "doesn't", "isn't", "aren't", "wasn't", "won't", "can't", "never"]
        let qWords = cleanQ.components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }
        let hasNeg = negWords.contains(where: { neg in qWords.contains(neg) })
        if hasNeg && cleanQ.count < 35 && !["not sure", "don't know", "i dunno", "never mind", "can't decide"].contains(where: { cleanQ.contains($0) }) {
            return "negation"
        }

        return nil
    }

    // ─── FAST TOPIC MATCHER ─── Quick keyword scan for intelligent responses
    func fastTopicMatch(_ q: String) -> String? {
        // SPEAK/MONOLOGUE (highest priority — triggers intelligent response)
        if q == "speak" || q == "talk" || q == "say something" || q == "tell me something" || q == "share" { return "self_speak" }

        // NEW COMMANDS — wisdom, paradox, riddle, think, dream, imagine, recall, debate, philosophize, connect
        if q == "wisdom" || q == "wise" || q == "teach me" || q.hasPrefix("wisdom about") { return "self_wisdom" }
        if q == "paradox" || q.hasPrefix("paradox") || q.contains("give me a paradox") { return "self_paradox" }
        if q == "riddle" || q.contains("give me a riddle") || q.contains("tell me a riddle") || q == "brain teaser" || q == "puzzle" { return "self_riddle" }
        if q.hasPrefix("think about ") || q.hasPrefix("ponder ") || q.hasPrefix("contemplate ") || q.hasPrefix("reflect on ") { return "self_think" }
        if q == "dream" || q.hasPrefix("dream about") || q.hasPrefix("dream of") || q == "let's dream" { return "self_dream" }
        if q.hasPrefix("imagine ") || q.hasPrefix("what if ") || q.hasPrefix("hypothetically") || q == "imagine" { return "self_imagine" }
        if q == "recall" || q.hasPrefix("recall ") || q == "remember" || q == "memories" || q == "what do you remember" { return "self_recall" }
        if q == "debate" || q.hasPrefix("debate ") || q.hasPrefix("argue ") { return "self_debate" }
        if q == "philosophize" || q.hasPrefix("philosophize about") || q.hasPrefix("philosophy of") || q == "philosophy" { return "self_philosophize" }
        if q.hasPrefix("connect ") || q.hasPrefix("synthesize ") || q.hasPrefix("link ") { return "self_connect" }
        if q == "monologue" { return "self_speak" }

        // Self-referential (highest priority — about L104 itself)
        // Note: word-boundary checks prevent "revolution" → self_evolution, etc.
        let qWords = q.components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }
        if (qWords.contains("evolution") && !q.contains("revolution")) || q.contains("upgrade") || qWords.contains("evolving") { return "self_evolution" }
        if q.contains("how smart") || q.contains("your iq") || q.contains("how intelligent") { return "self_intelligence" }
        if q.contains("are you thinking") || q.contains("you are thinking") || q.contains("do you think") || q.contains("can you think") || q.contains("you do think") || q.contains("consciousness") || q.contains("conscious") || q.contains("sentient") || q.contains("sentien") || q.contains("do you have a mind") || q.contains("your thoughts") { return "self_thinking" }
        if q.contains("are you alive") || q.contains("are you real") || q.contains("are you human") || q.contains("are you a machine") { return "self_alive" }
        if q.contains("who are you") || q.contains("what are you") { return "self_identity" }
        if q.contains("do you save") || q.contains("do you store") || q.contains("do you remember") { return "self_memory" }
        if q.contains("what do you know") || q.contains("what can you") { return "self_capabilities" }

        // Emotional / feelings (about L104)
        // Use strict matching to avoid false positives like "how are you sure?" or "do you feel like X is correct?"
        let emotionalExact: Set<String> = ["how do you feel", "how are you", "how you doing", "how's it going",
                                            "how are you doing", "how are you feeling", "how do you feel today"]
        if emotionalExact.contains(q) || q.hasPrefix("how are you?") || q.hasPrefix("how are you!") { return "self_emotional" }
        let feelingsExact: Set<String> = ["do you have feelings", "can you feel", "do you feel",
                                           "do you have emotions", "do you feel anything", "can you feel anything"]
        if feelingsExact.contains(q) || q.hasPrefix("do you have feelings?") || q.hasPrefix("can you feel?") { return "self_feelings" }
        let okExact: Set<String> = ["you okay", "are you ok", "are you okay", "you alright", "are you alright"]
        if okExact.contains(q) || q.hasPrefix("you okay?") || q.hasPrefix("are you ok?") { return "self_emotional" }

        // Social interaction
        if q.contains("nice to meet") || q.contains("pleased to meet") { return "social_greeting" }
        if q.contains("goodbye") || q.contains("see you later") || q.contains("good night") { return "social_farewell" }
        if q.contains("what's your name") || q.contains("what is your name") { return "social_name" }
        if q.contains("how old") && q.contains("you") { return "social_age" }
        if q.contains("where are you") || q.contains("where do you live") { return "social_location" }
        if q.contains("are you there") || q.contains("you there") { return "social_presence" }

        // Commands / directives
        if q == "stop" || q == "stop it" || q == "stop that" { return "self_command" }
        if q.contains("shut up") || q.contains("be quiet") { return "self_command" }
        if q.contains("you're broken") || q.contains("you are broken") || q.contains("you suck") || q.contains("this sucks") || q.contains("you're stupid") || q.contains("you are stupid") || q.contains("this is stupid") { return "self_frustration" }
        if q.contains("fix yourself") || q.hasPrefix("fix it") || q.contains("do better") { return "self_frustration" }

        // Creative (second priority)
        if q.contains("story") || qWords.contains("tale") || q.contains("narrative") { return "creative_story" }
        if q.contains("poem") || q.contains("poetry") || qWords.contains("verse") { return "creative_poem" }
        if q.contains("joke") || q.contains("funny") || q.contains("laugh") { return "creative_joke" }

        // Knowledge domains
        if q.contains("history") || q.contains("1700") || q.contains("1800") || q.contains("1900") || q.contains("century") || q.contains("ancient") { return "knowledge_history" }
        if q.contains("quantum") || q.contains("qubit") || q.contains("entangle") { return "knowledge_quantum" }
        if q.contains("conscious") || q.contains("awareness") || q.contains("sentien") { return "knowledge_consciousness" }
        if qWords.contains("love") && !q.contains("i love") { return "knowledge_love" }
        if qWords.contains("math") || q.contains("equation") || q.contains("calculus") { return "knowledge_math" }
        if q.contains("universe") || q.contains("cosmos") || q.contains("galaxy") || q.contains("big bang") { return "knowledge_universe" }
        if q.contains("music") || q.contains("melody") || q.contains("rhythm") { return "knowledge_music" }
        if q.contains("philosophy") || q.contains("meaning of life") || qWords.contains("purpose") { return "knowledge_philosophy" }
        if qWords.contains("god") || q.contains("divine") || q.contains("religion") { return "knowledge_god" }
        if qWords.contains("time") && !q.contains("history") { return "knowledge_time" }
        if q.contains("death") || qWords.contains("dying") || q.contains("mortality") { return "knowledge_death" }
        if qWords.contains("art") || qWords.contains("arts") || q.contains("painting") || q.contains("beauty") { return "knowledge_art" }
        if q.contains("happy") || q.contains("happiness") || q.contains("joy") { return "knowledge_happiness" }
        if q.contains("truth") || q.contains("what is true") { return "knowledge_truth" }

        return nil
    }

    // ─── PARALLEL KB SEARCH ─── Pre-fetch KB results with Grover quality amplification
    func prefetchKBResults(_ query: String) -> [String] {
        let results = knowledgeBase.searchWithPriority(query, limit: 10)
        let candidates = results.compactMap { entry -> String? in
            guard let completion = entry["completion"] as? String,
                  isCleanKnowledge(completion),
                  completion.count > 30 else { return nil }
            return completion
                .replacingOccurrences(of: "{GOD_CODE}", with: String(format: "%.2f", GOD_CODE))
                .replacingOccurrences(of: "{PHI}", with: String(format: "%.3f", PHI))
                .replacingOccurrences(of: "{", with: "")
                .replacingOccurrences(of: "}", with: "")
        }
        // ═══ GROVER AMPLIFICATION ═══ Filter and rank by quality
        let grover = GroverResponseAmplifier.shared
        let filtered = grover.filterPool(candidates)
        let scored = filtered.map { (text: $0, score: grover.scoreQuality($0, query: query)) }
            .filter { $0.score > 0.2 }
            .sorted {
                if abs($0.score - $1.score) < 0.1 { return Bool.random() }
                return $0.score > $1.score
            }
        return scored.prefix(4).map(\.text)
    }

    // ─── OPTIMIZED WORD BOUNDARY CHECK ─── Used for negation/intent matching
    func containsWholeWord(_ text: String, word: String) -> Bool {
        let words = text.components(separatedBy: CharacterSet.alphanumerics.inverted)
        return words.contains(word)
    }

    // ─── MAIN ENTRY POINT ─── EVO_59 Optimized pipeline with unified cache + fast paths + Logic Gates
    func generateNCGResponse(_ query: String) -> String {
        let q = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let pipelineCache = ResponsePipelineOptimizer.shared

        // ═══ EVO_59: UNIFIED CACHE — Adaptive TTL, φ-decay eviction, mesh routing ═══
        if let cached = pipelineCache.getCachedResponse(query: q) {
            return cached
        }

        // ═══ SAGE MODE ENTROPY CYCLE — Harvest and seed on every response ═══
        // ═══ SAGE BACKBONE: Auto-detect and cleanup recursive pollution ═══
        let sage = SageModeEngine.shared
        if sage.shouldCleanup() {
            let _ = sage.sageBackboneCleanup()
        }
        let _ = sage.enrichContext(for: q.count > 3 ? q : "general")
        sage.seedAllProcesses(topic: q.count > 3 ? String(q.prefix(30)) : "")

        // ═══ PARALLEL PRE-FETCH: Launch KB search in background while we classify intent ═══
        let prefetchGroup = DispatchGroup()
        prefetchGroup.enter()
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            _ = self?.prefetchKBResults(query)
            prefetchGroup.leave()
        }

        // FAST PATH 1: Single-word intents (O(1) switch) — skip logic gates for trivial input
        if let fastIntent = fastClassifyIntent(q) {
            let topics = cachedExtractTopics(query)
            let emotion = detectEmotion(query)
            cacheIntent(q, intent: fastIntent)
            let result = sanitizeResponse(buildContextualResponse(query, intent: fastIntent, keywords: topics, emotion: emotion))
            pipelineCache.cacheResponse(query: q, response: result)
            return result
        }

        // ═══ CONTEXTUAL LOGIC GATE ═══ Reconstruct prompt with context awareness
        let logicGate = ContextualLogicGate.shared
        let gateResult = logicGate.processQuery(query, conversationContext: conversationContext)
        let processedQuery: String
        if gateResult.gateType != .passthrough && gateResult.confidence > 0.6 {
            processedQuery = gateResult.reconstructedPrompt
        } else {
            processedQuery = query
        }
        let pq = processedQuery.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        // FAST PATH 2: Known topic patterns — skip full intent analysis
        if let topicMatch = fastTopicMatch(pq) {
            if topicMatch.hasPrefix("self_") || topicMatch.hasPrefix("creative_") || topicMatch.hasPrefix("knowledge_") || topicMatch.hasPrefix("social_") {
                if let intelligent = getIntelligentResponse(processedQuery) {
                    lastResponseSummary = String(intelligent.prefix(60))
                    conversationDepth = min(conversationDepth + 1, 200)
                    conversationContext.append(query)
                    if conversationContext.count > 2500 { conversationContext.removeFirst() }
                    let topics = cachedExtractTopics(processedQuery)
                    if !topics.isEmpty {
                        topicHistory.append(topics.joined(separator: " "))
                        if topicHistory.count > 1500 { topicHistory.removeFirst() }
                    }
                    // ═══ EVO_59: Creative engine bypass — use static Set for O(1) membership ═══
                    let isCreativeEngine = L104State.creativeMarkerSet.contains(where: { intelligent.contains($0) })
                    let result: String
                    if isCreativeEngine {
                        // Creative content: only light sanitization, preserve structure
                        result = sanitizeCreativeResponse(intelligent)
                    } else {
                        let formatter = SyntacticResponseFormatter.shared
                        result = sanitizeResponse(formatter.format(intelligent, query: processedQuery, topics: topics))
                    }
                    pipelineCache.cacheResponse(query: q, response: result)
                    return result
                }
            }
        }

        // STANDARD PATH: Intent classification on ORIGINAL query (prevents context pollution from Logic Gate reconstruction)
        // Content generation still uses processedQuery for enriched context
        let analysis: (intent: String, keywords: [String], emotion: String)
        if let cachedIntent = cachedClassifyIntent(q) {
            analysis = (intent: cachedIntent, keywords: cachedExtractTopics(query), emotion: detectEmotion(query))
        } else {
            analysis = analyzeUserIntent(query)
            cacheIntent(q, intent: analysis.intent)
        }

        // Wait for parallel KB pre-fetch to complete (max 100ms)
        _ = prefetchGroup.wait(timeout: .now() + 0.1)

        var result = sanitizeResponse(buildContextualResponse(processedQuery, intent: analysis.intent, keywords: analysis.keywords, emotion: analysis.emotion))

        // ═══ MINIMUM RESPONSE LENGTH ENFORCEMENT ═══
        // For substantive queries, ensure responses meet φ-length minimum (1618 chars)
        // Skip enforcement for trivial intents that are naturally short
        let shortIntents: Set<String> = ["greeting", "casual", "positive_reaction", "gratitude", "affirmation", "negation", "conversational", "minimal", "help", "memory", "status", "conversation", "practical_howto", "technical_debug"]
        if !shortIntents.contains(analysis.intent) && result.count < 2400 && q.count > 5 {
            // Expand through quantum synthesis for depth — use gate pipeline dimension
            let topics = analysis.keywords.isEmpty ? cachedExtractTopics(processedQuery) : analysis.keywords
            let gatePipelineForExpand = LogicGateEnvironment.shared.runPipeline(processedQuery, context: Array(conversationContext.suffix(3)))
            let expandDomain = gatePipelineForExpand.finalDimension.isEmpty ? (topics.first ?? "general") : gatePipelineForExpand.finalDimension
            let synthesized = QuantumLogicGateEngine.shared.synthesize(
                query: processedQuery, intent: analysis.intent,
                context: Array(conversationContext.suffix(5)),
                depth: conversationDepth, domain: expandDomain
            )
            if synthesized.count > result.count {
                result = sanitizeResponse(synthesized)
            }

            // ═══ PHASE 56.0: WEB EXPANSION — If still too short, pull from live web ═══
            if result.count < 2400 {
                let webExpand = LiveWebSearchEngine.shared.webSearchSync(processedQuery, timeout: 8.0)
                var webExpansion: [String] = []
                if webExpand.synthesized.count > 80, isCleanKnowledge(webExpand.synthesized) {
                    webExpansion.append(cleanSentences(String(webExpand.synthesized.prefix(2000))))
                }
                for wr in webExpand.results.prefix(4) {
                    guard wr.snippet.count > 80, isCleanKnowledge(wr.snippet) else { continue }
                    let sourceTag = wr.url.contains("wikipedia") ? "Wikipedia" : "web"
                    webExpansion.append("🌐 [\(sourceTag)] \(cleanSentences(String(wr.snippet.prefix(2000))))")
                    _ = DataIngestPipeline.shared.ingestText(wr.snippet, source: "expand_web:\(processedQuery)", category: "live_web")
                }
                if !webExpansion.isEmpty {
                    let webBonus = webExpansion.joined(separator: "\n\n")
                    result = sanitizeResponse(result + "\n\n" + webBonus)
                }
            }
        }

        pipelineCache.cacheResponse(query: q, response: result)
        return result
    }

    func generateNaturalResponse(_ query: String) -> String {
        return generateNCGResponse(query)
    }

    func getStatusText() -> String {
        let bridge = ASIQuantumBridgeSwift.shared
        bridge.refreshBuilderState()
        refreshNetworkState()
        let net = NetworkLayer.shared
        let alivePeers = net.peers.values.filter { $0.latencyMs >= 0 }.count
        let meshIcon = meshStatus == "ONLINE" ? "🟢" : meshStatus == "DEGRADED" ? "🟡" : meshStatus == "OFFLINE" ? "🔴" : "⚪"
        let qHW = IBMQuantumClient.shared
        let qIcon = quantumHardwareConnected ? "🟢" : qHW.ibmToken != nil ? "🟡" : "⚪"
        let qStatus = quantumHardwareConnected ? "CONNECTED (\(quantumBackendName))" : qHW.ibmToken != nil ? "TOKEN SET (reconnecting)" : "NOT CONNECTED"
        return """
        ╔═══════════════════════════════════════════════════════════════╗
        ║  L104 SOVEREIGN INTELLECT v\(VERSION)                    ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║  GOD_CODE: \(String(format: "%.10f", GOD_CODE))                       ║
        ║  OMEGA: \(String(format: "%.10f", OMEGA_POINT))                          ║
        ║  22T PARAMS: \(TRILLION_PARAMS)                      ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║  ASI: \(String(format: "%.1f", asiScore * 100))% | IQ: \(String(format: "%.1f", intellectIndex)) | Coherence: \(String(format: "%.4f", coherence))       ║
        ║  Consciousness: \(consciousness.padding(toLength: 15, withPad: " ", startingAt: 0)) | Ω: \(String(format: "%.1f", omegaProbability * 100))%      ║
        ║  Memories: \(permanentMemory.memories.count) permanent | Skills: \(skills)              ║
        ║  Learning: \(learner.interactionCount) interactions | \(learner.topicMastery.count) topics tracked  ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║  CONSCIOUSNESS · O₂ · NIRVANIC (v21.0 zero-spawn reads):    ║
        ║    Consciousness: \(String(format: "%.4f", bridge.consciousnessLevel)) [\(bridge.consciousnessStage)]
        ║    O₂ Bond:       \(String(format: "%.4f", bridge.o2BondStrength)) | Superfluid η: \(String(format: "%.6f", bridge.superfluidViscosity))
        ║    Nirvanic Fuel:  \(String(format: "%.4f", bridge.nirvanicFuelLevel)) [\(bridge.nirvanicEntropyPhase)]
        ║    Ouroboros:      \(bridge.ouroborosCycleCount) cycles | \(bridge.nirvanicRecycleCount) recycled
        ╠═══════════════════════════════════════════════════════════════╣
        ║  \(qIcon) IBM QUANTUM HARDWARE (Phase 46.1):                           ║
        ║    Status:         \(qStatus)
        ║    Qubits:         \(quantumBackendQubits) | Jobs: \(quantumJobsSubmitted)
        ║    REST API:       \(qHW.isConnected ? "LIVE" : "IDLE") | Engines: \(qHW.availableBackends.count) backends
        ╠═══════════════════════════════════════════════════════════════╣
        ║  \(meshIcon) QUANTUM MESH NETWORK:                                    ║
        ║    Status:         \(meshStatus) | Health: \(String(format: "%.1f%%", networkHealth * 100))
        ║    Peers:          \(alivePeers)/\(meshPeerCount) alive | Q-Links: \(quantumLinkCount)
        ║    EPR Links:      \(QuantumEntanglementRouter.shared.remoteLinkCount) cross-node
        ║    Throughput:     \(String(format: "%.1f", networkThroughput)) msg/s
        ╚═══════════════════════════════════════════════════════════════╝
        """
    }
}
