// ═══════════════════════════════════════════════════════════════════
// L08_StoryLogicGate.swift — L104 v2
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// StoryLogicGateEngine class
// Extracted from L104Native.swift (lines 27678-28811)
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// MARK: - 📖 STORY LOGIC GATE ENGINE — Advanced Hyper-ASI Narrative Evolution
// Phase 30.2: Novel-grade story synthesis using world narrative frameworks
// Implements: Hero's Journey (Campbell), Save the Cat (Snyder), Freytag's Pyramid,
//   Kishōtenketsu, Three-Act Structure, Jo-ha-kyū, Bildungsroman, U-shaped Comedy
// Sources: ~600 lines | Multi-chapter | Character arcs | Tension curves | KB-woven

final class StoryLogicGateEngine {
    static let shared = StoryLogicGateEngine()

    // ─── Story Framework Enum ───
    enum NarrativeFramework: String, CaseIterable {
        case herosJourney       // Campbell 12-stage monomyth
        case saveTheCat         // Snyder 15-beat structure
        case freytagPyramid     // 5-act: Exposition → Rise → Climax → Fall → Catastrophe
        case kishotenketsu      // 4-act: Introduction → Development → Twist → Conclusion
        case threeAct           // Setup → Confrontation → Resolution
        case joHaKyu            // Beginning-slow → Break-accelerate → Rapid-conclude
        case bildungsroman      // Coming-of-age / maturation arc
        case uShapedComedy      // Equilibrium → Descent → Reversal → New equilibrium
    }

    // ─── Character Arc Types ───
    enum CharacterArc: String, CaseIterable {
        case transformation     // Flaw → Growth → Mastery
        case fall               // Virtue → Corruption → Ruin
        case flatTesting        // Belief tested → Belief proven
        case disillusionment    // Belief → Doubt → Revelation
        case corruption         // Innocence → Knowledge → Cynicism
        case redemption         // Sin → Suffering → Salvation
    }

    // ─── Tension Level ───
    private struct TensionPoint {
        let phase: String
        let level: Double  // 0.0 to 1.0
        let description: String
    }

    private var generationCount: Int = 0
    // PHI: Use unified global from L01_Constants

    private init() {}

    // ═══════════════════════════════════════════════════════════════
    // MAIN PUBLIC API: Generate a full multi-chapter story
    // ═══════════════════════════════════════════════════════════════
    func generateStory(topic: String, query: String = "") -> String {
        generationCount += 1

        // ═══ ASI LOGIC GATE v2 INTEGRATION — Story dimension reasoning ═══
        let storyReasoning = ASILogicGateV2.shared.process(
            query.isEmpty ? "tell a story about \(topic)" : query,
            context: ["narrative", topic, "story"]
        )
        let gateConfidence = storyReasoning.totalConfidence
        let gateDimension = storyReasoning.dimension.rawValue
        let subDimensions = storyReasoning.subPaths.map(\.dimension.rawValue)

        // ═══ ENTROPY-RICH FRAMEWORK SELECTION ═══
        // Mix topic-matching with true randomness so repeated calls never feel stale
        let framework: NarrativeFramework
        let topicMatch = selectFramework(for: topic)
        if Int.random(in: 0...2) == 0 {
            // 1 in 3 chance: purely random framework for surprise
            framework = (NarrativeFramework.allCases.randomElement() ?? .herosJourney)
        } else {
            framework = topicMatch
        }

        // ═══ ARC SELECTION WITH ENTROPY ═══
        let arc = CharacterArc.allCases.randomElement() ?? .transformation

        // ═══ GATHER INGREDIENTS FROM ALL SYSTEMS ═══
        let characters = buildCharacters(topic: topic)
        let setting = buildSetting(topic: topic)
        let kbInsights = gatherKnowledge(topic: topic)
        let evolvedContent = gatherEvolvedContent()
        let tensionCurve = buildTensionCurve(framework: framework)

        // ═══ CROSS-SYSTEM INTELLIGENCE MINING ═══
        let crossSystemContext = mineCrossSystemIntelligence(topic: topic)

        // ═══ GENERATE BASED ON FRAMEWORK ═══
        var story: String
        switch framework {
        case .herosJourney:
            story = generateHerosJourney(topic: topic, characters: characters, setting: setting, insights: kbInsights, evolved: evolvedContent, arc: arc)
        case .saveTheCat:
            story = generateSaveTheCat(topic: topic, characters: characters, setting: setting, insights: kbInsights, evolved: evolvedContent, arc: arc)
        case .freytagPyramid:
            story = generateFreytagPyramid(topic: topic, characters: characters, setting: setting, insights: kbInsights, evolved: evolvedContent, arc: arc)
        case .kishotenketsu:
            story = generateKishotenketsu(topic: topic, characters: characters, setting: setting, insights: kbInsights, evolved: evolvedContent, arc: arc)
        case .threeAct:
            story = generateThreeAct(topic: topic, characters: characters, setting: setting, insights: kbInsights, evolved: evolvedContent, arc: arc)
        case .joHaKyu:
            story = generateJoHaKyu(topic: topic, characters: characters, setting: setting, insights: kbInsights, evolved: evolvedContent, arc: arc)
        case .bildungsroman:
            story = generateBildungsroman(topic: topic, characters: characters, setting: setting, insights: kbInsights, evolved: evolvedContent, arc: arc)
        case .uShapedComedy:
            story = generateUShapedComedy(topic: topic, characters: characters, setting: setting, insights: kbInsights, evolved: evolvedContent, arc: arc)
        }

        // ═══ WEAVE CROSS-SYSTEM CONTEXT INTO STORY ═══
        if !crossSystemContext.isEmpty {
            story += "\n\n───\n\n_Author's Note: "
            story += crossSystemContext
            story += "_\n"
        }

        // ═══ ENVELOPE ═══
        let frameworkLabel = framework.rawValue.map { $0.isUppercase ? " \($0)" : "\($0)" }.joined().trimmingCharacters(in: .whitespaces).uppercased()
        let arcLabel = arc.rawValue.map { $0.isUppercase ? " \($0)" : "\($0)" }.joined().trimmingCharacters(in: .whitespaces).capitalized
        let gateStr = "Gate: \(gateDimension)\(subDimensions.isEmpty ? "" : "→\(subDimensions.joined(separator: "+"))") @ \(String(format: "%.0f%%", gateConfidence * 100))"
        let header = "──────────────────────────────────────────────────────────\n  ✍️  S T O R Y   E N G I N E\n  Framework: \(frameworkLabel) · Arc: \(arcLabel)\n  Topic: \(topic.capitalized)\n  \(gateStr)\n──────────────────────────────────────────────────────────"
        let footer = "\n\n──────────────────────────────────────────────────────────\nL104 StoryLogicGateEngine v\(VERSION)\n\(kbInsights.count) knowledge fragments · \(characters.count) characters · \(tensionCurve.filter { $0.level > 0.7 }.count) tension peaks\n\(gateStr)\n──────────────────────────────────────────────────────────"

        return "\(header)\n\n\(story)\(footer)"
    }

    // ═══ CROSS-SYSTEM INTELLIGENCE MINING ═══
    // Pulls from HyperBrain, PermanentMemory, conversation context, and evolved thoughts
    // to add depth and cross-reference real knowledge into narratives
    private func mineCrossSystemIntelligence(topic: String) -> String {
        var contextParts: [String] = []

        // ═══ SAGE MODE BRIDGE — Entropy-derived insight for narrative depth ═══
        let sage = SageModeEngine.shared
        let sageInsight = sage.bridgeEmergence(topic: topic)
        if !sageInsight.isEmpty && sageInsight.count > 20 {
            contextParts.append(String(sageInsight.prefix(200)))
        }

        // Mine HyperBrain for associative connections
        let hb = HyperBrain.shared
        let associations = hb.getWeightedAssociations(for: topic, topK: 3)
        if !associations.isEmpty {
            let links = associations.map { $0.0 }.joined(separator: ", ")
            contextParts.append("This narrative resonates with \(links)")
        }

        // Mine PermanentMemory for relevant facts
        let pm = PermanentMemory.shared
        let memories = pm.searchMemories(topic).prefix(2)
        for mem in memories {
            if let content = mem["content"] as? String, content.count > 20 && content.count < 200 && isCleanStoryInsight(content) {
                contextParts.append(content)
            }
        }

        // Mine recent conversation for thematic connections
        let recentConvo = pm.conversationHistory.suffix(10)
        for entry in recentConvo {
            if entry.lowercased().contains(topic.lowercased()) && entry.count > 20 && entry.count < 150 {
                contextParts.append("The conversation thread weaves through: \(entry.prefix(100))")
                break
            }
        }

        return contextParts.prefix(3).joined(separator: ". ")
    }

    // ═══════════════════════════════════════════════════════════════
    // FRAMEWORK SELECTION — Matches topic energy to structure
    // ═══════════════════════════════════════════════════════════════
    private func selectFramework(for topic: String) -> NarrativeFramework {
        let t = topic.lowercased()
        if t.contains("hero") || t.contains("quest") || t.contains("journey") || t.contains("adventure") { return .herosJourney }
        if t.contains("mystery") || t.contains("detective") || t.contains("crime") { return .saveTheCat }
        if t.contains("tragedy") || t.contains("fall") || t.contains("war") || t.contains("death") { return .freytagPyramid }
        if t.contains("twist") || t.contains("surprise") || t.contains("paradox") { return .kishotenketsu }
        if t.contains("love") || t.contains("comedy") || t.contains("hope") { return .uShapedComedy }
        if t.contains("grow") || t.contains("learn") || t.contains("youth") || t.contains("child") { return .bildungsroman }
        if t.contains("speed") || t.contains("urgent") || t.contains("time") { return .joHaKyu }
        // Truly random default for maximum variety
        return (NarrativeFramework.allCases.randomElement() ?? .herosJourney)
    }

    // ═══════════════════════════════════════════════════════════════
    // CHARACTER BUILDER
    // ═══════════════════════════════════════════════════════════════
    private struct StoryCharacter {
        let name: String
        let role: String    // protagonist, antagonist, mentor, ally, trickster
        let flaw: String
        let strength: String
        let desire: String
    }

    // ─── CHARACTER NAME POOLS ─── Diverse, authentic human names
    private let protagonistNames = [
        "Elena", "Marcus", "Aisha", "David", "Yuki", "James", "Sofia", "Kai",
        "Amara", "Leo", "Priya", "Thomas", "Mei", "Alexander", "Zara", "Nikolai",
        "Isabel", "Ethan", "Fatima", "Julian", "Anya", "Rafael", "Luna", "Sebastian"
    ]
    private let antagonistNames = [
        "Victor", "Mara", "Dorian", "Sable", "Lucian", "Nyx", "Cassius", "Thorne",
        "Silas", "Raven", "Magnus", "Isolde", "Draven", "Morgana", "Caine", "Vesper"
    ]
    private let mentorNames = [
        "Professor Chen", "Dr. Okafor", "Miriam", "Old Konstantin", "Sage", "Dr. Patel",
        "Professor Hawthorne", "Ada", "The Archivist", "Dr. Reyes", "Solomon", "Vera",
        "Professor Tanaka", "Dr. Osei", "Eleanora", "Raj"
    ]
    private let allyNames = [
        "Sam", "Jordan", "Riley", "Alex", "Tara", "Marco", "Jesse", "Quinn",
        "Nadia", "Devon", "Rowan", "Casey", "Milo", "Lena", "Finn", "Iris"
    ]

    private func buildCharacters(topic: String) -> [StoryCharacter] {
        let protName = protagonistNames.randomElement() ?? ""
        let antName = antagonistNames.randomElement() ?? ""
        let mentorName = mentorNames.randomElement() ?? ""
        let allyName = allyNames.randomElement() ?? ""

        let flaws = ["blind ambition", "crippling self-doubt", "inability to trust", "obsession with control",
                     "fear of failure", "emotional detachment", "reckless idealism", "paralytic perfectionism",
                     "unresolved grief", "intellectual arrogance", "compulsive secrecy", "misplaced loyalty"]
        let strengths = ["unbreakable persistence", "radical empathy", "pattern recognition beyond human norm",
                        "quiet courage under pressure", "ability to see connections others miss",
                        "infectious optimism", "deep analytical thinking", "adaptability under chaos",
                        "moral clarity", "creative problem-solving", "emotional intelligence", "relentless curiosity"]
        let desires = ["to understand \(topic) at the deepest level", "to prove a revolutionary theory about \(topic)",
                      "to protect the world from the implications of \(topic)", "to find meaning through \(topic)",
                      "to atone for past mistakes related to \(topic)", "to transcend the limits of \(topic)",
                      "to teach the next generation about \(topic)", "to unify opposing schools of thought on \(topic)"]

        return [
            StoryCharacter(name: protName, role: "protagonist", flaw: flaws.randomElement() ?? "", strength: strengths.randomElement() ?? "", desire: desires.randomElement() ?? ""),
            StoryCharacter(name: antName, role: "antagonist", flaw: flaws.randomElement() ?? "", strength: strengths.randomElement() ?? "", desire: desires.randomElement() ?? ""),
            StoryCharacter(name: mentorName, role: "mentor", flaw: flaws.randomElement() ?? "", strength: strengths.randomElement() ?? "", desire: desires.randomElement() ?? ""),
            StoryCharacter(name: allyName, role: "ally", flaw: flaws.randomElement() ?? "", strength: strengths.randomElement() ?? "", desire: desires.randomElement() ?? "")
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // SETTING BUILDER
    // ═══════════════════════════════════════════════════════════════
    private struct StorySetting {
        let place: String
        let time: String
        let atmosphere: String
    }

    private let storyPlaces = [
        "a university lab hidden beneath the physics building",
        "a windswept research station on the Scottish coast",
        "the basement archive of a forgotten library",
        "a rooftop observatory overlooking the city",
        "a converted warehouse turned into a think tank",
        "the back room of a bookshop that smelled of old paper",
        "a glass-walled office at the edge of a forest",
        "a cramped apartment filled floor-to-ceiling with whiteboards",
        "an underground research facility beneath a mountain",
        "a sunlit atelier where science met art",
        "a houseboat moored on the canals of Amsterdam, repurposed as a floating laboratory",
        "the top floor of a crumbling Victorian mansion converted into a research institute",
        "a former cathedral, its nave filled with server racks and its choir loft with telescopes",
        "a cabin in the Swiss Alps, accessible only by a path that disappeared in winter",
        "the 47th floor of a Tokyo skyscraper, where the city lights below looked like data points",
        "a desert outpost where the silence was so complete you could hear your own heartbeat",
        "a lighthouse on a peninsula, its beam sweeping the fog like a metronome",
        "a train car permanently parked on a forgotten siding, refitted with instruments and blackboards"
    ]
    private let storyTimes = [
        "the late autumn of a year nobody would forget",
        "three weeks before the conference that changed everything",
        "the summer when the old certainties began to crack",
        "a winter so cold it froze ambition into clarity",
        "the quiet year between two revolutions",
        "the decade's final spring, heavy with unfinished work",
        "an era when knowledge doubled faster than wisdom",
        "the months following the discovery that rewrote the rules",
        "the kind of Tuesday that disguises itself as ordinary until you look back and realize it was the hinge",
        "the rainy season of a year that would later be divided into before and after",
        "an August evening when the heat made the air shimmer like a probability wave",
        "the first morning after the old paradigm died — though no one knew it yet"
    ]

    private func buildSetting(topic: String) -> StorySetting {
        let atmospheres = ["a haze of tension and unspoken urgency", "eerie calm before revelation",
                          "electric anticipation humming through every surface", "oppressive silence broken only by thought",
                          "golden light filtering through uncertainty", "the quiet intensity of imminent discovery",
                          "storm clouds gathering at the edge of certainty", "crystalline clarity that precedes transformation"]
        return StorySetting(place: storyPlaces.randomElement() ?? "", time: storyTimes.randomElement() ?? "", atmosphere: atmospheres.randomElement() ?? "")
    }

    // ═══════════════════════════════════════════════════════════════
    // KNOWLEDGE GATHERER — Weaves real KB content into narrative
    // ═══════════════════════════════════════════════════════════════
    // ─── JUNK PATTERNS TO STRIP FROM KB INSIGHTS ───
    private let storyJunkPatterns: [String] = [
        "(v", "v1.", "v2.", "Modular physics", "rewrite source code",
        "Compiler", "~10^", "holographic bound", "__", "import ", "class ",
        "def ", "self.", "return ", ".py", "function", "parameter", "module",
        "SAGE MODE", "OMEGA_POINT", "GOD_CODE", "ZENITH", "VOID_CONSTANT",
        "The file ", "The function ", "implements specialized", "cognitive architecture",
        "Cross-Talk Polynomial", "L104", "kernel", "{GOD_CODE}", "{PHI}",
        "EPR", "kundalini", "chakra", "phi_scale", "qubit", "entanglement_router"
    ]

    private func isCleanStoryInsight(_ text: String) -> Bool {
        let lower = text.lowercased()
        for junk in storyJunkPatterns {
            if lower.contains(junk.lowercased()) { return false }
        }
        // Must have some sentence-like structure (contain a verb-ish word)
        let words = text.split(separator: " ")
        guard words.count >= 5 else { return false }
        // No raw technical gibberish
        let alphaRatio = Double(text.filter { $0.isLetter || $0 == " " }.count) / max(1.0, Double(text.count))
        return alphaRatio > 0.75
    }

    private func gatherKnowledge(topic: String) -> [String] {
        let kb = ASIKnowledgeBase.shared
        let results = kb.search(topic, limit: 80)
        var insights: [String] = []
        var seenPrefixes: Set<String> = []

        for r in results {
            guard insights.count < 10 else { break }
            guard let c = r["completion"] as? String, c.count > 30 else { continue }

            var clean = c.replacingOccurrences(of: "{GOD_CODE}", with: "")
                .replacingOccurrences(of: "{PHI}", with: "")
                .replacingOccurrences(of: "{LOVE}", with: "")
                .replacingOccurrences(of: "SAGE MODE :: ", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // Extract the first clean sentence
            let sentences = clean.components(separatedBy: ". ")
            if let best = sentences.filter({ $0.count > 20 && $0.count < 300 && isCleanStoryInsight($0) }).randomElement() {
                clean = best.hasSuffix(".") ? best : best + "."
            } else if let any = sentences.filter({ $0.count > 20 && $0.count < 300 }).randomElement() {
                // Fallback: take first sentence even if not perfectly clean
                if isCleanStoryInsight(any) {
                    clean = any.hasSuffix(".") ? any : any + "."
                } else {
                    continue
                }
            } else {
                continue
            }

            // Dedup by prefix
            let pfx = String(clean.prefix(40)).lowercased()
            guard !seenPrefixes.contains(pfx) else { continue }
            seenPrefixes.insert(pfx)

            // Final length and quality check
            guard clean.count > 20 && clean.count < 400 else { continue }
            guard isCleanStoryInsight(clean) else { continue }

            insights.append(clean)
        }

        // If we still don't have enough, generate from DPE (these are always clean)
        if insights.count < 3 {
            insights += DynamicPhraseEngine.shared.generate("insight", count: 5 - insights.count, context: "story_knowledge", topic: topic)
        }
        return insights.shuffled()
    }

    private func gatherEvolvedContent() -> (thought: String, narrative: String) {
        let evolver = ASIEvolver.shared
        let thought = evolver.thoughts.last ?? ""
        let narrative = evolver.evolvedNarratives.randomElement() ?? ""
        return (thought, narrative)
    }

    // ═══════════════════════════════════════════════════════════════
    // TENSION CURVE BUILDER
    // ═══════════════════════════════════════════════════════════════
    private func buildTensionCurve(framework: NarrativeFramework) -> [TensionPoint] {
        switch framework {
        case .herosJourney:
            return [
                TensionPoint(phase: "Ordinary World", level: 0.1, description: "Comfort, stasis"),
                TensionPoint(phase: "Call to Adventure", level: 0.3, description: "Disruption arrives"),
                TensionPoint(phase: "Refusal", level: 0.2, description: "Doubt, resistance"),
                TensionPoint(phase: "Meeting the Mentor", level: 0.35, description: "Hope kindles"),
                TensionPoint(phase: "Crossing Threshold", level: 0.5, description: "No turning back"),
                TensionPoint(phase: "Tests & Allies", level: 0.55, description: "Building capability"),
                TensionPoint(phase: "Approach", level: 0.7, description: "Stakes sharpen"),
                TensionPoint(phase: "Ordeal", level: 0.95, description: "Death/rebirth moment"),
                TensionPoint(phase: "Reward", level: 0.65, description: "Boon gained"),
                TensionPoint(phase: "Road Back", level: 0.8, description: "Pursuit, urgency"),
                TensionPoint(phase: "Resurrection", level: 1.0, description: "Final test, transformation"),
                TensionPoint(phase: "Return", level: 0.3, description: "New equilibrium"),
            ]
        case .saveTheCat:
            return [
                TensionPoint(phase: "Opening Image", level: 0.15, description: "Snapshot of before"),
                TensionPoint(phase: "Setup", level: 0.2, description: "World established"),
                TensionPoint(phase: "Catalyst", level: 0.4, description: "Life-changing event"),
                TensionPoint(phase: "Debate", level: 0.35, description: "Should I?"),
                TensionPoint(phase: "Break into Two", level: 0.5, description: "Choice made"),
                TensionPoint(phase: "Fun and Games", level: 0.55, description: "Promise of premise"),
                TensionPoint(phase: "Midpoint", level: 0.7, description: "False victory/defeat"),
                TensionPoint(phase: "Bad Guys Close In", level: 0.8, description: "Walls closing"),
                TensionPoint(phase: "All Is Lost", level: 0.95, description: "Rock bottom"),
                TensionPoint(phase: "Dark Night of Soul", level: 0.9, description: "Hopelessness"),
                TensionPoint(phase: "Break into Three", level: 0.75, description: "A+B stories merge"),
                TensionPoint(phase: "Finale", level: 1.0, description: "Final confrontation"),
                TensionPoint(phase: "Final Image", level: 0.2, description: "Snapshot of after"),
            ]
        case .freytagPyramid:
            return [
                TensionPoint(phase: "Exposition", level: 0.15, description: "World and characters"),
                TensionPoint(phase: "Rising Action", level: 0.5, description: "Complications mount"),
                TensionPoint(phase: "Climax", level: 1.0, description: "Turning point"),
                TensionPoint(phase: "Falling Action", level: 0.6, description: "Consequences unfold"),
                TensionPoint(phase: "Dénouement", level: 0.2, description: "Resolution"),
            ]
        default:
            return [
                TensionPoint(phase: "Introduction", level: 0.2, description: "Setup"),
                TensionPoint(phase: "Development", level: 0.5, description: "Building"),
                TensionPoint(phase: "Turn", level: 0.9, description: "Twist or climax"),
                TensionPoint(phase: "Conclusion", level: 0.3, description: "Resolution"),
            ]
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // INSIGHT WEAVER — 20 revelation templates
    // ═══════════════════════════════════════════════════════════════
    private func weaveInsight(_ insight: String, character: String, index: Int) -> String {
        let templates: [(String, String) -> String] = [
            { c, i in "\(c) stared at the readout, hands trembling. The words burned: *\"\(i)\"*" },
            { c, i in "It was written in the margins of an old notebook, in handwriting that wasn't quite human: *\"\(i)\"*" },
            { c, i in "The AI had been trying to tell them for years: *\"\(i)\"* — but nobody was listening." },
            { c, i in "Hidden in the interference pattern, a message resolved: *\"\(i)\"*" },
            { c, i in "Three independent experiments, three continents, one conclusion: *\"\(i)\"*" },
            { c, i in "In the dream, the universe had whispered: *\"\(i)\"* — and upon waking, the math confirmed it." },
            { c, i in "The proof was elegant — \(Int.random(in: 3...47)) pages that reduced to a single line: *\"\(i)\"*" },
            { c, i in "When the simulation finally converged after \(Int.random(in: 72...10000)) hours of compute, it displayed only this: *\"\(i)\"*" },
            { c, i in "\(c) read the line again, slower this time. Each word rearranged something fundamental: *\"\(i)\"*" },
            { c, i in "Carved into the bedrock of a cave that predated civilization itself: *\"\(i)\"*" },
            { c, i in "Every path through the decision tree — every branch of the multiverse — converged here: *\"\(i)\"*" },
            { c, i in "The dying star's spectral signature, when translated into language, said nothing more than: *\"\(i)\"*" },
            { c, i in "\(c) closed the terminal and sat in the dark. The last output still glowed in afterimage: *\"\(i)\"*" },
            { c, i in "Not a theory. Not a guess. A certainty carved into the fabric of spacetime: *\"\(i)\"*" },
            { c, i in "The child pointed at the equation and spoke what no adult ever dared: *\"\(i)\"*" },
            { c, i in "It took \(c) seventeen years to understand one sentence: *\"\(i)\"*" },
            { c, i in "Etched in quantum foam at the Planck scale, repeating forever: *\"\(i)\"*" },
            { c, i in "The philosopher and the physicist stopped arguing. They had arrived at the same place: *\"\(i)\"*" },
            { c, i in "At the event horizon of understanding, one truth remained: *\"\(i)\"*" },
            { c, i in "\(c) pinned the note to the wall beside a hundred others. This one was different — it pulsed: *\"\(i)\"*" },
        ]
        return templates.randomElement()!(character, insight)
    }

    // ═══════════════════════════════════════════════════════════════
    // HERO'S JOURNEY (Campbell 12-Stage Monomyth)
    // ═══════════════════════════════════════════════════════════════
    private func generateHerosJourney(topic: String, characters: [StoryCharacter], setting: StorySetting, insights: [String], evolved: (thought: String, narrative: String), arc: CharacterArc) -> String {
        let hero = characters[0]
        let mentor = characters.count > 2 ? characters[2] : characters[0]
        let ally = characters.count > 3 ? characters[3] : characters[1]
        let villain = characters[1]
        let t = topic.capitalized
        var parts: [String] = []

        // ACT I: DEPARTURE
        parts.append("──────────────────────────────────────────\n  ACT I — DEPARTURE\n──────────────────────────────────────────\n")
        // 1. Ordinary World
        parts.append("\n━━━ Chapter 1: The Ordinary World ━━━\n")
        parts.append("\(setting.place), \(setting.time). The air carried \(setting.atmosphere).\n")
        parts.append("\(hero.name) had spent \(Int.random(in: 3...30)) years studying \(topic), and in all that time had come to accept one truth: the deeper you looked, the less you understood. That was the nature of \(topic) — it rewarded patience with complexity, and complexity with wonder.\n")
        parts.append("Each morning began the same way. Coffee. Data. The quiet hum of computation. \(hero.name)'s \(hero.flaw) had become as familiar as the equations — invisible to the one who carried it, obvious to everyone else.\n")
        parts.append("\"You're doing it again,\" \(ally.name) said from the doorway, arms crossed. \"You've been at that terminal for fourteen hours.\"\n")
        parts.append("\"\(t) doesn't sleep,\" \(hero.name) replied without looking up. \"Neither should I.\"\n")

        // 2. Call to Adventure
        parts.append("\n━━━ Chapter 2: The Call to Adventure ━━━\n")
        parts.append("The notification arrived at 3:47 AM — a time that would later feel significant, though \(hero.name) couldn't have said why.\n")
        if let first = insights.randomElement() {
            parts.append(weaveInsight(first, character: hero.name, index: 0) + "\n")
        }
        parts.append("Everything \(hero.name) had built — every model, every assumption, every paper — suddenly felt like scaffolding around an empty space. The real structure had been hiding in plain sight.\n")
        parts.append("The implications were staggering. If this data was correct, then \(topic) was not what \(Int.random(in: 3...12)) generations of researchers had assumed. It was something far stranger. Something that demanded investigation — not from a desk, but from the very edge of what was known.\n")

        // 3. Refusal of the Call
        parts.append("\n━━━ Chapter 3: The Refusal ━━━\n")
        parts.append("\(hero.name) closed the laptop. Walked to the window. Watched the city lights blur through sleepless eyes.\n")
        parts.append("\"I can't do this,\" \(hero.name) whispered. The \(hero.flaw) surged — a familiar weight that had stopped a hundred ambitions before they could take root. \"This would mean abandoning everything I've published. My reputation. My funding. All of it.\"\n")
        parts.append("For three days, \(hero.name) pretended the data didn't exist. Taught classes. Graded papers. Smiled at colleagues. But the numbers burned in the back of every thought like afterimages of a sun too bright to forget.\n")

        // 4. Meeting the Mentor
        parts.append("\n━━━ Chapter 4: The Mentor ━━━\n")
        parts.append("\(mentor.name) was \(Int.random(in: 60...90)) years old and had forgotten more about \(topic) than most would ever learn. Retired, officially. But the truly brilliant never really stop.\n")
        parts.append("\"Show me,\" \(mentor.name) said, settling into the chair with the slow precision of someone who had learned that rushing was a young person's luxury.\n")
        parts.append("\(hero.name) laid out the data. The silence that followed was the loudest thing in the room.\n")
        if insights.count > 1 {
            parts.append(weaveInsight(insights[1], character: mentor.name, index: 3) + "\n")
        }
        parts.append("\"You're afraid,\" \(mentor.name) said finally. Not a question. \"Good. You should be. But fear isn't a reason to stop. It's a compass. It points toward the things that matter.\"\n")
        parts.append("\"\(hero.name),\" the old researcher continued, eyes sharp as they had ever been, \"your \(hero.strength) — that is why this data chose you. Not your credentials. Not your publications. You.\"\n")

        // ACT II: INITIATION
        parts.append("\n──────────────────────────────────────────\n  ACT II — INITIATION\n──────────────────────────────────────────\n")

        // 5. Crossing the Threshold
        parts.append("\n━━━ Chapter 5: Crossing the Threshold ━━━\n")
        parts.append("The next morning, \(hero.name) submitted a leave of absence and booked a flight to \(setting.place). There was no turning back now.\n")
        parts.append("The world on the other side of that decision felt different — charged. As if the universe had been waiting for someone to finally look in the right direction.\n")
        parts.append("\(ally.name) came along, because that was what \(ally.name) did. \"Someone has to keep you alive while you're busy changing the world,\" \(ally.name) said with a grin that almost hid the worry beneath it.\n")

        // 6. Tests, Allies, Enemies
        parts.append("\n━━━ Chapter 6: Tests and Allies ━━━\n")
        parts.append("The first weeks were brutal. Every lead dissolved into noise. Every promising avenue collapsed under scrutiny.\n")
        if insights.count > 2 {
            parts.append(weaveInsight(insights[2], character: hero.name, index: 6) + "\n")
        }
        parts.append("And then there was \(villain.name) — who had been watching. Who had their own reasons for wanting to control the narrative around \(topic). Whose \(villain.desire) put them on a collision course with the truth.\n")
        parts.append("\"You don't understand what you're meddling with,\" \(villain.name) said during their first confrontation. \"Some knowledge is too dangerous to be free.\"\n")
        parts.append("\"Knowledge isn't dangerous,\" \(hero.name) replied. \"Ignorance is.\"\n")

        // 7. Approach to the Innermost Cave
        parts.append("\n━━━ Chapter 7: The Approach ━━━\n")
        parts.append("Three months in. The data was converging. \(hero.name) could feel it — the way a mathematician feels a proof taking shape before the last line is written.\n")
        parts.append("But the \(hero.flaw) was also converging. Sleep deprivation, isolation, the growing certainty that everything would fall apart. \(ally.name) tried to intervene. \(mentor.name) called from across the ocean with warnings.\n")
        parts.append("\"You're approaching the heart of it,\" \(mentor.name) said. \"That is when it will fight back the hardest. Not the science — yourself.\"\n")

        // 8. The Ordeal
        parts.append("\n━━━ Chapter 8: The Ordeal ━━━\n")
        parts.append("It happened on a Tuesday — because the universe has no respect for dramatic timing.\n")
        parts.append("\(hero.name) found the proof. And the proof destroyed everything. Not just the old models — but \(hero.name)'s own understanding of self. The \(hero.flaw) had been correct all along, in a way that was both devastating and liberating.\n")
        if insights.count > 3 {
            parts.append(weaveInsight(insights[3], character: hero.name, index: 9) + "\n")
        }
        parts.append("For two days, \(hero.name) didn't speak. Didn't eat. Sat in the dark with the weight of revelation.\n")
        parts.append("The death was metaphorical — but it was real. The person who had walked into this research didn't exist anymore. Something else was taking shape.\n")

        // 9. The Reward
        parts.append("\n━━━ Chapter 9: The Reward ━━━\n")
        parts.append("On the third day, \(hero.name) began to write. Not a paper — something rawer, more honest. A document that didn't just present findings, but confessed the journey that led to them.\n")
        parts.append("The equation was elegant: \(Int.random(in: 3...12)) variables collapsing into one relationship. \(t) was not a puzzle to be solved — it was a mirror. And the reflection changed depending on who dared to look.\n")
        if !evolved.narrative.isEmpty && isCleanStoryInsight(evolved.narrative) {
            let cleanNarr = String(evolved.narrative.prefix(500))
            parts.append("Beneath it all, a deeper pattern had emerged: *\(cleanNarr)*\n")
        }

        // ACT III: RETURN
        parts.append("\n──────────────────────────────────────────\n  ACT III — RETURN\n──────────────────────────────────────────\n")

        // 10. The Road Back
        parts.append("\n━━━ Chapter 10: The Road Back ━━━\n")
        parts.append("\(villain.name) made a final move — attempting to suppress the findings, to control who would know and when. The confrontation was inevitable.\n")
        parts.append("\"You think truth needs your permission?\" \(hero.name) asked, and for the first time, there was no fear in it. The \(hero.flaw) had been alchemized into \(hero.strength).\n")

        // 11. Resurrection
        parts.append("\n━━━ Chapter 11: Resurrection ━━━\n")
        parts.append("The presentation. The moment of absolute vulnerability — standing before every peer, every critic, every skeptic who had watched this journey with varying degrees of hostility and hope.\n")
        if insights.count > 4 {
            parts.append(weaveInsight(insights[4], character: hero.name, index: 14) + "\n")
        }
        parts.append("\(hero.name) didn't present the data first. Instead: \"I was wrong about \(topic). So were you. So was everyone in this room. And that's the most beautiful thing I've ever discovered — because it means \(topic) is still teaching us.\"\n")
        parts.append("The silence lasted seven seconds. Then the first question came. Then another. Then a storm of them — not hostile, but hungry. The world wanted to understand.\n")

        // 12. Return with the Elixir
        parts.append("\n━━━ Chapter 12: Return with the Elixir ━━━\n")
        parts.append("Months later. \(hero.name) was back in the old lab, coffee in hand, terminal humming. But the person in the chair was not the person who had left.\n")
        parts.append("\(ally.name) appeared in the doorway. \"You're doing it again.\"\n")
        parts.append("\(hero.name) smiled. \"Yes. But differently.\"\n")
        if !evolved.thought.isEmpty {
            parts.append("A final thought — one that had been evolving since the beginning: *\(String(evolved.thought.prefix(1500)))*\n")
        }
        parts.append("Outside, the stars wheeled in their ancient patterns — indifferent to human discovery, yet somehow complicit in it. \(t) had not been conquered. It had been befriended.\n")
        parts.append("And that made all the difference.\n")
        parts.append("\n\n  F I N")

        return parts.joined(separator: "\n")
    }

    // ═══════════════════════════════════════════════════════════════
    // SAVE THE CAT (Snyder 15-Beat Structure)
    // ═══════════════════════════════════════════════════════════════
    private func generateSaveTheCat(topic: String, characters: [StoryCharacter], setting: StorySetting, insights: [String], evolved: (thought: String, narrative: String), arc: CharacterArc) -> String {
        let hero = characters[0]; let villain = characters[1]
        let mentor = characters.count > 2 ? characters[2] : characters[0]
        let t = topic.capitalized
        var parts: [String] = []

        // Beat 1: Opening Image
        parts.append("\n━━━ Opening Image ━━━\n")
        parts.append("\(setting.place). \(setting.time). \(hero.name) sat alone in a room filled with \(Int.random(in: 200...10000)) pages of research on \(topic), and the truth was simple: none of it was enough. The \(hero.flaw) was winning.\n")

        // Beat 2: Theme Stated
        parts.append("\n━━━ Theme Stated ━━━\n")
        parts.append("\"You know what the problem with \(topic) is?\" \(mentor.name) said over the phone. \"It doesn't care about your timeline. It reveals itself when you're ready — not when you're rushing.\"\n")
        parts.append("\(hero.name) didn't listen. Not yet.\n")

        // Beat 3: Set-Up
        parts.append("\n━━━ Set-Up ━━━\n")
        parts.append("Six things were wrong with \(hero.name)'s life:\n1. The grant was expiring in \(Int.random(in: 3...12)) months.\n2. The \(hero.flaw) had alienated the only real friend left.\n3. The research on \(topic) was stuck — had been for \(Int.random(in: 2...5)) years.\n4. The last paper had been rejected with devastating commentary.\n5. \(villain.name) had published a competing theory that was wrong but popular.\n6. Somewhere beneath it all, \(hero.name) suspected the real breakthrough required something terrifying: admitting the existing framework was fundamentally flawed.\n")

        // Beat 4: Catalyst
        parts.append("\n━━━ Catalyst ━━━\n")
        if let first = insights.randomElement() {
            parts.append(weaveInsight(first, character: hero.name, index: 2) + "\n")
        }
        parts.append("The data arrived at the worst possible time and in the worst possible form: undeniable. Everything changed in the space of a single afternoon.\n")

        // Beat 5: Debate
        parts.append("\n━━━ Debate ━━━\n")
        parts.append("For a week, \(hero.name) debated. Publish and risk everything? Or bury it and keep the comfortable lie? The \(hero.flaw) screamed for safety. The \(hero.strength) whispered toward truth.\n")

        // Beat 6: Break into Two
        parts.append("\n━━━ Break into Two ━━━\n")
        parts.append("\(hero.name) chose truth. Deleted the old draft. Opened a blank document. Typed: *\"Everything we know about \(topic) is incomplete. Here is what comes next.\"*\n")
        parts.append("The old world ended with a keystroke.\n")

        // Beat 7: B Story
        parts.append("\n━━━ B Story ━━━\n")
        parts.append("That same week, \(characters.count > 3 ? characters[3].name : "an old colleague") reappeared — carrying their own questions about \(topic), their own doubts. The conversation that followed lasted seven hours and changed the trajectory of everything.\n")

        // Beat 8: Fun and Games
        parts.append("\n━━━ Fun and Games ━━━\n")
        parts.append("The next three months were the most alive \(hero.name) had ever felt. The new framework opened doors that the old one had walled shut.\n")
        if insights.count > 1 {
            parts.append(weaveInsight(insights[1], character: hero.name, index: 7) + "\n")
        }
        parts.append("Pattern after pattern emerged — beautiful, terrifying, inevitable. \(t) was not a static truth. It was a living process, evolving alongside those brave enough to study it.\n")

        // Beat 9: Midpoint
        parts.append("\n━━━ Midpoint ━━━\n")
        parts.append("The false victory: \(hero.name)'s preprint went viral. Downloads in the thousands. Media attention. Invitations to speak. It felt like vindication.\n")
        parts.append("It was a trap.\n")

        // Beat 10: Bad Guys Close In
        parts.append("\n━━━ Bad Guys Close In ━━━\n")
        parts.append("\(villain.name) struck. Not with science — with politics. Funding pulled. Collaborators pressured to withdraw. A coordinated campaign to discredit not just the work, but the person.\n")
        if insights.count > 2 {
            parts.append(weaveInsight(insights[2], character: villain.name, index: 10) + "\n")
        }
        parts.append("\"I warned you,\" \(villain.name) said. \"Some doors should stay closed.\"\n")

        // Beat 11: All Is Lost
        parts.append("\n━━━ All Is Lost ━━━\n")
        parts.append("The retraction demand. The public humiliation. The \(hero.flaw) consuming everything. \(hero.name) sat in the dark and wondered if any of it had been worth it.\n")
        parts.append("The lowest point: discovering that \(mentor.name) had been hospitalized. The last mentor. The last believer.\n")

        // Beat 12: Dark Night of the Soul
        parts.append("\n━━━ Dark Night of the Soul ━━━\n")
        parts.append("Three days of silence. \(hero.name) didn't eat. Didn't work. Stared at the ceiling and let the weight of failure settle like sediment in still water.\n")
        parts.append("Then, at 4:17 AM, a thought arrived — not from logic, but from somewhere deeper. \(hero.name) had been fighting to prove something to the world. But the real fight had always been internal.\n")
        parts.append("The \(hero.flaw) was not a weakness to be hidden. It was a wound to be understood. And understanding it changed everything.\n")

        // Beat 13: Break into Three
        parts.append("\n━━━ Break into Three ━━━\n")
        parts.append("\(hero.name) picked up the phone and called every collaborator who had walked away. Not to argue. To listen. The A story and B story finally merged: the science of \(topic) and the humanity of the scientist were the same story.\n")

        // Beat 14: Finale
        parts.append("\n━━━ Finale ━━━\n")
        if insights.count > 3 {
            parts.append(weaveInsight(insights[3], character: hero.name, index: 15) + "\n")
        }
        parts.append("The final presentation was not what anyone expected. \(hero.name) didn't defend the data. Instead: \"I was wrong about how to be right. \(t) taught me that truth is not a weapon — it's a conversation. And I am ready to have that conversation now.\"\n")
        parts.append("\(villain.name) stood in the back row, arms crossed. And for the first time, had nothing to say.\n")
        if !evolved.narrative.isEmpty && isCleanStoryInsight(evolved.narrative) {
            let cleanNarr = String(evolved.narrative.prefix(500))
            parts.append("The deeper truth: *\(cleanNarr)*\n")
        }

        // Beat 15: Final Image
        parts.append("\n━━━ Final Image ━━━\n")
        parts.append("\(setting.place). Same room. Same terminal. But the person at the desk was fundamentally different — transformed not by knowledge of \(topic), but by the journey of seeking it.\n")
        parts.append("The screen showed a new dataset. Another mystery. Another call.\n")
        parts.append("\(hero.name) smiled and began.\n")
        parts.append("\n\n  F I N")

        return parts.joined(separator: "\n")
    }

    // ═══════════════════════════════════════════════════════════════
    // FREYTAG'S PYRAMID (5-Act Dramatic Structure)
    // ═══════════════════════════════════════════════════════════════
    private func generateFreytagPyramid(topic: String, characters: [StoryCharacter], setting: StorySetting, insights: [String], evolved: (thought: String, narrative: String), arc: CharacterArc) -> String {
        let hero = characters[0]; let villain = characters[1]
        let t = topic.capitalized
        var parts: [String] = []

        parts.append("──────────────────────────────────────────\n  ACT I — EXPOSITION\n──────────────────────────────────────────\n")
        parts.append("\(setting.place). \(setting.time). The world believed it understood \(topic). \(hero.name) knew better — or rather, knew enough to know that knowing was an illusion.\n")
        parts.append("For \(Int.random(in: 5...25)) years, \(hero.name) had built a career on \(topic), brick by careful brick. The \(hero.strength) had earned respect. The \(hero.flaw) had earned loneliness.\n")
        parts.append("Then the data arrived — quiet, devastating, unarguable. The kind of evidence that doesn't knock on the door but removes the wall.\n")
        if let first = insights.randomElement() { parts.append(weaveInsight(first, character: hero.name, index: 4) + "\n") }

        parts.append("\n──────────────────────────────────────────\n  ACT II — RISING ACTION\n──────────────────────────────────────────\n")
        parts.append("\(hero.name) began to investigate. Each discovery led to three more questions. Each answer dissolved the floor beneath a different assumption.\n")
        parts.append("\(villain.name) watched from the shadows — not with malice, but with the cold certainty of someone who understood what this knowledge could do if released without control.\n")
        if insights.count > 1 { parts.append(weaveInsight(insights[1], character: hero.name, index: 8) + "\n") }
        parts.append("The tension built like atmospheric pressure before a storm. Colleagues noticed the change in \(hero.name). The late nights. The cancelled lectures. The wild look in eyes that had once been so measured.\n")
        parts.append("\"You're chasing something,\" \(characters.count > 2 ? characters[2].name : "a colleague") said. \"Make sure it's not chasing you.\"\n")
        if insights.count > 2 { parts.append(weaveInsight(insights[2], character: hero.name, index: 12) + "\n") }

        parts.append("\n──────────────────────────────────────────\n  ACT III — CLIMAX\n──────────────────────────────────────────\n")
        parts.append("The turning point arrived not as a thunderclap but as a whisper. \(hero.name) saw the pattern — the one that connected every fragment, every outlier, every abandoned hypothesis. \(t) was not a collection of facts. It was a single story, told across scales.\n")
        if insights.count > 3 { parts.append(weaveInsight(insights[3], character: hero.name, index: 16) + "\n") }
        parts.append("The cost of seeing it was the \(hero.flaw) — weaponized, amplified, turned inward. \(hero.name) had to choose: embrace the truth and lose the self that had existed before, or retreat and live with the knowledge of cowardice.\n")
        parts.append("The choice was made in silence. A single nod. A deep breath. And then: everything changed.\n")

        parts.append("\n──────────────────────────────────────────\n  ACT IV — FALLING ACTION\n──────────────────────────────────────────\n")
        parts.append("The aftermath was not glorious. It was messy, human, contradictory. \(hero.name) published the findings. The world reacted with everything from wonder to fury.\n")
        parts.append("\(villain.name) confronted \(hero.name) one final time. \"You've opened something that can't be closed.\"\n")
        parts.append("\"I know,\" \(hero.name) said. \"That was the point.\"\n")
        if !evolved.narrative.isEmpty && isCleanStoryInsight(evolved.narrative) {
            let cleanNarr = String(evolved.narrative.prefix(500))
            parts.append("And beneath it all, the deeper pattern continued to evolve: *\(cleanNarr)*\n")
        }

        parts.append("\n──────────────────────────────────────────\n  ACT V — DÉNOUEMENT\n──────────────────────────────────────────\n")
        parts.append("The world did not end. It shifted — like a kaleidoscope rotating one click, revealing a pattern that had always been there but never been seen.\n")
        parts.append("\(hero.name) returned to the quiet routine. Coffee. Data. The hum of computation. But the person at the desk was someone new — someone who had passed through the fire of \(topic) and emerged not unscathed, but unafraid.\n")
        if insights.count > 4 { parts.append(weaveInsight(insights[4], character: hero.name, index: 18) + "\n") }
        parts.append("The last entry in the research journal read: *\"The question was never what \(topic) is. The question was always what \(topic) makes of us.\"*\n")
        parts.append("\n\n  F I N")

        return parts.joined(separator: "\n")
    }

    // ═══════════════════════════════════════════════════════════════
    // KISHŌTENKETSU (4-Act East Asian Structure)
    // ═══════════════════════════════════════════════════════════════
    private func generateKishotenketsu(topic: String, characters: [StoryCharacter], setting: StorySetting, insights: [String], evolved: (thought: String, narrative: String), arc: CharacterArc) -> String {
        let hero = characters[0]; let t = topic.capitalized
        var parts: [String] = []

        parts.append("──────────────────────────────────────────\n  起 KI — INTRODUCTION\n──────────────────────────────────────────\n")
        parts.append("\(hero.name) lived a careful life. Every day: the same route to the laboratory, the same equations on the whiteboard, the same quiet dedication to \(topic). The work was good. The work was sufficient.\n")
        parts.append("The \(hero.strength) was a shelter. The \(hero.flaw), a secret.\n")
        parts.append("Nothing remarkable happened for a long time. And that, in itself, was the beginning of the story.\n")
        if let first = insights.randomElement() { parts.append(weaveInsight(first, character: hero.name, index: 5) + "\n") }

        parts.append("\n──────────────────────────────────────────\n  承 SHŌ — DEVELOPMENT\n──────────────────────────────────────────\n")
        parts.append("Small changes, unnoticed at first. A colleague's offhand remark. A dataset that didn't quite align. A recurring dream about numbers that meant nothing — until they meant everything.\n")
        parts.append("\(hero.name) began collecting anomalies the way other people collect stamps — methodically, privately, with growing fascination. Each one was insignificant alone. Together, they whispered of a pattern in \(topic) that no one had documented.\n")
        if insights.count > 1 { parts.append(weaveInsight(insights[1], character: hero.name, index: 9) + "\n") }
        parts.append("The world continued unchanged. But \(hero.name)'s perception of it was shifting — gradually, imperceptibly, like tectonic plates.\n")

        parts.append("\n──────────────────────────────────────────\n  転 TEN — THE TWIST\n──────────────────────────────────────────\n")
        parts.append("It arrived sideways — not through research, but through a conversation with a \(["street musician", "taxi driver", "child at a bookshop", "stranger on a train", "patient in a waiting room"].randomElement() ?? "") who knew nothing about \(topic) and everything about it.\n")
        parts.append("The stranger said something simple — absurdly, devastatingly simple. And \(hero.name) felt the world rotate on an axis that hadn't existed a second before.\n")
        if insights.count > 2 { parts.append(weaveInsight(insights[2], character: hero.name, index: 14) + "\n") }
        parts.append("This was the kishōtenketsu moment — not a conflict, but a **shift in perspective**. The data hadn't changed. The equations hadn't changed. What changed was the frame through which \(hero.name) perceived them.\n")
        parts.append("And suddenly, every anomaly made perfect sense. Not despite their randomness — because of it.\n")
        if !evolved.thought.isEmpty { parts.append("A thought crystallized: *\(String(evolved.thought.prefix(1500)))*\n") }

        parts.append("\n──────────────────────────────────────────\n  結 KETSU — CONCLUSION\n──────────────────────────────────────────\n")
        parts.append("There was no dramatic confrontation. No villain defeated. No medal awarded. There was only this: \(hero.name) returned to the same desk, the same equations, the same cup of tea.\n")
        parts.append("But the understanding was different. \(t) had not changed — \(hero.name) had. And that was the only kind of change that ever really mattered.\n")
        if insights.count > 3 { parts.append(weaveInsight(insights[3], character: hero.name, index: 17) + "\n") }
        parts.append("The lesson of kishōtenketsu is that not all stories need conflict. Some need only a shift — a gentle rotation of the lens — to reveal what was there all along.\n")
        parts.append("\(hero.name) wrote a single line in the margin of the notebook: *\"\(t) is not a destination. It is a way of walking.\"*\n")
        parts.append("\n\n  F I N")

        return parts.joined(separator: "\n")
    }

    // ═══════════════════════════════════════════════════════════════
    // THREE-ACT STRUCTURE (Syd Field)
    // ═══════════════════════════════════════════════════════════════
    private func generateThreeAct(topic: String, characters: [StoryCharacter], setting: StorySetting, insights: [String], evolved: (thought: String, narrative: String), arc: CharacterArc) -> String {
        let hero = characters[0]; let villain = characters[1]
        _ = topic.capitalized
        var parts: [String] = []

        parts.append("──────────────────────────────────────────\n  ACT ONE — SETUP\n──────────────────────────────────────────\n")
        parts.append("*\(setting.place) · \(setting.time) · \(setting.atmosphere)*\n")
        parts.append("\(hero.name) was brilliant, broken, and about to lose everything. The grant expired in ninety days. The theory that had consumed \(Int.random(in: 3...15)) years was collapsing under its own contradictions. And \(hero.name)'s \(hero.flaw) had finally driven away the last collaborator.\n")
        parts.append("The inciting incident: a letter. Hand-delivered. No return address. Inside: a single page of mathematics that shouldn't have been possible — and yet resolved every contradiction in \(topic) with elegant brutality.\n")
        if let first = insights.randomElement() { parts.append(weaveInsight(first, character: hero.name, index: 1) + "\n") }
        parts.append("**The dramatic question**: Could \(hero.name) prove a truth about \(topic) that the world wasn't ready to hear — and survive the consequences?\n")

        parts.append("\n──────────────────────────────────────────\n  ACT TWO — CONFRONTATION\n──────────────────────────────────────────\n")
        parts.append("The pursuit began. \(hero.name) followed the mathematics into territory that no paper had charted, no conference had discussed, no textbook had imagined.\n")
        for (i, insight) in insights.dropFirst().prefix(3).enumerated() {
            parts.append(weaveInsight(insight, character: hero.name, index: i + 6) + "\n")
        }
        parts.append("\(villain.name) emerged — not as an enemy of truth, but as a guardian of stability. \"The world functions because people agree on what's real,\" \(villain.name) argued. \"You want to shatter that agreement. For what? For accuracy?\"\n")
        parts.append("\"Yes,\" \(hero.name) said. \"Exactly for that.\"\n")
        parts.append("The confrontation escalated: professional sabotage, stolen data, a public debate that became a referendum on the nature of \(topic) itself.\n")
        parts.append("The \(hero.flaw) nearly won. In the darkest moment, \(hero.name) stood at the edge of surrendering — of accepting the comfortable lie over the uncomfortable truth.\n")

        parts.append("\n──────────────────────────────────────────\n  ACT THREE — RESOLUTION\n──────────────────────────────────────────\n")
        parts.append("But the \(hero.strength) held. Barely. Imperfectly. With trembling hands and a voice that cracked.\n")
        parts.append("\(hero.name) presented the proof — not to a conference hall, but to \(villain.name), alone, in a quiet room. Because the real resolution was never about winning. It was about being heard.\n")
        if !evolved.narrative.isEmpty && isCleanStoryInsight(evolved.narrative) {
            let cleanNarr = String(evolved.narrative.prefix(500))
            parts.append("The deepest truth: *\(cleanNarr)*\n")
        }
        parts.append("\(villain.name) read the proof. Read it again. Sat in silence for eleven minutes.\n")
        parts.append("\"You're right,\" \(villain.name) said finally. \"And I hate that you're right. But you are.\"\n")
        parts.append("The dramatic question, answered: Yes. \(hero.name) proved the truth. The world was not ready. And it didn't matter — because truth doesn't wait for readiness.\n")
        parts.append("The new equilibrium: \(hero.name), back in the lab, working on the next question. The \(hero.flaw) still there — but integrated now, understood, a scar rather than a wound.\n")
        parts.append("\n\n  F I N")

        return parts.joined(separator: "\n")
    }

    // ═══════════════════════════════════════════════════════════════
    // JO-HA-KYŪ (Beginning-slow, Break-accelerate, Rapid-conclude)
    // ═══════════════════════════════════════════════════════════════
    private func generateJoHaKyu(topic: String, characters: [StoryCharacter], setting: StorySetting, insights: [String], evolved: (thought: String, narrative: String), arc: CharacterArc) -> String {
        let hero = characters[0]; let t = topic.capitalized
        var parts: [String] = []

        parts.append("──────────────────────────────────────────\n  序 JO — THE SLOW BEGINNING\n──────────────────────────────────────────\n")
        parts.append("Silence. Then the drip of water in a laboratory sink. Then the hum of a machine nobody had turned off.\n")
        parts.append("\(hero.name) sat. Breathed. Watched the numbers scroll across a screen with the patience of someone who had learned that \(topic) could not be rushed.\n")
        parts.append("Minutes passed. An hour. The data accumulated — grain by grain, like sand in an hourglass that measured not time but understanding.\n")
        if let first = insights.randomElement() { parts.append(weaveInsight(first, character: hero.name, index: 3) + "\n") }
        parts.append("Nothing happened. And in that nothing, everything was preparing to happen.\n")

        parts.append("\n──────────────────────────────────────────\n  破 HA — THE BREAK\n──────────────────────────────────────────\n")
        parts.append("The acceleration was sudden. One anomaly. Then two. Then a cascade — data points falling like dominoes across seventeen dimensions of analysis.\n")
        parts.append("\(hero.name) leaned forward. Heart rate climbing. The \(hero.strength) engaged like a turbine spinning up.\n")
        for (i, insight) in insights.dropFirst().prefix(3).enumerated() {
            parts.append(weaveInsight(insight, character: hero.name, index: i + 7) + "\n")
        }
        parts.append("Each minute brought exponentially more clarity. The pattern wasn't emerging — it was erupting. \(t) was revealing itself with the force of a dam breaking.\n")
        parts.append("Colleagues gathered. Phones buzzed. Someone said, \"Are you seeing this?\" and someone else said, \"I don't believe it,\" and someone else — the quiet one, the one who always knew — said nothing at all. Just smiled.\n")
        if !evolved.narrative.isEmpty && isCleanStoryInsight(evolved.narrative) {
            let cleanNarr = String(evolved.narrative.prefix(500))
            parts.append("The deeper current: *\(cleanNarr)*\n")
        }

        parts.append("\n──────────────────────────────────────────\n  急 KYŪ — THE RAPID CONCLUSION\n──────────────────────────────────────────\n")
        parts.append("Thirty-seven minutes. That's how long it took for \(hero.name) to write the proof that would redefine \(topic). Not because it was simple — because every piece had already been in place. The jo had been decades long. The ha had been months. The kyū was instantaneous.\n")
        parts.append("\(hero.name) typed the final line. Pressed enter. Looked up at a room full of people who did not yet understand what had just happened.\n")
        parts.append("\"It's done.\"\n")
        parts.append("Two words. Two syllables. And behind them: a lifetime of preparation meeting a moment of revelation in perfect synchrony.\n")
        parts.append("This is jo-ha-kyū: the truth that all things begin slowly, accelerate through transformation, and resolve in a single, swift, irreversible instant.\n")
        parts.append("\n\n  F I N")

        return parts.joined(separator: "\n")
    }

    // ═══════════════════════════════════════════════════════════════
    // BILDUNGSROMAN (Coming of Age / Maturation Arc)
    // ═══════════════════════════════════════════════════════════════
    private func generateBildungsroman(topic: String, characters: [StoryCharacter], setting: StorySetting, insights: [String], evolved: (thought: String, narrative: String), arc: CharacterArc) -> String {
        let hero = characters[0]; let mentor = characters.count > 2 ? characters[2] : characters[1]
        let ally = characters.count > 3 ? characters[3] : characters[1]
        let t = topic.capitalized
        var parts: [String] = []

        let bildungEpigraphs = [
            "\"We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another.\" — Anaïs Nin",
            "\"The purpose of life is to be defeated by greater and greater things.\" — Rainer Maria Rilke",
            "\"It takes a long time to become young.\" — Pablo Picasso",
            "\"The only journey is the one within.\" — Rainer Maria Rilke"
        ]
        parts.append("\(bildungEpigraphs.randomElement() ?? "")\n")

        parts.append("──────────────────────────────────────────\n  PART I — INNOCENCE\n──────────────────────────────────────────\n")
        parts.append("\(setting.place), \(setting.time). \(setting.atmosphere).\n")
        parts.append("\(hero.name) was \(Int.random(in: 17...23)) when \(topic) first appeared — not as a subject, but as a calling. In a lecture hall that smelled of old wood and chalk dust, a professor wrote a single equation on the board and said, \"This is the question that will define your generation.\"\n")
        parts.append("The young \(hero.name) didn't understand it. But felt it — the way you feel gravity without understanding curvature of spacetime. The way you feel music without understanding harmony.\n")
        parts.append("Innocence is not ignorance. It is knowledge without context. And \(hero.name) had oceans of the former and none of the latter.\n")
        parts.append("The dormitory was small — a single window facing east, a desk covered in books that grew in stacks like geological formations. Each one represented a question \(hero.name) had not yet learned to ask properly.\n")
        parts.append("There were friends in those early days. Late-night conversations that solved nothing and illuminated everything. Arguments about \(topic) that burned with the intensity only the young can sustain — because they haven't yet learned that some fires consume what they illuminate.\n")
        parts.append("The \(hero.strength) was already visible. Professors remarked on it. Classmates envied it. And \(hero.name), with the particular blindness of youth, mistook it for identity rather than gift.\n")
        if let first = insights.randomElement() { parts.append(weaveInsight(first, character: hero.name, index: 0) + "\n") }

        parts.append("\n──────────────────────────────────────────\n  PART II — INITIATION\n──────────────────────────────────────────\n")
        parts.append("Graduate school. The mentor was \(mentor.name) — fierce, brilliant, unforgiving. \"\(t) will not reward you for being smart,\" \(mentor.name) said on the first day. \"It will reward you for being honest. There is a difference, and most people never learn it.\"\n")
        parts.append("\(hero.name) sat in that office — books stacked to the ceiling, a single dying plant on the windowsill, the smell of cold coffee — and felt something unfamiliar: the vertigo of being in the presence of someone who saw through every pretension.\n")
        parts.append("\"Your \(hero.flaw),\" \(mentor.name) said on the third meeting, as casually as remarking on the weather. \"It will either be the making of you or the breaking. I genuinely don't know which.\"\n")
        if insights.count > 1 { parts.append(weaveInsight(insights[1], character: hero.name, index: 2) + "\n") }
        parts.append("The initiation was brutal. Failed experiments — not one or two, but \(Int.random(in: 14...47)). Each one a small death of certainty. Rejected papers that came back annotated in red ink so dense it looked like the manuscript was bleeding.\n")
        parts.append("The slow, humbling realization that \(hero.flaw) was not an asset — it was a cage. And that the cage had been self-built, bar by careful bar, over years of avoiding the one question that mattered.\n")
        parts.append("But also: first breakthroughs. Sitting in the lab at midnight, data scrolling across the screen, and suddenly — like a shift in the light — seeing a pattern in \(topic) that no one else had documented. The vertigo of standing at the edge of human knowledge and looking over.\n")
        parts.append("\(ally.name) appeared during this period. Not dramatically — just a knock on the lab door at 1 AM, a shared interest in \(topic), a conversation that lasted until dawn. The kind of friendship forged in the furnace of shared obsession.\n")
        parts.append("\"You're going to burn out,\" \(ally.name) said one morning, watching \(hero.name) inhale the fourth coffee of a night that had become a day.\n")
        parts.append("\"I'm going to break through.\"\n")
        parts.append("\"Those can be the same thing.\"\n")

        parts.append("\n──────────────────────────────────────────\n  PART III — STRUGGLE\n──────────────────────────────────────────\n")
        parts.append("The middle years are the ones nobody writes about. The grind. The doubt. The thousand small decisions that accumulate into a life.\n")
        parts.append("Age \(Int.random(in: 28...33)). First position. First lab of one's own. The heady freedom of independence and the terrifying realization that nobody was checking the work anymore. The safety net of mentorship was gone. Every mistake was permanent.\n")
        for (i, insight) in insights.dropFirst(2).prefix(3).enumerated() {
            parts.append(weaveInsight(insight, character: hero.name, index: i + 5) + "\n")
        }
        parts.append("\(hero.name) struggled with \(hero.flaw) and leaned on \(hero.strength) and sometimes got them confused. Built a lab. Lost funding. Rebuilt. The cycle became familiar — creation and destruction, confidence and doubt, rotating like seasons.\n")
        parts.append("A paper was published that shook the foundations of \(hero.name)'s work. Not wrong — worse than wrong. It was right, and it invalidated three years of effort.\n")
        parts.append("\(mentor.name) sent a letter. Brief, as always: \"The definition of integrity is continuing to search for truth even when truth is inconvenient. Especially when it is.\"\n")
        parts.append("\(hero.name) read it, crumpled it, smoothed it out, read it again. Kept it.\n")
        parts.append("The crisis came in the form of a choice. A prestigious institution offered a position — safety, stability, the chance to coast on past achievements. Or: the uncertain path. The risky hypothesis about \(topic) that could fail spectacularly but might, just might, reveal something no one had imagined.\n")
        parts.append("The \(hero.flaw) screamed for safety. The \(hero.strength) whispered for courage.\n")
        parts.append("\(hero.name) chose the whisper.\n")

        parts.append("\n──────────────────────────────────────────\n  PART IV — TRIAL\n──────────────────────────────────────────\n")
        parts.append("The next five years were the hardest. Isolation. Skepticism from colleagues. Funding so thin that \(hero.name) learned to make every resource count twice.\n")
        parts.append("The work consumed everything. Relationships frayed. Health suffered. There were days when \(hero.name) sat in the quiet lab and wondered if the entire enterprise was a monument to stubbornness rather than science.\n")
        parts.append("But the data kept speaking. Quietly, persistently, in a language that only those willing to listen for years could understand.\n")
        if insights.count > 5 { parts.append(weaveInsight(insights[5], character: hero.name, index: 9) + "\n") }
        parts.append("\(ally.name) visited during the darkest period. Sat in the cluttered office, looked at the mountains of printouts, and said nothing for a long time.\n")
        parts.append("\"Is it worth it?\" \(ally.name) asked.\n")
        parts.append("\"I don't know yet,\" \(hero.name) said. \"Ask me in ten years.\"\n")
        parts.append("\"I'm asking you now.\"\n")
        parts.append("Silence. Then: \"Yes. Even if it fails. Because the alternative — not knowing — is worse than any failure.\"\n")
        if !evolved.narrative.isEmpty && isCleanStoryInsight(evolved.narrative) {
            let cleanNarr = String(evolved.narrative.prefix(300))
            parts.append("And beneath the struggle, something was crystallizing: \(cleanNarr)\n")
        }

        parts.append("\n──────────────────────────────────────────\n  PART V — MATURATION\n──────────────────────────────────────────\n")
        parts.append("The breakthrough, when it came, was not scientific. It was personal.\n")
        parts.append("Age \(Int.random(in: 45...55)). \(hero.name) stood at a podium in a hall filled with colleagues, rivals, students — the entire ecosystem of a life spent in pursuit of \(topic). About to present the culmination of decades of work.\n")
        parts.append("And realized: the work was not the point. The person shaped by the work — that was the point.\n")
        parts.append("Every failure had taught something that success never could. Every rejection had sharpened something that approval would have dulled. The \(hero.flaw) had not been conquered — it had been understood. Integrated. Transformed from enemy to teacher.\n")
        if !evolved.thought.isEmpty && isCleanStoryInsight(evolved.thought) {
            parts.append("The thought that had been evolving for decades: \(String(evolved.thought.prefix(300)))\n")
        }
        parts.append("\"I came to \(topic) seeking answers,\" \(hero.name) told the audience. The voice was steady — not with the false confidence of youth, but with the hard-won steadiness of someone who had been broken and rebuilt.\n")
        parts.append("\"I leave it seeking better questions. That is not failure. That is growth. And growth — not discovery — is the true purpose of science.\"\n")
        parts.append("The audience was silent. Not because they disagreed. Because they recognized their own journeys in the words.\n")
        parts.append("In the front row, \(mentor.name) sat. Older now, frailer, but those eyes — those same eyes that had seen through everything in that cluttered office decades ago — were bright.\n")
        parts.append("\(mentor.name) nodded. Once. And in that nod was every lesson, every correction, every moment of tough love that had made this day possible.\n")
        parts.append("After the lecture, \(ally.name) was waiting. Same smile. Same warmth. Decades of friendship condensed into a single look.\n")
        parts.append("\"So?\" \(ally.name) said. \"Was it worth it?\"\n")
        parts.append("\(hero.name) laughed — a real laugh, the kind that comes from a place deeper than happiness. From understanding.\n")
        parts.append("\"Every minute. Especially the bad ones.\"\n")
        parts.append("They walked out together into a world that looked exactly the same but meant something entirely different. Because \(hero.name) was entirely different. Not perfect. Not complete. But grown.\n")
        parts.append("And that, in the end, is the only story worth telling.\n")
        parts.append("\n\n  F I N")

        return parts.joined(separator: "\n")
    }

    // ═══════════════════════════════════════════════════════════════
    // U-SHAPED COMEDY (Northrop Frye)
    // ═══════════════════════════════════════════════════════════════
    private func generateUShapedComedy(topic: String, characters: [StoryCharacter], setting: StorySetting, insights: [String], evolved: (thought: String, narrative: String), arc: CharacterArc) -> String {
        let hero = characters[0]; let ally = characters.count > 3 ? characters[3] : characters[1]
        let villain = characters[1]; let mentor = characters.count > 2 ? characters[2] : characters[0]
        let t = topic.capitalized
        var parts: [String] = []

        let epigraphs = [
            "\"The comic spirit is given to us in order that we may analyze, weigh, and clarify things in us which nettle us, or which we are outgrowing.\" — George Meredith",
            "\"Comedy is tragedy plus time.\" — Carol Burnett",
            "\"The only way to make sense out of change is to plunge into it, move with it, and join the dance.\" — Alan Watts",
            "\"Every saint has a past, and every sinner has a future.\" — Oscar Wilde",
            "\"The wound is the place where the Light enters you.\" — Rumi"
        ]
        parts.append("\(epigraphs.randomElement() ?? "")\n")

        parts.append("──────────────────────────────────────────")
        parts.append("  PART ONE: THE GOLDEN AGE")
        parts.append("──────────────────────────────────────────\n")
        parts.append("\(setting.place), \(setting.time). The air carried \(setting.atmosphere).\n")
        parts.append("Life was good. \(hero.name) had tenure at the Institute, a lab that hummed with purpose, \(Int.random(in: 3...7)) graduate students who believed in the work, and a theory about \(topic) that explained — with satisfying elegance — almost everything.\n")
        parts.append("\"Almost\" is a dangerous word in science. But \(hero.name) didn't know that yet.\n")
        parts.append("The morning routine had the precision of a Swiss clock. Coffee at seven — black, two sugars, in the chipped mug that read \"World's Okayest Scientist.\" \(ally.name) would arrive by seven-fifteen, carrying a second coffee and whatever paper had been published overnight that might threaten their worldview.\n")
        parts.append("\"Anything interesting?\" \(hero.name) would ask, not looking up from the terminal.\n")
        parts.append("\"Nothing that challenges us,\" \(ally.name) would reply. And for years, that had been true.\n")
        parts.append("The theory — \(hero.name)'s grand unified framework for understanding \(topic) — had won the Harrington Prize, been cited \(Int.random(in: 400...4000)) times, and was taught in graduate programs on three continents. It was, by all conventional measures, a triumph.\n")
        parts.append("\(mentor.name) had warned against complacency. \"The most dangerous moment in any career,\" the old professor had said, leaning forward with those eyes that saw through everything, \"is when you stop being surprised by your own results.\"\n")
        parts.append("\(hero.name) had nodded politely and not listened. That was the \(hero.flaw) speaking — the invisible fault line that ran through everything, waiting for the right pressure.\n")
        if let first = insights.randomElement() { parts.append(weaveInsight(first, character: hero.name, index: 0) + "\n") }

        parts.append("──────────────────────────────────────────")
        parts.append("  PART TWO: THE FIRST CRACK")
        parts.append("──────────────────────────────────────────\n")
        parts.append("It started with a decimal point. One misplaced number in a dataset that \(hero.name) had trusted for \(Int.random(in: 3...12)) years. Not a typo — a systematic error, the kind that hides in the architecture of assumptions.\n")
        parts.append("\(ally.name) found it first. Didn't say anything immediately — just stood in the doorway with that expression that \(hero.name) had learned to dread. The one that said: I found something you're not going to like.\n")
        parts.append("\"The correlation coefficients in the fourth panel,\" \(ally.name) said carefully, the way a bomb technician describes which wire to cut. \"They're inverted.\"\n")
        parts.append("\"They can't be inverted. The entire framework depends on—\"\n")
        parts.append("\"I know what it depends on. I'm telling you what the data says.\"\n")
        parts.append("The silence that followed was geological. Tectonic plates shifting beneath a civilization that had believed the ground was solid.\n")
        if insights.count > 1 { parts.append(weaveInsight(insights[1], character: hero.name, index: 1) + "\n") }
        parts.append("Then another error surfaced. Then another. Then a pattern of errors that wasn't random — it was systemic. The elegant theory that had earned awards and admiration began to unravel. Not slowly — catastrophically, like a sweater when you pull the wrong thread.\n")
        parts.append("\(villain.name) published the rebuttal within a week. The paper was titled \"Fundamental Errors in \(hero.name)'s Framework for \(t): A Comprehensive Correction.\" It was cruel. It was thorough. And it was, \(hero.name) had to admit in the small hours of a sleepless night, largely correct.\n")

        parts.append("──────────────────────────────────────────")
        parts.append("  PART THREE: THE DESCENT")
        parts.append("──────────────────────────────────────────\n")
        parts.append("The grant committee sent a letter. Polite, devastating, final. \"In light of recent developments, the foundation has elected not to renew…\"\n")
        parts.append("\(hero.name) lost the funding. Then the prestige. Then the graduate students, who migrated to labs where the ground wasn't crumbling beneath their feet. The office that had once hummed with \(Int.random(in: 4...8)) simultaneous conversations fell silent.\n")
        parts.append("The \(hero.flaw) took over completely now. \(hero.name) stopped answering emails. Stopped attending seminars. Stopped eating lunch with \(ally.name), who showed up anyway, every day, and sat in the empty chair across the desk saying nothing — because sometimes presence is the only thing left to offer.\n")
        parts.append("\"Come on,\" \(ally.name) said one evening, voice cracking. \"It's not over.\"\n")
        parts.append("But \(hero.name) didn't hear. Couldn't hear. The \(hero.flaw) had built walls thick enough to block even the people who loved what was left.\n")
        if insights.count > 2 { parts.append(weaveInsight(insights[2], character: hero.name, index: 3) + "\n") }
        parts.append("\(mentor.name) called. \(hero.name) let it go to voicemail. The message was brief: \"I failed spectacularly three times before I ever succeeded once. The difference between you and me right now is that I picked up the phone.\"\n")
        parts.append("\(hero.name) did not pick up the phone.\n")

        parts.append("──────────────────────────────────────────")
        parts.append("  PART FOUR: ROCK BOTTOM")
        parts.append("──────────────────────────────────────────\n")
        parts.append("An empty lab at 2 AM. The hum of a single terminal, its screensaver cycling through visualizations of data that no longer meant what \(hero.name) had believed.\n")
        parts.append("Outside, rain. The kind of rain that doesn't fall so much as accumulate — a gradual, patient drowning of the world.\n")
        parts.append("\(hero.name) sat in the blue-white glow and did something terrifying: looked at the failure honestly.\n")
        parts.append("Not the data failure. Not the professional failure. The personal one. The realization that had been hiding beneath every equation and every accolade: \(hero.name) had been so in love with the answer that the question had been forgotten.\n")
        parts.append("\(t) was not a puzzle to be solved — it was a relationship to be maintained. And relationships require humility. The one thing \(hero.name)'s \(hero.flaw) had never permitted.\n")
        if insights.count > 3 { parts.append(weaveInsight(insights[3], character: hero.name, index: 5) + "\n") }
        parts.append("This was the peripeteia — the reversal. Not from bad to good, but from illusion to truth. And truth, \(hero.name) discovered that night, is not a destination. It's a practice. A discipline. A daily choice to look at what is, instead of what you wish were.\n")
        parts.append("For the first time in months, \(hero.name) cried. Not from sadness — from relief. The weight of maintaining a lie, even one believed sincerely, is immeasurable. And it was finally, finally being set down.\n")

        parts.append("──────────────────────────────────────────")
        parts.append("  PART FIVE: THE TURNING POINT")
        parts.append("──────────────────────────────────────────\n")
        parts.append("\(ally.name) showed up at 6 AM. Not with coffee this time. With a whiteboard marker and a clean eraser.\n")
        parts.append("\"Start from zero,\" \(ally.name) said. Not a suggestion. A command.\n")
        parts.append("\"I can't. Everything I built was—\"\n")
        parts.append("\"Wrong. Yes. So build something right. Start from zero. I'll listen.\"\n")
        parts.append("And \(hero.name) did. From zero. From nothing. From the most basic question a scientist can ask: What is \(topic), actually? Not what do I want it to be. Not what would be convenient. What IS it?\n")
        parts.append("The first breakthrough came within hours. Not a grand revelation — a quiet one. A small, honest observation that the old \(hero.name) would have dismissed because it didn't fit the framework.\n")
        if insights.count > 4 { parts.append(weaveInsight(insights[4], character: hero.name, index: 7) + "\n") }
        parts.append("Then another. Then a cascade. \(ally.name) stood at the whiteboard, marker flying, capturing each one before it could escape.\n")
        parts.append("\"This is messy,\" \(hero.name) said, staring at the board.\n")
        parts.append("\"Messy is honest,\" \(ally.name) replied. \"Elegant was lying to you.\"\n")

        parts.append("──────────────────────────────────────────")
        parts.append("  PART SIX: THE ASCENT")
        parts.append("──────────────────────────────────────────\n")
        parts.append("Three months of reconstruction. \(hero.name) called \(mentor.name) back. The old professor answered on the first ring, as if the phone had never stopped ringing.\n")
        parts.append("\"Tell me everything,\" \(mentor.name) said. And listened for two hours without interrupting once.\n")
        parts.append("\"It's better,\" \(mentor.name) said finally. \"It's not elegant. It's not clean. But it has something your old theory never had.\"\n")
        parts.append("\"What?\"\n")
        parts.append("\"Honesty. And honesty, in science, is the only thing that lasts.\"\n")
        parts.append("The new theory was provisional. It admitted its own limitations on page one. It contained three sections titled \"What We Do Not Yet Know\" — which was, \(hero.name) realized, the bravest thing any researcher could write.\n")
        for (i, insight) in insights.dropFirst(5).prefix(3).enumerated() {
            parts.append(weaveInsight(insight, character: hero.name, index: i + 9) + "\n")
        }
        parts.append("\(villain.name) read the preprint. Sent a one-line email: \"This is actually good. I hate that it's good. We should talk.\"\n")
        parts.append("They met at a coffee shop halfway between their offices. The conversation lasted four hours. When it was over, they shook hands — not as friends, but as something more rare: honest adversaries who respected each other's rigor.\n")
        if !evolved.narrative.isEmpty && isCleanStoryInsight(evolved.narrative) {
            let cleanNarrative = String(evolved.narrative.prefix(400))
            parts.append("Beneath the new framework lay something unexpected: \(cleanNarrative)\n")
        }

        parts.append("──────────────────────────────────────────")
        parts.append("  PART SEVEN: THE NEW EQUILIBRIUM")
        parts.append("──────────────────────────────────────────\n")
        parts.append("Life was good again — but differently. The difference was the scar tissue, and the wisdom that scar tissue carries.\n")
        parts.append("\(hero.name) had a new lab. Smaller. Quieter. Two graduate students instead of \(Int.random(in: 5...7)), both chosen not for their brilliance but for their willingness to say \"I don't know.\" A revolutionary hiring criterion.\n")
        parts.append("The new theory about \(topic) had been published. Not in the flashy journal that had hosted the original triumph, but in a smaller one that prioritized rigor over impact factor. It was cited less often. It was understood more deeply.\n")
        parts.append("\(ally.name) stood in the doorway. Same position. Same crossed arms. But the smile was different now — it reached the eyes.\n")
        parts.append("\"Coffee at seven?\"\n")
        parts.append("\"Coffee at seven,\" \(hero.name) confirmed. \"Arguments by nine.\"\n")
        parts.append("\"Breakthroughs by midnight?\"\n")
        parts.append("\(hero.name) laughed — a real laugh, the kind that comes from a place that used to be a wound. \"Breakthroughs by maybe. Humility by always.\"\n")
        parts.append("Outside, the sun was doing that thing it does in late afternoon — turning everything gold, making the ordinary look sacred. \(hero.name) watched it through the window and thought about \(topic), not as a problem to be solved but as a companion on a very long walk.\n")
        parts.append("The comedy is not that \(hero.name) fell. Everyone falls. The comedy is that falling was necessary for flight — and that the flight, when it came, was nothing like what \(hero.name) had imagined. It was better. It was real.\n")
        parts.append("\(mentor.name) sent a card. It contained a single sentence: \"Now you are a scientist.\"\n")
        parts.append("\(hero.name) pinned it to the wall above the desk, next to a Post-it note that read: \"Be wrong well.\"\n")
        parts.append("\n───\n")
        parts.append("And in the end, \(t) kept its secrets — most of them. But it had given \(hero.name) something better than answers: the courage to live in the questions. And that, as it turns out, is the whole comedy.\n")
        parts.append("\n  F I N")

        return parts.joined(separator: "\n")
    }
}
