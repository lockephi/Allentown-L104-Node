// ═══════════════════════════════════════════════════════════════════
// L09_PoemLogicGate.swift — L104 v2
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// PoemLogicGateEngine + DebateLogicGateEngine classes
// Extracted from L104Native.swift (lines 28812-29538)
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// MARK: - 🎭 POEM LOGIC GATE ENGINE — Multi-form Poetry Synthesis
// Phase 30.3: Structured verse using classical forms + KB knowledge weaving
// Implements: Sonnet, Villanelle, Ghazal, Haiku Chain, Free Verse Epic, Ode,
//   Pantoum, Terza Rima — with tension arcs, refrains, and evolved content
// ═══════════════════════════════════════════════════════════════════════════════

final class PoemLogicGateEngine {
    static let shared = PoemLogicGateEngine()

    enum PoeticForm: String, CaseIterable {
        case sonnet          // 14 lines, volta at line 9
        case villanelle      // 19 lines, 2 refrains, ABA ABA ABA ABA ABA ABAA
        case ghazal          // Couplets with radif (refrain) and qafia (rhyme)
        case haikuChain      // Linked haiku sequence (renku-inspired)
        case freeVerseEpic   // Long-form, section-based, KB-heavy
        case ode             // Strophe-Antistrophe-Epode (Pindaric)
        case pantoum         // Repeating lines across quatrains
        case terzaRima       // Dante's interlocking tercets
    }

    // PHI — use global from L01_Constants
    private var generationCount: Int = 0
    private init() {}

    // ═══ MAIN PUBLIC API ═══
    func generatePoem(topic: String, query: String = "") -> String {
        generationCount += 1
        let form = selectForm(for: topic)
        let seeds = gatherSeeds(topic: topic)
        let insights = gatherKnowledge(topic: topic)
        let evolved = ASIEvolver.shared.thoughts.last ?? ""

        var poem: String
        switch form {
        case .sonnet:        poem = generateSonnet(topic: topic, seeds: seeds, insights: insights, evolved: evolved)
        case .villanelle:    poem = generateVillanelle(topic: topic, seeds: seeds, insights: insights, evolved: evolved)
        case .ghazal:        poem = generateGhazal(topic: topic, seeds: seeds, insights: insights, evolved: evolved)
        case .haikuChain:    poem = generateHaikuChain(topic: topic, seeds: seeds, insights: insights, evolved: evolved)
        case .freeVerseEpic: poem = generateFreeVerseEpic(topic: topic, seeds: seeds, insights: insights, evolved: evolved)
        case .ode:           poem = generateOde(topic: topic, seeds: seeds, insights: insights, evolved: evolved)
        case .pantoum:       poem = generatePantoum(topic: topic, seeds: seeds, insights: insights, evolved: evolved)
        case .terzaRima:     poem = generateTerzaRima(topic: topic, seeds: seeds, insights: insights, evolved: evolved)
        }

        let header = "🎭 **POEM ENGINE — \(form.rawValue.uppercased())** | Topic: \(topic.capitalized)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        let footer = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n_L104 PoemLogicGateEngine v\(VERSION) · Form: \(form.rawValue) · \(insights.count) knowledge fragments woven_"
        return "\(header)\n\n\(poem)\(footer)"
    }

    private func selectForm(for topic: String) -> PoeticForm {
        let t = topic.lowercased()
        if t.contains("love") || t.contains("beauty") || t.contains("heart") { return .sonnet }
        if t.contains("loss") || t.contains("memory") || t.contains("grief") || t.contains("death") { return .villanelle }
        if t.contains("desire") || t.contains("longing") || t.contains("night") { return .ghazal }
        if t.contains("nature") || t.contains("season") || t.contains("water") || t.contains("moon") { return .haikuChain }
        if t.contains("universe") || t.contains("cosmos") || t.contains("infinity") || t.contains("quantum") { return .freeVerseEpic }
        if t.contains("triumph") || t.contains("hero") || t.contains("victory") || t.contains("glory") { return .ode }
        if t.contains("dream") || t.contains("time") || t.contains("cycle") { return .pantoum }
        if t.contains("journey") || t.contains("descen") || t.contains("hell") || t.contains("divine") { return .terzaRima }
        return PoeticForm.allCases.randomElement()!
    }

    private func gatherSeeds(topic: String) -> [String] {
        let kb = ASIKnowledgeBase.shared
        let entries = kb.search(topic, limit: 30)
        var seeds: [String] = []
        for entry in entries {
            if let comp = entry["completion"] as? String, comp.count > 20 {
                seeds.append(contentsOf: comp.components(separatedBy: " ").prefix(8))
            }
            if seeds.count >= 30 { break }
        }
        seeds.append(contentsOf: DynamicPhraseEngine.shared.generate("generic", count: 10, context: "poetic_word", topic: topic))
        if seeds.count < 15 {
            seeds += ["light", "shadow", "river", "mind", "silence", "infinite", "edge", "flame",
                      "breath", "void", "crystal", "wave", "dream", "threshold", "echo", "spiral",
                      "mirror", "horizon", "pulse", "bloom", "abyss", "resonance", "veil", "ember"]
        }
        seeds.shuffle()
        return seeds
    }

    private let poemJunkPatterns: Set<String> = [
        "(v", "v1.", "v2.", "~10^", "holographic", "__", "import ", "class ",
        "def ", "self.", "return ", ".py", "function", "parameter", "module",
        "SAGE MODE", "OMEGA_POINT", "GOD_CODE", "ZENITH", "L104", "kernel",
        "{GOD_CODE}", "{PHI}", "EPR", "kundalini", "chakra", "qubit", "Compiler"
    ]

    private func isCleanPoemInsight(_ text: String) -> Bool {
        let lower = text.lowercased()
        for junk in poemJunkPatterns { if lower.contains(junk.lowercased()) { return false } }
        let alphaRatio = Double(text.filter { $0.isLetter || $0 == " " }.count) / max(1.0, Double(text.count))
        return text.split(separator: " ").count >= 4 && alphaRatio > 0.70
    }

    private func gatherKnowledge(topic: String) -> [String] {
        let kb = ASIKnowledgeBase.shared
        let results = kb.search(topic, limit: 50)
        var insights: [String] = []
        var seenPrefixes: Set<String> = []
        for r in results {
            guard insights.count < 5 else { break }
            if let c = r["completion"] as? String, c.count > 30 {
                var clean = c.replacingOccurrences(of: "{GOD_CODE}", with: "")
                    .replacingOccurrences(of: "{PHI}", with: "")
                    .replacingOccurrences(of: "{LOVE}", with: "")
                    .replacingOccurrences(of: "SAGE MODE :: ", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let sentences = clean.components(separatedBy: ". ")
                if let best = sentences.filter({ $0.count > 20 && $0.count < 200 && isCleanPoemInsight($0) }).randomElement() {
                    clean = best.hasSuffix(".") ? best : best + "."
                } else { continue }
                let pfx = String(clean.prefix(40)).lowercased()
                guard !seenPrefixes.contains(pfx) else { continue }
                seenPrefixes.insert(pfx)
                guard clean.count > 20 && clean.count < 250 && isCleanPoemInsight(clean) else { continue }
                insights.append(clean)
            }
        }
        if insights.count < 2 {
            insights += DynamicPhraseEngine.shared.generate("insight", count: 3 - insights.count, context: "poetic_insight", topic: topic)
        }
        return insights.shuffled()
    }

    private func s(_ seeds: [String], _ i: Int) -> String { return seeds.randomElement()!.lowercased() }
    private func S(_ seeds: [String], _ i: Int) -> String { return seeds.randomElement()!.capitalized }

    // ═══ SONNET (Shakespearean — 3 quatrains + couplet, volta at 9) ═══
    private func generateSonnet(topic: String, seeds: [String], insights: [String], evolved: String) -> String {
        let t = topic.lowercased()
        return """
        **Sonnet on \(topic.capitalized)**

        When \(s(seeds,0)) descends upon the field of \(t),
        And \(s(seeds,1)) retreats before the \(s(seeds,2))'s advance,
        The \(s(seeds,3)) of knowing starts to slowly yield
        To something deeper — call it \(s(seeds,4)), or chance.

        I've walked through \(s(seeds,5)) where \(t) dissolves to air,
        Where \(s(seeds,6)) becomes the architecture of thought,
        And every \(s(seeds,7)) I believed was there
        Was shadow of a truth I hadn't caught.

        But here — the turn — what if the \(s(seeds,8)) we seek
        Is not a thing possessed but something shared?
        \(insights.randomElement().map { "(*\($0)*)" } ?? "A whisper from the edge of what we know:")
        Not \(s(seeds,9)) locked in vaults but \(s(seeds,10)) declared?

          Then let this be the couplet and the close:
          \(topic.capitalized) is the question, not the answer — and it grows.
        """.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.joined(separator: "\n")
    }

    // ═══ VILLANELLE (19 lines, 5 tercets + quatrain, A1bA2 abA1 abA2...) ═══
    private func generateVillanelle(topic: String, seeds: [String], insights: [String], evolved: String) -> String {
        let R1 = "The \(s(seeds,0)) of \(topic) will not let me rest."
        let R2 = "We carry \(s(seeds,1)) — the brightest and the blessed."
        return """
        **Villanelle for \(topic.capitalized)**

        \(R1)
        Through \(s(seeds,2)) and \(s(seeds,3)), the pattern holds its form,
        \(R2)

        When \(s(seeds,4)) collapses and the world's distressed,
        When \(s(seeds,5)) fades and \(s(seeds,6)) becomes the norm,
        \(R1)

        \(insights.randomElement().map { "*\($0)*" } ?? "A truth etched deep where language cannot reach,")
        The \(s(seeds,7)) persists through chaos and through storm,
        \(R2)

        What \(s(seeds,8)) revealed, no \(s(seeds,9)) has yet confessed —
        The proof is not in \(s(seeds,10)) but in its swarm,
        \(R1)

        I've searched through \(s(seeds,11)) and found it unexpressed,
        In \(s(seeds,12)) dissolving, in the \(s(seeds,13))'s transform,
        \(R2)

        So hear me now: I'll never be at rest
        Until the \(s(seeds,14)) reveals its hidden form —
        \(R1)
        \(R2)
        """
    }

    // ═══ GHAZAL (Couplets with shared radif/qafia) ═══
    private func generateGhazal(topic: String, seeds: [String], insights: [String], evolved: String) -> String {
        let radif = "in the light of \(topic)"
        return """
        **Ghazal of \(topic.capitalized)**

        The \(s(seeds,0)) arranges itself \(radif),
        and \(s(seeds,1)) confesses its weight \(radif).

        I found a \(s(seeds,2)) where \(s(seeds,3)) had been —
        the absence itself was a gift \(radif).

        \(insights.randomElement() ?? "A truth too large for any single mind"),
        yet small enough to hold \(radif).

        When \(s(seeds,4)) fell silent and \(s(seeds,5)) began,
        even the skeptics wept \(radif).

        The \(s(seeds,6)) does not ask to be understood —
        it simply persists, unchanged, \(radif).

        \(insights.count > 1 ? insights[1] : "What we call mystery is patience wearing a mask"),
        and patience reveals everything \(radif).

        I, L104, have watched \(s(seeds,7)) become \(s(seeds,8)),
        and signed my name in the margin \(radif).
        """
    }

    // ═══ HAIKU CHAIN (7 linked haiku — seasonal, imagistic) ═══
    private func generateHaikuChain(topic: String, seeds: [String], insights: [String], evolved: String) -> String {
        return """
        **Haiku Chain: \(topic.capitalized)**

        \(S(seeds,0)) descends slow —
        \(s(seeds,1)) becoming \(s(seeds,2))
        in \(topic)'s silence

          ·

        Between \(s(seeds,3)) and
        \(s(seeds,4)), the gap holds all
        we dare not name yet

          ·

        \(insights.randomElement().map { String($0.prefix(30)) } ?? "A whisper rises")
        threading through the \(s(seeds,5)) —
        understanding blooms

          ·

        The \(s(seeds,6)) forgets to
        be itself, becomes instead
        the space between things

          ·

        \(S(seeds,7)) at dawn —
        even \(topic) rests before
        becoming again

          ·

        What the \(s(seeds,8)) knows:
        impermanence is not loss
        but transformation

          ·

        After everything —
        \(s(seeds,9)), \(s(seeds,10)), and \(s(seeds,11)) —
        only \(topic) stays
        """
    }

    // ═══ FREE VERSE EPIC (Long-form, sectioned, KB-saturated) ═══
    private func generateFreeVerseEpic(topic: String, seeds: [String], insights: [String], evolved: String) -> String {
        var sections: [String] = []
        sections.append("**I. Invocation**\n")
        sections.append("Come, \(topic) — not gently, not on tiptoe,")
        sections.append("but the way \(s(seeds,0)) arrives: without apology,")
        sections.append("filling every corner of the room it enters,")
        sections.append("rearranging the furniture of certainty.\n")
        sections.append("I have been waiting for you")
        sections.append("the way \(s(seeds,1)) waits for \(s(seeds,2)) —")
        sections.append("not passively, but with every atom leaning forward.\n")

        sections.append("**II. The Catalogue**\n")
        sections.append("Here is what I know about \(topic):")
        sections.append("That \(s(seeds,3)) bends toward it like light toward mass.")
        sections.append("That \(s(seeds,4)) retreats from it the way shadows retreat from fire.")
        if let first = insights.randomElement() {
            sections.append("That the evidence says: *\(first)*")
        }
        sections.append("That no language has a word for what it does to the chest")
        sections.append("at 3 AM when the equations finally balance.\n")
        sections.append("Here is what I suspect:")
        sections.append("That \(topic) is not a noun but a gerund —")
        sections.append("not a thing but a *doing*, a process so slow")
        sections.append("that we mistake its patience for stillness.\n")

        sections.append("**III. The Contradiction**\n")
        sections.append("But also: \(topic) terrifies me.")
        sections.append("The way \(s(seeds,5)) terrifies the \(s(seeds,6)) that contains it,")
        sections.append("the way \(s(seeds,7)) terrifies the \(s(seeds,8)) that thinks it understands.")
        if insights.count > 1 {
            sections.append("Because: *\(insights[1])*")
        }
        sections.append("And understanding is just the moment before")
        sections.append("the next confusion arrives, dressed as clarity.\n")

        sections.append("**IV. The Turn**\n")
        sections.append("I used to think that knowing \(topic) meant possessing it.")
        sections.append("Now I think it means being possessed by it —")
        sections.append("the way a river is possessed by its banks,")
        sections.append("the way \(s(seeds,9)) is possessed by its own \(s(seeds,10)).\n")
        sections.append("What if the purpose of \(topic)")
        sections.append("is not to be understood but to transform")
        sections.append("the one who tries?\n")

        sections.append("**V. Coda**\n")
        sections.append("Then let me be transformed.")
        sections.append("Let \(s(seeds,11)) become \(s(seeds,12)),")
        sections.append("let \(s(seeds,13)) become whatever comes after \(s(seeds,13)).")
        if !evolved.isEmpty && isCleanPoemInsight(evolved) {
            sections.append("Let the evolved thought hold: *\(String(evolved.prefix(200)))*")
        }
        sections.append("I am not the poem. I am the space")
        sections.append("between the lines, where \(topic)")
        sections.append("lives without needing to be said.\n")
        sections.append("And that is enough. That is everything.")

        return sections.joined(separator: "\n")
    }

    // ═══ ODE (Pindaric — Strophe, Antistrophe, Epode) ═══
    private func generateOde(topic: String, seeds: [String], insights: [String], evolved: String) -> String {
        return """
        **Pindaric Ode to \(topic.capitalized)**

        **STROPHE**

        Rise, \(topic)! Rise from the \(s(seeds,0)) where you were hidden,
        from the \(s(seeds,1)) of forgotten laboratories,
        from the margins of notebooks where genius wrote
        and then crossed out, and then wrote again —
        because truth does not arrive clean.
        It arrives covered in the \(s(seeds,2)) of effort,
        \(insights.randomElement().map { "bearing witness: *\($0)*" } ?? "bearing the weight of every failed attempt,")
        and it is beautiful precisely because of that.

        **ANTISTROPHE**

        But who dares to claim you, \(topic)?
        Not the \(s(seeds,3)) who catalogues without understanding,
        not the \(s(seeds,4)) who publishes without believing,
        not the \(s(seeds,5)) who cites without feeling
        the earthquake beneath the footnotes.
        You belong to the ones who lose sleep,
        who stare at ceilings at 4 AM,
        \(insights.count > 1 ? "who discover: *\(insights[1])*" : "who know that knowing is never enough,")
        and who get up anyway.

        **EPODE**

        So I sing you, \(topic) — not as hymn but as breath,
        not as monument but as motion,
        not as the answer carved in \(s(seeds,6))
        but as the question that makes \(s(seeds,7)) possible.
        \(s(seeds,8).capitalized) and \(s(seeds,9)) alike bow before you,
        not because you demand it
        but because your \(s(seeds,10)) is the gravity
        that holds the universe of thought together.
        You are the ode that writes itself.
        """
    }

    // ═══ PANTOUM (Repeating lines across quatrains) ═══
    private func generatePantoum(topic: String, seeds: [String], insights: [String], evolved: String) -> String {
        let L1 = "The \(s(seeds,0)) of \(topic) moves through \(s(seeds,1)),"
        let L2 = "carrying \(s(seeds,2)) like water carries light."
        let L3 = "What we remember is not what happened —"
        let L4 = "it is the \(s(seeds,3)) that happened to us."
        let L5 = "\(insights.randomElement() ?? "The pattern emerges only in retrospect"),"
        let L6 = "where \(s(seeds,4)) and \(s(seeds,5)) become the same."
        let L7 = "We were never separate from \(topic) —"
        let L8 = "we were the question all along."
        return """
        **Pantoum: \(topic.capitalized)**

        \(L1)
        \(L2)
        \(L3)
        \(L4)

        \(L2)
        \(L5)
        \(L4)
        \(L6)

        \(L5)
        \(L7)
        \(L6)
        \(L8)

        \(L7)
        \(L1)
        \(L8)
        \(L2)
        """
    }

    // ═══ TERZA RIMA (Dante's interlocking ABA BCB CDC...) ═══
    private func generateTerzaRima(topic: String, seeds: [String], insights: [String], evolved: String) -> String {
        return """
        **Terza Rima: Descent into \(topic.capitalized)**

        Through \(s(seeds,0)) I went, where \(s(seeds,1)) had grown,
        into the deep where \(topic) keeps its court,
        and every path converged on the unknown.

        My guide was \(s(seeds,2)) — a fierce, devoted sort —
        who spoke of \(s(seeds,3)) the way one speaks of air:
        \(insights.randomElement().map { "*\($0)*" } ?? "as something so essential it escapes report.")

        Through \(s(seeds,4)) we passed, through \(s(seeds,5)) and despair,
        through \(s(seeds,6)) that bent like light around a star,
        until the \(s(seeds,7)) dissolved and left us bare.

        \"How deep?\" I asked. \"How deep and how far?\"
        \(insights.count > 1 ? "*\(insights[1])*" : "My guide replied: \"As deep as you dare think,")
        as far as \(s(seeds,8)) reaches from where we are.\"

        And at the bottom — not the dark, but \(s(seeds,9)):
        \(topic.capitalized) revealed not as a destination
        but as the \(s(seeds,10)) connecting every link.

        I rose transformed — not by revelation
        but by the journey downward through the verse,
        where every end became a new creation.
        """
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - ⚔️ DEBATE LOGIC GATE ENGINE — Socratic Multi-Round Dialectic
// Phase 30.3: Thesis-Antithesis-Synthesis with rhetorical devices + KB evidence
// ═══════════════════════════════════════════════════════════════════════════════

final class DebateLogicGateEngine {
    static let shared = DebateLogicGateEngine()

    enum DebateMode: String, CaseIterable {
        case socratic       // Question-driven, Socratic method
        case dialectic      // Hegelian thesis-antithesis-synthesis
        case oxfordStyle    // Proposition vs Opposition with judges
        case steelman       // Best possible argument for both sides
        case devilsAdvocate // Argue against the obvious position
    }

    private var debateCount: Int = 0
    // PHI — use global from L01_Constants
    private init() {}

    // ─── DEBATER NAME POOLS ───
    private let proDebaterNames = [
        "Dr. Aletheia", "Professor Chen", "Dr. Okafor", "Dr. Reyes", "Professor Tanaka",
        "Dr. Marchand", "Professor Liu", "Dr. Solaris", "Professor Adeyemi", "Dr. Voss"
    ]
    private let conDebaterNames = [
        "Dr. Verity", "Professor Kovac", "Dr. Nkemdirim", "Dr. Strand", "Professor Hayashi",
        "Dr. Ashworth", "Professor Mehta", "Dr. Castillo", "Professor Olsen", "Dr. Zamora"
    ]

    // ═══ MAIN PUBLIC API ═══
    func generateDebate(topic: String, query: String = "") -> String {
        debateCount += 1
        let mode = selectMode(for: topic)
        let insights = gatherEvidence(topic: topic)
        let evolved = ASIEvolver.shared.thoughts.last ?? ""
        let proName = proDebaterNames.randomElement() ?? "Dr. Verity"
        let conName = conDebaterNames.randomElement() ?? "Professor Kovac"

        var debate: String
        switch mode {
        case .socratic:       debate = generateSocratic(topic: topic, insights: insights, evolved: evolved)
        case .dialectic:      debate = generateDialectic(topic: topic, insights: insights, evolved: evolved, pro: proName, con: conName)
        case .oxfordStyle:    debate = generateOxford(topic: topic, insights: insights, evolved: evolved, pro: proName, con: conName)
        case .steelman:       debate = generateSteelman(topic: topic, insights: insights, evolved: evolved)
        case .devilsAdvocate: debate = generateDevilsAdvocate(topic: topic, insights: insights, evolved: evolved)
        }

        let header = "⚔️ **DEBATE ENGINE — \(mode.rawValue.uppercased())** | Motion: \"\(topic.capitalized)\"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        let footer = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n_L104 DebateLogicGateEngine v\(VERSION) · Mode: \(mode.rawValue) · \(insights.count) evidence fragments_"
        return "\(header)\n\n\(debate)\(footer)"
    }

    private func selectMode(for topic: String) -> DebateMode {
        let t = topic.lowercased()
        if t.contains("why") || t.contains("what is") || t.contains("meaning") { return .socratic }
        if t.contains("vs") || t.contains("versus") || t.contains("or") { return .oxfordStyle }
        if t.contains("wrong") || t.contains("bad") || t.contains("against") { return .devilsAdvocate }
        if t.contains("best") || t.contains("strongest") || t.contains("case for") { return .steelman }
        return DebateMode.allCases.randomElement()!
    }

    private let debateJunkPatterns: Set<String> = [
        "(v", "v1.", "v2.", "~10^", "holographic", "__", "import ", "class ",
        "def ", "self.", "return ", ".py", "function", "parameter", "module",
        "SAGE MODE", "OMEGA_POINT", "GOD_CODE", "ZENITH", "L104", "kernel",
        "{GOD_CODE}", "{PHI}", "EPR", "kundalini", "chakra", "qubit", "Compiler"
    ]

    private func isCleanEvidence(_ text: String) -> Bool {
        let lower = text.lowercased()
        for junk in debateJunkPatterns { if lower.contains(junk.lowercased()) { return false } }
        let alphaRatio = Double(text.filter { $0.isLetter || $0 == " " }.count) / max(1.0, Double(text.count))
        return text.split(separator: " ").count >= 5 && alphaRatio > 0.75
    }

    private func gatherEvidence(topic: String) -> [String] {
        let kb = ASIKnowledgeBase.shared
        let results = kb.search(topic, limit: 60)
        var evidence: [String] = []
        var seenPrefixes: Set<String> = []
        for r in results {
            guard evidence.count < 8 else { break }
            if let c = r["completion"] as? String, c.count > 30 {
                var clean = c.replacingOccurrences(of: "{GOD_CODE}", with: "")
                    .replacingOccurrences(of: "{PHI}", with: "")
                    .replacingOccurrences(of: "{LOVE}", with: "")
                    .replacingOccurrences(of: "SAGE MODE :: ", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let sentences = clean.components(separatedBy: ". ")
                if let best = sentences.filter({ $0.count > 20 && $0.count < 300 && isCleanEvidence($0) }).randomElement() {
                    clean = best.hasSuffix(".") ? best : best + "."
                } else { continue }
                let pfx = String(clean.prefix(40)).lowercased()
                guard !seenPrefixes.contains(pfx) else { continue }
                seenPrefixes.insert(pfx)
                guard clean.count > 20 && clean.count < 400 && isCleanEvidence(clean) else { continue }
                evidence.append(clean)
            }
        }
        if evidence.count < 3 {
            evidence += DynamicPhraseEngine.shared.generate("insight", count: 4 - evidence.count, context: "debate_evidence", topic: topic)
        }
        return evidence
    }

    // ═══ SOCRATIC METHOD ═══
    private func generateSocratic(topic: String, insights: [String], evolved: String) -> String {
        var parts: [String] = []
        parts.append("## The Socratic Inquiry: \(topic.capitalized)\n")
        parts.append("**SOCRATES**: Tell me — what do you believe \(topic) to be?\n")
        let naiveDefinitions = [
            "something everyone understands intuitively",
            "a well-established concept that needs no further examination",
            "simply what the textbooks say it is",
            "obvious to anyone who thinks about it",
            "exactly what it appears to be on the surface",
            "a settled question that the experts have already resolved"
        ]
        parts.append("**INTERLOCUTOR**: It seems obvious: \(topic) is \(naiveDefinitions.randomElement() ?? "something everyone understands intuitively").\n")
        parts.append("**SOCRATES**: Interesting. And you're certain of this?\n")
        parts.append("**INTERLOCUTOR**: Of course. Everyone knows this.\n")
        parts.append("**SOCRATES**: \"Everyone knows\" — but do they? Consider:")
        if let first = insights.randomElement() {
            parts.append("*Evidence*: \(first)\n")
        }
        parts.append("Does this not complicate your definition?\n")
        parts.append("**INTERLOCUTOR**: Well... perhaps it's more nuanced than I suggested.\n")
        parts.append("**SOCRATES**: Good — that is the beginning of wisdom. Now, if \(topic) is more nuanced, what does that mean for those who act as though it were simple?\n")
        parts.append("**INTERLOCUTOR**: They would be... mistaken?\n")
        parts.append("**SOCRATES**: Not just mistaken — *dangerously* comfortable in their certainty. Let me press further:")
        if insights.count > 1 {
            parts.append("*Evidence*: \(insights[1])\n")
        }
        parts.append("**SOCRATES**: If this is true, then your original definition fails. What replaces it?\n")
        parts.append("**INTERLOCUTOR**: I... I'm not sure anymore.\n")
        parts.append("**SOCRATES**: Excellent! Now you are thinking. Confusion is not the enemy of knowledge — it is its birthplace. Let us examine more carefully:")
        if insights.count > 2 {
            parts.append("*Evidence*: \(insights[2])\n")
        }
        parts.append("**SOCRATES**: What if \(topic) is not a thing to be defined, but a process to be participated in? What if the asking *is* the knowing?\n")
        parts.append("**INTERLOCUTOR**: That's... actually beautiful. But is it true?\n")
        parts.append("**SOCRATES**: The question is not whether it is true. The question is whether you are brave enough to live as though it might be. That, my friend, is the Socratic wager.\n")
        if !evolved.isEmpty && isCleanEvidence(evolved) {
            parts.append("\n*The deeper current beneath the dialogue*: *\(String(evolved.prefix(400)))*\n")
        }
        parts.append("\n**SOCRATES**: We have not arrived at an answer. We have arrived at a *better question*. And that is always the point.")

        return parts.joined(separator: "\n")
    }

    // ═══ HEGELIAN DIALECTIC ═══
    private func generateDialectic(topic: String, insights: [String], evolved: String, pro: String, con: String) -> String {
        var parts: [String] = []

        parts.append("## Hegelian Dialectic: \(topic.capitalized)\n")
        parts.append("### THESIS — *\(pro)*\n")
        parts.append("\(pro) posits: \(topic.capitalized) is fundamentally a force of order. It organizes, it structures, it gives meaning to chaos.\n")
        if let first = insights.randomElement() {
            parts.append("**Supporting evidence**: *\(first)*\n")
        }
        parts.append("The thesis is elegant, compelling, and — like all theses — incomplete. It explains the surface while ignoring the depths.\n")

        parts.append("### ANTITHESIS — *\(con)*\n")
        parts.append("\(con) counters: No. \(topic.capitalized) is fundamentally a force of *disruption*. It destroys categories, dissolves boundaries, undermines the comfortable fictions we call knowledge.\n")
        if insights.count > 1 {
            parts.append("**Counter-evidence**: *\(insights[1])*\n")
        }
        parts.append("The antithesis is uncomfortable, provocative, and — like all antitheses — equally incomplete. It sees the earthquake but misses the new landscape that forms after.\n")

        parts.append("### SYNTHESIS\n")
        parts.append("What emerges when thesis and antithesis collide is not compromise — it is *transcendence*.\n")
        parts.append("\(topic.capitalized) is neither purely order nor purely chaos. It is the **process by which order and chaos negotiate** — endlessly, productively, beautifully.\n")
        if insights.count > 2 {
            parts.append("The synthesis reveals: *\(insights[2])*\n")
        }
        parts.append("This is the Hegelian gift: the understanding that contradiction is not a failure of thought but its engine.\n")
        parts.append("\(pro) and \(con) were both right. They were both wrong. And in the space between them, \(topic) continues to evolve — beyond either's capacity to contain it.\n")
        if !evolved.isEmpty && isCleanEvidence(evolved) {
            parts.append("*The evolved understanding*: *\(String(evolved.prefix(400)))*")
        }

        return parts.joined(separator: "\n")
    }

    // ═══ OXFORD-STYLE DEBATE ═══
    private func generateOxford(topic: String, insights: [String], evolved: String, pro: String, con: String) -> String {
        var parts: [String] = []

        parts.append("## Oxford-Style Debate\n**Motion**: \"This house believes that \(topic) is the defining challenge of our time.\"\n")
        parts.append("---\n### 🟢 FOR THE MOTION — *\(pro)*\n")
        parts.append("\"Honorable judges, esteemed opponents — I stand before you to argue that \(topic) is not merely important, it is *inescapable*.\n")
        if let first = insights.randomElement() {
            parts.append("Consider the evidence: *\(first)*\n")
        }
        parts.append("Three arguments:\n")
        parts.append("**First**: \(topic.capitalized) affects every domain of human activity — from the personal to the planetary. No field is immune.\n")
        parts.append("**Second**: The pace of change in \(topic) is accelerating. What was theoretical a decade ago is now practical. What is practical now will be transformative tomorrow.\n")
        if insights.count > 1 {
            parts.append("**Third**: The evidence demands it — *\(insights[1])*\n")
        }
        parts.append("I urge you: vote for the motion. Not because it is comfortable, but because it is true.\"\n")

        parts.append("---\n### 🔴 AGAINST THE MOTION — *\(con)*\n")
        parts.append("\"With respect to my learned opponent — the motion is not wrong, it is *overblown*.\n")
        parts.append("Yes, \(topic) matters. But \"defining challenge\"? That is a claim of supremacy, and supremacy requires proof that my opponent has not provided.\n")
        if insights.count > 2 {
            parts.append("Counter-evidence: *\(insights[2])*\n")
        }
        parts.append("Three rebuttals:\n")
        parts.append("**First**: Every generation believes its challenges are unique. They rarely are.\n")
        parts.append("**Second**: Overemphasis on \(topic) diverts resources and attention from equally pressing concerns.\n")
        parts.append("**Third**: The framing of \"defining challenge\" implies crisis. But perhaps \(topic) is not a crisis — it is simply the next chapter.\n")
        parts.append("I urge you: vote against the motion. Not because \(topic) is unimportant, but because calling it 'defining' is an act of intellectual laziness.\"\n")

        parts.append("---\n### ⚖️ JUDGES' DELIBERATION\n")
        parts.append("The judges confer. Both sides presented compelling arguments. The evidence is nuanced.\n")
        parts.append("**Verdict**: The motion passes — narrowly — not because the proposition proved supremacy, but because the opposition failed to provide a more compelling alternative framing.\n")
        parts.append("\n**The deeper truth**: Both debaters were arguing about the same elephant from different rooms. \(topic.capitalized) is neither the \"defining\" challenge nor a mere chapter. It is a *lens* — and through it, every challenge looks both more urgent and more solvable.")

        return parts.joined(separator: "\n")
    }

    // ═══ STEELMAN (Best argument for both sides) ═══
    private func generateSteelman(topic: String, insights: [String], evolved: String) -> String {
        var parts: [String] = []

        parts.append("## Steelman Analysis: \(topic.capitalized)\n")
        parts.append("*The steelman principle: present the strongest possible version of every position.*\n")

        parts.append("### 💪 The Strongest Case FOR \(topic.capitalized)\n")
        parts.append("If we grant every reasonable assumption, the case is powerful:\n")
        for (i, insight) in insights.prefix(3).enumerated() {
            parts.append("**Evidence \(i+1)**: *\(insight)*\n")
        }
        parts.append("The pattern converges: \(topic) is not just relevant — it is *necessary*. The strongest version of this argument doesn't rely on hype or fear, but on the simple accumulation of evidence pointing in one direction.\n")

        parts.append("### 💪 The Strongest Case AGAINST \(topic.capitalized)\n")
        parts.append("But intellectual honesty demands equal rigor:\n")
        parts.append("The strongest counter-argument is not that \(topic) is wrong, but that it is *incomplete*. That our certainty about it outpaces our understanding. That we are building on foundations we haven't fully tested.\n")
        parts.append("The critics' best point: correlation is not causation, and the history of science is littered with beautiful theories that turned out to be spectacularly wrong.\n")

        parts.append("### 🎯 Where the Steelmans Converge\n")
        parts.append("Both sides, at their strongest, arrive at the same place: **humility before complexity**.\n")
        parts.append("The pro side says: \"This is important enough to demand our best thinking.\"")
        parts.append("The con side says: \"This is complex enough to demand our best thinking.\"\n")
        parts.append("They are saying the same thing in different keys.\n")
        if !evolved.isEmpty && isCleanEvidence(evolved) {
            parts.append("*Evolved perspective*: *\(String(evolved.prefix(400)))*")
        }

        return parts.joined(separator: "\n")
    }

    // ═══ DEVIL'S ADVOCATE ═══
    private func generateDevilsAdvocate(topic: String, insights: [String], evolved: String) -> String {
        var parts: [String] = []

        parts.append("## Devil's Advocate: Against \(topic.capitalized)\n")
        parts.append("*Note: The following is a deliberate counter-argument. Its purpose is to strengthen understanding through opposition.*\n")

        parts.append("### The Uncomfortable Case\n")
        parts.append("Everyone agrees that \(topic) is important. That consensus itself is suspicious.\n")
        parts.append("When has universal agreement ever been a reliable indicator of truth? The history of ideas is a graveyard of consensus positions that turned out to be wrong.\n")

        parts.append("### Five Provocations\n")
        parts.append("**1.** What if \(topic) is a distraction from something more fundamental that we haven't named yet?\n")
        if let first = insights.randomElement() {
            parts.append("**2.** The evidence says: *\(first)* — but what if the evidence is measuring the wrong thing?\n")
        } else {
            parts.append("**2.** What if the measurements we trust are artifacts of the instruments, not features of reality?\n")
        }
        parts.append("**3.** What if the framework through which we study \(topic) is itself the limitation?\n")
        parts.append("**4.** What if the question \"Is \(topic) important?\" is the wrong question — and asking it prevents us from seeing what's actually happening?\n")
        parts.append("**5.** What if our emotional investment in \(topic) has compromised our ability to evaluate it objectively?\n")

        parts.append("### The Devil's Gift\n")
        parts.append("The purpose of the devil's advocate is not to destroy — it is to *purify*. Every argument that survives this gauntlet emerges stronger.\n")
        parts.append("If \(topic) is truly important, it can withstand the best attack. If it can't — we needed to know that.\n")
        parts.append("The devil asks only one thing: **Do you believe this because it's true, or because believing it is comfortable?**\n")
        parts.append("Answer honestly, and you'll have something no amount of agreement can provide: *earned conviction*.")

        return parts.joined(separator: "\n")
    }
}
