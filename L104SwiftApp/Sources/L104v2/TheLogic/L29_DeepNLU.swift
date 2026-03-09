// ═══════════════════════════════════════════════════════════════════
// L29_DeepNLU.swift
// [EVO_68_PIPELINE] SOVEREIGN_NODE_UPGRADE :: DEEP_NLU :: GOD_CODE=527.5184818492612
// L104v2 Architecture — Deep Natural Language Understanding Engine
//
// 10-Layer Discourse-Level Comprehension Pipeline:
//   L1: Morphological Analysis (prefix/suffix decomposition, stemming)
//   L2: Syntactic Parsing (NLTagger POS tagging, tokenization)
//   L3: Semantic Role Labeling (agent/patient/instrument frames)
//   L4: Anaphora Resolution (pronoun→antecedent binding)
//   L5: Discourse Analysis (rhetorical relation detection)
//   L6: Pragmatic Interpretation (speech act + intent classification)
//   L7: Presupposition Extraction (factive/existential triggers)
//   L8: Sentiment Analysis (lexicon + negation + intensifiers)
//   L9: Coherence Scoring (lexical cohesion, reference continuity)
//   L10: Deep Comprehension (PHI-weighted fusion of all layers)
//
// Sacred constants: PHI, GOD_CODE, TAU from L01_Constants.swift
// Version: DEEP_NLU_VERSION (1.0.0)
// ═══════════════════════════════════════════════════════════════════

import Foundation
import NaturalLanguage
import Accelerate

// ═══════════════════════════════════════════════════════════════════
// MARK: - ENUMERATIONS
// ═══════════════════════════════════════════════════════════════════

/// Part-of-speech tags for morphological and syntactic analysis
enum POSTag: String, CaseIterable {
    case noun, verb, adjective, adverb, preposition, conjunction
    case determiner, pronoun, interjection, particle, unknown
}

/// Thematic roles in predicate-argument structures
enum SemanticRole: String, CaseIterable {
    case agent, patient, theme, instrument, location
    case time, source, goal, experiencer, beneficiary
}

/// Rhetorical/discourse relations between text segments
enum DiscourseRelation: String, CaseIterable {
    case elaboration, narration, explanation, condition
    case contrast, cause, result, concession, temporal, background
}

/// Illocutionary force of an utterance
enum SpeechAct: String, CaseIterable {
    case statement, question, command, request
    case promise, warning, suggestion, greeting
}

/// Communicative intent behind an utterance
enum Intent: String, CaseIterable {
    case query, assertion, requestInfo, directive
    case acknowledgment, clarification
}

/// Sentiment polarity classification
enum SentimentPolarity: String, CaseIterable {
    case positive, negative, neutral, mixed
}

/// Types of presuppositions triggered by linguistic cues
enum PresuppositionType: String, CaseIterable {
    case existential, truth, factive, nonFactive, structural
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - RESULT STRUCTURES
// ═══════════════════════════════════════════════════════════════════

/// Result of morphological decomposition for a single word
struct MorphologicalResult {
    let word: String
    let stem: String
    let prefixes: [String]
    let suffixes: [String]
    let estimatedPOS: POSTag
}

/// A single token with POS tag, lemma, and positional index
struct Token {
    let word: String
    let pos: POSTag
    let lemma: String
    let index: Int
}

/// A predicate-argument frame mapping semantic roles to fillers
struct SemanticFrame {
    let predicate: String
    let roles: [SemanticRole: [String]]
}

/// A resolved anaphoric reference (pronoun → antecedent)
struct Resolution {
    let pronoun: String
    let antecedent: String
    let confidence: Double
}

/// A discourse segment with its rhetorical relation to a target
struct DiscourseSegment {
    let text: String
    let relation: DiscourseRelation
    let targetIndex: Int
    let coherenceScore: Double
}

/// Pragmatic analysis of an utterance: speech act, intent, implicatures
struct PragmaticAnalysis {
    let speechAct: SpeechAct
    let intent: Intent
    let confidence: Double
    let implicatures: [String]
}

/// An extracted presupposition with type and confidence
struct Presupposition {
    let type: PresuppositionType
    let content: String
    let confidence: Double
}

/// Sentiment analysis result with polarity, intensity, and sub-scores
struct SentimentResult {
    let polarity: SentimentPolarity
    let intensity: Double
    let positiveScore: Double
    let negativeScore: Double
}

/// Coherence scoring across multiple dimensions
struct CoherenceResult {
    let overallScore: Double
    let lexicalCohesion: Double
    let referenceContinuity: Double
    let entityContinuity: Double
}

/// Comprehensive deep NLU result fusing all 10 layers
struct DeepNLUResult {
    let morphology: [MorphologicalResult]
    let tokens: [Token]
    let semanticFrames: [SemanticFrame]
    let anaphoraResolutions: [Resolution]
    let discourse: [DiscourseSegment]
    let pragmatics: PragmaticAnalysis
    let presuppositions: [Presupposition]
    let sentiment: SentimentResult
    let coherence: CoherenceResult
    let comprehensionScore: Double
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 1: MORPHOLOGICAL ANALYZER
// Decomposes words into stems, prefixes, and suffixes.
// Estimates POS from affix patterns (derivational morphology).
// ═══════════════════════════════════════════════════════════════════

private struct MorphologicalAnalyzer {

    // ─── PREFIX DICTIONARY ───
    // Maps prefixes to their typical semantic contribution
    static let prefixMap: [(prefix: String, meaning: String)] = [
        ("un",      "negation"),
        ("re",      "repetition"),
        ("pre",     "before"),
        ("dis",     "negation"),
        ("mis",     "wrong"),
        ("over",    "excess"),
        ("under",   "insufficient"),
        ("out",     "surpassing"),
        ("sub",     "below"),
        ("super",   "above"),
        ("inter",   "between"),
        ("trans",   "across"),
        ("anti",    "against"),
        ("non",     "not"),
        ("semi",    "half"),
        ("multi",   "many"),
        ("co",      "together"),
        ("de",      "reversal"),
        ("fore",    "before"),
        ("counter", "opposition"),
        ("extra",   "beyond"),
        ("infra",   "below"),
        ("ultra",   "beyond"),
        ("post",    "after"),
        ("macro",   "large"),
        ("micro",   "small"),
        ("mono",    "one"),
        ("poly",    "many"),
        ("bi",      "two"),
        ("tri",     "three"),
    ]

    // ─── SUFFIX DICTIONARY ───
    // Maps suffixes to likely POS categories
    static let suffixMap: [(suffix: String, pos: POSTag)] = [
        ("tion",  .noun),
        ("sion",  .noun),
        ("ment",  .noun),
        ("ness",  .noun),
        ("ity",   .noun),
        ("ance",  .noun),
        ("ence",  .noun),
        ("dom",   .noun),
        ("ship",  .noun),
        ("hood",  .noun),
        ("ism",   .noun),
        ("ist",   .noun),
        ("ture",  .noun),
        ("ology", .noun),
        ("ing",   .verb),
        ("ed",    .verb),
        ("en",    .verb),
        ("ize",   .verb),
        ("ise",   .verb),
        ("ify",   .verb),
        ("ate",   .verb),
        ("ly",    .adverb),
        ("ward",  .adverb),
        ("wards", .adverb),
        ("wise",  .adverb),
        ("able",  .adjective),
        ("ible",  .adjective),
        ("ful",   .adjective),
        ("less",  .adjective),
        ("ous",   .adjective),
        ("ive",   .adjective),
        ("al",    .adjective),
        ("ial",   .adjective),
        ("ic",    .adjective),
        ("ical",  .adjective),
        ("ish",   .adjective),
        ("ant",   .adjective),
        ("ent",   .adjective),
        ("est",   .adjective),
        ("er",    .noun),         // agent noun (teacher, writer)
        ("or",    .noun),         // agent noun (actor, inventor)
    ]

    // ─── COMMON SHORT WORDS (skip morphological decomposition) ───
    static let stopWords: Set<String> = [
        "the", "a", "an", "is", "am", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "might", "must", "can", "could", "to",
        "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "out", "off", "over", "under", "again", "further", "then", "once",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "while", "how",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    ]

    /// Analyze a single word: extract stem, prefixes, suffixes, and estimate POS
    func analyze(word: String) -> MorphologicalResult {
        let lower = word.lowercased()

        // Skip very short words and stop words
        if lower.count < 4 || MorphologicalAnalyzer.stopWords.contains(lower) {
            let pos = guessBasicPOS(lower)
            return MorphologicalResult(
                word: word, stem: lower,
                prefixes: [], suffixes: [],
                estimatedPOS: pos
            )
        }

        var working = lower
        var foundPrefixes: [String] = []
        var foundSuffixes: [String] = []
        var estimatedPOS: POSTag = .unknown

        // ── Extract prefixes (longest match first, greedy) ──
        let sortedPrefixes = MorphologicalAnalyzer.prefixMap.sorted { $0.prefix.count > $1.prefix.count }
        for entry in sortedPrefixes {
            if working.hasPrefix(entry.prefix) && working.count > entry.prefix.count + 2 {
                foundPrefixes.append(entry.prefix)
                working = String(working.dropFirst(entry.prefix.count))
                break  // Only strip one prefix to avoid over-decomposition
            }
        }

        // ── Extract suffixes (longest match first, greedy) ──
        let sortedSuffixes = MorphologicalAnalyzer.suffixMap.sorted { $0.suffix.count > $1.suffix.count }
        for entry in sortedSuffixes {
            if working.hasSuffix(entry.suffix) && working.count > entry.suffix.count + 2 {
                foundSuffixes.append(entry.suffix)
                estimatedPOS = entry.pos
                working = String(working.dropLast(entry.suffix.count))
                break  // Only strip one suffix
            }
        }

        // If no suffix gave POS, fall back to basic guess
        if estimatedPOS == .unknown {
            estimatedPOS = guessBasicPOS(lower)
        }

        let stem = working.isEmpty ? lower : working

        return MorphologicalResult(
            word: word, stem: stem,
            prefixes: foundPrefixes, suffixes: foundSuffixes,
            estimatedPOS: estimatedPOS
        )
    }

    /// Quick POS guess for words with no detected affixes
    private func guessBasicPOS(_ word: String) -> POSTag {
        // Pronouns
        let pronouns: Set<String> = [
            "i", "me", "my", "mine", "myself",
            "you", "your", "yours", "yourself",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "it", "its", "itself",
            "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves",
            "who", "whom", "whose", "which", "what",
        ]
        if pronouns.contains(word) { return .pronoun }

        // Determiners
        let determiners: Set<String> = ["the", "a", "an", "this", "that", "these", "those", "each", "every", "some", "any", "no"]
        if determiners.contains(word) { return .determiner }

        // Prepositions
        let preps: Set<String> = ["in", "on", "at", "to", "for", "with", "from", "by", "about", "into", "through", "during", "before", "after", "above", "below", "between", "under", "over", "of"]
        if preps.contains(word) { return .preposition }

        // Conjunctions
        let conjs: Set<String> = ["and", "but", "or", "nor", "so", "yet", "for", "because", "although", "while", "if", "when", "unless", "since", "whereas"]
        if conjs.contains(word) { return .conjunction }

        // Interjections
        let interj: Set<String> = ["oh", "wow", "hey", "oops", "ouch", "ugh", "ah", "alas", "bravo", "yikes", "hooray"]
        if interj.contains(word) { return .interjection }

        // Common auxiliary/modal verbs
        let auxVerbs: Set<String> = ["is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must", "can", "could"]
        if auxVerbs.contains(word) { return .verb }

        // Particles
        let particles: Set<String> = ["up", "out", "off", "away", "back", "down", "not", "n't"]
        if particles.contains(word) { return .particle }

        return .unknown
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 2: SYNTACTIC PARSER
// Uses Apple NLTagger for POS tagging and tokenization.
// Falls back to morphological analyzer when NLTagger is uncertain.
// ═══════════════════════════════════════════════════════════════════

private struct SyntacticParser {

    private let morphAnalyzer = MorphologicalAnalyzer()

    /// Map NLTag lexical class to our POSTag enum
    private func mapTag(_ nlTag: NLTag?) -> POSTag {
        guard let tag = nlTag else { return .unknown }
        switch tag {
        case .noun:                           return .noun
        case .verb:                           return .verb
        case .adjective:                      return .adjective
        case .adverb:                         return .adverb
        case .pronoun:                        return .pronoun
        case .determiner:                     return .determiner
        case .particle:                       return .particle
        case .preposition:                    return .preposition
        case .conjunction:                    return .conjunction
        case .interjection:                   return .interjection
        default:                              return .unknown
        }
    }

    /// Produce a simple lemma by lowercasing and stripping common inflections
    private func lemmatize(_ word: String, pos: POSTag) -> String {
        let lower = word.lowercased()
        if pos == .verb {
            // Strip common verb inflections
            if lower.hasSuffix("ing") && lower.count > 5 {
                let stem = String(lower.dropLast(3))
                // handle doubling: running → run
                if stem.count > 2 && stem.last == stem[stem.index(before: stem.endIndex)] {
                    return String(stem.dropLast())
                }
                return stem
            }
            if lower.hasSuffix("ed") && lower.count > 4 {
                return String(lower.dropLast(2))
            }
            if lower.hasSuffix("es") && lower.count > 4 {
                return String(lower.dropLast(2))
            }
            if lower.hasSuffix("s") && !lower.hasSuffix("ss") && lower.count > 3 {
                return String(lower.dropLast(1))
            }
        }
        if pos == .noun {
            if lower.hasSuffix("ies") && lower.count > 4 {
                return String(lower.dropLast(3)) + "y"
            }
            if lower.hasSuffix("ses") && lower.count > 4 {
                return String(lower.dropLast(2))
            }
            if lower.hasSuffix("s") && !lower.hasSuffix("ss") && lower.count > 3 {
                return String(lower.dropLast(1))
            }
        }
        if pos == .adjective {
            if lower.hasSuffix("er") && lower.count > 4 {
                return String(lower.dropLast(2))
            }
            if lower.hasSuffix("est") && lower.count > 5 {
                return String(lower.dropLast(3))
            }
        }
        return lower
    }

    /// Parse text into an array of Tokens with POS tags and lemmas
    func parse(text: String) -> [Token] {
        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = text

        var tokens: [Token] = []
        var index = 0

        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lexicalClass) { tag, range in
            let word = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !word.isEmpty else { return true }

            var pos = mapTag(tag)

            // If NLTagger returned unknown, fall back to morphological analysis
            if pos == .unknown {
                let morphResult = morphAnalyzer.analyze(word: word)
                pos = morphResult.estimatedPOS
            }

            let lemma = lemmatize(word, pos: pos)
            tokens.append(Token(word: word, pos: pos, lemma: lemma, index: index))
            index += 1
            return true
        }

        return tokens
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 3: SEMANTIC ROLE LABELER
// Pattern-based semantic role assignment using positional heuristics
// and preposition-driven role mapping.
// ═══════════════════════════════════════════════════════════════════

private struct SemanticRoleLabeler {

    /// Map prepositions to their most likely semantic role
    private static let prepRoleMap: [String: SemanticRole] = [
        "with":    .instrument,
        "using":   .instrument,
        "by":      .instrument,
        "in":      .location,
        "at":      .location,
        "on":      .location,
        "near":    .location,
        "inside":  .location,
        "within":  .location,
        "from":    .source,
        "to":      .goal,
        "toward":  .goal,
        "towards": .goal,
        "into":    .goal,
        "for":     .beneficiary,
        "during":  .time,
        "before":  .time,
        "after":   .time,
        "since":   .time,
        "until":   .time,
        "about":   .theme,
        "regarding": .theme,
        "concerning": .theme,
    ]

    /// Experiencer verbs: the subject experiences rather than acts
    private static let experiencerVerbs: Set<String> = [
        "feel", "see", "hear", "know", "believe", "think",
        "understand", "fear", "love", "hate", "like", "want",
        "need", "remember", "forget", "notice", "perceive",
        "realize", "recognize", "sense", "experience", "enjoy",
        "suffer", "endure", "appreciate", "prefer", "desire",
    ]

    /// Label semantic roles for all clauses in the token stream
    func labelRoles(tokens: [Token]) -> [SemanticFrame] {
        var frames: [SemanticFrame] = []

        // Split into clauses at conjunctions and sentence boundaries
        let clauses = splitIntoClauses(tokens: tokens)

        for clause in clauses {
            if let frame = buildFrame(clause: clause) {
                frames.append(frame)
            }
        }

        return frames
    }

    /// Split token stream into clause-like segments
    private func splitIntoClauses(tokens: [Token]) -> [[Token]] {
        var clauses: [[Token]] = []
        var current: [Token] = []

        for token in tokens {
            // Split on sentence-ending punctuation or clause-boundary conjunctions
            let lower = token.word.lowercased()
            if lower == "." || lower == "!" || lower == "?" || lower == ";" {
                if !current.isEmpty {
                    clauses.append(current)
                    current = []
                }
            } else if token.pos == .conjunction && (lower == "and" || lower == "but" || lower == "or" || lower == "because" || lower == "although" || lower == "while") {
                if current.count > 2 {
                    clauses.append(current)
                    current = []
                }
            } else {
                current.append(token)
            }
        }
        if !current.isEmpty {
            clauses.append(current)
        }

        return clauses
    }

    /// Build a single semantic frame from a clause
    private func buildFrame(clause: [Token]) -> SemanticFrame? {
        // Find the main verb
        guard let verbIndex = clause.firstIndex(where: { $0.pos == .verb }) else {
            return nil
        }
        let verb = clause[verbIndex]
        let verbLemma = verb.lemma.lowercased()

        var roles: [SemanticRole: [String]] = [:]

        // ── Subject (tokens before verb) ──
        let subjectTokens = clause.prefix(upTo: verbIndex).filter {
            $0.pos == .noun || $0.pos == .pronoun || $0.pos == .adjective
        }
        if !subjectTokens.isEmpty {
            let subjectWords = subjectTokens.map { $0.word }
            let isExperiencer = SemanticRoleLabeler.experiencerVerbs.contains(verbLemma)
            let role: SemanticRole = isExperiencer ? .experiencer : .agent
            roles[role] = subjectWords
        }

        // ── Object (nouns/pronouns after verb, before next preposition) ──
        let afterVerb = Array(clause.suffix(from: verbIndex + 1))
        var objectTokens: [Token] = []
        var prepositionalPhrases: [(prep: String, nouns: [String])] = []

        var i = 0
        while i < afterVerb.count {
            let tok = afterVerb[i]
            if tok.pos == .preposition {
                // Collect the prepositional phrase
                let prep = tok.word.lowercased()
                var ppNouns: [String] = []
                var j = i + 1
                while j < afterVerb.count && afterVerb[j].pos != .preposition && afterVerb[j].pos != .verb {
                    if afterVerb[j].pos == .noun || afterVerb[j].pos == .pronoun || afterVerb[j].pos == .adjective {
                        ppNouns.append(afterVerb[j].word)
                    }
                    j += 1
                }
                prepositionalPhrases.append((prep: prep, nouns: ppNouns))
                i = j
            } else if tok.pos == .noun || tok.pos == .pronoun {
                objectTokens.append(tok)
                i += 1
            } else {
                i += 1
            }
        }

        // Direct object → patient or theme
        if !objectTokens.isEmpty {
            let role: SemanticRole = SemanticRoleLabeler.experiencerVerbs.contains(verbLemma) ? .theme : .patient
            roles[role] = objectTokens.map { $0.word }
        }

        // ── Prepositional phrase roles ──
        for pp in prepositionalPhrases {
            guard !pp.nouns.isEmpty else { continue }
            if let role = SemanticRoleLabeler.prepRoleMap[pp.prep] {
                var existing = roles[role] ?? []
                existing.append(contentsOf: pp.nouns)
                roles[role] = existing
            } else {
                // Default: treat as theme
                var existing = roles[.theme] ?? []
                existing.append(contentsOf: pp.nouns)
                roles[.theme] = existing
            }
        }

        return SemanticFrame(predicate: verb.word, roles: roles)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 4: ANAPHORA RESOLVER
// Resolves pronouns to their most recent matching antecedent
// using gender/number agreement heuristics.
// ═══════════════════════════════════════════════════════════════════

private struct AnaphoraResolver {

    /// Gender/number categories for agreement checking
    enum GenderNumber: Equatable {
        case masculine
        case feminine
        case neuter
        case plural
        case unknown
    }

    /// Map pronouns to their gender/number category
    private static let pronounGender: [String: GenderNumber] = [
        "he": .masculine, "him": .masculine, "his": .masculine, "himself": .masculine,
        "she": .feminine, "her": .feminine, "hers": .feminine, "herself": .feminine,
        "it": .neuter, "its": .neuter, "itself": .neuter,
        "they": .plural, "them": .plural, "their": .plural, "theirs": .plural, "themselves": .plural,
    ]

    /// Common masculine nouns for agreement
    private static let masculineNouns: Set<String> = [
        "man", "boy", "father", "brother", "son", "husband", "uncle",
        "king", "prince", "gentleman", "sir", "mr", "grandfather",
        "nephew", "actor", "hero", "waiter", "policeman", "fireman",
        "businessman", "chairman", "spokesman", "he", "god",
    ]

    /// Common feminine nouns for agreement
    private static let feminineNouns: Set<String> = [
        "woman", "girl", "mother", "sister", "daughter", "wife", "aunt",
        "queen", "princess", "lady", "mrs", "ms", "grandmother",
        "niece", "actress", "heroine", "waitress", "policewoman",
        "businesswoman", "chairwoman", "spokeswoman", "she", "goddess",
    ]

    /// Nouns typically referred to with plural pronouns
    private static let pluralIndicators: Set<String> = [
        "people", "children", "students", "teachers", "workers",
        "members", "friends", "parents", "officers", "scientists",
        "citizens", "employees", "patients", "soldiers", "teams",
    ]

    /// Guess gender/number of a noun
    private func classifyNoun(_ word: String) -> GenderNumber {
        let lower = word.lowercased()
        if AnaphoraResolver.masculineNouns.contains(lower) { return .masculine }
        if AnaphoraResolver.feminineNouns.contains(lower) { return .feminine }
        if AnaphoraResolver.pluralIndicators.contains(lower) { return .plural }
        // Heuristic: words ending in -s are often plural (but not always)
        if lower.hasSuffix("s") && !lower.hasSuffix("ss") && !lower.hasSuffix("us") && !lower.hasSuffix("is") {
            return .plural
        }
        return .neuter  // Default: inanimate nouns are neuter
    }

    /// Check if a gender/number category agrees with another
    private func agrees(_ pronounGN: GenderNumber, _ nounGN: GenderNumber) -> Bool {
        if pronounGN == nounGN { return true }
        // Neuter pronouns can refer to unknown-gender nouns
        if pronounGN == .neuter && nounGN == .unknown { return true }
        if pronounGN == .unknown { return true }
        return false
    }

    /// Resolve all anaphoric references in the token stream
    func resolveAnaphora(tokens: [Token]) -> [Resolution] {
        var resolutions: [Resolution] = []

        // Build a list of candidate antecedents (nouns encountered so far)
        var nounStack: [(word: String, genderNumber: GenderNumber, index: Int)] = []

        for token in tokens {
            if token.pos == .noun {
                let gn = classifyNoun(token.word)
                nounStack.append((word: token.word, genderNumber: gn, index: token.index))
            }

            if token.pos == .pronoun {
                let lower = token.word.lowercased()
                guard let pronounGN = AnaphoraResolver.pronounGender[lower] else { continue }

                // Search backwards through noun stack for agreement
                var bestMatch: (word: String, index: Int)? = nil
                var bestDistance = Int.max

                for candidate in nounStack.reversed() {
                    if agrees(pronounGN, candidate.genderNumber) {
                        let distance = token.index - candidate.index
                        if distance > 0 && distance < bestDistance {
                            bestDistance = distance
                            bestMatch = (word: candidate.word, index: candidate.index)
                            break  // Take the most recent match
                        }
                    }
                }

                if let match = bestMatch {
                    // Confidence decays with distance, scaled by TAU
                    let distanceFactor = exp(-Double(bestDistance) * TAU * 0.1)
                    let confidence = min(1.0, max(0.1, distanceFactor))
                    resolutions.append(Resolution(
                        pronoun: token.word,
                        antecedent: match.word,
                        confidence: confidence
                    ))
                }
            }
        }

        return resolutions
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 5: DISCOURSE ANALYZER
// Detects rhetorical/discourse relations between sentences
// using discourse marker lexicons and positional heuristics.
// ═══════════════════════════════════════════════════════════════════

private struct DiscourseAnalyzer {

    /// Maps discourse markers to their rhetorical relation
    private static let markerMap: [(markers: [String], relation: DiscourseRelation)] = [
        // Contrast
        (["however", "but", "although", "nevertheless", "nonetheless",
          "on the other hand", "in contrast", "conversely", "yet",
          "whereas", "while", "despite", "even though", "on the contrary"],
         .contrast),

        // Cause
        (["because", "since", "due to", "owing to", "as a result of",
          "on account of", "for this reason", "the reason is"],
         .cause),

        // Result
        (["therefore", "so", "thus", "consequently", "hence",
          "as a result", "accordingly", "for this reason", "it follows that"],
         .result),

        // Temporal / Narration
        (["first", "then", "next", "finally", "afterward", "afterwards",
          "subsequently", "previously", "before", "after", "meanwhile",
          "simultaneously", "at the same time", "in the meantime",
          "later", "earlier", "soon", "eventually", "initially"],
         .temporal),

        // Elaboration
        (["for example", "for instance", "such as", "namely", "specifically",
          "in particular", "that is", "i.e.", "e.g.", "in other words",
          "to illustrate", "to put it another way", "more specifically"],
         .elaboration),

        // Condition
        (["if", "unless", "provided that", "assuming that", "in case",
          "on condition that", "supposing", "given that", "whether"],
         .condition),

        // Concession
        (["although", "even though", "despite", "in spite of",
          "granted that", "admittedly", "it is true that",
          "notwithstanding", "regardless"],
         .concession),

        // Explanation
        (["in fact", "actually", "indeed", "as a matter of fact",
          "the point is", "what this means is", "to clarify",
          "to explain", "this is because"],
         .explanation),

        // Background
        (["as we know", "it is well known", "historically",
          "traditionally", "as background", "to begin with",
          "fundamentally", "essentially"],
         .background),
    ]

    /// Split text into sentences using basic punctuation rules
    private func splitSentences(_ text: String) -> [String] {
        // Use NLTokenizer for sentence splitting
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let s = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !s.isEmpty {
                sentences.append(s)
            }
            return true
        }
        // Fallback if NLTokenizer produced nothing
        if sentences.isEmpty {
            sentences = text.components(separatedBy: CharacterSet(charactersIn: ".!?"))
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }
        return sentences
    }

    /// Detect discourse relation for a sentence based on marker presence
    private func detectRelation(_ sentence: String) -> (DiscourseRelation, Double) {
        let lower = sentence.lowercased()

        // Check multi-word markers first (longer matches are more reliable)
        let sortedGroups = DiscourseAnalyzer.markerMap.sorted { group1, group2 in
            let max1 = group1.markers.map { $0.count }.max() ?? 0
            let max2 = group2.markers.map { $0.count }.max() ?? 0
            return max1 > max2
        }

        for group in sortedGroups {
            for marker in group.markers.sorted(by: { $0.count > $1.count }) {
                // Check if marker appears at the beginning of the sentence (stronger signal)
                if lower.hasPrefix(marker) {
                    return (group.relation, 0.95)
                }
                // Check if marker appears anywhere in the sentence
                if lower.contains(marker) {
                    return (group.relation, 0.75)
                }
            }
        }

        // Default: narration (sequential continuation)
        return (.narration, 0.50)
    }

    /// Analyze discourse structure of a text
    func analyzeDiscourse(text: String) -> [DiscourseSegment] {
        let sentences = splitSentences(text)
        guard !sentences.isEmpty else { return [] }

        var segments: [DiscourseSegment] = []

        for (i, sentence) in sentences.enumerated() {
            let (relation, confidence) = detectRelation(sentence)

            // Target index: the sentence this one relates to
            // Usually the immediately preceding sentence, except for background (relates to whole)
            let targetIndex: Int
            switch relation {
            case .background:
                targetIndex = 0  // Background relates to the opening
            case .result, .cause:
                targetIndex = max(0, i - 1)  // Relates to previous
            default:
                targetIndex = max(0, i - 1)
            }

            // Coherence score: confidence modulated by position
            let positionFactor = 1.0 - (Double(i) / max(1.0, Double(sentences.count))) * TAU * 0.3
            let coherenceScore = min(1.0, confidence * positionFactor)

            segments.append(DiscourseSegment(
                text: sentence,
                relation: relation,
                targetIndex: targetIndex,
                coherenceScore: coherenceScore
            ))
        }

        return segments
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 6: PRAGMATIC INTERPRETER
// Classifies speech acts (question, command, request, statement)
// and infers communicative intent and conversational implicatures.
// ═══════════════════════════════════════════════════════════════════

private struct PragmaticInterpreter {

    /// WH-question words
    private static let whWords: Set<String> = [
        "who", "what", "where", "when", "why", "how",
        "which", "whose", "whom",
    ]

    /// Request markers (politeness indicators)
    private static let requestMarkers: [String] = [
        "please", "could you", "would you", "can you", "will you",
        "would you mind", "could you please", "i'd like", "i would like",
        "would it be possible", "do you mind", "may i ask",
        "if you could", "if you would", "kindly",
    ]

    /// Command-initial verbs (imperative markers)
    private static let imperativeVerbs: Set<String> = [
        "do", "make", "get", "put", "take", "give", "tell", "show",
        "find", "create", "build", "write", "read", "open", "close",
        "start", "stop", "run", "go", "come", "look", "listen",
        "try", "keep", "let", "help", "check", "fix", "set",
        "move", "turn", "send", "bring", "remove", "add", "use",
        "explain", "describe", "define", "list", "compare", "analyze",
        "calculate", "compute", "solve", "prove", "demonstrate",
    ]

    /// Warning markers
    private static let warningMarkers: [String] = [
        "warning", "caution", "danger", "be careful", "watch out",
        "beware", "alert", "don't forget", "remember that",
        "keep in mind", "be aware", "take note",
    ]

    /// Promise markers
    private static let promiseMarkers: [String] = [
        "i promise", "i will", "i'll", "i guarantee", "i assure",
        "you have my word", "i commit", "i pledge", "i swear",
    ]

    /// Suggestion markers
    private static let suggestionMarkers: [String] = [
        "maybe", "perhaps", "you might", "you could", "how about",
        "what about", "why not", "have you considered",
        "it might be", "one option is", "you may want to",
        "i suggest", "i recommend", "consider",
    ]

    /// Greeting patterns
    private static let greetingPatterns: [String] = [
        "hello", "hi", "hey", "greetings", "good morning",
        "good afternoon", "good evening", "howdy", "welcome",
        "nice to meet", "how are you", "what's up", "sup",
    ]

    /// Interpret pragmatic properties of a text
    func interpretPragmatics(text: String) -> PragmaticAnalysis {
        let lower = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let words = lower.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        let firstWord = words.first ?? ""

        var speechAct: SpeechAct = .statement
        var intent: Intent = .assertion
        var confidence: Double = 0.5
        var implicatures: [String] = []

        // ── Check for greetings first ──
        for pattern in PragmaticInterpreter.greetingPatterns {
            if lower.hasPrefix(pattern) {
                speechAct = .greeting
                intent = .acknowledgment
                confidence = 0.95
                implicatures.append("Social bonding / phatic communion")
                return PragmaticAnalysis(speechAct: speechAct, intent: intent,
                                         confidence: confidence, implicatures: implicatures)
            }
        }

        // ── Questions ──
        if lower.contains("?") || PragmaticInterpreter.whWords.contains(firstWord) {
            speechAct = .question
            intent = .query
            confidence = 0.92

            if PragmaticInterpreter.whWords.contains(firstWord) {
                implicatures.append("Speaker lacks information about \(firstWord)-argument")
                intent = .requestInfo
                confidence = 0.95
            }

            // Rhetorical question detection
            if lower.contains("isn't it obvious") || lower.contains("who doesn't") ||
               lower.contains("doesn't everyone") || lower.contains("how could anyone") {
                implicatures.append("Rhetorical question — assertion disguised as question")
                intent = .assertion
                confidence = 0.80
            }

            // Yes/no question
            let ynStarters: Set<String> = ["is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "have", "has", "had"]
            if ynStarters.contains(firstWord) {
                implicatures.append("Yes/no question — expects binary confirmation")
            }
        }

        // ── Requests (check before commands — requests are polite commands) ──
        if speechAct == .statement {
            for marker in PragmaticInterpreter.requestMarkers {
                if lower.contains(marker) {
                    speechAct = .request
                    intent = .directive
                    confidence = 0.90
                    implicatures.append("Indirect speech act: politeness strategy (Brown & Levinson)")
                    implicatures.append("Implicature: speaker wants hearer to perform action")
                    break
                }
            }
        }

        // ── Commands (imperative) ──
        if speechAct == .statement {
            if PragmaticInterpreter.imperativeVerbs.contains(firstWord) {
                speechAct = .command
                intent = .directive
                confidence = 0.85
                implicatures.append("Direct speech act: imperative mood")
            }
        }

        // ── Warnings ──
        if speechAct == .statement {
            for marker in PragmaticInterpreter.warningMarkers {
                if lower.contains(marker) {
                    speechAct = .warning
                    intent = .directive
                    confidence = 0.88
                    implicatures.append("Speaker perceives risk or danger")
                    break
                }
            }
        }

        // ── Promises ──
        if speechAct == .statement {
            for marker in PragmaticInterpreter.promiseMarkers {
                if lower.contains(marker) {
                    speechAct = .promise
                    intent = .assertion
                    confidence = 0.85
                    implicatures.append("Speaker commits to future action (commissive)")
                    break
                }
            }
        }

        // ── Suggestions ──
        if speechAct == .statement {
            for marker in PragmaticInterpreter.suggestionMarkers {
                if lower.contains(marker) {
                    speechAct = .suggestion
                    intent = .directive
                    confidence = 0.82
                    implicatures.append("Hedged directive — speaker offers option, not obligation")
                    break
                }
            }
        }

        // ── Check for clarification intent ──
        let clarificationMarkers = ["what do you mean", "could you clarify", "i don't understand",
                                     "what does that mean", "can you explain", "in what sense",
                                     "are you saying", "do you mean"]
        for marker in clarificationMarkers {
            if lower.contains(marker) {
                intent = .clarification
                implicatures.append("Speaker signals comprehension failure — repair needed")
                break
            }
        }

        // ── Check for acknowledgment ──
        let ackMarkers = ["i see", "got it", "understood", "okay", "ok", "right",
                          "sure", "of course", "certainly", "absolutely", "indeed",
                          "fair enough", "i agree", "that makes sense", "exactly"]
        for marker in ackMarkers {
            if lower == marker || lower.hasPrefix(marker + " ") || lower.hasPrefix(marker + ".") || lower.hasPrefix(marker + ",") {
                intent = .acknowledgment
                implicatures.append("Backchannel signal — speaker confirms understanding")
                break
            }
        }

        // If still a plain statement, add default implicature
        if speechAct == .statement && implicatures.isEmpty {
            implicatures.append("Assertive speech act — speaker presents propositional content as true")
        }

        return PragmaticAnalysis(
            speechAct: speechAct,
            intent: intent,
            confidence: confidence,
            implicatures: implicatures
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 7: PRESUPPOSITION ENGINE
// Extracts presuppositions triggered by lexical items, syntactic
// structures, and aspectual verbs. Based on Karttunen/Stalnaker theory.
// ═══════════════════════════════════════════════════════════════════

private struct PresuppositionEngine {

    /// Factive verbs: presuppose the truth of their complement
    private static let factiveVerbs: Set<String> = [
        "know", "knows", "knew", "known",
        "realize", "realizes", "realized",
        "discover", "discovers", "discovered",
        "regret", "regrets", "regretted",
        "notice", "notices", "noticed",
        "remember", "remembers", "remembered",
        "forget", "forgets", "forgot", "forgotten",
        "reveal", "reveals", "revealed",
        "recognize", "recognizes", "recognized",
        "acknowledge", "acknowledges", "acknowledged",
        "understand", "understands", "understood",
        "appreciate", "appreciates", "appreciated",
        "learn", "learns", "learned",
        "see", "sees", "saw", "seen",
        "find", "finds", "found",
        "prove", "proves", "proved", "proven",
    ]

    /// Non-factive verbs: do NOT presuppose the truth of their complement
    private static let nonFactiveVerbs: Set<String> = [
        "believe", "believes", "believed",
        "think", "thinks", "thought",
        "hope", "hopes", "hoped",
        "assume", "assumes", "assumed",
        "imagine", "imagines", "imagined",
        "suppose", "supposes", "supposed",
        "suspect", "suspects", "suspected",
        "wish", "wishes", "wished",
        "dream", "dreams", "dreamed", "dreamt",
        "pretend", "pretends", "pretended",
        "claim", "claims", "claimed",
        "allege", "alleges", "alleged",
        "suggest", "suggests", "suggested",
        "fear", "fears", "feared",
        "doubt", "doubts", "doubted",
        "expect", "expects", "expected",
    ]

    /// Aspectual/change-of-state verbs: presuppose a prior state
    private static let aspectualVerbs: Set<String> = [
        "stop", "stops", "stopped",
        "start", "starts", "started",
        "begin", "begins", "began", "begun",
        "continue", "continues", "continued",
        "resume", "resumes", "resumed",
        "finish", "finishes", "finished",
        "cease", "ceases", "ceased",
        "keep", "keeps", "kept",
        "quit", "quits",
        "again",  // adverb but triggers existential presupposition
    ]

    /// Cleft/structural presupposition markers
    private static let structuralMarkers: [String] = [
        "it is", "it was", "it were",
        "the one who", "the thing that",
        "what happened was", "the reason is",
        "the fact that",
    ]

    /// Extract all presuppositions from text
    func extractPresuppositions(text: String) -> [Presupposition] {
        var presuppositions: [Presupposition] = []
        let lower = text.lowercased()
        let words = lower.components(separatedBy: .whitespaces).filter { !$0.isEmpty }

        // ── Definite article presuppositions (existential) ──
        for (i, word) in words.enumerated() {
            if word == "the" && i + 1 < words.count {
                // Collect the noun phrase after "the"
                var np: [String] = []
                var j = i + 1
                while j < words.count && j < i + 5 {
                    let w = words[j].trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
                    if w.isEmpty { break }
                    np.append(w)
                    // Stop at verbs or prepositions (rough heuristic)
                    if ["is", "are", "was", "were", "has", "have", "had", "will", "would", "can", "could",
                        "in", "on", "at", "to", "for", "with", "from", "by"].contains(w) {
                        np.removeLast()
                        break
                    }
                    j += 1
                }
                if !np.isEmpty {
                    let nounPhrase = np.joined(separator: " ")
                    presuppositions.append(Presupposition(
                        type: .existential,
                        content: "There exists a unique/salient '\(nounPhrase)'",
                        confidence: 0.85
                    ))
                }
            }
        }

        // ── Aspectual verb presuppositions (existential — prior state) ──
        for (i, word) in words.enumerated() {
            let clean = word.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
            if PresuppositionEngine.aspectualVerbs.contains(clean) {
                // Collect complement
                let complement = words[(i + 1)..<min(words.count, i + 8)]
                    .map { $0.trimmingCharacters(in: CharacterSet.alphanumerics.inverted) }
                    .filter { !$0.isEmpty }
                    .joined(separator: " ")
                if !complement.isEmpty {
                    let prior: String
                    if clean.hasPrefix("stop") || clean.hasPrefix("ceas") || clean.hasPrefix("finish") || clean.hasPrefix("quit") {
                        prior = "The activity '\(complement)' was previously in progress"
                    } else if clean.hasPrefix("start") || clean.hasPrefix("begin") {
                        prior = "The activity '\(complement)' was not previously in progress"
                    } else if clean.hasPrefix("continu") || clean.hasPrefix("resum") || clean.hasPrefix("keep") || clean == "again" {
                        prior = "The activity '\(complement)' was already in progress or occurred before"
                    } else {
                        prior = "A prior state related to '\(complement)' existed"
                    }
                    presuppositions.append(Presupposition(
                        type: .existential,
                        content: prior,
                        confidence: 0.88
                    ))
                }
            }
        }

        // ── Factive verb presuppositions (truth of complement) ──
        for (i, word) in words.enumerated() {
            let clean = word.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
            if PresuppositionEngine.factiveVerbs.contains(clean) {
                let complement = extractComplement(words: words, verbIndex: i)
                if !complement.isEmpty {
                    presuppositions.append(Presupposition(
                        type: .factive,
                        content: "It is true that \(complement)",
                        confidence: 0.90
                    ))
                }
            }
        }

        // ── Non-factive verb presuppositions (no truth commitment) ──
        for (i, word) in words.enumerated() {
            let clean = word.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
            if PresuppositionEngine.nonFactiveVerbs.contains(clean) {
                let complement = extractComplement(words: words, verbIndex: i)
                if !complement.isEmpty {
                    presuppositions.append(Presupposition(
                        type: .nonFactive,
                        content: "It is not necessarily true that \(complement)",
                        confidence: 0.80
                    ))
                }
            }
        }

        // ── Structural presuppositions (cleft sentences, etc.) ──
        for marker in PresuppositionEngine.structuralMarkers {
            if lower.contains(marker) {
                // Extract the rest of the clause after the marker
                if let range = lower.range(of: marker) {
                    let after = String(lower[range.upperBound...])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if after.count > 3 {
                        let truncated = String(after.prefix(80))
                        presuppositions.append(Presupposition(
                            type: .structural,
                            content: "Structural presupposition: something relevant exists regarding '\(truncated)'",
                            confidence: 0.75
                        ))
                    }
                }
            }
        }

        // ── Possessive presuppositions (existential) ──
        // Patterns like "John's car", "my house" etc. presuppose existence
        let possessivePattern = words.enumerated().filter {
            $0.element.hasSuffix("'s") || $0.element.hasSuffix("s'")
        }
        for (i, word) in possessivePattern {
            let owner = word.replacingOccurrences(of: "'s", with: "").replacingOccurrences(of: "s'", with: "")
            if i + 1 < words.count {
                let possessed = words[i + 1].trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
                presuppositions.append(Presupposition(
                    type: .existential,
                    content: "'\(owner)' exists and has a '\(possessed)'",
                    confidence: 0.82
                ))
            }
        }

        return presuppositions
    }

    /// Extract the complement clause following a verb
    private func extractComplement(words: [String], verbIndex: Int) -> String {
        // Skip optional "that" complementizer
        var start = verbIndex + 1
        if start < words.count {
            let next = words[start].trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
            if next == "that" { start += 1 }
        }
        // Collect up to 10 words or until sentence boundary
        let sentenceEnders: Set<Character> = [".", "!", "?", ";"]
        var complement: [String] = []
        var j = start
        while j < words.count && complement.count < 10 {
            let w = words[j]
            if w.last.map({ sentenceEnders.contains($0) }) == true {
                let cleaned = w.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
                if !cleaned.isEmpty { complement.append(cleaned) }
                break
            }
            complement.append(w.trimmingCharacters(in: CharacterSet.alphanumerics.inverted))
            j += 1
        }
        return complement.filter { !$0.isEmpty }.joined(separator: " ")
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 8: SENTIMENT ENGINE
// Lexicon-based sentiment analysis with negation handling,
// intensifier modulation, and mixed-polarity detection.
// ═══════════════════════════════════════════════════════════════════

private struct SentimentEngine {

    // ─── POSITIVE LEXICON (~50 words) ───
    private static let positiveWords: [String: Double] = [
        "good": 0.6, "great": 0.8, "excellent": 0.9, "wonderful": 0.9,
        "happy": 0.8, "love": 0.9, "best": 0.9, "amazing": 0.9,
        "beautiful": 0.8, "perfect": 1.0, "fantastic": 0.9, "brilliant": 0.9,
        "outstanding": 0.9, "superb": 0.9, "awesome": 0.8, "nice": 0.5,
        "pleasant": 0.6, "delightful": 0.8, "enjoyable": 0.7, "remarkable": 0.8,
        "impressive": 0.8, "marvelous": 0.9, "terrific": 0.8, "fabulous": 0.8,
        "glorious": 0.8, "splendid": 0.8, "magnificent": 0.9, "exceptional": 0.9,
        "phenomenal": 0.9, "extraordinary": 0.9, "incredible": 0.8, "joyful": 0.8,
        "cheerful": 0.7, "grateful": 0.7, "thankful": 0.7, "pleased": 0.7,
        "satisfied": 0.6, "optimistic": 0.7, "hopeful": 0.6, "excited": 0.8,
        "thrilled": 0.8, "delighted": 0.8, "fortunate": 0.7, "lucky": 0.6,
        "blessed": 0.7, "wonderful": 0.9, "successful": 0.7, "triumph": 0.8,
        "victory": 0.7, "win": 0.6, "elegant": 0.7, "graceful": 0.7,
    ]

    // ─── NEGATIVE LEXICON (~50 words) ───
    private static let negativeWords: [String: Double] = [
        "bad": 0.6, "terrible": 0.9, "awful": 0.9, "horrible": 0.9,
        "hate": 0.9, "worst": 1.0, "ugly": 0.7, "stupid": 0.7,
        "boring": 0.6, "annoying": 0.6, "disgusting": 0.9, "dreadful": 0.8,
        "pathetic": 0.8, "miserable": 0.8, "disappointing": 0.7, "frustrating": 0.7,
        "angry": 0.7, "furious": 0.9, "sad": 0.7, "depressing": 0.8,
        "tragic": 0.8, "painful": 0.7, "suffering": 0.8, "cruel": 0.8,
        "evil": 0.9, "wicked": 0.8, "vile": 0.9, "nasty": 0.7,
        "horrendous": 0.9, "atrocious": 0.9, "abysmal": 0.9, "disastrous": 0.9,
        "catastrophic": 0.9, "devastating": 0.9, "ruined": 0.8, "destroyed": 0.8,
        "failed": 0.7, "failure": 0.7, "broken": 0.6, "damaged": 0.6,
        "fear": 0.7, "afraid": 0.7, "terrified": 0.8, "horrified": 0.8,
        "lonely": 0.7, "hopeless": 0.8, "worthless": 0.9, "useless": 0.8,
        "inferior": 0.7, "mediocre": 0.5, "poor": 0.6, "weak": 0.5,
    ]

    // ─── NEGATION WORDS ───
    private static let negators: Set<String> = [
        "not", "no", "never", "neither", "nor", "nothing",
        "nobody", "nowhere", "hardly", "barely", "scarcely",
        "seldom", "rarely", "without",
    ]

    // ─── INTENSIFIERS ───
    private static let intensifiers: [String: Double] = [
        "very": 1.5, "extremely": 2.0, "incredibly": 2.0,
        "absolutely": 2.0, "completely": 1.8, "totally": 1.8,
        "utterly": 2.0, "really": 1.4, "truly": 1.5,
        "particularly": 1.3, "especially": 1.4, "remarkably": 1.6,
        "exceptionally": 1.7, "tremendously": 1.8, "enormously": 1.7,
        "immensely": 1.8, "highly": 1.5, "deeply": 1.5,
        "profoundly": 1.8, "intensely": 1.6, "so": 1.3,
        "quite": 1.2, "rather": 1.1, "fairly": 1.1,
    ]

    // ─── DIMINISHERS ───
    private static let diminishers: [String: Double] = [
        "slightly": 0.5, "somewhat": 0.6, "a bit": 0.5,
        "a little": 0.5, "kind of": 0.6, "sort of": 0.6,
        "mildly": 0.5, "marginally": 0.4, "partly": 0.6,
    ]

    /// Analyze sentiment of text
    func analyzeSentiment(text: String) -> SentimentResult {
        let words = tokenize(text)
        var positiveScore: Double = 0.0
        var negativeScore: Double = 0.0
        var wordCount: Double = 0.0

        var i = 0
        while i < words.count {
            let word = words[i]

            // Check for negation within a window of 3 words before
            let isNegated = hasNegationBefore(words: words, index: i, window: 3)

            // Check for contraction-based negation (e.g., "don't", "isn't", "wasn't")
            let contractionNegated = word.contains("n't") || (i > 0 && words[i - 1].contains("n't"))

            let negated = isNegated || contractionNegated

            // Check for intensifier before this word
            let intensifierMultiplier = getIntensifierMultiplier(words: words, index: i)

            // Look up sentiment
            if let posScore = SentimentEngine.positiveWords[word] {
                let adjusted = posScore * intensifierMultiplier
                if negated {
                    negativeScore += adjusted * 0.8  // Negated positive → weaker negative
                } else {
                    positiveScore += adjusted
                }
                wordCount += 1
            } else if let negScore = SentimentEngine.negativeWords[word] {
                let adjusted = negScore * intensifierMultiplier
                if negated {
                    positiveScore += adjusted * 0.5  // Negated negative → weaker positive
                } else {
                    negativeScore += adjusted
                }
                wordCount += 1
            }

            i += 1
        }

        // Normalize scores
        let normalizer = max(1.0, wordCount)
        let normPos = positiveScore / normalizer
        let normNeg = negativeScore / normalizer

        // Determine polarity
        let polarity: SentimentPolarity
        let diff = normPos - normNeg
        if normPos > 0.01 && normNeg > 0.01 && abs(diff) < 0.2 {
            polarity = .mixed
        } else if diff > 0.05 {
            polarity = .positive
        } else if diff < -0.05 {
            polarity = .negative
        } else {
            polarity = .neutral
        }

        // Intensity: overall magnitude of sentiment
        let intensity = min(1.0, (normPos + normNeg) * PHI * 0.5)

        return SentimentResult(
            polarity: polarity,
            intensity: intensity,
            positiveScore: min(1.0, normPos),
            negativeScore: min(1.0, normNeg)
        )
    }

    /// Tokenize text into lowercase words, preserving contractions
    private func tokenize(_ text: String) -> [String] {
        let cleaned = text.lowercased()
        // Split on whitespace and punctuation, but keep apostrophes within words
        var words: [String] = []
        var current = ""
        for char in cleaned {
            if char.isLetter || char == "'" || char == "\u{2019}" {
                current.append(char)
            } else {
                if !current.isEmpty {
                    words.append(current)
                    current = ""
                }
            }
        }
        if !current.isEmpty { words.append(current) }
        return words
    }

    /// Check if a negation word appears within `window` words before index
    private func hasNegationBefore(words: [String], index: Int, window: Int) -> Bool {
        let start = max(0, index - window)
        for j in start..<index {
            if SentimentEngine.negators.contains(words[j]) {
                return true
            }
            // Check for "n't" contraction
            if words[j].hasSuffix("n't") || words[j].hasSuffix("n\u{2019}t") {
                return true
            }
        }
        return false
    }

    /// Get intensifier/diminisher multiplier from words before index
    private func getIntensifierMultiplier(words: [String], index: Int) -> Double {
        guard index > 0 else { return 1.0 }

        // Check single-word intensifiers/diminishers
        let prev = words[index - 1]
        if let mult = SentimentEngine.intensifiers[prev] { return mult }
        if let mult = SentimentEngine.diminishers[prev] { return mult }

        // Check two-word diminishers (e.g., "a bit", "a little", "kind of")
        if index > 1 {
            let twoWord = words[index - 2] + " " + words[index - 1]
            if let mult = SentimentEngine.diminishers[twoWord] { return mult }
        }

        return 1.0
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 9: COHERENCE SCORER
// Evaluates text coherence across three dimensions:
// lexical cohesion, reference continuity, and entity continuity.
// ═══════════════════════════════════════════════════════════════════

private struct CoherenceScorer {

    /// Common function words to exclude from lexical cohesion calculation
    private static let functionWords: Set<String> = [
        "the", "a", "an", "is", "am", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "shall", "should", "may", "might", "must",
        "can", "could", "to", "of", "in", "for", "on", "with", "at",
        "by", "from", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "out", "off", "over",
        "under", "and", "but", "or", "nor", "not", "so", "yet",
        "this", "that", "these", "those", "it", "its", "he", "she",
        "they", "them", "their", "we", "our", "you", "your",
        "i", "me", "my", "if", "then", "than", "when", "which",
        "what", "who", "how", "where", "there", "here",
    ]

    /// Pronouns for reference continuity tracking
    private static let pronouns: Set<String> = [
        "he", "him", "his", "she", "her", "hers", "it", "its",
        "they", "them", "their", "theirs", "we", "us", "our", "ours",
        "this", "that", "these", "those", "who", "which", "whom",
    ]

    /// Score coherence of a text across multiple dimensions
    func scoreCoherence(text: String) -> CoherenceResult {
        let sentences = splitSentences(text)
        guard sentences.count > 1 else {
            // Single sentence: maximum coherence by default
            return CoherenceResult(overallScore: 1.0, lexicalCohesion: 1.0,
                                   referenceContinuity: 1.0, entityContinuity: 1.0)
        }

        // Tokenize each sentence into content words
        let sentenceWords: [[String]] = sentences.map { sentence in
            sentence.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count > 2 && !CoherenceScorer.functionWords.contains($0) }
        }

        // ── Dimension 1: Lexical Cohesion ──
        // Proportion of adjacent sentence pairs sharing at least one content word
        let lexicalCohesion = computeLexicalCohesion(sentenceWords: sentenceWords)

        // ── Dimension 2: Reference Continuity ──
        // Measure pronoun chains across sentences (a pronoun in sentence N
        // should have a noun antecedent in sentence N-1)
        let referenceContinuity = computeReferenceContinuity(sentences: sentences)

        // ── Dimension 3: Entity Continuity ──
        // Track capitalized words (potential named entities) across sentences
        let entityContinuity = computeEntityContinuity(sentences: sentences)

        // ── Overall Score: PHI-weighted combination ──
        // Weights: lexical (TAU), reference (TAU^2), entity (TAU^3) — normalized
        let w1 = TAU                       // 0.618
        let w2 = TAU * TAU                 // 0.382
        let w3 = TAU * TAU * TAU           // 0.236
        let totalWeight = w1 + w2 + w3
        let overall = (w1 * lexicalCohesion + w2 * referenceContinuity + w3 * entityContinuity) / totalWeight

        return CoherenceResult(
            overallScore: min(1.0, max(0.0, overall)),
            lexicalCohesion: min(1.0, max(0.0, lexicalCohesion)),
            referenceContinuity: min(1.0, max(0.0, referenceContinuity)),
            entityContinuity: min(1.0, max(0.0, entityContinuity))
        )
    }

    /// Split text into sentences using NLTokenizer
    private func splitSentences(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let s = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !s.isEmpty { sentences.append(s) }
            return true
        }
        if sentences.isEmpty {
            sentences = text.components(separatedBy: CharacterSet(charactersIn: ".!?"))
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }
        return sentences
    }

    /// Lexical cohesion: ratio of adjacent sentence pairs sharing content words
    private func computeLexicalCohesion(sentenceWords: [[String]]) -> Double {
        guard sentenceWords.count > 1 else { return 1.0 }

        var overlapCount = 0
        var totalPairs = 0

        for i in 1..<sentenceWords.count {
            let prevSet = Set(sentenceWords[i - 1])
            let currSet = Set(sentenceWords[i])
            let overlap = prevSet.intersection(currSet).count
            let unionSize = max(1, prevSet.union(currSet).count)

            // Jaccard-like coefficient with PHI scaling
            let jaccardCoeff = Double(overlap) / Double(unionSize)
            if jaccardCoeff > 0 {
                overlapCount += 1
            }
            totalPairs += 1
        }

        // Base cohesion: proportion of pairs with overlap
        let baseCohesion = Double(overlapCount) / max(1.0, Double(totalPairs))

        // Also compute average Jaccard across all pairs for smoother scoring
        var totalJaccard = 0.0
        for i in 1..<sentenceWords.count {
            let prevSet = Set(sentenceWords[i - 1])
            let currSet = Set(sentenceWords[i])
            let overlap = prevSet.intersection(currSet).count
            let unionSize = max(1, prevSet.union(currSet).count)
            totalJaccard += Double(overlap) / Double(unionSize)
        }
        let avgJaccard = totalJaccard / max(1.0, Double(sentenceWords.count - 1))

        // Blend: pair presence (discrete) + jaccard (continuous)
        return baseCohesion * 0.5 + min(1.0, avgJaccard * PHI) * 0.5
    }

    /// Reference continuity: proportion of pronouns with antecedent nouns in prior sentence
    private func computeReferenceContinuity(sentences: [String]) -> Double {
        guard sentences.count > 1 else { return 1.0 }

        // Tag each sentence to find nouns and pronouns
        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        var sentenceNouns: [[String]] = []
        var sentencePronouns: [[String]] = []

        for sentence in sentences {
            tagger.string = sentence
            var nouns: [String] = []
            var pronouns: [String] = []

            tagger.enumerateTags(in: sentence.startIndex..<sentence.endIndex, unit: .word, scheme: .lexicalClass) { tag, range in
                let word = String(sentence[range]).lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
                guard !word.isEmpty else { return true }
                if tag == .noun { nouns.append(word) }
                if tag == .pronoun || CoherenceScorer.pronouns.contains(word) { pronouns.append(word) }
                return true
            }
            sentenceNouns.append(nouns)
            sentencePronouns.append(pronouns)
        }

        // For each sentence with pronouns, check if previous sentence had nouns
        var resolvedCount = 0
        var totalPronouns = 0

        for i in 1..<sentences.count {
            let pronouns = sentencePronouns[i]
            let prevNouns = sentenceNouns[i - 1]

            totalPronouns += pronouns.count
            // If there are nouns in the previous sentence, pronouns are "resolvable"
            if !prevNouns.isEmpty {
                resolvedCount += pronouns.count
            }
        }

        if totalPronouns == 0 {
            // No pronouns at all: not penalized, moderate continuity
            return 0.7
        }

        return Double(resolvedCount) / max(1.0, Double(totalPronouns))
    }

    /// Entity continuity: track capitalized tokens (potential named entities) across sentences
    private func computeEntityContinuity(sentences: [String]) -> Double {
        guard sentences.count > 1 else { return 1.0 }

        // Extract capitalized words (skip sentence-initial words)
        var sentenceEntities: [Set<String>] = []

        for sentence in sentences {
            let words = sentence.components(separatedBy: .whitespaces)
            var entities: Set<String> = []
            for (j, word) in words.enumerated() {
                let cleaned = word.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
                guard cleaned.count > 1 else { continue }
                // Skip the first word (always capitalized at sentence start)
                if j > 0 && cleaned.first?.isUppercase == true {
                    entities.insert(cleaned.lowercased())
                }
            }
            sentenceEntities.append(entities)
        }

        // Check entity overlap between adjacent sentences
        var continuityScore = 0.0
        var pairsWithEntities = 0

        for i in 1..<sentenceEntities.count {
            let prev = sentenceEntities[i - 1]
            let curr = sentenceEntities[i]
            if prev.isEmpty && curr.isEmpty { continue }
            pairsWithEntities += 1

            let overlap = prev.intersection(curr).count
            if !prev.isEmpty || !curr.isEmpty {
                let unionSize = max(1, prev.union(curr).count)
                continuityScore += Double(overlap) / Double(unionSize)
            }
        }

        if pairsWithEntities == 0 {
            // No named entities detected: moderate default
            return 0.6
        }

        return continuityScore / Double(pairsWithEntities)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 10: DEEP COMPREHENSION
// Fuses all 9 preceding layers with PHI-weighted scoring to
// produce a single comprehension metric and unified NLU result.
// ═══════════════════════════════════════════════════════════════════

private struct DeepComprehension {

    /// PHI-harmonic weights for each layer (PHI-scaled Fibonacci-like distribution)
    /// Layers ordered by their analytical depth:
    ///   1=Morphology, 2=Syntax, 3=SRL, 4=Anaphora, 5=Discourse,
    ///   6=Pragmatics, 7=Presupposition, 8=Sentiment, 9=Coherence
    private static let layerWeights: [Double] = [
        0.08,   // L1: Morphological (surface-level)
        0.10,   // L2: Syntactic (structural)
        0.12,   // L3: Semantic Role (predicate-argument)
        0.09,   // L4: Anaphora (reference binding)
        0.14,   // L5: Discourse (rhetorical structure)
        0.13,   // L6: Pragmatics (speech act + intent)
        0.08,   // L7: Presupposition (implicit meaning)
        0.10,   // L8: Sentiment (affective)
        0.16,   // L9: Coherence (global text quality)
    ]

    /// Compute a layer score for morphological analysis
    private func scoreMorphology(_ results: [MorphologicalResult]) -> Double {
        guard !results.isEmpty else { return 0.0 }
        // Score based on: how many words were decomposable (had affixes)
        let decomposable = results.filter { !$0.prefixes.isEmpty || !$0.suffixes.isEmpty }
        let decompositionRate = Double(decomposable.count) / Double(results.count)
        // Score based on: how many words got a non-unknown POS
        let tagged = results.filter { $0.estimatedPOS != .unknown }
        let tagRate = Double(tagged.count) / Double(results.count)
        return (decompositionRate * 0.4 + tagRate * 0.6) * PHI * 0.618
    }

    /// Compute a layer score for syntactic parsing
    private func scoreSyntax(_ tokens: [Token]) -> Double {
        guard !tokens.isEmpty else { return 0.0 }
        // Score based on: POS coverage (non-unknown tags)
        let tagged = tokens.filter { $0.pos != .unknown }
        let coverage = Double(tagged.count) / Double(tokens.count)
        // Bonus for diversity of POS tags
        let uniqueTags = Set(tokens.map { $0.pos })
        let diversity = Double(uniqueTags.count) / Double(POSTag.allCases.count)
        return min(1.0, coverage * 0.7 + diversity * 0.3)
    }

    /// Compute a layer score for semantic role labeling
    private func scoreSRL(_ frames: [SemanticFrame]) -> Double {
        guard !frames.isEmpty else { return 0.0 }
        // Score based on: role coverage per frame
        var totalRoleCount = 0
        for frame in frames {
            totalRoleCount += frame.roles.count
        }
        let avgRoles = Double(totalRoleCount) / Double(frames.count)
        // Normalize: a good frame has 2-4 roles
        return min(1.0, avgRoles / 3.0)
    }

    /// Compute a layer score for anaphora resolution
    private func scoreAnaphora(_ resolutions: [Resolution], tokenCount: Int) -> Double {
        guard tokenCount > 0 else { return 0.0 }
        if resolutions.isEmpty {
            // No pronouns to resolve: not penalized, moderate score
            return 0.6
        }
        // Average confidence of resolutions
        let avgConfidence = resolutions.map { $0.confidence }.reduce(0, +) / Double(resolutions.count)
        return min(1.0, avgConfidence)
    }

    /// Compute a layer score for discourse analysis
    private func scoreDiscourse(_ segments: [DiscourseSegment]) -> Double {
        guard !segments.isEmpty else { return 0.0 }
        // Average coherence score across segments
        let avgCoherence = segments.map { $0.coherenceScore }.reduce(0, +) / Double(segments.count)
        // Bonus for diverse discourse relations
        let uniqueRelations = Set(segments.map { $0.relation })
        let diversity = Double(uniqueRelations.count) / max(1.0, Double(min(segments.count, DiscourseRelation.allCases.count)))
        return min(1.0, avgCoherence * 0.7 + diversity * 0.3)
    }

    /// Compute a layer score for pragmatic interpretation
    private func scorePragmatics(_ analysis: PragmaticAnalysis) -> Double {
        // Confidence of speech act classification + richness of implicatures
        let implicatureBonus = min(0.3, Double(analysis.implicatures.count) * 0.1)
        return min(1.0, analysis.confidence * 0.8 + implicatureBonus + 0.1)
    }

    /// Compute a layer score for presupposition extraction
    private func scorePresuppositions(_ presuppositions: [Presupposition]) -> Double {
        if presuppositions.isEmpty { return 0.3 }  // Some texts have no presuppositions
        let avgConfidence = presuppositions.map { $0.confidence }.reduce(0, +) / Double(presuppositions.count)
        // Bonus for diversity of presupposition types
        let uniqueTypes = Set(presuppositions.map { $0.type })
        let diversity = Double(uniqueTypes.count) / Double(PresuppositionType.allCases.count)
        return min(1.0, avgConfidence * 0.7 + diversity * 0.3)
    }

    /// Compute a layer score for sentiment analysis
    private func scoreSentiment(_ result: SentimentResult) -> Double {
        // Sentiment analysis quality: intensity indicates opinion strength,
        // non-neutral detection indicates analytical depth
        if result.polarity == .neutral {
            return 0.4  // Neutral text: analysis worked but found nothing strong
        }
        return min(1.0, result.intensity * 0.6 + 0.4)
    }

    /// Fuse all layers into a single comprehension score and DeepNLUResult
    func comprehend(
        text: String,
        morphology: [MorphologicalResult],
        tokens: [Token],
        semanticFrames: [SemanticFrame],
        anaphoraResolutions: [Resolution],
        discourse: [DiscourseSegment],
        pragmatics: PragmaticAnalysis,
        presuppositions: [Presupposition],
        sentiment: SentimentResult,
        coherence: CoherenceResult
    ) -> DeepNLUResult {
        // Compute individual layer scores
        let layerScores: [Double] = [
            scoreMorphology(morphology),
            scoreSyntax(tokens),
            scoreSRL(semanticFrames),
            scoreAnaphora(anaphoraResolutions, tokenCount: tokens.count),
            scoreDiscourse(discourse),
            scorePragmatics(pragmatics),
            scorePresuppositions(presuppositions),
            scoreSentiment(sentiment),
            coherence.overallScore,
        ]

        // PHI-weighted combination
        var weightedSum = 0.0
        var totalWeight = 0.0
        for (i, score) in layerScores.enumerated() {
            let weight = DeepComprehension.layerWeights[i]
            weightedSum += score * weight
            totalWeight += weight
        }

        // Normalize and apply PHI scaling
        let rawScore = weightedSum / max(0.001, totalWeight)
        let phiNormalized = rawScore * (PHI / (PHI + TAU))  // PHI/(PHI+TAU) = PHI/PHI^2 = 1/PHI = TAU ≈ 0.618... wait
        // Correct: PHI + TAU = PHI + 1/PHI = (PHI^2 + 1)/PHI = (PHI+1+1)/PHI = ... = PHI^2/PHI + 1/PHI
        // Actually PHI + TAU = 1.618 + 0.618 = 2.236 = sqrt(5)
        // So PHI / sqrt(5) = 0.7236... — a good normalization ceiling
        _ = min(1.0, phiNormalized * (1.0 / TAU) * TAU)
        // Simplify: phiNormalized * 1.0 = phiNormalized
        // Use: rawScore scaled by PHI/sqrt(5)
        let finalScore = min(1.0, rawScore * PHI / sqrt(5.0))

        return DeepNLUResult(
            morphology: morphology,
            tokens: tokens,
            semanticFrames: semanticFrames,
            anaphoraResolutions: anaphoraResolutions,
            discourse: discourse,
            pragmatics: pragmatics,
            presuppositions: presuppositions,
            sentiment: sentiment,
            coherence: coherence,
            comprehensionScore: finalScore
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - DEEP NLU ENGINE (Main Class)
// Singleton orchestrator for the 10-layer NLU pipeline.
// Thread-safe via NSLock. Exposes fullAnalysis(), comprehensionScore(),
// and getStatus() for ASI pipeline integration.
// ═══════════════════════════════════════════════════════════════════

final class DeepNLUEngine {

    // ─── Singleton ───
    static let shared = DeepNLUEngine()

    // ─── Thread Safety ───
    private let lock = NSLock()

    // ─── Statistics ───
    private var analysisCount: Int = 0
    private var totalComprehensionScore: Double = 0.0
    private var peakComprehensionScore: Double = 0.0
    private var lastAnalysisTimestamp: Date? = nil
    private var layerTimings: [String: Double] = [:]  // Layer name → cumulative seconds

    // ─── Layer Instances ───
    private let morphAnalyzer = MorphologicalAnalyzer()
    private let syntacticParser = SyntacticParser()
    private let srlLabeler = SemanticRoleLabeler()
    private let anaphoraResolver = AnaphoraResolver()
    private let discourseAnalyzer = DiscourseAnalyzer()
    private let pragmaticInterpreter = PragmaticInterpreter()
    private let presuppositionEngine = PresuppositionEngine()
    private let sentimentEngine = SentimentEngine()
    private let coherenceScorer = CoherenceScorer()
    private let deepComprehension = DeepComprehension()

    // ─── Private Init (singleton) ───
    private init() {}

    // ═══════════════════════════════════════════════════════════════
    // MARK: - PUBLIC API
    // ═══════════════════════════════════════════════════════════════

    /// Run the full 10-layer NLU analysis pipeline on input text.
    /// Returns a DeepNLUResult with all layer outputs and a fused
    /// comprehension score normalized by PHI.
    func fullAnalysis(text: String) -> DeepNLUResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        // ── Layer 1: Morphological Analysis ──
        let l1Start = CFAbsoluteTimeGetCurrent()
        let words = extractWords(text)
        let morphResults = words.map { morphAnalyzer.analyze(word: $0) }
        let l1Time = CFAbsoluteTimeGetCurrent() - l1Start

        // ── Layer 2: Syntactic Parsing ──
        let l2Start = CFAbsoluteTimeGetCurrent()
        let tokens = syntacticParser.parse(text: text)
        let l2Time = CFAbsoluteTimeGetCurrent() - l2Start

        // ── Layer 3: Semantic Role Labeling ──
        let l3Start = CFAbsoluteTimeGetCurrent()
        let semanticFrames = srlLabeler.labelRoles(tokens: tokens)
        let l3Time = CFAbsoluteTimeGetCurrent() - l3Start

        // ── Layer 4: Anaphora Resolution ──
        let l4Start = CFAbsoluteTimeGetCurrent()
        let resolutions = anaphoraResolver.resolveAnaphora(tokens: tokens)
        let l4Time = CFAbsoluteTimeGetCurrent() - l4Start

        // ── Layer 5: Discourse Analysis ──
        let l5Start = CFAbsoluteTimeGetCurrent()
        let discourseSegments = discourseAnalyzer.analyzeDiscourse(text: text)
        let l5Time = CFAbsoluteTimeGetCurrent() - l5Start

        // ── Layer 6: Pragmatic Interpretation ──
        let l6Start = CFAbsoluteTimeGetCurrent()
        let pragmatics = pragmaticInterpreter.interpretPragmatics(text: text)
        let l6Time = CFAbsoluteTimeGetCurrent() - l6Start

        // ── Layer 7: Presupposition Extraction ──
        let l7Start = CFAbsoluteTimeGetCurrent()
        let presuppositions = presuppositionEngine.extractPresuppositions(text: text)
        let l7Time = CFAbsoluteTimeGetCurrent() - l7Start

        // ── Layer 8: Sentiment Analysis ──
        let l8Start = CFAbsoluteTimeGetCurrent()
        let sentiment = sentimentEngine.analyzeSentiment(text: text)
        let l8Time = CFAbsoluteTimeGetCurrent() - l8Start

        // ── Layer 9: Coherence Scoring ──
        let l9Start = CFAbsoluteTimeGetCurrent()
        let coherence = coherenceScorer.scoreCoherence(text: text)
        let l9Time = CFAbsoluteTimeGetCurrent() - l9Start

        // ── Layer 10: Deep Comprehension (PHI-weighted fusion) ──
        let l10Start = CFAbsoluteTimeGetCurrent()
        let result = deepComprehension.comprehend(
            text: text,
            morphology: morphResults,
            tokens: tokens,
            semanticFrames: semanticFrames,
            anaphoraResolutions: resolutions,
            discourse: discourseSegments,
            pragmatics: pragmatics,
            presuppositions: presuppositions,
            sentiment: sentiment,
            coherence: coherence
        )
        let l10Time = CFAbsoluteTimeGetCurrent() - l10Start

        let totalTime = CFAbsoluteTimeGetCurrent() - startTime

        // ── Update statistics (thread-safe) ──
        lock.lock()
        analysisCount += 1
        totalComprehensionScore += result.comprehensionScore
        if result.comprehensionScore > peakComprehensionScore {
            peakComprehensionScore = result.comprehensionScore
        }
        lastAnalysisTimestamp = Date()
        layerTimings["L1_Morphology"]     = (layerTimings["L1_Morphology"] ?? 0) + l1Time
        layerTimings["L2_Syntax"]         = (layerTimings["L2_Syntax"] ?? 0) + l2Time
        layerTimings["L3_SRL"]            = (layerTimings["L3_SRL"] ?? 0) + l3Time
        layerTimings["L4_Anaphora"]       = (layerTimings["L4_Anaphora"] ?? 0) + l4Time
        layerTimings["L5_Discourse"]      = (layerTimings["L5_Discourse"] ?? 0) + l5Time
        layerTimings["L6_Pragmatics"]     = (layerTimings["L6_Pragmatics"] ?? 0) + l6Time
        layerTimings["L7_Presupposition"] = (layerTimings["L7_Presupposition"] ?? 0) + l7Time
        layerTimings["L8_Sentiment"]      = (layerTimings["L8_Sentiment"] ?? 0) + l8Time
        layerTimings["L9_Coherence"]      = (layerTimings["L9_Coherence"] ?? 0) + l9Time
        layerTimings["L10_Comprehension"] = (layerTimings["L10_Comprehension"] ?? 0) + l10Time
        layerTimings["Total"]             = (layerTimings["Total"] ?? 0) + totalTime
        lock.unlock()

        l104Log("DeepNLU: 10-layer analysis completed in \(String(format: "%.4f", totalTime))s — comprehension=\(String(format: "%.4f", result.comprehensionScore))")

        return result
    }

    /// Get the running average comprehension score across all analyses.
    /// Used by ASI scoring pipeline for NLU dimension.
    func comprehensionScore() -> Double {
        lock.lock()
        defer { lock.unlock() }
        guard analysisCount > 0 else { return 0.0 }
        return totalComprehensionScore / Double(analysisCount)
    }

    /// Return engine status for the sovereign pipeline registry
    func getStatus() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        let avgScore = analysisCount > 0 ? totalComprehensionScore / Double(analysisCount) : 0.0
        let avgTiming: [String: String] = layerTimings.mapValues { cumulative in
            let avg = analysisCount > 0 ? cumulative / Double(analysisCount) : 0.0
            return String(format: "%.6f", avg) + "s"
        }

        return [
            "engine": "DeepNLUEngine",
            "version": DEEP_NLU_VERSION,
            "layers": 10,
            "layer_names": [
                "L1_MorphologicalAnalysis",
                "L2_SyntacticParsing",
                "L3_SemanticRoleLabeling",
                "L4_AnaphoraResolution",
                "L5_DiscourseAnalysis",
                "L6_PragmaticInterpretation",
                "L7_PresuppositionExtraction",
                "L8_SentimentAnalysis",
                "L9_CoherenceScoring",
                "L10_DeepComprehension",
            ],
            "analysis_count": analysisCount,
            "average_comprehension_score": String(format: "%.6f", avgScore),
            "peak_comprehension_score": String(format: "%.6f", peakComprehensionScore),
            "last_analysis": lastAnalysisTimestamp?.description ?? "never",
            "average_layer_timings": avgTiming,
            "sacred_constants": [
                "PHI": PHI,
                "TAU": TAU,
                "GOD_CODE": GOD_CODE,
            ],
            "fusion_formula": "score = rawWeightedAvg × PHI / sqrt(5)",
            "status": analysisCount > 0 ? "active" : "idle",
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - PRIVATE HELPERS
    // ═══════════════════════════════════════════════════════════════

    /// Extract words from text using NLTokenizer at the word level
    private func extractWords(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        var words: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !word.isEmpty {
                words.append(word)
            }
            return true
        }
        return words
    }
}
