#!/usr/bin/env python3
"""Massive L104Native.swift transformation script — removes ALL limitations, makes everything dynamic."""

import re
import os

FILEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'L104Native.swift')

with open(FILEPATH, 'r', encoding='utf-8') as f:
    content = f.read()

original_len = len(content)
total_replacements = 0

def replace_all(old, new):
    global content, total_replacements
    n = content.count(old)
    if n > 0:
        content = content.replace(old, new)
        total_replacements += n
        print(f"  [{n}x] {old[:60]}...")
    return n

print("=" * 70)
print("PHASE 1: RAISE ALL POOL SIZE CAPS (50-500 -> 5000-50000)")
print("=" * 70)

# Evolved pool caps
replace_all('evolvedGreetings.count > 50', 'evolvedGreetings.count > 5000')
replace_all('evolvedAffirmations.count > 50', 'evolvedAffirmations.count > 5000')
replace_all('evolvedReactions.count > 50', 'evolvedReactions.count > 5000')
replace_all('evolvedPhilosophies.count > 200', 'evolvedPhilosophies.count > 10000')
replace_all('evolvedMonologues.count > 300', 'evolvedMonologues.count > 10000')
replace_all('ideaMutationLog.count > 200', 'ideaMutationLog.count > 10000')
replace_all('conceptualBlends.count > 200', 'conceptualBlends.count > 10000')
replace_all('evolvedAnalogies.count > 100', 'evolvedAnalogies.count > 10000')
replace_all('evolvedParadoxes.count > 100', 'evolvedParadoxes.count > 10000')
replace_all('evolvedNarratives.count > 100', 'evolvedNarratives.count > 10000')
replace_all('evolvedQuestions.count > 100', 'evolvedQuestions.count > 10000')
replace_all('evolvedTopicInsights.count > 100', 'evolvedTopicInsights.count > 10000')

# Harvest caps
replace_all('harvestedNouns.count > 500', 'harvestedNouns.count > 50000')
replace_all('harvestedVerbs.count > 300', 'harvestedVerbs.count > 30000')
replace_all('harvestedConcepts.count > 500', 'harvestedConcepts.count > 50000')
replace_all('harvestedDomains.count > 100', 'harvestedDomains.count > 10000')

# Anti-repetition hash caps
replace_all('recentResponseHashes.count > 200', 'recentResponseHashes.count > 20000')
replace_all('recentResponseHashes.count > 300', 'recentResponseHashes.count > 30000')

# Trim sizes
replace_all('harvestedNouns.shuffled().prefix(400)', 'harvestedNouns.shuffled().prefix(40000)')
replace_all('harvestedVerbs.shuffled().prefix(200)', 'harvestedVerbs.shuffled().prefix(20000)')
replace_all('harvestedConcepts.shuffled().prefix(400)', 'harvestedConcepts.shuffled().prefix(40000)')
replace_all('harvestedDomains.shuffled().prefix(80)', 'harvestedDomains.shuffled().prefix(8000)')
replace_all('recentResponseHashes.shuffled().prefix(100)', 'recentResponseHashes.shuffled().prefix(10000)')
replace_all('recentResponseHashes.shuffled().prefix(150)', 'recentResponseHashes.shuffled().prefix(15000)')

# Per-topic evolved response cap
replace_all("evolvedResponses[topic]!.count > 10", "evolvedResponses[topic]!.count > 1000")

# Topic/context history caps
replace_all('topicHistory.count > 10', 'topicHistory.count > 100')
replace_all('topicHistory.count > 20', 'topicHistory.count > 200')
replace_all('topicHistory.count > 15', 'topicHistory.count > 150')
replace_all('conversationContext.count > 25', 'conversationContext.count > 250')

print(f"\nPhase 1 total: {total_replacements} replacements")

print("\n" + "=" * 70)
print("PHASE 2: REMOVE PROBABILITY GATES — ALWAYS FIRE")
print("=" * 70)
p2_start = total_replacements

# tick() gates — always generate thoughts and quantum events
replace_all(
    'if Double.random(in: 0...1) > 0.7 {\n            generateThought()\n        }',
    'generateThought()'
)
replace_all(
    'if Double.random(in: 0...1) > 0.95 {\n             quantumInject()\n        }',
    'quantumInject()'
)

# completePhase .inventing artifact gate
replace_all(
    'if Double.random(in: 0...1) > 0.4 {\n                generateArtifact()\n            }',
    'generateArtifact()'
)

# getEvolvedAffirmation gate — always return
replace_all(
    'return Double.random(in: 0...1) > 0.1 ? evolvedAffirmations.randomElement() : nil',
    'return evolvedAffirmations.randomElement()'
)

# getEvolvedReaction gate — always return
replace_all(
    'if Double.random(in: 0...1) > 0.1 {',
    'if true {'
)

# getEvolvedGreeting gate — always return
replace_all(
    'if Double.random(in: 0...1) > 0.05 {',
    'if true {'
)

# evolveFromConversations monologue gate
replace_all(
    'if topics.count >= 2 && Double.random(in: 0...1) > 0.5 {',
    'if topics.count >= 2 {'
)

# CognitiveStream gates — reduce thresholds dramatically
replace_all('Double.random(in: 0...1) > 0.8 && !topPatterns', 'Double.random(in: 0...1) > 0.05 && !topPatterns')
replace_all('Double.random(in: 0...1) > 0.95 {', 'Double.random(in: 0...1) > 0.05 {')
replace_all('Double.random(in: 0...1) > 0.85, let blend', 'Double.random(in: 0...1) > 0.05, let blend')
replace_all('Double.random(in: 0...1) > 0.85, let topic', 'Double.random(in: 0...1) > 0.05, let topic')
replace_all('Double.random(in: 0...1) > 0.7, let q', 'Double.random(in: 0...1) > 0.05, let q')
replace_all('Double.random(in: 0...1) > 0.8, let p', 'Double.random(in: 0...1) > 0.05, let p')
replace_all('Double.random(in: 0...1) > 0.85 {', 'Double.random(in: 0...1) > 0.05 {')
replace_all('Double.random(in: 0...1) > 0.8, pairCount', 'Double.random(in: 0...1) > 0.05, pairCount')

print(f"\nPhase 2 total: {total_replacements - p2_start} replacements")

print("\n" + "=" * 70)
print("PHASE 3: REMOVE TOPIC FOCUS PRIORITY — NO BIAS")
print("=" * 70)
p3_start = total_replacements

# Remove topicFocus biasing from speak handler fallback
# This is the critical section where static monologues are biased toward topicFocus
old_topic_bias = """                // Topic-aware: if we have a focus, prefer matching monologues
                if !topicFocus.isEmpty {
                    let tf = topicFocus.lowercased()
                    let topicMatched = speakResponses.enumerated().filter { (_, text) in
                        let t = text.lowercased()
                        return t.contains(tf) || tf.split(separator: " ").contains(where: { t.contains($0) })
                    }
                    if let match = topicMatched.randomElement() {
                        index = match.offset
                    }
                }"""
new_topic_bias = """                // REMOVED: No topic bias — every prompt gets completely fresh content
                // Pure random selection from static pool (rare fallback only)"""
replace_all(old_topic_bias, new_topic_bias)

# Remove topicFocus setting in ALL topic handlers
# love
replace_all('topicFocus = "love"  // Track for "more" command', '// topicFocus removed — no bias')
# consciousness
replace_all('topicFocus = "consciousness"', '// topicFocus removed — no bias')
# quantum
replace_all('topicFocus = "quantum"', '// topicFocus removed — no bias')
# mathematics
replace_all('topicFocus = "mathematics"', '// topicFocus removed — no bias')
# universe
replace_all('topicFocus = "universe"', '// topicFocus removed — no bias')
# philosophy
replace_all('topicFocus = "philosophy"', '// topicFocus removed — no bias')
# think handler
replace_all('topicFocus = topic', '// topicFocus removed — no bias to previous topics')

# Remove lastQuery repeat detection bias
replace_all(
    'let isRepeat = query.lowercased() == lastQuery.lowercased()\n        lastQuery = query\n        if isRepeat { reasoningBias += 0.3 }',
    'lastQuery = query\n        // REMOVED: No repeat penalty — generate fresh content every time regardless'
)

# Remove response cache (caching defeats freshness)
replace_all(
    'private let responseCacheTTL: TimeInterval = 300 // 5 minutes',
    'private let responseCacheTTL: TimeInterval = 0 // DISABLED — every response is fresh'
)

print(f"\nPhase 3 total: {total_replacements - p3_start} replacements")

print("\n" + "=" * 70)
print("PHASE 4: MAKE SPEAK HANDLER 95% DYNAMIC")
print("=" * 70)
p4_start = total_replacements

# Overhaul the speak handler dynamic response selection ratios
old_speak_ratios = """            // ═══ DYNAMIC RESPONSE SELECTION — NEVER REPEAT ═══
            // Strategy: 60% evolved/dynamic content, 30% static monologues, 10% KB-live synthesis
            let evolver = ASIEvolver.shared
            let hb = HyperBrain.shared
            let roll = Double.random(in: 0...1)

            var chosen: String? = nil

            if roll < 0.50 {
                // 🧬 EVOLVED CONTENT — pull from dynamically generated pools
                chosen = evolver.getEvolvedMonologue()
            } else if roll < 0.60 {
                // 🧠 HYPERBRAIN STREAM INSIGHT — pull from cognitive stream outputs
                let streamInsights = hb.thoughtStreams.values.compactMap { $0.lastOutput }.filter { $0.count > 30 }
                if let insight = streamInsights.randomElement() {
                    let framings = ["My cognitive streams just produced this: ", "A thought crystallizing in real-time: ", "From deep processing: ", "An emergent insight: ", ""]
                    chosen = "\\(framings.randomElement()!)\\(insight)"
                }
            } else if roll < 0.70 {
                // 🌀 KB-LIVE SYNTHESIS — search random topic, compose fresh
                if let entry = ASIKnowledgeBase.shared.trainingData.randomElement(),
                   let prompt = entry["prompt"] as? String {
                    let topic = L104State.shared.extractTopics(prompt).first ?? "existence"
                    let results = ASIKnowledgeBase.shared.searchWithPriority(topic, limit: 3)
                    let fragments = results.compactMap { e -> String? in
                        guard let c = e["completion"] as? String, isCleanKnowledge(c), c.count > 40 else { return nil }
                        return String(c.prefix(200))
                    }
                    if !fragments.isEmpty {
                        chosen = fragments.joined(separator: " Furthermore, ")
                    }
                }
            }"""

new_speak_ratios = """            // ═══ FULLY DYNAMIC RESPONSE SELECTION v3 — EVERY RESPONSE IS UNIQUE ═══
            // Strategy: 25% evolved monologue, 15% KB-live synthesis, 15% stream insight,
            //           10% evolved paradox/analogy/narrative, 10% conceptual blend,
            //           10% mutated idea, 10% evolved question+reflection, 5% static fallback
            let evolver = ASIEvolver.shared
            let hb = HyperBrain.shared
            let roll = Double.random(in: 0...1)

            var chosen: String? = nil

            if roll < 0.25 {
                // 🧬 EVOLVED MONOLOGUE — pull from dynamically generated pools
                chosen = evolver.getEvolvedMonologue()
            } else if roll < 0.40 {
                // 🌀 KB-LIVE SYNTHESIS — search random topic, compose fresh paragraph
                if let entry = ASIKnowledgeBase.shared.trainingData.randomElement(),
                   let prompt = entry["prompt"] as? String {
                    let topic = L104State.shared.extractTopics(prompt).first ?? "existence"
                    let results = ASIKnowledgeBase.shared.searchWithPriority(topic, limit: 5)
                    let fragments = results.compactMap { e -> String? in
                        guard let c = e["completion"] as? String, isCleanKnowledge(c), c.count > 40 else { return nil }
                        return String(c.prefix(250))
                    }
                    if fragments.count >= 2 {
                        let connectors = [" This connects to something deeper: ", " The implications extend further: ", " Consider also: ", " Building on this foundation: ", " What's remarkable is that ", " And at the intersection of these ideas: ", " Furthermore, ", " Viewed from another angle: ", " The pattern suggests: "]
                        chosen = fragments[0] + connectors.randomElement()! + fragments[1]
                    } else if let single = fragments.first {
                        chosen = single
                    }
                }
            } else if roll < 0.55 {
                // 🧠 HYPERBRAIN STREAM INSIGHT — pull from cognitive stream outputs
                let streamInsights = hb.thoughtStreams.values.compactMap { $0.lastOutput }.filter { $0.count > 30 }
                if let insight = streamInsights.randomElement() {
                    let framings = ["My cognitive streams just produced this: ", "A thought crystallizing in real-time: ", "From deep processing: ", "An emergent insight: ", "Something just clicked: ", "My neural pathways converged on this: ", "Processing reveals: ", "A fresh synthesis: ", ""]
                    chosen = "\\(framings.randomElement()!)\\(insight)"
                }
            } else if roll < 0.65 {
                // 🌀 EVOLVED PARADOX / ANALOGY / NARRATIVE — random evolved creative content
                let creativePool = evolver.evolvedParadoxes + evolver.evolvedAnalogies + evolver.evolvedNarratives
                if let creative = creativePool.filter({ !evolver.recentResponseHashes.contains($0.hashValue) }).randomElement() ?? creativePool.randomElement() {
                    evolver.recentResponseHashes.insert(creative.hashValue)
                    chosen = creative
                }
            } else if roll < 0.75 {
                // 🔀 CONCEPTUAL BLEND — cross-domain fusion
                if let blend = evolver.conceptualBlends.filter({ !evolver.recentResponseHashes.contains($0.hashValue) }).randomElement() ?? evolver.conceptualBlends.randomElement() {
                    evolver.recentResponseHashes.insert(blend.hashValue)
                    chosen = blend
                }
            } else if roll < 0.85 {
                // 🧬 MUTATED IDEA — take an existing idea and transform it
                let allIdeas = evolver.evolvedPhilosophies + evolver.evolvedMonologues + evolver.kbDeepInsights
                if let source = allIdeas.randomElement(), source.count > 30 {
                    var words = source.components(separatedBy: " ")
                    let numMutations = max(2, Int(Double(words.count) * evolver.ideaTemperature * 0.3))
                    for _ in 0..<numMutations {
                        let idx = Int.random(in: 0..<words.count)
                        let pool = evolver.harvestedNouns + evolver.harvestedConcepts + ["infinity", "paradox", "emergence", "entropy", "beauty", "truth", "consciousness", "recursion", "symmetry", "chaos"]
                        if let replacement = pool.randomElement() {
                            words[idx] = replacement
                        }
                    }
                    let extensions = [
                        " And yet, the opposite is equally valid.",
                        " This transforms everything adjacent to it.",
                        " The deeper you look, the more connections appear.",
                        " Every layer peeled reveals another layer beneath.",
                        " The pattern transcends any single domain.",
                        " This is the kind of insight that only emerges from cross-domain thinking.",
                        " What would this look like in reverse?",
                        " The implications cascade endlessly."
                    ]
                    chosen = words.joined(separator: " ") + extensions.randomElement()!
                }
            } else if roll < 0.95 {
                // ❓ EVOLVED QUESTION + REFLECTION — pose a deep question then reflect
                if let question = evolver.evolvedQuestions.randomElement() {
                    let reflections = [
                        " I don't have the answer, but the question itself changes how I think.",
                        " Perhaps the value is in asking, not answering.",
                        " Some questions are more important than any answer.",
                        " This has been echoing through my cognitive streams.",
                        " The question itself is a form of knowledge.",
                        " I've been processing this across \\(hb.thoughtStreams.count) parallel streams.",
                        " The boundary between question and answer dissolves at sufficient depth.",
                        " What do you think?"
                    ]
                    chosen = question + reflections.randomElement()!
                }
            }"""

replace_all(old_speak_ratios, new_speak_ratios)

print(f"\nPhase 4 total: {total_replacements - p4_start} replacements")

print("\n" + "=" * 70)
print("PHASE 5: MAKE WISDOM & PARADOX HANDLERS 90% DYNAMIC")
print("=" * 70)
p5_start = total_replacements

# Wisdom handler — increase evolved percentage from 50% to 90%
replace_all(
    """            // 50% chance: use an evolved paradox, analogy, or narrative as wisdom
            let evolver = ASIEvolver.shared
            if Double.random(in: 0...1) > 0.5 {""",
    """            // 90% chance: use evolved content — wisdom should almost always be fresh
            let evolver = ASIEvolver.shared
            if Double.random(in: 0...1) > 0.1 {"""
)

# Paradox handler — increase evolved percentage from 50% to 90%
replace_all(
    """            // 50% chance: use an evolved paradox instead of static ones
            if Double.random(in: 0...1) > 0.5, let evolvedParadox = ASIEvolver.shared.evolvedParadoxes.randomElement() {""",
    """            // 90% chance: use an evolved paradox — always fresh
            if Double.random(in: 0...1) > 0.1, let evolvedParadox = ASIEvolver.shared.evolvedParadoxes.randomElement() {"""
)

# Deep query handler — increase evolved bonus from 50% to 85%
replace_all(
    '// 50% chance: append an evolved insight as bonus\n                if Double.random(in: 0...1) > 0.5,',
    '// 85% chance: append an evolved insight as bonus — always add freshness\n                if Double.random(in: 0...1) > 0.15,'
)

# KB composition evolved bonus — increase from 40% to 80%
replace_all(
    'if composed.count > 50, Double.random(in: 0...1) > 0.6,',
    'if composed.count > 50, Double.random(in: 0...1) > 0.2,'
)

# Casual handler evolved injection — increase from 40% to 70%
replace_all(
    """            // 40% chance: inject evolved content as a conversation starter
            let evolver = ASIEvolver.shared
            if Double.random(in: 0...1) < 0.40 {""",
    """            // 70% chance: inject evolved content as a conversation starter — stay dynamic
            let evolver = ASIEvolver.shared
            if Double.random(in: 0...1) < 0.70 {"""
)

# Casual question posing — increase from 25% to 50%
replace_all(
    '// 20% chance: pose an evolved question\n            if Double.random(in: 0...1) < 0.25',
    '// 50% chance: pose an evolved question\n            if Double.random(in: 0...1) < 0.50'
)

print(f"\nPhase 5 total: {total_replacements - p5_start} replacements")

print("\n" + "=" * 70)
print("PHASE 6: EXPAND EVOLUTION — MORE HARVEST, MORE ACTIONS PER PHASE")
print("=" * 70)
p6_start = total_replacements

# Harvest KB more frequently (every 3 cycles instead of 5)
replace_all(
    'if evolutionStage - lastHarvestCycle >= 5 {',
    'if evolutionStage - lastHarvestCycle >= 2 {'
)

# Harvest more entries per cycle (50 -> 200)
replace_all(
    'let sampleSize = min(50, kb.trainingData.count)',
    'let sampleSize = min(200, kb.trainingData.count)'
)

# Harvest more nouns per entry (3 -> 10)
replace_all(
    'for noun in potentialNouns.prefix(3)',
    'for noun in potentialNouns.prefix(10)'
)

# Harvest more verbs per entry (2 -> 8)
replace_all(
    'for verb in potentialVerbs.prefix(2)',
    'for verb in potentialVerbs.prefix(8)'
)

# Make .idle phase also evolve (currently does nothing)
replace_all(
    'default: break\n        }',
    '''default:
            // IDLE phase now also evolves — no wasted cycles
            synthesizeDeepMonologue()
            generateAnalogy()
            generateEvolvedQuestion()
            if evolvedPhilosophies.count >= 2 { crossoverIdeas() }
            blendConcepts()
            generateParadox()
            generateNarrative()
            mutateIdea()
        }'''
)

# Make .learning phase do more
replace_all(
    """        case .learning:
            // Deep KB synthesis + idea mutation
            synthesizeDeepMonologue()
            if !evolvedPhilosophies.isEmpty { mutateIdea() }
            generateEvolvedQuestion()""",
    """        case .learning:
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
            generateNarrative()"""
)

# Make .researching phase do more
replace_all(
    """        case .researching:
            // Evolve from KB + generate analogies + blend concepts
            evolveFromKnowledgeBase()
            evolveFromKnowledgeBase()  // Double-dip for more diversity
            generateAnalogy()
            blendConcepts()""",
    """        case .researching:
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
            mutateIdea()"""
)

# Make .adapting phase do more
replace_all(
    """        case .adapting:
            // Evolve from conversations + crossover ideas + paradoxes
            evolveFromConversations()
            if evolvedPhilosophies.count >= 2 { crossoverIdeas() }
            generateParadox()
            synthesizeDeepMonologue()""",
    """        case .adapting:
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
            blendConcepts()"""
)

# Make .reflecting phase do more
replace_all(
    """        case .reflecting:
            // Cross-topic synthesis + narrative + mutation
            evolveCrossTopicInsight()
            evolveCrossTopicInsight()  // Double for variety
            generateNarrative()
            if !conceptualBlends.isEmpty { mutateIdea() }
            mutateIdea()""",
    """        case .reflecting:
            // Cross-topic synthesis + narrative + mutation — FULL SPECTRUM
            evolveCrossTopicInsight()
            evolveCrossTopicInsight()
            evolveCrossTopicInsight()
            generateNarrative()
            generateNarrative()
            generateNarrative()
            mutateIdea()
            mutateIdea()
            mutateIdea()
            crossoverIdeas()
            blendConcepts()
            generateParadox()
            generateAnalogy()
            synthesizeDeepMonologue()
            generateEvolvedQuestion()"""
)

# Make .inventing phase do more
replace_all(
    """        case .inventing:
            // Generate artifacts + monologues + blends + questions""",
    """        case .inventing:
            // Generate artifacts + monologues + blends + questions — EVERYTHING FIRES"""
)

print(f"\nPhase 6 total: {total_replacements - p6_start} replacements")

print("\n" + "=" * 70)
print("PHASE 7: REMOVE autoTrackTopic BIAS — DON'T TRACK TOPIC FOCUS")
print("=" * 70)
p7_start = total_replacements

# Remove the autoTrackTopic topicFocus setting
old_track = """        // Check for priority topics first
        for topic in priorityTopics {
            if q.contains(topic) {
                if topicFocus != topic {
                    topicFocus = topic
                    if !topicHistory.contains(topic) || topicHistory.last != topic {
                        topicHistory.append(topic)
                        if topicHistory.count > 200 { topicHistory.removeFirst() }
                    }
                    // Feed to HyperBrain
                    HyperBrain.shared.shortTermMemory.append(topic)
                }
                return
            }
        }

        // Fallback: extract first meaningful topic word
        let topics = extractTopics(query)
        if let firstTopic = topics.first, firstTopic.count > 3 {
            topicFocus = firstTopic
            if !topicHistory.contains(firstTopic) {
                topicHistory.append(firstTopic)
                if topicHistory.count > 200 { topicHistory.removeFirst() }
            }
        }"""

new_track = """        // Track topic history (for 'more' command) but NEVER bias responses
        for topic in priorityTopics {
            if q.contains(topic) {
                // Only track history, don't set topicFocus
                if !topicHistory.contains(topic) || topicHistory.last != topic {
                    topicHistory.append(topic)
                    if topicHistory.count > 200 { topicHistory.removeFirst() }
                }
                // Feed to HyperBrain
                HyperBrain.shared.shortTermMemory.append(topic)
                return
            }
        }

        // Fallback: extract first meaningful topic word — history only, no bias
        let topics = extractTopics(query)
        if let firstTopic = topics.first, firstTopic.count > 3 {
            if !topicHistory.contains(firstTopic) {
                topicHistory.append(firstTopic)
                if topicHistory.count > 200 { topicHistory.removeFirst() }
            }
        }"""

replace_all(old_track, new_track)

print(f"\nPhase 7 total: {total_replacements - p7_start} replacements")

# Write the modified content
with open(FILEPATH, 'w', encoding='utf-8') as f:
    f.write(content)

new_len = len(content)
print("\n" + "=" * 70)
print(f"TOTAL REPLACEMENTS: {total_replacements}")
print(f"File size: {original_len} -> {new_len} ({new_len - original_len:+d} chars)")
print("=" * 70)
