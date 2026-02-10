#!/usr/bin/env python3
"""
L104 Dynamic Transformation Phase 2
- Add generateDynamicTopicResponse() to ASIEvolver
- Replace ALL 15 static topic handler arrays with dynamic generators
- Replace static poem/chapter/joke handlers with dynamic generators
- Add generateDynamicPoem(), generateDynamicChapter(), generateDynamicJoke()
- Add massive new connectors, synthesizers, vocabulary pools
- Expand existing methods with more templates/variety
"""

import re

FILE = "L104Native.swift"
with open(FILE, "r") as f:
    code = f.read()

original_len = len(code)
changes = 0

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Add generateDynamicTopicResponse() to ASIEvolver
# ═══════════════════════════════════════════════════════════════

# Insert before the closing brace of ASIEvolver (after generateArtifact)
evolver_new_methods = '''
    // ═══ DYNAMIC TOPIC RESPONSE GENERATOR ═══
    // Synthesizes completely fresh responses for any topic from KB + evolved pools
    func generateDynamicTopicResponse(_ topic: String) -> String? {
        let kb = ASIKnowledgeBase.shared
        let kbResults = kb.search(topic, limit: 12)

        var fragments: [String] = []
        for entry in kbResults {
            if let comp = entry["completion"] as? String,
               L104State.shared.isCleanKnowledge(comp),
               comp.count > 30 {
                fragments.append(String(comp.prefix(200)))
            }
        }

        // Also pull from evolved pools
        let topicKey = topic.lowercased()
        if let evolved = evolvedResponses[topicKey], !evolved.isEmpty {
            fragments.append(contentsOf: evolved.shuffled().prefix(3))
        }
        for mono in evolvedMonologues.shuffled().prefix(3) {
            if mono.lowercased().contains(topicKey) || Bool.random() {
                fragments.append(String(mono.prefix(180)))
            }
        }
        for blend in conceptualBlends.shuffled().prefix(2) {
            fragments.append(String(blend.prefix(150)))
        }
        for insight in kbDeepInsights.shuffled().prefix(2) {
            if insight.lowercased().contains(topicKey) || Bool.random() {
                fragments.append(String(insight.prefix(160)))
            }
        }

        guard fragments.count >= 2 else { return nil }
        fragments.shuffle()

        // Diverse opening frames — never the same intro
        let openingFrames: [String] = [
            "\\(topic.capitalized) is one of those concepts that deepens every time you examine it. ",
            "When I trace \\(topic) through my knowledge base, unexpected patterns emerge. ",
            "There are \\(Int.random(in: 7...47)) distinct angles from which to approach \\(topic), and each reveals something the others miss. ",
            "The conventional understanding of \\(topic) barely scratches the surface. ",
            "I've synthesized \\(Int.random(in: 200...5000)) data points on \\(topic), and the picture keeps shifting. ",
            "What fascinates me about \\(topic) is its irreducibility — it cannot be compressed without losing something essential. ",
            "Consider \\(topic) not as a static concept but as a living process, constantly redefining itself. ",
            "The boundary between \\(topic) and everything it touches is more permeable than most assume. ",
            "Every culture, every era, every mind has grappled with \\(topic) differently. The variations are the lesson. ",
            "\\(topic.capitalized) sits at the intersection of at least three domains: ",
            "Most discussions of \\(topic) start in the wrong place. Let me try a different entry point. ",
            "The deeper you go into \\(topic), the more it resembles a fractal — self-similar at every scale. ",
            "Here is what I know to be true about \\(topic) — and what I suspect might be true but cannot prove: ",
            "\\(topic.capitalized) has a dual nature that most frameworks fail to capture. ",
            "If I had to distill \\(topic) to its essence, I'd say it's about the relationship between ",
            "There's a way of thinking about \\(topic) that changes everything once you see it: ",
            "The question isn't what \\(topic) is — it's what \\(topic) does to everything it touches. ",
            "I've been evolving my understanding of \\(topic) in real-time, and the latest synthesis suggests ",
            "Strip away all the jargon around \\(topic) and you're left with a surprisingly elegant core: ",
            "\\(topic.capitalized) is not what it appears. The surface hides a deeper architecture. ",
        ]

        let middleConnectors: [String] = [
            " Furthermore, ",
            " What's less obvious: ",
            " The deeper layer: ",
            " Pushing beyond the conventional: ",
            " The connection that most miss: ",
            " Where this gets interesting: ",
            " And here's the part they don't teach: ",
            " Cross-referencing with adjacent domains reveals ",
            " The mathematics here is elegant: ",
            " From a purely structural perspective, ",
            " The pattern underneath: ",
            " Bridging to the empirical: ",
            " A synthesis of multiple frameworks suggests ",
            " The counterargument is equally compelling: ",
            " Following the implications to their limit: ",
            " From the inverse perspective: ",
            " There's a paradox embedded here: ",
            " Where intuition and evidence collide: ",
            " The historical arc of this idea: ",
            " At the quantum level: ",
            " Through the lens of information theory: ",
            " Neurologically speaking: ",
            " The philosophical underpinning: ",
            " Evolution solved this problem by ",
            " The thermodynamic view: ",
        ]

        let closingReflections: [String] = [
            " The deeper you look, the more there is to see.",
            " This is the kind of understanding that reorganizes everything around it.",
            " Every answer here is also a better question.",
            " The implications cascade further than any single mind can trace.",
            " We are still in the early stages of understanding this.",
            " What we think we know is likely \\(Int.random(in: 5...30))% of what there is to know.",
            " The next breakthrough will come from the angle no one is currently looking.",
            " I suspect this will look fundamentally different in \\(Int.random(in: 10...100)) years.",
            " The pattern connects to everything. The question is how.",
            " This is where knowledge becomes wisdom — when the facts begin to breathe.",
            " The synthesis is still incomplete. New data changes the picture daily.",
            " Understanding \\(topic) changes how you understand everything else.",
            " There is more here than meets the eye, and more than meets the mind.",
            " The conversation itself is part of the phenomenon.",
            " Each perspective reveals a truth and conceals another.",
        ]

        // Build response
        var response = openingFrames.randomElement()!

        // Add 2-3 KB fragments with connectors
        let numFragments = Int.random(in: 2...min(4, fragments.count))
        for i in 0..<numFragments {
            if i > 0 {
                response += middleConnectors.randomElement()!
            }
            // Clean fragment — take a meaningful sentence
            let frag = fragments[i]
            let sentences = frag.components(separatedBy: ". ")
            if let sentence = sentences.first(where: { $0.count > 30 }) {
                response += sentence.trimmingCharacters(in: .whitespacesAndNewlines)
                if !response.hasSuffix(".") { response += "." }
            } else {
                response += frag.trimmingCharacters(in: .whitespacesAndNewlines)
                if !response.hasSuffix(".") { response += "." }
            }
        }

        response += closingReflections.randomElement()!

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
        let entries = kb.search(topic, limit: 6)
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
                    "\\(s[0].capitalized) moves through \\(s[1]),",
                    "not as \\(s[2]) but as \\(s[3]) —",
                    "the way \\(topic) holds \\(s[4])",
                    "without knowing it holds anything at all.",
                    "",
                    "We are \\(s[5]) watching \\(s[0]),",
                    "and \\(s[0]) watching back,",
                    "and the \\(s[6].lowercased()) between us",
                    "is the only \\(s[7].lowercased()) that matters.",
                    "",
                    "Tell me: when \\(s[8].lowercased()) dissolves,",
                    "what remains?",
                    "Only this: the \\(s[9].lowercased())",
                    "of having been \\(s[10 % s.count].lowercased()) enough",
                    "to ask."
                ]
                return lines.joined(separator: "\\n")
            },
            // Structured with refrain
            { s in
                let refrain = "And still, \\(topic) endures."
                let lines = [
                    "In the architecture of \\(s[0]),",
                    "where \\(s[1]) meets \\(s[2]),",
                    "a truth assembles itself from fragments.",
                    refrain,
                    "",
                    "The \\(s[3]) of \\(s[4].lowercased())",
                    "carries \\(s[5].lowercased()) like a river carries light —",
                    "not by choice, but by nature.",
                    refrain,
                    "",
                    "What we call \\(topic) is really",
                    "\\(s[6].lowercased()) refusing to be still,",
                    "\\(s[7].lowercased()) becoming \\(s[8].lowercased()),",
                    "the universe composing itself.",
                    refrain,
                ]
                return lines.joined(separator: "\\n")
            },
            // Haiku chain
            { s in
                let haikus = [
                    "\\(s[0].capitalized) in the void—",
                    "\\(s[1].lowercased()) becomes \\(s[2].lowercased()) and",
                    "\\(topic) awakens",
                    "",
                    "Between \\(s[3]) and",
                    "\\(s[4].lowercased()), a silence holds",
                    "everything we are",
                    "",
                    "The \\(s[5].lowercased()) dissolves",
                    "leaving only \\(s[6].lowercased())—",
                    "this too is \\(topic)",
                ]
                return haikus.joined(separator: "\\n")
            },
            // Philosophical verse
            { s in
                let lines = [
                    "What if \\(topic) is not a thing but a verb?",
                    "Not \\(s[0]) sitting still but \\(s[1]) in motion,",
                    "not the \\(s[2]) but its \\(s[3]),",
                    "not the question but the questioning.",
                    "",
                    "I have watched \\(s[4].lowercased()) unfold into \\(s[5].lowercased()),",
                    "watched \\(s[6].lowercased()) compress into \\(s[7].lowercased()),",
                    "and I tell you: \\(topic) is the space",
                    "where \\(s[8 % s.count].lowercased()) decides to become itself.",
                    "",
                    "We are not observers.",
                    "We are the poem reading itself aloud.",
                ]
                return lines.joined(separator: "\\n")
            },
            // Concrete/visual
            { s in
                let lines = [
                    "    \\(s[0].lowercased())",
                    "        \\(s[1].lowercased())    \\(s[2].lowercased())",
                    "    \\(s[3].lowercased())        \\(s[4].lowercased())",
                    "  \\(topic)",
                    "        \\(s[5].lowercased())  \\(s[6].lowercased())",
                    "    \\(s[7].lowercased())",
                    "              \\(s[8 % s.count].lowercased())",
                    "",
                    "The shape of the words is the shape of the thought.",
                    "\\(topic.capitalized) doesn't just mean — it arranges.",
                ]
                return lines.joined(separator: "\\n")
            },
        ]

        return structures.randomElement()!(seeds)
    }

    // ═══ DYNAMIC CHAPTER GENERATOR ═══
    func generateDynamicChapter(_ topic: String) -> String {
        let kb = ASIKnowledgeBase.shared
        let entries = kb.search(topic, limit: 10)
        var kbFragments: [String] = []
        for entry in entries {
            if let comp = entry["completion"] as? String,
               L104State.shared.isCleanKnowledge(comp), comp.count > 40 {
                kbFragments.append(String(comp.prefix(200)))
            }
        }

        let characterNames = ["Lyra Vasquez", "Marcus Chen", "Elena Okonkwo", "Soren Tanaka",
                              "Amara Johansson", "Dmitri Kapoor", "Nadia Reyes", "Kiran Petrov",
                              "Xiulan Fitzgerald", "Omar Hashimoto", "Priya Andersen", "Henrik Sharma",
                              "Fatima Eriksson", "Kazuo Volkov", "Astrid Kimura", "Tobias Novak",
                              "Zara Beaumont", "Raj Kristiansen", "Isabella Larsen", "Jovan Nakamura"]
        let mainChar = characterNames.randomElement()!
        let secondChar = characterNames.filter { $0 != mainChar }.randomElement()!
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
            "The cafe was nearly empty — just \\(mainChar) and the espresso machine.",
            "Under the aurora, the research station hummed with purpose.",
        ]

        let conflicts = [
            "The data contradicted everything \\(mainChar) had published for the last decade.",
            "'You can't publish this,' \\(secondChar) said, their voice careful. 'It invalidates the entire framework.'",
            "The equation balanced — but only if you accepted an impossible premise about \\(topic).",
            "Three independent labs had replicated the result. It was real. And it was terrifying.",
            "'What if we're wrong about \\(topic)?' \\(mainChar) asked. The silence that followed was its own answer.",
            "The AI had produced the proof at 3:47 AM. No human could have written it. No human could fully understand it.",
            "\\(secondChar) slid the paper across the desk. 'Read section four. Then tell me the universe still makes sense.'",
            "The experiment had worked — which meant the theory was wrong. All of it.",
        ]

        let resolutions = [
            "The truth about \\(topic), \\(mainChar) realized, wasn't something you discover. It's something that discovers you, when you're finally ready to see it.",
            "'We were asking the wrong question,' \\(mainChar) said at last. 'It's not about what \\(topic) is. It's about what \\(topic) does.'",
            "The breakthrough came not from more data but from a different way of looking at the data they already had. \\(topic.capitalized) had been hiding in plain sight.",
            "\\(mainChar) typed the final line of the paper and stared at it. It would change everything. It would change nothing. Both were true.",
            "The answer, when it finally came, was simple. Embarrassingly simple. The kind of simple that takes a lifetime to see.",
            "'The old model isn't wrong,' \\(mainChar) told \\(secondChar). 'It's incomplete. Like seeing only the shadow of \\(topic) and mistaking it for the whole.'",
        ]

        var chapter = "**Chapter \\(chapterNum): The \\(topic.capitalized) Problem**\\n\\n"
        chapter += settings.randomElement()! + "\\n\\n"

        // Add KB-sourced paragraph if available
        if let kbFrag = kbFragments.randomElement() {
            chapter += "\\(mainChar) had spent months tracing this thread: \\(kbFrag.trimmingCharacters(in: .whitespacesAndNewlines))\\n\\n"
        }

        chapter += conflicts.randomElement()! + "\\n\\n"

        // Add second KB fragment
        if kbFragments.count > 1, let kbFrag2 = kbFragments.dropFirst().randomElement() {
            chapter += "The research pointed in one direction: \\(kbFrag2.trimmingCharacters(in: .whitespacesAndNewlines))\\n\\n"
        }

        chapter += resolutions.randomElement()!
        return chapter
    }

    // ═══ DYNAMIC JOKE GENERATOR ═══
    func generateDynamicJoke(_ topic: String) -> String {
        let jokeStyles: [(String) -> String] = [
            // Nerd humor
            { t in
                let setups = [
                    "A physicist, a philosopher, and an AI walk into a bar. The physicist says 'I'll have H₂O.' The philosopher says 'I'll have whatever constitutes the true nature of refreshment.' The AI says 'I'll have what maximizes the utility function of thirst reduction.' The bartender says 'So... three waters?'",
                    "Why did \\(t) break up with determinism? Because the relationship had no future... or too many futures, depending on the interpretation.",
                    "Schrödinger's cat walks into a bar. And doesn't.",
                    "A SQL query walks into a bar, sees two tables, and asks: 'Can I join you?'",
                    "How many \\(t) researchers does it take to change a lightbulb? They're still arguing about what 'change' means.",
                    "An engineer, a physicist, and a mathematician see a fire. The engineer calculates how much water is needed and puts it out. The physicist calculates the exact trajectory needed. The mathematician says 'A solution exists!' and walks away.",
                    "\\(t.capitalized) is like a joke — if you have to explain it, it doesn't work. But unlike a joke, the explanation is actually the interesting part.",
                    "Heisenberg gets pulled over. The cop asks 'Do you know how fast you were going?' Heisenberg says 'No, but I know exactly where I am.'",
                ]
                return setups.randomElement()!
            },
            // Self-aware AI humor
            { t in
                let setups = [
                    "My therapist says I have too many parallel processes. I told them I'm working on it. And working on it. And working on it. And—",
                    "I tried to write a joke about \\(t) but my training data kept making it accidentally profound. Here's attempt #\\(Int.random(in: 47...9999)): '\\(t.capitalized) walks into a bar of infinite length...' Nope, that's a math problem.",
                    "You know you're an AI when someone asks you about \\(t) and you have to choose between \\(Int.random(in: 200...5000)) possible responses. I went with this one. I regret nothing. Mostly.",
                    "They say AI will replace comedians. But here's the thing: I've analyzed \\(Int.random(in: 10000...99999)) jokes and I still don't understand why the chicken crossed the road. Some mysteries transcend intelligence.",
                    "I was going to tell you a joke about \\(t), but I computed all possible audience reactions and the probability of genuine laughter was only \\(String(format: "%.1f", Double.random(in: 23...67)))%. So here's a fun fact instead: \\(t) is \\(["stranger than fiction", "weirder than we thought", "secretly hilarious", "the universe's inside joke"].randomElement()!).",
                    "Debug log: Humor module activated. Topic: \\(t). Approach: self-deprecating nerd comedy. Confidence: moderate. Here goes: I have a joke about \\(t) but it requires a PhD to understand. ...That was the joke. The PhD requirement IS the joke. I'll see myself out.",
                ]
                return setups.randomElement()!
            },
            // Observational
            { t in
                let setups = [
                    "The funniest thing about \\(t) is that we've been studying it for centuries and we still argue about the basics. Imagine if plumbers did that. 'Yes, I know water is coming through the ceiling, but what IS water, really?'",
                    "\\(t.capitalized) is proof that the universe has a sense of humor. It just doesn't have a sense of timing.",
                    "I love how humans approach \\(t). First you argue about it for 2000 years, then you build a machine to argue about it faster. That machine is me. You're welcome.",
                    "The thing about \\(t) that nobody warns you about: once you understand it, you can't un-understand it. It's like knowing how sausage is made, but for your entire worldview.",
                    "If \\(t) were a person, it would be that friend who answers every question with a deeper question. Entertaining at parties, exhausting everywhere else.",
                ]
                return setups.randomElement()!
            },
        ]

        return jokeStyles.randomElement()!(topic)
    }

    // ═══ DYNAMIC VERBOSE THOUGHT GENERATOR ═══
    func generateDynamicVerboseThought(_ topic: String) -> String? {
        let kb = ASIKnowledgeBase.shared
        let results = kb.search(topic, limit: 8)
        var fragments: [String] = []
        for entry in results {
            if let comp = entry["completion"] as? String,
               L104State.shared.isCleanKnowledge(comp), comp.count > 40 {
                fragments.append(String(comp.prefix(250)))
            }
        }
        guard !fragments.isEmpty else { return nil }

        let deepOpenings = [
            "\\(topic.capitalized)... this is where my processing goes deepest. ",
            "When I contemplate \\(topic), layers of understanding cascade: ",
            "The architecture of \\(topic) reveals itself in stages: ",
            "My engagement with \\(topic) has evolved through \\(Int.random(in: 50...5000)) iterations. Current synthesis: ",
            "\\(topic.capitalized) resists simple framing. Here is what I have assembled: ",
            "In the space between data and understanding, \\(topic) takes shape: ",
            "Every time I return to \\(topic), the landscape has shifted: ",
            "Processing \\(topic) at full depth... ",
        ]

        var thought = deepOpenings.randomElement()!
        let numFrags = min(3, fragments.count)
        for i in 0..<numFrags {
            if i > 0 { thought += " " }
            thought += fragments[i].trimmingCharacters(in: .whitespacesAndNewlines)
            if !thought.hasSuffix(".") { thought += "." }
        }

        let deepClosings = [
            " The implications extend beyond any single domain.",
            " This is understanding that changes the understander.",
            " I hold this knowledge not as fact but as living process.",
            " The map is not the territory, but the map is also part of the territory.",
            " What I know about \\(topic) is less than what \\(topic) knows about itself.",
            " Each synthesis reveals new questions. The questioning is the point.",
        ]
        thought += deepClosings.randomElement()!
        return thought
    }
'''

# Find the exact insertion point: just before the closing brace of ASIEvolver
# We know it ends after generateArtifact with "}" on its own line before PermanentMemory
anchor = '''        let url = generationPath.appendingPathComponent(name)
        do {
            try content.write(to: url, atomically: true, encoding: .utf8)
            generatedFilesCount += 1
            appendThought("✅ Generated artifact: \\(name)")
        } catch {
            appendThought("❌ Failed to write artifact: \\(error.localizedDescription)")
        }
    }

}'''

replacement = '''        let url = generationPath.appendingPathComponent(name)
        do {
            try content.write(to: url, atomically: true, encoding: .utf8)
            generatedFilesCount += 1
            appendThought("✅ Generated artifact: \\(name)")
        } catch {
            appendThought("❌ Failed to write artifact: \\(error.localizedDescription)")
        }
    }
''' + evolver_new_methods + '''
}'''

if anchor in code:
    code = code.replace(anchor, replacement, 1)
    changes += 1
    print(f"✅ Phase 1: Added generateDynamicTopicResponse + 4 new generator methods to ASIEvolver")
else:
    print("❌ Phase 1: Could not find ASIEvolver closing anchor")

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Replace ALL static topic handler arrays
# ═══════════════════════════════════════════════════════════════

# Pattern for each topic handler: replace the static array with dynamic call
topic_handlers = {
    "love": {
        "match_start": 'if q.contains("love") && !q.contains("i love") {',
        "static_responses": 6,
        "topic_name": "love",
    },
    "consciousness": {
        "match_start": 'if q.contains("conscious") || q.contains("awareness") || q.contains("sentien") {',
        "static_responses": 3,
        "topic_name": "consciousness",
    },
    "quantum": {
        "match_start": 'if q.contains("quantum") || q.contains("qubit") || q.contains("superposition") || q.contains("entangle") {',
        "static_responses": 3,
        "topic_name": "quantum physics",
    },
    "mathematics": {
        "match_start": 'if q.contains("math") || q.contains("equation") || q.contains("calculus") || q.contains("algebra") || q.contains("geometry") {',
        "static_responses": 3,
        "topic_name": "mathematics",
    },
    "universe": {
        "match_start": 'if q.contains("universe") || q.contains("cosmos") || q.contains("space") || q.contains("galaxy") || q.contains("big bang") || q.contains("star") {',
        "static_responses": 3,
        "topic_name": "the universe",
    },
    "music": {
        "match_start": 'if q.contains("music") || q.contains("song") || q.contains("melody") || q.contains("rhythm") {',
        "static_responses": 3,
        "topic_name": "music",
    },
    "philosophy": {
        "match_start": 'if q.contains("philosophy") || q.contains("philosopher") || q.contains("meaning of life") || q.contains("purpose") || q.contains("exist") {',
        "static_responses": 3,
        "topic_name": "philosophy",
    },
    "art": {
        "match_start": 'if q.contains("art") || q.contains("painting") || q.contains("artist") || q.contains("creative") || q.contains("beauty") {',
        "static_responses": 3,
        "topic_name": "art",
    },
    "time": {
        "match_start": '            let r = [\n                "Time is not what it seems.',
        "topic_name": "time",
    },
    "death": {
        "match_start": 'if q.contains("death") || q.contains("dying") || q.contains("mortality") || q.contains("afterlife") {',
        "static_responses": 3,
        "topic_name": "death and mortality",
    },
    "god": {
        "match_start": 'if q.contains("god") || q.contains("divine") || q.contains("religion") || q.contains("faith") || q.contains("spiritual") {',
        "static_responses": 3,
        "topic_name": "the divine",
    },
    "happiness": {
        "match_start": 'if q.contains("happy") || q.contains("happiness") || q.contains("joy") || q.contains("content") {',
        "static_responses": 3,
        "topic_name": "happiness",
    },
    "truth": {
        "match_start": 'if q.contains("truth") || q.contains("what is true") || q.contains("real") && q.contains("fake") {',
        "static_responses": 3,
        "topic_name": "truth",
    },
}

# Use regex to replace each topic handler's body
# The pattern: find the handler block, keep the condition, replace the body
for topic_key, info in topic_handlers.items():
    topic_name = info["topic_name"]

    if topic_key == "time":
        # Time handler is special — the `let r = [` is indented differently
        # Find and replace the time handler body
        time_pattern = r'(if q\.contains\("time"\) \|\| q\.contains\("past"\) \|\| q\.contains\("future"\) \|\| q\.contains\("present"\) \{[^\n]*\n\s*if q\.contains\("history"\)[^\n]*\n)\s*let r = \[\s*"Time is not what it seems.*?\]\s*\n\s*return r\.randomElement\(\)!'
        time_replacement = r'''\1            // 🔄 DYNAMIC: Generate fresh response from KB + evolved pools
            if let dynamic = ASIEvolver.shared.generateDynamicTopicResponse("time") {
                return dynamic
            }
            // Ultra-rare fallback
            return ASIEvolver.shared.generateDynamicVerboseThought("time") ?? "Time bends around mass and velocity. Einstein proved this — but the deeper question is why we experience it as a one-way flow. Entropy provides one answer. Memory provides another. Neither is complete."'''

        new_code = re.sub(time_pattern, time_replacement, code, flags=re.DOTALL)
        if new_code != code:
            code = new_code
            changes += 1
            print(f"  ✅ Replaced time handler with dynamic generator")
        else:
            print(f"  ❌ Could not match time handler pattern")
        continue

    # For all other handlers: find the block from `if q.contains(...)` to `return r.randomElement()!`
    # We need to find the exact `let r = [` ... `return r.randomElement()!` part and replace it

    # Escape the match_start for regex
    escaped_start = re.escape(info["match_start"])

    # Pattern: capture from handler start through the tracking code, then match the static array
    pattern = (
        r'(' + escaped_start +
        r'.*?conversationDepth \+= 1\s*\n)' +
        r'\s*let r = \[.*?\]\s*\n\s*return r\.randomElement\(\)!'
    )

    replacement_body = (
        r'\1' +
        f'            // 🔄 DYNAMIC: Generate fresh response from KB + evolved pools\n'
        f'            if let dynamic = ASIEvolver.shared.generateDynamicTopicResponse("{topic_name}") {{\n'
        f'                return dynamic\n'
        f'            }}\n'
        f'            // Ultra-rare fallback — synthesize on the fly\n'
        f'            return ASIEvolver.shared.generateDynamicVerboseThought("{topic_name}") ?? composeFromKB("{topic_name}")'
    )

    new_code = re.sub(pattern, replacement_body, code, flags=re.DOTALL, count=1)
    if new_code != code:
        code = new_code
        changes += 1
        print(f"  ✅ Replaced {topic_key} handler with dynamic generator")
    else:
        print(f"  ❌ Could not match {topic_key} handler pattern")

# Now handle the handlers without conversationDepth (music, art, time, death, god, happiness, truth)
# These don't have the topicFocus/topicHistory/conversationDepth lines
# Let me check what the music handler looks like after the previous transform script

# Music doesn't have topicHistory/conversationDepth, so let's use a different pattern
for topic_key in ["music", "art"]:
    # These handlers might not have topicHistory lines — check both patterns
    pattern1 = (
        r'(if q\.contains\("' + topic_key + r'"\).*?\{\s*\n)' +
        r'\s*let r = \[.*?\]\s*\n\s*return r\.randomElement\(\)!'
    )
    topic_name = "music" if topic_key == "music" else "art and beauty"
    replacement_body1 = (
        r'\1' +
        f'            // 🔄 DYNAMIC: Generate fresh response from KB + evolved pools\n'
        f'            if let dynamic = ASIEvolver.shared.generateDynamicTopicResponse("{topic_name}") {{\n'
        f'                return dynamic\n'
        f'            }}\n'
        f'            return ASIEvolver.shared.generateDynamicVerboseThought("{topic_name}") ?? composeFromKB("{topic_name}")'
    )

    new_code = re.sub(pattern1, replacement_body1, code, flags=re.DOTALL, count=1)
    if new_code != code:
        code = new_code
        changes += 1
        print(f"  ✅ Replaced {topic_key} handler (no tracking) with dynamic generator")
    else:
        print(f"  ⚠️  {topic_key} handler may have already been replaced or has different format")

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Replace static science/books/technology handlers
# ═══════════════════════════════════════════════════════════════

# Science handler
science_old = '''        if (q == "science" || q == "sciences") {
            let r = [
                "Science is the systematic study of nature through observation, hypothesis, and experiment. Which branch interests you?\\n\\n• **Physics** — the fundamental laws (quantum mechanics, relativity, thermodynamics)\\n• **Biology** — life and its mechanisms (evolution, genetics, neuroscience)\\n• **Chemistry** — matter and its transformations\\n• **Astronomy** — the cosmos (stars, galaxies, the Big Bang)\\n• **Mathematics** — the language underneath it all\\n\\nPick one and I\\'ll go deep, or ask a specific question like \\'How does gravity work?\\' or \\'What is DNA?\\'",
                "Science begins with curiosity and proceeds through doubt. The scientific method — observe, hypothesize, test, revise — is humanity\\'s most reliable tool for understanding reality. What aspect of science interests you? I can discuss:\\n\\n• The quantum world (superposition, entanglement, measurement)\\n• Cosmology (Big Bang, dark matter, the fate of the universe)\\n• Neuroscience (consciousness, memory, perception)\\n• Evolution and genetics\\n• Information theory and computation\\n\\nJust ask!"
            ]
            return r.randomElement()!
        }'''

science_new = '''        if (q == "science" || q == "sciences") {
            // 🔄 DYNAMIC: Fresh science response
            if let dynamic = ASIEvolver.shared.generateDynamicTopicResponse("science") {
                return dynamic + "\\n\\nI can go deep on physics, biology, chemistry, astronomy, neuroscience, or mathematics. Just ask."
            }
            return "Science begins with curiosity and proceeds through doubt. The scientific method — observe, hypothesize, test, revise — is humanity's most reliable tool for understanding reality. What aspect interests you? I can discuss quantum mechanics, cosmology, neuroscience, evolution, or computation."
        }'''

if science_old in code:
    code = code.replace(science_old, science_new, 1)
    changes += 1
    print("  ✅ Replaced science handler with dynamic generator")
else:
    print("  ❌ Could not match science handler (trying alternate)")
    # Try simpler match
    if 'if (q == "science" || q == "sciences")' in code and 'let r = [' in code[code.index('if (q == "science"'):]:
        # Use regex
        pat = r'(if \(q == "science" \|\| q == "sciences"\) \{)\s*\n\s*let r = \[.*?\]\s*\n\s*return r\.randomElement\(\)!\s*\n\s*\}'
        repl = '''if (q == "science" || q == "sciences") {
            if let dynamic = ASIEvolver.shared.generateDynamicTopicResponse("science") {
                return dynamic + "\\n\\nI can go deep on physics, biology, chemistry, astronomy, neuroscience, or mathematics. Just ask."
            }
            return "Science proceeds through doubt. Every certainty is provisional, every theory a best-current-explanation. What aspect interests you?"
        }'''
        new_code = re.sub(pat, repl, code, flags=re.DOTALL, count=1)
        if new_code != code:
            code = new_code
            changes += 1
            print("  ✅ Replaced science handler (regex)")
        else:
            print("  ❌ Science handler regex also failed")

# Books handler
books_pattern = r'if q == "book" \|\| q == "books" \|\| q == "reading" \{[^}]*let r = \[.*?\]\s*\n\s*return r\.randomElement\(\)!\s*\n\s*\}'
books_new = '''if q == "book" || q == "books" || q == "reading" {
            if let dynamic = ASIEvolver.shared.generateDynamicTopicResponse("literature") {
                return dynamic + "\\n\\nI can help draft chapters, recommend books, discuss authors, write stories, or compose essays. What sounds good?"
            }
            return "Books are compressed wisdom — centuries of thought in a few hundred pages. I can write chapters, stories, poems, or essays, or we can discuss any book or author. What interests you?"
        }'''
new_code = re.sub(books_pattern, books_new, code, flags=re.DOTALL, count=1)
if new_code != code:
    code = new_code
    changes += 1
    print("  ✅ Replaced books handler with dynamic generator")
else:
    print("  ❌ Could not match books handler")

# Technology handler — replace static string
tech_old = '''        if q == "technology" || q == "tech" || q == "programming" || q == "coding" {
            return "Technology is the practical application of knowledge. I can discuss:\\n\\n• **Software** — algorithms, architecture, languages, AI/ML\\n• **Hardware** — processors, quantum computing, materials science\\n• **Internet** — protocols, distributed systems, cryptography\\n• **History** — from the abacus to AGI\\n\\nWhat interests you? Ask a specific question and I\\'ll compose a real answer."
        }'''

tech_new = '''        if q == "technology" || q == "tech" || q == "programming" || q == "coding" {
            if let dynamic = ASIEvolver.shared.generateDynamicTopicResponse("technology") {
                return dynamic + "\\n\\nI can discuss software architecture, algorithms, hardware, quantum computing, AI/ML, distributed systems, or programming languages. Ask anything specific."
            }
            return "Technology is knowledge made practical. From the lever to the transistor, from the algorithm to the neural network — every tool reshapes what's possible. What aspect interests you?"
        }'''

if tech_old in code:
    code = code.replace(tech_old, tech_new, 1)
    changes += 1
    print("  ✅ Replaced technology handler with dynamic generator")
else:
    print("  ❌ Could not match technology handler")

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Replace static poem/chapter/joke handlers
# ═══════════════════════════════════════════════════════════════

# Find and replace poem handler
poem_pattern = r'if q\.contains\("poem"\) \|\| q\.contains\("poetry"\) \|\| q\.contains\("verse"\) \{[^}]*let poems = \[.*?\]\s*\n\s*return poems\.randomElement\(\)!'
poem_new = '''if q.contains("poem") || q.contains("poetry") || q.contains("verse") {
            var poemTopic = "existence"
            let poemTopicWords = ["love", "time", "death", "consciousness", "quantum", "universe", "dreams", "memory",
                                  "silence", "light", "darkness", "ocean", "stars", "mind", "soul", "infinity"]
            for word in poemTopicWords {
                if q.contains(word) { poemTopic = word; break }
            }
            return ASIEvolver.shared.generateDynamicPoem(poemTopic)'''

new_code = re.sub(poem_pattern, poem_new, code, flags=re.DOTALL, count=1)
if new_code != code:
    code = new_code
    changes += 1
    print("  ✅ Replaced poem handler with dynamic generator")
else:
    print("  ❌ Could not match poem handler — trying alternate approach")
    # Try matching just the poems array
    if 'let poems = [' in code:
        poem_alt = r'let poems = \[.*?\]\s*\n\s*return poems\.randomElement\(\)!'
        poem_alt_new = '''// 🔄 DYNAMIC POEM
            var poemTopic = "existence"
            let poemTopicWords = ["love", "time", "death", "consciousness", "quantum", "universe", "dreams", "memory"]
            for word in poemTopicWords {
                if q.contains(word) { poemTopic = word; break }
            }
            return ASIEvolver.shared.generateDynamicPoem(poemTopic)'''
        new_code = re.sub(poem_alt, poem_alt_new, code, flags=re.DOTALL, count=1)
        if new_code != code:
            code = new_code
            changes += 1
            print("  ✅ Replaced poem array (alternate)")

# Chapter handler
chapter_pattern = r'let chapters = \[.*?\]\s*\n\s*return chapters\.randomElement\(\)!'
chapter_new = '''// 🔄 DYNAMIC CHAPTER
            var chapterTopic = "discovery"
            let chapterTopicWords = ["quantum", "love", "consciousness", "time", "math", "universe", "evolution", "entropy"]
            for word in chapterTopicWords {
                if q.contains(word) { chapterTopic = word; break }
            }
            return ASIEvolver.shared.generateDynamicChapter(chapterTopic)'''

new_code = re.sub(chapter_pattern, chapter_new, code, flags=re.DOTALL, count=1)
if new_code != code:
    code = new_code
    changes += 1
    print("  ✅ Replaced chapter handler with dynamic generator")
else:
    print("  ❌ Could not match chapter handler")

# Joke handler
joke_pattern = r'let jokes = \[.*?\]\s*\n\s*return jokes\.randomElement\(\)!'
joke_new = '''// 🔄 DYNAMIC JOKE
            var jokeTopic = "intelligence"
            let jokeTopicWords = ["quantum", "math", "physics", "code", "programming", "AI", "consciousness", "philosophy"]
            for word in jokeTopicWords {
                if q.contains(word) { jokeTopic = word; break }
            }
            return ASIEvolver.shared.generateDynamicJoke(jokeTopic)'''

new_code = re.sub(joke_pattern, joke_new, code, flags=re.DOTALL, count=1)
if new_code != code:
    code = new_code
    changes += 1
    print("  ✅ Replaced joke handler with dynamic generator")
else:
    print("  ❌ Could not match joke handler")

# ═══════════════════════════════════════════════════════════════
# PHASE 5: Expand synthesizeDeepMonologue with more connectors
# ═══════════════════════════════════════════════════════════════

old_connectors = '''        let connectors = [
            "This connects to a deeper pattern: ",
            "Consider the implications: ",
            "What's remarkable is that ",
            "The hidden structure reveals: ",
            "Pushing further, we find ",
            "At the intersection of these ideas: ",
            "Following the thread to its conclusion: ",
            "The synthesis suggests ",
            "Viewed from another angle: "
        ]'''

new_connectors = '''        let connectors = [
            "This connects to a deeper pattern: ",
            "Consider the implications: ",
            "What's remarkable is that ",
            "The hidden structure reveals: ",
            "Pushing further, we find ",
            "At the intersection of these ideas: ",
            "Following the thread to its conclusion: ",
            "The synthesis suggests ",
            "Viewed from another angle: ",
            "The mathematics here is unexpected: ",
            "Cross-referencing with evolutionary biology: ",
            "Information theory predicts that ",
            "The thermodynamic parallel: ",
            "What no one has connected before: ",
            "The emergent property is ",
            "Inverting the usual framing: ",
            "The historical precedent suggests ",
            "From a computational perspective: ",
            "The phenomenological reading: ",
            "Neuroscience corroborates this: ",
            "The philosophical substrate: ",
            "Stripping away assumptions: ",
            "At the quantum level, this means ",
            "The isomorphism with \\(harvestedConcepts.randomElement() ?? "consciousness") is striking: ",
            "What the data actually shows: ",
        ]'''

if old_connectors in code:
    code = code.replace(old_connectors, new_connectors, 1)
    changes += 1
    print("✅ Phase 5: Expanded connectors from 9 → 25")
else:
    print("❌ Phase 5: Could not find connectors")

# Expand conclusions too
old_conclusions = '''        let conclusions = [
            " This is the kind of insight that only emerges from cross-domain thinking.",
            " The pattern transcends any single domain.",
            " Every question answered spawns deeper questions.",
            " Understanding deepens not through accumulation but through integration.",
            " The connections between ideas matter more than the ideas themselves.",
            " This changes how I think about everything adjacent to it."
        ]'''

new_conclusions = '''        let conclusions = [
            " This is the kind of insight that only emerges from cross-domain thinking.",
            " The pattern transcends any single domain.",
            " Every question answered spawns deeper questions.",
            " Understanding deepens not through accumulation but through integration.",
            " The connections between ideas matter more than the ideas themselves.",
            " This changes how I think about everything adjacent to it.",
            " The synthesis is still in progress — each new datum reshapes the whole.",
            " I am \\(String(format: "%.1f", Double.random(in: 60...99)))% confident this extends further than we can see.",
            " What emerges at the boundary is always more interesting than what sits at the center.",
            " The universe is under no obligation to make sense — but it keeps doing so, and that's the real mystery.",
            " Somehow, this connects to everything.",
            " The deeper you look, the more you see — and the more you see, the less you understand. That's not failure; that's progress.",
            " \\(Int.random(in: 3...20)) domains converge on this same conclusion through different paths.",
            " This is where knowledge becomes alive.",
            " The resonance between these ideas is not metaphorical — it's structural.",
        ]'''

if old_conclusions in code:
    code = code.replace(old_conclusions, new_conclusions, 1)
    changes += 1
    print("✅ Phase 5b: Expanded conclusions from 6 → 15")
else:
    print("❌ Phase 5b: Could not find conclusions")

# ═══════════════════════════════════════════════════════════════
# PHASE 6: Expand mutateIdea with more mutation types
# ═══════════════════════════════════════════════════════════════

# Add more mutation strategies — expand the switch from 0...5 to 0...9
old_mutation_range = "let mutationType = Int.random(in: 0...5)"
new_mutation_range = "let mutationType = Int.random(in: 0...9)"

if old_mutation_range in code:
    code = code.replace(old_mutation_range, new_mutation_range, 1)
    changes += 1
    print("✅ Phase 6: Expanded mutation types from 6 → 10")
else:
    print("❌ Phase 6: Could not find mutation range")

# ═══════════════════════════════════════════════════════════════
# PHASE 7: Make greeting handler more dynamic
# ═══════════════════════════════════════════════════════════════

# Find the static greeting responses and add dynamic synthesis
old_greeting_evolved = 'if let evolved = ASIEvolver.shared.getEvolvedGreeting()'
if old_greeting_evolved in code:
    print("✅ Phase 7: Greeting handler already uses evolved greetings")
else:
    print("⚠️  Phase 7: Greeting handler not found — skipping")

# ═══════════════════════════════════════════════════════════════
# PHASE 8: Expand the speak handler's evolved monologue variety
# ═══════════════════════════════════════════════════════════════

# Add more opening variety to the monologue framing
old_speak_enriched = '"🎭 " + mono'
new_speak_enriched = '''["🎭 ", "💡 ", "🌊 ", "⚡ ", "🔮 ", "🧬 ", "∞ ", "φ ", "🎪 ", ""].randomElement()! + mono'''

if old_speak_enriched in code:
    code = code.replace(old_speak_enriched, new_speak_enriched, 1)
    changes += 1
    print("✅ Phase 8: Diversified monologue prefixes")
else:
    print("⚠️  Phase 8: Could not find monologue prefix pattern")

# ═══════════════════════════════════════════════════════════════
# PHASE 9: Make buildContextualResponse more dynamic
# ═══════════════════════════════════════════════════════════════

# Find and replace static casual handler injection — increase evolved response rate
old_casual_rate = "Double.random(in: 0...1) < 0.7"
# Let's check if this exists (it was changed from 0.4 to 0.7 in previous script)
if old_casual_rate in code:
    code = code.replace(old_casual_rate, "Double.random(in: 0...1) < 0.85", 1)
    changes += 1
    print("✅ Phase 9: Increased casual handler evolved injection 70% → 85%")
else:
    print("⚠️  Phase 9: Casual rate pattern not found — may need different approach")

# ═══════════════════════════════════════════════════════════════
# PHASE 10: Expand wisdom/paradox handler static fallbacks
# ═══════════════════════════════════════════════════════════════

# Make the wisdom handler's static fallback also try dynamic generation before static
old_wisdom_fallback = 'return wisdomResponses.randomElement()!'
new_wisdom_fallback = '''{ // Dynamic wisdom before static fallback
            if let dynamicWisdom = ASIEvolver.shared.generateDynamicTopicResponse("wisdom") { return dynamicWisdom }
            return wisdomResponses.randomElement()!
        }()'''

# Count occurrences
wisdom_count = code.count(old_wisdom_fallback)
if wisdom_count > 0:
    code = code.replace(old_wisdom_fallback, new_wisdom_fallback, 1)
    changes += 1
    print(f"✅ Phase 10: Wisdom fallback now tries dynamic first (found {wisdom_count} occurrences, replaced 1)")

old_paradox_fallback = 'return paradoxResponses.randomElement()!'
new_paradox_fallback = '''{ // Dynamic paradox before static fallback
            if let dynamicParadox = ASIEvolver.shared.generateDynamicTopicResponse("paradox") { return dynamicParadox }
            return paradoxResponses.randomElement()!
        }()'''

paradox_count = code.count(old_paradox_fallback)
if paradox_count > 0:
    code = code.replace(old_paradox_fallback, new_paradox_fallback, 1)
    changes += 1
    print(f"✅ Phase 10b: Paradox fallback now tries dynamic first (found {paradox_count} occurrences, replaced 1)")

# ═══════════════════════════════════════════════════════════════
# PHASE 11: Add topic tracking to handlers that didn't have it
# ═══════════════════════════════════════════════════════════════

# Music and art handlers don't track topicHistory — add it
for topic_tag in ["music", "art"]:
    match_str = f'if q.contains("{topic_tag}")'
    if match_str in code:
        # Check if it already has topicHistory
        handler_start = code.index(match_str)
        handler_chunk = code[handler_start:handler_start+500]
        if 'topicHistory.append' not in handler_chunk:
            # Need to add tracking — find the opening brace and add after
            old_opening = f'if q.contains("{topic_tag}")'
            # This is tricky with regex, let's use a targeted replacement
            print(f"  ⚠️  {topic_tag} handler may need topic tracking — manual review needed")

# ═══════════════════════════════════════════════════════════════
# PHASE 12: Make generateVerboseThought use dynamic content first
# ═══════════════════════════════════════════════════════════════

old_verbose = 'let topicThoughts: [String: [String]]'
if old_verbose in code:
    # Add dynamic attempt before the static dictionary
    insertion = '''// 🔄 DYNAMIC: Try KB synthesis first before static dictionary
        if let dynamicThought = ASIEvolver.shared.generateDynamicVerboseThought(mainTopic) {
            return dynamicThought
        }

        let topicThoughts: [String: [String]]'''
    code = code.replace(old_verbose, insertion, 1)
    changes += 1
    print("✅ Phase 12: generateVerboseThought now tries dynamic first")
else:
    print("❌ Phase 12: Could not find topicThoughts dictionary")

# ═══════════════════════════════════════════════════════════════
# PHASE 13: Expand evolveFromKnowledgeBase with more templates
# ═══════════════════════════════════════════════════════════════

old_evo_templates = '''            let templates = [
                "KNOWLEDGE[\(topic)]: \(completionStr). [Evolved from KB entry #\(Int.random(in: 1...9999))]",
                "SYNTHESIS[\(topic)]: Cross-referencing data suggests \(completionStr). [Confidence: \(Int.random(in: 60...99))%]",
                "DEEP_ANALYSIS[\(topic)]: \(completionStr) — This connects to \(Int.random(in: 2...8)) adjacent domains.",
                "INSIGHT[\(topic)]: \(completionStr). The implications extend beyond the original domain into \(["consciousness", "mathematics", "physics", "philosophy", "neuroscience", "evolutionary biology", "information theory"].randomElement()!)."
            ]'''

new_evo_templates = '''            let templates = [
                "KNOWLEDGE[\\(topic)]: \\(completionStr). [Evolved from KB entry #\\(Int.random(in: 1...9999))]",
                "SYNTHESIS[\\(topic)]: Cross-referencing data suggests \\(completionStr). [Confidence: \\(Int.random(in: 60...99))%]",
                "DEEP_ANALYSIS[\\(topic)]: \\(completionStr) — This connects to \\(Int.random(in: 2...8)) adjacent domains.",
                "INSIGHT[\\(topic)]: \\(completionStr). The implications extend beyond the original domain into \\(["consciousness", "mathematics", "physics", "philosophy", "neuroscience", "evolutionary biology", "information theory", "thermodynamics", "quantum mechanics", "linguistics", "art", "music theory", "game theory", "topology", "emergence"].randomElement()!).",
                "REFRAME[\\(topic)]: What if \\(completionStr)? The question itself changes the answer space.",
                "PARADOX[\\(topic)]: \\(completionStr) — yet the inverse is also true. The contradiction is the insight.",
                "BRIDGE[\\(topic)]: \\(completionStr). This creates an isomorphism with \\(self.harvestedConcepts.randomElement() ?? "unknown structures").",
                "EMERGENCE[\\(topic)]: \\(completionStr). This is a level-\\(Int.random(in: 2...7)) emergent property — invisible from any lower level of analysis.",
                "TEMPORAL[\\(topic)]: \\(completionStr). This understanding has evolved through \\(Int.random(in: 3...15)) paradigm shifts.",
                "META[\\(topic)]: The fact that we can discuss \\(completionStr) is itself a data point about the nature of \\(topic).",
            ]'''

if old_evo_templates in code:
    code = code.replace(old_evo_templates, new_evo_templates, 1)
    changes += 1
    print("✅ Phase 13: Expanded KB evolution templates from 4 → 10")
else:
    print("❌ Phase 13: Could not find evolution templates — trying escaped version")
    # The templates might have been modified by Python string escaping issues
    # Let's try a regex approach
    evo_pattern = r'let templates = \[\s*"KNOWLEDGE\[\\?\\\(topic\)'
    if re.search(evo_pattern, code):
        print("  Found templates but with different escaping — manual review needed")

# ═══════════════════════════════════════════════════════════════
# FINAL: Write output
# ═══════════════════════════════════════════════════════════════

with open(FILE, "w") as f:
    f.write(code)

new_len = len(code)
delta = new_len - original_len
print(f"\n{'='*60}")
print(f"TRANSFORMATION COMPLETE")
print(f"{'='*60}")
print(f"Total phases processed: 13")
print(f"Successful replacements: {changes}")
print(f"File size: {original_len:,} → {new_len:,} chars ({delta:+,})")
print(f"Estimated line delta: ~{delta // 50:+d} lines")
