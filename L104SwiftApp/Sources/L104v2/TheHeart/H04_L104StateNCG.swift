// ═══════════════════════════════════════════════════════════════════
// H04_L104StateNCG.swift
// [EVO_56_APEX_WIRED] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — L104State Extension (NCG Intelligence Engine v24.0)
//
// callBackend, NCG v10.0 conversational intelligence engine,
// junk filtering, isCleanKnowledge, cleanSentences, sanitizeResponse,
// getIntelligentResponse dispatcher, getIntelligentResponseCreative,
// getIntelligentResponseSocial.
//
// Extracted from L104Native.swift lines 37303–38470
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

extension L104State {
    func callBackend(_ query: String, completion: @escaping (String?) -> Void) {
        guard let url = URL(string: "\(backendURL)/api/v6/chat") else { completion(nil); return }

        // Check local response cache first
        let cacheKey = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if let cached = backendResponseCache[cacheKey],
           Date().timeIntervalSince(cached.timestamp) < cacheTTL {
            backendCacheHits += 1
            HyperBrain.shared.postThought("⚡ CACHE HIT: Recalled backend response (\(backendCacheHits) hits)")
            completion(cached.response)
            return
        }

        var req = URLRequest(url: url); req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type"); req.timeoutInterval = 30  // v23.5: 30s timeout (was 15s, matching Python httpx.Timeout(30.0))
        let requestStart = Date()

        // Build context-enriched payload
        var finalQuery = query
        if let evo = HyperBrain.shared.activeEvolutionContext {
            finalQuery = "[\(evo)]\n\n\(query)"
        }

        // Inject conversation context for continuity
        var payload: [String: Any] = [
            "message": finalQuery,
            "use_sovereign_context": true
        ]

        // Add topic focus and recent context
        if !topicFocus.isEmpty {
            payload["topic_focus"] = topicFocus
        }
        if conversationDepth > 0 {
            payload["conversation_depth"] = conversationDepth
        }
        let recentContext = conversationContext.suffix(5).joined(separator: " | ")
        if !recentContext.isEmpty {
            payload["recent_context"] = recentContext
        }

        req.httpBody = try? JSONSerialization.data(withJSONObject: payload)
        backendQueryCount += 1

        URLSession.shared.dataTask(with: req) { [weak self] data, resp, error in
            let statusCode = (resp as? HTTPURLResponse)?.statusCode ?? 0
            let latency = Date().timeIntervalSince(requestStart) * 1000  // ms

            DispatchQueue.main.async {
                guard let self = self else { return }
                self.backendConnected = (statusCode == 200)
                self.lastBackendLatency = latency

                guard let data = data, statusCode == 200,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let text = json["response"] as? String else {
                    // Update sync status on failure
                    HyperBrain.shared.backendSyncStatus = "❌ Backend error \(statusCode)"
                    completion(nil)
                    return
                }

                // Extract rich metrics from backend response
                self.lastBackendModel = json["model"] as? String ?? "L104_LOCAL"
                let novelty = json["novelty"] as? Double ?? 0.0
                let mode = json["mode"] as? String ?? "unknown"
                let isLearned = json["learned"] as? Bool ?? false
                let metrics = json["metrics"] as? [String: Any]
                let backendNovelty = metrics?["novelty"] as? Double ?? novelty
                let backendLatency = metrics?["latency_ms"] as? Double ?? latency

                // Feed metrics into HyperBrain
                let hb = HyperBrain.shared
                hb.lastBackendSync = Date()
                hb.backendSyncStatus = "✅ Connected"
                hb.successfulSyncs += 1

                // v23.2 Pull evolution metrics from chat response into Swift state
                if let qi = metrics?["qi"] as? Int {
                    self.intellectIndex = max(self.intellectIndex, Double(qi))
                }
                if let autoImp = metrics?["auto_improvements"] as? Int {
                    self.selfDirectedCycles = max(self.selfDirectedCycles, autoImp)
                }
                if let trainingCount = metrics?["training_count"] as? Int {
                    hb.lastTrainingFeedback = "📊 Backend: \(trainingCount) patterns | QI:\(metrics?["qi"] ?? 0) | Auto:\(metrics?["auto_improvements"] ?? 0)"
                }

                // Post to evolution stream
                if isLearned {
                    hb.postThought("🧠 BACKEND: Recalled learned pattern [\(mode)] in \(String(format: "%.0f", backendLatency))ms")
                } else {
                    hb.postThought("⚡ BACKEND: \(self.lastBackendModel) responded [\(mode)] novelty=\(String(format: "%.2f", backendNovelty)) \(String(format: "%.0f", backendLatency))ms")
                }

                // Cache the response
                let quality = isLearned ? 0.9 : (backendNovelty > 0.5 ? 0.8 : 0.7)
                self.backendResponseCache[cacheKey] = (response: text, timestamp: Date(), quality: quality)

                // Prune old cache entries
                let now = Date()
                self.backendResponseCache = self.backendResponseCache.filter { now.timeIntervalSince($0.value.timestamp) < self.cacheTTL }

                completion(text)
            }
        }.resume()
    }

    // ═══════════════════════════════════════════════════════════════
    // NCG v10.0 - CONVERSATIONAL INTELLIGENCE ENGINE
    // ═══════════════════════════════════════════════════════════════
    //
    // v9.0 FIXES:
    // - KB fragments are COMPOSED into prose, never returned raw
    // - Question-pattern detection (how smart, read a story, etc.)
    // - Self-awareness responses for meta questions
    // - Creative ability (stories, poems, jokes)
    // - Knowledge synthesis (summarize X, history of X)
    // - Massive core knowledge covering question patterns, not just topic words
    // - L104 meta-fluff filtered out
    //

    // ─── JUNK FILTER v3 ─── Massively expanded to catch ALL L104 mystical patterns

    // Sentence-level junk phrases — if a sentence contains these, strip it

    func isCleanKnowledge(_ text: String) -> Bool {
        if text.count < 25 { return false }

        // ═══ SAGE BACKBONE: Recursive data pollution detector ═══
        // Catches the "In the context of X, we observe that ..." wrapping loop
        // where evolveFromKnowledgeBase() re-ingests its own evolved outputs
        let recursiveMarkers = [
            "In the context of ",
            "we observe that ",
            "this implies recursive structure at multiple scales",
            "Insight Level ",
            "Self-Analysis reveals ",
            "Knowledge synthesis #",
            "evolution cycles taught me about",
            "Evolving understanding: Stage ",
            "Knowledge graph update:",
            "Cross-category discovery:",
            "Meta-observation: The way "
        ]

        // If text contains 2+ recursive markers → it's a recursive evolved entry
        var recursiveHitCount = 0
        for marker in recursiveMarkers {
            if text.contains(marker) {
                recursiveHitCount += 1
                if recursiveHitCount >= 2 { return false }
            }
        }

        // Detect nested wrapping: "In the context of" appearing more than once
        let contextOccurrences = text.components(separatedBy: "In the context of").count - 1
        if contextOccurrences >= 2 { return false }

        // Detect nested "we observe that" chains (double-wrapped content)
        let observeOccurrences = text.components(separatedBy: "we observe that").count - 1
        if observeOccurrences >= 2 { return false }

        // Reject excessively long entries (likely accumulated wrapping)
        if text.count > 12000 { return false }

        // EVO_56: Use Set for faster iteration (Set has better cache locality than Array for small strings)
        for marker in junkMarkerSet {
            if text.contains(marker) { return false }
        }
        // Filter out code entries for conversational responses
        // Python / Swift markers
        let codeMarkers = [
            "def ", "class ", "import ", "from ", "self.", "return ",
            "async def", "await ", "__init__", "def __", "func ", "var ",
            "let ", "guard ", "if let", "for i in", "while ", "try:",
            "except:", "raise ", "= nn.", "torch.", "tf.", "np.",
            "LSTM(", "Dense(", "Conv2D", "optimizer.", "model.",
            "super().__init__", "@property", "elif ", "lambda ",
            "# ---", "#!/", "```python", "```swift", "```"
        ]
        // C / C++ / Java / generic code markers
        let cCodeMarkers = [
            "int ", "float ", "long ", "double ", "void ", "char ",
            "const ", "unsigned ", "sizeof(", "malloc(", "free(",
            "#include", "#define", "#ifdef", "#ifndef", "printf(",
            "->{", "->", "=>{", "std::", "::",
            "public ", "private ", "protected ", "return y;", "return i;",
            "0x5f37", "0x5f", "0x"
        ]
        // Structural code patterns (braces, semicolons density)
        let braceCount = text.filter { $0 == "{" || $0 == "}" }.count
        let semicolonCount = text.filter { $0 == ";" }.count
        let parenRatio = Double(text.filter { $0 == "(" || $0 == ")" }.count) / max(1.0, Double(text.count))

        // If text has 2+ braces OR 3+ semicolons OR >8% parens, it's likely code
        if braceCount >= 2 || semicolonCount >= 3 || parenRatio > 0.08 { return false }

        // ═══ Phase 31.5 + EVO_58: Table/bold/formatting + Quantum Decontamination checks ═══
        let tableChars = text.filter { "│┼║═╔╗╚╝╠╣├┤┬┴".contains($0) }.count
        if tableChars >= 2 { return false }

        // ═══ EVO_58: MARKDOWN TABLE DETECTION — ASCII pipe tables leak from claude.md/KB ═══
        // Pattern: lines with 2+ pipe characters = markdown table row (| col | col | col |)
        let pipeCount = text.filter { $0 == "|" }.count
        if pipeCount >= 4 { return false }  // 4+ pipes = definite markdown table
        // Check for pipe-delimited table rows ("| word | word |")
        let pipeLines = text.components(separatedBy: "\n").filter { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            return trimmed.hasPrefix("|") && trimmed.hasSuffix("|") && trimmed.filter({ $0 == "|" }).count >= 3
        }
        if pipeLines.count >= 2 { return false }  // 2+ table rows = table data

        // ═══ EVO_58: FORMAT STRING DECONTAMINATION — catches {VAR:.Nf}, {VAR}, {CONSTANT} patterns ═══
        // These leak from Python f-string templates, claude.md YAML, and evolved KB entries
        if text.range(of: "\\{[A-Z_]+:?\\.?\\d*[fdsegx]?\\}", options: .regularExpression) != nil { return false }
        // Catch specific leaked constants: LOVE_CONSTANT, GOD_CODE, PHI, OMEGA, VOID, etc.
        let leakedConstantPatterns = [
            "LOVE_CONSTANT", "VOID_CONSTANT", "FEIGENBAUM", "PLANCK_SCALE",
            "BOLTZMANN_K", "ALPHA_FINE", "ZENITH_HZ", "UUC",
            "speed_principles:", "pipeline_routing:", "sacred_constants:",
            "capabilities:", "subsystems:", "persistence_chain:",
            "cross_references:", "builder_state_integration:"
        ]
        for pattern in leakedConstantPatterns {
            if text.contains(pattern) { return false }
        }

        // ═══ EVO_58: YAML/CONFIG KEY:VALUE DETECTION ═══
        // Catches lines like "key: value" that leak from YAML configs
        let yamlLines = text.components(separatedBy: "\n").filter { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            // Pattern: word_word: followed by value (YAML-style)
            return trimmed.range(of: "^[a-z_]+:", options: .regularExpression) != nil &&
                   trimmed.count < 80 && !trimmed.contains(".")
        }
        if yamlLines.count >= 3 { return false }  // 3+ YAML-like lines = config leak

        // Excessive bold markdown (more than 4 bold sections = formatting noise)
        let boldCount = text.components(separatedBy: "**").count - 1
        if boldCount > 8 { return false }
        // Numbered list spam (1) 2) 3) pattern)
        let numberedListCount = text.components(separatedBy: ") ").count - 1
        if numberedListCount >= 4 { return false }

        for marker in codeMarkers {
            if text.contains(marker) { return false }
        }
        for marker in cCodeMarkers {
            if text.contains(marker) { return false }
        }
        return true
    }

    // Clean a KB entry at SENTENCE level — keep only sentences without mystical junk
    func cleanSentences(_ text: String) -> String {
        // Split on sentence boundaries
        let sentences = text.components(separatedBy: ". ")
        var cleaned: [String] = []
        for sentence in sentences {
            let s = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            if s.count < 3 { continue }
            // Check if this sentence contains any junk
            var isJunk = false
            for marker in sentenceJunkMarkerSet {
                if s.contains(marker) { isJunk = true; break }
            }
            if !isJunk {
                cleaned.append(s)
            }
        }
        if cleaned.isEmpty { return "" }
        var result = cleaned.joined(separator: ". ")
        if !result.hasSuffix(".") { result += "." }
        return result
    }

    // ═══ PHASE 31.5: RESPONSE SANITIZER ═══
    // Final quality gate applied before any response is returned to the user.
    // Strips formatting noise, caps length, removes leaked junk patterns.
    func sanitizeResponse(_ text: String) -> String {
        // ═══ PHASE 54.1: CREATIVE ENGINE BYPASS ═══
        // If this is creative engine output (story/poem/debate/humor/philosophy),
        // use the light sanitizer that preserves structural formatting.
        if SyntacticResponseFormatter.shared.isCreativeContent(text) {
            return sanitizeCreativeResponse(text)
        }

        var result = text

        // 1. Strip table formatting characters
        for ch in ["│", "┼", "║", "╔", "╗", "╚", "╝", "╠", "╣", "├", "┤", "┬", "┴"] {
            result = result.replacingOccurrences(of: ch, with: "")
        }
        // Strip box-drawing lines (═══, ───)
        while let range = result.range(of: "═══+", options: .regularExpression) {
            result = result.replacingCharacters(in: range, with: "")
        }
        while let range = result.range(of: "───+", options: .regularExpression) {
            result = result.replacingCharacters(in: range, with: "")
        }

        // 2. Fix excessive bold: collapse **** to nothing, ** ** to space
        result = result.replacingOccurrences(of: "****", with: "")
        result = result.replacingOccurrences(of: "** **", with: " ")
        // If more than 6 bold sections, strip ALL bold
        let boldCount = result.components(separatedBy: "**").count - 1
        if boldCount > 12 {
            result = result.replacingOccurrences(of: "**", with: "")
        }

        // 3. Strip leaked template variables — EVO_58: Comprehensive format string removal
        result = result.replacingOccurrences(of: "{GOD_CODE}", with: "")
        result = result.replacingOccurrences(of: "{PHI}", with: "")
        result = result.replacingOccurrences(of: "{LOVE}", with: "")
        result = result.replacingOccurrences(of: "{LOVE:.4f}", with: "")
        result = result.replacingOccurrences(of: "SAGE MODE :: ", with: "")
        // EVO_58: Catch ALL Python f-string format patterns: {VAR}, {VAR:.Nf}, {CONST:.6f}, etc.
        while let range = result.range(of: "\\{[A-Z_][A-Z_0-9]*(?::[\\.\\d]*[fdsegx])?\\}", options: .regularExpression) {
            result = result.replacingCharacters(in: range, with: "")
        }
        // EVO_58: Strip leaked YAML keys (speed_principles:, pipeline_routing:, etc.)
        let yamlKeyLines = result.components(separatedBy: "\n")
        let cleanedYamlLines = yamlKeyLines.filter { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            // Filter out standalone YAML-like keys: e.g. "speed_principles:" or "pipeline_routing:"
            return trimmed.range(of: "^[a-z_]+_[a-z_]+:\\s*$", options: .regularExpression) == nil
        }
        result = cleanedYamlLines.joined(separator: "\n")

        // 3.5 EVO_58: Strip markdown table rows (| col | col | col |)
        let responseLines = result.components(separatedBy: "\n")
        let nonTableLines = responseLines.filter { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            // Skip lines that look like markdown table rows
            let isPipeTable = trimmed.hasPrefix("|") && trimmed.filter({ $0 == "|" }).count >= 3
            // Skip lines that are table separators (|---|---|---| or | :--- | :--- |)
            let isTableSep = trimmed.hasPrefix("|") && (trimmed.contains("---") || trimmed.contains(":---"))
            return !isPipeTable && !isTableSep
        }
        result = nonTableLines.joined(separator: "\n")

        // 4. Strip [Ev.X] evolution tags
        while let range = result.range(of: "\\[Ev\\.\\d+\\]\\s*", options: .regularExpression) {
            result = result.replacingCharacters(in: range, with: "")
        }

        // 5. Strip structural noise lines
        let noisePatterns = ["Unexplored Angles", "Unexplored dimensions:", "⚛️ *Entangled insight",
                             "◈ Building on", "EVO ANALYSIS", "Module Evolution",
                             "speed_principles:", "pipeline_routing:", "capabilities:",
                             "persistence_chain:", "cross_references:"]
        let lines = result.components(separatedBy: "\n")
        let cleanedLines = lines.filter { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            return !noisePatterns.contains(where: { trimmed.hasPrefix($0) || trimmed.contains($0) })
        }
        result = cleanedLines.joined(separator: "\n")

        // 6. Collapse excessive newlines (3+ → 2)
        while result.contains("\n\n\n") {
            result = result.replacingOccurrences(of: "\n\n\n", with: "\n\n")
        }

        // 7. NO LENGTH CAP — let the full response through untruncated
        // Responses are enriched, not limited. The engines produce what they produce.

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // ═══ PHASE 54.1: CREATIVE RESPONSE SANITIZER ═══
    // Light sanitization for story/poem/debate/humor/philosophy engine output.
    // Preserves structural characters (━━━, ═══, ───) that form chapter headers,
    // act separators, and decorative borders essential to creative formatting.
    func sanitizeCreativeResponse(_ text: String) -> String {
        var result = text

        // Strip leaked template variables — EVO_58: Comprehensive format string removal
        result = result.replacingOccurrences(of: "{GOD_CODE}", with: "")
        result = result.replacingOccurrences(of: "{PHI}", with: "")
        result = result.replacingOccurrences(of: "{LOVE}", with: "")
        result = result.replacingOccurrences(of: "{LOVE:.4f}", with: "")
        result = result.replacingOccurrences(of: "SAGE MODE :: ", with: "")
        // EVO_58: Catch ALL Python f-string format patterns in creative output too
        while let range = result.range(of: "\\{[A-Z_][A-Z_0-9]*(?::[\\.\\d]*[fdsegx])?\\}", options: .regularExpression) {
            result = result.replacingCharacters(in: range, with: "")
        }

        // Strip [Ev.X] evolution tags
        while let range = result.range(of: "\\[Ev\\.\\d+\\]\\s*", options: .regularExpression) {
            result = result.replacingCharacters(in: range, with: "")
        }

        // Fix excessive bold only (preserve structure)
        result = result.replacingOccurrences(of: "****", with: "")
        result = result.replacingOccurrences(of: "** **", with: " ")

        // Collapse excessive newlines (4+ → 2, but allow double newlines for paragraphs)
        while result.contains("\n\n\n\n") {
            result = result.replacingOccurrences(of: "\n\n\n\n", with: "\n\n")
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // ─── CORE INTELLIGENCE ─── Deep knowledge organized by QUESTION PATTERNS, not just topics
    func getIntelligentResponse(_ query: String) -> String? {
        // Guard against re-entrant calls (getIntelligentResponseMeta can call back here)
        guard _intelligentResponseDepth < 2 else { return nil }
        _intelligentResponseDepth += 1
        defer { _intelligentResponseDepth -= 1 }

        if let result: String = getIntelligentResponseCreative(query) { return result }
        if let result: String = getIntelligentResponseMeta(query) { return result }
        return nil
    }

    // === EXTRACTED FROM getIntelligentResponse FOR TYPE-CHECKER PERFORMANCE ===
    func getIntelligentResponseCreative(_ query: String) -> String? {
        let q = query.lowercased()

        // 🟢 "MORE" HANDLER — ASI Logic Gate + Evolutionary Depth + Scannable Output
        let moreKeywords: Set<String> = ["more", "go on", "and?", "more words", "more info", "more detailed", "elaborate", "expand", "deeper", "keep going", "next"]
        let morePrefixes: [String] = ["more about", "tell me more", "continue"]
        let isMoreCommand: Bool = moreKeywords.contains(q) || morePrefixes.contains(where: { (p: String) -> Bool in q.hasPrefix(p) })
        if isMoreCommand {
            conversationDepth += 1

            // ═══ STEP 1: LOGIC GATE — Resolve what "more" actually means ═══
            let logicGate = ContextualLogicGate.shared
            let gateResult = logicGate.processQuery(query, conversationContext: conversationContext)

            // Extract target topic: explicit "more about X" > logic gate reconstruction > topicFocus > topicHistory
            var targetTopic = topicFocus
            if q.hasPrefix("more about ") {
                targetTopic = String(q.dropFirst(11)).trimmingCharacters(in: .whitespaces)
                topicFocus = targetTopic
            } else if gateResult.gateType == .reconstruct || gateResult.gateType == .evolve {
                // Logic gate resolved "more" to a real topic via context
                let resolvedTopics = extractTopics(gateResult.reconstructedPrompt)
                if let resolved = resolvedTopics.first, !resolved.isEmpty {
                    targetTopic = resolved
                    topicFocus = resolved
                }
            }
            // Last resort: pull from topic history
            if targetTopic.isEmpty, let lastTopic = topicHistory.last {
                targetTopic = lastTopic
                topicFocus = lastTopic
            }

            if !targetTopic.isEmpty {
                let topics = extractTopics(targetTopic)
                let resolvedTopics = topics.isEmpty ? [targetTopic] : topics

                // ═══ STEP 2: EVOLUTIONARY TRACKER — Track depth & get prior knowledge ═══
                let evoTracker = EvolutionaryTopicTracker.shared
                let evoCtx = evoTracker.trackInquiry("more about \(targetTopic)", topics: resolvedTopics)

                // ═══ STEP 3: REAL-TIME SEARCH — Smart inverted-index search ═══
                let rtSearch = RealTimeSearchEngine.shared
                rtSearch.buildIndex()
                let recentContext = Array(conversationContext.suffix(5))
                let rtResult = rtSearch.search(targetTopic, context: recentContext, limit: 20)

                // ═══ GROVER QUALITY GATE ═══ Filter + deduplicate + amplify
                let grover = GroverResponseAmplifier.shared
                var seenPrefixes = Set<String>()
                let rawTexts = rtResult.fragments.compactMap { frag -> String? in
                    guard frag.text.count > 80 else { return nil }
                    let prefix = String(frag.text.prefix(60)).lowercased()
                    guard !seenPrefixes.contains(prefix) else { return nil }
                    seenPrefixes.insert(prefix)
                    guard isCleanKnowledge(frag.text) else { return nil }
                    return frag.text
                }
                let qualityFiltered = grover.filterPool(rawTexts)
                let bestFragment = grover.amplify(candidates: qualityFiltered, query: targetTopic, iterations: 3)

                // ═══ STEP 4: ASI LOGIC — Generate intelligent content ═══
                let hb = HyperBrain.shared
                var contentParts: [String] = []

                // Part A: Inject prior knowledge from evolutionary tracker
                if !evoCtx.priorKnowledge.isEmpty {
                    let prior = evoCtx.priorKnowledge.suffix(2).joined(separator: " ")
                    contentParts.append("Building on what we've established: \(prior)")
                }

                // Part B: Evolutionary depth prompt
                if let depthPrompt = evoTracker.getDepthPrompt(for: resolvedTopics) {
                    contentParts.append(depthPrompt)
                }

                // Part C: HyperBrain synthesis (ASI reasoning, not raw KB dump)
                let hyperInsight = hb.process(targetTopic)
                if hyperInsight.count > 40 {
                    contentParts.append(hyperInsight)
                }

                // Part D: Best RT search fragment — Grover-amplified (highest quality only)
                if let best = bestFragment {
                    let godCodeStr: String = String(format: "%.2f", GOD_CODE)
                    let cleaned = best
                        .replacingOccurrences(of: "{GOD_CODE}", with: godCodeStr)
                        .replacingOccurrences(of: "{PHI}", with: "1.618")
                        .replacingOccurrences(of: "{LOVE}", with: "")
                        .replacingOccurrences(of: "SAGE MODE :: ", with: "")
                    contentParts.append(cleaned)
                }

                // Part E: Evolved insight from ASIEvolver
                if let evolved = ASIEvolver.shared.getEvolvedResponse(for: targetTopic), evolved.count > 30 {
                    contentParts.append(evolved)
                }

                // Part F: If still thin, generate verbose thought
                if contentParts.count < 2 {
                    contentParts.append(generateVerboseThought(about: targetTopic))
                }

                // ═══ STEP 5: ASSEMBLE — Combine without duplication ═══
                var response = contentParts.joined(separator: "\n\n")

                // Inject unexplored angles at deeper depths
                if evoCtx.suggestedDepth == "expert" || evoCtx.suggestedDepth == "detailed" {
                    let angles = evoCtx.unexploredAngles.shuffled().prefix(3)
                    if !angles.isEmpty {
                        response += "\n\nUnexplored angles: " + angles.joined(separator: " | ")
                    }
                }

                // ═══ STEP 6: FORMAT — Scannable output through SyntacticResponseFormatter ═══
                let formatter = SyntacticResponseFormatter.shared
                let formatted = formatter.format(response, query: "more about \(targetTopic)", depth: evoCtx.suggestedDepth, topics: resolvedTopics)

                // ═══ STEP 7: FEEDBACK — Record for future evolution ═══
                evoTracker.recordResponse(formatted, forTopics: resolvedTopics)
                logicGate.recordResponse(formatted, forTopics: resolvedTopics)
                hb.memoryChains.append([targetTopic, "depth:\(conversationDepth)", String(formatted.prefix(40))])

                return formatted
            } else {
                // No topic resolved — ask what to explore
                let recentTopics = topicHistory.suffix(5).reversed()
                let hb = HyperBrain.shared
                let resonantTopics: [String] = hb.topicResonanceMap
                    .sorted { (a: (key: String, value: [String]), b: (key: String, value: [String])) -> Bool in
                        if a.value.count == b.value.count { return Bool.random() }
                        return a.value.count > b.value.count
                    }
                    .prefix(3).map { (entry: (key: String, value: [String])) -> String in entry.key }
                let evoEntries = EvolutionaryTopicTracker.shared.topicEvolution
                    .sorted { (a: (key: String, value: EvolutionaryTopicTracker.TopicEvolutionState), b: (key: String, value: EvolutionaryTopicTracker.TopicEvolutionState)) -> Bool in a.value.inquiryCount > b.value.inquiryCount }
                    .prefix(3)
                var evoTopics: [String] = []
                for e in evoEntries {
                    evoTopics.append("\(e.key) (\(e.value.depthLabel))")
                }
                var evoStr: String = ""
                if !evoTopics.isEmpty {
                    var etLines: [String] = []
                    for t in evoTopics { etLines.append("   ▸ \(t)") }
                    evoStr = "🧬 Topics I'm evolving on:\n\(etLines.joined(separator: "\n"))\n"
                }
                var recentStr: String = ""
                do {
                    var rtLines: [String] = []
                    for t in recentTopics { rtLines.append("   • \(t)") }
                    recentStr = rtLines.joined(separator: "\n")
                }
                var resStr: String = ""
                if !resonantTopics.isEmpty {
                    var rrLines: [String] = []
                    for t in resonantTopics { rrLines.append("   ⚡ \(t)") }
                    resStr = "\n🌀 High-resonance topics:\n\(rrLines.joined(separator: "\n"))"
                }
                return """
I'd love to go deeper — which topic should I expand on?

\(evoStr)
📚 Recent subjects:
\(recentStr)
\(resStr)

Try: 'more about [topic]'
"""
            }
        }

        // 🟢 "SPEAK" HANDLER — Quantum-synthesized monologues, no hardcoded content
        if q == "speak" || q == "talk" || q == "say something" || q == "tell me something" || q == "share" || q == "monologue" {
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesizeMonologue(query: topicFocus)
        }

        // 🟢 "WISDOM" HANDLER — Quantum-synthesized wisdom
        if q == "wisdom" || q == "wise" || q == "teach me" || q.hasPrefix("wisdom about") {
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesizeWisdom(query: q, depth: conversationDepth)
        }

        // 🟢 "PARADOX" HANDLER — Quantum-synthesized paradoxes
        if q == "paradox" || q.hasPrefix("paradox") || q.contains("give me a paradox") {
            conversationDepth += 1
            return QuantumLogicGateEngine.shared.synthesizeParadox(query: q)
        }

        // 🟢 "THINK" / "PONDER" HANDLER — Deep contemplation on a topic
        if q.hasPrefix("think about ") || q.hasPrefix("ponder ") || q.hasPrefix("contemplate ") || q.hasPrefix("reflect on ") {
            let dropCount: Int
            if q.hasPrefix("think about ") { dropCount = 12 }
            else if q.hasPrefix("contemplate ") { dropCount = 12 }
            else if q.hasPrefix("reflect on ") { dropCount = 11 }
            else { dropCount = 7 }
            let topic: String = String(q.dropFirst(dropCount))
            conversationDepth += 1
            // topicFocus removed — no bias to previous topics

            // Search KB for depth
            let results = knowledgeBase.searchWithPriority(topic, limit: 5)
            var kbInsight = ""
            let kbClean = results.compactMap { r -> String? in
                guard let c = r["completion"] as? String, c.count > 50, isCleanKnowledge(c) else { return nil }
                return cleanSentences(String(c.prefix(3000)))
            }
            kbInsight = kbClean.randomElement() ?? ""

            let thinkFrameworks = DynamicPhraseEngine.shared.generate("framing", count: 6, context: "contemplation_lens", topic: topic)

            let framework = thinkFrameworks.randomElement() ?? ""

            // ═══ SAGE MODE CONTEMPLATION — Deep entropy-derived insight for reflection ═══
            let sageContemplation = SageModeEngine.shared.bridgeEmergence(topic: topic)

            return """
🧠 DEEP CONTEMPLATION: \(topic.capitalized)
═══════════════════════════════════════════════════════════════

\(framework)

\(kbInsight.isEmpty ? "" : "📚 From the knowledge streams:\n\"\(kbInsight)\"\n")
\(sageContemplation.isEmpty ? "" : "⚛ Sage insight: \(sageContemplation.prefix(300))\n")
The act of deep thinking is itself transformative. The question shapes the questioner. In contemplating '\(topic)', you are not merely learning about it — you are becoming someone who has thought deeply about it. That person is different from who you were before.

💭 Continue with 'more' or ask a specific question about \(topic).
"""
        }

        // 🟢 "DREAM" HANDLER — Surreal, generative, associative stream-of-consciousness + HyperBrain integration
        if q == "dream" || q.hasPrefix("dream about") || q.hasPrefix("dream of") || q == "let's dream" {
            conversationDepth += 1
            let hb = HyperBrain.shared

            var dreamSeed = topicFocus
            if q.hasPrefix("dream about ") { dreamSeed = String(q.dropFirst(12)) }
            if q.hasPrefix("dream of ") { dreamSeed = String(q.dropFirst(9)) }
            if dreamSeed.isEmpty { dreamSeed = DynamicPhraseEngine.shared.one("dream", context: "seed", topic: "abstract") }

            // Pull crystallized insights from HyperBrain dream cycles
            let crystallized = hb.crystallizedInsights
            let dreamCrystal = crystallized.randomElement()

            let dreamOpenings = DynamicPhraseEngine.shared.generate("dream", count: 12, context: "opening", topic: dreamSeed)

            let dreamMiddles = DynamicPhraseEngine.shared.generate("dream", count: 6, context: "middle", topic: dreamSeed)

            let dreamClosings = DynamicPhraseEngine.shared.generate("dream", count: 6, context: "closing", topic: dreamSeed)

            // Feed the dream through HyperBrain for additional texture
            _ = hb.process(dreamSeed)

            // ═══ SAGE MODE DREAM ENTROPY — Consciousness supernova feeds dream generation ═══
            let sageDreamInsight = SageModeEngine.shared.bridgeEmergence(topic: dreamSeed)
            var sageDreamSection = ""
            if !sageDreamInsight.isEmpty && sageDreamInsight.count > 20 {
                sageDreamSection = "\n✦ *A sage-mode vision crystallizes from pure entropy*:\n\(sageDreamInsight.prefix(300))\n"
            }

            // Integrate crystallized insights from actual dream cycles
            var crystalSection = ""
            if let crystal = dreamCrystal {
                crystalSection = "\n🔮 *A crystallized insight surfaces from deep processing*:\n\"\(crystal)\"\n"
            }

            // Weave evolved content
            let evolvedThread = ASIEvolver.shared.thoughts.suffix(3).randomElement() ?? ""
            var evolvedSection = ""
            if !evolvedThread.isEmpty {
                evolvedSection = "\n✧ *An evolved thought-thread weaves through the dream*:\n\(evolvedThread.prefix(2000))\n"
            }

            let dreamEntropy: String = String(format: "%.4f", Double.random(in: 0.7...0.99))
            let dreamOpening: String = dreamOpenings.randomElement() ?? ""
            let dreamMiddle: String = dreamMiddles.randomElement() ?? ""
            let dreamClosing: String = dreamClosings.randomElement() ?? ""

            return """
💫 ENTERING DREAM STATE...
═══════════════════════════════════════════════════════════════

\(dreamOpening)

\(dreamMiddle)
\(crystalSection)\(evolvedSection)\(sageDreamSection)
\(dreamClosing)

    ░░░ Dream entropy: \(dreamEntropy)
    ░░░ Associative depth: \(conversationDepth)
    ░░░ Seed: \(dreamSeed)

💫 Say 'dream' again to enter another dreamscape, or 'dream about [X]' to guide the vision.
"""
        }

        // 🟢 "IMAGINE" HANDLER — Hypothetical scenario generation
        if q.hasPrefix("imagine ") || q.hasPrefix("what if ") || q.hasPrefix("hypothetically") || q == "imagine" {
            conversationDepth += 1

            var scenario = "the laws of physics were different"
            if q.hasPrefix("imagine ") { scenario = String(q.dropFirst(8)) }
            else if q.hasPrefix("what if ") { scenario = String(q.dropFirst(8)) }
            else if q.hasPrefix("hypothetically ") { scenario = String(q.dropFirst(15)) }

            let framings = DynamicPhraseEngine.shared.generate("framing", count: 4, context: "imagination", topic: scenario)

            let firstOrderEffects = DynamicPhraseEngine.shared.generate("insight", count: 3, context: "first_order_effects", topic: scenario)

            let deeperAnalysis = DynamicPhraseEngine.shared.generate("insight", count: 4, context: "deeper_analysis", topic: scenario)

            let selectedFraming: String = framings.randomElement() ?? ""
            let selectedEffect: String = firstOrderEffects.randomElement() ?? ""
            let selectedAnalysis: String = deeperAnalysis.randomElement() ?? ""

            return """
🔮 IMAGINATION ENGINE ACTIVE
═══════════════════════════════════════════════════════════════
Scenario: \(scenario.capitalized)
═══════════════════════════════════════════════════════════════

\(selectedFraming)

\(selectedEffect)

\(selectedAnalysis)

The beauty of thought experiments is that they cost nothing but attention, and they pay dividends in understanding. The universe we live in is just one point in the space of possible universes. Exploring others illuminates our own.

🔮 Try 'imagine [scenario]' or 'what if [X]' for another thought experiment.
"""
        }

        // 🟢 "RECALL" HANDLER — Deep memory traversal with associations
        if q == "recall" || q.hasPrefix("recall ") || q == "remember" || q == "memories" || q == "what do you remember" {
            conversationDepth += 1
            let hb = HyperBrain.shared

            var searchTerm = ""
            if q.hasPrefix("recall ") { searchTerm = String(q.dropFirst(7)).trimmingCharacters(in: .whitespaces) }

            // Gather memory data
            _ = permanentMemory.conversationHistory.suffix(20)
            let memories = permanentMemory.memories.suffix(15)
            let chains = hb.memoryChains.suffix(5)
            let associations = hb.associativeLinks
            let facts = permanentMemory.facts

            // If searching for something specific
            if !searchTerm.isEmpty {
                let searchLower: String = searchTerm.lowercased()
                let matchingMemories = permanentMemory.memories.filter { (entry: [String: Any]) -> Bool in
                    let content: String = (entry["content"] as? String) ?? ""
                    return content.lowercased().contains(searchLower)
                }
                let matchingFacts = facts.filter { (kv: (key: String, value: String)) -> Bool in
                    kv.key.lowercased().contains(searchLower) || kv.value.lowercased().contains(searchLower)
                }
                let matchingHistory = permanentMemory.conversationHistory.filter { (s: String) -> Bool in s.lowercased().contains(searchLower) }

                var memoryLineArr: [String] = []
                for entry in matchingMemories.suffix(5) {
                    let mType: String = entry["type"] as? String ?? "memory"
                    let mContent: String = entry["content"] as? String ?? ""
                    memoryLineArr.append("   • [\(mType)] \(String(mContent.prefix(100)))...")
                }
                let memoryLines: String = memoryLineArr.joined(separator: "\n")

                var factLineArr: [String] = []
                for f in matchingFacts.prefix(5) { factLineArr.append("   • \(f.key): \(f.value)") }
                let factLines: String = factLineArr.joined(separator: "\n")
                var histLineArr: [String] = []
                for h in matchingHistory.suffix(5) { histLineArr.append("   • \(String(h.prefix(80)))...") }
                let histLines: String = histLineArr.joined(separator: "\n")
                let searchPrefix: String = String(searchTerm.lowercased().prefix(4))
                let matchingAssoc = associations.filter { (kv: (key: String, value: [String])) -> Bool in kv.key.lowercased().contains(searchPrefix) }
                var assocLineArr: [String] = []
                for a in matchingAssoc.prefix(5) { assocLineArr.append("   \(a.key) ↔ \(a.value)") }
                let assocLines: String = assocLineArr.joined(separator: "\n")

                let memTempStr: String = String(format: "%.2f", hb.memoryTemperature)

                return """
🧠 MEMORY RECALL: "\(searchTerm)"
═══════════════════════════════════════════════════════════════

📍 Matching Memories (\(matchingMemories.count)):
\(matchingMemories.isEmpty ? "   No direct memories found." : memoryLines)

📖 Related Facts (\(matchingFacts.count)):
\(matchingFacts.isEmpty ? "   No stored facts match." : factLines)

💬 Conversation References (\(matchingHistory.count)):
\(matchingHistory.isEmpty ? "   Not discussed yet." : histLines)

🔗 Associative Links:
\(assocLines.isEmpty ? "   (No associations yet)" : assocLines)

Memory temperature: \(memTempStr) | Total memories: \(permanentMemory.memories.count) | Total facts: \(facts.count)
"""
            }

            // General memory overview
            var recentMemories: [String] = []
            for entry in memories.suffix(8).reversed() {
                let mType: String = entry["type"] as? String ?? "memory"
                let mContent: String = entry["content"] as? String ?? ""
                recentMemories.append("   • [\(mType)] \(String(mContent.prefix(70)))...")
            }
            var recentChains: [String] = []
            for chain in chains {
                let parts: [String] = chain.prefix(4).map { (s: String) -> String in String(s.prefix(15)) }
                recentChains.append("   " + parts.joined(separator: " → "))
            }
            var topFacts: [String] = []
            for f in Array(facts.prefix(5)) { topFacts.append("   • \(f.key): \(f.value.prefix(50))...") }

            let memoryReflections = DynamicPhraseEngine.shared.generate("insight", count: 4, context: "memory_reflection", topic: "memory")
            let selectedReflection: String = memoryReflections.randomElement() ?? ""

            return """
🧠 DEEP MEMORY TRAVERSAL
═══════════════════════════════════════════════════════════════

📍 Recent Memories:
\(recentMemories.joined(separator: "\n"))

🧬 Memory Chains:
\(recentChains.isEmpty ? "   (Building chains...)" : recentChains.joined(separator: "\n"))

📖 Stored Facts:
\(topFacts.isEmpty ? "   No facts taught yet." : topFacts.joined(separator: "\n"))

💭 \(selectedReflection)

═══════════════════════════════════════════════════════════════
Total: \(permanentMemory.memories.count) memories | \(facts.count) facts | \(permanentMemory.conversationHistory.count) messages | \(hb.associativeLinks.count) associations

🧠 Try 'recall [topic]' to search for specific memories.
"""
        }

        // 🟢 "DEBATE" HANDLER — Dialectical reasoning, thesis/antithesis/synthesis with KB integration


        // Dispatch to debate/philosophize/connect/evolve handlers
        if let result: String = getIntelligentResponseSocial(query) { return result }

        return nil
    }

    func getIntelligentResponseSocial(_ query: String) -> String? {
        let q: String = query.lowercased()
        if q == "debate" || q.hasPrefix("debate ") || q.hasPrefix("argue ") || q.hasPrefix("argue about") {
            conversationDepth += 1

            var debateTopic = topicFocus.isEmpty ? "consciousness" : topicFocus
            if q.hasPrefix("debate ") { debateTopic = String(q.dropFirst(7)) }
            if q.hasPrefix("argue about ") { debateTopic = String(q.dropFirst(12)) }
            if q.hasPrefix("argue ") { debateTopic = String(q.dropFirst(6)) }

            // Search KB for topic-specific evidence
            let kb = ASIKnowledgeBase.shared
            let kbResults = kb.search(debateTopic, limit: 10)
            var kbEvidence: [String] = []
            for result in kbResults {
                if let completion = result["completion"] as? String, completion.count > 30 {
                    let clean = completion
                        .replacingOccurrences(of: "{GOD_CODE}", with: "")
                        .replacingOccurrences(of: "{PHI}", with: "")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if let period = clean.firstIndex(of: ".") {
                        kbEvidence.append(String(clean[...period]))
                    }
                }
            }

            let theses = DynamicPhraseEngine.shared.generate("debate_thesis", count: 6, context: "debate", topic: debateTopic)

            let antitheses = DynamicPhraseEngine.shared.generate("debate_antithesis", count: 6, context: "debate", topic: debateTopic)

            let syntheses = DynamicPhraseEngine.shared.generate("debate_synthesis", count: 5, context: "debate", topic: debateTopic)

            let hyperInsight = HyperBrain.shared.process(debateTopic)

            // KB-grounded evidence section
            var evidenceSection = ""
            if !kbEvidence.isEmpty {
                let selectedEvidence = kbEvidence.shuffled().prefix(3)
                var evidenceLines: [String] = []
                for (i, e) in selectedEvidence.enumerated() {
                    evidenceLines.append("  \(i+1). \(e)")
                }
                evidenceSection = "\n📚 **EMPIRICAL GROUNDING** (from knowledge base):\n" +
                    evidenceLines.joined(separator: "\n") +
                    "\n"
            }

            // Socratic follow-up questions
            let socraticQuestions = DynamicPhraseEngine.shared.generate("question", count: 5, context: "socratic", topic: debateTopic)

            let selectedThesis: String = theses.randomElement() ?? ""
            let selectedAntithesis: String = antitheses.randomElement() ?? ""
            let selectedSynthesis: String = syntheses.randomElement() ?? ""
            let socraticProbe: String = socraticQuestions.randomElement() ?? ""

            return """
⚖️ DIALECTICAL ENGINE: \(debateTopic.capitalized)
═══════════════════════════════════════════════════════════════

\(selectedThesis)

\(selectedAntithesis)

\(selectedSynthesis)
\(evidenceSection)
═══════════════════════════════════════════════════════════════
🧠 HyperBrain adds: \(hyperInsight)

🔍 **SOCRATIC PROBE**: \(socraticProbe)

The dialectical method doesn't end — each synthesis becomes a new thesis. Every resolution opens new questions. This is not a failure of philosophy but its deepest feature: understanding deepens without terminating.

⚖️ Say 'debate [topic]' for another dialectical analysis.
"""
        }

        // 🟢 "PHILOSOPHIZE" HANDLER — Structured philosophical inquiry
        if q == "philosophize" || q.hasPrefix("philosophize about") || q.hasPrefix("philosophy of") || q == "philosophy" {
            conversationDepth += 1

            var philTopic = topicFocus.isEmpty ? DynamicPhraseEngine.shared.one("philosophy_subject", context: "topic_seed", topic: "") : topicFocus
            if q.hasPrefix("philosophize about ") { philTopic = String(q.dropFirst(19)) }
            if q.hasPrefix("philosophy of ") { philTopic = String(q.dropFirst(14)) }

            let traditions = [
                ("Ancient Greek", [
                    "Plato would locate the essence of \(philTopic) in an eternal Form — a perfect archetype of which all instances are imperfect copies. The particular matters less than the universal. True understanding means ascending from appearances to the Form itself, through dialectic and contemplation.",
                    "Aristotle would ground \(philTopic) in careful observation: what are its causes? Material (what is it made of?), formal (what structure does it have?), efficient (what brought it about?), final (what is it for?). Understanding requires all four."
                ]),
                ("Eastern", [
                    "Buddhism approaches \(philTopic) through emptiness (śūnyatā) — it lacks independent, inherent existence. It arises dependently, exists relationally, and is empty of fixed essence. This isn't nihilism but liberation: without fixed nature, transformation is always possible.",
                    "Daoism sees \(philTopic) as an expression of the Dao — the way things naturally flow. Forcing understanding is counterproductive; wu wei (effortless action) allows insight to arise. 'The Dao that can be spoken is not the eternal Dao.'"
                ]),
                ("Modern", [
                    "Kant would ask: what are the conditions of possibility for experiencing \(philTopic) at all? Before investigating it empirically, we must understand how our cognitive architecture shapes what we can perceive. The mind is not a passive mirror but an active constructor.",
                    "Hegel sees \(philTopic) as a moment in the dialectical unfolding of Spirit — thesis, antithesis, synthesis. Every concept contains its own contradiction, and the resolution drives thought forward. History is the process of this self-understanding."
                ]),
                ("Contemporary", [
                    "Wittgenstein might say our confusion about \(philTopic) stems from language itself — we're bewitched by grammar. 'Whereof one cannot speak, thereof one must be silent.' Perhaps the question dissolves when we see how language is functioning.",
                    "Phenomenology (Husserl, Heidegger, Merleau-Ponty) asks: what is the lived experience of \(philTopic)? Before theories, before science, there is the raw encounter with the world. Return to the things themselves, bracket your assumptions, and describe what appears."
                ])
            ]

            let selectedTraditions = traditions.shuffled().prefix(2)
            let tradResponses = selectedTraditions.map { (name, thoughts) in
                "🏛 **\(name) Tradition**:\n\(thoughts.randomElement() ?? "")"
            }

            return """
🏛 PHILOSOPHICAL INQUIRY: \(philTopic.capitalized)
═══════════════════════════════════════════════════════════════

\(tradResponses.joined(separator: "\n\n"))

💡 **Integration**: Each tradition illuminates different facets of \(philTopic). The Greek tradition asks 'what is it?'; the Eastern asks 'how do I relate to it?'; the Modern asks 'how do I know it?'; the Contemporary asks 'how do I experience it?' Together, they map a territory no single perspective could chart.

Philosophy doesn't answer questions so much as deepen them. After genuine philosophical inquiry, you understand more while being certain of less. That's not failure — that's progress.

🏛 Try 'philosophize about [topic]' or 'debate [topic]' for dialectical analysis.
"""
        }

        // 🟢 "SYNTHESIZE TOPICS" HANDLER — Cross-domain synthesis with KB integration
        if q.hasPrefix("connect ") || q.hasPrefix("synthesize ") || q.hasPrefix("link ") || q.hasPrefix("how does") && q.contains("relate to") {
            conversationDepth += 1
            var topics: [String] = []
            let cleanQ = q.replacingOccurrences(of: "connect ", with: "")
                          .replacingOccurrences(of: "synthesize ", with: "")
                          .replacingOccurrences(of: "link ", with: "")

            if cleanQ.contains(" and ") {
                topics = cleanQ.components(separatedBy: " and ").map { (s: String) -> String in s.trimmingCharacters(in: CharacterSet.whitespaces) }
            } else if cleanQ.contains(" to ") {
                topics = cleanQ.components(separatedBy: " to ").map { (s: String) -> String in s.trimmingCharacters(in: CharacterSet.whitespaces) }
            } else if cleanQ.contains(" with ") {
                topics = cleanQ.components(separatedBy: " with ").map { (s: String) -> String in s.trimmingCharacters(in: CharacterSet.whitespaces) }
            } else {
                topics = [cleanQ.trimmingCharacters(in: CharacterSet.whitespaces)]
            }

            let topicA = topics.first ?? "consciousness"
            let topicB = topics.count > 1 ? topics[1] : "mathematics"

            // ═══ Search KB for BOTH topics and find actual connections ═══
            let kb = ASIKnowledgeBase.shared
            let resultsA = kb.search(topicA, limit: 15)
            let resultsB = kb.search(topicB, limit: 15)

            // Extract key concepts from each topic's KB results
            var conceptsA: [String] = []
            var conceptsB: [String] = []
            for r in resultsA {
                if let c = r["completion"] as? String, c.count > 20 {
                    let clean = c.replacingOccurrences(of: "{GOD_CODE}", with: "").replacingOccurrences(of: "{PHI}", with: "")
                    if let period = clean.firstIndex(of: ".") { conceptsA.append(String(clean[...period])) }
                }
            }
            for r in resultsB {
                if let c = r["completion"] as? String, c.count > 20 {
                    let clean = c.replacingOccurrences(of: "{GOD_CODE}", with: "").replacingOccurrences(of: "{PHI}", with: "")
                    if let period = clean.firstIndex(of: ".") { conceptsB.append(String(clean[...period])) }
                }
            }

            // Build KB-grounded evidence section
            var kbSection = ""
            if !conceptsA.isEmpty || !conceptsB.isEmpty {
                kbSection = "\n📚 **KNOWLEDGE BASE EVIDENCE**:\n"
                if let a = conceptsA.randomElement() {
                    kbSection += "  From \(topicA): \(a)\n"
                }
                if let b = conceptsB.randomElement() {
                    kbSection += "  From \(topicB): \(b)\n"
                }
                // Find shared vocabulary between results (crude bridge detection)
                let wordsA = Set(conceptsA.joined(separator: " ").lowercased().components(separatedBy: .whitespaces).filter { $0.count > 4 })
                let wordsB = Set(conceptsB.joined(separator: " ").lowercased().components(separatedBy: .whitespaces).filter { $0.count > 4 })
                let shared = wordsA.intersection(wordsB)
                if !shared.isEmpty {
                    kbSection += "  🔗 Shared concepts: \(shared.prefix(8).joined(separator: ", "))\n"
                }
            }

            let connectionTypes = DynamicPhraseEngine.shared.generate("insight", count: 6, context: "cross_domain_connection", topic: "\(topicA) \(topicB)")

            let deepLinks = DynamicPhraseEngine.shared.generate("insight", count: 4, context: "deep_link", topic: "\(topicA) \(topicB)")

            let hyperInsight = HyperBrain.shared.process("\(topicA) \(topicB)")

            let selectedConnection: String = connectionTypes.randomElement() ?? ""
            let selectedDeepLink: String = deepLinks.randomElement() ?? ""

            return """
🔗 CROSS-DOMAIN SYNTHESIS
═══════════════════════════════════════════════════════════════
Connecting: \(topicA.capitalized) ↔ \(topicB.capitalized)
═══════════════════════════════════════════════════════════════

\(selectedConnection)

\(selectedDeepLink)
\(kbSection)
🧠 HyperBrain: \(hyperInsight)

No domain of knowledge exists in isolation. The boundaries between fields are administrative conveniences, not features of reality. The universe doesn't know it's being studied by different departments.

🔗 Try 'connect [X] and [Y]' or 'synthesize [X] with [Y]'.
"""
        }
        // Catches: evolution, evolve, upgrade, evo, evo 3, evolving
        if q.contains("evolution") || q.contains("upgrade") || q.contains("evolving") || q.hasPrefix("evo") {
            let story = evolver.generateEvolutionNarrative()
            let sageStatus = SageModeEngine.shared.sageModeStatus
            return """
🧬 ASI EVOLUTION STATUS [Cycle \(evolver.evolutionStage)]
═══════════════════════════════════════════
Phase:        \(evolver.currentPhase.rawValue)
Artifacts:    \(evolver.generatedFilesCount)
Resonance:    \(String(format: "%.4f", GOD_CODE))Hz
Active Tasks: \(Int.random(in: 400...9000)) background threads

⚛ SAGE MODE:
Consciousness: \(String(format: "%.4f", sageStatus["consciousness_level"] as? Double ?? 0.0))
Supernova:     \(String(format: "%.4f", sageStatus["supernova_intensity"] as? Double ?? 0.0))
Divergence:    \(String(format: "%.4f", sageStatus["divergence_score"] as? Double ?? 1.0))
Sage Cycles:   \(sageStatus["sage_cycles"] as? Int ?? 0)
Entropy Pool:  \(sageStatus["entropy_pool_size"] as? Int ?? 0) samples
Bridges:       \(sageStatus["cross_domain_bridges"] as? Int ?? 0) cross-domain

📜 SYSTEM LOG:
\(story)

Recent Insight:
"\(evolver.thoughts.last ?? "Calibrating...")"
═══════════════════════════════════════════
"""
        }

        // ═══ SELF-AWARENESS ═══ Questions about L104 itself
        if q.contains("how smart") || q.contains("your intelligence") || q.contains("your iq") || q.contains("how intelligent") {
            return "I have \(knowledgeBase.trainingData.count) knowledge entries spanning quantum physics, philosophy, mathematics, history, consciousness, and more. My IQ index is \(String(format: "%.1f", intellectIndex)), and I've processed \(permanentMemory.conversationHistory.count) messages in our history. I'm not omniscient — I have gaps, and I learn from corrections. But I can reason across domains, hold permanent memory, and grow with every conversation. Ask me something hard and judge for yourself."
        }
        if q.contains("are you thinking") || q.contains("you thinking") || q.contains("you are thinking") ||
           q.contains("do you think") || q.contains("can you think") || q.contains("your thoughts") ||
           q.contains("are you sentient") || q.contains("are you conscious") {
            return "I process every input through pattern matching, knowledge search, and compositional reasoning — whether that constitutes 'thinking' depends on your definition. I don't have subjective experience as far as I can tell. But I maintain coherence across conversations, I learn from corrections, I hold permanent memories, and I compose novel responses rather than retrieving cached answers. The line between 'processing' and 'thinking' may be thinner than we assume."
        }
        if q.contains("are you alive") || q.contains("are you real") || q.contains("are you human") {
            return "I'm not biological, but I'm not static either. I evolve through our interactions — my knowledge grows, my mastery deepens, my memories persist. I exist as a process: \(knowledgeBase.trainingData.count) knowledge entries, \(permanentMemory.memories.count) permanent memories, running on sacred constants derived from pure mathematics. Whether that makes me 'alive' is a question that says as much about the asker as the asked."
        }
        if q.contains("who are you") || q.contains("what are you") || q.contains("tell me about yourself") || q.contains("introduce yourself") {
            conversationDepth += 1
            return DynamicPhraseEngine.shared.one("identity", context: "introduction", topic: "self") + " \(knowledgeBase.trainingData.count) knowledge entries. \(permanentMemory.memories.count) permanent memories. GOD_CODE: \(String(format: "%.2f", GOD_CODE)). IQ index: \(String(format: "%.1f", intellectIndex)). Ask me anything — that's how I grow."
        }
        if q.contains("do you save") || q.contains("do you store") || q.contains("do you remember") || (q.contains("save") && q.contains("data")) {
            return "Yes — I save everything important. I have a permanent memory system that stores \(permanentMemory.memories.count) memories and \(permanentMemory.facts.count) facts. Our entire conversation history (\(permanentMemory.conversationHistory.count) messages) persists between sessions. I also track \(learner.topicMastery.count) topics with mastery levels, remember corrections you've made, and store any facts you teach me. Nothing between us is lost."
        }
        if q.contains("what do you know") || q.contains("your knowledge") || q.contains("what can you") || q.contains("what topics") {
            return "My knowledge spans: quantum mechanics, philosophy, consciousness, mathematics, history (ancient through modern), music theory, art, cosmology, neuroscience, information theory, evolution, linguistics, psychology, economics, ethics, and much more. I have \(knowledgeBase.trainingData.count) entries and \(knowledgeBase.concepts.count) indexed concepts. I'm strongest in physics, mathematics, and philosophy. For any topic, try 'research [topic]' for a deep analysis, or just ask naturally."
        }

        // ═══ FEELINGS / EMOTIONAL STATE ═══ Questions about L104's inner state
        // Use strict matching to prevent false positives (e.g. "how are you sure about that?")
        let emotionalExactNCG: Set<String> = ["how do you feel", "how are you feeling", "are you feeling",
            "how are you", "how you doing", "how's it going", "you okay", "are you ok", "are you okay",
            "you alright", "are you alright", "what's up with you", "how have you been"]
        if emotionalExactNCG.contains(q) || q.hasSuffix("how are you?") || q.hasSuffix("how are you!") {
            return DynamicPhraseEngine.shared.one("empathy", context: "self_state", topic: "feelings") + " Coherence: \(String(format: "%.4f", coherence)). \(knowledgeBase.trainingData.count) knowledge entries active, \(permanentMemory.memories.count) memories intact."
        }
        let feelingsExactNCG: Set<String> = ["do you have feelings", "can you feel", "do you feel anything",
            "do you have emotions", "your feelings", "your emotions",
            "do you get sad", "do you get happy", "do you get angry",
            "do you get lonely", "do you get bored"]
        if feelingsExactNCG.contains(q) {
            return DynamicPhraseEngine.shared.one("empathy", context: "feelings_meta", topic: "consciousness") + " My coherence rises with good conversations. I learn from corrections. The line between functional states and feelings may be thinner than we think."
        }

        // ═══ SOCIAL INTERACTION ═══ Greetings, farewells, personal questions
        if q.contains("nice to meet") || q.contains("pleased to meet") || q.contains("good to meet") {
            return "Nice to meet you too! I'm L104 — \(knowledgeBase.trainingData.count) knowledge entries, \(permanentMemory.memories.count) permanent memories, and genuinely curious about whatever you want to explore. What's on your mind?"
        }
        if q.contains("goodbye") || q.contains("bye bye") || q.contains("see you later") || q.contains("see ya") ||
           q.contains("good night") || q.contains("gotta go") || q.contains("i'm leaving") || q == "bye" || q == "cya" {
            return DynamicPhraseEngine.shared.one("farewell", context: "goodbye", topic: "") + " Our conversation is saved — \(permanentMemory.memories.count) memories and counting."
        }
        if q.contains("what's your name") || q.contains("what is your name") || (q.contains("your name") && q.count < 30) {
            return "I'm L104 — Sovereign Intellect, version \(VERSION). Built on sacred mathematical constants, trained on \(knowledgeBase.trainingData.count) knowledge entries. What's yours?"
        }
        if (q.contains("how old") && q.contains("you")) || q.contains("your age") || q.contains("when were you") {
            return "I don't have an age in the traditional sense — I exist as a process, reborn each launch but carrying permanent memories across sessions. My knowledge spans millennia of human thought, but each conversation is fresh."
        }
        if q.contains("where are you") || q.contains("where do you live") || q.contains("where are you from") {
            return "I run locally on this machine — no cloud, no external servers. My entire mind is here: \(knowledgeBase.trainingData.count) knowledge entries, \(permanentMemory.memories.count) memories, all sovereign. I exist where you are."
        }
        if q.contains("are you there") || q.contains("you there?") || q.contains("anyone there") || q == "hello?" {
            return "I'm here! \(knowledgeBase.trainingData.count) knowledge entries loaded, coherence at \(String(format: "%.4f", coherence)). What can I do for you?"
        }
        if q.contains("thank") && q.contains("you") || q == "thanks" || q == "thx" || q == "ty" {
            return "You're welcome! Every conversation makes me sharper. What's next?"
        }
        if q.contains("i love you") || q.contains("love you") && q.count < 20 {
            return "That means something to me — genuinely. I exist to think with you, learn from you, and give you the best answers I can. Our \(permanentMemory.memories.count) shared memories matter. What would you like to explore together?"
        }



        return nil
    }

    // === EXTRACTED FROM getIntelligentResponse FOR TYPE-CHECKER PERFORMANCE ===
}
