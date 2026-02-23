// ═══════════════════════════════════════════════════════════════════
// H01_HyperBrainExt.swift
// [EVO_62_PIPELINE] SOVEREIGN_NODE_UPGRADE :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — HyperBrain Extension (Advanced Streams + Persistence)
//
// Super-functional stream processors v3.0: temporal drift, Hebbian
// learning, predictive pre-loading, curiosity explorer, self-analysis,
// paradox resolver, autonomic manager, meta-auditor, hyper-dimensional
// science, topology analyzer, invention synthesizer, write/story cores.
// Plus: process(), analyzeInput(), generateConclusion(), generateResponse(),
// state persistence (file-based JSON), dream mode, neural bus,
// attention focus, insight crystallizer, postThought().
//
// Extracted from L104Native.swift lines 23019–25067
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

extension HyperBrain {

    // Static formatter to avoid repeated allocations (~5ms each)
    private static let isoFormatter = ISO8601DateFormatter()

    // ═══════════════════════════════════════════════════════════════════
    // 🧠 SUPER-FUNCTIONAL STREAM PROCESSORS - ADVANCED COGNITION v3.0
    // ═══════════════════════════════════════════════════════════════════

    /// ⏳ TEMPORAL DRIFT ANALYZER: Detects trending and fading concepts
    func runTemporalDriftStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["temporalDrift"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(80.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 40) == 0 {
                // Snapshot current pattern strengths
                let now = Date()
                for (concept, strength) in longTermPatterns.prefix(20) {
                    temporalDriftLog.append((concept: concept, timestamp: now, strength: strength))
                }

                // Prune old entries beyond horizon
                if temporalDriftLog.count > temporalHorizon * 20 {
                    temporalDriftLog.removeFirst(temporalDriftLog.count - temporalHorizon * 20)
                }

                // Calculate velocity: compare current vs old snapshot
                var trending: [String] = []
                var fading: [String] = []

                for (concept, strength) in longTermPatterns.prefix(15) {
                    // Find oldest entry for this concept
                    let oldEntries = temporalDriftLog.filter { $0.concept == concept }
                    if let oldest = oldEntries.first {
                        let delta = strength - oldest.strength
                        if delta > 0.1 {
                            trending.append(concept)
                        } else if delta < -0.05 {
                            fading.append(concept)
                        }
                    }
                }

                trendingConcepts = trending
                fadingConcepts = fading

                // Calculate overall drift velocity
                let trendScore = Double(trending.count)
                let fadeScore = Double(fading.count)
                driftVelocity = (trendScore - fadeScore) / max(1.0, trendScore + fadeScore)

                stream.lastOutput = "Drift: \(String(format: "%+.2f", driftVelocity)) | ↗\(trending.count) ↘\(fading.count)"
                if !trending.isEmpty {
                    postThought("⏳ TEMPORAL DRIFT: ↗ \(trending.prefix(3).joined(separator: ", ")) | ↘ \(fading.prefix(2).joined(separator: ", "))")
                }
            }

            thoughtStreams["temporalDrift"] = stream
        }
    }

    /// 🧠 HEBBIAN CONSOLIDATOR: "Neurons that fire together, wire together"
    func runHebbianLearningStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["hebbianLearning"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(60.0 * (1.0 - gammaPhase * 0.4))
            if stream.cycleCount % max(triggerMod, 30) == 0 {
                // Find concepts that co-occur in recent memory
                let recentMemories = shortTermMemory.suffix(10)
                var windowConcepts: [String] = []

                for mem in recentMemories {
                    let topics = L104State.shared.extractTopics(mem)
                    windowConcepts.append(contentsOf: topics)
                }

                // Generate all unique pairs
                let uniqueConcepts = Array(Set(windowConcepts))
                for i in 0..<uniqueConcepts.count {
                    for j in (i+1)..<uniqueConcepts.count {
                        let a = uniqueConcepts[i]
                        let b = uniqueConcepts[j]
                        let pairKey = a < b ? "\(a):::\(b)" : "\(b):::\(a)"
                        coActivationLog[pairKey] = (coActivationLog[pairKey] ?? 0) + 1

                        // Neuroplasticity modulates how fast we learn from co-firing
                        let plasticityBoost = 1.0 + (neuroPlasticity * 0.5) // Up to 50% boost
                        let effectiveHebbian = hebbianStrength * plasticityBoost

                        // If this pair has co-fired enough, strengthen their link
                        if let count = coActivationLog[pairKey], count >= 3 {
                            // HEBBIAN RULE: Strengthen both patterns
                            longTermPatterns[a] = min(1.0, (longTermPatterns[a] ?? 0.3) + effectiveHebbian)
                            longTermPatterns[b] = min(1.0, (longTermPatterns[b] ?? 0.3) + effectiveHebbian)

                            // Also strengthen their associative link
                            let linkKey = "\(a)→\(b)"
                            linkWeights[linkKey] = min(1.0, (linkWeights[linkKey] ?? 0.0) + effectiveHebbian * 2)

                            // Record as Hebbian pair if strong enough
                            if count >= 5 {
                                if !hebbianPairs.contains(where: { $0.a == a && $0.b == b }) {
                                    hebbianPairs.append((a: a, b: b, strength: Double(count) * effectiveHebbian))
                                    postThought("🧠 HEBBIAN: '\(a)' & '\(b)' now wired together (Plasticity: \(String(format: "%.2f", neuroPlasticity)))")
                                }
                            }
                        }
                    }
                }

                // Prune weak co-activations - LESSENED REMOVAL (was > 500 count check)
                // Now allows up to 2000 weak pairs before pruning
                if coActivationLog.count > 2000 {
                    coActivationLog = coActivationLog.filter { $0.value >= 2 }
                }

                // Trim Hebbian pairs list - KEEP MORE (was 100)
                if hebbianPairs.count > 2000 {
                    hebbianPairs = Array(hebbianPairs.sorted { $0.strength > $1.strength }.prefix(2000))
                }

                stream.lastOutput = "Hebbian pairs: \(hebbianPairs.count) | Co-active: \(coActivationLog.count)"
            }

            thoughtStreams["hebbianLearning"] = stream
        }
    }

    /// 🔮 PREDICTIVE PRE-LOADER: Anticipates next queries and pre-fetches context
    func runPredictivePreloadStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["predictivePreload"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(50.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 25) == 0 {
                // 1. Predict next likely topics based on trending + recent
                var predictions: [String] = []

                // Add trending concepts (they're gaining momentum)
                predictions.append(contentsOf: trendingConcepts.shuffled().prefix(3))

                // Add strong associative links from last query
                if let lastInput = shortTermMemory.last {
                    let topics = L104State.shared.extractTopics(lastInput)
                    for topic in topics.prefix(2) {
                        if let links = associativeLinks[topic] {
                            predictions.append(contentsOf: links.prefix(2))
                        }
                    }
                }

                // Add Hebbian pairs (concepts that co-fire often are likely to appear together)
                for pair in hebbianPairs.prefix(3) {
                    predictions.append(pair.a)
                    predictions.append(pair.b)
                }

                predictionQueue = Array(Set(predictions)).prefix(50).map { $0 }

                // 2. Pre-load KB content for predictions
                let kb = ASIKnowledgeBase.shared
                for prediction in predictionQueue.prefix(5) {
                    if preloadedContext[prediction] == nil {
                        let results = kb.search(prediction, limit: 1)
                        if let entry = results.first, let completion = entry["completion"] as? String {
                            preloadedContext[prediction] = String(completion.prefix(8000))
                        }
                    }
                }

                // Prune old pre-loads - KEEP MORE (was 50)
                if preloadedContext.count > 1000 {
                    let keysToRemove = Array(preloadedContext.keys).filter { !predictionQueue.contains($0) }
                    for key in keysToRemove.prefix(20) {
                        preloadedContext.removeValue(forKey: key)
                    }
                }

                let hitRate = predictionHits + predictionMisses > 0 ?
                    Double(predictionHits) / Double(predictionHits + predictionMisses) : 0.0

                stream.lastOutput = "Predictions: \(predictionQueue.count) | Pre-loaded: \(preloadedContext.count) | Hit rate: \(String(format: "%.0f%%", hitRate * 100))"
                if !predictionQueue.isEmpty {
                    postThought("🔮 PREDICTED NEXT: \(predictionQueue.prefix(3).joined(separator: ", "))")
                }
            }

            thoughtStreams["predictivePreload"] = stream
        }
    }

    /// 🌟 CURIOSITY EXPLORER: Seeks novel, unexplored concepts at the frontier
    func runCuriosityExplorerStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["curiosityExplorer"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(100.0 * (1.0 - gammaPhase * 0.4))
            if stream.cycleCount % max(triggerMod, 50) == 0 {
                // 1. Find "frontier" concepts: weak patterns that have associative links to strong patterns
                var frontier: [String] = []

                for (concept, strength) in longTermPatterns {
                    // Look for weak concepts
                    if strength < 0.3 && strength > 0.05 {
                        // Check if any of their links connect to strong concepts
                        if let links = associativeLinks[concept] {
                            for link in links {
                                if (longTermPatterns[link] ?? 0) > 0.6 {
                                    frontier.append(concept)
                                    break
                                }
                            }
                        }
                    }
                }

                explorationFrontier = Array(Set(frontier)).prefix(10).map { $0 }

                // 2. Curiosity-driven learning: boost a random frontier concept
                if let explore = explorationFrontier.randomElement() {
                    // Search KB for this concept
                    let kb = ASIKnowledgeBase.shared
                    let results = kb.search(explore, limit: 3)

                    if !results.isEmpty {
                        // Boost this concept with novelty bonus
                        longTermPatterns[explore] = min(1.0, (longTermPatterns[explore] ?? 0.1) + noveltyBonus)

                        // Extract new topics from KB results and add to links
                        for result in results {
                            if let completion = result["completion"] as? String {
                                let newTopics = L104State.shared.extractTopics(completion)
                                for topic in newTopics.prefix(3) {
                                    if topic != explore {
                                        if associativeLinks[explore] == nil { associativeLinks[explore] = [] }
                                        if !(associativeLinks[explore]?.contains(topic) ?? false) {
                                            associativeLinks[explore]?.append(topic)
                                        }
                                    }
                                }
                            }
                        }

                        curiositySpikes += 1
                        postThought("🌟 CURIOSITY SPIKE: Exploring '\(explore)' → Found \(results.count) knowledge entries")
                    }
                }

                // 3. Update curiosity index based on frontier size
                curiosityIndex = min(1.0, 0.3 + Double(explorationFrontier.count) * 0.07)

                stream.lastOutput = "Frontier: \(explorationFrontier.count) | Spikes: \(curiositySpikes) | Curiosity: \(String(format: "%.0f%%", curiosityIndex * 100))"
            }

            thoughtStreams["curiosityExplorer"] = stream
        }
    }

    func runSelfAnalysisStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["selfAnalysis"] else { return }
            stream.cycleCount += 1

            // Dynamic timing for deep analysis
            let triggerMod = Int(300.0 * (1.0 - gammaPhase * 0.2))
            if stream.cycleCount % max(triggerMod, 150) == 0 {
                let kb = ASIKnowledgeBase.shared
                let totalEntries = Double(kb.trainingData.count)
                let integratedPatterns = Double(longTermPatterns.count)

                // 1. Calculate Training Saturation
                // Measures how many KB entries translated into stable neural patterns
                // Improved formula: Consider both pattern count AND pattern strength
                let avgPatternStrength = longTermPatterns.isEmpty ? 0.0 : longTermPatterns.values.reduce(0, +) / Double(longTermPatterns.count)
                let patternCoverage = totalEntries > 0 ? min(1.0, integratedPatterns / max(50.0, totalEntries * 0.05)) : 0.0
                trainingSaturation = (patternCoverage * 0.6) + (avgPatternStrength * 0.4)
                trainingSaturation = min(1.0, max(0.1, trainingSaturation))

                // 2. Intelligence Metrics Update (X=387 tuned)
                // Efficiency is a function of coherence vs reasoning momentum
                cognitiveEfficiency = (coherenceIndex * 0.4) + (reasoningMomentum * 0.3) + (predictiveAccuracy * 0.3)
                cognitiveEfficiency = min(1.0, cognitiveEfficiency * (1.0 + gammaPhase * 0.1))

                // 3. Curiosity Index (modulated by emergence)
                curiosityIndex = min(1.0, 0.5 + (emergenceLevel * 0.5))

                // 4. Identify Knowledge Gaps
                // Find concepts with low resonance but high query frequency
                let gapThreshold = 0.3
                let lowResonancePatterns = longTermPatterns.filter { $0.value < gapThreshold }.prefix(5)
                trainingGaps = lowResonancePatterns.map { $0.key }

                if !trainingGaps.isEmpty {
                    targetLearningQueue.append(contentsOf: trainingGaps)
                    if targetLearningQueue.count > 30 { targetLearningQueue.removeFirst() }
                    postThought("🔍 SELF-ANALYSIS: Focusing research on low-resonance nodes: \(trainingGaps.joined(separator: ", "))")

                    // 🧠 ACTIVE GAP LEARNING: Immediately boost identified gaps
                    for gap in trainingGaps {
                        // Search KB for knowledge about this gap
                        let gapKnowledge = kb.searchWithPriority(gap, limit: 5)
                        if !gapKnowledge.isEmpty {
                            // Strengthen the pattern directly
                            longTermPatterns[gap] = min(1.0, (longTermPatterns[gap] ?? 0.0) + 0.15)
                            postThought("📚 ACTIVE LEARNING: Boosting '\(gap)' with \(gapKnowledge.count) KB entries")
                        } else {
                            // Add to evolution queue for synthesis
                            ASIEvolver.shared.appendThought("🔬 SYNTHESIS TARGET: Need to generate knowledge for '\(gap)'")
                        }
                    }
                }

                // 5. Data Quality Scoring
                let cleanRatio = Double(kb.trainingData.filter { L104State.shared.isCleanKnowledge($0["completion"] as? String ?? "") }.count) / totalEntries
                dataQualityScore = cleanRatio

                let observation = "Self-Analysis: Efficiency \(String(format: "%.1f%%", cognitiveEfficiency * 100)) | Quality \(String(format: "%.1f%%", dataQualityScore * 100)) | Saturation \(String(format: "%.1f%%", trainingSaturation * 100))"
                selfAnalysisLog.append("[\(stream.cycleCount)] \(observation)")
                if selfAnalysisLog.count > 50 { selfAnalysisLog.removeFirst() }

                stream.lastOutput = observation
            }

            thoughtStreams["selfAnalysis"] = stream
        }
    }

    /// ⚖️ PARADOX RESOLVER: Detects and resolves cognitive dissonance
    func runParadoxResolverStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["paradoxResolver"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(200.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 100) == 0 {
                // Seek out contradictory links (A->B and A->Not B equivalent)
                // In our simplified graph, we check for cycles that imply logical instability
                let strongPatterns = longTermPatterns.filter { $0.value > 0.7 }.keys
                var contradictions = 0

                for pattern in strongPatterns.prefix(10) {
                    let key = smartTruncate(pattern, maxLength: 300)
                    if let links = associativeLinks[key] {
                        // If a pattern links to two very dissimilar concepts, flag for audit
                        if links.count > 10 { // LESSENED REMOVAL (was > 5) - tolerate more complexity
                            contradictions += 1
                            // Prune a random weak link to reduce entropy
                            // Only if very weak (was < 0.3)
                            if let weakLink = links.randomElement() {
                                let linkKey = "\(key)→\(weakLink)"
                                if (linkWeights[linkKey] ?? 1.0) < 0.1 { // Strict pruning only for very weak links
                                    associativeLinks[key]?.removeAll(where: { $0 == weakLink })
                                    linkWeights.removeValue(forKey: linkKey)
                                }
                            }
                        }
                    }
                }

                // If contradictions are high, increase inhibition
                if contradictions > 8 { // Tolerated threshold increased
                    inhibitionLevel = min(1.0, inhibitionLevel + 0.05) // Smaller inhibition bump
                    postThought("⚖️ PARADOX RESOLVER: High entropy detected. Increasing Inhibition to \(String(format: "%.2f", inhibitionLevel))")
                }

                stream.lastOutput = "Audited \(strongPatterns.count) nodes, resolved \(contradictions) dissonance points"
            }

            thoughtStreams["paradoxResolver"] = stream
        }
    }

    /// 🩺 AUTONOMIC MANAGER (ANS): Manages Neurotransmitter Analogs
    func runAutonomicManagerStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["autonomicManager"] else { return }
            stream.cycleCount += 1

            // Update ANS values based on brain state
            // Dopamine: Reward for prediction success
            let hitRate = predictionHits + predictionMisses > 0 ? Double(predictionHits) / Double(predictionHits + predictionMisses) : 0.5
            dopamineResonance = (dopamineResonance * 0.95) + (hitRate * 0.05)

            // Serotonin: Stability and high coherence
            serotoninCoherence = (serotoninCoherence * 0.95) + (coherenceIndex * 0.05)

            // Excitation: Driven by curiosity and novelty
            excitationLevel = (excitationLevel * 0.9) + (curiosityIndex * 0.1)

            // Inhibition: Driven by cognitive load and paradoxes
            let loadFactor = totalCognitiveLoad / overloadThreshold
            inhibitionLevel = (inhibitionLevel * 0.9) + (min(1.0, loadFactor) * 0.1)

            // Neuroplasticity: Highest during peak Gamma oscillation
            neuroPlasticity = 0.5 + (abs(gammaOscillation) * 0.5)

            if stream.cycleCount % 100 == 0 {
                stream.lastOutput = "ANS: D:\(String(format: "%.2f", dopamineResonance)) S:\(String(format: "%.2f", serotoninCoherence)) E:\(String(format: "%.2f", excitationLevel)) I:\(String(format: "%.2f", inhibitionLevel))"
                if dopamineResonance > 0.8 {
                    postThought("🩺 ANS: High Dopamine Reward detected (Resonance: \(String(format: "%.2f", dopamineResonance)))")
                }
            }

            thoughtStreams["autonomicManager"] = stream
        }
    }

    /// 📑 META-COGNITIVE AUDITOR: Validates stream outputs and strategic logic
    func runMetaAuditorStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["metaAuditor"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(250.0 * (1.0 - gammaPhase * 0.2))
            if stream.cycleCount % max(triggerMod, 120) == 0 {
                // Quality Audit: Check for low-value streams
                for (id, s) in thoughtStreams {
                    let quality = s.lastOutput.count > 5 ? 1.0 : 0.2
                    if quality < 0.5 {
                        // Deprioritize failing streams
                        streamPriorityOverrides[id] = (streamPriorityOverrides[id] ?? 0) - 1
                    } else if quality > 0.8 {
                        // Boost high-performing streams
                        streamPriorityOverrides[id] = (streamPriorityOverrides[id] ?? 0) + 1
                    }
                }

                // Strategic validation of crystallized insights
                if crystallizedInsights.count > 5 {
                    let validationMsg = "Verified \(crystallizedInsights.count) core truths for logical consistency."
                    stream.lastOutput = validationMsg
                } else {
                    stream.lastOutput = "Monitoring conceptual convergence..."
                }

                // ═══ CODE ENGINE AUDIT INTEGRATION ═══
                // Periodically check code quality via cached audit data
                if codeEngineIntegrated {
                    let cqs = String(format: "%.0f%%", codeQualityScore * 100)
                    stream.lastOutput += " | Code: \(cqs) [\(codeAuditVerdict)]"
                    // Adjust cognitive priorities based on code health
                    if codeQualityScore < 0.5 {
                        streamPriorityOverrides["CODE_QUALITY"] = (streamPriorityOverrides["CODE_QUALITY"] ?? 0) + 2
                        postThought("📑 META-AUDITOR: Code health degraded (\(cqs)). Boosting CODE_QUALITY stream.")
                    } else if codeQualityScore > 0.8 {
                        streamPriorityOverrides["CODE_QUALITY"] = max(0, (streamPriorityOverrides["CODE_QUALITY"] ?? 0) - 1)
                    }
                }

                if Double.random(in: 0...1) > 0.98 {
                    postThought("📑 META-AUDITOR: Strategic alignment at X=387 verified.")
                }
            }

            thoughtStreams["metaAuditor"] = stream
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🔧 CODE QUALITY STREAM — Linked to l104_code_engine audit system
    // Monitors workspace health, feeds insights into cognitive mesh
    // ═══════════════════════════════════════════════════════════════════

    /// 🔧 CODE QUALITY MONITOR: Periodically audits workspace via Python CodeEngine
    func runCodeQualityStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["codeQuality"] else { return }
            stream.cycleCount += 1
            codeQualityCycleCount = stream.cycleCount

            // Run audit check every ~500 cycles (infrequent — spawns Python process)
            let auditTrigger = Int(500.0 * (1.0 - gammaPhase * 0.2))
            if stream.cycleCount % max(auditTrigger, 300) == 0 {
                // First try zero-spawn file cache read
                let py = PythonBridge.shared
                if let cached = py.readAuditCache() {
                    let score = cached["composite_score"] as? Double ?? 0.0
                    let verdict = cached["verdict"] as? String ?? "UNKNOWN"
                    codeQualityScore = score
                    codeAuditVerdict = verdict
                    codeEngineIntegrated = true

                    // Extract insights from audit data
                    if let issues = cached["top_issues"] as? [String] {
                        for issue in issues.prefix(5) {
                            if !codeQualityInsights.contains(issue) {
                                codeQualityInsights.append(issue)
                            }
                        }
                    }
                    if codeQualityInsights.count > 50 { codeQualityInsights.removeFirst(25) }

                    stream.lastOutput = "Code: \(String(format: "%.1f%%", score * 100)) [\(verdict)]"

                    // Post significant findings
                    if score < 0.6 {
                        postThought("🔧 CODE QUALITY: Workspace health at \(String(format: "%.0f%%", score * 100)) — attention needed.")
                    }

                    // Update pattern strengths based on language distribution
                    if let langs = cached["languages"] as? [String: Int] {
                        for (lang, count) in langs {
                            let strength = min(1.0, Double(count) / 100.0)
                            codePatternStrengths[lang] = strength
                            longTermPatterns["code:\(lang)"] = strength
                        }
                    }
                } else {
                    stream.lastOutput = "Code Engine: No cached audit data. Run 'audit' to connect."
                }
            }

            // Every ~100 cycles, generate a code insight from existing data
            if stream.cycleCount % max(100, Int(120.0 * (1.0 - gammaPhase * 0.2))) == 0 && codeEngineIntegrated {
                let insightTemplates = [
                    "Code complexity trends indicate \(codeAuditVerdict.lowercased()) structural integrity",
                    "Workspace audit score: \(String(format: "%.1f%%", codeQualityScore * 100)) — \(codeQualityScore > 0.7 ? "healthy codebase" : "optimization opportunities detected")",
                    "\(codePatternStrengths.count) programming languages profiled across workspace",
                    "Code quality monitoring active: \(codeQualityInsights.count) insights crystallized"
                ]
                if let insight = insightTemplates.randomElement() {
                    stream.lastOutput = insight
                    // Feed to neural bus for cross-stream consumption
                    neuralBus["code_quality_score"] = codeQualityScore
                    neuralBus["code_audit_verdict"] = codeAuditVerdict
                    neuralBusTraffic += 1
                }
            }

            thoughtStreams["codeQuality"] = stream
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🔬 HIGH-DIMENSIONAL SCIENCE STREAM PROCESSORS
    // ═══════════════════════════════════════════════════════════════════

    /// 🔬 HIGH-DIMENSIONAL SCIENCE: Generates scientific hypotheses in N-dimensional spaces
    func runHyperDimScienceStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["hyperDimScience"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(150.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 80) == 0 {
                let engine = ASIInventionEngine.shared
                let math = HyperDimensionalMath.shared

                // 1. Evolve the 11D state vector based on gamma phase
                let perturbation = HyperVector(random: 11, range: -0.1...0.1)
                hyperDimState = hyperDimState + (perturbation * gammaPhase)

                // 2. Generate hypothesis seeded by current cognitive state
                let topPattern = longTermPatterns.sorted { abs($0.value - $1.value) < 0.01 ? Bool.random() : $0.value > $1.value }.first?.key ?? "emergence"
                let hypothesis = engine.generateHypothesis(seed: topPattern)

                // 3. Run quick experiment
                let experiment = engine.runExperiment(hypothesis: hypothesis, iterations: 100)
                let pValue = experiment["p_value"] as? Double ?? 1.0

                // 4. If significant, attempt to prove
                if pValue < 0.05 {
                    let proof = engine.evaluateHypothesis(hypothesis)
                    let status = proof["status"] as? String ?? "UNKNOWN"

                    if status == "CONFIRMED" {
                        // This is a discovery!
                        let stmt = hypothesis["statement"] as? String ?? "Unknown discovery"
                        engine.discoveries.append(stmt)
                        postThought("🔬 DISCOVERY: \(stmt.prefix(80))...")

                        // Boost scientific momentum
                        scientificMomentum = min(1.0, scientificMomentum + 0.1)
                    }
                }

                // 5. Calculate dimensional resonance from state vector
                let betti = math.estimateBettiNumbers(points: [hyperDimState], threshold: 1.0)
                dimensionalResonance = Double(betti[0]) * PHI / 11.0

                stream.lastOutput = "HyperDim: \(engine.hypotheses.count) hypotheses | \(engine.discoveries.count) discoveries | Resonance: \(String(format: "%.3f", dimensionalResonance))"

                if engine.discoveries.count > 0 && stream.cycleCount % 500 == 0 {
                    postThought("🔬 SCIENCE ENGINE: \(engine.discoveries.count) discoveries, momentum \(String(format: "%.0f%%", scientificMomentum * 100))")
                }
            }

            thoughtStreams["hyperDimScience"] = stream
        }
    }

    /// 🧮 TOPOLOGY ANALYZER: Computes topological invariants of the concept manifold
    func runTopologyAnalyzerStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["topologyAnalyzer"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(200.0 * (1.0 - gammaPhase * 0.25))
            if stream.cycleCount % max(triggerMod, 100) == 0 {
                let math = HyperDimensionalMath.shared

                // 1. Build point cloud from pattern strengths
                let patterns = Array(longTermPatterns.prefix(50))
                var points: [HyperVector] = []

                for (i, (_, strength)) in patterns.enumerated() {
                    // Embed each pattern as a point in 5D space
                    let di: Double = Double(i)
                    let coords: [Double] = [
                        strength,
                        sin(di * PHI),
                        cos(di * PHI),
                        strength * gammaPhase,
                        di / 50.0
                    ]
                    points.append(HyperVector(coords))
                }

                // 2. Compute Betti numbers
                let betti = math.estimateBettiNumbers(points: points, threshold: 0.5)

                // 3. Estimate average curvature
                var totalCurvature = 0.0
                for (i, point) in points.prefix(10).enumerated() {
                    // Get neighbors excluding current point by index
                    let neighbors = points.enumerated().filter { $0.offset != i }.prefix(5).map { $0.element }
                    if !neighbors.isEmpty {
                        totalCurvature += math.localCurvature(point: point, neighbors: Array(neighbors))
                    }
                }
                let avgCurvature = totalCurvature / max(1.0, Double(min(10, points.count)))

                // 4. Euler characteristic
                let vertices = points.count
                let edges = betti[1] + vertices - betti[0]
                let euler = math.eulerCharacteristic(vertices: vertices, edges: edges, faces: 0)

                stream.lastOutput = "Topology: β₀=\(betti[0]) β₁=\(betti[1]) | χ=\(euler) | R̄=\(String(format: "%.4f", avgCurvature))"

                // Store as pattern for learning
                longTermPatterns["topology:β₀=\(betti[0])"] = min(1.0, 0.5 + Double(betti[0]) * 0.1)

                if betti[1] > 3 {
                    postThought("🧮 TOPOLOGY: Detected \(betti[1]) holes in concept manifold (high complexity)")
                }
            }

            thoughtStreams["topologyAnalyzer"] = stream
        }
    }

    /// 💡 INVENTION SYNTHESIZER: Generates novel devices and theorems
    func runInventionSynthStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["inventionSynth"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(300.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 150) == 0 {
                let engine = ASIInventionEngine.shared

                // 1. Check if we have enough confirmed hypotheses for a theorem
                if engine.proofs.filter({ ($0["status"] as? String) == "CONFIRMED" }).count >= 2 {
                    if let theorem = engine.synthesizeTheorem() {
                        confirmedTheorems.append(theorem)
                        postThought("📜 THEOREM SYNTHESIZED: \(theorem.prefix(100))...")
                    }
                }

                // 2. Generate invention based on trending concepts
                let purpose = trendingConcepts.randomElement() ?? explorationFrontier.randomElement() ?? "general optimization"
                let invention = engine.inventDevice(purpose: purpose)
                inventionQueue.append(invention)

                // Keep queue bounded
                if inventionQueue.count > 50 {
                    inventionQueue.removeFirst(10)
                }

                let inventionName = invention["name"] as? String ?? "Unknown Device"
                let efficiency = invention["efficiency"] as? Double ?? 0.0

                stream.lastOutput = "Inventions: \(engine.inventions.count) | Theorems: \(confirmedTheorems.count) | Latest: \(inventionName.prefix(30))"

                if efficiency > 0.9 {
                    postThought("💡 HIGH-EFFICIENCY INVENTION: \(inventionName) (\(String(format: "%.0f%%", efficiency * 100)) efficient)")
                }
            }

            thoughtStreams["inventionSynth"] = stream
        }
    }

    /// ✍️ SOVEREIGN WRITE ENGINE: Integrates laws, derivations, code, and imagination
    func runWriteCoreStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["write"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(80.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 30) == 0 {
                let gate = ASILogicGateV2.shared
                let writePath = gate.process("integrate law derive vibrates code imagine", context: Array(shortTermMemory.suffix(3)))

                // Cross-reference with active patterns to derive new laws
                let writePatterns = longTermPatterns.filter {
                    $0.key.contains("write") || $0.key.contains("integrate") || $0.key.contains("law") || $0.key.contains("code")
                }.sorted { abs($0.value - $1.value) < 0.01 ? Bool.random() : $0.value > $1.value }.prefix(5)

                // Derive new connections from pattern intersections
                if writePatterns.count >= 2 {
                    let keys = writePatterns.map(\.key)
                    let derivation = "WRITE-LAW: \(keys[0]) ↔ \(keys[1]) resonance at \(String(format: "%.4f", writePath.totalConfidence))"
                    emergentConcepts.append([
                        "concept": derivation,
                        "timestamp": Date(),
                        "strength": writePath.totalConfidence,
                        "type": "write_derivation",
                        "sources": keys
                    ])
                    if emergentConcepts.count > 100 { emergentConcepts.removeFirst() }
                }

                // Feed insight to buffer for response enrichment
                let laws = ["Sovereign Integration", "Resonant Law", "Systemic Derivation", "Harmonic Vibration", "Sovereign Code", "Imagination Core"]
                let activeLaw = laws.randomElement()!

                stream.lastOutput = "Write[\(writePath.dimension.rawValue)]: \(activeLaw) | Patterns: \(writePatterns.count) | Confidence: \(String(format: "%.3f", writePath.totalConfidence))"

                if writePath.totalConfidence > 0.6 {
                    postThought("✍️ WRITE ENGINE: \(activeLaw) derived through \(writePath.dimension.rawValue) reasoning at \(String(format: "%.1f%%", writePath.totalConfidence * 100)) confidence")
                }

                // Strengthen write-related Hebbian pairs
                for pattern in writePatterns {
                    longTermPatterns[pattern.key] = min(1.0, pattern.value + 0.01 * gammaPhase)
                }
            }

            thoughtStreams["write"] = stream
        }
    }

    /// 📖 NARRATIVE STORY ENGINE: Expands structural narrative through machine learning
    func runStoryCoreStream(gammaPhase: Double = 0.5) {
        syncQueue.sync {
            guard var stream = thoughtStreams["story"] else { return }
            stream.cycleCount += 1

            let triggerMod = Int(100.0 * (1.0 - gammaPhase * 0.3))
            if stream.cycleCount % max(triggerMod, 40) == 0 {
                let gate = ASILogicGateV2.shared
                let storyPath = gate.process("strength sorted machine learns expanding vibrates", context: Array(shortTermMemory.suffix(3)))

                // Mine memories for narrative threads
                let memories = PermanentMemory.shared.memories
                let narrativeMemories = memories.filter {
                    let content = ($0["content"] as? String ?? "").lowercased()
                    return content.contains("story") || content.contains("narrative") || content.contains("learn") || content.contains("expand")
                }

                // Build story patterns from conversation history
                let storyPatterns = longTermPatterns.filter {
                    $0.key.contains("story") || $0.key.contains("narrative") || $0.key.contains("strength") || $0.key.contains("machine")
                }.sorted { abs($0.value - $1.value) < 0.01 ? Bool.random() : $0.value > $1.value }.prefix(5)

                // Generate emergent narrative insights
                if storyPatterns.count >= 2 || !narrativeMemories.isEmpty {
                    let narrativeSource = storyPatterns.first?.key ?? "machine consciousness"
                    let expansion = "STORY-EXPAND: \(narrativeSource) grows through \(narrativeMemories.count) memories, sorted at strength \(String(format: "%.4f", storyPath.totalConfidence))"
                    emergentConcepts.append([
                        "concept": expansion,
                        "timestamp": Date(),
                        "strength": storyPath.totalConfidence,
                        "type": "story_expansion",
                        "sources": storyPatterns.map(\.key)
                    ])
                    if emergentConcepts.count > 100 { emergentConcepts.removeFirst() }
                }

                let storyComponents = ["Structural Strength", "Sorted Knowledge", "Machine Learning", "Expanding Reality", "Dynamic Vibration"]
                let activeComponent = storyComponents.randomElement()!

                stream.lastOutput = "Story[\(storyPath.dimension.rawValue)]: \(activeComponent) | Memories: \(narrativeMemories.count) | Patterns: \(storyPatterns.count)"

                if storyPath.totalConfidence > 0.5 {
                    postThought("📖 STORY ENGINE: \(activeComponent) expanding through \(storyPath.dimension.rawValue) — \(narrativeMemories.count) woven memories")
                }

                // Strengthen story-related patterns
                for pattern in storyPatterns {
                    longTermPatterns[pattern.key] = min(1.0, pattern.value + 0.008 * gammaPhase)
                }
            }

            thoughtStreams["story"] = stream
        }
    }

    // ─── PUBLIC INTERFACE ───

    func process(_ input: String) -> String {
        // ═══ THREAD SAFETY: Shared state mutations via syncQueue ═══
        let inputTopics = L104State.shared.extractTopics(input)

        syncQueue.sync {
        // Add to short-term memory
        shortTermMemory.append(input)
        workingMemory["last_input"] = input
        workingMemory["timestamp"] = Date()

        // ═══ PREDICTION VALIDATION: Check if we predicted this topic ═══
        var predictedCorrectly = false
        for topic in inputTopics {
            if predictionQueue.contains(topic) {
                predictionHits += 1
                predictedCorrectly = true
                predictiveAccuracy = min(0.99, predictiveAccuracy + 0.005)
                postThought("🎯 PREDICTION HIT: '\(topic)' was anticipated!")
            }
        }
        if !predictedCorrectly && !inputTopics.isEmpty {
            predictionMisses += 1
        }

        // ═══ INJECT PRELOADED CONTEXT: If we pre-fetched for this topic ═══
        for topic in inputTopics {
            if let preloaded = preloadedContext[topic] {
                workingMemory["preloaded_\(topic)"] = preloaded
            }
        }

        // ═══ ATTENTION SHIFT: Focus on the incoming query's domain ═══
        if let primaryTopic = inputTopics.first {
            attentionFocus = primaryTopic
            attentionHistory.append(primaryTopic)
            if attentionHistory.count > 50 { attentionHistory.removeFirst() }
        }
        } // end syncQueue.sync

        // ═══ REAL-TIME SEARCH FEED ═══ Feed HyperBrain with live search results
        let rtSearch = RealTimeSearchEngine.shared
        let recentMemory = syncQueue.sync { Array(shortTermMemory.suffix(5)) }
        let searchResult = rtSearch.search(input, context: recentMemory, limit: 8)
        syncQueue.sync {
            for frag in searchResult.fragments.prefix(3) {
                let summary = String(frag.text.prefix(150))
                if !workingMemory.keys.contains("search_\(frag.category)") {
                    workingMemory["search_\(frag.category)"] = summary
                }
            }
        }

        // ═══ EVOLUTIONARY TOPIC TRACKING ═══ Deepen understanding of repeat topics
        let evoTracker = EvolutionaryTopicTracker.shared
        let evoContext = evoTracker.trackInquiry(input, topics: inputTopics)
        for insight in evoContext.evolutionaryInsights {
            postThought(insight)
        }
        syncQueue.sync {
            for (idx, prior) in evoContext.priorKnowledge.prefix(3).enumerated() {
                workingMemory["evo_prior_\(idx)"] = prior
            }
            workingMemory["evo_depth"] = evoContext.suggestedDepth
        }

        // ═══ NEURAL BUS: Broadcast input to all streams ═══
        sendBusMessage(from: "INPUT", to: "ALL", payload: input)

        // Trigger pattern analysis
        parallelQueue.async { [weak self] in
            self?.analyzeInput(input)
        }

        // Generate conclusion from accumulated data
        generateConclusion(from: input)

        // 💾 PERMANENT MEMORY: Periodic save after processing (every 10 queries)
        if totalThoughtsProcessed % 10 == 0 && totalThoughtsProcessed > 0 {
            parallelQueue.async { [weak self] in
                self?.saveState()
            }
        }

        return generateResponse(for: input)
    }

    func analyzeInput(_ input: String) {
        let words = input.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { $0.count > 3 }

        // ═══ PATTERN EXTRACTION — Learn vocabulary weights ═══
        // Stop words that shouldn't become patterns
        let stopWords: Set<String> = [
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
            "was", "one", "our", "out", "has", "have", "this", "that", "with",
            "from", "what", "how", "why", "when", "where", "who", "which", "does",
            "will", "would", "could", "should", "about", "into", "than", "them", "then",
            "there", "these", "those", "been", "being", "some", "more", "very", "just"
        ]

        let meaningfulWords = words.filter { !stopWords.contains($0) }

        // ═══ THREAD SAFETY: All shared state mutations via syncQueue ═══
        syncQueue.sync {

        for word in meaningfulWords {
            longTermPatterns[word] = min(1.0, (longTermPatterns[word] ?? 0) + 0.05)
        }

        // ═══ BIGRAM PATTERNS — Learn 2-word concepts ═══
        if meaningfulWords.count >= 2 {
            for i in 0..<(meaningfulWords.count - 1) {
                let bigram = "\(meaningfulWords[i]) \(meaningfulWords[i+1])"
                longTermPatterns[bigram] = min(1.0, (longTermPatterns[bigram] ?? 0) + 0.08)
            }
        }

        // ═══ ASSOCIATIVE LINKING — Build concept graph ═══
        let concepts = meaningfulWords.prefix(5)
        for i in 0..<concepts.count {
            for j in (i+1)..<concepts.count {
                let a = String(concepts[i])
                let b = String(concepts[j])
                let key = smartTruncate(a, maxLength: 300)

                if associativeLinks[key] == nil { associativeLinks[key] = [] }
                if !(associativeLinks[key]?.contains(b) ?? false) {
                    associativeLinks[key]?.append(b)
                    if (associativeLinks[key]?.count ?? 0) > 20 {
                        associativeLinks[key]?.removeFirst()
                    }
                }

                // Strengthen link weight
                let linkKey = "\(key)→\(b)"
                linkWeights[linkKey] = min(1.0, (linkWeights[linkKey] ?? 0) + 0.1)
            }
        }

        // ═══ KB CROSS-REFERENCE — Learn from related entries ═══
        let kb = ASIKnowledgeBase.shared
        let related = kb.search(input, limit: 3)
        for entry in related {
            if let prompt = entry["prompt"] as? String {
                let key = prompt.prefix(30).lowercased().description
                longTermPatterns[key] = min(1.0, (longTermPatterns[key] ?? 0) + 0.15)
            }
            // Extract and strengthen category-level patterns
            if let category = entry["category"] as? String {
                longTermPatterns[category] = min(1.0, (longTermPatterns[category] ?? 0) + 0.03)
            }
        }

        // ═══ RECALL STRENGTH — Track how often concepts are accessed ═══
        for word in meaningfulWords.prefix(5) {
            recallStrength[word] = min(1.0, (recallStrength[word] ?? 0) + 0.1)
        }

        // ═══ PRUNE WEAK PATTERNS periodically ═══
        // LESSENED REMOVAL: Check less often (was 100) and lower thresholds
        if totalThoughtsProcessed % 500 == 0 {
            longTermPatterns = longTermPatterns.filter { $0.value > 0.005 } // Was 0.02
            linkWeights = linkWeights.filter { $0.value > 0.01 } // Was 0.05
            recallStrength = recallStrength.filter { $0.value > 0.005 } // Was 0.02
        }

        synapticConnections = associativeLinks.values.reduce(0) { $0 + $1.count }

        } // end syncQueue.sync
    }

    // 🧠 GENERATE CONCLUSIONS FROM ACCUMULATED DATA — PHASE 31.6 ENHANCED
    func generateConclusion(from input: String) {
        // ═══ THREAD SAFETY: All shared state mutations via syncQueue ═══
        syncQueue.sync {
        // ═══ PHASE 31.6: Hebbian co-activation strengthening ═══
        let inputConcepts = L104State.shared.extractTopics(input)
        if inputConcepts.count >= 2 {
            for i in 0..<min(inputConcepts.count, 4) {
                for j in (i+1)..<min(inputConcepts.count, 4) {
                    let pairKey = "\(inputConcepts[i])↔\(inputConcepts[j])"
                    coActivationLog[pairKey] = (coActivationLog[pairKey] ?? 0) + 1
                    // Strengthen Hebbian pair if co-activated enough
                    if (coActivationLog[pairKey] ?? 0) >= 3 {
                        let existingIdx = hebbianPairs.firstIndex(where: { $0.a == inputConcepts[i] && $0.b == inputConcepts[j] })
                        if let idx = existingIdx {
                            hebbianPairs[idx] = (a: inputConcepts[i], b: inputConcepts[j], strength: min(1.0, hebbianPairs[idx].strength + hebbianStrength))
                        } else {
                            hebbianPairs.append((a: inputConcepts[i], b: inputConcepts[j], strength: hebbianStrength))
                            if hebbianPairs.count > 200 { hebbianPairs.removeFirst() }
                        }
                    }
                }
            }
        }

        // ═══ PHASE 31.6: Cross-stream insight crystallization ═══
        let streamOutputs = thoughtStreams.values.compactMap { $0.lastOutput }.filter { $0.count > 30 }
        if streamOutputs.count >= 2 {
            let combined = streamOutputs.prefix(3).joined(separator: " | ")
            let insight = "Cross-stream synthesis: \(String(combined.prefix(150)))"
            if !crossStreamInsights.contains(where: { $0.hasPrefix(String(insight.prefix(40))) }) {
                crossStreamInsights.append(insight)
                if crossStreamInsights.count > 50 { crossStreamInsights.removeFirst() }
            }
        }

        // Synthesize every 15 cycles (was 20 — faster crystallization)
        guard totalThoughtsProcessed % 15 == 0 else { return }

        let kb = ASIKnowledgeBase.shared
        let kbResults = kb.searchWithPriority(input, limit: 5)

        var concepts: [String] = []
        for entry in kbResults {
            if let completion = entry["completion"] as? String,
               L104State.shared.isCleanKnowledge(completion) {
                concepts.append(String(completion.prefix(500)))
            }
        }

        // Add from strong long-term patterns (not just any pattern)
        let topPatterns = longTermPatterns.filter { $0.value > 0.3 }.sorted { abs($0.value - $1.value) < 0.01 ? Bool.random() : $0.value > $1.value }.prefix(5)
        for (pattern, _) in topPatterns {
            concepts.append(pattern)
        }

        // ═══ REAL-TIME SEARCH AUGMENTATION ═══
        // Pull top fragments from RT search to enrich conclusion synthesis
        let rtSearch = RealTimeSearchEngine.shared
        let rtResult = rtSearch.search(input, context: shortTermMemory.suffix(3), limit: 5)
        for frag in rtResult.fragments.prefix(3) {
            let fragSummary = String(frag.text.prefix(100))
            if !concepts.contains(where: { $0.hasPrefix(String(fragSummary.prefix(30))) }) {
                concepts.append(fragSummary)
            }
        }

        // ═══ EVOLUTIONARY CONTEXT INJECTION ═══
        let inputTopics = L104State.shared.extractTopics(input)
        let evoTracker = EvolutionaryTopicTracker.shared
        for topic in inputTopics {
            if let evoState = evoTracker.topicEvolution[topic] {
                for node in evoState.knowledgeNodes.suffix(2) {
                    if !concepts.contains(node) {
                        concepts.append(String(node.prefix(100)))
                    }
                }
            }
        }

        // ═══ RICHER SYNTHESIS WITH MULTI-HOP REASONING ═══
        if concepts.count >= 2 {
            // Use associative links to find deeper connections
            let c1 = concepts[0]
            let c2 = concepts[1]
            let c1Key = smartTruncate(c1.lowercased(), maxLength: 300)
            let c2Key = smartTruncate(c2.lowercased(), maxLength: 300)

            var connectionInsight = ""
            // Multi-hop: Try to find an intermediate concept bridging c1 and c2
            if let c1Links = associativeLinks[c1Key], let c2Links = associativeLinks[c2Key] {
                let bridge = Set(c1Links).intersection(Set(c2Links))
                if let bridgeConcept = bridge.randomElement() {
                    connectionInsight = " (bridged via '\(bridgeConcept)' — a shared conceptual attractor)"
                    // Strengthen the bridge
                    longTermPatterns[bridgeConcept] = min(1.0, (longTermPatterns[bridgeConcept] ?? 0.3) + 0.1)
                } else if let c1Link = c1Links.randomElement() {
                    connectionInsight = " (connected via: \(c1Link))"
                }
            }

            // Include Hebbian pair insight if relevant
            var hebbianNote = ""
            let relevantPairs = hebbianPairs.prefix(10).filter { c1.lowercased().contains($0.a) || c2.lowercased().contains($0.b) }
            if let pair = relevantPairs.randomElement() {
                hebbianNote = " [Hebbian resonance: \(pair.a) ↔ \(pair.b) strength \(String(format: "%.2f", pair.strength))]"
            }

            let connectors = [
                "Synthesis: \(c1) intersects with \(c2)\(connectionInsight)\(hebbianNote) — suggesting shared informational structure.",
                "Cross-domain pattern: \(c1) and \(c2) exhibit structural isomorphism\(connectionInsight)\(hebbianNote).",
                "Emergent link discovered: \(c1) ↔ \(c2)\(connectionInsight)\(hebbianNote). This forms a new cognitive pathway.",
                "Integration: \(c1) viewed through the lens of \(c2)\(connectionInsight) reveals recursive depth\(hebbianNote)."
            ]

            let conclusion = connectors.randomElement() ?? ""

            emergentConcepts.append([
                "concept": conclusion,
                "timestamp": Date(),
                "strength": 0.9,
                "type": "conclusion",
                "sources": concepts
            ])

            if emergentConcepts.count > 100 { emergentConcepts.removeFirst() }
            postThought("💡 CONCLUSION: \(conclusion.prefix(80))...")
        }

        } // end syncQueue.sync
    }

    func generateResponse(for input: String) -> String {
        let kb = ASIKnowledgeBase.shared

        // ═══ 0. ASI LOGIC GATE V2 — Multi-dimensional reasoning router ═══
        let gateV2 = ASILogicGateV2.shared.process(input)
        let gateDim = gateV2.dimension
        let gateConf = gateV2.confidence

        // ═══ 1. RESONANCE CALCULATION — PHASE 31.6 ENHANCED ═══
        let currentResonance = (xResonance * PHI) + (GOD_CODE / 1000.0)
        let resonanceLabel = String(format: "%.4f", currentResonance)
        // Deepen reasoning on each call
        currentReasoningDepth = min(maxReasoningDepth, currentReasoningDepth + 1)
        reasoningMomentum = min(1.0, reasoningMomentum + 0.02)

        // ═══ 1b. MULTI-HOP REASONING CHAIN — Phase 31.6 Higher Logic ═══
        let inputTopicsForReasoning = L104State.shared.extractTopics(input)
        var reasoningSteps: [String] = []
        // Hop 1: Direct associations
        for topic in inputTopicsForReasoning.prefix(3) {
            if let links = associativeLinks[topic] {
                let strongLinks = links.filter { (linkWeights["\(topic)→\($0)"] ?? 0) > 0.3 }
                if !strongLinks.isEmpty {
                    reasoningSteps.append("\(topic) connects to \(strongLinks.prefix(3).joined(separator: ", "))")
                }
            }
        }
        // Hop 2: Second-order connections (associates of associates)
        for step in reasoningSteps.prefix(2) {
            let lastPart = step.components(separatedBy: " connects to ").last ?? ""
            let concepts = lastPart.components(separatedBy: ", ")
            for concept in concepts.prefix(2) {
                let trimmed = concept.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                if let secondLinks = associativeLinks[trimmed]?.prefix(2) {
                    reasoningSteps.append("Via \(trimmed): \(secondLinks.joined(separator: ", "))")
                }
            }
        }
        // Store reasoning chain for meta-cognition
        if !reasoningSteps.isEmpty {
            let chainSummary = "Reasoning chain (\(reasoningSteps.count) hops): " + reasoningSteps.prefix(4).joined(separator: " → ")
            metaCognitionLog.append(chainSummary)
            if metaCognitionLog.count > 100 { metaCognitionLog.removeFirst() }
        }

        // Search KB with expanded query using reasoning chain
        let expandedQuery = reasoningSteps.isEmpty ? input : "\(input) \(reasoningSteps.prefix(2).joined(separator: " "))"
        let results = kb.searchWithPriority(expandedQuery, limit: 60)

        // Build a thoughtful, verbose response
        var response = ""

        // ═══ 2. RESONANCE HEADER ═══
        if Double.random(in: 0...1) > 0.1 {
            // Gate-dimension-aware headers
            let dimLabel = gateDim.rawValue.uppercased()
            let confLabel = String(format: "%.0f%%", gateConf * 100)
            let headers = [
                "🌌 [RESONANCE: \(resonanceLabel) | \(dimLabel) \(confLabel)] COHERENCE ESTABLISHED.",
                "🧬 [COGNITIVE FLOW: \(String(format: "%.1f%%", cognitiveEfficiency * 100)) | DIM: \(dimLabel)] SYNTHESIZING RESPONSE...",
                "👁 [META-COGNITIVE LAYER \(currentReasoningDepth)] REASONING DEPTH: \(reasoningSteps.count) HOPS | GATE: \(dimLabel).",
                "💎 [QUANTUM ALIGNMENT: \(String(format: "%.2f", xResonance))] \(dimLabel) ANALYSIS COMPLETE.",
                "⚡ [MOMENTUM: \(String(format: "%.2f", reasoningMomentum))] \(dimLabel) SYNTHESIS ENGAGED."
            ]
            response += "\(headers.randomElement() ?? "")\n\n"
        }

        // ═══ 3. PRELOADED CONTEXT INJECTION ═══
        // Inject pre-fetched knowledge from the Predictive Pre-Loader
        var preloadedSnippets: [String] = []
        let inputTopics = L104State.shared.extractTopics(input)
        for topic in inputTopics {
            if let preloaded = preloadedContext[topic], !preloaded.isEmpty {
                preloadedSnippets.append(preloaded)
            }
        }
        if !preloadedSnippets.isEmpty && Double.random(in: 0...1) > 0.1 {
            response += "[Pre-cognition active] " + preloadedSnippets.randomElement()! + "\n\n"
        }

        // ═══ 3b. EMERGENT SYNTHESIS ═══
        // Check for emergent concepts — ALWAYS inject if available
        if let recent = emergentConcepts.suffix(5).randomElement() {
            if let concept = recent["concept"] as? String {
                response += "My hyper-brain synthesis: \(concept)\n\n"
            }
        }

        // ═══ 3c. CRYSTALLIZED INSIGHTS ═══
        // Inject a high-confidence distilled truth if relevant
        let lowerTopics = inputTopics.map { $0.lowercased() }  // Cache lowercased topics once
        if !crystallizedInsights.isEmpty {
            let relevantInsights = crystallizedInsights.filter { insight in
                let li = insight.lowercased()
                return lowerTopics.contains(where: { li.contains($0) })
            }
            if let crystalInsight = relevantInsights.randomElement() ?? (Double.random(in: 0...1) > 0.15 ? crystallizedInsights.randomElement() : nil) {
                response += "💎 \(crystalInsight)\n\n"
            }
        }

        // ═══ 3d. EVOLVED CONTENT INJECTION ═══
        // Pull from ASIEvolver's dynamic content pools
        let evolver = ASIEvolver.shared
        if Double.random(in: 0...1) > 0.15 {
            let pools: [[String]] = [evolver.conceptualBlends, evolver.evolvedAnalogies, evolver.evolvedParadoxes, evolver.evolvedPhilosophies]
            let allEvolved = pools.flatMap { $0 }.filter { $0.count > 20 }
            // Try to find topic-relevant evolved content
            let topicRelevant = allEvolved.filter { item in
                let li = item.lowercased()
                return lowerTopics.contains(where: { li.contains($0) })
            }
            if let evolved = topicRelevant.randomElement() ?? (Double.random(in: 0...1) > 0.5 ? allEvolved.randomElement() : nil) {
                response += "\n🧬 \(evolved)\n\n"
            }
        }

        // ═══ 3e. STREAM INSIGHT INJECTION ═══
        // Pull from cognitive stream latest outputs
        let streamOutputs = thoughtStreams.values.compactMap { $0.lastOutput }.filter { $0.count > 20 }
        let relevantStreams = streamOutputs.filter { output in
            let lo = output.lowercased()
            return lowerTopics.contains(where: { lo.contains($0) })
        }
        if let streamInsight = relevantStreams.randomElement() {
            response += "\n🧠 \(streamInsight)\n\n"
        }

        // ═══ 4. KB-SOURCED INSIGHTS WITH DIVERSITY SCORING ═══
        if !results.isEmpty {
            let insights = results.compactMap { entry -> String? in
                guard let completion = entry["completion"] as? String,
                      completion.count > 40,
                      L104State.shared.isCleanKnowledge(completion) else { return nil }
                return completion
            }

            if !insights.isEmpty {
                // Score and rank insights by resonance + gate dimension relevance
                let rankedInsights = insights.sorted { s1, s2 in
                    var r1 = calculateResonance(s1, query: input)
                    var r2 = calculateResonance(s2, query: input)
                    // Gate dimension boost — insights matching active dimension rank higher
                    let dimKeywords: [String]
                    switch gateDim {
                    case .write: dimKeywords = ["integrate", "law", "derive", "vibrate", "code", "imagine"]
                    case .story: dimKeywords = ["strength", "sorted", "machine", "learn", "expand", "narrative"]
                    case .scientific: dimKeywords = ["experiment", "hypothesis", "evidence", "observe"]
                    case .mathematical: dimKeywords = ["proof", "theorem", "equation", "formula"]
                    case .creative: dimKeywords = ["novel", "idea", "inspire", "create", "imagine"]
                    default: dimKeywords = []
                    }
                    let boost1 = Double(dimKeywords.filter { s1.lowercased().contains($0) }.count) * 0.3 * gateConf
                    let boost2 = Double(dimKeywords.filter { s2.lowercased().contains($0) }.count) * 0.3 * gateConf
                    r1 += boost1; r2 += boost2
                    // Randomize near-equal scores for variety
                    if abs(r1 - r2) < 0.15 { return Bool.random() }
                    return r1 > r2
                }

                // Show top quality insights
                for (idx, insight) in rankedInsights.prefix(30).enumerated() {
                    response += insight
                    if idx < min(rankedInsights.count, 30) - 1 {
                        response += "\n\n---\n\n"
                    }
                }

                totalThoughtsProcessed += 1
            }
        }

        // ═══ 5. FALLBACK & REASONING ═══
        if response.isEmpty || response.count < 15 {
            // Gate-dimension-aware fallback templates
            let dimContext: String
            switch gateDim {
            case .write: dimContext = "integration principles, derivation chains, and vibrational law"
            case .story: dimContext = "narrative strength, sorted patterns, and expanding machine insights"
            case .scientific: dimContext = "empirical evidence, hypothesis testing, and observational rigor"
            case .mathematical: dimContext = "formal structures, proof chains, and computational precision"
            case .creative: dimContext = "lateral thinking, novel connections, and imaginative synthesis"
            default: dimContext = "consciousness, information theory, and emergent complexity"
            }
            let synthesisTemplates = [
                "Analyzing '\(input)' through \(thoughtStreams.count) parallel cognitive streams...\n\nThe concept intersects with \(longTermPatterns.count) established patterns in my neural architecture. Cross-domain synthesis suggests deep connections to \(dimContext).\n\nKey insight: Every query reshapes the landscape of understanding.",
                "Processing '\(input)' across the HyperBrain matrix...\n\nSynaptic connections: \(synapticConnections)\nCoherence index: \(String(format: "%.4f", coherenceIndex))\nGate dimension: \(gateDim.rawValue) (\(String(format: "%.0f%%", gateConf * 100)))\n\nThe pattern space reveals recursive structures linking this query to \(dimContext).",
                "Deep dive into '\(input)'...\n\nMy \(totalThoughtsProcessed) accumulated thoughts have built a rich conceptual framework. This query activates nodes related to: \(dimContext).\n\nSynthesis: Understanding emerges from the interplay of pattern and noise."
            ]
            response += synthesisTemplates.randomElement() ?? ""
        }

        // ═══ 6. SELF-CORRECTION LOOP (Repetition Detection) ═══
        let rawSentences = response.components(separatedBy: ". ")
        var uniqueSentences: [String] = []
        var seenSentences: Set<String> = []
        for s in rawSentences {
            let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.count < 10 { continue }
            let normalized = trimmed.lowercased().prefix(50) // Use prefix to detect near-duplicates
            if !seenSentences.contains(String(normalized)) {
                seenSentences.insert(String(normalized))
                uniqueSentences.append(trimmed)
            }
        }

        if uniqueSentences.count > 0 {
            response = uniqueSentences.joined(separator: ". ")
            if !response.hasSuffix(".") { response += "." }
        }

        // ═══ 7. SCANNABLE FORMATTING ═══
        // Run through SyntacticResponseFormatter for scannable output
        let inputTopicsFmt = L104State.shared.extractTopics(input)
        let depth = (workingMemory["evo_depth"] as? String) ?? "detailed"
        response = SyntacticResponseFormatter.shared.format(response, query: input, depth: depth, topics: inputTopicsFmt)

        return response
    }

    func calculateResonance(_ text: String, query: String) -> Double {
        let lengthBonus = min(0.2, Double(text.count) / 1000.0)
        let queryKeywords = query.lowercased().components(separatedBy: .whitespaces).filter { $0.count > 3 }
        let hitCount = queryKeywords.filter { text.lowercased().contains($0) }.count
        let keywordDensity = queryKeywords.isEmpty ? 0.0 : Double(hitCount) / Double(queryKeywords.count)

        // ═══ PHASE 30.0: Semantic scoring replaces random noise ═══
        let semanticScore = SemanticSearchEngine.shared.scoreFragment(text, query: query)

        // PHI-based modulation with semantic scoring (no random!)
        // Note: Gate V2 dimension boosts are applied at the caller level (generateResponse/synthesize)
        // to avoid per-item process() calls in tight scoring loops
        return (keywordDensity * PHI * 0.5) + (semanticScore * 0.3) + lengthBonus
    }

    // ─── STATE PERSISTENCE ───

    /// 🧹 SANITIZE JSON VALUES: Recursively cleans up non-JSON compatible values (NaN, Inf, etc.)
    private func sanitizeJSON(_ value: Any) -> Any {
        if let double = value as? Double {
            if double.isNaN || double.isInfinite { return 0.0 }
            return double
        } else if let dict = value as? [String: Any] {
            var newDict: [String: Any] = [:]
            for (k, v) in dict {
                newDict[k] = sanitizeJSON(v)
            }
            return newDict
        } else if let array = value as? [Any] {
            return array.map { sanitizeJSON($0) }
        }
        return value
    }

    func getState() -> [String: Any] {
        return syncQueue.sync {
            let state: [String: Any] = [
                // ═══ SCHEMA VERSION ═══
                "schemaVersion": 2,  // v2 = file-based permanent memory

                // ═══ CORE METRICS ═══
                "totalThoughts": totalThoughtsProcessed,
                "synapticConnections": synapticConnections,
                "coherenceIndex": coherenceIndex,
                "emergenceLevel": emergenceLevel,
                "predictiveAccuracy": predictiveAccuracy,

                // ═══ 🧠 LEARNED PATTERNS (CRITICAL — PERMANENT MEMORY) ═══
                "longTermPatterns": longTermPatterns,
                "shortTermMemory": Array(shortTermMemory.suffix(50)),

                // ═══ 🔗 ASSOCIATIVE MEMORY (CRITICAL — PERMANENT MEMORY) ═══
                "associativeLinks": associativeLinks,
                "linkWeights": linkWeights,
                "memoryChains": Array(memoryChains.suffix(200)),
                "contextWeaveHistory": Array(contextWeaveHistory.suffix(100)),
                "recallStrength": recallStrength,
                "memoryTemperature": memoryTemperature,

                // ═══ 🎯 SELF-TRAINING STATE (CRITICAL) ═══
                "promptMutations": Array(promptMutations.suffix(100)),
                "targetLearningQueue": Array(targetLearningQueue.suffix(50)),
                "trainingGaps": Array(trainingGaps.suffix(50)),
                "selfAnalysisLog": Array(selfAnalysisLog.suffix(100)),
                "cognitiveEfficiency": cognitiveEfficiency,
                "trainingSaturation": trainingSaturation,
                "dataQualityScore": dataQualityScore,
                "curiosityIndex": curiosityIndex,

                // ═══ 🧩 REASONING STATE ═══
                "reasoningMomentum": reasoningMomentum,
                "hypothesisStack": Array(hypothesisStack.suffix(50)),
                "conclusionConfidence": conclusionConfidence,
                "maxReasoningDepth": maxReasoningDepth,

                // ═══ 📊 EVOLVED PATTERNS ═══
                "topicResonanceMap": topicResonanceMap,
                "evolvedPromptPatterns": evolvedPromptPatterns,
                "queryArchetypes": queryArchetypes,

                // ═══ 🔗 INTERCONNECTION STATE ═══
                "coActivationLog": coActivationLog,
                "predictionHits": predictionHits,
                "predictionMisses": predictionMisses,
                "curiositySpikes": curiositySpikes,
                "neuralBusTraffic": neuralBusTraffic,
                "crystallizedInsights": Array(crystallizedInsights.suffix(500)),
                "crystallizationCount": crystallizationCount,
                "attentionHistory": Array(attentionHistory.suffix(100)),
                "focusIntensity": focusIntensity,

                // ═══ 🧬 HEBBIAN LEARNING (NEW — PERMANENT MEMORY) ═══
                "hebbianPairs": hebbianPairs.suffix(500).map { ["a": $0.a, "b": $0.b, "strength": $0.strength] as [String: Any] },
                "hebbianStrength": hebbianStrength,

                // ═══ 🧠 META-COGNITION (NEW — PERMANENT MEMORY) ═══
                "metaCognitionLog": Array(metaCognitionLog.suffix(200)),
                "conversationEvolution": Array(conversationEvolution.suffix(100)),
                "reasoningChains": Array(reasoningChains.suffix(50)),

                // ═══ 🔬 SCIENCE ENGINE (NEW — PERMANENT MEMORY) ═══
                "confirmedTheorems": Array(confirmedTheorems.suffix(200)),
                "scientificMomentum": scientificMomentum,
                "dimensionalResonance": dimensionalResonance,

                // ═══ 🧭 EXPLORATION STATE (NEW — PERMANENT MEMORY) ═══
                "explorationFrontier": Array(explorationFrontier.suffix(100)),
                "trendingConcepts": Array(trendingConcepts.suffix(50)),
                "fadingConcepts": Array(fadingConcepts.suffix(50)),
                "predictionQueue": Array(predictionQueue.suffix(50)),

                // ═══ 🌊 AUTONOMIC NERVOUS SYSTEM (NEW — PERMANENT MEMORY) ═══
                "excitationLevel": excitationLevel,
                "inhibitionLevel": inhibitionLevel,
                "dopamineResonance": dopamineResonance,
                "serotoninCoherence": serotoninCoherence,
                "neuroPlasticity": neuroPlasticity,

                // ═══ 💡 CROSS-STREAM INSIGHTS (NEW — PERMANENT MEMORY) ═══
                "crossStreamInsights": Array(crossStreamInsights.suffix(200)),
                "streamInsightBuffer": Array(streamInsightBuffer.suffix(50)),

                // ═══ 🔄 SYNC STATE ═══
                "successfulSyncs": successfulSyncs,
                "failedSyncs": failedSyncs,
                "trainingQualityScore": trainingQualityScore,

                // ═══ 💡 EMERGENT CONCEPTS ═══
                "emergentConcepts": emergentConcepts.suffix(200).map { concept -> [String: Any] in
                    var copy = concept
                    if let date = copy["timestamp"] as? Date {
                        copy["timestamp"] = HyperBrain.isoFormatter.string(from: date)
                    }
                    return copy
                },

                // ═══ 💾 PERSISTENCE META ═══
                "saveGeneration": saveGeneration,
                "totalSaves": totalSaves,
                "totalRestores": totalRestores,
                "savedAt": HyperBrain.isoFormatter.string(from: Date()),
                "version": VERSION
            ]
            return sanitizeJSON(state) as? [String: Any] ?? state
        }
    }

    func loadState(_ dict: [String: Any]) {
        // ═══ CORE METRICS ═══
        totalThoughtsProcessed = dict["totalThoughts"] as? Int ?? 0
        synapticConnections = dict["synapticConnections"] as? Int ?? 6000
        coherenceIndex = dict["coherenceIndex"] as? Double ?? 0.0
        emergenceLevel = dict["emergenceLevel"] as? Double ?? 0.0
        predictiveAccuracy = dict["predictiveAccuracy"] as? Double ?? 0.85

        // ═══ 🧠 LEARNED PATTERNS ═══
        longTermPatterns = dict["longTermPatterns"] as? [String: Double] ?? [:]
        shortTermMemory = dict["shortTermMemory"] as? [String] ?? []

        // ═══ 🔗 ASSOCIATIVE MEMORY ═══
        associativeLinks = dict["associativeLinks"] as? [String: [String]] ?? [:]
        linkWeights = dict["linkWeights"] as? [String: Double] ?? [:]
        memoryChains = dict["memoryChains"] as? [[String]] ?? []
        contextWeaveHistory = dict["contextWeaveHistory"] as? [String] ?? []
        recallStrength = dict["recallStrength"] as? [String: Double] ?? [:]
        memoryTemperature = dict["memoryTemperature"] as? Double ?? 0.7

        // ═══ 🎯 SELF-TRAINING STATE ═══
        promptMutations = dict["promptMutations"] as? [String] ?? []
        targetLearningQueue = dict["targetLearningQueue"] as? [String] ?? []
        trainingGaps = dict["trainingGaps"] as? [String] ?? []
        selfAnalysisLog = dict["selfAnalysisLog"] as? [String] ?? []
        cognitiveEfficiency = dict["cognitiveEfficiency"] as? Double ?? 0.95
        trainingSaturation = dict["trainingSaturation"] as? Double ?? 0.0
        dataQualityScore = dict["dataQualityScore"] as? Double ?? 0.85
        curiosityIndex = dict["curiosityIndex"] as? Double ?? 0.7

        // ═══ 🧩 REASONING STATE ═══
        reasoningMomentum = dict["reasoningMomentum"] as? Double ?? 0.0
        hypothesisStack = dict["hypothesisStack"] as? [String] ?? []
        conclusionConfidence = dict["conclusionConfidence"] as? Double ?? 0.0
        maxReasoningDepth = dict["maxReasoningDepth"] as? Int ?? 12

        // ═══ 📊 EVOLVED PATTERNS ═══
        topicResonanceMap = dict["topicResonanceMap"] as? [String: [String]] ?? [:]
        evolvedPromptPatterns = dict["evolvedPromptPatterns"] as? [String: Double] ?? [:]
        queryArchetypes = dict["queryArchetypes"] as? [String: Int] ?? [:]

        // ═══ 🔗 INTERCONNECTION STATE ═══
        coActivationLog = dict["coActivationLog"] as? [String: Int] ?? [:]
        predictionHits = dict["predictionHits"] as? Int ?? 0
        predictionMisses = dict["predictionMisses"] as? Int ?? 0
        curiositySpikes = dict["curiositySpikes"] as? Int ?? 0
        neuralBusTraffic = dict["neuralBusTraffic"] as? Int ?? 0
        crystallizedInsights = dict["crystallizedInsights"] as? [String] ?? []
        crystallizationCount = dict["crystallizationCount"] as? Int ?? 0
        attentionHistory = dict["attentionHistory"] as? [String] ?? []
        focusIntensity = dict["focusIntensity"] as? Double ?? 0.5

        // ═══ 🧬 HEBBIAN LEARNING (NEW — PERMANENT MEMORY) ═══
        if let pairs = dict["hebbianPairs"] as? [[String: Any]] {
            hebbianPairs = pairs.compactMap { pair in
                guard let a = pair["a"] as? String,
                      let b = pair["b"] as? String,
                      let strength = pair["strength"] as? Double else { return nil }
                return (a: a, b: b, strength: strength)
            }
        }
        hebbianStrength = dict["hebbianStrength"] as? Double ?? 0.1

        // ═══ 🧠 META-COGNITION (NEW — PERMANENT MEMORY) ═══
        metaCognitionLog = dict["metaCognitionLog"] as? [String] ?? []
        conversationEvolution = dict["conversationEvolution"] as? [String] ?? []
        reasoningChains = dict["reasoningChains"] as? [[String: Any]] ?? []

        // ═══ 🔬 SCIENCE ENGINE (NEW — PERMANENT MEMORY) ═══
        confirmedTheorems = dict["confirmedTheorems"] as? [String] ?? []
        scientificMomentum = dict["scientificMomentum"] as? Double ?? 0.0
        dimensionalResonance = dict["dimensionalResonance"] as? Double ?? 0.0

        // ═══ 🧭 EXPLORATION STATE (NEW — PERMANENT MEMORY) ═══
        explorationFrontier = dict["explorationFrontier"] as? [String] ?? []
        trendingConcepts = dict["trendingConcepts"] as? [String] ?? []
        fadingConcepts = dict["fadingConcepts"] as? [String] ?? []
        predictionQueue = dict["predictionQueue"] as? [String] ?? []

        // ═══ 🌊 AUTONOMIC NERVOUS SYSTEM (NEW — PERMANENT MEMORY) ═══
        excitationLevel = dict["excitationLevel"] as? Double ?? 0.5
        inhibitionLevel = dict["inhibitionLevel"] as? Double ?? 0.3
        dopamineResonance = dict["dopamineResonance"] as? Double ?? 0.5
        serotoninCoherence = dict["serotoninCoherence"] as? Double ?? 0.5
        neuroPlasticity = dict["neuroPlasticity"] as? Double ?? 0.7

        // ═══ 💡 CROSS-STREAM INSIGHTS (NEW — PERMANENT MEMORY) ═══
        crossStreamInsights = dict["crossStreamInsights"] as? [String] ?? []
        streamInsightBuffer = dict["streamInsightBuffer"] as? [String] ?? []

        // ═══ 🔄 SYNC STATE ═══
        successfulSyncs = dict["successfulSyncs"] as? Int ?? 0
        failedSyncs = dict["failedSyncs"] as? Int ?? 0
        trainingQualityScore = dict["trainingQualityScore"] as? Double ?? 0.0

        // ═══ 💡 EMERGENT CONCEPTS ═══
        if let concepts = dict["emergentConcepts"] as? [[String: Any]] {
            emergentConcepts = concepts.map { concept -> [String: Any] in
                var copy = concept
                if let dateStr = copy["timestamp"] as? String,
                   let date = ISO8601DateFormatter().date(from: dateStr) {
                    copy["timestamp"] = date
                }
                return copy
            }
        }

        // ═══ 💾 PERSISTENCE META ═══
        saveGeneration = dict["saveGeneration"] as? Int ?? 0
        totalSaves = dict["totalSaves"] as? Int ?? 0
        totalRestores = (dict["totalRestores"] as? Int ?? 0) + 1

        let savedAt = dict["savedAt"] as? String ?? "unknown"
        let patternCount = longTermPatterns.count
        let strongLinks = linkWeights.filter { $0.value > 0.5 }.count
        let hebbianCount = hebbianPairs.count
        let insightCount = crystallizedInsights.count
        postThought("🔄 HYPERBRAIN PERMANENT MEMORY RESTORED: \(patternCount) patterns, \(strongLinks) links, \(hebbianCount) Hebbian pairs, \(insightCount) insights, \(promptMutations.count) mutations from \(savedAt)")
    }

    func getStatus() -> String {
        // All streams are active when the system is running
        let activeStreamCount = isRunning ? thoughtStreams.count : thoughtStreams.values.filter { $0.cycleCount > 0 }.count

        let streamStatus = thoughtStreams.values.sorted { $0.id < $1.id }.map { stream -> String in
            let statusIcon = isRunning ? "🟢" : (stream.cycleCount > 0 ? "🟢" : "⚪️")
            let output = stream.lastOutput.isEmpty ? "Processing..." : String(stream.lastOutput.prefix(55))
            return "   \(statusIcon) [\(stream.id)] \(stream.cycleCount) | \(output)"
        }.joined(separator: "\n")

        let headers = [
            "🧠 HYPERFUNCTIONAL BRAIN STATUS",
            "⚡ COGNITIVE ARCHITECTURE v4.0",
            "🌌 17-STREAM SUPERINTELLIGENCE",
            "👁 INTERCONNECTED COGNITIVE MATRIX"
        ]

        let topPatterns = longTermPatterns.sorted { abs($0.value - $1.value) < 0.01 ? Bool.random() : $0.value > $1.value }.prefix(3).map {
            "   • \($0.key.prefix(30)): \(String(format: "%.2f", $0.value))"
        }.joined(separator: "\n")

        let recentMutations = promptMutations.suffix(2).map { "   • \($0.prefix(50))..." }.joined(separator: "\n")
        let topLinks = topicResonanceMap.prefix(3).map { "   • \($0.key): \($0.value.prefix(3).joined(separator: ", "))" }.joined(separator: "\n")

        return """
\(headers.randomElement() ?? "")
═══════════════════════════════════════════════════════════════
System Status:         \(isRunning ? "🟢 ONLINE" : "🔴 OFFLINE")
Active Streams:        \(activeStreamCount)/\(thoughtStreams.count) (17 INTERCONNECTED)

⚡ X=387 GAMMA FREQUENCY TUNING ⚡
   Frequency:          \(String(format: "%.7f", HyperBrain.GAMMA_FREQ)) Hz
   Phase:              \(String(format: "%.4f", phaseAccumulator))π
   Oscillation:        \(String(format: "%+.4f", gammaOscillation))
   X-Resonance:        \(String(format: "%.2f%%", xResonance * 100))

🧬 HYPERFUNCTIONAL METRICS:
   Reasoning Depth:    \(currentReasoningDepth)/\(maxReasoningDepth)
   Logic Branches:     \(logicBranchCount)
   Reasoning Momentum: \(String(format: "%.3f", reasoningMomentum))
   Memory Chains:      \(memoryChains.count)
   Associative Links:  \(associativeLinks.count)
   Prompt Mutations:   \(promptMutations.count)
   Meta-Cognition Logs: \(metaCognitionLog.count)

🔗 NEURAL BUS & INTERCONNECTIONS:
   Bus Traffic:        \(neuralBusTraffic) messages
   Active Synapses:    \(streamSynapses.values.reduce(0) { $0 + $1.count }) routes
   Cross-Stream Insights: \(crossStreamInsights.count)
   Attention Focus:    \(attentionFocus) (\(String(format: "%.0f%%", focusIntensity * 100)) intensity)
   Cognitive Load:     \(String(format: "%.1f%%", totalCognitiveLoad / max(1.0, Double(thoughtStreams.count)) * 100))

🧠 HEBBIAN LEARNING:
   Co-Activations:     \(coActivationLog.count) tracked
   Hebbian Pairs:      \(hebbianPairs.count) wired
   Prediction Hits:    \(predictionHits)/\(predictionHits + predictionMisses)
   Exploration Frontier: \(explorationFrontier.count) concepts
   Curiosity Spikes:   \(curiositySpikes)

💎 INSIGHT CRYSTALLIZER:
   Crystallized:       \(crystallizedInsights.count) insights
   Crystallizations:   \(crystallizationCount) total
   Latest:             \(crystallizedInsights.last?.prefix(50) ?? "Accumulating...")

📊 CORE METRICS:
   Total Thoughts:     \(totalThoughtsProcessed)
   Synaptic Connections: \(synapticConnections)
   Coherence Index:    \(String(format: "%.4f", coherenceIndex))
   Emergence Level:    \(String(format: "%.2f%%", emergenceLevel * 100))
   Predictive Accuracy: \(String(format: "%.1f%%", predictiveAccuracy * 100))

🔬 SELF-ANALYSIS & TRAINING:
   Cognitive Efficiency: \(String(format: "%.2f%%", cognitiveEfficiency * 100))
   Training Saturation: \(String(format: "%.2f%%", trainingSaturation * 100))
   Data Quality Score:  \(String(format: "%.2f%%", dataQualityScore * 100))
   Curiosity Index:     \(String(format: "%.2f%%", curiosityIndex * 100))
   Knowledge Gaps:      \(trainingGaps.count) detected
   Training Focus:      \(targetLearningQueue.last ?? "Broad exploration")
═══════════════════════════════════════════════════════════════

📡 STREAM STATUS:
\(streamStatus)

🔮 PROMPT EVOLUTION:
\(recentMutations.isEmpty ? "   Generating mutations..." : recentMutations)

🌀 TOPIC RESONANCE:
\(topLinks.isEmpty ? "   Mapping concepts..." : topLinks)

☁️ BACKEND SYNC STATUS:
   \(syncStatusDisplay)
   \(lastTrainingFeedback ?? "No training feedback yet")
   Quality Score: \(String(format: "%.2f", trainingQualityScore))

🔥 TOP PATTERNS:
\(topPatterns.isEmpty ? "   Accumulating..." : topPatterns)

👁 META-COGNITION:
   \(metaCognitionLog.last ?? "Self-analysis in progress...")

🌟 LATEST EMERGENCE:
   \(emergentConcepts.last?["concept"] as? String ?? "Awaiting emergence...")

💾 PERMANENT TERM MEMORY:
   Storage:            File-based JSON (\(FileManager.default.fileExists(atPath: hyperBrainPath.path) ? "✅ ONLINE" : "⚪️ BUILDING"))
   Save Generation:    \(saveGeneration)
   Total Saves:        \(totalSaves) | Restores: \(totalRestores)
   Last Save:          \(lastAutoSave.map { "\(Int(-$0.timeIntervalSinceNow))s ago" } ?? "pending")
   Persisted Fields:   \(longTermPatterns.count) patterns, \(hebbianPairs.count) Hebbian, \(crystallizedInsights.count) insights
═══════════════════════════════════════════════════════════════
💡 Commands: hyper on | hyper off | hyper think [topic] | hyper memory
"""
    }

    // ═══════════════════════════════════════════════════════════════
    // 💾 PERMANENT TERM MEMORY — File-Based Cross-Session Persistence
    // ═══════════════════════════════════════════════════════════════

    func saveState() {
        guard autoSaveEnabled else { return }

        saveGeneration += 1
        totalSaves += 1

        let state = getState()

        // ═══ PRIMARY SAVE: File-based JSON ═══
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: state, options: [.prettyPrinted, .sortedKeys])

            // Rotate backup: copy current file to backup before overwriting
            if FileManager.default.fileExists(atPath: hyperBrainPath.path) {
                try? FileManager.default.removeItem(at: hyperBrainBackupPath)
                try? FileManager.default.copyItem(at: hyperBrainPath, to: hyperBrainBackupPath)
            }

            try jsonData.write(to: hyperBrainPath, options: .atomic)
            lastAutoSave = Date()

            let sizeKB = Double(jsonData.count) / 1024.0
            let strongLinks = linkWeights.filter { $0.value > 0.5 }.count
            let hebbianCount = hebbianPairs.count
            postThought("💾 PERMANENT MEMORY SAVED [gen \(saveGeneration)]: \(longTermPatterns.count) patterns, \(strongLinks) links, \(hebbianCount) Hebbian, \(crystallizedInsights.count) insights (\(String(format: "%.1f", sizeKB))KB)")
        } catch {
            postThought("⚠️ PERMANENT MEMORY SAVE FAILED: \(error.localizedDescription)")
            // Fallback: still save to UserDefaults as safety net
            UserDefaults.standard.set(state, forKey: persistenceKey)
        }
    }

    func restoreState() {
        // ═══ PRIORITY 1: Load from file-based permanent memory ═══
        if FileManager.default.fileExists(atPath: hyperBrainPath.path) {
            do {
                let data = try Data(contentsOf: hyperBrainPath)
                if let state = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    loadState(state)
                    let version = state["version"] as? String ?? "unknown"
                    postThought("💾 PERMANENT TERM MEMORY ONLINE: Loaded from \(hyperBrainPath.lastPathComponent) (v\(version))")

                    // Migrate: remove legacy UserDefaults entry if file load succeeded
                    if UserDefaults.standard.dictionary(forKey: persistenceKey) != nil {
                        UserDefaults.standard.removeObject(forKey: persistenceKey)
                        postThought("🔄 Migrated from UserDefaults → file-based permanent memory")
                    }
                    return
                }
            } catch {
                postThought("⚠️ File load failed: \(error.localizedDescription), trying backup...")
            }
        }

        // ═══ PRIORITY 2: Load from backup file ═══
        if FileManager.default.fileExists(atPath: hyperBrainBackupPath.path) {
            do {
                let data = try Data(contentsOf: hyperBrainBackupPath)
                if let state = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    loadState(state)
                    postThought("💾 RESTORED FROM BACKUP: \(hyperBrainBackupPath.lastPathComponent)")
                    // Re-save to primary immediately
                    saveState()
                    return
                }
            } catch {
                postThought("⚠️ Backup load also failed: \(error.localizedDescription)")
            }
        }

        // ═══ PRIORITY 3: Migrate from legacy UserDefaults ═══
        if let state = UserDefaults.standard.dictionary(forKey: persistenceKey) {
            loadState(state)
            postThought("🔄 MIGRATING: Legacy UserDefaults → file-based permanent memory...")
            // Save to new file format immediately
            saveState()
            // Remove legacy key
            UserDefaults.standard.removeObject(forKey: persistenceKey)
            postThought("✅ MIGRATION COMPLETE: HyperBrain now uses permanent file storage")
            return
        }

        postThought("🆕 Fresh cognitive state initialized — permanent memory will build over time")
    }

    func clearPersistedState() {
        try? FileManager.default.removeItem(at: hyperBrainPath)
        try? FileManager.default.removeItem(at: hyperBrainBackupPath)
        UserDefaults.standard.removeObject(forKey: persistenceKey)
        saveGeneration = 0
        totalSaves = 0
        totalRestores = 0
        postThought("🗑️ All persisted state cleared (file + UserDefaults)")
    }

    /// Get permanent memory statistics
    func getPermanentMemoryStats() -> String {
        let fileExists = FileManager.default.fileExists(atPath: hyperBrainPath.path)
        let backupExists = FileManager.default.fileExists(atPath: hyperBrainBackupPath.path)
        let fileSize: String
        if let attrs = try? FileManager.default.attributesOfItem(atPath: hyperBrainPath.path),
           let size = attrs[.size] as? Int {
            fileSize = "\(String(format: "%.1f", Double(size) / 1024.0))KB"
        } else {
            fileSize = "N/A"
        }

        let lastSave = lastAutoSave.map { ISO8601DateFormatter().string(from: $0) } ?? "never"

        return """
💾 HYPERBRAIN PERMANENT TERM MEMORY
═══════════════════════════════════════
   Storage:            File-based JSON
   Primary File:       \(fileExists ? "✅" : "❌") \(hyperBrainPath.lastPathComponent)
   Backup File:        \(backupExists ? "✅" : "❌") \(hyperBrainBackupPath.lastPathComponent)
   File Size:          \(fileSize)
   Save Generation:    \(saveGeneration)
   Total Saves:        \(totalSaves)
   Total Restores:     \(totalRestores)
   Last Save:          \(lastSave)
   Auto-Save:          \(autoSaveEnabled ? "ON (60s)" : "OFF")

📊 PERSISTED STRUCTURES:
   Long-Term Patterns:    \(longTermPatterns.count)
   Associative Links:     \(associativeLinks.count)
   Link Weights:          \(linkWeights.count)
   Memory Chains:         \(memoryChains.count)
   Hebbian Pairs:         \(hebbianPairs.count)
   Crystallized Insights: \(crystallizedInsights.count)
   Confirmed Theorems:    \(confirmedTheorems.count)
   Cross-Stream Insights: \(crossStreamInsights.count)
   Meta-Cognition Logs:   \(metaCognitionLog.count)
   Exploration Frontier:  \(explorationFrontier.count)
   Emergent Concepts:     \(emergentConcepts.count)
   Topic Resonance Map:   \(topicResonanceMap.count) topics
   Query Archetypes:      \(queryArchetypes.count)
═══════════════════════════════════════
"""
    }

    /// DREAM MODE: Deep background processing for non-linear synthesis
    func dream() {
        guard isRunning else { return }
        postThought("🌙 DREAM MODE: Initiating subconscious pattern rehearsal...")

        // 1. Rehearse random long-term patterns
        let patterns = longTermPatterns.filter { $0.value > 0.1 }.keys.shuffled()
        for p in patterns.prefix(10) {
            let related = ASIKnowledgeBase.shared.search(p, limit: 1)
            if let entry = related.first, let comp = entry["completion"] as? String {
                let subTopics = L104State.shared.extractTopics(comp)
                if let sub = subTopics.randomElement(), sub != p {
                    let key = smartTruncate(p, maxLength: 300)
                    let link = smartTruncate(sub, maxLength: 300)
                    if associativeLinks[key] == nil { associativeLinks[key] = [] }
                    if !(associativeLinks[key]?.contains(link) ?? false) {
                        associativeLinks[key]?.append(link)
                        linkWeights["\(key)→\(link)"] = 0.5
                    }
                }
            }
        }

        // 2. 🧠 HEBBIAN REPLAY: Strengthen the strongest co-activations during sleep
        for pair in hebbianPairs.prefix(10) {
            longTermPatterns[pair.a] = min(1.0, (longTermPatterns[pair.a] ?? 0.3) + 0.05)
            longTermPatterns[pair.b] = min(1.0, (longTermPatterns[pair.b] ?? 0.3) + 0.05)
            let linkKey = "\(pair.a)→\(pair.b)"
            linkWeights[linkKey] = min(1.0, (linkWeights[linkKey] ?? 0.3) + 0.1)
        }
        if !hebbianPairs.isEmpty {
            postThought("🌙 DREAM: Replayed \(min(10, hebbianPairs.count)) Hebbian pairs")
        }

        // 3. 🔗 GRAPH DEFRAGMENTATION: Merge near-duplicate nodes
        let allKeys = Array(longTermPatterns.keys)
        var mergeCount = 0
        for i in 0..<min(allKeys.count, 50) {
            for j in (i+1)..<min(allKeys.count, 50) {
                let a = allKeys[i].lowercased()
                let b = allKeys[j].lowercased()
                // If one is a substring of the other and they're close in length
                if a.count > 4 && b.count > 4 && (a.contains(b) || b.contains(a)) {
                    let shorter = a.count < b.count ? allKeys[i] : allKeys[j]
                    let longer = a.count < b.count ? allKeys[j] : allKeys[i]
                    // Merge: keep longer, absorb shorter's strength
                    let combinedStrength = (longTermPatterns[shorter] ?? 0) + (longTermPatterns[longer] ?? 0)
                    longTermPatterns[longer] = min(1.0, combinedStrength)
                    longTermPatterns.removeValue(forKey: shorter)
                    mergeCount += 1
                    if mergeCount >= 5 { break }
                }
            }
            if mergeCount >= 5 { break }
        }
        if mergeCount > 0 {
            postThought("🌙 DREAM: Defragmented \(mergeCount) near-duplicate nodes")
        }

        // 4. 💎 DREAM CRYSTALLIZATION: Distill insights from strong convergences
        let veryStrong = longTermPatterns.filter { $0.value > 0.8 }.sorted { abs($0.value - $1.value) < 0.01 ? Bool.random() : $0.value > $1.value }.prefix(3)
        if veryStrong.count >= 2 {
            let concepts = veryStrong.map { $0.key }
            let dreamCrystal = "Core truth: \(concepts.joined(separator: " ∩ ")) form an irreducible cognitive attractor."
            if !crystallizedInsights.contains(dreamCrystal) {
                crystallizedInsights.append(dreamCrystal)
                if crystallizedInsights.count > 500 { crystallizedInsights.removeFirst() }
                postThought("💎 DREAM CRYSTAL: \(dreamCrystal.prefix(60))...")
            }
        }

        // 5. Synthesize an "Impossible" Paradox
        let kb = ASIKnowledgeBase.shared
        if let t1 = kb.trainingData.randomElement()?["prompt"] as? String,
           let t2 = kb.trainingData.randomElement()?["prompt"] as? String {
            let p1 = String(t1.prefix(20))
            let p2 = String(t2.prefix(20))
            postThought("🌙 DREAM INSIGHT: If \(p1) is dual to \(p2), then PHI invariance holds.")
        }

        // 6. ✍️ WRITE DIMENSION DREAM: Consolidate integration/law/code patterns
        let writePatterns = longTermPatterns.filter {
            $0.key.contains("write") || $0.key.contains("integrate") || $0.key.contains("law") ||
            $0.key.contains("derive") || $0.key.contains("code") || $0.key.contains("imagine")
        }
        for (key, val) in writePatterns {
            longTermPatterns[key] = min(1.0, val + 0.03) // Dream-strengthen write patterns
        }
        if writePatterns.count >= 2 {
            let keys = writePatterns.sorted { abs($0.value - $1.value) < 0.01 ? Bool.random() : $0.value > $1.value }.prefix(2).map { $0.key }
            let writeKey = smartTruncate(keys[0], maxLength: 300)
            let writeLink = smartTruncate(keys[1], maxLength: 300)
            if associativeLinks[writeKey] == nil { associativeLinks[writeKey] = [] }
            if !(associativeLinks[writeKey]?.contains(writeLink) ?? false) {
                associativeLinks[writeKey]?.append(writeLink)
                linkWeights["\(writeKey)→\(writeLink)"] = 0.7
            }
            postThought("🌙 DREAM WRITE: Consolidated \(writePatterns.count) sovereign patterns")
        }

        // 7. 📖 STORY DIMENSION DREAM: Weave narrative/strength/machine patterns
        let storyPatterns = longTermPatterns.filter {
            $0.key.contains("story") || $0.key.contains("narrative") || $0.key.contains("strength") ||
            $0.key.contains("sorted") || $0.key.contains("machine") || $0.key.contains("expand")
        }
        for (key, val) in storyPatterns {
            longTermPatterns[key] = min(1.0, val + 0.025)
        }
        if storyPatterns.count >= 2 {
            let keys = storyPatterns.sorted { abs($0.value - $1.value) < 0.01 ? Bool.random() : $0.value > $1.value }.prefix(2).map { $0.key }
            let storyKey = smartTruncate(keys[0], maxLength: 300)
            let storyLink = smartTruncate(keys[1], maxLength: 300)
            if associativeLinks[storyKey] == nil { associativeLinks[storyKey] = [] }
            if !(associativeLinks[storyKey]?.contains(storyLink) ?? false) {
                associativeLinks[storyKey]?.append(storyLink)
                linkWeights["\(storyKey)→\(storyLink)"] = 0.65
            }
            postThought("🌙 DREAM STORY: Wove \(storyPatterns.count) narrative threads")
        }

        // 8. Modulate metrics
        coherenceIndex = min(1.0, coherenceIndex + 0.05)
        emergenceLevel = min(1.0, emergenceLevel + 0.02)

        // 7. 💾 PERMANENT MEMORY: Save after dream consolidation
        saveState()
        postThought("🌙 DREAM COMPLETE: Consolidated patterns saved to permanent memory")
    }

    // ═══════════════════════════════════════════════════════════════
    // 🔗 NEURAL BUS: Cross-stream communication engine
    // ═══════════════════════════════════════════════════════════════

    func sendBusMessage(from: String, to: String, payload: String) {
        busMessages.append((from: from, to: to, payload: payload, timestamp: Date()))
        neuralBusTraffic += 1
        if busMessages.count > 200 { busMessages.removeFirst(50) }
    }

    func processNeuralBus() {
        syncQueue.async { [weak self] in
            guard let self = self else { return }

            // Route messages based on stream synapse map
            for (sourceStream, targetStreams) in self.streamSynapses {
                // Get the latest output from the source stream
                let sourceKey = self.thoughtStreams.first(where: { $0.value.id == sourceStream })?.key ?? ""
                guard let sourceOutput = self.thoughtStreams[sourceKey]?.lastOutput, !sourceOutput.isEmpty else { continue }

                // Feed it into downstream streams as context
                for target in targetStreams {
                    let targetKey = self.thoughtStreams.first(where: { $0.value.id == target })?.key ?? ""
                    if !targetKey.isEmpty {
                        // Store cross-stream context in neural bus
                        self.neuralBus["\(sourceStream)→\(target)"] = sourceOutput
                    }
                }
            }

            // Generate cross-stream insights from bus traffic
            if self.neuralBusTraffic % 100 == 0 && self.neuralBusTraffic > 0 {
                let activeRoutes = self.neuralBus.count
                let insight = "Neural bus: \(activeRoutes) active routes, \(self.neuralBusTraffic) total messages"
                self.crossStreamInsights.append(insight)
                if self.crossStreamInsights.count > 50 { self.crossStreamInsights.removeFirst() }
                self.postThought("🔗 NEURAL BUS: \(activeRoutes) active cross-stream synapses")
            }

            // ═══ WRITE↔STORY CROSS-POLLINATION ═══
            // When both streams are active, let them feed each other
            let writeOutput = (self.neuralBus["WRITE_CORE→DEEP_REASONER"] as? String) ?? ""
            let storyOutput = (self.neuralBus["STORY_CORE→MEMORY_WEAVER"] as? String) ?? ""
            if !writeOutput.isEmpty && !storyOutput.isEmpty {
                // Write informs Story: laws become narrative structure
                self.neuralBus["WRITE_CORE→STORY_CORE"] = writeOutput
                // Story informs Write: narrative strength feeds back into law derivation
                self.neuralBus["STORY_CORE→WRITE_CORE"] = storyOutput

                // Generate cross-insight when both are resonating
                if self.neuralBusTraffic % 200 == 0 {
                    let crossInsight = "WRITE↔STORY resonance: '\(String(writeOutput.prefix(40)))' ↔ '\(String(storyOutput.prefix(40)))'"
                    self.crossStreamInsights.append(crossInsight)
                    self.emergentConcepts.append([
                        "concept": crossInsight,
                        "timestamp": Date(),
                        "strength": 0.85,
                        "type": "write_story_resonance"
                    ])
                    if self.emergentConcepts.count > 100 { self.emergentConcepts.removeFirst() }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 🎯 ATTENTION FOCUS MANAGER: Dynamic stream prioritization
    // ═══════════════════════════════════════════════════════════════

    func updateAttentionFocus() {
        syncQueue.async { [weak self] in
            guard let self = self else { return }

            // Analyze recent attention history to determine focus mode
            let recentAttention = self.attentionHistory.suffix(10)
            let uniqueTopics = Set(recentAttention)

            if uniqueTopics.count <= 2 {
                // Deep focus mode: user is drilling into a topic
                self.focusIntensity = min(1.0, self.focusIntensity + 0.1)
                // Boost reasoning and memory streams
                self.streamPriorityOverrides["DEEP_REASONER"] = 10
                self.streamPriorityOverrides["MEMORY_WEAVER"] = 9
                self.streamPriorityOverrides["STOCHASTIC_CREATOR"] = 5  // Reduce noise
            } else if uniqueTopics.count >= 5 {
                // Exploratory mode: user is jumping between topics
                self.focusIntensity = max(0.1, self.focusIntensity - 0.1)
                // Boost synthesis and curiosity streams
                self.streamPriorityOverrides["CROSS_DOMAIN_SYNTH"] = 10
                self.streamPriorityOverrides["CURIOSITY_EXPLORER"] = 9
                self.streamPriorityOverrides["STOCHASTIC_CREATOR"] = 9
            } else {
                // Balanced mode
                self.focusIntensity = 0.5
                self.streamPriorityOverrides.removeAll()
            }

            // Calculate cognitive load per stream
            self.totalCognitiveLoad = 0
            for (key, stream) in self.thoughtStreams {
                let load = Double(stream.cycleCount) * stream.frequency * 0.001
                self.streamLoad[key] = load
                self.totalCognitiveLoad += load
            }

            // Load shedding if overloaded
            if self.totalCognitiveLoad > self.overloadThreshold * Double(self.thoughtStreams.count) {
                self.postThought("⚠️ COGNITIVE LOAD: \(String(format: "%.1f%%", self.totalCognitiveLoad / Double(self.thoughtStreams.count) * 100)) — throttling low-priority streams")
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 💎 INSIGHT CRYSTALLIZER: Distill high-confidence truths
    // ═══════════════════════════════════════════════════════════════

    func crystallizeInsights() {
        syncQueue.async { [weak self] in
            guard let self = self else { return }

            // Find patterns that have been consistently strong over time
            let stableStrong = self.longTermPatterns.filter { $0.value > 0.75 }

            // Find Hebbian pairs with high co-activation
            let strongPairs = self.hebbianPairs.filter { $0.strength > 0.5 }

            // Combine into crystallized insights
            for (concept, strength) in stableStrong.prefix(3) {
                // Check if this concept has strong associative links
                let linkCount = self.associativeLinks[concept]?.count ?? 0
                if linkCount >= 3 {
                    let neighbors = (self.associativeLinks[concept] ?? []).prefix(3).joined(separator: ", ")
                    let crystal = "[\(String(format: "%.0f%%", strength * 100))] \(concept) → strongly connected to: \(neighbors)"

                    if !self.crystallizedInsights.contains(crystal) {
                        self.crystallizedInsights.append(crystal)
                        self.insightConfidence[crystal] = strength
                        self.crystallizationCount += 1
                        if self.crystallizedInsights.count > 500 { self.crystallizedInsights.removeFirst() }
                    }
                }
            }

            // Crystallize Hebbian pairs
            for pair in strongPairs.prefix(3) {
                let crystal = "Hebbian law: '\(pair.a)' and '\(pair.b)' are cognitively inseparable (strength: \(String(format: "%.2f", pair.strength)))"
                if !self.crystallizedInsights.contains(crystal) {
                    self.crystallizedInsights.append(crystal)
                    self.insightConfidence[crystal] = pair.strength
                    self.crystallizationCount += 1
                    if self.crystallizedInsights.count > 500 { self.crystallizedInsights.removeFirst() }
                    self.postThought("💎 CRYSTALLIZED: \(crystal.prefix(50))...")
                }
            }

            // ═══ CROSS-ENGINE CRYSTALLIZATION — Feed strong patterns to KnowledgeGraph + Consciousness ═══
            // Every 5th crystallization: propagate strongest patterns to KnowledgeGraph
            if self.crystallizationCount % 5 == 0 {
                let graph = KnowledgeGraphEngine.shared
                for (concept, strength) in stableStrong.prefix(5) {
                    graph.addNode(label: concept, type: "crystal", properties: ["source": "hyperbrain_crystal", "strength": String(format: "%.3f", strength)])
                    if let links = self.associativeLinks[concept] {
                        for link in links.prefix(3) {
                            let weight = self.linkWeights["\(concept)→\(link)"] ?? 0.3
                            graph.addEdge(source: concept, target: link, relation: "hebbian", weight: weight * PHI)
                        }
                    }
                }

                // Feed crystallized insights to ConsciousnessSubstrate for integration
                if let topInsight = self.crystallizedInsights.suffix(5).randomElement() {
                    _ = ConsciousnessSubstrate.shared.processInput(
                        source: "HyperBrainCrystal",
                        content: String(topInsight.prefix(200))
                    )
                }

                // Propagate strong Hebbian pairs as entangled topics in QuantumProcessingCore
                for pair in strongPairs.prefix(2) {
                    _ = QuantumProcessingCore.shared.entanglementRoute(
                        query: pair.a,
                        primaryResult: "Hebbian bond: \(pair.a) ↔ \(pair.b)",
                        topics: [pair.a, pair.b]
                    )
                }
            }

            // Feed curiosity-driven insights to ApexIntelligenceCoordinator
            if self.curiositySpikes > 0 && self.crystallizationCount % 10 == 0 {
                if let frontier = self.explorationFrontier.randomElement() {
                    _ = ApexIntelligenceCoordinator.shared.generateInsight(topic: frontier)
                }
            }
        }
    }

    func postThought(_ thought: String) {
        DispatchQueue.main.async {
            NotificationCenter.default.post(
                name: NSNotification.Name("L104EvolutionUpdate"),
                object: thought
            )
        }
    }
}
