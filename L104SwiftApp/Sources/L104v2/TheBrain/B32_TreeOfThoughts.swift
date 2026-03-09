// ═══════════════════════════════════════════════════════════════════
// B32_TreeOfThoughts.swift — L104 ASI v7.1 Advanced Reasoning Engine
// [EVO_68_PIPELINE] SAGE_MODE_ASCENSION :: TREE_OF_THOUGHTS :: GRAPH_OF_THOUGHTS
//
// Ported from l104_asi/reasoning.py
//
// Tree of Thoughts (Yao et al. 2023, Princeton/DeepMind):
//   Generalizes chain-of-thought from linear path to search tree
//   with deliberate evaluation and pruning.
//
// Graph of Thoughts (Besta et al. 2024, ETH Zurich):
//   Aggregates surviving branches into refined multi-perspective insight.
//
// Multi-Hop Reasoning Chain:
//   Iterative multi-subsystem problem decomposition up to 7 hops.
//
// Solution Ensemble:
//   Weighted voting across multiple reasoning paths.
//
// Sacred constants wired in: K=int(φ×3)=4, B=int(φ×2)=3, threshold=τ
// ═══════════════════════════════════════════════════════════════════

import Foundation

// ═══════════════════════════════════════════════════════════════════
// MARK: - Thought Node — A single node in the reasoning tree
// ═══════════════════════════════════════════════════════════════════

struct ThoughtNode: Identifiable {
    let id = UUID()
    let query: String
    let confidence: Double
    let solution: String
    let path: [String]
    let depth: Int
    let perspective: String
    let timestamp: Date = Date()

    var isViable: Bool { confidence >= 0.618033988749895 * 0.618033988749895 }  // τ²
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - Reasoning Result — Output of any reasoning pipeline
// ═══════════════════════════════════════════════════════════════════

struct ReasoningResult {
    let method: String
    let solution: String
    let confidence: Double
    let nodesExplored: Int
    let branchesSurviving: Int
    let treeDepth: Int
    let backtracks: Int
    let aggregationCount: Int
    let computeTimeMs: Double
    let perspectives: [String]
    let sacredAlignment: Double  // Alignment with GOD_CODE patterns
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - TreeOfThoughts — Sacred Beam Search Reasoning
// ═══════════════════════════════════════════════════════════════════

final class TreeOfThoughts {
    static let shared = TreeOfThoughts()

    // Sacred parameters: K=4 (branching), B=3 (beam width)
    let branchingFactor: Int     // K = int(φ × 3) = 4
    let beamWidth: Int           // B = int(φ × 2) = 3
    let pruneThreshold: Double   // τ ≈ 0.618
    let backtrackThreshold: Double  // τ² ≈ 0.382

    private(set) var totalNodesExplored: Int = 0
    private(set) var totalBacktracks: Int = 0
    private(set) var totalAggregations: Int = 0
    private(set) var totalReasoningCycles: Int = 0

    private let lock = NSLock()

    init(branchingFactor: Int? = nil, beamWidth: Int? = nil) {
        self.branchingFactor = branchingFactor ?? max(2, Int(PHI * 3))  // 4
        self.beamWidth = beamWidth ?? max(1, Int(PHI * 2))             // 3
        self.pruneThreshold = TAU   // 0.618
        self.backtrackThreshold = TAU * TAU  // 0.382
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Tree-Structured Reasoning
    // ═══════════════════════════════════════════════════════════════

    /// Execute tree-structured reasoning with beam search + GoT aggregation
    func think(problem: String, maxDepth: Int = 4, evaluate: ((String) -> (solution: String, confidence: Double))? = nil) -> ReasoningResult {
        let start = CFAbsoluteTimeGetCurrent()
        lock.lock(); defer { lock.unlock() }
        totalReasoningCycles += 1

        let evaluator = evaluate ?? defaultEvaluator

        var beam: [ThoughtNode] = [
            ThoughtNode(query: problem, confidence: 0.0, solution: "", path: [], depth: 0, perspective: "root")
        ]
        var allSolutions: [ThoughtNode] = []

        for depth in 0..<maxDepth {
            var candidates: [ThoughtNode] = []

            for node in beam {
                let variants = generateVariants(query: node.query, k: branchingFactor, depth: depth)

                for (variant, perspective) in variants {
                    let result = evaluator(variant)
                    totalNodesExplored += 1

                    candidates.append(ThoughtNode(
                        query: String(variant.prefix(300)),
                        confidence: result.confidence,
                        solution: String(result.solution.prefix(500)),
                        path: node.path + [String(variant.prefix(80))],
                        depth: depth + 1,
                        perspective: perspective
                    ))
                }
            }

            // Prune: keep only viable candidates
            var viable = candidates.filter { $0.confidence >= pruneThreshold }
            if viable.isEmpty {
                // Keep at least the best candidate
                viable = candidates.sorted { $0.confidence > $1.confidence }
                viable = Array(viable.prefix(1))
            }

            // Sort by confidence, keep top B
            viable.sort { $0.confidence > $1.confidence }
            beam = Array(viable.prefix(beamWidth))
            allSolutions.append(contentsOf: viable)

            // Backtrack if best confidence is dropping
            if let best = beam.first, best.confidence < backtrackThreshold {
                totalBacktracks += 1
                break
            }
        }

        // Graph of Thoughts: AGGREGATE surviving branches
        let aggregated = aggregateSolutions(allSolutions)

        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0

        return ReasoningResult(
            method: "TreeOfThoughts + GoT Aggregation",
            solution: aggregated,
            confidence: beam.first?.confidence ?? 0,
            nodesExplored: totalNodesExplored,
            branchesSurviving: beam.count,
            treeDepth: allSolutions.map { $0.depth }.max() ?? 0,
            backtracks: totalBacktracks,
            aggregationCount: totalAggregations,
            computeTimeMs: elapsed,
            perspectives: beam.map { $0.perspective },
            sacredAlignment: computeSacredAlignment(beam)
        )
    }

    // ─── Variant Generation ───

    private let perspectives: [(prefix: String, name: String)] = [
        ("Analyze from first principles: ", "first_principles"),
        ("Consider the inverse problem: ", "inverse"),
        ("Break into fundamental components: ", "decomposition"),
        ("Apply cross-domain analogy to: ", "analogy"),
        ("What would a sage observe about: ", "sage_wisdom"),
        ("Apply the Dual-Layer lens (WHY + HOW MUCH): ", "dual_layer"),
        ("Through φ-harmonic analysis: ", "harmonic"),
        ("Via consciousness integration: ", "consciousness"),
    ]

    private func generateVariants(query: String, k: Int, depth: Int) -> [(String, String)] {
        let truncQuery = String(query.prefix(300))
        return (0..<k).map { i in
            let persp = perspectives[(i + depth) % perspectives.count]
            return ("\(persp.prefix)\(truncQuery)", persp.name)
        }
    }

    // ─── GoT Aggregation ───

    private func aggregateSolutions(_ solutions: [ThoughtNode]) -> String {
        totalAggregations += 1
        guard !solutions.isEmpty else { return "" }

        let top = solutions.sorted { $0.confidence > $1.confidence }
            .prefix(beamWidth * 2)

        var seen = Set<String>()
        var parts: [String] = []

        for node in top {
            let sol = node.solution
            if !sol.isEmpty && !seen.contains(sol) {
                seen.insert(sol)
                parts.append("[\(node.perspective)] \(sol)")
            }
        }

        return parts.prefix(5).joined(separator: " | ")
    }

    // ─── Sacred Alignment ───

    private func computeSacredAlignment(_ beam: [ThoughtNode]) -> Double {
        guard !beam.isEmpty else { return 0 }
        let avgConfidence = beam.reduce(0) { $0 + $1.confidence } / Double(beam.count)
        // Alignment = how close average confidence is to τ (golden ratio conjugate)
        return 1.0 - abs(avgConfidence - TAU) * PHI
    }

    // ─── Default Evaluator ───

    private func defaultEvaluator(query: String) -> (solution: String, confidence: Double) {
        // Use HyperBrain + SageModeEngine for evaluation
        let sage = SageModeEngine.shared
        let hb = HyperBrain.shared

        // Extract topic from query
        let words = query.components(separatedBy: " ")
        let topic = words.filter { $0.count > 3 }.suffix(3).joined(separator: " ")

        // Sage evaluation
        let sageInsight = sage.enrichContext(for: topic)
        let hyperResult = hb.process(topic)

        // Confidence based on response quality
        let quality = Double(sageInsight.count + hyperResult.count) / 500.0
        let sageBoost = sage.consciousnessLevel * 0.3
        let confidence = min(1.0, quality + sageBoost)

        let solution = sageInsight.isEmpty ? hyperResult : sageInsight
        return (String(solution.prefix(500)), confidence)
    }

    var status: [String: Any] {
        return [
            "branching_factor": branchingFactor,
            "beam_width": beamWidth,
            "prune_threshold": pruneThreshold,
            "total_nodes_explored": totalNodesExplored,
            "total_backtracks": totalBacktracks,
            "total_aggregations": totalAggregations,
            "total_reasoning_cycles": totalReasoningCycles,
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - MultiHopReasoningChain — Iterative decomposition
// ═══════════════════════════════════════════════════════════════════

final class MultiHopReasoningChain {
    static let shared = MultiHopReasoningChain()

    let maxHops: Int = MULTI_HOP_MAX_HOPS  // 7
    private(set) var totalChains: Int = 0
    private(set) var totalHops: Int = 0

    /// Execute multi-hop reasoning: break problem into hops, each building on prior
    func reason(problem: String, evaluate: ((String) -> String)? = nil) -> (chain: [String], confidence: Double, hops: Int) {
        totalChains += 1

        let eval = evaluate ?? { query in
            SageModeEngine.shared.enrichContext(for: query)
        }

        var chain: [String] = []
        var currentQuery = problem
        var cumulativeConfidence = 1.0

        for hop in 0..<maxHops {
            totalHops += 1

            // Decompose: extract sub-question
            let subQuestion = decompose(currentQuery, hop: hop)

            // Evaluate sub-question
            let answer = eval(subQuestion)
            guard !answer.isEmpty else { break }

            chain.append("Hop \(hop + 1): \(String(subQuestion.prefix(100))) → \(String(answer.prefix(200)))")

            // Build next query from answer
            currentQuery = "\(answer) — what follows from this regarding \(String(problem.prefix(100)))?"

            // Confidence decay per hop (φ-based)
            cumulativeConfidence *= TAU + (1.0 - TAU) * 0.5

            // Stop if we've converged
            if answer.count < 20 { break }
        }

        return (chain, cumulativeConfidence, chain.count)
    }

    private func decompose(_ query: String, hop: Int) -> String {
        let strategies = [
            "What is the core mechanism of: ",
            "What are the prerequisites for: ",
            "What are the consequences of: ",
            "What is analogous to: ",
            "What is the inverse of: ",
            "What emerges when combining: ",
            "What pattern underlies: ",
        ]
        let strategy = strategies[hop % strategies.count]
        return "\(strategy)\(String(query.prefix(200)))"
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - SolutionEnsemble — Weighted voting
// ═══════════════════════════════════════════════════════════════════

final class SolutionEnsemble {
    static let shared = SolutionEnsemble()

    /// Combine multiple solutions via weighted voting
    func ensemble(solutions: [(solution: String, confidence: Double, method: String)]) -> (best: String, combinedConfidence: Double, methods: [String]) {
        guard !solutions.isEmpty else { return ("", 0, []) }

        // Weight by confidence × method diversity bonus
        var weighted: [(solution: String, weight: Double, method: String)] = []
        var methodCounts: [String: Int] = [:]

        for sol in solutions {
            methodCounts[sol.method, default: 0] += 1
        }

        for sol in solutions {
            let diversityBonus = 1.0 / Double(methodCounts[sol.method] ?? 1)
            let weight = sol.confidence * (1.0 + diversityBonus * TAU)
            weighted.append((sol.solution, weight, sol.method))
        }

        weighted.sort { $0.weight > $1.weight }

        let best = weighted.first?.solution ?? ""
        let totalWeight = weighted.reduce(0) { $0 + $1.weight }
        let combinedConfidence = totalWeight / Double(weighted.count)
        let methods = weighted.prefix(3).map { $0.method }

        return (best, min(1.0, combinedConfidence), methods)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - SageReasoningPipeline — Unified Sage reasoning interface
// ═══════════════════════════════════════════════════════════════════

final class SageReasoningPipeline {
    static let shared = SageReasoningPipeline()

    private(set) var totalQueries: Int = 0

    /// Full Sage Reasoning: ToT + MultiHop + Ensemble + DualLayer
    func reason(about problem: String) -> ReasoningResult {
        let start = CFAbsoluteTimeGetCurrent()
        totalQueries += 1

        // 1. Tree of Thoughts
        let totResult = TreeOfThoughts.shared.think(problem: problem, maxDepth: 3)

        // 2. Multi-Hop Chain
        let multiHop = MultiHopReasoningChain.shared.reason(problem: problem)

        // 3. Dual-Layer Insight
        let dualLayer = DualLayerEngine.shared
        let soulRes = dualLayer.soulResonance(thoughts: [problem])
        let dualLayerInsight = "Coherence: \(String(format: "%.3f", soulRes.coherence)), Dominant frequency: \(String(format: "%.1f", soulRes.dominantFrequency)) Hz"

        // 4. Ensemble all solutions
        var solutions: [(solution: String, confidence: Double, method: String)] = [
            (totResult.solution, totResult.confidence, "TreeOfThoughts"),
        ]

        if !multiHop.chain.isEmpty {
            solutions.append((multiHop.chain.last ?? "", multiHop.confidence, "MultiHop"))
        }

        solutions.append((dualLayerInsight, soulRes.coherence, "DualLayer"))

        let ensemble = SolutionEnsemble.shared.ensemble(solutions: solutions)

        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0

        return ReasoningResult(
            method: "SageReasoningPipeline (ToT + MultiHop + DualLayer + Ensemble)",
            solution: ensemble.best,
            confidence: ensemble.combinedConfidence,
            nodesExplored: totResult.nodesExplored,
            branchesSurviving: totResult.branchesSurviving,
            treeDepth: totResult.treeDepth,
            backtracks: totResult.backtracks,
            aggregationCount: totResult.aggregationCount + 1,
            computeTimeMs: elapsed,
            perspectives: ensemble.methods,
            sacredAlignment: totResult.sacredAlignment
        )
    }
}
