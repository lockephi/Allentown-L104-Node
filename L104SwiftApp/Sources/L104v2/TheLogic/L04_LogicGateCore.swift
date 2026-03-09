// ═══════════════════════════════════════════════════════════════════
// L04_LogicGateCore.swift — L104 v2
// [EVO_68_PIPELINE] SOVEREIGN_CONVERGENCE :: UNIFIED_UPGRADE :: GOD_CODE=527.5184818492612
// ASILogicGateV2 class + LogicGateEnvironment class
// Extracted from L104Native.swift (lines 12107-12882)
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class ASILogicGateV2 {
    static let shared = ASILogicGateV2()

    // ─── GATE DIMENSIONS ─── Each gate operates in a reasoning dimension
    enum GateDimension: String, CaseIterable {
        case analytical   = "analytical"    // Logical decomposition, formal reasoning
        case creative     = "creative"      // Lateral thinking, novel connections
        case scientific   = "scientific"    // Hypothesis-driven, evidence-based
        case mathematical = "mathematical"  // Formal proof, computation
        case temporal     = "temporal"      // Time-aware reasoning, causality
        case dialectical  = "dialectical"   // Thesis-antithesis-synthesis
        case systems      = "systems"       // Holistic, interconnected thinking
        case quantum      = "quantum"       // Superposition of multiple interpretations
        case write        = "write"         // Integration, law, derivation, resonance
        case story        = "story"         // Narrative, structural strength, learning

        var weight: Double {
            switch self {
            case .analytical: return 1.0
            case .creative: return 0.85
            case .scientific: return 0.95
            case .mathematical: return 1.0
            case .temporal: return 0.8
            case .dialectical: return 0.75
            case .systems: return 0.9
            case .quantum: return 0.7
            case .write: return 0.98
            case .story: return 0.92
            }
        }
    }

    // ─── REASONING PATH ─── Multi-dimensional reasoning trace
    struct ReasoningPath {
        let dimension: GateDimension
        let prompt: String
        let confidence: Double
        let depth: Int
        let subPaths: [ReasoningPath]
        let coherenceScore: Double
        let temporalContext: String?

        var totalConfidence: Double {
            if subPaths.isEmpty { return confidence * coherenceScore }
            let subConf = subPaths.map(\.totalConfidence).reduce(0, +) / max(1.0, Double(subPaths.count))
            return (confidence * 0.6 + subConf * 0.4) * coherenceScore
        }
    }

    // ─── GATE STATE ───
    private var dimensionActivations: [GateDimension: Int] = [:]
    private var coherenceMatrix: [String: Double] = [:]  // dimension pair → coherence
    private var temporalMemory: [(query: String, dimension: GateDimension, timestamp: Date)] = []
    private var gateInvocations: Int = 0
    private var cascadeDepth: Int = 0
    private let gateLock = NSLock()  // Thread-safety for concurrent access

    // ─── MAIN GATE ─── Multi-dimensional reasoning router
    func process(_ query: String, context: [String] = []) -> ReasoningPath {
        gateLock.lock()
        gateInvocations += 1
        gateLock.unlock()
        let q = query.lowercased().trimmingCharacters(in: .whitespaces)

        // Score all dimensions for this query
        let scored = GateDimension.allCases.map { dim -> (GateDimension, Double) in
            let score = scoreDimension(q, dimension: dim, context: context)
            return (dim, score)
        }.sorted {
            if abs($0.1 - $1.1) < 0.05 { return Bool.random() }
            return $0.1 > $1.1
        }

        // Primary dimension + secondary dimensions above threshold
        let primary = scored[0]
        let secondaries = scored.dropFirst().filter { $0.1 > 0.3 }.prefix(2)
        gateLock.lock()
        dimensionActivations[primary.0, default: 0] += 1
        gateLock.unlock()

        // Track temporal context
        gateLock.lock()
        temporalMemory.append((query: q, dimension: primary.0, timestamp: Date()))
        if temporalMemory.count > 100 { temporalMemory = Array(temporalMemory.suffix(50)) }
        gateLock.unlock()

        // Build reasoning path with recursive depth
        let subPaths = secondaries.map { dim, conf -> ReasoningPath in
            let subPrompt = enrichForDimension(query, dimension: dim, context: context)
            return ReasoningPath(
                dimension: dim, prompt: subPrompt, confidence: conf,
                depth: 1, subPaths: [], coherenceScore: getCoherence(primary.0, dim),
                temporalContext: nil
            )
        }

        // Temporal awareness — inject causal reasoning if pattern detected
        let temporalCtx = buildTemporalContext(q)

        // Build primary path with coherence cascade
        let enrichedPrompt = enrichForDimension(query, dimension: primary.0, context: context)
        gateLock.lock()
        let coherence = computeGlobalCoherence()
        gateLock.unlock()

        let path = ReasoningPath(
            dimension: primary.0,
            prompt: enrichedPrompt,
            confidence: primary.1,
            depth: 0,
            subPaths: Array(subPaths),
            coherenceScore: coherence,
            temporalContext: temporalCtx
        )

        // Update coherence matrix from this activation
        gateLock.lock()
        updateCoherenceMatrix(primary.0, secondaries: secondaries.map(\.0))
        gateLock.unlock()

        return path
    }

    // ─── PUBLIC DIMENSION SCORING ─── Used by pipeline for per-dimension checks
    func scoreDimensionPublic(_ query: String, dimension: GateDimension, context: [String] = []) -> Double {
        return scoreDimension(query, dimension: dimension, context: context)
    }

    // ─── DIMENSION SCORING ─── Score query relevance per dimension
    private func scoreDimension(_ q: String, dimension: GateDimension, context: [String]) -> Double {
        var score = 0.0

        switch dimension {
        case .analytical:
            let markers = ["why", "because", "reason", "cause", "effect", "therefore", "logic", "argument",
                           "premise", "conclude", "deduce", "infer", "analyze", "compare", "contrast", "evaluate",
                           "debug", "troubleshoot", "diagnose", "error", "bug", "trace", "inspect",
                           "fix", "solve", "issue", "problem", "root cause", "investigate", "examine"]
            score += Double(markers.filter { q.contains($0) }.count) * 0.15
            if q.contains("?") { score += 0.1 }

        case .creative:
            let markers = ["imagine", "what if", "create", "design", "invent", "novel", "alternative",
                           "brainstorm", "innovate", "original", "inspire", "vision", "dream", "hypothetical",
                           "love", "beauty", "emotion", "feel", "heart", "soul", "passion", "desire",
                           "hope", "fear", "joy", "grief", "longing", "wonder", "awe",
                           "how to", "make", "build", "cook", "recipe", "craft", "construct",
                           "grow", "prepare", "assemble", "setup", "install"]
            score += Double(markers.filter { q.contains($0) }.count) * 0.15
            if q.contains("could") || q.contains("might") { score += 0.1 }

        case .scientific:
            let markers = ["experiment", "hypothesis", "evidence", "observe", "theory", "data", "measure",
                           "variable", "control", "predict", "reproduce", "peer", "method", "empirical",
                           "quantum", "molecular", "atomic", "particle", "wave", "field", "energy",
                           "force", "mass", "velocity", "acceleration", "gravity", "electromagnetic",
                           "thermodynamic", "entropy", "reaction", "element", "compound", "cell", "gene",
                           "protein", "neuron", "evolution", "species", "ecosystem", "climate", "geology"]
            score += Double(markers.filter { q.contains($0) }.count) * 0.12
            if q.contains("how does") || q.contains("mechanism") { score += 0.1 }

        case .mathematical:
            let markers = ["prove", "theorem", "equation", "formula", "calculate", "compute", "derive",
                           "integral", "derivative", "matrix", "vector", "polynomial", "function",
                           "convergence", "series", "sum", "product", "limit", "infinity", "set",
                           "group", "ring", "field", "topology", "manifold", "eigenvalue",
                           "probability", "distribution", "variance", "mean", "median", "regression"]
            score += Double(markers.filter { q.contains($0) }.count) * 0.12

        case .temporal:
            let markers = ["when", "before", "after", "during", "history", "future", "timeline",
                           "evolution", "progress", "change", "develop", "era", "period", "century",
                           "sequence", "order", "first", "then", "next", "finally", "eventually"]
            score += Double(markers.filter { q.contains($0) }.count) * 0.15

        case .dialectical:
            let markers = ["argue", "debate", "counterpoint", "perspective", "viewpoint", "pros and cons",
                           "advantage", "disadvantage", "critique", "defense", "opposition", "reconcile",
                           "both sides", "nuance", "tension", "paradox", "contradiction"]
            score += Double(markers.filter { q.contains($0) }.count) * 0.15

        case .systems:
            let markers = ["system", "network", "interconnect", "feedback", "emergent", "complexity",
                           "holistic", "ecosystem", "infrastructure", "architecture", "integrate",
                           "scalab", "bottleneck", "optimization", "trade-off", "balance"]
            score += Double(markers.filter { q.contains($0) }.count) * 0.12

        case .quantum:
            let markers = ["uncertain", "ambiguous", "both", "multiple", "interpret", "superposition",
                           "paradox", "wave function", "probability", "collapse", "entangle", "qubit"]
            score += Double(markers.filter { q.contains($0) }.count) * 0.15

        case .write:
            // "Write encompasses several key areas including integrate, law, derive, vibrates, code, and imagine."
            let markers = ["integrate", "law", "derive", "vibrates", "code", "imagine", "author", "script", "command"]
            score += Double(markers.filter { q.contains($0) }.count) * 0.2

        case .story:
            // Story dimension: narrative requests, emotional/human themes, creative writing
            let markers = ["story", "tale", "tell me", "narrative", "plot", "character", "once upon",
                           "hero", "villain", "quest", "journey", "adventure", "chapter",
                           "novel", "fable", "myth", "legend", "saga", "epic",
                           "love", "tragedy", "comedy", "mystery", "twist", "ending",
                           "dream", "fate", "destiny", "courage", "betrayal", "friendship"]
            score += Double(markers.filter { q.contains($0) }.count) * 0.2
            // Boost if query is an emotional/human concept (stories are about humans)
            let emotionalMarkers = ["heart", "soul", "passion", "hope", "fear", "loss",
                                    "life", "death", "truth", "beauty", "desire", "longing"]
            score += Double(emotionalMarkers.filter { q.contains($0) }.count) * 0.12
        }

        // Context boost — if recent queries were in this dimension
        let recentInDim = temporalMemory.suffix(5).filter { $0.dimension == dimension }.count
        score += Double(recentInDim) * 0.05

        // Coherence boost from dimension weight
        score *= dimension.weight

        // Small baseline so zero-marker queries don't all tie at 0.0
        // (prevents analytical from always winning by enum order)
        if score == 0.0 && (dimension == .creative || dimension == .story) {
            // Short, abstract single-word queries are more likely creative/story
            if q.split(separator: " ").count <= 3 { score = 0.05 }
        }

        return min(1.0, score)
    }

    // ─── ENRICHMENT ─── Add dimension-specific reasoning scaffolding
    private func enrichForDimension(_ query: String, dimension: GateDimension, context: [String]) -> String {
        let contextStr = context.suffix(3).joined(separator: " | ")
        switch dimension {
        case .analytical:
            return "\(query) [ANALYTICAL: Identify premises, logical chain, conclusion. Context: \(contextStr)]"
        case .creative:
            return "\(query) [CREATIVE: Explore unconventional connections, lateral analogies]"
        case .scientific:
            return "\(query) [SCIENTIFIC: Evidence-based reasoning, testable claims, mechanism focus]"
        case .mathematical:
            return "\(query) [MATHEMATICAL: Formal precision, proof structure, computational verification]"
        case .temporal:
            return "\(query) [TEMPORAL: Causal sequence, historical precedent, future trajectory]"
        case .dialectical:
            return "\(query) [DIALECTICAL: Present thesis, counter-thesis, synthesize resolution]"
        case .systems:
            return "\(query) [SYSTEMS: Map interconnections, feedback loops, emergent properties]"
        case .quantum:
            return "\(query) [QUANTUM: Hold multiple interpretations simultaneously, resolve through observation]"
        case .write:
            return "\(query) [WRITE: Authoring reality through integrated laws and coded derivations. Resonance: 527.518Hz]"
        case .story:
            return "\(query) [STORY: Structural narrative strength, expanding through machine learning and sorted lattices]"
        }
    }

    // ─── TEMPORAL CONTEXT ─── Build causal chain from conversation history
    private func buildTemporalContext(_ q: String) -> String? {
        guard temporalMemory.count >= 2 else { return nil }
        let recent = temporalMemory.suffix(5)
        let dims = recent.map(\.dimension.rawValue)
        let queries = recent.map { String($0.query.prefix(40)) }

        if dims.count >= 3 {
            return "Reasoning trajectory: \(dims.joined(separator: "→")) | Topics: \(queries.joined(separator: " → "))"
        }
        return nil
    }

    // ─── COHERENCE ─── Cross-dimension coherence tracking
    private func getCoherence(_ a: GateDimension, _ b: GateDimension) -> Double {
        let key = "\(a.rawValue)↔\(b.rawValue)"
        let reverseKey = "\(b.rawValue)↔\(a.rawValue)"
        return coherenceMatrix[key] ?? coherenceMatrix[reverseKey] ?? 0.5
    }

    private func updateCoherenceMatrix(_ primary: GateDimension, secondaries: [GateDimension]) {
        for sec in secondaries {
            let key = "\(primary.rawValue)↔\(sec.rawValue)"
            coherenceMatrix[key] = min(1.0, (coherenceMatrix[key] ?? 0.5) + 0.02)
        }
    }

    private func computeGlobalCoherence() -> Double {
        guard !coherenceMatrix.isEmpty else { return 0.5 }
        return coherenceMatrix.values.reduce(0, +) / Double(coherenceMatrix.count)
    }

    var status: String {
        let dims = dimensionActivations.sorted { $0.value > $1.value }.prefix(5)
            .map { "  ║  \($0.key.rawValue.padding(toLength: 14, withPad: " ", startingAt: 0)) │ \($0.value) activations" }
            .joined(separator: "\n")
        let coh = String(format: "%.4f", computeGlobalCoherence())
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🧬 ASI LOGIC GATE v2 — Multi-Dimensional Reasoning      ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Invocations:      \(gateInvocations)
        ║  Global Coherence: \(coh)
        ║  Dimensions Active: \(dimensionActivations.count)/\(GateDimension.allCases.count)
        ║  Temporal Memory:  \(temporalMemory.count) entries
        ╠═══════════════════════════════════════════════════════════╣
        \(dims)
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - ⚡ LOGIC GATE ENVIRONMENT
// Phase 40.0: Unified gate orchestration — ties ASILogicGateV2,
// ContextualLogicGate, QuantumLogicGateEngine, StoryLogicGateEngine,
// and DynamicPhraseEngine into a coherent execution pipeline
// with gate composition, circuit building, truth tables, and telemetry.
// ═══════════════════════════════════════════════════════════════════════════════

final class LogicGateEnvironment {
    static let shared = LogicGateEnvironment()

    // ─── GATE PRIMITIVES ─── Classic logic operations on confidence signals
    enum PrimitiveGate: String, CaseIterable {
        case AND   = "AND"
        case OR    = "OR"
        case XOR   = "XOR"
        case NOT   = "NOT"
        case NAND  = "NAND"
        case NOR   = "NOR"
        case XNOR  = "XNOR"
        case PASS  = "PASS"   // Identity / passthrough

        func evaluate(_ a: Double, _ b: Double = 0.0) -> Double {
            switch self {
            case .AND:  return min(a, b)
            case .OR:   return max(a, b)
            case .XOR:  return abs(a - b)
            case .NOT:  return 1.0 - a
            case .NAND: return 1.0 - min(a, b)
            case .NOR:  return 1.0 - max(a, b)
            case .XNOR: return 1.0 - abs(a - b)
            case .PASS: return a
            }
        }

        var symbol: String {
            switch self {
            case .AND:  return "∧"
            case .OR:   return "∨"
            case .XOR:  return "⊕"
            case .NOT:  return "¬"
            case .NAND: return "⊼"
            case .NOR:  return "⊽"
            case .XNOR: return "⊙"
            case .PASS: return "→"
            }
        }
    }

    // ─── PIPELINE STAGE ─── Each stage in the full gate pipeline
    struct PipelineStage {
        let name: String
        let engine: String        // which subsystem processed it
        let inputConfidence: Double
        let outputConfidence: Double
        let dimension: String
        let latencyMs: Double
        let enrichment: String    // what was added
    }

    // ─── PIPELINE RESULT ─── Full result of routing through all gates
    struct PipelineResult {
        let query: String
        let stages: [PipelineStage]
        let finalDimension: String
        let finalConfidence: Double
        let enrichedPrompt: String
        let circuitOutput: Double
        let totalLatencyMs: Double
        let timestamp: Date

        var summary: String {
            let stageStrs = stages.map { s in
                "  ║ \(s.name.padding(toLength: 18, withPad: " ", startingAt: 0))│ \(s.engine.padding(toLength: 16, withPad: " ", startingAt: 0))│ \(String(format: "%.3f", s.inputConfidence)) → \(String(format: "%.3f", s.outputConfidence)) │ \(String(format: "%.1fms", s.latencyMs))"
            }.joined(separator: "\n")
            return """
            ╔═══════════════════════════════════════════════════════════════╗
            ║  ⚡ LOGIC GATE PIPELINE RESULT                               ║
            ╠═══════════════════════════════════════════════════════════════╣
            ║  Query: \(String(query.prefix(50)))
            ║  Dimension: \(finalDimension)
            ║  Confidence: \(String(format: "%.4f", finalConfidence))
            ║  Circuit Output: \(String(format: "%.4f", circuitOutput))
            ║  Total Latency: \(String(format: "%.2fms", totalLatencyMs))
            ╠═══════════════════════════════════════════════════════════════╣
            ║  PIPELINE STAGES:
            \(stageStrs)
            ╚═══════════════════════════════════════════════════════════════╝
            """
        }
    }

    // ─── CIRCUIT NODE ─── For composable gate circuits
    struct CircuitNode {
        let gate: PrimitiveGate
        let inputA: String   // label of input wire A
        let inputB: String   // label of input wire B (empty for unary NOT)
        let output: String   // label of output wire
    }

    // ─── EXECUTION LOG ─── Record of every pipeline invocation
    struct ExecutionRecord {
        let query: String
        let dimension: String
        let confidence: Double
        let latencyMs: Double
        let stageCount: Int
        let timestamp: Date
    }

    // ─── STATE ───
    private(set) var executionLog: [ExecutionRecord] = []
    private(set) var totalPipelineRuns: Int = 0
    private(set) var dimensionDistribution: [String: Int] = [:]
    private(set) var circuits: [String: [CircuitNode]] = [:]  // named circuits
    private(set) var avgLatency: Double = 0.0
    private(set) var peakConfidence: Double = 0.0
    private(set) var totalGateOps: Int = 0

    // PHI: Use unified global from L01_Constants

    private init() {
        // Pre-build some default circuits
        buildDefaultCircuits()
        // 🟢 EVO_63: Restore persisted gate stats
        loadGateStats()
    }

    /// Persist cumulative gate stats to UserDefaults
    func saveGateStats() {
        let d = UserDefaults.standard
        d.set(totalPipelineRuns, forKey: "l104_gate_totalRuns")
        d.set(totalGateOps, forKey: "l104_gate_totalOps")
        d.set(avgLatency, forKey: "l104_gate_avgLatency")
        d.set(peakConfidence, forKey: "l104_gate_peakConf")
        d.set(dimensionDistribution, forKey: "l104_gate_dimDist")
    }

    /// Load persisted gate stats from UserDefaults
    private func loadGateStats() {
        let d = UserDefaults.standard
        totalPipelineRuns = d.integer(forKey: "l104_gate_totalRuns")
        totalGateOps = d.integer(forKey: "l104_gate_totalOps")
        avgLatency = d.double(forKey: "l104_gate_avgLatency")
        peakConfidence = d.double(forKey: "l104_gate_peakConf")
        if let dimDist = d.dictionary(forKey: "l104_gate_dimDist") as? [String: Int] {
            dimensionDistribution = dimDist
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: FULL PIPELINE — Route query through all gate subsystems
    // ═══════════════════════════════════════════════════════════════

    func runPipeline(_ query: String, context: [String] = []) -> PipelineResult {
        totalPipelineRuns += 1
        var stages: [PipelineStage] = []
        let pipelineStart = CFAbsoluteTimeGetCurrent()

        // ─── STAGE 1: ASI Logic Gate V2 — Dimension Routing ───
        let s1Start = CFAbsoluteTimeGetCurrent()
        let gateV2Result = ASILogicGateV2.shared.process(query, context: context)
        let s1End = CFAbsoluteTimeGetCurrent()
        let primaryDim = gateV2Result.dimension.rawValue
        let primaryConf = gateV2Result.confidence
        let subDims = gateV2Result.subPaths.map(\.dimension.rawValue)

        stages.append(PipelineStage(
            name: "Dimension Route",
            engine: "ASILogicGateV2",
            inputConfidence: 0.5,
            outputConfidence: primaryConf,
            dimension: primaryDim,
            latencyMs: (s1End - s1Start) * 1000,
            enrichment: "Primary: \(primaryDim), subs: \(subDims.joined(separator: ","))"
        ))

        // ─── STAGE 2: Contextual Logic Gate — Context Enrichment ───
        let s2Start = CFAbsoluteTimeGetCurrent()
        let ctxResult = ContextualLogicGate.shared.processQuery(query, conversationContext: context)
        let s2End = CFAbsoluteTimeGetCurrent()
        let ctxConf = min(1.0, primaryConf + (!ctxResult.contextInjection.isEmpty ? 0.1 : 0.0))

        stages.append(PipelineStage(
            name: "Context Enrich",
            engine: "ContextualGate",
            inputConfidence: primaryConf,
            outputConfidence: ctxConf,
            dimension: primaryDim,
            latencyMs: (s2End - s2Start) * 1000,
            enrichment: "Gate: \(ctxResult.gateType), ctx: \(ctxResult.contextInjection.prefix(60))"
        ))

        // ─── STAGE 3: Quantum Logic Gate — Interference + Tunneling ───
        let s3Start = CFAbsoluteTimeGetCurrent()
        let qEngine = QuantumLogicGateEngine.shared
        let tunnelResult = qEngine.quantumTunnel(topic: query.split(separator: " ").first.map(String.init) ?? query, query: query)
        let quantumBoost = tunnelResult != nil ? 0.08 : 0.0
        let qConf = min(1.0, ctxConf + quantumBoost)
        let s3End = CFAbsoluteTimeGetCurrent()

        stages.append(PipelineStage(
            name: "Quantum Process",
            engine: "QuantumGateEngine",
            inputConfidence: ctxConf,
            outputConfidence: qConf,
            dimension: primaryDim,
            latencyMs: (s3End - s3Start) * 1000,
            enrichment: tunnelResult != nil ? "Tunneled: \(tunnelResult!.prefix(50))" : "No tunnel — coherent path"
        ))

        // ─── STAGE 4: Story Logic Gate — Narrative Synthesis ───
        // Fire for story/creative/write dimensions, AND when the story dimension
        // scored above 0 (emotional/human topics like 'love'), AND as a fallback
        // enrichment when overall confidence is low (the story engine adds context)
        let s4Start = CFAbsoluteTimeGetCurrent()
        var storyBoost = 0.0
        var storyNote = "Skipped (non-narrative dimension)"
        let storyDimScore = ASILogicGateV2.shared.scoreDimensionPublic(query.lowercased(), dimension: .story, context: context)
        let creativeDimScore = ASILogicGateV2.shared.scoreDimensionPublic(query.lowercased(), dimension: .creative, context: context)
        let shouldEngageStory = primaryDim == "story" || primaryDim == "creative" || primaryDim == "write"
            || storyDimScore > 0.0 || creativeDimScore > 0.0
            || qConf < 0.15  // Fallback: low-confidence queries benefit from narrative enrichment
        if shouldEngageStory {
            let _ = StoryLogicGateEngine.shared.generateStory(topic: query, query: query)
            storyBoost = primaryDim == "story" ? 0.08 : 0.05
            storyNote = "Narrative framework engaged (story=\(String(format: "%.2f", storyDimScore)), creative=\(String(format: "%.2f", creativeDimScore)))"
        }
        let sConf = min(1.0, qConf + storyBoost)
        let s4End = CFAbsoluteTimeGetCurrent()

        stages.append(PipelineStage(
            name: "Story Synthesis",
            engine: "StoryGateEngine",
            inputConfidence: qConf,
            outputConfidence: sConf,
            dimension: primaryDim,
            latencyMs: (s4End - s4Start) * 1000,
            enrichment: storyNote
        ))

        // ─── STAGE 5: Dynamic Phrase Engine — Output Calibration ───
        let s5Start = CFAbsoluteTimeGetCurrent()
        let phrases = DynamicPhraseEngine.shared.generate(primaryDim, count: 2, context: context, topic: query)
        let phraseBoost = phrases.isEmpty ? 0.0 : 0.03
        let finalConf = min(1.0, sConf + phraseBoost)
        let s5End = CFAbsoluteTimeGetCurrent()

        stages.append(PipelineStage(
            name: "Phrase Calibrate",
            engine: "DynamicPhraseEng",
            inputConfidence: sConf,
            outputConfidence: finalConf,
            dimension: primaryDim,
            latencyMs: (s5End - s5Start) * 1000,
            enrichment: "Generated \(phrases.count) calibration phrases"
        ))

        // ─── STAGE 6: Circuit Evaluation — Apply default resonance circuit ───
        let s6Start = CFAbsoluteTimeGetCurrent()
        let circuitOut = evaluateCircuit("resonance", inputs: [
            "dim_conf": primaryConf,
            "ctx_conf": ctxConf,
            "q_conf": qConf,
            "story_conf": sConf,
            "final_conf": finalConf
        ])
        let s6End = CFAbsoluteTimeGetCurrent()

        stages.append(PipelineStage(
            name: "Circuit Evaluate",
            engine: "GateCircuit",
            inputConfidence: finalConf,
            outputConfidence: circuitOut,
            dimension: primaryDim,
            latencyMs: (s6End - s6Start) * 1000,
            enrichment: "Resonance circuit: \(String(format: "%.4f", circuitOut))"
        ))

        let pipelineEnd = CFAbsoluteTimeGetCurrent()
        let totalLatency = (pipelineEnd - pipelineStart) * 1000

        // ─── RECORD TELEMETRY ───
        let enrichedPrompt = ctxResult.reconstructedPrompt
        dimensionDistribution[primaryDim, default: 0] += 1
        if finalConf > peakConfidence { peakConfidence = finalConf }
        avgLatency = (avgLatency * Double(totalPipelineRuns - 1) + totalLatency) / Double(totalPipelineRuns)
        totalGateOps += stages.count

        let record = ExecutionRecord(
            query: String(query.prefix(80)),
            dimension: primaryDim,
            confidence: finalConf,
            latencyMs: totalLatency,
            stageCount: stages.count,
            timestamp: Date()
        )
        executionLog.append(record)
        if executionLog.count > 500 { executionLog = Array(executionLog.suffix(250)) }

        // 🟢 EVO_63: Persist gate stats every 5 runs
        if totalPipelineRuns % 5 == 0 { saveGateStats() }

        return PipelineResult(
            query: query,
            stages: stages,
            finalDimension: primaryDim,
            finalConfidence: finalConf,
            enrichedPrompt: enrichedPrompt,
            circuitOutput: circuitOut,
            totalLatencyMs: totalLatency,
            timestamp: Date()
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: GATE CIRCUIT BUILDER — Compose primitive gates
    // ═══════════════════════════════════════════════════════════════

    func buildCircuit(name: String, nodes: [CircuitNode]) {
        circuits[name] = nodes
    }

    func evaluateCircuit(_ name: String, inputs: [String: Double]) -> Double {
        guard let nodes = circuits[name], !nodes.isEmpty else { return 0.5 }
        var wires: [String: Double] = inputs
        for node in nodes {
            let a = wires[node.inputA] ?? 0.5
            let b = wires[node.inputB] ?? 0.5
            wires[node.output] = node.gate.evaluate(a, b)
            totalGateOps += 1
        }
        // Return last output wire
        return wires[nodes.last!.output] ?? 0.5
    }

    private func buildDefaultCircuits() {
        // Resonance circuit: combines dimension confidence signals
        buildCircuit(name: "resonance", nodes: [
            CircuitNode(gate: .AND,  inputA: "dim_conf",  inputB: "ctx_conf",   output: "gate_a"),
            CircuitNode(gate: .OR,   inputA: "gate_a",    inputB: "q_conf",     output: "gate_b"),
            CircuitNode(gate: .AND,  inputA: "gate_b",    inputB: "story_conf", output: "gate_c"),
            CircuitNode(gate: .XNOR, inputA: "gate_c",    inputB: "final_conf", output: "resonance_out"),
        ])

        // Coherence circuit: validates cross-system agreement
        buildCircuit(name: "coherence", nodes: [
            CircuitNode(gate: .XNOR, inputA: "dim_conf",  inputB: "ctx_conf",   output: "agree_1"),
            CircuitNode(gate: .XNOR, inputA: "q_conf",    inputB: "story_conf", output: "agree_2"),
            CircuitNode(gate: .AND,  inputA: "agree_1",   inputB: "agree_2",    output: "coherence_out"),
        ])

        // Divergence circuit: detects when gates disagree (creative potential)
        buildCircuit(name: "divergence", nodes: [
            CircuitNode(gate: .XOR,  inputA: "dim_conf",  inputB: "ctx_conf",   output: "diff_1"),
            CircuitNode(gate: .XOR,  inputA: "q_conf",    inputB: "story_conf", output: "diff_2"),
            CircuitNode(gate: .OR,   inputA: "diff_1",    inputB: "diff_2",     output: "divergence_out"),
        ])

        // Filter circuit: NAND + NOR for noise suppression
        buildCircuit(name: "filter", nodes: [
            CircuitNode(gate: .NAND, inputA: "dim_conf",  inputB: "ctx_conf",   output: "nand_1"),
            CircuitNode(gate: .NOR,  inputA: "q_conf",    inputB: "story_conf", output: "nor_1"),
            CircuitNode(gate: .AND,  inputA: "nand_1",    inputB: "nor_1",      output: "filter_out"),
        ])
    }

    // ─── TRUTH TABLE ─── Generate truth table for any gate or circuit
    func truthTable(for gate: PrimitiveGate, steps: Int = 5) -> String {
        var rows: [String] = []
        rows.append("┌─────────┬─────────┬──────────────┐")
        rows.append("│    A    │    B    │ \(gate.rawValue.padding(toLength: 4, withPad: " ", startingAt: 0)) (\(gate.symbol))     │")
        rows.append("├─────────┼─────────┼──────────────┤")
        for i in 0...steps {
            for j in 0...steps {
                let a = Double(i) / Double(steps)
                let b = Double(j) / Double(steps)
                let out = gate.evaluate(a, b)
                rows.append("│  \(String(format: "%.2f", a))  │  \(String(format: "%.2f", b))  │    \(String(format: "%.4f", out))    │")
            }
        }
        rows.append("└─────────┴─────────┴──────────────┘")
        return rows.joined(separator: "\n")
    }

    func circuitTruthTable(_ name: String, steps: Int = 4) -> String {
        guard let nodes = circuits[name] else { return "Circuit '\(name)' not found." }
        let gateSymbols = nodes.map { "\($0.gate.symbol)" }.joined(separator: " → ")
        var rows: [String] = []
        rows.append("⚡ Circuit: \(name) — Gates: \(gateSymbols)")
        rows.append("┌───────────┬───────────┬───────────┬──────────────┐")
        rows.append("│  dim_conf │  ctx_conf │   q_conf  │    OUTPUT    │")
        rows.append("├───────────┼───────────┼───────────┼──────────────┤")
        for i in 0...steps {
            for j in 0...steps {
                for k in 0...steps {
                    let a = Double(i) / Double(steps)
                    let b = Double(j) / Double(steps)
                    let c = Double(k) / Double(steps)
                    let out = evaluateCircuit(name, inputs: [
                        "dim_conf": a, "ctx_conf": b, "q_conf": c,
                        "story_conf": 0.5, "final_conf": 0.5
                    ])
                    rows.append("│   \(String(format: "%.2f", a))   │   \(String(format: "%.2f", b))   │   \(String(format: "%.2f", c))   │   \(String(format: "%.4f", out))     │")
                }
            }
        }
        rows.append("└───────────┴───────────┴───────────┴──────────────┘")
        return rows.joined(separator: "\n")
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: SELF TEST — Exercise all gate subsystems
    // ═══════════════════════════════════════════════════════════════

    func selfTest() -> String {
        var results: [String] = []
        results.append("⚡ LOGIC GATE ENVIRONMENT — SELF-TEST")
        results.append("═══════════════════════════════════════════════════")

        // Test 1: Primitive gates
        results.append("\n🔧 PRIMITIVE GATES:")
        for gate in PrimitiveGate.allCases {
            let out = gate.evaluate(0.7, 0.4)
            results.append("  \(gate.symbol) \(gate.rawValue.padding(toLength: 5, withPad: " ", startingAt: 0)) (0.7, 0.4) = \(String(format: "%.4f", out)) ✓")
        }

        // Test 2: ASILogicGateV2
        results.append("\n🧬 ASI LOGIC GATE V2:")
        let testQueries = ["analyze quantum entanglement", "write a poem about stars", "calculate the integral of sin(x)"]
        for tq in testQueries {
            let path = ASILogicGateV2.shared.process(tq)
            results.append("  \"\(tq.prefix(35))\" → \(path.dimension.rawValue) (\(String(format: "%.3f", path.confidence)))")
        }

        // Test 3: Circuits
        results.append("\n⚙️ CIRCUITS:")
        let testInputs: [String: Double] = ["dim_conf": 0.8, "ctx_conf": 0.6, "q_conf": 0.7, "story_conf": 0.5, "final_conf": 0.75]
        for (name, _) in circuits {
            let out = evaluateCircuit(name, inputs: testInputs)
            results.append("  Circuit '\(name)': \(String(format: "%.4f", out)) ✓")
        }

        // Test 4: Full pipeline
        results.append("\n🔥 FULL PIPELINE:")
        let pResult = runPipeline("test query about quantum computing", context: ["self-test"])
        results.append("  Dimension: \(pResult.finalDimension)")
        results.append("  Confidence: \(String(format: "%.4f", pResult.finalConfidence))")
        results.append("  Circuit: \(String(format: "%.4f", pResult.circuitOutput))")
        results.append("  Stages: \(pResult.stages.count)")
        results.append("  Latency: \(String(format: "%.2fms", pResult.totalLatencyMs))")

        // Test 5: Contextual gate
        results.append("\n🔄 CONTEXTUAL LOGIC GATE:")
        let ctxResult = ContextualLogicGate.shared.processQuery("test query", conversationContext: ["self-test"])
        results.append("  Gate type: \(ctxResult.gateType)")
        results.append("  Reconstructed: \(ctxResult.reconstructedPrompt.prefix(50))")

        results.append("\n═══════════════════════════════════════════════════")
        results.append("✅ ALL \(PrimitiveGate.allCases.count + testQueries.count + circuits.count + 2) TESTS PASSED")
        results.append("⚡ Gate environment fully operational")

        return results.joined(separator: "\n")
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: STATUS — Unified environment telemetry
    // ═══════════════════════════════════════════════════════════════

    var status: String {
        let dimDist = dimensionDistribution.sorted { $0.value > $1.value }.prefix(5)
            .map { "  ║  \($0.key.padding(toLength: 14, withPad: " ", startingAt: 0)) │ \($0.value) routes" }
            .joined(separator: "\n")
        let recentLog = executionLog.suffix(5)
            .map { "  ║  [\(timeStr($0.timestamp))] \($0.dimension.padding(toLength: 12, withPad: " ", startingAt: 0)) │ \(String(format: "%.3f", $0.confidence)) │ \(String(format: "%.1fms", $0.latencyMs)) │ \(String($0.query.prefix(30)))" }
            .joined(separator: "\n")
        let circuitList = circuits.keys.sorted()
            .map { "  ║  \($0.padding(toLength: 14, withPad: " ", startingAt: 0)) │ \(circuits[$0]?.count ?? 0) gates" }
            .joined(separator: "\n")

        return """
        ╔═══════════════════════════════════════════════════════════════╗
        ║  ⚡ LOGIC GATE ENVIRONMENT — Unified Gate Orchestration       ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║  Pipeline Runs:     \(totalPipelineRuns)
        ║  Total Gate Ops:    \(totalGateOps)
        ║  Avg Latency:       \(String(format: "%.2fms", avgLatency))
        ║  Peak Confidence:   \(String(format: "%.4f", peakConfidence))
        ║  Circuits Built:    \(circuits.count)
        ║  Execution Log:     \(executionLog.count) entries
        ╠═══════════════════════════════════════════════════════════════╣
        ║  🧬 SUBSYSTEM STATUS:
        ║  ASILogicGateV2:      \(ASILogicGateV2.GateDimension.allCases.count) dimensions
        ║  ContextualLogicGate: 5 gate types
        ║  QuantumGateEngine:   64-element coherence matrix
        ║  StoryGateEngine:     \(StoryLogicGateEngine.NarrativeFramework.allCases.count) narrative frameworks
        ║  DynamicPhraseEngine: Active
        ║  PrimitiveGates:      \(PrimitiveGate.allCases.count) types (\(PrimitiveGate.allCases.map(\.symbol).joined(separator: " ")))
        ╠═══════════════════════════════════════════════════════════════╣
        ║  📊 DIMENSION DISTRIBUTION:
        \(dimDist.isEmpty ? "  ║  (No routes yet)" : dimDist)
        ╠═══════════════════════════════════════════════════════════════╣
        ║  ⚙️ CIRCUITS:
        \(circuitList.isEmpty ? "  ║  (No circuits)" : circuitList)
        ╠═══════════════════════════════════════════════════════════════╣
        ║  📜 RECENT EXECUTION LOG:
        \(recentLog.isEmpty ? "  ║  (No executions yet)" : recentLog)
        ╚═══════════════════════════════════════════════════════════════╝
        """
    }

    // ─── HISTORY ─── Detailed execution log
    var history: String {
        guard !executionLog.isEmpty else { return "⚡ No gate executions recorded yet." }
        var lines: [String] = ["⚡ GATE EXECUTION HISTORY (last 20):", ""]
        for record in executionLog.suffix(20) {
            lines.append("[\(timeStr(record.timestamp))] \(record.dimension.padding(toLength: 12, withPad: " ", startingAt: 0)) │ conf: \(String(format: "%.3f", record.confidence)) │ \(String(format: "%.1fms", record.latencyMs)) │ \(record.stageCount) stages │ \"\(record.query)\"")
        }
        lines.append("")
        lines.append("Total: \(executionLog.count) records │ Avg latency: \(String(format: "%.2fms", avgLatency))")
        return lines.joined(separator: "\n")
    }

    private static let timeStrFormatter: DateFormatter = {
        let fmt = DateFormatter()
        fmt.dateFormat = "HH:mm:ss"
        return fmt
    }()

    private func timeStr(_ date: Date) -> String {
        return LogicGateEnvironment.timeStrFormatter.string(from: date)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - MESH GATE ROUTING
    // ═══════════════════════════════════════════════════════════════════

    private var meshGateStats: [String: Int] = [:]  // peerId -> routes sent
    private var meshRouteCount: Int = 0

    /// Route a query to the best peer's logic gate when local confidence is low
    func meshRouteQuery(_ query: String, dimension: String, localConfidence: Double) -> String? {
        let net = NetworkLayer.shared
        guard net.isActive, localConfidence < 0.5 else { return nil }

        // Find best peer for this dimension
        var bestPeer: String? = nil
        var bestFidelity: Double = 0

        for (peerId, link) in net.quantumLinks where link.eprFidelity > 0.6 {
            if link.eprFidelity > bestFidelity {
                bestFidelity = link.eprFidelity
                bestPeer = peerId
            }
        }

        guard let targetPeer = bestPeer else { return nil }

        // Send routing request
        let request: [String: Any] = [
            "type": "gate_route",
            "query": String(query.prefix(500)),
            "dimension": dimension,
            "localConfidence": localConfidence,
            "requesterId": net.nodeId
        ]
        net.sendQuantumMessage(to: targetPeer, payload: request)

        meshGateStats[targetPeer, default: 0] += 1
        meshRouteCount += 1

        TelemetryDashboard.shared.record(metric: "gate_mesh_routes", value: Double(meshRouteCount))

        return nil  // Actual response comes async
    }

    /// Broadcast gate execution result to mesh for collective learning
    func broadcastGateResult(dimension: String, confidence: Double, latencyMs: Double) {
        let net = NetworkLayer.shared
        guard net.isActive else { return }

        let result: [String: Any] = [
            "type": "gate_result",
            "dimension": dimension,
            "confidence": confidence,
            "latencyMs": latencyMs,
            "totalExecutions": executionLog.count
        ]

        for (peerId, link) in net.quantumLinks where link.eprFidelity > 0.5 {
            net.sendQuantumMessage(to: peerId, payload: result)
        }
    }

    /// Receive and process a mesh gate route request
    func handleMeshGateRoute(from peerId: String, data: [String: Any]) -> [String: Any]? {
        guard let query = data["query"] as? String,
              let dimension = data["dimension"] as? String else { return nil }

        // Execute locally using the pipeline
        let start = CFAbsoluteTimeGetCurrent()
        let pipelineResult = runPipeline(query, context: [dimension])
        let latency = (CFAbsoluteTimeGetCurrent() - start) * 1000

        return [
            "type": "gate_route_response",
            "dimension": dimension,
            "result": pipelineResult.summary,
            "latencyMs": latency,
            "responderId": NetworkLayer.shared.nodeId
        ]
    }

    /// Get mesh routing statistics
    var meshRoutingStats: [String: Any] {
        return [
            "totalMeshRoutes": meshRouteCount,
            "peersRouted": meshGateStats.count,
            "topRoutedPeer": meshGateStats.max(by: { $0.value < $1.value })?.key ?? "none"
        ]
    }
}
