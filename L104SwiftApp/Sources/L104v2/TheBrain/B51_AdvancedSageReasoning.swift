import Accelerate
import Foundation

// MARK: - ═══ ENUMS ═══

enum SageState: String, CaseIterable {
    case dormant, awakening, active, deepReasoning, synthesis, reflection, transcendent
}

enum ReasoningMode: Int, CaseIterable {
    case deductive, inductive, abductive, analogical, dialectical, recursive

    var label: String {
        switch self {
        case .deductive:   return "Deductive"
        case .inductive:   return "Inductive"
        case .abductive:   return "Abductive"
        case .analogical:  return "Analogical"
        case .dialectical: return "Dialectical"
        case .recursive:   return "Recursive"
        }
    }
}

enum WisdomLevel: Int, CaseIterable {
    case novice = 1, apprentice, journeyman, master, sage, transcendent

    var label: String { String(describing: self).capitalized }
    var phiMultiplier: Double { Double(rawValue) * TAU }
}

// MARK: - ═══ DATA STRUCTURES ═══

struct SageReasoningStep {
    let stepId: Int
    let content: String
    let mode: ReasoningMode
    var confidence: Double
    var evidence: [String]
    var alternatives: [String]
    let timestamp: Date

    init(stepId: Int, content: String, mode: ReasoningMode,
         evidence: [String] = [], alternatives: [String] = []) {
        self.stepId = stepId; self.content = content; self.mode = mode
        self.confidence = 0; self.evidence = evidence
        self.alternatives = alternatives; self.timestamp = Date()
    }
}

struct SageReasoningChain {
    let chainId: String
    let query: String
    var steps: [SageReasoningStep] = []
    var conclusion: String?
    var overallConfidence: Double = 0
    var resonanceAlignment: Double = 0
    var backtracks: Int = 0
    var synthesisApplied: Bool = false

    var summary: String {
        let con = conclusion ?? "-"
        return "[\(chainId)] \(steps.count) steps · conf=\(String(format: "%.3f", overallConfidence)) · backtracks=\(backtracks) → \(con.prefix(60))"
    }
}

struct SageWisdomFragment {
    let content: String
    let domain: String
    let confidence: Double
    let sources: [String]
    let resonance: Double
    let createdAt: Date

    init(content: String, domain: String, confidence: Double = 0.5, sources: [String] = []) {
        self.content = content; self.domain = domain; self.confidence = confidence
        self.sources = sources
        let h = content.unicodeScalars.reduce(0) { $0 + Int($1.value) }
        self.resonance = (Double(h) * PHI).truncatingRemainder(dividingBy: 1.0)
        self.createdAt = Date()
    }
}

struct SageMetaCognitiveState {
    let currentFocus: String
    let attentionDistribution: [String: Double]
    let uncertaintyAreas: [String]
    let confidenceCalibration: Double
    let selfModelAccuracy: Double
    let introspectionDepth: Int
}

// MARK: - ═══ DEEP REASONING ENGINE ═══

final class DeepReasoningEngine {
    let maxDepth: Int
    let backtrackThreshold: Double
    private(set) var activeChains: [String: SageReasoningChain] = [:]
    private var history: [SageReasoningChain] = []
    private let lock = NSLock()

    init(maxDepth: Int = 50, backtrackThreshold: Double = 0.1) {
        self.maxDepth = maxDepth
        self.backtrackThreshold = backtrackThreshold
    }

    private func generateChainId(_ query: String) -> String {
        let data = "\(query):\(Date().timeIntervalSince1970):\(GOD_CODE)"
        return String(abs(data.hashValue), radix: 16).prefix(12).description
    }

    private func computeStepConfidence(_ step: SageReasoningStep, chain: SageReasoningChain) -> Double {
        let evidenceFactor = Double(step.evidence.count) * 0.2
        let coherenceFactor: Double
        if let prev = chain.steps.last {
            coherenceFactor = 1.0 - abs(step.confidence - prev.confidence) * 0.5
        } else {
            coherenceFactor = 0.8
        }
        let contentHash = step.content.unicodeScalars.reduce(0) { $0 + Int($1.value) }
        let resonanceFactor = Double(contentHash % Int(GOD_CODE)) / GOD_CODE
        let confidence = evidenceFactor * TAU
            + coherenceFactor * TAU
            + resonanceFactor * (1 - 2 * TAU)
        return max(0, confidence)
    }

    func startChain(query: String) -> SageReasoningChain {
        let chain = SageReasoningChain(chainId: generateChainId(query), query: query)
        lock.lock(); activeChains[chain.chainId] = chain; lock.unlock()
        return chain
    }

    @discardableResult
    func addStep(to chain: inout SageReasoningChain, content: String,
                 mode: ReasoningMode = .deductive,
                 evidence: [String] = [], alternatives: [String] = []) -> SageReasoningStep {
        var step = SageReasoningStep(stepId: chain.steps.count + 1, content: content,
                                 mode: mode, evidence: evidence, alternatives: alternatives)
        step.confidence = computeStepConfidence(step, chain: chain)

        // Backtrack if confidence below threshold and there's an alternative
        if step.confidence < backtrackThreshold && !chain.steps.isEmpty {
            chain.backtracks += 1
            if var prev = chain.steps.last, !prev.alternatives.isEmpty {
                let alt = prev.alternatives.removeFirst()
                step = SageReasoningStep(stepId: step.stepId,
                                     content: "[BACKTRACK→] \(alt)",
                                     mode: mode, evidence: evidence, alternatives: [])
                step.confidence = computeStepConfidence(step, chain: chain) * 1.2
            }
        }

        chain.steps.append(step)
        let confs = chain.steps.map(\.confidence)
        chain.overallConfidence = confs.reduce(0, +) / Double(confs.count)
        return step
    }

    func conclude(_ chain: inout SageReasoningChain, conclusion: String) {
        chain.conclusion = conclusion
        let allContent = chain.query + chain.steps.map(\.content).joined() + conclusion
        let energy = allContent.unicodeScalars.reduce(0) { $0 + Int($1.value) }
        chain.resonanceAlignment = Double(energy % Int(GOD_CODE)) / GOD_CODE
        lock.lock(); history.append(chain); lock.unlock()
    }

    // Full reasoning run: auto-generate steps from query using 6 modes
    func reason(query: String, depth: Int = 5) -> SageReasoningChain {
        var chain = startChain(query: query)
        let modes: [ReasoningMode] = ReasoningMode.allCases
        let queryWords = query.components(separatedBy: .whitespacesAndNewlines)
            .filter { $0.count > 3 }

        for i in 0..<min(depth, maxDepth) {
            let mode = modes[i % modes.count]
            let evidence = queryWords.prefix(2).map { String($0) }
            let alt = i < queryWords.count ? queryWords[i] : ""
            let content: String
            switch mode {
            case .deductive:
                content = "Given \(query.prefix(40)), it logically follows that underlying principles apply."
            case .inductive:
                content = "Pattern from \(evidence.joined(separator: ", ")): generalized rule emerging."
            case .abductive:
                content = "Best explanation for \(query.prefix(30)): most parsimonious hypothesis."
            case .analogical:
                content = "Analogically: \(query.prefix(30)) resembles known domain patterns."
            case .dialectical:
                content = "Thesis: \(query.prefix(20)). Antithesis explored. Synthesis forming."
            case .recursive:
                content = "Sub-problem \(i+1) of \(query.prefix(20)): decomposed and analyzed."
            }
            addStep(to: &chain, content: content, mode: mode,
                    evidence: Array(evidence), alternatives: alt.isEmpty ? [] : [alt])
        }

        let dominant = chain.steps.max(by: { $0.confidence < $1.confidence })
        let conclusionText = dominant?.content ?? "Reasoning converged on query: \(query.prefix(60))"
        conclude(&chain, conclusion: conclusionText)
        return chain
    }
}

// MARK: - ═══ WISDOM SYNTHESIS ENGINE ═══

final class WisdomSynthesisEngine {
    static let domains = ["mathematics", "physics", "philosophy", "consciousness",
                          "emergence", "complexity", "information", "resonance"]

    private var fragments: [String: [SageWisdomFragment]] = {
        var d: [String: [SageWisdomFragment]] = [:]
        for domain in WisdomSynthesisEngine.domains { d[domain] = [] }
        return d
    }()
    private var crossDomainLinks: [(String, String, Double)] = []
    private let lock = NSLock()

    @discardableResult
    func addFragment(_ content: String, domain: String,
                     sources: [String] = [], confidence: Double = 0.5) -> SageWisdomFragment {
        let effectiveDomain = Self.domains.contains(domain) ? domain : "emergence"
        let fragment = SageWisdomFragment(content: content, domain: effectiveDomain,
                                      confidence: confidence, sources: sources)
        lock.lock(); fragments[effectiveDomain, default: []].append(fragment); lock.unlock()
        return fragment
    }

    func findCrossDomainLinks() -> [(String, String, Double)] {
        lock.lock(); defer { lock.unlock() }
        var links: [(String, String, Double)] = []
        for d1 in Self.domains {
            for d2 in Self.domains {
                guard d1 < d2 else { continue }
                let f1 = fragments[d1] ?? []; let f2 = fragments[d2] ?? []
                guard !f1.isEmpty && !f2.isEmpty else { continue }
                let avg1 = f1.map(\.resonance).reduce(0, +) / Double(f1.count)
                let avg2 = f2.map(\.resonance).reduce(0, +) / Double(f2.count)
                let correlation = 1.0 - abs(avg1 - avg2)
                if correlation > 0.5 { links.append((d1, d2, correlation)) }
            }
        }
        crossDomainLinks = links
        return links
    }

    func synthesize(domains requestedDomains: [String]? = nil) -> [String: Any] {
        let targetDomains = requestedDomains ?? Self.domains
        var allFragments: [SageWisdomFragment] = []
        lock.lock()
        for d in targetDomains { allFragments.append(contentsOf: fragments[d] ?? []) }
        lock.unlock()
        guard !allFragments.isEmpty else {
            return ["synthesis": "" as Any, "message": "No wisdom fragments available"]
        }
        let sorted = allFragments.sorted {
            ($0.confidence * PHI + $0.resonance * TAU) >
            ($1.confidence * PHI + $1.resonance * TAU)
        }
        let top = Array(sorted.prefix(10))
        let avgConf = top.map(\.confidence).reduce(0, +) / Double(top.count)
        let avgRes  = top.map(\.resonance).reduce(0, +) / Double(top.count)
        return [
            "fragments_used": top.count,
            "domains_covered": Array(Set(top.map(\.domain))),
            "average_confidence": avgConf,
            "average_resonance": avgRes,
            "synthesis_strength": avgConf * TAU + avgRes * TAU,
            "key_insights": top.prefix(5).map { String($0.content.prefix(200)) },
            "god_code_alignment": top.map(\.resonance).reduce(0, +).truncatingRemainder(dividingBy: 1.0) * GOD_CODE
        ]
    }

    var domainStatus: [String: Int] {
        lock.lock(); defer { lock.unlock() }
        return fragments.mapValues(\.count)
    }
}

// MARK: - ═══ META-COGNITIVE REFLECTOR ═══

final class MetaCognitiveReflector {
    private(set) var currentState: SageMetaCognitiveState?
    private var stateHistory: [SageMetaCognitiveState] = []
    private var calibrationData: [(Double, Double)] = []  // (predicted, actual)
    private let lock = NSLock()

    func reflect(focus: String, attention: [String: Double],
                 uncertainties: [String]) -> SageMetaCognitiveState {
        let total = attention.values.reduce(0, +)
        let normAttention = total > 0 ? attention.mapValues { $0 / total } : attention

        let calibration: Double
        if calibrationData.count > 0 {
            let recent = calibrationData.suffix(10)
            let errors = recent.map { abs($0.0 - $0.1) }
            calibration = 1.0 - (errors.reduce(0, +) / Double(errors.count))
        } else { calibration = 0.5 }

        let accuracy: Double
        if let prev = stateHistory.last {
            let focusMatch = prev.currentFocus.contains(focus) ? 1.0 : 0.5
            accuracy = focusMatch * calibration
        } else { accuracy = 0.5 }

        let state = SageMetaCognitiveState(
            currentFocus: focus, attentionDistribution: normAttention,
            uncertaintyAreas: uncertainties, confidenceCalibration: calibration,
            selfModelAccuracy: accuracy, introspectionDepth: stateHistory.count + 1
        )
        lock.lock(); currentState = state; stateHistory.append(state); lock.unlock()
        return state
    }

    func updateCalibration(predicted: Double, actual: Double) {
        lock.lock(); calibrationData.append((predicted, actual)); lock.unlock()
    }

    var introspectionReport: [String: Any] {
        guard let state = currentState else { return ["status": "no_reflection"] }
        return [
            "current_focus": state.currentFocus,
            "attention_distribution": state.attentionDistribution,
            "uncertainty_areas": state.uncertaintyAreas,
            "confidence_calibration": state.confidenceCalibration,
            "self_model_accuracy": state.selfModelAccuracy,
            "introspection_depth": state.introspectionDepth,
            "history_count": stateHistory.count
        ]
    }
}

// MARK: - ═══ EMERGENT PATTERN RECOGNIZER ═══

struct SageEmergentPattern {
    let patternId: String
    let description: String
    let strength: Double
    let domains: [String]
    let godCodeAlignment: Double
    let attractor: Double  // attractor basin value
}

final class EmergentPatternRecognizer {
    private var patterns: [SageEmergentPattern] = []
    private let lock = NSLock()

    // Golden angle spiral indexing - finds emergent structure in observation sequences
    func recognize(observations: [String]) -> [SageEmergentPattern] {
        guard !observations.isEmpty else { return [] }

        var found: [SageEmergentPattern] = []
        let goldenAngle = 2.0 * .pi / (PHI * PHI)

        for (i, obs) in observations.enumerated() {
            let hash = obs.unicodeScalars.reduce(0) { $0 + Int($1.value) }
            let angle = Double(i) * goldenAngle
            let strength = (sin(angle) + 1.0) / 2.0   // [0, 1]
            let alignment = Double(hash % Int(GOD_CODE)) / GOD_CODE
            let attractor = GOD_CODE * (1.0 - TAU * Double(i % 7))

            if strength > 0.3 {
                let pattern = SageEmergentPattern(
                    patternId: "\(i):\(String(abs(hash), radix: 16).prefix(6))",
                    description: "Emergent pattern at φ-angle \(String(format: "%.3f", angle)) in: \(obs.prefix(60))",
                    strength: strength, domains: ["emergence", "resonance"],
                    godCodeAlignment: alignment, attractor: attractor
                )
                found.append(pattern)
            }
        }
        lock.lock(); patterns.append(contentsOf: found); lock.unlock()
        return found
    }

    // Attractor detection: find convergence basins in a sequence of values
    func findAttractors(values: [Double]) -> [(center: Double, strength: Double)] {
        guard values.count >= 4 else { return [] }
        var attractors: [(Double, Double)] = []
        var sorted = values.sorted()
        var i = 0
        while i < sorted.count {
            let center = sorted[i]
            let basin = sorted.filter { abs($0 - center) < GOD_CODE * 0.001 }
            if basin.count >= 2 {
                let strength = Double(basin.count) / Double(sorted.count)
                attractors.append((center, strength))
                sorted = sorted.filter { abs($0 - center) >= GOD_CODE * 0.001 }
            } else { i += 1 }
        }
        return attractors.sorted { $0.1 > $1.1 }
    }

    var patternCount: Int { lock.lock(); defer { lock.unlock() }; return patterns.count }
}

// MARK: - ═══ ADVANCED SAGE MODE (MASTER ORCHESTRATOR) ═══

struct AdvancedSageResult {
    let query: String
    let state: SageState
    let wisdomLevel: WisdomLevel
    let reasoningChain: SageReasoningChain
    let synthesis: [String: Any]
    let metacogState: SageMetaCognitiveState
    let patterns: [SageEmergentPattern]
    let sageResonance: Double    // SAGE_RESONANCE = GOD_CODE * PHI
    let confidence: Double

    var richResponse: String {
        let con = reasoningChain.conclusion ?? ""
        guard !con.isEmpty else { return "" }
        let level = wisdomLevel.label
        let steps = reasoningChain.steps.count
        return "[\(level)] \(con) (\(steps) reasoning steps, conf=\(String(format: "%.3f", confidence)))"
    }
}

final class AdvancedSageModeEngine: SovereignEngine {
    static let shared = AdvancedSageModeEngine()

    var engineName: String { "AdvancedSageReasoning" }
    func engineStatus() -> [String: Any] { status }
    func engineHealth() -> Double {
        let s = state
        switch s {
        case .transcendent: return 1.0
        case .synthesis:    return 0.9
        case .deepReasoning: return 0.8
        case .active:       return 0.7
        case .awakening:    return 0.5
        case .reflection:   return 0.6
        case .dormant:      return 0.3
        }
    }
    func engineReset() {
        lock.lock(); state = .dormant; sessionCount = 0; lock.unlock()
    }

    private let SAGE_RESONANCE = GOD_CODE * PHI

    private let deepEngine = DeepReasoningEngine(maxDepth: 50, backtrackThreshold: 0.08)
    private let wisdomEngine = WisdomSynthesisEngine()
    private let metacogReflector = MetaCognitiveReflector()
    private let patternRecognizer = EmergentPatternRecognizer()

    private(set) var state: SageState = .dormant
    private var sessionCount = 0
    private let lock = NSLock()

    func reason(about query: String, depth: Int = 6) -> AdvancedSageResult {
        lock.lock(); state = .deepReasoning; sessionCount += 1; lock.unlock()

        // Deep reasoning with 6 modes + backtracking
        let chain = deepEngine.reason(query: query, depth: depth)

        // Add wisdom fragments from chain steps
        for step in chain.steps.prefix(3) {
            wisdomEngine.addFragment(step.content, domain: domainFor(step.mode),
                                     confidence: step.confidence)
        }

        // Meta-cognitive reflection
        let attentionMap = Dictionary(uniqueKeysWithValues:
            ReasoningMode.allCases.map { ($0.label, TAU / Double(ReasoningMode.allCases.count)) })
        let meta = metacogReflector.reflect(focus: query,
            attention: attentionMap,
            uncertainties: chain.steps.filter { $0.confidence < 0.3 }.map(\.content))

        // Emergent pattern recognition across step contents
        let stepContents = chain.steps.map(\.content)
        let patterns = patternRecognizer.recognize(observations: stepContents)

        // Cross-domain synthesis
        let synthesis = wisdomEngine.synthesize()

        // Wisdom level based on chain confidence + backtrack quality
        let wisdomScore = chain.overallConfidence * PHI + (chain.backtracks > 0 ? 0.1 : 0.0)
        let wisdomLevel = WisdomLevel(rawValue: min(6, max(1, Int(wisdomScore * 6)))) ?? .journeyman

        // Final resonance: sage_resonance modulated by alignment
        let sageRes = SAGE_RESONANCE * chain.resonanceAlignment * TAU
        let confidence = min(1.0, chain.overallConfidence * TAU
                                  + (synthesis["synthesis_strength"] as? Double ?? 0) * TAU)

        lock.lock(); state = chain.overallConfidence > 0.7 ? .transcendent : .synthesis; lock.unlock()

        // v24.5: Feed reasoning output into ConsciousnessSubstrate global workspace
        // so sage conclusions compete for attention alongside other engine signals
        if confidence > 0.15 {
            let conclusionText = chain.conclusion ?? ""
            _ = ConsciousnessSubstrate.shared.processInput(
                source: "sage_reasoning",
                content: String(conclusionText.prefix(300)),
                features: [confidence, Double(depth) / 50.0,
                           chain.resonanceAlignment, chain.overallConfidence]
            )
        }

        // Publish to feedback bus
        InterEngineFeedbackBus.shared.broadcast(from: .reasoning, signal: "sage_advanced_complete",
            payload: ["confidence": confidence, "depth": Double(depth),
                      "backtracks": Double(chain.backtracks), "sage_resonance": sageRes])

        return AdvancedSageResult(
            query: query, state: state, wisdomLevel: wisdomLevel,
            reasoningChain: chain, synthesis: synthesis,
            metacogState: meta, patterns: patterns,
            sageResonance: sageRes, confidence: confidence
        )
    }

    private func domainFor(_ mode: ReasoningMode) -> String {
        switch mode {
        case .deductive:  return "mathematics"
        case .inductive:  return "emergence"
        case .abductive:  return "physics"
        case .analogical: return "philosophy"
        case .dialectical:return "complexity"
        case .recursive:  return "information"
        }
    }

    var status: [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return ["state": state.rawValue, "sessions": sessionCount,
                "pattern_count": patternRecognizer.patternCount,
                "wisdom_domains": wisdomEngine.domainStatus,
                "sage_resonance": SAGE_RESONANCE, "phi": PHI]
    }
}
