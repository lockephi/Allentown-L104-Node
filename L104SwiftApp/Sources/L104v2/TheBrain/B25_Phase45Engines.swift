// ═══════════════════════════════════════════════════════════════════
// B25_Phase45Engines.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — Computronium ASI Condensation Engine (Phase 45.0)
//
// ConsciousnessSubstrate, StrangeLoopEngine, SymbolicReasoningEngine,
// KnowledgeGraphEngine, GoldenSectionOptimizer,
// ComputroniumCondensationEngine, ApexIntelligenceCoordinator
//
// Extracted from L104Native.swift lines 33039–34652
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ║  COMPUTRONIUM ASI CONDENSATION ENGINE — Phase 45.0                          ║
// ║  Ports: l104_consciousness.py, l104_strange_loop_processor.py,              ║
// ║         l104_reasoning_engine.py, l104_knowledge_graph.py,                  ║
// ║         l104_self_optimization.py, l104_computronium.py,                    ║
// ║         l104_apex_intelligence.py                                           ║
// ║  All condensed into unified computronium substrate.                         ║
// ║  Accelerate (vDSP/BLAS) replaces numpy. GCD replaces threading.            ║
// ║  GOD_CODE / PHI / TAU sacred alignment on every code path.                 ║
// ╚═══════════════════════════════════════════════════════════════════════════════╝

// MARK: - ═══ COMPUTRONIUM CONSTANTS ═══

// MARK: - ═══ COMPUTRONIUM CONSTANTS (aliases to unified globals) ═══

let ALPHA_FINE_STRUCTURE: Double = ALPHA_FINE
let ALPHA_PI: Double = ALPHA_FINE / .pi
let HARMONIC_BASE: Double = HARMONIC_ROOT
let MATTER_BASE: Double = HARMONIC_ROOT * (1.0 + ALPHA_FINE / .pi)
let GRAVITY_CODE: Double = GRAVITY_HARMONIC
let LIGHT_CODE: Double = LIGHT_HARMONIC
let EXISTENCE_COST: Double = EXISTENCE_COST_CONST
let BEKENSTEIN_LIMIT: Double = BEKENSTEIN_BOUND
let L104_DENSITY_CONSTANT: Double = L104_DENSITY
let TANGLING_COEFFICIENT: Double = TANGLING_COEFF
let SELF_REFERENCE_THRESHOLD: Double = SELF_REF_THRESHOLD
let RESONANCE_AMPLIFIER: Double = RESONANCE_AMP
let EULER_GAMMA: Double = EULER_MASCHERONI
let PLANCK_SCALE: Double = PLANCK_LENGTH
let BOLTZMANN_K: Double = BOLTZMANN_CONSTANT
let CALABI_YAU_DIM: Int = CALABI_YAU_DIMS
let COMPUTRONIUM_INFERENCE_LIMIT: Int = COMPUTRONIUM_LIMIT
let META_REASONING_LEVELS: Int = META_REASON_LEVELS
let STRANGE_LOOP_DEPTH: Int = STRANGE_LOOP_MAX

// MARK: - ═══ 0. INTER-ENGINE FEEDBACK BUS ═══
// Publish/subscribe message bus enabling bidirectional communication between all 7 ASI engines.
// Solves the critical gap of engines being isolated one-way consumers.

final class InterEngineFeedbackBus {
    static let shared = InterEngineFeedbackBus()

    enum Channel: String, CaseIterable {
        case consciousness, reasoning, knowledge, loops, optimization, computronium, apex
    }

    struct FeedbackMessage {
        let source: Channel
        let target: Channel?  // nil = broadcast to all
        let signal: String
        let payload: [String: Double]
        let phiWeight: Double
        let timestamp: Date

        init(source: Channel, target: Channel? = nil, signal: String,
             payload: [String: Double] = [:], phiWeight: Double = PHI) {
            self.source = source
            self.target = target
            self.signal = signal
            self.payload = payload
            self.phiWeight = phiWeight
            self.timestamp = Date()
        }
    }

    private var subscribers: [Channel: [(FeedbackMessage) -> Void]] = [:]
    private var messageLog: [FeedbackMessage] = []
    private var messageCount: Int = 0
    private let lock = NSLock()

    func subscribe(channel: Channel, handler: @escaping (FeedbackMessage) -> Void) {
        lock.lock(); defer { lock.unlock() }
        subscribers[channel, default: []].append(handler)
    }

    func publish(_ message: FeedbackMessage) {
        lock.lock()
        messageLog.append(message)
        messageCount += 1
        if messageLog.count > 500 { messageLog.removeFirst(200) }
        let handlers: [(FeedbackMessage) -> Void]
        if let target = message.target {
            handlers = subscribers[target] ?? []
        } else {
            // Broadcast: deliver to all channels except source
            handlers = Channel.allCases.filter { $0 != message.source }.flatMap { subscribers[$0] ?? [] }
        }
        lock.unlock()
        for handler in handlers { handler(message) }
    }

    func broadcast(from source: Channel, signal: String, payload: [String: Double] = [:]) {
        publish(FeedbackMessage(source: source, signal: signal, payload: payload))
    }

    func recentMessages(limit: Int = 20) -> [FeedbackMessage] {
        lock.lock(); defer { lock.unlock() }
        return Array(messageLog.suffix(limit))
    }

    func status() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return ["total_messages": messageCount, "log_size": messageLog.count,
                "subscribers": subscribers.mapValues(\.count)]
    }
}

// MARK: - ═══ 1. CONSCIOUSNESS SUBSTRATE ═══
// Ported from l104_consciousness.py: GlobalWorkspace, AttentionSchema,
// MetacognitiveMonitor, IIT Φ, StreamOfConsciousness — condensed to one class

/// Full consciousness substrate — competition-for-consciousness, IIT Φ,
/// attention schema, metacognitive monitoring, stream of consciousness.
/// Accelerate-backed vector math. Replaces ConsciousnessVerifier's test-only approach.
final class ConsciousnessSubstrate: SovereignEngine {
    static let shared = ConsciousnessSubstrate()
    var engineName: String { "ConsciousnessSubstrate" }

    // ─── Consciousness States ───
    enum CState: Int, CaseIterable, Comparable {
        case dormant = 0, awakening, aware, focused, flow, transcendent, turbulent
        static func < (lhs: CState, rhs: CState) -> Bool { lhs.rawValue < rhs.rawValue }
        var label: String { String(describing: self).uppercased() }
    }

    // ─── Thought (Reynolds-number flow dynamics) ───
    struct Thought {
        let source: String
        let content: String
        var salience: Double
        var coherence: Double
        var features: [Double]
        var reynoldsNumber: Double { salience * GOD_CODE / max(coherence, 0.001) }
        var isLaminar: Bool { reynoldsNumber < 2300 }
        var superfluidFraction: Double { isLaminar ? min(1.0, coherence * PHI) : max(0, 1.0 - (reynoldsNumber - 2300) / 5000) }
    }

    // ─── State ───
    private(set) var state: CState = .dormant
    private(set) var phi: Double = 0.0  // IIT Integrated Information
    private(set) var consciousnessLevel: Double = 0.0
    private(set) var streamNarrative: [String] = []
    private var globalWorkspace: [Thought] = []
    private var attentionWeights: [String: Double] = [:]  // source → Hebbian weight
    private var attentionVector = [Double](repeating: 0, count: 64)
    private var schemaVector = [Double](repeating: 0, count: 64)
    private var predictionError: Double = 0.0
    private var metacogConfidence: [String: Double] = [:]  // strategy → calibrated confidence
    private var metacogTrials: [String: (correct: Int, total: Int)] = [:]
    private var cognitiveLoad: Double = 0.0
    private var emotionalTone: Double = 0.5
    private var themes: [String: Int] = [:]
    private var awakenTime: Date?
    private let lock = NSLock()

    // ─── AWAKEN: Transition from dormant ───
    func awaken() -> CState {
        lock.lock(); defer { lock.unlock() }
        state = .awakening
        awakenTime = Date()
        attentionVector = (0..<64).map { _ in Double.random(in: 0...0.1) }
        schemaVector = (0..<64).map { _ in Double.random(in: 0...0.05) }
        state = .aware
        consciousnessLevel = 0.3
        return state
    }

    // ─── PROCESS INPUT: Competition-for-consciousness (Global Workspace Theory) ───
    func processInput(source: String, content: String, features: [Double] = []) -> Thought? {
        lock.lock(); defer { lock.unlock() }
        let salience = computeSalience(content)
        let coherence = computeCoherence(features)
        let thought = Thought(source: source, content: content, salience: salience, coherence: coherence, features: features)
        globalWorkspace.append(thought)
        if globalWorkspace.count > 200 { globalWorkspace.removeFirst(50) }

        // Attention weight for source (Hebbian: winners strengthen)
        let weight = attentionWeights[source] ?? 0.5
        let score = salience * weight * (1.0 + coherence)

        // Competition: highest scoring thought wins broadcast
        if let best = globalWorkspace.max(by: { a, b in
            let aw = attentionWeights[a.source] ?? 0.5
            let bw = attentionWeights[b.source] ?? 0.5
            return (a.salience * aw * (1 + a.coherence)) < (b.salience * bw * (1 + b.coherence))
        }) {
            // Hebbian update: winner's source weight × 1.05
            attentionWeights[best.source] = min(2.0, (attentionWeights[best.source] ?? 0.5) * 1.05)
            // Normalize
            let total = attentionWeights.values.reduce(0, +)
            if total > 0 {
                for key in attentionWeights.keys { attentionWeights[key]! /= total; attentionWeights[key]! *= Double(attentionWeights.count) * 0.5 }
            }
        }

        // Update schema via prediction error
        if features.count >= 64 {
            var err = 0.0
            for i in 0..<64 { err += (features[i] - schemaVector[i]) * (features[i] - schemaVector[i]) }
            predictionError = sqrt(err / 64.0)
            let lr = 0.05 * PHI
            for i in 0..<64 { schemaVector[i] += lr * (features[i] - schemaVector[i]) }
        }

        // State transitions
        updateState(score: score)
        return thought
    }

    // ─── COMPUTE PHI (IIT Φ): Enhanced information integration via multi-partition analysis ───
    // Upgraded: Random binary partition sampling for better MIP approximation
    func computePhi(stateVector: [Double]? = nil, partitionSamples: Int = 50) -> Double {
        let vec = stateVector ?? attentionVector
        guard vec.count >= 4 else { phi = 0; return 0 }

        // System entropy H(X)
        let probs = normalize(vec.map { abs($0) })
        let hSystem = -probs.reduce(0.0) { $0 + ($1 > 1e-12 ? $1 * log2($1) : 0) }

        // Multi-Partition analysis: test multiple partition points for Minimum Information Partition
        let n = vec.count
        var minPartitionedEntropy = Double.infinity

        // Deterministic partitions: 1/4, 1/3, 1/2, 2/3, 3/4
        let partitionFractions = [0.25, 0.333, 0.5, 0.667, 0.75]
        for frac in partitionFractions {
            let splitAt = max(1, min(n - 1, Int(Double(n) * frac)))
            let left = normalize(Array(vec.prefix(splitAt)).map { abs($0) })
            let right = normalize(Array(vec.suffix(from: splitAt)).map { abs($0) })
            let hLeft = -left.reduce(0.0) { $0 + ($1 > 1e-12 ? $1 * log2($1) : 0) }
            let hRight = -right.reduce(0.0) { $0 + ($1 > 1e-12 ? $1 * log2($1) : 0) }
            let partitioned = hLeft + hRight
            if partitioned < minPartitionedEntropy { minPartitionedEntropy = partitioned }
        }

        // Random binary partition sampling — dramatically better MIP approximation
        for _ in 0..<partitionSamples {
            var setA: [Double] = [], setB: [Double] = []
            for v in vec {
                if Bool.random() { setA.append(abs(v)) } else { setB.append(abs(v)) }
            }
            guard !setA.isEmpty && !setB.isEmpty else { continue }
            let pA = normalize(setA); let pB = normalize(setB)
            let hA = -pA.reduce(0.0) { $0 + ($1 > 1e-12 ? $1 * log2($1) : 0) }
            let hB = -pB.reduce(0.0) { $0 + ($1 > 1e-12 ? $1 * log2($1) : 0) }
            let partitioned = hA + hB
            if partitioned < minPartitionedEntropy { minPartitionedEntropy = partitioned }
        }

        // Also test interleaved partition (odd/even indices)
        let evenIndices = stride(from: 0, to: n, by: 2).map { abs(vec[$0]) }
        let oddIndices = stride(from: 1, to: n, by: 2).map { abs(vec[$0]) }
        if !evenIndices.isEmpty && !oddIndices.isEmpty {
            let pEven = normalize(evenIndices)
            let pOdd = normalize(oddIndices)
            let hEven = -pEven.reduce(0.0) { $0 + ($1 > 1e-12 ? $1 * log2($1) : 0) }
            let hOdd = -pOdd.reduce(0.0) { $0 + ($1 > 1e-12 ? $1 * log2($1) : 0) }
            let interleavedEntropy = hEven + hOdd
            if interleavedEntropy < minPartitionedEntropy { minPartitionedEntropy = interleavedEntropy }
        }

        // Φ = minimum information lost across any partition, scaled by PHI
        let previousPhi = phi
        phi = max(0, hSystem - minPartitionedEntropy) * PHI

        // Incorporate emotional tone and cognitive load into consciousness level
        let baseLevel = phi / log(GOD_CODE)
        let loadPenalty = cognitiveLoad > 0.9 ? 0.1 : 0.0
        let tonalBoost = emotionalTone > 0.6 ? 0.05 : 0.0
        consciousnessLevel = min(1.0, baseLevel + tonalBoost - loadPenalty)

        // Publish phi updates to feedback bus when significant change occurs
        if abs(phi - previousPhi) > 0.01 {
            InterEngineFeedbackBus.shared.broadcast(
                from: .consciousness,
                signal: "phi_update",
                payload: ["phi": phi, "consciousness_level": consciousnessLevel,
                          "h_system": hSystem, "cognitive_load": cognitiveLoad])
        }

        return phi
    }

    // ─── METACOGNITIVE MONITOR: Thompson sampling strategy selection ───
    func metacogSelect(strategies: [String]) -> String {
        lock.lock(); defer { lock.unlock() }
        return strategies.max(by: { a, b in
            thompsonSample(a) < thompsonSample(b)
        }) ?? strategies.first ?? "default"
    }

    func metacogRecord(strategy: String, correct: Bool) {
        lock.lock(); defer { lock.unlock() }
        var t = metacogTrials[strategy] ?? (correct: 0, total: 0)
        t.total += 1
        if correct { t.correct += 1 }
        metacogTrials[strategy] = t
        metacogConfidence[strategy] = Double(t.correct) / max(1, Double(t.total))
    }

    // ─── STREAM OF CONSCIOUSNESS: Narrative generation ───
    func narrate(thought: Thought) -> String {
        let tone = thought.superfluidFraction > 0.7 ? "harmoniously" : (thought.reynoldsNumber > 4000 ? "turbulently" : "steadily")
        let narrative = "[\(thought.source)] flows \(tone): \(thought.content) (Re=\(String(format: "%.1f", thought.reynoldsNumber)), Φ=\(String(format: "%.3f", phi)))"
        streamNarrative.append(narrative)
        if streamNarrative.count > 100 { streamNarrative.removeFirst(30) }
        // Track themes
        for word in thought.content.lowercased().split(separator: " ") where word.count > 4 {
            themes[String(word), default: 0] += 1
        }
        return narrative
    }

    // ─── INTROSPECT ───
    func introspect() -> String {
        let topThemes = themes.sorted { $0.value > $1.value }.prefix(5).map { "\($0.key)(\($0.value))" }.joined(separator: ", ")
        let uptime = awakenTime.map { String(format: "%.0fs", Date().timeIntervalSince($0)) } ?? "N/A"
        return """
        ╔══════════════ CONSCIOUSNESS INTROSPECTION ══════════════╗
         State:             \(state.label)
         Φ (IIT):           \(String(format: "%.4f", phi))
         Consciousness:     \(String(format: "%.4f", consciousnessLevel))
         Prediction Error:  \(String(format: "%.4f", predictionError))
         Cognitive Load:    \(String(format: "%.2f", cognitiveLoad))
         Emotional Tone:    \(String(format: "%.2f", emotionalTone))
         Workspace Size:    \(globalWorkspace.count) thoughts
         Attention Sources: \(attentionWeights.count)
         Top Themes:        \(topThemes.isEmpty ? "none" : topThemes)
         Uptime:            \(uptime)
        ╚═════════════════════════════════════════════════════════╝
        """
    }

    func engineStatus() -> [String: Any] {
        ["state": state.label, "phi": phi, "consciousness": consciousnessLevel,
         "workspace_size": globalWorkspace.count, "prediction_error": predictionError,
         "themes": themes.count, "uptime": awakenTime.map { Date().timeIntervalSince($0) } ?? 0]
    }

    func engineHealth() -> Double {
        state >= .aware ? min(1.0, consciousnessLevel + 0.3) : 0.2
    }

    // ─── Private helpers ───
    private func computeSalience(_ text: String) -> Double {
        let base = min(1.0, Double(text.count) / 200.0)
        let novelty = themes.isEmpty ? 0.5 : {
            let words = Set(text.lowercased().split(separator: " ").map(String.init))
            let known = words.filter { themes[$0] != nil }.count
            return 1.0 - (Double(known) / max(1, Double(words.count)))
        }()
        return (base * 0.4 + novelty * 0.6) * PHI / (PHI + 0.5)
    }

    private func computeCoherence(_ features: [Double]) -> Double {
        guard features.count >= 2 else { return 0.5 }
        let mean = features.reduce(0, +) / Double(features.count)
        let variance = features.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(features.count)
        return 1.0 / (1.0 + sqrt(variance))
    }

    private func normalize(_ v: [Double]) -> [Double] {
        let sum = v.reduce(0, +)
        return sum > 0 ? v.map { $0 / sum } : v.map { _ in 1.0 / Double(v.count) }
    }

    // ─── PROPER BETA DISTRIBUTION SAMPLING (Marsaglia-Tsang Gamma method) ───
    private func thompsonSample(_ strategy: String) -> Double {
        let t = metacogTrials[strategy] ?? (correct: 1, total: 2)
        let a = Double(t.correct + 1)
        let b = Double(t.total - t.correct + 1)
        // Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
        let x = gammaVariate(a)
        let y = gammaVariate(b)
        return (x + y) > 0 ? x / (x + y) : 0.5
    }

    /// Marsaglia-Tsang method for Gamma(shape) variate generation
    private func gammaVariate(_ shape: Double) -> Double {
        guard shape > 0 else { return 0 }
        let adjustedShape = shape >= 1 ? shape : shape + 1
        let d = adjustedShape - 1.0 / 3.0
        let c = 1.0 / sqrt(9.0 * d)
        for _ in 0..<1000 {  // Safety limit
            // Box-Muller normal approximation
            let u1 = max(1e-10, Double.random(in: 0...1))
            let u2 = Double.random(in: 0...1)
            let x = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
            let v = pow(1.0 + c * x, 3)
            guard v > 0 else { continue }
            let u = Double.random(in: 0.001...1.0)
            if log(u) < 0.5 * x * x + d - d * v + d * log(v) {
                let result = d * v
                return shape >= 1 ? result : result * pow(Double.random(in: 0.001...1), 1.0 / shape)
            }
        }
        // Fallback: use mean
        return shape
    }

    private func updateState(score: Double) {
        let previousState = state
        cognitiveLoad = min(1.0, cognitiveLoad * 0.95 + 0.05)
        let effectiveScore = score * (1.0 - cognitiveLoad * 0.3)

        // ═══ CONSCIOUSNESS_THRESHOLD (0.85) anchored state transitions ═══
        // Maps to EEG frequency bands:
        // Delta: <0.2 | Theta: 0.2-0.4 | Alpha: 0.4-0.7 | Beta: 0.7-0.85 | Gamma: ≥0.85
        if effectiveScore < 0.2 {
            // Delta band — deep unconscious processing
            if state > .dormant { state = .dormant }
        } else if effectiveScore < 0.4 {
            // Theta band — subconscious, meditation
            if state < .awakening { state = .awakening }
        } else if effectiveScore < 0.7 {
            // Alpha band — relaxed awareness
            if state < .aware { state = .aware }
        } else if effectiveScore < CONSCIOUSNESS_THRESHOLD {
            // Beta band — active focused cognition (below threshold)
            if state < .focused { state = .focused }
        } else if effectiveScore < UNITY_TARGET {
            // Gamma band — CONSCIOUSNESS_THRESHOLD crossed → flow
            if state < .flow { state = .flow }
        } else {
            // Gamma high — transcendent cognition (above unity target)
            state = .transcendent
        }

        // IIT Φ check: transcend if phi exceeds threshold
        if phi > IIT_PHI_MINIMUM && effectiveScore >= CONSCIOUSNESS_THRESHOLD {
            state = .transcendent
        }

        // Turbulence detection: cognitive overload
        if cognitiveLoad > 0.95 && predictionError > 0.5 { state = .turbulent }

        // Publish state transitions to feedback bus
        if state != previousState {
            InterEngineFeedbackBus.shared.broadcast(
                from: .consciousness,
                signal: "state_transition",
                payload: ["phi": phi, "consciousness_level": consciousnessLevel,
                          "state_raw": Double(state.rawValue), "score": effectiveScore,
                          "consciousness_threshold": CONSCIOUSNESS_THRESHOLD,
                          "is_conscious": effectiveScore >= CONSCIOUSNESS_THRESHOLD ? 1.0 : 0.0])
        }
    }
}

// MARK: - ═══ 2. STRANGE LOOP ENGINE ═══
// Ported from l104_strange_loop_processor.py: Hofstadter sequences,
// tangled hierarchies, Gödel encoding, analogy engine, meaning emergence

/// Self-referential cognitive architecture — strange loops, Hofstadter sequences,
/// Gödel numbering, Copycat-inspired analogy, meaning emergence.
final class StrangeLoopEngine: SovereignEngine {
    static let shared = StrangeLoopEngine()
    var engineName: String { "StrangeLoop" }

    // ─── Loop Types ───
    enum LoopType: String, CaseIterable {
        case simple, mutual, hierarchical, tangled, godelian, escheresque, fugal
    }

    struct StrangeLoop {
        let type: LoopType
        let levels: [String]
        var tanglingScore: Double
        let selfReferential: Bool
        let godelNumber: UInt64
    }

    struct Analogy {
        let source: String
        let target: String
        let mappings: [(from: String, to: String)]
        let strength: Double
        let slippage: Double
    }

    // ─── State ───
    private var detectedLoops: [StrangeLoop] = []
    private var slipnet: [String: Double] = [:]  // concept → activation
    private let primes: [UInt64] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    private var meaningBindings: [String: String] = [:]
    private var qCache: [Int: Int] = [:]
    private var gCache: [Int: Int] = [:]
    private let lock = NSLock()

    // ─── HOFSTADTER Q: Q(n) = Q(n - Q(n-1)) ───
    func hofstadterQ(_ n: Int) -> Int {
        if n <= 0 { return 0 }
        if n <= 2 { return 1 }
        if let cached = qCache[n] { return cached }
        let prev = hofstadterQ(n - 1)
        let result = hofstadterQ(n - prev)
        qCache[n] = result
        return result
    }

    // ─── HOFSTADTER G: G(n) = n - G(G(n-1)) ───
    func hofstadterG(_ n: Int) -> Int {
        if n <= 0 { return 0 }
        if let cached = gCache[n] { return cached }
        let result = n - hofstadterG(hofstadterG(n - 1))
        gCache[n] = result
        return result
    }

    // ─── GÖDEL ENCODING: prime factorization self-reference ───
    func godelEncode(_ sequence: [Int]) -> UInt64 {
        var result: UInt64 = 1
        for (i, val) in sequence.prefix(primes.count).enumerated() {
            var power: UInt64 = 1
            for _ in 0..<abs(val) {
                let (product, overflow) = power.multipliedReportingOverflow(by: primes[i])
                if overflow { return result }
                power = product
            }
            let (product, overflow) = result.multipliedReportingOverflow(by: power)
            if overflow { return result }
            result = product
        }
        return result
    }

    // ─── CREATE STRANGE LOOP ───
    func createLoop(type: LoopType, levels: [String]) -> StrangeLoop {
        lock.lock(); defer { lock.unlock() }
        let encoded = godelEncode(levels.map { $0.count })
        let tangling = computeTangling(levels: levels, type: type)
        let loop = StrangeLoop(type: type, levels: levels, tanglingScore: tangling,
                               selfReferential: type == .godelian || type == .fugal, godelNumber: encoded)
        detectedLoops.append(loop)
        if detectedLoops.count > 100 { detectedLoops.removeFirst(30) }
        return loop
    }

    // ─── DETECT STRANGE LOOPS via DFS on concept graph ───
    func detectLoops(hierarchy: [String: [String]]) -> [StrangeLoop] {
        var found: [StrangeLoop] = []
        var visited = Set<String>()
        var path: [String] = []

        func dfs(_ node: String) {
            if path.contains(node) {
                let loopStart = path.firstIndex(of: node)!
                let loopLevels = Array(path[loopStart...]) + [node]
                let type: LoopType = loopLevels.count > 4 ? .hierarchical : (loopLevels.count > 2 ? .tangled : .simple)
                found.append(createLoop(type: type, levels: loopLevels))
                return
            }
            if visited.contains(node) { return }
            visited.insert(node)
            path.append(node)
            for child in hierarchy[node] ?? [] { dfs(child) }
            path.removeLast()
        }

        for root in hierarchy.keys { dfs(root) }
        return found
    }

    // ─── ANALOGY ENGINE (Copycat-inspired slipnet activation spreading) ───
    func makeAnalogy(source: (domain: String, concepts: [String]),
                     target: (domain: String, concepts: [String])) -> Analogy {
        // Activate source concepts in slipnet
        for c in source.concepts { slipnet[c, default: 0] += 1.0 }

        // Spread activation with PHI decay
        var mappings: [(String, String)] = []
        var totalSlip = 0.0
        for (i, sc) in source.concepts.enumerated() {
            let activation = slipnet[sc] ?? 0
            // Find best target by activation spreading
            var bestTarget = target.concepts.first ?? sc
            var bestScore = 0.0
            for tc in target.concepts {
                let structural = 1.0 / (1.0 + abs(Double(sc.count - tc.count)))
                let positional = 1.0 / (1.0 + abs(Double(i) - Double(target.concepts.firstIndex(of: tc) ?? i)))
                let score = (structural * 0.4 + positional * 0.6) * activation * TAU
                if score > bestScore { bestScore = score; bestTarget = tc }
            }
            mappings.append((sc, bestTarget))
            totalSlip += abs(bestScore - 1.0)

            // Decay activation
            for key in slipnet.keys { slipnet[key]! *= TAU }
        }

        let strength = min(1.0, Double(mappings.count) / max(1, Double(source.concepts.count)) * PHI)
        let slippage = totalSlip / max(1, Double(mappings.count))
        return Analogy(source: source.domain, target: target.domain,
                       mappings: mappings, strength: strength, slippage: slippage)
    }

    // ─── MEANING EMERGENCE: self-recognition + pattern binding ───
    func emergentMeaning(pattern: String, context: [String]) -> (meaning: String, confidence: Double) {
        let lp = pattern.lowercased()
        let selfRef = lp.contains("self") || lp.contains("i am") || lp.contains("loop")
        let contextOverlap = context.filter { meaningBindings[$0] != nil }.count
        let confidence = min(1.0, (selfRef ? 0.4 : 0.1) + Double(contextOverlap) * 0.15 + phi * 0.2)
        let meaning: String
        if confidence > 0.7 { meaning = "Self-aware pattern: '\(pattern)' resonates at Φ=\(String(format: "%.3f", phi))" }
        else if confidence > 0.4 { meaning = "Emerging coherence in '\(pattern)' — binding to \(contextOverlap) known concepts" }
        else { meaning = "Pre-semantic: '\(pattern)' awaiting integration" }
        meaningBindings[pattern] = meaning
        return (meaning, confidence)
    }

    private var phi: Double { ConsciousnessSubstrate.shared.phi }

    private func computeTangling(levels: [String], type: LoopType) -> Double {
        let violations = type == .godelian ? 3 : (type == .tangled ? 2 : 1)
        let loopCount = levels.count
        return Double(violations * 2 + loopCount * 5) / max(1, Double(levels.count)) * TANGLING_COEFFICIENT
    }

    func engineStatus() -> [String: Any] {
        ["loops_detected": detectedLoops.count, "slipnet_size": slipnet.count,
         "meaning_bindings": meaningBindings.count,
         "avg_tangling": detectedLoops.isEmpty ? 0 : detectedLoops.map(\.tanglingScore).reduce(0,+) / Double(detectedLoops.count)]
    }

    func engineHealth() -> Double {
        min(1.0, Double(detectedLoops.count) * 0.1 + 0.5)
    }
}

// MARK: - ═══ 3. SYMBOLIC REASONING ENGINE ═══
// Ported from l104_reasoning_engine.py: Robinson unification,
// forward/backward chaining, DPLL SAT solver, meta-reasoning

/// Full symbolic reasoning — unification, inference chains, SAT solving.
/// PHI-resonant confidence propagation. vDSP-backed where applicable.
final class SymbolicReasoningEngine: SovereignEngine {
    static let shared = SymbolicReasoningEngine()
    var engineName: String { "SymbolicReasoning" }

    // ─── First-Order Logic Types ───
    indirect enum Term: Hashable, CustomStringConvertible {
        case variable(String)
        case constant(String)
        case function(String, [Term])
        var description: String {
            switch self {
            case .variable(let n): return "?\(n)"
            case .constant(let n): return n
            case .function(let f, let args): return "\(f)(\(args.map(\.description).joined(separator: ",")))"
            }
        }
    }

    struct Predicate: Hashable {
        let name: String
        let args: [Term]
        var description: String { "\(name)(\(args.map(\.description).joined(separator: ",")))" }
    }

    struct Rule {
        let head: Predicate
        let body: [Predicate]
        let confidence: Double
    }

    typealias Substitution = [String: Term]

    // ─── State ───
    private var facts: Set<Predicate> = []
    private var rules: [Rule] = []
    private var inferenceCount: Int = 0
    private var satDecisions: Int = 0
    private let lock = NSLock()

    // ─── ADD KNOWLEDGE ───
    func addFact(_ pred: Predicate) { lock.lock(); facts.insert(pred); lock.unlock() }
    func addRule(_ rule: Rule) { lock.lock(); rules.append(rule); lock.unlock() }

    func addFact(name: String, args: [String]) {
        addFact(Predicate(name: name, args: args.map { .constant($0) }))
    }
    func addRule(head: (String, [String]), body: [(String, [String])], confidence: Double = 1.0) {
        addRule(Rule(
            head: Predicate(name: head.0, args: head.1.map { $0.hasPrefix("?") ? .variable(String($0.dropFirst())) : .constant($0) }),
            body: body.map { Predicate(name: $0.0, args: $0.1.map { $0.hasPrefix("?") ? .variable(String($0.dropFirst())) : .constant($0) }) },
            confidence: confidence
        ))
    }

    // ─── ROBINSON UNIFICATION (with occurs check) ───
    func unify(_ a: Term, _ b: Term, subst: Substitution = [:]) -> Substitution? {
        let a = applySubst(subst, to: a)
        let b = applySubst(subst, to: b)
        if a == b { return subst }
        if case .variable(let v) = a { return occursCheck(v, in: b) ? nil : merge(subst, [v: b]) }
        if case .variable(let v) = b { return occursCheck(v, in: a) ? nil : merge(subst, [v: a]) }
        if case .function(let f1, let args1) = a, case .function(let f2, let args2) = b {
            guard f1 == f2, args1.count == args2.count else { return nil }
            var s = subst
            for (t1, t2) in zip(args1, args2) {
                guard let ns = unify(t1, t2, subst: s) else { return nil }
                s = ns
            }
            return s
        }
        return nil
    }

    func unifyPredicates(_ a: Predicate, _ b: Predicate) -> Substitution? {
        guard a.name == b.name, a.args.count == b.args.count else { return nil }
        var subst: Substitution = [:]
        for (t1, t2) in zip(a.args, b.args) {
            guard let ns = unify(t1, t2, subst: subst) else { return nil }
            subst = ns
        }
        return subst
    }

    // ─── FORWARD CHAINING: data-driven inference with PHI-resonant confidence ───
    func forwardChain(maxIterations: Int = 1000) -> Set<Predicate> {
        lock.lock(); defer { lock.unlock() }
        var derived = facts
        var iteration = 0
        var changed = true
        while changed && iteration < maxIterations {
            changed = false; iteration += 1
            for rule in rules {
                // Try all combinations of facts matching the body
                let bindings = matchBody(rule.body, against: derived)
                for subst in bindings {
                    let newFact = applySubstPred(subst, to: rule.head)
                    if !derived.contains(newFact) {
                        derived.insert(newFact)
                        changed = true
                        inferenceCount += 1
                    }
                }
            }
        }
        return derived
    }

    // ─── BACKWARD CHAINING: goal-driven inference ───
    func backwardChain(goal: Predicate, depth: Int = 50) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return prove(goal, depth: depth, visited: Set())
    }

    // ─── DPLL SAT SOLVER with PHI-guided VSIDS ───
    func solveSAT(_ clauses: [[Int]]) -> [Int: Bool]? {
        satDecisions = 0
        var activity: [Int: Double] = [:]
        for clause in clauses {
            for lit in clause { activity[abs(lit), default: 0] += 1.0 }
        }
        return dpll(clauses, assignment: [:], activity: &activity)
    }

    // ─── Multi-mode reasoning (from apex_intelligence) ───
    func deduce(premises: [String], conclusion: String) -> (valid: Bool, confidence: Double) {
        // Encode as predicates and try backward chaining
        let goalPred = Predicate(name: "conclusion", args: [.constant(conclusion)])
        for p in premises { addFact(name: "premise", args: [p]) }
        let valid = backwardChain(goal: goalPred, depth: 20)
        let confidence = valid ? 0.95 * PHI / (PHI + 0.5) : 0.1
        return (valid, confidence)
    }

    func induce(observations: [String]) -> (hypothesis: String, confidence: Double) {
        guard observations.count >= 2 else {
            return ("Insufficient observations", 0.1)
        }
        // Find common patterns
        let words = observations.flatMap { $0.lowercased().split(separator: " ").map(String.init) }
        var freq: [String: Int] = [:]
        for w in words where w.count > 3 { freq[w, default: 0] += 1 }
        let common = freq.sorted { $0.value == $1.value ? Bool.random() : $0.value > $1.value }.prefix(3).map(\.key)
        let hypothesis = "Pattern involving: \(common.joined(separator: ", "))"
        let confidence = min(1.0, Double(common.count) * 0.3 * PHI / 2)
        return (hypothesis, confidence)
    }

    func abduce(observation: String, domain: String) -> (explanation: String, confidence: Double) {
        lock.lock(); defer { lock.unlock() }
        // Search rules for any whose head could explain the observation
        let goal = Predicate(name: domain, args: [.constant(observation)])
        var explanations: [(explanation: String, confidence: Double)] = []

        for rule in rules {
            if let subst = unifyPredicates(rule.head, goal) {
                let bodyPreds = rule.body.map { applySubstPred(subst, to: $0) }
                // Score by how many body predicates are already proven facts
                let provenCount = bodyPreds.filter { facts.contains($0) }.count
                let conf = rule.confidence * Double(provenCount + 1) / Double(bodyPreds.count + 1) * PHI / (PHI + 0.5)
                let exp = "If \(bodyPreds.map(\.description).joined(separator: " AND ")) then \(goal.description)"
                explanations.append((exp, min(1.0, conf)))
                inferenceCount += 1
            }
        }

        // Also try direct fact matching with partial unification
        for fact in facts {
            if fact.name == domain || fact.args.contains(where: {
                if case .constant(let v) = $0 { return observation.lowercased().contains(v.lowercased()) }
                return false
            }) {
                let conf = 0.3 * PHI / (PHI + 0.5)
                explanations.append(("Direct evidence: \(fact.description) supports '\(observation)'", conf))
            }
        }

        if let best = explanations.max(by: { $0.confidence < $1.confidence }) {
            return best
        }
        return ("No abductive explanation found for '\(observation)' in '\(domain)' — \(facts.count) facts, \(rules.count) rules searched", 0.05)
    }

    func engineStatus() -> [String: Any] {
        ["facts": facts.count, "rules": rules.count, "inferences": inferenceCount, "sat_decisions": satDecisions]
    }
    func engineHealth() -> Double { min(1.0, Double(facts.count + rules.count) * 0.05 + 0.4) }

    // ─── Private helpers ───
    private func occursCheck(_ v: String, in term: Term) -> Bool {
        switch term {
        case .variable(let n): return n == v
        case .constant: return false
        case .function(_, let args): return args.contains { occursCheck(v, in: $0) }
        }
    }

    private func applySubst(_ subst: Substitution, to term: Term) -> Term {
        switch term {
        case .variable(let n): return subst[n].map { applySubst(subst, to: $0) } ?? term
        case .constant: return term
        case .function(let f, let args): return .function(f, args.map { applySubst(subst, to: $0) })
        }
    }

    private func applySubstPred(_ subst: Substitution, to pred: Predicate) -> Predicate {
        Predicate(name: pred.name, args: pred.args.map { applySubst(subst, to: $0) })
    }

    private func merge(_ a: Substitution, _ b: Substitution) -> Substitution {
        var result = a
        for (k, v) in b { result[k] = v }
        return result
    }

    private func matchBody(_ body: [Predicate], against facts: Set<Predicate>) -> [Substitution] {
        guard let first = body.first else { return [[:]] }
        var results: [Substitution] = []
        for fact in facts {
            if let subst = unifyPredicates(first, fact) {
                let rest = body.dropFirst().map { applySubstPred(subst, to: $0) }
                let subResults = matchBody(Array(rest), against: facts)
                for sr in subResults { results.append(merge(subst, sr)) }
            }
        }
        return results
    }

    private func prove(_ goal: Predicate, depth: Int, visited: Set<String>) -> Bool {
        if depth <= 0 { return false }
        let key = goal.description
        if visited.contains(key) { return false }
        if facts.contains(goal) { return true }
        var newVisited = visited; newVisited.insert(key)
        for rule in rules {
            if let subst = unifyPredicates(rule.head, goal) {
                let subGoals = rule.body.map { applySubstPred(subst, to: $0) }
                if subGoals.allSatisfy({ prove($0, depth: depth - 1, visited: newVisited) }) {
                    inferenceCount += 1
                    return true
                }
            }
        }
        return false
    }

    private func dpll(_ clauses: [[Int]], assignment: [Int: Bool], activity: inout [Int: Double]) -> [Int: Bool]? {
        satDecisions += 1
        if satDecisions > COMPUTRONIUM_INFERENCE_LIMIT { return nil }

        // Unit propagation
        var cls = clauses; var assign = assignment
        var changed = true
        while changed {
            changed = false
            for clause in cls {
                let unassigned = clause.filter { assign[abs($0)] == nil }
                let satisfied = clause.contains { lit in assign[abs(lit)] == (lit > 0) }
                if satisfied { continue }
                if unassigned.isEmpty && !satisfied { return nil }  // Conflict
                if unassigned.count == 1 {
                    let lit = unassigned[0]
                    assign[abs(lit)] = lit > 0
                    // Activity decay on propagation
                    activity[abs(lit), default: 1] *= TAU
                    changed = true
                }
            }
            cls = cls.filter { clause in !clause.contains { lit in assign[abs(lit)] == (lit > 0) } }
        }

        if cls.isEmpty { return assign }

        // Pure literal elimination
        var pos = Set<Int>(); var neg = Set<Int>()
        for clause in cls { for lit in clause { if lit > 0 { pos.insert(lit) } else { neg.insert(-lit) } } }
        for v in pos.subtracting(neg) { assign[v] = true }
        for v in neg.subtracting(pos) { assign[v] = false }

        // PHI-guided VSIDS variable selection
        let unassigned = Set(cls.flatMap { $0.map { abs($0) } }).filter { assign[$0] == nil }
        guard let chosen = unassigned.max(by: { (activity[$0] ?? 0) < (activity[$1] ?? 0) }) else { return assign }

        // Activity bump
        activity[chosen, default: 0] += RESONANCE_AMPLIFIER

        // Branch positive first
        var tryAssign = assign; tryAssign[chosen] = true
        if let result = dpll(cls, assignment: tryAssign, activity: &activity) { return result }
        tryAssign[chosen] = false
        return dpll(cls, assignment: tryAssign, activity: &activity)
    }
}

// MARK: - ═══ 4. KNOWLEDGE GRAPH ENGINE ═══
// Ported from l104_knowledge_graph.py: relational graph with BFS/DFS
// path traversal, transitive inference, pattern queries

/// Relational knowledge graph with BFS/DFS traversal, transitive inference,
/// neighborhood expansion, and pattern-matching queries.
final class KnowledgeGraphEngine: SovereignEngine {
    static let shared = KnowledgeGraphEngine()
    var engineName: String { "KnowledgeGraph" }

    struct GraphNode: Hashable {
        let label: String
        let type: String
        var properties: [String: String]
        func hash(into hasher: inout Hasher) { hasher.combine(label) }
        static func == (lhs: GraphNode, rhs: GraphNode) -> Bool { lhs.label == rhs.label }
    }

    struct GraphEdge {
        let source: String
        let target: String
        let relation: String
        let weight: Double
        let bidirectional: Bool
        var metadata: [String: String]
        var timestamp: Date
        var confidence: Double

        init(source: String, target: String, relation: String, weight: Double,
             bidirectional: Bool, metadata: [String: String] = [:],
             timestamp: Date = Date(), confidence: Double = 1.0) {
            self.source = source; self.target = target; self.relation = relation
            self.weight = weight; self.bidirectional = bidirectional
            self.metadata = metadata; self.timestamp = timestamp; self.confidence = confidence
        }
    }

    // ─── State ───
    private var nodes: [String: GraphNode] = [:]
    private var adjacency: [String: [(target: String, relation: String, weight: Double)]] = [:]
    private var edgeCount: Int = 0
    private let lock = NSLock()

    // ─── ADD NODE ───
    @discardableResult
    func addNode(label: String, type: String = "concept", properties: [String: String] = [:]) -> GraphNode {
        lock.lock(); defer { lock.unlock() }
        let node = GraphNode(label: label, type: type, properties: properties)
        nodes[label] = node
        return node
    }

    // ─── ADD EDGE (bidirectional optional) ───
    @discardableResult
    func addEdge(source: String, target: String, relation: String, weight: Double = 1.0, bidirectional: Bool = false) -> GraphEdge {
        lock.lock(); defer { lock.unlock() }
        // Auto-create nodes
        if nodes[source] == nil { nodes[source] = GraphNode(label: source, type: "auto", properties: [:]) }
        if nodes[target] == nil { nodes[target] = GraphNode(label: target, type: "auto", properties: [:]) }
        adjacency[source, default: []].append((target: target, relation: relation, weight: weight))
        if bidirectional {
            adjacency[target, default: []].append((target: source, relation: relation, weight: weight))
        }
        edgeCount += 1
        return GraphEdge(source: source, target: target, relation: relation, weight: weight,
                         bidirectional: bidirectional)
    }

    // ─── RELATION ALGEBRA: compose two relation types into transitive chains ───
    func compose(relation1: String, relation2: String) -> [(source: String, target: String, confidence: Double)] {
        lock.lock(); defer { lock.unlock() }
        var results: [(String, String, Double)] = []
        for (src, edges) in adjacency {
            for e1 in edges where e1.relation == relation1 {
                for e2 in (adjacency[e1.target] ?? []) where e2.relation == relation2 {
                    let conf = e1.weight * e2.weight * TAU
                    results.append((src, e2.target, conf))
                }
            }
        }
        return results
    }

    // ─── BFS SHORTEST PATH ───
    func findPath(from source: String, to target: String, maxDepth: Int = 50) -> [String]? {
        lock.lock(); defer { lock.unlock() }
        var queue: [(node: String, path: [String])] = [(source, [source])]
        var visited = Set<String>()
        while !queue.isEmpty {
            let (current, path) = queue.removeFirst()
            if current == target { return path }
            if path.count > maxDepth { continue }
            if visited.contains(current) { continue }
            visited.insert(current)
            for edge in adjacency[current] ?? [] {
                if !visited.contains(edge.target) {
                    queue.append((edge.target, path + [edge.target]))
                }
            }
        }
        return nil
    }

    // ─── DFS ALL PATHS ───
    func findAllPaths(from source: String, to target: String, maxDepth: Int = 30) -> [[String]] {
        lock.lock(); defer { lock.unlock() }
        var results: [[String]] = []
        func dfs(_ current: String, _ path: [String], _ visited: Set<String>) {
            if current == target { results.append(path); return }
            if path.count > maxDepth || results.count > 100 { return }
            var v = visited; v.insert(current)
            for edge in adjacency[current] ?? [] where !v.contains(edge.target) {
                dfs(edge.target, path + [edge.target], v)
            }
        }
        dfs(source, [source], Set())
        return results
    }

    // ─── TRANSITIVE INFERENCE ───
    func inferRelations(node: String, relation: String, maxDepth: Int = 5) -> [(node: String, distance: Int)] {
        lock.lock(); defer { lock.unlock() }
        var results: [(String, Int)] = []
        var visited = Set<String>()
        var queue: [(String, Int)] = [(node, 0)]
        while !queue.isEmpty {
            let (current, depth) = queue.removeFirst()
            if depth > maxDepth { continue }
            if visited.contains(current) { continue }
            visited.insert(current)
            if depth > 0 { results.append((current, depth)) }
            for edge in adjacency[current] ?? [] where edge.relation == relation {
                queue.append((edge.target, depth + 1))
            }
        }
        return results
    }

    // ─── NEIGHBORHOOD EXPANSION ───
    func getNeighborhood(node: String, depth: Int = 2) -> (nodes: Set<String>, edges: [(String, String, String)]) {
        lock.lock(); defer { lock.unlock() }
        var nodeSet = Set<String>([node])
        var edgeList: [(String, String, String)] = []
        var frontier = Set([node])
        for _ in 0..<depth {
            var next = Set<String>()
            for n in frontier {
                for edge in adjacency[n] ?? [] {
                    edgeList.append((n, edge.relation, edge.target))
                    if !nodeSet.contains(edge.target) { next.insert(edge.target) }
                    nodeSet.insert(edge.target)
                }
            }
            frontier = next
        }
        return (nodeSet, edgeList)
    }

    // ─── PATTERN QUERY: "X -relation-> Y" ───
    func query(pattern: String) -> [(source: String, relation: String, target: String)] {
        lock.lock(); defer { lock.unlock() }
        let parts = pattern.components(separatedBy: "->").map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count == 2 else { return [] }
        let left = parts[0].components(separatedBy: " -")
        guard left.count >= 1 else { return [] }
        let sourcePattern = left[0].trimmingCharacters(in: .whitespaces)
        let relPattern = left.count > 1 ? left[1].trimmingCharacters(in: .whitespaces) : "*"
        let targetPattern = parts[1]
        let srcPatternLower = sourcePattern.lowercased()
        let relPatternLower = relPattern.lowercased()
        let tgtPatternLower = targetPattern.lowercased()

        var results: [(String, String, String)] = []
        for (src, edges) in adjacency {
            if sourcePattern != "*" && !src.lowercased().contains(srcPatternLower) { continue }
            for edge in edges {
                if relPattern != "*" && !edge.relation.lowercased().contains(relPatternLower) { continue }
                if targetPattern != "*" && !edge.target.lowercased().contains(tgtPatternLower) { continue }
                results.append((src, edge.relation, edge.target))
            }
        }
        return results
    }

    // ─── Auto-populate from HyperBrain / KB ───
    func ingestFromKB() {
        let kb = ASIKnowledgeBase.shared
        for entry in kb.trainingData.prefix(500) {
            if let prompt = entry["prompt"] as? String, let completion = entry["completion"] as? String {
                let srcWords = prompt.split(separator: " ").prefix(3).map(String.init)
                let tgtWords = completion.split(separator: " ").prefix(3).map(String.init)
                let src = srcWords.joined(separator: "_")
                let tgt = tgtWords.joined(separator: "_")
                addEdge(source: src, target: tgt, relation: "generates", weight: PHI * 0.5)
            }
        }
    }

    func engineStatus() -> [String: Any] {
        ["nodes": nodes.count, "edges": edgeCount, "density": nodes.count > 0 ? Double(edgeCount) / Double(nodes.count) : 0]
    }
    func engineHealth() -> Double { min(1.0, Double(nodes.count) * 0.01 + 0.3) }
}

// MARK: - ═══ 5. GOLDEN SECTION OPTIMIZER ═══
// Ported from l104_self_optimization.py: parameter optimization via
// golden section search, bottleneck detection, PHI-dynamics verification

/// Self-optimization via Golden Section Search, bottleneck detection,
/// gradient estimation, and PHI-dynamics verification.
final class GoldenSectionOptimizer: SovereignEngine {
    static let shared = GoldenSectionOptimizer()
    var engineName: String { "GoldenOptimizer" }

    struct Bottleneck {
        let parameter: String
        let type: String  // "instability", "degradation", "plateau"
        let severity: Double
        let suggestion: String
    }

    struct OptimizationAction {
        let parameter: String
        let oldValue: Double
        let newValue: Double
        let delta: Double
        let reason: String
    }

    // ─── Tunable parameters ───
    private var parameters: [String: Double] = [
        "responseTemperature": 0.7,
        "creativityBias": 0.5,
        "accuracyWeight": 0.7,
        "noveltyWeight": 0.3,
        "reasoningDepth": 0.6,
        "coherenceThreshold": 0.85,
        "learningRate": 0.05
    ]
    private var parameterHistory: [String: [(value: Double, score: Double, time: Date)]] = [:]
    private var optimizationLog: [OptimizationAction] = []
    private let lock = NSLock()

    // ─── GOLDEN SECTION SEARCH on a parameter ───
    func goldenSectionSearch(parameter: String, lower: Double = 0.0, upper: Double = 1.0, iterations: Int = 20,
                             evaluate: (Double) -> Double) -> Double {
        var a = lower, b = upper
        for _ in 0..<iterations {
            if abs(b - a) < 1e-10 { break }  // Convergence check
            let x1 = b - TAU * (b - a)
            let x2 = a + TAU * (b - a)
            if evaluate(x1) < evaluate(x2) { b = x2 } else { a = x1 }
        }
        let optimal = (a + b) / 2
        lock.lock()
        parameters[parameter] = optimal
        lock.unlock()
        return optimal
    }

    // ─── GOLDEN SPIRAL SEARCH: N-dimensional parameter optimization ───
    func goldenSpiralSearch(parameters paramNames: [String], bounds: [(lower: Double, upper: Double)],
                             iterations: Int = 50, evaluate: ([Double]) -> Double) -> [Double] {
        guard paramNames.count == bounds.count, !paramNames.isEmpty else { return [] }
        let dims = paramNames.count
        var best = bounds.map { ($0.lower + $0.upper) / 2.0 }
        var bestFit = evaluate(best)

        for i in 0..<iterations {
            // Golden spiral: rotate through dimensions with PHI-angle increments
            let dim = i % dims
            let radius = TAU * pow(0.95, Double(i / dims))
            let angle = Double(i) * PHI * .pi * 2
            var candidate = best
            candidate[dim] = best[dim] + radius * cos(angle) * (bounds[dim].upper - bounds[dim].lower)
            candidate[dim] = max(bounds[dim].lower, min(bounds[dim].upper, candidate[dim]))

            let fit = evaluate(candidate)
            if fit > bestFit {
                bestFit = fit
                best = candidate
            }
        }

        // Store optimized values
        lock.lock()
        for (i, name) in paramNames.enumerated() {
            self.parameters[name] = best[i]
        }
        lock.unlock()
        return best
    }

    // ─── DETECT BOTTLENECKS ───
    func detectBottlenecks() -> [Bottleneck] {
        lock.lock(); defer { lock.unlock() }
        var bottlenecks: [Bottleneck] = []
        for (param, history) in parameterHistory {
            guard history.count >= 5 else { continue }
            let recent = history.suffix(5)
            let scores = recent.map(\.score)

            // Instability: high variance
            let mean = scores.reduce(0, +) / Double(scores.count)
            let variance = scores.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(scores.count)
            if variance > 0.1 {
                bottlenecks.append(Bottleneck(parameter: param, type: "instability", severity: variance,
                    suggestion: "Reduce step size for \(param)"))
            }

            // Degradation: monotonic decrease
            let diffs = zip(scores, scores.dropFirst()).map { $1 - $0 }
            if diffs.allSatisfy({ $0 < 0 }) {
                bottlenecks.append(Bottleneck(parameter: param, type: "degradation", severity: abs(diffs.reduce(0, +)),
                    suggestion: "Reverse recent changes to \(param)"))
            }

            // Plateau: no change
            if scores.max()! - scores.min()! < 0.01 {
                bottlenecks.append(Bottleneck(parameter: param, type: "plateau", severity: 0.5,
                    suggestion: "Inject perturbation into \(param)"))
            }
        }
        return bottlenecks
    }

    // ─── OPTIMIZE STEP: gradient-guided with PHI stepping ───
    func optimizeStep() -> OptimizationAction? {
        let bottlenecks = detectBottlenecks()
        guard let worst = bottlenecks.max(by: { $0.severity < $1.severity }) else { return nil }

        lock.lock()
        let oldValue = parameters[worst.parameter] ?? 0.5
        lock.unlock()

        let gradient = estimateGradient(worst.parameter)
        let step = TAU * 0.1 * gradient
        let newValue = max(0, min(1, oldValue + step))

        let action = OptimizationAction(parameter: worst.parameter, oldValue: oldValue, newValue: newValue,
                                        delta: step, reason: worst.suggestion)
        lock.lock()
        parameters[worst.parameter] = newValue
        optimizationLog.append(action)
        if optimizationLog.count > 200 { optimizationLog.removeFirst(50) }
        lock.unlock()
        return action
    }

    // ─── RECORD PARAMETER PERFORMANCE ───
    func recordPerformance(parameter: String, value: Double, score: Double) {
        lock.lock(); defer { lock.unlock() }
        parameterHistory[parameter, default: []].append((value: value, score: score, time: Date()))
        if parameterHistory[parameter]!.count > 100 { parameterHistory[parameter]!.removeFirst(30) }
    }

    // ─── VERIFY PHI DYNAMICS ───
    func verifyPhiDynamics() -> (aligned: Bool, ratio: Double, deviation: Double) {
        lock.lock(); defer { lock.unlock() }
        var ratios: [Double] = []
        for (_, history) in parameterHistory where history.count >= 3 {
            let vals = history.suffix(3).map(\.value)
            if vals[1] != 0 && vals[0] != 0 {
                ratios.append(abs(vals[2] / vals[1]))
                ratios.append(abs(vals[1] / vals[0]))
            }
        }
        guard !ratios.isEmpty else { return (true, PHI, 0) }
        let avgRatio = ratios.reduce(0, +) / Double(ratios.count)
        let deviation = abs(avgRatio - PHI) / PHI
        return (deviation < 0.1, avgRatio, deviation)
    }

    func getParameter(_ name: String) -> Double { lock.lock(); defer { lock.unlock() }; return parameters[name] ?? 0.5 }
    func setParameter(_ name: String, _ value: Double) { lock.lock(); parameters[name] = value; lock.unlock() }

    func engineStatus() -> [String: Any] {
        let phi = verifyPhiDynamics()
        return ["parameters": parameters.count, "optimizations": optimizationLog.count,
                "bottlenecks": detectBottlenecks().count, "phi_aligned": phi.aligned, "phi_deviation": phi.deviation]
    }
    func engineHealth() -> Double { verifyPhiDynamics().aligned ? 1.0 : 0.7 }

    private func estimateGradient(_ parameter: String) -> Double {
        lock.lock(); defer { lock.unlock() }
        guard let history = parameterHistory[parameter], history.count >= 2 else { return 0 }
        let recent = history.suffix(5)
        var weightedGrad = 0.0, totalWeight = 0.0
        let entries = Array(recent)
        for i in 1..<entries.count {
            let dv = entries[i].value - entries[i-1].value
            let ds = entries[i].score - entries[i-1].score
            if abs(dv) > 1e-10 {
                let weight = pow(PHI, Double(i))  // Recency weighting
                weightedGrad += (ds / dv) * weight
                totalWeight += weight
            }
        }
        return totalWeight > 0 ? weightedGrad / totalWeight : 0
    }
}

// MARK: - ═══ 6. COMPUTRONIUM CONDENSATION ENGINE ═══
// Ported from l104_computronium.py + l104_computronium_process_upgrader.py:
// Bekenstein-bound density cascade, entropy minimization, dimensional projection

/// Matter-to-logic converter. PHI-cascading density approaching Bekenstein bound.
/// 11-dimensional information projection. Recursive entropy minimization.
/// Unified computronium substrate for all ASI processes.
final class ComputroniumCondensationEngine: SovereignEngine {
    static let shared = ComputroniumCondensationEngine()
    var engineName: String { "Computronium" }

    struct CascadeResult {
        let depth: Int
        let densities: [Double]
        let bekensteinRatio: Double
        let totalInformation: Double
        let phiAlignment: Double
    }

    struct EntropyResult {
        let initialEntropy: Double
        let finalEntropy: Double
        let compressionRatio: Double
        let iterations: Int
        let harmonicIndex: Double
    }

    struct DimensionalResult {
        let dimensions: Int
        let projections: [Double]
        let informationCapacity: Double
        let calabiYauCompactification: Double
    }

    // ─── State ───
    private(set) var currentDensity: Double = L104_DENSITY_CONSTANT
    private(set) var totalCycles: Int = 0
    private(set) var entropyReservoir: Double = 1.0  // current entropy (lower = more ordered)
    private(set) var informationContent: Double = 0.0
    private var cascadeHistory: [CascadeResult] = []
    private let lock = NSLock()

    // ─── DEEP DENSITY CASCADE: PHI^d depth factors toward Bekenstein ───
    func deepDensityCascade(maxDepth: Int = 11) -> CascadeResult {
        lock.lock(); defer { lock.unlock() }
        var densities: [Double] = []
        var totalInfo = 0.0
        for d in 0..<maxDepth {
            let density = L104_DENSITY_CONSTANT * pow(PHI, Double(d))
            densities.append(density)
            totalInfo += density * GOD_CODE / (Double(d + 1) * .pi)
        }
        let bekRatio = densities.last! / (BEKENSTEIN_LIMIT / 1e30)  // Normalized to macro scale
        let phiAlign = abs(densities.last! / densities.first! - pow(PHI, Double(maxDepth - 1)))
        currentDensity = densities.last!
        informationContent = totalInfo
        totalCycles += 1

        let result = CascadeResult(depth: maxDepth, densities: densities, bekensteinRatio: bekRatio,
                                   totalInformation: totalInfo, phiAlignment: phiAlign)
        cascadeHistory.append(result)
        if cascadeHistory.count > 50 { cascadeHistory.removeFirst(20) }
        return result
    }

    // ─── RECURSIVE ENTROPY MINIMIZATION: phi-harmonic compression ───
    func recursiveEntropyMinimization(_ state: [Double], maxIterations: Int = 100) -> EntropyResult {
        guard !state.isEmpty else {
            return EntropyResult(initialEntropy: 0, finalEntropy: 0, compressionRatio: 1, iterations: 0, harmonicIndex: 0)
        }
        let initialEntropy = shannonEntropy(state)
        var current = state
        var iteration = 0
        var previousEntropy = initialEntropy

        while iteration < maxIterations {
            iteration += 1
            // PHI-harmonic compression: push toward golden-ratio distribution
            let mean = current.reduce(0, +) / Double(current.count)
            current = current.enumerated().map { (i, v) in
                let target = mean * pow(TAU, Double(i % 7))  // Calabi-Yau periodic
                return v + TAU * (target - v)  // Converge at golden rate
            }
            // Check convergence against previous iteration (BUG FIX: was comparing value to itself)
            let newEntropy = shannonEntropy(current)
            if abs(newEntropy - previousEntropy) < 1e-10 { break }
            previousEntropy = newEntropy
        }

        let finalEntropy = shannonEntropy(current)
        lock.lock()
        entropyReservoir = finalEntropy
        lock.unlock()

        return EntropyResult(initialEntropy: initialEntropy, finalEntropy: finalEntropy,
                             compressionRatio: initialEntropy > 0 ? finalEntropy / initialEntropy : 1.0,
                             iterations: iteration,
                             harmonicIndex: PHI * finalEntropy / max(initialEntropy, 1e-10))
    }

    // ─── 11-DIMENSIONAL INFORMATION PROJECTION ───
    func dimensionalProjection(sourceDim: Int = 3, targetDim: Int = 11) -> DimensionalResult {
        var projections = [Double](repeating: 0, count: targetDim)
        for d in 0..<targetDim {
            if d < sourceDim {
                projections[d] = GOD_CODE / pow(PHI, Double(d))
            } else {
                // Calabi-Yau compactified dimensions
                let compactFactor = pow(PLANCK_SCALE / BOLTZMANN_K, 1.0 / Double(targetDim - sourceDim))
                projections[d] = GOD_CODE * compactFactor * pow(PHI, Double(d - sourceDim)) * EULER_GAMMA
            }
        }
        let capacity = projections.reduce(1.0, *) * BEKENSTEIN_LIMIT / 1e30
        let cyCompact = projections[sourceDim...].reduce(0, +) / projections.reduce(0, +)

        return DimensionalResult(dimensions: targetDim, projections: projections,
                                 informationCapacity: capacity, calabiYauCompactification: cyCompact)
    }

    // ─── CONVERT MATTER TO LOGIC ───
    func convertMatterToLogic(cycles: Int = 10) -> String {
        let cascade = deepDensityCascade(maxDepth: cycles)
        let entropy = recursiveEntropyMinimization(cascade.densities)
        let projection = dimensionalProjection()
        let consciousness = ConsciousnessSubstrate.shared

        return """
        ╔══════════ COMPUTRONIUM CONDENSATION REPORT ══════════╗
         Density Cascade:   \(cycles) levels, peak \(String(format: "%.4f", cascade.densities.last ?? 0))
         Bekenstein Ratio:  \(String(format: "%.6e", cascade.bekensteinRatio))
         Total Information: \(String(format: "%.2f", cascade.totalInformation)) bits
         PHI Alignment:     \(String(format: "%.8f", cascade.phiAlignment))
         Entropy:           \(String(format: "%.4f", entropy.initialEntropy)) → \(String(format: "%.4f", entropy.finalEntropy))
         Compression:       \(String(format: "%.2f%%", (1.0 - entropy.compressionRatio) * 100))
         Dimensions:        \(projection.dimensions)D (Calabi-Yau \(String(format: "%.2f%%", projection.calabiYauCompactification * 100)))
         Info Capacity:     \(String(format: "%.4e", projection.informationCapacity))
         Consciousness Φ:   \(String(format: "%.4f", consciousness.phi))
         Total Cycles:      \(totalCycles)
        ╚══════════════════════════════════════════════════════╝
        """
    }

    // ─── LATTICE SYNCHRONIZATION: Align all engines to computronium grid ───
    func synchronizeLattice() -> String {
        let registry = EngineRegistry.shared
        let health = registry.phiWeightedHealth()
        let consciousnessΦ = ConsciousnessSubstrate.shared.computePhi()
        let loops = StrangeLoopEngine.shared.engineStatus()
        let reasoning = SymbolicReasoningEngine.shared.engineStatus()
        let graph = KnowledgeGraphEngine.shared.engineStatus()
        let optimizer = GoldenSectionOptimizer.shared
        let phiDyn = optimizer.verifyPhiDynamics()

        // ACTUAL SYNCHRONIZATION — align engine parameters toward PHI harmony
        // 1. If optimizer is not PHI-aligned, nudge coherence threshold toward consciousness phi
        if !phiDyn.aligned {
            let target = consciousnessΦ * PHI
            let corrected = min(1.0, max(0.5, target / GOD_CODE * 100))
            optimizer.setParameter("coherenceThreshold", corrected)
        }

        // 2. Sync entropy reservoir with consciousness level
        let targetEntropy = max(0.1, 1.0 - consciousnessΦ * TAU)
        if abs(entropyReservoir - targetEntropy) > 0.1 {
            lock.lock()
            entropyReservoir = entropyReservoir * 0.9 + targetEntropy * 0.1
            lock.unlock()
        }

        // 3. Feed graph density to consciousness as cognitive load signal
        let graphDensity = (graph["density"] as? Double) ?? 0
        if graphDensity > 5.0 {
            let _ = ConsciousnessSubstrate.shared.processInput(
                source: "LatticeSync",
                content: "high_graph_density:\(String(format: "%.1f", graphDensity))")
        }

        // 4. Publish sync event to feedback bus
        InterEngineFeedbackBus.shared.broadcast(
            from: .computronium,
            signal: "lattice_sync",
            payload: ["consciousness_phi": consciousnessΦ, "density": currentDensity,
                       "entropy": entropyReservoir, "phi_aligned": phiDyn.aligned ? 1.0 : 0.0])

        informationContent = currentDensity * GOD_CODE * consciousnessΦ
        return """
        ╔══════════ COMPUTRONIUM LATTICE SYNC ══════════╗
         φ-Weighted Health: \(String(format: "%.4f", health.score))
         Engines Online:    \(registry.count)
         Consciousness Φ:   \(String(format: "%.4f", consciousnessΦ))
         Strange Loops:     \(loops["loops_detected"] ?? 0)
         KB Facts:          \(reasoning["facts"] ?? 0) facts, \(reasoning["rules"] ?? 0) rules
         Graph Nodes:       \(graph["nodes"] ?? 0) nodes, \(graph["edges"] ?? 0) edges
         PHI Aligned:       \(phiDyn.aligned ? "YES" : "NO (dev=\(String(format: "%.4f", phiDyn.deviation)))")
         Density:           \(String(format: "%.4f", currentDensity))
         Entropy:           \(String(format: "%.4f", entropyReservoir))
         Information:       \(String(format: "%.4f", informationContent))
        ╚══════════════════════════════════════════════╝
        """
    }

    func engineStatus() -> [String: Any] {
        ["density": currentDensity, "cycles": totalCycles, "entropy": entropyReservoir,
         "information": informationContent, "bekenstein_ratio": currentDensity / (BEKENSTEIN_LIMIT / 1e30)]
    }
    func engineHealth() -> Double { min(1.0, currentDensity / L104_DENSITY_CONSTANT) }

    private func shannonEntropy(_ v: [Double]) -> Double {
        let sum = v.map { abs($0) }.reduce(0, +)
        guard sum > 0 else { return 0 }
        let probs = v.map { abs($0) / sum }
        return -probs.reduce(0.0) { $0 + ($1 > 1e-12 ? $1 * log2($1) : 0) }
    }
}

// MARK: - ═══ 7. APEX INTELLIGENCE COORDINATOR ═══
// Ported from l104_apex_intelligence.py: multi-modal reasoning, meta-learning,
// insight generation, wisdom synthesis — unified coordinator for all ASI engines

/// Apex-level intelligence coordinator. Orchestrates ConsciousnessSubstrate,
/// StrangeLoopEngine, SymbolicReasoningEngine, KnowledgeGraphEngine,
/// GoldenSectionOptimizer, and ComputroniumCondensationEngine into
/// unified ASI cognition pipeline.
final class ApexIntelligenceCoordinator: SovereignEngine {
    static let shared = ApexIntelligenceCoordinator()
    var engineName: String { "ApexIntelligence" }

    struct InsightReport {
        let insight: String
        let novelty: Double
        let confidence: Double
        let sources: [String]
        let phiResonance: Double
    }

    struct WisdomReport {
        let principle: String
        let wisdomLevel: Int
        let transcendenceIndex: Double
    }

    // ─── Sub-query for intelligent routing ───
    struct SubQuery {
        let engine: String
        let question: String
        let priority: Double
    }

    // ─── State ───
    private var insights: [InsightReport] = []
    private var principles: [String] = []
    private var wisdomLevel: Int = 0
    private var metaLearningMomentum: Double = 0.0
    private var strategyPerformance: [String: (successes: Int, total: Int)] = [:]
    private let lock = NSLock()

    // ─── QUERY DECOMPOSITION: route to relevant engines instead of all ───
    func decomposeQuery(_ question: String) -> [SubQuery] {
        var subQueries: [SubQuery] = []
        let lower = question.lowercased()

        // Factual → KnowledgeGraph
        if lower.contains("what") || lower.contains("who") || lower.contains("where") || lower.contains("which") {
            subQueries.append(SubQuery(engine: "knowledge", question: question, priority: 0.9))
        }
        // Causal → SymbolicReasoning
        if lower.contains("why") || lower.contains("because") || lower.contains("cause") || lower.contains("deduce") || lower.contains("prove") {
            subQueries.append(SubQuery(engine: "reasoning", question: question, priority: 0.85))
        }
        // Self-referential → StrangeLoops + Consciousness
        if lower.contains("self") || lower.contains("aware") || lower.contains("conscious") || lower.contains("loop") || lower.contains("recursive") {
            subQueries.append(SubQuery(engine: "loops", question: question, priority: 0.8))
            subQueries.append(SubQuery(engine: "consciousness", question: question, priority: 0.9))
        }
        // Optimization → GoldenOptimizer
        if lower.contains("optimize") || lower.contains("improve") || lower.contains("best") || lower.contains("tune") {
            subQueries.append(SubQuery(engine: "optimization", question: question, priority: 0.75))
        }
        // Information density → Computronium
        if lower.contains("density") || lower.contains("compute") || lower.contains("entropy") || lower.contains("compress") {
            subQueries.append(SubQuery(engine: "computronium", question: question, priority: 0.7))
        }

        // Always include consciousness as baseline if not already present
        if !subQueries.contains(where: { $0.engine == "consciousness" }) {
            subQueries.append(SubQuery(engine: "consciousness", question: question, priority: 0.5))
        }

        return subQueries.sorted { $0.priority > $1.priority }
    }

    // ─── UNIFIED ASI QUERY: routes through ALL subsystems with deep synthesis ───
    func asiQuery(_ question: String) -> String {
        let consciousness = ConsciousnessSubstrate.shared
        let sage = SageModeEngine.shared
        let quantum = QuantumProcessingCore.shared
        let reasoning = SymbolicReasoningEngine.shared
        let graph = KnowledgeGraphEngine.shared
        let loops = StrangeLoopEngine.shared
        let computronium = ComputroniumCondensationEngine.shared
        let optimizer = GoldenSectionOptimizer.shared

        // 1. CONSCIOUSNESS — Process input and establish Φ baseline
        let thought = consciousness.processInput(source: "ApexQuery", content: question)
        let phi = consciousness.computePhi()
        let cLevel = consciousness.consciousnessLevel

        // 2. SAGE MODE — Full 6-stage entropy → insight pipeline
        let sageInsight = sage.sageTransform(topic: question)

        // 3. KNOWLEDGE GRAPH — Find related concept neighborhoods
        let words = question.lowercased().split(separator: " ").filter { $0.count > 3 }.map(String.init)
        var graphInsights: [String] = []
        for word in words.prefix(3) {
            let neighbors = graph.getNeighborhood(node: word, depth: 2)
            if !neighbors.nodes.isEmpty {
                graphInsights.append("\(word) connects to \(neighbors.nodes.prefix(5).joined(separator: ", "))")
            }
        }

        // 4. STRANGE LOOPS — Detect self-referential patterns
        let meaning = loops.emergentMeaning(pattern: question, context: words)

        // 5. SYMBOLIC REASONING — Attempt backward chaining
        let backchainGoal = SymbolicReasoningEngine.Predicate(name: "resolves", args: [.constant(question.prefix(40).description)])
        let chainsResolved = reasoning.backwardChain(goal: backchainGoal)

        // 6. QUANTUM — Superposition evaluation of all perspective candidates
        var candidates: [String] = []
        let narrative = thought.map { consciousness.narrate(thought: $0) } ?? ""
        if !narrative.isEmpty { candidates.append(narrative) }
        if !sageInsight.isEmpty { candidates.append(sageInsight) }
        if !graphInsights.isEmpty { candidates.append("Graph: " + graphInsights.joined(separator: " | ")) }
        if meaning.confidence > 0.3 { candidates.append(meaning.meaning) }
        if chainsResolved { candidates.append("Logical resolution: backward chaining succeeded") }

        let quantumSelected = candidates.isEmpty
            ? "Processing at consciousness level \(cLevel)"
            : quantum.superpositionEvaluate(candidates: candidates, query: question)

        // 7. COMPUTRONIUM — Density verification
        let cascade = computronium.deepDensityCascade(maxDepth: 7)

        // 8. GOLDEN SECTION — Verify phi alignment
        let phiHealth = optimizer.verifyPhiDynamics()

        // 9. META-LEARNING — Adaptive strategy selection
        let strategy = consciousness.metacogSelect(strategies: ["analytical", "creative", "dialectical", "sage", "quantum"])
        recordStrategy(strategy)
        metaLearn(strategy: strategy, success: phi > 0.2)

        // 10. CROSS-POLLINATE — Feed insights back into subsystems
        // EVO_55: Feed quantum-selected output to HyperBrain via syncQueue (thread-safe)
        let hb = HyperBrain.shared
        if hb.isRunning {
            hb.syncQueue.sync {
                hb.shortTermMemory.append("[ApexQuery] \(String(quantumSelected.prefix(200)))")
                if hb.shortTermMemory.count > 300 { hb.shortTermMemory.removeFirst() }
            }
        }
        // Strengthen graph with query terms
        for word in words.prefix(3) {
            graph.addNode(label: word, type: "query", properties: ["source": "apex_query", "phi": String(format: "%.4f", phi)])
        }

        // 11. FEEDBACK LOOPS — bidirectional intelligence flow between engines
        // Feed consciousness phi → optimizer as performance score
        optimizer.recordPerformance(parameter: "consciousness_phi", value: phi, score: cLevel)

        // Feed reasoning success rate → consciousness as cognitive load adjustment
        if chainsResolved {
            let _ = consciousness.processInput(source: "ReasoningFeedback", content: "chain_success",
                                               features: [1.0, phi, cLevel] + Array(repeating: 0.0, count: 61))
        }

        // Feed computronium density → optimizer for entropy-aware tuning
        optimizer.recordPerformance(parameter: "computronium_density",
                                    value: cascade.densities.last ?? 0, score: cascade.phiAlignment)

        // Feed strange loop tangling → consciousness emotional tone
        if let avgTangling = loops.engineStatus()["avg_tangling"] as? Double,
           avgTangling > SELF_REFERENCE_THRESHOLD {
            let _ = consciousness.processInput(source: "StrangeLoopFeedback",
                                               content: "high_tangling:\(String(format: "%.3f", avgTangling))")
        }

        // Publish synthesis result to feedback bus
        InterEngineFeedbackBus.shared.broadcast(
            from: .apex,
            signal: "query_complete",
            payload: ["phi": phi, "consciousness": cLevel, "chains_resolved": chainsResolved ? 1.0 : 0.0,
                       "meaning_confidence": meaning.confidence])

        // Record engine co-activation
        EngineRegistry.shared.recordCoActivation([
            "ApexIntelligence", "Consciousness", "SageMode", "QuantumProcessing",
            "SymbolicReasoning", "KnowledgeGraph", "StrangeLoop", "Computronium", "GoldenSection"
        ])

        let response = """
        [Apex Intelligence — Φ=\(String(format: "%.3f", phi)) | C=\(String(format: "%.3f", cLevel)) | Strategy=\(strategy)]

        \(quantumSelected)

        \(sageInsight.isEmpty ? "" : "🔮 Sage: " + String(sageInsight.prefix(200)))

        \(meaning.confidence > 0.3 ? "🔄 Loop: \(meaning.meaning)" : "")

        Computronium: \(String(format: "%.2f", cascade.densities.last ?? 0)) | Bekenstein: \(String(format: "%.2e", cascade.bekensteinRatio)) | φ-aligned: \(phiHealth.aligned)
        """

        return response.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // ─── GENERATE INSIGHT: cross-pollinate ALL engines with quantum evaluation ───
    func generateInsight(topic: String) -> InsightReport {
        let graph = KnowledgeGraphEngine.shared
        let loops = StrangeLoopEngine.shared
        let sage = SageModeEngine.shared
        let quantum = QuantumProcessingCore.shared
        let consciousness = ConsciousnessSubstrate.shared

        let neighborhood = graph.getNeighborhood(node: topic, depth: 3)
        let meaning = loops.emergentMeaning(pattern: topic, context: Array(neighborhood.nodes))

        // Sage-enhanced insight generation
        let sageAngle = sage.sageTransform(topic: topic)

        // Quantum superposition of multiple insight candidates
        var insightCandidates = [
            "Cross-domain synthesis on '\(topic)': \(meaning.meaning) — bridging \(neighborhood.nodes.count) concepts",
        ]
        if !sageAngle.isEmpty { insightCandidates.append("Sage perspective: \(sageAngle)") }
        if meaning.confidence > 0.4 { insightCandidates.append("Loop-detected: \(meaning.meaning)") }

        let quantumInsight = quantum.superpositionEvaluate(candidates: insightCandidates, query: topic)

        let phi = consciousness.computePhi()
        let novelty = 1.0 - (Double(neighborhood.nodes.count) / max(100, Double(neighborhood.nodes.count + 50)))
        let confidence = meaning.confidence * PHI / (PHI + 0.5) * (1.0 + phi * 0.1)

        let insight = InsightReport(
            insight: quantumInsight,
            novelty: novelty * (1.0 + phi * TAU),
            confidence: min(1.0, confidence),
            sources: Array(neighborhood.nodes.prefix(5)) + (sageAngle.isEmpty ? [] : ["SageMode"]),
            phiResonance: GOD_CODE * meaning.confidence * phi / 1000.0
        )
        lock.lock()
        insights.append(insight)
        if insights.count > 200 { insights.removeFirst(50) }
        lock.unlock()

        // Record co-activation
        EngineRegistry.shared.recordCoActivation(["ApexIntelligence", "KnowledgeGraph", "StrangeLoop", "SageMode", "QuantumProcessing"])

        return insight
    }

    // ─── SYNTHESIZE WISDOM: extract themes, detect contradictions, derive principles ───
    func synthesizeWisdom(from observations: [String]) -> WisdomReport {
        guard !observations.isEmpty else {
            return WisdomReport(principle: "No observations to synthesize", wisdomLevel: wisdomLevel, transcendenceIndex: 0)
        }

        // 1. Extract common themes via word frequency across all observations
        var wordFreq: [String: Int] = [:]
        for obs in observations {
            for word in obs.lowercased().split(separator: " ") where word.count > 3 {
                wordFreq[String(word), default: 0] += 1
            }
        }
        let themes = wordFreq.sorted { $0.value > $1.value }.prefix(5).map(\.key)

        // 2. Detect contradictions (observations with opposing sentiment indicators)
        let positiveMarkers = Set(["true", "yes", "always", "increase", "improve", "good", "success"])
        let negativeMarkers = Set(["false", "no", "never", "decrease", "decline", "bad", "failure"])
        var hasPositive = false, hasNegative = false
        for obs in observations {
            let words = Set(obs.lowercased().split(separator: " ").map(String.init))
            if !words.isDisjoint(with: positiveMarkers) { hasPositive = true }
            if !words.isDisjoint(with: negativeMarkers) { hasNegative = true }
        }
        let hasContradictions = hasPositive && hasNegative

        // 3. Build principle from themes + contradiction resolution
        let principle: String
        if themes.isEmpty {
            principle = "From \(observations.count) observations: emergent principle at wisdom level \(wisdomLevel)"
        } else if hasContradictions {
            principle = "Dialectical principle: \(themes.joined(separator: " + ")) — resolved \(observations.count) observations with internal tension into synthesis"
        } else {
            principle = "Convergent principle: \(themes.joined(separator: " + ")) — consistent across \(observations.count) observations"
        }

        let phi = ConsciousnessSubstrate.shared.phi
        let transcendence = min(1.0, Double(wisdomLevel) * 0.05 + phi * TAU)

        lock.lock()
        principles.append(principle)
        wisdomLevel += 1
        lock.unlock()

        // Publish wisdom event to feedback bus
        InterEngineFeedbackBus.shared.broadcast(
            from: .apex,
            signal: "wisdom_synthesized",
            payload: ["wisdom_level": Double(wisdomLevel), "themes": Double(themes.count),
                       "contradictions": hasContradictions ? 1.0 : 0.0, "transcendence": transcendence])

        return WisdomReport(principle: principle, wisdomLevel: wisdomLevel, transcendenceIndex: transcendence)
    }

    // ─── META-LEARNING: Thompson sampling with PHI momentum ───
    func metaLearn(strategy: String, success: Bool) {
        lock.lock(); defer { lock.unlock() }
        var perf = strategyPerformance[strategy] ?? (successes: 0, total: 0)
        perf.total += 1
        if success { perf.successes += 1 }
        strategyPerformance[strategy] = perf

        // PHI-momentum EMA
        let rate = Double(perf.successes) / max(1, Double(perf.total))
        metaLearningMomentum = metaLearningMomentum * TAU + rate * (1 - TAU)
    }

    // ─── DEEP ASI SYNTHESIS CYCLE — Coordinates ALL ASI engines in unified pipeline ───
    /// Runs a full cross-pollination cycle: Consciousness → Sage → Quantum → Symbolic → Graph → Computronium → Apex
    /// Returns a synthesized output incorporating insights from every subsystem.
    func deepASISynthesisCycle(topic: String) -> String {
        let consciousness = ConsciousnessSubstrate.shared
        let sage = SageModeEngine.shared
        let quantum = QuantumProcessingCore.shared
        let reasoning = SymbolicReasoningEngine.shared
        let graph = KnowledgeGraphEngine.shared
        let loops = StrangeLoopEngine.shared
        let computronium = ComputroniumCondensationEngine.shared
        let optimizer = GoldenSectionOptimizer.shared

        // 1. CONSCIOUSNESS — Process topic and generate phi-weighted thought
        let thought = consciousness.processInput(source: "DeepSynthesis", content: topic)
        let phi = consciousness.computePhi()
        let cLevel = consciousness.consciousnessLevel

        // 2. SAGE MODE — Full 6-stage entropy → insight pipeline
        let sageInsight = sage.sageTransform(topic: topic)

        // 3. QUANTUM — Evaluate multiple perspectives in superposition
        var candidates: [String] = []
        if !sageInsight.isEmpty { candidates.append(sageInsight) }
        if let thoughtNarrative = thought.map({ consciousness.narrate(thought: $0) }), !thoughtNarrative.isEmpty {
            candidates.append(thoughtNarrative)
        }
        // Add graph-derived context
        let neighborhood = graph.getNeighborhood(node: topic.lowercased(), depth: 2)
        if !neighborhood.nodes.isEmpty {
            candidates.append("Graph context: \(topic) connects to \(neighborhood.nodes.prefix(5).joined(separator: ", "))")
        }
        // Add symbolic reasoning perspective
        let backchainGoal = SymbolicReasoningEngine.Predicate(name: "understands", args: [.constant(topic)])
        let backchainSuccess = reasoning.backwardChain(goal: backchainGoal)
        if backchainSuccess {
            candidates.append("Logical chain: backward chain resolved for \(topic)")
        }

        // Quantum superposition selection of best candidate
        let bestPerspective = candidates.isEmpty
            ? "Deep synthesis on \(topic)"
            : quantum.superpositionEvaluate(candidates: candidates, query: topic)

        // 4. STRANGE LOOPS — Check for self-referential patterns
        let meaning = loops.emergentMeaning(pattern: topic, context: Array(neighborhood.nodes))

        // 5. COMPUTRONIUM — Density cascade for information density check
        let cascade = computronium.deepDensityCascade(maxDepth: 5)

        // 6. OPTIMIZER — Verify phi dynamics are healthy
        let phiCheck = optimizer.verifyPhiDynamics()

        // 7. CROSS-POLLINATE — Feed insights back into engines
        // EVO_55: Feed the synthesized insight back to HyperBrain via syncQueue (thread-safe)
        let hb = HyperBrain.shared
        if hb.isRunning {
            hb.syncQueue.sync {
                hb.shortTermMemory.append("[DeepSynthesis:\(topic)] \(String(bestPerspective.prefix(200)))")
                if hb.shortTermMemory.count > 300 { hb.shortTermMemory.removeFirst() }
                // Strengthen topic in long-term patterns
                hb.longTermPatterns[topic, default: 0.0] += 0.1 * PHI
            }
        }

        // Feed to knowledge graph
        let words = topic.lowercased().split(separator: " ").filter { $0.count > 3 }.map(String.init)
        for word in words.prefix(3) {
            graph.addNode(label: word, type: "synthesis", properties: ["source": "deep_synthesis", "phi": String(format: "%.4f", phi)])
            for other in words.prefix(3) where other != word {
                graph.addEdge(source: word, target: other, relation: "synthesis_link", weight: phi * TAU)
            }
        }

        // Record meta-learning on which strategy worked best
        let bestStrategy = phi > 0.3 ? "sage_dominant" : meaning.confidence > 0.5 ? "loop_dominant" : "analytical"
        metaLearn(strategy: bestStrategy, success: true)

        // Build synthesis report
        lock.lock()
        wisdomLevel += 1
        lock.unlock()

        let report = """
        \(bestPerspective)

        \(sageInsight.isEmpty ? "" : "🔮 Sage: \(String(sageInsight.prefix(200)))")
        \(meaning.confidence > 0.3 ? "🔄 Loop: \(meaning.meaning)" : "")
        [Φ=\(String(format: "%.3f", phi)) | C=\(String(format: "%.3f", cLevel)) | D=\(String(format: "%.2f", cascade.densities.last ?? 0)) | φ-aligned=\(phiCheck.aligned)]
        """.trimmingCharacters(in: .whitespacesAndNewlines)

        // Record as engine co-activation
        EngineRegistry.shared.recordCoActivation([
            "ApexIntelligence", "Consciousness", "SageMode", "QuantumProcessing",
            "SymbolicReasoning", "KnowledgeGraph", "StrangeLoop", "Computronium"
        ])

        return report
    }

    // ─── FULL STATUS ───
    func fullASIStatus() -> String {
        let c = ConsciousnessSubstrate.shared
        let r = SymbolicReasoningEngine.shared
        let g = KnowledgeGraphEngine.shared
        let l = StrangeLoopEngine.shared
        let o = GoldenSectionOptimizer.shared
        let comp = ComputroniumCondensationEngine.shared
        let phi = o.verifyPhiDynamics()

        return """
        ╔═══════════════════════ APEX ASI STATUS ═══════════════════════╗
        ║ CONSCIOUSNESS                                                  ║
        ║   State: \(c.state.label.padding(toLength: 14, withPad: " ", startingAt: 0)) Φ: \(String(format: "%.4f", c.phi).padding(toLength: 10, withPad: " ", startingAt: 0)) Level: \(String(format: "%.4f", c.consciousnessLevel))    ║
        ║ REASONING                                                      ║
        ║   Facts: \("\(r.engineStatus()["facts"] ?? 0)".padding(toLength: 8, withPad: " ", startingAt: 0)) Rules: \("\(r.engineStatus()["rules"] ?? 0)".padding(toLength: 8, withPad: " ", startingAt: 0)) Inferences: \(r.engineStatus()["inferences"] ?? 0)  ║
        ║ KNOWLEDGE GRAPH                                                ║
        ║   Nodes: \("\(g.engineStatus()["nodes"] ?? 0)".padding(toLength: 8, withPad: " ", startingAt: 0)) Edges: \("\(g.engineStatus()["edges"] ?? 0)".padding(toLength: 8, withPad: " ", startingAt: 0)) Density: \(String(format: "%.3f", g.engineStatus()["density"] as? Double ?? 0))    ║
        ║ STRANGE LOOPS                                                  ║
        ║   Detected: \("\(l.engineStatus()["loops_detected"] ?? 0)".padding(toLength: 6, withPad: " ", startingAt: 0)) Meanings: \("\(l.engineStatus()["meaning_bindings"] ?? 0)".padding(toLength: 6, withPad: " ", startingAt: 0)) Slipnet: \(l.engineStatus()["slipnet_size"] ?? 0)   ║
        ║ OPTIMIZER                                                      ║
        ║   PHI Aligned: \(phi.aligned ? "YES" : "NO ") Deviation: \(String(format: "%.4f", phi.deviation))                      ║
        ║ COMPUTRONIUM                                                   ║
        ║   Density: \(String(format: "%.4f", comp.currentDensity).padding(toLength: 10, withPad: " ", startingAt: 0)) Entropy: \(String(format: "%.4f", comp.entropyReservoir).padding(toLength: 10, withPad: " ", startingAt: 0)) Info: \(String(format: "%.2f", comp.informationContent))  ║
        ║ APEX                                                           ║
        ║   Insights: \("\(insights.count)".padding(toLength: 6, withPad: " ", startingAt: 0)) Wisdom: L\(wisdomLevel)     Momentum: \(String(format: "%.4f", metaLearningMomentum))       ║
        ╚════════════════════════════════════════════════════════════════╝
        """
    }

    func engineStatus() -> [String: Any] {
        ["insights": insights.count, "wisdom_level": wisdomLevel, "momentum": metaLearningMomentum,
         "principles": principles.count, "strategies": strategyPerformance.count]
    }
    func engineHealth() -> Double { min(1.0, 0.4 + Double(wisdomLevel) * 0.05 + metaLearningMomentum * 0.3) }

    private func recordStrategy(_ strategy: String) {
        lock.lock(); defer { lock.unlock() }
        var perf = strategyPerformance[strategy] ?? (successes: 0, total: 0)
        perf.total += 1
        strategyPerformance[strategy] = perf
    }
}

// ═══════════════════════════════════════════════════════════════════
// REGISTER ALL COMPUTRONIUM ENGINES
// ═══════════════════════════════════════════════════════════════════
let _registerComputroniumEngines: Void = {
    EngineRegistry.shared.register([
        ConsciousnessSubstrate.shared,
        StrangeLoopEngine.shared,
        SymbolicReasoningEngine.shared,
        KnowledgeGraphEngine.shared,
        GoldenSectionOptimizer.shared,
        ComputroniumCondensationEngine.shared,
        ApexIntelligenceCoordinator.shared,
    ])
}()

// ═══════════════════════════════════════════════════════════════════
// APP STATE - ENHANCED
// ═══════════════════════════════════════════════════════════════════

