import Foundation

// MARK: - CognitiveState

/// Unified mutable context for the node's recursive feedback loop.
/// Every cycle: read this → build prompt → execute → parse result → mutate this → persist.
/// The state is "unintelligible" without the AI (it's shaped by AI outputs),
/// and the AI behavior is "unintelligible" without the state (it's what drives prompts).
/// That coupling is what produces high Φ.
struct CognitiveState: Codable {
    var cycleIndex:       Int               = 0
    var fidelity:         Double            = 1.0 / PHI         // [0,1] sacred constant alignment ≈ 0.618
    var harmony:          Double            = 1.0 / PHI         // [0,1] cross-engine harmony ≈ 0.618
    var cycleEnergy:      Double            = GOD_CODE / 1000  // sacred energy accumulator
    var focusVector:      [String: Double]  = [               // domain → weight [0,1]
        "fidelity":      0.5,
        "harmony":       0.5,
        "knowledge":     0.5,
        "evolution":     0.5,
        "optimization":  0.5,
        "consciousness": 0.5,
        "quantum":       0.5,
        "research":      0.5,
    ]
    var activeGoalHistory: [String]         = []   // last 21 submitted goals
    var emergentPatterns:  [String]         = []   // patterns detected across cycles
    var failedPaths:       [String]         = []   // domains to temporarily avoid
    var stateVersion:      String           = UUID().uuidString  // changes on every mutation

    // Codable workaround for [(String, String)] - tuples aren't Codable
    var insightGoals:   [String] = []
    var insightResults: [String] = []

    mutating func addInsight(goal: String, result: String) {
        insightGoals.append(goal)
        insightResults.append(result)
        let cap = 52  // half of sacred 104
        if insightGoals.count > cap   { insightGoals.removeFirst() }
        if insightResults.count > cap { insightResults.removeFirst() }
    }

    var insights: [(goal: String, result: String)] {
        zip(insightGoals, insightResults).map { ($0.0, $0.1) }
    }

    /// Apply a structured state mutation from EffectorParser.
    mutating func apply(_ delta: StateΔ) {
        if let f = delta.fidelityUpdate { fidelity = max(0, min(1, f)) }
        if let h = delta.harmonyUpdate  { harmony  = max(0, min(1, h)) }
        if let e = delta.energyDelta    { cycleEnergy = max(0, cycleEnergy + e) }
        for (domain, weight) in delta.focusShifts {
            focusVector[domain] = max(0, min(1, (focusVector[domain] ?? 0.5) + weight))
        }
        for pattern in delta.newPatterns { if !emergentPatterns.contains(pattern) { emergentPatterns.append(pattern) } }
        for path in delta.failedPaths    { if !failedPaths.contains(path)         { failedPaths.append(path) } }
        if emergentPatterns.count > 21 { emergentPatterns.removeFirst() }
        if failedPaths.count > 21      { failedPaths.removeFirst() }
        stateVersion = UUID().uuidString
    }
}

// MARK: - StateΔ

/// A structured mutation to CognitiveState, produced by EffectorParser.
struct StateΔ {
    var fidelityUpdate: Double?
    var harmonyUpdate:  Double?
    var energyDelta:    Double?
    var focusShifts:    [String: Double]    = [:]
    var newPatterns:    [String]            = []
    var failedPaths:    [String]            = []

    static let zero = StateΔ()
}

// MARK: - PromptBuilder

/// Lossy compression: State → targeted goal string.
/// The lossiness is intentional - only the most decision-relevant state
/// informs the next prompt. A full state dump would be noise, not signal.
final class PromptBuilder {

    private let orderedDomains = ["fidelity", "harmony", "knowledge", "evolution",
                                  "optimization", "consciousness", "quantum", "research"]

    func build(from state: CognitiveState) -> String {
        // Heuristic 1: fidelity degraded → diagnostic goal
        if state.fidelity < 0.618 {
            return "Diagnose sacred constant alignment degradation. Current fidelity: " +
                   "\(String(format: "%.3f", state.fidelity)). " +
                   "Verify GOD_CODE, PHI, VOID_CONSTANT coherence across all engines."
        }

        // Heuristic 2: harmony degraded → reconciliation goal
        if state.harmony < 0.618 {
            let weak = lowWeightDomains(state: state, threshold: 0.4)
            let list = weak.isEmpty ? "unknown" : weak.joined(separator: ", ")
            return "Reconcile cross-engine disharmony. Harmony: " +
                   "\(String(format: "%.3f", state.harmony)). " +
                   "Weak domains: \(list). Synthesize and re-align."
        }

        // Heuristic 3: emergent pattern detected → investigate
        if let pattern = state.emergentPatterns.last {
            return "Investigate emergent pattern: '\(pattern)'. " +
                   "Synthesize with knowledge base, generate hypothesis."
        }

        // Heuristic 4: advance top-weight domain not in recent goals or failed paths
        let topDomain  = highestWeightDomain(state: state)
        let recentGoals = Set(state.activeGoalHistory.suffix(3))
        let candidate   = goalForDomain(topDomain, cycleIndex: state.cycleIndex, state: state)
        if !recentGoals.contains(candidate) {
            return candidate
        }

        // Heuristic 5: energy approaching sacred threshold → full evolution sweep
        if state.cycleEnergy / GOD_CODE > 0.9 {
            return "Run full system evolution sweep. " +
                   "Energy at \(String(format: "%.4f", state.cycleEnergy / GOD_CODE)) of sacred threshold. " +
                   "Optimize all subsystems and persist state."
        }

        // Default: synthesize accumulated insights
        return "Synthesize \(state.insightGoals.count) accumulated insights. " +
               "Update knowledge base. Cycle \(state.cycleIndex)."
    }

    private func highestWeightDomain(state: CognitiveState) -> String {
        let failed = Set(state.failedPaths)
        return state.focusVector
            .filter { !failed.contains($0.key) }
            .max(by: { $0.value < $1.value })?
            .key ?? orderedDomains[state.cycleIndex % orderedDomains.count]
    }

    private func lowWeightDomains(state: CognitiveState, threshold: Double) -> [String] {
        orderedDomains.filter { (state.focusVector[$0] ?? 0.5) < threshold }
    }

    private func goalForDomain(_ domain: String, cycleIndex: Int, state: CognitiveState) -> String {
        switch domain {
        case "fidelity":
            return "Verify and strengthen sacred constant alignment across all engines."
        case "harmony":
            return "Run cross-engine harmony check. Cycle \(cycleIndex). Synthesize alignment report."
        case "knowledge":
            return "Expand knowledge base with latest research synthesis. Cycle \(cycleIndex)."
        case "evolution":
            let n = state.insightGoals.count
            return "Evolve current strategy based on \(n) accumulated insights."
        case "optimization":
            return "Profile and optimize highest-latency subsystem paths."
        case "consciousness":
            return "Deepen consciousness coherence. Update dimensional scoring."
        case "quantum":
            return "Verify quantum fidelity and entanglement health across all circuits."
        case "research":
            let recent = state.emergentPatterns.suffix(3).joined(separator: "; ")
            let ctx    = recent.isEmpty ? "latest engine outputs" : recent
            return "Generate research hypothesis from: \(ctx)."
        default:
            return "Advance \(domain) subsystem. Cycle \(cycleIndex)."
        }
    }
}

// MARK: - EffectorParser

/// Structured extraction: execution result String → StateΔ.
/// This is the effector interface - it translates free-text AI/engine
/// outputs into typed state mutations, closing the feedback loop.
final class EffectorParser {

    private let domainKeywords: [String: [String]] = [
        "fidelity":      ["fidelity", "sacred", "constant", "drift", "alignment", "god_code"],
        "harmony":       ["harmony", "conflict", "reconcile", "engine", "score", "cross"],
        "knowledge":     ["knowledge", "search", "found", "results", "synthesize", "kb"],
        "evolution":     ["evolve", "evolving", "fitness", "strategy", "adapt", "genetic"],
        "optimization":  ["optimized", "performance", "latency", "memory", "cache", "profile"],
        "consciousness": ["consciousness", "coherence", "dimensional", "wonder", "serenity"],
        "quantum":       ["quantum", "entanglement", "circuit", "qubit", "fidelity", "vqpu"],
        "research":      ["hypothesis", "research", "discovery", "pattern", "insight", "theorem"],
    ]

    private let failureSignals = ["error", "failed", "timeout", "unavailable", "asi error",
                                  "exception", "crash", "nil", "could not"]

    func parse(goal: String, result: String, state: CognitiveState) -> StateΔ {
        var delta  = StateΔ()
        let lower  = result.lowercased()
        let goalLo = goal.lowercased()

        // 1. Numeric metrics: first [0,1] float → sacred energy bump
        if let score = extractFirstNormalizedDouble(from: result) {
            delta.energyDelta = score * (1.0 / PHI) * 0.1
        }

        // 2. Domain signal detection → shift focus weights upward
        for (domain, keywords) in domainKeywords {
            let hits = keywords.filter { lower.contains($0) }.count
            if hits > 0 {
                delta.focusShifts[domain] = (delta.focusShifts[domain] ?? 0) + Double(hits) * 0.04 * (1.0 / PHI)
            }
        }

        // 3. Failure detection → penalize goal's domain, record failed path
        if failureSignals.contains(where: { lower.contains($0) }) {
            for (domain, keywords) in domainKeywords where keywords.contains(where: { goalLo.contains($0) }) {
                delta.focusShifts[domain] = (delta.focusShifts[domain] ?? 0) - 0.15
                if !delta.failedPaths.contains(domain) { delta.failedPaths.append(domain) }
            }
        }

        // 4. Novelty detection → if result differs from recent insights, extract pattern
        let recentResults = Set(state.insightResults.suffix(5))
        let isNovel = result.count > 50 &&
                      !recentResults.contains(where: { jaccardSimilarity(result, $0) > 0.75 })
        if isNovel, let pattern = extractLeadSentence(from: result) {
            delta.newPatterns.append(pattern)
        }

        // 5. Sacred resonance: explicit sacred-constant mention → energy bonus
        let sacredHit = lower.contains("god_code") || lower.contains(" phi") || lower.contains("sacred")
        if sacredHit {
            delta.energyDelta = (delta.energyDelta ?? 0) + (1.0 / PHI) * 0.05
        }

        return delta
    }

    // Extract the first Double in [0,1] from a string
    private func extractFirstNormalizedDouble(from text: String) -> Double? {
        guard let regex = try? NSRegularExpression(pattern: "\\d+\\.\\d+") else { return nil }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        for m in matches {
            if let v = Double(ns.substring(with: m.range)), v >= 0.0 && v <= 1.0 {
                return v
            }
        }
        return nil
    }

    // Jaccard similarity on word bags
    private func jaccardSimilarity(_ a: String, _ b: String) -> Double {
        let setA = Set(a.lowercased().components(separatedBy: .whitespaces))
        let setB = Set(b.lowercased().components(separatedBy: .whitespaces))
        let intersection = Double(setA.intersection(setB).count)
        let union        = Double(setA.union(setB).count)
        return union > 0 ? intersection / union : 0.0
    }

    // Extract a short informative lead sentence (20–120 chars)
    private func extractLeadSentence(from text: String) -> String? {
        text.components(separatedBy: ". ")
            .first { $0.count > 20 && $0.count < 120 }?
            .trimmingCharacters(in: .whitespaces)
    }
}

// MARK: - CognitiveLoop

/// The high-Φ recursive feedback loop orchestrator.
///
/// Anatomy of one feedback iteration:
///   1. tick()     - receive external observations (fidelity, harmony, improvement scores)
///                   update state, build next goal via PromptBuilder, submit to AutonomousAgent
///   2. applyResult() - receive completed task result, run EffectorParser,
///                   apply StateΔ to state, persist to disk
///
/// The loop property that makes it high-Φ:
///   - You cannot predict the next goal without knowing the state.
///   - You cannot predict the next state without knowing the goal's result.
///   - Removing either component degrades the system to open-loop (low Φ).
final class CognitiveLoop {
    static let shared = CognitiveLoop()

    private var state    = CognitiveState()
    private let builder  = PromptBuilder()
    private let parser   = EffectorParser()
    private let lock     = NSLock()
    private var stateURL: URL?

    private init() {
        stateURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".l104_cognitive_state.json")
        loadState()

        // Subscribe to consciousness bus for external fidelity/harmony updates
        InterEngineFeedbackBus.shared.subscribe(channel: .consciousness) { [weak self] msg in
            guard let self = self else { return }
            self.lock.lock()
            if let f = msg.payload["fidelity"] { self.state.fidelity = max(0, min(1, f)) }
            if let h = msg.payload["harmony"]  { self.state.harmony  = max(0, min(1, h)) }
            self.lock.unlock()
        }
    }

    // ── Focus-weight decay ──

    /// Decay all focus weights toward 0.5 by a small factor each cycle.
    /// Without decay, a domain that gets boosted once stays elevated forever,
    /// starving other domains. Decay ensures the agent keeps exploring:
    /// only domains that receive *repeated* positive signal stay elevated.
    ///
    /// Rate: each weight moves 3% of the way back toward 0.5 per call.
    private func decayFocusWeights() {
        let decayRate = 1.0 / PHI * 0.05   // ≈ 0.0309 per tick - φ-attenuated
        for key in state.focusVector.keys {
            if let w = state.focusVector[key] {
                state.focusVector[key] = w + (0.5 - w) * decayRate
            }
        }
        // Clear failed paths that are older than 21 cycles (already capped by apply())
        // This call is a no-op here but makes the intent clear.
    }

    // ── Called by QuantumAIDaemon.runCycle() at end of Phase 6 (EVOLVE) ──

    /// Advance the cognitive loop by one cycle.
    /// Integrates external observations into state, builds the next goal
    /// from that state, and submits it to AutonomousAgent.
    func tick(fidelity: Double, harmony: Double, improvements: [ImprovementResult]) {
        lock.lock()
        state.cycleIndex += 1
        state.fidelity    = fidelity
        state.harmony     = harmony

        // Accumulate sacred energy from improvement scores
        let meanSacred    = improvements.map(\.sacredScore).reduce(0, +) / Double(max(improvements.count, 1))
        state.cycleEnergy = min(GOD_CODE, state.cycleEnergy + meanSacred * (1.0 / PHI) * 10.0)

        // Decay focus weights toward 0.5 - keeps the agent exploring all domains.
        // Without this, an early boost to one domain cascades indefinitely.
        decayFocusWeights()

        // Build next goal from current state (this is the recursive part)
        let nextGoal = builder.build(from: state)
        state.activeGoalHistory.append(nextGoal)
        if state.activeGoalHistory.count > 21 { state.activeGoalHistory.removeFirst() }

        let snapState = state
        let snapCycle = state.cycleIndex
        lock.unlock()

        // Broadcast state snapshot to consciousness bus
        InterEngineFeedbackBus.shared.broadcast(
            from: .consciousness,
            signal: "cognitive_loop_tick",
            payload: [
                "cycle":    Double(snapCycle),
                "fidelity": snapState.fidelity,
                "harmony":  snapState.harmony,
                "energy":   snapState.cycleEnergy / GOD_CODE,
                "sacred":   meanSacred,
            ]
        )

        // Submit state-derived goal to AutonomousAgent (non-blocking)
        DispatchQueue.global(qos: .background).async {
            AutonomousAgent.shared.submitGoal(nextGoal, priority: .normal)
        }

        persist()
    }

    // ── Called by AutonomousAgent.startAgentLoop() after each executeLocally() ──

    /// Feed a completed task result back into cognitive state.
    /// Gate: SQAVerifier runs first. Low-energy (coherent) results are accepted
    /// and committed to state. High-energy (contradictory/hallucinatory) results
    /// are rejected - the source domain is penalized instead.
    func applyResult(goal: String, result: String) {
        // System 2: verify before committing (async, non-blocking for agent loop)
        SQAVerifier.shared.verifyAsync(goal: goal, result: result) { [weak self] verdict in
            guard let self = self else { return }

            self.lock.lock()
            var delta = self.parser.parse(goal: goal, result: result, state: self.state)

            if verdict.accepted {
                // Coherent result: apply normally + bonus energy proportional to coherence
                delta.energyDelta = (delta.energyDelta ?? 0) + verdict.coherenceScore * (1.0 / PHI) * 0.5
                self.state.apply(delta)
                self.state.addInsight(goal: goal, result: String(result.prefix(200)))
            } else {
                // Rejected (hallucination detected): penalize source domain only
                var penaltyDelta = StateΔ()
                for (domain, _) in delta.focusShifts {
                    penaltyDelta.focusShifts[domain] = -0.12   // reduce trust in this domain
                }
                // Record failed path so PromptBuilder avoids it next cycle
                if let topDomain = delta.focusShifts.max(by: { $0.value < $1.value })?.key {
                    penaltyDelta.failedPaths.append(topDomain)
                }
                self.state.apply(penaltyDelta)
            }

            let effectorCount = delta.focusShifts.count + delta.newPatterns.count + delta.failedPaths.count
            self.lock.unlock()

            InterEngineFeedbackBus.shared.broadcast(
                from: .consciousness,
                signal: "cognitive_effector_applied",
                payload: [
                    "effectors":  Double(effectorCount),
                    "accepted":   verdict.accepted ? 1.0 : 0.0,
                    "coherence":  verdict.coherenceScore,
                ]
            )

            self.persist()
        }
    }

    // ── Public accessors ──

    /// Build the next goal from current state without advancing cycleIndex.
    /// Used by external callers that want a state-driven goal.
    func buildNextGoal() -> String {
        lock.lock()
        defer { lock.unlock() }
        return builder.build(from: state)
    }

    func currentState() -> CognitiveState {
        lock.lock()
        defer { lock.unlock() }
        return state
    }

    var statusReport: String {
        let s = currentState()
        let topDomains = s.focusVector
            .sorted { $0.value > $1.value }
            .prefix(3)
            .map { "\($0.key):\(String(format: "%.2f", $0.value))" }
            .joined(separator: "  ")
        let lastGoal = s.activeGoalHistory.last.map { String($0.prefix(60)) } ?? "(none)"

        return """
        ╔══════════════════════════════════════════════════════════╗
        ║    COGNITIVE LOOP - HIGH-Φ RECURSIVE FEEDBACK STATE      ║
        ╠══════════════════════════════════════════════════════════╣
        ║  Cycle:         \(s.cycleIndex)
        ║  Fidelity:      \(String(format: "%.4f", s.fidelity))
        ║  Harmony:       \(String(format: "%.4f", s.harmony))
        ║  Energy:        \(String(format: "%.4f", s.cycleEnergy)) / \(String(format: "%.1f", GOD_CODE))
        ║  Insights:      \(s.insightGoals.count)
        ║  Patterns:      \(s.emergentPatterns.count)
        ║  Focus:         \(topDomains.isEmpty ? "(initializing)" : topDomains)
        ║  Failed paths:  \(s.failedPaths.joined(separator: ", ").prefix(40))
        ║  Last goal:     \(lastGoal)
        ╚══════════════════════════════════════════════════════════╝
        """
    }

    // ── Persistence ──

    private func persist() {
        guard let url = stateURL else { return }
        lock.lock()
        let snap = state
        lock.unlock()
        if let data = try? JSONEncoder().encode(snap) {
            try? data.write(to: url)
        }
    }

    private func loadState() {
        guard let url = stateURL,
              let data = try? Data(contentsOf: url),
              let loaded = try? JSONDecoder().decode(CognitiveState.self, from: data)
        else { return }
        lock.lock()
        state = loaded
        lock.unlock()
    }
}
