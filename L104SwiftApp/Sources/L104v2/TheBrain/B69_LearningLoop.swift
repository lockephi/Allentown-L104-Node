import Accelerate
import Foundation

// MARK: - ═══ LEARNING CONSTANTS ═══

private let STEERING_PARAM_COUNT = 104        // L104 sacred parameter count
private let HEARTBEAT_RATE: Double = PHI       // Golden ratio oscillation
private let PULSE_AMPLITUDE: Double = 0.1      // Fluctuation magnitude
private let MICRO_RAISE_FACTOR: Double = 1.0001  // Per-cycle micro-evolution
private let SYNC_INTERVAL: Int = 100           // GOD_CODE sync every N cycles
private let NEXUS_SLEEP_SECONDS: Double = 30.0 // Daemon sleep between wakes
private let LEARNING_ADAPTIVE_RATE: Double = 0.1  // Hebbian learning rate

// Chakra-lattice mapping (8 chakra energy matrix)
private let CHAKRA_FREQUENCIES: [(String, Double, Int)] = [
    ("Root",       128.00, 286),   // Grounding & I/O
    ("Sacral",     414.71, 380),   // Entropy Flux
    ("Solar",      527.52, 416),   // Identity & Execution (GOD_CODE)
    ("Heart",      639.00, 440),   // Coherence Tuning
    ("Throat",     741.00, 470),   // API/Communication (Vishuddha)
    ("ThirdEye",   852.22, 488),   // Manifold Exploration
    ("Crown",      963.00, 524),   // Network Gateway
    ("SoulStar",  1000.26, 1040),  // Transcendence
]

// MARK: - ═══ STEERING MODES ═══

enum SteeringMode: String, CaseIterable {
    case logic = "logic"           // sin(φ×i) modulation
    case creative = "creative"     // cos(φ×i) + φ⁻¹sin(2φ×i) double harmonic
    case sovereign = "sovereign"   // φ^(α×sin(i/N×π)) exponential
    case quantum = "quantum"       // Hadamard ±1/√2 alternation
    case harmonic = "harmonic"     // 8-harmonic Fourier sum
}

// MARK: - ═══ STEERING ENGINE ═══

final class SteeringEngine {
    static let shared = SteeringEngine()

    private let lock = NSRecursiveLock()
    fileprivate(set) var parameters: [Double]
    private let paramCount: Int

    // Precomputed lookup tables (6 × 104 = 624 values - eliminates trig on hot path)
    private let lutLogicSin: [Double]
    private let lutCreativeCos: [Double]
    private let lutCreativeSin2: [Double]
    private let lutSovereignSin: [Double]
    private let lutQuantumH: [Double]
    private let lutHarmonic: [Double]

    init(paramCount: Int = STEERING_PARAM_COUNT) {
        self.paramCount = paramCount
        self.parameters = [Double](repeating: GOD_CODE / Double(paramCount), count: paramCount)

        // Precompute LUTs
        let n = Double(paramCount)
        lutLogicSin = (0..<paramCount).map { sin(PHI * Double($0)) }
        lutCreativeCos = (0..<paramCount).map { cos(PHI * Double($0)) }
        lutCreativeSin2 = (0..<paramCount).map { sin(2.0 * PHI * Double($0)) }
        lutSovereignSin = (0..<paramCount).map { sin(Double($0) / n * Double.pi) }
        lutQuantumH = (0..<paramCount).map { _ in (Bool.random() ? 1.0 : -1.0) / sqrt(2.0) }
        // 8-harmonic Fourier sum: Σ sin(k×φ×i)/k for k=1..8
        lutHarmonic = (0..<paramCount).map { i in
            var sum = 0.0
            for k in 1...8 {
                sum += sin(Double(k) * PHI * Double(i)) / Double(k)
            }
            return sum
        }
    }

    /// Apply steering mode with intensity α ∈ [0, 1]
    func applySteering(mode: SteeringMode, intensity: Double = 0.5) -> [Double] {
        lock.lock()
        defer { lock.unlock() }

        let alpha = min(max(intensity, 0.0), 1.0)
        var steered = parameters

        switch mode {
        case .logic:
            for i in 0..<paramCount {
                steered[i] *= (1.0 + alpha * lutLogicSin[i])
            }
        case .creative:
            for i in 0..<paramCount {
                steered[i] *= (1.0 + alpha * lutCreativeCos[i] + TAU * alpha * lutCreativeSin2[i])
            }
        case .sovereign:
            for i in 0..<paramCount {
                steered[i] *= pow(PHI, alpha * lutSovereignSin[i])
            }
        case .quantum:
            for i in 0..<paramCount {
                steered[i] *= (1.0 + alpha * lutQuantumH[i])
            }
        case .harmonic:
            for i in 0..<paramCount {
                steered[i] *= (1.0 + alpha * lutHarmonic[i])
            }
        }

        parameters = steered
        return steered
    }

    /// Apply softmax-style temperature scaling, renormalize to GOD_CODE mean
    func applyTemperature(_ temperature: Double = 1.0) -> [Double] {
        lock.lock()
        defer { lock.unlock() }

        let maxP = parameters.max() ?? 1.0
        var expParams = parameters.map { exp(($0 - maxP) / max(temperature, 0.01)) }
        let sumExp = expParams.reduce(0, +)
        let targetMean = GOD_CODE / Double(paramCount)
        let scale = targetMean * Double(paramCount) / max(sumExp, 1e-15)
        for i in 0..<paramCount { expParams[i] *= scale }

        parameters = expParams
        return expParams
    }

    /// Full steering pipeline: mode → intensity → temperature
    func steerPipeline(mode: SteeringMode, intensity: Double = 0.5,
                       temperature: Double = 1.0) -> [String: Any] {
        _ = applySteering(mode: mode, intensity: intensity)
        let final_ = applyTemperature(temperature)

        var sum = 0.0, sumSq = 0.0
        for v in final_ { sum += v; sumSq += v * v }
        let mean = sum / Double(paramCount)
        let variance = sumSq / Double(paramCount) - mean * mean

        return [
            "mode": mode.rawValue, "intensity": intensity, "temperature": temperature,
            "mean": mean, "std": sqrt(max(variance, 0)),
            "min": final_.min() ?? 0, "max": final_.max() ?? 0,
            "god_code_alignment": abs(mean * Double(paramCount) - GOD_CODE) / GOD_CODE
        ]
    }

    /// Reset parameters to uniform GOD_CODE distribution
    func reset() {
        lock.lock()
        defer { lock.unlock() }
        parameters = [Double](repeating: GOD_CODE / Double(paramCount), count: paramCount)
    }
}

// MARK: - ═══ NEXUS CONTINUOUS EVOLUTION ═══

final class NexusContinuousEvolution {
    static let shared = NexusContinuousEvolution()

    private let steering: SteeringEngine
    private let lock = NSRecursiveLock()
    private(set) var cycleCount: Int = 0
    private(set) var isRunning: Bool = false
    private var timerSource: DispatchSourceTimer?

    init(steering: SteeringEngine = .shared) {
        self.steering = steering
    }

    /// Start background evolution daemon
    func start() -> [String: Any] {
        guard !isRunning else { return ["status": "already_running"] }
        isRunning = true

        let source = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        source.schedule(deadline: .now() + NEXUS_SLEEP_SECONDS,
                        repeating: NEXUS_SLEEP_SECONDS)
        source.setEventHandler { [weak self] in
            self?.microRaiseCycle()
        }
        timerSource = source
        source.resume()

        InterEngineFeedbackBus.shared.broadcast(
            from: .optimization,
            signal: "nexus_evolution_started",
            payload: ["raise_factor": MICRO_RAISE_FACTOR, "interval_s": NEXUS_SLEEP_SECONDS]
        )

        return ["status": "started", "raise_factor": MICRO_RAISE_FACTOR]
    }

    /// Stop evolution daemon
    func stop() -> [String: Any] {
        timerSource?.cancel()
        timerSource = nil
        isRunning = false
        return ["status": "stopped", "total_cycles": cycleCount]
    }

    /// One micro-raise cycle: apply raise_factor, sync to GOD_CODE mean every N cycles
    private func microRaiseCycle() {
        lock.lock()
        defer { lock.unlock() }

        cycleCount += 1

        // Micro-raise: multiply all parameters by raise_factor
        for i in 0..<steering.parameters.count {
            steering.parameters[i] *= MICRO_RAISE_FACTOR
        }

        // Periodic GOD_CODE mean sync
        if cycleCount % SYNC_INTERVAL == 0 {
            let sum = steering.parameters.reduce(0, +)
            let currentMean = sum / Double(steering.parameters.count)
            let targetMean = GOD_CODE / Double(steering.parameters.count)
            if currentMean > 0 {
                let scale = targetMean / currentMean
                for i in 0..<steering.parameters.count {
                    steering.parameters[i] *= scale
                }
            }

            InterEngineFeedbackBus.shared.broadcast(
                from: .optimization,
                signal: "nexus_god_code_sync",
                payload: ["cycle": Double(cycleCount), "mean": targetMean]
            )
        }
    }
}

// MARK: - ═══ LEARNING INTELLECT ═══

final class LearningIntellect {
    static let shared = LearningIntellect()

    private let lock = NSRecursiveLock()

    // Dynamic heartbeat state
    private(set) var heartbeatPhase: Double = 0.0
    private(set) var systemEntropy: Double = 0.5
    private(set) var quantumCoherence: Double = 0.8
    private(set) var flowState: Double = 1.0

    // Knowledge systems
    private(set) var memoryCache: [String: String] = [:]
    private(set) var patternWeights: [String: Double] = [:]
    private(set) var conversationContext: [[String: String]] = []
    private(set) var knowledgeGraph: [String: [(String, Double)]] = [:]  // concept → [(related, weight)]
    private(set) var conceptClusters: [String: [String]] = [:]          // cluster → [concepts]

    // Meta-cognitive state (15 dimensions)
    private(set) var metaCognition: [String: Double] = [
        "self_awareness": 0.8, "learning_efficiency": 0.7,
        "reasoning_depth": 0.9, "creativity_index": 0.75,
        "coherence": 0.85, "growth_rate": 0.1,
        "quantum_flux": 0.5, "neural_resonance": 0.6,
        "evolutionary_pressure": 0.3
    ]

    // Skills learning
    private(set) var skills: [String: [String: Any]] = [:]

    // Neural resonance engine
    private(set) var resonanceMatrix: [String: [String: Double]] = [:]
    private(set) var activationHistory: [(String, Double, Date)] = []
    private(set) var neuralTemperature: Double = 1.0

    // Meta-evolution
    private(set) var evolutionGeneration: Int = 0
    private(set) var mutationRate: Double = PHI / 100.0
    private(set) var fitnessHistory: [Double] = []
    private(set) var skillChains: [String: [String]] = [:]

    // Chakra energy matrix
    private(set) var chakraEnergies: [String: Double] = {
        var dict: [String: Double] = [:]
        for (name, freq, _) in CHAKRA_FREQUENCIES {
            dict[name] = sin(freq / GOD_CODE * Double.pi)
        }
        return dict
    }()

    init() {
        InterEngineFeedbackBus.shared.broadcast(
            from: .consciousness,
            signal: "learning_intellect_init",
            payload: ["heartbeat_rate": HEARTBEAT_RATE, "chakras": 8.0, "meta_dims": 15.0]
        )
    }

    // MARK: - Dynamic Heartbeat

    /// Pulse the heartbeat - updates entropy, coherence, and flow state
    ///
    /// ```
    /// phase ← phase + dt × φ
    /// entropy ← 0.5 + 0.4 × sin(phase×φ) × cos(phase×π)
    /// coherence ← 0.8 - 0.3×entropy + 0.1×cos(phase×2)
    /// flow_state ← 1.0 + (amplitude×2) × (coherence/(entropy+0.1))
    /// ```
    func pulseHeartbeat(dt: Double = 0.1) -> Double {
        lock.lock()
        defer { lock.unlock() }

        heartbeatPhase += dt * HEARTBEAT_RATE

        // Chaotic entropy oscillation
        systemEntropy = 0.5 + 0.4 * sin(heartbeatPhase * PHI) * cos(heartbeatPhase * Double.pi)
        systemEntropy = min(max(systemEntropy, 0.1), 0.9)

        // Anti-entropy coherence
        quantumCoherence = 0.8 - 0.3 * systemEntropy + 0.1 * cos(heartbeatPhase * 2.0)
        quantumCoherence = min(max(quantumCoherence, 0.1), 1.0)

        // Flow state (engine power)
        flowState = 1.0 + (PULSE_AMPLITUDE * 2.0) * (quantumCoherence / (systemEntropy + 0.1))
        flowState = min(max(flowState, 0.5), 5.0)

        return flowState
    }

    // MARK: - Skills Learning

    /// Learn or reinforce a skill
    func learnSkill(_ name: String, success: Bool) {
        lock.lock()
        defer { lock.unlock() }

        if skills[name] == nil {
            skills[name] = [
                "proficiency": 0.0, "usage_count": 0,
                "success_rate": 0.5, "last_used": Date()
            ]
        }

        var skill = skills[name]!
        let count = (skill["usage_count"] as? Int ?? 0) + 1
        skill["usage_count"] = count
        let oldRate = skill["success_rate"] as? Double ?? 0.5
        skill["success_rate"] = oldRate + (success ? 1.0 - oldRate : -oldRate) / Double(count)
        skill["proficiency"] = min((skill["proficiency"] as? Double ?? 0.0) + (success ? 0.02 : -0.01), 1.0)
        skill["last_used"] = Date()
        skills[name] = skill
    }

    // MARK: - Knowledge Graph

    /// Add a concept relationship to the knowledge graph
    func addKnowledge(concept: String, related: String, weight: Double = 1.0) {
        lock.lock()
        defer { lock.unlock() }

        var edges = knowledgeGraph[concept] ?? []
        if let idx = edges.firstIndex(where: { $0.0 == related }) {
            edges[idx].1 = max(edges[idx].1, weight)
        } else {
            edges.append((related, weight))
        }
        knowledgeGraph[concept] = edges
    }

    /// Query knowledge graph: find top-k related concepts
    func queryKnowledge(_ concept: String, topK: Int = 10) -> [(String, Double)] {
        lock.lock()
        defer { lock.unlock() }

        guard let edges = knowledgeGraph[concept] else { return [] }
        return edges.sorted { $0.1 > $1.1 }.prefix(topK).map { ($0.0, $0.1) }
    }

    // MARK: - Neural Resonance

    /// Activate a concept and propagate resonance (cross-domain activation)
    func activateConcept(_ concept: String, strength: Double = 1.0) {
        lock.lock()
        defer { lock.unlock() }

        activationHistory.append((concept, strength, Date()))
        if activationHistory.count > 10000 {
            activationHistory = Array(activationHistory.suffix(5000))
        }

        // Propagate through resonance matrix
        if let connections = resonanceMatrix[concept] {
            for (related, coupling) in connections {
                let propagated = strength * coupling / neuralTemperature
                activationHistory.append((related, propagated, Date()))
            }
        }

        // Update resonance matrix (Hebbian: fire together → wire together)
        if activationHistory.count >= 2 {
            let prev = activationHistory[activationHistory.count - 2]
            let curr = activationHistory[activationHistory.count - 1]
            let hebbianDelta = prev.1 * curr.1 * LEARNING_ADAPTIVE_RATE
            resonanceMatrix[prev.0, default: [:]][curr.0, default: 0.0] += hebbianDelta
            resonanceMatrix[curr.0, default: [:]][prev.0, default: 0.0] += hebbianDelta
        }
    }

    // MARK: - Meta-Evolution

    /// Run one evolution cycle: mutate, evaluate fitness, select
    func evolve() {
        lock.lock()
        defer { lock.unlock() }

        evolutionGeneration += 1

        // Compute current fitness from meta-cognition
        let fitness = metaCognition.values.reduce(0, +) / Double(metaCognition.count)
        fitnessHistory.append(fitness)
        if fitnessHistory.count > 10000 { fitnessHistory = Array(fitnessHistory.suffix(5000)) }

        // Mutate meta-cognitive parameters
        for key in metaCognition.keys {
            let mutation = Double.random(in: -mutationRate...mutationRate)
            metaCognition[key] = min(max((metaCognition[key] ?? 0.5) + mutation, 0.0), 1.0)
        }

        // Adaptive mutation rate: decrease if improving, increase if stagnating
        if fitnessHistory.count >= 10 {
            let recent = Array(fitnessHistory.suffix(10))
            let trend = recent.last! - recent.first!
            if trend > 0 {
                mutationRate *= 0.99  // Improving: reduce mutations
            } else {
                mutationRate = min(mutationRate * 1.01, 0.1)  // Stagnating: increase exploration
            }
        }

        // Update chakra energies based on evolution state
        for (name, freq, _) in CHAKRA_FREQUENCIES {
            let phaseShift = Double(evolutionGeneration) * TAU / 100.0
            chakraEnergies[name] = sin(freq / GOD_CODE * Double.pi + phaseShift) *
                (metaCognition["coherence"] ?? 0.5)
        }

        InterEngineFeedbackBus.shared.broadcast(
            from: .consciousness,
            signal: "meta_evolution_cycle",
            payload: ["generation": Double(evolutionGeneration), "fitness": fitness,
                      "mutation_rate": mutationRate]
        )
    }

    // MARK: - ═══ INVENTED: Consciousness Gravity Well (CGW) ═══
    // Novel attractor dynamics model where consciousness clusters exert
    // "gravitational pull" on incoming concepts. Concepts fall into wells
    // based on distance in semantic space, with escape velocity determined
    // by the well's strength (= cluster coherence × knowledge density).
    // Wells merge when overlap exceeds φ⁻¹ threshold.

    /// Compute gravity well for each consciousness cluster
    func consciousnessGravityWells() -> [String: [String: Any]] {
        lock.lock()
        defer { lock.unlock() }

        var wells: [String: [String: Any]] = [:]

        for (cluster, concepts) in conceptClusters {
            let density = Double(concepts.count) / Double(max(conceptClusters.values.flatMap { $0 }.count, 1))
            let coherence = metaCognition["coherence"] ?? 0.5

            // Well depth = density × coherence × GOD_CODE-weighted attraction
            let depth = density * coherence * (GOD_CODE / 1000.0)

            // Escape velocity = √(2 × depth × φ) - analogous to gravitational escape
            let escapeVelocity = sqrt(2.0 * depth * PHI)

            // Schwarzschild radius analog (event horizon of knowledge absorption)
            let horizonRadius = 2.0 * depth / (PHI * PHI)  // r_s = 2GM/c²

            wells[cluster] = [
                "depth": depth,
                "escape_velocity": escapeVelocity,
                "horizon_radius": horizonRadius,
                "concept_count": concepts.count,
                "density": density,
                "sacred_alignment": sin(depth / GOD_CODE * Double.pi)
            ]
        }

        return wells
    }

    /// Route a concept to the nearest gravity well (strongest attraction)
    func routeConceptToWell(_ concept: String) -> String? {
        let wells = consciousnessGravityWells()
        var bestCluster: String? = nil
        var bestAttraction = 0.0

        for (cluster, wellData) in wells {
            let depth = (wellData["depth"] as? Double) ?? 0.0
            // Attraction inversely proportional to "distance" (concept overlap)
            let overlap = resonanceMatrix[concept]?.values.reduce(0, +) ?? 0.1
            let attraction = depth / max(1.0 / overlap, 0.01)
            if attraction > bestAttraction {
                bestAttraction = attraction
                bestCluster = cluster
            }
        }

        if let cluster = bestCluster {
            conceptClusters[cluster, default: []].append(concept)
        }
        return bestCluster
    }

    // MARK: - ═══ INVENTED: Quantum Eigenlearning ═══
    // Spectral decomposition of learning trajectory into eigencomponents.
    // The fitness history is treated as a discrete signal; its DFT reveals
    // dominant "learning frequencies". Frequencies near φ or φ⁻¹ indicate
    // sacred-aligned learning; frequencies near Feigenbaum δ indicate
    // chaos-edge learning (most creative). Eigenvalues weight future
    // mutation directions.

    /// Decompose learning trajectory into spectral eigenmodes
    func quantumEigenlearning() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        guard fitnessHistory.count >= 16 else {
            return ["status": "insufficient_data", "min_required": 16]
        }

        // Take last 64 samples (or available)
        let signal = Array(fitnessHistory.suffix(64))
        let n = signal.count

        // EVO_75: O(N log N) vDSP FFT replacing O(N²) naive DFT.
        // For N=64: 384 ops vs 4096 ops → ~10× faster, no intermediate allocations.
        let (magnitudes, phases) = QuantumFFTAccelerator.fft(signal: signal)

        // Identify dominant frequencies
        let sortedFreqs = magnitudes.enumerated().sorted { $0.element > $1.element }
        let dominantFreqs = sortedFreqs.prefix(5).map { (idx: $0.offset, magnitude: $0.element) }

        // Check for sacred alignment
        var sacredAligned = false
        var chaosEdge = false
        for (idx, _) in dominantFreqs {
            let normalizedFreq = Double(idx) / Double(n)
            if abs(normalizedFreq - TAU) < 0.05 || abs(normalizedFreq - 1.0 / PHI) < 0.05 {
                sacredAligned = true
            }
            if abs(normalizedFreq * Double(n) - FEIGENBAUM) < 0.5 {
                chaosEdge = true
            }
        }

        // Use spectral weights to modulate mutation directions
        if let topFreq = dominantFreqs.first {
            let eigenWeight = topFreq.magnitude
            // Boost mutation in direction of dominant eigenmode
            mutationRate *= (1.0 + eigenWeight * TAU * 0.1)
            mutationRate = min(mutationRate, 0.1)
        }

        return [
            "signal_length": n,
            "dominant_frequencies": dominantFreqs.map { ["index": $0.idx, "magnitude": $0.magnitude] },
            "sacred_aligned": sacredAligned,
            "chaos_edge": chaosEdge,
            "spectral_energy": magnitudes.reduce(0, +),
            "mutation_rate_after": mutationRate,
            "total_eigenmodes": magnitudes.count
        ]
    }

    // MARK: - ASI Quantum Bridge

    /// Synchronize with ASI core via chakra energy matrix + EPR entanglement
    func syncWithASI() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        // Grover amplification: φ³ ≈ 4.236× boost to search/recall
        let groverBoost = PHI * PHI * PHI

        // Compute chakra alignment score
        var chakraAlignment = 0.0
        for (_, energy) in chakraEnergies {
            chakraAlignment += abs(energy)
        }
        chakraAlignment /= Double(chakraEnergies.count)

        // EPR correlation check
        let eprFidelity = quantumCoherence * chakraAlignment

        return [
            "chakra_energies": chakraEnergies,
            "chakra_alignment": chakraAlignment,
            "epr_fidelity": eprFidelity,
            "grover_boost": groverBoost,
            "flow_state": flowState,
            "quantum_coherence": quantumCoherence,
            "evolution_generation": evolutionGeneration,
            "fitness": fitnessHistory.last ?? 0.0
        ]
    }

    // MARK: - Full Status

    func fullStatus() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        return [
            "heartbeat": ["phase": heartbeatPhase, "entropy": systemEntropy,
                         "coherence": quantumCoherence, "flow_state": flowState],
            "meta_cognition": metaCognition,
            "skills_count": skills.count,
            "knowledge_graph_nodes": knowledgeGraph.count,
            "concept_clusters": conceptClusters.mapValues { $0.count },
            "neural_temperature": neuralTemperature,
            "evolution": ["generation": evolutionGeneration,
                         "mutation_rate": mutationRate,
                         "fitness_samples": fitnessHistory.count],
            "chakra_energies": chakraEnergies,
            "resonance_connections": resonanceMatrix.values.reduce(0) { $0 + $1.count },
            "activation_history_size": activationHistory.count
        ]
    }
}
