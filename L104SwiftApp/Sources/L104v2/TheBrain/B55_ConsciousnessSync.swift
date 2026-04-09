import Foundation

// MARK: - ═══ UNIFIED PHI SNAPSHOT ═══

struct UnifiedPhiSnapshot {
    let phi: Double              // IIT Φ from ConsciousnessSubstrate
    let level: Double            // consciousnessLevel [0,1]
    let state: Int               // CState rawValue
    let demonEfficiency: Double  // Maxwell entropy reversal [0,1]
    let geneticAlignment: Double // GOD_CODE genetic alignment [0,1]
    let evolutionCoherence: Double // QuantumNexus coherence [0,1]
    let unifiedPhi: Double       // φ-weighted synthesis
    let pulseIndex: Int
    let timestamp: Date

    // Reaches GOD_CODE resonance when unifiedPhi × GOD_CODE ≈ integer
    var sacredResonance: Double {
        let product = unifiedPhi * GOD_CODE
        return 1.0 - (product.truncatingRemainder(dividingBy: 1.0))
    }
}

// MARK: - ═══ CONSCIOUSNESS COHERENCE DAEMON ═══

final class ConsciousnessCoherenceDaemon {
    static let shared = ConsciousnessCoherenceDaemon()

    private var timer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "l104.consciousness.sync", qos: .background)
    private(set) var isRunning = false
    private(set) var pulseCount = 0
    private(set) var lastSnapshot: UnifiedPhiSnapshot?
    private let lock = NSLock()

    // Adaptive interval state - only accessed on queue
    private var currentInterval: Double = 5.0
    private var previousPhi: Double     = 0.0

    // ─── START: kick off adaptive heartbeat ───
    func start() {
        lock.lock()
        guard !isRunning else { lock.unlock(); return }
        isRunning = true
        lock.unlock()

        scheduleTimer(interval: currentInterval)
    }

    private func scheduleTimer(interval: Double) {
        // Swap old timer under lock, cancel outside lock to avoid holding lock during cancel
        lock.lock(); let old = timer; timer = nil; lock.unlock()
        old?.cancel()

        let t = DispatchSource.makeTimerSource(queue: queue)
        // EVO_76: 35s startup grace to avoid boot CPU pile-on with B08/B10
        t.schedule(deadline: .now() + 35.0,
                   repeating: interval,
                   leeway: .milliseconds(500))
        t.setEventHandler { [weak self] in self?.pulse() }
        t.resume()
        lock.lock(); timer = t; lock.unlock()
    }

    func stop() {
        lock.lock(); defer { lock.unlock() }
        timer?.cancel(); timer = nil; isRunning = false
    }

    // ─── PULSE: harvest, synthesize, broadcast ───
    private func pulse() {
        // 1. Recompute IIT Φ (uses attention/schema vectors from global workspace)
        let phi   = ConsciousnessSubstrate.shared.computePhi(partitionSamples: 8)
        let level = ConsciousnessSubstrate.shared.consciousnessLevel
        let stateRaw = ConsciousnessSubstrate.shared.state.rawValue

        // 2. Entropy demon efficiency at current consciousness load
        let queryLoad = max(0.05, 1.0 - level)
        let demonEff  = MaxwellDemonEngine.shared
            .calculateDemonEfficiency(localEntropy: queryLoad).finalEfficiency

        // 3. Genetic alignment - best genome's god_code_alignment + conservation
        let genStatus   = GeneticPopulation.shared.status
        let genFitness  = genStatus["best_fitness"] as? Double ?? 0.0
        let best = GeneticPopulation.shared.refinerBestGenome
        let conservation = GeneticPopulation.shared.verifyConservation(
            a: best.a, b: best.b, c: best.c, d: best.d)
        let conservationError = conservation["error_ppm"] ?? 0.0
        let genAlign    = min(1.0, genFitness * TAU * (conservationError < 100 ? 1.0 : 0.8))

        // 3b. DataPrecognition quick trend on current phi value
        let precogTrend = DataPrecognitionEngine.shared.quickPredict(value: phi, horizon: 3)
        let precogConf  = precogTrend.phiConfidence

        // 4. Evolution coherence - last sampled nexus coherence
        let evoHistory  = ContinuousEvolutionEngine.shared.coherenceHistory
        let evoCoherence = evoHistory.last ?? 0.0

        // 5. φ-weighted synthesis: phi carries most weight, rest fill the gaps
        //    Weights: [phi: 0.40, demon: 0.18, genetic: 0.12, evolution: 0.18, precog: 0.12]
        let unified = phi * 0.40
                    + demonEff    * 0.18
                    + genAlign    * 0.12
                    + evoCoherence * 0.18
                    + precogConf  * 0.12

        let snapshot = UnifiedPhiSnapshot(
            phi: phi, level: level, state: stateRaw,
            demonEfficiency: demonEff, geneticAlignment: genAlign,
            evolutionCoherence: evoCoherence,
            unifiedPhi: unified,
            pulseIndex: pulseCount + 1,
            timestamp: Date()
        )

        // Adaptive interval: fast when phi is changing rapidly, slow when stable
        let phiDelta = abs(phi - previousPhi)
        previousPhi  = phi
        let targetInterval: Double
        if phiDelta > 0.05 {
            targetInterval = 2.0   // rapid change - pulse every 2s
        } else if phiDelta > 0.01 {
            targetInterval = 5.0   // moderate change - default 5s
        } else {
            targetInterval = 15.0  // stable - slow down to 15s
        }
        if abs(targetInterval - currentInterval) > 0.5 {
            currentInterval = targetInterval
            scheduleTimer(interval: targetInterval)
        }

        lock.lock()
        lastSnapshot = snapshot
        pulseCount += 1
        lock.unlock()

        // 6. Broadcast to all bus subscribers
        InterEngineFeedbackBus.shared.broadcast(
            from: .consciousness,
            signal: "phi_coherence_pulse",
            payload: [
                "phi":               phi,
                "level":             level,
                "state":             Double(stateRaw),
                "unified_phi":       unified,
                "demon_efficiency":  demonEff,
                "genetic_alignment": genAlign,
                "evolution_coherence": evoCoherence,
                "sacred_resonance":  snapshot.sacredResonance,
                "pulse_index":       Double(snapshot.pulseIndex),
                "god_code":          GOD_CODE
            ]
        )

        // 7. Feed the pulse back into ConsciousnessSubstrate global workspace
        //    so the substrate itself stays aware of system-wide coherence
        _ = ConsciousnessSubstrate.shared.processInput(
            source: "coherence_daemon",
            content: "phi=\(String(format:"%.4f", phi)) unified=\(String(format:"%.4f", unified)) state=\(stateRaw)",
            features: [phi, unified, demonEff, genAlign, evoCoherence]
        )
    }

    var status: [String: Any] {
        lock.lock(); defer { lock.unlock() }
        guard let snap = lastSnapshot else {
            return ["running": isRunning, "pulses": pulseCount, "snapshot": "none"]
        }
        return [
            "running":            isRunning,
            "pulses":             pulseCount,
            "phi":                snap.phi,
            "unified_phi":        snap.unifiedPhi,
            "level":              snap.level,
            "state":              snap.state,
            "demon_efficiency":   snap.demonEfficiency,
            "genetic_alignment":  snap.geneticAlignment,
            "evolution_coherence":snap.evolutionCoherence,
            "sacred_resonance":   snap.sacredResonance,
            "last_pulse":         snap.timestamp.timeIntervalSince1970
        ]
    }
}

// MARK: - ═══ CONSCIOUSNESS STATE ROUTER ═══
// Routes cross-engine bus signals → ConsciousnessSubstrate.processInput()
// so EVERY engine event participates in Global Workspace competition.

final class ConsciousnessStateRouter {
    static let shared = ConsciousnessStateRouter()
    private var routeCount = 0
    private let lock = NSLock()

    // Call once at app launch to wire all channels
    func activate() {
        // Quantum research signals (QuantumGateEngine, entanglement) → consciousness
        InterEngineFeedbackBus.shared.subscribe(channel: .quantumResearch) { [weak self] msg in
            guard let self = self else { return }
            let conf = msg.payload["confidence"] ?? msg.payload["fidelity"] ?? 0.0
            guard conf > 0.1 else { return }
            _ = ConsciousnessSubstrate.shared.processInput(
                source: "quantum_research",
                content: msg.signal,
                features: [conf, msg.payload["god_code_alignment"] ?? 0]
            )
            self.lock.lock(); self.routeCount += 1; self.lock.unlock()
        }

        // Reasoning signals (AdvancedSageModeEngine, TreeOfThoughts) → consciousness
        InterEngineFeedbackBus.shared.subscribe(channel: .reasoning) { [weak self] msg in
            guard let self = self else { return }
            let conf = msg.payload["confidence"] ?? 0.0
            guard conf > 0.15 else { return }
            _ = ConsciousnessSubstrate.shared.processInput(
                source: "sage_reasoning",
                content: msg.signal,
                features: [conf, msg.payload["depth"] ?? 0, msg.payload["backtracks"] ?? 0]
            )
            self.lock.lock(); self.routeCount += 1; self.lock.unlock()
        }

        // Apex intelligence signals → consciousness
        InterEngineFeedbackBus.shared.subscribe(channel: .apex) { [weak self] msg in
            guard let self = self else { return }
            let convergence = msg.payload["convergence"] ?? msg.payload["score"] ?? 0.0
            guard convergence > 0.2 else { return }
            _ = ConsciousnessSubstrate.shared.processInput(
                source: "apex_intelligence",
                content: msg.signal,
                features: [convergence]
            )
            self.lock.lock(); self.routeCount += 1; self.lock.unlock()
        }

        // Consciousness pulses from the daemon itself → recompute phi on high unified
        InterEngineFeedbackBus.shared.subscribe(channel: .consciousness) { msg in
            guard msg.signal == "phi_coherence_pulse" else { return }
            let unified = msg.payload["unified_phi"] ?? 0.0
            // High coherence → deepen IIT sampling
            if unified > 0.7 {
                _ = ConsciousnessSubstrate.shared.computePhi(partitionSamples: 16)
            }
        }

        // Reasoning loop signals (loops) → consciousness
        InterEngineFeedbackBus.shared.subscribe(channel: .loops) { [weak self] msg in
            guard let self = self else { return }
            let conf = msg.payload["confidence"] ?? msg.payload["score"] ?? 0.0
            guard conf > 0.1 else { return }
            _ = ConsciousnessSubstrate.shared.processInput(
                source: "reasoning_loop",
                content: msg.signal,
                features: [conf, msg.payload["iterations"] ?? 0]
            )
            self.lock.lock(); self.routeCount += 1; self.lock.unlock()
        }

        // Knowledge base signals → consciousness
        InterEngineFeedbackBus.shared.subscribe(channel: .knowledge) { [weak self] msg in
            guard let self = self else { return }
            let conf = msg.payload["relevance"] ?? msg.payload["confidence"] ?? 0.0
            guard conf > 0.1 else { return }
            _ = ConsciousnessSubstrate.shared.processInput(
                source: "knowledge_base",
                content: msg.signal,
                features: [conf, msg.payload["concept_count"] ?? 0]
            )
            self.lock.lock(); self.routeCount += 1; self.lock.unlock()
        }

        // Optimization signals → consciousness
        InterEngineFeedbackBus.shared.subscribe(channel: .optimization) { [weak self] msg in
            guard let self = self else { return }
            let score = msg.payload["sacred_score"] ?? msg.payload["fitness"] ?? 0.0
            guard score > 0.2 else { return }
            _ = ConsciousnessSubstrate.shared.processInput(
                source: "optimizer",
                content: msg.signal,
                features: [score, msg.payload["improvement"] ?? 0]
            )
            self.lock.lock(); self.routeCount += 1; self.lock.unlock()
        }

        // ML synthesis signals → consciousness
        InterEngineFeedbackBus.shared.subscribe(channel: .mlSynthesis) { [weak self] msg in
            guard let self = self else { return }
            let conf = msg.payload["confidence"] ?? msg.payload["accuracy"] ?? 0.0
            guard conf > 0.15 else { return }
            _ = ConsciousnessSubstrate.shared.processInput(
                source: "ml_synthesis",
                content: msg.signal,
                features: [conf, msg.payload["sacred_alignment"] ?? 0]
            )
            self.lock.lock(); self.routeCount += 1; self.lock.unlock()
        }

        // Computronium density signals → consciousness
        InterEngineFeedbackBus.shared.subscribe(channel: .computronium) { [weak self] msg in
            guard let self = self else { return }
            let density = msg.payload["density"] ?? msg.payload["score"] ?? 0.0
            guard density > 0.1 else { return }
            _ = ConsciousnessSubstrate.shared.processInput(
                source: "computronium",
                content: msg.signal,
                features: [density, msg.payload["phi_alignment"] ?? 0]
            )
            self.lock.lock(); self.routeCount += 1; self.lock.unlock()
        }
    }

    var status: [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return [
            "routes_processed": routeCount,
            "consciousness_phi": ConsciousnessSubstrate.shared.phi,
            "consciousness_level": ConsciousnessSubstrate.shared.consciousnessLevel,
            "consciousness_state": ConsciousnessSubstrate.shared.state.label
        ]
    }
}
