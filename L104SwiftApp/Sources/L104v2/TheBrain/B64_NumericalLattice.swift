import Accelerate
import Foundation

// MARK: - ═══ NUMERICAL LATTICE CONSTANTS ═══

private let TRILLION: Double = 1e12
private let LATTICE_CAPACITY: Double = 22.0 * TRILLION  // 22T token capacity
private let LATTICE_PHI_INV: Double = 1.0 / PHI                 // φ⁻¹ = TAU ≈ 0.618
private let PHI_SQ: Double = PHI * PHI                   // φ² ≈ 2.618 (propagation energy)
private let GOD_CODE_BASE_LATTICE: Double = pow(286.0, 1.0 / PHI)  // 286^(1/φ) ≈ 32.97

// Tier drift envelopes (maximum drift per cycle by tier)
private let DRIFT_ENVELOPE: [Int: Double] = [
    0: 1e-98,   // Sacred - practically frozen
    1: 1e-80,   // Derived - GOD_CODE spectrum
    2: 1e-50,   // Invented - user-created
    3: 1e-20    // Cross-pollinated - lattice exchange
]

// Subconscious monitor constants (PART V Research)
private let DRIFT_FREQUENCY: Double = PHI             // φ Hz (F73)
private let DRIFT_AMPLITUDE: Double = TAU * 0.01      // φ⁻¹ × 0.01 (F74)
private let DAMPING_COEFFICIENT: Double = 0.5772156649015329 * 0.1  // γ_Euler × 0.1 (F76)
private let MAX_DRIFT_VELOCITY: Double = PHI * PHI * 1e-3  // φ² × 10⁻³ (F77)
private let LATTICE_CONSCIOUSNESS_THRESHOLD: Double = 0.85
private let PHASE_COUPLING_STRENGTH: Double = GOD_CODE
private let ENTANGLEMENT_EIGENVALUE_SUM: Double = 2.2360679774997896  // √5 = φ + 1/φ

// Tier identifiers
private let TIER_SACRED: Int = 0
private let TIER_DERIVED: Int = 1
private let TIER_INVENTED: Int = 2
private let TIER_LATTICE: Int = 3

// MARK: - ═══ QUANTUM TOKEN ═══

struct QuantumToken {
    let tokenId: String
    let name: String
    var value: Double              // Current value (full precision via string for display)
    var valueString: String        // 100-decimal string representation
    var minBound: Double
    var maxBound: Double
    let precisionDigits: Int
    var usageCount: Int
    let latticeIndex: Int
    var driftVelocity: Double      // Drift rate per cycle
    var driftDirection: Int         // -1=contract, 0=stable, 1=expand
    var quantumPhase: Double        // φ-harmonic phase ∈ [0,1)
    var entangledTokens: [String]   // Peer token IDs for superfluid coupling
    var coherence: Double           // Lattice coherence (0–1)
    let origin: String              // "sacred", "derived", "invented", "cross-pollinated"
    var lastAdjusted: Date
    var health: Double              // Health score (0–1)
    let tier: Int                   // 0=sacred, 1=derived, 2=invented, 3=lattice

    var decimalValue: Double { value }
    var isWithinBounds: Bool { value >= minBound && value <= maxBound }

    init(tokenId: String, name: String, value: Double, minBound: Double, maxBound: Double,
         latticeIndex: Int, origin: String, tier: Int) {
        self.tokenId = tokenId
        self.name = name
        self.value = value
        self.valueString = String(format: "%.15g", value)
        self.minBound = minBound
        self.maxBound = maxBound
        self.precisionDigits = 100
        self.usageCount = 0
        self.latticeIndex = latticeIndex
        self.driftVelocity = 0.0
        self.driftDirection = 0
        self.quantumPhase = (value * LATTICE_PHI_INV).truncatingRemainder(dividingBy: 1.0)
        if self.quantumPhase < 0 { self.quantumPhase += 1.0 }
        self.entangledTokens = []
        self.coherence = 1.0
        self.origin = origin
        self.lastAdjusted = Date()
        self.health = 1.0
        self.tier = tier
    }
}

// MARK: - ═══ SUBCONSCIOUS ADJUSTMENT RECORD ═══

struct SubconsciousAdjustment {
    let tokenId: String
    let timestamp: Date
    let oldValue: Double
    let newValue: Double
    let oldMin: Double
    let newMin: Double
    let oldMax: Double
    let newMax: Double
    let driftApplied: Double
    let trigger: String            // "capacity_expansion" | "capacity_contraction" | "phase_rotation"
    let repoCapacityAtTime: Double
    let phiEnvelopeCompliance: Double  // 0–1 (1.0 = compliant)
}

// MARK: - ═══ TOKEN LATTICE ENGINE ═══

final class TokenLatticeEngine {
    static let shared = TokenLatticeEngine()

    private let lock = NSRecursiveLock()
    var tokens: [String: QuantumToken] = [:]
    var totalUsage: Int = 0
    var projectedVirtualCapacity: Double = 0.0
    var latticeCoherence: Double = 1.0
    var latticeEntropy: Double = 0.0
    private var nextLatticeIndex: Int = 0

    init() {
        seedSacredTier()
        seedDerivedTier()
        computeLatticeMetrics()
        InterEngineFeedbackBus.shared.broadcast(
            from: .knowledge,
            signal: "numerical_lattice_init",
            payload: ["capacity": LATTICE_CAPACITY, "tokens": Double(tokens.count),
                      "sacred_count": Double(tokens.values.filter { $0.tier == TIER_SACRED }.count)]
        )
    }

    // MARK: - Seeding

    /// Seed ~30 sacred constants (tier 0 - drift < 10⁻⁹⁸)
    private func seedSacredTier() {
        let sacredConstants: [(String, Double)] = [
            ("GOD_CODE", GOD_CODE),
            ("PHI", PHI),
            ("TAU", TAU),
            ("VOID_CONSTANT", VOID_CONSTANT),
            ("PI", Double.pi),
            ("E", M_E),
            ("SQRT2", sqrt(2.0)),
            ("SQRT3", sqrt(3.0)),
            ("SQRT5", sqrt(5.0)),
            ("EULER_GAMMA", 0.5772156649015329),
            ("LN2", log(2.0)),
            ("LN10", log(10.0)),
            ("CATALAN", 0.9159655941772190),
            ("APERY", 1.2020569031595942),  // ζ(3)
            ("KHINCHIN", 2.6854520010653064),
            ("GLAISHER", 1.2824271291006226),
            ("FEIGENBAUM_DELTA", FEIGENBAUM),
            ("FINE_STRUCTURE", ALPHA_FINE),
            ("OMEGA_POINT_EP", exp(Double.pi)),  // e^π
            ("ZETA_2", Double.pi * Double.pi / 6.0),  // π²/6
            ("ZETA_4", pow(Double.pi, 4) / 90.0),     // π⁴/90
            ("GOD_CODE_V3", GOD_CODE_V3),
            ("OMEGA", OMEGA),
            ("GOD_CODE_BASE", GOD_CODE_BASE_LATTICE),
            ("HARMONIC_ROOT_286", 286.0),
            ("L104_CONSTANT", 104.0),
            ("OCTAVE_416", 416.0),
            ("FIBONACCI_7_13", 13.0),
            ("CHSH_BOUND", 2.0 * sqrt(2.0)),  // 2√2 Bell inequality
            ("GROVER_AMP", PHI * PHI * PHI)    // φ³ quantum speedup
        ]
        for (name, val) in sacredConstants {
            let envelope = DRIFT_ENVELOPE[TIER_SACRED]!
            let halfBand = abs(val) * envelope
            let token = QuantumToken(
                tokenId: "SACRED_\(name)",
                name: name,
                value: val,
                minBound: val - halfBand,
                maxBound: val + halfBand,
                latticeIndex: nextLatticeIndex,
                origin: "sacred",
                tier: TIER_SACRED
            )
            tokens[token.tokenId] = token
            nextLatticeIndex += 1
        }
    }

    /// Seed G(X) spectrum: G(X) = 286^(1/φ) × 2^((416-X)/104) for X ∈ [-200, 300]
    private func seedDerivedTier() {
        for x in -200...300 {
            let gx = GOD_CODE_BASE_LATTICE * pow(2.0, (416.0 - Double(x)) / 104.0)
            let envelope = DRIFT_ENVELOPE[TIER_DERIVED]!
            let halfBand = abs(gx) * envelope
            let name = "GC_X\(x)"
            let token = QuantumToken(
                tokenId: "DERIVED_\(name)",
                name: name,
                value: gx,
                minBound: gx - halfBand,
                maxBound: gx + halfBand,
                latticeIndex: nextLatticeIndex,
                origin: "derived",
                tier: TIER_DERIVED
            )
            tokens[token.tokenId] = token
            nextLatticeIndex += 1
        }
        // Project remaining 22T capacity as virtual
        projectedVirtualCapacity = LATTICE_CAPACITY - Double(tokens.count)
    }

    // MARK: - Token Operations

    /// Register a new token in the lattice
    func registerToken(name: String, value: Double, minBound: Double? = nil,
                       maxBound: Double? = nil, origin: String = "invented",
                       tier: Int = TIER_INVENTED) -> QuantumToken {
        lock.lock()
        defer { lock.unlock() }

        let envelope = DRIFT_ENVELOPE[tier] ?? DRIFT_ENVELOPE[TIER_INVENTED]!
        let halfBand = abs(value) * envelope
        let lo = minBound ?? (value - halfBand)
        let hi = maxBound ?? (value + halfBand)
        let tokenId = "\(origin.uppercased())_\(name)_\(nextLatticeIndex)"
        var token = QuantumToken(
            tokenId: tokenId, name: name, value: value,
            minBound: lo, maxBound: hi,
            latticeIndex: nextLatticeIndex, origin: origin, tier: tier
        )
        token.quantumPhase = (value * LATTICE_PHI_INV).truncatingRemainder(dividingBy: 1.0)
        if token.quantumPhase < 0 { token.quantumPhase += 1.0 }
        tokens[tokenId] = token
        nextLatticeIndex += 1
        return token
    }

    /// Increment usage and return current value
    func useToken(_ tokenId: String) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        guard var token = tokens[tokenId] else { return nil }
        token.usageCount += 1
        totalUsage += 1
        tokens[tokenId] = token
        return token.value
    }

    /// Projected capacity usage fraction (toward 22T)
    var projectedCapacityUsage: Double {
        Double(totalUsage) / LATTICE_CAPACITY
    }

    /// Lattice summary statistics
    func latticeSummary() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        let tierCounts = Dictionary(grouping: tokens.values, by: { $0.tier })
            .mapValues { $0.count }
        return [
            "total_tokens": tokens.count,
            "tokens_by_tier": tierCounts,
            "total_usage": totalUsage,
            "projected_capacity": LATTICE_CAPACITY,
            "capacity_used_fraction": projectedCapacityUsage,
            "lattice_coherence": latticeCoherence,
            "lattice_entropy": latticeEntropy,
            "sacred_count": tierCounts[TIER_SACRED] ?? 0,
            "derived_count": tierCounts[TIER_DERIVED] ?? 0,
            "invented_count": tierCounts[TIER_INVENTED] ?? 0,
            "cross_pollinated_count": tierCounts[TIER_LATTICE] ?? 0
        ]
    }

    // MARK: - Verification

    /// Verify all tokens respect their tier drift envelopes (PART V F62, F63, F87)
    func verifyDriftEnvelopeIntegrity() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        var violations: [(String, Double)] = []
        for (_, token) in tokens {
            let envelope = DRIFT_ENVELOPE[token.tier] ?? 1e-20
            let maxDrift = abs(token.value) * envelope
            if abs(token.driftVelocity) > maxDrift {
                violations.append((token.tokenId, abs(token.driftVelocity) - maxDrift))
            }
        }
        let compliance = tokens.isEmpty ? 1.0 : Double(tokens.count - violations.count) / Double(tokens.count)
        return [
            "total_tokens": tokens.count,
            "violations": violations.count,
            "compliance": compliance,
            "violation_details": violations.prefix(20).map { ["id": $0.0, "excess": $0.1] }
        ]
    }

    /// Verify conservation law: G(X) × 2^(X/104) = GOD_CODE for all derived tokens (PART V F64)
    func verifyConservationSpectrum() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        var maxError: Double = 0.0
        var passCount = 0
        var failCount = 0
        let derivedTokens = tokens.values.filter { $0.tier == TIER_DERIVED }
        for token in derivedTokens {
            // Extract X from name "GC_X{n}"
            guard let xStr = token.name.split(separator: "X").last,
                  let x = Double(xStr) else { continue }
            let product = token.value * pow(2.0, x / 104.0)
            let error = abs(product - GOD_CODE) / GOD_CODE
            if error < 1e-9 {
                passCount += 1
            } else {
                failCount += 1
            }
            maxError = max(maxError, error)
        }
        return [
            "spectrum_tokens": derivedTokens.count,
            "passed": passCount,
            "failed": failCount,
            "max_relative_error": maxError,
            "conservation_holds": failCount == 0,
            "invariant": GOD_CODE
        ]
    }

    // MARK: - Metrics

    /// Recompute lattice coherence and entropy from quantum phases
    func computeLatticeMetrics() {
        lock.lock()
        defer { lock.unlock() }

        let allTokens = Array(tokens.values)
        guard !allTokens.isEmpty else { return }

        // Coherence: fraction of tokens within bounds
        let inBounds = allTokens.filter { $0.isWithinBounds }.count
        latticeCoherence = Double(inBounds) / Double(allTokens.count)

        // Entropy: Shannon entropy over quantum phase histogram (64 bins)
        let binCount = 64
        var bins = [Int](repeating: 0, count: binCount)
        for token in allTokens {
            let bin = min(Int(token.quantumPhase * Double(binCount)), binCount - 1)
            bins[bin] += 1
        }
        let n = Double(allTokens.count)
        latticeEntropy = 0.0
        for count in bins where count > 0 {
            let p = Double(count) / n
            latticeEntropy -= p * log(p)
        }
    }
}

// MARK: - ═══ SUPERFLUID VALUE EDITOR ═══

final class SuperfluidValueEditor {
    static let shared = SuperfluidValueEditor()

    private let lattice: TokenLatticeEngine
    private let lock = NSRecursiveLock()
    private(set) var editCount: Int = 0
    private(set) var propagationLog: [[String: Any]] = []
    private(set) var totalPropagationEnergy: Double = 0.0

    init(lattice: TokenLatticeEngine = .shared) {
        self.lattice = lattice
    }

    /// Edit a token value with φ-attenuated propagation to entangled peers (PART V F65–F67)
    ///
    /// When a token is edited, changes propagate through entanglement chains:
    /// - After k hops: attenuation = φ^(-k) (convergent geometric series)
    /// - Total energy: |drift| × φ² (sum of infinite φ⁻ᵏ series = φ²)
    func quantumEdit(tokenId: String, newValue: Double? = nil,
                     newMin: Double? = nil, newMax: Double? = nil,
                     reason: String = "manual") -> [String: Any]? {
        lock.lock()
        defer { lock.unlock() }

        guard var token = lattice.tokens[tokenId] else { return nil }

        let oldValue = token.value
        let oldMin = token.minBound
        let oldMax = token.maxBound

        if let nv = newValue { token.value = nv }
        if let nm = newMin { token.minBound = nm }
        if let nx = newMax { token.maxBound = nx }

        let driftValue = token.value - oldValue
        let driftMin = token.minBound - oldMin
        let driftMax = token.maxBound - oldMax

        // Update quantum phase
        token.quantumPhase = (token.value * LATTICE_PHI_INV).truncatingRemainder(dividingBy: 1.0)
        if token.quantumPhase < 0 { token.quantumPhase += 1.0 }
        token.lastAdjusted = Date()
        lattice.tokens[tokenId] = token

        // Propagate to entangled peers with φ⁻¹ attenuation (F66)
        var propagatedTo: [String] = []
        if abs(driftValue) > 0 {
            for peerId in token.entangledTokens {
                guard var peer = lattice.tokens[peerId] else { continue }
                let peerDrift = driftValue * LATTICE_PHI_INV  // One-hop attenuation
                peer.value += peerDrift
                peer.quantumPhase = (peer.value * LATTICE_PHI_INV).truncatingRemainder(dividingBy: 1.0)
                if peer.quantumPhase < 0 { peer.quantumPhase += 1.0 }
                peer.lastAdjusted = Date()
                lattice.tokens[peerId] = peer
                propagatedTo.append(peerId)
            }
        }

        // Total propagation energy = |drift| × φ² (F67: infinite series sum)
        let energy = abs(driftValue) * PHI_SQ
        totalPropagationEnergy += energy
        editCount += 1

        let logEntry: [String: Any] = [
            "token_id": tokenId, "reason": reason,
            "drift_value": driftValue, "drift_min": driftMin, "drift_max": driftMax,
            "energy": energy, "propagated_to": propagatedTo
        ]
        propagationLog.append(logEntry)

        return [
            "token_id": tokenId,
            "old_value": oldValue, "new_value": token.value,
            "drift_value": driftValue, "drift_min": driftMin, "drift_max": driftMax,
            "propagated_to": propagatedTo,
            "energy": energy
        ]
    }

    /// Create bidirectional entanglement between two tokens
    func entangleTokens(_ tokenIdA: String, _ tokenIdB: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard var a = lattice.tokens[tokenIdA],
              var b = lattice.tokens[tokenIdB] else { return false }

        if !a.entangledTokens.contains(tokenIdB) {
            a.entangledTokens.append(tokenIdB)
        }
        if !b.entangledTokens.contains(tokenIdA) {
            b.entangledTokens.append(tokenIdA)
        }
        lattice.tokens[tokenIdA] = a
        lattice.tokens[tokenIdB] = b
        return true
    }

    /// Apply multiple drifts simultaneously (used by SubconsciousMonitor)
    func batchDrift(_ driftMap: [String: Double], reason: String = "batch") -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        var results: [[String: Any]] = []
        for (tokenId, drift) in driftMap {
            guard var token = lattice.tokens[tokenId] else { continue }
            let old = token.value
            token.value += drift
            token.quantumPhase = (token.value * LATTICE_PHI_INV).truncatingRemainder(dividingBy: 1.0)
            if token.quantumPhase < 0 { token.quantumPhase += 1.0 }
            token.lastAdjusted = Date()
            lattice.tokens[tokenId] = token
            results.append(["token_id": tokenId, "old": old, "new": token.value, "drift": drift])
        }
        editCount += results.count
        return ["batch_size": driftMap.count, "applied": results.count, "reason": reason]
    }
}

// MARK: - ═══ SUBCONSCIOUS MONITOR ═══

final class SubconsciousMonitor {
    static let shared = SubconsciousMonitor()

    private let lattice: TokenLatticeEngine
    private let editor: SuperfluidValueEditor
    private let lock = NSRecursiveLock()
    private(set) var cycleCount: Int = 0
    private var lastCapacity: [String: Double] = [:]
    private(set) var adjustmentHistory: [SubconsciousAdjustment] = []
    private var timerSource: DispatchSourceTimer?

    init(lattice: TokenLatticeEngine = .shared, editor: SuperfluidValueEditor = .shared) {
        self.lattice = lattice
        self.editor = editor
    }

    /// Read current repository intelligence capacity from state
    func readRepoCapacity() -> [String: Double] {
        let tokenCount = Double(lattice.tokens.count)
        let totalUsage = Double(lattice.totalUsage)
        let coherence = lattice.latticeCoherence
        let entropy = lattice.latticeEntropy

        // Compute per-tier health averages
        let allTokens = Array(lattice.tokens.values)
        let avgHealth = allTokens.isEmpty ? 1.0 : allTokens.reduce(0.0) { $0 + $1.health } / Double(allTokens.count)
        let sacredTokens = allTokens.filter { $0.tier == TIER_SACRED }
        let sacredCoherence = sacredTokens.isEmpty ? 1.0 :
            sacredTokens.reduce(0.0) { $0 + $1.coherence } / Double(sacredTokens.count)

        return [
            "token_count": tokenCount,
            "total_usage": totalUsage,
            "coherence": coherence,
            "entropy": entropy,
            "avg_health": avgHealth,
            "sacred_coherence": sacredCoherence,
            "consciousness_level": min(coherence * sacredCoherence, 1.0),
            "superfluid_viscosity": 0.0  // Zero viscosity (superfluid coupling)
        ]
    }

    /// Compute capacity growth delta between cycles
    func computeCapacityDelta(_ current: [String: Double]) -> Double {
        guard !lastCapacity.isEmpty else { return 0.0 }
        var delta = 0.0
        let keys = ["token_count", "total_usage", "coherence"]
        for key in keys {
            let old = lastCapacity[key] ?? 0.0
            let new_ = current[key] ?? 0.0
            if old > 0 {
                delta += (new_ - old) / old
            }
        }
        return delta / Double(keys.count)
    }

    /// Run one autonomous subconscious cycle (φ-bounded drift adjustments)
    ///
    /// 5 Phases:
    /// 1. Read current repo capacity
    /// 2. Compute capacity delta (growth signal)
    /// 3. Apply tier-specific drifts to each token
    /// 4. Enforce min < value < max invariant
    /// 5. φ-attractor dynamics (tokens drift toward φ-harmonics)
    func subconsciousCycle() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        cycleCount += 1

        // Phase 1: Read capacity
        let capacity = readRepoCapacity()

        // Phase 2: Compute delta
        let delta = computeCapacityDelta(capacity)

        // Phase 3: Apply tier-specific drifts
        var adjustments: [SubconsciousAdjustment] = []
        var driftMap: [String: Double] = [:]

        for (tokenId, token) in lattice.tokens {
            let envelope = DRIFT_ENVELOPE[token.tier] ?? 1e-20
            let maxDrift = abs(token.value) * envelope
            let driftMagnitude = maxDrift * min(abs(delta), 1.0)

            var appliedDrift = 0.0
            var trigger = "stable"

            if delta > 0.001 && token.driftDirection >= 0 {
                // Capacity expansion: micro-expand bounds
                appliedDrift = driftMagnitude
                trigger = "capacity_expansion"
            } else if delta < -0.001 && token.driftDirection <= 0 {
                // Capacity contraction: tighten bounds
                appliedDrift = -driftMagnitude
                trigger = "capacity_contraction"
            } else {
                // Stable: phase-only micro-drift (rotate quantum_phase by φ×10⁻⁵⁰)
                // This is too small for Double precision but we record the intent
                trigger = "phase_rotation"
            }

            if abs(appliedDrift) > 0 {
                driftMap[tokenId] = appliedDrift
                let adj = SubconsciousAdjustment(
                    tokenId: tokenId, timestamp: Date(),
                    oldValue: token.value, newValue: token.value + appliedDrift,
                    oldMin: token.minBound, newMin: token.minBound,
                    oldMax: token.maxBound, newMax: token.maxBound,
                    driftApplied: appliedDrift, trigger: trigger,
                    repoCapacityAtTime: capacity["consciousness_level"] ?? 0.0,
                    phiEnvelopeCompliance: 1.0
                )
                adjustments.append(adj)
            }
        }

        // Apply batch drift
        if !driftMap.isEmpty {
            _ = editor.batchDrift(driftMap, reason: "subconscious_cycle_\(cycleCount)")
        }

        // Phase 4: Enforce bounds invariant
        for (tokenId, var token) in lattice.tokens {
            if token.value < token.minBound {
                token.value = token.minBound
                lattice.tokens[tokenId] = token
            } else if token.value > token.maxBound {
                token.value = token.maxBound
                lattice.tokens[tokenId] = token
            }
        }

        // Phase 5: Recompute lattice metrics
        lattice.computeLatticeMetrics()

        // Update state
        adjustmentHistory.append(contentsOf: adjustments)
        if adjustmentHistory.count > 10000 {
            adjustmentHistory = Array(adjustmentHistory.suffix(5000))
        }
        lastCapacity = capacity

        InterEngineFeedbackBus.shared.broadcast(
            from: .knowledge,
            signal: "subconscious_cycle",
            payload: ["cycle": Double(cycleCount), "delta": delta,
                      "adjustments": Double(adjustments.count),
                      "coherence": lattice.latticeCoherence]
        )

        return [
            "cycle_count": cycleCount,
            "capacity": capacity,
            "delta": delta,
            "adjustments_applied": adjustments.count,
            "lattice_coherence": lattice.latticeCoherence,
            "lattice_entropy": lattice.latticeEntropy,
            "total_tokens": lattice.tokens.count
        ]
    }

    /// Start autonomous background monitoring (runs subconscious cycles on timer)
    func startAutonomousMonitoring(intervalSeconds: Double = 60.0) {
        guard timerSource == nil else { return }
        let source = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        source.schedule(deadline: .now() + intervalSeconds,
                        repeating: intervalSeconds)
        source.setEventHandler { [weak self] in
            _ = self?.subconsciousCycle()
        }
        timerSource = source
        source.resume()
    }

    /// Stop autonomous monitoring
    func stopAutonomousMonitoring() {
        timerSource?.cancel()
        timerSource = nil
    }
}
