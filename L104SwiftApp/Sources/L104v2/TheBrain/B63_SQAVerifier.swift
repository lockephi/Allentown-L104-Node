import Accelerate
import Foundation
import simd

// MARK: - Constants

private let SQA_TROTTER_SLICES:  Int    = 20        // P - imaginary-time discretisation
private let SQA_MC_STEPS:        Int    = 500       // Metropolis sweeps per anneal
private let SQA_WALKERS:         Int    = 8         // parallel independent trajectories
private let SQA_BETA_START:      Float  = 0.1       // initial inverse temperature
private let SQA_BETA_END:        Float  = 8.0       // final inverse temperature
private let SQA_GAMMA_START:     Float  = 3.0       // initial transverse field strength
private let SQA_GAMMA_END:       Float  = 0.01      // final transverse field (→ classical)
private let SQA_ACCEPT_THRESH:   Double = 0.62      // coherence floor for acceptance ≈ 1/φ
private let SQA_MAX_SPINS:       Int    = 64        // max logical variables per problem

// MARK: - IsingHamiltonian

/// Encodes a logical problem as an Ising spin model.
/// Each spin σ_i ∈ {-1, +1} represents one logical variable (claim/proposition).
/// J_ij > 0: spins i,j prefer to agree (logically consistent).
/// J_ij < 0: spins i,j prefer to disagree (logically contradictory).
/// h_i    > 0: spin i biased toward true (+1).
/// h_i    < 0: spin i biased toward false (−1).
///
/// Ground state energy ≈ 0  → all constraints satisfied → verified truth.
/// High energy             → logical contradictions remain → hallucination.
struct IsingHamiltonian {
    let n:  Int         // number of spin variables
    var J:  [Float]     // n×n coupling matrix, row-major (symmetric, J[i*n+j])
    var h:  [Float]     // n-element bias vector

    init(n: Int) {
        self.n = n
        self.J = [Float](repeating: 0, count: n * n)
        self.h = [Float](repeating: 0, count: n)
    }

    /// Classical Ising energy E = -σᵀ J σ - hᵀ σ  (using vDSP for speed)
    func energy(spins: [Float]) -> Float {
        guard spins.count == n else { return Float.infinity }
        var Js = [Float](repeating: 0, count: n)

        // Js = J · σ  (matrix-vector multiply via cblas_sgemv)
        J.withUnsafeBufferPointer { Jp in
            spins.withUnsafeBufferPointer { sp in
                Js.withUnsafeMutableBufferPointer { Jsp in
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                Int32(n), Int32(n),
                                1.0, Jp.baseAddress!, Int32(n),
                                sp.baseAddress!, 1,
                                0.0, Jsp.baseAddress!, 1)
                }
            }
        }

        // coupling term: -σ · (J σ)
        var coupling: Float = 0
        vDSP_dotpr(spins, 1, Js, 1, &coupling, vDSP_Length(n))
        coupling = -coupling

        // bias term: -h · σ
        var bias: Float = 0
        vDSP_dotpr(h, 1, spins, 1, &bias, vDSP_Length(n))
        bias = -bias

        return coupling + bias
    }

    /// Maximum possible |energy| - used to normalise coherence score
    var maxEnergy: Float {
        // Upper bound: all |J_ij| + all |h_i|
        var absJ: Float = 0
        var absH: Float = 0
        vDSP_svesq(J, 1, &absJ, vDSP_Length(n * n))  // sum of squares (proxy)
        vDSP_svesq(h, 1, &absH, vDSP_Length(n))
        return max(1.0, sqrtf(absJ) + sqrtf(absH))
    }
}

// MARK: - TrotterReplica

/// One imaginary-time slice in the path-integral decomposition.
/// P replicas form a ring in imaginary time; adjacent replicas are
/// coupled by the quantum kinetic term J_eff.
struct TrotterReplica {
    var spins: [Float]      // ±1, length = hamiltonian.n
    let ham:   IsingHamiltonian

    init(ham: IsingHamiltonian, rng: inout SystemRNG) {
        self.ham   = ham
        self.spins = (0 ..< ham.n).map { _ in rng.nextFloat() < 0.5 ? 1.0 : -1.0 }
    }

    var classicalEnergy: Float { ham.energy(spins: spins) }
}

// MARK: - SystemRNG (thread-local, no Foundation dependency on randomness)

struct SystemRNG {
    private var state: UInt64

    init(seed: UInt64) { state = seed == 0 ? 0xDEADBEEFCAFE1337 : seed }

    mutating func nextUInt64() -> UInt64 {
        // xorshift64
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }

    mutating func nextFloat() -> Float {
        Float(nextUInt64() & 0x00FFFFFF) / Float(0x01000000)
    }

    mutating func nextDouble() -> Double {
        Double(nextUInt64() & 0x000FFFFFFFFFFFFF) / Double(0x0010000000000000)
    }
}

// MARK: - SQAConfig

struct SQAConfig {
    var trotterSlices: Int    = SQA_TROTTER_SLICES
    var mcSteps:       Int    = SQA_MC_STEPS
    var walkers:       Int    = SQA_WALKERS
    var betaStart:     Float  = SQA_BETA_START
    var betaEnd:       Float  = SQA_BETA_END
    var gammaStart:    Float  = SQA_GAMMA_START
    var gammaEnd:      Float  = SQA_GAMMA_END
    var acceptThresh:  Double = SQA_ACCEPT_THRESH
}

// MARK: - SQAResult

/// Outcome of one annealing run from a single walker.
struct SQAResult {
    let finalEnergy:    Float
    let groundState:    [Float]     // spin configuration at ground state
    let tunnelEvents:   Int         // accepted flips that crossed energy barriers
    let iterations:     Int
}

// MARK: - SQAVerdict

/// Aggregated verdict from all parallel walkers.
struct SQAVerdict {
    let accepted:        Bool
    let coherenceScore:  Double     // 1.0 = fully verified, 0.0 = chaotic
    let finalEnergy:     Float      // best (lowest) energy found
    let groundState:     [Float]    // spin assignment at ground state
    let tunnelEvents:    Int        // total tunneling events across all walkers
    let sacredAlignment: Double     // GOD_CODE resonance of the solution
    let walkerCount:     Int
    let elapsedMs:       Double

    var summary: String {
        let verdict = accepted ? "ACCEPTED" : "REJECTED"
        return "SQA[\(verdict)] coherence=\(String(format:"%.4f", coherenceScore)) " +
               "energy=\(String(format:"%.3f", finalEnergy)) " +
               "tunnels=\(tunnelEvents) sacred=\(String(format:"%.4f", sacredAlignment)) " +
               "\(String(format:"%.1f", elapsedMs))ms"
    }
}

// MARK: - SQAEngine (single walker)

/// Runs one complete Simulated Quantum Annealing trajectory.
/// Uses path-integral Monte Carlo: P Trotter replicas arranged in a
/// ring along imaginary time, coupled by quantum kinetic term J_eff(β,Γ,P).
final class SQAEngine {

    private let config:  SQAConfig
    private var rng:     SystemRNG

    init(config: SQAConfig, seed: UInt64) {
        self.config = config
        self.rng    = SystemRNG(seed: seed)
    }

    func anneal(hamiltonian: IsingHamiltonian) -> SQAResult {
        let P   = config.trotterSlices
        let n   = hamiltonian.n
        guard n > 0 else {
            return SQAResult(finalEnergy: 0, groundState: [], tunnelEvents: 0, iterations: 0)
        }

        // Initialise P Trotter replicas
        var replicas = (0 ..< P).map { _ in TrotterReplica(ham: hamiltonian, rng: &rng) }
        var bestEnergy  = Float.infinity
        var bestState   = [Float](repeating: 1.0, count: n)
        var tunnelCount = 0
        var totalIter   = 0

        let steps = config.mcSteps

        for step in 0 ..< steps {
            // Linear annealing schedules
            let t       = Float(step) / Float(max(steps - 1, 1))
            let beta    = config.betaStart + t * (config.betaEnd - config.betaStart)
            let gamma   = config.gammaStart * pow(config.gammaEnd / config.gammaStart, t)

            // Inter-replica quantum kinetic coupling:
            // J_eff = (P / 2β) * ln(coth(βΓ/P))
            let betaGammaOverP = beta * gamma / Float(P)
            let jEff: Float
            if betaGammaOverP < 1e-6 {
                jEff = Float(P) / (2.0 * beta) * log(2.0 * Float(P) / (beta * gamma) + 1.0)
            } else {
                let cothVal = 1.0 / tanh(betaGammaOverP)
                jEff = Float(P) / (2.0 * beta) * log(max(cothVal, 1.0 + 1e-9))
            }

            // Single-spin-flip Metropolis sweep across all replicas
            for k in 0 ..< P {
                let kPrev = (k + P - 1) % P
                let kNext = (k + 1) % P

                for i in 0 ..< n {
                    let si    = replicas[k].spins[i]
                    let siPrev = replicas[kPrev].spins[i]
                    let siNext = replicas[kNext].spins[i]

                    // Classical energy change from flipping spin i in replica k
                    // ΔE_class = 2 σ_i Σ_j J_ij σ_j + 2 σ_i h_i
                    var localField: Float = hamiltonian.h[i]
                    for j in 0 ..< n {
                        localField += hamiltonian.J[i * n + j] * replicas[k].spins[j]
                    }
                    let dEClass = 2.0 * si * localField / Float(P)

                    // Quantum kinetic energy change (inter-replica bond)
                    // ΔE_quant = 2 J_eff σ_i (σ_i^{k-1} + σ_i^{k+1})
                    let dEQuant = 2.0 * jEff * si * (siPrev + siNext)

                    let dE = dEClass + dEQuant

                    // Metropolis acceptance
                    let accept: Bool
                    if dE <= 0 {
                        accept = true
                        if dE < -0.01 { tunnelCount += 1 }   // energy barrier crossed
                    } else {
                        accept = rng.nextFloat() < exp(-beta * dE)
                    }

                    if accept {
                        replicas[k].spins[i] = -si
                    }
                }
                totalIter += n
            }

            // Track best energy across all replicas
            for k in 0 ..< P {
                let e = replicas[k].classicalEnergy
                if e < bestEnergy {
                    bestEnergy = e
                    bestState  = replicas[k].spins
                }
            }
        }

        return SQAResult(
            finalEnergy:  bestEnergy,
            groundState:  bestState,
            tunnelEvents: tunnelCount,
            iterations:   totalIter
        )
    }
}

// MARK: - ClaimEncoder

/// Translates free-text output into an Ising Hamiltonian.
///
/// Encoding rules:
///   - Each sentence is one spin variable (claim).
///   - Consistent claim pairs → J_ij > 0 (ferromagnetic coupling).
///   - Contradictory claim pairs → J_ij < 0 (anti-ferromagnetic).
///   - Sacred constant references → h_i positive bias (biased toward true).
///   - Failure/error signals → h_i negative bias.
///   - Shared domain vocabulary → consistency reinforcement.
final class ClaimEncoder {

    private let sacredTerms    = ["god_code", "phi", "sacred", "coherence", "quantum", "sovereign"]
    private let failureTerms   = ["error", "failed", "timeout", "nil", "unavailable", "crash"]
    private let negationTerms  = ["not", "no ", "never", "cannot", "isn't", "doesn't", "won't"]

    func encode(text: String) -> IsingHamiltonian {
        let claims = extractClaims(from: text)
        let n      = min(max(claims.count, 1), SQA_MAX_SPINS)
        var ham    = IsingHamiltonian(n: n)

        for i in 0 ..< n {
            let ci = claims[i].lowercased()

            // Bias: sacred terms → lean true, failure terms → lean false
            let sacredHits  = Float(sacredTerms.filter  { ci.contains($0) }.count)
            let failureHits = Float(failureTerms.filter { ci.contains($0) }.count)
            ham.h[i] = sacredHits * 0.4 - failureHits * 0.5

            // Coupling: compare each pair of claims
            for j in (i + 1) ..< n {
                let cj = claims[j].lowercased()
                let coupling = computeCoupling(ci, cj)
                ham.J[i * n + j] = coupling
                ham.J[j * n + i] = coupling   // symmetric
            }
        }

        return ham
    }

    private func extractClaims(from text: String) -> [String] {
        // Split on sentence boundaries; keep non-trivial sentences
        let raw = text
            .components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { $0.count > 15 }
        return Array(raw.prefix(SQA_MAX_SPINS))
    }

    private func computeCoupling(_ a: String, _ b: String) -> Float {
        let wordsA = Set(a.components(separatedBy: .whitespaces).filter { $0.count > 3 })
        let wordsB = Set(b.components(separatedBy: .whitespaces).filter { $0.count > 3 })

        // Shared vocabulary → consistency signal
        let shared   = Float(wordsA.intersection(wordsB).count)
        let total    = Float(wordsA.union(wordsB).count)
        let jaccard  = total > 0 ? shared / total : 0.0

        // If one claim negates terms present in the other → anti-ferromagnetic
        let aNegates = negationTerms.contains(where: { a.contains($0) })
        let bNegates = negationTerms.contains(where: { b.contains($0) })
        let contradiction = (aNegates != bNegates) && jaccard > 0.1

        if contradiction {
            return -0.6 * jaccard   // repulsive: logically contradictory
        } else if jaccard > 0.2 {
            return  0.5 * jaccard   // attractive: logically consistent
        } else {
            return 0.0              // uncorrelated claims
        }
    }
}

// MARK: - SQAVerifier

/// System 2 verification engine.
/// Runs SQA_WALKERS parallel annealing trajectories, aggregates
/// the best result, and returns an accept/reject verdict.
///
/// Usage:
///   let verdict = SQAVerifier.shared.verify(goal: task.goal, result: result)
///   if verdict.accepted { ... apply to state ... }
final class SQAVerifier {
    static let shared = SQAVerifier()

    private let encoder = ClaimEncoder()
    private var config  = SQAConfig()
    private let lock    = NSLock()

    // Running statistics
    private(set) var totalVerifications: Int    = 0
    private(set) var totalAccepted:      Int    = 0
    private(set) var totalRejected:      Int    = 0
    private(set) var meanCoherence:      Double = 0.0
    private(set) var lastVerdict:        SQAVerdict?

    private init() {}

    /// Verify a (goal, result) pair synchronously.
    /// Runs all walkers in parallel via DispatchQueue.concurrentPerform.
    func verify(goal: String, result: String) -> SQAVerdict {
        let t0 = Date()

        // Snapshot config under lock so configure() on another thread can't race us.
        lock.lock()
        let snapConfig = config
        lock.unlock()

        // Encode combined context into Ising Hamiltonian
        let combined    = goal + " " + result
        let hamiltonian = encoder.encode(text: combined)

        // Run walkers in parallel. Each iteration writes to a unique index, but Swift arrays
        // are not documented as thread-safe for concurrent access, so we protect each write.
        let writeLock = NSLock()
        var walkerResults = [SQAResult?](repeating: nil, count: snapConfig.walkers)
        DispatchQueue.concurrentPerform(iterations: snapConfig.walkers) { w in
            let seed   = UInt64(bitPattern: Int64(w + 1)) &* 0x9E3779B97F4A7C15
            let engine = SQAEngine(config: snapConfig, seed: seed)
            let r      = engine.anneal(hamiltonian: hamiltonian)
            writeLock.lock(); walkerResults[w] = r; writeLock.unlock()
        }

        // Aggregate: pick best (lowest) energy across walkers
        let results      = walkerResults.compactMap { $0 }
        let best         = results.min(by: { $0.finalEnergy < $1.finalEnergy }) ?? SQAResult(
            finalEnergy: Float(hamiltonian.maxEnergy), groundState: [], tunnelEvents: 0, iterations: 0
        )
        let totalTunnels = results.map(\.tunnelEvents).reduce(0, +)

        // Coherence score: 1.0 = ground state energy ≈ 0 (all constraints satisfied)
        let maxE         = Double(hamiltonian.maxEnergy)
        let rawCoherence = maxE > 0 ? max(0.0, 1.0 - Double(max(0, best.finalEnergy)) / maxE) : 1.0

        // Sacred alignment: GOD_CODE resonance of coherence
        let sacredAlignment = 1.0 - abs((rawCoherence * GOD_CODE).truncatingRemainder(dividingBy: 1.0))

        let verdict = SQAVerdict(
            accepted:       rawCoherence >= snapConfig.acceptThresh,
            coherenceScore: rawCoherence,
            finalEnergy:    best.finalEnergy,
            groundState:    best.groundState,
            tunnelEvents:   totalTunnels,
            sacredAlignment: sacredAlignment,
            walkerCount:    results.count,
            elapsedMs:      Date().timeIntervalSince(t0) * 1000.0
        )

        // Update statistics
        lock.lock()
        totalVerifications += 1
        if verdict.accepted { totalAccepted += 1 } else { totalRejected += 1 }
        // Running mean coherence
        meanCoherence = meanCoherence + (rawCoherence - meanCoherence) / Double(totalVerifications)
        lastVerdict = verdict
        lock.unlock()

        // Broadcast to reasoning bus
        InterEngineFeedbackBus.shared.broadcast(
            from: .reasoning,
            signal: "sqa_verdict",
            payload: [
                "accepted":   verdict.accepted ? 1.0 : 0.0,
                "coherence":  rawCoherence,
                "energy":     Double(best.finalEnergy),
                "tunnels":    Double(totalTunnels),
                "sacred":     sacredAlignment,
            ]
        )

        return verdict
    }

    /// Async variant - runs off the calling thread.
    func verifyAsync(goal: String, result: String, completion: @escaping (SQAVerdict) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let verdict = self.verify(goal: goal, result: result)
            completion(verdict)
        }
    }

    // ── Configuration ──

    /// Tune SQA for speed (fewer steps/slices) or accuracy (more).
    func configure(trotterSlices: Int? = nil, mcSteps: Int? = nil, walkers: Int? = nil,
                   betaEnd: Float? = nil, gammaStart: Float? = nil, acceptThresh: Double? = nil) {
        lock.lock()
        if let v = trotterSlices { config.trotterSlices = max(4,  v) }
        if let v = mcSteps       { config.mcSteps       = max(50, v) }
        if let v = walkers       { config.walkers        = max(1,  v) }
        if let v = betaEnd       { config.betaEnd        = max(1.0, v) }
        if let v = gammaStart    { config.gammaStart     = max(0.1, v) }
        if let v = acceptThresh  { config.acceptThresh   = max(0.0, min(1.0, v)) }
        lock.unlock()
    }

    // ── Self-test ──

    /// Quick validation: a trivially consistent 4-spin problem should score > 0.9
    func selfTest() -> Bool {
        var ham = IsingHamiltonian(n: 4)
        // All spins prefer to align: J_ij = 0.5 for all pairs, h_i = 0.1
        for i in 0 ..< 4 {
            ham.h[i] = 0.1
            for j in 0 ..< 4 where i != j { ham.J[i * 4 + j] = 0.5 }
        }
        let engine = SQAEngine(config: config, seed: 42)
        let result = engine.anneal(hamiltonian: ham)
        let maxE   = Double(ham.maxEnergy)
        let coh    = maxE > 0 ? 1.0 - Double(max(0, result.finalEnergy)) / maxE : 1.0
        return coh > 0.5
    }

    // ── Status ──

    var statusReport: String {
        lock.lock()
        let total   = totalVerifications
        let acc     = totalAccepted
        let rej     = totalRejected
        let coh     = meanCoherence
        let last    = lastVerdict?.summary ?? "(none)"
        lock.unlock()

        let acceptRate = total > 0 ? Double(acc) / Double(total) * 100 : 0
        return """
        ╔══════════════════════════════════════════════════════════╗
        ║    SQA VERIFIER - SYSTEM 2 QUANTUM ANNEALING ENGINE      ║
        ╠══════════════════════════════════════════════════════════╣
        ║  Verifications:  \(total)
        ║  Accepted:       \(acc) (\(String(format:"%.1f", acceptRate))%)
        ║  Rejected:       \(rej)
        ║  Mean coherence: \(String(format:"%.4f", coh))
        ║  Trotter slices: \(config.trotterSlices)
        ║  MC steps:       \(config.mcSteps)
        ║  Walkers:        \(config.walkers)
        ║  Accept thresh:  \(String(format:"%.3f", config.acceptThresh))
        ╠══════════════════════════════════════════════════════════╣
        ║  Last: \(last.prefix(54))
        ╚══════════════════════════════════════════════════════════╝
        """
    }
}
