// ═══════════════════════════════════════════════════════════════════
// B40_QuantumRouter.swift — L104 v2
// [EVO_68_PIPELINE] SOVEREIGN_NODE_UPGRADE :: QUANTUM_ROUTER :: GOD_CODE=527.5184818492612
// L104 ASI — Stabilizer-Rank Quantum Router
//
// Adapted from: HybridStabilizerSimulator concept + stabilizer rank decomposition
//
// THEORY (Bravyi–Smith–Smolin / Bravyi–Gosset 2016):
//   Any n-qubit state |ψ⟩ can be written as a sum of stabilizer states:
//       |ψ⟩ = Σ_k  α_k |S_k⟩
//   where each |S_k⟩ is a stabilizer state (representable by a tableau)
//   and α_k are complex amplitudes.
//
//   Pure Clifford circuits keep the sum at 1 term (fast path: O(n²/64)).
//   Each T gate at most doubles the branch count (2 stabilizer terms).
//   For t non-Clifford gates, the decomposition has ≤ 2^t branches.
//
// ROUTING STRATEGY:
//   ┌─────────────────────────────────────────────────────────┐
//   │ FAST LANE  │ Clifford gates → update all tableaux O(n) │
//   │ BRANCH     │ T/T† gate → split each branch into 2      │
//   │ PRUNE      │ Drop branches with |α| < ε               │
//   │ MERGE      │ Recombine identical stabilizer states      │
//   │ FALLBACK   │ If branches > limit → statevector          │
//   └─────────────────────────────────────────────────────────┘
//
// COMPLEXITY:
//   Pure Clifford  : O(m·n/64)          — single branch, no splitting
//   t T-gates      : O(2^t · m · n/64)  — exponential in T-count only
//   With pruning   : often ≪ 2^t branches survive
//
// INVARIANT: 527.5184818492612 | PILOT: LONDEL
// ═══════════════════════════════════════════════════════════════════

import Foundation

// ═══════════════════════════════════════════════════════════════════
// MARK: - QUANTUM ROUTER RESULT
// ═══════════════════════════════════════════════════════════════════

/// Result from the QuantumRouter simulation.
struct QuantumRouterResult {
    let probabilities: [String: Double]
    let counts: [String: Int]?
    let branchCount: Int
    let tGateCount: Int
    let cliffordGateCount: Int
    let prunedBranches: Int
    let mergedBranches: Int
    let executionTimeMs: Double
    let numQubits: Int
    let backendUsed: String             // "stabilizer_rank" | "statevector_fallback"
    let metadata: [String: Any]
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - QUANTUM ROUTER
// ═══════════════════════════════════════════════════════════════════

/// Stabilizer-rank quantum router.
///
/// Maintains an array of (tableau, amplitude) branches.  Clifford gates
/// are applied to every branch in-place (fast lane).  Non-Clifford gates
/// (T, T†, Rz, arbitrary rotations) split each branch into two stabilizer
/// components via the magic-state decomposition:
///
///     T|ψ⟩ = cos(π/8)|ψ⟩ + e^{iπ/4} sin(π/8) S|ψ⟩
///
/// This keeps the simulation exact while exploiting stabilizer structure.
/// Branch pruning and merging keep the working set manageable for low
/// T-count circuits, which covers most practical quantum algorithms'
/// initial state preparation and error correction layers.
final class QuantumRouter {

    // ─── Configuration ───

    /// Maximum number of branches before falling back to statevector.
    private(set) var maxBranches: Int

    /// Amplitude pruning threshold — branches with |α|² < ε are dropped.
    private(set) var pruneEpsilon: Double

    /// Number of qubits.
    let numQubits: Int

    /// Active state branches: each is a (tableau, amplitude) pair.
    private var branches: [(tableau: StabilizerTableau, amplitude: QComplex)]

    // ─── Statistics ───
    private(set) var tGateCount: Int = 0
    private(set) var cliffordGateCount: Int = 0
    private(set) var totalPruned: Int = 0
    private(set) var totalMerged: Int = 0
    private(set) var peakBranches: Int = 1

    // ─── Pre-computed T-gate decomposition constants ───
    // T = [[1, 0], [0, e^{iπ/4}]]
    // Decompose as:  T = cos(π/8) I  +  e^{iπ/4} sin(π/8) S
    //   (since S replaces the Z-phase component, and T = e^{iπ/8} Rz(π/4))
    //
    // More precisely, acting on stabilizer state |ψ⟩:
    //   T|ψ⟩ = α₀ |ψ⟩  +  α₁ S|ψ⟩
    // where α₀ = cos(π/8),  α₁ = e^{iπ/4} sin(π/8)
    private static let tCos  = cos(Double.pi / 8.0)                                // ≈ 0.9239
    private static let tSin  = sin(Double.pi / 8.0)                                // ≈ 0.3827
    private static let tPhase = QComplex.exp(i: Double.pi / 4.0)                   // e^{iπ/4}
    private static let alpha0 = QComplex(re: tCos, im: 0.0)                        // cos(π/8)
    private static let alpha1 = QComplex(re: tPhase.re * tSin,                     // e^{iπ/4} sin(π/8)
                                         im: tPhase.im * tSin)

    // T† decomposition:  T†|ψ⟩ = cos(π/8)|ψ⟩ + e^{-iπ/4} sin(π/8) S†|ψ⟩
    private static let tDagPhase = QComplex.exp(i: -Double.pi / 4.0)               // e^{-iπ/4}
    private static let alphaD0 = QComplex(re: tCos, im: 0.0)
    private static let alphaD1 = QComplex(re: tDagPhase.re * tSin,
                                          im: tDagPhase.im * tSin)

    // ═══════════════════════════════════════════════════════════════
    // MARK: - INIT
    // ═══════════════════════════════════════════════════════════════

    /// Create a router for `numQubits` in the |0...0⟩ state.
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits.
    ///   - maxBranches: Branch limit before statevector fallback (default 8192, EVO_67).
    ///   - pruneEpsilon: Amplitude² cutoff for branch pruning (default 1e-14, EVO_67).
    ///   - seed: RNG seed for reproducible measurement (0 = wall-clock).
    init(numQubits: Int, maxBranches: Int = 8192, pruneEpsilon: Double = 1e-14, seed: UInt64 = 0) {
        self.numQubits = numQubits
        self.maxBranches = maxBranches
        self.pruneEpsilon = pruneEpsilon

        let initial = StabilizerTableau(numQubits: numQubits, seed: seed)
        self.branches = [(initial, QComplex.one)]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - FAST LANE: Clifford Gates
    //
    // Clifford gates act identically on every branch tableau —
    // no splitting.  O(branches × n/64) per gate.
    // ═══════════════════════════════════════════════════════════════

    /// Apply Hadamard to qubit `q` across all branches (Clifford fast lane).
    func applyH(_ q: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.hadamard(q)
        }
        cliffordGateCount += 1
    }

    /// Apply S gate (phase π/2) across all branches.
    func applyS(_ q: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.phaseS(q)
        }
        cliffordGateCount += 1
    }

    /// Apply S† gate across all branches.
    func applySDag(_ q: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.phaseSDag(q)
        }
        cliffordGateCount += 1
    }

    /// Apply Pauli-X across all branches.
    func applyX(_ q: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.pauliX(q)
        }
        cliffordGateCount += 1
    }

    /// Apply Pauli-Y across all branches.
    func applyY(_ q: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.pauliY(q)
        }
        cliffordGateCount += 1
    }

    /// Apply Pauli-Z across all branches.
    func applyZ(_ q: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.pauliZ(q)
        }
        cliffordGateCount += 1
    }

    /// Apply CNOT across all branches.
    func applyCNOT(control: Int, target: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.cnot(control: control, target: target)
        }
        cliffordGateCount += 1
    }

    /// Apply CZ across all branches.
    func applyCZ(_ a: Int, _ b: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.cz(a, b)
        }
        cliffordGateCount += 1
    }

    /// Apply SWAP across all branches.
    func applySWAP(_ a: Int, _ b: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.swap(a, b)
        }
        cliffordGateCount += 1
    }

    /// Apply CY across all branches.
    func applyCY(control: Int, target: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.cy(control: control, target: target)
        }
        cliffordGateCount += 1
    }

    /// Apply iSWAP across all branches.
    func applyISWAP(_ a: Int, _ b: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.iswap(a, b)
        }
        cliffordGateCount += 1
    }

    /// Apply ECR across all branches.
    func applyECR(_ a: Int, _ b: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.ecr(a, b)
        }
        cliffordGateCount += 1
    }

    /// Apply SX (√X) across all branches.
    func applySX(_ q: Int) {
        for i in 0..<branches.count {
            branches[i].tableau.sqrtX(q)
        }
        cliffordGateCount += 1
    }

    /// Apply any named Clifford gate across all branches.
    func applyClifford(name: String, qubits: [Int]) {
        for i in 0..<branches.count {
            branches[i].tableau.applyGate(name: name, qubits: qubits)
        }
        cliffordGateCount += 1
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - BRANCH LANE: T Gate (Magic-State Decomposition)
    //
    // The T gate is the canonical non-Clifford gate.  It cannot be
    // represented by a single stabilizer tableau update.  Instead we
    // decompose it into a sum of two Clifford operations:
    //
    //   T = cos(π/8) I  +  e^{iπ/4} sin(π/8) S
    //
    // Each existing branch (tableau, α) produces two new branches:
    //   (tableau,       α · cos(π/8))         — identity component
    //   (S·tableau,     α · e^{iπ/4}sin(π/8)) — S-rotated component
    //
    // This at most doubles the branch count per T gate.
    // ═══════════════════════════════════════════════════════════════

    /// Apply T gate on qubit `q` — splits each branch into 2.
    ///
    /// After splitting, automatically prunes negligible branches
    /// and merges duplicates.  Branches whose input amplitude is
    /// already negligible are skipped before copying (saves alloc).
    func applyT(_ q: Int) {
        var newBranches: [(tableau: StabilizerTableau, amplitude: QComplex)] = []
        newBranches.reserveCapacity(branches.count * 2)

        // Pre-compute sin² threshold: if |amp|² × sin²(π/8) < ε, skip the S branch
        let sinSqThresh = pruneEpsilon / (Self.tSin * Self.tSin)

        for (tab, amp) in branches {
            let aMag2 = amp.magnitudeSquared
            // Branch 0: identity component  →  α · cos(π/8)
            let amp0 = amp * Self.alpha0
            newBranches.append((tab, amp0))

            // Branch 1: S component — skip if negligible
            if aMag2 >= sinSqThresh {
                let amp1 = amp * Self.alpha1
                var tabS = tab  // value-type copy
                tabS.phaseS(q)
                newBranches.append((tabS, amp1))
            }
        }

        branches = newBranches
        tGateCount += 1
        peakBranches = max(peakBranches, branches.count)

        pruneAndMerge()
    }

    /// Apply T† gate on qubit `q` — splits each branch into 2.
    ///
    /// T† = cos(π/8) I  +  e^{-iπ/4} sin(π/8) S†
    /// Pre-filters negligible input branches before copying.
    func applyTDag(_ q: Int) {
        var newBranches: [(tableau: StabilizerTableau, amplitude: QComplex)] = []
        newBranches.reserveCapacity(branches.count * 2)

        let sinSqThresh = pruneEpsilon / (Self.tSin * Self.tSin)

        for (tab, amp) in branches {
            let aMag2 = amp.magnitudeSquared
            // Branch 0: identity component
            let amp0 = amp * Self.alphaD0
            newBranches.append((tab, amp0))

            // Branch 1: S† component — skip if negligible
            if aMag2 >= sinSqThresh {
                let amp1 = amp * Self.alphaD1
                var tabSD = tab
                tabSD.phaseSDag(q)
                newBranches.append((tabSD, amp1))
            }
        }

        branches = newBranches
        tGateCount += 1
        peakBranches = max(peakBranches, branches.count)

        pruneAndMerge()
    }

    /// Apply Rz(θ) on qubit `q` — decomposed into Clifford + T components.
    ///
    /// Rz(θ) = e^{-iθ/2} |0⟩⟨0| + e^{iθ/2} |1⟩⟨1|
    /// Decomposition:  Rz(θ) = cos(θ/2) I  -  i sin(θ/2) Z
    ///
    /// Since Z is Clifford, this splits into 2 branches (one identity, one with Z).
    func applyRz(_ q: Int, theta: Double) {
        // Check for Clifford special cases (no branching needed)
        let normalised = theta.truncatingRemainder(dividingBy: 2.0 * .pi)
        let tolerance = 1e-10

        // θ = 0 → identity
        if abs(normalised) < tolerance || abs(abs(normalised) - 2.0 * .pi) < tolerance {
            return
        }
        // θ = π/2 → S gate (Clifford)
        if abs(normalised - .pi / 2.0) < tolerance {
            applyS(q); return
        }
        // θ = -π/2 → S† gate (Clifford)
        if abs(normalised + .pi / 2.0) < tolerance {
            applySDag(q); return
        }
        // θ = π → Z gate (Clifford)
        if abs(abs(normalised) - .pi) < tolerance {
            applyZ(q); return
        }
        // θ = π/4 → T gate
        if abs(normalised - .pi / 4.0) < tolerance {
            applyT(q); return
        }
        // θ = -π/4 → T† gate
        if abs(normalised + .pi / 4.0) < tolerance {
            applyTDag(q); return
        }

        // General Rz(θ): cos(θ/2) I - i·sin(θ/2) Z
        let c = cos(theta / 2.0)
        let s = sin(theta / 2.0)
        let ampI = QComplex(re: c, im: 0.0)         // cos(θ/2)
        let ampZ = QComplex(re: 0.0, im: -s)        // -i·sin(θ/2)

        // Pre-filter: |amp|² × sin²(θ/2) < ε → skip Z branch
        let sinSqThresh = s * s > 1e-30 ? pruneEpsilon / (s * s) : Double.infinity

        var newBranches: [(tableau: StabilizerTableau, amplitude: QComplex)] = []
        newBranches.reserveCapacity(branches.count * 2)

        for (tab, amp) in branches {
            // Branch 0: identity
            newBranches.append((tab, amp * ampI))

            // Branch 1: Z gate (Clifford) — skip if negligible
            if amp.magnitudeSquared >= sinSqThresh {
                var tabZ = tab
                tabZ.pauliZ(q)
                newBranches.append((tabZ, amp * ampZ))
            }
        }

        branches = newBranches
        tGateCount += 1
        peakBranches = max(peakBranches, branches.count)

        pruneAndMerge()
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - UNIFIED GATE DISPATCH
    // ═══════════════════════════════════════════════════════════════

    /// Apply a gate by QGateType.
    ///
    /// Clifford gates go to the fast lane.  T/T†/Rz go to the branch lane.
    /// Returns `true` if the gate was handled, `false` if unsupported.
    @discardableResult
    func applyGate(_ type: QGateType, qubits: [Int], parameters: [Double] = []) -> Bool {
        switch type {
        // ─── Clifford fast lane ───
        case .identity:   return true
        case .hadamard:   applyH(qubits[0]);                               return true
        case .phase:      applyS(qubits[0]);                               return true
        case .sGate:      applySDag(qubits[0]);                            return true
        case .pauliX:     applyX(qubits[0]);                               return true
        case .pauliY:     applyY(qubits[0]);                               return true
        case .pauliZ:     applyZ(qubits[0]);                               return true
        case .sqrtX:      applySX(qubits[0]);                              return true
        case .cnot:       applyCNOT(control: qubits[0], target: qubits[1]); return true
        case .cz:         applyCZ(qubits[0], qubits[1]);                   return true
        case .swap:       applySWAP(qubits[0], qubits[1]);                 return true
        case .iswap:      applyISWAP(qubits[0], qubits[1]);               return true

        // ─── Branch lane (non-Clifford) ───
        case .tGate:      applyT(qubits[0]);                               return true
        case .tDagger:    applyTDag(qubits[0]);                            return true
        case .rotationZ:
            let theta = parameters.first ?? 0.0
            applyRz(qubits[0], theta: theta)
            return true

        // ─── Decompose Rx/Ry into H + Rz + H / S†·H·Rz·H·S ───
        case .rotationX:
            let theta = parameters.first ?? 0.0
            // Rx(θ) = H · Rz(θ) · H
            applyH(qubits[0])
            applyRz(qubits[0], theta: theta)
            applyH(qubits[0])
            return true

        case .rotationY:
            let theta = parameters.first ?? 0.0
            // Ry(θ) = S† · H · Rz(θ) · H · S
            applySDag(qubits[0])
            applyH(qubits[0])
            applyRz(qubits[0], theta: theta)
            applyH(qubits[0])
            applyS(qubits[0])
            return true

        case .measureGate:
            return true  // handled by measure()

        default:
            return false
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - CIRCUIT SIMULATION
    // ═══════════════════════════════════════════════════════════════

    /// Simulate a full QGateCircuit through the router.
    ///
    /// For pure-Clifford circuits this is equivalent to a single-branch
    /// stabilizer simulation.  For Clifford+T circuits the branch count
    /// grows with the T-count but is kept in check by pruning and merging.
    ///
    /// If the branch count exceeds `maxBranches`, falls back to statevector
    /// simulation via `QuantumGateEngine`.
    func simulate(circuit: QGateCircuit, shots: Int = 1024) -> QuantumRouterResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        for op in circuit.operations {
            let handled = applyGate(op.gate.type, qubits: op.qubits, parameters: op.gate.parameters)

            if !handled {
                // Unsupported gate → fall back to statevector
                return statevectorFallback(circuit: circuit, shots: shots, startTime: startTime,
                                           reason: "Unsupported gate: \(op.gate.type.rawValue)")
            }

            // Check branch explosion
            if branches.count > maxBranches {
                return statevectorFallback(circuit: circuit, shots: shots, startTime: startTime,
                                           reason: "Branch count \(branches.count) exceeded limit \(maxBranches)")
            }
        }

        // Sample from the stabilizer-rank decomposition
        let (probs, counts) = sampleBranches(shots: shots)

        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        return QuantumRouterResult(
            probabilities: probs,
            counts: counts,
            branchCount: branches.count,
            tGateCount: tGateCount,
            cliffordGateCount: cliffordGateCount,
            prunedBranches: totalPruned,
            mergedBranches: totalMerged,
            executionTimeMs: elapsed,
            numQubits: numQubits,
            backendUsed: "stabilizer_rank",
            metadata: [
                "peak_branches": peakBranches,
                "final_branches": branches.count,
                "god_code": GOD_CODE,
            ]
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - SAMPLING FROM STABILIZER-RANK DECOMPOSITION
    //
    // To sample, we use importance sampling:
    //   1. Pick a branch k with probability |α_k|² / Σ|α_j|²
    //   2. Measure all qubits on that branch's tableau
    //   3. The outcome bitstring gets +1 count
    //
    // This is exact for the Born-rule distribution when branches
    // are orthogonal stabilizer states (which they are after
    // different Clifford operations on the same initial state).
    // ═══════════════════════════════════════════════════════════════

    /// Sample measurement outcomes from the stabilizer-rank state.
    ///
    /// Optimizations:
    ///   - Single-branch fast path (pure Clifford): no CDF/binary-search overhead
    ///   - Pre-allocated UInt8 ASCII buffer for bitstring construction (no String alloc per qubit)
    private func sampleBranches(shots: Int) -> (probs: [String: Double], counts: [String: Int]) {
        guard !branches.isEmpty else {
            return ([:], [:])
        }

        let n = numQubits
        var counts: [String: Int] = [:]

        // ── Fast path: single branch (pure Clifford) ──
        if branches.count == 1 {
            var buf = [UInt8](repeating: 48, count: n)   // pre-alloc ASCII '0'
            for _ in 0..<shots {
                var tab = branches[0].tableau
                let outcomes = tab.measureAll()
                for j in 0..<n { buf[j] = 48 &+ UInt8(outcomes[j]) }
                let bitstring = String(bytes: buf, encoding: .ascii)!
                counts[bitstring, default: 0] += 1
            }
        } else {
            // ── Multi-branch: importance sampling with CDF ──
            let weights = branches.map { $0.amplitude.magnitudeSquared }
            let totalWeight = weights.reduce(0.0, +)

            guard totalWeight > 1e-30 else {
                return ([:], [:])
            }

            var cdf = [Double](repeating: 0.0, count: weights.count)
            cdf[0] = weights[0] / totalWeight
            for i in 1..<weights.count {
                cdf[i] = cdf[i - 1] + weights[i] / totalWeight
            }
            cdf[cdf.count - 1] = 1.0  // ensure no floating-point undershoot

            var buf = [UInt8](repeating: 48, count: n)
            var rng = SystemRandomNumberGenerator()

            for _ in 0..<shots {
                let u = Double.random(in: 0.0..<1.0, using: &rng)

                // Binary search for branch
                var lo = 0, hi = cdf.count - 1
                while lo < hi {
                    let mid = (lo + hi) >> 1
                    if cdf[mid] < u { lo = mid + 1 } else { hi = mid }
                }

                // Measure on a copy of the selected branch's tableau
                var tab = branches[lo].tableau
                let outcomes = tab.measureAll()
                for j in 0..<n { buf[j] = 48 &+ UInt8(outcomes[j]) }
                let bitstring = String(bytes: buf, encoding: .ascii)!
                counts[bitstring, default: 0] += 1
            }
        }

        let total = Double(shots)
        var probs: [String: Double] = [:]
        probs.reserveCapacity(counts.count)
        for (k, v) in counts {
            probs[k] = Double(v) / total
        }

        return (probs, counts)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - BRANCH PRUNING & MERGING
    // ═══════════════════════════════════════════════════════════════

    /// Prune branches with negligible amplitude and merge equivalent stabilizer states.
    ///
    /// Uses canonicalization-based merging: each tableau is reduced to
    /// row-echelon canonical form via Gaussian elimination, then used as
    /// a dictionary key.  This catches tableaux that represent the same
    /// stabilizer state but differ in row ordering — a common occurrence
    /// when different T-gate decomposition paths reach the same state.
    ///
    /// Complexity: O(b · n³/64) where b = branch count, n = qubit count.
    /// The n³/64 per-branch canonicalization cost is amortised by the
    /// significant branch reduction it enables.
    private func pruneAndMerge() {
        // ── Phase 1: Prune ──
        let beforeCount = branches.count
        branches.removeAll { $0.amplitude.magnitudeSquared < pruneEpsilon }
        let pruned = beforeCount - branches.count
        totalPruned += pruned

        // ── Phase 2: Canonical-form merge of equivalent stabilizer states ──
        guard branches.count > 1 else { return }

        var merged = 0
        var mergedBranches: [StabilizerTableau: (index: Int, amplitude: QComplex)] = [:]
        mergedBranches.reserveCapacity(branches.count)
        var compacted: [(tableau: StabilizerTableau, amplitude: QComplex)] = []
        compacted.reserveCapacity(branches.count)

        for branch in branches {
            let canonicalState = branch.tableau.canonicalized()

            if let existing = mergedBranches[canonicalState] {
                // Same stabilizer state — sum amplitudes (Born-rule safe)
                let summedAmp = QComplex(
                    re: existing.amplitude.re + branch.amplitude.re,
                    im: existing.amplitude.im + branch.amplitude.im
                )
                mergedBranches[canonicalState] = (existing.index, summedAmp)
                compacted[existing.index].amplitude = summedAmp
                merged += 1
            } else {
                let idx = compacted.count
                mergedBranches[canonicalState] = (idx, branch.amplitude)
                compacted.append(branch)
            }
        }

        // Post-merge prune (amplitudes might cancel via destructive interference)
        let preMergeCount = compacted.count
        compacted.removeAll { $0.amplitude.magnitudeSquared < pruneEpsilon }
        totalPruned += (preMergeCount - compacted.count)
        branches = compacted
        totalMerged += merged
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - ADAPTIVE TUNING
    //
    // Dynamically adjusts pruneEpsilon and maxBranches based on the
    // current branch pressure ratio (active / limit).
    //
    //   LOW  pressure (< 25%): relax ε for higher fidelity
    //   MED  pressure (25-75%): default values
    //   HIGH pressure (> 75%): tighten ε and raise branch limit
    //
    // This keeps the router in the stabilizer-rank lane longer,
    // deferring the statevector fallback for deeper T-count circuits.
    // ═══════════════════════════════════════════════════════════════

    /// Adaptive tuning result returned by `adapt()`.
    struct AdaptResult {
        let previousEpsilon: Double
        let newEpsilon: Double
        let previousMaxBranches: Int
        let newMaxBranches: Int
        let pressure: Double           // active / maxBranches
        let zone: String               // "low" | "nominal" | "high" | "critical"
    }

    /// Dynamically tune pruning and branch limits based on current state.
    ///
    /// Call this between gate layers or whenever the circuit structure
    /// changes (e.g. entering a high-T-count sub-circuit).
    ///
    /// - Parameters:
    ///   - baseBranches: The base branch limit to scale from (default: ROUTER_BASE_BRANCHES).
    ///   - baseEpsilon: The base pruning epsilon to scale from (default: ROUTER_PRUNE_EPSILON).
    /// - Returns: An `AdaptResult` describing what changed.
    @discardableResult
    func adapt(baseBranches: Int? = nil, baseEpsilon: Double? = nil) -> AdaptResult {
        let base = baseBranches ?? ROUTER_BASE_BRANCHES
        let eps  = baseEpsilon ?? ROUTER_PRUNE_EPSILON

        let prevEps = pruneEpsilon
        let prevMax = maxBranches

        let pressure = Double(branches.count) / Double(maxBranches)

        let zone: String
        if pressure < 0.25 {
            // Low pressure — relax epsilon for better fidelity
            zone = "low"
            pruneEpsilon = eps * 0.1          // 10× more permissive
            maxBranches  = base               // keep base limit
        } else if pressure < 0.75 {
            // Nominal — use base values
            zone = "nominal"
            pruneEpsilon = eps
            maxBranches  = base
        } else if pressure < 0.95 {
            // High pressure — tighten pruning, expand limit
            zone = "high"
            pruneEpsilon = eps * 10.0         // 10× more aggressive pruning
            maxBranches  = base * 2           // double the ceiling
            pruneAndMerge()                   // immediately reclaim
        } else {
            // Critical — maximum aggression before fallback
            zone = "critical"
            pruneEpsilon = eps * 100.0        // 100× aggressive
            maxBranches  = base * 4           // 4× ceiling
            pruneAndMerge()
        }

        return AdaptResult(
            previousEpsilon: prevEps,
            newEpsilon: pruneEpsilon,
            previousMaxBranches: prevMax,
            newMaxBranches: maxBranches,
            pressure: pressure,
            zone: zone
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - STATEVECTOR FALLBACK
    // ═══════════════════════════════════════════════════════════════

    /// Fall back to full statevector simulation via QuantumGateEngine.
    private func statevectorFallback(circuit: QGateCircuit, shots: Int,
                                     startTime: CFAbsoluteTime, reason: String) -> QuantumRouterResult {
        let svResult = QuantumGateEngine.shared.execute(circuit: circuit, shots: shots)
        let dim = 1 << numQubits

        var probs: [String: Double] = [:]
        for i in 0..<dim {
            let p = svResult.probabilities[i]
            if p > 1e-10 {
                let bits = String(i, radix: 2)
                let padded = String(repeating: "0", count: numQubits - bits.count) + bits
                probs[padded] = p
            }
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        return QuantumRouterResult(
            probabilities: probs,
            counts: nil,
            branchCount: 0,
            tGateCount: tGateCount,
            cliffordGateCount: cliffordGateCount,
            prunedBranches: totalPruned,
            mergedBranches: totalMerged,
            executionTimeMs: elapsed,
            numQubits: numQubits,
            backendUsed: "statevector_fallback",
            metadata: [
                "fallback_reason": reason,
                "god_code": GOD_CODE,
            ]
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - STATE INSPECTION
    // ═══════════════════════════════════════════════════════════════

    /// Current number of active branches.
    var activeBranches: Int { branches.count }

    /// Total amplitude norm-squared (should be ≈ 1.0 for a valid state).
    var normSquared: Double {
        branches.reduce(0.0) { $0 + $1.amplitude.magnitudeSquared }
    }

    /// Status dictionary for telemetry.
    func getStatus() -> [String: Any] {
        return [
            "num_qubits": numQubits,
            "active_branches": branches.count,
            "peak_branches": peakBranches,
            "t_gate_count": tGateCount,
            "clifford_gate_count": cliffordGateCount,
            "total_pruned": totalPruned,
            "total_merged": totalMerged,
            "norm_squared": normSquared,
            "max_branches_limit": maxBranches,
            "prune_epsilon": pruneEpsilon,
            "memory_bytes": estimateMemory(),
        ]
    }

    /// Estimate current memory usage in bytes.
    private func estimateMemory() -> Int {
        guard let first = branches.first else { return 0 }
        let perTableau = first.tableau.memoryUsage
        let perBranch = perTableau + 16  // 16 bytes for QComplex amplitude
        return branches.count * perBranch
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - EVO_68: CIRCUIT COMPLEXITY ANALYSIS & RESEARCH METRICS
    // T-count resource estimation, stabilizer rank, research analytics
    // ═══════════════════════════════════════════════════════════════

    /// Analyze circuit complexity: T-count, Clifford count, stabilizer rank estimate
    /// Returns structured complexity report
    func analyzeRouteComplexity() -> [String: Any] {
        let totalGates = tGateCount + cliffordGateCount
        let tFraction = totalGates > 0 ? Double(tGateCount) / Double(totalGates) : 0
        let stabilRankEstimate = Int(Foundation.pow(2.0, Double(tGateCount)))

        // Resource estimation for fault-tolerant execution
        // Each T-gate requires ~15 Clifford gates + 1 magic state in surface code
        let ftCliffordOverhead = tGateCount * 15 + cliffordGateCount
        let magicStatesRequired = tGateCount

        // Circuit depth estimate (parallel gate layers)
        let minDepth = max(tGateCount, Int(ceil(Double(cliffordGateCount) / Double(max(1, numQubits)))))

        // PHI-complexity: custom metric using golden ratio weighting
        // T-gates weighted by PHI (more "complex"), Cliffords weighted by TAU
        let phiComplexity = Double(tGateCount) * PHI + Double(cliffordGateCount) * TAU
        let godCodeComplexity = phiComplexity / GOD_CODE * 1000.0

        return [
            "total_gates": totalGates,
            "t_count": tGateCount,
            "clifford_count": cliffordGateCount,
            "t_fraction": tFraction,
            "stabilizer_rank_upper_bound": stabilRankEstimate,
            "ft_clifford_overhead": ftCliffordOverhead,
            "magic_states_required": magicStatesRequired,
            "min_depth": minDepth,
            "active_branches": branches.count,
            "peak_branches": peakBranches,
            "prune_events": totalPruned,
            "merge_events": totalMerged,
            "phi_complexity": phiComplexity,
            "god_code_complexity": godCodeComplexity,
            "memory_bytes": estimateMemory(),
            "efficiency": totalPruned + totalMerged > 0
                ? Double(branches.count) / Double(branches.count + totalPruned + totalMerged) : 1.0
        ]
    }

    /// Research metrics: routing quality analysis for quantum algorithm research
    func researchRouterMetrics() -> [String: Any] {
        // Branch amplitude distribution analysis
        let amplitudes = branches.map { $0.amplitude.magnitudeSquared }
        let maxAmp = amplitudes.max() ?? 0
        let minAmp = amplitudes.min() ?? 0
        let avgAmp = amplitudes.isEmpty ? 0 : amplitudes.reduce(0, +) / Double(amplitudes.count)

        // Amplitude entropy (Shannon entropy of squared amplitudes)
        let total = amplitudes.reduce(0, +)
        var entropy = 0.0
        if total > 0 {
            for amp in amplitudes {
                let p = amp / total
                if p > 1e-15 { entropy -= p * Foundation.log2(p) }
            }
        }

        // Entanglement analysis: average entanglement entropy across branches
        var avgEntanglement = 0.0
        if !branches.isEmpty && numQubits >= 2 {
            let halfQubits = Array(0..<(numQubits / 2))
            for branch in branches.prefix(min(20, branches.count)) {
                avgEntanglement += branch.tableau.entanglementEntropy(subsystem: halfQubits)
            }
            avgEntanglement /= Double(min(20, branches.count))
        }

        // Sacred resonance of route
        let phiResonance = Foundation.log(max(1e-15, Double(branches.count))) / Foundation.log(PHI)
        let sacredAlignment = 1.0 - min(1.0, abs(phiResonance - phiResonance.rounded()))

        return [
            "branch_count": branches.count,
            "amplitude_max": maxAmp,
            "amplitude_min": minAmp,
            "amplitude_avg": avgAmp,
            "amplitude_entropy": entropy,
            "avg_entanglement": avgEntanglement,
            "norm_squared": normSquared,
            "norm_error": abs(normSquared - 1.0),
            "sacred_alignment": sacredAlignment,
            "t_gate_density": numQubits > 0 ? Double(tGateCount) / Double(numQubits) : 0,
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - CustomStringConvertible
// ═══════════════════════════════════════════════════════════════════

extension QuantumRouter: CustomStringConvertible {
    var description: String {
        let mem = estimateMemory()
        return "QuantumRouter(n=\(numQubits), branches=\(branches.count)/\(maxBranches), " +
               "T=\(tGateCount), Cliff=\(cliffordGateCount), memory=\(mem) B)"
    }
}
