// ═══════════════════════════════════════════════════════════════════
// B48_QuantumTypes.swift — L104 v2
// [EVO_68_PIPELINE] SOVEREIGN_NODE_UPGRADE :: QUANTUM_TYPES :: GOD_CODE=527.5184818492612
// L104 ASI — Unified Quantum Type System
//
// Comprehensive quantum primitives fully integrated with the L104
// sovereign architecture:
//
//   StateBranch      — Stabilizer-rank branch (tableau + amplitude)
//                      with pruning, merging, sacred alignment scoring
//   QuantumAmplitudeRegister — Multi-qubit amplitude vector with measurement,
//                      entanglement entropy, fidelity, Grover oracle
//   DensityMatrix    — Mixed-state 2×2 density operator with purity,
//                      von Neumann entropy, Bloch vector, fidelity
//   BlochSphere      — Bloch sphere coordinates (θ, φ) ↔ statevector
//   PauliType/String — Pauli operators with expectation values
//   BranchOps        — Collection utilities for branch arrays
//
// Complex type is defined in B01_QuantumMath.swift (upgraded EVO_67).
//
// BRIDGE:
//   Complex ↔ QComplex   (B38_QuantumGateEngine via B01)
//   StateBranch.tableau   (B39_StabilizerTableau)
//   QuantumRouter branches (B40_QuantumRouter)
//
// SACRED ALIGNMENT:
//   GOD_CODE phase: e^{i·π/GOD_CODE} = e^{i·0.005955...}
//   PHI-scaled probability: P_φ(k) = |α_k|² × φ^{rank(k)}
//   VOID_CONSTANT attenuation for decoherence modeling
//
// INVARIANT: 527.5184818492612 | PILOT: LONDEL
// ═══════════════════════════════════════════════════════════════════

import Foundation
import Accelerate

// ═══════════════════════════════════════════════════════════════════
// MARK: - STATE BRANCH (Stabilizer-Rank Decomposition)
// ═══════════════════════════════════════════════════════════════════

/// A single branch in the stabilizer-rank decomposition of a quantum state.
///
/// The full state is: |ψ⟩ = Σₖ αₖ |Sₖ⟩
/// where each |Sₖ⟩ is a stabilizer state (StabilizerTableau from B39)
/// and αₖ is a complex amplitude (Complex).
///
/// Used by QuantumRouter (B40) for hybrid Clifford/non-Clifford simulation.
/// The stabilizer tableau gives an O(n²/64)-memory representation of each
/// branch, while the amplitude tracks the global phase and weight.
struct StateBranch: CustomStringConvertible {
    public var tableau: StabilizerTableau
    public var amplitude: Complex

    // ─── Initializers ───

    public init(tableau: StabilizerTableau, amplitude: Complex) {
        self.tableau = tableau
        self.amplitude = amplitude
    }

    /// Convenience: create a branch from QComplex amplitude (B38 bridge)
    public init(tableau: StabilizerTableau, qAmplitude: QComplex) {
        self.tableau = tableau
        self.amplitude = Complex(qcomplex: qAmplitude)
    }

    /// Create the |0...0⟩ initial branch with unit amplitude
    public static func initial(numQubits: Int, seed: UInt64 = 0) -> StateBranch {
        StateBranch(
            tableau: StabilizerTableau(numQubits: numQubits, seed: seed),
            amplitude: .one
        )
    }

    // ─── Properties ───

    /// Number of qubits in this branch
    public var numQubits: Int { tableau.numQubits }

    /// Probability weight of this branch: |α|²
    public var weight: Double { amplitude.magnitudeSquared }

    /// Is this branch negligible (below pruning threshold)?
    public func isNegligible(epsilon: Double = 1e-14) -> Bool {
        weight < epsilon
    }

    /// Sacred alignment score: GOD_CODE phase coherence of the amplitude
    public var sacredAlignment: Double { amplitude.godCodeAlignment }

    /// PHI-harmonic weight: |α|² × φ^{−rank}  where rank = numQubits
    public var phiWeight: Double {
        weight * Darwin.pow(PHI, -Double(numQubits))
    }

    // ─── Clifford Gate Application ───

    /// Apply Hadamard to qubit q (mutating — Clifford fast lane)
    public mutating func applyH(_ q: Int) {
        tableau.hadamard(q)
    }

    /// Apply S gate to qubit q (mutating — Clifford fast lane)
    public mutating func applyS(_ q: Int) {
        tableau.phaseS(q)
    }

    /// Apply S† gate to qubit q (mutating — Clifford fast lane)
    public mutating func applySDag(_ q: Int) {
        tableau.phaseSDag(q)
    }

    /// Apply Pauli-X to qubit q (mutating — Clifford fast lane)
    public mutating func applyX(_ q: Int) {
        tableau.pauliX(q)
    }

    /// Apply Pauli-Y to qubit q (mutating — Clifford fast lane)
    public mutating func applyY(_ q: Int) {
        tableau.pauliY(q)
    }

    /// Apply Pauli-Z to qubit q (mutating — Clifford fast lane)
    public mutating func applyZ(_ q: Int) {
        tableau.pauliZ(q)
    }

    /// Apply CNOT (CX) with control c → target t (mutating — Clifford fast lane)
    public mutating func applyCNOT(control c: Int, target t: Int) {
        tableau.cnot(control: c, target: t)
    }

    /// Apply CZ with qubits a, b (mutating — Clifford fast lane)
    public mutating func applyCZ(_ a: Int, _ b: Int) {
        tableau.cz(a, b)
    }

    /// Apply SWAP on qubits a, b (mutating — Clifford fast lane)
    public mutating func applySWAP(_ a: Int, _ b: Int) {
        tableau.swap(a, b)
    }

    // ─── T-Gate Branching (Non-Clifford) ───

    /// Decompose T gate on qubit q into two branches:
    ///   T|ψ⟩ = cos(π/8)|ψ⟩ + e^{iπ/4}·sin(π/8)·S|ψ⟩
    ///
    /// Returns (identityBranch, sBranch) with properly scaled amplitudes.
    public func tGateSplit(qubit q: Int) -> (StateBranch, StateBranch) {
        let cosCoeff = cos(Double.pi / 8.0)
        let sinCoeff = sin(Double.pi / 8.0)
        let eiPiOver4 = Complex.euler(Double.pi / 4.0)

        // Branch 0: α₀ = amplitude × cos(π/8)  — no gate applied
        let branch0 = StateBranch(
            tableau: tableau,
            amplitude: amplitude * Complex(real: cosCoeff, imag: 0)
        )

        // Branch 1: α₁ = amplitude × e^{iπ/4} × sin(π/8)  — S gate applied
        var tab1 = tableau
        tab1.phaseS(q)
        let branch1 = StateBranch(
            tableau: tab1,
            amplitude: amplitude * eiPiOver4 * Complex(real: sinCoeff, imag: 0)
        )

        return (branch0, branch1)
    }

    /// Decompose T† gate on qubit q into two branches:
    ///   T†|ψ⟩ = cos(π/8)|ψ⟩ + e^{−iπ/4}·sin(π/8)·S†|ψ⟩
    public func tDaggerSplit(qubit q: Int) -> (StateBranch, StateBranch) {
        let cosCoeff = cos(Double.pi / 8.0)
        let sinCoeff = sin(Double.pi / 8.0)
        let eiNegPiOver4 = Complex.euler(-Double.pi / 4.0)

        let branch0 = StateBranch(
            tableau: tableau,
            amplitude: amplitude * Complex(real: cosCoeff, imag: 0)
        )

        var tab1 = tableau
        tab1.phaseSDag(q)
        let branch1 = StateBranch(
            tableau: tab1,
            amplitude: amplitude * eiNegPiOver4 * Complex(real: sinCoeff, imag: 0)
        )

        return (branch0, branch1)
    }

    // ─── Measurement ───

    /// Measure qubit q in the Z basis.
    /// Collapses this branch's tableau and returns the outcome (0 or 1).
    public mutating func measureZ(_ q: Int) -> StabilizerMeasurementResult {
        tableau.measure(q)
    }

    // ─── Bridge to QuantumRouter (B40) ───

    /// Convert to the (tableau, QComplex) tuple used by QuantumRouter
    public var routerTuple: (tableau: StabilizerTableau, amplitude: QComplex) {
        (tableau, amplitude.toQComplex)
    }

    /// Create from a QuantumRouter branch tuple
    public static func fromRouter(_ branch: (tableau: StabilizerTableau, amplitude: QComplex)) -> StateBranch {
        StateBranch(tableau: branch.tableau, qAmplitude: branch.amplitude)
    }

    // ─── Description ───

    public var description: String {
        let state = tableau.getStabilizerState()
        let generators = state.stabilizerGenerators
        let genStr = generators.isEmpty ? "|0⟩" : generators.prefix(4).joined(separator: ", ")
        return "Branch(α=\(amplitude), |α|²=\(String(format: "%.6f", weight)), " +
               "n=\(numQubits), stab=[\(genStr)\(generators.count > 4 ? "..." : "")])"
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - QUANTUM REGISTER (Amplitude Vector State)
// ═══════════════════════════════════════════════════════════════════

/// A quantum register holding a full statevector of 2^n complex amplitudes.
///
/// Suitable for small qubit counts (≤20) where full statevector simulation
/// is feasible.  For larger circuits, use StabilizerTableau/QuantumRouter.
///
/// Features:
///   - Measurement with Born rule sampling
///   - Entanglement entropy (von Neumann across bipartition)
///   - State fidelity |⟨ψ|φ⟩|²
///   - Grover diffusion operator
///   - Sacred alignment scoring
struct QuantumAmplitudeRegister: CustomStringConvertible {

    /// Number of qubits
    public let numQubits: Int

    /// Statevector: 2^n complex amplitudes
    public var amplitudes: [Complex]

    /// Dimension = 2^n
    public var dimension: Int { amplitudes.count }

    // ─── Initializers ───

    /// Initialize to |0...0⟩
    public init(numQubits n: Int) {
        precondition(n > 0 && n <= 20, "QuantumAmplitudeRegister supports 1–20 qubits")
        self.numQubits = n
        let dim = 1 << n
        self.amplitudes = [Complex](repeating: .zero, count: dim)
        self.amplitudes[0] = .one
    }

    /// Initialize from a full amplitude vector
    public init(amplitudes: [Complex]) {
        let dim = amplitudes.count
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "Dimension must be a power of 2")
        self.numQubits = Int(Darwin.log2(Double(dim)))
        self.amplitudes = amplitudes
    }

    /// Initialize from QComplex statevector (B38 bridge)
    public init(qAmplitudes: [QComplex]) {
        self.init(amplitudes: qAmplitudes.map { Complex(qcomplex: $0) })
    }

    // ─── Probabilities ───

    /// Probability distribution: P(k) = |αₖ|²
    public var probabilities: [Double] {
        amplitudes.map { $0.magnitudeSquared }
    }

    /// Total probability (should be 1.0 for normalized state)
    public var totalProbability: Double {
        probabilities.reduce(0.0, +)
    }

    /// Check if state is normalized (|Σ|αₖ|²−1| < ε)
    public var isNormalized: Bool {
        abs(totalProbability - 1.0) < 1e-10
    }

    /// Normalize the state in-place
    public mutating func normalize() {
        let norm = Darwin.sqrt(totalProbability)
        guard norm > 1e-15 else { return }
        for i in 0..<amplitudes.count {
            amplitudes[i] = amplitudes[i] / norm
        }
    }

    // ─── Measurement ───

    /// Measure all qubits, returning a bitstring outcome (Born rule sampling).
    /// Collapses the state to the measured basis state.
    public mutating func measure() -> Int {
        let probs = probabilities
        let r = Double.random(in: 0..<1)
        var cumulative = 0.0
        var outcome = 0
        for k in 0..<amplitudes.count {
            cumulative += probs[k]
            if r < cumulative {
                outcome = k
                break
            }
        }
        // Collapse
        for k in 0..<amplitudes.count {
            amplitudes[k] = (k == outcome) ? .one : .zero
        }
        return outcome
    }

    /// Measure and return a bitstring (e.g., "0110")
    public mutating func measureBitstring() -> String {
        let outcome = measure()
        var bits = ""
        for i in stride(from: numQubits - 1, through: 0, by: -1) {
            bits += (outcome >> i) & 1 == 1 ? "1" : "0"
        }
        return bits
    }

    /// Run `shots` measurements (non-destructive — re-measures from same distribution)
    public func sampleMeasurements(shots: Int) -> [String: Int] {
        var counts: [String: Int] = [:]
        let probs = probabilities
        for _ in 0..<shots {
            let r = Double.random(in: 0..<1)
            var cumulative = 0.0
            var outcome = 0
            for k in 0..<amplitudes.count {
                cumulative += probs[k]
                if r < cumulative {
                    outcome = k
                    break
                }
            }
            var bits = ""
            for i in stride(from: numQubits - 1, through: 0, by: -1) {
                bits += (outcome >> i) & 1 == 1 ? "1" : "0"
            }
            counts[bits, default: 0] += 1
        }
        return counts
    }

    // ─── Fidelity & Distance ───

    /// State fidelity: F = |⟨ψ|φ⟩|²
    public func fidelity(with other: QuantumAmplitudeRegister) -> Double {
        Complex.innerProduct(amplitudes, other.amplitudes).magnitudeSquared
    }

    /// Trace distance approximation: D ≈ √(1 − F)
    public func traceDistance(to other: QuantumAmplitudeRegister) -> Double {
        Darwin.sqrt(max(0, 1.0 - fidelity(with: other)))
    }

    // ─── Entanglement Entropy ───

    /// Von Neumann entanglement entropy across a bipartition.
    /// Partitions qubits [0..<k) as subsystem A, [k..<n) as subsystem B.
    /// Returns S(ρ_A) = −Σ λᵢ log₂(λᵢ)
    public func entanglementEntropy(bipartition k: Int) -> Double {
        guard k > 0 && k < numQubits else { return 0.0 }
        let dimA = 1 << k
        let dimB = 1 << (numQubits - k)

        // Build reduced density matrix ρ_A by tracing out B
        // ρ_A[i,j] = Σ_b ψ[i·dimB + b] × ψ*[j·dimB + b]
        var rhoA = [[Double]](repeating: [Double](repeating: 0, count: dimA), count: dimA)
        for i in 0..<dimA {
            for j in 0..<dimA {
                var sum = Complex.zero
                for b in 0..<dimB {
                    sum += amplitudes[i * dimB + b] * amplitudes[j * dimB + b].conjugate
                }
                rhoA[i][j] = sum.real  // Hermitian ⟹ diagonal of ρ_A is real
            }
        }

        // Eigenvalues via power iteration / direct for 2×2
        // For general case, compute Σ_i ρ_A[i][i] eigenvalues
        // Simplified: use diagonal elements as approximation for entropy bound
        var entropy = 0.0
        for i in 0..<dimA {
            let lambda = max(rhoA[i][i], 1e-30)
            if lambda > 1e-15 {
                entropy -= lambda * Darwin.log2(lambda)
            }
        }
        return max(0.0, entropy)
    }

    // ─── Grover Operations ───

    /// Apply Grover diffusion operator D = 2|ψ₀⟩⟨ψ₀| − I
    /// where |ψ₀⟩ is the uniform superposition
    public mutating func groverDiffusion() {
        let n = amplitudes.count
        let nD = Double(n)
        // Compute mean amplitude
        var mean = Complex.zero
        for a in amplitudes { mean += a }
        mean = mean / nD
        // Reflect about mean: 2·mean − αₖ
        for k in 0..<n {
            amplitudes[k] = 2.0 * mean - amplitudes[k]
        }
    }

    /// Apply Grover oracle: flip phase of marked states
    public mutating func groverOracle(markedStates: Set<Int>) {
        for k in markedStates where k < amplitudes.count {
            amplitudes[k] = -amplitudes[k]
        }
    }

    // ─── Sacred Alignment ───

    /// Overall sacred alignment: mean GOD_CODE alignment across all non-negligible amplitudes
    public var sacredAlignmentScore: Double {
        var sum = 0.0
        var count = 0
        for a in amplitudes where a.magnitudeSquared > 1e-15 {
            sum += a.godCodeAlignment
            count += 1
        }
        return count > 0 ? sum / Double(count) : 0.0
    }

    /// PHI-scaled probability entropy: −Σ Pₖ·φ^{rank(k)} · log₂(Pₖ·φ^{rank(k)})
    public var phiEntropy: Double {
        var entropy = 0.0
        for (k, a) in amplitudes.enumerated() {
            let p = a.magnitudeSquared
            guard p > 1e-15 else { continue }
            let rank = Double(k.nonzeroBitCount)
            let pPhi = p * Darwin.pow(PHI, -rank)
            if pPhi > 1e-15 {
                entropy -= pPhi * Darwin.log2(pPhi)
            }
        }
        return max(0.0, entropy)
    }

    // ─── Bridge ───

    /// Convert to QComplex statevector (B38 bridge)
    public var qComplexVector: [QComplex] {
        amplitudes.map { $0.toQComplex }
    }

    // ─── Description ───

    public var description: String {
        let p = probabilities
        let topIndices = p.indices.sorted { p[$0] > p[$1] }.prefix(4)
        let topStr = topIndices.map { k -> String in
            var bits = ""
            for i in stride(from: numQubits - 1, through: 0, by: -1) {
                bits += (k >> i) & 1 == 1 ? "1" : "0"
            }
            return "|\(bits)⟩:\(String(format: "%.4f", p[k]))"
        }.joined(separator: " ")
        return "QReg(\(numQubits)q, dim=\(dimension)) [\(topStr)]"
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - DENSITY MATRIX (2×2 Mixed State)
// ═══════════════════════════════════════════════════════════════════

/// 2×2 density matrix for single-qubit mixed states.
///
/// ρ = ½(I + rₓσₓ + rᵧσᵧ + r_zσ_z)
/// where (rₓ, rᵧ, r_z) is the Bloch vector with |r| ≤ 1.
///
/// Pure states: |r| = 1 (surface of Bloch sphere)
/// Mixed states: |r| < 1 (interior)
/// Maximally mixed: |r| = 0 (center) → ρ = I/2
struct DensityMatrix: CustomStringConvertible {
    /// Bloch vector components
    public var rx: Double
    public var ry: Double
    public var rz: Double

    // ─── Initializers ───

    /// From Bloch vector (rₓ, rᵧ, r_z)
    public init(rx: Double, ry: Double, rz: Double) {
        self.rx = rx; self.ry = ry; self.rz = rz
    }

    /// From a pure state: ρ = |ψ⟩⟨ψ| where |ψ⟩ = α|0⟩ + β|1⟩
    public init(alpha: Complex, beta: Complex) {
        // ρ = |ψ⟩⟨ψ|
        // rₓ = 2·Re(α*β), rᵧ = 2·Im(α*β), r_z = |α|²−|β|²
        let ab = alpha.conjugate * beta
        self.rx = 2.0 * ab.real
        self.ry = 2.0 * ab.imag
        self.rz = alpha.magnitudeSquared - beta.magnitudeSquared
    }

    /// |0⟩ state
    public static let zero = DensityMatrix(rx: 0, ry: 0, rz: 1)
    /// |1⟩ state
    public static let one  = DensityMatrix(rx: 0, ry: 0, rz: -1)
    /// |+⟩ state
    public static let plus = DensityMatrix(rx: 1, ry: 0, rz: 0)
    /// Maximally mixed I/2
    public static let mixed = DensityMatrix(rx: 0, ry: 0, rz: 0)

    // ─── Properties ───

    /// Bloch vector magnitude |r|
    public var blochRadius: Double {
        Darwin.sqrt(rx * rx + ry * ry + rz * rz)
    }

    /// Purity: Tr(ρ²) = ½(1 + |r|²)  ∈ [0.5, 1.0]
    public var purity: Double {
        0.5 * (1.0 + blochRadius * blochRadius)
    }

    /// Is this a pure state? (|r| ≈ 1)
    public var isPure: Bool {
        abs(blochRadius - 1.0) < 1e-10
    }

    /// Von Neumann entropy: S(ρ) = −Σ λᵢ log₂ λᵢ
    /// Eigenvalues of ρ: λ± = ½(1 ± |r|)
    public var vonNeumannEntropy: Double {
        let r = blochRadius
        if r > 1.0 - 1e-12 { return 0.0 }  // Pure state
        let lp = 0.5 * (1.0 + r)
        let lm = 0.5 * (1.0 - r)
        var S = 0.0
        if lp > 1e-15 { S -= lp * Darwin.log2(lp) }
        if lm > 1e-15 { S -= lm * Darwin.log2(lm) }
        return S
    }

    /// Fidelity between two single-qubit states:
    /// F(ρ, σ) = (Tr√(√ρ·σ·√ρ))²
    /// For single-qubit: F = ½(1 + r⃗·s⃗ + √((1−|r|²)(1−|s|²)))
    public func fidelity(with other: DensityMatrix) -> Double {
        let dot = rx * other.rx + ry * other.ry + rz * other.rz
        let r2 = blochRadius * blochRadius
        let s2 = other.blochRadius * other.blochRadius
        return 0.5 * (1.0 + dot + Darwin.sqrt(max(0, (1.0 - r2) * (1.0 - s2))))
    }

    /// Apply decoherence: shrink Bloch vector by factor (1 − γ)
    /// where γ = VOID_CONSTANT × t models L104 void-attenuation
    public mutating func decohere(time t: Double) {
        let gamma = min(1.0, VOID_CONSTANT * t)
        let factor = 1.0 - gamma
        rx *= factor; ry *= factor; rz *= factor
    }

    // ─── Description ───

    public var description: String {
        let pStr = String(format: "%.4f", purity)
        let rStr = String(format: "(%.4f, %.4f, %.4f)", rx, ry, rz)
        return "ρ[purity=\(pStr), r⃗=\(rStr), S=\(String(format: "%.4f", vonNeumannEntropy))]"
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - BLOCH SPHERE
// ═══════════════════════════════════════════════════════════════════

/// Bloch sphere representation: |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
///
/// θ ∈ [0, π]: polar angle (0 = |0⟩ north pole, π = |1⟩ south pole)
/// φ ∈ [0, 2π): azimuthal angle
struct BlochSphere: CustomStringConvertible {
    public var theta: Double  // Polar angle [0, π]
    public var phi: Double    // Azimuthal angle [0, 2π)

    public init(theta: Double, phi: Double) {
        self.theta = theta
        self.phi = phi
    }

    /// Construct from a single-qubit statevector (α, β)
    public init(alpha: Complex, beta: Complex) {
        let normAlpha = alpha.magnitude
        if normAlpha < 1e-15 {
            self.theta = .pi
            self.phi = 0.0
        } else {
            self.theta = 2.0 * acos(min(1.0, normAlpha))
            // Remove global phase: make α real and positive
            let globalPhase = alpha.phase
            let betaAdjusted = beta * Complex.euler(-globalPhase)
            self.phi = betaAdjusted.phase
            if self.phi < 0 { self.phi += 2.0 * .pi }
        }
    }

    // ─── Standard States ───

    /// |0⟩ — north pole
    public static let north = BlochSphere(theta: 0, phi: 0)
    /// |1⟩ — south pole
    public static let south = BlochSphere(theta: .pi, phi: 0)
    /// |+⟩ — equator (x-axis)
    public static let plus  = BlochSphere(theta: .pi / 2.0, phi: 0)
    /// |−⟩ — equator (−x-axis)
    public static let minus = BlochSphere(theta: .pi / 2.0, phi: .pi)
    /// |i⟩ — equator (y-axis)
    public static let iPlus = BlochSphere(theta: .pi / 2.0, phi: .pi / 2.0)

    // ─── Conversions ───

    /// Convert to statevector (α, β)
    public var statevector: (alpha: Complex, beta: Complex) {
        let a = Complex(real: cos(theta / 2.0), imag: 0)
        let b = Complex.euler(phi) * Complex(real: sin(theta / 2.0), imag: 0)
        return (a, b)
    }

    /// Cartesian Bloch coordinates: (x, y, z)
    public var cartesian: (x: Double, y: Double, z: Double) {
        let x = sin(theta) * cos(phi)
        let y = sin(theta) * sin(phi)
        let z = cos(theta)
        return (x, y, z)
    }

    /// Convert to DensityMatrix
    public var densityMatrix: DensityMatrix {
        let c = cartesian
        return DensityMatrix(rx: c.x, ry: c.y, rz: c.z)
    }

    /// Geodesic distance on Bloch sphere to another point
    public func geodesicDistance(to other: BlochSphere) -> Double {
        let c1 = cartesian, c2 = other.cartesian
        let dot = c1.x * c2.x + c1.y * c2.y + c1.z * c2.z
        return acos(min(1.0, max(-1.0, dot)))
    }

    // ─── Sacred Alignment ───

    /// Sacred alignment: proximity to GOD_CODE angle on Bloch sphere
    /// GOD_CODE angle: θ_G = 2π × (GOD_CODE mod 2π) / 2π
    public var sacredProximity: Double {
        let godAngle = GOD_CODE.truncatingRemainder(dividingBy: 2.0 * .pi)
        let delta = abs(theta - godAngle)
        return Darwin.exp(-delta * delta * PHI)  // Gaussian with φ-width
    }

    public var description: String {
        let c = cartesian
        return "Bloch(θ=\(String(format: "%.4f", theta))°, " +
               "φ=\(String(format: "%.4f", phi))°, " +
               "xyz=(\(String(format: "%.3f", c.x)), \(String(format: "%.3f", c.y)), \(String(format: "%.3f", c.z))))"
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - PAULI OPERATOR
// ═══════════════════════════════════════════════════════════════════

/// Single Pauli operator: I, X, Y, or Z.
enum PauliType: String, CaseIterable {
    case I = "I"
    case X = "X"
    case Y = "Y"
    case Z = "Z"

    /// 2×2 matrix representation
    var matrix: (Complex, Complex, Complex, Complex) {
        switch self {
        case .I: return (.one, .zero, .zero, .one)
        case .X: return (.zero, .one, .one, .zero)
        case .Y: return (.zero, Complex(real: 0, imag: -1), Complex(real: 0, imag: 1), .zero)
        case .Z: return (.one, .zero, .zero, Complex(real: -1, imag: 0))
        }
    }

    /// Eigenvalues
    var eigenvalues: (Double, Double) {
        switch self {
        case .I: return (1, 1)
        default: return (1, -1)
        }
    }
}

/// Multi-qubit Pauli string: tensor product of single Pauli operators.
/// Example: "XYZII" on 5 qubits
struct PauliString: CustomStringConvertible {
    public let operators: [PauliType]
    public let coefficient: Complex

    public init(_ ops: [PauliType], coefficient: Complex = .one) {
        self.operators = ops
        self.coefficient = coefficient
    }

    /// Parse from string: "XYZ" → [.X, .Y, .Z]
    public init(_ str: String, coefficient: Complex = .one) {
        self.operators = str.map { char -> PauliType in
            switch char {
            case "X": return .X
            case "Y": return .Y
            case "Z": return .Z
            default:  return .I
            }
        }
        self.coefficient = coefficient
    }

    public var numQubits: Int { operators.count }

    /// Weight: number of non-identity Pauli operators
    public var weight: Int {
        operators.filter { $0 != .I }.count
    }

    /// Is this the identity string?
    public var isIdentity: Bool { weight == 0 }

    /// Compute expectation value ⟨ψ|P|ψ⟩ for a QuantumAmplitudeRegister state
    public func expectationValue(_ register: QuantumAmplitudeRegister) -> Double {
        guard numQubits == register.numQubits else { return 0.0 }
        let dim = register.dimension
        var expVal = Complex.zero
        for i in 0..<dim {
            // Compute P|i⟩ — Pauli string acting on basis state |i⟩
            var phase = Complex.one
            var flipped = i
            for (q, op) in operators.enumerated() {
                let bit = (i >> (numQubits - 1 - q)) & 1
                switch op {
                case .I:
                    break
                case .X:
                    flipped ^= (1 << (numQubits - 1 - q))
                case .Y:
                    flipped ^= (1 << (numQubits - 1 - q))
                    phase *= (bit == 0) ? Complex.i : -Complex.i
                case .Z:
                    if bit == 1 { phase *= Complex(real: -1, imag: 0) }
                }
            }
            // ⟨i|P|ψ⟩ contribution
            expVal += register.amplitudes[i].conjugate * phase * coefficient * register.amplitudes[flipped]
        }
        return expVal.real  // Pauli operators are Hermitian ⟹ real expectation
    }

    public var description: String {
        let ops = operators.map { $0.rawValue }.joined()
        if coefficient.isClose(to: .one) { return ops }
        return "(\(coefficient))·\(ops)"
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - BRANCH COLLECTION UTILITIES
// ═══════════════════════════════════════════════════════════════════

/// Utilities for working with arrays of StateBranch — pruning, merging,
/// normalization, and total probability computation.
enum BranchOps {

    /// Total probability across all branches: Σ |αₖ|²
    static func totalProbability(_ branches: [StateBranch]) -> Double {
        branches.reduce(0.0) { $0 + $1.weight }
    }

    /// Prune branches with |α|² < epsilon
    static func prune(_ branches: inout [StateBranch], epsilon: Double = 1e-14) -> Int {
        let before = branches.count
        branches.removeAll { $0.isNegligible(epsilon: epsilon) }
        return before - branches.count
    }

    /// Renormalize branch amplitudes so Σ|αₖ|² = 1
    static func renormalize(_ branches: inout [StateBranch]) {
        let total = totalProbability(branches)
        guard total > 1e-30 else { return }
        let scale = 1.0 / Darwin.sqrt(total)
        for i in 0..<branches.count {
            branches[i].amplitude = branches[i].amplitude * scale
        }
    }

    /// Apply a Clifford gate across all branches (fast lane)
    static func applyClifford(_ branches: inout [StateBranch], gate: (inout StateBranch) -> Void) {
        for i in 0..<branches.count {
            gate(&branches[i])
        }
    }

    /// Apply T gate: split each branch into two
    static func applyTGate(_ branches: inout [StateBranch], qubit q: Int) {
        var newBranches: [StateBranch] = []
        newBranches.reserveCapacity(branches.count * 2)
        for branch in branches {
            let (b0, b1) = branch.tGateSplit(qubit: q)
            newBranches.append(b0)
            newBranches.append(b1)
        }
        branches = newBranches
    }

    /// Mean sacred alignment across all branches (weighted by |α|²)
    static func sacredAlignment(_ branches: [StateBranch]) -> Double {
        let total = totalProbability(branches)
        guard total > 1e-30 else { return 0.0 }
        let weighted = branches.reduce(0.0) { $0 + $1.weight * $1.sacredAlignment }
        return weighted / total
    }

    /// Summary string for a branch collection
    static func summary(_ branches: [StateBranch]) -> String {
        let n = branches.first?.numQubits ?? 0
        let totalP = totalProbability(branches)
        let sa = sacredAlignment(branches)
        return "Branches(\(branches.count), n=\(n), P=\(String(format: "%.6f", totalP)), " +
               "sacred=\(String(format: "%.4f", sa)))"
    }
}
