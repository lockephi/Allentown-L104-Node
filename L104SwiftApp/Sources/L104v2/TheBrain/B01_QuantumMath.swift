// ═══════════════════════════════════════════════════════════════════
// B01_QuantumMath.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — Quantum Simulation Engine
//
// Complex numbers, quantum states, multi-qubit registers,
// and pre-built quantum circuits (Bell, GHZ, teleportation,
// Deutsch-Jozsa, QFT).
//
// Extracted from L104Native.swift lines 1566-2239
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// MARK: - Complex Number

struct Complex: CustomStringConvertible {
    var real: Double
    var imag: Double

    init(_ real: Double, _ imag: Double = 0) { self.real = real; self.imag = imag }

    var description: String { imag >= 0 ? "\(String(format: "%.3f", real))+\(String(format: "%.3f", imag))i" : "\(String(format: "%.3f", real))\(String(format: "%.3f", imag))i" }
    var magnitude: Double { sqrt(real * real + imag * imag) }
    var phase: Double { atan2(imag, real) }
    var conjugate: Complex { Complex(real, -imag) }

    static func + (lhs: Complex, rhs: Complex) -> Complex { Complex(lhs.real + rhs.real, lhs.imag + rhs.imag) }
    static func - (lhs: Complex, rhs: Complex) -> Complex { Complex(lhs.real - rhs.real, lhs.imag - rhs.imag) }
    static func * (lhs: Complex, rhs: Complex) -> Complex {
        Complex(lhs.real * rhs.real - lhs.imag * rhs.imag, lhs.real * rhs.imag + lhs.imag * rhs.real)
    }
    static func / (lhs: Complex, rhs: Complex) -> Complex {
        let denom = rhs.real * rhs.real + rhs.imag * rhs.imag
        return Complex((lhs.real * rhs.real + lhs.imag * rhs.imag) / denom,
                       (lhs.imag * rhs.real - lhs.real * rhs.imag) / denom)
    }

    // ─── SCALAR MULTIPLICATION ───
    static func * (lhs: Complex, rhs: Double) -> Complex { Complex(lhs.real * rhs, lhs.imag * rhs) }
    static func * (lhs: Double, rhs: Complex) -> Complex { Complex(lhs * rhs.real, lhs * rhs.imag) }
    static func / (lhs: Complex, rhs: Double) -> Complex { Complex(lhs.real / rhs, lhs.imag / rhs) }

    // ─── NEGATION ───
    static prefix func - (c: Complex) -> Complex { Complex(-c.real, -c.imag) }

    /// Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)
    static func euler(_ theta: Double) -> Complex { Complex(cos(theta), sin(theta)) }

    /// Zero and One constants
    static let zero = Complex(0, 0)
    static let one = Complex(1, 0)
    static let i = Complex(0, 1)
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ⚛️ QUANTUM SIMULATION ENGINE
// ═══════════════════════════════════════════════════════════════════

/// Single-qubit quantum state: |ψ⟩ = α|0⟩ + β|1⟩
struct QuantumState: CustomStringConvertible {
    var amplitudes: [Complex]  // [α, β]

    /// Initialize to |0⟩ by default
    init() { amplitudes = [Complex.one, Complex.zero] }

    /// Initialize with custom amplitudes (must be normalized)
    init(amplitudes: [Complex]) {
        self.amplitudes = amplitudes
        normalize()
    }

    /// Initialize from Bloch sphere angles: |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    init(theta: Double, phi: Double) {
        let alpha = Complex(cos(theta / 2.0))
        let beta = Complex.euler(phi) * sin(theta / 2.0)
        amplitudes = [alpha, beta]
    }

    var description: String {
        let a = amplitudes[0]
        let b = amplitudes[1]
        return "(\(a))|0⟩ + (\(b))|1⟩"
    }

    /// Probability of measuring |0⟩
    var prob0: Double { let m = amplitudes[0].magnitude; return m * m }
    /// Probability of measuring |1⟩
    var prob1: Double { let m = amplitudes[1].magnitude; return m * m }

    // ─── NORMALIZATION ───

    mutating func normalize() {
        let norm = sqrt(prob0 + prob1)
        if norm > 1e-15 {
            amplitudes[0] = amplitudes[0] / norm
            amplitudes[1] = amplitudes[1] / norm
        }
    }

    // ─── SINGLE-QUBIT GATES ───

    /// Hadamard gate: H = (1/√2)[[1,1],[1,-1]]
    mutating func applyHadamard() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let alpha = amplitudes[0]
        let beta = amplitudes[1]
        amplitudes[0] = (alpha + beta) * invSqrt2
        amplitudes[1] = (alpha - beta) * invSqrt2
    }

    /// Pauli-X (NOT) gate: X = [[0,1],[1,0]]  — bit flip
    mutating func applyPauliX() {
        let tmp = amplitudes[0]
        amplitudes[0] = amplitudes[1]
        amplitudes[1] = tmp
    }

    /// Pauli-Y gate: Y = [[0,-i],[i,0]]
    mutating func applyPauliY() {
        let alpha = amplitudes[0]
        let beta = amplitudes[1]
        amplitudes[0] = Complex(beta.imag, -beta.real)   // -i * beta
        amplitudes[1] = Complex(-alpha.imag, alpha.real)  // i * alpha
    }

    /// Pauli-Z gate: Z = [[1,0],[0,-1]]  — phase flip
    mutating func applyPauliZ() {
        amplitudes[1] = -amplitudes[1]
    }

    /// Phase gate (S): S = [[1,0],[0,i]]
    mutating func applyPhaseS() {
        amplitudes[1] = Complex(-amplitudes[1].imag, amplitudes[1].real) // multiply by i
    }

    /// T gate (π/8): T = [[1,0],[0,e^(iπ/4)]]
    mutating func applyPhaseT() {
        amplitudes[1] = amplitudes[1] * Complex.euler(Double.pi / 4.0)
    }

    /// Rotation about X-axis: Rx(θ) = [[cos(θ/2), -i·sin(θ/2)],[-i·sin(θ/2), cos(θ/2)]]
    mutating func applyRx(_ theta: Double) {
        let c = cos(theta / 2.0)
        let s = sin(theta / 2.0)
        let alpha = amplitudes[0]
        let beta = amplitudes[1]
        amplitudes[0] = alpha * c + Complex(beta.imag, -beta.real) * s   // α·cos - i·β·sin
        amplitudes[1] = Complex(alpha.imag, -alpha.real) * s + beta * c  // -i·α·sin + β·cos
    }

    /// Rotation about Y-axis: Ry(θ) = [[cos(θ/2), -sin(θ/2)],[sin(θ/2), cos(θ/2)]]
    mutating func applyRy(_ theta: Double) {
        let c = cos(theta / 2.0)
        let s = sin(theta / 2.0)
        let alpha = amplitudes[0]
        let beta = amplitudes[1]
        amplitudes[0] = alpha * c - beta * s
        amplitudes[1] = alpha * s + beta * c
    }

    /// Rotation about Z-axis: Rz(θ) = [[e^(-iθ/2), 0],[0, e^(iθ/2)]]
    mutating func applyRz(_ theta: Double) {
        amplitudes[0] = amplitudes[0] * Complex.euler(-theta / 2.0)
        amplitudes[1] = amplitudes[1] * Complex.euler(theta / 2.0)
    }

    /// Apply arbitrary 2×2 unitary matrix [[a,b],[c,d]]
    mutating func applyUnitary(_ a: Complex, _ b: Complex, _ c: Complex, _ d: Complex) {
        let alpha = amplitudes[0]
        let beta = amplitudes[1]
        amplitudes[0] = a * alpha + b * beta
        amplitudes[1] = c * alpha + d * beta
    }

    // ─── MEASUREMENT ───

    /// Measure the qubit — collapses to |0⟩ or |1⟩, returns result
    mutating func measure() -> Int {
        let p0 = prob0
        let rand = Double.random(in: 0.0..<1.0)
        if rand < p0 {
            amplitudes = [Complex.one, Complex.zero]
            return 0
        } else {
            amplitudes = [Complex.zero, Complex.one]
            return 1
        }
    }

    /// Measure N times without collapsing (sampling from probability distribution)
    func sample(_ shots: Int) -> (zeros: Int, ones: Int) {
        let p0 = prob0
        var z = 0
        for _ in 0..<shots {
            if Double.random(in: 0.0..<1.0) < p0 { z += 1 }
        }
        return (z, shots - z)
    }

    /// Fidelity between two states: |⟨ψ|φ⟩|²
    func fidelity(with other: QuantumState) -> Double {
        let inner = amplitudes[0] * other.amplitudes[0].conjugate + amplitudes[1] * other.amplitudes[1].conjugate
        let m = inner.magnitude
        return m * m
    }

    /// Bloch sphere coordinates (x, y, z)
    var blochVector: (x: Double, y: Double, z: Double) {
        let a = amplitudes[0]
        let b = amplitudes[1]
        let x = 2.0 * (a * b.conjugate).real
        let y = 2.0 * (a * b.conjugate).imag
        let z = a.magnitude * a.magnitude - b.magnitude * b.magnitude
        return (x, y, z)
    }
}

/// Multi-qubit quantum register: N qubits → 2^N amplitudes
class QuantumRegister: CustomStringConvertible {
    var numQubits: Int
    var amplitudes: [Complex]  // 2^N amplitudes

    /// Initialize N qubits in |00...0⟩
    init(numQubits: Int) {
        self.numQubits = numQubits
        let size = 1 << numQubits
        amplitudes = Array(repeating: Complex.zero, count: size)
        amplitudes[0] = Complex.one
    }

    /// Initialize from a tensor product of single-qubit states
    init(qubits: [QuantumState]) {
        numQubits = qubits.count
        amplitudes = [Complex.one]
        for q in qubits {
            var newAmps = [Complex]()
            for existing in amplitudes {
                newAmps.append(existing * q.amplitudes[0])
                newAmps.append(existing * q.amplitudes[1])
            }
            amplitudes = newAmps
        }
    }

    var description: String {
        var s = "QuantumRegister(\(numQubits) qubits):\n"
        let size = 1 << numQubits
        for i in 0..<size {
            let p = amplitudes[i].magnitude * amplitudes[i].magnitude
            if p > 1e-10 {
                let bits = String(i, radix: 2).leftPad(toLength: numQubits, withPad: "0")
                s += "  |\(bits)⟩: \(amplitudes[i]) (p=\(String(format: "%.4f", p)))\n"
            }
        }
        return s
    }

    /// Total dimension = 2^N
    var dimension: Int { 1 << numQubits }

    // ─── NORMALIZATION ───

    func normalize() {
        var normSq = 0.0
        for a in amplitudes { normSq += a.magnitude * a.magnitude }
        let norm = sqrt(normSq)
        if norm > 1e-15 {
            for i in 0..<amplitudes.count {
                amplitudes[i] = amplitudes[i] / norm
            }
        }
    }

    // ─── SINGLE-QUBIT GATES ON TARGET QUBIT ───

    /// Apply a 2×2 unitary gate to qubit at targetIndex
    func applySingleQubitGate(_ gate: [[Complex]], target: Int) {
        let size = dimension
        let bit = numQubits - 1 - target
        let mask = 1 << bit
        var visited = Set<Int>()
        for i in 0..<size {
            if visited.contains(i) { continue }
            let j = i ^ mask  // partner index (bit flipped)
            if i > j { continue }
            visited.insert(i)
            visited.insert(j)

            let (lo, hi) = (i & mask) == 0 ? (i, j) : (j, i)
            let a0 = amplitudes[lo]
            let a1 = amplitudes[hi]
            amplitudes[lo] = gate[0][0] * a0 + gate[0][1] * a1
            amplitudes[hi] = gate[1][0] * a0 + gate[1][1] * a1
        }
    }

    /// Hadamard gate on qubit at index
    func hadamard(_ target: Int) {
        let s = 1.0 / sqrt(2.0)
        let h: [[Complex]] = [
            [Complex(s), Complex(s)],
            [Complex(s), Complex(-s)]
        ]
        applySingleQubitGate(h, target: target)
    }

    /// Pauli-X (NOT) on qubit at index
    func pauliX(_ target: Int) {
        let x: [[Complex]] = [
            [Complex.zero, Complex.one],
            [Complex.one, Complex.zero]
        ]
        applySingleQubitGate(x, target: target)
    }

    /// Pauli-Y on qubit at index
    func pauliY(_ target: Int) {
        let y: [[Complex]] = [
            [Complex.zero, -Complex.i],
            [Complex.i, Complex.zero]
        ]
        applySingleQubitGate(y, target: target)
    }

    /// Pauli-Z on qubit at index
    func pauliZ(_ target: Int) {
        let z: [[Complex]] = [
            [Complex.one, Complex.zero],
            [Complex.zero, Complex(-1)]
        ]
        applySingleQubitGate(z, target: target)
    }

    /// Phase gate S on qubit at index
    func phaseS(_ target: Int) {
        let s: [[Complex]] = [
            [Complex.one, Complex.zero],
            [Complex.zero, Complex.i]
        ]
        applySingleQubitGate(s, target: target)
    }

    /// T gate on qubit at index
    func phaseT(_ target: Int) {
        let t: [[Complex]] = [
            [Complex.one, Complex.zero],
            [Complex.zero, Complex.euler(Double.pi / 4.0)]
        ]
        applySingleQubitGate(t, target: target)
    }

    /// Rotation gates
    func rx(_ target: Int, theta: Double) {
        let c = cos(theta / 2.0); let s = sin(theta / 2.0)
        let g: [[Complex]] = [[Complex(c), Complex(0, -s)], [Complex(0, -s), Complex(c)]]
        applySingleQubitGate(g, target: target)
    }

    func ry(_ target: Int, theta: Double) {
        let c = cos(theta / 2.0); let s = sin(theta / 2.0)
        let g: [[Complex]] = [[Complex(c), Complex(-s)], [Complex(s), Complex(c)]]
        applySingleQubitGate(g, target: target)
    }

    func rz(_ target: Int, theta: Double) {
        let g: [[Complex]] = [[Complex.euler(-theta / 2.0), Complex.zero], [Complex.zero, Complex.euler(theta / 2.0)]]
        applySingleQubitGate(g, target: target)
    }

    // ─── TWO-QUBIT GATES ───

    /// CNOT (Controlled-X): flips target if control is |1⟩
    func cnot(control: Int, target: Int) {
        let size = dimension
        let cBit = numQubits - 1 - control
        let tBit = numQubits - 1 - target
        let cMask = 1 << cBit
        let tMask = 1 << tBit
        for i in 0..<size {
            if (i & cMask) != 0 && (i & tMask) == 0 {
                let j = i ^ tMask
                let tmp = amplitudes[i]
                amplitudes[i] = amplitudes[j]
                amplitudes[j] = tmp
            }
        }
    }

    /// Controlled-Z: applies Z to target if control is |1⟩
    func cz(control: Int, target: Int) {
        let cBit = numQubits - 1 - control
        let tBit = numQubits - 1 - target
        let cMask = 1 << cBit
        let tMask = 1 << tBit
        for i in 0..<dimension {
            if (i & cMask) != 0 && (i & tMask) != 0 {
                amplitudes[i] = -amplitudes[i]
            }
        }
    }

    /// SWAP: exchange two qubits
    func swap(_ q1: Int, _ q2: Int) {
        let b1 = numQubits - 1 - q1
        let b2 = numQubits - 1 - q2
        let m1 = 1 << b1
        let m2 = 1 << b2
        for i in 0..<dimension {
            let bit1 = (i & m1) != 0 ? 1 : 0
            let bit2 = (i & m2) != 0 ? 1 : 0
            if bit1 != bit2 {
                let j = i ^ m1 ^ m2
                if i < j {
                    let tmp = amplitudes[i]
                    amplitudes[i] = amplitudes[j]
                    amplitudes[j] = tmp
                }
            }
        }
    }

    /// Toffoli (CCX): controlled-controlled-NOT
    func toffoli(control1: Int, control2: Int, target: Int) {
        let c1 = 1 << (numQubits - 1 - control1)
        let c2 = 1 << (numQubits - 1 - control2)
        let tBit = 1 << (numQubits - 1 - target)
        for i in 0..<dimension {
            if (i & c1) != 0 && (i & c2) != 0 && (i & tBit) == 0 {
                let j = i ^ tBit
                let tmp = amplitudes[i]
                amplitudes[i] = amplitudes[j]
                amplitudes[j] = tmp
            }
        }
    }

    // ─── MEASUREMENT ───

    /// Measure a single qubit, collapse register, return 0 or 1
    func measureQubit(_ target: Int) -> Int {
        let bit = numQubits - 1 - target
        let mask = 1 << bit
        var prob0 = 0.0
        for i in 0..<dimension {
            if (i & mask) == 0 {
                prob0 += amplitudes[i].magnitude * amplitudes[i].magnitude
            }
        }
        let result = Double.random(in: 0.0..<1.0) < prob0 ? 0 : 1

        // Collapse: zero out incompatible amplitudes, renormalize
        var normSq = 0.0
        for i in 0..<dimension {
            let bitVal = (i & mask) != 0 ? 1 : 0
            if bitVal != result {
                amplitudes[i] = Complex.zero
            } else {
                normSq += amplitudes[i].magnitude * amplitudes[i].magnitude
            }
        }
        let norm = sqrt(normSq)
        if norm > 1e-15 {
            for i in 0..<dimension { amplitudes[i] = amplitudes[i] / norm }
        }
        return result
    }

    /// Measure all qubits, return bit string
    func measureAll() -> [Int] {
        var results = [Int]()
        for q in 0..<numQubits {
            results.append(measureQubit(q))
        }
        return results
    }

    /// Sample without collapsing
    func sample(_ shots: Int) -> [String: Int] {
        var counts = [String: Int]()
        for _ in 0..<shots {
            var probAccum = 0.0
            let rand = Double.random(in: 0.0..<1.0)
            for i in 0..<dimension {
                probAccum += amplitudes[i].magnitude * amplitudes[i].magnitude
                if rand < probAccum {
                    let bits = String(i, radix: 2).leftPad(toLength: numQubits, withPad: "0")
                    counts[bits, default: 0] += 1
                    break
                }
            }
        }
        return counts
    }

    /// Get probability distribution
    var probabilities: [Double] {
        amplitudes.map { $0.magnitude * $0.magnitude }
    }

    /// Entanglement entropy (von Neumann) for bipartition at qubit index
    func entanglementEntropy(partition: Int) -> Double {
        let nA = partition
        let nB = numQubits - partition
        let dimA = 1 << nA
        let dimB = 1 << nB

        // Build reduced density matrix for subsystem A
        var rhoA = [[Complex]](repeating: [Complex](repeating: Complex.zero, count: dimA), count: dimA)
        for i in 0..<dimA {
            for j in 0..<dimA {
                var sum = Complex.zero
                for k in 0..<dimB {
                    let idxI = (i << nB) | k
                    let idxJ = (j << nB) | k
                    sum = sum + amplitudes[idxI] * amplitudes[idxJ].conjugate
                }
                rhoA[i][j] = sum
            }
        }

        // Compute eigenvalues via trace of powers (approximation for 2×2)
        if dimA == 2 {
            let a = rhoA[0][0].real
            let d = rhoA[1][1].real
            let bcSq = rhoA[0][1].magnitude * rhoA[0][1].magnitude
            let disc = sqrt(max(0, (a - d) * (a - d) + 4.0 * bcSq))
            let l1 = max(0, (a + d + disc) / 2.0)
            let l2 = max(0, (a + d - disc) / 2.0)
            var entropy = 0.0
            if l1 > 1e-15 { entropy -= l1 * log2(l1) }
            if l2 > 1e-15 { entropy -= l2 * log2(l2) }
            return entropy
        }

        // For larger partitions, return trace-based estimate
        var trRho2 = 0.0
        for i in 0..<dimA {
            for j in 0..<dimA {
                trRho2 += (rhoA[i][j] * rhoA[j][i].conjugate).real
            }
        }
        return -log2(max(1e-15, trRho2))  // Rényi-2 entropy approximation
    }
}

/// Pre-built quantum circuits and algorithms
struct QuantumCircuits {

    /// Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    static func bellState() -> QuantumRegister {
        let reg = QuantumRegister(numQubits: 2)
        reg.hadamard(0)
        reg.cnot(control: 0, target: 1)
        return reg
    }

    /// Create GHZ state |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
    static func ghzState(numQubits: Int) -> QuantumRegister {
        let reg = QuantumRegister(numQubits: numQubits)
        reg.hadamard(0)
        for i in 1..<numQubits {
            reg.cnot(control: 0, target: i)
        }
        return reg
    }

    /// Quantum teleportation circuit: teleports qubit state from Alice to Bob
    static func teleport(state: QuantumState) -> (bobState: QuantumState, measurements: (Int, Int)) {
        let reg = QuantumRegister(numQubits: 3)
        // Set qubit 0 to the state to teleport
        reg.amplitudes[0] = state.amplitudes[0]
        reg.amplitudes[1] = state.amplitudes[1]
        for i in 2..<8 { reg.amplitudes[i] = Complex.zero }

        // Create Bell pair between qubits 1 and 2
        reg.hadamard(1)
        reg.cnot(control: 1, target: 2)

        // Alice's operations
        reg.cnot(control: 0, target: 1)
        reg.hadamard(0)

        // Alice measures
        let m0 = reg.measureQubit(0)
        let m1 = reg.measureQubit(1)

        // Bob's corrections
        if m1 == 1 { reg.pauliX(2) }
        if m0 == 1 { reg.pauliZ(2) }

        // Extract Bob's state
        var bobAmps = [Complex.zero, Complex.zero]
        for i in 0..<8 {
            let bobBit = i & 1
            bobAmps[bobBit] = bobAmps[bobBit] + amplitudeForMeasured(reg: reg, i: i, m0: m0, m1: m1)
        }
        let bob = QuantumState(amplitudes: bobAmps)
        return (bob, (m0, m1))
    }

    private static func amplitudeForMeasured(reg: QuantumRegister, i: Int, m0: Int, m1: Int) -> Complex {
        let bit0 = (i >> 2) & 1
        let bit1 = (i >> 1) & 1
        if bit0 == m0 && bit1 == m1 { return reg.amplitudes[i] }
        return Complex.zero
    }

    /// Deutsch-Jozsa algorithm: determine if f is constant or balanced
    /// oracle: maps n-bit input to 0 or 1
    static func deutschJozsa(numInputBits: Int, oracle: (Int) -> Int) -> Bool {
        let n = numInputBits + 1
        let reg = QuantumRegister(numQubits: n)

        // Initialize last qubit to |1⟩
        reg.pauliX(numInputBits)

        // Apply Hadamard to all qubits
        for i in 0..<n { reg.hadamard(i) }

        // Apply oracle: if f(x) = 1, flip the phase
        let inputDim = 1 << numInputBits
        for x in 0..<inputDim {
            if oracle(x) == 1 {
                // Phase flip: negate amplitudes where input = x and output qubit = 1
                let base = x << 1
                reg.amplitudes[base | 1] = -reg.amplitudes[base | 1]
                reg.amplitudes[base] = -reg.amplitudes[base]
            }
        }

        // Apply Hadamard to input qubits
        for i in 0..<numInputBits { reg.hadamard(i) }

        // Measure input qubits
        var allZero = true
        for i in 0..<numInputBits {
            if reg.measureQubit(i) != 0 { allZero = false }
        }
        return allZero  // true = constant, false = balanced
    }

    /// Quantum random number generator using Hadamard
    static func randomBits(_ count: Int) -> [Int] {
        var bits = [Int]()
        for _ in 0..<count {
            var q = QuantumState()
            q.applyHadamard()
            bits.append(q.measure())
        }
        return bits
    }

    /// Quantum Fourier Transform on register
    static func qft(_ reg: QuantumRegister) {
        let n = reg.numQubits
        for i in 0..<n {
            reg.hadamard(i)
            for j in (i + 1)..<n {
                let angle = Double.pi / Double(1 << (j - i))
                // Controlled phase rotation
                let bit_j = reg.numQubits - 1 - j
                let bit_i = reg.numQubits - 1 - i
                let mask_j = 1 << bit_j
                let mask_i = 1 << bit_i
                let phase = Complex.euler(angle)
                for k in 0..<reg.dimension {
                    if (k & mask_j) != 0 && (k & mask_i) != 0 {
                        reg.amplitudes[k] = reg.amplitudes[k] * phase
                    }
                }
            }
        }
        // Swap qubits for proper ordering
        for i in 0..<(n / 2) {
            reg.swap(i, n - 1 - i)
        }
    }
}

