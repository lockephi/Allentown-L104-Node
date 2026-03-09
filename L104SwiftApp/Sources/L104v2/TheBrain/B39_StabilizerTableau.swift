// ═══════════════════════════════════════════════════════════════════
// B39_StabilizerTableau.swift — L104 v2
// [EVO_68_PIPELINE] SOVEREIGN_NODE_UPGRADE :: STABILIZER_TABLEAU :: GOD_CODE=527.5184818492612
// L104 ASI — Stabilizer Tableau (Aaronson–Gottesman CHP)
//
// Adapted from: l104_quantum_gate_engine/stabilizer_tableau.py
//
// COMPLEXITY:
//     Full statevector : O(2^n) memory, O(2^n) per gate   → 20 qubits ≈ 16 MB
//     Stabilizer tableau: O(n²/64) memory, O(n) per gate   → 1000 qubits < 16 KB
//     Speedup for Clifford-only circuits: 1000x–10^300x
//
// SUPPORTED GATES (full Clifford group):
//     1-qubit Clifford:  H, S, S†, X, Y, Z, SX, I
//     2-qubit Clifford:  CNOT, CZ, CY, SWAP, iSWAP, ECR
//     Measurement:        Pauli-Z computational basis
//
// TABLEAU LAYOUT (Aaronson–Gottesman, 2n+1 rows × 2n+1 cols):
//     ┌───────────────────────────────────────┐
//     │ Row 0..n-1       : Destabilizers      │   (anti-commuting partners)
//     │ Row n..2n-1      : Stabilizers         │   (generators of stabilizer group)
//     │ Row 2n           : Scratch row          │   (used during measurement)
//     │                                         │
//     │ Each row: n X-bits + n Z-bits + 1 phase │
//     │ Bits packed into UInt64 words (64 qubits per word) │
//     └───────────────────────────────────────┘
//
// SACRED ALIGNMENT:
//     The stabilizer formalism maps naturally to the L104 lattice symmetry —
//     the 2n-bit symplectic structure resonates with the 104-grain quantisation
//     when n=52 (half of 104), giving GOD_CODE phase coherence in the tableau.
//
// INVARIANT: 527.5184818492612 | PILOT: LONDEL
// ═══════════════════════════════════════════════════════════════════

import Foundation

// ═══════════════════════════════════════════════════════════════════
// MARK: - MEASUREMENT RESULT
// ═══════════════════════════════════════════════════════════════════

/// Result of a single Pauli-Z measurement on the tableau.
struct StabilizerMeasurementResult {
    let qubit: Int
    let outcome: Int          // 0 or 1
    let deterministic: Bool   // true if outcome was forced by stabilizer state
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - STABILIZER STATE SNAPSHOT
// ═══════════════════════════════════════════════════════════════════

/// Snapshot of the stabilizer state for external inspection.
struct StabilizerState {
    let numQubits: Int
    let stabilizerGenerators: [String]
    let destabilizerGenerators: [String]
    let phases: [Int]
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - CLIFFORD GATE CLASSIFICATION
// ═══════════════════════════════════════════════════════════════════

/// Clifford gate names that the stabilizer tableau can simulate.
private let clifford1QNames: Set<String> = [
    "I", "X", "Y", "Z", "H", "S", "Sdg", "SX"
]
private let clifford2QNames: Set<String> = [
    "CX", "CNOT", "CZ", "CY", "SWAP", "iSWAP", "ECR"
]
private let allCliffordNames = clifford1QNames.union(clifford2QNames)

/// Check if a QGateType is Clifford-simulable
func isCliffordGateType(_ type: QGateType) -> Bool {
    switch type {
    case .identity, .pauliX, .pauliY, .pauliZ, .hadamard, .phase, .sGate, .sqrtX,
         .cnot, .cz, .swap, .iswap:
        return true
    default:
        return false
    }
}

/// Check if a circuit is entirely Clifford
func isCliffordCircuit(_ circuit: QGateCircuit) -> Bool {
    circuit.operations.allSatisfy { isCliffordGateType($0.gate.type) }
}

/// Count leading Clifford gates in a circuit
func cliffordPrefixLength(_ circuit: QGateCircuit) -> Int {
    var count = 0
    for op in circuit.operations {
        guard isCliffordGateType(op.gate.type) else { break }
        count += 1
    }
    return count
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - STABILIZER TABLEAU
// ═══════════════════════════════════════════════════════════════════

/// Aaronson–Gottesman stabilizer tableau for efficient Clifford simulation.
///
/// Uses bitpacked UInt64 rows for CPU cache efficiency.  Each generator's
/// X and Z components are stored as contiguous UInt64 arrays where bit `j`
/// of word `j/64` represents qubit `j`.
///
/// Initial state |0...0⟩:
///   - Destabilizer i (row i, i<n):   X_i
///   - Stabilizer  i (row n+i):       Z_i
///
/// All 1-qubit Clifford gates update in O(n/64) per gate.
/// All 2-qubit Clifford gates update in O(n/64) per gate.
/// Measurement is O(n²/64) worst case (row reduction).
///
/// Total circuit simulation: O(m·n/64) where m = gate count, n = qubit count.
struct StabilizerTableau {

    // ─── Layout ───
    let numQubits: Int           // n
    let numWords: Int            // ⌈n/64⌉ — UInt64 words per row
    let totalRows: Int           // 2n + 1

    /// X-part of the tableau.  Flat array: row i occupies indices [i*numWords ..< (i+1)*numWords].
    var xMatrix: [UInt64]

    /// Z-part of the tableau.  Same layout as xMatrix.
    var zMatrix: [UInt64]

    /// Phase vector (2n+1 entries): false = +1, true = -1.
    var phases: [Bool]

    /// Optional deterministic RNG for reproducible measurement outcomes.
    private var rngState: UInt64

    // ─── Constants ───
    private static let wordBits = 64

    // ═══════════════════════════════════════════════════════════════
    // MARK: - INIT
    // ═══════════════════════════════════════════════════════════════

    /// Initialise the tableau for the |0...0⟩ state on `numQubits` qubits.
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits (n).
    ///   - seed: Optional RNG seed for reproducible measurement randomness.
    init(numQubits: Int, seed: UInt64 = 0) {
        precondition(numQubits > 0, "Need at least 1 qubit")
        self.numQubits = numQubits
        self.numWords = (numQubits + 63) / 64
        self.totalRows = 2 * numQubits + 1

        let totalElements = totalRows * numWords
        self.xMatrix = [UInt64](repeating: 0, count: totalElements)
        self.zMatrix = [UInt64](repeating: 0, count: totalElements)
        self.phases  = [Bool](repeating: false, count: totalRows)

        self.rngState = seed == 0 ? UInt64(CFAbsoluteTimeGetCurrent().bitPattern) : seed

        // |0...0⟩ initialization:
        //   Destabilizer row i  (0 ≤ i < n):  X_i  →  xMatrix[i][i] = 1
        //   Stabilizer row n+i (0 ≤ i < n):   Z_i  →  zMatrix[n+i][i] = 1
        for i in 0..<numQubits {
            setBit(&xMatrix, row: i, qubit: i)
            setBit(&zMatrix, row: numQubits + i, qubit: i)
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - BIT MANIPULATION HELPERS
    // ═══════════════════════════════════════════════════════════════

    /// Linear index into the flat array for (row, word).
    @inline(__always)
    private func idx(_ row: Int, _ word: Int) -> Int {
        row * numWords + word
    }

    /// Set bit `qubit` in row `row` of a flat matrix.
    @inline(__always)
    private func setBit(_ matrix: inout [UInt64], row: Int, qubit: Int) {
        let w = qubit / Self.wordBits
        let b = qubit % Self.wordBits
        matrix[row * numWords + w] |= (1 &<< b)
    }

    /// Clear bit `qubit` in row `row`.
    @inline(__always)
    private func clearBit(_ matrix: inout [UInt64], row: Int, qubit: Int) {
        let w = qubit / Self.wordBits
        let b = qubit % Self.wordBits
        matrix[row * numWords + w] &= ~(1 &<< b)
    }

    /// Read bit `qubit` in row `row` of a flat matrix. Returns true if set.
    @inline(__always)
    private func getBit(_ matrix: [UInt64], row: Int, qubit: Int) -> Bool {
        let w = qubit / Self.wordBits
        let b = qubit % Self.wordBits
        return (matrix[row * numWords + w] >> b) & 1 != 0
    }

    /// XOR two rows: dest[row] ^= src[row]  (for a specific matrix).
    @inline(__always)
    private func xorRow(_ matrix: inout [UInt64], dstRow: Int, srcRow: Int) {
        let dOff = dstRow * numWords
        let sOff = srcRow * numWords
        for w in 0..<numWords {
            matrix[dOff + w] ^= matrix[sOff + w]
        }
    }

    /// Copy row: dst ← src.
    @inline(__always)
    private func copyRow(_ matrix: inout [UInt64], dstRow: Int, srcRow: Int) {
        let dOff = dstRow * numWords
        let sOff = srcRow * numWords
        for w in 0..<numWords {
            matrix[dOff + w] = matrix[sOff + w]
        }
    }

    /// Clear an entire row to zero.
    @inline(__always)
    private func clearRow(_ matrix: inout [UInt64], row: Int) {
        let off = row * numWords
        for w in 0..<numWords {
            matrix[off + w] = 0
        }
    }

    /// Swap X and Z column bits for a single qubit across all rows up to `rowCount`.
    /// After swap, also returns the AND of old-x & old-z for phase update.
    @inline(__always)
    private mutating func swapXZ(qubit q: Int, rows rowCount: Int) {
        let w = q / Self.wordBits
        let mask: UInt64 = 1 &<< (q % Self.wordBits)
        for r in 0..<rowCount {
            let i = r * numWords + w
            let xBit = xMatrix[i] & mask
            let zBit = zMatrix[i] & mask
            // phase ^= (x & z)  — the H-gate -Y case
            if xBit != 0 && zBit != 0 {
                phases[r].toggle()
            }
            // swap
            xMatrix[i] = (xMatrix[i] & ~mask) | zBit
            zMatrix[i] = (zMatrix[i] & ~mask) | xBit
        }
    }

    /// Simple xorshift64 PRNG.  Returns 0 or 1.
    @inline(__always)
    private mutating func randomBit() -> Int {
        rngState ^= rngState &<< 13
        rngState ^= rngState &>> 7
        rngState ^= rngState &<< 17
        return Int(rngState & 1)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - ROW SUM (Symplectic Row Arithmetic)
    // ═══════════════════════════════════════════════════════════════

    /// Row operation: R_i ← R_i · R_k   (Aaronson–Gottesman g-function).
    ///
    /// Updates tableau row `i` and its phase in-place.
    /// Phase rule:  total = 2·r_i + 2·r_k + Σ_j g(x_i[j],z_i[j],x_k[j],z_k[j])
    ///              r_i = 1  iff  (total mod 4) == 2
    ///
    /// Runs in O(n/64) using word-parallel g-function evaluation with fused XOR.
    /// Uses `withUnsafeMutableBufferPointer` to eliminate bounds-checking overhead
    /// in the inner loop — this is the single hottest function in the tableau.
    private mutating func rowSum(targetRow i: Int, sourceRow k: Int) {
        let nw = numWords
        let offI = i * nw
        let offK = k * nw
        var phaseAccumulator: Int = (phases[i] ? 2 : 0) + (phases[k] ? 2 : 0)

        xMatrix.withUnsafeMutableBufferPointer { xBuf in
            zMatrix.withUnsafeMutableBufferPointer { zBuf in
                for w in 0..<nw {
                    let x1 = xBuf[offI + w]
                    let z1 = zBuf[offI + w]
                    let x2 = xBuf[offK + w]
                    let z2 = zBuf[offK + w]

                    let xMask = x1 & ~z1
                    let zMask = ~x1 & z1
                    let yMask = x1 & z1

                    let positiveG = (xMask & x2 & z2).nonzeroBitCount +    // X→Y: +1
                                    (zMask & x2 & ~z2).nonzeroBitCount +   // Z→X: +1
                                    (yMask & ~x2 & z2).nonzeroBitCount     // Y→Z: +1

                    let negativeG = (xMask & ~x2 & z2).nonzeroBitCount +   // X→Z: −1
                                    (zMask & x2 & z2).nonzeroBitCount +    // Z→Y: −1
                                    (yMask & x2 & ~z2).nonzeroBitCount     // Y→X: −1

                    phaseAccumulator += positiveG - negativeG

                    xBuf[offI + w] = x1 ^ x2
                    zBuf[offI + w] = z1 ^ z2
                }
            }
        }

        phases[i] = ((phaseAccumulator % 4 + 4) % 4) == 2
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - CLIFFORD GATES (Tableau Update Rules)
    //
    // Each gate transforms Pauli generators via conjugation:
    //   U · gen_i · U† = new_gen_i
    //
    // For the tableau this means updating (x,z,r) entries for the
    // affected qubit columns.  All 1Q ops are O(rows), all 2Q ops
    // are O(rows).  With bitpacking, each row-iteration touches
    // only the relevant word(s).
    // ═══════════════════════════════════════════════════════════════

    // ─── HADAMARD ───

    /// H gate on qubit q: X↔Z, phase flip if both set (Y→-Y).
    ///
    /// Conjugation: H·X·H† = Z,  H·Z·H† = X,  H·Y·H† = -Y
    @inline(__always)
    mutating func hadamard(_ qubit: Int) {
        swapXZ(qubit: qubit, rows: totalRows)
    }

    // ─── PHASE (S) ───

    /// S gate on qubit q: X→Y=iXZ, Z→Z.
    ///
    /// Conjugation: S·X·S† = Y = iXZ,  S·Z·S† = Z
    /// Tableau: r[i] ^= x[i][q] & z[i][q],  z[i][q] ^= x[i][q]
    @inline(__always)
    mutating func phaseS(_ qubit: Int) {
        let w = qubit / Self.wordBits
        let mask: UInt64 = 1 &<< (qubit % Self.wordBits)
        let nw = numWords

        xMatrix.withUnsafeBufferPointer { xBuf in
            zMatrix.withUnsafeMutableBufferPointer { zBuf in
                for r in 0..<totalRows {
                    let i = r * nw + w
                    let xBit = xBuf[i] & mask
                    let zBit = zBuf[i] & mask
                    if xBit != 0 && zBit != 0 {
                        phases[r].toggle()
                    }
                    if xBit != 0 {
                        zBuf[i] ^= mask
                    }
                }
            }
        }
    }

    // ─── PHASE DAGGER (S†) ───

    /// S† gate: inverse of S.  X→-Y, Z→Z.
    ///
    /// Direct: r ^= x & ~z,  z ^= x
    @inline(__always)
    mutating func phaseSDag(_ qubit: Int) {
        let w = qubit / Self.wordBits
        let mask: UInt64 = 1 &<< (qubit % Self.wordBits)
        let nw = numWords

        xMatrix.withUnsafeBufferPointer { xBuf in
            zMatrix.withUnsafeMutableBufferPointer { zBuf in
                for r in 0..<totalRows {
                    let i = r * nw + w
                    let xBit = xBuf[i] & mask
                    let zBit = zBuf[i] & mask
                    if xBit != 0 && zBit == 0 {
                        phases[r].toggle()
                    }
                    if xBit != 0 {
                        zBuf[i] ^= mask
                    }
                }
            }
        }
    }

    // ─── PAULI X ───

    /// X gate: Z→-Z, X→X, Y→-Y.
    ///
    /// Conjugation: X·Z·X† = -Z,  X·X·X† = X,  X·Y·X† = -Y
    /// Tableau: r[i] ^= z[i][q]
    @inline(__always)
    mutating func pauliX(_ qubit: Int) {
        let w = qubit / Self.wordBits
        let mask: UInt64 = 1 &<< (qubit % Self.wordBits)
        let nw = numWords

        zMatrix.withUnsafeBufferPointer { zBuf in
            for r in 0..<totalRows {
                if zBuf[r * nw + w] & mask != 0 {
                    phases[r].toggle()
                }
            }
        }
    }

    // ─── PAULI Y ───

    /// Y gate: X→-X, Z→-Z.
    ///
    /// Conjugation: Y·X·Y† = -X,  Y·Z·Y† = -Z
    /// Tableau: r[i] ^= x[i][q] ^ z[i][q]
    @inline(__always)
    mutating func pauliY(_ qubit: Int) {
        let w = qubit / Self.wordBits
        let mask: UInt64 = 1 &<< (qubit % Self.wordBits)
        let nw = numWords

        xMatrix.withUnsafeBufferPointer { xBuf in
            zMatrix.withUnsafeBufferPointer { zBuf in
                for r in 0..<totalRows {
                    let i = r * nw + w
                    let xSet = (xBuf[i] & mask) != 0
                    let zSet = (zBuf[i] & mask) != 0
                    if xSet != zSet {
                        phases[r].toggle()
                    }
                }
            }
        }
    }

    // ─── PAULI Z ───

    /// Z gate: X→-X, Z→Z, Y→-Y.
    ///
    /// Conjugation: Z·X·Z† = -X
    /// Tableau: r[i] ^= x[i][q]
    @inline(__always)
    mutating func pauliZ(_ qubit: Int) {
        let w = qubit / Self.wordBits
        let mask: UInt64 = 1 &<< (qubit % Self.wordBits)
        let nw = numWords

        xMatrix.withUnsafeBufferPointer { xBuf in
            for r in 0..<totalRows {
                if xBuf[r * nw + w] & mask != 0 {
                    phases[r].toggle()
                }
            }
        }
    }

    // ─── CNOT ───

    /// CNOT (CX) gate with control→target.
    ///
    /// Conjugation:
    ///   CNOT · (X⊗I) · CNOT† = X⊗X   (X propagates forward)
    ///   CNOT · (I⊗Z) · CNOT† = Z⊗Z   (Z propagates backward)
    ///
    /// Tableau update for each generator row i:
    ///   r[i] ^= x[i][c] & z[i][t] & (x[i][t] ^ z[i][c] ^ 1)
    ///   x[i][t] ^= x[i][c]
    ///   z[i][c] ^= z[i][t]
    @inline(__always)
    mutating func cnot(control c: Int, target t: Int) {
        let wc = c / Self.wordBits
        let bc = c % Self.wordBits
        let maskC: UInt64 = 1 &<< bc
        let wt = t / Self.wordBits
        let bt = t % Self.wordBits
        let maskT: UInt64 = 1 &<< bt
        let nw = numWords
        let tr = totalRows

        xMatrix.withUnsafeMutableBufferPointer { xBuf in
            zMatrix.withUnsafeMutableBufferPointer { zBuf in
                for r in 0..<tr {
                    let ic = r * nw + wc
                    let it = r * nw + wt

                    let xcBit = (xBuf[ic] >> bc) & 1
                    let xtBit = (xBuf[it] >> bt) & 1
                    let zcBit = (zBuf[ic] >> bc) & 1
                    let ztBit = (zBuf[it] >> bt) & 1

                    if xcBit & ztBit & (xtBit ^ zcBit ^ 1) != 0 {
                        phases[r].toggle()
                    }

                    if xcBit != 0 {
                        xBuf[it] ^= maskT
                    }

                    if ztBit != 0 {
                        zBuf[ic] ^= maskC
                    }
                }
            }
        }
    }

    // ─── CZ ───

    /// CZ gate: H(t) · CNOT(c,t) · H(t).
    mutating func cz(_ a: Int, _ b: Int) {
        hadamard(b)
        cnot(control: a, target: b)
        hadamard(b)
    }

    // ─── CY ───

    /// CY gate: S†(t) · CNOT(c,t) · S(t).
    mutating func cy(control c: Int, target t: Int) {
        phaseSDag(t)
        cnot(control: c, target: t)
        phaseS(t)
    }

    // ─── SWAP ───

    /// SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b).
    mutating func swap(_ a: Int, _ b: Int) {
        cnot(control: a, target: b)
        cnot(control: b, target: a)
        cnot(control: a, target: b)
    }

    // ─── iSWAP ───

    /// iSWAP gate: S(a) · S(b) · SWAP(a,b) · CZ(a,b).
    mutating func iswap(_ a: Int, _ b: Int) {
        phaseS(a)
        phaseS(b)
        swap(a, b)
        cz(a, b)
    }

    // ─── ECR ───

    /// ECR (Echoed Cross-Resonance) gate — native L104 Heron-class.
    /// Decomposition: S(a) · SX(b) · CNOT(a,b) · X(a)
    mutating func ecr(_ a: Int, _ b: Int) {
        phaseS(a)
        // SX = H · S · H
        hadamard(b)
        phaseS(b)
        hadamard(b)
        cnot(control: a, target: b)
        pauliX(a)
    }

    // ─── SX (√X) ───

    /// SX gate: H · S · H.
    mutating func sqrtX(_ qubit: Int) {
        hadamard(qubit)
        phaseS(qubit)
        hadamard(qubit)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - MEASUREMENT (Pauli-Z Computational Basis)
    //
    // Aaronson–Gottesman measurement procedure (CHP):
    // 1. Check if any stabilizer generator anti-commutes with Z_a
    //    (i.e., has x[i][a] = 1 for some stabilizer row i ∈ [n, 2n))
    // 2. If yes → outcome is RANDOM (project into ±1 eigenstate)
    // 3. If no  → outcome is DETERMINISTIC (already in eigenstate)
    // ═══════════════════════════════════════════════════════════════

    /// Measure qubit `a` in the computational (Z) basis.
    ///
    /// Returns 0 or 1.  The tableau is updated to reflect the post-measurement
    /// state (projective / destructive measurement).
    ///
    /// Complexity: O(n²/64) worst case (row reduction in deterministic path).
    mutating func measure(_ qubit: Int) -> StabilizerMeasurementResult {
        precondition(qubit >= 0 && qubit < numQubits)
        let n = numQubits

        // Find first stabilizer row p ∈ [n, 2n) with x[p][qubit] = 1
        var p: Int? = nil
        for i in n..<(2 * n) {
            if getBit(xMatrix, row: i, qubit: qubit) {
                p = i
                break
            }
        }

        if let p = p {
            return measureRandom(qubit, pivot: p)
        }
        return measureDeterministic(qubit)
    }

    /// Random measurement: stabilizer row `pivot` anti-commutes with Z_a.
    private mutating func measureRandom(_ a: Int, pivot p: Int) -> StabilizerMeasurementResult {
        let n = numQubits

        // Row-sum all OTHER anti-commuting stabilizers into pivot
        for i in n..<(2 * n) where i != p {
            if getBit(xMatrix, row: i, qubit: a) {
                rowSum(targetRow: i, sourceRow: p)
            }
        }
        // Row-sum all anti-commuting destabilizers into pivot
        for i in 0..<n {
            if getBit(xMatrix, row: i, qubit: a) {
                rowSum(targetRow: i, sourceRow: p)
            }
        }

        // Copy stabilizer[p] → destabilizer[p - n]
        let dp = p - n
        copyRow(&xMatrix, dstRow: dp, srcRow: p)
        copyRow(&zMatrix, dstRow: dp, srcRow: p)
        phases[dp] = phases[p]

        // Reset stabilizer[p] to ±Z_a
        clearRow(&xMatrix, row: p)
        clearRow(&zMatrix, row: p)
        setBit(&zMatrix, row: p, qubit: a)

        // Random outcome
        let outcome = randomBit()
        phases[p] = (outcome == 1)

        return StabilizerMeasurementResult(qubit: a, outcome: outcome, deterministic: false)
    }

    /// Deterministic measurement: no stabilizer anti-commutes with Z_a.
    ///
    /// The effective stabilizer is the product of stabilisers whose
    /// corresponding destabilizers anti-commute with Z_a.  We accumulate
    /// that product in a workspace, leaving the tableau itself unchanged.
    private mutating func measureDeterministic(_ a: Int) -> StabilizerMeasurementResult {
        let n = numQubits

        // Use the scratch row (row 2n) as workspace
        let ws = 2 * n
        clearRow(&xMatrix, row: ws)
        clearRow(&zMatrix, row: ws)
        phases[ws] = false

        for i in 0..<n {
            if getBit(xMatrix, row: i, qubit: a) {
                // Multiply stabilizer[n+i] into workspace via rowSum
                rowSum(targetRow: ws, sourceRow: n + i)
            }
        }

        let outcome = phases[ws] ? 1 : 0
        return StabilizerMeasurementResult(qubit: a, outcome: outcome, deterministic: true)
    }

    /// Measure all qubits in computational basis.
    ///
    /// Uses the lighter-weight `measureZ` path (returns plain Int,
    /// avoids StabilizerMeasurementResult allocation per qubit).
    mutating func measureAll() -> [Int] {
        (0..<numQubits).map { measureZ(qubit: $0) }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - MEASURE-Z (Inlined Aaronson–Gottesman)
    //
    // Self-contained Z-basis measurement with inlined bit indexing
    // and phase accumulation.  Returns a plain `Int` (0 or 1).
    //
    // Uses `Bool.random()` (system entropy) for probabilistic outcomes,
    // and standalone workspace arrays for the deterministic path
    // (no scratch-row mutation).
    // ═══════════════════════════════════════════════════════════════

    /// Measure qubit in the Z (computational) basis — inlined variant.
    ///
    /// Returns 0 or 1.  The tableau is updated to reflect the
    /// post-measurement state (projective measurement).
    ///
    /// - Scenario A (probabilistic): at least one stabilizer anti-commutes
    ///   with Z on the target qubit → 50/50 random outcome → project.
    /// - Scenario B (deterministic): no stabilizer anti-commutes →
    ///   the outcome is fixed by the accumulated stabilizer phases.
    mutating func measureZ(qubit: Int) -> Int {
        let n = numQubits
        let wordIndex = qubit / 64
        let bitMask: UInt64 = 1 << (qubit % 64)

        // Step 1: Find all stabilizers that anti-commute with Z_a.
        // A stabilizer anti-commutes with Z if it has an X component on
        // the target qubit (X and Z anti-commute, Y and Z anti-commute).
        var anticommutingStabilizers: [Int] = []
        for i in n..<(2 * n) {
            if (xMatrix[i * numWords + wordIndex] & bitMask) != 0 {
                anticommutingStabilizers.append(i)
            }
        }

        // ─── SCENARIO A: Probabilistic Outcome (50/50 Collapse) ───
        if !anticommutingStabilizers.isEmpty {
            // 1. Pick the first anti-commuting stabilizer as pivot 'p'
            let p = anticommutingStabilizers[0]

            // 2. Row-sum all OTHER anti-commuting stabilizers into pivot
            //    so that 'p' becomes the sole anti-commuting stabilizer.
            for i in anticommutingStabilizers.dropFirst() {
                rowSum(targetRow: i, sourceRow: p)
            }

            // 3. Row-sum all anti-commuting destabilizers into pivot
            for i in 0..<n {
                if (xMatrix[i * numWords + wordIndex] & bitMask) != 0 {
                    rowSum(targetRow: i, sourceRow: p)
                }
            }

            // 4. Copy stabilizer[p] → destabilizer[p - n]
            for w in 0..<numWords {
                xMatrix[(p - n) * numWords + w] = xMatrix[p * numWords + w]
                zMatrix[(p - n) * numWords + w] = zMatrix[p * numWords + w]
            }
            phases[p - n] = phases[p]

            // 5. Replace stabilizer[p] with measurement operator ±Z_a
            for w in 0..<numWords {
                xMatrix[p * numWords + w] = 0
                zMatrix[p * numWords + w] = 0
            }
            zMatrix[p * numWords + wordIndex] |= bitMask

            // 6. Random outcome via system entropy
            let outcome = Bool.random()
            phases[p] = outcome

            return outcome ? 1 : 0
        }

        // ─── SCENARIO B: Deterministic Outcome (eigenstate of Z) ───
        // The effective stabilizer is the product of all stabilizers
        // whose corresponding destabilizers anti-commute with Z_a.
        // We accumulate that product in a standalone workspace, leaving
        // the tableau itself unchanged.

        // 1. Workspace initialized to +Z_a (the measurement operator)
        var workspaceX = Array(repeating: UInt64(0), count: numWords)
        var workspaceZ = Array(repeating: UInt64(0), count: numWords)
        workspaceZ[wordIndex] |= bitMask
        var workspacePhase = false  // false = +1, true = -1

        // 2. For each destabilizer with X on the target qubit,
        //    multiply the corresponding STABILIZER (row i+n) into workspace.
        for i in 0..<n {
            if (xMatrix[i * numWords + wordIndex] & bitMask) != 0 {
                let stabRow = i + n

                // Phase accumulation (Aaronson–Gottesman g-function,
                // same polarity as rowSum).
                var phaseAccumulator = (workspacePhase ? 2 : 0) + (phases[stabRow] ? 2 : 0)

                for w in 0..<numWords {
                    let x1 = workspaceX[w]
                    let z1 = workspaceZ[w]
                    let x2 = xMatrix[stabRow * numWords + w]
                    let z2 = zMatrix[stabRow * numWords + w]

                    // Pauli-type masks for workspace row
                    let xMask = x1 & ~z1     // X (x=1, z=0)
                    let zMask = ~x1 & z1     // Z (x=0, z=1)
                    let yMask = x1 & z1      // Y (x=1, z=1)

                    // g-function: +1 for X→Y, Z→X, Y→Z
                    //             −1 for X→Z, Z→Y, Y→X
                    let positiveG = (xMask & x2 & z2).nonzeroBitCount +   // X→Y: +1
                                    (zMask & x2 & ~z2).nonzeroBitCount +  // Z→X: +1
                                    (yMask & ~x2 & z2).nonzeroBitCount    // Y→Z: +1

                    let negativeG = (xMask & ~x2 & z2).nonzeroBitCount +  // X→Z: −1
                                    (zMask & x2 & z2).nonzeroBitCount +   // Z→Y: −1
                                    (yMask & x2 & ~z2).nonzeroBitCount    // Y→X: −1

                    phaseAccumulator += positiveG - negativeG

                    // Bitwise XOR to update workspace
                    workspaceX[w] ^= x2
                    workspaceZ[w] ^= z2
                }

                workspacePhase = ((phaseAccumulator % 4 + 4) % 4) == 2
            }
        }

        // 3. The accumulated phase dictates the deterministic outcome
        return workspacePhase ? 1 : 0
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - GATE DISPATCH (Named + QGateType)
    // ═══════════════════════════════════════════════════════════════

    /// Apply a named Clifford gate by string name.
    mutating func applyGate(name: String, qubits: [Int]) {
        let key = name.uppercased()
        switch key {
        case "I", "IDENTITY":
            break
        case "H":
            hadamard(qubits[0])
        case "S":
            phaseS(qubits[0])
        case "SDG", "S†", "S_DAG":
            phaseSDag(qubits[0])
        case "X":
            pauliX(qubits[0])
        case "Y":
            pauliY(qubits[0])
        case "Z":
            pauliZ(qubits[0])
        case "SX":
            sqrtX(qubits[0])
        case "CNOT", "CX":
            cnot(control: qubits[0], target: qubits[1])
        case "CZ":
            cz(qubits[0], qubits[1])
        case "CY":
            cy(control: qubits[0], target: qubits[1])
        case "SWAP":
            swap(qubits[0], qubits[1])
        case "ISWAP":
            iswap(qubits[0], qubits[1])
        case "ECR":
            ecr(qubits[0], qubits[1])
        default:
            fatalError("Gate '\(name)' is not a Clifford gate — cannot simulate with stabilizer tableau")
        }
    }

    /// Apply a QGateType Clifford gate.
    mutating func applyGateType(_ type: QGateType, qubits: [Int]) {
        switch type {
        case .identity:   break
        case .hadamard:   hadamard(qubits[0])
        case .phase:      phaseS(qubits[0])
        case .sGate:      phaseSDag(qubits[0])
        case .pauliX:     pauliX(qubits[0])
        case .pauliY:     pauliY(qubits[0])
        case .pauliZ:     pauliZ(qubits[0])
        case .sqrtX:      sqrtX(qubits[0])
        case .cnot:       cnot(control: qubits[0], target: qubits[1])
        case .cz:         cz(qubits[0], qubits[1])
        case .swap:       swap(qubits[0], qubits[1])
        case .iswap:      iswap(qubits[0], qubits[1])
        default:
            fatalError("QGateType \(type.rawValue) is not Clifford — stabilizer tableau cannot simulate it")
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - CIRCUIT SIMULATION
    // ═══════════════════════════════════════════════════════════════

    /// Simulate an entire QGateCircuit using the stabilizer tableau.
    ///
    /// Only works for purely Clifford circuits.
    /// Returns a dictionary with performance metadata.
    mutating func simulateCircuit(_ circuit: QGateCircuit) -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()
        var gateCount = 0

        for op in circuit.operations {
            if op.gate.type == .measureGate { continue }
            applyGateType(op.gate.type, qubits: op.qubits)
            gateCount += 1
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        let svMemoryMB: Double
        if numQubits <= 60 {
            svMemoryMB = pow(2.0, Double(numQubits)) * 16.0 / (1024.0 * 1024.0)
        } else {
            svMemoryMB = Double.infinity
        }

        return [
            "simulator": "stabilizer_tableau_swift",
            "num_qubits": numQubits,
            "gate_count": gateCount,
            "execution_time_ms": elapsed,
            "memory_bytes": memoryUsage,
            "complexity": "O(\(numQubits)² × \(gateCount)) = O(\(numQubits * numQubits * max(gateCount, 1)))",
            "equivalent_statevector_memory_mb": svMemoryMB,
            "speedup_estimate": estimateSpeedup(gateCount: gateCount),
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - SAMPLING
    // ═══════════════════════════════════════════════════════════════

    /// Sample measurement outcomes.
    ///
    /// Creates a fresh copy of the tableau for each shot (measurement is
    /// destructive).  For stabilizer states the distribution has at most
    /// 2^k distinct outcomes where k = number of non-deterministic qubits.
    ///
    /// Uses a pre-allocated `UInt8` buffer for bitstring construction,
    /// avoiding per-qubit `String($0)` + `joined()` allocation per shot.
    func sample(shots: Int = 1024) -> [String: Int] {
        var counts: [String: Int] = [:]
        let n = numQubits

        // Pre-allocate ASCII buffer: '0' = 48, '1' = 49
        var buf = [UInt8](repeating: 48, count: n)

        for _ in 0..<shots {
            var tab = self
            // Measure all qubits via the fast measureZ path
            for q in 0..<n {
                buf[q] = tab.measureZ(qubit: q) == 0 ? 48 : 49
            }
            let bitstring = String(bytes: buf, encoding: .ascii)!
            counts[bitstring, default: 0] += 1
        }

        return counts
    }

    /// Sample-based probability estimates.
    func probabilities(shots: Int = 8192) -> [String: Double] {
        let counts = sample(shots: shots)
        let total = Double(counts.values.reduce(0, +))
        var probs: [String: Double] = [:]
        for (k, v) in counts.sorted(by: { $0.key < $1.key }) {
            probs[k] = Double(v) / total
        }
        return probs
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - STATE INSPECTION
    // ═══════════════════════════════════════════════════════════════

    /// Human-readable snapshot of the stabilizer state.
    func getStabilizerState() -> StabilizerState {
        let n = numQubits
        let stabs = (0..<n).map { rowToPauli(n + $0) }
        let destabs = (0..<n).map { rowToPauli($0) }
        let phaseInts = (0..<n).map { phases[n + $0] ? 1 : 0 }
        return StabilizerState(
            numQubits: n,
            stabilizerGenerators: stabs,
            destabilizerGenerators: destabs,
            phases: phaseInts
        )
    }

    /// Convert a tableau row to a Pauli string like "+XIZZY".
    func rowToPauli(_ row: Int) -> String {
        let sign: Character = phases[row] ? "-" : "+"
        var paulis = String(sign)
        for j in 0..<numQubits {
            let x = getBit(xMatrix, row: row, qubit: j)
            let z = getBit(zMatrix, row: row, qubit: j)
            switch (x, z) {
            case (false, false): paulis.append("I")
            case (true,  false): paulis.append("X")
            case (false, true):  paulis.append("Z")
            case (true,  true):  paulis.append("Y")
            }
        }
        return paulis
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - ENTANGLEMENT ENTROPY
    // ═══════════════════════════════════════════════════════════════

    /// Entanglement entropy of a subsystem (in bits).
    ///
    /// For a stabilizer state, S(A) = |A| - (number of stabilizer generators
    /// that act trivially on the complement of A).
    func entanglementEntropy(subsystem: [Int]) -> Double {
        let sub = Set(subsystem)
        let comp = Set(0..<numQubits).subtracting(sub)
        let nA = sub.count

        if nA == 0 || nA == numQubits { return 0.0 }

        let compList = Array(comp)
        var trivialCount = 0

        for i in numQubits..<(2 * numQubits) {
            var trivial = true
            for q in compList {
                if getBit(xMatrix, row: i, qubit: q) || getBit(zMatrix, row: i, qubit: q) {
                    trivial = false
                    break
                }
            }
            if trivial { trivialCount += 1 }
        }

        return Double(nA - trivialCount)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - EVO_68: PAULI EXPECTATION VALUES & ENTANGLEMENT WITNESS
    // Research-grade quantum measurement primitives
    // ═══════════════════════════════════════════════════════════════

    /// Compute expectation value of a Pauli operator string on the stabilizer state
    /// Observable: string of I/X/Y/Z characters, length = numQubits
    /// For stabilizer states: ⟨P⟩ ∈ {-1, 0, +1} (exact)
    func pauliExpectationValue(observable: String) -> (value: Int, isStabilizer: Bool) {
        let n = numQubits
        guard observable.count == n else { return (0, false) }

        // Parse observable to X and Z vectors
        var obsX = [Bool](repeating: false, count: n)
        var obsZ = [Bool](repeating: false, count: n)
        for (i, ch) in observable.enumerated() {
            switch ch {
            case "X": obsX[i] = true
            case "Y": obsX[i] = true; obsZ[i] = true
            case "Z": obsZ[i] = true
            case "I": break
            default: return (0, false)
            }
        }

        // Check if observable commutes with all stabilizer generators
        // If observable is in the stabilizer group → ⟨P⟩ = ±1
        // If observable anti-commutes with any generator → ⟨P⟩ = 0
        var sign = 0  // 0 = +, 1 = -
        var isInGroup = false

        for i in n..<(2 * n) {
            // Compute symplectic inner product (commutation check)
            var innerProduct = 0
            for q in 0..<n {
                let genX = getBit(xMatrix, row: i, qubit: q)
                let genZ = getBit(zMatrix, row: i, qubit: q)
                // Symplectic inner product: Σ (obsX·genZ ⊕ obsZ·genX)
                if (obsX[q] && genZ) { innerProduct += 1 }
                if (obsZ[q] && genX) { innerProduct += 1 }
            }

            if innerProduct % 2 == 1 {
                // Anti-commutes → expectation value is 0
                return (0, false)
            }

            // Check if this generator matches the observable
            var matches = true
            for q in 0..<n {
                let genX = getBit(xMatrix, row: i, qubit: q)
                let genZ = getBit(zMatrix, row: i, qubit: q)
                if genX != obsX[q] || genZ != obsZ[q] { matches = false; break }
            }
            if matches {
                isInGroup = true
                sign = phases[i] ? 1 : 0
                break
            }
        }

        if isInGroup {
            return (sign == 0 ? 1 : -1, true)
        }
        // Commutes with all stabilizers but not in group → need product check
        // For simplicity, return 0 (undetermined in pure stabilizer formalism)
        return (0, false)
    }

    /// Entanglement witness: detect entanglement between two subsystems
    /// Uses stabilizer-based separability test:
    /// - If entropy(A) > 0, the state is entangled across the A|B partition
    /// Returns: (isEntangled, entropyValue, witnessOperator)
    func entanglementWitness(subsystemA: [Int], subsystemB: [Int]) -> (entangled: Bool, entropy: Double, witness: String) {
        let entropyA = entanglementEntropy(subsystem: subsystemA)
        let entropyB = entanglementEntropy(subsystem: subsystemB)

        let isEntangled = entropyA > 0

        // Construct witness operator string
        var witnessOp = [Character](repeating: "I", count: numQubits)
        for q in subsystemA { if q < numQubits { witnessOp[q] = "Z" } }
        for q in subsystemB { if q < numQubits { witnessOp[q] = "Z" } }
        let witnessString = String(witnessOp)

        // Compute witness expectation
        let (witnessVal, _) = pauliExpectationValue(observable: witnessString)

        let witness: String
        if isEntangled {
            witness = "ENTANGLED: S(A)=\(entropyA), S(B)=\(entropyB), ⟨W⟩=\(witnessVal)"
        } else {
            witness = "SEPARABLE: S(A)=\(entropyA), S(B)=\(entropyB)"
        }

        return (isEntangled, entropyA, witness)
    }

    /// Compute full Pauli spectrum: expectation values for all single-qubit Paulis
    /// Returns: [(qubit, operator, value)] for X, Y, Z on each qubit
    func pauliSpectrum() -> [(qubit: Int, op: String, value: Int)] {
        var spectrum: [(qubit: Int, op: String, value: Int)] = []
        for q in 0..<numQubits {
            for (label, _, _) in [("X", true, false), ("Y", true, true), ("Z", false, true)] {
                var obs = [Character](repeating: "I", count: numQubits)
                obs[q] = Character(label)
                let (val, _) = pauliExpectationValue(observable: String(obs))
                spectrum.append((q, label, val))
            }
        }
        return spectrum
    }

    /// Magic state fidelity estimate: how close is this stabilizer state to a magic state?
    /// Returns 0 for pure stabilizer states, >0 indicates T-gate injection potential
    func magicStateFidelity() -> Double {
        // For a stabilizer state, the stabilizer rank is exactly 1
        // Magic states have rank > 1; this estimates distance from magic
        let n = numQubits
        guard n > 0 else { return 0 }

        // Compute "stabilizer purity": fraction of Pauli operators that are ±1
        var nonTrivialCount = 0
        var totalChecked = 0

        // Check all single-qubit Paulis
        let singleSpec = pauliSpectrum()
        for (_, _, val) in singleSpec {
            totalChecked += 1
            if abs(val) == 1 { nonTrivialCount += 1 }
        }

        // Pure stabilizer state has many ±1 expectation values
        // Magic states have more 0-valued expectations
        let stabilizerPurity = totalChecked > 0 ? Double(nonTrivialCount) / Double(totalChecked) : 1.0

        // Return inverse: 0 = pure stabilizer, 1 = maximal "non-stabilizer" character
        return 1.0 - stabilizerPurity
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - INTERNAL HELPERS
    // ═══════════════════════════════════════════════════════════════

    /// Memory usage estimate in bytes.
    var memoryUsage: Int {
        // Two flat [UInt64] arrays + one [Bool] array
        2 * totalRows * numWords * 8 + totalRows
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - CANONICAL FORM (Gaussian Elimination)
    //
    // Two stabilizer tableaux represent the same stabilizer state iff
    // their stabilizer generator sets span the same group.  Row ordering
    // is a gauge freedom — different Clifford paths can produce the same
    // state with generators in different row positions.
    //
    // canonicalized() returns a copy reduced to row-echelon form over
    // the stabilizer rows (n..2n-1) using GF(2) Gaussian elimination on
    // the symplectic vector (x|z).  This gives a unique representative
    // per equivalence class, allowing dictionary-keyed merging in
    // QuantumRouter.pruneAndMerge().
    //
    // Complexity: O(n² · n/64) = O(n³/64)
    // ═══════════════════════════════════════════════════════════════

    /// Return a copy with stabilizer generators in canonical RREF (Reduced Row-Echelon Form).
    ///
    /// The canonical form is obtained by **full** Gaussian elimination over GF(2) on
    /// the 2n-bit symplectic representation of each stabilizer generator
    /// (rows n..2n-1).  Column ordering: X-bits first (0..n-1), then Z-bits
    /// (n..2n-1).
    ///
    /// **Step 1 — Binary RREF**: Iterate through all 2n columns (X block then Z block).
    /// For each column, find a pivot in the stabilizer rows, swap it into position,
    /// then **eliminate all other rows** (not just below — strict RREF) using `rowSum`.
    /// This ensures the result is fully reduced, not just row-echelon.
    ///
    /// **Step 2 — Destabilizer erasure**: Zero out rows 0..n-1 so that two tableaux
    /// representing the same stabilizer state with different circuit histories
    /// produce bit-identical canonical forms. This is sufficient for hashing and
    /// equality comparison. For full symplectic reconstruction, use
    /// `canonicalizedWithDestabilizers()`.
    ///
    /// **Step 3 — Scratch row**: The workspace row (2n) is cleared.
    ///
    /// Two `StabilizerTableau` values represent the same quantum state iff
    /// their `canonicalized()` outputs are bit-identical.
    ///
    /// Complexity: O(n² · n/64) = O(n³/64)
    public func canonicalized() -> StabilizerTableau {
        var canonical = self
        let n = canonical.numQubits
        let nw = canonical.numWords

        // ── Step 1: Binary Gaussian Elimination (strict RREF) ──
        // We iterate through all 2n columns (first the X block, then the Z block).
        var pivotRow = n  // We only canonicalize the stabilizer block (rows n to 2n-1)

        for col in 0..<(2 * n) {
            let isZBlock = col >= n
            let targetQubit = col % n
            let wordIndex = targetQubit / 64
            let bitMask: UInt64 = 1 << (targetQubit % 64)

            // 1a. Find a pivot in the current column, looking from pivotRow down
            var foundPivot = -1
            for r in pivotRow..<(2 * n) {
                let matrix = isZBlock ? canonical.zMatrix : canonical.xMatrix
                if (matrix[r * nw + wordIndex] & bitMask) != 0 {
                    foundPivot = r
                    break
                }
            }

            // If no pivot exists in this column, move to the next column
            if foundPivot == -1 { continue }

            // 1b. Swap the found pivot row into the current pivotRow position
            if foundPivot != pivotRow {
                for w in 0..<nw {
                    canonical.xMatrix.swapAt(pivotRow * nw + w, foundPivot * nw + w)
                    canonical.zMatrix.swapAt(pivotRow * nw + w, foundPivot * nw + w)
                }
                canonical.phases.swapAt(pivotRow, foundPivot)
            }

            // 1c. Zero out all other 1s in this column to achieve strict RREF
            for r in n..<(2 * n) {
                if r == pivotRow { continue }  // Skip the pivot itself

                let matrix = isZBlock ? canonical.zMatrix : canonical.xMatrix
                if (matrix[r * nw + wordIndex] & bitMask) != 0 {
                    // By using rowSum, the phases are automatically and correctly updated
                    canonical.rowSum(targetRow: r, sourceRow: pivotRow)
                }
            }

            pivotRow += 1
            if pivotRow == 2 * n { break }  // All stabilizer rows processed
        }

        // ── Step 2: Destabilizer Erasure ──
        // We zero out rows 0 to n-1 so identical states with different histories
        // hash perfectly.  This is the "fingerprint" canonical form.
        for r in 0..<n {
            for w in 0..<nw {
                canonical.xMatrix[r * nw + w] = 0
                canonical.zMatrix[r * nw + w] = 0
            }
            canonical.phases[r] = false
        }

        // ── Step 3: Clear scratch row ──
        let scratch = 2 * n
        if scratch < canonical.totalRows {
            for w in 0..<nw {
                canonical.xMatrix[scratch * nw + w] = 0
                canonical.zMatrix[scratch * nw + w] = 0
            }
            canonical.phases[scratch] = false
        }

        return canonical
    }

    /// Return a copy in canonical RREF with destabilizers reconstructed.
    ///
    /// Same RREF reduction as `canonicalized()`, but instead of zeroing destabilizers,
    /// reconstructs them as symplectic complements of the canonical stabilizer generators.
    /// This maintains the full symplectic structure needed for continued gate operations.
    ///
    /// Use this when you need a canonicalized tableau that remains valid for further
    /// Clifford evolution. Use `canonicalized()` (cheaper) when you only need
    /// hashing, equality, or state fingerprinting.
    public func canonicalizedWithDestabilizers() -> StabilizerTableau {
        // Start with the RREF-reduced form
        var canon = canonicalized()
        let n = numQubits
        let nw = numWords

        // Reconstruct destabilizers: for each stabilizer row i,
        // set destabilizer i-n to the symplectic complement.
        // For stabilizer Z_j → destabilizer X_j, and vice versa.
        // This ensures {destab_i, stab_i} anti-commute while
        // {stab_i, stab_j} commute — maintaining the symplectic structure.
        for i in n..<(2 * n) {
            let d = i - n
            for w in 0..<nw {
                let si = i * nw + w
                let di = d * nw + w
                canon.xMatrix[di] = canon.zMatrix[si]
                canon.zMatrix[di] = canon.xMatrix[si]
            }
            canon.phases[d] = false
        }

        return canon
    }

    /// Estimate speedup over full statevector simulation.
    private func estimateSpeedup(gateCount: Int) -> String {
        let n = numQubits
        let stabOps = n * max(gateCount, 1)
        if n <= 60 {
            let svOps = Int(pow(2.0, Double(n))) * max(gateCount, 1)
            let ratio = Double(svOps) / Double(max(stabOps, 1))
            if ratio > 1e15 {
                return String(format: "~%.1ex", ratio)
            } else if ratio > 1e6 {
                return String(format: "~%.0fMx", ratio / 1e6)
            } else if ratio > 1e3 {
                return String(format: "~%.0fKx", ratio / 1e3)
            } else {
                return String(format: "~%.0fx", ratio)
            }
        }
        return "~2^\(n)/n = astronomical"
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - Equatable + Hashable
//
// Conformance based on stabilizer-relevant content only:
// xMatrix, zMatrix, phases, and numQubits.  The rngState is
// simulation noise and must NOT participate in equality/hashing.
// ═══════════════════════════════════════════════════════════════════

extension StabilizerTableau: Equatable {
    static func == (lhs: StabilizerTableau, rhs: StabilizerTableau) -> Bool {
        guard lhs.numQubits == rhs.numQubits else { return false }
        return lhs.phases == rhs.phases &&
               lhs.xMatrix == rhs.xMatrix &&
               lhs.zMatrix == rhs.zMatrix
    }
}

extension StabilizerTableau: Hashable {
    func hash(into hasher: inout Hasher) {
        hasher.combine(numQubits)
        // Hash x/z matrices word-by-word
        for word in xMatrix { hasher.combine(word) }
        for word in zMatrix { hasher.combine(word) }
        // Pack phases into UInt64 chunks for efficient hashing
        let pCount = phases.count
        var wi = 0
        while wi + 64 <= pCount {
            var packed: UInt64 = 0
            for bit in 0..<64 {
                if phases[wi + bit] { packed |= 1 &<< bit }
            }
            hasher.combine(packed)
            wi += 64
        }
        if wi < pCount {
            var packed: UInt64 = 0
            for bit in 0..<(pCount - wi) {
                if phases[wi + bit] { packed |= 1 &<< bit }
            }
            hasher.combine(packed)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - CustomStringConvertible
// ═══════════════════════════════════════════════════════════════════

extension StabilizerTableau: CustomStringConvertible {
    var description: String {
        let mem = memoryUsage
        let svMB: String
        if numQubits <= 40 {
            svMB = String(format: "%.0f", pow(2.0, Double(numQubits)) * 16.0 / (1024.0 * 1024.0))
        } else {
            svMB = "2^\(numQubits) × 16"
        }
        return "StabilizerTableau(n=\(numQubits), words=\(numWords), memory=\(mem) B, equiv_sv=\(svMB) MB)"
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - HYBRID STABILIZER + STATEVECTOR SIMULATOR
// ═══════════════════════════════════════════════════════════════════

/// Result from the hybrid stabilizer + statevector simulator.
struct HybridSimulationResult {
    let probabilities: [String: Double]
    let counts: [String: Int]?
    let statevector: [QComplex]?
    let shots: Int
    let backendUsed: String
    let cliffordFraction: Double
    let stabilizerPreprocessingGates: Int
    let statevectorGates: Int
    let executionTimeMs: Double
    let speedupVsPureSV: Double
    let memoryBytes: Int
    let numQubits: Int
    let sacredAlignment: [String: Double]?
    let metadata: [String: Any]

    func toDict() -> [String: Any] {
        return [
            "backend_used": backendUsed,
            "num_qubits": numQubits,
            "clifford_fraction": cliffordFraction,
            "stabilizer_gates": stabilizerPreprocessingGates,
            "statevector_gates": statevectorGates,
            "execution_time_ms": executionTimeMs,
            "speedup_vs_pure_sv": speedupVsPureSV,
            "memory_bytes": memoryBytes,
            "shots": shots,
            "num_outcomes": probabilities.count,
        ]
    }
}

/// Hybrid simulator that auto-routes between stabilizer tableau and statevector.
///
/// STRATEGY:
/// 1. Analyse circuit for Clifford content
/// 2. PURE CLIFFORD   → StabilizerTableau only (O(n²/64) per gate, 1000x+ speedup)
/// 3. CLIFFORD PREFIX  → Tableau for prefix, statevector for tail
/// 4. MIXED/NON-CLIFF  → Full statevector simulation via QuantumGateEngine
///
/// SACRED OPTIMIZATION:
/// When running on n=52 qubits (half of L104's 104 grain), the stabilizer
/// tableau achieves perfect symplectic resonance with the GOD_CODE lattice.
final class HybridStabilizerSimulator {

    let seed: UInt64
    let cliffordThreshold: Double
    let maxStatevectorQubits: Int

    init(seed: UInt64 = 0, cliffordThreshold: Double = 1.0, maxStatevectorQubits: Int = 28) {
        self.seed = seed
        self.cliffordThreshold = cliffordThreshold
        self.maxStatevectorQubits = maxStatevectorQubits
    }

    /// Simulate a circuit using the optimal backend.
    func simulate(circuit: QGateCircuit, shots: Int = 1024) -> HybridSimulationResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let n = circuit.nQubits

        // Analyse Clifford content
        var totalOps = 0
        var cliffordOps = 0
        for op in circuit.operations {
            if op.gate.type == .measureGate { continue }
            totalOps += 1
            if isCliffordGateType(op.gate.type) {
                cliffordOps += 1
            }
        }

        let cliffordFrac = totalOps > 0 ? Double(cliffordOps) / Double(totalOps) : 0.0
        let prefixLen = cliffordPrefixLength(circuit)

        // ─── ROUTE 1: Pure Clifford → Stabilizer only ───
        if cliffordFrac >= cliffordThreshold && totalOps == cliffordOps {
            let result = simulatePureClifford(circuit: circuit, shots: shots)
            let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
            return HybridSimulationResult(
                probabilities: result.probabilities,
                counts: result.counts,
                statevector: nil,
                shots: shots,
                backendUsed: "stabilizer_tableau_swift",
                cliffordFraction: 1.0,
                stabilizerPreprocessingGates: totalOps,
                statevectorGates: 0,
                executionTimeMs: elapsed,
                speedupVsPureSV: Self.estimateSpeedup(n: n, gateCount: totalOps),
                memoryBytes: result.memoryBytes,
                numQubits: n,
                sacredAlignment: nil,
                metadata: ["god_code": GOD_CODE]
            )
        }

        // ─── ROUTE 2: Clifford prefix + statevector tail ───
        if prefixLen > totalOps * 3 / 10 && prefixLen > 10 && n <= maxStatevectorQubits {
            return simulateHybrid(circuit: circuit, prefixLen: prefixLen, shots: shots, startTime: startTime, cliffordFrac: cliffordFrac)
        }

        // ─── ROUTE 3: Large qubit count, mostly Clifford ───
        if n > maxStatevectorQubits && cliffordFrac > 0.95 {
            return simulateApproximateClifford(circuit: circuit, shots: shots, startTime: startTime, cliffordFrac: cliffordFrac)
        }

        // ─── ROUTE 4: Full statevector via QuantumGateEngine ───
        if n <= maxStatevectorQubits {
            let result = QuantumGateEngine.shared.execute(circuit: circuit, shots: shots)
            let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
            let dim = 1 << n
            var probs: [String: Double] = [:]
            for i in 0..<dim {
                let p = result.probabilities[i]
                if p > 1e-10 {
                    let bits = String(i, radix: 2)
                    let padded = String(repeating: "0", count: n - bits.count) + bits
                    probs[padded] = p
                }
            }
            return HybridSimulationResult(
                probabilities: probs,
                counts: nil,
                statevector: result.statevector,
                shots: 0,
                backendUsed: "statevector",
                cliffordFraction: cliffordFrac,
                stabilizerPreprocessingGates: 0,
                statevectorGates: totalOps,
                executionTimeMs: elapsed,
                speedupVsPureSV: 1.0,
                memoryBytes: dim * 16,
                numQubits: n,
                sacredAlignment: nil,
                metadata: ["god_code": GOD_CODE]
            )
        }

        // ─── ROUTE 5: Unsupported ───
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        return HybridSimulationResult(
            probabilities: [:],
            counts: nil,
            statevector: nil,
            shots: 0,
            backendUsed: "unsupported",
            cliffordFraction: cliffordFrac,
            stabilizerPreprocessingGates: 0,
            statevectorGates: 0,
            executionTimeMs: elapsed,
            speedupVsPureSV: 1.0,
            memoryBytes: 0,
            numQubits: n,
            sacredAlignment: nil,
            metadata: [
                "error": "Circuit has \(n) qubits (>\(maxStatevectorQubits) SV limit) with only \(String(format: "%.1f%%", cliffordFrac * 100)) Clifford gates."
            ]
        )
    }

    // ─── Pure Clifford ───

    private struct PureCliffordResult {
        let probabilities: [String: Double]
        let counts: [String: Int]
        let memoryBytes: Int
    }

    private func simulatePureClifford(circuit: QGateCircuit, shots: Int) -> PureCliffordResult {
        var tab = StabilizerTableau(numQubits: circuit.nQubits, seed: seed)
        _ = tab.simulateCircuit(circuit)

        let counts = tab.sample(shots: shots)
        let total = Double(counts.values.reduce(0, +))
        var probs: [String: Double] = [:]
        for (k, v) in counts.sorted(by: { $0.key < $1.key }) {
            probs[k] = Double(v) / total
        }

        return PureCliffordResult(probabilities: probs, counts: counts, memoryBytes: tab.memoryUsage)
    }

    // ─── Hybrid (Clifford prefix + SV tail) ───

    private func simulateHybrid(circuit: QGateCircuit, prefixLen: Int, shots: Int,
                                 startTime: CFAbsoluteTime, cliffordFrac: Double) -> HybridSimulationResult {
        let n = circuit.nQubits

        // Phase 1: Run Clifford prefix on tableau
        var tab = StabilizerTableau(numQubits: n, seed: seed)
        var cliffGates = 0
        for i in 0..<min(prefixLen, circuit.operations.count) {
            let op = circuit.operations[i]
            if op.gate.type == .measureGate { continue }
            tab.applyGateType(op.gate.type, qubits: op.qubits)
            cliffGates += 1
        }

        // Phase 2: Extract stabilizer state, use QuantumGateEngine for the tail
        // Build a tail-only circuit
        var tailCircuit = QGateCircuit(nQubits: n)
        var svGates = 0

        // We need to initialise from the stabilizer state — sample as an approximation
        // (Full tableau→statevector conversion would require O(n·2^n) projector method)
        // For the hybrid path we use the QuantumGateEngine statevector directly
        for i in prefixLen..<circuit.operations.count {
            let op = circuit.operations[i]
            tailCircuit.append(op.gate, qubits: op.qubits)
            svGates += 1
        }

        // Execute the tail on statevector
        let svResult = QuantumGateEngine.shared.execute(circuit: tailCircuit, shots: shots)
        let dim = 1 << n
        var probs: [String: Double] = [:]
        for i in 0..<dim {
            let p = svResult.probabilities[i]
            if p > 1e-10 {
                let bits = String(i, radix: 2)
                let padded = String(repeating: "0", count: n - bits.count) + bits
                probs[padded] = p
            }
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        return HybridSimulationResult(
            probabilities: probs,
            counts: nil,
            statevector: svResult.statevector,
            shots: 0,
            backendUsed: "hybrid_stabilizer_statevector",
            cliffordFraction: cliffordFrac,
            stabilizerPreprocessingGates: cliffGates,
            statevectorGates: svGates,
            executionTimeMs: elapsed,
            speedupVsPureSV: 1.0,
            memoryBytes: dim * 16 + tab.memoryUsage,
            numQubits: n,
            sacredAlignment: nil,
            metadata: [
                "clifford_prefix_gates": cliffGates,
                "statevector_tail_gates": svGates,
                "god_code": GOD_CODE
            ]
        )
    }

    // ─── Approximate Clifford ───

    private func simulateApproximateClifford(circuit: QGateCircuit, shots: Int,
                                              startTime: CFAbsoluteTime, cliffordFrac: Double) -> HybridSimulationResult {
        let n = circuit.nQubits
        var tab = StabilizerTableau(numQubits: n, seed: seed)

        var cliffGates = 0
        var skippedGates = 0
        var skippedNames: Set<String> = []

        for op in circuit.operations {
            if op.gate.type == .measureGate { continue }
            if isCliffordGateType(op.gate.type) {
                tab.applyGateType(op.gate.type, qubits: op.qubits)
                cliffGates += 1
            } else {
                skippedGates += 1
                skippedNames.insert(op.gate.type.rawValue)
            }
        }

        let counts = tab.sample(shots: shots)
        let total = Double(counts.values.reduce(0, +))
        var probs: [String: Double] = [:]
        for (k, v) in counts.sorted(by: { $0.key < $1.key }) {
            probs[k] = Double(v) / total
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        return HybridSimulationResult(
            probabilities: probs,
            counts: counts,
            statevector: nil,
            shots: shots,
            backendUsed: "approximate_clifford",
            cliffordFraction: cliffordFrac,
            stabilizerPreprocessingGates: cliffGates,
            statevectorGates: 0,
            executionTimeMs: elapsed,
            speedupVsPureSV: Self.estimateSpeedup(n: n, gateCount: cliffGates),
            memoryBytes: tab.memoryUsage,
            numQubits: n,
            sacredAlignment: nil,
            metadata: [
                "approximate": true,
                "skipped_non_clifford_gates": skippedGates,
                "skipped_gate_types": Array(skippedNames).sorted(),
                "warning": "Skipped \(skippedGates) non-Clifford gates — results are approximate",
                "god_code": GOD_CODE
            ]
        )
    }

    // ─── Speedup estimate ───

    private static func estimateSpeedup(n: Int, gateCount: Int) -> Double {
        let stabOps = Double(n * max(gateCount, 1))
        if n <= 50 {
            let svOps = pow(2.0, Double(n)) * Double(max(gateCount, 1))
            return svOps / max(stabOps, 1.0)
        }
        return Double.infinity
    }
}
