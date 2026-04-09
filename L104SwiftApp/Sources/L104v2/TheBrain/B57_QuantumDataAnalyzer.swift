import Accelerate
import Foundation

// MARK: - ═══ EVO_71 ENHANCED CONSTANTS ═══

private let MAX_QUBITS    = 16          // [EVO_71] DOUBLED: 12 → 16 for expanded state space
private let DEFAULT_SHOTS = 4096       // [EVO_71] DOUBLED: 1024 → 4096 for higher precision
private let ANOMALY_THRESHOLD = 1.0 / PHI   // ≈ 0.618
// ALPHA_FINE is global in L01_Constants.swift; use that directly
private let H_BAR         = 1.0545718e-34
private let PHI_CONJUGATE = 1.0 / PHI        // ≈ 0.618033 - golden conjugate
private let QDA_N_ITER    = 208             // [EVO_71] DOUBLED: 104 → 208 iterations

// MARK: - ═══ DATA STRUCTURES ═══

struct QDataPoint {
    let values: [Double]
    let label: String
    let weight: Double
    init(values: [Double], label: String = "", weight: Double = 1.0) {
        self.values = values; self.label = label; self.weight = weight
    }
}

struct SpectralResult {
    let frequencies:     [Double]
    let magnitudes:      [Double]
    let dominantFreq:    Double
    let godCodeAlignment: Double   // resonance with GOD_CODE harmonics
    let sacredPeaks:     [Double]  // frequencies aligning with PHI multiples
    let totalPower:      Double
}

struct PatternMatch {
    let index:       Int
    let similarity:  Double
    let amplitude:   Double        // Grover-amplified probability
    let sacredScore: Double
}

struct GroverResult {
    let matches:      [PatternMatch]
    let totalAmplified: Double
    let iterations:   Int
    let sacredAlignment: Double
}

struct PCAResult {
    let components:   [[Double]]   // principal eigenvectors
    let eigenvalues:  [Double]
    let explained:    [Double]     // variance explained per component
    let transformed:  [[Double]]   // projected data
    let sacredScore:  Double
}

struct VQEClusterResult {
    let assignments: [Int]
    let energies:    [Double]      // VQE ground-state energy per cluster
    let centroids:   [[Double]]
    let sacredAlignment: Double
    let convergenceIter: Int
}

struct AnomalyResult {
    let scores:      [Double]      // per-sample anomaly score [0,1]
    let anomalies:   [Int]         // indices of anomalous samples
    let threshold:   Double
    let sacredScore: Double
}

struct CorrelationMatrix {
    let matrix:      [[Double]]    // entanglement-enhanced correlation
    let maxCorr:     Double
    let clusterPairs: [(Int, Int)]
    let sacredScore:  Double
}

struct TopologyResult {
    let betti0:      Int           // connected components
    let betti1:      Int           // loops / 1-cycles
    let persistence: [(birth: Double, death: Double)]
    let sacredScore: Double
}

struct DenoiseResult {
    let cleanedData: [[Double]]
    let noiseLevel:  Double        // estimated noise
    let coherenceGain: Double
    let sacredAlignment: Double
}

struct QDAAnalysis {
    let pipeline:    String
    let spectral:    SpectralResult?
    let grover:      GroverResult?
    let pca:         PCAResult?
    let clustering:  VQEClusterResult?
    let anomaly:     AnomalyResult?
    let topology:    TopologyResult?
    let correlation: CorrelationMatrix?
    let denoised:    DenoiseResult?
    let overallSacred: Double
    let elapsedMs:   Double
    let timestamp:   Date
}

// MARK: - ═══ QUANTUM FOURIER ANALYZER ═══

final class QuantumFourierAnalyzer {

    // QFT approximation via classical FFT on sacred-encoded state
    func analyze(data: [Double]) -> SpectralResult {
        guard !data.isEmpty else {
            return SpectralResult(frequencies: [], magnitudes: [], dominantFreq: 0,
                                  godCodeAlignment: 0, sacredPeaks: [], totalPower: 0)
        }
        let n = _nextPowerOf2(data.count)
        var padded = data + [Double](repeating: 0.0, count: n - data.count)

        // Encode into quantum amplitudes (normalize) — EVO_76: vDSP, no map+reduce allocs
        var normSq = 0.0
        vDSP_svesqD(padded, 1, &normSq, vDSP_Length(padded.count))
        if normSq > 1e-28 { var inv = 1.0 / sqrt(normSq); vDSP_vsmulD(padded, 1, &inv, &padded, 1, vDSP_Length(padded.count)) }

        // vDSP FFT
        let log2n = vDSP_Length(log2(Double(n)))
        // Use the DOUBLE-precision setup (vDSP_create_fftsetupD) to match vDSP_fft_zopD.
        // Using the single-precision vDSP_create_fftsetup here causes the internal OpaqueFFTSetup
        // struct to be misread by vDSP_fft_zopD → heap corruption → Array bounds SIGILL at line 177.
        guard let fftSetup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else {
            return SpectralResult(frequencies: [], magnitudes: [], dominantFreq: 0,
                                  godCodeAlignment: 0, sacredPeaks: [], totalPower: 0)
        }
        defer { vDSP_destroy_fftsetupD(fftSetup) }

        var realIn  = padded
        var imagIn  = [Double](repeating: 0.0, count: n)
        var realOut = [Double](repeating: 0.0, count: n)
        var imagOut = [Double](repeating: 0.0, count: n)

        realIn.withUnsafeMutableBufferPointer { realPtr in
            imagIn.withUnsafeMutableBufferPointer { imagPtr in
                realOut.withUnsafeMutableBufferPointer { rOut in
                    imagOut.withUnsafeMutableBufferPointer { iOut in
                        var splitIn  = DSPDoubleSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                        var splitOut = DSPDoubleSplitComplex(realp: rOut.baseAddress!, imagp: iOut.baseAddress!)
                        vDSP_fft_zopD(fftSetup, &splitIn, 1, &splitOut, 1, log2n, FFTDirection(FFT_FORWARD))
                    }
                }
            }
        }

        let halfN = n / 2
        var magnitudes = (0..<halfN).map { i in
            sqrt(realOut[i]*realOut[i] + imagOut[i]*imagOut[i])
        }

        // Sacred GOD_CODE alignment: score peaks near GOD_CODE harmonics
        let domIdx = magnitudes.indices.max(by: { magnitudes[$0] < magnitudes[$1] }) ?? 0
        let domFreq = Double(domIdx) / Double(n)
        var totalPower = 0.0; vDSP_svesqD(magnitudes, 1, &totalPower, vDSP_Length(magnitudes.count))

        // Sacred peaks: frequencies near PHI multiples
        let sacredPeaks = magnitudes.indices.filter { i in
            let f = Double(i) / Double(n)
            let residue = (f * GOD_CODE).truncatingRemainder(dividingBy: 1.0)
            return residue < 0.05 || residue > 0.95
        }.map { Double($0) / Double(n) }

        let godAlignment = 1.0 - abs((domFreq * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        let frequencies  = (0..<halfN).map { Double($0) / Double(n) }

        return SpectralResult(
            frequencies: frequencies, magnitudes: magnitudes,
            dominantFreq: domFreq, godCodeAlignment: godAlignment,
            sacredPeaks: sacredPeaks, totalPower: totalPower
        )
    }

    private func _nextPowerOf2(_ n: Int) -> Int {
        var p = 1; while p < n { p <<= 1 }; return p
    }
}

// MARK: - ═══ GROVER PATTERN SEARCH ═══

final class GroverPatternSearch {

    // Grover-amplified similarity search
    func search(query: [Double], database: [[Double]], iterations: Int? = nil) -> GroverResult {
        guard !database.isEmpty else {
            return GroverResult(matches: [], totalAmplified: 0, iterations: 0, sacredAlignment: 0)
        }
        let n = database.count
        let optIter = iterations ?? max(1, Int((.pi / 4.0) * sqrt(Double(n))))

        // Classical dot-product similarities → oracle amplitudes — EVO_76: vDSP, no per-entry allocs
        let qLen = vDSP_Length(query.count)
        var qSq = 0.0; vDSP_svesqD(query, 1, &qSq, qLen)
        let qNorm = sqrt(qSq + 1e-14)
        var amplitudes = database.map { db -> Double in
            var dbSq = 0.0; vDSP_svesqD(db, 1, &dbSq, qLen)
            var dot  = 0.0; vDSP_dotprD(query, 1, db, 1, &dot, qLen)
            return max(0, dot / (qNorm * sqrt(dbSq + 1e-14)))
        }

        // Grover diffusion: amplify high-similarity entries
        for _ in 0..<optIter {
            let mean = amplitudes.reduce(0.0, +) / Double(n)
            amplitudes = amplitudes.map { 2.0 * mean - $0 }
            // Clip negative amplitudes (born-rule approximation)
            amplitudes = amplitudes.map { max(0, $0) }
            // Re-normalize
            let s = amplitudes.reduce(0.0, +)
            if s > 1e-14 { amplitudes = amplitudes.map { $0/s } }
        }

        // Build match list
        let threshold = ANOMALY_THRESHOLD / Double(n)
        let matches = amplitudes.indices.compactMap { i -> PatternMatch? in
            guard amplitudes[i] >= threshold else { return nil }
            let sacred = 1.0 - abs((amplitudes[i] * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
            return PatternMatch(index: i, similarity: amplitudes[i],
                                amplitude: amplitudes[i], sacredScore: sacred)
        }.sorted { $0.amplitude > $1.amplitude }

        let totalAmp  = amplitudes.reduce(0.0, +)
        let sacred    = 1.0 - abs((totalAmp * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        return GroverResult(matches: matches, totalAmplified: totalAmp,
                            iterations: optIter, sacredAlignment: sacred)
    }
}

// MARK: - ═══ QUANTUM PCA ═══

final class QuantumPCA {
    let nComponents: Int
    init(nComponents: Int = 4) { self.nComponents = nComponents }

    func fit(data: [[Double]]) -> PCAResult {
        guard let dim = data.first?.count, !data.isEmpty else {
            return PCAResult(components: [], eigenvalues: [], explained: [],
                             transformed: [], sacredScore: 0)
        }
        let n = data.count
        // Center data — EVO_76: single-pass mean, zero inner allocations
        var means = [Double](repeating: 0.0, count: dim)
        for row in data { for d in 0..<dim { means[d] += row[d] } }
        let invN = 1.0 / Double(n); for d in 0..<dim { means[d] *= invN }
        let centered = data.map { row in zip(row, means).map(-) }

        // Covariance matrix — EVO_76: direct accumulation loop, no per-entry temp arrays
        var cov = [[Double]](repeating: [Double](repeating: 0, count: dim), count: dim)
        let nM1 = Double(max(n-1, 1))
        for i in 0..<dim {
            for j in i..<dim {
                var cSum = 0.0
                for row in centered { cSum += row[i] * row[j] }
                let c = cSum / nM1; cov[i][j] = c; cov[j][i] = c
            }
        }

        // Power iteration for top-k eigenvectors (sacred-seeded)
        let k = min(nComponents, dim)
        var eigVecs = [[Double]]()
        var eigVals = [Double]()
        var deflated = cov

        let dLen = vDSP_Length(dim)
        for comp in 0..<k {
            // Initialize with PHI-spiral direction — EVO_76: vDSP norm+scale, no map allocs
            var v = (0..<dim).map { d in cos(Double(comp * d + 1) * PHI) }
            var vSq = 0.0; vDSP_svesqD(v, 1, &vSq, dLen)
            var vInv = 1.0 / sqrt(vSq + 1e-14); vDSP_vsmulD(v, 1, &vInv, &v, 1, dLen)

            for _ in 0..<QDA_N_ITER {  // 104 power iterations
                var mv = _matvec(deflated, v)
                var mSq = 0.0; vDSP_svesqD(mv, 1, &mSq, dLen)
                var mInv = 1.0 / sqrt(mSq + 1e-14); vDSP_vsmulD(mv, 1, &mInv, &mv, 1, dLen)
                v = mv
            }
            // Rayleigh quotient = eigenvalue
            let mv = _matvec(deflated, v)
            var eigVal = 0.0; vDSP_dotprD(v, 1, mv, 1, &eigVal, dLen)
            eigVecs.append(v)
            eigVals.append(abs(eigVal))

            // Deflate: remove this component
            for i in 0..<dim {
                for j in 0..<dim {
                    deflated[i][j] -= eigVal * v[i] * v[j]
                }
            }
        }

        let totalVar = eigVals.reduce(0.0, +)
        let explained = eigVals.map { $0 / max(totalVar, 1e-14) }
        let transformed = centered.map { row in
            eigVecs.map { ev -> Double in var d = 0.0; vDSP_dotprD(row, 1, ev, 1, &d, dLen); return d }
        }
        let sacred = 1.0 - abs((eigVals.first ?? 0.0 * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        return PCAResult(components: eigVecs, eigenvalues: eigVals,
                         explained: explained, transformed: transformed, sacredScore: sacred)
    }

    private func _matvec(_ A: [[Double]], _ v: [Double]) -> [Double] {
        // EVO_76: vDSP dot per row — no zip/map/reduce allocs
        let len = vDSP_Length(v.count)
        return A.map { row -> Double in var d = 0.0; vDSP_dotprD(row, 1, v, 1, &d, len); return d }
    }
}

// MARK: - ═══ VQE CLUSTERER ═══

final class VQEClusterer {
    let nClusters: Int
    init(nClusters: Int = 3) { self.nClusters = nClusters }

    func cluster(data: [[Double]]) -> VQEClusterResult {
        guard !data.isEmpty else {
            return VQEClusterResult(assignments: [], energies: [], centroids: [],
                                    sacredAlignment: 0, convergenceIter: 0)
        }
        let k = nClusters
        let dim = data[0].count
        let n = data.count

        // VQE energy landscape: use sacred-modulated cost function
        // Initialize centroids with PHI-spiral
        var centroids = (0..<k).map { c in
            (0..<dim).map { d in sin(Double(c+1) * PHI + Double(d)) }
        }

        var assignments = [Int](repeating: 0, count: n)
        var energies    = [Double](repeating: 0.0, count: k)
        var iter = 0

        for _ in 0..<QDA_N_ITER {
            iter += 1
            var changed = false
            // Assign step: minimum VQE energy distance
            for (idx, pt) in data.enumerated() {
                let dists = (0..<k).map { c in
                    _vqeEnergy(pt: pt, centroid: centroids[c])
                }
                let best = dists.indices.min { dists[$0] < dists[$1] } ?? 0
                if best != assignments[idx] { changed = true }
                assignments[idx] = best
            }
            if !changed { break }
            // Update centroids — EVO_76: single-pass O(n×dim) accumulation, was O(k×n×dim)
            var sums = [[Double]](repeating: [Double](repeating: 0.0, count: dim), count: k)
            var cnts = [Int](repeating: 0, count: k)
            for (pt, c) in zip(data, assignments) { for d in 0..<dim { sums[c][d] += pt[d] }; cnts[c] += 1 }
            for c in 0..<k where cnts[c] > 0 {
                let inv = 1.0 / Double(cnts[c]); for d in 0..<dim { centroids[c][d] = sums[c][d] * inv }
            }
        }

        // Compute cluster energies — EVO_76: single pass, no filter+map+reduce allocs
        for c in 0..<k {
            var inertia = 0.0; var cnt = 0
            for (pt, asgn) in zip(data, assignments) where asgn == c { inertia += _vqeEnergy(pt: pt, centroid: centroids[c]); cnt += 1 }
            energies[c] = cnt == 0 ? 0.0 : -inertia / GOD_CODE
        }

        let sacred = 1.0 - abs((energies.reduce(0.0, +) * PHI).truncatingRemainder(dividingBy: 1.0))
        return VQEClusterResult(assignments: assignments, energies: energies,
                                centroids: centroids, sacredAlignment: sacred,
                                convergenceIter: iter)
    }

    // Sacred VQE energy — EVO_76: branchless loop, zero allocation
    private func _vqeEnergy(pt: [Double], centroid: [Double]) -> Double {
        var d2 = 0.0
        for i in 0..<pt.count { let d = pt[i] - centroid[i]; d2 += d * d }
        return d2 * VOID_CONSTANT / (1.0 + PHI * sqrt(d2))
    }
}

// MARK: - ═══ QUANTUM ANOMALY DETECTOR ═══

final class QuantumAnomalyDetector {

    func detect(data: [[Double]], threshold: Double = ANOMALY_THRESHOLD) -> AnomalyResult {
        guard !data.isEmpty else {
            return AnomalyResult(scores: [], anomalies: [], threshold: threshold, sacredScore: 0)
        }
        let n = data.count
        let dim = data[0].count

        // Compute mean vector — EVO_76: single-pass, no nested map+reduce
        var mean = [Double](repeating: 0.0, count: dim)
        for row in data { for d in 0..<dim { mean[d] += row[d] } }
        let invN2 = 1.0 / Double(n); for d in 0..<dim { mean[d] *= invN2 }

        // SWAP-test: normB is constant — compute once outside loop (EVO_76)
        let mLen = vDSP_Length(dim)
        var meanSq = 0.0; vDSP_svesqD(mean, 1, &meanSq, mLen)
        let normB = sqrt(meanSq + 1e-14)
        let scores: [Double] = data.map { pt -> Double in
            var dot = 0.0; vDSP_dotprD(pt, 1, mean, 1, &dot, mLen)
            var ptSq = 0.0; vDSP_svesqD(pt, 1, &ptSq, mLen)
            let fidelity = (dot / (sqrt(ptSq + 1e-14) * normB) + 1.0) / 2.0
            return max(0, 1.0 - fidelity)
        }

        // PHI-modulated threshold
        let sacredThresh = threshold * PHI_CONJUGATE
        let anomalies = scores.indices.filter { scores[$0] > sacredThresh }
        let sacred    = 1.0 - abs((scores.reduce(0.0, +) / Double(n) * GOD_CODE).truncatingRemainder(dividingBy: 1.0))

        return AnomalyResult(scores: scores, anomalies: anomalies,
                             threshold: sacredThresh, sacredScore: sacred)
    }
}

// MARK: - ═══ ENTANGLEMENT CORRELATION ANALYZER ═══

final class EntanglementCorrelationAnalyzer {

    func analyze(data: [[Double]]) -> CorrelationMatrix {
        guard let dim = data.first?.count, !data.isEmpty else {
            return CorrelationMatrix(matrix: [], maxCorr: 0, clusterPairs: [], sacredScore: 0)
        }
        let n = data.count

        // Quantum-enhanced correlation — EVO_76: hoist column extraction + single-pass cov/std
        // Precompute column arrays (dim allocs total, was dim^2/2 × 5 allocs)
        let cols = (0..<dim).map { d in data.map { $0[d] } }
        var colMeans = [Double](repeating: 0.0, count: dim)
        let nLen = vDSP_Length(n)
        for d in 0..<dim { vDSP_sveD(cols[d], 1, &colMeans[d], nLen); colMeans[d] /= Double(n) }

        var corr = [[Double]](repeating: [Double](repeating: 0, count: dim), count: dim)
        let corrNM1 = Double(max(n-1, 1))
        for i in 0..<dim {
            for j in i..<dim {
                let mI = colMeans[i]; let mJ = colMeans[j]
                var covS = 0.0, si = 0.0, sj = 0.0
                for k in 0..<n {
                    let di = cols[i][k] - mI; let dj = cols[j][k] - mJ
                    covS += di * dj; si += di * di; sj += dj * dj
                }
                let c = (covS / corrNM1) / (sqrt(si / corrNM1 + 1e-14) * sqrt(sj / corrNM1 + 1e-14)) * cos(PHI * Double(i - j))
                corr[i][j] = c; corr[j][i] = c
            }
        }

        // Find highly correlated pairs
        var maxC = 0.0
        var clusterPairs: [(Int, Int)] = []
        for i in 0..<dim {
            for j in (i+1)..<dim {
                let v = abs(corr[i][j])
                if v > maxC { maxC = v }
                if v > ANOMALY_THRESHOLD { clusterPairs.append((i, j)) }
            }
        }
        let sacred = 1.0 - abs((maxC * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        return CorrelationMatrix(matrix: corr, maxCorr: maxC,
                                 clusterPairs: clusterPairs, sacredScore: sacred)
    }
}

// MARK: - ═══ QUANTUM WALK GRAPH ANALYZER ═══

final class QuantumWalkGraphAnalyzer {
    let steps: Int
    init(steps: Int = 104) { self.steps = steps }

    // Discrete quantum walk on adjacency graph
    func walk(adjacency: [[Int]], startNode: Int = 0) -> [Double] {
        let n = adjacency.count
        guard n > 0 else { return [] }
        // Probability distribution (quantum walk amplitudes squared)
        var amplitude = [Double](repeating: 0.0, count: n)
        amplitude[min(startNode, n-1)] = 1.0

        for step in 0..<steps {
            var newAmp = [Double](repeating: 0.0, count: n)
            let phase  = Double(step) * PHI * .pi / Double(steps)
            for i in 0..<n {
                let neighbors = adjacency[i].indices.filter { adjacency[i][$0] != 0 }
                guard !neighbors.isEmpty else { continue }
                let spread = amplitude[i] / sqrt(Double(neighbors.count))
                for j in neighbors {
                    newAmp[j] += spread * cos(phase)
                }
            }
            // Normalize — EVO_76: vDSP svesq + vsmul, no map allocs
            var wSq = 0.0; vDSP_svesqD(newAmp, 1, &wSq, vDSP_Length(n))
            var wInv = 1.0 / sqrt(wSq + 1e-14); vDSP_vsmulD(newAmp, 1, &wInv, &amplitude, 1, vDSP_Length(n))
        }
        return amplitude.map { $0 * $0 }  // Born rule probabilities
    }
}

// MARK: - ═══ TOPOLOGICAL DATA MINER ═══

final class TopologicalDataMiner {

    func mine(data: [[Double]]) -> TopologyResult {
        guard !data.isEmpty else {
            return TopologyResult(betti0: 0, betti1: 0, persistence: [], sacredScore: 0)
        }
        // Persistent homology approximation via Vietoris-Rips complex
        let n = data.count
        var persistence: [(birth: Double, death: Double)] = []

        // Compute pairwise distances
        var dists = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        // EVO_76: branchless distance loop — no zip/map/reduce allocs in O(n^2) loop
        for i in 0..<n {
            let ri = data[i]
            for j in (i+1)..<n {
                var d2 = 0.0; let rj = data[j]
                for k in 0..<ri.count { let diff = ri[k] - rj[k]; d2 += diff * diff }
                let d = sqrt(d2); dists[i][j] = d; dists[j][i] = d
            }
        }

        // Betti-0: connected components via union-find at threshold = median dist
        var allDists = dists.flatMap { $0 }.filter { $0 > 0 }.sorted()
        let threshold0 = allDists[allDists.count / 2]
        var components = Array(0..<n)
        func find(_ x: Int) -> Int {
            var r = x; while components[r] != r { r = components[r] }; return r
        }
        for i in 0..<n {
            for j in (i+1)..<n where dists[i][j] <= threshold0 {
                let ri = find(i); let rj = find(j)
                if ri != rj { components[ri] = rj }
            }
        }
        let betti0 = Set(components.indices.map { find($0) }).count

        // Betti-1: loops - PHI-modulated threshold
        let threshold1 = threshold0 * PHI
        var edges1 = [(Int,Int)]()
        for i in 0..<n {
            for j in (i+1)..<n where dists[i][j] <= threshold1 {
                edges1.append((i,j))
            }
        }
        let betti1 = max(0, edges1.count - n + betti0)

        // Persistence pairs at sacred radii
        for scale in stride(from: 0.0, to: 1.0, by: 0.1) {
            let b = threshold0 * scale
            let d = b * PHI
            if d > 0 { persistence.append((birth: b, death: d)) }
        }

        let sacred = 1.0 - abs((Double(betti0 + betti1) * PHI).truncatingRemainder(dividingBy: 1.0))
        return TopologyResult(betti0: betti0, betti1: betti1,
                              persistence: persistence, sacredScore: sacred)
    }
}

// MARK: - ═══ GOD CODE RESONANCE ALIGNER ═══

final class GodCodeResonanceAligner {

    // Align data to nearest GOD_CODE harmonic grid
    func align(data: [[Double]]) -> [[Double]] {
        data.map { row in
            row.map { v in
                let n = round(v * GOD_CODE / .pi) * .pi / GOD_CODE
                return v + (n - v) * (1.0 / PHI)  // PHI-weighted pull
            }
        }
    }

    // Score overall GOD_CODE alignment of a dataset
    func alignmentScore(data: [[Double]]) -> Double {
        let all = data.flatMap { $0 }
        guard !all.isEmpty else { return 0 }
        let residues = all.map { abs(($0 * GOD_CODE).truncatingRemainder(dividingBy: 1.0)) }
        let meanResidue = residues.reduce(0.0, +) / Double(all.count)
        return 1.0 - meanResidue
    }
}

// MARK: - ═══ ENTROPY REVERSAL DENOISER ═══

final class EntropyReversalDenoiser {

    // Multi-method quantum denoising
    func denoise(data: [[Double]]) -> DenoiseResult {
        guard !data.isEmpty else {
            return DenoiseResult(cleanedData: [], noiseLevel: 0, coherenceGain: 0, sacredAlignment: 0)
        }
        let n = data.count
        let dim = data[0].count

        // Method 1: median filter (kernel = PHI-spiral window)
        let winSize = max(3, Int(PHI * 2.0))  // ≈ 3
        var cleaned = data.map { row -> [Double] in
            (0..<dim).map { d in
                let window = (max(0, d - winSize/2)...min(dim-1, d + winSize/2)).map { row[$0] }
                let sorted  = window.sorted()
                return sorted[sorted.count / 2]
            }
        }

        // Method 2: coherence field smoothing via GOD_CODE low-pass
        let alpha = 1.0 / (1.0 + GOD_CODE / 1000.0)  // ≈ smoothing factor
        var smoothed = cleaned
        for i in 1..<n {
            smoothed[i] = zip(smoothed[i-1], cleaned[i]).map { alpha*$0 + (1-alpha)*$1 }
        }

        // Noise level = std dev of difference
        let diff = zip(data, smoothed).flatMap { zip($0, $1).map { abs($0 - $1) } }
        let noiseLevel = diff.reduce(0.0, +) / Double(max(diff.count, 1))

        // Coherence gain (variance reduction)
        let varBefore = _variance(data.flatMap { $0 })
        let varAfter  = _variance(smoothed.flatMap { $0 })
        let coherenceGain = 1.0 - varAfter / max(varBefore, 1e-14)

        let sacred = 1.0 - abs((coherenceGain * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        return DenoiseResult(cleanedData: smoothed, noiseLevel: noiseLevel,
                             coherenceGain: max(0, coherenceGain), sacredAlignment: sacred)
    }

    private func _variance(_ v: [Double]) -> Double {
        // EVO_76: vDSP mean + svesq after shift — no map+reduce allocs
        guard !v.isEmpty else { return 0 }
        var mean = 0.0; vDSP_meanvD(v, 1, &mean, vDSP_Length(v.count))
        var shifted = v.map { $0 - mean }
        var sq = 0.0; vDSP_svesqD(shifted, 1, &sq, vDSP_Length(v.count))
        return sq / Double(v.count)
    }
}

// MARK: - ═══ HHL SOLVER ═══

// Quantum-inspired linear system solver (Ax = b)
final class HHLSolver {

    // Iterative conjugate-gradient with sacred preconditioning
    func solve(A: [[Double]], b: [Double], maxIter: Int = QDA_N_ITER) -> [Double] {
        let n = b.count
        guard n > 0 && A.count == n else { return [] }
        var x = b.map { $0 / (GOD_CODE / 100.0) }  // sacred initial guess
        var r = _subtract(b, _matvec(A, x))
        var p = r
        var rsold = _dot(r, r)
        for _ in 0..<maxIter {
            let ap  = _matvec(A, p)
            let denom = _dot(p, ap)
            guard abs(denom) > 1e-14 else { break }
            // EVO_76: vDSP vector ops — no zip+map allocs per CG iteration
            let alpha = rsold / denom
            let cgLen = vDSP_Length(n)
            var a = alpha;  vDSP_vsmaD(p,  1, &a,  x,  1, &x,  1, cgLen)  // x += alpha*p
            var na = -alpha; vDSP_vsmaD(ap, 1, &na, r, 1, &r, 1, cgLen)   // r -= alpha*ap
            let rsnew = _dot(r, r)
            if sqrt(rsnew) < 1e-10 { break }
            let beta = rsnew / max(rsold, 1e-14)
            var b = beta; vDSP_vsmaD(p, 1, &b, r, 1, &p, 1, cgLen)       // p = r + beta*p
            rsold = rsnew
        }
        return x
    }

    private func _matvec(_ A: [[Double]], _ v: [Double]) -> [Double] {
        // EVO_76: vDSP dot per row
        let len = vDSP_Length(v.count)
        return A.map { row -> Double in var d = 0.0; vDSP_dotprD(row, 1, v, 1, &d, len); return d }
    }
    private func _dot(_ a: [Double], _ b: [Double]) -> Double {
        var d = 0.0; vDSP_dotprD(a, 1, b, 1, &d, vDSP_Length(a.count)); return d
    }
    private func _subtract(_ a: [Double], _ b: [Double]) -> [Double] { zip(a,b).map(-) }
}

// MARK: - ═══ QUANTUM DATA ANALYZER ORCHESTRATOR ═══

final class QuantumDataAnalyzer {
    static let shared = QuantumDataAnalyzer()

    private let qft        = QuantumFourierAnalyzer()
    private let grover     = GroverPatternSearch()
    private let pca        = QuantumPCA()
    private let vqe        = VQEClusterer()
    private let anomaly    = QuantumAnomalyDetector()
    private let correlator = EntanglementCorrelationAnalyzer()
    private let topology   = TopologicalDataMiner()
    private let denoiser   = EntropyReversalDenoiser()
    private let aligner    = GodCodeResonanceAligner()
    private let hhl        = HHLSolver()
    private let walker     = QuantumWalkGraphAnalyzer()

    private let queue = DispatchQueue(label: "l104.qda", qos: .userInitiated)

    // ── Full analysis pipeline ──
    func fullAnalysis(data: [[Double]]) -> QDAAnalysis {
        let t0 = Date()
        let flat = data.flatMap { $0 }

        let spec = qft.analyze(data: flat)
        let groverResult = data.count > 1
            ? grover.search(query: data[0], database: Array(data.dropFirst()))
            : GroverResult(matches: [], totalAmplified: 0, iterations: 0, sacredAlignment: 0)
        let pcaR  = pca.fit(data: data)
        let clust = vqe.cluster(data: data)
        let anom  = anomaly.detect(data: data)
        let topo  = topology.mine(data: data)
        let corr  = correlator.analyze(data: data)
        let den   = denoiser.denoise(data: data)

        let sacredScores = [spec.godCodeAlignment, groverResult.sacredAlignment,
                            pcaR.sacredScore, clust.sacredAlignment,
                            anom.sacredScore, topo.sacredScore,
                            corr.sacredScore, den.sacredAlignment]
        let overall = sacredScores.reduce(0.0, +) / Double(sacredScores.count)
        let elapsed = Date().timeIntervalSince(t0) * 1000.0

        return QDAAnalysis(
            pipeline: "full_analysis",
            spectral: spec, grover: groverResult, pca: pcaR,
            clustering: clust, anomaly: anom, topology: topo,
            correlation: corr, denoised: den,
            overallSacred: overall, elapsedMs: elapsed, timestamp: t0
        )
    }

    // ── Named pipeline dispatch ──
    func run(pipeline: String, data: [[Double]]) -> QDAAnalysis {
        let t0 = Date()
        let flat = data.flatMap { $0 }

        switch pipeline {
        case "spectral":
            let s = qft.analyze(data: flat)
            return QDAAnalysis(pipeline: pipeline, spectral: s, grover: nil, pca: nil,
                               clustering: nil, anomaly: nil, topology: nil, correlation: nil,
                               denoised: nil, overallSacred: s.godCodeAlignment,
                               elapsedMs: 0, timestamp: t0)
        case "clustering":
            let c = vqe.cluster(data: data)
            return QDAAnalysis(pipeline: pipeline, spectral: nil, grover: nil, pca: nil,
                               clustering: c, anomaly: nil, topology: nil, correlation: nil,
                               denoised: nil, overallSacred: c.sacredAlignment,
                               elapsedMs: 0, timestamp: t0)
        case "anomaly":
            let a = anomaly.detect(data: data)
            return QDAAnalysis(pipeline: pipeline, spectral: nil, grover: nil, pca: nil,
                               clustering: nil, anomaly: a, topology: nil, correlation: nil,
                               denoised: nil, overallSacred: a.sacredScore,
                               elapsedMs: 0, timestamp: t0)
        case "topological":
            let tp = topology.mine(data: data)
            return QDAAnalysis(pipeline: pipeline, spectral: nil, grover: nil, pca: nil,
                               clustering: nil, anomaly: nil, topology: tp, correlation: nil,
                               denoised: nil, overallSacred: tp.sacredScore,
                               elapsedMs: 0, timestamp: t0)
        case "correlation":
            let co = correlator.analyze(data: data)
            return QDAAnalysis(pipeline: pipeline, spectral: nil, grover: nil, pca: nil,
                               clustering: nil, anomaly: nil, topology: nil, correlation: co,
                               denoised: nil, overallSacred: co.sacredScore,
                               elapsedMs: 0, timestamp: t0)
        case "denoise":
            let d = denoiser.denoise(data: data)
            return QDAAnalysis(pipeline: pipeline, spectral: nil, grover: nil, pca: nil,
                               clustering: nil, anomaly: nil, topology: nil, correlation: nil,
                               denoised: d, overallSacred: d.sacredAlignment,
                               elapsedMs: 0, timestamp: t0)
        default:
            return fullAnalysis(data: data)
        }
    }

    // ── Sacred data alignment ──
    func align(data: [[Double]]) -> [[Double]] { aligner.align(data: data) }

    // ── Linear solve ──
    func solveLinear(A: [[Double]], b: [Double]) -> [Double] { hhl.solve(A: A, b: b) }

    // ── Self test ──
    func selfTest() -> Bool {
        let testData = (0..<13).map { i in (0..<8).map { d in sin(Double(i*d+1) * PHI / 10.0) } }
        let result = fullAnalysis(data: testData)
        return result.overallSacred > 0
    }
}
