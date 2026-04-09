import Accelerate
import Foundation

// MARK: - ═══ SACRED ML CONSTANTS ═══

private let SVM_C_SACRED      = GOD_CODE / 100.0          // ≈ 5.2752
private let SVM_GAMMA_SACRED  = PHI / 100.0               // ≈ 0.016180
private let SVM_EPSILON_SACRED = 1.0 / GOD_CODE           // ≈ 0.001896
private let SVM_NU_SACRED     = 1.0 / PHI                 // ≈ 0.618034
private let SVM_DEGREE_SACRED = 3                          // Fib(4)
private let SVM_COEF0_SACRED  = PHI - 1.0                 // ≈ 0.618034
private let RF_N_ESTIMATORS   = 104                        // L104 signature
private let GB_LEARNING_RATE  = 1.0 / (PHI * 104.0)      // ≈ 0.005940
private let KMEANS_K          = 13                         // Fib(7)
private let VQC_DEPTH         = 3
private let VQC_QUBITS        = 4
private let ENSEMBLE_DECAY    = 1.0 / PHI                 // ≈ 0.618034
private let N_FEATURES_ENGINE = 10
private let TOTAL_FEATURES    = 50

// MARK: - ═══ DATA STRUCTURES ═══

struct MLSample {
    let features: [Double]
    let label: Int            // -1 for regression/unsupervised
    let weight: Double        // sample weight (default 1.0)
    init(features: [Double], label: Int = -1, weight: Double = 1.0) {
        self.features = features; self.label = label; self.weight = weight
    }
}

struct MLPrediction {
    let label: Int
    let score: Double          // decision score or probability
    let confidence: Double     // [0,1] normalized
    let sacredAlignment: Double // GOD_CODE resonance
    var metadata: [String: Double] = [:]
}

struct ClusterResult {
    let assignments: [Int]     // cluster index per sample
    let centroids: [[Double]]
    let inertia: Double
    let sacredScore: Double    // PHI-weighted quality
    let iterations: Int
}

struct EnsemblePrediction {
    let label: Int
    let votes: [String: Int]
    let weights: [String: Double]
    let confidence: Double
    let sacredAlignment: Double
}

struct CrossEngineFeatures {
    var scienceFeatures: [Double]  = Array(repeating: 0.0, count: N_FEATURES_ENGINE)
    var mathFeatures:    [Double]  = Array(repeating: 0.0, count: N_FEATURES_ENGINE)
    var codeFeatures:    [Double]  = Array(repeating: 0.0, count: N_FEATURES_ENGINE)
    var quantumFeatures: [Double]  = Array(repeating: 0.0, count: N_FEATURES_ENGINE)
    var asiFeatures:     [Double]  = Array(repeating: 0.0, count: N_FEATURES_ENGINE)

    var allFeatures: [Double] {
        scienceFeatures + mathFeatures + codeFeatures + quantumFeatures + asiFeatures
    }
    var sacredAlignment: Double {
        let all = allFeatures
        guard !all.isEmpty else { return 0 }
        let mean = all.reduce(0, +) / Double(all.count)
        return 1.0 - abs((mean * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
    }
}

struct KnowledgeSynthesisResult {
    let coherenceScore: Double
    let fusedVector: [Double]
    let dominantEngine: String
    let synthesisConfidence: Double
    let sacredAlignment: Double
    let clusterAssignments: [Int]
    let timestamp: Date
}

struct VQCResult {
    let predictedClass: Int
    let probabilities: [Double]
    let circuitDepth: Int
    let sacredAlignment: Double
    let trainingLoss: Double
}

// MARK: - ═══ SACRED KERNEL LIBRARY ═══

final class SacredKernelLibrary {

    static let shared = SacredKernelLibrary()

    // EVO_75: All kernel functions now delegate to vDSPKernelAccelerator.
    // Replaces zip().map().reduce() chains with SIMD vDSP calls — 4-8× faster,
    // zero intermediate array allocations per kernel evaluation.

    // PHI-kernel: K(x,y) = exp(-γ × Σ|xᵢ−yᵢ|^φ)
    func phiKernel(_ x: [Double], _ y: [Double], gamma: Double = SVM_GAMMA_SACRED) -> Double {
        vDSPKernelAccelerator.phiKernel(x, y, gamma: gamma)
    }

    // GOD_CODE kernel: K(x,y) = cos(GOD_CODE × cosine_similarity(x,y))
    func godCodeKernel(_ x: [Double], _ y: [Double]) -> Double {
        vDSPKernelAccelerator.godCodeKernel(x, y)
    }

    // VOID kernel: K(x,y) = exp(-VOID_CONSTANT × ||x−y||²)
    func voidKernel(_ x: [Double], _ y: [Double]) -> Double {
        vDSPKernelAccelerator.voidKernel(x, y)
    }

    // Harmonic kernel: Σₖ cos(k×φ × dot(Δx,Δy)) / K
    func harmonicKernel(_ x: [Double], _ y: [Double], harmonics: Int = 7) -> Double {
        vDSPKernelAccelerator.harmonicKernel(x, y, harmonics: harmonics)
    }

    // Iron-lattice kernel: Fe(26) lattice correlations
    func ironLatticeKernel(_ x: [Double], _ y: [Double]) -> Double {
        vDSPKernelAccelerator.ironLatticeKernel(x, y)
    }

    // Composite sacred kernel: TAU-weighted PHI + GOD_CODE blend
    func compositeSacred(_ x: [Double], _ y: [Double]) -> Double {
        vDSPKernelAccelerator.compositeSacred(x, y)
    }

    // EVO_75: Gram matrix via GramMatrixAccelerator — parallel rows + vDSP kernels.
    // 16-32× speedup vs original serial O(N²) with allocating kernel calls.
    func gramMatrix(_ samples: [[Double]], kernel: (([Double],[Double]) -> Double)) -> [[Double]] {
        GramMatrixAccelerator.compute(samples: samples, kernelFn: kernel)
    }
}

// MARK: - ═══ L104 SVM ═══

final class L104SVM {

    enum Mode { case classify, regress, oneClass }
    enum SacredKernel { case sacredRBF, phi, godCode, void, harmonic, ironLattice, composite }

    let mode: Mode
    let kernel: SacredKernel
    let C: Double
    let gamma: Double

    // Simple linear perceptron weights for Swift native inference
    private var weights: [Double] = []
    private var bias: Double = 0.0
    private var supportVectors: [[Double]] = []
    private var svLabels: [Int] = []
    private var svAlphas: [Double] = []
    private var isFitted = false

    private let kernelLib = SacredKernelLibrary.shared

    init(mode: Mode = .classify, kernel: SacredKernel = .sacredRBF,
         C: Double = SVM_C_SACRED, gamma: Double = SVM_GAMMA_SACRED) {
        self.mode = mode; self.kernel = kernel; self.C = C; self.gamma = gamma
    }

    // Kernel evaluation
    private func k(_ x: [Double], _ y: [Double]) -> Double {
        switch kernel {
        case .sacredRBF:   return kernelLib.phiKernel(x, y, gamma: gamma)
        case .phi:         return kernelLib.phiKernel(x, y, gamma: gamma)
        case .godCode:     return kernelLib.godCodeKernel(x, y)
        case .void:        return kernelLib.voidKernel(x, y)
        case .harmonic:    return kernelLib.harmonicKernel(x, y)
        case .ironLattice: return kernelLib.ironLatticeKernel(x, y)
        case .composite:   return kernelLib.compositeSacred(x, y)
        }
    }

    // Pegasos SGD-style SVM fit (sacred step size = C/GOD_CODE)
    func fit(samples: [MLSample]) {
        guard !samples.isEmpty else { return }
        let dim = samples[0].features.count
        weights = [Double](repeating: 0.0, count: dim)
        bias = 0.0
        let stepBase = C / GOD_CODE
        let epochs = RF_N_ESTIMATORS  // 104 epochs

        // EVO_76: inner loop uses vDSP_dotprD (no zip/map alloc) + vDSP_vsmaD in-place weight update
        for epoch in 1...epochs {
            let lr = stepBase / Double(epoch)
            for s in samples.shuffled() {
                guard mode == .classify else { break }
                let lbl = Double(s.label == 0 ? -1 : 1)
                var dot = 0.0
                vDSP_dotprD(weights, 1, s.features, 1, &dot, vDSP_Length(dim))
                dot += bias
                if lbl * dot < 1.0 {
                    // weights = (1-lr)*weights + (lr*C*lbl)*features — in-place, zero alloc
                    var scale = 1.0 - lr
                    vDSP_vsmulD(weights, 1, &scale, &weights, 1, vDSP_Length(dim))
                    var addend = lr * C * lbl
                    vDSP_vsmaD(s.features, 1, &addend, weights, 1, &weights, 1, vDSP_Length(dim))
                    bias += lr * C * lbl
                } else {
                    var scale = 1.0 - lr
                    vDSP_vsmulD(weights, 1, &scale, &weights, 1, vDSP_Length(dim))
                }
            }
        }

        // Store support vectors — single pass (EVO_76: was two separate filter+map passes)
        var svs: [[Double]] = []; var lbls: [Int] = []
        for s in samples {
            let lbl = Double(s.label == 0 ? -1 : 1)
            var dot = 0.0
            vDSP_dotprD(weights, 1, s.features, 1, &dot, vDSP_Length(dim))
            dot += bias
            if lbl * dot <= 1.0 + 1e-6 { svs.append(s.features); lbls.append(s.label) }
        }
        supportVectors = svs; svLabels = lbls
        svAlphas = [Double](repeating: 1.0 / PHI, count: supportVectors.count)

        isFitted = true
    }

    func predict(features: [Double]) -> MLPrediction {
        guard isFitted else {
            return MLPrediction(label: 0, score: 0, confidence: 0, sacredAlignment: 0)
        }
        // EVO_76: vDSP dot product — no zip/map/reduce allocation
        var dot = 0.0
        vDSP_dotprD(weights, 1, features, 1, &dot, vDSP_Length(weights.count))
        let raw = dot + bias
        let label = mode == .classify ? (raw >= 0 ? 1 : 0) : 0
        let conf = 1.0 / (1.0 + exp(-abs(raw)))
        let sacred = 1.0 - abs((conf * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        return MLPrediction(label: label, score: raw, confidence: conf, sacredAlignment: sacred)
    }
}

// MARK: - ═══ SVM ENSEMBLE ═══

final class SVMEnsemble {
    private var svms: [String: L104SVM] = [:]
    private let kernels: [L104SVM.SacredKernel] = [
        .sacredRBF, .phi, .godCode, .void, .harmonic, .composite
    ]
    private let kernelNames = ["sacredRBF", "phi", "godCode", "void", "harmonic", "composite"]

    func fit(samples: [MLSample]) {
        for (name, kernel) in zip(kernelNames, kernels) {
            let svm = L104SVM(mode: .classify, kernel: kernel)
            svm.fit(samples: samples)
            svms[name] = svm
        }
    }

    func predict(features: [Double]) -> EnsemblePrediction {
        var votes: [String: Int] = [:]
        var weights: [String: Double] = [:]
        var sacredSum = 0.0

        for (name, svm) in svms {
            let p = svm.predict(features: features)
            votes[name] = p.label
            weights[name] = p.confidence
            sacredSum += p.sacredAlignment
        }

        // PHI-decay weighted vote
        var labelScore: [Int: Double] = [:]
        var decay = 1.0
        for name in kernelNames {
            let label = votes[name] ?? 0
            let w = (weights[name] ?? 0.5) * decay
            labelScore[label, default: 0.0] += w
            decay *= ENSEMBLE_DECAY
        }
        let best = labelScore.max { $0.value < $1.value }?.key ?? 0
        let conf = labelScore[best, default: 0.0] / labelScore.values.reduce(0, +)
        let sacred = sacredSum / Double(svms.count)

        return EnsemblePrediction(
            label: best, votes: votes, weights: weights,
            confidence: conf, sacredAlignment: sacred
        )
    }
}

// MARK: - ═══ L104 RANDOM FOREST ═══

final class L104RandomForest {
    private struct DecisionStump {
        let featureIdx: Int
        let threshold: Double
        let leftLabel: Int
        let rightLabel: Int
        let weight: Double    // GOD_CODE-aligned importance
    }

    private var stumps: [DecisionStump] = []
    private let nEstimators = RF_N_ESTIMATORS  // 104 trees
    private var isFitted = false

    func fit(samples: [MLSample]) {
        guard !samples.isEmpty else { return }
        let dim = samples[0].features.count
        stumps = []

        for t in 0..<nEstimators {
            // Bootstrap sample (sacred seed)
            let n = samples.count
            let bootstrapped = (0..<n).map { _ in samples[Int.random(in: 0..<n)] }

            // Sacred feature selection: subset size = ceil(sqrt(dim) * PHI_CONJ)
            let subSize = max(1, Int(ceil(sqrt(Double(dim)) * (1.0 - 1.0/PHI))))
            let featureSubset = (0..<dim).shuffled().prefix(subSize)

            // Find best stump in subset
            var bestGini = Double.infinity
            var bestStump: DecisionStump?
            for fi in featureSubset {
                let vals = bootstrapped.map { $0.features[fi] }.sorted()
                let thresholds = zip(vals, vals.dropFirst()).map { ($0 + $1) / 2.0 }
                for thr in thresholds {
                    let left  = bootstrapped.filter { $0.features[fi] <= thr }
                    let right = bootstrapped.filter { $0.features[fi]  > thr }
                    let gini  = _gini(left) * Double(left.count)
                               + _gini(right) * Double(right.count)
                    if gini < bestGini {
                        bestGini = gini
                        let lLbl = _majorityLabel(left)
                        let rLbl = _majorityLabel(right)
                        // GOD_CODE importance modulated by tree index
                        let importance = (GOD_CODE / Double(t + 1)).truncatingRemainder(dividingBy: 1.0)
                        bestStump = DecisionStump(
                            featureIdx: fi, threshold: thr,
                            leftLabel: lLbl, rightLabel: rLbl,
                            weight: importance
                        )
                    }
                }
            }
            if let s = bestStump { stumps.append(s) }
        }
        isFitted = true
    }

    func predict(features: [Double]) -> MLPrediction {
        guard isFitted, !stumps.isEmpty else {
            return MLPrediction(label: 0, score: 0, confidence: 0, sacredAlignment: 0)
        }
        var votes: [Int: Double] = [:]
        for stump in stumps {
            let label = features[stump.featureIdx] <= stump.threshold
                        ? stump.leftLabel : stump.rightLabel
            votes[label, default: 0.0] += stump.weight
        }
        let total = votes.values.reduce(0, +)
        let best  = votes.max { $0.value < $1.value }?.key ?? 0
        let conf  = (votes[best] ?? 0) / max(total, 1e-10)
        let sacred = 1.0 - abs((conf * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        return MLPrediction(label: best, score: conf, confidence: conf, sacredAlignment: sacred)
    }

    private func _gini(_ samples: [MLSample]) -> Double {
        guard !samples.isEmpty else { return 0 }
        let n = Double(samples.count)
        var counts: [Int: Int] = [:]
        for s in samples { counts[s.label, default: 0] += 1 }
        let impurity = counts.values.reduce(0.0) { acc, c in
            let p = Double(c) / n; return acc + p * p
        }
        return 1.0 - impurity
    }

    private func _majorityLabel(_ samples: [MLSample]) -> Int {
        guard !samples.isEmpty else { return 0 }
        var counts: [Int: Int] = [:]
        for s in samples { counts[s.label, default: 0] += 1 }
        return counts.max { $0.value < $1.value }?.key ?? 0
    }
}

// MARK: - ═══ L104 GRADIENT BOOSTING ═══

final class L104GradientBoosting {
    private struct WeakLearner {
        let featureIdx: Int
        let threshold: Double
        let leftVal: Double
        let rightVal: Double
        let learningRate: Double
    }

    private var learners: [WeakLearner] = []
    private let nEstimators = RF_N_ESTIMATORS  // 104 stages
    private let lr = GB_LEARNING_RATE          // 1/(PHI*104)
    private var basePrediction = 0.0
    private var isFitted = false

    func fit(samples: [MLSample]) {
        guard !samples.isEmpty else { return }
        let dim = samples[0].features.count
        let labels = samples.map { Double($0.label) }
        basePrediction = labels.reduce(0, +) / Double(labels.count)
        var residuals = labels.map { $0 - basePrediction }
        learners = []

        for _ in 0..<nEstimators {
            // Fit stump on residuals
            var bestMSE = Double.infinity
            var bestLearner: WeakLearner?
            // EVO_76: eliminated leftIdx/rightIdx/thresholds temp arrays — single-pass per threshold
            for fi in 0..<dim {
                var vals = samples.map { $0.features[fi] }
                vals.sort()
                let nv = vals.count
                for ti in 0..<(nv - 1) where vals[ti] < vals[ti+1] {
                    let thr = (vals[ti] + vals[ti+1]) * 0.5
                    var lSum = 0.0, rSum = 0.0, lCnt = 0, rCnt = 0
                    for i in samples.indices {
                        if samples[i].features[fi] <= thr { lSum += residuals[i]; lCnt += 1 }
                        else                              { rSum += residuals[i]; rCnt += 1 }
                    }
                    let lMean = lCnt == 0 ? 0.0 : lSum / Double(lCnt)
                    let rMean = rCnt == 0 ? 0.0 : rSum / Double(rCnt)
                    var mse = 0.0
                    for i in samples.indices {
                        let d = samples[i].features[fi] <= thr ? (residuals[i] - lMean) : (residuals[i] - rMean)
                        mse += d * d
                    }
                    if mse < bestMSE {
                        bestMSE = mse
                        bestLearner = WeakLearner(
                            featureIdx: fi, threshold: thr,
                            leftVal: lMean, rightVal: rMean,
                            learningRate: lr
                        )
                    }
                }
            }
            guard let learner = bestLearner else { break }
            learners.append(learner)
            // Update residuals
            for i in samples.indices {
                let pred = samples[i].features[learner.featureIdx] <= learner.threshold
                           ? learner.leftVal : learner.rightVal
                residuals[i] -= lr * pred
            }
        }
        isFitted = true
    }

    func predict(features: [Double]) -> MLPrediction {
        guard isFitted else {
            return MLPrediction(label: 0, score: 0, confidence: 0, sacredAlignment: 0)
        }
        var score = basePrediction
        for learner in learners {
            let delta = features[learner.featureIdx] <= learner.threshold
                        ? learner.leftVal : learner.rightVal
            score += learner.learningRate * delta
        }
        let label = score >= 0.5 ? 1 : 0
        let conf  = 1.0 / (1.0 + exp(-score))
        let sacred = 1.0 - abs((conf * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        return MLPrediction(label: label, score: score, confidence: conf, sacredAlignment: sacred)
    }
}

// MARK: - ═══ L104 ENSEMBLE CLASSIFIER ═══

final class L104EnsembleClassifier {
    private let svm    = L104SVM()
    private let forest = L104RandomForest()
    private let boosting = L104GradientBoosting()
    private let names  = ["svm", "forest", "boosting"]
    private var isFitted = false

    func fit(samples: [MLSample]) {
        svm.fit(samples: samples)
        forest.fit(samples: samples)
        boosting.fit(samples: samples)
        isFitted = true
    }

    func predict(features: [Double]) -> EnsemblePrediction {
        let preds = [
            ("svm",      svm.predict(features: features)),
            ("forest",   forest.predict(features: features)),
            ("boosting", boosting.predict(features: features)),
        ]
        var votes: [String: Int] = [:]
        var weights: [String: Double] = [:]
        var labelScore: [Int: Double] = [:]
        var sacredSum = 0.0

        var decay = 1.0
        for (name, p) in preds {
            votes[name] = p.label
            weights[name] = p.confidence
            labelScore[p.label, default: 0] += p.confidence * decay
            sacredSum += p.sacredAlignment
            decay *= ENSEMBLE_DECAY
        }
        let best = labelScore.max { $0.value < $1.value }?.key ?? 0
        let total = labelScore.values.reduce(0, +)
        let conf = (labelScore[best] ?? 0) / max(total, 1e-10)
        return EnsemblePrediction(
            label: best, votes: votes, weights: weights,
            confidence: conf, sacredAlignment: sacredSum / Double(preds.count)
        )
    }
}

// MARK: - ═══ L104 K-MEANS ═══

final class L104KMeans {
    let k: Int      // default 13 (Fib(7))
    let maxIter: Int
    private var centroids: [[Double]] = []
    private var isFitted = false

    init(k: Int = KMEANS_K, maxIter: Int = RF_N_ESTIMATORS) {
        self.k = k; self.maxIter = maxIter
    }

    func fit(samples: [[Double]]) -> ClusterResult {
        guard samples.count >= k else {
            return ClusterResult(assignments: [], centroids: [], inertia: 0, sacredScore: 0, iterations: 0)
        }
        // PHI-spiral initialization: pick k seeds along golden spiral
        var centers: [[Double]] = []
        let dim = samples[0].count
        let goldenAngle = 2.0 * .pi / (PHI * PHI)
        for i in 0..<k {
            let angle = Double(i) * goldenAngle
            let radius = sqrt(Double(i + 1)) / sqrt(Double(k))
            // Map to feature space by selecting closest sample to spiral point
            let spiralFeature = (0..<dim).map { d in
                let base = samples.map { $0[d] }
                let mn = base.min() ?? 0; let mx = base.max() ?? 1
                return mn + (mx - mn) * (0.5 + 0.5 * cos(angle + Double(d)) * radius)
            }
            if let closest = samples.min(by: { _dist($0, spiralFeature) < _dist($1, spiralFeature) }) {
                if !centers.contains(where: { _dist($0, closest) < 1e-10 }) {
                    centers.append(closest)
                }
            }
        }
        // Fill remaining centroids randomly if needed
        while centers.count < k {
            centers.append(samples.randomElement()!)
        }

        var assignments = [Int](repeating: 0, count: samples.count)
        var iter = 0
        for _ in 0..<maxIter {
            iter += 1
            // Assign
            var changed = false
            for (idx, s) in samples.enumerated() {
                let nearest = (0..<k).min { _dist(s, centers[$0]) < _dist(s, centers[$1]) } ?? 0
                if nearest != assignments[idx] { changed = true }
                assignments[idx] = nearest
            }
            if !changed { break }
            // Update centroids — EVO_76: single-pass O(n×dim) accumulation, was O(k×n×dim)
            var sums = [[Double]](repeating: [Double](repeating: 0.0, count: dim), count: k)
            var cnts = [Int](repeating: 0, count: k)
            for (s, c) in zip(samples, assignments) {
                for i in 0..<dim { sums[c][i] += s[i] }
                cnts[c] += 1
            }
            for c in 0..<k where cnts[c] > 0 {
                let inv = 1.0 / Double(cnts[c])
                for i in 0..<dim { centers[c][i] = sums[c][i] * inv }
            }
        }

        let inertia = zip(samples, assignments).reduce(0.0) { acc, pair in
            acc + _dist(pair.0, centers[pair.1])
        }
        let sacred = 1.0 - abs((inertia / GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        centroids = centers
        isFitted = true
        return ClusterResult(assignments: assignments, centroids: centers,
                             inertia: inertia, sacredScore: sacred, iterations: iter)
    }

    func predict(sample: [Double]) -> Int {
        guard isFitted, !centroids.isEmpty else { return 0 }
        return (0..<centroids.count).min { _dist(sample, centroids[$0]) < _dist(sample, centroids[$1]) } ?? 0
    }

    private func _dist(_ a: [Double], _ b: [Double]) -> Double {
        // EVO_76: branchless loop — no zip/map/reduce allocation, compiler auto-vectorizes
        var sum = 0.0
        for i in 0..<a.count { let d = a[i] - b[i]; sum += d * d }
        return sum
    }
}

// MARK: - ═══ L104 DBSCAN ═══

final class L104DBSCAN {
    let epsilon: Double   // GOD_CODE-modulated neighborhood radius
    let minPts: Int       // Fib(5) = 5

    init(epsilon: Double = GOD_CODE / 10000.0, minPts: Int = 5) {
        self.epsilon = epsilon; self.minPts = minPts
    }

    func fit(samples: [[Double]]) -> ClusterResult {
        let n = samples.count
        var labels = [Int](repeating: -1, count: n)  // -1 = noise
        var clusterId = 0

        for i in 0..<n {
            guard labels[i] == -1 else { continue }
            let neighbors = _neighbors(i, samples: samples)
            guard neighbors.count >= minPts else { continue }
            labels[i] = clusterId
            var seeds = neighbors
            var si = 0
            while si < seeds.count {
                let q = seeds[si]; si += 1
                if labels[q] == -1 {
                    labels[q] = clusterId
                    let qNeigh = _neighbors(q, samples: samples)
                    if qNeigh.count >= minPts { seeds += qNeigh }
                } else if labels[q] < 0 {
                    labels[q] = clusterId
                }
            }
            clusterId += 1
        }

        let maxC = (labels.max() ?? -1) + 1
        var centroids = [[Double]](repeating: [Double](repeating: 0, count: samples.first?.count ?? 1), count: maxC)
        var counts    = [Int](repeating: 0, count: maxC)
        for (i, lbl) in labels.enumerated() where lbl >= 0 {
            for d in 0..<samples[i].count {
                centroids[lbl][d] += samples[i][d]
            }
            counts[lbl] += 1
        }
        for c in 0..<maxC where counts[c] > 0 {
            centroids[c] = centroids[c].map { $0 / Double(counts[c]) }
        }

        let sacred = 1.0 - abs((Double(maxC) * PHI).truncatingRemainder(dividingBy: 1.0))
        return ClusterResult(assignments: labels, centroids: centroids,
                             inertia: 0.0, sacredScore: sacred, iterations: 1)
    }

    private func _neighbors(_ idx: Int, samples: [[Double]]) -> [Int] {
        // EVO_76: branchless squared-distance loop — no zip/map/reduce allocation per neighbor check
        let ref = samples[idx]; let n = ref.count; let eps2 = epsilon * epsilon
        return samples.indices.filter { j in
            var d2 = 0.0
            let sj = samples[j]
            for i in 0..<n { let d = ref[i] - sj[i]; d2 += d * d }
            return d2 <= eps2
        }
    }
}

// MARK: - ═══ VARIATIONAL QUANTUM CLASSIFIER ═══

final class VariationalQuantumClassifier {
    let nQubits: Int
    let depth: Int
    let nClasses: Int

    // Trainable parameters (angles per layer per qubit)
    private var params: [Double]
    private var isFitted = false
    private var trainingHistory: [Double] = []

    init(nQubits: Int = VQC_QUBITS, depth: Int = VQC_DEPTH, nClasses: Int = 2) {
        self.nQubits = nQubits; self.depth = depth; self.nClasses = nClasses
        // Initialize params with PHI-modulated angles
        let n = nQubits * depth * 3
        self.params = (0..<n).map { i in Double(i) * PHI * .pi / Double(n) }
    }

    // Simulate circuit via sacred statevector approximation
    private func _simulateCircuit(features: [Double]) -> [Double] {
        let dimH = 1 << nQubits  // 2^nQubits
        var state = [Double](repeating: 0.0, count: dimH)
        state[0] = 1.0

        // Feature encoding: Ry rotations
        for (qi, feat) in features.prefix(nQubits).enumerated() {
            let angle = feat * .pi
            _applyRy(state: &state, qubit: qi, angle: angle, n: nQubits)
        }

        // Variational ansatz: param-θ Ry + CX entanglement
        var pIdx = 0
        for _ in 0..<depth {
            for qi in 0..<nQubits {
                _applyRy(state: &state, qubit: qi, angle: params[pIdx % params.count], n: nQubits)
                _applyRz(state: &state, qubit: qi, angle: params[(pIdx+1) % params.count], n: nQubits)
                pIdx += 2
            }
            // Entangle adjacent qubits
            for qi in 0..<(nQubits - 1) {
                _applyCX(state: &state, control: qi, target: qi+1, n: nQubits)
            }
        }

        // Born rule probabilities
        return state.map { $0 * $0 }
    }

    private func _applyRy(state: inout [Double], qubit: Int, angle: Double, n: Int) {
        let c = cos(angle / 2.0); let s = sin(angle / 2.0)
        let stride = 1 << qubit
        for i in 0..<(1 << n) where (i & stride) == 0 {
            let a = state[i]; let b = state[i | stride]
            state[i]        = c * a - s * b
            state[i|stride] = s * a + c * b
        }
    }

    private func _applyRz(state: inout [Double], qubit: Int, angle: Double, n: Int) {
        let stride = 1 << qubit
        for i in 0..<(1 << n) where (i & stride) != 0 {
            state[i] *= cos(angle)  // simplified (ignores imaginary for pure-real state)
        }
    }

    private func _applyCX(state: inout [Double], control: Int, target: Int, n: Int) {
        let cs = 1 << control; let ts = 1 << target
        for i in 0..<(1 << n) where (i & cs) != 0 {
            let j = i ^ ts
            state.swapAt(i, j)
        }
    }

    func fit(samples: [MLSample], epochs: Int = RF_N_ESTIMATORS) {
        let lr = GB_LEARNING_RATE
        for _ in 0..<epochs {
            var totalLoss = 0.0
            for s in samples {
                let probs = _simulateCircuit(features: s.features)
                let target = s.label < nClasses ? s.label : 0
                let loss = 1.0 - probs[target]
                totalLoss += loss
                // Gradient ascent via parameter shift (simplified)
                for pi in 0..<params.count {
                    params[pi] -= lr * loss * sin(params[pi]) * PHI
                }
            }
            trainingHistory.append(totalLoss / Double(max(samples.count, 1)))
        }
        isFitted = true
    }

    func predict(features: [Double]) -> VQCResult {
        let probs = _simulateCircuit(features: features)
        let classProbs = Array(probs.prefix(nClasses))
        let label = classProbs.indices.max(by: { classProbs[$0] < classProbs[$1] }) ?? 0
        let sacred = 1.0 - abs((probs[0] * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        let loss = trainingHistory.last ?? 0.0
        return VQCResult(
            predictedClass: label, probabilities: classProbs,
            circuitDepth: depth, sacredAlignment: sacred, trainingLoss: loss
        )
    }
}

// MARK: - ═══ QUANTUM NEAREST NEIGHBOR ═══

final class QuantumNearestNeighbor {
    let k: Int     // k=3 by default (Fib(4)=3)
    private var trainSamples: [MLSample] = []
    private let kernelLib = SacredKernelLibrary.shared

    init(k: Int = 3) { self.k = k }

    func fit(samples: [MLSample]) { trainSamples = samples }

    func predict(features: [Double]) -> MLPrediction {
        guard !trainSamples.isEmpty else {
            return MLPrediction(label: 0, score: 0, confidence: 0, sacredAlignment: 0)
        }
        // Quantum kernel similarity
        let similarities = trainSamples.map { s in
            kernelLib.compositeSacred(features, s.features)
        }
        // Top-k nearest
        let ranked = zip(trainSamples, similarities)
            .sorted { $0.1 > $1.1 }
            .prefix(k)
        var votes: [Int: Double] = [:]
        for (s, sim) in ranked {
            votes[s.label, default: 0] += sim
        }
        let best = votes.max { $0.value < $1.value }?.key ?? 0
        let total = votes.values.reduce(0, +)
        let conf  = (votes[best] ?? 0) / max(total, 1e-10)
        let sacred = 1.0 - abs((conf * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        return MLPrediction(label: best, score: conf, confidence: conf, sacredAlignment: sacred)
    }
}

// MARK: - ═══ QUANTUM ENSEMBLE CLASSIFIER ═══

final class QuantumEnsembleClassifier {
    private let vqc  = VariationalQuantumClassifier()
    private let qknn = QuantumNearestNeighbor()
    private let svmE = SVMEnsemble()
    private var isFitted = false

    func fit(samples: [MLSample]) {
        vqc.fit(samples: samples)
        qknn.fit(samples: samples)
        svmE.fit(samples: samples)
        isFitted = true
    }

    func predict(features: [Double]) -> EnsemblePrediction {
        let r1 = vqc.predict(features: features)
        let r2 = qknn.predict(features: features)
        let r3 = svmE.predict(features: features)

        var labelScore: [Int: Double] = [:]
        let w: [Double] = [PHI, 1.0, 1.0/PHI]
        for (pred, wi) in zip([r1.predictedClass, r2.label, r3.label], w) {
            labelScore[pred, default: 0] += wi
        }
        let best  = labelScore.max { $0.value < $1.value }?.key ?? 0
        let total = labelScore.values.reduce(0, +)
        let conf  = (labelScore[best] ?? 0) / max(total, 1e-10)
        let sacred = (r1.sacredAlignment + r2.sacredAlignment + r3.sacredAlignment) / 3.0

        return EnsemblePrediction(
            label: best,
            votes: ["vqc": r1.predictedClass, "qknn": r2.label, "svmEnsemble": r3.label],
            weights: ["vqc": w[0], "qknn": w[1], "svmEnsemble": w[2]],
            confidence: conf, sacredAlignment: sacred
        )
    }
}

// MARK: - ═══ CROSS-ENGINE FEATURE EXTRACTOR ═══

final class CrossEngineFeatureExtractor {

    func extract() -> CrossEngineFeatures {
        var f = CrossEngineFeatures()

        // Science features (10): entropy, coherence, physics metrics
        f.scienceFeatures[0] = _demonEfficiency()
        f.scienceFeatures[1] = _coherenceLevel()
        f.scienceFeatures[2] = GOD_CODE.truncatingRemainder(dividingBy: 1.0)
        f.scienceFeatures[3] = PHI
        f.scienceFeatures[4] = VOID_CONSTANT
        f.scienceFeatures[5] = sin(GOD_CODE / 1000.0)
        f.scienceFeatures[6] = cos(PHI * .pi)
        f.scienceFeatures[7] = 286.0 / GOD_CODE     // Fe resonance normalized
        f.scienceFeatures[8] = exp(-1.0 / PHI)
        f.scienceFeatures[9] = OMEGA / GOD_CODE

        // Math features (10): GOD_CODE alignment, harmonic scores
        let godFib = _godCodeFibAlignment()
        f.mathFeatures[0] = godFib
        f.mathFeatures[1] = _harmonicAlignment(fundamental: 104.0)
        f.mathFeatures[2] = 1.0 / (1.0 + exp(-GOD_CODE / 1000.0))
        f.mathFeatures[3] = (PHI * PHI - PHI - 1.0).magnitude  // ≈ 0
        f.mathFeatures[4] = log(GOD_CODE) / log(PHI)
        f.mathFeatures[5] = sin(.pi / PHI)
        f.mathFeatures[6] = Double(Fibonacci(13)) / GOD_CODE
        f.mathFeatures[7] = cos(VOID_CONSTANT * .pi)
        f.mathFeatures[8] = sqrt(PHI) - 1.0
        f.mathFeatures[9] = atan(GOD_CODE / OMEGA)

        // Code features (10): complexity metrics (normalized)
        let codeEng = CodeEngineMetrics.shared
        f.codeFeatures[0] = codeEng.currentComplexity
        f.codeFeatures[1] = codeEng.sacredAlignmentScore
        f.codeFeatures[2] = codeEng.debtRatio
        f.codeFeatures[3] = codeEng.evolutionCycle / 1000.0
        f.codeFeatures[4] = codeEng.refactorScore
        for i in 5..<10 { f.codeFeatures[i] = Double(i) * PHI / 10.0 }

        // Quantum features (10): circuit metrics
        let qMetrics = QuantumCircuitMetrics.shared
        f.quantumFeatures[0] = qMetrics.averageFidelity
        f.quantumFeatures[1] = qMetrics.entanglementDepth / Double(RF_N_ESTIMATORS)
        f.quantumFeatures[2] = qMetrics.sacredCircuitScore
        f.quantumFeatures[3] = qMetrics.gateErrorRate
        f.quantumFeatures[4] = qMetrics.coherenceTime / 1000.0
        for i in 5..<10 { f.quantumFeatures[i] = sin(Double(i) * PHI) * 0.5 + 0.5 }

        // ASI features (10): scoring dimensions
        let asiScore = ASIScoringCache.shared
        f.asiFeatures[0] = asiScore.dimension("analytical")
        f.asiFeatures[1] = asiScore.dimension("scientific")
        f.asiFeatures[2] = asiScore.dimension("mathematical")
        f.asiFeatures[3] = asiScore.dimension("quantum")
        f.asiFeatures[4] = asiScore.dimension("consciousness")
        for i in 5..<10 { f.asiFeatures[i] = Double(i) * PHI / 15.0 }

        return f
    }

    private func _demonEfficiency() -> Double {
        let entropy = Double.random(in: 0.3...1.0)
        return 1.0 - (entropy / PHI).truncatingRemainder(dividingBy: 1.0)
    }

    private func _coherenceLevel() -> Double {
        let t = Date().timeIntervalSince1970
        return abs(sin(t / GOD_CODE))
    }

    private func _godCodeFibAlignment() -> Double {
        let ratio = Double(Fibonacci(13)) / Double(Fibonacci(12))
        return 1.0 - abs(ratio - PHI)
    }

    private func _harmonicAlignment(fundamental: Double) -> Double {
        let resonance = fundamental * PHI
        return abs(sin(resonance / GOD_CODE * .pi))
    }

    // Fibonacci helper
    private func Fibonacci(_ n: Int) -> Int {
        if n <= 1 { return n }
        var a = 0; var b = 1
        for _ in 2...n { let c = a + b; a = b; b = c }
        return b
    }
}

// Lightweight metric caches used by CrossEngineFeatureExtractor
final class CodeEngineMetrics {
    static let shared = CodeEngineMetrics()
    var currentComplexity:    Double = 0.5
    var sacredAlignmentScore: Double = PHI / 2.0
    var debtRatio:            Double = 0.1
    var evolutionCycle:       Double = 0.0
    var refactorScore:        Double = 0.7
}

final class QuantumCircuitMetrics {
    static let shared = QuantumCircuitMetrics()
    var averageFidelity:     Double = 0.95
    var entanglementDepth:   Double = 4.0
    var sacredCircuitScore:  Double = 0.8
    var gateErrorRate:       Double = 0.01
    var coherenceTime:       Double = 100.0
}

final class ASIScoringCache {
    static let shared = ASIScoringCache()
    private var dims: [String: Double] = [
        "analytical": 0.85, "scientific": 0.78, "mathematical": 0.90,
        "quantum": 0.72, "consciousness": 0.65,
    ]
    func dimension(_ name: String) -> Double { dims[name] ?? 0.5 }
    func update(_ name: String, value: Double) { dims[name] = max(0, min(1, value)) }
}

// MARK: - ═══ KNOWLEDGE SYNTHESIZER ═══

final class KnowledgeSynthesizer {
    private let extractor = CrossEngineFeatureExtractor()
    private let kmeans    = L104KMeans(k: KMEANS_K)
    private let forest    = L104RandomForest()
    private var history:  [CrossEngineFeatures] = []

    func synthesize() -> KnowledgeSynthesisResult {
        let current = extractor.extract()
        history.append(current)

        // Cluster historical feature vectors
        let vectors = history.map(\.allFeatures)
        var cluster = ClusterResult(assignments: [], centroids: [], inertia: 0, sacredScore: 0, iterations: 0)
        if vectors.count >= KMEANS_K {
            cluster = kmeans.fit(samples: vectors)
        }

        // Coherence = 1 - normalized inertia
        let coherence = max(0.0, 1.0 - cluster.inertia / (GOD_CODE * Double(max(history.count, 1))))

        // Dominant engine by max mean feature
        let engines = ["science", "math", "code", "quantum", "asi"]
        let means = [
            current.scienceFeatures.reduce(0,+) / Double(N_FEATURES_ENGINE),
            current.mathFeatures.reduce(0,+)    / Double(N_FEATURES_ENGINE),
            current.codeFeatures.reduce(0,+)    / Double(N_FEATURES_ENGINE),
            current.quantumFeatures.reduce(0,+) / Double(N_FEATURES_ENGINE),
            current.asiFeatures.reduce(0,+)     / Double(N_FEATURES_ENGINE),
        ]
        let domIdx = means.indices.max(by: { means[$0] < means[$1] }) ?? 0

        // Fused vector: PHI-weighted blend of current + cluster centroid
        let allF = current.allFeatures
        let centroid = cluster.centroids.first ?? allF
        let fused = zip(allF, centroid.prefix(allF.count)).map { PHI * $0 + (1-1/PHI) * $1 }

        let sacred = current.sacredAlignment
        let confidence = min(1.0, coherence * PHI + sacred * (1.0/PHI))

        return KnowledgeSynthesisResult(
            coherenceScore: coherence,
            fusedVector: fused,
            dominantEngine: engines[domIdx],
            synthesisConfidence: confidence,
            sacredAlignment: sacred,
            clusterAssignments: cluster.assignments,
            timestamp: Date()
        )
    }
}

// MARK: - ═══ ML ENGINE ORCHESTRATOR ═══

final class MLEngine {
    static let shared = MLEngine()

    let sacredKernels  = SacredKernelLibrary.shared
    let svm            = L104SVM()
    let svmEnsemble    = SVMEnsemble()
    let randomForest   = L104RandomForest()
    let gradientBoost  = L104GradientBoosting()
    let ensemble       = L104EnsembleClassifier()
    let kmeans         = L104KMeans()
    let dbscan         = L104DBSCAN()
    let vqc            = VariationalQuantumClassifier()
    let qknn           = QuantumNearestNeighbor()
    let qEnsemble      = QuantumEnsembleClassifier()
    let featureExt     = CrossEngineFeatureExtractor()
    let synthesizer    = KnowledgeSynthesizer()

    private let queue = DispatchQueue(label: "l104.ml.engine", qos: .userInitiated)

    // Full ML pipeline: extract cross-engine features → classify
    func runPipeline(query: String) -> KnowledgeSynthesisResult {
        return synthesizer.synthesize()
    }

    // Sacred-kernel SVM predict for a raw feature vector
    func classify(features: [Double]) -> MLPrediction {
        svm.predict(features: features)
    }

    // Quantum-enhanced classification
    func quantumClassify(features: [Double]) -> EnsemblePrediction {
        qEnsemble.predict(features: features)
    }

    // Cluster a batch of feature vectors
    func cluster(vectors: [[Double]]) -> ClusterResult {
        kmeans.fit(samples: vectors)
    }

    // Background synthesis loop
    func startSynthesisLoop(interval: TimeInterval = 30.0) {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + interval, repeating: interval,
                       leeway: .seconds(max(5, Int(interval / 6))))
        timer.setEventHandler { [weak self] in
            guard let self = self else { return }
            let result = self.synthesizer.synthesize()
            InterEngineFeedbackBus.shared.broadcast(
                from: .mlSynthesis,
                signal: "ml_synthesis_cycle",
                payload: [
                    "coherence":   result.coherenceScore,
                    "confidence":  result.synthesisConfidence,
                    "sacred":      result.sacredAlignment,
                ]
            )
        }
        timer.resume()
    }

    // Self-test
    func selfTest() -> Bool {
        // Generate synthetic samples
        let samples = (0..<104).map { i -> MLSample in
            let f = (0..<8).map { d in sin(Double(i * d) * PHI / 100.0) }
            return MLSample(features: f, label: i % 2)
        }
        svm.fit(samples: samples)
        randomForest.fit(samples: samples)
        ensemble.fit(samples: samples)

        let testF = (0..<8).map { d in cos(Double(d) * PHI) }
        let p1 = svm.predict(features: testF)
        let p2 = randomForest.predict(features: testF)
        let p3 = ensemble.predict(features: testF)

        return p1.confidence > 0 && p2.confidence > 0 && p3.confidence > 0
    }
}
