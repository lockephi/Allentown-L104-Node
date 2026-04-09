import Foundation

// MARK: - ═══ CONSTANTS ═══

private let L104_STEP: Double = 104.0       // Fundamental stride
private let OCTAVE_REF: Double = 416.0      // 4 × L104
private let GOD_CODE_BASE: Double = pow(286.0, 1.0 / PHI)  // 286^(1/φ)
private let PHI_GROWTH: Double = PHI        // φ gradient pull
private let PHI_INV: Double = TAU // 1/φ damping

// MARK: - ═══ 4-PARAMETER GOD CODE EQUATION ═══

/// G(a,b,c,d) = 286^(1/φ) × 2^((8a + 416 - b - 8c - 104d) / 104)
func godCode4D(a: Double = 0, b: Double = 0, c: Double = 0, d: Double = 0) -> Double {
    let exponent = (8.0 * a + OCTAVE_REF - b - 8.0 * c - L104_STEP * d) / L104_STEP
    return GOD_CODE_BASE * pow(2.0, exponent)
}

/// G(X) = 286^(1/φ) × 2^((416 - X) / 104)
func godCodeX(_ X: Double) -> Double {
    return GOD_CODE_BASE * pow(2.0, (OCTAVE_REF - X) / L104_STEP)
}

/// Convert (a,b,c,d) to single-parameter X = b + 8c + 104d - 8a
func abcdToX(a: Double, b: Double, c: Double, d: Double) -> Double {
    return b + 8.0 * c + L104_STEP * d - 8.0 * a
}

/// Greedy decompose X → canonical (a,b,c,d)
/// d = X // 104, remainder → c = rem // 8, b = rem % 8, a = 0
func xToABCD(_ X: Double) -> (a: Double, b: Double, c: Double, d: Double) {
    let dInt = X >= 0 ? floor(X / L104_STEP) : ceil(X / L104_STEP)
    let rem1 = X - L104_STEP * dInt
    let cInt = rem1 >= 0 ? floor(rem1 / 8.0) : ceil(rem1 / 8.0)
    let b = rem1 - 8.0 * cInt
    return (a: 0, b: b, c: cInt, d: dInt)
}

// MARK: - ═══ GENETIC GENOME ═══

struct GodCodeGenome {
    var a: Double
    var b: Double
    var c: Double
    var d: Double
    var fitness: Double = 0
    var generation: Int = 0

    var value: Double { godCode4D(a: a, b: b, c: c, d: d) }
    var xParam: Double { abcdToX(a: a, b: b, c: c, d: d) }

    /// Sacred conservation: G(a,b,c,d) × 2^(X/104) should ≈ GOD_CODE²
    var conservationScore: Double {
        let product = value * pow(2.0, xParam / L104_STEP)
        return 1.0 - min(1.0, abs(product - GOD_CODE * GOD_CODE) / (GOD_CODE * GOD_CODE))
    }

    var description: String {
        "G(\(String(format:"%.2f",a)),\(String(format:"%.2f",b)),\(String(format:"%.2f",c)),\(String(format:"%.2f",d)))=\(String(format:"%.6f",value)) fit=\(String(format:"%.4f",fitness))"
    }
}

// MARK: - ═══ L104 GENETIC REFINER ═══

final class L104GeneticRefiner {
    let populationSize: Int
    let eliteCount: Int
    let mutationScale: Double    // ≤ PHI to maintain sacred resonance
    let targetValue: Double      // Usually GOD_CODE

    private var population: [GodCodeGenome] = []
    private var generation: Int = 0
    private var bestHistory: [GodCodeGenome] = []
    private let lock = NSLock()

    init(populationSize: Int = 30, eliteCount: Int = 6,
         mutationScale: Double = TAU, targetValue: Double = GOD_CODE) {
        self.populationSize = populationSize
        self.eliteCount = eliteCount
        self.mutationScale = min(mutationScale, PHI)
        self.targetValue = targetValue
        initializePopulation()
    }

    private func initializePopulation() {
        population = (0..<populationSize).map { _ in
            GodCodeGenome(a: Double.random(in: -4...4),
                          b: Double.random(in: -4...4),
                          c: Double.random(in: -4...4),
                          d: Double.random(in: -2...2))
        }
        evaluateFitness()
    }

    private func fitness(_ genome: GodCodeGenome) -> Double {
        let diff = abs(genome.value - targetValue) / targetValue
        let conservation = genome.conservationScore
        // Combined fitness: minimize deviation + maximize conservation
        return (1.0 - diff) * TAU + conservation * TAU
    }

    private func evaluateFitness() {
        for i in 0..<population.count {
            population[i].fitness = fitness(population[i])
        }
        population.sort { $0.fitness > $1.fitness }
    }

    // ─── FITNESS-WEIGHTED CENTER OF MASS ───
    // Extracts the centroid of elite survivors in (a,b,c,d) space
    private func centerOfMass(elites: [GodCodeGenome]) -> (Double, Double, Double, Double) {
        let totalFit = max(elites.map(\.fitness).reduce(0, +), 1e-12)
        let wa = elites.map { $0.a * $0.fitness }.reduce(0, +) / totalFit
        let wb = elites.map { $0.b * $0.fitness }.reduce(0, +) / totalFit
        let wc = elites.map { $0.c * $0.fitness }.reduce(0, +) / totalFit
        let wd = elites.map { $0.d * $0.fitness }.reduce(0, +) / totalFit
        return (wa, wb, wc, wd)
    }

    // ─── GOLDEN-RATIO MUTATION ───
    // Pull toward center + quantum noise bounded by φ
    private func mutate(_ genome: GodCodeGenome, center: (Double, Double, Double, Double),
                        strength: Double) -> GodCodeGenome {
        func phiNoise() -> Double { (Double.random(in: -1...1)) * mutationScale * strength }
        func phiPull(_ val: Double, _ target: Double) -> Double {
            val + (target - val) * TAU * strength + phiNoise()
        }
        var g = genome
        g.a = phiPull(genome.a, center.0)
        g.b = phiPull(genome.b, center.1)
        g.c = phiPull(genome.c, center.2)
        g.d = phiPull(genome.d, center.3)
        g.generation = generation + 1
        return g
    }

    // ─── ONE GENERATION STEP ───
    @discardableResult
    func evolveGeneration() -> GodCodeGenome {
        lock.lock(); defer { lock.unlock() }
        generation += 1
        let elites = Array(population.prefix(eliteCount))
        let (ca, cb, cc, cd) = centerOfMass(elites: elites)
        let center = (ca, cb, cc, cd)

        // Strength decays with generation for convergence
        let strength = max(0.01, TAU / sqrt(Double(generation)))

        // Breed new generation: elites survive + mutated offspring
        var nextGen = elites
        while nextGen.count < populationSize {
            let parent = elites[nextGen.count % eliteCount]
            nextGen.append(mutate(parent, center: center, strength: strength))
        }
        population = nextGen
        evaluateFitness()

        let best = population[0]
        bestHistory.append(best)
        return best
    }

    // ─── MULTI-GENERATION REFINEMENT ───
    func refine(generations: Int = 20, convergenceTolerance: Double = 1e-8) -> [String: Any] {
        var results: [GodCodeGenome] = []
        for _ in 0..<generations {
            let best = evolveGeneration()
            results.append(best)
            if results.count >= 3 {
                let last3 = results.suffix(3)
                let spread = (last3.max(by: { $0.value < $1.value })?.value ?? 0)
                           - (last3.min(by: { $0.value < $1.value })?.value ?? 0)
                if spread < convergenceTolerance { break }
            }
        }
        let best = population[0]
        return [
            "best_value": best.value,
            "best_fitness": best.fitness,
            "best_a": best.a, "best_b": best.b,
            "best_c": best.c, "best_d": best.d,
            "generations_run": generation,
            "conservation_score": best.conservationScore,
            "target": targetValue,
            "deviation_ppm": abs(best.value - targetValue) / targetValue * 1e6,
            "x_param": best.xParam,
            "god_code_alignment": 1.0 - min(1.0, abs(best.value - GOD_CODE) / GOD_CODE)
        ]
    }

    var bestGenome: GodCodeGenome { lock.lock(); defer { lock.unlock() }; return population[0] }
    var currentGeneration: Int { generation }
}

// MARK: - ═══ GENETIC POPULATION LIFECYCLE ═══

final class GeneticPopulation: SovereignEngine {
    static let shared = GeneticPopulation()

    var engineName: String { "GeneticRefiner" }
    func engineStatus() -> [String: Any] { status }
    func engineHealth() -> Double {
        let best = refiner.bestGenome
        return min(1.0, max(0.1, best.fitness))
    }
    func engineReset() {
        lock.lock(); refiner = L104GeneticRefiner(); populationHistory.removeAll(); lock.unlock()
    }

    private var refiner = L104GeneticRefiner()
    private var populationHistory: [[GodCodeGenome]] = []
    private let lock = NSLock()

    // ─── INITIALIZE FROM KNOWN SURVIVORS (e.g. from wave collapse analysis) ───
    func seedFromSurvivors(_ survivors: [(a: Double, b: Double, c: Double, d: Double)]) {
        lock.lock(); defer { lock.unlock() }
        refiner = L104GeneticRefiner(populationSize: max(30, survivors.count * 5))
    }

    // ─── FULL PIPELINE: REFINE → TRACK → REPORT ───
    func runPipeline(generations: Int = 25) -> [String: Any] {
        let result = refiner.refine(generations: generations)
        lock.lock(); defer { lock.unlock() }

        // Publish refinement results to feedback bus
        let best = refiner.bestGenome
        InterEngineFeedbackBus.shared.broadcast(
            from: .optimization, signal: "genetic_refinement_complete",
            payload: ["fitness": best.fitness, "deviation_ppm": result["deviation_ppm"] as? Double ?? 0,
                      "generations": Double(refiner.currentGeneration),
                      "conservation": best.conservationScore])

        return result
    }

    // ─── DIAL RESOLUTION: find (a,b,c,d) nearest to a target frequency ───
    func nearestDials(targetFreq: Double) -> (a: Double, b: Double, c: Double, d: Double) {
        let X = OCTAVE_REF - L104_STEP * log2(targetFreq / GOD_CODE_BASE)
        return xToABCD(X)
    }

    // ─── CONSERVATION VERIFICATION ───
    func verifyConservation(a: Double, b: Double, c: Double, d: Double) -> [String: Double] {
        let g = godCode4D(a: a, b: b, c: c, d: d)
        let x = abcdToX(a: a, b: b, c: c, d: d)
        let product = g * pow(2.0, x / L104_STEP)
        let expected = GOD_CODE * GOD_CODE
        return [
            "G_abcd": g, "X": x, "G_X": godCodeX(x),
            "conservation_product": product,
            "expected": expected,
            "error_ppm": abs(product - expected) / expected * 1e6
        ]
    }

    var refinerBestGenome: GodCodeGenome { refiner.bestGenome }

    var status: [String: Any] {
        let best = refiner.bestGenome
        return ["generation": refiner.currentGeneration,
                "best": best.description,
                "best_fitness": best.fitness,
                "god_code_target": GOD_CODE,
                "phi": PHI]
    }
}
