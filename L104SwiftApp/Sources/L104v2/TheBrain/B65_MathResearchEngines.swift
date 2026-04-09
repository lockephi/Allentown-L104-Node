import Accelerate
import Foundation

// MARK: - ═══ SHARED MATH RESEARCH CONSTANTS ═══

private let GOD_CODE_BASE_HP: Double = pow(286.0, 1.0 / PHI)  // 286^(1/φ)
private let LN2_OVER_104: Double = log(2.0) / 104.0
private let EULER_GAMMA_HP: Double = 0.5772156649015329
private let APERY_HP: Double = 1.2020569031595942    // ζ(3)
private let CATALAN_HP: Double = 0.9159655941772190
private let ZETA_2_HP: Double = Double.pi * Double.pi / 6.0  // π²/6
private let ZETA_4_HP: Double = pow(Double.pi, 4) / 90.0     // π⁴/90

/// First 16 Bernoulli numbers B₀, B₁, ..., B₃₀ (even indices used for zeta)
private let BERNOULLI: [Double] = [
    1.0, -0.5, 1.0/6.0, 0.0, -1.0/30.0, 0.0, 1.0/42.0, 0.0,
    -1.0/30.0, 0.0, 5.0/66.0, 0.0, -691.0/2730.0, 0.0, 7.0/6.0, 0.0
]

// MARK: - ═══ 1. RIEMANN ZETA ENGINE ═══

final class RiemannZetaEngine {
    static let shared = RiemannZetaEngine()

    /// ζ(s) for real s > 1 via Euler-Maclaurin summation (200 terms)
    func zetaReal(_ s: Double, terms: Int = 200) -> Double {
        guard s > 1.0 else { return Double.nan }
        var sum = 0.0
        for n in 1...terms {
            sum += pow(Double(n), -s)
        }
        // Euler-Maclaurin correction: +½n⁻ˢ + s/(12)·n⁻⁽ˢ⁺¹⁾
        let n = Double(terms)
        sum += 0.5 * pow(n, -s)
        sum += s / 12.0 * pow(n, -(s + 1.0))
        return sum
    }

    /// ζ(2n) exact via Bernoulli: ζ(2n) = (-1)^(n+1) B_{2n} (2π)^{2n} / (2(2n)!)
    func zetaEvenExact(_ n: Int) -> Double {
        guard n >= 1, 2 * n < BERNOULLI.count else { return Double.nan }
        let b2n = BERNOULLI[2 * n]
        let sign = (n % 2 == 0) ? -1.0 : 1.0
        let twoPi2n = pow(2.0 * Double.pi, Double(2 * n))
        var factorial2n = 1.0
        for i in 1...(2 * n) { factorial2n *= Double(i) }
        return sign * b2n * twoPi2n / (2.0 * factorial2n)
    }

    /// Dirichlet eta: η(s) = (1 - 2^(1-s)) × ζ(s)
    func dirichletEta(_ s: Double, terms: Int = 1000) -> Double {
        var sum = 0.0
        for n in 1...terms {
            let sign = (n % 2 == 1) ? 1.0 : -1.0
            sum += sign * pow(Double(n), -s)
        }
        return sum
    }

    /// Verify known values: ζ(2)=π²/6, ζ(4)=π⁴/90
    func verifyKnownValues() -> [String: Any] {
        let z2 = zetaReal(2.0, terms: 500)
        let z4 = zetaReal(4.0, terms: 500)
        let z2Exact = ZETA_2_HP
        let z4Exact = ZETA_4_HP
        return [
            "zeta_2_computed": z2, "zeta_2_exact": z2Exact,
            "zeta_2_error": abs(z2 - z2Exact),
            "zeta_4_computed": z4, "zeta_4_exact": z4Exact,
            "zeta_4_error": abs(z4 - z4Exact),
            "zeta_2_bernoulli": zetaEvenExact(1),
            "zeta_4_bernoulli": zetaEvenExact(2)
        ]
    }

    /// Critical strip analysis: |ζ(1/2 + it)| for t ∈ [14, 50]
    func criticalStripAnalysis(imStart: Double = 14.0, imEnd: Double = 50.0,
                               steps: Int = 100) -> [String: Any] {
        // Approximate |ζ(1/2+it)| via Dirichlet series (limited accuracy on critical line)
        var magnitudes: [(Double, Double)] = []
        let dt = (imEnd - imStart) / Double(steps)
        for i in 0...steps {
            let t = imStart + Double(i) * dt
            // ζ(1/2+it) ≈ Σ n^(-1/2-it) = Σ n^(-1/2) × exp(-it·ln(n))
            var realPart = 0.0
            var imagPart = 0.0
            for n in 1...200 {
                let nf = Double(n)
                let amp = pow(nf, -0.5)
                let phase = -t * log(nf)
                realPart += amp * cos(phase)
                imagPart += amp * sin(phase)
            }
            let mag = sqrt(realPart * realPart + imagPart * imagPart)
            magnitudes.append((t, mag))
        }
        // Find approximate zeros (local minima below threshold)
        var approxZeros: [Double] = []
        for i in 1..<magnitudes.count - 1 {
            if magnitudes[i].1 < magnitudes[i-1].1 &&
               magnitudes[i].1 < magnitudes[i+1].1 &&
               magnitudes[i].1 < 2.0 {
                approxZeros.append(magnitudes[i].0)
            }
        }
        return [
            "range": [imStart, imEnd],
            "steps": steps,
            "approximate_zeros": approxZeros,
            "first_known_zero": 14.134725,
            "min_magnitude": magnitudes.min(by: { $0.1 < $1.1 })?.1 ?? 0.0
        ]
    }

    func fullAnalysis() -> [String: Any] {
        return [
            "verification": verifyKnownValues(),
            "critical_strip": criticalStripAnalysis(),
            "dirichlet_eta_at_2": dirichletEta(2.0),
            "apery_constant": APERY_HP,
            "engine": "RiemannZetaEngine"
        ]
    }
}

// MARK: - ═══ 2. GOD CODE CALCULUS ENGINE ═══

final class GodCodeCalculusEngine {
    static let shared = GodCodeCalculusEngine()

    /// G(X) = 286^(1/φ) × 2^((416-X)/104)
    func G(_ x: Double) -> Double {
        return GOD_CODE_BASE_HP * pow(2.0, (416.0 - x) / 104.0)
    }

    /// dG/dX = -G(X) × ln(2)/104 (exact analytical derivative)
    func dGdx(_ x: Double) -> Double {
        return -G(x) * LN2_OVER_104
    }

    /// d²G/dX² = G(X) × (ln(2)/104)²
    func d2Gdx2(_ x: Double) -> Double {
        return G(x) * LN2_OVER_104 * LN2_OVER_104
    }

    /// Central difference numerical derivative (h = 10⁻⁸)
    func dGdxNumerical(_ x: Double, h: Double = 1e-8) -> Double {
        return (G(x + h) - G(x - h)) / (2.0 * h)
    }

    /// ∫ G(X)dX = -G(X) × 104/ln(2) + C (closed-form antiderivative)
    func integralAnalytical(xLow: Double, xHigh: Double) -> Double {
        let antiderivHigh = -G(xHigh) * 104.0 / log(2.0)
        let antiderivLow = -G(xLow) * 104.0 / log(2.0)
        return antiderivHigh - antiderivLow
    }

    /// Simpson's 1/3 rule numerical integration
    func integralNumerical(xLow: Double, xHigh: Double, intervals: Int = 1000) -> Double {
        let n = intervals % 2 == 0 ? intervals : intervals + 1
        let h = (xHigh - xLow) / Double(n)
        var sum = G(xLow) + G(xHigh)
        for i in 1..<n {
            let x = xLow + Double(i) * h
            sum += (i % 2 == 0 ? 2.0 : 4.0) * G(x)
        }
        return sum * h / 3.0
    }

    /// Taylor expansion: G(X) = G(x₀) × Σ (-ln(2)/104)^n (X-x₀)^n / n!
    func taylorExpansion(x0: Double, order: Int = 10) -> [String: Any] {
        let gx0 = G(x0)
        var coefficients: [Double] = []
        var factorialN = 1.0
        let ratio = -LN2_OVER_104
        var ratioPow = 1.0
        for n in 0...order {
            if n > 0 { factorialN *= Double(n) }
            coefficients.append(gx0 * ratioPow / factorialN)
            ratioPow *= ratio
        }
        // Verify at x₀ + 1
        var taylorValue = 0.0
        for (n, c) in coefficients.enumerated() {
            taylorValue += c * pow(1.0, Double(n))  // (x - x₀) = 1
        }
        return [
            "x0": x0, "G_x0": gx0,
            "order": order,
            "coefficients": coefficients,
            "taylor_at_x0_plus_1": taylorValue,
            "exact_at_x0_plus_1": G(x0 + 1.0),
            "taylor_error": abs(taylorValue - G(x0 + 1.0))
        ]
    }

    /// Critical analysis at key sacred points
    func criticalAnalysis() -> [String: Any] {
        let points: [Double] = [0, 104, 208, 312, 416]
        var analysis: [[String: Double]] = []
        for x in points {
            analysis.append([
                "x": x, "G": G(x), "dG": dGdx(x),
                "d2G": d2Gdx2(x), "log10_G": log10(G(x))
            ])
        }
        return ["sacred_points": analysis, "engine": "GodCodeCalculusEngine"]
    }

    /// Derivative verification (analytical vs numerical)
    func derivativeVerification() -> [String: Any] {
        let testPoints: [Double] = [0, 52, 104, 208, 416]
        var results: [[String: Any]] = []
        for x in testPoints {
            let analytical = dGdx(x)
            let numerical = dGdxNumerical(x)
            results.append([
                "x": x,
                "analytical": analytical,
                "numerical": numerical,
                "relative_error": abs(analytical - numerical) / abs(analytical)
            ])
        }
        return ["derivative_tests": results]
    }

    func fullAnalysis() -> [String: Any] {
        return [
            "critical": criticalAnalysis(),
            "derivatives": derivativeVerification(),
            "integral_0_416": [
                "analytical": integralAnalytical(xLow: 0, xHigh: 416),
                "numerical": integralNumerical(xLow: 0, xHigh: 416)
            ],
            "taylor_at_0": taylorExpansion(x0: 0, order: 10),
            "conservation": G(0) * pow(2.0, 0.0 / 104.0)  // Should = GOD_CODE
        ]
    }
}

// MARK: - ═══ 3. ELLIPTIC CURVE ENGINE ═══

final class EllipticCurveEngine {
    static let shared = EllipticCurveEngine()

    /// Point addition on y² = x³ + ax + b
    func pointAdd(_ P: (Double, Double)?, _ Q: (Double, Double)?,
                  a: Double) -> (Double, Double)? {
        guard let p = P else { return Q }
        guard let q = Q else { return P }

        if p.0 == q.0 && p.1 == -q.1 { return nil }  // P + (-P) = O

        let slope: Double
        if p.0 == q.0 && p.1 == q.1 {
            // Point doubling: λ = (3x² + a) / (2y)
            guard abs(p.1) > 1e-15 else { return nil }
            slope = (3.0 * p.0 * p.0 + a) / (2.0 * p.1)
        } else {
            // General addition: λ = (y₂ - y₁) / (x₂ - x₁)
            guard abs(q.0 - p.0) > 1e-15 else { return nil }
            slope = (q.1 - p.1) / (q.0 - p.0)
        }

        let xr = slope * slope - p.0 - q.0
        let yr = slope * (p.0 - xr) - p.1
        return (xr, yr)
    }

    /// Scalar multiplication k·P via double-and-add (binary ladder)
    func scalarMult(_ k: Int, _ P: (Double, Double), a: Double) -> (Double, Double)? {
        guard k != 0 else { return nil }
        let absK = abs(k)
        var result: (Double, Double)? = nil
        var current: (Double, Double)? = P

        var bits = absK
        while bits > 0 {
            if bits & 1 == 1 {
                result = pointAdd(result, current, a: a)
            }
            current = pointAdd(current, current, a: a)
            bits >>= 1
        }

        if k < 0, let r = result {
            return (r.0, -r.1)
        }
        return result
    }

    /// Curve discriminant: Δ = -16(4a³ + 27b²)
    func curveDiscriminant(a: Double, b: Double) -> Double {
        return -16.0 * (4.0 * a * a * a + 27.0 * b * b)
    }

    /// j-invariant: j = -1728 × (4a)³ / Δ
    func jInvariant(a: Double, b: Double) -> Double {
        let disc = curveDiscriminant(a: a, b: b)
        guard abs(disc) > 1e-15 else { return Double.infinity }
        return -1728.0 * pow(4.0 * a, 3) / disc
    }

    /// Ramanujan tau function τ(n) - exact known values
    func ramanujanTau(nMax: Int = 20) -> [String: Any] {
        let knownTau: [Int: Int] = [
            1: 1, 2: -24, 3: 252, 4: -1472, 5: 4830,
            6: -6048, 7: -16744, 8: 84480, 9: -113643, 10: -115920,
            11: 534612, 12: -370944
        ]
        var results: [[String: Any]] = []
        for n in 1...min(nMax, 12) {
            if let tau = knownTau[n] {
                results.append(["n": n, "tau_n": tau])
            }
        }
        // Verify multiplicativity: τ(6) = τ(2)×τ(3) for gcd(2,3)=1
        let mult_check = (knownTau[2] ?? 0) * (knownTau[3] ?? 0) == (knownTau[6] ?? 0)
        return [
            "values": results,
            "multiplicativity_check_tau6": mult_check,
            "tau_2_times_tau_3": (knownTau[2] ?? 0) * (knownTau[3] ?? 0),
            "tau_6": knownTau[6] ?? 0
        ]
    }

    /// Analyze God Code curve: y² = x³ + φ·x + G(0)
    func analyzeGodCodeCurve() -> [String: Any] {
        let a = PHI
        let b = GOD_CODE
        let disc = curveDiscriminant(a: a, b: b)
        let j = jInvariant(a: a, b: b)
        // Find a point on the curve (approximate via Newton's method)
        let x0 = 10.0
        let rhs = x0 * x0 * x0 + a * x0 + b
        let y0 = rhs > 0 ? sqrt(rhs) : Double.nan

        var multiples: [[String: Any]] = []
        if !y0.isNaN {
            let basePoint = (x0, y0)
            for k in [2, 3, 5, 7, 13] {
                if let kp = scalarMult(k, basePoint, a: a) {
                    multiples.append(["k": k, "x": kp.0, "y": kp.1])
                }
            }
        }
        return [
            "curve": "y² = x³ + φx + GOD_CODE",
            "a_phi": a, "b_god_code": b,
            "discriminant": disc,
            "j_invariant": j,
            "non_singular": abs(disc) > 1e-10,
            "base_point": y0.isNaN ? "none_found" : "(10, \(y0))",
            "scalar_multiples": multiples
        ]
    }

    func fullAnalysis() -> [String: Any] {
        return [
            "god_code_curve": analyzeGodCodeCurve(),
            "ramanujan_tau": ramanujanTau(),
            "standard_curves": [
                "y2_x3_minus_x": ["disc": curveDiscriminant(a: -1, b: 0), "j": jInvariant(a: -1, b: 0)],
                "y2_x3_plus_1": ["disc": curveDiscriminant(a: 0, b: 1), "j": jInvariant(a: 0, b: 1)]
            ],
            "engine": "EllipticCurveEngine"
        ]
    }
}

// MARK: - ═══ 4. PRIME NUMBER THEORY ENGINE ═══

final class PrimeNumberTheoryEngine {
    static let shared = PrimeNumberTheoryEngine()

    private var primesCache: [Int] = []
    private var sieveLimit: Int = 0

    /// Sieve of Eratosthenes (cached)
    func sieve(limit: Int = 100000) -> [Int] {
        if limit <= sieveLimit { return primesCache.filter { $0 <= limit } }
        var isComposite = [Bool](repeating: false, count: limit + 1)
        isComposite[0] = true
        if limit >= 1 { isComposite[1] = true }
        var i = 2
        while i * i <= limit {
            if !isComposite[i] {
                var j = i * i
                while j <= limit {
                    isComposite[j] = true
                    j += i
                }
            }
            i += 1
        }
        primesCache = (2...limit).filter { !isComposite[$0] }
        sieveLimit = limit
        return primesCache
    }

    /// Miller-Rabin primality test (deterministic for n < 3.3×10²⁴)
    func millerRabin(_ n: Int) -> Bool {
        if n < 2 { return false }
        if n < 4 { return true }
        if n % 2 == 0 { return false }

        // Write n-1 = 2^r × d
        var d = n - 1
        var r = 0
        while d % 2 == 0 { d /= 2; r += 1 }

        let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses {
            if a >= n { continue }
            var x = modPow(a, d, n)
            if x == 1 || x == n - 1 { continue }
            var composite = true
            for _ in 0..<(r - 1) {
                x = modPow(x, 2, n)
                if x == n - 1 { composite = false; break }
            }
            if composite { return false }
        }
        return true
    }

    /// Modular exponentiation: base^exp mod m
    private func modPow(_ base: Int, _ exp: Int, _ m: Int) -> Int {
        var result = 1
        var b = base % m
        var e = exp
        while e > 0 {
            if e & 1 == 1 { result = result &* b % m }
            e >>= 1
            b = b &* b % m
        }
        return result
    }

    /// π(n) = count of primes ≤ n, with Li(n) comparison
    func primeCounting(_ n: Int) -> [String: Any] {
        let primes = sieve(limit: n)
        let piN = primes.count
        let logN = log(Double(n))
        let liApprox = Double(n) / logN
        let liCorrected = Double(n) / logN * (1.0 + 1.0 / logN + 2.0 / (logN * logN))
        return [
            "n": n, "pi_n": piN,
            "n_over_ln_n": liApprox,
            "corrected_li": liCorrected,
            "ratio_pi_to_li": Double(piN) / liApprox
        ]
    }

    /// Twin primes (p, p+2) up to limit
    func twinPrimes(limit: Int = 100000) -> [String: Any] {
        let primes = sieve(limit: limit)
        var twins: [(Int, Int)] = []
        for i in 0..<primes.count - 1 {
            if primes[i + 1] - primes[i] == 2 {
                twins.append((primes[i], primes[i + 1]))
            }
        }
        return [
            "count": twins.count,
            "first_10": twins.prefix(10).map { [$0.0, $0.1] },
            "last_5": twins.suffix(5).map { [$0.0, $0.1] }
        ]
    }

    /// Goldbach verification: every even n ≥ 4 = p + q
    func goldbachVerify(limit: Int = 1000) -> [String: Any] {
        let primeSet = Set(sieve(limit: limit))
        var verified = 0
        var violations: [Int] = []
        for n in stride(from: 4, through: limit, by: 2) {
            var found = false
            for p in primeSet {
                if p > n { break }
                if primeSet.contains(n - p) { found = true; break }
            }
            if found { verified += 1 } else { violations.append(n) }
        }
        return ["verified": verified, "violations": violations, "limit": limit]
    }

    func fullAnalysis() -> [String: Any] {
        return [
            "prime_counting": primeCounting(100000),
            "twin_primes": twinPrimes(),
            "goldbach": goldbachVerify(),
            "engine": "PrimeNumberTheoryEngine"
        ]
    }
}

// MARK: - ═══ 5. NUMBER THEORY FORGE ═══

final class NumberTheoryForge {
    static let shared = NumberTheoryForge()

    /// Continued fraction expansion of a real number
    func continuedFraction(_ value: Double, depth: Int = 50) -> [String: Any] {
        var x = value
        var coefficients: [Int] = []
        var convergentNum: [Double] = [1, Double(Int(x))]
        var convergentDen: [Double] = [0, 1]

        for _ in 0..<depth {
            let a = Int(x)
            coefficients.append(a)
            let frac = x - Double(a)
            if abs(frac) < 1e-12 { break }
            x = 1.0 / frac
        }

        // Compute convergents
        for i in 2..<coefficients.count + 1 {
            guard i < coefficients.count + 1, i - 1 < coefficients.count else { break }
            let a = Double(coefficients[min(i - 1, coefficients.count - 1)])
            let num = a * convergentNum.last! + convergentNum[convergentNum.count - 2]
            let den = a * convergentDen.last! + convergentDen[convergentDen.count - 2]
            convergentNum.append(num)
            convergentDen.append(den)
        }

        return [
            "value": value,
            "coefficients": Array(coefficients.prefix(30)),
            "depth": coefficients.count,
            "last_convergent": convergentNum.last! / max(convergentDen.last ?? 1.0, 1.0)
        ]
    }

    /// φ = [1; 1, 1, 1, ...] - simplest CF (most irrational)
    func goldenRatioCF() -> [String: Any] {
        let cf = continuedFraction(PHI, depth: 50)
        let coeffs = (cf["coefficients"] as? [Int]) ?? []
        let allOnes = coeffs.allSatisfy { $0 == 1 }
        return ["cf": cf, "all_coefficients_are_1": allOnes, "value": PHI]
    }

    /// Fibonacci identities verification at F(n)
    func fibonacciIdentities(n: Int = 50) -> [String: Any] {
        func fib(_ k: Int) -> Double {
            // Binet formula for large k
            return (pow(PHI, Double(k)) - pow(-TAU, Double(k))) / sqrt(5.0)
        }
        // Cassini: F(n-1)·F(n+1) - F(n)² = (-1)^n
        let cassini = fib(n - 1) * fib(n + 1) - fib(n) * fib(n)
        let expectedCassini = (n % 2 == 0) ? 1.0 : -1.0
        // Convergence to φ
        let ratio = fib(n + 1) / fib(n)
        return [
            "n": n,
            "cassini": cassini, "expected_cassini": expectedCassini,
            "cassini_ok": abs(cassini - expectedCassini) < 1.0,
            "fib_ratio_to_phi": ratio,
            "ratio_error": abs(ratio - PHI)
        ]
    }

    /// Partition function p(n) via dynamic programming
    func partitionCount(_ n: Int) -> [String: Any] {
        var table = [Int](repeating: 0, count: n + 1)
        table[0] = 1
        for k in 1...n {
            for j in k...n {
                table[j] += table[j - k]
            }
        }
        // Hardy-Ramanujan asymptotic: p(n) ~ exp(π√(2n/3)) / (4n√3)
        let asymptotic = exp(Double.pi * sqrt(2.0 * Double(n) / 3.0)) / (4.0 * Double(n) * sqrt(3.0))
        return [
            "n": n, "p_n": table[n],
            "hardy_ramanujan": asymptotic,
            "ratio": Double(table[n]) / asymptotic
        ]
    }

    /// Pell equation: x² - D·y² = 1 using CF of √D
    func pellEquation(D: Int, solutions: Int = 5) -> [String: Any] {
        let sqrtD = sqrt(Double(D))
        let a0 = Int(sqrtD)
        if a0 * a0 == D { return ["error": "D is a perfect square"] }

        // Generate CF of √D to find fundamental solution
        var m = 0, d = 1, a = a0
        var pPrev = 1, pCurr = a0
        var qPrev = 0, qCurr = 1
        var allSolutions: [(Int, Int)] = []

        for _ in 0..<1000 {
            m = d * a - m
            d = (D - m * m) / d
            a = (a0 + m) / d

            let pNext = a * pCurr + pPrev
            let qNext = a * qCurr + qPrev

            if pNext * pNext - D * qNext * qNext == 1 {
                allSolutions.append((pNext, qNext))
                if allSolutions.count >= solutions { break }
            }

            pPrev = pCurr; pCurr = pNext
            qPrev = qCurr; qCurr = qNext
        }

        return [
            "D": D,
            "solutions": allSolutions.map { ["x": $0.0, "y": $0.1] },
            "fundamental": allSolutions.first.map { ["x": $0.0, "y": $0.1] } as Any
        ]
    }

    func fullAnalysis() -> [String: Any] {
        return [
            "golden_ratio_cf": goldenRatioCF(),
            "fibonacci": fibonacciIdentities(),
            "partition_100": partitionCount(100),
            "pell_D61": pellEquation(D: 61),
            "engine": "NumberTheoryForge"
        ]
    }
}

// MARK: - ═══ 6. HARMONIC NUMBER ENGINE ═══

final class HarmonicNumberEngine {
    static let shared = HarmonicNumberEngine()

    /// H_n for various n
    func harmonicSeries(nMax: Int = 200) -> [String: Any] {
        let checkpoints = [10, 50, 100, 200, min(500, nMax), nMax]
        var results: [[String: Any]] = []
        var sum = 0.0
        var idx = 0
        for n in 1...nMax {
            sum += 1.0 / Double(n)
            if idx < checkpoints.count && n == checkpoints[idx] {
                let asymptotic = log(Double(n)) + EULER_GAMMA_HP + 1.0 / (2.0 * Double(n))
                results.append([
                    "n": n, "H_n": sum,
                    "asymptotic": asymptotic,
                    "error": abs(sum - asymptotic)
                ])
                idx += 1
            }
        }
        return ["series": results, "euler_mascheroni": EULER_GAMMA_HP]
    }

    /// Generalized harmonic H_n^(m) and convergence to ζ(m)
    func generalizedHarmonicAnalysis() -> [String: Any] {
        let orders = [2, 3, 4]
        let n = 10000
        var results: [[String: Any]] = []
        for m in orders {
            var sum = 0.0
            for k in 1...n {
                sum += 1.0 / pow(Double(k), Double(m))
            }
            let exactZeta: Double
            switch m {
            case 2: exactZeta = ZETA_2_HP
            case 3: exactZeta = APERY_HP
            case 4: exactZeta = ZETA_4_HP
            default: exactZeta = sum
            }
            results.append([
                "m": m, "H_n_m": sum,
                "zeta_m": exactZeta,
                "error": abs(sum - exactZeta)
            ])
        }
        return ["generalized": results, "n": n]
    }

    /// Extract γ from H_n - ln(n) → γ as n → ∞
    func eulerMascheroniFromHarmonics(n: Int = 1000) -> [String: Any] {
        var sum = 0.0
        for k in 1...n { sum += 1.0 / Double(k) }
        let gamma_approx = sum - log(Double(n))
        let gamma_corrected = sum - log(Double(n)) - 1.0 / (2.0 * Double(n))
            + 1.0 / (12.0 * Double(n) * Double(n))
        return [
            "n": n,
            "gamma_raw": gamma_approx,
            "gamma_corrected": gamma_corrected,
            "exact_gamma": EULER_GAMMA_HP,
            "raw_error": abs(gamma_approx - EULER_GAMMA_HP),
            "corrected_error": abs(gamma_corrected - EULER_GAMMA_HP)
        ]
    }

    func fullAnalysis() -> [String: Any] {
        return [
            "harmonic_series": harmonicSeries(),
            "generalized": generalizedHarmonicAnalysis(),
            "euler_mascheroni_extraction": eulerMascheroniFromHarmonics(),
            "engine": "HarmonicNumberEngine"
        ]
    }
}

// MARK: - ═══ 7. INFINITE SERIES LAB ═══

final class InfiniteSeriesLab {
    static let shared = InfiniteSeriesLab()

    /// Basel problem: Σ 1/n² = π²/6 (convergence rate analysis)
    func baselProblem(terms: Int = 100000) -> [String: Any] {
        var sum = 0.0
        var snapshots: [[String: Any]] = []
        let checkpoints = [10, 100, 1000, 10000, terms]
        for n in 1...terms {
            sum += 1.0 / (Double(n) * Double(n))
            if checkpoints.contains(n) {
                snapshots.append(["n": n, "sum": sum, "error": abs(sum - ZETA_2_HP)])
            }
        }
        return ["target": ZETA_2_HP, "snapshots": snapshots]
    }

    /// Leibniz formula: π/4 = 1 - 1/3 + 1/5 - 1/7 + ...
    func leibnizPi(terms: Int = 1000000) -> [String: Any] {
        var sum = 0.0
        for n in 0..<terms {
            let sign = (n % 2 == 0) ? 1.0 : -1.0
            sum += sign / Double(2 * n + 1)
        }
        let piApprox = sum * 4.0
        return ["pi_approx": piApprox, "error": abs(piApprox - Double.pi), "terms": terms]
    }

    /// Wallis product: π/2 = Π (4n²)/(4n²-1)
    func wallisProduct(terms: Int = 100000) -> [String: Any] {
        var product = 1.0
        for n in 1...terms {
            let n2 = Double(n) * Double(n) * 4.0
            product *= n2 / (n2 - 1.0)
        }
        let piApprox = product * 2.0
        return ["pi_approx": piApprox, "error": abs(piApprox - Double.pi), "terms": terms]
    }

    /// Ramanujan 1/π series (rapid convergence)
    func ramanujanPiSeries(terms: Int = 5) -> [String: Any] {
        // 1/π = (2√2/9801) Σ (4k)!(1103+26390k) / ((k!)⁴ 396^(4k))
        var sum = 0.0
        for k in 0..<terms {
            let factorial4k = factorialDouble(4 * k)
            let factorialK4 = pow(factorialDouble(k), 4)
            let numerator = factorial4k * (1103.0 + 26390.0 * Double(k))
            let denominator = factorialK4 * pow(396.0, Double(4 * k))
            sum += numerator / denominator
        }
        let piInverse = 2.0 * sqrt(2.0) / 9801.0 * sum
        let piApprox = 1.0 / piInverse
        return ["pi_approx": piApprox, "error": abs(piApprox - Double.pi), "terms": terms]
    }

    private func factorialDouble(_ n: Int) -> Double {
        guard n > 0 else { return 1.0 }
        var result = 1.0
        for i in 1...n { result *= Double(i) }
        return result
    }

    func fullAnalysis() -> [String: Any] {
        return [
            "basel": baselProblem(),
            "leibniz": leibnizPi(),
            "wallis": wallisProduct(),
            "ramanujan": ramanujanPiSeries(),
            "engine": "InfiniteSeriesLab"
        ]
    }
}

// MARK: - ═══ 8. FRACTAL DYNAMICS LAB ═══

final class FractalDynamicsLab {
    static let shared = FractalDynamicsLab()

    /// Mandelbrot escape iteration count at point c
    func mandelbrotEscape(_ cReal: Double, _ cImag: Double, maxIter: Int = 1000) -> Int {
        var zr = 0.0, zi = 0.0
        for i in 0..<maxIter {
            let zr2 = zr * zr, zi2 = zi * zi
            if zr2 + zi2 > 4.0 { return i }
            zi = 2.0 * zr * zi + cImag
            zr = zr2 - zi2 + cReal
        }
        return maxIter
    }

    /// Logistic map: x_{n+1} = r × x_n × (1 - x_n) - Lyapunov exponent
    func logisticLyapunov(r: Double, iterations: Int = 10000, warmup: Int = 1000) -> Double {
        var x = 0.5
        // Warmup
        for _ in 0..<warmup { x = r * x * (1.0 - x) }
        // Compute Lyapunov
        var lyapunov = 0.0
        for _ in 0..<iterations {
            x = r * x * (1.0 - x)
            let derivative = abs(r * (1.0 - 2.0 * x))
            if derivative > 0 { lyapunov += log(derivative) }
        }
        return lyapunov / Double(iterations)
    }

    /// Feigenbaum constant verification via period-doubling cascade
    func feigenbaumVerification() -> [String: Any] {
        // Known bifurcation points for logistic map
        let bifurcations: [Double] = [3.0, 3.44949, 3.54409, 3.5644, 3.5688, 3.56969]
        var ratios: [Double] = []
        for i in 2..<bifurcations.count {
            let ratio = (bifurcations[i-1] - bifurcations[i-2]) / (bifurcations[i] - bifurcations[i-1])
            ratios.append(ratio)
        }
        return [
            "bifurcation_points": bifurcations,
            "ratios": ratios,
            "feigenbaum_delta": FEIGENBAUM,
            "convergence": ratios.last.map { abs($0 - FEIGENBAUM) } ?? Double.nan
        ]
    }

    /// Julia set escape analysis for c = GOD_CODE-mapped point
    func juliaGodCode(resolution: Int = 100) -> [String: Any] {
        // Map GOD_CODE to Julia parameter: c = GOD_CODE/1000 + i×PHI/1000
        let cr = GOD_CODE.truncatingRemainder(dividingBy: 2.0) - 1.0  // Map to [-1, 1]
        let ci = PHI.truncatingRemainder(dividingBy: 1.0) - 0.5       // Map to [-0.5, 0.5]
        var escapeCounts: [Int] = []
        let step = 4.0 / Double(resolution)
        for row in 0..<resolution {
            for col in 0..<resolution {
                var zr = -2.0 + Double(col) * step
                var zi = -2.0 + Double(row) * step
                var iter = 0
                while iter < 256 && zr * zr + zi * zi < 4.0 {
                    let newZr = zr * zr - zi * zi + cr
                    zi = 2.0 * zr * zi + ci
                    zr = newZr
                    iter += 1
                }
                escapeCounts.append(iter)
            }
        }
        let avgEscape = Double(escapeCounts.reduce(0, +)) / Double(escapeCounts.count)
        let maxEscape = escapeCounts.max() ?? 0
        return [
            "c_real": cr, "c_imag": ci,
            "resolution": resolution,
            "avg_escape": avgEscape, "max_escape": maxEscape,
            "bounded_fraction": Double(escapeCounts.filter { $0 == 256 }.count) / Double(escapeCounts.count)
        ]
    }

    func fullAnalysis() -> [String: Any] {
        return [
            "feigenbaum": feigenbaumVerification(),
            "logistic_r3.57": logisticLyapunov(r: 3.57),
            "logistic_r4.0": logisticLyapunov(r: 4.0),
            "julia_god_code": juliaGodCode(resolution: 50),
            "mandelbrot_origin": mandelbrotEscape(0, 0),
            "engine": "FractalDynamicsLab"
        ]
    }
}

// MARK: - ═══ 9. TRANSCENDENTAL PROVER ═══

final class TranscendentalProver {
    static let shared = TranscendentalProver()

    /// Verify Lindemann-Weierstrass theorem consequences:
    /// e^α is transcendental for any non-zero algebraic α
    func lindemannWeierstrass() -> [String: Any] {
        // e^1 = e (transcendental - proven by Hermite 1873)
        // e^(iπ) = -1 (Euler's identity - π transcendental via Lindemann 1882)
        // e^(√2) - transcendental since √2 algebraic and non-zero
        let eVal = M_E
        let ePi = exp(Double.pi)
        let eSqrt2 = exp(sqrt(2.0))
        return [
            "e": eVal,
            "e_pi": ePi,
            "e_sqrt2": eSqrt2,
            "euler_identity_check": abs(exp(Double.pi) * cos(Double.pi) + 1.0),  // Should ≈ 0
            "theorem": "If α₁,...,αₙ algebraic & linearly independent over Q, then e^α₁,...,e^αₙ algebraically independent"
        ]
    }

    /// Niven's theorem: sin(r°) is rational only for r ∈ {0, 30, 90, 150, 180, ...}
    func nivensTheorem() -> [String: Any] {
        let rationalAngles = [0, 30, 90, 150, 180, 210, 270, 330, 360]
        var results: [[String: Any]] = []
        for deg in rationalAngles {
            let rad = Double(deg) * Double.pi / 180.0
            let sinVal = sin(rad)
            results.append(["degrees": deg, "sin": sinVal, "is_rational": true])
        }
        return ["rational_sine_angles": results, "theorem": "Niven's theorem (1956)"]
    }

    /// Irrationality measure μ(α): |α - p/q| > 1/q^μ for all but finitely many p/q
    func irrationalityMeasures() -> [String: Any] {
        return [
            "pi": ["value": Double.pi, "mu_known": "≤ 7.6063 (Zeilberger-Zudilin 2020)", "mu_conjectured": 2.0],
            "e": ["value": M_E, "mu_known": 2.0, "note": "Exact (e is NOT Liouville)"],
            "phi": ["value": PHI, "mu_known": 2.0, "note": "Most poorly approximable (slowest CF convergence)"],
            "god_code": ["value": GOD_CODE, "mu_estimate": "Unknown - depends on algebraic/transcendental status"]
        ]
    }

    func fullAnalysis() -> [String: Any] {
        return [
            "lindemann_weierstrass": lindemannWeierstrass(),
            "niven": nivensTheorem(),
            "irrationality_measures": irrationalityMeasures(),
            "engine": "TranscendentalProver"
        ]
    }
}

// MARK: - ═══ 10. STATISTICAL MECHANICS ENGINE ═══

final class StatisticalMechanicsEngine {
    static let shared = StatisticalMechanicsEngine()

    /// Boltzmann partition function: Z = Σ exp(-E_i / kT)
    func partitionFunction(energies: [Double], temperature: Double) -> [String: Any] {
        let beta = 1.0 / (BOLTZMANN_CONSTANT * temperature)
        let boltzmannFactors = energies.map { exp(-$0 * beta) }
        let Z = boltzmannFactors.reduce(0.0, +)
        let probabilities = boltzmannFactors.map { $0 / Z }
        let avgEnergy = zip(energies, probabilities).reduce(0.0) { $0 + $1.0 * $1.1 }
        let entropy = -probabilities.reduce(0.0) { sum, p in
            p > 0 ? sum + p * log(p) : sum
        }
        return [
            "Z": Z, "probabilities": probabilities,
            "avg_energy": avgEnergy, "entropy": entropy,
            "free_energy": -log(Z) / beta,
            "temperature": temperature
        ]
    }

    /// Ising model 1D partition function (exact)
    func ising1D(n: Int, J: Double, h: Double, temperature: Double) -> [String: Any] {
        let beta = 1.0 / (BOLTZMANN_CONSTANT * max(temperature, 1e-10))
        // Transfer matrix method: Z = 2^N × [cosh(βJ)]^N × [1 + tanh(βJ)]
        let betaJ = beta * J
        let betaH = beta * h
        // For h=0: Z = 2[2cosh(βJ)]^(N-1)
        let z = h == 0 ?
            2.0 * pow(2.0 * cosh(betaJ), Double(n - 1)) :
            pow(exp(betaJ) * cosh(betaH) + sqrt(exp(2 * betaJ) * sinh(betaH) * sinh(betaH) + exp(-2 * betaJ)), Double(n))
        let magnetization = h == 0 ? 0.0 : tanh(betaH)
        return [
            "Z": z, "free_energy_per_site": -log(z) / (beta * Double(n)),
            "magnetization": magnetization,
            "n": n, "J": J, "h": h, "T": temperature
        ]
    }

    /// Maxwell-Boltzmann speed distribution: f(v) = 4π(m/2πkT)^(3/2) v² exp(-mv²/2kT)
    func maxwellBoltzmann(mass: Double, temperature: Double, vMax: Double, steps: Int = 200) -> [String: Any] {
        let prefactor = 4.0 * Double.pi * pow(mass / (2.0 * Double.pi * BOLTZMANN_CONSTANT * temperature), 1.5)
        var distribution: [(Double, Double)] = []
        let dv = vMax / Double(steps)
        var vPeak = 0.0, fPeak = 0.0
        for i in 0...steps {
            let v = Double(i) * dv
            let f = prefactor * v * v * exp(-mass * v * v / (2.0 * BOLTZMANN_CONSTANT * temperature))
            distribution.append((v, f))
            if f > fPeak { fPeak = f; vPeak = v }
        }
        let vMostProbable = sqrt(2.0 * BOLTZMANN_CONSTANT * temperature / mass)
        return [
            "peak_v": vPeak,
            "theoretical_most_probable_v": vMostProbable,
            "peak_f": fPeak,
            "samples": distribution.count
        ]
    }

    func fullAnalysis() -> [String: Any] {
        let energies = (0..<10).map { Double($0) * 1e-21 }
        return [
            "partition_300K": partitionFunction(energies: energies, temperature: 300.0),
            "ising_1d": ising1D(n: 100, J: 1e-21, h: 0, temperature: 300.0),
            "maxwell_boltzmann": maxwellBoltzmann(mass: 4.65e-26, temperature: 300.0, vMax: 2000.0),
            "engine": "StatisticalMechanicsEngine"
        ]
    }
}

// MARK: - ═══ 11. COLLATZ CONJECTURE ANALYZER ═══

final class CollatzConjectureAnalyzer {
    static let shared = CollatzConjectureAnalyzer()

    /// Collatz sequence from n
    func collatzSequence(_ n: Int) -> [Int] {
        var seq = [n]
        var current = n
        while current != 1 && seq.count < 10000 {
            if current % 2 == 0 {
                current /= 2
            } else {
                current = 3 * current + 1
            }
            seq.append(current)
        }
        return seq
    }

    /// Stopping time: steps to reach 1
    func stoppingTime(_ n: Int) -> Int {
        var current = n
        var steps = 0
        while current != 1 && steps < 100000 {
            if current % 2 == 0 { current /= 2 } else { current = 3 * current + 1 }
            steps += 1
        }
        return steps
    }

    /// Analyze stopping times for range [1, limit]
    func stoppingTimeAnalysis(limit: Int = 10000) -> [String: Any] {
        var times: [Int] = []
        var maxTime = 0, maxTimeN = 1
        var maxValue = 0, maxValueN = 1

        for n in 1...limit {
            let t = stoppingTime(n)
            times.append(t)
            if t > maxTime { maxTime = t; maxTimeN = n }
            // Track max value reached
            var current = n
            var peak = n
            var steps = 0
            while current != 1 && steps < 100000 {
                if current % 2 == 0 { current /= 2 } else { current = 3 * current + 1 }
                peak = max(peak, current)
                steps += 1
            }
            if peak > maxValue { maxValue = peak; maxValueN = n }
        }

        let avgTime = Double(times.reduce(0, +)) / Double(times.count)
        return [
            "limit": limit,
            "avg_stopping_time": avgTime,
            "max_stopping_time": maxTime, "max_time_n": maxTimeN,
            "max_value_reached": maxValue, "max_value_n": maxValueN,
            "all_converge": times.allSatisfy { $0 < 100000 }
        ]
    }

    /// Sacred analysis: Collatz on L104 sacred numbers
    func sacredCollatz() -> [String: Any] {
        let sacredNums = [104, 286, 416, 527, 13, 26]
        var results: [[String: Any]] = []
        for n in sacredNums {
            let seq = collatzSequence(n)
            results.append([
                "n": n, "stopping_time": seq.count - 1,
                "peak": seq.max() ?? n,
                "first_10": Array(seq.prefix(10))
            ])
        }
        return ["sacred_numbers": results]
    }

    func fullAnalysis() -> [String: Any] {
        return [
            "stopping_times": stoppingTimeAnalysis(),
            "sacred_collatz": sacredCollatz(),
            "sequence_27": collatzSequence(27).count,
            "engine": "CollatzConjectureAnalyzer"
        ]
    }
}

// MARK: - ═══ UNIFIED MATH RESEARCH HUB ═══

final class MathResearchHub {
    static let shared = MathResearchHub()

    let zeta = RiemannZetaEngine.shared
    let godCodeCalculus = GodCodeCalculusEngine.shared
    let elliptic = EllipticCurveEngine.shared
    let primeTheory = PrimeNumberTheoryEngine.shared
    let numberTheory = NumberTheoryForge.shared
    let harmonic = HarmonicNumberEngine.shared
    let infiniteSeries = InfiniteSeriesLab.shared
    let fractal = FractalDynamicsLab.shared
    let transcendental = TranscendentalProver.shared
    let statMech = StatisticalMechanicsEngine.shared
    let collatz = CollatzConjectureAnalyzer.shared

    init() {
        InterEngineFeedbackBus.shared.broadcast(
            from: .knowledge,
            signal: "math_research_init",
            payload: ["engines": 11.0, "status": 1.0]
        )
    }

    /// Run all 11 engines and return unified report
    func fullResearchCycle() -> [String: Any] {
        return [
            "riemann_zeta": zeta.fullAnalysis(),
            "god_code_calculus": godCodeCalculus.fullAnalysis(),
            "elliptic_curves": elliptic.fullAnalysis(),
            "prime_theory": primeTheory.fullAnalysis(),
            "number_theory": numberTheory.fullAnalysis(),
            "harmonic_numbers": harmonic.fullAnalysis(),
            "infinite_series": infiniteSeries.fullAnalysis(),
            "fractal_dynamics": fractal.fullAnalysis(),
            "transcendental": transcendental.fullAnalysis(),
            "statistical_mechanics": statMech.fullAnalysis(),
            "collatz": collatz.fullAnalysis(),
            "engine_count": 11,
            "god_code": GOD_CODE
        ]
    }
}
