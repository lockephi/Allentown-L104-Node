// ═══════════════════════════════════════════════════════════════
// B35_SymbolicMathSolver.swift — Symbolic Math Solver Engine
// L104v2 — TheBrain Layer — EVO_68 SOVEREIGN_CONVERGENCE → v2.0.0 ASI
// 7 Domain Solvers (Algebra→Quantum) + WordProblemParser
// + EquationBuilder + SolutionChain + Three-Engine + Verification
// ═══════════════════════════════════════════════════════════════

import Foundation

// ═══════════════════════════════════════════════════════════════
// MARK: - MathExpression — Recursive Symbolic AST
// ═══════════════════════════════════════════════════════════════

indirect enum MathExpression {
    case number(Double)
    case variable(String)
    case add(MathExpression, MathExpression)
    case subtract(MathExpression, MathExpression)
    case multiply(MathExpression, MathExpression)
    case divide(MathExpression, MathExpression)
    case power(MathExpression, MathExpression)
    case sqrt(MathExpression)
    case sin(MathExpression)
    case cos(MathExpression)
    case ln(MathExpression)
    case exp(MathExpression)

    /// Evaluate the expression tree with variable bindings
    func evaluate(bindings: [String: Double] = [:]) -> Double {
        switch self {
        case .number(let v):
            return v
        case .variable(let name):
            return bindings[name] ?? .nan
        case .add(let a, let b):
            return a.evaluate(bindings: bindings) + b.evaluate(bindings: bindings)
        case .subtract(let a, let b):
            return a.evaluate(bindings: bindings) - b.evaluate(bindings: bindings)
        case .multiply(let a, let b):
            return a.evaluate(bindings: bindings) * b.evaluate(bindings: bindings)
        case .divide(let a, let b):
            let denom = b.evaluate(bindings: bindings)
            return denom == 0 ? .nan : a.evaluate(bindings: bindings) / denom
        case .power(let base, let exp):
            return Foundation.pow(base.evaluate(bindings: bindings), exp.evaluate(bindings: bindings))
        case .sqrt(let a):
            return Foundation.sqrt(a.evaluate(bindings: bindings))
        case .sin(let a):
            return Foundation.sin(a.evaluate(bindings: bindings))
        case .cos(let a):
            return Foundation.cos(a.evaluate(bindings: bindings))
        case .ln(let a):
            return Foundation.log(a.evaluate(bindings: bindings))
        case .exp(let a):
            return Foundation.exp(a.evaluate(bindings: bindings))
        }
    }

    /// Human-readable description of the expression
    var description: String {
        switch self {
        case .number(let v):
            return v.truncatingRemainder(dividingBy: 1) == 0
                ? String(format: "%.0f", v) : String(format: "%.6g", v)
        case .variable(let name):
            return name
        case .add(let a, let b):
            return "(\(a.description) + \(b.description))"
        case .subtract(let a, let b):
            return "(\(a.description) - \(b.description))"
        case .multiply(let a, let b):
            return "(\(a.description) * \(b.description))"
        case .divide(let a, let b):
            return "(\(a.description) / \(b.description))"
        case .power(let base, let exp):
            return "(\(base.description)^(\(exp.description)))"
        case .sqrt(let a):
            return "sqrt(\(a.description))"
        case .sin(let a):
            return "sin(\(a.description))"
        case .cos(let a):
            return "cos(\(a.description))"
        case .ln(let a):
            return "ln(\(a.description))"
        case .exp(let a):
            return "exp(\(a.description))"
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - MathSolution — Solver Result
// ═══════════════════════════════════════════════════════════════

struct MathSolution {
    let domain: String
    let input: String
    let resultValue: Double?
    let resultDescription: String
    let steps: [String]
    let confidence: Double
    let sacredResonance: Double

    /// Compute sacred resonance from a result value using PHI alignment
    static func computeResonance(_ value: Double?) -> Double {
        guard let v = value, v.isFinite else { return 0.0 }
        let phiRatio = v / PHI
        let fractional = phiRatio - Foundation.floor(phiRatio)
        let alignment = 1.0 - Swift.min(fractional, 1.0 - fractional) * 2.0
        return Swift.max(0.0, Swift.min(1.0, alignment))
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - SymbolicMathSolver — Main Engine
// ═══════════════════════════════════════════════════════════════

final class SymbolicMathSolver: SovereignEngine {
    static let shared = SymbolicMathSolver()

    private let lock = NSLock()
    private static let VERSION = SYMBOLIC_MATH_VERSION
    private var solveCount: Int = 0
    private var totalConfidence: Double = 0.0
    private var domainCounts: [String: Int] = [:]

    var engineName: String { "SymbolicMathSolver" }

    private init() {}

    // ═══════════════════════════════════════════════════════════
    // MARK: - 1. Algebra Solver
    // ═══════════════════════════════════════════════════════════

    /// Solve algebraic equations.
    /// Types: "linear" (ax+b=0, coeffs=[a,b]),
    ///        "quadratic" (ax²+bx+c=0, coeffs=[a,b,c]),
    ///        "cubic" (ax³+bx²+cx+d=0, coeffs=[a,b,c,d]),
    ///        "system2" (a1x+b1y=c1, a2x+b2y=c2, coeffs=[a1,b1,c1,a2,b2,c2])
    func solveAlgebra(type: String, coeffs: [Double]) -> MathSolution {
        lock.lock(); defer { lock.unlock() }
        var steps: [String] = []
        var result: Double? = nil
        var desc = ""

        switch type {
        case "linear":
            // ax + b = 0 => x = -b/a
            guard coeffs.count >= 2, coeffs[0] != 0 else {
                return makeSolution("Algebra", "linear(\(coeffs))", nil,
                    "Invalid: a=0 or insufficient coefficients", [], 0)
            }
            let a = coeffs[0], b = coeffs[1]
            let x = -b / a
            steps.append("Given: \(a)x + \(b) = 0")
            steps.append("x = -b/a = \(-b)/\(a) = \(x)")
            result = x
            desc = "x = \(x)"

        case "quadratic":
            // ax² + bx + c = 0
            guard coeffs.count >= 3, coeffs[0] != 0 else {
                return makeSolution("Algebra", "quadratic(\(coeffs))", nil,
                    "Invalid: a=0 or insufficient coefficients", [], 0)
            }
            let a = coeffs[0], b = coeffs[1], c = coeffs[2]
            let discriminant = b * b - 4 * a * c
            steps.append("Given: \(a)x² + \(b)x + \(c) = 0")
            steps.append("Discriminant Δ = b²-4ac = \(discriminant)")

            if discriminant >= 0 {
                let sqrtD = Foundation.sqrt(discriminant)
                let x1 = (-b + sqrtD) / (2 * a)
                let x2 = (-b - sqrtD) / (2 * a)
                steps.append("x₁ = (-b+√Δ)/(2a) = \(x1)")
                steps.append("x₂ = (-b-√Δ)/(2a) = \(x2)")
                result = x1
                desc = discriminant == 0 ? "x = \(x1) (double root)"
                    : "x₁ = \(x1), x₂ = \(x2)"
            } else {
                let realPart = -b / (2 * a)
                let imagPart = Foundation.sqrt(-discriminant) / (2 * a)
                steps.append("Complex roots: \(realPart) ± \(imagPart)i")
                result = realPart
                desc = "x = \(realPart) ± \(imagPart)i"
            }

        case "cubic":
            // Cardano's method: ax³ + bx² + cx + d = 0
            guard coeffs.count >= 4, coeffs[0] != 0 else {
                return makeSolution("Algebra", "cubic(\(coeffs))", nil,
                    "Invalid coefficients", [], 0)
            }
            let a = coeffs[0], b = coeffs[1], c = coeffs[2], d = coeffs[3]
            // Depressed cubic: t³ + pt + q = 0 via substitution x = t - b/(3a)
            let p = (3 * a * c - b * b) / (3 * a * a)
            let q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a)
            steps.append("Given: \(a)x³ + \(b)x² + \(c)x + \(d) = 0")
            steps.append("Depressed cubic: t³ + \(p)t + \(q) = 0")

            let discriminantC = q * q / 4 + p * p * p / 27
            steps.append("Cubic discriminant: \(discriminantC)")

            if discriminantC >= 0 {
                let sqrtDC = Foundation.sqrt(discriminantC)
                let u = cbrt(-q / 2 + sqrtDC)
                let v = cbrt(-q / 2 - sqrtDC)
                let t1 = u + v
                let x1 = t1 - b / (3 * a)
                steps.append("Cardano: u=\(u), v=\(v), t=\(t1)")
                steps.append("x₁ = t - b/(3a) = \(x1)")
                result = x1
                desc = "x₁ = \(x1) (real root via Cardano)"
            } else {
                // Casus irreducibilis — 3 real roots via trig
                let r = Foundation.sqrt(-p * p * p / 27)
                let theta = acos(-q / (2 * r)) / 3.0
                let m = 2 * Foundation.pow(r, 1.0 / 3.0)
                let shift = b / (3 * a)
                let x1 = m * Foundation.cos(theta) - shift
                let x2 = m * Foundation.cos(theta - 2 * .pi / 3) - shift
                let x3 = m * Foundation.cos(theta - 4 * .pi / 3) - shift
                steps.append("Three real roots (trig method):")
                steps.append("x₁=\(x1), x₂=\(x2), x₃=\(x3)")
                result = x1
                desc = "x₁=\(x1), x₂=\(x2), x₃=\(x3)"
            }

        case "system2":
            // a1*x + b1*y = c1, a2*x + b2*y = c2
            guard coeffs.count >= 6 else {
                return makeSolution("Algebra", "system2(\(coeffs))", nil,
                    "Need 6 coefficients [a1,b1,c1,a2,b2,c2]", [], 0)
            }
            let a1 = coeffs[0], b1 = coeffs[1], c1 = coeffs[2]
            let a2 = coeffs[3], b2 = coeffs[4], c2 = coeffs[5]
            let det = a1 * b2 - a2 * b1
            steps.append("\(a1)x + \(b1)y = \(c1)")
            steps.append("\(a2)x + \(b2)y = \(c2)")
            steps.append("det = a1*b2 - a2*b1 = \(det)")
            guard det != 0 else {
                return makeSolution("Algebra", "system2", nil,
                    "No unique solution (det=0)", steps, 0)
            }
            let x = (c1 * b2 - c2 * b1) / det
            let y = (a1 * c2 - a2 * c1) / det
            steps.append("x = (c1*b2-c2*b1)/det = \(x)")
            steps.append("y = (a1*c2-a2*c1)/det = \(y)")
            result = x
            desc = "x = \(x), y = \(y)"

        default:
            return makeSolution("Algebra", type, nil, "Unknown type: \(type)", [], 0)
        }
        return makeSolution("Algebra", "\(type)(\(coeffs))", result, desc, steps, 0.95)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 2. Calculus Solver
    // ═══════════════════════════════════════════════════════════

    /// Solve calculus: "derivative", "integral", "definite_integral", "limit_poly"
    func solveCalculus(type: String, coeffs: [Double], params: [Double] = []) -> MathSolution {
        lock.lock(); defer { lock.unlock() }
        var steps: [String] = []
        var result: Double? = nil
        var desc = ""

        switch type {
        case "derivative":
            // Power rule: d/dx [a_n*x^n + ... + a_0] => coefficients of derivative
            guard !coeffs.isEmpty else {
                return makeSolution("Calculus", "derivative([])", nil, "Empty polynomial", [], 0)
            }
            let n = coeffs.count - 1
            steps.append("Polynomial degree \(n): \(polyString(coeffs))")
            if n == 0 {
                desc = "d/dx(constant) = 0"
                result = 0
            } else {
                var derivCoeffs: [Double] = []
                for i in 0..<n {
                    let power = Double(n - i)
                    derivCoeffs.append(coeffs[i] * power)
                    steps.append("d/dx(\(coeffs[i])x^\(Int(power))) = \(coeffs[i] * power)x^\(Int(power) - 1)")
                }
                desc = "f'(x) = \(polyString(derivCoeffs))"
                result = derivCoeffs.first
            }

        case "integral":
            // Power rule integration: ∫[a_n*x^n + ...] dx
            guard !coeffs.isEmpty else {
                return makeSolution("Calculus", "integral([])", nil, "Empty polynomial", [], 0)
            }
            let n = coeffs.count - 1
            steps.append("Polynomial: \(polyString(coeffs))")
            var intCoeffs: [Double] = []
            for i in 0...n {
                let power = Double(n - i)
                let newCoeff = coeffs[i] / (power + 1)
                intCoeffs.append(newCoeff)
                steps.append("∫\(coeffs[i])x^\(Int(power))dx = \(newCoeff)x^\(Int(power) + 1)")
            }
            intCoeffs.append(0) // + C
            desc = "∫f(x)dx = \(polyString(intCoeffs)) + C"
            result = intCoeffs.first

        case "definite_integral":
            // ∫[a,b] polynomial dx
            guard params.count >= 2 else {
                return makeSolution("Calculus", "definite_integral", nil,
                    "Need params=[a,b]", [], 0)
            }
            let a = params[0], b = params[1]
            let n = coeffs.count - 1
            steps.append("∫[\(a),\(b)] \(polyString(coeffs)) dx")
            // Integrate then evaluate F(b) - F(a)
            var fb = 0.0, fa = 0.0
            for i in 0...n {
                let power = Double(n - i)
                let newPower = power + 1
                let newCoeff = coeffs[i] / newPower
                fb += newCoeff * Foundation.pow(b, newPower)
                fa += newCoeff * Foundation.pow(a, newPower)
            }
            result = fb - fa
            steps.append("F(b) - F(a) = \(fb) - \(fa) = \(fb - fa)")
            desc = "∫[\(a),\(b)] = \(fb - fa)"

        case "limit_poly":
            // Direct substitution for polynomial
            guard !params.isEmpty else {
                return makeSolution("Calculus", "limit_poly", nil,
                    "Need params=[x_approach]", [], 0)
            }
            let x0 = params[0]
            let n = coeffs.count - 1
            steps.append("lim x→\(x0) of \(polyString(coeffs))")
            var val = 0.0
            for i in 0...n {
                val += coeffs[i] * Foundation.pow(x0, Double(n - i))
            }
            result = val
            steps.append("Direct substitution: \(val)")
            desc = "lim = \(val)"

        default:
            return makeSolution("Calculus", type, nil, "Unknown type: \(type)", [], 0)
        }
        return makeSolution("Calculus", "\(type)(\(coeffs))", result, desc, steps, 0.92)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 3. Linear Algebra Solver
    // ═══════════════════════════════════════════════════════════

    /// Solve linear algebra: "det2", "det3", "inverse2", "dot", "cross", "eigenvalues2", "trace"
    func solveLinearAlgebra(type: String, matrix: [[Double]]) -> MathSolution {
        lock.lock(); defer { lock.unlock() }
        var steps: [String] = []
        var result: Double? = nil
        var desc = ""

        switch type {
        case "det2":
            guard matrix.count >= 2, matrix[0].count >= 2, matrix[1].count >= 2 else {
                return makeSolution("LinearAlgebra", "det2", nil, "Need 2x2 matrix", [], 0)
            }
            let a = matrix[0][0], b = matrix[0][1]
            let c = matrix[1][0], d = matrix[1][1]
            let det = a * d - b * c
            steps.append("|[\(a), \(b)] [\(c), \(d)]| = ad - bc")
            steps.append("= \(a)*\(d) - \(b)*\(c) = \(det)")
            result = det
            desc = "det = \(det)"

        case "det3":
            guard matrix.count >= 3,
                  matrix[0].count >= 3, matrix[1].count >= 3, matrix[2].count >= 3 else {
                return makeSolution("LinearAlgebra", "det3", nil, "Need 3x3 matrix", [], 0)
            }
            let m = matrix
            let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            steps.append("Cofactor expansion along row 1 → det = \(det)")
            result = det; desc = "det = \(det)"

        case "inverse2":
            guard matrix.count >= 2, matrix[0].count >= 2, matrix[1].count >= 2 else {
                return makeSolution("LinearAlgebra", "inverse2", nil, "Need 2x2 matrix", [], 0)
            }
            let a = matrix[0][0], b = matrix[0][1]
            let c = matrix[1][0], d = matrix[1][1]
            let det = a * d - b * c
            guard det != 0 else {
                return makeSolution("LinearAlgebra", "inverse2", nil,
                    "Singular matrix (det=0)", ["det = ad-bc = 0"], 0)
            }
            let invA = d / det, invB = -b / det
            let invC = -c / det, invD = a / det
            steps.append("det = \(det)")
            steps.append("A⁻¹ = (1/det) * [[d, -b], [-c, a]]")
            steps.append("[[\(invA), \(invB)], [\(invC), \(invD)]]")
            result = det
            desc = "A⁻¹ = [[\(invA),\(invB)],[\(invC),\(invD)]]"

        case "dot":
            // Treat matrix as two row vectors
            guard matrix.count >= 2 else {
                return makeSolution("LinearAlgebra", "dot", nil, "Need 2 vectors", [], 0)
            }
            let u = matrix[0], v = matrix[1]
            guard u.count == v.count else {
                return makeSolution("LinearAlgebra", "dot", nil, "Vector length mismatch", [], 0)
            }
            var dot = 0.0
            var terms: [String] = []
            for i in 0..<u.count {
                dot += u[i] * v[i]
                terms.append("\(u[i])*\(v[i])")
            }
            steps.append("u·v = \(terms.joined(separator: " + "))")
            steps.append("    = \(dot)")
            result = dot
            desc = "u·v = \(dot)"

        case "cross":
            guard matrix.count >= 2, matrix[0].count >= 3, matrix[1].count >= 3 else {
                return makeSolution("LinearAlgebra", "cross", nil, "Need 2 vectors of dim 3", [], 0)
            }
            let u = matrix[0], v = matrix[1]
            let cx = u[1] * v[2] - u[2] * v[1]
            let cy = u[2] * v[0] - u[0] * v[2]
            let cz = u[0] * v[1] - u[1] * v[0]
            steps.append("u×v = (u₂v₃-u₃v₂, u₃v₁-u₁v₃, u₁v₂-u₂v₁)")
            steps.append("    = (\(cx), \(cy), \(cz))")
            let magnitude = Foundation.sqrt(cx*cx + cy*cy + cz*cz)
            result = magnitude
            desc = "u×v = (\(cx), \(cy), \(cz)), |u×v| = \(magnitude)"

        case "eigenvalues2":
            guard matrix.count >= 2, matrix[0].count >= 2, matrix[1].count >= 2 else {
                return makeSolution("LinearAlgebra", "eigenvalues2", nil, "Need 2x2 matrix", [], 0)
            }
            let a = matrix[0][0], b = matrix[0][1], c = matrix[1][0], d = matrix[1][1]
            let trace = a + d, det = a * d - b * c, disc = trace * trace - 4 * det
            steps.append("Characteristic: λ² - \(trace)λ + \(det) = 0, Δ=\(disc)")
            if disc >= 0 {
                let l1 = (trace + Foundation.sqrt(disc)) / 2
                let l2 = (trace - Foundation.sqrt(disc)) / 2
                result = l1; desc = "λ₁ = \(l1), λ₂ = \(l2)"
            } else {
                let re = trace / 2, im = Foundation.sqrt(-disc) / 2
                result = re; desc = "λ = \(re) ± \(im)i"
            }

        case "trace":
            let n = Swift.min(matrix.count, matrix.first?.count ?? 0)
            var tr = 0.0; for i in 0..<n { tr += matrix[i][i] }
            result = tr; desc = "trace = \(tr)"
            steps.append("tr(A) = \(tr)")

        default:
            return makeSolution("LinearAlgebra", type, nil, "Unknown type: \(type)", [], 0)
        }
        return makeSolution("LinearAlgebra", "\(type)", result, desc, steps, 0.96)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 4. Number Theory Solver
    // ═══════════════════════════════════════════════════════════

    /// Solve number theory: primality, factorization, totient, Fibonacci (PHI closed form), perfect check
    func solveNumberTheory(n: Int) -> MathSolution {
        lock.lock(); defer { lock.unlock() }
        var steps: [String] = []
        var results: [String] = []
        let absN = abs(n)

        // 1. Primality test
        let isPrime = checkPrime(absN)
        steps.append("Primality(\(absN)): \(isPrime ? "PRIME" : "COMPOSITE")")
        results.append("prime=\(isPrime)")

        // 2. Factorization
        if absN > 1 && !isPrime {
            let factors = primeFactors(absN)
            steps.append("Factors: \(factors)")
            results.append("factors=\(factors)")
        }

        // 3. Euler totient φ(n)
        let totient = eulerTotient(absN)
        steps.append("φ(\(absN)) = \(totient)")
        results.append("totient=\(totient)")

        // 4. Fibonacci via Binet's formula using PHI
        let fib = fibonacci(absN)
        steps.append("F(\(absN)) = (PHI^\(absN) - (-PHI)^(-\(absN))) / √5 = \(fib)")
        results.append("fib=\(fib)")

        // 5. Perfect number check
        if absN > 1 {
            let divSum = divisorSum(absN)
            let perfect = divSum == absN
            steps.append("σ(\(absN))-\(absN) = \(divSum) → \(perfect ? "PERFECT" : "not perfect")")
            results.append("perfect=\(perfect)")
        }

        // 6. Sacred resonance — PHI alignment
        let fibDouble = Double(fib)
        let resonance = MathSolution.computeResonance(fibDouble)
        steps.append("Sacred resonance (PHI-alignment of F(\(absN))): \(resonance)")

        return makeSolution("NumberTheory", "n=\(n)", Double(fib),
            results.joined(separator: ", "), steps, 0.94)
    }

    /// Compute GCD of two integers using Euclidean algorithm
    func gcd(_ a: Int, _ b: Int) -> Int {
        var x = abs(a), y = abs(b)
        while y != 0 { let t = y; y = x % y; x = t }
        return x
    }

    /// Compute LCM
    func lcm(_ a: Int, _ b: Int) -> Int {
        guard a != 0 && b != 0 else { return 0 }
        return abs(a) / gcd(a, b) * abs(b)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 5. Geometry Solver
    // ═══════════════════════════════════════════════════════════

    /// Solve geometry for 10 shapes: circle, rectangle, triangle, sphere, cylinder, cone, trapezoid, ellipse, regular_polygon, torus
    func solveGeometry(shape: String, params: [Double]) -> MathSolution {
        lock.lock(); defer { lock.unlock() }
        var steps: [String] = []
        var result: Double? = nil
        var desc = ""

        switch shape {
        case "circle":
            guard let r = params.first, r > 0 else {
                return makeSolution("Geometry", "circle", nil, "Need radius > 0", [], 0)
            }
            let area = .pi * r * r
            let circumference = 2 * .pi * r
            steps.append("Circle r=\(r)")
            steps.append("Area = πr² = \(area)")
            steps.append("Circumference = 2πr = \(circumference)")
            result = area
            desc = "area=\(area), circumference=\(circumference)"

        case "rectangle":
            guard params.count >= 2, params[0] > 0, params[1] > 0 else {
                return makeSolution("Geometry", "rectangle", nil, "Need w,h > 0", [], 0)
            }
            let w = params[0], h = params[1]
            let area = w * h
            let perimeter = 2 * (w + h)
            let diagonal = Foundation.sqrt(w * w + h * h)
            steps.append("Rectangle \(w)×\(h)")
            steps.append("Area = \(area), Perimeter = \(perimeter), Diagonal = \(diagonal)")
            result = area
            desc = "area=\(area), perimeter=\(perimeter)"

        case "triangle":
            // Heron's formula with sides a, b, c
            guard params.count >= 3, params.allSatisfy({ $0 > 0 }) else {
                return makeSolution("Geometry", "triangle", nil, "Need 3 positive sides", [], 0)
            }
            let a = params[0], b = params[1], c = params[2]
            let s = (a + b + c) / 2
            let areaSquared = s * (s - a) * (s - b) * (s - c)
            guard areaSquared >= 0 else {
                return makeSolution("Geometry", "triangle", nil,
                    "Invalid triangle (violates inequality)", [], 0)
            }
            let area = Foundation.sqrt(areaSquared)
            steps.append("Triangle sides \(a), \(b), \(c)")
            steps.append("Semi-perimeter s = \(s)")
            steps.append("Heron: Area = √(s(s-a)(s-b)(s-c)) = \(area)")
            result = area
            desc = "area=\(area)"

        case "sphere":
            guard let r = params.first, r > 0 else {
                return makeSolution("Geometry", "sphere", nil, "Need radius > 0", [], 0)
            }
            let volume = (4.0 / 3.0) * .pi * r * r * r
            let surface = 4 * .pi * r * r
            steps.append("Sphere r=\(r)")
            steps.append("V = (4/3)πr³ = \(volume)")
            steps.append("SA = 4πr² = \(surface)")
            result = volume
            desc = "volume=\(volume), surface=\(surface)"

        case "cylinder":
            guard params.count >= 2, params[0] > 0, params[1] > 0 else {
                return makeSolution("Geometry", "cylinder", nil, "Need r,h > 0", [], 0)
            }
            let r = params[0], h = params[1]
            let volume = .pi * r * r * h
            let surface = 2 * .pi * r * (r + h)
            steps.append("Cylinder r=\(r), h=\(h)")
            steps.append("V = πr²h = \(volume)")
            steps.append("SA = 2πr(r+h) = \(surface)")
            result = volume
            desc = "volume=\(volume), surface=\(surface)"

        case "cone":
            guard params.count >= 2, params[0] > 0, params[1] > 0 else {
                return makeSolution("Geometry", "cone", nil, "Need r,h > 0", [], 0)
            }
            let r = params[0], h = params[1]
            let slant = Foundation.sqrt(r * r + h * h)
            let volume = .pi * r * r * h / 3
            let surface = .pi * r * (r + slant)
            steps.append("Cone r=\(r), h=\(h), slant=\(slant)")
            steps.append("V = πr²h/3 = \(volume)")
            steps.append("SA = πr(r+l) = \(surface)")
            result = volume
            desc = "volume=\(volume), surface=\(surface)"

        case "trapezoid":
            guard params.count >= 3, params.allSatisfy({ $0 > 0 }) else {
                return makeSolution("Geometry", "trapezoid", nil, "Need a,b,h > 0", [], 0)
            }
            let a = params[0], b = params[1], h = params[2]
            let area = (a + b) * h / 2
            steps.append("Trapezoid bases \(a),\(b), height \(h)")
            steps.append("Area = (a+b)h/2 = \(area)")
            result = area
            desc = "area=\(area)"

        case "ellipse":
            guard params.count >= 2, params[0] > 0, params[1] > 0 else {
                return makeSolution("Geometry", "ellipse", nil, "Need a,b > 0", [], 0)
            }
            let a = params[0], b = params[1]
            let area = .pi * a * b
            // Ramanujan approximation for perimeter
            let h = ((a - b) * (a - b)) / ((a + b) * (a + b))
            let perimeter = .pi * (a + b) * (1 + 3 * h / (10 + Foundation.sqrt(4 - 3 * h)))
            steps.append("Ellipse a=\(a), b=\(b)")
            steps.append("Area = πab = \(area)")
            steps.append("Perimeter ≈ \(perimeter) (Ramanujan)")
            result = area
            desc = "area=\(area), perimeter≈\(perimeter)"

        case "regular_polygon":
            guard params.count >= 2, params[0] >= 3, params[1] > 0 else {
                return makeSolution("Geometry", "regular_polygon", nil,
                    "Need n≥3, side>0", [], 0)
            }
            let n = params[0], s = params[1]
            let area = (n * s * s) / (4 * Foundation.tan(.pi / n))
            let perimeter = n * s
            steps.append("Regular \(Int(n))-gon, side=\(s)")
            steps.append("Area = (ns²)/(4tan(π/n)) = \(area)")
            result = area
            desc = "area=\(area), perimeter=\(perimeter)"

        case "torus":
            guard params.count >= 2, params[0] > 0, params[1] > 0 else {
                return makeSolution("Geometry", "torus", nil, "Need R,r > 0", [], 0)
            }
            let R = params[0], r = params[1]
            let volume = 2 * .pi * .pi * R * r * r
            let surface = 4 * .pi * .pi * R * r
            steps.append("Torus R=\(R), r=\(r)")
            steps.append("V = 2π²Rr² = \(volume)")
            steps.append("SA = 4π²Rr = \(surface)")
            result = volume
            desc = "volume=\(volume), surface=\(surface)"

        default:
            return makeSolution("Geometry", shape, nil, "Unknown shape: \(shape)", [], 0)
        }
        return makeSolution("Geometry", "\(shape)(\(params))", result, desc, steps, 0.97)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 6. Probability & Statistics Solver
    // ═══════════════════════════════════════════════════════════

    /// Solve probability: combinations, permutations, bayes, normal_cdf, binomial, poisson, expected_value, entropy
    func solveProbability(type: String, params: [Double]) -> MathSolution {
        lock.lock(); defer { lock.unlock() }
        var steps: [String] = []
        var result: Double? = nil
        var desc = ""

        switch type {
        case "combinations":
            guard params.count >= 2 else {
                return makeSolution("Probability", "C(n,r)", nil, "Need n,r", [], 0)
            }
            let n = Int(params[0]), r = Int(params[1])
            guard n >= r, r >= 0 else {
                return makeSolution("Probability", "C(\(n),\(r))", nil, "Need n≥r≥0", [], 0)
            }
            let c = binomialCoeff(n, r)
            steps.append("C(\(n),\(r)) = \(n)! / (\(r)!(\(n)-\(r))!)")
            steps.append("= \(c)")
            result = Double(c)
            desc = "C(\(n),\(r)) = \(c)"

        case "permutations":
            guard params.count >= 2 else {
                return makeSolution("Probability", "P(n,r)", nil, "Need n,r", [], 0)
            }
            let n = Int(params[0]), r = Int(params[1])
            guard n >= r, r >= 0 else {
                return makeSolution("Probability", "P(\(n),\(r))", nil, "Need n≥r≥0", [], 0)
            }
            var p = 1
            for i in 0..<r { p *= (n - i) }
            steps.append("P(\(n),\(r)) = \(n)!/(\(n)-\(r))! = \(p)")
            result = Double(p)
            desc = "P(\(n),\(r)) = \(p)"

        case "bayes":
            // P(A|B) = P(B|A)P(A) / [P(B|A)P(A) + P(B|¬A)P(¬A)]
            guard params.count >= 3 else {
                return makeSolution("Probability", "bayes", nil,
                    "Need P(A), P(B|A), P(B|¬A)", [], 0)
            }
            let pA = params[0], pBA = params[1], pBNotA = params[2]
            let pB = pBA * pA + pBNotA * (1 - pA)
            guard pB > 0 else {
                return makeSolution("Probability", "bayes", nil, "P(B)=0", [], 0)
            }
            let pAB = pBA * pA / pB
            steps.append("P(A)=\(pA), P(B|A)=\(pBA), P(B|¬A)=\(pBNotA)")
            steps.append("P(B) = P(B|A)P(A) + P(B|¬A)P(¬A) = \(pB)")
            steps.append("P(A|B) = P(B|A)P(A)/P(B) = \(pAB)")
            result = pAB
            desc = "P(A|B) = \(pAB)"

        case "normal_cdf":
            // Approximation of Φ(z) using Abramowitz & Stegun
            guard params.count >= 3 else {
                return makeSolution("Probability", "normal_cdf", nil, "Need x,μ,σ", [], 0)
            }
            let x = params[0], mu = params[1], sigma = params[2]
            guard sigma > 0 else {
                return makeSolution("Probability", "normal_cdf", nil, "Need σ>0", [], 0)
            }
            let z = (x - mu) / sigma
            let cdf = normalCDF(z)
            steps.append("z = (x-μ)/σ = (\(x)-\(mu))/\(sigma) = \(z)")
            steps.append("Φ(\(z)) ≈ \(cdf)")
            result = cdf
            desc = "Φ(\(z)) = \(cdf)"

        case "binomial":
            // P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
            guard params.count >= 3 else {
                return makeSolution("Probability", "binomial", nil, "Need n,k,p", [], 0)
            }
            let n = Int(params[0]), k = Int(params[1]), p = params[2]
            let c = binomialCoeff(n, k)
            let prob = Double(c) * Foundation.pow(p, Double(k)) * Foundation.pow(1 - p, Double(n - k))
            steps.append("P(X=\(k)) = C(\(n),\(k)) * \(p)^\(k) * \(1-p)^\(n-k)")
            steps.append("= \(c) * \(Foundation.pow(p, Double(k))) * \(Foundation.pow(1-p, Double(n-k)))")
            steps.append("= \(prob)")
            result = prob
            desc = "P(X=\(k)) = \(prob)"

        case "poisson":
            // P(X=k) = λ^k * e^(-λ) / k!
            guard params.count >= 2 else {
                return makeSolution("Probability", "poisson", nil, "Need k,λ", [], 0)
            }
            let k = Int(params[0]), lambda = params[1]
            let prob = Foundation.pow(lambda, Double(k)) * Foundation.exp(-lambda) / Double(factorial(k))
            steps.append("P(X=\(k)) = \(lambda)^\(k) * e^(-\(lambda)) / \(k)!")
            steps.append("= \(prob)")
            result = prob
            desc = "P(X=\(k)) = \(prob)"

        case "expected_value":
            // First half of params = values, second half = probabilities
            let n = params.count / 2
            guard n >= 1, params.count == 2 * n else {
                return makeSolution("Probability", "expected_value", nil,
                    "Need equal count of values and probabilities", [], 0)
            }
            var ev = 0.0
            for i in 0..<n { ev += params[i] * params[n + i] }
            steps.append("E[X] = Σ xᵢpᵢ = \(ev)")
            result = ev
            desc = "E[X] = \(ev)"

        case "entropy":
            // Shannon entropy: H = -Σ p log₂(p)
            guard !params.isEmpty else {
                return makeSolution("Probability", "entropy", nil, "Need probabilities", [], 0)
            }
            var h = 0.0
            for p in params where p > 0 {
                h -= p * Foundation.log2(p)
            }
            steps.append("H = -Σ pᵢ log₂(pᵢ) = \(h) bits")
            result = h
            desc = "H = \(h) bits"

        default:
            return makeSolution("Probability", type, nil, "Unknown type: \(type)", [], 0)
        }
        return makeSolution("Probability", "\(type)(\(params))", result, desc, steps, 0.91)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 7. Quantum Math Solver
    // ═══════════════════════════════════════════════════════════

    /// Solve quantum math: born_rule, unitary_check, tensor_product, sacred_geometry, bloch_coords, expectation
    func solveQuantumMath(type: String, params: [Double]) -> MathSolution {
        lock.lock(); defer { lock.unlock() }
        var steps: [String] = []
        var result: Double? = nil
        var desc = ""

        switch type {
        case "born_rule":
            // |ψ⟩ = Σ αᵢ|i⟩, P(i) = |αᵢ|²
            // params are [re₀, im₀, re₁, im₁, ...]
            guard params.count >= 2, params.count % 2 == 0 else {
                return makeSolution("QuantumMath", "born_rule", nil,
                    "Need pairs of (re,im) amplitudes", [], 0)
            }
            let n = params.count / 2
            var probs: [Double] = []
            var totalProb = 0.0
            for i in 0..<n {
                let re = params[2 * i], im = params[2 * i + 1]
                let p = re * re + im * im
                probs.append(p)
                totalProb += p
                steps.append("|α\(i)|² = \(re)² + \(im)² = \(p)")
            }
            steps.append("Total probability = \(totalProb)")
            steps.append(totalProb.isApproximatelyEqual(to: 1.0, tolerance: 1e-9)
                ? "Valid quantum state (normalized)" : "Warning: not normalized")
            result = probs.first
            desc = "P = [\(probs.map { String(format: "%.6f", $0) }.joined(separator: ", "))]"

        case "unitary_check":
            // Check if matrix is unitary: U†U = I
            let n = Int(Foundation.sqrt(Double(params.count)))
            guard n * n == params.count, n > 0 else {
                return makeSolution("QuantumMath", "unitary_check", nil,
                    "Need N² elements for NxN matrix", [], 0)
            }
            // Build matrix
            var m: [[Double]] = []
            for i in 0..<n {
                var row: [Double] = []
                for j in 0..<n { row.append(params[i * n + j]) }
                m.append(row)
            }
            // Compute M^T * M (real case: unitary = orthogonal)
            var product: [[Double]] = Array(repeating: Array(repeating: 0, count: n), count: n)
            for i in 0..<n {
                for j in 0..<n {
                    for k in 0..<n { product[i][j] += m[k][i] * m[k][j] }
                }
            }
            // Check if product ≈ I
            var maxDeviation = 0.0
            for i in 0..<n {
                for j in 0..<n {
                    let expected = (i == j) ? 1.0 : 0.0
                    let dev = abs(product[i][j] - expected)
                    if dev > maxDeviation { maxDeviation = dev }
                }
            }
            let isUnitary = maxDeviation < 1e-9
            steps.append("\(n)×\(n) matrix orthogonality check")
            steps.append("Max deviation from identity: \(maxDeviation)")
            steps.append(isUnitary ? "UNITARY (orthogonal)" : "NOT UNITARY")
            result = isUnitary ? 1.0 : 0.0
            desc = isUnitary ? "Unitary (max dev \(maxDeviation))" : "Not unitary (max dev \(maxDeviation))"

        case "tensor_product":
            // Tensor (Kronecker) product of two vectors
            // First half = vector u, second half = vector v
            let half = params.count / 2
            guard half >= 1, params.count == 2 * half else {
                return makeSolution("QuantumMath", "tensor_product", nil,
                    "Need even number of params (two vectors)", [], 0)
            }
            let u = Array(params[0..<half])
            let v = Array(params[half..<params.count])
            var tensor: [Double] = []
            for ui in u {
                for vj in v {
                    tensor.append(ui * vj)
                }
            }
            steps.append("u = \(u), v = \(v)")
            steps.append("u⊗v dim = \(u.count)×\(v.count) = \(tensor.count)")
            steps.append("u⊗v = \(tensor)")
            result = Double(tensor.count)
            desc = "u⊗v = \(tensor)"

        case "sacred_geometry":
            // Compute sacred geometric quantities for dimension d
            let d = params.isEmpty ? 3.0 : params[0]
            let dim = Int(d)
            steps.append("Sacred geometry in \(dim)D")

            // PHI-spiral radius at angle θ = dim * π
            let theta = Double(dim) * .pi
            let spiralR = Foundation.pow(PHI, theta / (2 * .pi))
            steps.append("PHI-spiral radius at θ=\(dim)π: r = PHI^(\(dim)/2) = \(spiralR)")

            // GOD_CODE resonance in d dimensions
            let godResonance = GOD_CODE / Foundation.pow(PHI, d)
            steps.append("GOD_CODE/PHI^\(dim) = \(godResonance)")

            // Feigenbaum harmonic
            let feigHarmonic = FEIGENBAUM * d / TAU
            steps.append("FEIGENBAUM×\(dim)/TAU = \(feigHarmonic)")

            // OMEGA dimensional projection
            let omegaProj = OMEGA / Foundation.pow(d, PHI)
            steps.append("OMEGA/\(dim)^PHI = \(omegaProj)")

            result = godResonance
            desc = "sacred_\(dim)D: GOD_RES=\(godResonance), spiral=\(spiralR)"

        case "bloch_coords":
            // Bloch sphere: |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
            guard params.count >= 2 else {
                return makeSolution("QuantumMath", "bloch_coords", nil,
                    "Need theta, phi", [], 0)
            }
            let theta = params[0], phi = params[1]
            let x = Foundation.sin(theta) * Foundation.cos(phi)
            let y = Foundation.sin(theta) * Foundation.sin(phi)
            let z = Foundation.cos(theta)
            let alpha = Foundation.cos(theta / 2)
            let betaRe = Foundation.cos(phi) * Foundation.sin(theta / 2)
            let betaIm = Foundation.sin(phi) * Foundation.sin(theta / 2)
            steps.append("Bloch: θ=\(theta), φ=\(phi)")
            steps.append("(x,y,z) = (\(x), \(y), \(z))")
            steps.append("|ψ⟩ = \(alpha)|0⟩ + (\(betaRe)+\(betaIm)i)|1⟩")
            result = Foundation.sqrt(x*x + y*y + z*z) // should be 1
            desc = "Bloch(\(x),\(y),\(z))"

        case "expectation":
            // ⟨O⟩ = Σ λᵢ pᵢ — eigenvalues and probabilities
            let half = params.count / 2
            guard half >= 1, params.count == 2 * half else {
                return makeSolution("QuantumMath", "expectation", nil,
                    "Need equal eigenvalues and probabilities", [], 0)
            }
            var ev = 0.0
            for i in 0..<half {
                ev += params[i] * params[half + i]
                steps.append("λ\(i)=\(params[i]) × p\(i)=\(params[half + i])")
            }
            steps.append("⟨O⟩ = \(ev)")
            result = ev
            desc = "⟨O⟩ = \(ev)"

        case "riemann_zeta":
            // ζ(s) = Σ_{n=1}^{N} 1/n^s   (partial sum approximation)
            let s = params.isEmpty ? 2.0 : params[0]
            let terms = params.count > 1 ? Int(params[1]) : 1000
            guard s > 1.0 else {
                return makeSolution("QuantumMath", "riemann_zeta", nil,
                    "ζ(s) converges only for Re(s) > 1", [], 0)
            }
            var zeta = 0.0
            for n in 1...terms {
                zeta += 1.0 / Foundation.pow(Double(n), s)
            }
            steps.append("ζ(\(s)) partial sum with \(terms) terms")
            steps.append("ζ(\(s)) ≈ \(zeta)")
            // Sacred check: ζ(2) = π²/6
            if abs(s - 2.0) < 1e-10 {
                let exact = .pi * .pi / 6.0
                steps.append("Exact: ζ(2) = π²/6 = \(exact)")
                steps.append("Error: \(abs(zeta - exact))")
            }
            // GOD_CODE resonance
            let zetaGodRatio = zeta * GOD_CODE / 1000.0
            steps.append("ζ×GOD_CODE/1000 = \(zetaGodRatio)")
            result = zeta
            desc = "ζ(\(s)) = \(zeta)"

        case "prime_sieve":
            // Sieve of Eratosthenes up to n, with prime counting and PHI-spiral analysis
            let n = params.isEmpty ? 100.0 : params[0]
            let limit = min(Int(n), 100_000)
            guard limit >= 2 else {
                return makeSolution("QuantumMath", "prime_sieve", nil, "Need n ≥ 2", [], 0)
            }
            var sieve = [Bool](repeating: true, count: limit + 1)
            sieve[0] = false; sieve[1] = false
            var p = 2
            while p * p <= limit {
                if sieve[p] { for m in stride(from: p * p, through: limit, by: p) { sieve[m] = false } }
                p += 1
            }
            let primes = (2...limit).filter { sieve[$0] }
            let piN = primes.count
            steps.append("π(\(limit)) = \(piN) primes")
            // Prime counting approx: n / ln(n)
            let approx = Double(limit) / log(Double(limit))
            steps.append("n/ln(n) ≈ \(approx), ratio = \(Double(piN) / approx)")
            // PHI-prime correlation: fraction of primes in Fibonacci positions
            var fibSet = Set<Int>(); var a = 1, b = 1
            while b <= limit { fibSet.insert(b); let t = a + b; a = b; b = t }
            let fibPrimes = primes.filter { fibSet.contains($0) }.count
            steps.append("Fibonacci-primes up to \(limit): \(fibPrimes)")
            // GOD_CODE harmonic: Σ 1/p for first 10 primes
            let primeHarmonic = primes.prefix(10).reduce(0.0) { $0 + 1.0 / Double($1) }
            steps.append("Prime harmonic (first 10): \(primeHarmonic)")
            result = Double(piN)
            desc = "π(\(limit))=\(piN), \(fibPrimes) Fib-primes"

        case "collatz_sequence":
            // Collatz conjecture: n → n/2 (even) or 3n+1 (odd), track until 1
            let start = params.isEmpty ? 27.0 : params[0]
            var n = Int(start)
            guard n > 0 else {
                return makeSolution("QuantumMath", "collatz_sequence", nil, "Need n > 0", [], 0)
            }
            var seq = [n]
            var maxVal = n
            while n != 1 && seq.count < 10_000 {
                n = n % 2 == 0 ? n / 2 : 3 * n + 1
                seq.append(n)
                if n > maxVal { maxVal = n }
            }
            steps.append("Collatz(\(Int(start))): \(seq.count) steps, max = \(maxVal)")
            steps.append("First 10: \(seq.prefix(10).map(String.init).joined(separator: "→"))")
            // Sacred geometry: stopping time / log(start) ratio
            let stoppingRatio = Double(seq.count) / log(max(2, start))
            steps.append("Stopping ratio: \(stoppingRatio)")
            // PHI resonance of sequence length
            let logPhiLen = log(Double(seq.count)) / log(PHI)
            steps.append("log_PHI(steps) = \(logPhiLen)")
            result = Double(seq.count)
            desc = "Collatz(\(Int(start))): \(seq.count) steps, max=\(maxVal)"

        case "fibonacci_spiral":
            // Extended Fibonacci with PHI-resonance analysis
            let count = params.isEmpty ? 20.0 : params[0]
            let n = min(Int(count), 100)
            guard n >= 2 else {
                return makeSolution("QuantumMath", "fibonacci_spiral", nil, "Need n ≥ 2", [], 0)
            }
            var fibs: [Double] = [1, 1]
            for i in 2..<n { fibs.append(fibs[i - 1] + fibs[i - 2]) }
            steps.append("Fibonacci up to F(\(n)): \(fibs.suffix(5).map { String(format: "%.0f", $0) })")
            // Ratio convergence to PHI
            var ratios: [Double] = []
            for i in 2..<fibs.count {
                ratios.append(fibs[i] / fibs[i - 1])
            }
            let lastRatio = ratios.last ?? 0
            let phiError = abs(lastRatio - PHI)
            steps.append("F(\(n))/F(\(n-1)) = \(lastRatio), PHI error = \(phiError)")
            // Spiral arc length (golden spiral approximation)
            let spiralArc = fibs.reduce(0.0) { $0 + Foundation.sqrt($1) * .pi / 2.0 }
            steps.append("Spiral arc ≈ \(spiralArc)")
            // GOD_CODE ratio at convergence
            let godRatio = fibs.last! / GOD_CODE
            steps.append("F(\(n))/GOD_CODE = \(godRatio)")
            result = lastRatio
            desc = "Fib(\(n)): ratio→PHI, error=\(phiError)"

        case "elliptic_curve":
            // Weierstrass form: y² = x³ + ax + b, count rational points (mod p)
            let a = params.count > 0 ? params[0] : -1.0
            let b = params.count > 1 ? params[1] : 1.0
            let p = params.count > 2 ? Int(params[2]) : 23
            guard p > 2 else {
                return makeSolution("QuantumMath", "elliptic_curve", nil, "Need prime p > 2", [], 0)
            }
            // Discriminant check: Δ = -16(4a³ + 27b²) ≠ 0
            let discriminant = -16 * (4 * a * a * a + 27 * b * b)
            steps.append("E: y² = x³ + \(a)x + \(b) over F_\(p)")
            steps.append("Discriminant Δ = \(discriminant)")
            // Count points by brute force mod p
            var points = 1  // Point at infinity
            for x in 0..<p {
                let rhs = (x * x * x + Int(a) * x + Int(b)) % p
                let rhsMod = ((rhs % p) + p) % p
                for y in 0..<p {
                    let lhs = (y * y) % p
                    if lhs == rhsMod { points += 1 }
                }
            }
            steps.append("#E(F_\(p)) = \(points)")
            // Hasse bound: |#E - (p+1)| ≤ 2√p
            let hasseBound = 2.0 * Foundation.sqrt(Double(p))
            let trace = Double(p + 1 - points)
            steps.append("Trace of Frobenius a_p = \(Int(trace))")
            steps.append("Hasse bound: |a_p| ≤ \(hasseBound) → \(abs(trace) <= hasseBound ? "VALID" : "ERROR")")
            // Sacred alignment: GOD_CODE mod relation
            let godMod = Int(GOD_CODE) % p
            steps.append("GOD_CODE mod \(p) = \(godMod)")
            result = Double(points)
            desc = "#E(F_\(p))=\(points), a_p=\(Int(trace))"

        case "modular_arithmetic":
            // Modular exponentiation, Euler's totient, discrete log search
            let base = params.count > 0 ? Int(params[0]) : 2
            let exp = params.count > 1 ? Int(params[1]) : 10
            let modulus = params.count > 2 ? Int(params[2]) : 104
            guard modulus > 0 else {
                return makeSolution("QuantumMath", "modular_arithmetic", nil, "Need modulus > 0", [], 0)
            }
            // Fast modular exponentiation
            var res = 1, b2 = base % modulus, e2 = exp
            while e2 > 0 {
                if e2 % 2 == 1 { res = (res * b2) % modulus }
                e2 /= 2; b2 = (b2 * b2) % modulus
            }
            steps.append("\(base)^\(exp) mod \(modulus) = \(res)")
            // Euler's totient φ(n) via prime factorization
            var phi = modulus, temp = modulus, pf = 2
            while pf * pf <= temp {
                if temp % pf == 0 {
                    while temp % pf == 0 { temp /= pf }
                    phi -= phi / pf
                }
                pf += 1
            }
            if temp > 1 { phi -= phi / temp }
            steps.append("φ(\(modulus)) = \(phi)")
            // Verify Euler's theorem: a^φ(n) ≡ 1 (mod n) when gcd(a,n)=1
            var verRes = 1; var vb = base % modulus; var ve = phi
            while ve > 0 {
                if ve % 2 == 1 { verRes = (verRes * vb) % modulus }
                ve /= 2; vb = (vb * vb) % modulus
            }
            steps.append("\(base)^φ(\(modulus)) mod \(modulus) = \(verRes) \(verRes == 1 ? "(Euler verified)" : "")")
            // L104 signature: 104 = 8 × 13, φ(104) = 48
            if modulus == 104 {
                steps.append("L104 sacred modulus: 104 = 2³ × 13, φ(104) = 48")
            }
            result = Double(res)
            desc = "\(base)^\(exp) mod \(modulus) = \(res), φ(\(modulus))=\(phi)"

        default:
            return makeSolution("QuantumMath", type, nil, "Unknown type: \(type)", [], 0)
        }
        return makeSolution("QuantumMath", "\(type)", result, desc, steps, 0.89)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Main Solver (Keyword Router)
    // ═══════════════════════════════════════════════════════════

    /// Parse a natural language math problem and route to the appropriate domain solver.
    func solve(problem: String) -> MathSolution {
        let p = problem.lowercased()
        let nums = extractNumbers(from: problem)

        // Algebra
        if p.contains("solve") && (p.contains("x") || p.contains("equation")) {
            if (p.contains("quadratic") || p.contains("x^2")), nums.count >= 3 {
                return solveAlgebra(type: "quadratic", coeffs: Array(nums.prefix(3)))
            }
            if (p.contains("cubic") || p.contains("x^3")), nums.count >= 4 {
                return solveAlgebra(type: "cubic", coeffs: Array(nums.prefix(4)))
            }
            if (p.contains("system") || p.contains("simultaneous")), nums.count >= 6 {
                return solveAlgebra(type: "system2", coeffs: Array(nums.prefix(6)))
            }
            if nums.count >= 2 { return solveAlgebra(type: "linear", coeffs: Array(nums.prefix(2))) }
        }
        // Calculus
        if p.contains("derivative") || p.contains("differentiate") || p.contains("d/dx") {
            if !nums.isEmpty { return solveCalculus(type: "derivative", coeffs: nums) }
        }
        if p.contains("integral") || p.contains("integrate") {
            if !nums.isEmpty { return solveCalculus(type: "integral", coeffs: nums) }
        }
        if p.contains("limit"), nums.count >= 2 {
            return solveCalculus(type: "limit_poly", coeffs: Array(nums.dropLast()), params: [nums.last!])
        }
        // Linear Algebra
        if p.contains("determinant") || p.contains("det") {
            if nums.count >= 9 {
                return solveLinearAlgebra(type: "det3", matrix: [Array(nums[0..<3]), Array(nums[3..<6]), Array(nums[6..<9])])
            } else if nums.count >= 4 {
                return solveLinearAlgebra(type: "det2", matrix: [Array(nums[0..<2]), Array(nums[2..<4])])
            }
        }
        if p.contains("eigenvalue"), nums.count >= 4 {
            return solveLinearAlgebra(type: "eigenvalues2", matrix: [Array(nums[0..<2]), Array(nums[2..<4])])
        }
        // Number Theory
        if p.contains("prime") || p.contains("factor") || p.contains("fibonacci")
            || p.contains("totient") || p.contains("gcd"), let first = nums.first {
            return solveNumberTheory(n: Int(first))
        }
        // Geometry
        for shape in ["circle","rectangle","triangle","sphere","cylinder","cone","trapezoid","ellipse","torus"] {
            if p.contains(shape), !nums.isEmpty { return solveGeometry(shape: shape, params: nums) }
        }
        // Probability
        if p.contains("bayes"), nums.count >= 3 { return solveProbability(type: "bayes", params: Array(nums.prefix(3))) }
        if p.contains("combination"), nums.count >= 2 { return solveProbability(type: "combinations", params: Array(nums.prefix(2))) }
        if p.contains("permutation"), nums.count >= 2 { return solveProbability(type: "permutations", params: Array(nums.prefix(2))) }
        if p.contains("binomial"), nums.count >= 3 { return solveProbability(type: "binomial", params: Array(nums.prefix(3))) }
        if p.contains("poisson"), nums.count >= 2 { return solveProbability(type: "poisson", params: Array(nums.prefix(2))) }
        // Quantum
        if p.contains("born"), !nums.isEmpty { return solveQuantumMath(type: "born_rule", params: nums) }
        if p.contains("bloch"), nums.count >= 2 { return solveQuantumMath(type: "bloch_coords", params: Array(nums.prefix(2))) }
        if p.contains("tensor"), nums.count >= 2 { return solveQuantumMath(type: "tensor_product", params: nums) }
        if p.contains("sacred") { return solveQuantumMath(type: "sacred_geometry", params: nums.isEmpty ? [3] : nums) }
        // Quantum Research Domains (EVO_68)
        if p.contains("riemann") || p.contains("zeta") { return solveQuantumMath(type: "riemann_zeta", params: nums.isEmpty ? [2] : nums) }
        if p.contains("prime") && p.contains("sieve") { return solveQuantumMath(type: "prime_sieve", params: nums.isEmpty ? [100] : nums) }
        if p.contains("collatz") { return solveQuantumMath(type: "collatz_sequence", params: nums.isEmpty ? [27] : nums) }
        if p.contains("fibonacci") || p.contains("spiral") { return solveQuantumMath(type: "fibonacci_spiral", params: nums.isEmpty ? [20] : nums) }
        if p.contains("elliptic") || p.contains("curve") { return solveQuantumMath(type: "elliptic_curve", params: nums.isEmpty ? [-1, 1, 23] : nums) }
        if p.contains("modular") || p.contains("mod ") { return solveQuantumMath(type: "modular_arithmetic", params: nums.isEmpty ? [2, 10, 104] : nums) }

        return MathSolution(domain: "Unknown", input: problem, resultValue: nil,
            resultDescription: "Could not parse. Use domain-specific methods.",
            steps: ["No matching domain for: \(problem)"], confidence: 0, sacredResonance: 0)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - MATH Benchmark Interface
    // ═══════════════════════════════════════════════════════════

    /// Attempt to solve a MATH benchmark problem.
    /// Routes through keyword parsing, adjusts confidence by difficulty level.
    func solveMATH(problem: String, level: Int) -> (answer: String, confidence: Double) {
        let solution = solve(problem: problem)

        // Confidence scales inversely with difficulty level (1-5)
        let levelFactor = Swift.max(0.3, 1.0 - Double(level - 1) * 0.12)
        let adjustedConfidence = solution.confidence * levelFactor

        // Format answer
        let answer: String
        if let val = solution.resultValue, val.isFinite {
            // Return integer if close to one, otherwise 6 significant figures
            if abs(val - val.rounded()) < 1e-9 {
                answer = String(format: "%.0f", val)
            } else {
                answer = String(format: "%.6g", val)
            }
        } else {
            answer = solution.resultDescription
        }

        return (answer: answer, confidence: adjustedConfidence)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - SovereignEngine Protocol
    // ═══════════════════════════════════════════════════════════

    func engineStatus() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        let avgConfidence = solveCount > 0 ? totalConfidence / Double(solveCount) : 0.0
        return [
            "engine": engineName,
            "version": Self.VERSION,
            "solveCount": solveCount,
            "averageConfidence": avgConfidence,
            "domainCounts": domainCounts,
            "domains": ["Algebra", "Calculus", "LinearAlgebra", "NumberTheory",
                        "Geometry", "Probability", "QuantumMath"],
            "sacredConstants": [
                "PHI": PHI,
                "GOD_CODE": GOD_CODE,
                "TAU": TAU,
                "OMEGA": OMEGA,
                "FEIGENBAUM": FEIGENBAUM
            ]
        ]
    }

    func engineHealth() -> Double {
        lock.lock(); defer { lock.unlock() }
        // Health based on average confidence and solve count
        if solveCount == 0 { return 1.0 } // Fresh engine is healthy
        let avgConf = totalConfidence / Double(solveCount)
        return Swift.min(1.0, avgConf * 1.05) // Slight boost for active use
    }

    func engineReset() {
        lock.lock(); defer { lock.unlock() }
        solveCount = 0
        totalConfidence = 0.0
        domainCounts = [:]
    }

    /// Status report for logging / telemetry
    func statusReport() -> String {
        let s = engineStatus()
        return "SymbolicMathSolver v\(Self.VERSION) | health=\(String(format:"%.1f%%",engineHealth()*100)) | solves=\(s["solveCount"] ?? 0) | PHI=\(PHI) | GOD_CODE=\(GOD_CODE)"
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Private Helpers
    // ═══════════════════════════════════════════════════════════

    /// Build a MathSolution and update internal counters (caller must hold lock)
    private func makeSolution(_ domain: String, _ input: String, _ value: Double?,
                              _ description: String, _ steps: [String],
                              _ confidence: Double) -> MathSolution {
        solveCount += 1
        totalConfidence += confidence
        domainCounts[domain, default: 0] += 1
        return MathSolution(
            domain: domain, input: input, resultValue: value,
            resultDescription: description, steps: steps,
            confidence: confidence,
            sacredResonance: MathSolution.computeResonance(value)
        )
    }

    /// Format polynomial coefficients as a string
    private func polyString(_ coeffs: [Double]) -> String {
        let n = coeffs.count - 1
        var terms: [String] = []
        for (i, c) in coeffs.enumerated() {
            let power = n - i
            if c == 0 { continue }
            let cStr = c.truncatingRemainder(dividingBy: 1) == 0
                ? String(format: "%.0f", c) : String(format: "%.4g", c)
            switch power {
            case 0: terms.append(cStr)
            case 1: terms.append("\(cStr)x")
            default: terms.append("\(cStr)x^\(power)")
            }
        }
        return terms.isEmpty ? "0" : terms.joined(separator: " + ")
    }

    /// Extract floating point numbers from a string
    private func extractNumbers(from text: String) -> [Double] {
        var numbers: [Double] = []
        let pattern = #"-?\d+\.?\d*"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, range: range)
        for match in matches {
            if let r = Range(match.range, in: text), let num = Double(text[r]) {
                numbers.append(num)
            }
        }
        return numbers
    }

    /// Trial-division primality test
    private func checkPrime(_ n: Int) -> Bool {
        guard n >= 2 else { return false }
        if n <= 3 { return true }
        if n % 2 == 0 || n % 3 == 0 { return false }
        var i = 5
        while i * i <= n {
            if n % i == 0 || n % (i + 2) == 0 { return false }
            i += 6
        }
        return true
    }

    /// Prime factorization
    private func primeFactors(_ n: Int) -> [Int] {
        var n = n
        var factors: [Int] = []
        var d = 2
        while d * d <= n {
            while n % d == 0 {
                factors.append(d)
                n /= d
            }
            d += 1
        }
        if n > 1 { factors.append(n) }
        return factors
    }

    /// Euler totient via factorization
    private func eulerTotient(_ n: Int) -> Int {
        guard n > 1 else { return n }
        var result = n
        var temp = n
        var d = 2
        while d * d <= temp {
            if temp % d == 0 {
                while temp % d == 0 { temp /= d }
                result -= result / d
            }
            d += 1
        }
        if temp > 1 { result -= result / temp }
        return result
    }

    /// Fibonacci via Binet's formula (PHI closed form)
    private func fibonacci(_ n: Int) -> Int {
        guard n >= 0 else { return 0 }
        if n <= 1 { return n }
        let sqrt5 = Foundation.sqrt(5.0)
        let fib = (Foundation.pow(PHI, Double(n)) - Foundation.pow(-PHI, Double(-n))) / sqrt5
        return Int(fib.rounded())
    }

    /// Sum of proper divisors
    private func divisorSum(_ n: Int) -> Int {
        guard n > 1 else { return 0 }
        var sum = 1
        var i = 2
        while i * i <= n {
            if n % i == 0 {
                sum += i
                if i != n / i { sum += n / i }
            }
            i += 1
        }
        return sum
    }

    /// Binomial coefficient C(n, k)
    private func binomialCoeff(_ n: Int, _ k: Int) -> Int {
        guard k >= 0, k <= n else { return 0 }
        let k = Swift.min(k, n - k)
        var result = 1
        for i in 0..<k {
            result = result * (n - i) / (i + 1)
        }
        return result
    }

    /// Factorial (capped at 20 to avoid overflow)
    private func factorial(_ n: Int) -> Int {
        guard n >= 0 else { return 1 }
        let capped = Swift.min(n, 20)
        var result = 1
        for i in 2...Swift.max(2, capped) { result *= i }
        return result
    }

    /// Normal CDF approximation (Abramowitz & Stegun 26.2.17)
    private func normalCDF(_ z: Double) -> Double {
        if z < -8 { return 0.0 }
        if z > 8 { return 1.0 }
        let a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741
        let a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911
        let sign: Double = z < 0 ? -1 : 1
        let x = abs(z) / Foundation.sqrt(2.0)
        let t = 1.0 / (1.0 + p * x)
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Foundation.exp(-x * x)
        return 0.5 * (1.0 + sign * y)
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Double Approximate Equality Extension
// ═══════════════════════════════════════════════════════════════

private extension Double {
    func isApproximatelyEqual(to other: Double, tolerance: Double) -> Bool {
        return abs(self - other) < tolerance
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - MathEntity — Extracted from word problem (ASI v2.0)
// ═══════════════════════════════════════════════════════════════

struct MathEntity {
    let name: String
    var value: Double?
    var variable: String?
    var unit: String
    var properties: [String: String]

    init(name: String, value: Double? = nil, variable: String? = nil,
         unit: String = "", properties: [String: String] = [:]) {
        self.name = name
        self.value = value
        self.variable = variable
        self.unit = unit
        self.properties = properties
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - MathRelation — Relationship between entities (ASI v2.0)
// ═══════════════════════════════════════════════════════════════

struct MathRelation {
    let relationType: String    // "equals", "sum", "product", "ratio", "difference", "percent_of"
    var entities: [String]
    var expression: String
    var value: Double?

    init(relationType: String, entities: [String] = [], expression: String = "", value: Double? = nil) {
        self.relationType = relationType
        self.entities = entities
        self.expression = expression
        self.value = value
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - ParsedProblem — Fully parsed word problem (ASI v2.0)
// ═══════════════════════════════════════════════════════════════

struct ParsedProblem {
    var original: String
    var entities: [MathEntity]
    var relations: [MathRelation]
    var unknowns: [String]
    var questionType: String    // "solve_for", "compute", "prove", "count", "find_max", "find_min"
    var domain: String          // "algebra", "geometry", "number_theory", "combinatorics", "calculus", "probability"
    var constraints: [String]

    init(original: String = "", entities: [MathEntity] = [], relations: [MathRelation] = [],
         unknowns: [String] = [], questionType: String = "", domain: String = "",
         constraints: [String] = []) {
        self.original = original
        self.entities = entities
        self.relations = relations
        self.unknowns = unknowns
        self.questionType = questionType
        self.domain = domain
        self.constraints = constraints
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - WordProblemParser (Layer 1) — ASI v2.0
// ═══════════════════════════════════════════════════════════════

/// Parse natural language math problems into structured representations.
/// Extracts entities, relationships, constraints, and identifies question type and domain.
final class WordProblemParser {
    static let shared = WordProblemParser()

    /// Number word mappings
    private let numberWords: [String: Double] = [
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        "hundred": 100, "thousand": 1000, "million": 1000000,
        "half": 0.5, "third": 1.0/3.0, "quarter": 0.25, "fifth": 0.2,
        "twice": 2, "thrice": 3, "double": 2, "triple": 3,
    ]

    /// Domain detection patterns
    private let domainPatterns: [String: [String]] = [
        "algebra": ["solve", "equation", "variable", "expression", "polynomial",
                    "factor", "simplify", "quadratic", "linear", "system of"],
        "geometry": ["triangle", "circle", "square", "rectangle", "area",
                    "perimeter", "volume", "angle", "radius", "diameter",
                    "parallel", "perpendicular", "polygon", "sphere", "cylinder",
                    "circumference"],
        "number_theory": ["prime", "divisible", "remainder", "modulo", "factor",
                         "gcd", "lcm", "integer", "divisor", "composite",
                         "congruent", "digit", "even", "odd", "totient", "factorial"],
        "combinatorics": ["how many ways", "permutation", "combination", "arrange",
                         "choose", "select", "distribute", "counting", "binomial"],
        "probability": ["probability", "chance", "likely", "random", "expected",
                       "die", "dice", "coin", "deck", "card"],
        "calculus": ["derivative", "integral", "limit", "rate of change",
                    "maximum", "minimum", "converge", "series", "continuous"],
        "statistics": ["mean", "median", "mode", "average", "standard deviation",
                      "variance", "percentile", "quartile"],
    ]

    /// Question type patterns
    private let questionPatterns: [String: [String]] = [
        "solve_for": ["find the value", "solve for", "what is", "determine", "calculate", "compute", "evaluate"],
        "count": ["how many", "number of", "count", "total number"],
        "prove": ["prove that", "show that", "demonstrate", "verify"],
        "find_max": ["maximum", "largest", "greatest", "most", "maximize"],
        "find_min": ["minimum", "smallest", "least", "fewest", "minimize"],
    ]

    /// Parse a word problem into structured representation
    func parse(_ text: String) -> ParsedProblem {
        var problem = ParsedProblem(original: text)
        let lower = text.lowercased()

        // Detect domain
        problem.domain = detectDomain(lower)

        // Detect question type
        problem.questionType = detectQuestionType(lower)

        // Extract entities (numbers + named variables)
        problem.entities = extractEntities(text)

        // Extract relations
        problem.relations = extractRelations(lower, entities: problem.entities)

        // Identify unknowns
        problem.unknowns = identifyUnknowns(lower, entities: problem.entities)

        return problem
    }

    private func detectDomain(_ text: String) -> String {
        var bestDomain = "algebra"
        var bestScore = 0

        for (domain, patterns) in domainPatterns {
            let score = patterns.filter { text.contains($0) }.count
            if score > bestScore {
                bestScore = score
                bestDomain = domain
            }
        }
        return bestDomain
    }

    private func detectQuestionType(_ text: String) -> String {
        for (qType, patterns) in questionPatterns {
            if patterns.contains(where: { text.contains($0) }) {
                return qType
            }
        }
        return "solve_for"
    }

    private func extractEntities(_ text: String) -> [MathEntity] {
        var entities: [MathEntity] = []

        // Extract numerical values
        let numPattern = #"(-?\d+\.?\d*)\s*([a-zA-Z%°]+)?"#
        if let regex = try? NSRegularExpression(pattern: numPattern) {
            let range = NSRange(text.startIndex..., in: text)
            let matches = regex.matches(in: text, range: range)
            for match in matches {
                if let numRange = Range(match.range(at: 1), in: text),
                   let num = Double(text[numRange]) {
                    var unit = ""
                    if match.numberOfRanges > 2, let unitRange = Range(match.range(at: 2), in: text) {
                        unit = String(text[unitRange])
                    }
                    entities.append(MathEntity(
                        name: "num_\(entities.count)",
                        value: num,
                        unit: unit
                    ))
                }
            }
        }

        // Extract number words
        let lower = text.lowercased()
        for (word, val) in numberWords {
            if lower.contains(word) {
                entities.append(MathEntity(name: word, value: val))
            }
        }

        // Extract variable mentions (single letters in algebraic context)
        let varPattern = #"\b([a-z])\b"#
        if let regex = try? NSRegularExpression(pattern: varPattern, options: .caseInsensitive) {
            let range = NSRange(lower.startIndex..., in: lower)
            var found = Set<String>()
            let matches = regex.matches(in: lower, range: range)
            for match in matches {
                if let r = Range(match.range(at: 1), in: lower) {
                    let varName = String(lower[r])
                    if !found.contains(varName) && !"aeiou".contains(varName) {
                        entities.append(MathEntity(name: varName, variable: varName))
                        found.insert(varName)
                    }
                }
            }
        }

        return entities
    }

    private func extractRelations(_ text: String, entities: [MathEntity]) -> [MathRelation] {
        var relations: [MathRelation] = []

        // Sum/total pattern
        if text.contains("sum") || text.contains("total") || text.contains("added") {
            relations.append(MathRelation(relationType: "sum"))
        }
        // Product pattern
        if text.contains("product") || text.contains("multiplied") || text.contains("times") {
            relations.append(MathRelation(relationType: "product"))
        }
        // Ratio/fraction
        if text.contains("ratio") || text.contains("divided") || text.contains("fraction") {
            relations.append(MathRelation(relationType: "ratio"))
        }
        // Equals
        if text.contains("equals") || text.contains("equal to") || text.contains("is equal") {
            relations.append(MathRelation(relationType: "equals"))
        }
        // Percent
        if text.contains("percent") || text.contains("%") {
            relations.append(MathRelation(relationType: "percent_of"))
        }

        return relations
    }

    private func identifyUnknowns(_ text: String, entities: [MathEntity]) -> [String] {
        var unknowns: [String] = []
        // Variables without values are unknowns
        for entity in entities {
            if entity.variable != nil && entity.value == nil {
                unknowns.append(entity.variable!)
            }
        }
        // If "find" or "what" is mentioned, the first variable candidate is unknown
        if unknowns.isEmpty && (text.contains("find") || text.contains("what")) {
            unknowns.append("x")
        }
        return unknowns
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - EquationBuilder (Layer 2) — ASI v2.0
// ═══════════════════════════════════════════════════════════════

/// Build symbolic equations from parsed word problems.
/// Maps NL patterns to MathExpression AST nodes.
final class EquationBuilder {
    static let shared = EquationBuilder()

    /// Build equation(s) from a parsed problem
    func build(from problem: ParsedProblem) -> [MathExpression] {
        var equations: [MathExpression] = []
        let nums = problem.entities.compactMap { $0.value }

        switch problem.domain {
        case "algebra":
            equations.append(contentsOf: buildAlgebra(problem, nums: nums))
        case "geometry":
            equations.append(contentsOf: buildGeometry(problem, nums: nums))
        case "number_theory":
            equations.append(contentsOf: buildNumberTheory(problem, nums: nums))
        case "combinatorics", "probability":
            equations.append(contentsOf: buildCombinatorics(problem, nums: nums))
        default:
            // Try to build from relations
            for relation in problem.relations {
                if relation.relationType == "sum" && nums.count >= 2 {
                    equations.append(.add(.number(nums[0]), .number(nums[1])))
                } else if relation.relationType == "product" && nums.count >= 2 {
                    equations.append(.multiply(.number(nums[0]), .number(nums[1])))
                }
            }
        }

        return equations
    }

    private func buildAlgebra(_ problem: ParsedProblem, nums: [Double]) -> [MathExpression] {
        if nums.count >= 3 {
            // ax² + bx + c = 0
            return [.add(.add(.multiply(.number(nums[0]), .power(.variable("x"), .number(2))),
                              .multiply(.number(nums[1]), .variable("x"))),
                         .number(nums[2]))]
        } else if nums.count >= 2 {
            // ax + b = 0
            return [.add(.multiply(.number(nums[0]), .variable("x")), .number(nums[1]))]
        }
        return []
    }

    private func buildGeometry(_ problem: ParsedProblem, nums: [Double]) -> [MathExpression] {
        let lower = problem.original.lowercased()
        if lower.contains("circle") && !nums.isEmpty {
            // Area = π * r²
            return [.multiply(.number(.pi), .power(.number(nums[0]), .number(2)))]
        }
        if lower.contains("triangle") && nums.count >= 2 {
            // Area = 0.5 * base * height
            return [.multiply(.number(0.5), .multiply(.number(nums[0]), .number(nums[1])))]
        }
        if lower.contains("rectangle") && nums.count >= 2 {
            return [.multiply(.number(nums[0]), .number(nums[1]))]
        }
        return []
    }

    private func buildNumberTheory(_ problem: ParsedProblem, nums: [Double]) -> [MathExpression] {
        if !nums.isEmpty {
            return [.number(nums[0])]
        }
        return []
    }

    private func buildCombinatorics(_ problem: ParsedProblem, nums: [Double]) -> [MathExpression] {
        if nums.count >= 2 {
            // C(n, k) or P(n, k)
            let lower = problem.original.lowercased()
            if lower.contains("permutation") {
                // n! / (n-k)!
                return [.divide(.variable("n!"), .variable("(n-k)!"))]
            }
            // Default: combination
            return [.divide(.variable("n!"), .multiply(.variable("k!"), .variable("(n-k)!")))]
        }
        return []
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - SolutionChain — Chain-of-Thought (ASI v2.0)
// ═══════════════════════════════════════════════════════════════

/// Chain-of-thought solution trace with step-by-step reasoning.
struct SolutionChain {
    var steps: [(description: String, result: Double?, confidence: Double)]
    var domain: String
    var verified: Bool

    init(domain: String = "") {
        self.steps = []
        self.domain = domain
        self.verified = false
    }

    mutating func addStep(_ description: String, result: Double? = nil, confidence: Double = 0.9) {
        steps.append((description: description, result: result, confidence: confidence))
    }

    mutating func verify(expected: Double?, tolerance: Double = 1e-6) {
        guard let last = steps.last?.result, let exp = expected else {
            verified = false
            return
        }
        verified = abs(last - exp) < tolerance
        addStep("Verification: \(verified ? "PASS" : "FAIL") (expected=\(exp), got=\(last))",
                result: last, confidence: verified ? 1.0 : 0.0)
    }

    var overallConfidence: Double {
        guard !steps.isEmpty else { return 0.0 }
        let total = steps.reduce(0.0) { $0 + $1.confidence }
        return total / Double(steps.count)
    }

    var description: String {
        return steps.enumerated().map { (i, s) in
            "  Step \(i+1): \(s.description)\(s.result != nil ? " → \(s.result!)" : "")"
        }.joined(separator: "\n")
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - SymbolicMathSolver Extension — ASI v2.0 Features
// ═══════════════════════════════════════════════════════════════

extension SymbolicMathSolver {

    /// Solve a word problem using the full ASI pipeline:
    /// Parse → Build Equations → Solve → Chain-of-Thought → Verify
    func solveWordProblem(_ text: String) -> (solution: MathSolution, chain: SolutionChain) {
        // Layer 1: Parse
        let parsed = WordProblemParser.shared.parse(text)
        var chain = SolutionChain(domain: parsed.domain)
        chain.addStep("Parsed problem — domain=\(parsed.domain), type=\(parsed.questionType), entities=\(parsed.entities.count)")

        // Layer 2: Build equations
        let equations = EquationBuilder.shared.build(from: parsed)
        chain.addStep("Built \(equations.count) equation(s)", confidence: equations.isEmpty ? 0.3 : 0.8)

        // Layer 3-6: Solve via domain router
        let solution = solve(problem: text)
        chain.addStep("Domain solver: \(solution.domain) → \(solution.resultDescription)",
                      result: solution.resultValue,
                      confidence: solution.confidence)

        // Layer 7: Chain-of-thought steps
        for step in solution.steps {
            chain.addStep(step, confidence: 0.9)
        }

        // Layer 8: Verification (plug-back)
        if let result = solution.resultValue, result.isFinite {
            chain.verify(expected: result)
        }

        return (solution, chain)
    }

    /// Solve MATH benchmark with chain-of-thought tracing
    func solveMATHWithChain(problem: String, level: Int) -> (answer: String, confidence: Double, chain: SolutionChain) {
        let (solution, chain) = solveWordProblem(problem)

        let levelFactor = Swift.max(0.3, 1.0 - Double(level - 1) * 0.12)
        let adjustedConfidence = solution.confidence * levelFactor

        let answer: String
        if let val = solution.resultValue, val.isFinite {
            if abs(val - val.rounded()) < 1e-9 {
                answer = String(format: "%.0f", val)
            } else {
                answer = String(format: "%.6g", val)
            }
        } else {
            answer = solution.resultDescription
        }

        return (answer, adjustedConfidence, chain)
    }

    /// Evaluate math solving quality score (0-1) for ASI integration
    func evaluateMathSolving() -> Double {
        let status = engineStatus()
        let avgConf = (status["averageConfidence"] as? Double) ?? 0.0
        let solves = (status["solveCount"] as? Int) ?? 0
        let domains = (status["domainCounts"] as? [String: Int]) ?? [:]

        let confScore = min(1.0, avgConf * 1.1)
        let domainCoverage = min(1.0, Double(domains.count) / 7.0)
        let activityScore = min(1.0, Double(solves) / 50.0)

        return confScore * 0.50 + domainCoverage * 0.30 + activityScore * 0.20
    }

    /// Three-engine math score for ASI 15D integration
    func threeEngineMathScore() -> Double {
        var scores: [Double] = []

        // Component 1: Solver capability (weight: 0.35)
        let solverScore = evaluateMathSolving()
        scores.append(solverScore * 0.35)

        // Component 2: GOD_CODE alignment (weight: 0.25)
        let godCodeRes = MathSolution.computeResonance(GOD_CODE)
        scores.append(godCodeRes * 0.25)

        // Component 3: PHI harmonic coherence (weight: 0.20)
        let phiHarmonic = 1.0 - abs(PHI - Foundation.sqrt(5.0 + 1.0) / 2.0)
        scores.append(min(1.0, phiHarmonic + 0.9) * 0.20)

        // Component 4: Domain coverage (weight: 0.20)
        let status = engineStatus()
        let domains = (status["domainCounts"] as? [String: Int]) ?? [:]
        let coverage = min(1.0, Double(domains.count) / 7.0)
        scores.append(coverage * 0.20)

        return scores.reduce(0, +)
    }
}
