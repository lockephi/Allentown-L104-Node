// ═══════════════════════════════════════════════════════════════
// B22_MathEngines2.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 · TheBrain · v2 Architecture
//
// Extracted from L104Native.swift lines 14128-15083
// Classes: TensorCalculusEngine, OptimizationEngine, ProbabilityEngine
// ═══════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class TensorCalculusEngine {
    static let shared = TensorCalculusEngine()
    private var computations: Int = 0

    /// Metric tensor type (4x4 for spacetime)
    typealias MetricTensor = [[Double]]

    // ─── PREDEFINED METRICS ───

    /// Minkowski metric: η = diag(-1, 1, 1, 1)
    func minkowskiMetric() -> MetricTensor {
        computations += 1
        return [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }

    /// Schwarzschild metric components at radius r
    /// ds² = -(1-rₛ/r)dt² + (1-rₛ/r)⁻¹dr² + r²dθ² + r²sin²θ dφ²
    func schwarzschildMetric(r: Double, rs: Double, theta: Double = .pi / 2) -> MetricTensor {
        computations += 1
        let f = 1.0 - rs / r
        return [
            [-f, 0, 0, 0],
            [0, 1.0 / f, 0, 0],
            [0, 0, r * r, 0],
            [0, 0, 0, r * r * sin(theta) * sin(theta)]
        ]
    }

    /// Friedmann-Lemaître-Robertson-Walker (FLRW) metric
    /// ds² = -dt² + a(t)²[dr²/(1-kr²) + r²dΩ²]
    func flrwMetric(a: Double, r: Double, k: Double, theta: Double = .pi / 2) -> MetricTensor {
        computations += 1
        let a2 = a * a
        return [
            [-1, 0, 0, 0],
            [0, a2 / (1.0 - k * r * r), 0, 0],
            [0, 0, a2 * r * r, 0],
            [0, 0, 0, a2 * r * r * sin(theta) * sin(theta)]
        ]
    }

    /// Kerr metric (equatorial slice, simplified)
    /// For a rotating black hole with mass M and angular momentum parameter a
    func kerrMetricEquatorial(r: Double, rs: Double, a: Double) -> MetricTensor {
        computations += 1
        let sigma = r * r + a * a  // Σ at θ=π/2 (a²cos²θ=0)
        let delta = r * r - rs * r + a * a
        return [
            [-(1.0 - rs * r / sigma), 0, 0, -rs * r * a / sigma],
            [0, sigma / delta, 0, 0],
            [0, 0, sigma, 0],
            [-rs * r * a / sigma, 0, 0, (r * r + a * a + rs * r * a * a / sigma)]
        ]
    }

    // ─── INVERSE METRIC ───

    /// Compute inverse metric tensor g^{μν} from g_{μν}
    func inverseMetric(_ g: MetricTensor) -> MetricTensor? {
        computations += 1
        return AdvancedMathEngine.shared.inverse(g)
    }

    // ─── CHRISTOFFEL SYMBOLS ───

    /// Christoffel symbols Γ^σ_{μν} via finite differences
    /// Γ^σ_{μν} = ½g^{σρ}(∂_μ g_{νρ} + ∂_ν g_{ρμ} - ∂_ρ g_{μν})
    func christoffelSymbols(metricAt: (Double) -> MetricTensor, coordinate: Int, value: Double, h: Double = 1e-6) -> [[[Double]]] {
        computations += 1
        let n = 4
        let gCenter = metricAt(value)
        guard let gInv = inverseMetric(gCenter) else {
            return [[[Double]]](repeating: [[Double]](repeating: [Double](repeating: 0, count: n), count: n), count: n)
        }

        // Numerical derivatives of metric components
        let gPlus = metricAt(value + h)
        let gMinus = metricAt(value - h)

        var gamma = [[[Double]]](repeating: [[Double]](repeating: [Double](repeating: 0, count: n), count: n), count: n)

        // For the varied coordinate axis only (simplified to single-coordinate variation)
        for sigma in 0..<n {
            for mu in 0..<n {
                for nu in 0..<n {
                    var sum = 0.0
                    for rho in 0..<n {
                        let dgNuRho = (gPlus[nu][rho] - gMinus[nu][rho]) / (2.0 * h)
                        let dgRhoMu = (gPlus[rho][mu] - gMinus[rho][mu]) / (2.0 * h)
                        let dgMuNu = (gPlus[mu][nu] - gMinus[mu][nu]) / (2.0 * h)
                        sum += gInv[sigma][rho] * (dgNuRho + dgRhoMu - dgMuNu)
                    }
                    gamma[sigma][mu][nu] = 0.5 * sum
                }
            }
        }
        return gamma
    }

    // ─── RICCI TENSOR & SCALAR ───

    /// Ricci scalar from metric (approximate numerical computation)
    func ricciScalar(metricAt: (Double) -> MetricTensor, coordinate: Int, value: Double, h: Double = 1e-4) -> Double {
        computations += 1
        let g = metricAt(value)
        guard let gInv = inverseMetric(g) else { return 0 }

        // Compute via numerical double derivatives of the metric
        let gPlus = metricAt(value + h)
        let gMinus = metricAt(value - h)
        let gCenter = g

        // Second derivatives approximation: ∂²g_{μν}/∂x² ≈ (g+ - 2g + g-)/h²
        var R = 0.0
        for mu in 0..<4 {
            for nu in 0..<4 {
                let d2g = (gPlus[mu][nu] - 2.0 * gCenter[mu][nu] + gMinus[mu][nu]) / (h * h)
                R += gInv[mu][nu] * d2g
            }
        }
        return R
    }

    /// Kretschner scalar K = R_{αβγδ}R^{αβγδ} for Schwarzschild: K = 48M²/r⁶
    func kretschnerScalar(mass: Double, radius: Double) -> Double {
        computations += 1
        let G = 6.67430e-11
        let c = 299_792_458.0
        let rs = 2.0 * G * mass / (c * c)
        let M = rs / 2.0  // geometric mass (rs = 2M in geometric units)
        return 48.0 * M * M / pow(radius, 6)
    }

    /// Geodesic equation: d²x^μ/dτ² + Γ^μ_{αβ}(dx^α/dτ)(dx^β/dτ) = 0
    /// Solve numerically for radial geodesic in Schwarzschild spacetime
    func solveRadialGeodesic(rs: Double, r0: Double, dr0: Double, steps: Int = 500, dtau: Double = 0.01) -> [(tau: Double, r: Double, drDtau: Double)] {
        computations += 1
        var r = r0
        var dr = dr0
        var tau = 0.0
        var result: [(Double, Double, Double)] = [(tau, r, dr)]

        for _ in 0..<steps {
            // Christoffel: Γ^r_{tt} = rs(r-rs)/(2r³), Γ^r_{rr} = -rs/(2r(r-rs))
            let gammaRtt = rs * (r - rs) / (2.0 * r * r * r)
            let gammaRrr = -rs / (2.0 * r * (r - rs))

            // Energy conservation gives dt/dτ
            let f = 1.0 - rs / r
            let E = 1.0  // unit energy per rest mass
            let dtDtau = E / f

            // Geodesic equation: d²r/dτ² = -Γ^r_{tt}(dt/dτ)² - Γ^r_{rr}(dr/dτ)²
            let d2r = -gammaRtt * dtDtau * dtDtau - gammaRrr * dr * dr

            // Leapfrog integration
            dr += d2r * dtau
            r += dr * dtau
            tau += dtau

            guard r > rs * 1.01 else { break }  // Stop near horizon
            result.append((tau, r, dr))
        }
        return result
    }

    /// Proper distance between two radial coordinates in Schwarzschild
    func properDistance(rs: Double, r1: Double, r2: Double, steps: Int = 1000) -> Double {
        computations += 1
        let h = (r2 - r1) / Double(steps)
        var dist = 0.0
        for i in 0..<steps {
            let r = r1 + (Double(i) + 0.5) * h
            dist += Foundation.sqrt(1.0 / (1.0 - rs / r)) * h
        }
        return dist
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  📐 TENSOR CALCULUS & DIFFERENTIAL GEOMETRY v41.2          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Metrics:
        ║    • Minkowski, Schwarzschild, FLRW, Kerr
        ║  Tensor Operations:
        ║    • Inverse metric, Christoffel symbols
        ║    • Ricci scalar, Kretschner scalar
        ║    • Geodesic equation (radial Schwarzschild)
        ║    • Proper distance computation
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - ⚙️ OPTIMIZATION & NUMERICAL METHODS ENGINE
// Phase 41.3: Gradient descent, Newton's method, root finding,
// Lagrange multipliers, interpolation, quadrature, stiff ODEs
// ═══════════════════════════════════════════════════════════════════════════════

class OptimizationEngine {
    static let shared = OptimizationEngine()
    private var computations: Int = 0

    // ═══════════════════════════════════════════════════════════════
    // MARK: ROOT FINDING
    // ═══════════════════════════════════════════════════════════════

    /// Bisection method: find root of f(x) = 0 in [a, b]
    func bisection(f: (Double) -> Double, a: Double, b: Double, tol: Double = 1e-12, maxIter: Int = 100) -> (root: Double, iterations: Int)? {
        computations += 1
        var lo = a, hi = b
        guard f(lo) * f(hi) < 0 else { return nil }
        for iter in 1...maxIter {
            let mid = (lo + hi) / 2.0
            let fm = f(mid)
            if abs(fm) < tol || (hi - lo) / 2.0 < tol { return (mid, iter) }
            if f(lo) * fm < 0 { hi = mid } else { lo = mid }
        }
        return ((lo + hi) / 2.0, maxIter)
    }

    /// Newton-Raphson method: x_{n+1} = x_n - f(x_n)/f'(x_n)
    func newtonRaphson(f: (Double) -> Double, df: (Double) -> Double, x0: Double, tol: Double = 1e-12, maxIter: Int = 100) -> (root: Double, iterations: Int)? {
        computations += 1
        var x = x0
        for iter in 1...maxIter {
            let fx = f(x)
            let dfx = df(x)
            guard abs(dfx) > 1e-15 else { return nil }
            let xNew = x - fx / dfx
            if abs(xNew - x) < tol { return (xNew, iter) }
            x = xNew
        }
        return (x, maxIter)
    }

    /// Secant method: x_{n+1} = x_n - f(x_n)·(x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    func secant(f: (Double) -> Double, x0: Double, x1: Double, tol: Double = 1e-12, maxIter: Int = 100) -> (root: Double, iterations: Int)? {
        computations += 1
        var xPrev = x0, xCurr = x1
        for iter in 1...maxIter {
            let fPrev = f(xPrev), fCurr = f(xCurr)
            guard abs(fCurr - fPrev) > 1e-15 else { return nil }
            let xNext = xCurr - fCurr * (xCurr - xPrev) / (fCurr - fPrev)
            if abs(xNext - xCurr) < tol { return (xNext, iter) }
            xPrev = xCurr
            xCurr = xNext
        }
        return (xCurr, maxIter)
    }

    /// Brent's method: hybrid bisection + secant + inverse quadratic interpolation
    func brent(f: (Double) -> Double, a: Double, b: Double, tol: Double = 1e-12, maxIter: Int = 100) -> (root: Double, iterations: Int)? {
        computations += 1
        var a = a, b = b
        var fa = f(a), fb = f(b)
        guard fa * fb < 0 else { return nil }
        if abs(fa) < abs(fb) { swap(&a, &b); swap(&fa, &fb) }
        var c = a, fc = fa, d = b - a, e = d
        for iter in 1...maxIter {
            if fb == 0 || abs(b - a) < tol { return (b, iter) }
            if abs(fc) < abs(fb) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa }
            let tolM = 2.0 * 1e-15 * abs(b) + tol / 2.0
            let m = (c - b) / 2.0
            if abs(m) <= tolM || fb == 0 { return (b, iter) }
            if abs(e) >= tolM && abs(fa) > abs(fb) {
                let s = fb / fa
                var p: Double, q: Double
                if a == c {
                    p = 2.0 * m * s; q = 1.0 - s
                } else {
                    let q2 = fa / fc; let r = fb / fc
                    p = s * (2.0 * m * q2 * (q2 - r) - (b - a) * (r - 1.0))
                    q = (q2 - 1.0) * (r - 1.0) * (s - 1.0)
                }
                if p > 0 { q = -q } else { p = -p }
                if 2.0 * p < min(3.0 * m * q - abs(tolM * q), abs(e * q)) {
                    e = d; d = p / q
                } else { d = m; e = m }
            } else { d = m; e = m }
            a = b; fa = fb
            b += abs(d) > tolM ? d : (m > 0 ? tolM : -tolM)
            fb = f(b)
            if (fb > 0 && fc > 0) || (fb < 0 && fc < 0) { c = a; fc = fa; d = b - a; e = d }
        }
        return (b, maxIter)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: OPTIMIZATION
    // ═══════════════════════════════════════════════════════════════

    /// Gradient descent: minimize f(x) using numerical gradient
    func gradientDescent(f: ([Double]) -> Double, x0: [Double], learningRate: Double = 0.01, tol: Double = 1e-8, maxIter: Int = 10000) -> (minimum: [Double], value: Double, iterations: Int) {
        computations += 1
        var x = x0
        let h = 1e-7
        for iter in 1...maxIter {
            // Numerical gradient
            var grad = [Double](repeating: 0, count: x.count)
            for i in 0..<x.count {
                var xPlus = x; xPlus[i] += h
                var xMinus = x; xMinus[i] -= h
                grad[i] = (f(xPlus) - f(xMinus)) / (2.0 * h)
            }
            let gradNorm = Foundation.sqrt(grad.map { $0 * $0 }.reduce(0, +))
            if gradNorm < tol { return (x, f(x), iter) }
            for i in 0..<x.count { x[i] -= learningRate * grad[i] }
        }
        return (x, f(x), maxIter)
    }

    /// Golden section search: minimize f(x) on [a, b]
    func goldenSection(f: (Double) -> Double, a: Double, b: Double, tol: Double = 1e-10, maxIter: Int = 200) -> (minimum: Double, value: Double, iterations: Int) {
        computations += 1
        let gr = (Foundation.sqrt(5.0) - 1.0) / 2.0
        var a = a, b = b
        var c = b - gr * (b - a)
        var d = a + gr * (b - a)
        for iter in 1...maxIter {
            if abs(b - a) < tol { let mid = (a + b) / 2; return (mid, f(mid), iter) }
            if f(c) < f(d) { b = d } else { a = c }
            c = b - gr * (b - a)
            d = a + gr * (b - a)
        }
        let mid = (a + b) / 2
        return (mid, f(mid), maxIter)
    }

    /// Nelder-Mead simplex method for multidimensional unconstrained optimization
    func nelderMead(f: ([Double]) -> Double, x0: [Double], tol: Double = 1e-8, maxIter: Int = 5000) -> (minimum: [Double], value: Double, iterations: Int) {
        computations += 1
        let n = x0.count
        let alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5

        // Initialize simplex
        var simplex = [x0]
        for i in 0..<n {
            var v = x0
            v[i] += (abs(x0[i]) > 1e-10 ? 0.05 * x0[i] : 0.00025)
            simplex.append(v)
        }
        var fValues = simplex.map { f($0) }

        for iter in 1...maxIter {
            // Sort
            let indices = fValues.enumerated().sorted { $0.element < $1.element }.map(\.offset)
            simplex = indices.map { simplex[$0] }
            fValues = indices.map { fValues[$0] }

            // Check convergence
            let fRange = fValues.last! - fValues.first!
            if fRange < tol { return (simplex[0], fValues[0], iter) }

            // Centroid (excluding worst)
            var centroid = [Double](repeating: 0, count: n)
            for i in 0..<n { for j in 0..<n { centroid[j] += simplex[i][j] } }
            centroid = centroid.map { $0 / Double(n) }

            // Reflection
            let worst = simplex[n]
            let reflected: [Double] = (0..<n).map { (i: Int) -> Double in centroid[i] + alpha * (centroid[i] - worst[i]) }
            let fReflected = f(reflected)

            if fReflected < fValues[0] {
                // Expansion
                let expanded: [Double] = (0..<n).map { (i: Int) -> Double in centroid[i] + gamma * (reflected[i] - centroid[i]) }
                let fExpanded = f(expanded)
                if fExpanded < fReflected { simplex[n] = expanded; fValues[n] = fExpanded }
                else { simplex[n] = reflected; fValues[n] = fReflected }
            } else if fReflected < fValues[n - 1] {
                simplex[n] = reflected; fValues[n] = fReflected
            } else {
                // Contraction
                let contracted: [Double] = (0..<n).map { (i: Int) -> Double in centroid[i] + rho * (worst[i] - centroid[i]) }
                let fContracted = f(contracted)
                if fContracted < fValues[n] {
                    simplex[n] = contracted; fValues[n] = fContracted
                } else {
                    // Shrink
                    for i in 1...n {
                        simplex[i] = (0..<n).map { (j: Int) -> Double in simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) }
                        fValues[i] = f(simplex[i])
                    }
                }
            }
        }
        return (simplex[0], fValues[0], maxIter)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: INTERPOLATION
    // ═══════════════════════════════════════════════════════════════

    /// Lagrange interpolation polynomial evaluation
    func lagrangeInterpolate(xPoints: [Double], yPoints: [Double], at x: Double) -> Double {
        computations += 1
        let n = xPoints.count
        guard n > 0, n == yPoints.count else { return .nan }
        // Guard against duplicate x-points (would cause division by zero)
        guard Set(xPoints).count == n else { return .nan }
        var result = 0.0
        for i in 0..<n {
            var basis = yPoints[i]
            for j in 0..<n where j != i {
                let denom = xPoints[i] - xPoints[j]
                guard denom != 0 else { return .nan }
                basis *= (x - xPoints[j]) / denom
            }
            result += basis
        }
        return result
    }

    /// Newton's divided differences interpolation
    func newtonInterpolate(xPoints: [Double], yPoints: [Double], at x: Double) -> Double {
        computations += 1
        let n = xPoints.count
        guard n > 0, n == yPoints.count else { return .nan }
        guard Set(xPoints).count == n else { return .nan }
        var dd = yPoints  // divided differences table (1D diagonal)
        var result = dd[0]
        var product = 1.0
        for i in 1..<n {
            // Compute next level of divided differences
            for j in stride(from: n - 1, through: i, by: -1) {
                let denom = xPoints[j] - xPoints[j - i]
                guard denom != 0 else { return .nan }
                dd[j] = (dd[j] - dd[j - 1]) / denom
            }
            product *= (x - xPoints[i - 1])
            result += dd[i] * product
        }
        return result
    }

    /// Cubic spline interpolation (natural boundary conditions)
    func cubicSpline(xPoints: [Double], yPoints: [Double], at x: Double) -> Double {
        computations += 1
        let n = xPoints.count
        guard n >= 3, n == yPoints.count else { return lagrangeInterpolate(xPoints: xPoints, yPoints: yPoints, at: x) }

        // Compute h[i] = x[i+1] - x[i] — guard against zero-width intervals
        let h: [Double] = (0..<n-1).map { (i: Int) -> Double in xPoints[i + 1] - xPoints[i] }
        // Validate: all intervals must be positive (strictly monotonic x-points)
        guard h.allSatisfy({ $0 > 0 }) else { return .nan }

        // Solve for second derivatives (tridiagonal system)
        var alpha = [Double](repeating: 0, count: n)
        for i in 1..<n-1 {
            alpha[i] = 3.0 / h[i] * (yPoints[i+1] - yPoints[i]) - 3.0 / h[i-1] * (yPoints[i] - yPoints[i-1])
        }
        var l = [Double](repeating: 1, count: n)
        var mu = [Double](repeating: 0, count: n)
        var z = [Double](repeating: 0, count: n)
        for i in 1..<n-1 {
            l[i] = 2.0 * (xPoints[i+1] - xPoints[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
        }
        var c = [Double](repeating: 0, count: n)
        var b = [Double](repeating: 0, count: n - 1)
        var d = [Double](repeating: 0, count: n - 1)
        for j in stride(from: n - 2, through: 0, by: -1) {
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (yPoints[j+1] - yPoints[j]) / h[j] - h[j] * (c[j+1] + 2.0 * c[j]) / 3.0
            d[j] = (c[j+1] - c[j]) / (3.0 * h[j])
        }

        // Find interval and evaluate (clamp to valid range for out-of-bounds x)
        var idx = 0
        let xClamped = min(max(x, xPoints[0]), xPoints[n - 1])
        for i in 0..<n-1 {
            if xClamped >= xPoints[i] && xClamped <= xPoints[i+1] { idx = i; break }
        }
        let dx = xClamped - xPoints[idx]
        return yPoints[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: NUMERICAL INTEGRATION (QUADRATURE)
    // ═══════════════════════════════════════════════════════════════

    /// Gaussian quadrature (5-point Gauss-Legendre)
    func gaussianQuadrature(f: (Double) -> Double, a: Double, b: Double) -> Double {
        computations += 1
        let nodes: [Double] = [-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459]
        let weights: [Double] = [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]
        let halfWidth = (b - a) / 2.0
        let midpoint = (a + b) / 2.0
        var sum = 0.0
        for i in 0..<5 {
            sum += weights[i] * f(midpoint + halfWidth * nodes[i])
        }
        return halfWidth * sum
    }

    /// Adaptive Simpson's quadrature with error control
    func adaptiveSimpson(f: (Double) -> Double, a: Double, b: Double, tol: Double = 1e-10, maxDepth: Int = 50) -> Double {
        computations += 1
        func simpsonRule(_ a: Double, _ b: Double) -> Double {
            let c = (a + b) / 2.0
            return (b - a) / 6.0 * (f(a) + 4.0 * f(c) + f(b))
        }
        func adaptive(_ a: Double, _ b: Double, _ whole: Double, _ depth: Int) -> Double {
            let c = (a + b) / 2.0
            let left = simpsonRule(a, c)
            let right = simpsonRule(c, b)
            if depth >= maxDepth || abs(left + right - whole) <= 15.0 * tol {
                return left + right + (left + right - whole) / 15.0
            }
            return adaptive(a, c, left, depth + 1) + adaptive(c, b, right, depth + 1)
        }
        return adaptive(a, b, simpsonRule(a, b), 0)
    }

    /// Romberg integration
    func romberg(f: (Double) -> Double, a: Double, b: Double, maxOrder: Int = 10, tol: Double = 1e-12) -> Double {
        computations += 1
        var R = [[Double]](repeating: [Double](repeating: 0, count: maxOrder), count: maxOrder)
        R[0][0] = (b - a) / 2.0 * (f(a) + f(b))  // Trapezoidal
        for i in 1..<maxOrder {
            let n = 1 << i  // 2^i
            let h = (b - a) / Double(n)
            var sum = 0.0
            for k in stride(from: 1, to: n, by: 2) {
                sum += f(a + Double(k) * h)
            }
            R[i][0] = R[i-1][0] / 2.0 + h * sum
            for j in 1...i {
                let factor = pow(4.0, Double(j))
                R[i][j] = (factor * R[i][j-1] - R[i-1][j-1]) / (factor - 1.0)
            }
            if i > 1 && abs(R[i][i] - R[i-1][i-1]) < tol { return R[i][i] }
        }
        return R[maxOrder-1][maxOrder-1]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: STIFF ODE SOLVERS
    // ═══════════════════════════════════════════════════════════════

    /// Implicit Euler method for stiff ODEs: y_{n+1} = y_n + h·f(t_{n+1}, y_{n+1})
    /// Uses Newton iteration to solve the implicit equation
    func implicitEuler(f: (Double, Double) -> Double, dfdy: (Double, Double) -> Double, t0: Double, y0: Double, tEnd: Double, steps: Int = 500) -> [(t: Double, y: Double)] {
        computations += 1
        let h = (tEnd - t0) / Double(steps)
        var t = t0, y = y0
        var result: [(Double, Double)] = [(t, y)]
        for _ in 0..<steps {
            let tNext = t + h
            // Newton iteration: solve g(Y) = Y - y - h·f(tNext, Y) = 0
            var Y = y + h * f(t, y)  // explicit Euler as initial guess
            for _ in 0..<10 {
                let g = Y - y - h * f(tNext, Y)
                let dg = 1.0 - h * dfdy(tNext, Y)
                guard abs(dg) > 1e-15 else { break }
                let correction = g / dg
                Y -= correction
                if abs(correction) < 1e-12 { break }
            }
            y = Y; t = tNext
            result.append((t, y))
        }
        return result
    }

    /// BDF-2 (Backward Differentiation Formula order 2) for stiff ODEs
    func bdf2(f: (Double, Double) -> Double, dfdy: (Double, Double) -> Double, t0: Double, y0: Double, tEnd: Double, steps: Int = 500) -> [(t: Double, y: Double)] {
        computations += 1
        let h = (tEnd - t0) / Double(steps)
        var t = t0, y = y0
        var result: [(Double, Double)] = [(t, y)]
        // First step with implicit Euler
        let firstStep = implicitEuler(f: f, dfdy: dfdy, t0: t0, y0: y0, tEnd: t0 + h, steps: 1)
        guard firstStep.count > 1 else { return result }
        var yPrev = y0
        y = firstStep[1].y
        t = firstStep[1].t
        result.append((t, y))
        // Subsequent steps with BDF-2: (3/2)y_{n+1} - 2y_n + (1/2)y_{n-1} = h·f(t_{n+1}, y_{n+1})
        for _ in 2...steps {
            let tNext = t + h
            var Y = (4.0 * y - yPrev) / 3.0 + 2.0 * h * f(t, y) / 3.0  // predictor
            for _ in 0..<10 {
                let g = 1.5 * Y - 2.0 * y + 0.5 * yPrev - h * f(tNext, Y)
                let dg = 1.5 - h * dfdy(tNext, Y)
                guard abs(dg) > 1e-15 else { break }
                let correction = g / dg
                Y -= correction
                if abs(correction) < 1e-12 { break }
            }
            yPrev = y; y = Y; t = tNext
            result.append((t, y))
        }
        return result
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  ⚙️ OPTIMIZATION & NUMERICAL METHODS v41.3                ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Root Finding:
        ║    • Bisection, Newton-Raphson, Secant, Brent
        ║  Optimization:
        ║    • Gradient descent, golden section, Nelder-Mead
        ║  Interpolation:
        ║    • Lagrange, Newton divided differences, cubic spline
        ║  Quadrature:
        ║    • Gauss-Legendre, adaptive Simpson, Romberg
        ║  Stiff ODEs:
        ║    • Implicit Euler, BDF-2
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - 🎲 PROBABILITY & STOCHASTIC PROCESSES ENGINE
// Phase 42.0: Bayes' theorem, Markov chains, random walks, distributions,
// Poisson processes, queuing theory, Monte Carlo, stochastic calculus
// ═══════════════════════════════════════════════════════════════════════════════

class ProbabilityEngine {
    static let shared = ProbabilityEngine()
    private var computations: Int = 0

    // ═══════════════════════════════════════════════════════════════
    // MARK: CORE PROBABILITY
    // ═══════════════════════════════════════════════════════════════

    /// Bayes' theorem: P(A|B) = P(B|A)·P(A) / P(B)
    func bayes(priorA: Double, likelihoodBA: Double, evidenceB: Double) -> Double {
        computations += 1
        guard evidenceB > 0 else { return 0 }
        return likelihoodBA * priorA / evidenceB
    }

    /// Extended Bayes with total probability: P(A|B) = P(B|A)P(A) / [P(B|A)P(A) + P(B|¬A)P(¬A)]
    func bayesExtended(priorA: Double, likelihoodBA: Double, likelihoodBNotA: Double) -> Double {
        computations += 1
        let pB = likelihoodBA * priorA + likelihoodBNotA * (1.0 - priorA)
        guard pB > 0 else { return 0 }
        return likelihoodBA * priorA / pB
    }

    /// Law of total probability: P(B) = Σ P(B|Aᵢ)·P(Aᵢ)
    func totalProbability(conditionals: [Double], priors: [Double]) -> Double {
        computations += 1
        return zip(conditionals, priors).reduce(0.0) { $0 + $1.0 * $1.1 }
    }

    /// Inclusion-exclusion for 2 events: P(A∪B) = P(A) + P(B) - P(A∩B)
    func unionProbability(pA: Double, pB: Double, pIntersect: Double) -> Double {
        computations += 1
        return pA + pB - pIntersect
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: DISTRIBUTIONS
    // ═══════════════════════════════════════════════════════════════

    /// Poisson PMF: P(X=k) = (λ^k · e^(-λ)) / k!
    func poissonPMF(lambda: Double, k: Int) -> Double {
        computations += 1
        guard k >= 0 else { return 0 }
        var logP: Double = -lambda
        for i in 1...max(1, k) {
            logP += log(lambda) - log(Double(i))
        }
        return k == 0 ? exp(-lambda) : exp(logP)
    }

    /// Poisson CDF: P(X ≤ k) = Σ P(X=i) for i=0..k
    func poissonCDF(lambda: Double, k: Int) -> Double {
        computations += 1
        var cdf: Double = 0
        for i in 0...max(0, k) {
            cdf += poissonPMF(lambda: lambda, k: i)
        }
        return cdf
    }

    /// Geometric PMF: P(X=k) = (1-p)^(k-1) · p — trials until first success
    func geometricPMF(p: Double, k: Int) -> Double {
        computations += 1
        guard k >= 1, p > 0, p <= 1 else { return 0 }
        return pow(1.0 - p, Double(k - 1)) * p
    }

    /// Exponential PDF: f(x) = λ·e^(-λx) for x ≥ 0
    func exponentialPDF(lambda: Double, x: Double) -> Double {
        computations += 1
        guard x >= 0, lambda > 0 else { return 0 }
        return lambda * exp(-lambda * x)
    }

    /// Exponential CDF: F(x) = 1 - e^(-λx)
    func exponentialCDF(lambda: Double, x: Double) -> Double {
        computations += 1
        guard x >= 0 else { return 0 }
        return 1.0 - exp(-lambda * x)
    }

    /// Chi-squared PDF (simplified via gamma): f(x;k) = x^(k/2-1)·e^(-x/2) / (2^(k/2)·Γ(k/2))
    func chiSquaredPDF(degreesOfFreedom k: Int, x: Double) -> Double {
        computations += 1
        guard x > 0, k > 0 else { return 0 }
        let halfK: Double = Double(k) / 2.0
        let logPdf: Double = (halfK - 1.0) * log(x) - x / 2.0 - halfK * log(2.0) - lgamma(halfK)
        return exp(logPdf)
    }

    /// Student's t-distribution PDF: f(t;ν) = Γ((ν+1)/2) / (√(νπ)·Γ(ν/2)) · (1+t²/ν)^(-(ν+1)/2)
    func studentTPDF(degreesOfFreedom nu: Int, t: Double) -> Double {
        computations += 1
        let v: Double = Double(nu)
        let coeff: Double = exp(lgamma((v + 1.0) / 2.0) - lgamma(v / 2.0)) / Foundation.sqrt(v * .pi)
        return coeff * pow(1.0 + t * t / v, -(v + 1.0) / 2.0)
    }

    /// Beta function: B(α,β) = Γ(α)Γ(β)/Γ(α+β)
    func betaFunction(alpha: Double, beta: Double) -> Double {
        computations += 1
        return exp(lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta))
    }

    /// Beta distribution PDF: f(x;α,β) = x^(α-1)·(1-x)^(β-1) / B(α,β)
    func betaPDF(alpha: Double, beta: Double, x: Double) -> Double {
        computations += 1
        guard x > 0, x < 1 else { return 0 }
        return pow(x, alpha - 1.0) * pow(1.0 - x, beta - 1.0) / betaFunction(alpha: alpha, beta: beta)
    }

    /// Uniform distribution: E[X] = (a+b)/2, Var[X] = (b-a)²/12
    func uniformStats(a: Double, b: Double) -> (mean: Double, variance: Double, entropy: Double) {
        computations += 1
        return ((a + b) / 2.0, (b - a) * (b - a) / 12.0, log(b - a))
    }

    /// Log-normal PDF: f(x;μ,σ) = (1/(xσ√(2π)))·e^(-(ln(x)-μ)²/(2σ²))
    func logNormalPDF(mu: Double, sigma: Double, x: Double) -> Double {
        computations += 1
        guard x > 0, sigma > 0 else { return 0 }
        let logX: Double = log(x)
        let exponent: Double = -(logX - mu) * (logX - mu) / (2.0 * sigma * sigma)
        return exp(exponent) / (x * sigma * Foundation.sqrt(2.0 * .pi))
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: MARKOV CHAINS
    // ═══════════════════════════════════════════════════════════════

    /// Markov chain state after n steps: π(n) = π(0) · P^n
    func markovEvolve(initialState: [Double], transitionMatrix: [[Double]], steps: Int) -> [Double] {
        computations += 1
        var state = initialState
        let n = state.count
        for _ in 0..<steps {
            var newState = [Double](repeating: 0, count: n)
            for j in 0..<n {
                for i in 0..<n {
                    newState[j] += state[i] * transitionMatrix[i][j]
                }
            }
            state = newState
        }
        return state
    }

    /// Steady-state distribution: solve πP = π, Σπᵢ = 1
    /// Uses power iteration to approximate stationary distribution
    func markovSteadyState(transitionMatrix: [[Double]], maxIter: Int = 1000, tol: Double = 1e-10) -> [Double] {
        computations += 1
        let n = transitionMatrix.count
        var state = [Double](repeating: 1.0 / Double(n), count: n)
        for _ in 0..<maxIter {
            var newState = [Double](repeating: 0, count: n)
            for j in 0..<n {
                for i in 0..<n {
                    newState[j] += state[i] * transitionMatrix[i][j]
                }
            }
            var diff: Double = 0
            for i in 0..<n { diff += abs(newState[i] - state[i]) }
            state = newState
            if diff < tol { break }
        }
        return state
    }

    /// Absorbing Markov chain: expected steps to absorption from each transient state
    /// Returns expected number of steps from each transient state to absorption
    func markovAbsorptionTime(transitionMatrix: [[Double]], absorbingStates: Set<Int>) -> [Double] {
        computations += 1
        let n = transitionMatrix.count
        let transientStates = (0..<n).filter { !absorbingStates.contains($0) }
        let t = transientStates.count
        guard t > 0 else { return [] }

        // Extract Q matrix (transient-to-transient transitions)
        var Q = [[Double]](repeating: [Double](repeating: 0, count: t), count: t)
        for i in 0..<t {
            for j in 0..<t {
                Q[i][j] = transitionMatrix[transientStates[i]][transientStates[j]]
            }
        }
        // N = (I - Q)^(-1), expected steps = N·1
        var IminusQ = Q
        for i in 0..<t {
            for j in 0..<t {
                IminusQ[i][j] = (i == j ? 1.0 : 0.0) - Q[i][j]
            }
        }
        guard let N = AdvancedMathEngine.shared.inverse(IminusQ) else { return [] }
        return (0..<t).map { i in N[i].reduce(0, +) }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: RANDOM WALKS & STOCHASTIC PROCESSES
    // ═══════════════════════════════════════════════════════════════

    /// 1D symmetric random walk: E[position after n steps] = 0, Var = n
    /// Return probability of being at position k after n steps
    func randomWalkProbability(steps n: Int, position k: Int) -> Double {
        computations += 1
        // Must have same parity: n and k
        guard (n + k) % 2 == 0, abs(k) <= n else { return 0 }
        let r = (n + k) / 2  // number of right steps
        let binom: Double = Double(AdvancedMathEngine.shared.binomial(n, r))
        return binom * pow(0.5, Double(n))
    }

    /// Gambler's ruin: probability of reaching N starting from k with prob p of winning each round
    func gamblersRuin(startingWealth k: Int, targetWealth N: Int, winProb p: Double) -> Double {
        computations += 1
        guard k > 0, k < N else { return k >= N ? 1.0 : 0.0 }
        if abs(p - 0.5) < 1e-10 {
            return Double(k) / Double(N)
        }
        let r: Double = (1.0 - p) / p
        return (pow(r, Double(k)) - 1.0) / (pow(r, Double(N)) - 1.0)
    }

    /// Brownian motion properties: E[B(t)] = 0, Var[B(t)] = t, B(t) ~ N(0,t)
    func brownianMotionVariance(time: Double) -> (mean: Double, variance: Double, stdDev: Double) {
        computations += 1
        return (0.0, time, Foundation.sqrt(time))
    }

    /// Geometric Brownian Motion: S(t) = S₀·exp((μ-σ²/2)t + σW(t))
    /// Expected value: E[S(t)] = S₀·exp(μt)
    func geometricBrownianExpected(s0: Double, drift mu: Double, time t: Double) -> Double {
        computations += 1
        return s0 * exp(mu * t)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: QUEUING THEORY
    // ═══════════════════════════════════════════════════════════════

    /// M/M/1 queue: arrival rate λ, service rate μ
    func mm1Queue(arrivalRate lambda: Double, serviceRate mu: Double) -> (utilization: Double, avgQueue: Double, avgSystem: Double, avgWaitTime: Double, avgSystemTime: Double)? {
        computations += 1
        let rho: Double = lambda / mu
        guard rho < 1.0 else { return nil }  // unstable
        let Lq: Double = rho * rho / (1.0 - rho)
        let Ls: Double = rho / (1.0 - rho)
        let Wq: Double = Lq / lambda
        let Ws: Double = Ls / lambda
        return (rho, Lq, Ls, Wq, Ws)
    }

    /// M/M/c queue: c servers, arrival rate λ, service rate μ
    func mmcQueueUtilization(arrivalRate lambda: Double, serviceRate mu: Double, servers c: Int) -> Double {
        computations += 1
        return lambda / (Double(c) * mu)
    }

    /// Erlang C formula: probability of having to wait (M/M/c queue)
    func erlangC(arrivalRate lambda: Double, serviceRate mu: Double, servers c: Int) -> Double {
        computations += 1
        let a: Double = lambda / mu  // offered load
        let rho: Double = a / Double(c)
        guard rho < 1.0 else { return 1.0 }

        // Compute (a^c/c!) / (1-ρ)
        var acOverCFact: Double = 1
        for i in 1...c { acOverCFact *= a / Double(i) }
        let numerator: Double = acOverCFact / (1.0 - rho)

        // Compute Σ_{k=0}^{c-1} a^k/k!
        var sum: Double = 0
        var term: Double = 1
        sum += term
        for k in 1..<c {
            term *= a / Double(k)
            sum += term
        }
        return numerator / (sum + numerator)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: MONTE CARLO
    // ═══════════════════════════════════════════════════════════════

    /// Estimate π via Monte Carlo: π ≈ 4 · (points in circle) / (total points)
    func monteCarloPI(samples: Int = 100000) -> Double {
        computations += 1
        var inside = 0
        // Use deterministic quasi-random for reproducibility
        for i in 0..<samples {
            let x: Double = Double(i * 1103515245 + 12345) / Double(Int.max)
            let hash: Int = (i * 6364136223846793005 + 1442695040888963407) & Int.max
            let y: Double = Double(hash) / Double(Int.max)
            if x * x + y * y <= 1.0 { inside += 1 }
        }
        return 4.0 * Double(inside) / Double(samples)
    }

    /// Monte Carlo integration: ∫_a^b f(x) dx ≈ (b-a)/N · Σf(xᵢ)
    func monteCarloIntegrate(f: (Double) -> Double, a: Double, b: Double, samples: Int = 10000) -> Double {
        computations += 1
        var sum: Double = 0
        let width: Double = b - a
        for i in 0..<samples {
            let frac: Double = Double(i) / Double(samples) + 0.5 / Double(samples)
            let x: Double = a + frac * width
            sum += f(x)
        }
        return width * sum / Double(samples)
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🎲 PROBABILITY & STOCHASTIC PROCESSES v42.0              ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Probability:
        ║    • Bayes (simple + extended), total probability
        ║  Distributions:
        ║    • Poisson, geometric, exponential, chi-squared
        ║    • Student's t, beta, log-normal, uniform
        ║  Stochastic Processes:
        ║    • Markov chains (evolve, steady-state, absorption)
        ║    • Random walks, gambler's ruin, Brownian motion
        ║    • Geometric Brownian motion (GBM)
        ║  Queuing Theory:
        ║    • M/M/1 queue, M/M/c utilization, Erlang C
        ║  Monte Carlo:
        ║    • π estimation, numerical integration
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
