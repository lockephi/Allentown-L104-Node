// ═══════════════════════════════════════════════════════════════════════════════
// B21_MathEngines1.swift — L104 · TheBrain · v2 Architecture
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// Extracted from L104Native.swift lines 12883-14127
// Classes: AdvancedMathEngine, FluidWaveEngine, InformationSignalEngine
// ═══════════════════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - 📐 ADVANCED MATH ENGINE
// Phase 29.0: Symbolic calculus, linear algebra, number theory,
// statistics, differential equations, series & sequences
// ═══════════════════════════════════════════════════════════════════════════════

class AdvancedMathEngine {
    static let shared = AdvancedMathEngine()

    private var computations: Int = 0

    func derivative(of expression: String, withRespectTo variable: String = "x") -> String {
        computations += 1
        let expr = expression.trimmingCharacters(in: .whitespaces)
        if Double(expr) != nil { return "0" }
        if expr == variable { return "1" }
        if let range = expr.range(of: "^\(variable)\\^(-?[\\d.]+)", options: .regularExpression) {
            let matched = String(expr[range])
            let nStr = matched.components(separatedBy: "^").last ?? "1"
            if let n = Double(nStr) {
                if n == 2 { return "2\(variable)" }
                if n == 1 { return "1" }
                return "\(formatNum(n))\(variable)^\(formatNum(n - 1))"
            }
        }
        let polyRegex = try? NSRegularExpression(pattern: "^(-?[\\d.]+)\\*?\(variable)\\^(-?[\\d.]+)$")
        if let regex = polyRegex,
           let match = regex.firstMatch(in: expr, range: NSRange(expr.startIndex..., in: expr)),
           match.numberOfRanges == 3 {
            let aStr = String(expr[Range(match.range(at: 1), in: expr)!])
            let nStr = String(expr[Range(match.range(at: 2), in: expr)!])
            if let a = Double(aStr), let n = Double(nStr) {
                let coeff = a * n
                let newExp = n - 1
                if newExp == 0 { return formatNum(coeff) }
                if newExp == 1 { return "\(formatNum(coeff))\(variable)" }
                return "\(formatNum(coeff))\(variable)^\(formatNum(newExp))"
            }
        }
        if expr == "sin(\(variable))" { return "cos(\(variable))" }
        if expr == "cos(\(variable))" { return "-sin(\(variable))" }
        if expr == "tan(\(variable))" { return "sec²(\(variable))" }
        if expr == "sec(\(variable))" { return "sec(\(variable))tan(\(variable))" }
        if expr == "csc(\(variable))" { return "-csc(\(variable))cot(\(variable))" }
        if expr == "cot(\(variable))" { return "-csc²(\(variable))" }
        if expr == "arcsin(\(variable))" || expr == "asin(\(variable))" { return "1/√(1-\(variable)²)" }
        if expr == "arccos(\(variable))" || expr == "acos(\(variable))" { return "-1/√(1-\(variable)²)" }
        if expr == "arctan(\(variable))" || expr == "atan(\(variable))" { return "1/(1+\(variable)²)" }
        if expr == "e^\(variable)" || expr == "exp(\(variable))" { return "e^\(variable)" }
        if expr == "ln(\(variable))" || expr == "log(\(variable))" { return "1/\(variable)" }
        if expr == "\(variable)^" + variable { return "\(variable)^\(variable)(ln(\(variable))+1)" }
        let aExpRegex = try? NSRegularExpression(pattern: "^([\\d.]+)\\^\(variable)$")
        if let regex = aExpRegex,
           let match = regex.firstMatch(in: expr, range: NSRange(expr.startIndex..., in: expr)),
           match.numberOfRanges == 2 {
            let aStr = String(expr[Range(match.range(at: 1), in: expr)!])
            return "\(aStr)^\(variable)·ln(\(aStr))"
        }
        if let plusIdx = findTopLevelOperator(expr, op: "+") {
            let left = String(expr[expr.startIndex..<plusIdx]).trimmingCharacters(in: .whitespaces)
            let right = String(expr[expr.index(after: plusIdx)...]).trimmingCharacters(in: .whitespaces)
            let dLeft = derivative(of: left, withRespectTo: variable)
            let dRight = derivative(of: right, withRespectTo: variable)
            if dLeft == "0" { return dRight }
            if dRight == "0" { return dLeft }
            return "\(dLeft) + \(dRight)"
        }
        if let mulIdx = findTopLevelOperator(expr, op: "*") {
            let left = String(expr[expr.startIndex..<mulIdx]).trimmingCharacters(in: .whitespaces)
            let right = String(expr[expr.index(after: mulIdx)...]).trimmingCharacters(in: .whitespaces)
            let dLeft = derivative(of: left, withRespectTo: variable)
            let dRight = derivative(of: right, withRespectTo: variable)
            return "(\(dLeft))(\(right)) + (\(left))(\(dRight))"
        }
        return "d/d\(variable)[\(expr)]"
    }

    func integral(of expression: String, withRespectTo variable: String = "x") -> String {
        computations += 1
        let expr = expression.trimmingCharacters(in: .whitespaces)
        if let a = Double(expr) { return "\(formatNum(a))\(variable) + C" }
        if expr == variable { return "\(variable)²/2 + C" }
        if let range = expr.range(of: "^\(variable)\\^(-?[\\d.]+)$", options: .regularExpression) {
            let matched = String(expr[range])
            let nStr = matched.components(separatedBy: "^").last ?? "1"
            if let n = Double(nStr), n != -1 {
                let newExp = n + 1
                return "\(variable)^\(formatNum(newExp))/\(formatNum(newExp)) + C"
            }
            if let n = Double(nStr), n == -1 { return "ln|\(variable)| + C" }
        }
        let polyRegex = try? NSRegularExpression(pattern: "^(-?[\\d.]+)\\*?\(variable)\\^(-?[\\d.]+)$")
        if let regex = polyRegex,
           let match = regex.firstMatch(in: expr, range: NSRange(expr.startIndex..., in: expr)),
           match.numberOfRanges == 3 {
            let aStr = String(expr[Range(match.range(at: 1), in: expr)!])
            let nStr = String(expr[Range(match.range(at: 2), in: expr)!])
            if let a = Double(aStr), let n = Double(nStr), n != -1 {
                let newExp = n + 1
                let coeff = a / newExp
                return "\(formatNum(coeff))\(variable)^\(formatNum(newExp)) + C"
            }
        }
        if expr == "sin(\(variable))" { return "-cos(\(variable)) + C" }
        if expr == "cos(\(variable))" { return "sin(\(variable)) + C" }
        if expr == "sec²(\(variable))" || expr == "sec^2(\(variable))" { return "tan(\(variable)) + C" }
        if expr == "csc²(\(variable))" || expr == "csc^2(\(variable))" { return "-cot(\(variable)) + C" }
        if expr == "sec(\(variable))tan(\(variable))" { return "sec(\(variable)) + C" }
        if expr == "csc(\(variable))cot(\(variable))" { return "-csc(\(variable)) + C" }
        if expr == "tan(\(variable))" { return "-ln|cos(\(variable))| + C" }
        if expr == "cot(\(variable))" { return "ln|sin(\(variable))| + C" }
        if expr == "e^\(variable)" || expr == "exp(\(variable))" { return "e^\(variable) + C" }
        if expr == "1/\(variable)" { return "ln|\(variable)| + C" }
        return "∫\(expr) d\(variable) + C"
    }

    func definiteIntegral(of f: (Double) -> Double, from a: Double, to b: Double, intervals: Int = 1000) -> Double {
        computations += 1
        let n = intervals % 2 == 0 ? intervals : intervals + 1
        let h = (b - a) / Double(n)
        var sum = f(a) + f(b)
        for i in 1..<n { sum += (i % 2 == 0 ? 2.0 : 4.0) * f(a + Double(i) * h) }
        return sum * h / 3.0
    }

    /// EVO_58: O(n³) determinant via LU decomposition (LAPACK dgetrf)
    /// Replaces O(n!) cofactor expansion — safe for any matrix size
    func determinant(_ matrix: [[Double]]) -> Double {
        computations += 1
        let n = matrix.count
        guard n > 0 else { return 0 }
        if n == 1 { return matrix[0][0] }
        if n == 2 { return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] }

        // Flatten to column-major for LAPACK
        var a = [Double](repeating: 0.0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                a[j * n + i] = matrix[i][j]  // column-major
            }
        }

        var M = __CLPK_integer(n)
        var N = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgetrf_(&M, &N, &a, &lda, &ipiv, &info)

        guard info == 0 else {
            // Singular or error — fall back to manual for small matrices
            if n == 3 {
                return matrix[0][0] * (matrix[1][1]*matrix[2][2] - matrix[1][2]*matrix[2][1])
                     - matrix[0][1] * (matrix[1][0]*matrix[2][2] - matrix[1][2]*matrix[2][0])
                     + matrix[0][2] * (matrix[1][0]*matrix[2][1] - matrix[1][1]*matrix[2][0])
            }
            return 0.0
        }

        // Product of diagonal elements × sign from permutation
        var det = 1.0
        var swaps = 0
        for i in 0..<n {
            det *= a[i * n + i]  // diagonal of U
            if ipiv[i] != __CLPK_integer(i + 1) { swaps += 1 }
        }
        return swaps % 2 == 0 ? det : -det
    }

    func inverse(_ matrix: [[Double]]) -> [[Double]]? {
        computations += 1
        let n = matrix.count
        guard n > 0 else { return nil }
        for row in matrix { if row.count != n { return nil } }
        var aug = matrix.enumerated().map { i, row in row + (0..<n).map { $0 == i ? 1.0 : 0.0 } }
        for col in 0..<n {
            var maxRow = col
            for row in (col+1)..<n { if abs(aug[row][col]) > abs(aug[maxRow][col]) { maxRow = row } }
            if maxRow != col { aug.swapAt(col, maxRow) }
            guard abs(aug[col][col]) > 1e-14 else { return nil }
            let pivot = aug[col][col]
            for j in 0..<(2*n) { aug[col][j] /= pivot }
            for row in 0..<n where row != col {
                let factor = aug[row][col]
                for j in 0..<(2*n) { aug[row][j] -= factor * aug[col][j] }
            }
        }
        return aug.map { Array($0[n...]) }
    }

    func eigenvalues(_ matrix: [[Double]]) -> [Complex] {
        computations += 1
        let n = matrix.count
        if n == 2 {
            let a = matrix[0][0], b = matrix[0][1], c = matrix[1][0], d = matrix[1][1]
            let trace = a + d
            let det = a * d - b * c
            let disc = trace * trace - 4.0 * det
            if disc >= 0 {
                return [Complex((trace + Foundation.sqrt(disc)) / 2.0, 0),
                        Complex((trace - Foundation.sqrt(disc)) / 2.0, 0)]
            } else {
                let realPart = trace / 2.0
                let imagPart = Foundation.sqrt(-disc) / 2.0
                return [Complex(realPart, imagPart), Complex(realPart, -imagPart)]
            }
        }
        if n == 3 {
            let tr = matrix[0][0] + matrix[1][1] + matrix[2][2]
            let m00 = matrix[1][1]*matrix[2][2] - matrix[1][2]*matrix[2][1]
            let m11 = matrix[0][0]*matrix[2][2] - matrix[0][2]*matrix[2][0]
            let m22 = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
            let minorSum = m00 + m11 + m22
            let det = determinant(matrix)
            return solveCubic(a: 1, b: -tr, c: minorSum, d: -det)
        }
        return powerIteration(matrix, iterations: 200)
    }

    func singularValues(_ matrix: [[Double]]) -> [Double] {
        computations += 1
        let at = transpose(matrix)
        let ata = matMul(at, matrix)
        let eigVals = eigenvalues(ata)
        return eigVals.map { Foundation.sqrt(max(0, $0.real)) }.sorted(by: >)
    }

    func rank(_ matrix: [[Double]], tolerance: Double = 1e-10) -> Int {
        let svs = singularValues(matrix)
        return svs.filter { $0 > tolerance }.count
    }

    func solveLinearSystem(_ A: [[Double]], _ b: [Double]) -> [Double]? {
        computations += 1
        let n = A.count
        guard n > 0, b.count == n else { return nil }
        var aug = A.enumerated().map { i, row in row + [b[i]] }
        for col in 0..<n {
            var maxRow = col
            for row in (col+1)..<n { if abs(aug[row][col]) > abs(aug[maxRow][col]) { maxRow = row } }
            aug.swapAt(col, maxRow)
            guard abs(aug[col][col]) > 1e-14 else { return nil }
            for row in (col+1)..<n {
                let factor = aug[row][col] / aug[col][col]
                for j in col..<(n+1) { aug[row][j] -= factor * aug[col][j] }
            }
        }
        var x = [Double](repeating: 0, count: n)
        for i in stride(from: n-1, through: 0, by: -1) {
            var sum = aug[i][n]
            for j in (i+1)..<n { sum -= aug[i][j] * x[j] }
            x[i] = sum / aug[i][i]
        }
        return x
    }

    func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? abs(a) : gcd(b, a % b) }
    func lcm(_ a: Int, _ b: Int) -> Int { abs(a * b) / gcd(a, b) }

    func extendedGCD(_ a: Int, _ b: Int) -> (gcd: Int, x: Int, y: Int) {
        if b == 0 { return (a, 1, 0) }
        let result = extendedGCD(b, a % b)
        return (result.gcd, result.y, result.x - (a / b) * result.y)
    }

    func modPow(_ base: Int, _ exp: Int, _ mod: Int) -> Int {
        guard mod > 1 else { return 0 }
        var result = 1
        var b = base % mod
        var e = exp
        while e > 0 {
            if e % 2 == 1 { result = result * b % mod }
            e /= 2
            b = b * b % mod
        }
        return result
    }

    func modInverse(_ a: Int, _ m: Int) -> Int? {
        let result = extendedGCD(a, m)
        guard result.gcd == 1 else { return nil }
        return ((result.x % m) + m) % m
    }

    func eulerTotient(_ n: Int) -> Int {
        guard n > 1 else { return n }
        var result = n
        var num = n
        var d = 2
        while d * d <= num {
            if num % d == 0 {
                while num % d == 0 { num /= d }
                result -= result / d
            }
            d += 1
        }
        if num > 1 { result -= result / num }
        return result
    }

    func primeFactors(_ n: Int) -> [(prime: Int, power: Int)] {
        computations += 1
        guard n > 1 else { return [] }
        var factors: [(Int, Int)] = []
        var num = n
        var d = 2
        while d * d <= num {
            var count = 0
            while num % d == 0 { count += 1; num /= d }
            if count > 0 { factors.append((d, count)) }
            d += 1
        }
        if num > 1 { factors.append((num, 1)) }
        return factors
    }

    func primeSieve(_ n: Int) -> [Int] {
        guard n >= 2 else { return [] }
        var sieve = [Bool](repeating: true, count: n + 1)
        sieve[0] = false; sieve[1] = false
        for i in 2...Int(Foundation.sqrt(Double(n))) + 1 where i <= n {
            if sieve[i] { for j in stride(from: i*i, through: n, by: i) { sieve[j] = false } }
        }
        return sieve.enumerated().compactMap { $0.element ? $0.offset : nil }
    }

    func classifyNumber(_ n: Int) -> String {
        guard n > 1 else { return "N/A" }
        var sumDivisors = 1
        for d in 2...Int(Foundation.sqrt(Double(n))) {
            if n % d == 0 {
                sumDivisors += d
                if d != n / d { sumDivisors += n / d }
            }
        }
        if sumDivisors == n { return "perfect" }
        if sumDivisors > n { return "abundant (σ=\(sumDivisors))" }
        return "deficient (σ=\(sumDivisors))"
    }

    func statistics(_ data: [Double]) -> [String: Double] {
        computations += 1
        guard !data.isEmpty else { return [:] }
        let sorted: [Double] = data.sorted()
        let n: Double = Double(data.count)
        let mean: Double = data.reduce(0, +) / n
        var varianceSum: Double = 0
        for val in data { let diff: Double = val - mean; varianceSum += diff * diff }
        let variance: Double = varianceSum / n
        let stddev: Double = Foundation.sqrt(variance)
        let median: Double
        if data.count % 2 == 0 { median = (sorted[data.count/2 - 1] + sorted[data.count/2]) / 2.0 }
        else { median = sorted[data.count/2] }
        let q1: Double = sorted[data.count / 4]
        let q3: Double = sorted[3 * data.count / 4]
        let iqr: Double = q3 - q1
        let safeStddev: Double = max(stddev, 1e-10)
        var skewnessSum: Double = 0
        var kurtosisSum: Double = 0
        for val in data {
            let z: Double = (val - mean) / safeStddev
            skewnessSum += z * z * z
            kurtosisSum += z * z * z * z
        }
        let skewness: Double = skewnessSum / n
        let kurtosis: Double = kurtosisSum / n - 3.0
        return ["count": n, "mean": mean, "median": median, "variance": variance, "stddev": stddev,
                "min": sorted.first!, "max": sorted.last!, "q1": q1, "q3": q3, "iqr": iqr,
                "skewness": skewness, "kurtosis": kurtosis, "range": sorted.last! - sorted.first!]
    }

    func linearRegression(_ x: [Double], _ y: [Double]) -> (slope: Double, intercept: Double, rSquared: Double)? {
        computations += 1
        guard x.count == y.count, x.count > 1 else { return nil }
        let n: Double = Double(x.count)
        let sumX: Double = x.reduce(0, +)
        let sumY: Double = y.reduce(0, +)
        var sumXY: Double = 0; var sumX2: Double = 0
        for i in 0..<x.count { sumXY += x[i] * y[i]; sumX2 += x[i] * x[i] }
        let denom: Double = n * sumX2 - sumX * sumX
        guard abs(denom) > 1e-14 else { return nil }
        let slope: Double = (n * sumXY - sumX * sumY) / denom
        let intercept: Double = (sumY - slope * sumX) / n
        var ssRes: Double = 0; let meanY: Double = sumY / n; var ssTot: Double = 0
        for i in 0..<x.count {
            let predicted: Double = slope * x[i] + intercept
            ssRes += (predicted - y[i]) * (predicted - y[i])
            ssTot += (y[i] - meanY) * (y[i] - meanY)
        }
        let rSquared: Double = ssTot > 0 ? 1.0 - ssRes / ssTot : 1.0
        return (slope, intercept, rSquared)
    }

    func normalPDF(_ x: Double, mean: Double = 0, stddev: Double = 1) -> Double {
        let z = (x - mean) / stddev
        return exp(-0.5 * z * z) / (stddev * Foundation.sqrt(2.0 * .pi))
    }

    func normalCDF(_ x: Double, mean: Double = 0, stddev: Double = 1) -> Double {
        let z = (x - mean) / (stddev * Foundation.sqrt(2.0))
        return 0.5 * (1.0 + erf(z))
    }

    func binomial(_ n: Int, _ k: Int) -> Int {
        guard k >= 0 && k <= n else { return 0 }
        if k == 0 || k == n { return 1 }
        var result = 1
        for i in 0..<min(k, n - k) { result = result * (n - i) / (i + 1) }
        return result
    }

    func permutations(_ n: Int, _ k: Int) -> Int {
        guard k >= 0 && k <= n else { return 0 }
        var result = 1
        for i in 0..<k { result *= (n - i) }
        return result
    }

    func taylorSeries(function: String, around a: Double = 0, terms: Int = 8) -> String {
        computations += 1
        switch function.lowercased() {
        case "e^x", "exp":
            let coeffs = (0..<terms).map { n in "x^\(n)/\(factorial(n))" }
            return "e^x = " + coeffs.joined(separator: " + ") + " + ..."
        case "sin":
            let coeffs = (0..<terms).map { n -> String in
                let sign = n % 2 == 0 ? "" : "-"
                return "\(sign)x^\(2*n+1)/\(factorial(2*n+1))"
            }
            return "sin(x) = " + coeffs.joined(separator: " + ") + " + ..."
        case "cos":
            let coeffs = (0..<terms).map { n -> String in
                let sign = n % 2 == 0 ? "" : "-"
                return "\(sign)x^\(2*n)/\(factorial(2*n))"
            }
            return "cos(x) = " + coeffs.joined(separator: " + ") + " + ..."
        case "ln(1+x)", "log(1+x)":
            let coeffs = (1...terms).map { n -> String in
                let sign = n % 2 == 1 ? "" : "-"
                return "\(sign)x^\(n)/\(n)"
            }
            return "ln(1+x) = " + coeffs.joined(separator: " + ") + " + ..."
        case "1/(1-x)", "geometric":
            let coeffs = (0..<terms).map { "x^\($0)" }
            return "1/(1-x) = " + coeffs.joined(separator: " + ") + " + ... (|x| < 1)"
        default:
            return "Taylor[\(function)] — Use: exp, sin, cos, ln(1+x), 1/(1-x)"
        }
    }

    func continuedFraction(_ x: Double, maxTerms: Int = 15) -> [Int] {
        var result: [Int] = []
        var val = x
        for _ in 0..<maxTerms {
            let intPart = Int(Foundation.floor(val))
            result.append(intPart)
            let frac = val - Double(intPart)
            if abs(frac) < 1e-10 { break }
            val = 1.0 / frac
        }
        return result
    }

    func seriesSum(from start: Int, to end: Int, f: (Int) -> Double) -> Double {
        (start...end).map { f($0) }.reduce(0, +)
    }

    func solveODE(f: (Double, Double) -> Double, x0: Double, y0: Double, xEnd: Double, steps: Int = 1000) -> [(x: Double, y: Double)] {
        computations += 1
        let h = (xEnd - x0) / Double(steps)
        var x = x0, y = y0
        var result: [(Double, Double)] = [(x, y)]
        for _ in 0..<steps {
            let k1 = h * f(x, y)
            let k2 = h * f(x + h/2, y + k1/2)
            let k3 = h * f(x + h/2, y + k2/2)
            let k4 = h * f(x + h, y + k3)
            y += (k1 + 2*k2 + 2*k3 + k4) / 6
            x += h
            result.append((x, y))
        }
        return result
    }

    func solveODESystem(
        f: (Double, Double, Double) -> Double,
        g: (Double, Double, Double) -> Double,
        t0: Double, x0: Double, y0: Double, tEnd: Double, steps: Int = 1000
    ) -> [(t: Double, x: Double, y: Double)] {
        computations += 1
        let h = (tEnd - t0) / Double(steps)
        var t = t0, x = x0, y = y0
        var result: [(Double, Double, Double)] = [(t, x, y)]
        for _ in 0..<steps {
            let kx1 = h * f(t, x, y);         let ky1 = h * g(t, x, y)
            let kx2 = h * f(t+h/2, x+kx1/2, y+ky1/2); let ky2 = h * g(t+h/2, x+kx1/2, y+ky1/2)
            let kx3 = h * f(t+h/2, x+kx2/2, y+ky2/2); let ky3 = h * g(t+h/2, x+kx2/2, y+ky2/2)
            let kx4 = h * f(t+h, x+kx3, y+ky3);       let ky4 = h * g(t+h, x+kx3, y+ky3)
            x += (kx1 + 2*kx2 + 2*kx3 + kx4) / 6
            y += (ky1 + 2*ky2 + 2*ky3 + ky4) / 6
            t += h
            result.append((t, x, y))
        }
        return result
    }

    private func formatNum(_ x: Double) -> String {
        x == Foundation.floor(x) ? "\(Int(x))" : String(format: "%.4g", x)
    }

    private func factorial(_ n: Int) -> Int {
        guard n > 1 else { return 1 }
        return (2...n).reduce(1, *)
    }

    private func findTopLevelOperator(_ expr: String, op: Character) -> String.Index? {
        var depth = 0
        for (idx, ch) in zip(expr.indices, expr) {
            if ch == "(" { depth += 1 }
            else if ch == ")" { depth -= 1 }
            else if ch == op && depth == 0 { return idx }
        }
        return nil
    }

    private func submatrix(_ m: [[Double]], excludingRow row: Int, excludingCol col: Int) -> [[Double]] {
        var result = [[Double]]()
        for i in 0..<m.count {
            guard i != row else { continue }
            var newRow = [Double]()
            for j in 0..<m[i].count {
                guard j != col else { continue }
                newRow.append(m[i][j])
            }
            result.append(newRow)
        }
        return result
    }

    private func transpose(_ m: [[Double]]) -> [[Double]] {
        guard let first = m.first else { return [] }
        return (0..<first.count).map { j in m.map { $0[j] } }
    }

    private func matMul(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
        let m = a.count, n = b[0].count, p = b.count
        var c = [[Double]](repeating: [Double](repeating: 0, count: n), count: m)
        for i in 0..<m { for j in 0..<n { for k in 0..<p { c[i][j] += a[i][k] * b[k][j] } } }
        return c
    }

    private func solveCubic(a: Double, b: Double, c: Double, d: Double) -> [Complex] {
        let p: Double = (3*a*c - b*b) / (3*a*a)
        let q: Double = (2*b*b*b - 9*a*b*c + 27*a*a*d) / (27*a*a*a)
        let disc: Double = q*q/4 + p*p*p/27
        let shift: Double = -b / (3*a)
        if disc > 0 {
            let sqrtD: Double = Foundation.sqrt(disc)
            let u: Double = cbrt(-q/2 + sqrtD)
            let v: Double = cbrt(-q/2 - sqrtD)
            let r1: Double = u + v + shift
            let realPart: Double = -(u+v)/2 + shift
            let imagPart: Double = (u-v) * Foundation.sqrt(3.0) / 2
            return [Complex(r1, 0), Complex(realPart, imagPart), Complex(realPart, -imagPart)]
        } else {
            let r: Double = Foundation.sqrt(-p*p*p/27)
            let theta: Double = acos(-q / (2*r))
            let m: Double = 2 * cbrt(r)
            let r1: Double = m * cos(theta/3) + shift
            let r2: Double = m * cos((theta + 2 * .pi)/3) + shift
            let r3: Double = m * cos((theta + 4 * .pi)/3) + shift
            return [Complex(r1, 0), Complex(r2, 0), Complex(r3, 0)]
        }
    }

    private func cbrt(_ x: Double) -> Double { x >= 0 ? pow(x, 1.0/3.0) : -pow(-x, 1.0/3.0) }

    private func powerIteration(_ matrix: [[Double]], iterations: Int) -> [Complex] {
        let n = matrix.count
        var v = [Double](repeating: 1.0/Foundation.sqrt(Double(n)), count: n)
        var eigenvalue = 0.0
        for _ in 0..<iterations {
            var w = [Double](repeating: 0, count: n)
            for i in 0..<n { for j in 0..<n { w[i] += matrix[i][j] * v[j] } }
            let norm = Foundation.sqrt(w.map { $0*$0 }.reduce(0, +))
            guard norm > 1e-14 else { break }
            eigenvalue = w.enumerated().map { $0.element * v[$0.offset] }.reduce(0, +)
            v = w.map { $0 / norm }
        }
        return [Complex(eigenvalue, 0)]
    }

    private func erf(_ x: Double) -> Double {
        let t = 1.0 / (1.0 + 0.3275911 * abs(x))
        let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        let result = 1.0 - poly * exp(-x * x)
        return x >= 0 ? result : -result
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  📐 ADVANCED MATH ENGINE v29.0                            ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Capabilities:
        ║    • Symbolic Calculus (derivatives, integrals)
        ║    • Linear Algebra (det, inv, eigenvalues, SVD, rank)
        ║    • Number Theory (GCD, LCM, totient, factorization)
        ║    • Statistics (regression, distributions, descriptive)
        ║    • Differential Equations (RK4 ODE solver)
        ║    • Series & Sequences (Taylor, continued fractions)
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - 🌊 FLUID DYNAMICS & WAVE MECHANICS ENGINE
// Phase 41.0: Reynolds, Bernoulli, Poiseuille, Navier-Stokes,
// Doppler, Snell's law, diffraction, interference, standing waves
// ═══════════════════════════════════════════════════════════════════════════════

class FluidWaveEngine {
    static let shared = FluidWaveEngine()
    private var computations: Int = 0

    // ═══════════════════════════════════════════════════════════════
    // MARK: FLUID DYNAMICS
    // ═══════════════════════════════════════════════════════════════

    /// Reynolds number: Re = ρvL/μ — predicts laminar vs turbulent flow
    func reynoldsNumber(density: Double, velocity: Double, length: Double, viscosity: Double) -> (Re: Double, regime: String) {
        computations += 1
        let Re = density * velocity * length / viscosity
        let regime: String
        if Re < 2300 { regime = "Laminar" }
        else if Re < 4000 { regime = "Transitional" }
        else { regime = "Turbulent" }
        return (Re, regime)
    }

    /// Bernoulli's equation: P₁ + ½ρv₁² + ρgh₁ = P₂ + ½ρv₂² + ρgh₂
    /// Solves for unknown pressure given other quantities
    func bernoulli(p1: Double, v1: Double, h1: Double, v2: Double, h2: Double, density: Double, g: Double = 9.80665) -> Double {
        computations += 1
        // P₂ = P₁ + ½ρ(v₁² - v₂²) + ρg(h₁ - h₂)
        return p1 + 0.5 * density * (v1 * v1 - v2 * v2) + density * g * (h1 - h2)
    }

    /// Hagen-Poiseuille equation: Q = πr⁴ΔP / (8μL) — laminar pipe flow
    func poiseuille(radius: Double, pressureDrop: Double, viscosity: Double, length: Double) -> Double {
        computations += 1
        return .pi * pow(radius, 4) * pressureDrop / (8.0 * viscosity * length)
    }

    /// Drag force: F_D = ½ρv²C_D·A
    func dragForce(density: Double, velocity: Double, dragCoeff: Double, area: Double) -> Double {
        computations += 1
        return 0.5 * density * velocity * velocity * dragCoeff * area
    }

    /// Terminal velocity: v_t = √(2mg / (ρC_D·A))
    func terminalVelocity(mass: Double, gravity: Double = 9.80665, density: Double, dragCoeff: Double, area: Double) -> Double {
        computations += 1
        return Foundation.sqrt(2.0 * mass * gravity / (density * dragCoeff * area))
    }

    /// Stokes drag (low Re): F = 6πμrv
    func stokesDrag(viscosity: Double, radius: Double, velocity: Double) -> Double {
        computations += 1
        return 6.0 * .pi * viscosity * radius * velocity
    }

    /// Mach number: M = v/c_s
    func machNumber(velocity: Double, soundSpeed: Double) -> (mach: Double, regime: String) {
        computations += 1
        let M = velocity / soundSpeed
        let regime: String
        if M < 0.8 { regime = "Subsonic" }
        else if M < 1.2 { regime = "Transonic" }
        else if M < 5.0 { regime = "Supersonic" }
        else { regime = "Hypersonic" }
        return (M, regime)
    }

    /// Continuity equation: A₁v₁ = A₂v₂
    func continuity(a1: Double, v1: Double, a2: Double) -> Double {
        computations += 1
        return a1 * v1 / a2
    }

    /// Torricelli's theorem: v = √(2gh)
    func torricelliVelocity(height: Double, gravity: Double = 9.80665) -> Double {
        computations += 1
        return Foundation.sqrt(2.0 * gravity * height)
    }

    /// Navier-Stokes viscous stress term: τ = μ(∂u/∂y) — 1D shear stress
    func viscousShearStress(viscosity: Double, velocityGradient: Double) -> Double {
        computations += 1
        return viscosity * velocityGradient
    }

    /// Froude number: Fr = v / √(gL) — gravitational flow regime
    func froudeNumber(velocity: Double, gravity: Double = 9.80665, length: Double) -> (Fr: Double, regime: String) {
        computations += 1
        let Fr = velocity / Foundation.sqrt(gravity * length)
        let regime = Fr < 1.0 ? "Subcritical" : (Fr == 1.0 ? "Critical" : "Supercritical")
        return (Fr, regime)
    }

    /// Weber number: We = ρv²L/σ — inertial vs surface tension forces
    func weberNumber(density: Double, velocity: Double, length: Double, surfaceTension: Double) -> Double {
        computations += 1
        return density * velocity * velocity * length / surfaceTension
    }

    /// Euler number: Eu = ΔP / (½ρv²) — pressure losses in flow
    func eulerNumber(pressureDrop: Double, density: Double, velocity: Double) -> Double {
        computations += 1
        return pressureDrop / (0.5 * density * velocity * velocity)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: WAVE MECHANICS
    // ═══════════════════════════════════════════════════════════════

    /// Fundamental wave equation: v = fλ
    func waveSpeed(frequency: Double, wavelength: Double) -> Double {
        computations += 1
        return frequency * wavelength
    }

    /// Doppler effect (source moving): f' = f · v_s / (v_s ± v_source)
    func dopplerFrequency(sourceFreq: Double, soundSpeed: Double, sourceVelocity: Double, approaching: Bool) -> Double {
        computations += 1
        return sourceFreq * soundSpeed / (soundSpeed + (approaching ? -sourceVelocity : sourceVelocity))
    }

    /// Doppler effect (observer moving): f' = f · (v_s ± v_obs) / v_s
    func dopplerObserver(sourceFreq: Double, soundSpeed: Double, observerVelocity: Double, approaching: Bool) -> Double {
        computations += 1
        return sourceFreq * (soundSpeed + (approaching ? observerVelocity : -observerVelocity)) / soundSpeed
    }

    /// Standing wave on string: fₙ = n·v / (2L) — harmonic frequencies
    func standingWaveHarmonics(waveSpeed: Double, length: Double, harmonics: Int = 8) -> [(n: Int, freq: Double, wavelength: Double)] {
        computations += 1
        return (1...harmonics).map { n in
            let f = Double(n) * waveSpeed / (2.0 * length)
            let lambda = 2.0 * length / Double(n)
            return (n, f, lambda)
        }
    }

    /// Beat frequency: f_beat = |f₁ - f₂|
    func beatFrequency(f1: Double, f2: Double) -> Double {
        computations += 1
        return abs(f1 - f2)
    }

    /// Snell's law: n₁ sin(θ₁) = n₂ sin(θ₂)
    func snellsLaw(n1: Double, theta1: Double, n2: Double) -> Double? {
        computations += 1
        let sinTheta2 = n1 * sin(theta1) / n2
        guard abs(sinTheta2) <= 1.0 else { return nil } // Total internal reflection
        return asin(sinTheta2)
    }

    /// Critical angle for total internal reflection: θ_c = arcsin(n₂/n₁)
    func criticalAngle(n1: Double, n2: Double) -> Double? {
        computations += 1
        guard n1 > n2 else { return nil } // Only when going from denser to rarer medium
        return asin(n2 / n1)
    }

    /// Single-slit diffraction: minima at sin(θ) = mλ/a
    func diffractionMinima(slitWidth: Double, wavelength: Double, orders: Int = 5) -> [(order: Int, angle: Double)] {
        computations += 1
        return (1...orders).compactMap { m in
            let sinTheta = Double(m) * wavelength / slitWidth
            guard abs(sinTheta) <= 1.0 else { return nil }
            return (m, asin(sinTheta))
        }
    }

    /// Double-slit interference: maxima at d·sin(θ) = mλ
    func interferenceMaxima(slitSeparation: Double, wavelength: Double, orders: Int = 5) -> [(order: Int, angle: Double)] {
        computations += 1
        return (0...orders).compactMap { m in
            let sinTheta = Double(m) * wavelength / slitSeparation
            guard abs(sinTheta) <= 1.0 else { return nil }
            return (m, asin(sinTheta))
        }
    }

    /// Sound intensity level: β = 10·log₁₀(I/I₀) dB
    func soundIntensityLevel(intensity: Double, reference: Double = 1e-12) -> Double {
        computations += 1
        return 10.0 * log10(intensity / reference)
    }

    /// Inverse square law for wave intensity: I₂ = I₁(r₁/r₂)²
    func inverseSquareIntensity(intensity1: Double, r1: Double, r2: Double) -> Double {
        computations += 1
        return intensity1 * (r1 * r1) / (r2 * r2)
    }

    /// Wave superposition: compute resultant amplitude for two waves
    /// A = √(A₁² + A₂² + 2·A₁·A₂·cos(δ))
    func waveSuperposition(a1: Double, a2: Double, phaseDifference: Double) -> Double {
        computations += 1
        return Foundation.sqrt(a1 * a1 + a2 * a2 + 2.0 * a1 * a2 * cos(phaseDifference))
    }

    /// Group velocity vs phase velocity: v_g = v_p - λ(dv_p/dλ)
    /// Returns (phase velocity, group velocity) for a dispersive medium
    func groupVelocity(phaseVelocity: Double, wavelength: Double, dvdLambda: Double) -> (vPhase: Double, vGroup: Double) {
        computations += 1
        return (phaseVelocity, phaseVelocity - wavelength * dvdLambda)
    }

    /// Wave energy density: u = ½ρω²A²
    func waveEnergyDensity(density: Double, angularFreq: Double, amplitude: Double) -> Double {
        computations += 1
        return 0.5 * density * angularFreq * angularFreq * amplitude * amplitude
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🌊 FLUID DYNAMICS & WAVE MECHANICS v41.0                 ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Fluid Dynamics:
        ║    • Reynolds, Bernoulli, Poiseuille, Navier-Stokes
        ║    • Drag (form + Stokes), terminal velocity
        ║    • Mach, Froude, Weber, Euler numbers
        ║    • Continuity, Torricelli
        ║  Wave Mechanics:
        ║    • Doppler (source + observer), standing waves
        ║    • Snell's law, critical angle, diffraction
        ║    • Interference, beats, superposition
        ║    • Group/phase velocity, energy density
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - 📡 INFORMATION THEORY & SIGNAL PROCESSING ENGINE
// Phase 41.1: Shannon entropy, mutual information, channel capacity,
// KL divergence, DFT, convolution, autocorrelation, filtering
// ═══════════════════════════════════════════════════════════════════════════════

class InformationSignalEngine {
    static let shared = InformationSignalEngine()
    private var computations: Int = 0

    // ═══════════════════════════════════════════════════════════════
    // MARK: INFORMATION THEORY
    // ═══════════════════════════════════════════════════════════════

    /// Shannon entropy: H(X) = -Σ p(x)·log₂(p(x))
    func shannonEntropy(_ probabilities: [Double]) -> Double {
        computations += 1
        return -probabilities.reduce(0.0) { sum, p in
            p > 0 ? sum + p * log2(p) : sum
        }
    }

    /// Joint entropy: H(X,Y) = -Σ p(x,y)·log₂(p(x,y))
    func jointEntropy(_ jointProbs: [[Double]]) -> Double {
        computations += 1
        var H = 0.0
        for row in jointProbs {
            for p in row where p > 0 {
                H -= p * log2(p)
            }
        }
        return H
    }

    /// Conditional entropy: H(Y|X) = H(X,Y) - H(X)
    func conditionalEntropy(jointProbs: [[Double]], marginalX: [Double]) -> Double {
        computations += 1
        return jointEntropy(jointProbs) - shannonEntropy(marginalX)
    }

    /// Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    func mutualInformation(probsX: [Double], probsY: [Double], jointProbs: [[Double]]) -> Double {
        computations += 1
        return shannonEntropy(probsX) + shannonEntropy(probsY) - jointEntropy(jointProbs)
    }

    /// KL Divergence: D_KL(P‖Q) = Σ P(x)·ln(P(x)/Q(x))
    func klDivergence(p: [Double], q: [Double]) -> Double {
        computations += 1
        guard p.count == q.count else { return Double.infinity }
        var kl = 0.0
        for i in 0..<p.count {
            if p[i] > 0 && q[i] > 0 {
                kl += p[i] * log(p[i] / q[i])
            } else if p[i] > 0 && q[i] == 0 {
                return Double.infinity
            }
        }
        return kl
    }

    /// Jensen-Shannon Divergence: JSD(P‖Q) = ½D_KL(P‖M) + ½D_KL(Q‖M), M = ½(P+Q)
    func jsDivergence(p: [Double], q: [Double]) -> Double {
        computations += 1
        guard p.count == q.count else { return Double.infinity }
        let m = zip(p, q).map { ($0 + $1) / 2.0 }
        return 0.5 * klDivergence(p: p, q: m) + 0.5 * klDivergence(p: q, q: m)
    }

    /// Cross-entropy: H(P,Q) = -Σ P(x)·log₂(Q(x))
    func crossEntropy(p: [Double], q: [Double]) -> Double {
        computations += 1
        guard p.count == q.count else { return Double.infinity }
        var ce = 0.0
        for i in 0..<p.count where p[i] > 0 && q[i] > 0 {
            ce -= p[i] * log2(q[i])
        }
        return ce
    }

    /// Shannon channel capacity: C = B·log₂(1 + S/N) bits/sec
    func channelCapacity(bandwidth: Double, signalToNoise: Double) -> Double {
        computations += 1
        return bandwidth * log2(1.0 + signalToNoise)
    }

    /// Data compression bound: minimum bits = n·H(X)
    func compressionBound(symbolCount: Int, probabilities: [Double]) -> Double {
        computations += 1
        return Double(symbolCount) * shannonEntropy(probabilities)
    }

    /// Rényi entropy: H_α(X) = (1/(1-α))·log₂(Σ p(x)^α)
    func renyiEntropy(_ probabilities: [Double], alpha: Double) -> Double {
        computations += 1
        guard alpha != 1.0 else { return shannonEntropy(probabilities) }
        let sum = probabilities.reduce(0.0) { $0 + pow($1, alpha) }
        return log2(sum) / (1.0 - alpha)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: SIGNAL PROCESSING
    // ═══════════════════════════════════════════════════════════════

    /// Discrete Fourier Transform: X[k] = Σ x[n]·e^(-j2πkn/N)
    func dft(_ signal: [Double]) -> [Complex] {
        computations += 1
        let N = signal.count
        var result = [Complex](repeating: Complex.zero, count: N)
        for k in 0..<N {
            var sum = Complex.zero
            for n in 0..<N {
                let angle = -2.0 * .pi * Double(k) * Double(n) / Double(N)
                sum = sum + Complex(signal[n], 0) * Complex.euler(angle)
            }
            result[k] = sum
        }
        return result
    }

    /// Inverse DFT: x[n] = (1/N)·Σ X[k]·e^(j2πkn/N)
    func idft(_ spectrum: [Complex]) -> [Double] {
        computations += 1
        let N = spectrum.count
        var result = [Double](repeating: 0, count: N)
        for n in 0..<N {
            var sum = Complex.zero
            for k in 0..<N {
                let angle = 2.0 * .pi * Double(k) * Double(n) / Double(N)
                sum = sum + spectrum[k] * Complex.euler(angle)
            }
            result[n] = sum.real / Double(N)
        }
        return result
    }

    /// Power spectral density: |X[k]|² / N
    func powerSpectrum(_ signal: [Double]) -> [Double] {
        computations += 1
        let spectrum = dft(signal)
        let n = Double(signal.count)
        return spectrum.map { $0.magnitude * $0.magnitude / n }
    }

    /// Linear convolution: (f * g)[n] = Σ f[m]·g[n-m]
    func convolve(_ f: [Double], _ g: [Double]) -> [Double] {
        computations += 1
        let outLen = f.count + g.count - 1
        var result = [Double](repeating: 0, count: outLen)
        for i in 0..<f.count {
            for j in 0..<g.count {
                result[i + j] += f[i] * g[j]
            }
        }
        return result
    }

    /// Cross-correlation: (f ⋆ g)[n] = Σ f[m]·g[m+n]
    func crossCorrelation(_ f: [Double], _ g: [Double]) -> [Double] {
        computations += 1
        let outLen = f.count + g.count - 1
        var result = [Double](repeating: 0, count: outLen)
        for lag in 0..<outLen {
            let shift = lag - g.count + 1
            for m in 0..<f.count {
                let gIdx = m - shift
                if gIdx >= 0 && gIdx < g.count {
                    result[lag] += f[m] * g[gIdx]
                }
            }
        }
        return result
    }

    /// Autocorrelation: R[τ] = Σ x[n]·x[n+τ]
    func autocorrelation(_ signal: [Double]) -> [Double] {
        return crossCorrelation(signal, signal)
    }

    /// Moving average filter: y[n] = (1/M)·Σ x[n-k] for k=0..M-1
    func movingAverage(_ signal: [Double], windowSize: Int) -> [Double] {
        computations += 1
        guard windowSize > 0, signal.count >= windowSize else { return signal }
        var result = [Double]()
        var sum = signal[0..<windowSize].reduce(0, +)
        result.append(sum / Double(windowSize))
        for i in windowSize..<signal.count {
            sum += signal[i] - signal[i - windowSize]
            result.append(sum / Double(windowSize))
        }
        return result
    }

    /// Exponential moving average: y[n] = α·x[n] + (1-α)·y[n-1]
    func exponentialMovingAverage(_ signal: [Double], alpha: Double) -> [Double] {
        computations += 1
        guard !signal.isEmpty else { return [] }
        var result = [signal[0]]
        for i in 1..<signal.count {
            result.append(alpha * signal[i] + (1.0 - alpha) * result[i - 1])
        }
        return result
    }

    /// Nyquist frequency: f_N = f_s / 2
    func nyquistFrequency(sampleRate: Double) -> Double {
        computations += 1
        return sampleRate / 2.0
    }

    /// Signal-to-noise ratio in dB: SNR = 10·log₁₀(P_signal / P_noise)
    func snrDB(signalPower: Double, noisePower: Double) -> Double {
        computations += 1
        return 10.0 * log10(signalPower / noisePower)
    }

    /// Window functions for spectral analysis
    func hanningWindow(_ n: Int) -> [Double] {
        computations += 1
        var result = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let x: Double = Double(i) / Double(n - 1)
            result[i] = 0.5 * (1.0 - cos(2.0 * .pi * x))
        }
        return result
    }

    func hammingWindow(_ n: Int) -> [Double] {
        computations += 1
        var result = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let x: Double = Double(i) / Double(n - 1)
            result[i] = 0.54 - 0.46 * cos(2.0 * .pi * x)
        }
        return result
    }

    func blackmanWindow(_ n: Int) -> [Double] {
        computations += 1
        var result = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let x: Double = Double(i) / Double(n - 1)
            result[i] = 0.42 - 0.5 * cos(2.0 * .pi * x) + 0.08 * cos(4.0 * .pi * x)
        }
        return result
    }

    /// Generate test signal: sum of sinusoids
    func generateSignal(frequencies: [Double], amplitudes: [Double], sampleRate: Double, duration: Double) -> [Double] {
        computations += 1
        let N = Int(sampleRate * duration)
        var result = [Double](repeating: 0, count: N)
        for n in 0..<N {
            let t: Double = Double(n) / sampleRate
            var sample: Double = 0
            for j in 0..<min(frequencies.count, amplitudes.count) {
                sample += amplitudes[j] * sin(2.0 * .pi * frequencies[j] * t)
            }
            result[n] = sample
        }
        return result
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  📡 INFORMATION THEORY & SIGNAL PROCESSING v41.1          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Information Theory:
        ║    • Shannon, joint, conditional, Rényi entropy
        ║    • Mutual information, KL & JS divergence
        ║    • Cross-entropy, channel capacity, compression
        ║  Signal Processing:
        ║    • DFT/IDFT, power spectrum
        ║    • Convolution, cross/auto-correlation
        ║    • Moving avg, EMA, window functions
        ║    • SNR, Nyquist, signal generation
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
