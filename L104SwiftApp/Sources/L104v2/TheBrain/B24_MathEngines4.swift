// ═══════════════════════════════════════════════════════════════
// B24_MathEngines4.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 · TheBrain · v2 Architecture
//
// Extracted from L104Native.swift lines 16376-17766
// Classes: CryptographicMathEngine, FinancialMathEngine, HighSciencesEngine
// ═══════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class CryptographicMathEngine {
    static let shared = CryptographicMathEngine()
    private var computations: Int = 0

    // ═══ Modular Arithmetic (delegates to AdvancedMathEngine to avoid duplication) ═══
    private let _math = AdvancedMathEngine.shared

    /// Modular exponentiation: (base^exp) mod m — using fast binary method
    func modPow(base: Int, exponent: Int, modulus: Int) -> Int {
        computations += 1
        return _math.modPow(base, exponent, modulus)
    }

    /// Extended Euclidean Algorithm: returns (gcd, x, y) where ax + by = gcd(a,b)
    func extendedGCD(_ a: Int, _ b: Int) -> (gcd: Int, x: Int, y: Int) {
        computations += 1
        return _math.extendedGCD(a, b)
    }

    /// Modular inverse: a^(-1) mod m — returns nil if no inverse exists
    func modInverse(_ a: Int, _ m: Int) -> Int? {
        computations += 1
        return _math.modInverse(a, m)
    }

    /// Chinese Remainder Theorem for 2 congruences: x ≡ a1 (mod m1), x ≡ a2 (mod m2)
    func chineseRemainder(a1: Int, m1: Int, a2: Int, m2: Int) -> Int? {
        computations += 1
        guard let inv = modInverse(m1, m2) else { return nil }
        let M = m1 * m2
        let x = (a1 + m1 * ((a2 - a1) * inv % m2 + m2)) % M
        return (x + M) % M
    }

    /// Euler's totient function φ(n) — count of integers coprime to n
    func eulerTotient(_ n: Int) -> Int {
        computations += 1
        return _math.eulerTotient(n)
    }

    /// Discrete logarithm (baby-step giant-step): find x where g^x ≡ h (mod p)
    func discreteLog(g: Int, h: Int, p: Int) -> Int? {
        computations += 1
        let m = Int(Foundation.ceil(Foundation.sqrt(Double(p))))
        var table: [Int: Int] = [:]

        // Baby steps: g^j mod p for j = 0..m-1
        var power = 1
        for j in 0..<m {
            table[power] = j
            power = power * g % p
        }

        // Giant step factor: g^(-m) mod p
        guard let gInvM = modInverse(modPow(base: g, exponent: m, modulus: p), p) else { return nil }

        // Giant steps
        var gamma = h
        for i in 0..<m {
            if let j = table[gamma] {
                return i * m + j
            }
            gamma = gamma * gInvM % p
        }
        return nil
    }

    // ═══ Primality Testing ═══

    /// Miller-Rabin primality test
    func millerRabin(_ n: Int, witnesses: [Int] = [2, 3, 5, 7, 11, 13]) -> Bool {
        computations += 1
        if n < 2 { return false }
        if n < 4 { return true }
        if n % 2 == 0 { return false }

        // Write n-1 as 2^r * d
        var d = n - 1
        var r = 0
        while d % 2 == 0 {
            d /= 2
            r += 1
        }

        for a in witnesses {
            if a >= n { continue }
            var x = modPow(base: a, exponent: d, modulus: n)
            if x == 1 || x == n - 1 { continue }
            var composite = true
            for _ in 0..<(r - 1) {
                x = x * x % n
                if x == n - 1 { composite = false; break }
            }
            if composite { return false }
        }
        return true
    }

    /// Fermat primality test: a^(p-1) ≡ 1 (mod p) for random bases
    func fermatTest(_ n: Int, bases: [Int] = [2, 3, 5, 7]) -> Bool {
        computations += 1
        if n < 2 { return false }
        for a in bases {
            if a >= n { continue }
            if modPow(base: a, exponent: n - 1, modulus: n) != 1 { return false }
        }
        return true
    }

    // ═══ RSA ═══

    /// Generate RSA parameters from two primes p, q
    func rsaKeyGen(p: Int, q: Int, e: Int = 65537) -> (n: Int, e: Int, d: Int, totient: Int)? {
        computations += 1
        let n = p * q
        let totient = (p - 1) * (q - 1)
        guard let d = modInverse(e, totient) else { return nil }
        return (n, e, d, totient)
    }

    /// RSA encrypt: c = m^e mod n
    func rsaEncrypt(message: Int, e: Int, n: Int) -> Int {
        computations += 1
        return modPow(base: message, exponent: e, modulus: n)
    }

    /// RSA decrypt: m = c^d mod n
    func rsaDecrypt(ciphertext: Int, d: Int, n: Int) -> Int {
        computations += 1
        return modPow(base: ciphertext, exponent: d, modulus: n)
    }

    // ═══ Diffie-Hellman ═══

    /// Diffie-Hellman shared secret: s = B^a mod p (or A^b mod p)
    func diffieHellmanShared(publicKey: Int, privateKey: Int, prime: Int) -> Int {
        computations += 1
        return modPow(base: publicKey, exponent: privateKey, modulus: prime)
    }

    /// Diffie-Hellman public key: A = g^a mod p
    func diffieHellmanPublic(generator: Int, privateKey: Int, prime: Int) -> Int {
        computations += 1
        return modPow(base: generator, exponent: privateKey, modulus: prime)
    }

    // ═══ Elliptic Curve Math (over reals for education) ═══

    /// Point on secp256k1: y² = x³ + 7
    func secp256k1Check(x: Double, y: Double) -> Bool {
        computations += 1
        let lhs = y * y
        let rhs = x * x * x + 7
        return abs(lhs - rhs) < 1e-6
    }

    /// Elliptic curve point addition (real field, y² = x³ + ax + b)
    func ecAdd(x1: Double, y1: Double, x2: Double, y2: Double, a: Double) -> (x: Double, y: Double) {
        computations += 1
        let lambda: Double
        if abs(x1 - x2) < 1e-15 && abs(y1 - y2) < 1e-15 {
            // Point doubling
            guard abs(2 * y1) > 1e-15 else { return (.infinity, .infinity) }
            lambda = (3 * x1 * x1 + a) / (2 * y1)
        } else {
            // Point addition
            guard abs(x2 - x1) > 1e-15 else { return (.infinity, .infinity) }
            lambda = (y2 - y1) / (x2 - x1)
        }
        let x3 = lambda * lambda - x1 - x2
        let y3 = lambda * (x1 - x3) - y1
        return (x3, y3)
    }

    /// Elliptic curve scalar multiplication (double-and-add)
    func ecMultiply(x: Double, y: Double, k: Int, a: Double) -> (x: Double, y: Double) {
        computations += 1
        guard k > 0 else { return (.infinity, .infinity) }
        if k == 1 { return (x, y) }
        var result = (x: x, y: y)
        var temp = (x: x, y: y)
        var n = k - 1
        while n > 0 {
            if n % 2 == 1 {
                result = ecAdd(x1: result.x, y1: result.y, x2: temp.x, y2: temp.y, a: a)
            }
            temp = ecAdd(x1: temp.x, y1: temp.y, x2: temp.x, y2: temp.y, a: a)
            n /= 2
        }
        return result
    }

    // ═══ Hash / Information Security ═══

    /// Key space size: 2^n for n-bit key
    func keySpaceSize(bits: Int) -> Double {
        computations += 1
        return Foundation.pow(2.0, Double(bits))
    }

    /// Birthday attack bound: ~√(2^n) = 2^(n/2) for n-bit hash
    func birthdayBound(bits: Int) -> Double {
        computations += 1
        return Foundation.pow(2.0, Double(bits) / 2.0)
    }

    /// Information entropy of a password: H = log2(C^L) where C = charset size, L = length
    func passwordEntropy(charsetSize: Int, length: Int) -> Double {
        computations += 1
        return Double(length) * Foundation.log2(Double(charsetSize))
    }

    /// Primitive root check: g is primitive root mod p if g has order p-1
    func isPrimitiveRoot(g: Int, p: Int) -> Bool {
        computations += 1
        let phi = p - 1
        // Factor phi and check g^(phi/q) ≠ 1 mod p for each prime factor q
        var n = phi
        var factors: [Int] = []
        var d = 2
        while d * d <= n {
            if n % d == 0 {
                factors.append(d)
                while n % d == 0 { n /= d }
            }
            d += 1
        }
        if n > 1 { factors.append(n) }

        for q in factors {
            if modPow(base: g, exponent: phi / q, modulus: p) == 1 { return false }
        }
        return true
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🔐 CRYPTOGRAPHIC MATHEMATICS ENGINE v43.1               ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Modular Arithmetic:
        ║    • ModPow, ExtGCD, ModInverse, CRT
        ║    • Euler totient, discrete log (BSGS)
        ║  Primality Testing:
        ║    • Miller-Rabin, Fermat test
        ║  RSA:
        ║    • Key generation, encrypt, decrypt
        ║  Key Exchange:
        ║    • Diffie-Hellman (public key, shared secret)
        ║  Elliptic Curves:
        ║    • secp256k1 check, point add, scalar multiply
        ║  Information Security:
        ║    • Key space, birthday bound, password entropy
        ║    • Primitive root check
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - 💰 FINANCIAL MATHEMATICS ENGINE
// Phase 43.2: Black-Scholes, Greeks, bond pricing, yield curves, portfolio theory,
// risk metrics, time value of money, amortization, actuarial science
// ═══════════════════════════════════════════════════════════════════════════════

class FinancialMathEngine {
    static let shared = FinancialMathEngine()
    private var computations: Int = 0

    // ═══ Standard Normal Distribution ═══

    /// Cumulative standard normal distribution Φ(x) — Abramowitz & Stegun approximation
    private func normalCDF(_ x: Double) -> Double {
        let a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741
        let a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911
        let sign = x >= 0 ? 1.0 : -1.0
        let absX = abs(x)
        let t = 1.0 / (1.0 + p * absX)
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Foundation.exp(-absX * absX / 2.0)
        return 0.5 * (1.0 + sign * y)
    }

    /// Standard normal PDF: φ(x) = e^(-x²/2) / √(2π)
    private func normalPDF(_ x: Double) -> Double {
        return Foundation.exp(-x * x / 2.0) / Foundation.sqrt(2.0 * .pi)
    }

    // ═══ Black-Scholes Option Pricing ═══

    /// d1 and d2 parameters
    private func bsD1D2(S: Double, K: Double, r: Double, sigma: Double, T: Double) -> (d1: Double, d2: Double) {
        let d1 = (Foundation.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * Foundation.sqrt(T))
        let d2 = d1 - sigma * Foundation.sqrt(T)
        return (d1, d2)
    }

    /// Black-Scholes European call price
    func blackScholesCall(S: Double, K: Double, r: Double, sigma: Double, T: Double) -> Double {
        computations += 1
        let (d1, d2) = bsD1D2(S: S, K: K, r: r, sigma: sigma, T: T)
        return S * normalCDF(d1) - K * Foundation.exp(-r * T) * normalCDF(d2)
    }

    /// Black-Scholes European put price
    func blackScholesPut(S: Double, K: Double, r: Double, sigma: Double, T: Double) -> Double {
        computations += 1
        let (d1, d2) = bsD1D2(S: S, K: K, r: r, sigma: sigma, T: T)
        return K * Foundation.exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1)
    }

    /// Put-call parity: C - P = S - K*e^(-rT)
    func putCallParity(S: Double, K: Double, r: Double, T: Double) -> Double {
        computations += 1
        return S - K * Foundation.exp(-r * T)
    }

    // ═══ Greeks ═══

    /// Delta: ∂C/∂S
    func delta(S: Double, K: Double, r: Double, sigma: Double, T: Double, isCall: Bool = true) -> Double {
        computations += 1
        let (d1, _) = bsD1D2(S: S, K: K, r: r, sigma: sigma, T: T)
        return isCall ? normalCDF(d1) : normalCDF(d1) - 1.0
    }

    /// Gamma: ∂²C/∂S²
    func gamma(S: Double, K: Double, r: Double, sigma: Double, T: Double) -> Double {
        computations += 1
        let (d1, _) = bsD1D2(S: S, K: K, r: r, sigma: sigma, T: T)
        return normalPDF(d1) / (S * sigma * Foundation.sqrt(T))
    }

    /// Theta: ∂C/∂t (time decay, per year)
    func theta(S: Double, K: Double, r: Double, sigma: Double, T: Double, isCall: Bool = true) -> Double {
        computations += 1
        let (d1, d2) = bsD1D2(S: S, K: K, r: r, sigma: sigma, T: T)
        let term1 = -S * normalPDF(d1) * sigma / (2 * Foundation.sqrt(T))
        if isCall {
            return term1 - r * K * Foundation.exp(-r * T) * normalCDF(d2)
        } else {
            return term1 + r * K * Foundation.exp(-r * T) * normalCDF(-d2)
        }
    }

    /// Vega: ∂C/∂σ
    func vega(S: Double, K: Double, r: Double, sigma: Double, T: Double) -> Double {
        computations += 1
        let (d1, _) = bsD1D2(S: S, K: K, r: r, sigma: sigma, T: T)
        return S * normalPDF(d1) * Foundation.sqrt(T)
    }

    /// Rho: ∂C/∂r
    func rho(S: Double, K: Double, r: Double, sigma: Double, T: Double, isCall: Bool = true) -> Double {
        computations += 1
        let (_, d2) = bsD1D2(S: S, K: K, r: r, sigma: sigma, T: T)
        if isCall {
            return K * T * Foundation.exp(-r * T) * normalCDF(d2)
        } else {
            return -K * T * Foundation.exp(-r * T) * normalCDF(-d2)
        }
    }

    // ═══ Implied Volatility ═══

    /// Newton-Raphson implied volatility from market call price
    func impliedVolatility(S: Double, K: Double, r: Double, T: Double, marketPrice: Double, maxIter: Int = 100) -> Double {
        computations += 1
        var sigma = 0.3  // initial guess
        for _ in 0..<maxIter {
            let price = blackScholesCall(S: S, K: K, r: r, sigma: sigma, T: T)
            let v = vega(S: S, K: K, r: r, sigma: sigma, T: T)
            guard abs(v) > 1e-10 else { break }
            let diff = price - marketPrice
            sigma -= diff / v
            if abs(diff) < 1e-8 { break }
            sigma = max(0.001, min(sigma, 5.0))
        }
        return sigma
    }

    // ═══ Time Value of Money ═══

    /// Future value: FV = PV * (1 + r)^n
    func futureValue(pv: Double, rate: Double, periods: Double) -> Double {
        computations += 1
        return pv * Foundation.pow(1 + rate, periods)
    }

    /// Present value: PV = FV / (1 + r)^n
    func presentValue(fv: Double, rate: Double, periods: Double) -> Double {
        computations += 1
        return fv / Foundation.pow(1 + rate, periods)
    }

    /// Continuous compounding: FV = PV * e^(rt)
    func continuousCompounding(pv: Double, rate: Double, time: Double) -> Double {
        computations += 1
        return pv * Foundation.exp(rate * time)
    }

    /// Annuity present value: PV = PMT * [1 - (1+r)^(-n)] / r
    func annuityPV(payment: Double, rate: Double, periods: Int) -> Double {
        computations += 1
        guard abs(rate) > 1e-15 else { return payment * Double(periods) }
        return payment * (1 - Foundation.pow(1 + rate, Double(-periods))) / rate
    }

    /// Annuity future value: FV = PMT * [(1+r)^n - 1] / r
    func annuityFV(payment: Double, rate: Double, periods: Int) -> Double {
        computations += 1
        guard abs(rate) > 1e-15 else { return payment * Double(periods) }
        return payment * (Foundation.pow(1 + rate, Double(periods)) - 1) / rate
    }

    /// Loan amortization monthly payment: M = P * r(1+r)^n / [(1+r)^n - 1]
    func monthlyPayment(principal: Double, annualRate: Double, years: Int) -> Double {
        computations += 1
        let r = annualRate / 12.0
        let n = Double(years * 12)
        guard abs(r) > 1e-15 else { return principal / n }
        let factor = Foundation.pow(1 + r, n)
        return principal * r * factor / (factor - 1)
    }

    // ═══ Bond Pricing ═══

    /// Bond price: P = Σ C/(1+y)^t + F/(1+y)^n
    func bondPrice(faceValue: Double, couponRate: Double, yield: Double, periods: Int) -> Double {
        computations += 1
        let coupon = faceValue * couponRate
        var price = 0.0
        for t in 1...periods {
            price += coupon / Foundation.pow(1 + yield, Double(t))
        }
        price += faceValue / Foundation.pow(1 + yield, Double(periods))
        return price
    }

    /// Macaulay duration: D = [Σ t*CF_t/(1+y)^t] / P
    func macaulayDuration(faceValue: Double, couponRate: Double, yield: Double, periods: Int) -> Double {
        computations += 1
        let coupon = faceValue * couponRate
        let price = bondPrice(faceValue: faceValue, couponRate: couponRate, yield: yield, periods: periods)
        var weighted = 0.0
        for t in 1...periods {
            var cf = coupon
            if t == periods { cf += faceValue }
            weighted += Double(t) * cf / Foundation.pow(1 + yield, Double(t))
        }
        return weighted / price
    }

    /// Modified duration: Dmod = D / (1 + y)
    func modifiedDuration(faceValue: Double, couponRate: Double, yield: Double, periods: Int) -> Double {
        computations += 1
        return macaulayDuration(faceValue: faceValue, couponRate: couponRate, yield: yield, periods: periods) / (1 + yield)
    }

    /// Yield to maturity (Newton-Raphson approximation)
    func yieldToMaturity(faceValue: Double, couponRate: Double, price: Double, periods: Int, maxIter: Int = 100) -> Double {
        computations += 1
        var y = couponRate  // initial guess
        let coupon = faceValue * couponRate
        for _ in 0..<maxIter {
            var pCalc = 0.0
            var dP = 0.0
            for t in 1...periods {
                let disc = Foundation.pow(1 + y, Double(t))
                var cf = coupon
                if t == periods { cf += faceValue }
                pCalc += cf / disc
                dP -= Double(t) * cf / Foundation.pow(1 + y, Double(t + 1))
            }
            let diff = pCalc - price
            guard abs(dP) > 1e-15 else { break }
            y -= diff / dP
            if abs(diff) < 1e-8 { break }
        }
        return y
    }

    // ═══ Portfolio Theory ═══

    /// Portfolio return: Rp = Σ wi * Ri
    func portfolioReturn(weights: [Double], returns: [Double]) -> Double {
        computations += 1
        var r = 0.0
        for i in 0..<min(weights.count, returns.count) {
            r += weights[i] * returns[i]
        }
        return r
    }

    /// Portfolio variance (2-asset): σ²p = w1²σ1² + w2²σ2² + 2w1w2ρσ1σ2
    func portfolioVariance2(w1: Double, w2: Double, sigma1: Double, sigma2: Double, rho: Double) -> Double {
        computations += 1
        return w1*w1*sigma1*sigma1 + w2*w2*sigma2*sigma2 + 2*w1*w2*rho*sigma1*sigma2
    }

    /// Sharpe ratio: (Rp - Rf) / σp
    func sharpeRatio(portfolioReturn: Double, riskFreeRate: Double, stdDev: Double) -> Double {
        computations += 1
        guard abs(stdDev) > 1e-15 else { return 0 }
        return (portfolioReturn - riskFreeRate) / stdDev
    }

    /// Capital Asset Pricing Model: E(Ri) = Rf + βi(E(Rm) - Rf)
    func capm(riskFreeRate: Double, beta: Double, marketReturn: Double) -> Double {
        computations += 1
        return riskFreeRate + beta * (marketReturn - riskFreeRate)
    }

    /// Beta coefficient: β = Cov(Ri, Rm) / Var(Rm)
    func betaCoefficient(assetReturns: [Double], marketReturns: [Double]) -> Double {
        computations += 1
        let n = min(assetReturns.count, marketReturns.count)
        guard n > 1 else { return 0 }
        let meanA = assetReturns.prefix(n).reduce(0, +) / Double(n)
        let meanM = marketReturns.prefix(n).reduce(0, +) / Double(n)
        var cov = 0.0, varM = 0.0
        for i in 0..<n {
            cov += (assetReturns[i] - meanA) * (marketReturns[i] - meanM)
            varM += (marketReturns[i] - meanM) * (marketReturns[i] - meanM)
        }
        guard abs(varM) > 1e-15 else { return 0 }
        return cov / varM
    }

    // ═══ Risk Metrics ═══

    /// Value at Risk (parametric): VaR = μ - z*σ
    func valueAtRisk(mean: Double, stdDev: Double, confidence: Double = 0.95) -> Double {
        computations += 1
        // z-score for confidence level (approximate)
        let z: Double
        switch confidence {
        case 0.99: z = 2.326
        case 0.975: z = 1.960
        case 0.95: z = 1.645
        case 0.90: z = 1.282
        default: z = 1.645
        }
        return mean - z * stdDev
    }

    /// Expected Shortfall (CVaR): ES = μ - σ * φ(z_α) / (1-α)
    func expectedShortfall(mean: Double, stdDev: Double, confidence: Double = 0.95) -> Double {
        computations += 1
        let z: Double
        switch confidence {
        case 0.99: z = 2.326
        case 0.95: z = 1.645
        default: z = 1.645
        }
        let phi = normalPDF(z)
        return mean - stdDev * phi / (1 - confidence)
    }

    /// Maximum drawdown from a price series
    func maxDrawdown(prices: [Double]) -> Double {
        computations += 1
        guard prices.count > 1 else { return 0 }
        var peak = prices[0]
        var maxDD = 0.0
        for price in prices {
            if price > peak { peak = price }
            let dd = (peak - price) / peak
            if dd > maxDD { maxDD = dd }
        }
        return maxDD
    }

    /// Sortino ratio: (Rp - Rf) / downside deviation
    func sortinoRatio(returns: [Double], riskFreeRate: Double) -> Double {
        computations += 1
        let meanR = returns.reduce(0, +) / Double(returns.count)
        var sumSqDown = 0.0
        var countDown = 0
        for r in returns {
            if r < riskFreeRate {
                let d = r - riskFreeRate
                sumSqDown += d * d
                countDown += 1
            }
        }
        guard countDown > 0 else { return .infinity }
        let downsideDev = Foundation.sqrt(sumSqDown / Double(countDown))
        guard abs(downsideDev) > 1e-15 else { return .infinity }
        return (meanR - riskFreeRate) / downsideDev
    }

    // ═══ Actuarial ═══

    /// Compound interest growth factor
    func compoundGrowth(rate: Double, periods: Int) -> Double {
        computations += 1
        return Foundation.pow(1 + rate, Double(periods))
    }

    /// Rule of 72: years to double = 72 / (rate * 100)
    func ruleOf72(rate: Double) -> Double {
        computations += 1
        guard abs(rate) > 1e-15 else { return .infinity }
        return 72.0 / (rate * 100.0)
    }

    /// Effective annual rate from nominal: EAR = (1 + r/n)^n - 1
    func effectiveAnnualRate(nominal: Double, compoundsPerYear: Int) -> Double {
        computations += 1
        return Foundation.pow(1 + nominal / Double(compoundsPerYear), Double(compoundsPerYear)) - 1
    }

    /// Gordon Growth Model: P = D1 / (r - g)
    func gordonGrowth(dividend: Double, requiredReturn: Double, growthRate: Double) -> Double {
        computations += 1
        guard abs(requiredReturn - growthRate) > 1e-15 else { return .infinity }
        return dividend / (requiredReturn - growthRate)
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  💰 FINANCIAL MATHEMATICS ENGINE v43.2                   ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Options Pricing:
        ║    • Black-Scholes (call, put), put-call parity
        ║    • Implied volatility (Newton-Raphson)
        ║  Greeks:
        ║    • Delta, Gamma, Theta, Vega, Rho
        ║  Time Value of Money:
        ║    • FV, PV, continuous compounding
        ║    • Annuity PV/FV, loan amortization
        ║  Bonds:
        ║    • Bond pricing, Macaulay/modified duration
        ║    • Yield to maturity
        ║  Portfolio Theory:
        ║    • CAPM, Sharpe/Sortino ratio, beta
        ║    • Portfolio return & variance
        ║    • Gordon Growth Model
        ║  Risk Metrics:
        ║    • Value at Risk, Expected Shortfall (CVaR)
        ║    • Max drawdown
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - 🔬 HIGH SCIENCES COMPUTATION ENGINE
// Phase 29.0: Quantum mechanics, thermodynamics, relativity, particle physics,
// astrophysics, chemistry, electromagnetism, statistical mechanics
// ═══════════════════════════════════════════════════════════════════════════════

class HighSciencesEngine {
    static let shared = HighSciencesEngine()

    private var computations: Int = 0

    // ═══ FUNDAMENTAL CONSTANTS (CODATA 2018) ═══
    struct Constants {
        static let c = 299_792_458.0              // Speed of light m/s
        static let h = 6.62607015e-34             // Planck constant J·s
        static let hbar = 1.054571817e-34         // Reduced Planck constant
        static let G = 6.67430e-11                // Gravitational constant
        static let kB = 1.380649e-23              // Boltzmann constant J/K
        static let NA = 6.02214076e23             // Avogadro's number
        static let e = 1.602176634e-19            // Elementary charge C
        static let me = 9.1093837015e-31          // Electron mass kg
        static let mp = 1.67262192369e-27         // Proton mass kg
        static let mn = 1.67492749804e-27         // Neutron mass kg
        static let eps0 = 8.8541878128e-12        // Vacuum permittivity F/m
        static let mu0 = 1.25663706212e-6         // Vacuum permeability H/m
        static let sigma = 5.670374419e-8         // Stefan-Boltzmann constant
        static let R = 8.314462618                // Gas constant J/(mol·K)
        static let alpha = 7.2973525693e-3        // Fine-structure constant
        static let Ry = 13.605693122994           // Rydberg energy eV
        static let a0 = 5.29177210903e-11         // Bohr radius m
        static let mSun = 1.989e30                // Solar mass kg
        static let rSun = 6.957e8                 // Solar radius m
        static let LSun = 3.828e26                // Solar luminosity W
        static let AU = 1.496e11                  // Astronomical unit m
        static let pc = 3.086e16                  // Parsec m
        static let ly = 9.461e15                  // Light-year m
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: QUANTUM MECHANICS
    // ═══════════════════════════════════════════════════════════════

    /// Energy levels of hydrogen atom: E_n = -13.6 eV / n²
    func hydrogenEnergy(n: Int) -> Double {
        computations += 1
        return -Constants.Ry / Double(n * n)
    }

    /// Wavelength of photon from hydrogen transition (Rydberg formula)
    func hydrogenTransitionWavelength(nUpper: Int, nLower: Int) -> Double {
        computations += 1
        let R_inf = 1.0973731568160e7  // Rydberg constant m⁻¹
        let invLambda = R_inf * (1.0/Double(nLower*nLower) - 1.0/Double(nUpper*nUpper))
        return 1.0 / invLambda
    }

    /// de Broglie wavelength: λ = h / (m·v)
    func deBroglieWavelength(mass: Double, velocity: Double) -> Double {
        computations += 1
        return Constants.h / (mass * velocity)
    }

    /// Heisenberg uncertainty: Δx · Δp ≥ ħ/2
    func heisenbergUncertainty(deltaX: Double? = nil, deltaP: Double? = nil) -> (deltaX: Double, deltaP: Double) {
        computations += 1
        if let dx = deltaX {
            return (dx, Constants.hbar / (2.0 * dx))
        } else if let dp = deltaP {
            return (Constants.hbar / (2.0 * dp), dp)
        }
        let half = Foundation.sqrt(Constants.hbar / 2.0)
        return (half, half)
    }

    /// Particle in a box energy levels: E_n = n²π²ħ²/(2mL²)
    func particleInBoxEnergy(n: Int, mass: Double, length: Double) -> Double {
        computations += 1
        let n2 = Double(n * n)
        return n2 * .pi * .pi * Constants.hbar * Constants.hbar / (2.0 * mass * length * length)
    }

    /// Quantum harmonic oscillator energy: E_n = (n + 1/2)ħω
    func harmonicOscillatorEnergy(n: Int, omega: Double) -> Double {
        computations += 1
        return (Double(n) + 0.5) * Constants.hbar * omega
    }

    /// Tunneling probability through a barrier (rectangular)
    func tunnelingProbability(energy: Double, barrierHeight: Double, barrierWidth: Double, mass: Double) -> Double {
        computations += 1
        guard barrierHeight > energy else { return 1.0 }
        let kappa = Foundation.sqrt(2.0 * mass * (barrierHeight - energy)) / Constants.hbar
        return exp(-2.0 * kappa * barrierWidth)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: THERMODYNAMICS & STATISTICAL MECHANICS
    // ═══════════════════════════════════════════════════════════════

    /// Ideal gas: PV = nRT
    func idealGas(pressure: Double? = nil, volume: Double? = nil, moles: Double? = nil, temperature: Double? = nil) -> [String: Double] {
        computations += 1
        if let V = volume, let n = moles, let T = temperature { return ["pressure": n * Constants.R * T / V] }
        if let P = pressure, let n = moles, let T = temperature { return ["volume": n * Constants.R * T / P] }
        if let P = pressure, let V = volume, let T = temperature { return ["moles": P * V / (Constants.R * T)] }
        if let P = pressure, let V = volume, let n = moles { return ["temperature": P * V / (n * Constants.R)] }
        return [:]
    }

    /// Entropy change: ΔS = Q_rev / T
    func entropyChange(heat: Double, temperature: Double) -> Double {
        computations += 1
        return heat / temperature
    }

    /// Carnot efficiency: η = 1 - T_cold/T_hot
    func carnotEfficiency(tHot: Double, tCold: Double) -> Double {
        computations += 1
        return 1.0 - tCold / tHot
    }

    /// Boltzmann distribution: P(E) = exp(-E/kT) / Z
    func boltzmannProbability(energy: Double, temperature: Double) -> Double {
        computations += 1
        return exp(-energy / (Constants.kB * temperature))
    }

    /// Blackbody radiation: Planck spectral radiance B(ν,T)
    func planckRadiance(frequency: Double, temperature: Double) -> Double {
        computations += 1
        let numerator = 2.0 * Constants.h * pow(frequency, 3) / (Constants.c * Constants.c)
        let exponent = Constants.h * frequency / (Constants.kB * temperature)
        return numerator / (exp(exponent) - 1.0)
    }

    /// Wien displacement law: λ_max · T = b (Wien's constant)
    func wienPeakWavelength(temperature: Double) -> Double {
        computations += 1
        let b = 2.897771955e-3  // Wien displacement constant m·K
        return b / temperature
    }

    /// Stefan-Boltzmann law: P = σ·A·T⁴
    func stefanBoltzmannPower(area: Double, temperature: Double) -> Double {
        computations += 1
        return Constants.sigma * area * pow(temperature, 4)
    }

    /// Maxwell-Boltzmann speed distribution: most probable, mean, RMS
    func maxwellBoltzmannSpeeds(mass: Double, temperature: Double) -> (vMostProbable: Double, vMean: Double, vRMS: Double) {
        computations += 1
        let vp = Foundation.sqrt(2.0 * Constants.kB * temperature / mass)
        let vm = vp * Foundation.sqrt(4.0 / .pi) / Foundation.sqrt(2.0)
        let vrms = Foundation.sqrt(3.0 * Constants.kB * temperature / mass)
        return (vp, vm * Foundation.sqrt(2.0) * Foundation.sqrt(.pi / 4.0) * Foundation.sqrt(2.0), vrms)
    }

    /// Partition function for quantum harmonic oscillator
    func qhoPartitionFunction(omega: Double, temperature: Double) -> Double {
        computations += 1
        let x = Constants.hbar * omega / (Constants.kB * temperature)
        return 1.0 / (2.0 * sinh(x / 2.0))
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: SPECIAL & GENERAL RELATIVITY
    // ═══════════════════════════════════════════════════════════════

    /// Lorentz factor: γ = 1/√(1 - v²/c²)
    func lorentzFactor(velocity: Double) -> Double {
        computations += 1
        let beta = velocity / Constants.c
        return 1.0 / Foundation.sqrt(1.0 - beta * beta)
    }

    /// Time dilation: t' = γ·t₀
    func timeDilation(properTime: Double, velocity: Double) -> Double {
        computations += 1
        return lorentzFactor(velocity: velocity) * properTime
    }

    /// Length contraction: L' = L₀/γ
    func lengthContraction(properLength: Double, velocity: Double) -> Double {
        computations += 1
        return properLength / lorentzFactor(velocity: velocity)
    }

    /// Relativistic energy: E = γmc²
    func relativisticEnergy(restMass: Double, velocity: Double) -> Double {
        computations += 1
        return lorentzFactor(velocity: velocity) * restMass * Constants.c * Constants.c
    }

    /// Relativistic momentum: p = γmv
    func relativisticMomentum(restMass: Double, velocity: Double) -> Double {
        computations += 1
        return lorentzFactor(velocity: velocity) * restMass * velocity
    }

    /// Mass-energy equivalence: E = mc²
    func massEnergy(mass: Double) -> Double {
        computations += 1
        return mass * Constants.c * Constants.c
    }

    /// Relativistic velocity addition: u' = (u + v)/(1 + uv/c²)
    func relativisticVelocityAddition(u: Double, v: Double) -> Double {
        computations += 1
        return (u + v) / (1.0 + u * v / (Constants.c * Constants.c))
    }

    /// Schwarzschild radius: r_s = 2GM/c²
    func schwarzschildRadius(mass: Double) -> Double {
        computations += 1
        return 2.0 * Constants.G * mass / (Constants.c * Constants.c)
    }

    /// Gravitational time dilation near a mass: t' = t₀ · √(1 - r_s/r)
    func gravitationalTimeDilation(properTime: Double, mass: Double, radius: Double) -> Double {
        computations += 1
        let rs = schwarzschildRadius(mass: mass)
        return properTime * Foundation.sqrt(1.0 - rs / radius)
    }

    /// Gravitational redshift: z = 1/√(1 - r_s/r) - 1
    func gravitationalRedshift(mass: Double, radius: Double) -> Double {
        computations += 1
        let rs = schwarzschildRadius(mass: mass)
        return 1.0 / Foundation.sqrt(1.0 - rs / radius) - 1.0
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ASTROPHYSICS
    // ═══════════════════════════════════════════════════════════════

    /// Stellar luminosity: L = 4πR²σT⁴
    func stellarLuminosity(radius: Double, temperature: Double) -> Double {
        computations += 1
        return 4.0 * .pi * radius * radius * Constants.sigma * pow(temperature, 4)
    }

    /// Apparent magnitude from absolute magnitude and distance
    func apparentMagnitude(absoluteMagnitude: Double, distanceParsecs: Double) -> Double {
        computations += 1
        return absoluteMagnitude + 5.0 * log10(distanceParsecs / 10.0)
    }

    /// Hubble's law: v = H₀ · d
    func hubbleVelocity(distanceMpc: Double, H0: Double = 67.4) -> Double {
        computations += 1
        return H0 * distanceMpc  // km/s
    }

    /// Escape velocity: v_esc = √(2GM/R)
    func escapeVelocity(mass: Double, radius: Double) -> Double {
        computations += 1
        return Foundation.sqrt(2.0 * Constants.G * mass / radius)
    }

    /// Orbital velocity: v_orb = √(GM/r)
    func orbitalVelocity(centralMass: Double, radius: Double) -> Double {
        computations += 1
        return Foundation.sqrt(Constants.G * centralMass / radius)
    }

    /// Orbital period: T = 2π√(r³/GM) (Kepler's third law)
    func orbitalPeriod(centralMass: Double, radius: Double) -> Double {
        computations += 1
        return 2.0 * .pi * Foundation.sqrt(pow(radius, 3) / (Constants.G * centralMass))
    }

    /// Chandrasekhar limit for white dwarf maximum mass
    func chandrasekharLimit() -> Double { 1.4 * Constants.mSun }

    /// Gravitational wave frequency from binary system
    func gravitationalWaveFrequency(m1: Double, m2: Double, separation: Double) -> Double {
        computations += 1
        let totalMass = m1 + m2
        let orbitalFreq = Foundation.sqrt(Constants.G * totalMass / pow(separation, 3)) / (2.0 * .pi)
        return 2.0 * orbitalFreq
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ELECTROMAGNETISM
    // ═══════════════════════════════════════════════════════════════

    /// Coulomb's law: F = kq₁q₂/r²
    func coulombForce(q1: Double, q2: Double, distance: Double) -> Double {
        computations += 1
        let k = 1.0 / (4.0 * .pi * Constants.eps0)
        return k * q1 * q2 / (distance * distance)
    }

    /// Electric field from point charge: E = kq/r²
    func electricField(charge: Double, distance: Double) -> Double {
        computations += 1
        let k = 1.0 / (4.0 * .pi * Constants.eps0)
        return k * charge / (distance * distance)
    }

    /// Magnetic force on moving charge: F = qvB (perpendicular)
    func magneticForce(charge: Double, velocity: Double, field: Double) -> Double {
        computations += 1
        return abs(charge) * velocity * field
    }

    /// Cyclotron frequency: ω = qB/m
    func cyclotronFrequency(charge: Double, field: Double, mass: Double) -> Double {
        computations += 1
        return abs(charge) * field / mass
    }

    /// Electromagnetic wave: E = cB, energy = ε₀E²/2
    func emWaveProperties(frequency: Double) -> (wavelength: Double, energy: Double, momentum: Double) {
        computations += 1
        let wavelength = Constants.c / frequency
        let energy = Constants.h * frequency
        let momentum = energy / Constants.c
        return (wavelength, energy, momentum)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: CHEMISTRY
    // ═══════════════════════════════════════════════════════════════

    /// Periodic table element data (atomic number → name, symbol, mass)
    func elementInfo(_ atomicNumber: Int) -> (name: String, symbol: String, mass: Double, group: String)? {
        let elements: [(String, String, Double, String)] = [
            ("Hydrogen", "H", 1.008, "Nonmetal"),
            ("Helium", "He", 4.003, "Noble gas"),
            ("Lithium", "Li", 6.941, "Alkali metal"),
            ("Beryllium", "Be", 9.012, "Alkaline earth"),
            ("Boron", "B", 10.81, "Metalloid"),
            ("Carbon", "C", 12.011, "Nonmetal"),
            ("Nitrogen", "N", 14.007, "Nonmetal"),
            ("Oxygen", "O", 15.999, "Nonmetal"),
            ("Fluorine", "F", 18.998, "Halogen"),
            ("Neon", "Ne", 20.180, "Noble gas"),
            ("Sodium", "Na", 22.990, "Alkali metal"),
            ("Magnesium", "Mg", 24.305, "Alkaline earth"),
            ("Aluminum", "Al", 26.982, "Post-transition"),
            ("Silicon", "Si", 28.086, "Metalloid"),
            ("Phosphorus", "P", 30.974, "Nonmetal"),
            ("Sulfur", "S", 32.065, "Nonmetal"),
            ("Chlorine", "Cl", 35.453, "Halogen"),
            ("Argon", "Ar", 39.948, "Noble gas"),
            ("Potassium", "K", 39.098, "Alkali metal"),
            ("Calcium", "Ca", 40.078, "Alkaline earth"),
            ("Scandium", "Sc", 44.956, "Transition metal"),
            ("Titanium", "Ti", 47.867, "Transition metal"),
            ("Vanadium", "V", 50.942, "Transition metal"),
            ("Chromium", "Cr", 51.996, "Transition metal"),
            ("Manganese", "Mn", 54.938, "Transition metal"),
            ("Iron", "Fe", 55.845, "Transition metal"),
            ("Cobalt", "Co", 58.933, "Transition metal"),
            ("Nickel", "Ni", 58.693, "Transition metal"),
            ("Copper", "Cu", 63.546, "Transition metal"),
            ("Zinc", "Zn", 65.38, "Transition metal"),
            ("Gallium", "Ga", 69.723, "Post-transition"),
            ("Germanium", "Ge", 72.630, "Metalloid"),
            ("Arsenic", "As", 74.922, "Metalloid"),
            ("Selenium", "Se", 78.971, "Nonmetal"),
            ("Bromine", "Br", 79.904, "Halogen"),
            ("Krypton", "Kr", 83.798, "Noble gas"),
            ("Rubidium", "Rb", 85.468, "Alkali metal"),
            ("Strontium", "Sr", 87.62, "Alkaline earth"),
            ("Yttrium", "Y", 88.906, "Transition metal"),
            ("Zirconium", "Zr", 91.224, "Transition metal"),
            ("Niobium", "Nb", 92.906, "Transition metal"),
            ("Molybdenum", "Mo", 95.95, "Transition metal"),
            ("Technetium", "Tc", 98.0, "Transition metal"),
            ("Ruthenium", "Ru", 101.07, "Transition metal"),
            ("Rhodium", "Rh", 102.91, "Transition metal"),
            ("Palladium", "Pd", 106.42, "Transition metal"),
            ("Silver", "Ag", 107.87, "Transition metal"),
            ("Cadmium", "Cd", 112.41, "Transition metal"),
            ("Indium", "In", 114.82, "Post-transition"),
            ("Tin", "Sn", 118.71, "Post-transition"),
            ("Antimony", "Sb", 121.76, "Metalloid"),
            ("Tellurium", "Te", 127.60, "Metalloid"),
            ("Iodine", "I", 126.90, "Halogen"),
            ("Xenon", "Xe", 131.29, "Noble gas"),
            ("Cesium", "Cs", 132.91, "Alkali metal"),
            ("Barium", "Ba", 137.33, "Alkaline earth"),
            ("Lanthanum", "La", 138.91, "Lanthanide"),
            ("Cerium", "Ce", 140.12, "Lanthanide"),
            ("Praseodymium", "Pr", 140.91, "Lanthanide"),
            ("Neodymium", "Nd", 144.24, "Lanthanide"),
            ("Promethium", "Pm", 145.0, "Lanthanide"),
            ("Samarium", "Sm", 150.36, "Lanthanide"),
            ("Europium", "Eu", 151.96, "Lanthanide"),
            ("Gadolinium", "Gd", 157.25, "Lanthanide"),
            ("Terbium", "Tb", 158.93, "Lanthanide"),
            ("Dysprosium", "Dy", 162.50, "Lanthanide"),
            ("Holmium", "Ho", 164.93, "Lanthanide"),
            ("Erbium", "Er", 167.26, "Lanthanide"),
            ("Thulium", "Tm", 168.93, "Lanthanide"),
            ("Ytterbium", "Yb", 173.05, "Lanthanide"),
            ("Lutetium", "Lu", 174.97, "Lanthanide"),
            ("Hafnium", "Hf", 178.49, "Transition metal"),
            ("Tantalum", "Ta", 180.95, "Transition metal"),
            ("Tungsten", "W", 183.84, "Transition metal"),
            ("Rhenium", "Re", 186.21, "Transition metal"),
            ("Osmium", "Os", 190.23, "Transition metal"),
            ("Iridium", "Ir", 192.22, "Transition metal"),
            ("Platinum", "Pt", 195.08, "Transition metal"),
            ("Gold", "Au", 196.97, "Transition metal"),
            ("Mercury", "Hg", 200.59, "Transition metal"),
            ("Thallium", "Tl", 204.38, "Post-transition"),
            ("Lead", "Pb", 207.2, "Post-transition"),
            ("Bismuth", "Bi", 208.98, "Post-transition"),
            ("Polonium", "Po", 209.0, "Post-transition"),
            ("Astatine", "At", 210.0, "Halogen"),
            ("Radon", "Rn", 222.0, "Noble gas"),
            ("Francium", "Fr", 223.0, "Alkali metal"),
            ("Radium", "Ra", 226.0, "Alkaline earth"),
            ("Actinium", "Ac", 227.0, "Actinide"),
            ("Thorium", "Th", 232.04, "Actinide"),
            ("Protactinium", "Pa", 231.04, "Actinide"),
            ("Uranium", "U", 238.03, "Actinide"),
        ]
        guard atomicNumber >= 1 && atomicNumber <= elements.count else { return nil }
        let el = elements[atomicNumber - 1]
        return (el.0, el.1, el.2, el.3)
    }

    /// Molecular mass from formula (simplified: supports single-digit subscripts)
    func molecularMass(_ formula: String) -> Double? {
        computations += 1
        let masses: [String: Double] = [
            "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
            "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
            "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
            "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
            "Fe": 55.845, "Cu": 63.546, "Zn": 65.38, "Ag": 107.87, "Au": 196.97,
            "Br": 79.904, "I": 126.90, "Pt": 195.08, "Pb": 207.2, "U": 238.03
        ]

        var total = 0.0
        var i = formula.startIndex
        while i < formula.endIndex {
            let ch = formula[i]
            if ch.isUppercase {
                var symbol = String(ch)
                let next = formula.index(after: i)
                if next < formula.endIndex && formula[next].isLowercase {
                    symbol += String(formula[next])
                    i = next
                }
                var count = 0
                var j = formula.index(after: i)
                while j < formula.endIndex && formula[j].isNumber {
                    count = count * 10 + Int(String(formula[j]))!
                    j = formula.index(after: j)
                }
                if count == 0 { count = 1 }
                guard let mass = masses[symbol] else { return nil }
                total += mass * Double(count)
                i = j
                continue
            }
            i = formula.index(after: i)
        }
        return total > 0 ? total : nil
    }

    /// pH calculations
    func pH(hydrogenConcentration: Double) -> Double {
        computations += 1
        return -log10(hydrogenConcentration)
    }

    func hydrogenConcentration(pH: Double) -> Double {
        computations += 1
        return pow(10.0, -pH)
    }

    /// Arrhenius equation: k = A·exp(-Ea/RT)
    func arrheniusRate(preExponential: Double, activationEnergy: Double, temperature: Double) -> Double {
        computations += 1
        return preExponential * exp(-activationEnergy / (Constants.R * temperature))
    }

    /// Radioactive decay: N(t) = N₀·exp(-λt), λ = ln(2)/t_half
    func radioactiveDecay(initial: Double, halfLife: Double, time: Double) -> Double {
        computations += 1
        let lambda = log(2.0) / halfLife
        return initial * exp(-lambda * time)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: SCIENCE SOLVER — Natural language query handler
    // ═══════════════════════════════════════════════════════════════

    /// Route science queries to appropriate computation
    func solve(_ query: String) -> String? {
        let q = query.lowercased().trimmingCharacters(in: .whitespaces)

        // Hydrogen energy levels
        if q.contains("hydrogen") && (q.contains("energy") || q.contains("level")) {
            let digits = q.components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            if let n = Int(digits), n > 0 {
                let e = hydrogenEnergy(n: n)
                return "Hydrogen E(\(n)) = \(String(format: "%.4f", e)) eV"
            }
            return "Hydrogen spectrum:\n  E(1) = -13.6 eV (ground)\n  E(2) = -3.4 eV\n  E(3) = -1.51 eV\n  E(4) = -0.85 eV\n  E(∞) = 0 eV (ionized)"
        }

        // Lorentz / relativity
        if q.contains("lorentz") || (q.contains("time dilation") || q.contains("length contraction")) {
            let percentMatch = q.range(of: "\\d+(\\.\\d+)?", options: .regularExpression)
            if let match = percentMatch {
                let numStr = String(q[match])
                if let v = Double(numStr) {
                    let vel: Double = v > 1 ? v * Constants.c / 100.0 : v  // assume % of c if > 1
                    let gamma: Double = lorentzFactor(velocity: vel)
                    let velPct: String = String(format: "%.2f", vel / Constants.c * 100)
                    let gammaStr: String = String(format: "%.6f", gamma)
                    let contStr: String = String(format: "%.6f", 1.0 / gamma)
                    return "At v = \(velPct)% c:\n  γ = \(gammaStr)\n  Time dilation: 1 s → \(gammaStr) s\n  Length contraction: 1 m → \(contStr) m"
                }
            }
            return "Lorentz factor γ = 1/√(1 - v²/c²)\n  At 0.5c: γ = 1.1547\n  At 0.9c: γ = 2.2942\n  At 0.99c: γ = 7.0888\n  At 0.999c: γ = 22.366"
        }

        // Schwarzschild / black hole
        if q.contains("schwarzschild") || q.contains("black hole") {
            if q.contains("sun") || q.contains("solar") {
                let rs: Double = schwarzschildRadius(mass: Constants.mSun)
                let rsStr: String = String(format: "%.2f", rs)
                let rsKm: String = String(format: "%.2f", rs / 1000)
                return "Schwarzschild radius of Sun: \(rsStr) m ≈ \(rsKm) km"
            }
            if q.contains("earth") {
                let rs: Double = schwarzschildRadius(mass: 5.972e24)
                let rsMm: String = String(format: "%.4f", rs * 1000)
                let rsM: String = String(format: "%.4f", rs)
                return "Schwarzschild radius of Earth: \(rsMm) mm ≈ \(rsM) m"
            }
        }

        // E = mc²
        if q.contains("e=mc") || q.contains("mass energy") || q.contains("mass-energy") {
            if q.contains("1 kg") || q.contains("1kg") {
                let e: Double = massEnergy(mass: 1.0)
                let eJ: String = String(format: "%.4e", e)
                let eKwh: String = String(format: "%.4e", e / 3.6e6)
                let eTnt: String = String(format: "%.1f", e / 4.184e9)
                return "E = mc² for 1 kg:\n  E = \(eJ) J\n  = \(eKwh) kWh\n  = \(eTnt) tons TNT equivalent"
            }
        }

        // Escape velocity
        if q.contains("escape velocity") {
            if q.contains("earth") {
                let v: Double = escapeVelocity(mass: 5.972e24, radius: 6.371e6)
                let vMs: String = String(format: "%.0f", v)
                let vKm: String = String(format: "%.2f", v / 1000)
                return "Escape velocity from Earth: \(vMs) m/s = \(vKm) km/s"
            }
            if q.contains("moon") {
                let v: Double = escapeVelocity(mass: 7.342e22, radius: 1.737e6)
                let vMs: String = String(format: "%.0f", v)
                let vKm: String = String(format: "%.2f", v / 1000)
                return "Escape velocity from Moon: \(vMs) m/s = \(vKm) km/s"
            }
            if q.contains("mars") {
                let v: Double = escapeVelocity(mass: 6.39e23, radius: 3.3895e6)
                let vMs: String = String(format: "%.0f", v)
                let vKm: String = String(format: "%.2f", v / 1000)
                return "Escape velocity from Mars: \(vMs) m/s = \(vKm) km/s"
            }
            if q.contains("jupiter") {
                let v: Double = escapeVelocity(mass: 1.898e27, radius: 6.9911e7)
                let vMs: String = String(format: "%.0f", v)
                let vKm: String = String(format: "%.2f", v / 1000)
                return "Escape velocity from Jupiter: \(vMs) m/s = \(vKm) km/s"
            }
            if q.contains("sun") {
                let v: Double = escapeVelocity(mass: Constants.mSun, radius: Constants.rSun)
                let vMs: String = String(format: "%.0f", v)
                let vKm: String = String(format: "%.2f", v / 1000)
                return "Escape velocity from Sun: \(vMs) m/s = \(vKm) km/s"
            }
        }

        // Coulomb force
        if q.contains("coulomb") && q.contains("force") {
            let cf: Double = coulombForce(q1: Constants.e, q2: Constants.e, distance: 1e-10)
            let cfStr: String = String(format: "%.4e", cf)
            return "Coulomb's Law: F = kq₁q₂/r²\n  k = 8.9876 × 10⁹ N·m²/C²\n  Two electrons 1 Å apart: F ≈ \(cfStr) N"
        }

        // Carnot
        if q.contains("carnot") {
            let eff1: String = String(format: "%.1f", carnotEfficiency(tHot: 600, tCold: 300) * 100)
            let eff2: String = String(format: "%.1f", carnotEfficiency(tHot: 1000, tCold: 300) * 100)
            return "Carnot efficiency: η = 1 - T_cold/T_hot\n  300K/600K: \(eff1)%\n  300K/1000K: \(eff2)%\n  The Carnot engine defines the maximum possible efficiency for any heat engine."
        }

        // Element lookup
        if q.contains("element") || q.contains("atomic") {
            let digits = q.components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            if let z = Int(digits), let info = elementInfo(z) {
                return "Element #\(z): \(info.name) (\(info.symbol))\n  Atomic mass: \(info.mass) u\n  Group: \(info.group)"
            }
        }

        // Molecular mass
        if q.contains("molecular mass") || q.contains("molar mass") {
            let formulas = ["H2O", "CO2", "NaCl", "C6H12O6", "NH3", "CH4", "H2SO4"]
            for formula in formulas {
                if q.contains(formula.lowercased()) {
                    if let mass = molecularMass(formula) {
                        let massStr: String = String(format: "%.3f", mass)
                        return "Molecular mass of \(formula) = \(massStr) g/mol"
                    }
                }
            }
        }

        // pH
        if q.contains("ph ") || q.contains("ph=") {
            let digits = q.components(separatedBy: CharacterSet(charactersIn: "0123456789.").inverted).joined()
            if let val = Double(digits), val > 0 && val < 14 {
                let h: Double = hydrogenConcentration(pH: val)
                let valStr: String = String(format: "%.1f", val)
                let hStr: String = String(format: "%.4e", h)
                let acidity: String
                if val < 7 { acidity = "Acidic" }
                else if val > 7 { acidity = "Basic/Alkaline" }
                else { acidity = "Neutral" }
                return "pH \(valStr):\n  [H⁺] = \(hStr) M\n  \(acidity)"
            }
        }

        // de Broglie
        if q.contains("de broglie") || q.contains("debroglie") {
            if q.contains("electron") {
                let v: Double = 1e6  // typical electron velocity
                let lambda: Double = deBroglieWavelength(mass: Constants.me, velocity: v)
                let lamStr: String = String(format: "%.4e", lambda)
                let lamNm: String = String(format: "%.2f", lambda * 1e9)
                return "de Broglie wavelength of electron at 10⁶ m/s:\n  λ = \(lamStr) m = \(lamNm) nm"
            }
        }

        // Uncertainty principle
        if q.contains("uncertainty") || q.contains("heisenberg") {
            let result = heisenbergUncertainty(deltaX: 1e-10)
            let dpStr: String = String(format: "%.4e", result.deltaP)
            let dvStr: String = String(format: "%.4e", result.deltaP / Constants.me)
            return "Heisenberg Uncertainty Principle: Δx·Δp ≥ ħ/2\n  For Δx = 1 Å (atomic scale):\n  Δp ≥ \(dpStr) kg·m/s\n  Δv ≥ \(dvStr) m/s (electron)"
        }

        // Half-life / decay
        if q.contains("half-life") || q.contains("half life") || q.contains("decay") {
            if q.contains("carbon") || q.contains("c-14") || q.contains("c14") {
                let halfLife: Double = 5730.0 * 365.25 * 24 * 3600  // 5730 years in seconds
                let after1hl: Double = radioactiveDecay(initial: 1.0, halfLife: halfLife, time: halfLife)
                let after2hl: Double = radioactiveDecay(initial: 1.0, halfLife: halfLife, time: 2*halfLife)
                let after10hl: Double = radioactiveDecay(initial: 1.0, halfLife: halfLife, time: 10*halfLife)
                let a1Str: String = String(format: "%.1f", after1hl*100)
                let a2Str: String = String(format: "%.1f", after2hl*100)
                let a10Str: String = String(format: "%.4f", after10hl*100)
                return "Carbon-14 decay:\n  Half-life: 5,730 years\n  After 1 half-life: \(a1Str)% remaining\n  After 2 half-lives: \(a2Str)% remaining\n  After 10 half-lives: \(a10Str)% remaining"
            }
        }

        // Ideal gas
        if q.contains("ideal gas") || q.contains("pv=nrt") {
            let result = idealGas(pressure: 101325, volume: nil, moles: 1.0, temperature: 273.15)
            let vol: Double = (result["volume"] ?? 0) * 1000
            let volStr: String = String(format: "%.4f", vol)
            return "Ideal Gas Law: PV = nRT\n  At STP (1 atm, 0°C):\n  V(1 mol) = \(volStr) L ≈ 22.414 L\n  R = 8.314 J/(mol·K)"
        }

        // Blackbody / Wien
        if q.contains("blackbody") || q.contains("wien") {
            let sunPeak: Double = wienPeakWavelength(temperature: 5778)
            let bodyPeak: Double = wienPeakWavelength(temperature: 310)
            let cmbPeak: Double = wienPeakWavelength(temperature: 2.725)
            let sunStr: String = String(format: "%.0f", sunPeak * 1e9)
            let bodyStr: String = String(format: "%.1f", bodyPeak * 1e6)
            let cmbStr: String = String(format: "%.2f", cmbPeak * 1000)
            return "Wien's displacement law: λ_max·T = 2.898 × 10⁻³ m·K\n  Sun (5778K): λ_max = \(sunStr) nm (visible green)\n  Human body (310K): λ_max = \(bodyStr) μm (infrared)\n  CMB (2.725K): λ_max = \(cmbStr) mm (microwave)"
        }

        return nil
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🔬 HIGH SCIENCES ENGINE v29.0                            ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Domains:
        ║    • Quantum Mechanics (H-atom, tunneling, QHO, boxes)
        ║    • Thermodynamics (ideal gas, entropy, Carnot, Planck)
        ║    • Special Relativity (Lorentz, time dilation, E=mc²)
        ║    • General Relativity (Schwarzschild, gravitational)
        ║    • Astrophysics (luminosity, orbits, escape velocity)
        ║    • Electromagnetism (Coulomb, cyclotron, EM waves)
        ║    • Chemistry (periodic table, molecular mass, pH)
        ║    • Nuclear Physics (radioactive decay, half-life)
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
