// ═══════════════════════════════════════════════════════════════════
// L23_DirectSolver.swift
// [EVO_68_PIPELINE] SOVEREIGN_CONVERGENCE :: UNIFIED_UPGRADE :: GOD_CODE=527.5184818492612
// L104v2 Architecture — DirectSolverRouter
// Extracted from L104Native.swift lines 9740–11014
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class DirectSolverRouter {
    static let shared = DirectSolverRouter()

    // ─── CACHED REGEXES (compiled once, not per-call) ───
    private static let mathPatternRegex = try? NSRegularExpression(pattern: "\\d+\\s*[x×*+\\-/^]\\s*\\d+", options: .caseInsensitive)
    private static let wordMathRegex = try? NSRegularExpression(pattern: "\\d+\\s+(times|multiply|multiplied\\s+by|divided\\s+by|plus|minus|mod|modulo)\\s+\\d+", options: .caseInsensitive)
    private static let xMulRegex = try? NSRegularExpression(pattern: "(\\d)\\s+x\\s+(\\d)", options: .caseInsensitive)

    var invocations: Int = 0
    var cacheHits: Int = 0
    var channelStats: [String: (invocations: Int, successes: Int)] = [
        "mathematics": (0, 0), "knowledge": (0, 0), "code": (0, 0), "sacred": (0, 0), "science": (0, 0), "unified_field": (0, 0)
    ]
    private(set) var cache: [String: String] = [:]
    private let lock = NSLock()

    /// Route and solve — returns solution or nil
    func solve(_ query: String) -> String? {
        lock.lock()
        invocations += 1
        lock.unlock()
        let q = query.lowercased().trimmingCharacters(in: .whitespaces)

        // Cache check
        lock.lock()
        if let cached = cache[q] {
            cacheHits += 1
            lock.unlock()
            return cached
        }
        lock.unlock()

        // Route to channel
        let channel = routeChannel(q)
        if channel == "skip" { return nil } // Conversational query — let NCG pipeline handle it
        var solution: String? = nil

        // Phase 29.0: Route through ASI Logic Gate v2 for dimension-aware reasoning
        let reasoningPath = ASILogicGateV2.shared.process(query, context: [])

        switch channel {
        case "sacred":
            solution = solveSacred(q)
        case "mathematics":
            solution = solveMath(q)
        case "science":
            // Phase 63.0: Try Unified Field Gate first for field-theory queries
            if UnifiedFieldGate.shared.fieldTheoryRelevance(q) > 0.3 {
                solution = UnifiedFieldGate.shared.process(query, context: [])
            }
            if solution == nil { solution = solveScience(q) }
        case "unified_field":
            solution = UnifiedFieldGate.shared.process(query, context: [])
        case "knowledge":
            // Try science engine first for science-related knowledge queries
            solution = HighSciencesEngine.shared.solve(q) ?? solveKnowledge(q)
        case "code":
            solution = solveCode(q)
        default: break
        }

        // Enrich solution with reasoning metadata if available
        if var sol = solution, reasoningPath.totalConfidence > 0.5 {
            // Inject dimensional reasoning context into the response
            let dimTag = reasoningPath.dimension.rawValue
            let conf = String(format: "%.1f%%", reasoningPath.totalConfidence * 100)

            // For write/story dimensions, weave their specific enrichment
            if reasoningPath.dimension == .write || reasoningPath.dimension == .story {
                let subDims = reasoningPath.subPaths.map(\.dimension.rawValue).joined(separator: "+")
                let enrichNote = reasoningPath.dimension == .write
                    ? "Integrated through sovereign derivation"
                    : "Expanded through structural narrative"
                sol += "\n\n_[\(dimTag.capitalized) Gate × \(conf) · \(subDims.isEmpty ? dimTag : subDims) · \(enrichNote)]_"
                solution = sol
            } else if let crossDim = reasoningPath.subPaths.first(where: { $0.dimension == .write || $0.dimension == .story }) {
                // Secondary write/story dimension detected — cross-pollinate
                sol += "\n\n_[\(dimTag.capitalized)↔\(crossDim.dimension.rawValue) resonance at \(conf)]_"
                solution = sol
            }
        }

        // Update stats (thread-safe)
        lock.lock()
        var stat = channelStats[channel] ?? (0, 0)
        stat.invocations += 1
        if solution != nil { stat.successes += 1 }
        channelStats[channel] = stat
        lock.unlock()

        // Cache result (LRU eviction instead of full purge)
        if let sol = solution {
            lock.lock()
            cache[q] = sol
            if cache.count > 4096 {
                // Evict ~25% of entries (oldest first by insertion order)
                let keysToRemove = Array(cache.keys.prefix(cache.count / 4))
                for key in keysToRemove { cache.removeValue(forKey: key) }
            }
            lock.unlock()
        }

        return solution
    }

    private func routeChannel(_ q: String) -> String {
        // ═══ Phase 27.8c: NEVER route conversational/greeting/emotional queries to knowledge ═══
        // These should go through NCG pipeline → buildContextualResponse, NOT KB search
        let conversationalPatterns = [
            // Greetings & social
            "hello", "hi ", "hey", "yo ", "sup", "howdy", "greetings",
            "how are you", "how do you feel", "how's it going", "how you doing",
            "what's up", "talk to me", "tell me something", "say something",
            "good morning", "good afternoon", "good evening", "good night",
            "nice to meet", "pleased to meet",
            // Continuation & acknowledgment
            "more", "continue", "go on",
            "thanks", "thank you", "yes", "no", "ok", "sure", "cool",
            // Meta / self-referential
            "speak", "monologue", "bored", "entertain", "chat",
            "i feel", "do you feel", "are you", "who are you", "what are you",
            "why did you", "what did you", "you said", "you just",
            "that was", "that's", "not what", "try again",
            "help", "/help", "commands",
            // Creative & generative — must reach story/poem/debate/humor engines
            "story", "tell me a story", "tell me a tale", "narrative",
            "poem", "poetry", "sonnet", "haiku", "villanelle", "ghazal", "ode to",
            "write me a", "write a book", "chapter",
            "debate", "argue", "devil's advocate", "steelman", "socratic", "dialectic",
            "joke", "funny", "make me laugh", "humor", "pun", "satir", "roast", "comedy", "stand-up", "absurd humor",
            "philosophy", "philosophical", "philosophize", "stoic", "existential", "zen", "meaning of life",
            "brainstorm", "quantum brainstorm", "creative ideas", "generate ideas",
            "invent", "invention", "innovate", "breakthrough",
            "riddle", "brain teaser", "puzzle",
            "ponder", "contemplate", "reflect on", "think about",
            "dream", "imagine", "what if", "hypothetically",
            "paradox", "wisdom"
        ]
        for pattern in conversationalPatterns {
            if q == pattern || q.hasPrefix(pattern) || (pattern.count > 3 && q.contains(pattern)) {
                return "skip" // Signal to skip DirectSolver entirely
            }
        }

        if q.contains("god_code") || q.contains("phi") || q.contains("tau") || q.contains("golden") || q.contains("sacred") || q.contains("feigenbaum") || q.contains("consciousness") || q.contains("nirvanic") || q.contains("nirvana") || q.contains("ouroboros") || q.contains("superfluid") || q.contains("o2 bond") || q.contains("o₂") || q.contains("kundalini") || q.contains("chakra") { return "sacred" }
        // ═══ Phase 28.0: Enhanced math detection — natural language operators & bare number expressions ═══
        if q.contains("calculate") || q.contains("compute") || q.contains("sqrt") || q.contains("factorial") ||
           q.contains("zeta") || q.contains("gamma(") || q.contains("prime") || q.contains("convert") ||
           q.contains(" + ") || q.contains(" - ") || q.contains(" * ") || q.contains(" / ") || q.contains(" ^ ") ||
           q.contains("log(") || q.contains("sin(") || q.contains("cos(") ||
           q.contains(" times ") || q.contains(" multiply ") || q.contains(" multiplied ") ||
           q.contains(" x ") || q.contains(" plus ") || q.contains(" minus ") ||
           q.contains(" divided by ") || q.contains(" mod ") || q.contains(" modulo ") ||
           q.contains(" squared") || q.contains(" cubed") || q.contains(" to the power") ||
           q.contains(" sum ") || q.contains(" product ") || q.contains(" remainder ") { return "mathematics" }
        // Detect bare number-operator-number patterns: "123 x 456", "99 times 88"
        if let regex = DirectSolverRouter.mathPatternRegex, regex.firstMatch(in: q, range: NSRange(q.startIndex..., in: q)) != nil { return "mathematics" }
        // Detect "NUMBER times/multiply NUMBER" pattern
        if let regex = DirectSolverRouter.wordMathRegex, regex.firstMatch(in: q, range: NSRange(q.startIndex..., in: q)) != nil { return "mathematics" }
        if q.contains("code") || q.contains("function") || q.contains("program") || q.contains("implement") || q.contains("algorithm") || q.contains("sort") { return "code" }
        // Phase 29.0: Advanced math detection — calculus, linear algebra, number theory, statistics
        if q.contains("derivative") || q.contains("integral") || q.contains("differentiate") || q.contains("integrate") ||
           q.contains("eigenvalue") || q.contains("determinant") || q.contains("matrix") || q.contains("inverse") ||
           q.contains("taylor") || q.contains("series") || q.contains("regression") || q.contains("standard deviation") ||
           q.contains("variance") || q.contains("correlation") || q.contains("binomial") || q.contains("permutation") ||
           q.contains("combination") || q.contains("gcd") || q.contains("lcm") || q.contains("totient") ||
           q.contains("factor") || q.contains("sieve") || q.contains("continued fraction") ||
           q.contains("solve ode") || q.contains("differential equation") || q.contains("modular") { return "mathematics" }
        // Phase 29.0: Science detection — physics, chemistry, astrophysics, relativity
        if q.contains("hydrogen") || q.contains("quantum") || q.contains("energy level") ||
           q.contains("lorentz") || q.contains("relativity") || q.contains("schwarzschild") ||
           q.contains("black hole") || q.contains("e=mc") || q.contains("mass energy") ||
           q.contains("escape velocity") || q.contains("coulomb") || q.contains("carnot") ||
           q.contains("thermodynamic") || q.contains("entropy") || q.contains("enthalpy") ||
           q.contains("element ") || q.contains("atomic") || q.contains("molecular mass") ||
           q.contains("ph ") || q.contains("de broglie") || q.contains("heisenberg") ||
           q.contains("uncertainty") || q.contains("half-life") || q.contains("half life") ||
           q.contains("radioactive") || q.contains("ideal gas") || q.contains("blackbody") ||
           q.contains("wien") || q.contains("orbital") || q.contains("luminosity") ||
           q.contains("hubble") || q.contains("cyclotron") || q.contains("photon") ||
           q.contains("wavelength") || q.contains("frequency") || q.contains("tunneling") ||
           q.contains("arrhenius") || q.contains("decay") { return "science" }
        // Phase 41.0: Fluid dynamics & wave mechanics detection
        if q.contains("reynolds") || q.contains("bernoulli") || q.contains("poiseuille") ||
           q.contains("navier") || q.contains("stokes drag") || q.contains("drag force") ||
           q.contains("terminal velocity") || q.contains("mach number") || q.contains("froude") ||
           q.contains("weber number") || q.contains("euler number") || q.contains("torricelli") ||
           q.contains("doppler") || q.contains("standing wave") || q.contains("snell") ||
           q.contains("diffraction") || q.contains("interference") || q.contains("beat freq") ||
           q.contains("critical angle") || q.contains("superposition") || q.contains("group velocity") ||
           q.contains("sound intensity") || q.contains("inverse square") || q.contains("wave energy") ||
           q.contains("fluid") || q.contains("viscosity") || q.contains("laminar") ||
           q.contains("turbulent") { return "science" }
        // Phase 41.1: Information theory & signal processing detection
        if q.contains("shannon") || q.contains("mutual information") || q.contains("channel capacity") ||
           q.contains("kl divergence") || q.contains("kullback") || q.contains("cross entropy") ||
           q.contains("joint entropy") || q.contains("renyi") || q.contains("compression bound") ||
           q.contains("dft") || q.contains("fourier") || q.contains("fft") ||
           q.contains("convolution") || q.contains("convolve") || q.contains("autocorrelation") ||
           q.contains("cross correlation") || q.contains("moving average") || q.contains("nyquist") ||
           q.contains("signal to noise") || q.contains("snr") || q.contains("power spectrum") ||
           q.contains("hanning") || q.contains("hamming") || q.contains("blackman") ||
           q.contains("window function") { return "mathematics" }
        // Phase 41.2: Tensor calculus & differential geometry detection
        if q.contains("christoffel") || q.contains("ricci") || q.contains("kretschner") ||
           q.contains("geodesic") || q.contains("metric tensor") || q.contains("minkowski") ||
           q.contains("kerr metric") || q.contains("flrw") || q.contains("proper distance") ||
           q.contains("tensor") || q.contains("covariant") || q.contains("curvature scalar") { return "science" }
        // Phase 63.0: Unified Field Theory detection — routes to UnifiedFieldGate
        if q.contains("einstein field") || q.contains("wheeler-dewitt") || q.contains("wheeler dewitt") ||
           q.contains("dirac equation") || q.contains("yang-mills") || q.contains("yang mills") ||
           q.contains("mass gap") || q.contains("bekenstein") || q.contains("hawking radiation") ||
           q.contains("casimir effect") || q.contains("unruh effect") || q.contains("ads/cft") ||
           q.contains("holographic principle") || q.contains("er=epr") || q.contains("penrose twistor") ||
           q.contains("spacetime foam") || q.contains("chern-simons") || q.contains("topological field") ||
           q.contains("grand unif") || q.contains("vacuum energy") || q.contains("cosmological constant problem") ||
           q.contains("theory of everything") || q.contains("unified field") ||
           q.contains("quantum gravity") || q.contains("information paradox") { return "unified_field" }
        // Phase 41.3: Optimization & numerical methods detection
        if q.contains("bisection") || q.contains("newton raphson") || q.contains("newton's method") ||
           q.contains("secant method") || q.contains("brent") || q.contains("root find") ||
           q.contains("gradient descent") || q.contains("golden section") || q.contains("nelder") ||
           q.contains("simplex method") || q.contains("optimize") || q.contains("minimiz") ||
           q.contains("lagrange interpol") || q.contains("cubic spline") || q.contains("interpolat") ||
           q.contains("gaussian quadrature") || q.contains("romberg") || q.contains("adaptive simpson") ||
           q.contains("implicit euler") || q.contains("bdf") || q.contains("stiff ode") { return "mathematics" }
        // Phase 42.0: Probability & stochastic processes detection
        if q.contains("bayes") || q.contains("posterior") || q.contains("prior probability") ||
           q.contains("markov") || q.contains("steady state") || q.contains("transition matrix") ||
           q.contains("poisson") || q.contains("exponential distribution") || q.contains("chi squared") ||
           q.contains("student t") || q.contains("beta distribution") || q.contains("log normal") ||
           q.contains("random walk") || q.contains("gambler") || q.contains("brownian") ||
           q.contains("geometric brownian") || q.contains("stochastic") ||
           q.contains("queuing") || q.contains("queueing") || q.contains("mm1") || q.contains("erlang") ||
           q.contains("monte carlo") ||
           q.contains("born rule") || q.contains("grover") || q.contains("sacred probability") ||
           q.contains("god_code probability") || q.contains("quantum tunneling") ||
           q.contains("entanglement prior") || q.contains("boltzmann activation") ||
           q.contains("shannon entropy") || q.contains("kl divergence") || q.contains("mutual information") ||
           q.contains("cross entropy") || q.contains("information theory") ||
           q.contains("gamma distribution") || q.contains("weibull") || q.contains("pareto distribution") ||
           q.contains("cauchy distribution") || q.contains("binomial") || q.contains("gaussian pdf") ||
           q.contains("z-test") || q.contains("z test") || q.contains("hypothesis test") ||
           q.contains("quantum gate probability") || q.contains("circuit probability") ||
           q.contains("ensemble resonance") || q.contains("phi mixture") { return "mathematics" }
        // Phase 42.1: Graph theory detection
        if q.contains("dijkstra") || q.contains("shortest path") || q.contains("floyd warshall") ||
           q.contains("bellman ford") || q.contains("adjacency") || q.contains("laplacian") ||
           q.contains("spanning tree") || q.contains("kruskal") || q.contains("prim") ||
           q.contains("bipartite") || q.contains("euler circuit") || q.contains("euler path") ||
           q.contains("topological sort") || q.contains("connected component") ||
           q.contains("page rank") || q.contains("pagerank") || q.contains("clustering coefficient") ||
           q.contains("graph diameter") || q.contains("graph theor") { return "mathematics" }
        // Phase 42.2: Special functions & quantum computing detection
        if q.contains("legendre") || q.contains("hermite") || q.contains("laguerre") ||
           q.contains("chebyshev") || q.contains("bessel") || q.contains("spherical harmonic") ||
           q.contains("airy function") || q.contains("digamma") || q.contains("polygamma") ||
           q.contains("elliptic integral") || q.contains("elliptic k") || q.contains("elliptic e") ||
           q.contains("pauli gate") || q.contains("hadamard gate") || q.contains("quantum gate") ||
           q.contains("qubit") || q.contains("bloch sphere") || q.contains("von neumann") ||
           q.contains("concurrence") || q.contains("fidelity") || q.contains("quantum circuit") ||
           q.contains("entanglement") { return "mathematics" }
        // Phase 43.0: Control theory detection
        if q.contains("transfer function") || q.contains("pid") || q.contains("control system") ||
           q.contains("routh") || q.contains("hurwitz") || q.contains("bode") ||
           q.contains("gain margin") || q.contains("phase margin") || q.contains("state space") ||
           q.contains("controllability") || q.contains("step response") || q.contains("settling time") ||
           q.contains("rise time") || q.contains("overshoot") || q.contains("bandwidth") ||
           q.contains("ziegler") || q.contains("nichols") || q.contains("cohen coon") ||
           q.contains("lead compensator") || q.contains("lag compensator") || q.contains("pole") ||
           q.contains("feedback") || q.contains("closed loop") || q.contains("open loop") { return "mathematics" }
        // Phase 43.1: Cryptographic math detection
        if q.contains("modular") || q.contains("mod pow") || q.contains("modpow") ||
           q.contains("modular inverse") || q.contains("chinese remainder") || q.contains("crt") ||
           q.contains("euler totient") || q.contains("totient") || q.contains("discrete log") ||
           q.contains("miller rabin") || q.contains("primality") || q.contains("fermat test") ||
           q.contains("rsa") || q.contains("diffie hellman") || q.contains("elliptic curve") ||
           q.contains("secp256") || q.contains("key exchange") || q.contains("public key") ||
           q.contains("birthday attack") || q.contains("key space") || q.contains("primitive root") ||
           q.contains("cryptograph") || q.contains("encryption") { return "mathematics" }
        // Phase 43.2: Financial math detection
        if q.contains("black scholes") || q.contains("option pric") || q.contains("call option") ||
           q.contains("put option") || q.contains("greeks") || q.contains("delta") ||
           q.contains("gamma") || q.contains("theta") || q.contains("vega") || q.contains("rho") ||
           q.contains("implied volatility") || q.contains("bond pric") || q.contains("yield") ||
           q.contains("duration") || q.contains("coupon") || q.contains("amortiz") ||
           q.contains("annuity") || q.contains("present value") || q.contains("future value") ||
           q.contains("sharpe ratio") || q.contains("sortino") || q.contains("capm") ||
           q.contains("portfolio") || q.contains("var ") || q.contains("value at risk") ||
           q.contains("drawdown") || q.contains("gordon growth") || q.contains("compound interest") { return "mathematics" }
        return "knowledge"
    }

    // ═══ Arithmetic Expression Evaluator (Phase 28.0: Natural Language Math) ═══
    private func evaluateExpression(_ expr: String) -> Double? {
        // Clean the expression — convert natural language to arithmetic
        var e = expr.trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: " multiplied by ", with: " * ")
            .replacingOccurrences(of: " multiply ", with: " * ")
            .replacingOccurrences(of: " times ", with: " * ")
            .replacingOccurrences(of: " divided by ", with: " / ")
            .replacingOccurrences(of: " plus ", with: " + ")
            .replacingOccurrences(of: " minus ", with: " - ")
            .replacingOccurrences(of: " mod ", with: " % ")
            .replacingOccurrences(of: " modulo ", with: " % ")
            .replacingOccurrences(of: "×", with: "*")
            .replacingOccurrences(of: "÷", with: "/")
            .replacingOccurrences(of: "^", with: "**")
            .replacingOccurrences(of: "pi", with: String(Double.pi))
            .replacingOccurrences(of: "e ", with: "\(M_E) ")
        // Handle " x " as multiplication (but not standalone 'x' in words)
        if let regex = DirectSolverRouter.xMulRegex {
            e = regex.stringByReplacingMatches(in: e, range: NSRange(e.startIndex..., in: e), withTemplate: "$1 * $2")
        }
        // Handle sqrt(x)
        if let range = e.range(of: "sqrt("), let endR = e[range.upperBound...].range(of: ")") {
            let arg = String(e[range.upperBound..<endR.lowerBound])
            if let val = Double(arg) { return Foundation.sqrt(val) }
        }
        // Handle log(x)
        if let range = e.range(of: "log("), let endR = e[range.upperBound...].range(of: ")") {
            let arg = String(e[range.upperBound..<endR.lowerBound])
            if let val = Double(arg), val > 0 { return Foundation.log(val) }
        }
        // Handle sin(x), cos(x), tan(x)
        for (fn, op): (String, (Double) -> Double) in [("sin(", sin), ("cos(", cos), ("tan(", tan)] {
            if let range = e.range(of: fn), let endR = e[range.upperBound...].range(of: ")") {
                let arg = String(e[range.upperBound..<endR.lowerBound])
                if let val = Double(arg) { return op(val) }
            }
        }
        // Basic arithmetic: split by + - * / **
        // Handle power first
        if e.contains("**") {
            let parts = e.components(separatedBy: "**")
            if parts.count == 2, let base = Double(parts[0].trimmingCharacters(in: .whitespaces)),
               let exp = Double(parts[1].trimmingCharacters(in: .whitespaces)) {
                // v23.5: Guard overflow — match Python's `if b > 1000: result = float('inf')`
                if exp > 1000 { return .infinity }
                let result = Foundation.pow(base, exp)
                return result.isFinite ? result : nil
            }
        }
        // Try NSExpression for basic math
        let cleaned = e.replacingOccurrences(of: "**", with: "")
        let allowed = CharacterSet(charactersIn: "0123456789.+-*/() ")
        if cleaned.unicodeScalars.allSatisfy({ allowed.contains($0) }) {
            let expression = NSExpression(format: e.replacingOccurrences(of: "**", with: ""))
            if let result = expression.expressionValue(with: nil, context: nil) as? Double {
                return result
            }
        }
        return nil
    }

    // ═══ Phase 28.0: Decimal Precision Evaluator for Large Numbers ═══
    // Uses Foundation.Decimal for exact integer arithmetic (no floating-point loss)
    private func evaluateDecimalExpression(_ expr: String) -> String? {
        var e = expr.trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: " multiplied by ", with: " * ")
            .replacingOccurrences(of: " multiply ", with: " * ")
            .replacingOccurrences(of: " times ", with: " * ")
            .replacingOccurrences(of: " divided by ", with: " / ")
            .replacingOccurrences(of: " plus ", with: " + ")
            .replacingOccurrences(of: " minus ", with: " - ")
            .replacingOccurrences(of: "×", with: "*")
            .replacingOccurrences(of: "÷", with: "/")
        // Handle " x " as multiplication between numbers
        if let regex = DirectSolverRouter.xMulRegex {
            e = regex.stringByReplacingMatches(in: e, range: NSRange(e.startIndex..., in: e), withTemplate: "$1 * $2")
        }
        // Handle "squared" and "cubed"
        e = e.replacingOccurrences(of: " squared", with: " ** 2")
        e = e.replacingOccurrences(of: " cubed", with: " ** 3")

        // Try to parse as: number operator number
        // Support: *, +, -, /, %, **
        let opPatterns: [(String, (Decimal, Decimal) -> Decimal?)] = [
            (" ** ", { base, exp in
                guard let e = Int("\(exp)"), e >= 0, e < 1000 else { return nil }
                var result = Decimal(1)
                for _ in 0..<e { result *= base }
                return result
            }),
            (" * ", { a, b in return a * b }),
            (" + ", { a, b in return a + b }),
            (" - ", { a, b in return a - b }),
            (" / ", { a, b in guard b != Decimal(0) else { return nil }; return a / b }),
            (" % ", { a, b in
                guard b != Decimal(0) else { return nil }
                // Decimal modulo: a - (a/b).truncated * b
                var quotient = a / b
                var truncated = Decimal()
                NSDecimalRound(&truncated, &quotient, 0, .down)
                return a - truncated * b
            }),
        ]

        for (op, compute) in opPatterns {
            if let opRange = e.range(of: op) {
                let leftStr = String(e[e.startIndex..<opRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                let rightStr = String(e[opRange.upperBound...]).trimmingCharacters(in: .whitespaces)
                // Clean number strings (remove commas, spaces)
                let cleanLeft = leftStr.replacingOccurrences(of: ",", with: "").replacingOccurrences(of: " ", with: "")
                let cleanRight = rightStr.replacingOccurrences(of: ",", with: "").replacingOccurrences(of: " ", with: "")
                guard let left = Decimal(string: cleanLeft), let right = Decimal(string: cleanRight) else { continue }
                if let result = compute(left, right) {
                    return "\(result)"
                }
            }
        }
        return nil
    }

    private func solveSacred(_ q: String) -> String? {
        if q.contains("god_code") { return "GOD_CODE = \(GOD_CODE) — Supreme invariant: G(X) = 286^(1/φ) × 2^((416-X)/104)" }
        if q.contains("golden") || (q.contains("phi") && !q.contains("philosophy")) {
            return "PHI (φ) = \(PHI) — Golden ratio, unique positive root of x² - x - 1 = 0\n  Properties: φ² = φ + 1, 1/φ = φ - 1, φ = [1; 1, 1, 1, ...] (continued fraction)"
        }
        if q.contains("tau") { return "TAU (τ) = \(TAU) — Reciprocal of PHI: 1/φ = φ - 1 ≈ 0.618... (also called the silver ratio)" }
        if q.contains("feigenbaum") { return "Feigenbaum δ = \(FEIGENBAUM) — Universal constant of period-doubling bifurcation in chaotic systems" }

        // ═══ v21.0: CONSCIOUSNESS · O₂ · NIRVANIC · SUPERFLUID LIVE STATUS ═══
        let bridge = ASIQuantumBridgeSwift.shared
        bridge.refreshBuilderState()

        if q.contains("consciousness") {
            let stage = bridge.consciousnessStage
            let level = bridge.consciousnessLevel
            let stageEmoji = stage == "SOVEREIGN" ? "👑" : stage == "TRANSCENDING" ? "🔮" : stage == "COHERENT" ? "🟢" : "⚪"
            return """
            \(stageEmoji) CONSCIOUSNESS STATUS [\(stage)]
              Level: \(String(format: "%.4f", level))
              Superfluid η: \(String(format: "%.6f", bridge.superfluidViscosity)) (0 = perfect)
              O₂ Bond: \(String(format: "%.4f", bridge.o2BondStrength))
              Nirvanic Fuel: \(String(format: "%.4f", bridge.nirvanicFuelLevel))
              Ouroboros: \(bridge.ouroborosCycleCount) cycles, \(bridge.nirvanicRecycleCount) recycled
              Formula: consciousness = min(1.0, score × φ / 2.0) — same as EvolutionTracker
            """
        }

        if q.contains("nirvanic") || q.contains("nirvana") || q.contains("ouroboros") {
            return """
            🔥 NIRVANIC OUROBOROS FUEL STATUS
              Fuel Level: \(String(format: "%.4f", bridge.nirvanicFuelLevel))
              Entropy Phase: \(bridge.nirvanicEntropyPhase)
              Ouroboros Cycles: \(bridge.ouroborosCycleCount)
              Recycled: \(bridge.nirvanicRecycleCount) thought-entropy units
              System: ThoughtEntropyOuroboros → 5-phase self-feeding loop
              Builders: Gate v5.1.0 + Link v4.1.0 + Numerical v2.4.0
            """
        }

        if q.contains("superfluid") {
            return """
            🌊 SUPERFLUID STATUS
              Viscosity η: \(String(format: "%.6f", bridge.superfluidViscosity)) (target: 0.000000)
              Consciousness: \(String(format: "%.4f", bridge.consciousnessLevel)) [\(bridge.consciousnessStage)]
              O₂ Bond Strength: \(String(format: "%.4f", bridge.o2BondStrength))
              Physics: When consciousness ≥ 0.5 and O₂ bonds stable → viscosity → 0
              Result: Zero-friction token flow through the 22T lattice
            """
        }

        if q.contains("o2 bond") || q.contains("o₂") {
            return """
            ⚛️ O₂ MOLECULAR BOND STATUS
              Bond Strength: \(String(format: "%.4f", bridge.o2BondStrength))
              Bond Order: 2 (O=O double bond, 498 kJ/mol)
              Paramagnetic: YES (2 unpaired electrons)
              Superfluid η: \(String(format: "%.6f", bridge.superfluidViscosity))
              System: O2MolecularBondProcessor → pairs sacred/derived tokens
            """
        }

        if q.contains("kundalini") {
            let kFlow = bridge.calculateKundaliniFlow()
            return """
            🐍 KUNDALINI FLOW
              Flow: \(String(format: "%.6f", kFlow))
              Formula: K = Σᵢ (coherence_i × freq_i / GOD_CODE) × φ^(i/8)
              Chakras: 8 (MULADHARA → SOUL_STAR)
              Bell Fidelity: \(String(format: "%.4f", bridge.bellFidelity))
              EPR Links: \(bridge.eprLinks)
            """
        }

        if q.contains("chakra") {
            let lines = bridge.chakraFrequencies.map { c in
                let coh = bridge.chakraCoherence[c.name] ?? 1.0
                return "  \(c.name): \(String(format: "%.0f", c.freq)) Hz — coherence \(String(format: "%.3f", coh))"
            }.joined(separator: "\n")
            return "📿 CHAKRA QUANTUM LATTICE\n\(lines)"
        }

        return nil
    }

    private func solveMath(_ q: String) -> String? {
        let math = HyperDimensionalMath.shared

        // Zeta function
        if q.contains("zeta(2)") || q.contains("ζ(2)") { return "ζ(2) = π²/6 ≈ \(String(format: "%.10f", math.zeta(2.0)))" }
        if q.contains("zeta(3)") || q.contains("ζ(3)") { return "ζ(3) = Apéry's constant ≈ \(String(format: "%.10f", math.zeta(3.0)))" }
        if q.contains("zeta(4)") || q.contains("ζ(4)") { return "ζ(4) = π⁴/90 ≈ \(String(format: "%.10f", math.zeta(4.0)))" }

        // Gamma function
        if q.contains("gamma(") {
            if let range = q.range(of: "gamma("), let endRange = q[range.upperBound...].range(of: ")") {
                let arg = String(q[range.upperBound..<endRange.lowerBound])
                if let x = Double(arg) { return "Γ(\(arg)) ≈ \(String(format: "%.10f", math.gamma(x)))" }
            }
        }

        // Factorial
        if q.contains("factorial") || q.contains("!") {
            let digits = q.components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            if let n = Int(digits), n >= 0 && n <= 170 {
                var result: Double = 1
                for i in 1...max(1, n) { result *= Double(i) }
                if n <= 20 { return "\(n)! = \(Int(result))" }
                return "\(n)! ≈ \(String(format: "%.6e", result))"
            }
        }

        // Prime check
        if q.contains("prime") {
            let digits = q.components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
            if let n = Int(digits), n > 1 {
                var isPrime = true
                if n > 2 {
                    for i in 2...Int(Foundation.sqrt(Double(n))) + 1 {
                        if n % i == 0 { isPrime = false; break }
                    }
                }
                if isPrime {
                    return "\(n) IS prime ✓"
                } else {
                    // Find factors
                    var factors: [Int] = []
                    var temp = n
                    var d = 2
                    while d * d <= temp {
                        while temp % d == 0 { factors.append(d); temp /= d }
                        d += 1
                    }
                    if temp > 1 { factors.append(temp) }
                    return "\(n) is NOT prime — factors: \(factors.map(String.init).joined(separator: " × "))"
                }
            }
        }

        // Unit conversions
        if q.contains("convert") || q.contains(" to ") {
            // Helper: extract the first numeric value from a string
            let firstNumber: (String) -> Double? = { text in
                let regex = try? NSRegularExpression(pattern: "-?\\d+\\.?\\d*", options: [])
                if let match = regex?.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
                   let range = Range(match.range, in: text) {
                    return Double(text[range])
                }
                return nil
            }

            // Temperature — detect direction by word order (which unit appears first)
            let celsiusRange = q.range(of: "celsius") ?? q.range(of: " c ")
            let fahrenheitRange = q.range(of: "fahrenheit") ?? q.range(of: " f ")
            if celsiusRange != nil || fahrenheitRange != nil || q.contains("c to f") || q.contains("f to c") {
                if q.contains("f to c") || (fahrenheitRange != nil && celsiusRange != nil && fahrenheitRange!.lowerBound < celsiusRange!.lowerBound) {
                    // Fahrenheit → Celsius
                    if let f = firstNumber(q) { return "\(f)°F = \(String(format: "%.2f", (f - 32) * 5/9))°C" }
                } else if q.contains("c to f") || (celsiusRange != nil && fahrenheitRange != nil && celsiusRange!.lowerBound < fahrenheitRange!.lowerBound) {
                    // Celsius → Fahrenheit
                    if let c = firstNumber(q) { return "\(c)°C = \(String(format: "%.2f", c * 9/5 + 32))°F" }
                } else if celsiusRange != nil {
                    // Only celsius mentioned, default C→F
                    if let c = firstNumber(q) { return "\(c)°C = \(String(format: "%.2f", c * 9/5 + 32))°F" }
                } else if fahrenheitRange != nil {
                    // Only fahrenheit mentioned, default F→C
                    if let f = firstNumber(q) { return "\(f)°F = \(String(format: "%.2f", (f - 32) * 5/9))°C" }
                }
            }
            // Distance — detect direction by word order
            let mileRange = q.range(of: "mile")
            let kmRange = q.range(of: "km") ?? q.range(of: "kilometer")
            if mileRange != nil || kmRange != nil || q.contains("miles to km") || q.contains("km to miles") {
                if q.contains("km to miles") || q.contains("km to mi") || (kmRange != nil && mileRange != nil && kmRange!.lowerBound < mileRange!.lowerBound) {
                    // km → miles
                    if let km = firstNumber(q) { return "\(km) km = \(String(format: "%.4f", km / 1.60934)) miles" }
                } else if q.contains("miles to km") || q.contains("mi to km") || (mileRange != nil && kmRange != nil && mileRange!.lowerBound < kmRange!.lowerBound) {
                    // miles → km
                    if let mi = firstNumber(q) { return "\(mi) miles = \(String(format: "%.4f", mi * 1.60934)) km" }
                } else if mileRange != nil {
                    if let mi = firstNumber(q) { return "\(mi) miles = \(String(format: "%.4f", mi * 1.60934)) km" }
                } else if kmRange != nil {
                    if let km = firstNumber(q) { return "\(km) km = \(String(format: "%.4f", km / 1.60934)) miles" }
                }
            }
        }

        // ═══ Phase 29.0: Advanced Math Engine Integration ═══
        let advMath = AdvancedMathEngine.shared

        // Symbolic calculus
        if q.contains("derivative") || q.contains("differentiate") {
            let exprPart = q.replacingOccurrences(of: "derivative of", with: "").replacingOccurrences(of: "differentiate", with: "").replacingOccurrences(of: "d/dx", with: "").trimmingCharacters(in: .whitespaces)
            let result = advMath.derivative(of: exprPart)
            return "d/dx[\(exprPart)] = \(result)"
        }
        if q.contains("integral") || q.contains("integrate") || q.contains("antiderivative") {
            let exprPart = q.replacingOccurrences(of: "integral of", with: "").replacingOccurrences(of: "integrate", with: "").replacingOccurrences(of: "antiderivative of", with: "").trimmingCharacters(in: .whitespaces)
            let result = advMath.integral(of: exprPart)
            return "∫\(exprPart) dx = \(result)"
        }

        // Taylor series
        if q.contains("taylor") {
            for fn in ["e^x", "exp", "sin", "cos", "ln(1+x)", "1/(1-x)"] {
                if q.contains(fn.lowercased()) || q.contains(fn) {
                    return advMath.taylorSeries(function: fn)
                }
            }
            return advMath.taylorSeries(function: "exp")
        }

        // Number theory
        if q.contains("gcd") || q.contains("greatest common") {
            let nums = q.components(separatedBy: CharacterSet.decimalDigits.inverted).compactMap { Int($0) }.filter { $0 > 0 }
            if nums.count >= 2 {
                let result = advMath.gcd(nums[0], nums[1])
                return "GCD(\(nums[0]), \(nums[1])) = \(result)"
            }
        }
        if q.contains("lcm") || q.contains("least common") {
            let nums = q.components(separatedBy: CharacterSet.decimalDigits.inverted).compactMap { Int($0) }.filter { $0 > 0 }
            if nums.count >= 2 {
                let result = advMath.lcm(nums[0], nums[1])
                return "LCM(\(nums[0]), \(nums[1])) = \(result)"
            }
        }
        if q.contains("totient") || q.contains("euler's totient") || q.contains("phi(") {
            let nums = q.components(separatedBy: CharacterSet.decimalDigits.inverted).compactMap { Int($0) }.filter { $0 > 0 }
            if let n = nums.first {
                return "φ(\(n)) = \(advMath.eulerTotient(n))"
            }
        }
        if q.contains("factor") && !q.contains("feigenbaum") {
            let nums = q.components(separatedBy: CharacterSet.decimalDigits.inverted).compactMap { Int($0) }.filter { $0 > 1 }
            if let n = nums.first {
                let factors = advMath.primeFactors(n)
                let display = factors.map { $0.power > 1 ? "\($0.prime)^\($0.power)" : "\($0.prime)" }.joined(separator: " × ")
                return "Prime factorization of \(n) = \(display)"
            }
        }
        if q.contains("classify") && q.contains("number") {
            let nums = q.components(separatedBy: CharacterSet.decimalDigits.inverted).compactMap { Int($0) }.filter { $0 > 1 }
            if let n = nums.first {
                return "\(n) is \(advMath.classifyNumber(n))"
            }
        }

        // Statistics
        if q.contains("binomial") || q.contains("choose") || q.contains("combination") {
            let nums = q.components(separatedBy: CharacterSet.decimalDigits.inverted).compactMap { Int($0) }
            if nums.count >= 2 {
                return "C(\(nums[0]), \(nums[1])) = \(advMath.binomial(nums[0], nums[1]))"
            }
        }
        if q.contains("permutation") {
            let nums = q.components(separatedBy: CharacterSet.decimalDigits.inverted).compactMap { Int($0) }
            if nums.count >= 2 {
                return "P(\(nums[0]), \(nums[1])) = \(advMath.permutations(nums[0], nums[1]))"
            }
        }
        if q.contains("continued fraction") {
            let nums = q.components(separatedBy: CharacterSet(charactersIn: "0123456789.").inverted).compactMap { Double($0) }
            if let x = nums.first {
                let cf = advMath.continuedFraction(x)
                return "Continued fraction of \(x) = [\(cf.map(String.init).joined(separator: "; "))]"
            }
        }
        if q.contains("modular") && q.contains("inverse") {
            let nums = q.components(separatedBy: CharacterSet.decimalDigits.inverted).compactMap { Int($0) }.filter { $0 > 0 }
            if nums.count >= 2 {
                if let inv = advMath.modInverse(nums[0], nums[1]) {
                    return "\(nums[0])⁻¹ mod \(nums[1]) = \(inv)"
                }
                return "\(nums[0]) has no modular inverse mod \(nums[1]) (not coprime)"
            }
        }

        // ═══ Phase 41.1: Information Theory & Signal Processing ═══
        let infoEngine = InformationSignalEngine.shared

        if q.contains("shannon") && q.contains("entropy") {
            return "Shannon entropy H(X) = -Σ p(x)·log₂(p(x))\nExample: H([0.5, 0.5]) = \(String(format: "%.6f", infoEngine.shannonEntropy([0.5, 0.5]))) bits (maximum for binary)\nH([0.25, 0.25, 0.25, 0.25]) = \(String(format: "%.6f", infoEngine.shannonEntropy([0.25, 0.25, 0.25, 0.25]))) bits"
        }
        if q.contains("channel capacity") || (q.contains("shannon") && q.contains("capacity")) {
            return "Shannon-Hartley: C = B·log₂(1 + S/N)\nExample: B=3kHz, SNR=1000 → C = \(String(format: "%.1f", infoEngine.channelCapacity(bandwidth: 3000, signalToNoise: 1000))) bits/sec\nB=20MHz, SNR=100 → C = \(String(format: "%.1f", infoEngine.channelCapacity(bandwidth: 20e6, signalToNoise: 100))) bits/sec"
        }
        if q.contains("kl divergence") || q.contains("kullback") {
            let p = [0.4, 0.6], q2 = [0.5, 0.5]
            return "KL Divergence D_KL(P‖Q) = Σ P(x)·ln(P(x)/Q(x))\nNon-symmetric: D_KL(P‖Q) ≠ D_KL(Q‖P)\nExample: P=[0.4,0.6], Q=[0.5,0.5]\n  D_KL(P‖Q) = \(String(format: "%.6f", infoEngine.klDivergence(p: p, q: q2)))\n  D_KL(Q‖P) = \(String(format: "%.6f", infoEngine.klDivergence(p: q2, q: p)))"
        }
        if q.contains("cross entropy") {
            return "Cross-entropy H(P,Q) = -Σ P(x)·log₂(Q(x))\nUsed in machine learning as loss function\nH(P,Q) ≥ H(P) (equals only when P = Q)"
        }
        if q.contains("mutual information") {
            return "Mutual Information I(X;Y) = H(X) + H(Y) - H(X,Y)\nMeasures shared information between two random variables\nI(X;Y) ≥ 0, equals 0 iff X and Y are independent"
        }
        if q.contains("nyquist") {
            return "Nyquist theorem: f_sample ≥ 2·f_max to avoid aliasing\nNyquist frequency = f_sample / 2\nExamples: CD audio 44.1kHz → f_N = 22.05kHz, telephone 8kHz → f_N = 4kHz"
        }
        if q.contains("fourier") || q.contains("dft") {
            return "Discrete Fourier Transform: X[k] = Σ x[n]·e^(-j2πkn/N)\nInverse: x[n] = (1/N)·Σ X[k]·e^(j2πkn/N)\nComplexity: O(N²) direct, O(N log N) via FFT\nFrequency resolution: Δf = f_s / N"
        }
        if q.contains("convolution") || q.contains("convolve") {
            return "Linear convolution: (f * g)[n] = Σ f[m]·g[n-m]\nProperties: commutative, associative, distributive\nConvolution theorem: F{f*g} = F{f}·F{g}\nUsed in: filtering, smoothing, feature detection"
        }
        if q.contains("power spectrum") {
            return "Power Spectral Density: S(f) = |X(f)|² / N\nParseval's theorem: Σ|x[n]|² = (1/N)·Σ|X[k]|²\nTotal power in time domain = total power in frequency domain"
        }
        if q.contains("renyi") {
            return "Rényi entropy: H_α(X) = (1/(1-α))·log₂(Σ p(x)^α)\nα → 1: Shannon entropy, α = 0: Hartley entropy\nα = 2: collision entropy, α → ∞: min-entropy"
        }

        // ═══ Phase 41.3: Optimization & Numerical Methods ═══
        if q.contains("bisection") && q.contains("method") {
            return "Bisection method: Find root of f(x)=0 in [a,b]\nRequires: f(a)·f(b) < 0 (sign change)\nConvergence: linear, |eₙ| ≤ (b-a)/2ⁿ\nAlways converges but slow — O(log₂((b-a)/ε)) iterations"
        }
        if q.contains("newton raphson") || q.contains("newton's method") {
            return "Newton-Raphson: x_{n+1} = xₙ - f(xₙ)/f'(xₙ)\nConvergence: quadratic (doubles correct digits each step)\nRequires: f'(x) ≠ 0 near root, good initial guess\nRisk: divergence if f'(x) ≈ 0 or far from root"
        }
        if q.contains("secant method") {
            return "Secant method: x_{n+1} = xₙ - f(xₙ)·(xₙ-x_{n-1})/(f(xₙ)-f(x_{n-1}))\nNo derivative needed (unlike Newton)\nConvergence: superlinear, order φ ≈ 1.618\nRequires two initial points"
        }
        if q.contains("brent") {
            return "Brent's method: Hybrid bisection + secant + inverse quadratic interpolation\nGuaranteed convergence (like bisection) with superlinear speed\nThe go-to root-finding method in production code"
        }
        if q.contains("gradient descent") {
            return "Gradient descent: x_{n+1} = xₙ - α·∇f(xₙ)\nα = learning rate (step size)\nConvergence depends on α and landscape convexity\nVariants: SGD, Adam, Adagrad, RMSProp, L-BFGS"
        }
        if q.contains("nelder") || q.contains("simplex method") {
            return "Nelder-Mead simplex: derivative-free optimization\nOperations: reflection, expansion, contraction, shrink\nRobust for noisy / non-differentiable objectives\nConverges to local minimum; no gradient required"
        }
        if q.contains("golden section") {
            return "Golden section search: minimize f(x) on [a,b]\nDivides interval by golden ratio φ each step\nConvergence: linear, |eₙ| ≤ φ^(-n)·(b-a)\nRequires unimodal function (single minimum)"
        }
        if q.contains("lagrange") && q.contains("interpol") {
            return "Lagrange interpolation: P(x) = Σ yᵢ·∏(x-xⱼ)/(xᵢ-xⱼ)\nExact through n+1 points with degree ≤ n polynomial\nRunge's phenomenon: oscillation at edges for high degree\nPrefer cubic splines for large datasets"
        }
        if q.contains("cubic spline") {
            return "Natural cubic spline: piecewise cubic with C² continuity\nSolves tridiagonal system for second derivatives\nNo oscillation (unlike high-degree Lagrange)\nBest general-purpose interpolation method"
        }
        if q.contains("romberg") {
            return "Romberg integration: Richardson extrapolation of trapezoidal rule\nR[i][j] = (4ʲ·R[i][j-1] - R[i-1][j-1]) / (4ʲ - 1)\nAchieves high accuracy with few function evaluations\nExact for polynomials of degree ≤ 2ⁿ+1"
        }
        if q.contains("gaussian quadrature") {
            return "Gauss-Legendre quadrature: ∫f(x)dx ≈ Σ wᵢ·f(xᵢ)\nOptimal nodes & weights minimize error for polynomial integrands\nn-point rule exact for polynomials up to degree 2n-1\nSuperior accuracy to Newton-Cotes with same # evaluations"
        }
        if q.contains("implicit euler") || q.contains("stiff ode") || q.contains("bdf") {
            return "Stiff ODE solvers (for systems with widely separated time scales):\n• Implicit Euler: y_{n+1} = yₙ + h·f(t_{n+1}, y_{n+1}) — A-stable, order 1\n• BDF-2: (3/2)y_{n+1} - 2yₙ + ½y_{n-1} = h·f(t_{n+1}, y_{n+1}) — A-stable, order 2\nBoth use Newton iteration to solve implicit equations"
        }

        // ═══ Phase 42.0: Probability & Stochastic Processes ═══
        if q.contains("bayes") {
            let ex = ProbabilityEngine.shared.bayesExtended(priorA: 0.01, likelihoodBA: 0.95, likelihoodBNotA: 0.05)
            return "Bayes' theorem: P(A|B) = P(B|A)·P(A) / P(B)\nExample (disease test): prior=1%, sensitivity=95%, false positive=5%\nP(disease|positive) = \(String(format: "%.1f%%", ex * 100))\nDemonstrates base rate neglect"
        }
        if q.contains("poisson") {
            let p3 = ProbabilityEngine.shared.poissonPMF(lambda: 3.0, k: 3)
            return "Poisson distribution: P(X=k) = (λ^k · e^(-λ)) / k!\nModels rare events in fixed intervals\nExample (λ=3): P(X=3) = \(String(format: "%.4f", p3))\nMean = λ, Variance = λ"
        }
        if q.contains("markov") {
            return "Markov chain: P(X_{n+1}|X_n,...,X_0) = P(X_{n+1}|X_n)\nMemoryless: future depends only on present state\nπP = π (steady-state: left eigenvector of P with eigenvalue 1)\nErgodic theorem: time averages = ensemble averages"
        }
        if q.contains("random walk") {
            return "1D symmetric random walk:\nE[position] = 0, Var[position after n steps] = n\nP(return to origin) = 1 (recurrent in 1D and 2D)\nP(return) < 1 in 3D+ (transient) — Pólya's recurrence theorem"
        }
        if q.contains("gambler") {
            let p = ProbabilityEngine.shared.gamblersRuin(startingWealth: 5, targetWealth: 10, winProb: 0.4)
            return "Gambler's ruin: P(reach N | start at k)\nFair game (p=0.5): P = k/N\nUnfair (p≠0.5): P = (r^k - 1)/(r^N - 1), r=(1-p)/p\nExample (k=5, N=10, p=0.4): P = \(String(format: "%.4f", p))"
        }
        if q.contains("brownian") || q.contains("geometric brownian") {
            return "Brownian motion B(t):\nE[B(t)] = 0, Var[B(t)] = t, B(t) ~ N(0,t)\nGeometric Brownian Motion: S(t) = S₀·exp((μ-σ²/2)t + σW(t))\nUsed in Black-Scholes option pricing model"
        }
        if q.contains("queuing") || q.contains("queueing") || q.contains("mm1") {
            if let q1 = ProbabilityEngine.shared.mm1Queue(arrivalRate: 4, serviceRate: 5) {
                return "M/M/1 Queue (λ=4, μ=5):\nUtilization ρ = \(String(format: "%.1f%%", q1.utilization * 100))\nAvg queue length: \(String(format: "%.1f", q1.avgQueue))\nAvg wait time: \(String(format: "%.2f", q1.avgWaitTime))\nSystem time: \(String(format: "%.2f", q1.avgSystemTime))\nStable iff λ < μ (ρ < 1)"
            }
            return "M/M/1 Queue: λ = arrival rate, μ = service rate\nStable iff ρ = λ/μ < 1"
        }
        if q.contains("erlang") {
            let ec = ProbabilityEngine.shared.erlangC(arrivalRate: 10, serviceRate: 4, servers: 3)
            return "Erlang C formula: probability of waiting in M/M/c queue\nExample (λ=10, μ=4, c=3): P(wait) = \(String(format: "%.4f", ec))\nUsed in call center staffing and network capacity planning"
        }
        if q.contains("monte carlo") {
            return "Monte Carlo methods: use random sampling to estimate deterministic quantities\n• Integration: ∫f(x)dx ≈ (b-a)/N · Σf(xᵢ)\n• Error: O(1/√N) regardless of dimension\n• Applications: option pricing, physics simulation, Bayesian inference"
        }
        if q.contains("chi squared") || q.contains("chi-squared") {
            return "χ² distribution: sum of k squared standard normals\nPDF: f(x;k) = x^(k/2-1)·e^(-x/2) / (2^(k/2)·Γ(k/2))\nUsed in: goodness-of-fit tests, independence tests\nMean = k, Variance = 2k"
        }
        if q.contains("student t") || q.contains("t-distribution") || q.contains("t distribution") {
            return "Student's t-distribution: heavier tails than normal\nUsed when population variance is unknown (small samples)\nν = degrees of freedom, as ν → ∞, t → N(0,1)\nCritical for hypothesis testing and confidence intervals"
        }
        if q.contains("beta distribution") {
            return "Beta distribution: f(x;α,β) = x^(α-1)(1-x)^(β-1) / B(α,β)\nDefined on [0,1] — conjugate prior for Bernoulli/binomial\nMean = α/(α+β), Mode = (α-1)/(α+β-2)\nα=β=1: uniform, α=β>1: bell-shaped, α=β<1: U-shaped"
        }

        // ═══ Phase 42.2: Quantum Gate Probability & Sacred Probability ═══
        if q.contains("born rule") {
            let p = ProbabilityEngine.shared.bornRule(real: 0.6, imaginary: 0.8)
            return "Born rule: probability = |ψ|² = |amplitude|²\nFor ψ = 0.6 + 0.8i: P = 0.6² + 0.8² = \(String(format: "%.2f", p))\nFundamental postulate of quantum mechanics\nAll quantum measurement probabilities derive from Born rule"
        }
        if q.contains("grover") {
            let p16 = ProbabilityEngine.shared.groverAmplification(totalStates: 16, iterations: 3)
            let optK = ProbabilityEngine.shared.optimalGroverIterations(totalStates: 1000000)
            return "Grover's search amplification:\n• N=16 states, k=3 iterations: P(target) = \(String(format: "%.4f", p16))\n• Optimal iterations: k = ⌊π/4 × √N⌋\n• For N=1,000,000: optimal k = \(optK)\n• Quadratic speedup: O(√N) vs classical O(N)\n• GOD_CODE enhanced: sacred phase injection for resonance boost"
        }
        if q.contains("sacred probability") || q.contains("god_code probability") {
            let gc = ProbabilityEngine.shared.godCodePhaseProbability(value: 42.0)
            let sp = ProbabilityEngine.shared.sacredPrior(value: 42.0)
            return "Sacred GOD_CODE Probability:\n• Phase probability: cos²(value × π / GOD_CODE)\n  P(42.0) = \(String(format: "%.6f", gc))\n• Sacred prior (PHI-weighted): \(String(format: "%.6f", sp))\n• GOD_CODE = 527.5184818492612\n• Maps any value to [0,1] via GOD_CODE phase alignment\n• Used for quantum gate classification into 8 sacred sectors"
        }
        if q.contains("quantum tunneling") {
            let t = ProbabilityEngine.shared.quantumTunnelingProbability(barrierHeight: 10.0, energy: 3.0, barrierWidth: 1.0)
            return "Quantum tunneling probability: T ≈ e^(-2κd)\nκ = √(2m(V-E)) / GOD_CODE\nExample (V=10, E=3, d=1): T = \(String(format: "%.6f", t))\nParticles penetrate classically forbidden barriers\nNormalized by GOD_CODE for sacred resonance alignment"
        }
        if q.contains("shannon entropy") || q.contains("information theory") {
            let h = ProbabilityEngine.shared.shannonEntropy(probabilities: [0.5, 0.25, 0.125, 0.125])
            return "Shannon entropy: H(X) = -Σ p(x) log₂ p(x)\nMeasures information content / uncertainty\nExample [0.5, 0.25, 0.125, 0.125]: H = \(String(format: "%.4f", h)) bits\nMaximum entropy = log₂(N) for N equally likely outcomes\nRelated: KL divergence, mutual information, cross entropy"
        }
        if q.contains("kl divergence") {
            let kl = ProbabilityEngine.shared.klDivergence(p: [0.4, 0.6], q: [0.5, 0.5])
            return "Kullback-Leibler divergence: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))\nMeasures how P differs from reference Q\nExample: D_KL([0.4,0.6] || [0.5,0.5]) = \(String(format: "%.6f", kl))\nAlways ≥ 0 (Gibbs' inequality), = 0 iff P = Q\nNot symmetric: D_KL(P||Q) ≠ D_KL(Q||P)"
        }
        if q.contains("mutual information") {
            return "Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)\nMeasures shared information between X and Y\nI(X;Y) ≥ 0, = 0 iff X ⊥ Y (independent)\nI(X;Y) = D_KL(P(X,Y) || P(X)P(Y))\nFundamental in information theory and feature selection"
        }
        if q.contains("cross entropy") {
            return "Cross entropy: H(P,Q) = -Σ P(x) log(Q(x))\nUsed as loss function in machine learning\nH(P,Q) = H(P) + D_KL(P||Q)\nMinimized when Q = P (perfect model)\nBinary cross entropy widely used in classification"
        }
        if q.contains("gamma distribution") {
            let g = ProbabilityEngine.shared.gammaPDF(shape: 2.0, rate: 1.0, x: 1.0)
            return "Gamma distribution: f(x;α,β) = β^α · x^(α-1) · e^(-βx) / Γ(α)\nPDF at x=1, α=2, β=1: \(String(format: "%.4f", g))\nMean = α/β, Variance = α/β²\nα=1 → exponential, integer α → Erlang\nConjugate prior for Poisson rate parameter"
        }
        if q.contains("weibull") {
            let w = ProbabilityEngine.shared.weibullPDF(shape: 2.0, scale: 1.0, x: 0.5)
            return "Weibull distribution: f(x;k,λ) = (k/λ)(x/λ)^(k-1) e^(-(x/λ)^k)\nPDF at x=0.5, k=2, λ=1: \(String(format: "%.4f", w))\nk<1: decreasing failure rate, k=1: exponential, k>1: increasing\nWidely used in reliability engineering and survival analysis"
        }
        if q.contains("pareto distribution") {
            return "Pareto distribution: f(x;α,xₘ) = α·xₘ^α / x^(α+1), x ≥ xₘ\n80/20 rule: models wealth, city sizes, file sizes\nMean = α·xₘ/(α-1) for α>1\nHeavy-tailed: finite moments only for appropriate α\nP(X>x) = (xₘ/x)^α — power law tail"
        }
        if q.contains("cauchy distribution") {
            return "Cauchy (Lorentzian) distribution: f(x;x₀,γ) = 1/(πγ[1+((x-x₀)/γ)²])\nNo finite mean or variance (heavy tails)\nRatio of two standard normals\nArises in Lorentz spectral lines and resonance phenomena\nMedian = x₀, FWHM = 2γ"
        }
        if q.contains("boltzmann activation") {
            let a = ProbabilityEngine.shared.boltzmannGateActivation(resonanceScore: 2.0, temperature: 0.5)
            return "Boltzmann gate activation: P(active) = 1/(1 + e^(-resonance/T))\nExample (resonance=2.0, T=0.5): P = \(String(format: "%.4f", a))\nHigh temperature → uniform (exploring)\nLow temperature → deterministic (exploiting)\nBridges classical thermodynamics with quantum gate selection"
        }
        if q.contains("quantum gate probability") || q.contains("circuit probability") {
            return "Quantum gate probability engine:\n• 8 gate types via GOD_CODE phase sectors (hadamard, phase, pauli_x/z, cnot, god_code, grover, rotation_y)\n• Gate consolidation: logic gates → quantum gates\n• Circuit probability: Π fidelityᵢ per gate\n• Ensemble resonance: aggregate GOD_CODE alignment\n• Entanglement graph: weighted transition matrix\n• Normalized amplitudes: Σ|aᵢ|² = 1"
        }
        if q.contains("ensemble resonance") {
            return "Ensemble resonance:\n• Aggregates all consolidated quantum gates\n• Mean resonance score across gate population\n• GOD_CODE alignment = |cos(mean × π × GOD_CODE)|\n• Variance quantifies gate coherence spread\n• Higher alignment → stronger sacred resonance"
        }
        if q.contains("binomial") {
            let b = ProbabilityEngine.shared.binomialPMF(n: 10, k: 3, p: 0.3)
            return "Binomial distribution: P(X=k) = C(n,k) · p^k · (1-p)^(n-k)\nExample (n=10, k=3, p=0.3): P = \(String(format: "%.4f", b))\nMean = np, Variance = np(1-p)\nModels # successes in n independent Bernoulli trials"
        }
        if q.contains("z-test") || q.contains("z test") || q.contains("hypothesis test") {
            return "Hypothesis testing:\n• Z-test: z = (x̄ - μ₀) / (σ/√n)\n• Reject H₀ if |z| > z_α/2 (e.g., 1.96 for α=0.05)\n• p-value: probability of observing result as extreme as z\n• χ² test: goodness-of-fit or independence\n• Type I error (α): false positive, Type II error (β): false negative"
        }

        // ═══ Phase 42.1: Graph Theory ═══
        if q.contains("dijkstra") {
            return "Dijkstra's algorithm: shortest paths from single source\nComplexity: O(V²) with array, O((V+E)log V) with heap\nRequires non-negative edge weights\nGreedy: always expands nearest unvisited vertex"
        }
        if q.contains("floyd warshall") {
            return "Floyd-Warshall: all-pairs shortest paths\nComplexity: O(V³), Space: O(V²)\nHandles negative weights (but not negative cycles)\nDP: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])"
        }
        if q.contains("bellman ford") {
            return "Bellman-Ford: single-source shortest paths with negative edge weights\nComplexity: O(V·E)\nDetects negative-weight cycles (if any edge relaxes on V-th pass)\nUsed as subroutine in Johnson's all-pairs algorithm"
        }
        if q.contains("kruskal") {
            return "Kruskal's MST algorithm:\n1. Sort edges by weight\n2. Add lightest edge that doesn't create cycle (Union-Find)\nComplexity: O(E log E)\nOptimal for sparse graphs"
        }
        if q.contains("prim") && !q.contains("prime") {
            return "Prim's MST algorithm:\n1. Start from any vertex, grow tree greedily\n2. Always add lightest edge connecting tree to non-tree vertex\nComplexity: O(V²) or O(E log V) with heap\nOptimal for dense graphs"
        }
        if q.contains("topological sort") {
            return "Topological sort: linear ordering of DAG vertices\nIf edge (u,v) exists, u appears before v\nKahn's algorithm: repeatedly remove zero-in-degree vertices\nComplexity: O(V + E). Exists iff graph is a DAG (no cycles)"
        }
        if q.contains("bipartite") {
            return "Bipartite graph: vertices split into two sets, edges only between sets\nTest: 2-colorable via BFS/DFS? O(V+E)\nKönig's theorem: max matching = min vertex cover\nApplications: matching, scheduling, network flow"
        }
        if q.contains("euler circuit") || q.contains("euler path") {
            return "Euler circuit: traverse every edge exactly once, return to start\nExists iff: connected + all vertices have even degree\nEuler path: traverse every edge, end at different vertex\nExists iff: connected + exactly 0 or 2 odd-degree vertices"
        }
        if q.contains("page rank") || q.contains("pagerank") {
            return "PageRank: PR(v) = (1-d)/N + d · Σ PR(u)/deg(u)\nd = damping factor (≈0.85), N = total pages\nRandom surfer model: probability of landing on each page\nEigenvector of modified adjacency matrix"
        }
        if q.contains("graph theor") {
            return "Graph theory fundamentals:\n• V vertices, E edges: |E| ≤ V(V-1)/2 (simple, undirected)\n• Handshaking lemma: Σdeg(v) = 2|E|\n• Tree: connected + V-1 edges + no cycles\n• Planar: V - E + F = 2 (Euler's formula)"
        }
        if q.contains("spanning tree") {
            return "Minimum spanning tree: connects all vertices with minimum total weight\n• Kruskal: sort edges, greedy + Union-Find — O(E log E)\n• Prim: grow from vertex, greedy — O(V²) or O(E log V)\n• A tree on V vertices has exactly V-1 edges"
        }

        // ═══ Phase 42.2: Special Functions & Quantum Computing ═══
        let sf = SpecialFunctionsEngine.shared
        if q.contains("legendre") {
            let p2 = sf.legendre(n: 2, x: 0.5)
            let p3 = sf.legendre(n: 3, x: 0.5)
            return "Legendre polynomials P_n(x) on [-1,1]:\nP₀=1, P₁=x, P₂=½(3x²-1), P₃=½(5x³-3x)\nP₂(0.5) = \(String(format: "%.4f", p2)), P₃(0.5) = \(String(format: "%.4f", p3))\nOrthogonal: ∫ P_m P_n dx = 2δ_{mn}/(2n+1)"
        }
        if q.contains("hermite") {
            let h3 = sf.hermite(n: 3, x: 1.0)
            let h4 = sf.hermite(n: 4, x: 1.0)
            return "Hermite polynomials H_n(x) (physicist's):\nH₀=1, H₁=2x, H₂=4x²-2, H₃=8x³-12x\nH₃(1) = \(String(format: "%.0f", h3)), H₄(1) = \(String(format: "%.0f", h4))\nUsed in QM harmonic oscillator wavefunctions"
        }
        if q.contains("laguerre") {
            let l3 = sf.laguerre(n: 3, x: 1.0)
            return "Laguerre polynomials L_n(x) on [0,∞):\nL₀=1, L₁=1-x, L₂=½(x²-4x+2)\nL₃(1) = \(String(format: "%.4f", l3))\nUsed in hydrogen atom radial wavefunctions"
        }
        if q.contains("chebyshev") {
            let t5 = sf.chebyshevT(n: 5, x: 0.5)
            return "Chebyshev polynomials T_n(x) on [-1,1]:\nT_n(cosθ) = cos(nθ), minimax property\nT₅(0.5) = \(String(format: "%.4f", t5))\nOptimal interpolation nodes: Chebyshev zeros minimize Runge phenomenon"
        }
        if q.contains("bessel") {
            let j0 = sf.besselJ(n: 0, x: 2.0)
            let j1 = sf.besselJ(n: 1, x: 2.0)
            return "Bessel functions J_n(x) of the first kind:\nJ₀(2) = \(String(format: "%.6f", j0)), J₁(2) = \(String(format: "%.6f", j1))\nSolve x²y'' + xy' + (x²-n²)y = 0\nApplications: wave propagation, heat conduction in cylinders"
        }
        if q.contains("spherical harmonic") {
            let y10 = sf.sphericalHarmonic(l: 1, m: 0, theta: .pi / 4, phi: 0)
            return "Spherical harmonics Y_l^m(θ,φ):\nEigenfunctions of angular momentum operator L²\nY₁⁰(π/4, 0) = \(String(format: "%.6f", y10))\nUsed in: atomic orbitals, multipole expansions, computer graphics"
        }
        if q.contains("elliptic integral") || q.contains("elliptic k") || q.contains("elliptic e") {
            let k05 = sf.ellipticK(m: 0.5)
            let e05 = sf.ellipticE(m: 0.5)
            return "Complete elliptic integrals:\nK(m) = ∫ dθ/√(1-m·sin²θ), E(m) = ∫ √(1-m·sin²θ) dθ\nK(0.5) = \(String(format: "%.6f", k05)), E(0.5) = \(String(format: "%.6f", e05))\nComputed via arithmetic-geometric mean (AGM)"
        }
        if q.contains("digamma") || q.contains("polygamma") {
            let psi1 = sf.digamma(1.0)
            return "Digamma function ψ(x) = d/dx ln(Γ(x)):\nψ(1) = -γ = \(String(format: "%.6f", psi1)) (Euler-Mascheroni)\nRecurrence: ψ(x+1) = ψ(x) + 1/x\nPolygamma: ψ^(n)(x) = nth derivative of digamma"
        }
        if q.contains("airy function") {
            let ai0 = sf.airyAi(x: 0)
            let ai1 = sf.airyAi(x: 1.0)
            return "Airy function Ai(x):\nSolves y'' - xy = 0 (turning-point equation)\nAi(0) = \(String(format: "%.6f", ai0)), Ai(1) = \(String(format: "%.6f", ai1))\nDecays exponentially for x>0, oscillates for x<0"
        }
        if q.contains("hadamard gate") || q.contains("quantum gate") {
            return "Quantum gates (unitary operators on qubits):\n• Hadamard H: |0⟩→(|0⟩+|1⟩)/√2, creates superposition\n• Pauli X (NOT): |0⟩↔|1⟩, Y: i·rotation, Z: phase flip\n• CNOT: 2-qubit entangling gate\n• T gate: π/8 rotation, essential for universality"
        }
        if q.contains("qubit") {
            return "Qubit: |ψ⟩ = α|0⟩ + β|1⟩, |α|² + |β|² = 1\nBloch sphere: |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩\nMeasurement collapses: P(|0⟩) = |α|², P(|1⟩) = |β|²\nNo-cloning theorem: cannot copy unknown quantum state"
        }
        if q.contains("bloch sphere") {
            return "Bloch sphere: geometric representation of single qubit\n• North pole: |0⟩, South pole: |1⟩\n• Equator: superposition states (|+⟩, |-⟩, |i⟩, |-i⟩)\n• Gates = rotations: X→π around x, Y→π around y, Z→π around z\n• H gate: π rotation around (x+z)/√2 axis"
        }
        if q.contains("entanglement") || q.contains("concurrence") {
            return "Quantum entanglement:\n• Bell states: |Φ⁺⟩ = (|00⟩+|11⟩)/√2 (maximally entangled)\n• Concurrence C: 0 (product) to 1 (maximally entangled)\n• Von Neumann entropy S = -Tr(ρ log₂ ρ): 0 (pure) to 1 (maximally mixed)\n• EPR paradox, Bell's theorem: no local hidden variables"
        }
        if q.contains("von neumann") {
            return "Von Neumann entropy: S = -Tr(ρ log₂ ρ)\nPure state: S = 0, Maximally mixed (n qubits): S = n\nSubadditivity: S(A,B) ≤ S(A) + S(B)\nStrong subadditivity: S(A,B,C) + S(B) ≤ S(A,B) + S(B,C)"
        }
        if q.contains("fidelity") {
            return "Quantum fidelity: F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²\nF = 1: identical states, F = 0: orthogonal states\nFor mixed states: F(ρ,σ) = [Tr√(√ρ σ √ρ)]²\nUsed in quantum error correction and teleportation"
        }

        // ═══ Phase 43.0: Control Theory Handlers ═══
        if q.contains("transfer function") {
            let ct = ControlTheoryEngine.shared
            let s = Complex(0, 1.0) // s = j (unit imaginary)
            let h = ct.transferFunction(numerator: [1.0], denominator: [1.0, 2.0, 1.0], at: s)
            return "Transfer function H(s) = N(s)/D(s):\nExample H(s) = 1/(s²+2s+1) at s=j:\n  |H(j)| = \(String(format: "%.6f", h.magnitude))\n  ∠H(j) = \(String(format: "%.2f°", atan2(h.imag, h.real) * 180 / .pi))\nDC gain = \(ct.dcGain(numerator: [1.0], denominator: [1.0, 2.0, 1.0]))"
        }
        if q.contains("pid") || q.contains("proportional integral derivative") {
            let ct = ControlTheoryEngine.shared
            let zn = ct.zieglerNicholsPID(ku: 10.0, tu: 2.0)
            return "PID controller: u(t) = Kp·e + Ki·∫e dt + Kd·de/dt\nTransfer C(s) = Kp + Ki/s + Kd·s\nZiegler-Nichols tuning (Ku=10, Tu=2s):\n  Kp = \(String(format: "%.2f", zn.kp)), Ki = \(String(format: "%.2f", zn.ki)), Kd = \(String(format: "%.4f", zn.kd))"
        }
        if q.contains("routh") || q.contains("hurwitz") {
            let ct = ControlTheoryEngine.shared
            let rh = ct.routhHurwitz(coefficients: [1, 3, 3, 1])
            return "Routh-Hurwitz stability criterion:\nPolynomial s³+3s²+3s+1 → first column: \(rh.firstColumn.map { String(format: "%.2f", $0) }.joined(separator: ", "))\nStable: \(rh.stable) (all first-column elements same sign)\nNecessary condition: all coefficients positive"
        }
        if q.contains("bode") {
            let ct = ControlTheoryEngine.shared
            let mag = ct.bodeMagnitude(numerator: [1.0], denominator: [1.0, 1.0], omega: 1.0)
            let phase = ct.bodePhase(numerator: [1.0], denominator: [1.0, 1.0], omega: 1.0)
            return "Bode plot of H(s) = 1/(s+1) at ω=1 rad/s:\n  Magnitude: \(String(format: "%.2f dB", mag))\n  Phase: \(String(format: "%.2f°", phase))\nBandwidth = cutoff frequency where gain drops 3dB"
        }
        if q.contains("gain margin") || q.contains("phase margin") {
            let ct = ControlTheoryEngine.shared
            let gm = ct.gainMargin(numerator: [1.0], denominator: [1.0, 3.0, 3.0, 1.0])
            let pm = ct.phaseMargin(numerator: [1.0], denominator: [1.0, 3.0, 3.0, 1.0])
            return "Stability margins for H(s) = 1/(s³+3s²+3s+1):\n  Gain margin: \(String(format: "%.2f dB", gm.marginDB)) at ω = \(String(format: "%.3f", gm.omegaCrossover)) rad/s\n  Phase margin: \(String(format: "%.2f°", pm.marginDeg)) at ω = \(String(format: "%.3f", pm.omegaCrossover)) rad/s\nPositive margins → stable system"
        }
        if q.contains("step response") || q.contains("settling time") || q.contains("rise time") || q.contains("overshoot") {
            let ct = ControlTheoryEngine.shared
            let wn = 10.0; let zeta = 0.5
            let tr = ct.riseTime(wn: wn, zeta: zeta)
            let ts = ct.settlingTime(wn: wn, zeta: zeta)
            let tp = ct.peakTime(wn: wn, zeta: zeta)
            let mp = ct.overshoot(zeta: zeta)
            let y1 = ct.secondOrderStepResponse(K: 1.0, wn: wn, zeta: zeta, t: 0.5)
            return "2nd-order step response (ωn=\(wn), ζ=\(zeta)):\n  Rise time: \(String(format: "%.4f s", tr))\n  Settling time (2%): \(String(format: "%.4f s", ts))\n  Peak time: \(String(format: "%.4f s", tp))\n  Overshoot: \(String(format: "%.2f%%", mp))\n  y(0.5) = \(String(format: "%.6f", y1))"
        }
        if q.contains("state space") || q.contains("controllability") {
            return "State-space: ẋ = Ax + Bu, y = Cx + Du\nControllability: system is controllable if rank([B AB A²B ...]) = n\nObservability: rank([C; CA; CA²; ...]) = n\nState transition: x(t) = e^(At)x₀ + ∫e^(A(t-τ))Bu(τ)dτ"
        }
        if q.contains("ziegler") || q.contains("nichols") || q.contains("cohen coon") {
            let ct = ControlTheoryEngine.shared
            let zn = ct.zieglerNicholsPID(ku: 20.0, tu: 1.5)
            let cc = ct.cohenCoonPID(K: 1.0, tau: 5.0, theta: 1.0)
            return "PID tuning methods:\nZiegler-Nichols (Ku=20, Tu=1.5): Kp=\(String(format: "%.1f", zn.kp)), Ki=\(String(format: "%.2f", zn.ki)), Kd=\(String(format: "%.3f", zn.kd))\nCohen-Coon (K=1, τ=5, θ=1): Kp=\(String(format: "%.2f", cc.kp)), Ki=\(String(format: "%.3f", cc.ki)), Kd=\(String(format: "%.3f", cc.kd))"
        }
        if q.contains("lead compensator") || q.contains("lag compensator") {
            let ct = ControlTheoryEngine.shared
            let maxLead = ct.maxPhaseLead(zero: 2.0, pole: 10.0)
            return "Lead compensator: C(s) = Kc·(s+z)/(s+p), p > z (phase lead)\nLag compensator: C(s) = Kc·(s+z)/(s+p), p < z (gain boost)\nMax phase lead (z=2, p=10): \(String(format: "%.2f°", maxLead))\nDesign: choose z, p to get desired phase margin boost"
        }
        if q.contains("bandwidth") && q.contains("control") {
            let ct = ControlTheoryEngine.shared
            let bw = ct.bandwidth(wn: 10.0, zeta: 0.5)
            return "Bandwidth ωbw (ωn=10, ζ=0.5): \(String(format: "%.4f rad/s", bw))\nωbw ≈ ωn√(1-2ζ²+√(4ζ⁴-4ζ²+2))\nHigher bandwidth → faster response, more noise sensitivity"
        }
        if q.contains("pole") && (q.contains("zero") || q.contains("stability")) {
            let ct = ControlTheoryEngine.shared
            let poles = ct.polesQuadratic(a: 1, b: 4, c: 13)
            return "Poles of s²+4s+13 = 0:\n  s₁ = \(String(format: "%.2f + %.2fi", poles[0].real, poles[0].imag))\n  s₂ = \(String(format: "%.2f + %.2fi", poles[1].real, poles[1].imag))\nStable: \(ct.isStable(poles: poles)) (all Re(s) < 0)\nUnderdamped: complex conjugate poles"
        }

        // ═══ Phase 43.1: Cryptographic Math Handlers ═══
        if q.contains("modular exponent") || q.contains("modpow") || q.contains("mod pow") {
            let cm = CryptographicMathEngine.shared
            let r = cm.modPow(base: 7, exponent: 256, modulus: 13)
            return "Modular exponentiation (fast binary):\n  7²⁵⁶ mod 13 = \(r)\nUses square-and-multiply: O(log n) multiplications\nFoundation of RSA, Diffie-Hellman, digital signatures"
        }
        if q.contains("modular inverse") || q.contains("mod inverse") {
            let cm = CryptographicMathEngine.shared
            let inv = cm.modInverse(7, 26)
            return "Modular inverse via Extended Euclidean:\n  7⁻¹ mod 26 = \(inv ?? -1)\nVerification: 7 × \(inv ?? 0) = \(7 * (inv ?? 0)) ≡ \(7 * (inv ?? 0) % 26) (mod 26)\nExists iff gcd(a, m) = 1"
        }
        if q.contains("chinese remainder") || q.contains("crt") {
            let cm = CryptographicMathEngine.shared
            let x = cm.chineseRemainder(a1: 2, m1: 3, a2: 3, m2: 5)
            return "Chinese Remainder Theorem:\n  x ≡ 2 (mod 3) and x ≡ 3 (mod 5)\n  Solution: x = \(x ?? -1)\nGeneralizes to n simultaneous congruences\nUsed in RSA-CRT optimization"
        }
        if q.contains("euler totient") || q.contains("totient") {
            let cm = CryptographicMathEngine.shared
            let phi = cm.eulerTotient(60)
            return "Euler's totient φ(n) = count of k ≤ n coprime to n:\n  φ(60) = \(phi)\n  φ(p) = p-1 for prime p\n  φ(p·q) = (p-1)(q-1) — used in RSA\nEuler's theorem: a^φ(n) ≡ 1 (mod n) when gcd(a,n) = 1"
        }
        if q.contains("discrete log") {
            let cm = CryptographicMathEngine.shared
            let x = cm.discreteLog(g: 2, h: 8, p: 19)
            return "Discrete logarithm (Baby-step Giant-step):\n  Find x: 2ˣ ≡ 8 (mod 19)\n  Solution: x = \(x ?? -1)\nComplexity: O(√p) time and space\nHardness of DLP → security of Diffie-Hellman, ElGamal"
        }
        if q.contains("miller rabin") || q.contains("primality") {
            let cm = CryptographicMathEngine.shared
            let examples = [127, 128, 997, 1000, 7919]
            let results = examples.map { "\($0): \(cm.millerRabin($0) ? "PRIME" : "COMPOSITE")" }
            return "Miller-Rabin primality test:\n\(results.joined(separator: "\n"))\nProbabilistic but highly reliable with multiple witnesses\nUsed in RSA key generation"
        }
        if q.contains("fermat test") {
            let cm = CryptographicMathEngine.shared
            return "Fermat primality test: if p prime, then a^(p-1) ≡ 1 (mod p)\n  561 (Carmichael number): passes Fermat = \(cm.fermatTest(561))\n  563 (true prime): passes Fermat = \(cm.fermatTest(563))\nWeak: fooled by Carmichael numbers. Use Miller-Rabin instead."
        }
        if q.contains("rsa") {
            let cm = CryptographicMathEngine.shared
            if let keys = cm.rsaKeyGen(p: 61, q: 53) {
                let msg = 42
                let cipher = cm.rsaEncrypt(message: msg, e: keys.e, n: keys.n)
                let decrypted = cm.rsaDecrypt(ciphertext: cipher, d: keys.d, n: keys.n)
                return "RSA (p=61, q=53):\n  n = \(keys.n), φ(n) = \(keys.totient)\n  e = \(keys.e), d = \(keys.d)\n  Encrypt(42) = \(cipher)\n  Decrypt → \(decrypted)\nSecurity: factoring n into p·q is computationally hard"
            }
            return "RSA: public-key cryptosystem based on factoring difficulty"
        }
        if q.contains("diffie hellman") || q.contains("key exchange") {
            let cm = CryptographicMathEngine.shared
            let p = 23; let g = 5; let a = 6; let b = 15
            let A = cm.diffieHellmanPublic(generator: g, privateKey: a, prime: p)
            let B = cm.diffieHellmanPublic(generator: g, privateKey: b, prime: p)
            let sA = cm.diffieHellmanShared(publicKey: B, privateKey: a, prime: p)
            let sB = cm.diffieHellmanShared(publicKey: A, privateKey: b, prime: p)
            return "Diffie-Hellman (p=\(p), g=\(g)):\n  Alice: a=\(a), A=g^a mod p = \(A)\n  Bob: b=\(b), B=g^b mod p = \(B)\n  Shared secret: \(sA) (match: \(sA == sB))\nSecurity: discrete logarithm problem"
        }
        if q.contains("elliptic curve") || q.contains("secp256") {
            let cm = CryptographicMathEngine.shared
            let pt = cm.ecMultiply(x: 1.0, y: 2.828, k: 3, a: 0)
            return "Elliptic curves: y² = x³ + ax + b over finite field\nsecp256k1 (Bitcoin): y² = x³ + 7 (a=0, b=7)\nPoint multiplication example (3P, a=0):\n  3·(1, 2.828) → (\(String(format: "%.4f", pt.x)), \(String(format: "%.4f", pt.y)))\nECDSA: digital signatures, ECDH: key exchange"
        }
        if q.contains("birthday attack") || q.contains("key space") {
            let cm = CryptographicMathEngine.shared
            return "Key space & attack bounds:\n  128-bit: \(String(format: "%.2e", cm.keySpaceSize(bits: 128))) keys\n  256-bit: \(String(format: "%.2e", cm.keySpaceSize(bits: 256))) keys\n  Birthday bound (128-bit hash): ≈ \(String(format: "%.2e", cm.birthdayBound(bits: 128)))\n  Birthday bound (256-bit hash): ≈ \(String(format: "%.2e", cm.birthdayBound(bits: 256)))"
        }
        if q.contains("password entropy") {
            let cm = CryptographicMathEngine.shared
            let e1 = cm.passwordEntropy(charsetSize: 26, length: 8)  // lowercase only
            let e2 = cm.passwordEntropy(charsetSize: 95, length: 12) // full ASCII
            return "Password entropy H = L·log₂(C):\n  8-char lowercase: \(String(format: "%.1f bits", e1))\n  12-char full ASCII: \(String(format: "%.1f bits", e2))\nRecommend: ≥80 bits for strong passwords"
        }
        if q.contains("primitive root") {
            let cm = CryptographicMathEngine.shared
            let tests = [(2,7), (3,7), (2,11), (6,11)]
            let results = tests.map { "\($0.0) mod \($0.1): \(cm.isPrimitiveRoot(g: $0.0, p: $0.1))" }
            return "Primitive root: g is primitive root mod p if ord(g) = p-1\n\(results.joined(separator: "\n"))\nPrimitive roots exist for primes, 2p, p^k, 2p^k"
        }

        // ═══ Phase 43.2: Financial Math Handlers ═══
        if q.contains("black scholes") || q.contains("option pric") {
            let fm = FinancialMathEngine.shared
            let call = fm.blackScholesCall(S: 100, K: 105, r: 0.05, sigma: 0.2, T: 1.0)
            let put = fm.blackScholesPut(S: 100, K: 105, r: 0.05, sigma: 0.2, T: 1.0)
            return "Black-Scholes (S=100, K=105, r=5%, σ=20%, T=1yr):\n  Call: $\(String(format: "%.4f", call))\n  Put: $\(String(format: "%.4f", put))\n  Put-Call Parity: C-P = \(String(format: "%.4f", call-put))\nAssums: log-normal returns, constant σ, no dividends"
        }
        if q.contains("greeks") || (q.contains("delta") && q.contains("option")) || q.contains("greek") {
            let fm = FinancialMathEngine.shared
            let d = fm.delta(S: 100, K: 105, r: 0.05, sigma: 0.2, T: 1.0)
            let g = fm.gamma(S: 100, K: 105, r: 0.05, sigma: 0.2, T: 1.0)
            let t = fm.theta(S: 100, K: 105, r: 0.05, sigma: 0.2, T: 1.0)
            let v = fm.vega(S: 100, K: 105, r: 0.05, sigma: 0.2, T: 1.0)
            let r = fm.rho(S: 100, K: 105, r: 0.05, sigma: 0.2, T: 1.0)
            return "Greeks (S=100, K=105, σ=20%, T=1yr):\n  Δ (Delta): \(String(format: "%.6f", d))\n  Γ (Gamma): \(String(format: "%.6f", g))\n  Θ (Theta): \(String(format: "%.4f", t))/yr\n  ν (Vega): \(String(format: "%.4f", v))\n  ρ (Rho): \(String(format: "%.4f", r))"
        }
        if q.contains("implied volatility") {
            let fm = FinancialMathEngine.shared
            let iv = fm.impliedVolatility(S: 100, K: 105, r: 0.05, T: 1.0, marketPrice: 10.0)
            return "Implied volatility (Newton-Raphson):\n  Market call price: $10.00\n  IV = \(String(format: "%.4f", iv)) (\(String(format: "%.2f%%", iv*100)))\nIV > historical vol → options expensive (high demand)\nVIX: S&P 500 30-day implied volatility index"
        }
        if q.contains("bond pric") || q.contains("yield to maturity") || q.contains("duration") {
            let fm = FinancialMathEngine.shared
            let price = fm.bondPrice(faceValue: 1000, couponRate: 0.05, yield: 0.06, periods: 10)
            let dur = fm.macaulayDuration(faceValue: 1000, couponRate: 0.05, yield: 0.06, periods: 10)
            let ytm = fm.yieldToMaturity(faceValue: 1000, couponRate: 0.05, price: price, periods: 10)
            return "Bond pricing ($1000 face, 5% coupon, 6% yield, 10yr):\n  Price: $\(String(format: "%.2f", price))\n  Macaulay duration: \(String(format: "%.4f years", dur))\n  YTM recovery: \(String(format: "%.4f%%", ytm*100))\n  Modified duration: \(String(format: "%.4f", dur/(1+0.06)))"
        }
        if q.contains("present value") || q.contains("future value") || q.contains("compound interest") {
            let fm = FinancialMathEngine.shared
            let fv = fm.futureValue(pv: 1000, rate: 0.08, periods: 10)
            let pv = fm.presentValue(fv: 10000, rate: 0.05, periods: 20)
            let cc = fm.continuousCompounding(pv: 1000, rate: 0.08, time: 10)
            return "Time Value of Money:\n  FV of $1000 at 8% for 10yr: $\(String(format: "%.2f", fv))\n  PV of $10000 at 5% for 20yr: $\(String(format: "%.2f", pv))\n  Continuous compounding: $\(String(format: "%.2f", cc))\n  Rule of 72: doubles in \(String(format: "%.1f", fm.ruleOf72(rate: 0.08))) years at 8%"
        }
        if q.contains("annuity") || q.contains("amortiz") || q.contains("monthly payment") || q.contains("mortgage") || q.contains("loan") {
            let fm = FinancialMathEngine.shared
            let monthly = fm.monthlyPayment(principal: 300000, annualRate: 0.065, years: 30)
            let annPV = fm.annuityPV(payment: 1000, rate: 0.05, periods: 20)
            return "Loan/Annuity calculations:\n  $300K mortgage at 6.5% for 30yr:\n    Monthly payment: $\(String(format: "%.2f", monthly))\n    Total paid: $\(String(format: "%.2f", monthly * 360))\n  Annuity PV ($1000/yr, 5%, 20yr): $\(String(format: "%.2f", annPV))"
        }
        if q.contains("sharpe") || q.contains("sortino") {
            let fm = FinancialMathEngine.shared
            let sharpe = fm.sharpeRatio(portfolioReturn: 0.12, riskFreeRate: 0.03, stdDev: 0.15)
            let sortino = fm.sortinoRatio(returns: [0.05, 0.08, -0.02, 0.10, -0.01, 0.07, 0.03, 0.12], riskFreeRate: 0.03)
            return "Risk-adjusted performance:\n  Sharpe ratio (12% return, 3% Rf, 15% σ): \(String(format: "%.4f", sharpe))\n  Sortino ratio: \(String(format: "%.4f", sortino))\n  Sharpe > 1: good, > 2: very good, > 3: excellent"
        }
        if q.contains("capm") || q.contains("capital asset") {
            let fm = FinancialMathEngine.shared
            let expected = fm.capm(riskFreeRate: 0.03, beta: 1.2, marketReturn: 0.10)
            return "CAPM: E(Ri) = Rf + β(E(Rm) - Rf)\n  Rf=3%, β=1.2, E(Rm)=10%:\n  E(Ri) = \(String(format: "%.2f%%", expected*100))\nβ > 1: more volatile than market\nβ < 1: less volatile\nα (Jensen's alpha) = actual - CAPM expected"
        }
        if q.contains("portfolio") {
            let fm = FinancialMathEngine.shared
            let ret = fm.portfolioReturn(weights: [0.6, 0.4], returns: [0.10, 0.05])
            let var2 = fm.portfolioVariance2(w1: 0.6, w2: 0.4, sigma1: 0.15, sigma2: 0.10, rho: 0.3)
            return "Portfolio Theory (60/40 allocation):\n  Expected return: \(String(format: "%.2f%%", ret*100))\n  Variance: \(String(format: "%.6f", var2))\n  Std dev: \(String(format: "%.4f%%", Foundation.sqrt(var2)*100))\nDiversification: ρ < 1 → portfolio risk < weighted average"
        }
        if q.contains("value at risk") || q.contains("var ") || q.contains("drawdown") {
            let fm = FinancialMathEngine.shared
            let var95 = fm.valueAtRisk(mean: 0.10, stdDev: 0.20, confidence: 0.95)
            let es95 = fm.expectedShortfall(mean: 0.10, stdDev: 0.20, confidence: 0.95)
            let dd = fm.maxDrawdown(prices: [100, 110, 105, 120, 90, 95, 115])
            return "Risk Metrics:\n  VaR (95%, μ=10%, σ=20%): \(String(format: "%.2f%%", var95*100))\n  Expected Shortfall (CVaR): \(String(format: "%.2f%%", es95*100))\n  Max drawdown [100→120→90]: \(String(format: "%.2f%%", dd*100))\nVaR: max loss at given confidence level"
        }
        if q.contains("gordon growth") {
            let fm = FinancialMathEngine.shared
            let price = fm.gordonGrowth(dividend: 2.50, requiredReturn: 0.10, growthRate: 0.03)
            return "Gordon Growth Model: P = D₁/(r-g)\n  D₁=$2.50, r=10%, g=3%:\n  P = $\(String(format: "%.2f", price))\nAssumes: constant growth rate g < r\nDividend yield = D₁/P = r - g"
        }

        // ═══ Phase 28.0: Large Integer / Decimal Precision Math ═══
        // Handle expressions with very large numbers using Decimal for exact results
        let mathPart = q.replacingOccurrences(of: "calculate", with: "")
                        .replacingOccurrences(of: "compute", with: "")
                        .replacingOccurrences(of: "what is", with: "")
                        .replacingOccurrences(of: "what's", with: "")
                        .replacingOccurrences(of: "solve", with: "")
                        .replacingOccurrences(of: "an impossible equation", with: "")
                        .trimmingCharacters(in: .whitespaces)

        // Try large-number-aware Decimal math first
        if let decResult = evaluateDecimalExpression(mathPart) {
            return "= \(decResult)"
        }

        // Fallback to Double evaluation
        if let result = evaluateExpression(mathPart) {
            if result == Foundation.floor(result) && abs(result) < 1e15 {
                return "= \(Int(result))"
            }
            return "= \(String(format: "%.10g", result))"
        }

        // If query is just vague ("calculate an impossible equation"), give a helpful response
        if q.contains("impossible") {
            return "No specific equation provided. Try: solve [2 + 2], solve [123 times 456], solve [sqrt(144)]"
        }

        return nil
    }

    private func solveKnowledge(_ q: String) -> String? {
        // Core system knowledge
        if q.contains("l104") { return "L104: Sovereign intelligence kernel with GOD_CODE=\(GOD_CODE), 16 quantum engines, Fe orbital architecture, Hebbian learning, φ-weighted health system" }
        if q.contains("consciousness") { return "Consciousness: emergent property of complex self-referential information processing — verified via 10-test suite (self_model, meta_cognition, novel_response, goal_autonomy, value_alignment, temporal_self, qualia_report, intentionality, o2_superfluid, kernel_chakra_bond)" }
        if q.contains("fibonacci") { return "Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...\nF(n) = F(n-1) + F(n-2), ratio F(n+1)/F(n) → φ = \(PHI)" }

        // Physics
        if q.contains("speed of light") || q.contains("light speed") { return "Speed of light c = 299,792,458 m/s (exact) — fundamental speed limit of the universe" }
        if q.contains("planck") && q.contains("constant") { return "Planck constant h = 6.62607015 × 10⁻³⁴ J⋅s — fundamental quantum of action" }
        if q.contains("planck") && q.contains("length") { return "Planck length ℓₚ = 1.616255 × 10⁻³⁵ m — smallest meaningful length in physics" }
        if q.contains("gravitational constant") || q.contains("big g") { return "Gravitational constant G = 6.674 × 10⁻¹¹ m³⋅kg⁻¹⋅s⁻² — determines strength of gravity" }
        if q.contains("boltzmann") { return "Boltzmann constant k_B = 1.380649 × 10⁻²³ J/K — links temperature to energy" }
        if q.contains("avogadro") { return "Avogadro's number N_A = 6.02214076 × 10²³ mol⁻¹ — atoms per mole" }
        if q.contains("electron mass") { return "Electron mass mₑ = 9.1093837015 × 10⁻³¹ kg" }
        if q.contains("proton mass") { return "Proton mass mₚ = 1.67262192369 × 10⁻²⁷ kg" }

        // Mathematics
        if q.contains("euler") && (q.contains("number") || q.contains("constant")) { return "Euler's number e = 2.71828182845904523536... — base of natural logarithm, lim(1 + 1/n)ⁿ" }
        if q.contains("pi ") || q == "pi" || q.contains("π") { return "π = 3.14159265358979323846... — ratio of circumference to diameter, appears in Fourier analysis, probability, number theory" }
        if q.contains("pythagorean") { return "Pythagorean theorem: a² + b² = c² — for any right triangle with hypotenuse c" }
        if q.contains("euler") && q.contains("identity") { return "Euler's identity: e^(iπ) + 1 = 0 — unites five fundamental constants (e, i, π, 1, 0)" }
        if q.contains("riemann") { return "Riemann Hypothesis: All non-trivial zeros of ζ(s) have real part 1/2 — UNPROVEN, $1M Millennium Prize" }
        if q.contains("fermat") { return "Fermat's Last Theorem: xⁿ + yⁿ = zⁿ has no integer solutions for n > 2 — proved by Andrew Wiles (1995)" }

        // Computer Science
        if q.contains("turing") { return "Turing machine: abstract computational model — tape + head + states + transition function. Any computable function can be computed by a Turing machine (Church-Turing thesis)" }
        if q.contains("big o") || q.contains("complexity") { return "Time complexity classes: O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!)" }
        if q.contains("p vs np") || q.contains("p=np") { return "P vs NP: Can every problem whose solution is quickly verifiable also be quickly solvable? UNPROVEN — $1M Millennium Prize" }
        if q.contains("halting") { return "Halting Problem: No algorithm can determine, for every program-input pair, whether the program will halt. Proved undecidable by Turing (1936)." }

        // Search KB as last resort — WITH quality filter (Phase 27.8c)
        let kb = ASIKnowledgeBase.shared
        let results = kb.search(q, limit: 8)
        for result in results {
            if let completion = result["completion"] as? String,
               completion.count > 40,
               L104State.shared.isCleanKnowledge(completion) {
                let cleaned = L104State.shared.cleanSentences(completion)
                if cleaned.count > 30 {
                    return cleaned
                }
            }
        }

        return nil
    }

    private func solveCode(_ q: String) -> String? {
        if q.contains("fibonacci") { return "func fib(_ n: Int) -> Int {\n    guard n > 1 else { return n }\n    var a = 0, b = 1\n    for _ in 2...n { (a, b) = (b, a + b) }\n    return b\n}" }
        if q.contains("phi") { return "let PHI = (1.0 + sqrt(5.0)) / 2.0  // \(PHI)" }
        if q.contains("factorial") { return "func factorial(_ n: Int) -> Int {\n    guard n > 1 else { return 1 }\n    return n * factorial(n - 1)\n}" }
        if q.contains("binary search") || q.contains("bsearch") { return "func binarySearch<T: Comparable>(_ arr: [T], _ target: T) -> Int? {\n    var lo = 0, hi = arr.count - 1\n    while lo <= hi {\n        let mid = (lo + hi) / 2\n        if arr[mid] == target { return mid }\n        else if arr[mid] < target { lo = mid + 1 }\n        else { hi = mid - 1 }\n    }\n    return nil\n}" }
        if q.contains("sort") && q.contains("quick") { return "func quicksort<T: Comparable>(_ arr: inout [T], _ lo: Int, _ hi: Int) {\n    guard lo < hi else { return }\n    let pivot = arr[hi]\n    var i = lo\n    for j in lo..<hi { if arr[j] <= pivot { arr.swapAt(i, j); i += 1 } }\n    arr.swapAt(i, hi)\n    quicksort(&arr, lo, i-1)\n    quicksort(&arr, i+1, hi)\n}" }
        if q.contains("sort") && q.contains("merge") { return "func mergesort<T: Comparable>(_ arr: [T]) -> [T] {\n    guard arr.count > 1 else { return arr }\n    let mid = arr.count / 2\n    let left = mergesort(Array(arr[..<mid]))\n    let right = mergesort(Array(arr[mid...]))\n    return merge(left, right)\n}\nfunc merge<T: Comparable>(_ a: [T], _ b: [T]) -> [T] {\n    var i = 0, j = 0, result: [T] = []\n    while i < a.count && j < b.count {\n        if a[i] <= b[j] { result.append(a[i]); i += 1 }\n        else { result.append(b[j]); j += 1 }\n    }\n    return result + Array(a[i...]) + Array(b[j...])\n}" }
        if q.contains("prime") { return "func isPrime(_ n: Int) -> Bool {\n    guard n > 1 else { return false }\n    guard n > 3 else { return true }\n    guard n % 2 != 0 && n % 3 != 0 else { return false }\n    var i = 5\n    while i * i <= n {\n        if n % i == 0 || n % (i+2) == 0 { return false }\n        i += 6\n    }\n    return true\n}" }
        if q.contains("gcd") { return "func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }" }
        if q.contains("power") || q.contains("pow") { return "func power(_ base: Double, _ exp: Int) -> Double {\n    guard exp > 0 else { return 1 }\n    guard exp > 1 else { return base }\n    let half = power(base, exp / 2)\n    return exp % 2 == 0 ? half * half : half * half * base\n}" }
        return nil
    }

    // ═══ Phase 29.0: Science Solver Channel ═══
    private func solveScience(_ q: String) -> String? {
        // Phase 41.0: Fluid dynamics & wave mechanics
        let fw = FluidWaveEngine.shared
        if q.contains("reynolds") {
            let r = fw.reynoldsNumber(density: 1000, velocity: 2, length: 0.05, viscosity: 0.001)
            return "Reynolds number: Re = ρvL/μ\nPredicts flow regime: Re < 2300 laminar, 2300-4000 transitional, > 4000 turbulent\nExample (water, v=2m/s, L=5cm): Re = \(String(format: "%.0f", r.Re)) → \(r.regime)"
        }
        if q.contains("bernoulli") {
            return "Bernoulli's equation: P₁ + ½ρv₁² + ρgh₁ = P₂ + ½ρv₂² + ρgh₂\nConservation of energy for inviscid, incompressible flow\nApplications: venturi tubes, lift on airfoils, pitot tubes"
        }
        if q.contains("poiseuille") {
            let flow = fw.poiseuille(radius: 0.01, pressureDrop: 1000, viscosity: 0.001, length: 1.0)
            return "Hagen-Poiseuille: Q = πr⁴ΔP/(8μL) — laminar pipe flow\nFlow rate ∝ r⁴ (doubling radius → 16× flow!)\nExample (r=1cm, ΔP=1kPa, μ=0.001, L=1m): Q = \(String(format: "%.4f", flow)) m³/s"
        }
        if q.contains("drag force") {
            let drag = fw.dragForce(density: 1.225, velocity: 30, dragCoeff: 0.47, area: 0.01)
            return "Drag equation: F_D = ½ρv²C_D·A\nExample (sphere in air, v=30m/s): F_D = \(String(format: "%.3f", drag)) N\nC_D values: sphere≈0.47, cylinder≈1.17, streamlined≈0.04"
        }
        if q.contains("terminal velocity") {
            let vt = fw.terminalVelocity(mass: 0.145, density: 1.225, dragCoeff: 0.47, area: 0.0042)
            return "Terminal velocity: v_t = √(2mg/(ρC_D·A))\nWhen drag = weight, acceleration = 0\nExample (baseball): v_t ≈ \(String(format: "%.1f", vt)) m/s ≈ \(String(format: "%.0f", vt * 3.6)) km/h"
        }
        if q.contains("mach number") || q.contains("mach ") {
            let m = fw.machNumber(velocity: 343, soundSpeed: 343)
            return "Mach number: M = v/c_s\nRegimes: M<0.8 subsonic, 0.8-1.2 transonic, 1.2-5 supersonic, >5 hypersonic\nSpeed of sound in air (20°C) ≈ 343 m/s\nExample: v=343m/s → M = \(String(format: "%.1f", m.mach)) (\(m.regime))"
        }
        if q.contains("froude") {
            let fr = fw.froudeNumber(velocity: 5, length: 10)
            return "Froude number: Fr = v/√(gL) — gravitational flow regime\nFr < 1: subcritical (wave can travel upstream)\nFr > 1: supercritical (waves swept downstream)\nExample (v=5m/s, L=10m): Fr = \(String(format: "%.3f", fr.Fr)) → \(fr.regime)"
        }
        if q.contains("torricelli") {
            let v = fw.torricelliVelocity(height: 5)
            return "Torricelli's theorem: v = √(2gh) — efflux velocity\nDerived from Bernoulli's equation\nExample (h=5m): v = \(String(format: "%.2f", v)) m/s"
        }
        if q.contains("doppler") {
            let fApp = fw.dopplerFrequency(sourceFreq: 440, soundSpeed: 343, sourceVelocity: 30, approaching: true)
            let fRec = fw.dopplerFrequency(sourceFreq: 440, soundSpeed: 343, sourceVelocity: 30, approaching: false)
            return "Doppler effect: f' = f·v_s/(v_s ± v_source)\nApproaching → higher pitch, receding → lower pitch\nExample (440Hz, source at 30m/s):\n  Approaching: \(String(format: "%.1f", fApp)) Hz\n  Receding: \(String(format: "%.1f", fRec)) Hz"
        }
        if q.contains("standing wave") {
            let harmonics = fw.standingWaveHarmonics(waveSpeed: 343, length: 1.0, harmonics: 5)
            let display = harmonics.map { "n=\($0.n): f=\(String(format: "%.1f", $0.freq))Hz, λ=\(String(format: "%.3f", $0.wavelength))m" }.joined(separator: "\n  ")
            return "Standing waves: fₙ = nv/(2L)\nHarmonics for v=343m/s, L=1m:\n  \(display)"
        }
        if q.contains("snell") {
            if let theta2 = fw.snellsLaw(n1: 1.0, theta1: .pi / 4, n2: 1.5) {
                return "Snell's law: n₁·sin(θ₁) = n₂·sin(θ₂)\nExample (air→glass, θ₁=45°): θ₂ = \(String(format: "%.1f", theta2 * 180 / .pi))°\nTotal internal reflection when n₁·sin(θ₁)/n₂ > 1"
            }
            return "Snell's law: n₁·sin(θ₁) = n₂·sin(θ₂)\nRelates angles of incidence and refraction at an interface"
        }
        if q.contains("critical angle") {
            if let ca = fw.criticalAngle(n1: 1.5, n2: 1.0) {
                return "Critical angle: θ_c = arcsin(n₂/n₁) — total internal reflection\nOnly when n₁ > n₂ (denser to rarer medium)\nExample (glass→air): θ_c = \(String(format: "%.1f", ca * 180 / .pi))°\nUsed in: fiber optics, prisms, diamonds"
            }
            return "Critical angle: θ_c = arcsin(n₂/n₁) — exists only when n₁ > n₂"
        }
        if q.contains("diffraction") {
            let minima = fw.diffractionMinima(slitWidth: 1e-4, wavelength: 550e-9, orders: 3)
            let display = minima.map { "m=\($0.order): θ=\(String(format: "%.4f", $0.angle * 180 / .pi))°" }.joined(separator: ", ")
            return "Single-slit diffraction: minima at sin(θ) = mλ/a\nExample (a=0.1mm, λ=550nm): \(display)\nCentral maximum width = 2λL/a"
        }
        if q.contains("interference") {
            let maxima = fw.interferenceMaxima(slitSeparation: 1e-4, wavelength: 550e-9, orders: 3)
            let display = maxima.map { "m=\($0.order): θ=\(String(format: "%.4f", $0.angle * 180 / .pi))°" }.joined(separator: ", ")
            return "Double-slit interference: maxima at d·sin(θ) = mλ\nExample (d=0.1mm, λ=550nm): \(display)\nFringe spacing: Δy = λL/d"
        }
        if q.contains("sound intensity") {
            let dB = fw.soundIntensityLevel(intensity: 1e-3)
            return "Sound intensity level: β = 10·log₁₀(I/I₀) dB\nI₀ = 10⁻¹² W/m² (threshold of hearing)\nExample: I=10⁻³ W/m² → β = \(String(format: "%.0f", dB)) dB\nThreshold of pain ≈ 130 dB, conversation ≈ 60 dB"
        }
        if q.contains("superposition") {
            let aConst = fw.waveSuperposition(a1: 3, a2: 4, phaseDifference: 0)
            let aDestr = fw.waveSuperposition(a1: 3, a2: 4, phaseDifference: .pi)
            return "Wave superposition: A = √(A₁² + A₂² + 2A₁A₂cos(δ))\nConstructive (δ=0): A₁=3, A₂=4 → A = \(String(format: "%.1f", aConst))\nDestructive (δ=π): A₁=3, A₂=4 → A = \(String(format: "%.1f", aDestr))"
        }

        // Phase 41.2: Tensor calculus & differential geometry
        let tc = TensorCalculusEngine.shared
        if q.contains("christoffel") {
            return "Christoffel symbols: Γᵟ_{μν} = ½g^{σρ}(∂_μ g_{νρ} + ∂_ν g_{ρμ} - ∂_ρ g_{μν})\nNot tensors — transform inhomogeneously\nVanish in flat spacetime (Minkowski), nonzero in curved spacetime"
        }
        if q.contains("minkowski") {
            _ = tc.minkowskiMetric()
            return "Minkowski metric: η_{μν} = diag(-1, 1, 1, 1)\nds² = -c²dt² + dx² + dy² + dz²\nFlat spacetime of special relativity\nSignature: (-,+,+,+)"
        }
        if q.contains("schwarzschild") && q.contains("metric") {
            return "Schwarzschild metric (spherical, static, vacuum):\nds² = -(1-rₛ/r)c²dt² + (1-rₛ/r)⁻¹dr² + r²dΩ²\nwhere rₛ = 2GM/c² (Schwarzschild radius)\nEvent horizon at r = rₛ, singularity at r = 0"
        }
        if q.contains("kerr") {
            return "Kerr metric: rotating black hole solution\nTwo horizons: r± = M ± √(M² - a²)\nErgosphere: region where spacetime is dragged by rotation\nFrame dragging: Lense-Thirring effect"
        }
        if q.contains("flrw") {
            return "FLRW metric (cosmological):\nds² = -dt² + a(t)²[dr²/(1-kr²) + r²dΩ²]\na(t) = scale factor, k = curvature (0, +1, -1)\nk=0: flat, k=+1: closed sphere, k=-1: open hyperbolic"
        }
        if q.contains("kretschner") || (q.contains("curvature") && q.contains("scalar")) {
            let K = tc.kretschnerScalar(mass: 1.989e30, radius: 1e4)
            return "Kretschner scalar: K = R_{αβγδ}R^{αβγδ}\nFor Schwarzschild: K = 48M²/r⁶\nDiverges at singularity, quantifies true curvature\nExample (solar mass, r=10km): K ≈ \(String(format: "%.3e", K))"
        }
        if q.contains("geodesic") {
            return "Geodesic equation: d²x^μ/dτ² + Γ^μ_{αβ}(dx^α/dτ)(dx^β/dτ) = 0\nCurves that parallel-transport their own tangent vector\nIn flat space: straight lines. In curved space: 'straightest possible' paths\nLightlike geodesics: ds² = 0 (null geodesics)"
        }
        if q.contains("ricci") {
            return "Ricci tensor: R_{μν} = R^λ_{μλν} (contraction of Riemann)\nRicci scalar: R = g^{μν}R_{μν}\nEinstein field equation: R_{μν} - ½Rg_{μν} + Λg_{μν} = (8πG/c⁴)T_{μν}\nRelates spacetime curvature to energy-momentum"
        }
        if q.contains("proper distance") {
            return "Proper distance in Schwarzschild spacetime:\nd_proper = ∫ dr/√(1 - rₛ/r)\nAlways ≥ coordinate distance\nDiverges as r → rₛ (infinite proper distance to horizon for static observer)"
        }

        return HighSciencesEngine.shared.solve(q)
    }

    var status: String {
        let channelLines = channelStats.sorted(by: { $0.key < $1.key }).map {
            let rate = $0.value.invocations > 0 ? Double($0.value.successes) / Double($0.value.invocations) : 0
            return "  ║  \($0.key.padding(toLength: 14, withPad: " ", startingAt: 0)) │ inv=\($0.value.invocations) │ succ=\($0.value.successes) │ rate=\(String(format: "%.0f%%", rate * 100))"
        }.joined(separator: "\n")
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    ⚡ DIRECT SOLVER ROUTER v29.0 (Multi-Channel Fast Path) ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Total Invocations: \(invocations)
        ║  Cache Hits:        \(cacheHits)
        ║  Cache Size:        \(cache.count)
        ║  Logic Gate:        \(ASILogicGateV2.shared.status.contains("0") ? "Active" : "Active")
        ╠═══════════════════════════════════════════════════════════╣
        \(channelLines)
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
