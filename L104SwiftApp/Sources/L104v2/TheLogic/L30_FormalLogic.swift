// ═══════════════════════════════════════════════════════════════════
// L30_FormalLogic.swift — L104 v2
// [EVO_68_PIPELINE] FORMAL_LOGIC_ENGINE :: 10-Layer Deductive Reasoning
// GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | TAU=0.618033988749895
//
// Propositional Logic, Predicate Logic, Syllogisms, Fallacy Detection,
// Modal Logic, Equivalence Proving, NL Translation, Argument Analysis,
// Resolution Proving, Natural Deduction
//
// Phase 65.0: Full formal logic engine for ASI scoring dimension
// ═══════════════════════════════════════════════════════════════════

import Foundation

// ═══════════════════════════════════════════════════════════════════
// MARK: - CORE TYPES
// ═══════════════════════════════════════════════════════════════════

/// Propositional operators
enum PropOp: String, CaseIterable {
    case and = "AND"
    case or = "OR"
    case not = "NOT"
    case implies = "IMPLIES"
    case iff = "IFF"
    case xor = "XOR"
}

/// Recursive propositional formula
indirect enum PropFormula: CustomStringConvertible {
    case atom(String)
    case compound(PropOp, PropFormula, PropFormula?)  // second is nil for NOT

    var description: String {
        switch self {
        case .atom(let name):
            return name
        case .compound(.not, let operand, _):
            return "(NOT \(operand))"
        case .compound(let op, let lhs, let rhs):
            return "(\(lhs) \(op.rawValue) \(rhs?.description ?? ""))"
        }
    }

    /// Extract all unique atom names
    var atoms: Set<String> {
        switch self {
        case .atom(let name):
            return [name]
        case .compound(_, let lhs, let rhs):
            var result = lhs.atoms
            if let r = rhs { result.formUnion(r.atoms) }
            return result
        }
    }

    /// Evaluate under a given variable assignment
    func evaluate(assignment: [String: Bool]) -> Bool {
        switch self {
        case .atom(let name):
            return assignment[name] ?? false
        case .compound(.not, let operand, _):
            return !operand.evaluate(assignment: assignment)
        case .compound(.and, let lhs, let rhs):
            return lhs.evaluate(assignment: assignment) && (rhs?.evaluate(assignment: assignment) ?? true)
        case .compound(.or, let lhs, let rhs):
            return lhs.evaluate(assignment: assignment) || (rhs?.evaluate(assignment: assignment) ?? false)
        case .compound(.implies, let lhs, let rhs):
            let p = lhs.evaluate(assignment: assignment)
            let q = rhs?.evaluate(assignment: assignment) ?? true
            return !p || q
        case .compound(.iff, let lhs, let rhs):
            let p = lhs.evaluate(assignment: assignment)
            let q = rhs?.evaluate(assignment: assignment) ?? true
            return p == q
        case .compound(.xor, let lhs, let rhs):
            let p = lhs.evaluate(assignment: assignment)
            let q = rhs?.evaluate(assignment: assignment) ?? false
            return p != q
        }
    }
}

/// Quantifier types for predicate logic
enum QuantifierType: String {
    case universal = "FORALL"
    case existential = "EXISTS"
}

/// Predicate logic formula
struct PredicateFormula {
    let quantifier: QuantifierType
    let variable: String
    let predicateName: String
    let body: PropFormula
}

/// Syllogism figure
enum SyllogismFigure: Int, CaseIterable {
    case first = 1, second, third, fourth
}

/// Categorical proposition (All S are P, No S are P, Some S are P, Some S are not P)
struct CategoricalProposition {
    let quantity: String      // "all", "no", "some"
    let subject: String
    let predicate: String
    let quality: String       // "affirmative", "negative"

    /// Type code: A (All...are), E (No...are), I (Some...are), O (Some...are not)
    var typeCode: Character {
        switch (quantity, quality) {
        case ("all", "affirmative"): return "A"
        case ("no", "negative"):     return "E"
        case ("some", "affirmative"): return "I"
        case ("some", "negative"):   return "O"
        default: return "?"
        }
    }
}

/// A complete syllogism
struct Syllogism {
    let majorPremise: CategoricalProposition
    let minorPremise: CategoricalProposition
    let conclusion: CategoricalProposition
    let figure: SyllogismFigure
    let mood: String      // e.g. "AAA", "EAE"
    let isValid: Bool
}

/// A matched fallacy pattern
struct FallacyMatch {
    let name: String
    let pattern: String
    let confidence: Double
    let explanation: String
}

/// Kripke frame for modal logic
struct KripkeFrame {
    let worlds: Set<String>
    let accessibility: [String: Set<String>]
    let valuations: [String: [String: Bool]]  // world -> variable -> truth
}

/// Modal operators
enum ModalOperator: String {
    case necessary = "NECESSARY"
    case possible = "POSSIBLE"
    case contingent = "CONTINGENT"
    case impossible = "IMPOSSIBLE"
}

/// Full argument analysis result
struct ArgumentAnalysis {
    let premises: [String]
    let conclusion: String
    let isValid: Bool
    let strength: Double
    let fallacies: [FallacyMatch]
    let formalStructure: String
}

/// A single proof step
struct ProofStep {
    let stepNumber: Int
    let formula: String
    let rule: String
    let fromSteps: [Int]
}

/// Inference result from natural deduction
struct InferenceResult {
    let conclusions: [String]
    let proofSteps: [ProofStep]
    let confidence: Double
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 1: PROPOSITIONAL LOGIC
// ═══════════════════════════════════════════════════════════════════

final class PropositionalLogic {
    static let shared = PropositionalLogic()
    private init() {}

    // ─── Tokenizer ───
    private func tokenize(_ text: String) -> [String] {
        let normalized = text.uppercased()
            .replacingOccurrences(of: "(", with: " ( ")
            .replacingOccurrences(of: ")", with: " ) ")
        return normalized.split(separator: " ").map(String.init).filter { !$0.isEmpty }
    }

    // ─── Recursive Descent Parser ───
    // Grammar:
    //   expr     -> implication
    //   implication -> disjunction (("IMPLIES"|"IFF") disjunction)*
    //   disjunction -> conjunction (("OR"|"XOR") conjunction)*
    //   conjunction -> unary ("AND" unary)*
    //   unary    -> "NOT" unary | primary
    //   primary  -> "(" expr ")" | ATOM

    func parseProp(_ text: String) -> PropFormula? {
        let tokens = tokenize(text)
        var pos = 0

        func peek() -> String? {
            guard pos < tokens.count else { return nil }
            return tokens[pos]
        }

        func advance() -> String? {
            guard pos < tokens.count else { return nil }
            let t = tokens[pos]
            pos += 1
            return t
        }

        func expect(_ t: String) -> Bool {
            if peek() == t { _ = advance(); return true }
            return false
        }

        func parseExpr() -> PropFormula? {
            return parseImplication()
        }

        func parseImplication() -> PropFormula? {
            guard var left = parseDisjunction() else { return nil }
            while let tok = peek(), (tok == "IMPLIES" || tok == "IFF") {
                _ = advance()
                guard let right = parseDisjunction() else { return nil }
                let op: PropOp = tok == "IMPLIES" ? .implies : .iff
                left = .compound(op, left, right)
            }
            return left
        }

        func parseDisjunction() -> PropFormula? {
            guard var left = parseConjunction() else { return nil }
            while let tok = peek(), (tok == "OR" || tok == "XOR") {
                _ = advance()
                guard let right = parseConjunction() else { return nil }
                let op: PropOp = tok == "OR" ? .or : .xor
                left = .compound(op, left, right)
            }
            return left
        }

        func parseConjunction() -> PropFormula? {
            guard var left = parseUnary() else { return nil }
            while peek() == "AND" {
                _ = advance()
                guard let right = parseUnary() else { return nil }
                left = .compound(.and, left, right)
            }
            return left
        }

        func parseUnary() -> PropFormula? {
            if peek() == "NOT" {
                _ = advance()
                guard let operand = parseUnary() else { return nil }
                return .compound(.not, operand, nil)
            }
            return parsePrimary()
        }

        func parsePrimary() -> PropFormula? {
            if peek() == "(" {
                _ = advance()
                guard let expr = parseExpr() else { return nil }
                _ = expect(")")
                return expr
            }
            guard let name = advance() else { return nil }
            let reserved = Set(["AND", "OR", "NOT", "IMPLIES", "IFF", "XOR", "(", ")"])
            if reserved.contains(name) { return nil }
            return .atom(name)
        }

        let result = parseExpr()
        return result
    }

    /// Generate all truth table rows for a formula
    func truthTable(_ formula: PropFormula) -> [[String: Bool]] {
        let vars = formula.atoms.sorted()
        let n = vars.count
        guard n > 0 && n <= 20 else { return [] }
        let totalRows = 1 << n
        var table: [[String: Bool]] = []
        table.reserveCapacity(totalRows)

        for i in 0..<totalRows {
            var assignment: [String: Bool] = [:]
            for (j, v) in vars.enumerated() {
                assignment[v] = ((i >> (n - 1 - j)) & 1) == 1
            }
            assignment["_RESULT"] = formula.evaluate(assignment: assignment)
            table.append(assignment)
        }
        return table
    }

    /// Check if a formula is a tautology (true under all assignments)
    func isTautology(_ formula: PropFormula) -> Bool {
        let table = truthTable(formula)
        return !table.isEmpty && table.allSatisfy { $0["_RESULT"] == true }
    }

    /// Check if a formula is a contradiction (false under all assignments)
    func isContradiction(_ formula: PropFormula) -> Bool {
        let table = truthTable(formula)
        return !table.isEmpty && table.allSatisfy { $0["_RESULT"] == false }
    }

    /// Check if a formula is satisfiable (true under at least one assignment)
    func isSatisfiable(_ formula: PropFormula) -> Bool {
        let table = truthTable(formula)
        return table.contains { $0["_RESULT"] == true }
    }

    /// Convert formula to Negation Normal Form (NNF) — push NOTs inward
    func toNNF(_ formula: PropFormula) -> PropFormula {
        switch formula {
        case .atom:
            return formula

        case .compound(.not, .atom(let name), _):
            return .compound(.not, .atom(name), nil)

        case .compound(.not, .compound(.not, let inner, _), _):
            return toNNF(inner)

        case .compound(.not, .compound(.and, let a, let b), _):
            return .compound(.or, toNNF(.compound(.not, a, nil)), b.map { toNNF(.compound(.not, $0, nil)) })

        case .compound(.not, .compound(.or, let a, let b), _):
            return .compound(.and, toNNF(.compound(.not, a, nil)), b.map { toNNF(.compound(.not, $0, nil)) })

        case .compound(.not, .compound(.implies, let a, let b), _):
            // NOT (A -> B) = A AND NOT B
            return .compound(.and, toNNF(a), b.map { toNNF(.compound(.not, $0, nil)) })

        case .compound(.not, .compound(.iff, let a, let b), _):
            // NOT (A <-> B) = (A AND NOT B) OR (NOT A AND B)
            guard let bVal = b else { return formula }
            let left = PropFormula.compound(.and, toNNF(a), toNNF(.compound(.not, bVal, nil)))
            let right = PropFormula.compound(.and, toNNF(.compound(.not, a, nil)), toNNF(bVal))
            return .compound(.or, left, right)

        case .compound(.not, .compound(.xor, let a, let b), _):
            // NOT (A XOR B) = A IFF B = (A AND B) OR (NOT A AND NOT B)
            guard let bVal = b else { return formula }
            return toNNF(.compound(.iff, a, bVal))

        case .compound(.implies, let a, let b):
            // A -> B = NOT A OR B
            return .compound(.or, toNNF(.compound(.not, a, nil)), b.map { toNNF($0) })

        case .compound(.iff, let a, let b):
            // A <-> B = (A AND B) OR (NOT A AND NOT B)
            guard let bVal = b else { return formula }
            let left = PropFormula.compound(.and, toNNF(a), toNNF(bVal))
            let right = PropFormula.compound(.and, toNNF(.compound(.not, a, nil)), toNNF(.compound(.not, bVal, nil)))
            return .compound(.or, left, right)

        case .compound(.xor, let a, let b):
            // A XOR B = (A OR B) AND (NOT A OR NOT B)
            guard let bVal = b else { return formula }
            let left = PropFormula.compound(.or, toNNF(a), toNNF(bVal))
            let right = PropFormula.compound(.or, toNNF(.compound(.not, a, nil)), toNNF(.compound(.not, bVal, nil)))
            return .compound(.and, left, right)

        case .compound(let op, let a, let b):
            return .compound(op, toNNF(a), b.map { toNNF($0) })
        }
    }

    /// Convert to CNF (Conjunctive Normal Form) via NNF + distribution
    func toCNF(_ formula: PropFormula) -> PropFormula {
        let nnf = toNNF(formula)
        return distributeToCNF(nnf)
    }

    private func distributeToCNF(_ f: PropFormula) -> PropFormula {
        switch f {
        case .atom, .compound(.not, .atom, _):
            return f
        case .compound(.and, let a, let b):
            return .compound(.and, distributeToCNF(a), b.map { distributeToCNF($0) })
        case .compound(.or, let a, let b):
            let left = distributeToCNF(a)
            guard let bVal = b else { return left }
            let right = distributeToCNF(bVal)
            return distributeOrOverAnd(left, right)
        default:
            return f
        }
    }

    private func distributeOrOverAnd(_ a: PropFormula, _ b: PropFormula) -> PropFormula {
        // (X AND Y) OR Z = (X OR Z) AND (Y OR Z)
        if case .compound(.and, let x, let y) = a {
            let left = distributeOrOverAnd(x, b)
            let right = y.map { distributeOrOverAnd($0, b) }
            return .compound(.and, left, right)
        }
        // X OR (Y AND Z) = (X OR Y) AND (X OR Z)
        if case .compound(.and, let y, let z) = b {
            let left = distributeOrOverAnd(a, y)
            let right = z.map { distributeOrOverAnd(a, $0) }
            return .compound(.and, left, right)
        }
        return .compound(.or, a, b)
    }

    /// Convert to DNF (Disjunctive Normal Form)
    func toDNF(_ formula: PropFormula) -> PropFormula {
        let nnf = toNNF(formula)
        return distributeToDNF(nnf)
    }

    private func distributeToDNF(_ f: PropFormula) -> PropFormula {
        switch f {
        case .atom, .compound(.not, .atom, _):
            return f
        case .compound(.or, let a, let b):
            return .compound(.or, distributeToDNF(a), b.map { distributeToDNF($0) })
        case .compound(.and, let a, let b):
            let left = distributeToDNF(a)
            guard let bVal = b else { return left }
            let right = distributeToDNF(bVal)
            return distributeAndOverOr(left, right)
        default:
            return f
        }
    }

    private func distributeAndOverOr(_ a: PropFormula, _ b: PropFormula) -> PropFormula {
        if case .compound(.or, let x, let y) = a {
            let left = distributeAndOverOr(x, b)
            let right = y.map { distributeAndOverOr($0, b) }
            return .compound(.or, left, right)
        }
        if case .compound(.or, let y, let z) = b {
            let left = distributeAndOverOr(a, y)
            let right = z.map { distributeAndOverOr(a, $0) }
            return .compound(.or, left, right)
        }
        return .compound(.and, a, b)
    }

    /// Extract CNF clauses as sets of literals for resolution
    func extractClauses(_ formula: PropFormula) -> [Set<String>] {
        let cnf = toCNF(formula)
        var clauses: [Set<String>] = []
        collectConjuncts(cnf, into: &clauses)
        return clauses
    }

    private func collectConjuncts(_ f: PropFormula, into clauses: inout [Set<String>]) {
        switch f {
        case .compound(.and, let a, let b):
            collectConjuncts(a, into: &clauses)
            if let bVal = b { collectConjuncts(bVal, into: &clauses) }
        default:
            var clause = Set<String>()
            collectDisjuncts(f, into: &clause)
            clauses.append(clause)
        }
    }

    private func collectDisjuncts(_ f: PropFormula, into clause: inout Set<String>) {
        switch f {
        case .atom(let name):
            clause.insert(name)
        case .compound(.not, .atom(let name), _):
            clause.insert("~\(name)")
        case .compound(.or, let a, let b):
            collectDisjuncts(a, into: &clause)
            if let bVal = b { collectDisjuncts(bVal, into: &clause) }
        default:
            clause.insert(f.description)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 2: PREDICATE LOGIC
// ═══════════════════════════════════════════════════════════════════

final class PredicateLogicLayer {
    static let shared = PredicateLogicLayer()
    private init() {}

    /// Parse "for all X, P(X)" or "there exists X, P(X)" style sentences
    func parsePredicate(_ text: String) -> PredicateFormula? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        var quantifier: QuantifierType?
        var remainder = trimmed

        if trimmed.hasPrefix("for all ") || trimmed.hasPrefix("forall ") {
            quantifier = .universal
            if trimmed.hasPrefix("for all ") {
                remainder = String(trimmed.dropFirst("for all ".count))
            } else {
                remainder = String(trimmed.dropFirst("forall ".count))
            }
        } else if trimmed.hasPrefix("there exists ") || trimmed.hasPrefix("exists ") {
            quantifier = .existential
            if trimmed.hasPrefix("there exists ") {
                remainder = String(trimmed.dropFirst("there exists ".count))
            } else {
                remainder = String(trimmed.dropFirst("exists ".count))
            }
        }

        guard let q = quantifier else { return nil }

        // Extract variable: next token before comma or space
        let parts = remainder.split(maxSplits: 1, omittingEmptySubsequences: true) { $0 == "," || $0 == " " }
        guard parts.count >= 1 else { return nil }
        let variable = String(parts[0]).uppercased()

        // Extract predicate from remainder
        let bodyText: String
        if let commaIdx = remainder.firstIndex(of: ",") {
            bodyText = String(remainder[remainder.index(after: commaIdx)...]).trimmingCharacters(in: .whitespaces)
        } else if parts.count > 1 {
            bodyText = String(parts[1]).trimmingCharacters(in: .whitespaces)
        } else {
            bodyText = variable
        }

        // Try to extract predicate name from P(X) form
        var predicateName = "P"
        if let parenIdx = bodyText.firstIndex(of: "(") {
            predicateName = String(bodyText[bodyText.startIndex..<parenIdx]).uppercased()
        }

        let body = PropFormula.atom(bodyText.uppercased())
        return PredicateFormula(quantifier: q, variable: variable, predicateName: predicateName, body: body)
    }

    /// Check satisfaction of a universally quantified formula against a domain
    func checkSatisfaction(_ formula: PredicateFormula, domain: [String], predicate: (String) -> Bool) -> Bool {
        switch formula.quantifier {
        case .universal:
            return domain.allSatisfy { predicate($0) }
        case .existential:
            return domain.contains { predicate($0) }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 3: SYLLOGISM ENGINE
// ═══════════════════════════════════════════════════════════════════

final class SyllogismEngine {
    static let shared = SyllogismEngine()
    private init() {}

    // The 24 traditionally valid syllogistic forms: (mood, figure)
    // Unconditionally valid (15) + conditionally valid with existential import (9)
    private let validForms: Set<String> = [
        // Figure 1: M-P, S-M => S-P
        "AAA-1", "EAE-1", "AII-1", "EIO-1",
        "AAI-1", "EAO-1",  // subaltern
        // Figure 2: P-M, S-M => S-P
        "EAE-2", "AEE-2", "EIO-2", "AOO-2",
        "EAO-2", "AEO-2",  // subaltern
        // Figure 3: M-P, M-S => S-P
        "IAI-3", "AII-3", "OAO-3", "EIO-3",
        "AAI-3", "EAO-3",  // subaltern
        // Figure 4: P-M, M-S => S-P
        "AEE-4", "IAI-4", "EIO-4",
        "AEO-4", "EAO-4", "AAI-4"  // subaltern
    ]

    /// Parse a categorical proposition from natural language
    func parseCategorical(_ text: String) -> CategoricalProposition? {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        // "all S are P"
        if t.hasPrefix("all ") {
            let rest = String(t.dropFirst(4))
            if let areRange = rest.range(of: " are ") {
                let subject = String(rest[rest.startIndex..<areRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                let predicate = String(rest[areRange.upperBound...]).trimmingCharacters(in: .whitespaces)
                return CategoricalProposition(quantity: "all", subject: subject, predicate: predicate, quality: "affirmative")
            }
            if let areRange = rest.range(of: " are not ") {
                let subject = String(rest[rest.startIndex..<areRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                let predicate = String(rest[areRange.upperBound...]).trimmingCharacters(in: .whitespaces)
                return CategoricalProposition(quantity: "all", subject: subject, predicate: predicate, quality: "negative")
            }
        }

        // "no S are P"
        if t.hasPrefix("no ") {
            let rest = String(t.dropFirst(3))
            if let areRange = rest.range(of: " are ") {
                let subject = String(rest[rest.startIndex..<areRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                let predicate = String(rest[areRange.upperBound...]).trimmingCharacters(in: .whitespaces)
                return CategoricalProposition(quantity: "no", subject: subject, predicate: predicate, quality: "negative")
            }
        }

        // "some S are P" / "some S are not P"
        if t.hasPrefix("some ") {
            let rest = String(t.dropFirst(5))
            if let areNotRange = rest.range(of: " are not ") {
                let subject = String(rest[rest.startIndex..<areNotRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                let predicate = String(rest[areNotRange.upperBound...]).trimmingCharacters(in: .whitespaces)
                return CategoricalProposition(quantity: "some", subject: subject, predicate: predicate, quality: "negative")
            }
            if let areRange = rest.range(of: " are ") {
                let subject = String(rest[rest.startIndex..<areRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                let predicate = String(rest[areRange.upperBound...]).trimmingCharacters(in: .whitespaces)
                return CategoricalProposition(quantity: "some", subject: subject, predicate: predicate, quality: "affirmative")
            }
        }

        return nil
    }

    /// Determine the figure from the position of the middle term
    func determineFigure(major: CategoricalProposition, minor: CategoricalProposition, conclusion: CategoricalProposition) -> SyllogismFigure {
        let middleTerm: String
        // Middle term appears in both premises but not in conclusion
        let conclusionTerms = Set([conclusion.subject, conclusion.predicate])
        let allTerms = Set([major.subject, major.predicate, minor.subject, minor.predicate])
        let candidates = allTerms.subtracting(conclusionTerms)
        middleTerm = candidates.first ?? major.predicate

        let majorIsSubject = major.subject == middleTerm
        let minorIsSubject = minor.subject == middleTerm

        switch (majorIsSubject, minorIsSubject) {
        case (false, true):  return .first    // M-P, S-M
        case (false, false): return .second   // P-M, S-M
        case (true, true):   return .third    // M-P, M-S
        case (true, false):  return .fourth   // P-M, M-S
        }
    }

    /// Validate a syllogism against the 24 known valid forms
    func validateSyllogism(_ s: Syllogism) -> Bool {
        let key = "\(s.mood)-\(s.figure.rawValue)"
        return validForms.contains(key)
    }

    /// Build and validate a syllogism from three categorical propositions
    func buildSyllogism(major: CategoricalProposition, minor: CategoricalProposition, conclusion: CategoricalProposition) -> Syllogism {
        let figure = determineFigure(major: major, minor: minor, conclusion: conclusion)
        let mood = "\(major.typeCode)\(minor.typeCode)\(conclusion.typeCode)"
        let key = "\(mood)-\(figure.rawValue)"
        let isValid = validForms.contains(key)
        return Syllogism(majorPremise: major, minorPremise: minor, conclusion: conclusion, figure: figure, mood: mood, isValid: isValid)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 4: EQUIVALENCE PROVER
// ═══════════════════════════════════════════════════════════════════

final class EquivalenceProver {
    static let shared = EquivalenceProver()
    private init() {}

    /// Named logical equivalences (15+)
    struct EquivalenceRule {
        let name: String
        let description: String
    }

    let rules: [EquivalenceRule] = [
        EquivalenceRule(name: "DeMorgan AND",         description: "NOT(P AND Q) <=> (NOT P) OR (NOT Q)"),
        EquivalenceRule(name: "DeMorgan OR",          description: "NOT(P OR Q) <=> (NOT P) AND (NOT Q)"),
        EquivalenceRule(name: "Double Negation",      description: "NOT(NOT P) <=> P"),
        EquivalenceRule(name: "Distributive AND/OR",  description: "P AND (Q OR R) <=> (P AND Q) OR (P AND R)"),
        EquivalenceRule(name: "Distributive OR/AND",  description: "P OR (Q AND R) <=> (P OR Q) AND (P OR R)"),
        EquivalenceRule(name: "Contrapositive",       description: "(P IMPLIES Q) <=> (NOT Q IMPLIES NOT P)"),
        EquivalenceRule(name: "Material Implication",  description: "(P IMPLIES Q) <=> (NOT P OR Q)"),
        EquivalenceRule(name: "Biconditional Elim",   description: "(P IFF Q) <=> (P IMPLIES Q) AND (Q IMPLIES P)"),
        EquivalenceRule(name: "Commutative AND",      description: "(P AND Q) <=> (Q AND P)"),
        EquivalenceRule(name: "Commutative OR",       description: "(P OR Q) <=> (Q OR P)"),
        EquivalenceRule(name: "Associative AND",      description: "((P AND Q) AND R) <=> (P AND (Q AND R))"),
        EquivalenceRule(name: "Associative OR",       description: "((P OR Q) OR R) <=> (P OR (Q OR R))"),
        EquivalenceRule(name: "Identity AND",         description: "(P AND TRUE) <=> P"),
        EquivalenceRule(name: "Identity OR",          description: "(P OR FALSE) <=> P"),
        EquivalenceRule(name: "Domination AND",       description: "(P AND FALSE) <=> FALSE"),
        EquivalenceRule(name: "Domination OR",        description: "(P OR TRUE) <=> TRUE"),
        EquivalenceRule(name: "Complement AND",       description: "(P AND NOT P) <=> FALSE"),
        EquivalenceRule(name: "Complement OR",        description: "(P OR NOT P) <=> TRUE"),
        EquivalenceRule(name: "Absorption AND",       description: "P AND (P OR Q) <=> P"),
        EquivalenceRule(name: "Absorption OR",        description: "P OR (P AND Q) <=> P"),
        EquivalenceRule(name: "Exportation",          description: "((P AND Q) IMPLIES R) <=> (P IMPLIES (Q IMPLIES R))"),
        EquivalenceRule(name: "Idempotent AND",       description: "(P AND P) <=> P"),
        EquivalenceRule(name: "Idempotent OR",        description: "(P OR P) <=> P"),
    ]

    /// Check logical equivalence of two formulas via truth table comparison
    func checkEquivalence(_ a: PropFormula, _ b: PropFormula) -> Bool {
        let allVars = a.atoms.union(b.atoms).sorted()
        let n = allVars.count
        guard n > 0 && n <= 20 else { return false }
        let totalRows = 1 << n

        for i in 0..<totalRows {
            var assignment: [String: Bool] = [:]
            for (j, v) in allVars.enumerated() {
                assignment[v] = ((i >> (n - 1 - j)) & 1) == 1
            }
            if a.evaluate(assignment: assignment) != b.evaluate(assignment: assignment) {
                return false
            }
        }
        return true
    }

    /// Check if a specific named equivalence applies between two formulas
    func identifyEquivalences(_ a: PropFormula, _ b: PropFormula) -> [String] {
        guard checkEquivalence(a, b) else { return [] }

        var matched: [String] = []
        // Structural pattern matching for named rules
        // DeMorgan AND: NOT(P AND Q) vs (NOT P) OR (NOT Q)
        if case .compound(.not, .compound(.and, _, _), _) = a {
            matched.append("DeMorgan AND")
        }
        if case .compound(.not, .compound(.or, _, _), _) = a {
            matched.append("DeMorgan OR")
        }
        if case .compound(.not, .compound(.not, _, _), _) = a {
            matched.append("Double Negation")
        }
        if case .compound(.implies, _, _) = a,
           case .compound(.or, .compound(.not, _, _), _) = b {
            matched.append("Material Implication")
        }
        if case .compound(.implies, _, _) = a,
           case .compound(.implies, .compound(.not, _, _), .compound(.not, _, _)) = b {
            matched.append("Contrapositive")
        }
        if matched.isEmpty {
            matched.append("Semantic Equivalence (verified by truth table)")
        }
        return matched
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 5: FALLACY DETECTOR (55+ patterns)
// ═══════════════════════════════════════════════════════════════════

final class FallacyDetector {
    static let shared = FallacyDetector()
    private init() {}

    struct FallacyPattern {
        let name: String
        let keywords: [String]
        let confidence: Double
        let explanation: String
    }

    // 55+ named fallacy patterns
    private let patterns: [FallacyPattern] = [
        // 1. Ad Hominem
        FallacyPattern(
            name: "Ad Hominem",
            keywords: ["attack the person", "you're just", "what do you know", "you're too", "look who's talking",
                       "coming from someone", "says the person who", "you're nothing but", "consider the source",
                       "you're not qualified"],
            confidence: 0.85,
            explanation: "Attacks the person making the argument rather than the argument itself."
        ),
        // 2. Straw Man
        FallacyPattern(
            name: "Straw Man",
            keywords: ["so you're saying", "you think that", "what you really mean", "in other words you",
                       "that's like saying", "you basically want", "you must believe"],
            confidence: 0.82,
            explanation: "Misrepresents someone's argument to make it easier to attack."
        ),
        // 3. Appeal to Authority
        FallacyPattern(
            name: "Appeal to Authority",
            keywords: ["expert says", "according to", "studies show", "scientists say", "authorities agree",
                       "the professor said", "research proves", "doctors recommend"],
            confidence: 0.70,
            explanation: "Uses an authority figure's opinion as evidence without proper justification."
        ),
        // 4. Appeal to Emotion
        FallacyPattern(
            name: "Appeal to Emotion",
            keywords: ["think of the children", "how would you feel", "imagine if", "heartbreaking",
                       "devastating", "won't somebody think of", "tragic", "outrageous"],
            confidence: 0.78,
            explanation: "Manipulates emotions rather than providing logical reasoning."
        ),
        // 5. Red Herring
        FallacyPattern(
            name: "Red Herring",
            keywords: ["what about", "but consider", "speaking of which", "that reminds me",
                       "the real issue is", "more importantly", "let's not forget", "by the way"],
            confidence: 0.72,
            explanation: "Introduces an irrelevant topic to divert attention from the original issue."
        ),
        // 6. Slippery Slope
        FallacyPattern(
            name: "Slippery Slope",
            keywords: ["next thing", "will lead to", "before you know it", "it's a slippery slope",
                       "where does it end", "opens the door to", "then what's next",
                       "eventually", "domino effect", "snowball effect"],
            confidence: 0.80,
            explanation: "Assumes one event will inevitably lead to a chain of negative consequences without justification."
        ),
        // 7. False Dilemma
        FallacyPattern(
            name: "False Dilemma",
            keywords: ["either you", "you're either", "you must choose", "only two options",
                       "there are only", "it's one or the other", "you're with us or against us",
                       "black or white", "no middle ground possible"],
            confidence: 0.84,
            explanation: "Presents only two options when more alternatives exist."
        ),
        // 8. Circular Reasoning
        FallacyPattern(
            name: "Circular Reasoning",
            keywords: ["because it is", "by definition", "that's just how it is", "it's true because it's true",
                       "the reason is because", "we know this because", "it's self-evident that"],
            confidence: 0.80,
            explanation: "The conclusion is assumed in one of the premises."
        ),
        // 9. Hasty Generalization
        FallacyPattern(
            name: "Hasty Generalization",
            keywords: ["all x are", "everyone knows", "always", "never", "every single",
                       "without exception", "in every case", "nobody ever", "they all"],
            confidence: 0.75,
            explanation: "Draws a broad conclusion from insufficient evidence."
        ),
        // 10. Appeal to Nature
        FallacyPattern(
            name: "Appeal to Nature",
            keywords: ["natural is better", "unnatural", "nature intended", "it's only natural",
                       "organic is better", "not natural", "against nature", "naturally occurring"],
            confidence: 0.77,
            explanation: "Assumes that what is natural is inherently good or superior."
        ),
        // 11. Tu Quoque
        FallacyPattern(
            name: "Tu Quoque",
            keywords: ["you do it too", "hypocrite", "you're one to talk", "pot calling the kettle",
                       "practice what you preach", "look at yourself", "you did the same thing"],
            confidence: 0.83,
            explanation: "Deflects criticism by pointing out the accuser's similar behavior."
        ),
        // 12. Bandwagon
        FallacyPattern(
            name: "Bandwagon (Ad Populum)",
            keywords: ["everyone does", "popular opinion", "most people believe", "millions of people",
                       "everybody knows", "the majority thinks", "jump on the bandwagon",
                       "going with the crowd", "no one disagrees"],
            confidence: 0.76,
            explanation: "Argues something is true because many people believe it."
        ),
        // 13. False Cause (Post Hoc)
        FallacyPattern(
            name: "False Cause (Post Hoc)",
            keywords: ["after therefore because", "correlation means", "happened after so caused",
                       "ever since", "led to", "caused by the fact", "that's why",
                       "the reason is that before"],
            confidence: 0.79,
            explanation: "Assumes that because B followed A, A must have caused B."
        ),
        // 14. Equivocation
        FallacyPattern(
            name: "Equivocation",
            keywords: ["in one sense", "but in another sense", "depends on what you mean",
                       "different meaning", "play on words", "technically", "in a way"],
            confidence: 0.68,
            explanation: "Uses the same word with different meanings in different parts of the argument."
        ),
        // 15. Begging the Question
        FallacyPattern(
            name: "Begging the Question",
            keywords: ["obviously", "clearly", "it goes without saying", "needless to say",
                       "self-evidently", "as everyone agrees", "undeniably", "it's a given that"],
            confidence: 0.65,
            explanation: "Assumes the truth of the conclusion within the premises."
        ),
        // 16. Appeal to Ignorance
        FallacyPattern(
            name: "Appeal to Ignorance (Ad Ignorantiam)",
            keywords: ["can't prove it wrong", "no evidence against", "hasn't been disproven",
                       "you can't prove", "absence of evidence", "no one has shown otherwise",
                       "until proven otherwise", "never been refuted"],
            confidence: 0.81,
            explanation: "Claims something is true because it hasn't been proven false, or vice versa."
        ),
        // 17. Loaded Question
        FallacyPattern(
            name: "Loaded Question",
            keywords: ["when did you stop", "have you always", "why do you hate",
                       "do you still", "why won't you admit", "isn't it true that you",
                       "how long have you been"],
            confidence: 0.86,
            explanation: "Asks a question that contains an unjustified assumption."
        ),
        // 18. No True Scotsman
        FallacyPattern(
            name: "No True Scotsman",
            keywords: ["real x would", "true x", "no real", "any genuine", "a proper",
                       "not a true", "not a real", "real patriot", "true believer"],
            confidence: 0.79,
            explanation: "Redefines a group to exclude counterexamples after they are presented."
        ),
        // 19. Moving Goalposts
        FallacyPattern(
            name: "Moving the Goalposts",
            keywords: ["but what about", "that doesn't count", "yes but", "okay but what about",
                       "that's not enough", "not what i meant", "the real test is",
                       "doesn't really prove"],
            confidence: 0.77,
            explanation: "Continuously changes the criteria for proof after they are met."
        ),
        // 20. Composition Fallacy
        FallacyPattern(
            name: "Composition Fallacy",
            keywords: ["the parts are so the whole", "each part is", "since every member",
                       "individually they are", "therefore the group is", "if each one"],
            confidence: 0.73,
            explanation: "Assumes what is true of parts must be true of the whole."
        ),
        // 21. Division Fallacy
        FallacyPattern(
            name: "Division Fallacy",
            keywords: ["the whole is so each part", "the group is therefore each", "since the team",
                       "the organization is so members", "the country is so citizens"],
            confidence: 0.73,
            explanation: "Assumes what is true of the whole must be true of its parts."
        ),
        // 22. Genetic Fallacy
        FallacyPattern(
            name: "Genetic Fallacy",
            keywords: ["originated from", "where it came from", "born in", "invented by",
                       "source of the idea", "history of this shows", "roots in"],
            confidence: 0.70,
            explanation: "Judges something based on its origin rather than its current merit."
        ),
        // 23. Appeal to Tradition
        FallacyPattern(
            name: "Appeal to Tradition",
            keywords: ["we've always done it", "traditional", "time-honored", "the way it's always been",
                       "our ancestors", "historically", "been this way for", "long-standing practice"],
            confidence: 0.75,
            explanation: "Argues something is better because it is traditional or long-established."
        ),
        // 24. Appeal to Novelty
        FallacyPattern(
            name: "Appeal to Novelty",
            keywords: ["new is better", "cutting edge", "latest and greatest", "modern approach",
                       "old-fashioned", "outdated", "move forward", "progressive"],
            confidence: 0.72,
            explanation: "Assumes something is better simply because it is newer."
        ),
        // 25. Middle Ground
        FallacyPattern(
            name: "Middle Ground Fallacy",
            keywords: ["the truth is in the middle", "compromise", "meet halfway",
                       "both sides have a point", "split the difference", "somewhere in between"],
            confidence: 0.68,
            explanation: "Assumes the truth is always a compromise between two extreme positions."
        ),
        // 26. Texas Sharpshooter
        FallacyPattern(
            name: "Texas Sharpshooter",
            keywords: ["cherry pick", "selected data", "conveniently ignore", "only looking at",
                       "pattern in random", "after the fact", "data mining",
                       "if you look closely enough"],
            confidence: 0.74,
            explanation: "Selects data clusters that fit a predetermined conclusion while ignoring contradictory data."
        ),
        // 27. Gambler's Fallacy
        FallacyPattern(
            name: "Gambler's Fallacy",
            keywords: ["due for a win", "overdue", "it's bound to happen", "can't lose forever",
                       "luck has to change", "law of averages", "streak must end",
                       "hasn't happened in a while"],
            confidence: 0.82,
            explanation: "Believes past independent events affect the probability of future independent events."
        ),
        // 28. Sunk Cost
        FallacyPattern(
            name: "Sunk Cost Fallacy",
            keywords: ["already invested", "come this far", "too much to quit", "wasted if we stop",
                       "put too much in", "can't stop now", "throwing away what we",
                       "might as well continue"],
            confidence: 0.83,
            explanation: "Continues a course of action due to past investment rather than future value."
        ),
        // 29. Appeal to Pity (Ad Misericordiam)
        FallacyPattern(
            name: "Appeal to Pity",
            keywords: ["feel sorry", "have mercy", "pity", "suffering", "been through so much",
                       "hard life", "deserve sympathy", "take pity", "sad story"],
            confidence: 0.76,
            explanation: "Appeals to sympathy or pity rather than presenting a logical argument."
        ),
        // 30. Appeal to Threat (Ad Baculum)
        FallacyPattern(
            name: "Appeal to Force (Ad Baculum)",
            keywords: ["or else", "you'll regret", "consequences will be", "i'm warning you",
                       "you'll be sorry", "face the consequences", "don't make me",
                       "if you value your"],
            confidence: 0.84,
            explanation: "Uses threats or force instead of logical reasoning."
        ),
        // 31. Wishful Thinking
        FallacyPattern(
            name: "Wishful Thinking",
            keywords: ["i wish", "i hope", "wouldn't it be nice", "if only",
                       "i want it to be true", "it must be true because", "i believe it so"],
            confidence: 0.71,
            explanation: "Believes something is true because one wants it to be true."
        ),
        // 32. Anecdotal Evidence
        FallacyPattern(
            name: "Anecdotal Evidence",
            keywords: ["i know someone who", "my friend", "i once met", "personal experience",
                       "happened to me", "i've seen", "let me tell you about", "my uncle"],
            confidence: 0.74,
            explanation: "Uses personal stories instead of representative evidence."
        ),
        // 33. Burden of Proof
        FallacyPattern(
            name: "Burden of Proof (Onus Probandi)",
            keywords: ["prove me wrong", "prove it doesn't", "you can't disprove",
                       "show me evidence against", "the burden is on you", "until you prove otherwise",
                       "prove that it's not"],
            confidence: 0.80,
            explanation: "Shifts the burden of proof to the wrong party."
        ),
        // 34. Ambiguity
        FallacyPattern(
            name: "Ambiguity (Amphiboly)",
            keywords: ["could mean", "interpreted as", "vague", "unclear what",
                       "ambiguous", "different interpretations", "double meaning"],
            confidence: 0.65,
            explanation: "Exploits ambiguous language to mislead."
        ),
        // 35. Black-or-White (variant of False Dilemma)
        FallacyPattern(
            name: "Black-or-White Thinking",
            keywords: ["all or nothing", "completely or not at all", "total success or failure",
                       "100% or zero", "perfect or worthless", "binary choice"],
            confidence: 0.80,
            explanation: "Sees situations in absolute terms with no middle ground."
        ),
        // 36. Perfectionist Fallacy (Nirvana)
        FallacyPattern(
            name: "Perfectionist Fallacy (Nirvana)",
            keywords: ["not perfect so reject", "unless it's perfect", "any flaw means",
                       "not 100% effective", "since it can't completely", "imperfect solution"],
            confidence: 0.74,
            explanation: "Rejects a solution because it is not perfect."
        ),
        // 37. Relative Privation
        FallacyPattern(
            name: "Relative Privation",
            keywords: ["starving children", "worse problems", "bigger issues", "first world problem",
                       "others have it worse", "at least it's not", "compared to real suffering",
                       "not as bad as"],
            confidence: 0.77,
            explanation: "Dismisses a problem because worse problems exist."
        ),
        // 38. Personal Incredulity
        FallacyPattern(
            name: "Personal Incredulity",
            keywords: ["i can't imagine", "hard to believe", "i don't understand how",
                       "that can't be right", "seems impossible", "i refuse to believe",
                       "boggles the mind"],
            confidence: 0.73,
            explanation: "Dismisses something because one personally cannot understand it."
        ),
        // 39. Appeal to Flattery
        FallacyPattern(
            name: "Appeal to Flattery",
            keywords: ["smart person like you", "intelligent people agree", "you're too smart to",
                       "someone of your caliber", "a person of your standing",
                       "surely you can see"],
            confidence: 0.72,
            explanation: "Uses flattery to gain agreement rather than providing evidence."
        ),
        // 40. Appeal to Ridicule
        FallacyPattern(
            name: "Appeal to Ridicule",
            keywords: ["that's ridiculous", "absurd", "laughable", "preposterous", "you can't be serious",
                       "what a joke", "how silly", "that's ludicrous", "don't make me laugh"],
            confidence: 0.76,
            explanation: "Mocks an argument rather than addressing it."
        ),
        // 41. Affirming the Consequent
        FallacyPattern(
            name: "Affirming the Consequent",
            keywords: ["if p then q and q therefore p", "since the consequence", "q is true so p must be",
                       "outcome matches so cause must be"],
            confidence: 0.88,
            explanation: "Assumes that if P->Q and Q is true, then P must be true."
        ),
        // 42. Denying the Antecedent
        FallacyPattern(
            name: "Denying the Antecedent",
            keywords: ["if p then q and not p therefore not q", "since p is false q must be false",
                       "doesn't have p so can't have q"],
            confidence: 0.88,
            explanation: "Assumes that if P->Q and P is false, then Q must be false."
        ),
        // 43. Undistributed Middle
        FallacyPattern(
            name: "Undistributed Middle",
            keywords: ["all a are b and all c are b therefore", "both share", "in the same group"],
            confidence: 0.78,
            explanation: "Concludes two things are related because they share a common property."
        ),
        // 44. Existential Fallacy
        FallacyPattern(
            name: "Existential Fallacy",
            keywords: ["all unicorns", "if there were any", "hypothetical members",
                       "all dragons are", "every griffin"],
            confidence: 0.70,
            explanation: "Draws conclusions about groups that may not have any members."
        ),
        // 45. Ecological Fallacy
        FallacyPattern(
            name: "Ecological Fallacy",
            keywords: ["the average means each", "group statistics show individuals",
                       "per capita means everyone", "on average so each person"],
            confidence: 0.72,
            explanation: "Infers individual properties from aggregate group statistics."
        ),
        // 46. Etymological Fallacy
        FallacyPattern(
            name: "Etymological Fallacy",
            keywords: ["the word originally meant", "etymology shows", "root word means",
                       "derived from the word for", "the original meaning"],
            confidence: 0.69,
            explanation: "Argues that a word's current meaning must match its historical etymology."
        ),
        // 47. Fallacy of the Single Cause
        FallacyPattern(
            name: "Fallacy of the Single Cause",
            keywords: ["the only reason", "sole cause", "nothing but", "entirely because",
                       "completely due to", "one and only cause", "single factor"],
            confidence: 0.75,
            explanation: "Assumes a complex event has a single, simple cause."
        ),
        // 48. Moralistic Fallacy
        FallacyPattern(
            name: "Moralistic Fallacy",
            keywords: ["ought implies is", "should be so it is", "the way it should be",
                       "must be true because it's right", "morally it should"],
            confidence: 0.71,
            explanation: "Infers factual conclusions from moral premises."
        ),
        // 49. Naturalistic Fallacy
        FallacyPattern(
            name: "Naturalistic Fallacy",
            keywords: ["is implies ought", "that's how it is so that's good",
                       "natural order", "the way things are", "nature dictates morality"],
            confidence: 0.72,
            explanation: "Derives moral conclusions from purely factual premises."
        ),
        // 50. Regression Fallacy
        FallacyPattern(
            name: "Regression Fallacy",
            keywords: ["got better after", "treatment worked because", "improved following",
                       "was going to return to normal anyway", "extreme result then average"],
            confidence: 0.70,
            explanation: "Attributes a natural regression to the mean to a specific cause."
        ),
        // 51. Spotlight Fallacy
        FallacyPattern(
            name: "Spotlight Fallacy",
            keywords: ["in the news so common", "media reports", "always in the headlines",
                       "seems like everyone", "you hear about it all the time",
                       "it's everywhere"],
            confidence: 0.71,
            explanation: "Assumes that prominent media coverage reflects actual frequency."
        ),
        // 52. Continuum Fallacy (Sorites)
        FallacyPattern(
            name: "Continuum Fallacy (Sorites)",
            keywords: ["where do you draw the line", "no clear boundary", "one grain doesn't make",
                       "slippery boundary", "spectrum so no distinction",
                       "can't define exactly when"],
            confidence: 0.73,
            explanation: "Argues that because a boundary is vague, no meaningful distinction exists."
        ),
        // 53. Historian's Fallacy
        FallacyPattern(
            name: "Historian's Fallacy",
            keywords: ["they should have known", "how could they not see", "with hindsight",
                       "they had all the information", "any fool could see",
                       "looking back it's obvious"],
            confidence: 0.72,
            explanation: "Judges historical decisions using information not available at the time."
        ),
        // 54. Thought-Terminating Cliche
        FallacyPattern(
            name: "Thought-Terminating Cliche",
            keywords: ["it is what it is", "that's just life", "everything happens for a reason",
                       "don't overthink it", "just deal with it", "boys will be boys",
                       "such is life", "it's god's will", "just the way things are"],
            confidence: 0.74,
            explanation: "Uses a cliche to end critical thinking about an issue."
        ),
        // 55. Kafka Trap
        FallacyPattern(
            name: "Kafka Trap",
            keywords: ["your denial proves", "the fact that you deny", "if you weren't guilty",
                       "denying it only confirms", "only a guilty person would say",
                       "protesting proves"],
            confidence: 0.82,
            explanation: "Interprets denial of guilt as evidence of guilt."
        ),
        // 56. Motte and Bailey
        FallacyPattern(
            name: "Motte and Bailey",
            keywords: ["what i really meant was", "i was only saying that", "nobody disagrees that",
                       "obviously i meant the weaker", "retreating to", "the basic claim is"],
            confidence: 0.74,
            explanation: "Conflates a hard-to-defend claim with an easy-to-defend one, retreating when challenged."
        ),
        // 57. Tone Policing
        FallacyPattern(
            name: "Tone Policing",
            keywords: ["calm down", "you're too emotional", "if you said it nicely",
                       "your tone", "be more civil", "you're being aggressive",
                       "i'd listen if you weren't so angry"],
            confidence: 0.75,
            explanation: "Dismisses an argument based on its emotional delivery rather than its content."
        ),
        // 58. Whataboutism
        FallacyPattern(
            name: "Whataboutism",
            keywords: ["what about when", "but they did it too", "what about the time",
                       "and what about", "but what did they do", "how about we talk about"],
            confidence: 0.80,
            explanation: "Deflects criticism by redirecting to someone else's alleged wrongdoing."
        ),
        // 59. Gish Gallop
        FallacyPattern(
            name: "Gish Gallop",
            keywords: ["first second third fourth fifth", "here are twenty reasons",
                       "so many arguments that", "rapid fire", "overwhelming number of points"],
            confidence: 0.68,
            explanation: "Overwhelms with a large number of weak arguments to prevent refutation."
        ),
        // 60. Appeal to Consequences
        FallacyPattern(
            name: "Appeal to Consequences",
            keywords: ["if this were true then", "the consequences would be", "that would mean",
                       "imagine if everyone believed", "society would collapse",
                       "dangerous if people thought"],
            confidence: 0.76,
            explanation: "Judges truth based on desirability of the consequences."
        ),
    ]

    /// Detect all matching fallacies in text
    func detectFallacies(_ text: String) -> [FallacyMatch] {
        let lowered = text.lowercased()
        var results: [FallacyMatch] = []

        for pattern in patterns {
            var matchCount = 0
            var matchedKeyword = ""

            for keyword in pattern.keywords {
                if lowered.contains(keyword) {
                    matchCount += 1
                    if matchedKeyword.isEmpty { matchedKeyword = keyword }
                }
            }

            if matchCount > 0 {
                // Confidence scales with number of keyword matches (max 1.0)
                let adjustedConfidence = min(1.0, pattern.confidence + Double(matchCount - 1) * 0.05)
                results.append(FallacyMatch(
                    name: pattern.name,
                    pattern: matchedKeyword,
                    confidence: adjustedConfidence,
                    explanation: pattern.explanation
                ))
            }
        }

        // Sort by confidence descending
        results.sort { $0.confidence > $1.confidence }
        return results
    }

    /// Total number of registered fallacy patterns
    var patternCount: Int { patterns.count }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 6: MODAL LOGIC (Kripke Semantics)
// ═══════════════════════════════════════════════════════════════════

final class ModalLogicLayer {
    static let shared = ModalLogicLayer()
    private init() {}

    /// Create a default Kripke frame with specified worlds
    func createFrame(worlds: [String], accessibility: [String: [String]], valuations: [String: [String: Bool]]) -> KripkeFrame {
        var accessSet: [String: Set<String>] = [:]
        for (k, v) in accessibility {
            accessSet[k] = Set(v)
        }
        return KripkeFrame(worlds: Set(worlds), accessibility: accessSet, valuations: valuations)
    }

    /// Evaluate if a proposition is true in a specific world
    func evaluateInWorld(_ prop: String, world: String, frame: KripkeFrame) -> Bool {
        return frame.valuations[world]?[prop] ?? false
    }

    /// Evaluate necessity (box): true in all accessible worlds
    func evaluateNecessity(_ prop: String, world: String, frame: KripkeFrame) -> Bool {
        guard let accessible = frame.accessibility[world] else { return true }
        return accessible.allSatisfy { w in
            evaluateInWorld(prop, world: w, frame: frame)
        }
    }

    /// Evaluate possibility (diamond): true in at least one accessible world
    func evaluatePossibility(_ prop: String, world: String, frame: KripkeFrame) -> Bool {
        guard let accessible = frame.accessibility[world] else { return false }
        return accessible.contains { w in
            evaluateInWorld(prop, world: w, frame: frame)
        }
    }

    /// Global necessity: true in ALL worlds
    func evaluateGlobalNecessity(_ prop: String, frame: KripkeFrame) -> Bool {
        return frame.worlds.allSatisfy { w in
            evaluateInWorld(prop, world: w, frame: frame)
        }
    }

    /// Global possibility: true in at least one world
    func evaluateGlobalPossibility(_ prop: String, frame: KripkeFrame) -> Bool {
        return frame.worlds.contains { w in
            evaluateInWorld(prop, world: w, frame: frame)
        }
    }

    /// Classify a proposition's modal status in a frame
    func classifyModal(_ prop: String, frame: KripkeFrame) -> ModalOperator {
        let necessarilyTrue = evaluateGlobalNecessity(prop, frame: frame)
        let possiblyTrue = evaluateGlobalPossibility(prop, frame: frame)

        if necessarilyTrue { return .necessary }
        if !possiblyTrue { return .impossible }
        return .contingent
    }

    /// Check if frame is reflexive (S5-like)
    func isReflexive(_ frame: KripkeFrame) -> Bool {
        return frame.worlds.allSatisfy { w in
            frame.accessibility[w]?.contains(w) ?? false
        }
    }

    /// Check if frame is transitive
    func isTransitive(_ frame: KripkeFrame) -> Bool {
        for w in frame.worlds {
            guard let accessible = frame.accessibility[w] else { continue }
            for v in accessible {
                guard let vAccessible = frame.accessibility[v] else { continue }
                for u in vAccessible {
                    if !(frame.accessibility[w]?.contains(u) ?? false) {
                        return false
                    }
                }
            }
        }
        return true
    }

    /// Check if frame is symmetric
    func isSymmetric(_ frame: KripkeFrame) -> Bool {
        for w in frame.worlds {
            guard let accessible = frame.accessibility[w] else { continue }
            for v in accessible {
                if !(frame.accessibility[v]?.contains(w) ?? false) {
                    return false
                }
            }
        }
        return true
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 7: NL-TO-LOGIC TRANSLATOR
// ═══════════════════════════════════════════════════════════════════

final class NLToLogicTranslator {
    static let shared = NLToLogicTranslator()
    private init() {}

    private struct TranslationPattern {
        let regex: String
        let builder: ([String]) -> PropFormula?
    }

    /// Translate a natural language sentence to a propositional formula
    func translate(_ sentence: String) -> PropFormula? {
        let s = sentence.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        // "if P then Q" -> P IMPLIES Q
        if let match = matchPattern(s, pattern: "^if\\s+(.+?)\\s+then\\s+(.+)$") {
            let pAtom = atomize(match[0])
            let qAtom = atomize(match[1])
            return .compound(.implies, pAtom, qAtom)
        }

        // "P if and only if Q" -> P IFF Q
        if let match = matchPattern(s, pattern: "^(.+?)\\s+if and only if\\s+(.+)$") {
            let pAtom = atomize(match[0])
            let qAtom = atomize(match[1])
            return .compound(.iff, pAtom, qAtom)
        }

        // "P only if Q" -> P IMPLIES Q
        if let match = matchPattern(s, pattern: "^(.+?)\\s+only if\\s+(.+)$") {
            let pAtom = atomize(match[0])
            let qAtom = atomize(match[1])
            return .compound(.implies, pAtom, qAtom)
        }

        // "either P or Q" -> P XOR Q
        if let match = matchPattern(s, pattern: "^either\\s+(.+?)\\s+or\\s+(.+)$") {
            let pAtom = atomize(match[0])
            let qAtom = atomize(match[1])
            return .compound(.xor, pAtom, qAtom)
        }

        // "neither P nor Q" -> NOT P AND NOT Q
        if let match = matchPattern(s, pattern: "^neither\\s+(.+?)\\s+nor\\s+(.+)$") {
            let pAtom = atomize(match[0])
            let qAtom = atomize(match[1])
            let notP = PropFormula.compound(.not, pAtom, nil)
            let notQ = PropFormula.compound(.not, qAtom, nil)
            return .compound(.and, notP, notQ)
        }

        // "P unless Q" -> NOT Q IMPLIES P (= Q OR P)
        if let match = matchPattern(s, pattern: "^(.+?)\\s+unless\\s+(.+)$") {
            let pAtom = atomize(match[0])
            let qAtom = atomize(match[1])
            return .compound(.or, qAtom, pAtom)
        }

        // "P and Q" -> P AND Q
        if let match = matchPattern(s, pattern: "^(.+?)\\s+and\\s+(.+)$") {
            let pAtom = atomize(match[0])
            let qAtom = atomize(match[1])
            return .compound(.and, pAtom, qAtom)
        }

        // "P or Q" -> P OR Q
        if let match = matchPattern(s, pattern: "^(.+?)\\s+or\\s+(.+)$") {
            let pAtom = atomize(match[0])
            let qAtom = atomize(match[1])
            return .compound(.or, pAtom, qAtom)
        }

        // "not P" -> NOT P
        if let match = matchPattern(s, pattern: "^not\\s+(.+)$") {
            let pAtom = atomize(match[0])
            return .compound(.not, pAtom, nil)
        }

        // "it is not the case that P" -> NOT P
        if let match = matchPattern(s, pattern: "^it is not the case that\\s+(.+)$") {
            let pAtom = atomize(match[0])
            return .compound(.not, pAtom, nil)
        }

        // "P implies Q" -> P IMPLIES Q
        if let match = matchPattern(s, pattern: "^(.+?)\\s+implies\\s+(.+)$") {
            let pAtom = atomize(match[0])
            let qAtom = atomize(match[1])
            return .compound(.implies, pAtom, qAtom)
        }

        // "P therefore Q" -> P IMPLIES Q
        if let match = matchPattern(s, pattern: "^(.+?)\\s+therefore\\s+(.+)$") {
            let pAtom = atomize(match[0])
            let qAtom = atomize(match[1])
            return .compound(.implies, pAtom, qAtom)
        }

        // Fallback: treat the entire sentence as a single atom
        return atomize(s)
    }

    private func matchPattern(_ text: String, pattern: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        guard let match = regex.firstMatch(in: text, range: range) else { return nil }
        var groups: [String] = []
        for i in 1..<match.numberOfRanges {
            if let r = Range(match.range(at: i), in: text) {
                groups.append(String(text[r]))
            }
        }
        return groups.isEmpty ? nil : groups
    }

    /// Convert a text phrase to an atom name (capitalized, spaces removed)
    private func atomize(_ text: String) -> PropFormula {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: " ", with: "_")
            .uppercased()
        return .atom(cleaned)
    }

    /// Batch translate multiple sentences
    func translateAll(_ sentences: [String]) -> [PropFormula?] {
        return sentences.map { translate($0) }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 8: ARGUMENT ANALYZER
// ═══════════════════════════════════════════════════════════════════

final class ArgumentAnalyzer {
    static let shared = ArgumentAnalyzer()
    private init() {}

    /// Analyze an argument from premises and conclusion
    func analyzeArgument(premises: [String], conclusion: String) -> ArgumentAnalysis {
        let translator = NLToLogicTranslator.shared
        let propLogic = PropositionalLogic.shared
        let fallacyDetector = FallacyDetector.shared

        // Translate premises and conclusion to formulas
        let premiseFormulas = premises.compactMap { translator.translate($0) }
        let conclusionFormula = translator.translate(conclusion)

        // Build formal structure string
        var structureParts: [String] = []
        for (i, pf) in premiseFormulas.enumerated() {
            structureParts.append("P\(i+1): \(pf)")
        }
        if let cf = conclusionFormula {
            structureParts.append("C: \(cf)")
        }
        let formalStructure = structureParts.joined(separator: "; ")

        // Check validity: conjunction of premises implies conclusion should be tautology
        var isValid = false
        var strength = 0.0

        if !premiseFormulas.isEmpty, let concl = conclusionFormula {
            // Build (P1 AND P2 AND ... AND Pn) IMPLIES C
            var conjunction = premiseFormulas[0]
            for i in 1..<premiseFormulas.count {
                conjunction = .compound(.and, conjunction, premiseFormulas[i])
            }
            let implication = PropFormula.compound(.implies, conjunction, concl)
            isValid = propLogic.isTautology(implication)

            // Compute strength: proportion of truth table rows where implication holds
            let table = propLogic.truthTable(implication)
            let trueCount = table.filter { $0["_RESULT"] == true }.count
            strength = table.isEmpty ? 0.0 : Double(trueCount) / Double(table.count)
        }

        // Detect fallacies in all text
        let allText = (premises + [conclusion]).joined(separator: " ")
        let fallacies = fallacyDetector.detectFallacies(allText)

        // Reduce strength if fallacies are detected
        if !fallacies.isEmpty {
            let maxFallacyConfidence = fallacies.map(\.confidence).max() ?? 0.0
            strength *= (1.0 - maxFallacyConfidence * 0.5)
        }

        // Apply PHI-scaled scoring
        strength = min(1.0, strength * PHI / PHI)  // Identity but acknowledges the constant

        return ArgumentAnalysis(
            premises: premises,
            conclusion: conclusion,
            isValid: isValid,
            strength: strength,
            fallacies: fallacies,
            formalStructure: formalStructure
        )
    }

    /// Quick validity check (just tests tautology of implication)
    func isValidArgument(premises: [String], conclusion: String) -> Bool {
        let analysis = analyzeArgument(premises: premises, conclusion: conclusion)
        return analysis.isValid
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 9: RESOLUTION PROVER
// ═══════════════════════════════════════════════════════════════════

final class ResolutionProver {
    static let shared = ResolutionProver()
    private init() {}

    /// Negate a literal: "P" -> "~P", "~P" -> "P"
    private func negateLiteral(_ literal: String) -> String {
        if literal.hasPrefix("~") {
            return String(literal.dropFirst())
        }
        return "~\(literal)"
    }

    /// Check if a clause set is trivially true (contains P and ~P)
    private func isTrivial(_ clause: Set<String>) -> Bool {
        for literal in clause {
            if clause.contains(negateLiteral(literal)) {
                return true
            }
        }
        return false
    }

    /// Resolve two clauses on a complementary pair, returning resolvent or nil
    private func resolve(_ clause1: Set<String>, _ clause2: Set<String>) -> Set<String>? {
        for literal in clause1 {
            let complement = negateLiteral(literal)
            if clause2.contains(complement) {
                var resolvent = clause1.union(clause2)
                resolvent.remove(literal)
                resolvent.remove(complement)
                if isTrivial(resolvent) { return nil }
                return resolvent
            }
        }
        return nil
    }

    /// Prove by resolution refutation:
    /// Add negation of conclusion to premises, derive empty clause
    func prove(premises: [PropFormula], conclusion: PropFormula) -> [ProofStep]? {
        let propLogic = PropositionalLogic.shared

        // Collect all premise clauses
        var allClauses: [Set<String>] = []
        var proofSteps: [ProofStep] = []
        var stepNum = 0

        for premise in premises {
            let clauses = propLogic.extractClauses(premise)
            for clause in clauses {
                stepNum += 1
                allClauses.append(clause)
                proofSteps.append(ProofStep(
                    stepNumber: stepNum,
                    formula: clauseToString(clause),
                    rule: "Premise",
                    fromSteps: []
                ))
            }
        }

        // Negate the conclusion and add its CNF clauses
        let negatedConclusion = PropFormula.compound(.not, conclusion, nil)
        let negClauses = propLogic.extractClauses(negatedConclusion)
        for clause in negClauses {
            stepNum += 1
            allClauses.append(clause)
            proofSteps.append(ProofStep(
                stepNumber: stepNum,
                formula: clauseToString(clause),
                rule: "Negated Conclusion",
                fromSteps: []
            ))
        }

        // Apply resolution rule iteratively
        var clauseSet = allClauses
        var iterations = 0
        let maxIterations = RESOLUTION_DEPTH_LIMIT * 10  // Safety limit

        while iterations < maxIterations {
            iterations += 1
            var newClauses: [Set<String>] = []

            for i in 0..<clauseSet.count {
                for j in (i + 1)..<clauseSet.count {
                    if let resolvent = resolve(clauseSet[i], clauseSet[j]) {
                        // Empty clause = contradiction = proof complete
                        if resolvent.isEmpty {
                            stepNum += 1
                            proofSteps.append(ProofStep(
                                stepNumber: stepNum,
                                formula: "[] (empty clause)",
                                rule: "Resolution",
                                fromSteps: [i + 1, j + 1]
                            ))
                            return proofSteps
                        }

                        // Check if resolvent is genuinely new
                        if !clauseSet.contains(resolvent) && !newClauses.contains(resolvent) {
                            newClauses.append(resolvent)
                            stepNum += 1
                            proofSteps.append(ProofStep(
                                stepNumber: stepNum,
                                formula: clauseToString(resolvent),
                                rule: "Resolution",
                                fromSteps: [i + 1, j + 1]
                            ))
                        }
                    }
                }
            }

            // If no new clauses generated, the conclusion cannot be proven
            if newClauses.isEmpty {
                return nil
            }

            clauseSet.append(contentsOf: newClauses)

            // Safety: limit total clauses
            if clauseSet.count > 500 {
                return nil
            }
        }

        return nil
    }

    /// Convert a clause (set of literals) to a readable string
    private func clauseToString(_ clause: Set<String>) -> String {
        if clause.isEmpty { return "[]" }
        return "{\(clause.sorted().joined(separator: ", "))}"
    }

    /// Convenience: prove from string premises and conclusion
    func proveFromStrings(premises: [String], conclusion: String) -> [ProofStep]? {
        let propLogic = PropositionalLogic.shared
        let premiseFormulas = premises.compactMap { propLogic.parseProp($0) }
        guard let conclusionFormula = propLogic.parseProp(conclusion) else { return nil }
        return prove(premises: premiseFormulas, conclusion: conclusionFormula)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - LAYER 10: NATURAL DEDUCTION (Fitch-style)
// ═══════════════════════════════════════════════════════════════════

final class NaturalDeductionEngine {
    static let shared = NaturalDeductionEngine()
    private init() {}

    /// Inference rules for natural deduction
    enum InferenceRule: String {
        case modusPonens = "Modus Ponens"
        case modusTollens = "Modus Tollens"
        case disjunctiveSyllogism = "Disjunctive Syllogism"
        case hypotheticalSyllogism = "Hypothetical Syllogism"
        case constructiveDilemma = "Constructive Dilemma"
        case simplification = "Simplification"
        case conjunction = "Conjunction"
        case addition = "Addition"
        case assumption = "Assumption"
    }

    /// Apply natural deduction rules to premises and derive conclusions
    func naturalDeduction(premises: [String]) -> InferenceResult {
        let propLogic = PropositionalLogic.shared
        var parsedPremises: [(formula: PropFormula, text: String)] = []
        var proofSteps: [ProofStep] = []
        var conclusions: [String] = []
        var stepNum = 0

        // Parse all premises
        for p in premises {
            if let f = propLogic.parseProp(p) {
                parsedPremises.append((f, p))
                stepNum += 1
                proofSteps.append(ProofStep(
                    stepNumber: stepNum,
                    formula: p,
                    rule: InferenceRule.assumption.rawValue,
                    fromSteps: []
                ))
            }
        }

        // Apply rules exhaustively (bounded)
        var derived: [(formula: PropFormula, text: String)] = parsedPremises
        var appliedRules = 0
        let maxRules = 50

        while appliedRules < maxRules {
            var newDerived = false

            // Modus Ponens: from (P IMPLIES Q) and P, derive Q
            for i in 0..<derived.count {
                if case .compound(.implies, let antecedent, let consequent) = derived[i].formula,
                   let q = consequent {
                    for j in 0..<derived.count where j != i {
                        if formulasEquivalent(derived[j].formula, antecedent) {
                            let qText = "\(q)"
                            if !derived.contains(where: { formulasEquivalent($0.formula, q) }) {
                                derived.append((q, qText))
                                stepNum += 1
                                proofSteps.append(ProofStep(
                                    stepNumber: stepNum,
                                    formula: qText,
                                    rule: InferenceRule.modusPonens.rawValue,
                                    fromSteps: [i + 1, j + 1]
                                ))
                                conclusions.append(qText)
                                newDerived = true
                            }
                        }
                    }
                }
            }

            // Modus Tollens: from (P IMPLIES Q) and NOT Q, derive NOT P
            for i in 0..<derived.count {
                if case .compound(.implies, let p, let q) = derived[i].formula,
                   let qVal = q {
                    for j in 0..<derived.count where j != i {
                        if case .compound(.not, let inner, _) = derived[j].formula,
                           formulasEquivalent(inner, qVal) {
                            let notP = PropFormula.compound(.not, p, nil)
                            let notPText = "\(notP)"
                            if !derived.contains(where: { formulasEquivalent($0.formula, notP) }) {
                                derived.append((notP, notPText))
                                stepNum += 1
                                proofSteps.append(ProofStep(
                                    stepNumber: stepNum,
                                    formula: notPText,
                                    rule: InferenceRule.modusTollens.rawValue,
                                    fromSteps: [i + 1, j + 1]
                                ))
                                conclusions.append(notPText)
                                newDerived = true
                            }
                        }
                    }
                }
            }

            // Disjunctive Syllogism: from (P OR Q) and NOT P, derive Q
            for i in 0..<derived.count {
                if case .compound(.or, let p, let q) = derived[i].formula,
                   let qVal = q {
                    for j in 0..<derived.count where j != i {
                        if case .compound(.not, let inner, _) = derived[j].formula {
                            if formulasEquivalent(inner, p) {
                                let qText = "\(qVal)"
                                if !derived.contains(where: { formulasEquivalent($0.formula, qVal) }) {
                                    derived.append((qVal, qText))
                                    stepNum += 1
                                    proofSteps.append(ProofStep(
                                        stepNumber: stepNum,
                                        formula: qText,
                                        rule: InferenceRule.disjunctiveSyllogism.rawValue,
                                        fromSteps: [i + 1, j + 1]
                                    ))
                                    conclusions.append(qText)
                                    newDerived = true
                                }
                            }
                            if formulasEquivalent(inner, qVal) {
                                let pText = "\(p)"
                                if !derived.contains(where: { formulasEquivalent($0.formula, p) }) {
                                    derived.append((p, pText))
                                    stepNum += 1
                                    proofSteps.append(ProofStep(
                                        stepNumber: stepNum,
                                        formula: pText,
                                        rule: InferenceRule.disjunctiveSyllogism.rawValue,
                                        fromSteps: [i + 1, j + 1]
                                    ))
                                    conclusions.append(pText)
                                    newDerived = true
                                }
                            }
                        }
                    }
                }
            }

            // Hypothetical Syllogism: from (P IMPLIES Q) and (Q IMPLIES R), derive (P IMPLIES R)
            for i in 0..<derived.count {
                if case .compound(.implies, let p, let q) = derived[i].formula,
                   let qVal = q {
                    for j in 0..<derived.count where j != i {
                        if case .compound(.implies, let q2, let r) = derived[j].formula,
                           let rVal = r,
                           formulasEquivalent(qVal, q2) {
                            let chain = PropFormula.compound(.implies, p, rVal)
                            let chainText = "\(chain)"
                            if !derived.contains(where: { formulasEquivalent($0.formula, chain) }) {
                                derived.append((chain, chainText))
                                stepNum += 1
                                proofSteps.append(ProofStep(
                                    stepNumber: stepNum,
                                    formula: chainText,
                                    rule: InferenceRule.hypotheticalSyllogism.rawValue,
                                    fromSteps: [i + 1, j + 1]
                                ))
                                conclusions.append(chainText)
                                newDerived = true
                            }
                        }
                    }
                }
            }

            // Simplification: from (P AND Q), derive P and Q
            for i in 0..<derived.count {
                if case .compound(.and, let p, let q) = derived[i].formula {
                    let pText = "\(p)"
                    if !derived.contains(where: { formulasEquivalent($0.formula, p) }) {
                        derived.append((p, pText))
                        stepNum += 1
                        proofSteps.append(ProofStep(
                            stepNumber: stepNum,
                            formula: pText,
                            rule: InferenceRule.simplification.rawValue,
                            fromSteps: [i + 1]
                        ))
                        conclusions.append(pText)
                        newDerived = true
                    }
                    if let qVal = q {
                        let qText = "\(qVal)"
                        if !derived.contains(where: { formulasEquivalent($0.formula, qVal) }) {
                            derived.append((qVal, qText))
                            stepNum += 1
                            proofSteps.append(ProofStep(
                                stepNumber: stepNum,
                                formula: qText,
                                rule: InferenceRule.simplification.rawValue,
                                fromSteps: [i + 1]
                            ))
                            conclusions.append(qText)
                            newDerived = true
                        }
                    }
                }
            }

            appliedRules += 1
            if !newDerived { break }
        }

        // Confidence based on how many rules were successfully applied
        let confidence = conclusions.isEmpty ? 0.3 : min(1.0, 0.5 + Double(conclusions.count) * 0.1 * TAU)

        return InferenceResult(
            conclusions: conclusions,
            proofSteps: proofSteps,
            confidence: confidence
        )
    }

    /// Structural equality check for formulas
    private func formulasEquivalent(_ a: PropFormula, _ b: PropFormula) -> Bool {
        switch (a, b) {
        case (.atom(let n1), .atom(let n2)):
            return n1 == n2
        case (.compound(let op1, let l1, let r1), .compound(let op2, let l2, let r2)):
            if op1 != op2 { return false }
            if !formulasEquivalent(l1, l2) { return false }
            switch (r1, r2) {
            case (nil, nil): return true
            case (let a?, let b?): return formulasEquivalent(a, b)
            default: return false
            }
        default:
            return false
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - FORMAL LOGIC ENGINE (Main Class — Singleton)
// ═══════════════════════════════════════════════════════════════════

final class FormalLogicEngine {
    static let shared = FormalLogicEngine()
    private let lock = NSLock()

    // ─── Statistics ───
    private var analysisCount: Int = 0
    private var fallaciesDetected: Int = 0
    private var proofsCompleted: Int = 0
    private var syllogismsChecked: Int = 0
    private var truthTablesGenerated: Int = 0
    private var translationsPerformed: Int = 0
    private var equivalencesChecked: Int = 0
    private var modalEvaluations: Int = 0
    private var deductionRuns: Int = 0
    private var resolutionRuns: Int = 0
    private var startTime: Date = Date()

    // ─── Layer References ───
    private let propLogic = PropositionalLogic.shared
    private let predicateLogic = PredicateLogicLayer.shared
    private let syllogismEngine = SyllogismEngine.shared
    private let equivalenceProver = EquivalenceProver.shared
    private let fallacyDetector = FallacyDetector.shared
    private let modalLogic = ModalLogicLayer.shared
    private let translator = NLToLogicTranslator.shared
    private let argumentAnalyzer = ArgumentAnalyzer.shared
    private let resolutionProver = ResolutionProver.shared
    private let naturalDeduction = NaturalDeductionEngine.shared

    private init() {
        startTime = Date()
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - PUBLIC API
    // ═══════════════════════════════════════════════════════════════

    /// Analyze an argument from premises and conclusion (Layer 8)
    func analyzeArgument(premises: [String], conclusion: String) -> ArgumentAnalysis {
        lock.lock()
        analysisCount += 1
        lock.unlock()

        let analysis = argumentAnalyzer.analyzeArgument(premises: premises, conclusion: conclusion)

        lock.lock()
        fallaciesDetected += analysis.fallacies.count
        lock.unlock()

        return analysis
    }

    /// Detect fallacies in text (Layer 5)
    func detectFallacies(text: String) -> [FallacyMatch] {
        let result = fallacyDetector.detectFallacies(text)
        lock.lock()
        fallaciesDetected += result.count
        lock.unlock()
        return result
    }

    /// Prove a syllogism from three natural language propositions (Layer 3)
    func proveSyllogism(major: String, minor: String, conclusion: String) -> Bool {
        lock.lock()
        syllogismsChecked += 1
        lock.unlock()

        guard let majorProp = syllogismEngine.parseCategorical(major),
              let minorProp = syllogismEngine.parseCategorical(minor),
              let conclusionProp = syllogismEngine.parseCategorical(conclusion) else {
            return false
        }

        let syllogism = syllogismEngine.buildSyllogism(
            major: majorProp,
            minor: minorProp,
            conclusion: conclusionProp
        )
        return syllogism.isValid
    }

    /// Generate and evaluate a truth table for a formula string (Layer 1)
    func evaluateTruthTable(formula: String) -> [[String: Bool]] {
        lock.lock()
        truthTablesGenerated += 1
        lock.unlock()

        guard let parsed = propLogic.parseProp(formula) else { return [] }
        return propLogic.truthTable(parsed)
    }

    /// Parse and check if a formula is a tautology (Layer 1)
    func isTautology(formula: String) -> Bool {
        guard let parsed = propLogic.parseProp(formula) else { return false }
        return propLogic.isTautology(parsed)
    }

    /// Parse and check if a formula is a contradiction (Layer 1)
    func isContradiction(formula: String) -> Bool {
        guard let parsed = propLogic.parseProp(formula) else { return false }
        return propLogic.isContradiction(parsed)
    }

    /// Parse and check if a formula is satisfiable (Layer 1)
    func isSatisfiable(formula: String) -> Bool {
        guard let parsed = propLogic.parseProp(formula) else { return false }
        return propLogic.isSatisfiable(parsed)
    }

    /// Check logical equivalence of two formula strings (Layer 4)
    func checkEquivalence(formula1: String, formula2: String) -> Bool {
        lock.lock()
        equivalencesChecked += 1
        lock.unlock()

        guard let f1 = propLogic.parseProp(formula1),
              let f2 = propLogic.parseProp(formula2) else { return false }
        return equivalenceProver.checkEquivalence(f1, f2)
    }

    /// Translate natural language to propositional formula (Layer 7)
    func translateToLogic(sentence: String) -> String? {
        lock.lock()
        translationsPerformed += 1
        lock.unlock()

        guard let formula = translator.translate(sentence) else { return nil }
        return "\(formula)"
    }

    /// Evaluate modal necessity in a Kripke frame (Layer 6)
    func evaluateNecessity(proposition: String, world: String, frame: KripkeFrame) -> Bool {
        lock.lock()
        modalEvaluations += 1
        lock.unlock()

        return modalLogic.evaluateNecessity(proposition, world: world, frame: frame)
    }

    /// Evaluate modal possibility in a Kripke frame (Layer 6)
    func evaluatePossibility(proposition: String, world: String, frame: KripkeFrame) -> Bool {
        lock.lock()
        modalEvaluations += 1
        lock.unlock()

        return modalLogic.evaluatePossibility(proposition, world: world, frame: frame)
    }

    /// Prove by resolution refutation (Layer 9)
    func proveByResolution(premises: [String], conclusion: String) -> [ProofStep]? {
        lock.lock()
        resolutionRuns += 1
        lock.unlock()

        let result = resolutionProver.proveFromStrings(premises: premises, conclusion: conclusion)

        if result != nil {
            lock.lock()
            proofsCompleted += 1
            lock.unlock()
        }

        return result
    }

    /// Perform natural deduction on premises (Layer 10)
    func performNaturalDeduction(premises: [String]) -> InferenceResult {
        lock.lock()
        deductionRuns += 1
        lock.unlock()

        let result = naturalDeduction.naturalDeduction(premises: premises)

        if !result.conclusions.isEmpty {
            lock.lock()
            proofsCompleted += result.conclusions.count
            lock.unlock()
        }

        return result
    }

    /// Parse a predicate logic formula (Layer 2)
    func parsePredicate(text: String) -> PredicateFormula? {
        return predicateLogic.parsePredicate(text)
    }

    /// Parse a categorical proposition (Layer 3)
    func parseCategorical(text: String) -> CategoricalProposition? {
        return syllogismEngine.parseCategorical(text)
    }

    /// Convert a formula string to CNF (Layer 1)
    func toCNF(formula: String) -> String? {
        guard let parsed = propLogic.parseProp(formula) else { return nil }
        return "\(propLogic.toCNF(parsed))"
    }

    /// Convert a formula string to DNF (Layer 1)
    func toDNF(formula: String) -> String? {
        guard let parsed = propLogic.parseProp(formula) else { return nil }
        return "\(propLogic.toDNF(parsed))"
    }

    /// Identify which equivalence rules apply between two formulas (Layer 4)
    func identifyEquivalences(formula1: String, formula2: String) -> [String] {
        guard let f1 = propLogic.parseProp(formula1),
              let f2 = propLogic.parseProp(formula2) else { return [] }
        return equivalenceProver.identifyEquivalences(f1, f2)
    }

    /// Classify modal status of a proposition across a Kripke frame (Layer 6)
    func classifyModal(proposition: String, frame: KripkeFrame) -> String {
        lock.lock()
        modalEvaluations += 1
        lock.unlock()

        return modalLogic.classifyModal(proposition, frame: frame).rawValue
    }

    /// Validate Kripke frame properties (Layer 6)
    func validateFrameProperties(frame: KripkeFrame) -> [String: Bool] {
        return [
            "reflexive": modalLogic.isReflexive(frame),
            "transitive": modalLogic.isTransitive(frame),
            "symmetric": modalLogic.isSymmetric(frame)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - ASI SCORING
    // ═══════════════════════════════════════════════════════════════

    /// Overall logic engine score for ASI scoring dimension (0.0 - 1.0)
    func overallScore() -> Double {
        lock.lock()
        let total = analysisCount + fallaciesDetected + proofsCompleted +
                    syllogismsChecked + truthTablesGenerated + translationsPerformed +
                    equivalencesChecked + modalEvaluations + deductionRuns + resolutionRuns
        lock.unlock()

        guard total > 0 else { return 0.0 }

        // PHI-weighted score: more operations = higher score, saturating at 1.0
        let raw = Double(total) / (Double(total) + GOD_CODE * TAU)
        let scaled = raw * PHI
        return min(1.0, scaled)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - STATUS
    // ═══════════════════════════════════════════════════════════════

    func getStatus() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        let uptime = Date().timeIntervalSince(startTime)

        return [
            "engine": "FormalLogicEngine",
            "version": FORMAL_LOGIC_VERSION,
            "layers": 10,
            "layer_names": [
                "L1_PropositionalLogic",
                "L2_PredicateLogic",
                "L3_SyllogismEngine",
                "L4_EquivalenceProver",
                "L5_FallacyDetector",
                "L6_ModalLogic",
                "L7_NLToLogicTranslator",
                "L8_ArgumentAnalyzer",
                "L9_ResolutionProver",
                "L10_NaturalDeduction"
            ],
            "statistics": [
                "analyses": analysisCount,
                "fallacies_detected": fallaciesDetected,
                "proofs_completed": proofsCompleted,
                "syllogisms_checked": syllogismsChecked,
                "truth_tables_generated": truthTablesGenerated,
                "translations_performed": translationsPerformed,
                "equivalences_checked": equivalencesChecked,
                "modal_evaluations": modalEvaluations,
                "deduction_runs": deductionRuns,
                "resolution_runs": resolutionRuns
            ],
            "capabilities": [
                "fallacy_patterns": fallacyDetector.patternCount,
                "valid_syllogism_forms": 24,
                "equivalence_rules": equivalenceProver.rules.count,
                "resolution_depth_limit": RESOLUTION_DEPTH_LIMIT,
                "inference_rules": 9
            ],
            "sacred_constants": [
                "PHI": PHI,
                "GOD_CODE": GOD_CODE,
                "TAU": TAU
            ],
            "overall_score": overallScore(),
            "uptime_seconds": uptime
        ]
    }
}
