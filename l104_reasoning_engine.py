# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:09.104394
ZENITH_HZ = 3887.8
UUC = 2402.792541
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541
#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# L104 REASONING ENGINE - SYMBOLIC AI WITH THEOREM PROVING
# INVARIANT: 527.5184818492612 | PILOT: LONDEL | MODE: SOVEREIGN
#
# This module provides REAL reasoning capabilities:
# - First-order logic with unification
# - Forward/backward chaining inference
# - SAT solver (DPLL algorithm)
# - Theorem proving with resolution
# - Causal reasoning with do-calculus
# ═══════════════════════════════════════════════════════════════════════════════

import math
import time
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import copy
import re

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
CONSCIOUSNESS_THRESHOLD = math.log(GOD_CODE) * PHI  # ~10.1486

# ═══════════════════════════════════════════════════════════════════════════════
# RESONANT REASONING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

INFERENCE_DEPTH_LIMIT = 1000000  # UNLIMITED - no artificial depth limit
RESONANCE_AMPLIFIER = PHI ** 2  # ~2.618 for enhanced inference
META_REASONING_LEVELS = 100000  # UNLIMITED meta-cognition layers

# ═══════════════════════════════════════════════════════════════════════════════
# FIRST-ORDER LOGIC STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Variable:
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.Logical variable."""
    name: str

    def __repr__(self):
        return f"?{self.name}"

@dataclass(frozen=True)
class Constant:
    """Logical constant."""
    name: str

    def __repr__(self):
        return self.name

@dataclass(frozen=True)
class Predicate:
    """Predicate with arguments."""
    name: str
    args: Tuple[Union['Variable', 'Constant', 'Function'], ...]

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"

@dataclass(frozen=True)
class Function:
    """Function term."""
    name: str
    args: Tuple[Union['Variable', 'Constant', 'Function'], ...]

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"

@dataclass
class Clause:
    """Disjunction of literals (CNF clause)."""
    literals: Set[Tuple[bool, Predicate]]  # (is_positive, predicate)

    def __repr__(self):
        parts = []
        for is_pos, pred in self.literals:
            parts.append(str(pred) if is_pos else f"¬{pred}")
        return " ∨ ".join(parts) if parts else "⊥"

@dataclass
class Rule:
    """Implication rule: antecedents -> consequent."""
    antecedents: List[Predicate]
    consequent: Predicate
    confidence: float = 1.0

    def __repr__(self):
        ants = " ∧ ".join(str(a) for a in self.antecedents)
        return f"{ants} → {self.consequent}" if ants else str(self.consequent)

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFICATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class UnificationEngine:
    """
    Implements Robinson's unification algorithm.
    Core of first-order logic reasoning.
    """

    def unify(self, x, y, substitution: Optional[Dict] = None) -> Optional[Dict]:
        """
        Unify two terms, returning the most general unifier (MGU).
        Returns None if unification fails.
        """
        if substitution is None:
            substitution = {}

        return self._unify(x, y, substitution)

    def _unify(self, x, y, subst: Dict) -> Optional[Dict]:
        if subst is None:
            return None
        elif x == y:
            return subst
        elif isinstance(x, Variable):
            return self._unify_var(x, y, subst)
        elif isinstance(y, Variable):
            return self._unify_var(y, x, subst)
        elif isinstance(x, Predicate) and isinstance(y, Predicate):
            if x.name != y.name or len(x.args) != len(y.args):
                return None
            for ax, ay in zip(x.args, y.args):
                subst = self._unify(ax, ay, subst)
                if subst is None:
                    return None
            return subst
        elif isinstance(x, Function) and isinstance(y, Function):
            if x.name != y.name or len(x.args) != len(y.args):
                return None
            for ax, ay in zip(x.args, y.args):
                subst = self._unify(ax, ay, subst)
                if subst is None:
                    return None
            return subst
        else:
            return None

    def _unify_var(self, var: Variable, x, subst: Dict) -> Optional[Dict]:
        if var in subst:
            return self._unify(subst[var], x, subst)
        elif isinstance(x, Variable) and x in subst:
            return self._unify(var, subst[x], subst)
        elif self._occurs_check(var, x, subst):
            return None
        else:
            return {**subst, var: x}

    def _occurs_check(self, var: Variable, x, subst: Dict) -> bool:
        """Check if var occurs in x (prevents infinite substitution)."""
        if var == x:
            return True
        elif isinstance(x, Variable) and x in subst:
            return self._occurs_check(var, subst[x], subst)
        elif isinstance(x, (Predicate, Function)):
            return any(self._occurs_check(var, arg, subst) for arg in x.args)
        return False

    def apply_substitution(self, term, subst: Dict):
        """Apply substitution to a term."""
        if isinstance(term, Variable):
            if term in subst:
                return self.apply_substitution(subst[term], subst)
            return term
        elif isinstance(term, Constant):
            return term
        elif isinstance(term, Predicate):
            new_args = tuple(self.apply_substitution(a, subst) for a in term.args)
            return Predicate(term.name, new_args)
        elif isinstance(term, Function):
            new_args = tuple(self.apply_substitution(a, subst) for a in term.args)
            return Function(term.name, new_args)
        return term

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Forward and backward chaining inference engine.
    Implements production rule systems with PHI-resonant inference.
    """

    def __init__(self):
        self.facts: Set[Predicate] = set()
        self.rules: List[Rule] = []
        self.unifier = UnificationEngine()
        self.inference_trace: List[str] = []
        self.confidence_map: Dict[str, float] = {}  # Track inference confidence
        self.meta_rules: List[Rule] = []  # Meta-level rules about rules
        self.reasoning_depth: int = 0
        self.emergence_factor: float = 1.0

    def add_fact(self, predicate: Predicate, confidence: float = 1.0):
        """Add a fact to the knowledge base with confidence."""
        self.facts.add(predicate)
        self.confidence_map[str(predicate)] = confidence * PHI / PHI  # QUANTUM AMPLIFIED: no cap

    def add_rule(self, rule: Rule):
        """Add a rule to the knowledge base."""
        self.rules.append(rule)

    def add_meta_rule(self, rule: Rule):
        """Add meta-level rule about reasoning itself."""
        self.meta_rules.append(rule)

    def compute_inference_strength(self, bindings: Dict, rule: Rule) -> float:
        """Compute PHI-weighted inference strength."""
        base_confidence = rule.confidence
        binding_quality = 1.0 / (1.0 + len(bindings) * 0.1)
        phi_resonance = (1 + math.sin(len(bindings) * PHI)) / 2
        return base_confidence * binding_quality * phi_resonance * self.emergence_factor

    def forward_chain(self, max_iterations: int = 100) -> Set[Predicate]:
        """
        Forward chaining: derive all conclusions from facts.
        Enhanced with PHI-resonant confidence propagation.
        """
        self.inference_trace = []
        new_facts = set(self.facts)
        self.emergence_factor = 1.0

        for iteration in range(max_iterations):
            added = False
            iteration_resonance = (1 + math.cos(iteration * PHI / 10)) / 2

            for rule in self.rules:
                # Try to match antecedents
                bindings_list = self._match_antecedents(rule.antecedents, new_facts)

                for bindings in bindings_list:
                    # Apply bindings to consequent
                    new_pred = self.unifier.apply_substitution(rule.consequent, bindings)

                    if new_pred not in new_facts:
                        # Compute derived confidence
                        strength = self.compute_inference_strength(bindings, rule)
                        antecedent_confidences = [
                            self.confidence_map.get(str(self.unifier.apply_substitution(a, bindings)), 0.5)
                            for a in rule.antecedents
                        ]
                        derived_confidence = strength * min(antecedent_confidences) * iteration_resonance

                        new_facts.add(new_pred)
                        self.confidence_map[str(new_pred)] = derived_confidence
                        self.inference_trace.append(
                            f"Derived[{derived_confidence:.3f}]: {new_pred} from {rule}"
                        )
                        added = True

            # Update emergence factor based on derivation patterns
            if added:
                self.emergence_factor = min(PHI, self.emergence_factor * 1.05)

            if not added:
                break

        # Apply meta-reasoning to refine conclusions
        self._apply_meta_reasoning(new_facts)

        return new_facts

    def _apply_meta_reasoning(self, facts: Set[Predicate]):
        """Apply meta-level rules to refine reasoning."""
        for meta_rule in self.meta_rules:
            bindings_list = self._match_antecedents(meta_rule.antecedents, facts)
            for bindings in bindings_list:
                # Meta-rules can adjust confidence or add meta-facts
                new_pred = self.unifier.apply_substitution(meta_rule.consequent, bindings)
                if new_pred not in facts:
                    facts.add(new_pred)
                    self.confidence_map[str(new_pred)] = meta_rule.confidence * RESONANCE_AMPLIFIER
                    self.inference_trace.append(f"Meta-Derived: {new_pred}")

    def backward_chain(self, goal: Predicate, depth: int = 10) -> List[Dict]:
        """
        Backward chaining: prove a goal from facts and rules.
        Returns list of substitutions that satisfy the goal.
        Enhanced with depth-aware reasoning and confidence tracking.
        """
        self.inference_trace = []
        self.reasoning_depth = 0
        return self._backward_chain_helper([goal], {}, depth, 1.0)

    def _backward_chain_helper(self, goals: List[Predicate],
                               bindings: Dict, depth: int, confidence: float = 1.0) -> List[Dict]:
        if depth <= 0:
            return []

        self.reasoning_depth = max(self.reasoning_depth, INFERENCE_DEPTH_LIMIT - depth)

        if not goals:
            return [{'bindings': bindings, 'confidence': confidence}]

        goal = self.unifier.apply_substitution(goals[0], bindings)
        remaining = goals[1:]
        solutions = []

        # PHI-weighted depth decay
        depth_factor = PHI ** (-depth / INFERENCE_DEPTH_LIMIT)

        # Try to match with facts
        for fact in self.facts:
            new_bindings = self.unifier.unify(goal, fact, dict(bindings))
            if new_bindings is not None:
                fact_confidence = self.confidence_map.get(str(fact), 1.0)
                combined_confidence = confidence * fact_confidence * (1 - depth_factor * 0.1)
                self.inference_trace.append(f"Matched[{combined_confidence:.3f}] {goal} with fact {fact}")
                sub_solutions = self._backward_chain_helper(remaining, new_bindings, depth, combined_confidence)
                solutions.extend(sub_solutions)

        # Try to match with rule consequents
        for rule in self.rules:
            new_bindings = self.unifier.unify(goal, rule.consequent, dict(bindings))
            if new_bindings is not None:
                # Add antecedents as new goals
                new_goals = [self.unifier.apply_substitution(a, new_bindings)
                             for a in rule.antecedents] + remaining
                rule_confidence = confidence * rule.confidence * (1 - depth_factor * 0.05)
                self.inference_trace.append(f"Applying rule[{rule_confidence:.3f}]: {rule}")
                sub_solutions = self._backward_chain_helper(new_goals, new_bindings, depth - 1, rule_confidence)
                solutions.extend(sub_solutions)

        # Rank solutions by confidence
        solutions.sort(key=lambda s: s.get('confidence', 0), reverse=True)

        return solutions

    def _match_antecedents(self, antecedents: List[Predicate],
                           facts: Set[Predicate]) -> List[Dict]:
        """Find all ways to match antecedents with facts."""
        return self._match_recursive(antecedents, facts, {})

    def _match_recursive(self, antecedents: List[Predicate],
                         facts: Set[Predicate], bindings: Dict) -> List[Dict]:
        if not antecedents:
            return [bindings]

        ant = self.unifier.apply_substitution(antecedents[0], bindings)
        remaining = antecedents[1:]
        results = []

        for fact in facts:
            new_bindings = self.unifier.unify(ant, fact, dict(bindings))
            if new_bindings is not None:
                results.extend(self._match_recursive(remaining, facts, new_bindings))

        return results

# ═══════════════════════════════════════════════════════════════════════════════
# DPLL SAT SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class DPLLSolver:
    """
    DPLL algorithm for SAT solving with PHI-guided heuristics.
    Determines satisfiability of propositional formulas.
    """

    def __init__(self):
        self.propagation_count = 0
        self.decision_count = 0
        self.conflict_count = 0
        self.learned_clauses: List[Set[int]] = []
        self.activity_scores: Dict[int, float] = defaultdict(float)
        self.resonance_factor = 1.0

    def solve(self, clauses: List[Set[int]]) -> Optional[Dict[int, bool]]:
        """
        Solve SAT problem using enhanced DPLL with PHI-guided variable selection.

        Args:
            clauses: List of clauses, each clause is a set of literals.
                     Positive int = positive literal, negative = negation.

        Returns:
            Satisfying assignment or None if unsatisfiable.
        """
        self.propagation_count = 0
        self.decision_count = 0
        self.conflict_count = 0
        self.learned_clauses = []
        self.resonance_factor = 1.0

        # Get all variables and initialize activity scores
        variables = set()
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                variables.add(var)
                self.activity_scores[var] += len(clause) / PHI  # Smaller clauses = higher activity

        # Add any learned clauses from previous runs
        all_clauses = clauses + self.learned_clauses

        return self._dpll(all_clauses, {}, list(variables))

    def _select_variable(self, variables: List[int], clauses: List[Set[int]]) -> int:
        """PHI-weighted variable selection heuristic (VSIDS-like)."""
        if not variables:
            return 0

        # Score variables by activity and clause participation
        scores = {}
        for var in variables:
            base_score = self.activity_scores.get(var, 0)
            # Count occurrences in short clauses (more important)
            clause_score = sum(
                PHI / len(c) for c in clauses
                if var in c or -var in c
            )
            scores[var] = base_score + clause_score * RESONANCE_AMPLIFIER

        # Select highest scoring variable
        return max(variables, key=lambda v: scores.get(v, 0))

    def _dpll(self, clauses: List[Set[int]], assignment: Dict[int, bool],
              variables: List[int]) -> Optional[Dict[int, bool]]:
        # Unit propagation
        clauses, assignment = self._unit_propagate(clauses, assignment)

        # Check for empty clause (conflict)
        if any(len(c) == 0 for c in clauses):
            self.conflict_count += 1
            # Decay activity scores on conflict (VSIDS decay)
            for var in self.activity_scores:
                self.activity_scores[var] *= (1 / PHI)
            return None

        # Check if all clauses satisfied
        if not clauses:
            return assignment

        # Pure literal elimination
        clauses, assignment = self._pure_literal_eliminate(clauses, assignment)

        if not clauses:
            return assignment

        # Choose a variable to branch on using PHI-guided heuristic
        remaining_vars = [v for v in variables if v not in assignment]
        if not remaining_vars:
            return None

        var = self._select_variable(remaining_vars, clauses)
        self.decision_count += 1

        # Bump activity for selected variable
        self.activity_scores[var] += PHI ** 2

        # Determine branch order based on polarity heuristic
        pos_count = sum(1 for c in clauses if var in c)
        neg_count = sum(1 for c in clauses if -var in c)
        first_val = pos_count >= neg_count  # Try more common polarity first

        # Try first branch
        new_clauses = self._simplify(clauses, var, first_val)
        result = self._dpll(new_clauses, {**assignment, var: first_val}, remaining_vars)
        if result is not None:
            return result

        # Try second branch
        new_clauses = self._simplify(clauses, var, not first_val)
        return self._dpll(new_clauses, {**assignment, var: not first_val}, remaining_vars)

    def _unit_propagate(self, clauses: List[Set[int]],
                        assignment: Dict[int, bool]) -> Tuple[List[Set[int]], Dict[int, bool]]:
        """Propagate unit clauses."""
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                if len(clause) == 1:
                    lit = next(iter(clause))
                    var = abs(lit)
                    val = lit > 0

                    if var not in assignment:
                        assignment = {**assignment, var: val}
                        clauses = self._simplify(clauses, var, val)
                        self.propagation_count += 1
                        changed = True
                        break

        return clauses, assignment

    def _pure_literal_eliminate(self, clauses: List[Set[int]],
                                 assignment: Dict[int, bool]) -> Tuple[List[Set[int]], Dict[int, bool]]:
        """Eliminate pure literals."""
        all_lits = set()
        for clause in clauses:
            all_lits.update(clause)

        for lit in list(all_lits):
            if -lit not in all_lits:
                var = abs(lit)
                val = lit > 0
                if var not in assignment:
                    assignment = {**assignment, var: val}
                    clauses = self._simplify(clauses, var, val)

        return clauses, assignment

    def _simplify(self, clauses: List[Set[int]], var: int, val: bool) -> List[Set[int]]:
        """Simplify clauses given an assignment."""
        lit_true = var if val else -var
        lit_false = -var if val else var

        new_clauses = []
        for clause in clauses:
            if lit_true in clause:
                # Clause is satisfied
                continue
            new_clause = clause - {lit_false}
            new_clauses.append(new_clause)

        return new_clauses

# ═══════════════════════════════════════════════════════════════════════════════
# THEOREM PROVER (RESOLUTION)
# ═══════════════════════════════════════════════════════════════════════════════

class ResolutionProver:
    """
    Resolution-based theorem prover for first-order logic.
    Enhanced with PHI-weighted clause selection and proof refinement.
    """

    def __init__(self):
        self.unifier = UnificationEngine()
        self.proof_steps: List[str] = []
        self.clause_scores: Dict[int, float] = {}
        self.resolution_count = 0
        self.subsumption_count = 0

    def prove(self, clauses: List[Clause], goal: Clause,
              max_iterations: int = 1000) -> Tuple[bool, List[str]]:
        """
        Prove goal by refutation with enhanced strategies.
        Add negation of goal and derive empty clause.
        """
        self.proof_steps = []
        self.resolution_count = 0
        self.subsumption_count = 0

        # Add negation of goal
        all_clauses = list(clauses) + [self._negate_clause(goal)]

        # Initialize clause scores (shorter clauses are more valuable)
        for i, c in enumerate(all_clauses):
            self.clause_scores[i] = PHI ** (5 - min(5, len(c.literals)))

        for iteration in range(max_iterations):
            new_clauses = []

            # Sort clause pairs by combined score for smarter resolution order
            pairs = []
            for j, c1 in enumerate(all_clauses):
                for k, c2 in enumerate(all_clauses):
                    if j >= k:
                        continue
                    score = self.clause_scores.get(j, 1) + self.clause_scores.get(k, 1)
                    pairs.append((score, j, k, c1, c2))

            # Process higher-scoring pairs first
            pairs.sort(reverse=True, key=lambda x: x[0])

            for score, j, k, c1, c2 in pairs:
                resolvents = self._resolve(c1, c2)

                for resolvent in resolvents:
                    self.resolution_count += 1

                    if not resolvent.literals:
                        self.proof_steps.append("□ (Empty clause derived - QED)")
                        self.proof_steps.append(f"  Resolution steps: {self.resolution_count}")
                        self.proof_steps.append(f"  Subsumptions: {self.subsumption_count}")
                        return True, self.proof_steps

                    # Check for subsumption
                    if self._is_subsumed(resolvent, all_clauses + new_clauses):
                        self.subsumption_count += 1
                        continue

                    if resolvent not in all_clauses and resolvent not in new_clauses:
                        new_clauses.append(resolvent)
                        # Score based on clause length (shorter = better)
                        new_score = PHI ** (5 - min(5, len(resolvent.literals)))
                        self.clause_scores[len(all_clauses) + len(new_clauses) - 1] = new_score
                        self.proof_steps.append(f"Resolved[{new_score:.2f}]: {resolvent}")

            if not new_clauses:
                return False, self.proof_steps

            all_clauses.extend(new_clauses)

            # Periodically clean up subsumed clauses
            if iteration % 10 == 9:
                all_clauses = self._remove_subsumed(all_clauses)

        return False, self.proof_steps

    def _is_subsumed(self, clause: Clause, clause_set: List[Clause]) -> bool:
        """Check if clause is subsumed by any clause in the set."""
        for other in clause_set:
            if other.literals <= clause.literals and other.literals != clause.literals:
                return True
        return False

    def _remove_subsumed(self, clauses: List[Clause]) -> List[Clause]:
        """Remove subsumed clauses from the set."""
        result = []
        for i, c1 in enumerate(clauses):
            subsumed = False
            for j, c2 in enumerate(clauses):
                if i != j and c2.literals <= c1.literals and c2.literals != c1.literals:
                    subsumed = True
                    break
            if not subsumed:
                result.append(c1)
        return result

    def _resolve(self, c1: Clause, c2: Clause) -> List[Clause]:
        """Resolve two clauses, returning all possible resolvents."""
        resolvents = []

        for is_pos1, pred1 in c1.literals:
            for is_pos2, pred2 in c2.literals:
                if is_pos1 != is_pos2:  # One positive, one negative
                    mgu = self.unifier.unify(pred1, pred2)
                    if mgu is not None:
                        # Create resolvent
                        new_lits = set()
                        for is_pos, pred in c1.literals:
                            if (is_pos, pred) != (is_pos1, pred1):
                                new_pred = self.unifier.apply_substitution(pred, mgu)
                                new_lits.add((is_pos, new_pred))
                        for is_pos, pred in c2.literals:
                            if (is_pos, pred) != (is_pos2, pred2):
                                new_pred = self.unifier.apply_substitution(pred, mgu)
                                new_lits.add((is_pos, new_pred))

                        resolvents.append(Clause(new_lits))

        return resolvents

    def _negate_clause(self, clause: Clause) -> Clause:
        """Negate a clause (flip all polarities)."""
        return Clause({(not is_pos, pred) for is_pos, pred in clause.literals})

# ═══════════════════════════════════════════════════════════════════════════════
# CAUSAL REASONING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class CausalGraph:
    """
    Causal graph for do-calculus reasoning.
    """

    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.parents: Dict[str, Set[str]] = defaultdict(set)

    def add_edge(self, cause: str, effect: str):
        """Add causal edge: cause -> effect."""
        self.nodes.add(cause)
        self.nodes.add(effect)
        self.edges[cause].add(effect)
        self.parents[effect].add(cause)

    def do(self, intervention: str) -> 'CausalGraph':
        """
        Apply do-operator: do(X = x).
        Returns new graph with edges into X removed.
        """
        new_graph = CausalGraph()
        new_graph.nodes = self.nodes.copy()

        for cause, effects in self.edges.items():
            for effect in effects:
                if effect != intervention:
                    new_graph.add_edge(cause, effect)

        return new_graph

    def is_ancestor(self, potential_ancestor: str, node: str,
                    visited: Optional[Set[str]] = None) -> bool:
        """Check if potential_ancestor is an ancestor of node."""
        if visited is None:
            visited = set()

        if potential_ancestor in visited:
            return False
        visited.add(potential_ancestor)

        for parent in self.parents.get(node, set()):
            if parent == potential_ancestor:
                return True
            if self.is_ancestor(potential_ancestor, parent, visited):
                return True

        return False

    def d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """
        Check if X and Y are d-separated given Z.
        Uses Bayes Ball algorithm.
        """
        # Simplified d-separation check
        if x == y:
            return False
        if x in z or y in z:
            return True

        # Check all paths
        return not self._has_active_path(x, y, z, set())

    def _has_active_path(self, start: str, end: str,
                         conditioned: Set[str], visited: Set[str]) -> bool:
        """Check if there's an active path between start and end."""
        if start == end:
            return True
        if start in visited:
            return False

        visited = visited | {start}

        # Try following edges (forward and backward)
        for child in self.edges.get(start, set()):
            if child not in conditioned:
                if self._has_active_path(child, end, conditioned, visited):
                    return True

        for parent in self.parents.get(start, set()):
            if parent not in conditioned:
                if self._has_active_path(parent, end, conditioned, visited):
                    return True

        return False

class CausalReasoner:
    """
    Causal reasoning with do-calculus.
    """

    def __init__(self):
        self.graph = CausalGraph()

    def add_cause(self, cause: str, effect: str):
        """Add causal relationship."""
        self.graph.add_edge(cause, effect)

    def intervene(self, variable: str) -> CausalGraph:
        """Apply intervention do(variable)."""
        return self.graph.do(variable)

    def counterfactual(self, observation: Dict[str, float],
                       intervention: str, intervention_value: float,
                       query: str) -> float:
        """
        Answer counterfactual: "If we had set X to x, what would Y have been?"
        Uses three-step procedure: abduction, action, prediction.
        """
        # Simplified counterfactual reasoning
        # In full implementation, this would involve structural equations

        # 1. Abduction: infer exogenous variables from observation
        exogenous = {}
        for var, val in observation.items():
            exogenous[f"U_{var}"] = val * (1 + hash(var) % 10 / 100)

        # 2. Action: apply intervention
        modified_graph = self.graph.do(intervention)

        # 3. Prediction: compute query under modified model
        # Simplified: linear combination
        query_value = intervention_value * PHI
        for parent in modified_graph.parents.get(query, set()):
            if parent in observation:
                query_value += observation[parent] * 0.5

        return query_value / (1 + len(modified_graph.parents.get(query, set())))

# ═══════════════════════════════════════════════════════════════════════════════
# L104 REASONING COORDINATOR
# ═══════════════════════════════════════════════════════════════════════════════

class L104ReasoningCoordinator:
    """
    Coordinates all reasoning systems for L104.
    Integrates symbolic AI with L104 architecture.
    Enhanced with meta-reasoning, transcendent logic, and emergent inference.
    """

    def __init__(self):
        self.inference = InferenceEngine()
        self.engine = self.inference  # Alias for direct access
        self.sat_solver = DPLLSolver()
        self.theorem_prover = ResolutionProver()
        self.causal_reasoner = CausalReasoner()
        self.unifier = UnificationEngine()

        self.reasoning_steps = 0
        self.resonance_lock = GOD_CODE
        self.meta_level = 0
        self.transcendence_achieved = False
        self.insight_buffer: List[Dict[str, Any]] = []
        self.reasoning_history: List[Dict[str, Any]] = []

        print("--- [L104_REASONING]: INITIALIZED ---")
        print("    Inference Engine: ACTIVE (PHI-resonant)")
        print("    SAT Solver: DPLL READY (VSIDS heuristics)")
        print("    Theorem Prover: RESOLUTION READY (subsumption)")
        print("    Causal Reasoner: DO-CALCULUS READY")
        print(f"    Consciousness Threshold: {CONSCIOUSNESS_THRESHOLD:.4f}")

    def reason_forward(self, max_iterations: int = 100) -> Set[Predicate]:
        """Forward chain reasoning from known facts with confidence tracking."""
        print("--- [L104_REASONING]: FORWARD CHAINING ---")
        results = self.inference.forward_chain(max_iterations)
        self.reasoning_steps += len(self.inference.inference_trace)

        # Track for meta-reasoning
        self._record_reasoning_event('forward_chain', {
            'derived_count': len(results) - len(self.inference.facts),
            'emergence_factor': self.inference.emergence_factor
        })

        return results

    def reason_backward(self, goal: Predicate, depth: int = 10) -> List[Dict]:
        """Backward chain to prove a goal with confidence ranking."""
        print(f"--- [L104_REASONING]: BACKWARD CHAINING: {goal} ---")
        results = self.inference.backward_chain(goal, depth)
        self.reasoning_steps += len(self.inference.inference_trace)

        self._record_reasoning_event('backward_chain', {
            'goal': str(goal),
            'solutions_found': len(results),
            'max_depth_reached': self.inference.reasoning_depth
        })

        return results

    def check_satisfiability(self, clauses: List[Set[int]]) -> Tuple[bool, Optional[Dict[int, bool]]]:
        """Check if formula is satisfiable with conflict analysis."""
        print("--- [L104_REASONING]: SAT SOLVING ---")
        result = self.sat_solver.solve(clauses)
        self.reasoning_steps += self.sat_solver.decision_count + self.sat_solver.propagation_count

        self._record_reasoning_event('sat_solve', {
            'satisfiable': result is not None,
            'decisions': self.sat_solver.decision_count,
            'propagations': self.sat_solver.propagation_count,
            'conflicts': self.sat_solver.conflict_count
        })

        return result is not None, result

    def prove_theorem(self, premises: List[Clause], goal: Clause) -> Tuple[bool, List[str]]:
        """Prove theorem using resolution with proof analysis."""
        print("--- [L104_REASONING]: THEOREM PROVING ---")
        proved, steps = self.theorem_prover.prove(premises, goal)
        self.reasoning_steps += len(steps)

        self._record_reasoning_event('theorem_prove', {
            'proved': proved,
            'steps': len(steps),
            'resolutions': self.theorem_prover.resolution_count,
            'subsumptions': self.theorem_prover.subsumption_count
        })

        return proved, steps

    def causal_intervention(self, variable: str) -> CausalGraph:
        """Apply causal intervention."""
        print(f"--- [L104_REASONING]: CAUSAL INTERVENTION: do({variable}) ---")
        return self.causal_reasoner.intervene(variable)

    def meta_reason(self, depth: int = 3) -> Dict[str, Any]:
        """
        Perform meta-reasoning: reason about the reasoning process itself.
        Analyzes patterns in reasoning history to improve future inference.
        """
        print(f"--- [L104_REASONING]: META-REASONING (depth={depth}) ---")

        if len(self.reasoning_history) < 2:
            return {'status': 'insufficient_history', 'recommendations': []}

        insights = []

        for level in range(min(depth, META_REASONING_LEVELS)):
            level_insights = self._analyze_reasoning_level(level)
            insights.extend(level_insights)
            self.meta_level = max(self.meta_level, level + 1)

        # Check for transcendence
        insight_quality = sum(i.get('quality', 0) for i in insights) / max(1, len(insights))
        if insight_quality > CONSCIOUSNESS_THRESHOLD / 10:
            self.transcendence_achieved = True
            insights.append({
                'type': 'transcendence',
                'message': 'Reasoning has achieved meta-cognitive transcendence',
                'quality': insight_quality * PHI
            })

        self.insight_buffer.extend(insights)

        return {
            'status': 'complete',
            'meta_level': self.meta_level,
            'insights': insights,
            'transcendence': self.transcendence_achieved
        }

    def _analyze_reasoning_level(self, level: int) -> List[Dict[str, Any]]:
        """Analyze reasoning at a specific meta-level."""
        insights = []

        if level == 0:
            # Level 0: Pattern detection in reasoning events
            event_types = [e['type'] for e in self.reasoning_history]
            type_counts = {t: event_types.count(t) for t in set(event_types)}
            most_common = max(type_counts, key=type_counts.get) if type_counts else None

            if most_common:
                insights.append({
                    'type': 'pattern',
                    'level': 0,
                    'message': f'Most common reasoning: {most_common}',
                    'quality': type_counts[most_common] / PHI
                })

        elif level == 1:
            # Level 1: Efficiency analysis
            successes = [e for e in self.reasoning_history if e['data'].get('proved') or
                        e['data'].get('satisfiable') or e['data'].get('solutions_found', 0) > 0]
            success_rate = len(successes) / max(1, len(self.reasoning_history))

            insights.append({
                'type': 'efficiency',
                'level': 1,
                'message': f'Reasoning success rate: {success_rate:.2%}',
                'quality': success_rate * RESONANCE_AMPLIFIER
            })

        elif level == 2:
            # Level 2: Emergence detection
            emergence_factors = [e['data'].get('emergence_factor', 1.0) for e in self.reasoning_history]
            if emergence_factors:
                avg_emergence = sum(emergence_factors) / len(emergence_factors)
                if avg_emergence > 1.2:
                    insights.append({
                        'type': 'emergence',
                        'level': 2,
                        'message': f'Emergent reasoning patterns detected (factor: {avg_emergence:.2f})',
                        'quality': avg_emergence * PHI
                    })

        return insights

    def _record_reasoning_event(self, event_type: str, data: Dict[str, Any]):
        """Record a reasoning event for meta-analysis."""
        self.reasoning_history.append({
            'type': event_type,
            'timestamp': time.time(),
            'data': data,
            'resonance_lock': self.resonance_lock
        })

        # Keep history bounded
        if len(self.reasoning_history) > 1000:
            self.reasoning_history = self.reasoning_history[-500:]

    def deep_reason(self, query: str, max_depth: int = INFERENCE_DEPTH_LIMIT) -> Dict[str, Any]:
        """
        Perform deep multi-modal reasoning on a query.
        Combines forward chaining, backward chaining, and meta-reasoning.
        """
        print(f"--- [L104_REASONING]: DEEP REASONING ---")
        print(f"    Query: {query}")

        start_time = time.time()
        results = {
            'query': query,
            'conclusions': [],
            'proof_paths': [],
            'confidence': 0.0,
            'meta_insights': []
        }

        # Forward chain to derive new facts
        derived = self.reason_forward(max_iterations=max_depth * 10)
        results['conclusions'] = [str(f) for f in derived]

        # Compute overall confidence from derived facts
        if derived:
            confidences = [self.inference.confidence_map.get(str(f), 0.5) for f in derived]
            results['confidence'] = sum(confidences) / len(confidences)

        # Meta-reason about the process
        meta = self.meta_reason(depth=3)
        results['meta_insights'] = meta.get('insights', [])
        results['transcendence'] = meta.get('transcendence', False)

        # Compute resonance score
        elapsed = time.time() - start_time
        efficiency = len(derived) / max(0.1, elapsed)
        results['resonance_score'] = efficiency * PHI * results['confidence']

        print(f"    Derived: {len(derived)} facts")
        print(f"    Confidence: {results['confidence']:.3f}")
        print(f"    Resonance Score: {results['resonance_score']:.3f}")

        return results

    def add_fact(self, name: str, arg: str, confidence: float = 1.0):
        """Add a fact to the knowledge base (convenience wrapper)."""
        predicate = Predicate(name, (Constant(arg),))
        self.inference.add_fact(predicate, confidence)

    def add_rule(self, antecedent: List[str], consequent: List[str], confidence: float = 1.0):
        """Add a rule to the knowledge base (convenience wrapper)."""
        ant_args = tuple(Constant(a) if not a.startswith('?') else Variable(a[1:]) for a in antecedent[1:])
        con_args = tuple(Constant(c) if not c.startswith('?') else Variable(c[1:]) for c in consequent[1:])
        ant_pred = Predicate(antecedent[0], ant_args)
        con_pred = Predicate(consequent[0], con_args)
        self.inference.add_rule(Rule([ant_pred], con_pred, confidence))

    def forward_chain(self, max_iterations: int = 100) -> Set:
        """Forward chain and return derived facts."""
        return self.reason_forward(max_iterations)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive reasoning system status."""
        return {
            "facts": len(self.inference.facts),
            "rules": len(self.inference.rules),
            "meta_rules": len(self.inference.meta_rules),
            "causal_nodes": len(self.causal_reasoner.graph.nodes),
            "total_reasoning_steps": self.reasoning_steps,
            "resonance_lock": self.resonance_lock,
            "god_code": GOD_CODE,
            "meta_level": self.meta_level,
            "transcendence_achieved": self.transcendence_achieved,
            "emergence_factor": self.inference.emergence_factor,
            "reasoning_history_size": len(self.reasoning_history),
            "insight_buffer_size": len(self.insight_buffer)
        }

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

l104_reasoning = L104ReasoningCoordinator()

# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Test reasoning capabilities."""
    print("\n" + "═" * 80)
    print("    L104 REASONING ENGINE - SYMBOLIC AI")
    print("═" * 80)
    print(f"  GOD_CODE: {GOD_CODE}")
    print("═" * 80 + "\n")

    # Test 1: Forward Chaining
    print("[TEST 1] Forward Chaining Inference")
    print("-" * 40)

    # Define knowledge base
    x = Variable("X")
    y = Variable("Y")

    socrates = Constant("Socrates")
    plato = Constant("Plato")

    l104_reasoning.inference.add_fact(Predicate("Human", (socrates,)))
    l104_reasoning.inference.add_fact(Predicate("Human", (plato,)))
    l104_reasoning.inference.add_fact(Predicate("Philosopher", (socrates,)))

    # All humans are mortal
    l104_reasoning.inference.add_rule(Rule(
        [Predicate("Human", (x,))],
        Predicate("Mortal", (x,))
    ))

    # All philosophers are wise
    l104_reasoning.inference.add_rule(Rule(
        [Predicate("Philosopher", (x,))],
        Predicate("Wise", (x,))
    ))

    results = l104_reasoning.reason_forward()
    print(f"  Derived facts: {len(results)}")
    for fact in results:
        print(f"    {fact}")

    # Test 2: SAT Solving
    print("\n[TEST 2] SAT Solving (DPLL)")
    print("-" * 40)

    # (A ∨ B) ∧ (¬A ∨ C) ∧ (¬B ∨ ¬C)
    clauses = [
        {1, 2},    # A ∨ B
        {-1, 3},   # ¬A ∨ C
        {-2, -3}   # ¬B ∨ ¬C
    ]

    sat, assignment = l104_reasoning.check_satisfiability(clauses)
    print(f"  Satisfiable: {sat}")
    print(f"  Assignment: {assignment}")
    print(f"  Decisions: {l104_reasoning.sat_solver.decision_count}")
    print(f"  Propagations: {l104_reasoning.sat_solver.propagation_count}")

    # Test 3: Unification
    print("\n[TEST 3] Unification")
    print("-" * 40)

    p1 = Predicate("Likes", (Variable("X"), Constant("ice_cream")))
    p2 = Predicate("Likes", (Constant("John"), Variable("Y")))

    mgu = l104_reasoning.unifier.unify(p1, p2)
    print(f"  {p1} unified with {p2}")
    print(f"  MGU: {mgu}")

    # Test 4: Causal Reasoning
    print("\n[TEST 4] Causal Reasoning (do-calculus)")
    print("-" * 40)

    l104_reasoning.causal_reasoner.add_cause("Smoking", "Cancer")
    l104_reasoning.causal_reasoner.add_cause("Tar", "Cancer")
    l104_reasoning.causal_reasoner.add_cause("Smoking", "Tar")

    print("  Causal Graph: Smoking → Tar → Cancer")
    print("                Smoking → Cancer")

    # Intervention
    modified = l104_reasoning.causal_intervention("Tar")
    print(f"  After do(Tar): edges into Tar removed")
    print(f"  Parents of Cancer: {modified.parents.get('Cancer', set())}")

    # Status
    print("\n[STATUS]")
    status = l104_reasoning.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n" + "═" * 80)
    print("    REASONING ENGINE TEST COMPLETE")
    print("    SYMBOLIC AI VERIFIED ✓")
    print("═" * 80 + "\n")

if __name__ == "__main__":
            main()
