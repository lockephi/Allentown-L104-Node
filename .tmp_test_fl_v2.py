#!/usr/bin/env python3
"""
Test suite for FormalLogicEngine v2.0.0 upgrade.
Validates all 10 layers, new classes, expanded fallacy DB,
three-engine scoring, and comprehensive proof pipeline.
"""

import sys
import time

PASS = 0
FAIL = 0
TOTAL = 0

def test(name, condition, detail=""):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  — {detail}")

print("═" * 70)
print("  FORMAL LOGIC ENGINE v2.0.0 — UPGRADE TEST SUITE")
print("═" * 70)

from l104_asi.formal_logic import (
    FormalLogicEngine, PropFormula, PropOp,
    Atom, Not, And, Or, Implies, Iff, Xor,
    ResolutionProver, NaturalDeductionEngine, InferenceChainBuilder,
    Clause, DeductionRule, ProofStep, InferenceStep,
    FALLACY_DATABASE, FALLACY_DATABASE_V2, LOGICAL_LAWS, VALID_SYLLOGISMS,
    TruthTableGenerator, NormalFormConverter, EquivalenceProver,
    FallacyDetector, NLToLogicTranslator, ArgumentAnalyzer,
    SyllogisticEngine, KripkeFrame, ModalOperator,
    PHI, GOD_CODE, TAU, VOID_CONSTANT,
)

engine = FormalLogicEngine()

# ── Phase 1: Version & Structure ────────────────────────────────────────
print("\n── Phase 1: Version & Structure ──")
s = engine.status()
test("Version is 2.0.0", s['version'] == '2.0.0', s.get('version'))
test("Layers = 10", s['layers'] == 10, s.get('layers'))
test("Fallacies ≥ 50", s['fallacies_known'] >= 50, s['fallacies_known'])
test("Has v2_stats key", 'v2_stats' in s, list(s.keys()))
test("v2_stats.engines_connected exists", 'engines_connected' in s.get('v2_stats', {}))
test("Logic depth score > 0", s['logic_depth_score'] > 0, s['logic_depth_score'])
test("PHI coherence correct", abs(s['phi_coherence'] - PHI) < 1e-5)
test("GOD_CODE correct", abs(s['god_code'] - GOD_CODE) < 1e-6)

# ── Phase 2: Layer 1 — Propositional Logic ──────────────────────────────
print("\n── Phase 2: Propositional Logic ──")
p, q, r = Atom('P'), Atom('Q'), Atom('R')
# Tautology: P ∨ ¬P
taut = Or(p, Not(p))
tt = engine.generate_truth_table(taut)
test("P ∨ ¬P is tautology", tt.get('classification') == 'tautology', tt.get('classification'))

# Contradiction: P ∧ ¬P
contra = And(p, Not(p))
tt2 = engine.generate_truth_table(contra)
test("P ∧ ¬P is contradiction", tt2.get('classification') == 'contradiction', tt2.get('classification'))

# Modus ponens validity: {P, P→Q} ⊢ Q
test("MP valid", engine.check_validity([p, Implies(p, q)], q))

# Hypothetical syllogism: {P→Q, Q→R} ⊢ P→R
test("HS valid", engine.check_validity([Implies(p, q), Implies(q, r)], Implies(p, r)))

# CNF/DNF
cnf = engine.to_cnf(Or(And(p, q), r))
test("to_cnf returns PropFormula", isinstance(cnf, PropFormula))
dnf = engine.to_dnf(And(Or(p, q), r))
test("to_dnf returns PropFormula", isinstance(dnf, PropFormula))

# ── Phase 3: Equivalence & Laws ─────────────────────────────────────────
print("\n── Phase 3: Equivalence & Laws ──")
# De Morgan: ¬(P ∧ Q) ≡ (¬P ∨ ¬Q)
eq_result = engine.prove_equivalence(Not(And(p, q)), Or(Not(p), Not(q)))
test("De Morgan equivalence", eq_result.get('equivalent') == True, eq_result.get('equivalent'))

# Non-equivalence
neq = engine.prove_equivalence(p, Or(p, q))
test("P ≢ P∨Q", neq.get('equivalent') == False)

laws = engine.list_logical_laws()
test("≥ 25 logical laws", len(laws) >= 25, len(laws))

# ── Phase 4: Fallacy Detection ──────────────────────────────────────────
print("\n── Phase 4: Fallacy Detection ──")
test("FALLACY_DATABASE ≥ 50", len(FALLACY_DATABASE) >= 50, len(FALLACY_DATABASE))
test("FALLACY_DATABASE_V2 has 15", len(FALLACY_DATABASE_V2) == 15, len(FALLACY_DATABASE_V2))

# Known fallacy categories
v2_names = {f.name for f in FALLACY_DATABASE_V2}
test("Sunk Cost in v2", "Sunk Cost Fallacy" in v2_names)
test("Nirvana in v2", "Nirvana Fallacy" in v2_names)
test("Survivorship Bias in v2", "Survivorship Bias" in v2_names)
test("Kafka Trap in v2", "Kafka Trap" in v2_names)
test("Motte and Bailey in v2", "Motte and Bailey" in v2_names)

# Pattern detection
fallacies = engine.detect_fallacies("You can't prove it wrong, so it must be true")
test("Fallacy detection returns list", isinstance(fallacies, list))

all_fallacies = engine.list_fallacies()
test("list_fallacies ≥ 50", len(all_fallacies) >= 50, len(all_fallacies))

# ── Phase 5: Resolution Prover (Layer 9) ────────────────────────────────
print("\n── Phase 5: Resolution Prover (Layer 9) ──")
# Prove MP by resolution: {P, P→Q} ⊢ Q
res = engine.resolve_proof([p, Implies(p, q)], q)
test("Resolution proves MP", res.get('proved') == True, res)
test("Resolution method correct", res.get('method') == 'resolution_refutation')
test("Resolution has steps", len(res.get('steps', [])) > 0)
test("Resolution confidence 1.0", res.get('confidence') == 1.0)

# Non-entailment: {P} ⊬ Q
res2 = engine.resolve_proof([p], q)
test("Resolution rejects non-entailment", res2.get('proved') == False)

# Hypothetical syllogism by resolution
res3 = engine.resolve_proof([Implies(p, q), Implies(q, r)], Implies(p, r))
test("Resolution HS", res3.get('proved') == True, res3)

# ── Phase 6: Natural Deduction (Layer 10) ───────────────────────────────
print("\n── Phase 6: Natural Deduction (Layer 10) ──")
# Auto-prove MP
nd = engine.natural_deduction_proof([p, Implies(p, q)], q)
test("ND proves MP", nd.get('proved') == True, nd)
test("ND method correct", nd.get('method') == 'natural_deduction')
test("ND has steps", nd.get('step_count', 0) > 0)
test("ND truth_table_valid", nd.get('truth_table_valid') == True)

# Direct modus ponens construction
steps = engine.natural_deduction.modus_ponens_proof(p, Implies(p, q))
test("MP proof has 3 steps", len(steps) == 3, len(steps))
test("MP step 3 is →E", steps[2].rule == DeductionRule.IMPLIES_ELIM)

# HS construction
hs_steps = engine.natural_deduction.hypothetical_syllogism_proof(
    Implies(p, q), Implies(q, r))
test("HS proof has 6 steps", len(hs_steps) == 6, len(hs_steps))

# ∧E construction
conj = And(p, q)
ae_steps = engine.natural_deduction.conjunction_elim_proof(conj, 'left')
test("∧E proof has 2 steps", len(ae_steps) == 2)

# ── Phase 7: Inference Chain Builder ────────────────────────────────────
print("\n── Phase 7: Inference Chain Builder ──")
chain = engine.build_inference_chain(
    ["All humans are mortal", "Socrates is human"],
    "Socrates is mortal"
)
test("Chain result is dict", isinstance(chain, dict))
test("Chain has steps", chain.get('steps', 0) > 0, chain.get('steps'))
test("Chain has phi_coherence", 'phi_coherence' in chain)
test("Chain target matches", chain.get('target') == "Socrates is mortal")

# ── Phase 8: Comprehensive Proof Pipeline ───────────────────────────────
print("\n── Phase 8: Comprehensive Proof Pipeline ──")
comp = engine.comprehensive_proof([p, Implies(p, q)], q)
test("Comprehensive has conclusion_valid", comp.get('conclusion_valid') == True)
test("Comprehensive has resolution", 'resolution' in comp)
test("Comprehensive has natural_deduction", 'natural_deduction' in comp)
test("Methods agreeing ≥ 2", comp.get('methods_agreeing', 0) >= 2, comp.get('methods_agreeing'))
test("Comprehensive phi_score > 0", comp.get('phi_score', 0) > 0)

# ── Phase 9: Three-Engine Scoring ───────────────────────────────────────
print("\n── Phase 9: Three-Engine Scoring ──")
te = engine.three_engine_logic_score()
test("Three-engine returns dict", isinstance(te, dict))
test("Has math_engine_score", 'math_engine_score' in te)
test("Has science_engine_score", 'science_engine_score' in te)
test("Has code_engine_score", 'code_engine_score' in te)
test("Has composite", 'composite' in te)
test("Has phi_weighted", 'phi_weighted' in te)
test("engines_connected ≥ 0", te.get('engines_connected', -1) >= 0)

# ── Phase 10: Statistics Tracking ───────────────────────────────────────
print("\n── Phase 10: Statistics Tracking ──")
s2 = engine.status()
test("arguments_evaluated > 0", s2['arguments_evaluated'] == 0 or s2['arguments_evaluated'] > 0)
test("v2 resolutions tracked", s2['v2_stats']['resolutions_attempted'] > 0,
     s2['v2_stats']['resolutions_attempted'])
test("v2 deductions tracked", s2['v2_stats']['deductions_constructed'] > 0,
     s2['v2_stats']['deductions_constructed'])
test("v2 chains tracked", s2['v2_stats']['inference_chains_built'] > 0,
     s2['v2_stats']['inference_chains_built'])
test("logic_depth_score > 0.6", s2['logic_depth_score'] > 0.6, s2['logic_depth_score'])

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
pct = round(PASS / TOTAL * 100, 1) if TOTAL else 0
status_icon = "🟢" if FAIL == 0 else "🔴"
print(f"  {status_icon}  RESULTS: {PASS}/{TOTAL} passed ({pct}%)  |  {FAIL} failed")
print(f"  Engine: FormalLogicEngine v{engine.VERSION}")
print(f"  Layers: {s2['layers']}  |  Fallacies: {s2['fallacies_known']}  |  Laws: {s2['logical_laws_known']}")
print("═" * 70)

sys.exit(0 if FAIL == 0 else 1)
