#!/usr/bin/env python3
"""L104 Sovereign Node — Full Benchmark Suite (Formal Logic + Deep NLU + ASI v10.0)"""
import json, time, sys

print("=" * 70)
print("  L104 SOVEREIGN NODE — FULL BENCHMARK SUITE")
print("  Formal Logic + Deep NLU + ASI v10.0 Benchmarks")
print("=" * 70)
print()

# ── Phase 1: ASI Benchmark Harness (MMLU, HumanEval, MATH, ARC) ──
print("[PHASE 1] Running ASI Benchmark Harness (offline)...")
from l104_asi import asi_core
report = asi_core.run_benchmarks()
print(f"  Composite Score: {report.get('composite_score', 'N/A')}")
print(f"  Verdict: {report.get('verdict', 'N/A')}")
for bm, data in report.get("benchmarks", {}).items():
    print(f"    {bm}: {data.get('correct',0)}/{data.get('total',0)} = {data.get('score',0):.4f}")
print()

# ── Phase 2: Formal Logic Engine ──
print("[PHASE 2] Formal Logic Engine Benchmarks...")
from l104_asi.formal_logic import FormalLogicEngine, Atom, Not, And, Or, Implies, Iff
fle = FormalLogicEngine()

# 2a: Truth table generation
t0 = time.time()
p, q, r = Atom("P"), Atom("Q"), Atom("R")
tables_generated = 0
for combo in [(And(p, q), "P^Q"), (Or(p, Not(q)), "PvnQ"), (Implies(p, q), "P->Q"),
              (Iff(p, q), "P<->Q"), (And(Or(p,q), Not(r)), "(PvQ)^nR")]:
    fle.generate_truth_table(combo[0])
    tables_generated += 1
tt_time = time.time() - t0
print(f"  Truth Tables: {tables_generated}/5 generated in {tt_time:.3f}s")

# 2b: Equivalence proving
t0 = time.time()
equiv_tests = [
    (Not(And(p, q)), Or(Not(p), Not(q)), "De Morgan 1", True),
    (Not(Or(p, q)), And(Not(p), Not(q)), "De Morgan 2", True),
    (Implies(p, q), Implies(Not(q), Not(p)), "Contrapositive", True),
    (Implies(p, q), Or(Not(p), q), "Material Conditional", True),
    (Implies(p, q), Implies(q, p), "Converse (should fail)", False),
]
equiv_correct = 0
for lhs, rhs, name, expected in equiv_tests:
    result = fle.prove_equivalence(lhs, rhs)
    actual = result.get("equivalent", False)
    if actual == expected:
        equiv_correct += 1
    else:
        print(f"    FAIL: {name} — expected {expected}, got {actual}")
eq_time = time.time() - t0
print(f"  Equivalence Proving: {equiv_correct}/{len(equiv_tests)} correct in {eq_time:.3f}s")

# 2c: Fallacy detection
t0 = time.time()
fallacy_texts = [
    "You cant trust his policy because hes a bad person.",
    "Everyone is doing it so it must be right.",
    "Either you support the bill or you hate freedom.",
    "He wants to reduce spending so he wants to abolish everything.",
    "A famous actor said this product works so it must be true.",
    "We should not listen to him because he is young.",
    "If we allow this then everything will collapse.",
    "That cant be true because it would be inconvenient.",
]
total_detected = 0
for text in fallacy_texts:
    detected = fle.detect_fallacies(text)
    total_detected += len(detected)
fd_time = time.time() - t0
print(f"  Fallacy Detection: {total_detected} fallacies found in {len(fallacy_texts)} texts ({fd_time:.3f}s)")

# 2d: NL->Logic translation
t0 = time.time()
nl_tests = [
    "If it rains then the ground is wet",
    "All humans are mortal",
    "It is not the case that the sun is cold",
    "Either we go to the park or we stay home",
    "No reptiles are mammals",
]
translations_ok = 0
for text in nl_tests:
    result = fle.translate_to_logic(text)
    if "formalization" in result:
        translations_ok += 1
nl_time = time.time() - t0
print(f"  NL->Logic Translation: {translations_ok}/{len(nl_tests)} translated in {nl_time:.3f}s")

# 2e: Argument analysis
t0 = time.time()
arguments = [
    (["If it rains the ground is wet", "It is raining"], "The ground is wet"),
    (["All men are mortal", "Socrates is a man"], "Socrates is mortal"),
    (["All birds can fly", "Penguins are birds"], "Penguins can fly"),
]
args_analyzed = 0
for premises, conclusion in arguments:
    result = fle.analyze_argument(premises, conclusion)
    if isinstance(result, dict):
        args_analyzed += 1
aa_time = time.time() - t0
print(f"  Argument Analysis: {args_analyzed}/{len(arguments)} analyzed in {aa_time:.3f}s")

logic_status = fle.status()
print(f"  Logic Depth Score: {logic_status['logic_depth_score']}")
print(f"  Known Fallacies: {logic_status['fallacies_known']}")
print(f"  Known Laws: {logic_status['logical_laws_known']}")
print(f"  Valid Syllogism Forms: {logic_status['valid_syllogism_forms']}")
print()

# ── Phase 3: Deep NLU Engine ──
print("[PHASE 3] Deep NLU Engine Benchmarks...")
from l104_asi.deep_nlu import DeepNLUEngine
nlu = DeepNLUEngine()

# 3a: Sentiment analysis
t0 = time.time()
sentiment_tests = [
    ("I absolutely love this amazing product!", "positive"),
    ("This is terrible and I hate it.", "negative"),
    ("The meeting is at 3pm.", "neutral"),
    ("What a wonderful surprise!", "positive"),
    ("I am devastated by the tragic news.", "negative"),
    ("Water boils at 100 degrees Celsius.", "neutral"),
]
sentiment_correct = 0
for text, expected in sentiment_tests:
    result = nlu.analyze_sentiment(text)
    score = result.get("score", 0)
    if expected == "positive" and score > 0:
        sentiment_correct += 1
    elif expected == "negative" and score < 0:
        sentiment_correct += 1
    elif expected == "neutral" and abs(score) <= 1.5:
        sentiment_correct += 1
sa_time = time.time() - t0
print(f"  Sentiment Analysis: {sentiment_correct}/{len(sentiment_tests)} correct in {sa_time:.3f}s")

# 3b: Pragmatics / Intent classification
t0 = time.time()
intent_tests = [
    "What time is the meeting?",
    "Please close the door.",
    "I promise to finish this by Friday.",
    "Could you help me with this task?",
    "The report has been submitted.",
    "Wow, that is incredible!",
]
intents_classified = 0
for text in intent_tests:
    result = nlu.classify_intent(text)
    if "intent" in result:
        intents_classified += 1
pr_time = time.time() - t0
print(f"  Intent Classification: {intents_classified}/{len(intent_tests)} classified in {pr_time:.3f}s")

# 3c: Discourse analysis
t0 = time.time()
discourse_texts = [
    ["The economy grew last year.", "However, unemployment also rose.", "Therefore, the growth was uneven."],
    ["First, preheat the oven.", "Then, mix the ingredients.", "Finally, bake for 30 minutes."],
    ["Studies show coffee is healthy.", "But some researchers disagree.", "More research is needed."],
]
discourse_ok = 0
for sentences in discourse_texts:
    result = nlu.analyze_discourse(sentences)
    if "relations" in result:
        discourse_ok += 1
dc_time = time.time() - t0
print(f"  Discourse Analysis: {discourse_ok}/{len(discourse_texts)} analyzed in {dc_time:.3f}s")

# 3d: Morphological analysis
t0 = time.time()
morph_words = ["unhappiness", "reorganization", "misbehaving", "counterproductive", "internationalize"]
morph_ok = 0
for word in morph_words:
    result = nlu.analyze_morphology(word)
    if result.get("prefixes") or result.get("suffixes"):
        morph_ok += 1
mr_time = time.time() - t0
print(f"  Morphological Analysis: {morph_ok}/{len(morph_words)} decomposed in {mr_time:.3f}s")

# 3e: Semantic role labeling
t0 = time.time()
srl_tests = [
    "The cat chased the mouse across the yard.",
    "She gave the book to her friend yesterday.",
    "The engineer built a bridge with steel.",
]
srl_ok = 0
for text in srl_tests:
    result = nlu.label_semantic_roles(text)
    if result.get("roles"):
        srl_ok += 1
srl_time = time.time() - t0
print(f"  Semantic Role Labeling: {srl_ok}/{len(srl_tests)} frames extracted in {srl_time:.3f}s")

# 3f: Deep comprehension fusion
t0 = time.time()
deep_texts = [
    "Although the weather was terrible, Sarah decided to go hiking because she needed to clear her mind. She grabbed her rain jacket and headed out.",
    "John told Mary that he would finish the report by Friday. She was relieved because the deadline was Monday.",
]
deep_ok = 0
for text in deep_texts:
    result = nlu.deep_analyze(text)
    if "comprehension_depth" in result:
        deep_ok += 1
dp_time = time.time() - t0
print(f"  Deep Comprehension: {deep_ok}/{len(deep_texts)} fused in {dp_time:.3f}s")

nlu_status = nlu.status()
print(f"  NLU Depth Score: {nlu_status['nlu_depth_score']}")
print(f"  Layers: {nlu_status['layers']}")
print(f"  Analyses Performed: {nlu_status['analyses_performed']}")
print()

# ── Phase 4: ASI Score with new dimensions ──
print("[PHASE 4] ASI Score with Formal Logic + Deep NLU dimensions...")
asi_score = asi_core.compute_asi_score()
print(f"  ASI Score: {round(asi_score, 6) if isinstance(asi_score, float) else asi_score}")
print(f"  Formal Logic Score: {round(asi_core.formal_logic_score(), 4)}")
print(f"  Deep NLU Score: {round(asi_core.deep_nlu_score(), 4)}")
status = asi_core.get_status()
print(f"  Scoring Dimensions: {status.get('scoring_dimensions', '?')}")
print(f"  Version: {status.get('version', '?')}")
print()

# ── Phase 5: Test Suite ──
print("[PHASE 5] Unit Test Suite (84 tests)...")
import subprocess
test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_formal_logic_and_deep_nlu.py", "-v", "--tb=line", "-q"],
    capture_output=True, text=True, cwd="/Users/carolalvarez/Applications/Allentown-L104-Node"
)
# Extract summary line
for line in test_result.stdout.strip().split("\n")[-5:]:
    print(f"  {line}")
test_passed = "passed" in test_result.stdout and "failed" not in test_result.stdout
print()

# ── Final Summary ──
print("=" * 70)
print("  BENCHMARK RESULTS SUMMARY")
print("=" * 70)
composite = report.get("composite_score", 0)
logic_score = logic_status["logic_depth_score"]
nlu_score_val = nlu_status["nlu_depth_score"]

summary = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "asi_benchmark_harness": {
        "composite_score": composite,
        "verdict": report.get("verdict", "N/A"),
        "benchmarks": report.get("benchmarks", {}),
    },
    "formal_logic_engine": {
        "version": logic_status["version"],
        "layers": 8,
        "truth_tables": f"{tables_generated}/5",
        "equivalence_proving": f"{equiv_correct}/{len(equiv_tests)}",
        "fallacy_detection": f"{total_detected} in {len(fallacy_texts)} texts",
        "nl_translation": f"{translations_ok}/{len(nl_tests)}",
        "argument_analysis": f"{args_analyzed}/{len(arguments)}",
        "logic_depth_score": logic_score,
        "fallacies_known": logic_status["fallacies_known"],
        "laws_known": logic_status["logical_laws_known"],
    },
    "deep_nlu_engine": {
        "version": nlu_status["version"],
        "layers": 10,
        "sentiment_accuracy": f"{sentiment_correct}/{len(sentiment_tests)}",
        "intent_classification": f"{intents_classified}/{len(intent_tests)}",
        "discourse_analysis": f"{discourse_ok}/{len(discourse_texts)}",
        "morphological_analysis": f"{morph_ok}/{len(morph_words)}",
        "semantic_role_labeling": f"{srl_ok}/{len(srl_tests)}",
        "deep_comprehension": f"{deep_ok}/{len(deep_texts)}",
        "nlu_depth_score": nlu_score_val,
    },
    "asi_integration": {
        "asi_score": round(asi_score, 6) if isinstance(asi_score, float) else str(asi_score),
        "formal_logic_dimension": round(asi_core.formal_logic_score(), 4),
        "deep_nlu_dimension": round(asi_core.deep_nlu_score(), 4),
        "scoring_dimensions": status.get("scoring_dimensions", "?"),
    },
    "test_suite": {
        "total_tests": 84,
        "all_passed": test_passed,
    },
}

print(json.dumps(summary, indent=2, default=str))

# Save to file
with open("benchmark_logic_nlu_results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print()
print("Results saved to benchmark_logic_nlu_results.json")
print("=" * 70)
