#!/usr/bin/env python3
"""Validation tests for v8.0.0 language comprehension algorithms."""
import sys
sys.path.insert(0, '.')

from l104_asi.language_comprehension import (
    CoreferenceResolver,
    SentimentAnalyzer,
    SemanticFrameAnalyzer,
    TaxonomyClassifier,
    CausalChainReasoner,
    PragmaticInferenceEngine,
    ConceptNetLinker,
    LanguageComprehensionEngine,
)

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))

# ── Test 1: CoreferenceResolver ──────────────────────────────────────────────
print("\n=== TEST 1: CoreferenceResolver ===")
coref = CoreferenceResolver()
r = coref.resolve("Einstein developed relativity. He won the Nobel Prize.")
test("Pronoun resolved", r["resolution_count"] > 0, f"{r['resolution_count']} resolutions")
if r["resolutions"]:
    test("He→Einstein", "Einstein" in r["resolutions"][0]["antecedent"],
         f"antecedent={r['resolutions'][0]['antecedent']}")
resolved = coref.resolve_for_scoring("Marie Curie studied radioactivity. She won two Nobel Prizes.")
test("She→Marie Curie", "Marie" in resolved or "Curie" in resolved, f"resolved={resolved[:80]}")

# ── Test 2: SentimentAnalyzer ────────────────────────────────────────────────
print("\n=== TEST 2: SentimentAnalyzer ===")
sa = SentimentAnalyzer()
pos = sa.analyze("This is an excellent and wonderful achievement.")
test("Positive detected", pos["label"] == "positive", f"polarity={pos['polarity']}")
neg = sa.analyze("This is a terrible and awful disaster.")
test("Negative detected", neg["label"] == "negative", f"polarity={neg['polarity']}")
negated = sa.analyze("This is not good at all.")
test("Negation flips", negated["polarity"] < 0, f"polarity={negated['polarity']}")
cmp = sa.compare_sentiment("Great work!", "Terrible failure.")
test("Compare works", not cmp["agree"], f"diff={cmp['polarity_difference']}")

# ── Test 3: SemanticFrameAnalyzer ────────────────────────────────────────────
print("\n=== TEST 3: SemanticFrameAnalyzer ===")
sfa = SemanticFrameAnalyzer()
r1 = sfa.analyze("What causes global warming?")
test("CAUSE_EFFECT detected", r1["primary_frame"] == "CAUSE_EFFECT", f"frame={r1['primary_frame']}")
r2 = sfa.analyze("What is the definition of entropy?")
test("DEFINITION detected", r2["primary_frame"] == "DEFINITION", f"frame={r2['primary_frame']}")
r3 = sfa.analyze("Where is the Great Barrier Reef located?")
test("LOCATION detected", r3["primary_frame"] == "LOCATION", f"frame={r3['primary_frame']}")
r4 = sfa.analyze("How many chromosomes do humans have?")
test("QUANTITY detected", r4["primary_frame"] == "QUANTITY", f"frame={r4['primary_frame']}")
fit = sfa.score_choice_frame_fit("What causes rust?", "Oxidation of iron in the presence of moisture")
test("Frame fit score > 0", fit > 0, f"fit={fit}")

# ── Test 4: TaxonomyClassifier ───────────────────────────────────────────────
print("\n=== TEST 4: TaxonomyClassifier ===")
tax = TaxonomyClassifier()
test("mitosis is-a cell division", tax.is_a("mitosis", "cell division"))
test("mitosis is-a biological process", tax.is_a("mitosis", "biological process"))
test("bird NOT is-a cell division", not tax.is_a("bird", "cell division"))
test("proton part-of atom", tax.part_of("proton", "atom"))
test("nucleus part-of cell", tax.part_of("nucleus", "cell"))
sim = tax.taxonomic_similarity("mitosis", "meiosis")
test("mitosis~meiosis similar", sim > 0.5, f"sim={sim}")
dist = tax.taxonomic_distance("mitosis", "meiosis")
test("distance < 0.5", dist < 0.5, f"dist={dist}")

# ── Test 5: CausalChainReasoner ──────────────────────────────────────────────
print("\n=== TEST 5: CausalChainReasoner ===")
ccr = CausalChainReasoner()
fwd = ccr.forward_chain("smoking")
test("Forward chain from smoking", len(fwd) > 0, f"effects={len(fwd)}")
test("Lung cancer reachable", any(e["effect"] == "lung cancer" for e in fwd))
bwd = ccr.backward_chain("evolution")
test("Backward chain to evolution", len(bwd) > 0, f"causes={len(bwd)}")
strength = ccr.causal_link_strength("smoking", "lung cancer")
test("Smoking→lung cancer strength", strength > 0.5, f"strength={strength}")
multi = ccr.causal_link_strength("mutation", "evolution")
test("Mutation→evolution multi-hop", multi > 0, f"strength={multi}")

# ── Test 6: PragmaticInferenceEngine ─────────────────────────────────────────
print("\n=== TEST 6: PragmaticInferenceEngine ===")
pie = PragmaticInferenceEngine()
impl = pie.detect_implicatures("Some students passed the exam.")
test("Scalar implicature: some→not all", len(impl) > 0, f"implicatures={len(impl)}")
if impl:
    test("Trigger is 'some'", impl[0]["trigger"] == "some")
presup = pie.detect_presuppositions("He stopped smoking last year.")
test("Presupposition: stopped→prior activity", len(presup) > 0, f"presuppositions={len(presup)}")
speech = pie.classify_speech_act("What is the atomic number of iron?")
test("Speech act: question", speech["type"] == "question", f"type={speech['type']}")
hedges = pie.detect_hedges("The result is probably approximately correct.")
test("Hedges detected", hedges["hedge_count"] >= 2, f"count={hedges['hedge_count']}")
align = pie.pragmatic_alignment("Some students passed", "All students passed")
test("Implicature penalty", align < 0, f"score={align}")

# ── Test 7: ConceptNetLinker ─────────────────────────────────────────────────
print("\n=== TEST 7: ConceptNetLinker ===")
cn = ConceptNetLinker()
bird_rels = cn.query("bird")
test("Bird has relations", len(bird_rels) > 0, f"relations={list(bird_rels.keys())}")
test("Bird HasA wings", "wings" in bird_rels.get("HasA", []))
test("Bird CapableOf fly", "fly" in bird_rels.get("CapableOf", []))
rev = cn.reverse_query("wings", "HasA")
test("Reverse: wings→bird", "bird" in rev, f"subjects={rev}")
related = cn.related("heat", "expansion")
test("Heat Causes expansion", len(related) > 0, f"relations={related}")
score = cn.score_choice_commonsense("What does a bird have?", "wings")
test("Commonsense score > 0", score > 0, f"score={score}")

# ── Test 8: Version & Layer Count ────────────────────────────────────────────
print("\n=== TEST 8: Version & Layers ===")
engine = LanguageComprehensionEngine()
status = engine.get_status()
test("Version 13.0.0", status["version"] == "13.0.0", f"version={status['version']}")
layer_count = len(status["layers"])
test("28 layer keys", layer_count == 28, f"layers={layer_count}")
test("v8_algorithms present", "v8_algorithms" in status)
v8 = status.get("v8_algorithms", {})
test("All 7 v8 algorithms", all([
    v8.get("coreference_resolver"),
    v8.get("sentiment_analyzer"),
    v8.get("semantic_frame_analyzer"),
    v8.get("taxonomy_classifier"),
    v8.get("causal_chain_reasoner"),
    v8.get("pragmatic_inference"),
    v8.get("commonsense_linker"),
]), f"v8={v8}")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*60}")
sys.exit(0 if failed == 0 else 1)
