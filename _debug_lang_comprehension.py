#!/usr/bin/env python3
"""
L104 Language Comprehension Pipeline — Full Debug Suite
═══════════════════════════════════════════════════════════
Exercises every layer of the 25-layer LanguageComprehensionEngine
and the 12-stage MCQSolver pipeline with diagnostic output.
"""

import sys
import time
import traceback

sys.path.insert(0, ".")

# ─────────────────────────────────────────────────────────────────────
#  PHASE 1: Engine Boot + Initialization
# ─────────────────────────────────────────────────────────────────────

def banner(text):
    w = 70
    print(f"\n{'═' * w}")
    print(f"  {text}")
    print(f"{'═' * w}")


def phase_header(n, text):
    print(f"\n  ━━━ Phase {n}: {text} ━━━")


def ok(msg):
    print(f"    ✓ {msg}")


def fail(msg):
    print(f"    ✗ {msg}")


def info(msg):
    print(f"    · {msg}")


banner("L104 LANGUAGE COMPREHENSION PIPELINE — DEBUG SUITE")

t_total = time.time()
errors = []

# ── Phase 1: Boot ────────────────────────────────────────────────────
phase_header(1, "Engine Boot & Initialization")

t0 = time.time()
try:
    from l104_asi.language_comprehension import LanguageComprehensionEngine
    ok("Import LanguageComprehensionEngine")
except Exception as e:
    fail(f"Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

lce = LanguageComprehensionEngine()
ok(f"Instantiated v{lce.VERSION}")

t_init = time.time()
try:
    lce.initialize()
    dt = (time.time() - t_init) * 1000
    ok(f"initialize() completed in {dt:.0f}ms")
except Exception as e:
    dt = (time.time() - t_init) * 1000
    fail(f"initialize() FAILED after {dt:.0f}ms: {e}")
    traceback.print_exc()
    errors.append(("Phase 1: initialize()", str(e)))

# ── Phase 2: Layer Health Check ──────────────────────────────────────
phase_header(2, "Layer Health Check (25 layers)")

status = lce.get_status()
layers = status.get("layers", {})
total_layers = len(layers)
active_layers = sum(1 for v in layers.values() if v)
info(f"{active_layers}/{total_layers} layers active")

for layer_name, active in sorted(layers.items()):
    sym = "✓" if active else "✗"
    print(f"      {sym}  {layer_name}")

# Engine support
engines = status.get("engine_support", {})
connected = sum(1 for v in engines.values() if v)
info(f"Engine connections: {connected}/{len(engines)}")
for eng, active in engines.items():
    sym = "✓" if active else "·"
    print(f"      {sym}  {eng}")

# v9 pipeline features
v9 = status.get("v9_pipeline", {})
info(f"v9 pipeline: cache={v9.get('query_cache_size', '?')}, "
     f"nlu_cache={v9.get('nlu_cache_size', '?')}, "
     f"enrichment_guard={v9.get('enrichment_guard', '?')}")

# ── Phase 3: Knowledge Base Diagnostics ──────────────────────────────
phase_header(3, "Knowledge Base Diagnostics")

kb = lce.knowledge_base
kb_status = kb.get_status()
info(f"Nodes: {kb_status.get('total_nodes', '?')}")
info(f"Total facts: {kb_status.get('total_facts', '?')}")
info(f"Relations: {kb_status.get('total_relations', '?')}")
info(f"Initialized: {kb_status.get('initialized', '?')}")

# Sample a few subjects
try:
    nodes_list = list(kb.nodes.items())
    info(f"Sample subjects ({min(5, len(nodes_list))}):")
    for key, node in nodes_list[:5]:
        print(f"      {key}: {len(node.facts)} facts, def={node.definition[:60]}...")
except Exception as e:
    fail(f"KB node sampling failed: {e}")

# Test query
try:
    t0 = time.time()
    hits = kb.query("What is the speed of light?", top_k=5)
    dt = (time.time() - t0) * 1000
    ok(f"KB query returned {len(hits)} hits in {dt:.1f}ms")
    for key, node, score in hits[:3]:
        print(f"      {score:.4f}  {key}")
except Exception as e:
    fail(f"KB query failed: {e}")
    errors.append(("Phase 3: KB query", str(e)))

# ── Phase 4: Tokenizer ──────────────────────────────────────────────
phase_header(4, "Tokenizer (Layer 1)")

try:
    tokens = lce.tokenizer.tokenize("The mitochondria is the powerhouse of the cell")
    ok(f"Tokenized → {len(tokens)} tokens: {tokens[:10]}...")
    info(f"Vocab size: {lce.tokenizer.vocab_count}")
except Exception as e:
    fail(f"Tokenizer failed: {e}")
    errors.append(("Phase 4: Tokenizer", str(e)))

# ── Phase 5: Semantic Encoder ────────────────────────────────────────
phase_header(5, "Semantic Encoder (Layer 2)")

try:
    enc = lce.semantic_encoder
    vec = enc.encode("quantum entanglement")
    ok(f"Encoded → vector dim={len(vec)}, norm={sum(x*x for x in vec)**0.5:.4f}")
except Exception as e:
    fail(f"Semantic encoding failed: {e}")
    errors.append(("Phase 5: Semantic Encoder", str(e)))

# ── Phase 6: N-gram Matcher ─────────────────────────────────────────
phase_header(6, "N-gram Matcher (Layer 2b)")

try:
    ngram_hits = lce.ngram_matcher.match("cell membrane transport proteins", top_k=5)
    ok(f"N-gram matches: {len(ngram_hits)}")
    for key, score in ngram_hits[:3]:
        print(f"      {score:.4f}  {key}")
except Exception as e:
    fail(f"N-gram matching failed: {e}")
    errors.append(("Phase 6: N-gram Matcher", str(e)))

# ── Phase 7: Subject Detector ───────────────────────────────────────
phase_header(7, "Subject Detector (Layer 4b)")

try:
    det = lce.subject_detector
    test_q = "What is the half-life of carbon-14?"
    test_choices = ["5730 years", "1000 years", "10000 years", "100 years"]
    subj = det.detect(test_q, test_choices)
    ok(f"Detected subject: '{subj}' for '{test_q[:50]}'")
except Exception as e:
    fail(f"Subject detection failed: {e}")
    errors.append(("Phase 7: Subject Detector", str(e)))

# ── Phase 8: BM25 Ranker ────────────────────────────────────────────
phase_header(8, "BM25 Ranker (Layer 4)")

try:
    bm25 = lce.mcq_solver.bm25
    docs = ["The speed of light is 3e8 m/s", "Gravity pulls objects down",
            "Photons have no rest mass", "E=mc² relates energy and mass"]
    bm25.fit(docs)
    ranked = bm25.rank("What is the speed of light?", top_k=3)
    ok(f"BM25 ranked {len(ranked)} results")
    for idx, score in ranked:
        print(f"      [{idx}] score={score:.4f}  '{docs[idx][:50]}'")
except Exception as e:
    fail(f"BM25 ranking failed: {e}")
    errors.append(("Phase 8: BM25 Ranker", str(e)))

# ── Phase 9: Numerical Reasoner ─────────────────────────────────────
phase_header(9, "Numerical Reasoner (Layer 4c)")

try:
    nr = lce.numerical_reasoner
    test = nr.extract("The boiling point of water is 100°C or 212°F")
    ok(f"Extracted values: {test}")
except Exception as e:
    fail(f"Numerical reasoner failed: {e}")
    errors.append(("Phase 9: Numerical Reasoner", str(e)))

# ── Phase 10: Textual Entailment ────────────────────────────────────
phase_header(10, "Textual Entailment (Layer 12)")

try:
    ent = lce.entailment_engine
    r = ent.entail("All mammals are warm-blooded", "Dogs are warm-blooded")
    ok(f"Entailment: '{r['label']}' (conf={r['confidence']:.3f})")
    r2 = ent.entail("It is raining", "The ground is dry")
    ok(f"Contradiction test: '{r2['label']}' (conf={r2['confidence']:.3f})")
except Exception as e:
    fail(f"Entailment failed: {e}")
    errors.append(("Phase 10: Entailment", str(e)))

# ── Phase 11: Analogical Reasoning ──────────────────────────────────
phase_header(11, "Analogical Reasoner (Layer 13)")

try:
    ar = lce.analogical_reasoner
    result = ar.score_analogy("cat", "kitten", "dog", "puppy")
    ok(f"cat:kitten :: dog:puppy → score={result:.4f}")
except Exception as e:
    fail(f"Analogical reasoning failed: {e}")
    errors.append(("Phase 11: Analogy", str(e)))

# ── Phase 12: NER ───────────────────────────────────────────────────
phase_header(12, "Named Entity Recognition (Layer 15)")

try:
    ner = lce.ner
    entities = ner.extract_entity_types("Albert Einstein published his theory in Berlin in 1905")
    ok(f"Entities found: {entities}")
except Exception as e:
    fail(f"NER failed: {e}")
    errors.append(("Phase 12: NER", str(e)))

# ── Phase 13: Fuzzy Match ───────────────────────────────────────────
phase_header(13, "Levenshtein Fuzzy Matcher (Layer 16)")

try:
    fm = lce.fuzzy_matcher
    sim = fm.similarity("mitochondria", "mitocondria")
    ok(f"'mitochondria' vs 'mitocondria' → similarity={sim:.4f}")
except Exception as e:
    fail(f"Fuzzy matcher failed: {e}")
    errors.append(("Phase 13: Fuzzy Match", str(e)))

# ── Phase 14: LSA ───────────────────────────────────────────────────
phase_header(14, "Latent Semantic Analysis (Layer 17)")

try:
    lsa = lce.lsa
    info(f"LSA fitted: {lsa._fitted}")
    if lsa._fitted:
        hits = lsa.query_similarity("quantum tunnelling effect", top_k=3)
        ok(f"LSA query → {len(hits)} hits")
        for idx, sim in hits[:3]:
            print(f"      [{idx}] sim={sim:.4f}")
    else:
        fail("LSA not fitted — comprehend() will lack concept-level similarity")
        errors.append(("Phase 14: LSA", "Not fitted"))
except Exception as e:
    fail(f"LSA failed: {e}")
    errors.append(("Phase 14: LSA", str(e)))

# ── Phase 15: Lesk WSD ──────────────────────────────────────────────
phase_header(15, "Lesk Word Sense Disambiguation (Layer 18)")

try:
    lesk = lce.lesk
    results = lesk.disambiguate_all("She went to the bank to deposit money")
    ok(f"WSD results: {len(results)} senses")
    for r in results[:3]:
        if isinstance(r, dict):
            print(f"      '{r.get('word', '?')}' → {r.get('sense', '?')}")
        else:
            print(f"      {r}")
except Exception as e:
    fail(f"Lesk WSD failed: {e}")
    errors.append(("Phase 15: Lesk WSD", str(e)))

# ── Phase 16: TextRank Summarizer ────────────────────────────────────
phase_header(16, "TextRank Summarizer (Layer 14)")

try:
    tr = lce.textrank
    long_text = (
        "The mitochondria is the powerhouse of the cell. It produces ATP through "
        "oxidative phosphorylation. The inner membrane contains electron transport "
        "chains. Cellular respiration converts glucose to energy. The Krebs cycle "
        "occurs in the mitochondrial matrix. ATP synthase uses the proton gradient "
        "to generate ATP. Mitochondria have their own DNA."
    )
    result = tr.summarize(long_text, num_sentences=2)
    ok(f"Summary: '{result['summary'][:80]}...'")
    info(f"Compression: {result.get('compression_ratio', '?')}")
except Exception as e:
    fail(f"TextRank failed: {e}")
    errors.append(("Phase 16: TextRank", str(e)))

# ── Phase 17: Coreference Resolution ────────────────────────────────
phase_header(17, "Coreference Resolution (Layer 19)")

try:
    coref = lce.coref_resolver
    result = coref.resolve("Albert Einstein was a physicist. He developed general relativity.")
    ok(f"Coreference result: {result}")
except Exception as e:
    fail(f"Coreference failed: {e}")
    errors.append(("Phase 17: Coreference", str(e)))

# ── Phase 18: Sentiment Analysis ─────────────────────────────────────
phase_header(18, "Sentiment Analysis (Layer 20)")

try:
    sent = lce.sentiment_analyzer
    result = sent.analyze("The experiment was a brilliant success with remarkable results")
    ok(f"Sentiment: {result}")
except Exception as e:
    fail(f"Sentiment failed: {e}")
    errors.append(("Phase 18: Sentiment", str(e)))

# ── Phase 19: Semantic Frame Analyzer ────────────────────────────────
phase_header(19, "Semantic Frame Analyzer (Layer 21)")

try:
    frames = lce.frame_analyzer
    result = frames.analyze("What causes the greenhouse effect?")
    ok(f"Frame analysis: {result}")
except Exception as e:
    fail(f"Frame analysis failed: {e}")
    errors.append(("Phase 19: Frames", str(e)))

# ── Phase 20: Taxonomy Classifier ────────────────────────────────────
phase_header(20, "Taxonomy Classifier (Layer 22)")

try:
    tax = lce.taxonomy
    result = tax.classify("photosynthesis")
    ok(f"Taxonomy: {result}")
except Exception as e:
    fail(f"Taxonomy failed: {e}")
    errors.append(("Phase 20: Taxonomy", str(e)))

# ── Phase 21: Causal Chain Reasoner ──────────────────────────────────
phase_header(21, "Causal Chain Reasoner (Layer 23)")

try:
    cc = lce.causal_chain
    result = cc.infer("Global warming leads to ice caps melting which causes sea level rise")
    ok(f"Causal chain: {result}")
except Exception as e:
    fail(f"Causal chain failed: {e}")
    errors.append(("Phase 21: Causal Chain", str(e)))

# ── Phase 22: Pragmatic Inference ────────────────────────────────────
phase_header(22, "Pragmatic Inference (Layer 24)")

try:
    prag = lce.pragmatics
    result = prag.analyze("Can you pass the salt?")
    ok(f"Pragmatic: {result}")
except Exception as e:
    fail(f"Pragmatic inference failed: {e}")
    errors.append(("Phase 22: Pragmatics", str(e)))

# ── Phase 23: Commonsense Knowledge ─────────────────────────────────
phase_header(23, "ConceptNet Commonsense Linker (Layer 25)")

try:
    cs = lce.commonsense
    result = cs.relate("fire", "heat")
    ok(f"Commonsense: fire→heat = {result}")
except Exception as e:
    fail(f"Commonsense failed: {e}")
    errors.append(("Phase 23: Commonsense", str(e)))

# ── Phase 24: DeepNLU Integration ────────────────────────────────────
phase_header(24, "DeepNLU Integration (Layers 9-11)")

if lce._deep_nlu is not None:
    try:
        nlu = lce._deep_nlu
        result = nlu.analyze("The electron was discovered by J.J. Thomson in 1897")
        ok(f"DeepNLU result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        if hasattr(nlu, 'temporal'):
            temp = nlu.temporal.analyze("World War II ended in 1945 after the atomic bombs")
            ok(f"Temporal: {temp}")
        if hasattr(nlu, 'causal'):
            caus = nlu.causal.analyze("Heating water causes it to boil")
            ok(f"Causal: {caus}")
    except Exception as e:
        fail(f"DeepNLU analysis failed: {e}")
        errors.append(("Phase 24: DeepNLU", str(e)))
else:
    info("DeepNLU not connected (optional)")

# ── Phase 25: comprehend() Full Pipeline ─────────────────────────────
phase_header(25, "comprehend() — Full Pipeline Test")

test_texts = [
    "Quantum entanglement allows two particles to be correlated regardless of distance",
    "The French Revolution began in 1789 and led to major political changes in Europe",
    "Mitosis is a type of cell division that produces two genetically identical daughter cells",
]

for text in test_texts:
    try:
        t0 = time.time()
        result = lce.comprehend(text)
        dt = (time.time() - t0) * 1000
        ok(f"comprehend() in {dt:.0f}ms: depth={result.get('comprehension_depth', '?'):.3f}")
        info(f"  concepts={result.get('key_concepts', [])[:5]}")
        info(f"  knowledge_hits={len(result.get('knowledge_relevance', []))}")
        info(f"  entities={result.get('entities', {})}")
        if result.get('entailment_signals'):
            info(f"  entailment={result.get('entailment_signals')}")
        if result.get('temporal'):
            info(f"  temporal={result.get('temporal')}")
    except Exception as e:
        fail(f"comprehend('{text[:40]}...') failed: {e}")
        traceback.print_exc()
        errors.append(("Phase 25: comprehend()", str(e)))

# ── Phase 26: MCQ Solver — Full Pipeline ─────────────────────────────
phase_header(26, "MCQ Solver — Full Pipeline")

mcq_tests = [
    {
        "question": "What is the primary function of mitochondria in a cell?",
        "choices": ["Protein synthesis", "Energy production", "Cell division", "DNA replication"],
        "expected": 1,  # B: Energy production
        "subject": "college_biology",
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
        "expected": 1,  # B: Mars
        "subject": "astronomy",
    },
    {
        "question": "What is the speed of light in vacuum?",
        "choices": ["3 × 10^6 m/s", "3 × 10^8 m/s", "3 × 10^10 m/s", "3 × 10^4 m/s"],
        "expected": 1,  # B: 3 × 10^8
        "subject": "college_physics",
    },
    {
        "question": "In economics, what does GDP stand for?",
        "choices": ["Gross Domestic Product", "General Development Plan",
                     "Global Distribution Protocol", "Government Debt Program"],
        "expected": 0,  # A: Gross Domestic Product
        "subject": "high_school_macroeconomics",
    },
    {
        "question": "What is the chemical formula for water?",
        "choices": ["CO2", "H2O", "NaCl", "O2"],
        "expected": 1,  # B: H2O
        "subject": "high_school_chemistry",
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
        "expected": 1,  # B: Shakespeare
        "subject": None,
    },
    {
        "question": "What is the main gas in Earth's atmosphere?",
        "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"],
        "expected": 2,  # C: Nitrogen
        "subject": "high_school_geography",
    },
    {
        "question": "What is Newton's second law of motion?",
        "choices": ["F = ma", "E = mc²", "PV = nRT", "V = IR"],
        "expected": 0,  # A: F = ma
        "subject": "college_physics",
    },
]

correct = 0
total = len(mcq_tests)
for i, test in enumerate(mcq_tests):
    try:
        t0 = time.time()
        result = lce.mcq_solver.solve(
            test["question"], test["choices"], subject=test["subject"]
        )
        dt = (time.time() - t0) * 1000

        selected = result.get("selected_index", -1)
        confidence = result.get("confidence", 0)
        is_correct = selected == test["expected"]
        if is_correct:
            correct += 1

        sym = "✓" if is_correct else "✗"
        expected_label = chr(65 + test["expected"])
        selected_label = chr(65 + selected) if 0 <= selected < len(test["choices"]) else "?"

        print(f"    {sym} Q{i+1}: selected={selected_label} expected={expected_label} "
              f"conf={confidence:.3f} ({dt:.0f}ms)")

        if not is_correct:
            info(f"  Question: {test['question'][:60]}")
            info(f"  Expected: [{expected_label}] {test['choices'][test['expected']]}")
            info(f"  Selected: [{selected_label}] {test['choices'][selected] if 0 <= selected < len(test['choices']) else '?'}")
            # Show score breakdown
            scores = result.get("choice_scores", result.get("scores", []))
            if scores:
                info(f"  Scores: {[f'{s:.4f}' for s in scores]}")
            reasoning = result.get("reasoning", result.get("reasoning_chain", ""))
            if reasoning:
                chain = reasoning if isinstance(reasoning, str) else str(reasoning)[:200]
                info(f"  Reasoning: {chain[:150]}")
    except Exception as e:
        fail(f"MCQ Q{i+1} failed: {e}")
        traceback.print_exc()
        errors.append((f"Phase 26: MCQ Q{i+1}", str(e)))

accuracy = correct / total * 100 if total > 0 else 0
info(f"MCQ accuracy: {correct}/{total} = {accuracy:.1f}%")

# MCQ solver stats
mcq_status = lce.mcq_solver.get_status()
info(f"MCQ stats: answered={mcq_status.get('questions_answered', 0)}, "
     f"subject_detections={lce.mcq_solver._subject_detections}, "
     f"numerical_assists={lce.mcq_solver._numerical_assists}, "
     f"cross_verifications={lce.mcq_solver._cross_verifications}")
info(f"  ngram_boosts={lce.mcq_solver._ngram_boosts}, "
     f"logic_assists={lce.mcq_solver._logic_assists}, "
     f"nlu_assists={lce.mcq_solver._nlu_assists}, "
     f"early_exits={lce.mcq_solver._early_exits}")

# ── Phase 27: Three-Engine Scoring ───────────────────────────────────
phase_header(27, "Three-Engine Comprehension Score")

try:
    score = lce.three_engine_comprehension_score()
    ok(f"Three-engine score: {score:.6f}")
except Exception as e:
    fail(f"Three-engine scoring failed: {e}")
    errors.append(("Phase 27: Three-Engine Score", str(e)))

try:
    te_status = lce.three_engine_status()
    info(f"Three-engine engines: {te_status.get('engines', {})}")
    info(f"Three-engine scores: {te_status.get('scores', {})}")
except Exception as e:
    fail(f"Three-engine status failed: {e}")

# ── Summary ──────────────────────────────────────────────────────────
dt_total = (time.time() - t_total) * 1000

banner("RESULTS")
print(f"  Total time:  {dt_total:.0f}ms ({dt_total/1000:.1f}s)")
print(f"  Layers:      {active_layers}/{total_layers} active")
print(f"  Engines:     {connected}/{len(engines)} connected")
print(f"  MCQ:         {correct}/{total} correct ({accuracy:.1f}%)")
print(f"  Errors:      {len(errors)}")

if errors:
    print(f"\n  ── Errors ──")
    for phase, msg in errors:
        print(f"    ✗ {phase}: {msg}")

print(f"\n{'═' * 70}")
print(f"  DONE")
print(f"{'═' * 70}")
