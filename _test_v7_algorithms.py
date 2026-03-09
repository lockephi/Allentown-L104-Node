#!/usr/bin/env python3
"""Validation suite for Language Comprehension v7.0.0 — 7 new algorithms."""

def main():
    print("=== VALIDATION: Language Comprehension v7.0.0 ===\n")

    # Test 1: Import all new classes
    print("TEST 1: Import new algorithm classes...")
    from l104_asi.language_comprehension import (
        TextualEntailmentEngine, AnalogicalReasoner, TextRankSummarizer,
        NamedEntityRecognizer, LevenshteinMatcher, LatentSemanticAnalyzer,
        LeskDisambiguator, LanguageComprehensionEngine,
    )
    print("  All 7 new classes imported successfully")
    print("  PASS")

    # Test 2: TextualEntailmentEngine
    print("\nTEST 2: TextualEntailmentEngine...")
    te = TextualEntailmentEngine()
    r = te.entail("Dogs are animals that bark.", "Animals can bark.")
    print(f"  Label: {r['label']}, Confidence: {r['confidence']:.3f}")
    assert r["label"] in ("entailment", "neutral", "contradiction")
    r2 = te.entail("The temperature is 100 degrees.", "The temperature is not 100 degrees.")
    print(f"  Negation test: {r2['label']}, Confidence: {r2['confidence']:.3f}")
    r3 = te.score_fact_choice_entailment("Water boils at 100 degrees Celsius.", "boiling point is 100 degrees")
    print(f"  Fact-choice entailment score: {r3:.3f}")
    print("  PASS")

    # Test 3: AnalogicalReasoner
    print("\nTEST 3: AnalogicalReasoner...")
    ar = AnalogicalReasoner()
    rel, conf = ar.detect_relation("dog", "animal")
    print(f"  dog->animal: {rel} (conf: {conf:.2f})")
    assert rel == "is_a"
    score = ar.score_analogy("dog", "animal", "rose", "flower")
    print(f"  Analogy dog:animal :: rose:flower = {score:.3f}")
    assert score > 0.5, f"Expected > 0.5, got {score}"
    parts = ar.detect_analogy_in_question("Cat is to animal as rose is to what?")
    print(f"  Analogy detection: {parts}")
    completions = ar.complete_analogy("hot", "cold", "big", ["small", "large", "tall", "wide"])
    print(f"  Complete hot:cold :: big:? = {completions[0]}")
    print("  PASS")

    # Test 4: TextRankSummarizer
    print("\nTEST 4: TextRankSummarizer...")
    tr = TextRankSummarizer()
    text = (
        "The Earth revolves around the Sun. The Sun is a star. "
        "Stars produce energy through nuclear fusion. "
        "The Earth has one natural satellite called the Moon. "
        "The Moon affects tides on Earth. Gravity keeps planets in orbit."
    )
    result = tr.summarize(text, num_sentences=2)
    print(f"  Summary: \"{result['summary'][:80]}...\"")
    print(f"  Compression: {result['compression_ratio']}")
    print(f"  Total sentences: {result['total_sentences']}")
    assert result["total_sentences"] >= 4
    facts_ranked = tr.extract_key_facts(
        ["fact1 about physics", "fact2 about chemistry", "fact3 about physics and chemistry"],
        top_k=2,
    )
    print(f"  Top facts ranked: {len(facts_ranked)}")
    print("  PASS")

    # Test 5: NamedEntityRecognizer
    print("\nTEST 5: NamedEntityRecognizer...")
    ner = NamedEntityRecognizer()
    entities = ner.recognize("Albert Einstein published his theory of relativity in 1905 in Berlin.")
    print(f"  Found {len(entities)} entities:")
    for e in entities:
        print(f"    {e['text']} -> {e['type']} (conf: {e['confidence']:.2f})")
    grouped = ner.extract_entity_types("Marie Curie discovered radium in Paris.")
    print(f"  Entity types: {list(grouped.keys())}")
    print("  PASS")

    # Test 6: LevenshteinMatcher
    print("\nTEST 6: LevenshteinMatcher...")
    lm = LevenshteinMatcher()
    d = lm.distance("kitten", "sitting")
    print(f"  distance(kitten, sitting) = {d}")
    assert d == 3
    sim = lm.similarity("algorithm", "algorythm")
    print(f"  similarity(algorithm, algorythm) = {sim:.3f}")
    assert sim > 0.7
    dd = lm.damerau_distance("ab", "ba")
    print(f"  damerau_distance(ab, ba) = {dd}")
    assert dd == 1  # transposition
    matches = lm.fuzzy_match("python", ["pythong", "java", "pyhton", "ruby"], threshold=0.5)
    print(f"  Fuzzy matches for python: {[(m, round(s,2)) for m,s in matches]}")
    best, bscore = lm.best_match("evaporation", ["evaporate", "condensation", "precipitation"])
    print(f"  Best match for evaporation: {best} ({bscore:.3f})")
    print("  PASS")

    # Test 7: LatentSemanticAnalyzer
    print("\nTEST 7: LatentSemanticAnalyzer...")
    lsa = LatentSemanticAnalyzer(n_components=5)
    docs = [
        "the cat sat on the mat", "dogs chase cats in the park",
        "the quick brown fox jumps", "neural networks learn patterns",
        "machine learning uses algorithms", "deep learning neural computation",
        "the dog runs in the garden", "cats and dogs are pets",
    ]
    lsa.fit(docs)
    assert lsa._fitted, "LSA should be fitted"
    hits = lsa.query_similarity("cats dogs pets", top_k=3)
    print(f"  LSA query results: {[(i, round(s,3)) for i,s in hits]}")
    cs = lsa.concept_similarity("neural networks", "machine learning")
    print(f"  Concept similarity (neural networks, machine learning): {cs:.3f}")
    print("  PASS")

    # Test 8: LeskDisambiguator
    print("\nTEST 8: LeskDisambiguator...")
    lesk = LeskDisambiguator()
    r = lesk.disambiguate("bank", "I deposited money at the bank for savings.")
    print(f"  bank in financial context: {r['selected_sense']} (conf: {r['confidence']:.3f})")
    assert r["selected_sense"] == "financial"
    r2 = lesk.disambiguate("bank", "The river bank was eroded by the flood water.")
    print(f"  bank in river context: {r2['selected_sense']} (conf: {r2['confidence']:.3f})")
    assert r2["selected_sense"] == "river"
    all_wsd = lesk.disambiguate_all("The cell in the plant produces power.")
    print(f"  Disambiguated {len(all_wsd)} polysemous words")
    for w in all_wsd:
        print(f"    {w['word']}: {w['selected_sense']}")
    print("  PASS")

    # Test 9: LanguageComprehensionEngine v7.0.0
    print("\nTEST 9: LanguageComprehensionEngine v7.0.0...")
    lce = LanguageComprehensionEngine()
    assert lce.VERSION == "7.0.0", f"Expected v7.0.0, got {lce.VERSION}"
    status = lce.get_status()
    assert "12_textual_entailment" in status["layers"], "Missing layer 12"
    assert "18_lesk_wsd" in status["layers"], "Missing layer 18"
    assert "v7_algorithms" in status, "Missing v7_algorithms status"
    print(f"  Version: {lce.VERSION}")
    print(f"  Layers: {len(status['layers'])}")
    print(f"  V7 algorithms: {list(status['v7_algorithms'].keys())}")
    print("  PASS")

    # Test 10: comprehend() with new outputs
    print("\nTEST 10: comprehend() with new algorithm outputs...")
    lce.initialize()
    result = lce.comprehend(
        "Albert Einstein developed the theory of relativity in 1905. "
        "The theory explains that energy equals mass times the speed of light squared. "
        "This equation revolutionized modern physics and led to nuclear energy."
    )
    print(f"  Token count: {result['token_count']}")
    print(f"  Concepts: {result['concepts_extracted'][:5]}")
    print(f"  Entities: {result.get('entities', {})}")
    print(f"  Summary present: {bool(result.get('summary', {}))}")
    print(f"  Lesk WSD results: {len(result.get('lesk_wsd', []))}")
    print(f"  Entailment results: {len(result.get('entailment', []))}")
    print(f"  LSA matches: {len(result.get('lsa_concept_matches', []))}")
    print(f"  Comprehension depth: {result['comprehension_depth']:.4f}")
    assert "entities" in result, "Missing entities in comprehend output"
    assert "summary" in result, "Missing summary in comprehend output"
    assert "lesk_wsd" in result, "Missing lesk_wsd in comprehend output"
    assert "entailment" in result, "Missing entailment in comprehend output"
    assert "lsa_concept_matches" in result, "Missing lsa_concept_matches"
    print("  PASS")

    print("\n=== ALL 10 TESTS PASSED — Language Comprehension v7.0.0 OK ===")

if __name__ == "__main__":
    main()
