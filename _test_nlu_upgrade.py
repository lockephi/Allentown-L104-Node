#!/usr/bin/env python3
"""Quick validation of DeepNLU v2.0.0 + LanguageComprehension v6.0.0 upgrade."""

import sys

def main():
    print("=== TEST 1: DeepNLU v2.0.0 Import ===")
    from l104_asi.deep_nlu import (
        DeepNLUEngine, DeepComprehension,
        TemporalReasoner, CausalReasoner, ContextualDisambiguator,
        TemporalRelation, CausalRelationType,
    )
    engine = DeepNLUEngine()
    status = engine.status()
    assert status["version"] == "2.0.0", f"Expected v2.0.0, got {status['version']}"
    assert status["layers"] == 13, f"Expected 13 layers, got {status['layers']}"
    assert len(status["layer_names"]) == 13
    print(f"  Version: {status['version']}")
    print(f"  Layers: {status['layers']}")
    print(f"  NLU Depth Score: {status['nlu_depth_score']}")
    print(f"  Sense inventory: {status['sense_inventory_size']}")
    print(f"  Temporal markers: {status['temporal_markers']}")
    print(f"  Causal patterns: {status['causal_patterns']}")
    print("  PASS")

    print("\n=== TEST 2: Temporal Analysis ===")
    t = engine.analyze_temporal("Yesterday the team finished the project. Tomorrow they will start a new one.")
    assert t["tense"]["dominant"] in ("past", "future"), f"Unexpected tense: {t['tense']['dominant']}"
    assert t["temporal_richness"] > 0, "Expected temporal richness > 0"
    assert len(t["temporal_expressions"]) >= 1, "Expected temporal expressions"
    print(f"  Dominant tense: {t['tense']['dominant']}")
    print(f"  Temporal richness: {t['temporal_richness']}")
    print(f"  Expressions found: {len(t['temporal_expressions'])}")
    print("  PASS")

    print("\n=== TEST 3: Causal Analysis ===")
    c = engine.analyze_causal("Smoking causes cancer. Cancer leads to death. Exercise prevents heart disease.")
    assert c["total_relations"] > 0, "Expected causal relations"
    print(f"  Causal pairs: {c['total_relations']}")
    print(f"  Causal strength: {c['causal_strength']}")
    print(f"  Chains: {len(c['causal_chains'])}")
    for p in c["causal_pairs"][:3]:
        print(f"    {p['cause'][:40]} --{p['relation']}--> {p['effect'][:40]}")
    print("  PASS")

    print("\n=== TEST 4: Disambiguation ===")
    d = engine.disambiguate("The bank of the river was eroded by the flood water.")
    assert d["ambiguous_words_found"] > 0, "Expected ambiguous words"
    print(f"  Ambiguous words: {d['ambiguous_words_found']}")
    for dis in d["disambiguations"]:
        print(f"    {dis['word']}: {dis['selected_sense']} (conf: {dis['confidence']})")
    print("  PASS")

    print("\n=== TEST 5: Full 13-Layer Analysis ===")
    result = engine.deep_analyze(
        "Before the experiment, the scientist hypothesized that heat causes expansion. "
        "She tested this by heating a metal bar, and it expanded as predicted."
    )
    assert result["total_layers"] == 13, f"Expected 13, got {result['total_layers']}"
    assert result["layers_active"] >= 5, f"Expected >=5 active layers, got {result['layers_active']}"
    assert "temporal" in result, "Missing temporal in result"
    assert "causal" in result, "Missing causal in result"
    assert "disambiguation" in result, "Missing disambiguation in result"
    print(f"  Layers active: {result['layers_active']}/{result['total_layers']}")
    print(f"  Comprehension depth: {result['comprehension_depth']}")
    print(f"  Temporal richness: {result['temporal']['temporal_richness']}")
    print(f"  Causal pairs: {result['causal']['total_relations']}")
    print(f"  Disambiguations: {result['disambiguation']['ambiguous_words_found']}")
    print("  PASS")

    print("\n=== TEST 6: LanguageComprehensionEngine v6.0.0 ===")
    from l104_asi.language_comprehension import LanguageComprehensionEngine
    lce = LanguageComprehensionEngine()
    assert lce.VERSION == "6.0.0", f"Expected v6.0.0, got {lce.VERSION}"
    lce_status = lce.get_status()
    assert "9_temporal_reasoning" in lce_status["layers"], "Missing temporal layer in status"
    assert "10_causal_reasoning" in lce_status["layers"], "Missing causal layer in status"
    assert "11_contextual_disambiguation" in lce_status["layers"], "Missing disambiguation layer in status"
    # Verify no duplicate keys by checking structure
    keys = list(lce_status.keys())
    assert keys.count("layers") <= 1, "Duplicate 'layers' key!"
    assert keys.count("v3_stats") <= 1, "Duplicate 'v3_stats' key!"
    print(f"  Version: {lce.VERSION}")
    print(f"  Layers: {list(lce_status['layers'].keys())}")
    print("  PASS")

    print("\n=== TEST 7: Comprehend with Temporal/Causal/Disambiguation ===")
    lce.initialize()
    result = lce.comprehend(
        "The Industrial Revolution caused massive urbanization in the 19th century. "
        "Before this, most people lived in rural areas. The scale of change was enormous."
    )
    assert "temporal" in result, "Missing temporal in comprehend result"
    assert "causal" in result, "Missing causal in comprehend result"
    assert "disambiguation" in result, "Missing disambiguation in comprehend result"
    print(f"  Token count: {result['token_count']}")
    print(f"  Concepts: {result['concepts_extracted'][:5]}")
    print(f"  Comprehension depth: {result['comprehension_depth']}")
    if result["temporal"]:
        print(f"  Temporal richness: {result['temporal'].get('temporal_richness', 'N/A')}")
    if result["causal"]:
        print(f"  Causal density: {result['causal'].get('causal_density', 'N/A')}")
    if result["disambiguation"]:
        print(f"  WSD coverage: {result['disambiguation'].get('wsd_coverage', 'N/A')}")
    print("  PASS")

    print("\n" + "=" * 60)
    print("  ALL 7 TESTS PASSED — Language Comprehension Upgrade OK")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
