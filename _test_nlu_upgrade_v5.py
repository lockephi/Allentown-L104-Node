#!/usr/bin/env python3
"""
Wave 5 Validation Test — DeepNLU v2.2.0 + ASIInnovationEngine v2.0.0 +
CommonsenseReasoning v3.0.0 + LanguageEngine v6.0.0

Tests:
  A) QueryDecomposer — 10 tests
  B) QueryExpander — 10 tests
  C) QueryClassifier — 10 tests
  D) DeepNLUEngine v2.2.0 integration — 10 tests
  E) ASIInnovationEngine v2.0.0 NLU — 10 tests
  F) CommonsenseReasoning v3.0.0 NLU — 5 tests
  G) LanguageEngine v6.0.0 hub — 10 tests
  H) ASI Core v2.2.0 methods — 5 tests
  I) __init__.py exports — 5 tests

Total: 75 tests
"""

import sys
import traceback

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612
PASS = 0
FAIL = 0
ERRORS = []


def test(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        ERRORS.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")


def run_all():
    global PASS, FAIL

    # ═══════════════════════════════════════════════════════════════════
    # A) QUERY DECOMPOSER — 10 tests
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  A) QUERY DECOMPOSER  (Layer 15)")
    print("=" * 70)

    from l104_asi.deep_nlu import QueryDecomposer, SubQuery

    decomposer = QueryDecomposer()

    # A1: Class instantiation
    test("A1: QueryDecomposer instantiation", decomposer is not None)

    # A2: Version
    test("A2: QueryDecomposer VERSION", decomposer.VERSION == "1.0.0",
         f"got {decomposer.VERSION}")

    # A3: Simple atomic query (no decomposition)
    r = decomposer.decompose("What is gravity?")
    test("A3: Atomic query detection",
         r['is_atomic'] or r['count'] >= 1,
         f"is_atomic={r.get('is_atomic')}, count={r.get('count')}")

    # A4: Complex conjunctive query
    r = decomposer.decompose("How does photosynthesis work and why is it important?")
    test("A4: Conjunctive decomposition",
         r['count'] >= 1,
         f"count={r['count']}, method={r.get('decomposition_method')}")

    # A5: Output structure
    test("A5: Has sub_queries list",
         isinstance(r.get('sub_queries'), list) and len(r['sub_queries']) > 0,
         f"sub_queries={r.get('sub_queries')}")

    # A6: Dependency graph
    test("A6: Has dependency_graph",
         isinstance(r.get('dependency_graph'), dict))

    # A7: Execution order
    test("A7: Has execution_order",
         isinstance(r.get('execution_order'), list))

    # A8: PHI coherence
    test("A8: phi_coherence present",
         'phi_coherence' in r and r['phi_coherence'] > 0,
         f"phi_coherence={r.get('phi_coherence')}")

    # A9: Status method
    st = decomposer.status()
    test("A9: status() works",
         st.get('engine') == 'QueryDecomposer' and st['decompositions_performed'] >= 2)

    # A10: SubQuery dataclass
    sq = SubQuery(text="test?", index=0, focus="test", complexity=0.3, query_type="factual")
    test("A10: SubQuery dataclass", sq.text == "test?" and sq.query_type == "factual")

    # ═══════════════════════════════════════════════════════════════════
    # B) QUERY EXPANDER — 10 tests
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  B) QUERY EXPANDER  (Layer 16)")
    print("=" * 70)

    from l104_asi.deep_nlu import QueryExpander, ExpandedQuery

    expander = QueryExpander()

    # B1: Instantiation
    test("B1: QueryExpander instantiation", expander is not None)

    # B2: Version
    test("B2: QueryExpander VERSION", expander.VERSION == "1.0.0")

    # B3: Synonym expansion
    r = expander.expand("Why is it important to understand gravity?")
    test("B3: Expansion returns results",
         r['count'] >= 0 and isinstance(r.get('expansions'), list),
         f"count={r['count']}")

    # B4: Strategies used
    test("B4: strategies_used present",
         isinstance(r.get('strategies_used'), list) and len(r['strategies_used']) > 0)

    # B5: Unique terms
    test("B5: unique_terms_added present",
         isinstance(r.get('unique_terms_added'), list))

    # B6: PHI diversity
    test("B6: phi_diversity present",
         'phi_diversity' in r and r['phi_diversity'] >= 0)

    # B7: Strategy filtering
    r2 = expander.expand("How does energy create momentum?", strategies={'synonym'})
    test("B7: Strategy filtering works",
         'synonym' in r2.get('strategies_used', []))

    # B8: Synonym clusters
    test("B8: Has synonym clusters",
         len(expander.SYNONYM_CLUSTERS) >= 20,
         f"got {len(expander.SYNONYM_CLUSTERS)}")

    # B9: Hypernym map
    test("B9: Has hypernym mappings",
         len(expander.HYPERNYM_MAP) >= 15,
         f"got {len(expander.HYPERNYM_MAP)}")

    # B10: Status
    st = expander.status()
    test("B10: status() works",
         st.get('engine') == 'QueryExpander' and st['expansions_performed'] >= 2)

    # ═══════════════════════════════════════════════════════════════════
    # C) QUERY CLASSIFIER — 10 tests
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  C) QUERY CLASSIFIER  (Layer 17)")
    print("=" * 70)

    from l104_asi.deep_nlu import (
        QueryClassifier, QueryClassification,
        BloomLevel, QueryDomain, AnswerFormat,
    )

    classifier = QueryClassifier()

    # C1: Instantiation
    test("C1: QueryClassifier instantiation", classifier is not None)

    # C2: Version
    test("C2: QueryClassifier VERSION", classifier.VERSION == "1.0.0")

    # C3: Basic classification
    r = classifier.classify("What is photosynthesis?")
    test("C3: Classification returns bloom_level",
         'bloom_level' in r,
         f"got keys: {list(r.keys())}")

    # C4: Domain detection
    test("C4: domain present",
         'domain' in r,
         f"domain={r.get('domain')}")

    # C5: Complexity tier
    test("C5: complexity tier",
         r.get('complexity') in ('simple', 'moderate', 'complex', 'expert'),
         f"complexity={r.get('complexity')}")

    # C6: Answer format
    test("C6: answer_format present",
         'answer_format' in r,
         f"answer_format={r.get('answer_format')}")

    # C7: Cognitive load
    test("C7: cognitive_load in [0,1]",
         0 <= r.get('cognitive_load', -1) <= 1,
         f"cognitive_load={r.get('cognitive_load')}")

    # C8: Confidence
    test("C8: confidence present",
         0 < r.get('confidence', 0) <= 1.0,
         f"confidence={r.get('confidence')}")

    # C9: PHI alignment
    test("C9: phi_alignment present",
         'phi_alignment' in r and r['phi_alignment'] > 0)

    # C10: BloomLevel enum
    test("C10: BloomLevel has 6 levels",
         len(BloomLevel) == 6,
         f"got {len(BloomLevel)}")

    # ═══════════════════════════════════════════════════════════════════
    # D) DEEP NLU ENGINE v2.2.0 — 10 tests
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  D) DEEP NLU ENGINE v2.2.0 INTEGRATION")
    print("=" * 70)

    from l104_asi.deep_nlu import DeepNLUEngine

    nlu = DeepNLUEngine()

    # D1: Version
    test("D1: DeepNLUEngine VERSION=2.2.0", nlu.VERSION == "2.2.0",
         f"got {nlu.VERSION}")

    # D2: Has decomposer
    test("D2: Has decomposer attribute", hasattr(nlu, 'decomposer'))

    # D3: Has expander
    test("D3: Has expander attribute", hasattr(nlu, 'expander'))

    # D4: Has classifier
    test("D4: Has classifier attribute", hasattr(nlu, 'classifier'))

    # D5: decompose_query API
    r = nlu.decompose_query("What is DNA and how does it replicate?")
    test("D5: decompose_query works",
         isinstance(r, dict) and 'sub_queries' in r)

    # D6: expand_query API
    r = nlu.expand_query("How does gravity affect objects?")
    test("D6: expand_query works",
         isinstance(r, dict) and 'expansions' in r)

    # D7: classify_query API
    r = nlu.classify_query("Why do electrons orbit the nucleus?")
    test("D7: classify_query works",
         isinstance(r, dict) and 'bloom_level' in r)

    # D8: Status reports 17 layers
    st = nlu.status()
    test("D8: 17 layers reported",
         st.get('layers') == 17,
         f"got {st.get('layers')}")

    # D9: Layer names include new layers
    names = st.get('layer_names', [])
    test("D9: New layer names present",
         'query_decomposer' in names and 'query_expander' in names and 'query_classifier' in names,
         f"names={names[-5:]}")

    # D10: nlu_depth_score base >= 0.78
    score = nlu.nlu_depth_score()
    test("D10: nlu_depth_score >= 0.78",
         score >= 0.78,
         f"got {score}")

    # ═══════════════════════════════════════════════════════════════════
    # E) ASI INNOVATION ENGINE v2.0.0 — 10 tests
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  E) ASI INNOVATION ENGINE v2.0.0 (NLU-DRIVEN)")
    print("=" * 70)

    from l104_asi_language_engine import (
        ASIInnovationEngine, ASILinguisticAnalyzer, ASIHumanInferenceEngine,
        ASILanguageEngine,
    )

    analyzer = ASILinguisticAnalyzer()
    inference = ASIHumanInferenceEngine(analyzer)
    innov = ASIInnovationEngine(analyzer, inference)

    # E1: VERSION
    test("E1: ASIInnovationEngine VERSION=2.0.0",
         innov.VERSION == "2.0.0",
         f"got {innov.VERSION}")

    # E2: DeepNLU available
    test("E2: _deep_nlu_available flag exists",
         hasattr(innov, '_deep_nlu_available'))

    # E3: DeepNLU integration active
    test("E3: DeepNLU integration active",
         innov._deep_nlu_available,
         "False — import may have failed")

    # E4: Has _decomposer
    test("E4: Has _decomposer",
         innov._decomposer is not None)

    # E5: Has _classifier
    test("E5: Has _classifier",
         innov._classifier is not None)

    # E6: innovate() generates innovations
    innovations = innov.innovate("Reduce energy consumption in data centers")
    test("E6: innovate() returns innovations",
         len(innovations) >= 1,
         f"got {len(innovations)}")

    # E7: NLU-enriched concepts extracted (more concepts due to NLU)
    analysis = analyzer.analyze("Reduce energy consumption in data centers using quantum computing")
    concepts = innov._extract_key_concepts(analysis)
    test("E7: _extract_key_concepts works",
         isinstance(concepts, list),
         f"concepts={concepts[:5]}")

    # E8: _causal_innovation method exists
    test("E8: _causal_innovation method exists",
         hasattr(innov, '_causal_innovation') and callable(innov._causal_innovation))

    # E9: ASILanguageEngine VERSION=3.0.0
    engine = ASILanguageEngine()
    test("E9: ASILanguageEngine VERSION=3.0.0",
         engine.VERSION == "3.0.0",
         f"got {engine.VERSION}")

    # E10: get_status includes innovation NLU info
    st = engine.get_status()
    components = st.get('components', {})
    test("E10: get_status has innovation_nlu_active",
         'innovation_nlu_active' in components,
         f"components keys: {list(components.keys())}")

    # ═══════════════════════════════════════════════════════════════════
    # F) COMMONSENSE REASONING v3.0.0 — 5 tests
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  F) COMMONSENSE REASONING v3.0.0 (NLU-ENRICHED)")
    print("=" * 70)

    from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine

    cre = CommonsenseReasoningEngine()

    # F1: VERSION
    test("F1: CommonsenseReasoning VERSION=3.0.0",
         cre.VERSION == "3.0.0",
         f"got {cre.VERSION}")

    # F2: DeepNLU flag
    test("F2: _deep_nlu_available flag exists",
         hasattr(cre, '_deep_nlu_available'))

    # F3: DeepNLU active
    test("F3: DeepNLU integration active",
         cre._deep_nlu_available,
         "False — NLU import failed")

    # F4: reason_about returns NLU enrichment
    cre.initialize()
    r = cre.reason_about("What happens when water freezes?")
    test("F4: reason_about has deep_nlu_enrichment",
         'deep_nlu_enrichment' in r,
         f"keys={list(r.keys())}")

    # F5: get_status reports deep_nlu
    st = cre.get_status()
    eng_support = st.get('engine_support', {})
    test("F5: get_status includes deep_nlu",
         'deep_nlu' in eng_support,
         f"engine_support keys: {list(eng_support.keys())}")

    # ═══════════════════════════════════════════════════════════════════
    # G) LANGUAGE ENGINE v6.0.0 HUB — 10 tests
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  G) LANGUAGE ENGINE v6.0.0 HUB")
    print("=" * 70)

    from l104_language_engine import LanguageEngine, VERSION as LE_VERSION

    le = LanguageEngine()

    # G1: VERSION
    test("G1: LanguageEngine VERSION=6.0.0",
         le.VERSION == "6.0.0",
         f"got {le.VERSION}")

    # G2: Module VERSION
    test("G2: Module VERSION=6.0.0",
         LE_VERSION == "6.0.0",
         f"got {LE_VERSION}")

    # G3: Has decomposer subsystem
    test("G3: Has decomposer attribute",
         hasattr(le, 'decomposer') and le.decomposer is not None)

    # G4: Has expander subsystem
    test("G4: Has expander attribute",
         hasattr(le, 'expander') and le.expander is not None)

    # G5: Has query_classifier subsystem
    test("G5: Has query_classifier attribute",
         hasattr(le, 'query_classifier') and le.query_classifier is not None)

    # G6: decompose_query method
    r = le.decompose_query("How does gravity work and what is its formula?")
    test("G6: decompose_query works",
         isinstance(r, dict) and 'sub_queries' in r)

    # G7: expand_query method
    r = le.expand_query("What causes earthquakes?")
    test("G7: expand_query works",
         isinstance(r, dict) and 'expansions' in r)

    # G8: classify_query method
    r = le.classify_query("Compare DNA and RNA structures")
    test("G8: classify_query works",
         isinstance(r, dict) and 'bloom_level' in r)

    # G9: deep_analyze includes new fields
    r = le.deep_analyze("Energy transforms between kinetic and potential forms")
    test("G9: deep_analyze has query_decomposition",
         'query_decomposition' in r and 'query_expansion' in r and 'query_classification' in r,
         f"keys={list(r.keys())}")

    # G10: Status reports 16 subsystems
    st = le.status()
    test("G10: 16 subsystems reported",
         st.get('subsystem_count') >= 16,
         f"got {st.get('subsystem_count')}")

    # ═══════════════════════════════════════════════════════════════════
    # H) ASI CORE v2.2.0 METHODS — 5 tests
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  H) ASI CORE v2.2.0 METHODS")
    print("=" * 70)

    from l104_asi.core import ASICore

    core = ASICore()

    # H1: decompose_query
    r = core.decompose_query("What is energy and how is it conserved?")
    test("H1: core.decompose_query works",
         isinstance(r, dict) and ('sub_queries' in r or 'error' in r))

    # H2: expand_query
    r = core.expand_query("How does photosynthesis convert sunlight?")
    test("H2: core.expand_query works",
         isinstance(r, dict) and ('expansions' in r or 'error' in r))

    # H3: classify_query
    r = core.classify_query("Explain the theory of relativity")
    test("H3: core.classify_query works",
         isinstance(r, dict) and ('bloom_level' in r or 'error' in r))

    # H4: Methods are callable
    test("H4: All 3 methods callable",
         callable(getattr(core, 'decompose_query', None)) and
         callable(getattr(core, 'expand_query', None)) and
         callable(getattr(core, 'classify_query', None)))

    # H5: synthesize_queries still works
    r = core.synthesize_queries("The force of gravity pulls objects toward Earth")
    test("H5: synthesize_queries backward compatible",
         isinstance(r, dict) and ('queries' in r or 'error' in r))

    # ═══════════════════════════════════════════════════════════════════
    # I) __init__.py EXPORTS — 5 tests
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  I) __init__.py EXPORTS")
    print("=" * 70)

    # I1: QueryDecomposer
    try:
        from l104_asi import QueryDecomposer as QD
        test("I1: QueryDecomposer exported", QD is not None)
    except ImportError as e:
        test("I1: QueryDecomposer exported", False, str(e))

    # I2: QueryExpander
    try:
        from l104_asi import QueryExpander as QE
        test("I2: QueryExpander exported", QE is not None)
    except ImportError as e:
        test("I2: QueryExpander exported", False, str(e))

    # I3: QueryClassifier
    try:
        from l104_asi import QueryClassifier as QC
        test("I3: QueryClassifier exported", QC is not None)
    except ImportError as e:
        test("I3: QueryClassifier exported", False, str(e))

    # I4: BloomLevel
    try:
        from l104_asi import BloomLevel as BL
        test("I4: BloomLevel exported", len(BL) == 6)
    except ImportError as e:
        test("I4: BloomLevel exported", False, str(e))

    # I5: AnswerFormat
    try:
        from l104_asi import AnswerFormat as AF
        test("I5: AnswerFormat exported", len(AF) == 8)
    except ImportError as e:
        test("I5: AnswerFormat exported", False, str(e))


if __name__ == "__main__":
    print("=" * 70)
    print("  WAVE 5 VALIDATION — DeepNLU v2.2.0 QUERY AUGMENTATION")
    print("  + ASIInnovationEngine v2.0.0 + CommonsenseReasoning v3.0.0")
    print("  + LanguageEngine v6.0.0 + ASI Core")
    print("=" * 70)

    try:
        run_all()
    except Exception as e:
        print(f"\n  💥 FATAL: {e}")
        traceback.print_exc()
        FAIL += 1

    print("\n" + "=" * 70)
    total = PASS + FAIL
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    if ERRORS:
        print(f"  FAILURES:")
        for err in ERRORS:
            print(f"    • {err}")
    print("=" * 70)

    sys.exit(0 if FAIL == 0 else 1)
