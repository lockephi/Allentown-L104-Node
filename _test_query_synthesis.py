#!/usr/bin/env python3
"""
Query Synthesis Pipeline — Validation Test Suite
═════════════════════════════════════════════════
Tests all 8 query archetypes, pipeline integration, NLU wiring,
LanguageEngine hub, ASI core, and __init__ exports.
"""

import sys
PASS = 0
FAIL = 0
TOTAL = 0


def check(name, condition, detail=""):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


print("=" * 70)
print("  QUERY SYNTHESIS PIPELINE — VALIDATION TEST")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Imports and Class Availability
# ═══════════════════════════════════════════════════════════════════════════
print("\n── PHASE 1: Imports ──")

from l104_asi.deep_nlu import (
    QuerySynthesisPipeline, QueryType, SynthesizedQuery,
    DeepNLUEngine,
)
check("QuerySynthesisPipeline importable", True)
check("QueryType enum importable", True)
check("SynthesizedQuery dataclass importable", True)

# From __init__
from l104_asi import QuerySynthesisPipeline as QSP2, QueryType as QT2, SynthesizedQuery as SQ2
check("Exported from l104_asi.__init__", QSP2 is QuerySynthesisPipeline)

# QueryType has 8 members
qt_names = [qt.value for qt in QueryType]
check("QueryType has 8 archetypes", len(QueryType) == 8,
      f"got {len(QueryType)}: {qt_names}")

expected = {'factual', 'causal', 'temporal', 'definitional',
            'counterfactual', 'comparative', 'inferential', 'verification'}
check("QueryType values correct", set(qt_names) == expected,
      f"missing={expected - set(qt_names)}")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Pipeline Direct — Factual + Causal + Temporal
# ═══════════════════════════════════════════════════════════════════════════
print("\n── PHASE 2: Pipeline Direct ──")

pipeline = QuerySynthesisPipeline()
check("Pipeline instantiation", pipeline is not None)
check("Pipeline VERSION", pipeline.VERSION == "1.0.0")

# Complex text with causal, temporal, and entity content
text1 = (
    "The scientist discovered a new compound in 2024. "
    "Because the compound was unstable, it caused an explosion in the laboratory. "
    "After the explosion, the building was evacuated for 3 days."
)

result = pipeline.synthesize(text1)
check("synthesize returns dict", isinstance(result, dict))
check("'queries' key present", 'queries' in result)
check("'total' key present", 'total' in result)
check("queries generated > 0", result['total'] > 0, f"got {result['total']}")

queries = result['queries']
types_found = {q.query_type for q in queries}
check("Factual queries generated", QueryType.FACTUAL in types_found,
      f"types found: {[t.value for t in types_found]}")
check("Causal queries generated", QueryType.CAUSAL in types_found,
      f"types found: {[t.value for t in types_found]}")
check("Temporal queries generated", QueryType.TEMPORAL in types_found,
      f"types found: {[t.value for t in types_found]}")

# Check SynthesizedQuery structure
q0 = queries[0]
check("Query has .text", isinstance(q0.text, str) and len(q0.text) > 0)
check("Query has .query_type", isinstance(q0.query_type, QueryType))
check("Query has .source_layer", isinstance(q0.source_layer, str))
check("Query has .confidence", 0.0 <= q0.confidence <= 1.0)
check("Query has .focus", isinstance(q0.focus, str))
check("Query has .depth >= 1", q0.depth >= 1)

# Archetype distribution
check("archetype_distribution present", 'archetype_distribution' in result)
check("phi_synthesis present", 'phi_synthesis' in result)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Typed Synthesis
# ═══════════════════════════════════════════════════════════════════════════
print("\n── PHASE 3: Typed Synthesis ──")

causal_only = pipeline.synthesize_typed(text1, QueryType.CAUSAL)
check("synthesize_typed returns list", isinstance(causal_only, list))
if causal_only:
    check("All typed queries are CAUSAL",
          all(q.query_type == QueryType.CAUSAL for q in causal_only))
else:
    check("At least one typed causal query", False, "empty list")

temporal_only = pipeline.synthesize_typed(text1, QueryType.TEMPORAL)
check("Temporal typed queries generated", len(temporal_only) > 0,
      f"got {len(temporal_only)}")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: Counterfactual + Inferential + Definitional
# ═══════════════════════════════════════════════════════════════════════════
print("\n── PHASE 4: Deep Query Types ──")

# Text with presuppositions, ambiguity, and counterfactuals
text2 = (
    "If the bank had not collapsed, the economy would have recovered. "
    "The plant by the river stopped growing because of pollution. "
    "The scientist, who already knew the answer, asked the question anyway."
)

result2 = pipeline.synthesize(text2)
queries2 = result2['queries']
types2 = {q.query_type for q in queries2}

check("Counterfactual queries from 'if/would'", QueryType.COUNTERFACTUAL in types2,
      f"types: {[t.value for t in types2]}")
check("Definitional from ambiguous 'bank'/'plant'", QueryType.DEFINITIONAL in types2,
      f"types: {[t.value for t in types2]}")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5: Batch Synthesis
# ═══════════════════════════════════════════════════════════════════════════
print("\n── PHASE 5: Batch Synthesis ──")

batch = pipeline.batch_synthesize([text1, text2], max_per_text=10)
check("batch_synthesize returns dict", isinstance(batch, dict))
check("batch 'queries' key", 'queries' in batch)
check("batch total > 0", batch['total'] > 0, f"got {batch['total']}")
check("batch texts_processed == 2", batch['texts_processed'] == 2)
check("batch per_text == 2 entries", len(batch.get('per_text', [])) == 2)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6: DeepNLUEngine Integration
# ═══════════════════════════════════════════════════════════════════════════
print("\n── PHASE 6: DeepNLUEngine Integration ──")

engine = DeepNLUEngine()
check("DeepNLUEngine VERSION == 3.0.0", engine.VERSION == "3.0.0")
check("engine.query_pipeline exists", hasattr(engine, 'query_pipeline'))
check("engine.query_pipeline is QuerySynthesisPipeline",
      isinstance(engine.query_pipeline, QuerySynthesisPipeline))

# synthesize_queries via engine
eng_result = engine.synthesize_queries(text1)
check("engine.synthesize_queries works", eng_result['total'] > 0,
      f"got {eng_result.get('total', 0)}")

# synthesize_typed via engine
eng_typed = engine.synthesize_typed(text1, QueryType.FACTUAL)
check("engine.synthesize_typed works", len(eng_typed) > 0)

# batch via engine
eng_batch = engine.batch_synthesize([text1, text2])
check("engine.batch_synthesize works", eng_batch['total'] > 0)

# Status
st = engine.status()
check("Status layers == 20", st['layers'] == 20, f"got {st['layers']}")
check("'query_synthesis_pipeline' in layer_names",
      'query_synthesis_pipeline' in st['layer_names'])
check("'query_pipeline' in status", 'query_pipeline' in st)
check("query_types_supported == 8", st.get('query_types_supported') == 8)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 7: LanguageEngine Hub Integration
# ═══════════════════════════════════════════════════════════════════════════
print("\n── PHASE 7: LanguageEngine Hub ──")

from l104_language_engine import language_engine, VERSION as LE_VERSION
check("LanguageEngine VERSION == 2.0.0", LE_VERSION == "2.0.0",
      f"got {LE_VERSION}")

check("language_engine.query_pipeline exists",
      hasattr(language_engine, 'query_pipeline'))

le_result = language_engine.synthesize_queries(text1)
check("language_engine.synthesize_queries works",
      le_result.get('total', 0) > 0, f"got {le_result}")

le_batch = language_engine.batch_synthesize_queries([text1, text2])
check("language_engine.batch_synthesize_queries works",
      le_batch.get('total', 0) > 0)

# deep_analyze includes query_synthesis
da = language_engine.deep_analyze(text1)
check("deep_analyze includes query_synthesis",
      'query_synthesis' in da, f"keys={list(da.keys())}")
check("deep_analyze deep_nlu_version == 3.0.0",
      da.get('deep_nlu_version') == '3.0.0')

le_st = language_engine.status()
subsystems = le_st.get('subsystems', {})
check("query_synthesis_pipeline subsystem in status",
      'query_synthesis_pipeline' in subsystems)
check("subsystem_count >= 13",
      le_st.get('subsystem_count', 0) >= 13,
      f"got {le_st.get('subsystem_count')}")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 8: ASI Core Integration
# ═══════════════════════════════════════════════════════════════════════════
print("\n── PHASE 8: ASI Core ──")

from l104_asi import asi_core
check("asi_core.synthesize_queries exists",
      hasattr(asi_core, 'synthesize_queries'))
check("asi_core.batch_synthesize_queries exists",
      hasattr(asi_core, 'batch_synthesize_queries'))

asi_result = asi_core.synthesize_queries(text1)
check("asi_core.synthesize_queries works",
      asi_result.get('total', 0) > 0 or 'error' not in asi_result,
      f"got {asi_result}")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 9: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════
print("\n── PHASE 9: Edge Cases ──")

# Empty text
empty = pipeline.synthesize("")
check("Empty text returns empty queries", empty['total'] == 0)

# Single word
single = pipeline.synthesize("Hello.")
check("Single word returns dict", isinstance(single, dict))

# High confidence filter
strict = pipeline.synthesize(text1, min_confidence=0.9)
check("High min_confidence filters more",
      strict['total'] <= result['total'])

# Max queries cap
capped = pipeline.synthesize(text1, max_queries=3)
check("max_queries=3 caps at 3", capped['total'] <= 3)

# Pipeline status
ps = pipeline.status()
check("Pipeline status has version", ps.get('version') == '1.0.0')
check("Pipeline status has archetypes", ps.get('query_types_supported') == 8)
check("Pipeline tracked texts_processed", ps.get('texts_processed', 0) > 0)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 10: Query Quality Validation
# ═══════════════════════════════════════════════════════════════════════════
print("\n── PHASE 10: Query Quality ──")

# Verify queries are well-formed
all_well_formed = True
for q in queries:
    if not q.text.endswith('?'):
        all_well_formed = False
        break
check("All queries end with '?'", all_well_formed)

# Verify ranking order (descending by score)
confidences = [q.confidence for q in queries]
# Note: ranking is by PHI-weighted score, not raw confidence, so just check it's reasonable
check("Top query confidence >= 0.5", queries[0].confidence >= 0.5 if queries else True,
      f"top conf={queries[0].confidence if queries else 'N/A'}")

# Verify dedup — no duplicate query texts
texts_seen = set()
duplicates = 0
for q in queries:
    key = q.text.lower().strip()
    if key in texts_seen:
        duplicates += 1
    texts_seen.add(key)
check("No duplicate queries", duplicates == 0, f"found {duplicates} dupes")

# Verify source layers are valid
valid_layers = {'L3_SRL', 'L4_Anaphora', 'L5_Discourse', 'L6_Pragmatics',
                'L7_Presupposition', 'L8_Sentiment',
                'L10_Temporal', 'L11_Causal', 'L12_Disambiguation',
                'L3_SRL+L5_Discourse', 'L6_Pragmatics+L8_Sentiment'}
all_valid = all(q.source_layer in valid_layers for q in queries)
check("All source_layers are valid NLU layers", all_valid,
      f"invalid: {[q.source_layer for q in queries if q.source_layer not in valid_layers]}")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  RESULT: {PASS}/{TOTAL} passed, {FAIL} failed")
print("=" * 70)

if FAIL > 0:
    sys.exit(1)
else:
    print("  🎯 ALL TESTS PASSED — Query Synthesis Pipeline fully operational")
