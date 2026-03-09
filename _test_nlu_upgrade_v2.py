#!/usr/bin/env python3
"""Validation test for NLU Upgrade Wave 2 — all files, all new capabilities."""
import sys

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}")

# ═══════════════════════════════════════════════════════════════════════
# 1. DeepNLU v2.0.0 — 13-layer engine with expanded inventory
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 1: DeepNLU v2.0.0 ===")
from l104_asi.deep_nlu import (
    DeepNLUEngine, TemporalReasoner, CausalReasoner, ContextualDisambiguator,
    TemporalRelation, CausalRelationType,
)
nlu = DeepNLUEngine()
check("VERSION is 2.0.0", nlu.VERSION == "2.0.0")

st = nlu.status()
check("Reports 13 layers", st.get('layers') == 13)
check("Sense inventory >= 21 words", st.get('sense_inventory_size', 0) >= 21)

# Test temporal
tr = nlu.analyze_temporal("Yesterday I went to the store and tomorrow I will go again.")
check("Temporal has tense", 'tense' in tr)

# Test causal
cr = nlu.analyze_causal("The rain caused the flood because the dam broke.")
check("Causal has causal_pairs", 'causal_pairs' in cr)

# Test disambiguation
dr = nlu.disambiguate("I need to write better code for the genetic code project.")
check("Disambiguation has results", 'disambiguations' in dr)

# Test expanded inventory
inv = ContextualDisambiguator.SENSE_INVENTORY
check("Has 'wave' word", 'wave' in inv)
check("Has 'code' word", 'code' in inv)
check("Has 'bridge' word", 'bridge' in inv)
check("Has 'drive' word", 'drive' in inv)

# ═══════════════════════════════════════════════════════════════════════
# 2. LanguageComprehensionEngine v6.0.0
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 2: LanguageComprehensionEngine v6.0.0 ===")
from l104_asi.language_comprehension import LanguageComprehensionEngine
lce = LanguageComprehensionEngine()
check("LCE VERSION is 7.0.0", getattr(lce, 'VERSION', None) == "7.0.0")

comp = lce.comprehend("The rain caused flooding yesterday and will continue tomorrow.")
check("comprehend has temporal", 'temporal' in comp)
check("comprehend has causal", 'causal' in comp)
check("comprehend has disambiguation", 'disambiguation' in comp)

# ═══════════════════════════════════════════════════════════════════════
# 3. ASILinguisticAnalyzer v2.0.0
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 3: ASILinguisticAnalyzer v2.0.0 ===")
from l104_asi_language_engine import ASILinguisticAnalyzer
la = ASILinguisticAnalyzer()
check("Analyzer VERSION is 2.0.0", la.VERSION == "2.0.0")

ar = la.analyze("Heavy rain caused the river to flood the bank.")
check("analyze has temporal", 'temporal' in ar)
check("analyze has causal", 'causal' in ar)
check("analyze has disambiguation", 'disambiguation' in ar)
check("analyze has deep_nlu_available", 'deep_nlu_available' in ar)

# New methods
check("has analyze_temporal()", hasattr(la, 'analyze_temporal'))
check("has analyze_causal()", hasattr(la, 'analyze_causal'))
check("has disambiguate()", hasattr(la, 'disambiguate'))

# ═══════════════════════════════════════════════════════════════════════
# 4. ASILanguageEngine v2.0.0
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 4: ASILanguageEngine v2.0.0 ===")
from l104_asi_language_engine import ASILanguageEngine
ale = ASILanguageEngine()
check("ASILanguageEngine VERSION is 2.0.0", ale.VERSION == "2.0.0")
ales = ale.get_status()
check("Status has deep_nlu_temporal", 'deep_nlu_temporal' in ales.get('components', {}))
check("Status has deep_nlu_causal", 'deep_nlu_causal' in ales.get('components', {}))

# ═══════════════════════════════════════════════════════════════════════
# 5. LanguageEngine hub v4.0.0
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 5: LanguageEngine v4.0.0 ===")
from l104_language_engine import language_engine, VERSION
check("VERSION is 4.0.0", VERSION == "4.0.0")
check("Hub VERSION is 4.0.0", language_engine.VERSION == "4.0.0")
check("DeepNLU available", language_engine._deep_nlu_available)

les = language_engine.status()
check("Status has 12 subsystems", les.get('subsystem_count') == 12)
check("Status reports deep_nlu_available", les.get('deep_nlu_available') is True)
check("Has temporal_reasoner subsystem", 'temporal_reasoner' in les.get('subsystems', {}))
check("Has causal_reasoner subsystem", 'causal_reasoner' in les.get('subsystems', {}))
check("Has contextual_disambiguator", 'contextual_disambiguator' in les.get('subsystems', {}))

# Test new hub methods
ta = language_engine.analyze_temporal("I will go to the store tomorrow after lunch.")
check("analyze_temporal works", 'tense' in ta)

ca = language_engine.analyze_causal("The earthquake caused a tsunami.")
check("analyze_causal works", 'causal_pairs' in ca)

da = language_engine.disambiguate("The bright light from the wave illuminated the bridge.")
check("disambiguate works", 'disambiguations' in da)

# Test deep_analyze composite
deep = language_engine.deep_analyze("The rain caused flooding at the river bank yesterday.")
check("deep_analyze has sentiment", 'sentiment' in deep)
check("deep_analyze has temporal", 'temporal' in deep)
check("deep_analyze has causal", 'causal' in deep)
check("deep_analyze has disambiguation", 'disambiguation' in deep)
check("deep_analyze reports v2.0.0", deep.get('deep_nlu_version') == '2.0.0')

# ═══════════════════════════════════════════════════════════════════════
# 6. l104_asi __init__ exports
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 6: Package exports ===")
from l104_asi import (
    TemporalReasoner as TR, CausalReasoner as CR,
    ContextualDisambiguator as CD,
    TemporalRelation as TRel, CausalRelationType as CRT,
)
check("TemporalReasoner exported", TR is TemporalReasoner)
check("CausalReasoner exported", CR is CausalReasoner)
check("ContextualDisambiguator exported", CD is ContextualDisambiguator)
check("TemporalRelation exported", TRel is TemporalRelation)
check("CausalRelationType exported", CRT is CausalRelationType)

# ═══════════════════════════════════════════════════════════════════════
# 7. ASI Core integration
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 7: ASI Core NLU integration ===")
from l104_asi import asi_core
check("asi_core has analyze_temporal", hasattr(asi_core, 'analyze_temporal'))
check("asi_core has analyze_causal", hasattr(asi_core, 'analyze_causal'))
check("asi_core has disambiguate", hasattr(asi_core, 'disambiguate'))

# deep_nlu_score should benefit from 13-layer base (0.72 vs old 0.6)
score = asi_core.deep_nlu_score()
check("deep_nlu_score >= 0.72", score >= 0.72)

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
