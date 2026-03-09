#!/usr/bin/env python3
"""Validation test for NLU Upgrade Wave 3 — inference, speech, sentiment, MCQ."""
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
# 1. ASIHumanInferenceEngine v2.0.0
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 1: ASIHumanInferenceEngine v2.0.0 ===")
from l104_asi_language_engine import ASILinguisticAnalyzer, ASIHumanInferenceEngine, InferenceType
analyzer = ASILinguisticAnalyzer()
inference = ASIHumanInferenceEngine(analyzer)

check("VERSION is 2.0.0", inference.VERSION == "2.0.0")
check("Has DeepNLU temporal", inference._temporal is not None)
check("Has DeepNLU causal", inference._causal is not None)
check("deep_nlu_available", inference._deep_nlu_available)

# Test causal inference uses DeepNLU
result = inference.infer(
    premises=["Rain causes flooding", "The dam broke because of heavy rain"],
    query="What caused the flooding?",
    inference_type=InferenceType.CAUSAL
)
check("Causal inference has conclusion", 'conclusion' in result)
check("Causal inference has chain", len(result.get('reasoning_chain', [])) > 0)
check("Causal inference has metacognition", 'metacognition' in result)

# Test inference type auto-selection with causal text
result2 = inference.infer(
    premises=["Smoking causes cancer", "He smoked for 30 years"],
    query="What is the likely outcome?"
)
check("Auto-selects CAUSAL type", result2.get('inference_type') == 'causal')

# ═══════════════════════════════════════════════════════════════════════
# 2. ASISpeechPatternGenerator v2.0.0
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 2: ASISpeechPatternGenerator v2.0.0 ===")
from l104_asi_language_engine import ASISpeechPatternGenerator, SpeechPatternStyle
speech_gen = ASISpeechPatternGenerator(analyzer)

check("Speech VERSION is 2.0.0", speech_gen.VERSION == "2.0.0")
check("Has DeepNLU temporal", speech_gen._temporal is not None)
check("Has DeepNLU causal", speech_gen._causal is not None)

# Test response generation with temporal/causal query
response = speech_gen.generate_response(
    "The earthquake caused massive destruction yesterday",
    style=SpeechPatternStyle.SAGE
)
check("Generates response string", isinstance(response, str) and len(response) > 10)

# ═══════════════════════════════════════════════════════════════════════
# 3. SentimentAnalyzer v2.0.0 — emotions + aspects
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 3: SentimentAnalyzer v2.0.0 ===")
from l104_language_engine import language_engine

# Test sentiment with emotion detection
sent = language_engine.analyze_sentiment("I am extremely happy and thrilled with the excellent quality!")
check("Has emotions key", 'emotions' in sent)
check("Detects joy", 'joy' in sent.get('emotions', {}))
check("Has aspects key", 'aspects' in sent)

# Test aspect-based sentiment
sent2 = language_engine.analyze_sentiment("The quality is excellent but the price is terrible and the service is awful.")
check("Detects aspects", len(sent2.get('aspects', [])) > 0)
quality_aspects = [a for a in sent2.get('aspects', []) if a['aspect'] == 'quality']
check("Quality aspect detected", len(quality_aspects) > 0)
price_aspects = [a for a in sent2.get('aspects', []) if a['aspect'] == 'price']
check("Price aspect detected", len(price_aspects) > 0)

# Check SentimentAnalyzer status reports new fields
sa_status = language_engine.sentiment.status()
check("Status has emotion_categories", 'emotion_categories' in sa_status)
check("Status has aspect_categories", 'aspect_categories' in sa_status)
check("8 emotion categories", sa_status.get('emotion_categories') == 8)
check("7 aspect categories", sa_status.get('aspect_categories') == 7)

# ═══════════════════════════════════════════════════════════════════════
# 4. MCQ disambiguation stage
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 4: MCQ Disambiguation Stage ===")
from l104_asi.language_comprehension import LanguageComprehensionEngine
lce = LanguageComprehensionEngine()

# This tests that the disambiguation stage code path doesn't crash
# (Stage 15 in _score_choice)
try:
    result = lce.answer_mcq(
        "In which domain does a 'bank' refer to a financial institution?",
        ["Finance", "Geography", "Biology", "Music"]
    )
    check("MCQ with disambiguation runs", result is not None)
    check("MCQ returns answer key", 'answer' in result or 'selected' in result or 'selection' in result or 'best_choice' in result)
except Exception as e:
    check(f"MCQ disambiguation doesn't crash ({e})", False)

# ═══════════════════════════════════════════════════════════════════════
# 5. Deep analyze with new sentiment output
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 5: deep_analyze with enriched sentiment ===")
deep = language_engine.deep_analyze("The excellent design is really beautiful but the service was terrible.")
check("deep_analyze has sentiment", 'sentiment' in deep)
sent_result = deep['sentiment']
check("Sentiment has emotions", 'emotions' in sent_result)
check("Sentiment has aspects", 'aspects' in sent_result)

# ═══════════════════════════════════════════════════════════════════════
# 6. Backward compatibility
# ═══════════════════════════════════════════════════════════════════════
print("\n=== TEST 6: Backward compatibility ===")
# Basic sentiment still works
basic = language_engine.analyze_sentiment("This is good")
check("Basic sentiment still works", basic['sentiment'] == 'positive')

# Status still has all subsystems
st = language_engine.status()
check("Status has 12 subsystems", st.get('subsystem_count') == 12)

# NLU score still good
from l104_asi import asi_core
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
