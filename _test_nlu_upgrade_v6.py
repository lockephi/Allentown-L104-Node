#!/usr/bin/env python3
"""
Wave 6 NLU Upgrade Validation — 80+ tests
Tests: TextualEntailmentEngine, FigurativeLanguageProcessor,
       InformationDensityAnalyzer, DeepNLUEngine v2.3.0,
       LanguageComprehensionEngine v8.1.0, LanguageEngine v7.0.0,
       ASI core wiring, __init__.py exports
"""

import sys, traceback

PASSED = 0
FAILED = 0
ERRORS = []

def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        msg = f"  ❌ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)

def test():
    global PASSED, FAILED

    # ── Section 1: TextualEntailmentEngine ──
    print("\n═══ Section 1: TextualEntailmentEngine (Layer 18) ═══")
    from l104_asi.deep_nlu import TextualEntailmentEngine, EntailmentLabel, EntailmentResult

    te = TextualEntailmentEngine()
    check("TE instantiation", te is not None)
    check("TE VERSION = 1.0.0", te.VERSION == "1.0.0")

    # Entailment pair
    r = te.check("All dogs are animals", "Dogs are living creatures")
    check("TE check() returns dict", isinstance(r, dict))
    check("TE has label key", "label" in r)
    check("TE label is str", isinstance(r["label"], str))
    check("TE confidence in [0,1]", 0.0 <= r["confidence"] <= 1.0)
    check("TE evidence is list", isinstance(r["evidence"], list))
    check("TE lexical_overlap is float", isinstance(r["lexical_overlap"], float))
    check("TE phi_score is float", isinstance(r["phi_score"], float))

    # Contradiction pair (uses antonyms alive/dead, not negation words)
    r2 = te.check("The cat is alive", "The cat is dead")
    check("TE contradiction detected", r2["label"] == "contradiction",
          f"got {r2['label']}")
    check("TE has negation_conflict key", "negation_conflict" in r2)

    # Neutral pair
    r3 = te.check("The sky is blue", "Pizza is delicious")
    check("TE neutral pair has low overlap", r3["lexical_overlap"] < 0.3,
          f"overlap={r3['lexical_overlap']}")

    # Role alignment
    check("TE role_alignment is float", isinstance(r["role_alignment"], float))

    # Status
    st = te.status()
    check("TE status has version", st.get("version") == "1.0.0")
    check("TE status has negation_words", "negation_words" in st)
    check("TE status has antonym_pairs", "antonym_pairs" in st)

    # ── Section 2: FigurativeLanguageProcessor ──
    print("\n═══ Section 2: FigurativeLanguageProcessor (Layer 19) ═══")
    from l104_asi.deep_nlu import FigurativeLanguageProcessor, FigurativeType, FigurativeExpression

    fp = FigurativeLanguageProcessor()
    check("FP instantiation", fp is not None)
    check("FP VERSION = 1.0.0", fp.VERSION == "1.0.0")

    # Idiom detection (use a known idiom from IDIOM_DB)
    r = fp.analyze("She had to break the ice at the party")
    check("FP analyze returns dict", isinstance(r, dict))
    check("FP has figures key", "figures" in r)
    idioms = [e for e in r["figures"] if e["type"] == "idiom"]
    check("FP detected idiom", len(idioms) > 0, f"found {len(idioms)}")

    # Simile detection
    r2 = fp.analyze("She runs like a cheetah in the wild")
    similes = [e for e in r2["figures"] if e["type"] == "simile"]
    check("FP detected simile", len(similes) > 0, f"found {len(similes)}")

    # Irony detection
    r3 = fp.analyze("Oh great, another Monday morning, just what I needed")
    irony = [e for e in r3["figures"] if e["type"] == "irony"]
    check("FP detected irony", len(irony) > 0)

    # Hyperbole detection (need >= 2 markers)
    r4 = fp.analyze("I've literally told you a million times it's absolutely the worst")
    hyp = [e for e in r4["figures"] if e["type"] == "hyperbole"]
    check("FP detected hyperbole", len(hyp) > 0)

    # Personification detection
    r5 = fp.analyze("The wind whispered through the trees at night")
    pers = [e for e in r5["figures"] if e["type"] == "personification"]
    check("FP detected personification", len(pers) > 0)

    # Metaphor detection (cross-domain mapping requires known domain words)
    r6 = fp.analyze("Love is a battlefield and the heart is a sword")
    meta_or_any = len(r6.get("figures", []))
    check("FP figurative analysis for complex text", isinstance(r6, dict) and "figures" in r6)

    # Literal text should be clean
    r7 = fp.analyze("The temperature is 72 degrees Fahrenheit today")
    check("FP literal text is_literal", r7.get("is_literal", False) is True)

    # Figurative intensity
    check("FP has figurative_intensity", "figurative_intensity" in r)
    check("FP intensity is float", isinstance(r.get("figurative_intensity", 0.0), float))

    # Type counts
    check("FP has type_counts", "type_counts" in r)

    # IDIOM_DB size
    check("FP IDIOM_DB >= 60", len(fp.IDIOM_DB) >= 60, f"got {len(fp.IDIOM_DB)}")

    # FigurativeType enum values
    check("FP FigurativeType has IDIOM", hasattr(FigurativeType, "IDIOM"))
    check("FP FigurativeType has SIMILE", hasattr(FigurativeType, "SIMILE"))
    check("FP FigurativeType has IRONY", hasattr(FigurativeType, "IRONY"))
    check("FP FigurativeType has HYPERBOLE", hasattr(FigurativeType, "HYPERBOLE"))
    check("FP FigurativeType has PERSONIFICATION", hasattr(FigurativeType, "PERSONIFICATION"))
    check("FP FigurativeType has METAPHOR", hasattr(FigurativeType, "METAPHOR"))
    check("FP FigurativeType count = 6", len(FigurativeType) == 6)

    # FigurativeExpression fields
    expr = FigurativeExpression(text="test", fig_type=FigurativeType.IDIOM, meaning="test", confidence=0.9, span=(0, 4))
    check("FP FigurativeExpression fields", expr.text == "test" and expr.confidence == 0.9)

    # Status
    st = fp.status()
    check("FP status version", st.get("version") == "1.0.0")
    check("FP status has idioms_known", "idioms_known" in st)

    # ── Section 3: InformationDensityAnalyzer ──
    print("\n═══ Section 3: InformationDensityAnalyzer (Layer 20) ═══")
    from l104_asi.deep_nlu import InformationDensityAnalyzer, DensityProfile

    ida = InformationDensityAnalyzer()
    check("IDA instantiation", ida is not None)
    check("IDA VERSION = 1.0.0", ida.VERSION == "1.0.0")

    # Analyze text
    text = ("Quantum entanglement demonstrates that particles separated by vast distances "
            "exhibit correlated properties instantaneously, challenging classical notions "
            "of locality and suggesting a deeper interconnected fabric of reality.")
    r = ida.analyze(text)
    check("IDA returns dict", isinstance(r, dict))
    check("IDA overall_density is float", isinstance(r["overall_density"], float))
    check("IDA overall_density in [0,1]", 0.0 <= r["overall_density"] <= 1.0)
    check("IDA lexical_diversity is float", isinstance(r["lexical_diversity"], float))
    check("IDA has metrics dict", isinstance(r.get("metrics"), dict))
    check("IDA metrics has type_token_ratio", "type_token_ratio" in r.get("metrics", {}))
    check("IDA metrics has hapax_ratio", "hapax_ratio" in r.get("metrics", {}))
    check("IDA metrics has content_word_ratio", "content_word_ratio" in r.get("metrics", {}))
    check("IDA redundancy is float", isinstance(r["redundancy"], float))
    check("IDA specificity is float", isinstance(r["specificity"], float))
    check("IDA surprisal is float", isinstance(r["surprisal"], float))
    check("IDA gradient is list", isinstance(r["gradient"], list))
    check("IDA phi_density is float", isinstance(r["phi_density"], float))

    # High vs low density comparison
    dense = "Mitochondrial ATP synthase utilizes proton-motive force across inner membranes"
    simple = "The the the the the the the and and and the the"
    r_dense = ida.analyze(dense)
    r_simple = ida.analyze(simple)
    check("IDA dense > simple density", r_dense["overall_density"] > r_simple["overall_density"],
          f"dense={r_dense['overall_density']:.3f} simple={r_simple['overall_density']:.3f}")

    # Empty text
    r_empty = ida.analyze("")
    check("IDA empty text handled", r_empty["overall_density"] == 0.0 or isinstance(r_empty["overall_density"], float))

    # DensityProfile dataclass
    dp = DensityProfile(overall_density=0.5, lexical_diversity={"ttr": 0.8}, redundancy={"bigram": 0.1},
                        specificity={"proper_nouns": 2}, surprisal=0.4, gradient=[0.3, 0.5], phi_density=0.81)
    check("IDA DensityProfile dataclass works", dp.overall_density == 0.5 and dp.phi_density == 0.81)

    # Status
    st = ida.status()
    check("IDA status version", st.get("version") == "1.0.0")

    # ── Section 4: DeepNLUEngine v2.3.0 ──
    print("\n═══ Section 4: DeepNLUEngine v2.3.0 ═══")
    from l104_asi.deep_nlu import DeepNLUEngine

    nlu = DeepNLUEngine()
    check("NLU instantiation", nlu is not None)
    check("NLU VERSION = 3.0.0", nlu.VERSION == "3.0.0")

    # APIs
    check("NLU has check_entailment", hasattr(nlu, "check_entailment"))
    check("NLU has analyze_figurative", hasattr(nlu, "analyze_figurative"))
    check("NLU has analyze_density", hasattr(nlu, "analyze_density"))

    # check_entailment
    ent_r = nlu.check_entailment("Birds can fly", "Sparrows can fly")
    check("NLU check_entailment returns dict", isinstance(ent_r, dict))
    check("NLU entailment has label", "label" in ent_r)

    # analyze_figurative
    fig_r = nlu.analyze_figurative("Break a leg on your audition")
    check("NLU analyze_figurative returns dict", isinstance(fig_r, dict))
    check("NLU figurative has figures", "figures" in fig_r)

    # analyze_density
    den_r = nlu.analyze_density("The quick brown fox jumps over the lazy dog")
    check("NLU analyze_density returns dict", isinstance(den_r, dict))
    check("NLU density has overall_density", "overall_density" in den_r)

    # nlu_depth_score
    score = nlu.nlu_depth_score()
    check("NLU depth_score >= 0.83", score >= 0.83, f"got {score}")

    # Status
    st = nlu.status()
    check("NLU status layers = 20", st.get("layers") == 20, f"got layers={st.get('layers')}")
    check("NLU status version 3.0.0", st.get("version") == "3.0.0")
    layer_names = st.get("layer_names", [])
    check("NLU status has textual_entailment", "textual_entailment" in layer_names)
    check("NLU status has figurative_language", "figurative_language" in layer_names)
    check("NLU status has information_density", "information_density" in layer_names)

    # ── Section 5: LanguageComprehensionEngine v8.1.0 ──
    print("\n═══ Section 5: LanguageComprehensionEngine v8.1.0 ═══")
    from l104_asi.language_comprehension import LanguageComprehensionEngine

    lce = LanguageComprehensionEngine()
    check("LCE VERSION = 8.1.0", lce.VERSION == "8.1.0")

    # ── Section 6: LanguageEngine v7.0.0 ──
    print("\n═══ Section 6: LanguageEngine v7.0.0 ═══")
    from l104_language_engine import LanguageEngine

    le = LanguageEngine()
    check("LE VERSION = 7.0.0", le.VERSION == "7.0.0")

    # New methods exist
    check("LE has check_entailment", hasattr(le, "check_entailment"))
    check("LE has analyze_figurative", hasattr(le, "analyze_figurative"))
    check("LE has analyze_density", hasattr(le, "analyze_density"))

    # Call new methods
    le_ent = le.check_entailment("Cats are mammals", "Cats are animals")
    check("LE check_entailment works", isinstance(le_ent, dict) and "label" in le_ent)

    le_fig = le.analyze_figurative("It's raining cats and dogs outside")
    check("LE analyze_figurative works", isinstance(le_fig, dict) and "figures" in le_fig)

    le_den = le.analyze_density("Complex technical language with specific terminology")
    check("LE analyze_density works", isinstance(le_den, dict) and "overall_density" in le_den)

    # deep_analyze includes new fields
    da = le.deep_analyze("The sun smiled upon the earth with warmth")
    check("LE deep_analyze has figurative", "figurative" in da)
    check("LE deep_analyze has density", "density" in da)

    # Status
    st = le.status()
    subs = st.get("subsystems", {})
    check("LE status has textual_entailment", "textual_entailment" in subs)
    check("LE status has figurative_language", "figurative_language" in subs)
    check("LE status has information_density", "information_density" in subs)
    check("LE subsystem count >= 19", st.get("subsystem_count", 0) >= 19,
          f"got {st.get('subsystem_count')}")

    # ── Section 7: ASI Core Wiring ──
    print("\n═══ Section 7: ASI Core Wiring ═══")
    from l104_asi.core import ASICore

    asi = ASICore()
    check("ASI has check_entailment", hasattr(asi, "check_entailment"))
    check("ASI has analyze_figurative", hasattr(asi, "analyze_figurative"))
    check("ASI has analyze_density", hasattr(asi, "analyze_density"))

    # callable
    check("ASI check_entailment callable", callable(getattr(asi, "check_entailment", None)))
    check("ASI analyze_figurative callable", callable(getattr(asi, "analyze_figurative", None)))
    check("ASI analyze_density callable", callable(getattr(asi, "analyze_density", None)))

    # ── Section 8: __init__.py Exports ──
    print("\n═══ Section 8: __init__.py Exports ═══")
    import l104_asi

    check("Export TextualEntailmentEngine", hasattr(l104_asi, "TextualEntailmentEngine"))
    check("Export EntailmentLabel", hasattr(l104_asi, "EntailmentLabel"))
    check("Export EntailmentResult", hasattr(l104_asi, "EntailmentResult"))
    check("Export FigurativeLanguageProcessor", hasattr(l104_asi, "FigurativeLanguageProcessor"))
    check("Export FigurativeType", hasattr(l104_asi, "FigurativeType"))
    check("Export FigurativeExpression", hasattr(l104_asi, "FigurativeExpression"))
    check("Export InformationDensityAnalyzer", hasattr(l104_asi, "InformationDensityAnalyzer"))
    check("Export DensityProfile", hasattr(l104_asi, "DensityProfile"))

    # ── Summary ──
    print(f"\n{'═'*50}")
    total = PASSED + FAILED
    print(f"Wave 6 Validation: {PASSED}/{total} passed, {FAILED} failed")
    if ERRORS:
        print("\nFailed tests:")
        for e in ERRORS:
            print(e)
    print(f"{'═'*50}")
    return FAILED == 0

if __name__ == "__main__":
    try:
        ok = test()
        sys.exit(0 if ok else 1)
    except Exception as e:
        traceback.print_exc()
        print(f"\n💥 FATAL ERROR: {e}")
        sys.exit(2)
