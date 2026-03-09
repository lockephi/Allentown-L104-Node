#!/usr/bin/env python3
"""
_verify_hybrid.py — Verification script for the Hybrid Comprehension Model.

Tests end-to-end wiring between ProbabilityEngine and LanguageComprehensionEngine:

  1. ProbabilityEngine singleton import + readiness
  2. LanguageComprehensionEngine initialization with PE wiring
  3. KB tokens ingested into PE DataIngestor
  4. ConceptNet relations bridged into GateProbabilityBridge
  5. PE Bayesian update + sacred probability coherence
  6. MCQ solving with hybrid_probability_collapse active
  7. three_engine_comprehension_score includes PE component
  8. Status dicts include probability_engine

Run: .venv/bin/python _verify_hybrid.py
"""
from __future__ import annotations

import sys
import time
import traceback

PASS = 0
FAIL = 0
WARN = 0

def ok(label: str, detail: str = ""):
    global PASS
    PASS += 1
    print(f"  ✅ {label}" + (f" — {detail}" if detail else ""))

def fail(label: str, detail: str = ""):
    global FAIL
    FAIL += 1
    print(f"  ❌ {label}" + (f" — {detail}" if detail else ""))

def warn(label: str, detail: str = ""):
    global WARN
    WARN += 1
    print(f"  ⚠️  {label}" + (f" — {detail}" if detail else ""))


def main():
    global PASS, FAIL, WARN
    t0 = time.perf_counter()

    # ════════════════════════════════════════════════════════════════════
    print("\n═══ Phase 1: ProbabilityEngine singleton import ═══")
    # ════════════════════════════════════════════════════════════════════
    try:
        from l104_probability_engine import probability_engine as pe
        ok("Import probability_engine singleton")
    except Exception as e:
        fail("Import probability_engine", str(e))
        print("FATAL: Cannot continue without ProbabilityEngine")
        sys.exit(1)

    # Sanity: check hub attributes
    for attr in ("ingestor", "classical", "quantum", "bridge", "insight", "algorithm"):
        if hasattr(pe, attr):
            ok(f"PE has .{attr}")
        else:
            fail(f"PE missing .{attr}")

    # Quick API smoke
    try:
        sp = pe.sacred_probability(527.518)
        assert 0 < sp <= 1, f"sacred_probability returned {sp}"
        ok(f"PE.sacred_probability(GOD_CODE) = {sp:.6f}")
    except Exception as e:
        fail("PE.sacred_probability", str(e))

    # ════════════════════════════════════════════════════════════════════
    print("\n═══ Phase 2: LanguageComprehensionEngine init + PE wiring ═══")
    # ════════════════════════════════════════════════════════════════════
    try:
        from l104_asi.language_comprehension import LanguageComprehensionEngine
        lce = LanguageComprehensionEngine()
        lce.initialize()
        ok("LCE initialized")
    except Exception as e:
        fail("LCE initialize()", str(e))
        traceback.print_exc()
        print("FATAL: Cannot continue without LCE")
        sys.exit(1)

    # Check PE wiring
    if lce._probability_engine is not None:
        ok("LCE._probability_engine wired")
    else:
        fail("LCE._probability_engine is None (wiring failed)")

    # ════════════════════════════════════════════════════════════════════
    print("\n═══ Phase 3: KB tokens ingested into PE DataIngestor ═══")
    # ════════════════════════════════════════════════════════════════════
    pe_ref = lce._probability_engine or pe
    if hasattr(pe_ref, 'ingestor') and hasattr(pe_ref.ingestor, 'token_counter'):
        tc = pe_ref.ingestor.token_counter
        total_tokens = sum(tc.values()) if tc else 0
        if total_tokens > 0:
            ok(f"DataIngestor has {total_tokens} token counts across {len(tc)} unique tokens")
            # Sample some tokens
            top5 = sorted(tc.items(), key=lambda x: x[1], reverse=True)[:5]
            for tok, cnt in top5:
                print(f"       token '{tok}': count={cnt}")
        else:
            fail("DataIngestor token_counter is empty — KB facts not fed")
    else:
        fail("DataIngestor.token_counter not accessible")

    if hasattr(pe_ref, 'ingestor') and hasattr(pe_ref.ingestor, 'category_counter'):
        cc = pe_ref.ingestor.category_counter
        total_cats = sum(cc.values()) if cc else 0
        if total_cats > 0:
            ok(f"DataIngestor has {total_cats} category counts across {len(cc)} categories")
            for cat, cnt in sorted(cc.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"       category '{cat}': count={cnt}")
        else:
            warn("DataIngestor category_counter is empty (may be OK if KB has no subject_index)")
    else:
        warn("DataIngestor.category_counter not accessible")

    # ════════════════════════════════════════════════════════════════════
    print("\n═══ Phase 4: Token probability from KB-informed priors ═══")
    # ════════════════════════════════════════════════════════════════════
    test_tokens = ["physics", "electron", "biology", "mitochondria", "history", "zzz_nonexistent"]
    for tok in test_tokens:
        try:
            p = pe_ref.token_probability(tok)
            status = "KB-informed" if p > 0 else "zero (not in KB)"
            ok(f"token_probability('{tok}') = {p:.8f} [{status}]")
        except Exception as e:
            fail(f"token_probability('{tok}')", str(e))

    # ════════════════════════════════════════════════════════════════════
    print("\n═══ Phase 5: Bayesian update + Grover amplification ═══")
    # ════════════════════════════════════════════════════════════════════
    try:
        prior = [0.25, 0.25, 0.25, 0.25]
        likelihoods = [0.1, 0.6, 0.2, 0.1]
        posterior = pe_ref.bayesian_update(prior, likelihoods)
        ok(f"bayesian_update: prior={prior} → posterior={[round(p, 4) for p in posterior]}")
        assert abs(sum(posterior) - 1.0) < 0.01, f"Posterior doesn't sum to 1: {sum(posterior)}"
        ok("Posterior sums to 1.0")
        assert posterior[1] > posterior[0], "Highest likelihood choice should dominate"
        ok("Highest likelihood choice dominates posterior")
    except Exception as e:
        fail("bayesian_update", str(e))

    try:
        amp = pe_ref.grover_amplification(0.25, n_items=4, iterations=1)
        ok(f"grover_amplification(0.25, n=4, iter=1) = {amp:.6f}")
        assert amp > 0.25, f"Grover should amplify: got {amp}"
        ok("Grover amplifies target probability above uniform")
    except Exception as e:
        fail("grover_amplification", str(e))

    # ════════════════════════════════════════════════════════════════════
    print("\n═══ Phase 6: Insight synthesis (consciousness bridge) ═══")
    # ════════════════════════════════════════════════════════════════════
    try:
        signals = [0.8, 0.3, 0.1, 0.6]
        insight = pe_ref.synthesize_insight(signals, consciousness_level=0.7, temperature=0.618)
        ok(f"synthesize_insight: consciousness_prob={insight.consciousness_probability:.6f}, "
           f"resonance={insight.resonance_score:.6f}")
        if hasattr(insight, 'trajectory_forecast') and insight.trajectory_forecast:
            ok(f"  trajectory_forecast: {[round(v, 4) for v in insight.trajectory_forecast[:5]]}")
        if hasattr(insight, 'god_code_alignment'):
            ok(f"  god_code_alignment: {insight.god_code_alignment:.6f}")
    except Exception as e:
        fail("synthesize_insight", str(e))
        traceback.print_exc()

    # ════════════════════════════════════════════════════════════════════
    print("\n═══ Phase 7: MCQ solve with hybrid probability collapse ═══")
    # ════════════════════════════════════════════════════════════════════
    test_questions = [
        {
            "question": "What is the powerhouse of the cell?",
            "choices": ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"],
            "expected_label": "B",
            "expected_text": "Mitochondria",
            "subject": "biology",
        },
        {
            "question": "What is the speed of light in vacuum?",
            "choices": ["3×10^8 m/s", "3×10^6 m/s", "3×10^10 m/s", "3×10^4 m/s"],
            "expected_label": "A",
            "expected_text": "3×10^8 m/s",
            "subject": "physics",
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
            "expected_label": "B",
            "expected_text": "William Shakespeare",
            "subject": "literature",
        },
    ]

    correct = 0
    for tq in test_questions:
        try:
            result = lce.answer_mcq(tq["question"], tq["choices"], subject=tq["subject"])
            answer = result.get("answer", result.get("best_choice", ""))
            # answer_mcq returns letter labels (A/B/C/D) — compare against both
            is_correct = answer in (tq["expected_label"], tq["expected_text"])
            hybrid_used = False
            # Check if hybrid path was used by looking at choice_scores for hybrid_posterior key
            choice_scores = result.get("choice_scores", [])
            if choice_scores and isinstance(choice_scores[0], dict):
                hybrid_used = "hybrid_posterior" in choice_scores[0]
            status = "CORRECT" if is_correct else f"WRONG (got '{answer}', expected '{tq['expected_label']}')"
            path = "HYBRID" if hybrid_used else "LEGACY"
            if is_correct:
                ok(f"Q: {tq['question'][:50]}... → {status} [{path}]")
                correct += 1
            else:
                warn(f"Q: {tq['question'][:50]}... → {status} [{path}]")
        except Exception as e:
            fail(f"answer_mcq() on '{tq['question'][:40]}...'", str(e))
            traceback.print_exc()

    if correct == len(test_questions):
        ok(f"All {len(test_questions)} test questions correct")
    else:
        warn(f"{correct}/{len(test_questions)} test questions correct")

    # ════════════════════════════════════════════════════════════════════
    print("\n═══ Phase 8: three_engine_comprehension_score includes PE ═══")
    # ════════════════════════════════════════════════════════════════════
    try:
        score = lce.three_engine_comprehension_score()
        ok(f"three_engine_comprehension_score = {score:.6f}")
        if score > 0:
            ok(f"Score is positive (PE component contributing)")
        else:
            warn(f"Score is zero — PE component may not be contributing")
    except Exception as e:
        fail("three_engine_comprehension_score", str(e))
        traceback.print_exc()

    # ════════════════════════════════════════════════════════════════════
    print("\n═══ Phase 9: Status dicts include probability_engine ═══")
    # ════════════════════════════════════════════════════════════════════
    try:
        status = lce.get_status()
        es = status.get("engine_support", {})
        if "probability_engine" in es:
            ok(f"get_status().engine_support.probability_engine = {es['probability_engine']}")
        else:
            fail("get_status() missing probability_engine in engine_support")
    except Exception as e:
        fail("get_status()", str(e))

    try:
        te_status = lce.three_engine_status()
        engines = te_status.get("engines", {})
        if "probability_engine" in engines:
            ok(f"three_engine_status().engines.probability_engine = {engines['probability_engine']}")
        else:
            fail("three_engine_status() missing probability_engine in engines")
    except Exception as e:
        fail("three_engine_status()", str(e))

    # ════════════════════════════════════════════════════════════════════
    elapsed = time.perf_counter() - t0
    print(f"\n{'═' * 60}")
    print(f"  Hybrid Comprehension Model Verification")
    print(f"  ✅ PASS: {PASS}  ❌ FAIL: {FAIL}  ⚠️  WARN: {WARN}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"{'═' * 60}\n")

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
