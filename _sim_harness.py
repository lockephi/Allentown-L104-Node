#!/usr/bin/env python3
"""
L104 Commonsense Reasoning Simulation Harness

Tests proposed changes against the FULL question set BEFORE committing.
Provides:
  1. Before/after comparison for every question
  2. Regression detection (was right, now wrong)
  3. Fix detection (was wrong, now right)
  4. Net impact summary

Usage:
    # Run full audit on a range
    python _sim_harness.py 0 450

    # Run quick audit on specific offsets
    python _sim_harness.py 0 50 100 150 200 250 300 350 400

    # Run with detailed mode (shows every question)
    python _sim_harness.py --detail 0 50
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset

def load_questions(offsets, batch_size=50):
    """Load ARC-Easy questions at given offsets."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    questions = []
    for offset in offsets:
        for i in range(batch_size):
            idx = offset + i
            if idx >= len(ds):
                break
            item = ds[idx]
            q_text = item["question"]
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            correct = item["answerKey"]
            choices = dict(zip(labels, texts))
            questions.append({
                "idx": idx,
                "question": q_text,
                "choices": choices,
                "correct": correct,
            })
    return questions


def run_engine_on_questions(questions, engine=None):
    """Run the commonsense reasoning engine on all questions.
    Returns dict mapping idx → {predicted, correct, right, question, choices}
    """
    if engine is None:
        from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
        engine = CommonsenseReasoningEngine()

    results = {}
    for i, q in enumerate(questions):
        try:
            # Build the MCQ query
            formatted = q["question"] + "\n"
            for label in sorted(q["choices"].keys()):
                formatted += f"{label}. {q['choices'][label]}\n"

            result = engine.solve_mcq(formatted)
            predicted = result.get("answer", "?")
            if len(predicted) > 1:
                predicted = predicted[0]

            correct = q["correct"]
            is_right = predicted == correct

            results[q["idx"]] = {
                "predicted": predicted,
                "correct": correct,
                "right": is_right,
                "question": q["question"][:80],
                "choices": q["choices"],
            }
        except Exception as e:
            results[q["idx"]] = {
                "predicted": "ERR",
                "correct": q["correct"],
                "right": False,
                "question": q["question"][:80],
                "error": str(e),
            }

        # Progress
        if (i + 1) % 10 == 0:
            right_so_far = sum(1 for r in results.values() if r["right"])
            print(f"  [{i+1}/{len(questions)}] {right_so_far}/{i+1} correct "
                  f"({100*right_so_far/(i+1):.1f}%)")

    return results


def compare_runs(baseline: dict, current: dict) -> dict:
    """Compare baseline vs current results. Returns analysis."""
    regressions = []  # Was right, now wrong
    fixes = []        # Was wrong, now right
    stable_right = 0
    stable_wrong = 0

    all_idxs = sorted(set(baseline.keys()) | set(current.keys()))

    for idx in all_idxs:
        b = baseline.get(idx, {})
        c = current.get(idx, {})

        b_right = b.get("right", False)
        c_right = c.get("right", False)

        if b_right and not c_right:
            regressions.append({
                "idx": idx,
                "question": c.get("question", "?"),
                "was": b.get("predicted", "?"),
                "now": c.get("predicted", "?"),
                "correct": c.get("correct", "?"),
            })
        elif not b_right and c_right:
            fixes.append({
                "idx": idx,
                "question": c.get("question", "?"),
                "was": b.get("predicted", "?"),
                "now": c.get("predicted", "?"),
                "correct": c.get("correct", "?"),
            })
        elif b_right:
            stable_right += 1
        else:
            stable_wrong += 1

    baseline_total = sum(1 for r in baseline.values() if r["right"])
    current_total = sum(1 for r in current.values() if r["right"])

    return {
        "baseline_score": baseline_total,
        "current_score": current_total,
        "total_questions": len(all_idxs),
        "baseline_pct": 100 * baseline_total / len(all_idxs) if all_idxs else 0,
        "current_pct": 100 * current_total / len(all_idxs) if all_idxs else 0,
        "net_change": current_total - baseline_total,
        "regressions": regressions,
        "fixes": fixes,
        "stable_right": stable_right,
        "stable_wrong": stable_wrong,
    }


def save_baseline(results: dict, filename: str = "_sim_baseline.json"):
    """Save results as baseline for future comparisons."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline saved: {filename} ({len(results)} questions)")


def load_baseline(filename: str = "_sim_baseline.json") -> dict:
    """Load baseline results."""
    if not os.path.exists(filename):
        return {}
    with open(filename) as f:
        data = json.load(f)
    # Convert string keys to int
    return {int(k): v for k, v in data.items()}


def main():
    args = sys.argv[1:]
    detail = "--detail" in args
    save_as_baseline = "--save-baseline" in args
    compare_to_baseline = "--compare" in args
    args = [a for a in args if not a.startswith("--")]

    if not args:
        print("Usage: python _sim_harness.py [--detail] [--save-baseline] [--compare] offset1 [offset2 ...]")
        print("  Or:  python _sim_harness.py 0 450  (for range)")
        sys.exit(1)

    offsets = [int(a) for a in args]

    # If two args and second is much larger, treat as range
    if len(offsets) == 2 and offsets[1] - offsets[0] >= 50:
        start, end = offsets
        offsets = list(range(start, end, 50))

    print(f"{'='*60}")
    print(f"L104 Simulation Harness — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Testing offsets: {offsets}")
    print(f"{'='*60}")

    # Load questions
    t0 = time.time()
    questions = load_questions(offsets)
    print(f"Loaded {len(questions)} questions in {time.time()-t0:.1f}s")

    # Run engine
    print(f"\nRunning commonsense reasoning engine...")
    t0 = time.time()
    results = run_engine_on_questions(questions)
    elapsed = time.time() - t0

    # Summary
    right = sum(1 for r in results.values() if r["right"])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {right}/{total} ({100*right/total:.1f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.2f}s/question)")
    print(f"{'='*60}")

    # Per-offset breakdown
    print(f"\nPer-offset breakdown:")
    for offset in offsets:
        batch = {k: v for k, v in results.items() if offset <= k < offset + 50}
        batch_right = sum(1 for r in batch.values() if r["right"])
        batch_total = len(batch)
        if batch_total > 0:
            print(f"  Offset {offset:4d}: {batch_right}/{batch_total} ({100*batch_right/batch_total:.1f}%)")

    # Show failures
    failures = {k: v for k, v in results.items() if not v["right"]}
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for idx in sorted(failures.keys()):
            f = failures[idx]
            print(f"  [{idx}] pred={f['predicted']} correct={f['correct']} — {f['question']}")

    # Save baseline if requested
    if save_as_baseline:
        save_baseline(results)

    # Compare to baseline if requested or if baseline exists
    if compare_to_baseline or os.path.exists("_sim_baseline.json"):
        baseline = load_baseline()
        if baseline:
            comparison = compare_runs(baseline, results)
            print(f"\n{'='*60}")
            print(f"COMPARISON TO BASELINE")
            print(f"{'='*60}")
            print(f"Baseline: {comparison['baseline_score']}/{comparison['total_questions']} ({comparison['baseline_pct']:.1f}%)")
            print(f"Current:  {comparison['current_score']}/{comparison['total_questions']} ({comparison['current_pct']:.1f}%)")
            print(f"Net:      {comparison['net_change']:+d}")

            if comparison['regressions']:
                print(f"\n⚠️  REGRESSIONS ({len(comparison['regressions'])}):")
                for r in comparison['regressions']:
                    print(f"  [{r['idx']}] {r['was']}→{r['now']} (correct={r['correct']}) — {r['question']}")

            if comparison['fixes']:
                print(f"\n✅ FIXES ({len(comparison['fixes'])}):")
                for r in comparison['fixes']:
                    print(f"  [{r['idx']}] {r['was']}→{r['now']} (correct={r['correct']}) — {r['question']}")

    # Detail mode
    if detail:
        print(f"\n{'='*60}")
        print("DETAILED RESULTS")
        print(f"{'='*60}")
        for idx in sorted(results.keys()):
            r = results[idx]
            status = "✓" if r["right"] else "✗"
            print(f"  [{idx}] {status} pred={r['predicted']} correct={r['correct']} — {r['question']}")

    return right, total


if __name__ == "__main__":
    main()
