#!/usr/bin/env python3
"""ARC benchmark — 100 questions with progress tracking."""
import os, sys, json, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)

from l104_asi.benchmark_harness import _HuggingFaceFetcher
import importlib

print("Fetching ARC data (100 questions)...", flush=True)
data = _HuggingFaceFetcher.fetch_arc(max_questions=100)
print(f"Fetched {len(data)} questions", flush=True)

cr_mod = importlib.import_module("l104_asi.commonsense_reasoning")
engine = cr_mod.CommonsenseReasoningEngine()

correct = 0
total = 0
wrong_samples = []
t0 = time.time()

for item in data:
    q = item["question"]
    choices = item["choices"]
    answer_idx = item["answer"]
    result = engine.answer_mcq(q, choices)
    sel = result.get("selected_index", result.get("answer_index", -1))
    is_correct = (sel == answer_idx)
    if is_correct:
        correct += 1
    else:
        if len(wrong_samples) < 8:
            wrong_samples.append({
                "q": q[:100],
                "correct": choices[answer_idx] if answer_idx < len(choices) else "?",
                "picked": choices[sel] if 0 <= sel < len(choices) else "?",
            })
    total += 1
    if total % 10 == 0:
        elapsed = time.time() - t0
        rate = total / elapsed
        eta = (len(data) - total) / rate if rate > 0 else 0
        print(f"  [{total}/{len(data)}] {correct}/{total} = {correct/total*100:.1f}% | {elapsed:.0f}s elapsed | ETA {eta:.0f}s", flush=True)

elapsed = time.time() - t0
print(f"\nARC RESULT: {correct}/{total} = {correct/total*100:.1f}% in {elapsed:.1f}s", flush=True)
print(f"\nSample wrong answers:", flush=True)
for w in wrong_samples[:5]:
    print(f"  Q: {w['q']}")
    print(f"  Correct: {w['correct']}, Picked: {w['picked']}")
    print()

with open("/tmp/arc_result.json", "w") as f:
    json.dump({"correct": correct, "total": total, "pct": round(correct/total*100, 1), "wrong": wrong_samples, "elapsed_s": round(elapsed, 1)}, f, indent=2)
print("Result saved to /tmp/arc_result.json", flush=True)
