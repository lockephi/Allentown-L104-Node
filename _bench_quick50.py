#!/usr/bin/env python3
"""Quick MMLU+ARC benchmark with progress reporting. 50 questions each for fast iteration."""
import json, os, sys, time, re
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Suppress all logging to speed up
import logging
logging.disable(logging.CRITICAL)

print("=" * 60, flush=True)
print("  Quick MMLU+ARC Benchmark (50 each)", flush=True)
print("=" * 60, flush=True)

print("\n[INIT] Loading engines...", flush=True)
t0 = time.time()

# Redirect stderr during init
import io
_old_stderr = sys.stderr
sys.stderr = io.StringIO()

from l104_asi.benchmark_harness import BenchmarkHarness, _HuggingFaceFetcher as fetcher

h = BenchmarkHarness()
sys.stderr = _old_stderr
print(f"[INIT] Engines loaded in {time.time()-t0:.1f}s", flush=True)

# ── MMLU: 50 questions ──
print("\n[MMLU] Fetching 50 questions from HuggingFace...", flush=True)
mmlu_data = fetcher.fetch_mmlu(max_questions=50)
print(f"[MMLU] Got {len(mmlu_data)} questions", flush=True)

engine = h._mmlu._get_engine()
correct_mmlu = 0
total_mmlu = len(mmlu_data)
predicted_dist = {0: 0, 1: 0, 2: 0, 3: 0}

print("[MMLU] Evaluating...", flush=True)
for i, sample in enumerate(mmlu_data):
    try:
        result = engine.answer_mcq(sample["question"], sample["choices"],
                                   subject=sample.get("subject", "unknown"))
        predicted = result.get("selected_index", result.get("answer_index", -1))
        is_correct = predicted == sample["answer"]
    except Exception:
        predicted = -1
        is_correct = False

    if is_correct:
        correct_mmlu += 1
    if 0 <= predicted <= 3:
        predicted_dist[predicted] = predicted_dist.get(predicted, 0) + 1

    if (i + 1) % 10 == 0:
        acc = correct_mmlu / (i + 1)
        print(f"  [{i+1}/{total_mmlu}] accuracy so far: {acc*100:.1f}%  "
              f"dist: A={predicted_dist[0]} B={predicted_dist[1]} "
              f"C={predicted_dist[2]} D={predicted_dist[3]}", flush=True)

mmlu_score = correct_mmlu / max(total_mmlu, 1)
print(f"[MMLU] Final: {mmlu_score*100:.1f}% ({correct_mmlu}/{total_mmlu})", flush=True)
print(f"[MMLU] Distribution: A={predicted_dist[0]} B={predicted_dist[1]} "
      f"C={predicted_dist[2]} D={predicted_dist[3]}", flush=True)

# ── ARC: 50 questions ──
print("\n[ARC] Fetching 50 questions from HuggingFace...", flush=True)
arc_data = fetcher.fetch_arc(max_questions=25, include_easy=True)  # 25 challenge + 25 easy
if len(arc_data) > 50:
    arc_data = arc_data[:50]
print(f"[ARC] Got {len(arc_data)} questions", flush=True)

arc_engine = h._arc._get_engine()
correct_arc = 0
total_arc = len(arc_data)
arc_dist = {0: 0, 1: 0, 2: 0, 3: 0}

print("[ARC] Evaluating...", flush=True)
for i, sample in enumerate(arc_data):
    try:
        result = arc_engine.answer_mcq(sample["question"], sample["choices"])
        predicted = result.get("selected_index", result.get("answer_index", -1))
        is_correct = predicted == sample["answer"]
    except Exception:
        predicted = -1
        is_correct = False

    if is_correct:
        correct_arc += 1
    if 0 <= predicted <= 3:
        arc_dist[predicted] = arc_dist.get(predicted, 0) + 1

    if (i + 1) % 10 == 0:
        acc = correct_arc / (i + 1)
        print(f"  [{i+1}/{total_arc}] accuracy so far: {acc*100:.1f}%  "
              f"dist: A={arc_dist[0]} B={arc_dist[1]} "
              f"C={arc_dist[2]} D={arc_dist[3]}", flush=True)

arc_score = correct_arc / max(total_arc, 1)
print(f"[ARC] Final: {arc_score*100:.1f}% ({correct_arc}/{total_arc})", flush=True)
print(f"[ARC] Distribution: A={arc_dist[0]} B={arc_dist[1]} "
      f"C={arc_dist[2]} D={arc_dist[3]}", flush=True)

# ── Summary ──
print("\n" + "=" * 60, flush=True)
print(f"  MMLU: {mmlu_score*100:5.1f}%  ({correct_mmlu}/{total_mmlu})  [was 29.6%]", flush=True)
print(f"  ARC:  {arc_score*100:5.1f}%  ({correct_arc}/{total_arc})  [was 28.7%]", flush=True)
print(f"  Time: {time.time()-t0:.0f}s", flush=True)
print("=" * 60, flush=True)
