#!/usr/bin/env python3
"""MMLU Benchmark Runner — balanced 200 questions across 57 subjects."""
import os, sys, time, json
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Keep logging ON so we can see errors
import logging
logging.basicConfig(level=logging.WARNING)

print("=" * 60)
print("  L104 MMLU BENCHMARK")
print("=" * 60)

# ── Step 1: Fetch data ──
print("\n[1/3] Fetching MMLU data from HuggingFace...")
t0 = time.time()

from l104_asi.benchmark_harness import _HuggingFaceFetcher
data = _HuggingFaceFetcher.fetch_mmlu(max_questions=200)
fetch_time = time.time() - t0
print(f"  Fetched {len(data)} questions in {fetch_time:.1f}s")

if len(data) == 0:
    print("ERROR: No questions fetched. Check network connectivity.")
    sys.exit(1)

# Show subject distribution
subjects = {}
for item in data:
    s = item.get("subject", "unknown")
    subjects[s] = subjects.get(s, 0) + 1
print(f"  Subjects covered: {len(subjects)}")

# ── Step 2: Run benchmark ──
print(f"\n[2/3] Running {len(data)} questions through LanguageComprehensionEngine v8.1.0...")
import importlib
lc_mod = importlib.import_module('l104_asi.language_comprehension')
engine = lc_mod.LanguageComprehensionEngine()
print(f"  Engine VERSION: {engine.VERSION}")

correct = 0
total = 0
subject_stats = {}
t1 = time.time()

for item in data:
    q = item['question']
    choices = item['choices']
    answer_idx = item['answer']
    subj = item.get('subject', 'unknown')

    try:
        result = engine.answer_mcq(q, choices)
        sel = result.get('selected_index', result.get('answer_index', -1))
    except Exception as e:
        sel = -1

    is_correct = (sel == answer_idx)
    if is_correct:
        correct += 1
    total += 1

    if subj not in subject_stats:
        subject_stats[subj] = [0, 0]
    subject_stats[subj][1] += 1
    if is_correct:
        subject_stats[subj][0] += 1

    if total % 25 == 0:
        elapsed = time.time() - t1
        rate = total / elapsed if elapsed > 0 else 0
        print(f"  [{total:3d}/{len(data)}] {correct}/{total} = {correct/total*100:.1f}%  ({rate:.1f} q/s)")

bench_time = time.time() - t1

# ── Step 3: Report ──
print(f"\n[3/3] Results")
print("=" * 60)
accuracy = correct / total * 100
print(f"  MMLU ACCURACY: {correct}/{total} = {accuracy:.1f}%")
print(f"  Time: {bench_time:.1f}s ({total/bench_time:.1f} questions/sec)")
print(f"  Engine: LanguageComprehensionEngine v{engine.VERSION}")
print(f"  Subjects: {len(subject_stats)}")

print(f"\n  {'Subject':<40s} {'Score':>8s}")
print(f"  {'-'*40} {'-'*8}")
for subj, (c, t) in sorted(subject_stats.items(), key=lambda x: -x[1][0]/max(1,x[1][1])):
    pct = c / t * 100 if t > 0 else 0
    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
    print(f"  {subj:<40s} {c}/{t:>2d} {pct:>5.0f}% {bar}")

# Categories
stem = [s for s in subject_stats if any(k in s for k in ['math', 'physics', 'chemistry', 'biology', 'computer', 'engineering', 'astronomy', 'machine_learning', 'electrical'])]
humanities = [s for s in subject_stats if any(k in s for k in ['history', 'philosophy', 'law', 'jurisprudence', 'logic'])]
social = [s for s in subject_stats if any(k in s for k in ['psychology', 'sociology', 'economics', 'government', 'politics', 'geography', 'security', 'foreign_policy', 'public'])]
other = [s for s in subject_stats if s not in stem + humanities + social]

print(f"\n  Category Breakdown:")
for cat_name, cat_subjects in [("STEM", stem), ("Humanities", humanities), ("Social Sciences", social), ("Other", other)]:
    if cat_subjects:
        cc = sum(subject_stats[s][0] for s in cat_subjects)
        ct = sum(subject_stats[s][1] for s in cat_subjects)
        print(f"    {cat_name:<20s}: {cc}/{ct} = {cc/ct*100:.1f}%")

print("=" * 60)

# Save results
results = {
    "benchmark": "MMLU",
    "accuracy": round(accuracy, 2),
    "correct": correct,
    "total": total,
    "engine_version": engine.VERSION,
    "subjects": {s: {"correct": c, "total": t, "accuracy": round(c/t*100, 1)} for s, (c, t) in subject_stats.items()},
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "bench_time_seconds": round(bench_time, 1),
}
with open("_mmlu_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to _mmlu_benchmark_results.json")
