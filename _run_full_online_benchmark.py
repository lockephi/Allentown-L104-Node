#!/usr/bin/env python3
"""
L104 Full Online Benchmark — 1000+ Questions
═══════════════════════════════════════════════════════════════════════════════
Fetches real benchmark data from HuggingFace and runs the L104 ASI benchmark
harness with expanded question counts:

  • MMLU:      1000 questions  (cais/mmlu — Hendrycks et al., ICLR 2021)
  • ARC:       1000+ questions (allenai/ai2_arc — challenge + easy)
  • HumanEval: 164 problems    (openai/openai_humaneval — full dataset)
  • MATH:      120+ problems   (expanded hardcoded — Hendrycks et al., NeurIPS 2021)
  ─────────────────────────────────────────────────────────────────────────────
  TOTAL:       2,280+ questions

Run:  .venv/bin/python _run_full_online_benchmark.py
GOD_CODE: 527.5184818492612
"""

import os, sys, json, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_asi.benchmark_harness import BenchmarkHarness

SEP = "═" * 72

print(f"""
{SEP}
  L104 SOVEREIGN NODE — FULL ONLINE BENCHMARK (1000+ Questions)
{SEP}
  Sources: HuggingFace Datasets API (peer-reviewed academic benchmarks)
  MMLU: 1000q | ARC: 1000+q | HumanEval: 164p | MATH: 120+p
{SEP}
""", flush=True)

t0 = time.time()

h = BenchmarkHarness()
# Run with expanded counts: 1000 MMLU + 500 ARC per config (challenge+easy=1000+)
results = h.run_all(online=True, mmlu_count=1000, arc_count=500)

elapsed = time.time() - t0

# ── Print results ──
print(f"\n{SEP}", flush=True)
print(f"  BENCHMARK RESULTS — {elapsed:.1f}s elapsed", flush=True)
print(f"{SEP}", flush=True)

# Sources
sources = results.get("sources", {})
if sources:
    print("\n  Data Sources:", flush=True)
    for name, src in sources.items():
        print(f"    {name}: {src}", flush=True)

# Scores
bm = results.get("benchmarks", {})
total_questions = 0
total_correct = 0

print(f"\n  {'Benchmark':>12}  {'Score':>8}  {'Correct':>8}  {'Total':>8}", flush=True)
print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}", flush=True)

for k, v in bm.items():
    if isinstance(v, dict):
        score = v.get("score", 0.0)
        total = v.get("total", 0)
        correct = v.get("correct", v.get("passed", 0))
        err = v.get("error")
        total_questions += total
        total_correct += correct
        if err:
            print(f"  {k:>12}: ERROR — {err}", flush=True)
        else:
            print(f"  {k:>12}  {score*100:>7.1f}%  {correct:>8}  {total:>8}", flush=True)

print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}", flush=True)
overall_pct = (total_correct / max(total_questions, 1)) * 100
print(f"  {'TOTAL':>12}  {overall_pct:>7.1f}%  {total_correct:>8}  {total_questions:>8}", flush=True)

composite = results.get("composite_score", 0)
verdict = results.get("verdict", "?")
god_code_bonus = results.get("god_code_bonus", 0)

print(f"""
  COMPOSITE (weighted):  {composite*100:.1f}%
  GOD_CODE bonus:        {god_code_bonus*100:.4f}%
  VERDICT:               {verdict}
  Total Questions:       {total_questions}
  Elapsed:               {elapsed:.1f}s
{SEP}
""", flush=True)

# Save results
out_file = "benchmark_online_results.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Results saved to: {out_file}", flush=True)

# Also save a copy with timestamp
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
ts_file = f"/tmp/l104_benchmark_{ts}.json"
with open(ts_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Timestamped copy:  {ts_file}", flush=True)
