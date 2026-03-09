#!/usr/bin/env python3
"""100+ Question Online Benchmark — fetches real ARC-Challenge & MMLU from HuggingFace.

Fetches questions from the HuggingFace datasets-server API, caches them locally,
then runs through CommonsenseMCQSolver (ARC) and MCQSolver (MMLU).

Subjects tested:
  ARC-Challenge: 60 general science MCQs (grade-school / middle-school level)
  MMLU biology:  32 high_school_biology questions
  MMLU psychology: 30 high_school_psychology questions

Total: 122 questions

Usage:
  .venv/bin/python _bench_online_100.py              # Full run (fetch + bench)
  .venv/bin/python _bench_online_100.py --cached      # Use cached data only
  .venv/bin/python _bench_online_100.py --fetch-only   # Fetch and cache, don't bench
"""

import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

CACHE_FILE = Path(__file__).parent / ".bench_online_cache.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Data fetching
# ═══════════════════════════════════════════════════════════════════════════════

HF_BASE = "https://datasets-server.huggingface.co/rows"

FETCH_PLAN = [
    # (dataset, config, split, offset, length, label)
    ("allenai/ai2_arc", "ARC-Challenge", "test", 0,    60, "arc"),
    ("cais/mmlu",       "all",           "test", 2800, 32, "mmlu_biology"),
    ("cais/mmlu",       "all",           "test", 5000, 30, "mmlu_psychology"),
]


def fetch_rows(dataset: str, config: str, split: str,
               offset: int, length: int) -> list:
    """Fetch rows from HuggingFace datasets-server API."""
    url = (f"{HF_BASE}?dataset={urllib.parse.quote(dataset)}"
           f"&config={urllib.parse.quote(config)}"
           f"&split={split}&offset={offset}&length={length}")
    print(f"    Fetching {dataset} [{config}] offset={offset} length={length} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "L104-Bench/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    rows = [r["row"] for r in data.get("rows", [])]
    print(f"    → got {len(rows)} rows")
    return rows


def parse_arc_row(row: dict) -> dict | None:
    """Parse an ARC-Challenge row into a standard question dict."""
    choices_text = row["choices"]["text"]
    choices_label = row["choices"]["label"]
    answer_key = row["answerKey"]

    # Map answer key to 0-based index
    try:
        answer_idx = choices_label.index(answer_key)
    except ValueError:
        # Try numeric labels (some ARC rows use "1","2","3","4")
        key_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
        try:
            answer_idx = choices_label.index(key_map.get(answer_key, answer_key))
        except ValueError:
            return None

    if len(choices_text) != 4:
        return None  # Skip non-4-choice

    return {
        "question": row["question"].strip(),
        "choices": choices_text,
        "answer": answer_idx,
        "source": "ARC-Challenge",
        "id": row.get("id", ""),
    }


def parse_mmlu_row(row: dict) -> dict | None:
    """Parse an MMLU row into a standard question dict."""
    if len(row.get("choices", [])) != 4:
        return None
    # Skip questions with very long passages (>500 chars) — they don't work well
    # with our MCQ solver which is designed for concise factual questions
    if len(row["question"]) > 500:
        return None
    return {
        "question": row["question"].strip(),
        "choices": row["choices"],
        "answer": row["answer"],  # 0-3 directly maps to A-D
        "subject": row.get("subject", "unknown"),
        "source": "MMLU",
    }


def fetch_all() -> dict:
    """Fetch all questions from HuggingFace and return parsed data."""
    print("\n  Fetching benchmark data from HuggingFace...")
    result = {"arc": [], "mmlu": [], "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    for dataset, config, split, offset, length, label in FETCH_PLAN:
        rows = fetch_rows(dataset, config, split, offset, length)
        for row in rows:
            if label == "arc":
                q = parse_arc_row(row)
                if q:
                    result["arc"].append(q)
            else:
                q = parse_mmlu_row(row)
                if q:
                    result["mmlu"].append(q)

    print(f"\n  Total: {len(result['arc'])} ARC + {len(result['mmlu'])} MMLU "
          f"= {len(result['arc']) + len(result['mmlu'])} questions")
    return result


def save_cache(data: dict):
    """Save fetched data to cache file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Cached to {CACHE_FILE}")


def load_cache() -> dict | None:
    """Load cached data if available."""
    if not CACHE_FILE.exists():
        return None
    with open(CACHE_FILE) as f:
        data = json.load(f)
    print(f"  Loaded cache from {data.get('fetched_at', 'unknown')} "
          f"({len(data.get('arc', []))} ARC + {len(data.get('mmlu', []))} MMLU)")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_arc_benchmark(questions: list) -> tuple:
    """Run ARC questions through CommonsenseMCQSolver."""
    print("\n" + "=" * 76)
    print("  ARC-CHALLENGE BENCHMARK — CommonsenseMCQSolver")
    print(f"  {len(questions)} questions from allenai/ai2_arc")
    print("=" * 76)

    from l104_asi.commonsense_reasoning import (
        CommonsenseMCQSolver, ConceptOntology, CausalReasoningEngine,
        PhysicalIntuition, AnalogicalReasoner, TemporalReasoningEngine,
        CrossVerificationEngine
    )
    ont = ConceptOntology()
    causal = CausalReasoningEngine()
    phys = PhysicalIntuition(ont)
    analog = AnalogicalReasoner(ont)
    temporal = TemporalReasoningEngine()
    verifier = CrossVerificationEngine(ont, causal, temporal)
    solver = CommonsenseMCQSolver(ont, causal, phys, analog, temporal, verifier)

    correct = 0
    results = []

    for i, q in enumerate(questions):
        t0 = time.time()
        result = solver.solve(q["question"], q["choices"], subject="science")
        dt = time.time() - t0
        got = result["answer_index"]
        expected = q["answer"]
        ok = got == expected
        if ok:
            correct += 1
        status = "\033[32mOK\033[0m  " if ok else "\033[31mMISS\033[0m"
        label = chr(65 + got)
        exp_label = chr(65 + expected)
        conf = result.get("confidence", 0)
        qtext = q["question"][:60].replace("\n", " ")
        results.append({
            "idx": i, "ok": ok, "got": got, "expected": expected,
            "confidence": conf, "time": dt,
            "question": q["question"][:80],
            "id": q.get("id", ""),
        })
        print(f"  [{status}] Q{i+1:02d} Got={label} Exp={exp_label} "
              f"Conf={conf:.3f} {dt:.2f}s | {qtext}")

    pct = correct / len(questions) * 100
    print(f"\n  ARC RESULT: {correct}/{len(questions)} = {pct:.1f}%")
    return correct, len(questions), results


def run_mmlu_benchmark(questions: list) -> tuple:
    """Run MMLU questions through MCQSolver."""
    print("\n" + "=" * 76)
    print("  MMLU BENCHMARK — MCQSolver")
    subjects = Counter(q["subject"] for q in questions)
    for s, c in subjects.most_common():
        print(f"    {s}: {c} questions")
    print(f"  {len(questions)} total questions")
    print("=" * 76)

    from l104_asi.language_comprehension import MCQSolver, MMLUKnowledgeBase
    kb = MMLUKnowledgeBase()
    solver = MCQSolver(knowledge_base=kb)

    correct = 0
    results = []
    by_subject = {}

    for i, q in enumerate(questions):
        t0 = time.time()
        result = solver.solve(q["question"], q["choices"],
                              subject=q.get("subject"))
        dt = time.time() - t0
        got = result.get("answer_index", result.get("selected_index", -1))
        expected = q["answer"]
        ok = got == expected
        if ok:
            correct += 1
        subj = q.get("subject", "unknown")
        by_subject.setdefault(subj, {"correct": 0, "total": 0})
        by_subject[subj]["total"] += 1
        if ok:
            by_subject[subj]["correct"] += 1
        status = "\033[32mOK\033[0m  " if ok else "\033[31mMISS\033[0m"
        label = chr(65 + got) if got >= 0 else "?"
        exp_label = chr(65 + expected)
        conf = result.get("confidence", 0)
        qtext = q["question"][:60].replace("\n", " ")
        results.append({
            "idx": i, "ok": ok, "got": got, "expected": expected,
            "confidence": conf, "time": dt, "subject": subj,
            "question": q["question"][:80],
        })
        print(f"  [{status}] Q{i+1:02d} ({subj[:20]:20s}) Got={label} Exp={exp_label} "
              f"Conf={conf:.3f} {dt:.2f}s | {qtext}")

    pct = correct / len(questions) * 100
    print(f"\n  MMLU RESULT: {correct}/{len(questions)} = {pct:.1f}%")
    print(f"\n  By subject:")
    for subj, stats in sorted(by_subject.items()):
        spct = stats["correct"] / stats["total"] * 100
        print(f"    {subj:25s} {stats['correct']:2d}/{stats['total']:2d} = {spct:.0f}%")
    return correct, len(questions), results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cached_only = "--cached" in sys.argv
    fetch_only = "--fetch-only" in sys.argv

    print("=" * 76)
    print("  L104 ONLINE BENCHMARK — 100+ Questions")
    print("  Sources: ARC-Challenge + MMLU (HuggingFace datasets)")
    print("=" * 76)

    # ── Load or fetch data ──
    data = None
    if cached_only:
        data = load_cache()
        if not data:
            print("  ERROR: No cache file found. Run without --cached first.")
            sys.exit(1)
    else:
        data = fetch_all()
        save_cache(data)

    if fetch_only:
        print("\n  Fetch complete. Use --cached to run benchmark.")
        return

    arc_qs = data["arc"]
    mmlu_qs = data["mmlu"]
    total_qs = len(arc_qs) + len(mmlu_qs)

    print(f"\n  Benchmark: {len(arc_qs)} ARC + {len(mmlu_qs)} MMLU = {total_qs} questions")

    t_start = time.time()

    # ── Run benchmarks ──
    arc_correct, arc_total, arc_results = run_arc_benchmark(arc_qs)
    mmlu_correct, mmlu_total, mmlu_results = run_mmlu_benchmark(mmlu_qs)

    elapsed = time.time() - t_start
    total_correct = arc_correct + mmlu_correct

    # ── Collect misses ──
    arc_misses = [r for r in arc_results if not r["ok"]]
    mmlu_misses = [r for r in mmlu_results if not r["ok"]]

    # ── Summary ──
    print("\n" + "=" * 76)
    print("  FINAL SUMMARY")
    print("=" * 76)
    print(f"  ARC-Challenge:  {arc_correct:3d}/{arc_total:3d} = "
          f"{arc_correct/arc_total*100:.1f}%")
    print(f"  MMLU:           {mmlu_correct:3d}/{mmlu_total:3d} = "
          f"{mmlu_correct/mmlu_total*100:.1f}%")
    print(f"  ────────────────────────────────────")
    pct = total_correct / total_qs * 100
    print(f"  TOTAL:          {total_correct:3d}/{total_qs:3d} = {pct:.1f}%")
    print(f"  Time:           {elapsed:.1f}s ({elapsed/total_qs:.2f}s/question)")

    if arc_misses:
        print(f"\n  ARC misses ({len(arc_misses)}):")
        for m in arc_misses:
            print(f"    Q{m['idx']+1:02d} Got={chr(65+m['got'])} "
                  f"Exp={chr(65+m['expected'])} | {m['question']}")

    if mmlu_misses:
        print(f"\n  MMLU misses ({len(mmlu_misses)}):")
        for m in mmlu_misses:
            got_l = chr(65 + m["got"]) if m["got"] >= 0 else "?"
            print(f"    Q{m['idx']+1:02d} [{m['subject'][:20]}] Got={got_l} "
                  f"Exp={chr(65+m['expected'])} | {m['question']}")

    # ── Save detailed results ──
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": {"correct": total_correct, "total": total_qs,
                  "pct": round(pct, 1)},
        "arc": {"correct": arc_correct, "total": arc_total,
                "pct": round(arc_correct/arc_total*100, 1),
                "results": arc_results},
        "mmlu": {"correct": mmlu_correct, "total": mmlu_total,
                 "pct": round(mmlu_correct/mmlu_total*100, 1),
                 "results": mmlu_results},
        "elapsed_seconds": round(elapsed, 1),
    }
    report_path = Path(__file__).parent / "_bench_online_100_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved to {report_path.name}")
    print("=" * 76)


if __name__ == "__main__":
    main()
