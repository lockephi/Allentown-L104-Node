#!/usr/bin/env python3
"""Run the full 1000+ question benchmark (online mode).

Fetches ~500 MMLU + ~500 ARC from HuggingFace, plus 110 MATH (expanded hardcoded)
and 164 HumanEval problems.  Saves detailed results to JSON for analysis.
"""

import json, sys, time, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from l104_asi.benchmark_harness import BenchmarkHarness

def main():
    print("=" * 70)
    print("L104 FULL BENCHMARK — ONLINE MODE (1000+ questions)")
    print("=" * 70)
    t0 = time.time()

    harness = BenchmarkHarness()
    report = harness.run_all(online=True, mmlu_count=500, arc_count=500)

    harness.print_report()

    # ── Per-category breakdown ──
    detailed = report.get("detailed_results", {})

    # MMLU per-subject
    mmlu_detail = detailed.get("MMLU", {})
    mmlu_results = mmlu_detail.get("details", mmlu_detail.get("results", []))
    if mmlu_results:
        from collections import Counter
        subj_correct = Counter()
        subj_total = Counter()
        subj_wrong = []
        for r in mmlu_results:
            s = r.get("subject", "unknown")
            subj_total[s] += 1
            if r.get("correct"):
                subj_correct[s] += 1
            else:
                subj_wrong.append(r)

        print("\n── MMLU Per-Subject Breakdown ──")
        for s in sorted(subj_total.keys()):
            c = subj_correct[s]
            t = subj_total[s]
            pct = c / t * 100 if t else 0
            marker = " ✗" if pct < 50 else ""
            print(f"  {s:>45}: {c}/{t} = {pct:5.1f}%{marker}")

        # Show some wrong answers
        if subj_wrong:
            print(f"\n── MMLU Failures ({len(subj_wrong)} total) — first 15 ──")
            for r in subj_wrong[:15]:
                q = r.get("question", "")[:80]
                print(f"  [{r.get('subject','?')}] Q: {q}")
                print(f"    Expected: {r.get('expected',r.get('answer','?'))}  Got: {r.get('predicted',r.get('chosen','?'))}")

    # ARC per-category
    arc_detail = detailed.get("ARC", {})
    arc_results = arc_detail.get("details", arc_detail.get("results", []))
    if arc_results:
        from collections import Counter
        cat_correct = Counter()
        cat_total = Counter()
        cat_wrong = []
        for r in arc_results:
            c = r.get("category", "unknown")
            cat_total[c] += 1
            if r.get("correct"):
                cat_correct[c] += 1
            else:
                cat_wrong.append(r)

        print("\n── ARC Per-Category Breakdown ──")
        for c in sorted(cat_total.keys()):
            cr = cat_correct[c]
            t = cat_total[c]
            pct = cr / t * 100 if t else 0
            print(f"  {c:>20}: {cr}/{t} = {pct:5.1f}%")

        if cat_wrong:
            print(f"\n── ARC Failures ({len(cat_wrong)} total) — first 15 ──")
            for r in cat_wrong[:15]:
                q = r.get("question", "")[:80]
                print(f"  [{r.get('category','?')}] Q: {q}")
                print(f"    Expected: {r.get('expected',r.get('answer','?'))}  Got: {r.get('predicted',r.get('chosen','?'))}")

    # MATH per-domain
    math_detail = detailed.get("MATH", {})
    math_results = math_detail.get("details", math_detail.get("results", []))
    if math_results:
        from collections import Counter
        dom_correct = Counter()
        dom_total = Counter()
        for r in math_results:
            d = r.get("domain", "unknown")
            dom_total[d] += 1
            if r.get("correct"):
                dom_correct[d] += 1

        print("\n── MATH Per-Domain Breakdown ──")
        for d in sorted(dom_total.keys()):
            c = dom_correct[d]
            t = dom_total[d]
            pct = c / t * 100 if t else 0
            print(f"  {d:>25}: {c}/{t} = {pct:5.1f}%")

    # Save full report
    out_path = "benchmark_full_online_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
