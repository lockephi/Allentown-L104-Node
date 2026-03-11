#!/usr/bin/env python3
"""Show all benchmark results we have."""
import json
import os
import subprocess
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
SEP = "=" * 72

DETAIL_FILES = [
    "benchmark_online_results.json",
    "benchmark_full_online_results.json",
    "_benchmark_quantum_results.json",
    "_benchmark_clean_results.json",
]

SUMMARY_FILES = [
    "_mmlu_benchmark_results.json",
    "benchmark_results.json",
    "benchmark_logic_nlu_results.json",
    "benchmark_report.json",
]


def show_detailed(fname):
    """Show per-benchmark accuracy from detailed results."""
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)
    size_kb = os.path.getsize(path) / 1024

    print("\n" + SEP)
    print("  " + fname + "  (" + str(int(size_kb)) + " KB)")
    print(SEP)

    dr = data.get("detailed_results", {})
    total_q = 0
    total_c = 0
    for bench, info in dr.items():
        if not isinstance(info, dict):
            continue
        acc = info.get("accuracy", info.get("pass_rate", 0))
        n = info.get("total", info.get("count", info.get("n_questions", "?")))
        corr = info.get("correct", info.get("passed", "?"))
        line = "  {:12s}  accuracy={:>7}  correct={}/{}".format(bench, acc, corr, n)
        print(line)
        if isinstance(n, int):
            total_q += n
        if isinstance(corr, int):
            total_c += corr

    if total_q > 0:
        pct = total_c / total_q * 100
        print("  {:12s}  accuracy={:5.1f}%  correct={}/{}".format("TOTAL", pct, total_c, total_q))

    cs = data.get("composite_score", 0)
    print("  Composite (weighted): {:.4f} ({:.1f}%)".format(cs, cs * 100))
    vd = data.get("verdict", "N/A")
    print("  Verdict: " + str(vd))
    el = data.get("elapsed_seconds", 0)
    print("  Time: {:.0f}s".format(el))
    tq = data.get("total_questions", total_q)
    print("  Total questions: {}".format(tq))


def show_summary(fname):
    """Show summary-level benchmark files."""
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)
    size_kb = os.path.getsize(path) / 1024

    print("\n" + SEP)
    print("  " + fname + "  (" + str(int(size_kb)) + " KB)")
    print(SEP)

    if not isinstance(data, dict):
        print("  (top-level is {}, len={})".format(type(data).__name__, len(data)))
        return

    for key in sorted(data.keys()):
        val = data[key]
        if isinstance(val, (int, float)):
            if isinstance(val, float) and 0 < val < 1:
                print("  {}: {:.4f}  ({:.1f}%)".format(key, val, val * 100))
            else:
                print("  {}: {}".format(key, val))
        elif isinstance(val, str) and len(val) < 120:
            print("  {}: {}".format(key, val))
        elif isinstance(val, dict):
            acc_fields = {k: v for k, v in val.items() if isinstance(v, (int, float))}
            if acc_fields:
                print("  {}:".format(key))
                for k, v in acc_fields.items():
                    if isinstance(v, float) and v < 10:
                        print("    {}: {:.4f}".format(k, v))
                    else:
                        print("    {}: {}".format(k, v))
            else:
                print("  {}: dict({} keys)".format(key, len(val)))
        elif isinstance(val, list):
            print("  {}: {} items".format(key, len(val)))


def main():
    print("\n" + SEP)
    print("  L104 SOVEREIGN NODE — ALL BENCHMARK RESULTS")
    print(SEP)

    print("\n--- DETAILED BENCHMARK RESULTS (per-category accuracy) ---")
    for fname in DETAIL_FILES:
        try:
            show_detailed(fname)
        except Exception as e:
            print("  {} Error: {}".format(fname, e))

    print("\n\n--- SUMMARY BENCHMARK RESULTS ---")
    for fname in SUMMARY_FILES:
        try:
            show_summary(fname)
        except Exception as e:
            print("  {} Error: {}".format(fname, e))

    # Check running benchmark
    result = subprocess.run(["pgrep", "-f", "_bench_1000q"], capture_output=True, text=True)
    if result.stdout.strip():
        pids = result.stdout.strip().split("\n")
        print("\n" + SEP)
        print("  NOTE: _bench_1000q.py is STILL RUNNING (PID {})".format(pids[0]))
        print("  Results will appear when it completes.")
        print(SEP)


if __name__ == "__main__":
    main()
