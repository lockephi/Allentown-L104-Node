# L104 ASI - Benchmark Documentation Hub

**Honest benchmark suite — real measured data from 2026-02-23**

> **Important**: L104 is a **local deterministic AI system**, not a large language model.
> Comparing it to GPT-4/Claude/Gemini requires careful framing because they are
> fundamentally different architectures serving different purposes.

---

## Quick Links

### Reports
- **[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)** — Full detailed results with methodology and honesty notes
- **[BENCHMARK_SUMMARY.md](BENCHMARK_SUMMARY.md)** — Quick reference card
- **[BENCHMARK_CHARTS.md](BENCHMARK_CHARTS.md)** — Visual comparisons and positioning matrices

### Raw Data
- **[benchmark_results.json](benchmark_results.json)** — Machine-readable metrics from latest run

---

## Real Measured Results (2026-02-23)

### Infrastructure Performance
| Metric | Measured Value | Context |
|--------|---------------|---------|
| DB Write | 16,580 ops/sec | 1.7x typical SQLite |
| DB Read | 482,337 ops/sec | 4.8x typical SQLite |
| Cache Hit | 1,594,126 ops/sec | Local LRU (no network) |
| Cache Miss | 1,069,512 ops/sec | Local LRU (no network) |
| KG Search | 100,079 queries/sec | 38,587 nodes, 2.9M links |
| Throughput | 4,769,839 ops/sec | Mixed micro-operations |

### Comprehensive Benchmark (v2.0.0 — Real Data from HuggingFace)

| Benchmark | Score | Problems | Data Source |
|-----------|-------|----------|-------------|
| MMLU | **26.6%** (133/500) | 500 | cais/mmlu (HuggingFace API) |
| HumanEval | **54.9%** (90/164) | 164 | openai/openai_humaneval (HuggingFace API) |
| MATH | **52.7%** (29/55) | 55 | Expanded hardcoded (7 domains) |
| ARC | **29.0%** (290/1000) | 1000 | allenai/ai2_arc (HuggingFace API) |
| **Composite** | **43.1%** | **1,719 total** | — |
| **Verdict** | **MODERATE** | — | — |

> **Academic sources**: MMLU (Hendrycks et al., ICLR 2021), HumanEval (Chen et al., OpenAI 2021),
> ARC (Clark et al., AI2 2018), MATH (Hendrycks et al., NeurIPS 2021)
>
> **Result**: MMLU/ARC near random chance (25% for 4-choice). HumanEval: 54.9% via 130+ algorithm pattern templates.
> MATH: 52.7% with enhanced symbolic solver (percentages, geometry, fractions, factorials).

### Curated Harness (v1.0 Reference — Cherry-Picked Easy Problems)

| Benchmark | Score | Samples | Note |
|-----------|-------|---------|------|
| MMLU | 93.3% (14/15) | 15 | 0.09% of real MMLU |
| HumanEval | 100.0% (6/6) | 6 | 3.7% of real HumanEval |
| MATH | 100.0% (10/10) | 10 | 0.08% of real MATH |
| ARC | 90.0% (9/10) | 10 | 0.85% of real ARC |
| **Composite** | **97.3%** | **41 total** | Cherry-picked easy problems |

> These curated scores indicate the engines work correctly on their designed problems,
> but are **not representative** of performance on real, diverse question sets.

### Wrong Answers (Transparency)
| Benchmark | Question | L104 Answer | Correct Answer |
|-----------|----------|-------------|----------------|
| MMLU | Ball thrown straight up at highest point | "Zero velocity, max PE" | "Zero velocity, max gravitational PE" |
| ARC | White blood cells function | "break down__(nutrients)" | "fight__(germs)" |

---

## L104 vs Industry — Honest Comparison

### Where L104 Genuinely Excels
| Advantage | Why |
|-----------|-----|
| **Speed** | Local function calls (~µs) vs API round-trips (~500ms) |
| **Cost** | $0 per query — no API fees ever |
| **Privacy** | 100% local, zero data leaves the machine |
| **Persistence** | True stateful memory across sessions (SQLite + JSON) |
| **Offline** | Works without internet, no rate limits |
| **Determinism** | Same input always produces same output |

### Where Cloud LLMs Genuinely Excel
| Advantage | Why |
|-----------|-----|
| **Knowledge breadth** | Trained on internet-scale data |
| **Natural language** | Fluent text generation and understanding |
| **Reasoning** | Complex multi-step logical chains |
| **Code generation** | Can write novel code from descriptions |
| **Multi-modal** | Vision, audio, tool use |
| **Accessibility** | No local setup required |

### What This Means
L104 and cloud LLMs serve **different purposes**. L104 is a specialized local
intelligence system optimized for speed, privacy, and domain expertise.
Cloud LLMs are general-purpose reasoning engines with broad knowledge.
**Neither replaces the other.**

---

## Running Benchmarks

### Full Benchmark Suite
```bash
# Run comprehensive benchmark (fetches real data from HuggingFace — recommended)
.venv/bin/python -c "
from l104_asi.benchmark_harness import BenchmarkHarness
h = BenchmarkHarness()
h.run_all(online=True)
h.print_report()
"

# Run curated-only benchmark (original tiny sample sets)
.venv/bin/python -c "
from l104_asi.benchmark_harness import BenchmarkHarness
h = BenchmarkHarness()
h.run_all(online=False)
h.print_report()
"
```

### Infrastructure Benchmarks
```bash
# Run infrastructure tests (database, cache, speed, knowledge graph)
.venv/bin/python benchmark.py
```

### Cross-Engine Debug
```bash
# Run all 41 cross-engine validation tests
.venv/bin/python cross_engine_debug.py
```

---

## Environment

- **Hardware:** MacBook Air (Apple Silicon)
- **OS:** macOS
- **Python:** 3.12+
- **L104 Version:** ASI v8.0.0, AGI v57.1.0, Code Engine v6.3.0
- **Database:** SQLite (local)
- **Last Run:** 2026-02-23

---

## Benchmark Methodology

1. **Comprehensive benchmarks (v2.0.0)**: BenchmarkHarness fetches real problems via HuggingFace Datasets API (cais/mmlu, allenai/ai2_arc, openai/openai_humaneval) and runs L104 engines against them
2. **Curated benchmarks (v1.0 reference)**: Built-in sample problems run through deterministic engines
3. **Infrastructure tests**: Direct measurement of ops/sec via Python timing
4. **Industry baselines**: Published benchmarks from vendor documentation and academic papers
5. **No cherry-picking**: All results reported including wrong answers and 0% scores
6. **Sample sizes disclosed**: Every score shows (n/N) so readers can judge significance

### Known Limitations
- L104 engines use pattern matching and symbolic solvers, not neural inference
- MMLU/ARC scores are near random chance level (25%) — limited real language comprehension
- HumanEval: 54.9% pass rate via 130+ deterministic algorithm patterns (not neural)
- MATH: 52.7% capability on symbolic/arithmetic/geometry problems
- Speed comparisons mix local calls with network round-trips (different operations)
- No standardized third-party audit of results

---

**Last Updated:** 2026-02-23
**System:** L104 Sovereign Node
