# L104 Performance Limits Audit — Complete Report

> **Generated**: 2026-03-04
> **Scope**: l104_asi, l104_agi, l104_science_engine, l104_quantum_gate_engine, l104_code_engine, l104_intellect

---

## Table of Contents

1. [l104_asi/ — ASI Core Package](#1-l104_asi)
2. [l104_agi/ — AGI Core Package](#2-l104_agi)
3. [l104_science_engine/ — Science Engine](#3-l104_science_engine)
4. [l104_quantum_gate_engine/ — Quantum Gate Engine](#4-l104_quantum_gate_engine)
5. [l104_code_engine/ — Code Engine](#5-l104_code_engine)
6. [l104_intellect/ — Local Intellect](#6-l104_intellect)
7. [Grover/Quantum Search Infrastructure](#7-grover-quantum-search)
8. [Summary: Top Priority Unlimiting Targets](#8-summary)

---

## 1. l104_asi/

### 1A. `l104_asi/constants.py` — Central Limits Configuration

| Line | Constant | Value | Constrains | Recommendation |
|------|----------|-------|------------|----------------|
| 115 | `_detect_system_max_qubits(max_cap=25)` | 25 | Max qubit ceiling for all quantum ops | Raise to 30+ on high-RAM systems |
| 291 | `ASI_SELF_MODIFICATION_DEPTH` | 0xFFFF | Self-mod depth | ✅ Already unlimited |
| 311 | `THEOREM_AXIOM_DEPTH` | 5 | Symbolic reasoning chain length | Raise to 13 (PHI×8) |
| 312 | `SELF_MOD_MAX_ROLLBACK` | 10 | Rollback buffer | Raise to 50+ |
| 320 | `MULTI_HOP_MAX_HOPS` | 7 | Multi-hop reasoning chain | Raise to 13-21 |
| 330 | `VQE_ANSATZ_DEPTH` | 4 | VQE circuit layers | Raise to 8-12 |
| 332 | `VQE_MAX_QUBITS` | min(12, SYS) | VQE width | Raise to min(20, SYS) |
| 334 | `QAOA_SUBSYSTEM_QUBITS` | min(8, SYS) | QAOA routing space | Raise to min(14, SYS) |
| 335 | `QRC_RESERVOIR_QUBITS` | min(10, SYS) | Reservoir computing size | Raise to min(16, SYS) |
| 336 | `QRC_RESERVOIR_DEPTH` | 8 | Random unitary depth | Raise to 16 |
| 337 | `QKM_FEATURE_QUBITS` | min(8, SYS) | Feature map qubits | Raise to min(12, SYS) |
| 338 | `QPE_PRECISION_QUBITS` | min(8, SYS-1) | QPE precision bits | Raise to min(12, SYS-1) |
| 430 | `CONSCIOUSNESS_SPIRAL_DEPTH` | 13 | Spiral recursion depth | Raise to 21 (fib) |
| 431 | `CONSCIOUSNESS_PHI_TRAJECTORY_WINDOW` | 50 | Trajectory sliding window | Raise to 200 |
| 438 | `TRAJECTORY_WINDOW_SIZE` | 20 | Score history for regression | Raise to 100 |
| 439 | `TRAJECTORY_PREDICTION_HORIZON` | 5 | Prediction forward steps | Raise to 13 |
| 455 | `RESILIENCE_MAX_RETRY` | 3 | Failed subsystem retries | Raise to 5-7 |

### 1B. `l104_asi/reasoning.py` — Reasoning Engine Limits

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 54 | `MCTSReasoner.__init__(max_iterations=50)` | 50 | MCTS search iterations | Raise to 200 |
| 57 | `self.max_depth = int(PHI * 5)` | 8 | MCTS tree depth | Raise to 13 |
| 110 | `path = [root.query[:100]]` | 100 chars | Query truncation in path | Raise to 500 |
| 123 | `path[:5]` | 5 | Reasoning path displayed | Raise to 13 |
| 162 | `query[:250]` | 250 chars | Action query truncation | Raise to 1000 |
| 269 | `ReflectionRefinementLoop(max_reflections=6)` | 6 | Reflection iterations | Raise to 13 |
| 294 | `solution[:200]` | 200 chars | Solution preview truncation | Raise to 1000 |
| 311-315 | `solution[:100]`, `problem[:300]`, `problem[:250]` | Various | Multi-hop query building truncation | Raise all to 500+ |
| 360 | `TreeOfThoughts.K = int(PHI * 3)` | 4 | Branching factor | Raise to 8 |
| 361 | `TreeOfThoughts.B = int(PHI * 2)` | 3 | Beam width | Raise to 5 |
| 368 | `MCTSReasoner(max_iterations=30)` | 30 | TOT's internal MCTS | Raise to 100 |
| 369 | `ReflectionRefinementLoop(max_reflections=4)` | 4 | TOT's reflector | Raise to 8 |
| 371 | `think(..., max_depth=4)` | 4 | Default TOT depth | Raise to 8 |
| 376 | `solve_fn, max_iterations=20` | 20 | Complex problem MCTS | Raise to 100 |
| 399 | `variant[:300]` | 300 chars | Variant query truncation | Raise to 1000 |
| 401 | `str(result...)[:500]` | 500 chars | Solution truncation | Raise to 2000 |
| 402 | `variant[:80]` | 80 chars | Path step truncation | Raise to 300 |
| 408 | `sorted(...)[:1]` | 1 | Fallback viable candidates | Raise to 3 |
| 443 | `query[:300]` | 300 chars | Variant generation truncation | Raise to 1000 |
| 458 | `parts[:5]` | 5 | GoT aggregation merge | Raise to 13 |
| 490 | `MCTSReasoner(max_iterations=15)` | 15 | Multi-hop MCTS fallback | Raise to 50 |
| 512 | `max_depth=3` | 3 | First-hop TOT depth | Raise to 5 |
| 524 | `max_iterations=10` | 10 | Stuck-hop MCTS | Raise to 30 |
| 539 | `current_query[:200]` | 200 chars | Hop query recording | Raise to 500 |
| 565 | `solution_text[:200]`, `problem[:200]` | 200 chars | Inter-hop query building | Raise to 500 |
| 711 | `str(c['solution'])[:200]` | 200 chars | Candidate solutions to consensus | Raise to 1000 |
| 747 | `candidates[:5]` | 5 | Candidate analysis limit | Raise to 13 |

### 1C. `l104_asi/core.py` — ASI Core Pipeline Limits

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 973 | `dict(list(self._pipeline_metrics.items())[:10])` | 10 | Pipeline metrics display | Raise to 50 |
| 1029 | `_search_all_knowledge(message, max_results=5)` | 5 | Knowledge search results | Raise to 25 |
| 1032 | `[str(f)[:200] for f in all_k[:3]]` | 3 facts, 200 chars | Top facts recorded | Raise to 10/500 |
| 1054 | `prompt[:100] → completion[:100]` | 100 chars | Training data content truncation | Raise to 500 |
| 1062 | `manifold_hit[:200]` | 200 chars | Knowledge manifold truncation | Raise to 1000 |
| 1070 | `vault_hit[:200]` | 200 chars | Knowledge vault truncation | Raise to 1000 |
| 1079 | `str(h)[:200]` | 200 chars | JSON knowledge truncation | Raise to 1000 |
| 2224 | `time.sleep(min(backoff, 1.0))` | 1s cap | Retry backoff ceiling | Consider 2-3s |
| 2584-2585 | `_asi_score_history > 100` → trim to 100 | 100 | Score history length | Raise to 500 |
| 3858 | `errors[:10]` | 10 | Error details reported | Raise to 50 |
| 3883 | `routed_subsystems[:3]` | 3 | Subsystem routing depth | Raise to 5-7 |
| 3999 | `insights[:3]` | 3 | SG insights reported | Raise to 10 |
| 4066 | `_search_training_data(query_str)[:5]` | 5 | Training hits per query | Raise to 15 |
| 4095 | `[str(f)[:150] for f in li_facts[:5]]` | 5 facts, 150 chars | LI facts truncation | Raise to 10/500 |
| 4130 | `str(...)[:500]` | 500 chars | Logic text truncation | Raise to 2000 |
| 4135 | `fallacies[:3]` | 3 | Fallacy detection limit | Raise to 10 |
| 4216 | `query_str.lower().split()...[:5]` | 5 | Keywords for KB search | Raise to 10 |
| 4292 | `_reflect_text[:300]` | 300 chars | Reflection text truncation | Raise to 1000 |
| 4305 | `_lang_text[:300]` | 300 chars | Language text truncation | Raise to 1000 |
| 4344 | `str(result['solution'])[:1000]` | 1000 chars | Hallucination purge input | Raise to 5000 |
| 4363 | `query_str[:100]`, `str(result['solution'])[:500]` | 100/500 chars | LI cache key/value truncation | Raise to 300/2000 |
| 4438 | `keywords...[:5]` | 5 | Router feedback keywords | Raise to 10 |
| 4447 | `input_data=query_str[:200]` | 200 chars | Pipeline audit input | Raise to 500 |
| 4456 | `routed_subsystems[:3]` | 3 | Top routes recorded | Raise to 5 |
| 4566 | `result['research'][:1000]` | 1000 chars | Gemini research truncation | Raise to 5000 |
| 4574 | `combined_research[:500]` | 500 chars | Research analysis truncation | Raise to 2000 |
| 4639 | `time.sleep(1)` | 1s | Fixed sleep in processing loop | Remove or make conditional |
| 4907 | `result["research"]["insights"][:10]` | 10 | Research insights limit | Raise to 25 |
| 5886 | `input_data=problem[:200]` | 200 chars | Multi-hop audit input | Raise to 500 |
| 5988-5989 | `query[:200]`, `routes[:10]` | 200/10 | Route debug info | Raise to 500/25 |
| 6361 | `updated_keys[:100]` | 100 | KB keys reported | Raise to 500 |

### 1D. `l104_asi/pipeline.py` — Pipeline Orchestrator

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 35 | `_cache_max_size = 256` | 256 | Pipeline cache entries | Raise to 1024 |
| 486 | `expert_load...[:5]` | 5 | Expert load display | Raise to 15 |
| 630 | `keyword_stats...[:10]` | 10 | Keyword stats displayed | Raise to 25 |
| 655 | `PipelineReplayBuffer(capacity=1000)` | 1000 | Replay buffer capacity | Raise to 5000 |
| 665 | `query[:200]` | 200 chars | Query truncation in replay | Raise to 500 |
| 778 | `keywords...[:5]` | 5 | Router feedback keywords | Raise to 10 |
| 782 | `routes[:3]` | 3 | Routes returned to caller | Raise to 5 |
| 927 | `indexed[:3]` | 3 | ML classifier top results | Raise to 5 |

### 1E. `l104_asi/consciousness.py` — Consciousness Engine

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 122 | `_consciousness_history[-20:]` | 20 | History window for trend analysis | Raise to 50 |
| 183-195 | `CONSCIOUSNESS_SPIRAL_DEPTH = 13` | 13 | Phi spiral recursion depth | Raise to 21 |
| 242 | `spiral_values[:5]` | 5 | Spiral values inspection | Raise to 13 |
| 294 | `overtone_scores...[:5]` | 5 | Top overtone scores | Raise to 10 |
| 447 | `min(0.3, len(history) * 0.03)` | 0.3 cap | History bonus ceiling | Raise to 0.618 (TAU) |
| 495 | `self.TESTS[:16]` | 16 | Consciousness test count | Allow full test set |

### 1F. `l104_asi/quantum.py` — Quantum Subsystem

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 396 | `ranked[:3]` | 3 | Selected subsystems for routing | Raise to 5 |
| 467 | `(...)[:3]` | 3 | Quantum route results | Raise to 5 |
| 992 | `coherence_vqe(max_iterations=50)` | 50 | VQE optimization iterations | Raise to 200 |

### 1G. `l104_asi/dual_layer.py` — Dual-Layer Engine

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 1283 | `violations[:5]` | 5 | Violation details reported | Raise to 20 |
| 1562 | `errors[:5]` | 5 | Best errors displayed | Raise to 20 |
| 2056-2059 | `by_physics[:5]`, `by_improvement[:5]`, `by_thought[:5]` | 5 each | Top-N layer rankings | Raise to 13 |
| 2196 | `convergences[:20]` | 20 | Convergence records | Raise to 50 |
| 2624 | `rankings[:5]` | 5 | Best rankings | Raise to 13 |
| 2668 | `hits[:5]` | 5 | Best hits | Raise to 13 |
| 2898 | `cross_correlations[:5]` | 5 | Strongest correlations | Raise to 13 |
| 3254 | `samples[:5]` | 5 | Sample inspection | Raise to 13 |
| 3320 | `peaks[:10]` | 10 | Resonance peaks | Raise to 25 |
| 4030 | `time.sleep(min(backoff, 1.0))` | 1s cap | Retry backoff | Consider 2-3s |

### 1H. `l104_asi/formal_logic.py` — Formal Logic Engine

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 700 | `applicable[:5]` | 5 | Applicable laws returned | Raise to 13 |
| 1443 | `_max_iterations = 5000` | 5000 | Resolution safety cap | ✅ Reasonable for safety |
| 1758 | `forward_chain(max_steps=20)` | 20 | Forward chaining steps | Raise to 50 |
| 1774 | `rules match...[:3]` | 3 | Rules applied per step | Raise to 8 |
| 1790 | `build_chain(max_steps=15)` | 15 | Backward chain building | Raise to 50 |

### 1I. `l104_asi/kb_reconstruction.py` — Knowledge Base

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 729 | `entries[:10]` | 10 | Per-category intake cap | Raise to 50 |
| 732 | `completion[:200]` | 200 chars | Fact truncation | Raise to 1000 |
| 738 | `entries[:15]` | 15 | Graph entries cap | Raise to 50 |
| 742 | `definition = prompt[:200]` | 200 chars | Definition truncation | Raise to 1000 |
| 812 | `facts[:20]` | 20 | Facts per node | Raise to 100 |
| 849 | `facts[:15]` | 15 | Facts per node (alt path) | Raise to 100 |

### 1J. `l104_asi/theorem_gen.py` — Theorem Generator

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 64 | `available[:8]` | 8 | Axiom pairs checked per step | Raise to 21 |
| 262 | `t.statement[:80]` | 80 chars | Theorem statement truncation | Raise to 500 |

---

## 2. l104_agi/

### 2A. `l104_agi/core.py` — AGI Core

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 132 | `_telemetry_capacity = 500` | 500 | Telemetry log max entries | Raise to 2000 |
| 174 | `_consciousness_cache_ttl = 10.0` | 10s | Consciousness cache TTL | Consider 30s+ |
| 184 | `_reasoning_history = deque(maxlen=100)` | 100 | Reasoning history buffer | Raise to 500 |
| 187 | `_replay_buffer = deque(maxlen=200)` | 200 | Pipeline replay buffer | Raise to 1000 |
| 251 | `_scheduler_pattern_buffer = deque(maxlen=500)` | 500 | Scheduler pattern buffer | Raise to 2000 |
| 266 | `_coherence_history = deque(maxlen=100)` | 100 | Coherence history buffer | Raise to 500 |
| 383 | `thought_vec = [... for c in thought[:64]]` | 64 chars | Thought vector length | Raise to 256 |
| 394 | `_search_training_data(thought[:200], max_results=3)` | 200c/3 results | KB search truncation | Raise to 500/10 |
| 396 | `[r.get('completion', '')[:200] for r in kb_results[:2]]` | 200c/2 results | KB context truncation | Raise to 500/5 |
| 430 | `min(0.24, fused * 0.02)` | 0.24 cap | Synthesis boost ceiling | Raise to 0.618 (TAU) |
| 951 | `real_probs[:8]` | 8 | QPU probabilities displayed | Raise to 16 |
| 1023 | `ranking[:3]` | 3 | Grover-ranked subsystems | Raise to 5 |
| 1111 | `real_probs[:8]` | 8 | QPU probabilities displayed | Raise to 16 |
| 1916 | `.pop(0)` when `> _telemetry_capacity` | FIFO drop | O(n) eviction on list | Convert to deque(maxlen=N) |
| 3113 | `score = min(1.0, demon_eff * 2.0)` | 1.0 cap | Demon efficiency score | Score normalization (keep) |
| 3163 | `chaos_amp = min(0.5, entropy / 20.0)` | 0.5 cap | Chaos amplitude cap | Raise to 0.618 |
| 3267 | `min(1.0, self.intellect_index / 1e12)` | 1.0 cap | Intellect dimension cap | Score normalization (review if index > 1e12) |
| 3300 | `min(1.0, fused / 12.0 * 0.7 + min(boost, 1.0) * 0.3)` | 1.0 cap | Synthesis dimension cap | Score normalization (keep) |
| 3685 | `(...)[:5]` | 5 | Diagnostic display | Raise to 13 |

---

## 3. l104_science_engine/

### 3A. `l104_science_engine/coherence.py` — Coherence Subsystem

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 74 | `BRAID_DEPTH = 4` | 4 | Anyon braid ops per evolve step | Raise to 8-13 |
| 172 | `limited_seeds = seed_thoughts[:50]` | 50 | Seed thoughts for initialization | Raise to 200 |
| 269 | `range(min(n, 20))` | 20 | PHI-spiral pattern search cap | Raise to full `range(n)` |
| 363 | `spectrum[:20]` | 20 | Spectrum display | Raise to 50 |
| 383 | `range(min(5, len(sorted_modes) - 1))` | 5 | Gap analysis modes | Raise to 13 |
| 405 | `sorted_modes[:5]` | 5 | Dominant modes reported | Raise to 13 |
| 577 | `sorted_probs[:min(n, 10)]` | 10 | Outcome probability display | Raise to 25 |
| 642 | **`braid_ops = min(braid_ops, 32)` — "Cap for performance"** | 32 | **Braid operations ceiling** | **Raise to 64-104** |

### 3B. `l104_science_engine/entropy.py` — Entropy Subsystem

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 249 | `trajectory[:5] + trajectory[-5:]` | 10 total | Trajectory sample display | Raise to 20+20 |
| 327 | `trajectory[:3] + trajectory[-3:]` | 6 total | Trajectory sample display | Raise to 10+10 |
| 359 | `min(len(chaos_products), i + 4)` | +4 window | Chaos correlation window | Raise to +8 |
| 801 | `min(history_depth, n // 2, n - 1)` | n//2 | Hurst exponent window | ✅ Mathematically correct |
| 867 | `steps_ahead = min(3, n - i - 1)` | 3 | Transfer entropy lookahead | Raise to 5 |
| 1019 | `attractors[:5] + attractors[-5:]` | 10 total | Grid detail sampling | Raise to 20+20 |

### 3C. `l104_science_engine/quantum_25q.py` — Quantum 25Q

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 636-637 | `max_depth = max(1, min(max_depth, 1000))` | 1000 | Circuit depth budget ceiling | Raise to 5000 |

### 3D. `l104_science_engine/bridge.py` — Science-Quantum Bridge

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 310-311 | `max_depth = max(1, min(max_depth, 1000))` | 1000 | Depth budget ceiling | Raise to 5000 |
| 442 | `depth = min(circuit.get("depth", 50), ...)` | 50 default | Default circuit depth | Raise to 100 |

---

## 4. l104_quantum_gate_engine/

### 4A. `l104_quantum_gate_engine/compiler.py` — Gate Compiler

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 708 | **`range(min(n_braids, 20))` — "Cap at 20 braids"** | 20 | **Fibonacci anyon braid count** | **Raise to 50-104** |

### 4B. `l104_quantum_gate_engine/orchestrator.py` — Orchestrator

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 710 | `max_iterations=50` | 50 | Sacred VQE iterations | Raise to 200 |
| 1158 | `range(min(n_qubits, 20))` | 20 | Coherence seed qubits | Raise to full range |
| 1159 | `circuit.operations[:30]` | 30 | Gate seeds for coherence | Raise to 100 |
| 1184 | `min(n_qubits, 6)` | 6 | Iron lattice Hamiltonian qubits | Raise to min(n_qubits, 12) |
| 1187 | `se.coherence.evolve(steps=min(depth, 10))` | 10 | **Coherence evolution steps capped at 10** | **Raise to min(depth, 50)** |

### 4C. `l104_quantum_gate_engine/tensor_network.py` — Tensor Network MPS

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 71-72 | `DEFAULT_MAX_BOND = 64` | 64 | Default MPS bond dimension | Raise to 128-256 |
| 785 | `batch_size = min(shots, 256)` | 256 | Sampling batch size | Raise to 1024 |

### 4D. `l104_quantum_gate_engine/constants.py` — Engine Constants

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 111 | `MPS_SVD_CUTOFF = 1e-16` | 1e-16 | SVD truncation threshold | ✅ Already loosened |

### 4E. `l104_quantum_gate_engine/gates.py` — Gate Algebra

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 1032 | `range(1, min(max_k + 1, 10001))` | 10000 | Series expansion order | ✅ Reasonable ceiling |

---

## 5. l104_code_engine/

### 5A. `l104_code_engine/quantum.py` — Code Quantum Intelligence

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 361 | `scores...[:20]` | 20 | Score display | Raise to 50 |
| 411 | **`range(min(steps - 1, 10))`** | 10 | **Quantum walk operator repetitions** | **Raise to min(steps-1, 50)** |
| 729 | `range(min(n_qubits, 4))` | 4 | Superposition qubit limit | Raise to 8 |
| 1037 | `code_files[:4]` — "Max 4 files for tractability" | 4 | **Quantum analysis file limit** | **Raise to 8-16** |
| 1038 | `file_features.values()[:4]` | 4 | Features per file | Raise to 8 |
| 1077 | `range(min(len(file_states), 4))` | 4 | File states analyzed | Raise to 8 |
| 1787 | `range(min(max_lines, len(attention_weights)))` | max_lines | Attention analysis | ✅ Uses parameter |
| 2089 | `clean_features = clean_features[:16]` | 16 | Feature vector truncation | Raise to 32 |

### 5B. `l104_code_engine/hub.py` — Code Engine Hub

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 522 | `security[:3]` | 3 | Security vulns displayed | Raise to 10 |
| 526 | `issues[:3]` | 3 | Concurrency issues | Raise to 10 |
| 530 | `smells[:3]` | 3 | Code smells displayed | Raise to 10 |
| 538 | `drifts[:3]` | 3 | Contract drifts | Raise to 10 |
| 542 | `gaps[:3]` | 3 | Type flow gaps | Raise to 10 |
| 546 | `violations[:2]` | 2 | SOLID violations | Raise to 5 |
| 592 | `actions[:25]` | 25 | Review actions | Raise to 50 |
| 985 | `security[:5]` | 5 | Audit security vulns | Raise to 15 |
| 989 | `violations[:3]` | 3 | SOLID violations in audit | Raise to 10 |
| 992 | `hotspots[:3]` | 3 | Performance hotspots | Raise to 10 |
| 995 | `suggestions[:3]` | 3 | Refactoring suggestions | Raise to 10 |
| 1034 | `actions[:15]` | 15 | Audit actions | Raise to 50 |
| 1756 | `code_lines[:50]` | 50 | Coherence seed lines | Raise to 200 |
| 1872 | `actions[:25]` | 25 | Review actions cap | Raise to 50 |

### 5C. `l104_code_engine/training_kernel.py` — Training Kernel

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 1079 | `range(min(len(_quantum_params), GRADIENT_PARAMS))` | GRADIENT_PARAMS | Parameter optimization limit | Check constant value |
| 1244 | **`range(min(5, self.MAX_EPOCHS))`** | 5 | **Training epochs capped at 5** | **Raise to MAX_EPOCHS** |
| 1641 | `list(self._corpus_cache.items())[:10]` | 10 | Corpus cache inspection | Raise to 50 |

---

## 6. l104_intellect/

### 6A. `l104_intellect/constants.py` — Intellect Constants

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 85 | `MAX_SAVE_STATES = 50` | 50 | Evolution checkpoints | Raise to 200 |
| 87 | `HIGHER_LOGIC_DEPTH = 25` | 25 | Higher logic reasoning depth | Raise to 50 (was 5) |
| 161 | `SAGE_SCOUR_MAX_FILES = 500` | 500 | File scan limit | Raise to 2000 |
| 218 | `HL_META_REASONING_TOP_K = 8` | 8 | Meta-reasoning top-K | Raise to 15 |
| ~148 | `SAGE_VOID_DEPTH_MAX = 13` | 13 | Creative void depth | Raise to 21 |
| ~152 | `NON_LOCALITY_BRIDGE_DEPTH = 5` | 5 | Non-local bridge hops | Raise to 8 |

### 6B. `l104_intellect/local_intellect_core.py` — Local Intellect Core

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 600 | `self.training_data[:500]` | 500 | Training index build scope | Raise to full dataset |
| 603 | `tokens...[:50]` | 50 | Token count per entry | Raise to 100 |
| 661 | `tokens...[:20]` | 20 | Message tokens for analysis | Raise to 50 |
| 1100 | `concepts[:50]` | 50 | Grover search concepts | ✅ Raised from 8 |
| 1213 | `concepts[:25]` | 25 | Quantum amplified concepts | Raise to 50 |
| 1221 | `_search_training_data(query, max_results=5)` | 5 | Training search in synthesis | Raise to 15 |
| 1227 | `training_matches[:3]` | 3 | Training matches used | Raise to 8 |
| 1229 | `match["completion"][:500]` | 500 chars | Completion truncation | Raise to 2000 |
| 1233 | `list(all_entangled)[:5]` | 5 | Entangled concepts used | Raise to 13 |
| 1235 | `self.knowledge[entangled_concept][:200]` | 200 chars | Entangled knowledge truncation | Raise to 1000 |
| 1245 | `list(all_entangled)[:10]` | 10 | Entangled concepts reported | Raise to 25 |
| 2005 | `str(value)[:1500]` | 1500 chars | JSON value truncation | ✅ Reasonable |
| 2011 | `obj[:100]` | 100 | List iteration limit in search | Raise to 500 |
| 2605 | `query_terms = query_terms[:8]` | 8 | **Query term cap in training search** | **Raise to 15** |
| 2624 | `self.training_index[term][:30]` | 30 | Entries per term in index | Raise to 100 |
| 7275 | `_search_training_data(message, max_results=8)` | 8 | Training search in higher-logic | Raise to 20 |
| 7297 | `_search_chat_conversations(message, max_results=5)` | 5 | Chat search | Raise to 15 |
| 7306 | `_search_knowledge_manifold(message, max_results=5)` | 5 | Manifold search | Raise to 15 |
| 7321 | `_search_knowledge_vault(message, max_results=5)` | 5 | Vault search | Raise to 15 |
| 7795/7853/7884/7907/7974/8013/8111 | `max_results=5` | 5 | Multiple sage search calls | Raise all to 15 |

### 6C. `l104_intellect/optimization.py` — Optimization Engine

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 722 | **`range(min(max_iter, 30))`** | 30 | **Bayesian optimization iterations capped at 30** | **Raise to max_iter (remove cap)** |

### 6D. `l104_intellect/quantum_recompiler.py` — Quantum Recompiler

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 140 | `message[:200]` | 200 chars | Query recording | Raise to 500 |
| 141 | `response[:500]` | 500 chars | Response recording | Raise to 2000 |
| 157 | `keywords...[:30]` | 30 | Keyword extraction | ✅ Reasonable |
| 248 | `query_context_index(query, max_results=depth * 2)` | depth×2 | Context results | Raise multiplier to 5 |
| 284 | `part.get("concepts", [])[:3]` | 3 | Related concepts per part | Raise to 8 |
| 287 | `list(related_concepts)[:5]` | 5 | Related concepts shown | Raise to 13 |
| 334 | `query_context_index(query, max_results=10)` | 10 | Sage context results | Raise to 25 |
| 344 | `sage_patterns[:5]` | 5 | Sage patterns used | Raise to 13 |
| 345 | `pattern...[:300]` | 300 chars | Pattern wisdom truncation | Raise to 1000 |
| 353 | `wisdom_parts[1][:200]` | 200 chars | Deeper insight truncation | Raise to 500 |
| 450 | `_search_training_data(topic, max_results=10)` | 10 | Research training search | Raise to 25 |
| 453 | `training_results[:3]` | 3 | Training results used | Raise to 8 |
| 456 | `tr.get("completion", "")[:500]` | 500 chars | Completion truncation | Raise to 2000 |

### 6E. `l104_intellect/distributed.py` — Distributed Intellect

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 427 | `list(effective)[:20]` | 20 | Effective set display | Raise to 50 |
| 693 | `spread_results[:10]` | 10 | Spread results display | Raise to 25 |
| 770 | `message[:100]` | 100 chars | Message truncation | Raise to 500 |
| 775 | `deliveries[:8]` | 8 | Delivery display | Raise to 20 |

### 6F. `l104_intellect/hardware.py` — Hardware Tuning

| Line | Limit | Value | Constrains | Recommendation |
|------|-------|-------|------------|----------------|
| 45 | `memory_limit_mb: 512` | 512MB | Default memory limit | Dynamic based on system |

---

## 7. Grover/Quantum Search Infrastructure

### Existing Grover Implementations (can be leveraged for unlimiting)

| File | Line | Status | Description |
|------|------|--------|-------------|
| `l104_asi/self_mod.py:392` | `grover_amplified_transform_select()` | ✅ ACTIVE | Grover amplitude amplification for AST pass selection |
| `l104_asi/self_mod.py:451-464` | Grover oracle + diffusion | ✅ ACTIVE | Real Grover iterations on pass fitness amplitudes |
| `l104_agi/core.py:957-1023` | `_grover_amplified_route()` | ✅ ACTIVE | Grover-amplified subsystem routing (3-qubit, 8 states) |
| `l104_intellect/local_intellect_core.py:1075` | `grover_amplified_search()` | ✅ ACTIVE | Grover-amplified KB concept search |
| `l104_asi/quantum.py:40-45` | `_GROVER_NERVE_AVAILABLE` | ⚠️ OPTIONAL | Grover Nerve Link import (may not be installed) |
| `l104_asi/language_comprehension/mcq_solver.py` | KB fact Grover search | ✅ ACTIVE | Grover oracle over KB facts for knowledge retrieval |

### Quantum Search Stubs That Could Use Grover But Don't

| File | Line | Current Approach | Grover Opportunity |
|------|------|-----------------|-------------------|
| `l104_asi/core.py:4066` | `_search_training_data[:5]` | Linear BM25 search → slice | Grover amplification of best matches |
| `l104_asi/pipeline.py:778` | `keywords[:5]` | Linear keyword extraction | Grover-amplified keyword selection |
| `l104_code_engine/quantum.py:1037` | `code_files[:4]` | Linear file slice | Grover search over file relevance |
| `l104_intellect/local_intellect_core.py:2624` | `training_index[term][:30]` | Linear index slice | Grover-amplified entry ranking |
| `l104_asi/formal_logic.py:700` | `applicable[:5]` | Linear law selection | Grover amplification of best law |

---

## 8. Summary: Top Priority Unlimiting Targets

### 🔴 Critical (Active Performance Bottlenecks)

| # | File | Line | Current | Impact | Fix |
|---|------|------|---------|--------|-----|
| 1 | `l104_code_engine/training_kernel.py` | 1244 | `min(5, MAX_EPOCHS)` | **Training always capped at 5 epochs** | Remove cap: `range(MAX_EPOCHS)` |
| 2 | `l104_intellect/optimization.py` | 722 | `min(max_iter, 30)` | **Bayesian opt always ≤30 iterations** | Remove cap: `range(max_iter)` |
| 3 | `l104_code_engine/quantum.py` | 411 | `min(steps-1, 10)` | **Quantum walk capped at 10 repetitions** | `min(steps-1, 50)` or remove |
| 4 | `l104_code_engine/quantum.py` | 1037 | `code_files[:4]` | **Only 4 files analyzed quantumly** | Raise to 16 |
| 5 | `l104_quantum_gate_engine/compiler.py` | 708 | `min(n_braids, 20)` | **Fibonacci braids capped at 20** | Raise to 104 |
| 6 | `l104_science_engine/coherence.py` | 642 | `min(braid_ops, 32)` | **Error mitigation braid cap** | Raise to 104 |
| 7 | `l104_science_engine/coherence.py` | 74 | `BRAID_DEPTH = 4` | **Low braid depth per step** | Raise to 8-13 |
| 8 | `l104_science_engine/coherence.py` | 269 | `range(min(n, 20))` | **Pattern search capped at 20** | `range(n)` |
| 9 | `l104_quantum_gate_engine/orchestrator.py` | 1187 | `min(depth, 10)` | **Coherence evolve max 10 steps** | Raise to 50 |
| 10 | `l104_intellect/local_intellect_core.py` | 2605 | `query_terms[:8]` | **Training search term cap** | Raise to 15 |

### 🟡 High Impact (Data Retention / Reasoning Depth)

| # | File | Line | Current | Impact | Fix |
|---|------|------|---------|--------|-----|
| 11 | `l104_asi/reasoning.py` | 54 | `MCTSReasoner(50)` | MCTS limited to 50 iterations | 200 |
| 12 | `l104_asi/reasoning.py` | 57 | `max_depth = 8` | MCTS depth limited | 13 |
| 13 | `l104_asi/constants.py` | 320 | `MULTI_HOP_MAX_HOPS = 7` | Multi-hop depth | 13-21 |
| 14 | `l104_asi/constants.py` | 438 | `TRAJECTORY_WINDOW_SIZE = 20` | Score regression window | 100 |
| 15 | `l104_asi/core.py` | 2584 | `_asi_score_history > 100` | Score history trimmed to 100 | 500 |
| 16 | `l104_agi/core.py` | 184 | `deque(maxlen=100)` | Reasoning history | 500 |
| 17 | `l104_agi/core.py` | 132 | `_telemetry_capacity = 500` | Telemetry capacity | 2000 |
| 18 | `l104_asi/pipeline.py` | 35 | `_cache_max_size = 256` | Pipeline cache | 1024 |
| 19 | `l104_asi/pipeline.py` | 655 | `capacity=1000` | Replay buffer | 5000 |
| 20 | `l104_intellect/constants.py` | 161 | `SAGE_SCOUR_MAX_FILES = 500` | File scan limit | 2000 |

### 🟢 Display/Reporting Limits (Low Risk, Easy Fix)

All `[:3]`, `[:5]`, `[:10]` truncations on **output/display** data (not processing) can be uniformly raised to `[:13]` or `[:25]` with zero risk. These are scattered across every file — over 100 instances total across the 6 packages.

---

*End of audit. 200+ individual limits catalogued across 6 packages.*
