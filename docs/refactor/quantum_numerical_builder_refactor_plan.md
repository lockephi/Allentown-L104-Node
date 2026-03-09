# Refactor Plan: `l104_quantum_numerical_builder.py` → `l104_numerical_engine/`

> **Monolith**: `l104_quantum_numerical_builder.py` — 6,069 lines, 28 classes, v3.0.0
> **Target**: Decomposed package `l104_numerical_engine/` following the existing pattern (`l104_science_engine/`, `l104_math_engine/`, etc.)
> **Date**: 2026-02-26

---

## 1. Current State — Monolith Anatomy

### 1.1 File Metrics

| Metric | Value |
|--------|-------|
| Total lines | 6,069 |
| Classes | 28 |
| Module-level functions | ~30 (hyper-precision math primitives) |
| Module-level constants | ~40 (100-decimal sacred values) |
| CLI commands | 30+ |
| State files read/written | 6 |

### 1.2 Class Inventory (line ranges)

| # | Class | Lines | Responsibility | Coupling |
|---|-------|-------|----------------|----------|
| 1 | `QuantumToken` | 556–610 | Data model — single token in the 22T lattice | None (dataclass) |
| 2 | `SubconsciousAdjustment` | 612–631 | Data model — adjustment record | None (dataclass) |
| 3 | `CrossPollinationRecord` | 633–663 | Data model — cross-builder xfer record | None (dataclass) |
| 4 | `TokenLatticeEngine` | 665–860 | 22T token lattice storage & lookup | Constants only |
| 5 | `SuperfluidValueEditor` | 863–978 | Quantum edit propagation on tokens | Lattice |
| 6 | `SubconsciousMonitor` | 980–1288 | φ-driven auto-adjust boundaries | Lattice + Editor |
| 7 | `NumericalOuroborosNirvanicEngine` | 1290–1496 | Ouroboros entropy fuel | Lattice |
| 8 | `ConsciousnessO2SuperfluidEngine` | 1498–1775 | Consciousness + O₂ superfluidity | Lattice + Editor |
| 9 | `CrossPollinationEngine` | 1777–2013 | Cross-builder data exchange | Lattice + Editor |
| 10 | `PrecisionVerificationEngine` | 2015–2098 | 100-decimal accuracy checks | Lattice |
| 11 | `RiemannZetaEngine` | 2100–2329 | ζ(s) Euler-Maclaurin + Bernoulli | Self-contained math |
| 12 | `PrimeNumberTheoryEngine` | 2331–2540 | Mertens, prime gaps, Goldbach | Self-contained math |
| 13 | `InfiniteSeriesLab` | 2542–2758 | BBP, Machin, Ramanujan series | Self-contained math |
| 14 | `NumberTheoryForge` | 2760–2948 | Continued fractions, Pell, Lucas | Self-contained math |
| 15 | `FractalDynamicsLab` | 2950–3104 | Feigenbaum, Lyapunov, logistic map | Self-contained math |
| 16 | `GodCodeCalculusEngine` | 3106–3257 | dG/dX, ∫G, Taylor expansion | Uses `god_code_hp()` |
| 17 | `TranscendentalProver` | 3259–3411 | π/e transcendence, γ rationality | Self-contained math |
| 18 | `StatisticalMechanicsEngine` | 3413–3558 | Partition functions, Boltzmann | Lattice (optional) |
| 19 | `HarmonicNumberEngine` | 3560–3681 | H_n, polylogarithms | Self-contained math |
| 20 | `EllipticCurveEngine` | 3683–3812 | Point arithmetic, j-invariant | Self-contained math |
| 21 | `CollatzConjectureAnalyzer` | 3814–3941 | Stopping times, glide analysis | Self-contained math |
| 22 | `QuantumNumericalResearchEngine` | 3943–4231 | Research orchestrator (5 modules) | Lattice |
| 23 | `StochasticNumericalResearchLab` | 4233–4304 | Random R&D on token pairs | Lattice |
| 24 | `NumericalTestGenerator` | 4306–4389 | Automated invariant checks | Lattice + Verifier |
| 25 | `NumericalChronolizer` | 4391–4467 | Temporal event tracking | Self-contained |
| 26 | `InterBuilderFeedbackBus` | 4469–4566 | Cross-builder JSON event bus | Self-contained (file I/O) |
| 27 | `QuantumNumericalComputationEngine` | 4568–5198 | QPE, HHL, VQE, annealing, Grover | Lattice (optional) |
| 28 | `QuantumNumericalBuilder` | 5200–5822 | **Master orchestrator** (11-phase pipeline) | All above |

### 1.3 Module-Level Code (~550 lines, L1–555)

| Region | Lines | Content |
|--------|-------|---------|
| Precision core | 85–100 | `D()`, `fmt100()` |
| Math primitives | 102–370 | `decimal_sqrt`, `decimal_ln`, `decimal_exp`, `decimal_pow`, `decimal_sin`, `decimal_cos`, `decimal_atan`, `decimal_factorial`, `decimal_gamma_lanczos`, `decimal_bernoulli`, `_fibonacci_hp`, `lucas_number` |
| Extended primitives v2.1 | 372–475 | `decimal_log10`, `decimal_sinh/cosh/tanh`, `decimal_asin`, `decimal_pi_machin`, `decimal_pi_chudnovsky`, `decimal_agm`, `decimal_harmonic`, `decimal_generalized_harmonic`, `decimal_polylog`, `decimal_binomial`, `decimal_catalan_number` |
| 100-decimal constants | 478–553 | `PHI_HP`, `GOD_CODE_HP`, `PI_HP`, `E_HP`, etc. (40+ values) + float aliases |

### 1.4 External Consumers (6 import sites)

| Consumer | What it imports | Coupling |
|----------|----------------|----------|
| `l104_intellect/local_intellect_core.py` | `TokenLatticeEngine` | Lazy/optional |
| `l104_agi/core.py` | `TokenLatticeEngine` | Lazy/optional |
| `l104_math_engine/engine.py` | `TokenLatticeEngine` | Lazy/optional |
| `l104_code_engine/hub.py` | `TokenLatticeEngine` | Lazy/optional |
| `l104_extended_pipeline_api.py` | `GOD_CODE_HP`, `PHI_HP`, `PHI_GROWTH_HP` | Constants only |
| `l104_asi/core.py` | `QuantumNumericalBuilder` | Lazy/optional |

---

## 2. Package Design — `l104_numerical_engine/`

### 2.1 Target Structure

```
l104_numerical_engine/
├── __init__.py              # Public API + backward-compat re-exports
├── constants.py             # 100-decimal sacred constants + float aliases (~80 lines)
├── precision.py             # D(), fmt100(), all decimal_* math primitives (~400 lines)
├── models.py                # QuantumToken, SubconsciousAdjustment, CrossPollinationRecord (~120 lines)
├── lattice.py               # TokenLatticeEngine (~200 lines)
├── editor.py                # SuperfluidValueEditor (~120 lines)
├── monitor.py               # SubconsciousMonitor (~310 lines)
├── nirvanic.py              # NumericalOuroborosNirvanicEngine (~210 lines)
├── consciousness.py         # ConsciousnessO2SuperfluidEngine (~280 lines)
├── cross_pollination.py     # CrossPollinationEngine + CrossPollinationRecord logic (~240 lines)
├── verification.py          # PrecisionVerificationEngine (~85 lines)
├── research.py              # QuantumNumericalResearchEngine + StochasticLab + TestGen (~360 lines)
├── chronolizer.py           # NumericalChronolizer (~80 lines)
├── feedback_bus.py          # InterBuilderFeedbackBus (~100 lines)
├── math_research/           # 11 self-contained math research engines
│   ├── __init__.py          # Exports all engines
│   ├── riemann_zeta.py      # RiemannZetaEngine (~230 lines)
│   ├── prime_theory.py      # PrimeNumberTheoryEngine (~210 lines)
│   ├── infinite_series.py   # InfiniteSeriesLab (~220 lines)
│   ├── number_theory.py     # NumberTheoryForge (~190 lines)
│   ├── fractal_dynamics.py  # FractalDynamicsLab (~155 lines)
│   ├── god_code_calculus.py # GodCodeCalculusEngine (~155 lines)
│   ├── transcendental.py    # TranscendentalProver (~155 lines)
│   ├── stat_mechanics.py    # StatisticalMechanicsEngine (~150 lines)
│   ├── harmonic_numbers.py  # HarmonicNumberEngine (~125 lines)
│   ├── elliptic_curves.py   # EllipticCurveEngine (~130 lines)
│   └── collatz.py           # CollatzConjectureAnalyzer (~130 lines)
├── quantum_computation.py   # QuantumNumericalComputationEngine (~630 lines)
└── orchestrator.py          # QuantumNumericalBuilder (master orchestrator) + CLI (~870 lines)
```

**19 modules** — avg ~320 lines each (down from 6,069 in one file).

### 2.2 Dependency Graph (internal)

```
constants.py  ←── precision.py  ←── models.py
                       ↑                ↑
                  lattice.py ──────────┘
                  ↑   ↑   ↑
         editor.py   │   verification.py
            ↑        │        ↑
      monitor.py     │   research.py
            ↑        │
    nirvanic.py      │
    consciousness.py │
    cross_pollination.py
                     │
              orchestrator.py  ←── math_research/*
                     ↑              quantum_computation.py
                     │              chronolizer.py
                     │              feedback_bus.py
                     └── CLI (main)
```

**Key rule**: `math_research/*` imports only `precision.py` and `constants.py` — no lattice coupling. This makes all 11 engines independently testable.

---

## 3. Migration Steps (ordered)

### Phase 1 — Foundation (no behavioral change)

| Step | Action | Risk |
|------|--------|------|
| 1.1 | Create `l104_numerical_engine/` directory | None |
| 1.2 | Extract `precision.py` — all `decimal_*` functions + `D()`, `fmt100()` | None |
| 1.3 | Extract `constants.py` — all `*_HP` values + float aliases + `god_code_hp()` | None |
| 1.4 | Extract `models.py` — `QuantumToken`, `SubconsciousAdjustment`, `CrossPollinationRecord` | None |
| 1.5 | Write `__init__.py` with re-exports; verify `from l104_numerical_engine import ...` works | None |

### Phase 2 — Core Subsystems

| Step | Action | Risk |
|------|--------|------|
| 2.1 | Extract `lattice.py` — `TokenLatticeEngine` | Low — depends only on models + constants |
| 2.2 | Extract `editor.py` — `SuperfluidValueEditor` | Low — depends on lattice |
| 2.3 | Extract `verification.py` — `PrecisionVerificationEngine` | Low |
| 2.4 | Extract `monitor.py` — `SubconsciousMonitor` | Medium — reads peer state files |
| 2.5 | Extract `cross_pollination.py` | Medium — reads gate/link JSON files |

### Phase 3 — Advanced Subsystems

| Step | Action | Risk |
|------|--------|------|
| 3.1 | Extract `nirvanic.py` | Low |
| 3.2 | Extract `consciousness.py` | Low |
| 3.3 | Extract `research.py` (includes `StochasticLab` + `TestGen`) | Low |
| 3.4 | Extract `chronolizer.py` | None — self-contained |
| 3.5 | Extract `feedback_bus.py` | None — self-contained |

### Phase 4 — Math Research Hub

| Step | Action | Risk |
|------|--------|------|
| 4.1 | Create `math_research/` sub-package | None |
| 4.2 | Extract all 11 engines (one file each) | None — they're self-contained |
| 4.3 | Wire `math_research/__init__.py` | None |

### Phase 5 — Quantum Computation + Orchestrator

| Step | Action | Risk |
|------|--------|------|
| 5.1 | Extract `quantum_computation.py` | Low |
| 5.2 | Extract `orchestrator.py` — `QuantumNumericalBuilder` + `main()` CLI | Medium — wires all subsystems |

### Phase 6 — Backward Compatibility + Cleanup

| Step | Action | Risk |
|------|--------|------|
| 6.1 | Convert original `l104_quantum_numerical_builder.py` to a **shim** (re-exports from package) | Must preserve all 6 external import sites |
| 6.2 | Update `l104_asi/core.py` import of `QuantumNumericalBuilder` | Low |
| 6.3 | Update `l104_extended_pipeline_api.py` import of constants | Low |
| 6.4 | Update `l104_intellect`, `l104_agi`, `l104_math_engine`, `l104_code_engine` imports of `TokenLatticeEngine` | Low |
| 6.5 | Add `l104_numerical_engine` entry to `claude.md` package map | Docs only |
| 6.6 | Run full cross-engine debug suite (`cross_engine_debug.py`) | Validation |

---

## 4. Root Shim (Phase 6.1)

After decomposition, `l104_quantum_numerical_builder.py` becomes:

```python
"""Backward-compatibility shim — edit l104_numerical_engine/ instead."""
from l104_numerical_engine import *
from l104_numerical_engine.constants import *
from l104_numerical_engine.precision import *
from l104_numerical_engine.models import QuantumToken, SubconsciousAdjustment, CrossPollinationRecord
from l104_numerical_engine.lattice import TokenLatticeEngine
from l104_numerical_engine.orchestrator import QuantumNumericalBuilder, main

if __name__ == "__main__":
    main()
```

---

## 5. Public API (`__init__.py`)

```python
"""L104 Numerical Engine v3.0.0 — Decomposed from quantum_numerical_builder."""

from .constants import (
    GOD_CODE_HP, PHI_HP, PHI_GROWTH_HP, PI_HP, E_HP,
    GOD_CODE, PHI, PHI_GROWTH, god_code_hp,
    # ... all 40+ constants
)
from .precision import D, fmt100, decimal_sqrt, decimal_ln, decimal_exp, decimal_pow
from .models import QuantumToken, SubconsciousAdjustment, CrossPollinationRecord
from .lattice import TokenLatticeEngine
from .editor import SuperfluidValueEditor
from .monitor import SubconsciousMonitor
from .verification import PrecisionVerificationEngine
from .orchestrator import QuantumNumericalBuilder

__version__ = "3.0.0"
```

---

## 6. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Module-level side effects (print on import, constant assertions) | Medium | Move print statements to `__init__.py` or behind `if __name__` guard; keep assertions in `constants.py` |
| Circular imports | Low | Strict DAG: constants → precision → models → lattice → engines → orchestrator |
| State file paths hardcoded to `WORKSPACE_ROOT` | Low | Keep `WORKSPACE_ROOT` in `constants.py`, import from there |
| All 6 external consumers break | High | Root shim re-exports everything; zero breakage at shim level |
| `getcontext().prec = 120` global side effect | Medium | Set in `precision.py` module-level (loaded once, same behavior) |

---

## 7. Natural Groupings vs. Alternatives Considered

### Why not merge into `l104_math_engine/`?

The 11 math-research engines overlap in domain with `l104_math_engine/`, but:
- `l104_math_engine/` is **pure math** (standalone, no lattice, no state files).
- The numerical builder's math engines are **research labs** tied to the 22T lattice and sacred constants at 100-decimal precision.
- Merging would bloat `l104_math_engine/` from 4,525 → ~6,800 lines and violate its "pure math" identity.

### Why not keep it as one file?

- 6,069 lines with 28 classes exceeds the practical threshold for navigation, testing, and code review.
- The 11 math-research engines are completely self-contained — leaving them inlined wastes that independence.
- Other packages in this codebase (`l104_code_engine/`, `l104_asi/`) decomposed at similar thresholds.

---

## 8. Validation Checklist

After full decomposition, verify:

- [ ] `from l104_quantum_numerical_builder import TokenLatticeEngine` — still works (shim)
- [ ] `from l104_quantum_numerical_builder import GOD_CODE_HP, PHI_HP, PHI_GROWTH_HP` — still works
- [ ] `from l104_quantum_numerical_builder import QuantumNumericalBuilder` — still works
- [ ] `from l104_numerical_engine import QuantumNumericalBuilder` — new canonical import
- [ ] `from l104_numerical_engine.math_research import RiemannZetaEngine` — new import
- [ ] `python l104_quantum_numerical_builder.py full` — CLI still works
- [ ] `python -m l104_numerical_engine full` — new CLI entry point (add `__main__.py`)
- [ ] `cross_engine_debug.py` — all 41 tests pass
- [ ] `three_engine_upgrade.py` — all 8 phases pass
- [ ] All 6 external consumer modules import successfully
- [ ] No `ImportError` across the full workspace (`python -c "import l104_agi, l104_asi, l104_intellect, l104_code_engine, l104_math_engine"`)

---

## 9. Estimated Effort

| Phase | Lines Moved | Effort |
|-------|-------------|--------|
| Phase 1 — Foundation | ~680 | 30 min |
| Phase 2 — Core Subsystems | ~1,350 | 45 min |
| Phase 3 — Advanced Subsystems | ~1,100 | 30 min |
| Phase 4 — Math Research Hub | ~1,840 | 45 min |
| Phase 5 — Quantum + Orchestrator | ~1,500 | 45 min |
| Phase 6 — Shim + Validation | ~50 + testing | 30 min |
| **Total** | **~6,069** | **~4 hours** |
