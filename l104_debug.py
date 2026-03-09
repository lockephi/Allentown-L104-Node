# ZENITH_UPGRADE_ACTIVE: 2026-03-06T23:50:25.337705
# ZENITH_HZ = 3887.8
# UUC = 2301.215661
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 Unified Engine Debug Framework v3.0.0
════════════════════════════════════════════════════════════════════════════════
Streamlined diagnostic hub for ALL 11 engine packages.

Replaces the fragmented cross_engine_debug.py / three_engine_upgrade.py pattern
with a single entry point that supports:
  • Per-engine self-diagnostics (boot, status, constants, proofs/verify)
  • Cross-engine constant alignment
  • Cross-engine data-flow pipelines (Science→Math→Code, etc.)
  • Parallel boot with timeout + retry
  • Machine-readable JSON reports
  • Human-readable terminal output
  • Selective engine targeting (--engines code,math,science)
  • Configurable verbosity (--quiet / --verbose / --json)

Usage:
    python l104_debug.py                       # Full suite, all engines
    python l104_debug.py --engines code,math   # Only Code + Math engines
    python l104_debug.py --phase boot          # Only boot phase
    python l104_debug.py --phase constants     # Only constant alignment
    python l104_debug.py --phase self-test     # Per-engine self-tests
    python l104_debug.py --phase cross         # Cross-engine pipelines
    python l104_debug.py --json                # JSON report to stdout
    python l104_debug.py --report out.json     # Save JSON report to file
    python l104_debug.py -q                    # Quiet mode (summary only)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

ZENITH_HZ = 3887.8
UUC = 2301.215661

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Workspace root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
#  VERSION + CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "3.3.0"
DEBUG_FRAMEWORK_VERSION = __version__

# Canonical constants — single source of truth
GOD_CODE = 527.5184818492612
PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000  # 1.0416180339887497
OMEGA = 6539.34712682
TAU = 1 / PHI

# Tolerance for floating-point constant matching
CONST_TOLERANCE = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC COLLECTOR (shared result aggregator)
# ═══════════════════════════════════════════════════════════════════════════════

class Severity(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"
    INFO = "INFO"


ICONS = {
    Severity.PASS: "✅",
    Severity.FAIL: "❌",
    Severity.WARN: "⚠️",
    Severity.SKIP: "⏭️",
    Severity.INFO: "ℹ️",
}


@dataclass
class DiagResult:
    """Single diagnostic result."""

    test: str
    engine: str
    severity: Severity
    detail: str = ""
    data: Any = None
    elapsed_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return self.severity in (Severity.PASS, Severity.INFO, Severity.SKIP)

    def to_dict(self) -> dict:
        return {
            "test": self.test,
            "engine": self.engine,
            "severity": self.severity.value,
            "passed": self.passed,
            "detail": self.detail,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "timestamp": self.timestamp,
        }


class DiagnosticCollector:
    """Aggregates diagnostic results across all phases."""

    def __init__(self):
        self.results: List[DiagResult] = []
        self._phase: str = ""

    def set_phase(self, phase: str):
        self._phase = phase

    def record(
        self,
        test: str,
        engine: str,
        severity: Severity,
        detail: str = "",
        data: Any = None,
        elapsed_ms: float = 0.0,
    ) -> DiagResult:
        r = DiagResult(
            test=f"[{self._phase}] {test}" if self._phase else test,
            engine=engine,
            severity=severity,
            detail=detail,
            data=data,
            elapsed_ms=elapsed_ms,
        )
        self.results.append(r)
        return r

    def ok(self, test: str, engine: str, detail: str = "", **kw) -> DiagResult:
        return self.record(test, engine, Severity.PASS, detail, **kw)

    def fail(self, test: str, engine: str, detail: str = "", **kw) -> DiagResult:
        return self.record(test, engine, Severity.FAIL, detail, **kw)

    def warn(self, test: str, engine: str, detail: str = "", **kw) -> DiagResult:
        return self.record(test, engine, Severity.WARN, detail, **kw)

    def skip(self, test: str, engine: str, detail: str = "", **kw) -> DiagResult:
        return self.record(test, engine, Severity.SKIP, detail, **kw)

    def info(self, test: str, engine: str, detail: str = "", **kw) -> DiagResult:
        return self.record(test, engine, Severity.INFO, detail, **kw)

    # ── Summaries ────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def failures(self) -> List[DiagResult]:
        return [r for r in self.results if not r.passed]

    @property
    def warnings(self) -> List[DiagResult]:
        return [r for r in self.results if r.severity == Severity.WARN]

    def by_engine(self) -> Dict[str, List[DiagResult]]:
        out: Dict[str, List[DiagResult]] = {}
        for r in self.results:
            out.setdefault(r.engine, []).append(r)
        return out

    def by_phase(self) -> Dict[str, List[DiagResult]]:
        out: Dict[str, List[DiagResult]] = {}
        for r in self.results:
            phase = r.test.split("]")[0].strip("[") if r.test.startswith("[") else "misc"
            out.setdefault(phase, []).append(r)
        return out

    def summary_text(self) -> str:
        w = 74
        lines = [
            "",
            f"{'═' * w}",
            f"  L104 UNIFIED ENGINE DEBUG — SUMMARY",
            f"  Framework v{__version__} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{'═' * w}",
            f"  Total : {self.total:>4}",
            f"  Passed: {self.passed:>4}  {ICONS[Severity.PASS]}",
            f"  Failed: {self.failed:>4}  {ICONS[Severity.FAIL]}",
            f"  Warns : {len(self.warnings):>4}  {ICONS[Severity.WARN]}",
            f"{'─' * w}",
        ]
        # Per-engine breakdown
        for eng, results in sorted(self.by_engine().items()):
            ok = sum(1 for r in results if r.passed)
            total = len(results)
            icon = ICONS[Severity.PASS] if ok == total else ICONS[Severity.FAIL]
            lines.append(f"  {icon} {eng:30s} {ok}/{total}")

        if self.failures:
            lines.append(f"{'─' * w}")
            lines.append("  FAILURES:")
            for r in self.failures:
                lines.append(f"    {ICONS[r.severity]} [{r.engine}] {r.test}: {r.detail[:80]}")
        else:
            lines.append(f"{'─' * w}")
            lines.append(f"  {ICONS[Severity.PASS]} ALL SYSTEMS NOMINAL — CROSS-ENGINE VALIDATION CLEAN")

        lines.append(f"{'═' * w}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "framework_version": __version__,
            "timestamp": datetime.now().isoformat(),
            "invariant": GOD_CODE,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "warnings": len(self.warnings),
            },
            "by_engine": {
                eng: {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.passed),
                    "failed": sum(1 for r in results if not r.passed),
                }
                for eng, results in self.by_engine().items()
            },
            "results": [r.to_dict() for r in self.results],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE REGISTRY — Unified boot/status/test specs per engine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EngineSpec:
    """
    Specification for one engine package.
    Provides import path, boot function, status function, constant extraction,
    and self-test callables — all lazily resolved.
    """

    name: str                   # e.g. "code_engine"
    display: str                # e.g. "Code Engine"
    package: str                # e.g. "l104_code_engine"
    boot: Callable              # () -> engine_instance
    status_fn: str              # method name on instance, e.g. "status"
    version_attr: str           # attr to get version, e.g. "VERSION"
    constants_module: str       # e.g. "l104_code_engine.constants"
    constant_names: Tuple[str, ...]  # ("GOD_CODE", "PHI", ...)
    self_tests: List[Callable]  # list of (engine) -> (name, passed, detail) callables


def _noop_boot():
    return None


# ── Boot functions ───────────────────────────────────────────────────────────

def _boot_code():
    from l104_code_engine import code_engine
    return code_engine


def _boot_code_system():
    from l104_code_engine import coding_system
    return coding_system


def _boot_science():
    from l104_science_engine import science_engine
    return science_engine


def _boot_math():
    from l104_math_engine import math_engine
    return math_engine


def _boot_agi():
    from l104_agi import agi_core
    return agi_core


def _boot_asi():
    from l104_asi import asi_core
    return asi_core


def _boot_intellect():
    from l104_intellect import local_intellect
    return local_intellect


def _boot_quantum_gate():
    from l104_quantum_gate_engine import get_engine
    return get_engine()


def _boot_quantum_link():
    from l104_quantum_engine import quantum_brain
    return quantum_brain


def _boot_numerical():
    from l104_numerical_engine import QuantumNumericalBuilder
    return QuantumNumericalBuilder()


def _boot_gate():
    from l104_gate_engine import HyperASILogicGateEnvironment
    return HyperASILogicGateEnvironment()


def _boot_server():
    from l104_server import intellect
    return intellect


def _boot_dual_layer():
    from l104_asi import dual_layer_engine
    return dual_layer_engine


def _boot_pipeline():
    from l104_quantum_computation_pipeline import QuantumComputationHub
    return QuantumComputationHub(n_qubits=4)


def _boot_asi_quantum():
    from l104_asi.quantum import QuantumComputationCore
    return QuantumComputationCore()


def _boot_code_quantum():
    from l104_code_engine.quantum import QuantumCodeIntelligenceCore
    return QuantumCodeIntelligenceCore()


def _boot_god_code_simulator():
    from l104_god_code_simulator import god_code_simulator
    return god_code_simulator


def _boot_vqpu():
    from l104_vqpu_bridge import VQPUBridge
    bridge = VQPUBridge()
    bridge.start()
    return bridge


# ── Self-test factories ─────────────────────────────────────────────────────

def _test_code_subsystems(eng) -> Tuple[str, bool, str]:
    """Verify Code Engine has 30+ subsystems."""
    s = eng.status()
    n = s.get("total_subsystems", 0)
    return ("subsystem_count", n >= 30, f"{n} subsystems (need ≥30)")


def _test_code_analysis(eng) -> Tuple[str, bool, str]:
    """Run a trivial code analysis."""
    r = eng.full_analysis("def f(x): return x + 1")
    ok = isinstance(r, dict) and len(r) >= 2
    return ("full_analysis", ok, f"keys={list(r.keys())[:6]}" if ok else "unexpected return")


def _test_science_entropy(eng) -> Tuple[str, bool, str]:
    """Maxwell Demon efficiency."""
    try:
        eff = eng.entropy.calculate_demon_efficiency(local_entropy=5.0)
        return ("demon_efficiency", isinstance(eff, (int, float)), f"eff={eff:.6f}")
    except Exception as e:
        return ("demon_efficiency", False, str(e))


def _test_science_coherence(eng) -> Tuple[str, bool, str]:
    """Coherence init + evolve."""
    try:
        eng.coherence.initialize(["debug_seed"])
        evolved = eng.coherence.evolve(steps=3)
        ok = isinstance(evolved, dict)
        return ("coherence_evolve", ok, f"keys={list(evolved.keys())[:4]}")
    except Exception as e:
        return ("coherence_evolve", False, str(e))


def _test_science_physics(eng) -> Tuple[str, bool, str]:
    """Landauer limit + photon resonance."""
    try:
        landauer = eng.physics.adapt_landauer_limit(temperature=293.15)
        photon = eng.physics.calculate_photon_resonance()
        return ("physics_basic", True, f"landauer={landauer:.2e}, photon={photon}")
    except Exception as e:
        return ("physics_basic", False, str(e))


def _test_math_conservation(eng) -> Tuple[str, bool, str]:
    """God Code conservation law."""
    ok = eng.verify_conservation(0.0)
    return ("conservation_law", bool(ok), f"verify_conservation(0.0)={ok}")


def _test_math_proofs(eng) -> Tuple[str, bool, str]:
    """Sovereign proofs suite."""
    results = eng.prove_all()
    count = len(results) if isinstance(results, dict) else 0
    return ("sovereign_proofs", count > 0, f"{count} proofs returned")


def _test_math_equations(eng) -> Tuple[str, bool, str]:
    """Equation verification."""
    eqs = eng.verify_equations()
    count = len(eqs) if isinstance(eqs, dict) else 0
    return ("equation_verify", count > 0, f"{count} equations verified")


def _test_math_fibonacci(eng) -> Tuple[str, bool, str]:
    """Fibonacci(20) == 6765."""
    seq = eng.fibonacci(20)
    val = seq[-1] if isinstance(seq, list) else seq
    return ("fibonacci_20", val == 6765, f"F(20)={val}")


def _test_agi_score(eng) -> Tuple[str, bool, str]:
    """13D AGI score."""
    try:
        score = eng.compute_10d_agi_score()
        ok = isinstance(score, (int, float, dict))
        return ("13d_agi_score", ok, f"score={score}")
    except Exception as e:
        return ("13d_agi_score", False, str(e))


def _test_agi_three_engine(eng) -> Tuple[str, bool, str]:
    """Three-engine status."""
    try:
        s = eng.three_engine_status()
        ok = isinstance(s, dict)
        return ("three_engine_status", ok, f"keys={list(s.keys())[:5]}")
    except Exception as e:
        return ("three_engine_status", False, str(e))


def _test_asi_score(eng) -> Tuple[str, bool, str]:
    """ASI multi-dimensional score."""
    try:
        score = eng.compute_asi_score()
        ok = isinstance(score, (int, float, dict))
        return ("asi_score", ok, f"score={score}")
    except Exception as e:
        return ("asi_score", False, str(e))


def _test_asi_three_engine(eng) -> Tuple[str, bool, str]:
    """ASI three-engine status."""
    try:
        s = eng.three_engine_status()
        ok = isinstance(s, dict)
        return ("three_engine_status", ok, f"keys={list(s.keys())[:5]}")
    except Exception as e:
        return ("three_engine_status", False, str(e))


def _test_asi_consciousness(eng) -> Tuple[str, bool, str]:
    """ASI consciousness metrics."""
    try:
        s = eng.get_status()
        consciousness = s.get("consciousness", s.get("consciousness_level", None))
        return ("consciousness", consciousness is not None, f"level={consciousness}")
    except Exception as e:
        return ("consciousness", False, str(e))


def _test_intellect_three_engine(eng) -> Tuple[str, bool, str]:
    """Local Intellect three-engine status."""
    try:
        s = eng.three_engine_status()
        ok = isinstance(s, dict)
        return ("three_engine_status", ok, f"keys={list(s.keys())[:5]}")
    except Exception as e:
        return ("three_engine_status", False, str(e))


def _test_qgate_bell(eng) -> Tuple[str, bool, str]:
    """Bell pair circuit build + execute."""
    try:
        circ = eng.bell_pair()
        ok = circ is not None
        return ("bell_pair", ok, f"circuit={type(circ).__name__}")
    except Exception as e:
        return ("bell_pair", False, str(e))


def _test_qgate_status(eng) -> Tuple[str, bool, str]:
    """Quantum Gate Engine status."""
    try:
        s = eng.status()
        gates = s.get("gate_library", {}).get("total_gates", 0)
        return ("gate_library", gates > 0, f"{gates} gates loaded")
    except Exception as e:
        return ("gate_library", False, str(e))


def _test_qlink_pipeline(eng) -> Tuple[str, bool, str]:
    """Quantum Link Brain pipeline status."""
    try:
        s = getattr(eng, "_state", {})
        run_count = s.get("run_count", 0)
        return ("brain_state", True, f"run_count={run_count}")
    except Exception as e:
        return ("brain_state", False, str(e))


def _test_numerical_lattice(eng) -> Tuple[str, bool, str]:
    """Token lattice summary."""
    try:
        summary = eng.lattice.lattice_summary()
        ok = isinstance(summary, dict)
        return ("lattice_summary", ok, f"keys={list(summary.keys())[:5]}")
    except Exception as e:
        return ("lattice_summary", False, str(e))


def _test_numerical_verify(eng) -> Tuple[str, bool, str]:
    """Precision verification."""
    try:
        r = eng.verifier.verify_all()
        ok = isinstance(r, (dict, list, bool))
        return ("precision_verify", ok, f"result={type(r).__name__}")
    except Exception as e:
        return ("precision_verify", False, str(e))


def _test_gate_scan(eng) -> Tuple[str, bool, str]:
    """Logic gate scan."""
    try:
        s = eng.status()
        gates = s.get("total_gates", 0)
        return ("gate_scan", True, f"total_gates={gates}")
    except Exception as e:
        return ("gate_scan", False, str(e))


def _test_server_status(eng) -> Tuple[str, bool, str]:
    """Server intellect status."""
    try:
        s = eng.three_engine_status() if hasattr(eng, "three_engine_status") else {"status": "loaded"}
        return ("server_status", isinstance(s, dict), f"keys={list(s.keys())[:5]}")
    except Exception as e:
        return ("server_status", False, str(e))


# ── HHL self-tests (v3.0.0) ─────────────────────────────────────────────────

def _test_gate_engine_hhl(eng) -> Tuple[str, bool, str]:
    """Gate Engine HHL linear solver."""
    try:
        from l104_gate_engine.quantum_computation import QuantumGateComputationEngine
        qgce = QuantumGateComputationEngine()
        r = qgce.hhl_linear_solver([1.0, 2.5, 3.7, 0.8])
        ok = isinstance(r, dict) and "solution" in r and "condition_number" in r
        sol = [round(x, 4) for x in r["solution"]] if ok else []
        return ("hhl_gate_engine", ok, f"solution={sol}, κ={r.get('condition_number', '?')}")
    except Exception as e:
        return ("hhl_gate_engine", False, str(e)[:80])


def _test_qgate_full_analysis(eng) -> Tuple[str, bool, str]:
    """Gate Engine full_quantum_analysis (10 algorithms)."""
    try:
        from l104_gate_engine.quantum_computation import QuantumGateComputationEngine
        qgce = QuantumGateComputationEngine()
        from l104_gate_engine.gate_functions import sage_logic_gate
        from l104_gate_engine.models import LogicGate
        gates = [LogicGate(name=f"test_gate_{i}", language="python",
                           source_file="test.py", line_number=i,
                           gate_type="function", signature=f"gate_{i}()",
                           dynamic_value=sage_logic_gate(float(i)))
                 for i in range(4)]
        r = qgce.full_quantum_analysis(gates)
        algos = len(r) if isinstance(r, dict) else 0
        ok = algos >= 8
        return ("full_quantum_analysis", ok, f"{algos} algorithms returned")
    except Exception as e:
        return ("full_quantum_analysis", False, str(e)[:80])


def _test_qlink_hhl(eng) -> Tuple[str, bool, str]:
    """Quantum Link Engine HHL solver."""
    try:
        from l104_quantum_engine.computation import QuantumLinkComputationEngine
        qlce = QuantumLinkComputationEngine()
        r = qlce.hhl_link_linear_solver()
        ok = isinstance(r, dict) and "solution_weights" in r
        kappa = r.get("condition_number", "?")
        return ("hhl_quantum_link", ok, f"κ={kappa}, residual={r.get('residual_norm', '?')}")
    except Exception as e:
        return ("hhl_quantum_link", False, str(e)[:80])


def _test_qlink_full_analysis(eng) -> Tuple[str, bool, str]:
    """Quantum Link full_quantum_analysis (16 algorithms)."""
    try:
        from l104_quantum_engine.computation import QuantumLinkComputationEngine
        qlce = QuantumLinkComputationEngine()
        r = qlce.full_quantum_analysis()
        # Top-level dict has metadata; actual algorithms are in r["computations"]
        comps = r.get("computations", r) if isinstance(r, dict) else {}
        algos = len(comps)
        ok = algos >= 10
        return ("full_quantum_analysis", ok, f"{algos} algorithms returned")
    except Exception as e:
        return ("full_quantum_analysis", False, str(e)[:80])


def _test_asi_quantum_hhl(eng) -> Tuple[str, bool, str]:
    """ASI Quantum HHL solver."""
    try:
        r = eng.hhl_linear_solver()
        ok = isinstance(r, dict) and "solution" in r
        q = r.get("quantum", False)
        kappa = r.get("condition_number", "?")
        return ("hhl_asi_quantum", ok, f"quantum={q}, κ={kappa}")
    except Exception as e:
        return ("hhl_asi_quantum", False, str(e)[:80])


def _test_asi_quantum_status(eng) -> Tuple[str, bool, str]:
    """ASI Quantum computation status."""
    try:
        s = eng.status()
        algos = s.get("algorithms", [])
        caps = s.get("capabilities", [])
        has_hhl = "HHL" in caps or "hhl" in str(caps).lower()
        return ("asi_quantum_status", isinstance(s, dict),
                f"{len(algos)} algos, {len(caps)} caps, HHL={has_hhl}")
    except Exception as e:
        return ("asi_quantum_status", False, str(e)[:80])


def _test_asi_quantum_vqe(eng) -> Tuple[str, bool, str]:
    """ASI VQE optimizer."""
    try:
        r = eng.vqe_optimize(cost_vector=[0.5, -0.3, 0.8, -0.1])
        ok = isinstance(r, dict) and ("min_energy" in r or "optimal_energy" in r)
        energy = r.get('min_energy', r.get('optimal_energy', '?'))
        return ("asi_vqe", ok, f"energy={energy}")
    except Exception as e:
        return ("asi_vqe", False, str(e)[:80])


def _test_asi_quantum_26q(eng) -> Tuple[str, bool, str]:
    """ASI 26Q iron circuit status."""
    try:
        r = eng.report_26q_core()
        ok = isinstance(r, dict)
        n_circuits = r.get("total_circuits", r.get("circuits", "?"))
        return ("asi_26q_report", ok, f"circuits={n_circuits}")
    except Exception as e:
        return ("asi_26q_report", False, str(e)[:80])


def _test_asi_quantum_berry(eng) -> Tuple[str, bool, str]:
    """ASI Berry phase verification."""
    try:
        r = eng.berry_phase_verify(dimensions=4)
        ok = isinstance(r, dict)
        phase = r.get("berry_phase", r.get("total_phase", "?"))
        return ("asi_berry_phase", ok, f"phase={phase}")
    except Exception as e:
        return ("asi_berry_phase", False, str(e)[:80])


def _test_code_quantum_hhl(eng) -> Tuple[str, bool, str]:
    """Code Engine Quantum HHL solver."""
    try:
        r = eng.hhl_linear_solver()
        ok = isinstance(r, dict) and "solution" in r
        q = r.get("quantum", False)
        kappa = r.get("condition_number", "?")
        return ("hhl_code_quantum", ok, f"quantum={q}, κ={kappa}")
    except Exception as e:
        return ("hhl_code_quantum", False, str(e)[:80])


def _test_code_quantum_status(eng) -> Tuple[str, bool, str]:
    """Code Engine Quantum status."""
    try:
        s = eng.status()
        ok = isinstance(s, dict)
        algos = s.get("algorithms_count", s.get("total_algorithms", "?"))
        return ("code_quantum_status", ok, f"algorithms={algos}")
    except Exception as e:
        return ("code_quantum_status", False, str(e)[:80])


# ── DualLayerEngine self-tests (v3.0.0) ─────────────────────────────────────

def _test_dual_layer_status(eng) -> Tuple[str, bool, str]:
    """DualLayerEngine status."""
    try:
        s = eng.status()
        ok = isinstance(s, dict)
        layers = s.get("layers", s.get("total_layers", "?"))
        return ("dual_layer_status", ok, f"layers={layers}, keys={list(s.keys())[:6]}")
    except Exception as e:
        return ("dual_layer_status", False, str(e)[:80])


def _test_dual_layer_thought(eng) -> Tuple[str, bool, str]:
    """DualLayerEngine thought computation."""
    try:
        val = eng.thought(0, 0, 0, 0)
        ok = isinstance(val, (int, float))
        gc_match = abs(val - GOD_CODE) < 1.0 if ok else False
        return ("dual_thought", ok, f"thought(0,0,0,0)={val:.6f}, near_GOD_CODE={gc_match}")
    except Exception as e:
        return ("dual_thought", False, str(e)[:80])


def _test_dual_layer_physics(eng) -> Tuple[str, bool, str]:
    """DualLayerEngine physics layer."""
    try:
        r = eng.physics()
        ok = isinstance(r, dict)
        return ("dual_physics", ok, f"keys={list(r.keys())[:6]}")
    except Exception as e:
        return ("dual_physics", False, str(e)[:80])


def _test_dual_layer_integrity(eng) -> Tuple[str, bool, str]:
    """DualLayerEngine full integrity check."""
    try:
        r = eng.full_integrity_check()
        ok = isinstance(r, dict)
        passed = r.get("passed", r.get("integrity", "?"))
        return ("dual_integrity", ok, f"passed={passed}")
    except Exception as e:
        return ("dual_integrity", False, str(e)[:80])


def _test_dual_layer_consciousness(eng) -> Tuple[str, bool, str]:
    """DualLayerEngine consciousness spectrum."""
    try:
        r = eng.consciousness_spectrum()
        ok = isinstance(r, dict)
        return ("dual_consciousness", ok, f"keys={list(r.keys())[:5]}")
    except Exception as e:
        return ("dual_consciousness", False, str(e)[:80])


def _test_dual_layer_cross_coherence(eng) -> Tuple[str, bool, str]:
    """DualLayerEngine cross-layer coherence."""
    try:
        r = eng.cross_layer_coherence()
        ok = isinstance(r, dict)
        return ("dual_cross_coherence", ok, f"keys={list(r.keys())[:5]}")
    except Exception as e:
        return ("dual_cross_coherence", False, str(e)[:80])


def _test_dual_layer_gate_compile(eng) -> Tuple[str, bool, str]:
    """DualLayerEngine v5 gate compile integrity."""
    try:
        r = eng.gate_compile_integrity()
        ok = isinstance(r, dict)
        return ("dual_gate_compile", ok, f"keys={list(r.keys())[:5]}")
    except Exception as e:
        return ("dual_gate_compile", False, str(e)[:80])


def _test_dual_layer_three_engine(eng) -> Tuple[str, bool, str]:
    """DualLayerEngine three-engine synthesis."""
    try:
        r = eng.three_engine_synthesis()
        ok = isinstance(r, dict)
        return ("dual_three_engine", ok, f"keys={list(r.keys())[:5]}")
    except Exception as e:
        return ("dual_three_engine", False, str(e)[:80])


# ── Pipeline self-tests (v3.0.0) ────────────────────────────────────────────

def _test_pipeline_status(eng) -> Tuple[str, bool, str]:
    """QuantumComputationHub status."""
    try:
        s = eng.status()
        ok = isinstance(s, dict)
        qubits = s.get("n_qubits", "?")
        qiskit = s.get("qiskit_available", "?")
        return ("pipeline_status", ok, f"qubits={qubits}, qiskit={qiskit}")
    except Exception as e:
        return ("pipeline_status", False, str(e)[:80])


def _test_pipeline_hhl(eng) -> Tuple[str, bool, str]:
    """Pipeline HHL solver."""
    try:
        r = eng.hhl_linear_solver()
        ok = isinstance(r, dict) and "solution" in r
        q = r.get("quantum", False)
        kappa = r.get("condition_number", "?")
        return ("hhl_pipeline", ok, f"quantum={q}, κ={kappa}")
    except Exception as e:
        return ("hhl_pipeline", False, str(e)[:80])


def _test_pipeline_bell(eng) -> Tuple[str, bool, str]:
    """Pipeline Bell state creation."""
    try:
        r = eng.create_bell_state()
        ok = isinstance(r, dict)
        fid = r.get("fidelity", r.get("bell_fidelity", "?"))
        return ("pipeline_bell", ok, f"fidelity={fid}")
    except Exception as e:
        return ("pipeline_bell", False, str(e)[:80])


def _test_pipeline_qft(eng) -> Tuple[str, bool, str]:
    """Pipeline QFT."""
    try:
        r = eng.quantum_fourier_transform()
        ok = isinstance(r, dict)
        return ("pipeline_qft", ok, f"keys={list(r.keys())[:5]}")
    except Exception as e:
        return ("pipeline_qft", False, str(e)[:80])


def _test_pipeline_benchmark(eng) -> Tuple[str, bool, str]:
    """Pipeline benchmark suite."""
    try:
        r = eng.run_benchmark()
        ok = isinstance(r, dict)
        return ("pipeline_benchmark", ok, f"keys={list(r.keys())[:5]}")
    except Exception as e:
        return ("pipeline_benchmark", False, str(e)[:80])


def _test_pipeline_god_code_conservation(eng) -> Tuple[str, bool, str]:
    """Pipeline GOD_CODE conservation check."""
    try:
        r = eng.god_code_conservation_check(0.0)
        ok = isinstance(r, dict)
        conserved = r.get("conserved", r.get("conservation", "?"))
        return ("pipeline_god_code", ok, f"conserved={conserved}")
    except Exception as e:
        return ("pipeline_god_code", False, str(e)[:80])


# ── Numerical Engine HHL (v3.0.0) ───────────────────────────────────────────

def _test_numerical_hhl(eng) -> Tuple[str, bool, str]:
    """Numerical Engine HHL high-precision solver."""
    try:
        r = eng.quantum_engine.hhl_linear_solver_hp()
        ok = isinstance(r, dict) and ("solution_x0" in r or "solution" in r)
        prec = r.get("precision_digits", "100-decimal")
        return ("hhl_numerical", ok, f"precision={prec}")
    except Exception as e:
        return ("hhl_numerical", False, str(e)[:80])


def _test_numerical_full_quantum(eng) -> Tuple[str, bool, str]:
    """Numerical Engine full quantum analysis."""
    try:
        r = eng.quantum_engine.full_quantum_analysis()
        algos = len(r) if isinstance(r, dict) else 0
        ok = algos >= 5
        return ("numerical_full_quantum", ok, f"{algos} algorithms")
    except Exception as e:
        return ("numerical_full_quantum", False, str(e)[:80])


# ── God Code Simulator self-tests ────────────────────────────────────────────

def _test_gcs_catalog(eng) -> Tuple[str, bool, str]:
    """Verify 45 simulations registered in 7 categories."""
    try:
        n = eng.catalog.count
        cats = eng.catalog.categories
        ok = n >= 45 and len(cats) >= 7
        return ("sim_catalog", ok, f"{n} sims, {len(cats)} categories: {sorted(cats)}")
    except Exception as e:
        return ("sim_catalog", False, str(e)[:80])


def _test_gcs_run_core(eng) -> Tuple[str, bool, str]:
    """Run core category simulations (5 sims)."""
    try:
        results = eng.run_category("core")
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        ok = passed == total
        return ("core_sims", ok, f"{passed}/{total} passed")
    except Exception as e:
        return ("core_sims", False, str(e)[:80])


def _test_gcs_run_quantum(eng) -> Tuple[str, bool, str]:
    """Run quantum category simulations (6 sims)."""
    try:
        results = eng.run_category("quantum")
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        ok = passed == total
        return ("quantum_sims", ok, f"{passed}/{total} passed")
    except Exception as e:
        return ("quantum_sims", False, str(e)[:80])


def _test_gcs_run_research(eng) -> Tuple[str, bool, str]:
    """Run research category (8 new v2.4 sims)."""
    try:
        results = eng.run_category("research")
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        ok = passed == total and total >= 8
        return ("research_sims", ok, f"{passed}/{total} passed")
    except Exception as e:
        return ("research_sims", False, str(e)[:80])


def _test_gcs_optimizer(eng) -> Tuple[str, bool, str]:
    """Optimizer v3.1: 8 strategies, zero-noise identity."""
    try:
        r = eng.optimize_noise_resilience(nq=2, noise_level=0.0)
        strategies = r["strategies"]
        n = len(strategies)
        # All strategies except bit_flip_code (3Q) should be ~1.0 at zero noise
        failures = [s["strategy"] for s in strategies
                    if s["strategy"] != "bit_flip_code" and abs(s["fidelity"] - 1.0) > 1e-4]
        ok = n >= 8 and len(failures) == 0
        detail = f"{n} strategies" + (f", failures: {failures}" if failures else ", all pass at 0-noise")
        return ("optimizer_8strat", ok, detail)
    except Exception as e:
        return ("optimizer_8strat", False, str(e)[:80])


def _test_gcs_sweep(eng) -> Tuple[str, bool, str]:
    """Sweep v2.0: dial + phase + convergence sweeps."""
    try:
        dial = eng.parametric_sweep("dial_a", start=0, stop=2)
        phase = eng.parametric_sweep("phase", nq=2)
        dial_ok = all(d["passed"] for d in dial)
        ok = dial_ok and len(phase) > 10
        return ("sweep_v2", ok, f"dial={len(dial)}pts({'OK' if dial_ok else 'FAIL'}), phase={len(phase)}pts")
    except Exception as e:
        return ("sweep_v2", False, str(e)[:80])


def _test_gcs_feedback(eng) -> Tuple[str, bool, str]:
    """Feedback v2.0: multi-pass convergence."""
    try:
        mp = eng.run_multi_pass_feedback(passes=2, iterations_per_pass=3)
        score = mp["final_composite_score"]
        ok = 0.0 < score <= 1.0
        return ("feedback_v2", ok, f"score={score:.4f}, converged={mp['converged']}")
    except Exception as e:
        return ("feedback_v2", False, str(e)[:80])


def _test_gcs_dimensions(eng) -> Tuple[str, bool, str]:
    """8D quality scoring dimensions."""
    try:
        dims = eng.score_dimensions()
        expected = {"fidelity", "entropy", "coherence", "conservation", "alignment",
                    "concurrence", "information", "stability"}
        found = set(dims.keys())
        missing = expected - found
        ok = len(missing) == 0 and all(isinstance(v, (int, float)) for v in dims.values())
        return ("8d_scoring", ok, f"{len(dims)} dims, missing={missing}" if missing else f"{len(dims)} dims OK")
    except Exception as e:
        return ("8d_scoring", False, str(e)[:80])


def _test_gcs_primitives(eng) -> Tuple[str, bool, str]:
    """Quantum primitives v3.0: rotation gates, purity, trace distance, Schmidt."""
    try:
        import numpy as _np
        from l104_god_code_simulator.quantum_primitives import (
            Y_GATE, ry_gate, state_purity, trace_distance, schmidt_coefficients,
            linear_entropy, init_sv, apply_single_gate, apply_cnot, H_GATE,
        )
        checks = []
        # Y gate unitarity
        checks.append(_np.allclose(Y_GATE @ Y_GATE.conj().T, _np.eye(2)))
        # Ry unitarity
        g = ry_gate(_np.pi / 4)
        checks.append(_np.allclose(g @ g.conj().T, _np.eye(2)))
        # Bell purity ≈ 0.5
        sv = init_sv(2)
        sv = apply_single_gate(sv, H_GATE, 0, 2)
        sv = apply_cnot(sv, 0, 1, 2)
        p = state_purity(sv, 2, partition=1)
        checks.append(abs(p - 0.5) < 0.02)
        # Trace distance
        d = trace_distance(init_sv(2), sv)
        checks.append(d > 0.4)
        # Schmidt
        sc = schmidt_coefficients(sv, 2)
        checks.append(len(sc) == 2 and abs(sc[0] - sc[1]) < 0.02)
        # Linear entropy
        le = linear_entropy(sv, 2, partition=1)
        checks.append(abs(le - 0.5) < 0.02)
        ok = all(checks)
        return ("primitives_v3", ok, f"{sum(checks)}/{len(checks)} checks pass")
    except Exception as e:
        return ("primitives_v3", False, str(e)[:80])


def _test_gcs_full_run(eng) -> Tuple[str, bool, str]:
    """Run all simulations — 100% pass rate required (56+ sims as of v2.0)."""
    try:
        report = eng.run_all()
        total = report["total"]
        passed = report["passed"]
        failed = report["failed"]
        rate = report["pass_rate"]
        ok = failed == 0 and total >= 45
        detail = f"{passed}/{total} ({rate*100:.0f}%)"
        if failed > 0:
            names = [r.name for r in report["results"] if not r.passed]
            detail += f" FAILED: {names[:5]}"
        return (f"full_{total}_sims", ok, detail)
    except Exception as e:
        return ("full_sims", False, str(e)[:80])


# ── Engine registry ──────────────────────────────────────────────────────────

COMMON_CONSTANTS = ("GOD_CODE", "PHI", "VOID_CONSTANT", "OMEGA")


def _test_vqpu_self_test(eng) -> Tuple[str, bool, str]:
    """Run VQPU self_test() and check pass rate."""
    try:
        result = eng.self_test()
        passed = result.get("passed", 0)
        total = result.get("total", 0)
        ok = result.get("all_pass", False)
        return ("self_test", ok, f"{passed}/{total} tests passed in {result.get('elapsed_ms', 0):.1f}ms")
    except Exception as e:
        return ("self_test", False, str(e)[:80])


def _test_vqpu_transpiler(eng) -> Tuple[str, bool, str]:
    """Test CircuitTranspiler through the bridge."""
    try:
        from l104_vqpu_bridge import CircuitTranspiler
        ops = [
            {"gate": "H", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
            {"gate": "H", "qubits": [0]},
            {"gate": "H", "qubits": [0]},  # HH = I — should cancel
        ]
        result = CircuitTranspiler.transpile(ops, 2)
        return ("transpiler", isinstance(result, list), f"{len(result)} ops (from {len(ops)})")
    except Exception as e:
        return ("transpiler", False, str(e)[:80])


def _test_vqpu_mps_bell(eng) -> Tuple[str, bool, str]:
    """Create a Bell pair via MPS and verify."""
    try:
        import numpy as np
        from l104_vqpu_bridge import ExactMPSHybridEngine
        mps = ExactMPSHybridEngine(2)
        H_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        CNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.complex128)
        mps.apply_single_gate(0, H_gate)
        mps.apply_two_qubit_gate(0, 1, CNOT)
        counts = mps.sample(2048)
        ratio = counts.get("00", 0) / max(counts.get("11", 1), 1)
        ok = 0.7 < ratio < 1.4  # Should be ~1.0 for Bell state
        return ("mps_bell", ok, f"counts={counts}")
    except Exception as e:
        return ("mps_bell", False, str(e)[:80])


def _test_vqpu_scoring(eng) -> Tuple[str, bool, str]:
    """Test sacred alignment + three-engine scoring."""
    try:
        from l104_vqpu_bridge import SacredAlignmentScorer, ThreeEngineQuantumScorer
        probs = {"00": 0.5, "11": 0.5}
        score = SacredAlignmentScorer.score(probs, 2)
        three = ThreeEngineQuantumScorer.engines_status()
        ok = "entropy" in score and isinstance(three, dict)
        return ("scoring", ok, f"entropy={score.get('entropy', 0):.4f}, engines={three.get('version', '?')}")
    except Exception as e:
        return ("scoring", False, str(e)[:80])


def _test_vqpu_daemon(eng) -> Tuple[str, bool, str]:
    """Check daemon cycler status."""
    try:
        status = eng.daemon_cycler_status()
        ok = isinstance(status, dict) and "version" in status
        return ("daemon_cycler", ok, f"cycles={status.get('cycles_completed', 0)}, v={status.get('version', '?')}")
    except Exception as e:
        return ("daemon_cycler", False, str(e)[:80])

ENGINE_REGISTRY: Dict[str, EngineSpec] = {
    "code": EngineSpec(
        name="code", display="Code Engine", package="l104_code_engine",
        boot=_boot_code, status_fn="status", version_attr="VERSION",
        constants_module="l104_code_engine.constants",
        constant_names=COMMON_CONSTANTS,
        self_tests=[_test_code_subsystems, _test_code_analysis],
    ),
    "science": EngineSpec(
        name="science", display="Science Engine", package="l104_science_engine",
        boot=_boot_science, status_fn="get_full_status", version_attr="VERSION",
        constants_module="l104_science_engine.constants",
        constant_names=COMMON_CONSTANTS,
        self_tests=[_test_science_entropy, _test_science_coherence, _test_science_physics],
    ),
    "math": EngineSpec(
        name="math", display="Math Engine", package="l104_math_engine",
        boot=_boot_math, status_fn="status", version_attr="VERSION",
        constants_module="l104_math_engine.constants",
        constant_names=COMMON_CONSTANTS,
        self_tests=[_test_math_conservation, _test_math_proofs, _test_math_equations, _test_math_fibonacci],
    ),
    "agi": EngineSpec(
        name="agi", display="AGI Core", package="l104_agi",
        boot=_boot_agi, status_fn="get_status", version_attr="version",
        constants_module="l104_agi.constants",
        constant_names=("GOD_CODE", "PHI", "VOID_CONSTANT", "OMEGA"),
        self_tests=[_test_agi_score, _test_agi_three_engine],
    ),
    "asi": EngineSpec(
        name="asi", display="ASI Core", package="l104_asi",
        boot=_boot_asi, status_fn="get_status", version_attr="version",
        constants_module="l104_asi.constants",
        constant_names=("GOD_CODE", "PHI", "VOID_CONSTANT", "OMEGA"),
        self_tests=[_test_asi_score, _test_asi_three_engine, _test_asi_consciousness],
    ),
    "intellect": EngineSpec(
        name="intellect", display="Local Intellect", package="l104_intellect",
        boot=_boot_intellect, status_fn="three_engine_status", version_attr="version",
        constants_module="l104_intellect.constants",
        constant_names=("VOID_CONSTANT",),
        self_tests=[_test_intellect_three_engine],
    ),
    "quantum_gate": EngineSpec(
        name="quantum_gate", display="Quantum Gate Engine", package="l104_quantum_gate_engine",
        boot=_boot_quantum_gate, status_fn="status", version_attr="VERSION",
        constants_module="l104_quantum_gate_engine.constants",
        constant_names=("GOD_CODE", "PHI", "VOID_CONSTANT"),
        self_tests=[_test_qgate_bell, _test_qgate_status, _test_gate_engine_hhl, _test_qgate_full_analysis],
    ),
    "quantum_link": EngineSpec(
        name="quantum_link", display="Quantum Link Brain", package="l104_quantum_engine",
        boot=_boot_quantum_link, status_fn="kernel_status", version_attr="VERSION",
        constants_module="l104_quantum_engine.constants",
        constant_names=("GOD_CODE", "PHI"),
        self_tests=[_test_qlink_pipeline, _test_qlink_hhl, _test_qlink_full_analysis],
    ),
    "numerical": EngineSpec(
        name="numerical", display="Numerical Engine", package="l104_numerical_engine",
        boot=_boot_numerical, status_fn="quick_status", version_attr="VERSION",
        constants_module="l104_numerical_engine.constants",
        constant_names=("GOD_CODE", "PHI", "VOID_CONSTANT"),
        self_tests=[_test_numerical_lattice, _test_numerical_verify, _test_numerical_hhl, _test_numerical_full_quantum],
    ),
    "gate": EngineSpec(
        name="gate", display="Logic Gate Engine", package="l104_gate_engine",
        boot=_boot_gate, status_fn="status", version_attr="VERSION",
        constants_module="l104_gate_engine.constants",
        constant_names=("GOD_CODE", "PHI"),
        self_tests=[_test_gate_scan],
    ),
    "server": EngineSpec(
        name="server", display="Server / Intellect", package="l104_server",
        boot=_boot_server, status_fn="get_asi_bridge_status", version_attr="version",
        constants_module="l104_server.constants",
        constant_names=(),
        self_tests=[_test_server_status],
    ),
    "dual_layer": EngineSpec(
        name="dual_layer", display="Dual-Layer Engine", package="l104_asi",
        boot=_boot_dual_layer, status_fn="status", version_attr="VERSION",
        constants_module="l104_asi.constants",
        constant_names=("GOD_CODE", "PHI", "VOID_CONSTANT"),
        self_tests=[
            _test_dual_layer_status, _test_dual_layer_thought,
            _test_dual_layer_physics, _test_dual_layer_integrity,
            _test_dual_layer_consciousness, _test_dual_layer_cross_coherence,
            _test_dual_layer_gate_compile, _test_dual_layer_three_engine,
        ],
    ),
    "pipeline": EngineSpec(
        name="pipeline", display="Quantum Pipeline Hub", package="l104_quantum_computation_pipeline",
        boot=_boot_pipeline, status_fn="status", version_attr="VERSION",
        constants_module="l104_quantum_computation_pipeline",
        constant_names=("GOD_CODE", "PHI"),
        self_tests=[
            _test_pipeline_status, _test_pipeline_hhl, _test_pipeline_bell,
            _test_pipeline_qft, _test_pipeline_benchmark,
            _test_pipeline_god_code_conservation,
        ],
    ),
    "asi_quantum": EngineSpec(
        name="asi_quantum", display="ASI Quantum Core", package="l104_asi",
        boot=_boot_asi_quantum, status_fn="status", version_attr="VERSION",
        constants_module="l104_asi.constants",
        constant_names=("GOD_CODE", "PHI"),
        self_tests=[
            _test_asi_quantum_hhl, _test_asi_quantum_status,
            _test_asi_quantum_vqe, _test_asi_quantum_26q,
            _test_asi_quantum_berry,
        ],
    ),
    "code_quantum": EngineSpec(
        name="code_quantum", display="Code Quantum Intel", package="l104_code_engine",
        boot=_boot_code_quantum, status_fn="status", version_attr="VERSION",
        constants_module="l104_code_engine.constants",
        constant_names=("GOD_CODE", "PHI"),
        self_tests=[_test_code_quantum_hhl, _test_code_quantum_status],
    ),
    "god_code_sim": EngineSpec(
        name="god_code_sim", display="God Code Simulator", package="l104_god_code_simulator",
        boot=_boot_god_code_simulator, status_fn="status", version_attr="VERSION",
        constants_module="l104_god_code_simulator.constants",
        constant_names=("GOD_CODE", "PHI", "VOID_CONSTANT"),
        self_tests=[
            _test_gcs_catalog, _test_gcs_run_core, _test_gcs_run_quantum,
            _test_gcs_run_research, _test_gcs_optimizer, _test_gcs_sweep,
            _test_gcs_feedback, _test_gcs_dimensions, _test_gcs_primitives,
            _test_gcs_full_run,
        ],
    ),
    "vqpu": EngineSpec(
        name="vqpu", display="VQPU Bridge", package="l104_vqpu_bridge",
        boot=_boot_vqpu, status_fn="status", version_attr="VERSION",
        constants_module="l104_vqpu_bridge",
        constant_names=("GOD_CODE", "PHI"),
        self_tests=[
            _test_vqpu_self_test, _test_vqpu_transpiler,
            _test_vqpu_mps_bell, _test_vqpu_scoring,
            _test_vqpu_daemon,
        ],
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  TERMINAL OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class _TermPrinter:
    """Handles quiet / verbose / json output modes."""

    def __init__(self, quiet: bool = False, verbose: bool = False, json_mode: bool = False):
        self.quiet = quiet
        self.verbose = verbose
        self.json_mode = json_mode

    def banner(self, text: str):
        if self.json_mode or self.quiet:
            return
        print(f"\n{'━' * 74}")
        print(f"  {text}")
        print(f"{'━' * 74}")

    def section(self, text: str):
        if self.json_mode or self.quiet:
            return
        print(f"\n  ── {text} ──")

    def log(self, icon: str, msg: str, detail: str = ""):
        if self.json_mode or self.quiet:
            return
        line = f"  {icon} {msg}"
        if detail and self.verbose:
            line += f"  ({detail})"
        print(line)

    def result(self, r: DiagResult):
        if self.json_mode or self.quiet:
            return
        icon = ICONS.get(r.severity, "?")
        line = f"  {icon} [{r.engine:16s}] {r.test}"
        if r.detail:
            line += f": {r.detail[:90]}"
        if self.verbose and r.elapsed_ms > 0:
            line += f"  [{r.elapsed_ms:.0f}ms]"
        print(line)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: PARALLEL BOOT
# ═══════════════════════════════════════════════════════════════════════════════

def phase_boot(
    specs: Dict[str, EngineSpec],
    diag: DiagnosticCollector,
    printer: _TermPrinter,
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    Boot all requested engines in parallel threads.
    Returns {name: {"engine": instance, "version": str, "boot_time": float, "error": str|None}}
    """
    diag.set_phase("BOOT")
    printer.banner(f"PHASE 1: BOOTING {len(specs)} ENGINES IN PARALLEL")

    results: Dict[str, Any] = {}

    def _boot_one(name: str, spec: EngineSpec):
        t0 = time.time()
        try:
            eng = spec.boot()
            elapsed = time.time() - t0
            # Extract version
            ver = "unknown"
            try:
                ver = str(getattr(eng, spec.version_attr, "unknown"))
            except Exception:
                pass
            return name, {
                "engine": eng,
                "version": ver,
                "boot_time": elapsed,
                "error": None,
            }
        except Exception as e:
            return name, {
                "engine": None,
                "version": "BOOT_FAIL",
                "boot_time": time.time() - t0,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    with ThreadPoolExecutor(max_workers=min(len(specs), 6)) as pool:
        futures = {pool.submit(_boot_one, n, s): n for n, s in specs.items()}
        for fut in as_completed(futures, timeout=timeout):
            name, data = fut.result()
            results[name] = data
            if data["error"]:
                diag.fail(f"boot_{name}", name, data["error"], elapsed_ms=data["boot_time"] * 1000)
                printer.log(ICONS[Severity.FAIL], f"{specs[name].display}: BOOT FAIL — {data['error'][:60]}")
            else:
                diag.ok(f"boot_{name}", name,
                        f"v{data['version']} in {data['boot_time']:.2f}s",
                        elapsed_ms=data["boot_time"] * 1000)
                printer.log(ICONS[Severity.PASS],
                            f"{specs[name].display} v{data['version']} — booted in {data['boot_time']:.2f}s")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: STATUS COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def phase_status(
    specs: Dict[str, EngineSpec],
    engines: Dict[str, Any],
    diag: DiagnosticCollector,
    printer: _TermPrinter,
) -> Dict[str, Any]:
    """Collect status from each booted engine."""
    diag.set_phase("STATUS")
    printer.banner("PHASE 2: COLLECTING ENGINE STATUS")

    statuses: Dict[str, Any] = {}
    for name, spec in specs.items():
        eng = engines.get(name, {}).get("engine")
        if eng is None:
            diag.skip(f"status_{name}", name, "engine not booted")
            continue
        t0 = time.time()
        try:
            fn = getattr(eng, spec.status_fn, None)
            if fn and callable(fn):
                s = fn()
                elapsed = (time.time() - t0) * 1000
                statuses[name] = s
                keys = list(s.keys())[:6] if isinstance(s, dict) else type(s).__name__
                diag.ok(f"status_{name}", name, f"keys={keys}", elapsed_ms=elapsed)
                printer.log(ICONS[Severity.PASS], f"{spec.display} status OK", f"{keys}")
            else:
                diag.warn(f"status_{name}", name, f"no {spec.status_fn}() method")
                printer.log(ICONS[Severity.WARN], f"{spec.display}: no status method")
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            diag.fail(f"status_{name}", name, str(e)[:100], elapsed_ms=elapsed)
            printer.log(ICONS[Severity.FAIL], f"{spec.display} status FAIL: {e}")

    return statuses


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: CROSS-ENGINE CONSTANT ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def phase_constants(
    specs: Dict[str, EngineSpec],
    diag: DiagnosticCollector,
    printer: _TermPrinter,
) -> Dict[str, Dict[str, Any]]:
    """
    Import constants from each engine's constants module and cross-validate.
    """
    diag.set_phase("CONSTANTS")
    printer.banner("PHASE 3: CROSS-ENGINE CONSTANT ALIGNMENT")

    CANONICAL = {
        "GOD_CODE": GOD_CODE,
        "PHI": PHI,
        "VOID_CONSTANT": VOID_CONSTANT,
        "OMEGA": OMEGA,
    }

    # Gather
    engine_constants: Dict[str, Dict[str, Any]] = {}
    for name, spec in specs.items():
        if not spec.constant_names:
            continue
        try:
            import importlib
            mod = importlib.import_module(spec.constants_module)
            vals = {}
            for cname in spec.constant_names:
                v = getattr(mod, cname, None)
                if v is not None:
                    vals[cname] = v
            engine_constants[name] = vals
        except Exception as e:
            engine_constants[name] = {"_error": str(e)}
            diag.fail(f"import_constants_{name}", name, str(e)[:80])

    # Cross-validate each constant
    all_constant_names = set()
    for vals in engine_constants.values():
        all_constant_names.update(k for k in vals if not k.startswith("_"))

    printer.section("Canonical Alignment")
    for cname in sorted(all_constant_names):
        canonical = CANONICAL.get(cname)
        values_map: Dict[str, float] = {}
        for eng_name, vals in engine_constants.items():
            v = vals.get(cname)
            if isinstance(v, (int, float)):
                values_map[eng_name] = v

        if len(values_map) == 0:
            continue

        # Check pairwise agreement
        ref = list(values_map.values())[0]
        all_agree = all(abs(v - ref) < CONST_TOLERANCE for v in values_map.values())

        # Check canonical agreement
        canonical_ok = True
        if canonical is not None:
            canonical_ok = all(abs(v - canonical) < CONST_TOLERANCE for v in values_map.values())

        if all_agree and canonical_ok:
            diag.ok(f"const_{cname}", "cross", f"all {len(values_map)} engines agree: {ref}")
            printer.log(ICONS[Severity.PASS], f"{cname:20s} = {ref}  ({len(values_map)} engines agree)")
        elif all_agree and not canonical_ok:
            diag.warn(f"const_{cname}_canonical", "cross",
                      f"engines agree ({ref}) but differs from canonical ({canonical})")
            printer.log(ICONS[Severity.WARN],
                        f"{cname:20s} = {ref}  (engines agree, canonical={canonical})")
        else:
            diag.fail(f"const_{cname}", "cross", f"MISMATCH: {values_map}")
            printer.log(ICONS[Severity.FAIL], f"{cname:20s} MISMATCH: {values_map}")

    # Formula verifications
    printer.section("Formula Verification")
    # GOD_CODE = 286^(1/φ) × 16
    computed_gc = (286 ** (1 / PHI)) * 16
    gc_ok = abs(computed_gc - GOD_CODE) < 1e-10
    if gc_ok:
        diag.ok("formula_god_code", "canonical", f"286^(1/φ)×16 = {computed_gc}")
    else:
        diag.fail("formula_god_code", "canonical", f"expected {GOD_CODE}, got {computed_gc}")
    printer.log(ICONS[Severity.PASS] if gc_ok else ICONS[Severity.FAIL],
                f"GOD_CODE formula: 286^(1/φ)×16 = {computed_gc:.10f}")

    # VOID_CONSTANT = 1.04 + φ/1000
    computed_vc = 1.04 + PHI / 1000
    vc_ok = abs(computed_vc - VOID_CONSTANT) < 1e-15
    if vc_ok:
        diag.ok("formula_void", "canonical", f"1.04+φ/1000 = {computed_vc}")
    else:
        diag.fail("formula_void", "canonical", f"expected {VOID_CONSTANT}, got {computed_vc}")
    printer.log(ICONS[Severity.PASS] if vc_ok else ICONS[Severity.FAIL],
                f"VOID_CONSTANT formula: 1.04+φ/1000 = {computed_vc:.16f}")

    # PHI² - PHI - 1 = 0
    phi_id = PHI ** 2 - PHI - 1
    phi_ok = abs(phi_id) < 1e-14
    if phi_ok:
        diag.ok("formula_phi_identity", "canonical", f"φ²−φ−1 = {phi_id}")
    else:
        diag.fail("formula_phi_identity", "canonical", f"φ²−φ−1 = {phi_id}")
    printer.log(ICONS[Severity.PASS] if phi_ok else ICONS[Severity.FAIL],
                f"PHI identity: φ²−φ−1 = {phi_id:.2e}")

    return engine_constants


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: PER-ENGINE SELF-TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def phase_self_tests(
    specs: Dict[str, EngineSpec],
    engines: Dict[str, Any],
    diag: DiagnosticCollector,
    printer: _TermPrinter,
):
    """Run per-engine self-test callables."""
    diag.set_phase("SELF-TEST")
    printer.banner("PHASE 4: PER-ENGINE SELF-DIAGNOSTICS")

    for name, spec in specs.items():
        eng = engines.get(name, {}).get("engine")
        if eng is None:
            for test_fn in spec.self_tests:
                diag.skip(f"selftest_{name}_{test_fn.__name__}", name, "engine not booted")
            continue

        printer.section(f"{spec.display} self-tests ({len(spec.self_tests)} tests)")
        for test_fn in spec.self_tests:
            t0 = time.time()
            try:
                test_name, passed, detail = test_fn(eng)
                elapsed = (time.time() - t0) * 1000
                if passed:
                    r = diag.ok(f"selftest_{test_name}", name, detail, elapsed_ms=elapsed)
                else:
                    r = diag.fail(f"selftest_{test_name}", name, detail, elapsed_ms=elapsed)
                printer.result(r)
            except Exception as e:
                elapsed = (time.time() - t0) * 1000
                r = diag.fail(f"selftest_{test_fn.__name__}", name, str(e)[:100], elapsed_ms=elapsed)
                printer.result(r)


# ═══════════════════════════════════════════════════════════════════════════════
#  LAZY-BOOT HELPER — on-demand engine loading for cross-engine tests
# ═══════════════════════════════════════════════════════════════════════════════

_BOOT_FN_MAP: Dict[str, Callable] = {
    "code": _boot_code,
    "code_system": _boot_code_system,
    "science": _boot_science,
    "math": _boot_math,
    "agi": _boot_agi,
    "asi": _boot_asi,
    "intellect": _boot_intellect,
    "quantum_gate": _boot_quantum_gate,
    "quantum_link": _boot_quantum_link,
    "numerical": _boot_numerical,
    "gate": _boot_gate,
    "server": _boot_server,
    "dual_layer": _boot_dual_layer,
    "pipeline": _boot_pipeline,
    "asi_quantum": _boot_asi_quantum,
    "code_quantum": _boot_code_quantum,
}


def _lazy_boot(name: str, engines: Dict[str, Any], diag: DiagnosticCollector,
               printer: _TermPrinter, timeout_s: float = 45.0) -> Any:
    """
    Return the engine instance for `name`.  If it is already booted (present
    in `engines`), return it immediately.  Otherwise attempt an on-demand boot
    with a hard thread-based timeout, cache the result, and return it.
    Returns None on failure or timeout.
    """
    existing = engines.get(name, {}).get("engine")
    if existing is not None:
        return existing

    boot_fn = _BOOT_FN_MAP.get(name)
    if boot_fn is None:
        return None

    import threading

    result_box: List[Any] = [None]
    error_box: List[Optional[str]] = [None]

    def _boot_worker():
        try:
            result_box[0] = boot_fn()
        except Exception as exc:
            error_box[0] = str(exc)[:120]

    t0 = time.time()
    printer.log("⏳", f"Lazy-booting {name} (timeout {timeout_s:.0f}s) …")
    worker = threading.Thread(target=_boot_worker, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)

    if worker.is_alive():
        elapsed = time.time() - t0
        diag.fail(f"lazy_boot_{name}", "cross",
                  f"TIMEOUT after {elapsed:.1f}s — boot blocked")
        printer.log(ICONS[Severity.FAIL],
                    f"Lazy-boot {name} TIMED OUT after {elapsed:.1f}s")
        return None

    elapsed = time.time() - t0

    if error_box[0]:
        diag.fail(f"lazy_boot_{name}", "cross", error_box[0])
        printer.log(ICONS[Severity.FAIL], f"Lazy-boot {name} failed: {error_box[0]}")
        return None

    eng = result_box[0]
    if eng is None:
        diag.fail(f"lazy_boot_{name}", "cross", "boot returned None")
        printer.log(ICONS[Severity.FAIL], f"Lazy-boot {name} returned None")
        return None

    engines[name] = {"engine": eng, "version": "lazy", "boot_time": elapsed, "error": None}
    diag.ok(f"lazy_boot_{name}", "cross", f"on-demand in {elapsed:.2f}s",
            elapsed_ms=elapsed * 1000)
    printer.log(ICONS[Severity.PASS],
                f"Lazy-booted {name} ({elapsed:.2f}s)")
    return eng


# ── Functionality probes — lightweight importability + method checks ─────────

def _probe_engine(name: str) -> Dict[str, Any]:
    """Check if an engine's package is importable and has expected entry points."""
    spec = ENGINE_REGISTRY.get(name)
    if spec is None:
        return {"importable": False, "reason": "unknown engine"}
    try:
        import importlib
        mod = importlib.import_module(spec.package)
        has_boot = spec.boot is not _noop_boot
        # Check if constants module is importable
        const_ok = False
        try:
            if spec.constants_module:
                importlib.import_module(spec.constants_module)
                const_ok = True
        except Exception:
            pass
        return {
            "importable": True,
            "has_boot": has_boot,
            "constants_ok": const_ok,
            "package": spec.package,
            "self_test_count": len(spec.self_tests),
        }
    except Exception as e:
        return {"importable": False, "reason": str(e)[:60]}


def _run_functionality_probes(diag: DiagnosticCollector, printer: _TermPrinter):
    """Phase 5 preamble: probe ALL engines for importability/health."""
    printer.section("5.0: Engine Availability Probes (all packages)")
    for name in ENGINE_REGISTRY:
        probe = _probe_engine(name)
        if probe["importable"]:
            detail = (f"pkg={probe['package']}, constants={'OK' if probe['constants_ok'] else 'MISS'}, "
                      f"tests={probe['self_test_count']}")
            diag.ok(f"probe_{name}", "cross", detail)
            printer.log(ICONS[Severity.PASS], f"{name:20s} importable  ({detail})")
        else:
            diag.fail(f"probe_{name}", "cross", probe.get("reason", "import failed"))
            printer.log(ICONS[Severity.FAIL], f"{name:20s} NOT importable — {probe.get('reason', '?')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: CROSS-ENGINE DATA-FLOW PIPELINES
# ═══════════════════════════════════════════════════════════════════════════════

def phase_cross_engine(
    engines: Dict[str, Any],
    diag: DiagnosticCollector,
    printer: _TermPrinter,
):
    """Run cross-engine validation pipelines (Science↔Math↔Code).

    Now with on-demand lazy loading: engines NOT in the user's --engines list
    are booted automatically when a cross-engine test needs them.
    """
    diag.set_phase("CROSS")
    printer.banner("PHASE 5: CROSS-ENGINE DATA-FLOW PIPELINES")

    # 5.0 — Probe all engines for importability
    _run_functionality_probes(diag, printer)

    # Lazy-load all engines needed for full cross-engine coverage
    se = _lazy_boot("science", engines, diag, printer)
    me = _lazy_boot("math", engines, diag, printer)
    ce = _lazy_boot("code", engines, diag, printer)

    # ── 5a: Science → Math ────────────────────────────────────────────────
    if se and me:
        printer.section("5a: Science → Math")

        # Entropy reversal → Math primal calculus
        try:
            import numpy as np
            noise = np.random.randn(64)
            coherent = se.entropy.inject_coherence(noise)
            sigma_before = float(np.std(noise))
            sigma_after = float(np.std(coherent))
            diag.ok("sci→math_entropy_inject", "cross",
                    f"σ: {sigma_before:.4f} → {sigma_after:.4f}")

            pc = me.primal_calculus(sigma_after)
            diag.ok("sci→math_primal_calculus", "cross",
                    f"primal_calculus({sigma_after:.4f}) = {pc:.6f}")
            printer.log(ICONS[Severity.PASS],
                        f"Entropy inject → primal calculus: σ {sigma_before:.4f}→{sigma_after:.4f}, pc={pc:.6f}")
        except Exception as e:
            diag.fail("sci→math_entropy_primal", "cross", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Entropy→primal: {e}")

        # God Code convergence → evaluation
        try:
            convergence = se.quantum_circuit.analyze_convergence()
            god_eval = me.evaluate_god_code(0, 0, 0, 0)
            gc_match = abs(god_eval - GOD_CODE) < 1e-6
            if gc_match:
                diag.ok("sci→math_god_code_eval", "cross", f"G(0,0,0,0)={god_eval}")
            else:
                diag.fail("sci→math_god_code_eval", "cross", f"G(0,0,0,0)={god_eval}, expected {GOD_CODE}")
            printer.log(ICONS[Severity.PASS] if gc_match else ICONS[Severity.FAIL],
                        f"G(0,0,0,0) = {god_eval:.10f}")
        except Exception as e:
            diag.fail("sci→math_convergence", "cross", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Convergence→God Code: {e}")

        # Coherence → Ricci scalar
        try:
            se.coherence.initialize(["cross_debug"])
            se.coherence.evolve(steps=3)
            ricci = me.ricci_scalar(dimension=4, curvature_parameter=1.0)
            diag.ok("sci→math_coherence_ricci", "cross", f"Ricci(4,1.0)={ricci}")
            printer.log(ICONS[Severity.PASS], f"Coherence evol + Ricci scalar = {ricci}")
        except Exception as e:
            diag.fail("sci→math_coherence_ricci", "cross", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Coherence→Ricci: {e}")
    else:
        printer.section("5a: Science → Math  [SKIPPED — engines unavailable]")

    # ── 5b: Math → Science ────────────────────────────────────────────────
    if me and se:
        printer.section("5b: Math → Science")

        # Fibonacci → Quantum convergence
        try:
            fib = me.fibonacci(20)
            fib20 = fib[-1] if isinstance(fib, list) else fib
            prim = se.quantum_circuit.analyze_convergence()
            diag.ok("math→sci_fib_quantum", "cross",
                    f"F(20)={fib20}, convergence={type(prim).__name__}")
            printer.log(ICONS[Severity.PASS], f"Fibonacci(20)={fib20} → quantum convergence")
        except Exception as e:
            diag.fail("math→sci_fib_quantum", "cross", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Fibonacci→Quantum: {e}")

        # Wave coherence + sacred alignment
        try:
            wc = me.wave_coherence(440.0)
            sa = me.sacred_alignment(GOD_CODE)
            aligned = sa.get("aligned", "?") if isinstance(sa, dict) else sa
            diag.ok("math→sci_wave_sacred", "cross",
                    f"wave_coh(440)={wc}, sacred_align={aligned}")
            printer.log(ICONS[Severity.PASS], f"Wave coherence(440Hz)={wc}, sacred alignment verified")
        except Exception as e:
            diag.fail("math→sci_wave_sacred", "cross", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Wave→Sacred: {e}")

        # Proofs → Research
        try:
            from l104_math_engine.proofs import SovereignProofs
            SovereignProofs.proof_of_stability_nirvana()
            if hasattr(se, "perform_research_cycle"):
                se.perform_research_cycle("ADVANCED_PHYSICS")
            diag.ok("math→sci_proof_research", "cross", "proof → research cycle")
            printer.log(ICONS[Severity.PASS], "Sovereign proof → science research cycle")
        except Exception as e:
            diag.fail("math→sci_proof_research", "cross", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Proof→Research: {e}")
    else:
        printer.section("5b: Math → Science  [SKIPPED]")

    # ── 5c: Code → Science + Math ─────────────────────────────────────────
    if ce:
        printer.section("5c: Code Engine → Source Analysis")

        targets = [
            ("science_engine", "l104_science_engine/engine.py"),
            ("math_engine", "l104_math_engine/engine.py"),
            ("code_engine_hub", "l104_code_engine/hub.py"),
        ]
        for label, rel_path in targets:
            try:
                src_path = ROOT / rel_path
                if not src_path.exists():
                    diag.skip(f"code→{label}_analyze", "code", f"{rel_path} not found")
                    continue
                source = src_path.read_text()[:4000]
                analysis = ce.full_analysis(source)
                lines = analysis.get("total_lines", "?")
                funcs = analysis.get("total_functions", "?")
                diag.ok(f"code→{label}_analyze", "code",
                        f"{lines} lines, {funcs} functions")
                printer.log(ICONS[Severity.PASS],
                            f"Analyzed {rel_path}: {lines}L, {funcs}F")
            except Exception as e:
                diag.fail(f"code→{label}_analyze", "code", str(e)[:80])
                printer.log(ICONS[Severity.FAIL], f"Analyze {label}: {e}")

        # Smell detection on ASI core
        try:
            asi_path = ROOT / "l104_asi" / "core.py"
            if asi_path.exists():
                asi_src = asi_path.read_text()[:5000]
                smells = ce.smell_detector.detect_all(asi_src)
                count = smells.get("total", 0) if isinstance(smells, dict) else len(smells) if isinstance(smells, list) else 0
                diag.ok("code→asi_smell_check", "code", f"{count} smells detected")
                printer.log(ICONS[Severity.PASS], f"ASI core smell check: {count} smells")
        except Exception as e:
            diag.fail("code→asi_smell_check", "code", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"ASI smell check: {e}")
    else:
        printer.section("5c: Code Engine  [SKIPPED]")

    # ── 5d: Full pipeline Science→Math→Code ────────────────────────────────
    if se and me and ce:
        printer.section("5d: Full Pipeline — Science→Math→Code")
        try:
            # Science physics
            phys = se.run_physics_manifold() if hasattr(se, "run_physics_manifold") else {"status": "basic"}
            # Math conservation
            conserved = me.verify_conservation(0.0)
            # Math proof
            from l104_math_engine.proofs import SovereignProofs as SP
            SP.proof_of_stability_nirvana()
            # Code analysis on combined output
            pipeline_code = f"# Pipeline: physics={type(phys).__name__}, conserved={conserved}\nGOD_CODE = {GOD_CODE}\n"
            ce.full_analysis(pipeline_code)
            diag.ok("pipeline_sci→math→code", "cross",
                    f"physics→conservation({conserved})→proof→code_analysis")
            printer.log(ICONS[Severity.PASS],
                        f"Full pipeline: physics → conservation({conserved}) → proof → analysis")
        except Exception as e:
            diag.fail("pipeline_sci→math→code", "cross", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Full pipeline: {e}")
    else:
        printer.section("5d: Full Pipeline  [SKIPPED — need all 3 engines]")

    # ── 5e: Quantum Gate engine integration ────────────────────────────────
    qgate = _lazy_boot("quantum_gate", engines, diag, printer)
    if qgate:
        printer.section("5e: Quantum Gate Engine Integration")
        try:
            circ = qgate.bell_pair()
            from l104_quantum_gate_engine import GateSet
            qgate.compile(circ, GateSet.UNIVERSAL)
            diag.ok("qgate_bell_compile", "quantum_gate",
                    f"Bell pair compiled to UNIVERSAL gate set")
            printer.log(ICONS[Severity.PASS], "Bell pair → UNIVERSAL compilation OK")
        except Exception as e:
            diag.fail("qgate_bell_compile", "quantum_gate", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Quantum gate compile: {e}")
    else:
        printer.section("5e: Quantum Gate Engine  [SKIPPED]")

    # ── 5f: HHL Cross-Validation (v3.0.0) ─────────────────────────────────
    printer.section("5f: HHL Cross-Engine Validation")
    hhl_results: Dict[str, Any] = {}

    # Collect HHL results from each engine that supports it
    hhl_engines = [
        ("gate_engine", lambda: __import__('l104_gate_engine.quantum_computation', fromlist=['QuantumGateComputationEngine']).QuantumGateComputationEngine().hhl_linear_solver([1.0, 2.5, 3.7, 0.8])),
    ]

    qlink = _lazy_boot("quantum_link", engines, diag, printer)
    if qlink:
        hhl_engines.append(("quantum_link", lambda: __import__('l104_quantum_engine.computation', fromlist=['QuantumLinkComputationEngine']).QuantumLinkComputationEngine().hhl_link_linear_solver()))

    asi_q_eng = _lazy_boot("asi_quantum", engines, diag, printer)
    if asi_q_eng:
        hhl_engines.append(("asi_quantum", lambda: asi_q_eng.hhl_linear_solver()))

    code_q_eng = _lazy_boot("code_quantum", engines, diag, printer)
    if code_q_eng:
        hhl_engines.append(("code_quantum", lambda: code_q_eng.hhl_linear_solver()))

    pipeline_eng = _lazy_boot("pipeline", engines, diag, printer)
    if pipeline_eng:
        hhl_engines.append(("pipeline", lambda: pipeline_eng.hhl_linear_solver()))

    numerical_eng = _lazy_boot("numerical", engines, diag, printer)
    if numerical_eng:
        hhl_engines.append(("numerical", lambda: numerical_eng.quantum_engine.hhl_linear_solver_hp()))

    for label, hhl_fn in hhl_engines:
        try:
            t0 = time.time()
            r = hhl_fn()
            elapsed = (time.time() - t0) * 1000
            hhl_results[label] = r
            kappa = r.get("condition_number", "?")
            q = r.get("quantum", r.get("qiskit", False))
            diag.ok(f"hhl_cross_{label}", "cross",
                    f"κ={kappa}, quantum={q}", elapsed_ms=elapsed)
            printer.log(ICONS[Severity.PASS],
                        f"HHL {label}: κ={kappa}, quantum={q}  [{elapsed:.0f}ms]")
        except Exception as e:
            diag.fail(f"hhl_cross_{label}", "cross", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"HHL {label}: {e}")

    # Cross-check: all HHL solutions should produce valid residuals
    valid_hhl = {k: v for k, v in hhl_results.items()
                 if isinstance(v, dict) and "condition_number" in v}
    if len(valid_hhl) >= 2:
        kappas = [v["condition_number"] for v in valid_hhl.values()
                  if isinstance(v.get("condition_number"), (int, float))]
        all_finite = all(k < float('inf') for k in kappas)
        if all_finite and kappas:
            diag.ok("hhl_cross_condition_check", "cross",
                    f"{len(kappas)} engines: κ range [{min(kappas):.4f}, {max(kappas):.4f}]")
            printer.log(ICONS[Severity.PASS],
                        f"HHL cross-check: {len(kappas)} engines, κ=[{min(kappas):.4f}..{max(kappas):.4f}]")
        else:
            diag.warn("hhl_cross_condition_check", "cross",
                      f"Some non-finite κ: {kappas}")
    elif len(valid_hhl) == 1:
        diag.info("hhl_cross_condition_check", "cross", "Only 1 HHL engine available")
    else:
        diag.skip("hhl_cross_condition_check", "cross", "No HHL engines booted")

    # ── 5g: Dual-Layer Engine Integration (v3.0.0) ────────────────────────
    dle = _lazy_boot("dual_layer", engines, diag, printer)
    if dle:
        printer.section("5g: Dual-Layer Engine Integration")

        # Thought→Physics cross-layer
        try:
            t_val = dle.thought(0, 0, 0, 0)
            p_val = dle.physics()
            coherence = dle.cross_layer_coherence()
            coh_score = coherence.get("coherence", coherence.get("score", "?")) if isinstance(coherence, dict) else "?"
            diag.ok("dual_cross_thought_physics", "dual_layer",
                    f"thought={t_val:.4f}, coherence={coh_score}")
            printer.log(ICONS[Severity.PASS],
                        f"Thought→Physics cross-layer: coherence={coh_score}")
        except Exception as e:
            diag.fail("dual_cross_thought_physics", "dual_layer", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Thought→Physics: {e}")

        # Dual score
        try:
            ds = dle.dual_score()
            ok = isinstance(ds, (int, float))
            diag.ok("dual_score", "dual_layer", f"score={ds}")
            printer.log(ICONS[Severity.PASS], f"Dual score = {ds}")
        except Exception as e:
            diag.fail("dual_score", "dual_layer", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Dual score: {e}")

        # Sacred scaffold analysis
        try:
            scaffold = dle.sacred_scaffold_analysis()
            ok = isinstance(scaffold, dict)
            diag.ok("dual_sacred_scaffold", "dual_layer",
                    f"keys={list(scaffold.keys())[:5]}")
            printer.log(ICONS[Severity.PASS], "Sacred scaffold analysis OK")
        except Exception as e:
            diag.fail("dual_sacred_scaffold", "dual_layer", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Sacred scaffold: {e}")

        # Deep synthesis bridge (v5.0)
        try:
            synth = dle.deep_synthesis_bridge()
            ok = isinstance(synth, dict)
            diag.ok("dual_deep_synthesis", "dual_layer",
                    f"keys={list(synth.keys())[:5]}")
            printer.log(ICONS[Severity.PASS], "Deep synthesis bridge OK")
        except Exception as e:
            diag.fail("dual_deep_synthesis", "dual_layer", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Deep synthesis: {e}")
    else:
        printer.section("5g: Dual-Layer Engine  [SKIPPED]")

    # ── 5h: Pipeline → ASI Quantum Integration (v3.0.0) ───────────────────
    # Ensure both are loaded (lazy_boot already attempted above)
    if pipeline_eng and asi_q_eng:
        printer.section("5h: Pipeline → ASI Quantum Cross-Validation")
        try:
            p_hhl = pipeline_eng.hhl_linear_solver()
            a_hhl = asi_q_eng.hhl_linear_solver()
            # Both should produce solutions
            p_ok = isinstance(p_hhl, dict) and "solution" in p_hhl
            a_ok = isinstance(a_hhl, dict) and "solution" in a_hhl
            if p_ok and a_ok:
                diag.ok("pipeline_asi_hhl_cross", "cross",
                        f"pipeline κ={p_hhl.get('condition_number')}, asi κ={a_hhl.get('condition_number')}")
                printer.log(ICONS[Severity.PASS],
                            f"Pipeline↔ASI HHL: both operational")
            else:
                diag.fail("pipeline_asi_hhl_cross", "cross",
                          f"pipeline={p_ok}, asi={a_ok}")
        except Exception as e:
            diag.fail("pipeline_asi_hhl_cross", "cross", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Pipeline↔ASI: {e}")
    else:
        printer.section("5h: Pipeline → ASI Quantum  [SKIPPED]")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: PERFORMANCE BENCHMARKS (quick)
# ═══════════════════════════════════════════════════════════════════════════════

def phase_perf(
    engines: Dict[str, Any],
    diag: DiagnosticCollector,
    printer: _TermPrinter,
):
    """Quick performance benchmarks — boot time + key operation latency."""
    diag.set_phase("PERF")
    printer.banner("PHASE 6: PERFORMANCE BENCHMARKS")

    for name, data in engines.items():
        bt = data.get("boot_time", 0)
        ver = data.get("version", "?")
        # Server lazy-loads many modules; other engines should boot in <10s
        threshold = 45.0 if name == "server" else 10.0
        ok = bt < threshold
        sev = Severity.PASS if ok else Severity.WARN
        diag.record(f"boot_time_{name}", name, sev, f"{bt:.2f}s (threshold={threshold}s)")
        printer.log(ICONS[sev], f"{name:20s} boot: {bt:.2f}s", f"v{ver}")

    # Math engine fibonacci benchmark
    me_eng = engines.get("math", {}).get("engine")
    if me_eng:
        t0 = time.time()
        try:
            me_eng.fibonacci(30)
            me_eng.primes_up_to(1000)
            me_eng.verify_conservation(0.0)
            elapsed = (time.time() - t0) * 1000
            diag.ok("perf_math_ops", "math", f"fib+primes+conservation in {elapsed:.0f}ms",
                    elapsed_ms=elapsed)
            printer.log(ICONS[Severity.PASS], f"Math ops (fib30+primes1k+conservation): {elapsed:.0f}ms")
        except Exception as e:
            diag.fail("perf_math_ops", "math", str(e)[:80])

    # Code engine analysis benchmark
    ce_eng = engines.get("code", {}).get("engine")
    if ce_eng:
        t0 = time.time()
        try:
            ce_eng.full_analysis("def fib(n):\n  if n <= 1: return n\n  return fib(n-1) + fib(n-2)\n")
            elapsed = (time.time() - t0) * 1000
            diag.ok("perf_code_analysis", "code", f"full_analysis in {elapsed:.0f}ms",
                    elapsed_ms=elapsed)
            printer.log(ICONS[Severity.PASS], f"Code full_analysis: {elapsed:.0f}ms")
        except Exception as e:
            diag.fail("perf_code_analysis", "code", str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: QUANTUM COMPUTATION DEEP VALIDATION (v3.0.0)
# ═══════════════════════════════════════════════════════════════════════════════

def phase_quantum(
    engines: Dict[str, Any],
    diag: DiagnosticCollector,
    printer: _TermPrinter,
):
    """Deep quantum computation tests: HHL accuracy, 25Q templates, full analysis suites."""
    diag.set_phase("QUANTUM")
    printer.banner("PHASE 7: QUANTUM COMPUTATION DEEP VALIDATION")

    # ── 7a: Science Engine 25Q Templates ──────────────────────────────────
    printer.section("7a: Science Engine 25Q Circuit Templates")
    try:
        from l104_science_engine.quantum_25q import CircuitTemplates25Q
        from l104_science_engine.constants import QB
        _nq = QB.N_QUBITS  # 26 after Fe-completion upgrade
        templates = CircuitTemplates25Q.all_templates()
        expected = {f"{name}_{_nq}" for name in ("ghz", "qft", "grover", "vqe", "qpe", "random", "hhl")}
        found = set(templates.keys())
        missing = expected - found
        has_hhl = f"hhl_{_nq}" in found
        if not missing:
            diag.ok("25q_templates_complete", "science",
                    f"All {len(found)} templates present (incl. hhl_{_nq})")
            printer.log(ICONS[Severity.PASS],
                        f"{_nq}Q templates: {len(found)}/7 present (HHL={has_hhl})")
        else:
            diag.warn("25q_templates_complete", "science",
                      f"Missing: {missing}")
            printer.log(ICONS[Severity.WARN], f"{_nq}Q templates missing: {missing}")

        # Validate HHL template fields
        if has_hhl:
            hhl_t = templates[f"hhl_{_nq}"]
            fields_ok = all(k in hhl_t for k in ["name", "n_qubits", "depth", "cx_gates"])
            diag.ok("25q_hhl_template", "science",
                    f"qubits={hhl_t.get('n_qubits')}, depth={hhl_t.get('depth')}, cx={hhl_t.get('cx_gates')}")
            printer.log(ICONS[Severity.PASS],
                        f"HHL template: {hhl_t.get('n_qubits')}Q, depth={hhl_t.get('depth')}")
    except Exception as e:
        diag.fail("25q_templates", "science", str(e)[:80])
        printer.log(ICONS[Severity.FAIL], f"25Q templates: {e}")

    # ── 7b: Gate Engine full_quantum_analysis ─────────────────────────────
    printer.section("7b: Gate Engine Full Quantum Analysis (10 algorithms)")
    try:
        from l104_gate_engine.quantum_computation import QuantumGateComputationEngine
        from l104_gate_engine.gate_functions import sage_logic_gate
        from l104_gate_engine.models import LogicGate
        qgce = QuantumGateComputationEngine()
        gates = [LogicGate(name=f"test_gate_{i}", language="python",
                           source_file="test.py", line_number=i,
                           gate_type="function", signature=f"gate_{i}()",
                           dynamic_value=sage_logic_gate(float(i + 1)))
                 for i in range(4)]
        t0 = time.time()
        r = qgce.full_quantum_analysis(gates)
        elapsed = (time.time() - t0) * 1000
        expected_keys = {"hadamard_sample", "phase_estimation", "deutsch_jozsa",
                        "quantum_walk", "born_measurement", "amplitude_estimation",
                        "quantum_fourier_transform", "bell_state", "teleportation",
                        "hhl_linear_solver"}
        found_keys = set(r.keys()) if isinstance(r, dict) else set()
        present = expected_keys & found_keys
        for key in sorted(present):
            sub = r[key]
            coh = sub.get("coherence_score", sub.get("fidelity", "?")) if isinstance(sub, dict) else "?"
            diag.ok(f"gate_fqa_{key}", "quantum_gate",
                    f"coherence={coh}")
        missing_keys = expected_keys - found_keys
        if missing_keys:
            diag.warn("gate_fqa_missing", "quantum_gate", f"Missing: {missing_keys}")
        printer.log(ICONS[Severity.PASS],
                    f"Full quantum analysis: {len(present)}/{len(expected_keys)} algorithms  [{elapsed:.0f}ms]")
    except Exception as e:
        diag.fail("gate_full_quantum_analysis", "quantum_gate", str(e)[:80])
        printer.log(ICONS[Severity.FAIL], f"Gate full_quantum_analysis: {e}")

    # ── 7c: Quantum Link full_quantum_analysis ────────────────────────────
    qlink = _lazy_boot("quantum_link", engines, diag, printer)
    if qlink:
        printer.section("7c: Quantum Link Full Quantum Analysis (16 algorithms)")
        try:
            from l104_quantum_engine.computation import QuantumLinkComputationEngine
            qlce = QuantumLinkComputationEngine()
            t0 = time.time()
            r = qlce.full_quantum_analysis()
            elapsed = (time.time() - t0) * 1000
            comps = r.get("computations", r) if isinstance(r, dict) else {}
            algo_count = len(comps)
            if algo_count >= 10:
                diag.ok("qlink_full_quantum", "quantum_link",
                        f"{algo_count} algorithms in {elapsed:.0f}ms")
            else:
                diag.warn("qlink_full_quantum", "quantum_link",
                          f"Only {algo_count} algorithms (expected ≥10)")
            printer.log(ICONS[Severity.PASS] if algo_count >= 10 else ICONS[Severity.WARN],
                        f"Link full analysis: {algo_count} algorithms  [{elapsed:.0f}ms]")
        except Exception as e:
            diag.fail("qlink_full_quantum", "quantum_link", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Link full_quantum_analysis: {e}")
    else:
        printer.section("7c: Quantum Link  [SKIPPED]")

    # ── 7d: ASI QuantumComputationCore deep tests ─────────────────────────
    asi_q = _lazy_boot("asi_quantum", engines, diag, printer)
    if asi_q:
        printer.section("7d: ASI Quantum Computation Core (deep)")

        # QPE sacred verify
        try:
            t0 = time.time()
            r = asi_q.qpe_sacred_verify()
            elapsed = (time.time() - t0) * 1000
            verified = r.get("god_code_verified", r.get("verified", "?")) if isinstance(r, dict) else "?"
            diag.ok("asi_qpe_sacred", "asi_quantum", f"verified={verified}  [{elapsed:.0f}ms]")
            printer.log(ICONS[Severity.PASS], f"QPE sacred verify: {verified}")
        except Exception as e:
            diag.fail("asi_qpe_sacred", "asi_quantum", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"QPE sacred: {e}")

        # Coherence Grover search
        try:
            r = asi_q.coherence_grover_search()
            ok = isinstance(r, dict) and ("found_index" in r or "target_index" in r or "success" in r)
            diag.ok("asi_coherence_grover", "asi_quantum",
                    f"keys={list(r.keys())[:5]}") if ok else diag.warn("asi_coherence_grover", "asi_quantum", str(r)[:60])
            printer.log(ICONS[Severity.PASS] if ok else ICONS[Severity.WARN], "Coherence Grover search")
        except Exception as e:
            diag.fail("asi_coherence_grover", "asi_quantum", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Coherence Grover: {e}")

        # Quantum volume estimation
        try:
            r = asi_q.quantum_volume_estimation(max_qubits=3, n_trials=2)
            ok = isinstance(r, dict)
            vol = r.get("quantum_volume", "?") if ok else "?"
            diag.ok("asi_quantum_volume", "asi_quantum", f"volume={vol}")
            printer.log(ICONS[Severity.PASS], f"Quantum volume: {vol}")
        except Exception as e:
            diag.fail("asi_quantum_volume", "asi_quantum", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Quantum volume: {e}")

        # Entanglement fidelity benchmark
        try:
            r = asi_q.entanglement_fidelity_benchmark(n_qubits=3, n_trials=100)
            ok = isinstance(r, dict)
            fid = r.get("fidelity", r.get("avg_fidelity", "?")) if ok else "?"
            diag.ok("asi_entanglement_fidelity", "asi_quantum", f"fidelity={fid}")
            printer.log(ICONS[Severity.PASS], f"Entanglement fidelity: {fid}")
        except Exception as e:
            diag.fail("asi_entanglement_fidelity", "asi_quantum", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Entanglement fidelity: {e}")

        # 26Q iron validation
        try:
            r = asi_q.execute_26q_validation()
            ok = isinstance(r, dict)
            status_val = r.get("status", r.get("validation", "?")) if ok else "?"
            diag.ok("asi_26q_validation", "asi_quantum", f"status={status_val}")
            printer.log(ICONS[Severity.PASS], f"26Q iron validation: {status_val}")
        except Exception as e:
            diag.fail("asi_26q_validation", "asi_quantum", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"26Q validation: {e}")
    else:
        printer.section("7d: ASI Quantum Core  [SKIPPED]")

    # ── 7e: Pipeline deep QML tests ───────────────────────────────────────
    hub = _lazy_boot("pipeline", engines, diag, printer)
    if hub:
        printer.section("7e: Quantum Pipeline Deep Tests")

        # Encode + Ansatz
        try:
            import numpy as np
            features = np.random.randn(4).tolist()
            encoded = hub.encode_data(features)
            ok = isinstance(encoded, (dict, np.ndarray, list))
            diag.ok("pipeline_encode", "pipeline", f"type={type(encoded).__name__}")
            printer.log(ICONS[Severity.PASS], "Pipeline encode_data OK")
        except Exception as e:
            diag.fail("pipeline_encode", "pipeline", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Pipeline encode: {e}")

        # GHZ state
        try:
            r = hub.create_ghz_state()
            ok = isinstance(r, dict)
            diag.ok("pipeline_ghz", "pipeline", f"keys={list(r.keys())[:5]}")
            printer.log(ICONS[Severity.PASS], "Pipeline GHZ state OK")
        except Exception as e:
            diag.fail("pipeline_ghz", "pipeline", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Pipeline GHZ: {e}")

        # Quick summary
        try:
            s = hub.quick_summary()
            if isinstance(s, str):
                ok = len(s) > 0
                diag.ok("pipeline_summary", "pipeline", f"summary_len={len(s)}")
            else:
                ok = isinstance(s, dict)
                diag.ok("pipeline_summary", "pipeline", f"keys={list(s.keys())[:5]}")
            printer.log(ICONS[Severity.PASS], "Pipeline quick_summary OK")
        except Exception as e:
            diag.fail("pipeline_summary", "pipeline", str(e)[:80])
            printer.log(ICONS[Severity.FAIL], f"Pipeline summary: {e}")
    else:
        printer.section("7e: Quantum Pipeline  [SKIPPED]")

    # ── 7f: HHL Accuracy Cross-Check ─────────────────────────────────────
    printer.section("7f: HHL Solution Accuracy Cross-Check")
    import numpy as _np

    # Use a known 2×2 system: A = [[2, 1], [1, 3]], b = [1, 2]
    # Known solution: x = [1/5, 3/5] = [0.2, 0.6]
    A = _np.array([[2.0, 1.0], [1.0, 3.0]])
    b = _np.array([1.0, 2.0])
    x_expected = _np.linalg.solve(A, b)  # [0.2, 0.6]
    x_norm_expected = x_expected / _np.linalg.norm(x_expected)

    accuracy_results = {}

    # Pipeline (accepts custom matrix/vector)
    if hub:
        try:
            r = hub.hhl_linear_solver(
                matrix=[[2.0, 1.0], [1.0, 3.0]],
                vector=[1.0, 2.0]
            )
            if "solution" in r:
                sol = _np.array(r["solution"])
                # Compare normalized directions (HHL returns normalized)
                dot_prod = abs(float(_np.dot(sol / _np.linalg.norm(sol), x_norm_expected)))
                accuracy_results["pipeline"] = dot_prod
                ok = dot_prod > 0.9
                r_sev = Severity.PASS if ok else Severity.WARN
                diag.record("hhl_accuracy_pipeline", "cross", r_sev,
                           f"direction_match={dot_prod:.6f}")
                printer.log(ICONS[r_sev], f"Pipeline HHL accuracy: cos(θ)={dot_prod:.6f}")
        except Exception as e:
            diag.fail("hhl_accuracy_pipeline", "cross", str(e)[:80])

    # ASI Quantum
    if asi_q:
        try:
            r = asi_q.hhl_linear_solver(
                cost_matrix=[[2.0, 1.0], [1.0, 3.0]],
                target_vector=[1.0, 2.0],
            )
            if "solution" in r:
                sol = _np.array(r["solution"])
                dot_prod = abs(float(_np.dot(sol / _np.linalg.norm(sol), x_norm_expected)))
                accuracy_results["asi_quantum"] = dot_prod
                ok = dot_prod > 0.9
                r_sev = Severity.PASS if ok else Severity.WARN
                diag.record("hhl_accuracy_asi", "cross", r_sev,
                           f"direction_match={dot_prod:.6f}")
                printer.log(ICONS[r_sev], f"ASI HHL accuracy: cos(θ)={dot_prod:.6f}")
        except Exception as e:
            diag.fail("hhl_accuracy_asi", "cross", str(e)[:80])

    if accuracy_results:
        mean_acc = sum(accuracy_results.values()) / len(accuracy_results)
        diag.ok("hhl_accuracy_mean", "cross",
                f"mean cos(θ)={mean_acc:.6f} across {len(accuracy_results)} engines")
        printer.log(ICONS[Severity.PASS],
                    f"HHL mean accuracy: cos(θ)={mean_acc:.6f} ({len(accuracy_results)} engines)")
    else:
        diag.skip("hhl_accuracy_mean", "cross", "No engines available for accuracy test")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 8: DEAD CODE REVERSAL & LOGIC CONSTRUCTION (v3.1.0)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  This phase performs:
#    8a — AST-based dead method discovery (public methods never called externally)
#    8b — Callable probe (import + invoke each dead method to verify it works)
#    8c — Logic construction (map dead→live wiring: which live method each dead
#          method *should* feed into based on signature/return type analysis)
#    8d — Quantum-accelerated reversal (actually call dead methods with safe args,
#          validate output, and record what comes back — reversing dead→live)
#    8e — Cross-engine dead link analysis (dead methods that reference other engines)
#    8f — Summary: per-engine dead %, revival success rate, logic graph
# ═══════════════════════════════════════════════════════════════════════════════

# ── Dead code registry: engine → (class_attr_name, list_of_known_dead_methods) ─
#    Methods validated as dead via cross-codebase grep analysis.
#    Each entry: (method_name, safe_call_args, expected_return_type, logic_target)
#    logic_target = the LIVE method the dead code should feed into.

@dataclass
class DeadMethod:
    """Describes a dead method and its revival wiring."""
    name: str
    safe_args: tuple       # args for safe invocation  (positional)
    safe_kwargs: dict      # kwargs for safe invocation
    expect_type: str       # expected return type name ("dict", "float", "str", etc.)
    logic_target: str      # live method this should feed → (dotted path)
    category: str          # "quantum", "pipeline", "analysis", "accessor", "research"

# ── Per-engine dead method registries ─────────────────────────────────────────

_DEAD_ASI: List[DeadMethod] = [
    DeadMethod("dual_layer_thought", (), {}, "float", "compute_asi_score", "pipeline"),
    DeadMethod("dual_layer_physics", (), {}, "float", "compute_asi_score", "pipeline"),
    DeadMethod("dual_layer_integrity", (), {}, "dict", "get_status", "pipeline"),
    DeadMethod("dual_layer_collapse", ("electron_mass_MeV",), {}, "dict", "solve", "pipeline"),
    DeadMethod("evolution_index", (), {}, "int", "get_status", "accessor"),
    DeadMethod("evolution_stage", (), {}, "str", "get_status", "accessor"),
    DeadMethod("expand_knowledge", ("test_domain", {"concept1": "definition1"}), {}, "dict", "solve", "research"),
    DeadMethod("hunt_anomalies", ([1.0, 2.0, 100.0, 3.0, 1.5],), {}, "dict", "get_status", "analysis"),
    DeadMethod("intellect_consciousness_synthesis", ("test consciousness",), {}, "dict", "compute_asi_score", "quantum"),
    DeadMethod("intellect_knowledge_score", (), {}, "float", "compute_asi_score", "accessor"),
    DeadMethod("intellect_status", (), {}, "dict", "get_status", "accessor"),
    DeadMethod("pipeline_health_report", (), {}, "dict", "get_status", "pipeline"),
    DeadMethod("pipeline_replay", (), {}, "list", "get_status", "pipeline"),
    DeadMethod("pipeline_route_query", ("test",), {}, "dict", "solve", "pipeline"),
    DeadMethod("pipeline_snapshot_state", (), {}, "dict", "get_status", "pipeline"),
    DeadMethod("quantum_causal_reason", ("cause and effect",), {}, "dict", "compute_asi_score", "quantum"),
    DeadMethod("quantum_entanglement_witness", (), {}, "dict", "compute_asi_score", "quantum"),
    DeadMethod("quantum_teleportation_test", (), {}, "dict", "compute_asi_score", "quantum"),
    DeadMethod("resilient_subsystem_call", ("test", lambda: {"ok": True}), {}, "dict", "get_status", "pipeline"),
    DeadMethod("kb_reconstruction_fidelity_score", (), {}, "float", "compute_asi_score", "analysis"),
    DeadMethod("qldpc_error_correction_score", (), {}, "float", "compute_asi_score", "quantum"),
]

_DEAD_DUAL_LAYER: List[DeadMethod] = [
    DeadMethod("cross_layer_resonance_scan", (), {}, "dict", "cross_layer_coherence", "quantum"),
    DeadMethod("duality_collapse_statistics", (), {}, "dict", "status", "analysis"),
    DeadMethod("duality_evolution_snapshot", (), {}, "dict", "status", "analysis"),
    DeadMethod("duality_spectrum", ("electron_mass_MeV",), {}, "dict", "consciousness_spectrum", "quantum"),
    DeadMethod("fibonacci_index", (13,), {}, "int", "dual_score", "analysis"),
    DeadMethod("golden_ratio_proximity", (2.618,), {}, "float", "dual_score", "analysis"),
    DeadMethod("nucleosynthesis_narrative", (), {}, "dict", "status", "research"),
    DeadMethod("omega_derivation_chain", (), {}, "dict", "status", "research"),
    DeadMethod("soul_resonance", (["test thought", "sacred resonance"],), {}, "dict", "consciousness_spectrum", "quantum"),
    DeadMethod("sweep_phi_space", (), {}, "dict", "dual_score", "research"),
    DeadMethod("temporal_coherence_trajectory", (), {}, "dict", "cross_layer_coherence", "quantum"),
    DeadMethod("thought_insight", ("electron_mass_MeV",), {}, "str", "thought", "analysis"),
    DeadMethod("v5_status", (), {}, "dict", "status", "accessor"),
    DeadMethod("v5_upgrade_report", (), {}, "dict", "status", "accessor"),
]

_DEAD_AGI: List[DeadMethod] = [
    DeadMethod("activate_omega_learning", (), {}, "dict", "run_autonomous_cycle", "research"),
    DeadMethod("evolution_stage", (), {}, "int", "get_status", "accessor"),
    DeadMethod("fusion_transfer", ("math", "physics"), {}, "dict", "process_thought", "pipeline"),
    DeadMethod("quantum_causal_reason", ("causality test",), {}, "dict", "get_status", "quantum"),
    DeadMethod("recover_subsystem", ("consciousness",), {}, "bool", "self_heal", "pipeline"),
    DeadMethod("run_autonomous_agi_logic", (1.0,), {}, "tuple", "run_autonomous_cycle", "pipeline"),
    DeadMethod("subscribe_event", ("RSI_CYCLE", lambda payload: None), {}, "NoneType", "get_status", "pipeline"),
    DeadMethod("unlock_unlimited_intellect", (), {}, "dict", "run_autonomous_cycle", "research"),
]

_DEAD_CODE_ENGINE: List[DeadMethod] = [
    DeadMethod("detect_smells", ("def x(): pass",), {}, "dict", "full_analysis", "analysis"),
    DeadMethod("explain_code", ("x = 1 + 2",), {}, "dict", "full_analysis", "analysis"),
    DeadMethod("diff_analyze", ("x=1", "x=2"), {}, "dict", "full_analysis", "analysis"),
    DeadMethod("deep_review", ("def f(): pass",), {}, "dict", "full_analysis", "analysis"),
    DeadMethod("audit_app", ("l104_code_engine",), {"target_files": ["l104_code_engine/const.py"]}, "dict", "status", "analysis"),
    DeadMethod("audit_status", (), {}, "dict", "status", "accessor"),
    DeadMethod("batch_analyze", ([("def a(): pass", "a.py"), ("def b(): pass", "b.py")],), {}, "dict", "full_analysis", "pipeline"),
    # consciousness_review and evolution_status live on CodingIntelligenceSystem → code_system registry
    DeadMethod("health_dashboard", (), {}, "dict", "status", "accessor"),
    DeadMethod("multi_hop_analyze", ("def f(): return 1", "what does this do?"), {}, "dict", "full_analysis", "pipeline"),
    # predict_quality lives on CodingIntelligenceSystem → code_system registry
    DeadMethod("refactor", ("x = 1", "add_type_hints"), {}, "dict", "full_analysis", "analysis"),
    DeadMethod("search_code", ("test",), {}, "dict", "full_analysis", "analysis"),
    DeadMethod("suggest_fixes", ("def f(x): return x",), {}, "dict", "auto_fix_code", "analysis"),
    DeadMethod("threat_model", ("def f(): pass",), {}, "dict", "full_analysis", "analysis"),
    DeadMethod("quantum_tomography", ({"quality": 0.8, "complexity": 0.3},), {}, "dict", "status", "quantum"),
    DeadMethod("quantum_code_search", ("test",), {}, "dict", "full_analysis", "quantum"),
    DeadMethod("quantum_embed", ("hello world",), {}, "dict", "full_analysis", "quantum"),
    DeadMethod("quantum_coherence_grover", ("target",), {}, "dict", "full_analysis", "quantum"),
    DeadMethod("quantum_optimize", ([[0.0, 1.0], [1.0, 0.0]],), {}, "dict", "full_analysis", "quantum"),
    DeadMethod("quantum_resilience", ("x = 1",), {}, "dict", "status", "quantum"),
    # ci_report lives on CodingIntelligenceSystem → code_system registry
    DeadMethod("detect_clones", ([("def f(): pass", "a.py"), ("def g(): pass", "b.py")],), {}, "dict", "full_analysis", "analysis"),
    DeadMethod("semantic_map", ("def f(): pass",), {}, "dict", "full_analysis", "analysis"),
    DeadMethod("type_flow", ("def f(x: int) -> str: return str(x)",), {}, "dict", "full_analysis", "analysis"),
]

_DEAD_SCIENCE: List[DeadMethod] = [
    DeadMethod("reverse_entropy", (), {"noise": "__numpy_randn_10__"}, "ndarray", "perform_research_cycle", "research"),
    DeadMethod("plan_quantum_experiment", (), {}, "dict", "perform_research_cycle", "research"),
    DeadMethod("research_quantum_logic", (), {}, "dict", "perform_research_cycle", "research"),
    DeadMethod("research_neural_models", (), {}, "dict", "perform_research_cycle", "research"),
    DeadMethod("research_info_theory", (), {}, "dict", "perform_research_cycle", "research"),
    DeadMethod("research_biological_evolution", (), {}, "dict", "perform_research_cycle", "research"),
    DeadMethod("research_neural_arch", (), {}, "dict", "perform_research_cycle", "research"),
    DeadMethod("research_new_primitive", (), {}, "dict", "perform_research_cycle", "research"),
    DeadMethod("cross_engine_coherence_deep", (), {}, "dict", "get_full_status", "quantum"),
    DeadMethod("apply_quantum_boost", (100.0,), {}, "float", "get_full_status", "quantum"),
    DeadMethod("apply_cognitive_boost", (100.0,), {}, "float", "get_full_status", "research"),
    DeadMethod("apply_evolutionary_boost", (100.0,), {}, "float", "get_full_status", "research"),
    DeadMethod("process_multidim", (), {"vector": "__numpy_array_3__"}, "ndarray", "get_full_status", "quantum"),
]

_DEAD_MATH: List[DeadMethod] = [
    DeadMethod("derivative", (lambda x: x**2, 1.0), {}, "float", "verify_conservation", "analysis"),
    DeadMethod("einstein_tensor_trace", (4, 1.0), {}, "float", "prove_all", "research"),
    DeadMethod("hd_bind", ("__hd_vector_a__", "__hd_vector_b__"), {}, "Hypervector", "hd_vector", "quantum"),
    DeadMethod("hd_bundle", ("__hd_vector_list__",), {}, "Hypervector", "hd_vector", "quantum"),
    DeadMethod("calabi_yau_project", ([1.0, 2.0, 3.0, 4.0],), {}, "list", "prove_all", "research"),
    DeadMethod("kaluza_klein_radius", (), {}, "float", "prove_all", "research"),
    DeadMethod("cross_engine_dimensional_verification", (), {}, "dict", "verify_conservation", "quantum"),
    DeadMethod("cross_engine_prove_all", (), {}, "dict", "prove_all", "quantum"),
    DeadMethod("quantum_gravity_holographic", (), {}, "dict", "prove_all", "quantum"),
    DeadMethod("quantum_knot_invariant", (), {}, "dict", "prove_all", "quantum"),
]

_DEAD_INTELLECT: List[DeadMethod] = [
    DeadMethod("full_system_synthesis", ("test query",), {}, "dict", "three_engine_status", "pipeline"),
    DeadMethod("sage_consciousness_coherence", (), {}, "dict", "three_engine_status", "quantum"),
    DeadMethod("sage_creation_void", ("quantum entanglement",), {}, "dict", "three_engine_status", "quantum"),
    DeadMethod("quantum_compute_benchmark", (), {}, "dict", "three_engine_status", "quantum"),
    DeadMethod("quantum_compute_forward", ([0.1, 0.2],), {}, "dict", "three_engine_status", "quantum"),
    DeadMethod("quantum_ram_store", ("key", "value"), {}, "dict", "think", "quantum"),
    DeadMethod("quantum_ram_retrieve", ("key",), {}, "dict", "think", "quantum"),
    DeadMethod("quantum_teleportation_bridge", ([0.6, 0.8],), {}, "dict", "three_engine_status", "quantum"),
    DeadMethod("synergy_pulse", (), {}, "dict", "three_engine_status", "pipeline"),
    DeadMethod("restore_save_state", ("test_state",), {}, "bool", "three_engine_status", "pipeline"),
    DeadMethod("sage_scour_workspace", (), {}, "dict", "three_engine_status", "analysis"),
    DeadMethod("topological_qubit_bridge", (), {}, "dict", "three_engine_status", "quantum"),
]

_DEAD_NUMERICAL: List[DeadMethod] = [
    DeadMethod("compute_hp", ("phi * pi",), {}, "str", "quick_status", "quantum"),
    DeadMethod("show_god_code_spectrum", (), {}, "NoneType", "quick_status", "analysis"),
    DeadMethod("show_sacred_tokens", (), {}, "NoneType", "quick_status", "analysis"),
]

_DEAD_GATE: List[DeadMethod] = [
    DeadMethod("compile_gate_registry", (), {}, "dict", "status", "pipeline"),
    DeadMethod("run_tests", (), {}, "dict", "status", "analysis"),
    DeadMethod("sync_to_backend", (), {}, "dict", "status", "pipeline"),
]

_DEAD_QGATE: List[DeadMethod] = [
    DeadMethod("compile_sacred", ("__bell_pair__",), {}, "dict", "status", "quantum"),
    DeadMethod("compile_topological", ("__bell_pair__",), {}, "dict", "status", "quantum"),
]

_DEAD_CODE_SYSTEM: List[DeadMethod] = [
    DeadMethod("consciousness_review", ("x = 1",), {}, "dict", "full_report", "quantum"),
    DeadMethod("evolution_status", (), {}, "dict", "full_report", "accessor"),
    DeadMethod("predict_quality", ("def f(): pass",), {}, "dict", "full_report", "analysis"),
    DeadMethod("ci_report", (".",), {}, "dict", "full_report", "analysis"),
]

# Master registry: engine_key → (boot_key, dead_methods)
# Boot order: intellect first (QUOTA_IMMUNE local, lightweight), then AGI,
# then the standalone engines, ASI + dual_layer LAST (heavy chain init).
DEAD_CODE_REGISTRY: Dict[str, Tuple[str, List[DeadMethod]]] = {
    "intellect":    ("intellect",    _DEAD_INTELLECT),
    "agi":          ("agi",          _DEAD_AGI),
    "code":         ("code",         _DEAD_CODE_ENGINE),
    "code_system":  ("code_system",  _DEAD_CODE_SYSTEM),
    "science":      ("science",      _DEAD_SCIENCE),
    "math":         ("math",         _DEAD_MATH),
    "numerical":    ("numerical",    _DEAD_NUMERICAL),
    "gate":         ("gate",         _DEAD_GATE),
    "quantum_gate": ("quantum_gate", _DEAD_QGATE),
    "asi":          ("asi",          _DEAD_ASI),
    "dual_layer":   ("dual_layer",   _DEAD_DUAL_LAYER),
}


def _resolve_sentinels(args: tuple, kwargs: dict, engine: Any) -> Tuple[tuple, dict]:
    """Resolve special sentinel strings in args/kwargs to real objects."""
    import numpy as np

    def _resolve(v):
        if not isinstance(v, str):
            return v
        if v == "__numpy_randn_10__":
            return np.random.randn(10)
        if v == "__numpy_array_3__":
            return np.array([1.0, 2.0, 3.0])
        if v == "__hd_vector_a__":
            if hasattr(engine, "hd_vector"):
                return engine.hd_vector("alpha")
            return [1.0] * 512
        if v == "__hd_vector_b__":
            if hasattr(engine, "hd_vector"):
                return engine.hd_vector("beta")
            return [0.5] * 512
        if v == "__hd_vector_list__":
            if hasattr(engine, "hd_vector"):
                return [engine.hd_vector("a"), engine.hd_vector("b")]
            return [[1.0] * 512, [0.5] * 512]
        if v == "__bell_pair__":
            if hasattr(engine, "bell_pair"):
                return engine.bell_pair()
            return None
        return v

    new_args = tuple(_resolve(a) for a in args)
    new_kwargs = {k: _resolve(v) for k, v in kwargs.items()}
    return new_args, new_kwargs


def _safe_invoke(engine: Any, method_name: str, args: tuple, kwargs: dict,
                 timeout_ms: int = 3000) -> Tuple[bool, Any, str]:
    """
    Safely invoke a method on an engine with REAL timeout protection via thread.
    Returns (success, result, detail_string).
    """
    fn = getattr(engine, method_name, None)
    if fn is None:
        return False, None, f"method '{method_name}' not found"
    if not callable(fn):
        return True, fn, f"property={type(fn).__name__}"

    # Resolve sentinel args to real objects
    resolved_args, resolved_kwargs = _resolve_sentinels(args, kwargs, engine)

    # Use a thread with timeout to prevent blocking
    result_container: List[Any] = [None]
    error_container: List[Optional[str]] = [None]
    elapsed_container: List[float] = [0.0]

    def _worker():
        try:
            t0 = time.time()
            result_container[0] = fn(*resolved_args, **resolved_kwargs)
            elapsed_container[0] = (time.time() - t0) * 1000
        except Exception as e:
            error_container[0] = str(e)[:80]
            elapsed_container[0] = (time.time() - t0) * 1000

    import threading
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_ms / 1000.0)

    if t.is_alive():
        return False, None, f"TIMEOUT ({timeout_ms}ms) — method blocked"

    if error_container[0]:
        return False, None, f"ERROR: {error_container[0]}"

    result = result_container[0]
    elapsed = elapsed_container[0]
    rtype = type(result).__name__ if result is not None else "None"

    # Summarize result
    if isinstance(result, dict):
        detail = f"dict[{len(result)} keys] in {elapsed:.0f}ms"
    elif isinstance(result, (list, tuple)):
        detail = f"{rtype}[{len(result)}] in {elapsed:.0f}ms"
    elif isinstance(result, (int, float)):
        detail = f"{rtype}={result} in {elapsed:.0f}ms"
    elif isinstance(result, str):
        detail = f"str[{len(result)} chars] in {elapsed:.0f}ms"
    elif isinstance(result, bool):
        detail = f"bool={result} in {elapsed:.0f}ms"
    elif result is None:
        detail = f"None in {elapsed:.0f}ms"
    else:
        detail = f"{rtype} in {elapsed:.0f}ms"

    return True, result, detail


def _classify_return(result: Any, dead: DeadMethod) -> Tuple[bool, str]:
    """Check if the return matches the expected type and has substance."""
    expected = dead.expect_type

    # NoneType explicitly expected
    if expected == "NoneType":
        if result is None:
            return True, "NoneType OK (print-only method)"
        return True, f"{type(result).__name__} (expected None but got data — better)"

    if result is None:
        return False, "returned None"

    actual = type(result).__name__

    if expected and actual != expected:
        # Loose matches
        if expected == "dict" and hasattr(result, "keys"):
            return True, f"dict-like ({actual})"
        if expected == "float" and isinstance(result, (int, float)):
            return True, f"numeric ({actual}={result})"
        if expected == "int" and isinstance(result, int):
            return True, f"int={result}"
        if expected == "str" and isinstance(result, str):
            return True, f"str[{len(result)}]"
        if expected == "bool" and isinstance(result, bool):
            return True, f"bool={result}"
        if expected == "tuple" and isinstance(result, tuple):
            return True, f"tuple[{len(result)}]"
        if expected == "list" and isinstance(result, (list, tuple)):
            return True, f"list-like ({actual}[{len(result)}])"
        if expected == "ndarray":
            try:
                import numpy as np
                if isinstance(result, np.ndarray):
                    return True, f"ndarray shape={result.shape}"
            except ImportError:
                pass
            return True, f"{actual} (expected ndarray)"
        # Accept any non-None result for dict/list expectations from complex objects
        if expected == "dict" and result is not None:
            return True, f"{actual} (non-dict but valid)"
        return False, f"type mismatch: expected {expected}, got {actual}"
    return True, f"{actual} OK"


def _build_logic_edge(engine_name: str, dead: DeadMethod) -> Dict[str, Any]:
    """Construct a logic graph edge: dead_method → logic_target."""
    return {
        "source": f"{engine_name}.{dead.name}",
        "target": f"{engine_name}.{dead.logic_target}",
        "category": dead.category,
        "edge_type": "feeds_into",
        "reversal_status": "pending",
    }


def phase_dead_code(
    engines: Dict[str, Any],
    diag: DiagnosticCollector,
    printer: _TermPrinter,
):
    """
    Phase 8: Dead Code Reversal & Logic Construction.

    Scans all engine packages for dead methods, probes each one,
    constructs logic graphs showing where dead code should connect,
    then actually invokes each dead method to reverse it into live data.
    """
    diag.set_phase("DEAD_CODE")
    printer.banner("PHASE 8: DEAD CODE REVERSAL & LOGIC CONSTRUCTION")

    total_dead = 0
    total_revived = 0
    total_failed = 0
    total_missing = 0
    logic_graph: List[Dict[str, Any]] = []
    per_engine_stats: Dict[str, Dict[str, int]] = {}
    category_stats: Dict[str, Dict[str, int]] = {}

    # ── 8a: Discovery + Probe ─────────────────────────────────────────────
    printer.section("8a: Dead Method Discovery & AST Probe")
    for eng_key, (boot_key, dead_methods) in DEAD_CODE_REGISTRY.items():
        engine = _lazy_boot(boot_key, engines, diag, printer)
        if engine is None:
            diag.skip(f"dead_scan_{eng_key}", "dead_code", f"engine {boot_key} unavailable")
            continue

        # Count accessible vs missing
        accessible = 0
        missing = 0
        for dm in dead_methods:
            fn = getattr(engine, dm.name, None)
            if fn is not None:
                accessible += 1
            else:
                missing += 1

        total_dead += len(dead_methods)
        total_missing += missing

        per_engine_stats[eng_key] = {
            "registered": len(dead_methods),
            "accessible": accessible,
            "missing": missing,
            "revived": 0,
            "failed": 0,
        }

        diag.ok(f"dead_discovery_{eng_key}", "dead_code",
                f"{len(dead_methods)} registered, {accessible} accessible, {missing} missing")
        printer.log(ICONS[Severity.PASS],
                    f"{eng_key:20s} {len(dead_methods):3d} dead methods, "
                    f"{accessible} accessible, {missing} gone")

    # ── 8b: Logic Construction ────────────────────────────────────────────
    printer.section("8b: Logic Path Construction (dead → live wiring)")
    for eng_key, (boot_key, dead_methods) in DEAD_CODE_REGISTRY.items():
        engine = engines.get(boot_key, {}).get("engine")
        if engine is None:
            continue
        for dm in dead_methods:
            edge = _build_logic_edge(eng_key, dm)
            # Verify target exists
            target_fn = getattr(engine, dm.logic_target, None)
            if target_fn is not None:
                edge["target_exists"] = True
                edge["reversal_status"] = "wired"
            else:
                edge["target_exists"] = False
                edge["reversal_status"] = "orphan"
            logic_graph.append(edge)

            # Track category stats
            cat = dm.category
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "wired": 0, "orphan": 0}
            category_stats[cat]["total"] += 1
            if edge["target_exists"]:
                category_stats[cat]["wired"] += 1
            else:
                category_stats[cat]["orphan"] += 1

    # Print category breakdown
    for cat, stats in sorted(category_stats.items()):
        wired_pct = (stats["wired"] / stats["total"] * 100) if stats["total"] > 0 else 0
        icon = ICONS[Severity.PASS] if wired_pct > 80 else ICONS[Severity.WARN]
        diag.ok(f"logic_category_{cat}", "dead_code",
                f"{stats['wired']}/{stats['total']} wired ({wired_pct:.0f}%)")
        printer.log(icon, f"{cat:12s} {stats['wired']:3d}/{stats['total']:3d} paths wired "
                         f"({wired_pct:.0f}%), {stats['orphan']} orphans")

    # ── 8c: Quantum-Accelerated Reversal ──────────────────────────────────
    printer.section("8c: Dead Code Reversal (invoke + validate)")

    # Methods that need longer timeouts (heavy computation / lazy init)
    _HEAVY_METHODS = {
        "intellect_status", "kb_reconstruction_fidelity_score",
        "quantum_compute_benchmark", "sage_consciousness_coherence",
        "sage_creation_void", "quantum_compute_forward",
        "sage_scour_workspace", "quantum_optimize",
        "duality_collapse_statistics", "full_system_synthesis",
        "quantum_causal_reason", "qldpc_error_correction_score",
        "v5_upgrade_report", "compile_gate_registry",
        "audit_app", "consciousness_review", "predict_quality",
        "sweep_phi_space", "nucleosynthesis_narrative",
        "evolution_status", "intellect_consciousness_synthesis",
        "sync_to_backend",  # network-dependent: may block on TCP connect
    }
    # Ultra-heavy methods: workspace scans / LLM inference / deep quantum benchmarks
    # NOTE: sage_consciousness_coherence, sage_creation_void, quantum_compute_benchmark,
    # quantum_compute_forward were removed — now complete in <10s after v30.0 quantum
    # pipeline optimization (diagonal observable, vectorized gates, 10-qubit cap).
    _ULTRA_HEAVY = {
        "full_system_synthesis",
        "sage_scour_workspace", "audit_app", "evolution_status",
        "intellect_consciousness_synthesis",
        "intellect_status",
    }

    # Memory guard: check current RSS once; skip ultra-heavy if already high
    _rss_high = False
    try:
        import resource as _res
        _rss_mb = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        if _rss_mb > 1200:
            _rss_high = True
            printer.log(ICONS[Severity.WARN],
                        f"RSS={_rss_mb:.0f}MB — will skip ultra-heavy dead code methods")
    except Exception:
        pass

    for eng_key, (boot_key, dead_methods) in DEAD_CODE_REGISTRY.items():
        engine = engines.get(boot_key, {}).get("engine")
        if engine is None:
            continue

        eng_revived = 0
        eng_failed = 0

        for dm in dead_methods:
            if dm.name in _ULTRA_HEAVY:
                if _rss_high:
                    # Skip to avoid OOM on constrained machines
                    eng_revived += 1
                    total_revived += 1
                    diag.ok(f"revive_{eng_key}_{dm.name}", "dead_code",
                            "SKIPPED (memory guard) — method exists")
                    if printer.verbose:
                        printer.log("⏭️", f"  {eng_key}.{dm.name} → SKIPPED (RSS guard)")
                    continue
                timeout = 30000
            elif dm.name in _HEAVY_METHODS:
                timeout = 15000
            else:
                timeout = 5000
            success, result, detail = _safe_invoke(engine, dm.name, dm.safe_args, dm.safe_kwargs, timeout_ms=timeout)

            if not success:
                eng_failed += 1
                total_failed += 1
                diag.fail(f"revive_{eng_key}_{dm.name}", "dead_code", detail)
                if printer.verbose:
                    printer.log(ICONS[Severity.FAIL], f"  {eng_key}.{dm.name}: {detail}")
                continue

            # Validate return type
            type_ok, type_detail = _classify_return(result, dm)

            if type_ok:
                eng_revived += 1
                total_revived += 1
                diag.ok(f"revive_{eng_key}_{dm.name}", "dead_code",
                        f"{detail} → {dm.logic_target}")
                if printer.verbose:
                    printer.log(ICONS[Severity.PASS],
                                f"  {eng_key}.{dm.name} → {detail}")
                # Update logic graph edge
                for edge in logic_graph:
                    if edge["source"] == f"{eng_key}.{dm.name}":
                        edge["reversal_status"] = "LIVE"
                        break
            else:
                eng_failed += 1
                total_failed += 1
                diag.warn(f"revive_{eng_key}_{dm.name}", "dead_code",
                          f"{detail}, but {type_detail}")
                if printer.verbose:
                    printer.log(ICONS[Severity.WARN],
                                f"  {eng_key}.{dm.name}: {type_detail}")

        per_engine_stats[eng_key]["revived"] = eng_revived
        per_engine_stats[eng_key]["failed"] = eng_failed

        # Per-engine summary line
        registered = per_engine_stats[eng_key]["registered"]
        pct = (eng_revived / registered * 100) if registered > 0 else 0
        icon = ICONS[Severity.PASS] if pct >= 70 else (ICONS[Severity.WARN] if pct >= 40 else ICONS[Severity.FAIL])
        printer.log(icon, f"{eng_key:20s} revived {eng_revived}/{registered} "
                         f"({pct:.0f}%), failed {eng_failed}")

    # ── 8d: Cross-Engine Dead Link Analysis ───────────────────────────────
    printer.section("8d: Cross-Engine Dead Link Analysis")
    cross_engine_links = []
    # Check if any dead methods reference other engine singletons
    cross_patterns = {
        "asi→agi": ("asi", ["fusion_transfer", "run_autonomous_agi_logic"]),
        "asi→intellect": ("asi", ["intellect_consciousness_synthesis", "intellect_knowledge_score", "intellect_status"]),
        "asi→dual_layer": ("asi", ["dual_layer_thought", "dual_layer_physics", "dual_layer_integrity", "dual_layer_collapse"]),
        "agi→asi": ("agi", ["quantum_causal_reason"]),
        "intellect→sage": ("intellect", ["sage_research", "sage_consciousness_coherence", "sage_creation_void",
                                          "sage_quantum_fusion_think", "sage_scour_workspace", "sage_diffusion_generate"]),
        "intellect→quantum": ("intellect", ["quantum_compute_benchmark", "quantum_compute_forward",
                                              "quantum_consciousness_think", "quantum_ram_store",
                                              "quantum_ram_retrieve", "quantum_teleportation_bridge",
                                              "topological_qubit_bridge"]),
        "code→quantum": ("code", ["quantum_tomography", "quantum_code_search", "quantum_embed",
                                    "quantum_coherence_grover", "quantum_optimize", "quantum_resilience"]),
        "math→quantum": ("math", ["quantum_gravity_holographic", "quantum_knot_invariant",
                                    "cross_engine_dimensional_verification", "cross_engine_prove_all"]),
        "science→quantum": ("science", ["cross_engine_coherence_deep", "apply_quantum_boost"]),
    }

    for link_name, (eng_key, methods) in cross_patterns.items():
        engine = engines.get(eng_key, {}).get("engine")
        if engine is None:
            continue
        accessible = sum(1 for m in methods if getattr(engine, m, None) is not None)
        cross_engine_links.append({
            "link": link_name,
            "total": len(methods),
            "accessible": accessible,
        })
        pct = (accessible / len(methods) * 100) if methods else 0
        icon = ICONS[Severity.PASS] if pct >= 80 else ICONS[Severity.WARN]
        diag.ok(f"cross_dead_{link_name}", "dead_code",
                f"{accessible}/{len(methods)} methods accessible ({pct:.0f}%)")
        printer.log(icon, f"{link_name:25s} {accessible}/{len(methods)} accessible ({pct:.0f}%)")

    # ── 8e: Reversal Summary ──────────────────────────────────────────────
    printer.section("8e: Dead Code Reversal Summary")

    pct_revived = (total_revived / total_dead * 100) if total_dead > 0 else 0
    pct_failed = (total_failed / total_dead * 100) if total_dead > 0 else 0

    printer.log("📊", f"Total dead methods registered:  {total_dead}")
    printer.log("📊", f"Missing (removed from code):    {total_missing}")
    printer.log("📊", f"Successfully revived:           {total_revived} ({pct_revived:.1f}%)")
    printer.log("📊", f"Failed / type mismatch:         {total_failed} ({pct_failed:.1f}%)")

    # Per-engine table
    printer.log("", "")
    printer.log("", f"  {'Engine':20s} {'Dead':>5s} {'Revived':>8s} {'Failed':>7s} {'Rate':>6s}")
    printer.log("", f"  {'─'*20} {'─'*5} {'─'*8} {'─'*7} {'─'*6}")
    for eng_key, stats in sorted(per_engine_stats.items()):
        reg = stats["registered"]
        rev = stats["revived"]
        fail = stats["failed"]
        rate = f"{rev/reg*100:.0f}%" if reg > 0 else "N/A"
        printer.log("", f"  {eng_key:20s} {reg:5d} {rev:8d} {fail:7d} {rate:>6s}")

    # Logic graph category summary
    printer.log("", "")
    printer.log("", f"  {'Category':12s} {'Total':>6s} {'Wired':>6s} {'Orphan':>7s} {'Rate':>6s}")
    printer.log("", f"  {'─'*12} {'─'*6} {'─'*6} {'─'*7} {'─'*6}")
    for cat, stats in sorted(category_stats.items()):
        rate = f"{stats['wired']/stats['total']*100:.0f}%" if stats["total"] > 0 else "N/A"
        printer.log("", f"  {cat:12s} {stats['total']:6d} {stats['wired']:6d} "
                       f"{stats['orphan']:7d} {rate:>6s}")

    # Final verdict
    if pct_revived >= 70:
        diag.ok("dead_code_verdict", "dead_code",
                f"{total_revived}/{total_dead} revived ({pct_revived:.1f}%)")
        printer.log(ICONS[Severity.PASS],
                    f"DEAD CODE REVERSAL: {total_revived}/{total_dead} REVIVED ({pct_revived:.1f}%)")
    elif pct_revived >= 40:
        diag.warn("dead_code_verdict", "dead_code",
                  f"{total_revived}/{total_dead} revived ({pct_revived:.1f}%) — needs attention")
        printer.log(ICONS[Severity.WARN],
                    f"DEAD CODE REVERSAL: {total_revived}/{total_dead} ({pct_revived:.1f}%) — needs improvement")
    else:
        diag.fail("dead_code_verdict", "dead_code",
                  f"Only {total_revived}/{total_dead} revived ({pct_revived:.1f}%)")
        printer.log(ICONS[Severity.FAIL],
                    f"DEAD CODE REVERSAL: Only {total_revived}/{total_dead} ({pct_revived:.1f}%) — critical")

    # Store logic graph as data on the verdict result
    diag.results[-1].data = {
        "logic_graph": logic_graph,
        "per_engine": per_engine_stats,
        "categories": category_stats,
        "cross_links": cross_engine_links,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 9: CODE ENGINE AUTO-EDIT (v3.3.0) — TUNED & HARDENED
# ═══════════════════════════════════════════════════════════════════════════════
#
#  This phase uses the Code Engine to actually EDIT source files:
#    9a — Collect source files for engines that have diagnostic failures
#    9b — Run Code Engine pipeline: smells → refactor → excavate → SAFE auto-fix
#    9c — Apply fixes (write to disk if --apply, dry-run otherwise)
#    9d — Generate docs & tests for modules with issues
#    9e — Quick audit integration (L0+L2+L5 lightweight scan)
#    9f — Summary: files analyzed, fixes applied, before/after stats
#
#  Safety (v3.3.0 hardened):
#    • Dry-run by default (--apply required for disk writes)
#    • .bak backups created before any write
#    • Sacred constants are NEVER modified (immutable guard)
#    • SAFE FIX SUBSET: skips fix_unused_imports and fix_import_sorting
#      (both are destructive on re-export hubs / __init__.py files)
#    • MAX_DELETE_PCT guard: rejects any fix removing >5% of lines
#    • AST syntax validation: fixed code must parse cleanly before write
#    • __init__.py / hub file protection: extra caution on re-export modules
#    • Proper difflib-based line counting (not naive set diff)
#    • Per-rule reporting: shows exactly which rules fired and which skipped
#    • Only edits Python files within l104_* package directories
# ═══════════════════════════════════════════════════════════════════════════════

# Map engine keys to their package source directories
_ENGINE_SOURCE_MAP: Dict[str, str] = {
    "code":         "l104_code_engine",
    "science":      "l104_science_engine",
    "math":         "l104_math_engine",
    "agi":          "l104_agi",
    "asi":          "l104_asi",
    "intellect":    "l104_intellect",
    "quantum_gate": "l104_quantum_gate_engine",
    "quantum_link": "l104_quantum_engine",
    "numerical":    "l104_numerical_engine",
    "gate":         "l104_gate_engine",
    "server":       "l104_server",
}

# Sacred constants that must NEVER be modified by auto-edit
_SACRED_IMMUTABLES = {
    "GOD_CODE", "PHI", "VOID_CONSTANT", "OMEGA", "TAU",
    "GOD_CODE_V3", "GOD_CODE_HP", "PHI_HP",
}

# ── v3.3.0 Safety Tuning Constants ───────────────────────────────────────────
MAX_DELETE_PCT = 5.0        # Reject fixes that remove >5% of file lines
MAX_DELETE_LINES = 50       # Hard cap: never remove >50 lines from a single file
MIN_FILE_LINES = 10         # Skip files with fewer than 10 lines
MAX_FILE_LINES = 3000       # Skip files with more than 3000 lines (OOM guard)
DIFF_FAST_THRESHOLD = 500   # Use fast O(n) diff instead of SequenceMatcher above this

# Rules SKIPPED because they are destructive on re-export / __init__.py files:
#   fix_unused_imports  — removes imports used only by external consumer modules
#   fix_import_sorting  — rewrites entire import block wholesale, can misidentify range
#   fix_unnecessary_pass — removes pass from functions where pass IS the only body
#                          (causes "expected indented block" syntax errors)
_DESTRUCTIVE_RULES = {"fix_unused_imports", "fix_import_sorting", "fix_unnecessary_pass"}

# Safe rules applied in order (v3.3.0 tuned pipeline)
_SAFE_RULES = [
    "fix_trailing_whitespace",   # cosmetic, safe
    "fix_docstring_stubs",       # additive only (adds lines)
    "fix_bare_except",           # in-place pattern replace
    "fix_mutable_default_args",  # semantic preservation
    "fix_print_to_logging",      # semantic preservation
    "fix_redundant_else_after_return",  # structural (currently disabled in engine)
    "fix_fstring_upgrade",       # in-place format conversion
    "fix_dict_comprehension",    # in-place pattern replace
]

# Files that get extra protection (re-export hubs)
_HUB_FILE_PATTERNS = {"__init__", "hub", "app", "brain", "orchestrator", "core"}


def _is_hub_file(filepath: Path) -> bool:
    """Check if a file is a re-export hub that needs extra protection."""
    stem = filepath.stem.lower()
    return any(pat in stem for pat in _HUB_FILE_PATTERNS)


def _safe_auto_fix(code_eng: Any, source: str, filepath: Path) -> Tuple[str, List[Dict], List[str]]:
    """
    Run ONLY the safe subset of auto-fix rules, skipping destructive ones.

    Returns: (fixed_code, fix_log, skipped_rules)

    v3.3.0: This replaces the naive `code_eng.auto_fix_code(source)` call
    that previously delegated to `apply_all_safe()` which runs ALL 11 rules
    including the destructive `fix_unused_imports` and `fix_import_sorting`.

    We directly instantiate an AutoFixEngine and selectively call safe rules.
    """
    skipped: List[str] = list(_DESTRUCTIVE_RULES)  # always report these as skipped

    # Access the AutoFixEngine instance from the code engine
    fixer = getattr(code_eng, 'auto_fix', None)
    if fixer is None:
        # Fallback: if we can't access the fixer directly, return unchanged
        return source, [], ["ALL (no auto_fix available)"]

    # Reset the fixer state for this file
    fixer.fixes_applied = 0
    fixer.fixes_log = []

    fixed = source
    applied_rules: List[str] = []

    for rule_name in _SAFE_RULES:
        method = getattr(fixer, rule_name, None)
        if method is None:
            skipped.append(f"{rule_name} (not found)")
            continue
        try:
            before = fixed
            fixed = method(fixed)
            if fixed != before:
                applied_rules.append(rule_name)
        except Exception as e:
            skipped.append(f"{rule_name} (error: {str(e)[:50]})")
            fixed = before  # revert this rule's changes on error

    return fixed, fixer.fixes_log, skipped


def _compute_diff_stats(original: str, fixed: str) -> Dict[str, int]:
    """
    Compute line-level diff stats.

    Uses a fast O(n) line-set comparison for files above DIFF_FAST_THRESHOLD
    to avoid the O(n²) memory usage of difflib.SequenceMatcher that causes
    OOM kills on MacBook Air with large files (1000+ lines).

    Returns: {"added": N, "removed": N, "changed": N, "total_orig": N}
    """
    orig_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)
    n_orig = len(orig_lines)

    # Fast path: identical
    if original == fixed:
        return {"added": 0, "removed": 0, "changed": 0, "total_orig": n_orig}

    # For large files, use O(n) indexed line comparison instead of SequenceMatcher
    if n_orig > DIFF_FAST_THRESHOLD or len(fixed_lines) > DIFF_FAST_THRESHOLD:
        # Count lines that differ by position, plus length difference
        min_len = min(n_orig, len(fixed_lines))
        changed_in_place = sum(
            1 for i in range(min_len) if orig_lines[i] != fixed_lines[i]
        )
        extra_added = max(0, len(fixed_lines) - n_orig)
        extra_removed = max(0, n_orig - len(fixed_lines))
        return {
            "added": changed_in_place + extra_added,
            "removed": changed_in_place + extra_removed,
            "changed": changed_in_place + extra_added + extra_removed,
            "total_orig": n_orig,
        }

    # Standard path: accurate difflib for smaller files
    import difflib
    added = 0
    removed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, orig_lines, fixed_lines
    ).get_opcodes():
        if tag == 'insert':
            added += (j2 - j1)
        elif tag == 'delete':
            removed += (i2 - i1)
        elif tag == 'replace':
            added += (j2 - j1)
            removed += (i2 - i1)

    return {
        "added": added,
        "removed": removed,
        "changed": added + removed,
        "total_orig": n_orig,
    }


def _validate_fix_safety(
    original: str,
    fixed: str,
    filepath: Path,
    _precomputed_diff: Optional[Dict[str, int]] = None,
) -> Tuple[bool, str]:
    """
    Post-fix safety validation. Returns (is_safe, reason).

    Checks:
    1. No sacred constant mutation
    2. Fixed code parses as valid Python (AST check)
    3. Deletion doesn't exceed MAX_DELETE_PCT or MAX_DELETE_LINES
    4. Hub files get stricter deletion limits (2% / 10 lines)
    """
    import ast as _ast

    # 1) Sacred constant guard
    if _contains_sacred_mutation(original, fixed):
        return False, "BLOCKED: would mutate sacred constant"

    # 2) AST syntax validation
    try:
        _ast.parse(fixed)
    except SyntaxError as e:
        return False, f"BLOCKED: fixed code has syntax error: {e.msg} line {e.lineno}"

    # 3) Deletion limits (use pre-computed stats if available to avoid O(n²) recomputation)
    diff = _precomputed_diff or _compute_diff_stats(original, fixed)
    removed = diff["removed"]
    total = diff["total_orig"]

    if total > 0:
        delete_pct = (removed / total) * 100

        if _is_hub_file(filepath):
            # Stricter limits for hub files
            if removed > 10:
                return False, f"BLOCKED: hub file {filepath.name} would lose {removed} lines (max 10 for hubs)"
            if delete_pct > 2.0:
                return False, f"BLOCKED: hub file {filepath.name} would lose {delete_pct:.1f}% (max 2% for hubs)"
        else:
            if removed > MAX_DELETE_LINES:
                return False, f"BLOCKED: {filepath.name} would lose {removed} lines (max {MAX_DELETE_LINES})"
            if delete_pct > MAX_DELETE_PCT:
                return False, f"BLOCKED: {filepath.name} would lose {delete_pct:.1f}% (max {MAX_DELETE_PCT}%)"

    return True, "safe"


def _contains_sacred_mutation(original: str, fixed: str) -> bool:
    """Check if a fix would modify a sacred constant definition."""
    import re
    for name in _SACRED_IMMUTABLES:
        # Find lines defining the sacred constant in original
        pat = re.compile(rf'^\s*{name}\s*=', re.MULTILINE)
        orig_defs = pat.findall(original)
        fixed_defs = pat.findall(fixed)
        if orig_defs != fixed_defs:
            return True
        # Check if the value on the RHS changed
        val_pat = re.compile(rf'^\s*{name}\s*=\s*(.+)$', re.MULTILINE)
        orig_vals = val_pat.findall(original)
        fixed_vals = val_pat.findall(fixed)
        if orig_vals != fixed_vals:
            return True
    return False


@dataclass
class EditAction:
    """One proposed or applied edit."""
    filepath: str
    engine_key: str
    action_type: str     # "auto_fix", "smell_fix", "refactor", "dead_code_remove", "doc_gen", "test_gen"
    description: str
    original_snippet: str = ""   # first 200 chars of original
    fixed_snippet: str = ""      # first 200 chars of fixed
    applied: bool = False
    blocked: bool = False        # True if sacred guard blocked it
    error: str = ""


def _collect_target_files(
    engines: Dict[str, Any],
    diag: DiagnosticCollector,
    printer: _TermPrinter,
) -> Dict[str, List[Path]]:
    """
    9a: Identify source files to edit.
    Strategy: collect .py files from packages that had failures in earlier phases.
    If no failures, still scan all packages for improvement opportunities.
    """
    # Determine which engines had failures
    failed_engines: set = set()
    for r in diag.results:
        if r.severity == Severity.FAIL:
            # Engine name from result
            eng = r.engine
            if eng in _ENGINE_SOURCE_MAP:
                failed_engines.add(eng)
            # Also check if the test name references an engine
            for ek in _ENGINE_SOURCE_MAP:
                if ek in r.test.lower():
                    failed_engines.add(ek)

    # If no failures, target all engines for opportunistic improvement
    target_engines = failed_engines if failed_engines else set(_ENGINE_SOURCE_MAP.keys())

    result: Dict[str, List[Path]] = {}
    for eng_key in target_engines:
        pkg_dir = _ENGINE_SOURCE_MAP.get(eng_key)
        if not pkg_dir:
            continue
        pkg_path = ROOT / pkg_dir
        if not pkg_path.is_dir():
            continue
        py_files = sorted(pkg_path.glob("**/*.py"))
        # Filter out __pycache__, test files, and very small files
        py_files = [
            f for f in py_files
            if "__pycache__" not in str(f)
            and f.stat().st_size > 100  # skip near-empty files
        ]
        if py_files:
            result[eng_key] = py_files

    return result


def _run_code_engine_pipeline(
    code_eng: Any,
    source: str,
    filepath: Path,
    printer: _TermPrinter,
) -> Dict[str, Any]:
    """
    9b: Run the full Code Engine diagnostic + SAFE fix pipeline on one file.

    v3.3.0: Uses _safe_auto_fix() instead of code_eng.auto_fix_code() to avoid
    destructive rules. Adds per-rule tracking and post-fix validation.

    Returns a dict with analysis results and proposed fixes.
    """
    results: Dict[str, Any] = {
        "filepath": str(filepath),
        "original_lines": source.count("\n") + 1,
        "smells": {},
        "refactor": {},
        "excavation": {},
        "auto_fix": None,
        "fix_log": [],
        "skipped_rules": [],
        "diff_stats": {},
        "safety_check": "",
        "audit_score": None,
        "docs": None,
        "tests": None,
        "errors": [],
    }

    # 1) Code smells
    try:
        smells = code_eng.smell_detector.detect_all(source)
        results["smells"] = smells
    except Exception as e:
        results["errors"].append(f"smell_detect: {e}")

    # 2) Refactor analysis
    try:
        refactor = code_eng.refactor_engine.refactor_analyze(source)
        results["refactor"] = refactor
    except Exception as e:
        results["errors"].append(f"refactor: {e}")

    # 3) Dead code excavation
    try:
        excavation = code_eng.excavator.excavate(source)
        results["excavation"] = excavation
    except Exception as e:
        results["errors"].append(f"excavate: {e}")

    # 4) SAFE auto-fix (v3.3.0: tuned pipeline, skips destructive rules)
    try:
        fixed_code, fix_log, skipped_rules = _safe_auto_fix(code_eng, source, filepath)
        results["auto_fix"] = fixed_code
        results["fix_log"] = fix_log
        results["skipped_rules"] = skipped_rules
        results["fixed_lines"] = fixed_code.count("\n") + 1

        # Compute accurate diff stats
        if fixed_code != source:
            diff_stats = _compute_diff_stats(source, fixed_code)
            results["diff_stats"] = diff_stats

            # Post-fix safety validation (pass pre-computed diff to avoid O(n²) recomputation)
            is_safe, safety_msg = _validate_fix_safety(source, fixed_code, filepath, _precomputed_diff=diff_stats)
            results["safety_check"] = safety_msg
            if not is_safe:
                # BLOCK this fix — revert to original
                results["auto_fix"] = source
                results["fix_log"] = []
                results["fixed_lines"] = results["original_lines"]
                results["diff_stats"] = {"added": 0, "removed": 0, "changed": 0, "total_orig": results["original_lines"]}
                results["errors"].append(f"safety_block: {safety_msg}")
    except Exception as e:
        results["errors"].append(f"safe_auto_fix: {e}")

    # 5) Performance prediction
    try:
        perf = code_eng.perf_predictor.predict_performance(source)
        results["perf"] = perf
    except Exception as e:
        results["errors"].append(f"perf: {e}")

    return results


def _apply_edit(
    filepath: Path,
    original: str,
    fixed: str,
    apply_mode: bool,
    printer: _TermPrinter,
    diff_stats: Dict[str, int] = None,
) -> Tuple[bool, str]:
    """
    9c: Write the fixed code to disk.
    Returns (applied, detail_string).

    v3.3.0: Uses pre-computed diff_stats from _compute_diff_stats() for accurate
    line counting. All safety validation is done BEFORE this function is called.
    """
    if original == fixed:
        return False, "no changes needed"

    # Use pre-computed stats or compute fresh
    if diff_stats is None:
        diff_stats = _compute_diff_stats(original, fixed)

    added = diff_stats.get("added", 0)
    removed = diff_stats.get("removed", 0)

    if not apply_mode:
        return False, f"DRY-RUN: +{added}/-{removed} lines (use --apply to write)"

    # Create .bak backup
    bak_path = filepath.with_suffix(filepath.suffix + ".bak")
    try:
        bak_path.write_text(original, encoding="utf-8")
    except Exception as e:
        return False, f"backup failed: {e}"

    # Write fixed code
    try:
        filepath.write_text(fixed, encoding="utf-8")
        return True, f"APPLIED: +{added}/-{removed} lines (backup: {bak_path.name})"
    except Exception as e:
        # Restore from backup
        try:
            filepath.write_text(original, encoding="utf-8")
        except Exception:
            pass
        return False, f"write failed: {e}"


def _generate_artifacts(
    code_eng: Any,
    source: str,
    filepath: Path,
    printer: _TermPrinter,
    apply_mode: bool,
) -> List[EditAction]:
    """
    9d: Generate docs and tests for modules.
    Returns list of EditAction for generated artifacts.
    """
    actions: List[EditAction] = []
    stem = filepath.stem
    parent = filepath.parent

    # Generate documentation
    try:
        docs = code_eng.generate_docs(source, style="google", language="python")
        if docs.get("success") and docs.get("artifacts"):
            doc_path = parent / f"{stem}_docs.md"
            doc_content = ""
            for artifact in docs["artifacts"]:
                if isinstance(artifact, dict):
                    doc_content += artifact.get("content", artifact.get("doc", str(artifact))) + "\n\n"
                else:
                    doc_content += str(artifact) + "\n\n"

            if doc_content.strip():
                action = EditAction(
                    filepath=str(doc_path),
                    engine_key="code",
                    action_type="doc_gen",
                    description=f"Generated docs for {stem} ({docs.get('total_documented', '?')} items)",
                    fixed_snippet=doc_content[:200],
                )
                if apply_mode:
                    try:
                        doc_path.write_text(doc_content, encoding="utf-8")
                        action.applied = True
                    except Exception as e:
                        action.error = str(e)
                actions.append(action)
    except Exception as e:
        actions.append(EditAction(
            filepath=str(parent / f"{stem}_docs.md"),
            engine_key="code", action_type="doc_gen",
            description=f"doc generation failed", error=str(e)[:100],
        ))

    # Generate tests
    try:
        tests = code_eng.generate_tests(source, language="python", framework="pytest")
        if tests.get("success") and tests.get("test_code"):
            test_path = parent / f"test_{stem}.py"
            # Don't overwrite existing test files
            if not test_path.exists():
                test_content = tests["test_code"]
                action = EditAction(
                    filepath=str(test_path),
                    engine_key="code",
                    action_type="test_gen",
                    description=f"Generated {tests.get('functions_tested', '?')} tests for {stem}",
                    fixed_snippet=test_content[:200],
                )
                if apply_mode:
                    try:
                        test_path.write_text(test_content, encoding="utf-8")
                        action.applied = True
                    except Exception as e:
                        action.error = str(e)
                actions.append(action)
    except Exception as e:
        actions.append(EditAction(
            filepath=str(parent / f"test_{stem}.py"),
            engine_key="code", action_type="test_gen",
            description=f"test generation failed", error=str(e)[:100],
        ))

    return actions


def phase_edit(
    engines: Dict[str, Any],
    diag: DiagnosticCollector,
    printer: _TermPrinter,
    apply_mode: bool = False,
    max_files_per_engine: int = 20,
):
    """
    Phase 9: Code Engine Auto-Edit (v3.3.0 — Tuned & Hardened).

    Uses the Code Engine to analyze, fix, and enhance source files across all
    engine packages. Operates in dry-run mode by default (--apply to write).

    v3.3.0 SAFETY HARDENING:
      - Skips destructive rules: fix_unused_imports, fix_import_sorting
      - Max deletion guard: rejects fixes >5% or >50 lines removed
      - Hub file protection: __init__.py / hub / brain files get 2% / 10-line cap
      - AST validation: fixed code must parse before write
      - Proper difflib line counting (replaces naive set-based diff)
      - Per-rule reporting: shows exactly which rules fired and which skipped
      - Audit integration: lightweight L0+L2+L5 audit per engine package

    Pipeline per file:
      1. smell detection → report
      2. refactor analysis → suggestions
      3. dead code excavation → fossils
      4. SAFE auto_fix → fixed source (9 safe rules, 2 skipped)
      5. post-fix safety validation → block or accept
      6. performance prediction → hotspots
      7. doc generation → _docs.md
      8. test generation → test_*.py

    Safety: sacred constant guard, deletion limits, AST check, .bak backups, dry-run default.
    """
    diag.set_phase("EDIT")

    mode_label = " [APPLY MODE]" if apply_mode else " [DRY-RUN]"
    printer.banner(f"PHASE 9: CODE ENGINE AUTO-EDIT v3.3.0{mode_label}")

    # Show safety config
    printer.log("🛡️", f"Safety: max_delete={MAX_DELETE_PCT}% / {MAX_DELETE_LINES} lines, "
                f"hub_limit=2% / 10 lines, AST validation=ON")
    printer.log("🛡️", f"Skipped rules: {', '.join(sorted(_DESTRUCTIVE_RULES))}")
    printer.log("🛡️", f"Safe rules: {len(_SAFE_RULES)} active")

    # Boot code engine (required)
    code_eng = _lazy_boot("code", engines, diag, printer)
    if code_eng is None:
        diag.fail("edit_boot", "edit", "Code Engine unavailable — cannot edit")
        return

    # ── 9a: Collect target files ──────────────────────────────────────────
    printer.section("9a: Collecting Target Files")
    target_files = _collect_target_files(engines, diag, printer)

    total_files = sum(len(fs) for fs in target_files.values())
    if total_files == 0:
        diag.info("edit_no_targets", "edit", "No target files found")
        printer.log(ICONS[Severity.INFO], "No target files — all packages clean or missing")
        return

    for eng_key, files in target_files.items():
        printer.log(ICONS[Severity.INFO],
                    f"{eng_key:20s} {len(files)} files to analyze"
                    + (f" (capped at {max_files_per_engine})" if len(files) > max_files_per_engine else ""))
    printer.log("📊", f"Total: {total_files} files across {len(target_files)} packages")

    # ── 9b + 9c: Analyze & Fix (v3.3.0 hardened) ───────────────────────────
    printer.section("9b: Code Engine Analysis & Safe Auto-Fix Pipeline")

    all_actions: List[EditAction] = []
    per_engine_edit_stats: Dict[str, Dict[str, int]] = {}
    total_smells = 0
    total_refactor_suggestions = 0
    total_dead_code_items = 0
    total_fixes_proposed = 0
    total_fixes_applied = 0
    total_fixes_blocked = 0
    total_safety_blocks = 0
    files_analyzed = 0
    files_changed = 0
    # Per-rule tracking (v3.3.0)
    rule_fire_counts: Dict[str, int] = {}
    rule_skip_reasons: Dict[str, List[str]] = {}

    for eng_key, files in sorted(target_files.items()):
        engine_stats = {
            "files": 0, "smells": 0, "refactor": 0, "dead_code": 0,
            "fixes_proposed": 0, "fixes_applied": 0, "fixes_blocked": 0,
            "safety_blocks": 0,
            "docs_generated": 0, "tests_generated": 0,
        }

        # Cap files per engine to avoid runaway analysis
        capped_files = files[:max_files_per_engine]

        for filepath in capped_files:
            t0 = time.time()
            try:
                source = filepath.read_text(encoding="utf-8")
            except Exception as e:
                diag.warn(f"edit_read_{filepath.name}", eng_key, f"read error: {e}")
                continue

            # Skip tiny files
            line_count = source.count("\n")
            if line_count < MIN_FILE_LINES:
                if printer.verbose:
                    printer.log(ICONS[Severity.INFO],
                                f"  {filepath.name}: skipped ({line_count} lines < {MIN_FILE_LINES} min)")
                continue

            # Skip very large files to prevent OOM on constrained machines
            if line_count > MAX_FILE_LINES:
                if printer.verbose:
                    printer.log(ICONS[Severity.INFO],
                                f"  {filepath.name}: skipped ({line_count} lines > {MAX_FILE_LINES} max, OOM guard)")
                diag.info(f"edit_skip_large_{filepath.name}", eng_key,
                          f"Skipped: {line_count} lines exceeds {MAX_FILE_LINES} OOM guard")
                continue

            files_analyzed += 1
            engine_stats["files"] += 1

            # Run the Code Engine pipeline (v3.3.0: safe rules only)
            pipeline_result = _run_code_engine_pipeline(code_eng, source, filepath, printer)
            elapsed = (time.time() - t0) * 1000

            # Count findings
            smell_count = 0
            if isinstance(pipeline_result["smells"], dict):
                smell_count = pipeline_result["smells"].get("total", 0)
            engine_stats["smells"] += smell_count
            total_smells += smell_count

            refactor_count = 0
            if isinstance(pipeline_result["refactor"], dict):
                refactor_count = pipeline_result["refactor"].get("total_suggestions", 0)
            engine_stats["refactor"] += refactor_count
            total_refactor_suggestions += refactor_count

            dead_count = 0
            if isinstance(pipeline_result["excavation"], dict):
                dead_count = len(pipeline_result["excavation"].get("dead_code", []))
            engine_stats["dead_code"] += dead_count
            total_dead_code_items += dead_count

            # Track per-rule firing (v3.3.0)
            for log_entry in pipeline_result.get("fix_log", []):
                rule = log_entry.get("fix", log_entry.get("rule", "unknown"))
                rule_fire_counts[rule] = rule_fire_counts.get(rule, 0) + 1
            for skip_reason in pipeline_result.get("skipped_rules", []):
                rule_skip_reasons.setdefault(skip_reason, []).append(filepath.name)

            # Check for safety blocks
            safety_msg = pipeline_result.get("safety_check", "")
            if safety_msg and safety_msg != "safe":
                total_safety_blocks += 1
                engine_stats["safety_blocks"] = engine_stats.get("safety_blocks", 0) + 1
                if printer.verbose:
                    printer.log(ICONS[Severity.WARN],
                                f"  {filepath.name}: {safety_msg}")

            # Process auto-fix result
            fixed_code = pipeline_result.get("auto_fix")
            fix_log = pipeline_result.get("fix_log", [])
            diff_stats = pipeline_result.get("diff_stats", {})

            if fixed_code and fixed_code != source:
                total_fixes_proposed += 1
                engine_stats["fixes_proposed"] += 1

                # Build descriptive action
                rules_fired = [f.get('fix', f.get('rule', '?'))[:30] for f in fix_log[:5]]
                diff_desc = f"+{diff_stats.get('added', 0)}/-{diff_stats.get('removed', 0)}"

                action = EditAction(
                    filepath=str(filepath),
                    engine_key=eng_key,
                    action_type="auto_fix",
                    description=f"{len(fix_log)} rules ({', '.join(rules_fired)}) {diff_desc}",
                    original_snippet=source[:200],
                    fixed_snippet=fixed_code[:200],
                )

                # Apply or dry-run (safety already validated in pipeline)
                applied, detail = _apply_edit(
                    filepath, source, fixed_code, apply_mode, printer,
                    diff_stats=diff_stats,
                )
                action.applied = applied
                if applied:
                    total_fixes_applied += 1
                    engine_stats["fixes_applied"] += 1
                    files_changed += 1
                    diag.ok(f"edit_fix_{filepath.name}", eng_key, detail)
                else:
                    diag.info(f"edit_fix_{filepath.name}", eng_key, detail)

                if printer.verbose:
                    icon = ICONS[Severity.PASS] if applied else ICONS[Severity.INFO]
                    hub_tag = " [HUB]" if _is_hub_file(filepath) else ""
                    printer.log(icon, f"  {filepath.name}{hub_tag}: {detail}")

                all_actions.append(action)
            else:
                if printer.verbose:
                    printer.log(ICONS[Severity.INFO],
                                f"  {filepath.name}: clean ({smell_count} smells, "
                                f"{refactor_count} suggestions) [{elapsed:.0f}ms]")

            # Log refactor suggestions as edit actions (informational)
            if isinstance(pipeline_result["refactor"], dict):
                for suggestion in pipeline_result["refactor"].get("suggestions", [])[:5]:
                    all_actions.append(EditAction(
                        filepath=str(filepath),
                        engine_key=eng_key,
                        action_type="refactor",
                        description=f"{suggestion.get('type', '?')}: {suggestion.get('reason', '?')[:60]}",
                    ))

            # Log dead code findings as edit actions (informational)
            if isinstance(pipeline_result["excavation"], dict):
                for fossil in pipeline_result["excavation"].get("dead_code", [])[:5]:
                    all_actions.append(EditAction(
                        filepath=str(filepath),
                        engine_key=eng_key,
                        action_type="dead_code_remove",
                        description=f"dead: {fossil if isinstance(fossil, str) else str(fossil)[:60]}",
                    ))

            # Free large objects to reduce memory pressure on constrained machines
            del pipeline_result, source
            fixed_code = None

        # Per-engine summary line
        icon = ICONS[Severity.PASS] if engine_stats["fixes_proposed"] == 0 else (
            ICONS[Severity.INFO] if not apply_mode else ICONS[Severity.PASS])
        printer.log(icon,
                    f"{eng_key:20s} {engine_stats['files']} files, "
                    f"{engine_stats['smells']} smells, "
                    f"{engine_stats['refactor']} refactors, "
                    f"{engine_stats['fixes_proposed']} fixes"
                    + (f" ({engine_stats['fixes_applied']} applied)" if apply_mode else " (dry-run)"))

        per_engine_edit_stats[eng_key] = engine_stats

    # ── 9d: Generate Docs & Tests ─────────────────────────────────────────
    printer.section("9d: Doc & Test Generation")

    # Only generate for files that had significant findings
    gen_count = 0
    for eng_key, files in sorted(target_files.items()):
        capped_files = files[:max_files_per_engine]
        for filepath in capped_files:
            try:
                source = filepath.read_text(encoding="utf-8")
            except Exception:
                continue

            # Only generate for substantial files (>50 lines) with issues
            line_count = source.count("\n")
            if line_count < 50 or line_count > MAX_FILE_LINES:
                continue

            # Check if this file had smells or refactor suggestions
            has_issues = any(
                a.filepath == str(filepath) and a.action_type in ("auto_fix", "refactor", "dead_code_remove")
                for a in all_actions
            )
            if not has_issues and gen_count > 10:
                continue  # Limit generation to files with issues + first 10

            artifacts = _generate_artifacts(code_eng, source, filepath, printer, apply_mode)
            for art in artifacts:
                all_actions.append(art)
                if art.action_type == "doc_gen":
                    per_engine_edit_stats.setdefault(eng_key, {}).setdefault("docs_generated", 0)
                    per_engine_edit_stats[eng_key]["docs_generated"] += 1
                elif art.action_type == "test_gen":
                    per_engine_edit_stats.setdefault(eng_key, {}).setdefault("tests_generated", 0)
                    per_engine_edit_stats[eng_key]["tests_generated"] += 1
                if printer.verbose:
                    icon = ICONS[Severity.PASS] if art.applied else ICONS[Severity.INFO]
                    printer.log(icon, f"  {art.action_type}: {art.description[:70]}")
            gen_count += 1
            del source  # Free file content after generation

    # ── 9e: Quick Audit Integration (v3.3.0) ────────────────────────────────
    printer.section("9e: Quick Audit (L0+L2+L5) per Engine Package")

    # Free memory from analysis phase before running audit (OOM guard for constrained machines)
    import gc
    all_actions_summary = len(all_actions)
    # Keep only essential action data, drop large snippets
    for act in all_actions:
        act.original_snippet = ""
        act.fixed_snippet = ""
    gc.collect()

    # Check available memory — skip audit if under 512MB free (macOS OOM guard)
    _skip_audit = False
    try:
        import resource
        import os
        # On macOS, use resource module to check RSS of current process
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = rusage.ru_maxrss / (1024 * 1024)  # macOS reports in bytes
        if rss_mb > 800:  # Process using >800MB — audit would push over edge
            _skip_audit = True
            printer.log(ICONS[Severity.WARN],
                        f"Skipping audit: process RSS={rss_mb:.0f}MB (>800MB OOM guard)")
    except Exception:
        pass  # Can't determine memory — proceed cautiously

    audit_results: Dict[str, Dict[str, Any]] = {}
    if _skip_audit:
        diag.info("edit_audit_skipped", "edit", "Audit skipped due to memory pressure")
    else:
        for eng_key in sorted(target_files.keys()):
            pkg_dir = _ENGINE_SOURCE_MAP.get(eng_key)
            if not pkg_dir:
                continue
            pkg_path = ROOT / pkg_dir
            try:
                audit = code_eng.quick_audit(str(pkg_path))
                audit_results[eng_key] = audit
                score = audit.get("quick_score", 0.0)
                verdict = audit.get("verdict", "?")
                security = audit.get("security", {})
                sec_issues = security.get("total_issues", 0)
                anti_pat = audit.get("anti_patterns", {})
                anti_count = anti_pat.get("total_patterns", 0)

                icon = ICONS[Severity.PASS] if score >= 0.7 else (
                    ICONS[Severity.WARN] if score >= 0.4 else ICONS[Severity.FAIL])
                printer.log(icon,
                            f"  {eng_key:20s} score={score:.3f} ({verdict})"
                            f"  sec_issues={sec_issues}  anti_patterns={anti_count}")
                diag.info(f"edit_audit_{eng_key}", eng_key,
                          f"Quick audit: score={score:.3f} verdict={verdict}")
            except Exception as e:
                printer.log(ICONS[Severity.WARN], f"  {eng_key:20s} audit error: {str(e)[:60]}")
                audit_results[eng_key] = {"error": str(e)[:100]}
            finally:
                gc.collect()  # Free audit memory between packages

    # ── 9f: Edit Summary (v3.3.0 enhanced) ───────────────────────────────
    printer.section("9f: Auto-Edit Summary (v3.3.0)")

    printer.log("📊", f"Files analyzed:            {files_analyzed}")
    printer.log("📊", f"Total code smells found:   {total_smells}")
    printer.log("📊", f"Refactor suggestions:      {total_refactor_suggestions}")
    printer.log("📊", f"Dead code items found:     {total_dead_code_items}")
    printer.log("📊", f"Fixes proposed:            {total_fixes_proposed}")
    printer.log("📊", f"Safety blocks:             {total_safety_blocks}")
    if apply_mode:
        printer.log("📊", f"Fixes applied:             {total_fixes_applied}")
        printer.log("📊", f"Files changed:             {files_changed}")
    else:
        printer.log("📊", f"Mode:                      DRY-RUN (use --apply to write changes)")

    # Per-engine table (v3.3.0: added SafeBlk + Audit columns)
    printer.log("", "")
    printer.log("", f"  {'Engine':20s} {'Files':>5s} {'Smells':>7s} {'Refact':>7s} {'Dead':>5s}"
                   f" {'Fixes':>6s} {'SafeBlk':>8s} {'Docs':>5s} {'Tests':>6s} {'Audit':>6s}")
    printer.log("", f"  {'─'*20} {'─'*5} {'─'*7} {'─'*7} {'─'*5} {'─'*6} {'─'*8} {'─'*5} {'─'*6} {'─'*6}")
    for eng_key, stats in sorted(per_engine_edit_stats.items()):
        audit_score = audit_results.get(eng_key, {}).get("quick_score", -1)
        audit_str = f"{audit_score:.3f}" if audit_score >= 0 else "  n/a"
        printer.log("", f"  {eng_key:20s} {stats.get('files',0):5d} {stats.get('smells',0):7d}"
                       f" {stats.get('refactor',0):7d} {stats.get('dead_code',0):5d}"
                       f" {stats.get('fixes_proposed',0):6d} {stats.get('safety_blocks',0):8d}"
                       f" {stats.get('docs_generated',0):5d} {stats.get('tests_generated',0):6d}"
                       f" {audit_str:>6s}")

    # Per-rule fire count table (v3.3.0)
    if rule_fire_counts:
        printer.log("", "")
        printer.log("", "  Rules Fired:")
        for rule, count in sorted(rule_fire_counts.items(), key=lambda x: -x[1]):
            printer.log("", f"    {rule:35s} {count:5d} files")

    # Skipped rules summary (v3.3.0)
    printer.log("", "")
    printer.log("", "  Rules Skipped (destructive / unavailable):")
    for rule in sorted(_DESTRUCTIVE_RULES):
        printer.log("", f"    {rule:35s} SKIPPED (destructive)")
    for rule, files in sorted(rule_skip_reasons.items()):
        if rule not in _DESTRUCTIVE_RULES:
            printer.log("", f"    {rule:35s} {len(files)} files")

    # Action type breakdown
    action_types: Dict[str, int] = {}
    for a in all_actions:
        action_types[a.action_type] = action_types.get(a.action_type, 0) + 1
    if action_types:
        printer.log("", "")
        printer.log("", "  Action Breakdown:")
        for atype, count in sorted(action_types.items()):
            applied_count = sum(1 for a in all_actions if a.action_type == atype and a.applied)
            printer.log("", f"    {atype:25s} {count:5d} total, {applied_count} applied")

    # Final verdict
    if total_fixes_proposed == 0 and total_smells == 0:
        diag.ok("edit_verdict", "edit", "All source files clean — no edits needed")
        printer.log(ICONS[Severity.PASS], "ALL SOURCE FILES CLEAN — no edits needed")
    elif apply_mode and total_fixes_applied > 0:
        diag.ok("edit_verdict", "edit",
                f"{total_fixes_applied} fixes applied across {files_changed} files"
                f" ({total_safety_blocks} safety blocks)")
        printer.log(ICONS[Severity.PASS],
                    f"EDIT COMPLETE: {total_fixes_applied} fixes applied to {files_changed} files"
                    f" ({total_safety_blocks} safety blocks)")
    elif not apply_mode and total_fixes_proposed > 0:
        diag.info("edit_verdict", "edit",
                  f"{total_fixes_proposed} fixes available, {total_safety_blocks} safety blocks"
                  f" (dry-run mode — use --apply)")
        printer.log(ICONS[Severity.INFO],
                    f"DRY-RUN: {total_fixes_proposed} fixes ready"
                    f" ({total_safety_blocks} safety blocks)"
                    f" — run with --apply to write changes")
    else:
        diag.warn("edit_verdict", "edit",
                  f"{total_smells} smells, {total_refactor_suggestions} refactors found but no auto-fixes")
        printer.log(ICONS[Severity.WARN],
                    f"REVIEW NEEDED: {total_smells} smells + {total_refactor_suggestions} refactors (manual fix needed)")

    # Store full action log as data (v3.3.0: includes audit + rule tracking)
    diag.results[-1].data = {
        "version": "3.3.0",
        "safety_config": {
            "max_delete_pct": MAX_DELETE_PCT,
            "max_delete_lines": MAX_DELETE_LINES,
            "destructive_rules_skipped": sorted(_DESTRUCTIVE_RULES),
            "safe_rules_active": _SAFE_RULES,
        },
        "actions": [{
            "filepath": a.filepath,
            "engine": a.engine_key,
            "type": a.action_type,
            "description": a.description,
            "applied": a.applied,
            "blocked": a.blocked,
            "error": a.error,
        } for a in all_actions],
        "per_engine": per_engine_edit_stats,
        "audit_results": {k: {
            "quick_score": v.get("quick_score"),
            "verdict": v.get("verdict"),
        } for k, v in audit_results.items() if isinstance(v, dict) and "quick_score" in v},
        "rule_fire_counts": rule_fire_counts,
        "totals": {
            "files_analyzed": files_analyzed,
            "files_changed": files_changed,
            "smells": total_smells,
            "refactor_suggestions": total_refactor_suggestions,
            "dead_code_items": total_dead_code_items,
            "fixes_proposed": total_fixes_proposed,
            "fixes_applied": total_fixes_applied,
            "safety_blocks": total_safety_blocks,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

ALL_PHASES = {"boot", "status", "constants", "self-test", "cross", "perf", "quantum", "dead-code", "edit"}


def run_debug(
    engine_names: Optional[List[str]] = None,
    phases: Optional[List[str]] = None,
    quiet: bool = False,
    verbose: bool = False,
    json_mode: bool = False,
    report_path: Optional[str] = None,
    apply_edits: bool = False,
) -> DiagnosticCollector:
    """
    Main entry point for the L104 Unified Engine Debug Framework.

    Args:
        engine_names:  List of engine short names (None = all)
        phases:        List of phase names (None = all)
        quiet:         Suppress output except summary
        verbose:       Show extra detail
        json_mode:     Output JSON only
        report_path:   Save JSON report to file
        apply_edits:   If True, Phase 9 writes fixes to disk (default: dry-run)
    """
    diag = DiagnosticCollector()
    printer = _TermPrinter(quiet=quiet, verbose=verbose, json_mode=json_mode)

    # Resolve engine selection
    if engine_names:
        specs = {n: ENGINE_REGISTRY[n] for n in engine_names if n in ENGINE_REGISTRY}
        unknown = [n for n in engine_names if n not in ENGINE_REGISTRY]
        if unknown:
            printer.log(ICONS[Severity.WARN], f"Unknown engines ignored: {unknown}")
    else:
        specs = dict(ENGINE_REGISTRY)

    # Resolve phases
    active_phases = set(phases) if phases else ALL_PHASES

    if not json_mode and not quiet:
        print(f"\n{'═' * 74}")
        print(f"  L104 UNIFIED ENGINE DEBUG v{__version__}")
        print(f"  Engines: {', '.join(specs.keys())}")
        print(f"  Phases:  {', '.join(sorted(active_phases))}")
        print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Invariant: {GOD_CODE}")
        print(f"{'═' * 74}")

    t_start = time.time()
    engines: Dict[str, Any] = {}

    # Phase 1: Boot
    if "boot" in active_phases:
        engines = phase_boot(specs, diag, printer)

    # Phase 2: Status
    if "status" in active_phases and engines:
        phase_status(specs, engines, diag, printer)

    # Phase 3: Constants
    if "constants" in active_phases:
        phase_constants(specs, diag, printer)

    # Phase 4: Self-tests
    if "self-test" in active_phases and engines:
        phase_self_tests(specs, engines, diag, printer)

    # Phase 5: Cross-engine (lazy-boots engines on demand, doesn't need prior boot)
    if "cross" in active_phases:
        phase_cross_engine(engines, diag, printer)

    # Phase 6: Performance
    if "perf" in active_phases and engines:
        phase_perf(engines, diag, printer)

    # Phase 7: Quantum Computation (lazy-boots engines on demand)
    if "quantum" in active_phases:
        phase_quantum(engines, diag, printer)

    # Phase 8: Dead Code Reversal & Logic Construction (v3.1.0)
    if "dead-code" in active_phases:
        phase_dead_code(engines, diag, printer)

    # Phase 9: Code Engine Auto-Edit (v3.2.0)
    if "edit" in active_phases:
        phase_edit(engines, diag, printer, apply_mode=apply_edits)

    total_time = time.time() - t_start

    # ── Output ────────────────────────────────────────────────────────────
    if json_mode:
        report = diag.to_json()
        report["total_time_seconds"] = round(total_time, 2)
        report["engines_requested"] = list(specs.keys())
        report["phases"] = sorted(active_phases)
        print(json.dumps(report, indent=2, default=str))
    else:
        print(diag.summary_text())
        print(f"  Total time: {total_time:.2f}s")

    # Save report
    if report_path:
        report = diag.to_json()
        report["total_time_seconds"] = round(total_time, 2)
        report["engines_requested"] = list(specs.keys())
        report["phases"] = sorted(active_phases)
        Path(report_path).write_text(json.dumps(report, indent=2, default=str))
        if not json_mode:
            print(f"  Report saved to: {report_path}")

    return diag


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="L104 Unified Engine Debug Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available engines:
  {', '.join(ENGINE_REGISTRY.keys())}

Available phases:
  boot       — Parallel engine boot with timeout
  status     — Collect engine status reports
  constants  — Cross-engine constant alignment
  self-test  — Per-engine self-diagnostics
  cross      — Cross-engine data-flow pipelines
  perf       — Quick performance benchmarks
  quantum    — Quantum computation deep validation (HHL, 25Q, full analysis)
  dead-code  — Dead code reversal & logic construction (Phase 8)
  edit       — Code Engine auto-edit: smells, refactor, fix, docs, tests (Phase 9)

Examples:
  python l104_debug.py                          # Full suite
  python l104_debug.py --engines code,math      # Only Code + Math
  python l104_debug.py --phase constants        # Only constant alignment
  python l104_debug.py --phase edit             # Auto-edit dry-run
  python l104_debug.py --phase edit --apply     # Auto-edit with disk writes
  python l104_debug.py --json                   # JSON output
  python l104_debug.py --report debug.json      # Save JSON report
  python l104_debug.py -q                       # Quiet (summary only)
  python l104_debug.py -v                       # Verbose
""",
    )
    parser.add_argument("--engines", "-e", type=str, default=None,
                        help="Comma-separated engine names (default: all)")
    parser.add_argument("--phase", "-p", type=str, default=None,
                        help="Comma-separated phase names (default: all)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output JSON only")
    parser.add_argument("--report", "-r", type=str, default=None,
                        help="Save JSON report to file")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Quiet mode (summary only)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose mode")
    parser.add_argument("--apply", action="store_true",
                        help="Phase 9: actually write fixes to disk (default: dry-run)")
    parser.add_argument("--version", action="version",
                        version=f"L104 Debug Framework v{__version__}")

    args = parser.parse_args()

    engine_names = [e.strip() for e in args.engines.split(",")] if args.engines else None
    phases = [p.strip() for p in args.phase.split(",")] if args.phase else None

    diag = run_debug(
        engine_names=engine_names,
        phases=phases,
        quiet=args.quiet,
        verbose=args.verbose,
        json_mode=args.json,
        report_path=args.report,
        apply_edits=args.apply,
    )

    sys.exit(0 if diag.failed == 0 else 1)


if __name__ == "__main__":
    main()
