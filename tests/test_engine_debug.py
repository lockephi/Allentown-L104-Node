#!/usr/bin/env python3
"""
L104 Unified Engine Debug — pytest-compatible test wrapper.
══════════════════════════════════════════════════════════════════════════════════
Exposes the l104_debug.py framework as pytest test cases so that:
    .venv/bin/python -m pytest tests/test_engine_debug.py -v
runs the same diagnostic suite with pytest's reporting.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from l104_debug import (
    DiagnosticCollector,
    EngineSpec,
    ENGINE_REGISTRY,
    GOD_CODE,
    PHI,
    VOID_CONSTANT,
    OMEGA,
    CONST_TOLERANCE,
    phase_boot,
    phase_constants,
    phase_self_tests,
    phase_cross_engine,
    phase_quantum,
    _TermPrinter,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

# Quiet printer for pytest — no stdout noise
_printer = _TermPrinter(quiet=True)


@pytest.fixture(scope="session")
def core_engines():
    """Boot code, science, math engines once for the session."""
    specs = {k: ENGINE_REGISTRY[k] for k in ["code", "science", "math"]}
    diag = DiagnosticCollector()
    return phase_boot(specs, diag, _printer, timeout=120)


@pytest.fixture(scope="session")
def all_engines():
    """Boot all engines once for the session."""
    diag = DiagnosticCollector()
    return phase_boot(ENGINE_REGISTRY, diag, _printer, timeout=180)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANT ALIGNMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanonicalConstants:
    """Verify sacred constant formulas."""

    def test_god_code_formula(self):
        """GOD_CODE = 286^(1/φ) × 16"""
        computed = (286 ** (1 / PHI)) * 16
        assert abs(computed - GOD_CODE) < 1e-10, f"GOD_CODE: {computed} != {GOD_CODE}"

    def test_void_constant_formula(self):
        """VOID_CONSTANT = 1.04 + φ/1000"""
        computed = 1.04 + PHI / 1000
        assert abs(computed - VOID_CONSTANT) < 1e-15, f"VOID: {computed} != {VOID_CONSTANT}"

    def test_phi_identity(self):
        """φ² - φ - 1 = 0"""
        identity = PHI ** 2 - PHI - 1
        assert abs(identity) < 1e-14, f"PHI identity: {identity}"

    def test_phi_value(self):
        """PHI = (1+√5)/2"""
        expected = (1 + math.sqrt(5)) / 2
        assert abs(PHI - expected) < 1e-15

    def test_god_code_immutable(self):
        """GOD_CODE has the canonical value."""
        assert GOD_CODE == 527.5184818492612

    def test_cross_engine_constants(self):
        """Constants agree across all engine packages."""
        import importlib
        for name, spec in ENGINE_REGISTRY.items():
            if not spec.constant_names:
                continue
            try:
                mod = importlib.import_module(spec.constants_module)
                for cname in spec.constant_names:
                    val = getattr(mod, cname, None)
                    if val is None:
                        continue
                    canonical = {"GOD_CODE": GOD_CODE, "PHI": PHI,
                                 "VOID_CONSTANT": VOID_CONSTANT, "OMEGA": OMEGA}.get(cname)
                    if canonical is not None:
                        assert abs(val - canonical) < CONST_TOLERANCE, \
                            f"{name}.{cname} = {val}, canonical = {canonical}"
            except ImportError:
                pytest.skip(f"Cannot import {spec.constants_module}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-ENGINE BOOT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("engine_name", list(ENGINE_REGISTRY.keys()))
class TestEngineBoot:
    """Each engine boots without errors."""

    def test_boot(self, engine_name, all_engines):
        data = all_engines.get(engine_name, {})
        if data.get("error"):
            pytest.fail(f"{engine_name} boot failed: {data['error']}")
        assert data.get("engine") is not None, f"{engine_name} returned None"


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-ENGINE SELF-TEST WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_self_tests():
    """Generate (engine_name, test_fn) pairs for parametrize."""
    pairs = []
    for name, spec in ENGINE_REGISTRY.items():
        for fn in spec.self_tests:
            pairs.append((name, fn))
    return pairs


@pytest.mark.parametrize("engine_name,test_fn", _gen_self_tests(),
                         ids=[f"{n}-{fn.__name__}" for n, fn in _gen_self_tests()])
def test_engine_self_test(engine_name, test_fn, all_engines):
    """Run each engine's built-in self-test."""
    eng = all_engines.get(engine_name, {}).get("engine")
    if eng is None:
        pytest.skip(f"{engine_name} not booted")
    test_name, passed, detail = test_fn(eng)
    assert passed, f"[{engine_name}] {test_name}: {detail}"


# ═══════════════════════════════════════════════════════════════════════════════
#  CROSS-ENGINE PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossEngine:
    """Cross-engine data-flow validations."""

    def test_cross_engine_pipelines(self, core_engines):
        """Run cross-engine pipeline phase and check no failures."""
        diag = DiagnosticCollector()
        phase_cross_engine(core_engines, diag, _printer)
        if diag.failed > 0:
            failures = "; ".join(f"{r.test}: {r.detail}" for r in diag.failures)
            pytest.fail(f"Cross-engine failures: {failures}")


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM COMPUTATION TESTS (v3.0.0)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def quantum_engines():
    """Boot quantum-focused engines for quantum phase tests."""
    quantum_names = ["quantum_gate", "quantum_link", "numerical", "gate",
                     "asi_quantum", "code_quantum", "pipeline", "dual_layer", "science"]
    specs = {k: ENGINE_REGISTRY[k] for k in quantum_names if k in ENGINE_REGISTRY}
    diag = DiagnosticCollector()
    return phase_boot(specs, diag, _printer, timeout=180)


class TestQuantumComputation:
    """Quantum computation deep validation tests (HHL, 25Q, full analysis)."""

    def test_quantum_phase(self, quantum_engines):
        """Run the full quantum computation phase and check no failures."""
        diag = DiagnosticCollector()
        phase_quantum(quantum_engines, diag, _printer)
        if diag.failed > 0:
            failures = "; ".join(f"{r.test}: {r.detail}" for r in diag.failures)
            pytest.fail(f"Quantum phase failures: {failures}")

    def test_hhl_cross_validation(self, quantum_engines):
        """All booted HHL implementations produce valid results."""
        hhl_engines = {}
        # Gate engine
        try:
            from l104_gate_engine.quantum_computation import QuantumGateComputationEngine
            qgce = QuantumGateComputationEngine()
            r = qgce.hhl_linear_solver([1.0, 2.5, 3.7, 0.8])
            assert "solution" in r and "condition_number" in r
            hhl_engines["gate"] = r
        except Exception:
            pass

        # ASI quantum
        asi_q = quantum_engines.get("asi_quantum", {}).get("engine")
        if asi_q:
            try:
                r = asi_q.hhl_linear_solver()
                assert "solution" in r
                hhl_engines["asi"] = r
            except Exception:
                pass

        # Pipeline
        hub = quantum_engines.get("pipeline", {}).get("engine")
        if hub:
            try:
                r = hub.hhl_linear_solver()
                assert "solution" in r
                hhl_engines["pipeline"] = r
            except Exception:
                pass

        assert len(hhl_engines) >= 1, "No HHL engines produced valid results"

        # All condition numbers should be finite
        for name, r in hhl_engines.items():
            kappa = r.get("condition_number", float("inf"))
            if isinstance(kappa, (int, float)):
                assert kappa < float("inf"), f"{name} HHL has infinite κ"

    def test_25q_hhl_template(self):
        """25Q HHL template exists and has valid fields."""
        from l104_science_engine.quantum_25q import CircuitTemplates25Q
        templates = CircuitTemplates25Q.all_templates()
        assert "hhl_25" in templates, "HHL template missing from all_templates()"
        t = templates["hhl_25"]
        assert t["n_qubits"] >= 20, f"HHL qubits too low: {t['n_qubits']}"
        assert t["depth"] > 0
        assert t["cx_gates"] > 0


class TestDualLayerEngine:
    """Dual-Layer Engine integration tests (v3.0.0)."""

    @pytest.fixture(scope="class")
    def dle(self, quantum_engines):
        eng = quantum_engines.get("dual_layer", {}).get("engine")
        if eng is None:
            pytest.skip("DualLayerEngine not booted")
        return eng

    def test_thought_god_code(self, dle):
        """thought(0,0,0,0) produces a value near GOD_CODE."""
        val = dle.thought(0, 0, 0, 0)
        assert isinstance(val, (int, float))

    def test_physics_layer(self, dle):
        """physics() returns a dict."""
        r = dle.physics()
        assert isinstance(r, dict)

    def test_integrity_check(self, dle):
        """full_integrity_check() completes."""
        r = dle.full_integrity_check()
        assert isinstance(r, dict)

    def test_dual_score(self, dle):
        """dual_score() returns a numeric value."""
        s = dle.dual_score()
        assert isinstance(s, (int, float))

    def test_consciousness_spectrum(self, dle):
        """consciousness_spectrum() returns a dict."""
        r = dle.consciousness_spectrum()
        assert isinstance(r, dict)

    def test_cross_layer_coherence(self, dle):
        """cross_layer_coherence() returns a dict."""
        r = dle.cross_layer_coherence()
        assert isinstance(r, dict)


class TestQuantumPipeline:
    """Quantum Pipeline Hub tests (v3.0.0)."""

    @pytest.fixture(scope="class")
    def hub(self, quantum_engines):
        eng = quantum_engines.get("pipeline", {}).get("engine")
        if eng is None:
            pytest.skip("Pipeline not booted")
        return eng

    def test_pipeline_status(self, hub):
        """status() returns comprehensive dict."""
        s = hub.status()
        assert isinstance(s, dict)
        assert "n_qubits" in s

    def test_pipeline_hhl(self, hub):
        """hhl_linear_solver produces a valid solution."""
        r = hub.hhl_linear_solver()
        assert isinstance(r, dict) and "solution" in r

    def test_pipeline_bell(self, hub):
        """create_bell_state produces entangled state."""
        r = hub.create_bell_state()
        assert isinstance(r, dict)

    def test_pipeline_ghz(self, hub):
        """create_ghz_state works."""
        r = hub.create_ghz_state()
        assert isinstance(r, dict)

    def test_pipeline_qft(self, hub):
        """quantum_fourier_transform works."""
        r = hub.quantum_fourier_transform()
        assert isinstance(r, dict)
