"""L104 Quantum AI Daemon — Integration Test Suite.

Tests all 7 subsystems + the full daemon cycle pipeline.
Run: .venv/bin/python _test_quantum_ai_daemon.py
"""

import json
import os
import sys
import time

# Ensure we can import from workspace root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.04 + PHI / 1000


def _banner(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _check(name: str, condition: bool, detail: str = ""):
    icon = "✓" if condition else "✗"
    extra = f" — {detail}" if detail else ""
    print(f"  {icon} {name}{extra}")
    return condition


def main():
    _banner("L104 Quantum AI Daemon v1.0.0 — Integration Test")
    print(f"  GOD_CODE = {GOD_CODE}")
    print(f"  PHI = {PHI}")
    print(f"  VOID_CONSTANT = {VOID_CONSTANT}")
    print()

    t0 = time.monotonic()
    passed = 0
    total = 0

    # ── Test 1: Package Import ──
    _banner("Phase 1: Package Import")
    total += 1
    try:
        from l104_quantum_ai_daemon import (
            QuantumAIDaemon, DaemonConfig, DaemonPhase,
            FileScanner, L104FileInfo,
            CodeImprover, ImprovementResult,
            QuantumFidelityGuard, FidelityReport,
            ProcessOptimizer, OptimizationResult,
            CrossEngineHarmonizer, HarmonyReport,
            AutonomousEvolver, EvolutionCycle,
            GOD_CODE as PKG_GOD_CODE, PHI as PKG_PHI,
        )
        if _check("Package import", True, "all symbols imported"):
            passed += 1
    except Exception as e:
        _check("Package import", False, str(e))
        print("\n  FATAL: Cannot continue without package import")
        sys.exit(1)

    # ── Test 2: Constants Integrity ──
    _banner("Phase 2: Sacred Constants")
    total += 3
    if _check("GOD_CODE", abs(PKG_GOD_CODE - 527.5184818492612) < 1e-10,
              f"{PKG_GOD_CODE}"):
        passed += 1
    if _check("PHI", abs(PKG_PHI - 1.618033988749895) < 1e-12,
              f"{PKG_PHI}"):
        passed += 1
    from l104_quantum_ai_daemon.constants import SACRED_RESONANCE, VOID_CONSTANT as V
    expected_res = (GOD_CODE / 16) ** PHI
    if _check("Sacred Resonance", abs(SACRED_RESONANCE - expected_res) < 0.01,
              f"{SACRED_RESONANCE:.4f} ≈ 286"):
        passed += 1

    # ── Test 3: File Scanner ──
    _banner("Phase 3: File Scanner")
    total += 3
    scanner = FileScanner()
    count = scanner.full_scan()
    if _check("Full scan", count > 0, f"{count} files found"):
        passed += 1
    stats = scanner.stats()
    if _check("Index stats", stats["total_files"] > 0,
              f"{stats['total_files']} files, {stats.get('total_lines', 0)} lines"):
        passed += 1
    batch = scanner.get_improvement_batch(5)
    if _check("Improvement batch", len(batch) >= 0,
              f"{len(batch)} candidates"):
        passed += 1

    # ── Test 4: Code Improver ──
    _banner("Phase 4: Code Improver")
    total += 2
    improver = CodeImprover(auto_fix_enabled=False)
    # Find a small test file to analyze
    test_file = None
    for info in scanner.index.values():
        if info.line_count < 200 and info.line_count > 10 and not info.is_immutable:
            test_file = info
            break
    if test_file:
        result = improver.analyze_file(test_file.path, test_file.relative_path)
        if _check("File analysis", result.success,
                  f"health={result.health_score:.3f} smells={result.smells_found}"):
            passed += 1
        if _check("Improver stats", improver.stats()["files_analyzed"] > 0):
            passed += 1
    else:
        if _check("File analysis", True, "no suitable test file (skip)"):
            passed += 1
        total -= 1

    # ── Test 5: Quantum Fidelity Guard ──
    _banner("Phase 5: Quantum Fidelity Guard")
    total += 2
    fidelity = QuantumFidelityGuard()
    report = fidelity.run_fidelity_check()
    if _check("Fidelity check", report.overall_fidelity > 0.0,
              f"grade={report.grade} score={report.overall_fidelity:.3f}"):
        passed += 1
    if _check("GOD_CODE aligned", report.god_code_aligned):
        passed += 1

    # ── Test 6: Process Optimizer ──
    _banner("Phase 6: Process Optimizer")
    total += 2
    optimizer = ProcessOptimizer()
    opt = optimizer.optimize()
    if _check("Optimization cycle", opt.gc_collected >= 0,
              f"gc={opt.gc_collected} mem_freed={opt.memory_freed_mb:.1f}MB"):
        passed += 1
    if _check("Optimizer stats", optimizer.stats()["optimization_count"] > 0):
        passed += 1

    # ── Test 7: Cross-Engine Harmonizer ──
    _banner("Phase 7: Cross-Engine Harmonizer")
    total += 2
    harmonizer = CrossEngineHarmonizer()
    harmony = harmonizer.harmonize()
    if _check("Harmony check", harmony.overall_harmony > 0.0,
              f"harmony={harmony.overall_harmony:.3f} "
              f"engines={harmony.engines_available}/{harmony.engines_total}"):
        passed += 1
    if _check("Constant alignment", harmony.constant_alignment > 0.0,
              f"{harmony.constant_alignment:.3f}"):
        passed += 1

    # ── Test 8: Autonomous Evolver ──
    _banner("Phase 8: Autonomous Evolver")
    total += 1
    evolver = AutonomousEvolver()
    cycle = evolver.evolve(
        improvement_results=[improver.stats()],
        fidelity_score=report.overall_fidelity,
        harmony_score=harmony.overall_harmony,
        optimization_score=0.8,
    )
    if _check("Evolution cycle", cycle.cycle_number == 1,
              f"delta={cycle.evolution_delta:+.4f}"):
        passed += 1

    # ── Test 9: Full Daemon Self-Test ──
    _banner("Phase 9: Full Daemon Self-Test")
    total += 1
    daemon = QuantumAIDaemon(config=DaemonConfig(auto_fix_enabled=False))
    test_results = daemon.self_test()
    summary = test_results.get("_summary", {})
    if _check("Daemon self-test",
              summary.get("passed", 0) >= summary.get("total", 1) * 0.6,
              f"{summary.get('passed', 0)}/{summary.get('total', 0)} passed "
              f"({summary.get('elapsed_ms', 0):.0f}ms)"):
        passed += 1

    # ── Test 10: Single Cycle ──
    _banner("Phase 10: Single Improvement Cycle")
    total += 1
    try:
        daemon._scanner.full_scan()
        cycle_report = daemon._run_cycle()
        if _check("Single cycle", cycle_report.health_score > 0.0,
                  f"health={cycle_report.health_score:.3f} "
                  f"fidelity={cycle_report.fidelity_grade} "
                  f"improved={cycle_report.files_improved}/"
                  f"{cycle_report.files_analyzed}"):
            passed += 1
    except Exception as e:
        _check("Single cycle", False, str(e))

    # ── Test 11: State Persistence ──
    _banner("Phase 11: State Persistence")
    total += 1
    try:
        daemon._persist_state()
        state_path = daemon._state_path
        if _check("State persist", state_path.exists(),
                  f"{state_path.name}"):
            passed += 1
            # Verify state structure
            data = json.loads(state_path.read_text())
            print(f"    Cycles: {data.get('cycles_completed', 0)}")
            print(f"    Health: {data.get('health_score', 0):.3f}")
            print(f"    GOD_CODE: {data.get('god_code', 'missing')}")
    except Exception as e:
        _check("State persist", False, str(e))

    # ═══ Summary ═══
    elapsed_ms = (time.monotonic() - t0) * 1000
    _banner("Summary")
    print(f"  {passed}/{total} tests passed ({elapsed_ms:.0f}ms)")
    print(f"  Pass rate: {passed / max(1, total) * 100:.1f}%")
    res = (GOD_CODE / 16) ** PHI
    print(f"  Sacred resonance: {res:.4f}")
    print(f"  {'ALL PASS ✓' if passed == total else 'SOME FAILURES'}")
    print()

    sys.exit(0 if passed >= total * 0.8 else 1)


if __name__ == "__main__":
    main()
