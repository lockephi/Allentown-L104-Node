#!/usr/bin/env python3
"""
L104 God Code Simulator — GodCodeSimulator Orchestrator v4.0
═══════════════════════════════════════════════════════════════════════════

Master orchestrator that wires together all decomposed subsystems:
  constants         — Sacred constants (GOD_CODE, PHI, phases)
  result            — SimulationResult dataclass (+ VQPU metrics)
  catalog           — SimulationCatalog registry
  quantum_primitives — Gates, statevector ops, VQPU-derived primitives
  simulations/      — 55 simulations in 8 categories (incl. vqpu_findings)
  sweep             — ParametricSweepEngine v3.0 (12 sweep types)
  optimizer         — AdaptiveOptimizer
  feedback          — FeedbackLoopEngine v3.0 (15D scoring)

v4.0 UPGRADES (from VQPU findings):
  - 55 simulations (was 23): +10 VQPU-derived, +22 from prior expansions
  - 8 categories (was 4): +vqpu_findings, +research, +circuits, +transpiler
  - 12 sweep types (was 8): +trotter, +qfi_scaling, +tomography, +zne
  - 15D feedback scoring (was 8D): +qfi, +topo, +trotter, +loschmidt
  - VQPU metric fields on SimulationResult

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from .catalog import SimulationCatalog
from .constants import GOD_CODE, PHI  # noqa: F401 — re-exported for compat
from .feedback import FeedbackLoopEngine
from .optimizer import AdaptiveOptimizer
from .sacred_transpiler import SacredTranspilerEngine
from .result import SimulationResult
from .simulations import ALL_SIMULATIONS
from .sweep import ParametricSweepEngine


class GodCodeSimulator:
    """
    Advanced God Code Full Simulator v4.0 — Unified orchestrator.

    v4.0: Adapted VQPU v8.0 findings into the simulation package:
      - 55 simulations across 8 categories (incl. vqpu_findings)
      - VQPU-derived quantum primitives (QFI, Loschmidt, tomography, etc.)
      - 12 parametric sweep types (incl. trotter, qfi_scaling, tomography, zne)
      - 15-dimension feedback scoring (QFI, purity, topology, Trotter)
      - SimulationResult with VQPU metric fields + to_vqpu_metrics()

    Usage:
        sim = GodCodeSimulator()
        result = sim.run("quantum_fisher_sensing")  # VQPU-derived
        report = sim.run_all()                       # All 55 sims
        sweep = sim.parametric_sweep("qfi_scaling")  # VQPU sweep
    """

    VERSION = "4.0.0"

    def __init__(self):
        self.catalog = SimulationCatalog()
        self.sweep = ParametricSweepEngine()
        self.optimizer = AdaptiveOptimizer()
        self.feedback = FeedbackLoopEngine()
        self.transpiler = SacredTranspilerEngine()
        self._run_count = 0
        self._total_elapsed_ms = 0.0
        self._last_results: List[SimulationResult] = []

        # ══════ Cross-engine integration (lazy via connect_engines) ══════
        self._science_engine = None
        self._quantum_gate_engine = None
        self._vqpu_bridge = None

        # Register all built-in simulations from the simulations subpackage
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register all 55 built-in simulations from the decomposed subpackage."""
        for name, fn, category, desc, nq in ALL_SIMULATIONS:
            self.catalog.register(name, fn, category, desc, nq)
    def status(self) -> dict:
        """Return engine status for l104_debug integration."""
        return {
            "version": self.VERSION,
            "simulations": self.catalog.count,
            "categories": sorted(self.catalog.categories),
            "runs": self._run_count,
            "total_elapsed_ms": round(self._total_elapsed_ms, 1),
            "optimizer_strategies": 8,
            "sweep_types": 12,
            "feedback_version": "3.0",
            "feedback_dimensions": 15,
            "primitives_version": "4.0",
            "vqpu_adapted": True,
        }
    # ── Run Interface ───────────────────────────────────────────────────────

    def run(self, name: str, **kwargs) -> SimulationResult:
        """Run a single named simulation."""
        entry = self.catalog.get(name)
        if entry is None:
            return SimulationResult(
                name=name, category="unknown", passed=False,
                detail=f"Unknown simulation: {name}. Available: {', '.join(self.catalog.list_all()[:5])}...",
            )
        try:
            result = entry["fn"](**kwargs)
            self._run_count += 1
            self._total_elapsed_ms += result.elapsed_ms
            self._last_results.append(result)
            return result
        except Exception as e:
            return SimulationResult(
                name=name, category=entry["category"], passed=False,
                detail=f"Error: {e}",
            )

    def run_category(self, category: str, **kwargs) -> List[SimulationResult]:
        """Run all simulations in a category (parallel, v5.1)."""
        names = self.catalog.list_by_category(category)
        return self._parallel_run_names(names, **kwargs)

    def _parallel_run_names(self, names: List[str], **kwargs) -> List[SimulationResult]:
        """Run named simulations in parallel via ThreadPoolExecutor.

        v5.1 PERFORMANCE: NumPy releases the GIL during BLAS matmul, so
        threading provides genuine speedup for compute-bound simulations.
        Workers default to min(CPU_COUNT, 8) for optimal throughput.
        """
        max_workers = min(os.cpu_count() or 4, 8, len(names))
        if max_workers <= 1 or len(names) <= 1:
            return [self.run(name, **kwargs) for name in names]
        results = [None] * len(names)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.run, name, **kwargs): i
                       for i, name in enumerate(names)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results

    def run_all(self, **kwargs) -> Dict[str, Any]:
        """Run all simulations (parallel) and return full report.

        v5.1 PERFORMANCE: Parallelized via ThreadPoolExecutor — 3-6× faster
        on multicore systems.  All 55 simulations are independent.
        """
        names = self.catalog.list_all()
        results = self._parallel_run_names(names, **kwargs)

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        total_ms = sum(r.elapsed_ms for r in results)

        return {
            "version": self.VERSION,
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "total_elapsed_ms": round(total_ms, 1),
            "categories": {
                cat: {
                    "total": len(self.catalog.list_by_category(cat)),
                    "passed": sum(1 for r in results if r.category == cat and r.passed),
                }
                for cat in self.catalog.categories
            },
            "results": results,
            "summaries": [r.summary() for r in results],
        }

    # ── Parametric Sweep Interface ──────────────────────────────────────────

    def parametric_sweep(self, sweep_type: str, **kwargs) -> Any:
        """
        Run parametric sweep.

        Types:
          - "dial_a" / "dial_b" / "dial_c" / "dial_d": God Code dial sweeps
          - "noise": Noise level sweep
          - "depth": Circuit depth sweep
          - "qubit_scaling": Scale a simulation across qubit counts
          - "strategy": Compare protection strategies across noise regimes
          - "phase": Sweep sacred phase angles
          - "cross_simulation": Compare metrics across all simulations
          - "convergence": Optimizer convergence characteristics
          v3.0 (VQPU-derived):
          - "trotter_convergence": Trotter error vs step count
          - "qfi_scaling": QFI vs qubit count (Heisenberg limit)
          - "tomography": Reconstruction fidelity across qubit counts
          - "zne": Zero-noise extrapolation cost-benefit
        """
        if sweep_type.startswith("dial_"):
            dial = sweep_type.split("_")[1]
            return self.sweep.dial_sweep(dial=dial, **kwargs)
        elif sweep_type == "noise":
            return self.sweep.noise_sweep(simulator=self, **kwargs)
        elif sweep_type == "depth":
            return self.sweep.depth_sweep(**kwargs)
        elif sweep_type == "qubit_scaling":
            sim_name = kwargs.pop("sim_name", "entanglement_entropy")
            entry = self.catalog.get(sim_name)
            if entry:
                return self.sweep.qubit_scaling(entry["fn"], **kwargs)
            return [{"error": f"Unknown simulation: {sim_name}"}]
        elif sweep_type == "strategy":
            return self.sweep.strategy_sweep(simulator=self, **kwargs)
        elif sweep_type == "phase":
            return self.sweep.phase_sweep(**kwargs)
        elif sweep_type == "cross_simulation":
            return self.sweep.cross_simulation_sweep(simulator=self, **kwargs)
        elif sweep_type == "convergence":
            return self.sweep.convergence_sweep(simulator=self, **kwargs)
        # v3.0 VQPU-derived sweep types
        elif sweep_type == "trotter_convergence":
            return self.sweep.trotter_convergence_sweep(**kwargs)
        elif sweep_type == "qfi_scaling":
            return self.sweep.qfi_scaling_sweep(**kwargs)
        elif sweep_type == "tomography":
            return self.sweep.tomography_accuracy_sweep(**kwargs)
        elif sweep_type == "zne":
            return self.sweep.zne_overhead_sweep(**kwargs)
        else:
            return [{"error": f"Unknown sweep type: {sweep_type}"}]

    # ── Adaptive Optimization Interface ─────────────────────────────────────

    def adaptive_optimize(self, target_fidelity: float = 0.99,
                          nq: int = 4, depth: int = 4) -> Dict[str, Any]:
        """Run adaptive circuit optimization."""
        self.optimizer.target_fidelity = target_fidelity
        return self.optimizer.optimize_sacred_circuit(nq=nq, depth=depth)

    def optimize_noise_resilience(self, nq: int = 2, noise_level: float = 0.1) -> Dict[str, Any]:
        """Find best noise protection strategy."""
        return self.optimizer.optimize_noise_resilience(nq=nq, noise_level=noise_level)

    # ── Feedback Loop Interface ─────────────────────────────────────────────

    def run_feedback_loop(self, sim_names: List[str] = None, iterations: int = 5) -> Dict[str, Any]:
        """
        Run multi-engine feedback loop.

        Runs specified simulations, feeds results through coherence → entropy → scoring.
        Auto-connects Science/Math engines on first call if not already connected.

        v4.2: Auto-connect engines lazily so feedback always runs with live data.
        v4.1: Default sim set expanded to include VQPU sims for richer feedback.
        """
        # v4.2: Auto-connect engines if not yet wired
        self._auto_connect_engines()

        if sim_names is None:
            sim_names = ["entanglement_entropy", "sacred_cascade", "decoherence_model",
                         "phase_interference", "iron_manifold",
                         "quantum_fisher_sensing", "loschmidt_chaos",
                         "kitaev_preskill_topo"]

        results = [self.run(name) for name in sim_names[:iterations]]
        return self.feedback.run_feedback_cycle(results, iterations=iterations)

    def _auto_connect_engines(self) -> None:
        """Lazily import and connect Science + Math engines if not already wired."""
        if (self.feedback._coherence_subsystem is not None
                and self.feedback._entropy_subsystem is not None
                and self.feedback._math_engine is not None):
            return  # already connected
        try:
            if self.feedback._coherence_subsystem is None or self.feedback._entropy_subsystem is None:
                from l104_science_engine import ScienceEngine
                se = ScienceEngine()
                if self.feedback._coherence_subsystem is None:
                    self.feedback.connect_coherence(se.coherence)
                if self.feedback._entropy_subsystem is None:
                    self.feedback.connect_entropy(se.entropy)
                if self._science_engine is None:
                    self._science_engine = se
        except Exception:
            pass  # Science engine not available — local fallback used
        try:
            if self.feedback._math_engine is None:
                from l104_math_engine import MathEngine
                self.feedback.connect_math(MathEngine())
        except Exception:
            pass  # Math engine not available — local fallback used

    def connect_engines(self, coherence=None, entropy=None, math_engine=None,
                         science_engine=None, quantum_gate_engine=None,
                         vqpu_bridge=None) -> None:
        """Connect live engine subsystems for real feedback loops.

        Args:
            coherence: CoherenceSubsystem from ScienceEngine
            entropy: EntropySubsystem from ScienceEngine
            math_engine: MathEngine instance
            science_engine: Full ScienceEngine instance (v3.0)
            quantum_gate_engine: QuantumGateEngine for circuit compilation (v3.0)
            vqpu_bridge: VQPUBridge for Metal GPU execution (v3.0)
        """
        if coherence:
            self.feedback.connect_coherence(coherence)
        if entropy:
            self.feedback.connect_entropy(entropy)
        if math_engine:
            self.feedback.connect_math(math_engine)
        if science_engine:
            self._science_engine = science_engine
        if quantum_gate_engine:
            self._quantum_gate_engine = quantum_gate_engine
        if vqpu_bridge:
            self._vqpu_bridge = vqpu_bridge

    def run_multi_pass_feedback(self, sim_names: List[str] = None,
                                 passes: int = 3, iterations_per_pass: int = 5) -> Dict[str, Any]:
        """
        Run multi-pass feedback loop with convergence detection.

        Each pass refines the scoring with diminishing learning rate.
        Stops early on convergence or oscillation.
        """
        self._auto_connect_engines()
        if sim_names is None:
            sim_names = ["entanglement_entropy", "sacred_cascade", "decoherence_model",
                         "phase_interference", "iron_manifold",
                         "quantum_fisher_sensing", "loschmidt_chaos",
                         "kitaev_preskill_topo"]
        results = [self.run(name) for name in sim_names[:iterations_per_pass]]
        return self.feedback.run_multi_pass(results, passes=passes,
                                            iterations_per_pass=iterations_per_pass)

    def score_dimensions(self, sim_names: List[str] = None) -> Dict[str, float]:
        """
        Score simulation results across 15 quality dimensions.

        v4.1: Uses a curated representative set (20 sims across all 8 categories)
        to ensure all 15 dimensions receive real data — especially VQPU-derived
        dimensions (QFI, topo_entropy, Loschmidt, mutual information).

        Returns per-dimension scores for: fidelity, entropy, coherence,
        conservation, alignment, concurrence, information, stability,
        purity, gate_fidelity, decoherence_resilience,
        qfi, topo_entropy, trotter_quality, loschmidt.
        """
        if sim_names is None:
            sim_names = self._scoring_representative_set()
        results = [self.run(name) for name in sim_names]
        return self.feedback.score_by_dimension(results)

    def _scoring_representative_set(self) -> List[str]:
        """
        Curated 20-sim set covering all 8 categories and all 15 scoring dimensions.

        Ensures VQPU-derived dimensions (QFI, topo_entropy, Loschmidt, MI)
        receive data from simulations that actually produce those metrics.
        """
        curated_per_category = {
            "core": ["conservation_proof", "dial_sweep_a"],
            "quantum": ["bell_chsh_violation", "mutual_information", "entanglement_entropy"],
            "advanced": ["grover_search", "teleportation", "qec_bit_flip"],
            "discovery": ["iron_manifold", "decoherence_model"],
            "transpiler": ["unitary_verification"],
            "circuits": ["vqe_sacred", "noise_resilience"],
            "research": ["quantum_chaos", "topological_braiding"],
            "vqpu_findings": [
                "quantum_fisher_sensing", "loschmidt_chaos",
                "kitaev_preskill_topo", "swap_test_fidelity",
                "trotter_error_analysis",
            ],
        }
        all_registered = set(self.catalog.list_all())
        curated = []
        for cat_sims in curated_per_category.values():
            for name in cat_sims:
                if name in all_registered:
                    curated.append(name)
        return curated if curated else self.catalog.list_all()[:10]

    def simulate_for_vqpu(self, sim_name: str = "quantum_fisher_sensing") -> Dict[str, Any]:
        """Run simulation and return VQPU metrics payload."""
        result = self.run(sim_name)
        return result.to_vqpu_metrics()

    # ── Engine Integration Methods ──────────────────────────────────────────

    def simulate_for_coherence(self, sim_name: str = "entanglement_entropy") -> Dict[str, Any]:
        """Run simulation and return payload for coherence.ingest_simulation_result()."""
        result = self.run(sim_name)
        return result.to_coherence_payload()

    def simulate_for_entropy(self, sim_name: str = "decoherence_model") -> float:
        """Run simulation and return entropy value for entropy.calculate_demon_efficiency()."""
        result = self.run(sim_name)
        return result.to_entropy_input()

    def simulate_for_asi(self, sim_name: str = "sacred_cascade") -> Dict[str, Any]:
        """Run simulation and return scoring payload for ASI pipeline."""
        result = self.run(sim_name)
        return result.to_asi_scoring()

    def batch_simulate_for_coherence(self, n: int = 5) -> List[Dict[str, Any]]:
        """Run N simulations and return payloads for coherence.run_feedback_loop()."""
        names = self.catalog.list_all()[:n]
        return [self.run(name).to_coherence_payload() for name in names]

    # ── Qiskit Transpiler Interface ─────────────────────────────────────────

    def build_circuit(self, circuit_type: str = "sacred", **kwargs) -> Any:
        """
        Build a GOD_CODE quantum circuit.

        Types: "1q", "1q_decomposed", "sacred", "dial"
        """
        if circuit_type == "1q":
            return self.transpiler.build_1q()
        elif circuit_type == "1q_decomposed":
            return self.transpiler.build_1q_decomposed()
        elif circuit_type == "sacred":
            return self.transpiler.build_sacred(**kwargs)
        elif circuit_type == "dial":
            return self.transpiler.build_dial(**kwargs)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

    def transpile_circuit(self, circuit, basis_gates: List[str] = None,
                          label: str = "") -> Dict[str, Any]:
        """
        Transpile a circuit to a hardware basis gate set.

        If basis_gates is None, transpiles to all 5 standard sets.
        """
        if basis_gates is None:
            return self.transpiler.transpile_all(circuit, label)
        return self.transpiler.transpile(circuit, basis_gates, label=label)[1]

    def verify_circuit(self, circuit, label: str = "") -> Dict[str, Any]:
        """Full unitary verification of a circuit."""
        return self.transpiler.verify_unitary(circuit, label)

    def verify_conservation(self, n_points: int = 7) -> Dict[str, Any]:
        """Verify GOD_CODE conservation law across octave X values."""
        return self.transpiler.verify_conservation(n_points)

    def full_transpilation_report(self) -> Dict[str, Any]:
        """Run full transpilation pipeline: build → transpile → verify → analyze."""
        return self.transpiler.full_report()

    # ── Status & Metrics ────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return simulator status and metrics."""
        return {
            "version": self.VERSION,
            "simulations_registered": self.catalog.count,
            "categories": self.catalog.categories,
            "total_runs": self._run_count,
            "total_elapsed_ms": round(self._total_elapsed_ms, 1),
            "feedback_engines": {
                "coherence": self.feedback._coherence_subsystem is not None,
                "entropy": self.feedback._entropy_subsystem is not None,
                "math": self.feedback._math_engine is not None,
                "science_engine": self._science_engine is not None,
                "quantum_gate_engine": self._quantum_gate_engine is not None,
                "vqpu_bridge": self._vqpu_bridge is not None,
            },
            "transpiler": self.transpiler.status(),
            "last_results_count": len(self._last_results),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

god_code_simulator = GodCodeSimulator()
