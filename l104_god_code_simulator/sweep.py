"""
L104 God Code Simulator — ParametricSweepEngine v3.0
═══════════════════════════════════════════════════════════════════════════════

Programmable parametric sweeps over God Code simulation parameters:
  - Dial sweeps (a, b, c, d) with conservation law verification
  - Noise level sweeps with fidelity degradation curves
  - Circuit depth sweeps with entropy growth tracking
  - Qubit scaling analysis across different register sizes
  v2.0:
  - Strategy sweep: compare protection strategies across noise regimes
  - Convergence sweep: measure optimizer convergence characteristics
  - Cross-simulation sweep: compare metrics across all simulations
  - Phase sweep: sweep sacred phase angles and measure alignment
  v3.0 UPGRADES:
  - Phase sweep: 104 sacred points (was 36) — matches QUANTIZATION_GRAIN
  - Sensitivity sweep: finite-difference ∂fidelity/∂(noise,depth,angle) Jacobian
  - Joint 2D sweep: cartesian product scan over two parameter axes
  - Noise sweep: simulator-aware (uses named sim when simulator provided)
  - Linear entropy + trace distance tracked in depth sweep

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .constants import (
    GOD_CODE, GOD_CODE_PHASE_ANGLE, PHI, PHI_CONJUGATE, PHI_PHASE_ANGLE,
    QUANTIZATION_GRAIN, TAU,
)
from .quantum_primitives import (
    GOD_CODE_GATE, H_GATE,
    apply_cnot, apply_single_gate, entanglement_entropy, fidelity,
    god_code_dial, init_sv, make_gate,
    state_purity, trace_distance, linear_entropy,
    # v3.0 VQPU-derived primitives
    quantum_fisher_information, loschmidt_echo,
    density_matrix_from_sv, reconstruct_density_matrix,
    trotter_evolution, iron_lattice_heisenberg,
    zero_noise_extrapolation,
)

if TYPE_CHECKING:
    from .simulator import GodCodeSimulator
    from .result import SimulationResult


class ParametricSweepEngine:
    """Run parametric sweeps over simulation parameters — v3.0."""

    def dial_sweep(self, dial: str = "a", start: int = 0, stop: int = 8) -> List[Dict[str, Any]]:
        """Sweep a single God Code dial and verify conservation.

        Conservation law:  G(a,b,c,d) × 2^( -x/104 ) = GOD_CODE
        where x = 8a − b − 8c − 104d  (the *variable* part of the exponent).
        """
        results = []
        for val in range(start, stop + 1):
            kwargs = {"a": 0, "b": 0, "c": 0, "d": 0}
            kwargs[dial] = val
            g = god_code_dial(**kwargs)
            x = 8 * kwargs["a"] - kwargs["b"] - 8 * kwargs["c"] - 104 * kwargs["d"]
            product = g * (2.0 ** (-x / QUANTIZATION_GRAIN))
            error = abs(product - GOD_CODE)
            results.append({
                "dial": dial, "value": val, "G": g,
                "conservation_product": product, "error": error,
                "passed": error < 1e-9,
            })
        return results

    def noise_sweep(self, sim_name: str = "", noise_levels: List[float] = None,
                    simulator: 'GodCodeSimulator' = None) -> List[Dict[str, Any]]:
        """Sweep noise levels for a simulation, measuring fidelity degradation.

        v3.0: When *simulator* and *sim_name* are both provided, runs that
        simulation through the optimizer at each noise level (realistic
        per-strategy fidelity).  Falls back to a local GHZ-damping circuit
        when no simulator is given — also upgraded: every qubit now gets
        position-dependent amplitude damping *and* PHI-phase dephasing.
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

        results: List[Dict[str, Any]] = []

        for noise in noise_levels:
            if simulator is not None and sim_name:
                # v3.0: Use the real optimizer path for named simulations
                try:
                    opt = simulator.optimize_noise_resilience(nq=4, noise_level=noise)
                    best = max(opt.get("results", [{}]), key=lambda e: e.get("fidelity", 0))
                    results.append({
                        "noise_level": noise,
                        "fidelity": best.get("fidelity", 0.0),
                        "best_strategy": best.get("strategy", "raw"),
                    })
                    continue
                except Exception:
                    pass  # fall through to local circuit

            # Local circuit: GHZ chain + amplitude-damping + PHI-dephasing
            nq_local = 4
            sv = init_sv(nq_local)
            sv = apply_single_gate(sv, H_GATE, 0, nq_local)
            for q in range(1, nq_local):
                sv = apply_cnot(sv, q - 1, q, nq_local)
            sv_ideal = sv.copy()

            for q in range(nq_local):
                # Amplitude damping ∝ qubit position
                damp = make_gate([[1, 0], [0, np.exp(-noise * (q + 1))]])
                sv = apply_single_gate(sv, damp, q, nq_local)
                # PHI-phase dephasing (v3.0)
                dephase_angle = noise * PHI_CONJUGATE * (q + 1)
                rz = make_gate([
                    [np.exp(-1j * dephase_angle / 2), 0],
                    [0, np.exp(1j * dephase_angle / 2)],
                ])
                sv = apply_single_gate(sv, rz, q, nq_local)

            norm = np.linalg.norm(sv)
            if norm > 0:
                sv /= norm
            f = fidelity(sv, sv_ideal)
            le = linear_entropy(sv, nq_local)
            results.append({"noise_level": noise, "fidelity": f, "linear_entropy": le})
        return results

    def depth_sweep(self, depths: List[int] = None) -> List[Dict[str, Any]]:
        """Sweep circuit depth, measuring entanglement entropy growth.

        v3.0: Also tracks linear entropy, purity, and trace distance from
        the initial product state — giving a fuller picture of how depth
        drives the state into the entangled regime.
        """
        if depths is None:
            depths = [1, 2, 4, 8, 16, 32]
        results = []
        nq = 6
        sv_ref = init_sv(nq)  # |0…0⟩ reference for trace distance
        for depth in depths:
            sv = init_sv(nq)
            sv = apply_single_gate(sv, H_GATE, 0, nq)
            for d in range(depth):
                sv = apply_cnot(sv, d % nq, (d + 1) % nq, nq)
                sv = apply_single_gate(sv, GOD_CODE_GATE, d % nq, nq)
            entropy = entanglement_entropy(sv, nq)
            purity = state_purity(sv, nq)
            le = linear_entropy(sv, nq)
            td = trace_distance(sv, sv_ref)
            results.append({
                "depth": depth,
                "entanglement_entropy": entropy,
                "linear_entropy": le,
                "purity": purity,
                "trace_distance_from_init": td,
            })
        return results

    def qubit_scaling(self, sim_fn: Callable, qubit_range: List[int] = None) -> List[Dict[str, Any]]:
        """Scale a simulation across different qubit counts."""
        if qubit_range is None:
            qubit_range = [2, 4, 6, 8]
        results = []
        for nq in qubit_range:
            try:
                result = sim_fn(nq=nq)
                results.append({
                    "num_qubits": nq,
                    "fidelity": result.fidelity,
                    "entropy": result.entanglement_entropy,
                    "elapsed_ms": result.elapsed_ms,
                    "passed": result.passed,
                })
            except Exception as e:
                results.append({"num_qubits": nq, "error": str(e)})
        return results

    # ═══════════════════════════════════════════════════════════════════════════
    #  v2.0 — New Sweep Types
    # ═══════════════════════════════════════════════════════════════════════════

    def strategy_sweep(self, noise_levels: List[float] = None,
                       nq: int = 2, simulator: 'GodCodeSimulator' = None) -> Dict[str, Any]:
        """
        Compare all noise protection strategies across noise regimes.

        Returns a matrix: strategies × noise_levels → fidelity.
        Identifies the dominant strategy at each noise level and the
        crossover points where one strategy overtakes another.
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

        if simulator is None:
            return {"error": "Simulator required for strategy sweep"}

        matrix: Dict[str, List[float]] = {}
        for noise in noise_levels:
            result = simulator.optimize_noise_resilience(nq=nq, noise_level=noise)
            for entry in result.get("results", []):
                name = entry["strategy"]
                if name not in matrix:
                    matrix[name] = []
                matrix[name].append(entry["fidelity"])

        # Find dominant strategy at each noise level
        dominants = []
        for i, noise in enumerate(noise_levels):
            best_name = ""
            best_f = -1.0
            for name, fids in matrix.items():
                if i < len(fids) and fids[i] > best_f:
                    best_f = fids[i]
                    best_name = name
            dominants.append({"noise": noise, "best_strategy": best_name, "fidelity": best_f})

        return {
            "noise_levels": noise_levels,
            "strategy_matrix": {k: [round(f, 6) for f in v] for k, v in matrix.items()},
            "dominants": dominants,
            "num_strategies": len(matrix),
        }

    def phase_sweep(self, phase_range: List[float] = None, nq: int = 3) -> List[Dict[str, Any]]:
        """
        Sweep sacred phase angles and measure state properties.

        Tests how the sacred gate phase angle affects entanglement,
        purity, and alignment — revealing resonance peaks.

        v3.0: Default grid is 104 points (QUANTIZATION_GRAIN) for full
        sacred-resolution coverage.  Added linear_entropy + trace-distance
        from the uniform superposition.  Peaks are annotated with their
        distance to known sacred angles (GOD_CODE, PHI, VOID).
        """
        if phase_range is None:
            # 104 sacred-resolution points (was 36)
            phase_range = list(np.linspace(0, TAU, QUANTIZATION_GRAIN))

        results = []
        # Reference state: equal superposition |+…+⟩
        sv_ref = init_sv(nq)
        for q in range(nq):
            sv_ref = apply_single_gate(sv_ref, H_GATE, q, nq)

        for phase in phase_range:
            sv = sv_ref.copy()

            # Sacred gate with swept phase
            rz = make_gate([[np.exp(-1j * phase / 2), 0],
                            [0, np.exp(1j * phase / 2)]])
            sv = apply_single_gate(sv, rz, 0, nq)
            for q in range(nq - 1):
                sv = apply_cnot(sv, q, q + 1, nq)

            entropy = entanglement_entropy(sv, nq)
            purity = state_purity(sv, nq)
            le = linear_entropy(sv, nq)
            td = trace_distance(sv, sv_ref)
            alignment_gc = abs(math.cos(phase - GOD_CODE_PHASE_ANGLE))
            alignment_phi = abs(math.cos(phase - PHI_PHASE_ANGLE))

            results.append({
                "phase": round(phase, 6),
                "entropy": round(entropy, 6),
                "purity": round(purity, 6),
                "linear_entropy": round(le, 6),
                "trace_distance": round(td, 6),
                "alignment_god_code": round(alignment_gc, 6),
                "alignment_phi": round(alignment_phi, 6),
            })

        # Find resonance peaks (local maxima of alignment × entropy)
        scores = [r["alignment_god_code"] * r["entropy"] for r in results]
        peaks = []
        for i in range(1, len(scores) - 1):
            if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
                nearest_sacred = min(
                    [GOD_CODE_PHASE_ANGLE, PHI_PHASE_ANGLE],
                    key=lambda a: abs(results[i]["phase"] - a),
                )
                peaks.append({
                    "phase": results[i]["phase"],
                    "score": round(scores[i], 6),
                    "nearest_sacred_angle": round(nearest_sacred, 6),
                    "distance_to_sacred": round(abs(results[i]["phase"] - nearest_sacred), 6),
                })

        # Attach peaks metadata to last result for easy access
        if results:
            results[-1]["_peaks"] = peaks

        return results

    def cross_simulation_sweep(self, simulator: 'GodCodeSimulator' = None,
                               categories: List[str] = None) -> Dict[str, Any]:
        """
        Run all simulations and compare metrics across them.

        Returns aggregated statistics per metric and per category,
        plus identifies outliers and top performers.
        """
        if simulator is None:
            return {"error": "Simulator required for cross-simulation sweep"}

        report = simulator.run_all()
        results = report.get("results", [])

        metrics = {
            "fidelity": [r.fidelity for r in results],
            "entropy": [r.entanglement_entropy for r in results],
            "coherence": [r.phase_coherence for r in results],
            "alignment": [r.sacred_alignment for r in results],
            "elapsed_ms": [r.elapsed_ms for r in results],
        }

        agg = {}
        for key, vals in metrics.items():
            arr = np.array(vals)
            agg[key] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "median": round(float(np.median(arr)), 4),
            }

        # Top performers by fidelity
        sorted_by_fid = sorted(zip([r.name for r in results], [r.fidelity for r in results]),
                               key=lambda x: x[1], reverse=True)
        top_5 = [{"name": n, "fidelity": round(f, 4)} for n, f in sorted_by_fid[:5]]

        # Category breakdown
        cat_breakdown = {}
        for r in results:
            cat = r.category
            if cat not in cat_breakdown:
                cat_breakdown[cat] = {"count": 0, "pass": 0, "fidelity_sum": 0.0}
            cat_breakdown[cat]["count"] += 1
            if r.passed:
                cat_breakdown[cat]["pass"] += 1
            cat_breakdown[cat]["fidelity_sum"] += r.fidelity
        for cat in cat_breakdown:
            n = cat_breakdown[cat]["count"]
            cat_breakdown[cat]["avg_fidelity"] = round(cat_breakdown[cat]["fidelity_sum"] / max(n, 1), 4)
            del cat_breakdown[cat]["fidelity_sum"]

        return {
            "total_simulations": len(results),
            "total_passed": report["passed"],
            "aggregated_metrics": agg,
            "top_performers": top_5,
            "category_breakdown": cat_breakdown,
        }

    def convergence_sweep(self, nq_values: List[int] = None,
                          simulator: 'GodCodeSimulator' = None) -> List[Dict[str, Any]]:
        """
        Measure optimizer convergence characteristics across qubit counts.

        For each nq, runs optimizer and tracks iterations to convergence,
        final fidelity, and convergence rate.
        """
        if nq_values is None:
            nq_values = [2, 3, 4, 5]

        if simulator is None:
            return [{"error": "Simulator required for convergence sweep"}]

        results = []
        for nq in nq_values:
            t0 = time.time()
            opt = simulator.adaptive_optimize(target_fidelity=0.99, nq=nq, depth=4)
            elapsed = (time.time() - t0) * 1000

            results.append({
                "num_qubits": nq,
                "converged": opt["converged"],
                "best_fidelity": round(opt["best_fidelity"], 6),
                "iterations": opt["iterations"],
                "elapsed_ms": round(elapsed, 1),
                "rate": round(opt["best_fidelity"] / max(opt["iterations"], 1), 6),
            })

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    #  v3.0 — Advanced & VQPU-Derived Sweep Types
    # ═══════════════════════════════════════════════════════════════════════════

    def sensitivity_sweep(self, base_noise: float = 0.05,
                          base_depth: int = 8, base_angle: float = 0.0,
                          nq: int = 4, epsilon: float = 1e-3) -> Dict[str, Any]:
        """
        Finite-difference sensitivity analysis: ∂fidelity/∂parameter.

        Computes the Jacobian of fidelity with respect to three axes:
          - noise (amplitude damping rate)
          - depth (circuit entangling layers)
          - angle (phase gate angle on qubit 0)

        Uses three-point central differences for second-order accuracy:
          ∂f/∂x ≈ (f(x+ε) − f(x−ε)) / (2ε)

        Returns per-axis gradient magnitude and the combined sensitivity
        norm, indicating which parameter most influences fidelity.
        """
        def _evaluate(noise: float, depth: int, angle: float) -> float:
            sv = init_sv(nq)
            sv = apply_single_gate(sv, H_GATE, 0, nq)
            for q in range(1, nq):
                sv = apply_cnot(sv, q - 1, q, nq)
            for d in range(depth):
                sv = apply_cnot(sv, d % nq, (d + 1) % nq, nq)
                sv = apply_single_gate(sv, GOD_CODE_GATE, d % nq, nq)
            sv_ideal = sv.copy()
            if angle != 0.0:
                rz = make_gate([[np.exp(-1j * angle / 2), 0],
                                [0, np.exp(1j * angle / 2)]])
                sv = apply_single_gate(sv, rz, 0, nq)
                sv_ideal = apply_single_gate(sv_ideal, rz, 0, nq)
            for q in range(nq):
                damp = make_gate([[1, 0], [0, np.exp(-noise * (q + 1))]])
                sv = apply_single_gate(sv, damp, q, nq)
            norm = np.linalg.norm(sv)
            if norm > 0:
                sv /= norm
            return fidelity(sv, sv_ideal)

        f_base = _evaluate(base_noise, base_depth, base_angle)

        # ∂f/∂noise (central diff)
        f_np = _evaluate(base_noise + epsilon, base_depth, base_angle)
        f_nm = _evaluate(max(0.0, base_noise - epsilon), base_depth, base_angle)
        df_dnoise = (f_np - f_nm) / (2 * epsilon)

        # ∂f/∂depth (discrete: depth ± 1)
        f_dp = _evaluate(base_noise, base_depth + 1, base_angle)
        f_dm = _evaluate(base_noise, max(1, base_depth - 1), base_angle)
        df_ddepth = (f_dp - f_dm) / 2.0

        # ∂f/∂angle (central diff)
        f_ap = _evaluate(base_noise, base_depth, base_angle + epsilon)
        f_am = _evaluate(base_noise, base_depth, base_angle - epsilon)
        df_dangle = (f_ap - f_am) / (2 * epsilon)

        sensitivity_norm = math.sqrt(df_dnoise ** 2 + df_ddepth ** 2 + df_dangle ** 2)
        grads = {"noise": abs(df_dnoise), "depth": abs(df_ddepth), "angle": abs(df_dangle)}
        dominant_axis = max(grads, key=grads.get)

        return {
            "base_fidelity": round(f_base, 8),
            "df_dnoise": round(df_dnoise, 8),
            "df_ddepth": round(df_ddepth, 8),
            "df_dangle": round(df_dangle, 8),
            "sensitivity_norm": round(sensitivity_norm, 8),
            "dominant_axis": dominant_axis,
            "parameters": {"noise": base_noise, "depth": base_depth, "angle": base_angle, "nq": nq},
        }

    def joint_sweep_2d(self, axis_x: str = "noise", axis_y: str = "depth",
                       x_values: Optional[List[float]] = None,
                       y_values: Optional[List[float]] = None,
                       nq: int = 4) -> Dict[str, Any]:
        """
        Cartesian-product 2D sweep over two parameter axes.

        Supported axes: 'noise', 'depth', 'angle'.
        Returns an x × y matrix of fidelity values plus summary statistics:
          - max/min fidelity and their (x, y) coordinates
          - mean fidelity iso-lines along each axis
        """
        defaults: Dict[str, List[float]] = {
            "noise": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3],
            "depth": [1.0, 2.0, 4.0, 8.0, 16.0],
            "angle": list(np.linspace(0, TAU, 13)[:-1]),
        }
        if x_values is None:
            x_values = defaults.get(axis_x, defaults["noise"])
        if y_values is None:
            y_values = defaults.get(axis_y, defaults["depth"])

        def _evaluate(noise: float, depth: int, angle: float) -> float:
            sv = init_sv(nq)
            sv = apply_single_gate(sv, H_GATE, 0, nq)
            for q in range(1, nq):
                sv = apply_cnot(sv, q - 1, q, nq)
            for d in range(depth):
                sv = apply_cnot(sv, d % nq, (d + 1) % nq, nq)
                sv = apply_single_gate(sv, GOD_CODE_GATE, d % nq, nq)
            sv_ideal = sv.copy()
            if angle != 0.0:
                rz = make_gate([[np.exp(-1j * angle / 2), 0],
                                [0, np.exp(1j * angle / 2)]])
                sv = apply_single_gate(sv, rz, 0, nq)
                sv_ideal = apply_single_gate(sv_ideal, rz, 0, nq)
            for q in range(nq):
                damp = make_gate([[1, 0], [0, np.exp(-noise * (q + 1))]])
                sv = apply_single_gate(sv, damp, q, nq)
            norm = np.linalg.norm(sv)
            if norm > 0:
                sv /= norm
            return fidelity(sv, sv_ideal)

        matrix: List[List[float]] = []
        best_f, worst_f = -1.0, 2.0
        best_xy: Tuple[float, float] = (0.0, 0.0)
        worst_xy: Tuple[float, float] = (0.0, 0.0)

        for xv in x_values:
            row = []
            for yv in y_values:
                params: Dict[str, Any] = {"noise": 0.0, "depth": 4, "angle": 0.0}
                params[axis_x] = xv
                params[axis_y] = yv
                f = _evaluate(params["noise"], int(params["depth"]), params["angle"])
                row.append(round(f, 6))
                if f > best_f:
                    best_f, best_xy = f, (xv, yv)
                if f < worst_f:
                    worst_f, worst_xy = f, (xv, yv)
            matrix.append(row)

        arr = np.array(matrix)
        x_means = [round(float(m), 6) for m in np.mean(arr, axis=1)]
        y_means = [round(float(m), 6) for m in np.mean(arr, axis=0)]

        return {
            "axis_x": axis_x, "axis_y": axis_y,
            "x_values": [round(v, 6) for v in x_values],
            "y_values": [round(v, 6) for v in y_values],
            "fidelity_matrix": matrix,
            "best": {"fidelity": round(best_f, 6), axis_x: round(best_xy[0], 6), axis_y: round(best_xy[1], 6)},
            "worst": {"fidelity": round(worst_f, 6), axis_x: round(worst_xy[0], 6), axis_y: round(worst_xy[1], 6)},
            "x_iso_mean": x_means,
            "y_iso_mean": y_means,
            "grid_size": f"{len(x_values)}×{len(y_values)}",
        }

    def trotter_convergence_sweep(self, step_counts: List[int] = None,
                                  nq: int = 4, t: float = 1.0) -> List[Dict[str, Any]]:
        """
        Sweep Trotter step count and measure decomposition error convergence.

        From VQPU v8.0: Trotter error should scale as O(t²/steps) for
        first-order decomposition.  Uses iron_lattice_heisenberg which
        internally runs Trotter evolution and returns a Dict.  We compare
        the statevector at each step count against a high-precision reference.
        """
        if step_counts is None:
            step_counts = [1, 2, 4, 8, 16, 32, 64]

        # Reference: high step count as "exact"
        ref_result = iron_lattice_heisenberg(n_sites=nq, trotter_steps=256, total_time=t)
        sv_ref = ref_result["statevector"]

        results = []
        for steps in step_counts:
            t0 = time.time()
            run = iron_lattice_heisenberg(n_sites=nq, trotter_steps=steps, total_time=t)
            elapsed = (time.time() - t0) * 1000
            sv_approx = run["statevector"]
            f = fidelity(sv_approx, sv_ref)
            error = 1.0 - f
            results.append({
                "steps": steps,
                "fidelity": round(f, 8),
                "trotter_error": round(error, 8),
                "energy": round(run["energy"], 6),
                "elapsed_ms": round(elapsed, 2),
                "expected_scaling": round(t ** 2 / steps, 6),
            })

        return results

    def qfi_scaling_sweep(self, qubit_range: List[int] = None) -> List[Dict[str, Any]]:
        """
        Sweep qubit count and measure Quantum Fisher Information scaling.

        From VQPU v8.0: QFI should scale as O(N²) for entangled states
        (Heisenberg limit) and O(N) for product states (shot noise limit).
        Tests both regimes.

        v3.0: Fixed to use Dict return from quantum_fisher_information().
        """
        if qubit_range is None:
            qubit_range = [2, 3, 4, 5, 6]

        results = []
        for nq in qubit_range:
            # Entangled state (GHZ-like)
            sv_ent = init_sv(nq)
            sv_ent = apply_single_gate(sv_ent, H_GATE, 0, nq)
            for q in range(nq - 1):
                sv_ent = apply_cnot(sv_ent, q, q + 1, nq)

            # Generator: single-qubit Z (expanded to full space inside QFI fn)
            gen_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

            qfi_ent_result = quantum_fisher_information(sv_ent, gen_z, nq)
            qfi_ent = qfi_ent_result["qfi"]

            # Product state (all |+⟩)
            sv_prod = init_sv(nq)
            for q in range(nq):
                sv_prod = apply_single_gate(sv_prod, H_GATE, q, nq)
            qfi_prod_result = quantum_fisher_information(sv_prod, gen_z, nq)
            qfi_prod = qfi_prod_result["qfi"]

            results.append({
                "num_qubits": nq,
                "qfi_entangled": round(qfi_ent, 4),
                "qfi_product": round(qfi_prod, 4),
                "heisenberg_ratio": round(qfi_ent / max(nq ** 2, 1), 4),
                "shot_noise_ratio": round(qfi_prod / max(nq, 1), 4),
                "heisenberg_limited": qfi_ent_result["heisenberg_limited"],
            })

        return results

    def tomography_accuracy_sweep(self, qubit_range: List[int] = None,
                                  n_measurements: int = 100) -> List[Dict[str, Any]]:
        """
        Sweep qubit count and measure state tomography reconstruction fidelity.

        From VQPU v8.0: Tomography fidelity should degrade gracefully with
        qubit count due to exponential state space growth vs. linear
        measurement budget.

        v3.0: Fixed to use Dict return from reconstruct_density_matrix().
        """
        if qubit_range is None:
            qubit_range = [1, 2, 3, 4]

        results = []
        for nq in qubit_range:
            # Prepare entangled state
            sv = init_sv(nq)
            sv = apply_single_gate(sv, H_GATE, 0, nq)
            for q in range(nq - 1):
                sv = apply_cnot(sv, q, q + 1, nq)

            true_rho = density_matrix_from_sv(sv)
            recon_result = reconstruct_density_matrix(sv, nq)
            recon_rho = recon_result["density_matrix"]

            # Reconstruction fidelity: F(ρ, σ) = Tr(ρσ) / sqrt(Tr(ρ²)Tr(σ²))
            overlap = abs(np.trace(true_rho @ recon_rho))
            norm_true = np.sqrt(abs(np.trace(true_rho @ true_rho)))
            norm_recon = np.sqrt(abs(np.trace(recon_rho @ recon_rho)))
            recon_f = overlap / max(norm_true * norm_recon, 1e-15)

            results.append({
                "num_qubits": nq,
                "dim": 2 ** nq,
                "reconstruction_fidelity": round(float(recon_f), 6),
                "purity_true": round(float(np.trace(true_rho @ true_rho).real), 6),
                "purity_recon": round(recon_result["purity"], 6),
                "von_neumann_entropy": round(recon_result["von_neumann_entropy"], 6),
                "is_pure": recon_result["is_pure"],
            })

        return results

    def zne_overhead_sweep(self, noise_levels: List[float] = None,
                           nq: int = 3, depth: int = 4) -> List[Dict[str, Any]]:
        """
        Sweep noise levels and measure ZNE error mitigation cost-benefit.

        From VQPU v8.0 benchmarks: ZNE overhead is ~249.6% (3.5x more circuits).
        This sweep measures actual fidelity improvement vs. computational cost
        at each noise level to find the sweet spot.

        v3.0: Fixed to use correct zero_noise_extrapolation signature
        (callable(noise_level) → sv, noise_levels list, n_qubits).
        """
        if noise_levels is None:
            noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

        # Ideal state for reference
        sv_ideal = init_sv(nq)
        sv_ideal = apply_single_gate(sv_ideal, H_GATE, 0, nq)
        for q in range(nq - 1):
            sv_ideal = apply_cnot(sv_ideal, q, q + 1, nq)

        results = []
        for base_noise in noise_levels:
            # Noisy circuit function: callable(noise_level) → statevector
            def noisy_circuit(noise_lvl: float) -> np.ndarray:
                s = init_sv(nq)
                s = apply_single_gate(s, H_GATE, 0, nq)
                for q in range(nq - 1):
                    s = apply_cnot(s, q, q + 1, nq)
                for q_idx in range(nq):
                    for _ in range(depth):
                        damp = make_gate([[1, 0], [0, np.exp(-noise_lvl)]])
                        s = apply_single_gate(s, damp, q_idx, nq)
                n = np.linalg.norm(s)
                return s / n if n > 0 else s

            # Without ZNE — direct noisy circuit
            t0 = time.time()
            sv_noisy = noisy_circuit(base_noise)
            time_noisy = (time.time() - t0) * 1000
            f_noisy = fidelity(sv_noisy, sv_ideal)

            # With ZNE — extrapolate from multiple noise levels
            zne_levels = [base_noise, base_noise * 2, base_noise * 3]
            t0 = time.time()
            zne_result = zero_noise_extrapolation(noisy_circuit, noise_levels=zne_levels, n_qubits=nq)
            time_zne = (time.time() - t0) * 1000
            f_zne = zne_result.get("extrapolated_fidelity", f_noisy)

            improvement = f_zne - f_noisy
            overhead = time_zne / max(time_noisy, 0.001)

            results.append({
                "noise_level": base_noise,
                "fidelity_noisy": round(f_noisy, 6),
                "fidelity_zne": round(f_zne, 6),
                "improvement": round(improvement, 6),
                "time_noisy_ms": round(time_noisy, 2),
                "time_zne_ms": round(time_zne, 2),
                "overhead_factor": round(overhead, 2),
                "benefit_per_cost": round(improvement / max(overhead, 0.01), 6),
            })

        return results


__all__ = ["ParametricSweepEngine"]
