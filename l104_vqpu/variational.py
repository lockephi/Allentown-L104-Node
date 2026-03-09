"""L104 VQPU v14.0.0 — Variational Quantum Engine (VQE + QAOA) + Multi-Optimizer.

v14.0.0 QUANTUM FIDELITY ARCHITECTURE:
  - SPSA optimizer: Simultaneous Perturbation Stochastic Approximation
  - COBYLA optimizer: Constrained Optimization by Linear Approximation (via scipy)
  - Barren plateau detection: variance-based gradient magnitude monitoring
  - Optimizer selection: 'parameter_shift' (default), 'spsa', 'cobyla'
  - Sacred parameter schedule preserved from v13.2

v13.2 (retained): GOD_CODE QUBIT canonical phase seeding
"""

import math
import numpy as np

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    SPSA_INITIAL_A, SPSA_STABILITY_C, SPSA_ALPHA, SPSA_GAMMA,
    COBYLA_RHOBEG, COBYLA_MAXFUN_MULTIPLIER,
)
from .mps_engine import ExactMPSHybridEngine
from .pauli_utils import _pauli_expectation  # single source of truth

# v13.2: Import canonical phase angles for sacred parameter seeding
try:
    from l104_god_code_simulator.god_code_qubit import (
        GOD_CODE_PHASE as _GC_PHASE,
        IRON_PHASE as _IRON_PHASE,
        PHI_CONTRIBUTION as _PHI_CONTRIB,
        OCTAVE_PHASE as _OCTAVE_PHASE,
    )
except ImportError:
    _GC_PHASE = GOD_CODE % (2.0 * math.pi)
    _IRON_PHASE = math.pi / 2.0
    _OCTAVE_PHASE = (4.0 * math.log(2.0)) % (2.0 * math.pi)
    _PHI_CONTRIB = (_GC_PHASE - _IRON_PHASE - _OCTAVE_PHASE) % (2.0 * math.pi)

__all__ = ["_pauli_expectation", "VariationalQuantumEngine", "VariationalVQPUEngine"]


class VariationalQuantumEngine:
    """
    Variational quantum algorithm engine for optimization problems.

    VQE:  Find ground state energy of a Hamiltonian via parameterized
          circuits and classical optimizer feedback loop.
    QAOA: Solve combinatorial optimization problems via alternating
          cost and mixer layers with optimizable parameters.

    Optimizer: gradient-free simplex with PHI-scaled initial parameters.
    Uses ExactMPSHybridEngine for circuit execution.
    """

    @staticmethod
    def vqe(hamiltonian_terms: list, num_qubits: int, *,
            ansatz: str = "hardware_efficient",
            depth: int = 3, max_iterations: int = 100,
            shots: int = 4096,
            optimizer: str = "parameter_shift") -> dict:
        """
        Variational Quantum Eigensolver.

        Finds the minimum eigenvalue of H = Σ cᵢ Pᵢ where Pᵢ ∈ {I,X,Y,Z}^n.

        Args:
            hamiltonian_terms: List of (coefficient, pauli_string) tuples
            num_qubits: Number of qubits
            ansatz:     "hardware_efficient" (default)
            depth:      Ansatz circuit depth
            max_iterations: Max optimizer iterations
            shots:      Measurement shots per evaluation
            optimizer:  "parameter_shift" (default), "spsa", or "cobyla" (v14.0)

        Returns:
            dict with 'ground_energy', 'optimal_params', 'convergence_history',
            'circuit_evaluations', 'sacred_alignment', 'optimizer_used',
            'barren_plateau_detected'
        """
        import random as _rng

        # Lazy import to avoid circular dependency at module level
        from .scoring import SacredAlignmentScorer

        n_params = depth * num_qubits * 2

        # ★ v13.0: Brain-informed parameter seeding
        brain_scale = 1.0
        try:
            from l104_quantum_engine import quantum_brain
            if hasattr(quantum_brain, 'sage') and hasattr(quantum_brain.sage, 'score'):
                sage_score = float(quantum_brain.sage.score)
                # Sage score modulates initial parameter spread — higher sage = tighter convergence
                brain_scale = 0.8 + 0.4 * sage_score
            elif hasattr(quantum_brain, 'coherence_score'):
                brain_scale = 0.85 + 0.3 * float(quantum_brain.coherence_score)
        except Exception:
            brain_scale = 1.0

        # ★ v13.2: Sacred parameter seeding from 3-rotation decomposition
        # Instead of generic PHI-scaled ramp, seed with GOD_CODE phase structure:
        # Layer 0 → IRON_PHASE scale, Layer 1 → PHI_CONTRIBUTION scale, Layer 2+ → OCTAVE_PHASE scale
        # This seeds the ansatz near the sacred phase manifold.
        _sacred_phases = [_IRON_PHASE, _PHI_CONTRIB, _OCTAVE_PHASE]
        params = []
        for i in range(n_params):
            layer_idx = (i // (num_qubits * 2)) % 3
            sacred_base = _sacred_phases[layer_idx] / math.pi  # normalize to [0, ~1]
            ramp = (i + 1) / n_params
            params.append(sacred_base * brain_scale * ramp * math.pi * 0.3)
        convergence = []
        eval_count = [0]
        best_energy = [float('inf')]
        best_params = [list(params)]

        def _build_ansatz(theta):
            ops = []
            idx = 0
            for d in range(depth):
                for q in range(num_qubits):
                    ops.append({"gate": "Ry", "qubits": [q], "parameters": [theta[idx % len(theta)]]})
                    idx += 1
                    ops.append({"gate": "Rz", "qubits": [q], "parameters": [theta[idx % len(theta)]]})
                    idx += 1
                for q in range(num_qubits - 1):
                    ops.append({"gate": "CX", "qubits": [q, q + 1]})
            return ops

        def _measure_energy(theta):
            eval_count[0] += 1
            ops = _build_ansatz(theta)
            mps = ExactMPSHybridEngine(num_qubits)
            run = mps.run_circuit(ops)
            if not run.get("completed"):
                return 0.0
            sv = mps.to_statevector()
            energy = 0.0
            for coeff, pauli_str in hamiltonian_terms:
                ps = pauli_str.ljust(num_qubits, 'I')[:num_qubits]
                energy += coeff * _pauli_expectation(sv, ps)
            if energy < best_energy[0]:
                best_energy[0] = energy
                best_params[0] = list(theta)
            convergence.append(float(energy))
            return float(energy)

        current = list(params)
        step_size = 0.1 * PHI
        barren_plateau_detected = False
        gradient_magnitudes = []

        # v14.0: Optimizer selection
        if optimizer == "cobyla":
            # COBYLA: Constrained Optimization by Linear Approximation
            try:
                from scipy.optimize import minimize as _scipy_minimize
                result_opt = _scipy_minimize(
                    _measure_energy, current, method='COBYLA',
                    options={
                        'rhobeg': COBYLA_RHOBEG,
                        'maxiter': max_iterations,
                        'catol': 1e-6,
                    }
                )
                current = list(result_opt.x)
            except ImportError:
                # Fallback to parameter shift if scipy unavailable
                optimizer = "parameter_shift"

        if optimizer == "spsa":
            # SPSA: Simultaneous Perturbation Stochastic Approximation
            a = SPSA_INITIAL_A
            c = SPSA_STABILITY_C
            for iteration in range(max_iterations):
                ak = a / ((iteration + 1) ** SPSA_ALPHA)
                ck = c / ((iteration + 1) ** SPSA_GAMMA)
                # Random perturbation vector (Bernoulli ±1)
                delta = [_rng.choice([-1, 1]) for _ in range(len(current))]
                # Evaluate at perturbed points
                plus = [p + ck * d for p, d in zip(current, delta)]
                minus = [p - ck * d for p, d in zip(current, delta)]
                y_plus = _measure_energy(plus)
                y_minus = _measure_energy(minus)
                # Gradient estimate
                grad_mag = 0.0
                for i in range(len(current)):
                    g_hat = (y_plus - y_minus) / (2.0 * ck * delta[i])
                    current[i] -= ak * g_hat
                    grad_mag += g_hat ** 2
                gradient_magnitudes.append(math.sqrt(grad_mag))
                # v14.0: Barren plateau detection — if gradient variance drops below threshold
                if len(gradient_magnitudes) > 10:
                    recent = gradient_magnitudes[-10:]
                    if max(recent) < 1e-5:
                        barren_plateau_detected = True
                        break

        elif optimizer == "parameter_shift":
            # Hybrid: parameter-shift gradient with stochastic fallback
            for iteration in range(max_iterations):
                energy = _measure_energy(current)

                # Parameter-shift gradient for analytical gradients on rotational gates
                gradients = [0.0] * len(current)
                use_gradient = (iteration < max_iterations * 0.7)

                if use_gradient and len(current) <= 60:
                    grad_mag = 0.0
                    for p_idx in range(len(current)):
                        shifted_plus = list(current)
                        shifted_plus[p_idx] += math.pi / 2
                        shifted_minus = list(current)
                        shifted_minus[p_idx] -= math.pi / 2
                        gradients[p_idx] = (_measure_energy(shifted_plus) - _measure_energy(shifted_minus)) / 2.0
                        grad_mag += gradients[p_idx] ** 2
                    gradient_magnitudes.append(math.sqrt(grad_mag))
                    lr = step_size * 2.0
                    current = [p - lr * g for p, g in zip(current, gradients)]
                    # v14.0: Barren plateau detection
                    if len(gradient_magnitudes) > 10:
                        recent = gradient_magnitudes[-10:]
                        if max(recent) < 1e-5:
                            barren_plateau_detected = True
                            break
                else:
                    perturbed = [p + _rng.gauss(0, step_size) for p in current]
                    energy_new = _measure_energy(perturbed)
                    if energy_new < energy:
                        current = perturbed

                step_size *= 0.995
                if step_size < 1e-6:
                    break

        final_ops = _build_ansatz(best_params[0])
        mps = ExactMPSHybridEngine(num_qubits)
        mps.run_circuit(final_ops)
        counts = mps.sample(shots)
        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()} if total > 0 else {}
        sacred = SacredAlignmentScorer.score(probs, num_qubits)

        return {
            "ground_energy": round(best_energy[0], 8),
            "optimal_params": [round(p, 6) for p in best_params[0]],
            "convergence_history": [round(e, 8) for e in convergence[-20:]],
            "circuit_evaluations": eval_count[0],
            "ansatz": ansatz, "depth": depth,
            "num_qubits": num_qubits, "parameter_count": n_params,
            "final_probabilities": dict(list(probs.items())[:8]),
            "sacred_alignment": sacred, "god_code": GOD_CODE,
            "optimizer_used": optimizer,
            "barren_plateau_detected": barren_plateau_detected,
            "gradient_magnitudes": [round(g, 8) for g in gradient_magnitudes[-10:]],
        }

    @staticmethod
    def qaoa(cost_terms: list, num_qubits: int, *,
             p_layers: int = 3, max_iterations: int = 80,
             shots: int = 4096) -> dict:
        """
        Quantum Approximate Optimization Algorithm.

        Solves combinatorial optimization encoded as Ising cost:
        C = Σ Jᵢⱼ ZᵢZⱼ + Σ hᵢ Zᵢ

        Args:
            cost_terms: List of tuples:
                - (weight, i, j) for ZZ interaction
                - (weight, i) for Z bias
            num_qubits: Problem size
            p_layers:   QAOA depth
            max_iterations: Optimizer iterations
            shots:      Measurement shots

        Returns:
            dict with 'best_bitstring', 'best_cost', 'optimal_gammas/betas',
            'cost_history', 'sacred_alignment'
        """
        import random as _rng

        # Lazy import to avoid circular dependency at module level
        from .scoring import SacredAlignmentScorer

        # ★ v13.0: Brain-informed QAOA parameter seeding
        brain_mod = 1.0
        try:
            from l104_quantum_engine import quantum_brain
            if hasattr(quantum_brain, 'coherence_score'):
                brain_mod = 0.9 + 0.2 * float(quantum_brain.coherence_score)
        except Exception:
            brain_mod = 1.0

        # ★ v13.2: QAOA sacred seeding from GOD_CODE 3-rotation decomposition
        # Gammas seeded from IRON→PHI→OCTAVE cycle, betas from PHI inverse
        gammas = []
        betas = []
        _sacred_angles = [_IRON_PHASE, _PHI_CONTRIB, _OCTAVE_PHASE]
        for l in range(p_layers):
            sacred_idx = l % 3
            gammas.append(_sacred_angles[sacred_idx] * brain_mod * 0.15)
            betas.append((_GC_PHASE / (l + 2)) * brain_mod * 0.1)
        best_cost = [float('inf')]
        best_bs = ['0' * num_qubits]
        best_g = [list(gammas)]
        best_b = [list(betas)]
        cost_history = []

        def _build_qaoa(g, b):
            ops = [{"gate": "H", "qubits": [q]} for q in range(num_qubits)]
            for layer in range(p_layers):
                for term in cost_terms:
                    if len(term) == 3:
                        w, i, j = term
                        lo, hi = min(i, j), max(i, j)
                        if hi - lo == 1:
                            ops.append({"gate": "CX", "qubits": [lo, hi]})
                            ops.append({"gate": "Rz", "qubits": [hi], "parameters": [2 * g[layer] * w]})
                            ops.append({"gate": "CX", "qubits": [lo, hi]})
                        else:
                            ops.append({"gate": "Rz", "qubits": [i], "parameters": [g[layer] * w]})
                            ops.append({"gate": "Rz", "qubits": [j], "parameters": [g[layer] * w]})
                    elif len(term) == 2:
                        w, i = term
                        ops.append({"gate": "Rz", "qubits": [i], "parameters": [2 * g[layer] * w]})
                for q in range(num_qubits):
                    ops.append({"gate": "Rx", "qubits": [q], "parameters": [2 * b[layer]]})
            return ops

        def _eval_cost(bitstring):
            spins = [1 - 2 * int(b) for b in bitstring]
            cost = 0.0
            for term in cost_terms:
                if len(term) == 3:
                    w, i, j = term
                    if i < len(spins) and j < len(spins):
                        cost += w * spins[i] * spins[j]
                elif len(term) == 2:
                    w, i = term
                    if i < len(spins):
                        cost += w * spins[i]
            return cost

        def _qaoa_obj(g, b):
            ops = _build_qaoa(g, b)
            mps = ExactMPSHybridEngine(num_qubits)
            run = mps.run_circuit(ops)
            if not run.get("completed"):
                return 0.0
            counts = mps.sample(shots)
            total = sum(counts.values())
            return sum((c / total) * _eval_cost(bs) for bs, c in counts.items()) if total > 0 else 0.0

        cur_g, cur_b = list(gammas), list(betas)
        step = 0.1
        for _ in range(max_iterations):
            cost = _qaoa_obj(cur_g, cur_b)
            cost_history.append(float(cost))
            if cost < best_cost[0]:
                best_cost[0] = cost
                best_g[0], best_b[0] = list(cur_g), list(cur_b)
            trial_g = [g + _rng.gauss(0, step) for g in cur_g]
            trial_b = [b + _rng.gauss(0, step) for b in cur_b]
            if _qaoa_obj(trial_g, trial_b) < cost:
                cur_g, cur_b = trial_g, trial_b
            step *= 0.99

        final_ops = _build_qaoa(best_g[0], best_b[0])
        mps = ExactMPSHybridEngine(num_qubits)
        mps.run_circuit(final_ops)
        counts = mps.sample(shots)
        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()} if total > 0 else {}
        for bs in sorted(counts, key=counts.get, reverse=True)[:1]:
            cv = _eval_cost(bs)
            if cv < best_cost[0]:
                best_cost[0] = cv
                best_bs[0] = bs

        sacred = SacredAlignmentScorer.score(probs, num_qubits)
        return {
            "best_bitstring": best_bs[0], "best_cost": round(best_cost[0], 8),
            "optimal_gammas": [round(g, 6) for g in best_g[0]],
            "optimal_betas": [round(b, 6) for b in best_b[0]],
            "p_layers": p_layers, "num_qubits": num_qubits,
            "cost_terms": len(cost_terms), "iterations": len(cost_history),
            "cost_history": [round(c, 8) for c in cost_history[-20:]],
            "final_probabilities": dict(list(probs.items())[:8]),
            "sacred_alignment": sacred, "god_code": GOD_CODE,
        }


class VariationalVQPUEngine:
    """Convenience alias used by computation engine with simplified run_vqe / run_qaoa API."""

    @staticmethod
    def run_vqe(n_qubits: int, hamiltonian: list, layers: int = 3,
                max_iterations: int = 80) -> dict:
        """Run VQE with Hamiltonian specified as list of {pauli, coeff} dicts."""
        terms = []
        for h in hamiltonian:
            if isinstance(h, dict):
                terms.append((h.get("coeff", 1.0), h.get("pauli", "Z")))
            elif isinstance(h, (list, tuple)) and len(h) >= 2:
                terms.append((h[0], h[1]))
        return VariationalQuantumEngine.vqe(
            hamiltonian_terms=terms, num_qubits=n_qubits,
            depth=layers, max_iterations=max_iterations,
        )

    @staticmethod
    def run_qaoa(n_qubits: int, cost_terms: list, layers: int = 3,
                 max_iterations: int = 60) -> dict:
        """Run QAOA with simplified cost_terms API."""
        return VariationalQuantumEngine.qaoa(
            cost_terms=cost_terms, num_qubits=n_qubits,
            p_layers=layers, max_iterations=max_iterations,
        )