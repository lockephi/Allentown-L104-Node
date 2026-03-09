"""
===============================================================================
L104 SIMULATOR — QUANTUM TRAJECTORY SIMULATION (MONTE CARLO WAVEFUNCTION)
===============================================================================

Monte Carlo Wavefunction (MCWF) method for simulating open quantum systems.
Instead of evolving the full density matrix (O(4^n)), evolves individual
state vectors and stochastically applies quantum jumps.

For N trajectories, the density matrix is reconstructed as:
  ρ ≈ (1/N) Σ |ψ_k⟩⟨ψ_k|

Each trajectory:
  1. Evolve |ψ⟩ under non-Hermitian effective Hamiltonian H_eff
  2. At each step, compute jump probability p_jump
  3. If random < p_jump: apply jump operator L_k and renormalize
  4. Else: renormalize the no-jump evolution

This gives O(N × 2^n) scaling instead of O(4^n) for density matrix,
making it practical for larger qubit counts with noise.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from .simulator import Simulator, QuantumCircuit, SimulationResult


class TrajectorySimulator:
    """Monte Carlo Wavefunction simulator for noisy circuits.

    Runs N independent trajectory simulations in parallel, each evolving
    a pure state with stochastic quantum jumps representing decoherence.

    Usage:
        traj = TrajectorySimulator(
            noise_model={'depolarizing': 0.01, 'amplitude_damping': 0.005},
            n_trajectories=100
        )
        qc = QuantumCircuit(6, 'noisy_bell')
        qc.h(0).cx(0, 1).cx(1, 2)
        result = traj.run(qc)
    """

    def __init__(self, noise_model: Dict[str, float],
                 n_trajectories: int = 100,
                 n_workers: int = 4):
        self.noise_model = noise_model
        self.n_trajectories = n_trajectories
        self.n_workers = n_workers
        self._clean_sim = Simulator()  # Noiseless simulator for gate application

    def run(self, circuit: QuantumCircuit,
            seed: Optional[int] = None) -> Dict[str, Any]:
        """Run Monte Carlo wavefunction simulation.

        Returns:
            Dict with averaged probabilities, purity, entropy, trajectory stats
        """
        t0 = time.time()
        n = circuit.n_qubits
        rng = np.random.default_rng(seed)

        # Generate seeds for each trajectory
        seeds = rng.integers(0, 2**31, size=self.n_trajectories)

        # Run trajectories in parallel
        def run_single(traj_seed):
            return self._single_trajectory(circuit, int(traj_seed))

        if self.n_trajectories <= 4:
            trajectories = [run_single(s) for s in seeds]
        else:
            with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                trajectories = list(pool.map(run_single, seeds))

        elapsed = (time.time() - t0) * 1000

        # Average the density matrices
        dim = 2 ** n
        rho_avg = np.zeros((dim, dim), dtype=complex)
        jump_counts = []
        for traj in trajectories:
            psi = traj["statevector"]
            rho_avg += np.outer(psi, psi.conj())
            jump_counts.append(traj["n_jumps"])
        rho_avg /= self.n_trajectories

        # Extract statistics
        probs_arr = np.real(np.diag(rho_avg))
        probs = {format(i, f'0{n}b'): float(p)
                 for i, p in enumerate(probs_arr) if p > 1e-15}

        purity = float(np.real(np.trace(rho_avg @ rho_avg)))

        eigvals = np.linalg.eigvalsh(rho_avg)
        eigvals = eigvals[eigvals > 1e-15]
        entropy = float(-np.sum(eigvals * np.log2(eigvals + 1e-30)))

        return {
            "probabilities": probs,
            "density_matrix": rho_avg,
            "purity": purity,
            "von_neumann_entropy": entropy,
            "n_trajectories": self.n_trajectories,
            "n_qubits": n,
            "gate_count": circuit.gate_count,
            "mean_jumps": float(np.mean(jump_counts)),
            "max_jumps": int(np.max(jump_counts)),
            "execution_time_ms": elapsed,
            "noise_model": self.noise_model,
        }

    def _single_trajectory(self, circuit: QuantumCircuit,
                           seed: int) -> Dict[str, Any]:
        """Run a single trajectory with stochastic quantum jumps."""
        rng = np.random.default_rng(seed)
        n = circuit.n_qubits
        dim = 2 ** n

        # Initialize |0...0⟩
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        n_jumps = 0

        for gate_rec in circuit.gates:
            # Apply gate (noiseless)
            state = self._clean_sim._apply_gate(
                state, gate_rec.matrix, gate_rec.qubits, n
            )

            # Stochastic jump check for each noise channel
            for qubit in gate_rec.qubits:
                # Amplitude damping jump
                p_amp = self.noise_model.get("amplitude_damping", 0.0)
                if p_amp > 0:
                    # Jump operator: L = sqrt(gamma) |0⟩⟨1|
                    # Jump probability: p = gamma × ⟨1|ρ|1⟩ on target qubit
                    p1 = self._qubit_prob(state, qubit, n, value=1)
                    p_jump = p_amp * p1
                    if rng.random() < p_jump:
                        # Apply jump: project qubit to |0⟩
                        state = self._project_qubit(state, qubit, n, target=0)
                        norm = np.linalg.norm(state)
                        if norm > 1e-30:
                            state /= norm
                        n_jumps += 1
                    else:
                        # No-jump: apply sqrt(1-gamma) damping on |1⟩ component
                        state = self._damp_qubit(state, qubit, n,
                                                 math.sqrt(1 - p_amp))
                        norm = np.linalg.norm(state)
                        if norm > 1e-30:
                            state /= norm

                # Depolarizing channel (as random Pauli jump)
                p_dep = self.noise_model.get("depolarizing", 0.0)
                if p_dep > 0:
                    r = rng.random()
                    if r < p_dep / 4:
                        # X jump
                        state = self._apply_pauli(state, qubit, n, 'X')
                        n_jumps += 1
                    elif r < p_dep / 2:
                        # Y jump
                        state = self._apply_pauli(state, qubit, n, 'Y')
                        n_jumps += 1
                    elif r < 3 * p_dep / 4:
                        # Z jump
                        state = self._apply_pauli(state, qubit, n, 'Z')
                        n_jumps += 1
                    # else: no jump (identity)

                # Phase damping (dephasing jump)
                p_phase = self.noise_model.get("phase_damping", 0.0)
                if p_phase > 0 and rng.random() < p_phase:
                    state = self._apply_pauli(state, qubit, n, 'Z')
                    n_jumps += 1

        # Final normalization
        norm = np.linalg.norm(state)
        if norm > 1e-30:
            state /= norm

        return {
            "statevector": state,
            "n_jumps": n_jumps,
        }

    @staticmethod
    def _qubit_prob(state: np.ndarray, qubit: int, n: int,
                    value: int = 1) -> float:
        """Marginal probability of a single qubit being |value⟩."""
        probs = np.abs(state) ** 2
        total = 0.0
        for i in range(len(probs)):
            bit = (i >> (n - qubit - 1)) & 1
            if bit == value:
                total += probs[i]
        return total

    @staticmethod
    def _project_qubit(state: np.ndarray, qubit: int, n: int,
                       target: int) -> np.ndarray:
        """Project qubit to |target⟩ by zeroing the other component."""
        result = state.copy()
        for i in range(len(result)):
            bit = (i >> (n - qubit - 1)) & 1
            if bit != target:
                result[i] = 0.0
        return result

    @staticmethod
    def _damp_qubit(state: np.ndarray, qubit: int, n: int,
                    factor: float) -> np.ndarray:
        """Multiply |1⟩ component of qubit by factor."""
        result = state.copy()
        shape = (2 ** qubit, 2, 2 ** (n - qubit - 1))
        r = result.reshape(shape)
        r[:, 1, :] *= factor
        return r.reshape(-1)

    @staticmethod
    def _apply_pauli(state: np.ndarray, qubit: int, n: int,
                     pauli: str) -> np.ndarray:
        """Apply Pauli X, Y, or Z to a single qubit."""
        result = state.copy()
        shape = (2 ** qubit, 2, 2 ** (n - qubit - 1))
        r = result.reshape(shape)
        if pauli == 'X':
            r[:, 0, :], r[:, 1, :] = r[:, 1, :].copy(), r[:, 0, :].copy()
        elif pauli == 'Y':
            temp0 = r[:, 0, :].copy()
            temp1 = r[:, 1, :].copy()
            r[:, 0, :] = -1j * temp1
            r[:, 1, :] = 1j * temp0
        elif pauli == 'Z':
            r[:, 1, :] *= -1
        return r.reshape(-1)

    def __repr__(self) -> str:
        return (f"TrajectorySimulator(n_traj={self.n_trajectories}, "
                f"noise={self.noise_model})")
