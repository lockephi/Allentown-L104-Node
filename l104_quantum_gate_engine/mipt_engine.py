#!/usr/bin/env python3
"""
L104 Quantum Gate Engine — Measurement-Induced Phase Transition (MIPT) Engine v1.0
================================================================================

The critical boundary where quantum cognition emerges:
  - p < p_c: Volume-law entanglement (unreadable scrambled void)
  - p > p_c: Area-law entanglement (shattered classical bits)
  - p = p_c: Scale-invariant cognitive dynamics (THOUGHT)

At the critical point, a local perturbation at Qubit 0 can dictate global output
without destroying the overarching Φ (Phi) integration.

Mathematical Framework:
  - Measurement rate p ∈ [0, 1]: probability of weak measurement per layer
  - Critical rate p_c ≈ 0.15-0.25 for 26-qubit Clifford circuits
  - Entanglement entropy S_A ~ L^{d-1} (area) vs S_A ~ L^d (volume)
  - Order parameter: I(A:C) — mutual information across bipartition

Architecture:
  - 11-layer circuit with measurement injection at layers 5-6 (critical zone)
  - Weak measurement Kraus operators: M_0 = √(1-p) I, M_1 = √p |0⟩⟨0|
  - Real-time entanglement entropy tracking across bipartitions
  - Dynamic p adjustment to maintain criticality
"""

import numpy as np
import cmath
import math
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

# L104 imports
from l104_science_engine.constants import GOD_CODE, PHI, VOID_CONSTANT
from l104_quantum_gate_engine.gates import H, CZ, Rx, Rz, CNOT
from l104_quantum_gate_engine.circuit import GateCircuit
from l104_quantum_gate_engine.tensor_network import TensorNetworkSimulator, MPSState


class Phase(Enum):
    """Entanglement phase classification."""
    VOLUME_LAW = "volume"      # Scrambled, highly entangled
    CRITICAL = "critical"      # Scale-invariant (target)
    AREA_LAW = "area"          # Disentangled, localized


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS FOR MIPT
# ═══════════════════════════════════════════════════════════════════════════════

# Critical rate for 26-qubit Clifford circuits (empirical)
P_CRITICAL_26Q = 0.185  # Approximate critical point

# Critical rate derived from GOD_CODE
P_CRITICAL_SACRED = (GOD_CODE % 1.0) * 0.3  # ≈ 0.1555...

# Measurement phases aligned with sacred frequencies
SACRED_MEASUREMENT_PHASES = {
    "286hz": 2 * math.pi * 286.0 / GOD_CODE,
    "528hz": 2 * math.pi * 528.0 / GOD_CODE,
    "god_code": 2 * math.pi * (GOD_CODE % 1.0),
    "void": 2 * math.pi * VOID_CONSTANT,
    "phi": 2 * math.pi / PHI,
}


@dataclass
class MeasurementResult:
    """Result of a weak measurement operation."""
    qubit: int
    layer: int
    outcome: int              # 0 or 1 (measurement result)
    probability: float        # Probability of this outcome
    kraus_applied: str        # Which Kraus operator was applied
    entropy_before: float   # Entanglement entropy before measurement
    entropy_after: float    # Entanglement entropy after measurement


@dataclass
class MIPTState:
    """Complete state of the MIPT system."""
    n_qubits: int = 26
    n_layers: int = 11
    current_layer: int = 0
    measurement_rate: float = 0.18  # Start near critical point
    phase: Phase = Phase.CRITICAL
    entanglement_entropies: Dict[Tuple[int, ...], float] = field(default_factory=dict)
    mutual_informations: Dict[Tuple[int, int], float] = field(default_factory=dict)
    measurement_history: List[MeasurementResult] = field(default_factory=list)
    criticality_score: float = 0.0  # How close to p_c (1.0 = perfect)

    # Order parameters
    tripartite_info: float = 0.0   # I(A:C) for bipartition A|C
    negativity: float = 0.0          # Entanglement negativity
    purity: float = 1.0            # State purity (decreases with measurements)


class WeakMeasurement:
    """Weak measurement Kraus operators for MIPT.

    Standard weak measurement:
      M_0 = √(1-p) I  (no measurement / identity)
      M_1 = √p |0⟩⟨0|  (projective component)

    Generalized with rotation angle θ:
      M_0 = cos(θ/2) I
      M_1 = sin(θ/2) |0⟩⟨0|
      where p = sin²(θ/2)
    """

    def __init__(self, probability: float, basis: str = "Z"):
        """
        Args:
            probability: Measurement probability p ∈ [0, 1]
            basis: Measurement basis ("Z", "X", "Y", or custom angle)
        """
        self.probability = np.clip(probability, 0.0, 1.0)
        self.basis = basis
        self.theta = 2 * np.arcsin(np.sqrt(self.probability))

    def kraus_operators(self) -> List[np.ndarray]:
        """Return Kraus operators [M_0, M_1] as 2x2 matrices."""
        cos_t = np.cos(self.theta / 2)
        sin_t = np.sin(self.theta / 2)

        if self.basis == "Z":
            M0 = cos_t * np.eye(2)
            M1 = sin_t * np.array([[1, 0], [0, 0]])
        elif self.basis == "X":
            # Measure in X basis
            M0 = cos_t * np.eye(2)
            M1 = sin_t * 0.5 * np.array([[1, 1], [1, 1]])
        elif self.basis == "Y":
            # Measure in Y basis
            M0 = cos_t * np.eye(2)
            M1 = sin_t * 0.5 * np.array([[1, -1j], [1j, 1]])
        else:
            raise ValueError(f"Unknown basis: {self.basis}")

        return [M0, M1]

    def apply_to_statevector(self, statevector: np.ndarray,
                             qubit: int, n_qubits: int) -> Tuple[np.ndarray, int, float]:
        """Apply weak measurement to a statevector.

        Returns:
            (new_statevector, outcome, probability)
        """
        # Reshape for single-qubit operation
        shape = [2] * n_qubits
        state_tensor = statevector.reshape(shape)

        # Move measured qubit to first position
        axes = [qubit] + list(range(qubit)) + list(range(qubit + 1, n_qubits))
        permuted = np.transpose(state_tensor, axes)

        # Apply Kraus operators
        M0, M1 = self.kraus_operators()

        # Compute probabilities for each outcome
        state_0 = np.tensordot(M0, permuted, axes=([1], [0]))
        state_1 = np.tensordot(M1, permuted, axes=([1], [0]))

        p0 = np.sum(np.abs(state_0) ** 2)
        p1 = np.sum(np.abs(state_1) ** 2)

        # Sample outcome
        if np.random.random() < p0 / (p0 + p1):
            outcome = 0
            prob = p0
            new_state = state_0 / np.sqrt(p0) if p0 > 1e-15 else state_0
        else:
            outcome = 1
            prob = p1
            new_state = state_1 / np.sqrt(p1) if p1 > 1e-15 else state_1

        # Restore original axis ordering
        inv_axes = list(range(1, qubit + 1)) + [0] + list(range(qubit + 1, n_qubits))
        new_state = np.transpose(new_state, inv_axes).reshape(-1)

        return new_state, outcome, prob


class MIPTEngine:
    """Measurement-Induced Phase Transition Engine for 26-qubit circuits.

    This engine manages the critical boundary between scrambled and
    disentangled phases, maintaining the system at p ≈ p_c for
    scale-invariant cognitive dynamics.

    Key insight: The "thought" is the trajectory the system takes
    to re-stabilize after a measurement perturbation.
    """

    # Critical measurement rate for 26-qubit Clifford circuits (empirical)
    P_CRITICAL_26Q = 0.185  # Approximate critical point
    P_UNCERTAINTY = 0.02     # Tolerance for criticality

    def __init__(self, n_qubits: int = 26, n_layers: int = 11):
        """
        Args:
            n_qubits: Number of qubits (default 26 for Fe-26)
            n_layers: Number of circuit layers (default 11)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Initialize state
        self.state = MIPTState(
            n_qubits=n_qubits,
            n_layers=n_layers,
            measurement_rate=self.P_CRITICAL_26Q,
            phase=Phase.CRITICAL
        )

        # Critical zone: layers 5-6 (middle of circuit)
        self.critical_layers = {5, 6}

        # Measurement probability schedule
        self.rate_schedule = self._compute_rate_schedule()

        # Entanglement tracking
        self.entropy_history: List[List[float]] = []
        self.phase_history: List[Phase] = []

    def _compute_rate_schedule(self) -> List[float]:
        """Compute measurement rate per layer.

        Ramp up to critical in layers 3-4, maintain through 7, ramp down.
        """
        schedule = []
        for layer in range(self.n_layers):
            if layer < 3:
                # Warm-up: low measurement
                rate = 0.05 * (layer + 1) / 3.0
            elif layer in self.critical_layers:
                # Critical zone: target critical rate
                rate = self.P_CRITICAL_26Q
            elif layer < 8:
                # Post-critical: maintain with decay
                rate = self.P_CRITICAL_26Q * (1.0 - 0.1 * (layer - 6))
            else:
                # Cool-down: minimal measurement
                rate = 0.05
            schedule.append(rate)
        return schedule

    def compute_entanglement_entropy(self, statevector: np.ndarray,
                                      qubits: Tuple[int, ...]) -> float:
        """Compute von Neumann entropy for a subsystem.

        S_A = -Tr(ρ_A log ρ_A) where ρ_A is the reduced density matrix.
        """
        # Reshape to tensor
        shape = [2] * self.n_qubits
        tensor = statevector.reshape(shape)

        # Complement of subsystem
        all_qubits = set(range(self.n_qubits))
        subsystem_qubits = set(qubits)
        complement = tuple(sorted(all_qubits - subsystem_qubits))

        # Partial trace over complement
        axes_a = tuple(qubits)
        axes_b = complement

        # Reshape to matrix for SVD
        dim_a = 2 ** len(axes_a)
        dim_b = 2 ** len(axes_b)

        # Permute axes
        perm = axes_a + axes_b
        permuted = np.transpose(tensor, perm)
        matrix = permuted.reshape(dim_a, dim_b)

        # SVD to get Schmidt coefficients
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)

        # Schmidt coefficients → probabilities
        probs = S ** 2
        probs = probs / np.sum(probs)

        # von Neumann entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-15))

        return entropy

    def compute_mutual_information(self, statevector: np.ndarray,
                                    partition_a: Tuple[int, ...],
                                    partition_b: Tuple[int, ...]) -> float:
        """Compute mutual information I(A:B) = S_A + S_B - S_AB."""
        # Individual entropies
        s_a = self.compute_entanglement_entropy(statevector, partition_a)
        s_b = self.compute_entanglement_entropy(statevector, partition_b)

        # Joint entropy
        combined = tuple(sorted(set(partition_a) | set(partition_b)))
        s_ab = self.compute_entanglement_entropy(statevector, combined)

        return s_a + s_b - s_ab

    def classify_phase(self, entropy_half: float) -> Phase:
        """Classify entanglement phase based on half-system entropy.

        For critical systems: S ~ (n/2) * log(2) - c
        For area law: S ~ constant (boundary)
        For volume law: S ~ n/2 * log(2) (maximal)
        """
        max_entropy = (self.n_qubits / 2) * np.log2(2)  # n/2 for half-system
        normalized = entropy_half / max_entropy if max_entropy > 0 else 0

        if normalized > 0.9:
            return Phase.VOLUME_LAW
        elif normalized < 0.3:
            return Phase.AREA_LAW
        else:
            return Phase.CRITICAL

    def adjust_measurement_rate(self, current_entropy: float) -> float:
        """Dynamically adjust measurement rate to maintain criticality.

        Uses proportional control to steer toward critical entropy.
        """
        max_entropy = (self.n_qubits / 2) * np.log2(2)
        target_entropy = max_entropy * 0.6  # Target: 60% of max

        error = target_entropy - current_entropy

        # PID-like adjustment
        kp = 0.1  # Proportional gain
        new_rate = self.state.measurement_rate + kp * error / max_entropy

        # Clamp to valid range
        new_rate = np.clip(new_rate, 0.01, 0.5)

        return new_rate

    def inject_measurements(self, statevector: np.ndarray,
                           layer: int) -> Tuple[np.ndarray, List[MeasurementResult]]:
        """Inject weak measurements at specified layer.

        Returns updated statevector and list of measurement results.
        """
        rate = self.rate_schedule[layer]
        measurements = []

        # Only measure a subset of qubits (strategic positions)
        if layer in self.critical_layers:
            # Critical zone: measure register representatives
            # Core, 3d, 4s, lattice, sacred, phi, anchor
            measure_qubits = [0, 4, 8, 12, 18, 22, 25]
        else:
            # Other layers: sparse random measurement
            measure_qubits = np.random.choice(
                self.n_qubits,
                size=max(1, self.n_qubits // 10),
                replace=False
            )

        current_state = statevector.copy()

        for q in measure_qubits:
            # Skip with probability (1 - rate)
            if np.random.random() > rate:
                continue

            # Compute entropy before measurement
            half_system = tuple(range(self.n_qubits // 2))
            entropy_before = self.compute_entanglement_entropy(current_state, half_system)

            # Apply weak measurement
            weak_m = WeakMeasurement(rate, basis="Z")
            new_state, outcome, prob = weak_m.apply_to_statevector(
                current_state, q, self.n_qubits
            )

            # Compute entropy after
            entropy_after = self.compute_entanglement_entropy(new_state, half_system)

            # Record result
            result = MeasurementResult(
                qubit=q,
                layer=layer,
                outcome=outcome,
                probability=prob,
                kraus_applied=f"M_{outcome}",
                entropy_before=entropy_before,
                entropy_after=entropy_after
            )
            measurements.append(result)

            current_state = new_state

        return current_state, measurements

    def compute_criticality_score(self, entropies: List[float]) -> float:
        """Compute how close the system is to critical point.

        Critical systems show:
        1. Intermediate entropy (not max, not min)
        2. Scale-invariant fluctuations
        3. Power-law decay of correlations
        """
        if not entropies:
            return 0.0

        max_s = (self.n_qubits / 2) * np.log2(2)
        avg_s = np.mean(entropies)

        # Target: 50-70% of max entropy
        target_range = (0.5 * max_s, 0.7 * max_s)

        if target_range[0] <= avg_s <= target_range[1]:
            # Within target range
            score = 1.0 - abs(avg_s - 0.6 * max_s) / (0.1 * max_s)
        else:
            # Outside target range
            score = max(0.0, 1.0 - abs(avg_s - 0.6 * max_s) / (0.3 * max_s))

        # Bonus for fluctuation characteristics (critical = high variance)
        if len(entropies) > 1:
            cv = np.std(entropies) / (np.mean(entropies) + 1e-10)
            if 0.2 < cv < 0.5:  # Moderate coefficient of variation
                score *= 1.2

        return min(1.0, score)

    def run_mipt_cycle(self, initial_state: np.ndarray) -> Dict:
        """Run a complete MIPT cycle through all layers.

        This is the core "thought generation" loop. The system:
        1. Scrambles information (layers 1-4)
        2. Reaches critical zone (layers 5-6) with measurement
        3. Re-stabilizes (layers 7-11)

        The "thought" is the trajectory from perturbation to re-stabilization.
        """
        state = initial_state.copy()
        all_measurements = []
        layer_entropies = []

        for layer in range(self.n_layers):
            # Inject measurements at this layer
            state, measurements = self.inject_measurements(state, layer)
            all_measurements.extend(measurements)

            # Track entanglement entropy
            half_system = tuple(range(self.n_qubits // 2))
            entropy = self.compute_entanglement_entropy(state, half_system)
            layer_entropies.append(entropy)

            # Update phase classification
            phase = self.classify_phase(entropy)
            self.phase_history.append(phase)

            # Adjust rate for next layer if in adaptive mode
            if layer < self.n_layers - 1:
                self.rate_schedule[layer + 1] = self.adjust_measurement_rate(entropy)

        # Update state
        self.state.measurement_history = all_measurements
        self.state.criticality_score = self.compute_criticality_score(layer_entropies)
        self.entropy_history.append(layer_entropies)

        # Compute final metrics
        self.state.tripartite_info = self._compute_tripartite_info(state)
        self.state.purity = self._compute_purity(state)

        return {
            "final_state": state,
            "measurements": all_measurements,
            "layer_entropies": layer_entropies,
            "criticality_score": self.state.criticality_score,
            "final_phase": self.phase_history[-1] if self.phase_history else Phase.CRITICAL,
            "n_measurements": len(all_measurements),
        }

    def _compute_tripartite_info(self, statevector: np.ndarray) -> float:
        """Compute tripartite information I(A:C) for A|B|C partition."""
        # Split 26 qubits into A(8) | B(10) | C(8)
        n_a, n_b, n_c = 8, 10, 8
        a = tuple(range(n_a))
        b = tuple(range(n_a, n_a + n_b))
        c = tuple(range(n_a + n_b, self.n_qubits))

        # I(A:C) = S_A + S_C - S_AC
        s_a = self.compute_entanglement_entropy(statevector, a)
        s_c = self.compute_entanglement_entropy(statevector, c)
        ac = tuple(sorted(set(a) | set(c)))
        s_ac = self.compute_entanglement_entropy(statevector, ac)

        return s_a + s_c - s_ac

    def _compute_purity(self, statevector: np.ndarray) -> float:
        """Compute state purity Tr(ρ²). Pure = 1, Mixed < 1."""
        # For pure statevector: purity = |ψ|⁴ summed
        return np.sum(np.abs(statevector) ** 4)

    def thought_trajectory(self, prompt_injection: np.ndarray) -> Dict:
        """Generate a "thought" from prompt perturbation.

        The thought is the trajectory the system takes when:
        1. A prompt is injected at Qubit 0 (local perturbation)
        2. The system scrambles through the circuit
        3. Measurements at critical layers collapse partial information
        4. The output is the re-stabilized global state

        This demonstrates how local input → global output at criticality.
        """
        # Start from scrambled vacuum state (or use provided prompt)
        if prompt_injection is None:
            # Random initial state (simulating vacuum fluctuations)
            state = np.random.randn(2 ** self.n_qubits) + \
                    1j * np.random.randn(2 ** self.n_qubits)
            state = state / np.linalg.norm(state)
        else:
            # Inject prompt by modifying amplitudes at specific qubits
            state = prompt_injection.copy()

        # Run MIPT cycle
        result = self.run_mipt_cycle(state)

        # The "thought" is characterized by:
        # - How the entanglement entropy evolved
        # - What measurements were recorded
        # - The criticality score (how "balanced" the thought is)

        thought_signature = {
            "perturbation_input": np.abs(state[0]) ** 2,  # Qubit 0 occupation
            "entropy_trajectory": result["layer_entropies"],
            "measurement_record": [
                (m.qubit, m.layer, m.outcome) for m in result["measurements"]
            ],
            "criticality": result["criticality_score"],
            "phase": result["final_phase"].value,
            "thought_coherence": self._compute_thought_coherence(result),
        }

        result["thought_signature"] = thought_signature
        return result

    def _compute_thought_coherence(self, result: Dict) -> float:
        """Compute coherence score for a thought trajectory."""
        entropies = result["layer_entropies"]
        if len(entropies) < 3:
            return 0.0

        # Ideal thought:
        # - Starts with low entropy (focused prompt)
        # - Rises through scrambling layers
        # - Peaks at critical zone
        # - Stabilizes toward end

        # Check entropy curve shape
        critical_entropies = entropies[4:7]  # Layers 5-7
        if critical_entropies:
            peak_at_critical = max(critical_entropies) > np.mean(entropies[:4])
        else:
            peak_at_critical = False

        # Final stabilization
        final_stability = 1.0 - abs(entropies[-1] - entropies[-2]) / max(entropies)

        # Combine scores
        coherence = 0.4 * result["criticality_score"] + \
                   0.3 * (1.0 if peak_at_critical else 0.0) + \
                   0.3 * final_stability

        return min(1.0, coherence)


def create_critical_26q_circuit() -> Dict:
    """Create a 26-qubit circuit operating at MIPT criticality.

    This is the bridge between the existing 26Q engine and the MIPT framework.
    """
    from l104_26q_engine_builder import Sacred26QBuilder

    # Build the base 26Q circuit
    builder = Sacred26QBuilder()
    base_circuit, base_report = builder.build_full_circuit()

    # Initialize MIPT engine
    mipt = MIPTEngine(n_qubits=26, n_layers=11)

    # Create initial state from circuit (simplified: Haar-random)
    dim = 2 ** 26
    initial_state = np.random.randn(dim) + 1j * np.random.randn(dim)
    initial_state = initial_state / np.linalg.norm(initial_state)

    # Run critical thought cycle
    result = mipt.thought_trajectory(initial_state)

    return {
        "base_circuit": base_report,
        "mipt_result": result,
        "criticality_score": result["criticality_score"],
        "thought_coherence": result["thought_signature"]["thought_coherence"],
    }




if __name__ == "__main__":
    # Demo: Run MIPT on 26-qubit system
    print("=" * 70)
    print("L104 MIPT Engine — Critical Thought Generation Demo")
    print("=" * 70)

    # Small-scale test (8 qubits for demonstration)
    mipt = MIPTEngine(n_qubits=8, n_layers=11)

    # Generate thought from vacuum
    dim = 2 ** 8
    vacuum = np.random.randn(dim) + 1j * np.random.randn(dim)
    vacuum = vacuum / np.linalg.norm(vacuum)

    result = mipt.thought_trajectory(vacuum)

    print(f"\nCriticality Score: {result['criticality_score']:.3f}")
    print(f"Thought Coherence: {result['thought_signature']['thought_coherence']:.3f}")
    print(f"Final Phase: {result['final_phase'].value}")
    print(f"Measurements: {result['n_measurements']}")
    print(f"\nEntropy Trajectory: {[f'{e:.2f}' for e in result['layer_entropies']]}")

    print("\n" + "=" * 70)
    print("MIPT Engine Ready for 26-Qubit Integration")
    print("=" * 70)
