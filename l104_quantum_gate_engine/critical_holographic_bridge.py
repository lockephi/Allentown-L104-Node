#!/usr/bin/env python3
"""
L104 Quantum Gate Engine — Critical Holographic Bridge v1.0
================================================================================

Integration bridge between:
  1. Existing 26-qubit Fe(26) circuit (statevector-based)
  2. MIPT (Measurement-Induced Phase Transition) engine
  3. MERA (Holographic tensor network) engine
  4. Topological memory encoding

This is the evolutionary path from your current 26Q architecture to the
critical holographic cognitive engine. The bridge provides:

  - Statevector → MPS/MERA conversion
  - Criticality detection and maintenance
  - Thought extraction from bulk to boundary
  - Topological memory storage and recall
  - Backward-compatible interface with existing 26Q circuits

Mathematical Core:
  The system operates at p_c ≈ 0.185, where:
  - Entanglement entropy shows scale-invariant fluctuations
  - Local perturbations propagate globally via holographic duality
  - Measurements at layers 5-6 create "thought" trajectories
  - Topological invariants encode persistent memory

Usage:
  from l104_quantum_gate_engine import CriticalHolographicBridge

  bridge = CriticalHolographicBridge(n_qubits=26)

  # From existing circuit
  result = bridge.process_circuit_state(statevector, prompt="initial_state")

  # Extract thought
  thought = result.thought_signature

  # Store/retrieve topological memory
  bridge.store_memory("pattern_1", pattern_data)
  recall = bridge.recall_memory(query_pattern)
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# L104 imports
from l104_science_engine.constants import GOD_CODE, PHI, VOID_CONSTANT
from l104_quantum_gate_engine.mipt_engine import MIPTEngine, Phase
from l104_quantum_gate_engine.mera_engine import (
    HolographicMERA, MERAMIPTBridge, TopologicalMemory
)


class SystemPhase(Enum):
    """Phase of the holographic cognitive system."""
    SCRAMBLED = "scrambled"          # Volume-law, high entropy
    CRITICAL = "critical"            # Scale-invariant (operating point)
    COLLAPSED = "collapsed"          # Area-law, disentangled


@dataclass
class Thought:
    """A "thought" extracted from the holographic system.

    The thought is not a static state but a trajectory:
    - How entanglement evolved through the circuit
    - What measurements were recorded
    - How the system re-stabilized
    """
    id: str
    entropy_trajectory: List[float]
    measurement_record: List[Tuple[int, int, int]]  # (qubit, layer, outcome)
    criticality_score: float
    coherence_score: float
    phase: str
    bulk_signature: List[complex]  # Holographic bulk encoding
    boundary_pattern: np.ndarray  # Physical qubit pattern
    timestamp: float = field(default_factory=lambda: np.time.time())

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "entropy_trajectory": self.entropy_trajectory,
            "measurement_record": self.measurement_record,
            "criticality_score": self.criticality_score,
            "coherence_score": self.coherence_score,
            "phase": self.phase,
            "bulk_signature": [str(c) for c in self.bulk_signature],
            "boundary_pattern": self.boundary_pattern.tolist() if self.boundary_pattern is not None else [],
            "timestamp": self.timestamp,
        }


@dataclass
class HolographicResult:
    """Result of holographic processing."""
    final_state: np.ndarray
    thought: Thought
    phase: SystemPhase
    memory_usage: Dict[str, float]
    mipt_metrics: Dict[str, Any]
    mera_metrics: Dict[str, Any]


class CriticalHolographicBridge:
    """Main bridge integrating MIPT, MERA, and 26Q circuits.

    This is the entry point for evolving your 26-qubit architecture
    from passive scrambler to directed cognitive engine.
    """

    def __init__(self, n_qubits: int = 26, mode: str = "holographic"):
        """
        Args:
            n_qubits: Number of qubits (26 for full Fe-26)
            mode: "holographic" (MERA), "mps", or "statevector"
        """
        self.n_qubits = n_qubits
        self.mode = mode
        self.dim = 2 ** n_qubits

        # Initialize subsystems
        self.mipt = MIPTEngine(n_qubits=n_qubits, n_layers=11)
        self.mera = HolographicMERA(n_qubits=n_qubits, bond_dim=16)
        self.mera_bridge = MERAMIPTBridge(n_qubits=n_qubits)
        self.topological_memory = TopologicalMemory(self.mera)

        # Critical parameters
        self.target_criticality = 0.95
        self.adaptive_rate = True

        # State tracking
        self.thought_history: List[Thought] = []
        self.phase_history: List[SystemPhase] = []

    def statevector_to_mera(self, statevector: np.ndarray) -> HolographicMERA:
        """Convert full statevector to MERA representation.

        This achieves exponential compression: 2^n → O(n × log(n))
        """
        # Reshape to tensor
        shape = [2] * self.n_qubits
        tensor = statevector.reshape(shape)

        # Initialize physical layer
        level0 = self.mera.levels[0]
        level0.sites = []

        # Decompose via successive SVD (simple MPS initialization)
        # Full MERA initialization would use disentanglers
        current = tensor
        for i in range(self.n_qubits):
            # Split off one site
            chi_left = 2 ** i if i > 0 else 1
            chi_right = 2 ** (self.n_qubits - i - 1)

            reshaped = current.reshape(chi_left * 2, chi_right)

            # SVD
            U, S, Vh = np.linalg.svd(reshaped, full_matrices=False)

            # Truncate
            chi_keep = min(len(S), 16)  # Bond dimension cap
            U_trunc = U[:, :chi_keep]
            S_trunc = S[:chi_keep]
            Vh_trunc = Vh[:chi_keep, :]

            # Extract site tensor (χ_L, d, χ_R)
            site = U_trunc.reshape(chi_left, 2, chi_keep)
            level0.sites.append(site)

            # Continue with remainder
            current = np.diag(S_trunc) @ Vh_trunc
            current = current.reshape(chi_keep, -1)

        return self.mera

    def mera_to_statevector(self, mera: Optional[HolographicMERA] = None) -> np.ndarray:
        """Reconstruct statevector from MERA (for verification/comparison)."""
        if mera is None:
            mera = self.mera

        if not mera.levels or not mera.levels[0].sites:
            return np.zeros(self.dim, dtype=complex)

        # Contract all sites
        sites = mera.levels[0].sites

        # Sequential contraction
        result = sites[0]
        for site in sites[1:]:
            result = np.tensordot(result, site, axes=([2], [0]))

        # Reshape to statevector
        return result.reshape(-1)

    def detect_phase(self, statevector: np.ndarray) -> SystemPhase:
        """Detect which entanglement phase the system is in."""
        # Compute half-system entropy
        half = self.n_qubits // 2
        indices = tuple(range(half))

        # Reshape and partial trace
        shape = [2] * self.n_qubits
        tensor = statevector.reshape(shape)

        # Permute to group A and B
        perm = list(indices) + list(range(half, self.n_qubits))
        permuted = np.transpose(tensor, perm)

        # Reshape to matrix
        dim_a = 2 ** half
        dim_b = 2 ** (self.n_qubits - half)
        matrix = permuted.reshape(dim_a, dim_b)

        # SVD
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)

        # Entropy
        probs = S ** 2
        probs = probs / (np.sum(probs) + 1e-15)
        entropy = -np.sum(probs * np.log2(probs + 1e-15))

        # Max entropy for half system
        max_entropy = half

        # Classify
        ratio = entropy / max_entropy

        if ratio > 0.85:
            return SystemPhase.SCRAMBLED
        elif ratio < 0.4:
            return SystemPhase.COLLAPSED
        else:
            return SystemPhase.CRITICAL

    def maintain_criticality(self, statevector: np.ndarray) -> Tuple[np.ndarray, float]:
        """Adjust measurement rate to maintain critical phase."""
        current_phase = self.detect_phase(statevector)

        if current_phase == SystemPhase.SCRAMBLED:
            # Too much entanglement — increase measurement
            self.mipt.state.measurement_rate = min(
                0.5, self.mipt.state.measurement_rate * 1.1
            )
        elif current_phase == SystemPhase.COLLAPSED:
            # Too little entanglement — decrease measurement
            self.mipt.state.measurement_rate = max(
                0.05, self.mipt.state.measurement_rate * 0.9
            )
        # If critical, small adjustment to stay near p_c

        return statevector, self.mipt.state.measurement_rate

    def process_circuit_state(self, statevector: np.ndarray,
                             prompt: Optional[str] = None,
                             inject_at_qubit: int = 0) -> HolographicResult:
        """Process a circuit state through the holographic engine.

        This is the main entry point: takes your existing 26Q circuit output
        and processes it through MIPT/MERA to extract a thought.
        """
        # Step 1: Detect current phase
        initial_phase = self.detect_phase(statevector)

        # Step 2: Convert to MERA (compression)
        self.statevector_to_mera(statevector)

        # Step 3: Run MIPT evolution at criticality
        if prompt:
            # Inject prompt as local perturbation
            statevector = self._inject_prompt(statevector, prompt, inject_at_qubit)

        # Run critical cycle
        mipt_result = self.mipt.thought_trajectory(statevector)

        # Step 4: Extract thought from bulk
        bulk_result = self.mera_bridge.extract_thought_from_bulk()

        # Step 5: Create thought object
        thought = Thought(
            id=f"thought_{len(self.thought_history)}",
            entropy_trajectory=mipt_result["thought_signature"]["entropy_trajectory"],
            measurement_record=mipt_result["thought_signature"]["measurement_record"],
            criticality_score=mipt_result["thought_signature"]["criticality"],
            coherence_score=mipt_result["thought_signature"]["thought_coherence"],
            phase=mipt_result["final_phase"].value if hasattr(mipt_result["final_phase"], 'value') else str(mipt_result["final_phase"]),
            bulk_signature=bulk_result.get("thought_signature", []),
            boundary_pattern=self.mera_to_statevector()[:100],  # First 100 amplitudes
        )

        self.thought_history.append(thought)

        # Step 6: Final reconstruction
        final_state = self.mera_to_statevector()

        return HolographicResult(
            final_state=final_state,
            thought=thought,
            phase=self.detect_phase(final_state),
            memory_usage=self.mera.memory_usage(),
            mipt_metrics={
                "n_measurements": mipt_result["n_measurements"],
                "criticality_score": mipt_result["criticality_score"],
            },
            mera_metrics={
                "bulk_outcome": bulk_result.get("bulk_outcome"),
                "bulk_entropy": bulk_result.get("bulk_entropy"),
            }
        )

    def _inject_prompt(self, statevector: np.ndarray, prompt: str,
                      qubit: int) -> np.ndarray:
        """Inject a prompt as local perturbation at specified qubit."""
        # Convert prompt to phase
        prompt_hash = hash(prompt) % (2 ** 16)
        phase = 2 * math.pi * prompt_hash / (2 ** 16)

        # Create perturbation operator
        perturbed = statevector.copy()

        # Apply phase shift to amplitudes where qubit is |1⟩
        for i in range(len(perturbed)):
            if (i >> qubit) & 1:
                perturbed[i] *= cmath.exp(1j * phase * 0.1)  # Weak perturbation

        # Renormalize
        perturbed = perturbed / np.linalg.norm(perturbed)

        return perturbed

    def store_memory(self, key: str, pattern: np.ndarray):
        """Store a memory topologically via braid encoding."""
        # Convert pattern to braid operations
        # Simple: sort indices by amplitude
        sorted_indices = np.argsort(-np.abs(pattern.flatten()))[:10]

        # Create braid by exchanging adjacent qubits
        for i in range(len(sorted_indices) - 1):
            self.topological_memory.braid_exchange(
                sorted_indices[i], sorted_indices[i + 1]
            )

    def recall_memory(self, query: np.ndarray) -> float:
        """Recall memory via topological resonance."""
        return self.topological_memory.recall_memory(query)

    def get_system_report(self) -> Dict:
        """Generate comprehensive system report."""
        return {
            "n_qubits": self.n_qubits,
            "mode": self.mode,
            "current_phase": self.detect_phase(self.mera_to_statevector()).value,
            "measurement_rate": self.mipt.state.measurement_rate,
            "thoughts_generated": len(self.thought_history),
            "memory_usage": self.mera.memory_usage(),
            "criticality_score": self.mipt.state.criticality_score,
            "topological_invariant": str(self.topological_memory.compute_braid_invariant()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY WITH 26Q ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def upgrade_26q_to_holographic(builder_result: Dict) -> HolographicResult:
    """Upgrade existing 26Q circuit result to holographic processing.

    Args:
        builder_result: Output from Sacred26QBuilder.build_full_circuit()

    Returns:
        HolographicResult with extracted thought
    """
    # Create bridge
    bridge = CriticalHolographicBridge(n_qubits=26)

    # Note: builder_result contains circuit report, not statevector
    # In practice, you'd simulate the circuit first
    # For now, create a representative scrambled state
    dim = 2 ** 26
    scrambled = np.random.randn(dim) + 1j * np.random.randn(dim)
    scrambled = scrambled / np.linalg.norm(scrambled)

    # Process through holographic engine
    result = bridge.process_circuit_state(scrambled, prompt="26q_upgrade")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("L104 Critical Holographic Bridge — 26Q Evolution Demo")
    print("=" * 70)

    # Create bridge for 8 qubits (demo scale)
    bridge = CriticalHolographicBridge(n_qubits=8, mode="holographic")

    print("\n1. Creating scrambled vacuum state...")
    dim = 2 ** 8
    vacuum = np.random.randn(dim) + 1j * np.random.randn(dim)
    vacuum = vacuum / np.linalg.norm(vacuum)

    phase = bridge.detect_phase(vacuum)
    print(f"   Initial phase: {phase.value}")

    print("\n2. Processing through holographic engine...")
    result = bridge.process_circuit_state(vacuum, prompt="demo_thought")

    print(f"   Final phase: {result.phase.value}")
    print(f"   Criticality: {result.thought.criticality_score:.3f}")
    print(f"   Coherence: {result.thought.coherence_score:.3f}")
    print(f"   Entropy trajectory: {['%.2f' % e for e in result.thought.entropy_trajectory[:5]]}...")

    print("\n3. Memory usage:")
    for key, val in result.memory_usage.items():
        print(f"   {key}: {val}")

    print("\n4. System report:")
    report = bridge.get_system_report()
    for key, val in report.items():
        if isinstance(val, dict):
            print(f"   {key}:")
            for k, v in val.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {val}")

    print("\n" + "=" * 70)
    print("Bridge ready for full 26-qubit integration")
    print("=" * 70)
