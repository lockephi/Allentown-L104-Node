"""
L104 God Code Simulator — SimulationResult
═══════════════════════════════════════════════════════════════════════════════

Structured result dataclass for all God Code simulations.
Provides engine-ready payloads for Coherence, Entropy, Math, and ASI subsystems.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .constants import PHI, GOD_CODE


@dataclass
class SimulationResult:
    """
    Structured result from any God Code simulation.

    Designed to produce payloads compatible with:
      - CoherenceSubsystem.ingest_simulation_result(payload)
      - EntropySubsystem.calculate_demon_efficiency(entropy)
      - MathEngine verification methods
      - ASI pipeline scoring dimensions
    """

    name: str
    category: str  # "discovery", "quantum", "advanced", "core"
    passed: bool = True
    elapsed_ms: float = 0.0
    detail: str = ""

    # ── Quantum metrics ──
    fidelity: float = 1.0
    gate_fidelity: float = 1.0
    decoherence_fidelity: float = 1.0
    circuit_depth: int = 0
    num_qubits: int = 0
    noise_variance: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)

    # ── Conservation / Physics metrics ──
    conservation_error: float = 0.0
    entropy_value: float = 0.0
    phase_coherence: float = 1.0
    bloch_vector: Optional[Tuple[float, float, float]] = None
    entanglement_entropy: float = 0.0
    concurrence: float = 0.0
    mutual_information: float = 0.0
    sacred_alignment: float = 0.0

    # ── God Code specific ──
    god_code_measured: float = 0.0
    god_code_error: float = 0.0
    dial_values: Dict[str, int] = field(default_factory=dict)

    # ── VQPU-derived metrics (v4.0) ──
    qfi: float = 0.0                          # Quantum Fisher Information
    purity: float = 1.0                       # Density matrix purity Tr(ρ²)
    trace_dist: float = 0.0                   # Trace distance to reference
    bures_dist: float = 0.0                   # Bures distance to reference
    topo_entropy: float = 0.0                 # Topological entanglement entropy
    decay_rate: float = 0.0                   # Loschmidt echo decay rate
    approx_ratio: float = 0.0                # QAOA approximation ratio
    trotter_error: float = 0.0                # Trotter decomposition error bound

    # ── Superconductivity metrics (v5.0) ──
    cooper_pair_amplitude: float = 0.0         # Average singlet fraction (Cooper pairing)
    sc_order_parameter: float = 0.0            # Δ_SC — superconducting order parameter
    energy_gap_eV: float = 0.0                 # BCS energy gap Δ₀ (eV)
    critical_temperature_K: float = 0.0        # BCS critical temperature T_c (K)
    meissner_fraction: float = 0.0             # Meissner diamagnetic response (0=none, 1=perfect)
    london_depth_nm: float = 0.0               # London penetration depth (nm)
    pairing_symmetry: str = ""                 # SC pairing symmetry type (e.g., "s±")

    # ── Raw data ──
    raw_statevector: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── Engine-ready payload converters ──────────────────────────────────────

    def to_coherence_payload(self) -> Dict[str, Any]:
        """Format for CoherenceSubsystem.ingest_simulation_result()."""
        payload = {
            "total_fidelity": self.fidelity,
            "gate_fidelity": self.gate_fidelity,
            "decoherence_fidelity": self.decoherence_fidelity,
            "circuit_depth": self.circuit_depth,
            "noise_variance": self.noise_variance,
            "probabilities": self.probabilities,
        }
        if self.entropy_value > 0:
            demon_factor = PHI / (GOD_CODE / 416.0)
            payload["demon_efficiency"] = max(0.0, min(1.0,
                demon_factor / (1.0 + self.entropy_value)))
        return payload

    def to_entropy_input(self) -> float:
        """Return entropy value for EntropySubsystem.calculate_demon_efficiency()."""
        if self.entropy_value > 0:
            return self.entropy_value
        return self.noise_variance + (1.0 - self.decoherence_fidelity) * 2.0

    def to_math_verification(self) -> Dict[str, Any]:
        """Format for MathEngine God Code verification."""
        return {
            "god_code_measured": self.god_code_measured,
            "god_code_error": self.god_code_error,
            "conservation_error": self.conservation_error,
            "phase_coherence": self.phase_coherence,
            "sacred_alignment": self.sacred_alignment,
            "dial_values": self.dial_values,
        }

    def to_asi_scoring(self) -> Dict[str, Any]:
        """Format for ASI pipeline scoring dimension input."""
        return {
            "simulation_name": self.name,
            "category": self.category,
            "passed": self.passed,
            "fidelity": self.fidelity,
            "entanglement_entropy": self.entanglement_entropy,
            "concurrence": self.concurrence,
            "phase_coherence": self.phase_coherence,
            "conservation_verified": self.conservation_error < 1e-9,
            "sacred_alignment": self.sacred_alignment,
            "god_code_accuracy": max(0.0, 1.0 - self.god_code_error / GOD_CODE) if GOD_CODE else 0.0,
            # VQPU-derived scoring dimensions (v4.0)
            "qfi": self.qfi,
            "purity": self.purity,
            "topo_entropy": self.topo_entropy,
            "decay_rate": self.decay_rate,
            "approx_ratio": self.approx_ratio,
        }

    def to_vqpu_metrics(self) -> Dict[str, Any]:
        """Format for VQPU bridge integration — all quantum information metrics."""
        return {
            "qfi": self.qfi,
            "purity": self.purity,
            "trace_distance": self.trace_dist,
            "bures_distance": self.bures_dist,
            "topological_entropy": self.topo_entropy,
            "decay_rate": self.decay_rate,
            "approximation_ratio": self.approx_ratio,
            "trotter_error": self.trotter_error,
            "fidelity": self.fidelity,
            "entanglement_entropy": self.entanglement_entropy,
            "phase_coherence": self.phase_coherence,
            "sacred_alignment": self.sacred_alignment,
            # v5.0 Superconductivity metrics
            "cooper_pair_amplitude": self.cooper_pair_amplitude,
            "sc_order_parameter": self.sc_order_parameter,
            "energy_gap_eV": self.energy_gap_eV,
            "critical_temperature_K": self.critical_temperature_K,
            "meissner_fraction": self.meissner_fraction,
        }

    def to_superconductivity_payload(self) -> Dict[str, Any]:
        """Format for superconductivity research database persistence."""
        return {
            "simulation_name": self.name,
            "passed": self.passed,
            "cooper_pair_amplitude": self.cooper_pair_amplitude,
            "sc_order_parameter": self.sc_order_parameter,
            "energy_gap_eV": self.energy_gap_eV,
            "critical_temperature_K": self.critical_temperature_K,
            "meissner_fraction": self.meissner_fraction,
            "london_depth_nm": self.london_depth_nm,
            "pairing_symmetry": self.pairing_symmetry,
            "entanglement_entropy": self.entanglement_entropy,
            "sacred_alignment": self.sacred_alignment,
            "god_code_coupling": self.god_code_measured,
            "num_qubits": self.num_qubits,
            "fidelity": self.fidelity,
        }

    def to_daemon_telemetry(self) -> Dict[str, Any]:
        """v9.0: Format for VQPUDaemonCycler telemetry persistence and state tracking."""
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "fidelity": round(self.fidelity, 6),
            "sacred_alignment": round(self.sacred_alignment, 6),
            "god_code_measured": round(self.god_code_measured, 6),
            "entropy_value": round(self.entropy_value, 6),
            "entanglement_entropy": round(self.entanglement_entropy, 6),
            "num_qubits": self.num_qubits,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "sc_order_parameter": self.sc_order_parameter,
            "cooper_pair_amplitude": self.cooper_pair_amplitude,
            "meissner_fraction": self.meissner_fraction,
            "timestamp": time.time() if hasattr(time, "time") else 0.0,
        }

    def summary(self) -> str:
        """One-line summary."""
        status = "PASS" if self.passed else "FAIL"
        return (f"[{status}] {self.name} ({self.category}) — "
                f"fidelity={self.fidelity:.4f}, entropy={self.entropy_value:.4f}, "
                f"sacred={self.sacred_alignment:.4f}, {self.elapsed_ms:.1f}ms")


__all__ = ["SimulationResult"]
