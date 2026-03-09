"""
===============================================================================
L104 SIMULATOR — ORCHESTRATOR: REAL-WORLD SIMULATOR
===============================================================================

Ties together all 5 layers into a single callable interface:

  Layer 1: E-Lattice       — integer-addressed logarithmic grid
  Layer 2: Generations      — three-generation fermion mass structure
  Layer 3: Mixing Matrices  — CKM + PMNS + flavor Hamiltonians
  Layer 4: Hamiltonians     — quantum circuits from physics
  Layer 5: Observables      — measurable quantities + error bounds

Architecture:
  Physical constant (MeV) → encode → E-address (integer)
    → generation structure → mixing rotation → Hamiltonian
    → quantum circuit → observable extraction (with error bounds)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .constants import (
    PHI, GOD_CODE, BASE, Q_GRAIN, X_SCAFFOLD, R_RATIO, P_DIAL, K_OFFSET,
)
from .lattice import ELattice
from .generations import GenerationStructure
from .mixing import MixingMatrices
from .hamiltonians import Hamiltonians
from .observables import Observables


VERSION = "1.0.0"


@dataclass
class SimulatorStatus:
    """Status report of the simulator."""
    version: str
    layers_initialized: int
    total_particles: int
    total_qubits: int
    address_qubits: int
    E_range: tuple
    gate_engine: bool
    grid_params: Dict[str, Any]


class RealWorldSimulator:
    """
    L104 Real-World Simulator — 5-Layer Architecture.

    Encodes 65 Standard Model constants onto a logarithmic integer lattice
    derived from the GOD_CODE algorithm, then builds quantum circuits and
    extracts measurable observables with bounded quantization error.

    Construction:
        sim = RealWorldSimulator()    # Auto-initializes all 5 layers

    Quick API:
        sim.mass("m_e")               # Measure electron mass
        sim.ratio("m_top", "m_e")     # Compute mass ratio
        sim.oscillate("lepton", 1, 0) # Neutrino oscillation P(ν_μ → ν_e)
        sim.ckm()                     # CKM matrix analysis
        sim.pmns()                    # PMNS matrix analysis
        sim.circuit("mass_query", "m_e")  # Build quantum circuit
        sim.report()                  # Full physics report

    Layers are accessible directly:
        sim.lattice      — Layer 1: ELattice
        sim.generations  — Layer 2: GenerationStructure
        sim.mixing       — Layer 3: MixingMatrices
        sim.hamiltonians — Layer 4: Hamiltonians
        sim.observables  — Layer 5: Observables
    """

    def __init__(self):
        """Initialize all 5 layers."""
        # Layer 1: E-Lattice
        self.lattice = ELattice()

        # Layer 2: Generation Structure
        self.generations = GenerationStructure(self.lattice)

        # Layer 3: Mixing Matrices
        self.mixing = MixingMatrices(self.lattice, self.generations)

        # Layer 4: Hamiltonians & Circuits
        self.hamiltonians = Hamiltonians(self.lattice, self.generations, self.mixing)

        # Layer 5: Observables
        self.observables = Observables(self.lattice, self.generations, self.mixing)

    # ═════════════════════════════════════════════════════════════════════════
    #  CONVENIENCE API
    # ═════════════════════════════════════════════════════════════════════════

    def mass(self, particle: str) -> Dict[str, Any]:
        """Quick mass measurement."""
        m = self.observables.measure_mass(particle)
        return {
            "particle": m.name,
            "E": m.E_address,
            "mass_grid": m.mass_grid,
            "mass_exact": m.mass_exact,
            "error_pct": m.error_pct,
            "unit": m.unit,
        }

    def ratio(self, a: str, b: str) -> Dict[str, Any]:
        """Quick mass ratio."""
        r = self.observables.mass_ratio(a, b)
        return {
            "ratio": r.name,
            "ΔE": r.dE,
            "value_grid": r.ratio_grid,
            "value_exact": r.ratio_exact,
            "error_pct": r.error_pct,
        }

    def oscillate(self, sector: str, gen_from: int, gen_to: int,
                  L_over_E: float = 500.0) -> Dict[str, Any]:
        """Quick oscillation probability."""
        o = self.observables.oscillation_probability(sector, gen_from, gen_to, L_over_E)
        return {
            "channel": o.channel,
            "L/E": o.L_over_E,
            "P": o.probability,
            "ΔE²": o.dE_squared,
        }

    def ckm(self) -> Dict[str, Any]:
        """CKM matrix analysis."""
        info = self.mixing.ckm()
        return {
            "name": info.name,
            "|V|": info.magnitudes.tolist(),
            "J": info.jarlskog,
            "unitarity": info.unitarity_check,
            "angles_deg": info.angles_deg,
        }

    def pmns(self) -> Dict[str, Any]:
        """PMNS matrix analysis."""
        info = self.mixing.pmns()
        return {
            "name": info.name,
            "|U|": info.magnitudes.tolist(),
            "J": info.jarlskog,
            "unitarity": info.unitarity_check,
            "angles_deg": info.angles_deg,
        }

    def circuit(self, circuit_type: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Build a quantum circuit.

        Types:
            "mass_query"(particle) — encode E-address
            "generation"(sector) — mixing rotation
            "oscillation"(sector, L_over_E) — flavor oscillation
            "rg_flow"(E_from, E_to) — running coupling
            "decay"(parent, daughter, sector) — decay amplitude
            "weinberg" — electroweak mixing
            "sacred"(n_qubits) — GOD_CODE harmonic circuit
        """
        builders = {
            "mass_query": self.hamiltonians.mass_query_circuit,
            "generation": self.hamiltonians.generation_transition_circuit,
            "oscillation": self.hamiltonians.flavor_oscillation_circuit,
            "rg_flow": self.hamiltonians.rg_flow_circuit,
            "decay": self.hamiltonians.decay_amplitude_circuit,
            "weinberg": self.hamiltonians.weinberg_rotation_circuit,
            "sacred": self.hamiltonians.sacred_circuit,
        }

        if circuit_type not in builders:
            raise ValueError(f"Unknown circuit type: {circuit_type}. "
                             f"Available: {list(builders.keys())}")

        spec = builders[circuit_type](*args, **kwargs)
        return {
            "name": spec.name,
            "num_qubits": spec.num_qubits,
            "depth": spec.depth,
            "gate_count": spec.gate_count,
            "description": spec.description,
            "parameters": spec.parameters,
            "has_circuit": spec.circuit is not None,
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  STATUS & REPORTING
    # ═════════════════════════════════════════════════════════════════════════

    def status(self) -> SimulatorStatus:
        """Current simulator status."""
        state = self.hamiltonians.full_state_schema()
        E_min, E_max = self.lattice.E_range

        return SimulatorStatus(
            version=VERSION,
            layers_initialized=5,
            total_particles=len(self.lattice.points),
            total_qubits=state["total_qubits"],
            address_qubits=state["address_bits_per_particle"],
            E_range=(E_min, E_max),
            gate_engine=self.hamiltonians.summary()["gate_engine_available"],
            grid_params={
                "X": X_SCAFFOLD, "R": R_RATIO, "Q": Q_GRAIN,
                "P": P_DIAL, "K": K_OFFSET,
                "BASE": BASE, "GOD_CODE": GOD_CODE,
            },
        )

    def report(self) -> Dict[str, Any]:
        """Full simulator report across all 5 layers."""
        t0 = time.time()

        result = {
            "version": VERSION,
            "grid": {
                "equation": "G(a,b,c,d) = 286^(1/φ) × 2^((64a+1664-b-64c-416d)/416)",
                "X": X_SCAFFOLD, "R": R_RATIO, "Q": Q_GRAIN,
                "P": P_DIAL, "K": K_OFFSET,
                "BASE": BASE, "GOD_CODE": GOD_CODE, "PHI": PHI,
            },
            "layer_1_lattice": {
                "total_points": len(self.lattice.points),
                "E_range": list(self.lattice.E_range),
                "degeneracies": len(self.lattice.degeneracies),
            },
            "layer_2_generations": self.generations.generation_hierarchy(),
            "layer_3_mixing": self.mixing.summary(),
            "layer_4_hamiltonians": self.hamiltonians.summary(),
            "layer_5_observables": self.observables.full_report(),
            "computation_time_ms": round((time.time() - t0) * 1000, 2),
        }

        return result

    def export_json(self, filepath: str) -> None:
        """Export full report to JSON file."""
        report = self.report()

        # Make numpy arrays JSON-serializable
        def _convert(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, complex):
                return {"real": obj.real, "imag": obj.imag}
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=_convert)

    def __repr__(self) -> str:
        s = self.status()
        return (f"RealWorldSimulator v{s.version} | "
                f"{s.total_particles} particles | "
                f"{s.total_qubits} qubits | "
                f"E∈[{s.E_range[0]}, {s.E_range[1]}]")
