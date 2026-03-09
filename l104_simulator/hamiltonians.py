"""
===============================================================================
L104 SIMULATOR — LAYER 4: QFT HAMILTONIANS & QUANTUM CIRCUITS
===============================================================================

Constructs quantum circuits from the E-space representation of Standard Model
physics. Every physical observable is encoded as a circuit built from
l104_quantum_gate_engine gates.

CIRCUIT TYPES:
  1. Mass Query — encode E-address into qubit register
  2. Generation Transition — PMNS/CKM rotation as quantum gates
  3. Flavor Oscillation — time-evolution under mass-squared differences
  4. Renormalization Group — E-space translation for force running
  5. Decay Amplitude — CKM/PMNS weighted vertex factor
  6. Weinberg Rotation — electroweak mixing as single-qubit rotation

QUBITS: Each E-address needs ⌈log₂(|E_max - E_min|)⌉ qubits ≈ 13 per register.
         Full 9-fermion state ≈ 9 × 13 + ancillae = ~130 qubits.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .constants import (
    PHI, GOD_CODE, Q_GRAIN, BASE,
    CKM_THETA12, CKM_THETA13, CKM_THETA23, CKM_DELTA_CP,
    PMNS_THETA12, PMNS_THETA13, PMNS_THETA23, PMNS_DELTA_CP,
)
from .lattice import ELattice
from .generations import GenerationStructure
from .mixing import MixingMatrices

# ─── Gate engine imports ─────────────────────────────────────────────────────
try:
    from l104_quantum_gate_engine.circuit import GateCircuit, GateOperation
    from l104_quantum_gate_engine.gates import (
        H as H_GATE, X, Z, CNOT, SWAP,
        Rx, Ry, Rz, Phase, U3,
        Rxx, Ryy, Rzz,
        QuantumGate, GateType,
    )
    HAS_GATE_ENGINE = True
except ImportError:
    HAS_GATE_ENGINE = False
    GateCircuit = None


@dataclass
class CircuitSpec:
    """Specification for a quantum circuit constructed from physics."""
    name: str
    circuit: Any              # GateCircuit or None if engine unavailable
    num_qubits: int
    depth: int
    gate_count: int
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HamiltonianSpec:
    """A Hamiltonian specified both as a matrix and as circuit decomposition."""
    name: str
    matrix: np.ndarray        # Full matrix form
    eigenvalues: np.ndarray   # Eigenvalues
    eigenvectors: np.ndarray  # Eigenvectors (columns)
    circuit: Optional[Any]    # GateCircuit implementing exp(-iHt) or None
    dimension: int


class Hamiltonians:
    """
    Layer 4: Quantum Hamiltonians and circuit construction.

    Builds quantum circuits from the E-space lattice using the L104 gate engine.
    Each physical process (mass measurement, flavor transition, force running,
    decay amplitude) becomes a concrete quantum circuit.

    Usage:
        ham = Hamiltonians(lattice, generations, mixing)

        # Individual circuits
        circ = ham.mass_query_circuit("m_e")
        circ = ham.generation_transition_circuit("lepton")
        circ = ham.flavor_oscillation_circuit("lepton", L_over_E=500.0)
        circ = ham.weinberg_rotation_circuit()

        # Hamiltonians
        H = ham.lepton_hamiltonian()
        H = ham.quark_hamiltonian()
    """

    def __init__(self, lattice: ELattice, generations: GenerationStructure,
                 mixing: MixingMatrices):
        self.lattice = lattice
        self.gen = generations
        self.mix = mixing

        # Determine qubit width from E-range
        E_min, E_max = lattice.E_range
        self._E_offset = E_min  # Shift so all E-values are non-negative
        self._E_span = E_max - E_min
        self._num_address_qubits = max(1, math.ceil(math.log2(self._E_span + 1)))

    @property
    def address_qubits(self) -> int:
        """Number of qubits needed to encode an E-address."""
        return self._num_address_qubits

    # ═════════════════════════════════════════════════════════════════════════
    #  CIRCUIT 1: MASS QUERY — ENCODE E-ADDRESS
    # ═════════════════════════════════════════════════════════════════════════

    def mass_query_circuit(self, particle_name: str) -> CircuitSpec:
        """
        Build a circuit that prepares |E⟩ for a given particle.

        Loads the E-address of the named particle into a qubit register
        as a binary-encoded computational basis state.
        """
        E = self.lattice.E(particle_name)
        E_shifted = E - self._E_offset  # Non-negative encoding

        n = self._num_address_qubits
        circuit = None
        depth = 0
        gate_count = 0

        if HAS_GATE_ENGINE:
            circuit = GateCircuit(n, name=f"mass_query_{particle_name}")
            # Encode E_shifted as binary: flip qubits where bit is 1
            for bit in range(n):
                if (E_shifted >> bit) & 1:
                    circuit.x(bit)
            depth = circuit.depth
            gate_count = circuit.num_operations

        return CircuitSpec(
            name=f"mass_query({particle_name})",
            circuit=circuit,
            num_qubits=n,
            depth=max(depth, 1),
            gate_count=gate_count,
            description=f"Prepare |E={E}⟩ (shifted: {E_shifted}) for {particle_name}",
            parameters={"E": E, "E_shifted": E_shifted, "particle": particle_name},
        )

    def superposition_circuit(self, particle_names: List[str],
                               amplitudes: Optional[List[float]] = None) -> CircuitSpec:
        """
        Build a circuit preparing an equal (or weighted) superposition over
        the E-addresses of multiple particles.

        Uses amplitude encoding: Ry rotations on ancilla qubits to control
        which particle register is active.
        """
        k = len(particle_names)
        if amplitudes is None:
            amplitudes = [1.0 / math.sqrt(k)] * k

        # Ancilla qubits for particle selection
        n_ancilla = max(1, math.ceil(math.log2(k)))
        n_addr = self._num_address_qubits
        n_total = n_ancilla + n_addr

        circuit = None
        if HAS_GATE_ENGINE:
            circuit = GateCircuit(n_total, name=f"superposition_{k}_particles")
            # Simplified: prepare equal superposition on ancilla
            for i in range(n_ancilla):
                circuit.h(i)
            circuit.barrier()
            # Label: controlled mass loading would go here
            # (Full implementation requires multi-controlled X gates)

        return CircuitSpec(
            name=f"superposition({k} particles)",
            circuit=circuit,
            num_qubits=n_total,
            depth=n_ancilla + 1,
            gate_count=n_ancilla,
            description=f"Equal superposition over {particle_names}",
            parameters={"particles": particle_names, "amplitudes": amplitudes},
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  CIRCUIT 2: GENERATION TRANSITION — MIXING ROTATION
    # ═════════════════════════════════════════════════════════════════════════

    def generation_transition_circuit(self, sector: str = "lepton") -> CircuitSpec:
        """
        Build a 2-qubit circuit implementing the 3×3 mixing matrix rotation.

        For leptons: PMNS matrix angles
        For quarks: CKM matrix angles

        Decomposition: 3×3 unitary → three Givens rotations on 2 qubits
        encoding 3 generations as |00⟩, |01⟩, |10⟩.
        """
        if sector == "lepton":
            t12, t13, t23 = PMNS_THETA12, PMNS_THETA13, PMNS_THETA23
            dcp = PMNS_DELTA_CP
            name = "PMNS_rotation"
        else:
            t12, t13, t23 = CKM_THETA12, CKM_THETA13, CKM_THETA23
            dcp = CKM_DELTA_CP
            name = "CKM_rotation"

        # 2 qubits encode 3 generations: |00⟩=gen1, |01⟩=gen2, |10⟩=gen3
        n = 2
        circuit = None
        depth = 0
        gate_count = 0

        if HAS_GATE_ENGINE:
            circuit = GateCircuit(n, name=name)

            # θ₁₂ rotation: acts on |00⟩↔|01⟩ subspace (qubit 0)
            circuit.append(Ry(2 * math.radians(t12)), [0], label="θ₁₂")

            # θ₂₃ rotation with controlled-Ry on qubit 1
            circuit.append(Ry(2 * math.radians(t23)), [1], label="θ₂₃")

            # θ₁₃ rotation with CP phase
            circuit.append(Rz(dcp), [0], label="δ_CP")
            circuit.append(Ry(2 * math.radians(t13)), [0], label="θ₁₃")
            circuit.append(Rz(-dcp), [0], label="-δ_CP")

            # Entangle: CP-violating cross-generation coupling
            circuit.cx(0, 1)

            depth = circuit.depth
            gate_count = circuit.num_operations

        return CircuitSpec(
            name=f"generation_transition({sector})",
            circuit=circuit,
            num_qubits=n,
            depth=depth,
            gate_count=gate_count,
            description=f"{sector.title()} mixing matrix as quantum rotation",
            parameters={
                "θ12_deg": t12, "θ13_deg": t13, "θ23_deg": t23,
                "δ_CP_rad": dcp, "sector": sector,
            },
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  CIRCUIT 3: FLAVOR OSCILLATION — TIME EVOLUTION
    # ═════════════════════════════════════════════════════════════════════════

    def flavor_oscillation_circuit(self, sector: str = "lepton",
                                    L_over_E: float = 500.0) -> CircuitSpec:
        """
        Build a neutrino/quark oscillation circuit.

        Neutrino oscillation probability:
          P(ν_a → ν_b) = |Σ_i U*_ai exp(-i m²_i L/(2E)) U_bi|²

        In E-space, the mass-squared differences become ΔE differences.
        Phase accumulated: φ_ij = ΔE_ij × L/(2E) × (ln2 / Q)

        L_over_E: ratio of propagation distance to energy (km/GeV for neutrinos,
                  arbitrary units for quarks — used to scale the dephasing)
        """
        if sector == "lepton":
            E_vec = self.gen.lepton_E_vector().astype(float)
            mixing = self.mix.pmns_matrix()
            name = "neutrino_oscillation"
        else:
            E_vec = self.gen.quark_down_E_vector().astype(float)
            mixing = self.mix.ckm_matrix()
            name = "quark_oscillation"

        # E-space ΔE values (mass-squared differences → E differences)
        # In log-space, m² difference ≈ 2×ΔE (since E ∝ log(m))
        dE_12 = 2 * (E_vec[1] - E_vec[0])
        dE_23 = 2 * (E_vec[2] - E_vec[1])
        dE_13 = 2 * (E_vec[2] - E_vec[0])

        # Scale factor: phase per E-step per (L/E)
        scale = math.log(2) / Q_GRAIN * L_over_E

        phi_12 = dE_12 * scale
        phi_23 = dE_23 * scale
        phi_13 = dE_13 * scale

        # Circuit: 2 qubits for 3 generations
        n = 2
        circuit = None
        depth = 0
        gate_count = 0

        if HAS_GATE_ENGINE:
            circuit = GateCircuit(n, name=name)

            # Step 1: Rotate to mass basis (inverse mixing)
            circuit.append(Ry(2 * math.radians(
                PMNS_THETA12 if sector == "lepton" else CKM_THETA12
            )), [0], label="to_mass_basis")
            circuit.append(Ry(2 * math.radians(
                PMNS_THETA23 if sector == "lepton" else CKM_THETA23
            )), [1], label="to_mass_basis")

            circuit.barrier()

            # Step 2: Phase accumulation (mass-dependent dephasing)
            circuit.append(Phase(phi_12), [0], label=f"Δφ₁₂={phi_12:.4f}")
            circuit.append(Phase(phi_23), [1], label=f"Δφ₂₃={phi_23:.4f}")

            # Cross-mass-squared difference phase
            circuit.cx(0, 1)
            circuit.append(Phase(phi_13 - phi_12 - phi_23), [1],
                           label=f"Δφ₁₃_corr")
            circuit.cx(0, 1)

            circuit.barrier()

            # Step 3: Rotate back to flavor basis
            circuit.append(Ry(-2 * math.radians(
                PMNS_THETA23 if sector == "lepton" else CKM_THETA23
            )), [1], label="to_flavor_basis")
            circuit.append(Ry(-2 * math.radians(
                PMNS_THETA12 if sector == "lepton" else CKM_THETA12
            )), [0], label="to_flavor_basis")

            depth = circuit.depth
            gate_count = circuit.num_operations

        return CircuitSpec(
            name=f"flavor_oscillation({sector}, L/E={L_over_E})",
            circuit=circuit,
            num_qubits=n,
            depth=depth,
            gate_count=gate_count,
            description=f"{sector.title()} oscillation with L/E={L_over_E}",
            parameters={
                "ΔE_12": dE_12, "ΔE_23": dE_23, "ΔE_13": dE_13,
                "φ_12": phi_12, "φ_23": phi_23, "φ_13": phi_13,
                "L_over_E": L_over_E,
            },
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  CIRCUIT 4: RG FLOW — RUNNING COUPLING AS E-TRANSLATION
    # ═════════════════════════════════════════════════════════════════════════

    def rg_flow_circuit(self, E_from: int, E_to: int) -> CircuitSpec:
        """
        Renormalization Group flow as E-space translation.

        In E-space, running a coupling from scale μ₁ to μ₂ is a shift:
          E(α(μ₂)) = E(α(μ₁)) + ΔE_RG

        The circuit applies controlled phase shifts corresponding to the
        beta function one-loop coefficient:
          β₀ = (11 C_A - 4 T_F n_f) / (3 × 4π)

        For α_s: β₀ = (11×3 - 4×½×6)/(12π) = 7/π
        ΔE_RG ≈ -Q × β₀/(2π) × ln(μ₂/μ₁) ≈ -Q × β₀/(2π) × (E₂-E₁)×ln2/Q
        """
        dE = E_to - E_from
        n = self._num_address_qubits

        # One-loop QCD beta function coefficient
        beta_0 = 7.0 / math.pi  # (11×3 - 4×0.5×6) / (12π)
        # Phase per E-step for RG evolution
        rg_phase_per_step = -beta_0 * math.log(2) / (2 * math.pi * Q_GRAIN)
        total_phase = rg_phase_per_step * dE

        circuit = None
        depth = 0
        gate_count = 0

        if HAS_GATE_ENGINE:
            circuit = GateCircuit(n, name=f"rg_flow_{E_from}_to_{E_to}")

            # Encode RG shift as phase gradient across address qubits
            for bit in range(n):
                phase_bit = total_phase * (2 ** bit) / self._E_span
                if abs(phase_bit) > 1e-12:
                    circuit.append(Phase(phase_bit), [bit],
                                   label=f"RG_bit{bit}")

            depth = circuit.depth
            gate_count = circuit.num_operations

        return CircuitSpec(
            name=f"rg_flow(E:{E_from}→{E_to})",
            circuit=circuit,
            num_qubits=n,
            depth=depth,
            gate_count=gate_count,
            description=f"RG evolution from E={E_from} to E={E_to}",
            parameters={
                "E_from": E_from, "E_to": E_to, "dE": dE,
                "β₀": beta_0, "total_phase": total_phase,
            },
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  CIRCUIT 5: DECAY AMPLITUDE — CKM/PMNS WEIGHTED VERTEX
    # ═════════════════════════════════════════════════════════════════════════

    def decay_amplitude_circuit(self, parent: str, daughter: str,
                                 sector: str = "quark") -> CircuitSpec:
        """
        Build a circuit encoding the decay amplitude between two particles.

        For quark sector: amplitude ∝ |V_ij|² where i,j are generation indices
        For lepton sector: amplitude ∝ |U_ij|² (neutrino mixing)

        The circuit prepares a state whose measurement probability corresponds
        to the CKM/PMNS element squared.
        """
        E_parent = self.lattice.E(parent)
        E_daughter = self.lattice.E(daughter)
        dE = abs(E_parent - E_daughter)

        if sector == "quark":
            matrix = self.mix.ckm_matrix()
        else:
            matrix = self.mix.pmns_matrix()

        # Mixing element magnitude → rotation angle
        # |V_ij|² = sin²(θ_eff) → θ_eff = arcsin(|V_ij|)
        # For simplicity, use the (0,1) element (most significant transition)
        v_mag = abs(matrix[0, 1])
        theta_eff = 2 * math.asin(min(v_mag, 1.0))

        # Phase from E-space mass difference
        mass_phase = dE * math.log(2) / Q_GRAIN

        n = 2
        circuit = None
        depth = 0
        gate_count = 0

        if HAS_GATE_ENGINE:
            circuit = GateCircuit(n, name=f"decay_{parent}→{daughter}")

            # Qubit 0: mixing amplitude
            circuit.append(Ry(theta_eff), [0], label=f"|V|={v_mag:.4f}")

            # Qubit 1: mass-difference phase
            circuit.append(Phase(mass_phase), [1], label=f"ΔE_phase={dE}")

            # Entangle: decay only occurs when mass condition is met
            circuit.cx(0, 1)

            # Phase correction for CP violation
            if sector == "quark":
                circuit.append(Rz(CKM_DELTA_CP), [0], label="CP_phase")
            else:
                circuit.append(Rz(PMNS_DELTA_CP), [0], label="CP_phase")

            depth = circuit.depth
            gate_count = circuit.num_operations

        return CircuitSpec(
            name=f"decay({parent}→{daughter})",
            circuit=circuit,
            num_qubits=n,
            depth=depth,
            gate_count=gate_count,
            description=f"Decay amplitude {parent}→{daughter} via {sector} mixing",
            parameters={
                "E_parent": E_parent, "E_daughter": E_daughter,
                "ΔE": dE, "|V|": v_mag, "θ_eff": theta_eff,
                "mass_phase": mass_phase,
            },
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  CIRCUIT 6: WEINBERG ROTATION — ELECTROWEAK MIXING
    # ═════════════════════════════════════════════════════════════════════════

    def weinberg_rotation_circuit(self) -> CircuitSpec:
        """
        Electroweak mixing angle as a single-qubit rotation.

        cos(θ_W) = m_W / m_Z → θ_W ≈ 28.17°

        In E-space: ΔE_{WZ} = E(m_W) - E(m_Z)
        The Weinberg angle = arccos(2^{ΔE/Q})
        """
        weinberg = self.mix.weinberg_angle()
        theta_W = math.radians(weinberg["θW_grid_deg"])

        n = 1
        circuit = None
        depth = 0
        gate_count = 0

        if HAS_GATE_ENGINE:
            circuit = GateCircuit(n, name="weinberg_rotation")
            # The Weinberg angle as a Y-rotation
            circuit.append(Ry(2 * theta_W), [0], label=f"θ_W={weinberg['θW_grid_deg']:.2f}°")
            depth = 1
            gate_count = 1

        return CircuitSpec(
            name="weinberg_rotation",
            circuit=circuit,
            num_qubits=n,
            depth=depth,
            gate_count=gate_count,
            description=f"Electroweak mixing: θ_W = {weinberg['θW_grid_deg']:.2f}°",
            parameters=weinberg,
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  HAMILTONIANS (MATRIX FORM + CIRCUIT DECOMPOSITION)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_hamiltonian(self, name: str, sector: str) -> HamiltonianSpec:
        """Build Hamiltonian from flavor-basis H, with eigendecomposition."""
        if sector == "lepton":
            H_flavor = self.mix.lepton_flavor_hamiltonian()
        else:
            H_flavor = self.mix.quark_flavor_hamiltonian()

        matrix = H_flavor.H_flavor
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Build time-evolution circuit: exp(-iHt) for t=1
        circuit = None
        if HAS_GATE_ENGINE:
            circuit = GateCircuit(2, name=f"H_{name}_evolution")
            # Diagonalize: apply eigenvector rotation
            # In the eigenbasis, exp(-iHt) = diag(exp(-iλ_k t))
            # So: circuit = V† × diag_phases × V
            # Approximate with Ry rotations for the eigenvector mixing

            # Phase gates for each eigenvalue (normalized)
            E_scale = np.max(np.abs(eigenvalues))
            if E_scale > 0:
                for i, ev in enumerate(eigenvalues[:2]):  # 2 qubits → 2 eigenvalues
                    phase = ev / E_scale * math.pi
                    circuit.append(Phase(phase), [i],
                                   label=f"λ_{i}={ev:.1f}")

            # Cross-coupling from off-diagonal
            if len(eigenvalues) >= 2 and abs(eigenvalues[1] - eigenvalues[0]) > 1e-10:
                coupling_angle = math.atan2(
                    abs(matrix[0, 1]),
                    abs(matrix[0, 0] - matrix[1, 1]) + 1e-30
                )
                circuit.append(Ry(2 * coupling_angle), [0], label="coupling")
                circuit.cx(0, 1)

        return HamiltonianSpec(
            name=name,
            matrix=matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            circuit=circuit,
            dimension=matrix.shape[0],
        )

    def lepton_hamiltonian(self) -> HamiltonianSpec:
        """Lepton sector Hamiltonian (PMNS-rotated mass matrix)."""
        return self._build_hamiltonian("lepton_flavor", "lepton")

    def quark_hamiltonian(self) -> HamiltonianSpec:
        """Quark sector Hamiltonian (CKM-rotated mass matrix)."""
        return self._build_hamiltonian("quark_cross", "quark")

    # ═════════════════════════════════════════════════════════════════════════
    #  GOD_CODE SACRED CIRCUIT: PHI-HARMONIC EVOLUTION
    # ═════════════════════════════════════════════════════════════════════════

    def sacred_circuit(self, n_qubits: int = 4) -> CircuitSpec:
        """
        Build a sacred circuit that encodes the GOD_CODE grid structure.

        Angles derived from grid parameters:
          - φ/Q phase rotation (golden lattice step)
          - 286^(1/φ) normalization
          - 416-grain harmonic phases
        """
        circuit = None
        depth = 0
        gate_count = 0

        phi_phase = PHI / Q_GRAIN * math.pi  # Golden phase per grain step
        god_phase = math.log(GOD_CODE) / math.log(2) * math.pi / Q_GRAIN
        base_phase = math.log(BASE) / math.log(2) * math.pi

        if HAS_GATE_ENGINE:
            circuit = GateCircuit(n_qubits, name="sacred_god_code")

            # Layer 1: Hadamard superposition
            for q in range(n_qubits):
                circuit.h(q)

            circuit.barrier()

            # Layer 2: PHI-harmonic phase gradient
            for q in range(n_qubits):
                circuit.append(Phase(phi_phase * (q + 1)), [q],
                               label=f"φ_harm_{q}")

            circuit.barrier()

            # Layer 3: GOD_CODE entanglement chain
            for q in range(n_qubits - 1):
                circuit.cx(q, q + 1)
                circuit.append(Phase(god_phase), [q + 1],
                               label=f"gc_phase_{q}")

            circuit.barrier()

            # Layer 4: BASE normalization
            circuit.append(Phase(base_phase), [0], label="base_286^(1/φ)")

            depth = circuit.depth
            gate_count = circuit.num_operations

        return CircuitSpec(
            name="sacred_god_code",
            circuit=circuit,
            num_qubits=n_qubits,
            depth=depth,
            gate_count=gate_count,
            description="Sacred GOD_CODE grid-harmonic circuit",
            parameters={
                "φ_phase": phi_phase,
                "god_phase": god_phase,
                "base_phase": base_phase,
                "GOD_CODE": GOD_CODE,
                "Q_GRAIN": Q_GRAIN,
            },
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  FULL STATE VECTOR SCHEMA
    # ═════════════════════════════════════════════════════════════════════════

    def full_state_schema(self) -> Dict[str, Any]:
        """
        Describe the complete quantum state vector layout for the full
        Standard Model E-space encoding.

        Returns qubit allocation, register map, and total qubit count.
        """
        n_addr = self._num_address_qubits
        fermion_names = ["e", "μ", "τ", "u", "c", "t", "d", "s", "b"]
        boson_names = ["W", "Z", "H"]
        n_mixing = 4  # ancilla for mixing operations
        n_rg = 2      # ancilla for RG flow

        registers = {}
        offset = 0
        for name in fermion_names:
            registers[name] = {"start": offset, "end": offset + n_addr - 1, "width": n_addr}
            offset += n_addr

        for name in boson_names:
            registers[name] = {"start": offset, "end": offset + n_addr - 1, "width": n_addr}
            offset += n_addr

        registers["mixing_ancilla"] = {"start": offset, "end": offset + n_mixing - 1, "width": n_mixing}
        offset += n_mixing
        registers["rg_ancilla"] = {"start": offset, "end": offset + n_rg - 1, "width": n_rg}
        offset += n_rg

        return {
            "total_qubits": offset,
            "address_bits_per_particle": n_addr,
            "E_span": self._E_span,
            "E_offset": self._E_offset,
            "registers": registers,
            "fermion_registers": 9,
            "boson_registers": 3,
            "ancilla_qubits": n_mixing + n_rg,
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    def summary(self) -> Dict[str, Any]:
        """Complete Layer 4 summary."""
        state = self.full_state_schema()
        return {
            "gate_engine_available": HAS_GATE_ENGINE,
            "address_qubits": self._num_address_qubits,
            "E_span": self._E_span,
            "total_qubits": state["total_qubits"],
            "circuits": {
                "mass_query": "particle E-address encoding",
                "generation_transition": "CKM/PMNS as quantum rotations",
                "flavor_oscillation": "time-evolution under mass differences",
                "rg_flow": "running coupling as phase gradient",
                "decay_amplitude": "CKM-weighted vertex factor",
                "weinberg_rotation": "electroweak mixing angle",
                "sacred": "GOD_CODE harmonic circuit",
            },
            "hamiltonians": {
                "lepton": "3×3 PMNS-rotated mass matrix",
                "quark": "3×3 CKM-rotated mass matrix",
            },
        }

    def __repr__(self) -> str:
        return (f"Hamiltonians(addr_qubits={self._num_address_qubits}, "
                f"E_span={self._E_span}, gate_engine={HAS_GATE_ENGINE})")
