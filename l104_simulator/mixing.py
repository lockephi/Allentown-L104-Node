"""
===============================================================================
L104 SIMULATOR — LAYER 3: MIXING MATRICES (CKM + PMNS)
===============================================================================

Encodes the flavor mixing structure of the Standard Model:
  - CKM matrix (quark mixing) — Cabibbo-Kobayashi-Maskawa
  - PMNS matrix (lepton mixing) — Pontecorvo-Maki-Nakagawa-Sakata
  - Flavor-basis Hamiltonians constructed from E-space mass matrices
  - Jarlskog invariant (CP violation measure)

The mixing matrices rotate MASS eigenstates → FLAVOR eigenstates.
In E-space, this generates off-diagonal couplings in the flavor basis
that represent the "E-cost" of a flavor transition.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .constants import (
    CKM_THETA12, CKM_THETA13, CKM_THETA23, CKM_DELTA_CP,
    PMNS_THETA12, PMNS_THETA13, PMNS_THETA23, PMNS_DELTA_CP,
)
from .lattice import ELattice
from .generations import GenerationStructure


@dataclass
class MixingMatrixInfo:
    """Complete info about a mixing matrix."""
    name: str                      # "CKM" or "PMNS"
    matrix: np.ndarray             # 3×3 complex unitary matrix
    magnitudes: np.ndarray         # |V_ij|
    angles_deg: Dict[str, float]   # θ₁₂, θ₁₃, θ₂₃
    delta_cp: float                # CP phase (radians)
    jarlskog: float                # J = Im(V_us V_cb V*_ub V*_cs)
    unitarity_check: float         # max |V†V - I|


@dataclass
class FlavorHamiltonian:
    """Hamiltonian in the flavor basis, constructed from E-space masses + mixing."""
    sector: str                    # "lepton" or "quark"
    H_mass: np.ndarray             # Diagonal mass-basis matrix (E-values)
    H_flavor: np.ndarray           # U × H_mass × U† (off-diagonal couplings)
    mixing_matrix: np.ndarray      # The U used to rotate
    transition_amplitudes: Dict[str, float]  # off-diagonal |H_ij| values


class MixingMatrices:
    """
    Layer 3: CKM and PMNS mixing matrices with E-space Hamiltonians.

    Constructs the physical mixing matrices from measured angles,
    then rotates the E-space mass matrices into the flavor basis to
    produce Hamiltonians with off-diagonal transition amplitudes.

    Usage:
        mix = MixingMatrices(lattice, generations)
        ckm = mix.ckm()                        # CKM matrix info
        pmns = mix.pmns()                       # PMNS matrix info
        H_lep = mix.lepton_flavor_hamiltonian() # flavor-basis H
        H_qrk = mix.quark_flavor_hamiltonian()  # CKM-rotated H
    """

    def __init__(self, lattice: ELattice, generations: GenerationStructure):
        self.lattice = lattice
        self.gen = generations
        self._ckm: Optional[np.ndarray] = None
        self._pmns: Optional[np.ndarray] = None

    # ─── Parameterized Mixing Matrix Construction ────────────────────────

    @staticmethod
    def _build_mixing_matrix(theta12_deg: float, theta13_deg: float,
                              theta23_deg: float, delta_cp_rad: float) -> np.ndarray:
        """
        Build a 3×3 unitary mixing matrix from angles and CP phase.
        Standard parameterization (PDG convention).
        """
        s12 = math.sin(math.radians(theta12_deg))
        s13 = math.sin(math.radians(theta13_deg))
        s23 = math.sin(math.radians(theta23_deg))
        c12 = math.cos(math.radians(theta12_deg))
        c13 = math.cos(math.radians(theta13_deg))
        c23 = math.cos(math.radians(theta23_deg))

        # CP phase
        exp_id = complex(math.cos(delta_cp_rad), math.sin(delta_cp_rad))
        exp_mid = complex(math.cos(delta_cp_rad), -math.sin(delta_cp_rad))

        U = np.array([
            [c12*c13,                    s12*c13,                   s13*exp_mid],
            [-s12*c23 - c12*s23*s13*exp_id, c12*c23 - s12*s23*s13*exp_id, s23*c13],
            [s12*s23 - c12*c23*s13*exp_id, -c12*s23 - s12*c23*s13*exp_id, c23*c13],
        ], dtype=complex)

        return U

    @staticmethod
    def _jarlskog(U: np.ndarray) -> float:
        """Compute Jarlskog invariant J = Im(V_us V_cb V*_ub V*_cs)."""
        return abs(float(np.imag(U[0, 1] * U[1, 2] * np.conj(U[0, 2]) * np.conj(U[1, 1]))))

    # ─── CKM Matrix ─────────────────────────────────────────────────────

    def ckm_matrix(self) -> np.ndarray:
        """Get the CKM mixing matrix (cached)."""
        if self._ckm is None:
            self._ckm = self._build_mixing_matrix(
                CKM_THETA12, CKM_THETA13, CKM_THETA23, CKM_DELTA_CP
            )
        return self._ckm

    def ckm(self) -> MixingMatrixInfo:
        """Full CKM matrix analysis."""
        V = self.ckm_matrix()
        return MixingMatrixInfo(
            name="CKM",
            matrix=V,
            magnitudes=np.abs(V),
            angles_deg={"θ12": CKM_THETA12, "θ13": CKM_THETA13, "θ23": CKM_THETA23},
            delta_cp=CKM_DELTA_CP,
            jarlskog=self._jarlskog(V),
            unitarity_check=float(np.max(np.abs(V @ V.conj().T - np.eye(3)))),
        )

    # ─── PMNS Matrix ────────────────────────────────────────────────────

    def pmns_matrix(self) -> np.ndarray:
        """Get the PMNS mixing matrix (cached)."""
        if self._pmns is None:
            self._pmns = self._build_mixing_matrix(
                PMNS_THETA12, PMNS_THETA13, PMNS_THETA23, PMNS_DELTA_CP
            )
        return self._pmns

    def pmns(self) -> MixingMatrixInfo:
        """Full PMNS matrix analysis."""
        U = self.pmns_matrix()
        return MixingMatrixInfo(
            name="PMNS",
            matrix=U,
            magnitudes=np.abs(U),
            angles_deg={"θ12": PMNS_THETA12, "θ13": PMNS_THETA13, "θ23": PMNS_THETA23},
            delta_cp=PMNS_DELTA_CP,
            jarlskog=self._jarlskog(U),
            unitarity_check=float(np.max(np.abs(U @ U.conj().T - np.eye(3)))),
        )

    # ─── Flavor-Basis Hamiltonians ───────────────────────────────────────

    def _flavor_hamiltonian(self, E_vector: np.ndarray,
                             U: np.ndarray, sector: str) -> FlavorHamiltonian:
        """
        Construct flavor-basis Hamiltonian: H_flavor = U × diag(E) × U†

        E_vector: diagonal mass-basis E-addresses
        U: mixing matrix (unitary)
        """
        H_mass = np.diag(E_vector.astype(float))
        # Hermitian matrix — do NOT apply np.real() (preserves CP-violating phases)
        H_flavor = U @ H_mass @ U.conj().T

        # Extract transition amplitudes (off-diagonal elements)
        labels_map = {
            "lepton": [("e↔μ", 0, 1), ("e↔τ", 0, 2), ("μ↔τ", 1, 2)],
            "quark":  [("u↔d", 0, 0), ("u↔s", 0, 1), ("u↔b", 0, 2),
                       ("c↔d", 1, 0), ("c↔s", 1, 1), ("c↔b", 1, 2),
                       ("t↔d", 2, 0), ("t↔s", 2, 1), ("t↔b", 2, 2)],
        }

        transitions = {}
        if sector == "lepton":
            for label, i, j in labels_map["lepton"]:
                transitions[label] = abs(H_flavor[i, j])
        else:
            # For quarks, the Hamiltonian is CKM × down_mass × CKM†
            for i in range(3):
                for j in range(3):
                    if i != j:
                        transitions[f"row{i}↔col{j}"] = abs(H_flavor[i, j])

        return FlavorHamiltonian(
            sector=sector,
            H_mass=H_mass,
            H_flavor=H_flavor,
            mixing_matrix=U,
            transition_amplitudes=transitions,
        )

    def lepton_flavor_hamiltonian(self) -> FlavorHamiltonian:
        """
        Lepton flavor Hamiltonian: U_PMNS × diag(E_e, E_μ, E_τ) × U_PMNS†.

        Off-diagonal elements give PMNS-weighted transition "costs" in E-units.
        """
        E_lep = self.gen.lepton_E_vector()
        U = self.pmns_matrix()
        return self._flavor_hamiltonian(E_lep, U, "lepton")

    def quark_flavor_hamiltonian(self) -> FlavorHamiltonian:
        """
        Quark cross-sector Hamiltonian: V_CKM × diag(E_d, E_s, E_b) × V_CKM†.

        Represents effective down-type mass matrix in the up-type flavor basis.
        Off-diagonals encode CKM transition amplitudes in E-units.
        """
        E_dn = self.gen.quark_down_E_vector()
        V = self.ckm_matrix()
        return self._flavor_hamiltonian(E_dn, V, "quark")

    # ─── Weinberg Angle ──────────────────────────────────────────────────

    def weinberg_angle(self) -> Dict[str, float]:
        """
        Electroweak mixing: cos(θ_W) = m_W/m_Z.
        Computed both from exact masses and from E-space.
        """
        E_W = self.lattice.E("m_W")
        E_Z = self.lattice.E("m_Z")
        dE = E_W - E_Z

        # In E-space, the ratio is 2^(ΔE/Q)
        ratio_grid = 2 ** (dE / 416)
        ratio_exact = 80369.2 / 91187.6
        theta_W_grid = math.acos(ratio_grid)
        theta_W_exact = math.acos(ratio_exact)

        return {
            "cos_θW_exact": ratio_exact,
            "cos_θW_grid": ratio_grid,
            "θW_exact_deg": math.degrees(theta_W_exact),
            "θW_grid_deg": math.degrees(theta_W_grid),
            "ΔE_WZ": dE,
            "error_pct": abs(ratio_grid - ratio_exact) / ratio_exact * 100,
        }

    # ─── Summary ─────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Complete mixing layer summary."""
        ckm_info = self.ckm()
        pmns_info = self.pmns()
        H_lep = self.lepton_flavor_hamiltonian()
        H_qrk = self.quark_flavor_hamiltonian()

        return {
            "CKM": {
                "|V|": ckm_info.magnitudes.tolist(),
                "jarlskog": ckm_info.jarlskog,
                "unitarity": ckm_info.unitarity_check,
            },
            "PMNS": {
                "|U|": pmns_info.magnitudes.tolist(),
                "jarlskog": pmns_info.jarlskog,
                "unitarity": pmns_info.unitarity_check,
            },
            "lepton_transitions": H_lep.transition_amplitudes,
            "quark_transitions": H_qrk.transition_amplitudes,
            "weinberg": self.weinberg_angle(),
        }

    def __repr__(self) -> str:
        return "MixingMatrices(CKM + PMNS, flavor Hamiltonians)"
