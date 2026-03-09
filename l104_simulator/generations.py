"""
===============================================================================
L104 SIMULATOR — LAYER 2: GENERATION STRUCTURE
===============================================================================

Encodes the 3-generation structure of the Standard Model in E-space:
  - Lepton mass matrix (e, μ, τ)
  - Up-type quark mass matrix (u, c, t)
  - Down-type quark mass matrix (d, s, b)
  - Generation gap analysis and Koide formula verification
  - Full 9×9 fermion transition matrix

All matrices work in E-space (integers) where mass ratios = ΔE values.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .lattice import ELattice, ParticleType


@dataclass
class GenerationGap:
    """Gap between two consecutive generations in E-space."""
    gen_low: int
    gen_high: int
    particle_low: str
    particle_high: str
    delta_E: int           # E(gen_high) - E(gen_low)
    mass_ratio: float      # m_high / m_low
    octaves: float         # ΔE / Q (how many doublings)


@dataclass
class KoideResult:
    """Result of Koide formula check."""
    exact_value: float     # Q = (Σm)/(Σ√m)² from exact masses
    grid_value: float      # Q from grid-snapped masses
    target: float          # 2/3
    deviation_exact: float # |exact - 2/3|
    deviation_grid: float  # |grid - 2/3|
    grid_preserves: bool   # grid deviation < 2× exact deviation


class GenerationStructure:
    """
    Layer 2: Three-generation fermion mass structure in E-space.

    Constructs integer-valued mass matrices for all fermion sectors,
    computes generation gaps, and verifies empirical mass relations
    (Koide formula) on the lattice.

    Usage:
        gen = GenerationStructure(lattice)
        M = gen.lepton_transition_matrix()    # 3×3 integer ΔE matrix
        gaps = gen.lepton_gaps()              # generation spacings
        koide = gen.koide_check()             # Koide formula on grid
        F = gen.full_fermion_matrix()         # 9×9 transition matrix
    """

    # Fermion ordering within sectors
    LEPTONS = ["m_e", "m_μ", "m_τ"]
    QUARKS_UP = ["m_u", "m_c", "m_top"]
    QUARKS_DN = ["m_d", "m_s", "m_b"]
    ALL_FERMIONS = LEPTONS + QUARKS_UP + QUARKS_DN  # canonical 9-element order

    def __init__(self, lattice: ELattice):
        self.lattice = lattice

    # ─── Transition Matrices (ΔE_ij = E_i - E_j) ────────────────────────

    def _transition_matrix(self, names: List[str]) -> np.ndarray:
        """Build N×N transition matrix: M_ij = E(name_i) - E(name_j)."""
        Es = np.array([self.lattice.E(name) for name in names])
        return np.subtract.outer(Es, Es).astype(int)

    def lepton_transition_matrix(self) -> np.ndarray:
        """3×3 lepton transition matrix ΔE_ij (e, μ, τ)."""
        return self._transition_matrix(self.LEPTONS)

    def quark_up_transition_matrix(self) -> np.ndarray:
        """3×3 up-type quark transition matrix ΔE_ij (u, c, t)."""
        return self._transition_matrix(self.QUARKS_UP)

    def quark_down_transition_matrix(self) -> np.ndarray:
        """3×3 down-type quark transition matrix ΔE_ij (d, s, b)."""
        return self._transition_matrix(self.QUARKS_DN)

    def full_fermion_matrix(self) -> np.ndarray:
        """9×9 full fermion transition matrix in E-space."""
        return self._transition_matrix(self.ALL_FERMIONS)

    # ─── E-Address Vectors ───────────────────────────────────────────────

    def lepton_E_vector(self) -> np.ndarray:
        """E-addresses of leptons: [E_e, E_μ, E_τ]."""
        return np.array([self.lattice.E(n) for n in self.LEPTONS])

    def quark_up_E_vector(self) -> np.ndarray:
        """E-addresses of up-type quarks: [E_u, E_c, E_t]."""
        return np.array([self.lattice.E(n) for n in self.QUARKS_UP])

    def quark_down_E_vector(self) -> np.ndarray:
        """E-addresses of down-type quarks: [E_d, E_s, E_b]."""
        return np.array([self.lattice.E(n) for n in self.QUARKS_DN])

    def full_fermion_E_vector(self) -> np.ndarray:
        """All 9 fermion E-addresses in canonical order."""
        return np.array([self.lattice.E(n) for n in self.ALL_FERMIONS])

    # ─── Generation Gaps ─────────────────────────────────────────────────

    def _gaps(self, names: List[str]) -> List[GenerationGap]:
        """Compute generation gaps for a sector."""
        gaps = []
        Q = 416  # grain
        for i in range(len(names) - 1):
            E_lo = self.lattice.E(names[i])
            E_hi = self.lattice.E(names[i + 1])
            dE = E_hi - E_lo
            mass_lo = self.lattice.get(names[i]).value
            mass_hi = self.lattice.get(names[i + 1]).value
            gaps.append(GenerationGap(
                gen_low=i + 1, gen_high=i + 2,
                particle_low=names[i], particle_high=names[i + 1],
                delta_E=dE,
                mass_ratio=mass_hi / mass_lo,
                octaves=dE / Q,
            ))
        return gaps

    def lepton_gaps(self) -> List[GenerationGap]:
        """Generation gaps for leptons: e→μ and μ→τ."""
        return self._gaps(self.LEPTONS)

    def quark_up_gaps(self) -> List[GenerationGap]:
        """Generation gaps for up-type quarks: u→c and c→t."""
        return self._gaps(self.QUARKS_UP)

    def quark_down_gaps(self) -> List[GenerationGap]:
        """Generation gaps for down-type quarks: d→s and s→b."""
        return self._gaps(self.QUARKS_DN)

    def all_gaps(self) -> Dict[str, List[GenerationGap]]:
        """All generation gaps by sector."""
        return {
            "lepton": self.lepton_gaps(),
            "quark_up": self.quark_up_gaps(),
            "quark_down": self.quark_down_gaps(),
        }

    # ─── Koide Formula ───────────────────────────────────────────────────

    def koide_check(self, names: Optional[List[str]] = None) -> KoideResult:
        """
        Check Koide formula: Q = (Σm)/(Σ√m)² =? 2/3

        Default: check for charged leptons (e, μ, τ).
        """
        if names is None:
            names = self.LEPTONS

        # Exact masses
        masses = [self.lattice.get(n).value for n in names]
        sum_m = sum(masses)
        sum_sqrt_m = sum(math.sqrt(m) for m in masses)
        Q_exact = sum_m / (sum_sqrt_m ** 2)

        # Grid-snapped masses
        grid_masses = [self.lattice.get(n).grid_value for n in names]
        sum_gm = sum(grid_masses)
        sum_sqrt_gm = sum(math.sqrt(m) for m in grid_masses)
        Q_grid = sum_gm / (sum_sqrt_gm ** 2)

        target = 2.0 / 3.0
        return KoideResult(
            exact_value=Q_exact,
            grid_value=Q_grid,
            target=target,
            deviation_exact=abs(Q_exact - target),
            deviation_grid=abs(Q_grid - target),
            grid_preserves=abs(Q_grid - target) < 2 * abs(Q_exact - target),
        )

    # ─── Cross-Generation Analysis ───────────────────────────────────────

    def up_down_cross_matrix(self) -> np.ndarray:
        """
        3×3 up↔down cross-transition matrix.
        M_ij = E(up_i) - E(down_j), relevant for CKM transitions.
        """
        up_Es = self.quark_up_E_vector()
        dn_Es = self.quark_down_E_vector()
        return np.subtract.outer(up_Es, dn_Es)

    def generation_hierarchy(self) -> Dict[str, Any]:
        """
        Full generation hierarchy analysis.

        Returns dict with:
          - fermion_E_addresses: all 9 E-values
          - sector_gaps: gaps per sector
          - gap_ratios: Δ₂/Δ₁ for each sector
          - koide: Koide formula check
          - cross_matrix: up↔down transitions
        """
        gaps = self.all_gaps()
        gap_ratios = {}
        for sector, sector_gaps in gaps.items():
            if len(sector_gaps) == 2:
                gap_ratios[sector] = sector_gaps[1].delta_E / sector_gaps[0].delta_E

        return {
            "fermion_E_addresses": {
                n: self.lattice.E(n) for n in self.ALL_FERMIONS
            },
            "sector_gaps": gaps,
            "gap_ratios": gap_ratios,
            "koide": self.koide_check(),
            "cross_matrix": self.up_down_cross_matrix(),
        }

    def __repr__(self) -> str:
        E_vec = self.full_fermion_E_vector()
        return f"GenerationStructure(9 fermions, E∈[{E_vec.min()},{E_vec.max()}])"
