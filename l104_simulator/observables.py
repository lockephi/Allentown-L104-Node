"""
===============================================================================
L104 SIMULATOR — LAYER 5: OBSERVABLE EXTRACTION
===============================================================================

Extracts measurable physics from the quantum state:
  1. Mass Measurement — decode E-address → physical mass (MeV/c²)
  2. Mass Ratios — exact integer subtraction in E-space
  3. Decay Rates — CKM/PMNS element × phase space factor
  4. Oscillation Probabilities — flavor transition amplitudes
  5. Cross Sections — σ ∝ |V|² × phase_space × propagator
  6. Running Couplings — α(μ) from RG-evolved E-address
  7. Koide/Sum Rule Verification — validate grid faithfulness

All observables are computed from the E-lattice representation,
with error bounds from the grid quantization (±0.0834% max).

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .constants import (
    PHI, GOD_CODE, Q_GRAIN, BASE, MAX_GRID_ERROR,
    M_W, M_Z, ALPHA_EM, ALPHA_INV,
    CKM_THETA12, CKM_THETA13, CKM_THETA23,
    PMNS_THETA12, PMNS_THETA13, PMNS_THETA23,
)
from .lattice import ELattice
from .generations import GenerationStructure
from .mixing import MixingMatrices


@dataclass
class MassObservable:
    """A particle mass extracted from the E-lattice."""
    name: str
    E_address: int          # Integer lattice address
    mass_grid: float        # Mass from grid (MeV/c²)
    mass_exact: float       # Exact measured mass (MeV/c²)
    error_pct: float        # Grid quantization error (%)
    unit: str = "MeV/c²"


@dataclass
class RatioObservable:
    """A dimensionless ratio computed via E-space subtraction."""
    name: str
    numerator: str
    denominator: str
    dE: int                 # E(num) - E(den) — exact integer
    ratio_grid: float       # 2^(ΔE/Q) × (BASE_num/BASE_den)
    ratio_exact: float      # PDG value
    error_pct: float


@dataclass
class DecayObservable:
    """A decay rate observable."""
    process: str            # e.g. "top → W + bottom"
    ckm_element: float      # |V_ij|²
    phase_space: float      # Phase space factor (E-space estimate)
    rate_relative: float    # Relative rate vs reference
    branching_ratio: float  # Estimated BR


@dataclass
class OscillationObservable:
    """A flavor oscillation probability."""
    channel: str            # e.g. "ν_μ → ν_e"
    L_over_E: float         # Propagation L/E (km/GeV)
    probability: float      # P(a→b)
    dE_squared: float       # ΔE² driving the oscillation (E-space units)


class Observables:
    """
    Layer 5: Observable extraction from the E-lattice.

    Computes physically measurable quantities from the lattice representation,
    with quantized error bounds from the grid resolution.

    Usage:
        obs = Observables(lattice, generations, mixing)

        # Mass measurements
        masses = obs.all_masses()
        m = obs.measure_mass("m_e")

        # Ratios (exact in E-space)
        r = obs.mass_ratio("m_top", "m_e")

        # Decay rates
        decays = obs.quark_decay_rates()

        # Oscillation
        P = obs.oscillation_probability("lepton", 0, 1, L_over_E=500)

        # Summary
        report = obs.full_report()
    """

    def __init__(self, lattice: ELattice, generations: GenerationStructure,
                 mixing: MixingMatrices):
        self.lattice = lattice
        self.gen = generations
        self.mix = mixing

    # ═════════════════════════════════════════════════════════════════════════
    #  1. MASS MEASUREMENTS
    # ═════════════════════════════════════════════════════════════════════════

    def measure_mass(self, name: str) -> MassObservable:
        """Extract a particle mass from the E-lattice with error bound."""
        pt = self.lattice.get(name)
        E = pt.E
        mass_grid = self.lattice.decode(E)
        mass_exact = pt.value
        error = abs(mass_grid - mass_exact) / mass_exact * 100 if mass_exact > 0 else 0.0

        return MassObservable(
            name=name,
            E_address=E,
            mass_grid=mass_grid,
            mass_exact=mass_exact,
            error_pct=error,
            unit=pt.unit,
        )

    def all_masses(self) -> List[MassObservable]:
        """Measure all registered particle masses."""
        fermions = ["m_e", "m_μ", "m_τ", "m_u", "m_c", "m_top",
                    "m_d", "m_s", "m_b"]
        bosons = ["m_W", "m_Z", "m_H"]
        hadrons = ["m_p", "m_n", "m_π±", "m_π0", "m_K", "m_D"]
        all_names = fermions + bosons + hadrons

        results = []
        for name in all_names:
            try:
                results.append(self.measure_mass(name))
            except KeyError:
                pass
        return results

    def mass_accuracy_report(self) -> Dict[str, Any]:
        """Summary statistics of mass measurement accuracy."""
        masses = self.all_masses()
        errors = [m.error_pct for m in masses]
        return {
            "num_particles": len(masses),
            "mean_error_pct": np.mean(errors),
            "max_error_pct": np.max(errors),
            "min_error_pct": np.min(errors),
            "std_error_pct": np.std(errors),
            "theoretical_max_error": MAX_GRID_ERROR,
            "all_within_theory": all(e <= MAX_GRID_ERROR * 1.01 for e in errors),
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  2. MASS RATIOS (EXACT IN E-SPACE)
    # ═════════════════════════════════════════════════════════════════════════

    def mass_ratio(self, name_a: str, name_b: str) -> RatioObservable:
        """
        Compute a mass ratio via E-space subtraction.

        Exact: ratio_grid = 2^(ΔE/Q) where ΔE = E(a) - E(b) is an integer.
        The homomorphism guarantees: E(a/b) = E(a) - E(b).
        """
        E_a = self.lattice.E(name_a)
        E_b = self.lattice.E(name_b)
        dE = E_a - E_b

        ratio_grid = 2 ** (dE / Q_GRAIN)
        pt_a = self.lattice.get(name_a)
        pt_b = self.lattice.get(name_b)
        ratio_exact = pt_a.value / pt_b.value if pt_b.value > 0 else float('inf')

        error = abs(ratio_grid - ratio_exact) / ratio_exact * 100 if ratio_exact > 0 else 0.0

        return RatioObservable(
            name=f"{name_a}/{name_b}",
            numerator=name_a,
            denominator=name_b,
            dE=dE,
            ratio_grid=ratio_grid,
            ratio_exact=ratio_exact,
            error_pct=error,
        )

    def key_ratios(self) -> List[RatioObservable]:
        """Standard Model key mass ratios."""
        pairs = [
            ("m_top", "m_e"),       # Top/electron ≈ 338,000
            ("m_top", "m_b"),       # Top/bottom ≈ 41
            ("m_τ", "m_e"),         # Tau/electron ≈ 3477
            ("m_μ", "m_e"),         # Muon/electron ≈ 207
            ("m_W", "m_Z"),         # cos(θ_W) ≈ 0.882
            ("m_H", "m_W"),         # Higgs/W ≈ 1.56
            ("m_p", "m_e"),         # Proton/electron ≈ 1836
            ("m_n", "m_p"),         # Neutron/proton ≈ 1.0014
            ("m_c", "m_s"),         # Charm/strange ≈ 13.6
            ("m_top", "m_u"),       # Top/up ≈ 75,000
        ]
        return [self.mass_ratio(a, b) for a, b in pairs]

    # ═════════════════════════════════════════════════════════════════════════
    #  3. DECAY RATES
    # ═════════════════════════════════════════════════════════════════════════

    def quark_decay_rates(self) -> List[DecayObservable]:
        """
        Estimate relative quark decay rates from CKM elements.

        Rate ∝ |V_ij|² × (mass_parent)^5 / (mass_W)^4  (for semileptonic)
        In E-space:
          Rate_E ∝ |V_ij|² × 2^(5×E_parent/Q) / 2^(4×E_W/Q)
        """
        V = self.mix.ckm_matrix()
        V_mag2 = np.abs(V) ** 2  # |V_ij|²

        up_quarks = [("m_u", "u"), ("m_c", "c"), ("m_top", "t")]
        down_quarks = [("m_d", "d"), ("m_s", "s"), ("m_b", "b")]

        E_W = self.lattice.E("m_W")
        results = []

        for i, (up_name, up_label) in enumerate(up_quarks):
            E_parent = self.lattice.E(up_name)
            for j, (dn_name, dn_label) in enumerate(down_quarks):
                ckm = float(V_mag2[i, j])
                # Phase space factor: (m_parent/m_W)^4 in E-space
                phase_space = 2 ** (4 * (E_parent - E_W) / Q_GRAIN)
                rate = ckm * phase_space

                results.append(DecayObservable(
                    process=f"{up_label} → {dn_label} + W",
                    ckm_element=ckm,
                    phase_space=phase_space,
                    rate_relative=rate,
                    branching_ratio=0.0,  # to be normalized
                ))

        # Normalize branching ratios per parent
        for i in range(3):
            group = results[i*3:(i+1)*3]
            total = sum(d.rate_relative for d in group)
            if total > 0:
                for d in group:
                    d.branching_ratio = d.rate_relative / total

        return results

    # ═════════════════════════════════════════════════════════════════════════
    #  4. OSCILLATION PROBABILITIES
    # ═════════════════════════════════════════════════════════════════════════

    def oscillation_probability(self, sector: str,
                                 gen_from: int, gen_to: int,
                                 L_over_E: float = 500.0) -> OscillationObservable:
        """
        Compute flavor oscillation probability P(gen_from → gen_to).

        P(a→b) = δ_ab - 4 Σ_{i>j} Re(U*_ai U_bi U_aj U*_bj) sin²(ΔE_ij L/4E)
                      + 2 Σ_{i>j} Im(U*_ai U_bi U_aj U*_bj) sin(ΔE_ij L/2E)

        Simplified two-flavor approximation for quick computation.
        """
        if sector == "lepton":
            U = self.mix.pmns_matrix()
            E_vec = self.gen.lepton_E_vector().astype(float)
        else:
            U = self.mix.ckm_matrix()
            E_vec = self.gen.quark_down_E_vector().astype(float)

        a, b = gen_from, gen_to

        # Full three-flavor oscillation formula
        P = float(a == b)  # δ_ab base
        scale = math.log(2) / Q_GRAIN * L_over_E

        for i in range(3):
            for j in range(i + 1, 3):
                dE = 2 * (E_vec[j] - E_vec[i])  # ΔE² proxy
                phase = dE * scale / 2

                # Jarlskog-like product
                product = U[a, i].conj() * U[b, i] * U[a, j] * U[b, j].conj()
                P -= 4 * float(product.real) * math.sin(phase) ** 2
                P += 2 * float(product.imag) * math.sin(2 * phase)

        # Clamp to physical range
        P = max(0.0, min(1.0, P))

        dE_max = 2 * (E_vec[2] - E_vec[0])

        channel_names = {
            "lepton": {(0, 1): "ν_e → ν_μ", (0, 2): "ν_e → ν_τ",
                       (1, 0): "ν_μ → ν_e", (1, 2): "ν_μ → ν_τ",
                       (2, 0): "ν_τ → ν_e", (2, 1): "ν_τ → ν_μ",
                       (0, 0): "ν_e → ν_e", (1, 1): "ν_μ → ν_μ",
                       (2, 2): "ν_τ → ν_τ"},
            "quark": {(i, j): f"q{i} → q{j}" for i in range(3) for j in range(3)},
        }

        channel = channel_names.get(sector, {}).get((a, b), f"{a}→{b}")

        return OscillationObservable(
            channel=channel,
            L_over_E=L_over_E,
            probability=P,
            dE_squared=dE_max,
        )

    def oscillation_spectrum(self, sector: str = "lepton",
                              gen_from: int = 1, gen_to: int = 0,
                              L_over_E_range: Optional[List[float]] = None
                              ) -> List[OscillationObservable]:
        """Compute oscillation probability over a range of L/E values."""
        if L_over_E_range is None:
            L_over_E_range = [10, 50, 100, 200, 500, 1000, 2000, 5000]

        return [
            self.oscillation_probability(sector, gen_from, gen_to, LE)
            for LE in L_over_E_range
        ]

    # ═════════════════════════════════════════════════════════════════════════
    #  5. RUNNING COUPLINGS
    # ═════════════════════════════════════════════════════════════════════════

    def running_alpha_em(self, E_scale: int) -> Dict[str, float]:
        """
        Estimate electromagnetic coupling α(μ) at an E-space scale.

        One-loop running:
          α⁻¹(μ₂) = α⁻¹(μ₁) - b₀/(2π) × ln(μ₂/μ₁)

        In E-space:
          Δ(α⁻¹) = -b₀/(2π) × ΔE × ln(2)/Q
        """
        # Reference: α⁻¹(m_Z) ≈ 127.95
        E_Z = self.lattice.E("m_Z")
        alpha_inv_Z = 127.95
        b0_em = -80.0 / (9.0 * math.pi)  # QED one-loop (leptons + quarks)

        dE = E_scale - E_Z
        delta_alpha_inv = -b0_em / (2 * math.pi) * dE * math.log(2) / Q_GRAIN
        alpha_inv_mu = alpha_inv_Z + delta_alpha_inv
        alpha_mu = 1.0 / alpha_inv_mu if alpha_inv_mu > 0 else 0.0

        return {
            "E_scale": E_scale,
            "α⁻¹(μ)": alpha_inv_mu,
            "α(μ)": alpha_mu,
            "reference_E": E_Z,
            "ΔE": dE,
            "b₀": b0_em,
        }

    def running_alpha_s(self, E_scale: int) -> Dict[str, float]:
        """
        Estimate strong coupling α_s(μ) at an E-space scale.

        One-loop: α_s⁻¹(μ₂) = α_s⁻¹(μ₁) + b₀/(2π) × ln(μ₂/μ₁)
        b₀ = (11C_A - 4T_F n_f) / (12π) = (33 - 2n_f) / (12π)
        """
        E_Z = self.lattice.E("m_Z")
        alpha_s_Z = 0.1179  # α_s(m_Z)
        alpha_s_inv_Z = 1.0 / alpha_s_Z
        n_f = 5  # Active flavors at m_Z
        b0 = (33 - 2 * n_f) / (12 * math.pi)

        dE = E_scale - E_Z
        delta = b0 / (2 * math.pi) * dE * math.log(2) / Q_GRAIN
        alpha_s_inv_mu = alpha_s_inv_Z + delta
        alpha_s_mu = 1.0 / alpha_s_inv_mu if alpha_s_inv_mu > 0 else float('inf')

        return {
            "E_scale": E_scale,
            "α_s⁻¹(μ)": alpha_s_inv_mu,
            "α_s(μ)": alpha_s_mu,
            "reference_E": E_Z,
            "ΔE": dE,
            "n_f": n_f,
            "b₀": b0,
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  6. KOIDE & SUM RULE VERIFICATION
    # ═════════════════════════════════════════════════════════════════════════

    def koide_observable(self) -> Dict[str, Any]:
        """
        Verify Koide's formula from E-lattice masses.

        Koide: (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² = 2/3
        """
        koide = self.gen.koide_check()
        return {
            "koide_exact": koide.exact_value,
            "koide_grid": koide.grid_value,
            "target": koide.target,
            "deviation_exact": koide.deviation_exact,
            "deviation_grid": koide.deviation_grid,
            "grid_preserves": koide.grid_preserves,
        }

    def sum_rules(self) -> Dict[str, Any]:
        """Check various E-space sum rules."""
        # Proton mass sum rule: m_p ≈ 2m_u + m_d + binding
        E_p = self.lattice.E("m_p")
        E_u = self.lattice.E("m_u")
        E_d = self.lattice.E("m_d")

        # In E-space, m_p = 2m_u × m_d × binding
        # → E_p ≈ 2E_u + E_d + E_binding (approximately)
        # The excess E captures QCD binding energy
        E_constituent = 2 * E_u + E_d
        E_binding_excess = E_p - E_constituent

        # Neutron-proton mass difference
        E_n = self.lattice.E("m_n")
        dE_np = E_n - E_p

        return {
            "proton_composition": {
                "E(p)": E_p,
                "2E(u) + E(d)": E_constituent,
                "E_binding_excess": E_binding_excess,
                "note": "Excess E captures QCD dynamics",
            },
            "neutron_proton_split": {
                "ΔE(n-p)": dE_np,
                "mass_diff_grid_MeV": self.lattice.decode(E_n) - self.lattice.decode(E_p),
                "mass_diff_exact_MeV": 1293.332,  # PDG value (keV → MeV)
            },
            "koide": self.koide_observable(),
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  FULL REPORT
    # ═════════════════════════════════════════════════════════════════════════

    def full_report(self) -> Dict[str, Any]:
        """Comprehensive observable report."""
        return {
            "mass_accuracy": self.mass_accuracy_report(),
            "key_ratios": [
                {"name": r.name, "ΔE": r.dE,
                 "ratio_grid": r.ratio_grid, "ratio_exact": r.ratio_exact,
                 "error_pct": r.error_pct}
                for r in self.key_ratios()
            ],
            "sum_rules": self.sum_rules(),
            "oscillation_example": {
                "channel": "ν_μ → ν_e",
                "probabilities": [
                    {"L/E": o.L_over_E, "P": o.probability}
                    for o in self.oscillation_spectrum("lepton", 1, 0)
                ],
            },
            "running_couplings": {
                "α_em(m_Z)": self.running_alpha_em(self.lattice.E("m_Z")),
                "α_s(m_Z)": self.running_alpha_s(self.lattice.E("m_Z")),
            },
        }

    def __repr__(self) -> str:
        return "Observables(mass + ratio + decay + oscillation + running)"
