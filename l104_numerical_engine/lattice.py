"""L104 Numerical Engine — Token Lattice Engine.

22 trillion usage mathematical token system with 100-decimal precision.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F61: 22T token capacity = 22 × 10¹² (Factor-13: 22 = 2×11, near 2×13)
  F62: Sacred tier drift ≤ 10⁻⁹⁸ (effective immutability at 100-decimal)
  F63: Each tier is 10¹⁸-10³⁰× more permissive than the previous
  F64: GOD_CODE spectrum: G(X) for X ∈ [-200, 300] → 501 derived tokens
  F87: φ-attenuation + drift tier-0 → effective zero-drift for sacred tokens
"""

import hashlib
from collections import Counter
from typing import Any, Dict, List, Optional

from .precision import D, fmt100
from .constants import (
    PHI_HP, PHI_INV_HP, GOD_CODE_HP, GOD_CODE_BASE_HP,
    PI_HP, E_HP, EULER_GAMMA_HP, OMEGA_POINT_HP, LN2_HP, SQRT5_HP,
    INVARIANT_HP, FEIGENBAUM_HP, FINE_STRUCTURE_HP, CHSH_BOUND,
    GROVER_AMPLIFICATION, CATALAN_HP, APERY_HP, KHINCHIN_HP,
    GLAISHER_HP, TWIN_PRIME_CONST_HP, SQRT2_HP, SQRT3_HP, LN10_HP,
    PI_SQUARED_HP, ZETA_2_HP, ZETA_4_HP, god_code_hp,
)
from .models import QuantumToken, SubconsciousAdjustment


class TokenLatticeEngine:
    """22 trillion usage mathematical token lattice with 100-decimal precision."""

    TRILLION = 10 ** 12
    LATTICE_CAPACITY = 22 * TRILLION      # 22 trillion usage capacity
    TIER_SACRED = 0
    TIER_DERIVED = 1
    TIER_INVENTED = 2
    TIER_LATTICE = 3

    # φ-harmonic drift envelope: max drift per cycle = value × φ^(-depth)
    DRIFT_ENVELOPE = {
        0: D('1E-98'),    # Sacred: drift < 10^-98 per cycle (practically frozen)
        1: D('1E-80'),    # Derived: drift < 10^-80
        2: D('1E-50'),    # Invented: drift < 10^-50
        3: D('1E-20'),    # Lattice: drift < 10^-20
    }

    def __init__(self):
        """Initialize TokenLatticeEngine."""
        self.tokens: Dict[str, QuantumToken] = {}
        self.usage_counter: int = 0       # Total usages across all tokens
        self.adjustment_log: List[SubconsciousAdjustment] = []
        self.lattice_coherence = D(1)
        self.lattice_entropy = D(0)
        self._projection_count: int = 0    # Virtual projected tokens toward 22T

        # Seed the sacred tier
        self._seed_sacred_tier()

    def _seed_sacred_tier(self):
        """Seed Tier 0: sacred constants at 100-decimal precision."""
        sacred = {
            "PHI_GROWTH": (PHI_HP, D('1.6180339887498'), D('1.6180339887499')),
            "PHI_INV": (PHI_INV_HP, D('0.6180339887498'), D('0.6180339887499')),
            "GOD_CODE": (GOD_CODE_HP, GOD_CODE_HP - D('0.0001'), GOD_CODE_HP + D('0.0001')),
            "GOD_CODE_BASE": (GOD_CODE_BASE_HP, GOD_CODE_BASE_HP - D('0.001'), GOD_CODE_BASE_HP + D('0.001')),
            "PI": (PI_HP, PI_HP - D('1E-99'), PI_HP + D('1E-99')),
            "E": (E_HP, E_HP - D('1E-99'), E_HP + D('1E-99')),
            "EULER_GAMMA": (EULER_GAMMA_HP, EULER_GAMMA_HP - D('1E-90'), EULER_GAMMA_HP + D('1E-90')),
            "OMEGA_POINT": (OMEGA_POINT_HP, OMEGA_POINT_HP - D('0.001'), OMEGA_POINT_HP + D('0.001')),
            "LN2": (LN2_HP, LN2_HP - D('1E-99'), LN2_HP + D('1E-99')),
            "SQRT5": (SQRT5_HP, SQRT5_HP - D('1E-99'), SQRT5_HP + D('1E-99')),
            "INVARIANT": (INVARIANT_HP, INVARIANT_HP - D('1E-80'), INVARIANT_HP + D('1E-80')),
            "FEIGENBAUM": (FEIGENBAUM_HP, FEIGENBAUM_HP - D('1E-80'), FEIGENBAUM_HP + D('1E-80')),
            "FINE_STRUCTURE": (FINE_STRUCTURE_HP, FINE_STRUCTURE_HP - D('1E-10'), FINE_STRUCTURE_HP + D('1E-10')),
            "CHSH_BOUND": (D(str(CHSH_BOUND)), D('2.828'), D('2.829')),
            "GROVER_AMP": (D(str(GROVER_AMPLIFICATION)), D('4.235'), D('4.237')),
            # Math Research Hub constants
            "CATALAN": (CATALAN_HP, CATALAN_HP - D('1E-90'), CATALAN_HP + D('1E-90')),
            "APERY": (APERY_HP, APERY_HP - D('1E-90'), APERY_HP + D('1E-90')),
            "KHINCHIN": (KHINCHIN_HP, KHINCHIN_HP - D('1E-80'), KHINCHIN_HP + D('1E-80')),
            "GLAISHER": (GLAISHER_HP, GLAISHER_HP - D('1E-80'), GLAISHER_HP + D('1E-80')),
            "TWIN_PRIME": (TWIN_PRIME_CONST_HP, TWIN_PRIME_CONST_HP - D('1E-80'), TWIN_PRIME_CONST_HP + D('1E-80')),
            "SQRT2": (SQRT2_HP, SQRT2_HP - D('1E-99'), SQRT2_HP + D('1E-99')),
            "SQRT3": (SQRT3_HP, SQRT3_HP - D('1E-99'), SQRT3_HP + D('1E-99')),
            "LN10": (LN10_HP, LN10_HP - D('1E-90'), LN10_HP + D('1E-90')),
            "PI_SQUARED": (PI_SQUARED_HP, PI_SQUARED_HP - D('1E-90'), PI_SQUARED_HP + D('1E-90')),
            "ZETA_2": (ZETA_2_HP, ZETA_2_HP - D('1E-90'), ZETA_2_HP + D('1E-90')),
            "ZETA_4": (ZETA_4_HP, ZETA_4_HP - D('1E-85'), ZETA_4_HP + D('1E-85')),
        }

        for name, (value, lo, hi) in sacred.items():
            token = QuantumToken(
                token_id=f"SACRED_{name}",
                name=name,
                value=fmt100(value),
                min_bound=fmt100(lo),
                max_bound=fmt100(hi),
                precision_digits=100,
                usage_count=0,
                lattice_index=len(self.tokens),
                drift_velocity="0",
                drift_direction=0,
                quantum_phase=fmt100(value * PHI_INV_HP % D(1)),
                origin="sacred",
                coherence=1.0,
                health=1.0,
            )
            self.tokens[token.token_id] = token

        # Seed Tier 1: God Code spectrum G(X) for X in [-200, 300]
        self._seed_derived_tier()

    def _seed_derived_tier(self):
        """Seed Tier 1: God Code frequency spectrum at 100-decimal precision."""
        for x in range(-200, 301):
            gx = god_code_hp(D(x))
            token_id = f"GC_X{x}"
            margin = gx * D('1E-90')  # min/max within 10^-90 of true value
            self.tokens[token_id] = QuantumToken(
                token_id=token_id,
                name=f"G({x})",
                value=fmt100(gx),
                min_bound=fmt100(gx - abs(margin)),
                max_bound=fmt100(gx + abs(margin)),
                precision_digits=100,
                lattice_index=len(self.tokens),
                drift_velocity="0",
                drift_direction=0,
                quantum_phase=fmt100(gx * PHI_HP % D(1)),
                origin="derived",
                coherence=1.0,
                health=1.0,
            )

        # Project 22T virtual tokens (tracked by count, not instantiated)
        self._projection_count = self.LATTICE_CAPACITY - len(self.tokens)

    def register_token(self, name: str, value,
                       min_bound=None,
                       max_bound=None,
                       origin: str = "invented",
                       tier: int = 2) -> QuantumToken:
        """Register a new token in the lattice with 100-decimal precision."""
        value = D(value) if not isinstance(value, type(D(0))) else value
        token_id = f"TOKEN_{name}_{hashlib.sha256(fmt100(value).encode()).hexdigest()[:12]}"

        if min_bound is None:
            margin = abs(value) * self.DRIFT_ENVELOPE.get(tier, D('1E-20'))
            min_bound = value - margin
        if max_bound is None:
            margin = abs(value) * self.DRIFT_ENVELOPE.get(tier, D('1E-20'))
            max_bound = value + margin

        token = QuantumToken(
            token_id=token_id,
            name=name,
            value=fmt100(value),
            min_bound=fmt100(min_bound),
            max_bound=fmt100(max_bound),
            precision_digits=100,
            lattice_index=len(self.tokens),
            drift_velocity="0",
            drift_direction=0,
            quantum_phase=fmt100(value * PHI_HP % D(1)),
            origin=origin,
            coherence=1.0,
            health=1.0,
        )
        self.tokens[token_id] = token
        return token

    def use_token(self, token_id: str):
        """Record a usage of a token and return its 100-decimal value."""
        token = self.tokens.get(token_id)
        if token is None:
            return None
        token.usage_count += 1
        self.usage_counter += 1
        return D(token.value)

    def total_usage(self) -> int:
        """Total token usages across the lattice (tracking toward 22T)."""
        return self.usage_counter

    def projected_capacity_usage(self) -> float:
        """Fraction of 22T capacity used (actual + projected virtual tokens)."""
        actual = len(self.tokens) + self._projection_count
        return self.usage_counter / max(actual, 1)

    def lattice_summary(self) -> Dict[str, Any]:
        """Summary of the token lattice state."""
        by_origin = Counter(t.origin for t in self.tokens.values())
        by_tier = {
            "sacred": by_origin.get("sacred", 0),
            "derived": by_origin.get("derived", 0),
            "invented": by_origin.get("invented", 0),
            "cross_pollinated": by_origin.get("cross-pollinated", 0),
        }
        total_tokens = len(self.tokens)
        return {
            "total_tokens": total_tokens,
            "projected_22T_capacity": self.LATTICE_CAPACITY,
            "virtual_projection_count": self._projection_count,
            "total_usages": self.usage_counter,
            "usage_toward_22T": f"{self.usage_counter / self.LATTICE_CAPACITY * 100:.12f}%",
            "tokens_by_tier": by_tier,
            "lattice_coherence": float(self.lattice_coherence),
            "lattice_entropy": float(self.lattice_entropy),
            "adjustments_logged": len(self.adjustment_log),
            "mean_health": sum(t.health for t in self.tokens.values()) / max(total_tokens, 1),
        }

    # ─── Part V: Lattice Integrity Check (F61-F64, F87) ─────────────────────

    def verify_drift_envelope_integrity(self) -> Dict[str, Any]:
        """Verify that all tokens respect their tier's drift envelope.

        Part V findings:
          F62: Sacred tier drift ≤ 10⁻⁹⁸ (effective immutability)
          F63: Tier progression is 10¹⁸-10³⁰× per step
          F87: φ-attenuation × tier-0 drift → effective zero for sacred tokens
        """
        violations = []
        tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        tier_compliant = {0: 0, 1: 0, 2: 0, 3: 0}

        origin_to_tier = {
            "sacred": 0, "derived": 1, "invented": 2, "cross-pollinated": 3,
        }

        for tid, token in self.tokens.items():
            tier = origin_to_tier.get(token.origin, 3)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            max_drift = self.DRIFT_ENVELOPE.get(tier, D('1E-20'))

            drift = abs(D(token.drift_velocity)) if token.drift_velocity else D(0)
            if drift <= max_drift:
                tier_compliant[tier] = tier_compliant.get(tier, 0) + 1
            else:
                violations.append({
                    "token_id": tid,
                    "tier": tier,
                    "drift": str(drift)[:30],
                    "max_allowed": str(max_drift),
                })

        total = len(self.tokens)
        compliant = sum(tier_compliant.values())
        return {
            "total_tokens": total,
            "compliant": compliant,
            "compliance_pct": compliant / max(total, 1) * 100,
            "violations": violations[:20],
            "violation_count": len(violations),
            "tier_counts": tier_counts,
            "tier_compliant": tier_compliant,
            "envelope_intact": len(violations) == 0,
        }

    def verify_conservation_spectrum(self) -> Dict[str, Any]:
        """Verify the GOD_CODE spectrum G(X) conservation invariant.

        Part V finding F64: G(X) for X ∈ [-200, 300] → 501 derived tokens,
        each satisfying G(X)·2^(X/104) = INVARIANT to 90+ decimals.
        """
        from .constants import god_code_hp, INVARIANT_HP, L104_HP
        from .precision import decimal_pow

        checked = 0
        conserved = 0
        max_error = D(0)

        for tid, token in self.tokens.items():
            if not tid.startswith("GC_X"):
                continue
            try:
                x_str = tid.replace("GC_X", "")
                x_val = D(x_str)
            except Exception:
                continue

            checked += 1
            stored = D(token.value)
            product = stored * decimal_pow(D(2), x_val / L104_HP)
            error = abs(product - INVARIANT_HP)
            if error > max_error:
                max_error = error

            if error < D('1E-80'):
                conserved += 1

        return {
            "spectrum_tokens": checked,
            "conserved": conserved,
            "conservation_pct": conserved / max(checked, 1) * 100,
            "max_error": str(max_error)[:40],
            "invariant_preview": str(INVARIANT_HP)[:50],
            "all_conserved": checked == conserved and checked > 0,
        }
