"""L104 Numerical Engine — Superfluid Value Editor.

Quantum min/max value editor with zero-viscosity drift propagation.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F65: Quantum edit propagates to entangled peers with φ-attenuation
  F66: After k hops, attenuation = φ^(-k) — convergent geometric series
  F67: Total propagation energy = drift × Σ_{k=0}^∞ φ^(-k) = drift × φ²
  F85: Superfluid η=0 + Sacred noise ε→0: dual zero-limit convergence
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from decimal import Decimal

from .precision import D, fmt100
from .constants import PHI_INV_HP
from .lattice import TokenLatticeEngine


class SuperfluidValueEditor:
    """Quantum min/max value editor with zero-viscosity drift propagation.

    Part V (F65-F67): Edits propagate through entangled tokens via φ-attenuated
    superfluid coupling. Total energy = drift × φ² (convergent geometric series).
    """

    def __init__(self, lattice: TokenLatticeEngine):
        """Initialize SuperfluidValueEditor."""
        self.lattice = lattice
        self.edit_count = 0
        self.propagation_log: List[Dict] = []
        self.total_propagation_energy = D(0)  # F67: tracks Σ drift×φ²

    def quantum_edit(self, token_id: str,
                     new_value: Optional[Decimal] = None,
                     new_min: Optional[Decimal] = None,
                     new_max: Optional[Decimal] = None,
                     reason: str = "manual") -> Dict:
        """Edit a token's value and/or boundaries with quantum propagation.

        If the token is entangled with others, the edit propagates to peers
        with φ-attenuated strength (superfluid coupling).
        """
        token = self.lattice.tokens.get(token_id)
        if token is None:
            return {"error": f"Token {token_id} not found"}

        old_value = D(token.value)
        old_min = D(token.min_bound)
        old_max = D(token.max_bound)

        # Apply edits
        if new_value is not None:
            token.value = fmt100(new_value)
        if new_min is not None:
            token.min_bound = fmt100(new_min)
        if new_max is not None:
            token.max_bound = fmt100(new_max)

        token.last_adjusted = datetime.now(timezone.utc).isoformat()
        self.edit_count += 1

        # Compute drift vector
        drift_v = D(token.value) - old_value
        drift_min = D(token.min_bound) - old_min
        drift_max = D(token.max_bound) - old_max

        result = {
            "token_id": token_id,
            "drift_value": str(drift_v),
            "drift_min": str(drift_min),
            "drift_max": str(drift_max),
            "reason": reason,
            "propagated_to": [],
        }

        # Superfluid propagation to entangled peers (F65-F67)
        if token.entangled_tokens:
            phi_attenuation = PHI_INV_HP  # F66: each hop attenuates by φ⁻¹
            for peer_id in token.entangled_tokens:
                peer = self.lattice.tokens.get(peer_id)
                if peer is None:
                    continue
                # Propagate with φ-attenuated drift
                peer_drift_v = drift_v * phi_attenuation
                peer_drift_min = drift_min * phi_attenuation
                peer_drift_max = drift_max * phi_attenuation

                peer_val = D(peer.value) + peer_drift_v
                peer_lo = D(peer.min_bound) + peer_drift_min
                peer_hi = D(peer.max_bound) + peer_drift_max

                peer.value = fmt100(peer_val)
                peer.min_bound = fmt100(peer_lo)
                peer.max_bound = fmt100(peer_hi)
                peer.last_adjusted = datetime.now(timezone.utc).isoformat()
                result["propagated_to"].append(peer_id)

            # F67: Total propagation energy = drift × φ² (geometric series sum)
            if abs(drift_v) > 0:
                energy = abs(drift_v) * PHI_INV_HP * PHI_INV_HP  # φ⁻² series total for peers
                self.total_propagation_energy += energy

        self.propagation_log.append(result)
        return result

    def entangle_tokens(self, token_id_a: str, token_id_b: str) -> bool:
        """Create quantum entanglement between two tokens.
        Edits to one will propagate to the other with φ-attenuation."""
        a = self.lattice.tokens.get(token_id_a)
        b = self.lattice.tokens.get(token_id_b)
        if a is None or b is None:
            return False
        if token_id_b not in a.entangled_tokens:
            a.entangled_tokens.append(token_id_b)
        if token_id_a not in b.entangled_tokens:
            b.entangled_tokens.append(token_id_a)
        return True

    def batch_drift(self, drift_map: Dict[str, Decimal], reason: str = "batch") -> Dict:
        """Apply a batch of drifts to multiple tokens simultaneously.
        Used by the subconscious monitor for coordinated adjustments."""
        results = []
        for token_id, drift in drift_map.items():
            token = self.lattice.tokens.get(token_id)
            if token is None:
                continue
            new_val = D(token.value) + drift
            res = self.quantum_edit(token_id, new_value=new_val, reason=reason)
            results.append(res)
        return {"batch_size": len(drift_map), "applied": len(results), "results": results}
