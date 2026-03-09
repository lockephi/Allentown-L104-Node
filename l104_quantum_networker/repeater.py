"""L104 Quantum Networker v1.0.0 — Quantum Repeater Chain.

Implements quantum repeater chains for long-distance quantum communication.
Uses entanglement swapping at relay nodes and nested purification to extend
quantum links beyond direct-connection range while maintaining high fidelity.

Repeater Architecture:
  ┌───┐     ┌───┐     ┌───┐     ┌───┐     ┌───┐
  │ A │─────│R1 │─────│R2 │─────│R3 │─────│ B │
  └───┘     └───┘     └───┘     └───┘     └───┘
    ↕ Bell    ↕ Bell    ↕ Bell    ↕ Bell
    pairs     pairs     pairs     pairs

  Level 0: Generate Bell pairs on each segment
  Level 1: Swap at R1 → A↔R2, swap at R3 → R2↔B
  Level 2: Swap at R2 → A↔B (end-to-end entanglement)
  Each level: optional purification round

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import (
    QuantumNode, QuantumChannel, EntangledPair,
    GOD_CODE, PHI, PHI_INV,
)
from .entanglement_router import EntanglementRouter


class QuantumRepeaterChain:
    """Quantum repeater chain for long-distance entanglement distribution.

    Implements a hierarchical swap+purify protocol:
      1. Generate Bell pairs on each segment
      2. Purify each segment to target fidelity
      3. Swap at alternating relay nodes (binary tree of swaps)
      4. Purify the longer-range pairs
      5. Repeat until end-to-end pair achieved
    """

    def __init__(self, router: EntanglementRouter,
                 purification_rounds: int = 2,
                 target_fidelity: float = 0.95):
        """
        Args:
            router: EntanglementRouter for pair management
            purification_rounds: Number of purification rounds per level
            target_fidelity: Minimum acceptable fidelity for the output pair
        """
        self.router = router
        self.purification_rounds = purification_rounds
        self.target_fidelity = target_fidelity
        self._chains_executed: List[Dict] = []

    def establish_chain(
        self,
        node_ids: List[str],
        pairs_per_segment: int = 8,
    ) -> Dict:
        """Establish an end-to-end entangled pair through a repeater chain.

        Args:
            node_ids: Ordered list of node IDs [A, R1, R2, ..., B]
            pairs_per_segment: Initial Bell pairs per segment

        Returns:
            Dict with chain results, final pair, fidelity metrics
        """
        t0 = time.time()
        n = len(node_ids)

        if n < 2:
            return {"success": False, "error": "Need at least 2 nodes"}

        if n == 2:
            # Direct connection, just generate pairs
            ch = self.router.get_channel(node_ids[0], node_ids[1])
            if not ch:
                ch = self.router.create_channel(node_ids[0], node_ids[1],
                                                  pairs_per_segment)
            pair = ch.best_pair
            if pair:
                return {
                    "success": True,
                    "final_pair": pair.to_dict(),
                    "fidelity": pair.current_fidelity,
                    "hops": 1,
                    "route": node_ids,
                    "levels": 0,
                    "execution_time_ms": (time.time() - t0) * 1000,
                }
            return {"success": False, "error": "No usable pairs"}

        # Step 1: Ensure channels exist on each segment with sufficient pairs
        segments = []
        for i in range(n - 1):
            ch = self.router.get_channel(node_ids[i], node_ids[i + 1])
            if not ch:
                ch = self.router.create_channel(
                    node_ids[i], node_ids[i + 1], pairs_per_segment
                )
            else:
                self.router.replenish_channel(ch.channel_id, pairs_per_segment)
            segments.append(ch)

        # Step 2: Purify each segment
        purification_log = []
        for seg in segments:
            if self.purification_rounds > 0 and len(seg.usable_pairs) >= 2:
                result = self.router.purify_channel(
                    seg.channel_id, self.purification_rounds
                )
                purification_log.append(result)

        # Step 3: Hierarchical entanglement swapping
        # Binary tree: pair up segments and swap at intermediate relays
        current_level_nodes = list(node_ids)
        level = 0
        swap_log = []

        while len(current_level_nodes) > 2:
            next_level = [current_level_nodes[0]]
            i = 0
            while i < len(current_level_nodes) - 2:
                a = current_level_nodes[i]
                relay = current_level_nodes[i + 1]
                b = current_level_nodes[i + 2]

                pair = self.router.entanglement_swap(a, relay, b)
                if pair:
                    swap_log.append({
                        "level": level,
                        "a": a, "relay": relay, "b": b,
                        "fidelity": pair.current_fidelity,
                    })
                    next_level.append(b)
                    i += 2
                else:
                    # Swap failed, keep intermediate node
                    next_level.append(relay)
                    next_level.append(b)
                    i += 2

            # Add remaining nodes not yet processed
            if i == len(current_level_nodes) - 2:
                next_level.append(current_level_nodes[-1])

            current_level_nodes = next_level
            level += 1

            # Purify after each swap level
            if self.purification_rounds > 0 and len(current_level_nodes) >= 2:
                src, dst = current_level_nodes[0], current_level_nodes[-1]
                ch = self.router.get_channel(src, dst)
                if ch and len(ch.usable_pairs) >= 2:
                    self.router.purify_channel(ch.channel_id, 1)

        # Step 4: Extract final end-to-end pair
        src, dst = node_ids[0], node_ids[-1]
        final_ch = self.router.get_channel(src, dst)
        final_pair = None
        if final_ch:
            final_pair = final_ch.best_pair

        exec_time = (time.time() - t0) * 1000

        result = {
            "success": final_pair is not None and (
                final_pair.current_fidelity >= self.target_fidelity * 0.8
            ) if final_pair else False,
            "final_pair": final_pair.to_dict() if final_pair else None,
            "fidelity": final_pair.current_fidelity if final_pair else 0.0,
            "meets_target": (
                final_pair.current_fidelity >= self.target_fidelity
                if final_pair else False
            ),
            "target_fidelity": self.target_fidelity,
            "hops": n - 1,
            "route": node_ids,
            "levels": level,
            "swaps": swap_log,
            "purifications": purification_log,
            "execution_time_ms": exec_time,
        }

        self._chains_executed.append(result)
        return result

    def estimate_chain_fidelity(self, num_segments: int,
                                 segment_fidelity: float = 0.995,
                                 purification: bool = True) -> float:
        """Estimate the final fidelity of a repeater chain.

        Model: F_final = product of swap fidelities × purification boost

        For n segments with swap fidelity F_swap at each relay:
          F_final ≈ F_segment^n (without purification)
          F_purified ≈ (F^2) / (F^2 + (1-F)^2) per round
        """
        f = segment_fidelity

        # Compute cascaded swap fidelity
        n_swaps = num_segments - 1
        # Each swap: F_swap = F1·F2 + (1-F1)(1-F2)/3
        f_current = f
        for _ in range(n_swaps):
            f_current = f_current * f + (1 - f_current) * (1 - f) / 3

        # Apply purification boost
        if purification and self.purification_rounds > 0:
            for _ in range(self.purification_rounds):
                f_current = (f_current ** 2) / (f_current ** 2 + (1 - f_current) ** 2)

        return f_current

    def status(self) -> Dict:
        return {
            "chains_executed": len(self._chains_executed),
            "successful": sum(1 for c in self._chains_executed if c.get("success")),
            "purification_rounds": self.purification_rounds,
            "target_fidelity": self.target_fidelity,
            "mean_final_fidelity": (
                sum(c["fidelity"] for c in self._chains_executed) / len(self._chains_executed)
                if self._chains_executed else 0.0
            ),
        }
