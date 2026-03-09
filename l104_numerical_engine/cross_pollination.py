"""L104 Numerical Engine — Cross-Pollination Engine.

Synergize inventions across gate builder, link builder, and numerical builder.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F78: 12 gate constants with ±0.01% GOD_CODE guard bounds
  F79: GOD_CODE guard: ±0.01% = ±0.0527518... tolerance
  F83: Consciousness SOVEREIGN × Grover φ³ = φ⁴ cross-builder resonance
"""

import json
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from .precision import D, fmt100
from .constants import (
    GOD_CODE_HP, PHI_HP, WORKSPACE_ROOT,
    GATE_REGISTRY_PATH, LINK_STATE_PATH,
)
from .lattice import TokenLatticeEngine
from .editor import SuperfluidValueEditor
from .models import CrossPollinationRecord


class CrossPollinationEngine:
    """Synergize inventions across gate builder, link builder, and numerical builder."""

    def __init__(self, lattice: TokenLatticeEngine, editor: SuperfluidValueEditor):
        """Initialize CrossPollinationEngine."""
        self.lattice = lattice
        self.editor = editor
        self.records: List[CrossPollinationRecord] = []

    def pollinate_from_gates(self) -> Dict:
        """Ingest gate builder data and cross-pollinate into numerical tokens."""
        results = {"new_tokens": 0, "gates_ingested": 0, "entanglements": 0}

        try:
            if not GATE_REGISTRY_PATH.exists():
                return results
            registry = json.loads(GATE_REGISTRY_PATH.read_text())
            gates = registry.get("gates", [])
        except Exception:
            return results

        for gate in gates:
            name = gate.get("name", "unknown")
            complexity = gate.get("complexity", 0)
            entropy = gate.get("entropy_score", 0.0)
            language = gate.get("language", "python")
            results["gates_ingested"] += 1

            sig_value = GOD_CODE_HP * D(str(max(complexity, 1))) / D(str(max(complexity, 1) + 1))
            sig_value += D(str(entropy)) * PHI_HP

            if complexity > 10:
                margin = D('1E-80')
            elif complexity > 5:
                margin = D('1E-60')
            else:
                margin = D('1E-40')

            token_name = f"GATE_{language}_{name}"
            token_id = f"TOKEN_{token_name}_{hashlib.sha256(name.encode()).hexdigest()[:12]}"

            if token_id not in self.lattice.tokens:
                self.lattice.register_token(
                    name=token_name,
                    value=sig_value,
                    min_bound=sig_value - margin,
                    max_bound=sig_value + margin,
                    origin="cross-pollinated",
                    tier=2,
                )
                results["new_tokens"] += 1

                self.records.append(CrossPollinationRecord(
                    source_builder="gate_builder",
                    target_builder="numerical_builder",
                    invention_type="token",
                    invention_id=token_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    fidelity=min(1.0, complexity / 20.0),
                    details={"gate_name": name, "complexity": complexity, "entropy": entropy},
                ))

        # Entangle tokens from the same source file
        by_file: Dict[str, List[str]] = defaultdict(list)
        for tid, token in self.lattice.tokens.items():
            if token.origin == "cross-pollinated" and token.name.startswith("GATE_"):
                by_file["cross_pollinated"].append(tid)

        for group_tokens in by_file.values():
            for i, tid_a in enumerate(group_tokens[:50]):
                for tid_b in group_tokens[i + 1:i + 4]:
                    if self.editor.entangle_tokens(tid_a, tid_b):
                        results["entanglements"] += 1

        return results

    def pollinate_from_links(self) -> Dict:
        """Ingest quantum link data and cross-pollinate into tokens."""
        results = {"new_tokens": 0, "links_ingested": 0, "entanglements": 0}

        try:
            if not LINK_STATE_PATH.exists():
                return results
            state = json.loads(LINK_STATE_PATH.read_text())
            links = state.get("links", [])
        except Exception:
            return results

        for link in links:
            fidelity = link.get("fidelity", 0.0)
            source = link.get("source_symbol", "?")
            target = link.get("target_symbol", "?")
            link_type = link.get("link_type", "unknown")
            results["links_ingested"] += 1

            link_val = GOD_CODE_HP * D(str(max(fidelity, 0.001)))

            if fidelity > 0.9:
                margin = D('1E-85')
            elif fidelity > 0.5:
                margin = D('1E-60')
            else:
                margin = D('1E-30')

            token_name = f"LINK_{link_type}_{source[:20]}_{target[:20]}"
            token_id = f"TOKEN_{token_name}_{hashlib.sha256(f'{source}{target}'.encode()).hexdigest()[:12]}"

            if token_id not in self.lattice.tokens:
                self.lattice.register_token(
                    name=token_name,
                    value=link_val,
                    min_bound=link_val - margin,
                    max_bound=link_val + margin,
                    origin="cross-pollinated",
                    tier=2,
                )
                results["new_tokens"] += 1

                self.records.append(CrossPollinationRecord(
                    source_builder="link_builder",
                    target_builder="numerical_builder",
                    invention_type="token",
                    invention_id=token_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    fidelity=fidelity,
                    details={"source": source, "target": target, "link_type": link_type},
                ))

        return results

    def pollinate_to_gates(self) -> Dict:
        """Export numerical insights for the gate builder to consume."""
        research_targets = []
        for tid, token in self.lattice.tokens.items():
            drift = abs(D(token.drift_velocity)) if token.drift_velocity else D(0)
            if drift > D('1E-90'):
                research_targets.append({
                    "token_id": tid,
                    "name": token.name,
                    "drift": str(drift),
                    "direction": token.drift_direction,
                    "coherence": token.coherence,
                    "origin": token.origin,
                })

        payload = {
            "source": "quantum_numerical_builder",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lattice_coherence": float(self.lattice.lattice_coherence),
            "lattice_entropy": float(self.lattice.lattice_entropy),
            "total_tokens": len(self.lattice.tokens),
            "research_targets": research_targets[:50],
            "high_drift_count": len(research_targets),
            "mean_token_health": sum(t.health for t in self.lattice.tokens.values()) / max(len(self.lattice.tokens), 1),
            "precision_grade": "100-decimal" if all(t.precision_digits >= 100 for t in self.lattice.tokens.values()) else "mixed",
        }

        out_path = WORKSPACE_ROOT / ".l104_numerical_to_gates.json"
        try:
            out_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            pass

        return payload

    def pollinate_to_links(self) -> Dict:
        """Export numerical insights for the quantum link builder to consume."""
        anchor_tokens = []
        for tid, token in self.lattice.tokens.items():
            if token.origin == "sacred" and token.coherence >= 0.99:
                anchor_tokens.append({
                    "token_id": tid,
                    "name": token.name,
                    "value_preview": token.value[:30],
                    "precision": token.precision_digits,
                    "coherence": token.coherence,
                })

        payload = {
            "source": "quantum_numerical_builder",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "anchor_tokens": anchor_tokens,
            "lattice_coherence": float(self.lattice.lattice_coherence),
            "total_entanglements": sum(
                len(t.entangled_tokens) for t in self.lattice.tokens.values()
            ) // 2,
            "cross_pollination_records": len(self.records),
        }

        out_path = WORKSPACE_ROOT / ".l104_numerical_to_links.json"
        try:
            out_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            pass

        return payload

    def full_cross_pollination(self) -> Dict:
        """Run bidirectional cross-pollination with both builders."""
        results = {}
        results["from_gates"] = self.pollinate_from_gates()
        results["from_links"] = self.pollinate_from_links()
        results["to_gates"] = self.pollinate_to_gates()
        results["to_links"] = self.pollinate_to_links()
        results["total_records"] = len(self.records)
        return results
