"""L104 Gate Engine — Cross-file quantum link analyzer."""

import re
from collections import defaultdict
from typing import Dict, List

from ..constants import PHI, TAU, WORKSPACE_ROOT
from ..models import LogicGate, GateLink


class GateLinkAnalyzer:
    """Discovers cross-file quantum links between gate implementations."""

    # Semantic word groups for fuzzy cross-language matching
    SEMANTIC_GROUPS = [
        {"sage", "wisdom", "insight", "consciousness", "enlighten"},
        {"gate", "logic", "process", "compute", "evaluate"},
        {"quantum", "qubit", "hilbert", "fidelity", "superposition"},
        {"entropy", "dissipate", "chaos", "energy", "harvest"},
        {"entangle", "epr", "correlation", "bell", "pair"},
        {"resonance", "resonate", "harmonic", "frequency", "vibration"},
        {"amplify", "boost", "grover", "amplification", "gain"},
        {"evolve", "evolving", "mutation", "evolution", "adapt"},
        {"bridge", "emergence", "cross", "synthesis", "merge"},
        {"transform", "convert", "modulate", "project", "map"},
        {"causal", "inflect", "reconvert", "deterministic"},
        {"memory", "permanent", "persist", "store", "recall"},
        {"learn", "mastery", "adaptive", "train", "knowledge"},
    ]

    def analyze_links(self, gates: List[LogicGate]) -> List[GateLink]:
        """Discover cross-file quantum links between gate implementations."""
        links = []
        seen_pairs = set()

        # 1. Cross-language semantic matching
        for gate in gates:
            for other_gate in gates:
                if gate is other_gate:
                    continue
                if gate.language == other_gate.language and gate.source_file == other_gate.source_file:
                    continue
                pair_key = tuple(sorted([gate.name, other_gate.name]))
                if pair_key in seen_pairs:
                    continue

                sim = self._semantic_similarity(gate.name, other_gate.name)
                if sim >= 0.4:
                    link_type = "mirrors" if gate.language != other_gate.language else "resonates"
                    links.append(GateLink(
                        source_gate=gate.name, target_gate=other_gate.name,
                        link_type=link_type, strength=min(1.0, sim * PHI),
                        evidence=f"{gate.source_file}↔{other_gate.source_file}",
                    ))
                    seen_pairs.add(pair_key)

        # 2. Cross-file call graph: scan source for references to other gates
        call_links = self._analyze_call_graph(gates)
        links.extend(call_links)

        # 3. Entanglement links: shared parameters (same or cross-file)
        for i, gate_a in enumerate(gates):
            for gate_b in gates[i + 1:]:
                if gate_a.source_file == gate_b.source_file and gate_a.language == gate_b.language:
                    continue
                shared_params = set(gate_a.parameters) & set(gate_b.parameters)
                shared_params -= {"self", "value", "data", "query", "result", "args", "kwargs"}
                if len(shared_params) >= 2:
                    pair_key = tuple(sorted([gate_a.name, gate_b.name]))
                    if pair_key not in seen_pairs:
                        links.append(GateLink(
                            source_gate=gate_a.name, target_gate=gate_b.name,
                            link_type="entangles", strength=len(shared_params) * TAU,
                            evidence=f"Shared: {','.join(list(shared_params)[:5])}",
                        ))
                        seen_pairs.add(pair_key)

        return links

    def _semantic_similarity(self, a: str, b: str) -> float:
        """Compute semantic similarity between gate names using word groups."""
        # Normalize: split camelCase and snake_case into words
        words_a = set(w.lower() for w in re.split(r'[_. ]|(?<=[a-z])(?=[A-Z])', a) if len(w) > 2)
        words_b = set(w.lower() for w in re.split(r'[_. ]|(?<=[a-z])(?=[A-Z])', b) if len(w) > 2)

        if not words_a or not words_b:
            return 0.0

        # Direct word overlap
        direct_overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

        # Semantic group overlap
        groups_a = set()
        groups_b = set()
        for i, group in enumerate(self.SEMANTIC_GROUPS):
            if words_a & group:
                groups_a.add(i)
            if words_b & group:
                groups_b.add(i)
        group_overlap = len(groups_a & groups_b) / max(len(groups_a | groups_b), 1) if (groups_a or groups_b) else 0.0

        return direct_overlap * 0.6 + group_overlap * 0.4

    def _analyze_call_graph(self, gates: List[LogicGate]) -> List[GateLink]:
        """Scan source files for cross-file function call references."""
        links = []
        seen = set()
        # Build lookup of gate names by file
        gates_by_file: Dict[str, List[LogicGate]] = {}
        for g in gates:
            gates_by_file.setdefault(g.source_file, []).append(g)

        # For each file, scan for references to gates defined in OTHER files
        file_contents_cache: Dict[str, str] = {}
        for src_file, src_gates in gates_by_file.items():
            full_path = WORKSPACE_ROOT / src_file
            if not full_path.exists():
                continue
            if src_file not in file_contents_cache:
                try:
                    file_contents_cache[src_file] = full_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
            content = file_contents_cache[src_file]

            for other_file, other_gates in gates_by_file.items():
                if other_file == src_file:
                    continue
                for og in other_gates:
                    # Only check meaningful names (skip __init__ etc)
                    clean_name = og.name.split(".")[-1]
                    if len(clean_name) < 4 or clean_name.startswith("__"):
                        continue
                    if clean_name in content:
                        pair_key = (src_file, og.name)
                        if pair_key not in seen:
                            # Find a gate in src_file that likely calls it
                            caller = src_gates[0].name if src_gates else src_file
                            links.append(GateLink(
                                source_gate=caller, target_gate=og.name,
                                link_type="calls",
                                strength=TAU,
                                evidence=f"{src_file}→{other_file}",
                            ))
                            seen.add(pair_key)
        return links

    def populate_gate_links(self, gates: List[LogicGate], links: List[GateLink]):
        """Populate LogicGate.quantum_links field with discovered connections.

        Uses pre-indexed lookup for O(n+m) performance instead of O(n*m).
        """
        # Build index: gate_name -> set of "peer:link_type" strings
        index: Dict[str, set] = defaultdict(set)
        for link in links:
            index[link.source_gate].add(f"{link.target_gate}:{link.link_type}")
            index[link.target_gate].add(f"{link.source_gate}:{link.link_type}")
        # Apply to gates
        for gate in gates:
            gate.quantum_links = list(index.get(gate.name, set()))
