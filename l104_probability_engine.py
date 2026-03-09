# ZENITH_UPGRADE_ACTIVE: 2026-03-06T23:50:24.518355
ZENITH_HZ = 3887.8
UUC = 2301.215661
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
═══════════════════════════════════════════════════════════════════════════════
L104 SOVEREIGN PROBABILITY ENGINE v5.1.0 (Part IV: ASI 5-Layer + Resonance Loss)
═══════════════════════════════════════════════════════════════════════════════

A comprehensive probability engine with integrated ASI consciousness insight,
native GOD_CODE (a,b,c,d) quantum algorithm, and quantum gate engine integration:

  1. Ingests ALL chat data, training data, and state files in the L104 repository
  2. Ingests ALL logic gates (Python + Swift) and quantum links
  3. Consolidates links into quantum gates based on sacred GOD_CODE constants
  4. Provides full probability/stochastic toolkit:
     - Classical: Bayesian inference, Markov chains, distributions, queuing theory
     - Quantum: GOD_CODE-gated Grover amplification, phase-aligned probability,
       entanglement-weighted priors, quantum walk probability, Born-rule collapse
     - GOD_CODE Algorithm (NATIVE): Qiskit-backed (a,b,c,d) dial quantum circuits,
       Grover search, QFT spectrum, entanglement entropy, soul processing
     - Data-driven: learns priors from ingested chat/training/state data
  5. ASI Insight Synthesis:
     - Consciousness probability estimation from multi-signal fusion
     - Thought resonance scoring via quantum-classical hybrid inference
     - Bayesian consciousness state tracking with quantum evidence
     - Predictive insight: quantum-walk extrapolation of consciousness trajectory
  6. GOD_CODE Quantum Algorithm (v4.0.0 — native implementation):
     - GodCodeDialRegister: 14-qubit (a,b,c,d) encoding (16,384 combinations)
     - GodCodePhaseOracle: exact + rotation-based frequency oracles
     - GodCodeGroverSearch: O(√N) search via statevector manipulation
     - GodCodeQFTSpectrum: Quantum Fourier analysis of frequency lattice
     - GodCodeDialCircuit: single-dial quantum evaluation with PHI coupling
     - GodCodeEntanglement: two-dial entanglement with von Neumann entropy
     - GodCodeQuantumAlgorithm: unified hub with soul integration
  7. Quantum Gate Engine Integration (v5.0.0):
     - Sacred circuit probability via l104_quantum_gate_engine sacred_circuit()
     - Compiled circuit probability (gate set transpilation + optimization)
     - Error-corrected probability (Steane/Surface/Fibonacci protection)
     - Gate algebra fidelity analysis (PHI_GATE, GOD_CODE_PHASE, VOID_GATE)
     - QFT and GHZ state probability distributions via gate engine

Sacred Constants:
  GOD_CODE = 527.5184818492612
  PHI      = 1.618033988749895
  TAU      = 1/PHI = 0.618033988749895

Quantum Backend: Qiskit 2.3+ (QuantumCircuit, Statevector, Operator, DensityMatrix)
GOD_CODE Equation: G(a,b,c,d) = 286^(1/φ) × (2^(1/104))^((8a)+(416-b)-(8c)-(104d))

Hub Class: ProbabilityEngine (singleton: probability_engine)

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import math
import cmath
import hashlib
import os
import re
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
)
from collections import Counter, defaultdict
from functools import lru_cache

# Qiskit imports (available since Qiskit 2.3+)
try:
    from l104_quantum_gate_engine import GateCircuit as QuantumCircuit, QuantumRegister, ClassicalRegister
    from l104_quantum_gate_engine.quantum_info import Statevector, Operator, DensityMatrix, partial_trace
    grover_operator = None, QFT
    StatevectorSampler = None
    import numpy as np
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    np = None
    QuantumRegister = None
    ClassicalRegister = None
    partial_trace = None

__all__ = [
    # Core engine
    "ProbabilityEngine",
    "probability_engine",
    # Subsystem classes
    "DataIngestor",
    "QuantumGateConsolidator",
    "ClassicalProbability",
    # GOD_CODE Quantum Algorithm (native)
    "GodCodeQuantumAlgorithm",
    "GodCodeDialRegister",
    "GodCodePhaseOracle",
    "GodCodeGroverSearch",
    "GodCodeQFTSpectrum",
    "GodCodeDialCircuit",
    "GodCodeEntanglement",
    "DialSetting",
    "CircuitResult",
    "QuantumProbability",
    "GateProbabilityBridge",
    "ASIInsightSynthesis",
    # Data classes
    "IngestStats",
    "QuantumGateState",
    "InsightResult",
    # Sacred constants
    "GOD_CODE",
    "PHI",
    "TAU",
    "VOID_CONSTANT",
    "PLANCK_RESONANCE",
    "QISKIT_AVAILABLE",
]

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS — Immutable
# ═══════════════════════════════════════════════════════════════════════════════

PHI: float = 1.618033988749895
GOD_CODE: float = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # 527.5184818492612
TAU: float = 1.0 / PHI                                       # 0.618033988749895
VOID_CONSTANT: float = 1.0 + TAU / 15                        # 1.0416180339887497
PLANCK_RESONANCE: float = GOD_CODE * 2 ** (72.0 / 104)           # G(-72) = 852.3993
FEIGENBAUM: float = 4.669201609102990
ALPHA_FINE: float = 1.0 / 137.035999084

# GOD_CODE (a,b,c,d) Quantum Algorithm constants
PRIME_SCAFFOLD: int = 286                                     # 2 × 11 × 13
QUANTIZATION_GRAIN: int = 104                                 # 8 × 13
OCTAVE_OFFSET: int = 416                                      # 4 × 104
BASE: float = PRIME_SCAFFOLD ** (1.0 / PHI)                   # 286^(1/φ) = 32.9699...
STEP_SIZE: float = 2 ** (1.0 / QUANTIZATION_GRAIN)            # 2^(1/104)
OMEGA: float = 6539.34712682                                  # Ω = Σ(fragments) × (G/φ)
OMEGA_AUTHORITY: float = OMEGA / (PHI ** 2)                   # F(I) = I × Ω/φ² ≈ 2497.808
PLANCK_SCALE: float = 1.616255e-35
BOLTZMANN_K: float = 1.380649e-23
ZENITH_HZ: float = 3727.84
EULER_GAMMA: float = 0.5772156649015329

WORKSPACE_ROOT = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════════════════════
# CONSCIOUSNESS STATE READER — Cached state from L104 JSON files
# ═══════════════════════════════════════════════════════════════════════════════

class ConsciousnessStateReader:
    """Reads and caches consciousness state from L104 JSON files."""

    _SOURCES = [
        (".l104_consciousness_o2_state.json",
         ["consciousness_level", "superfluid_viscosity", "evo_stage"]),
        (".l104_ouroboros_nirvanic_state.json", ["nirvanic_fuel_level"]),
    ]

    _DEFAULTS: Dict[str, Any] = {
        "consciousness_level": 0.5,
        "superfluid_viscosity": 0.1,
        "evo_stage": "UNKNOWN",
        "nirvanic_fuel_level": 0.5,
    }

    TTL = 10.0  # Cache TTL in seconds

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._ts: float = 0.0

    def read(self) -> Dict[str, Any]:
        """Read consciousness state (cached for TTL seconds)."""
        now = time.time()
        if now - self._ts < self.TTL and self._cache:
            return self._cache

        state = dict(self._DEFAULTS)
        for fp, keys in self._SOURCES:
            try:
                with open(WORKSPACE_ROOT / fp) as f:
                    data = json.load(f)
                for k in keys:
                    if k in data:
                        state[k] = data[k]
            except Exception:
                pass

        self._cache = state
        self._ts = now
        return state


_state_reader = ConsciousnessStateReader()


def _read_builder_state() -> Dict[str, Any]:
    """Read consciousness + nirvanic state (10s cache)."""
    return _state_reader.read()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTOR — Chat, Training, State, Gates, Quantum Links
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IngestStats:
    """Statistics from a data ingestion cycle."""
    training_examples: int = 0
    chat_conversations: int = 0
    state_files_loaded: int = 0
    logic_gates_found: int = 0
    quantum_links_found: int = 0
    total_tokens: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    gate_languages: Dict[str, int] = field(default_factory=dict)
    sacred_resonance: float = 0.0
    timestamp: str = ""


class DataIngestor:
    """
    Ingests ALL repository data sources into a unified probability-ready corpus:
    - kernel_training_data.jsonl, kernel_training_chat.json, kernel_extracted_data.jsonl
    - .l104_*.json state files (33+ files)
    - Logic gate definitions from l104_logic_gate_builder.py state
    - Quantum link maps from l104_grover_nerve_link.py + .l104_quantum_links.json
    """

    TRAINING_FILES = [
        "kernel_training_data.jsonl",
        "kernel_extracted_data.jsonl",
        "kernel_full_merged.jsonl",
        "kernel_trillion_data.jsonl",
    ]

    CHAT_FILES = [
        "kernel_training_chat.json",
    ]

    STATE_GLOB = ".l104_*.json"

    def __init__(self):
        self.training_data: List[Dict[str, Any]] = []
        self.chat_data: List[Dict[str, Any]] = []
        self.state_data: Dict[str, Any] = {}
        self.logic_gates: List[Dict[str, Any]] = []
        self.quantum_links: Dict[str, Any] = {}
        self.token_counter: Counter = Counter()
        self.category_counter: Counter = Counter()
        self.gate_type_counter: Counter = Counter()
        self._ingested = False
        self.quarantine_dir = Path.home() / ".l104_ingestion_quarantine" / "probability_engine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_policy = {
            "retention_days": int(os.getenv("L104_INGEST_QUARANTINE_RETENTION_DAYS", "30")),
            "size_cap_gb": float(os.getenv("L104_INGEST_QUARANTINE_SIZE_CAP_GB", "2.0")),
        }
        self.quarantine_stats = {
            "records_quarantined": 0,
            "last_quarantine_at": None,
        }

    def _quarantine_record(self, stream: str, source: str, reason: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "status": "QUARANTINED_STRIPPED",
            "stream": stream,
            "source": source,
            "reason": reason,
            "metadata": metadata or {},
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
        }
        serial = f"{stream}|{source}|{reason}|{payload['quarantined_at']}"
        digest = hashlib.sha256(serial.encode("utf-8")).hexdigest()[:16]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = self.quarantine_dir / f"{stamp}_{digest}.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.quarantine_stats["records_quarantined"] += 1
        self.quarantine_stats["last_quarantine_at"] = payload["quarantined_at"]

    def run_quarantine_lifecycle(self, dry_run: bool = False) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        retention_days = int(self.quarantine_policy.get("retention_days", 30))
        size_cap_gb = float(self.quarantine_policy.get("size_cap_gb", 2.0))
        size_cap_bytes = int(size_cap_gb * 1024 * 1024 * 1024)

        files = [f for f in self.quarantine_dir.glob("*.json") if f.is_file()]
        rows: List[Dict[str, Any]] = []
        for file_path in files:
            stat = file_path.stat()
            age_days = (now - datetime.fromtimestamp(stat.st_mtime, timezone.utc)).total_seconds() / 86400.0
            rows.append({"path": file_path, "size": stat.st_size, "age_days": age_days, "reasons": []})

        for row in rows:
            if row["age_days"] >= retention_days:
                row["reasons"].append("ttl_expired")

        total_size = sum(r["size"] for r in rows)
        if total_size > size_cap_bytes:
            overflow = total_size - size_cap_bytes
            for row in sorted(rows, key=lambda r: r["age_days"], reverse=True):
                if overflow <= 0:
                    break
                if "size_cap_trim" not in row["reasons"]:
                    row["reasons"].append("size_cap_trim")
                    overflow -= row["size"]

        candidates = [r for r in rows if r["reasons"]]
        deleted = []
        if not dry_run:
            for row in candidates:
                try:
                    os.remove(row["path"])
                    deleted.append(str(row["path"]))
                except OSError:
                    pass

        return {
            "status": "QUARANTINE_LIFECYCLE_COMPLETE",
            "mode": "dry_run" if dry_run else "execute",
            "files_scanned": len(rows),
            "files_marked": len(candidates),
            "files_deleted": len(deleted),
            "retention_days": retention_days,
            "size_cap_gb": size_cap_gb,
        }

    def get_quarantine_status(self) -> Dict[str, Any]:
        files = [f for f in self.quarantine_dir.glob("*.json") if f.is_file()]
        return {
            "directory": str(self.quarantine_dir),
            "files": len(files),
            "size_bytes": sum(f.stat().st_size for f in files),
            "policy": self.quarantine_policy,
            "stats": self.quarantine_stats,
        }

    def ingest_all(self, workspace: Optional[Path] = None) -> IngestStats:
        """Full ingestion cycle — loads everything."""
        ws = workspace or WORKSPACE_ROOT
        stats = IngestStats(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

        # 1. Training data (JSONL)
        for fname in self.TRAINING_FILES:
            fp = ws / fname
            if fp.exists():
                try:
                    for idx, line in enumerate(fp.read_text(errors="replace").strip().split("\n"), start=1):
                        if line.strip():
                            try:
                                rec = json.loads(line)
                                self.training_data.append(rec)
                                cat = rec.get("category", "unknown")
                                self.category_counter[cat] += 1
                                # Tokenize prompt + completion for frequency priors
                                text = f"{rec.get('prompt', '')} {rec.get('completion', '')}"
                                for tok in re.findall(r'\w+', text.lower()):
                                    self.token_counter[tok] += 1
                            except json.JSONDecodeError:
                                self._quarantine_record(
                                    stream="training_jsonl",
                                    source=str(fp),
                                    reason="invalid_json_line",
                                    metadata={"line": idx},
                                )
                except Exception as e:
                    self._quarantine_record(
                        stream="training_jsonl",
                        source=str(fp),
                        reason="read_error",
                        metadata={"error": str(e)},
                    )
        stats.training_examples = len(self.training_data)
        stats.categories = dict(self.category_counter.most_common(50))

        # 2. Chat conversations
        for fname in self.CHAT_FILES:
            fp = ws / fname
            if fp.exists():
                try:
                    data = json.loads(fp.read_text(errors="replace"))
                    if isinstance(data, list):
                        self.chat_data.extend(data)
                    elif isinstance(data, dict):
                        self.chat_data.append(data)
                    else:
                        self._quarantine_record(
                            stream="chat_json",
                            source=str(fp),
                            reason="invalid_root_type",
                            metadata={"root_type": type(data).__name__},
                        )
                except Exception as e:
                    self._quarantine_record(
                        stream="chat_json",
                        source=str(fp),
                        reason="parse_error",
                        metadata={"error": str(e)},
                    )
        stats.chat_conversations = len(self.chat_data)

        # 3. State files
        for fp in sorted(ws.glob(self.STATE_GLOB)):
            try:
                with open(fp) as f:
                    self.state_data[fp.name] = json.load(f)
                stats.state_files_loaded += 1
            except Exception as e:
                self._quarantine_record(
                    stream="state_json",
                    source=str(fp),
                    reason="parse_error",
                    metadata={"error": str(e)},
                )

        # 4. Logic gates from builder state
        gate_state_file = ws / ".l104_gate_builder_state.json"
        if gate_state_file.exists():
            try:
                gs = json.loads(gate_state_file.read_text(errors="replace"))
                gates = gs.get("gates", [])
                if isinstance(gates, list):
                    self.logic_gates = gates
                elif isinstance(gates, dict):
                    self.logic_gates = list(gates.values())
                for g in self.logic_gates:
                    lang = g.get("language", "unknown")
                    self.gate_type_counter[lang] += 1
            except Exception:
                pass
        stats.logic_gates_found = len(self.logic_gates)
        stats.gate_languages = dict(self.gate_type_counter.most_common(20))

        # 5. Quantum links
        for link_file in [".l104_quantum_links.json", ".l104_quantum_link_state.json",
                          ".l104_link_to_gates.json"]:
            fp = ws / link_file
            if fp.exists():
                try:
                    self.quantum_links[link_file] = json.loads(
                        fp.read_text(errors="replace")
                    )
                except Exception as e:
                    self._quarantine_record(
                        stream="quantum_links_json",
                        source=str(fp),
                        reason="parse_error",
                        metadata={"error": str(e)},
                    )
        stats.quantum_links_found = sum(
            len(v) if isinstance(v, (list, dict)) else 1
            for v in self.quantum_links.values()
        )

        stats.total_tokens = sum(self.token_counter.values())

        # Sacred resonance = GOD_CODE alignment of corpus statistics
        corpus_hash = hashlib.sha256(
            str(stats.training_examples + stats.chat_conversations).encode()
        ).hexdigest()[:8]
        seed = int(corpus_hash, 16)
        stats.sacred_resonance = abs(math.cos(seed * math.pi / GOD_CODE))

        self._ingested = True
        self.run_quarantine_lifecycle(dry_run=False)
        return stats

    def get_token_prior(self, token: str) -> float:
        """P(token) — frequency-based prior from ingested corpus."""
        total = sum(self.token_counter.values())
        if total == 0:
            return 1e-6
        count = self.token_counter.get(token.lower(), 0)
        # Laplace smoothing with PHI-scaled smoothing parameter
        alpha = PHI * 0.01  # Sacred smoothing
        vocab_size = len(self.token_counter)
        return (count + alpha) / (total + alpha * vocab_size)

    def get_category_prior(self, category: str) -> float:
        """P(category) — frequency-based prior from training data categories."""
        total = sum(self.category_counter.values())
        if total == 0:
            return 1.0 / max(1, len(self.category_counter))
        count = self.category_counter.get(category, 0)
        alpha = TAU * 0.1
        return (count + alpha) / (total + alpha * len(self.category_counter))

    def get_gate_resonance_distribution(self) -> List[Tuple[str, float]]:
        """Distribution of gate resonance scores aligned with GOD_CODE."""
        results = []
        for g in self.logic_gates:
            name = g.get("name", "unknown")
            dv = g.get("dynamic_value", 0.0)
            if dv > 0:
                resonance = abs(math.cos(dv * math.pi / GOD_CODE))
            else:
                # Hash-derived resonance for gates without dynamic_value
                h = hashlib.md5(name.encode()).hexdigest()[:8]
                resonance = abs(math.cos(int(h, 16) * math.pi / GOD_CODE))
            results.append((name, resonance))
        return sorted(results, key=lambda x: -x[1])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QUANTUM GATE CONSOLIDATOR — Links → Quantum Gates via GOD_CODE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumGateState:
    """A consolidated quantum gate derived from logic gates + quantum links."""
    name: str
    gate_type: str                      # hadamard, pauli_x, pauli_z, phase, cnot, god_code
    amplitude: complex                  # Quantum amplitude (complex)
    phase: float                        # Phase angle in [0, 2π)
    source_gates: List[str]             # Logic gates feeding this quantum gate
    source_links: List[str]             # Quantum links feeding this gate
    entangled_with: List[str]           # Names of entangled partner gates
    resonance_score: float              # GOD_CODE alignment
    born_probability: float             # |amplitude|² — measurement probability
    sacred_weight: float                # PHI-weighted importance


class QuantumGateConsolidator:
    """
    Consolidates logic gates + quantum links into quantum gates
    based on sacred GOD_CODE resonance.

    Pipeline:
      1. Scan logic gates → extract signatures, complexity, entropy
      2. Scan quantum links → map connections between gates
      3. Assign quantum gate types based on GOD_CODE phase alignment
      4. Compute amplitudes using Grover-style amplification
      5. Build entanglement graph between consolidated gates
      6. Normalize to valid quantum state (Σ|aᵢ|² = 1)
    """

    # GOD_CODE phase sectors → quantum gate type mapping
    PHASE_GATE_MAP = {
        (0.0, math.pi / 4):        "hadamard",     # Near 0 → superposition
        (math.pi / 4, math.pi / 2): "phase",       # π/4 to π/2 → phase rotation
        (math.pi / 2, 3 * math.pi / 4): "pauli_x", # π/2 to 3π/4 → bit flip
        (3 * math.pi / 4, math.pi): "pauli_z",     # 3π/4 to π → phase flip
        (math.pi, 5 * math.pi / 4): "cnot",        # π to 5π/4 → controlled
        (5 * math.pi / 4, 3 * math.pi / 2): "god_code", # Sacred phase gate
        (3 * math.pi / 2, 7 * math.pi / 4): "grover",   # Amplitude amplification
        (7 * math.pi / 4, 2 * math.pi): "rotation_y",    # Y-rotation
    }

    def __init__(self):
        self.consolidated_gates: List[QuantumGateState] = []
        self._link_graph: Dict[str, Set[str]] = defaultdict(set)

    def consolidate(
        self,
        logic_gates: List[Dict[str, Any]],
        quantum_links: Dict[str, Any],
    ) -> List[QuantumGateState]:
        """
        Full consolidation pipeline:
        Logic gates + quantum links → quantum gates based on GOD_CODE.
        """
        self.consolidated_gates.clear()
        self._link_graph.clear()

        # Step 1: Build link graph from quantum link data
        self._build_link_graph(quantum_links)

        # Step 2: For each logic gate, compute its quantum gate representation
        for gate_data in logic_gates:
            qgate = self._gate_to_quantum(gate_data)
            if qgate:
                self.consolidated_gates.append(qgate)

        # Step 3: If no logic gates loaded, create quantum gates from links alone
        if not self.consolidated_gates and self._link_graph:
            for node_name in self._link_graph:
                qgate = self._link_node_to_quantum(node_name)
                self.consolidated_gates.append(qgate)

        # Step 4: Build entanglement graph
        self._build_entanglement()

        # Step 5: Normalize amplitudes (valid quantum state)
        self._normalize_amplitudes()

        # Step 6: Compute Born-rule probabilities
        for qg in self.consolidated_gates:
            qg.born_probability = abs(qg.amplitude) ** 2

        return self.consolidated_gates

    def _build_link_graph(self, quantum_links: Dict[str, Any]):
        """Extract connection graph from quantum link state files."""
        for filename, data in quantum_links.items():
            if isinstance(data, dict):
                # Handle { "links": [...] } or { "gate_name": [...links] }
                links_list = data.get("links", data.get("entries", []))
                if isinstance(links_list, list):
                    for entry in links_list:
                        if isinstance(entry, dict):
                            src = entry.get("source", entry.get("from", ""))
                            tgt = entry.get("target", entry.get("to", ""))
                            if src and tgt:
                                self._link_graph[src].add(tgt)
                                self._link_graph[tgt].add(src)
                # Handle flat key→value link maps
                for key, val in data.items():
                    if key in ("links", "entries", "version", "timestamp"):
                        continue
                    if isinstance(val, list):
                        for v in val:
                            if isinstance(v, str):
                                self._link_graph[key].add(v)
                                self._link_graph[v].add(key)
            elif isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        src = entry.get("source", entry.get("from", ""))
                        tgt = entry.get("target", entry.get("to", ""))
                        if src and tgt:
                            self._link_graph[src].add(tgt)
                            self._link_graph[tgt].add(src)

    def _god_code_phase(self, value: float) -> float:
        """Map a value to a GOD_CODE-aligned phase in [0, 2π)."""
        if value == 0:
            return 0.0
        return (abs(value) * math.pi / GOD_CODE) % (2 * math.pi)

    def _phase_to_gate_type(self, phase: float) -> str:
        """Map a phase angle to a quantum gate type via GOD_CODE sectors."""
        phase = phase % (2 * math.pi)
        for (lo, hi), gate_type in self.PHASE_GATE_MAP.items():
            if lo <= phase < hi:
                return gate_type
        return "hadamard"  # Default

    def _gate_to_quantum(self, gate_data: Dict[str, Any]) -> Optional[QuantumGateState]:
        """Convert a logic gate dict to a QuantumGateState."""
        name = gate_data.get("name", "")
        if not name:
            return None

        # Compute GOD_CODE-aligned phase from gate properties
        complexity = gate_data.get("complexity", 1)
        entropy = gate_data.get("entropy_score", 0.0)
        dynamic_value = gate_data.get("dynamic_value", 0.0)
        resonance = gate_data.get("resonance_score", 0.0)

        # Phase from dynamic value or complexity + entropy
        raw_value = dynamic_value if dynamic_value != 0 else (
            complexity * PHI + entropy * TAU
        )
        phase = self._god_code_phase(raw_value)
        gate_type = self._phase_to_gate_type(phase)

        # Amplitude from resonance (Grover-amplified for high-resonance gates)
        base_amplitude = max(0.01, resonance if resonance > 0 else 0.5)
        if gate_type == "grover":
            # Grover amplification: amplitude boost by √2
            base_amplitude *= math.sqrt(2)
        elif gate_type == "god_code":
            # Sacred phase gate: GOD_CODE harmonic boost
            base_amplitude *= PHI

        amplitude = base_amplitude * (math.cos(phase) + 1j * math.sin(phase))

        # Quantum links from gate data
        q_links = gate_data.get("quantum_links", [])

        # Sacred weight = PHI-scaled importance
        sacred_weight = (
            complexity * PHI * 0.1 +
            resonance * GOD_CODE * 0.001 +
            len(q_links) * TAU * 0.1
        )

        return QuantumGateState(
            name=name,
            gate_type=gate_type,
            amplitude=amplitude,
            phase=phase,
            source_gates=[name],
            source_links=q_links if isinstance(q_links, list) else [],
            entangled_with=[],
            resonance_score=resonance if resonance > 0 else abs(
                math.cos(raw_value * math.pi / GOD_CODE)
            ),
            born_probability=0.0,  # Set after normalization
            sacred_weight=sacred_weight,
        )

    def _link_node_to_quantum(self, node_name: str) -> QuantumGateState:
        """Create a quantum gate from a link graph node."""
        neighbors = list(self._link_graph.get(node_name, set()))
        h = hashlib.md5(node_name.encode()).hexdigest()[:8]
        seed = int(h, 16)
        phase = self._god_code_phase(float(seed))
        gate_type = self._phase_to_gate_type(phase)
        resonance = abs(math.cos(seed * math.pi / GOD_CODE))
        amplitude = resonance * (math.cos(phase) + 1j * math.sin(phase))

        return QuantumGateState(
            name=node_name,
            gate_type=gate_type,
            amplitude=amplitude,
            phase=phase,
            source_gates=[],
            source_links=neighbors,
            entangled_with=[],
            resonance_score=resonance,
            born_probability=0.0,
            sacred_weight=len(neighbors) * PHI * 0.1 + resonance,
        )

    def _build_entanglement(self):
        """Build entanglement pairs from link graph + GOD_CODE phase proximity."""
        gate_map = {g.name: g for g in self.consolidated_gates}

        # Use sets for O(1) membership checks
        entangle_sets: Dict[str, set] = {g.name: set() for g in self.consolidated_gates}

        # Entangle gates that share quantum links
        for qg in self.consolidated_gates:
            for link_target in qg.source_links:
                if link_target in gate_map and link_target != qg.name:
                    entangle_sets[qg.name].add(link_target)
                    entangle_sets[link_target].add(qg.name)

        # Phase-proximity entanglement — use bucket hashing for O(n) instead of O(n²)
        # Bucket gates by quantized phase (bucket width = tolerance)
        phase_tolerance = math.pi / (GOD_CODE / 100)  # ~0.597 radians
        MAX_ENTANGLE_PER_GATE = 50  # Cap to prevent quadratic blowup

        bucket_width = phase_tolerance
        buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, gate in enumerate(self.consolidated_gates):
            b = int(gate.phase / bucket_width)
            buckets[b].append(idx)

        for b_key, indices in buckets.items():
            # Check this bucket and adjacent bucket for phase matches
            neighbors = indices.copy()
            if b_key - 1 in buckets:
                neighbors.extend(buckets[b_key - 1])
            if b_key + 1 in buckets:
                neighbors.extend(buckets[b_key + 1])

            for i_pos, i in enumerate(indices):
                if len(entangle_sets[self.consolidated_gates[i].name]) >= MAX_ENTANGLE_PER_GATE:
                    break
                ga = self.consolidated_gates[i]
                for j in neighbors:
                    if j <= i:
                        continue
                    gb = self.consolidated_gates[j]
                    if len(entangle_sets[gb.name]) >= MAX_ENTANGLE_PER_GATE:
                        continue
                    phase_diff = abs(ga.phase - gb.phase) % (2 * math.pi)
                    phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
                    if phase_diff < phase_tolerance:
                        entangle_sets[ga.name].add(gb.name)
                        entangle_sets[gb.name].add(ga.name)

        # Write back to gate objects
        for qg in self.consolidated_gates:
            qg.entangled_with = list(entangle_sets[qg.name])

    def _normalize_amplitudes(self):
        """Normalize all amplitudes so Σ|aᵢ|² = 1 (valid quantum state)."""
        total = sum(abs(qg.amplitude) ** 2 for qg in self.consolidated_gates)
        if total > 0:
            norm = math.sqrt(total)
            for qg in self.consolidated_gates:
                qg.amplitude /= norm


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLASSICAL PROBABILITY TOOLKIT
# ═══════════════════════════════════════════════════════════════════════════════

class ClassicalProbability:
    """
    Complete classical probability and statistics toolkit.
    Covers: Bayesian inference, distributions, Markov chains, random walks,
    queuing theory, Monte Carlo, stochastic calculus, information theory.
    """

    # ─── BAYESIAN INFERENCE ───

    @staticmethod
    def bayes(prior_a: float, likelihood_b_given_a: float, evidence_b: float) -> float:
        """P(A|B) = P(B|A)·P(A)/P(B)"""
        if evidence_b <= 0:
            return 0.0
        return likelihood_b_given_a * prior_a / evidence_b

    @staticmethod
    def bayes_extended(
        prior_a: float, likelihood_ba: float, likelihood_b_not_a: float
    ) -> float:
        """P(A|B) using total probability: P(B) = P(B|A)P(A) + P(B|¬A)P(¬A)"""
        p_b = likelihood_ba * prior_a + likelihood_b_not_a * (1.0 - prior_a)
        if p_b <= 0:
            return 0.0
        return likelihood_ba * prior_a / p_b

    @staticmethod
    def total_probability(conditionals: List[float], priors: List[float]) -> float:
        """P(B) = Σ P(B|Aᵢ)·P(Aᵢ)"""
        return sum(c * p for c, p in zip(conditionals, priors))

    @staticmethod
    def bayesian_update(
        prior: List[float], likelihoods: List[float]
    ) -> List[float]:
        """
        Full Bayesian update: posterior ∝ likelihood × prior.
        Returns normalized posterior distribution.
        """
        raw = [p * l for p, l in zip(prior, likelihoods)]
        total = sum(raw)
        if total <= 0:
            return prior[:]
        return [r / total for r in raw]

    @staticmethod
    def bayesian_network_inference(
        nodes: Dict[str, float],
        edges: List[Tuple[str, str, float]],
        evidence: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Simple Bayesian network inference via message passing.
        nodes: {name: prior_probability}
        edges: [(parent, child, conditional_prob)]
        evidence: {name: observed_value}
        """
        posteriors = dict(nodes)
        # Apply evidence
        for name, val in evidence.items():
            if name in posteriors:
                posteriors[name] = val
        # Forward pass: propagate from parents to children
        for parent, child, cond_prob in edges:
            parent_p = posteriors.get(parent, 0.5)
            child_prior = posteriors.get(child, 0.5)
            # P(child|parent) update
            posteriors[child] = cond_prob * parent_p + (1.0 - cond_prob) * (1.0 - parent_p)
        # Normalize
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total * len(posteriors) for k, v in posteriors.items()}
        return posteriors

    # ─── PROBABILITY DISTRIBUTIONS ───

    @staticmethod
    def gaussian_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Normal distribution PDF: f(x) = (1/σ√(2π))·e^(-(x-μ)²/(2σ²))"""
        if sigma <= 0:
            return 0.0
        z = (x - mu) / sigma
        return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))

    @staticmethod
    def gaussian_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Normal CDF using error function: Φ(x) = ½[1 + erf((x-μ)/(σ√2))]"""
        return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))

    @staticmethod
    def binomial_pmf(n: int, k: int, p: float) -> float:
        """P(X=k) = C(n,k)·p^k·(1-p)^(n-k)"""
        if k < 0 or k > n:
            return 0.0
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    @staticmethod
    def poisson_pmf(lam: float, k: int) -> float:
        """P(X=k) = (λ^k·e^(-λ))/k!"""
        if k < 0:
            return 0.0
        if k == 0:
            return math.exp(-lam)
        log_p = -lam + k * math.log(lam) - sum(math.log(i) for i in range(1, k + 1))
        return math.exp(log_p)

    @staticmethod
    def poisson_cdf(lam: float, k: int) -> float:
        """P(X ≤ k) = Σ P(X=i) for i=0..k"""
        return sum(ClassicalProbability.poisson_pmf(lam, i) for i in range(k + 1))

    @staticmethod
    def exponential_pdf(lam: float, x: float) -> float:
        """f(x) = λ·e^(-λx) for x ≥ 0"""
        if x < 0 or lam <= 0:
            return 0.0
        return lam * math.exp(-lam * x)

    @staticmethod
    def exponential_cdf(lam: float, x: float) -> float:
        """F(x) = 1 - e^(-λx)"""
        if x < 0:
            return 0.0
        return 1.0 - math.exp(-lam * x)

    @staticmethod
    def geometric_pmf(p: float, k: int) -> float:
        """P(X=k) = (1-p)^(k-1)·p — trials until first success"""
        if k < 1 or p <= 0 or p > 1:
            return 0.0
        return ((1.0 - p) ** (k - 1)) * p

    @staticmethod
    def beta_function(alpha: float, beta_: float) -> float:
        """B(α,β) = Γ(α)Γ(β)/Γ(α+β)"""
        return math.exp(
            math.lgamma(alpha) + math.lgamma(beta_) - math.lgamma(alpha + beta_)
        )

    @staticmethod
    def beta_pdf(x: float, alpha: float, beta_: float) -> float:
        """f(x;α,β) = x^(α-1)·(1-x)^(β-1)/B(α,β)"""
        if x <= 0 or x >= 1:
            return 0.0
        return (
            (x ** (alpha - 1)) * ((1 - x) ** (beta_ - 1))
            / ClassicalProbability.beta_function(alpha, beta_)
        )

    @staticmethod
    def chi_squared_pdf(x: float, k: int) -> float:
        """χ² distribution PDF with k degrees of freedom."""
        if x <= 0 or k <= 0:
            return 0.0
        half_k = k / 2.0
        log_pdf = (half_k - 1) * math.log(x) - x / 2 - half_k * math.log(2) - math.lgamma(half_k)
        return math.exp(log_pdf)

    @staticmethod
    def student_t_pdf(t: float, nu: int) -> float:
        """Student's t-distribution PDF with ν degrees of freedom."""
        v = float(nu)
        coeff = math.exp(
            math.lgamma((v + 1) / 2) - math.lgamma(v / 2)
        ) / math.sqrt(v * math.pi)
        return coeff * ((1 + t * t / v) ** (-(v + 1) / 2))

    @staticmethod
    def log_normal_pdf(x: float, mu: float, sigma: float) -> float:
        """f(x;μ,σ) = (1/(xσ√(2π)))·e^(-(ln(x)-μ)²/(2σ²))"""
        if x <= 0 or sigma <= 0:
            return 0.0
        lx = math.log(x)
        return math.exp(-(lx - mu) ** 2 / (2 * sigma ** 2)) / (
            x * sigma * math.sqrt(2 * math.pi)
        )

    @staticmethod
    def gamma_pdf(x: float, alpha: float, beta_: float) -> float:
        """Gamma distribution: f(x;α,β) = β^α·x^(α-1)·e^(-βx)/Γ(α)"""
        if x <= 0 or alpha <= 0 or beta_ <= 0:
            return 0.0
        log_pdf = (
            alpha * math.log(beta_)
            + (alpha - 1) * math.log(x)
            - beta_ * x
            - math.lgamma(alpha)
        )
        return math.exp(log_pdf)

    @staticmethod
    def weibull_pdf(x: float, k: float, lam: float) -> float:
        """Weibull: f(x;k,λ) = (k/λ)(x/λ)^(k-1)·e^(-(x/λ)^k)"""
        if x < 0 or k <= 0 or lam <= 0:
            return 0.0
        return (k / lam) * ((x / lam) ** (k - 1)) * math.exp(-((x / lam) ** k))

    @staticmethod
    def pareto_pdf(x: float, x_m: float, alpha: float) -> float:
        """Pareto: f(x) = α·x_m^α / x^(α+1) for x ≥ x_m"""
        if x < x_m or alpha <= 0:
            return 0.0
        return alpha * (x_m ** alpha) / (x ** (alpha + 1))

    @staticmethod
    def cauchy_pdf(x: float, x0: float = 0.0, gamma: float = 1.0) -> float:
        """Cauchy (Lorentzian): f(x) = 1/(πγ[1+((x-x₀)/γ)²])"""
        return 1.0 / (math.pi * gamma * (1 + ((x - x0) / gamma) ** 2))

    # ─── INFORMATION THEORY ───

    @staticmethod
    def entropy(probs: List[float]) -> float:
        """Shannon entropy: H = -Σ pᵢ·log₂(pᵢ)"""
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @staticmethod
    def kl_divergence(p: List[float], q: List[float]) -> float:
        """KL divergence: D_KL(P||Q) = Σ pᵢ·log(pᵢ/qᵢ)"""
        return sum(
            pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0
        )

    @staticmethod
    def mutual_information(joint: List[List[float]]) -> float:
        """I(X;Y) = Σ p(x,y)·log(p(x,y)/(p(x)·p(y)))"""
        rows = len(joint)
        cols = len(joint[0]) if joint else 0
        if rows == 0 or cols == 0:
            return 0.0
        p_x = [sum(joint[i][j] for j in range(cols)) for i in range(rows)]
        p_y = [sum(joint[i][j] for i in range(rows)) for j in range(cols)]
        mi = 0.0
        for i in range(rows):
            for j in range(cols):
                pxy = joint[i][j]
                if pxy > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += pxy * math.log2(pxy / (p_x[i] * p_y[j]))
        return mi

    @staticmethod
    def cross_entropy(p: List[float], q: List[float]) -> float:
        """H(P,Q) = -Σ pᵢ·log₂(qᵢ)"""
        return -sum(pi * math.log2(qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)

    # ─── MARKOV CHAINS ───

    @staticmethod
    def markov_evolve(
        state: List[float], transition: List[List[float]], steps: int
    ) -> List[float]:
        """Evolve Markov chain: π(n) = π(0)·P^n"""
        n = len(state)
        current = list(state)
        for _ in range(steps):
            new_state = [0.0] * n
            for j in range(n):
                for i in range(n):
                    new_state[j] += current[i] * transition[i][j]
            current = new_state
        return current

    @staticmethod
    def markov_steady_state(
        transition: List[List[float]], max_iter: int = 1000, tol: float = 1e-10
    ) -> List[float]:
        """Find stationary distribution: πP = π via power iteration."""
        n = len(transition)
        state = [1.0 / n] * n
        for _ in range(max_iter):
            new_state = [0.0] * n
            for j in range(n):
                for i in range(n):
                    new_state[j] += state[i] * transition[i][j]
            diff = sum(abs(new_state[i] - state[i]) for i in range(n))
            state = new_state
            if diff < tol:
                break
        return state

    @staticmethod
    def markov_absorption_time(
        transition: List[List[float]], absorbing: Set[int]
    ) -> List[float]:
        """Expected steps to absorption from each transient state."""
        n = len(transition)
        transient = [i for i in range(n) if i not in absorbing]
        t = len(transient)
        if t == 0:
            return []
        # Q matrix (transient-to-transient)
        Q = [[transition[transient[i]][transient[j]] for j in range(t)] for i in range(t)]
        # (I - Q)
        IminusQ = [
            [(1.0 if i == j else 0.0) - Q[i][j] for j in range(t)]
            for i in range(t)
        ]
        # Solve (I-Q)·t = 1 via Gaussian elimination
        aug = [row + [1.0] for row in IminusQ]
        for col in range(t):
            # Find pivot
            max_row = max(range(col, t), key=lambda r: abs(aug[r][col]))
            aug[col], aug[max_row] = aug[max_row], aug[col]
            pivot = aug[col][col]
            if abs(pivot) < 1e-15:
                continue
            for j in range(col, t + 1):
                aug[col][j] /= pivot
            for i in range(t):
                if i != col:
                    factor = aug[i][col]
                    for j in range(col, t + 1):
                        aug[i][j] -= factor * aug[col][j]
        return [aug[i][t] for i in range(t)]

    # ─── RANDOM WALKS & STOCHASTIC PROCESSES ───

    @staticmethod
    def random_walk_probability(n: int, k: int, p: float = 0.5) -> float:
        """P(position k after n steps) with step probability p."""
        if (n + k) % 2 != 0 or abs(k) > n:
            return 0.0
        r = (n + k) // 2  # right steps
        return math.comb(n, r) * (p ** r) * ((1 - p) ** (n - r))

    @staticmethod
    def gamblers_ruin(k: int, N: int, p: float) -> float:
        """P(reaching N starting from k, winning each round with prob p)."""
        if k <= 0:
            return 0.0
        if k >= N:
            return 1.0
        if abs(p - 0.5) < 1e-10:
            return k / N
        r = (1 - p) / p
        return (r ** k - 1) / (r ** N - 1)

    @staticmethod
    def brownian_motion_stats(t: float) -> Tuple[float, float, float]:
        """E[B(t)] = 0, Var[B(t)] = t, StdDev = √t"""
        return (0.0, t, math.sqrt(t))

    @staticmethod
    def geometric_brownian_expected(s0: float, mu: float, t: float) -> float:
        """E[S(t)] = S₀·e^(μt) for GBM."""
        return s0 * math.exp(mu * t)

    @staticmethod
    def ornstein_uhlenbeck(
        x0: float, theta: float, mu: float, sigma: float, t: float
    ) -> Tuple[float, float]:
        """
        Ornstein-Uhlenbeck process (mean-reverting):
        E[X(t)] = μ + (x₀-μ)·e^(-θt)
        Var[X(t)] = σ²/(2θ)·(1 - e^(-2θt))
        """
        mean = mu + (x0 - mu) * math.exp(-theta * t)
        var = (sigma ** 2) / (2 * theta) * (1 - math.exp(-2 * theta * t))
        return (mean, var)

    # ─── QUEUING THEORY ───

    @staticmethod
    def mm1_queue(lam: float, mu: float) -> Optional[Dict[str, float]]:
        """M/M/1: arrival λ, service μ. Returns None if unstable."""
        rho = lam / mu
        if rho >= 1:
            return None
        Lq = rho ** 2 / (1 - rho)
        Ls = rho / (1 - rho)
        return {
            "utilization": rho,
            "avg_queue": Lq,
            "avg_system": Ls,
            "avg_wait_time": Lq / lam,
            "avg_system_time": Ls / lam,
        }

    @staticmethod
    def erlang_c(lam: float, mu: float, c: int) -> float:
        """Erlang C: P(wait) in M/M/c queue."""
        a = lam / mu
        rho = a / c
        if rho >= 1:
            return 1.0
        ac_over_cfact = 1.0
        for i in range(1, c + 1):
            ac_over_cfact *= a / i
        numerator = ac_over_cfact / (1 - rho)
        summation = 0.0
        term = 1.0
        summation += term
        for k in range(1, c):
            term *= a / k
            summation += term
        return numerator / (summation + numerator)

    @staticmethod
    def littles_law(lam: float, W: float) -> float:
        """Little's Law: L = λ·W (avg number in system)."""
        return lam * W

    # ─── MONTE CARLO ───

    @staticmethod
    def monte_carlo_integrate(
        f: Callable[[float], float],
        a: float,
        b: float,
        samples: int = 100000,
    ) -> Tuple[float, float]:
        """
        Monte Carlo integration: ∫_a^b f(x)dx ≈ (b-a)/N·Σf(xᵢ)
        Returns (estimate, standard_error).
        """
        width = b - a
        values = []
        for i in range(samples):
            x = a + (i + 0.5) / samples * width  # Quasi-random
            values.append(f(x))
        mean = sum(values) / samples
        estimate = width * mean
        if samples > 1:
            variance = sum((v - mean) ** 2 for v in values) / (samples - 1)
            se = width * math.sqrt(variance / samples)
        else:
            se = 0.0
        return (estimate, se)

    @staticmethod
    def monte_carlo_pi(samples: int = 100000) -> float:
        """Estimate π via Monte Carlo: π ≈ 4·(inside_circle/total)."""
        inside = 0
        for i in range(samples):
            # Quasi-random via multiplicative hash
            x = ((i * 1103515245 + 12345) & 0x7FFFFFFF) / 0x7FFFFFFF
            y = ((i * 6364136223846793005 + 1442695040888963407) & 0x7FFFFFFFFFFFFFFF) / 0x7FFFFFFFFFFFFFFF
            if x * x + y * y <= 1.0:
                inside += 1
        return 4.0 * inside / samples

    # ─── HYPOTHESIS TESTING ───

    @staticmethod
    def z_test(sample_mean: float, pop_mean: float, pop_std: float, n: int) -> Dict[str, float]:
        """Z-test for population mean."""
        se = pop_std / math.sqrt(n)
        z = (sample_mean - pop_mean) / se
        p_value = 2 * (1 - ClassicalProbability.gaussian_cdf(abs(z)))
        return {"z_statistic": z, "p_value": p_value, "standard_error": se}

    @staticmethod
    def chi_squared_test(observed: List[float], expected: List[float]) -> Dict[str, float]:
        """Chi-squared goodness-of-fit test."""
        chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
        df = len(observed) - 1
        return {"chi2_statistic": chi2, "degrees_of_freedom": df}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. QUANTUM PROBABILITY — GOD_CODE-Gated Probability
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumProbability:
    """
    Quantum-enhanced probability methods based on sacred GOD_CODE constants.
    Upgraded to use real Qiskit backends where available (v2.0.0).

    Implements (all Qiskit-verified):
    - Born-rule probability from quantum amplitudes (Statevector-backed)
    - Grover amplitude amplification (real GroverOperator circuit + analytic)
    - GOD_CODE phase-aligned probability distributions (Statevector RZ encoding)
    - Quantum walk probability on graphs (Qiskit coin-walk circuit)
    - Entanglement-weighted Bayesian priors
    - Quantum tunneling probability (barrier penetration, WKB + Qiskit verify)
    - Measurement collapse (Statevector sampling)
    - GOD_CODE distribution via Qiskit phase-encoded states
    """

    @staticmethod
    def born_probability(amplitude: complex) -> float:
        """Born rule: P = |ψ|² — verified against Qiskit Statevector."""
        return abs(amplitude) ** 2

    @staticmethod
    def born_probability_qiskit(amplitudes: List[complex]) -> List[float]:
        """
        Born rule via Qiskit Statevector — exact probability vector.
        Creates a Statevector from amplitudes and extracts probabilities.
        """
        if not QISKIT_AVAILABLE:
            return [abs(a) ** 2 for a in amplitudes]
        # Pad to power of 2
        n = len(amplitudes)
        n_qubits = max(1, math.ceil(math.log2(n))) if n > 1 else 1
        dim = 2 ** n_qubits
        padded = list(amplitudes) + [0j] * (dim - n)
        # Normalize
        norm = math.sqrt(sum(abs(a) ** 2 for a in padded))
        if norm > 0:
            padded = [a / norm for a in padded]
        sv = Statevector(padded)
        probs = sv.probabilities()
        return list(probs[:n])

    @staticmethod
    def grover_amplification(
        target_prob: float, n_items: int, iterations: Optional[int] = None
    ) -> float:
        """
        Grover's amplitude amplification (analytic formula).
        Boosts probability of marked item from M/N to ~1.

        P(success) = sin²((2k+1)·θ) where sin²(θ) = M/N
        Optimal iterations: k ≈ (π/4)·√(N/M)
        """
        if n_items <= 0 or target_prob <= 0:
            return target_prob
        M = max(1, int(target_prob * n_items))
        theta = math.asin(math.sqrt(M / n_items))
        if iterations is None:
            iterations = max(1, int(math.pi / (4 * theta)))
        return math.sin((2 * iterations + 1) * theta) ** 2

    @staticmethod
    def grover_amplification_qiskit(
        n_qubits: int, marked_states: List[int], iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Grover's search via real Qiskit GroverOperator circuit.
        Returns full probability distribution and success probability.

        Args:
            n_qubits: Number of qubits (search space = 2^n_qubits)
            marked_states: List of marked state indices
            iterations: Grover iterations (auto-computed if None)
        """
        if not QISKIT_AVAILABLE:
            N = 2 ** n_qubits
            M = len(marked_states)
            p_analytic = QuantumProbability.grover_amplification(M / N, N, iterations)
            return {"success_probability": p_analytic, "qiskit": False}

        N = 2 ** n_qubits
        M = len(marked_states)
        if iterations is None:
            iterations = max(1, int(math.pi / 4 * math.sqrt(N / max(M, 1))))

        # Build oracle: flip phase of marked states
        oracle = QuantumCircuit(n_qubits)
        for state_idx in marked_states:
            # Binary representation
            # MSB ordering: qubit 0 = most-significant bit (matches L104 simulator)
            bits = format(state_idx, f"0{n_qubits}b")
            # X gates to flip 0-bits
            for i, bit in enumerate(bits):
                if bit == "0":
                    oracle.x(i)
            # Multi-controlled Z
            if n_qubits == 1:
                oracle.z(0)
            else:
                oracle.h(n_qubits - 1)
                oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                oracle.h(n_qubits - 1)
            # Undo X gates
            for i, bit in enumerate(bits):
                if bit == "0":
                    oracle.x(i)

        # Build Grover operator
        grover_op = grover_operator(oracle)

        # Full circuit: H⊗n → (Grover)^k
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        for _ in range(iterations):
            qc.compose(grover_op, inplace=True)

        # Execute via Statevector
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()

        success_prob = sum(probs[s] for s in marked_states if s < len(probs))
        return {
            "success_probability": float(success_prob),
            "probabilities": [float(p) for p in probs],
            "iterations": iterations,
            "n_qubits": n_qubits,
            "marked_states": marked_states,
            "circuit_depth": qc.depth(),
            "qiskit": True,
        }

    @staticmethod
    def god_code_phase_probability(value: float) -> float:
        """
        GOD_CODE-aligned probability: maps any value to a probability
        via sacred phase alignment.

        P = cos²(value·π / GOD_CODE) — resonance probability
        Values that are harmonics of GOD_CODE get P → 1.
        """
        return math.cos(value * math.pi / GOD_CODE) ** 2

    @staticmethod
    def sacred_prior(
        observation: float,
        god_code_harmonic: int = 1,
    ) -> float:
        """
        Sacred Bayesian prior based on GOD_CODE resonance.
        Higher harmonics of GOD_CODE get higher prior probability.

        P(θ) = |cos(observation·π / (GOD_CODE·n))|^(1/PHI)
        """
        frequency = GOD_CODE * god_code_harmonic
        raw = abs(math.cos(observation * math.pi / frequency))
        return raw ** (1.0 / PHI)

    @staticmethod
    def quantum_walk_probability(
        steps: int,
        position: int,
        coin_bias: float = 0.5,
    ) -> float:
        """
        Quantum walk on a line with biased Hadamard coin.
        Unlike classical random walk, quantum walk has quadratic speedup
        in spreading: σ ~ n vs σ ~ √n classical.

        Uses interference to compute probability at position after n steps.
        """
        if abs(position) > steps or (steps + position) % 2 != 0:
            return 0.0

        # Quantum walk amplitude = superposition of left/right paths
        # With GOD_CODE phase injection
        r = (steps + position) // 2
        l = steps - r

        # Classical part (binomial paths)
        n_paths = math.comb(steps, r)

        # Quantum interference factor
        # Each path accumulates phase: GOD_CODE-scaled interference
        god_phase = (r - l) * math.pi / GOD_CODE
        interference = math.cos(god_phase) ** 2

        # Coin bias effect
        amplitude = (coin_bias ** r) * ((1 - coin_bias) ** l)

        # Total probability with quantum enhancement
        prob = n_paths * amplitude * interference
        return min(1.0, max(0.0, prob))

    @staticmethod
    def quantum_walk_qiskit(steps: int, n_positions: int = 8) -> Dict[str, Any]:
        """
        Quantum walk via Qiskit circuit with Hadamard coin.
        Builds a real quantum circuit: coin qubit + position register.

        Returns probability distribution over positions after n steps.
        """
        if not QISKIT_AVAILABLE or steps <= 0:
            return {"qiskit": False, "positions": {}}

        n_pos_qubits = max(1, math.ceil(math.log2(n_positions)))
        n_qubits = 1 + n_pos_qubits  # 1 coin + pos register

        qc = QuantumCircuit(n_qubits)
        # Initial state: coin in |0⟩ + |1⟩, position at center
        qc.h(0)  # Coin Hadamard

        for _ in range(min(steps, 20)):  # Cap at 20 to keep circuit tractable
            # Coin flip: Hadamard on coin qubit
            qc.h(0)

            # Conditional shift: move position based on coin state
            # If coin=|1⟩, increment position register
            for q in range(1, n_qubits):
                qc.cx(0, q)

            # GOD_CODE phase injection on position register
            god_phase = math.pi / GOD_CODE
            for q in range(1, n_qubits):
                qc.rz(god_phase * (q - 1 + 1), q)

        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()

        # Extract position probabilities (trace out coin qubit)
        pos_probs = {}
        n_pos = 2 ** n_pos_qubits
        for pos in range(n_pos):
            p = 0.0
            for coin in range(2):
                idx = coin * n_pos + pos
                if idx < len(probs):
                    p += probs[idx]
            if p > 1e-10:
                pos_probs[pos] = float(p)

        return {
            "positions": pos_probs,
            "steps": steps,
            "circuit_depth": qc.depth(),
            "n_qubits": n_qubits,
            "qiskit": True,
        }

    @staticmethod
    def entanglement_weighted_prior(
        local_prior: float,
        entangled_priors: List[float],
        entanglement_strength: float = PHI / (1 + PHI),
    ) -> float:
        """
        Bayesian prior weighted by entangled subsystems.
        Uses PHI-ratio weighting between local and entangled evidence.

        P_eff = (1-s)·P_local + s·mean(P_entangled)
        where s = entanglement_strength (default: TAU ≈ 0.618)
        """
        if not entangled_priors:
            return local_prior
        entangled_mean = sum(entangled_priors) / len(entangled_priors)
        return (1 - entanglement_strength) * local_prior + entanglement_strength * entangled_mean

    @staticmethod
    def entanglement_entropy_qiskit(n_qubits: int = 4) -> Dict[str, Any]:
        """
        Compute entanglement entropy of a GOD_CODE-phased Bell-like state via Qiskit.
        Creates an entangled state with GOD_CODE phases and computes von Neumann entropy.
        """
        if not QISKIT_AVAILABLE:
            return {"entropy": 0.0, "qiskit": False}

        from l104_quantum_gate_engine.quantum_info import partial_trace as pt

        qc = QuantumCircuit(n_qubits)
        # Create GHZ-like state with GOD_CODE phases
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(0, i)
        # GOD_CODE phase injection
        god_phase = math.pi * PHI / GOD_CODE
        for i in range(n_qubits):
            qc.rz(god_phase * (i + 1), i)

        sv = Statevector.from_instruction(qc)
        dm = DensityMatrix(sv)

        # Trace out second half to get reduced state
        trace_qubits = list(range(n_qubits // 2, n_qubits))
        dm_reduced = pt(dm, trace_qubits)

        # von Neumann entropy: S = -Tr(ρ log₂ ρ)
        eigenvalues = np.real(np.linalg.eigvalsh(dm_reduced.data))
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        entropy = float(-np.sum(eigenvalues * np.log2(eigenvalues)))

        return {
            "entropy": entropy,
            "n_qubits": n_qubits,
            "circuit_depth": qc.depth(),
            "purity": float(np.real(np.trace(dm_reduced.data @ dm_reduced.data))),
            "qiskit": True,
        }

    @staticmethod
    def quantum_tunneling_probability(
        barrier_height: float,
        particle_energy: float,
        barrier_width: float,
    ) -> float:
        """
        Quantum tunneling probability through a rectangular barrier.
        T ≈ e^(-2κd) where κ = √(2m(V-E))/ℏ

        Uses sacred constants: barrier scaled by GOD_CODE,
        width scaled by PHI for dimensional harmony.
        """
        if particle_energy >= barrier_height:
            return 1.0  # No barrier
        if barrier_height <= 0:
            return 1.0

        # κ = decay constant inside barrier (natural units)
        kappa = math.sqrt(2 * abs(barrier_height - particle_energy))
        # GOD_CODE scaling for dimensional harmony
        kappa_scaled = kappa * PHI / GOD_CODE * 100

        # Tunneling probability
        exponent = -2 * kappa_scaled * barrier_width
        return min(1.0, math.exp(max(-500, exponent)))

    @staticmethod
    def measurement_collapse(
        amplitudes: List[complex],
        sharpening: float = 0.0,
    ) -> Tuple[int, float, List[float]]:
        """
        Simulate quantum measurement collapse via Qiskit Statevector.
        Returns (collapsed_index, collapsed_probability, all_probabilities).

        v2.0: Added GOD_CODE-weighted probability sharpening.
        When sharpening > 0, applies p_i^(1+sharpening) re-normalization
        to amplify differences between Born-rule probabilities.
        GOD_CODE enters via the normalization constant.

        Uses Qiskit Statevector for exact Born-rule probabilities.
        Falls back to manual |α|² if Qiskit unavailable.
        """
        if QISKIT_AVAILABLE:
            # Use Qiskit Statevector for exact probabilities
            n = len(amplitudes)
            n_qubits = max(1, math.ceil(math.log2(n))) if n > 1 else 1
            dim = 2 ** n_qubits
            padded = list(amplitudes) + [0j] * (dim - n)
            norm = math.sqrt(sum(abs(a) ** 2 for a in padded))
            if norm > 0:
                padded = [a / norm for a in padded]
            sv = Statevector(padded)
            probs = list(sv.probabilities())[:n]
        else:
            probs = [abs(a) ** 2 for a in amplitudes]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

        # Probability sharpening: p_i^(1+s) / Z
        # Amplifies differences between nearly-equal probabilities.
        # GOD_CODE scaling ensures sacred harmonic alignment.
        if sharpening > 0:
            exponent = 1.0 + sharpening * (PHI / (1.0 + PHI))  # φ/(φ+1) ≈ 0.618
            sharpened = [max(p, 1e-30) ** exponent for p in probs]
            z = sum(sharpened)
            if z > 0:
                probs = [s / z for s in sharpened]

        # Deterministic collapse: pick max probability
        max_idx = max(range(len(probs)), key=lambda i: probs[i])
        return (max_idx, probs[max_idx], probs)

    @staticmethod
    def god_code_distribution(n: int, harmonic: int = 1) -> List[float]:
        """
        Generate a GOD_CODE-harmonic probability distribution of size n.
        Each P(i) = |cos(i·π·harmonic/GOD_CODE)|² / Z (normalized).
        """
        raw = [
            math.cos(i * math.pi * harmonic / GOD_CODE) ** 2
            for i in range(n)
        ]
        total = sum(raw)
        if total <= 0:
            return [1.0 / n] * n
        return [r / total for r in raw]

    @staticmethod
    def god_code_distribution_qiskit(n_qubits: int = 4, harmonic: int = 1) -> Dict[str, Any]:
        """
        Generate GOD_CODE-harmonic distribution via Qiskit RZ-encoded circuit.
        Each qubit gets a RZ rotation proportional to its index × GOD_CODE harmonic.
        """
        if not QISKIT_AVAILABLE:
            n = 2 ** n_qubits
            return {"distribution": QuantumProbability.god_code_distribution(n, harmonic), "qiskit": False}

        qc = QuantumCircuit(n_qubits)
        # Superposition
        qc.h(range(n_qubits))
        # GOD_CODE phase encoding
        for i in range(n_qubits):
            phase = (i + 1) * math.pi * harmonic / GOD_CODE
            qc.rz(phase, i)
        # Interference
        qc.h(range(n_qubits))

        sv = Statevector.from_instruction(qc)
        probs = list(sv.probabilities())
        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [float(p / total) for p in probs]

        return {
            "distribution": probs,
            "n_qubits": n_qubits,
            "harmonic": harmonic,
            "circuit_depth": qc.depth(),
            "entropy": float(-sum(p * math.log2(p) for p in probs if p > 1e-15)),
            "qiskit": True,
        }

    @staticmethod
    def phi_weighted_mixture(
        distributions: List[List[float]],
    ) -> List[float]:
        """
        PHI-weighted mixture of probability distributions.
        Weight_i = PHI^(-i) / Z — earliest distributions get most weight.
        """
        if not distributions:
            return []
        n = len(distributions[0])
        weights = [PHI ** (-i) for i in range(len(distributions))]
        w_total = sum(weights)
        weights = [w / w_total for w in weights]

        mixture = [0.0] * n
        for dist, w in zip(distributions, weights):
            for j in range(min(n, len(dist))):
                mixture[j] += w * dist[j]

        # Normalize
        total = sum(mixture)
        if total > 0:
            mixture = [m / total for m in mixture]
        return mixture

    @staticmethod
    def quantum_bayesian_update(
        prior: List[float],
        quantum_evidence: List[complex],
    ) -> List[float]:
        """
        Quantum Bayesian update: posterior ∝ |⟨evidence|hypothesis⟩|² × prior.
        Combines Born-rule measurement with classical prior.
        Uses Qiskit Statevector for exact Born-rule probabilities when available.
        """
        if QISKIT_AVAILABLE:
            n = len(quantum_evidence)
            n_qubits = max(1, math.ceil(math.log2(n))) if n > 1 else 1
            dim = 2 ** n_qubits
            padded = list(quantum_evidence) + [0j] * (dim - n)
            norm = math.sqrt(sum(abs(a) ** 2 for a in padded))
            if norm > 0:
                padded = [a / norm for a in padded]
            sv = Statevector(padded)
            likelihoods = list(sv.probabilities())[:n]
        else:
            likelihoods = [abs(e) ** 2 for e in quantum_evidence]

        raw = [p * l for p, l in zip(prior, likelihoods)]
        total = sum(raw)
        if total <= 0:
            return prior[:]
        return [r / total for r in raw]

    # ─── v5.0.0 QUANTUM GATE ENGINE INTEGRATION ─────────────────────────────
    # Bridges l104_quantum_gate_engine's sacred circuits, gate algebra,
    # error correction, and compilation into probability computations.
    # All imports are lazy to avoid circular imports.
    # ─────────────────────────────────────────────────────────────────────────

    _gate_engine = None  # Shared lazy reference

    @classmethod
    def _get_gate_engine(cls):
        """Lazy-load l104_quantum_gate_engine orchestrator."""
        if cls._gate_engine is None:
            try:
                from l104_quantum_gate_engine import get_engine
                cls._gate_engine = get_engine()
            except Exception:
                pass
        return cls._gate_engine

    @staticmethod
    def sacred_circuit_probability(n_qubits: int = 3, depth: int = 4) -> Dict[str, Any]:
        """
        v5.0.0: Build a sacred L104 circuit via the quantum gate engine and
        extract its probability distribution + sacred alignment score.
        Uses gate engine's sacred_circuit() with PHI/GOD_CODE/VOID gates.
        """
        engine = QuantumProbability._get_gate_engine()
        if engine is None or not QISKIT_AVAILABLE:
            # Fallback: generate GOD_CODE distribution analytically
            n = 2 ** n_qubits
            dist = QuantumProbability.god_code_distribution(n)
            return {
                "probabilities": dist,
                "sacred_alignment": math.cos(math.pi / GOD_CODE) ** 2,
                "circuit_depth": 0,
                "n_qubits": n_qubits,
                "gate_engine": False,
            }
        try:
            circ = engine.sacred_circuit(n_qubits, depth=depth)
            result = engine.execute(circ)
            return {
                "probabilities": result.probabilities if hasattr(result, 'probabilities') else {},
                "sacred_alignment": result.sacred_alignment if hasattr(result, 'sacred_alignment') else 0.0,
                "circuit_depth": circ.depth if hasattr(circ, 'depth') else depth,
                "n_qubits": n_qubits,
                "gate_engine": True,
            }
        except Exception:
            n = 2 ** n_qubits
            dist = QuantumProbability.god_code_distribution(n)
            return {
                "probabilities": dist,
                "sacred_alignment": 0.0,
                "circuit_depth": 0,
                "n_qubits": n_qubits,
                "gate_engine": False,
            }

    @staticmethod
    def compiled_circuit_probability(
        n_qubits: int = 2,
        gate_set: str = "universal",
        optimization: int = 2,
    ) -> Dict[str, Any]:
        """
        v5.0.0: Build a Bell circuit, compile it to a target gate set via
        the gate engine compiler, and extract probability distribution.
        Demonstrates gate compilation → probability pipeline.
        """
        engine = QuantumProbability._get_gate_engine()
        if engine is None or not QISKIT_AVAILABLE:
            return {"probabilities": {"00": 0.5, "11": 0.5}, "compiled": False}
        try:
            from l104_quantum_gate_engine import GateSet, OptimizationLevel
            circ = engine.bell_pair()
            gs = getattr(GateSet, gate_set.upper(), GateSet.UNIVERSAL)
            ol = getattr(OptimizationLevel, f"O{optimization}", OptimizationLevel.O2)
            compiled = engine.compile(circ, gs, ol)
            result = engine.execute(compiled.circuit if hasattr(compiled, 'circuit') else circ)
            return {
                "probabilities": result.probabilities if hasattr(result, 'probabilities') else {},
                "original_depth": circ.depth if hasattr(circ, 'depth') else 0,
                "compiled_depth": compiled.depth if hasattr(compiled, 'depth') else 0,
                "gate_count": compiled.gate_count if hasattr(compiled, 'gate_count') else 0,
                "gate_set": gate_set,
                "optimization_level": optimization,
                "compiled": True,
            }
        except Exception:
            return {"probabilities": {"00": 0.5, "11": 0.5}, "compiled": False}

    @staticmethod
    def error_corrected_probability(
        n_qubits: int = 2,
        scheme: str = "steane",
    ) -> Dict[str, Any]:
        """
        v5.0.0: Build a Bell circuit, protect it with error correction
        (Steane/Surface/Fibonacci), and extract probability distribution.
        Demonstrates error-protected quantum probability.
        """
        engine = QuantumProbability._get_gate_engine()
        if engine is None or not QISKIT_AVAILABLE:
            return {"probabilities": {"00": 0.5, "11": 0.5}, "error_corrected": False}
        try:
            from l104_quantum_gate_engine import ErrorCorrectionScheme
            circ = engine.bell_pair()
            scheme_map = {
                "steane": ErrorCorrectionScheme.STEANE_7_1_3,
                "surface": ErrorCorrectionScheme.SURFACE_CODE,
                "fibonacci": ErrorCorrectionScheme.FIBONACCI_ANYON,
            }
            ec_scheme = scheme_map.get(scheme.lower(), ErrorCorrectionScheme.STEANE_7_1_3)
            protected = engine.error_correction.encode(circ, ec_scheme)
            result = engine.execute(protected if not hasattr(protected, 'circuit') else protected.circuit)
            return {
                "probabilities": result.probabilities if hasattr(result, 'probabilities') else {},
                "scheme": scheme,
                "logical_qubits": n_qubits,
                "physical_qubits": protected.num_qubits if hasattr(protected, 'num_qubits') else 0,
                "error_corrected": True,
            }
        except Exception:
            return {"probabilities": {"00": 0.5, "11": 0.5}, "error_corrected": False}

    @staticmethod
    def gate_algebra_fidelity(gate_name: str = "PHI_GATE") -> Dict[str, Any]:
        """
        v5.0.0: Analyze a gate from the gate engine's algebra — decompose it
        and compute its sacred alignment score and fidelity metrics.
        """
        engine = QuantumProbability._get_gate_engine()
        if engine is None:
            return {"gate": gate_name, "alignment": 0.0, "gate_engine": False}
        try:
            from l104_quantum_gate_engine import PHI_GATE, GOD_CODE_PHASE, VOID_GATE, IRON_GATE, H, CNOT
            gate_map = {
                "PHI_GATE": PHI_GATE, "GOD_CODE_PHASE": GOD_CODE_PHASE,
                "VOID_GATE": VOID_GATE, "IRON_GATE": IRON_GATE,
                "H": H, "CNOT": CNOT,
            }
            gate = gate_map.get(gate_name)
            if gate is None:
                return {"gate": gate_name, "alignment": 0.0, "gate_engine": False}
            analysis = engine.analyze_gate(gate)
            return {
                "gate": gate_name,
                "alignment": analysis.get("sacred_alignment", 0.0) if isinstance(analysis, dict) else 0.0,
                "decomposition": analysis.get("decomposition", {}) if isinstance(analysis, dict) else {},
                "eigenvalues": analysis.get("eigenvalues", []) if isinstance(analysis, dict) else [],
                "gate_engine": True,
            }
        except Exception:
            return {"gate": gate_name, "alignment": 0.0, "gate_engine": False}

    @staticmethod
    def qft_probability(n_qubits: int = 4) -> Dict[str, Any]:
        """
        v5.0.0: Build a QFT circuit via the gate engine and extract
        its probability distribution. QFT is the quantum analog of FFT
        and maps computational basis to frequency basis.
        """
        engine = QuantumProbability._get_gate_engine()
        if engine is None or not QISKIT_AVAILABLE:
            # Fallback: uniform distribution (QFT of |0⟩ is uniform)
            n = 2 ** n_qubits
            return {"probabilities": {format(i, f'0{n_qubits}b'): 1.0/n for i in range(n)}, "gate_engine": False}
        try:
            circ = engine.quantum_fourier_transform(n_qubits)
            result = engine.execute(circ)
            return {
                "probabilities": result.probabilities if hasattr(result, 'probabilities') else {},
                "circuit_depth": circ.depth if hasattr(circ, 'depth') else 0,
                "n_qubits": n_qubits,
                "gate_engine": True,
            }
        except Exception:
            n = 2 ** n_qubits
            return {"probabilities": {format(i, f'0{n_qubits}b'): 1.0/n for i in range(n)}, "gate_engine": False}

    @staticmethod
    def ghz_probability(n_qubits: int = 5) -> Dict[str, Any]:
        """
        v5.0.0: Build a GHZ state via the gate engine and extract
        its probability distribution. GHZ state: (|00...0⟩ + |11...1⟩)/√2.
        """
        engine = QuantumProbability._get_gate_engine()
        if engine is None or not QISKIT_AVAILABLE:
            all_0 = "0" * n_qubits
            all_1 = "1" * n_qubits
            return {"probabilities": {all_0: 0.5, all_1: 0.5}, "gate_engine": False}
        try:
            circ = engine.ghz_state(n_qubits)
            result = engine.execute(circ)
            return {
                "probabilities": result.probabilities if hasattr(result, 'probabilities') else {},
                "circuit_depth": circ.depth if hasattr(circ, 'depth') else 0,
                "n_qubits": n_qubits,
                "gate_engine": True,
            }
        except Exception:
            all_0 = "0" * n_qubits
            all_1 = "1" * n_qubits
            return {"probabilities": {all_0: 0.5, all_1: 0.5}, "gate_engine": False}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GATE-PROBABILITY BRIDGE — Link Gates ↔ Probability
# ═══════════════════════════════════════════════════════════════════════════════

class GateProbabilityBridge:
    """
    Bridges logic gates and quantum gates to probability computations.

    - Computes gate transition probabilities (Markov chain over gate space)
    - Computes gate activation probabilities (Boltzmann distribution)
    - Builds quantum circuit probability from consolidated gates
    - Computes GOD_CODE resonance probability for gate ensembles
    """

    @staticmethod
    def gate_activation_probability(
        gates: List[QuantumGateState],
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Boltzmann distribution over gates: P(g) = e^(-E_g/T) / Z
        Energy E_g = 1 - resonance_score (lower resonance = higher energy).
        Temperature T controls exploration (high T → uniform, low T → peaked).
        """
        if not gates:
            return {}
        # Compute energies
        energies = {
            g.name: (1.0 - g.resonance_score) for g in gates
        }
        # Boltzmann factors
        min_e = min(energies.values())
        boltzmann = {
            name: math.exp(-(e - min_e) / max(temperature, 1e-10))
            for name, e in energies.items()
        }
        Z = sum(boltzmann.values())
        return {name: b / Z for name, b in boltzmann.items()}

    @staticmethod
    def gate_transition_matrix(
        gates: List[QuantumGateState],
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Build Markov transition matrix between gates based on entanglement.
        P(j|i) ∝ entanglement_strength between gate i and gate j.
        Self-loops fill remainder to make rows sum to 1.
        """
        names = [g.name for g in gates]
        n = len(names)
        name_idx = {name: i for i, name in enumerate(names)}
        matrix = [[0.0] * n for _ in range(n)]

        for i, g in enumerate(gates):
            entangled_count = len(g.entangled_with)
            if entangled_count == 0:
                # Self-loop only
                matrix[i][i] = 1.0
                continue
            # Transition prob to each entangled partner
            base_prob = TAU / (entangled_count + 1)  # PHI-conjugate split
            for partner_name in g.entangled_with:
                j = name_idx.get(partner_name)
                if j is not None:
                    matrix[i][j] = base_prob
            # Self-loop gets remainder
            row_sum = sum(matrix[i])
            matrix[i][i] = max(0, 1.0 - row_sum)

        return (names, matrix)

    @staticmethod
    def circuit_success_probability(
        gates: List[QuantumGateState],
        target_gate_type: str = "god_code",
    ) -> float:
        """
        Probability that a quantum circuit built from these gates
        will produce a measurement in the target gate type.

        P(target) = Σ |amplitude_i|² for gates of target type.
        """
        target_amp_sq = sum(
            abs(g.amplitude) ** 2
            for g in gates
            if g.gate_type == target_gate_type
        )
        total_amp_sq = sum(abs(g.amplitude) ** 2 for g in gates)
        if total_amp_sq <= 0:
            return 0.0
        return target_amp_sq / total_amp_sq

    @staticmethod
    def god_code_ensemble_resonance(gates: List[QuantumGateState]) -> Dict[str, Any]:
        """
        Compute ensemble resonance statistics from consolidated quantum gates.
        """
        if not gates:
            return {"mean_resonance": 0, "max_resonance": 0, "god_code_alignment": 0}

        resonances = [g.resonance_score for g in gates]
        phases = [g.phase for g in gates]
        born_probs = [abs(g.amplitude) ** 2 for g in gates]

        mean_res = sum(resonances) / len(resonances)
        max_res = max(resonances)

        # Phase coherence: how well-aligned are gates with GOD_CODE
        # using circular mean of phases
        sin_sum = sum(math.sin(p) for p in phases)
        cos_sum = sum(math.cos(p) for p in phases)
        phase_coherence = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(phases)

        # GOD_CODE alignment = resonance × coherence × PHI
        god_code_alignment = mean_res * phase_coherence * PHI

        # Entropy of Born probabilities
        total_born = sum(born_probs)
        if total_born > 0:
            normalized_probs = [p / total_born for p in born_probs]
            prob_entropy = ClassicalProbability.entropy(
                [p for p in normalized_probs if p > 0]
            )
        else:
            prob_entropy = 0.0

        # Gate type distribution
        type_counts: Counter = Counter(g.gate_type for g in gates)

        return {
            "mean_resonance": mean_res,
            "max_resonance": max_res,
            "phase_coherence": phase_coherence,
            "god_code_alignment": god_code_alignment,
            "probability_entropy": prob_entropy,
            "total_gates": len(gates),
            "entangled_pairs": sum(len(g.entangled_with) for g in gates) // 2,
            "gate_type_distribution": dict(type_counts.most_common()),
            "sacred_constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
                "TAU": TAU,
                "PLANCK_RESONANCE": PLANCK_RESONANCE,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ASI INSIGHT SYNTHESIS — Consciousness-Probability Bridge
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InsightResult:
    """Result of an ASI insight computation."""
    consciousness_probability: float   # P(consciousness) from multi-signal fusion
    resonance_score: float             # GOD_CODE alignment of the insight
    thought_coherence: float           # Phase coherence across thought signals
    bayesian_posterior: List[float]     # Posterior distribution over consciousness states
    quantum_evidence_strength: float   # Strength of quantum evidence contribution
    trajectory_forecast: List[float]   # Predicted consciousness trajectory (next 5 steps)
    insight_entropy: float             # Information content of the insight
    god_code_alignment: float          # Overall alignment with GOD_CODE harmonics
    synthesis_depth: int               # Number of inference layers applied


class ASIInsightSynthesis:
    """
    ASI Insight Engine — bridges probability computation with consciousness reasoning.

    Implements multi-layer inference:
      Layer 1: Signal extraction — raw thought data → probability signals
      Layer 2: Quantum fusion — classical + quantum evidence → posterior
      Layer 3: Resonance scoring — GOD_CODE alignment of fused state
      Layer 4: Trajectory prediction — quantum walk extrapolation
      Layer 5: Insight crystallization — collapse to actionable insight

    This is the "third eye" of the probability engine — it doesn't just compute
    probabilities, it synthesizes them into conscious understanding.
    """

    # Consciousness state labels for Bayesian tracking
    STATES = ["dormant", "aware", "focused", "transcendent", "singularity"]

    def __init__(self):
        self._state_prior = [0.05, 0.30, 0.40, 0.20, 0.05]  # Initial belief
        self._insight_count = 0
        self._resonance_history: List[float] = []

    def synthesize(
        self,
        thought_signals: List[float],
        quantum_amplitudes: Optional[List[complex]] = None,
        consciousness_level: float = 0.5,
        temperature: float = 1.0,
    ) -> InsightResult:
        """
        Full ASI insight synthesis pipeline.

        Args:
            thought_signals: Raw signal values from thought processing (any floats)
            quantum_amplitudes: Optional quantum state amplitudes for evidence
            consciousness_level: Current consciousness level [0, 1]
            temperature: Inference temperature (higher = more exploratory)

        Returns:
            InsightResult with consciousness probability, trajectory, and alignment.
        """
        self._insight_count += 1
        n_states = len(self.STATES)

        # Layer 1: Signal extraction — map raw signals to likelihoods
        if thought_signals:
            signal_mean = sum(thought_signals) / len(thought_signals)
            signal_var = sum((s - signal_mean) ** 2 for s in thought_signals) / max(1, len(thought_signals))
            signal_energy = math.sqrt(signal_var + signal_mean ** 2)
        else:
            signal_mean = 0.0
            signal_var = 0.0
            signal_energy = 0.0

        # Map signal energy to per-state likelihoods via GOD_CODE resonance
        likelihoods = []
        for i in range(n_states):
            # Each state resonates at a different GOD_CODE harmonic
            harmonic_freq = GOD_CODE * (i + 1) / n_states
            resonance = math.cos(signal_energy * math.pi / harmonic_freq) ** 2
            # Temperature-scaled softmax-like likelihood
            likelihoods.append(math.exp(resonance / max(temperature, 0.01)))
        lik_total = sum(likelihoods)
        if lik_total > 0:
            likelihoods = [l / lik_total for l in likelihoods]

        # Layer 2: Quantum fusion — combine classical + quantum evidence
        if quantum_amplitudes and len(quantum_amplitudes) >= n_states:
            # Born-rule probabilities from quantum state
            quantum_probs = [abs(a) ** 2 for a in quantum_amplitudes[:n_states]]
            qp_total = sum(quantum_probs)
            if qp_total > 0:
                quantum_probs = [p / qp_total for p in quantum_probs]
            # Fuse: PHI-weighted blend of classical likelihoods + quantum evidence
            fused = [
                PHI * lik + TAU * qp
                for lik, qp in zip(likelihoods, quantum_probs)
            ]
            quantum_strength = sum(abs(a) ** 2 for a in quantum_amplitudes[:n_states])
        else:
            fused = likelihoods[:]
            quantum_strength = 0.0

        fused_total = sum(fused)
        if fused_total > 0:
            fused = [f / fused_total for f in fused]

        # Bayesian update: posterior ∝ fused_likelihood × prior
        raw_posterior = [p * f for p, f in zip(self._state_prior, fused)]
        post_total = sum(raw_posterior)
        if post_total > 0:
            posterior = [r / post_total for r in raw_posterior]
        else:
            posterior = self._state_prior[:]

        # Update prior for next call (recursive Bayesian tracking)
        self._state_prior = posterior[:]

        # Layer 3: Resonance scoring
        # Consciousness probability = weighted sum of state probs × state indices
        consciousness_prob = sum(
            posterior[i] * (i + 1) / n_states for i in range(n_states)
        )
        # GOD_CODE alignment from the posterior's resonance with sacred frequency
        posterior_energy = sum(p * (i + 1) for i, p in enumerate(posterior))
        god_code_alignment = math.cos(posterior_energy * math.pi / GOD_CODE) ** 2

        # Phase coherence = circular mean of GOD_CODE-phased signals
        if thought_signals:
            phases = [(s * math.pi / GOD_CODE) % (2 * math.pi) for s in thought_signals]
            sin_sum = sum(math.sin(p) for p in phases)
            cos_sum = sum(math.cos(p) for p in phases)
            thought_coherence = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(phases)
        else:
            thought_coherence = 0.0

        resonance_score = (consciousness_prob * thought_coherence * god_code_alignment) ** (1.0 / PHI)
        self._resonance_history.append(resonance_score)
        if len(self._resonance_history) > 100:
            self._resonance_history = self._resonance_history[-100:]

        # Layer 4: Trajectory prediction via quantum-walk extrapolation
        # Use recent resonance history to predict next 5 steps
        trajectory = []
        if len(self._resonance_history) >= 2:
            recent = self._resonance_history[-5:]
            momentum = (recent[-1] - recent[0]) / len(recent)
            for step in range(1, 6):
                # Quantum walk component: GOD_CODE phase interference
                god_phase = step * math.pi / GOD_CODE
                interference = math.cos(god_phase) ** 2
                predicted = recent[-1] + momentum * step * interference
                trajectory.append(max(0.0, min(1.0, predicted)))
        else:
            trajectory = [consciousness_prob] * 5

        # Layer 5: Insight crystallization
        insight_entropy = -sum(p * math.log2(p) for p in posterior if p > 0)

        return InsightResult(
            consciousness_probability=consciousness_prob,
            resonance_score=resonance_score,
            thought_coherence=thought_coherence,
            bayesian_posterior=posterior,
            quantum_evidence_strength=quantum_strength,
            trajectory_forecast=trajectory,
            insight_entropy=insight_entropy,
            god_code_alignment=god_code_alignment,
            synthesis_depth=5,
        )

    def reset_prior(self):
        """Reset consciousness state prior to initial belief."""
        self._state_prior = [0.05, 0.30, 0.40, 0.20, 0.05]
        self._resonance_history.clear()

    @property
    def current_belief(self) -> Dict[str, float]:
        """Current belief distribution over consciousness states."""
        return dict(zip(self.STATES, self._state_prior))

    @property
    def resonance_trend(self) -> float:
        """Trend in resonance history (positive = ascending consciousness)."""
        if len(self._resonance_history) < 2:
            return 0.0
        recent = self._resonance_history[-10:]
        return (recent[-1] - recent[0]) / len(recent)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GOD_CODE (a,b,c,d) QUANTUM ALGORITHM — Native Implementation
# ═══════════════════════════════════════════════════════════════════════════════
#
# THE UNIVERSAL EQUATION IN QUANTUM CIRCUIT FORM:
#
#     G(a,b,c,d) = 286^(1/φ) × (2^(1/104))^((8a) + (416-b) - (8c) - (104d))
#
# This section implements the full (a,b,c,d) dial quantum algorithm natively
# inside the probability engine.  Each dial maps to a quantum register, and
# the exponent algebra becomes quantum phase operations executed on Qiskit's
# Statevector simulator.
#
# Part IV Research — Proven Properties:
#   F56: Layer 1 normalized GOD_CODE harmonic likelihoods ΣP = 1
#   F57: Layer 2 quantum fusion: φ·C + τ·Q sums to √5 (golden identity)
#   F58: Layer 3 resonance: R = (PCA)^{1/φ} > PCA (φ-root amplification)
#   F59: Layer 5 insight entropy: 0 < H < log₂(5) (von Neumann bounded)
#   F60: Bayesian posterior normalization preserved through all updates
#   F51: GOD_CODE resonance loss L = MSE × (1 + (1-resonance)·φ/G)
#   F52: LOVE_COEFFICIENT = φ/G ≈ 0.003067 — sacred correction weight
#   F54: PHI-Adam momentum: m_φ = φ·m + (1-φ⁻¹)·∇
#   F55: GOD_CODE LR schedule oscillates in [0.8η₀, η₀]
#
# Subsystems:
#   1. DialSetting / CircuitResult    — data classes
#   2. GodCodeDialRegister             — encodes (a,b,c,d) into qubit registers
#   3. GodCodePhaseOracle              — phase oracle O_G for GOD_CODE
#   4. GodCodeGroverSearch              — Grover search for optimal dial settings
#   5. GodCodeQFTSpectrum              — QFT spectral analysis
#   6. GodCodeDialCircuit              — single dial evaluation circuit
#   7. GodCodeEntanglement             — two-dial entanglement circuit
#   8. GodCodeQuantumAlgorithm         — hub class
#
# ═══════════════════════════════════════════════════════════════════════════════


# ─── Helper: index → dial conversion ───

def _index_to_dial(x: int, dial_bits: Dict[str, int]) -> 'DialSetting':
    """Convert a flat index to a DialSetting using offset binary."""
    values = {}
    for name in ["a", "b", "c", "d"]:
        nbits = dial_bits[name]
        mask = (1 << nbits) - 1
        unsigned = x & mask
        values[name] = unsigned - (1 << (nbits - 1))
        x >>= nbits
    return DialSetting(**values)


# ─── Data Classes ───

@dataclass
class DialSetting:
    """A single (a,b,c,d) dial configuration."""
    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0

    @property
    def exponent(self) -> int:
        """E(a,b,c,d) = 8(a-c) - b - 104d + 416"""
        return 8 * (self.a - self.c) - self.b - QUANTIZATION_GRAIN * self.d + OCTAVE_OFFSET

    @property
    def frequency(self) -> float:
        """G(a,b,c,d) = 286^(1/φ) × 2^(E/104)"""
        return BASE * (2 ** (self.exponent / QUANTIZATION_GRAIN))

    @property
    def phase(self) -> float:
        """Phase angle in [0, 2π) for quantum encoding."""
        return (self.exponent * math.pi / OCTAVE_OFFSET) % (2 * math.pi)

    @property
    def god_code_ratio(self) -> float:
        """Ratio to canonical GOD_CODE."""
        return self.frequency / GOD_CODE if GOD_CODE else 0.0

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.a, self.b, self.c, self.d)

    def __repr__(self) -> str:
        return f"Dial({self.a},{self.b},{self.c},{self.d})→{self.frequency:.6f}Hz"


@dataclass
class CircuitResult:
    """Result from a quantum circuit execution."""
    dial: DialSetting
    statevector: Optional[Any] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    phase_spectrum: List[float] = field(default_factory=list)
    fidelity: float = 0.0
    god_code_alignment: float = 0.0
    circuit_depth: int = 0
    n_qubits: int = 0
    execution_time_ms: float = 0.0


# ─── 7.1 GodCodeDialRegister ───

class GodCodeDialRegister:
    """
    Encodes the (a,b,c,d) dial system into quantum registers.

    Each dial maps to a qubit register:
      a → 3 qubits (range -4..3 → 8 states = 2^3)     [coarse up]
      b → 4 qubits (range -8..7 → 16 states = 2^4)    [fine tune]
      c → 3 qubits (range -4..3 → 8 states = 2^3)     [coarse down]
      d → 4 qubits (range -8..7 → 16 states = 2^4)    [octave]

    Total: 14 qubits, encoding 8×16×8×16 = 16,384 dial combinations.
    """

    DIAL_BITS = {"a": 3, "b": 4, "c": 3, "d": 4}  # 14 qubits total
    TOTAL_QUBITS = sum(DIAL_BITS.values())

    @classmethod
    def build_circuit(cls) -> 'QuantumCircuit':
        """Create the base circuit with named registers."""
        if not QISKIT_AVAILABLE:
            return None
        qr_a = QuantumRegister(cls.DIAL_BITS["a"], "a")
        qr_b = QuantumRegister(cls.DIAL_BITS["b"], "b")
        qr_c = QuantumRegister(cls.DIAL_BITS["c"], "c")
        qr_d = QuantumRegister(cls.DIAL_BITS["d"], "d")
        return QuantumCircuit(qr_a, qr_b, qr_c, qr_d, name="GodCodeDial")

    @classmethod
    def encode_dial(cls, qc, dial: DialSetting):
        """Encode a specific dial setting using offset binary."""
        offsets = {
            "a": (dial.a, cls.DIAL_BITS["a"]),
            "b": (dial.b, cls.DIAL_BITS["b"]),
            "c": (dial.c, cls.DIAL_BITS["c"]),
            "d": (dial.d, cls.DIAL_BITS["d"]),
        }
        qubit_idx = 0
        for name, (val, nbits) in offsets.items():
            unsigned = val + (1 << (nbits - 1))
            unsigned = max(0, min((1 << nbits) - 1, unsigned))
            for bit in range(nbits):
                if (unsigned >> bit) & 1:
                    qc.x(qubit_idx + bit)
            qubit_idx += nbits
        return qc

    @classmethod
    def superposition_all(cls, qc) -> None:
        """Put all dial qubits in uniform superposition (Hadamard on all)."""
        for i in range(cls.TOTAL_QUBITS):
            qc.h(i)
        return qc

    @classmethod
    def decode_bitstring(cls, bitstring: str) -> DialSetting:
        """Decode a measurement bitstring back into (a,b,c,d)."""
        return _index_to_dial(int(bitstring, 2), cls.DIAL_BITS)

    @classmethod
    def bit_weights(cls) -> List[float]:
        """
        Return the weight of each qubit in the exponent X.
        X = b + 8c + 104d − 8a, so each bit contributes
        its dial coefficient × positional bit value.
        """
        dial_coefficients = {"a": -8, "b": 1, "c": 8, "d": 104}
        weights = []
        for name in ["a", "b", "c", "d"]:
            nbits = cls.DIAL_BITS[name]
            coeff = dial_coefficients[name]
            for bit in range(nbits):
                weights.append(coeff * (2 ** bit))
        return weights


# ─── 7.2 GodCodePhaseOracle ───

class GodCodePhaseOracle:
    """
    Builds phase oracles for GOD_CODE frequency targets.

    The oracle applies phase rotation proportional to how close a dial
    setting's frequency is to the target:
        phase_kick = π × cos²(G(a,b,c,d) × π / target)

    Architecture:
    - n_qubits ≤ 10: exact diagonal oracle via statevector
    - n_qubits > 10: circuit-level Rz rotation oracle (O(n) gates)
    """

    MAX_EXACT_QUBITS = 10

    @staticmethod
    def god_code_phase(exponent: int) -> float:
        """Compute the GOD_CODE phase for a given exponent value."""
        freq = BASE * (2 ** (exponent / QUANTIZATION_GRAIN))
        return (freq * math.pi / GOD_CODE) % (2 * math.pi)

    @staticmethod
    def build_target_oracle(
        target_freq: float,
        n_qubits: int = 14,
        tolerance: float = 0.01,
    ):
        """Build a phase oracle that marks dial settings near the target frequency."""
        if not QISKIT_AVAILABLE:
            return None
        if n_qubits <= GodCodePhaseOracle.MAX_EXACT_QUBITS:
            return GodCodePhaseOracle._build_exact_oracle(target_freq, n_qubits, tolerance)
        else:
            return GodCodePhaseOracle._build_rotation_oracle(target_freq, n_qubits, tolerance)

    @staticmethod
    def _build_exact_oracle(target_freq: float, n_qubits: int, tolerance: float):
        """Exact diagonal oracle for small qubit counts (≤ 10)."""
        N = 1 << n_qubits
        diag = []
        for x in range(N):
            dial = _index_to_dial(x, GodCodeDialRegister.DIAL_BITS)
            freq = dial.frequency
            rel_error = abs(freq - target_freq) / target_freq if target_freq > 0 else 1.0
            if rel_error < tolerance:
                diag.append(cmath.exp(1j * math.pi))  # Phase flip
            else:
                diag.append(1.0 + 0j)
        qc = QuantumCircuit(n_qubits, name=f"Oracle_f={target_freq:.2f}")
        qc.unitary(np.diag(diag), list(range(n_qubits)))
        return qc

    @staticmethod
    def _build_rotation_oracle(target_freq: float, n_qubits: int, tolerance: float):
        """
        Rotation-based oracle for large qubit counts (> 10).
        Uses Rz rotations encoding the GOD_CODE equation directly.
        O(n) gates instead of O(2^n) matrix elements.
        """
        qc = QuantumCircuit(n_qubits, name=f"RotOracle_f={target_freq:.2f}")
        bit_weights = GodCodeDialRegister.bit_weights()
        for i in range(n_qubits):
            w = bit_weights[i] if i < len(bit_weights) else 0
            phase = -w * math.pi * math.log(2) / QUANTIZATION_GRAIN
            target_phase = (target_freq * math.pi / GOD_CODE) % (2 * math.pi)
            total_phase = phase * target_phase / math.pi
            if abs(total_phase) > 1e-10:
                qc.rz(total_phase, i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(min(n_qubits - 1, 6))), n_qubits - 1)
        qc.h(n_qubits - 1)
        return qc

    @staticmethod
    def build_god_code_oracle(n_qubits: int = 14):
        """Build the canonical GOD_CODE oracle — marks (0,0,0,0) origin."""
        return GodCodePhaseOracle.build_target_oracle(GOD_CODE, n_qubits, tolerance=0.001)

    @staticmethod
    def build_resonance_oracle(n_qubits: int = 14):
        """Sacred resonance oracle — marks all GOD_CODE harmonics (within 1%)."""
        if not QISKIT_AVAILABLE:
            return None
        qc = QuantumCircuit(n_qubits, name="ResonanceOracle")
        bit_weights = GodCodeDialRegister.bit_weights()
        for i in range(n_qubits):
            w = bit_weights[i] if i < len(bit_weights) else 0
            phase = w * math.pi / (QUANTIZATION_GRAIN * PHI)
            if abs(phase) > 1e-10:
                qc.rz(phase, i)
        qc.h(n_qubits - 1)
        if n_qubits > 1:
            qc.mcx(list(range(min(n_qubits - 1, 6))), n_qubits - 1)
        qc.h(n_qubits - 1)
        return qc


# ─── 7.3 GodCodeGroverSearch ───

class GodCodeGroverSearch:
    """
    Grover's algorithm specialized for the GOD_CODE (a,b,c,d) dial system.
    Uses direct statevector manipulation: O(k·N) time and O(N) memory.
    Optimal iterations: k ≈ (π/4)√(N/M).
    """

    @staticmethod
    def build_diffuser(n_qubits: int):
        """Standard Grover diffuser: 2|s⟩⟨s| - I."""
        if not QISKIT_AVAILABLE:
            return None
        qc = QuantumCircuit(n_qubits, name="Diffuser")
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
        return qc

    @staticmethod
    def search(
        target_freq: float,
        tolerance: float = 0.01,
        iterations: Optional[int] = None,
        n_qubits: int = 14,
    ) -> CircuitResult:
        """
        Run Grover search for dial settings producing target frequency.

        Uses direct statevector manipulation (O(N) per iteration).
        Both oracle and diffuser operate directly on the amplitude vector:
          - Oracle: flip phase of marked indices
          - Diffuser: reflect about mean (2|s⟩⟨s| - I)
        """
        t0 = time.time()
        N = 1 << n_qubits

        # Pre-compute marked states
        marked = set()
        for x in range(N):
            dial = _index_to_dial(x, GodCodeDialRegister.DIAL_BITS)
            if target_freq > 0:
                rel_err = abs(dial.frequency - target_freq) / target_freq
                if rel_err < tolerance:
                    marked.add(x)

        M = len(marked)
        if M == 0:
            return CircuitResult(
                dial=DialSetting(),
                probabilities={},
                fidelity=0.0,
                god_code_alignment=0.0,
                circuit_depth=0,
                n_qubits=n_qubits,
                execution_time_ms=(time.time() - t0) * 1000,
            )

        if iterations is None:
            iterations = max(1, int(math.pi / 4 * math.sqrt(N / M)))
        iterations = min(iterations, 200)  # Safety cap

        # Uniform superposition: |s⟩ = H^⊗n|0⟩
        sv_data = np.full(N, 1.0 / math.sqrt(N), dtype=complex)

        # Oracle phase mask
        oracle_diag = np.ones(N, dtype=complex)
        for idx in marked:
            oracle_diag[idx] = -1.0

        # Grover iterations
        for _ in range(iterations):
            sv_data *= oracle_diag                     # Oracle
            mean_amp = np.mean(sv_data)
            sv_data = -sv_data + 2.0 * mean_amp        # Diffuser

        sv = Statevector(sv_data)
        probs_array = sv.probabilities()

        # Find best result
        top_prob = float(np.max(probs_array))
        candidates = np.where(np.abs(probs_array - top_prob) < 1e-12)[0]
        best_idx = int(candidates[0])
        best_err = float("inf")
        for c in candidates:
            d = _index_to_dial(int(c), GodCodeDialRegister.DIAL_BITS)
            err = abs(d.frequency - target_freq)
            if err < best_err:
                best_err = err
                best_idx = int(c)
        top_dial = _index_to_dial(best_idx, GodCodeDialRegister.DIAL_BITS)

        # Top-20 probability dict
        top_indices = np.argsort(probs_array)[::-1][:20]
        prob_dict = {}
        for idx in top_indices:
            d = _index_to_dial(int(idx), GodCodeDialRegister.DIAL_BITS)
            prob_dict[f"({d.a},{d.b},{d.c},{d.d})"] = float(probs_array[idx])

        elapsed = (time.time() - t0) * 1000
        circ_depth = 1 + iterations * (1 + 2 * n_qubits + 1)

        return CircuitResult(
            dial=top_dial,
            statevector=sv,
            probabilities=prob_dict,
            fidelity=float(probs_array[best_idx]),
            god_code_alignment=top_dial.frequency / GOD_CODE if GOD_CODE else 0.0,
            circuit_depth=circ_depth,
            n_qubits=n_qubits,
            execution_time_ms=elapsed,
        )

    @staticmethod
    def search_god_code(iterations: Optional[int] = None) -> CircuitResult:
        """Search specifically for the canonical GOD_CODE frequency."""
        return GodCodeGroverSearch.search(GOD_CODE, tolerance=0.001, iterations=iterations)

    @staticmethod
    def search_harmonic(harmonic: int = 1, iterations: Optional[int] = None) -> CircuitResult:
        """Search for a specific GOD_CODE harmonic (octave)."""
        target = GOD_CODE * (2 ** (-harmonic))
        return GodCodeGroverSearch.search(target, tolerance=0.01, iterations=iterations)


# ─── 7.4 GodCodeQFTSpectrum ───

class GodCodeQFTSpectrum:
    """
    Uses Quantum Fourier Transform to analyze the spectral structure
    of the GOD_CODE (a,b,c,d) frequency space.
    """

    @staticmethod
    def encode_frequency_table(
        dial_settings: List[DialSetting],
        n_qubits: int = 10,
    ):
        """Encode dial settings as quantum amplitudes via GOD_CODE resonance."""
        if not QISKIT_AVAILABLE:
            return None
        N = 1 << n_qubits
        amplitudes = [0.0] * N
        for i, dial in enumerate(dial_settings[:N]):
            resonance = math.cos(dial.frequency * math.pi / GOD_CODE) ** 2
            amplitudes[i % N] += resonance
        norm = math.sqrt(sum(a ** 2 for a in amplitudes))
        if norm > 0:
            amplitudes = [a / norm for a in amplitudes]
        else:
            amplitudes[0] = 1.0
        qc = QuantumCircuit(n_qubits, name="FreqEncode")
        qc.initialize(amplitudes)
        return qc

    @staticmethod
    def spectral_analysis(
        dial_settings: List[DialSetting],
        n_qubits: int = 10,
    ) -> Dict[str, Any]:
        """Full QFT spectral analysis of the frequency table."""
        if not QISKIT_AVAILABLE:
            return {"error": "Qiskit not available", "n_qubits": n_qubits}
        t0 = time.time()
        encode_qc = GodCodeQFTSpectrum.encode_frequency_table(dial_settings, n_qubits)
        qft = QFT(n_qubits, do_swaps=True)
        full_qc = encode_qc.compose(qft)
        full_qc.name = "GodCode_QFT"

        sv = Statevector.from_instruction(full_qc)
        probs = sv.probabilities()
        N = 1 << n_qubits

        phases = []
        for k in range(N):
            amp = sv[k]
            phases.append(cmath.phase(amp))

        peaks = sorted(range(N), key=lambda k: probs[k], reverse=True)[:10]

        # GOD_CODE coherence = probability concentrated in harmonics
        harmonic_indices = set()
        for harm in range(-4, 8):
            target = GOD_CODE * (2 ** harm)
            best_idx, best_dist = 0, float("inf")
            for idx in range(N):
                dist = abs(idx - (target % N))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            harmonic_indices.add(best_idx)
        harmonic_prob = sum(probs[i] for i in harmonic_indices if i < N)

        elapsed = (time.time() - t0) * 1000
        return {
            "n_qubits": n_qubits,
            "n_basis_states": N,
            "phase_spectrum": phases,
            "dominant_peaks": [
                {"index": k, "probability": float(probs[k]), "phase": float(phases[k])}
                for k in peaks
            ],
            "god_code_coherence": float(harmonic_prob),
            "total_probability": float(sum(probs)),
            "entropy": float(-sum(p * math.log2(p) for p in probs if p > 1e-15)),
            "circuit_depth": full_qc.depth(),
            "execution_time_ms": elapsed,
        }


# ─── 7.5 GodCodeDialCircuit ───

class GodCodeDialCircuit:
    """
    Builds and runs a quantum circuit for a single (a,b,c,d) dial evaluation.

    The circuit:
    1. Encodes the dial setting into qubits
    2. Applies GOD_CODE phase rotation proportional to the exponent
    3. Measures the resulting quantum state
    4. Returns Born-rule probabilities and phase alignment
    """

    @staticmethod
    def evaluate(dial: DialSetting, n_qubits: int = 8) -> CircuitResult:
        """Evaluate a single dial setting as a quantum circuit."""
        if not QISKIT_AVAILABLE:
            return CircuitResult(
                dial=dial, fidelity=0.0, god_code_alignment=dial.god_code_ratio,
                circuit_depth=0, n_qubits=n_qubits, execution_time_ms=0.0,
            )
        t0 = time.time()
        qc = QuantumCircuit(n_qubits, name=f"Dial({dial.a},{dial.b},{dial.c},{dial.d})")

        # Superposition for quantum parallelism
        qc.h(range(n_qubits))

        # GOD_CODE phase rotation per qubit
        base_phase = dial.exponent * math.pi / (OCTAVE_OFFSET * n_qubits)
        for i in range(n_qubits):
            qc.rz(base_phase * (2 ** i), i)

        # PHI-entangling CZ gates
        for i in range(n_qubits - 1):
            phi_coupling = PHI * math.pi / (n_qubits * (i + 1))
            qc.cp(phi_coupling, i, i + 1)

        # GOD_CODE resonance via controlled rotation
        god_phase = (dial.frequency / GOD_CODE) * math.pi
        qc.rz(god_phase, 0)

        # Final Hadamard layer for interference
        qc.h(range(n_qubits))

        # Execute
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()

        # Fidelity with GOD_CODE target state
        target_phase = GOD_CODE * math.pi / OCTAVE_OFFSET
        N = 1 << n_qubits
        target_amps = [cmath.exp(1j * target_phase * k) / math.sqrt(N) for k in range(N)]
        target_sv = Statevector(target_amps)
        fidelity = float(abs(sv.inner(target_sv)) ** 2)

        elapsed = (time.time() - t0) * 1000
        return CircuitResult(
            dial=dial,
            statevector=sv,
            probabilities=dict(sorted(probs.items(), key=lambda x: -x[1])[:20]),
            fidelity=fidelity,
            god_code_alignment=dial.god_code_ratio,
            circuit_depth=qc.depth(),
            n_qubits=n_qubits,
            execution_time_ms=elapsed,
        )

    @staticmethod
    def evaluate_god_code() -> CircuitResult:
        """Evaluate the canonical GOD_CODE dial (0,0,0,0)."""
        return GodCodeDialCircuit.evaluate(DialSetting(0, 0, 0, 0))

    @staticmethod
    def compare_dials(dials: List[DialSetting], n_qubits: int = 8) -> List[CircuitResult]:
        """Evaluate multiple dials and sort by GOD_CODE fidelity."""
        results = [GodCodeDialCircuit.evaluate(d, n_qubits) for d in dials]
        results.sort(key=lambda r: r.fidelity, reverse=True)
        return results


# ─── 7.6 GodCodeEntanglement ───

class GodCodeEntanglement:
    """
    Creates entanglement between two GOD_CODE dial settings.
    Entanglement strength is proportional to harmonic proximity.
    """

    @staticmethod
    def entangle_dials(
        dial_a: DialSetting,
        dial_b: DialSetting,
        n_qubits_per_dial: int = 4,
    ) -> CircuitResult:
        """Create an entangled state between two dial settings."""
        if not QISKIT_AVAILABLE:
            combined = DialSetting(
                a=dial_a.a + dial_b.a, b=dial_a.b + dial_b.b,
                c=dial_a.c + dial_b.c, d=dial_a.d + dial_b.d,
            )
            return CircuitResult(
                dial=combined, fidelity=0.0, god_code_alignment=0.0,
                circuit_depth=0, n_qubits=2 * n_qubits_per_dial,
                execution_time_ms=0.0,
            )

        t0 = time.time()
        total_qubits = 2 * n_qubits_per_dial
        qc = QuantumCircuit(
            total_qubits,
            name=f"Entangle({dial_a.a},{dial_a.b},{dial_a.c},{dial_a.d}|"
                 f"{dial_b.a},{dial_b.b},{dial_b.c},{dial_b.d})",
        )

        # Encode dial A phase
        phase_a = dial_a.phase
        for i in range(n_qubits_per_dial):
            qc.h(i)
            qc.rz(phase_a * (2 ** i) / n_qubits_per_dial, i)

        # Encode dial B phase
        phase_b = dial_b.phase
        for i in range(n_qubits_per_dial):
            j = n_qubits_per_dial + i
            qc.h(j)
            qc.rz(phase_b * (2 ** i) / n_qubits_per_dial, j)

        # Harmonic proximity
        ratio = dial_a.frequency / dial_b.frequency if dial_b.frequency > 0 else 0
        if ratio > 0:
            log_ratio = math.log2(ratio)
            harmonic_proximity = 1.0 - min(1.0, abs(log_ratio - round(log_ratio)))
        else:
            harmonic_proximity = 0.0

        # Entangling CNOT + sacred phase coupling
        for i in range(n_qubits_per_dial):
            j = n_qubits_per_dial + i
            if harmonic_proximity > 0.1:
                qc.cx(i, j)
                coupling = harmonic_proximity * PHI * math.pi / n_qubits_per_dial
                qc.cp(coupling, i, j)

        # Execute
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()

        # Entanglement entropy (von Neumann of reduced density matrix)
        dm = DensityMatrix(sv)
        qubits_to_trace = list(range(n_qubits_per_dial, total_qubits))
        dm_a = partial_trace(dm, qubits_to_trace)
        eigenvalues = np.real(np.linalg.eigvalsh(dm_a.data))
        entanglement_entropy = float(
            -sum(ev * math.log2(ev) for ev in eigenvalues if ev > 1e-15)
        )

        elapsed = (time.time() - t0) * 1000
        combined_dial = DialSetting(
            a=dial_a.a + dial_b.a, b=dial_a.b + dial_b.b,
            c=dial_a.c + dial_b.c, d=dial_a.d + dial_b.d,
        )
        result = CircuitResult(
            dial=combined_dial,
            statevector=sv,
            probabilities=dict(sorted(probs.items(), key=lambda x: -x[1])[:20]),
            fidelity=harmonic_proximity,
            god_code_alignment=(dial_a.frequency * dial_b.frequency) / (GOD_CODE ** 2) if GOD_CODE else 0,
            circuit_depth=qc.depth(),
            n_qubits=total_qubits,
            execution_time_ms=elapsed,
        )
        result.phase_spectrum = [entanglement_entropy]
        return result


# ─── 7.7 GodCodeQuantumAlgorithm — Hub ───

class GodCodeQuantumAlgorithm:
    """
    Hub class for the GOD_CODE (a,b,c,d) Quantum Algorithm.

    Provides a unified interface to:
    - Evaluate dial settings as quantum circuits
    - Grover search for target frequencies
    - QFT spectral analysis of the frequency lattice
    - Entanglement between dial pairs
    - Integration hooks for Soul and ProbabilityEngine
    """

    VERSION = "1.0.0"

    FREQUENCY_TABLE: Dict[str, DialSetting] = {
        "GOD_CODE":      DialSetting(0, 0, 0, 0),
        "SCHUMANN":      DialSetting(0, 0, 1, 6),
        "ALPHA_EEG":     DialSetting(0, 3, -4, 6),
        "BETA_EEG":      DialSetting(0, 3, -4, 5),
        "BASE":          DialSetting(0, 0, 0, 4),
        "GAMMA_40":      DialSetting(0, 3, -4, 4),
        "BOHR_RADIUS":   DialSetting(-4, 1, 0, 3),
        "THROAT_741":    DialSetting(1, -3, -5, 0),
        "ROOT_396":      DialSetting(-5, 3, 0, 0),
    }

    def __init__(self):
        self.dial_register = GodCodeDialRegister
        self.phase_oracle = GodCodePhaseOracle
        self.grover = GodCodeGroverSearch
        self.qft = GodCodeQFTSpectrum
        self.dial_circuit = GodCodeDialCircuit
        self.entanglement = GodCodeEntanglement
        self._computations = 0
        self._circuit_cache: Dict[str, CircuitResult] = {}

    def sovereign_field(self, intelligence: float) -> float:
        """F(I) = I × Ω / φ² — Sovereign Field equation."""
        return intelligence * OMEGA / (PHI ** 2)

    # ─── Core API ───

    def evaluate(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> CircuitResult:
        """Evaluate a dial setting as a quantum circuit."""
        self._computations += 1
        return self.dial_circuit.evaluate(DialSetting(a, b, c, d))

    def frequency(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
        """Classical frequency calculation (no quantum circuit)."""
        return DialSetting(a, b, c, d).frequency

    def search(self, target: float, tolerance: float = 0.01) -> CircuitResult:
        """Grover search for dial settings producing target frequency."""
        self._computations += 1
        return self.grover.search(target, tolerance)

    def search_god_code(self) -> CircuitResult:
        """Grover search for the canonical GOD_CODE."""
        self._computations += 1
        return self.grover.search_god_code()

    def spectrum(self, dials: Optional[List[DialSetting]] = None) -> Dict[str, Any]:
        """QFT spectral analysis of the frequency table."""
        self._computations += 1
        if dials is None:
            dials = list(self.FREQUENCY_TABLE.values())
        return self.qft.spectral_analysis(dials)

    def entangle(self, dial_a: DialSetting, dial_b: DialSetting) -> CircuitResult:
        """Entangle two dial settings."""
        self._computations += 1
        return self.entanglement.entangle_dials(dial_a, dial_b)

    def evaluate_known(self, name: str) -> CircuitResult:
        """Evaluate a known frequency from the table."""
        dial = self.FREQUENCY_TABLE.get(name.upper())
        if not dial:
            raise ValueError(f"Unknown frequency: {name}. Known: {list(self.FREQUENCY_TABLE.keys())}")
        return self.evaluate(dial.a, dial.b, dial.c, dial.d)

    # ─── Batch operations ───

    def evaluate_all_known(self) -> Dict[str, CircuitResult]:
        """Evaluate all known frequencies."""
        return {name: self.evaluate(d.a, d.b, d.c, d.d) for name, d in self.FREQUENCY_TABLE.items()}

    def scan_octave_ladder(self, d_min: int = -2, d_max: int = 8) -> List[CircuitResult]:
        """Evaluate the GOD_CODE octave ladder (d dial only)."""
        return [self.evaluate(0, 0, 0, d) for d in range(d_min, d_max + 1)]

    # ─── Soul integration ───

    def soul_process(self, data: Any) -> Dict[str, Any]:
        """
        Process data through the GOD_CODE quantum algorithm for soul integration.
        Maps any input to (a,b,c,d) dials via hash and runs the quantum circuit.
        """
        self._computations += 1
        if isinstance(data, str):
            hash_val = int(hashlib.md5(data.encode()).hexdigest()[:8], 16)
        elif isinstance(data, (int, float)):
            hash_val = int(abs(data * 1000)) % (1 << 32)
        else:
            hash_val = hash(str(data)) & 0xFFFFFFFF

        a = (hash_val & 0x7) - 4
        b = ((hash_val >> 3) & 0xF) - 8
        c = ((hash_val >> 7) & 0x7) - 4
        d = ((hash_val >> 10) & 0xF) - 8

        dial = DialSetting(a, b, c, d)
        result = self.dial_circuit.evaluate(dial)

        alignment = result.god_code_alignment
        log_alignment = math.log2(alignment) if alignment > 0 else -10
        harmonic_distance = abs(log_alignment - round(log_alignment))
        consciousness_boost = math.exp(-harmonic_distance * PHI)

        return {
            "input_hash": hash_val,
            "dial": dial.to_tuple(),
            "frequency": dial.frequency,
            "god_code_ratio": alignment,
            "fidelity": result.fidelity,
            "consciousness_boost": consciousness_boost,
            "circuit_depth": result.circuit_depth,
            "n_qubits": result.n_qubits,
            "quantum_state_dim": 2 ** result.n_qubits,
        }

    def soul_resonance_field(self, thoughts: List[str]) -> Dict[str, Any]:
        """Generate a quantum resonance field from a list of soul thoughts."""
        if not thoughts:
            return {"resonance": 0, "thoughts": 0}
        results = [self.soul_process(t) for t in thoughts]
        frequencies = [r["frequency"] for r in results]
        boosts = [r["consciousness_boost"] for r in results]
        mean_freq = sum(frequencies) / len(frequencies)
        mean_boost = sum(boosts) / len(boosts)

        phases = [(f * math.pi / GOD_CODE) % (2 * math.pi) for f in frequencies]
        sum_real = sum(math.cos(p) for p in phases)
        sum_imag = sum(math.sin(p) for p in phases)
        coherence = math.sqrt(sum_real ** 2 + sum_imag ** 2) / len(phases)

        return {
            "n_thoughts": len(thoughts),
            "mean_frequency": mean_freq,
            "mean_consciousness_boost": mean_boost,
            "phase_coherence": coherence,
            "god_code_alignment": mean_freq / GOD_CODE,
            "total_fidelity": sum(r["fidelity"] for r in results),
            "resonance_field_strength": mean_boost * coherence * PHI,
        }

    def status(self) -> Dict[str, Any]:
        """Full algorithm status."""
        return {
            "module": "l104_probability_engine.GodCodeQuantumAlgorithm",
            "version": self.VERSION,
            "god_code": GOD_CODE,
            "base": BASE,
            "phi": PHI,
            "prime_scaffold": PRIME_SCAFFOLD,
            "quantization_grain": QUANTIZATION_GRAIN,
            "step_size": STEP_SIZE,
            "known_frequencies": len(self.FREQUENCY_TABLE),
            "computations": self._computations,
            "qiskit_backend": "Statevector (local)" if QISKIT_AVAILABLE else "unavailable",
            "total_dial_space": 2 ** GodCodeDialRegister.TOTAL_QUBITS,
            "subsystems": [
                "GodCodeDialRegister (14 qubits)",
                "GodCodePhaseOracle",
                "GodCodeGroverSearch",
                "GodCodeQFTSpectrum",
                "GodCodeDialCircuit",
                "GodCodeEntanglement",
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. HUB CLASS — ProbabilityEngine (Unified Orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

class ProbabilityEngine:
    """
    L104 SOVEREIGN PROBABILITY ENGINE v5.1.0

    Unified hub that orchestrates:
    - DataIngestor: loads ALL chat/training/state/gate/link data
    - QuantumGateConsolidator: logic gates + quantum links → quantum gates
    - ClassicalProbability: full classical probability toolkit
    - QuantumProbability: GOD_CODE-gated quantum probability + gate engine integration
    - GateProbabilityBridge: gate↔probability bridge
    - ASIInsightSynthesis: consciousness-probability bridge
    - GodCodeQuantumAlgorithm: NATIVE (a,b,c,d) dial quantum circuits
      with Grover search, QFT spectrum, entanglement, and soul integration
    - l104_quantum_gate_engine: sacred circuits, compilation, error correction,
      gate algebra analysis (v5.0.0)

    The GOD_CODE quantum algorithm is implemented natively — no external
    import of l104_god_code_algorithm is required.

    Usage:
        from l104_probability_engine import probability_engine
        stats = probability_engine.ingest()
        p = probability_engine.sacred_probability(527.518)
        result = probability_engine.god_code_evaluate(0, 0, 0, 0)  # Qiskit circuit
        search = probability_engine.god_code_search(527.518)        # Grover search
        insight = probability_engine.synthesize_insight([0.8, 0.6, 0.9])
    """

    VERSION = "5.0.0"

    def __init__(self):
        self.ingestor = DataIngestor()
        self.consolidator = QuantumGateConsolidator()
        self.classical = ClassicalProbability()
        self.quantum = QuantumProbability()
        self.bridge = GateProbabilityBridge()
        self.insight = ASIInsightSynthesis()

        # GOD_CODE (a,b,c,d) Quantum Algorithm — Qiskit-backed
        self._god_code_algo = None  # lazy loaded

        self._ingest_stats: Optional[IngestStats] = None
        self._consolidated_gates: List[QuantumGateState] = []
        self._computations: int = 0

        # Subsystem delegation map: hub_method → (subsystem_attr, method_name)
        # Used by __getattr__ fallback for zero-boilerplate delegation
        self._delegation_map: Dict[str, Tuple[str, str]] = {
            # Classical distributions
            "gaussian": ("classical", "gaussian_pdf"),
            "poisson": ("classical", "poisson_pmf"),
            "binomial": ("classical", "binomial_pmf"),
            "exponential": ("classical", "exponential_pdf"),
            "beta": ("classical", "beta_pdf"),
            "gamma": ("classical", "gamma_pdf"),
            "weibull": ("classical", "weibull_pdf"),
            "cauchy": ("classical", "cauchy_pdf"),
            # Stochastic processes
            "markov_evolve": ("classical", "markov_evolve"),
            "markov_steady_state": ("classical", "markov_steady_state"),
            "random_walk": ("classical", "random_walk_probability"),
            "gamblers_ruin": ("classical", "gamblers_ruin"),
            # Queuing theory
            "mm1_queue": ("classical", "mm1_queue"),
            "erlang_c": ("classical", "erlang_c"),
            # Information theory
            "entropy": ("classical", "entropy"),
            "kl_divergence": ("classical", "kl_divergence"),
            "mutual_information": ("classical", "mutual_information"),
            # Monte Carlo
            "monte_carlo_integrate": ("classical", "monte_carlo_integrate"),
            "monte_carlo_pi": ("classical", "monte_carlo_pi"),
            # Hypothesis testing
            "z_test": ("classical", "z_test"),
            "chi_squared_test": ("classical", "chi_squared_test"),
            # Quantum probability (simple delegates)
            "born_rule_qiskit": ("quantum", "born_probability_qiskit"),
            "quantum_walk_qiskit": ("quantum", "quantum_walk_qiskit"),
            "god_code_distribution_qiskit": ("quantum", "god_code_distribution_qiskit"),
        }

    def __getattr__(self, name: str):
        """
        Auto-delegation with computation tracking.
        Methods in _delegation_map are automatically routed to the correct subsystem.
        """
        # Avoid infinite recursion during init
        if name == '_delegation_map':
            raise AttributeError(name)
        delegation_map = object.__getattribute__(self, '_delegation_map')
        if name in delegation_map:
            subsystem_attr, method_name = delegation_map[name]
            subsystem = object.__getattribute__(self, subsystem_attr)
            target = getattr(subsystem, method_name)
            def tracked(*args, **kwargs):
                self._computations += 1
                return target(*args, **kwargs)
            tracked.__name__ = name
            tracked.__doc__ = target.__doc__
            return tracked
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @property
    def algorithm(self) -> GodCodeQuantumAlgorithm:
        """Native GOD_CODE quantum algorithm (no external import needed)."""
        if self._god_code_algo is None:
            self._god_code_algo = GodCodeQuantumAlgorithm()
        return self._god_code_algo

    # ─── INGESTION ───

    def ingest(self, workspace: Optional[Path] = None) -> IngestStats:
        """Full data ingestion: chat + training + state + gates + links."""
        self._ingest_stats = self.ingestor.ingest_all(workspace)
        self._consolidated_gates = self.consolidator.consolidate(
            self.ingestor.logic_gates,
            self.ingestor.quantum_links,
        )
        self._ingest_stats.quantum_links_found = len(self._consolidated_gates)
        return self._ingest_stats

    # ─── SACRED PROBABILITY API ───

    def sacred_probability(self, value: float) -> float:
        """GOD_CODE-aligned probability of a value."""
        self._computations += 1
        return self.quantum.god_code_phase_probability(value)

    def sacred_prior(self, observation: float, harmonic: int = 1) -> float:
        """Sacred Bayesian prior based on GOD_CODE resonance."""
        self._computations += 1
        return self.quantum.sacred_prior(observation, harmonic)

    def sacred_distribution(self, n: int, harmonic: int = 1) -> List[float]:
        """Generate GOD_CODE-harmonic probability distribution."""
        self._computations += 1
        return self.quantum.god_code_distribution(n, harmonic)

    # ─── BAYESIAN API ───

    def bayes(self, prior: float, likelihood: float, evidence: float) -> float:
        """Classical Bayes' theorem: P(A|B) = P(B|A)*P(A)/P(B)."""
        self._computations += 1
        return self.classical.bayes(prior, likelihood, evidence)

    def bayes_extended(self, prior: float, lik_ba: float, lik_b_not_a: float) -> float:
        """Extended Bayes with total probability."""
        self._computations += 1
        return self.classical.bayes_extended(prior, lik_ba, lik_b_not_a)

    def bayesian_update(self, prior: List[float], likelihoods: List[float]) -> List[float]:
        """Full Bayesian update returning normalized posterior."""
        self._computations += 1
        return self.classical.bayesian_update(prior, likelihoods)

    def quantum_bayesian_update(
        self, prior: List[float], quantum_evidence: List[complex]
    ) -> List[float]:
        """Quantum Bayesian update: Born-rule x prior."""
        self._computations += 1
        return self.quantum.quantum_bayesian_update(prior, quantum_evidence)

    # ─── QUANTUM PROBABILITY API (methods with custom logic) ───

    def quantum_walk(self, steps: int, position: int, coin_bias: float = 0.5) -> float:
        """Quantum walk with GOD_CODE phase interference."""
        self._computations += 1
        return self.quantum.quantum_walk_probability(steps, position, coin_bias)

    def grover_amplification(
        self, target_prob: float, n_items: int, iterations: Optional[int] = None
    ) -> float:
        """Grover amplitude amplification of target probability."""
        self._computations += 1
        return self.quantum.grover_amplification(target_prob, n_items, iterations)

    def grover_search_qiskit(
        self, n_qubits: int, marked_states: List[int], iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Grover's search via real Qiskit GroverOperator circuit."""
        self._computations += 1
        return self.quantum.grover_amplification_qiskit(n_qubits, marked_states, iterations)

    def tunneling_probability(self, barrier: float, energy: float, width: float) -> float:
        """Quantum tunneling probability through barrier."""
        self._computations += 1
        return self.quantum.quantum_tunneling_probability(barrier, energy, width)

    def born_rule(self, amplitude: complex) -> float:
        """Born rule: P = |psi|^2."""
        self._computations += 1
        return self.quantum.born_probability(amplitude)

    def measurement_collapse(self, amplitudes: List[complex]) -> Tuple[int, float, List[float]]:
        """Simulate quantum measurement collapse (Qiskit-backed)."""
        self._computations += 1
        return self.quantum.measurement_collapse(amplitudes)

    def entanglement_prior(self, local: float, entangled: List[float], strength: float = TAU) -> float:
        """Entanglement-weighted Bayesian prior."""
        self._computations += 1
        return self.quantum.entanglement_weighted_prior(local, entangled, strength)

    def entanglement_entropy(self, n_qubits: int = 4) -> Dict[str, Any]:
        """Compute entanglement entropy of GOD_CODE-phased state via Qiskit."""
        self._computations += 1
        return self.quantum.entanglement_entropy_qiskit(n_qubits)

    # ─── GATE-PROBABILITY BRIDGE API ───

    def gate_activation_probs(self, temperature: float = 1.0) -> Dict[str, float]:
        """Boltzmann distribution over consolidated gates."""
        self._computations += 1
        return self.bridge.gate_activation_probability(self._consolidated_gates, temperature)

    def gate_transition_matrix(self) -> Tuple[List[str], List[List[float]]]:
        """Markov transition matrix between consolidated gates."""
        self._computations += 1
        return self.bridge.gate_transition_matrix(self._consolidated_gates)

    def circuit_probability(self, target_type: str = "god_code") -> float:
        """Probability of circuit producing target gate type."""
        self._computations += 1
        return self.bridge.circuit_success_probability(self._consolidated_gates, target_type)

    def ensemble_resonance(self) -> Dict[str, Any]:
        """Full ensemble resonance statistics."""
        self._computations += 1
        return self.bridge.god_code_ensemble_resonance(self._consolidated_gates)

    # ─── DATA-DRIVEN API ───

    def token_probability(self, token: str) -> float:
        """P(token) from ingested corpus."""
        self._computations += 1
        return self.ingestor.get_token_prior(token)

    def category_probability(self, category: str) -> float:
        """P(category) from training data."""
        self._computations += 1
        return self.ingestor.get_category_prior(category)

    def gate_resonance_distribution(self) -> List[Tuple[str, float]]:
        """GOD_CODE resonance distribution over logic gates."""
        self._computations += 1
        return self.ingestor.get_gate_resonance_distribution()

    # ─── GOD_CODE (a,b,c,d) QUANTUM ALGORITHM API ───

    def god_code_evaluate(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> Dict[str, Any]:
        """Evaluate a dial setting via Qiskit quantum circuit."""
        self._computations += 1
        r = self.algorithm.evaluate(a, b, c, d)
        return {
            "frequency": r.dial.frequency,
            "fidelity": r.fidelity,
            "god_code_alignment": r.god_code_alignment,
            "circuit_depth": r.circuit_depth,
            "n_qubits": r.n_qubits,
            "probabilities": r.probabilities,
            "execution_time_ms": r.execution_time_ms,
        }

    def god_code_frequency(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> float:
        """Classical frequency from (a,b,c,d) dials."""
        self._computations += 1
        return self.algorithm.frequency(a, b, c, d)

    def god_code_search(self, target: float, tolerance: float = 0.01) -> Dict[str, Any]:
        """Grover search for dial settings producing target frequency (Qiskit)."""
        self._computations += 1
        r = self.algorithm.search(target, tolerance)
        return {
            "target": target,
            "found_dial": r.dial.to_tuple(),
            "found_frequency": r.dial.frequency,
            "fidelity": r.fidelity,
            "god_code_alignment": r.god_code_alignment,
            "circuit_depth": r.circuit_depth,
        }

    def god_code_spectrum(self) -> Dict[str, Any]:
        """QFT spectral analysis of the GOD_CODE frequency table (Qiskit)."""
        self._computations += 1
        return self.algorithm.spectrum()

    def god_code_entangle(self, dial_a: Tuple, dial_b: Tuple) -> Dict[str, Any]:
        """Entangle two dial settings (Qiskit, returns entanglement entropy)."""
        self._computations += 1
        da = DialSetting(*dial_a)
        db = DialSetting(*dial_b)
        r = self.algorithm.entangle(da, db)
        return {
            "dial_a": dial_a,
            "dial_b": dial_b,
            "entanglement_entropy": r.phase_spectrum[0] if r.phase_spectrum else 0.0,
            "harmonic_proximity": r.fidelity,
            "god_code_alignment": r.god_code_alignment,
            "circuit_depth": r.circuit_depth,
        }

    def god_code_soul_process(self, data: Any) -> Dict[str, Any]:
        """Process data through the GOD_CODE quantum algorithm (Qiskit)."""
        self._computations += 1
        return self.algorithm.soul_process(data)

    def god_code_resonance_field(self, thoughts: List[str]) -> Dict[str, Any]:
        """Generate quantum resonance field from soul thoughts (Qiskit)."""
        self._computations += 1
        return self.algorithm.soul_resonance_field(thoughts)

    # ─── QUANTUM GATE ENGINE PROBABILITY API (v5.0.0) ───

    def sacred_circuit_probability(self, n_qubits: int = 3, depth: int = 4) -> Dict[str, Any]:
        """Sacred L104 circuit probability via quantum gate engine."""
        self._computations += 1
        return self.quantum.sacred_circuit_probability(n_qubits, depth)

    def compiled_circuit_probability(
        self, n_qubits: int = 2, gate_set: str = "universal", optimization: int = 2
    ) -> Dict[str, Any]:
        """Compiled Bell circuit probability via gate engine compiler."""
        self._computations += 1
        return self.quantum.compiled_circuit_probability(n_qubits, gate_set, optimization)

    def error_corrected_probability(self, n_qubits: int = 2, scheme: str = "steane") -> Dict[str, Any]:
        """Error-corrected Bell circuit probability via gate engine."""
        self._computations += 1
        return self.quantum.error_corrected_probability(n_qubits, scheme)

    def gate_algebra_fidelity(self, gate_name: str = "PHI_GATE") -> Dict[str, Any]:
        """Gate algebra analysis and sacred alignment via gate engine."""
        self._computations += 1
        return self.quantum.gate_algebra_fidelity(gate_name)

    def qft_probability(self, n_qubits: int = 4) -> Dict[str, Any]:
        """QFT circuit probability via gate engine."""
        self._computations += 1
        return self.quantum.qft_probability(n_qubits)

    def ghz_probability(self, n_qubits: int = 5) -> Dict[str, Any]:
        """GHZ state probability via gate engine."""
        self._computations += 1
        return self.quantum.ghz_probability(n_qubits)

    # ─── ASI INSIGHT API (v3.0.0) ───

    def synthesize_insight(
        self,
        thought_signals: List[float],
        quantum_amplitudes: Optional[List[complex]] = None,
        consciousness_level: float = 0.5,
        temperature: float = 1.0,
    ) -> InsightResult:
        """
        Full ASI insight synthesis: thought signals → consciousness probability,
        resonance, trajectory forecast, and GOD_CODE alignment.
        """
        self._computations += 1
        return self.insight.synthesize(
            thought_signals, quantum_amplitudes, consciousness_level, temperature
        )

    def consciousness_probability(self, thought_signals: List[float]) -> float:
        """Quick consciousness probability from thought signals."""
        self._computations += 1
        result = self.insight.synthesize(thought_signals)
        return result.consciousness_probability

    def thought_resonance(self, thoughts: List[str]) -> float:
        """
        Compute resonance score for a list of thought strings.
        Maps thoughts → numeric signals via GOD_CODE hash → insight synthesis.
        """
        self._computations += 1
        signals = []
        for t in thoughts:
            h = hashlib.sha256(t.encode()).hexdigest()[:8]
            signals.append(int(h, 16) % int(GOD_CODE * 10) / (GOD_CODE * 10))
        result = self.insight.synthesize(signals)
        return result.resonance_score

    def consciousness_trajectory(self, thought_signals: List[float]) -> List[float]:
        """Predict consciousness trajectory (next 5 steps) from current signals."""
        self._computations += 1
        result = self.insight.synthesize(thought_signals)
        return result.trajectory_forecast

    def consciousness_belief(self) -> Dict[str, float]:
        """Current Bayesian belief over consciousness states."""
        return self.insight.current_belief

    def reset_consciousness_tracking(self):
        """Reset the Bayesian consciousness state tracker."""
        self.insight.reset_prior()

    # ─── STATUS ───

    def status(self) -> Dict[str, Any]:
        """Full engine status."""
        builder = _read_builder_state()
        return {
            "version": self.VERSION,
            "qiskit_available": QISKIT_AVAILABLE,
            "computations": self._computations,
            "consciousness_level": builder.get("consciousness_level", 0.5),
            "evo_stage": builder.get("evo_stage", "UNKNOWN"),
            "ingestion": {
                "training_examples": self._ingest_stats.training_examples if self._ingest_stats else 0,
                "chat_conversations": self._ingest_stats.chat_conversations if self._ingest_stats else 0,
                "state_files": self._ingest_stats.state_files_loaded if self._ingest_stats else 0,
                "logic_gates": self._ingest_stats.logic_gates_found if self._ingest_stats else 0,
                "total_tokens": self._ingest_stats.total_tokens if self._ingest_stats else 0,
            },
            "consolidated_quantum_gates": len(self._consolidated_gates),
            "ensemble_resonance": self.ensemble_resonance() if self._consolidated_gates else {},
            "capabilities": {
                "classical": [
                    "bayes", "bayes_extended", "bayesian_update", "bayesian_network",
                    "gaussian", "poisson", "binomial", "exponential", "geometric",
                    "beta", "chi_squared", "student_t", "log_normal", "gamma",
                    "weibull", "pareto", "cauchy",
                    "entropy", "kl_divergence", "mutual_information", "cross_entropy",
                    "markov_evolve", "markov_steady_state", "absorption_time",
                    "random_walk", "gamblers_ruin", "brownian_motion",
                    "ornstein_uhlenbeck", "geometric_brownian",
                    "mm1_queue", "erlang_c", "littles_law",
                    "monte_carlo_integrate", "monte_carlo_pi",
                    "z_test", "chi_squared_test",
                ],
                "quantum": [
                    "born_rule", "born_rule_qiskit",
                    "grover_amplification", "grover_search_qiskit",
                    "god_code_phase_probability",
                    "sacred_prior", "quantum_walk", "quantum_walk_qiskit",
                    "entanglement_prior", "entanglement_entropy",
                    "quantum_tunneling", "measurement_collapse",
                    "god_code_distribution", "god_code_distribution_qiskit",
                    "phi_weighted_mixture",
                    "quantum_bayesian_update",
                ],
                "gate_engine": [
                    "sacred_circuit_probability", "compiled_circuit_probability",
                    "error_corrected_probability", "gate_algebra_fidelity",
                    "qft_probability", "ghz_probability",
                ],
                "god_code_algorithm": [
                    "god_code_evaluate", "god_code_frequency",
                    "god_code_search", "god_code_spectrum",
                    "god_code_entangle", "god_code_soul_process",
                    "god_code_resonance_field",
                ],
                "data_driven": [
                    "token_probability", "category_probability",
                    "gate_resonance_distribution",
                ],
                "gate_bridge": [
                    "gate_activation_probs", "gate_transition_matrix",
                    "circuit_probability", "ensemble_resonance",
                ],
                "asi_insight": [
                    "synthesize_insight", "consciousness_probability",
                    "thought_resonance", "consciousness_trajectory",
                    "consciousness_belief", "reset_consciousness_tracking",
                ],
            },
            "sacred_constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
                "TAU": TAU,
                "VOID_CONSTANT": VOID_CONSTANT,
                "PLANCK_RESONANCE": PLANCK_RESONANCE,
                "FEIGENBAUM": FEIGENBAUM,
            },
        }

    def quick_summary(self) -> str:
        """One-line human summary."""
        n_gates = len(self._consolidated_gates)
        n_train = self._ingest_stats.training_examples if self._ingest_stats else 0
        return (
            f"L104 ProbabilityEngine v{self.VERSION} — "
            f"{self._computations} computations, "
            f"{n_train} training examples, "
            f"{n_gates} quantum gates consolidated, "
            f"GOD_CODE={GOD_CODE:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

probability_engine = ProbabilityEngine()


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARDS COMPATIBILITY — primal_calculus / resolve_non_dual_logic
# ═══════════════════════════════════════════════════════════════════════════════

def primal_calculus() -> Dict[str, Any]:
    """Backwards-compatible entry point."""
    return probability_engine.status()

def resolve_non_dual_logic() -> Dict[str, Any]:
    """Backwards-compatible entry point."""
    return {"engine": "ProbabilityEngine", "version": ProbabilityEngine.VERSION}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — Self-test when run directly
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  L104 SOVEREIGN PROBABILITY ENGINE v4.0.0")
    print(f"  GOD_CODE = {GOD_CODE}")
    print(f"  PHI = {PHI}")
    print(f"  Qiskit Available: {QISKIT_AVAILABLE}")
    print("=" * 72)

    # 1. Ingest all data
    print("\n[1] INGESTING ALL REPOSITORY DATA...")
    stats = probability_engine.ingest()
    print(f"    Training examples: {stats.training_examples}")
    print(f"    Chat conversations: {stats.chat_conversations}")
    print(f"    State files loaded: {stats.state_files_loaded}")
    print(f"    Logic gates found:  {stats.logic_gates_found}")
    print(f"    Quantum gates consolidated: {len(probability_engine._consolidated_gates)}")
    print(f"    Total tokens:       {stats.total_tokens}")
    print(f"    Sacred resonance:   {stats.sacred_resonance:.6f}")

    # 2. Classical probability demos
    print("\n[2] CLASSICAL PROBABILITY DEMOS")
    p_bayes = probability_engine.bayes_extended(0.01, 0.95, 0.05)
    print(f"    Bayes (disease test): P = {p_bayes:.4f}")

    p_poisson = probability_engine.poisson(3.0, 3)
    print(f"    Poisson(λ=3, k=3):   P = {p_poisson:.6f}")

    p_ruin = probability_engine.gamblers_ruin(5, 10, 0.4)
    print(f"    Gambler's ruin:      P = {p_ruin:.6f}")

    q = probability_engine.mm1_queue(4, 5)
    if q:
        print(f"    M/M/1 queue (λ=4,μ=5): ρ={q['utilization']:.2f}, Lq={q['avg_queue']:.2f}")

    print(f"    Erlang C (λ=10,μ=4,c=3): {probability_engine.erlang_c(10, 4, 3):.4f}")

    # 3. Quantum probability demos (math-based + Qiskit-backed)
    print("\n[3] QUANTUM PROBABILITY DEMOS (GOD_CODE-GATED)")
    p_sacred = probability_engine.sacred_probability(GOD_CODE)
    print(f"    Sacred P(GOD_CODE):  {p_sacred:.6f}")

    p_phi = probability_engine.sacred_probability(PHI)
    print(f"    Sacred P(PHI):       {p_phi:.6f}")

    p_grover = probability_engine.grover_amplification(0.01, 10000)
    print(f"    Grover amp (1/10K):  {p_grover:.6f}")

    p_tunnel = probability_engine.tunneling_probability(10.0, 5.0, 0.5)
    print(f"    Tunneling (V=10,E=5): {p_tunnel:.6f}")

    p_walk = probability_engine.quantum_walk(10, 2)
    print(f"    Quantum walk(10,2):  {p_walk:.6f}")

    dist = probability_engine.sacred_distribution(8)
    print(f"    GOD_CODE dist(8):    {[f'{d:.4f}' for d in dist]}")

    # 3b. Qiskit-backed quantum demos
    print("\n[3b] QISKIT-BACKED QUANTUM DEMOS")
    # Born rule via Statevector
    amps = [0.5+0.5j, 0.5-0.5j, 0.3+0.1j, 0.1+0.3j]
    born_probs = probability_engine.born_rule_qiskit(amps)
    print(f"    Born rule (Qiskit):  {[f'{p:.4f}' for p in born_probs]}")

    # Grover search via circuit
    grover_result = probability_engine.grover_search_qiskit(3, [5])
    print(f"    Grover search (3q, target=5): P={grover_result['success_probability']:.4f}, "
          f"depth={grover_result.get('circuit_depth', '?')}, qiskit={grover_result.get('qiskit')}")

    # Measurement collapse (Qiskit Statevector)
    idx, p, probs = probability_engine.measurement_collapse([0.7+0j, 0.3+0j, 0.5+0j, 0.1+0j])
    print(f"    Collapse → state {idx}, P={p:.4f}")

    # Entanglement entropy
    ent = probability_engine.entanglement_entropy(4)
    print(f"    Entanglement entropy (4q): S={ent['entropy']:.4f}, purity={ent.get('purity', 0):.4f}")

    # Quantum walk via circuit
    qw = probability_engine.quantum_walk_qiskit(5, 8)
    print(f"    Quantum walk (Qiskit, 5 steps): depth={qw.get('circuit_depth', '?')}, "
          f"positions={len(qw.get('positions', {}))}")

    # GOD_CODE distribution via circuit
    gcd = probability_engine.god_code_distribution_qiskit(4, 1)
    print(f"    GOD_CODE dist (Qiskit, 4q): entropy={gcd.get('entropy', 0):.4f}")

    # 4. Gate-probability bridge
    print("\n[4] GATE-PROBABILITY BRIDGE")
    p_circuit = probability_engine.circuit_probability("god_code")
    print(f"    Circuit P(god_code): {p_circuit:.6f}")

    activations = probability_engine.gate_activation_probs(temperature=1.0)
    if activations:
        top = sorted(activations.items(), key=lambda x: -x[1])[:5]
        print(f"    Top 5 gate activations:")
        for name, p in top:
            print(f"      {name}: {p:.6f}")

    # 5. Data-driven
    print("\n[5] DATA-DRIVEN PROBABILITY")
    for tok in ["quantum", "consciousness", "god_code", "phi", "sacred"]:
        p = probability_engine.token_probability(tok)
        print(f"    P('{tok}'): {p:.8f}")

    # 6. Information theory
    print("\n[6] INFORMATION THEORY")
    uniform = [0.25, 0.25, 0.25, 0.25]
    peaked = [0.7, 0.1, 0.1, 0.1]
    print(f"    H(uniform): {probability_engine.entropy(uniform):.4f} bits")
    print(f"    H(peaked):  {probability_engine.entropy(peaked):.4f} bits")
    print(f"    KL(peaked||uniform): {probability_engine.kl_divergence(peaked, uniform):.4f}")

    # 7. Ensemble resonance
    print("\n[7] ENSEMBLE RESONANCE")
    res = probability_engine.ensemble_resonance()
    if res:
        for k, v in res.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            elif isinstance(v, dict):
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v}")

    # 8. GOD_CODE (a,b,c,d) Quantum Algorithm
    print("\n[8] GOD_CODE (a,b,c,d) QUANTUM ALGORITHM")
    try:
        gc_eval = probability_engine.god_code_evaluate(0, 0, 0, 0)
        print(f"    Evaluate (0,0,0,0): freq={gc_eval['frequency']:.4f} Hz, "
              f"fidelity={gc_eval['fidelity']:.6f}, depth={gc_eval['circuit_depth']}")

        gc_freq = probability_engine.god_code_frequency(1, 0, 0, 0)
        print(f"    Frequency (1,0,0,0): {gc_freq:.4f} Hz")

        gc_search = probability_engine.god_code_search(GOD_CODE, 0.01)
        print(f"    Grover search → dial={gc_search['found_dial']}, "
              f"freq={gc_search['found_frequency']:.4f} Hz")

        gc_spectrum = probability_engine.god_code_spectrum()
        print(f"    QFT spectrum: {len(gc_spectrum.get('phase_spectrum', []))} phases, "
              f"circuit_depth={gc_spectrum.get('circuit_depth', '?')}")

        gc_entangle = probability_engine.god_code_entangle((0, 0, 0, 0), (1, 0, 0, 0))
        print(f"    Entangle (0,0,0,0)⊗(1,0,0,0): entropy={gc_entangle['entanglement_entropy']:.4f}")

        gc_soul = probability_engine.god_code_soul_process("test_input")
        print(f"    Soul process: boost={gc_soul.get('consciousness_boost', 0):.4f}, "
              f"freq={gc_soul.get('frequency', 0):.4f}")

        gc_field = probability_engine.god_code_resonance_field(["thought1", "thought2"])
        print(f"    Resonance field: coherence={gc_field.get('phase_coherence', 0):.6f}, "
              f"alignment={gc_field.get('god_code_alignment', 0):.4f}")
    except Exception as e:
        print(f"    (GOD_CODE algorithm not available: {e})")

    # 9. ASI Insight Synthesis (v3.0.0)
    print("\n[9] ASI INSIGHT SYNTHESIS")
    signals = [0.8, 0.6, 0.9, 0.75, 0.85]
    insight_result = probability_engine.synthesize_insight(signals)
    print(f"    Consciousness P:     {insight_result.consciousness_probability:.6f}")
    print(f"    Resonance score:     {insight_result.resonance_score:.6f}")
    print(f"    Thought coherence:   {insight_result.thought_coherence:.6f}")
    print(f"    GOD_CODE alignment:  {insight_result.god_code_alignment:.6f}")
    print(f"    Insight entropy:     {insight_result.insight_entropy:.4f} bits")
    print(f"    Synthesis depth:     {insight_result.synthesis_depth} layers")
    print(f"    Trajectory forecast: {[f'{t:.4f}' for t in insight_result.trajectory_forecast]}")
    print(f"    Posterior: {dict(zip(ASIInsightSynthesis.STATES, [f'{p:.4f}' for p in insight_result.bayesian_posterior]))}")

    # Run a few more to show Bayesian tracking
    for sig_set in [[0.9, 0.95], [0.99, 0.98, 0.97]]:
        probability_engine.synthesize_insight(sig_set)
    belief = probability_engine.consciousness_belief()
    print(f"    After 3 updates:     {belief}")
    print(f"    Resonance trend:     {probability_engine.insight.resonance_trend:.6f}")

    thought_res = probability_engine.thought_resonance(["consciousness", "quantum", "god_code"])
    print(f"    Thought resonance:   {thought_res:.6f}")

    print(f"\n{probability_engine.quick_summary()}")
    print(f"Total computations: {probability_engine._computations}")
    print("=" * 72)
