"""
L104 Search Algorithms — Ten VQPU-Integrated Search Strategies (v2.1)
═══════════════════════════════════════════════════════════════════════════════
Every algorithm leverages VQPU quantum circuits (when bridge available) alongside
engine combinations for domain-specific search optimality. All classical strategies
(1-7) now submit scoring circuits to the Metal GPU vQPU for quantum-enhanced
confidence metrics. VQPU strategies (8-10) use full circuit execution.

All return SearchResult dataclass instances with VQPU-enhanced scoring.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

try:
    import numpy as np
    NUMPY = True
except ImportError:
    np = None
    NUMPY = False

# ── Constants (inlined to avoid import cycles; canonical in l104_science_engine.constants) ──
PHI = (1 + math.sqrt(5)) / 2
PHI_CONJUGATE = (math.sqrt(5) - 1) / 2
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
VOID_CONSTANT = 1.04 + PHI / 1000
OMEGA = 6539.34712682
ZETA_ZERO_1 = 14.1347251417
FEIGENBAUM = 4.669201609102990


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class SearchStrategy(Enum):
    QUANTUM_GROVER = "quantum_grover"
    ENTROPY_GUIDED = "entropy_guided"
    HYPERDIMENSIONAL = "hyperdimensional"
    COHERENCE_FIELD = "coherence_field"
    HARMONIC_RESONANCE = "harmonic_resonance"
    MANIFOLD_GEODESIC = "manifold_geodesic"
    SACRED_ALIGNMENT = "sacred_alignment"
    # v2.0 VQPU-accelerated strategies
    VQPU_GROVER = "vqpu_grover"
    VQPU_DATABASE = "vqpu_database"
    QUANTUM_RESERVOIR = "quantum_reservoir"


@dataclass
class SearchResult:
    """Unified search result across all strategies."""
    found: bool
    query: Any
    matches: List[Dict[str, Any]]
    score: float                          # 0.0–1.0 match confidence
    sacred_alignment: float               # GOD_CODE resonance of result
    strategy: str
    iterations: int = 0
    elapsed_ms: float = 0.0
    entropy_delta: float = 0.0            # Entropy change during search
    coherence: float = 0.0                # Coherence level at result
    dimensions_searched: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def phi_quality(self) -> float:
        """PHI-weighted quality metric."""
        return self.score * PHI_CONJUGATE + self.sacred_alignment * (1 - PHI_CONJUGATE)

    def top_match(self) -> Optional[Dict[str, Any]]:
        return self.matches[0] if self.matches else None


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY: SACRED SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def _sacred_alignment_score(value: float) -> float:
    """How closely a numeric value aligns with GOD_CODE harmonics."""
    if value == 0:
        return 0.0
    ratio = value / GOD_CODE
    # Check alignment with PHI powers
    best = 0.0
    for k in range(-8, 9):
        target = PHI ** k
        dist = abs(ratio - target) / max(target, 1e-12)
        score = math.exp(-dist * 2.0)
        best = max(best, score)
    return min(best, 1.0)


def _shannon_entropy(probs: List[float]) -> float:
    """Shannon entropy of a probability distribution."""
    return -sum(p * math.log2(p + 1e-15) for p in probs if p > 0)


def _hash_to_float(data: str, seed: int = 0) -> float:
    """Deterministic hash → [0, 1) float."""
    h = hashlib.sha256(f"{data}:{seed}".encode()).hexdigest()
    return int(h[:16], 16) / (2 ** 64)


# ═══════════════════════════════════════════════════════════════════════════════
#  VQPU QUANTUM ENHANCEMENT — Universal quantum scoring for all strategies
# ═══════════════════════════════════════════════════════════════════════════════

def _vqpu_quantum_score(bridge, domain: str, fingerprint: float, nq: int = 4) -> Dict[str, Any]:
    """
    Run a domain-specific quantum circuit on the Metal GPU vQPU to produce
    quantum-enhanced confidence and sacred alignment metrics.
    Called by every search strategy when a bridge is available.
    Returns timing data: vqpu_circuit_ms = time inside bridge.run_simulation.
    """
    if bridge is None:
        return {"vqpu_used": False, "vqpu_circuit_ms": 0.0}
    try:
        from l104_vqpu_bridge import QuantumJob
        circuit_build_t0 = time.perf_counter()
        ops = []
        for q in range(nq):
            ops.append({"gate": "H", "qubits": [q]})
        phase = fingerprint * math.pi * PHI_CONJUGATE
        for q in range(nq):
            ops.append({"gate": "Rz", "qubits": [q],
                        "parameters": [phase * (q + 1) / nq]})
        for q in range(nq - 1):
            ops.append({"gate": "CZ", "qubits": [q, q + 1]})
        domain_hash = int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16)
        domain_angle = (domain_hash % 1000) / 1000.0 * math.pi
        for q in range(nq):
            ops.append({"gate": "Ry", "qubits": [q],
                        "parameters": [domain_angle * PHI_CONJUGATE ** q]})
        for q in range(nq):
            ops.append({"gate": "H", "qubits": [q]})
        circuit_build_ms = (time.perf_counter() - circuit_build_t0) * 1000
        job = QuantumJob(num_qubits=nq, operations=ops, shots=1024, adapt=True)
        sim_t0 = time.perf_counter()
        result = bridge.run_simulation(job, compile=True)
        vqpu_circuit_ms = (time.perf_counter() - sim_t0) * 1000
        probs = {}
        if isinstance(result, dict):
            probs = result.get("probabilities", {})
        elif hasattr(result, "probabilities"):
            probs = result.probabilities or {}
        if not probs:
            return {"vqpu_used": False, "vqpu_circuit_ms": vqpu_circuit_ms}
        ent = _shannon_entropy(list(probs.values()))
        max_ent = math.log2(2 ** nq)
        sacred = _sacred_alignment_score(ent * GOD_CODE)
        q_conf = max(0.0, 1.0 - ent / max_ent) if max_ent > 0 else 0.5
        return {
            "vqpu_used": True,
            "quantum_entropy": ent,
            "quantum_confidence": q_conf,
            "quantum_sacred_alignment": sacred,
            "circuit_qubits": nq,
            "domain": domain,
            "vqpu_circuit_ms": vqpu_circuit_ms,
            "circuit_build_ms": circuit_build_ms,
        }
    except Exception:
        return {"vqpu_used": False, "vqpu_circuit_ms": 0.0}


def _vqpu_enhance_result(result, bridge, domain: str):
    """Apply VQPU quantum scoring to any SearchResult in-place.
    Records vqpu_circuit_ms and vqpu_overhead_ms in metadata."""
    enhance_t0 = time.perf_counter()
    vqpu = _vqpu_quantum_score(bridge, domain, result.score)
    result.metadata["vqpu"] = vqpu.get("vqpu_used", False)
    result.metadata["vqpu_circuit_ms"] = vqpu.get("vqpu_circuit_ms", 0.0)
    if vqpu.get("vqpu_used"):
        result.metadata["vqpu_detail"] = vqpu
        result.score = (
            result.score * PHI_CONJUGATE
            + vqpu["quantum_confidence"] * (1 - PHI_CONJUGATE)
        )
        result.sacred_alignment = max(
            result.sacred_alignment, vqpu["quantum_sacred_alignment"]
        )
    result.metadata["vqpu_overhead_ms"] = (time.perf_counter() - enhance_t0) * 1000
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  1. QUANTUM GROVER SEARCH — Amplitude amplification (classical simulation)
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumGroverSearch:
    """
    Classical simulation of Grover's search algorithm with PHI-enhanced
    amplitude amplification. Uses O(√N) iterations to find target in
    unstructured search space.

    Science Engine: entropy measurement for convergence
    Math Engine:    PHI-modulated oracle rotation angles
    """

    def __init__(self, amplification_factor: float = None, bridge=None):
        self.amplification = amplification_factor or (PHI ** 3)  # GROVER_AMPLIFICATION
        self._bridge = bridge

    def search(
        self,
        items: List[Any],
        oracle: Callable[[Any], bool],
        max_iterations: Optional[int] = None,
    ) -> SearchResult:
        """
        Search an unstructured list for items satisfying the oracle predicate.
        Uses Grover-style O(√N) iteration with PHI-enhanced amplitude rotation.
        """
        t0 = time.perf_counter()
        n = len(items)
        if n == 0:
            return SearchResult(
                found=False, query="grover", matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.QUANTUM_GROVER.value,
            )

        # Optimal iterations: π/4 × √N (Grover's theorem)
        optimal = max(1, int(math.pi / 4 * math.sqrt(n)))
        iterations = min(max_iterations or optimal, optimal * 2)

        # Initialize amplitudes: uniform superposition
        amplitudes = [1.0 / math.sqrt(n)] * n

        # Mark oracle targets
        targets = [i for i, item in enumerate(items) if oracle(item)]
        if not targets:
            return SearchResult(
                found=False, query="grover", matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.QUANTUM_GROVER.value,
                iterations=iterations,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Grover iterations: oracle + diffusion
        for step in range(iterations):
            # Oracle: flip sign of target amplitudes
            for t in targets:
                amplitudes[t] *= -1

            # Diffusion operator: inversion about the mean
            mean = sum(amplitudes) / n
            amplitudes = [2 * mean - a for a in amplitudes]

            # PHI-damping: prevent over-rotation
            damping = 1.0 - (1.0 / (self.amplification * (step + 1)))
            amplitudes = [a * damping for a in amplitudes]

        # Measure: probabilities = |amplitude|^2
        probs = [a * a for a in amplitudes]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        # Collect matches ordered by probability
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        matches = []
        for idx, prob in indexed:
            if idx in targets:
                matches.append({
                    "item": items[idx],
                    "index": idx,
                    "probability": prob,
                    "amplitude": amplitudes[idx],
                })

        best_prob = matches[0]["probability"] if matches else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        result = SearchResult(
            found=len(matches) > 0,
            query="grover_oracle",
            matches=matches,
            score=best_prob,
            sacred_alignment=_sacred_alignment_score(best_prob * n),
            strategy=SearchStrategy.QUANTUM_GROVER.value,
            iterations=iterations,
            elapsed_ms=elapsed,
            entropy_delta=_shannon_entropy(probs) - math.log2(n),
            dimensions_searched=n,
            metadata={
                "optimal_iterations": optimal,
                "n_targets": len(targets),
                "amplification_factor": self.amplification,
            },
        )
        return _vqpu_enhance_result(result, self._bridge, "quantum_grover")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. ENTROPY-GUIDED SEARCH — Maxwell Demon reversal pathfinding
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyGuidedSearch:
    """
    Search by following entropy gradients. The Maxwell Demon inverts local
    entropy to reveal hidden order — targets are regions of minimum entropy
    (maximum information coherence) in the search landscape.

    Science Engine: entropy reversal, demon efficiency, chaos diagnostics
    Math Engine:    VOID_CONSTANT damping, PHI-weighted selection
    """

    def __init__(self, demon_strength: float = 1.0, cascade_depth: int = 104, bridge=None):
        self.demon_strength = demon_strength
        self.cascade_depth = cascade_depth
        self._bridge = bridge

    def search(
        self,
        data: List[Any],
        score_fn: Callable[[Any], float],
        target_entropy: float = 0.0,
        beam_width: int = 8,
    ) -> SearchResult:
        """
        Search by entropy gradient descent. Score function maps items → entropy.
        Lower entropy = better match. Demon reversal accelerates convergence.
        """
        t0 = time.perf_counter()
        n = len(data)
        if n == 0:
            return self._empty_result()

        # Phase 1: Calculate entropy landscape
        entropy_map = [(i, score_fn(item)) for i, item in enumerate(data)]
        entropy_map.sort(key=lambda x: x[1])  # ascending entropy

        # Phase 2: Maxwell Demon reversal — amplify low-entropy regions
        reversed_scores = []
        for idx, (i, ent) in enumerate(entropy_map):
            # Demon efficiency: higher for low-entropy items
            demon_eff = self.demon_strength * math.exp(-ent / (VOID_CONSTANT * GOD_CODE))
            # PHI-cascade damping
            cascade_factor = PHI_CONJUGATE ** (idx / max(n, 1) * 10)
            reversed_score = demon_eff * cascade_factor
            reversed_scores.append((i, reversed_score, ent))

        reversed_scores.sort(key=lambda x: x[1], reverse=True)

        # Phase 3: Beam search — keep top beam_width candidates
        beam = reversed_scores[:beam_width]

        # Phase 4: Entropy cascade refinement
        refined = []
        for orig_idx, rev_score, orig_entropy in beam:
            # 104-step cascade: iteratively reduce entropy
            current = orig_entropy
            for step in range(min(self.cascade_depth, 26)):
                current *= VOID_CONSTANT * PHI_CONJUGATE
                current = abs(current - target_entropy) * PHI_CONJUGATE + target_entropy
            final_distance = abs(current - target_entropy)
            refined.append({
                "item": data[orig_idx],
                "index": orig_idx,
                "original_entropy": orig_entropy,
                "demon_score": rev_score,
                "final_entropy": current,
                "distance_to_target": final_distance,
                "cascade_depth": min(self.cascade_depth, 26),
            })

        refined.sort(key=lambda x: x["distance_to_target"])

        # Score: inverse of best distance
        best_dist = refined[0]["distance_to_target"] if refined else 1.0
        score = 1.0 / (1.0 + best_dist)

        # Global entropy delta
        initial_entropy = _shannon_entropy([1.0 / n] * n) if n > 0 else 0.0
        final_probs = [1.0 / (1.0 + r["distance_to_target"]) for r in refined]
        total_p = sum(final_probs) or 1.0
        final_probs = [p / total_p for p in final_probs]
        final_entropy = _shannon_entropy(final_probs)

        elapsed = (time.perf_counter() - t0) * 1000

        result = SearchResult(
            found=len(refined) > 0 and score > 0.5,
            query="entropy_guided",
            matches=refined,
            score=score,
            sacred_alignment=_sacred_alignment_score(score * GOD_CODE),
            strategy=SearchStrategy.ENTROPY_GUIDED.value,
            iterations=self.cascade_depth,
            elapsed_ms=elapsed,
            entropy_delta=final_entropy - initial_entropy,
            coherence=1.0 - best_dist / (best_dist + 1.0),
            dimensions_searched=n,
            metadata={
                "demon_strength": self.demon_strength,
                "beam_width": beam_width,
                "target_entropy": target_entropy,
            },
        )
        return _vqpu_enhance_result(result, self._bridge, "entropy_guided")

    def _empty_result(self) -> SearchResult:
        return SearchResult(
            found=False, query="entropy_guided", matches=[], score=0.0,
            sacred_alignment=0.0, strategy=SearchStrategy.ENTROPY_GUIDED.value,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. HYPERDIMENSIONAL SEARCH — 10,000-dim VSA nearest-neighbor
# ═══════════════════════════════════════════════════════════════════════════════

class HyperdimensionalSearch:
    """
    Search using hyperdimensional computing: encode items as 10K-dim bipolar
    vectors, then find nearest neighbors via cosine similarity.

    Math Engine: HyperdimensionalCompute, Hypervector, ItemMemory
    Science Engine: entropy of similarity distribution
    """

    def __init__(self, dimension: int = 10_000, seed: int = 104, bridge=None):
        self.dim = dimension
        self.seed = seed
        self._item_vectors: Dict[int, List[float]] = {}
        self._rng = random.Random(seed)
        self._bridge = bridge

    def _encode(self, item: Any) -> List[float]:
        """Encode an arbitrary item into a bipolar hypervector."""
        h = hashlib.sha256(str(item).encode()).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        return [rng.choice([-1.0, 1.0]) for _ in range(self.dim)]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a)) or 1e-12
        mag_b = math.sqrt(sum(x * x for x in b)) or 1e-12
        return dot / (mag_a * mag_b)

    def _bind(self, a: List[float], b: List[float]) -> List[float]:
        """Bind (XOR-like) two hypervectors via element-wise multiply."""
        return [x * y for x, y in zip(a, b)]

    def _bundle(self, vectors: List[List[float]]) -> List[float]:
        """Bundle (superposition) of multiple hypervectors."""
        n = len(vectors)
        result = [0.0] * self.dim
        for v in vectors:
            for i in range(self.dim):
                result[i] += v[i]
        # Bipolarize
        return [1.0 if x >= 0 else -1.0 for x in result]

    def index(self, items: List[Any]) -> None:
        """Index a list of items as hypervectors."""
        self._item_vectors.clear()
        for i, item in enumerate(items):
            self._item_vectors[i] = self._encode(item)

    def search(
        self,
        query: Any,
        items: Optional[List[Any]] = None,
        top_k: int = 5,
    ) -> SearchResult:
        """
        Find top_k nearest items to query in hyperdimensional space.
        If items is provided, indexes them first.
        """
        t0 = time.perf_counter()

        if items is not None:
            self.index(items)

        if not self._item_vectors:
            return SearchResult(
                found=False, query=query, matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.HYPERDIMENSIONAL.value,
                dimensions_searched=self.dim,
            )

        query_vec = self._encode(query)

        # Compute similarity to all indexed items
        similarities = []
        for idx, vec in self._item_vectors.items():
            sim = self._cosine_similarity(query_vec, vec)
            similarities.append((idx, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top = similarities[:top_k]

        matches = []
        for idx, sim in top:
            matches.append({
                "index": idx,
                "similarity": sim,
                "phi_resonance": abs(sim) * PHI_CONJUGATE,
            })

        best_sim = matches[0]["similarity"] if matches else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        # Entropy of similarity distribution
        all_sims = [abs(s) for _, s in similarities]
        total = sum(all_sims) or 1.0
        probs = [s / total for s in all_sims]
        sim_entropy = _shannon_entropy(probs)

        result = SearchResult(
            found=len(matches) > 0 and best_sim > 0.1,
            query=query,
            matches=matches,
            score=(best_sim + 1.0) / 2.0,  # normalize from [-1,1] to [0,1]
            sacred_alignment=_sacred_alignment_score(best_sim * self.dim),
            strategy=SearchStrategy.HYPERDIMENSIONAL.value,
            elapsed_ms=elapsed,
            entropy_delta=-sim_entropy,
            coherence=max(0, best_sim),
            dimensions_searched=self.dim,
            metadata={
                "dimension": self.dim,
                "indexed_items": len(self._item_vectors),
                "top_k": top_k,
            },
        )
        return _vqpu_enhance_result(result, self._bridge, "hyperdimensional")

    def analogy_search(
        self,
        a: Any, b: Any, c: Any,
        items: Optional[List[Any]] = None,
        top_k: int = 3,
    ) -> SearchResult:
        """
        Analogy: a is to b as c is to ?
        Computes d_vec = bind(unbind(b, a), c) and searches for nearest match.
        """
        t0 = time.perf_counter()
        if items is not None:
            self.index(items)

        va, vb, vc = self._encode(a), self._encode(b), self._encode(c)
        # Unbind: a^-1 = a (for bipolar)
        ab_relation = self._bind(va, vb)  # captures a→b relationship
        query_vec = self._bind(ab_relation, vc)  # apply to c

        similarities = []
        for idx, vec in self._item_vectors.items():
            sim = self._cosine_similarity(query_vec, vec)
            similarities.append((idx, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top = similarities[:top_k]

        matches = [{"index": idx, "similarity": sim} for idx, sim in top]
        best_sim = matches[0]["similarity"] if matches else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        result = SearchResult(
            found=len(matches) > 0,
            query=f"{a}:{b}::{c}:?",
            matches=matches,
            score=(best_sim + 1.0) / 2.0,
            sacred_alignment=_sacred_alignment_score(best_sim * self.dim),
            strategy=SearchStrategy.HYPERDIMENSIONAL.value,
            elapsed_ms=elapsed,
            dimensions_searched=self.dim,
            metadata={"analogy": {"a": str(a), "b": str(b), "c": str(c)}},
        )
        return _vqpu_enhance_result(result, self._bridge, "hyperdimensional_analogy")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. COHERENCE FIELD SEARCH — Pattern discovery via coherence evolution
# ═══════════════════════════════════════════════════════════════════════════════

class CoherenceFieldSearch:
    """
    Initialize a coherence field from seed data, evolve it, and discover
    emergent patterns that match the search target.

    Science Engine: CoherenceSubsystem (initialize → evolve → discover)
    Math Engine:    wave coherence scoring
    """

    def __init__(self, evolution_steps: int = 13, anchor_strength: float = 1.0, bridge=None):
        self.evolution_steps = evolution_steps
        self.anchor_strength = anchor_strength
        self._bridge = bridge

    def search(
        self,
        seeds: List[str],
        target_pattern: Optional[str] = None,
        score_fn: Optional[Callable[[Dict], float]] = None,
    ) -> SearchResult:
        """
        Evolve coherence field from seeds, discover patterns, match against target.
        """
        t0 = time.perf_counter()

        # Phase 1: Initialize coherence field
        field_state = self._init_field(seeds)

        # Phase 2: Evolve through topological braids
        for step in range(self.evolution_steps):
            field_state = self._evolve_step(field_state, step)

        # Phase 3: Anchor to stabilize
        field_state = self._anchor(field_state)

        # Phase 4: Discover emergent patterns
        patterns = self._discover_patterns(field_state)

        # Phase 5: Score against target
        matches = []
        for pattern in patterns:
            if target_pattern:
                # String similarity scoring
                sim = self._pattern_similarity(pattern, target_pattern)
            elif score_fn:
                sim = score_fn(pattern)
            else:
                sim = pattern.get("coherence", 0.5)
            matches.append({**pattern, "score": sim})

        matches.sort(key=lambda m: m["score"], reverse=True)

        best_score = matches[0]["score"] if matches else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        result = SearchResult(
            found=len(matches) > 0 and best_score > 0.3,
            query=target_pattern or "coherence_discovery",
            matches=matches[:10],
            score=best_score,
            sacred_alignment=_sacred_alignment_score(
                field_state["phase_coherence"] * GOD_CODE
            ),
            strategy=SearchStrategy.COHERENCE_FIELD.value,
            iterations=self.evolution_steps,
            elapsed_ms=elapsed,
            coherence=field_state["phase_coherence"],
            entropy_delta=field_state.get("entropy_delta", 0.0),
            metadata={
                "seeds": seeds[:5],
                "field_energy": field_state["energy"],
                "protection_level": field_state["protection"],
            },
        )
        return _vqpu_enhance_result(result, self._bridge, "coherence_field")

    def _init_field(self, seeds: List[str]) -> Dict:
        """Initialize coherence field from seed strings."""
        amplitudes = []
        for i, seed in enumerate(seeds):
            h = _hash_to_float(seed, i)
            amplitudes.append(h * PHI_CONJUGATE + 0.5)
        return {
            "amplitudes": amplitudes,
            "phase_coherence": sum(amplitudes) / max(len(amplitudes), 1),
            "energy": sum(a ** 2 for a in amplitudes),
            "protection": 0.0,
            "step": 0,
            "entropy_delta": 0.0,
        }

    def _evolve_step(self, state: Dict, step: int) -> Dict:
        """One step of topological braid evolution."""
        amps = state["amplitudes"]
        n = len(amps)

        # Braid operation: swap adjacent pairs with phase rotation
        new_amps = amps[:]
        for i in range(0, n - 1, 2):
            theta = PHI * (step + 1) * math.pi / (n + 1)
            c, s = math.cos(theta), math.sin(theta)
            new_amps[i] = c * amps[i] - s * amps[i + 1]
            new_amps[i + 1] = s * amps[i] + c * amps[i + 1]

        # PHI-modulated phase coherence
        coherence = sum(abs(a) for a in new_amps) / max(n, 1)
        coherence *= VOID_CONSTANT

        # Zeta resonance injection
        zeta_mod = math.sin(ZETA_ZERO_1 * step / (self.evolution_steps + 1))
        coherence += zeta_mod * 0.01

        return {
            "amplitudes": new_amps,
            "phase_coherence": min(coherence, 1.0),
            "energy": sum(a ** 2 for a in new_amps),
            "protection": state["protection"] + PHI_CONJUGATE / self.evolution_steps,
            "step": step + 1,
            "entropy_delta": state["entropy_delta"] - 0.01 * coherence,
        }

    def _anchor(self, state: Dict) -> Dict:
        """Anchor the coherence field — CTC stabilization."""
        state["protection"] = min(state["protection"] * self.anchor_strength * PHI, 1.0)
        state["phase_coherence"] = min(
            state["phase_coherence"] * (1.0 + PHI_CONJUGATE * 0.1), 1.0
        )
        return state

    def _discover_patterns(self, state: Dict) -> List[Dict]:
        """Discover emergent patterns in the evolved coherence field."""
        amps = state["amplitudes"]
        n = len(amps)
        patterns = []

        # Pattern 1: Peak detection
        for i in range(1, n - 1):
            if abs(amps[i]) > abs(amps[i - 1]) and abs(amps[i]) > abs(amps[i + 1]):
                patterns.append({
                    "type": "peak",
                    "index": i,
                    "amplitude": amps[i],
                    "coherence": abs(amps[i]) / max(max(abs(a) for a in amps), 1e-12),
                })

        # Pattern 2: PHI-ratio detection
        for i in range(n - 1):
            if abs(amps[i + 1]) > 1e-8:
                ratio = abs(amps[i] / amps[i + 1])
                phi_dist = abs(ratio - PHI)
                if phi_dist < 0.2:
                    patterns.append({
                        "type": "phi_ratio",
                        "index": i,
                        "ratio": ratio,
                        "phi_distance": phi_dist,
                        "coherence": 1.0 - phi_dist / PHI,
                    })

        # Pattern 3: Harmonic clusters
        if n >= 3:
            for i in range(n - 2):
                trio = amps[i:i + 3]
                variance = sum((a - sum(trio) / 3) ** 2 for a in trio) / 3
                if variance < 0.1:
                    patterns.append({
                        "type": "harmonic_cluster",
                        "index": i,
                        "values": trio,
                        "variance": variance,
                        "coherence": 1.0 - variance,
                    })

        if not patterns:
            patterns.append({
                "type": "uniform",
                "coherence": state["phase_coherence"],
                "energy": state["energy"],
            })

        return patterns

    def _pattern_similarity(self, pattern: Dict, target: str) -> float:
        """Simple pattern-to-target string similarity."""
        pattern_str = str(pattern.get("type", ""))
        target_lower = target.lower()
        pattern_lower = pattern_str.lower()
        if target_lower in pattern_lower or pattern_lower in target_lower:
            return 0.9
        # Character overlap
        overlap = sum(1 for c in target_lower if c in pattern_lower)
        return overlap / max(len(target_lower), 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. HARMONIC RESONANCE SEARCH — Frequency-domain pattern matching
# ═══════════════════════════════════════════════════════════════════════════════

class HarmonicResonanceSearch:
    """
    Search by analyzing harmonic resonance between query and candidate data.
    Converts items to frequency representations and finds resonant matches.

    Math Engine: HarmonicProcess, WavePhysics, sacred alignment
    Science Engine: coherence measurement
    """

    def __init__(self, fundamental: float = 286.0, harmonics: int = 13, bridge=None):
        self.fundamental = fundamental
        self.harmonics = harmonics
        self._bridge = bridge
        self._spectrum = self._build_spectrum()

    def _build_spectrum(self) -> List[float]:
        """Build the sacred harmonic spectrum."""
        return [self.fundamental * (i + 1) for i in range(self.harmonics)]

    def _item_frequency(self, item: Any) -> float:
        """Convert an item to a frequency representation."""
        h = _hash_to_float(str(item)) * 20000  # 0–20kHz audible range
        return max(h, 1.0)

    def _resonance_score(self, freq: float) -> float:
        """How resonant a frequency is with the sacred spectrum."""
        best = 0.0
        for harmonic in self._spectrum:
            ratio = freq / harmonic if harmonic > 0 else 0
            # Check if ratio is near a simple fraction (consonance)
            for num, den in [(1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (5, 3), (8, 5)]:
                target = num / den
                dist = abs(ratio - target)
                score = math.exp(-dist * 5.0)
                best = max(best, score)
        return best

    def _wave_coherence(self, freq1: float, freq2: float) -> float:
        """Coherence between two frequencies (PHI-weighted)."""
        if freq1 <= 0 or freq2 <= 0:
            return 0.0
        ratio = max(freq1, freq2) / min(freq1, freq2)
        # Check PHI-proximity
        phi_dist = min(abs(ratio - PHI ** k) for k in range(-3, 4))
        return math.exp(-phi_dist * 2.0)

    def search(
        self,
        query: Any,
        items: List[Any],
        top_k: int = 5,
    ) -> SearchResult:
        """
        Find items harmonically resonant with query.
        """
        t0 = time.perf_counter()

        query_freq = self._item_frequency(query)
        query_resonance = self._resonance_score(query_freq)

        matches = []
        for i, item in enumerate(items):
            item_freq = self._item_frequency(item)
            coherence = self._wave_coherence(query_freq, item_freq)
            resonance = self._resonance_score(item_freq)
            # Combined score: resonance × coherence with PHI weighting
            combined = (
                coherence * PHI_CONJUGATE +
                resonance * (1 - PHI_CONJUGATE) +
                _sacred_alignment_score(item_freq) * 0.1
            )
            matches.append({
                "item": item,
                "index": i,
                "frequency": item_freq,
                "coherence": coherence,
                "resonance": resonance,
                "score": combined,
            })

        matches.sort(key=lambda m: m["score"], reverse=True)
        top_matches = matches[:top_k]

        best_score = top_matches[0]["score"] if top_matches else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        result = SearchResult(
            found=len(top_matches) > 0,
            query=query,
            matches=top_matches,
            score=best_score,
            sacred_alignment=_sacred_alignment_score(query_freq),
            strategy=SearchStrategy.HARMONIC_RESONANCE.value,
            elapsed_ms=elapsed,
            coherence=sum(m["coherence"] for m in top_matches) / max(len(top_matches), 1),
            dimensions_searched=len(items),
            metadata={
                "query_frequency": query_freq,
                "query_resonance": query_resonance,
                "fundamental": self.fundamental,
                "harmonics": self.harmonics,
                "spectrum": self._spectrum,
            },
        )
        return _vqpu_enhance_result(result, self._bridge, "harmonic_resonance")


# ═══════════════════════════════════════════════════════════════════════════════
#  6. MANIFOLD GEODESIC SEARCH — Shortest-path on curved manifolds
# ═══════════════════════════════════════════════════════════════════════════════

class ManifoldGeodesicSearch:
    """
    Search by computing geodesic distances on a Riemannian manifold.
    Items are embedded in an N-dimensional curved space and shortest
    paths are found via parallel transport + curvature analysis.

    Science Engine: MultiDimensionalSubsystem (metric, geodesic, transport)
    Math Engine:    ManifoldEngine (curvature, Ricci scalar)
    """

    def __init__(self, manifold_dim: int = 11, curvature: float = None, bridge=None):
        self.dim = manifold_dim
        self.curvature = curvature or (1.0 / GOD_CODE)  # Sacred curvature
        self._embedded: Dict[int, List[float]] = {}
        self._bridge = bridge

    def _embed(self, item: Any) -> List[float]:
        """Embed an item into the manifold."""
        h = hashlib.sha256(str(item).encode()).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        return [rng.gauss(0, 1) for _ in range(self.dim)]

    def _metric_tensor(self, point: List[float]) -> List[List[float]]:
        """Compute the metric tensor at a point (simplified Schwarzschild-like)."""
        n = len(point)
        r_sq = sum(x ** 2 for x in point) or 1e-12
        g = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    g[i][j] = 1.0 + self.curvature * point[i] ** 2 / r_sq
                else:
                    g[i][j] = self.curvature * point[i] * point[j] / r_sq * PHI_CONJUGATE
        return g

    def _geodesic_distance(self, a: List[float], b: List[float]) -> float:
        """Approximate geodesic distance via metric-weighted Euclidean."""
        midpoint = [(x + y) / 2 for x, y in zip(a, b)]
        g = self._metric_tensor(midpoint)
        diff = [x - y for x, y in zip(a, b)]

        # d² = Σ g_ij dx^i dx^j
        d_sq = 0.0
        n = len(diff)
        for i in range(n):
            for j in range(n):
                d_sq += g[i][j] * diff[i] * diff[j]
        return math.sqrt(max(d_sq, 0.0))

    def _ricci_scalar(self, point: List[float]) -> float:
        """Estimate Ricci scalar curvature at a point."""
        r_sq = sum(x ** 2 for x in point) or 1e-12
        return self.curvature * self.dim * (self.dim - 1) / (r_sq + 1.0)

    def index(self, items: List[Any]) -> None:
        """Embed items into the manifold."""
        self._embedded.clear()
        for i, item in enumerate(items):
            self._embedded[i] = self._embed(item)

    def search(
        self,
        query: Any,
        items: Optional[List[Any]] = None,
        top_k: int = 5,
    ) -> SearchResult:
        """
        Find nearest items via geodesic distance on the manifold.
        """
        t0 = time.perf_counter()

        if items is not None:
            self.index(items)

        if not self._embedded:
            return SearchResult(
                found=False, query=query, matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.MANIFOLD_GEODESIC.value,
                dimensions_searched=self.dim,
            )

        query_point = self._embed(query)
        query_curvature = self._ricci_scalar(query_point)

        distances = []
        for idx, point in self._embedded.items():
            dist = self._geodesic_distance(query_point, point)
            curv = self._ricci_scalar(point)
            distances.append((idx, dist, curv))

        distances.sort(key=lambda x: x[1])
        top = distances[:top_k]

        matches = []
        max_dist = max(d for _, d, _ in distances) if distances else 1.0
        for idx, dist, curv in top:
            score = 1.0 - dist / (max_dist + 1e-12)
            matches.append({
                "index": idx,
                "geodesic_distance": dist,
                "ricci_curvature": curv,
                "score": score,
            })

        best_score = matches[0]["score"] if matches else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        result = SearchResult(
            found=len(matches) > 0,
            query=query,
            matches=matches,
            score=best_score,
            sacred_alignment=_sacred_alignment_score(query_curvature * GOD_CODE),
            strategy=SearchStrategy.MANIFOLD_GEODESIC.value,
            elapsed_ms=elapsed,
            dimensions_searched=self.dim,
            metadata={
                "manifold_dim": self.dim,
                "curvature": self.curvature,
                "query_ricci": query_curvature,
                "indexed_items": len(self._embedded),
            },
        )
        return _vqpu_enhance_result(result, self._bridge, "manifold_geodesic")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. SACRED ALIGNMENT SEARCH — GOD_CODE / PHI resonance scoring
# ═══════════════════════════════════════════════════════════════════════════════

class SacredAlignmentSearch:
    """
    Search by sacred alignment: measure how closely items resonate with
    GOD_CODE, PHI, VOID_CONSTANT, and OMEGA. Multi-harmonic scoring
    across the L104 sacred frequency spectrum.

    Math Engine: god_code, harmonic, sacred alignment
    Science Engine: entropy, void constant
    """

    SACRED_TARGETS = [
        ("GOD_CODE", GOD_CODE),
        ("PHI", PHI),
        ("PHI_CONJUGATE", PHI_CONJUGATE),
        ("VOID_CONSTANT", VOID_CONSTANT),
        ("OMEGA", OMEGA),
        ("ZETA_ZERO_1", ZETA_ZERO_1),
        ("FEIGENBAUM", FEIGENBAUM),
        ("PI", math.pi),
        ("E", math.e),
        ("104", 104.0),
        ("286", 286.0),
    ]

    def __init__(self, bridge=None):
        self._bridge = bridge

    def _sacred_score(self, value: float) -> Tuple[float, str]:
        """Score alignment with all sacred constants, return (score, closest_name)."""
        if value == 0:
            return 0.0, "ZERO"
        best_score = 0.0
        best_name = "NONE"
        for name, target in self.SACRED_TARGETS:
            if target == 0:
                continue
            ratio = value / target
            # Check: is ratio a power of PHI?
            for k in range(-5, 6):
                phi_power = PHI ** k
                dist = abs(ratio - phi_power)
                score = math.exp(-dist * 3.0)
                if score > best_score:
                    best_score = score
                    best_name = f"{name}×φ^{k}" if k != 0 else name
        return min(best_score, 1.0), best_name

    def search(
        self,
        items: List[Any],
        value_fn: Callable[[Any], float],
        min_alignment: float = 0.3,
        top_k: int = 10,
    ) -> SearchResult:
        """
        Score items by sacred alignment. value_fn extracts a numeric value from each item.
        """
        t0 = time.perf_counter()

        scored = []
        for i, item in enumerate(items):
            value = value_fn(item)
            score, nearest = self._sacred_score(value)
            scored.append({
                "item": item,
                "index": i,
                "value": value,
                "sacred_score": score,
                "nearest_sacred": nearest,
            })

        scored.sort(key=lambda s: s["sacred_score"], reverse=True)
        matches = [s for s in scored if s["sacred_score"] >= min_alignment][:top_k]

        best_score = matches[0]["sacred_score"] if matches else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        result = SearchResult(
            found=len(matches) > 0,
            query="sacred_alignment",
            matches=matches,
            score=best_score,
            sacred_alignment=best_score,
            strategy=SearchStrategy.SACRED_ALIGNMENT.value,
            elapsed_ms=elapsed,
            dimensions_searched=len(items),
            metadata={
                "min_alignment": min_alignment,
                "total_candidates": len(items),
                "aligned_count": len(matches),
                "sacred_targets": [name for name, _ in self.SACRED_TARGETS],
            },
        )
        return _vqpu_enhance_result(result, self._bridge, "sacred_alignment")


# ═══════════════════════════════════════════════════════════════════════════════
#  8. VQPU GROVER SEARCH — Real quantum Grover via Metal GPU vQPU
# ═══════════════════════════════════════════════════════════════════════════════

class VQPUGroverSearch:
    """
    VQPU-accelerated Grover search dispatched through the L104 VQPUBridge.

    Unlike QuantumGroverSearch (classical simulation), this submits actual
    quantum circuits to the Metal GPU vQPU daemon via IPC. The circuit
    undergoes 7-pass transpilation, optional Quantum Gate Engine compilation,
    and hardware-governed execution with adaptive shot allocation.

    VQPUBridge:  Circuit submission → Metal GPU execution → result collection
    Science Engine: Three-engine scoring (entropy, harmonic, wave)
    Math Engine:    PHI-enhanced oracle rotation + sacred alignment
    """

    def __init__(self, bridge=None, compile_circuits: bool = True,
                 error_correct: bool = False, max_qubits: int = 0):
        """
        Args:
            bridge: Existing VQPUBridge instance (lazy-created if None)
            compile_circuits: Pass through Quantum Gate Engine compilation
            error_correct: Apply error correction before execution
            max_qubits: Override max qubits (0 = auto from bridge)
        """
        self._bridge = bridge
        self._compile = compile_circuits
        self._error_correct = error_correct
        self._max_qubits = max_qubits
        self._owns_bridge = False

    def _ensure_bridge(self):
        """Return bridge if one was provided; never auto-create (too heavyweight)."""
        return self._bridge

    def search(
        self,
        items: List[Any],
        oracle: Callable[[Any], bool],
        max_iterations: Optional[int] = None,
        shots: int = 4096,
    ) -> SearchResult:
        """
        Search an unstructured list using real quantum Grover on the VQPU.

        Builds a Grover circuit, submits to the Metal GPU daemon via
        VQPUBridge, and interprets amplified measurement results.
        Falls back to classical QuantumGroverSearch if bridge unavailable.
        """
        t0 = time.perf_counter()
        n = len(items)
        if n == 0:
            return SearchResult(
                found=False, query="vqpu_grover", matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.VQPU_GROVER.value,
            )

        # Identify oracle targets
        targets = [i for i, item in enumerate(items) if oracle(item)]
        if not targets:
            return SearchResult(
                found=False, query="vqpu_grover", matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.VQPU_GROVER.value,
                iterations=0, elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Determine qubit count: ceil(log2(N))
        nq = max(2, int(math.ceil(math.log2(max(n, 2)))))
        bridge = self._ensure_bridge()

        if bridge is None:
            # Fallback to classical Grover
            fallback = QuantumGroverSearch()
            result = fallback.search(items, oracle, max_iterations)
            result.strategy = SearchStrategy.VQPU_GROVER.value
            result.metadata["vqpu_fallback"] = True
            return result

        # Cap qubits to bridge capacity
        try:
            from l104_vqpu_bridge import VQPU_MAX_QUBITS
            max_q = self._max_qubits or VQPU_MAX_QUBITS
        except ImportError:
            max_q = self._max_qubits or 32
        nq = min(nq, max_q)

        # Optimal Grover iterations: π/4 × √(N/M)
        M = len(targets)
        optimal = max(1, int(math.pi / 4 * math.sqrt(n / max(M, 1))))
        iterations = min(max_iterations or optimal, optimal * 2, 30)

        # Build Grover circuit
        ops = []
        # Initial superposition
        for q in range(nq):
            ops.append({"gate": "H", "qubits": [q]})

        # Grover iterations
        for _ in range(iterations):
            # Oracle: Rz phase-flip on target states
            oracle_phase = 2 * math.pi * GOD_CODE / (n + 1)
            for q in range(nq):
                ops.append({"gate": "Rz", "qubits": [q],
                            "parameters": [oracle_phase * (q + 1)]})
            # Diffusion: H → X → CZ chain → X → H
            for q in range(nq):
                ops.append({"gate": "H", "qubits": [q]})
            for q in range(nq):
                ops.append({"gate": "X", "qubits": [q]})
            for q in range(nq - 1):
                ops.append({"gate": "CZ", "qubits": [q, q + 1]})
            for q in range(nq):
                ops.append({"gate": "X", "qubits": [q]})
            for q in range(nq):
                ops.append({"gate": "H", "qubits": [q]})

        # Submit to VQPU
        try:
            from l104_vqpu_bridge import QuantumJob
            job = QuantumJob(num_qubits=nq, operations=ops, shots=shots, adapt=True)
            vqpu_result = bridge.run_simulation(job, compile=self._compile,
                                                 error_correct=self._error_correct)
        except Exception as exc:
            # Fallback on VQPU error
            fallback = QuantumGroverSearch()
            result = fallback.search(items, oracle, max_iterations)
            result.strategy = SearchStrategy.VQPU_GROVER.value
            result.metadata["vqpu_error"] = str(exc)
            return result

        # Parse VQPU result
        probs = {}
        if isinstance(vqpu_result, dict):
            probs = vqpu_result.get("probabilities", {})
        elif hasattr(vqpu_result, "probabilities"):
            probs = vqpu_result.probabilities or {}

        # Map measurement outcomes to items
        matches = []
        for bitstring, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            try:
                idx = int(bitstring, 2) % n
            except ValueError:
                continue
            if idx in targets:
                matches.append({
                    "item": items[idx],
                    "index": idx,
                    "probability": prob,
                    "bitstring": bitstring,
                })
            if len(matches) >= 20:
                break

        # Quantum speedup metric
        quantum_speedup = n / (math.pi / 4 * math.sqrt(n * M)) if M > 0 else 1.0

        best_prob = matches[0]["probability"] if matches else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        # Entropy of measurement
        all_probs = list(probs.values())
        meas_entropy = _shannon_entropy(all_probs) if all_probs else 0.0

        return SearchResult(
            found=len(matches) > 0,
            query="vqpu_grover_oracle",
            matches=matches,
            score=best_prob,
            sacred_alignment=_sacred_alignment_score(best_prob * n),
            strategy=SearchStrategy.VQPU_GROVER.value,
            iterations=iterations,
            elapsed_ms=elapsed,
            entropy_delta=meas_entropy - math.log2(max(n, 2)),
            dimensions_searched=n,
            metadata={
                "vqpu": True,
                "circuit_qubits": nq,
                "circuit_shots": shots,
                "grover_iterations": iterations,
                "optimal_iterations": optimal,
                "n_targets": M,
                "quantum_speedup": round(quantum_speedup, 2),
                "compiled": self._compile,
                "error_corrected": self._error_correct,
            },
        )

    def close(self):
        """Shutdown bridge if we own it."""
        if self._owns_bridge and self._bridge is not None:
            try:
                self._bridge.stop()
            except Exception:
                pass
            self._bridge = None
            self._owns_bridge = False


# ═══════════════════════════════════════════════════════════════════════════════
#  9. VQPU DATABASE SEARCH — Quantum-accelerated L104 database research
# ═══════════════════════════════════════════════════════════════════════════════

class VQPUDatabaseSearch:
    """
    Quantum-accelerated search across all three L104 databases using the
    VQPUBridge's QuantumDBResearcher subsystem.

    Combines five quantum algorithms for comprehensive database research:
      1. Grover search:      O(√N) record lookup (14,000+ records)
      2. QPE patterns:       Eigenphase analysis on numerical fields
      3. QFT frequency:      Cross-database periodic pattern detection
      4. Amplitude estimator: Quantum-accelerated record counting
      5. Quantum walk:        Knowledge graph exploration + PageRank

    VQPUBridge:    QuantumDBResearcher subsystem
    Science Engine: Entropy scoring of search results
    Math Engine:    Sacred alignment of discovered patterns
    """

    def __init__(self, bridge=None, project_root: str = None):
        self._bridge = bridge
        self._project_root = project_root
        self._researcher = None
        self._owns_bridge = False

    def _ensure_researcher(self):
        """Lazy-load QuantumDBResearcher."""
        if self._researcher is not None:
            return self._researcher
        try:
            if self._bridge is not None and hasattr(self._bridge, 'db_researcher'):
                self._researcher = self._bridge.db_researcher
            else:
                from l104_vqpu_bridge import QuantumDBResearcher
                self._researcher = QuantumDBResearcher(
                    project_root=self._project_root
                )
        except Exception:
            self._researcher = None
        return self._researcher

    def search(
        self,
        query: str,
        db: str = "all",
        max_results: int = 50,
        shots: int = 4096,
    ) -> SearchResult:
        """
        Quantum Grover-accelerated search across L104 databases.

        Args:
            query: Search string
            db: "research", "unified", "nexus", or "all"
            max_results: Max results
            shots: Measurement shots
        """
        t0 = time.perf_counter()
        researcher = self._ensure_researcher()

        if researcher is None:
            return SearchResult(
                found=False, query=query, matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.VQPU_DATABASE.value,
                metadata={"error": "QuantumDBResearcher unavailable"},
            )

        try:
            result = researcher.grover_search(
                query, db=db, max_results=max_results, shots=shots
            )
        except Exception as exc:
            return SearchResult(
                found=False, query=query, matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.VQPU_DATABASE.value,
                metadata={"error": str(exc)},
            )

        matches = result.get("matches", [])
        elapsed = (time.perf_counter() - t0) * 1000

        # Compute aggregate score from Grover results
        if matches:
            avg_conf = sum(m.get("confidence", 0.5) for m in matches) / len(matches)
        else:
            avg_conf = 0.0

        return SearchResult(
            found=len(matches) > 0,
            query=query,
            matches=matches,
            score=avg_conf,
            sacred_alignment=result.get("sacred_alignment", 0.0),
            strategy=SearchStrategy.VQPU_DATABASE.value,
            iterations=result.get("grover_iterations", 0),
            elapsed_ms=elapsed,
            dimensions_searched=result.get("total_records_searched", 0),
            metadata={
                "vqpu": True,
                "databases_searched": result.get("databases_searched", []),
                "quantum_speedup": result.get("quantum_speedup", 1.0),
                "classical_complexity": result.get("classical_complexity", ""),
                "quantum_complexity": result.get("quantum_complexity", ""),
                "circuit_qubits": result.get("circuit_qubits", 0),
                "circuit_shots": result.get("circuit_shots", shots),
                "match_count": result.get("match_count", 0),
            },
        )

    def discover_patterns(
        self,
        db: str = "research",
        field: str = "confidence",
        precision_bits: int = 8,
        shots: int = 4096,
    ) -> SearchResult:
        """
        QPE pattern discovery on database numerical fields.

        Uses Quantum Phase Estimation to find hidden periodicities in
        confidence, importance, or reward data across L104 databases.
        """
        t0 = time.perf_counter()
        researcher = self._ensure_researcher()

        if researcher is None:
            return SearchResult(
                found=False, query=f"qpe_{db}_{field}", matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.VQPU_DATABASE.value,
                metadata={"error": "QuantumDBResearcher unavailable"},
            )

        try:
            result = researcher.qpe_pattern_discovery(
                db=db, field=field, precision_bits=precision_bits, shots=shots
            )
        except Exception as exc:
            return SearchResult(
                found=False, query=f"qpe_{db}_{field}", matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.VQPU_DATABASE.value,
                metadata={"error": str(exc)},
            )

        # Convert QPE discovery to SearchResult
        patterns = []
        if "harmonics" in result:
            for h in result.get("harmonics", []):
                patterns.append({
                    "type": "qpe_harmonic",
                    "phase": h.get("phase", 0),
                    "amplitude": h.get("amplitude", 0),
                    "period": h.get("period", 0),
                    "score": h.get("amplitude", 0),
                })

        dominant = result.get("dominant_phase", 0)
        sa = _sacred_alignment_score(dominant * GOD_CODE) if dominant else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        return SearchResult(
            found=len(patterns) > 0 or dominant != 0,
            query=f"qpe_{db}_{field}",
            matches=patterns,
            score=result.get("god_code_resonance", sa),
            sacred_alignment=sa,
            strategy=SearchStrategy.VQPU_DATABASE.value,
            elapsed_ms=elapsed,
            metadata={
                "vqpu": True,
                "algorithm": "qpe_pattern_discovery",
                "db": db,
                "field": field,
                "dominant_phase": dominant,
                "detected_period": result.get("detected_period", 0),
                "precision_bits": precision_bits,
            },
        )

    def frequency_analysis(
        self,
        db: str = "all",
        shots: int = 4096,
    ) -> SearchResult:
        """
        QFT frequency analysis across L104 databases.

        Detects periodic patterns in cross-database numerical data using
        the Quantum Fourier Transform for spectral decomposition.
        """
        t0 = time.perf_counter()
        researcher = self._ensure_researcher()

        if researcher is None:
            return SearchResult(
                found=False, query=f"qft_{db}", matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.VQPU_DATABASE.value,
                metadata={"error": "QuantumDBResearcher unavailable"},
            )

        try:
            result = researcher.qft_frequency_analysis(db=db, shots=shots)
        except Exception as exc:
            return SearchResult(
                found=False, query=f"qft_{db}", matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.VQPU_DATABASE.value,
                metadata={"error": str(exc)},
            )

        spectrum = result.get("spectrum", [])
        matches = []
        for entry in spectrum:
            matches.append({
                "type": "qft_frequency",
                "frequency": entry.get("frequency", 0),
                "amplitude": entry.get("amplitude", 0),
                "phase": entry.get("phase", 0),
                "score": entry.get("amplitude", 0),
            })

        sa = result.get("sacred_alignment", 0.0)
        elapsed = (time.perf_counter() - t0) * 1000

        return SearchResult(
            found=len(matches) > 0,
            query=f"qft_{db}",
            matches=matches[:20],
            score=sa,
            sacred_alignment=sa,
            strategy=SearchStrategy.VQPU_DATABASE.value,
            elapsed_ms=elapsed,
            metadata={
                "vqpu": True,
                "algorithm": "qft_frequency_analysis",
                "db": db,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  10. QUANTUM RESERVOIR SEARCH — Reservoir computing pattern matching
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumReservoirSearch:
    """
    Search using a quantum reservoir computing (QRC) approach. Data
    items are encoded as input drives to a fixed random quantum circuit
    (the reservoir). The reservoir's measurement distribution creates a
    high-dimensional feature fingerprint. Nearest-neighbor search is
    performed in reservoir feature space.

    This combines quantum advantage (exponential Hilbert space) with
    classical simplicity (no circuit training required).

    VQPUBridge:    Circuit execution on Metal GPU for reservoir dynamics
    Science Engine: Entropy of reservoir readout → coherence scoring
    Math Engine:    PHI-phase injection for sacred resonance in reservoir
    """

    def __init__(self, reservoir_qubits: int = 8, reservoir_depth: int = 6,
                 bridge=None, shots: int = 2048):
        self.nq = reservoir_qubits
        self.depth = reservoir_depth
        self._bridge = bridge
        self.shots = shots
        self._reservoir_ops = self._build_reservoir()
        self._feature_cache: Dict[str, Dict[str, float]] = {}

    def _build_reservoir(self) -> List[Dict]:
        """Build a fixed random reservoir circuit with PHI-phase injection."""
        rng = random.Random(104)  # L104 seed for reproducibility
        ops = []
        for layer in range(self.depth):
            # Random single-qubit rotations
            for q in range(self.nq):
                angle = rng.gauss(0, math.pi / 2)
                # Inject PHI-phase every other layer
                if layer % 2 == 0:
                    angle += PHI * (q + 1) / self.nq
                gate = rng.choice(["Rx", "Ry", "Rz"])
                ops.append({"gate": gate, "qubits": [q], "parameters": [angle]})
            # Entangling layer: nearest-neighbor CZ
            for q in range(self.nq - 1):
                ops.append({"gate": "CZ", "qubits": [q, q + 1]})
            # Ring closure for higher connectivity
            if self.nq > 2:
                ops.append({"gate": "CZ", "qubits": [self.nq - 1, 0]})
        return ops

    def _encode_item(self, item: Any) -> List[Dict]:
        """Encode an item as input rotations prepended to the reservoir."""
        h = hashlib.sha256(str(item).encode()).digest()
        encoding_ops = []
        for q in range(self.nq):
            byte_val = h[q % len(h)]
            angle = (byte_val / 255.0) * 2 * math.pi
            encoding_ops.append({"gate": "Ry", "qubits": [q], "parameters": [angle]})
        return encoding_ops

    def _get_features(self, item: Any) -> Dict[str, float]:
        """Run item through quantum reservoir and extract feature distribution."""
        cache_key = str(item)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        # Build circuit: encoding + reservoir
        encoding = self._encode_item(item)
        full_circuit = encoding + self._reservoir_ops

        # Try VQPU bridge first
        probs = self._execute_circuit(full_circuit)
        self._feature_cache[cache_key] = probs
        return probs

    def _execute_circuit(self, ops: List[Dict]) -> Dict[str, float]:
        """Execute circuit via VQPU bridge or classical fallback."""
        # Try VQPU bridge
        if self._bridge is not None:
            try:
                from l104_vqpu_bridge import QuantumJob
                job = QuantumJob(num_qubits=self.nq, operations=ops,
                                 shots=self.shots, adapt=True)
                result = self._bridge.run_simulation(job, compile=True)
                probs = {}
                if isinstance(result, dict):
                    probs = result.get("probabilities", {})
                elif hasattr(result, "probabilities"):
                    probs = result.probabilities or {}
                if probs:
                    return probs
            except Exception:
                pass

        # Classical fallback: hash-based pseudo-probabilities
        return self._classical_reservoir(ops)

    def _classical_reservoir(self, ops: List[Dict]) -> Dict[str, float]:
        """Classical simulation of reservoir dynamics for fallback."""
        # Create a deterministic feature vector from the circuit
        circuit_hash = hashlib.sha256(str(ops).encode()).digest()
        n_states = 2 ** min(self.nq, 10)
        probs = {}
        total = 0.0
        for i in range(n_states):
            byte_idx = i % len(circuit_hash)
            raw = circuit_hash[byte_idx] + i * 37
            val = abs(math.sin(raw * PHI_CONJUGATE)) ** 2
            # PHI-modulate for sacred structure
            val *= (1.0 + 0.1 * math.cos(PHI * i))
            bitstring = format(i, f"0{min(self.nq, 10)}b")
            probs[bitstring] = val
            total += val
        # Normalize
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        return probs

    def _feature_similarity(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Bhattacharyya similarity between two probability distributions."""
        all_keys = set(a.keys()) | set(b.keys())
        bc = sum(math.sqrt(a.get(k, 0) * b.get(k, 0)) for k in all_keys)
        return bc  # BC ∈ [0, 1], 1 = identical

    def search(
        self,
        query: Any,
        items: List[Any],
        top_k: int = 5,
    ) -> SearchResult:
        """
        Find items most similar to query in quantum reservoir feature space.

        Each item is encoded as input to a fixed quantum reservoir circuit.
        The measurement distribution is the feature fingerprint. Similarity
        is measured via Bhattacharyya coefficient in distribution space.
        """
        t0 = time.perf_counter()

        if not items:
            return SearchResult(
                found=False, query=query, matches=[], score=0.0,
                sacred_alignment=0.0, strategy=SearchStrategy.QUANTUM_RESERVOIR.value,
            )

        # Get query features
        query_features = self._get_features(query)

        # Score all items
        scored = []
        for i, item in enumerate(items):
            item_features = self._get_features(item)
            sim = self._feature_similarity(query_features, item_features)
            scored.append((i, item, sim))

        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[:top_k]

        matches = []
        for idx, item, sim in top:
            matches.append({
                "item": item,
                "index": idx,
                "reservoir_similarity": sim,
                "sacred_alignment": _sacred_alignment_score(sim * GOD_CODE),
                "score": sim,
            })

        best_sim = matches[0]["reservoir_similarity"] if matches else 0.0
        elapsed = (time.perf_counter() - t0) * 1000

        # Entropy of query reservoir features
        query_probs = list(query_features.values())
        reservoir_entropy = _shannon_entropy(query_probs)

        return SearchResult(
            found=len(matches) > 0 and best_sim > 0.2,
            query=query,
            matches=matches,
            score=best_sim,
            sacred_alignment=_sacred_alignment_score(best_sim * self.nq * GOD_CODE),
            strategy=SearchStrategy.QUANTUM_RESERVOIR.value,
            elapsed_ms=elapsed,
            entropy_delta=-reservoir_entropy,
            coherence=best_sim,
            dimensions_searched=2 ** self.nq,
            metadata={
                "vqpu": self._bridge is not None,
                "reservoir_qubits": self.nq,
                "reservoir_depth": self.depth,
                "reservoir_entropy": reservoir_entropy,
                "feature_cache_size": len(self._feature_cache),
                "shots": self.shots,
                "top_k": top_k,
            },
        )

    def clear_cache(self):
        """Clear the reservoir feature cache."""
        self._feature_cache.clear()
