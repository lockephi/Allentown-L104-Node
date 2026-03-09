"""
L104 ASI KB Reconstruction Engine v1.0.0
═══════════════════════════════════════════════════════════════════════════════
Quantum Probability Knowledge Base Data Reconstruction

Reconstructs missing or degraded knowledge nodes using quantum probability
amplitudes propagated through the KB relation graph. Three-phase pipeline:

  Phase 1 — VECTORIZATION: TF-IDF semantic embedding + GOD_CODE quantum states
  Phase 2 — PROPAGATION: BFS amplitude propagation with entanglement blend,
            Born-rule collapse, Grover amplification, GOD_CODE phase alignment
  Phase 3 — COLLECTION: Fact reconstruction from high-amplitude neighbors

Uses l104_quantum_engine.QuantumMathCore for Grover amplification and
quantum state operations. Falls back to analytic formulas when unavailable.

Integration:
  - ASI scoring dimension: kb_reconstruction_fidelity (weight 0.02)
  - Data source: l104_asi.knowledge_data (191 nodes, 1609 facts, 141 edges)
  - Quantum math: l104_quantum_engine.QuantumMathCore (Grover, Bell, density)
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import (
    PHI, GOD_CODE, TAU, VOID_CONSTANT, PHI_CONJUGATE,
)

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

KB_RECONSTRUCTION_VERSION = "1.0.0"

PROPAGATION_DEPTH = 3                           # Max hops through relation graph
AMPLITUDE_DECAY_PER_HOP = TAU                   # ~0.618 — PHI-conjugate decay per hop
MIN_RECONSTRUCTION_CONFIDENCE = 0.15            # Below this, node is "unrecoverable"
GOD_CODE_PHASE_SCALE = math.pi / GOD_CODE       # Phase alignment scale factor
GROVER_BOOST_THRESHOLD = 5                      # Min neighbor count for Grover amplification
ENTANGLEMENT_STRENGTH = PHI / (1 + PHI)         # TAU ~ 0.618
SOFT_EDGE_THRESHOLD = 0.15                      # Min cosine similarity for soft edge
VOID_DAMPING = VOID_CONSTANT                    # 1.0416... damping per hop
EMBEDDING_DIM = 256                             # TF-IDF + quantum state dimensionality
SCAN_TTL = 60.0                                 # Seconds before stale scan

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ReconstructionResult:
    """Result of a single node reconstruction attempt."""
    node_key: str
    original_confidence: float
    reconstructed_confidence: float
    reconstructed_facts: List[str]
    source_nodes: List[str]
    born_probability: float
    grover_amplified: bool
    god_code_alignment: float
    propagation_depth_used: int
    timestamp: str = ""


@dataclass
class KBHealthReport:
    """Overall health of the knowledge base after reconstruction scan."""
    total_nodes: int = 0
    healthy_nodes: int = 0
    degraded_nodes: int = 0
    missing_nodes: int = 0
    reconstructed_count: int = 0
    avg_reconstruction_confidence: float = 0.0
    graph_connectivity: float = 0.0
    god_code_resonance: float = 0.0
    fidelity_score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  KB VECTORIZER — Phase 1: Semantic + Quantum Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class KBVectorizer:
    """Vectorization layer that bridges TF-IDF semantic embeddings with
    GOD_CODE-phased quantum state vectors.

    Reuses the SemanticEncoder pattern from language_comprehension.py:524
    (TF-IDF + cosine similarity) and the GOD_CODE phase-seeding pattern
    from quantum_embedding.py:100 (QuantumTokenEmbedding).
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self._semantic_vectors: Dict[str, Any] = {}     # key -> real ndarray (D,)
        self._quantum_states: Dict[str, Any] = {}       # key -> complex ndarray (D,)
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        self._fitted = False

    def vectorize_kb(self, kb_nodes: Dict[str, Dict]) -> None:
        """TF-IDF vectorize all KB nodes, then wrap in quantum state vectors.

        Each node's text = definition + ' '.join(facts).
        Semantic vector: TF-IDF real-valued (D,).
        Quantum state: |node_k> = magnitude(tfidf) * exp(i * GOD_CODE phase).
        """
        if not _NP_AVAILABLE or not kb_nodes:
            return

        keys = list(kb_nodes.keys())
        texts = []
        for key in keys:
            node = kb_nodes[key]
            text = node.get("definition", "") + " " + " ".join(node.get("facts", []))
            texts.append(text)

        # ── TF-IDF vectorization (inline, no external dep) ──
        semantic_matrix = self._tfidf_vectorize(texts)

        for i, key in enumerate(keys):
            self._semantic_vectors[key] = semantic_matrix[i]

            # ── Quantum state: magnitude from TF-IDF, phase from GOD_CODE ──
            key_hash = sum(ord(c) for c in key)
            phase = key_hash * GOD_CODE_PHASE_SCALE

            # Add GOD_CODE harmonic overtone (matches quantum_embedding.py:126)
            phase += (GOD_CODE / 286.0) * math.sin(key_hash * math.pi / 104.0)

            magnitude = semantic_matrix[i]
            quantum_state = magnitude * np.exp(1j * phase)
            # Normalize to unit vector in Hilbert space
            norm = np.linalg.norm(quantum_state)
            if norm > 0:
                quantum_state = quantum_state / norm
            self._quantum_states[key] = quantum_state

        self._fitted = True

    def _tfidf_vectorize(self, texts: List[str]) -> Any:
        """Minimal TF-IDF vectorizer (matches SemanticEncoder pattern).
        Returns (N, D) numpy array of TF-IDF vectors."""
        # Build vocabulary from corpus
        vocab: Dict[str, int] = {}
        doc_freq: Dict[str, int] = defaultdict(int)
        tokenized = []

        for text in texts:
            tokens = text.lower().split()
            unique_tokens = set(tokens)
            tokenized.append(tokens)
            for tok in unique_tokens:
                doc_freq[tok] += 1
                if tok not in vocab and len(vocab) < self.embedding_dim:
                    vocab[tok] = len(vocab)

        n_docs = len(texts)
        dim = min(len(vocab), self.embedding_dim)
        matrix = np.zeros((n_docs, self.embedding_dim), dtype=np.float64)

        for i, tokens in enumerate(tokenized):
            tf: Dict[str, int] = defaultdict(int)
            for tok in tokens:
                tf[tok] += 1
            for tok, count in tf.items():
                if tok in vocab and vocab[tok] < dim:
                    idf = math.log(1.0 + n_docs / (1.0 + doc_freq.get(tok, 0)))
                    matrix[i, vocab[tok]] = (count / max(len(tokens), 1)) * idf

            # L2 normalize
            row_norm = np.linalg.norm(matrix[i])
            if row_norm > 0:
                matrix[i] /= row_norm

        return matrix

    def semantic_similarity(self, key_a: str, key_b: str) -> float:
        """Cosine similarity between two node TF-IDF vectors."""
        cache_key = (key_a, key_b) if key_a < key_b else (key_b, key_a)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        vec_a = self._semantic_vectors.get(key_a)
        vec_b = self._semantic_vectors.get(key_b)
        if vec_a is None or vec_b is None:
            return 0.0

        dot = float(np.dot(vec_a, vec_b))
        self._similarity_cache[cache_key] = dot
        return dot

    def quantum_overlap(self, key_a: str, key_b: str) -> complex:
        """<node_a|node_b> inner product in quantum Hilbert space."""
        state_a = self._quantum_states.get(key_a)
        state_b = self._quantum_states.get(key_b)
        if state_a is None or state_b is None:
            return 0j
        return complex(np.vdot(state_a, state_b))

    def discover_soft_edges(self, kb_keys: List[str],
                            threshold: float = SOFT_EDGE_THRESHOLD
                            ) -> List[Tuple[str, str, float]]:
        """Find semantically similar KB node pairs above threshold.
        Uses vectorized matrix multiplication for O(N²) efficiency.
        Returns (key_a, key_b, similarity) triples."""
        if not self._fitted or not _NP_AVAILABLE:
            return []

        # Build matrix of all semantic vectors in key order
        vecs = []
        valid_keys = []
        for key in kb_keys:
            vec = self._semantic_vectors.get(key)
            if vec is not None:
                vecs.append(vec)
                valid_keys.append(key)

        if len(vecs) < 2:
            return []

        # Vectorized pairwise cosine similarity via matrix multiply
        mat = np.array(vecs)              # (N, D)
        sim_matrix = mat @ mat.T          # (N, N) — cosine sim (already L2-normalized)

        edges = []
        n = len(valid_keys)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    edges.append((valid_keys[i], valid_keys[j], sim))
                    # Cache for later lookups
                    cache_key = (valid_keys[i], valid_keys[j]) if valid_keys[i] < valid_keys[j] else (valid_keys[j], valid_keys[i])
                    self._similarity_cache[cache_key] = sim
        return edges

    def get_quantum_amplitude(self, key: str) -> complex:
        """Get the GOD_CODE-phase quantum amplitude for a node.
        Magnitude = L2 norm of TF-IDF, phase = GOD_CODE hash alignment."""
        state = self._quantum_states.get(key)
        if state is None:
            return 0j
        # Scalar amplitude = mean of quantum state components
        return complex(np.mean(state))


# ═══════════════════════════════════════════════════════════════════════════════
#  KB RECONSTRUCTION ENGINE — Phases 2-3: Propagation + Collection
# ═══════════════════════════════════════════════════════════════════════════════

class KBReconstructionEngine:
    """Quantum Probability KB Data Reconstruction Engine.

    Reconstructs missing/degraded knowledge nodes using quantum probability
    amplitudes propagated through the KB cross-subject relation graph,
    augmented by semantic soft edges from TF-IDF vectorization.

    Algorithm:
      1. Vectorize: TF-IDF + GOD_CODE quantum states for all 191 nodes
      2. Build combined graph: hard edges (CROSS_SUBJECT_RELATIONS) + soft edges (cosine sim)
      3. Assign quantum amplitudes from vectorized states
      4. For each node: propagate amplitudes via BFS, apply Born rule,
         Grover-amplify if neighborhood large, weight by GOD_CODE alignment
      5. Collect reconstructed facts from high-amplitude neighbors

    Uses QuantumMathCore from l104_quantum_engine for Grover operator and
    quantum state manipulation when available.
    """

    VERSION = KB_RECONSTRUCTION_VERSION

    def __init__(self):
        self._vectorizer: Optional[KBVectorizer] = None
        self._kb_nodes: Dict[str, Dict[str, Any]] = {}
        # Weighted relation graph: key -> {neighbor_key: weight}
        self._relation_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._node_amplitudes: Dict[str, complex] = {}
        self._initialized = False
        self._fidelity_score = 0.0
        self._reconstruction_cache: Dict[str, ReconstructionResult] = {}
        self._total_reconstructions = 0
        self._last_scan_time: float = 0.0
        # Quantum math core (lazy)
        self._qmc = None

    # ── Lazy Loaders ──

    def _get_quantum_math_core(self):
        """Lazy-load QuantumMathCore from l104_quantum_engine."""
        if self._qmc is None:
            try:
                from l104_quantum_engine import QuantumMathCore
                self._qmc = QuantumMathCore
            except Exception:
                pass
        return self._qmc

    # ── Initialization ──

    def initialize(self) -> None:
        """Load KB data, vectorize, build combined graph, compute amplitudes."""
        if self._initialized:
            return

        from .knowledge_data import KNOWLEDGE_NODES, CROSS_SUBJECT_RELATIONS

        # Index nodes by composite key (matches MMLUKnowledgeBase pattern)
        for node_data in KNOWLEDGE_NODES:
            key = f"{node_data['subject']}/{node_data['concept']}".lower().replace(" ", "_")
            self._kb_nodes[key] = node_data

        # Phase 1: Vectorize all nodes
        self._vectorizer = KBVectorizer(EMBEDDING_DIM)
        self._vectorizer.vectorize_kb(self._kb_nodes)

        # Phase 2a: Build combined relation graph
        self._build_combined_graph(CROSS_SUBJECT_RELATIONS)

        # Phase 2b: Compute quantum amplitudes from vectorized states
        self._compute_node_amplitudes()

        self._initialized = True

    def _build_combined_graph(self, cross_subject_relations: List[Tuple[str, str]]) -> None:
        """Merge hard edges (CROSS_SUBJECT_RELATIONS) with soft edges from
        semantic similarity. Hard edges get weight 1.0, soft edges get their
        cosine similarity as weight."""

        # Hard edges from CROSS_SUBJECT_RELATIONS (weight 1.0)
        for key_a, key_b in cross_subject_relations:
            if key_a in self._kb_nodes and key_b in self._kb_nodes:
                self._relation_graph[key_a][key_b] = 1.0
                self._relation_graph[key_b][key_a] = 1.0

        # Intra-subject auto-links (weight 0.8)
        subject_groups: Dict[str, List[str]] = defaultdict(list)
        for key, node in self._kb_nodes.items():
            subject_groups[node["subject"]].append(key)
        for subject, keys in subject_groups.items():
            for i in range(len(keys)):
                for j in range(i + 1, min(i + 3, len(keys))):
                    if keys[j] not in self._relation_graph[keys[i]]:
                        self._relation_graph[keys[i]][keys[j]] = 0.8
                        self._relation_graph[keys[j]][keys[i]] = 0.8

        # Soft edges from semantic similarity (weight = cosine similarity)
        if self._vectorizer and self._vectorizer._fitted:
            kb_keys = list(self._kb_nodes.keys())
            soft_edges = self._vectorizer.discover_soft_edges(kb_keys, SOFT_EDGE_THRESHOLD)
            for key_a, key_b, sim in soft_edges:
                # Only add if not already connected (hard edges take priority)
                if key_b not in self._relation_graph[key_a]:
                    self._relation_graph[key_a][key_b] = sim
                    self._relation_graph[key_b][key_a] = sim

    def _compute_node_amplitudes(self) -> None:
        """Assign complex quantum amplitude to each KB node.

        Amplitude = vectorizer quantum amplitude + fact-density correction.
        Uses QuantumMathCore Bell state initialization pattern when available.
        """
        max_facts = max(
            (len(n.get("facts", [])) for n in self._kb_nodes.values()), default=1
        )

        qmc = self._get_quantum_math_core()

        for key, node in self._kb_nodes.items():
            n_facts = len(node.get("facts", []))
            fact_density = math.sqrt(max(0.01, n_facts / max(max_facts, 1)))

            # Get vectorizer quantum amplitude (GOD_CODE-phased)
            if self._vectorizer:
                vec_amp = self._vectorizer.get_quantum_amplitude(key)
            else:
                vec_amp = 0j

            # Phase from key hash via GOD_CODE
            key_hash = sum(ord(c) for c in key)
            phase = key_hash * GOD_CODE_PHASE_SCALE

            # Combine: fact density as magnitude correction, vectorizer as base
            base_magnitude = abs(vec_amp) if abs(vec_amp) > 0 else 0.5
            amplitude = (base_magnitude * fact_density) * (
                math.cos(phase) + 1j * math.sin(phase)
            )

            # Use QuantumMathCore density matrix trace as quality check
            if qmc and _NP_AVAILABLE and abs(amplitude) > 0:
                try:
                    state = [amplitude, complex(math.sqrt(max(0, 1.0 - abs(amplitude)**2)))]
                    rho = qmc.density_matrix(state)
                    trace = sum(rho[i][i].real for i in range(len(rho)))
                    if trace > 0:
                        amplitude *= math.sqrt(trace)
                except Exception:
                    pass

            self._node_amplitudes[key] = amplitude

    # ── Propagation ──

    def _god_code_phase_coupling(self, key_a: str, key_b: str) -> float:
        """Compute GOD_CODE phase coupling between two KB nodes.
        Nodes in the same category resonate more strongly (PHI-conjugate boost).
        Uses quantum overlap from vectorizer when available."""
        cat_a = self._kb_nodes.get(key_a, {}).get("category", "")
        cat_b = self._kb_nodes.get(key_b, {}).get("category", "")

        base_phase = (
            sum(ord(c) for c in key_a) + sum(ord(c) for c in key_b)
        ) * GOD_CODE_PHASE_SCALE

        # Same-category bonus: stronger phase coherence
        if cat_a == cat_b and cat_a:
            base_phase *= PHI_CONJUGATE

        # Quantum overlap modulation (from vectorizer Hilbert space)
        if self._vectorizer:
            overlap = self._vectorizer.quantum_overlap(key_a, key_b)
            # Use overlap phase as correction
            base_phase += math.atan2(overlap.imag, overlap.real) * TAU

        return base_phase

    def _propagate_amplitude(self, target_key: str,
                             depth: int = PROPAGATION_DEPTH) -> complex:
        """BFS amplitude propagation from neighbors to target node.

        For each neighbor at hop distance h:
          contribution = A_neighbor * TAU^h * edge_weight * e^(i * phase_coupling) / VOID_DAMPING^h

        Uses QuantumMathCore.grover_operator on the accumulated state when the
        neighborhood is large enough (>= GROVER_BOOST_THRESHOLD)."""
        if target_key not in self._kb_nodes:
            return 0j

        accumulated = 0j
        visited: Set[str] = {target_key}
        frontier: List[Tuple[str, int]] = [(target_key, 0)]

        while frontier:
            current_key, hop = frontier.pop(0)
            if hop >= depth:
                continue

            neighbors = self._relation_graph.get(current_key, {})
            for neighbor_key, edge_weight in neighbors.items():
                if neighbor_key in visited:
                    continue
                visited.add(neighbor_key)

                # Amplitude decays by TAU per hop, damped by VOID_CONSTANT
                decay = (AMPLITUDE_DECAY_PER_HOP ** (hop + 1)) / (VOID_DAMPING ** hop)
                neighbor_amp = self._node_amplitudes.get(neighbor_key, 0j)

                # GOD_CODE phase coupling rotation
                coupling = self._god_code_phase_coupling(target_key, neighbor_key)
                phase_factor = math.cos(coupling) + 1j * math.sin(coupling)

                contribution = neighbor_amp * decay * edge_weight * phase_factor
                accumulated += contribution

                frontier.append((neighbor_key, hop + 1))

        return accumulated

    def _grover_amplify(self, target_prob: float, n_neighbors: int,
                        use_qmc: bool = False) -> float:
        """Apply Grover amplification when neighborhood is large enough.

        Uses analytic Grover rotation formula by default for speed.
        When use_qmc=True, uses QuantumMathCore.grover_operator from
        l104_quantum_engine (involves Qiskit circuit construction)."""
        if n_neighbors < GROVER_BOOST_THRESHOLD:
            return target_prob

        if use_qmc:
            qmc = self._get_quantum_math_core()
            if qmc and _NP_AVAILABLE:
                try:
                    N = max(n_neighbors, 2)
                    state = [complex(1.0 / math.sqrt(N))] * N
                    oracle_indices = [0]
                    state[0] = complex(math.sqrt(max(0.001, target_prob)))
                    norm = math.sqrt(sum(abs(a)**2 for a in state))
                    if norm > 0:
                        state = [a / norm for a in state]
                    k = max(1, int(math.pi / 4 * math.sqrt(N)))
                    amplified = qmc.grover_operator(state, oracle_indices, iterations=k)
                    return min(1.0, abs(amplified[0])**2)
                except Exception:
                    pass

        # Analytic Grover rotation formula (fast, no Qiskit dependency)
        theta = math.asin(math.sqrt(max(0.001, min(1.0, target_prob))))
        k = max(1, int(math.pi / (4 * theta)))
        return min(1.0, math.sin((2 * k + 1) * theta) ** 2)

    def _compute_god_code_alignment(self, node_key: str) -> float:
        """GOD_CODE phase alignment: P = cos^2(hash * pi / GOD_CODE).
        This is the resonance probability — nodes whose hash aligns with
        GOD_CODE harmonics get higher reconstruction confidence."""
        key_hash = sum(ord(c) for c in node_key)
        return math.cos(key_hash * math.pi / GOD_CODE) ** 2

    # ── Fact Collection ──

    def _collect_neighbor_facts(self, node_key: str,
                                top_k: int = 5) -> Tuple[List[str], List[str]]:
        """Collect facts from the top-k highest-amplitude neighbors.
        Number of facts taken per neighbor is proportional to their Born probability."""
        neighbors = self._relation_graph.get(node_key, {})
        if not neighbors:
            return [], []

        # Rank neighbors by Born probability
        scored: List[Tuple[str, float]] = []
        for nk in neighbors:
            amp = self._node_amplitudes.get(nk, 0j)
            scored.append((nk, abs(amp) ** 2))
        scored.sort(key=lambda x: x[1], reverse=True)

        facts: List[str] = []
        sources: List[str] = []
        target_category = self._kb_nodes.get(node_key, {}).get("category", "")

        for nk, score in scored[:top_k]:
            node = self._kb_nodes.get(nk, {})
            node_facts = node.get("facts", [])
            if not node_facts:
                continue

            # Category-aware selection: same category = more facts
            n_take = max(1, int(score * 3))
            if node.get("category", "") == target_category and target_category:
                n_take = min(len(node_facts), n_take + 1)

            facts.extend(node_facts[:n_take])
            sources.append(nk)

        return facts, sources

    # ── Single Node Reconstruction ──

    def reconstruct_node(self, node_key: str) -> ReconstructionResult:
        """Full reconstruction pipeline for a single KB node.

        1. Get current node amplitude (may be low if degraded)
        2. Propagate amplitudes from neighbors through relation graph
        3. Entanglement-weighted blend of local + propagated
        4. Born rule: P = |A_combined|^2
        5. Grover amplification if neighborhood >= threshold
        6. GOD_CODE phase alignment weighting
        7. Collect reconstructed facts from high-amplitude neighbors
        """
        self.initialize()

        # Check cache
        if node_key in self._reconstruction_cache:
            if time.time() - self._last_scan_time < SCAN_TTL:
                return self._reconstruction_cache[node_key]

        node = self._kb_nodes.get(node_key)
        if node is None:
            return ReconstructionResult(
                node_key=node_key, original_confidence=0.0,
                reconstructed_confidence=0.0, reconstructed_facts=[],
                source_nodes=[], born_probability=0.0,
                grover_amplified=False, god_code_alignment=0.0,
                propagation_depth_used=0,
            )

        # Original confidence from fact density (avg ~8.4 facts per node)
        n_facts = len(node.get("facts", []))
        original_confidence = min(1.0, n_facts / 10.0)

        # Step 1-2: Local + propagated amplitudes
        local_amp = self._node_amplitudes.get(node_key, 0j)
        propagated_amp = self._propagate_amplitude(node_key)

        # Step 3: Entanglement-weighted blend (matches QuantumProbability pattern)
        combined_amp = (
            (1 - ENTANGLEMENT_STRENGTH) * local_amp +
            ENTANGLEMENT_STRENGTH * propagated_amp
        )

        # Step 4: Born rule
        born_prob = min(1.0, abs(combined_amp) ** 2)

        # Step 5: Grover amplification
        n_neighbors = len(self._relation_graph.get(node_key, {}))
        grover_applied = n_neighbors >= GROVER_BOOST_THRESHOLD
        if grover_applied:
            born_prob = self._grover_amplify(born_prob, n_neighbors)

        # Step 6: GOD_CODE phase alignment
        god_code_align = self._compute_god_code_alignment(node_key)

        # Final reconstruction confidence: Born probability weighted by alignment
        reconstructed_confidence = born_prob * (0.7 + 0.3 * god_code_align)
        reconstructed_confidence = min(1.0, max(0.0, reconstructed_confidence))

        # Step 7: Collect facts from neighbors
        reconstructed_facts, source_nodes = self._collect_neighbor_facts(node_key)

        result = ReconstructionResult(
            node_key=node_key,
            original_confidence=original_confidence,
            reconstructed_confidence=reconstructed_confidence,
            reconstructed_facts=reconstructed_facts,
            source_nodes=source_nodes,
            born_probability=born_prob,
            grover_amplified=grover_applied,
            god_code_alignment=god_code_align,
            propagation_depth_used=min(PROPAGATION_DEPTH, max(1, n_neighbors)),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        self._reconstruction_cache[node_key] = result
        self._total_reconstructions += 1
        return result

    # ── Full KB Scan ──

    def full_scan(self) -> KBHealthReport:
        """Scan entire KB, reconstruct all nodes, return aggregate health report."""
        self.initialize()
        self._last_scan_time = time.time()

        report = KBHealthReport(total_nodes=len(self._kb_nodes))
        confidences: List[float] = []
        god_code_scores: List[float] = []
        connected_count = 0

        for key in self._kb_nodes:
            result = self.reconstruct_node(key)
            confidences.append(result.reconstructed_confidence)
            god_code_scores.append(result.god_code_alignment)

            if len(self._relation_graph.get(key, {})) > 0:
                connected_count += 1

            if result.reconstructed_confidence >= 0.7:
                report.healthy_nodes += 1
            elif result.reconstructed_confidence >= MIN_RECONSTRUCTION_CONFIDENCE:
                report.degraded_nodes += 1
                report.reconstructed_count += 1
            else:
                report.missing_nodes += 1

        report.avg_reconstruction_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        report.graph_connectivity = connected_count / max(report.total_nodes, 1)
        report.god_code_resonance = (
            sum(god_code_scores) / len(god_code_scores) if god_code_scores else 0.0
        )

        # Fidelity score: weighted composite for ASI scoring dimension
        report.fidelity_score = (
            0.35 * report.avg_reconstruction_confidence +
            0.25 * report.graph_connectivity +
            0.20 * report.god_code_resonance +
            0.20 * (report.healthy_nodes / max(report.total_nodes, 1))
        )
        self._fidelity_score = report.fidelity_score
        return report

    # ── ASI Scoring Interface ──

    def fidelity_score(self) -> float:
        """Return KB reconstruction fidelity score [0, 1] for ASI scoring.
        Runs full_scan if stale (>60s since last scan)."""
        if time.time() - self._last_scan_time > SCAN_TTL:
            self.full_scan()
        return self._fidelity_score

    # ── Live Data Ingestion (Intellect → AGI → ASI chain) ──

    def ingest_intellect_data(self, local_intellect) -> int:
        """Ingest live training data from LocalIntellect into the KB graph.

        Converts LocalIntellect's BM25-indexed training entries into KB nodes
        that participate in quantum probability reconstruction. This ensures
        the reconstruction engine uses the full live knowledge corpus, not
        just the static KNOWLEDGE_NODES.

        Called by ASI Core to complete the data flow:
          LocalIntellect (BM25 corpus) → AGI (enrichment) → ASI (reconstruction)

        Returns number of nodes ingested."""
        self.initialize()
        ingested = 0

        try:
            # Use whatever training_data is already loaded from __init__ (fast JSONL).
            # Do NOT call _ensure_training_extended / _ensure_training_index here —
            # those trigger heavy SQLite / MMLU / BM25 loads that block for minutes.
            training_data = getattr(local_intellect, 'training_data', [])
            if not training_data:
                return 0

            # Convert training entries to KB nodes
            # Group by category/source for coherent node creation
            category_groups: Dict[str, List[Dict]] = defaultdict(list)
            for entry in training_data:
                cat = entry.get('category', entry.get('source', 'intellect_general'))
                category_groups[cat].append(entry)

            for category, entries in category_groups.items():
                # Create a KB node per category group
                key = f"intellect/{category}".lower().replace(" ", "_")
                if key in self._kb_nodes:
                    # Merge new facts into existing node
                    existing = self._kb_nodes[key]
                    for entry in entries[:50]:  # Cap per-category intake  # (was 10)
                        completion = entry.get('completion', '')
                        if completion and completion not in existing.get('facts', []):
                            existing.setdefault('facts', []).append(completion[:1000])  # (was 200)
                    continue

                # New node from category
                facts = []
                definition = ""
                for entry in entries[:50]:  # Cap to keep graph manageable  # (was 15)
                    prompt = entry.get('prompt', '')
                    completion = entry.get('completion', '')
                    if not definition and prompt:
                        definition = prompt[:1000]  # (was 200)
                    if completion:
                        facts.append(completion[:1000])  # (was 200)

                if not facts:
                    continue

                node = {
                    "concept": category,
                    "subject": "intellect_corpus",
                    "category": "intellect",
                    "definition": definition,
                    "facts": facts,
                }
                self._kb_nodes[key] = node
                ingested += 1

            # If new nodes were added, revectorize and rebuild graph
            if ingested > 0:
                self._revectorize_and_rebuild()

        except Exception:
            pass

        self._intellect_nodes_ingested = ingested
        return ingested

    def ingest_agi_data(self, agi_core) -> int:
        """Ingest AGI Core enrichment data into the KB graph.

        Pulls AGI scoring dimensions, engine wiring info, and cognitive
        mesh state as supplementary KB nodes. This closes the middle link
        in the Intellect → AGI → ASI data flow.

        Returns number of nodes ingested."""
        self.initialize()
        ingested = 0

        try:
            # AGI enrichment data (uses get_kb_enrichment_data if available,
            # falls back to full_engine_status + compute_10d_agi_score)
            agi_data = None
            if hasattr(agi_core, 'get_kb_enrichment_data'):
                try:
                    agi_data = agi_core.get_kb_enrichment_data()
                except Exception:
                    pass
            if agi_data is None and hasattr(agi_core, 'full_engine_status'):
                try:
                    agi_data = agi_core.full_engine_status()
                except Exception:
                    pass

            if agi_data:
                # Scoring dimensions node
                key = "agi/scoring_dimensions"
                if key not in self._kb_nodes:
                    facts = []
                    dims = agi_data.get('dimensions', {})
                    for dim_name, dim_val in dims.items():
                        facts.append(f"AGI dimension {dim_name}: {dim_val:.4f}" if isinstance(dim_val, float) else f"AGI dimension {dim_name}: {dim_val}")
                    composite = agi_data.get('composite_score', 0.0)
                    if composite:
                        facts.insert(0, f"AGI composite score: {composite:.4f}")
                    if facts:
                        self._kb_nodes[key] = {
                            "concept": "scoring_dimensions",
                            "subject": "agi_core",
                            "category": "agi",
                            "definition": f"AGI Core v{agi_data.get('agi_version', '?')} scoring with {len(dims)} dimensions",
                            "facts": facts[:100],  # (was 20)
                        }
                        ingested += 1

                # Engine wiring node
                engines = agi_data.get('engines', {})
                if engines:
                    key = "agi/engine_wiring"
                    if key not in self._kb_nodes:
                        facts = [
                            f"Engine {name}: {'connected' if connected else 'disconnected'}"
                            for name, connected in engines.items()
                        ]
                        self._kb_nodes[key] = {
                            "concept": "engine_wiring",
                            "subject": "agi_core",
                            "category": "agi",
                            "definition": "AGI engine connection status across all subsystems",
                            "facts": facts,
                        }
                        ingested += 1

            # AGI pipeline health as knowledge
            if hasattr(agi_core, '_pipeline_health'):
                key = "agi/pipeline_health"
                if key not in self._kb_nodes:
                    health = agi_core._pipeline_health
                    facts = [
                        f"Pipeline {name}: {'healthy' if ok else 'degraded'}"
                        for name, ok in health.items()
                    ]
                    if facts:
                        self._kb_nodes[key] = {
                            "concept": "pipeline_health",
                            "subject": "agi_core",
                            "category": "agi",
                            "definition": "AGI pipeline subsystem health status",
                            "facts": facts[:100],  # (was 15)
                        }
                        ingested += 1

            if ingested > 0:
                self._revectorize_and_rebuild()

        except Exception:
            pass

        self._agi_nodes_ingested = ingested
        return ingested

    def _revectorize_and_rebuild(self) -> None:
        """Re-run vectorization and graph building after new nodes are ingested.
        Invalidates caches to force fresh reconstruction."""
        from .knowledge_data import CROSS_SUBJECT_RELATIONS

        self._vectorizer = KBVectorizer(EMBEDDING_DIM)
        self._vectorizer.vectorize_kb(self._kb_nodes)
        self._relation_graph = defaultdict(dict)
        self._build_combined_graph(CROSS_SUBJECT_RELATIONS)
        self._compute_node_amplitudes()
        # Invalidate caches
        self._reconstruction_cache.clear()
        self._last_scan_time = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Status dict for pipeline health tracking."""
        hard_edges = sum(
            1 for neighbors in self._relation_graph.values()
            for w in neighbors.values() if w >= 1.0
        ) // 2
        soft_edges = sum(
            1 for neighbors in self._relation_graph.values()
            for w in neighbors.values() if w < 1.0
        ) // 2
        return {
            "version": self.VERSION,
            "initialized": self._initialized,
            "total_nodes": len(self._kb_nodes),
            "hard_edges": hard_edges,
            "soft_edges": soft_edges,
            "total_reconstructions": self._total_reconstructions,
            "fidelity_score": self._fidelity_score,
            "cache_size": len(self._reconstruction_cache),
            "quantum_math_core_available": self._qmc is not None,
            "intellect_nodes_ingested": getattr(self, '_intellect_nodes_ingested', 0),
            "agi_nodes_ingested": getattr(self, '_agi_nodes_ingested', 0),
        }
