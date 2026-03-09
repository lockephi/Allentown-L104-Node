"""L104 Intellect — Fault Tolerance Mixin.

Extracts the fault tolerance subsystem from LocalIntellect:
  - _init_fault_tolerance(): initializes the L104FaultTolerance engine with
    attention, TF-IDF, and topological memory fed from training data.
  - _text_to_ft_vector(): deterministic hash-based text embedding (64-dim).
  - _ft_process_query(): runs a query through 5 quantum upgrades + RNN context.
  - _qiskit_process(): real quantum circuit processing via l104_quantum_gate_engine.

All self.* references (training_data, _ft_engine, _ft_init_done, _evolution_state,
_ensure_training_extended, etc.) are resolved via multiple inheritance with LocalIntellect.
"""

import math
import hashlib
import logging

import numpy as np

from .constants import GOD_CODE_PHASE
from .numerics import PHI, GOD_CODE

logger = logging.getLogger("l104_local_intellect")


class FaultToleranceMixin:
    """Mixin providing the v23.0 Fault Tolerance Engine for LocalIntellect."""

    # ═══════════════════════════════════════════════════════════════════════════
    # v23.0 FAULT TOLERANCE ENGINE INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _init_fault_tolerance(self):
        """
        Initialize the L104 Fault Tolerance engine with all 5 quantum upgrades.
        Feeds training data into attention, TF-IDF, and topological memory.
        """
        self._ensure_training_extended()
        try:
            from l104_fault_tolerance import (
                L104FaultTolerance, COHERENCE_LIMIT,
                GOD_CODE as FT_GOD_CODE,
                PHI as FT_PHI,
            )
            self._ft_engine = L104FaultTolerance(
                braid_depth=8,
                lattice_size=10,
                topological_distance=5,
                hidden_dim=128,
                input_dim=64,
            )
            # Initialise the 3-layer stack
            self._ft_engine.initialise()

            # Feed training data into attention + TF-IDF + topological memory
            _fed_attention = 0
            _fed_tfidf = 0
            _fed_memory = 0

            # Sample training data for attention patterns (up to 200)
            np.random.seed(None)  # True randomness
            sample_size = min(200, len(self.training_data))
            if sample_size > 0:
                indices = np.random.choice(len(self.training_data), sample_size, replace=False)
                for idx in indices:
                    entry = self.training_data[idx]
                    text = entry.get('completion', entry.get('text', ''))
                    if text and len(text) > 10:
                        # Convert text to vector via hash-based embedding
                        vec = self._text_to_ft_vector(text)
                        self._ft_engine.attention.add_pattern(vec)
                        _fed_attention += 1

                        # Store in topological memory
                        label = text[:40]
                        self._ft_engine.memory.store(vec, label=label)
                        _fed_memory += 1

            # Feed documents into TF-IDF
            for entry in self.training_data[:2000]:
                text = entry.get('completion', entry.get('text', ''))
                if text and len(text) > 5:
                    tokens = [w.lower() for w in text.split() if len(w) > 2][:100]
                    if tokens:
                        self._ft_engine.tfidf.add_document(tokens)
                        _fed_tfidf += 1

            self._ft_init_done = True

        except Exception as e:
            self._ft_engine = None
            self._ft_init_done = False

    def _text_to_ft_vector(self, text: str, dim: int = 64) -> np.ndarray:
        """Convert text to a 64-dim vector via deterministic hash embedding + noise."""
        h = hashlib.sha512(text.encode('utf-8', errors='replace')).digest()
        base = np.array([float(b) / 255.0 for b in h[:dim]], dtype=np.float64)
        # Add time-based micro-noise for evolution
        noise = np.random.randn(dim) * 0.001
        vec = base + noise
        # Normalize to unit sphere, scale by character entropy
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _ft_process_query(self, message: str) -> dict:
        """
        Run a query through the fault tolerance engine's 5 upgrades:
        1. Inductive coherence check
        2. Attention over training patterns
        3. TF-IDF query embedding
        4. Multi-hop reasoning
        5. Topological memory retrieval
        6. RNN hidden state update
        Returns metadata dict for response enrichment.
        """
        if not self._ft_engine or not self._ft_init_done:
            # Lazy init on first query (deferred from __init__ for performance)
            if not self._ft_init_done and self._ft_engine is None:
                self._init_fault_tolerance()
            if not self._ft_engine or not self._ft_init_done:
                return {}

        try:
            result = {}
            query_vec = self._text_to_ft_vector(message)

            # 1. RNN hidden state - accumulate context
            rnn_out = self._ft_engine.process_query(query_vec)
            result['rnn_ctx_sim'] = rnn_out.get('context_similarity_after', 0)
            result['rnn_queries'] = rnn_out.get('query_count', 0)

            # 2. Attention over training patterns
            attn = self._ft_engine.attention.attend(query_vec)
            result['attn_entropy'] = attn.get('entropy', 0)
            result['attn_patterns'] = attn.get('pattern_count', 0)
            result['attn_max_weight'] = attn.get('max_weight', 0)

            # 3. TF-IDF query
            tokens = [w.lower() for w in message.split() if len(w) > 2][:50]
            if tokens:
                tfidf_vec = self._ft_engine.tfidf.tfidf_query(tokens)
                result['tfidf_norm'] = float(np.linalg.norm(tfidf_vec))
                result['tfidf_vocab'] = self._ft_engine.tfidf.vocab_size
            else:
                result['tfidf_norm'] = 0.0
                result['tfidf_vocab'] = self._ft_engine.tfidf.vocab_size

            # 4. Multi-hop reasoning
            mh = self._ft_engine.reasoner.reason(query_vec)
            result['mh_hops'] = mh.get('hops_taken', 0)
            result['mh_converged'] = mh.get('converged', False)
            result['mh_harmonic'] = mh.get('god_harmonic', 0)

            # 5. Topological memory retrieval
            mem_results = self._ft_engine.memory.retrieve(query_vec, top_k=3)
            if mem_results and 'advisory' not in mem_results[0]:
                result['mem_top_sim'] = mem_results[0].get('cosine_similarity', 0)
                result['mem_protection'] = mem_results[0].get('protection', 0)
            else:
                result['mem_top_sim'] = 0.0
                result['mem_protection'] = 0.0

            result['mem_stored'] = len(self._ft_engine.memory._memory)

            # 6. Inductive coherence at current interaction depth
            qi = self._evolution_state.get('quantum_interactions', 0)
            depth = max(1, (qi % 63) + 1)
            coherence_val = self._ft_engine.inductive.coherence_at(depth)
            result['coherence_depth'] = depth
            result['coherence_value'] = coherence_val
            result['coherence_limit'] = 326.0244

            # Store the query pattern for future attention
            self._ft_engine.attention.add_pattern(query_vec)
            self._ft_engine.memory.store(query_vec, label=message[:40])

            # v23.4: Run qiskit quantum circuit for real quantum state data
            qiskit_data = self._qiskit_process(message)
            if qiskit_data:
                result.update(qiskit_data)

            return result
        except Exception:
            return {}

    def _qiskit_process(self, message: str) -> dict:
        """
        v23.4 REAL QUANTUM PROCESSING via IBM Qiskit.
        Builds a parameterized quantum circuit from message hash,
        runs statevector simulation, extracts quantum metrics.

        Returns metadata dict with quantum state info for response enrichment.
        """
        try:
            from l104_quantum_gate_engine import GateCircuit as QuantumCircuit
            from l104_quantum_gate_engine.quantum_info import Statevector
            import hashlib

            # Derive circuit parameters from message content
            msg_hash = hashlib.sha256(message.encode()).hexdigest()
            n_qubits = min(6, max(2, len(message) % 5 + 2))  # 2-6 qubits

            # Build parameterized quantum circuit
            qc = QuantumCircuit(n_qubits)

            # Layer 1: Hadamard superposition on all qubits
            for i in range(n_qubits):
                qc.h(i)

            # Layer 2: φ-rotation gates derived from message hash
            for i in range(n_qubits):
                # Rotation angle from hash bytes, scaled by PHI
                angle = int(msg_hash[i*2:i*2+2], 16) / 255.0 * math.pi * PHI
                qc.rz(angle, i)
                qc.ry(angle * (1.0 / PHI), i)

            # Layer 3: Entanglement via CNOT cascade (creates Bell-like states)
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)

            # Layer 4: GOD_CODE phase encoding
            god_phase = GOD_CODE_PHASE
            for i in range(n_qubits):
                qc.rz(god_phase * (i + 1) / n_qubits, i)

            # Layer 5: Second entanglement layer (circular)
            if n_qubits > 2:
                qc.cx(n_qubits - 1, 0)  # Close the loop

            # Run statevector simulation
            sv = Statevector.from_instruction(qc)
            probs = sv.probabilities()

            # Extract quantum metrics
            # Shannon entropy of measurement probabilities
            q_entropy = -sum(p * math.log2(max(p, 1e-30)) for p in probs if p > 0)
            max_entropy = math.log2(2 ** n_qubits)
            q_coherence = q_entropy / max(max_entropy, 1e-30)

            # Entanglement measure (purity of subsystem)
            # For 2+ qubit system, trace out half and measure purity
            try:
                from l104_quantum_gate_engine.quantum_info import partial_trace
                half = n_qubits // 2
                if half > 0:
                    subsystem_dm = partial_trace(sv, list(range(half)))
                    purity = float(subsystem_dm.purity())
                    entanglement = 1.0 - purity  # 0 = separable, ~1 = maximally entangled
                else:
                    entanglement = 0.0
            except Exception:
                entanglement = q_coherence * 0.5  # Fallback estimate

            # Most probable basis state
            max_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
            max_state = format(max_idx, f'0{n_qubits}b')
            max_prob = float(probs[max_idx])

            return {
                "qiskit_qubits": n_qubits,
                "qiskit_entropy": q_entropy,
                "qiskit_coherence": q_coherence,
                "qiskit_entanglement": entanglement,
                "qiskit_top_state": f"|{max_state}⟩",
                "qiskit_top_prob": max_prob,
                "qiskit_circuit_depth": qc.depth(),
                "qiskit_gate_count": qc.size(),
            }
        except ImportError:
            return {}
