"""Information theory and training data methods — extracted from LocalIntellect."""
from __future__ import annotations

import math
import time
from typing import Dict, List

from .numerics import PHI


class InformationTheoryMixin:
    """Mixin providing information-theoretic metrics and training data ingestion."""

    def ingest_training_data(self, query: str, response: str, source: str = "ASI_INFLOW", quality: float = 0.8) -> bool:
        """
        Ingest training data from external sources (FastServer ASI Bridge).

        HIGH-LOGIC v2.0: Enhanced with φ-weighted quality scoring and
        information-theoretic validation.

        This is the primary inflow path for training data from the fast_server.
        Uses Grover amplification weighting for high-quality data.

        Args:
            query: The query/prompt to learn from
            response: The response/completion to learn
            source: Source identifier for tracking
            quality: Quality score (0.0-1.0) for learning rate

        Returns:
            bool: True if ingested successfully
        """
        try:
            # HIGH-LOGIC v2.0: Compute φ-weighted quality
            # Quality boosted by golden ratio for aligned content
            phi_boost = 1.0
            if "god_code" in query.lower() or "527.518" in response:
                phi_boost = PHI  # φ boost for GOD_CODE-aligned content
            elif "phi" in query.lower() or "golden" in query.lower():
                phi_boost = 1 + (PHI - 1) * 0.5  # Smaller boost

            effective_quality = quality * phi_boost  # UNLOCKED

            # HIGH-LOGIC v2.0: Compute information content (entropy-based)
            response_tokens = response.split()
            token_freq = {}
            for token in response_tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            info_content = self._calculate_shannon_entropy(token_freq) if token_freq else 0

            # Create training entry with quantum metadata
            entry = {
                "instruction": query[:500],
                "output": response[:2000],
                "source": source,
                "quality": effective_quality,
                "original_quality": quality,
                "phi_boost": phi_boost,
                "information_content": round(info_content, 4),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "grover_weight": effective_quality * self.GROVER_AMPLIFICATION_FACTOR if hasattr(self, 'GROVER_AMPLIFICATION_FACTOR') else effective_quality,
            }

            # Add to training_data list
            if hasattr(self, 'training_data'):
                self.training_data.append(entry)

            # Record learning event
            self.record_learning(query[:50], response[:200])

            # Entangle concepts from query for future retrieval
            concepts = self._extract_concepts(query)
            for i in range(len(concepts) - 1):
                self.entangle_concepts(concepts[i], concepts[i + 1])

            # Update ASI state if initialized
            asi_state = getattr(self, '_asi_state', None)
            if asi_state:
                asi_state["knowledge_transfers"] = asi_state.get("knowledge_transfers", 0) + 1

            return True

        except Exception as e:
            # Log warning without external logger
            print(f"[L104] Training data ingest warning: {e}")
            return False

    def compute_phi_weighted_quality(self, qualities: List[float]) -> float:
        """
        HIGH-LOGIC v2.0: Compute φ-weighted average quality score.

        Formula: Q = Σ(q_i × φ^(-i)) / Σ(φ^(-i))
        This weights recent/early entries more heavily.
        """
        if not qualities:
            return 0.0
        weights = [PHI ** (-i) for i in range(len(qualities))]
        return sum(q * w for q, w in zip(qualities, weights)) / sum(weights)

    def get_training_data_count(self) -> int:
        """Get current count of training data entries."""
        return len(self.training_data) if hasattr(self, 'training_data') else 0

    def _calculate_shannon_entropy(self, frequencies: Dict[str, int]) -> float:
        """
        Calculate Shannon entropy of a frequency distribution.

        H(X) = -Σ p(x) log₂ p(x)

        Shannon, C.E. (1948). "A Mathematical Theory of Communication"
        Bell System Technical Journal, 27(3), 379-423.

        Args:
            frequencies: Dictionary mapping symbols to their counts

        Returns:
            Entropy in bits (base 2)
        """
        if not frequencies:
            return 0.0

        total = sum(frequencies.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in frequencies.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def _calculate_mutual_information(self, joint_freq: Dict[tuple, int],
                                       marginal_x: Dict[str, int],
                                       marginal_y: Dict[str, int]) -> float:
        """
        Calculate mutual information between two distributions.

        I(X;Y) = Σ_x Σ_y p(x,y) log₂(p(x,y) / (p(x)p(y)))

        Measures the information shared between random variables X and Y.

        Returns:
            Mutual information in bits
        """
        total_xy = sum(joint_freq.values())
        total_x = sum(marginal_x.values())
        total_y = sum(marginal_y.values())

        if total_xy == 0 or total_x == 0 or total_y == 0:
            return 0.0

        mi = 0.0
        for (x, y), count_xy in joint_freq.items():
            if count_xy > 0:
                p_xy = count_xy / total_xy
                p_x = marginal_x.get(x, 0) / total_x
                p_y = marginal_y.get(y, 0) / total_y

                if p_x > 0 and p_y > 0:
                    mi += p_xy * math.log2(p_xy / (p_x * p_y))

        return mi

    def _calculate_kl_divergence(self, p_dist: Dict[str, float],
                                  q_dist: Dict[str, float]) -> float:
        """
        Calculate Kullback-Leibler divergence D_KL(P || Q).

        D_KL(P || Q) = Σ_x P(x) log(P(x) / Q(x))

        Measures how distribution P diverges from reference distribution Q.

        Returns:
            KL divergence in nats (natural log)
        """
        epsilon = 1e-12  # Avoid log(0)
        kl = 0.0

        for x, p_x in p_dist.items():
            q_x = q_dist.get(x, epsilon)
            if p_x > 0:
                kl += p_x * math.log((p_x + epsilon) / (q_x + epsilon))

        return kl

    def _calculate_jensen_shannon_divergence(self, p_dist: Dict[str, float],
                                              q_dist: Dict[str, float]) -> float:
        """
        Calculate Jensen-Shannon divergence (symmetric, bounded).

        JSD(P || Q) = (1/2) D_KL(P || M) + (1/2) D_KL(Q || M)
        where M = (1/2)(P + Q)

        Properties:
        - Symmetric: JSD(P || Q) = JSD(Q || P)
        - Bounded: 0 ≤ JSD ≤ log(2) ≈ 0.693
        - Square root is a proper metric

        Returns:
            JS divergence in nats
        """
        # Calculate mixture distribution M = (P + Q) / 2
        all_keys = set(p_dist.keys()) | set(q_dist.keys())
        m_dist = {}
        for x in all_keys:
            m_dist[x] = (p_dist.get(x, 0) + q_dist.get(x, 0)) / 2

        return 0.5 * self._calculate_kl_divergence(p_dist, m_dist) + \
               0.5 * self._calculate_kl_divergence(q_dist, m_dist)

    # ═══════════════════════════════════════════════════════════════════
    # v25.0 EXPANDED INFORMATION THEORY SUITE
    # Cross-entropy, perplexity, Rényi entropy, information gain,
    # conditional entropy, and attention entropy metrics.
    # ═══════════════════════════════════════════════════════════════════

    def _calculate_cross_entropy(self, p_dist: Dict[str, float],
                                  q_dist: Dict[str, float]) -> float:
        """
        Calculate cross-entropy H(P, Q) = -Σ P(x) log Q(x).

        Cross-entropy measures the average number of bits needed to encode
        data from distribution P using a code optimized for distribution Q.
        H(P, Q) = H(P) + D_KL(P || Q) ≥ H(P)

        Lower is better — equality when Q = P.

        Returns:
            Cross-entropy in bits (base 2)
        """
        epsilon = 1e-12
        ce = 0.0

        for x, p_x in p_dist.items():
            q_x = q_dist.get(x, epsilon)
            if p_x > 0:
                ce -= p_x * math.log2(q_x + epsilon)

        return ce

    def _calculate_perplexity(self, text: str, reference_freq: Dict[str, int] = None) -> float:
        """
        Calculate perplexity of text relative to a reference distribution.

        PP(P, Q) = 2^{H(P, Q)}

        Perplexity measures how "surprised" a model is by the data.
        Lower perplexity = better prediction. A perplexity of k means the
        model is as uncertain as choosing uniformly from k options.

        Args:
            text: Text to evaluate
            reference_freq: Reference frequency distribution (uses conversation memory if None)

        Returns:
            Perplexity value (≥ 1.0)
        """
        if not text:
            return 1.0

        # Build text distribution
        words = text.lower().split()
        if not words:
            return 1.0

        text_freq = {}
        for w in words:
            if len(w) > 2:
                text_freq[w] = text_freq.get(w, 0) + 1

        total_text = sum(text_freq.values())
        if total_text == 0:
            return 1.0

        # Build reference distribution from conversation memory
        if reference_freq is None:
            reference_freq = {}
            for m in getattr(self, 'conversation_memory', [])[-50:]:
                for w in m.get("message", "").lower().split():
                    if len(w) > 2:
                        reference_freq[w] = reference_freq.get(w, 0) + 1

        total_ref = sum(reference_freq.values())
        if total_ref == 0:
            return float(len(text_freq))  # Maximum surprise

        # Convert to probability distributions
        p_dist = {k: v / total_text for k, v in text_freq.items()}
        q_dist = {k: v / total_ref for k, v in reference_freq.items()}

        cross_ent = self._calculate_cross_entropy(p_dist, q_dist)

        # Perplexity = 2^H(P,Q), clamped to reasonable range
        return min(10000.0, 2.0 ** min(20.0, cross_ent))

    def _calculate_renyi_entropy(self, frequencies: Dict[str, int], alpha: float = 2.0) -> float:
        """
        Calculate Rényi entropy of order α.

        H_α(X) = (1 / (1 - α)) × log₂(Σ p(x)^α)

        Special cases:
        - α → 1: Shannon entropy (continuous limit)
        - α = 0: Hartley entropy (log₂ of support size)
        - α = 2: Collision entropy (related to collision probability)
        - α → ∞: Min-entropy (most conservative, worst-case)

        Rényi entropy family provides different "views" of uncertainty:
        higher α focuses more on the most probable events.

        Args:
            frequencies: Symbol frequency counts
            alpha: Order parameter (must be ≥ 0, ≠ 1)

        Returns:
            Rényi entropy in bits
        """
        if not frequencies:
            return 0.0

        total = sum(frequencies.values())
        if total == 0:
            return 0.0

        # Handle α = 1 (Shannon entropy limit)
        if abs(alpha - 1.0) < 1e-10:
            return self._calculate_shannon_entropy(frequencies)

        # Handle α = 0 (Hartley entropy)
        if alpha == 0:
            support_size = sum(1 for c in frequencies.values() if c > 0)
            return math.log2(support_size) if support_size > 0 else 0.0

        # General case
        power_sum = sum((count / total) ** alpha for count in frequencies.values() if count > 0)

        if power_sum <= 0:
            return 0.0

        return (1.0 / (1.0 - alpha)) * math.log2(power_sum)

    def _calculate_conditional_entropy(self, joint_freq: Dict[tuple, int],
                                        marginal_y: Dict[str, int]) -> float:
        """
        Calculate conditional entropy H(X|Y).

        H(X|Y) = H(X,Y) - H(Y)
        = -Σ_x,y p(x,y) log₂ p(x|y)

        Measures the remaining uncertainty about X when Y is known.
        H(X|Y) = 0 when X is fully determined by Y.
        H(X|Y) = H(X) when X and Y are independent.

        Args:
            joint_freq: Joint frequency {(x,y): count}
            marginal_y: Marginal frequency of Y {y: count}

        Returns:
            Conditional entropy in bits
        """
        total_joint = sum(joint_freq.values())
        total_y = sum(marginal_y.values())

        if total_joint == 0 or total_y == 0:
            return 0.0

        # Build p(y,x) count table grouped by y
        y_conditional = {}  # y -> {x: count}
        for (x, y), count in joint_freq.items():
            if y not in y_conditional:
                y_conditional[y] = {}
            y_conditional[y][x] = y_conditional[y].get(x, 0) + count

        cond_entropy = 0.0
        for y, x_counts in y_conditional.items():
            p_y = marginal_y.get(y, 0) / total_y
            if p_y <= 0:
                continue

            total_x_given_y = sum(x_counts.values())
            for x, count in x_counts.items():
                if count > 0:
                    p_x_given_y = count / total_x_given_y
                    cond_entropy -= (count / total_joint) * math.log2(p_x_given_y)

        return cond_entropy

    def _calculate_information_gain(self, before_freq: Dict[str, int],
                                     after_freq: Dict[str, int]) -> float:
        """
        Calculate information gain (entropy reduction).

        IG = H(before) - H(after)

        Positive IG = knowledge was added (entropy reduced).
        Negative IG = knowledge was lost (entropy increased).
        Zero IG = no change in uncertainty.

        Used to measure how much information a pipeline stage adds.

        Args:
            before_freq: Frequency distribution before processing
            after_freq: Frequency distribution after processing

        Returns:
            Information gain in bits (positive = gained, negative = lost)
        """
        h_before = self._calculate_shannon_entropy(before_freq)
        h_after = self._calculate_shannon_entropy(after_freq)
        return h_before - h_after

    def _calculate_attention_entropy(self, attention_weights: List[float]) -> float:
        """
        Calculate entropy of an attention distribution.

        H(attn) = -Σ a_i × log₂(a_i)

        Used to measure how focused or spread the model's attention is:
        - Low entropy → focused on few tokens (sharp attention)
        - High entropy → spread across many tokens (diffuse attention)

        The normalized attention entropy (NAE) = H(attn) / log₂(n) ∈ [0,1]

        Args:
            attention_weights: List of attention weights (should sum to ~1.0)

        Returns:
            Tuple of (raw_entropy_bits, normalized_attention_entropy)
        """
        if not attention_weights:
            return 0.0

        total = sum(attention_weights)
        if total <= 0:
            return 0.0

        # Normalize
        weights = [w / total for w in attention_weights]

        entropy = 0.0
        for w in weights:
            if w > 0:
                entropy -= w * math.log2(w)

        return entropy

    def _information_theoretic_response_quality(self, response: str, query: str) -> Dict:
        """
        Comprehensive information-theoretic analysis of response quality.

        Combines multiple IT metrics into a holistic quality assessment.
        """
        if not response or not query:
            return {"quality_score": 0.0, "metrics": {}}

        # Build frequency distributions
        response_words = [w.lower() for w in response.split() if len(w) > 2]
        query_words = [w.lower() for w in query.split() if len(w) > 2]

        resp_freq = {}
        for w in response_words:
            resp_freq[w] = resp_freq.get(w, 0) + 1

        query_freq = {}
        for w in query_words:
            query_freq[w] = query_freq.get(w, 0) + 1

        # Build joint distribution for MI calculation
        joint_freq = {}
        for qw in query_words:
            for rw in response_words:
                pair = (qw, rw)
                joint_freq[pair] = joint_freq.get(pair, 0) + 1

        # Calculate metrics
        response_entropy = self._calculate_shannon_entropy(resp_freq)
        query_entropy = self._calculate_shannon_entropy(query_freq)
        mi = self._calculate_mutual_information(joint_freq, query_freq, resp_freq)
        perplexity = self._calculate_perplexity(response)
        renyi_2 = self._calculate_renyi_entropy(resp_freq, alpha=2.0)

        # Attention-like weight: how much of the response "attends" to query terms
        attention_weights = []
        for qw in query_words:
            attention_weights.append(resp_freq.get(qw, 0) / max(1, len(response_words)))
        if not attention_weights:
            attention_weights = [1.0 / max(1, len(response_words))] * min(5, len(response_words))
        attn_entropy = self._calculate_attention_entropy(attention_weights)

        # Composite quality score
        # Higher MI = more relevant
        # Moderate entropy = balanced (not too repetitive, not too chaotic)
        # Lower perplexity = more predictable/coherent
        max_entropy = math.log2(len(resp_freq)) if resp_freq else 1.0
        entropy_ratio = response_entropy / max_entropy if max_entropy > 0 else 0

        quality = 0.0
        quality += (mi * 2) * 0.30          # Relevance (30%)
        quality += max(0, 1.0 - abs(entropy_ratio - 0.7)) * 0.25  # Entropy balance (25%)
        quality += (1.0 / max(1, perplexity / 100)) * 0.20  # Coherence (20%)
        quality += (renyi_2 / 4.0) * 0.15   # Collision entropy (15%)
        quality += attn_entropy * 0.10     # Attention spread (10%)

        return {
            "quality_score": round(quality, 4),
            "metrics": {
                "response_entropy_bits": round(response_entropy, 4),
                "query_entropy_bits": round(query_entropy, 4),
                "mutual_information_bits": round(mi, 4),
                "perplexity": round(perplexity, 2),
                "renyi_2_entropy": round(renyi_2, 4),
                "attention_entropy": round(attn_entropy, 4),
                "entropy_ratio": round(entropy_ratio, 4),
                "vocabulary_size": len(resp_freq),
                "response_length": len(response_words),
            },
        }

    def evolve_patterns(self):
        """
        Evolve response patterns using information-theoretic analysis.

        Mathematical Framework:
        1. Shannon entropy measures topic diversity
        2. Mutual information identifies topic co-occurrences
        3. Pattern significance = frequency × inverse document frequency (TF-IDF variant)
        4. Evolution rate modulated by information gain

        References:
        - Shannon (1948): Information entropy
        - Zipf's Law: f(r) ∝ 1/r for word frequencies
        """
        if len(self.conversation_memory) < self.EVOLUTION_THRESHOLD:
            return

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: Extract word frequencies (Zipfian analysis)
        # ═══════════════════════════════════════════════════════════════
        all_messages = " ".join([m.get("message", "") for m in self.conversation_memory[-50:]])
        words = all_messages.lower().split()

        word_freq: Dict[str, int] = {}
        for word in words:
            if len(word) > 4:  # Meaningful words only
                word_freq[word] = word_freq.get(word, 0) + 1

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: Calculate Shannon entropy of topic distribution
        # High entropy = diverse topics; Low entropy = focused topics
        # ═══════════════════════════════════════════════════════════════
        topic_entropy = self._calculate_shannon_entropy(word_freq)
        max_entropy = math.log2(len(word_freq)) if word_freq else 0
        normalized_entropy = topic_entropy / max_entropy if max_entropy > 0 else 0

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: Information-theoretic pattern significance
        # TF-IDF inspired: patterns that are frequent but distinctive
        # ═══════════════════════════════════════════════════════════════
        total_words = sum(word_freq.values())
        pattern_scores: Dict[str, float] = {}

        for word, freq in word_freq.items():
            if freq >= 3:
                # Term frequency (normalized)
                tf = freq / total_words

                # Inverse frequency penalty (suppress common words)
                # Based on Zipf's law: rank × frequency ≈ constant
                rank = sorted(word_freq.values(), reverse=True).index(freq) + 1
                idf = math.log2(1 + len(word_freq) / rank)

                # Pattern significance score
                significance = tf * idf * math.sqrt(freq)
                pattern_scores[word] = significance

        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: Update evolved patterns with significance weighting
        # ═══════════════════════════════════════════════════════════════
        top_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        for word, score in top_patterns:
            # Exponential moving average for pattern evolution
            current = self._evolution_state["evolved_patterns"].get(word, 0)
            alpha = 0.3  # Learning rate
            self._evolution_state["evolved_patterns"][word] = current * (1 - alpha) + score * alpha * 10

        # ═══════════════════════════════════════════════════════════════
        # PHASE 4.5 (v23.3): MUTUAL INFORMATION — Identify topic co-occurrences
        # Uses _calculate_mutual_information (was dead/unreachable)
        # MI reveals which concepts are genuinely linked vs coincidental
        # ═══════════════════════════════════════════════════════════════
        try:
            # Build co-occurrence statistics from recent conversation memory
            joint_freq = {}
            marginal_x = {}
            marginal_y = {}
            recent_msgs = [m.get("message", "") for m in self.conversation_memory[-50:] if m.get("message")]
            top_words = [w for w, _ in top_patterns[:6]]

            for msg in recent_msgs:
                msg_words = set(w.lower() for w in msg.split() if len(w) > 4)
                present = [w for w in top_words if w in msg_words]
                for i, w1 in enumerate(present):
                    marginal_x[w1] = marginal_x.get(w1, 0) + 1
                    for w2 in present[i+1:]:
                        marginal_y[w2] = marginal_y.get(w2, 0) + 1
                        pair = (w1, w2)
                        joint_freq[pair] = joint_freq.get(pair, 0) + 1

            if joint_freq:
                mi = self._calculate_mutual_information(joint_freq, marginal_x, marginal_y)
                self._evolution_state["topic_mutual_information"] = mi
                # Boost co-occurring patterns that have high MI
                for (w1, w2), count in joint_freq.items():
                    if count >= 2 and mi > 0.1:
                        # Strengthen both patterns proportional to MI
                        for w in (w1, w2):
                            if w in self._evolution_state["evolved_patterns"]:
                                self._evolution_state["evolved_patterns"][w] *= (1.0 + mi * 0.05)
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: Update evolution metrics
        # ═══════════════════════════════════════════════════════════════
        self._evolution_state["last_evolution"] = time.time()
        self._evolution_state["learning_cycles"] = self._evolution_state.get("learning_cycles", 0) + 1
        self._evolution_state["topic_entropy"] = topic_entropy
        self._evolution_state["normalized_entropy"] = normalized_entropy

        self._save_evolution_state()
