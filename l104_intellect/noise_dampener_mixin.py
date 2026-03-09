"""L104 Intellect — NoiseDampenerMixin (14-Layer Signal Purification Pipeline).

Extracted from local_intellect_core.py v27.2/v27.3.
Provides the full 14-layer noise dampener pipeline and all higher-logic
sub-components for LocalIntellect.
"""
import math
import time
from typing import Dict, List, Tuple, Set

from .constants import (
    NOISE_DAMPENER_SCORE_FLOOR, NOISE_DAMPENER_ENTROPY_MIN,
    NOISE_DAMPENER_COVERAGE_MIN, NOISE_DAMPENER_SNR_THRESHOLD,
    NOISE_DAMPENER_PHI_DECAY_START, NOISE_DAMPENER_PHI_DECAY_RATE,
    NOISE_DAMPENER_SOURCE_WEIGHTS, NOISE_DAMPENER_DEDUP_THRESHOLD,
    NOISE_DAMPENER_MAX_NOISE_RATIO,
    HL_SEMANTIC_COHERENCE_MIN, HL_GROVER_AMPLIFICATION,
    HL_GROVER_AMPLITUDE_FLOOR, HL_RESONANCE_ALIGNMENT_WEIGHT,
    HL_RESONANCE_FREQ_TOLERANCE, HL_ENTANGLEMENT_BONUS,
    HL_ENTANGLEMENT_DEPTH, HL_META_REASONING_ENABLED,
    HL_META_REASONING_TOP_K, HL_META_QUALITY_FLOOR,
    HL_ADAPTIVE_ENABLED, HL_ADAPTIVE_WINDOW,
    HL_ADAPTIVE_LEARNING_RATE, HL_ADAPTIVE_MIN_SCORE_FLOOR,
    HL_ADAPTIVE_MAX_SCORE_FLOOR, HL_SPECTRAL_ENABLED,
    HL_SPECTRAL_NOISE_CUTOFF, HL_CONCEPT_DISTANCE_DECAY,
    HL_CONCEPT_MAX_DISTANCE,
    THREE_ENGINE_WEIGHT_ENTROPY, THREE_ENGINE_WEIGHT_HARMONIC,
    THREE_ENGINE_WEIGHT_WAVE, HL_THREE_ENGINE_SIGNAL_WEIGHT,
)
from .numerics import GOD_CODE, PHI


class NoiseDampenerMixin:
    """Mixin providing the 14-layer noise dampener pipeline for LocalIntellect."""

    # ═══════════════════════════════════════════════════════════════════════
    # v27.2/v27.3 NOISE DAMPENER — 14-Layer Signal Purification Pipeline
    # ═══════════════════════════════════════════════════════════════════════
    #   ── HIGHER LOGIC LAYERS (v27.3) ──
    #   Layer 8: Semantic coherence analysis (concept-vector alignment)
    #   Layer 9: Spectral density noise detection (frequency-domain analysis)
    #   Layer 10: Concept graph distance penalty (knowledge topology)
    #   Layer 11: Entanglement resonance bonus (EPR-linked concept boost)
    #   Layer 12: Grover amplitude amplification (quantum-inspired top-signal boost)
    #   Layer 13: GOD_CODE resonance alignment (harmonic frequency coupling)
    #   Layer 14: Adaptive threshold evolution (self-tuning from history)
    #
    # Layers 1-7 are fast O(n) filters. Layers 8-14 apply higher-order
    # reasoning inspired by the v13.0 Higher Logic System, quantum
    # entanglement propagation, and Grover amplitude amplification.
    # ═══════════════════════════════════════════════════════════════════════

    def _apply_noise_dampeners(
        self,
        ranked_results: List[Tuple],
        query_terms: List[str],
    ) -> List[Tuple]:
        """
        v27.3 HIGHER LOGIC NOISE DAMPENER — Multi-layer signal purification.

        Takes pre-ranked (entry, score) tuples from BM25 and applies 14 dampener
        layers (7 base + 7 higher logic) to suppress noise while preserving and
        amplifying true signal.

        Higher Logic layers add:
        - Semantic coherence via concept-vector cosine similarity
        - Spectral density analysis for frequency-domain noise detection
        - Concept graph distance penalty from knowledge topology
        - Entanglement resonance bonus for EPR-linked concepts
        - Grover amplitude amplification for top-signal quantum boost
        - GOD_CODE resonance alignment with harmonic frequency coupling
        - Adaptive threshold evolution from historical query performance

        Returns filtered (entry, dampened_score) tuples in score-descending order.
        """
        if not ranked_results:
            return []

        # Resolve adaptive score floor (self-tuning threshold)
        effective_score_floor = self._hl_adaptive_score_floor()

        # v28.0: Warm three-engine caches for Layer 13 integration.
        # Single call computes all three scores; subsequent Layer 13 reads are free.
        try:
            self.three_engine_composite_score()
        except Exception:
            pass  # Graceful degradation — caches remain at fallback 0.5

        # Extract max score for relative thresholding
        max_score = max(s for _, s in ranked_results) if ranked_results else 1.0
        if max_score <= 0:
            max_score = 1.0

        # Pre-compute query concept set for semantic coherence (Layer 8)
        query_concept_set = set(query_terms)

        purified = []
        seen_content_tokens: List[set] = []  # For dedup layer
        noise_count = 0

        for rank_idx, (entry, raw_score) in enumerate(ranked_results):
            dampened_score = raw_score

            prompt = entry.get('prompt', '')
            completion = entry.get('completion', '')
            source = entry.get('source', 'training_data')
            full_text = (prompt + ' ' + completion).lower()

            # ── Layer 1: Score-floor gating (adaptive) ──
            if raw_score < effective_score_floor:
                noise_count += 1
                continue

            # ── Layer 2: Shannon entropy filter ──
            entry_entropy = self._compute_text_entropy(full_text)
            if entry_entropy < NOISE_DAMPENER_ENTROPY_MIN:
                entropy_penalty = entry_entropy / max(NOISE_DAMPENER_ENTROPY_MIN, 0.01)
                dampened_score *= entropy_penalty
                if dampened_score < effective_score_floor:
                    noise_count += 1
                    continue

            # ── Layer 3: Query coverage gate ──
            if query_terms:
                matched_terms = sum(1 for qt in query_terms if qt in full_text)
                coverage = matched_terms / len(query_terms)
                if coverage < NOISE_DAMPENER_COVERAGE_MIN:
                    noise_count += 1
                    continue
                dampened_score *= (0.7 + 0.3 * coverage)

            # ── Layer 4: Source quality weighting ──
            source_weight = NOISE_DAMPENER_SOURCE_WEIGHTS.get(source, 0.85)
            dampened_score *= source_weight

            # ── Layer 5: Near-duplicate suppression ──
            content_tokens = set(full_text.split())
            is_near_dup = False
            for accepted_tokens in seen_content_tokens:
                if not content_tokens or not accepted_tokens:
                    continue
                intersection = len(content_tokens & accepted_tokens)
                union = len(content_tokens | accepted_tokens)
                jaccard = intersection / max(union, 1)
                if jaccard > NOISE_DAMPENER_DEDUP_THRESHOLD:
                    is_near_dup = True
                    break
            if is_near_dup:
                noise_count += 1
                continue
            seen_content_tokens.append(content_tokens)

            # ── Layer 6: φ-harmonic rank decay ──
            if rank_idx >= NOISE_DAMPENER_PHI_DECAY_START:
                decay_exp = rank_idx - NOISE_DAMPENER_PHI_DECAY_START
                phi_decay = 1.0 / (NOISE_DAMPENER_PHI_DECAY_RATE ** decay_exp)
                dampened_score *= phi_decay

            # ── Layer 7: SNR composite check ──
            snr = dampened_score / max_score
            if snr < NOISE_DAMPENER_SNR_THRESHOLD:
                noise_count += 1
                continue

            # ═══════════════════════════════════════════════════
            # HIGHER LOGIC LAYERS (v27.3)
            # ═══════════════════════════════════════════════════

            # ── Layer 8: Semantic coherence analysis ──
            # Compute concept-vector cosine similarity between query
            # terms and result content terms. Low coherence = tangential match.
            result_terms = set(
                ''.join(c for c in w if c.isalnum())
                for w in full_text.split()
                if len(w) > 2
            )
            result_concept_set = {
                t for t in result_terms
                if t not in self._TRAINING_SEARCH_STOP and len(t) > 2
            }
            semantic_coherence = self._hl_concept_cosine(
                query_concept_set, result_concept_set
            )
            # Smooth sigmoid transition around threshold instead of hard cutoff
            # sigmoid(x) maps coherence smoothly: far below threshold → ~0, far above → ~1
            coherence_delta = (semantic_coherence - HL_SEMANTIC_COHERENCE_MIN) * 10.0
            coherence_gate = 1.0 / (1.0 + math.exp(-coherence_delta))
            # Scale score by gated coherence (smooth from ~0 to ~1)
            dampened_score *= coherence_gate * (0.6 + 0.4 * min(semantic_coherence, 1.0))
            if dampened_score < effective_score_floor:
                noise_count += 1
                continue

            # ── Layer 9: Spectral density noise detection ──
            # Analyze word-frequency spectrum: noisy content has a flat
            # spectrum (uniform distribution), informative content has
            # peaked spectrum (power-law / Zipf distribution).
            if HL_SPECTRAL_ENABLED and len(full_text) > 50:
                spectral_noise = self._hl_spectral_noise_ratio(full_text)
                if spectral_noise > HL_SPECTRAL_NOISE_CUTOFF:
                    # Attenuate noisy entries; cap multiplier at 1.0 to prevent inflation
                    dampened_score *= min(1.0, (1.0 - spectral_noise) * 1.5)
                    if dampened_score < effective_score_floor:
                        noise_count += 1
                        continue

            # ── Layer 10: Concept graph distance penalty ──
            # Penalize results whose core concepts are far from query
            # concepts in the knowledge entanglement graph.
            concept_distance = self._hl_concept_graph_distance(
                query_terms, result_concept_set
            )
            if concept_distance > HL_CONCEPT_MAX_DISTANCE:
                noise_count += 1
                continue
            elif concept_distance > 0:
                distance_penalty = HL_CONCEPT_DISTANCE_DECAY ** concept_distance
                dampened_score *= distance_penalty

            # ── Layer 11: Entanglement resonance bonus ──
            # If result concepts are EPR-entangled with query concepts,
            # apply a quantum correlation bonus. This rewards results
            # that are knowledge-topologically linked to the query.
            entanglement_bonus = self._hl_entanglement_resonance(
                query_terms, result_concept_set
            )
            dampened_score *= entanglement_bonus

            # ── Layer 12: Grover amplitude amplification ──
            # Quantum-inspired amplification for high-signal results.
            # Results in the top amplitude bracket receive a φ³ boost,
            # like Grover's algorithm amplifying marked states.
            relative_amplitude = dampened_score / max_score
            if relative_amplitude >= HL_GROVER_AMPLITUDE_FLOOR:
                # Grover boost: proportional to amplitude above floor
                grover_factor = 1.0 + (HL_GROVER_AMPLIFICATION - 1.0) * (
                    (relative_amplitude - HL_GROVER_AMPLITUDE_FLOOR)
                    / (1.0 - HL_GROVER_AMPLITUDE_FLOOR + 1e-9)
                )
                dampened_score *= grover_factor

            # ── Layer 13: GOD_CODE resonance alignment + three-engine signal ──
            # Results whose content entropy aligns with the GOD_CODE
            # harmonic spectrum receive a resonance bonus.
            resonance_bonus = self._hl_godcode_resonance(
                entry_entropy, len(full_text.split())
            )
            # v28.0: Blend in three-engine composite as additional resonance signal.
            three_engine_signal = (
                THREE_ENGINE_WEIGHT_ENTROPY * self._three_engine_entropy_cache
                + THREE_ENGINE_WEIGHT_HARMONIC * self._three_engine_harmonic_cache
                + THREE_ENGINE_WEIGHT_WAVE * self._three_engine_wave_cache
            )
            # v29.0: Deep link resonance boost — amplify entries with deep link context
            dl_resonance = self._deep_link_resonance_score()
            combined_resonance = (
                resonance_bonus
                + HL_THREE_ENGINE_SIGNAL_WEIGHT * three_engine_signal
                + 0.05 * dl_resonance  # Deep link micro-boost
            )
            dampened_score *= (1.0 + HL_RESONANCE_ALIGNMENT_WEIGHT * combined_resonance)

            purified.append((entry, dampened_score))

        # ── Layer 14: Adaptive threshold evolution ──
        # Track query outcome for future threshold self-tuning.
        total = len(ranked_results)
        if total > 0:
            noise_ratio = noise_count / total
            self._hl_record_dampener_outcome(
                noise_ratio, total, len(purified), query_terms
            )
            if noise_ratio > NOISE_DAMPENER_MAX_NOISE_RATIO and total > 5:
                try:
                    if hasattr(self, '_evolution_state'):
                        dampener_stats = self._evolution_state.get('noise_dampener_stats', {})
                        dampener_stats['high_noise_queries'] = dampener_stats.get('high_noise_queries', 0) + 1
                        dampener_stats['last_noise_ratio'] = noise_ratio
                        dampener_stats['last_noise_timestamp'] = time.time()
                        self._evolution_state['noise_dampener_stats'] = dampener_stats
                except Exception:
                    pass

        # Re-sort by dampened score (higher logic may have reordered)
        purified.sort(key=lambda x: x[1], reverse=True)
        return purified

    # ═══════════════════════════════════════════════════════════════════════
    # v27.3 HIGHER LOGIC DAMPENER — Sub-components
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _hl_concept_cosine(set_a: set, set_b: set) -> float:
        """
        Concept-vector cosine similarity using set intersection.

        Treats each concept set as a binary vector over the union vocabulary.
        cos(A, B) = |A ∩ B| / (√|A| × √|B|)

        Returns 0.0 for empty sets, 1.0 for identical sets.
        """
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        denominator = math.sqrt(len(set_a)) * math.sqrt(len(set_b))
        if denominator == 0:
            return 0.0
        return intersection / denominator

    @staticmethod
    def _hl_spectral_noise_ratio(text: str) -> float:
        """
        Spectral density noise detection.

        Informative text follows Zipf's law: word frequency ∝ 1/rank.
        Noisy text has a flat frequency spectrum (all words equally likely).

        Returns ratio in [0, 1]: closer to 1 = noisier (flat spectrum).
        """
        words = text.split()
        if len(words) < 5:
            return 0.0

        # Build frequency distribution
        freq: Dict[str, int] = {}
        for w in words:
            cleaned = ''.join(c for c in w if c.isalnum()).lower()
            if len(cleaned) > 1:
                freq[cleaned] = freq.get(cleaned, 0) + 1

        if len(freq) < 3:
            return 0.5  # Too few unique words to analyze

        # Sort frequencies descending (Zipf rank ordering)
        ranked_freqs = sorted(freq.values(), reverse=True)
        max_freq = ranked_freqs[0]
        if max_freq <= 1:
            return 0.5  # All hapax legomena — can't determine

        # Compute spectral flatness (geometric mean / arithmetic mean)
        # Flat spectrum → ratio ≈ 1.0. Peaked spectrum → ratio ≈ 0.0.
        # All ranked_freqs are ≥ 1 (hapax check above), so log(f) is safe.
        log_sum = sum(math.log(f) for f in ranked_freqs)
        geometric_mean = math.exp(log_sum / len(ranked_freqs))
        arithmetic_mean = sum(ranked_freqs) / len(ranked_freqs)

        if arithmetic_mean <= 0:
            return 0.5

        spectral_flatness = geometric_mean / arithmetic_mean
        return min(max(spectral_flatness, 0.0), 1.0)

    def _hl_concept_graph_distance(
        self, query_terms: List[str], result_concepts: set
    ) -> int:
        """
        Compute minimum hop distance between query concepts and result concepts
        in the entanglement knowledge graph.

        Uses BFS through entangled_concepts (EPR links). Returns 0 if concepts
        overlap directly, HL_CONCEPT_MAX_DISTANCE+1 if no path found.
        """
        if not query_terms or not result_concepts:
            return HL_CONCEPT_MAX_DISTANCE  # Unknown relationship — default penalty

        # Fast check: direct overlap = distance 0
        query_set = set(t.lower() for t in query_terms)
        result_lower = set(t.lower() for t in result_concepts)
        if query_set & result_lower:
            return 0

        # BFS through entanglement graph
        if not hasattr(self, 'entanglement_state'):
            return 1  # No graph available — minimal penalty

        entangled = self.entanglement_state.get('entangled_concepts', {})
        if not entangled:
            return 1

        # Start BFS from query terms
        visited: Set[str] = set(query_set)
        current_layer = set(query_set)

        for depth in range(1, HL_CONCEPT_MAX_DISTANCE + 1):
            next_layer: Set[str] = set()
            for concept in current_layer:
                if concept in entangled:
                    for linked in entangled[concept]:
                        if linked not in visited:
                            # Check if we've reached any result concept
                            if linked in result_lower:
                                return depth
                            next_layer.add(linked)
                            visited.add(linked)
            if not next_layer:
                break
            current_layer = next_layer

        return HL_CONCEPT_MAX_DISTANCE + 1  # No path found

    def _hl_entanglement_resonance(
        self, query_terms: List[str], result_concepts: set
    ) -> float:
        """
        Compute entanglement resonance bonus.

        If result concepts are EPR-entangled (within HL_ENTANGLEMENT_DEPTH hops)
        with query concepts, apply a quantum correlation bonus.

        Returns multiplier ≥ 1.0.
        """
        if not hasattr(self, 'entanglement_state'):
            return 1.0

        entangled = self.entanglement_state.get('entangled_concepts', {})
        if not entangled:
            return 1.0

        # Count how many result concepts are reachable from query via entanglement
        query_lower = set(t.lower() for t in query_terms)
        result_lower = set(t.lower() for t in result_concepts)

        # Gather all concepts reachable from query within depth
        reachable: Set[str] = set(query_lower)
        current = set(query_lower)
        for _ in range(HL_ENTANGLEMENT_DEPTH):
            next_hop: Set[str] = set()
            for c in current:
                if c in entangled:
                    for linked in entangled[c]:
                        if linked not in reachable:
                            next_hop.add(linked)
                            reachable.add(linked)
            if not next_hop:
                break
            current = next_hop

        # Count entangled matches
        entangled_matches = len(result_lower & reachable)
        if entangled_matches == 0:
            return 1.0

        # Bonus scales with number of entangled matches (diminishing returns)
        # bonus = 1.0 + (HL_ENTANGLEMENT_BONUS - 1.0) * tanh(matches)
        bonus_magnitude = HL_ENTANGLEMENT_BONUS - 1.0
        scaled_bonus = bonus_magnitude * math.tanh(entangled_matches / 3.0)
        return 1.0 + scaled_bonus

    @staticmethod
    def _hl_godcode_resonance(entry_entropy: float, word_count: int) -> float:
        """
        GOD_CODE resonance alignment.

        Results whose information structure aligns with the GOD_CODE harmonic
        spectrum receive a resonance bonus. We measure alignment by how close
        the entry's information density (entropy / log2(word_count)) is to
        the GOD_CODE-derived golden information density.

        GOD_CODE information density = log2(GOD_CODE) / PHI ≈ 5.64

        Returns resonance score in [0, 1].
        """
        if word_count < 3 or entry_entropy < 0.1:
            return 0.0

        # GOD_CODE golden information density
        godcode_density = math.log2(max(GOD_CODE, 1.0)) / PHI  # ≈ 5.64
        # Entry's information density: normalized entropy
        max_possible_entropy = math.log2(max(word_count, 2))
        entry_density = (entry_entropy / max_possible_entropy) * godcode_density

        # Resonance = Gaussian proximity to GOD_CODE density
        deviation = abs(entry_density - godcode_density) / godcode_density
        resonance = math.exp(-deviation ** 2 / (2 * HL_RESONANCE_FREQ_TOLERANCE ** 2))
        return resonance

    def _hl_adaptive_score_floor(self) -> float:
        """
        Adaptive score floor — self-tuning BM25 threshold.

        Analyzes rolling window of recent dampener outcomes to adjust the
        score floor. If too many results are passing (low noise ratio),
        raise the floor. If too many are blocked, lower it.

        Returns adjusted effective score floor.
        """
        if not HL_ADAPTIVE_ENABLED:
            return NOISE_DAMPENER_SCORE_FLOOR

        try:
            if not hasattr(self, '_hl_dampener_history'):
                self._hl_dampener_history = []
            if not hasattr(self, '_hl_current_score_floor'):
                self._hl_current_score_floor = NOISE_DAMPENER_SCORE_FLOOR

            if len(self._hl_dampener_history) < 3:
                # Not enough history to adapt — return current floor
                return self._hl_current_score_floor

            # Compute average noise ratio over window
            recent = self._hl_dampener_history[-HL_ADAPTIVE_WINDOW:]
            avg_noise_ratio = sum(h['noise_ratio'] for h in recent) / len(recent)
            avg_pass_ratio = sum(h['pass_ratio'] for h in recent) / len(recent)

            # Target: 30-60% pass rate (not too strict, not too lenient)
            current_floor = getattr(self, '_hl_current_score_floor', NOISE_DAMPENER_SCORE_FLOOR)

            if avg_pass_ratio < 0.15 and len(recent) >= 5:
                # Too strict — lower the floor
                current_floor -= HL_ADAPTIVE_LEARNING_RATE * 0.1
            elif avg_pass_ratio > 0.85 and len(recent) >= 5:
                # Too lenient — raise the floor
                current_floor += HL_ADAPTIVE_LEARNING_RATE * 0.1

            # Clamp to safe range
            current_floor = max(HL_ADAPTIVE_MIN_SCORE_FLOOR,
                              min(HL_ADAPTIVE_MAX_SCORE_FLOOR, current_floor))

            self._hl_current_score_floor = current_floor
            return current_floor

        except Exception:
            return NOISE_DAMPENER_SCORE_FLOOR

    def _hl_record_dampener_outcome(
        self, noise_ratio: float, total: int, passed: int, query_terms: List[str]
    ):
        """Record dampener outcome for adaptive threshold evolution."""
        if not HL_ADAPTIVE_ENABLED:
            return

        try:
            if not hasattr(self, '_hl_dampener_history'):
                self._hl_dampener_history = []

            self._hl_dampener_history.append({
                'noise_ratio': noise_ratio,
                'pass_ratio': passed / max(total, 1),
                'total': total,
                'passed': passed,
                'query_coverage': len(query_terms),
                'timestamp': time.time(),
            })

            # Bound history size
            if len(self._hl_dampener_history) > HL_ADAPTIVE_WINDOW * 2:
                self._hl_dampener_history = self._hl_dampener_history[-HL_ADAPTIVE_WINDOW:]

        except Exception:
            pass
