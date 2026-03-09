"""L104 Intellect — KnowledgeSearchMixin

Extracts the knowledge search/retrieval subsystem from local_intellect_core.py.
Handles searching and ranking across all knowledge sources:
  - BM25 + TF-IDF training data search with noise dampening
  - Training index builder (inverted index with IDF weights)
  - Chat conversation search
  - Knowledge manifold pattern search
  - Knowledge vault proof/documentation search
  - Unified all-knowledge recursive JSON search
  - GQA noise dampener (Higher Logic v27.3)
  - Shannon entropy text scorer
"""

import hashlib
import math
from typing import Any, Dict, List, Optional, Set

from .constants import (
    NOISE_DAMPENER_ENTROPY_MIN,
    NOISE_DAMPENER_COVERAGE_MIN,
    NOISE_DAMPENER_SOURCE_WEIGHTS,
    NOISE_DAMPENER_PHI_DECAY_START,
    NOISE_DAMPENER_PHI_DECAY_RATE,
    HL_SEMANTIC_COHERENCE_MIN,
    HL_GROVER_AMPLIFICATION,
    HL_GROVER_AMPLITUDE_FLOOR,
    HL_RESONANCE_ALIGNMENT_WEIGHT,
    HL_SPECTRAL_ENABLED,
    HL_SPECTRAL_NOISE_CUTOFF,
)


class KnowledgeSearchMixin:
    """Mixin providing knowledge-search methods for LocalIntellect."""

    # v23.4 Common single-word intents + instruction verbs that should NOT match training data
    # (these are handled by exact_matches / kernel_synthesis instead)
    _TRAINING_SEARCH_STOP = frozenset({
        'status', 'hello', 'help', 'state', 'running', 'alive', 'health',
        'test', 'ping', 'info', 'about', 'what', 'your', 'with', 'that',
        'this', 'have', 'from', 'will', 'been', 'they', 'them', 'does',
        'were', 'into', 'more', 'some', 'than', 'each', 'make', 'like',
        'just', 'over', 'such', 'also', 'back', 'much', 'when', 'only',
        # v23.4: Instruction/command words — match TOPIC words not VERBS
        'tell', 'know', 'explain', 'describe', 'please', 'could', 'would',
        'should', 'talk', 'give', 'show', 'want', 'need', 'think', 'mean',
        'these', 'those', 'there', 'here', 'very', 'really', 'thing',
        'things', 'something', 'anything', 'everything', 'nothing',
    })

    def _build_training_index(self) -> Dict[str, List]:
        """
        Build keyword index for fast training data lookup.
        v26.0 QUANTUM UPGRADE: TF-IDF weighted index with document frequency tracking.
        - Computes IDF (inverse document frequency) for all terms
        - Stores term→entries mapping for O(1) lookup
        - Pre-computes document norms for cosine similarity
        """
        index = {}
        doc_freq = {}  # term → count of docs containing term
        doc_terms = {}  # doc_idx → set of terms (for IDF computation)
        N = len(self.training_data)

        # Pass 1: Build inverted index + count document frequencies
        for doc_idx, entry in enumerate(self.training_data):
            prompt = entry.get('prompt', '').lower()
            completion = entry.get('completion', '').lower()
            full_text = prompt + ' ' + completion

            # Extract and clean terms
            terms_in_doc = set()
            for word in full_text.split():
                term = ''.join(c for c in word if c.isalnum())
                if len(term) > 2 and term not in self._TRAINING_SEARCH_STOP:
                    terms_in_doc.add(term)
                    if term not in index:
                        index[term] = []
                    index[term].append(entry)
                    # Cap per-term entries to prevent memory bloat
                    if len(index[term]) > 50:
                        index[term] = index[term][-50:]

            doc_terms[doc_idx] = terms_in_doc
            for t in terms_in_doc:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        # Pass 2: Compute IDF weights — log(N / df) with smoothing
        idf = {}
        for term, df in doc_freq.items():
            idf[term] = math.log((N + 1) / (df + 1)) + 1.0  # Smoothed IDF

        # Store IDF weights and doc count for TF-IDF scoring
        self._idf_weights = idf
        self._training_doc_count = N
        self._doc_freq = doc_freq

        return index

    def _search_training_data(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search training data for relevant entries.
        v26.0 QUANTUM UPGRADE: TF-IDF + BM25 hybrid ranking with cosine similarity.
        - BM25 term frequency saturation (k1=1.5, b=0.75)
        - IDF-weighted term importance
        - Prompt-boost: matches in prompts score 2x
        - Length normalization prevents long-doc bias
        - Phrase proximity bonus for multi-word matches
        """
        self._ensure_training_index()
        query_lower = query.lower()
        # Filter stop words and extract query terms
        query_terms = []
        for w in query_lower.split():
            cleaned = ''.join(c for c in w if c.isalnum())
            if len(cleaned) > 2 and cleaned not in self._TRAINING_SEARCH_STOP:
                query_terms.append(cleaned)
        query_terms = query_terms[:15]  # Cap query terms (was 8)

        if not query_terms:
            return []

        # BM25 parameters
        k1 = 1.5   # Term frequency saturation
        b = 0.75    # Length normalization strength
        avg_dl = 50  # Approximate average document length

        # Collect candidate entries from inverted index
        candidates = {}  # entry_id → (entry, score)
        seen_prompts = set()

        idf = getattr(self, '_idf_weights', {})

        for term in query_terms:
            term_idf = idf.get(term, 1.0)
            if term in self.training_index:
                for entry in self.training_index[term][:100]:  # (was 60)
                    prompt = entry.get('prompt', '')
                    prompt_key = prompt[:60]
                    if prompt_key in seen_prompts:
                        # Update score for already-seen entry
                        if prompt_key in candidates:
                            old_entry, old_score = candidates[prompt_key]
                            candidates[prompt_key] = (old_entry, old_score + term_idf * 0.3)
                        continue
                    seen_prompts.add(prompt_key)

                    # Compute BM25-like score
                    prompt_lower = prompt.lower()
                    completion = entry.get('completion', '')
                    completion_lower = completion.lower()
                    full_text = prompt_lower + ' ' + completion_lower
                    doc_len = len(full_text.split())

                    score = 0.0
                    prompt_matches = 0
                    completion_matches = 0

                    for qt in query_terms:
                        qt_idf = idf.get(qt, 1.0)

                        # Count term frequency in full document
                        tf_full = full_text.count(qt)
                        if tf_full > 0:
                            # BM25 TF saturation
                            tf_saturated = (tf_full * (k1 + 1)) / (tf_full + k1 * (1 - b + b * doc_len / avg_dl))
                            score += qt_idf * tf_saturated

                        # Prompt-level bonus (2x weight — prompt matches are more relevant)
                        tf_prompt = prompt_lower.count(qt)
                        if tf_prompt > 0:
                            score += qt_idf * 0.5
                            prompt_matches += 1

                        # Track completion matches
                        if qt in completion_lower:
                            completion_matches += 1

                    # Phrase proximity bonus: if query terms appear near each other
                    if len(query_terms) >= 2:
                        query_phrase = ' '.join(query_terms[:3])
                        if query_phrase in full_text:
                            score *= 1.5  # Exact phrase match bonus

                    # Coverage bonus: what fraction of query terms matched?
                    total_matches = prompt_matches + completion_matches
                    coverage = total_matches / max(1, len(query_terms))
                    score *= (1.0 + coverage * 0.3)

                    # Content quality bonus: prefer entries with substantial completions
                    if len(completion) > 100:
                        score *= 1.1
                    elif len(completion) < 20:
                        score *= 0.5

                    if score > 0:
                        candidates[prompt_key] = (entry, score)

        # Sort by BM25 score descending, return top N
        ranked = sorted(candidates.values(), key=lambda x: x[1], reverse=True)

        # ═══════════════════════════════════════════════════════════════════
        # v27.2 NOISE DAMPENER PASS — purify signal before returning
        # ═══════════════════════════════════════════════════════════════════
        dampened = self._apply_noise_dampeners(ranked[:max_results], query_terms)
        return [entry for entry, _score in dampened]

    def _search_all_knowledge(self, query: str, max_results: int = 100) -> List[str]:
        """Deep search all JSON knowledge for relevant content. (Unlimited Mode: max_results=100)"""
        self._ensure_json_knowledge()
        query_lower = query.lower()
        query_words = set(w for w in query_lower.split() if len(w) > 2)
        results = []

        if not hasattr(self, '_all_json_knowledge'):
            self._all_json_knowledge = self._load_all_json_knowledge()

        def search_recursive(obj, path=""):
            """Recursively search nested structures."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_lower = str(key).lower()
                    # Check if key matches any query word
                    if any(w in key_lower for w in query_words):
                        content = f"{path}/{key}: {str(value)[:1500]}"
                        matches = sum(1 for w in query_words if w in content.lower())
                        results.append((matches, content))
                    # Recurse
                    search_recursive(value, f"{path}/{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:500]):  # Expanded list iteration (was 100)
                    search_recursive(item, f"{path}[{i}]")
            elif isinstance(obj, str) and len(obj) > 20:
                obj_lower = obj.lower()
                if any(w in obj_lower for w in query_words):
                    matches = sum(1 for w in query_words if w in obj_lower)
                    results.append((matches, f"{path}: {obj[:1500]}"))

        for source_name, data in self._all_json_knowledge.items():
            search_recursive(data, source_name)

        # Sort by relevance and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:max_results] if r[0] >= 2]

    @staticmethod
    def _compute_text_entropy(text: str) -> float:
        """
        Compute Shannon entropy of word distribution in text.

        H = -Σ p(w) * log2(p(w))

        High entropy → diverse vocabulary → likely informative content.
        Low entropy → repetitive/boilerplate → noise candidate.
        """
        if not text or len(text) < 10:
            return 0.0

        words = text.lower().split()
        if not words:
            return 0.0

        # Frequency distribution
        freq: Dict[str, int] = {}
        for w in words:
            cleaned = ''.join(c for c in w if c.isalnum())
            if len(cleaned) > 1:
                freq[cleaned] = freq.get(cleaned, 0) + 1

        total = sum(freq.values())
        if total == 0:
            return 0.0

        # Shannon entropy
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _apply_gqa_noise_dampeners(self, results: list, query: str) -> list:
        """
        v27.3 Higher Logic Noise dampener for GQA search results.

        Applies base dampening + higher logic layers adapted for the heterogeneous
        GQA result format. Adds semantic coherence, spectral analysis, entanglement
        resonance, and Grover amplification on top of v27.2 base filters.
        """
        if not results:
            return results

        query_lower = query.lower()
        query_terms = [
            ''.join(c for c in w if c.isalnum())
            for w in query_lower.split()
            if len(w) > 2
        ]
        query_terms = [t for t in query_terms if t and t not in self._TRAINING_SEARCH_STOP][:8]

        if not query_terms:
            return results

        query_concept_set = set(query_terms)
        effective_score_floor = self._hl_adaptive_score_floor()

        purified = []
        seen_hashes: Set[str] = set()
        max_score = max(
            (r.get('_gqa_score', r.get('score', 0.5)) for r in results
             if isinstance(r.get('_gqa_score', r.get('score', 0.5)), (int, float))),
            default=1.0,
        )
        if max_score <= 0:
            max_score = 1.0

        for rank_idx, result in enumerate(results):
            content = str(
                result.get('completion',
                    result.get('content',
                        result.get('response', '')))
            ).lower()
            source = result.get('_gqa_source', 'unknown')

            # ── Base: Entropy filter ──
            entropy = self._compute_text_entropy(content)
            if entropy < NOISE_DAMPENER_ENTROPY_MIN and len(content) > 20:
                continue

            # ── Base: Coverage gate ──
            matched = sum(1 for qt in query_terms if qt in content)
            coverage = matched / max(len(query_terms), 1)
            if coverage < NOISE_DAMPENER_COVERAGE_MIN and len(content) > 50:
                continue

            # ── Base: Near-duplicate suppression ──
            content_hash = hashlib.md5(content[:200].encode()).hexdigest()[:16]
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # ── Base: Source quality weight ──
            source_weight = NOISE_DAMPENER_SOURCE_WEIGHTS.get(source, 0.85)
            gqa_score = result.get('_gqa_score', result.get('score', 0.5))
            if isinstance(gqa_score, (int, float)):
                gqa_score *= source_weight
            else:
                gqa_score = 0.5

            # ── Base: φ-decay for tail results ──
            if rank_idx >= NOISE_DAMPENER_PHI_DECAY_START:
                decay_exp = rank_idx - NOISE_DAMPENER_PHI_DECAY_START
                phi_decay = 1.0 / (NOISE_DAMPENER_PHI_DECAY_RATE ** decay_exp)
                gqa_score *= phi_decay

            # ═══ Higher Logic: Semantic coherence ═══
            result_terms = set(
                ''.join(c for c in w if c.isalnum())
                for w in content.split() if len(w) > 2
            )
            result_concepts = {
                t for t in result_terms
                if t not in self._TRAINING_SEARCH_STOP and len(t) > 2
            }
            coherence = self._hl_concept_cosine(query_concept_set, result_concepts)
            if coherence < HL_SEMANTIC_COHERENCE_MIN:
                continue
            gqa_score *= (0.6 + 0.4 * min(coherence, 1.0))

            # ═══ Higher Logic: Spectral noise detection ═══
            if HL_SPECTRAL_ENABLED and len(content) > 50:
                spectral_noise = self._hl_spectral_noise_ratio(content)
                if spectral_noise > HL_SPECTRAL_NOISE_CUTOFF:
                    gqa_score *= (1.0 - spectral_noise) * 1.5

            # ═══ Higher Logic: Entanglement resonance bonus ═══
            ent_bonus = self._hl_entanglement_resonance(query_terms, result_concepts)
            gqa_score *= ent_bonus

            # ═══ Higher Logic: Grover amplification ═══
            relative_amp = gqa_score / max_score
            if relative_amp >= HL_GROVER_AMPLITUDE_FLOOR:
                grover_factor = 1.0 + (HL_GROVER_AMPLIFICATION - 1.0) * (
                    (relative_amp - HL_GROVER_AMPLITUDE_FLOOR)
                    / (1.0 - HL_GROVER_AMPLITUDE_FLOOR + 1e-9)
                )
                gqa_score *= grover_factor

            # ═══ Higher Logic: GOD_CODE resonance ═══
            word_count = len(content.split())
            res_bonus = self._hl_godcode_resonance(entropy, word_count)
            gqa_score *= (1.0 + HL_RESONANCE_ALIGNMENT_WEIGHT * res_bonus)

            result['_gqa_score'] = gqa_score
            purified.append(result)

        # Re-sort by higher-logic dampened GQA score
        purified.sort(
            key=lambda x: x.get('_gqa_score', x.get('score', 0)),
            reverse=True,
        )
        return purified

    def _search_chat_conversations(self, query: str, max_results: int = 100) -> List[str]:
        """Search chat conversations for relevant responses. (Unlimited Mode: max_results=100)"""
        query_lower = query.lower()
        query_words = set(w for w in query_lower.split() if len(w) > 3)
        results = []

        for conv in self.chat_conversations:
            messages = conv.get('messages', [])
            conv_text = ' '.join(m.get('content', '') for m in messages).lower()

            # Score by word matches
            matches = sum(1 for w in query_words if w in conv_text)
            if matches >= 2:
                # Find the assistant response
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if len(content) > 50:
                            results.append((matches, content))
                            break

        # Sort by relevance and return top
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:max_results]]

    def _search_knowledge_manifold(self, query: str) -> Optional[str]:
        """Search knowledge manifold for matching patterns."""
        query_lower = query.lower()
        patterns = self.knowledge_manifold.get('patterns', {})

        for pattern_name, pattern_data in patterns.items():
            if pattern_name.lower() in query_lower or query_lower in pattern_name.lower():
                if isinstance(pattern_data, dict):
                    return f"Pattern: {pattern_name}\n{str(pattern_data)[:1500]}"
                elif isinstance(pattern_data, str):
                    return f"Pattern: {pattern_name}\n{pattern_data[:1500]}"

        return None

    def _search_knowledge_vault(self, query: str) -> Optional[str]:
        """Search knowledge vault for proofs and documentation."""
        query_lower = query.lower()

        # Search proofs
        proofs = self.knowledge_vault.get('proofs', [])
        for proof in proofs:
            if isinstance(proof, dict):
                proof_text = str(proof).lower()
                if any(w in proof_text for w in query_lower.split() if len(w) > 3):
                    return f"From Knowledge Vault:\n{str(proof)[:1500]}"

        # Search documentation
        docs = self.knowledge_vault.get('documentation', {})
        for doc_name, doc_content in docs.items():
            if doc_name.lower() in query_lower or any(w in doc_name.lower() for w in query_lower.split()):
                return f"Documentation: {doc_name}\n{str(doc_content)[:1500]}"

        return None
