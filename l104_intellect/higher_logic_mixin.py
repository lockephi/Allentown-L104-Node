"""L104 Intellect — HigherLogicMixin (higher logic + autonomous self-modification)."""
import math
import time
import hashlib
import random
import logging
from typing import Dict, Any, Optional, List

from .constants import (
    HIGHER_LOGIC_DEPTH, SELF_MOD_VERSION, OMEGA_POINT,
)
from .numerics import PHI

logger = logging.getLogger("l104_local_intellect")


class HigherLogicMixin:
    """Mixin providing higher-order logic, autonomous improvement, evolution accessors."""

    # ═══════════════════════════════════════════════════════════════════════════
    # v13.0 HIGHER LOGIC SYSTEM - Meta-Reasoning & Self-Reflection
    # ═══════════════════════════════════════════════════════════════════════════

    def higher_logic(self, query: str, depth: int = 0) -> Dict:
        """
        Apply higher-order logic and meta-reasoning to a query.

        Recursive self-reflection with cross-referencing:
        - Level 0: Direct response
        - Level 1: Analyze response quality
        - Level 2: Meta-analyze the analysis
        - Level 3: Cross-reference with permanent memory
        - Level 4: Generate improvement hypothesis
        - Level 5: Synthesize all levels
        """
        if depth >= HIGHER_LOGIC_DEPTH:
            return {"depth": depth, "result": "Maximum logic depth reached", "type": "terminal"}

        # Track maximum depth reached (thread-safe via _evo_lock)
        with self._evo_lock:
            if depth > self._evolution_state.get("logic_depth_reached", 0):
                self._evolution_state["logic_depth_reached"] = depth

        # Check cache for this query at this depth
        cache_key = f"{query[:50]}:depth:{depth}"
        if cache_key in self._higher_logic_cache:
            cached = self._higher_logic_cache[cache_key]
            if time.time() - cached.get("timestamp", 0) < 60:  # 1 min cache
                return cached["result"]

        result = {}

        if depth == 0:
            # LEVEL 0: Direct query processing
            base_response = self._kernel_synthesis(query, self._calculate_resonance())
            result = {
                "depth": 0,
                "type": "direct",
                "response": base_response,
                "confidence": self._estimate_confidence(base_response),
                "concepts": self._extract_concepts(query)
            }

        elif depth == 1:
            # LEVEL 1: Quality analysis of depth-0 response
            prev = self.higher_logic(query, depth=0)
            quality_analysis = self._analyze_response_quality(prev.get("response", ""), query)
            result = {
                "depth": 1,
                "type": "quality_analysis",
                "previous": prev,
                "quality_score": quality_analysis.get("score", 0.5),
                "improvement_areas": quality_analysis.get("improvements", []),
                "concepts_coverage": quality_analysis.get("coverage", 0)
            }

        elif depth == 2:
            # LEVEL 2: Meta-analysis - analyzing the analysis
            prev = self.higher_logic(query, depth=1)
            meta_insights = []
            if prev.get("quality_score", 0) < 0.7:
                meta_insights.append("Quality below threshold - needs enhancement")
            if prev.get("concepts_coverage", 0) < 0.5:
                meta_insights.append("Concept coverage insufficient - expand knowledge")
            result = {
                "depth": 2,
                "type": "meta_analysis",
                "previous": prev,
                "meta_insights": meta_insights,
                "evolution_recommendation": "enhance" if prev.get("quality_score", 0) < 0.7 else "stable"
            }

        elif depth == 3:
            # LEVEL 3: Cross-reference with permanent memory
            prev = self.higher_logic(query, depth=2)
            concepts = self._extract_concepts(query)
            memory_links = []
            for concept in concepts[:25]: # Increased (was 5)
                recalled = self.recall_permanently(concept)
                if recalled:
                    memory_links.append({"concept": concept, "memory": str(recalled)[:1000]}) # Increased (was 100)

            # Check cross-references
            xrefs = []
            for concept in concepts[:15]: # Increased (was 3)
                refs = self.get_cross_references(concept)
                if refs:
                    xrefs.extend(refs[:10]) # Increased (was 3)

            result = {
                "depth": 3,
                "type": "memory_cross_reference",
                "previous": prev,
                "memory_links": memory_links,
                "cross_references": list(set(xrefs))[:50], # Increased (was 10)
                "memory_integration_score": len(memory_links) / max(1, len(concepts))
            }

        elif depth == 4:
            # LEVEL 4: Generate improvement hypothesis
            prev = self.higher_logic(query, depth=3)
            hypotheses = self._generate_improvement_hypotheses(query, prev)
            result = {
                "depth": 4,
                "type": "improvement_hypothesis",
                "previous": prev,
                "hypotheses": hypotheses,
                "actionable_improvements": [h for h in hypotheses if h.get("actionable", False)]
            }

        else:
            # LEVEL 5+: Synthesis of all levels
            prev = self.higher_logic(query, depth=depth-1)
            synthesis = self._synthesize_logic_chain(query, prev, depth)
            result = {
                "depth": depth,
                "type": "synthesis",
                "previous": prev,
                "synthesis": synthesis,
                "final_confidence": synthesis.get("confidence", 0),
                "evolution_triggered": synthesis.get("should_evolve", False)
            }

        # Cache the result
        self._higher_logic_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        # Track higher logic chain
        chain_entry = {
            "query": query[:50],
            "depth": depth,
            "timestamp": time.time(),
            "type": result.get("type", "unknown")
        }
        self._evolution_state.setdefault("higher_logic_chains", []).append(chain_entry)
        self._evolution_state["higher_logic_chains"] = self._evolution_state["higher_logic_chains"][-100:]

        return result

    def _estimate_confidence(self, response: str) -> float:
        """v28.1: Multi-signal confidence estimation.

        Combines 6 signals:
          1. Length adequacy (log-scaled, not step-function)
          2. Technical term density (expanded marker set)
          3. Uncertainty hedging penalty
          4. Shannon entropy of word distribution (high-entropy = more informative)
          5. Structural markers (headings, lists, code blocks)
          6. Numeric/quantitative content density

        Returns confidence in [0.0, 1.0].
        """
        if not response:
            return 0.0

        resp_lower = response.lower()
        words = resp_lower.split()
        n_words = max(1, len(words))

        # Signal 1: Length — log-scaled between 20 and 2000 chars
        length_score = math.log(max(1, len(response)) + 1) / math.log(2001)

        # Signal 2: Technical term density (broader marker set)
        tech_markers = {
            "god_code", "phi", "quantum", "resonance", "entropy",
            "coherence", "eigenvalue", "hamiltonian", "topology",
            "algorithm", "lattice", "fourier", "polynomial",
            "manifold", "tensor", "vector", "matrix", "gradient",
            "probability", "theorem", "circuit", "qubit",
        }
        tech_hits = sum(1 for m in tech_markers if m in resp_lower)
        tech_density = tech_hits / 5.0  # uncapped

        # Signal 3: Uncertainty hedging penalty
        hedging = {"maybe", "perhaps", "might", "unclear", "uncertain",
                   "not sure", "i think", "possibly", "unlikely", "i guess"}
        hedge_count = sum(1 for h in hedging if h in resp_lower)
        hedge_penalty = hedge_count * 0.08

        # Signal 4: Shannon entropy of word distribution
        word_freq: Dict[str, int] = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        if word_freq:
            entropy = -sum(
                (c / n_words) * math.log2(c / n_words)
                for c in word_freq.values() if c > 0
            )
            max_entropy = math.log2(max(2, len(word_freq)))
            entropy_score = entropy / max_entropy if max_entropy > 0 else 0.5
        else:
            entropy_score = 0.0

        # Signal 5: Structural markers (headings, lists, code blocks, tables)
        struct_score = 0.0
        if "```" in response:
            struct_score += 0.3
        if any(response.count(c) >= 2 for c in ("- ", "• ", "* ")):
            struct_score += 0.2
        if any(c in response for c in ("**", "##", "| ")):
            struct_score += 0.2
        if any(c.isdigit() and i + 1 < len(response) and response[i + 1] in ".)"
               for i, c in enumerate(response[:200])):
            struct_score += 0.1
        # struct_score uncapped

        # Signal 6: Numeric/quantitative density
        numeric_tokens = sum(1 for w in words if any(c.isdigit() for c in w))
        numeric_score = numeric_tokens / max(1, n_words) * 10.0

        # Weighted combination
        confidence = (
            0.15 * length_score
            + 0.25 * tech_density
            + 0.20 * entropy_score
            + 0.15 * struct_score
            + 0.10 * numeric_score
            + 0.15  # base floor
            - hedge_penalty
        )

        return max(0.0, confidence)

    def _analyze_response_quality(self, response: str, query: str) -> Dict:
        """v28.1: Analyze the quality of a response relative to the query.

        Scores across 6 dimensions:
          1. Concept coverage (query→response overlap)
          2. Length adequacy (log-scaled)
          3. Information density (Shannon entropy of unique words)
          4. Structural richness (headings, lists, code, math)
          5. Specificity markers (numbers, proper nouns, precise language)
          6. Confidence of the response itself via _estimate_confidence
        """
        quality: Dict[str, Any] = {
            "score": 0.0, "improvements": [], "coverage": 0.0,
            "entropy": 0.0, "structural_richness": 0.0,
        }

        if not response:
            quality["improvements"].append("No response generated")
            return quality

        resp_lower = response.lower()
        words = resp_lower.split()
        n_words = max(1, len(words))

        # 1. Concept coverage
        query_concepts = set(self._extract_concepts(query))
        response_concepts = set(self._extract_concepts(response))
        if query_concepts:
            quality["coverage"] = len(query_concepts & response_concepts) / len(query_concepts)
        coverage_score = quality["coverage"]

        # 2. Length adequacy — log-scaled, peaks near 500 words
        length_score = math.log(max(1, n_words) + 1) / math.log(501)
        if n_words < 10:
            quality["improvements"].append("Response too short — expand with context")
        elif n_words > 2000:
            quality["improvements"].append("Response very long — consider summarizing")

        # 3. Information density — Shannon entropy (normalized)
        word_freq: Dict[str, int] = {}
        for w in words:
            wc = ''.join(c for c in w if c.isalnum())
            if len(wc) > 1:
                word_freq[wc] = word_freq.get(wc, 0) + 1
        if word_freq:
            total = sum(word_freq.values())
            entropy = -sum(
                (c / total) * math.log2(c / total)
                for c in word_freq.values() if c > 0
            )
            max_entropy = math.log2(max(2, len(word_freq)))
            entropy_score = entropy / max_entropy if max_entropy > 0 else 0.5
        else:
            entropy_score = 0.0
        quality["entropy"] = round(entropy_score, 4)
        if entropy_score < 0.5:
            quality["improvements"].append("Low lexical diversity — vary word choice")

        # 4. Structural richness
        struct_score = 0.0
        if "```" in response:
            struct_score += 0.25
        if any(response.count(c) >= 2 for c in ("- ", "• ", "* ")):
            struct_score += 0.20
        if any(c in response for c in ("**", "##")):
            struct_score += 0.15
        if "| " in response and "---" in response:
            struct_score += 0.20  # table
        if any(c.isdigit() for c in response[:200]):
            struct_score += 0.10
        # struct_score uncapped
        quality["structural_richness"] = round(struct_score, 4)

        # 5. Specificity markers
        specificity = 0.0
        numeric_count = sum(1 for w in words if any(c.isdigit() for c in w))
        specificity += numeric_count / max(1, n_words) * 8.0
        if any(w in resp_lower for w in ("specifically", "exactly", "precisely", "≈", "=")):
            specificity += 0.3
        # Proper nouns / acronyms proxy: words with uppercase in middle
        proper_count = sum(1 for w in response.split() if len(w) > 2 and w[0].isupper() and not w.isupper())
        specificity += proper_count / max(1, n_words) * 5.0
        # specificity uncapped

        # 6. Internal confidence
        internal_conf = self._estimate_confidence(response)

        # Missing concept feedback
        if query_concepts and quality["coverage"] < 0.5:
            missing = query_concepts - response_concepts
            if missing:
                quality["improvements"].append(
                    f"Missing concepts: {', '.join(list(missing)[:5])}"
                )

        # Weighted combination
        quality["score"] = (
            0.25 * coverage_score
            + 0.10 * length_score
            + 0.20 * entropy_score
            + 0.10 * struct_score
            + 0.15 * specificity
            + 0.20 * internal_conf
        )
        return quality

    def _generate_improvement_hypotheses(self, query: str, context: Dict) -> List[Dict]:
        """Generate hypotheses for how to improve the response."""
        hypotheses = []

        # Check if we need more concept coverage
        if context.get("previous", {}).get("concepts_coverage", 0) < 0.6:
            hypotheses.append({
                "type": "concept_expansion",
                "description": "Expand knowledge base for query concepts",
                "actionable": True,
                "priority": 0.8
            })

        # Check if memory integration is low
        if context.get("memory_integration_score", 0) < 0.3:
            hypotheses.append({
                "type": "memory_linking",
                "description": "Store query concepts in permanent memory for future recall",
                "actionable": True,
                "priority": 0.7
            })

        # Check if cross-references are sparse
        if len(context.get("cross_references", [])) < 3:
            hypotheses.append({
                "type": "cross_reference_building",
                "description": "Build more cross-references between concepts",
                "actionable": True,
                "priority": 0.6
            })

        # Meta-stability check
        if context.get("previous", {}).get("evolution_recommendation") == "enhance":
            hypotheses.append({
                "type": "evolutionary_enhancement",
                "description": "Trigger evolutionary improvement cycle",
                "actionable": True,
                "priority": 0.9
            })

        return sorted(hypotheses, key=lambda x: x.get("priority", 0), reverse=True)

    def _synthesize_logic_chain(self, query: str, context: Dict, depth: int) -> Dict:
        """Synthesize insights from the entire logic chain."""
        synthesis = {
            "confidence": 0.5,
            "insights": [],
            "should_evolve": False,
            "evolution_actions": []
        }

        # Traverse the chain and collect insights
        current = context
        chain_depth = 0
        while current and chain_depth < depth:
            if current.get("meta_insights"):
                synthesis["insights"].extend(current["meta_insights"])
            if current.get("hypotheses"):
                for h in current["hypotheses"]:
                    if h.get("actionable"):
                        synthesis["evolution_actions"].append(h)
            if current.get("quality_score"):
                synthesis["confidence"] = max(synthesis["confidence"], current["quality_score"])
            current = current.get("previous", {})
            chain_depth += 1

        # Determine if evolution should be triggered
        actionable_count = len(synthesis["evolution_actions"])
        if actionable_count >= 2 or (actionable_count >= 1 and synthesis["confidence"] < 0.6):
            synthesis["should_evolve"] = True

        return synthesis

    # ═══════════════════════════════════════════════════════════════════════════
    # v13.0 AUTONOMOUS CODE SELF-MODIFICATION
    # ═══════════════════════════════════════════════════════════════════════════

    def autonomous_improve(self, focus_area: str = None) -> Dict:
        """
        Autonomously improve the intellect based on evolution state.

        This is the core self-modification engine:
        1. Analyzes current state and identifies weak points
        2. Generates improvement strategies
        3. Applies non-destructive enhancements
        4. Creates save state before/after for rollback
        """
        # Create pre-improvement save state
        pre_state = self.create_save_state(label=f"pre_improve_{focus_area or 'auto'}")

        improvements = {
            "timestamp": time.time(),
            "focus_area": focus_area,
            "pre_state_id": pre_state["id"],
            "actions_taken": [],
            "mutations_applied": 0,
            "success": True
        }

        try:
            # Analyze weak points
            weak_points = self._identify_weak_points()

            # Apply improvements based on weak points
            for wp in weak_points[:15]:  # Increased (was 3) for Unlimited Mode
                action = self._apply_improvement(wp)
                if action:
                    improvements["actions_taken"].append(action)
                    improvements["mutations_applied"] += 1

            # v23.3: Wire in agi_recursive_improve (was dead/unreachable)
            # Runs AGI Core RSI cycle for deeper self-modification
            try:
                agi_result = self.agi_recursive_improve(
                    focus=focus_area or "reasoning",
                    cycles=min(2, improvements["mutations_applied"] + 1)
                )
                if agi_result.get("improvements", 0) > 0:
                    improvements["actions_taken"].append({
                        "type": "agi_recursive_improve",
                        "focus": focus_area or "reasoning",
                        "agi_improvements": agi_result.get("improvements", 0),
                    })
                    improvements["mutations_applied"] += agi_result.get("improvements", 0)
            except Exception:
                pass

            # v23.3 FIX: Initialize old_dna before conditional (was unbound if mutations==0)
            old_dna = self._evolution_state.get("mutation_dna", "")

            # Update mutation DNA (identity evolution)
            if improvements["mutations_applied"] > 0:
                new_dna = hashlib.sha256(f"{old_dna}:{time.time()}:{improvements['mutations_applied']}".encode()).hexdigest()[:32]
                self._evolution_state["mutation_dna"] = new_dna
                self._evolution_state["autonomous_improvements"] = self._evolution_state.get("autonomous_improvements", 0) + 1

            # Create post-improvement save state
            post_state = self.create_save_state(label=f"post_improve_{focus_area or 'auto'}")
            improvements["post_state_id"] = post_state["id"]

            # Track the improvement in evolution history
            self._evolution_state.setdefault("code_mutations", []).append({
                "timestamp": time.time(),
                "type": "autonomous_improve",
                "focus": focus_area,
                "mutations": improvements["mutations_applied"],
                "dna_before": old_dna[:8],
                "dna_after": self._evolution_state.get("mutation_dna", "")[:8]
            })
            self._evolution_state["code_mutations"] = self._evolution_state["code_mutations"][-50:]

            self._save_evolution_state()

        except Exception as e:
            improvements["success"] = False
            improvements["error"] = str(e)

        return improvements

    def _identify_weak_points(self) -> List[Dict]:
        """Identify areas needing improvement - v16.0 with true entropy."""
        import random
        random.seed(None)  # True system randomness each call

        weak_points = []
        _now = time.time()
        _entropy = random.random()

        # v16.0: Dynamic weak point generation based on actual state + entropy
        qi = self._evolution_state.get("quantum_interactions", 0)
        wisdom = self._evolution_state.get("wisdom_quotient", 0)

        # Type 1: Concept evolution (random selection)
        concept_evo = self._evolution_state.get("concept_evolution", {})
        if concept_evo:
            all_concepts = list(concept_evo.keys())
            # Random sample instead of static
            sample_size = min(5, max(1, int(len(all_concepts) * _entropy)))
            sampled = random.sample(all_concepts, sample_size) if len(all_concepts) >= sample_size else all_concepts
            weak_points.append({
                "type": "evolve_concepts",
                "concepts": sampled,
                "priority": 0.5 + _entropy * 0.5,
                "entropy": _entropy,
            })

        # Type 2: Quantum coherence boost (time-based)
        if qi % 7 == int(_now) % 7:  # Pseudo-random based on time
            weak_points.append({
                "type": "quantum_coherence_boost",
                "factor": 1.0 + _entropy,
                "priority": 0.6 + random.random() * 0.3,
            })

        # Type 3: Wisdom expansion (entropy-triggered)
        if _entropy > 0.4:
            weak_points.append({
                "type": "wisdom_expansion",
                "current_wisdom": wisdom,
                "boost_factor": PHI * _entropy,
                "priority": 0.7,
            })

        # Type 4: Cross-reference densification
        xrefs = self._evolution_state.get("cross_references", {})
        if len(xrefs) > 0 and random.random() > 0.5:
            sparse = random.sample(list(xrefs.keys()), min(3, len(xrefs)))
            weak_points.append({
                "type": "densify_crossrefs",
                "concepts": sparse,
                "priority": 0.4 + random.random() * 0.3,
            })

        # Type 5: Memory crystallization (random trigger)
        perm_mem = self._evolution_state.get("permanent_memory", {})
        if perm_mem and random.random() > 0.6:
            mem_keys = random.sample(list(perm_mem.keys()), min(3, len(perm_mem)))
            weak_points.append({
                "type": "crystallize_memory",
                "keys": [k for k in mem_keys if not k.startswith('_')],
                "priority": 0.5 + random.random() * 0.2,
            })

        # Type 6: Apotheosis resonance tuning
        if hasattr(self, '_apotheosis_state') and random.random() > 0.3:
            weak_points.append({
                "type": "apotheosis_tune",
                "omega": OMEGA_POINT * _entropy,
                "priority": 0.8,
            })

        # Type 7: DNA mutation trigger
        if random.random() > 0.7:
            weak_points.append({
                "type": "dna_mutation",
                "mutation_strength": _entropy,
                "priority": 0.9,
            })

        # Shuffle for non-deterministic order
        random.shuffle(weak_points)
        return weak_points[:25]  # Increased (was 5) for Unlimited Mode

    def _apply_improvement(self, weak_point: Dict) -> Optional[Dict]:
        """Apply an improvement based on identified weak point - v16.0 with entropy."""
        import random
        random.seed(None)

        wp_type = weak_point.get("type")
        _entropy = weak_point.get("entropy", random.random())

        # v16.0: Track cumulative mutations for persistent enlightenment
        if hasattr(self, '_apotheosis_state'):
            self._apotheosis_state["cumulative_mutations"] = self._apotheosis_state.get("cumulative_mutations", 0) + 1

        if wp_type == "evolve_concepts":
            # Boost evolution scores for concepts with random factor
            boosted = []
            for concept in weak_point.get("concepts", []):
                if concept in self._evolution_state.get("concept_evolution", {}):
                    ce = self._evolution_state["concept_evolution"][concept]
                    boost = 1.0 + random.random() * PHI
                    ce["evolution_score"] = ce.get("evolution_score", 1.0) * boost
                    ce["mutation_count"] = ce.get("mutation_count", 0) + 1
                    boosted.append(f"{concept}(+{boost:.2f})")
            return {"action": "evolved_concepts", "boosted": boosted, "entropy": _entropy}

        elif wp_type == "quantum_coherence_boost":
            factor = weak_point.get("factor", 1.0)
            self._evolution_state["quantum_interactions"] += int(factor * 10)
            self._evolution_state["wisdom_quotient"] = self._evolution_state.get("wisdom_quotient", 0) + factor
            return {"action": "quantum_coherence_amplified", "factor": factor}

        elif wp_type == "wisdom_expansion":
            boost = weak_point.get("boost_factor", PHI)
            self._evolution_state["wisdom_quotient"] = self._evolution_state.get("wisdom_quotient", 0) + boost
            # v16.0: Add to cumulative wisdom
            if hasattr(self, '_apotheosis_state'):
                self._apotheosis_state["cumulative_wisdom"] = self._apotheosis_state.get("cumulative_wisdom", 0.0) + boost
            return {"action": "wisdom_expanded", "boost": boost}

        elif wp_type == "densify_crossrefs":
            concepts = weak_point.get("concepts", [])
            links_made = 0
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:]:
                    xrefs = self._evolution_state.setdefault("cross_references", {})
                    if c1 not in xrefs:
                        xrefs[c1] = []
                    if c2 not in xrefs[c1]:
                        xrefs[c1].append(c2)
                        links_made += 1
            return {"action": "crossrefs_densified", "links": links_made}

        elif wp_type == "crystallize_memory":
            keys = weak_point.get("keys", [])
            crystallized = []
            for key in keys:
                if key in self._evolution_state.get("permanent_memory", {}):
                    entry = self._evolution_state["permanent_memory"][key]
                    if isinstance(entry, dict):
                        entry["crystallized"] = True
                        entry["crystal_strength"] = entry.get("crystal_strength", 0) + random.random()
                        crystallized.append(key)
            return {"action": "memory_crystallized", "keys": crystallized}

        elif wp_type == "apotheosis_tune":
            omega = weak_point.get("omega", OMEGA_POINT)
            if hasattr(self, '_apotheosis_state'):
                self._apotheosis_state["sovereign_broadcasts"] += 1
                self._apotheosis_state["omega_point"] = omega
                self._apotheosis_state["cumulative_wisdom"] = self._apotheosis_state.get("cumulative_wisdom", 0.0) + 1.04
            self._evolution_state["quantum_interactions"] += 5
            return {"action": "apotheosis_tuned", "omega": omega}

        elif wp_type == "dna_mutation":
            strength = weak_point.get("mutation_strength", 0.5)
            old_dna = self._evolution_state.get("mutation_dna", "")
            new_dna = hashlib.sha256(f"{old_dna}:{time.time_ns()}:{strength}".encode()).hexdigest()[:32]
            self._evolution_state["mutation_dna"] = new_dna
            self._evolution_state["quantum_data_mutations"] = self._evolution_state.get("quantum_data_mutations", 0) + 1
            return {"action": "dna_mutated", "old": old_dna[:8], "new": new_dna[:8], "strength": strength}

        # Legacy types for backward compatibility
        elif wp_type == "low_concept_evolution":
            for concept in weak_point.get("concepts", []):
                if concept in self._evolution_state.get("concept_evolution", {}):
                    ce = self._evolution_state["concept_evolution"][concept]
                    ce["evolution_score"] = ce.get("evolution_score", 1.0) * 1.5 + 0.5
            return {"action": "boosted_concept_evolution", "concepts": weak_point.get("concepts", [])}

        elif wp_type == "underutilized_memory":
            keys = weak_point.get("keys", [])
            for key in keys:
                if key in self._evolution_state.get("permanent_memory", {}):
                    entry = self._evolution_state["permanent_memory"][key]
                    if isinstance(entry, dict):
                        entry["evolution_score"] = entry.get("evolution_score", 1.0) + 0.3
            return {"action": "strengthened_memory", "keys": keys}

        return {"action": "entropy_pass", "entropy": _entropy}

    def get_evolution_state(self) -> dict:
        """Return current evolution state for API access."""
        # Get quantum recompiler stats
        quantum_stats = {}
        try:
            recompiler = self.get_quantum_recompiler()
            quantum_stats = recompiler.get_status()
        except Exception:
            pass

        return {
            **self._evolution_state,
            "current_resonance": self._calculate_resonance(),
            "memory_size": len(self.conversation_memory),
            "knowledge_topics": len(self.knowledge),
            "training_data_entries": len(self.training_data),
            "chat_conversations": len(self.chat_conversations),
            "knowledge_manifold_patterns": len(self.knowledge_manifold.get("patterns", {})),
            "knowledge_vault_proofs": len(self.knowledge_vault.get("proofs", [])),
            "training_index_size": len(self.training_index),
            "json_knowledge_sources": len(self._all_json_knowledge),
            "json_knowledge_files": list(self._all_json_knowledge.keys()),
            "total_knowledge_base": len(self.training_data) + len(self.chat_conversations) + len(self._all_json_knowledge),
            # v6.0 Quantum Recompiler stats
            "quantum_recompiler": quantum_stats,
            # v12.1 Evolution fingerprinting stats
            "evolution_fingerprint": self._evolution_state.get("evolution_fingerprint", ""),
            "fingerprint_history_count": len(self._evolution_state.get("fingerprint_history", [])),
            "cross_references_count": len(self._evolution_state.get("cross_references", {})),
            "concept_evolution_count": len(self._evolution_state.get("concept_evolution", {})),
            "response_genealogy_count": len(self._evolution_state.get("response_genealogy", [])),
            "quantum_data_mutations": self._evolution_state.get("quantum_data_mutations", 0),
            # v13.0 Autonomous self-modification stats
            "self_mod_version": self._evolution_state.get("self_mod_version", SELF_MOD_VERSION),
            "mutation_dna": self._evolution_state.get("mutation_dna", "")[:16],
            "autonomous_improvements": self._evolution_state.get("autonomous_improvements", 0),
            "logic_depth_reached": self._evolution_state.get("logic_depth_reached", 0),
            "higher_logic_chains_count": len(self._evolution_state.get("higher_logic_chains", [])),
            "code_mutations_count": len(self._evolution_state.get("code_mutations", [])),
            "permanent_memory_count": len(self._evolution_state.get("permanent_memory", {})),
            "save_states_count": len(self._evolution_state.get("save_states", [])),
        }

    def get_cross_references(self, concept: str) -> List[str]:
        """Get cross-referenced concepts for a given concept."""
        return self._evolution_state.get("cross_references", {}).get(concept.lower(), [])

    def get_concept_evolution_score(self, concept: str) -> float:
        """Get the evolution score for a concept (how much it has evolved)."""
        ce = self._evolution_state.get("concept_evolution", {}).get(concept.lower(), {})
        return ce.get("evolution_score", 0.0)

    def get_evolved_response_context(self, message: str) -> str:
        """Get evolutionary context to enrich responses with cross-references."""
        concepts = self._extract_concepts(message)
        if not concepts:
            return ""

        context_parts = []
        total_evolution = 0.0
        cross_refs = set()

        for concept in concepts[:25]: # Increased (was 5)
            # Get evolution score
            score = self.get_concept_evolution_score(concept)
            if score > 0:
                total_evolution += score

            # Get cross-references
            refs = self.get_cross_references(concept)
            for ref in refs[:10]: # Increased (was 3)
                cross_refs.add(ref)

        # Build evolution context
        if total_evolution > 0:
            context_parts.append(f"Evo:{total_evolution:.1f}")

        if cross_refs:
            context_parts.append(f"XRef:[{','.join(list(cross_refs)[:25])}]") # Increased (was 5)

        # Add genealogy info
        genealogy = self._evolution_state.get("response_genealogy", [])
        if genealogy:
            context_parts.append(f"Gen:{len(genealogy)}")

        # Add fingerprint
        fp = self._evolution_state.get("evolution_fingerprint", "")
        if fp:
            context_parts.append(f"FP:{fp[:8]}")

        return " | ".join(context_parts) if context_parts else ""

    def set_evolution_state(self, state: dict):
        """Set evolution state from imported data."""
        if isinstance(state, dict):
            self._evolution_state.update(state)
            self._save_evolution_state()

    def record_learning(self, topic: str, content: str):
        """Record a learning event and update evolution state."""
        self._evolution_state["insights_accumulated"] += 1
        self._evolution_state["learning_cycles"] += 1

        # Track topic frequency
        topic_lower = topic.lower()
        self._evolution_state["topic_frequencies"][topic_lower] = \
            self._evolution_state["topic_frequencies"].get(topic_lower, 0) + 1

        # Increase wisdom quotient
        self._evolution_state["wisdom_quotient"] += len(content) / 1000.0

        self._save_evolution_state()
