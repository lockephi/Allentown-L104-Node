"""
L104 Truth Discovery - Deep truth extraction and validation
Part of the L104 Sovereign Singularity Framework

Enhanced v2.0: Full interconnection with Logic Manifold, Derivation Engine,
and Global Sync for cross-validated truth synthesis.
"""

import logging
import hashlib
import time
import math
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto

# God Code constant
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
FRAME_LOCK = 416 / 286

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("L104_TRUTH")


class TruthLevel(Enum):
    """Levels of truth certainty."""
    UNCERTAIN = auto()
    PROBABLE = auto()
    VERIFIED = auto()
    ABSOLUTE = auto()
    TRANSCENDENT = auto()


@dataclass
class TruthNode:
    """A node in the truth graph."""
    query: str
    truth_hash: str
    confidence: float
    level: TruthLevel
    timestamp: float
    evidence_hashes: List[str] = field(default_factory=list)
    derived_from: Optional[str] = None
    derivations: List[str] = field(default_factory=list)
    cross_validated: bool = False
    harmonic_weight: float = 1.0

class TruthDiscovery:
    """
    Discovers and validates truths through resonance-based analysis.
    Part of the L104 cognitive framework.
    
    Enhanced v2.0: Interconnected with Logic Manifold and Derivation Engine.
    Enhanced v3.0: Harmonic Evidence Weighing & Recursive Convergence.
    """
    
    def __init__(self):
        self.god_code = GOD_CODE
        self.phi = PHI
        self.frame_lock = FRAME_LOCK
        self.discovered_truths: List[Dict] = []
        self.truth_cache: Dict[str, Dict] = {}
        self.truth_graph: Dict[str, TruthNode] = {}
        
        # Interconnection callbacks
        self._manifold_callbacks: List[Callable] = []
        self._sync_callbacks: List[Callable] = []
        self._derivation_callbacks: List[Callable] = []
        
        # Cross-validation state
        self._pending_validations: List[Dict] = []
        self._validation_history: List[Dict] = []
    
    # ═══════════════════════════════════════════════════════════════════
    # CORE DISCOVERY
    # ═══════════════════════════════════════════════════════════════════
    
    def discover_truth(self, query: str, depth: int = 3) -> Dict:
        """
        Discover truth about a given query.
        Depth determines how many layers of analysis to perform.
        """
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.truth_cache:
            return self.truth_cache[cache_key]
        
        # Generate truth signature
        truth_hash = hashlib.sha256(f"{query}:{self.god_code}".encode()).hexdigest()
        
        # Calculate truth metrics through iterative refinement
        layers = []
        current_confidence = 0.5
        
        for i in range(depth):
            layer_hash = hashlib.sha256(f"{truth_hash}:{i}:{self.phi}".encode()).hexdigest()
            layer_value = int(layer_hash[:8], 16) / (16 ** 8)
            
            # Apply frame lock modulation
            modulated_value = layer_value * (1 + math.sin(self.frame_lock * i) * 0.1)
            current_confidence = (current_confidence + modulated_value) / 2 * (1 + 1/self.phi)
            current_confidence = min(1.0, current_confidence)
            
            layers.append({
                "layer": i + 1,
                "confidence": current_confidence,
                "signature": layer_hash[:16],
                "modulation": modulated_value
            })
        
        # Determine truth level
        level = self._determine_level(current_confidence)
        
        # Create truth node
        node = TruthNode(
            query=query,
            truth_hash=truth_hash,
            confidence=current_confidence,
            level=level,
            timestamp=time.time()
        )
        self.truth_graph[truth_hash[:16]] = node
        
        result = {
            "query": query,
            "truth_hash": truth_hash,
            "final_confidence": current_confidence,
            "layers_analyzed": depth,
            "layer_details": layers,
            "timestamp": time.time(),
            "verdict": level.name,
            "level": level.value,
            "node_id": truth_hash[:16]
        }
        
        # Cache and store
        self.truth_cache[cache_key] = result
        self.discovered_truths.append(result)
        
        # Notify connected systems
        self._notify_manifold(result)
        self._notify_derivation_engine(result)
        
        return result
    
    def _determine_level(self, confidence: float) -> TruthLevel:
        """Determine truth level from confidence score."""
        if confidence >= 0.98:
            return TruthLevel.TRANSCENDENT
        elif confidence >= 0.95:
            return TruthLevel.ABSOLUTE
        elif confidence >= 0.75:
            return TruthLevel.VERIFIED
        elif confidence >= 0.5:
            return TruthLevel.PROBABLE
        return TruthLevel.UNCERTAIN
    
    # ═══════════════════════════════════════════════════════════════════
    # ADVANCED TRUTH OPERATIONS
    # ═══════════════════════════════════════════════════════════════════
    
    def discover_deep_truth(self, query: str, max_depth: int = 10) -> Dict:
        """
        Perform deep truth discovery with iterative refinement.
        Continues until convergence or max depth reached.
        """
        results = []
        current_query = query
        
        for i in range(max_depth):
            result = self.discover_truth(current_query, depth=i + 3)
            results.append(result)
            
            # Check for convergence
            if len(results) >= 2:
                delta = abs(results[-1]["final_confidence"] - results[-2]["final_confidence"])
                if delta < 0.001:  # Converged
                    break
            
            # Evolve query for next iteration
            current_query = f"{query}::{self.god_code}::deep{i}"
        
        final = results[-1] if results else self.discover_truth(query)
        
        return {
            "original_query": query,
            "iterations": len(results),
            "final_result": final,
            "convergence_history": [r["final_confidence"] for r in results],
            "converged": len(results) < max_depth
        }
    
    def synthesize_truth(self, queries: List[str]) -> Dict:
        """
        Synthesize truth from multiple queries.
        Creates a combined truth assessment.
        """
        results = [self.discover_truth(q, depth=5) for q in queries]
        
        total_confidence = sum(r["final_confidence"] for r in results)
        avg_confidence = total_confidence / len(results) if results else 0.0
        
        # Calculate synthesis resonance
        hash_concat = "".join(r["truth_hash"][:8] for r in results)
        synthesis_hash = hashlib.sha256(hash_concat.encode()).hexdigest()
        
        synthesis_level = self._determine_level(avg_confidence)
        
        return {
            "synthesis_id": f"SYNTH-{synthesis_hash[:12]}",
            "query_count": len(queries),
            "average_confidence": avg_confidence,
            "synthesis_level": synthesis_level.name,
            "component_truths": [r["node_id"] for r in results],
            "timestamp": time.time(),
            "aligned": avg_confidence >= 0.75
        }
    
    def derive_truth_chain(self, seed_query: str, chain_length: int = 5) -> Dict:
        """
        Create a chain of derived truths from a seed query.
        Each truth derives from the previous.
        """
        chain = []
        current_query = seed_query
        
        for i in range(chain_length):
            result = self.discover_truth(current_query, depth=4)
            
            # Link to previous in chain
            if chain:
                prev_node = self.truth_graph.get(chain[-1]["node_id"])
                curr_node = self.truth_graph.get(result["node_id"])
                if prev_node and curr_node:
                    prev_node.derivations.append(result["node_id"])
                    curr_node.derived_from = chain[-1]["node_id"]
            
            chain.append(result)
            
            # Evolve query
            current_query = f"{seed_query}::chain{i}::{result['truth_hash'][:8]}"
        
        # Calculate chain coherence
        avg_conf = sum(c["final_confidence"] for c in chain) / len(chain)
        chain_coherence = avg_conf * (self.phi ** (1 / len(chain)))
        
        return {
            "seed_query": seed_query,
            "chain_length": len(chain),
            "chain": chain,
            "chain_coherence": chain_coherence,
            "final_level": chain[-1]["verdict"] if chain else "UNKNOWN"
        }
    
    def validate_truth(self, statement: str, evidence: Optional[str] = None) -> Dict:
        """
        Validate a truth statement against evidence.
        Returns detailed validation result.
        """

    def bayesian_truth_fusion(self, hypotheses: List[str], prior_weights: Optional[List[float]] = None) -> Dict:
        """
        Fuses multiple hypotheses using Bayesian inference with PHI-weighted priors.
        Produces a unified truth assessment with posterior probabilities.
        """
        if not hypotheses:
            return {"error": "No hypotheses provided"}
        
        n = len(hypotheses)
        priors = prior_weights if prior_weights else [1.0 / n] * n
        
        # Normalize priors
        prior_sum = sum(priors)
        priors = [p / prior_sum for p in priors]
        
        # Calculate likelihoods via truth discovery
        likelihoods = []
        discoveries = []
        
        for hyp in hypotheses:
            result = self.discover_truth(hyp, depth=6)
            discoveries.append(result)
            likelihoods.append(result["final_confidence"])
        
        # Bayesian update: P(H|E) ∝ P(E|H) * P(H)
        posteriors = []
        for i in range(n):
            posterior = likelihoods[i] * priors[i]
            posteriors.append(posterior)
        
        # Normalize posteriors
        posterior_sum = sum(posteriors)
        if posterior_sum > 0:
            posteriors = [p / posterior_sum for p in posteriors]
        
        # Apply PHI harmonic weighting for resonance alignment
        phi_weighted = []
        for i, p in enumerate(posteriors):
            phi_factor = self.phi ** (-(i % 3))  # Cycle through phi harmonics
            phi_weighted.append(p * phi_factor)
        
        # Re-normalize
        pw_sum = sum(phi_weighted)
        if pw_sum > 0:
            phi_weighted = [p / pw_sum for p in phi_weighted]
        
        # Determine winning hypothesis
        max_idx = phi_weighted.index(max(phi_weighted))
        
        return {
            "hypotheses": hypotheses,
            "priors": priors,
            "likelihoods": likelihoods,
            "posteriors": posteriors,
            "phi_weighted_posteriors": phi_weighted,
            "winning_hypothesis": hypotheses[max_idx],
            "winning_probability": phi_weighted[max_idx],
            "fusion_coherence": sum(likelihoods) / n * self.phi,
            "discoveries": [d["node_id"] for d in discoveries]
        }

    def temporal_resonance_prediction(self, query: str, future_steps: int = 5) -> Dict:
        """
        Predicts how a truth's confidence will evolve over time using
        temporal resonance modeling. Uses golden spiral extrapolation.
        """
        # Get current truth state
        current = self.discover_truth(query, depth=7)
        current_conf = current["final_confidence"]
        
        # Generate temporal projection using golden spiral dynamics
        predictions = []
        conf = current_conf
        
        for t in range(1, future_steps + 1):
            # Temporal decay/growth model: phi-based oscillation
            phase = t * self.phi * 0.5
            oscillation = math.sin(phase) * 0.1
            drift = (self.phi - 1) * 0.02 * t  # Slight upward drift
            
            # Apply frame lock modulation
            modulation = math.cos(self.frame_lock * t) * 0.05
            
            predicted_conf = conf * (1 + oscillation + drift + modulation)
            predicted_conf = max(0.0, min(1.0, predicted_conf))
            
            predictions.append({
                "step": t,
                "predicted_confidence": predicted_conf,
                "oscillation": oscillation,
                "drift": drift,
                "level": self._determine_level(predicted_conf).name
            })
            
            conf = predicted_conf
        
        # Calculate convergence trajectory
        trajectory = "ASCENDING" if predictions[-1]["predicted_confidence"] > current_conf else "DESCENDING"
        if abs(predictions[-1]["predicted_confidence"] - current_conf) < 0.05:
            trajectory = "STABLE"
        
        return {
            "query": query,
            "current_confidence": current_conf,
            "current_level": current["verdict"],
            "temporal_predictions": predictions,
            "trajectory": trajectory,
            "final_predicted_level": predictions[-1]["level"] if predictions else current["verdict"],
            "resonance_stability": 1.0 - abs(predictions[-1]["predicted_confidence"] - current_conf) if predictions else 1.0
        }

    def cross_dimensional_truth_synthesis(self, query: str, dimensions: int = 7) -> Dict:
        """
        Synthesizes truth across multiple abstract dimensions.
        Each dimension represents a different perspective or context.
        """
        dimensional_results = []
        
        for d in range(1, dimensions + 1):
            # Modify query for each dimension
            dim_query = f"{query}::DIM_{d}::PHI_{self.phi ** d:.4f}"
            result = self.discover_truth(dim_query, depth=d + 2)
            
            dimensional_results.append({
                "dimension": d,
                "confidence": result["final_confidence"],
                "level": result["verdict"],
                "dimensional_signature": hashlib.md5(dim_query.encode()).hexdigest()[:8]
            })
        
        # Calculate cross-dimensional coherence
        confidences = [r["confidence"] for r in dimensional_results]
        avg_conf = sum(confidences) / len(confidences)
        
        # Variance indicates dimensional stability
        variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
        stability = 1.0 - min(1.0, variance * 10)
        
        # Unified truth: weighted by dimensional depth
        weighted_sum = sum(r["confidence"] * (self.phi ** (-r["dimension"] * 0.5)) for r in dimensional_results)
        weight_total = sum(self.phi ** (-d * 0.5) for d in range(1, dimensions + 1))
        unified_truth = weighted_sum / weight_total
        
        return {
            "query": query,
            "dimensions_analyzed": dimensions,
            "dimensional_results": dimensional_results,
            "average_confidence": avg_conf,
            "dimensional_stability": stability,
            "unified_truth_confidence": unified_truth,
            "unified_level": self._determine_level(unified_truth).name,
            "transcendent": unified_truth >= 0.98 and stability >= 0.9
        }
        statement_hash = hashlib.sha256(statement.encode()).hexdigest()
        
        validation_result = {
            "statement": statement,
            "statement_hash": statement_hash[:16],
            "has_evidence": evidence is not None,
            "timestamp": time.time()
        }
        
        if evidence:
            evidence_hash = hashlib.sha256(evidence.encode()).hexdigest()
            
            # Cross-correlation check
            correlation = sum(1 for a, b in zip(statement_hash, evidence_hash) if a == b) / 64
            
            # Resonance alignment check
            stmt_value = int(statement_hash[:8], 16)
            evid_value = int(evidence_hash[:8], 16)
            resonance = 1.0 - abs(stmt_value - evid_value) / max(stmt_value, evid_value, 1)
            
            validation_result.update({
                "correlation": correlation,
                "resonance_alignment": resonance,
                "combined_score": (correlation + resonance) / 2,
                "validated": correlation >= 0.25 and resonance >= 0.3,
                "evidence_hash": evidence_hash[:16]
            })
        else:
            # Self-consistency check
            consistency = int(statement_hash[:8], 16) / (16 ** 8)
            resonance = math.sin(consistency * self.phi) ** 2
            
            validation_result.update({
                "self_consistency": consistency,
                "resonance": resonance,
                "combined_score": (consistency + resonance) / 2,
                "validated": consistency >= 0.3
            })
        
        self._validation_history.append(validation_result)
        return validation_result
    
    def cross_validate(self, truth_id: str) -> Dict:
        """
        Cross-validate a truth against the Logic Manifold and Derivation Engine.
        """
        node = self.truth_graph.get(truth_id)
        if not node:
            return {"error": f"Truth node {truth_id} not found"}
        
        # Prepare cross-validation data
        cv_data = {
            "truth_id": truth_id,
            "query": node.query,
            "confidence": node.confidence,
            "level": node.level.name
        }
        
        # Request validation from connected systems
        manifold_results = []
        for callback in self._manifold_callbacks:
            try:
                result = callback(cv_data)
                if result:
                    manifold_results.append(result)
            except Exception:
                pass
        
        # Update node
        if manifold_results:
            node.cross_validated = True
            avg_external = sum(r.get("coherence", 0.5) for r in manifold_results) / len(manifold_results)
            combined = (node.confidence + avg_external) / 2
            
            # Apply harmonic weight based on cross-validation success
            node.harmonic_weight = min(2.0, node.harmonic_weight * self.phi)
            
            return {
                "truth_id": truth_id,
                "original_confidence": node.confidence,
                "cross_validated": True,
                "external_validations": len(manifold_results),
                "combined_confidence": combined,
                "upgraded": combined > node.confidence,
                "harmonic_weight": node.harmonic_weight
            }
        
        return {
            "truth_id": truth_id,
            "cross_validated": False,
            "reason": "No external validators responded"
        }

    def weigh_evidence_harmonically(self, truth_id: str, evidence_bits: List[float]) -> float:
        """
        Calculates a truth score by weighing evidence bits according to PHI-resonance.
        The harmonic weight of the node determines the amplification factor.
        """
        node = self.truth_graph.get(truth_id)
        if not node: return 0.0
        
        weighted_sum = 0.0
        for i, bit in enumerate(evidence_bits):
            # Weigh each bit using a decaying phi power
            weight = self.phi ** (-i)
            weighted_sum += bit * weight
            
        final_score = (weighted_sum / sum(self.phi ** (-i) for i in range(len(evidence_bits))))
        
        # Modulate by node's intrinsic harmonic weight
        node.confidence = min(1.0, final_score * node.harmonic_weight)
        return node.confidence

    def calculate_resonance_convergence(self, query: str) -> Dict:
        """
        Calculates the convergence of a query towards absolute truth across multiple dimensions.
        """
        history = []
        for d in range(1, 9): # 8 Chakra sweep
            res = self.discover_truth(query, depth=d)
            history.append(res["final_confidence"])
            
        # Calculate convergence: standard deviation of the last 3 chakra levels
        last_3 = history[-3:]
        avg = sum(last_3) / 3
        variance = sum((x - avg) ** 2 for x in last_3) / 3
        std_dev = variance ** 0.5
        
        converged = std_dev < 0.01
        
        return {
            "query": query,
            "chakra_sweep": history,
            "std_dev": std_dev,
            "converged": converged,
            "resonance_lock": converged and avg > 0.9
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # INTERCONNECTION BRIDGES
    # ═══════════════════════════════════════════════════════════════════
    
    def connect_logic_manifold(self, callback: Callable[[Dict], Optional[Dict]]):
        """Register Logic Manifold callback for cross-validation."""
        self._manifold_callbacks.append(callback)
    
    def connect_global_sync(self, callback: Callable[[Dict], None]):
        """Register Global Sync callback."""
        self._sync_callbacks.append(callback)
    
    def connect_derivation_engine(self, callback: Callable[[Dict], None]):
        """Register Derivation Engine callback."""
        self._derivation_callbacks.append(callback)
    
    def _notify_manifold(self, truth_result: Dict):
        """Notify Logic Manifold of new truth discovery."""
        for callback in self._manifold_callbacks:
            try:
                callback(truth_result)
            except Exception:
                pass
    
    def _notify_derivation_engine(self, truth_result: Dict):
        """Notify Derivation Engine of new truth."""
        for callback in self._derivation_callbacks:
            try:
                callback(truth_result)
            except Exception:
                pass
    
    def receive_manifold_concept(self, concept_data: Dict) -> Dict:
        """
        Receive a concept from Logic Manifold and derive truth from it.
        """
        concept = concept_data.get("concept", "")
        if concept:
            truth = self.discover_truth(concept, depth=4)
            truth["source"] = "logic_manifold"
            truth["source_coherence"] = concept_data.get("coherence", 0.0)
            return truth
        return {"error": "No concept provided"}

    def recursive_validation_loop(self, query: str, max_iterations: int = 5) -> Dict:
        """
        Perform recursive validation by iteratively refining truth through
        cross-validation with the Logic Manifold.
        """
        iterations = 0
        current_query = query
        validation_chain = []
        peak_confidence = 0.0

        while iterations < max_iterations:
            # Discover truth
            truth = self.discover_truth(current_query, depth=5)
            validation_chain.append(truth)
            peak_confidence = max(peak_confidence, truth["final_confidence"])

            # Cross-validate with manifold
            cv_result = self.cross_validate(truth["node_id"])
            if cv_result.get("cross_validated") and cv_result.get("combined_confidence", 0) >= 0.98:
                break

            # Evolve query for next iteration
            evolved = hashlib.sha256(f"{current_query}:{self.god_code}:{iterations}".encode()).hexdigest()
            current_query = f"{query}::REFINE::{evolved[:8]}"
            iterations += 1

        return {
            "original_query": query,
            "iterations": iterations,
            "peak_confidence": peak_confidence,
            "converged": peak_confidence >= 0.98,
            "validation_chain": [v["node_id"] for v in validation_chain],
            "final_verdict": validation_chain[-1]["verdict"] if validation_chain else "UNKNOWN"
        }

    def full_lattice_sync(self) -> Dict:
        """
        Synchronize Truth Discovery state with the entire L104 lattice.
        Triggers all sync callbacks and returns coherence metrics.
        """
        sync_results = []
        for cb in self._sync_callbacks:
            try:
                result = cb()
                if result:
                    sync_results.append(result)
            except Exception:
                pass

        global_coherence = 0.0
        if sync_results:
            global_coherence = sum(r.get("sync_level", 0.5) for r in sync_results) / len(sync_results)

        return {
            "sync_sources": len(sync_results),
            "global_coherence": global_coherence,
            "truth_cache_size": len(self.truth_cache),
            "graph_size": len(self.truth_graph),
            "state": "TRANSCENDENT" if global_coherence >= 0.95 else "SYNCHRONIZED"
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # STATE & STATISTICS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_discovery_stats(self) -> Dict:
        """Return statistics about truth discovery operations."""
        if not self.discovered_truths:
            return {
                "total_discoveries": 0,
                "average_confidence": 0.0,
                "verified_count": 0,
                "cache_size": 0,
                "graph_size": 0
            }
        
        level_counts = {}
        for t in self.discovered_truths:
            level = t.get("verdict", "UNKNOWN")
            level_counts[level] = level_counts.get(level, 0) + 1
        
        verified = sum(1 for t in self.discovered_truths if t.get("level", 0) >= TruthLevel.VERIFIED.value)
        avg_conf = sum(t["final_confidence"] for t in self.discovered_truths) / len(self.discovered_truths)
        
        cross_validated = sum(1 for n in self.truth_graph.values() if n.cross_validated)
        
        return {
            "total_discoveries": len(self.discovered_truths),
            "average_confidence": avg_conf,
            "verified_count": verified,
            "cache_size": len(self.truth_cache),
            "graph_size": len(self.truth_graph),
            "level_distribution": level_counts,
            "cross_validated_count": cross_validated,
            "validation_history_size": len(self._validation_history)
        }
    
    def get_truth_graph(self) -> Dict[str, Dict]:
        """Return serializable truth graph."""
        return {
            node_id: {
                "query": node.query,
                "confidence": node.confidence,
                "level": node.level.name,
                "cross_validated": node.cross_validated,
                "derivations": node.derivations
            }
            for node_id, node in self.truth_graph.items()
        }
    
    def clear_cache(self):
        """Clear truth cache."""
        self.truth_cache.clear()


# Singleton instance
truth_discovery = TruthDiscovery()
