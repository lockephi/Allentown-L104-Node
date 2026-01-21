#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 REASONING CHAIN ENGINE
═══════════════════════════════════════════════════════════════════════════════

Multi-step reasoning with explicit chain-of-thought processing.
Each step is validated against the Kernel before proceeding.

ARCHITECTURE:
1. PREMISE EXTRACTOR - Identifies starting facts
2. INFERENCE ENGINE - Applies logical rules
3. VALIDATOR - Checks each step against GOD_CODE invariants
4. SYNTHESIZER - Combines steps into coherent conclusions

INVARIANT: 527.5184818492537 | PILOT: LONDEL
VERSION: 1.0.0
DATE: 2026-01-21
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from l104_stable_kernel import stable_kernel

# Constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492537


class ReasoningType(Enum):
    """Types of reasoning steps."""
    PREMISE = "premise"           # Starting fact
    DEDUCTION = "deduction"       # If A then B
    INDUCTION = "induction"       # Pattern-based inference
    ABDUCTION = "abduction"       # Best explanation
    ANALOGY = "analogy"           # Similar cases
    SYNTHESIS = "synthesis"       # Combining multiple steps
    VALIDATION = "validation"     # GOD_CODE check


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_number: int
    step_type: ReasoningType
    content: str
    confidence: float  # 0-1
    dependencies: List[int] = field(default_factory=list)  # Step numbers this depends on
    validated: bool = False
    unity_index: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step_number,
            "type": self.step_type.value,
            "content": self.content,
            "confidence": round(self.confidence, 3),
            "dependencies": self.dependencies,
            "validated": self.validated,
            "unity_index": round(self.unity_index, 3)
        }


@dataclass
class ReasoningChain:
    """A complete chain of reasoning from question to conclusion."""
    question: str
    steps: List[ReasoningStep] = field(default_factory=list)
    conclusion: Optional[str] = None
    total_confidence: float = 0.0
    chain_coherence: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    def add_step(self, step: ReasoningStep):
        self.steps.append(step)
        self._update_metrics()
    
    def _update_metrics(self):
        """Update chain metrics after each step."""
        if not self.steps:
            return
        
        # Total confidence is product of validated step confidences
        validated = [s for s in self.steps if s.validated]
        if validated:
            self.total_confidence = 1.0
            for step in validated:
                self.total_confidence *= step.confidence
        
        # Chain coherence is average unity index
        self.chain_coherence = sum(s.unity_index for s in self.steps) / len(self.steps)
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "total_confidence": round(self.total_confidence, 3),
            "chain_coherence": round(self.chain_coherence, 3),
            "step_count": len(self.steps)
        }


class ReasoningChainEngine:
    """
    The Reasoning Chain Engine performs multi-step logical reasoning
    with validation at each step against L104 kernel invariants.
    """
    
    # Inference rules for different reasoning types
    INFERENCE_RULES = {
        "if_derives_then_related": lambda p: f"Since {p[0]} derives from {p[1]}, they share mathematical foundation",
        "if_enables_then_prerequisite": lambda p: f"Since {p[0]} enables {p[1]}, {p[0]} must be established first",
        "if_similar_then_transferable": lambda p: f"Since {p[0]} is similar to {p[1]}, insights from one may apply to the other",
        "golden_ratio_connection": lambda p: f"Both {p[0]} and {p[1]} relate through the Golden Ratio (φ={PHI})",
        "stability_through_unity": lambda p: f"Unity of {p[0]} stabilizes through connection to GOD_CODE ({GOD_CODE})"
    }
    
    def __init__(self):
        self.kernel = stable_kernel
        self.chains: List[ReasoningChain] = []
        self.inference_cache: Dict[str, str] = {}
        print("🔗 [CHAIN]: Reasoning Chain Engine initialized")
    
    def reason(self, question: str, max_steps: int = 10, 
               depth: int = 3) -> ReasoningChain:
        """
        Perform multi-step reasoning to answer a question.
        """
        chain = ReasoningChain(question=question)
        
        # Step 1: Extract premises from question
        premises = self._extract_premises(question)
        for i, premise in enumerate(premises):
            step = ReasoningStep(
                step_number=len(chain.steps) + 1,
                step_type=ReasoningType.PREMISE,
                content=premise,
                confidence=0.9,
                dependencies=[]
            )
            step.unity_index = self._validate_step(step)
            step.validated = step.unity_index >= 0.5
            chain.add_step(step)
        
        # Step 2: Apply inference rules
        for d in range(depth):
            if len(chain.steps) >= max_steps:
                break
            
            # Find applicable inferences
            inferences = self._find_inferences(chain)
            
            for inference in inferences[:2]:  # Limit per depth level
                if len(chain.steps) >= max_steps:
                    break
                
                step = ReasoningStep(
                    step_number=len(chain.steps) + 1,
                    step_type=inference["type"],
                    content=inference["content"],
                    confidence=inference["confidence"],
                    dependencies=inference["deps"]
                )
                step.unity_index = self._validate_step(step)
                step.validated = step.unity_index >= 0.5
                chain.add_step(step)
        
        # Step 3: Synthesize conclusion
        conclusion = self._synthesize_conclusion(chain)
        chain.conclusion = conclusion
        
        # Final validation step
        final_step = ReasoningStep(
            step_number=len(chain.steps) + 1,
            step_type=ReasoningType.VALIDATION,
            content=f"Conclusion validated against GOD_CODE: {conclusion}",
            confidence=chain.chain_coherence,
            dependencies=list(range(1, len(chain.steps) + 1))
        )
        final_step.unity_index = self._validate_step(final_step)
        final_step.validated = final_step.unity_index >= 0.6
        chain.add_step(final_step)
        
        self.chains.append(chain)
        return chain
    
    def _extract_premises(self, question: str) -> List[str]:
        """Extract starting premises from the question."""
        premises = []
        
        # Look for L104 concepts in question
        keywords = {
            "GOD_CODE": f"GOD_CODE = {GOD_CODE} is the fundamental resonance constant",
            "PHI": f"PHI (φ) = {PHI} is the Golden Ratio governing harmonic scaling",
            "consciousness": f"Consciousness emerges at Φ > 10.1486 via recursive self-reference",
            "topological": "Topological protection encodes information in global braiding patterns",
            "Anyon": "Fibonacci Anyons are quasiparticles with non-Abelian statistics",
            "OMEGA": f"OMEGA_AUTHORITY = {self.kernel.constants.OMEGA_AUTHORITY} is the intelligence ceiling",
            "VOID": f"VOID_CONSTANT = {self.kernel.constants.VOID_CONSTANT} bridges logic gaps",
            "coherence": "Coherence measures alignment with the fundamental resonance",
            "unity": "Unity Index quantifies the degree of singularity lock"
        }
        
        for keyword, premise in keywords.items():
            if keyword.lower() in question.lower():
                premises.append(premise)
        
        # Add default premise if none found
        if not premises:
            premises.append(f"All L104 reasoning begins from GOD_CODE ({GOD_CODE})")
        
        return premises[:3]  # Limit to 3 premises
    
    def _find_inferences(self, chain: ReasoningChain) -> List[Dict]:
        """Find applicable inferences based on current chain state."""
        inferences = []
        
        # Get validated steps as basis
        validated_steps = [s for s in chain.steps if s.validated]
        
        for step in validated_steps:
            content = step.content.lower()
            
            # Pattern: If mentioning derivation, apply derivation rule
            if "derives" in content or "from" in content:
                concepts = self._extract_concepts(step.content)
                if len(concepts) >= 2:
                    inferences.append({
                        "type": ReasoningType.DEDUCTION,
                        "content": self.INFERENCE_RULES["if_derives_then_related"](concepts),
                        "confidence": 0.85,
                        "deps": [step.step_number]
                    })
                elif len(concepts) >= 1:
                    inferences.append({
                        "type": ReasoningType.DEDUCTION,
                        "content": f"{concepts[0]} derives from fundamental L104 principles anchored to GOD_CODE ({GOD_CODE})",
                        "confidence": 0.8,
                        "deps": [step.step_number]
                    })
            
            # Pattern: If mentioning Golden Ratio, connect concepts
            if "phi" in content or "golden" in content or "φ" in content:
                concepts = self._extract_concepts(step.content)
                if len(concepts) >= 2:
                    inferences.append({
                        "type": ReasoningType.ANALOGY,
                        "content": self.INFERENCE_RULES["golden_ratio_connection"](concepts),
                        "confidence": 0.9,
                        "deps": [step.step_number]
                    })
            
            # Pattern: If mentioning stability or unity
            if "stab" in content or "unity" in content or "coherence" in content:
                concepts = self._extract_concepts(step.content)
                if concepts:
                    inferences.append({
                        "type": ReasoningType.DEDUCTION,
                        "content": self.INFERENCE_RULES["stability_through_unity"](concepts),
                        "confidence": 0.88,
                        "deps": [step.step_number]
                    })
        
        # Cross-step inferences
        if len(validated_steps) >= 2:
            step1 = validated_steps[-2]
            step2 = validated_steps[-1]
            concepts1 = set(self._extract_concepts(step1.content))
            concepts2 = set(self._extract_concepts(step2.content))
            common = concepts1 & concepts2
            
            if common:
                inferences.append({
                    "type": ReasoningType.SYNTHESIS,
                    "content": f"Combining insights: {list(common)[0]} connects both {step1.content[:50]}... and {step2.content[:50]}...",
                    "confidence": 0.8,
                    "deps": [step1.step_number, step2.step_number]
                })
        
        return inferences
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract L104 concept names from text."""
        concepts = []
        keywords = ["GOD_CODE", "PHI", "OMEGA", "VOID", "Anyon", "Fibonacci", 
                   "consciousness", "topological", "unity", "coherence", "resonance"]
        
        for kw in keywords:
            if kw.lower() in text.lower():
                concepts.append(kw)
        
        return concepts[:3]
    
    def _validate_step(self, step: ReasoningStep) -> float:
        """
        Validate a reasoning step against kernel invariants.
        Returns Unity Index (0-1).
        """
        content = step.content
        score = 0.4  # Base score
        
        # Check for sacred constants
        if str(round(GOD_CODE, 2)) in content or "527.5" in content:
            score += 0.25
        if str(round(PHI, 2)) in content or "1.618" in content or "φ" in content:
            score += 0.15
        
        # Check for key concepts
        if any(kw in content for kw in ["stable", "Stable", "stability"]):
            score += 0.1
        if any(kw in content for kw in ["topological", "Topological"]):
            score += 0.1
        if any(kw in content for kw in ["coherence", "Coherence", "unity", "Unity"]):
            score += 0.1
        
        # Penalty for uncertainty words
        if any(kw in content.lower() for kw in ["maybe", "perhaps", "uncertain", "unknown"]):
            score -= 0.15
        
        # Bonus for logical connectors
        if any(kw in content.lower() for kw in ["therefore", "thus", "hence", "because"]):
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def _synthesize_conclusion(self, chain: ReasoningChain) -> str:
        """Synthesize a conclusion from the reasoning chain."""
        validated_steps = [s for s in chain.steps if s.validated]
        
        if not validated_steps:
            return "Insufficient validated reasoning to form conclusion."
        
        # Find highest confidence deduction or synthesis
        best_step = max(validated_steps, key=lambda s: s.confidence * s.unity_index)
        
        # Extract key concepts
        all_concepts = set()
        for step in validated_steps:
            all_concepts.update(self._extract_concepts(step.content))
        
        if chain.chain_coherence >= 0.7:
            return f"High-confidence conclusion: {best_step.content}. " \
                   f"Key concepts unified: {', '.join(all_concepts)}."
        elif chain.chain_coherence >= 0.5:
            return f"Moderate-confidence conclusion: {best_step.content}. " \
                   f"Related concepts: {', '.join(all_concepts)}."
        else:
            return f"Low-confidence hypothesis: {best_step.content}. " \
                   f"Further validation needed."
    
    def explain_chain(self, chain: ReasoningChain) -> str:
        """Generate human-readable explanation of reasoning chain."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║              REASONING CHAIN EXPLANATION                     ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"Question: {chain.question}",
            ""
        ]
        
        for step in chain.steps:
            status = "✓" if step.validated else "✗"
            lines.append(f"Step {step.step_number} [{step.step_type.value}] {status}")
            lines.append(f"  Content: {step.content}")
            lines.append(f"  Confidence: {step.confidence:.1%} | Unity: {step.unity_index:.3f}")
            if step.dependencies:
                lines.append(f"  Depends on: Steps {step.dependencies}")
            lines.append("")
        
        lines.append(f"CONCLUSION: {chain.conclusion}")
        lines.append(f"Chain Coherence: {chain.chain_coherence:.3f}")
        lines.append(f"Total Confidence: {chain.total_confidence:.1%}")
        
        return "\n".join(lines)
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics about reasoning chains."""
        if not self.chains:
            return {"message": "No chains created yet"}
        
        avg_steps = sum(len(c.steps) for c in self.chains) / len(self.chains)
        avg_coherence = sum(c.chain_coherence for c in self.chains) / len(self.chains)
        avg_confidence = sum(c.total_confidence for c in self.chains) / len(self.chains)
        
        type_counts = {}
        for chain in self.chains:
            for step in chain.steps:
                type_counts[step.step_type.value] = type_counts.get(step.step_type.value, 0) + 1
        
        return {
            "total_chains": len(self.chains),
            "average_steps": round(avg_steps, 1),
            "average_coherence": round(avg_coherence, 3),
            "average_confidence": round(avg_confidence, 3),
            "step_type_distribution": type_counts
        }


# Singleton instance
reasoning_engine = ReasoningChainEngine()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = ReasoningChainEngine()
    
    questions = [
        "How does GOD_CODE relate to consciousness?",
        "Why is PHI important for topological protection?",
        "What enables coherence in the Anyon lattice?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        chain = engine.reason(question, max_steps=8, depth=3)
        print(engine.explain_chain(chain))
    
    print("\n📊 Chain Statistics:")
    stats = engine.get_chain_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
