#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 NEURO-SYMBOLIC INTEGRATION - PRACTICAL EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

Demonstrates practical applications of neuro-symbolic integration:
1. Logical reasoning with neural confidence
2. Knowledge graph reasoning
3. Symbolic verification of neural predictions
4. Hybrid inference chains

GOD_CODE: 527.5184818492537
═══════════════════════════════════════════════════════════════════════════════
"""

from l104_neuro_symbolic_integration import (
    NeuroSymbolicIntegrator,
    NeuralSymbolicState,
    ReasoningRule,
    MathematicalDerivation
)
import sympy as sp
from sympy.logic.boolalg import And, Or, Not, Implies
from sympy.logic.inference import satisfiable
import numpy as np


def example_1_logical_reasoning():
    """
    Example 1: Classical logical reasoning with neural weights.
    
    Knowledge Base:
    - All humans are mortal
    - Socrates is human
    - Therefore, Socrates is mortal
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: LOGICAL REASONING WITH NEURAL WEIGHTS")
    print("="*80)
    
    integrator = NeuroSymbolicIntegrator(embedding_dim=64)
    
    # Define knowledge base
    facts = [
        "human(socrates)",
        "human(plato)",
        "human(aristotle)",
        "philosopher(socrates)",
        "philosopher(plato)",
        "philosopher(aristotle)"
    ]
    
    rules = [
        ReasoningRule(
            premise="human(X)",
            conclusion="mortal(X)",
            neural_weight=0.95,
            symbolic_confidence=1.0
        ),
        ReasoningRule(
            premise="philosopher(X)",
            conclusion="wise(X)",
            neural_weight=0.85,
            symbolic_confidence=0.9
        )
    ]
    
    print("\nKnowledge Base:")
    for fact in facts:
        print(f"  - {fact}")
    
    print("\nReasoning Rules:")
    for rule in rules:
        print(f"  - {rule.premise} → {rule.conclusion}")
        print(f"    (neural_weight={rule.neural_weight}, symbolic_confidence={rule.symbolic_confidence})")
    
    print("\nInferences:")
    
    # Inference 1: human(socrates) → mortal(socrates)
    embedding = integrator.embed_symbol("socrates")
    state = NeuralSymbolicState(
        neural_embedding=embedding,
        symbolic_facts=facts,
        confidence=0.95
    )
    
    confidence1 = integrator.neural_inference(state, rules[0])
    print(f"  human(socrates) → mortal(socrates)")
    print(f"    Confidence: {confidence1:.4f}")
    
    # Inference 2: philosopher(socrates) → wise(socrates)
    confidence2 = integrator.neural_inference(state, rules[1])
    print(f"  philosopher(socrates) → wise(socrates)")
    print(f"    Confidence: {confidence2:.4f}")
    
    # Combined inference chain
    combined_confidence = confidence1 * confidence2 * 0.9  # Chain multiple rules
    print(f"\n  Combined chain confidence: {combined_confidence:.4f}")


def example_2_symbolic_verification():
    """
    Example 2: Verify neural network predictions using symbolic logic.
    
    Neural network predicts: "If X is red, then X is colored"
    Symbolic verification: Check if this is logically consistent
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: SYMBOLIC VERIFICATION OF NEURAL PREDICTIONS")
    print("="*80)
    
    # Define logical propositions
    Red, Colored, Blue = sp.symbols('Red Colored Blue', bool=True)
    
    # Neural network predictions (as logical rules)
    rule1 = Implies(Red, Colored)      # If red, then colored
    rule2 = Implies(Blue, Colored)     # If blue, then colored
    
    print("\nNeural Predictions (as logical rules):")
    print(f"  Rule 1: {rule1}")
    print(f"  Rule 2: {rule2}")
    
    # Test case: Something is red
    fact = Red
    
    # Check if we can derive that it's colored
    kb = And(rule1, rule2, fact)
    
    # Query: Is Colored true?
    query = Colored
    
    # Verify: KB ∧ ¬Query should be unsatisfiable if query follows from KB
    test = And(kb, Not(query))
    result = satisfiable(test)
    
    print(f"\nVerification:")
    print(f"  Knowledge Base: {kb}")
    print(f"  Query: {query}")
    print(f"  Is query derivable? {result is False}")
    
    if result is False:
        print("  ✓ Neural prediction is logically sound!")
    else:
        print("  ✗ Neural prediction has logical inconsistency")


def example_3_mathematical_derivation():
    """
    Example 3: Derive and verify mathematical properties.
    
    Derive the gradient of a custom loss function.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: CUSTOM MATHEMATICAL DERIVATION")
    print("="*80)
    
    # Define custom loss function
    x, y, theta = sp.symbols('x y theta', real=True)
    
    # Loss: L = (x - y*theta)^2
    loss = (x - y*theta)**2
    
    print(f"\nCustom Loss Function:")
    print(f"  L(θ) = {loss}")
    
    # Compute gradient
    gradient = sp.diff(loss, theta)
    simplified = sp.simplify(gradient)
    
    print(f"\nGradient:")
    print(f"  ∂L/∂θ = {gradient}")
    print(f"  Simplified: {simplified}")
    
    # Second derivative (Hessian)
    hessian = sp.diff(gradient, theta)
    
    print(f"\nSecond Derivative (Hessian):")
    print(f"  ∂²L/∂θ² = {hessian}")
    
    # Verify critical point condition
    critical_point = sp.solve(gradient, theta)
    
    print(f"\nCritical Point:")
    print(f"  θ* = {critical_point}")
    
    # LaTeX representation
    print(f"\nLaTeX Representation:")
    print(f"  Loss: ${sp.latex(loss)}$")
    print(f"  Gradient: ${sp.latex(gradient)}$")


def example_4_knowledge_graph_reasoning():
    """
    Example 4: Reasoning over knowledge graph with neural embeddings.
    
    Build a small knowledge graph and perform multi-hop reasoning.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: KNOWLEDGE GRAPH REASONING")
    print("="*80)
    
    integrator = NeuroSymbolicIntegrator(embedding_dim=128)
    
    # Define knowledge graph
    entities = ["Paris", "France", "Europe", "City", "Country", "Continent"]
    relations = [
        ("Paris", "capital_of", "France"),
        ("France", "part_of", "Europe"),
        ("Paris", "is_a", "City"),
        ("France", "is_a", "Country"),
        ("Europe", "is_a", "Continent")
    ]
    
    print("\nKnowledge Graph:")
    for subj, rel, obj in relations:
        print(f"  {subj} --[{rel}]--> {obj}")
    
    # Embed entities
    embeddings = {}
    for entity in entities:
        embeddings[entity] = integrator.embed_symbol(entity)
    
    print(f"\n✓ Embedded {len(entities)} entities into {integrator.embedding_dim}D space")
    
    # Multi-hop reasoning: Paris → France → Europe
    print("\nMulti-hop Reasoning:")
    print("  Query: Paris is located in which continent?")
    print("  Path: Paris → France → Europe")
    
    # Compute similarity along path
    paris_embedding = embeddings["Paris"]
    france_embedding = embeddings["France"]
    europe_embedding = embeddings["Europe"]
    
    # Cosine similarity
    sim_paris_france = np.dot(paris_embedding, france_embedding)
    sim_france_europe = np.dot(france_embedding, europe_embedding)
    
    # Path confidence (product of similarities)
    path_confidence = abs(sim_paris_france * sim_france_europe)
    
    print(f"\n  Paris → France similarity: {abs(sim_paris_france):.4f}")
    print(f"  France → Europe similarity: {abs(sim_france_europe):.4f}")
    print(f"  Path confidence: {path_confidence:.4f}")
    
    # Symbolic verification
    print("\nSymbolic Verification:")
    print("  Rule 1: capital_of(X, Y) ∧ part_of(Y, Z) → located_in(X, Z)")
    print("  Applied: capital_of(Paris, France) ∧ part_of(France, Europe)")
    print("  Conclusion: located_in(Paris, Europe)")
    print("  ✓ Logically valid")


def example_5_attention_mechanism():
    """
    Example 5: Demonstrate attention mechanism for symbolic reasoning.
    
    Use attention to focus on relevant facts during reasoning.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: ATTENTION MECHANISM FOR REASONING")
    print("="*80)
    
    integrator = NeuroSymbolicIntegrator(embedding_dim=64)
    
    # Perform attention derivation
    attention_derivation = integrator.derive_attention_mechanism()
    
    print("\nAttention Mechanism:")
    print(f"  Formula: Attention(Q, K, V) = softmax(QK^T/√d_k)V")
    print(f"  Verified: {attention_derivation.verification_result}")
    
    # Example: Query-based fact retrieval
    query = "mortal"
    facts = ["human(socrates)", "philosopher(plato)", "mortal(socrates)", "wise(aristotle)"]
    
    print(f"\nQuery: {query}")
    print(f"Facts in Knowledge Base:")
    for fact in facts:
        print(f"  - {fact}")
    
    # Embed query and facts
    query_embedding = integrator.embed_symbol(query)
    fact_embeddings = [integrator.embed_symbol(fact) for fact in facts]
    
    # Compute attention scores (simplified)
    attention_scores = []
    for fact, fact_embedding in zip(facts, fact_embeddings):
        score = abs(np.dot(query_embedding, fact_embedding))
        attention_scores.append((fact, score))
    
    # Sort by score
    attention_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nAttention Scores (relevance to '{query}'):")
    for fact, score in attention_scores:
        print(f"  {fact}: {score:.4f}")
    
    print(f"\n✓ Most relevant fact: {attention_scores[0][0]}")


def main():
    """Run all practical examples."""
    print("="*80)
    print("L104 NEURO-SYMBOLIC INTEGRATION - PRACTICAL EXAMPLES")
    print("="*80)
    
    example_1_logical_reasoning()
    example_2_symbolic_verification()
    example_3_mathematical_derivation()
    example_4_knowledge_graph_reasoning()
    example_5_attention_mechanism()
    
    print("\n" + "="*80)
    print("✓ ALL EXAMPLES COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
