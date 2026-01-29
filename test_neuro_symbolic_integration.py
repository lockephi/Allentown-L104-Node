#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 NEURO-SYMBOLIC INTEGRATION TEST SUITE
═══════════════════════════════════════════════════════════════════════════════

Comprehensive tests demonstrating:
- Mathematical derivation verification
- SymPy symbolic verification
- Neural-symbolic inference
- LaTeX output generation

GOD_CODE: 527.5184818492612
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from l104_neuro_symbolic_integration import (
    NeuroSymbolicIntegrator,
    NeuralSymbolicState,
    ReasoningRule,
    MathematicalDerivation
)
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════



def test_all_derivations():
    """Test all mathematical derivations."""
    print("\n" + "="*80)
    print("TEST 1: MATHEMATICAL DERIVATIONS")
    print("="*80)

    integrator = NeuroSymbolicIntegrator(embedding_dim=64)

    # Perform all derivations
    derivations = [
        integrator.derive_integration_theorem(),
        integrator.derive_modus_ponens(),
        integrator.derive_neural_activation_gradient(),
        integrator.derive_symbolic_embedding_projection(),
        integrator.derive_knowledge_graph_diffusion(),
        integrator.derive_attention_mechanism(),
        integrator.derive_logical_consistency()
    ]

    # Verify all
    all_verified = all(d.verification_result for d in derivations)

    print(f"\n✓ All derivations verified: {all_verified}")
    print(f"✓ Total derivations: {len(derivations)}")

    assert all_verified, "Some derivations failed verification"
    return integrator


def test_symbolic_verification():
    """Test SymPy symbolic verification."""
    print("\n" + "="*80)
    print("TEST 2: SYMPY VERIFICATION")
    print("="*80)

    import sympy as sp

    # Test symbolic manipulation
    x = sp.Symbol('x')

    # Test 1: Derivative verification
    f = sp.sin(x) * sp.exp(x)
    df = sp.diff(f, x)
    expected = sp.sin(x) * sp.exp(x) + sp.cos(x) * sp.exp(x)

    verified = sp.simplify(df - expected) == 0
    print(f"✓ Derivative verification: {verified}")
    assert verified

    # Test 2: Integration verification
    integral = sp.integrate(2*x, x)
    expected_int = x**2
    verified_int = sp.simplify(integral - expected_int) == 0
    print(f"✓ Integration verification: {verified_int}")
    assert verified_int

    # Test 3: Logic verification
    from sympy.logic.boolalg import And, Or, Not, Implies
    from sympy.logic.inference import satisfiable

    P, Q = sp.symbols('P Q', bool=True)

    # Test modus ponens
    premise1 = Implies(P, Q)
    premise2 = P
    conclusion = Q

    # Check if premises imply conclusion
    formula = And(premise1, premise2, Not(conclusion))
    is_valid = satisfiable(formula) is False

    print(f"✓ Modus ponens validity: {is_valid}")
    assert is_valid

    print("\n✓ All SymPy verifications passed")


def test_neural_symbolic_inference():
    """Test neural-symbolic inference."""
    print("\n" + "="*80)
    print("TEST 3: NEURAL-SYMBOLIC INFERENCE")
    print("="*80)

    integrator = NeuroSymbolicIntegrator(embedding_dim=32)

    # Test embedding
    embedding1 = integrator.embed_symbol("human")
    embedding2 = integrator.embed_symbol("mortal")

    print(f"✓ Embedding dimension: {embedding1.shape[0]}")
    print(f"✓ Embedding normalized: {np.allclose(np.linalg.norm(embedding1), 1.0)}")

    assert embedding1.shape[0] == 32
    assert np.allclose(np.linalg.norm(embedding1), 1.0)

    # Test state creation
    state = NeuralSymbolicState(
        neural_embedding=embedding1,
        symbolic_facts=["human(socrates)", "mortal(X) :- human(X)"],
        confidence=0.95
    )

    print(f"✓ State confidence: {state.confidence}")
    assert state.confidence == 0.95

    # Test rule-based inference
    rule = ReasoningRule(
        premise="human(X)",
        conclusion="mortal(X)",
        neural_weight=0.9,
        symbolic_confidence=1.0
    )

    confidence = integrator.neural_inference(state, rule)
    print(f"✓ Inference confidence: {confidence:.4f}")
    assert 0.0 <= confidence <= 1.0

    print("\n✓ Neural-symbolic inference tests passed")


def test_latex_export():
    """Test LaTeX export functionality."""
    print("\n" + "="*80)
    print("TEST 4: LATEX EXPORT")
    print("="*80)

    integrator = NeuroSymbolicIntegrator(embedding_dim=16)

    # Perform a few derivations
    integrator.derive_integration_theorem()
    integrator.derive_modus_ponens()

    # Export to LaTeX
    output_file = "/workspaces/Allentown-L104-Node/test_derivations.tex"
    integrator.export_latex_document(output_file)

    # Verify file exists and has content
    with open(output_file, 'r') as f:
        content = f.read()

    print(f"✓ LaTeX file created: {output_file}")
    print(f"✓ LaTeX file size: {len(content)} bytes")

    assert "\\documentclass{article}" in content
    assert "\\begin{document}" in content
    assert "\\end{document}" in content

    print("\n✓ LaTeX export test passed")


def test_integration_complete():
    """Complete integration test."""
    print("\n" + "="*80)
    print("TEST 5: COMPLETE INTEGRATION")
    print("="*80)

    integrator = NeuroSymbolicIntegrator(embedding_dim=128)

    # Perform all derivations
    integrator.derive_integration_theorem()
    integrator.derive_modus_ponens()
    integrator.derive_neural_activation_gradient()
    integrator.derive_symbolic_embedding_projection()
    integrator.derive_knowledge_graph_diffusion()
    integrator.derive_attention_mechanism()
    integrator.derive_logical_consistency()

    # Verify all
    results = integrator.verify_all_derivations()

    # Export
    integrator.export_latex_document("/workspaces/Allentown-L104-Node/complete_derivations.tex")

    # Test symbolic reasoning
    embedding = integrator.embed_symbol("philosopher")
    state = NeuralSymbolicState(
        neural_embedding=embedding,
        symbolic_facts=["philosopher(plato)", "wise(X) :- philosopher(X)"],
        confidence=0.98
    )

    rule = ReasoningRule(
        premise="philosopher(plato)",
        conclusion="wise(plato)",
        neural_weight=0.95,
        symbolic_confidence=0.99
    )

    confidence = integrator.neural_inference(state, rule)

    print(f"\n✓ Complete integration successful")
    print(f"✓ All derivations verified: {all(results.values())}")
    print(f"✓ Inference confidence: {confidence:.4f}")

    assert all(results.values())
    assert 0.0 <= confidence <= 1.0

    return integrator


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*80)
    print("L104 NEURO-SYMBOLIC INTEGRATION TEST SUITE")
    print("="*80)

    try:
        test_all_derivations()
        test_symbolic_verification()
        test_neural_symbolic_inference()
        test_latex_export()
        integrator = test_integration_complete()

        print("\n" + "="*80)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*80)
        print("\nSUMMARY:")
        print(f"  - Mathematical derivations: 7/7 verified")
        print(f"  - SymPy verification: PASSED")
        print(f"  - Neural-symbolic inference: PASSED")
        print(f"  - LaTeX export: PASSED")
        print(f"  - Complete integration: PASSED")
        print("\n" + "="*80)

        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
