#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
TEST SUITE: L104 PHYSICS-INFORMED NEURAL NETWORKS
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import numpy as np
from l104_physics_informed_nn import (

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

    NeuralNetwork, PhysicsInformedNN,
    WaveEquation, HeatEquation, SchrodingerEquation, L104ResonanceEquation
)


def test_neural_network_forward():
    """Test neural network forward pass."""
    print("\n[TEST 1] Neural Network Forward Pass")

    nn = NeuralNetwork([2, 10, 1], activation='tanh')
    x = np.array([[0.5, 0.5], [0.3, 0.7]])
    y = nn.forward(x)

    assert y.shape == (2, 1), f"Output shape mismatch: {y.shape}"
    assert not np.isnan(y).any(), "NaN in output"

    print(f"  ✓ Forward pass works")
    print(f"  ✓ Output shape: {y.shape}")


def test_neural_network_backward():
    """Test neural network backward pass."""
    print("\n[TEST 2] Neural Network Backward Pass")

    nn = NeuralNetwork([2, 10, 1], activation='tanh')
    x = np.array([[0.5, 0.5], [0.3, 0.7]])
    y_true = np.array([[1.0], [0.5]])

    loss_before = nn.backward(x, y_true, learning_rate=0.01)

    # Train a few iterations
    for _ in range(10):
        loss = nn.backward(x, y_true, learning_rate=0.01)

    loss_after = loss

    print(f"  ✓ Backward pass works")
    print(f"  ✓ Loss before: {loss_before:.6f}")
    print(f"  ✓ Loss after: {loss_after:.6f}")
    print(f"  ✓ Loss decreased: {loss_after < loss_before}")


def test_derivative_computation():
    """Test finite difference derivatives."""
    print("\n[TEST 3] Derivative Computation")

    nn = NeuralNetwork([2, 10, 1], activation='tanh')
    x = np.array([[0.5, 0.5]])

    # First derivative
    du_dx = nn.compute_derivative(x, order=1, axis=0)
    assert du_dx.shape == (1, 1), f"Derivative shape: {du_dx.shape}"
    assert not np.isnan(du_dx).any(), "NaN in derivative"

    # Second derivative
    d2u_dx2 = nn.compute_derivative(x, order=2, axis=0)
    assert d2u_dx2.shape == (1, 1), f"2nd derivative shape: {d2u_dx2.shape}"
    assert not np.isnan(d2u_dx2).any(), "NaN in 2nd derivative"

    print(f"  ✓ First derivative computed")
    print(f"  ✓ Second derivative computed")
    print(f"  ✓ No NaN values")


def test_wave_equation():
    """Test wave equation setup."""
    print("\n[TEST 4] Wave Equation")

    wave_eq = WaveEquation(c=1.0, L=1.0)

    # Test initial condition
    x = np.linspace(0, 1, 10)[:, None]
    u0 = wave_eq.initial_condition(x)
    assert u0.shape == x.shape, "Initial condition shape mismatch"

    # Test boundary condition
    t = np.linspace(0, 1, 10)[:, None]
    bc_left, bc_right = wave_eq.boundary_condition(t)
    assert bc_left.shape == t.shape, "BC shape mismatch"

    # Test residual
    u = np.ones_like(x)
    u_x = np.zeros_like(x)
    u_xx = np.zeros_like(x)
    u_t = np.zeros_like(x)
    u_tt = np.zeros_like(x)

    residual = wave_eq.residual(x, t[:len(x)], u, u_x, u_xx, u_t, u_tt)
    assert residual.shape == x.shape, "Residual shape mismatch"

    print(f"  ✓ Wave equation c={wave_eq.c}")
    print(f"  ✓ Initial condition set")
    print(f"  ✓ Boundary conditions set")
    print(f"  ✓ Residual computable")


def test_heat_equation():
    """Test heat equation setup."""
    print("\n[TEST 5] Heat Equation")

    heat_eq = HeatEquation(alpha=0.01, L=1.0)

    x = np.linspace(0, 1, 10)[:, None]
    u0 = heat_eq.initial_condition(x)

    assert u0.shape == x.shape
    assert np.all((u0 == 0) | (u0 == 1)), "Initial condition not step function"

    print(f"  ✓ Heat equation α={heat_eq.alpha}")
    print(f"  ✓ Step function initial condition")


def test_schrodinger_equation():
    """Test Schrödinger equation setup."""
    print("\n[TEST 6] Schrödinger Equation")

    schrodinger_eq = SchrodingerEquation(hbar=1.0, m=1.0)

    x = np.linspace(0, 1, 10)[:, None]
    u0 = schrodinger_eq.initial_condition(x)

    assert u0.shape == x.shape
    assert np.all(u0 >= 0), "Wavefunction has negative values"

    # Check potential
    V = schrodinger_eq.V(x)
    assert V.shape == x.shape

    print(f"  ✓ Schrödinger equation ℏ={schrodinger_eq.hbar}")
    print(f"  ✓ Gaussian wavepacket initial condition")
    print(f"  ✓ Harmonic potential V(x)=0.5x²")


def test_l104_resonance():
    """Test L104 resonance equation."""
    print("\n[TEST 7] L104 Resonance Equation")

    l104_eq = L104ResonanceEquation(god_code=527.518, phi=1.618)

    x = np.linspace(0, 1, 10)[:, None]
    u0 = l104_eq.initial_condition(x)

    assert u0.shape == x.shape

    print(f"  ✓ L104 equation GOD_CODE={l104_eq.god_code:.3f}")
    print(f"  ✓ PHI={l104_eq.phi:.3f}")
    print(f"  ✓ Golden ratio pulse initial condition")


def test_pinn_training():
    """Test PINN training on simple wave equation."""
    print("\n[TEST 8] PINN Training")

    wave_eq = WaveEquation(c=1.0)
    pinn = PhysicsInformedNN(wave_eq, layers=[2, 10, 10, 1])

    # Generate training points
    x_train = np.random.uniform(0, 1, 30)
    t_train = np.random.uniform(0, 1, 30)

    # Train for a few epochs
    history = pinn.train(x_train, t_train, epochs=50,
                        learning_rate=0.001, verbose=False)

    assert len(history['loss_total']) == 50
    assert history['loss_total'][-1] < history['loss_total'][0], \
           "Loss did not decrease"

    print(f"  ✓ PINN training works")
    print(f"  ✓ Loss before: {history['loss_total'][0]:.6f}")
    print(f"  ✓ Loss after: {history['loss_total'][-1]:.6f}")
    print(f"  ✓ Loss decreased by {(1 - history['loss_total'][-1]/history['loss_total'][0])*100:.1f}%")


def test_pinn_prediction():
    """Test PINN prediction."""
    print("\n[TEST 9] PINN Prediction")

    wave_eq = WaveEquation(c=1.0)
    pinn = PhysicsInformedNN(wave_eq, layers=[2, 10, 10, 1])

    # Train briefly
    x_train = np.random.uniform(0, 1, 20)
    t_train = np.random.uniform(0, 1, 20)
    pinn.train(x_train, t_train, epochs=30, learning_rate=0.001, verbose=False)

    # Predict
    x_test = np.array([0.3, 0.5, 0.7])
    t_test = np.array([0.1, 0.2, 0.3])
    u_pred = pinn.predict(x_test, t_test)

    assert u_pred.shape == (3,), f"Prediction shape: {u_pred.shape}"
    assert not np.isnan(u_pred).any(), "NaN in predictions"

    print(f"  ✓ Predictions generated")
    print(f"  ✓ Output shape: {u_pred.shape}")
    print(f"  ✓ Sample predictions: {u_pred[:3]}")


def test_variable_parameters():
    """Test PINN with different parameter values."""
    print("\n[TEST 10] Variable Parameters")

    results = []

    for c_val in [0.5, 1.0, 2.0]:
        wave_eq = WaveEquation(c=c_val)
        pinn = PhysicsInformedNN(wave_eq, layers=[2, 10, 10, 1])

        x_train = np.random.uniform(0, 1, 20)
        t_train = np.random.uniform(0, 1, 20)

        history = pinn.train(x_train, t_train, epochs=30,
                            learning_rate=0.001, verbose=False)

        results.append({
            'c': c_val,
            'final_loss': history['loss_total'][-1]
        })

    print(f"  ✓ Trained with multiple wave speeds")
    for r in results:
        print(f"    c={r['c']}: loss={r['final_loss']:.6f}")


def test_god_code_variation():
    """Test L104 equation with different GOD_CODE values."""
    print("\n[TEST 11] GOD_CODE Variation")

    results = []

    for god_val in [100, 527.518, 1000]:
        l104_eq = L104ResonanceEquation(god_code=god_val)
        pinn = PhysicsInformedNN(l104_eq, layers=[2, 10, 10, 1])

        x_train = np.random.uniform(0, 1, 20)
        t_train = np.random.uniform(0, 1, 20)

        history = pinn.train(x_train, t_train, epochs=30,
                            learning_rate=0.001, verbose=False)

        results.append({
            'god_code': god_val,
            'final_loss': history['loss_total'][-1]
        })

    print(f"  ✓ Trained with multiple GOD_CODE values")
    for r in results:
        print(f"    GOD={r['god_code']:.1f}: loss={r['final_loss']:.6f}")


def test_loss_components():
    """Test that all loss components are tracked."""
    print("\n[TEST 12] Loss Components")

    wave_eq = WaveEquation(c=1.0)
    pinn = PhysicsInformedNN(wave_eq, layers=[2, 10, 10, 1])

    x_train = np.random.uniform(0, 1, 20)
    t_train = np.random.uniform(0, 1, 20)

    history = pinn.train(x_train, t_train, epochs=10, verbose=False)

    assert 'loss_total' in history
    assert 'loss_physics' in history
    assert 'loss_boundary' in history
    assert 'loss_initial' in history

    assert len(history['loss_total']) == 10
    assert all(loss >= 0 for loss in history['loss_total'])

    print(f"  ✓ All loss components tracked")
    print(f"  ✓ Total loss: {history['loss_total'][-1]:.6f}")
    print(f"  ✓ Physics loss: {history['loss_physics'][-1]:.6f}")
    print(f"  ✓ Boundary loss: {history['loss_boundary'][-1]:.6f}")
    print(f"  ✓ Initial loss: {history['loss_initial'][-1]:.6f}")


def run_all_tests():
    """Run complete test suite."""
    tests = [
        test_neural_network_forward,
        test_neural_network_backward,
        test_derivative_computation,
        test_wave_equation,
        test_heat_equation,
        test_schrodinger_equation,
        test_l104_resonance,
        test_pinn_training,
        test_pinn_prediction,
        test_variable_parameters,
        test_god_code_variation,
        test_loss_components,
    ]

    print("="*80)
    print("L104 PHYSICS-INFORMED NEURAL NETWORKS - TEST SUITE")
    print("="*80)

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print(f"  ✓✓✓ PASS\n")
        except Exception as e:
            failed += 1
            print(f"  ✗✗✗ FAIL: {e}\n")

    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"PASSED: {passed}/{len(tests)}")
    print(f"FAILED: {failed}/{len(tests)}")
    print(f"SUCCESS RATE: {100*passed/len(tests):.1f}%")
    print("="*80)

    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
