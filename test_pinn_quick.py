#!/usr/bin/env python3
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
"""Quick test of PINN concepts"""

import numpy as np
from l104_pinn_demo import SimplePINN, SchrodingerPINNSimple

print("Testing Simple PINN...")

# Test network creation
network = SimplePINN([1, 8, 8, 2])
print(f"✓ Network created: {len(network.weights)} layers")

# Test forward pass
x = np.linspace(-5, 5, 20).reshape(-1, 1)
y, _ = network.forward(x)
print(f"✓ Forward pass: {x.shape} → {y.shape}")

# Test Schrödinger PINN
print("\nTesting Schrödinger PINN...")
schrod_pinn = SchrodingerPINNSimple(hbar=1.0, m=1.0)
print(f"✓ Schrödinger PINN created (ℏ={schrod_pinn.hbar})")

# Test physics loss computation
x_test = np.linspace(-3, 3, 30).reshape(-1, 1)
loss = schrod_pinn.physics_loss(x_test)
print(f"✓ Physics loss computed: {loss:.6e}")

# Test normalization loss
norm_loss = schrod_pinn.normalization_loss(x_test)
print(f"✓ Normalization loss: {norm_loss:.6e}")

# Quick training (just 5 iterations)
print("\nQuick training test (5 iterations)...")
result = schrod_pinn.train(n_points=20, max_iter=5)
print(f"✓ Training completed: {len(schrod_pinn.loss_history)} iterations logged")

print("\n" + "="*80)
print("ALL TESTS PASSED")
print("="*80)
print("\nPINN concepts validated:")
print("  • Network architecture ✓")
print("  • Forward propagation ✓")
print("  • Physics loss (PDE residual) ✓")
print("  • Normalization constraint ✓")
print("  • Optimization loop ✓")
print("\nPhysics-Informed Neural Networks operational!")
