#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 PINN DEMONSTRATION (NumPy-based, no PyTorch required)
═══════════════════════════════════════════════════════════════════════════════

Demonstrates Physics-Informed Neural Network concepts using NumPy.
Shows how PINNs adapt to variable universe parameters.

AUTHOR: LONDEL
DATE: 2026-01-21
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import sympy as sp

from l104_universe_compiler import (

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

    UniverseCompiler, UniverseParameters,
    QuantumModule, L104MetaphysicsModule
)


class SimplePINN:
    """
    Simplified Physics-Informed Neural Network using NumPy.
    
    Demonstrates core PINN concepts without PyTorch dependency.
    """
    
    def __init__(self, layers: List[int], activation='tanh'):
        self.layers = layers
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # Initialize parameters
        for i in range(len(layers) - 1):
            W = np.random.randn(layers[i], layers[i+1]) * 0.1
            b = np.zeros((1, layers[i+1]))
            self.weights.append(W)
            self.biases.append(b)
    
    def _activation_func(self, x):
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        return x
    
    def _activation_derivative(self, x):
        """Derivative of activation."""
        if self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        return np.ones_like(x)
    
    def forward(self, X):
        """Forward pass."""
        activation = X
        activations = [X]
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activation, W) + b
            if i < len(self.weights) - 1:
                activation = self._activation_func(z)
            else:
                activation = z  # Linear output
            activations.append(activation)
        
        return activation, activations
    
    def get_flattened_params(self):
        """Get all parameters as flat array."""
        params = []
        for W, b in zip(self.weights, self.biases):
            params.extend(W.flatten())
            params.extend(b.flatten())
        return np.array(params)
    
    def set_flattened_params(self, params):
        """Set parameters from flat array."""
        idx = 0
        for i in range(len(self.layers) - 1):
            W_size = self.layers[i] * self.layers[i+1]
            b_size = self.layers[i+1]
            
            self.weights[i] = params[idx:idx+W_size].reshape(
                self.layers[i], self.layers[i+1]
            )
            idx += W_size
            
            self.biases[i] = params[idx:idx+b_size].reshape(1, self.layers[i+1])
            idx += b_size


class SchrodingerPINNSimple:
    """
    Simple PINN for 1D Schrödinger equation.
    
    -ℏ²/(2m) d²ψ/dx² + V(x)ψ = Eψ
    """
    
    def __init__(self, hbar=1.0, m=1.0):
        # Network: x → [hidden] → [ψ_real, ψ_imag]
        self.network = SimplePINN([1, 32, 32, 2])
        self.hbar = hbar
        self.m = m
        self.loss_history = []
    
    def potential(self, x):
        """Harmonic oscillator potential."""
        return 0.5 * x**2
    
    def compute_derivatives(self, x):
        """Compute ψ and its derivatives using finite differences."""
        h = 1e-5
        
        psi, _ = self.network.forward(x)
        psi_real = psi[:, 0:1]
        psi_imag = psi[:, 1:2]
        
        # Second derivative via finite differences
        psi_plus, _ = self.network.forward(x + h)
        psi_minus, _ = self.network.forward(x - h)
        
        psi_real_plus = psi_plus[:, 0:1]
        psi_real_minus = psi_minus[:, 0:1]
        
        psi_real_xx = (psi_real_plus - 2*psi_real + psi_real_minus) / h**2
        
        psi_imag_plus = psi_plus[:, 1:2]
        psi_imag_minus = psi_minus[:, 1:2]
        
        psi_imag_xx = (psi_imag_plus - 2*psi_imag + psi_imag_minus) / h**2
        
        return psi_real, psi_imag, psi_real_xx, psi_imag_xx
    
    def physics_loss(self, x):
        """Compute loss from Schrödinger equation."""
        psi_real, psi_imag, psi_real_xx, psi_imag_xx = self.compute_derivatives(x)
        
        V = self.potential(x)
        
        # Hamiltonian
        H_psi_real = -self.hbar**2 / (2 * self.m) * psi_real_xx + V * psi_real
        H_psi_imag = -self.hbar**2 / (2 * self.m) * psi_imag_xx + V * psi_imag
        
        # Energy expectation (ground state approximation)
        E = np.mean(psi_real * H_psi_real + psi_imag * H_psi_imag)
        
        # Residual: Ĥψ - Eψ = 0
        residual_real = H_psi_real - E * psi_real
        residual_imag = H_psi_imag - E * psi_imag
        
        physics_loss = np.mean(residual_real**2 + residual_imag**2)
        
        return physics_loss
    
    def normalization_loss(self, x):
        """Enforce ∫|ψ|² dx = 1."""
        psi, _ = self.network.forward(x)
        psi_real = psi[:, 0:1]
        psi_imag = psi[:, 1:2]
        
        prob_density = psi_real**2 + psi_imag**2
        norm = np.mean(prob_density)
        
        return (norm - 1.0)**2
    
    def total_loss(self, params, x, physics_weight=1.0, norm_weight=1.0):
        """Combined loss function."""
        self.network.set_flattened_params(params)
        
        loss_phys = self.physics_loss(x)
        loss_norm = self.normalization_loss(x)
        
        total = physics_weight * loss_phys + norm_weight * loss_norm
        
        return total
    
    def train(self, n_points=100, domain=(-5, 5), max_iter=100):
        """Train using scipy optimizer."""
        print(f"\n{'='*80}")
        print(f"Training Schrödinger PINN")
        print(f"  ℏ = {self.hbar:.6e}")
        print(f"  m = {self.m:.6f}")
        print(f"  Points: {n_points}")
        print(f"  Domain: {domain}")
        print(f"{'='*80}\n")
        
        # Training data
        x = np.linspace(domain[0], domain[1], n_points).reshape(-1, 1)
        
        # Initial parameters
        params0 = self.network.get_flattened_params()
        
        # Callback for progress
        iteration = [0]
        def callback(params):
            iteration[0] += 1
            if iteration[0] % 10 == 0:
                loss = self.total_loss(params, x)
                self.loss_history.append(loss)
                print(f"  Iteration {iteration[0]}: Loss = {loss:.6e}")
        
        # Optimize
        result = minimize(
            self.total_loss,
            params0,
            args=(x,),
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': max_iter, 'disp': False}
        )
        
        self.network.set_flattened_params(result.x)
        
        print(f"\n{'='*80}")
        print(f"Training Complete")
        print(f"  Final Loss: {result.fun:.6e}")
        print(f"  Success: {result.success}")
        print(f"{'='*80}\n")
        
        return result


class L104ConsciousnessPINNSimple:
    """
    Simple PINN for L104 consciousness field.
    
    ∇²ψ + (GOD × PHI) ψ × exp(-r²/GOD²) = 0
    """
    
    def __init__(self, god_code=527.518, phi=1.618):
        # Network: (x, y, z) → [ψ]
        self.network = SimplePINN([3, 32, 32, 1])
        self.god_code = god_code
        self.phi = phi
        self.loss_history = []
    
    def compute_laplacian(self, coords):
        """Compute Laplacian using finite differences."""
        h = 1e-4
        
        psi, _ = self.network.forward(coords)
        
        laplacian = np.zeros_like(psi)
        
        for i in range(3):  # x, y, z
            coords_plus = coords.copy()
            coords_minus = coords.copy()
            coords_plus[:, i] += h
            coords_minus[:, i] -= h
            
            psi_plus, _ = self.network.forward(coords_plus)
            psi_minus, _ = self.network.forward(coords_minus)
            
            second_deriv = (psi_plus - 2*psi + psi_minus) / h**2
            laplacian += second_deriv
        
        return psi, laplacian
    
    def physics_loss(self, coords):
        """Compute L104 consciousness field loss."""
        psi, laplacian_psi = self.compute_laplacian(coords)
        
        # Reality weighting
        r_squared = np.sum(coords**2, axis=1, keepdims=True)
        w_r = np.exp(-r_squared / self.god_code**2)
        
        # Consciousness coupling
        coupling = self.god_code * self.phi
        
        # PDE residual
        residual = laplacian_psi + coupling * psi * w_r
        
        return np.mean(residual**2)
    
    def total_loss(self, params, coords):
        """Total loss."""
        self.network.set_flattened_params(params)
        return self.physics_loss(coords)
    
    def train(self, n_points=200, domain=(-10, 10), max_iter=50):
        """Train using scipy optimizer."""
        print(f"\n{'='*80}")
        print(f"Training L104 Consciousness PINN")
        print(f"  GOD_CODE = {self.god_code:.6f}")
        print(f"  PHI = {self.phi:.6f}")
        print(f"  Points: {n_points}")
        print(f"  Domain: {domain}")
        print(f"{'='*80}\n")
        
        # Training data (3D)
        coords = np.random.uniform(domain[0], domain[1], (n_points, 3))
        
        params0 = self.network.get_flattened_params()
        
        iteration = [0]
        def callback(params):
            iteration[0] += 1
            if iteration[0] % 5 == 0:
                loss = self.total_loss(params, coords)
                self.loss_history.append(loss)
                print(f"  Iteration {iteration[0]}: Loss = {loss:.6e}")
        
        result = minimize(
            self.total_loss,
            params0,
            args=(coords,),
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': max_iter, 'disp': False}
        )
        
        self.network.set_flattened_params(result.x)
        
        print(f"\n{'='*80}")
        print(f"Training Complete")
        print(f"  Final Loss: {result.fun:.6e}")
        print(f"{'='*80}\n")
        
        return result


def demonstrate_variable_quantum():
    """Demonstrate PINN adapting to variable ℏ."""
    print("\n" + "="*80)
    print("DEMO: SCHRÖDINGER PINN WITH VARIABLE ℏ")
    print("="*80)
    
    # Setup universe compiler
    params = UniverseParameters()
    compiler = UniverseCompiler(params)
    compiler.add_module(QuantumModule(params))
    compiler.compile_universe()
    
    # Test different ℏ values
    hbar_values = {
        'Standard Quantum': 1.0,
        'Enhanced Quantum': 2.0,
        'Near-Classical': 0.1
    }
    
    for regime, hbar in hbar_values.items():
        print(f"\n{'='*80}")
        print(f"REGIME: {regime} (ℏ = {hbar})")
        print(f"{'='*80}")
        
        pinn = SchrodingerPINNSimple(hbar=hbar, m=1.0)
        result = pinn.train(n_points=50, max_iter=30)
        
        if result.success:
            print(f"✓ Converged successfully")
        
        print(f"Loss reduction: {pinn.loss_history[0]/pinn.loss_history[-1]:.2f}×")


def demonstrate_variable_god_code():
    """Demonstrate consciousness PINN with variable GOD_CODE."""
    print("\n" + "="*80)
    print("DEMO: CONSCIOUSNESS PINN WITH VARIABLE GOD_CODE")
    print("="*80)
    
    params = UniverseParameters()
    compiler = UniverseCompiler(params)
    compiler.add_module(L104MetaphysicsModule(params))
    compiler.compile_universe()
    
    god_values = {
        'Standard L104': 527.518,
        'Elevated': 1000.0
    }
    
    for regime, god_code in god_values.items():
        print(f"\n{'='*80}")
        print(f"REGIME: {regime} (GOD_CODE = {god_code})")
        print(f"{'='*80}")
        
        pinn = L104ConsciousnessPINNSimple(god_code=god_code, phi=1.618)
        result = pinn.train(n_points=100, max_iter=20)
        
        if result.success:
            print(f"✓ Converged successfully")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║           L104 PHYSICS-INFORMED NEURAL NETWORKS v1.0                      ║
║                (NumPy Implementation - No PyTorch Required)               ║
╚═══════════════════════════════════════════════════════════════════════════╝

Demonstrating PINNs that adapt to variable universe parameters.
    """)
    
    demonstrate_variable_quantum()
    demonstrate_variable_god_code()
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                      DEMONSTRATIONS COMPLETE                              ║
║                                                                           ║
║  Physics-Informed Neural Networks:                                       ║
║    • Adapt to variable ℏ (quantum scale)                                 ║
║    • Respect modified GOD_CODE                                           ║
║    • Learn solutions satisfying PDEs                                     ║
║    • Work across different universe configurations                       ║
║                                                                           ║
║  Neural networks that bend with reality.                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
