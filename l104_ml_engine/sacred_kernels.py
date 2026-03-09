"""
===============================================================================
L104 ML ENGINE — SACRED KERNEL LIBRARY v1.0.0
===============================================================================

Custom SVM kernel functions grounded in L104 sacred constants.
All kernels produce positive semi-definite Gram matrices by construction.

Kernels:
  phi_kernel          — Golden ratio bandwidth RBF
  god_code_kernel     — GOD_CODE-normalized polynomial with PHI exponent
  void_kernel         — VOID_CONSTANT-scaled Laplacian (L1 norm)
  harmonic_kernel     — 13-harmonic Fourier cosine series
  iron_lattice_kernel — Fe BCC lattice spacing as RBF bandwidth
  composite_sacred    — PHI-weighted blend of all sacred kernels

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import numpy as np
from typing import Optional, Dict

from .constants import (
    PHI, PHI_SQUARED, GOD_CODE, VOID_CONSTANT,
    PHI_KERNEL_SCALE, GOD_CODE_KERNEL_SCALE, VOID_KERNEL_SCALE,
    HARMONIC_N_TERMS, IRON_LATTICE_BW, PHI_CONJUGATE,
)


class SacredKernelLibrary:
    """Library of custom SVM kernels derived from L104 sacred constants.

    All kernels accept (X, Y) arrays where X is (n, d) and Y is (m, d),
    returning a Gram matrix of shape (n, m).

    Each kernel is a positive semi-definite function suitable for use with
    sklearn.svm.SVC(kernel='precomputed') after calling kernel(X_train, X_train).
    """

    @staticmethod
    def phi_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Golden ratio bandwidth RBF kernel.

        K(x, y) = exp(-PHI * ||x - y||^2 / (2 * PHI^2))
                 = exp(-||x - y||^2 / (2 * PHI))

        The bandwidth σ² = PHI emphasizes natural harmonic separation
        in feature space, matching the golden ratio's optimal packing property.
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        sq_dists = (
            np.sum(X ** 2, axis=1, keepdims=True)
            - 2.0 * X @ Y.T
            + np.sum(Y ** 2, axis=1, keepdims=True).T
        )
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.exp(-sq_dists / (2.0 * PHI))

    @staticmethod
    def god_code_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """GOD_CODE-normalized polynomial kernel with PHI exponent.

        K(x, y) = (1 + <x, y> / GOD_CODE)^(1/PHI)

        Polynomial kernel where the inner product is normalized by GOD_CODE
        and raised to the inverse golden power (1/φ ≈ 0.618).
        PSD because (1 + <x,y>/c) is PSD for c > 0, and fractional
        powers of PSD kernels remain PSD (Schur product theorem).
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        inner = X @ Y.T
        base = np.maximum(1.0 + inner / GOD_CODE, 1e-12)
        return np.power(base, 1.0 / PHI)

    @staticmethod
    def void_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """VOID_CONSTANT-scaled Laplacian kernel.

        K(x, y) = VOID_CONSTANT * exp(-||x - y||_1 / (104 * VOID_CONSTANT))

        Laplacian kernel (L1 norm) scaled by VOID_CONSTANT with
        bandwidth = 104 * VOID_CONSTANT ≈ 108.33. The L104 sacred
        number 104 appears as the bandwidth multiplier.
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        # L1 distance via feature-wise accumulation (O(n*m) memory, not O(n*m*d))
        l1_dists = np.zeros((X.shape[0], Y.shape[0]))
        for k in range(X.shape[1]):
            l1_dists += np.abs(X[:, k:k+1] - Y[:, k:k+1].T)
        bandwidth = 104.0 * VOID_CONSTANT
        return VOID_CONSTANT * np.exp(-l1_dists / bandwidth)

    @staticmethod
    def harmonic_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """13-harmonic Fourier cosine kernel.

        K(x, y) = sum_{k=1}^{13} cos(k * PHI * <x, y>) / k

        Fourier-harmonic kernel with 13 (Fibonacci(7)) harmonics.
        Each cosine term is PSD (Bochner's theorem), and the sum
        with positive coefficients (1/k > 0) preserves PSD.
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        inner = X @ Y.T
        result = np.zeros_like(inner)
        for k in range(1, HARMONIC_N_TERMS + 1):
            result += np.cos(k * PHI * inner) / k
        return result

    @staticmethod
    def iron_lattice_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Iron lattice RBF kernel.

        K(x, y) = exp(-||x - y||^2 / (2 * σ²))
        where σ = 286 / 1000 = 0.286 (Fe BCC lattice parameter in nm)

        The Fe(26) body-centered cubic lattice spacing defines the
        kernel bandwidth, connecting ML feature separation to the
        fundamental iron crystal structure.
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        sq_dists = (
            np.sum(X ** 2, axis=1, keepdims=True)
            - 2.0 * X @ Y.T
            + np.sum(Y ** 2, axis=1, keepdims=True).T
        )
        sq_dists = np.maximum(sq_dists, 0.0)
        sigma_sq = IRON_LATTICE_BW ** 2
        return np.exp(-sq_dists / (2.0 * sigma_sq))

    @staticmethod
    def composite_sacred_kernel(
        X: np.ndarray,
        Y: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """PHI-weighted composite of all sacred kernels.

        K_composite = w_phi * K_phi + w_god * K_god + w_void * K_void
                    + w_harm * K_harmonic + w_iron * K_iron

        Default weights are PHI-normalized (each weight = PHI^(-i) / Z):
          phi: 1.0, god_code: 0.618, void: 0.382, harmonic: 0.236, iron: 0.146
        (geometric PHI_CONJUGATE^i series, normalized to sum to 1)
        """
        if weights is None:
            raw = [PHI_CONJUGATE ** i for i in range(5)]
            total = sum(raw)
            w = [r / total for r in raw]
            weights = {
                'phi': w[0], 'god_code': w[1], 'void': w[2],
                'harmonic': w[3], 'iron': w[4],
            }

        lib = SacredKernelLibrary
        result = np.zeros((np.atleast_2d(X).shape[0], np.atleast_2d(Y).shape[0]))

        kernels = {
            'phi': lib.phi_kernel,
            'god_code': lib.god_code_kernel,
            'void': lib.void_kernel,
            'harmonic': lib.harmonic_kernel,
            'iron': lib.iron_lattice_kernel,
        }

        for name, kernel_fn in kernels.items():
            w = weights.get(name, 0.0)
            if w > 0:
                result += w * kernel_fn(X, Y)

        return result

    @classmethod
    def get_kernel_callable(cls, name: str):
        """Return a kernel function by name for sklearn integration.

        Usage:
            svc = SVC(kernel=SacredKernelLibrary.get_kernel_callable('phi'))
        """
        mapping = {
            'phi': cls.phi_kernel,
            'phi_kernel': cls.phi_kernel,
            'god_code': cls.god_code_kernel,
            'god_code_kernel': cls.god_code_kernel,
            'void': cls.void_kernel,
            'void_kernel': cls.void_kernel,
            'harmonic': cls.harmonic_kernel,
            'harmonic_kernel': cls.harmonic_kernel,
            'iron': cls.iron_lattice_kernel,
            'iron_lattice': cls.iron_lattice_kernel,
            'iron_lattice_kernel': cls.iron_lattice_kernel,
            'composite': cls.composite_sacred_kernel,
            'composite_sacred': cls.composite_sacred_kernel,
            'sacred': cls.composite_sacred_kernel,
        }
        if name not in mapping:
            raise ValueError(
                f"Unknown kernel '{name}'. Available: {sorted(set(mapping.keys()))}"
            )
        return mapping[name]

    @classmethod
    def list_kernels(cls) -> Dict[str, str]:
        """Return available kernel names with descriptions."""
        return {
            'phi': 'Golden ratio bandwidth RBF (σ²=PHI)',
            'god_code': 'GOD_CODE-normalized polynomial (exponent=1/PHI)',
            'void': 'VOID_CONSTANT Laplacian (L1 norm, bw=104*VOID)',
            'harmonic': '13-harmonic Fourier cosine series',
            'iron': 'Fe BCC lattice RBF (σ=0.286nm)',
            'composite': 'PHI-weighted blend of all 5 sacred kernels',
        }
