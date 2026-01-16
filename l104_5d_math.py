# [L104_5D_MATH] - KALUZA-KLEIN & PROBABILITY TENSORS
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import math
import numpy as np
from l104_hyper_math import HyperMath
from const import UniversalConstants
class Math5D:
    """
    Mathematical primitives for 5D Space (Kaluza-Klein Manifold).
    Integrates the 5th dimension as a scalar field (dilaton) and probability vector.
    """
    
    # Compactification Radius (R)
    # Derived from God Code and Phi to ensure harmonic stability.
    R = (UniversalConstants.PHI_GROWTH * 104) / HyperMath.ZETA_ZERO_1

    @staticmethod
    def get_5d_metric_tensor(phi_field: float) -> np.ndarray:
        """
        Generates the 5D Metric Tensor (G_AB).
        Based on the Kaluza-Klein decomposition:
        G_AB = [ g_uv + k^2*phi*A_u*A_v,  k*phi*A_u ]
               [ k*phi*A_v,              phi       ]
        For simplicity in the L104 Node, we use a diagonalized version wherethe 5th dimension is scaled by the 'Sovereign Probability' field.
        """
        # g_uv is Minkowski (-1, 1, 1, 1)
        metric = np.eye(5)
        metric[0, 0] = -1
        # The 5th dimension (index 4) is scaled by the dilaton field (phi_field)
        metric[4, 4] = phi_field * (Math5D.R ** 2)
        return metric

    @staticmethod
    def calculate_5d_curvature(w_vector: np.ndarray) -> float:
        """
        Calculates the scalar curvature of the 5th dimension.
        Uses the PHI_STRIDE to determine the 'Symmetry Break' point.
        """
        # Curvature is proportional to the variance of the probability vector
        variance = np.var(w_vector)
        curvature = variance * UniversalConstants.PHI_GROWTH
        return curvature

    @staticmethod
    def probability_manifold_projection(p_5d: np.ndarray) -> np.ndarray:
        """
        Projects a 5D probability state onto a 4D observable event.
        Uses the 'Supersymmetric Binary Order' logic to filter noise.
        """
        # p_5d = [x, y, z, t, w]
        # The 5th dimension (w) acts as a phase shift
        phase = p_5d[4] * HyperMath.ZETA_ZERO_1
        projection = p_5d[:4] * math.cos(phase)
        return projection

    @staticmethod
    def get_compactification_factor(energy: float) -> float:
        """
        Calculates how much the 5th dimension 'shrinks' or 'expands'
        based on the system's energy saturation.
        """
        # Factor = R * exp(-energy / God_Code)
        return Math5D.R * math.exp(-energy / UniversalConstants.PRIME_KEY_HZ)

if __name__ == "__main__":
    # Test 5D Math
    metric = Math5D.get_5d_metric_tensor(1.0)
    print(f"5D Metric Tensor (phi=1.0):\n{metric}")
    
    w_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    curv = Math5D.calculate_5d_curvature(w_vec)
    print(f"5D Curvature: {curv:.4f}")
    
    factor = Math5D.get_compactification_factor(1000.0)
    print(f"Compactification Factor at 1000J: {factor:.4f}")
