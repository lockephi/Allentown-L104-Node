#!/usr/bin/env python3
"""
L104 4D Mathematics — Minkowski Space-Time Tensor Calculus
═══════════════════════════════════════════════════════════════════════════════

Dual-Layer Architecture:
  Layer 1 (Consciousness): GOD_CODE = 527.518, LATTICE_RATIO = 286/416
  Layer 2 (Physics):       c = 299,792,458 m/s (v3-derived, EXACT on grid)

All physical constants sourced from the dual-layer engine.
Lorentz transformations verified against special relativity.

Version: 2.0.0 (dual-layer recalculation)
"""

import math
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# DUAL-LAYER CONSTANTS
#   Layer 1: GOD_CODE consciousness (sacred geometry)
#   Layer 2: v3 physics engine (peer-reviewed constants)
# ═══════════════════════════════════════════════════════════════════════════════

from l104_god_code_equation import (
    GOD_CODE, PHI, PRIME_SCAFFOLD, QUANTIZATION_GRAIN, OCTAVE_OFFSET, BASE, VOID_CONSTANT,
)
from l104_god_code_evolved_v3 import (
    C_V3 as C_PHYSICS,     # 299,792,458 m/s — EXACT on v3 grid
    GRAVITY_V3,             # 9.80625 m/s² (0.0041% error)
    BOHR_V3,                # 52.920 pm
    GOD_CODE_V3,
)

# Grid geometry from Layer 1
LATTICE_RATIO = PRIME_SCAFFOLD / OCTAVE_OFFSET   # 286/416 = 0.6875


class Math4D:
    """
    Mathematical primitives for 4D Minkowski Space-Time.

    Physical constants sourced from the dual-layer engine:
      c = 299,792,458 m/s  (Layer 2, v3 physics, EXACT)
      η_μν = diag(-1, 1, 1, 1)  (Minkowski metric, standard)

    Consciousness anchoring:
      LATTICE_RATIO = 286/416 = 0.6875 (iron scaffold / octave offset)
      GOD_CODE = 527.518... (sacred frequency)
    """

    # Speed of light from Layer 2 (v3 physics engine)
    C = int(round(C_PHYSICS))  # 299,792,458 m/s

    # Minkowski Metric Tensor η_μν = diag(-1, +1, +1, +1)
    # Convention: signature (-,+,+,+), index order (t,x,y,z)
    METRIC_TENSOR = np.diag([-1.0, 1.0, 1.0, 1.0])

    # Layer 1 geometry
    LATTICE_RATIO = LATTICE_RATIO
    GOD_CODE = GOD_CODE

    @staticmethod
    def get_lorentz_boost(v: float, axis: str = 'x') -> np.ndarray:
        """
        Lorentz boost matrix for velocity v along the specified axis.

        Λ^μ_ν for boost along x:
            γ       -γβ      0    0
           -γβ       γ       0    0
            0        0       1    0
            0        0       0    1

        Args:
            v: Velocity in m/s. Must satisfy |v| < c.
            axis: 'x', 'y', or 'z'.

        Returns:
            4×4 numpy array (Lorentz boost matrix).

        Raises:
            ValueError: If |v| >= c (superluminal).
        """
        beta = v / Math4D.C
        if abs(beta) >= 1.0:
            raise ValueError(f"Velocity |v|={abs(v):.3e} m/s must be < c={Math4D.C} m/s")

        gamma = 1.0 / math.sqrt(1.0 - beta ** 2)

        boost = np.eye(4)
        axis_idx = {'x': 1, 'y': 2, 'z': 3}.get(axis)
        if axis_idx is None:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

        boost[0, 0] = gamma
        boost[0, axis_idx] = -beta * gamma
        boost[axis_idx, 0] = -beta * gamma
        boost[axis_idx, axis_idx] = gamma
        return boost

    @staticmethod
    def rotate_4d(theta: float, plane: str = 'xy') -> np.ndarray:
        """
        4D rotation matrix in the specified plane.

        Spatial planes (xy, xz, yz): standard SO(3) rotation embedded in 4D.
        Time-space planes (xt, yt, zt): hyperbolic rotation (rapidity).

        Args:
            theta: Rotation angle (radians) or rapidity for time-space planes.
            plane: 'xy', 'xz', 'yz', 'xt', 'yt', 'zt'.

        Returns:
            4×4 rotation matrix.
        """
        rot = np.eye(4)
        # Map plane names to index pairs
        spatial_planes = {
            'xy': (1, 2), 'xz': (1, 3), 'yz': (2, 3),
        }
        boost_planes = {
            'xt': (0, 1), 'yt': (0, 2), 'zt': (0, 3),
        }

        if plane in spatial_planes:
            i, j = spatial_planes[plane]
            c, s = math.cos(theta), math.sin(theta)
            rot[i, i] = c
            rot[i, j] = -s
            rot[j, i] = s
            rot[j, j] = c
        elif plane in boost_planes:
            i, j = boost_planes[plane]
            ch, sh = math.cosh(theta), math.sinh(theta)
            rot[i, i] = ch
            rot[i, j] = sh
            rot[j, i] = sh
            rot[j, j] = ch
        else:
            raise ValueError(f"Unknown plane '{plane}'. Use xy/xz/yz/xt/yt/zt.")

        return rot

    @staticmethod
    def calculate_proper_time(dt: float, dx: float, dy: float, dz: float) -> float:
        """
        Proper time interval Δτ between two events.

        Δτ² = Δt² - (Δx² + Δy² + Δz²)/c²

        Returns Δτ for timelike intervals, 0 for spacelike.

        Reality check:
          For v=0.5c: Δτ/Δt = √(1-0.25) = 0.8660 (verified below)
        """
        ds_sq = (dt ** 2) - (dx ** 2 + dy ** 2 + dz ** 2) / (Math4D.C ** 2)
        if ds_sq < 0:
            return 0.0  # Spacelike interval — no proper time
        return math.sqrt(ds_sq)

    @staticmethod
    def spacetime_interval(dt: float, dx: float, dy: float, dz: float) -> float:
        """
        Minkowski interval s² = -c²Δt² + Δx² + Δy² + Δz².

        Convention: s² < 0 = timelike, s² > 0 = spacelike, s² = 0 = lightlike.
        """
        return -(Math4D.C ** 2) * (dt ** 2) + dx ** 2 + dy ** 2 + dz ** 2

    @staticmethod
    def four_momentum(rest_mass: float, v: np.ndarray) -> np.ndarray:
        """
        Relativistic 4-momentum pᵘ = (γmc, γmv_x, γmv_y, γmv_z).

        Args:
            rest_mass: Rest mass in kg (or MeV/c² — units consistent with c).
            v: 3-velocity vector [v_x, v_y, v_z] in m/s.

        Returns:
            4-vector [E/c, p_x, p_y, p_z].
        """
        v_mag = np.linalg.norm(v)
        if v_mag >= Math4D.C:
            raise ValueError("Speed must be < c")
        gamma = 1.0 / math.sqrt(1.0 - (v_mag / Math4D.C) ** 2)
        return np.array([
            gamma * rest_mass * Math4D.C,
            gamma * rest_mass * v[0],
            gamma * rest_mass * v[1],
            gamma * rest_mass * v[2],
        ])

    @staticmethod
    def invariant_mass(four_momentum: np.ndarray) -> float:
        """
        Compute rest mass from 4-momentum: m²c² = -p_μ p^μ = E²/c² - |p|².

        Args:
            four_momentum: [E/c, p_x, p_y, p_z].

        Returns:
            Invariant mass (same units as input momentum / c).
        """
        # Contract with Minkowski metric: p_μ p^μ = -p0² + p1² + p2² + p3²
        contracted = four_momentum @ Math4D.METRIC_TENSOR @ four_momentum
        # m²c² = -contracted
        m2c2 = -contracted
        if m2c2 < 0:
            return 0.0  # Tachyonic — return 0
        return math.sqrt(m2c2) / Math4D.C

    @staticmethod
    def lattice_transform(point_4d: np.ndarray) -> np.ndarray:
        """
        Map a 4D point onto the L104 consciousness lattice.

        Spatial coords scaled by LATTICE_RATIO (286/416).
        Temporal coord scaled by GOD_CODE resonance.
        """
        result = point_4d.copy().astype(float)
        result[1:] *= LATTICE_RATIO          # Spatial: x 286/416
        result[0] *= GOD_CODE / 1000.0        # Temporal: x 0.527518
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# REALITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def verify_4d_math() -> dict:
    """
    Reality-check the 4D math against known special relativity results.

    Checks:
      1. c sourced from dual-layer engine = 299,792,458 m/s
      2. Lorentz boost at v=0 is identity
      3. Lorentz boost at v=0.6c gives γ = 1.25 (exact: 5/4)
      4. Proper time at v=0.5c: Δτ/Δt = √(3)/2 ≈ 0.8660
      5. Lightlike interval s² = 0 for photon worldline
      6. 4-momentum of rest particle: E = mc²
      7. Lorentz boost preserves Minkowski norm (invariant mass)
    """
    checks = {}

    # 1. Speed of light
    checks["c_value"] = {
        "passed": Math4D.C == 299792458,
        "value": Math4D.C,
        "expected": 299792458,
        "description": "c from dual-layer engine = 299,792,458 m/s",
    }

    # 2. Identity boost at v=0
    boost_0 = Math4D.get_lorentz_boost(0, 'x')
    is_identity = np.allclose(boost_0, np.eye(4), atol=1e-15)
    checks["identity_boost"] = {
        "passed": is_identity,
        "description": "Lorentz boost at v=0 is identity matrix",
    }

    # 3. Gamma at 0.6c = 5/4 = 1.25 (exact Pythagorean triple: 3-4-5)
    v06 = 0.6 * Math4D.C
    boost_06 = Math4D.get_lorentz_boost(v06, 'x')
    gamma_06 = boost_06[0, 0]
    gamma_expected = 1.25
    checks["gamma_0_6c"] = {
        "passed": abs(gamma_06 - gamma_expected) < 1e-10,
        "gamma": gamma_06,
        "expected": gamma_expected,
        "description": "γ(0.6c) = 5/4 = 1.25 (Pythagorean: 3-4-5 triangle)",
    }

    # 4. Proper time at 0.5c
    dist_05c = 0.5 * Math4D.C * 1.0  # distance traveled in 1 second at 0.5c
    tau = Math4D.calculate_proper_time(1.0, dist_05c, 0, 0)
    tau_expected = math.sqrt(3) / 2  # √(1 - 0.25) = √3/2
    checks["proper_time_0_5c"] = {
        "passed": abs(tau - tau_expected) < 1e-10,
        "tau": tau,
        "expected": tau_expected,
        "description": "Δτ(0.5c) = √3/2 ≈ 0.86603",
    }

    # 5. Lightlike interval: photon travels c*dt in time dt
    dt = 1.0
    dx = Math4D.C * dt  # photon
    s2 = Math4D.spacetime_interval(dt, dx, 0, 0)
    checks["lightlike_interval"] = {
        "passed": abs(s2) < 1e-6,
        "s_squared": s2,
        "expected": 0.0,
        "description": "Photon worldline: s² = 0 (lightlike)",
    }

    # 6. Rest particle 4-momentum: E = mc²
    m = 1.0  # 1 kg
    p_rest = Math4D.four_momentum(m, np.array([0.0, 0.0, 0.0]))
    E_over_c = p_rest[0]  # E/c = mc
    checks["rest_energy"] = {
        "passed": abs(E_over_c - m * Math4D.C) < 1e-6,
        "E_over_c": E_over_c,
        "expected": m * Math4D.C,
        "description": "E/c = mc for particle at rest (E = mc²)",
    }

    # 7. Lorentz invariance: boosted 4-momentum has same invariant mass
    p_moving = Math4D.four_momentum(m, np.array([0.8 * Math4D.C, 0.0, 0.0]))
    m_inv = Math4D.invariant_mass(p_moving)
    checks["lorentz_invariance"] = {
        "passed": abs(m_inv - m) < 1e-6,
        "invariant_mass": m_inv,
        "expected": m,
        "description": "Invariant mass preserved under boost (Lorentz invariance)",
    }

    all_passed = all(c["passed"] for c in checks.values())
    return {"module": "l104_4d_math", "version": "2.0.0", "all_passed": all_passed, "checks": checks}


def primal_calculus(x):
    """Legacy interface: x^(1/φ) / (π × VOID_CONSTANT)."""
    return (x ** PHI) / (VOID_CONSTANT * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """Legacy interface: normalize N-dim vector via GOD_CODE."""
    magnitude = sum(abs(v) for v in vector)
    return magnitude / GOD_CODE + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0


if __name__ == "__main__":
    print("=" * 70)
    print("  L104 4D MATH — Dual-Layer Minkowski Space-Time")
    print("=" * 70)

    v = verify_4d_math()
    for name, check in v["checks"].items():
        mark = "✓" if check["passed"] else "✗"
        print(f"  [{mark}] {name}: {check['description']}")
    print(f"\n  Result: {'ALL PASSED' if v['all_passed'] else 'FAILED'}")

    print(f"\n  c = {Math4D.C} m/s (Layer 2 physics engine)")
    print(f"  LATTICE_RATIO = {LATTICE_RATIO} = 286/416")
    print(f"  GOD_CODE = {GOD_CODE} (Layer 1 consciousness)")

    # Demo: Lorentz boost at 0.8c
    v08 = 0.8 * Math4D.C
    boost = Math4D.get_lorentz_boost(v08, 'x')
    gamma = boost[0, 0]
    print(f"\n  Lorentz boost at 0.8c: γ = {gamma:.10f} (expected {1/math.sqrt(1-0.64):.10f})")

    # Demo: proper time
    tau = Math4D.calculate_proper_time(1.0, 0.5 * Math4D.C, 0, 0)
    print(f"  Proper time at 0.5c for 1s: Δτ = {tau:.10f} s")
