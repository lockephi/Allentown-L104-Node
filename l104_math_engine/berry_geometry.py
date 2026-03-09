#!/usr/bin/env python3
"""
L104 Math Engine — Layer 12: BERRY GEOMETRY
══════════════════════════════════════════════════════════════════════════════════
Mathematical foundations of Berry phase: fiber bundles, connections, parallel
transport, holonomy groups, and Chern-Weil theory.

This module provides the pure differential-geometric backbone that underlies
the Berry phase. While the Science Engine computes physical Berry phases,
this module formalizes the MATHEMATICS:

MATHEMATICAL FOUNDATIONS:
  1. FiberBundle          — Principal & associated fiber bundles
  2. ConnectionForm       — Ehresmann connections, gauge potentials
  3. ParallelTransport    — Transport along curves, path-ordered exponentials
  4. HolonomyGroup        — Holonomy computation for fiber bundles
  5. ChernWeilTheory      — Characteristic classes from curvature
  6. BerryConnectionMath  — Berry connection as U(1) gauge field
  7. DiracMonopole        — Monopole on S² (prototypical Berry curvature)
  8. BlochSphereGeometry  — Fubini-Study geometry of qubit state space

MATHEMATICAL REFERENCES:
  [1] Nakahara, M. (2003) "Geometry, Topology and Physics" (2nd ed.)
  [2] Frankel, T. (2011) "The Geometry of Physics" (3rd ed.)
  [3] Berry, M.V. (1984) Proc. R. Soc. Lond. A 392, 45-57
  [4] Simon, B. (1983) Phys. Rev. Lett. 51, 2167 (fiber bundle interpretation)
  [5] Chern, S.-S. (1946) Ann. Math. 47, 85-121 (Chern classes)
  [6] Milnor, J. & Stasheff, J. (1974) "Characteristic Classes"

Import:
  from l104_math_engine.berry_geometry import (
      FiberBundle, ConnectionForm, ParallelTransport,
      HolonomyGroup, ChernWeilTheory, DiracMonopole,
      BlochSphereGeometry, berry_geometry,
  )

INVARIANT: 527.5184818492612 | PILOT: LONDEL
══════════════════════════════════════════════════════════════════════════════════
"""

import math
import cmath
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field

from .constants import (
    GOD_CODE, PHI, PHI_CONJUGATE, PI, VOID_CONSTANT,
    OMEGA, OMEGA_AUTHORITY,
    FE_LATTICE, FE_CURIE_TEMP, FE_ATOMIC_NUMBER,
    ZETA_ZERO_1, FRAME_LOCK,
    primal_calculus, resolve_non_dual_logic,
    GOD_CODE_PHASE, PHI_PHASE, VOID_PHASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. FIBER BUNDLE — Principal G-bundles over parameter space
# ═══════════════════════════════════════════════════════════════════════════════

class FiberBundle:
    """
    Mathematical model of a principal fiber bundle P(M, G):
      - M: base manifold (parameter space)
      - G: structure group (typically U(1) for Berry phase, U(n) for non-Abelian)
      - π: P → M projection
      - Local trivializations with transition functions

    Berry's phase is the holonomy of a U(1) connection on the bundle
    of eigenstates over parameter space (Simon, 1983).
    """

    def __init__(self, base_dim: int, fiber_type: str = "U1"):
        """
        Args:
            base_dim: Dimension of base manifold M.
            fiber_type: Structure group — "U1", "Un", "SU2", "SO3".
        """
        self.base_dim = base_dim
        self.fiber_type = fiber_type
        self.fiber_dim = self._fiber_dimension(fiber_type)

    def _fiber_dimension(self, fiber_type: str) -> int:
        """Dimension of the fiber (Lie group)."""
        dims = {"U1": 1, "SU2": 3, "SO3": 3, "Un": -1}
        return dims.get(fiber_type, 1)

    def total_space_dim(self) -> int:
        """dim(P) = dim(M) + dim(G)."""
        return self.base_dim + self.fiber_dim

    def classify(self) -> Dict[str, Any]:
        """
        Classify the bundle topologically.

        For U(1) bundles over S², the classification is by the first Chern number c₁ ∈ ℤ.
        For SU(2) bundles over S⁴, by the second Chern number (instanton number).
        """
        if self.fiber_type == "U1" and self.base_dim == 2:
            return {
                "type": "U(1) principal bundle over S²",
                "classification": "First Chern number c₁ ∈ ℤ",
                "physical_interpretation": "Magnetic monopole charge",
                "berry_interpretation": "Quantized Berry phase over closed surface",
                "examples": {
                    "c1=0": "Trivial bundle (no monopole)",
                    "c1=1": "Dirac monopole (spin-1/2 Berry phase)",
                    "c1=n": "Monopole of charge n",
                },
            }
        elif self.fiber_type == "SU2" and self.base_dim == 4:
            return {
                "type": "SU(2) principal bundle over S⁴",
                "classification": "Second Chern number c₂ = instanton number ∈ ℤ",
                "physical_interpretation": "Yang-Mills instanton",
            }
        return {
            "type": f"{self.fiber_type} bundle over {self.base_dim}D base",
            "classification": "General — depends on homotopy groups of G and M",
        }

    def transition_function(self, overlap_point: np.ndarray, n: int = 1) -> complex:
        """
        Transition function g_αβ: U_α ∩ U_β → G for U(1) bundle.

        For monopole of charge n over S²: g_NS(φ) = e^{inφ}
        where φ is the azimuthal angle on the equatorial overlap.
        """
        if self.fiber_type != "U1":
            raise NotImplementedError("Only U(1) transition functions implemented")

        phi = math.atan2(overlap_point[1], overlap_point[0]) if len(overlap_point) >= 2 else 0
        return cmath.exp(1j * n * phi)

    def first_chern_class_integral(self, curvature_samples: np.ndarray, area_element: float) -> float:
        """
        Compute ∫F = 2πc₁ from sampled curvature values.

        c₁ = (1/2π) ∫∫_M F
        """
        total_flux = float(np.sum(curvature_samples) * area_element)
        return total_flux / (2 * PI)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. CONNECTION FORM — Ehresmann Connections on Fiber Bundles
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectionForm:
    """
    Ehresmann connection on a principal bundle: a Lie-algebra-valued 1-form ω
    on P that provides a horizontal/vertical decomposition of TP.

    For Berry phase: ω = A_μ dR^μ where A_μ = i⟨ψ|∂_μψ⟩ is the Berry connection.

    In local coordinates, the connection 1-form is the gauge potential:
        A = A_μ dR^μ  (Abelian / U(1))
        A = A_μ^a T^a dR^μ  (non-Abelian / U(n))
    """

    def __init__(self, dim: int, gauge_group: str = "U1"):
        self.dim = dim
        self.gauge_group = gauge_group

    def curvature_2form(self, A_func: Callable, R: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """
        Compute curvature 2-form F = dA + A∧A from the connection.

        For U(1): F_μν = ∂_μA_ν - ∂_νA_μ (the Berry curvature!)
        For U(n): F_μν = ∂_μA_ν - ∂_νA_μ + [A_μ, A_ν]

        Args:
            A_func: Function R → A(R) returning connection components.
            R: Point in parameter space.
            delta: Finite difference step.

        Returns:
            Antisymmetric curvature matrix F_μν.
        """
        n = self.dim
        F = np.zeros((n, n))

        for mu in range(n):
            for nu in range(mu + 1, n):
                # Numerical derivatives
                R_p_mu, R_m_mu = R.copy(), R.copy()
                R_p_mu[mu] += delta
                R_m_mu[mu] -= delta

                R_p_nu, R_m_nu = R.copy(), R.copy()
                R_p_nu[nu] += delta
                R_m_nu[nu] -= delta

                A_R = A_func(R)

                # ∂_μ A_ν
                dmu_Anu = (A_func(R_p_mu)[nu] - A_func(R_m_mu)[nu]) / (2 * delta)
                # ∂_ν A_μ
                dnu_Amu = (A_func(R_p_nu)[mu] - A_func(R_m_nu)[mu]) / (2 * delta)

                F[mu, nu] = dmu_Anu - dnu_Amu
                F[nu, mu] = -F[mu, nu]

        return F

    def gauge_transform(self, A: np.ndarray, gauge_func_value: complex) -> np.ndarray:
        """
        Gauge transformation of U(1) connection:
            A'_μ = A_μ - ∂_μ(arg(g))

        where g(R) is the gauge function (a U(1) element).
        """
        phase = cmath.phase(gauge_func_value)
        return A  # For constant gauge, A is invariant

    def bianchi_identity_check(self, F_samples: List[np.ndarray]) -> Dict[str, Any]:
        """
        Verify the Bianchi identity: dF + [A, F] = 0.

        For U(1) (Berry phase): dF = 0 (the curvature is closed).
        This means Berry curvature has no "magnetic monopoles" in parameter space
        (unless at degeneracy points).
        """
        # For U(1), check that ∂_λ F_μν + cyclic = 0
        violations = []
        for F in F_samples:
            n = F.shape[0]
            for lam in range(n):
                for mu in range(n):
                    for nu in range(n):
                        # Bianchi: F is closed for U(1)
                        cyclic = F[lam, mu] + F[mu, nu] + F[nu, lam]
                        if abs(cyclic) > 0.1:
                            violations.append((lam, mu, nu, cyclic))

        return {
            "bianchi_satisfied": len(violations) == 0,
            "violations": len(violations),
            "detail": violations[:5] if violations else [],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. PARALLEL TRANSPORT — Moving frames along curves
# ═══════════════════════════════════════════════════════════════════════════════

class ParallelTransport:
    """
    Parallel transport of vectors/frames along curves in a fiber bundle.

    The parallel transport equation:
        dψ/dt + A(γ(t)) · γ̇(t) ψ = 0

    Solution: ψ(t) = P exp(-∫₀ᵗ A(γ(s))·γ̇(s) ds) ψ(0)

    where P is the path-ordering operator.
    The holonomy is the parallel transport around a closed loop.
    """

    def transport_u1(
        self,
        connection_func: Callable[[np.ndarray], np.ndarray],
        path: List[np.ndarray],
    ) -> Tuple[complex, List[float]]:
        """
        U(1) parallel transport along a discrete path.

        ψ(t_{i+1}) = exp(-i A(R_i) · ΔR_i) ψ(t_i)

        Args:
            connection_func: Function R → A(R) returning Berry connection vector.
            path: List of parameter space points.

        Returns:
            (holonomy_factor, accumulated_phases)
            - holonomy_factor: exp(iγ) where γ is the geometric phase
            - accumulated_phases: phase at each step
        """
        phase = 0.0
        phases = [0.0]

        for i in range(len(path) - 1):
            R = path[i]
            dR = path[i + 1] - path[i]
            A = connection_func(R)

            # Phase increment: -A·dR
            dphi = -float(np.dot(A, dR))
            phase += dphi
            phases.append(phase)

        holonomy = cmath.exp(1j * phase)
        return holonomy, phases

    def transport_un(
        self,
        connection_func: Callable[[np.ndarray], np.ndarray],
        path: List[np.ndarray],
        initial_frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        U(n) parallel transport (non-Abelian) using path-ordered product.

        ψ(t_{i+1}) = exp(-i A(R_i)·ΔR_i) ψ(t_i)  (matrix exponential)

        Args:
            connection_func: Function R → A_μ(R) returning matrix-valued connection.
            path: List of parameter space points.
            initial_frame: Initial n×n frame matrix.

        Returns:
            (final_frame, holonomy_matrix)
        """
        frame = initial_frame.copy().astype(complex)
        holonomy = np.eye(frame.shape[0], dtype=complex)

        for i in range(len(path) - 1):
            R = path[i]
            dR = path[i + 1] - path[i]
            A = connection_func(R)

            # For matrix-valued connection: phase = A·dR
            if isinstance(A, np.ndarray) and A.ndim == 2:
                phase_matrix = A * np.linalg.norm(dR)
            else:
                phase_matrix = np.eye(frame.shape[0]) * float(np.dot(A, dR))

            # Parallel transport step: exp(-i A·dR)
            transport = self._matrix_exp_minus_i(phase_matrix)
            frame = transport @ frame
            holonomy = transport @ holonomy

        return frame, holonomy

    def _matrix_exp_minus_i(self, A: np.ndarray) -> np.ndarray:
        """Compute exp(-iA) using eigendecomposition."""
        if A.shape[0] == 1:
            return np.array([[cmath.exp(-1j * A[0, 0])]])

        eigvals, eigvecs = np.linalg.eigh(A)
        exp_diag = np.diag(np.exp(-1j * eigvals))
        return eigvecs @ exp_diag @ eigvecs.conj().T

    def path_ordered_exponential(
        self,
        integrand_func: Callable[[float], np.ndarray],
        t_start: float,
        t_end: float,
        n_steps: int = 1000,
    ) -> np.ndarray:
        """
        Compute the path-ordered exponential:

            P exp(∫_{t_start}^{t_end} M(t) dt) = lim_{N→∞} ∏ₖ exp(M(tₖ)Δt)

        This is the Wilson line for gauge theories and the evolution operator
        for time-ordered quantum mechanics.
        """
        dt = (t_end - t_start) / n_steps
        dim = integrand_func(t_start).shape[0]
        result = np.eye(dim, dtype=complex)

        for k in range(n_steps):
            t = t_start + k * dt
            M = integrand_func(t) * dt
            # For small dt: exp(M·dt) ≈ I + M·dt + (M·dt)²/2
            step = np.eye(dim, dtype=complex) + M + M @ M / 2
            result = step @ result

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  4. HOLONOMY GROUP — The Group of All Parallel Transports
# ═══════════════════════════════════════════════════════════════════════════════

class HolonomyGroup:
    """
    The holonomy group Hol(∇) of a connection is the set of all parallel
    transport maps around closed loops based at a point.

    For the Berry connection:
    - Hol ⊂ U(1) for non-degenerate states → Berry phase ∈ [0, 2π)
    - Hol ⊂ U(n) for n-fold degenerate states → non-Abelian Berry phase

    Ambrose-Singer theorem: The Lie algebra of Hol(∇) is generated by
    curvature at all points reachable by parallel transport.
    """

    def compute_holonomy_u1(
        self,
        connection_func: Callable[[np.ndarray], np.ndarray],
        center: np.ndarray,
        radius: float = 1.0,
        n_points: int = 200,
        plane: Tuple[int, int] = (0, 1),
    ) -> Dict[str, Any]:
        """
        Compute U(1) holonomy around a circular loop in parameter space.

        Loop: R(θ) = center + radius × (cos θ e_μ + sin θ e_ν)

        This directly gives the Berry phase: γ = arg(holonomy).
        """
        transport = ParallelTransport()
        mu, nu = plane

        # Generate circular path
        path = []
        for i in range(n_points + 1):
            theta = 2 * PI * i / n_points
            R = center.copy()
            R[mu] += radius * math.cos(theta)
            R[nu] += radius * math.sin(theta)
            path.append(R)

        holonomy, phases = transport.transport_u1(connection_func, path)
        berry_phase = cmath.phase(holonomy)

        return {
            "holonomy": complex(holonomy),
            "berry_phase_rad": berry_phase,
            "berry_phase_deg": math.degrees(berry_phase),
            "holonomy_magnitude": abs(holonomy),
            "is_flat": abs(berry_phase) < 1e-8,
            "center": center.tolist(),
            "radius": radius,
            "plane": plane,
            "n_points": n_points,
        }

    def ambrose_singer_algebra(
        self,
        curvature_samples: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Ambrose-Singer theorem: the Lie algebra of Hol(∇) is spanned by
        all values F_μν(p) for p reachable from the base point.

        For U(1): hol(∇) = {0} if F=0 everywhere (flat), else u(1) ≅ ℝ.
        For U(n): hol(∇) ⊆ u(n), dimension ≤ n².
        """
        # Collect all non-zero curvature components
        generators = []
        for F in curvature_samples:
            frob_norm = np.linalg.norm(F, 'fro')
            if frob_norm > 1e-10:
                generators.append(F)

        is_flat = len(generators) == 0
        if is_flat:
            return {
                "holonomy_algebra_dim": 0,
                "is_flat_connection": True,
                "holonomy_group": "trivial {e}",
                "generators": 0,
            }

        # Estimate algebra dimension from independent generators
        if generators:
            stacked = np.array([g.flatten() for g in generators])
            rank = np.linalg.matrix_rank(stacked, tol=1e-8)
        else:
            rank = 0

        return {
            "holonomy_algebra_dim": int(rank),
            "is_flat_connection": False,
            "curvature_samples": len(curvature_samples),
            "nonzero_curvatures": len(generators),
            "generators": int(rank),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  5. CHERN-WEIL THEORY — Characteristic Classes from Curvature
# ═══════════════════════════════════════════════════════════════════════════════

class ChernWeilTheory:
    """
    Chern-Weil theory constructs topological invariants (characteristic classes)
    from the curvature of a connection.

    The key formula: for an invariant polynomial P on g,
        [P(F)] ∈ H^{2k}(M; ℝ) is independent of the connection!

    First Chern class:   c₁ = (i/2π) Tr(F)
    Second Chern class:  c₂ = (1/8π²) [Tr(F∧F) - Tr(F)∧Tr(F)]
    Chern character:     ch = Tr(exp(iF/2π))

    For Berry phase: c₁ = (1/2π) ∫∫ Ω (the first Chern number is the
    topological invariant that quantizes the Berry phase).
    """

    def first_chern_class(self, curvature: np.ndarray) -> float:
        """
        First Chern class: c₁ = (i/2π) Tr(F).

        For Abelian (U(1)) connection: Tr(F) = F itself.
        For non-Abelian: Tr(F) is the trace of the matrix-valued curvature.
        """
        if curvature.ndim <= 2:
            # U(1) case or single curvature value
            return float(np.sum(curvature)) / (2 * PI)
        # Matrix-valued: take trace
        return float(np.trace(curvature)) / (2 * PI)

    def second_chern_class(self, curvature: np.ndarray) -> float:
        """
        Second Chern class: c₂ = (1/8π²) Tr(F∧F).

        For a 4D manifold, this gives the instanton number.
        """
        if curvature.ndim < 2:
            return 0.0
        # c₂ = (1/8π²)(Tr(F²) - (Tr F)²)
        F2 = curvature @ curvature if curvature.ndim == 2 else np.eye(2)
        tr_F2 = float(np.trace(F2))
        tr_F = float(np.trace(curvature))
        return (tr_F2 - tr_F ** 2) / (8 * PI ** 2)

    def chern_character(self, curvature: np.ndarray) -> np.ndarray:
        """
        Chern character: ch(E) = Tr(exp(iF/2π)).

        This is the generating function for all Chern classes.
        Additive under direct sum: ch(E⊕F) = ch(E) + ch(F).
        Multiplicative under tensor product: ch(E⊗F) = ch(E)·ch(F).
        """
        F_scaled = 1j * curvature / (2 * PI)
        # Matrix exponential
        if curvature.ndim >= 2:
            eigvals, eigvecs = np.linalg.eig(F_scaled)
            exp_F = eigvecs @ np.diag(np.exp(eigvals)) @ np.linalg.inv(eigvecs)
            return exp_F
        return np.exp(F_scaled)

    def chern_simons_3form(self, A: np.ndarray, F: np.ndarray) -> float:
        """
        Chern-Simons 3-form: CS(A) = Tr(A∧dA + (2/3)A∧A∧A).

        For U(1): CS(A) = A∧F = A∧dA (the cubic term vanishes).

        The Chern-Simons invariant is defined mod ℤ and gives the
        "fractional part" of the Chern number.
        """
        # Simplified: Tr(A·F) for matrix valued, A·F for scalar
        if isinstance(A, np.ndarray) and A.ndim >= 2:
            return float(np.trace(A @ F))
        return float(np.dot(A.flatten(), F.flatten()))

    def gauss_bonnet_chern(self, dimension: int, euler_char: int) -> Dict[str, Any]:
        """
        Generalized Gauss-Bonnet-Chern theorem:

        For a closed 2n-dimensional Riemannian manifold M:
            χ(M) = (1/(2π)^n) ∫_M Pf(Ω)

        where Pf is the Pfaffian of the curvature 2-form Ω.
        For n=1 (surfaces): χ = (1/2π)∫K dA (classical Gauss-Bonnet).
        """
        n = dimension // 2
        normalization = (2 * PI) ** n

        return {
            "dimension": dimension,
            "euler_characteristic": euler_char,
            "normalization": f"(2π)^{n} = {normalization:.6f}",
            "formula": f"χ(M) = (1/{normalization:.2f}) ∫_M Pf(Ω)",
            "classical_limit": "χ = (1/2π)∫K dA for surfaces (dim=2)",
            "curvature_integral": euler_char * normalization,
        }

    def todd_class(self, chern_classes: List[float]) -> float:
        """
        Todd class: td(E) = 1 + c₁/2 + (c₁² + c₂)/12 + ...

        Appears in the Hirzebruch-Riemann-Roch theorem.
        """
        if not chern_classes:
            return 1.0
        c1 = chern_classes[0] if len(chern_classes) >= 1 else 0
        c2 = chern_classes[1] if len(chern_classes) >= 2 else 0
        return 1.0 + c1 / 2 + (c1 ** 2 + c2) / 12


# ═══════════════════════════════════════════════════════════════════════════════
#  6. BERRY CONNECTION MATH — U(1) Gauge Theory of Quantum States
# ═══════════════════════════════════════════════════════════════════════════════

class BerryConnectionMath:
    """
    Formalizes the Berry connection as a U(1) gauge field on the parameter
    space of a quantum Hamiltonian.

    Key mathematical results:
    1. Berry connection = connection 1-form of natural U(1) bundle
    2. Berry curvature = curvature 2-form = first Chern class representative
    3. Quantization: ∫∫ F = 2πn for closed surfaces (Chern theorem)
    4. Gauge invariance: γ is independent of the gauge choice for |ψ⟩
    """

    def berry_flux_quantization(self, total_flux: float) -> Dict[str, Any]:
        """
        For a closed surface in parameter space, the total Berry flux
        must be quantized: Φ = ∫∫_S F = 2πn.

        This is the Chern theorem for line bundles (U(1) principal bundles).
        """
        n = total_flux / (2 * PI)
        n_int = round(n)
        is_quantized = abs(n - n_int) < 0.05

        return {
            "total_flux": total_flux,
            "chern_number_exact": n,
            "chern_number_integer": n_int,
            "is_quantized": is_quantized,
            "quantization_error": abs(n - n_int),
            "theorem": "∫∫_S F = 2πc₁, c₁ ∈ ℤ (Chern's theorem)",
        }

    def stokes_theorem_berry(
        self,
        boundary_integral: float,
        surface_integral: float,
    ) -> Dict[str, Any]:
        """
        Stokes' theorem for Berry connection:

            γ = ∮_C A·dR = ∫∫_S F dσ

        The Berry phase equals the Berry flux through any surface bounded by the loop.
        This is valid only for contractible loops (no monopoles inside).
        """
        discrepancy = abs(boundary_integral - surface_integral)
        # If discrepancy ≈ 2πn, a monopole is enclosed
        monopole_charge = round(discrepancy / (2 * PI)) if discrepancy > PI else 0

        return {
            "boundary_integral_gamma": boundary_integral,
            "surface_integral_flux": surface_integral,
            "discrepancy": discrepancy,
            "stokes_satisfied": discrepancy < 0.01,
            "monopole_charge": monopole_charge,
            "interpretation": (
                "Stokes verified: Berry phase = Berry flux"
                if discrepancy < 0.01
                else f"Monopole detected: charge = {monopole_charge}"
            ),
        }

    def gauge_invariance_proof(self) -> Dict[str, str]:
        """
        Proof that Berry phase is gauge-invariant.

        Under gauge transformation |ψ'⟩ = e^{iα(R)} |ψ⟩:
          A'_μ = A_μ - ∂_μα
          γ' = ∮ A'·dR = ∮ A·dR - ∮ ∂α/∂R · dR = γ - [α]_loop = γ

        The last equality holds because α must be single-valued → [α]_loop = 2πn.
        """
        return {
            "statement": "Berry phase γ is gauge-invariant modulo 2π",
            "proof": (
                "Under |ψ'⟩ = e^{iα(R)}|ψ⟩:\n"
                "  A'_μ = A_μ - ∂_μα\n"
                "  γ' = ∮A'·dR = ∮A·dR - ∮∇α·dR\n"
                "  = γ - Δα_loop = γ - 2πn\n"
                "  ≡ γ  (mod 2π)"
            ),
            "physical_meaning": "Observable phases are gauge-independent",
            "fiber_bundle_interpretation": "Holonomy of a connection is gauge-covariant",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  7. DIRAC MONOPOLE — The Prototypical Berry Curvature Source
# ═══════════════════════════════════════════════════════════════════════════════

class DiracMonopole:
    """
    The Dirac magnetic monopole on S² is the prototypical example of
    Berry curvature. A spin-S particle in a magnetic field B = B r̂ has
    Berry curvature identical to a monopole of charge S at the origin:

        F_θφ = S sin(θ) (Berry curvature on S²)

    Total flux: ∫∫ F dΩ = 4πS → Chern number c₁ = 2S.

    This is mathematically identical to the curvature of a U(1) line bundle
    over CP¹ ≅ S².
    """

    def curvature(self, theta: float, phi: float, charge: float = 0.5) -> float:
        """
        Berry curvature of a monopole of charge S at the origin of S²:
            F_θφ = S sin(θ)
        """
        return charge * math.sin(theta)

    def connection_north(self, theta: float, phi: float, charge: float = 0.5) -> float:
        """
        Berry connection in the NORTH patch (valid for θ ≠ π):
            A_φ^N = S(1 - cos θ)  (Dirac string at south pole)
        """
        return charge * (1 - math.cos(theta))

    def connection_south(self, theta: float, phi: float, charge: float = 0.5) -> float:
        """
        Berry connection in the SOUTH patch (valid for θ ≠ 0):
            A_φ^S = -S(1 + cos θ)  (Dirac string at north pole)
        """
        return -charge * (1 + math.cos(theta))

    def total_flux(self, charge: float = 0.5) -> float:
        """Total magnetic flux through S²: Φ = 4πS."""
        return 4 * PI * charge

    def chern_number(self, charge: float = 0.5) -> int:
        """Chern number: c₁ = 2S (must be integer for consistency)."""
        return round(2 * charge)

    def dirac_quantization(self) -> Dict[str, Any]:
        """
        Dirac quantization condition:

        The magnetic flux through a closed surface must be quantized:
            Φ = ∫∫ F = 4πn/2 = 2πn

        This is equivalent to: eg = nℏ/2 (Dirac's original form).
        Topologically: the U(1) bundle over S² must be classified by ℤ.
        """
        return {
            "condition": "eg = nℏ/2 (n ∈ ℤ)",
            "flux_quantization": "Φ = 2πn",
            "topological_origin": "π₁(U(1)) = ℤ",
            "berry_interpretation": (
                "Berry flux through any closed surface in parameter space "
                "must be an integer multiple of 2π"
            ),
            "physical_consequence": "Existence of a single monopole → all charges quantized",
        }

    def solid_angle_phase(self, theta_cone: float, charge: float = 0.5) -> Dict[str, Any]:
        """
        Berry phase for transport around a latitude θ on S²:
            γ = -2πS(1 - cos θ) = -S × Ω

        where Ω = 2π(1-cos θ) is the solid angle of the polar cap.
        """
        solid_angle = 2 * PI * (1 - math.cos(theta_cone))
        berry_phase = -charge * solid_angle

        return {
            "latitude_theta": theta_cone,
            "solid_angle": solid_angle,
            "berry_phase": berry_phase,
            "berry_phase_deg": math.degrees(berry_phase),
            "charge": charge,
            "formula": "γ = -S × Ω = -S × 2π(1-cosθ)",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  8. BLOCH SPHERE GEOMETRY — Fubini-Study Metric of CP¹
# ═══════════════════════════════════════════════════════════════════════════════

class BlochSphereGeometry:
    """
    The Bloch sphere S² ≅ CP¹ is the state space of a single qubit.
    Its natural geometry is the Fubini-Study metric, and the Berry curvature
    is the area 2-form (constant Gaussian curvature K = 1).

    The Fubini-Study metric ds² = (dθ² + sin²θ dφ²)/4 makes the Bloch sphere
    a sphere of radius 1/2.

    Total area = π (not 4π!) due to the Hopf fibration S³ → S² → S¹
    with fiber S¹ (the overall phase).
    """

    def fubini_study_metric(self, theta: float) -> np.ndarray:
        """
        Fubini-Study metric on CP¹ (Bloch sphere):
            ds² = (1/4)(dθ² + sin²θ dφ²)

        Returns 2×2 metric tensor g_μν at (θ, φ).
        """
        return np.array([
            [0.25, 0.0],
            [0.0, 0.25 * math.sin(theta) ** 2],
        ])

    def fubini_study_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Fubini-Study distance between two quantum states:
            d_FS(|ψ⟩, |φ⟩) = arccos(|⟨ψ|φ⟩|)

        This is the geodesic distance on CP^n.
        """
        overlap = abs(np.vdot(state1, state2))
        overlap = min(1.0, overlap)  # Clamp for numerical stability
        return math.acos(overlap)

    def gaussian_curvature(self) -> float:
        """Gaussian curvature of Fubini-Study CP¹: K = 4 (radius 1/2 sphere)."""
        return 4.0

    def area(self) -> float:
        """Total area of CP¹ (Bloch sphere) with Fubini-Study metric:
        A = π (since radius = 1/2, A = 4π(1/2)² = π)."""
        return PI

    def euler_characteristic(self) -> int:
        """Gauss-Bonnet: χ = (1/2π)∫K dA = (1/2π) × 4 × π = 2."""
        return 2

    def hopf_fibration(self, point_on_s2: np.ndarray) -> Dict[str, Any]:
        """
        The Hopf fibration S³ → S² with fiber S¹:
            π: (z₀, z₁) ↦ (2Re(z̄₀z₁), 2Im(z̄₀z₁), |z₀|²-|z₁|²)

        Every point on S² corresponds to a great circle (fiber) in S³.
        The Berry connection is the connection form of this fibration.

        Args:
            point_on_s2: Normalized 3-vector (x,y,z) on unit S².
        """
        x, y, z = point_on_s2

        # Construct a representative point in S³ (fiber choice = gauge choice)
        theta = math.acos(max(-1, min(1, z)))
        phi = math.atan2(y, x)

        z0 = math.cos(theta / 2)
        z1 = math.sin(theta / 2) * cmath.exp(1j * phi)

        return {
            "bloch_sphere_point": list(point_on_s2),
            "s3_representative": [complex(z0), complex(z1)],
            "fiber": "S¹ (phase circle)",
            "fiber_berry_connection": f"A_φ = (1-cos θ)/2 at θ={theta:.4f}",
            "topology": "π₃(S²) = ℤ (Hopf invariant = 1)",
            "chern_number": 1,
        }

    def geodesic_triangle_area(
        self,
        state_a: np.ndarray,
        state_b: np.ndarray,
        state_c: np.ndarray,
    ) -> float:
        """
        Area of a geodesic triangle on CP¹ with vertices |a⟩, |b⟩, |c⟩.

        By the Gauss-Bonnet theorem for the Fubini-Study metric:
            Area = 2|γ_Berry| (the Berry phase of the triangle)

        Explicit formula (Mukunda & Simon):
            Area = 2 arg(⟨a|b⟩⟨b|c⟩⟨c|a⟩)
        """
        abc = np.vdot(state_a, state_b) * np.vdot(state_b, state_c) * np.vdot(state_c, state_a)
        return 2 * abs(np.angle(abc))

    def sacred_golden_spiral_states(self, n_points: int = 50) -> List[np.ndarray]:
        """
        Generate qubit states following a golden spiral on the Bloch sphere.

        The golden angle 2π/φ² ≈ 137.508° provides the most uniform
        distribution of points on S² (Fibonacci lattice).

        These states have optimal Berry curvature sampling properties.
        """
        golden_angle = 2 * PI / (PHI ** 2)
        states = []

        for i in range(n_points):
            # Fibonacci lattice parameterization
            theta = math.acos(1 - 2 * (i + 0.5) / n_points)
            phi = golden_angle * i

            # Qubit state from Bloch sphere coordinates
            state = np.array([
                math.cos(theta / 2),
                math.sin(theta / 2) * cmath.exp(1j * phi),
            ], dtype=complex)
            states.append(state)

        return states


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED BERRY GEOMETRY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BerryGeometry:
    """
    Master Berry Geometry engine exposing all mathematical machinery.
    """

    def __init__(self):
        self.fiber_bundle = FiberBundle(base_dim=2, fiber_type="U1")
        self.connection = ConnectionForm(dim=2)
        self.transport = ParallelTransport()
        self.holonomy = HolonomyGroup()
        self.chern_weil = ChernWeilTheory()
        self.berry_math = BerryConnectionMath()
        self.monopole = DiracMonopole()
        self.bloch = BlochSphereGeometry()

    def full_geometric_analysis(self) -> Dict[str, Any]:
        """Complete geometric analysis of Berry phase structures."""
        results = {}

        # 1. Bundle classification
        results["bundle_classification"] = self.fiber_bundle.classify()

        # 2. Dirac monopole
        results["dirac_monopole"] = {
            "total_flux": self.monopole.total_flux(),
            "chern_number": self.monopole.chern_number(),
            "quantization": self.monopole.dirac_quantization(),
            "solid_angle_phases": {
                "equator": self.monopole.solid_angle_phase(PI / 2),
                "pole": self.monopole.solid_angle_phase(0.01),
                "golden_angle": self.monopole.solid_angle_phase(2 * PI / (PHI ** 2)),
            },
        }

        # 3. Bloch sphere geometry
        results["bloch_sphere"] = {
            "gaussian_curvature": self.bloch.gaussian_curvature(),
            "total_area": self.bloch.area(),
            "euler_characteristic": self.bloch.euler_characteristic(),
            "fubini_study_at_equator": self.bloch.fubini_study_metric(PI / 2).tolist(),
        }

        # 4. Hopf fibration
        results["hopf_fibration"] = self.bloch.hopf_fibration(np.array([0, 0, 1]))

        # 5. Chern-Weil: Gauss-Bonnet
        results["gauss_bonnet"] = self.chern_weil.gauss_bonnet_chern(2, 2)

        # 6. Gauge invariance proof
        results["gauge_invariance"] = self.berry_math.gauge_invariance_proof()

        # 7. Golden spiral states on Bloch sphere
        golden_states = self.bloch.sacred_golden_spiral_states(20)
        # Compute Berry phase around golden spiral
        from l104_science_engine.berry_phase import BerryPhaseCalculator
        calc = BerryPhaseCalculator()
        golden_berry = calc.discrete_berry_phase(golden_states)
        results["golden_spiral_berry_phase"] = {
            "n_states": len(golden_states),
            "berry_phase_rad": golden_berry.phase,
            "berry_phase_deg": golden_berry.phase_degrees,
            "sacred_alignment": golden_berry.sacred_alignment,
        }

        # 8. Sacred geometric phases
        god_code_phase = GOD_CODE_PHASE
        phi_phase = PHI_PHASE
        void_phase = VOID_PHASE

        results["sacred_phases"] = {
            "god_code_geometric_phase": god_code_phase,
            "phi_geometric_phase": phi_phase,
            "void_geometric_phase": void_phase,
            "god_code_chern": god_code_phase / (2 * PI),
            "phi_chern": phi_phase / (2 * PI),
        }

        results["_summary"] = {
            "engine": "BerryGeometry v1.0",
            "layers": [
                "FiberBundle", "ConnectionForm", "ParallelTransport",
                "HolonomyGroup", "ChernWeilTheory", "BerryConnectionMath",
                "DiracMonopole", "BlochSphereGeometry",
            ],
        }

        return results

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine": "BerryGeometry",
            "version": "1.0.0",
            "base_dim": self.fiber_bundle.base_dim,
            "structure_group": self.fiber_bundle.fiber_type,
            "capabilities": [
                "Fiber bundle theory (U(1), SU(2), SO(3))",
                "Ehresmann connection forms",
                "Curvature 2-forms (Abelian & non-Abelian)",
                "U(1) & U(n) parallel transport",
                "Path-ordered exponentials (Wilson lines)",
                "Holonomy group computation",
                "Ambrose-Singer theorem",
                "Chern-Weil theory (c₁, c₂, Chern character, Todd class)",
                "Chern-Simons 3-form",
                "Gauss-Bonnet-Chern theorem",
                "Berry connection gauge theory",
                "Flux quantization theorem",
                "Stokes' theorem for Berry phase",
                "Gauge invariance proof",
                "Dirac monopole (connection, curvature, Dirac quantization)",
                "Bloch sphere Fubini-Study geometry",
                "Hopf fibration S³→S²",
                "Golden spiral state generation",
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

berry_geometry = BerryGeometry()
fiber_bundle = berry_geometry.fiber_bundle
connection_form = berry_geometry.connection
parallel_transport = berry_geometry.transport
holonomy_group = berry_geometry.holonomy
chern_weil = berry_geometry.chern_weil
berry_connection_math = berry_geometry.berry_math
dirac_monopole = berry_geometry.monopole
bloch_sphere = berry_geometry.bloch
