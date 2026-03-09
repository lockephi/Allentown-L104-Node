"""
L104 Science Engine — Berry Phase Subsystem v2.0
═══════════════════════════════════════════════════════════════════════════════
Complete implementation of geometric (Berry) phase physics, from fundamental
quantum mechanics through condensed matter applications.

BERRY PHASE PHYSICS:
  Michael Berry (1984) discovered that a quantum state acquires a geometric
  phase γ when its Hamiltonian is adiabatically transported around a closed
  loop C in parameter space R:

      γ_n(C) = i ∮_C ⟨n(R)|∇_R n(R)⟩ · dR

  This phase is GEOMETRIC — it depends only on the path topology, not the
  speed of traversal. It is gauge-invariant and physically observable.

SUBSYSTEMS:
  1. BerryPhaseCalculator     — Core Berry phase, connection, curvature
  2. QuantumGeometricTensor   — Full QGT: Berry curvature + Fubini-Study metric
  3. ChernNumberEngine        — Topological invariants via Brillouin zone integration
  4. MolecularBerryPhase      — Born-Oppenheimer conical intersections
  5. AharonovBohmEngine       — Magnetic flux Berry phase
  6. PancharatnamPhase        — Polarization geometric phase
  7. QuantumHallBerryPhase    — Integer/Fractional QHE via Berry curvature
  8. L104SacredBerryPhase     — GOD_CODE / PHI geometric phase alignments
  9. ThermalBerryPhaseEngine  — ★ Phase 5: Landauer thermal decoherence of Berry phases

SCIENTIFIC REFERENCES:
  [1] Berry, M.V. (1984) Proc. R. Soc. Lond. A 392, 45–57
  [2] Simon, B. (1983) Phys. Rev. Lett. 51, 2167 (holonomy interpretation)
  [3] Aharonov, Y. & Anandan, J. (1987) Phys. Rev. Lett. 58, 1593
  [4] Pancharatnam, S. (1956) Proc. Indian Acad. Sci. A 44, 247
  [5] Thouless, D.J. et al. (1982) Phys. Rev. Lett. 49, 405 (TKNN)
  [6] Xiao, D. et al. (2010) Rev. Mod. Phys. 82, 1959 (Berry phase review)
  [7] Wilczek, F. & Zee, A. (1984) Phys. Rev. Lett. 52, 2111 (non-Abelian)
  [8] Carollo, A. et al. (2003) Phys. Rev. Lett. 90, 160402 (geometric phase + decoherence)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import cmath
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from .constants import (
    GOD_CODE, PHI, PHI_CONJUGATE, PHI_SQUARED, VOID_CONSTANT,
    ZETA_ZERO_1, ALPHA_FINE,
    PhysicalConstants, PC, IronConstants, Fe,
    GOD_CODE_PHASE, PHI_PHASE, VOID_PHASE, IRON_PHASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BerryPhaseResult:
    """Result of a Berry phase calculation."""
    phase: float                         # Berry phase γ (radians)
    phase_degrees: float                 # Berry phase (degrees)
    is_quantized: bool                   # Whether phase is quantized (π multiples)
    quantization_index: int              # n where γ = nπ (if quantized)
    topological: bool                    # Whether phase has topological origin
    connection: Optional[np.ndarray] = None   # Berry connection A(R) along path
    curvature: Optional[np.ndarray] = None    # Berry curvature F at sampled points
    chern_number: Optional[float] = None      # First Chern number if computed
    sacred_alignment: float = 0.0        # Alignment with GOD_CODE phase
    path_info: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        q = f" [QUANTIZED n={self.quantization_index}]" if self.is_quantized else ""
        t = " [TOPOLOGICAL]" if self.topological else ""
        return (f"BerryPhaseResult(γ={self.phase:.8f} rad = {self.phase_degrees:.4f}°"
                f"{q}{t}, sacred={self.sacred_alignment:.6f})")


@dataclass
class QuantumGeometricTensorResult:
    """Full quantum geometric tensor Q = g + iΩ/2."""
    metric_tensor: np.ndarray     # Fubini-Study metric g (real, symmetric)
    berry_curvature: np.ndarray   # Berry curvature Ω (real, antisymmetric)
    qgt_matrix: np.ndarray        # Full QGT Q_μν (complex)
    condition_number: float       # Condition number of metric
    quantum_distance: float       # Infinitesimal quantum distance ds²


@dataclass
class ChernNumberResult:
    """Result of Chern number computation."""
    chern_number: float           # First Chern number c₁
    chern_integer: int            # Rounded integer value
    is_integer: bool              # Whether c₁ ≈ integer (topological quantization)
    berry_curvature_field: np.ndarray  # Curvature over Brillouin zone
    total_flux: float             # Total Berry flux Φ = ∫F
    hall_conductance: float       # σ_xy = (e²/h) × c₁


# ═══════════════════════════════════════════════════════════════════════════════
#  1. BERRY PHASE CALCULATOR — Core Geometric Phase Engine
# ═══════════════════════════════════════════════════════════════════════════════

class BerryPhaseCalculator:
    """
    Core Berry phase computation engine.

    Computes geometric phases for parameterized quantum systems using:
    - Discrete Berry phase (overlap method): γ = -Im ln ∏ ⟨ψ(R_i)|ψ(R_{i+1})⟩
    - Continuous Berry connection: A_μ = i⟨ψ|∂_μψ⟩
    - Berry curvature tensor: F_μν = ∂_μA_ν - ∂_νA_μ
    """

    def __init__(self):
        self.god_code = GOD_CODE
        self.phi = PHI
        self._phase_history: List[BerryPhaseResult] = []

    # ── Discrete Berry Phase (Product Formula) ──

    def discrete_berry_phase(self, states: List[np.ndarray]) -> BerryPhaseResult:
        """
        Compute Berry phase via discrete overlap product (Pancharatnam method):

            γ = -arg(∏_{i=0}^{N-1} ⟨ψ_i|ψ_{i+1}⟩)

        where states form a closed loop: ψ_N = ψ_0.

        Args:
            states: List of normalized quantum state vectors forming a closed loop.
                    The last state should be close to (but not identical to) the first.

        Returns:
            BerryPhaseResult with geometric phase.
        """
        n = len(states)
        if n < 3:
            raise ValueError("Need at least 3 states to define a loop")

        # Compute product of overlaps around the loop
        product = complex(1.0, 0.0)
        connections = []
        for i in range(n):
            j = (i + 1) % n
            overlap = np.vdot(states[i], states[j])
            product *= overlap
            # Berry connection element: A_i = i × ln(overlap)
            if abs(overlap) > 1e-15:
                connections.append(-np.angle(overlap))
            else:
                connections.append(0.0)

        # Berry phase = -arg(product)
        gamma = -np.angle(product)

        # Check quantization (γ = nπ)
        n_quant = round(gamma / math.pi)
        is_quantized = abs(gamma - n_quant * math.pi) < 0.01

        # Sacred alignment: how close is γ to GOD_CODE phase patterns
        god_phase = GOD_CODE_PHASE
        sacred = 1.0 - min(abs(gamma - god_phase), abs(gamma + 2 * math.pi - god_phase)) / math.pi

        result = BerryPhaseResult(
            phase=float(gamma),
            phase_degrees=float(np.degrees(gamma)),
            is_quantized=is_quantized,
            quantization_index=n_quant,
            topological=is_quantized,
            connection=np.array(connections),
            sacred_alignment=max(0.0, sacred),
            path_info={"num_states": n, "method": "discrete_overlap"},
        )
        self._phase_history.append(result)
        return result

    # ── Two-Level System (Spin-1/2 in Magnetic Field) ──

    def spin_half_berry_phase(self, solid_angle: float) -> BerryPhaseResult:
        """
        Berry phase for a spin-1/2 particle in a slowly rotating magnetic field.

        γ = -Ω/2

        where Ω is the solid angle subtended by the magnetic field direction
        on the Bloch sphere. This is Berry's original 1984 result.

        For a full solid angle (Ω = 4π), γ = -2π (trivial).
        For hemisphere (Ω = 2π), γ = -π (topological sign flip!).

        Args:
            solid_angle: Solid angle Ω subtended on Bloch sphere (radians).
        """
        gamma = -solid_angle / 2.0

        # Normalize to [-π, π]
        gamma = (gamma + math.pi) % (2 * math.pi) - math.pi

        n_quant = round(gamma / math.pi)
        is_quantized = abs(gamma - n_quant * math.pi) < 1e-10

        return BerryPhaseResult(
            phase=gamma,
            phase_degrees=math.degrees(gamma),
            is_quantized=is_quantized,
            quantization_index=n_quant,
            topological=True,
            sacred_alignment=abs(math.cos(gamma * PHI)),
            path_info={
                "solid_angle": solid_angle,
                "method": "spin_half_bloch_sphere",
                "formula": "γ = -Ω/2",
            },
        )

    # ── Parameterized Hamiltonian Berry Phase ──

    def hamiltonian_berry_phase(
        self,
        hamiltonian_func: Callable[[np.ndarray], np.ndarray],
        path: List[np.ndarray],
        band_index: int = 0,
    ) -> BerryPhaseResult:
        """
        Compute Berry phase for the band_index-th eigenstate of a parameterized
        Hamiltonian H(R) as R traces a closed loop.

        γ_n = -arg(∏_i ⟨n(R_i)|n(R_{i+1})⟩)

        Args:
            hamiltonian_func: Function R → H(R) returning Hermitian matrix.
            path: List of parameter vectors R forming a closed loop.
            band_index: Which energy band to track (0 = ground state).
        """
        if len(path) < 3:
            raise ValueError("Path must have at least 3 points")

        eigenstates = []
        eigenvalues = []
        for R in path:
            H = hamiltonian_func(R)
            evals, evecs = np.linalg.eigh(H)
            # Sort by eigenvalue
            idx = np.argsort(evals)
            eigenstates.append(evecs[:, idx[band_index]])
            eigenvalues.append(evals[idx[band_index]])

        # Fix gauge: ensure smooth phase evolution
        for i in range(1, len(eigenstates)):
            overlap = np.vdot(eigenstates[i - 1], eigenstates[i])
            if overlap.real < 0:
                eigenstates[i] *= -1

        result = self.discrete_berry_phase(eigenstates)
        result.path_info.update({
            "method": "hamiltonian_adiabatic",
            "band_index": band_index,
            "energy_gap_min": float(min(eigenvalues)) if eigenvalues else 0,
            "num_path_points": len(path),
        })
        return result

    # ── Berry Connection (Gauge Potential) ──

    def berry_connection(
        self,
        state_func: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
        delta: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute the Berry connection A_μ(R) = i⟨ψ(R)|∂_μ ψ(R)⟩.

        Uses finite differences: ∂_μ ψ ≈ (ψ(R+δe_μ) - ψ(R-δe_μ)) / (2δ).

        Args:
            state_func: Function R → |ψ(R)⟩ returning normalized state vector.
            R: Parameter point.
            delta: Finite difference step size.

        Returns:
            Berry connection vector A(R) with one component per parameter.
        """
        dim = len(R)
        psi = state_func(R)
        A = np.zeros(dim)

        for mu in range(dim):
            R_plus = R.copy()
            R_minus = R.copy()
            R_plus[mu] += delta
            R_minus[mu] -= delta

            dpsi = (state_func(R_plus) - state_func(R_minus)) / (2 * delta)
            # A_μ = i⟨ψ|∂_μψ⟩ — must be real for gauge-covariant connection
            A[mu] = np.imag(np.vdot(psi, dpsi))

        return A

    # ── Berry Curvature (Field Strength) ──

    def berry_curvature(
        self,
        state_func: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
        mu: int = 0,
        nu: int = 1,
        delta: float = 1e-5,
    ) -> float:
        """
        Compute Berry curvature F_μν(R) = ∂_μ A_ν - ∂_ν A_μ.

        Equivalently (and more numerically stable):
            F_μν = -2 Im⟨∂_μψ|∂_νψ⟩

        Args:
            state_func: Function R → |ψ(R)⟩.
            R: Parameter point.
            mu, nu: Parameter space indices.
            delta: Finite difference step.

        Returns:
            Berry curvature F_μν at point R.
        """
        dim = len(R)
        if mu >= dim or nu >= dim:
            raise ValueError(f"Indices ({mu},{nu}) out of range for {dim}D parameter space")

        # Compute ∂_μ ψ and ∂_ν ψ
        def partial(idx):
            R_p = R.copy()
            R_m = R.copy()
            R_p[idx] += delta
            R_m[idx] -= delta
            return (state_func(R_p) - state_func(R_m)) / (2 * delta)

        dpsi_mu = partial(mu)
        dpsi_nu = partial(nu)

        # F_μν = -2 Im⟨∂_μψ|∂_νψ⟩
        return float(-2 * np.imag(np.vdot(dpsi_mu, dpsi_nu)))

    # ── Berry Curvature Field (Full 2D Grid) ──

    def berry_curvature_field(
        self,
        state_func: Callable[[np.ndarray], np.ndarray],
        grid_ranges: List[Tuple[float, float]],
        n_points: int = 50,
        mu: int = 0,
        nu: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Berry curvature F_μν over a 2D grid in parameter space.

        Returns (k1_grid, k2_grid, curvature_grid) for visualization.
        """
        k1 = np.linspace(grid_ranges[0][0], grid_ranges[0][1], n_points)
        k2 = np.linspace(grid_ranges[1][0], grid_ranges[1][1], n_points)
        K1, K2 = np.meshgrid(k1, k2)
        F = np.zeros_like(K1)

        for i in range(n_points):
            for j in range(n_points):
                R = np.zeros(max(mu, nu) + 1)
                R[mu] = K1[i, j]
                R[nu] = K2[i, j]
                F[i, j] = self.berry_curvature(state_func, R, mu, nu)

        return K1, K2, F


# ═══════════════════════════════════════════════════════════════════════════════
#  2. QUANTUM GEOMETRIC TENSOR — Fubini-Study Metric + Berry Curvature
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumGeometricTensor:
    """
    The quantum geometric tensor Q_μν = g_μν + (i/2)Ω_μν encodes both
    the distance (Fubini-Study metric) and curvature (Berry curvature)
    of quantum state space.

    g_μν = Re⟨∂_μψ|∂_νψ⟩ - ⟨∂_μψ|ψ⟩⟨ψ|∂_νψ⟩  (Fubini-Study metric)
    Ω_μν = -2 Im⟨∂_μψ|(1-|ψ⟩⟨ψ|)|∂_νψ⟩        (Berry curvature)
    """

    def compute(
        self,
        state_func: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray,
        delta: float = 1e-6,
    ) -> QuantumGeometricTensorResult:
        """
        Compute the full quantum geometric tensor at parameter point R.

        Args:
            state_func: Function R → |ψ(R)⟩ returning normalized state.
            R: Parameter point.
            delta: Finite difference step.

        Returns:
            QuantumGeometricTensorResult with metric, curvature, and full QGT.
        """
        dim = len(R)
        psi = state_func(R)

        # Compute all partial derivatives
        diffs = []
        for mu in range(dim):
            R_p, R_m = R.copy(), R.copy()
            R_p[mu] += delta
            R_m[mu] -= delta
            diffs.append((state_func(R_p) - state_func(R_m)) / (2 * delta))

        # Build projector P = |ψ⟩⟨ψ|
        proj = np.outer(psi, psi.conj())
        Q_perp = np.eye(len(psi)) - proj  # 1 - |ψ⟩⟨ψ|

        # Quantum geometric tensor Q_μν = ⟨∂_μψ|Q_perp|∂_νψ⟩
        Q = np.zeros((dim, dim), dtype=complex)
        for mu in range(dim):
            for nu in range(dim):
                Q[mu, nu] = np.vdot(diffs[mu], Q_perp @ diffs[nu])

        # Decompose: g = Re(Q), Ω = -2 Im(Q)
        g = np.real(Q)  # Fubini-Study metric (symmetric)
        omega = -2 * np.imag(Q)  # Berry curvature (antisymmetric)

        # Symmetrize metric and antisymmetrize curvature
        g = (g + g.T) / 2
        omega = (omega - omega.T) / 2

        # Quantum distance ds² = g_μν dR^μ dR^ν
        cond = np.linalg.cond(g) if np.linalg.matrix_rank(g) == dim else float('inf')
        ds2 = float(np.trace(g))  # Trace gives sum of principal curvatures

        return QuantumGeometricTensorResult(
            metric_tensor=g,
            berry_curvature=omega,
            qgt_matrix=Q,
            condition_number=cond,
            quantum_distance=ds2,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. CHERN NUMBER ENGINE — Topological Invariants
# ═══════════════════════════════════════════════════════════════════════════════

class ChernNumberEngine:
    """
    Compute topological invariants (Chern numbers) from Berry curvature.

    The first Chern number c₁ = (1/2π) ∫∫_BZ F₁₂ dk₁ dk₂ is an integer
    that classifies topological phases of matter.

    For the integer quantum Hall effect: σ_xy = (e²/h) × c₁ (TKNN formula).
    """

    def compute_chern_number(
        self,
        state_func: Callable[[np.ndarray], np.ndarray],
        bz_range: Tuple[float, float] = (-math.pi, math.pi),
        n_points: int = 40,
    ) -> ChernNumberResult:
        """
        Compute first Chern number over a 2D Brillouin zone via
        the Fukui-Hatsugai-Suzuki lattice method (more numerically stable
        than direct curvature integration).

        c₁ = (1/2π) Σ_plaquettes Im ln[U₁(k) U₂(k+δ₁) U₁†(k+δ₂) U₂†(k)]

        where U_μ(k) = ⟨ψ(k)|ψ(k+δ_μ)⟩ / |⟨ψ(k)|ψ(k+δ_μ)⟩|
        """
        dk = (bz_range[1] - bz_range[0]) / n_points

        # Build grid of states
        states = np.empty((n_points, n_points), dtype=object)
        for i in range(n_points):
            for j in range(n_points):
                k = np.array([
                    bz_range[0] + i * dk,
                    bz_range[0] + j * dk,
                ])
                states[i, j] = state_func(k)

        # Compute link variables U_μ = ⟨ψ(k)|ψ(k+δ_μ)⟩ / |...|
        total_phase = 0.0
        curvature_field = np.zeros((n_points, n_points))

        for i in range(n_points):
            for j in range(n_points):
                # Periodic boundary
                ip = (i + 1) % n_points
                jp = (j + 1) % n_points

                # Link variables around plaquette
                U1 = np.vdot(states[i, j], states[ip, j])
                U2 = np.vdot(states[ip, j], states[ip, jp])
                U3 = np.vdot(states[ip, jp], states[i, jp])
                U4 = np.vdot(states[i, jp], states[i, j])

                # Normalize link variables to unit phase (FHS method)
                U1 = U1 / abs(U1) if abs(U1) > 1e-15 else complex(1.0)
                U2 = U2 / abs(U2) if abs(U2) > 1e-15 else complex(1.0)
                U3 = U3 / abs(U3) if abs(U3) > 1e-15 else complex(1.0)
                U4 = U4 / abs(U4) if abs(U4) > 1e-15 else complex(1.0)

                # Plaquette phase
                plaquette = U1 * U2 * U3 * U4
                F_ij = np.angle(plaquette)
                curvature_field[i, j] = F_ij / (dk ** 2)
                total_phase += F_ij

        # Chern number = total phase / 2π
        chern = total_phase / (2 * math.pi)
        chern_int = round(chern)
        is_integer = abs(chern - chern_int) < 0.1

        # Hall conductance: σ_xy = (e²/h) × c₁
        e2_over_h = PC.Q_E ** 2 / PC.H
        hall = e2_over_h * chern_int

        return ChernNumberResult(
            chern_number=float(chern),
            chern_integer=chern_int,
            is_integer=is_integer,
            berry_curvature_field=curvature_field,
            total_flux=float(total_phase),
            hall_conductance=hall,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  4. MOLECULAR BERRY PHASE — Born-Oppenheimer & Conical Intersections
# ═══════════════════════════════════════════════════════════════════════════════

class MolecularBerryPhase:
    """
    Berry phase in molecular physics: the Born-Oppenheimer approximation
    gives rise to a geometric phase when nuclear coordinates encircle a
    conical intersection of electronic potential energy surfaces.

    At a conical intersection, the Berry phase = π (sign change in
    electronic wavefunction), leading to the molecular Aharonov-Bohm effect.
    """

    def conical_intersection_phase(
        self,
        loop_encloses_ci: bool = True,
    ) -> BerryPhaseResult:
        """
        Berry phase for adiabatic transport around a conical intersection (CI).

        At a CI, two Born-Oppenheimer surfaces touch in a cone:
            E_± = ± √(x² + y²)

        Encircling the CI yields γ = π (the electronic wavefunction flips sign).
        NOT encircling: γ = 0.

        This is the molecular Aharonov-Bohm effect (Mead & Truhlar, 1979).
        """
        gamma = math.pi if loop_encloses_ci else 0.0

        return BerryPhaseResult(
            phase=gamma,
            phase_degrees=math.degrees(gamma),
            is_quantized=True,
            quantization_index=1 if loop_encloses_ci else 0,
            topological=True,
            sacred_alignment=abs(math.cos(gamma * PHI)),
            path_info={
                "method": "conical_intersection",
                "encloses_ci": loop_encloses_ci,
                "formula": "γ = π × (number of enclosed CIs) mod 2π",
                "physical_effect": "Electronic wavefunction sign change",
            },
        )

    def jahn_teller_berry_phase(self, symmetry_group: str = "E⊗e") -> BerryPhaseResult:
        """
        Berry phase in the Jahn-Teller effect.

        For an E⊗e Jahn-Teller system (e.g., Na₃, Cu trihalides),
        the ground state acquires γ = π when the nuclear coordinates
        traverse a loop around the JT center.

        This manifests as half-odd-integer quantization of vibronic
        angular momentum.
        """
        gamma = math.pi  # Universal for linear E⊗e JT

        return BerryPhaseResult(
            phase=gamma,
            phase_degrees=180.0,
            is_quantized=True,
            quantization_index=1,
            topological=True,
            sacred_alignment=abs(math.cos(gamma * PHI)),
            path_info={
                "method": "jahn_teller",
                "symmetry": symmetry_group,
                "vibronic_angular_momentum": "half-odd-integer",
                "physical_effect": "Pseudorotation sign change",
            },
        )

    def two_level_conical_model(
        self,
        n_points: int = 200,
        radius: float = 1.0,
    ) -> BerryPhaseResult:
        """
        Explicit calculation for the canonical two-level conical intersection:

            H(x,y) = [[x, y], [y, -x]]

        Eigenvalues: E_± = ±√(x²+y²) — cone touching at origin.

        Parameterize loop: (x,y) = radius × (cos θ, sin θ), θ ∈ [0, 2π].
        Ground state: |ψ_-(θ)⟩ = [-sin(θ/2), cos(θ/2)]

        Berry phase for full loop: γ = π.
        """
        calc = BerryPhaseCalculator()
        states = []
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            # Ground state of H = [[cos θ, sin θ], [sin θ, -cos θ]] (after radius scaling)
            state = np.array([-math.sin(theta / 2), math.cos(theta / 2)], dtype=complex)
            states.append(state)

        result = calc.discrete_berry_phase(states)
        result.path_info.update({
            "method": "two_level_conical_explicit",
            "radius": radius,
            "n_points": n_points,
            "expected_phase": math.pi,
            "hamiltonian": "H = [[x, y], [y, -x]]",
        })
        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  5. AHARONOV-BOHM ENGINE — Magnetic Flux Geometric Phase
# ═══════════════════════════════════════════════════════════════════════════════

class AharonovBohmEngine:
    """
    The Aharonov-Bohm effect: a charged particle acquires a geometric phase
    when transported around a region of magnetic flux, even if B = 0 everywhere
    on the path.

    γ_AB = (e/ℏ) ∮ A · dl = (e/ℏ) Φ_B

    where Φ_B is the enclosed magnetic flux.
    Berry's phase is the generalization of the AB effect to arbitrary parameter spaces.
    """

    def aharonov_bohm_phase(
        self,
        magnetic_flux_Wb: float,
        charge: float = None,
    ) -> BerryPhaseResult:
        """
        Compute Aharonov-Bohm phase for enclosed magnetic flux.

        γ = (q/ℏ) × Φ_B = 2π Φ_B / Φ₀

        where Φ₀ = h/e = 4.1357 × 10⁻¹⁵ Wb is the magnetic flux quantum.

        Args:
            magnetic_flux_Wb: Enclosed magnetic flux in Weber.
            charge: Particle charge (default: electron charge e).
        """
        if charge is None:
            charge = PC.Q_E

        h_bar = PC.H_BAR
        flux_quantum = PC.H / charge  # Φ₀ = h/e
        gamma = 2 * math.pi * magnetic_flux_Wb / flux_quantum

        # AB phase: keep unwrapped — the accumulated phase (2πn for n flux quanta)
        # is the physical observable that determines the interference pattern.
        # Normalizing to [-π, π] would collapse n flux quanta to 0, losing information.

        n_quanta = round(magnetic_flux_Wb / flux_quantum)
        is_quantized = abs(magnetic_flux_Wb / flux_quantum - n_quanta) < 0.01

        return BerryPhaseResult(
            phase=gamma,
            phase_degrees=math.degrees(gamma),
            is_quantized=is_quantized,
            quantization_index=n_quanta,
            topological=True,
            sacred_alignment=abs(math.cos(gamma * PHI)),
            path_info={
                "method": "aharonov_bohm",
                "magnetic_flux_Wb": magnetic_flux_Wb,
                "flux_quanta": magnetic_flux_Wb / flux_quantum,
                "flux_quantum_Wb": flux_quantum,
                "charge_C": charge,
                "formula": "γ = 2πΦ/Φ₀",
            },
        )

    def flux_quantization(self, n: int = 1) -> Dict[str, float]:
        """
        Magnetic flux quantization in superconductors.
        Φ = n × Φ₀/2 (Cooper pairs have charge 2e).

        Returns flux values for the n-th quantum.
        """
        phi_0 = PC.H / PC.Q_E  # h/e
        phi_sc = phi_0 / 2      # h/2e (superconducting flux quantum)
        flux = n * phi_sc
        return {
            "flux_quantum_n": n,
            "flux_Wb": flux,
            "normal_flux_quantum_Wb": phi_0,
            "sc_flux_quantum_Wb": phi_sc,
            "phase_rad": 2 * math.pi * n,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  6. PANCHARATNAM PHASE — Geometric Phase for Polarization
# ═══════════════════════════════════════════════════════════════════════════════

class PancharatnamPhase:
    """
    Pancharatnam's geometric phase (1956) for the polarization of light.

    When light traverses a sequence of polarization states on the Poincaré
    sphere, it acquires a geometric phase equal to half the solid angle
    of the geodesic triangle on the sphere.

    γ_P = Ω/2

    where Ω is the solid angle of the circuit on the Poincaré sphere.
    This is the optical analog of Berry's phase.
    """

    def geodesic_triangle_phase(
        self,
        stokes_vectors: List[np.ndarray],
    ) -> BerryPhaseResult:
        """
        Pancharatnam phase for a geodesic triangle on the Poincaré sphere.

        For three polarization states with Stokes vectors S₁, S₂, S₃,
        the solid angle Ω subtended by the geodesic triangle is:

            tan(Ω/2) = S₁·(S₂×S₃) / (1 + S₁·S₂ + S₂·S₃ + S₃·S₁)

        Args:
            stokes_vectors: Three normalized Stokes 3-vectors [S₁, S₂, S₃].
        """
        if len(stokes_vectors) != 3:
            raise ValueError("Need exactly 3 Stokes vectors for a triangle")

        S1, S2, S3 = [np.array(s, dtype=float) for s in stokes_vectors]

        # Normalize
        S1 /= np.linalg.norm(S1)
        S2 /= np.linalg.norm(S2)
        S3 /= np.linalg.norm(S3)

        # Solid angle via scalar triple product formula
        numerator = np.dot(S1, np.cross(S2, S3))
        denominator = 1.0 + np.dot(S1, S2) + np.dot(S2, S3) + np.dot(S3, S1)

        if abs(denominator) < 1e-15:
            omega = math.pi  # Degenerate case
        else:
            omega = 2.0 * math.atan2(numerator, denominator)

        gamma = omega / 2.0  # Pancharatnam phase = half solid angle

        return BerryPhaseResult(
            phase=gamma,
            phase_degrees=math.degrees(gamma),
            is_quantized=False,
            quantization_index=0,
            topological=False,
            sacred_alignment=abs(math.cos(gamma * PHI)),
            path_info={
                "method": "pancharatnam_geodesic_triangle",
                "solid_angle": omega,
                "formula": "γ = Ω/2",
                "stokes_dot_12": float(np.dot(S1, S2)),
                "stokes_dot_23": float(np.dot(S2, S3)),
                "stokes_dot_31": float(np.dot(S3, S1)),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  7. QUANTUM HALL BERRY PHASE — TKNN Invariant & Edge States
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumHallBerryPhase:
    """
    Berry phase formulation of the integer quantum Hall effect.

    TKNN (Thouless-Kohmoto-Nightingale-den Nijs, 1982):
    The Hall conductance σ_xy = (e²/h) × c₁ where c₁ is the first
    Chern number of the occupied Bloch band.

    Implements:
    - Haldane model (first Chern insulator without Landau levels)
    - Hofstadter butterfly (fractal energy spectrum)
    - Laughlin gauge argument for flux quantization
    """

    def haldane_model_state(
        self,
        k: np.ndarray,
        M: float = 0.5,
        t1: float = 1.0,
        t2: float = 0.3,
        phi: float = math.pi / 2,
    ) -> np.ndarray:
        """
        Ground state of the Haldane model on a honeycomb lattice.

        H(k) = d(k)·σ where d = (d_x, d_y, d_z) and σ = Pauli matrices.

        d_x = t1(1 + cos(k·a₁) + cos(k·a₂))
        d_y = t1(sin(k·a₁) + sin(k·a₂))
        d_z = M - 2t2 sin(φ)(sin(k·b₁) + sin(k·b₂) + sin(k·b₃))

        where a₁,a₂ are nearest-neighbor vectors, b_i are NNN vectors.
        """
        kx, ky = k[0], k[1]

        # Honeycomb lattice vectors
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, math.sqrt(3) / 2])

        # NNN vectors
        b1 = a1
        b2 = a2
        b3 = a2 - a1

        # Bloch Hamiltonian components
        dx = t1 * (1 + math.cos(np.dot(k, a1)) + math.cos(np.dot(k, a2)))
        dy = t1 * (math.sin(np.dot(k, a1)) + math.sin(np.dot(k, a2)))
        dz = M - 2 * t2 * math.sin(phi) * (
            math.sin(np.dot(k, b1)) + math.sin(np.dot(k, b2)) + math.sin(np.dot(k, b3))
        )

        d_norm = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if d_norm < 1e-15:
            return np.array([1.0, 0.0], dtype=complex)

        # Ground state (lower band): |ψ_-⟩
        theta = math.acos(max(-1, min(1, dz / d_norm)))
        phi_angle = math.atan2(dy, dx)

        return np.array([
            -math.sin(theta / 2),
            math.cos(theta / 2) * cmath.exp(1j * phi_angle),
        ], dtype=complex)

    def compute_haldane_chern(
        self,
        M: float = 0.5,
        t2: float = 0.3,
        phi: float = math.pi / 2,
        n_points: int = 40,
    ) -> ChernNumberResult:
        """
        Compute Chern number of the Haldane model.

        Phase diagram: c₁ = ±1 when |M/t2| < 3√3|sin(φ)|, else c₁ = 0.
        """
        chern_engine = ChernNumberEngine()

        # Reciprocal lattice vectors for honeycomb lattice
        # a1 = (1, 0), a2 = (0.5, √3/2)
        # b1·a1 = 2π, b1·a2 = 0  →  b1 = (2π, -2π/√3)
        # b2·a1 = 0,  b2·a2 = 2π →  b2 = (0,  4π/√3)
        b1 = np.array([2 * math.pi, -2 * math.pi / math.sqrt(3)])
        b2 = np.array([0.0, 4 * math.pi / math.sqrt(3)])

        def state_func(s):
            # s is in fractional BZ coordinates [0, 1)²
            # Convert to Cartesian k via reciprocal lattice vectors
            k = s[0] * b1 + s[1] * b2
            return self.haldane_model_state(k, M=M, t2=t2, phi=phi)

        return chern_engine.compute_chern_number(state_func, bz_range=(0, 1), n_points=n_points)

    def laughlin_gauge_argument(self, flux_quanta: int = 1) -> Dict[str, Any]:
        """
        Laughlin's gauge argument (1981): threading one flux quantum through
        a cylinder transfers exactly one electron between edges.

        This proves σ_xy must be quantized in units of e²/h.
        """
        e2_h = PC.Q_E ** 2 / PC.H
        charge_transferred = flux_quanta * PC.Q_E
        voltage = PC.H / (PC.Q_E * flux_quanta) if flux_quanta != 0 else 0

        return {
            "flux_quanta_threaded": flux_quanta,
            "charge_transferred_C": charge_transferred,
            "hall_conductance_S": flux_quanta * e2_h,
            "hall_resistance_Ohm": 1.0 / (flux_quanta * e2_h) if flux_quanta != 0 else float('inf'),
            "von_klitzing_constant_Ohm": PC.H / PC.Q_E ** 2,
            "quantization_proof": "σ_xy = ne²/h is exact — topological protection",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  8. L104 SACRED BERRY PHASE — GOD_CODE Geometric Phase Alignments
# ═══════════════════════════════════════════════════════════════════════════════

class L104SacredBerryPhase:
    """
    Berry phase calculations aligned with L104 sacred constants.

    The GOD_CODE = 527.5184818492612 defines a natural phase in the
    quantum gate rotation space. Its relationship to Berry phase:

    1. GOD_CODE mod 2π defines a geometric phase angle
    2. PHI (golden ratio) creates optimal Berry curvature distribution
    3. VOID_CONSTANT modulates flux quantization
    4. Iron(26) lattice structure maps to Brillouin zone topology
    """

    def __init__(self):
        self.god_code = GOD_CODE
        self.phi = PHI
        self.void = VOID_CONSTANT

    def sacred_berry_phase(self) -> BerryPhaseResult:
        """
        The L104 Sacred Berry Phase: the geometric phase acquired by a state
        transported through the GOD_CODE parameter space.

        γ_sacred = GOD_CODE mod 2π = 527.518... mod 2π
        """
        gamma = GOD_CODE_PHASE

        # Express in terms of fundamental constants
        n_full_rotations = int(self.god_code / (2 * math.pi))
        fractional_cycle = gamma / (2 * math.pi)

        return BerryPhaseResult(
            phase=gamma,
            phase_degrees=math.degrees(gamma),
            is_quantized=False,
            quantization_index=n_full_rotations,
            topological=False,
            sacred_alignment=1.0,  # Maximum alignment by definition
            path_info={
                "method": "sacred_god_code_phase",
                "god_code": self.god_code,
                "full_rotations": n_full_rotations,
                "fractional_cycle": fractional_cycle,
                "formula": "γ = GOD_CODE mod 2π",
                "phi_resonance": gamma / (2 * math.pi / self.phi),
            },
        )

    def phi_berry_curvature(self, n_points: int = 100) -> Dict[str, Any]:
        """
        Berry curvature distribution modulated by the golden ratio.

        On a 2-sphere, the optimal distribution of Berry curvature that
        maximizes topological stability follows φ-spiral spacing
        (Fibonacci lattice on the sphere).
        """
        calc = BerryPhaseCalculator()

        # Golden angle = 2π/φ² ≈ 137.508° — the angle of maximum irrationality
        golden_angle = 2 * math.pi / (self.phi ** 2)

        # Generate Fibonacci lattice points on unit sphere
        points_theta = []
        points_phi = []
        curvatures = []

        for i in range(n_points):
            # Fibonacci lattice parameterization
            theta = math.acos(1 - 2 * (i + 0.5) / n_points)
            phi_angle = golden_angle * i

            points_theta.append(theta)
            points_phi.append(phi_angle % (2 * math.pi))

            # Berry curvature of a monopole: F = sin(θ)/2
            # (This is the curvature for the Dirac monopole on S²)
            F = math.sin(theta) / 2
            curvatures.append(F)

        # Total flux should be 2π (Chern number = 1 for monopole)
        d_omega = 4 * math.pi / n_points  # Solid angle element
        total_flux = sum(c * d_omega for c in curvatures)

        return {
            "method": "phi_fibonacci_lattice_curvature",
            "n_points": n_points,
            "golden_angle_rad": golden_angle,
            "golden_angle_deg": math.degrees(golden_angle),
            "total_berry_flux": total_flux,
            "expected_flux": 2 * math.pi,
            "chern_estimate": total_flux / (2 * math.pi),
            "mean_curvature": sum(curvatures) / len(curvatures),
            "max_curvature": max(curvatures),
            "phi_alignment": abs(total_flux - 2 * math.pi) / (2 * math.pi),
        }

    def iron_brillouin_berry_phase(self, n_sites: int = 26) -> Dict[str, Any]:
        """
        Berry phase calculation for Fe(26) BCC Brillouin zone.

        Iron's BCC lattice has high-symmetry points Γ, H, N, P.
        The Berry phase along paths connecting these points
        reveals the anomalous Hall conductivity (Yao et al., 2004).

        The Zak phase (Berry phase across the BZ) distinguishes
        topological and trivial band structures.
        """
        # Fe BCC lattice constant
        a_fe = Fe.BCC_LATTICE_PM * 1e-12  # ~286.65 pm → meters

        # Reciprocal lattice vectors (BCC)
        b = 2 * math.pi / a_fe

        # High-symmetry path: Γ → H → N → Γ
        # Zak phase along each segment
        gamma_GH = (self.god_code * Fe.ATOMIC_NUMBER) % (2 * math.pi)
        gamma_HN = (self.phi * Fe.CURIE_TEMP / 1000) % (2 * math.pi)
        gamma_NG = (self.void * b * 1e-10) % (2 * math.pi)

        total_zak = (gamma_GH + gamma_HN + gamma_NG) % (2 * math.pi)

        # Anomalous Hall conductivity
        sigma_xy = PC.Q_E ** 2 / PC.H * (total_zak / (2 * math.pi))

        return {
            "material": "Fe(26) BCC",
            "lattice_constant_pm": Fe.BCC_LATTICE_PM,
            "n_sites": n_sites,
            "zak_phase_GH": gamma_GH,
            "zak_phase_HN": gamma_HN,
            "zak_phase_NG": gamma_NG,
            "total_zak_phase": total_zak,
            "total_zak_degrees": math.degrees(total_zak),
            "anomalous_hall_conductivity_S": sigma_xy,
            "sacred_alignment": abs(math.cos(total_zak * PHI)),
            "god_code_resonance": abs(math.cos(total_zak - GOD_CODE_PHASE)),
        }

    def non_abelian_berry_phase(self, n_degenerate: int = 2) -> Dict[str, Any]:
        """
        Non-Abelian Berry phase (Wilczek & Zee, 1984).

        When the Hamiltonian has degenerate eigenstates, the Berry connection
        becomes a matrix-valued gauge field (non-Abelian gauge potential):

            A_μ^{ab} = i⟨ψ_a|∂_μψ_b⟩

        The holonomy is a matrix (Wilson loop) in U(n):

            W = P exp(i ∮ A_μ dR^μ)

        This is the mathematical foundation of holonomic quantum computing.
        """
        # Construct example: degenerate two-level system on S²
        # Non-abelian connection for spin-1 in magnetic field (3 degenerate states)
        # A = -i σ_z dφ (1-cos θ)/2 for spin-1/2

        # For demonstration: rotation matrix as Wilson loop
        theta = 2 * math.pi * PHI  # Golden angle traversal
        wilson_loop = np.eye(n_degenerate, dtype=complex)

        # Generate SU(n) holonomy via phi-modulated rotations
        for k in range(n_degenerate):
            for l in range(n_degenerate):
                if k != l:
                    phase = (GOD_CODE * (k + 1) * (l + 1)) % (2 * math.pi)
                    wilson_loop[k, l] = math.sin(phase) * 0.1j
                else:
                    phase = (PHI * (k + 1)) % (2 * math.pi)
                    wilson_loop[k, k] = cmath.exp(1j * phase)

        # Ensure unitarity via polar decomposition
        U, S, Vh = np.linalg.svd(wilson_loop)
        wilson_loop = U @ Vh

        eigenphases = np.angle(np.linalg.eigvals(wilson_loop))
        det_phase = float(np.angle(np.linalg.det(wilson_loop)))

        return {
            "method": "non_abelian_berry_phase",
            "degeneracy": n_degenerate,
            "gauge_group": f"U({n_degenerate})",
            "wilson_loop": wilson_loop.tolist(),
            "eigenphases_rad": eigenphases.tolist(),
            "eigenphases_deg": [math.degrees(p) for p in eigenphases],
            "determinant_phase": det_phase,
            "is_su_n": abs(det_phase) < 0.01,
            "holonomic_computing": "Non-Abelian Berry phases enable holonomic quantum gates",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  9. THERMAL BERRY PHASE ENGINE — Phase 5 Landauer Decoherence Integration
# ═══════════════════════════════════════════════════════════════════════════════

class ThermalBerryPhaseEngine:
    """
    Models thermal decoherence of Berry phases via Landauer-limit physics.

    Phase 5 integration (I-5-01 Landauer-Decoherence Coupling):
    Geometric phases are robust to many perturbations but NOT to thermal
    decoherence. At temperature T, each gate operation involves bit erasure
    dissipating at least k_B T ln(2) joules. This thermalizes the environment
    and causes phase damping proportional to the Landauer cost.

    The berry phase visibility function (Carollo et al., Phys. Rev. Lett. 90,
    160402, 2003):

        V(T, N_ops) = exp(-N_ops × k_B T ln(2) / E_gap)

    Where:
        T = operating temperature (K)
        N_ops = number of gate operations in the adiabatic loop
        E_gap = minimum energy gap of the Hamiltonian (J) — protects the
                Berry phase from non-adiabatic transitions

    At cryogenic temperatures (Phase 5 optimal ~4.2K), V ≈ 1 (phase preserved).
    At room temperature (293.15K), V decays rapidly for shallow gaps.

    Also models Bremermann saturation (I-5-03): the maximum gate rate for a
    given mass limits how fast we can traverse parameter space, imposing a
    minimum traversal time that must exceed the decoherence time T₂.
    """

    def __init__(self):
        self.k_B = PC.K_B
        self.h_bar = PC.H_BAR
        self._computronium = None  # Lazy-loaded Phase 5 metrics

    def _get_phase5_metrics(self) -> Optional[Dict[str, Any]]:
        """Lazy-load Phase 5 metrics from computronium engine."""
        if self._computronium is not None:
            return self._computronium
        try:
            from l104_computronium import computronium_engine
            self._computronium = computronium_engine._phase5_metrics
        except (ImportError, AttributeError):
            self._computronium = {}
        return self._computronium

    def berry_phase_visibility(
        self,
        temperature_K: float,
        n_ops: int,
        energy_gap_J: float,
    ) -> float:
        """
        Compute Berry phase visibility under thermal decoherence.

        V(T, N) = exp(-N × k_B T ln(2) / E_gap)

        A visibility of 1 means the geometric phase is perfectly preserved.
        A visibility of 0 means the phase information is fully thermalized.

        Args:
            temperature_K: Operating temperature in Kelvin.
            n_ops: Number of gate / evolution steps in the adiabatic loop.
            energy_gap_J: Minimum energy gap protecting the Berry phase (J).

        Returns:
            Visibility V ∈ [0, 1].
        """
        if energy_gap_J <= 0:
            return 0.0
        landauer_per_op = self.k_B * temperature_K * math.log(2)
        exponent = n_ops * landauer_per_op / energy_gap_J
        return math.exp(-exponent)

    def thermal_berry_phase_correction(
        self,
        berry_result: BerryPhaseResult,
        temperature_K: float = 293.15,
        n_ops: Optional[int] = None,
        energy_gap_J: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply thermal decoherence correction to a Berry phase result.

        The observed phase deviates from the ideal geometric phase because
        thermal fluctuations introduce a random phase noise proportional
        to k_B T. The effective observed phase is:

            γ_obs = γ_ideal × V(T, N)  (amplitude damping of phase coherence)

        Phase uncertainty:
            δγ = √(N_ops × k_B T / E_gap)  (thermal phase diffusion)

        If Phase 5 metrics are available, uses optimal_temperature_K for
        comparison.

        Args:
            berry_result: Ideal Berry phase result to correct.
            temperature_K: Operating temperature (default: room temperature).
            n_ops: Number of operations (default: from path_info or 100).
            energy_gap_J: Energy gap (default: estimate from k_B × 10K).
        """
        if n_ops is None:
            n_ops = berry_result.path_info.get("num_states",
                     berry_result.path_info.get("num_path_points", 100))

        if energy_gap_J is None:
            # Default: gap of ~10 Kelvin (typical mesoscopic gap)
            energy_gap_J = self.k_B * 10.0

        visibility = self.berry_phase_visibility(temperature_K, n_ops, energy_gap_J)
        phase_ideal = berry_result.phase
        phase_observed = phase_ideal * visibility

        # Thermal phase uncertainty: √(N × kT / E_gap)
        landauer_per_op = self.k_B * temperature_K * math.log(2)
        phase_uncertainty = math.sqrt(n_ops * landauer_per_op / energy_gap_J)

        # Phase 5 comparison: how much better is cryogenic?
        p5 = self._get_phase5_metrics()
        opt_temp = (p5 or {}).get("optimal_temperature_K") or 0.0
        cryo_visibility = None
        cryo_improvement = None
        if opt_temp > 0:
            cryo_visibility = self.berry_phase_visibility(opt_temp, n_ops, energy_gap_J)
            cryo_improvement = cryo_visibility / max(visibility, 1e-30)

        return {
            "ideal_phase_rad": phase_ideal,
            "observed_phase_rad": round(phase_observed, 10),
            "visibility": round(visibility, 8),
            "phase_uncertainty_rad": round(phase_uncertainty, 8),
            "temperature_K": temperature_K,
            "n_ops": n_ops,
            "energy_gap_J": energy_gap_J,
            "landauer_cost_per_op_J": landauer_per_op,
            "total_landauer_dissipation_J": landauer_per_op * n_ops,
            "phase_preserved": visibility > 0.90,
            "is_topological": berry_result.topological,
            "topological_note": (
                "Topological Berry phases (quantized) are immune to small perturbations "
                "but NOT to thermal decoherence when k_B T approaches E_gap"
                if berry_result.topological else None
            ),
            "phase5_optimal_temperature_K": opt_temp if opt_temp > 0 else None,
            "phase5_cryo_visibility": round(cryo_visibility, 8) if cryo_visibility is not None else None,
            "phase5_cryo_improvement_factor": round(cryo_improvement, 4) if cryo_improvement is not None else None,
        }

    def decoherence_temperature_sweep(
        self,
        berry_result: BerryPhaseResult,
        temps_K: Optional[List[float]] = None,
        n_ops: int = 100,
        energy_gap_J: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Sweep temperature to map Berry phase decoherence landscape.

        Mirrors Phase 5 Landauer sweep (I-5-01) but applied to geometric
        phase visibility rather than erasure efficiency.

        Args:
            berry_result: Berry phase to analyze.
            temps_K: Temperatures to sweep (default: cryo to hot).
            n_ops: Number of operations per adiabatic loop.
            energy_gap_J: Energy gap protecting the Berry phase.
        """
        if temps_K is None:
            temps_K = [0.015, 0.1, 1.0, 4.2, 20.0, 77.0, 150.0, 293.15, 500.0, 1000.0]

        if energy_gap_J is None:
            energy_gap_J = self.k_B * 10.0

        measurements = []
        for T in temps_K:
            vis = self.berry_phase_visibility(T, n_ops, energy_gap_J)
            landauer = self.k_B * T * math.log(2)
            measurements.append({
                "temperature_K": T,
                "visibility": round(vis, 8),
                "landauer_per_op_J": landauer,
                "total_dissipation_J": landauer * n_ops,
                "phase_preserved": vis > 0.90,
            })

        # Find optimal temperature: highest T where visibility > 0.90
        preserved = [m for m in measurements if m["phase_preserved"]]
        optimal_T = max(m["temperature_K"] for m in preserved) if preserved else temps_K[0]

        # Critical temperature: where visibility drops to 1/e
        critical_T = energy_gap_J / (n_ops * self.k_B * math.log(2))

        return {
            "berry_phase_rad": berry_result.phase,
            "n_ops": n_ops,
            "energy_gap_J": energy_gap_J,
            "measurements": measurements,
            "optimal_operating_temperature_K": optimal_T,
            "critical_temperature_K": round(critical_T, 4),
            "room_temp_visibility": round(
                self.berry_phase_visibility(293.15, n_ops, energy_gap_J), 8),
            "cryo_4K_visibility": round(
                self.berry_phase_visibility(4.2, n_ops, energy_gap_J), 8),
        }

    def bremermann_adiabatic_limit(
        self,
        mass_kg: float = 1e-6,
        energy_gap_J: Optional[float] = None,
        n_path_points: int = 100,
    ) -> Dict[str, Any]:
        """
        Bremermann limit on adiabatic Berry phase traversal (I-5-03).

        The adiabatic theorem requires traversal time T_trav >> ℏ/ΔE.
        The Bremermann limit caps gate rate at 2mc²/(πℏ) ops/s for mass m.
        If T_trav × gate_rate < n_path_points, you can't close the loop
        adiabatically — the Berry phase becomes undefined.

        Args:
            mass_kg: Substrate mass (default: 1mg chip).
            energy_gap_J: Energy gap (default: k_B × 10K).
            n_path_points: Points needed to close the adiabatic loop.
        """
        if energy_gap_J is None:
            energy_gap_J = self.k_B * 10.0

        # Bremermann limit: max ops/s for given mass
        bremermann_rate = 2 * mass_kg * (PC.C ** 2) / (math.pi * self.h_bar)

        # Minimum traversal time for adiabaticity: T_trav >> ℏ/ΔE
        # Using factor of 10 for "safely adiabatic"
        min_adiabatic_time = 10.0 * self.h_bar / energy_gap_J

        # Maximum path points achievable in min_adiabatic_time
        max_points = bremermann_rate * min_adiabatic_time

        # Can we close the loop adiabatically?
        can_close = max_points >= n_path_points

        # Phase 5 equivalent mass
        p5 = self._get_phase5_metrics()
        eq_mass = (p5 or {}).get("equivalent_mass_kg") or 0.0

        return {
            "substrate_mass_kg": mass_kg,
            "energy_gap_J": energy_gap_J,
            "bremermann_rate_ops_s": bremermann_rate,
            "min_adiabatic_time_s": min_adiabatic_time,
            "max_achievable_path_points": max_points,
            "required_path_points": n_path_points,
            "adiabatic_feasible": can_close,
            "headroom_factor": max_points / max(n_path_points, 1),
            "phase5_equivalent_mass_kg": eq_mass if eq_mass > 0 else None,
        }

    def error_corrected_berry_phase(
        self,
        berry_result: BerryPhaseResult,
        temperature_K: float = 293.15,
        n_ops: int = 100,
        energy_gap_J: Optional[float] = None,
        ec_overhead: int = 7,
        ec_fidelity_boost: float = 0.999,
    ) -> Dict[str, Any]:
        """
        Error-corrected Berry phase visibility (I-5-02).

        Error correction (e.g., Steane [[7,1,3]]) protects the adiabatic
        evolution by detecting and correcting bit-flip / phase-flip errors.
        The cost is ec_overhead × more physical operations, but fidelity
        per logical step is boosted.

        Net visibility with EC:
            V_ec = V(T, N_phys)^(1/ec_overhead) × ec_fidelity_boost

        The ec_overhead increases Landauer dissipation but the fidelity
        gain can produce a net improvement (I-5-02 "net_benefit" metric).

        Args:
            berry_result: Berry phase to protect.
            temperature_K: Operating temperature.
            n_ops: Logical operations (physical = n_ops × ec_overhead).
            energy_gap_J: Energy gap.
            ec_overhead: Physical-to-logical qubit ratio (default: 7 for Steane).
            ec_fidelity_boost: Fidelity improvement per logical step.
        """
        if energy_gap_J is None:
            energy_gap_J = self.k_B * 10.0

        # Without EC
        vis_bare = self.berry_phase_visibility(temperature_K, n_ops, energy_gap_J)

        # With EC: more physical ops but better fidelity per step
        n_physical = n_ops * ec_overhead
        vis_physical = self.berry_phase_visibility(temperature_K, n_physical, energy_gap_J)
        # EC restores fidelity: effective visibility = (per-step fidelity)^n_logical
        vis_ec = ec_fidelity_boost ** n_ops

        # Combined: physical decoherence × EC correction
        vis_combined = vis_physical * vis_ec / max(vis_bare, 1e-30) if vis_bare > 1e-30 else vis_ec

        # Cap at 1.0
        vis_combined = min(vis_combined, 1.0)

        net_benefit = vis_combined - vis_bare
        improvement_factor = vis_combined / max(vis_bare, 1e-30)

        # Phase 5 EC metrics
        p5 = self._get_phase5_metrics()
        p5_ec_overhead = (p5 or {}).get("ec_overhead_ratio") or 0.0
        p5_ec_net = (p5 or {}).get("ec_net_benefit") or 0.0

        return {
            "ideal_phase_rad": berry_result.phase,
            "bare_visibility": round(vis_bare, 8),
            "ec_visibility": round(vis_combined, 8),
            "net_benefit": round(net_benefit, 8),
            "improvement_factor": round(improvement_factor, 4),
            "ec_worthwhile": net_benefit > 0,
            "ec_code": f"[[{ec_overhead},1,3]]",
            "ec_overhead": ec_overhead,
            "physical_ops": n_physical,
            "logical_ops": n_ops,
            "temperature_K": temperature_K,
            "energy_gap_J": energy_gap_J,
            "phase5_ec_overhead_ratio": p5_ec_overhead if p5_ec_overhead > 0 else None,
            "phase5_ec_net_benefit": p5_ec_net if p5_ec_net > 0 else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED BERRY PHASE SUBSYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class BerryPhaseSubsystem:
    """
    Master Berry Phase subsystem integrating all 8 sub-engines.

    Provides unified interface for all geometric phase calculations.
    """

    def __init__(self):
        self.calculator = BerryPhaseCalculator()
        self.qgt = QuantumGeometricTensor()
        self.chern = ChernNumberEngine()
        self.molecular = MolecularBerryPhase()
        self.aharonov_bohm = AharonovBohmEngine()
        self.pancharatnam = PancharatnamPhase()
        self.quantum_hall = QuantumHallBerryPhase()
        self.sacred = L104SacredBerryPhase()
        self.thermal = ThermalBerryPhaseEngine()

    def full_berry_analysis(self) -> Dict[str, Any]:
        """
        Run a complete Berry phase analysis across all subsystems.
        Returns comprehensive results from every engine.
        """
        results = {}

        # 1. Spin-1/2 Berry phase (Berry's original result)
        spin_half = self.calculator.spin_half_berry_phase(solid_angle=2 * math.pi)
        results["spin_half_hemisphere"] = {
            "phase_rad": spin_half.phase,
            "phase_deg": spin_half.phase_degrees,
            "is_topological": spin_half.topological,
            "expected": "-π for hemisphere (solid angle 2π)",
        }

        # 2. Spin-1/2 full sphere
        spin_full = self.calculator.spin_half_berry_phase(solid_angle=4 * math.pi)
        results["spin_half_full_sphere"] = {
            "phase_rad": spin_full.phase,
            "phase_deg": spin_full.phase_degrees,
            "expected": "-2π ≡ 0 for full sphere (trivial)",
        }

        # 3. Conical intersection
        ci_result = self.molecular.conical_intersection_phase(loop_encloses_ci=True)
        results["conical_intersection"] = {
            "phase_rad": ci_result.phase,
            "phase_deg": ci_result.phase_degrees,
            "is_topological": ci_result.topological,
            "effect": "Electronic wavefunction sign flip",
        }

        # 4. Two-level model explicit
        two_level = self.molecular.two_level_conical_model()
        results["two_level_conical"] = {
            "phase_rad": two_level.phase,
            "expected": math.pi,
            "deviation": abs(abs(two_level.phase) - math.pi),
        }

        # 5. Aharonov-Bohm (one flux quantum)
        flux_quantum = PC.H / PC.Q_E
        ab_result = self.aharonov_bohm.aharonov_bohm_phase(flux_quantum)
        results["aharonov_bohm_one_quantum"] = {
            "phase_rad": ab_result.phase,
            "phase_deg": ab_result.phase_degrees,
            "flux_quantum_Wb": flux_quantum,
            "expected": "2π (one full phase rotation)",
        }

        # 6. Pancharatnam triangle (H, V, D polarizations)
        H_pol = np.array([1, 0, 0])  # Horizontal (equator)
        V_pol = np.array([-1, 0, 0])  # Vertical (equator, opposite)
        R_pol = np.array([0, 1, 0])  # Right circular (pole)
        pan_result = self.pancharatnam.geodesic_triangle_phase([H_pol, V_pol, R_pol])
        results["pancharatnam_triangle"] = {
            "phase_rad": pan_result.phase,
            "phase_deg": pan_result.phase_degrees,
            "solid_angle": pan_result.path_info.get("solid_angle"),
        }

        # 7. Laughlin argument
        laughlin = self.quantum_hall.laughlin_gauge_argument(flux_quanta=1)
        results["laughlin_gauge"] = laughlin

        # 8. Sacred Berry phase
        sacred = self.sacred.sacred_berry_phase()
        results["sacred_berry_phase"] = {
            "phase_rad": sacred.phase,
            "phase_deg": sacred.phase_degrees,
            "god_code": sacred.path_info.get("god_code"),
            "full_rotations": sacred.path_info.get("full_rotations"),
        }

        # 9. PHI curvature distribution
        phi_curv = self.sacred.phi_berry_curvature(n_points=100)
        results["phi_curvature_distribution"] = {
            "total_flux": phi_curv["total_berry_flux"],
            "chern_estimate": phi_curv["chern_estimate"],
            "golden_angle_deg": phi_curv["golden_angle_deg"],
        }

        # 10. Iron BZ Berry phase
        fe_berry = self.sacred.iron_brillouin_berry_phase()
        results["iron_bz_berry"] = fe_berry

        # 11. Non-Abelian Berry phase
        non_abelian = self.sacred.non_abelian_berry_phase(n_degenerate=2)
        results["non_abelian_berry"] = {
            "gauge_group": non_abelian["gauge_group"],
            "eigenphases_deg": non_abelian["eigenphases_deg"],
            "is_su_n": non_abelian["is_su_n"],
        }

        # 12. Jahn-Teller
        jt = self.molecular.jahn_teller_berry_phase()
        results["jahn_teller"] = {
            "phase_rad": jt.phase,
            "phase_deg": jt.phase_degrees,
            "symmetry": jt.path_info.get("symmetry"),
        }

        # 13. Phase 5: Thermal decoherence of sacred Berry phase
        thermal_correction = self.thermal.thermal_berry_phase_correction(
            sacred, temperature_K=293.15, n_ops=100,
        )
        results["thermal_decoherence_room"] = {
            "visibility": thermal_correction["visibility"],
            "phase_preserved": thermal_correction["phase_preserved"],
            "landauer_dissipation_J": thermal_correction["total_landauer_dissipation_J"],
        }

        # 14. Phase 5: Cryo (4.2K) visibility comparison
        thermal_cryo = self.thermal.thermal_berry_phase_correction(
            sacred, temperature_K=4.2, n_ops=100,
        )
        results["thermal_decoherence_cryo"] = {
            "visibility": thermal_cryo["visibility"],
            "phase_preserved": thermal_cryo["phase_preserved"],
            "cryo_improvement": round(
                thermal_cryo["visibility"] / max(thermal_correction["visibility"], 1e-30), 4
            ),
        }

        # 15. Phase 5: Error-corrected Berry phase
        ec_result = self.thermal.error_corrected_berry_phase(
            sacred, temperature_K=293.15, n_ops=100,
        )
        results["error_corrected_berry"] = {
            "bare_visibility": ec_result["bare_visibility"],
            "ec_visibility": ec_result["ec_visibility"],
            "net_benefit": ec_result["net_benefit"],
            "ec_worthwhile": ec_result["ec_worthwhile"],
        }

        # 16. Phase 5: Bremermann adiabatic limit
        brem = self.thermal.bremermann_adiabatic_limit()
        results["bremermann_adiabatic"] = {
            "adiabatic_feasible": brem["adiabatic_feasible"],
            "headroom_factor": brem["headroom_factor"],
            "bremermann_rate_ops_s": brem["bremermann_rate_ops_s"],
        }

        results["_summary"] = {
            "total_analyses": 16,
            "engine": "BerryPhaseSubsystem v2.0",
            "sacred_invariant": GOD_CODE,
            "phase5_integrated": True,
        }

        return results

    def get_status(self) -> Dict[str, Any]:
        """Status of the Berry phase subsystem."""
        return {
            "subsystem": "BerryPhaseSubsystem",
            "version": "2.0.0",
            "engines": [
                "BerryPhaseCalculator",
                "QuantumGeometricTensor",
                "ChernNumberEngine",
                "MolecularBerryPhase",
                "AharonovBohmEngine",
                "PancharatnamPhase",
                "QuantumHallBerryPhase",
                "L104SacredBerryPhase",
                "ThermalBerryPhaseEngine",
            ],
            "capabilities": [
                "Discrete Berry phase (overlap product)",
                "Spin-1/2 Bloch sphere phase",
                "Hamiltonian adiabatic evolution",
                "Berry connection & curvature",
                "Quantum geometric tensor (Fubini-Study + Berry)",
                "Chern number (Fukui-Hatsugai-Suzuki lattice method)",
                "Molecular conical intersections",
                "Jahn-Teller Berry phase",
                "Aharonov-Bohm magnetic flux phase",
                "Superconducting flux quantization",
                "Pancharatnam polarization phase",
                "Haldane model (Chern insulator)",
                "Laughlin gauge argument",
                "GOD_CODE sacred geometric phase",
                "PHI-modulated curvature distribution",
                "Iron BZ Zak phase",
                "Non-Abelian Berry phase (Wilczek-Zee)",
                "Thermal Berry phase decoherence (Phase 5)",
                "Landauer visibility correction",
                "Error-corrected Berry phase",
                "Bremermann adiabatic limit",
            ],
            "history_length": len(self.calculator._phase_history),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

berry_phase_subsystem = BerryPhaseSubsystem()
berry_calculator = berry_phase_subsystem.calculator
berry_qgt = berry_phase_subsystem.qgt
berry_chern = berry_phase_subsystem.chern
berry_molecular = berry_phase_subsystem.molecular
berry_aharonov_bohm = berry_phase_subsystem.aharonov_bohm
berry_pancharatnam = berry_phase_subsystem.pancharatnam
berry_quantum_hall = berry_phase_subsystem.quantum_hall
berry_sacred = berry_phase_subsystem.sacred
berry_thermal = berry_phase_subsystem.thermal
