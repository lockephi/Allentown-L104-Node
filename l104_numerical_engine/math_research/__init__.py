"""Math Research sub-package — 11 self-contained research engines.

Each engine depends only on ``l104_numerical_engine.precision`` and
``l104_numerical_engine.constants``; there is no lattice coupling.
"""

from .riemann_zeta import RiemannZetaEngine
from .prime_theory import PrimeNumberTheoryEngine
from .infinite_series import InfiniteSeriesLab
from .number_theory import NumberTheoryForge
from .fractal_dynamics import FractalDynamicsLab
from .god_code_calculus import GodCodeCalculusEngine
from .transcendental import TranscendentalProver
from .stat_mechanics import StatisticalMechanicsEngine
from .harmonic_numbers import HarmonicNumberEngine
from .elliptic_curves import EllipticCurveEngine
from .collatz import CollatzConjectureAnalyzer

__all__ = [
    "RiemannZetaEngine",
    "PrimeNumberTheoryEngine",
    "InfiniteSeriesLab",
    "NumberTheoryForge",
    "FractalDynamicsLab",
    "GodCodeCalculusEngine",
    "TranscendentalProver",
    "StatisticalMechanicsEngine",
    "HarmonicNumberEngine",
    "EllipticCurveEngine",
    "CollatzConjectureAnalyzer",
]
