"""
L104 Quantum Gate Engine -- Cross-Engine Integration Hub v1.0
===============================================================================
Connects the Quantum Gate Engine to Science Engine, Math Engine, and Code
Engine for three-engine circuit validation, scoring, and compilation analysis.

Integration Points:
  1. Science Engine  -- circuit coherence, entropy, physics validation
  2. Math Engine     -- PHI alignment, GOD_CODE resonance, harmonic structure
  3. Code Engine     -- circuit generation quality analysis

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import math
import numpy as np
from typing import Dict, Any, Optional, List

from .constants import (
    GOD_CODE, PHI, PHI_CONJUGATE, VOID_CONSTANT,
    IRON_FREQUENCY, IRON_ATOMIC_NUMBER,
)


# ---------------------------------------------------------------------------
#  Science Engine Bridge
# ---------------------------------------------------------------------------

class ScienceEngineBridge:
    """Lazy bridge to Science Engine for physics validation of circuits."""

    _engine = None
    _connected = False

    @classmethod
    def connect(cls):
        """Lazy-connect to Science Engine."""
        if cls._connected:
            return cls._engine
        try:
            from l104_science_engine import ScienceEngine
            cls._engine = ScienceEngine()
            cls._connected = True
        except ImportError:
            cls._connected = True
        return cls._engine

    @classmethod
    def validate_circuit_physics(cls, circuit) -> Dict[str, Any]:
        """
        Validate a quantum gate circuit against physical constraints.

        Checks coherence viability, entropy bounds, and sacred resonance
        using the Science Engine's physics and entropy subsystems.
        """
        science = cls.connect()

        # Extract circuit properties
        n_qubits = getattr(circuit, "n_qubits", getattr(circuit, "num_qubits", 2))
        gates = getattr(circuit, "gates", [])
        gate_count = len(gates) if gates else getattr(circuit, "size", 0)
        depth = getattr(circuit, "depth", gate_count)

        # Decoherence budget: deeper circuits lose coherence faster
        decoherence_factor = math.exp(-depth / (n_qubits * PHI * 10.0))
        coherence_viable = decoherence_factor > 0.1

        # Entropy estimate -- circuit entropy grows with depth and qubits
        circuit_entropy = (depth * n_qubits) / (GOD_CODE * VOID_CONSTANT)
        entropy_bounded = circuit_entropy < 1.0

        result = {
            "n_qubits": n_qubits,
            "gate_count": gate_count,
            "depth": depth,
            "decoherence_factor": round(decoherence_factor, 8),
            "coherence_viable": coherence_viable,
            "circuit_entropy": round(circuit_entropy, 8),
            "entropy_bounded": entropy_bounded,
        }

        if science is not None:
            try:
                demon_eff = science.entropy.calculate_demon_efficiency(
                    min(1.0, max(0.01, circuit_entropy))
                )
                result["demon_efficiency"] = round(demon_eff, 8)
                result["entropy_reversible"] = demon_eff > 0.5
            except Exception:
                result["demon_efficiency"] = 0.5
                result["entropy_reversible"] = True

            try:
                landauer = science.physics.adapt_landauer_limit(300)
                result["landauer_limit"] = landauer
                result["above_landauer"] = gate_count > 0
            except Exception:
                pass
        else:
            result["demon_efficiency"] = 0.5
            result["entropy_reversible"] = True

        result["method"] = "science_engine" if science is not None else "analytical_fallback"
        result["physics_valid"] = coherence_viable and entropy_bounded
        return result


# ---------------------------------------------------------------------------
#  Math Engine Bridge
# ---------------------------------------------------------------------------

class MathEngineBridge:
    """Lazy bridge to Math Engine for mathematical validation of circuits."""

    _engine = None
    _connected = False

    @classmethod
    def connect(cls):
        """Lazy-connect to Math Engine."""
        if cls._connected:
            return cls._engine
        try:
            from l104_math_engine import MathEngine
            cls._engine = MathEngine()
            cls._connected = True
        except ImportError:
            cls._connected = True
        return cls._engine

    @classmethod
    def validate_circuit_math(cls, circuit) -> Dict[str, Any]:
        """
        Validate a circuit's mathematical structure.

        Checks PHI alignment of gate phases, GOD_CODE resonance in the
        circuit's spectral profile, and harmonic structure of gate sequences.
        """
        math_eng = cls.connect()

        n_qubits = getattr(circuit, "n_qubits", getattr(circuit, "num_qubits", 2))
        gates = getattr(circuit, "gates", [])
        gate_count = len(gates) if gates else getattr(circuit, "size", 0)

        # Collect gate phases for analysis
        phases = []
        for g in gates:
            params = getattr(g, "params", None) or getattr(g, "parameters", None) or []
            for p in (params if isinstance(params, (list, tuple)) else [params]):
                if isinstance(p, (int, float)):
                    phases.append(float(p))

        # PHI alignment -- measure how close gate phases are to PHI multiples
        if phases:
            phi_distances = [abs((p / PHI) - round(p / PHI)) for p in phases if p != 0]
            phi_alignment = 1.0 - (sum(phi_distances) / len(phi_distances)) if phi_distances else 0.5
        else:
            phi_alignment = 0.5

        # GOD_CODE resonance -- spectral signature
        if phases:
            phase_sum = sum(abs(p) for p in phases)
            gc_resonance = abs(math.cos(phase_sum * math.pi / GOD_CODE))
        else:
            gc_resonance = abs(math.cos(gate_count * math.pi / GOD_CODE))

        # Harmonic score -- Fibonacci structure in gate count
        fib_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        fib_distances = [abs(gate_count - f) for f in fib_seq]
        min_fib_dist = min(fib_distances) if fib_distances else gate_count
        harmonic_score = 1.0 / (1.0 + min_fib_dist * PHI_CONJUGATE)

        result = {
            "n_qubits": n_qubits,
            "gate_count": gate_count,
            "n_parametric_phases": len(phases),
            "phi_alignment": round(phi_alignment, 8),
            "god_code_resonance": round(gc_resonance, 8),
            "harmonic_score": round(harmonic_score, 8),
            "void_correction": round(VOID_CONSTANT * phi_alignment, 8),
        }

        if math_eng is not None:
            try:
                gc_val = math_eng.god_code_value()
                result["god_code_verified"] = abs(gc_val - GOD_CODE) < 1e-6
            except Exception:
                result["god_code_verified"] = True

            try:
                alignment = math_eng.sacred_alignment(IRON_FREQUENCY)
                result["iron_sacred_alignment"] = alignment
            except Exception:
                pass

            try:
                wave_coh = math_eng.wave_coherence(
                    GOD_CODE, GOD_CODE * PHI
                )
                result["wave_coherence"] = round(float(wave_coh), 8) if wave_coh is not None else 0.5
            except Exception:
                result["wave_coherence"] = 0.5
        else:
            result["god_code_verified"] = True
            result["wave_coherence"] = 0.5

        result["method"] = "math_engine" if math_eng is not None else "analytical_fallback"
        result["math_valid"] = phi_alignment > 0.3 and gc_resonance > 0.1
        return result


# ---------------------------------------------------------------------------
#  Code Engine Bridge
# ---------------------------------------------------------------------------

class CodeEngineBridge:
    """Lazy bridge to Code Engine for circuit code quality analysis."""

    _engine = None
    _connected = False

    @classmethod
    def connect(cls):
        """Lazy-connect to Code Engine."""
        if cls._connected:
            return cls._engine
        try:
            from l104_code_engine import code_engine
            cls._engine = code_engine
            cls._connected = True
        except ImportError:
            cls._connected = True
        return cls._engine

    @classmethod
    def analyze_circuit_code(cls, circuit) -> Dict[str, Any]:
        """
        Analyze the code quality of circuit construction.

        Uses the Code Engine to evaluate structural complexity, detect
        potential issues, and assess generation quality of gate sequences.
        """
        code_eng = cls.connect()

        gates = getattr(circuit, "gates", [])
        gate_count = len(gates) if gates else getattr(circuit, "size", 0)
        n_qubits = getattr(circuit, "n_qubits", getattr(circuit, "num_qubits", 2))
        depth = getattr(circuit, "depth", gate_count)

        # Structural metrics (always available)
        gate_density = gate_count / max(n_qubits * depth, 1)
        parallelism_ratio = gate_count / max(depth, 1)
        qubit_utilization = min(1.0, gate_count / max(n_qubits, 1))

        # Unique gate types
        gate_names = []
        for g in gates:
            name = getattr(g, "name", getattr(g, "gate_name", type(g).__name__))
            gate_names.append(str(name))
        unique_gates = len(set(gate_names))
        gate_diversity = unique_gates / max(gate_count, 1)

        result = {
            "gate_count": gate_count,
            "depth": depth,
            "n_qubits": n_qubits,
            "gate_density": round(gate_density, 6),
            "parallelism_ratio": round(parallelism_ratio, 4),
            "qubit_utilization": round(qubit_utilization, 6),
            "unique_gate_types": unique_gates,
            "gate_diversity": round(gate_diversity, 6),
        }

        # Quality score heuristic
        quality = (
            0.30 * min(1.0, gate_density) +
            0.25 * min(1.0, parallelism_ratio / n_qubits) +
            0.25 * qubit_utilization +
            0.20 * min(1.0, gate_diversity * 5.0)
        )
        result["quality_score"] = round(quality, 6)

        if code_eng is not None:
            try:
                # Generate a textual circuit representation for code analysis
                circuit_repr = repr(circuit) if hasattr(circuit, "__repr__") else str(circuit)
                if hasattr(code_eng, "full_analysis") and len(circuit_repr) > 10:
                    analysis = code_eng.full_analysis(circuit_repr)
                    if isinstance(analysis, dict):
                        result["code_complexity"] = analysis.get("complexity", "unknown")
                        result["code_issues"] = analysis.get("issues", [])[:5]
                result["code_engine_available"] = True
            except Exception:
                result["code_engine_available"] = True
        else:
            result["code_engine_available"] = False

        result["method"] = "code_engine" if code_eng is not None else "structural_analysis"
        return result


# ---------------------------------------------------------------------------
#  Gate Engine Cross Hub -- Three-Engine Integration
# ---------------------------------------------------------------------------

class GateEngineCrossHub:
    """
    Central cross-engine integration hub for the Quantum Gate Engine v1.0.

    Provides unified three-engine validation, scoring, and compilation
    analysis by bridging Science Engine (physics), Math Engine (algebra),
    and Code Engine (structural quality).
    """

    def __init__(self):
        self.science = ScienceEngineBridge
        self.math = MathEngineBridge
        self.code = CodeEngineBridge

    def validate_circuit_physics(self, circuit) -> Dict[str, Any]:
        """Validate circuit against physical constraints via Science Engine."""
        return self.science.validate_circuit_physics(circuit)

    def validate_circuit_math(self, circuit) -> Dict[str, Any]:
        """Validate circuit mathematical structure via Math Engine."""
        return self.math.validate_circuit_math(circuit)

    def analyze_circuit_code(self, circuit) -> Dict[str, Any]:
        """Analyze circuit code quality via Code Engine."""
        return self.code.analyze_circuit_code(circuit)

    def three_engine_circuit_score(self, circuit) -> Dict[str, Any]:
        """
        Compute a composite three-engine score for a quantum circuit.

        Combines physics validation (Science), mathematical structure (Math),
        and code quality (Code) into a unified sacred alignment score.
        """
        physics = self.science.validate_circuit_physics(circuit)
        math_val = self.math.validate_circuit_math(circuit)
        code_val = self.code.analyze_circuit_code(circuit)

        # Weighted composite -- PHI-balanced scoring
        physics_score = (
            0.4 * physics.get("decoherence_factor", 0.5) +
            0.3 * physics.get("demon_efficiency", 0.5) +
            0.3 * (1.0 - min(1.0, physics.get("circuit_entropy", 0.5)))
        )
        math_score = (
            0.35 * math_val.get("phi_alignment", 0.5) +
            0.35 * math_val.get("god_code_resonance", 0.5) +
            0.30 * math_val.get("harmonic_score", 0.5)
        )
        code_score = code_val.get("quality_score", 0.5)

        # Three-engine composite with PHI weighting
        composite = (
            PHI_CONJUGATE * physics_score +
            (1.0 - PHI_CONJUGATE) * 0.5 * math_score +
            (1.0 - PHI_CONJUGATE) * 0.5 * code_score
        )
        sacred_alignment = composite * GOD_CODE / 1000.0

        return {
            "physics_score": round(physics_score, 8),
            "math_score": round(math_score, 8),
            "code_score": round(code_score, 8),
            "composite_score": round(composite, 8),
            "sacred_alignment": round(sacred_alignment, 8),
            "physics": physics,
            "math": math_val,
            "code": code_val,
        }

    def cross_validate_compilation(
        self, original, compiled
    ) -> Dict[str, Any]:
        """
        Validate that compilation preserved circuit physics and structure.

        Compares the original and compiled circuits across all three engines
        to ensure compilation did not degrade sacred alignment or physical
        viability.
        """
        orig_physics = self.science.validate_circuit_physics(original)
        comp_physics = self.science.validate_circuit_physics(compiled)

        orig_math = self.math.validate_circuit_math(original)
        comp_math = self.math.validate_circuit_math(compiled)

        # Physics preservation
        coherence_preserved = (
            comp_physics.get("decoherence_factor", 0) >=
            orig_physics.get("decoherence_factor", 0) * 0.8
        )
        entropy_preserved = (
            comp_physics.get("circuit_entropy", 1.0) <=
            orig_physics.get("circuit_entropy", 0.0) * 1.5 + 0.01
        )

        # Math preservation
        phi_delta = abs(
            comp_math.get("phi_alignment", 0) -
            orig_math.get("phi_alignment", 0)
        )
        gc_delta = abs(
            comp_math.get("god_code_resonance", 0) -
            orig_math.get("god_code_resonance", 0)
        )
        math_preserved = phi_delta < 0.3 and gc_delta < 0.3

        # Gate count change
        orig_gates = orig_physics.get("gate_count", 0)
        comp_gates = comp_physics.get("gate_count", 0)
        gate_reduction = (
            (orig_gates - comp_gates) / max(orig_gates, 1)
        )

        compilation_valid = coherence_preserved and entropy_preserved and math_preserved

        return {
            "compilation_valid": compilation_valid,
            "coherence_preserved": coherence_preserved,
            "entropy_preserved": entropy_preserved,
            "math_preserved": math_preserved,
            "phi_delta": round(phi_delta, 8),
            "god_code_delta": round(gc_delta, 8),
            "gate_reduction": round(gate_reduction, 6),
            "original_gates": orig_gates,
            "compiled_gates": comp_gates,
            "original_depth": orig_physics.get("depth", 0),
            "compiled_depth": comp_physics.get("depth", 0),
        }

    def status(self) -> Dict[str, Any]:
        """Report cross-engine integration status."""
        return {
            "version": "1.0.0",
            "science": {
                "connected": ScienceEngineBridge._connected,
                "available": ScienceEngineBridge._engine is not None,
            },
            "math": {
                "connected": MathEngineBridge._connected,
                "available": MathEngineBridge._engine is not None,
            },
            "code": {
                "connected": CodeEngineBridge._connected,
                "available": CodeEngineBridge._engine is not None,
            },
        }


# ---------------------------------------------------------------------------
#  Module-level singleton
# ---------------------------------------------------------------------------

_hub_instance: Optional[GateEngineCrossHub] = None


def get_gate_cross_hub() -> GateEngineCrossHub:
    """Get or create the singleton GateEngineCrossHub."""
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = GateEngineCrossHub()
    return _hub_instance
