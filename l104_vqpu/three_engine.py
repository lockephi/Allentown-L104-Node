"""L104 VQPU v14.0.0 — Three-Engine Quantum Scorer + Engine Integration + Brain Bridge.

v14.0.0: Version strings centralized via constants.VERSION.
v13.2 (retained): QPU fidelity calibration in composite scoring.
"""

import time
import math
import logging
import threading

_ei_logger = logging.getLogger("L104_ENGINE_INTEGRATION")

from .constants import (
    VERSION, GOD_CODE, PHI, VOID_CONSTANT,
    THREE_ENGINE_WEIGHT_ENTROPY, THREE_ENGINE_WEIGHT_HARMONIC,
    THREE_ENGINE_WEIGHT_WAVE, THREE_ENGINE_WEIGHT_SC,
    THREE_ENGINE_FALLBACK_SCORE,
    BRAIN_INTEGRATION_WEIGHT_SAGE, BRAIN_INTEGRATION_WEIGHT_MANIFOLD,
    BRAIN_INTEGRATION_WEIGHT_ENTANGLE, BRAIN_INTEGRATION_WEIGHT_ORACLE,
    BRAIN_FEEDBACK_FIDELITY_FLOOR, BRAIN_FEEDBACK_MAX_LATENCY_S,
)
from .scoring import SacredAlignmentScorer
from .cache import ScoringCache

__all__ = [
    "ThreeEngineQuantumScorer", "EngineIntegration", "BrainIntegration",
    "_apply_gate_to_circuit", "_circuit_to_ops",
]


class ThreeEngineQuantumScorer:
    """
    Integrates Science Engine, Math Engine, and Code Engine into VQPU
    result scoring.

    Three-Engine Quantum Scoring Dimensions:
    - Entropy:   Science Engine Maxwell's Demon reversal efficiency
                 applied to circuit measurement entropy
    - Harmonic:  Math Engine GOD_CODE sacred alignment + wave
                 coherence at 104 Hz (L104 signature frequency)
    - Wave:      Math Engine PHI-harmonic phase-lock between
                 VOID_CONSTANT and GOD_CODE carrier
    - SC (v9.0): Superconductivity BCS-Heisenberg order parameter
                 from iron-based Fe(26) simulation

    Composite = 0.30×entropy + 0.30×harmonic + 0.20×wave + 0.20×sc
    """

    _science_engine = None
    _math_engine = None
    _code_engine = None
    _init_lock = threading.Lock()

    @classmethod
    def _get_science(cls):
        """Lazy-load ScienceEngine for entropy reversal and coherence."""
        if cls._science_engine is None:
            with cls._init_lock:
                if cls._science_engine is None:
                    try:
                        from l104_science_engine import ScienceEngine
                        cls._science_engine = ScienceEngine()
                    except Exception:
                        pass
        return cls._science_engine

    @classmethod
    def _get_math(cls):
        """Lazy-load MathEngine for harmonic calibration and wave coherence."""
        if cls._math_engine is None:
            with cls._init_lock:
                if cls._math_engine is None:
                    try:
                        from l104_math_engine import MathEngine
                        cls._math_engine = MathEngine()
                    except Exception:
                        pass
        return cls._math_engine

    @classmethod
    def _get_code(cls):
        """Lazy-load code_engine for circuit analysis intelligence."""
        if cls._code_engine is None:
            with cls._init_lock:
                if cls._code_engine is None:
                    try:
                        from l104_code_engine import code_engine
                        cls._code_engine = code_engine
                    except Exception:
                        pass
        return cls._code_engine

    @classmethod
    def entropy_score(cls, measurement_entropy: float) -> float:
        """
        Compute entropy reversal score via Science Engine's Maxwell's Demon.

        Maps circuit measurement Shannon entropy to a demon reversal
        efficiency metric. Higher efficiency = more ordered output.
        """
        se = cls._get_science()
        if se is None:
            return THREE_ENGINE_FALLBACK_SCORE
        try:
            # Clamp entropy to sensible range (0.1–5.0)
            local_entropy = max(0.1, min(5.0, measurement_entropy))
            demon_eff = se.entropy.calculate_demon_efficiency(local_entropy)
            return demon_eff * 2.0
        except Exception:
            return THREE_ENGINE_FALLBACK_SCORE

    @classmethod
    def harmonic_score(cls) -> float:
        """
        Compute harmonic resonance score via Math Engine.

        Validates GOD_CODE sacred alignment and wave coherence at
        104 Hz — the L104 signature frequency.
        """
        me = cls._get_math()
        if me is None:
            return THREE_ENGINE_FALLBACK_SCORE
        try:
            alignment = me.sacred_alignment(GOD_CODE)
            aligned = 1.0 if alignment.get('aligned', False) else 0.0
            wc = me.wave_coherence(104.0, GOD_CODE)
            return aligned * 0.6 + wc * 0.4
        except Exception:
            return THREE_ENGINE_FALLBACK_SCORE

    @classmethod
    def wave_score(cls) -> float:
        """
        Compute wave coherence score from PHI-harmonic phase-locking.

        Tests coherence between PHI carrier and GOD_CODE, and between
        VOID_CONSTANT×1000 carrier and GOD_CODE.
        """
        me = cls._get_math()
        if me is None:
            return THREE_ENGINE_FALLBACK_SCORE
        try:
            wc_phi = me.wave_coherence(PHI, GOD_CODE)
            wc_void = me.wave_coherence(VOID_CONSTANT * 1000, GOD_CODE)
            return (wc_phi + wc_void) / 2.0
        except Exception:
            return THREE_ENGINE_FALLBACK_SCORE

    # v12.3: Background SC computation — avoids 2.2s block on first call
    _sc_bg_thread = None
    _sc_bg_result = None
    _sc_lock = threading.Lock()

    @classmethod
    def _compute_sc_score_impl(cls) -> float:
        """Internal: run the heavy SC simulation (2.2s on Intel i5)."""
        try:
            from l104_god_code_simulator.simulations.vqpu_findings import (
                sim_superconductivity_heisenberg,
            )
            result = sim_superconductivity_heisenberg(4)
            if not result.passed:
                return THREE_ENGINE_FALLBACK_SCORE * 0.5
            import math
            op_score = math.log1p(result.sc_order_parameter * 100) / math.log1p(25)
            cp_score = math.log1p(result.cooper_pair_amplitude * 100) / math.log1p(25)
            ms_score = result.meissner_fraction * 2.0
            return op_score * 0.40 + cp_score * 0.35 + ms_score * 0.25
        except Exception:
            return THREE_ENGINE_FALLBACK_SCORE

    @classmethod
    def _bg_sc_worker(cls):
        """Background worker that computes SC score without blocking pipeline."""
        cls._sc_bg_result = cls._compute_sc_score_impl()

    @classmethod
    def sc_score(cls) -> float:
        """
        v12.3: Non-blocking superconductivity score.

        First call: launches background computation and returns fallback (0.75)
        immediately. Second call onward: returns the real cached score.
        This eliminates the 2.2s first-call penalty on the hot path.

        Thread-safe via double-checked locking on _sc_lock.
        """
        # Fast path: result already computed (no lock needed)
        if cls._sc_bg_result is not None:
            return cls._sc_bg_result
        # Slow path: launch background thread under lock
        with cls._sc_lock:
            if cls._sc_bg_result is not None:
                return cls._sc_bg_result
            if cls._sc_bg_thread is None:
                cls._sc_bg_thread = threading.Thread(
                    target=cls._bg_sc_worker, daemon=True, name="vqpu-sc-bg"
                )
                cls._sc_bg_thread.start()
        # Check if thread finished (non-blocking)
        if cls._sc_bg_thread.is_alive():
            return THREE_ENGINE_FALLBACK_SCORE  # Use fallback until ready
        return cls._sc_bg_result if cls._sc_bg_result is not None else THREE_ENGINE_FALLBACK_SCORE

    @classmethod
    def composite_score(cls, measurement_entropy: float) -> dict:
        """
        Full three-engine composite score for a VQPU circuit result.

        v9.0: Added SC scoring dimension from superconductivity simulation.
        v8.0: Uses ScoringCache for harmonic and wave scores (deterministic)
        and bucketed caching for entropy scores — fixes 96% scoring bottleneck.

        Returns dict with individual + composite scores.
        """
        entropy_s = ScoringCache.get_entropy(measurement_entropy, cls.entropy_score)
        harmonic_s = ScoringCache.get_harmonic(cls.harmonic_score)
        wave_s = ScoringCache.get_wave(cls.wave_score)
        sc_s = ScoringCache.get_sc(cls.sc_score)

        composite = (
            THREE_ENGINE_WEIGHT_ENTROPY * entropy_s
            + THREE_ENGINE_WEIGHT_HARMONIC * harmonic_s
            + THREE_ENGINE_WEIGHT_WAVE * wave_s
            + THREE_ENGINE_WEIGHT_SC * sc_s
        )

        return {
            "entropy_reversal": round(entropy_s, 6),
            "harmonic_resonance": round(harmonic_s, 6),
            "wave_coherence": round(wave_s, 6),
            "sc_heisenberg": round(sc_s, 6),
            "composite": round(composite, 6),
            "qpu_calibrated_composite": round(composite * 0.97475930, 6),  # v13.2: QPU-scaled
            "engines_active": {
                "science": cls._science_engine is not None,
                "math": cls._math_engine is not None,
                "code": cls._code_engine is not None,
                "superconductivity": True,
            },
            "cached": True,
        }

    @classmethod
    def engines_status(cls) -> dict:
        """Return the connection status of all three engines."""
        return {
            "science_engine": cls._get_science() is not None,
            "math_engine": cls._get_math() is not None,
            "code_engine": cls._get_code() is not None,
            "superconductivity": True,
            "manifold_intelligence": True,
            "version": VERSION,
        }


# ═══════════════════════════════════════════════════════════════════
# ENGINE INTEGRATION — Full L104 Engine + Core Pipeline (v11.0)
# ═══════════════════════════════════════════════════════════════════

class EngineIntegration:
    """
    Centralised access to ALL L104 engines and cores for VQPU simulations.

    v11.0 integrates (11 engines + v11.0 daemon cycler):
      - Quantum Gate Engine:     circuit compilation, gate algebra, error correction
      - Quantum Engine:          quantum brain 22-phase pipeline, 26 subsystems
      - Science Engine:          entropy reversal, coherence, physics
      - Math Engine:             harmonic resonance, sacred alignment, wave coherence
      - Code Engine:             circuit code analysis + intelligence
      - ASI Core:                15-dimension scoring, dual-layer engine
      - AGI Core:                13-dimension scoring, cognitive mesh
      - Quantum Data Storage:    QRAM, Shor code, state tomography
      - Quantum Data Analyzer:   15 quantum algorithms for data analysis
      - God Code Simulator:      23 simulations, parametric sweep, feedback loop
      - Manifold Intelligence:   kernel PCA, entanglement network, predictive oracle (v11.0 NEW)

    All engines are lazy-loaded and cached. Missing engines degrade
    gracefully — simulation proceeds with reduced scoring fidelity.
    """

    _gate_engine = None
    _quantum_brain = None
    _science_engine = None
    _math_engine = None
    _code_engine = None
    _asi_core = None
    _agi_core = None
    _quantum_data_storage = None
    _quantum_data_analyzer = None
    _god_code_simulator = None
    _ml_engine = None
    _simulator = None
    _numerical_engine = None
    _logic_gate_engine = None
    _audio_simulation = None
    _search_precog = None

    # ─── Lazy Loaders ───

    @classmethod
    def gate_engine(cls):
        """Quantum Gate Engine: compilation, error correction, gate algebra."""
        if cls._gate_engine is False:
            return None
        if cls._gate_engine is None:
            try:
                from l104_quantum_gate_engine import get_engine
                cls._gate_engine = get_engine()
            except Exception as e:
                _ei_logger.debug("gate_engine unavailable: %s", e)
                cls._gate_engine = False
                return None
        return cls._gate_engine

    @classmethod
    def quantum_brain(cls):
        """Quantum Engine brain: 22-phase pipeline orchestrator (26 subsystems)."""
        if cls._quantum_brain is False:
            return None
        if cls._quantum_brain is None:
            try:
                from l104_quantum_engine import quantum_brain
                cls._quantum_brain = quantum_brain
            except Exception as e:
                _ei_logger.debug("quantum_brain unavailable: %s", e)
                cls._quantum_brain = False
                return None
        return cls._quantum_brain

    @classmethod
    def science_engine(cls):
        """Science Engine: entropy, coherence, physics."""
        if cls._science_engine is False:
            return None
        if cls._science_engine is None:
            try:
                from l104_science_engine import ScienceEngine
                cls._science_engine = ScienceEngine()
            except Exception as e:
                _ei_logger.debug("science_engine unavailable: %s", e)
                cls._science_engine = False
                return None
        return cls._science_engine

    @classmethod
    def math_engine(cls):
        """Math Engine: harmonic, sacred alignment, wave coherence."""
        if cls._math_engine is False:
            return None
        if cls._math_engine is None:
            try:
                from l104_math_engine import MathEngine
                cls._math_engine = MathEngine()
            except Exception as e:
                _ei_logger.debug("math_engine unavailable: %s", e)
                cls._math_engine = False
                return None
        return cls._math_engine

    @classmethod
    def code_engine(cls):
        """Code Engine: analysis, intelligence."""
        if cls._code_engine is False:
            return None
        if cls._code_engine is None:
            try:
                from l104_code_engine import code_engine
                cls._code_engine = code_engine
            except Exception as e:
                _ei_logger.debug("code_engine unavailable: %s", e)
                cls._code_engine = False
                return None
        return cls._code_engine

    @classmethod
    def asi_core(cls):
        """ASI Core: 15-dimension scoring, dual-layer engine."""
        if cls._asi_core is False:
            return None
        if cls._asi_core is None:
            try:
                from l104_asi import asi_core
                cls._asi_core = asi_core
            except Exception as e:
                _ei_logger.debug("asi_core unavailable: %s", e)
                cls._asi_core = False
                return None
        return cls._asi_core

    @classmethod
    def agi_core(cls):
        """AGI Core: 13-dimension scoring, cognitive mesh."""
        if cls._agi_core is False:
            return None
        if cls._agi_core is None:
            try:
                from l104_agi import agi_core
                cls._agi_core = agi_core
            except Exception as e:
                _ei_logger.debug("agi_core unavailable: %s", e)
                cls._agi_core = False
                return None
        return cls._agi_core

    @classmethod
    def quantum_data_storage(cls):
        """Quantum Data Storage: QRAM, Shor code, state tomography."""
        if cls._quantum_data_storage is False:
            return None
        if cls._quantum_data_storage is None:
            try:
                from l104_quantum_data_storage import QuantumDataStorage
                cls._quantum_data_storage = QuantumDataStorage()
            except Exception as e:
                _ei_logger.debug("quantum_data_storage unavailable: %s", e)
                cls._quantum_data_storage = False
                return None
        return cls._quantum_data_storage

    @classmethod
    def quantum_data_analyzer(cls):
        """Quantum Data Analyzer: 15 quantum algorithms for data analysis."""
        if cls._quantum_data_analyzer is False:
            return None
        if cls._quantum_data_analyzer is None:
            try:
                from l104_quantum_data_analyzer import QuantumDataAnalyzer
                cls._quantum_data_analyzer = QuantumDataAnalyzer()
            except Exception as e:
                _ei_logger.debug("quantum_data_analyzer unavailable: %s", e)
                cls._quantum_data_analyzer = False
                return None
        return cls._quantum_data_analyzer

    @classmethod
    def god_code_simulator(cls):
        """God Code Simulator: 23 sims, parametric sweep, feedback loop (v7.0)."""
        if cls._god_code_simulator is False:
            return None
        if cls._god_code_simulator is None:
            try:
                from l104_god_code_simulator import god_code_simulator
                cls._god_code_simulator = god_code_simulator
            except Exception as e:
                _ei_logger.debug("god_code_simulator unavailable: %s", e)
                cls._god_code_simulator = False
                return None
        return cls._god_code_simulator

    @classmethod
    def ml_engine(cls):
        """ML Engine: sacred SVM, random forest, gradient boosting, quantum classifiers."""
        if cls._ml_engine is False:
            return None
        if cls._ml_engine is None:
            try:
                from l104_ml_engine import ml_engine
                cls._ml_engine = ml_engine
            except Exception as e:
                _ei_logger.debug("ml_engine unavailable: %s", e)
                cls._ml_engine = False
                return None
        return cls._ml_engine

    @classmethod
    def simulator(cls):
        """Real-World Simulator: Standard Model physics on GOD_CODE lattice."""
        if cls._simulator is False:
            return None
        if cls._simulator is None:
            try:
                from l104_simulator import RealWorldSimulator
                cls._simulator = RealWorldSimulator()
            except Exception as e:
                _ei_logger.debug("simulator unavailable: %s", e)
                cls._simulator = False
                return None
        return cls._simulator

    @classmethod
    def numerical_engine(cls):
        """Quantum Numerical Builder: 22T token lattice, 100-decimal precision."""
        if cls._numerical_engine is False:
            return None
        if cls._numerical_engine is None:
            try:
                from l104_numerical_engine import QuantumNumericalBuilder
                cls._numerical_engine = QuantumNumericalBuilder()
            except Exception as e:
                _ei_logger.debug("numerical_engine unavailable: %s", e)
                cls._numerical_engine = False
                return None
        return cls._numerical_engine

    @classmethod
    def logic_gate_engine(cls):
        """Logic Gate Builder: analyzers, dynamism, nirvanic, consciousness."""
        if cls._logic_gate_engine is False:
            return None
        if cls._logic_gate_engine is None:
            try:
                from l104_gate_engine import HyperASILogicGateEnvironment
                cls._logic_gate_engine = HyperASILogicGateEnvironment(auto_sync=False)
            except Exception as e:
                _ei_logger.debug("logic_gate_engine unavailable: %s", e)
                cls._logic_gate_engine = False
                return None
        return cls._logic_gate_engine

    @classmethod
    def audio_simulation(cls):
        """Quantum Audio DAW: 17-layer VQPU pipeline, sequencer, mixer, synth."""
        if cls._audio_simulation is False:
            return None
        if cls._audio_simulation is None:
            try:
                from l104_audio_simulation import quantum_daw
                cls._audio_simulation = quantum_daw
            except Exception as e:
                _ei_logger.debug("audio_simulation unavailable: %s", e)
                cls._audio_simulation = False
                return None
        return cls._audio_simulation

    @classmethod
    def search_precog(cls):
        """Three-Engine Search + Precognition: 10 strategies, 8 predictors."""
        if cls._search_precog is False:
            return None
        if cls._search_precog is None:
            try:
                from l104_search import ThreeEngineSearchPrecog
                cls._search_precog = ThreeEngineSearchPrecog()
            except Exception as e:
                _ei_logger.debug("search_precog unavailable: %s", e)
                cls._search_precog = False
                return None
        return cls._search_precog

    # ─── Compilation via Quantum Gate Engine ───

    @classmethod
    def compile_circuit(cls, operations: list, num_qubits: int,
                        gate_set: str = "UNIVERSAL",
                        optimization_level: int = 2) -> dict:
        """
        Compile a circuit through the Quantum Gate Engine.

        Converts raw operations into a GateCircuit, compiles to the
        target gate set with the specified optimization level, and
        returns the compiled operations + compilation metrics.

        Args:
            operations: List of gate operation dicts
            num_qubits: Number of qubits
            gate_set: Target gate set (UNIVERSAL, IBM_EAGLE, CLIFFORD_T,
                      L104_SACRED, IONQ_NATIVE, RIGETTI_ASPEN)
            optimization_level: 0-3 (O0=none, O1=light, O2=standard, O3=aggressive)

        Returns:
            dict with 'operations', 'metrics', 'compiled' flag
        """
        engine = cls.gate_engine()
        if engine is None:
            return {"operations": operations, "compiled": False,
                    "reason": "quantum_gate_engine_unavailable"}

        try:
            from l104_quantum_gate_engine import (
                GateCircuit, GateSet, OptimizationLevel,
            )

            # Build GateCircuit from raw operations
            circ = GateCircuit(num_qubits, name="vqpu_simulation")
            for op in operations:
                gate_name = op.get("gate", "") if isinstance(op, dict) else op.gate
                qubits = op.get("qubits", []) if isinstance(op, dict) else op.qubits
                params = op.get("parameters", []) if isinstance(op, dict) else (op.parameters or [])
                _apply_gate_to_circuit(circ, gate_name, qubits, params)

            # Resolve gate set and optimization level
            gs_map = {
                "UNIVERSAL": GateSet.UNIVERSAL,
                "IBM_EAGLE": GateSet.IBM_EAGLE,
                "CLIFFORD_T": GateSet.CLIFFORD_T,
                "L104_SACRED": GateSet.L104_SACRED,
            }
            ol_map = {
                0: OptimizationLevel.O0,
                1: OptimizationLevel.O1,
                2: OptimizationLevel.O2,
                3: OptimizationLevel.O3,
            }
            target_gs = gs_map.get(gate_set.upper(), GateSet.UNIVERSAL)
            opt_level = ol_map.get(optimization_level, OptimizationLevel.O2)

            # Compile
            result = engine.compile(circ, target_gs, opt_level)

            # Extract compiled ops back to dict format
            compiled_ops = _circuit_to_ops(result.circuit if hasattr(result, 'circuit') else circ)

            return {
                "operations": compiled_ops if compiled_ops else operations,
                "compiled": True,
                "gate_set": gate_set,
                "optimization_level": optimization_level,
                "original_gate_count": len(operations),
                "compiled_gate_count": len(compiled_ops) if compiled_ops else len(operations),
                "depth": getattr(result, 'depth', 0),
                "sacred_alignment": getattr(result, 'sacred_alignment', None),
            }
        except Exception as e:
            return {"operations": operations, "compiled": False,
                    "reason": f"compilation_error: {e}"}

    @classmethod
    def apply_error_correction(cls, operations: list, num_qubits: int,
                               scheme: str = "STEANE_7_1_3",
                               distance: int = 3) -> dict:
        """
        Apply error correction encoding via the Quantum Gate Engine.

        Schemes: SURFACE_CODE, STEANE_7_1_3, FIBONACCI_ANYON, SHOR_9_1_3

        Returns dict with protected operations and encoding metrics.
        """
        engine = cls.gate_engine()
        if engine is None:
            return {"operations": operations, "protected": False,
                    "reason": "quantum_gate_engine_unavailable"}

        try:
            from l104_quantum_gate_engine import (
                GateCircuit, ErrorCorrectionScheme,
            )

            circ = GateCircuit(num_qubits, name="vqpu_ec")
            for op in operations:
                gate_name = op.get("gate", "") if isinstance(op, dict) else op.gate
                qubits = op.get("qubits", []) if isinstance(op, dict) else op.qubits
                params = op.get("parameters", []) if isinstance(op, dict) else (op.parameters or [])
                _apply_gate_to_circuit(circ, gate_name, qubits, params)

            ec_map = {
                "SURFACE_CODE": ErrorCorrectionScheme.SURFACE_CODE,
                "STEANE_7_1_3": ErrorCorrectionScheme.STEANE_7_1_3,
                "FIBONACCI_ANYON": ErrorCorrectionScheme.FIBONACCI_ANYON,
            }
            ec_scheme = ec_map.get(scheme.upper(), ErrorCorrectionScheme.STEANE_7_1_3)

            protected = engine.error_correction.encode(circ, ec_scheme, distance=distance)
            protected_ops = _circuit_to_ops(protected) if protected else operations

            return {
                "operations": protected_ops,
                "protected": True,
                "scheme": scheme,
                "distance": distance,
                "logical_qubits": num_qubits,
                "physical_qubits": len(protected_ops) // max(len(operations), 1) * num_qubits if protected_ops else num_qubits,
            }
        except Exception as e:
            return {"operations": operations, "protected": False,
                    "reason": f"error_correction_error: {e}"}

    @classmethod
    def execute_via_gate_engine(cls, operations: list, num_qubits: int,
                                shots: int = 1024,
                                target: str = "LOCAL_STATEVECTOR") -> dict:
        """
        Execute a circuit through the Quantum Gate Engine's execution targets.

        Targets: LOCAL_STATEVECTOR, COHERENCE_ENGINE, ASI_QUANTUM

        Returns dict with probabilities, sacred alignment, and execution metrics.
        """
        engine = cls.gate_engine()
        if engine is None:
            return {"executed": False, "reason": "quantum_gate_engine_unavailable"}

        try:
            from l104_quantum_gate_engine import GateCircuit, ExecutionTarget

            circ = GateCircuit(num_qubits, name="vqpu_exec")
            for op in operations:
                gate_name = op.get("gate", "") if isinstance(op, dict) else op.gate
                qubits = op.get("qubits", []) if isinstance(op, dict) else op.qubits
                params = op.get("parameters", []) if isinstance(op, dict) else (op.parameters or [])
                _apply_gate_to_circuit(circ, gate_name, qubits, params)

            target_map = {
                "LOCAL_STATEVECTOR": ExecutionTarget.LOCAL_STATEVECTOR,
            }
            # Add targets that may exist
            for t_name in ("COHERENCE_ENGINE", "ASI_QUANTUM", "QISKIT_AER"):
                if hasattr(ExecutionTarget, t_name):
                    target_map[t_name] = getattr(ExecutionTarget, t_name)

            exec_target = target_map.get(target.upper(), ExecutionTarget.LOCAL_STATEVECTOR)
            result = engine.execute(circ, exec_target, shots=shots)

            return {
                "executed": True,
                "probabilities": getattr(result, 'probabilities', {}),
                "counts": getattr(result, 'counts', None),
                "sacred_alignment": getattr(result, 'sacred_alignment', None),
                "fidelity": getattr(result, 'fidelity', None),
                "backend": target,
                "execution_time_ms": getattr(result, 'execution_time_ms', 0.0),
            }
        except Exception as e:
            return {"executed": False, "reason": f"execution_error: {e}"}

    # ─── ASI/AGI Core Scoring ───

    @classmethod
    def asi_score(cls, probabilities: dict, num_qubits: int = 0) -> dict:
        """
        Score simulation results using ASI Core's 15-dimension scoring.

        v12.3: Protected against import hangs on low-RAM systems.
        l104_ml_engine imports sklearn→scipy which can exhaust 4GB RAM.
        Uses threading timeout to abort if scoring takes > 10s.
        """
        asi = cls.asi_core()
        if asi is None:
            return {"available": False, "score": 0.5}

        try:
            import threading
            result_box = [None]
            error_box = [None]

            def _score():
                try:
                    entropy_s = asi.three_engine_entropy_score()
                    harmonic_s = asi.three_engine_harmonic_score()
                    wave_s = asi.three_engine_wave_coherence_score()
                    asi_full = asi.compute_asi_score()
                    composite = asi_full.get("total_score", 0.0) if isinstance(asi_full, dict) else float(asi_full)
                    result_box[0] = {
                        "available": True,
                        "score": composite,
                        "entropy_reversal": entropy_s,
                        "harmonic_resonance": harmonic_s,
                        "wave_coherence": wave_s,
                        "dimensions": 15,
                        "version": getattr(asi, 'version', 'unknown'),
                    }
                except Exception as e:
                    error_box[0] = e

            # v12.3: 10s timeout prevents import-cascade hangs on 4GB Macs
            t = threading.Thread(target=_score, daemon=True)
            t.start()
            t.join(timeout=10.0)

            if t.is_alive():
                # Scoring timed out (likely scipy import hang on low RAM)
                return {"available": False, "score": 0.5, "reason": "timeout_low_ram"}
            if error_box[0]:
                return {"available": False, "score": 0.5}
            return result_box[0] or {"available": False, "score": 0.5}
        except Exception:
            return {"available": False, "score": 0.5}

    @classmethod
    def agi_score(cls, probabilities: dict, num_qubits: int = 0) -> dict:
        """
        Score simulation results using AGI Core's 13-dimension scoring.

        v12.3: Protected against import hangs on low-RAM systems.
        Uses threading timeout to abort if scoring takes > 10s.
        """
        agi = cls.agi_core()
        if agi is None:
            return {"available": False, "score": 0.5}

        try:
            import threading
            result_box = [None]
            error_box = [None]

            def _score():
                try:
                    agi_full = agi.compute_10d_agi_score()
                    composite = agi_full.get("total", 0.0) if isinstance(agi_full, dict) else float(agi_full)
                    result_box[0] = {
                        "available": True,
                        "score": composite,
                        "dimensions": 13,
                        "version": getattr(agi, 'version', 'unknown'),
                    }
                except Exception as e:
                    error_box[0] = e

            t = threading.Thread(target=_score, daemon=True)
            t.start()
            t.join(timeout=10.0)

            if t.is_alive():
                return {"available": False, "score": 0.5, "reason": "timeout_low_ram"}
            if error_box[0]:
                return {"available": False, "score": 0.5}
            return result_box[0] or {"available": False, "score": 0.5}
        except Exception:
            return {"available": False, "score": 0.5}

    # ─── Coherence Evolution ───

    @classmethod
    def evolve_coherence(cls, seed_values: list, steps: int = 10) -> dict:
        """
        Evolve quantum coherence using Science Engine.

        Seeds the coherence subsystem with initial values and evolves
        for N steps, returning the coherence trajectory.
        """
        se = cls.science_engine()
        if se is None:
            return {"evolved": False, "reason": "science_engine_unavailable"}

        try:
            se.coherence.initialize(seed_values)
            se.coherence.evolve(steps)
            state = se.coherence.discover()
            return {
                "evolved": True,
                "steps": steps,
                "coherence_state": state,
            }
        except Exception as e:
            return {"evolved": False, "reason": str(e)}

    # ─── Quantum Data Operations (v6.0) ───

    @classmethod
    def encode_to_quantum(cls, data: bytes, num_qubits: int = 8) -> dict:
        """Encode classical data into quantum state via Quantum Data Storage."""
        storage = cls.quantum_data_storage()
        if storage is None:
            return {"encoded": False, "reason": "quantum_data_storage_unavailable"}
        try:
            encoder = storage.encoder if hasattr(storage, 'encoder') else storage
            result = encoder.encode(data, num_qubits) if hasattr(encoder, 'encode') else {}
            return {"encoded": True, "result": result}
        except Exception as e:
            return {"encoded": False, "reason": str(e)}

    @classmethod
    def run_sc_simulation(cls, n_qubits: int = 4) -> dict:
        """v9.0: Run superconductivity Heisenberg simulation and return payload."""
        try:
            from l104_god_code_simulator.simulations.vqpu_findings import (
                sim_superconductivity_heisenberg,
            )
            result = sim_superconductivity_heisenberg(n_qubits)
            return {
                "passed": result.passed,
                "sc_payload": result.to_superconductivity_payload(),
                "vqpu_metrics": result.to_vqpu_metrics(),
                "scoring": result.to_asi_scoring(),
            }
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def run_vqpu_findings_cycle(cls) -> dict:
        """v9.0: Run all 11 VQPU findings simulations and return summary."""
        try:
            from l104_god_code_simulator.simulations.vqpu_findings import (
                VQPU_FINDINGS_SIMULATIONS,
            )
            results = []
            passed = 0
            for entry in VQPU_FINDINGS_SIMULATIONS:
                name, fn = entry[0], entry[1]
                try:
                    r = fn()
                    results.append({
                        "name": name,
                        "passed": r.passed,
                        "fidelity": round(r.fidelity, 6),
                        "sacred_alignment": round(r.sacred_alignment, 6),
                        "elapsed_ms": round(r.elapsed_ms, 2),
                    })
                    if r.passed:
                        passed += 1
                except Exception as e:
                    results.append({"name": name, "passed": False, "error": str(e)})
            return {
                "total": len(results),
                "passed": passed,
                "pass_rate": round(passed / max(len(results), 1), 4),
                "results": results,
            }
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def analyze_data_quantum(cls, data: list, algorithm: str = "qft") -> dict:
        """Run quantum data analysis algorithm on a dataset."""
        analyzer = cls.quantum_data_analyzer()
        if analyzer is None:
            return {"analyzed": False, "reason": "quantum_data_analyzer_unavailable"}
        try:
            if hasattr(analyzer, 'analyze'):
                return {"analyzed": True, "result": analyzer.analyze(data, algorithm)}
            elif hasattr(analyzer, 'run_algorithm'):
                return {"analyzed": True, "result": analyzer.run_algorithm(algorithm, data)}
            return {"analyzed": False, "reason": "no_compatible_analyze_method"}
        except Exception as e:
            return {"analyzed": False, "reason": str(e)}

    # ─── Status ───

    @classmethod
    def status(cls) -> dict:
        """Connection status of all engines and cores."""
        return {
            "quantum_gate_engine": cls._gate_engine not in (None, False) or cls.gate_engine() is not None,
            "quantum_brain": cls._quantum_brain not in (None, False) or cls.quantum_brain() is not None,
            "science_engine": cls._science_engine not in (None, False) or cls.science_engine() is not None,
            "math_engine": cls._math_engine not in (None, False) or cls.math_engine() is not None,
            "code_engine": cls._code_engine not in (None, False) or cls.code_engine() is not None,
            "asi_core": cls._asi_core not in (None, False) or cls.asi_core() is not None,
            "agi_core": cls._agi_core not in (None, False) or cls.agi_core() is not None,
            "quantum_data_storage": cls._quantum_data_storage not in (None, False) or cls.quantum_data_storage() is not None,
            "quantum_data_analyzer": cls._quantum_data_analyzer not in (None, False) or cls.quantum_data_analyzer() is not None,
            "god_code_simulator": cls._god_code_simulator not in (None, False) or cls.god_code_simulator() is not None,
            "ml_engine": cls._ml_engine not in (None, False) or cls.ml_engine() is not None,
            "simulator": cls._simulator not in (None, False) or cls.simulator() is not None,
            "numerical_engine": cls._numerical_engine not in (None, False) or cls.numerical_engine() is not None,
            "logic_gate_engine": cls._logic_gate_engine not in (None, False) or cls.logic_gate_engine() is not None,
            "audio_simulation": cls._audio_simulation not in (None, False) or cls.audio_simulation() is not None,
            "search_precog": cls._search_precog not in (None, False) or cls.search_precog() is not None,
            "brain_integration": BrainIntegration.is_available(),
            "version": VERSION,
            "engine_count": 18,
            "sc_simulation": True,
            "vqpu_findings": True,
        }


# ═══════════════════════════════════════════════════════════════════
# BRAIN INTEGRATION — Bidirectional VQPU ↔ Quantum Brain Bridge (v13.0)
# ═══════════════════════════════════════════════════════════════════

class BrainIntegration:
    """
    Bidirectional bridge between VQPU and L104QuantumBrain.

    v13.0 integration capabilities:
    - VQPU→Brain: Feed VQPU simulation results into brain link fidelity
    - Brain→VQPU: Brain Sage/manifold/entanglement/oracle enrich VQPU scoring
    - Unified scoring: blended VQPU three-engine + brain multi-phase score
    - Coherence amplification: cross-system coherence feedback loop
    - Brain self-test: diagnostic probes for l104_debug.py integration

    Thread-safe with timeout protection to prevent cross-import deadlocks.
    """

    _brain = None
    _brain_checked = False
    _lock = threading.Lock()
    _last_brain_scores = None
    _last_brain_score_time = 0.0
    _cache_ttl = 30.0  # Cache brain scores for 30s (brain pipeline is expensive)

    @classmethod
    def _get_brain(cls):
        """Lazy-load quantum brain with import deadlock protection."""
        if cls._brain_checked:
            return cls._brain
        with cls._lock:
            if cls._brain_checked:
                return cls._brain
            try:
                from l104_quantum_engine import quantum_brain
                cls._brain = quantum_brain
            except Exception:
                cls._brain = None
            cls._brain_checked = True
        return cls._brain

    @classmethod
    def is_available(cls) -> bool:
        """Check if quantum brain is available for integration."""
        return cls._get_brain() is not None

    @classmethod
    def brain_sage_score(cls) -> dict:
        """
        Get the latest Sage consensus score from the quantum brain.

        Returns cached results if brain was scored within cache TTL.
        Falls back to lightweight scoring if full pipeline not available.
        """
        # Return cached scores if fresh
        now = time.time()
        if (cls._last_brain_scores is not None
                and now - cls._last_brain_score_time < cls._cache_ttl):
            return cls._last_brain_scores

        brain = cls._get_brain()
        if brain is None:
            return {"available": False, "sage_score": 0.0}

        try:
            # Use latest results if brain has run before
            results = getattr(brain, 'results', {})
            sage = results.get("sage", {})
            manifold = results.get("manifold_learning", {})
            entanglement = results.get("entanglement_network", {})
            oracle = results.get("predictive_oracle", {})

            scores = {
                "available": True,
                "sage_score": sage.get("unified_score", 0.0),
                "sage_grade": sage.get("grade", "?"),
                "mean_fidelity": sage.get("mean_fidelity", 0.0),
                "god_code_alignment": sage.get("god_code_alignment", 0.0),
                "phi_resonance": sage.get("phi_resonance", 0.0),
                "manifold_health": manifold.get("manifold_health", 0.0),
                "manifold_dimension": manifold.get("manifold_dimension", 0),
                "phi_fractal_dim": manifold.get("phi_fractal_dimension", 0.0),
                "ghz_fidelity": entanglement.get("mean_ghz_fidelity", 0.0),
                "gmc_score": entanglement.get("mean_gmc", 0.0),
                "network_score": entanglement.get("network_entanglement_score", 0.0),
                "oracle_status": oracle.get("status", "unknown"),
                "oracle_trajectory": oracle.get("alignment_trajectory", "unknown"),
                "total_links": sage.get("total_links", len(getattr(brain, 'links', []))),
                "brain_runs": getattr(brain, 'run_count', 0),
                "brain_version": getattr(brain, 'VERSION', '?'),
            }

            # Cache
            cls._last_brain_scores = scores
            cls._last_brain_score_time = now
            return scores

        except Exception as e:
            return {"available": False, "sage_score": 0.0, "error": str(e)}

    @classmethod
    def brain_composite_score(cls) -> float:
        """
        Compute a composite brain score for VQPU pipeline enrichment.

        Weighted blend:
          0.35 × Sage unified score
          0.25 × Manifold topology health
          0.20 × Multipartite entanglement network score
          0.20 × Oracle confidence (if predictions available)
        """
        data = cls.brain_sage_score()
        if not data.get("available"):
            return THREE_ENGINE_FALLBACK_SCORE

        sage_s = max(0.0, data.get("sage_score", 0.0))
        manifold_s = max(0.0, data.get("manifold_health", 0.0))
        entangle_s = max(0.0, data.get("network_score", 0.0))
        # Oracle confidence: 1.0 if stable/strong trajectory, lower if degrading
        trajectory = data.get("oracle_trajectory", "unknown")
        oracle_map = {
            "strong_ascending": 1.0, "ascending": 0.85, "stable": 0.75,
            "oscillating": 0.6, "descending": 0.4, "strong_descending": 0.2,
        }
        oracle_s = oracle_map.get(trajectory, 0.5)

        composite = (
            BRAIN_INTEGRATION_WEIGHT_SAGE * sage_s
            + BRAIN_INTEGRATION_WEIGHT_MANIFOLD * manifold_s
            + BRAIN_INTEGRATION_WEIGHT_ENTANGLE * entangle_s
            + BRAIN_INTEGRATION_WEIGHT_ORACLE * oracle_s
        )
        return round(composite, 6)

    @classmethod
    def feed_simulation_to_brain(cls, simulation_result: dict) -> dict:
        """
        Feed VQPU simulation results back to the quantum brain.

        Injects VQPU sacred alignment, three-engine scores, and entanglement
        metrics into the brain's temporal memory bank and predictive oracle
        for cross-system coherence amplification.

        Args:
            simulation_result: Output from VQPUBridge.run_simulation()

        Returns:
            Feedback acknowledgment with coherence delta
        """
        brain = cls._get_brain()
        if brain is None:
            return {"fed": False, "reason": "brain_unavailable"}

        try:
            sacred = simulation_result.get("sacred", {})
            three_eng = simulation_result.get("three_engine", {})
            pipeline = simulation_result.get("pipeline", {})

            sacred_score = sacred.get("sacred_score", 0.0)
            if sacred_score < BRAIN_FEEDBACK_FIDELITY_FLOOR:
                return {"fed": False, "reason": "below_fidelity_floor",
                        "sacred_score": sacred_score}

            # Feed to temporal memory bank
            if hasattr(brain, 'temporal_memory'):
                brain.temporal_memory.record({
                    "source": "vqpu_simulation",
                    "sacred_score": sacred_score,
                    "entropy": sacred.get("entropy", 0),
                    "three_engine_composite": three_eng.get("composite", 0),
                    "pipeline_ms": pipeline.get("total_ms", 0),
                    "timestamp": time.time(),
                })

            # Feed to predictive oracle
            if hasattr(brain, 'predictive_oracle'):
                brain.predictive_oracle.record_observation({
                    "vqpu_sacred_score": sacred_score,
                    "vqpu_entropy_reversal": three_eng.get("entropy_reversal", 0),
                    "vqpu_harmonic": three_eng.get("harmonic_resonance", 0),
                    "vqpu_wave": three_eng.get("wave_coherence", 0),
                    "vqpu_sc": three_eng.get("sc_heisenberg", 0),
                    "source": "vqpu_bridge",
                })

            # Feed to feedback bus for cross-builder sync
            if hasattr(brain, 'feedback_bus'):
                brain.feedback_bus.broadcast({
                    "type": "vqpu_simulation_complete",
                    "sender": "vqpu_bridge_v13",
                    "payload": {
                        "sacred_score": sacred_score,
                        "composite": three_eng.get("composite", 0),
                        "pipeline_ms": pipeline.get("total_ms", 0),
                        "event": "vqpu_sim_feedback",
                    },
                })

            return {
                "fed": True,
                "sacred_score": sacred_score,
                "three_engine_composite": three_eng.get("composite", 0),
                "brain_links": len(getattr(brain, 'links', [])),
                "brain_runs": getattr(brain, 'run_count', 0),
            }
        except Exception as e:
            return {"fed": False, "reason": str(e)}

    @classmethod
    def unified_vqpu_brain_score(cls, measurement_entropy: float) -> dict:
        """
        Compute a unified score blending VQPU three-engine and brain intelligence.

        Formula:
          unified = 0.60 × three_engine_composite + 0.40 × brain_composite

        The brain composite includes Sage consensus, manifold topology,
        multipartite entanglement, and predictive oracle dimensions.
        """
        three_eng = ThreeEngineQuantumScorer.composite_score(measurement_entropy)
        brain_data = cls.brain_sage_score()
        brain_composite = cls.brain_composite_score()

        three_eng_score = three_eng.get("composite", THREE_ENGINE_FALLBACK_SCORE)

        # Blend: 60% VQPU engines + 40% brain intelligence
        unified = 0.60 * three_eng_score + 0.40 * brain_composite

        return {
            "unified_score": round(unified, 6),
            "three_engine_score": round(three_eng_score, 6),
            "brain_composite": round(brain_composite, 6),
            "three_engine_detail": three_eng,
            "brain_detail": {
                "sage_score": brain_data.get("sage_score", 0),
                "manifold_health": brain_data.get("manifold_health", 0),
                "network_score": brain_data.get("network_score", 0),
                "oracle_trajectory": brain_data.get("oracle_trajectory", "unknown"),
                "brain_available": brain_data.get("available", False),
            },
            "blend_weights": {"three_engine": 0.60, "brain": 0.40},
            "version": VERSION,
        }

    @classmethod
    def run_brain_self_test(cls) -> dict:
        """
        Run diagnostic probes on the quantum brain for l104_debug.py integration.

        Tests 10 brain subsystem accessibility probes:
        1. Brain singleton availability
        2. Sacred constants alignment
        3. QuantumMathCore accessibility
        4. Scanner subsystem
        5. Link builder subsystem
        6. Sage inference engine
        7. Predictive oracle
        8. Manifold learner
        9. Entanglement network
        10. qLDPC error correction
        """
        results = []
        t0 = time.monotonic()

        brain = cls._get_brain()

        # 1. Brain singleton
        try:
            assert brain is not None, "Brain singleton is None"
            assert hasattr(brain, 'VERSION'), "No VERSION attribute"
            results.append({"test": "brain_singleton", "pass": True,
                            "detail": f"v{brain.VERSION}"})
        except Exception as e:
            results.append({"test": "brain_singleton", "pass": False, "error": str(e)})

        if brain is None:
            return {
                "engine": "quantum_brain",
                "version": "unavailable",
                "tests": results,
                "passed": 0, "total": 1, "all_pass": False,
                "elapsed_ms": round((time.monotonic() - t0) * 1000, 2),
            }

        # 2. Sacred constants
        try:
            from l104_quantum_engine.constants import GOD_CODE as BRAIN_GC, PHI as BRAIN_PHI
            assert abs(BRAIN_GC - GOD_CODE) < 1e-10, "GOD_CODE mismatch VQPU↔Brain"
            assert abs(BRAIN_PHI - PHI) < 1e-10, "PHI mismatch VQPU↔Brain"
            results.append({"test": "constants_alignment", "pass": True,
                            "detail": f"GOD_CODE={GOD_CODE:.10f}"})
        except Exception as e:
            results.append({"test": "constants_alignment", "pass": False, "error": str(e)})

        # 3. QuantumMathCore
        try:
            assert hasattr(brain, 'qmath'), "No qmath attribute"
            bell = brain.qmath.bell_state()
            assert bell is not None, "bell_state() returned None"
            results.append({"test": "quantum_math_core", "pass": True,
                            "detail": "Bell state OK"})
        except Exception as e:
            results.append({"test": "quantum_math_core", "pass": False, "error": str(e)})

        # 4. Scanner
        try:
            assert hasattr(brain, 'scanner'), "No scanner attribute"
            results.append({"test": "scanner_subsystem", "pass": True,
                            "detail": f"Scanner ready"})
        except Exception as e:
            results.append({"test": "scanner_subsystem", "pass": False, "error": str(e)})

        # 5. Link builder
        try:
            assert hasattr(brain, 'link_builder'), "No link_builder"
            results.append({"test": "link_builder", "pass": True,
                            "detail": "Builder ready"})
        except Exception as e:
            results.append({"test": "link_builder", "pass": False, "error": str(e)})

        # 6. Sage inference
        try:
            assert hasattr(brain, 'sage'), "No sage attribute"
            results.append({"test": "sage_inference", "pass": True,
                            "detail": "Sage ready"})
        except Exception as e:
            results.append({"test": "sage_inference", "pass": False, "error": str(e)})

        # 7. Predictive oracle
        try:
            assert hasattr(brain, 'predictive_oracle'), "No predictive_oracle"
            results.append({"test": "predictive_oracle", "pass": True,
                            "detail": "Oracle ready"})
        except Exception as e:
            results.append({"test": "predictive_oracle", "pass": False, "error": str(e)})

        # 8. Manifold learner
        try:
            assert hasattr(brain, 'manifold_learner'), "No manifold_learner"
            results.append({"test": "manifold_learner", "pass": True,
                            "detail": "Manifold ready"})
        except Exception as e:
            results.append({"test": "manifold_learner", "pass": False, "error": str(e)})

        # 9. Entanglement network
        try:
            assert hasattr(brain, 'entanglement_network'), "No entanglement_network"
            results.append({"test": "entanglement_network", "pass": True,
                            "detail": "Network ready"})
        except Exception as e:
            results.append({"test": "entanglement_network", "pass": False, "error": str(e)})

        # 10. qLDPC error correction
        try:
            assert hasattr(brain, 'qldpc_sacred'), "No qldpc_sacred"
            results.append({"test": "qldpc_error_correction", "pass": True,
                            "detail": "qLDPC ready"})
        except Exception as e:
            results.append({"test": "qldpc_error_correction", "pass": False, "error": str(e)})

        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
        passed = sum(1 for r in results if r["pass"])

        return {
            "engine": "quantum_brain",
            "version": brain.VERSION,
            "tests": results,
            "passed": passed,
            "total": len(results),
            "all_pass": passed == len(results),
            "elapsed_ms": elapsed_ms,
            "god_code": GOD_CODE,
            "brain_runs": getattr(brain, 'run_count', 0),
            "brain_links": len(getattr(brain, 'links', [])),
        }

    @classmethod
    def status(cls) -> dict:
        """Brain integration status dashboard."""
        brain = cls._get_brain()
        brain_scores = cls.brain_sage_score() if brain else {}
        return {
            "brain_available": brain is not None,
            "brain_version": getattr(brain, 'VERSION', None) if brain else None,
            "brain_runs": getattr(brain, 'run_count', 0) if brain else 0,
            "brain_links": len(getattr(brain, 'links', [])) if brain else 0,
            "sage_score": brain_scores.get("sage_score", 0),
            "manifold_health": brain_scores.get("manifold_health", 0),
            "network_score": brain_scores.get("network_score", 0),
            "brain_composite": cls.brain_composite_score() if brain else 0,
            "cache_ttl": cls._cache_ttl,
            "version": VERSION,
        }


# ─── Helper: Apply gate to GateCircuit ───

def _apply_gate_to_circuit(circ, gate_name: str, qubits: list, params: list):
    """Map a gate dict to a GateCircuit method call."""
    g = gate_name.upper()
    try:
        if g == "H":
            circ.h(qubits[0])
        elif g in ("X", "NOT"):
            circ.x(qubits[0])
        elif g == "Y":
            circ.y(qubits[0])
        elif g == "Z":
            circ.z(qubits[0])
        elif g == "S":
            circ.s(qubits[0])
        elif g == "T":
            circ.t(qubits[0])
        elif g in ("CX", "CNOT"):
            circ.cx(qubits[0], qubits[1])
        elif g == "CZ":
            circ.cz(qubits[0], qubits[1])
        elif g == "SWAP":
            circ.swap(qubits[0], qubits[1])
        elif g in ("RZ", "ROTATIONZ"):
            circ.rz(params[0] if params else 0, qubits[0])
        elif g in ("RX", "ROTATIONX"):
            circ.rx(params[0] if params else 0, qubits[0])
        elif g in ("RY", "ROTATIONY"):
            circ.ry(params[0] if params else 0, qubits[0])
        elif g == "I":
            pass  # Identity — no-op
        else:
            # Attempt generic append for sacred/topological gates
            from l104_quantum_gate_engine import GateAlgebra
            gate_obj = GateAlgebra.get(gate_name)
            if gate_obj is not None:
                circ.append(gate_obj, qubits)
    except (IndexError, AttributeError):
        pass  # Skip unresolvable gates gracefully


def _circuit_to_ops(circ) -> list:
    """Convert a GateCircuit back to list of operation dicts."""
    ops = []
    try:
        for instruction in circ.instructions:
            op = {
                "gate": instruction.gate.name if hasattr(instruction.gate, 'name') else str(instruction.gate),
                "qubits": list(instruction.qubits) if hasattr(instruction, 'qubits') else [],
            }
            if hasattr(instruction, 'params') and instruction.params:
                op["parameters"] = list(instruction.params)
            elif hasattr(instruction.gate, 'params') and instruction.gate.params:
                op["parameters"] = list(instruction.gate.params)
            ops.append(op)
    except (AttributeError, TypeError):
        pass
    return ops
