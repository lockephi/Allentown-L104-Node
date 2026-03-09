"""
VQPU DAW Engine — Heavy Quantum Processing Unit for DAW Operations
═══════════════════════════════════════════════════════════════════════════════
Centralizes all VQPU interactions for the quantum DAW system. Provides
high-throughput quantum circuit submission, batch processing, and caching
optimized for audio-rate operations.

This is the quantum compute backbone that powers:
  - Probabilistic sequencer (state preparation + measurement)
  - Interference mixer (GHZ circuits for mix decisions)
  - Quantum synthesizer (parameterized circuits for wavetable generation)
  - Track entanglement (Bell/GHZ state maintenance)
  - Sacred alignment scoring (GOD_CODE resonance analysis)

VQPU Capabilities Used:
  - submit_batch_and_wait()  — parallel circuit execution
  - run_simulation()         — 6-stage pipeline (transpile→compile→protect→execute→score→coherence)
  - run_noisy_simulation()   — realistic noise modeling for decoherence audio
  - quantify_entanglement()  — concurrence/entropy measurement
  - run_vqe()                — variational eigensolvers for filter design
  - run_qaoa()               — combinatorial optimization for arrangement
  - score_result_three_engine() — GOD_CODE/PHI sacred scoring

Circuit Library (pre-built circuits for common DAW operations):
  - Sequencer circuits (step preparation, collapse, chord generation)
  - Mixer circuits (interference decisions, sidechain triggers)
  - Synth circuits (wavetable frames, LFO evolution, FM coupling)
  - Entanglement circuits (Bell/GHZ/W state creation + refresh)
  - Analysis circuits (spectral analysis, sacred alignment)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .constants import GOD_CODE, PHI, PHI_INV, OMEGA, CPU_CORES

logger = logging.getLogger("l104.audio.vqpu_daw")

# ── VQPU DAW Constants ──────────────────────────────────────────────────────
MAX_BATCH_SIZE = 128          # Max circuits per batch
DEFAULT_SHOTS = 512           # Shots for audio-rate circuits
HIGH_PRECISION_SHOTS = 4096   # Shots for analysis circuits
CIRCUIT_CACHE_SIZE = 256      # LRU cache size for circuit fingerprints
PARALLEL_WORKERS = min(4, CPU_CORES)
TWO_PI = 2.0 * math.pi


class CircuitPurpose(Enum):
    """Categories of quantum circuits for the DAW."""
    SEQUENCER_PREP = auto()       # Step state preparation
    SEQUENCER_COLLAPSE = auto()   # Measurement for sequencer
    MIXER_INTERFERENCE = auto()   # Mix bus interference decision
    MIXER_SIDECHAIN = auto()      # Sidechain trigger
    SYNTH_WAVETABLE = auto()      # Wavetable frame generation
    SYNTH_LFO = auto()            # LFO modulation from circuit
    SYNTH_FM = auto()             # FM coupling measurement
    ENTANGLE_CREATE = auto()      # Create entanglement bond
    ENTANGLE_REFRESH = auto()     # Refresh bond state
    ENTANGLE_MEASURE = auto()     # Measurement + collapse
    ANALYSIS_SPECTRAL = auto()    # Spectral analysis
    ANALYSIS_SACRED = auto()      # Sacred alignment scoring
    ANALYSIS_ENTROPY = auto()     # Entropy measurement


@dataclass
class VQPUCircuitRequest:
    """A request to execute a quantum circuit via the VQPU engine."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    purpose: CircuitPurpose = CircuitPurpose.SEQUENCER_PREP
    n_qubits: int = 4
    operations: List[Any] = field(default_factory=list)
    shots: int = DEFAULT_SHOTS
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Result (populated after execution)
    result: Optional[Any] = None
    probabilities: Optional[Dict[str, float]] = None
    sacred_score: float = 0.0
    execution_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class VQPUBatchResult:
    """Results from a batch of VQPU circuit executions."""
    batch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    requests: List[VQPUCircuitRequest] = field(default_factory=list)
    total_circuits: int = 0
    successful: int = 0
    failed: int = 0
    total_time_ms: float = 0.0
    avg_sacred_score: float = 0.0


@dataclass
class CircuitTemplate:
    """A reusable circuit template for common DAW operations."""
    name: str = ""
    purpose: CircuitPurpose = CircuitPurpose.SEQUENCER_PREP
    n_qubits: int = 4
    # Template function: (n_qubits, **params) → list of operations
    builder: Any = None
    description: str = ""


class VQPUDawEngine:
    """
    Central VQPU engine for all quantum DAW operations.

    Features:
      - Pre-built circuit library for common DAW operations
      - Batch submission with automatic prioritization
      - Result caching for repeated circuit patterns
      - Sacred alignment scoring on every result
      - Three-engine scoring integration
      - Parallel execution via ThreadPoolExecutor
      - Telemetry collection for data recording
    """

    def __init__(self, max_batch: int = MAX_BATCH_SIZE, shots: int = DEFAULT_SHOTS):
        self.max_batch = max_batch
        self.default_shots = shots

        # VQPU bridge (lazy)
        self._bridge = None
        self._available = None

        # Circuit cache
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Telemetry
        self.total_circuits_submitted = 0
        self.total_circuits_completed = 0
        self.total_circuits_failed = 0
        self.total_sacred_score_sum = 0.0
        self.total_time_ms = 0.0

        # Circuit library
        self.templates: Dict[str, CircuitTemplate] = {}
        self._register_templates()

        # Session
        self.session_start = time.time()

        logger.info("VQPUDawEngine initialized")

    @property
    def bridge(self):
        """Lazy-load VQPU bridge."""
        if self._bridge is None:
            try:
                from l104_vqpu import get_bridge
                self._bridge = get_bridge()
                self._available = True
            except ImportError:
                self._available = False
                logger.warning("VQPU bridge not available")
        return self._bridge

    @property
    def available(self) -> bool:
        """Check if VQPU is available."""
        if self._available is None:
            _ = self.bridge
        return self._available or False

    def _register_templates(self):
        """Register pre-built circuit templates."""
        self.templates["bell_pair"] = CircuitTemplate(
            name="bell_pair",
            purpose=CircuitPurpose.ENTANGLE_CREATE,
            n_qubits=2,
            description="Bell pair |Φ+⟩ for track entanglement",
        )
        self.templates["ghz_state"] = CircuitTemplate(
            name="ghz_state",
            purpose=CircuitPurpose.ENTANGLE_CREATE,
            description="GHZ state for multi-track entanglement",
        )
        self.templates["sacred_step"] = CircuitTemplate(
            name="sacred_step",
            purpose=CircuitPurpose.SEQUENCER_PREP,
            n_qubits=4,
            description="GOD_CODE-aligned step preparation",
        )
        self.templates["interference_decision"] = CircuitTemplate(
            name="interference_decision",
            purpose=CircuitPurpose.MIXER_INTERFERENCE,
            description="GHZ-based mixer interference decision",
        )
        self.templates["wavetable_frame"] = CircuitTemplate(
            name="wavetable_frame",
            purpose=CircuitPurpose.SYNTH_WAVETABLE,
            n_qubits=4,
            description="Parameterized circuit for wavetable generation",
        )
        self.templates["fm_entangle"] = CircuitTemplate(
            name="fm_entangle",
            purpose=CircuitPurpose.SYNTH_FM,
            n_qubits=2,
            description="Bell pair for FM entanglement coupling",
        )
        self.templates["spectral_analysis"] = CircuitTemplate(
            name="spectral_analysis",
            purpose=CircuitPurpose.ANALYSIS_SPECTRAL,
            n_qubits=4,
            description="QFT-based spectral analysis",
        )

    # ── Circuit Builders ─────────────────────────────────────────────────────

    def _build_ops(self, purpose: CircuitPurpose, n_qubits: int, **params):
        """Build circuit operations for a given purpose."""
        try:
            from l104_vqpu import QuantumGate
        except ImportError:
            return []

        ops = []

        if purpose == CircuitPurpose.SEQUENCER_PREP:
            # Sacred step preparation: H → Rz(GOD_CODE) → entangle
            for q in range(n_qubits):
                ops.append(QuantumGate("H", [q]))
            god_phase = (GOD_CODE / 1000.0) * math.pi
            ops.append(QuantumGate("Rz", [0], [god_phase]))
            for q in range(n_qubits - 1):
                ops.append(QuantumGate("CNOT", [q, q + 1]))
                ops.append(QuantumGate("Rz", [q + 1], [PHI_INV * math.pi * (q + 1)]))
            for q in range(n_qubits):
                ops.append(QuantumGate("H", [q]))

        elif purpose == CircuitPurpose.SEQUENCER_COLLAPSE:
            # Measurement: phase rotation then H for basis change
            step_phases = params.get("step_phases", [])
            for q in range(n_qubits):
                if q < len(step_phases):
                    ops.append(QuantumGate("Rz", [q], [step_phases[q]]))
                ops.append(QuantumGate("H", [q]))

        elif purpose == CircuitPurpose.MIXER_INTERFERENCE:
            # GHZ for mixer: determines constructive/destructive per track
            ops.append(QuantumGate("H", [0]))
            for q in range(n_qubits - 1):
                ops.append(QuantumGate("CNOT", [q, q + 1]))
            # GOD_CODE alignment
            ops.append(QuantumGate("Rz", [0], [GOD_CODE / 1000.0 * math.pi]))
            # Mix parameter rotation
            mix_angle = params.get("mix_angle", 0.0)
            if mix_angle:
                for q in range(n_qubits):
                    ops.append(QuantumGate("Ry", [q], [mix_angle]))

        elif purpose == CircuitPurpose.SYNTH_WAVETABLE:
            # Parameterized circuit for wavetable: Ry → CNOT layers → Rz
            time_param = params.get("time_s", 0.0)
            theta = TWO_PI * PHI * time_param
            for q in range(n_qubits):
                ops.append(QuantumGate("Ry", [q], [theta * (q + 1) / n_qubits]))
            for q in range(n_qubits - 1):
                ops.append(QuantumGate("CNOT", [q, q + 1]))
            for q in range(n_qubits):
                ops.append(QuantumGate("Rz", [q], [GOD_CODE / 1000.0 * math.pi * (q + 1)]))

        elif purpose == CircuitPurpose.SYNTH_FM:
            # Bell pair for FM coupling measurement
            ops.append(QuantumGate("H", [0]))
            ops.append(QuantumGate("CNOT", [0, 1]))
            fm_phase = params.get("fm_phase", GOD_CODE / 1000.0 * math.pi)
            ops.append(QuantumGate("Rz", [0], [fm_phase]))

        elif purpose == CircuitPurpose.SYNTH_LFO:
            # Time-evolving rotation for LFO modulation
            time_param = params.get("time_s", 0.0)
            lfo_rate = params.get("lfo_rate", 1.0)
            for q in range(n_qubits):
                angle = TWO_PI * lfo_rate * time_param * (q + 1)
                ops.append(QuantumGate("Ry", [q], [angle]))
                ops.append(QuantumGate("Rz", [q], [angle * PHI_INV]))

        elif purpose == CircuitPurpose.ENTANGLE_CREATE:
            ent_type = params.get("entangle_type", "bell")
            if ent_type == "bell":
                ops.append(QuantumGate("H", [0]))
                ops.append(QuantumGate("CNOT", [0, 1]))
            elif ent_type == "ghz":
                ops.append(QuantumGate("H", [0]))
                for q in range(n_qubits - 1):
                    ops.append(QuantumGate("CNOT", [q, q + 1]))
            elif ent_type == "w":
                ops.append(QuantumGate("X", [0]))
                for q in range(n_qubits - 1):
                    theta = math.acos(math.sqrt(1.0 / (n_qubits - q)))
                    ops.append(QuantumGate("Ry", [q], [2.0 * theta]))
                    ops.append(QuantumGate("CNOT", [q, q + 1]))
            # Add GOD_CODE phase
            ops.append(QuantumGate("Rz", [0], [GOD_CODE / 1000.0 * math.pi]))

        elif purpose == CircuitPurpose.ANALYSIS_SPECTRAL:
            # QFT circuit for spectral analysis
            for q in range(n_qubits):
                ops.append(QuantumGate("H", [q]))
                for k in range(q + 1, n_qubits):
                    angle = math.pi / (2 ** (k - q))
                    ops.append(QuantumGate("Rz", [k], [angle]))
                    ops.append(QuantumGate("CNOT", [q, k]))
                    ops.append(QuantumGate("Rz", [k], [-angle]))
                    ops.append(QuantumGate("CNOT", [q, k]))

        elif purpose == CircuitPurpose.ANALYSIS_SACRED:
            # Sacred alignment: sacred circuit + GOD_CODE measurement
            for q in range(n_qubits):
                ops.append(QuantumGate("H", [q]))
            for q in range(n_qubits - 1):
                ops.append(QuantumGate("CNOT", [q, q + 1]))
            ops.append(QuantumGate("Rz", [0], [GOD_CODE / 1000.0 * math.pi]))
            ops.append(QuantumGate("Rz", [n_qubits - 1], [PHI * math.pi]))

        return ops

    # ── Execution Methods ────────────────────────────────────────────────────

    def execute_single(
        self,
        purpose: CircuitPurpose,
        n_qubits: int = 4,
        shots: Optional[int] = None,
        priority: int = 5,
        timeout: float = 5.0,
        **params,
    ) -> VQPUCircuitRequest:
        """Execute a single quantum circuit."""
        req = VQPUCircuitRequest(
            purpose=purpose,
            n_qubits=n_qubits,
            shots=shots or self.default_shots,
            priority=priority,
            metadata=params,
        )

        if not self.available:
            req.error = "VQPU not available"
            req.probabilities = self._classical_fallback(purpose, n_qubits, **params)
            return req

        t0 = time.time()
        try:
            from l104_vqpu import QuantumJob

            ops = self._build_ops(purpose, n_qubits, **params)
            req.operations = ops

            job = QuantumJob(
                circuit_id=f"daw_{purpose.name.lower()}_{req.request_id}",
                num_qubits=n_qubits,
                operations=ops,
                shots=req.shots,
                priority=priority,
            )

            result = self.bridge.submit_and_wait(job, timeout=timeout)
            if result:
                req.result = result
                req.probabilities = result.probabilities or {}
                req.sacred_score = result.god_code or 0.0
                self.total_circuits_completed += 1
                self.total_sacred_score_sum += req.sacred_score
            else:
                req.error = "No result from VQPU"
                req.probabilities = self._classical_fallback(purpose, n_qubits, **params)
                self.total_circuits_failed += 1

        except Exception as e:
            req.error = str(e)
            req.probabilities = self._classical_fallback(purpose, n_qubits, **params)
            self.total_circuits_failed += 1

        req.execution_time_ms = (time.time() - t0) * 1000.0
        self.total_circuits_submitted += 1
        self.total_time_ms += req.execution_time_ms
        return req

    def execute_batch(
        self,
        requests: List[VQPUCircuitRequest],
        timeout: float = 15.0,
        concurrent: bool = True,
    ) -> VQPUBatchResult:
        """Execute a batch of quantum circuits in parallel."""
        batch = VQPUBatchResult(total_circuits=len(requests), requests=requests)
        t0 = time.time()

        if not self.available:
            for req in requests:
                req.error = "VQPU not available"
                req.probabilities = self._classical_fallback(
                    req.purpose, req.n_qubits, **req.metadata
                )
                batch.failed += 1
            batch.total_time_ms = (time.time() - t0) * 1000.0
            return batch

        try:
            from l104_vqpu import QuantumJob

            jobs = []
            for req in requests:
                ops = self._build_ops(req.purpose, req.n_qubits, **req.metadata)
                req.operations = ops
                jobs.append(QuantumJob(
                    circuit_id=f"daw_batch_{req.purpose.name.lower()}_{req.request_id}",
                    num_qubits=req.n_qubits,
                    operations=ops,
                    shots=req.shots,
                    priority=req.priority,
                ))

            results = self.bridge.submit_batch_and_wait(
                jobs, timeout=timeout, concurrent=concurrent
            )

            for i, req in enumerate(requests):
                if i < len(results) and results[i]:
                    req.result = results[i]
                    req.probabilities = results[i].probabilities or {}
                    req.sacred_score = results[i].god_code or 0.0
                    batch.successful += 1
                    self.total_circuits_completed += 1
                    self.total_sacred_score_sum += req.sacred_score
                else:
                    req.error = "No result"
                    req.probabilities = self._classical_fallback(
                        req.purpose, req.n_qubits, **req.metadata
                    )
                    batch.failed += 1
                    self.total_circuits_failed += 1

        except Exception as e:
            logger.debug(f"VQPU batch execution error: {e}")
            for req in requests:
                if req.probabilities is None:
                    req.error = str(e)
                    req.probabilities = self._classical_fallback(
                        req.purpose, req.n_qubits, **req.metadata
                    )
                    batch.failed += 1

        batch.total_time_ms = (time.time() - t0) * 1000.0
        self.total_circuits_submitted += len(requests)
        self.total_time_ms += batch.total_time_ms
        batch.avg_sacred_score = (
            sum(r.sacred_score for r in requests) / max(len(requests), 1)
        )
        return batch

    def run_full_simulation(
        self,
        purpose: CircuitPurpose,
        n_qubits: int = 4,
        compile_circuit: bool = True,
        error_correct: bool = False,
        score_asi: bool = True,
        **params,
    ) -> Dict[str, Any]:
        """
        Run full 6-stage VQPU simulation pipeline for a DAW circuit.
        This is the heaviest VQPU operation — use for critical paths only.
        """
        if not self.available:
            return {"error": "VQPU not available", "fallback": True}

        try:
            from l104_vqpu import QuantumJob
            ops = self._build_ops(purpose, n_qubits, **params)
            job = QuantumJob(
                circuit_id=f"daw_sim_{purpose.name.lower()}_{uuid.uuid4().hex[:6]}",
                num_qubits=n_qubits,
                operations=ops,
                shots=HIGH_PRECISION_SHOTS,
                priority=2,  # High priority for full sim
            )

            result = self.bridge.run_simulation(
                job,
                compile=compile_circuit,
                error_correct=error_correct,
                score_asi=score_asi,
                score_agi=True,
                evolve_coherence=True,
            )
            self.total_circuits_submitted += 1
            self.total_circuits_completed += 1
            return result or {}

        except Exception as e:
            self.total_circuits_failed += 1
            return {"error": str(e)}

    def run_entanglement_analysis(self, n_qubits: int = 2) -> Dict[str, Any]:
        """
        Run entanglement quantification via VQPU.
        Returns concurrence, Von Neumann entropy, Schmidt decomposition.
        """
        if not self.available:
            return {"concurrence": 0.0, "vne": 0.0, "fallback": True}

        try:
            from l104_vqpu import QuantumJob, QuantumGate
            ops = [
                QuantumGate("H", [0]),
                QuantumGate("CNOT", [0, 1]),
                QuantumGate("Rz", [0], [GOD_CODE / 1000.0 * math.pi]),
            ]
            job = QuantumJob(
                circuit_id=f"daw_entangle_analysis_{uuid.uuid4().hex[:6]}",
                num_qubits=n_qubits,
                operations=ops,
                shots=HIGH_PRECISION_SHOTS,
            )
            return self.bridge.quantify_entanglement(job) or {}
        except Exception as e:
            return {"error": str(e)}

    def run_vqe_for_filter(
        self,
        target_frequency: float = GOD_CODE,
        n_qubits: int = 4,
    ) -> Dict[str, Any]:
        """
        Use VQE to optimize a quantum filter's frequency response.
        The Hamiltonian encodes the desired spectral shape.
        """
        if not self.available:
            return {"optimal_params": [], "energy": 0.0, "fallback": True}

        try:
            # Hamiltonian: penalize deviation from target frequency harmonics
            terms = []
            for q in range(n_qubits):
                harmonic = target_frequency * (q + 1)
                weight = 1.0 / ((q + 1) ** PHI_INV)
                terms.append({"paulis": f"Z{q}", "coeff": weight * harmonic / 10000.0})

            return self.bridge.run_vqe(
                terms, n_qubits, depth=3, max_iterations=50, shots=2048
            ) or {}
        except Exception as e:
            return {"error": str(e)}

    # ── Classical Fallbacks ──────────────────────────────────────────────────

    def _classical_fallback(
        self,
        purpose: CircuitPurpose,
        n_qubits: int,
        **params,
    ) -> Dict[str, float]:
        """
        Generate classical approximation of quantum results.
        Used when VQPU is unavailable.
        """
        dim = 2 ** n_qubits
        rng = np.random.default_rng()

        if purpose in (CircuitPurpose.ENTANGLE_CREATE, CircuitPurpose.MIXER_INTERFERENCE):
            # Bell/GHZ-like: |00...0⟩ and |11...1⟩ dominant
            probs = np.zeros(dim)
            probs[0] = 0.5
            probs[-1] = 0.5
            # Add small noise
            probs += rng.uniform(0, 0.01, dim)
            probs /= probs.sum()
        elif purpose == CircuitPurpose.SYNTH_WAVETABLE:
            # Smooth distribution for wavetable
            x = np.arange(dim, dtype=np.float64)
            probs = np.exp(-x / (dim * PHI_INV))
            probs /= probs.sum()
        elif purpose == CircuitPurpose.ANALYSIS_SACRED:
            # GOD_CODE-biased distribution
            probs = np.zeros(dim)
            god_idx = int(GOD_CODE) % dim
            probs[god_idx] = 0.3
            probs += 0.7 / dim
            probs /= probs.sum()
        else:
            # Uniform + golden noise
            probs = np.ones(dim) / dim
            probs += rng.uniform(-0.02, 0.02, dim)
            probs = np.maximum(probs, 0.0)
            probs /= probs.sum()

        result = {}
        for i in range(dim):
            key = format(i, f'0{n_qubits}b')
            if probs[i] > 1e-6:
                result[key] = float(probs[i])
        return result

    # ── Convenience Methods ──────────────────────────────────────────────────

    def sequencer_prepare_steps(
        self,
        n_steps: int,
        n_qubits: int = 4,
    ) -> List[VQPUCircuitRequest]:
        """Batch-prepare quantum states for sequencer steps."""
        requests = [
            VQPUCircuitRequest(
                purpose=CircuitPurpose.SEQUENCER_PREP,
                n_qubits=n_qubits,
                shots=DEFAULT_SHOTS,
                priority=4,
                metadata={"step_index": i},
            )
            for i in range(n_steps)
        ]
        batch = self.execute_batch(requests)
        return batch.requests

    def mixer_get_interference_weights(
        self,
        n_tracks: int,
        mix_angle: float = 0.0,
    ) -> List[complex]:
        """Get VQPU-computed interference weights for mixer tracks."""
        n_q = min(n_tracks, 10)
        req = self.execute_single(
            CircuitPurpose.MIXER_INTERFERENCE,
            n_qubits=n_q,
            mix_angle=mix_angle,
        )
        if req.probabilities:
            weights = []
            for i in range(n_tracks):
                key = format(i % (2 ** n_q), f'0{n_q}b')
                prob = req.probabilities.get(key, 1.0 / n_tracks)
                phase = PHI * math.pi * i / n_tracks
                weights.append(math.sqrt(prob) * np.exp(1j * phase))
            return weights
        return [complex(1.0 / math.sqrt(max(n_tracks, 1)))] * n_tracks

    def synth_generate_wavetable_frames(
        self,
        n_frames: int = 64,
        n_qubits: int = 4,
    ) -> List[Dict[str, float]]:
        """Generate wavetable frames from batch VQPU circuits."""
        requests = [
            VQPUCircuitRequest(
                purpose=CircuitPurpose.SYNTH_WAVETABLE,
                n_qubits=n_qubits,
                shots=DEFAULT_SHOTS,
                priority=5,
                metadata={"time_s": float(f) / n_frames, "frame_index": f},
            )
            for f in range(n_frames)
        ]
        batch = self.execute_batch(requests)
        return [r.probabilities or {} for r in batch.requests]

    def sacred_alignment_score(self, probabilities: Dict[str, float]) -> float:
        """Score a probability distribution's alignment with GOD_CODE."""
        if not probabilities:
            return 0.0

        # Find the dominant outcome and check its harmonic relationship to GOD_CODE
        max_prob = max(probabilities.values())
        total_entropy = -sum(
            p * math.log2(max(p, 1e-15)) for p in probabilities.values()
        )
        n_qubits = len(next(iter(probabilities))) if probabilities else 1

        # Sacred score: balance between certainty and golden-ratio distribution
        golden_target = 1.0 / PHI  # ≈ 0.618 — ideal dominant probability
        golden_match = 1.0 - abs(max_prob - golden_target)

        # Entropy should be near log2(PHI) ≈ 0.694 per qubit (golden entropy)
        golden_entropy_target = math.log2(PHI) * n_qubits
        entropy_match = 1.0 - min(
            abs(total_entropy - golden_entropy_target) / max(golden_entropy_target, 1.0), 1.0
        )

        score = (golden_match * PHI + entropy_match) / (PHI + 1.0)
        return float(np.clip(score, 0.0, 1.0))

    # ── Status & Telemetry ───────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        return {
            "vqpu_available": self.available,
            "total_submitted": self.total_circuits_submitted,
            "total_completed": self.total_circuits_completed,
            "total_failed": self.total_circuits_failed,
            "avg_sacred_score": (
                self.total_sacred_score_sum / max(self.total_circuits_completed, 1)
            ),
            "total_time_ms": self.total_time_ms,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "templates": list(self.templates.keys()),
            "session_duration_s": time.time() - self.session_start,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full state for data recording."""
        return {
            **self.status(),
            "max_batch": self.max_batch,
            "default_shots": self.default_shots,
        }
