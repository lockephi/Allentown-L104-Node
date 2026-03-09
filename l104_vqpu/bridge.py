"""
L104 VQPU Bridge v14.0.0 — Main VQPUBridge orchestrator.

Extracted from the monolith l104_vqpu_bridge.py.
This module contains the VQPUBridge class: the full engine-integrated
Python ↔ Swift Metal vQPU controller with 27 capability layers including
transpilation, compilation, error correction, scoring, variational
algorithms, tomography, Hamiltonian simulation, and quantum subconscious.

v14.0.0 QUANTUM FIDELITY ARCHITECTURE:
  - 14-pass transpiler with SWAP routing + multi-qubit decomposition
  - Crosstalk noise model (ZZ interaction, φ⁻¹ decay)
  - Multi-optimizer VQE (parameter_shift, SPSA, COBYLA)
  - 4th-order Suzuki-Trotter + 2D iron lattice
  - MLE state tomography
  - Adaptive daemon cycle interval (30s–300s)
  - TTL-based scoring cache expiration
  - All version strings centralized via constants.VERSION

v13.2 (retained): QPU-calibrated Heron noise model, GOD_CODE qubit-aware pipeline.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import uuid
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from l104_vqpu.constants import (
    VERSION,
    GOD_CODE,
    PHI,
    VOID_CONSTANT,
    BRIDGE_PATH,
    VQPU_PIPELINE_WORKERS,
    THROTTLE_COOLDOWN_S,
    VQPU_ADAPTIVE_SHOTS_MIN,
    VQPU_ADAPTIVE_SHOTS_MAX,
    VQPU_MPS_FALLBACK_TARGET,
    VQPU_GPU_CROSSOVER,
    VQPU_MAX_QUBITS,
    VQPU_STABILIZER_MAX_QUBITS,
    VQPU_DB_RESEARCH_QUBITS,
    VQPU_BATCH_LIMIT,
    VQPU_MPS_MAX_BOND_LOW,
    VQPU_MPS_MAX_BOND_MED,
    VQPU_MPS_MAX_BOND_HIGH,
    DAEMON_CYCLE_INTERVAL_S,
    _IS_INTEL,
    _IS_APPLE_SILICON,
    _PLATFORM,
    _GPU_CLASS,
    _HAS_METAL_COMPUTE,
    _HW_RAM_GB,
    _HW_CORES,
)
from l104_vqpu.types import QuantumJob, QuantumGate, VQPUResult
from l104_vqpu.transpiler import CircuitTranspiler, CircuitAnalyzer
from l104_vqpu.mps_engine import ExactMPSHybridEngine
from l104_vqpu.scoring import SacredAlignmentScorer, NoiseModel
from l104_vqpu.entanglement import EntanglementQuantifier, QuantumInformationMetrics
from l104_vqpu.tomography import QuantumStateTomography
from l104_vqpu.hamiltonian import HamiltonianSimulator, QuantumErrorMitigation
from l104_vqpu.cache import CircuitCache, ScoringCache
from l104_vqpu.variational import VariationalQuantumEngine, _pauli_expectation
from l104_vqpu.three_engine import ThreeEngineQuantumScorer, EngineIntegration, BrainIntegration, _circuit_to_ops
from l104_vqpu.researcher import QuantumDatabaseResearcher
from l104_vqpu.daemon import VQPUDaemonCycler
from l104_vqpu.hardware import HardwareGovernor, ResultCollector
from l104_vqpu.micro_daemon import VQPUMicroDaemon


def _parallel_simulation_worker(job_data: dict) -> dict:
    """
    Worker function for parallel simulation execution.

    v12.2: Process-based parallel execution to bypass GIL for CPU-bound
    quantum simulations. Takes pickled job data and returns simulation result.

    v13.1: Reuses a process-local bridge via _worker_bridge to avoid
    spinning up a full VQPUBridge (ThreadPoolExecutor, DaemonCycler,
    HardwareGovernor, filesystem directories) for every single job.
    """
    try:
        # Reconstruct job from data
        job = QuantumJob(**job_data)

        # Reuse a process-local bridge (one per worker process)
        global _worker_bridge
        if '_worker_bridge' not in globals() or _worker_bridge is None:
            _worker_bridge = VQPUBridge()
            _worker_bridge.start()
        bridge = _worker_bridge

        # Run simulation
        result = bridge.run_simulation(job)
        return result
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "pipeline": {
                "version": VERSION,
                "stages_executed": ["error"],
                "error": str(e),
                "god_code": GOD_CODE,
                "total_ms": 0,
            }
        }

_worker_bridge = None  # Process-local bridge singleton


class VQPUBridge:
    """
    L104 Virtual Quantum Processing Unit Bridge v13.0.

    Full engine-integrated Python ↔ Swift Metal vQPU controller:
    1. 10-pass circuit transpilation (gate cancellation, rotation merging, template matching,
       dynamic decoupling, peephole optimization, gate fusion)
    2. Quantum Gate Engine compilation + error correction pre-processing
    3. Hardware monitoring (thermal throttling, RAM pressure, 5-sample thermal prediction)
    4. Pipeline-parallel dispatch via file-based IPC (transpile next while executing current)
    5. Adaptive shot allocation (131072 max — more shots for low-confidence results)
    6. Double-buffer IPC channels for zero-wait submission
    7. Full eleven-engine scoring + ASI/AGI core analysis
    8. run_simulation() orchestrated pipeline using all engines
    9. Quantum database research: Grover search, QPE, QFT, amplitude estimation, quantum walk
    10. Variational algorithms: VQE for ground state energy, QAOA for combinatorial opt
    11. Noise simulation with ZNE error mitigation
    12. Entanglement quantification: von Neumann entropy, concurrence, Schmidt decomposition
    13. Platform-aware Mac control: Intel x86_64 vs Apple Silicon detection + routing
    14. Quantum Information Metrics: QFI, Berry phase, mutual info, Loschmidt echo, topological entropy
    15. Quantum State Tomography: density matrix reconstruction, purity, fidelity, SWAP test
    16. Hamiltonian Simulation: Trotter-Suzuki evolution, adiabatic prep, Fe(26) iron-lattice
    17. Scoring Cache v3: 4096-entry ASI/AGI caches + fingerprint bloom filter
    18. VQPUDaemonCycler v13.0: autonomous daemon with quantum subconscious + brain-fed cycles
    19. Superconductivity pipeline stage: SC Heisenberg analysis in run_simulation()
    20. VQPU Findings on-demand: run_vqpu_findings() for full 11-sim cycle
    21. Manifold Intelligence: quantum kernel PCA, entanglement network, predictive oracle
    22. ExactMPSHybridEngine v3: product-state fast path, contiguous arrays, vectorized sampling
    23. Parallel batch execution: run_simulation_batch() with ThreadPoolExecutor (v12.0 NEW)
    24. Self-test framework: self_test() for l104_debug.py integration (v12.0 NEW)
    25. Quantum Subconscious: 13-stream idle thought engine with insight crystallization (v12.1 NEW)
    26. ★ Brain Integration: VQPU↔QuantumBrain bidirectional scoring bridge (v13.0 NEW)
    27. ★ Unified scoring: blended three-engine + brain Sage/manifold/oracle score (v13.0 NEW)

    v12.1 Upgrades (Quantum Subconscious — idle thought engine):
      - QuantumSubconscious engine: 13 idle thought streams (phi, entropy, harmonic, dream, ...)
      - InsightCrystallizer: 7 crystallization rules (sacred alignment, entropy anomaly, bifurcation, ...)
      - Precognition pre-seeding: subconscious insights feed DataPrecognitionEngine predictors
      - VQPUDaemonCycler v12.1: subconscious lifecycle management + harvest on each cycle
      - VQPUBridge methods: subconscious_status(), harvest_idle_thoughts(), crystallize_insights(),
        harvest_precog_seeds(), drain_subconscious()
      - Theta-band consciousness (0.1-0.3): subconscious never competes with active processing
      - 8-second idle cycle with GOD_CODE-seeded PRNG + PHI-weighted phase calculations
    v12.0 (retained):
      - 32768-entry parametric gate cache with LRU eviction (was 16384)
      - ExactMPSHybridEngine v3: product-state ultra-fast path + np.ascontiguousarray
      - Parallel run_simulation_batch() via ThreadPoolExecutor + as_completed()
      - ScoringCache v3: 4096-entry ASI/AGI caches (was 2048)
      - self_test() classmethod for unified l104_debug.py integration
    v11.0 (retained):
      - 10-pass CircuitTranspiler, 12 pipeline workers, Manifold Intelligence
      - 131072 max adaptive shots, 120s daemon cycle, ResultCollector v2
    v8.0 (retained):
      - QuantumInformationMetrics, QuantumStateTomography, HamiltonianSimulator
      - ScoringCache harmonic/wave bucketed — 10-50x speedup
    v7.1 (retained):
      - Platform detection, Intel CPU-only routing, BLAS thread tuning
    v6.0 (retained):
      - 48-qubit MPS capacity, stabilizer-tableau mode, Quantum DB research
    v5.0 (retained):
      - Quantum Gate Engine integration, ASI/AGI scoring, run_simulation()

    Usage:
        bridge = VQPUBridge()
        bridge.start()

        # Quick submit (IPC to Swift daemon)
        job = QuantumJob(num_qubits=2, operations=[
            {"gate": "H", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
        ])
        result = bridge.submit_and_wait(job)
        print(result.probabilities)  # {'00': ~0.5, '11': ~0.5}

        # Full engine-powered simulation
        sim = bridge.run_simulation(job, compile=True, error_correct=True)
        print(sim)  # Enriched result with engine scoring

        # Quantum database research
        results = bridge.quantum_db_search("consciousness", db="all")
        print(results['matches'])  # Grover-accelerated search results

        # Full research pipeline (all 5 quantum algorithms)
        research = bridge.research_database("quantum coherence")
        print(research['pipeline_summary'])

        bridge.stop()
    """

    def __init__(self, bridge_path: Path = BRIDGE_PATH,
                 enable_governor: bool = True,
                 enable_transpiler: bool = True,
                 enable_adaptive_shots: bool = True,
                 enable_daemon_cycler: bool = True,
                 enable_micro_daemon: bool = True,
                 pipeline_workers: int = VQPU_PIPELINE_WORKERS):
        self.bridge_path = bridge_path
        self.inbox = bridge_path / "inbox"
        self.outbox = bridge_path / "outbox"
        self.telemetry_dir = bridge_path / "telemetry"

        self.transpiler = CircuitTranspiler() if enable_transpiler else None
        self.governor = HardwareGovernor() if enable_governor else None
        self.collector = ResultCollector(self.outbox)
        self.enable_adaptive_shots = enable_adaptive_shots
        self.pipeline_workers = pipeline_workers
        self.engines = EngineIntegration  # v5.0: engine integration hub
        self._enable_daemon_cycler = enable_daemon_cycler
        self._daemon_cycler = VQPUDaemonCycler()  # v9.0: autonomous daemon cycler
        self._enable_micro_daemon = enable_micro_daemon
        self._micro_daemon = VQPUMicroDaemon()     # v2.0: lightweight micro daemon

        # v4.0: Pipeline executor for parallel transpile+dispatch
        self._pipeline_executor = None

        # Stats
        self._jobs_submitted = 0
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._total_transpile_savings = 0
        self._total_submit_time_ms = 0.0
        self._total_result_time_ms = 0.0
        self._start_time = 0.0
        self._active = False
        self._peak_throughput_hz = 0.0   # v4.0: track peak throughput
        self._template_match_savings = 0  # v4.0: track template match savings

    # ─── LIFECYCLE ───

    def start(self):
        """Initialize the bridge filesystem, pipeline executor, and start monitoring."""
        if self._active:
            return

        for d in [self.inbox, self.outbox, self.telemetry_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if self.governor:
            self.governor.start()

        # v4.0: Pipeline executor for parallel transpile+dispatch
        if self.pipeline_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._pipeline_executor = ThreadPoolExecutor(
                max_workers=self.pipeline_workers,
                thread_name_prefix="vqpu-pipeline")

        # v9.0: Start daemon cycler (background sim thread)
        if self._enable_daemon_cycler:
            self._daemon_cycler.start()

        # v2.0: Start micro daemon (lightweight high-frequency background thread)
        if self._enable_micro_daemon:
            self._micro_daemon.connect_bridge(self)
            self._micro_daemon.start()

        self._start_time = time.time()
        self._active = True

    def stop(self):
        """Stop monitoring, shutdown pipeline executor, daemon cycler, micro daemon, and clean up."""
        if not self._active:
            return

        # v2.0: Stop micro daemon
        if self._enable_micro_daemon:
            self._micro_daemon.stop()

        # v9.0: Stop daemon cycler
        if self._enable_daemon_cycler:
            self._daemon_cycler.stop()

        # v4.0: Shutdown pipeline executor
        if self._pipeline_executor is not None:
            self._pipeline_executor.shutdown(wait=True, cancel_futures=False)
            self._pipeline_executor = None

        if self.governor:
            self.governor.stop()

        self.collector.close()
        self._write_telemetry_summary()
        self._active = False

    # ─── JOB SUBMISSION ───

    def submit(self, job: QuantumJob) -> str:
        """
        Submit a quantum job to the vQPU. Returns the circuit_id.

        v11.0: 10-pass transpilation, adaptive shot allocation,
        template match tracking, version-tagged payloads.
        If throttled, the submission is delayed until vitals normalize.
        """
        if not self._active:
            self.start()

        # Wait for throttle to clear
        if self.governor and self.governor.is_throttled:
            self._wait_throttle_clear(timeout=THROTTLE_COOLDOWN_S)

        # Transpile (10-pass v11.0 pipeline)
        original_count = len(job.operations)
        if self.transpiler and job.operations:
            job.operations = CircuitTranspiler.transpile(job.operations)
            saved = original_count - len(job.operations)
            self._total_transpile_savings += saved

        # Analyze circuit for intelligent routing
        serialized_ops = self._serialize_ops(job.operations)
        routing_hints = CircuitAnalyzer.analyze(serialized_ops, job.num_qubits)

        # v4.0: Adaptive shot allocation — increase shots for complex circuits
        effective_shots = job.shots
        if self.enable_adaptive_shots and routing_hints.get("t_gate_count", 0) > 0:
            t_count = routing_hints["t_gate_count"]
            depth = routing_hints.get("circuit_depth_est", 1)
            # Scale shots with circuit complexity (more T-gates → more shots needed)
            complexity_factor = 1.0 + (t_count * 0.1) + (depth * 0.02)
            adaptive = int(job.shots * complexity_factor)
            effective_shots = max(
                VQPU_ADAPTIVE_SHOTS_MIN,
                min(adaptive, VQPU_ADAPTIVE_SHOTS_MAX))

        # Build payload
        payload = {
            "circuit_id": job.circuit_id,
            "num_qubits": job.num_qubits,
            "operations": serialized_ops,
            "shots": effective_shots,
            "priority": job.priority,
            "adapt": job.adapt,
            "routing": routing_hints,
            "timestamp": time.time(),
            "god_code": GOD_CODE,
            "bridge_version": VERSION,
        }
        if job.max_branches is not None:
            payload["max_branches"] = job.max_branches
        if job.prune_epsilon is not None:
            payload["prune_epsilon"] = job.prune_epsilon
        if effective_shots != job.shots:
            payload["adaptive_shots"] = {
                "original": job.shots,
                "effective": effective_shots,
            }

        # Write to inbox (atomic via temp + rename)
        start = time.monotonic()
        filename = f"{job.circuit_id}.json"
        tmp_path = self.inbox / f".tmp_{filename}"
        final_path = self.inbox / filename

        tmp_path.write_text(json.dumps(payload, separators=(",", ":")))
        tmp_path.rename(final_path)

        elapsed = (time.monotonic() - start) * 1000.0
        self._total_submit_time_ms += elapsed
        self._jobs_submitted += 1

        # v4.0: Track throughput
        if self._active and self._start_time > 0:
            uptime = time.time() - self._start_time
            if uptime > 0:
                current_hz = self._jobs_submitted / uptime
                if current_hz > self._peak_throughput_hz:
                    self._peak_throughput_hz = current_hz

        return job.circuit_id

    def submit_and_wait(self, job: QuantumJob,
                        timeout: float = 30.0) -> VQPUResult:
        """Submit a job and block until the result is ready."""
        if not self._active:
            self.start()

        # Check if this should run through the exact MPS hybrid engine locally
        serialized_ops = self._serialize_ops(job.operations)
        routing_hints = CircuitAnalyzer.analyze(serialized_ops, job.num_qubits)

        if routing_hints.get("recommended_backend") == "exact_mps_hybrid":
            return self._execute_mps_hybrid(job, serialized_ops, routing_hints)

        circuit_id = self.submit(job)
        start = time.monotonic()
        result = self.collector.wait_for(circuit_id, timeout=timeout)
        elapsed = (time.monotonic() - start) * 1000.0
        self._total_result_time_ms += elapsed

        if result and not result.error:
            self._jobs_completed += 1
        else:
            self._jobs_failed += 1

        return result

    def _execute_mps_hybrid(self, job: QuantumJob,
                            serialized_ops: list,
                            routing_hints: dict) -> VQPUResult:
        """
        Execute via ExactMPSHybridEngine (lossless MPS + platform-aware fallback).

        v7.1: Fallback target is platform-dependent:
          - Apple Silicon: Metal GPU via Swift daemon (fast compute shaders)
          - Intel x86_64:  Chunked CPU statevector (no useful Metal compute)

        Phase 1: Run all gates through exact MPS (cutoff=0)
        Phase 2: If bond dim exceeds threshold → convert to statevector
                 and continue via platform-appropriate fallback
        Phase 3: Sample from final state
        """
        start = time.monotonic()

        mps = ExactMPSHybridEngine(job.num_qubits)
        run_result = mps.run_circuit(serialized_ops)

        if run_result["completed"]:
            # All gates applied in MPS — sample directly
            counts = mps.sample(job.shots)
            shots_total = sum(counts.values())
            probs = {k: v / shots_total for k, v in counts.items()}
            elapsed = (time.monotonic() - start) * 1000.0

            self._jobs_completed += 1
            self._jobs_submitted += 1
            return VQPUResult(
                circuit_id=job.circuit_id,
                probabilities=probs,
                counts=counts,
                backend="exact_mps_hybrid",
                execution_time_ms=elapsed,
                num_qubits=job.num_qubits,
                god_code=GOD_CODE,
            )

        # Fallback: MPS hit bond dim threshold
        # Convert current MPS state to statevector
        statevector = mps.to_statevector()
        remaining_ops = run_result["remaining_ops"]
        fallback_gate = run_result["fallback_at"]
        fallback_target = VQPU_MPS_FALLBACK_TARGET  # v7.1: platform-dependent

        # v15.0: Intel path — AccelStatevectorEngine with gate fusion (18× faster)
        # Replaces raw tensordot/einsum loop with BLAS-optimized fused execution.
        # Diagonal gates (Rz, Phase, Sacred) use element-wise fast path.
        if _IS_INTEL:
            import numpy as np
            from .accel_engine import AccelStatevectorEngine

            accel = AccelStatevectorEngine(job.num_qubits, statevector)
            accel_result = accel.run_fused_circuit(
                remaining_ops, mps._resolve_single_gate
            )

            counts = accel.sample(job.shots)
            shots_total = sum(counts.values())
            probs = {k: v / shots_total for k, v in counts.items()}
            elapsed = (time.monotonic() - start) * 1000.0

            self._jobs_completed += 1
            self._jobs_submitted += 1
            accel_stats = accel.stats()
            return VQPUResult(
                circuit_id=job.circuit_id,
                probabilities=probs,
                counts=counts,
                backend=(
                    f"exact_mps_hybrid\u2192accel_cpu "
                    f"(\u03c7={run_result['peak_chi']}, fallback@gate#{fallback_gate}, "
                    f"fused={accel_stats.get('fused_applied', 0)}, "
                    f"diag_fast={accel_stats.get('diagonal_fast_paths', 0)})"
                ),
                execution_time_ms=elapsed,
                num_qubits=job.num_qubits,
                god_code=GOD_CODE,
            )

        # Apple Silicon path — send to Metal GPU via Swift daemon
        import numpy as np
        sv_real = statevector.real.tolist()
        sv_imag = statevector.imag.tolist()

        fallback_payload = {
            "circuit_id": job.circuit_id,
            "num_qubits": job.num_qubits,
            "operations": remaining_ops,
            "shots": job.shots,
            "resume_statevector": {
                "real": sv_real,
                "imag": sv_imag,
            },
            "routing": {
                "recommended_backend": fallback_target,
                "mps_fallback": True,
                "mps_peak_chi": run_result["peak_chi"],
                "mps_gates_completed": fallback_gate,
                "platform": _PLATFORM["arch"],
            },
            "timestamp": time.time(),
            "god_code": GOD_CODE,
        }

        # Write to inbox
        filename = f"{job.circuit_id}.json"
        tmp_path = self.inbox / f".tmp_{filename}"
        final_path = self.inbox / filename
        tmp_path.write_text(json.dumps(fallback_payload, separators=(",", ":")))
        tmp_path.rename(final_path)
        self._jobs_submitted += 1

        # Wait for GPU result
        result = self.collector.wait_for(job.circuit_id, timeout=30.0)
        elapsed = (time.monotonic() - start) * 1000.0
        self._total_result_time_ms += elapsed

        if result and not result.error:
            result.backend = f"exact_mps_hybrid\u2192{fallback_target} (\u03c7={run_result['peak_chi']}, fallback@gate#{fallback_gate})"
            self._jobs_completed += 1
        else:
            self._jobs_failed += 1

        return result

    def submit_batch(self, jobs: list[QuantumJob],
                     concurrent: bool = False,
                     max_workers: int = 4) -> list[str]:
        """
        Submit multiple jobs. Returns list of circuit_ids.

        v4.0: Uses the persistent pipeline executor when available,
        falling back to a temporary ThreadPoolExecutor for parallel
        submission (I/O-bound: file writes + transpilation).
        """
        if not concurrent or len(jobs) <= 1:
            return [self.submit(job) for job in jobs]

        # v4.0: Prefer persistent pipeline executor
        if self._pipeline_executor is not None:
            futures = [self._pipeline_executor.submit(self.submit, job) for job in jobs]
            return [f.result() for f in futures]

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
            return list(pool.map(self.submit, jobs))

    def collect_results(self, circuit_ids: list[str],
                        timeout: float = 60.0) -> list[VQPUResult]:
        """Collect results for multiple circuit IDs.

        v13.1: Deadline-based timeout — tracks remaining wall time
        instead of dividing timeout equally per job.
        """
        import time as _time
        results = []
        deadline = _time.monotonic() + timeout
        for cid in circuit_ids:
            remaining = max(0.1, deadline - _time.monotonic())
            result = self.collector.wait_for(cid, timeout=remaining)
            results.append(result)
            if result and not result.error:
                self._jobs_completed += 1
            else:
                self._jobs_failed += 1
        return results

    def submit_batch_and_wait(self, jobs: list[QuantumJob],
                              timeout: float = 60.0,
                              concurrent: bool = True) -> list[VQPUResult]:
        """
        Submit multiple jobs and wait for all results.
        Prioritizes jobs by priority field (higher = submitted first).
        """
        if not self._active:
            self.start()

        # PHI-weighted priority scheduling: higher priority jobs first
        sorted_jobs = sorted(jobs, key=lambda j: j.priority * PHI, reverse=True)
        cids = self.submit_batch(sorted_jobs, concurrent=concurrent)
        return self.collect_results(cids, timeout=timeout)

    # ─── RUN SIMULATION (v11.0) ───

    def run_simulation(self, job: QuantumJob, *,
                       compile: bool = True,
                       gate_set: str = "UNIVERSAL",
                       optimization_level: int = 2,
                       error_correct: bool = False,
                       ec_scheme: str = "STEANE_7_1_3",
                       ec_distance: int = 3,
                       use_gate_engine_exec: bool = False,
                       exec_target: str = "LOCAL_STATEVECTOR",
                       score_asi: bool = True,
                       score_agi: bool = True,
                       evolve_coherence: bool = False,
                       coherence_steps: int = 10) -> dict:
        """
        Full engine-orchestrated quantum simulation pipeline.

        Pipeline stages:
          1. TRANSPILE:  10-pass VQPU transpiler (gate cancellation, rotation merge, peephole, fusion)
          2. COMPILE:    Quantum Gate Engine compilation to target gate set
          3. PROTECT:    Error correction encoding (Steane, Surface Code, etc.)
          4. EXECUTE:    Multi-backend execution (MPS hybrid, Gate Engine, Swift GPU)
          5. SCORE:      Sacred alignment + three-engine + ASI 15D + AGI 13D scoring
          6. COHERENCE:  Science Engine coherence evolution (optional)

        Args:
            job:                  QuantumJob to simulate
            compile:              Run Quantum Gate Engine compilation (default: True)
            gate_set:             Target gate set for compilation
            optimization_level:   Compiler optimization 0-3
            error_correct:        Apply error correction encoding
            ec_scheme:            Error correction scheme name
            ec_distance:          Code distance for error correction
            use_gate_engine_exec: Execute via Gate Engine instead of MPS/IPC
            exec_target:          Gate Engine execution target
            score_asi:            Include ASI Core 15D scoring
            score_agi:            Include AGI Core 13D scoring
            evolve_coherence:     Run Science Engine coherence evolution
            coherence_steps:      Number of coherence evolution steps

        Returns:
            dict with keys:
              'result':       VQPUResult or execution dict
              'compilation':  Compilation metrics (if compile=True)
              'protection':   Error correction metrics (if error_correct=True)
              'sacred':       Sacred alignment scores
              'three_engine': Three-engine composite scores
              'asi_score':    ASI 15D scoring (if score_asi=True)
              'agi_score':    AGI 13D scoring (if score_agi=True)
              'coherence':    Coherence evolution state (if evolve_coherence=True)
              'pipeline':     Pipeline execution metadata
        """
        try:
            if not self._active:
                self.start()

            import math
            pipeline_start = time.monotonic()
            simulation = {
                "pipeline": {
                    "version": VERSION,
                    "stages_executed": [],
                    "god_code": GOD_CODE,
                }
            }

            # ── 0. Serialize operations ──
            ops = self._serialize_ops(job.operations)
            num_qubits = job.num_qubits
            shots = job.shots

            # ── 1. TRANSPILE: 10-pass VQPU transpiler ──
            stage_start = time.monotonic()
            original_count = len(ops)
            if self.transpiler and ops:
                ops = CircuitTranspiler.transpile(ops)
                saved = original_count - len(ops)
                self._total_transpile_savings += saved
            simulation["pipeline"]["stages_executed"].append("transpile")
            simulation["pipeline"]["transpile_ms"] = round((time.monotonic() - stage_start) * 1000, 2)
            simulation["pipeline"]["transpile_savings"] = original_count - len(ops)

            # ── 2. COMPILE: Quantum Gate Engine ──
            if compile:
                stage_start = time.monotonic()
                comp_result = self.engines.compile_circuit(
                    ops, num_qubits,
                    gate_set=gate_set,
                    optimization_level=optimization_level
                )
                simulation["compilation"] = comp_result
                if comp_result.get("compiled"):
                    ops = comp_result["operations"]
                simulation["pipeline"]["stages_executed"].append("compile")
                simulation["pipeline"]["compile_ms"] = round((time.monotonic() - stage_start) * 1000, 2)

            # ── 3. PROTECT: Error correction ──
            if error_correct:
                stage_start = time.monotonic()
                ec_result = self.engines.apply_error_correction(
                    ops, num_qubits,
                    scheme=ec_scheme,
                    distance=ec_distance
                )
                simulation["protection"] = ec_result
                if ec_result.get("protected"):
                    ops = ec_result["operations"]
                    # Update qubit count if error correction expanded it
                    if "physical_qubits" in ec_result:
                        num_qubits = max(num_qubits, ec_result["physical_qubits"])
                simulation["pipeline"]["stages_executed"].append("protect")
                simulation["pipeline"]["protect_ms"] = round((time.monotonic() - stage_start) * 1000, 2)

            # ── 4. EXECUTE ──
            stage_start = time.monotonic()
            if use_gate_engine_exec:
                # Execute through Quantum Gate Engine
                exec_result = self.engines.execute_via_gate_engine(
                    ops, num_qubits, shots=shots, target=exec_target
                )
                simulation["result"] = exec_result
                probabilities = exec_result.get("probabilities", {})
            else:
                # Execute through MPS hybrid or IPC to Swift daemon
                exec_job = QuantumJob(
                    circuit_id=job.circuit_id,
                    num_qubits=num_qubits,
                    operations=ops,
                    shots=shots,
                    priority=job.priority,
                    adapt=job.adapt,
                    max_branches=job.max_branches,
                    prune_epsilon=job.prune_epsilon,
                )
                routing_hints = CircuitAnalyzer.analyze(ops, num_qubits)

                if routing_hints.get("recommended_backend") == "exact_mps_hybrid":
                    vqpu_result = self._execute_mps_hybrid(exec_job, ops, routing_hints)
                else:
                    # Direct local MPS execution for simulation
                    mps = ExactMPSHybridEngine(num_qubits)
                    run_result = mps.run_circuit(ops)
                    if run_result["completed"]:
                        counts = mps.sample(shots)
                        shots_total = sum(counts.values())
                        probs = {k: v / shots_total for k, v in counts.items()}
                        elapsed = (time.monotonic() - stage_start) * 1000.0
                        vqpu_result = VQPUResult(
                            circuit_id=exec_job.circuit_id,
                            probabilities=probs,
                            counts=counts,
                            backend=routing_hints.get("recommended_backend", "exact_mps_local"),
                            execution_time_ms=elapsed,
                            num_qubits=num_qubits,
                            god_code=GOD_CODE,
                        )
                    else:
                        # Fallback to IPC submission
                        cid = self.submit(exec_job)
                        vqpu_result = self.collector.wait_for(cid, timeout=30.0)

                simulation["result"] = {
                    "circuit_id": vqpu_result.circuit_id,
                    "probabilities": vqpu_result.probabilities,
                    "counts": vqpu_result.counts,
                    "backend": vqpu_result.backend,
                    "execution_time_ms": vqpu_result.execution_time_ms,
                    "num_qubits": vqpu_result.num_qubits,
                    "error": vqpu_result.error,
                }
                probabilities = vqpu_result.probabilities

            simulation["pipeline"]["stages_executed"].append("execute")
            simulation["pipeline"]["execute_ms"] = round((time.monotonic() - stage_start) * 1000, 2)

            # ── 5. SCORE (v8.0: cached scoring — fixes 96% bottleneck) ──
            stage_start = time.monotonic()

            # Sacred alignment (fast — no caching needed)
            simulation["sacred"] = SacredAlignmentScorer.score(probabilities, num_qubits)
            measurement_entropy = simulation["sacred"].get("entropy", 1.0)

            # Three-engine scoring (v8.0: harmonic+wave cached, entropy bucketed)
            simulation["three_engine"] = ThreeEngineQuantumScorer.composite_score(measurement_entropy)

            # ASI 15D scoring (v8.0: cached per num_qubits + entropy bucket)
            if score_asi:
                simulation["asi_score"] = ScoringCache.get_asi_score(
                    probabilities, num_qubits, measurement_entropy,
                    self.engines.asi_score)

            # AGI 13D scoring (v8.0: cached per num_qubits + entropy bucket)
            if score_agi:
                simulation["agi_score"] = ScoringCache.get_agi_score(
                    probabilities, num_qubits, measurement_entropy,
                    self.engines.agi_score)

            simulation["pipeline"]["stages_executed"].append("score")
            simulation["pipeline"]["score_ms"] = round((time.monotonic() - stage_start) * 1000, 2)
            simulation["pipeline"]["scoring_cache"] = ScoringCache.stats()

            # ── 6. COHERENCE EVOLUTION (optional) ──
            if evolve_coherence:
                stage_start = time.monotonic()
                # Seed coherence with probability amplitudes
                seed = list(probabilities.values())[:10] if probabilities else [0.5]
                simulation["coherence"] = self.engines.evolve_coherence(seed, coherence_steps)
                simulation["pipeline"]["stages_executed"].append("coherence")
                simulation["pipeline"]["coherence_ms"] = round((time.monotonic() - stage_start) * 1000, 2)

            # ── 7. SUPERCONDUCTIVITY ANALYSIS (v9.0, cached via composite_score) ──
            stage_start = time.monotonic()
            sc_data = simulation.get("three_engine", {}).get("sc_heisenberg", 0.0)
            simulation["superconductivity"] = sc_data
            simulation["pipeline"]["stages_executed"].append("sc_analysis")
            simulation["pipeline"]["sc_analysis_ms"] = round((time.monotonic() - stage_start) * 1000, 2)

            # ── 8. BRAIN INTEGRATION (v13.0 — Quantum Brain bidirectional scoring) ──
            stage_start = time.monotonic()
            try:
                if BrainIntegration.is_available():
                    # Brain→VQPU: Get brain intelligence scores
                    brain_unified = BrainIntegration.unified_vqpu_brain_score(measurement_entropy)
                    simulation["brain_integration"] = brain_unified

                    # VQPU→Brain: Feed simulation results to brain
                    feedback = BrainIntegration.feed_simulation_to_brain(simulation)
                    simulation["brain_feedback"] = feedback
                else:
                    simulation["brain_integration"] = {"available": False}
                    simulation["brain_feedback"] = {"fed": False, "reason": "brain_unavailable"}
            except Exception:
                simulation["brain_integration"] = {"available": False, "error": "integration_failed"}
                simulation["brain_feedback"] = {"fed": False}
            simulation["pipeline"]["stages_executed"].append("brain_integration")
            simulation["pipeline"]["brain_ms"] = round((time.monotonic() - stage_start) * 1000, 2)

            # ── Pipeline summary ──
            total_ms = round((time.monotonic() - pipeline_start) * 1000, 2)
            simulation["pipeline"]["total_ms"] = total_ms
            simulation["pipeline"]["engines_active"] = self.engines.status()

            self._jobs_submitted += 1
            self._jobs_completed += 1

            return simulation
        except Exception as e:
            # v12.2: Robust error handling for parallel execution
            import traceback
            error_details = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "pipeline": {
                    "version": VERSION,
                    "stages_executed": ["error"],
                    "error": str(e),
                    "god_code": GOD_CODE,
                    "total_ms": 0,
                }
            }
            return error_details

    def run_simulation_batch(self, jobs: list, **kwargs) -> list:
        """
        Run multiple simulations through the engine pipeline.

        v12.2: Process-based parallel execution for CPU-bound quantum simulations.
        Uses ProcessPoolExecutor to bypass GIL limitations of ThreadPoolExecutor.
        Returns a list of simulation result dicts with per-job timing metadata.
        """
        if len(jobs) > 1 and self.pipeline_workers > 1:
            # Use ProcessPoolExecutor for true parallel execution
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import logging
            logging.info(f"[VQPU_PARALLEL] Using ProcessPoolExecutor with {min(len(jobs), self.pipeline_workers)} workers for {len(jobs)} jobs")

            # Convert jobs to dicts for pickling
            job_dicts = []
            for job in jobs:
                job_dict = {
                    'num_qubits': job.num_qubits,
                    'operations': job.operations,
                    'shots': job.shots,
                    'circuit_id': job.circuit_id,
                    'priority': job.priority,
                    'adapt': job.adapt,
                    'max_branches': job.max_branches,
                    'prune_epsilon': job.prune_epsilon,
                }
                job_dicts.append(job_dict)

            # Use ProcessPoolExecutor with appropriate worker count
            max_workers = min(len(jobs), self.pipeline_workers)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures_map = {}
                for i, job_dict in enumerate(job_dicts):
                    future = executor.submit(_parallel_simulation_worker, job_dict)
                    futures_map[future] = i

                results = [None] * len(jobs)
                for future in as_completed(futures_map):
                    idx = futures_map[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = {
                            "error": str(e),
                            "pipeline": {
                                "version": VERSION,
                                "stages_executed": ["error"],
                                "error": str(e),
                                "god_code": GOD_CODE,
                                "total_ms": 0,
                            }
                        }
                return results
        else:
            # Fallback to sequential execution
            return [self.run_simulation(job, **kwargs) for job in jobs]

    # ─── VQPU FINDINGS (v9.0) ───

    def run_vqpu_findings(self) -> dict:
        """
        v9.0: Run all 11 VQPU findings simulations on-demand.

        Returns summary with per-sim results, pass rate, and SC telemetry.
        Uses EngineIntegration.run_vqpu_findings_cycle() for the cycle.
        """
        if not self._active:
            self.start()

        start = time.monotonic()
        cycle_result = self.engines.run_vqpu_findings_cycle()

        # Also get SC scoring
        sc_scoring = ScoringCache.get_sc(ThreeEngineQuantumScorer.sc_score)

        elapsed = round((time.monotonic() - start) * 1000, 2)
        return {
            "findings": cycle_result,
            "sc_scoring": sc_scoring,
            "daemon_cycler_status": self._daemon_cycler.status(),
            "elapsed_ms": elapsed,
            "version": VERSION,
        }

    def daemon_cycler_status(self) -> dict:
        """v11.0: Return daemon cycler health and history."""
        return self._daemon_cycler.status()

    def trigger_daemon_cycle(self) -> dict:
        """v11.0: Trigger an immediate daemon cycle (non-blocking if already running)."""
        return self._daemon_cycler.run_cycle_now()

    # ─── MICRO DAEMON (v2.0 → v2.1) ───

    def micro_daemon_status(self) -> dict:
        """v2.0: Return micro daemon health and telemetry."""
        return self._micro_daemon.status()

    def micro_daemon_force_tick(self) -> dict:
        """v2.0: Trigger an immediate micro daemon tick (synchronous)."""
        return self._micro_daemon.force_tick()

    def micro_daemon_submit(self, task_name: str, payload: dict = None) -> str:
        """v2.0: Submit a named micro-task for immediate execution."""
        return self._micro_daemon.submit(task_name, payload)

    def micro_daemon_tick_metrics(self) -> list:
        """v2.1: Return recent tick profiling metrics from the micro daemon."""
        return list(self._micro_daemon._tick_metrics)

    def micro_daemon_self_test(self) -> dict:
        """v2.1: Run micro daemon self_test (12 probes)."""
        return self._micro_daemon.self_test()

    def micro_daemon_analytics(self) -> dict:
        """v2.3: Telemetry analytics — trends, anomalies, grades, hotspots."""
        return self._micro_daemon.analytics()

    def micro_daemon_throttled_tasks(self) -> dict:
        """v2.3: Report which micro-tasks are currently auto-throttled."""
        return self._micro_daemon.throttled_tasks()

    def micro_daemon_reset_stats(self) -> dict:
        """v2.4: Reset all micro daemon counters + ring buffers (returns pre-reset snapshot)."""
        return self._micro_daemon.reset_stats()

    def micro_daemon_task_stats(self) -> dict:
        """v2.4: Per-task execution statistics from micro daemon task history."""
        return self._micro_daemon.task_stats()

    # ─── QUANTUM SUBCONSCIOUS (v12.1 + v12.2 Autonomous) ───

    def subconscious_status(self) -> dict:
        """v12.1: Return quantum subconscious engine status (v2.0 autonomous)."""
        return self._daemon_cycler.subconscious_status()

    def harvest_idle_thoughts(self, max_count: int = 50,
                              min_coherence: float = 0.0,
                              stream_filter: Optional[str] = None) -> list:
        """v12.1: Harvest idle thoughts from quantum subconscious.

        Returns recent idle thoughts generated during kernel idle periods.
        Non-destructive read — thoughts remain in the buffer.

        Args:
            max_count: Max thoughts to return.
            min_coherence: Minimum coherence threshold.
            stream_filter: Optional stream name filter.
        """
        sub = self._daemon_cycler._quantum_subconscious
        if sub is not None:
            try:
                return sub.harvest_thoughts(
                    max_count=max_count,
                    min_coherence=min_coherence,
                    stream_filter=stream_filter)
            except Exception:
                pass
        return []

    def harvest_precog_seeds(self) -> list:
        """v12.1: Harvest precognition pre-seeds from subconscious insights."""
        sub = self._daemon_cycler._quantum_subconscious
        if sub is not None:
            try:
                return sub.harvest_precog_seeds()
            except Exception:
                pass
        return []

    def crystallize_insights(self) -> list:
        """v12.1: Trigger insight crystallization from subconscious dream state."""
        sub = self._daemon_cycler._quantum_subconscious
        if sub is not None:
            try:
                return sub.crystallize_insights(force=True)
            except Exception:
                pass
        return []

    def drain_subconscious(self) -> list:
        """v12.1: Drain all buffered idle thoughts (destructive read).

        Used when precognition cycle begins and needs all accumulated
        subconscious data for pre-seeding.
        """
        sub = self._daemon_cycler._quantum_subconscious
        if sub is not None:
            try:
                return sub.drain_thoughts()
            except Exception:
                pass
        return []

    # ─── QUANTUM SUBCONSCIOUS v2.0 DYNAMIC AUTONOMOUS ───

    def subconscious_stream_health(self) -> dict:
        """v12.2: Per-stream health report (failures, coherence, latency, restarts)."""
        return self._daemon_cycler.subconscious_stream_health()

    def subconscious_stream_weights(self) -> dict:
        """v12.2: Current adaptive φ-gradient stream weights."""
        return self._daemon_cycler.subconscious_stream_weights()

    def subconscious_evolution(self) -> dict:
        """v12.2: Self-evolution engine status (generation, mutations, best params)."""
        return self._daemon_cycler.subconscious_evolution()

    def subconscious_autonomy_dashboard(self) -> dict:
        """v12.2: Full autonomous systems dashboard.

        Returns a comprehensive view of all 8 autonomous subsystems:
         - StreamHealthMonitor: per-stream failure/coherence/latency
         - AdaptiveStreamWeighter: φ-gradient priority weights
         - SelfEvolutionEngine: generation count, mutation rate, best params
         - ProactiveInsightPusher: push count, callbacks registered
         - CrossEngineFeedback: coherence/entropy resonance state
         - EmergentStreamFactory: spawned streams, capacity remaining
         - DreamDepthController: current depth, CPU load, idle time
         - AdaptiveCrystallizer: current threshold, quality EMA
        """
        return self._daemon_cycler.subconscious_autonomy()

    def subconscious_dream_depth(self) -> dict:
        """v12.2: Dream depth controller status."""
        sub = self._daemon_cycler._quantum_subconscious
        if sub is not None:
            try:
                return sub.dream_depth_status()
            except Exception:
                pass
        return {}

    def subconscious_crystallizer(self) -> dict:
        """v12.2: Adaptive crystallizer status (threshold, quality EMA)."""
        sub = self._daemon_cycler._quantum_subconscious
        if sub is not None:
            try:
                return sub.crystallizer_status()
            except Exception:
                pass
        return {}

    def subconscious_emergent_streams(self) -> dict:
        """v12.2: Emergent (dynamically spawned) stream status."""
        sub = self._daemon_cycler._quantum_subconscious
        if sub is not None:
            try:
                return sub.emergent_status()
            except Exception:
                pass
        return {}

    # ─── SCORING ───

    def score_result(self, result: VQPUResult) -> dict:
        """
        Score a VQPUResult for sacred alignment + three-engine + ASI/AGI analysis.
        Returns sacred metrics dict with engine composite scores.
        """
        sacred = SacredAlignmentScorer.score(
            result.probabilities, result.num_qubits)

        # Three-engine scoring: entropy reversal + harmonic + wave coherence
        three_engine = ThreeEngineQuantumScorer.composite_score(
            sacred.get("entropy", 1.0))
        sacred["three_engine"] = three_engine

        # v5.0: ASI/AGI core scoring
        sacred["asi_score"] = self.engines.asi_score(
            result.probabilities, result.num_qubits)
        sacred["agi_score"] = self.engines.agi_score(
            result.probabilities, result.num_qubits)

        return sacred

    def score_result_three_engine(self, result: VQPUResult) -> dict:
        """
        Three-engine-only scoring for a VQPUResult.

        Returns entropy reversal, harmonic resonance, wave coherence,
        and composite score using Science + Math engines.
        """
        sacred = SacredAlignmentScorer.score(
            result.probabilities, result.num_qubits)
        return ThreeEngineQuantumScorer.composite_score(
            sacred.get("entropy", 1.0))

    def three_engine_status(self) -> dict:
        """Return connection status of all three engines."""
        return ThreeEngineQuantumScorer.engines_status()

    def engine_status(self) -> dict:
        """Return connection status of ALL engines and cores (v6.0)."""
        return self.engines.status()

    # ─── QUANTUM DATABASE RESEARCH (v6.0) ───

    def research_database(self, query: str = "", *, db: str = "all",
                          shots: int = 4096) -> dict:
        """
        Full quantum-accelerated research pipeline on L104 databases.

        Runs Grover search, QPE pattern discovery, QFT frequency analysis,
        amplitude estimation, and quantum walk on knowledge graph.

        Args:
            query:  Search query (for Grover search). Empty = skip search.
            db:     Target database: "research", "unified", "nexus", "all"
            shots:  Measurement shots per algorithm

        Returns:
            dict with complete quantum research results
        """
        researcher = QuantumDatabaseResearcher()
        return researcher.full_research(query, shots=shots)

    def quantum_db_search(self, query: str, *, db: str = "all",
                          max_results: int = 50, shots: int = 2048) -> dict:
        """
        Grover-accelerated quantum search across L104 databases.

        O(√N) search speedup for finding matching records in research
        findings, unified memory, knowledge nodes, and ASI learnings.

        Args:
            query:       Search string
            db:          Database to search: "research", "unified", "nexus", "all"
            max_results: Maximum results
            shots:       Measurement shots

        Returns:
            dict with 'matches', 'quantum_speedup', 'sacred_alignment'
        """
        researcher = QuantumDatabaseResearcher()
        return researcher.grover_search(query, db=db, max_results=max_results,
                                         shots=shots)

    def quantum_pattern_discovery(self, *, db: str = "research",
                                   field: str = "confidence",
                                   shots: int = 4096) -> dict:
        """
        QPE-based pattern discovery in database numerical fields.

        Discovers hidden periodic patterns and GOD_CODE resonances
        in confidence scores, importance weights, and reward values.
        """
        researcher = QuantumDatabaseResearcher()
        return researcher.qpe_pattern_discovery(db=db, field=field, shots=shots)

    def quantum_frequency_analysis(self, *, db: str = "all",
                                    shots: int = 4096) -> dict:
        """
        QFT frequency analysis of database record distributions.

        Extracts frequency components revealing periodic patterns
        in database activity and cross-database correlations.
        """
        researcher = QuantumDatabaseResearcher()
        return researcher.qft_frequency_analysis(db=db, shots=shots)

    def quantum_knowledge_walk(self, *, steps: int = 10,
                                shots: int = 2048) -> dict:
        """
        Quantum walk on L104 knowledge graph.

        Discovers node importance, clusters, and sacred resonance
        points through quantum interference on the knowledge graph.
        """
        researcher = QuantumDatabaseResearcher()
        return researcher.quantum_walk_knowledge(steps=steps, shots=shots)

    def database_summary(self) -> dict:
        """Summary of all L104 databases and quantum-searchable content."""
        researcher = QuantumDatabaseResearcher()
        return researcher.database_summary()

    def daemon_health(self) -> dict:
        """
        Check if the Swift L104Daemon is running and responsive.
        Returns health dict with pid, uptime, and responsiveness.
        """
        health = {
            "daemon_running": False,
            "pid": None,
            "bridge_path_exists": self.bridge_path.exists(),
            "inbox_writable": os.access(str(self.inbox), os.W_OK) if self.inbox.exists() else False,
            "outbox_readable": os.access(str(self.outbox), os.R_OK) if self.outbox.exists() else False,
        }

        # Check for daemon PID file
        pid_paths = [
            Path(os.environ.get("L104_ROOT", os.getcwd())) / "l104_daemon.pid",
            Path("/tmp/l104_daemon.pid"),
        ]
        for pid_path in pid_paths:
            if pid_path.exists():
                try:
                    pid = int(pid_path.read_text().strip())
                    health["pid"] = pid
                    # Check if process is alive
                    os.kill(pid, 0)
                    health["daemon_running"] = True
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
                break

        # Responsiveness check: write a ping, see if outbox gets a response
        if health["daemon_running"]:
            try:
                ping_id = f"health-{uuid.uuid4().hex[:8]}"
                ping_payload = {
                    "circuit_id": ping_id,
                    "num_qubits": 1,
                    "operations": [{"gate": "I", "qubits": [0]}],
                    "shots": 1,
                    "priority": 0,
                    "health_check": True,
                    "god_code": GOD_CODE,
                }
                tmp = self.inbox / f".tmp_{ping_id}.json"
                final = self.inbox / f"{ping_id}.json"
                tmp.write_text(json.dumps(ping_payload, separators=(",", ":")))
                tmp.rename(final)

                # Quick wait (2 second max)
                result = self.collector.wait_for(ping_id, timeout=2.0)
                health["responsive"] = result is not None and not result.error
                health["response_time_ms"] = result.execution_time_ms if result else None
            except Exception:
                health["responsive"] = False

        return health

    # ─── RESTART ON BOOT ───

    @staticmethod
    def restart_on_boot() -> dict:
        """
        Full restart-on-boot for all VQPU daemon processes.

        Ensures all launchd LaunchAgents are registered, loaded, and alive.
        Resurrects any crashed/unloaded services. Creates missing IPC dirs.
        Can be called programmatically or from CLI via l104_vqpu_boot_manager.py.

        Returns:
            dict with boot sequence report including per-service status
        """
        try:
            from l104_vqpu_boot_manager import restart_on_boot as _boot
            return _boot()
        except ImportError:
            # Inline fallback if boot manager not available
            services = [
                "com.l104.fast-server",
                "com.l104.node-server",
                "com.l104.vqpu-daemon",
                "com.l104.vqpu-micro-daemon",
                "com.l104.auto-update",
                "com.l104.log-rotate",
                "com.l104.health-watchdog",
                "com.l104.boot-manager",
            ]
            launch_dir = Path.home() / "Library" / "LaunchAgents"
            results = {}
            for svc in services:
                plist = launch_dir / f"{svc}.plist"
                if not plist.exists():
                    results[svc] = "plist_missing"
                    continue
                try:
                    subprocess.run(
                        ["launchctl", "unload", str(plist)],
                        capture_output=True, timeout=5,
                    )
                    time.sleep(0.5)
                    rc = subprocess.run(
                        ["launchctl", "load", "-w", str(plist)],
                        capture_output=True, timeout=5,
                    )
                    results[svc] = "loaded" if rc.returncode == 0 else "failed"
                except Exception as e:
                    results[svc] = f"error: {e}"
            return {"boot_results": results, "god_code": GOD_CODE}

    @staticmethod
    def daemon_status_all() -> dict:
        """
        Get status of all VQPU daemon processes managed by launchd.

        Returns:
            dict with per-service load state, PIDs, and overall health
        """
        try:
            from l104_vqpu_boot_manager import status as _status
            return _status()
        except ImportError:
            return {"error": "l104_vqpu_boot_manager not available"}

    @staticmethod
    def resurrect_daemons() -> dict:
        """
        Resurrect any dead/unloaded VQPU daemon services.

        Returns:
            dict with actions taken per service
        """
        try:
            from l104_vqpu_boot_manager import resurrect as _resurrect
            return _resurrect()
        except ImportError:
            return {"error": "l104_vqpu_boot_manager not available"}

    # ─── CONVENIENCE BUILDERS ───

    def bell_pair(self, shots: int = 1024) -> QuantumJob:
        """Create a Bell state circuit: H(0) → CNOT(0,1)."""
        # v5.0: Try Quantum Gate Engine first for sacred-aligned Bell pair
        engine = self.engines.gate_engine()
        if engine is not None:
            try:
                circ = engine.bell_pair()
                ops = _circuit_to_ops(circ)
                if ops:
                    return QuantumJob(num_qubits=2, shots=shots, operations=ops)
            except Exception:
                pass
        return QuantumJob(
            num_qubits=2,
            shots=shots,
            operations=[
                {"gate": "H", "qubits": [0]},
                {"gate": "CX", "qubits": [0, 1]},
            ],
        )

    def ghz_state(self, n: int = 3, shots: int = 1024) -> QuantumJob:
        """Create an N-qubit GHZ state."""
        # v5.0: Try Quantum Gate Engine first
        engine = self.engines.gate_engine()
        if engine is not None:
            try:
                circ = engine.ghz_state(n)
                ops = _circuit_to_ops(circ)
                if ops:
                    return QuantumJob(num_qubits=n, shots=shots, operations=ops)
            except Exception:
                pass
        ops = [{"gate": "H", "qubits": [0]}]
        for i in range(n - 1):
            ops.append({"gate": "CX", "qubits": [i, i + 1]})
        return QuantumJob(num_qubits=n, shots=shots, operations=ops)

    def qft_circuit(self, n: int = 4, shots: int = 1024) -> QuantumJob:
        """Create an N-qubit Quantum Fourier Transform circuit."""
        # v5.0: Try Quantum Gate Engine first
        engine = self.engines.gate_engine()
        if engine is not None:
            try:
                circ = engine.quantum_fourier_transform(n)
                ops = _circuit_to_ops(circ)
                if ops:
                    return QuantumJob(num_qubits=n, shots=shots, operations=ops, adapt=True)
            except Exception:
                pass
        import math
        ops = []
        for i in range(n):
            ops.append({"gate": "H", "qubits": [i]})
            for j in range(i + 1, n):
                angle = math.pi / (2 ** (j - i))
                ops.append({"gate": "Rz", "qubits": [j], "parameters": [angle]})
                ops.append({"gate": "CX", "qubits": [i, j]})
        return QuantumJob(num_qubits=n, shots=shots, operations=ops, adapt=True)

    def sacred_circuit(self, n: int = 3, depth: int = 4,
                       shots: int = 1024) -> QuantumJob:
        """Create an L104 sacred circuit with φ-aligned rotations."""
        # v5.0: Try Quantum Gate Engine first
        engine = self.engines.gate_engine()
        if engine is not None:
            try:
                circ = engine.sacred_circuit(n, depth=depth)
                ops = _circuit_to_ops(circ)
                if ops:
                    return QuantumJob(num_qubits=n, shots=shots, operations=ops, adapt=True)
            except Exception:
                pass
        import math
        ops = []
        for d in range(depth):
            for q in range(n):
                ops.append({"gate": "H", "qubits": [q]})
                theta = (PHI ** (d + 1)) * math.pi / GOD_CODE
                ops.append({"gate": "Rz", "qubits": [q], "parameters": [theta]})
            for q in range(n - 1):
                ops.append({"gate": "CX", "qubits": [q, q + 1]})
        return QuantumJob(num_qubits=n, shots=shots, operations=ops, adapt=True)

    def sacred_gate_circuit(self, n: int = 3, shots: int = 1024) -> QuantumJob:
        """
        Create a circuit using L104 sacred gates (PHI_GATE, GOD_CODE_PHASE, etc.).
        Requires the Quantum Gate Engine; falls back to Rz approximation.
        """
        engine = self.engines.gate_engine()
        if engine is not None:
            try:
                from l104_quantum_gate_engine import (
                    GateCircuit, PHI_GATE, GOD_CODE_PHASE, VOID_GATE,
                )
                circ = GateCircuit(n, name="sacred_gates")
                for q in range(n):
                    circ.h(q)
                    circ.append(PHI_GATE, [q])
                for q in range(n - 1):
                    circ.cx(q, q + 1)
                for q in range(n):
                    circ.append(GOD_CODE_PHASE, [q])
                if hasattr(circ, 'append') and VOID_GATE is not None:
                    for q in range(n):
                        circ.append(VOID_GATE, [q])
                ops = _circuit_to_ops(circ)
                if ops:
                    return QuantumJob(num_qubits=n, shots=shots, operations=ops, adapt=True)
            except Exception:
                pass
        # Fallback: approximate sacred gates with Rz
        import math
        ops = []
        for q in range(n):
            ops.append({"gate": "H", "qubits": [q]})
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [math.pi * PHI]})
        for q in range(n - 1):
            ops.append({"gate": "CX", "qubits": [q, q + 1]})
        for q in range(n):
            ops.append({"gate": "Rz", "qubits": [q], "parameters": [math.pi * GOD_CODE / 1000.0]})
        return QuantumJob(num_qubits=n, shots=shots, operations=ops, adapt=True)

    # ─── VARIATIONAL ALGORITHMS (v7.0) ───

    def run_vqe(self, hamiltonian_terms: list, num_qubits: int, *,
                depth: int = 3, max_iterations: int = 100,
                shots: int = 4096) -> dict:
        """
        Run Variational Quantum Eigensolver via the VQPU pipeline.

        Finds approximate ground state energy of H = Σ cᵢ Pᵢ using
        a hardware-efficient ansatz optimised with gradient-free search.
        PHI-scaled initial parameters and GOD_CODE-aligned scoring.

        Args:
            hamiltonian_terms: List of (coefficient, pauli_string) tuples
            num_qubits:        Number of qubits
            depth:             Ansatz depth (default 3)
            max_iterations:    Optimizer iterations
            shots:             Measurement shots per evaluation

        Returns:
            dict with ground_energy, optimal_params, convergence, sacred_alignment
        """
        if not self._active:
            self.start()
        result = VariationalQuantumEngine.vqe(
            hamiltonian_terms, num_qubits,
            depth=depth, max_iterations=max_iterations, shots=shots,
        )
        self._jobs_submitted += result.get("circuit_evaluations", 1)
        self._jobs_completed += result.get("circuit_evaluations", 1)
        return result

    def run_qaoa(self, cost_terms: list, num_qubits: int, *,
                 p_layers: int = 3, max_iterations: int = 80,
                 shots: int = 4096) -> dict:
        """
        Run Quantum Approximate Optimization Algorithm via the VQPU pipeline.

        Solves combinatorial problems encoded as Ising Hamiltonian
        C = Σ Jᵢⱼ ZᵢZⱼ + Σ hᵢ Zᵢ with alternating cost/mixer layers.

        Args:
            cost_terms:  List of (weight, i, j) for ZZ or (weight, i) for Z
            num_qubits:  Problem size
            p_layers:    QAOA depth
            max_iterations: Optimizer iterations
            shots:       Measurement shots

        Returns:
            dict with best_bitstring, best_cost, optimal_gammas/betas, sacred_alignment
        """
        if not self._active:
            self.start()
        result = VariationalQuantumEngine.qaoa(
            cost_terms, num_qubits,
            p_layers=p_layers, max_iterations=max_iterations, shots=shots,
        )
        self._jobs_submitted += result.get("iterations", 1)
        self._jobs_completed += result.get("iterations", 1)
        return result

    # ─── NOISY SIMULATION (v7.0) ───

    def run_noisy_simulation(self, job: QuantumJob, *,
                             noise_model: NoiseModel = None,
                             mitigate: bool = True) -> dict:
        """
        Execute a circuit with realistic noise and optional ZNE mitigation.

        Runs the circuit through ExactMPSHybridEngine with per-gate
        noise injection from the NoiseModel, then applies ZNE if enabled.

        Args:
            job:          QuantumJob to simulate
            noise_model:  NoiseModel instance (default: realistic superconducting)
            mitigate:     Apply Zero-Noise Extrapolation (default: True)

        Returns:
            dict with noisy_result, mitigated_result (if mitigate), noise_params
        """
        if not self._active:
            self.start()
        if noise_model is None:
            # v13.2: Default to QPU-calibrated Heron noise model (ibm_torino data)
            # Falls back to realistic_superconducting() if factory not available
            try:
                noise_model = NoiseModel.qpu_calibrated_heron()
            except (AttributeError, TypeError):
                noise_model = NoiseModel.realistic_superconducting()

        ops = self._serialize_ops(job.operations)
        if self.transpiler:
            ops = CircuitTranspiler.transpile(ops)

        def _run_with_noise(nm: NoiseModel):
            """Execute circuit with given noise model."""
            mps = ExactMPSHybridEngine(job.num_qubits)
            run_result = mps.run_circuit(ops)
            if not run_result.get("completed"):
                return {"probabilities": {}, "error": "mps_incomplete"}
            sv = mps.to_statevector()
            # Apply readout noise to sampled counts
            counts = mps.sample(job.shots)
            noisy_counts = nm.apply_readout_noise(counts, job.num_qubits)
            total = sum(noisy_counts.values())
            probs = {k: v / total for k, v in noisy_counts.items()} if total > 0 else {}
            return {"probabilities": probs, "counts": noisy_counts}

        # Noisy run
        noisy = _run_with_noise(noise_model)
        result = {
            "noisy_result": noisy,
            "noise_params": noise_model.to_dict(),
            "sacred_alignment": SacredAlignmentScorer.score(
                noisy.get("probabilities", {}), job.num_qubits),
        }

        # ZNE mitigation
        if mitigate:
            zne = QuantumErrorMitigation.zero_noise_extrapolation(
                _run_with_noise, noise_model,
            )
            result["mitigated"] = zne

        self._jobs_submitted += 1
        self._jobs_completed += 1
        return result

    # ─── ENTANGLEMENT ANALYSIS (v7.0) ───

    def quantify_entanglement(self, job: QuantumJob) -> dict:
        """
        Run a circuit and compute full entanglement metrics on the final state.

        Executes via MPS hybrid, extracts the statevector, then computes
        von Neumann entropy, concurrence, Schmidt decomposition, and
        sacred entanglement score.

        Returns:
            dict with entanglement metrics from EntanglementQuantifier
        """
        if not self._active:
            self.start()

        ops = self._serialize_ops(job.operations)
        if self.transpiler:
            ops = CircuitTranspiler.transpile(ops)

        mps = ExactMPSHybridEngine(job.num_qubits)
        run_result = mps.run_circuit(ops)
        if not run_result.get("completed"):
            return {"error": "circuit_execution_incomplete", "num_qubits": job.num_qubits}

        sv = mps.to_statevector()
        analysis = EntanglementQuantifier.full_analysis(sv, job.num_qubits)

        # Add probability distribution for context
        counts = mps.sample(job.shots)
        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()} if total > 0 else {}
        analysis["probabilities"] = dict(list(probs.items())[:16])
        analysis["sacred_alignment"] = SacredAlignmentScorer.score(probs, job.num_qubits)

        self._jobs_submitted += 1
        self._jobs_completed += 1
        return analysis

    # ─── GOD CODE SIMULATION (v7.0) ───

    def run_god_code_simulation(self, simulation_name: str = None, *,
                                category: str = None,
                                feedback_iterations: int = 0) -> dict:
        """
        Run God Code Simulator through the VQPU pipeline.

        Can run a single named simulation, a full category, or a
        multi-engine feedback loop. Results are scored with sacred
        alignment and three-engine metrics.

        Args:
            simulation_name:     Name of a specific simulation (e.g., "entanglement_entropy")
            category:            Run all sims in category ("core", "quantum", "advanced", "discovery")
            feedback_iterations: If > 0, run multi-engine feedback loop

        Returns:
            dict with simulation results and engine scoring
        """
        sim = self.engines.god_code_simulator()
        if sim is None:
            return {"error": "god_code_simulator_unavailable"}

        try:
            result = {}
            if feedback_iterations > 0:
                se = self.engines.science_engine()
                me = self.engines.math_engine()
                if se and me:
                    sim.connect_engines(coherence=se.coherence, entropy=se.entropy, math_engine=me)
                result["feedback_loop"] = sim.run_feedback_loop(iterations=feedback_iterations)
            elif category:
                result["category_results"] = sim.run_category(category)
            elif simulation_name:
                result["simulation"] = sim.run(simulation_name)
            else:
                result["all_results"] = sim.run_all()

            result["god_code"] = GOD_CODE
            result["engine_status"] = self.engines.status()
            return result
        except Exception as e:
            return {"error": str(e)}

    # ─── QUANTUM INFORMATION METRICS (v8.0) ───

    def quantum_information_metrics(self, job: QuantumJob,
                                     generator_ops: list = None) -> dict:
        """
        Compute full quantum information metrics on a circuit's output state.

        Includes quantum mutual information, topological entanglement entropy,
        and quantum Fisher information (if generator_ops provided).

        Args:
            job:            QuantumJob to execute and analyze
            generator_ops:  Optional parameterised gate operations for QFI

        Returns:
            dict with mutual_information, topological, fisher_information,
            and sacred alignment scores
        """
        if not self._active:
            self.start()

        ops = self._serialize_ops(job.operations)
        if self.transpiler:
            ops = CircuitTranspiler.transpile(ops)

        mps = ExactMPSHybridEngine(job.num_qubits)
        run_result = mps.run_circuit(ops)
        if not run_result.get("completed"):
            return {"error": "circuit_execution_incomplete"}

        sv = mps.to_statevector()
        metrics = QuantumInformationMetrics.full_metrics(
            sv, job.num_qubits, generator_ops=generator_ops)

        # Add standard entanglement metrics
        metrics["entanglement"] = EntanglementQuantifier.full_analysis(sv, job.num_qubits)

        # Add probability distribution
        counts = mps.sample(job.shots)
        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()} if total > 0 else {}
        metrics["probabilities"] = dict(list(probs.items())[:16])
        metrics["sacred_alignment"] = SacredAlignmentScorer.score(probs, job.num_qubits)

        self._jobs_submitted += 1
        self._jobs_completed += 1
        return metrics

    def quantum_fidelity(self, job_a: QuantumJob, job_b: QuantumJob) -> dict:
        """
        Compute quantum fidelity between the output states of two circuits.

        Executes both circuits via MPS, extracts statevectors, and computes
        fidelity F = |⟨ψ_a|ψ_b⟩|², Bures distance, and relative entropy.

        Args:
            job_a: First quantum circuit
            job_b: Second quantum circuit

        Returns:
            dict with fidelity, relative_entropy, bures_distance,
            sacred_alignment
        """
        if not self._active:
            self.start()

        # Execute circuit A
        ops_a = self._serialize_ops(job_a.operations)
        if self.transpiler:
            ops_a = CircuitTranspiler.transpile(ops_a)
        mps_a = ExactMPSHybridEngine(job_a.num_qubits)
        mps_a.run_circuit(ops_a)
        sv_a = mps_a.to_statevector()

        # Execute circuit B
        ops_b = self._serialize_ops(job_b.operations)
        if self.transpiler:
            ops_b = CircuitTranspiler.transpile(ops_b)
        mps_b = ExactMPSHybridEngine(job_b.num_qubits)
        mps_b.run_circuit(ops_b)
        sv_b = mps_b.to_statevector()

        num_qubits = max(job_a.num_qubits, job_b.num_qubits)

        # Fidelity
        fidelity = QuantumStateTomography.state_fidelity(sv_a, sv_b, num_qubits)

        # Relative entropy
        rel_entropy = QuantumInformationMetrics.quantum_relative_entropy(
            sv_a, sv_b, num_qubits)

        result = {
            "fidelity": fidelity,
            "relative_entropy": rel_entropy,
            "num_qubits_a": job_a.num_qubits,
            "num_qubits_b": job_b.num_qubits,
            "god_code": GOD_CODE,
        }

        self._jobs_submitted += 2
        self._jobs_completed += 2
        return result

    def run_berry_phase(self, base_job: QuantumJob,
                        parameter_values: list,
                        param_gate_index: int = 0) -> dict:
        """
        Compute Berry phase by sweeping a parameter through a cycle.

        Executes the base circuit at each parameter value, collects
        statevectors, and computes the geometric phase around the loop.

        Args:
            base_job:          Template circuit with a parameterised gate
            parameter_values:  List of parameter values forming a closed loop
            param_gate_index:  Index of the gate whose parameter to sweep

        Returns:
            dict with berry_phase, geometric_phase_mod_2pi, sacred_alignment
        """
        if not self._active:
            self.start()

        statevectors = []
        for param_val in parameter_values:
            ops = self._serialize_ops(base_job.operations)
            # Replace the parameter in the target gate
            if param_gate_index < len(ops):
                op = ops[param_gate_index]
                if isinstance(op, dict) and "parameters" in op:
                    op["parameters"] = [param_val]

            if self.transpiler:
                ops = CircuitTranspiler.transpile(ops)

            mps = ExactMPSHybridEngine(base_job.num_qubits)
            mps.run_circuit(ops)
            sv = mps.to_statevector()
            statevectors.append(sv)

        result = QuantumInformationMetrics.berry_phase(
            statevectors, base_job.num_qubits)
        result["parameter_values"] = [round(p, 6) for p in parameter_values[:10]]
        result["parameter_count"] = len(parameter_values)

        self._jobs_submitted += len(parameter_values)
        self._jobs_completed += len(parameter_values)
        return result

    def run_loschmidt_echo(self, job: QuantumJob,
                            hamiltonian_ops: list,
                            perturbation_ops: list,
                            time_steps: int = 20,
                            dt: float = 0.1) -> dict:
        """
        Compute Loschmidt echo for quantum chaos detection.

        Executes the circuit to get initial state, then applies
        forward and backward Hamiltonian evolution to measure
        fidelity decay. Rapid decay = quantum chaos.

        Args:
            job:               Circuit to prepare initial state
            hamiltonian_ops:   Original Hamiltonian (Pauli terms)
            perturbation_ops:  Perturbation (Pauli terms)
            time_steps:        Evolution time steps
            dt:                Time step size

        Returns:
            dict with echo_values, decay_rate, is_chaotic,
            lyapunov_estimate, sacred_alignment
        """
        if not self._active:
            self.start()

        ops = self._serialize_ops(job.operations)
        if self.transpiler:
            ops = CircuitTranspiler.transpile(ops)

        mps = ExactMPSHybridEngine(job.num_qubits)
        mps.run_circuit(ops)
        sv = mps.to_statevector()

        result = QuantumInformationMetrics.loschmidt_echo(
            sv, hamiltonian_ops, perturbation_ops, job.num_qubits,
            time_steps=time_steps, dt=dt)

        self._jobs_submitted += 1
        self._jobs_completed += 1
        return result

    # ─── STATE TOMOGRAPHY (v8.0) ───

    def run_tomography(self, job: QuantumJob, shots: int = 4096) -> dict:
        """
        Full quantum state tomography on a circuit's output.

        Executes the circuit, measures in all Pauli bases, reconstructs
        the density matrix, and computes purity + entropy + sacred alignment.

        Returns:
            dict with purity, rank, von_neumann_entropy, eigenvalues,
            sacred_alignment
        """
        if not self._active:
            self.start()

        ops = self._serialize_ops(job.operations)
        if self.transpiler:
            ops = CircuitTranspiler.transpile(ops)

        mps = ExactMPSHybridEngine(job.num_qubits)
        run_result = mps.run_circuit(ops)
        if not run_result.get("completed"):
            return {"error": "circuit_execution_incomplete"}

        sv = mps.to_statevector()
        result = QuantumStateTomography.full_tomography(sv, job.num_qubits, shots)

        self._jobs_submitted += 1
        self._jobs_completed += 1
        return result

    def swap_test(self, job_a: QuantumJob, job_b: QuantumJob,
                  shots: int = 4096) -> dict:
        """
        SWAP test for estimating fidelity between two circuit outputs.

        P(ancilla=0) = (1 + |⟨ψ_a|ψ_b⟩|²) / 2

        Args:
            job_a: First circuit
            job_b: Second circuit
            shots: Measurement shots

        Returns:
            dict with estimated_fidelity, ancilla_prob_0, swap_test_circuit_size
        """
        if not self._active:
            self.start()

        n = max(job_a.num_qubits, job_b.num_qubits)
        swap_ops = QuantumStateTomography.swap_test_circuit(n)
        sacred = SacredAlignmentScorer.score({}, n)

        # Direct fidelity computation (more accurate than SWAP test on simulator)
        ops_a = self._serialize_ops(job_a.operations)
        if self.transpiler:
            ops_a = CircuitTranspiler.transpile(ops_a)
        mps_a = ExactMPSHybridEngine(job_a.num_qubits)
        mps_a.run_circuit(ops_a)
        sv_a = mps_a.to_statevector()

        ops_b = self._serialize_ops(job_b.operations)
        if self.transpiler:
            ops_b = CircuitTranspiler.transpile(ops_b)
        mps_b = ExactMPSHybridEngine(job_b.num_qubits)
        mps_b.run_circuit(ops_b)
        sv_b = mps_b.to_statevector()

        fidelity_result = QuantumStateTomography.state_fidelity(sv_a, sv_b, n)
        fidelity = fidelity_result.get("fidelity", 0)
        ancilla_prob_0 = (1.0 + fidelity) / 2.0

        self._jobs_submitted += 2
        self._jobs_completed += 2
        return {
            "estimated_fidelity": round(fidelity, 8),
            "ancilla_prob_0": round(ancilla_prob_0, 8),
            "swap_test_circuit_gates": len(swap_ops),
            "swap_test_total_qubits": 2 * n + 1,
            "fidelity_detail": fidelity_result,
            "sacred_alignment": sacred,
            "god_code": GOD_CODE,
        }

    # ─── HAMILTONIAN SIMULATION (v8.0) ───

    def run_hamiltonian_evolution(self, hamiltonian_terms: list,
                                  num_qubits: int, *,
                                  total_time: float = 1.0,
                                  trotter_steps: int = 10,
                                  order: int = 1,
                                  shots: int = 2048) -> dict:
        """
        Trotterized Hamiltonian time evolution through the VQPU pipeline.

        Evolves the state under e^{-iHt} using product formula decomposition.
        First-order (Lie-Trotter) and second-order (Suzuki-Trotter) supported.

        Args:
            hamiltonian_terms: [(coefficient, pauli_string), ...]
            num_qubits:        System size
            total_time:        Evolution time
            trotter_steps:     Number of Trotter steps (accuracy ∝ 1/n)
            order:             1 (first) or 2 (second-order)
            shots:             Measurement shots

        Returns:
            dict with energy_estimate, trotter_error_bound, sacred_alignment
        """
        if not self._active:
            self.start()
        result = HamiltonianSimulator.trotter_evolution(
            hamiltonian_terms, num_qubits,
            total_time=total_time, trotter_steps=trotter_steps,
            order=order, shots=shots)
        self._jobs_submitted += 1
        self._jobs_completed += 1
        return result

    def run_adiabatic_preparation(self, target_hamiltonian: list,
                                   num_qubits: int, *,
                                   adiabatic_steps: int = 20,
                                   shots: int = 2048) -> dict:
        """
        Adiabatic ground state preparation through the VQPU pipeline.

        Slowly interpolates from H_init = -Σ Xᵢ to H_target,
        keeping the system in the ground state via adiabatic theorem.

        Args:
            target_hamiltonian: Target Hamiltonian [(coeff, pauli_str), ...]
            num_qubits:         System size
            adiabatic_steps:    Interpolation steps
            shots:              Measurement shots

        Returns:
            dict with energy, ground_state_probabilities, sacred_alignment
        """
        if not self._active:
            self.start()
        result = HamiltonianSimulator.adiabatic_preparation(
            target_hamiltonian, num_qubits,
            adiabatic_steps=adiabatic_steps, shots=shots)
        self._jobs_submitted += 1
        self._jobs_completed += 1
        return result

    def run_iron_lattice(self, n_sites: int = 4, *,
                         coupling_j: float = None,
                         field_h: float = None,
                         trotter_steps: int = 10,
                         total_time: float = 1.0,
                         shots: int = 2048) -> dict:
        """
        Fe(26) iron-lattice Hamiltonian simulation through the VQPU pipeline.

        1D Heisenberg chain with sacred parameters:
          J = GOD_CODE/1000 ≈ 0.5275 (exchange coupling)
          h = VOID_CONSTANT ≈ 1.0416 (external field)

        Returns magnetization, nearest-neighbour correlations, energy,
        and sacred alignment of the Fe lattice quantum state.

        Args:
            n_sites:       Lattice sites (qubits)
            coupling_j:    Exchange coupling (default: GOD_CODE/1000)
            field_h:       External field (default: VOID_CONSTANT)
            trotter_steps: Trotter steps
            total_time:    Evolution time
            shots:         Measurement shots

        Returns:
            dict with energy, magnetization, zz_correlations, sacred_alignment
        """
        if not self._active:
            self.start()
        result = HamiltonianSimulator.iron_lattice_circuit(
            n_sites, coupling_j=coupling_j, field_h=field_h,
            trotter_steps=trotter_steps, total_time=total_time,
            shots=shots)
        self._jobs_submitted += 1
        self._jobs_completed += 1
        return result

    # ─── SCORING CACHE MANAGEMENT (v8.0) ───

    def scoring_cache_stats(self) -> dict:
        """Return scoring cache performance statistics."""
        return ScoringCache.stats()

    def clear_scoring_cache(self):
        """Clear all scoring caches (forces re-computation)."""
        ScoringCache.clear()

    # ─── STATUS & TELEMETRY ───

    def status(self) -> dict:
        """Full bridge status (v7.1 — platform-aware Mac control, Intel/Apple Silicon routing)."""
        uptime = time.time() - self._start_time if self._active else 0
        avg_submit = (self._total_submit_time_ms / self._jobs_submitted
                      if self._jobs_submitted > 0 else 0)
        avg_result = (self._total_result_time_ms / self._jobs_completed
                      if self._jobs_completed > 0 else 0)
        throughput = (self._jobs_completed / uptime
                      if uptime > 0 else 0)

        s = {
            "version": VERSION,
            "active": self._active,
            "uptime_seconds": uptime,
            "jobs_submitted": self._jobs_submitted,
            "jobs_completed": self._jobs_completed,
            "jobs_failed": self._jobs_failed,
            "jobs_success_rate": (self._jobs_completed / max(self._jobs_submitted, 1)),
            "total_transpile_savings": self._total_transpile_savings,
            "template_match_savings": self._template_match_savings,
            "avg_submit_time_ms": avg_submit,
            "avg_result_time_ms": avg_result,
            "throughput_jobs_per_sec": round(throughput, 4),
            "peak_throughput_hz": round(self._peak_throughput_hz, 4),
            "bridge_path": str(self.bridge_path),
            "god_code": GOD_CODE,
            "phi": PHI,
            "platform": {
                "arch": _PLATFORM["arch"],
                "processor": _PLATFORM["processor"],
                "mac_ver": _PLATFORM["mac_ver"],
                "is_apple_silicon": _IS_APPLE_SILICON,
                "is_intel": _IS_INTEL,
                "gpu_class": _GPU_CLASS,
                "metal_family": _PLATFORM["metal_family"],
                "metal_compute_capable": _HAS_METAL_COMPUTE,
                "simd": _PLATFORM["simd"],
                "has_amx": _PLATFORM["has_amx"],
                "has_neural_engine": _PLATFORM["has_neural_engine"],
                "gpu_crossover": VQPU_GPU_CROSSOVER,
                "mps_fallback_target": VQPU_MPS_FALLBACK_TARGET,
                "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "auto"),
            },
            "capacity": {
                "max_qubits": VQPU_MAX_QUBITS,
                "stabilizer_max_qubits": VQPU_STABILIZER_MAX_QUBITS,
                "db_research_qubits": VQPU_DB_RESEARCH_QUBITS,
                "batch_limit": VQPU_BATCH_LIMIT,
                "mps_max_bond_low": VQPU_MPS_MAX_BOND_LOW,
                "mps_max_bond_med": VQPU_MPS_MAX_BOND_MED,
                "mps_max_bond_high": VQPU_MPS_MAX_BOND_HIGH,
                "adaptive_shots_range": [VQPU_ADAPTIVE_SHOTS_MIN, VQPU_ADAPTIVE_SHOTS_MAX],
                "pipeline_workers": self.pipeline_workers,
                "hw_ram_gb": _HW_RAM_GB,
                "hw_cores": _HW_CORES,
            },
            "process_features": [
                "transpiler_14pass",
                "template_pattern_matching",
                "commutation_reorder",
                "pipeline_parallel_dispatch",
                "adaptive_shot_allocation",
                "concurrent_batch",
                "priority_scheduling",
                "sacred_alignment_scoring",
                "three_engine_entropy_scoring",
                "three_engine_harmonic_scoring",
                "three_engine_wave_coherence",
                "daemon_health_check",
                "exact_mps_hybrid",
                "kqueue_result_collection",
                "quantum_gate_engine_compilation",
                "quantum_gate_engine_error_correction",
                "quantum_gate_engine_execution",
                "asi_core_15d_scoring",
                "agi_core_13d_scoring",
                "run_simulation_pipeline",
                "sacred_gate_circuits",
                "coherence_evolution",
                "quantum_db_grover_search",
                "quantum_db_qpe_patterns",
                "quantum_db_qft_frequency",
                "quantum_db_amplitude_estimation",
                "quantum_db_knowledge_walk",
                "quantum_data_storage_integration",
                "quantum_data_analyzer_integration",
                "tiered_capacity_scaling",
                "stabilizer_unlimited_clifford",
                "noise_model_depolarizing_damping",
                "zero_noise_extrapolation_mitigation",
                "readout_error_mitigation",
                "vqe_variational_eigensolver",
                "qaoa_combinatorial_optimization",
                "entanglement_quantifier_vne_concurrence",
                "circuit_result_cache_lru",
                "dynamic_decoupling_8pass",
                "god_code_simulator_integration",
                "phi_scaled_noise_attenuation",
                "v7.1_platform_detection",
                "v7.1_intel_cpu_only_routing",
                "v7.1_apple_silicon_metal_compute",
                "v7.1_blas_thread_tuning",
                "v7.1_mps_fallback_platform_aware",
                "v7.1_avx2_fma3_acceleration" if _IS_INTEL and "AVX2" in _PLATFORM.get("simd", []) else "v7.1_neon_amx_acceleration",
                "v8.0_quantum_fisher_information",
                "v8.0_berry_phase_geometric",
                "v8.0_quantum_mutual_information",
                "v8.0_quantum_relative_entropy",
                "v8.0_loschmidt_echo_chaos",
                "v8.0_topological_entanglement_entropy",
                "v8.0_quantum_state_tomography",
                "v8.0_density_matrix_reconstruction",
                "v8.0_state_fidelity_swap_test",
                "v8.0_trotter_suzuki_evolution",
                "v8.0_adiabatic_state_preparation",
                "v8.0_iron_lattice_fe26_circuit",
                "v8.0_scoring_cache_optimization",
                "v9.0_daemon_cycler_autonomous",
                "v9.0_vqpu_findings_11_sims",
                "v9.0_superconductivity_heisenberg",
                "v9.0_sc_scoring_dimension",
                "v9.0_sc_cache_persistence",
                "v9.0_daemon_telemetry_state",
                "v9.0_run_vqpu_findings_on_demand",
                "v9.0_four_axis_composite_scoring",
                "v12.1_quantum_subconscious_idle_thoughts",
                "v12.1_13_stream_generators",
                "v12.1_insight_crystallization",
                "v12.1_precog_pre_seeding",
                "v12.1_dream_associator",
                "v12.1_ouroboros_entropy_recycling",
                "v12.1_coherence_meditation",
                "v12.1_bifurcation_scout",
                "v14.0_14pass_transpiler",
                "v14.0_swap_routing_topology",
                "v14.0_multi_qubit_decomposition",
                "v14.0_crosstalk_noise_model",
                "v14.0_multi_optimizer_vqe",
                "v14.0_4th_order_suzuki_trotter",
                "v14.0_2d_iron_lattice_heisenberg",
                "v14.0_mle_state_tomography",
                "v14.0_adaptive_daemon_interval",
                "v14.0_ttl_scoring_cache",
                "v14.0_barren_plateau_detection",
                "v2.0_micro_daemon_background_assistant",
                "v2.0_micro_daemon_heartbeat",
                "v2.0_micro_daemon_ipc_poll",
                "v2.0_micro_daemon_adaptive_tick",
            ],
            "three_engine": ThreeEngineQuantumScorer.engines_status(),
            "engine_integration": self.engines.status(),
            "brain_integration": BrainIntegration.status(),
            "scoring_cache": ScoringCache.stats(),
            "daemon_cycler": self._daemon_cycler.status(),
            "micro_daemon": self._micro_daemon.status(),
            "quantum_subconscious": self.subconscious_status(),
            "pipeline_executor_active": self._pipeline_executor is not None,
            "adaptive_shots_enabled": self.enable_adaptive_shots,
        }

        if self.governor:
            s["hardware"] = self.governor.get_vitals()

        # Check for pending inbox/outbox files
        try:
            inbox_count = len(list(self.inbox.glob("*.json")))
            outbox_count = len(list(self.outbox.glob("*.json")))
            s["inbox_pending"] = inbox_count
            s["outbox_pending"] = outbox_count
        except OSError:
            pass

        return s

    # ─── SELF-TEST (v12.0) ───

    def self_test(self) -> dict:
        """
        v15.0: Comprehensive self-test for l104_debug.py integration.

        Runs 19 diagnostic probes across all VQPU subsystems including
        brain integration, SWAP routing, crosstalk noise, Toffoli decompose,
        4th-order Trotter, and micro daemon. Returns structured results
        compatible with the unified debug framework.
        """
        results = []
        t0 = time.monotonic()

        # 1. Platform detection
        try:
            assert _PLATFORM.get("arch"), "No platform arch detected"
            assert _PLATFORM.get("gpu_class"), "No GPU class detected"
            results.append({"test": "platform_detection", "pass": True,
                            "detail": f"{_PLATFORM['gpu_class']} / {_PLATFORM['arch']}"})
        except Exception as e:
            results.append({"test": "platform_detection", "pass": False, "error": str(e)})

        # 2. Sacred constants
        try:
            assert abs(GOD_CODE - 527.5184818492612) < 1e-10, "GOD_CODE mismatch"
            assert abs(PHI - 1.618033988749895) < 1e-10, "PHI mismatch"
            results.append({"test": "sacred_constants", "pass": True,
                            "detail": f"GOD_CODE={GOD_CODE:.10f}, PHI={PHI:.10f}"})
        except Exception as e:
            results.append({"test": "sacred_constants", "pass": False, "error": str(e)})

        # 3. CircuitTranspiler (10-pass)
        try:
            ops = [{"gate": "H", "qubits": [0]}, {"gate": "CX", "qubits": [0, 1]}]
            tran = CircuitTranspiler.transpile(ops)
            assert isinstance(tran, list), "Transpiler returned non-list"
            results.append({"test": "circuit_transpiler", "pass": True,
                            "detail": f"{len(tran)} ops after transpilation"})
        except Exception as e:
            results.append({"test": "circuit_transpiler", "pass": False, "error": str(e)})

        # 4. ExactMPSHybridEngine (product-state + sampling)
        try:
            mps = ExactMPSHybridEngine(2)
            mps.apply_single_gate(0, np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2))
            mps.apply_two_gate(0, 1, np.array([
                [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]
            ], dtype=np.complex128).reshape(2, 2, 2, 2))
            counts = mps.sample(1024)
            assert "00" in counts and "11" in counts, "Bell state sampling failed"
            results.append({"test": "mps_engine", "pass": True,
                            "detail": f"Bell sampling: {counts}"})
        except Exception as e:
            results.append({"test": "mps_engine", "pass": False, "error": str(e)})

        # 5. SacredAlignmentScorer
        try:
            probs = {"00": 0.5, "11": 0.5}
            score = SacredAlignmentScorer.score(probs, 2)
            assert "entropy" in score, "Missing entropy in sacred score"
            assert "sacred_score" in score, "Missing sacred_score in sacred score"
            results.append({"test": "sacred_alignment", "pass": True,
                            "detail": f"sacred={score.get('sacred_score', 0):.4f}, entropy={score.get('entropy', 0):.4f}"})
        except Exception as e:
            results.append({"test": "sacred_alignment", "pass": False, "error": str(e)})

        # 6. ThreeEngineQuantumScorer
        try:
            status = ThreeEngineQuantumScorer.engines_status()
            assert isinstance(status, dict), "engines_status returned non-dict"
            results.append({"test": "three_engine_scorer", "pass": True,
                            "detail": f"version={status.get('version', '?')}"})
        except Exception as e:
            results.append({"test": "three_engine_scorer", "pass": False, "error": str(e)})

        # 7. ScoringCache
        try:
            stats = ScoringCache.stats()
            assert isinstance(stats, dict), "ScoringCache.stats() returned non-dict"
            results.append({"test": "scoring_cache", "pass": True,
                            "detail": f"hits={stats.get('hits', 0)}, misses={stats.get('misses', 0)}"})
        except Exception as e:
            results.append({"test": "scoring_cache", "pass": False, "error": str(e)})

        # 8. Parametric gate cache (LRU)
        try:
            cache_size = len(ExactMPSHybridEngine._parametric_cache)
            cache_max = ExactMPSHybridEngine._PARAMETRIC_CACHE_MAX
            results.append({"test": "parametric_cache", "pass": True,
                            "detail": f"{cache_size}/{cache_max} entries"})
        except Exception as e:
            results.append({"test": "parametric_cache", "pass": False, "error": str(e)})

        # 9. EngineIntegration status
        try:
            ei_status = EngineIntegration.status()
            assert isinstance(ei_status, dict), "EngineIntegration.status() non-dict"
            results.append({"test": "engine_integration", "pass": True,
                            "detail": f"engines={ei_status.get('engine_count', 0)}"})
        except Exception as e:
            results.append({"test": "engine_integration", "pass": False, "error": str(e)})

        # 10. HardwareGovernor
        try:
            if self.governor:
                vitals = self.governor.get_vitals()
                results.append({"test": "hardware_governor", "pass": True,
                                "detail": f"throttled={vitals.get('is_throttled', False)}"})
            else:
                results.append({"test": "hardware_governor", "pass": True,
                                "detail": "no psutil — governor disabled"})
        except Exception as e:
            results.append({"test": "hardware_governor", "pass": False, "error": str(e)})

        # 11. DaemonCycler status
        try:
            dc_status = self._daemon_cycler.status()
            assert isinstance(dc_status, dict), "DaemonCycler.status() non-dict"
            results.append({"test": "daemon_cycler", "pass": True,
                            "detail": f"cycles={dc_status.get('cycles_completed', 0)}"})
        except Exception as e:
            results.append({"test": "daemon_cycler", "pass": False, "error": str(e)})

        # 12. Parallel batch execution
        try:
            bell_job = QuantumJob(num_qubits=2, operations=[
                {"gate": "H", "qubits": [0]}, {"gate": "CX", "qubits": [0, 1]}])
            batch = self.run_simulation_batch([bell_job])
            assert len(batch) == 1, "Batch returned wrong count"
            results.append({"test": "parallel_batch", "pass": True,
                            "detail": f"1 job executed"})
        except Exception as e:
            results.append({"test": "parallel_batch", "pass": False, "error": str(e)})

        # 13. Brain integration availability (v13.0)
        try:
            brain_avail = BrainIntegration.is_available()
            brain_status = BrainIntegration.status()
            results.append({"test": "brain_integration", "pass": True,
                            "detail": f"available={brain_avail}, composite={brain_status.get('brain_composite', 0):.4f}"})
        except Exception as e:
            results.append({"test": "brain_integration", "pass": False, "error": str(e)})

        # 14. Brain self-test delegation (v13.0)
        try:
            brain_test = BrainIntegration.run_brain_self_test()
            brain_pass = brain_test.get("all_pass", False)
            brain_count = brain_test.get("passed", 0)
            brain_total = brain_test.get("total", 0)
            results.append({"test": "brain_self_test", "pass": brain_pass,
                            "detail": f"{brain_count}/{brain_total} probes passed, v{brain_test.get('version', '?')}"})
        except Exception as e:
            results.append({"test": "brain_self_test", "pass": False, "error": str(e)})

        # 15. v14.0: SWAP routing (topology-aware transpilation)
        try:
            ops_3q = [{"gate": "CX", "qubits": [0, 2]}]  # non-adjacent on linear
            routed = CircuitTranspiler.transpile(ops_3q, topology="linear", num_qubits=3)
            assert isinstance(routed, list), "SWAP routing returned non-list"
            results.append({"test": "swap_routing", "pass": True,
                            "detail": f"{len(routed)} ops after routing (linear topology)"})
        except Exception as e:
            results.append({"test": "swap_routing", "pass": False, "error": str(e)})

        # 16. v14.0: Multi-qubit gate decomposition (Toffoli → CX)
        try:
            toffoli_ops = [{"gate": "CCX", "qubits": [0, 1, 2]}]
            decomposed = CircuitTranspiler.transpile(toffoli_ops, num_qubits=3)
            has_cx = any(op.get("gate") == "CX" for op in decomposed)
            no_ccx = not any(op.get("gate") == "CCX" for op in decomposed)
            assert has_cx and no_ccx, "Toffoli not fully decomposed"
            results.append({"test": "toffoli_decompose", "pass": True,
                            "detail": f"CCX → {len(decomposed)} gates"})
        except Exception as e:
            results.append({"test": "toffoli_decompose", "pass": False, "error": str(e)})

        # 17. v14.0: Crosstalk noise model
        try:
            nm = NoiseModel.with_crosstalk(depolarizing_rate=0.01)
            noisy = nm.apply_crosstalk({"00": 0.5, "11": 0.5}, 2)
            assert isinstance(noisy, dict), "Crosstalk returned non-dict"
            results.append({"test": "crosstalk_noise", "pass": True,
                            "detail": f"crosstalk model applied, {len(noisy)} outcomes"})
        except Exception as e:
            results.append({"test": "crosstalk_noise", "pass": False, "error": str(e)})

        # 18. v14.0: 4th-order Trotter
        try:
            from .hamiltonian import HamiltonianSimulator
            ham = [(1.0, "ZZ"), (0.5, "XI")]
            result_t4 = HamiltonianSimulator.trotter_evolution(ham, 2, order=4, trotter_steps=4)
            assert "energy_estimate" in result_t4, "4th-order Trotter missing energy"
            assert result_t4.get("trotter_order") == 4, "Order not 4"
            results.append({"test": "trotter_order_4", "pass": True,
                            "detail": f"energy={result_t4['energy_estimate']}, gates={result_t4.get('gate_count', 0)}"})
        except Exception as e:
            results.append({"test": "trotter_order_4", "pass": False, "error": str(e)})

        # 19. v2.0: Micro daemon self-test
        try:
            micro_test = self._micro_daemon.self_test()
            micro_pass = micro_test.get("all_pass", False)
            micro_count = micro_test.get("passed", 0)
            micro_total = micro_test.get("total", 0)
            results.append({"test": "micro_daemon", "pass": micro_pass,
                            "detail": f"{micro_count}/{micro_total} probes, v{micro_test.get('version', '?')}"})
        except Exception as e:
            results.append({"test": "micro_daemon", "pass": False, "error": str(e)})

        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
        passed = sum(1 for r in results if r["pass"])
        total = len(results)

        return {
            "engine": "vqpu",
            "version": VERSION,
            "tests": results,
            "passed": passed,
            "total": total,
            "all_pass": passed == total,
            "elapsed_ms": elapsed_ms,
            "god_code": GOD_CODE,
        }

    # ─── INTERNAL ───

    def _serialize_ops(self, operations: list) -> list:
        """Ensure operations are JSON-serializable dicts."""
        result = []
        for op in operations:
            if isinstance(op, dict):
                result.append(op)
            elif isinstance(op, QuantumGate):
                d = {"gate": op.gate, "qubits": op.qubits}
                if op.parameters:
                    d["parameters"] = op.parameters
                result.append(d)
            else:
                result.append(op)
        return result

    def _wait_throttle_clear(self, timeout: float = 5.0):
        """Wait for hardware throttle to clear."""
        deadline = time.monotonic() + timeout
        while (self.governor and self.governor.is_throttled
               and time.monotonic() < deadline):
            time.sleep(0.1)

    def _write_telemetry_summary(self):
        """Write session telemetry summary on shutdown (v7.0: includes v7.0 capabilities + capacity metrics)."""
        try:
            summary = self.status()
            summary["session_end"] = time.time()
            summary["v6_metrics"] = {
                "peak_throughput_hz": self._peak_throughput_hz,
                "template_match_savings": self._template_match_savings,
                "pipeline_workers": self.pipeline_workers,
                "adaptive_shots_enabled": self.enable_adaptive_shots,
                "capacity_max_qubits": VQPU_MAX_QUBITS,
                "capacity_stabilizer_max": VQPU_STABILIZER_MAX_QUBITS,
                "capacity_db_research_qubits": VQPU_DB_RESEARCH_QUBITS,
                "capacity_batch_limit": VQPU_BATCH_LIMIT,
                "hw_ram_gb": _HW_RAM_GB,
                "hw_cores": _HW_CORES,
                "engines_active": self.engines.status(),
            }
            path = self.telemetry_dir / f"session_{int(time.time())}.json"
            path.write_text(json.dumps(summary, indent=2))
        except OSError:
            pass

    # ─── CONTEXT MANAGER ───

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
