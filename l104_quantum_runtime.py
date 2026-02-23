#!/usr/bin/env python3
"""
===============================================================================
L104 QUANTUM RUNTIME BRIDGE v1.0.0
===============================================================================

Centralized real IBM QPU execution bridge for the entire L104 ASI system.

This module provides a single entry point for ALL quantum subsystems to:
  1. Execute circuits on REAL IBM QPU hardware (ibm_fez, ibm_torino, ibm_marrakesh)
  2. Fall back gracefully to local Statevector simulation when QPU is unavailable
  3. Cache results, track telemetry, manage sessions

ARCHITECTURE:
  QuantumRuntime (singleton)
    ├── connect()              → IAM auth + backend selection
    ├── execute(circuit)       → Real QPU or Statevector fallback → probabilities
    ├── execute_sampler(circuit, shots) → Raw measurement counts from real QPU
    ├── get_backend_info()     → Current backend details
    ├── get_telemetry()        → Execution stats, queue depths, fidelity
    └── disconnect()           → Clean shutdown

CONSUMERS:
  l104_quantum_coherence.py    — Grover, QAOA, VQE, QPE, walk, kernel, etc.
  l104_asi/quantum.py          — VQE, QAOA, ZNE, QRC, QKM, QPE
  l104_agi/core.py             — pipeline health, subsystem routing
  l104_code_engine/quantum.py  — code intelligence quantum methods
  l104_server/engines_quantum.py — consciousness engine, memory bank

TOKEN:     Set IBMQ_TOKEN or IBM_QUANTUM_TOKEN env var
CHANNEL:   ibm_quantum_platform (default for 2025+ accounts)
BACKENDS:  ibm_fez (156q), ibm_torino (133q), ibm_marrakesh (156q)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import os
import math
import time
import sys
import threading
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from pathlib import Path
import numpy as np

# ═══ LOAD .env INTO os.environ (before any token reads) ═══
try:
    from dotenv import load_dotenv
    # Walk up from this file to find .env at project root
    _env_path = Path(__file__).resolve().parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
    else:
        load_dotenv(override=False)  # search CWD
except ImportError:
    # Manual .env fallback if python-dotenv isn't installed
    _env_path = Path(__file__).resolve().parent / '.env'
    if _env_path.exists():
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _, _v = _line.partition('=')
                    os.environ.setdefault(_k.strip(), _v.strip())

# Sacred Constants
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # 527.5184818492612

# ═══ QISKIT 2.3.0 IMPORTS ═══
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = type('QuantumCircuit', (), {})

# ═══ QISKIT IBM RUNTIME IMPORTS ═══
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False
    QiskitRuntimeService = None
    Session = None
    SamplerV2 = None

# ═══ AER SIMULATOR FALLBACK ═══
try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionMode(Enum):
    """How a circuit was executed."""
    REAL_QPU = "real_qpu"
    AER_SIMULATOR = "aer_simulator"
    STATEVECTOR = "statevector"
    FAILED = "failed"


@dataclass
class BackendInfo:
    """Information about the selected quantum backend."""
    name: str = "none"
    num_qubits: int = 0
    quantum_volume: int = 0
    pending_jobs: int = 0
    is_real: bool = False
    is_simulator: bool = False
    error_rate: float = 0.0
    basis_gates: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Result from a quantum circuit execution."""
    probabilities: Dict[str, float]          # bitstring → probability
    counts: Dict[str, int]                    # bitstring → count (shots)
    mode: ExecutionMode                       # how it was executed
    backend_name: str = "statevector"
    shots: int = 0
    execution_time_ms: float = 0.0
    job_id: str = ""
    transpiled_depth: int = 0
    transpiled_gate_count: int = 0
    num_qubits: int = 0
    fidelity_estimate: float = 1.0

    def prob_array(self, n_qubits: int) -> np.ndarray:
        """Convert probabilities dict to numpy array indexed by integer state."""
        arr = np.zeros(2 ** n_qubits)
        for bitstring, prob in self.probabilities.items():
            try:
                idx = int(bitstring, 2)
                if idx < len(arr):
                    arr[idx] = prob
            except ValueError:
                continue
        # Normalize
        total = arr.sum()
        if total > 0:
            arr /= total
        return arr

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON/Swift consumption."""
        return {
            "mode": self.mode.value,
            "backend": self.backend_name,
            "shots": self.shots,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "job_id": self.job_id,
            "transpiled_depth": self.transpiled_depth,
            "transpiled_gate_count": self.transpiled_gate_count,
            "num_qubits": self.num_qubits,
            "fidelity_estimate": round(self.fidelity_estimate, 6),
            "top_5_counts": dict(sorted(self.counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_5_probs": dict(sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)[:5]),
        }


@dataclass
class RuntimeTelemetry:
    """Telemetry counters for tracking quantum execution."""
    total_executions: int = 0
    real_qpu_executions: int = 0
    simulator_executions: int = 0
    statevector_executions: int = 0
    failed_executions: int = 0
    total_shots_consumed: int = 0
    total_circuits_submitted: int = 0
    total_execution_time_ms: float = 0.0
    avg_queue_wait_ms: float = 0.0
    cache_hits: int = 0
    last_execution_time: float = 0.0
    last_backend: str = ""
    last_job_id: str = ""
    backends_used: Dict[str, int] = field(default_factory=dict)
    algorithm_counts: Dict[str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM RUNTIME — SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumRuntime:
    """
    Centralized real IBM QPU execution bridge for the L104 ASI system.

    Singleton that manages:
    - Connection to IBM Quantum Platform (IAM auth + backend selection)
    - Circuit execution on real QPU or Statevector fallback
    - Result caching and telemetry
    - Graceful degradation when hardware is unavailable
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._service: Optional[Any] = None
        self._backend: Optional[Any] = None
        self._backend_info = BackendInfo()
        self._telemetry = RuntimeTelemetry()
        self._connected = False
        self._use_real_hardware = True  # Prefer real QPU
        self._default_shots = 4096
        self._optimization_level = 2   # Transpiler optimization (0-3)
        self._max_qubits_for_real = 25  # Allow 25q ASI runs by default
        self._result_cache: Dict[str, ExecutionResult] = {}
        self._cache_max = 1000
        self._execution_log: deque = deque(maxlen=500)
        self._session: Optional[Any] = None
        self._session_lock = threading.Lock()

        # Auto-connect on init
        self._auto_connect()
        self._initialized = True

    # ═══════════════════════════════════════════════════════════════════
    # CONNECTION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _auto_connect(self):
        """Attempt automatic connection using environment token."""
        token = os.environ.get('IBMQ_TOKEN') or os.environ.get('IBM_QUANTUM_TOKEN')
        if token and RUNTIME_AVAILABLE:
            try:
                self.connect(token)
            except Exception as e:
                print(f"[QUANTUM_RUNTIME] Auto-connect failed: {e}")
                self._setup_fallback()
        else:
            self._setup_fallback()

    def connect(self, token: str = None, channel: str = None,
                prefer_backend: str = None) -> Dict[str, Any]:
        """
        Connect to IBM Quantum Platform.

        Args:
            token: IBM Quantum API token (falls back to env var)
            channel: Channel name (default: ibm_quantum_platform)
            prefer_backend: Preferred backend name (e.g., 'ibm_fez')

        Returns:
            Connection status dict
        """
        token = token or os.environ.get('IBMQ_TOKEN') or os.environ.get('IBM_QUANTUM_TOKEN')
        channel = channel or os.environ.get('IBM_QUANTUM_CHANNEL', 'ibm_quantum_platform')
        instance = os.environ.get('IBM_QUANTUM_INSTANCE', None)

        if not token:
            self._setup_fallback()
            return {"connected": False, "error": "No IBM Quantum token provided"}

        if not RUNTIME_AVAILABLE:
            self._setup_fallback()
            return {"connected": False, "error": "qiskit-ibm-runtime not installed"}

        try:
            svc_kwargs = dict(channel=channel, token=token)
            if instance:
                svc_kwargs['instance'] = instance
            self._service = QiskitRuntimeService(**svc_kwargs)

            # Get available real backends
            backends = self._service.backends(
                simulator=False,
                operational=True,
                min_num_qubits=20
            )

            if not backends:
                print("[QUANTUM_RUNTIME] No real QPU backends available, using simulator")
                self._setup_fallback()
                return {"connected": False, "error": "No operational backends found"}

            # Select backend
            if prefer_backend:
                selected = [b for b in backends if b.name == prefer_backend]
                if selected:
                    self._backend = selected[0]
                else:
                    # Fall back to least-busy
                    backends_sorted = sorted(backends, key=lambda b: b.status().pending_jobs)
                    self._backend = backends_sorted[0]
            else:
                # Select least-busy backend
                backends_sorted = sorted(backends, key=lambda b: b.status().pending_jobs)
                self._backend = backends_sorted[0]

            # Populate backend info
            status = self._backend.status()
            self._backend_info = BackendInfo(
                name=self._backend.name,
                num_qubits=self._backend.num_qubits,
                quantum_volume=getattr(self._backend, 'quantum_volume', 0) or 0,
                pending_jobs=status.pending_jobs,
                is_real=True,
                is_simulator=False,
                error_rate=self._estimate_error_rate(),
                basis_gates=list(getattr(self._backend, 'basis_gates', []) or [])
            )

            self._connected = True
            self._use_real_hardware = True

            available_names = [b.name for b in backends]
            print(f"[QUANTUM_RUNTIME] ✓ Connected to {self._backend.name} "
                  f"({self._backend.num_qubits}q, queue: {status.pending_jobs})")
            print(f"[QUANTUM_RUNTIME]   Available backends: {available_names}")

            return {
                "connected": True,
                "backend": self._backend.name,
                "num_qubits": self._backend.num_qubits,
                "queue_depth": status.pending_jobs,
                "available_backends": available_names,
                "mode": "real_qpu"
            }

        except Exception as e:
            print(f"[QUANTUM_RUNTIME] Connection error: {e}")
            self._setup_fallback()
            return {"connected": False, "error": str(e)}

    def _setup_fallback(self):
        """Setup Statevector fallback when no real QPU available."""
        self._connected = False
        self._use_real_hardware = False
        self._backend = None
        self._backend_info = BackendInfo(
            name="statevector_simulator",
            num_qubits=25,
            is_simulator=True
        )
        print("[QUANTUM_RUNTIME] Using Statevector simulator (local)")

    def _estimate_error_rate(self) -> float:
        """Estimate average gate error rate for the backend."""
        if not self._backend:
            return 0.0
        try:
            props = self._backend.properties()
            if props:
                errors = [g.error for g in props.gates if g.error is not None]
                return sum(errors) / len(errors) if errors else 0.01
        except Exception:
            pass
        return 0.01

    def disconnect(self):
        """Disconnect from IBM Quantum Platform."""
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        self._service = None
        self._backend = None
        self._connected = False
        self._use_real_hardware = False
        self._backend_info = BackendInfo(name="disconnected")
        print("[QUANTUM_RUNTIME] Disconnected")

    def switch_backend(self, backend_name: str) -> Dict[str, Any]:
        """Switch to a different backend."""
        if not self._service:
            return {"error": "Not connected to IBM Quantum"}
        try:
            self._backend = self._service.backend(backend_name)
            status = self._backend.status()
            self._backend_info = BackendInfo(
                name=self._backend.name,
                num_qubits=self._backend.num_qubits,
                quantum_volume=getattr(self._backend, 'quantum_volume', 0) or 0,
                pending_jobs=status.pending_jobs,
                is_real=True,
                is_simulator=False,
                error_rate=self._estimate_error_rate(),
            )
            print(f"[QUANTUM_RUNTIME] Switched to {backend_name}")
            return {"backend": backend_name, "num_qubits": self._backend.num_qubits}
        except Exception as e:
            return {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    # CIRCUIT EXECUTION — THE CORE METHOD
    # ═══════════════════════════════════════════════════════════════════

    def execute(self, circuit: 'QuantumCircuit', shots: int = None,
                algorithm_name: str = "unknown",
                force_simulator: bool = False,
                force_real: bool = False) -> ExecutionResult:
        """
        Execute a quantum circuit — routes to real QPU or Statevector automatically.

        This is the primary method all ASI subsystems should call.

        Args:
            circuit: Qiskit QuantumCircuit (measurements will be added if needed)
            shots: Number of measurement shots (default: 4096)
            algorithm_name: Name for telemetry tracking
            force_simulator: Force local Statevector simulation
            force_real: Force real QPU execution (will fail if not connected)

        Returns:
            ExecutionResult with probabilities, counts, and metadata
        """
        shots = shots or self._default_shots
        n_qubits = circuit.num_qubits
        start_time = time.time()

        # Update telemetry
        self._telemetry.total_executions += 1
        self._telemetry.algorithm_counts[algorithm_name] = \
            self._telemetry.algorithm_counts.get(algorithm_name, 0) + 1

        # Check cache
        cache_key = self._circuit_hash(circuit, shots)
        if cache_key in self._result_cache:
            self._telemetry.cache_hits += 1
            cached = self._result_cache[cache_key]
            return cached

        # Decide execution path
        use_real = (
            self._connected
            and self._use_real_hardware
            and not force_simulator
            and n_qubits <= self._max_qubits_for_real
        ) or force_real

        if use_real and self._connected and self._backend:
            result = self._execute_real_qpu(circuit, shots, algorithm_name)
        else:
            result = self._execute_statevector(circuit, n_qubits, algorithm_name)

        result.execution_time_ms = (time.time() - start_time) * 1000
        self._telemetry.total_execution_time_ms += result.execution_time_ms
        self._telemetry.last_execution_time = time.time()
        self._telemetry.last_backend = result.backend_name

        # Cache result
        if len(self._result_cache) >= self._cache_max:
            # Remove oldest entries
            oldest_keys = list(self._result_cache.keys())[:100]
            for k in oldest_keys:
                del self._result_cache[k]
        self._result_cache[cache_key] = result

        # Log execution
        self._execution_log.append({
            "time": time.time(),
            "algorithm": algorithm_name,
            "mode": result.mode.value,
            "backend": result.backend_name,
            "qubits": n_qubits,
            "shots": shots,
            "time_ms": round(result.execution_time_ms, 2),
            "job_id": result.job_id,
        })

        return result

    def _execute_real_qpu(self, circuit: 'QuantumCircuit', shots: int,
                          algorithm_name: str) -> ExecutionResult:
        """Execute circuit on real IBM QPU via SamplerV2."""
        n_qubits = circuit.num_qubits

        try:
            # Clone circuit and add measurements if not present
            qc = circuit.copy()
            if not qc.count_ops().get('measure', 0):
                qc.measure_all()

            # Transpile for target backend
            pm = generate_preset_pass_manager(
                backend=self._backend,
                optimization_level=self._optimization_level
            )
            qc_transpiled = pm.run(qc)

            transpiled_depth = qc_transpiled.depth()
            transpiled_gates = sum(qc_transpiled.count_ops().values())

            print(f"[QUANTUM_RUNTIME] Submitting to {self._backend.name}: "
                  f"{n_qubits}q, depth={transpiled_depth}, gates={transpiled_gates}, shots={shots}")

            # Execute via SamplerV2 in job mode (compatible with Open plan)
            # Open plan doesn't support Sessions — pass backend directly
            sampler = SamplerV2(mode=self._backend)
            job = sampler.run([qc_transpiled], shots=shots)
            print(f"[QUANTUM_RUNTIME] Job submitted: {job.job_id()}")
            result = job.result()

            job_id = job.job_id() if hasattr(job, 'job_id') else str(id(job))

            # Parse results — SamplerV2 returns PubResult
            counts = {}
            try:
                pub_result = result[0]
                # SamplerV2 returns BitArray in data
                data_dict = pub_result.data
                # Get the first classical register's bit array
                for attr_name in dir(data_dict):
                    if not attr_name.startswith('_'):
                        bit_array = getattr(data_dict, attr_name, None)
                        if bit_array is not None and hasattr(bit_array, 'get_counts'):
                            counts = bit_array.get_counts()
                            break

                if not counts:
                    # Fallback: try older API
                    counts = result.get_counts() if hasattr(result, 'get_counts') else {}
            except Exception as parse_err:
                print(f"[QUANTUM_RUNTIME] Result parsing: {parse_err}")
                # Try multiple fallback approaches
                try:
                    counts = result[0].data.meas.get_counts()
                except Exception:
                    try:
                        counts = result[0].data.c.get_counts()
                    except Exception:
                        counts = {}

            if not counts:
                print("[QUANTUM_RUNTIME] WARNING: No counts from QPU, falling back to Statevector")
                return self._execute_statevector(circuit, n_qubits, algorithm_name)

            # Convert counts to probabilities
            total_shots = sum(counts.values())
            probabilities = {k: v / total_shots for k, v in counts.items()}

            self._telemetry.real_qpu_executions += 1
            self._telemetry.total_shots_consumed += total_shots
            self._telemetry.total_circuits_submitted += 1
            self._telemetry.last_job_id = job_id
            self._telemetry.backends_used[self._backend.name] = \
                self._telemetry.backends_used.get(self._backend.name, 0) + 1

            print(f"[QUANTUM_RUNTIME] ✓ Real QPU result: {len(counts)} unique states, "
                  f"job={job_id[:16]}...")

            return ExecutionResult(
                probabilities=probabilities,
                counts=counts,
                mode=ExecutionMode.REAL_QPU,
                backend_name=self._backend.name,
                shots=total_shots,
                job_id=job_id,
                transpiled_depth=transpiled_depth,
                transpiled_gate_count=transpiled_gates,
                num_qubits=n_qubits,
                fidelity_estimate=max(0.0, 1.0 - self._backend_info.error_rate * transpiled_depth * 0.1),
            )

        except Exception as e:
            print(f"[QUANTUM_RUNTIME] QPU execution failed: {e}, falling back to Statevector")
            self._telemetry.failed_executions += 1
            return self._execute_statevector(circuit, n_qubits, algorithm_name)

    def _execute_statevector(self, circuit: 'QuantumCircuit', n_qubits: int,
                             algorithm_name: str) -> ExecutionResult:
        """Execute circuit using local Statevector simulation."""
        try:
            # Remove measurement gates for Statevector simulation
            qc = circuit.copy()
            qc.remove_final_measurements(inplace=True)

            sv = Statevector.from_label('0' * n_qubits).evolve(qc)
            probs = np.abs(sv.data) ** 2

            # Convert to bitstring probabilities
            probabilities = {}
            counts = {}
            synthetic_shots = 4096
            for i in range(len(probs)):
                if probs[i] > 1e-10:
                    bitstring = format(i, f'0{n_qubits}b')
                    probabilities[bitstring] = float(probs[i])
                    counts[bitstring] = int(round(probs[i] * synthetic_shots))

            self._telemetry.statevector_executions += 1

            return ExecutionResult(
                probabilities=probabilities,
                counts=counts,
                mode=ExecutionMode.STATEVECTOR,
                backend_name="statevector_simulator",
                shots=0,
                num_qubits=n_qubits,
                fidelity_estimate=1.0,  # Perfect simulation
            )

        except Exception as e:
            print(f"[QUANTUM_RUNTIME] Statevector execution failed: {e}")
            self._telemetry.failed_executions += 1
            return ExecutionResult(
                probabilities={},
                counts={},
                mode=ExecutionMode.FAILED,
                backend_name="none",
                num_qubits=n_qubits,
            )

    def benchmark_asi_25_qubits(self,
                                runs: int = 3,
                                shots: int = 4096,
                                force_simulator: bool = False) -> Dict[str, Any]:
        """
        Run a dedicated ASI benchmark on a 25-qubit GHZ-style circuit.

        Returns summary with per-run timings, execution modes, and peak state.
        """
        if not QISKIT_AVAILABLE:
            return {
                "success": False,
                "error": "Qiskit not available",
                "requested_qubits": 25,
            }

        target_qubits = 25
        runs = max(1, int(runs))
        shots = max(128, int(shots))

        # Ensure real-QPU path can accept 25q when connected
        self.set_max_qubits(max(self._max_qubits_for_real, target_qubits))

        benchmark_runs: List[Dict[str, Any]] = []
        t0 = time.time()

        telemetry_before = self.get_telemetry()

        for i in range(runs):
            qc = QuantumCircuit(target_qubits)
            qc.h(0)
            for q in range(1, target_qubits):
                qc.cx(q - 1, q)
            # Make each run circuit-unique to avoid cache reuse in timing benchmarks
            qc.rz(((GOD_CODE / 1000.0) + (i * 1e-6)) % (2 * math.pi), target_qubits - 1)

            run_start = time.time()
            result = self.execute(
                qc,
                shots=shots,
                algorithm_name="asi_25q_benchmark",
                force_simulator=force_simulator,
            )
            run_ms = (time.time() - run_start) * 1000.0

            if result.probabilities:
                top_state, top_prob = max(result.probabilities.items(), key=lambda x: x[1])
            else:
                top_state, top_prob = "", 0.0

            benchmark_runs.append({
                "run": i + 1,
                "mode": result.mode.value,
                "backend": result.backend_name,
                "time_ms": round(run_ms, 3),
                "execution_time_ms": round(result.execution_time_ms, 3),
                "num_qubits": result.num_qubits,
                "job_id": result.job_id,
                "top_state": top_state,
                "top_probability": round(float(top_prob), 8),
                "transpiled_depth": result.transpiled_depth,
                "transpiled_gate_count": result.transpiled_gate_count,
                "fidelity_estimate": round(float(result.fidelity_estimate), 8),
            })

        telemetry_after = self.get_telemetry()

        avg_ms = sum(r["time_ms"] for r in benchmark_runs) / max(1, len(benchmark_runs))

        return {
            "success": True,
            "benchmark": "asi_25q",
            "requested_qubits": target_qubits,
            "runs": benchmark_runs,
            "summary": {
                "runs": runs,
                "shots": shots,
                "avg_time_ms": round(avg_ms, 3),
                "total_time_ms": round((time.time() - t0) * 1000.0, 3),
                "mode_counts": {
                    "real_qpu": sum(1 for r in benchmark_runs if r["mode"] == "real_qpu"),
                    "statevector": sum(1 for r in benchmark_runs if r["mode"] == "statevector"),
                    "aer_simulator": sum(1 for r in benchmark_runs if r["mode"] == "aer_simulator"),
                    "failed": sum(1 for r in benchmark_runs if r["mode"] == "failed"),
                },
            },
            "telemetry_delta": {
                "total_executions": telemetry_after["total_executions"] - telemetry_before["total_executions"],
                "real_qpu_executions": telemetry_after["real_qpu_executions"] - telemetry_before["real_qpu_executions"],
                "statevector_executions": telemetry_after["statevector_executions"] - telemetry_before["statevector_executions"],
                "failed_executions": telemetry_after["failed_executions"] - telemetry_before["failed_executions"],
                "cache_hits": telemetry_after["cache_hits"] - telemetry_before["cache_hits"],
                "total_shots_consumed": telemetry_after["total_shots_consumed"] - telemetry_before["total_shots_consumed"],
            },
            "runtime_mode": self.mode,
            "connected": self._connected,
            "backend": self.get_backend_info(),
            "sacred_constants": {
                "god_code": GOD_CODE,
                "phi": PHI,
            },
        }

    def _circuit_hash(self, circuit: 'QuantumCircuit', shots: int) -> str:
        """Generate a hash key for circuit caching."""
        try:
            qasm = circuit.qasm() if hasattr(circuit, 'qasm') else str(circuit)
        except Exception:
            qasm = str(circuit)
        key = f"{qasm}:{shots}:{self._backend_info.name}"
        return hashlib.md5(key.encode()).hexdigest()

    # ═══════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS FOR ASI SUBSYSTEMS
    # ═══════════════════════════════════════════════════════════════════

    def execute_and_get_probs(self, circuit: 'QuantumCircuit',
                               n_qubits: int = None,
                               algorithm_name: str = "unknown",
                               force_simulator: bool = False) -> Tuple[np.ndarray, ExecutionResult]:
        """
        Execute circuit and return probability array + full result.

        This is the drop-in replacement for:
            sv = Statevector.from_label('0' * n).evolve(qc)
            probs = np.abs(sv.data) ** 2

        Returns:
            (probs_array, execution_result) where probs_array is shape (2^n,)
        """
        n = n_qubits or circuit.num_qubits
        result = self.execute(circuit, algorithm_name=algorithm_name,
                             force_simulator=force_simulator)
        probs = result.prob_array(n)
        return probs, result

    def execute_and_get_counts(self, circuit: 'QuantumCircuit',
                                shots: int = None,
                                algorithm_name: str = "unknown") -> Tuple[Dict[str, int], ExecutionResult]:
        """
        Execute circuit and return measurement counts + full result.

        Returns:
            (counts_dict, execution_result) where counts_dict is {"bitstring": count}
        """
        result = self.execute(circuit, shots=shots, algorithm_name=algorithm_name)
        return result.counts, result

    # ═══════════════════════════════════════════════════════════════════
    # STATUS & TELEMETRY
    # ═══════════════════════════════════════════════════════════════════

    @property
    def is_connected(self) -> bool:
        """Whether connected to real IBM QPU."""
        return self._connected

    @property
    def is_real_hardware(self) -> bool:
        """Whether using real QPU (vs simulator)."""
        return self._connected and self._use_real_hardware

    @property
    def backend_name(self) -> str:
        """Current backend name."""
        return self._backend_info.name

    @property
    def mode(self) -> str:
        """Current execution mode string."""
        if self._connected and self._use_real_hardware:
            return "real_qpu"
        return "statevector"

    def set_real_hardware(self, enabled: bool):
        """Toggle real hardware usage (when connected)."""
        self._use_real_hardware = enabled
        if enabled and not self._connected:
            self._auto_connect()

    def set_optimization_level(self, level: int):
        """Set transpiler optimization level (0-3)."""
        self._optimization_level = max(0, min(3, level))

    def set_default_shots(self, shots: int):
        """Set default shot count."""
        self._default_shots = max(100, min(100000, shots))

    def set_max_qubits(self, max_q: int):
        """Set maximum qubit count for real QPU submission."""
        self._max_qubits_for_real = max(1, min(156, max_q))

    def get_backend_info(self) -> Dict[str, Any]:
        """Get current backend information."""
        info = {
            "name": self._backend_info.name,
            "num_qubits": self._backend_info.num_qubits,
            "quantum_volume": self._backend_info.quantum_volume,
            "pending_jobs": self._backend_info.pending_jobs,
            "is_real": self._backend_info.is_real,
            "is_simulator": self._backend_info.is_simulator,
            "error_rate": round(self._backend_info.error_rate, 6),
            "connected": self._connected,
            "mode": self.mode,
        }

        # Refresh queue depth if connected
        if self._connected and self._backend:
            try:
                status = self._backend.status()
                info["pending_jobs"] = status.pending_jobs
                self._backend_info.pending_jobs = status.pending_jobs
            except Exception:
                pass

        return info

    def get_available_backends(self) -> List[Dict[str, Any]]:
        """List all available backends."""
        if not self._service:
            return []
        try:
            backends = self._service.backends(simulator=False, operational=True)
            return [
                {
                    "name": b.name,
                    "num_qubits": b.num_qubits,
                    "pending_jobs": b.status().pending_jobs,
                    "quantum_volume": getattr(b, 'quantum_volume', 0) or 0,
                }
                for b in backends
            ]
        except Exception:
            return []

    def get_telemetry(self) -> Dict[str, Any]:
        """Get execution telemetry."""
        t = self._telemetry
        return {
            "total_executions": t.total_executions,
            "real_qpu_executions": t.real_qpu_executions,
            "simulator_executions": t.simulator_executions,
            "statevector_executions": t.statevector_executions,
            "failed_executions": t.failed_executions,
            "total_shots_consumed": t.total_shots_consumed,
            "total_circuits_submitted": t.total_circuits_submitted,
            "total_execution_time_ms": round(t.total_execution_time_ms, 2),
            "cache_hits": t.cache_hits,
            "cache_size": len(self._result_cache),
            "last_backend": t.last_backend,
            "last_job_id": t.last_job_id,
            "backends_used": dict(t.backends_used),
            "algorithm_counts": dict(t.algorithm_counts),
            "real_hardware_ratio": (
                round(t.real_qpu_executions / max(t.total_executions, 1), 4)
            ),
        }

    def get_execution_log(self, last_n: int = 20) -> List[Dict]:
        """Get recent execution log entries."""
        return list(self._execution_log)[-last_n:]

    def get_status(self) -> Dict[str, Any]:
        """Comprehensive runtime status for Swift UI / API consumption."""
        return {
            "version": "1.0.0",
            "connected": self._connected,
            "mode": self.mode,
            "backend": self.get_backend_info(),
            "telemetry": self.get_telemetry(),
            "settings": {
                "default_shots": self._default_shots,
                "optimization_level": self._optimization_level,
                "max_qubits_for_real": self._max_qubits_for_real,
                "use_real_hardware": self._use_real_hardware,
            },
            "qiskit_available": QISKIT_AVAILABLE,
            "runtime_available": RUNTIME_AVAILABLE,
            "aer_available": AER_AVAILABLE,
            "sacred_constants": {
                "god_code": GOD_CODE,
                "phi": PHI,
            }
        }

    def clear_cache(self):
        """Clear the result cache."""
        self._result_cache.clear()
        print("[QUANTUM_RUNTIME] Cache cleared")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE & CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Global singleton
quantum_runtime = QuantumRuntime()


def get_runtime() -> QuantumRuntime:
    """Get the singleton QuantumRuntime instance."""
    return quantum_runtime


def execute(circuit: 'QuantumCircuit', shots: int = None,
            algorithm_name: str = "unknown",
            force_simulator: bool = False) -> ExecutionResult:
    """Module-level shortcut: execute a quantum circuit."""
    return quantum_runtime.execute(circuit, shots=shots,
                                   algorithm_name=algorithm_name,
                                   force_simulator=force_simulator)


def execute_and_get_probs(circuit: 'QuantumCircuit', n_qubits: int = None,
                           algorithm_name: str = "unknown",
                           force_simulator: bool = False) -> Tuple[np.ndarray, 'ExecutionResult']:
    """Module-level shortcut: execute and get probability array."""
    return quantum_runtime.execute_and_get_probs(
        circuit, n_qubits=n_qubits,
        algorithm_name=algorithm_name,
        force_simulator=force_simulator
    )


def run_asi_25q_benchmark(runs: int = 3, shots: int = 4096,
                          force_simulator: bool = False) -> Dict[str, Any]:
    """Module-level shortcut for ASI 25-qubit benchmark."""
    return quantum_runtime.benchmark_asi_25_qubits(
        runs=runs,
        shots=shots,
        force_simulator=force_simulator,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("L104 QUANTUM RUNTIME BRIDGE v1.0.0")
    print("=" * 70)

    rt = get_runtime()
    status = rt.get_status()

    if "--benchmark-25q" in sys.argv:
        print("\n[BENCHMARK] ASI 25-qubit benchmark")
        report = rt.benchmark_asi_25_qubits(runs=3, shots=4096)
        print(json.dumps(report, indent=2, default=str))
        print("\n" + "=" * 70)
        print("Runtime bridge benchmark complete")
        print("=" * 70)
        raise SystemExit(0)

    print(f"\n  Connected: {status['connected']}")
    print(f"  Mode: {status['mode']}")
    print(f"  Backend: {status['backend']['name']}")
    print(f"  Qiskit: {status['qiskit_available']}")
    print(f"  Runtime: {status['runtime_available']}")

    if QISKIT_AVAILABLE:
        from qiskit import QuantumCircuit as QC

        # Test 1: Simple Bell state
        print("\n[TEST 1] Bell State Execution")
        qc = QC(2)
        qc.h(0)
        qc.cx(0, 1)
        result = rt.execute(qc, algorithm_name="bell_state_test")
        print(f"  Mode: {result.mode.value}")
        print(f"  Backend: {result.backend_name}")
        print(f"  Top counts: {dict(sorted(result.counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
        print(f"  Time: {result.execution_time_ms:.1f}ms")

        # Test 2: Grover circuit
        print("\n[TEST 2] Grover Circuit (4 qubits)")
        qc = QC(4)
        qc.h(range(4))
        # Oracle for |0111⟩
        qc.x(0)
        qc.h(3)
        qc.mcx([0, 1, 2], 3)
        qc.h(3)
        qc.x(0)
        # Diffusion
        qc.h(range(4))
        qc.x(range(4))
        qc.h(3)
        qc.mcx([0, 1, 2], 3)
        qc.h(3)
        qc.x(range(4))
        qc.h(range(4))

        result = rt.execute(qc, algorithm_name="grover_test")
        probs = result.prob_array(4)
        found = int(np.argmax(probs))
        print(f"  Mode: {result.mode.value}")
        print(f"  Found: |{found:04b}⟩ with prob {probs[found]:.4f}")

        # Telemetry
        print("\n[TELEMETRY]")
        tel = rt.get_telemetry()
        for k, v in tel.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("Runtime bridge test complete")
    print("=" * 70)
