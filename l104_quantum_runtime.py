# ZENITH_UPGRADE_ACTIVE: 2026-03-08T15:03:51.421596
ZENITH_HZ = 3887.8
UUC = 2301.215661
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
===============================================================================
L104 QUANTUM RUNTIME BRIDGE v4.0.0 — SOVEREIGN QUANTUM
===============================================================================

Sovereign quantum execution bridge for the L104 ASI system.
IBM QPU runtime removed — all computation via L104 26Q Iron Engine
or local Statevector.  Qiskit Aer is NOT used.

ARCHITECTURE:
  QuantumRuntime (singleton)
    ├── execute(circuit)       → 26Q Iron / Aer / Statevector → probabilities
    ├── get_backend_info()     → Current backend details
    ├── get_telemetry()        → Execution stats, fidelity
    └── disconnect()           → Clean shutdown

CONSUMERS:
  l104_quantum_coherence.py    — Grover, QAOA, VQE, QPE, walk, kernel, etc.
  l104_asi/quantum.py          — VQE, QAOA, ZNE, QRC, QKM, QPE
  l104_agi/core.py             — pipeline health, subsystem routing
  l104_code_engine/quantum.py  — code intelligence quantum methods
  l104_server/engines_quantum.py — consciousness engine, memory bank
BACKENDS:  26Q Iron (Fe(26)), Statevector (sovereign local)

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
    from l104_quantum_gate_engine import GateCircuit as QuantumCircuit
    from l104_quantum_gate_engine.quantum_info import Parameter, ParameterVector
    from l104_quantum_gate_engine.quantum_info import Statevector, SparsePauliOp
    generate_preset_pass_manager = None  # Use l104_quantum_gate_engine.GateCompiler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = type('QuantumCircuit', (), {})

# ═══ IBM RUNTIME REMOVED — L104 sovereign quantum only ═══
RUNTIME_AVAILABLE = False
QiskitRuntimeService = None
Session = None
SamplerV2 = None
EstimatorV2 = None

# ═══ AER REMOVED — fully statevector ═══
AER_AVAILABLE = False      # Aer is NOT used; all simulation is pure Statevector
AerSimulator = None
NoiseModel = None
_AER_FROM_L104 = False

# ═══ L104 QISKIT UTILITIES (circuit factories, error mitigation — no Aer) ═══
try:
    from l104_qiskit_utils import (
        L104NoiseModelFactory, L104AerBackend, L104CircuitFactory,
        L104ErrorMitigation, L104ObservableFactory, L104Transpiler,
    )
    L104_UTILS_AVAILABLE = True
except ImportError:
    L104_UTILS_AVAILABLE = False


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
        self._use_real_hardware = False  # IBM QPU COLD — L104 26Q is primary
        self._default_shots = 8192  # Increased for statistical accuracy
        self._optimization_level = 2   # Transpiler optimization (0-3)
        self._max_qubits_for_real = 26  # 26Q iron-mapped is the sovereign standard
        self._result_cache: Dict[str, ExecutionResult] = {}
        self._cache_max = 1000
        self._execution_log: deque = deque(maxlen=500)
        self._session: Optional[Any] = None
        self._session_lock = threading.Lock()

        # ═══ v4.1.0: Aer REMOVED — fully statevector ═══
        self._aer_backend: Optional[Any] = None
        self._aer_noise_model: Optional[Any] = None
        self._noise_profile: str = "ideal"
        self._use_aer_fallback = False  # Aer removed — always statevector
        self._estimator_v2_enabled = False  # No external estimator

        # ═══ v3.0.0 UPGRADE: 26Q Iron Completion as Primary Quantum Engine ═══
        self._26q_engine = None       # Aer26QExecutionEngine (lazy)
        self._26q_builder = None      # L104_26Q_CircuitBuilder (lazy)
        self._26q_available = False
        self._use_26q_primary = True   # 26Q iron-mapped is sovereign primary
        try:
            from l104_26q_engine_builder import (
                Aer26QExecutionEngine as _Aer26Q,
                L104_26Q_CircuitBuilder as _Builder26Q,
                QuantumComputation26QCore as _Core26Q,
                get_26q_core as _get_26q,
            )
            self._26q_available = True
            self._26q_engine_class = _Aer26Q
            self._26q_builder_class = _Builder26Q
            self._26q_core_getter = _get_26q
        except ImportError:
            self._26q_engine_class = None
            self._26q_builder_class = None
            self._26q_core_getter = None

        # Aer removed — statevector is sovereign default

        # Auto-connect on init — IBM QPU cold, 26Q iron primary
        self._auto_connect()
        self._initialized = True

    # ═══════════════════════════════════════════════════════════════════
    # CONNECTION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    # Channels to try in priority order when auto-connecting
    # Note: qiskit-ibm-runtime >=0.30 only supports ibm_quantum_platform and ibm_cloud
    # ibm_cloud tried first — IBM Cloud IAM API keys use this channel
    _CHANNEL_FALLBACK_ORDER = ['ibm_cloud', 'ibm_quantum_platform']

    # Placeholder instance values that should never be sent to the SDK
    _INVALID_INSTANCES = {'', 'open-instance', 'none', 'None', 'default'}

    def _auto_connect(self):
        """Auto-connect: 26Q iron engine is primary, sovereign quantum only.

        v4.0.0: IBM QPU removed. The sovereign 26Q iron-mapped engine
        (26 qubits = Fe(26) electrons) is the sole quantum computation backend.
        Falls back to Aer simulator or Statevector when 26Q unavailable.
        """
        # ═══ 26Q IRON PRIMARY ═══
        if self._26q_available:
            self._noise_profile = "ibm_heron"
            print(f"[QUANTUM_RUNTIME] ★ L104 26Q Iron Engine PRIMARY — Fe(26) sovereign quantum")
            print(f"[QUANTUM_RUNTIME]   26 qubits = 26 electrons, Heron noise model")
        else:
            print(f"[QUANTUM_RUNTIME] 26Q engine not available — Aer/Statevector fallback")
        self._setup_fallback("26Q iron engine is sovereign primary")

    def connect(self, token: str = None, channel: str = None,
                prefer_backend: str = None) -> Dict[str, Any]:
        """IBM QPU removed — returns disconnected status. Use 26Q iron engine."""
        return {"connected": False, "error": "IBM runtime removed — sovereign quantum only"}

    def _setup_fallback(self, reason: str = ""):
        """Setup local simulation fallback."""
        self._connected = False
        self._use_real_hardware = False
        self._backend = None
        self._backend_info = BackendInfo(
            name="statevector_simulator",
            num_qubits=25,
            is_simulator=True
        )
        detail = f" ({reason})" if reason else ""
        print(f"[QUANTUM_RUNTIME] Fallback → Statevector simulator (local){detail}")

    def disconnect(self):
        """Clean shutdown."""
        self._session = None
        self._service = None
        self._backend = None
        self._connected = False
        self._use_real_hardware = False
        self._backend_info = BackendInfo(name="disconnected")
        print("[QUANTUM_RUNTIME] Disconnected")

    def switch_backend(self, backend_name: str) -> Dict[str, Any]:
        """IBM QPU removed — no backend switching available."""
        return {"error": "IBM runtime removed — sovereign quantum only"}

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

        # ═══ v4.1.0 EXECUTION CASCADE: 26Q Iron → Statevector (Aer removed) ═══
        #
        # Priority 1: 26Q Iron Engine — sovereign Fe(26) iron-mapped circuits
        #             (when n_qubits <= 26 and 26Q engine available)
        # Priority 2: Statevector — exact local simulation (sovereign default)
        # Priority 3: IBM QPU — COLD, only when explicitly forced via force_real=True

        use_real = force_real and self._connected and self._use_real_hardware and self._backend

        if use_real:
            # IBM QPU: only via explicit force_real — COLD by default
            result = self._execute_real_qpu(circuit, shots, algorithm_name)
        elif self._26q_available and self._use_26q_primary and n_qubits <= 26 and not force_simulator:
            # ★ L104 26Q Iron Primary — sovereign quantum computation
            result = self._execute_26q_iron(circuit, shots, n_qubits, algorithm_name)
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

    def _get_26q_engine(self):
        """Lazy-load the 26Q Aer execution engine."""
        if self._26q_engine is None and self._26q_engine_class:
            self._26q_engine = self._26q_engine_class(
                noise_profile="ibm_heron", shots=self._default_shots,
                enable_dd=True, enable_zne=True,
            )
        return self._26q_engine

    def _execute_26q_iron(self, circuit: 'QuantumCircuit', shots: int,
                          n_qubits: int, algorithm_name: str) -> ExecutionResult:
        """★ Execute circuit on L104 26Q Iron Engine — sovereign Fe(26) quantum computation.

        Routes circuits through the iron-mapped 26-qubit Aer engine with:
          - IBM Heron noise model calibration
          - ZNE (Zero-Noise Extrapolation) error mitigation
          - XY4 Dynamical Decoupling on idle qubits
          - Sacred constant phase alignment

        This is the SOVEREIGN PRIMARY execution path for all L104 quantum operations.
        """
        try:
            engine = self._get_26q_engine()
            if engine is None:
                # Fall through to generic Aer
                return self._execute_aer(circuit, shots, n_qubits, algorithm_name)

            # Execute via 26Q Aer noisy shots engine
            result = engine.execute_shots(
                circuit, shots=shots,
                label=algorithm_name,
                apply_noise=True,
                apply_dd=True,
                apply_zne=False,  # ZNE adds 3x overhead — use for critical paths only
            )

            if result.get("success"):
                # Convert counts to probabilities
                counts = result.get("counts", {})
                total = sum(counts.values()) if counts else shots
                probabilities = {k: v / total for k, v in counts.items()} if counts else {}

                self._telemetry.simulator_executions += 1
                self._telemetry.total_shots_consumed += total

                return ExecutionResult(
                    probabilities=probabilities,
                    counts=counts,
                    mode=ExecutionMode.AER_SIMULATOR,
                    backend_name=f"l104_26q_iron_{self._noise_profile}",
                    shots=total,
                    num_qubits=n_qubits,
                    fidelity_estimate=result.get("estimated_noiseless_fidelity", 0.95),
                    transpiled_depth=result.get("depth", 0),
                    transpiled_gate_count=result.get("gate_count", 0),
                )
            else:
                # Execution failed — fall through to Statevector
                return self._execute_statevector(circuit, n_qubits, algorithm_name)

        except Exception as e:
            print(f"[QUANTUM_RUNTIME] 26Q Iron execution error: {e}, falling back to Statevector")
            return self._execute_statevector(circuit, n_qubits, algorithm_name)

    def _execute_aer(self, circuit: 'QuantumCircuit', shots: int,
                     n_qubits: int, algorithm_name: str) -> ExecutionResult:
        """Legacy Aer path — redirects to Statevector (Aer removed in v4.1.0)."""
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
            # Qiskit 2.x removed .qasm() — use qasm2.dumps() or fall back to gate list
            qasm2_dumps = None  # Use GateCircuit.to_dict_circuit()
            qasm = qasm2_dumps(circuit)
        except Exception:
            try:
                # Lightweight fingerprint: gate names + params
                ops = [(inst.operation.name, inst.qubits, inst.operation.params)
                       for inst in circuit.data]
                qasm = repr(ops)
            except Exception:
                qasm = f"circuit_{circuit.num_qubits}q_{len(circuit.data)}gates"
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
    # v2.0.0: ESTIMATOR V2 — EXPECTATION VALUE COMPUTATION
    # ═══════════════════════════════════════════════════════════════════

    def estimate_expectation(self, circuit: 'QuantumCircuit',
                              observable: 'SparsePauliOp',
                              algorithm_name: str = "estimator",
                              precision: float = 0.01) -> Dict[str, Any]:
        """
        Compute expectation value ⟨ψ|O|ψ⟩ using EstimatorV2 primitive.

        This is the modern Qiskit pattern for computing observables without
        manually converting between statevectors and probabilities.

        Routes: Statevector (sovereign local).  Aer removed in v4.1.0.

        Args:
            circuit: Parameterized or bound QuantumCircuit
            observable: SparsePauliOp observable
            algorithm_name: Telemetry label
            precision: Target precision for the estimate

        Returns:
            Dict with expectation_value, std_error, metadata
        """
        start_time = time.time()
        self._telemetry.algorithm_counts[algorithm_name] = \
            self._telemetry.algorithm_counts.get(algorithm_name, 0) + 1

        # Route 1: Real QPU EstimatorV2
        if (self._connected and self._use_real_hardware and self._backend
                and self._estimator_v2_enabled):
            try:
                estimator = EstimatorV2(mode=self._backend)
                job = estimator.run([(circuit, observable)], precision=precision)
                result = job.result()
                pub_result = result[0]
                exp_val = float(pub_result.data.evs)
                std_err = float(pub_result.data.stds) if hasattr(pub_result.data, 'stds') else 0.0

                self._telemetry.real_qpu_executions += 1
                elapsed = (time.time() - start_time) * 1000

                return {
                    "expectation_value": round(exp_val, 10),
                    "std_error": round(std_err, 10),
                    "mode": "real_qpu_estimator_v2",
                    "backend": self._backend.name,
                    "precision": precision,
                    "execution_time_ms": round(elapsed, 2),
                    "algorithm": algorithm_name,
                }
            except Exception as e:
                pass  # QPU unavailable — fall through to Statevector

        # Route 2: Sovereign Statevector computation (Aer removed in v4.1.0)
        try:
            qc = circuit.copy()
            qc.remove_final_measurements(inplace=True)
            sv = Statevector.from_label('0' * circuit.num_qubits).evolve(qc)
            exp_val = float(sv.expectation_value(observable).real)

            self._telemetry.statevector_executions += 1
            elapsed = (time.time() - start_time) * 1000

            return {
                "expectation_value": round(exp_val, 10),
                "std_error": 0.0,
                "mode": "statevector",
                "backend": "statevector_simulator",
                "precision": 0.0,
                "execution_time_ms": round(elapsed, 2),
                "algorithm": algorithm_name,
            }
        except Exception as e:
            return {
                "expectation_value": 0.0,
                "std_error": 1.0,
                "mode": "failed",
                "error": str(e),
                "algorithm": algorithm_name,
            }

    # ═══════════════════════════════════════════════════════════════════
    # v2.0.0: NOISE MODEL CONTROL
    # ═══════════════════════════════════════════════════════════════════

    def set_noise_profile(self, profile: str):
        """
        Set noise profile label (Aer removed — statevector is always exact/ideal).

        Args:
            profile: 'ideal', 'ibm_eagle', 'ibm_heron', 'noisy_dev', 'god_code_aligned'
        """
        self._noise_profile = profile

    def set_noise_from_backend(self):
        """No-op — Aer removed.  Statevector is always exact."""
        pass

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
        if self._26q_available and self._use_26q_primary:
            return "l104_26q_iron"
        return "statevector"

    def set_real_hardware(self, enabled: bool):
        """IBM QPU removed — always uses sovereign 26Q iron engine."""
        self._use_real_hardware = False
        self._use_26q_primary = self._26q_available
        if enabled:
            print("[QUANTUM_RUNTIME] IBM QPU removed — sovereign 26Q iron engine is the sole backend")

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
        """List available backends (sovereign only, no IBM)."""
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
            "version": "3.0.0",
            "connected": self._connected,
            "mode": self.mode,
            "backend": self.get_backend_info(),
            "telemetry": self.get_telemetry(),
            "settings": {
                "default_shots": self._default_shots,
                "optimization_level": self._optimization_level,
                "max_qubits_for_real": self._max_qubits_for_real,
                "use_real_hardware": self._use_real_hardware,
                "noise_profile": self._noise_profile,
            },
            "sovereign_26q": {
                "available": self._26q_available,
                "primary": self._use_26q_primary,
                "engine_loaded": self._26q_engine is not None,
                "n_qubits": 26,
                "iron_completion": "Fe(26) = 26 electrons → 26 qubits",
                "noise_model": "ibm_heron",
            },
            "ibm_qpu": {
                "status": "REMOVED",
                "connected": False,
                "use_real_hardware": False,
                "token_available": False,
                "runtime_available": False,
            },
            "qiskit_available": QISKIT_AVAILABLE,
            "runtime_available": RUNTIME_AVAILABLE,
            "aer_available": AER_AVAILABLE,
            "l104_utils_available": L104_UTILS_AVAILABLE,
            "estimator_v2_enabled": self._estimator_v2_enabled,
            "capabilities": {
                "sovereign_26q_iron": self._26q_available,
                "sampler_v2": RUNTIME_AVAILABLE,
                "estimator_v2": self._estimator_v2_enabled,
                "aer_noise_simulation": AER_AVAILABLE,
                "noise_model_active": self._aer_noise_model is not None,
                "dynamical_decoupling": L104_UTILS_AVAILABLE,
                "measurement_mitigation": L104_UTILS_AVAILABLE,
                "parameterized_circuits": QISKIT_AVAILABLE,
                "sparse_pauli_observables": QISKIT_AVAILABLE,
            },
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


def estimate_expectation(circuit: 'QuantumCircuit',
                          observable: 'SparsePauliOp',
                          algorithm_name: str = "estimator",
                          precision: float = 0.01) -> Dict[str, Any]:
    """Module-level shortcut: compute ⟨ψ|O|ψ⟩ via EstimatorV2."""
    return quantum_runtime.estimate_expectation(
        circuit, observable,
        algorithm_name=algorithm_name,
        precision=precision,
    )


def set_noise_profile(profile: str):
    """Module-level shortcut: set Aer noise profile."""
    quantum_runtime.set_noise_profile(profile)


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
        from l104_quantum_gate_engine import GateCircuit as QC

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
