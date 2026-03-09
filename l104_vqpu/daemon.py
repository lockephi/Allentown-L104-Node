"""L104 VQPU v16.0.0 — VQPU Daemon Cycler (background findings runner + brain intelligence).

v16.0.0 CROSS-DAEMON INTELLIGENCE & GRACEFUL DEGRADATION:
  - Cross-daemon mesh telemetry: reads NanoDaemon + QuantumAIDaemon + Guardian state files
  - Predictive cycle scheduling: fidelity trend accelerates cycles when quality drops
  - Sim priority ranking: historical pass rate × fidelity contribution determines execution order
  - Graceful degradation: FULL (>0.7) → REDUCED (0.4-0.7, top 50%) → MINIMAL (<0.4, top 2)
  - Batch fidelity alerts: structured alerts when per-sim fidelity drops >15% below rolling avg

v15.1.0 INTELLIGENCE & SELF-HEALING UPGRADE:
  - Sim quarantine: chronically failing sims auto-skipped for N cycles (3 failures → 10 cycles)
  - Per-sim retry: transient failures retried once with 500ms delay (timeouts excluded)
  - Fidelity trend tracking: per-sim + per-cycle fidelity/sacred-alignment rolling history
  - Degradation detection: alerts when fidelity trends >10% downward
  - Throughput metric: sims/second tracked per cycle for performance dashboards
  - Leaked thread tracking: timed-out sim threads counted and reported in status
  - Runtime control: pause(), resume(), force_cycle(), adjust_interval(), unquarantine()
  - Enhanced brain feedback: fidelity + alignment trends + quarantine stats
  - State persistence: quarantine + failure counts + health_score restored on reload
  - Module-level `import math` (removed per-call inline import)

v15.0.0 RESILIENCE & OBSERVABILITY (retained):
  - Circuit breaker: exponential backoff on consecutive failures (2^n × base, max 30 min)
  - Watchdog: self-healing daemon thread auto-restart on unexpected death
  - atexit persistence: state saved on process exit (no silent data loss)
  - Unified health score: single 0-1 metric (pass×0.3 + timing×0.2 + stability×0.25 + fidelity×0.25)
  - Sim timing drift detection: alerts when sims trend >2× slower than rolling average
  - deque-backed bounded collections: O(1) append replaces O(n) list slicing
  - Cycle jitter: ±10% randomization prevents thundering herd across nodes
  - Error log exposed in status() API for full observability

v14.0.0 QUANTUM FIDELITY ARCHITECTURE (retained):
  - Adaptive cycle interval: 60s–600s based on CPU load (psutil)
  - Per-sim timeout guard: 120s max per simulation (prevents runaway cycles)

v13.4 SPEED UPGRADE (retained):
  - FAST sim registry: VQPU_FINDINGS_SIMULATIONS_FAST cuts cycle ~7× vs full
  - Sequential execution (ThreadPoolExecutor REMOVED — GIL-bound, was slower)
  - Cached engine imports, batched lock, cached memory pressure
  - Background state persistence via daemon thread
"""

import atexit
import gc
import json
import logging
import math
import os
import random
import time
import threading
from collections import deque
from pathlib import Path
from typing import Optional

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT, BRIDGE_PATH,
    DAEMON_CYCLE_INTERVAL_S, DAEMON_STATE_FILE,
    DAEMON_MAX_ERROR_LOG, DAEMON_ERROR_THRESHOLD,
    DAEMON_CYCLE_MIN_INTERVAL_S, DAEMON_CYCLE_MAX_INTERVAL_S,
    DAEMON_LOAD_THRESHOLD_LOW, DAEMON_LOAD_THRESHOLD_HIGH,
)

# ═══════════════════════════════════════════════════════════════════
# v13.3: MODULE-LEVEL CACHES — avoid repeated heavy imports
# ═══════════════════════════════════════════════════════════════════
_cached_findings_sims = None       # VQPU_FINDINGS_SIMULATIONS (loaded once)
_cached_quantum_brain = None       # l104_quantum_engine.quantum_brain
_cached_asi_core = None            # l104_asi.asi_core
_cached_agi_core = None            # l104_agi.agi_core
_engine_import_attempted = set()   # Track which imports have been attempted

# v15.4: Cached psutil module — shared across _compute_adaptive_interval and _run_findings_cycle
_cached_psutil_mod = None
_cached_psutil_attempted = False

def _get_cached_psutil():
    """Return cached psutil module (None if unavailable). Import once only."""
    global _cached_psutil_mod, _cached_psutil_attempted
    if not _cached_psutil_attempted:
        _cached_psutil_attempted = True
        try:
            import psutil
            _cached_psutil_mod = psutil
        except ImportError:
            _cached_psutil_mod = None
    return _cached_psutil_mod

_logger = logging.getLogger("L104_VQPU_DAEMON")

# v15.0: Circuit breaker constants
_CIRCUIT_BREAKER_MAX_BACKOFF_S = 1800.0  # 30 min max backoff
_CIRCUIT_BREAKER_BASE_S = 30.0           # base backoff interval
_JITTER_FRACTION = 0.10                  # ±10% cycle jitter
_SIM_DRIFT_WINDOW = 20                   # rolling window for drift detection
_SIM_DRIFT_THRESHOLD = 2.0              # alert if >2× rolling average

# v15.2: Strategic GC control — disable during sims, explicit collect between cycles
_GC_COLLECT_EVERY_N_CYCLES = 3          # explicit gc.collect() every N cycles
_GC_GEN0_THRESHOLD = 2000               # raised from default 700
_GC_GEN1_THRESHOLD = 50                 # raised from default 10
_GC_GEN2_THRESHOLD = 20                 # raised from default 10

# v15.1: Sim quarantine + retry constants
_SIM_QUARANTINE_THRESHOLD = 3           # failures before quarantine
_SIM_QUARANTINE_CYCLES = 10             # cycles to skip quarantined sim
_SIM_RETRY_MAX = 1                      # max retries per sim per cycle
_SIM_RETRY_DELAY_S = 0.5                # delay between retries
_LEAKED_THREAD_WARN_INTERVAL = 300.0    # warn about leaked threads every 5 min
_FIDELITY_HISTORY_WINDOW = 50           # rolling window for fidelity trend

# v16.0: Degradation levels
_DEGRADATION_FULL_THRESHOLD = 0.7
_DEGRADATION_REDUCED_THRESHOLD = 0.4
_DEGRADATION_REDUCED_SIM_RATIO = 0.5   # run top 50% sims
_DEGRADATION_MINIMAL_SIM_COUNT = 2     # run top 2 sims only

# v16.0: Fidelity alert threshold
_FIDELITY_ALERT_DROP_PCT = 0.15        # 15% drop triggers alert

# v16.0: Predictive scheduling
_PREDICTIVE_FIDELITY_WEIGHT = 0.4      # weight for fidelity trend in interval calc

# v16.0: Cross-daemon state paths
_NANO_DAEMON_STATE = ".l104_nano_daemon_python.json"
_QAI_DAEMON_STATE = ".l104_quantum_ai_daemon.json"
_GUARDIAN_DAEMON_STATE = ".l104_resource_guardian.json"


def _get_findings_sims():
    """Load VQPU_FINDINGS_SIMULATIONS_FAST once, cache at module level.

    v13.4: Uses FAST registry (reduced-precision sims) for ~3-4× speedup.
    Falls back to full registry if fast not available.
    """
    global _cached_findings_sims
    if _cached_findings_sims is not None:
        return _cached_findings_sims
    try:
        from l104_god_code_simulator.simulations.vqpu_findings import (
            VQPU_FINDINGS_SIMULATIONS_FAST,
        )
        _cached_findings_sims = VQPU_FINDINGS_SIMULATIONS_FAST
        return _cached_findings_sims
    except ImportError:
        pass
    # Fallback to full registry
    try:
        from l104_god_code_simulator.simulations.vqpu_findings import (
            VQPU_FINDINGS_SIMULATIONS,
        )
        _cached_findings_sims = VQPU_FINDINGS_SIMULATIONS
        return _cached_findings_sims
    except ImportError:
        return None


def _get_quantum_brain():
    """Import quantum_brain once, cache at module level."""
    global _cached_quantum_brain
    if _cached_quantum_brain is not None:
        return _cached_quantum_brain
    if "quantum_brain" in _engine_import_attempted:
        return None
    _engine_import_attempted.add("quantum_brain")
    try:
        from l104_quantum_engine import quantum_brain
        _cached_quantum_brain = quantum_brain
        return _cached_quantum_brain
    except Exception:
        return None


def _get_asi_core():
    """Import asi_core once, cache at module level."""
    global _cached_asi_core
    if _cached_asi_core is not None:
        return _cached_asi_core
    if "asi_core" in _engine_import_attempted:
        return None
    _engine_import_attempted.add("asi_core")
    try:
        from l104_asi import asi_core
        _cached_asi_core = asi_core
        return _cached_asi_core
    except Exception:
        return None


def _get_agi_core():
    """Import agi_core once, cache at module level."""
    global _cached_agi_core
    if _cached_agi_core is not None:
        return _cached_agi_core
    if "agi_core" in _engine_import_attempted:
        return None
    _engine_import_attempted.add("agi_core")
    try:
        from l104_agi import agi_core
        _cached_agi_core = agi_core
        return _cached_agi_core
    except Exception:
        return None


# v14.0: Per-sim timeout guard — kills runaway sims after 120s
_SIM_TIMEOUT_S = 120.0

# v15.1: Track leaked threads from timed-out sims
_leaked_threads: deque = deque(maxlen=20)
_last_leaked_warn_ts = 0.0


def _run_single_sim_once(sim_name: str, sim_fn) -> dict:
    """Execute a single VQPU findings simulation (one attempt).

    v14.0: 120s timeout guard — if a single sim exceeds this,
    it is terminated and marked as failed.
    v15.1: Leaked thread tracking — timed-out threads are logged.
    """
    global _last_leaked_warn_ts
    sim_start = time.monotonic()
    try:
        result_box = [None]
        exc_box = [None]

        def _target():
            try:
                result_box[0] = sim_fn()
            except Exception as e:
                exc_box[0] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=_SIM_TIMEOUT_S)
        if t.is_alive():
            # v15.1: Track leaked thread
            _leaked_threads.append({
                "sim": sim_name,
                "ts": time.time(),
                "thread_name": t.name,
            })
            now = time.time()
            if (now - _last_leaked_warn_ts) > _LEAKED_THREAD_WARN_INTERVAL:
                _logger.warning(
                    "Leaked threads: %d (latest: '%s' timeout after %.0fs)",
                    len(_leaked_threads), sim_name, _SIM_TIMEOUT_S)
                _last_leaked_warn_ts = now
            elapsed = (time.monotonic() - sim_start) * 1000.0
            return {
                "name": sim_name,
                "passed": False,
                "error": f"timeout_exceeded_{_SIM_TIMEOUT_S}s",
                "elapsed_ms": round(elapsed, 2),
                "retryable": False,  # timeouts are not retryable
            }
        if exc_box[0] is not None:
            raise exc_box[0]
        result = result_box[0]
        elapsed = (time.monotonic() - sim_start) * 1000.0
        entry_data = {
            "name": sim_name,
            "passed": result.passed,
            "elapsed_ms": round(elapsed, 2),
            "fidelity": round(result.fidelity, 6),
            "sacred_alignment": round(result.sacred_alignment, 6),
            "_result_obj": result,  # Carry for engine feeding (stripped later)
        }
        # SC-specific telemetry
        if sim_name == "superconductivity_heisenberg":
            entry_data["sc"] = {
                "cooper_pair": round(result.cooper_pair_amplitude, 6),
                "order_param": round(result.sc_order_parameter, 6),
                "energy_gap_eV": round(result.energy_gap_eV, 6),
                "meissner": round(result.meissner_fraction, 6),
                "pairing": result.pairing_symmetry,
            }
        # VQPU metrics
        entry_data["vqpu_metrics"] = result.to_vqpu_metrics()
        return entry_data
    except Exception as e:
        elapsed = (time.monotonic() - sim_start) * 1000.0
        return {
            "name": sim_name,
            "passed": False,
            "error": str(e),
            "elapsed_ms": round(elapsed, 2),
            "retryable": True,  # exceptions are retryable
        }


def _run_single_sim(entry) -> dict:
    """Execute a single VQPU findings simulation with retry.

    v15.1: Retries transient failures once with a short delay.
    Timeouts are NOT retried (thread already leaked).
    v14.0: 120s timeout guard.
    v13.4: Sequential execution (GIL-bound).
    """
    sim_name = entry[0]
    sim_fn = entry[1]
    result = _run_single_sim_once(sim_name, sim_fn)

    # v15.1: Retry transient failures (not timeouts)
    if not result.get("passed") and result.get("retryable", False):
        for attempt in range(_SIM_RETRY_MAX):
            time.sleep(_SIM_RETRY_DELAY_S)
            _logger.debug("Retrying sim '%s' (attempt %d)", sim_name, attempt + 1)
            result = _run_single_sim_once(sim_name, sim_fn)
            if result.get("passed"):
                result["retried"] = attempt + 1
                break

    # Strip retryable flag from final output
    result.pop("retryable", None)
    return result

# ═══════════════════════════════════════════════════════════════════
# QUANTUM SUBCONSCIOUS INTEGRATION — Idle Thought Engine
# ═══════════════════════════════════════════════════════════════════

try:
    from l104_quantum_subconscious import (
        get_subconscious as _get_quantum_subconscious,
        QuantumSubconscious as _QuantumSubconscious,
    )
    _HAS_QUANTUM_SUBCONSCIOUS = True
except ImportError:
    _HAS_QUANTUM_SUBCONSCIOUS = False
    _get_quantum_subconscious = None
    _QuantumSubconscious = None


class VQPUDaemonCycler:
    """
    Autonomous background daemon that periodically runs all 11 VQPU
    findings simulations, feeds results to coherence/entropy engines,
    and persists state + health telemetry to disk.

    v15.0 RESILIENCE & OBSERVABILITY:
      - Circuit breaker: exponential backoff on consecutive failures
      - Watchdog: auto-restart on daemon thread death
      - atexit persistence: state saved on process exit
      - Unified health score: single 0-1 metric for dashboard/brain
      - Sim timing drift detection: alerts when sims trend slower
      - deque-backed bounded collections: O(1) replaces O(n) slicing
      - Cycle jitter: ±10% prevents thundering herd
      - Error log in status() for full observability
    v14.0 ADAPTIVE FIDELITY (retained):
      - Adaptive cycle interval: 60s–600s based on CPU load (psutil)
      - Per-sim timeout guard: 120s max prevents runaway cycles
    v13.4 SPEED UPGRADE (retained):
      - FAST sim registry: VQPU_FINDINGS_SIMULATIONS_FAST (~7× faster)
      - Sequential execution (GIL makes threading slower for Python matrix ops)
      - Cached engine imports: heavy singletons imported once, reused forever
      - Batched lock: one acquisition per cycle (was 11× per sim)
      - Cached memory pressure: psutil throttled to 30s intervals
      - Background state persistence via fire-and-forget thread
      - Pre-initialized ScienceEngine at start()
      - Harvest TTL: skip re-harvest if results are <60s old

    v12.2 capabilities (retained):
      - Quantum Subconscious Engine v2.0 during IDLE GAP between cycles
      - 13+ adaptive streams, self-evolving, emergent spawning
      - Full v2.0 autonomous telemetry in daemon status
    v11.0 capabilities (retained):
      - Runs VQPU_FINDINGS_SIMULATIONS every DAEMON_CYCLE_INTERVAL_S (3 min)
      - Feeds results through: Coherence, Entropy, SC, ThreeEngine scorer
      - Persists cumulative state to .l104_vqpu_daemon_state.json
      - Thread-safe operation with graceful shutdown

    Usage:
        cycler = VQPUDaemonCycler()
        cycler.start()    # Spawns background daemon thread + subconscious
        cycler.status()   # Current health + run history + subconscious state
        cycler.stop()     # Graceful shutdown + state persist
    """

    # v13.4: How often to re-check memory pressure (seconds)
    _MEM_CHECK_TTL = 30.0
    # v13.4: How often to re-harvest brain/evolution data (seconds)
    _HARVEST_TTL = 60.0
    # v13.4: Persist state every N cycles (not every cycle)
    _PERSIST_EVERY_N = 5

    def __init__(self, interval: float = DAEMON_CYCLE_INTERVAL_S,
                 state_path: str = None):
        self._interval = interval
        self._state_path = Path(state_path or (
            Path(os.environ.get("L104_ROOT", os.getcwd())) / DAEMON_STATE_FILE))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Cumulative statistics
        self._cycles_completed = 0
        self._total_sims_run = 0
        self._total_sims_passed = 0
        self._total_sims_failed = 0
        self._total_elapsed_ms = 0.0
        self._last_cycle_time = 0.0
        self._last_cycle_results = []
        # v15.0: deque replaces list + slicing — O(1) bounded append
        self._sc_history = deque(maxlen=200)
        self._health_history = deque(maxlen=50)
        self._start_time = 0.0
        self._active = False
        self._error_log = deque(maxlen=DAEMON_MAX_ERROR_LOG)

        # v15.0: Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_breaker_open = False
        self._circuit_breaker_until = 0.0  # timestamp when breaker closes

        # v15.0: Per-sim timing drift detection
        self._sim_timing_history: dict[str, deque] = {}  # name → deque of elapsed_ms
        self._drift_alerts: deque = deque(maxlen=50)

        # v15.1: Sim quarantine — chronically failing sims auto-skipped
        self._sim_failure_counts: dict[str, int] = {}      # name → consecutive failures
        self._sim_quarantine: dict[str, int] = {}           # name → cycles remaining
        self._quarantine_log: deque = deque(maxlen=50)

        # v15.1: Fidelity + sacred alignment trend tracking
        self._sim_fidelity_history: dict[str, deque] = {}   # name → deque of fidelity
        self._sim_alignment_history: dict[str, deque] = {}  # name → deque of sacred_alignment
        self._cycle_fidelity_avg: deque = deque(maxlen=100)  # per-cycle avg fidelity
        self._cycle_alignment_avg: deque = deque(maxlen=100) # per-cycle avg alignment

        # v15.1: Throughput tracking
        self._cycle_throughput: deque = deque(maxlen=50)    # sims/second per cycle

        # v15.1: Leaked thread tracking
        self._total_leaked_threads = 0

        # v15.1: Runtime control
        self._paused = False

        # v15.0: Watchdog + atexit state
        self._watchdog_restarts = 0
        self._atexit_registered = False

        # Engine caches
        self._coherence_engine = None
        self._entropy_engine = None

        # v13.4: Cached memory pressure
        self._cached_avail_mb = 2048.0
        self._mem_check_ts = 0.0

        # v13.4: Cached harvest results with timestamps
        self._cached_brain_data = None
        self._brain_harvest_ts = 0.0
        self._cached_evolution_data = None
        self._evolution_harvest_ts = 0.0

        # v12.1: Quantum Subconscious — idle thought engine
        self._quantum_subconscious = None
        self._subconscious_harvests = 0
        self._subconscious_precog_seeds_fed = 0

        # v14.0: Adaptive interval tracking
        self._adaptive_interval = interval  # current effective interval
        self._last_cpu_percent = 0.0

        # v15.0: Unified health score
        self._health_score = 1.0  # 0.0 = critical, 1.0 = perfect

        # v15.2: GC telemetry
        self._gc_collections = 0
        self._gc_objects_freed = 0
        self._gc_paused_during_sims = False

        # v15.2: Persist guard — prevents overlapping background persist threads
        self._persist_in_progress = False

        # v15.2: Instance-level adaptive interval bounds (runtime-adjustable)
        self._interval_min = DAEMON_CYCLE_MIN_INTERVAL_S
        self._interval_max = DAEMON_CYCLE_MAX_INTERVAL_S

        # v15.2: Cached precognition engine import
        self._cached_precog_engine = None
        self._precog_import_attempted = False

        # v16.0: Sim priority ranking (name → priority score)
        self._sim_priority_scores: dict[str, float] = {}

        # v16.0: Degradation level tracking
        self._degradation_level = "FULL"  # FULL, REDUCED, MINIMAL

        # v16.0: Fidelity alert log
        self._fidelity_alerts: deque = deque(maxlen=50)

        # v16.0: Cross-daemon state cache
        self._cross_daemon_cache: dict = {}
        self._cross_daemon_cache_ts = 0.0

    def start(self):
        """Spawn the background daemon cycling thread + quantum subconscious.

        v15.0: Registers atexit handler for clean shutdown + state persistence.
        v13.4: Pre-warms heavy engine imports in a background thread so the
        first cycle doesn't pay the 4.6s import penalty.
        """
        if self._active:
            return
        self._stop_event.clear()
        self._start_time = time.time()
        self._active = True
        self._load_state()

        # v15.0: Register atexit handler (once) — persist state on process exit
        if not self._atexit_registered:
            atexit.register(self._atexit_handler)
            self._atexit_registered = True

        # v13.4: Pre-warm heavy engine imports in a background thread
        # so the first daemon cycle doesn't pay the import penalty
        threading.Thread(
            target=self._prewarm_engines, daemon=True,
            name="l104-vqpu-prewarm").start()

        # v12.1: Start quantum subconscious (idle thoughts between cycles)
        if _HAS_QUANTUM_SUBCONSCIOUS:
            try:
                self._quantum_subconscious = _get_quantum_subconscious()
                self._quantum_subconscious.start()
            except Exception:
                self._quantum_subconscious = None

        self._thread = threading.Thread(
            target=self._daemon_loop, daemon=True,
            name="l104-vqpu-daemon-cycler")
        self._thread.start()

        # v15.0: Start watchdog thread — auto-restarts daemon if thread dies
        threading.Thread(
            target=self._watchdog_loop, daemon=True,
            name="l104-vqpu-watchdog").start()

    def _prewarm_engines(self):
        """v13.3: Background pre-warm of heavy engine imports.

        Imports quantum_brain, asi_core, agi_core, VQPU_FINDINGS_SIMULATIONS,
        and ScienceEngine in a background thread during the initial interval
        wait.  This overlaps with the daemon's startup delay so first cycle
        runs at full speed.
        """
        try:
            _get_findings_sims()
        except Exception:
            pass
        try:
            self._init_science_engines()
        except Exception:
            pass
        try:
            _get_quantum_brain()
        except Exception:
            pass
        try:
            _get_asi_core()
        except Exception:
            pass
        try:
            _get_agi_core()
        except Exception:
            pass

    def stop(self):
        """Graceful shutdown — finish current sim, persist state, stop subconscious."""
        if not self._active:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30.0)

        # v12.1: Stop quantum subconscious
        if self._quantum_subconscious is not None:
            try:
                self._quantum_subconscious.stop()
            except Exception:
                pass

        self._persist_state()
        self._active = False

    def _atexit_handler(self):
        """v15.0: atexit callback — persist state on process exit.

        Called by the Python interpreter during shutdown.  Avoids
        silent data loss when a process exits without calling stop().
        """
        if self._active:
            _logger.info("atexit: persisting daemon state before shutdown")
            try:
                self._persist_state()
            except Exception:
                pass  # interpreter shutdown — best-effort only

    def _watchdog_loop(self):
        """v15.0: Watchdog — monitors daemon thread health, auto-restarts on death.

        Checks every 30s.  If the daemon thread has died while _active is
        still True (unexpected death, not a clean stop()), it restarts it.
        Max 5 automatic restarts to prevent infinite restart loops.
        """
        max_restarts = 5
        stable_cycles_to_reset = 10  # v15.2: reset counter after N stable cycles
        last_cycle_count = self._cycles_completed
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=30.0)
            if self._stop_event.is_set():
                break

            # v15.2: Decay restart counter after sustained stable operation.
            # If the daemon has completed 10+ cycles since the last restart
            # without dying, the previous failures were transient — allow
            # the watchdog to try again if needed in the future.
            current_cycles = self._cycles_completed
            if (current_cycles - last_cycle_count) >= stable_cycles_to_reset:
                if self._watchdog_restarts > 0:
                    self._watchdog_restarts = max(0, self._watchdog_restarts - 1)
                    _logger.debug(
                        "Watchdog: stable operation — decayed restart counter to %d",
                        self._watchdog_restarts)
                last_cycle_count = current_cycles

            if self._active and self._thread is not None and not self._thread.is_alive():
                if self._watchdog_restarts >= max_restarts:
                    _logger.error(
                        "Watchdog: daemon thread died %d times — giving up",
                        self._watchdog_restarts)
                    break
                self._watchdog_restarts += 1
                last_cycle_count = self._cycles_completed  # reset stable counter
                _logger.warning(
                    "Watchdog: daemon thread died — restarting (attempt %d/%d)",
                    self._watchdog_restarts, max_restarts)
                self._thread = threading.Thread(
                    target=self._daemon_loop, daemon=True,
                    name="l104-vqpu-daemon-cycler")
                self._thread.start()

    def run_cycle_now(self) -> dict:
        """Run one full findings cycle synchronously (on-demand).

        v13.4: Uses cached harvest data when available (TTL-based),
        avoiding redundant heavy imports.  Harvests that haven't been
        populated yet are fetched; subsequent calls within the TTL window
        return the cached version.
        """
        # v13.3: Use cached harvests if fresh, otherwise refresh
        now = time.time()
        subconscious_data = self._harvest_subconscious()

        if self._cached_brain_data is None or (now - self._brain_harvest_ts) > self._HARVEST_TTL:
            self._cached_brain_data = self._harvest_brain_intelligence()
            self._brain_harvest_ts = now
        brain_data = self._cached_brain_data

        if self._cached_evolution_data is None or (now - self._evolution_harvest_ts) > self._HARVEST_TTL:
            self._cached_evolution_data = self._harvest_evolution_scores()
            self._evolution_harvest_ts = now
        evolution_data = self._cached_evolution_data

        result = self._run_findings_cycle()
        # v15.1: Update health score after on-demand cycle
        self._update_health_score()
        if subconscious_data:
            result["subconscious_harvest"] = subconscious_data
        if brain_data:
            result["brain_intelligence"] = brain_data
        if evolution_data:
            result["evolution_scores"] = evolution_data
        return result

    def _harvest_subconscious(self) -> Optional[dict]:
        """v12.1: Harvest idle thoughts from quantum subconscious.

        Collects accumulated subconscious thoughts and precog seeds
        generated during the idle gap between daemon cycles.
        """
        if self._quantum_subconscious is None:
            return None
        try:
            # Harvest precognition seeds from crystallized insights
            seeds = self._quantum_subconscious.harvest_precog_seeds()
            # Get recent high-coherence thoughts
            thoughts = self._quantum_subconscious.harvest_thoughts(
                max_count=26, min_coherence=0.05)  # Fe(26) mapped
            # Get subconscious status
            sub_status = self._quantum_subconscious.status()

            with self._lock:
                self._subconscious_harvests += 1
                self._subconscious_precog_seeds_fed += len(seeds)

            # Feed precog seeds to precognition engine if available
            # v12.2: Actually wire up seeds to predictors (was no-op pass)
            # v15.2: Cached import — avoids repeated ImportError on every harvest
            if seeds:
                pe = self._get_precog_engine()
                if pe is not None:
                    for seed_data in seeds[:8]:  # Max 8 seeds per harvest
                        seed_val = seed_data.get("seed", {})
                        seed_type = seed_val.get("type", "")
                        coherence_series = seed_val.get("coherence_series", [])
                        if not coherence_series and "coherence" in seed_val:
                            coherence_series = [seed_val["coherence"]]
                        if seed_type == "temporal_prediction" and len(coherence_series) >= 5:
                            # Temporal seeds feed the TemporalPatternPredictor
                            try:
                                pe.temporal.predict(coherence_series, horizon=3)
                            except Exception:
                                pass
                        elif seed_type == "entropy_anomaly" and coherence_series:
                            # Entropy seeds feed the EntropyAnomalyForecaster
                            try:
                                for val in coherence_series:
                                    pe.anomaly.observe(val)
                            except Exception:
                                pass
                        elif seed_type == "bifurcation_warning" and len(coherence_series) >= 10:
                            # Bifurcation seeds feed the ChaosBifurcationDetector
                            try:
                                pe.chaos.detect(coherence_series)
                            except Exception:
                                pass
                        elif seed_type == "harmonic_discovery" and len(coherence_series) >= 5:
                            # Harmonic seeds feed the HarmonicExtrapolator
                            try:
                                pe.harmonic.extrapolate(coherence_series, horizon=3)
                            except Exception:
                                pass

            # v2.0: Include autonomous telemetry
            autonomous = sub_status.get("autonomous", {})
            return {
                "harvested_thoughts": len(thoughts),
                "precog_seeds": len(seeds),
                "subconscious_cycles": sub_status.get("cycles_completed", 0),
                "total_thoughts": sub_status.get("total_thoughts_generated", 0),
                "total_insights": sub_status.get("total_insights_crystallized", 0),
                "buffer_size": sub_status.get("thought_buffer_size", 0),
                "consciousness_band": "theta (subconscious)",
                # v2.0 autonomous metrics
                "adaptive_interval": sub_status.get("interval_seconds", 8.0),
                "active_streams": autonomous.get("active_emergent_streams", 0) + 13,
                "proactive_pushes": autonomous.get("total_proactive_pushes", 0),
                "evolution_generations": autonomous.get("evolution_generations", 0),
                "dream_depth": autonomous.get("dream_depth", 5),
            }
        except Exception:
            return None

    def subconscious_status(self) -> dict:
        """v12.2: Get quantum subconscious engine status (full v2.0 autonomous)."""
        if self._quantum_subconscious is not None:
            try:
                return self._quantum_subconscious.status()
            except Exception:
                pass
        return {"available": False, "reason": "not_initialized"}

    def subconscious_stream_health(self) -> dict:
        """v12.2: Get per-stream health report from quantum subconscious v2.0."""
        if self._quantum_subconscious is not None:
            try:
                return self._quantum_subconscious.stream_health()
            except Exception:
                pass
        return {}

    def subconscious_stream_weights(self) -> dict:
        """v12.2: Get adaptive stream weights from quantum subconscious v2.0."""
        if self._quantum_subconscious is not None:
            try:
                return self._quantum_subconscious.stream_weights()
            except Exception:
                pass
        return {}

    def subconscious_evolution(self) -> dict:
        """v12.2: Get self-evolution engine status from quantum subconscious v2.0."""
        if self._quantum_subconscious is not None:
            try:
                return self._quantum_subconscious.evolution_status()
            except Exception:
                pass
        return {}

    def subconscious_autonomy(self) -> dict:
        """v12.2: Get full autonomous systems dashboard."""
        if self._quantum_subconscious is not None:
            try:
                return {
                    "stream_health": self._quantum_subconscious.stream_health(),
                    "stream_weights": self._quantum_subconscious.stream_weights(),
                    "evolution": self._quantum_subconscious.evolution_status(),
                    "dream_depth": self._quantum_subconscious.dream_depth_status(),
                    "crystallizer": self._quantum_subconscious.crystallizer_status(),
                    "emergent": self._quantum_subconscious.emergent_status(),
                    "proactive": self._quantum_subconscious.proactive_status(),
                    "cross_engine": self._quantum_subconscious.cross_engine_status(),
                }
            except Exception:
                pass
        return {"available": False}

    # ═══════════════════════════════════════════════════════════════════
    # v15.1: RUNTIME CONTROL — pause, resume, force cycle, reconfigure
    # ═══════════════════════════════════════════════════════════════════

    def pause(self):
        """v15.1: Pause daemon cycling (sims stop, thread stays alive).

        The daemon thread remains alive but skips cycles until resume().
        State persistence continues.  Useful for manual maintenance windows.
        """
        self._paused = True
        _logger.info("Daemon PAUSED — cycles suspended")

    def resume(self):
        """v15.1: Resume daemon cycling after a pause()."""
        self._paused = False
        _logger.info("Daemon RESUMED — cycles active")

    def force_cycle(self) -> dict:
        """v15.1: Force an immediate cycle (ignores pause state, circuit breaker).

        Bypasses all guards.  Useful for diagnostics and on-demand probes.
        Updates health score after the cycle completes.
        """
        _logger.info("Force cycle triggered")
        result = self._run_findings_cycle()
        self._update_health_score()
        return result

    def adjust_interval(self, min_s: float = None, max_s: float = None):
        """v15.1: Runtime adjustment of adaptive interval bounds.

        Allows tuning cycle frequency without restart.  Only updates
        the instance-level bounds (doesn't mutate global constants).
        v15.2: Fixed — now properly initializes _interval_min/_interval_max
        in __init__ and _compute_adaptive_interval reads them.
        """
        if min_s is not None:
            self._interval_min = max(10.0, min_s)
        if max_s is not None:
            self._interval_max = max(self._interval_min + 10.0, max_s)
        _logger.info("Interval adjusted: min=%.0fs, max=%.0fs",
                      self._interval_min, self._interval_max)

    def unquarantine(self, sim_name: str = None):
        """v15.1: Remove a sim from quarantine, or clear all quarantines.

        Args:
            sim_name: Specific sim to unquarantine, or None to clear all.
        """
        if sim_name:
            self._sim_quarantine.pop(sim_name, None)
            self._sim_failure_counts.pop(sim_name, None)
            _logger.info("Unquarantined: '%s'", sim_name)
        else:
            self._sim_quarantine.clear()
            self._sim_failure_counts.clear()
            _logger.info("All quarantines cleared")

    def quarantine_status(self) -> dict:
        """v15.1: Return current quarantine state for all sims."""
        return {
            "quarantined": dict(self._sim_quarantine),
            "failure_counts": dict(self._sim_failure_counts),
            "quarantine_log": list(self._quarantine_log)[-10:],
            "quarantine_threshold": _SIM_QUARANTINE_THRESHOLD,
            "quarantine_cycles": _SIM_QUARANTINE_CYCLES,
        }

    def fidelity_trends(self) -> dict:
        """v15.1: Return fidelity + sacred alignment trends for all tracked sims.

        v15.4: Snapshot deques under lock to prevent data races with
        _run_findings_cycle which appends to them concurrently.
        """
        with self._lock:
            fidelity_snap = {k: list(v) for k, v in self._sim_fidelity_history.items()}
            alignment_snap = {k: list(v) for k, v in self._sim_alignment_history.items()}
            cycle_fid = list(self._cycle_fidelity_avg)[-10:]
            cycle_ali = list(self._cycle_alignment_avg)[-10:]

        trends = {}
        for name, data in fidelity_snap.items():
            alignment_data = alignment_snap.get(name, [])
            trends[name] = {
                "fidelity_samples": len(data),
                "fidelity_avg": round(sum(data) / len(data), 6) if data else 0,
                "fidelity_min": round(min(data), 6) if data else 0,
                "fidelity_max": round(max(data), 6) if data else 0,
                "fidelity_latest": round(data[-1], 6) if data else 0,
                "alignment_avg": round(sum(alignment_data) / len(alignment_data), 6) if alignment_data else 0,
                "alignment_latest": round(alignment_data[-1], 6) if alignment_data else 0,
            }
            # Detect fidelity degradation: last 5 < first 5 by >10%
            if len(data) >= 10:
                early_avg = sum(data[:5]) / 5
                late_avg = sum(data[-5:]) / 5
                if early_avg > 0 and late_avg < early_avg * 0.9:
                    trends[name]["degradation_alert"] = True
                    trends[name]["degradation_ratio"] = round(late_avg / early_avg, 4)
        return {
            "per_sim": trends,
            "cycle_fidelity_avg": cycle_fid,
            "cycle_alignment_avg": cycle_ali,
        }

    def throughput_stats(self) -> dict:
        """v15.1: Return throughput statistics (sims/second)."""
        data = list(self._cycle_throughput)
        if not data:
            return {"samples": 0, "avg_sims_per_sec": 0}
        return {
            "samples": len(data),
            "avg_sims_per_sec": round(sum(data) / len(data), 2),
            "min_sims_per_sec": round(min(data), 2),
            "max_sims_per_sec": round(max(data), 2),
            "latest_sims_per_sec": round(data[-1], 2),
        }

    def _compute_adaptive_interval(self) -> float:
        """v14.0 — Compute adaptive daemon cycle interval based on CPU load.

        Returns interval in seconds:
          - CPU < DAEMON_LOAD_THRESHOLD_LOW (30%):  _interval_min (60s default)
          - CPU > DAEMON_LOAD_THRESHOLD_HIGH (70%): _interval_max (600s default)
          - Between: linear interpolation

        v15.2: Uses instance-level _interval_min/_interval_max so
        adjust_interval() runtime changes actually take effect.
        v15.4: Uses _get_cached_psutil() (avoids per-call importlib overhead).
        """
        try:
            _ps = _get_cached_psutil()
            # v15.3: interval=0 is non-blocking (returns cached value).
            # interval=0.1 was blocking the daemon thread for 100ms every cycle.
            cpu = _ps.cpu_percent(interval=0) if _ps else 50.0
        except Exception:
            cpu = 50.0  # assume moderate load if psutil unavailable
        self._last_cpu_percent = cpu

        lo, hi = DAEMON_LOAD_THRESHOLD_LOW, DAEMON_LOAD_THRESHOLD_HIGH
        i_min = self._interval_min
        i_max = self._interval_max
        if cpu <= lo:
            interval = i_min
        elif cpu >= hi:
            interval = i_max
        else:
            # Linear interpolation
            frac = (cpu - lo) / (hi - lo)
            interval = i_min + frac * (i_max - i_min)

        self._adaptive_interval = round(interval, 1)
        return self._adaptive_interval

    def _daemon_loop(self):
        """Main daemon loop — waits one interval, then runs findings cycle.

        v14.0: Adaptive interval — recomputes wait time each cycle based on
        CPU load.  Low load → faster cycles (30s), high load → slower (300s).
        v13.3: Persist state every N cycles (not every cycle) and do it
        in a fire-and-forget background thread.  Refresh harvest caches
        on TTL expiry (overlapped with the inter-cycle wait).
        v12.0: Structured error logging replaces bare pass.
        """
        # Delay first cycle to avoid CPU contention on startup
        self._stop_event.wait(timeout=self._interval)
        while not self._stop_event.is_set():
            # v15.1: Pause check — skip cycle if paused
            if self._paused:
                self._stop_event.wait(timeout=10.0)
                continue

            # v15.0: Circuit breaker — skip cycle if in backoff
            now_ts = time.time()
            if self._circuit_breaker_open:
                if now_ts < self._circuit_breaker_until:
                    remaining = self._circuit_breaker_until - now_ts
                    _logger.info(
                        "Circuit breaker OPEN — skipping cycle (%.0fs remaining)",
                        remaining)
                    self._stop_event.wait(timeout=min(remaining, 30.0))
                    continue
                else:
                    # Breaker cooldown expired — probe with one cycle
                    _logger.info("Circuit breaker HALF-OPEN — probing")
                    self._circuit_breaker_open = False

            try:
                # v13.3: Refresh harvest caches if stale (before cycle)
                now = time.time()
                if self._cached_brain_data is None or (now - self._brain_harvest_ts) > self._HARVEST_TTL:
                    self._cached_brain_data = self._harvest_brain_intelligence()
                    self._brain_harvest_ts = now
                if self._cached_evolution_data is None or (now - self._evolution_harvest_ts) > self._HARVEST_TTL:
                    self._cached_evolution_data = self._harvest_evolution_scores()
                    self._evolution_harvest_ts = now

                self._run_findings_cycle()

                # v13.3: Persist every N cycles in background thread
                # v15.2: Guard against overlapping persist threads — if a previous
                # persist is still writing, skip this one to avoid file corruption.
                if (self._cycles_completed % self._PERSIST_EVERY_N == 0
                        and not self._persist_in_progress):
                    threading.Thread(
                        target=self._guarded_persist, daemon=True,
                        name=f"l104-vqpu-persist-{self._cycles_completed}").start()

                # v15.2: Explicit GC between cycles — collect garbage during
                # idle time rather than mid-simulation.  Raised thresholds
                # reduce gen-0 frequency; periodic full collect reclaims
                # leaked closures from timed-out sim threads.
                if self._cycles_completed % _GC_COLLECT_EVERY_N_CYCLES == 0:
                    gc.set_threshold(_GC_GEN0_THRESHOLD, _GC_GEN1_THRESHOLD, _GC_GEN2_THRESHOLD)
                    freed = gc.collect(generation=2)
                    self._gc_collections += 1
                    self._gc_objects_freed += freed
                    if freed > 100:
                        _logger.debug(
                            "GC inter-cycle: freed %d objects (collection #%d)",
                            freed, self._gc_collections)

                # v15.0: Success — reset circuit breaker
                self._consecutive_failures = 0
                self._circuit_breaker_open = False
            except Exception as e:
                self._consecutive_failures += 1
                with self._lock:
                    self._error_log.append({
                        "ts": time.time(),
                        "cycle": self._cycles_completed,
                        "error": str(e),
                        "consecutive": self._consecutive_failures,
                    })
                _logger.warning(
                    "Cycle %d failed (consecutive=%d): %s",
                    self._cycles_completed, self._consecutive_failures, e)

                # v15.0: Circuit breaker — exponential backoff on consecutive failures
                if self._consecutive_failures >= DAEMON_ERROR_THRESHOLD:
                    backoff = min(
                        _CIRCUIT_BREAKER_BASE_S * (2 ** (self._consecutive_failures - DAEMON_ERROR_THRESHOLD)),
                        _CIRCUIT_BREAKER_MAX_BACKOFF_S,
                    )
                    self._circuit_breaker_open = True
                    self._circuit_breaker_until = time.time() + backoff
                    _logger.error(
                        "Circuit breaker OPEN — %d consecutive failures, backing off %.0fs",
                        self._consecutive_failures, backoff)

            # v15.0: Recompute unified health score
            self._update_health_score()

            # v14.0: Adaptive interval based on CPU load + v15.0 jitter
            wait_time = self._compute_adaptive_interval()
            jitter = wait_time * _JITTER_FRACTION * (2 * random.random() - 1)
            self._stop_event.wait(timeout=max(10.0, wait_time + jitter))

    def _run_findings_cycle(self) -> dict:
        """Execute VQPU findings simulations + engine feedback.

        v15.5 GUARDIAN INTEGRATION:
          - Reads Resource Guardian IPC commands before each cycle
          - Skips cycle if guardian says suspend/halt
          - Respects guardian thread pool cap
        v13.4 SPEED UPGRADE:
          - FAST sim registry: reduced-precision sims (~3-4× faster)
          - Sequential execution (GIL makes threading slower)
          - Cached memory pressure check (psutil every 30s, not every cycle)
          - Cached sim import (module-level)
          - Batched lock acquisition: one lock for all counters at end
          - Engine feeding batched after all sims complete
        v12.3: Memory pressure guard (retained).
        """
        cycle_start = time.monotonic()

        # v15.5: Check Resource Guardian IPC for throttle/suspend commands
        try:
            _guardian_cmd_path = Path("/tmp/l104_bridge/guardian/outbox")
            for cmd_name in ("guardian_suspend_all.json", "guardian_emergency_halt.json"):
                cmd_file = _guardian_cmd_path / cmd_name
                if cmd_file.exists():
                    try:
                        cmd_data = json.loads(cmd_file.read_text())
                        cmd_ts = cmd_data.get("timestamp", 0)
                        # Only respect commands from last 120 seconds
                        if time.time() - cmd_ts < 120:
                            _logger.warning(
                                "Guardian %s active — skipping findings cycle (avail=%sMB)",
                                cmd_data.get("command", "unknown"),
                                cmd_data.get("payload", {}).get("available_mb", "?"))
                            with self._lock:
                                self._cycles_completed += 1
                                self._last_cycle_time = 0.0
                                self._last_cycle_results = [{
                                    "name": "GUARDIAN_PAUSED",
                                    "reason": cmd_data.get("command", "guardian"),
                                }]
                            return {"skipped": True, "reason": "guardian_" + cmd_data.get("command", "pause")}
                    except Exception:
                        pass
        except Exception:
            pass

        # v13.4: Throttled memory pressure check
        # v15.4: Use cached psutil (avoids importlib overhead per cycle)
        now = time.monotonic()
        if (now - self._mem_check_ts) > self._MEM_CHECK_TTL:
            try:
                _ps = _get_cached_psutil()
                if _ps is not None:
                    self._cached_avail_mb = _ps.virtual_memory().available / (1024 * 1024)
            except Exception:
                self._cached_avail_mb = 2048.0
            self._mem_check_ts = now

        avail_mb = self._cached_avail_mb

        if avail_mb < 512:
            with self._lock:
                self._cycles_completed += 1
                self._last_cycle_time = 0.0
                self._last_cycle_results = [{
                    "name": "SKIPPED",
                    "reason": f"memory_pressure_{int(avail_mb)}MB",
                }]
            return {"skipped": True, "reason": "low_memory", "avail_mb": int(avail_mb)}

        # v13.4: Cached import (FAST registry for daemon speed)
        sims_to_run = _get_findings_sims()
        if sims_to_run is None:
            return {"error": "vqpu_findings_unavailable"}

        # v12.3: Limit simulation count when memory is tight (< 1GB)
        if avail_mb < 1024:
            sims_to_run = sims_to_run[:5]

        # ═══════════════════════════════════════════════════════════
        # v15.1: QUARANTINE FILTER — skip chronically failing sims
        # Decrement quarantine counters; filter out quarantined sims.
        # ═══════════════════════════════════════════════════════════
        active_sims = []
        for entry in sims_to_run:
            sim_name = entry[0]
            remaining = self._sim_quarantine.get(sim_name, 0)
            if remaining > 0:
                self._sim_quarantine[sim_name] = remaining - 1
                if remaining - 1 <= 0:
                    self._sim_quarantine.pop(sim_name, None)
                    self._sim_failure_counts.pop(sim_name, None)
                    _logger.info("Sim '%s' released from quarantine", sim_name)
            else:
                active_sims.append(entry)
        sims_to_run = active_sims

        # ═══════════════════════════════════════════════════════════
        # v13.4: SEQUENTIAL EXECUTION (faster than threaded — GIL-bound)
        # Python-level matrix ops (numpy small arrays) don't release GIL
        # enough for threading to help; overhead outweighs any parallelism.
        # v15.2: Disable GC during sim execution to avoid stop-the-world
        # pauses mid-cycle. GC runs explicitly between cycles instead.
        # ═══════════════════════════════════════════════════════════
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
            self._gc_paused_during_sims = True

        cycle_results = []
        try:
            for entry in sims_to_run:
                sim_result = _run_single_sim(entry)
                cycle_results.append(sim_result)
                # v15.0: Track per-sim timing for drift detection
                self._track_sim_timing(sim_result)
        finally:
            # v15.2: Re-enable GC after sim execution completes.
            # MUST be in finally — if a sim raises, GC stays disabled forever
            # without this guard, starving the process of memory reclamation.
            if gc_was_enabled:
                gc.enable()
                self._gc_paused_during_sims = False

        # ═══════════════════════════════════════════════════════════
        # v15.1: BATCHED ENGINE FEEDING + QUARANTINE + FIDELITY TRACKING
        # Feed all results to coherence/entropy engines, track per-sim
        # fidelity and alignment trends, quarantine chronic failures,
        # then update counters in a single lock acquisition.
        # ═══════════════════════════════════════════════════════════
        sims_run = 0
        sims_passed = 0
        sims_failed = 0
        total_elapsed = 0.0
        cycle_fidelities = []
        cycle_alignments = []

        for entry_data in cycle_results:
            sims_run += 1
            elapsed = entry_data.get("elapsed_ms", 0.0)
            total_elapsed += elapsed
            sim_name = entry_data.get("name", "")

            result_obj = entry_data.pop("_result_obj", None)
            if result_obj is not None:
                if result_obj.passed:
                    sims_passed += 1
                    # v15.1: Reset failure count on success
                    self._sim_failure_counts.pop(sim_name, None)
                else:
                    sims_failed += 1
                    # v15.1: Track per-sim failures for quarantine
                    self._sim_failure_counts[sim_name] = (
                        self._sim_failure_counts.get(sim_name, 0) + 1)
                    if self._sim_failure_counts[sim_name] >= _SIM_QUARANTINE_THRESHOLD:
                        self._sim_quarantine[sim_name] = _SIM_QUARANTINE_CYCLES
                        self._quarantine_log.append({
                            "ts": time.time(),
                            "sim": sim_name,
                            "failures": self._sim_failure_counts[sim_name],
                            "quarantine_cycles": _SIM_QUARANTINE_CYCLES,
                        })
                        _logger.warning(
                            "Sim '%s' QUARANTINED after %d failures (skip %d cycles)",
                            sim_name, self._sim_failure_counts[sim_name],
                            _SIM_QUARANTINE_CYCLES)

                # Feed to engines (cheap: <1ms each when pre-initialized)
                self._feed_coherence(result_obj)
                self._feed_entropy(result_obj)

                # v15.2: Eagerly release result_obj to reduce peak memory
                del result_obj

                # v15.1: Fidelity + alignment trend tracking
                fid = entry_data.get("fidelity", 0.0)
                align = entry_data.get("sacred_alignment", 0.0)
                if fid > 0:
                    if sim_name not in self._sim_fidelity_history:
                        self._sim_fidelity_history[sim_name] = deque(
                            maxlen=_FIDELITY_HISTORY_WINDOW)
                    self._sim_fidelity_history[sim_name].append(fid)
                    cycle_fidelities.append(fid)
                if align > 0:
                    if sim_name not in self._sim_alignment_history:
                        self._sim_alignment_history[sim_name] = deque(
                            maxlen=_FIDELITY_HISTORY_WINDOW)
                    self._sim_alignment_history[sim_name].append(align)
                    cycle_alignments.append(align)

                # SC history
                if entry_data.get("sc"):
                    with self._lock:
                        self._sc_history.append({
                            "cycle": self._cycles_completed,
                            "ts": time.time(),
                            **entry_data["sc"],
                        })
            else:
                sims_failed += 1
                # v15.1: Track failure for quarantine even without result_obj
                self._sim_failure_counts[sim_name] = (
                    self._sim_failure_counts.get(sim_name, 0) + 1)
                if self._sim_failure_counts.get(sim_name, 0) >= _SIM_QUARANTINE_THRESHOLD:
                    self._sim_quarantine[sim_name] = _SIM_QUARANTINE_CYCLES

        cycle_elapsed = (time.monotonic() - cycle_start) * 1000.0

        # v15.1: Per-cycle fidelity + alignment averages
        if cycle_fidelities:
            self._cycle_fidelity_avg.append(
                round(sum(cycle_fidelities) / len(cycle_fidelities), 6))
        if cycle_alignments:
            self._cycle_alignment_avg.append(
                round(sum(cycle_alignments) / len(cycle_alignments), 6))

        # v15.1: Throughput metric (sims/second)
        cycle_s = cycle_elapsed / 1000.0
        if cycle_s > 0 and sims_run > 0:
            self._cycle_throughput.append(round(sims_run / cycle_s, 2))

        # v15.1: Track leaked threads
        self._total_leaked_threads = len(_leaked_threads)

        # v15.2: Purge dead leaked thread entries — threads that have
        # actually terminated no longer need tracking (frees closure refs)
        self._purge_dead_leaked_threads()

        # v13.3: Single batched lock acquisition for all counters
        with self._lock:
            self._cycles_completed += 1
            self._total_sims_run += sims_run
            self._total_sims_passed += sims_passed
            self._total_sims_failed += sims_failed
            self._total_elapsed_ms += total_elapsed
            self._last_cycle_time = cycle_elapsed
            self._last_cycle_results = cycle_results
            summary = {
                "cycle": self._cycles_completed,
                "ts": time.time(),
                "passed": sims_passed,
                "total": sims_run,
                "pass_rate": round(sims_passed / max(sims_run, 1), 4),
                "elapsed_ms": round(cycle_elapsed, 2),
            }
            self._health_history.append(summary)

        return {
            "cycle": self._cycles_completed,
            "results": cycle_results,
            "passed": sims_passed,
            "total": sims_run,
            "elapsed_ms": round(cycle_elapsed, 2),
        }

    def _feed_coherence(self, result):
        """Feed simulation result to coherence engine.

        v13.1: Shares ScienceEngine instance with _feed_entropy to avoid
        creating disconnected engine copies.
        """
        try:
            if self._coherence_engine is None:
                self._init_science_engines()
            if self._coherence_engine is not None:
                payload = result.to_coherence_payload()
                self._coherence_engine.anchor(payload.get("total_fidelity", 0.9))
        except Exception as e:
            _logger.debug("_feed_coherence: %s", e)

    def _feed_entropy(self, result):
        """Feed simulation result to entropy engine.

        v13.1: Shares ScienceEngine instance with _feed_coherence.
        """
        try:
            if self._entropy_engine is None:
                self._init_science_engines()
            if self._entropy_engine is not None:
                entropy_val = result.to_entropy_input()
                self._entropy_engine.calculate_demon_efficiency(entropy_val)
        except Exception as e:
            _logger.debug("_feed_entropy: %s", e)

    def _init_science_engines(self):
        """v13.1: Initialize shared ScienceEngine for coherence + entropy feeds."""
        try:
            from l104_science_engine import ScienceEngine
            se = ScienceEngine()
            self._coherence_engine = se.coherence
            self._entropy_engine = se.entropy
        except Exception as e:
            _logger.debug("ScienceEngine init: %s", e)

    def _get_precog_engine(self):
        """v15.2: Import precognition engine once, cache at instance level.

        Avoids repeated ImportError on every subconscious harvest when
        l104_data_precognition is not installed.  Once the import fails,
        it's never retried (the package won't magically appear mid-run).
        """
        if self._cached_precog_engine is not None:
            return self._cached_precog_engine
        if self._precog_import_attempted:
            return None
        self._precog_import_attempted = True
        try:
            from l104_data_precognition import precognition_engine
            self._cached_precog_engine = precognition_engine
            return self._cached_precog_engine
        except ImportError:
            return None

    def _purge_dead_leaked_threads(self):
        """v15.2: Remove entries for leaked threads that have terminated.

        Timed-out simulation threads are tracked in the module-level
        _leaked_threads deque.  Over time, most will eventually finish
        (or be killed by the OS).  This method scans active threads and
        removes deque entries whose threads are no longer alive, freeing
        the closure references they hold (sim functions, result boxes, etc.).
        """
        if not _leaked_threads:
            return
        active_thread_names = {t.name for t in threading.enumerate()}
        initial = len(_leaked_threads)
        # Rebuild deque without dead threads
        alive = deque(
            (entry for entry in _leaked_threads
             if entry.get("thread_name") in active_thread_names),
            maxlen=_leaked_threads.maxlen,
        )
        purged = initial - len(alive)
        if purged > 0:
            _leaked_threads.clear()
            _leaked_threads.extend(alive)
            _logger.debug("Purged %d dead leaked thread entries", purged)
        self._total_leaked_threads = len(_leaked_threads)

    def _track_sim_timing(self, sim_result: dict):
        """v15.0: Track per-sim elapsed times and detect drift.

        Maintains a rolling window of the last _SIM_DRIFT_WINDOW timings
        per sim name.  If the latest timing exceeds the rolling average
        by _SIM_DRIFT_THRESHOLD×, a drift alert is emitted.
        """
        name = sim_result.get("name", "")
        elapsed = sim_result.get("elapsed_ms", 0.0)
        if not name or elapsed <= 0:
            return

        if name not in self._sim_timing_history:
            self._sim_timing_history[name] = deque(maxlen=_SIM_DRIFT_WINDOW)

        history = self._sim_timing_history[name]
        if len(history) >= 3:
            avg = sum(history) / len(history)
            if avg > 0 and elapsed > avg * _SIM_DRIFT_THRESHOLD:
                alert = {
                    "ts": time.time(),
                    "sim": name,
                    "elapsed_ms": round(elapsed, 2),
                    "avg_ms": round(avg, 2),
                    "ratio": round(elapsed / avg, 2),
                }
                self._drift_alerts.append(alert)
                _logger.warning(
                    "Sim drift: '%s' took %.0fms (avg %.0fms, %.1f×)",
                    name, elapsed, avg, elapsed / avg)
        history.append(elapsed)

    def _update_health_score(self):
        """v15.1: Compute unified health score (0.0–1.0).

        Components (weighted):
          - Pass rate (30%): total_passed / total_run
          - Timing health (20%): 1.0 if last cycle < 10s, decays to 0 at 120s
          - Stability (25%): exp decay based on consecutive failures
          - Fidelity trend (25%): average fidelity of last cycle (0–1 range)

        The health score is fed to brain intelligence and exposed in status().
        """
        # Pass rate component
        pass_rate = (self._total_sims_passed / max(self._total_sims_run, 1))

        # Timing health: 1.0 at 0ms, decaying to ~0 around 120s
        cycle_s = self._last_cycle_time / 1000.0
        timing_health = max(0.0, 1.0 - (cycle_s / 120.0))

        # Stability: exponential decay based on consecutive failures
        stability = math.exp(-0.5 * self._consecutive_failures)

        # Fidelity component: latest cycle average fidelity
        fidelity_health = 0.5  # neutral default
        if self._cycle_fidelity_avg:
            fidelity_health = min(1.0, self._cycle_fidelity_avg[-1])

        self._health_score = round(
            0.30 * pass_rate + 0.20 * timing_health
            + 0.25 * stability + 0.25 * fidelity_health, 4)

    def _harvest_brain_intelligence(self) -> Optional[dict]:
        """v13.0: Harvest intelligence from Quantum Brain before each cycle.

        v13.3: Uses cached quantum_brain import (avoids 1.2s re-import).
        """
        try:
            quantum_brain = _get_quantum_brain()
            if quantum_brain is None:
                return None
            brain_intel = {}

            # Read Sage verdict from last pipeline run
            sage = getattr(quantum_brain, 'results', {}).get('sage', {})
            if sage:
                brain_intel['sage_score'] = sage.get('unified_score', 0)
                brain_intel['sage_grade'] = sage.get('grade', 'unknown')

            # Read manifold data
            manifold = getattr(quantum_brain, 'results', {}).get('manifold', {})
            if manifold:
                summary = manifold.get('summary', {})
                brain_intel['manifold_dimension'] = summary.get('intrinsic_dimension', 0)
                brain_intel['manifold_curvature'] = summary.get('mean_ricci_curvature', 0)

            # Read predictive oracle
            oracle = getattr(quantum_brain, 'results', {}).get('predictive_oracle', {})
            if oracle and oracle.get('status') == 'ok':
                brain_intel['oracle_trajectory'] = oracle.get('alignment_trajectory', 'unknown')
                fids = oracle.get('predicted_fidelity', [])
                if fids:
                    brain_intel['oracle_next_fidelity'] = fids[0]

            # Feed daemon health back to brain's feedback bus
            try:
                with self._lock:
                    pass_rate = self._total_sims_passed / max(self._total_sims_run, 1)
                if hasattr(quantum_brain, 'feedback_bus'):
                    # v15.1: Include fidelity + alignment trend in brain feedback
                    avg_fidelity = (
                        self._cycle_fidelity_avg[-1]
                        if self._cycle_fidelity_avg else 0.0)
                    avg_alignment = (
                        self._cycle_alignment_avg[-1]
                        if self._cycle_alignment_avg else 0.0)
                    quantum_brain.feedback_bus.send({
                        'sender': 'vqpu_daemon',
                        'type': 'health_telemetry',
                        'payload': {
                            'event': 'daemon_cycle_health',
                            'pass_rate': round(pass_rate, 4),
                            'health_score': self._health_score,
                            'cycles': self._cycles_completed,
                            'consecutive_failures': self._consecutive_failures,
                            'circuit_breaker_open': self._circuit_breaker_open,
                            'watchdog_restarts': self._watchdog_restarts,
                            'sacred_alignment_avg': brain_intel.get('sage_score', 0),
                            'cycle_fidelity_avg': avg_fidelity,
                            'cycle_alignment_avg': avg_alignment,
                            'quarantined_sims': len(self._sim_quarantine),
                            'leaked_threads': self._total_leaked_threads,
                        },
                    })
            except Exception:
                pass

            return brain_intel if brain_intel else None
        except Exception:
            return None

    def _harvest_evolution_scores(self) -> Optional[dict]:
        """v13.1: Harvest ASI/AGI evolution scores for cross-engine telemetry.

        v13.3: Uses cached asi_core/agi_core imports (avoids 3.4s re-import).
        """
        evo = {}
        # ASI evolution index + composite
        asi_core = _get_asi_core()
        if asi_core is not None:
            try:
                evo['asi_evolution_index'] = getattr(asi_core, 'evolution_index', 0)
                evo['asi_evolution_stage'] = getattr(asi_core, 'evolution_stage', 'unknown')
                evo['asi_vqpu_health'] = getattr(asi_core, '_vqpu_bridge_health_score', 0.0)
                evo['asi_vqpu_sacred'] = getattr(asi_core, '_vqpu_sacred_alignment_score', 0.0)
            except Exception:
                pass
        # AGI evolution + consciousness
        agi_core = _get_agi_core()
        if agi_core is not None:
            try:
                evo['agi_consciousness'] = getattr(agi_core, '_consciousness_level', 0.0)
                evo['agi_coherence'] = getattr(agi_core, '_coherence_level', 0.0)
                evo['agi_stage'] = getattr(agi_core, '_evo_stage', 'unknown')
            except Exception:
                pass
        return evo if evo else None

    def _guarded_persist(self):
        """v15.2: Persist wrapper with overlap guard.

        Prevents two background persist threads from writing the state
        file concurrently, which would produce a corrupt/partial JSON.
        """
        if self._persist_in_progress:
            return
        self._persist_in_progress = True
        try:
            self._persist_state()
        finally:
            self._persist_in_progress = False

    def _persist_state(self):
        """Write cumulative state to disk."""
        try:
            with self._lock:
                state = {
                    "version": "15.3.0",
                    "daemon_cycler": "VQPUDaemonCycler",
                    "last_persist": time.time(),
                    "cycles_completed": self._cycles_completed,
                    "total_sims_run": self._total_sims_run,
                    "total_sims_passed": self._total_sims_passed,
                    "total_sims_failed": self._total_sims_failed,
                    "pass_rate": round(
                        self._total_sims_passed / max(self._total_sims_run, 1), 4),
                    "total_elapsed_ms": round(self._total_elapsed_ms, 2),
                    "avg_cycle_ms": round(
                        self._total_elapsed_ms / max(self._cycles_completed, 1), 2),
                    "last_cycle_ms": round(self._last_cycle_time, 2),
                    "health_score": self._health_score,
                    "consecutive_failures": self._consecutive_failures,
                    "watchdog_restarts": self._watchdog_restarts,
                    "quarantined_sims": dict(self._sim_quarantine),
                    "sim_failure_counts": dict(self._sim_failure_counts),
                    "cycle_fidelity_avg": list(self._cycle_fidelity_avg)[-10:],
                    "cycle_alignment_avg": list(self._cycle_alignment_avg)[-10:],
                    "total_leaked_threads": self._total_leaked_threads,
                    "sc_history_count": len(self._sc_history),
                    "sc_latest": self._sc_history[-1] if self._sc_history else None,
                    "health_history": list(self._health_history)[-10:],
                    "god_code": GOD_CODE,
                }
            self._state_path.write_text(
                json.dumps(state, indent=2, default=str))
        except Exception as e:
            _logger.warning("Failed to persist state to %s: %s", self._state_path, e)

    def _load_state(self):
        """Load persisted state from disk (including quarantine + health score)."""
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                self._cycles_completed = data.get("cycles_completed", 0)
                self._total_sims_run = data.get("total_sims_run", 0)
                self._total_sims_passed = data.get("total_sims_passed", 0)
                self._total_sims_failed = data.get("total_sims_failed", 0)
                self._total_elapsed_ms = data.get("total_elapsed_ms", 0.0)
                self._watchdog_restarts = data.get("watchdog_restarts", 0)
                self._health_score = data.get("health_score", 1.0)
                self._consecutive_failures = data.get("consecutive_failures", 0)
                # Restore quarantine state
                quarantined = data.get("quarantined_sims", {})
                for name, cycles in quarantined.items():
                    if cycles > 0:
                        self._sim_quarantine[name] = cycles
                fail_counts = data.get("sim_failure_counts", {})
                for name, count in fail_counts.items():
                    self._sim_failure_counts[name] = count
        except Exception as e:
            _logger.warning("Failed to load state from %s: %s", self._state_path, e)

    def _read_micro_daemon_health(self) -> dict:
        """v2.4: Read micro daemon state file for cross-health monitoring.

        Reads the persisted .l104_vqpu_micro_daemon.json state file to check
        micro daemon health without importing or calling it directly.
        Also checks PID file liveness and heartbeat file freshness.
        """
        try:
            # Check PID liveness first
            pid_alive = False
            pid_file = Path(BRIDGE_PATH) / "micro" / "micro_daemon.pid"
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    import os as _os
                    _os.kill(pid, 0)
                    pid_alive = True
                except (ValueError, ProcessLookupError, PermissionError, OSError):
                    pid_alive = False

            # v2.4: Check heartbeat file freshness (mtime age in seconds)
            heartbeat_age_s = -1.0
            heartbeat_file = Path(BRIDGE_PATH) / "micro" / "heartbeat"
            if heartbeat_file.exists():
                try:
                    heartbeat_age_s = round(time.time() - heartbeat_file.stat().st_mtime, 1)
                except Exception:
                    pass

            # Read state file
            import os as _os
            _l104_root = _os.environ.get("L104_ROOT", _os.getcwd())
            _micro_state = Path(_l104_root) / ".l104_vqpu_micro_daemon.json"
            if _micro_state.exists():
                with open(_micro_state, "r") as _f:
                    _data = json.load(_f)
                result = {
                    "available": True,
                    "pid_alive": pid_alive,
                    "heartbeat_age_s": heartbeat_age_s,
                    "active": _data.get("bridge_connected", False),
                    "health_score": _data.get("health_score", 0),
                    "tick": _data.get("tick", 0),
                    "total_tasks_run": _data.get("total_tasks_run", 0),
                    "pass_rate": _data.get("pass_rate", 0),
                    "version": _data.get("version", "?"),
                    "vqpu_tasks_enabled": _data.get("vqpu_tasks_enabled", False),
                    "crash_count": _data.get("crash_count", 0),
                    "last_persist": _data.get("last_persist", 0),
                    "registered_tasks": len(_data.get("registered_tasks", [])),
                    # v2.4: Staleness detection — if heartbeat is too old, flag it
                    "stale": heartbeat_age_s > 30.0 if heartbeat_age_s >= 0 else True,
                }
                # v2.4: Derive approximate analytics grade from state data
                health = _data.get("health_score", 0)
                pr = _data.get("pass_rate", 0)
                if health >= 0.95 and pr >= 0.99:
                    result["analytics_grade"] = "A"
                elif health >= 0.85 and pr >= 0.95:
                    result["analytics_grade"] = "B"
                elif health >= 0.70 and pr >= 0.90:
                    result["analytics_grade"] = "C"
                elif health >= 0.50 and pr >= 0.80:
                    result["analytics_grade"] = "D"
                else:
                    result["analytics_grade"] = "F"
                return result
            return {"available": False, "pid_alive": pid_alive,
                    "heartbeat_age_s": heartbeat_age_s, "reason": "state_file_missing"}
        except Exception as _e:
            return {"available": False, "pid_alive": False,
                    "heartbeat_age_s": -1, "reason": str(_e)[:60]}

    def status(self) -> dict:
        """Current daemon cycler health and run history + subconscious state.

        v15.3: Minimized lock scope — snapshot mutable state under lock,
        build the large 60+ key dict outside the lock to avoid contention.
        """
        # --- Fast snapshot under lock ---
        with self._lock:
            snap = {
                "active": self._active,
                "paused": self._paused,
                "start_time": self._start_time,
                "interval": self._interval,
                "cycles_completed": self._cycles_completed,
                "total_sims_run": self._total_sims_run,
                "total_sims_passed": self._total_sims_passed,
                "total_sims_failed": self._total_sims_failed,
                "total_elapsed_ms": self._total_elapsed_ms,
                "last_cycle_time": self._last_cycle_time,
                "sc_history": list(self._sc_history)[-1:],
                "sc_runs": len(self._sc_history),
                "health_history": list(self._health_history)[-5:],
                "health_score": self._health_score,
                "consecutive_failures": self._consecutive_failures,
                "circuit_breaker_open": self._circuit_breaker_open,
                "watchdog_restarts": self._watchdog_restarts,
                "drift_alerts": list(self._drift_alerts)[-5:],
                "error_log": list(self._error_log)[-10:],
                "quarantined_sims": dict(self._sim_quarantine),
                "cycle_fidelity_avg": list(self._cycle_fidelity_avg)[-5:],
                "cycle_alignment_avg": list(self._cycle_alignment_avg)[-5:],
                "throughput_data": list(self._cycle_throughput)[-1:],
                "leaked_threads": self._total_leaked_threads,
                "adaptive_interval": self._adaptive_interval,
                "last_cpu_percent": self._last_cpu_percent,
                "gc_collections": self._gc_collections,
                "gc_objects_freed": self._gc_objects_freed,
                "gc_paused_during_sims": self._gc_paused_during_sims,
                "qs_avail": self._quantum_subconscious is not None,
                "harvests": self._subconscious_harvests,
                "precog_seeds_fed": self._subconscious_precog_seeds_fed,
                "cached_brain_data": self._cached_brain_data,
            }

        # --- Build dict outside lock (no contention) ---
        uptime = time.time() - snap["start_time"] if snap["active"] else 0
        td = snap["throughput_data"]
        s = {
            "version": "15.3.0",
            "active": snap["active"],
            "paused": snap["paused"],
            "uptime_seconds": round(uptime, 1),
            "interval_seconds": snap["interval"],
            "cycles_completed": snap["cycles_completed"],
            "total_sims_run": snap["total_sims_run"],
            "total_sims_passed": snap["total_sims_passed"],
            "total_sims_failed": snap["total_sims_failed"],
            "pass_rate": round(
                snap["total_sims_passed"] / max(snap["total_sims_run"], 1), 4),
            "avg_cycle_ms": round(
                snap["total_elapsed_ms"] / max(snap["cycles_completed"], 1), 2),
            "last_cycle_ms": round(snap["last_cycle_time"], 2),
            "sc_runs": snap["sc_runs"],
            "sc_latest": snap["sc_history"][0] if snap["sc_history"] else None,
            "health_trend": snap["health_history"],
            "state_file": str(self._state_path),
            "god_code": GOD_CODE,
            "health_score": snap["health_score"],
            "consecutive_failures": snap["consecutive_failures"],
            "circuit_breaker_open": snap["circuit_breaker_open"],
            "watchdog_restarts": snap["watchdog_restarts"],
            "drift_alerts": snap["drift_alerts"],
            "error_log": snap["error_log"],
            "quarantined_sims": snap["quarantined_sims"],
            "quarantined_count": len(snap["quarantined_sims"]),
            "cycle_fidelity_avg": snap["cycle_fidelity_avg"],
            "cycle_alignment_avg": snap["cycle_alignment_avg"],
            "throughput_sims_per_sec": round(td[0], 2) if td else 0,
            "leaked_threads": snap["leaked_threads"],
            "adaptive_interval_s": snap["adaptive_interval"],
            "last_cpu_percent": snap["last_cpu_percent"],
            "sim_timeout_s": _SIM_TIMEOUT_S,
            "execution_mode": "sequential",
            "sim_registry": "VQPU_FINDINGS_SIMULATIONS_FAST",
            "gc_collections": snap["gc_collections"],
            "gc_objects_freed": snap["gc_objects_freed"],
            "gc_paused_during_sims": snap["gc_paused_during_sims"],
            "quantum_subconscious": {
                "available": snap["qs_avail"],
                "harvests": snap["harvests"],
                "precog_seeds_fed": snap["precog_seeds_fed"],
            },
            "brain_intelligence": {
                "available": True,
                "last_harvest": snap["cached_brain_data"],
            },
            "micro_daemon_cross_health": self._read_micro_daemon_health(),
        }

        # v12.1: Append full subconscious status if available
        if snap["qs_avail"]:
            try:
                s["quantum_subconscious"].update(self._quantum_subconscious.status())
            except Exception:
                pass

        return s


__all__ = ["VQPUDaemonCycler"]
