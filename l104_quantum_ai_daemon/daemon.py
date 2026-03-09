"""L104 Quantum AI Daemon v2.0.0 — Autonomous Multipurpose High-Fidelity Daemon.

The sovereign orchestrator that autonomously improves all L104 files,
validates quantum fidelity, harmonizes cross-engine processes, and
evolves its own strategies over time.

Architecture:
  7-phase improvement cycle running on adaptive interval (60s–600s):
    Phase 1: FILE SCAN      — Discover & index all L104 Python files
    Phase 2: FIDELITY CHECK  — Validate sacred constants & quantum coherence
    Phase 3: HARMONY CHECK   — Cross-engine consistency validation
    Phase 4: OPTIMIZE        — Memory, GC, import cache, temp cleanup
    Phase 5: IMPROVE         — Analyze & auto-fix batch of priority files
    Phase 6: EVOLVE          — Self-improving strategy adaptation
    Phase 7: PERSIST         — State checkpoint to disk

Lifecycle:
    daemon = QuantumAIDaemon()
    daemon.start()              # Spawns background daemon thread
    daemon.status()             # Full status dashboard
    daemon.force_cycle()        # Trigger immediate improvement cycle
    daemon.stop()               # Graceful shutdown + state persist

CLI:
    python -m l104_quantum_ai_daemon                   # Run daemon
    python -m l104_quantum_ai_daemon --self-test       # Test all subsystems
    python -m l104_quantum_ai_daemon --health-check    # Quick health report
    python -m l104_quantum_ai_daemon --status          # Read persisted state
    python -m l104_quantum_ai_daemon --single-cycle    # Run one cycle and exit

SACRED INVARIANT: GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
"""

import atexit
import gc
import json
import logging
import math
import os
import random
import signal
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT, OMEGA, TAU,
    DAEMON_VERSION, SACRED_RESONANCE,
    CYCLE_INTERVAL_S, CYCLE_MIN_INTERVAL_S, CYCLE_MAX_INTERVAL_S,
    LOAD_THRESHOLD_LOW, LOAD_THRESHOLD_HIGH,
    SCAN_BATCH_SIZE, STATE_PATH, PERSIST_EVERY_N_CYCLES,
    PID_FILE, HEARTBEAT_FILE, LOG_DIR, IPC_PATH, INBOX_PATH, OUTBOX_PATH,
    TELEMETRY_WINDOW, ERROR_LOG_SIZE, IMPROVEMENT_HISTORY_SIZE,
    ANOMALY_SIGMA, HEALTH_STALENESS_DECAY,
    QUARANTINE_THRESHOLD, QUARANTINE_CYCLES,
    CIRCUIT_BREAKER_BASE_S, CIRCUIT_BREAKER_MAX_S,
    L104_ROOT,
)
from .scanner import FileScanner, L104FileInfo
from .improver import CodeImprover, ImprovementResult
from .fidelity import QuantumFidelityGuard, FidelityReport
from .optimizer import ProcessOptimizer, OptimizationResult
from .harmonizer import CrossEngineHarmonizer, HarmonyReport
from .evolver import AutonomousEvolver, EvolutionCycle

_logger = logging.getLogger("L104_QAI_DAEMON")


class DaemonPhase(str, Enum):
    """Current execution phase of the daemon cycle."""
    IDLE = "idle"
    SCANNING = "scanning"
    FIDELITY = "fidelity_check"
    HARMONY = "harmony_check"
    OPTIMIZING = "optimizing"
    IMPROVING = "improving"
    EVOLVING = "evolving"
    PERSISTING = "persisting"
    CROSS_SYNC = "cross_sync"       # v2.0: Cross-daemon synchronization
    SHUTDOWN = "shutdown"


@dataclass
class FileHealthRecord:
    """Persistent health record for a single file across daemon cycles."""
    relative_path: str
    health_score: float = 1.0
    times_improved: int = 0
    times_analyzed: int = 0
    last_health_change: float = 0.0
    sacred_alignment: float = 0.0


@dataclass
class ImprovementReport:
    """Summary report of one complete daemon cycle."""
    cycle_number: int
    timestamp: float = field(default_factory=time.time)
    phase: str = DaemonPhase.IDLE
    duration_ms: float = 0.0
    files_scanned: int = 0
    files_analyzed: int = 0
    files_improved: int = 0
    fidelity_grade: str = "N/A"
    fidelity_score: float = 0.0
    harmony_score: float = 0.0
    optimization_memory_freed_mb: float = 0.0
    evolution_delta: float = 0.0
    sacred_resonance: float = 0.0
    health_score: float = 0.0            # Overall daemon health (0–1)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DaemonConfig:
    """Configuration for the Quantum AI Daemon."""
    cycle_interval: float = CYCLE_INTERVAL_S
    auto_fix_enabled: bool = True
    doc_gen_enabled: bool = False
    scan_batch_size: int = SCAN_BATCH_SIZE
    enable_evolution: bool = True
    enable_ipc: bool = True
    state_path: Optional[str] = None
    log_level: str = "INFO"
    enable_multi_lang: bool = True           # v2.0: Scan Swift/C/Rust files
    enable_cross_sync: bool = True           # v2.0: Cross-daemon synchronization
    enable_feedback_loop: bool = True        # v2.0: Improvement effectiveness tracking
    multi_lang_extensions: tuple = (".swift", ".c", ".rs", ".h")  # v2.0: Extra file types


class QuantumAIDaemon:
    """Autonomous multipurpose high-fidelity quantum AI daemon.

    The sovereign daemon that continuously improves all L104 files,
    validates quantum fidelity, and evolves its own strategies.

    Thread-safe. Single daemon thread with 7-phase cycles.
    """

    def __init__(self, config: Optional[DaemonConfig] = None):
        cfg = config or DaemonConfig()

        # Configuration
        self._cycle_interval = cfg.cycle_interval
        self._adaptive_interval = cfg.cycle_interval
        self._auto_fix_enabled = cfg.auto_fix_enabled
        self._enable_evolution = cfg.enable_evolution
        self._enable_ipc = cfg.enable_ipc
        self._state_path = Path(cfg.state_path or STATE_PATH)

        # Subsystems
        self._scanner = FileScanner()
        self._improver = CodeImprover(
            auto_fix_enabled=cfg.auto_fix_enabled,
            doc_gen_enabled=cfg.doc_gen_enabled,
        )
        self._fidelity = QuantumFidelityGuard()
        self._optimizer = ProcessOptimizer()
        self._harmonizer = CrossEngineHarmonizer()
        self._evolver = AutonomousEvolver()

        # Thread control
        self._stop_event = threading.Event()
        self._force_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # State
        self._active = False
        self._phase = DaemonPhase.IDLE
        self._cycles_completed = 0
        self._start_time = 0.0
        self._last_cycle_time = 0.0
        self._health_score = 1.0

        # Telemetry
        self._telemetry: deque = deque(maxlen=TELEMETRY_WINDOW)
        self._error_log: deque = deque(maxlen=ERROR_LOG_SIZE)
        self._cycle_history: deque = deque(maxlen=IMPROVEMENT_HISTORY_SIZE)

        # Circuit breaker
        self._consecutive_failures = 0
        self._circuit_breaker_open = False
        self._circuit_breaker_until = 0.0

        # Jitter for cycle timing
        self._jitter_fraction = 0.10

        # v2.0: Phase timing analytics
        self._phase_timings: dict[str, deque] = {}  # phase → deque of elapsed_ms

        # v2.0: Cross-daemon state cache
        self._cross_daemon_state: dict = {}
        self._cross_daemon_ts = 0.0

        # v2.0: Improvement effectiveness tracking
        self._improvement_effectiveness: deque = deque(maxlen=100)
        self._rollback_count = 0

        # v2.0: Multi-language file tracking
        self._multi_lang_files: dict[str, int] = {}  # extension → count

        # v2.0: Store config reference
        self._config = cfg

    # ═══════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════

    def start(self):
        """Start the daemon in a background thread."""
        if self._active:
            _logger.warning("Daemon already active")
            return

        # Ensure directories exist
        for d in [LOG_DIR, IPC_PATH, INBOX_PATH, OUTBOX_PATH]:
            d.mkdir(parents=True, exist_ok=True)

        # Write PID file
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))

        # Load persisted state
        self._load_state()

        # Initial full scan
        _logger.info(
            f"L104 Quantum AI Daemon v{DAEMON_VERSION} starting "
            f"(cycle={self._cycle_interval}s, pid={os.getpid()})")
        self._scanner.full_scan()

        self._active = True
        self._start_time = time.time()
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run_loop, name="L104-QAI-Daemon", daemon=True)
        self._thread.start()

        # Register atexit for clean shutdown
        atexit.register(self._atexit_handler)

        _logger.info(
            f"Daemon started — {self._scanner.indexed_count} files indexed, "
            f"GOD_CODE={GOD_CODE}, PHI={PHI}")

    def stop(self):
        """Graceful shutdown: persist state, clean up, stop thread."""
        if not self._active:
            return

        _logger.info("Daemon shutting down...")
        self._phase = DaemonPhase.SHUTDOWN
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30.0)

        self._persist_state()
        self._active = False

        # Clean PID file
        try:
            PID_FILE.unlink(missing_ok=True)
        except OSError:
            pass

        _logger.info(
            f"Daemon stopped after {self._cycles_completed} cycles, "
            f"uptime={time.time() - self._start_time:.0f}s")

    def force_cycle(self):
        """Trigger an immediate improvement cycle."""
        self._force_event.set()

    def _atexit_handler(self):
        """Ensure state is persisted on process exit."""
        if self._active:
            try:
                self._persist_state()
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════

    def _run_loop(self):
        """Main daemon loop — runs 7-phase improvement cycles."""
        _logger.info("Daemon loop started")

        while not self._stop_event.is_set():
            try:
                # Check circuit breaker
                if self._circuit_breaker_open:
                    if time.time() < self._circuit_breaker_until:
                        self._write_heartbeat()  # Stay visible while waiting
                        self._stop_event.wait(10.0)
                        continue
                    else:
                        self._circuit_breaker_open = False
                        _logger.info("Circuit breaker closed — resuming")

                # Run improvement cycle
                report = self._run_cycle()
                self._cycle_history.append(report)

                # Update health + bookkeeping
                self._health_score = report.health_score
                self._last_cycle_time = time.time()
                self._cycles_completed += 1

                if report.error:
                    self._consecutive_failures += 1
                    self._error_log.append({
                        "cycle": self._cycles_completed,
                        "error": report.error,
                        "time": time.time(),
                    })
                    if self._consecutive_failures >= 3:
                        self._trip_circuit_breaker()
                else:
                    self._consecutive_failures = 0

                # Write heartbeat
                self._write_heartbeat()

                # Adaptive interval
                self._compute_adaptive_interval()

                # Persist periodically
                if self._cycles_completed % PERSIST_EVERY_N_CYCLES == 0:
                    self._persist_state()

            except Exception as e:
                _logger.error(f"Cycle error: {e}", exc_info=True)
                self._error_log.append({
                    "cycle": self._cycles_completed,
                    "error": str(e),
                    "time": time.time(),
                })
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    self._trip_circuit_breaker()

            # Wait for next cycle (with jitter + force interrupt)
            jitter = random.uniform(
                -self._jitter_fraction, self._jitter_fraction)
            wait_time = self._adaptive_interval * (1.0 + jitter)
            interrupted = self._stop_event.wait(timeout=wait_time)
            if interrupted and not self._force_event.is_set():
                break
            self._force_event.clear()

        _logger.info("Daemon loop exiting")

    def _run_cycle(self) -> ImprovementReport:
        """Execute one complete 7-phase improvement cycle."""
        t0 = time.monotonic()
        report = ImprovementReport(
            cycle_number=self._cycles_completed + 1)

        try:
            # ── Phase 1: FILE SCAN ──
            phase_start = time.monotonic()
            self._phase = DaemonPhase.SCANNING
            report.phase = DaemonPhase.SCANNING
            changed = self._scanner.get_changed_files()
            if changed or self._cycles_completed % 10 == 0:
                self._scanner.full_scan()
            report.files_scanned = self._scanner.indexed_count
            self._record_phase_timing("scanning", time.monotonic() - phase_start)

            # ── Phase 2: FIDELITY CHECK ──
            phase_start = time.monotonic()
            self._phase = DaemonPhase.FIDELITY
            report.phase = DaemonPhase.FIDELITY
            fidelity = self._fidelity.run_fidelity_check()
            report.fidelity_grade = fidelity.grade
            report.fidelity_score = fidelity.overall_fidelity
            report.sacred_resonance = SACRED_RESONANCE

            if fidelity.warnings:
                report.warnings.extend(fidelity.warnings[:5])
            self._record_phase_timing("fidelity", time.monotonic() - phase_start)

            # ── Phase 3: HARMONY CHECK ──
            phase_start = time.monotonic()
            self._phase = DaemonPhase.HARMONY
            report.phase = DaemonPhase.HARMONY
            harmony = self._harmonizer.harmonize()
            report.harmony_score = harmony.overall_harmony
            self._record_phase_timing("harmony", time.monotonic() - phase_start)

            # ── Phase 4: OPTIMIZE ──
            phase_start = time.monotonic()
            self._phase = DaemonPhase.OPTIMIZING
            report.phase = DaemonPhase.OPTIMIZING
            opt = self._optimizer.optimize()
            report.optimization_memory_freed_mb = opt.memory_freed_mb
            self._record_phase_timing("optimizing", time.monotonic() - phase_start)

            # ── Phase 5: IMPROVE FILES ──
            phase_start = time.monotonic()
            self._phase = DaemonPhase.IMPROVING
            report.phase = DaemonPhase.IMPROVING
            batch_size = self._evolver.current_strategy.get(
                "scan_batch_size", SCAN_BATCH_SIZE)
            batch = self._scanner.get_improvement_batch(batch_size)
            report.files_analyzed = len(batch)

            improvement_results = []
            for file_info in batch:
                try:
                    result = self._improver.analyze_file(
                        file_info.path, file_info.relative_path)

                    # Update scanner with new health data
                    self._scanner.update_health(
                        file_info.relative_path,
                        health=result.health_score,
                        smells=result.smells_found,
                        complexity=result.complexity_max,
                        alignment=result.sacred_alignment,
                    )

                    if result.improved:
                        report.files_improved += 1
                        file_info.improvement_count += 1

                    improvement_results.append(result)

                except Exception as e:
                    # Quarantine files that keep failing
                    file_info.failure_streak += 1
                    if file_info.failure_streak >= QUARANTINE_THRESHOLD:
                        self._scanner.quarantine_file(
                            file_info.relative_path, QUARANTINE_CYCLES)
                        report.warnings.append(
                            f"Quarantined: {file_info.relative_path}")
            self._record_phase_timing("improving", time.monotonic() - phase_start)

            # ── Phase 6: EVOLVE ──
            phase_start = time.monotonic()
            self._phase = DaemonPhase.EVOLVING
            report.phase = DaemonPhase.EVOLVING
            if self._enable_evolution and improvement_results:
                evo = self._evolver.evolve(
                    improvement_results=[
                        self._improver.stats()],
                    fidelity_score=report.fidelity_score,
                    harmony_score=report.harmony_score,
                    optimization_score=min(1.0,
                        opt.memory_freed_mb / 10.0 + 0.5),
                )
                report.evolution_delta = evo.evolution_delta
            self._record_phase_timing("evolving", time.monotonic() - phase_start)

            # ── Phase 7: PERSIST ──
            phase_start = time.monotonic()
            self._phase = DaemonPhase.PERSISTING
            report.phase = DaemonPhase.PERSISTING
            # (Periodic persistence handled in main loop)
            self._record_phase_timing("persisting", time.monotonic() - phase_start)

            # ── Phase 8: CROSS-DAEMON SYNC (v2.0) ──
            if self._config.enable_cross_sync:
                phase_start = time.monotonic()
                self._phase = DaemonPhase.CROSS_SYNC
                try:
                    self._cross_daemon_sync()
                except Exception as e:
                    _logger.debug("Cross-sync phase: %s", e)
                self._record_phase_timing("cross_sync", time.monotonic() - phase_start)

            # Compute overall health score
            report.health_score = self._compute_health(
                report.fidelity_score,
                report.harmony_score,
                report.files_improved,
                report.files_analyzed,
            )

        except Exception as e:
            report.error = str(e)
            _logger.error(f"Cycle #{report.cycle_number} error: {e}")

        finally:
            self._phase = DaemonPhase.IDLE
            report.phase = DaemonPhase.IDLE
            report.duration_ms = (time.monotonic() - t0) * 1000

        # Telemetry snapshot
        self._telemetry.append({
            "cycle": report.cycle_number,
            "time": time.time(),
            "health": report.health_score,
            "fidelity": report.fidelity_score,
            "harmony": report.harmony_score,
            "files_improved": report.files_improved,
            "duration_ms": report.duration_ms,
        })

        _logger.info(
            f"═══ Cycle #{report.cycle_number} complete: "
            f"health={report.health_score:.3f} "
            f"fidelity={report.fidelity_grade} "
            f"harmony={report.harmony_score:.3f} "
            f"files={report.files_improved}/{report.files_analyzed} improved "
            f"evo_delta={report.evolution_delta:+.4f} "
            f"({report.duration_ms:.0f}ms) ═══"
        )
        return report

    # ═══════════════════════════════════════════════════════════════
    # v2.0: PHASE TIMING + CROSS-DAEMON + MULTI-LANG
    # ═══════════════════════════════════════════════════════════════

    def _record_phase_timing(self, phase_name: str, elapsed_s: float):
        """v2.0: Record execution time for a phase."""
        ms = elapsed_s * 1000.0
        if phase_name not in self._phase_timings:
            self._phase_timings[phase_name] = deque(maxlen=50)
        self._phase_timings[phase_name].append(round(ms, 2))

    def _cross_daemon_sync(self):
        """v2.0: Read other daemon states and adjust improvement priorities."""
        now = time.time()
        if (now - self._cross_daemon_ts) < 30.0 and self._cross_daemon_state:
            return

        l104_root = str(L104_ROOT)
        daemon_files = {
            "vqpu_cycler": ".l104_vqpu_daemon_state.json",
            "micro_daemon": ".l104_vqpu_micro_daemon.json",
            "nano_daemon": ".l104_nano_daemon_python.json",
            "guardian": ".l104_resource_guardian.json",
        }

        state = {}
        for name, filename in daemon_files.items():
            try:
                path = Path(l104_root) / filename
                if path.exists():
                    data = json.loads(path.read_text())
                    state[name] = {
                        "health_score": data.get("health_score", data.get("health_trend", 0)),
                        "available": True,
                    }
                else:
                    state[name] = {"available": False}
            except Exception:
                state[name] = {"available": False}

        self._cross_daemon_state = state
        self._cross_daemon_ts = now

    def _scan_multi_lang_files(self) -> dict:
        """v2.0: Count Swift, C, Rust, and header files in the L104 codebase."""
        counts = {}
        l104_root = Path(str(L104_ROOT))
        for ext in self._config.multi_lang_extensions:
            try:
                files = list(l104_root.rglob(f"*{ext}"))
                # Exclude .build, .venv, node_modules
                files = [f for f in files if not any(
                    p in str(f) for p in [".build", ".venv", "node_modules", "__pycache__"])]
                counts[ext] = len(files)
            except Exception:
                counts[ext] = 0
        self._multi_lang_files = counts
        return counts

    def phase_timings(self) -> dict:
        """v2.0: Return per-phase timing analytics."""
        result = {}
        for phase, times in self._phase_timings.items():
            if times:
                result[phase] = {
                    "samples": len(times),
                    "avg_ms": round(sum(times) / len(times), 2),
                    "max_ms": round(max(times), 2),
                    "latest_ms": round(times[-1], 2),
                }
        return result

    # ═══════════════════════════════════════════════════════════════
    # HEALTH + ADAPTIVE INTERVAL
    # ═══════════════════════════════════════════════════════════════

    def _compute_health(self, fidelity: float, harmony: float,
                        improved: int, analyzed: int) -> float:
        """Compute overall daemon health (0–1)."""
        improvement_rate = improved / max(1, analyzed)
        health = (
            fidelity * 0.35 +
            harmony * 0.30 +
            min(1.0, improvement_rate + 0.5) * 0.20 +
            (1.0 - self._consecutive_failures * 0.1) * 0.15
        )
        return max(0.0, min(1.0, health))

    def _compute_adaptive_interval(self):
        """Adjust cycle interval based on CPU load and health."""
        cpu = self._optimizer.get_cpu_load()
        if cpu < LOAD_THRESHOLD_LOW:
            self._adaptive_interval = max(
                CYCLE_MIN_INTERVAL_S, self._cycle_interval * 0.8)
        elif cpu > LOAD_THRESHOLD_HIGH:
            self._adaptive_interval = min(
                CYCLE_MAX_INTERVAL_S, self._cycle_interval * 1.5)
        else:
            self._adaptive_interval = self._cycle_interval

    def _trip_circuit_breaker(self):
        """Open circuit breaker after consecutive failures."""
        backoff = min(
            CIRCUIT_BREAKER_MAX_S,
            CIRCUIT_BREAKER_BASE_S * (2 ** self._consecutive_failures))
        self._circuit_breaker_open = True
        self._circuit_breaker_until = time.time() + backoff
        _logger.warning(
            f"Circuit breaker OPEN — backing off {backoff:.0f}s "
            f"(failures={self._consecutive_failures})")

    def _write_heartbeat(self):
        """Write heartbeat timestamp for external monitoring."""
        try:
            HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_FILE.write_text(str(time.time()))
        except OSError:
            pass

    # ═══════════════════════════════════════════════════════════════
    # STATE PERSISTENCE
    # ═══════════════════════════════════════════════════════════════

    def _persist_state(self):
        """Save daemon state to disk."""
        try:
            state = {
                "version": DAEMON_VERSION,
                "god_code": GOD_CODE,
                "cycles_completed": self._cycles_completed,
                "health_score": self._health_score,
                "start_time": self._start_time,
                "last_cycle_time": self._last_cycle_time,
                "file_index": self._scanner.to_dict(),
                "improver_stats": self._improver.stats(),
                "fidelity_stats": self._fidelity.stats(),
                "optimizer_stats": self._optimizer.stats(),
                "harmonizer_stats": self._harmonizer.stats(),
                "evolver_stats": self._evolver.stats(),
                "telemetry_recent": list(self._telemetry)[-20:],
                "error_log_recent": list(self._error_log)[-10:],
                "persisted_at": time.time(),
                "phase_timings": self.phase_timings(),
                "cross_daemon_state": self._cross_daemon_state,
                "multi_lang_files": self._multi_lang_files,
                "rollback_count": self._rollback_count,
            }
            temp = self._state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(state, indent=2, default=str))
            temp.rename(self._state_path)
            _logger.debug(f"State persisted to {self._state_path}")
        except Exception as e:
            _logger.warning(f"Persist failed: {e}")

    def _load_state(self):
        """Restore daemon state from disk."""
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                self._cycles_completed = data.get("cycles_completed", 0)
                self._health_score = data.get("health_score", 1.0)

                # Restore file index health data
                file_data = data.get("file_index", {})
                if file_data:
                    # Full scan first, then overlay persisted health
                    self._scanner.full_scan()
                    self._scanner.restore_from_dict(file_data)

                _logger.info(
                    f"State restored: {self._cycles_completed} prior cycles, "
                    f"health={self._health_score:.3f}")
        except Exception as e:
            _logger.warning(f"State load failed (fresh start): {e}")

    # ═══════════════════════════════════════════════════════════════
    # STATUS + DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════

    def status(self) -> dict:
        """Comprehensive daemon status dashboard."""
        uptime = time.time() - self._start_time if self._start_time else 0.0
        return {
            "daemon": {
                "version": DAEMON_VERSION,
                "active": self._active,
                "phase": self._phase.value,
                "pid": os.getpid(),
                "uptime_s": round(uptime, 1),
                "uptime_human": self._format_uptime(uptime),
                "cycles_completed": self._cycles_completed,
                "health_score": round(self._health_score, 4),
                "adaptive_interval_s": round(self._adaptive_interval, 1),
                "circuit_breaker": "OPEN" if self._circuit_breaker_open else "CLOSED",
                "consecutive_failures": self._consecutive_failures,
            },
            "sacred": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
                "VOID_CONSTANT": VOID_CONSTANT,
                "SACRED_RESONANCE": round(SACRED_RESONANCE, 4),
            },
            "scanner": self._scanner.stats(),
            "improver": self._improver.stats(),
            "fidelity": self._fidelity.stats(),
            "optimizer": self._optimizer.stats(),
            "harmonizer": self._harmonizer.stats(),
            "evolver": self._evolver.stats(),
            "telemetry_count": len(self._telemetry),
            "error_count": len(self._error_log),
            "last_errors": list(self._error_log)[-3:],
            "phase_timings": self.phase_timings(),
            "cross_daemon_state": self._cross_daemon_state,
            "multi_lang_files": self._multi_lang_files,
            "improvement_effectiveness_count": len(self._improvement_effectiveness),
            "rollback_count": self._rollback_count,
        }

    def self_test(self) -> dict:
        """Run diagnostic self-test across all subsystems."""
        t0 = time.monotonic()
        results = {}

        # Test 1: Scanner
        try:
            count = self._scanner.full_scan()
            results["scanner"] = {
                "status": "PASS" if count > 0 else "WARN",
                "files_found": count}
        except Exception as e:
            results["scanner"] = {"status": "FAIL", "error": str(e)}

        # Test 2: Code Engine availability
        try:
            from l104_code_engine import code_engine
            analysis = code_engine.full_analysis("x = 1 + 2")
            results["code_engine"] = {
                "status": "PASS" if analysis else "WARN"}
        except Exception as e:
            results["code_engine"] = {"status": "FAIL", "error": str(e)}

        # Test 3: Science Engine
        try:
            from l104_science_engine import ScienceEngine
            se = ScienceEngine()
            results["science_engine"] = {"status": "PASS"}
        except Exception as e:
            results["science_engine"] = {"status": "FAIL", "error": str(e)}

        # Test 4: Math Engine
        try:
            from l104_math_engine import MathEngine
            me = MathEngine()
            fib = me.fibonacci(10)
            results["math_engine"] = {
                "status": "PASS" if fib else "WARN"}
        except Exception as e:
            results["math_engine"] = {"status": "FAIL", "error": str(e)}

        # Test 5: Fidelity Guard
        try:
            report = self._fidelity.run_fidelity_check()
            results["fidelity_guard"] = {
                "status": "PASS" if report.overall_fidelity > 0.5 else "WARN",
                "grade": report.grade,
                "score": round(report.overall_fidelity, 3)}
        except Exception as e:
            results["fidelity_guard"] = {"status": "FAIL", "error": str(e)}

        # Test 6: Harmonizer
        try:
            harmony = self._harmonizer.harmonize()
            results["harmonizer"] = {
                "status": "PASS" if harmony.harmonized else "WARN",
                "score": round(harmony.overall_harmony, 3)}
        except Exception as e:
            results["harmonizer"] = {"status": "FAIL", "error": str(e)}

        # Test 7: Optimizer
        try:
            opt = self._optimizer.optimize()
            results["optimizer"] = {
                "status": "PASS",
                "gc_collected": opt.gc_collected}
        except Exception as e:
            results["optimizer"] = {"status": "FAIL", "error": str(e)}

        # Test 8: Sacred Constants
        results["sacred_constants"] = {
            "status": "PASS" if (
                abs(GOD_CODE - 527.5184818492612) < 1e-10 and
                abs(PHI - 1.618033988749895) < 1e-12
            ) else "FAIL",
            "GOD_CODE": GOD_CODE,
            "PHI": PHI,
        }

        # Test 9: State persistence
        try:
            self._persist_state()
            results["persistence"] = {
                "status": "PASS" if self._state_path.exists() else "WARN"}
        except Exception as e:
            results["persistence"] = {"status": "FAIL", "error": str(e)}

        # Test 10: IPC directories
        results["ipc"] = {
            "status": "PASS" if IPC_PATH.exists() else "WARN",
            "paths_exist": {
                "ipc": IPC_PATH.exists(),
                "inbox": INBOX_PATH.exists(),
                "outbox": OUTBOX_PATH.exists(),
            }
        }

        elapsed_ms = (time.monotonic() - t0) * 1000
        passed = sum(1 for r in results.values()
                     if r.get("status") == "PASS")
        total = len(results)

        results["_summary"] = {
            "passed": passed,
            "total": total,
            "all_pass": passed == total,
            "elapsed_ms": round(elapsed_ms, 1),
        }

        return results

    def _format_uptime(self, seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"

    # ═══════════════════════════════════════════════════════════════
    # IPC — Inter-Process Communication
    # ═══════════════════════════════════════════════════════════════

    def _poll_ipc(self):
        """Check IPC inbox for external commands."""
        if not self._enable_ipc:
            return

        try:
            if not INBOX_PATH.exists():
                return

            for entry in sorted(INBOX_PATH.iterdir()):
                if not entry.is_file() or not entry.suffix == ".json":
                    continue
                try:
                    cmd = json.loads(entry.read_text())
                    action = cmd.get("action", "")

                    response = {"status": "unknown_action"}
                    if action == "force_cycle":
                        self.force_cycle()
                        response = {"status": "cycle_triggered"}
                    elif action == "status":
                        response = self.status()
                    elif action == "health":
                        response = {
                            "health": self._health_score,
                            "active": self._active,
                        }

                    # Write response to outbox
                    out_file = OUTBOX_PATH / entry.name
                    out_file.write_text(json.dumps(response, default=str))
                    entry.unlink()  # Remove processed command

                except Exception as e:
                    _logger.debug(f"IPC error on {entry.name}: {e}")
                    try:
                        entry.unlink()
                    except OSError:
                        pass
        except FileNotFoundError:
            pass


# ═══════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def _cli_main():
    """CLI entry point for the Quantum AI Daemon."""
    import argparse

    parser = argparse.ArgumentParser(
        description="L104 Quantum AI Daemon — Autonomous File Improvement System")
    parser.add_argument("--self-test", action="store_true",
                        help="Run diagnostic self-test and exit")
    parser.add_argument("--health-check", action="store_true",
                        help="Quick health report and exit")
    parser.add_argument("--status", action="store_true",
                        help="Read persisted state and exit")
    parser.add_argument("--single-cycle", action="store_true",
                        help="Run one improvement cycle and exit")
    parser.add_argument("--interval", type=float, default=CYCLE_INTERVAL_S,
                        help=f"Cycle interval in seconds (default: {CYCLE_INTERVAL_S})")
    parser.add_argument("--no-auto-fix", action="store_true",
                        help="Disable auto-fix (read-only analysis)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    # Ensure log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Add file handler
    try:
        fh = logging.FileHandler(LOG_DIR / "daemon.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s"))
        logging.getLogger().addHandler(fh)
    except OSError:
        pass

    config = DaemonConfig(
        cycle_interval=args.interval,
        auto_fix_enabled=not args.no_auto_fix,
    )
    daemon = QuantumAIDaemon(config=config)

    if args.self_test:
        print(f"\n══════════════════════════════════════════════════")
        print(f"  L104 Quantum AI Daemon v{DAEMON_VERSION} — Self-Test")
        print(f"  GOD_CODE = {GOD_CODE} | PHI = {PHI}")
        print(f"══════════════════════════════════════════════════\n")

        results = daemon.self_test()
        for name, result in results.items():
            if name == "_summary":
                continue
            status = result.get("status", "?")
            icon = "✓" if status == "PASS" else "⚠" if status == "WARN" else "✗"
            detail = {k: v for k, v in result.items() if k != "status"}
            print(f"  {icon} {name:<20s} {status:<6s} {detail}")

        summary = results.get("_summary", {})
        print(f"\n  {'─' * 40}")
        p, t = summary.get("passed", 0), summary.get("total", 0)
        print(f"  {p}/{t} passed ({summary.get('elapsed_ms', 0):.0f}ms)")
        sys.exit(0 if summary.get("all_pass") else 1)

    elif args.health_check:
        # Try reading persisted state
        if STATE_PATH.exists():
            data = json.loads(STATE_PATH.read_text())
            health = data.get("health_score", 0)
            cycles = data.get("cycles_completed", 0)
            print(f"Health: {health:.3f} | Cycles: {cycles}")
            sys.exit(0 if health > 0.5 else 1)
        else:
            print("No state file — running quick check...")
            report = daemon._fidelity.run_fidelity_check()
            print(f"Fidelity: {report.grade} ({report.overall_fidelity:.3f})")
            sys.exit(0 if report.overall_fidelity > 0.5 else 1)

    elif args.status:
        status = daemon.status()
        print(json.dumps(status, indent=2, default=str))
        sys.exit(0)

    elif args.single_cycle:
        print(f"\n  L104 Quantum AI Daemon — Single Cycle Mode\n")
        daemon._scanner.full_scan()
        report = daemon._run_cycle()
        daemon._persist_state()
        print(f"\n  Cycle #{report.cycle_number}: "
              f"health={report.health_score:.3f} "
              f"fidelity={report.fidelity_grade} "
              f"harmony={report.harmony_score:.3f} "
              f"improved={report.files_improved}/{report.files_analyzed}")
        if report.warnings:
            for w in report.warnings[:10]:
                print(f"    ⚠ {w}")
        sys.exit(0)

    else:
        # Run as persistent daemon
        print(f"\n{'═' * 60}")
        print(f"  L104 Quantum AI Daemon v{DAEMON_VERSION}")
        print(f"  GOD_CODE = {GOD_CODE}")
        print(f"  PHI = {PHI}")
        print(f"  VOID_CONSTANT = {VOID_CONSTANT}")
        print(f"  Sacred Resonance = {SACRED_RESONANCE:.4f}")
        print(f"  Cycle Interval = {args.interval}s (adaptive)")
        print(f"  Auto-Fix = {'ON' if config.auto_fix_enabled else 'OFF'}")
        print(f"  PID = {os.getpid()}")
        print(f"{'═' * 60}\n")

        # Handle signals
        def _handle_sigterm(sig, frame):
            _logger.info("SIGTERM received — shutting down")
            daemon.stop()
            sys.exit(0)

        def _handle_sigusr1(sig, frame):
            """SIGUSR1 → dump status to log."""
            _logger.info(f"Status dump:\n{json.dumps(daemon.status(), indent=2, default=str)}")

        def _handle_sigusr2(sig, frame):
            """SIGUSR2 → force immediate cycle."""
            daemon.force_cycle()
            _logger.info("SIGUSR2 → forced cycle")

        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)
        try:
            signal.signal(signal.SIGUSR1, _handle_sigusr1)
            signal.signal(signal.SIGUSR2, _handle_sigusr2)
        except (AttributeError, OSError):
            pass  # SIGUSR not available on all platforms

        daemon.start()

        # Block main thread until stop
        try:
            while daemon._active:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            daemon.stop()


if __name__ == "__main__":
    _cli_main()
