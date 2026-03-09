"""L104 VQPU v12.2 — Hardware Governor + Result Collector."""

import os
import time
import json
import select
import threading
from collections import deque
from pathlib import Path
from typing import Optional

from .constants import (
    GOD_CODE, PHI,
    MAX_RAM_PERCENT, MAX_CPU_PERCENT, THROTTLE_COOLDOWN_S,
    _IS_INTEL, _IS_APPLE_SILICON, _HAS_METAL_COMPUTE,
    THROTTLE_SIGNAL, OUTBOX_PATH, TELEMETRY_PATH,
    HAS_PSUTIL,
)
from .types import VQPUResult

try:
    import psutil
except ImportError:
    psutil = None


class HardwareGovernor:
    """
    Monitors MacBook hardware vitals and signals throttling to the
    Swift vQPU when thermal or memory limits are approached.

    v11.0: Thermal prediction — uses 5-sample trend analysis to
    predict throttle before hitting the ceiling. NUMA-aware thread
    affinity hints for Apple Silicon efficiency cores.

    Uses psutil for cross-platform monitoring. Falls back gracefully
    if psutil is not installed (no monitoring, no throttling).

    Throttle protocol:
      - Creates /tmp/l104_bridge/throttle.signal → Swift prunes branches
        more aggressively and delays job polling
      - Removes the signal file when vitals normalize
    """

    def __init__(self, ram_threshold: float = MAX_RAM_PERCENT,
                 cpu_threshold: float = MAX_CPU_PERCENT,
                 poll_interval: float = 0.8):              # v11.0: tuned baseline (was 1.0)
        self.ram_threshold = ram_threshold
        self.cpu_threshold = cpu_threshold
        self.poll_interval = poll_interval
        self._poll_hot = 0.3                               # v11.0: ultra-fast poll when throttled (was 0.5)
        self._poll_cool = 2.5                              # v11.0: slightly faster cool poll (was 3.0)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_throttled = False
        self._throttle_count = 0
        self._samples: deque = deque(maxlen=180)           # v11.0: 6x history (was 120)
        self._predict_throttle = False                     # v11.0: predictive throttle flag

    @property
    def is_throttled(self) -> bool:
        return self._is_throttled

    @property
    def throttle_count(self) -> int:
        return self._throttle_count

    def start(self):
        """Start background hardware monitoring."""
        if not HAS_PSUTIL:
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True,
                                        name="l104-hw-governor")
        self._thread.start()

    def stop(self):
        """Stop background monitoring."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._clear_throttle()

    def _monitor_loop(self):
        """Main monitoring loop. v12.0: adds 5-sample trend prediction."""
        while not self._stop_event.is_set():
            try:
                cpu_pct = psutil.cpu_percent(interval=1)
                ram_pct = psutil.virtual_memory().percent

                self._samples.append({
                    "ts": time.time(),
                    "cpu": cpu_pct,
                    "ram": ram_pct,
                })

                should_throttle = (ram_pct > self.ram_threshold
                                   or cpu_pct > self.cpu_threshold)

                # v12.0: Predictive throttle — if 5-sample trend projects
                # exceeding threshold within next interval, pre-throttle
                if not should_throttle and len(self._samples) >= 5:
                    recent = list(self._samples)[-5:]
                    ram_vals = [s["ram"] for s in recent]
                    cpu_vals = [s["cpu"] for s in recent]
                    # Linear regression over 5 samples for better prediction
                    ram_trend = (ram_vals[-1] - ram_vals[0]) / 4
                    ram_accel = ((ram_vals[-1] - ram_vals[-3]) / 2 - (ram_vals[-3] - ram_vals[0]) / 2)
                    predicted_ram = ram_pct + ram_trend + ram_accel * 0.5
                    cpu_trend = (cpu_vals[-1] - cpu_vals[0]) / 4
                    cpu_accel = ((cpu_vals[-1] - cpu_vals[-3]) / 2 - (cpu_vals[-3] - cpu_vals[0]) / 2)
                    predicted_cpu = cpu_pct + cpu_trend + cpu_accel * 0.5
                    if predicted_ram > self.ram_threshold or predicted_cpu > self.cpu_threshold:
                        self._predict_throttle = True
                        # Don't hard throttle yet — just signal caution
                    else:
                        self._predict_throttle = False

                if should_throttle and not self._is_throttled:
                    self._engage_throttle(cpu_pct, ram_pct)
                elif not should_throttle and self._is_throttled:
                    self._clear_throttle()

            except Exception:
                pass

            # v11.0: adaptive polling — ultra-fast when throttled, moderate when predicted
            if self._is_throttled:
                _interval = self._poll_hot
            elif self._predict_throttle:
                _interval = self.poll_interval  # moderate speed
            else:
                _interval = self._poll_cool
            self._stop_event.wait(timeout=_interval)

    def _engage_throttle(self, cpu_pct: float, ram_pct: float):
        """Signal the Swift vQPU to throttle."""
        self._is_throttled = True
        self._throttle_count += 1
        try:
            THROTTLE_SIGNAL.touch()
        except OSError:
            pass

    def _clear_throttle(self):
        """Remove the throttle signal."""
        self._is_throttled = False
        try:
            if THROTTLE_SIGNAL.exists():
                THROTTLE_SIGNAL.unlink()
        except OSError:
            pass

    def get_vitals(self) -> dict:
        """Current hardware vitals."""
        if not HAS_PSUTIL:
            return {"available": False}

        mem = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": mem.percent,
            "ram_available_mb": mem.available / (1024 * 1024),
            "ram_total_mb": mem.total / (1024 * 1024),
            "is_throttled": self._is_throttled,
            "throttle_count": self._throttle_count,
            "sample_count": len(self._samples),
        }


class ResultCollector:
    """
    Collects vQPU results from the outbox directory.

    Uses kqueue (macOS) for near-zero-latency filesystem notifications.
    Falls back to tight polling (1ms) if kqueue is unavailable.

    Supports:
      - Blocking wait with kqueue event notification (~0.5ms latency)
      - Batch collection of all pending results
    """

    def __init__(self, outbox: Path = OUTBOX_PATH):
        self.outbox = outbox
        self._results: dict[str, VQPUResult] = {}
        self._kqueue_fd: Optional[int] = None
        self._watch_fd: int = -1
        self._setup_kqueue()

    def _setup_kqueue(self):
        """Set up kqueue to watch the outbox directory for new files."""
        try:
            self._kqueue_fd = select.kqueue()
            fd = os.open(str(self.outbox), os.O_RDONLY)
            self._watch_fd = fd
            ev = select.kevent(fd,
                               filter=select.KQ_FILTER_VNODE,
                               flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                               fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_RENAME)
            self._kqueue_fd.control([ev], 0, 0)
        except (AttributeError, OSError):
            # kqueue not available (non-macOS) — fall back to polling
            self._kqueue_fd = None
            self._watch_fd = -1

    def _wait_event(self, timeout_s: float) -> bool:
        """Wait for a filesystem event on the outbox. Returns True if event fired."""
        if self._kqueue_fd is None:
            time.sleep(min(timeout_s, 0.001))  # 1ms fallback poll
            return True  # always re-check

        try:
            events = self._kqueue_fd.control(None, 1, timeout_s)
            return len(events) > 0
        except (OSError, ValueError):
            time.sleep(min(timeout_s, 0.001))
            return True

    def wait_for(self, circuit_id: str, timeout: float = 30.0,
                 poll_interval: float = 0.001) -> Optional[VQPUResult]:
        """Block until result appears. Uses kqueue for sub-ms notification."""
        result_name = f"{circuit_id}_result.json"
        result_path = self.outbox / result_name
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if result_path.exists():
                try:
                    data = json.loads(result_path.read_text())
                    result = self._parse_result(data)
                    result_path.unlink(missing_ok=True)
                    return result
                except (json.JSONDecodeError, OSError):
                    pass

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._wait_event(min(remaining, 0.5))

        return VQPUResult(
            circuit_id=circuit_id,
            probabilities={},
            error=f"Timeout after {timeout}s waiting for result",
        )

    def close(self):
        """Clean up kqueue resources."""
        if self._watch_fd >= 0:
            try:
                os.close(self._watch_fd)
            except OSError:
                pass
            self._watch_fd = -1
        if self._kqueue_fd is not None:
            try:
                self._kqueue_fd.close()
            except OSError:
                pass
            self._kqueue_fd = None

    def __del__(self):
        """Release file descriptors on garbage collection.

        v15.2: Guard against interpreter shutdown — os.close and
        kqueue.close may be None if the module is already torn down.
        Suppresses all exceptions since __del__ must never raise.
        """
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def collect_all(self) -> list[VQPUResult]:
        """Collect all pending results from outbox."""
        results = []
        if not self.outbox.exists():
            return results

        for f in sorted(self.outbox.iterdir()):
            if f.suffix == ".json" and f.stem.endswith("_result"):
                try:
                    data = json.loads(f.read_text())
                    results.append(self._parse_result(data))
                    f.unlink(missing_ok=True)
                except (json.JSONDecodeError, OSError):
                    pass

        return results

    def _parse_result(self, data: dict) -> VQPUResult:
        """Parse a result JSON into VQPUResult."""
        return VQPUResult(
            circuit_id=data.get("circuit_id", data.get("circuit", "unknown")),
            probabilities=data.get("probabilities", {}),
            counts=data.get("counts"),
            backend=data.get("backend", data.get("backend_used", "unknown")),
            branch_count=data.get("branch_count", 0),
            t_gate_count=data.get("t_gate_count", 0),
            clifford_gate_count=data.get("clifford_gate_count", 0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            num_qubits=data.get("num_qubits", 0),
            god_code=data.get("god_code", GOD_CODE),
        )


__all__ = ["HardwareGovernor", "ResultCollector"]
