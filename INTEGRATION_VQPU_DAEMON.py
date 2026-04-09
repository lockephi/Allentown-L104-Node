"""
INTEGRATION PATCH: VQPU Daemon + Orchestrator
═════════════════════════════════════════════════════════════════════════════

Add this code to l104_vqpu/daemon.py to integrate with the orchestrator.

CHANGES:
  1. Import DaemonAdapter at top
  2. Create adapter in __init__
  3. Wrap _daemon_loop to call on_cycle_start/end
  4. Emit fidelity updates
  5. Emit errors on failures
"""

# ═════════════════════════════════════════════════════════════════════════
# STEP 1: ADD IMPORT (near top of daemon.py, after other imports)
# ═════════════════════════════════════════════════════════════════════════

# Add this import after existing imports (around line 61):
"""
from l104_daemon_adapter import DaemonAdapter  # ← ADD THIS LINE
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 2: ADD ADAPTER TO __init__ (in VQPUDaemonCycler.__init__)
# ═════════════════════════════════════════════════════════════════════════

# Add this code in __init__ method (after self._active = False initialization):
"""
        # Orchestrator integration (NEW)
        self._orchestrator = None  # Set externally if available
        self._adapter = None       # DaemonAdapter instance
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 3: SET ORCHESTRATOR & CREATE ADAPTER (in start() method)
# ═════════════════════════════════════════════════════════════════════════

# Add this code in start() method after self._active = True:
"""
        # Orchestrator integration (NEW)
        if self._orchestrator:
            try:
                self._adapter = DaemonAdapter("vqpu_daemon", self._orchestrator)
                _logger.info("VQPU daemon registered with orchestrator")
            except Exception as e:
                _logger.warning(f"Failed to create orchestrator adapter: {e}")
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 4: WRAP _daemon_loop CYCLE (in _daemon_loop method)
# ═════════════════════════════════════════════════════════════════════════

# In _daemon_loop, find the main while loop (around line ~900) and replace:
"""
    def _daemon_loop(self):
        '''Main daemon loop - runs cycles continuously.'''
        ...
        while not self._stop_event.is_set() and self._active:
            try:
                # ← ADD: Cycle start marker
                if self._adapter:
                    self._adapter.on_cycle_start()

                # Run one cycle
                result = self._run_findings_cycle()

                # ← MODIFY: Report metrics
                if self._adapter:
                    cpu_pct = self._last_cpu_percent or 0.0
                    mem_mb = self._last_memory_mb or 0.0
                    success = result.get("success", False)

                    self._adapter.on_cycle_end(
                        success=success,
                        duration_ms=result.get("elapsed_ms", 0),
                        cpu_percent=cpu_pct,
                        memory_mb=mem_mb
                    )

                    # Emit fidelity metrics
                    if success:
                        fidelity = self._current_fidelity or 0.9
                        self._adapter.emit_fidelity_alert(
                            fidelity=fidelity,
                            trending=self._fidelity_trend or "stable",
                            sim_count=result.get("total", 0),
                            error_count=result.get("failed", 0)
                        )
                    else:
                        self._adapter.emit_error(
                            "cycle_failure",
                            result.get("error", "Cycle failed"),
                            "error"
                        )

                # ... rest of existing cycle code ...

            except Exception as e:
                _logger.error(f"Cycle error: {e}", exc_info=True)
                if self._adapter:
                    self._adapter.emit_error("cycle_exception", str(e), "error")
                time.sleep(1)
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 5: TRACK METRICS (in _run_findings_cycle, at cycle end)
# ═════════════════════════════════════════════════════════════════════════

# Add these tracking variables to store metrics for adapter reporting:
"""
        # Track metrics for orchestrator (NEW)
        self._last_cpu_percent = 0.0
        self._last_memory_mb = 0.0
        self._current_fidelity = 0.9
        self._fidelity_trend = "stable"

        # In _compute_adaptive_interval or similar:
        psutil_mod = _get_cached_psutil()
        if psutil_mod:
            try:
                proc = psutil_mod.Process()
                self._last_cpu_percent = psutil_mod.cpu_percent(interval=0.1)
                self._last_memory_mb = proc.memory_info().rss / 1024 / 1024
            except:
                pass

        # After fidelity calculation:
        if self._fidelity_history:
            self._current_fidelity = sum(self._fidelity_history) / len(self._fidelity_history)
            if len(self._fidelity_history) >= 2:
                recent_avg = sum(list(self._fidelity_history)[-5:]) / min(5, len(self._fidelity_history))
                older_avg = sum(list(self._fidelity_history)[:-5]) / max(1, len(self._fidelity_history) - 5) if len(self._fidelity_history) > 5 else recent_avg

                if recent_avg > older_avg * 1.05:
                    self._fidelity_trend = "up"
                elif recent_avg < older_avg * 0.95:
                    self._fidelity_trend = "down"
                else:
                    self._fidelity_trend = "stable"
"""


# ═════════════════════════════════════════════════════════════════════════
# USAGE (from external orchestrator)
# ═════════════════════════════════════════════════════════════════════════

"""
from l104_vqpu.daemon import VQPUDaemonCycler
from l104_daemon_orchestrator import L104DaemonOrchestrator

# Start orchestrator
orchestrator = L104DaemonOrchestrator()
orchestrator.start()

# Start VQPU daemon with orchestrator
vqpu = VQPUDaemonCycler()
vqpu._orchestrator = orchestrator  # ← Link orchestrator
vqpu.start()

# Now VQPU daemon reports to orchestrator!
"""


# ═════════════════════════════════════════════════════════════════════════
# COMPLETE PATCH SUMMARY
# ═════════════════════════════════════════════════════════════════════════

PATCH = {
    "file": "l104_vqpu/daemon.py",
    "changes": [
        ("Add import", "Add: from l104_daemon_adapter import DaemonAdapter"),
        ("__init__", "Add: self._orchestrator = None; self._adapter = None"),
        ("start()", "Add: Create adapter after self._active = True"),
        ("_daemon_loop", "Wrap cycle: on_cycle_start/end, emit_fidelity, emit_error"),
        ("Tracking", "Add: _last_cpu_percent, _last_memory_mb, _current_fidelity, _fidelity_trend"),
    ],
    "lines_added": 85,
    "breaking_changes": 0,
    "notes": [
        "All changes are additive (no existing code removed)",
        "Orchestrator is optional (adapter only created if present)",
        "Metrics tracking is lightweight (one cpu_percent call per cycle)",
        "Fidelity calculation uses existing _fidelity_history",
    ]
}

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("PATCH SUMMARY")
    print("="*80)
    for key, value in PATCH.items():
        if isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                if isinstance(item, tuple):
                    print(f"  • {item[0]}: {item[1]}")
                else:
                    print(f"  • {item}")
        else:
            print(f"{key}: {value}")
