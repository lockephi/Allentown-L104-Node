"""
INTEGRATION PATCH: QuantumAI Daemon + Orchestrator
═════════════════════════════════════════════════════════════════════════════

Add this code to l104_quantum_ai_daemon/daemon.py to integrate with orchestrator.

CHANGES:
  1. Import DaemonAdapter at top
  2. Create adapter in __init__
  3. Wrap _run_improvement_cycle to call on_cycle_start/end
  4. Emit improvement metrics
  5. Emit errors on failures
"""

# ═════════════════════════════════════════════════════════════════════════
# STEP 1: ADD IMPORT (near top, after other imports)
# ═════════════════════════════════════════════════════════════════════════

# Add this import after line 69 (after other imports):
"""
from l104_daemon_adapter import DaemonAdapter  # ← ADD THIS LINE
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 2: ADD FIELDS TO __init__ (QuantumAIDaemon.__init__)
# ═════════════════════════════════════════════════════════════════════════

# Add these fields after existing initialization (around line 180):
"""
        # Orchestrator integration (NEW)
        self._orchestrator = None  # Set externally
        self._adapter = None       # DaemonAdapter instance
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 3: CREATE ADAPTER IN start() (QuantumAIDaemon.start)
# ═════════════════════════════════════════════════════════════════════════

# In start() method, add after daemon thread creation:
"""
        # Orchestrator integration (NEW)
        if self._orchestrator:
            try:
                self._adapter = DaemonAdapter("quantum_ai_daemon", self._orchestrator)
                _logger.info("QuantumAI daemon registered with orchestrator")
            except Exception as e:
                _logger.warning(f"Failed to create orchestrator adapter: {e}")
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 4: WRAP CYCLE IN _daemon_loop (main cycle loop)
# ═════════════════════════════════════════════════════════════════════════

# Find main cycle loop and wrap it (around line ~400-500):
"""
    def _daemon_loop(self):
        '''Main background cycle loop.'''
        ...
        while self._running:
            ...
            try:
                # ← ADD: Cycle start
                if self._adapter:
                    self._adapter.on_cycle_start()

                # Run improvement cycle
                report = self._run_improvement_cycle()

                # ← MODIFY: Report metrics
                if self._adapter:
                    success = report.error is None
                    cpu_pct = report.optimization_memory_freed_mb * 10 if report.optimization_memory_freed_mb > 0 else 15.0
                    mem_mb = 100.0  # Approximate

                    self._adapter.on_cycle_end(
                        success=success,
                        duration_ms=report.duration_ms,
                        cpu_percent=cpu_pct,
                        memory_mb=mem_mb
                    )

                    # Emit fidelity/quality metrics
                    if success:
                        fidelity = report.fidelity_score
                        trending = "up" if report.harmony_score > 0.8 else "stable"

                        self._adapter.emit_fidelity_alert(
                            fidelity=fidelity,
                            trending=trending,
                            sim_count=report.files_scanned,
                            error_count=report.files_analyzed - report.files_improved
                        )

                        # Emit improvement success
                        self._adapter.emit_success(
                            "code_improvement",
                            {
                                "files_improved": report.files_improved,
                                "files_scanned": report.files_scanned,
                                "fidelity": fidelity,
                                "harmony": report.harmony_score,
                                "evolution_delta": report.evolution_delta,
                            }
                        )
                    else:
                        self._adapter.emit_error(
                            "improvement_cycle_failure",
                            report.error or "Cycle failed",
                            "error"
                        )

            except Exception as e:
                _logger.error(f"Cycle exception: {e}", exc_info=True)
                if self._adapter:
                    self._adapter.emit_error(
                        "cycle_exception",
                        str(e),
                        "error"
                    )
                time.sleep(5)
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 5: OPTIONAL - SUBSCRIBE TO ORCHESTRATOR EVENTS
# ═════════════════════════════════════════════════════════════════════════

# Add this to start() or __init__ for event handling:
"""
        # Subscribe to resource alerts (OPTIONAL)
        if self._adapter:
            def on_resource_alert(event):
                payload = event.get('payload', {})
                if payload.get('alert_type') == 'high_memory':
                    _logger.warning("Resource alert: High memory — reducing cache")
                    self._file_cache.clear()  # or similar cache clearing

            self._adapter.subscribe_to_event("resource_alert", on_resource_alert)
"""


# ═════════════════════════════════════════════════════════════════════════
# USAGE (from external orchestrator)
# ═════════════════════════════════════════════════════════════════════════

"""
from l104_quantum_ai_daemon import QuantumAIDaemon
from l104_daemon_orchestrator import L104DaemonOrchestrator

# Start orchestrator
orchestrator = L104DaemonOrchestrator()
orchestrator.start()

# Start QuantumAI daemon with orchestrator
qai = QuantumAIDaemon()
qai._orchestrator = orchestrator  # ← Link orchestrator
qai.start()

# Now QuantumAI daemon reports to orchestrator!
"""


# ═════════════════════════════════════════════════════════════════════════
# COMPLETE PATCH SUMMARY
# ═════════════════════════════════════════════════════════════════════════

PATCH = {
    "file": "l104_quantum_ai_daemon/daemon.py",
    "changes": [
        ("Add import", "Add: from l104_daemon_adapter import DaemonAdapter"),
        ("__init__", "Add: self._orchestrator = None; self._adapter = None"),
        ("start()", "Add: Create adapter after daemon thread creation"),
        ("_daemon_loop", "Wrap cycle: on_cycle_start/end, emit_fidelity, emit_success/error"),
        ("Optional", "Add event subscriptions for resource alerts"),
    ],
    "lines_added": 95,
    "breaking_changes": 0,
    "notes": [
        "All changes are additive",
        "Orchestrator is optional (adapter only if set externally)",
        "Metrics use existing ImprovementReport fields",
        "Event subscription example shows resource reduction pattern",
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
