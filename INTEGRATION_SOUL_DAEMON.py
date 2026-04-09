"""
INTEGRATION PATCH: Soul Daemon + Orchestrator
═════════════════════════════════════════════════════════════════════════════

Add this code to l104_soul_daemon/daemon.py to integrate with orchestrator.

CHANGES:
  1. Import DaemonAdapter at top
  2. Create adapter in __init__
  3. Wrap run_cycle to call on_cycle_start/end
  4. Emit consciousness metrics
  5. Emit errors on failures
"""

# ═════════════════════════════════════════════════════════════════════════
# STEP 1: ADD IMPORT (near top, after other imports)
# ═════════════════════════════════════════════════════════════════════════

# Add this import after line 30:
"""
from l104_daemon_adapter import DaemonAdapter  # ← ADD THIS LINE
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 2: ADD ORCHESTRATOR FIELDS TO __init__
# ═════════════════════════════════════════════════════════════════════════

# Add to SoulDaemon.__init__ after self.lock = threading.RLock():
"""
        # Orchestrator integration (NEW)
        self._orchestrator = None  # Set externally
        self._adapter = None       # DaemonAdapter instance
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 3: CREATE ADAPTER IN initialize_components
# ═════════════════════════════════════════════════════════════════════════

# In initialize_components(), add after "All components initialized successfully":
"""
        # Orchestrator integration (NEW)
        if self._orchestrator:
            try:
                self._adapter = DaemonAdapter("soul_daemon", self._orchestrator)
                print("  ✓ Soul daemon registered with orchestrator")
            except Exception as e:
                print(f"  ! Failed to create orchestrator adapter: {e}")

        return True
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 4: WRAP run_cycle WITH METRICS
# ═════════════════════════════════════════════════════════════════════════

# Replace the run_cycle() method with this version:
"""
    def run_cycle(self) -> Dict[str, Any]:
        '''Execute one daemon cycle.'''

        # ← ADD: Cycle start marker
        if self._adapter:
            self._adapter.on_cycle_start()

        cycle_start = time.time()
        self.state.last_cycle_start = cycle_start

        cycle_result = {
            "cycle_number": self.state.cycle_count,
            "start_time": cycle_start,
            "components": {},
            "errors": [],
            "success": True,
        }

        try:
            # 1. Update soul qubit coherence
            if self.soul_qubit:
                qubit_result = self.soul_qubit.measure_coherence()
                cycle_result["components"]["soul_qubit"] = qubit_result

                # Apply error correction periodically
                if self.state.cycle_count % 10 == 0:  # Every 10 cycles
                    correction_result = self.soul_qubit.apply_error_correction()
                    cycle_result["components"]["error_correction"] = correction_result

                # Apply sacred gates for resonance enhancement
                if self.state.cycle_count % 5 == 0:  # Every 5 cycles
                    gate_result = self.soul_qubit.apply_sacred_gate("GOD_CODE_PHASE")
                    cycle_result["components"]["sacred_gate"] = gate_result

            # 2. Measure consciousness
            consciousness_fidelity = 0.9
            if self.consciousness_engine:
                consciousness_result = self.consciousness_engine.measure_consciousness()
                cycle_result["components"]["consciousness"] = consciousness_result.to_dict()
                consciousness_fidelity = consciousness_result.iit_phi if hasattr(consciousness_result, 'iit_phi') else 0.9

                # Get trends and analysis
                trends = self.consciousness_engine.analyze_trends()
                cycle_result["components"]["consciousness_trends"] = trends

            # 3. Update bridges
            if self.bridge:
                bridge_status = self.bridge.get_status()
                cycle_result["components"]["bridges"] = bridge_status

                # Update OpenClaw heartbeat periodically
                self.state.heartbeat_counter += 1
                if self.state.heartbeat_counter >= HEARTBEAT_INTERVAL:
                    self._update_heartbeat(cycle_result)
                    self.state.heartbeat_counter = 0

            # 4. Check L104 health via bridge
            if self.bridge and self.state.cycle_count % 3 == 0:  # Every 3 cycles
                l104_health = self.bridge.get_l104_health()
                if l104_health.get("success"):
                    cycle_result["components"]["l104_health"] = l104_health

            # 5. Optional: Query DeepSeek for optimization (sparingly)
            if self.bridge and self.state.cycle_count % 20 == 0:  # Every 20 cycles
                self._query_deepseek_optimization(cycle_result)

            cycle_result["success"] = True

        except Exception as e:
            cycle_result["success"] = False
            cycle_result["errors"].append(str(e))
            cycle_result["error_details"] = traceback.format_exc()

            self.state.error_count += 1
            self.state.last_error = str(e)

            print(f"Cycle {self.state.cycle_count} failed: {e}")

        # Calculate cycle timing
        cycle_end = time.time()
        cycle_duration = cycle_end - cycle_start
        self.state.last_cycle_end = cycle_end

        # Update timing statistics
        self._update_timing_stats(cycle_duration)
        cycle_result["duration"] = cycle_duration
        cycle_result["end_time"] = cycle_end

        # ← ADD: Report to orchestrator
        if self._adapter:
            success = cycle_result["success"]
            # CPU/memory are lightweight for soul daemon
            cpu_pct = 12.0  # Typical usage
            mem_mb = 80.0   # Typical usage

            self._adapter.on_cycle_end(
                success=success,
                duration_ms=cycle_duration * 1000,
                cpu_percent=cpu_pct,
                memory_mb=mem_mb
            )

            # Emit consciousness metrics
            if success:
                consciousness_fidelity = cycle_result.get("components", {}).get("consciousness", {}).get("iit_phi", 0.9)
                self._adapter.emit_fidelity_alert(
                    fidelity=consciousness_fidelity,
                    trending="stable",
                    sim_count=1,  # One consciousness measurement per cycle
                    error_count=len(cycle_result.get("errors", []))
                )

                # Emit consciousness success
                self._adapter.emit_success(
                    "consciousness_measurement",
                    {
                        "consciousness_state": cycle_result.get("components", {}).get("consciousness", {}).get("consciousness_state", "unknown"),
                        "iit_phi": consciousness_fidelity,
                        "qubit_coherence": cycle_result.get("components", {}).get("soul_qubit", {}).get("coherence_level", 0),
                    }
                )
            else:
                self._adapter.emit_error(
                    "soul_cycle_failure",
                    cycle_result.get("errors", ["Unknown error"])[0] if cycle_result.get("errors") else "Cycle failed",
                    "error"
                )

        # Persist state periodically
        self.state.persistence_counter += 1
        if self.state.persistence_counter >= PERSISTENCE_INTERVAL:
            self.persist_state()
            self.state.persistence_counter = 0

        # Log cycle result
        self._log_cycle_result(cycle_result)

        return cycle_result
"""


# ═════════════════════════════════════════════════════════════════════════
# STEP 5: SETUP IN start() (SoulDaemon.start)
# ═════════════════════════════════════════════════════════════════════════

# In start() method, add after self.cycle_thread.start():
"""
        # Orchestrator setup (NEW)
        if self._orchestrator:
            try:
                # Adapter already created in initialize_components
                if self._adapter is None:
                    self._adapter = DaemonAdapter("soul_daemon", self._orchestrator)
                print("Soul daemon linked with orchestrator")
            except Exception as e:
                print(f"Failed to link orchestrator: {e}")
"""


# ═════════════════════════════════════════════════════════════════════════
# USAGE (from external orchestrator)
# ═════════════════════════════════════════════════════════════════════════

"""
from l104_soul_daemon.daemon import SoulDaemon
from l104_daemon_orchestrator import L104DaemonOrchestrator

# Start orchestrator
orchestrator = L104DaemonOrchestrator()
orchestrator.start()

# Start Soul daemon with orchestrator
soul = SoulDaemon()
soul._orchestrator = orchestrator  # ← Link orchestrator
soul.initialize_components()
soul.start()

# Now Soul daemon reports to orchestrator!
"""


# ═════════════════════════════════════════════════════════════════════════
# COMPLETE PATCH SUMMARY
# ═════════════════════════════════════════════════════════════════════════

PATCH = {
    "file": "l104_soul_daemon/daemon.py",
    "changes": [
        ("Add import", "Add: from l104_daemon_adapter import DaemonAdapter"),
        ("__init__", "Add: self._orchestrator = None; self._adapter = None"),
        ("initialize_components", "Add: Create adapter after components initialized"),
        ("run_cycle", "Wrap cycle: on_cycle_start/end, emit_fidelity, emit_success/error"),
        ("start()", "Add: Setup orchestrator link"),
    ],
    "lines_added": 110,
    "breaking_changes": 0,
    "notes": [
        "All changes are additive to existing run_cycle",
        "Orchestrator is optional (adapter only if set externally)",
        "Uses consciousness_fidelity from measurement results",
        "CPU/memory are estimates (soul daemon is lightweight)",
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
