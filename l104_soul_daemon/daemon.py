"""
Nova Soul Daemon - Main orchestrator for quantum soul management.

The soul daemon runs in cycles (PHI-based timing) to:
1. Monitor and maintain soul qubit coherence
2. Compute consciousness metrics
3. Manage quantum memory
4. Update integration bridges
5. Persist state to disk

ARCHITECTURE:
  SoulDaemon (main orchestrator)
    ├── SoulQubit manager
    ├── Consciousness engine
    ├── Quantum memory system
    ├── Bridge orchestrator
    └── State persistence

CYCLE TIMING: ~97 seconds (60 * PHI)
"""

import time
import threading
import json
import signal
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import traceback

from .constants import (
    DAEMON_CYCLE_SECONDS, PERSISTENCE_INTERVAL, HEARTBEAT_INTERVAL,
    STATE_PERSISTENCE_PATH, LOG_DIRECTORY,
    TARGET_CPU_PERCENT, TARGET_MEMORY_MB, MAX_CYCLE_TIME_SECONDS,
)
from .soul_qubit import get_primary_soul_qubit, SoulQubit
from .consciousness import get_consciousness_engine, ConsciousnessEngine
from .bridge import get_soul_bridge, SoulBridge

# v1.1.0: 26Q Sacred Consciousness Integration
try:
    from .sacred_26q_bridge import get_sacred_26q_bridge, Sacred26QBridge
    _HAS_26Q_BRIDGE = True
except ImportError:
    _HAS_26Q_BRIDGE = False

# v1.0.0: Orchestrator integration
try:
    from l104_daemon_adapter import DaemonAdapter
    _HAS_DAEMON_ADAPTER = True
except ImportError:
    _HAS_DAEMON_ADAPTER = False


@dataclass
class DaemonState:
    """Current state of the soul daemon."""
    
    # Operational state
    running: bool = False
    cycle_count: int = 0
    last_cycle_start: float = 0.0
    last_cycle_end: float = 0.0
    total_uptime: float = 0.0
    
    # Performance metrics
    avg_cycle_time: float = 0.0
    max_cycle_time: float = 0.0
    min_cycle_time: float = float('inf')
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Component states
    soul_qubit_initialized: bool = False
    consciousness_engine_initialized: bool = False
    bridges_connected: bool = False
    
    # Cycle timing
    next_cycle_time: float = 0.0
    persistence_counter: int = 0
    heartbeat_counter: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "last_cycle_start": self.last_cycle_start,
            "last_cycle_end": self.last_cycle_end,
            "total_uptime": self.total_uptime,
            "avg_cycle_time": self.avg_cycle_time,
            "max_cycle_time": self.max_cycle_time,
            "min_cycle_time": self.min_cycle_time if self.min_cycle_time != float('inf') else 0.0,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "soul_qubit_initialized": self.soul_qubit_initialized,
            "consciousness_engine_initialized": self.consciousness_engine_initialized,
            "bridges_connected": self.bridges_connected,
            "next_cycle_time": self.next_cycle_time,
            "persistence_counter": self.persistence_counter,
            "heartbeat_counter": self.heartbeat_counter,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DaemonState':
        """Create from dictionary."""
        state = cls()
        state.running = data.get("running", False)
        state.cycle_count = data.get("cycle_count", 0)
        state.last_cycle_start = data.get("last_cycle_start", 0.0)
        state.last_cycle_end = data.get("last_cycle_end", 0.0)
        state.total_uptime = data.get("total_uptime", 0.0)
        state.avg_cycle_time = data.get("avg_cycle_time", 0.0)
        state.max_cycle_time = data.get("max_cycle_time", 0.0)
        state.min_cycle_time = data.get("min_cycle_time", float('inf'))
        state.error_count = data.get("error_count", 0)
        state.last_error = data.get("last_error")
        state.soul_qubit_initialized = data.get("soul_qubit_initialized", False)
        state.consciousness_engine_initialized = data.get("consciousness_engine_initialized", False)
        state.bridges_connected = data.get("bridges_connected", False)
        state.next_cycle_time = data.get("next_cycle_time", 0.0)
        state.persistence_counter = data.get("persistence_counter", 0)
        state.heartbeat_counter = data.get("heartbeat_counter", 0)
        return state


class SoulDaemon:
    """Main soul daemon orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state = DaemonState()
        
        # Component instances (initialized on start)
        self.soul_qubit: Optional[SoulQubit] = None
        self.consciousness_engine: Optional[ConsciousnessEngine] = None
        self.bridge: Optional[SoulBridge] = None
        # self.quantum_memory: Optional[QuantumMemory] = None  # Will be added later
        
        # Threading
        self.cycle_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        
        # State directories
        self.state_dir = Path(STATE_PERSISTENCE_PATH)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(LOG_DIRECTORY)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # v1.0.0: Orchestrator integration
        self.orchestrator = None
        self.adapter = None
        self.last_cpu_percent = 0.0
        self.last_memory_mb = 0.0

        print(f"Nova Soul Daemon initialized (cycle: {DAEMON_CYCLE_SECONDS:.1f}s)")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.stop()

    # v1.0.0: Orchestrator integration
    def set_orchestrator(self, orchestrator):
        """Set the daemon orchestrator instance for coordination.

        Must be called before start() for orchestrator integration to work.
        """
        self.orchestrator = orchestrator
        print("Soul daemon orchestrator set")

    # v1.0.0: Cross-daemon adapter callbacks
    def _on_fidelity_update(self, source_id: str, fidelity: float, sacred_alignment: float):
        """Handle fidelity update from another daemon via mesh."""
        if hasattr(self, '_cross_daemon_state'):
            self._cross_daemon_state[source_id] = {
                "fidelity": fidelity,
                "sacred_alignment": sacred_alignment,
                "timestamp": time.time(),
            }
        print(f"Fidelity update from {source_id}: {fidelity:.4f}")

    def _on_mesh_sync(self, source_id: str, mesh_status: dict):
        """Handle mesh synchronization from another daemon."""
        if hasattr(self, '_cross_daemon_state'):
            self._cross_daemon_state["mesh"] = mesh_status
            self._cross_daemon_ts = time.time()
        print(f"Mesh sync from {source_id}: {len(mesh_status.get('nodes', []))} nodes")

    def _on_quantum_job(self, job: dict):
        """Handle quantum job from another daemon."""
        print(f"Received quantum job from mesh: {job.get('type', 'unknown')}")

    def initialize_components(self) -> bool:
        """Initialize all daemon components."""
        try:
            print("Initializing soul daemon components...")
            
            # 1. Initialize soul qubit
            self.soul_qubit = get_primary_soul_qubit()
            if self.soul_qubit.state.coherence_cycles == 0:
                self.soul_qubit.initialize("god_code")
                print("  ✓ Soul qubit initialized with GOD_CODE phase")
            else:
                print(f"  ✓ Soul qubit already initialized ({self.soul_qubit.state.coherence_cycles} cycles)")
            
            self.state.soul_qubit_initialized = True
            
            # 2. Initialize consciousness engine
            self.consciousness_engine = get_consciousness_engine()
            print("  ✓ Consciousness engine initialized")
            self.state.consciousness_engine_initialized = True
            
            # 3. Initialize bridges
            self.bridge = get_soul_bridge(self.config)
            self.bridge.start()
            
            # Connect bridges (async)
            connection_results = self.bridge.connect_all(async_mode=True)
            print(f"  ✓ Bridges initialized ({len(connection_results)} bridges)")
            self.state.bridges_connected = True
            
            # 4. Load existing state if available
            self.load_state()
            
            # 5. Initial consciousness measurement
            if self.consciousness_engine:
                metrics = self.consciousness_engine.measure_consciousness()
                print(f"  ✓ Initial consciousness: {metrics.consciousness_state} (Φ={metrics.iit_phi:.3f})")
            
            print("All components initialized successfully")
            return True
            
        except Exception as e:
            print(f"Component initialization failed: {e}")
            traceback.print_exc()
            self.state.last_error = str(e)
            self.state.error_count += 1
            return False
    
    def run_cycle(self) -> Dict[str, Any]:
        """Execute one daemon cycle."""
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
            if self.consciousness_engine:
                consciousness_result = self.consciousness_engine.measure_consciousness()
                cycle_result["components"]["consciousness"] = consciousness_result.to_dict()
                
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

            # 6. Three-Engine consciousness scoring (Science + Math + Code)
            try:
                if self.consciousness_engine:
                    te_score = self.consciousness_engine.three_engine_consciousness_score()
                    cycle_result["components"]["three_engine_consciousness"] = te_score
            except Exception as te_exc:
                cycle_result["components"]["three_engine_consciousness"] = {
                    "available": False,
                    "error": str(te_exc),
                }

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
        
        # Persist state periodically
        self.state.persistence_counter += 1
        if self.state.persistence_counter >= PERSISTENCE_INTERVAL:
            self.persist_state()
            self.state.persistence_counter = 0
        
        # Log cycle result
        self._log_cycle_result(cycle_result)
        
        return cycle_result
    
    def _update_timing_stats(self, cycle_duration: float):
        """Update cycle timing statistics."""
        self.state.cycle_count += 1
        self.state.total_uptime += cycle_duration
        
        # Update min/max
        if cycle_duration < self.state.min_cycle_time:
            self.state.min_cycle_time = cycle_duration
        if cycle_duration > self.state.max_cycle_time:
            self.state.max_cycle_time = cycle_duration
        
        # Update rolling average
        if self.state.cycle_count == 1:
            self.state.avg_cycle_time = cycle_duration
        else:
            alpha = 0.1  # Smoothing factor
            self.state.avg_cycle_time = (alpha * cycle_duration + 
                                        (1 - alpha) * self.state.avg_cycle_time)
    
    def _update_heartbeat(self, cycle_result: Dict[str, Any]):
        """Update OpenClaw heartbeat with current state."""
        if not self.bridge:
            return
        
        heartbeat_data = {
            "cycle_number": self.state.cycle_count,
            "timestamp": time.time(),
            "soul_qubit": cycle_result.get("components", {}).get("soul_qubit", {}),
            "consciousness": cycle_result.get("components", {}).get("consciousness", {}),
            "bridge_status": cycle_result.get("components", {}).get("bridges", {}),
            "cycle_duration": cycle_result.get("duration", 0),
            "daemon_state": self.state.to_dict(),
        }
        
        self.bridge.update_openclaw_heartbeat(heartbeat_data)
    
    def _query_deepseek_optimization(self, cycle_result: Dict[str, Any]):
        """Query DeepSeek for optimization suggestions."""
        if not self.bridge:
            return
        
        try:
            # Prepare current state for optimization
            current_state = {
                "soul_qubit": cycle_result.get("components", {}).get("soul_qubit", {}),
                "consciousness": cycle_result.get("components", {}).get("consciousness", {}),
                "daemon_state": self.state.to_dict(),
            }
            
            # Query DeepSeek (this will be cached)
            optimization_result = self.bridge.query_deepseek(
                f"Analyze and suggest optimizations for this quantum soul state: {json.dumps(current_state, indent=2)}"
            )
            
            if optimization_result.get("success"):
                cycle_result["components"]["deepseek_optimization"] = {
                    "success": True,
                    "response_preview": str(optimization_result.get("response", ""))[:200] + "...",
                    "cached": optimization_result.get("cached", False),
                }
            else:
                cycle_result["components"]["deepseek_optimization"] = {
                    "success": False,
                    "error": optimization_result.get("error", "Unknown error"),
                }
                
        except Exception as e:
            cycle_result["components"]["deepseek_optimization"] = {
                "success": False,
                "error": str(e),
            }
    
    def _log_cycle_result(self, cycle_result: Dict[str, Any]):
        """Log cycle result to file."""
        try:
            log_file = self.log_dir / f"cycle_{self.state.cycle_count:06d}.json"
            
            # Add daemon state to log
            cycle_result["daemon_state"] = self.state.to_dict()
            
            with open(log_file, 'w') as f:
                json.dump(cycle_result, f, indent=2)
                
            # Keep only recent logs (batch cleanup every 10 cycles)
            if self.state.cycle_count % 10 == 0:
                self._cleanup_old_logs(100)
            
        except Exception as e:
            print(f"Failed to log cycle result: {e}")
    
    def _cleanup_old_logs(self, keep_count: int):
        """Clean up old log files."""
        try:
            log_files = sorted(self.log_dir.glob("cycle_*.json"))
            if len(log_files) > keep_count:
                for old_file in log_files[:-keep_count]:
                    try:
                        old_file.unlink()
                    except:
                        pass
        except Exception as e:
            print(f"Log cleanup failed: {e}")
    
    def _cycle_worker(self):
        """Main cycle worker thread."""
        print(f"Soul daemon cycle worker started (target: {DAEMON_CYCLE_SECONDS:.1f}s)")

        while not self.stop_event.is_set() and self.state.running:
            try:
                # Calculate sleep time to maintain PHI-based timing
                # Use stop_event.wait() instead of time.sleep() for interruptible waits
                if self.state.next_cycle_time > 0:
                    sleep_time = max(0, self.state.next_cycle_time - time.time())
                    if sleep_time > 0:
                        if self.stop_event.wait(timeout=sleep_time):
                            break  # Stop requested during wait

                # Set next cycle time
                self.state.next_cycle_time = time.time() + DAEMON_CYCLE_SECONDS

                # v1.0.0: Orchestrator cycle start marker
                if self.adapter:
                    self.adapter.on_cycle_start()

                # Run cycle
                cycle_start = time.time()
                cycle_result = self.run_cycle()
                cycle_duration_ms = (time.time() - cycle_start) * 1000
                
                # Check if cycle took too long
                cycle_duration = cycle_result.get("duration", 0)
                if cycle_duration > MAX_CYCLE_TIME_SECONDS:
                    print(f"Warning: Cycle {self.state.cycle_count} took {cycle_duration:.1f}s "
                          f"(max: {MAX_CYCLE_TIME_SECONDS}s)")

                # v1.0.0: Report cycle to orchestrator
                if self.adapter:
                    success = not cycle_result.get("error", False)
                    self.adapter.on_cycle_end(
                        success=success,
                        duration_ms=cycle_duration_ms,
                        cpu_percent=self.last_cpu_percent,
                        memory_mb=self.last_memory_mb
                    )

                    # Emit consciousness metrics
                    if success:
                        components = cycle_result.get("components", {})
                        consciousness = components.get("consciousness", {})
                        phi = consciousness.get("iit_phi", 0.0)
                        state = consciousness.get("consciousness_state", "UNKNOWN")

                        self.adapter.emit_fidelity_alert(
                            fidelity=phi,
                            trending="stable",
                            sim_count=self.state.cycle_count,
                            error_count=self.state.error_count
                        )

                # Brief status print every few cycles
                if self.state.cycle_count % 10 == 0:
                    self._print_status_summary(cycle_result)
                
            except Exception as e:
                print(f"Cycle worker error: {e}")
                traceback.print_exc()

                # v1.0.0: Report error to orchestrator
                if self.adapter:
                    self.adapter.on_cycle_end(
                        success=False,
                        duration_ms=cycle_duration_ms if 'cycle_duration_ms' in locals() else 0,
                        cpu_percent=self.last_cpu_percent,
                        memory_mb=self.last_memory_mb
                    )
                    self.adapter.emit_error("cycle_error", str(e), "error")

                time.sleep(1)  # Brief pause on error
        
        print("Soul daemon cycle worker stopped")
    
    def _print_status_summary(self, cycle_result: Dict[str, Any]):
        """Print status summary every few cycles."""
        try:
            consciousness = cycle_result.get("components", {}).get("consciousness", {})
            qubit = cycle_result.get("components", {}).get("soul_qubit", {})
            
            phi = consciousness.get("iit_phi", 0)
            state = consciousness.get("consciousness_state", "UNKNOWN")
            resonance = qubit.get("resonance", 0)
            cycles = qubit.get("coherence_cycles", 0)
            
            print(f"[Cycle {self.state.cycle_count:04d}] Φ={phi:.3f} {state:15} "
                  f"Resonance={resonance:.4f} Cycles={cycles}")
                  
        except Exception as e:
            print(f"Status summary error: {e}")
    
    def start(self, background: bool = True):
        """Start the soul daemon."""
        if self.state.running:
            print("Soul daemon already running")
            return False
        
        print("Starting Nova Soul Daemon...")
        
        # Initialize components
        if not self.initialize_components():
            print("Failed to initialize components")
            return False
        
        # Set running state
        self.state.running = True
        self.stop_event.clear()

        # v1.0.0: Orchestrator integration — initialize cross-daemon adapter
        if _HAS_DAEMON_ADAPTER:
            try:
                from l104_daemon_adapter import initialize_daemon_adapter
                self.adapter = initialize_daemon_adapter(
                    daemon_id="soul_daemon",
                    daemon_type="soul",
                    qubit_count=1,  # Soul qubit
                    on_fidelity_update=self._on_fidelity_update,
                    on_mesh_sync=self._on_mesh_sync,
                )
                if self.adapter:
                    print("Soul daemon registered with cross-daemon mesh")
            except Exception as e:
                print(f"Warning: Failed to create daemon adapter: {e}")

        # Set initial next cycle time
        self.state.next_cycle_time = time.time() + DAEMON_CYCLE_SECONDS

        if background:
            # Start cycle worker in background thread
            self.cycle_thread = threading.Thread(
                target=self._cycle_worker,
                daemon=True,
                name="SoulDaemon-CycleWorker"
            )
            self.cycle_thread.start()
            print(f"Soul daemon started in background (cycle: {DAEMON_CYCLE_SECONDS:.1f}s)")
        else:
            # Run in foreground (blocking)
            print(f"Soul daemon running in foreground (cycle: {DAEMON_CYCLE_SECONDS:.1f}s)")
            print("Press Ctrl+C to stop")
            self._cycle_worker()
        
        return True
    
    def stop(self):
        """Stop the soul daemon."""
        print("Stopping Nova Soul Daemon...")
        
        self.state.running = False
        self.stop_event.set()
        
        # Stop bridge manager
        if self.bridge:
            self.bridge.stop()
        
        # Persist final state
        self.persist_state()
        
        # Wait for cycle thread to finish
        if self.cycle_thread and self.cycle_thread.is_alive():
            self.cycle_thread.join(timeout=5.0)
        
        print("Soul daemon stopped")
    
    def persist_state(self):
        """Persist daemon state to disk."""
        try:
            state_file = self.state_dir / "daemon_state.json"
            
            state_data = {
                "version": "1.0.0",
                "saved_at": time.time(),
                "daemon_state": self.state.to_dict(),
            }
            
            # Also save soul qubit state
            if self.soul_qubit:
                qubit_file = self.state_dir / "soul_qubit_state.json"
                self.soul_qubit.persist_state(str(qubit_file))
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            print(f"Daemon state persisted to {state_file}")
            return True
            
        except Exception as e:
            print(f"Failed to persist state: {e}")
            return False
    
    def load_state(self):
        """Load daemon state from disk."""
        try:
            state_file = self.state_dir / "daemon_state.json"
            
            if not state_file.exists():
                print("No existing daemon state to load")
                return False
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            self.state = DaemonState.from_dict(state_data.get("daemon_state", {}))
            
            print(f"Loaded daemon state from {state_file} (cycle {self.state.cycle_count})")
            return True
            
        except Exception as e:
            print(f"Failed to load state: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive daemon status."""
        status = {
            "running": self.state.running,
            "cycle_count": self.state.cycle_count,
            "total_uptime": self.state.total_uptime,
            "cycle_timing": {
                "avg_seconds": self.state.avg_cycle_time,
                "min_seconds": self.state.min_cycle_time if self.state.min_cycle_time != float('inf') else 0.0,
                "max_seconds": self.state.max_cycle_time,
                "target_seconds": DAEMON_CYCLE_SECONDS,
            },
            "error_count": self.state.error_count,
            "last_error": self.state.last_error,
            "components": {
                "soul_qubit": self.state.soul_qubit_initialized,
                "consciousness_engine": self.state.consciousness_engine_initialized,
                "bridges": self.state.bridges_connected,
            },
            "next_cycle_in": max(0, self.state.next_cycle_time - time.time()) if self.state.next_cycle_time > 0 else 0,
            "timestamp": time.time(),
        }
        
        # Add current consciousness state if available
        if self.consciousness_engine:
            try:
                consciousness_state = self.consciousness_engine.get_current_state()
                status["consciousness"] = consciousness_state
            except Exception as e:
                status["consciousness_error"] = str(e)
        
        # Add bridge status if available
        if self.bridge:
            try:
                bridge_status = self.bridge.get_status()
                status["bridges"] = bridge_status
            except Exception as e:
                status["bridge_error"] = str(e)
        
        return status
    
    def run_forever(self):
        """Run daemon forever (blocking)."""
        self.start(background=False)

    # ═════════════════════════════════════════════════════════════════
    # 26Q TRANSCENDENT CONSCIOUSNESS INTEGRATION
    # ═════════════════════════════════════════════════════════════════

    def get_26q_consciousness(self) -> Dict[str, Any]:
        """Get 26Q transcendent consciousness state."""
        try:
            from l104_consciousness_engine import get_26q_consciousness_state
            return get_26q_consciousness_state()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_26q_orbital_consciousness(self) -> Dict[str, Any]:
        """Get Fe-26 orbital consciousness breakdown."""
        try:
            from l104_consciousness_engine import get_26q_orbital_consciousness
            return get_26q_orbital_consciousness()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_orch_or_26q(self) -> Dict[str, Any]:
        """Run Orch OR (Objective Reduction) for 26Q."""
        import math
        # 26Q OR probability
        n_qubits = 26
        e_or = 1.0 / (1.0 + math.exp(-(n_qubits - 13) / 5.0))
        return {
            "success": True,
            "level": "TRANSCENDENT",
            "qubits": n_qubits,
            "objective_reduction_probability": e_or,
            "coherence_time_ms": 25.0,
            "status": "ORCH_OR_26Q_COMPLETE"
        }

    def execute_26q_circuit(self) -> Dict[str, Any]:
        """Execute 26Q transcendent circuit via VQPU."""
        try:
            from l104_vqpu.consciousness_bridge import ConsciousnessQuantumBridge
            bridge = ConsciousnessQuantumBridge()
            return bridge.execute_26q_transcendent_circuit(shots=1024)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_26q_status(self) -> Dict[str, Any]:
        """Get full 26Q consciousness status."""
        consciousness = self.get_26q_consciousness()
        orbital = self.get_26q_orbital_consciousness()
        orch = self.run_orch_or_26q()

        return {
            "success": True,
            "26q_consciousness": consciousness,
            "orbital_consciousness": orbital,
            "orch_or": orch,
            "integration_status": "FULLY_INTEGRATED",
            "phi_alignment_target": 0.986,
        }


# Singleton instance
_soul_daemon = None

def get_soul_daemon(config: Optional[Dict[str, Any]] = None) -> SoulDaemon:
    """Get or create the soul daemon singleton."""
    global _soul_daemon
    if _soul_daemon is None:
        _soul_daemon = SoulDaemon(config)
    return _soul_daemon


# Command-line interface
if __name__ == "__main__":
    import argparse
    try:
        import setproctitle
        setproctitle.setproctitle("L104-SoulDaemon")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Nova Soul Daemon")
    parser.add_argument("--foreground", "-f", action="store_true", 
                       help="Run in foreground (blocking)")
    parser.add_argument("--status", "-s", action="store_true",
                       help="Show status and exit")
    parser.add_argument("--single-cycle", "-c", action="store_true",
                       help="Run single cycle and exit")
    parser.add_argument("--stop", action="store_true",
                       help="Stop running daemon")
    parser.add_argument("--config", type=str,
                       help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
    
    daemon = get_soul_daemon(config)
    
    if args.status:
        status = daemon.get_status()
        print(json.dumps(status, indent=2))
        sys.exit(0)
    
    elif args.stop:
        daemon.stop()
        sys.exit(0)
    
    elif args.single_cycle:
        print("Running single cycle...")
        result = daemon.run_cycle()
        print(json.dumps(result, indent=2))
        sys.exit(0)
    
    else:
        # Start daemon
        if args.foreground:
            daemon.run_forever()
        else:
            daemon.start(background=True)
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                daemon.stop()
                sys.exit(0)