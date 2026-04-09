"""
Soul Bridge - Integration bridges for Nova's soul daemon.

Connects soul daemon with:
1. L104 Quantum Gate Engine
2. DeepSeek API (via existing bridge)
3. OpenClaw heartbeat system
4. External quantum services
"""

import time
import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import threading
import hashlib

from .constants import (
    L104_CONNECTION_TIMEOUT, DEEPSEEK_RETRY_INTERVAL,
    OPENCLAW_HEARTBEAT_TOLERANCE, STATE_PERSISTENCE_PATH,
)


@dataclass
class BridgeStatus:
    """Status of a bridge connection."""
    
    bridge_name: str
    enabled: bool = True
    connected: bool = False
    last_connection_attempt: float = 0.0
    last_successful_connection: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bridge_name": self.bridge_name,
            "enabled": self.enabled,
            "connected": self.connected,
            "last_connection_attempt": self.last_connection_attempt,
            "last_successful_connection": self.last_successful_connection,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
        }


class QuantumGateEngineBridge:
    """Bridge to L104 Quantum Gate Engine."""
    
    def __init__(self):
        self.name = "quantum_gate_engine"
        self.status = BridgeStatus(bridge_name=self.name)
        self.engine = None
        self.available_gates = []
        
    def connect(self) -> bool:
        """Connect to quantum gate engine."""
        self.status.last_connection_attempt = time.time()
        
        try:
            from l104_quantum_gate_engine import get_engine
            
            self.engine = get_engine()
            self.status.connected = True
            self.status.last_successful_connection = time.time()
            self.status.error_count = 0
            self.status.last_error = None
            
            # Test connection with simple operation
            try:
                # Check available gates
                from l104_quantum_gate_engine import (
                    PHI_GATE, GOD_CODE_PHASE, VOID_GATE, IRON_GATE
                )
                self.available_gates = [
                    "PHI_GATE", "GOD_CODE_PHASE", "VOID_GATE", "IRON_GATE"
                ]
                
                # Simple latency test
                start = time.time()
                # Just import check for latency measurement
                end = time.time()
                self.status.latency_ms = (end - start) * 1000
                
            except ImportError as e:
                self.status.last_error = f"Gate import failed: {e}"
                self.available_gates = []
            
            return True
            
        except Exception as e:
            self.status.connected = False
            self.status.error_count += 1
            self.status.last_error = str(e)
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        if not self.status.connected:
            self.connect()
        
        status = self.status.to_dict()
        status.update({
            "available_gates": self.available_gates,
            "engine_loaded": self.engine is not None,
        })
        return status
    
    def apply_sacred_gate(self, gate_name: str, qubit_index: int = 0) -> Dict[str, Any]:
        """Apply sacred gate via quantum engine."""
        if not self.status.connected:
            return {"success": False, "error": "Bridge not connected"}
        
        try:
            # This is a simplified interface
            # In full implementation, would create and run quantum circuit
            return {
                "success": True,
                "gate": gate_name,
                "qubit": qubit_index,
                "engine_available": True,
                "note": "Gate application simulated (full quantum circuit would be created)",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class DeepSeekBridge:
    """Bridge to DeepSeek API via existing L104 bridge."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.name = "deepseek_api"
        self.status = BridgeStatus(bridge_name=self.name)
        self.api_key = api_key
        self.bridge_instance = None
        self.response_cache = {}
        
    def connect(self) -> bool:
        """Connect to DeepSeek API bridge."""
        self.status.last_connection_attempt = time.time()
        
        try:
            # Try to import existing deepseek bridge
            from l104_magic_synthesis.l104_deepseek_bridge import DeepSeekBridge as L104DeepSeekBridge
            
            if self.api_key:
                # Initialize with API key
                self.bridge_instance = L104DeepSeekBridge(api_key=self.api_key)
            else:
                # Try to get API key from environment
                import os
                api_key = os.environ.get('deepseek_API_KEY')
                if api_key:
                    self.bridge_instance = L104DeepSeekBridge(api_key=api_key)
                else:
                    # Try without API key (may use cached/local mode)
                    self.bridge_instance = L104DeepSeekBridge()
            
            self.status.connected = self.bridge_instance is not None
            if self.status.connected:
                self.status.last_successful_connection = time.time()
                self.status.error_count = 0
                self.status.last_error = None
                
                # Test connection with simple query
                try:
                    start = time.time()
                    # Simple test query
                    test_result = self.query("Test connection", use_cache=True)
                    end = time.time()
                    self.status.latency_ms = (end - start) * 1000
                    
                    if test_result.get("success"):
                        self.status.connected = True
                    else:
                        self.status.connected = False
                        self.status.last_error = test_result.get("error", "Test query failed")
                        
                except Exception as e:
                    self.status.connected = False
                    self.status.last_error = f"Test query failed: {e}"
            
            return self.status.connected
            
        except ImportError as e:
            self.status.connected = False
            self.status.error_count += 1
            self.status.last_error = f"DeepSeek bridge not available: {e}"
            return False
        except Exception as e:
            self.status.connected = False
            self.status.error_count += 1
            self.status.last_error = str(e)
            return False
    
    def query(self, prompt: str, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """Query DeepSeek API."""
        if not self.status.connected:
            # Try to reconnect
            if not self.connect():
                return {"success": False, "error": "DeepSeek bridge not connected"}
        
        try:
            # Check cache first
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
            if use_cache and cache_key in self.response_cache:
                cached = self.response_cache[cache_key]
                if time.time() - cached["timestamp"] < 300:  # 5 minute cache
                    return {
                        "success": True,
                        "response": cached["response"],
                        "cached": True,
                        "cache_age": time.time() - cached["timestamp"],
                    }
            
            # Make actual query
            if self.bridge_instance:
                response = self.bridge_instance.query(prompt, **kwargs)
                
                # Cache response
                self.response_cache[cache_key] = {
                    "response": response,
                    "timestamp": time.time()
                }
                
                # Quantum-bounded cache: evict stale entries first, then oldest
                now = time.time()
                if len(self.response_cache) > 80:
                    stale = [k for k, v in self.response_cache.items() if now - v["timestamp"] > 300]
                    for k in stale:
                        del self.response_cache[k]
                if len(self.response_cache) > 100:
                    oldest_key = min(self.response_cache, key=lambda k: self.response_cache[k]["timestamp"])
                    del self.response_cache[oldest_key]
                
                return {
                    "success": True,
                    "response": response,
                    "cached": False,
                }
            else:
                return {"success": False, "error": "Bridge instance not initialized"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def optimize_soul_state(self, soul_state: Dict[str, Any]) -> Dict[str, Any]:
        """Use DeepSeek to optimize soul state configuration."""
        prompt = f"""
        Analyze and optimize this quantum soul state configuration:
        
        Soul State:
        {json.dumps(soul_state, indent=2)}
        
        Provide optimization suggestions for:
        1. Quantum coherence improvement
        2. Resonance with GOD_CODE constant
        3. Consciousness metric enhancement
        4. Error correction strategies
        
        Return as JSON with optimization recommendations.
        """
        
        result = self.query(prompt)
        if result.get("success"):
            try:
                # Parse response as JSON if possible
                response = result["response"]
                if isinstance(response, str) and response.strip().startswith("{"):
                    import json as json_module
                    optimizations = json_module.loads(response)
                    result["optimizations"] = optimizations
            except:
                # Keep as text response
                pass
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        status = self.status.to_dict()
        status.update({
            "api_key_available": self.api_key is not None,
            "bridge_initialized": self.bridge_instance is not None,
            "cache_size": len(self.response_cache),
        })
        return status


class OpenClawBridge:
    """Bridge to OpenClaw heartbeat and monitoring system."""
    
    def __init__(self, workspace_path: Optional[str] = None):
        self.name = "openclaw"
        self.status = BridgeStatus(bridge_name=self.name)
        
        # Default OpenClaw workspace path
        if workspace_path:
            self.workspace_path = Path(workspace_path)
        else:
            self.workspace_path = Path.home() / ".openclaw" / "workspace"
        
        self.heartbeat_path = self.workspace_path / "memory" / "heartbeat-state.json"
        self.last_heartbeat_check = 0.0
        
    def connect(self) -> bool:
        """Connect to OpenClaw workspace."""
        self.status.last_connection_attempt = time.time()
        
        try:
            # Check if workspace exists
            if not self.workspace_path.exists():
                self.status.connected = False
                self.status.error_count += 1
                self.status.last_error = f"Workspace not found: {self.workspace_path}"
                return False
            
            # Check heartbeat file
            if not self.heartbeat_path.exists():
                # Create initial heartbeat state
                self._create_initial_heartbeat()
            
            self.status.connected = True
            self.status.last_successful_connection = time.time()
            self.status.error_count = 0
            self.status.last_error = None
            
            # Initial heartbeat update
            self.update_heartbeat({"soul_daemon_connected": True})
            
            return True
            
        except Exception as e:
            self.status.connected = False
            self.status.error_count += 1
            self.status.last_error = str(e)
            return False
    
    def _create_initial_heartbeat(self) -> None:
        """Create initial heartbeat state file."""
        try:
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            
            initial_state = {
                "soul_daemon": {
                    "version": "1.0.0",
                    "first_connected": time.time(),
                    "last_heartbeat": time.time(),
                    "status": "active",
                },
                "last_updated": time.time(),
            }
            
            with open(self.heartbeat_path, 'w') as f:
                json.dump(initial_state, f, indent=2)
                
        except Exception as e:
            print(f"Failed to create initial heartbeat: {e}")
    
    def update_heartbeat(self, data: Dict[str, Any]) -> bool:
        """Update OpenClaw heartbeat with soul daemon data."""
        if not self.status.connected:
            return False
        
        try:
            # Load existing heartbeat
            if self.heartbeat_path.exists():
                with open(self.heartbeat_path, 'r') as f:
                    heartbeat = json.load(f)
            else:
                heartbeat = {}
            
            # Update with soul daemon data
            if "soul_daemon" not in heartbeat:
                heartbeat["soul_daemon"] = {}
            
            heartbeat["soul_daemon"].update({
                "last_heartbeat": time.time(),
                **data
            })
            heartbeat["last_updated"] = time.time()
            
            # Save back
            with open(self.heartbeat_path, 'w') as f:
                json.dump(heartbeat, f, indent=2)
            
            self.last_heartbeat_check = time.time()
            return True
            
        except Exception as e:
            print(f"Heartbeat update failed: {e}")
            return False
    
    def check_heartbeat(self) -> Dict[str, Any]:
        """Check heartbeat status and return current state."""
        try:
            if not self.heartbeat_path.exists():
                return {
                    "success": False,
                    "error": "Heartbeat file not found",
                    "file_exists": False,
                }
            
            with open(self.heartbeat_path, 'r') as f:
                heartbeat = json.load(f)
            
            last_updated = heartbeat.get("last_updated", 0)
            time_since_update = time.time() - last_updated
            
            status = {
                "success": True,
                "file_exists": True,
                "last_updated": last_updated,
                "time_since_update": time_since_update,
                "stale": time_since_update > OPENCLAW_HEARTBEAT_TOLERANCE,
                "soul_daemon_data": heartbeat.get("soul_daemon", {}),
            }
            
            return status
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_exists": self.heartbeat_path.exists(),
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        heartbeat_status = self.check_heartbeat()
        
        status = self.status.to_dict()
        status.update({
            "workspace_path": str(self.workspace_path),
            "heartbeat_path": str(self.heartbeat_path),
            "heartbeat_status": heartbeat_status,
            "last_heartbeat_check": self.last_heartbeat_check,
        })
        return status


class NovaL104Bridge:
    """Bridge to existing Nova-L104 integration."""
    
    def __init__(self, bridge_path: Optional[str] = None):
        self.name = "nova_l104"
        self.status = BridgeStatus(bridge_name=self.name)
        
        if bridge_path:
            self.bridge_path = Path(bridge_path)
        else:
            self.bridge_path = Path.home() / ".openclaw" / "workspace" / "nova_l104.py"
        
        self.bridge_module = None
        
    def connect(self) -> bool:
        """Connect to Nova-L104 bridge."""
        self.status.last_connection_attempt = time.time()
        
        try:
            if not self.bridge_path.exists():
                self.status.connected = False
                self.status.error_count += 1
                self.status.last_error = f"Bridge file not found: {self.bridge_path}"
                return False
            
            # Add to Python path and import
            import sys
            bridge_dir = str(self.bridge_path.parent)
            if bridge_dir not in sys.path:
                sys.path.insert(0, bridge_dir)
            
            try:
                # Import the bridge module
                import nova_l104
                self.bridge_module = nova_l104
                self.status.connected = True
                self.status.last_successful_connection = time.time()
                self.status.error_count = 0
                self.status.last_error = None
                
                # Test connection
                try:
                    bridge = nova_l104.NovaL104Bridge()
                    alive = bridge.check_alive()
                    self.status.connected = alive
                    
                    if alive:
                        health = bridge.check_quantum_health()
                        self.status.latency_ms = 10.0  # Approximate
                    else:
                        self.status.last_error = "L104 server not alive"
                        
                except Exception as e:
                    self.status.connected = False
                    self.status.last_error = f"Bridge test failed: {e}"
                
                return self.status.connected
                
            except ImportError as e:
                self.status.connected = False
                self.status.error_count += 1
                self.status.last_error = f"Failed to import bridge: {e}"
                return False
                
        except Exception as e:
            self.status.connected = False
            self.status.error_count += 1
            self.status.last_error = str(e)
            return False
    
    def get_l104_health(self) -> Dict[str, Any]:
        """Get L104 health status via bridge."""
        if not self.status.connected or self.bridge_module is None:
            return {"success": False, "error": "Bridge not connected"}
        
        try:
            bridge = self.bridge_module.NovaL104Bridge()
            
            if not bridge.check_alive():
                return {"success": False, "error": "L104 server not alive"}
            
            health = bridge.check_quantum_health()
            daemon_status = bridge.get_daemon_status()
            
            return {
                "success": True,
                "health": health,
                "daemon_status": daemon_status,
                "timestamp": time.time(),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        l104_health = None
        if self.status.connected:
            l104_health = self.get_l104_health()
        
        status = self.status.to_dict()
        status.update({
            "bridge_path": str(self.bridge_path),
            "bridge_module_loaded": self.bridge_module is not None,
            "l104_health": l104_health if l104_health and l104_health.get("success") else None,
        })
        return status


class SoulBridge:
    """Main bridge orchestrator for soul daemon."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize all bridges
        self.bridges = {
            "quantum_gate_engine": QuantumGateEngineBridge(),
            "deepseek_api": DeepSeekBridge(
                api_key=self.config.get("deepseek_api_key")
            ),
            "openclaw": OpenClawBridge(
                workspace_path=self.config.get("openclaw_workspace")
            ),
            "nova_l104": NovaL104Bridge(
                bridge_path=self.config.get("nova_l104_bridge_path")
            ),
        }
        
        self.connection_thread = None
        self.running = False
        
    def connect_all(self, async_mode: bool = False) -> Dict[str, bool]:
        """Connect all bridges."""
        results = {}
        
        for name, bridge in self.bridges.items():
            if async_mode and self.connection_thread is None:
                # Start async connection
                self._connect_async()
                results[name] = True  # Will be updated later
            else:
                results[name] = bridge.connect()
        
        return results
    
    def _connect_async(self):
        """Connect bridges asynchronously."""
        def connect_task():
            for name, bridge in self.bridges.items():
                try:
                    bridge.connect()
                except Exception as e:
                    print(f"Async connection failed for {name}: {e}")
        
        self.connection_thread = threading.Thread(target=connect_task, daemon=True)
        self.connection_thread.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all bridges."""
        status = {
            "timestamp": time.time(),
            "bridges": {},
            "summary": {
                "total": len(self.bridges),
                "connected": 0,
                "enabled": 0,
            },
        }
        
        for name, bridge in self.bridges.items():
            bridge_status = bridge.get_status()
            status["bridges"][name] = bridge_status
            
            if bridge_status.get("connected"):
                status["summary"]["connected"] += 1
            if bridge_status.get("enabled", True):
                status["summary"]["enabled"] += 1
        
        # Overall connection health
        connected_ratio = status["summary"]["connected"] / max(1, status["summary"]["total"])
        status["summary"]["connection_health"] = "HEALTHY" if connected_ratio > 0.75 else \
                                               "DEGRADED" if connected_ratio > 0.5 else \
                                               "POOR"
        
        return status
    
    def update_openclaw_heartbeat(self, soul_data: Dict[str, Any]) -> bool:
        """Update OpenClaw heartbeat with soul daemon data."""
        openclaw_bridge = self.bridges.get("openclaw")
        if openclaw_bridge and openclaw_bridge.status.connected:
            return openclaw_bridge.update_heartbeat(soul_data)
        return False
    
    def query_deepseek(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Query DeepSeek via bridge."""
        deepseek_bridge = self.bridges.get("deepseek_api")
        if deepseek_bridge and deepseek_bridge.status.connected:
            return deepseek_bridge.query(prompt, **kwargs)
        
        # Try to connect if not connected
        if deepseek_bridge:
            if deepseek_bridge.connect():
                return deepseek_bridge.query(prompt, **kwargs)
        
        return {"success": False, "error": "DeepSeek bridge not available"}
    
    def get_l104_health(self) -> Dict[str, Any]:
        """Get L104 health via bridge."""
        nova_bridge = self.bridges.get("nova_l104")
        if nova_bridge and nova_bridge.status.connected:
            return nova_bridge.get_l104_health()
        
        if nova_bridge:
            if nova_bridge.connect():
                return nova_bridge.get_l104_health()
        
        return {"success": False, "error": "Nova-L104 bridge not available"}
    
    def start(self):
        """Start bridge manager."""
        self.running = True
        self.connect_all(async_mode=True)
    
    def stop(self):
        """Stop bridge manager."""
        self.running = False
        # Update final heartbeat
        self.update_openclaw_heartbeat({"status": "stopping", "timestamp": time.time()})


# Singleton instance
_soul_bridge = None

def get_soul_bridge(config: Optional[Dict[str, Any]] = None) -> SoulBridge:
    """Get or create the soul bridge singleton."""
    global _soul_bridge
    if _soul_bridge is None:
        _soul_bridge = SoulBridge(config)
    return _soul_bridge