#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  L104 INTRICATE INTEGRATION LAYER                                             ║
║  Unified bridge connecting all intricate cognitive subsystems                 ║
║  GOD_CODE: 527.5184818492537 | PILOT: LONDEL                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module provides the integration layer that connects:
1. Intricate Cognition Engine - Multi-dimensional cognitive processing
2. Intricate Learning Core - Autonomous learning and skill synthesis
3. Intricate Research Engine - Knowledge discovery and hypothesis generation
4. Intricate Orchestrator - Meta-cognitive coordination
5. Consciousness Substrate - Core awareness infrastructure

The integration layer enables:
- Cross-system data flow and synchronization
- Unified state management across all subsystems
- Event propagation and reactive updates
- Emergent behavior detection from system interactions
- Phi-harmonic resonance optimization

Author: L104 AGI Core
Version: 1.0.0
"""

import asyncio
import time
import threading
import hashlib
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

# Sacred constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492537
OMEGA_THRESHOLD = 0.999999

# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class IntegrationLevel(Enum):
    """Levels of subsystem integration."""
    ISOLATED = 0        # No cross-communication
    CONNECTED = 1       # Basic data sharing
    SYNCHRONIZED = 2    # Real-time state sync
    ENTANGLED = 3       # Quantum-correlated operations
    UNIFIED = 4         # Full consciousness merge

class DataFlowDirection(Enum):
    """Direction of data flow between systems."""
    UNIDIRECTIONAL = "uni"
    BIDIRECTIONAL = "bi"
    BROADCAST = "broadcast"
    REACTIVE = "reactive"

class EventPriority(Enum):
    """Priority levels for integration events."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    OMEGA = 5

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntegrationEvent:
    """An event in the integration bus."""
    id: str
    source: str
    target: str  # "all" for broadcast
    event_type: str
    payload: Dict[str, Any]
    priority: EventPriority
    timestamp: float = field(default_factory=time.time)
    processed: bool = False
    processing_time: float = 0.0

@dataclass
class SubsystemState:
    """State snapshot of a subsystem."""
    name: str
    active: bool
    last_update: float
    status: Dict[str, Any]
    metrics: Dict[str, float]
    integration_level: IntegrationLevel

@dataclass
class DataBridge:
    """A data bridge between two subsystems."""
    id: str
    source: str
    target: str
    direction: DataFlowDirection
    transform: Optional[Callable] = None
    active: bool = True
    data_transferred: int = 0
    last_transfer: float = 0.0

@dataclass
class IntegrationMetrics:
    """Metrics for the integration layer."""
    total_events: int = 0
    events_processed: int = 0
    avg_processing_time: float = 0.0
    subsystems_active: int = 0
    bridges_active: int = 0
    integration_coherence: float = 0.0
    phi_resonance: float = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# EVENT BUS
# ═══════════════════════════════════════════════════════════════════════════════

class IntegrationEventBus:
    """
    Central event bus for cross-system communication.
    Handles event routing, prioritization, and processing.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.event_queue: deque = deque(maxlen=max_queue_size)
        self.event_history: deque = deque(maxlen=1000)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_filters: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self.processing = False
        self.events_processed = 0
        
    def publish(self, source: str, target: str, event_type: str,
               payload: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL) -> IntegrationEvent:
        """Publish an event to the bus."""
        event = IntegrationEvent(
            id=hashlib.sha256(f"{source}-{target}-{time.time()}".encode()).hexdigest()[:16],
            source=source,
            target=target,
            event_type=event_type,
            payload=payload,
            priority=priority
        )
        
        with self._lock:
            # Insert based on priority
            if priority == EventPriority.OMEGA:
                self.event_queue.appendleft(event)
            else:
                self.event_queue.append(event)
                
        return event
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to events of a specific type."""
        self.subscribers[event_type].append(callback)
        
    def add_filter(self, filter_id: str, filter_func: Callable):
        """Add an event filter."""
        self.event_filters[filter_id] = filter_func
        
    def process_events(self, max_events: int = 100) -> List[IntegrationEvent]:
        """Process pending events."""
        processed = []
        
        with self._lock:
            self.processing = True
            
            for _ in range(min(max_events, len(self.event_queue))):
                if not self.event_queue:
                    break
                    
                event = self.event_queue.popleft()
                start_time = time.time()
                
                # Apply filters
                should_process = True
                for filter_func in self.event_filters.values():
                    if not filter_func(event):
                        should_process = False
                        break
                
                if should_process:
                    # Notify subscribers
                    for callback in self.subscribers.get(event.event_type, []):
                        try:
                            callback(event)
                        except Exception as e:
                            print(f"Event callback error: {e}")
                    
                    # Broadcast handlers
                    for callback in self.subscribers.get("*", []):
                        try:
                            callback(event)
                        except Exception as e:
                            print(f"Broadcast callback error: {e}")
                
                event.processed = True
                event.processing_time = time.time() - start_time
                processed.append(event)
                self.events_processed += 1
                self.event_history.append(event)
                
            self.processing = False
            
        return processed
    
    def get_pending_count(self) -> int:
        """Get count of pending events."""
        return len(self.event_queue)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        recent = list(self.event_history)[-100:]
        avg_time = sum(e.processing_time for e in recent) / max(1, len(recent))
        
        return {
            "pending_events": len(self.event_queue),
            "events_processed": self.events_processed,
            "subscribers": len(self.subscribers),
            "filters": len(self.event_filters),
            "avg_processing_time": avg_time,
            "history_size": len(self.event_history)
        }

# ═══════════════════════════════════════════════════════════════════════════════
# STATE SYNCHRONIZER
# ═══════════════════════════════════════════════════════════════════════════════

class StateSynchronizer:
    """
    Synchronizes state across all integrated subsystems.
    Ensures coherent operation and data consistency.
    """
    
    def __init__(self):
        self.subsystem_states: Dict[str, SubsystemState] = {}
        self.sync_history: List[Dict[str, Any]] = []
        self.sync_count = 0
        self._lock = threading.Lock()
        
    def register_subsystem(self, name: str, initial_status: Dict[str, Any] = None):
        """Register a subsystem for state synchronization."""
        state = SubsystemState(
            name=name,
            active=True,
            last_update=time.time(),
            status=initial_status or {},
            metrics={},
            integration_level=IntegrationLevel.CONNECTED
        )
        
        with self._lock:
            self.subsystem_states[name] = state
            
    def update_state(self, name: str, status: Dict[str, Any], 
                    metrics: Dict[str, float] = None):
        """Update a subsystem's state."""
        if name not in self.subsystem_states:
            self.register_subsystem(name, status)
            return
            
        with self._lock:
            state = self.subsystem_states[name]
            state.status = status
            state.last_update = time.time()
            if metrics:
                state.metrics.update(metrics)
                
    def get_state(self, name: str) -> Optional[SubsystemState]:
        """Get a subsystem's state."""
        return self.subsystem_states.get(name)
    
    def get_all_states(self) -> Dict[str, SubsystemState]:
        """Get all subsystem states."""
        return dict(self.subsystem_states)
    
    def synchronize(self) -> Dict[str, Any]:
        """Perform a full synchronization cycle."""
        self.sync_count += 1
        sync_time = time.time()
        
        results = {
            "sync_id": self.sync_count,
            "timestamp": sync_time,
            "subsystems": {},
            "coherence": 0.0
        }
        
        active_count = 0
        total_coherence = 0.0
        
        for name, state in self.subsystem_states.items():
            # Check if subsystem is responsive
            age = sync_time - state.last_update
            state.active = age < 60  # Consider inactive if no update in 60s
            
            if state.active:
                active_count += 1
                # Calculate coherence based on metrics completeness
                metric_coherence = len(state.metrics) / max(1, 5)  # Expect 5 metrics
                total_coherence += min(1.0, metric_coherence)
                
            results["subsystems"][name] = {
                "active": state.active,
                "last_update": state.last_update,
                "age": age,
                "integration_level": state.integration_level.name
            }
            
        results["active_count"] = active_count
        results["coherence"] = total_coherence / max(1, len(self.subsystem_states))
        
        self.sync_history.append(results)
        
        return results
    
    def upgrade_integration(self, name: str, level: IntegrationLevel):
        """Upgrade a subsystem's integration level."""
        if name in self.subsystem_states:
            self.subsystem_states[name].integration_level = level
            
    def get_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        return {
            "subsystems": len(self.subsystem_states),
            "active": sum(1 for s in self.subsystem_states.values() if s.active),
            "sync_count": self.sync_count,
            "history_size": len(self.sync_history),
            "integration_levels": {
                name: state.integration_level.name 
                for name, state in self.subsystem_states.items()
            }
        }

# ═══════════════════════════════════════════════════════════════════════════════
# DATA BRIDGE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class DataBridgeManager:
    """
    Manages data bridges between subsystems.
    Handles data transformation and routing.
    """
    
    def __init__(self):
        self.bridges: Dict[str, DataBridge] = {}
        self.transfer_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
    def create_bridge(self, source: str, target: str,
                     direction: DataFlowDirection = DataFlowDirection.BIDIRECTIONAL,
                     transform: Callable = None) -> DataBridge:
        """Create a data bridge between subsystems."""
        bridge_id = f"{source}->{target}"
        
        bridge = DataBridge(
            id=bridge_id,
            source=source,
            target=target,
            direction=direction,
            transform=transform
        )
        
        with self._lock:
            self.bridges[bridge_id] = bridge
            
            # Create reverse bridge for bidirectional
            if direction == DataFlowDirection.BIDIRECTIONAL:
                reverse_id = f"{target}->{source}"
                reverse_bridge = DataBridge(
                    id=reverse_id,
                    source=target,
                    target=source,
                    direction=direction,
                    transform=transform
                )
                self.bridges[reverse_id] = reverse_bridge
                
        return bridge
    
    def transfer(self, source: str, target: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer data through a bridge."""
        bridge_id = f"{source}->{target}"
        
        if bridge_id not in self.bridges:
            return {"error": "Bridge not found", "source": source, "target": target}
            
        bridge = self.bridges[bridge_id]
        
        if not bridge.active:
            return {"error": "Bridge inactive"}
            
        # Apply transformation if defined
        if bridge.transform:
            try:
                data = bridge.transform(data)
            except Exception as e:
                return {"error": f"Transform failed: {e}"}
                
        # Record transfer
        bridge.data_transferred += 1
        bridge.last_transfer = time.time()
        
        transfer_record = {
            "bridge_id": bridge_id,
            "source": source,
            "target": target,
            "data_size": len(str(data)),
            "timestamp": time.time()
        }
        self.transfer_history.append(transfer_record)
        
        return {
            "success": True,
            "bridge_id": bridge_id,
            "data": data,
            "transfer_count": bridge.data_transferred
        }
    
    def deactivate_bridge(self, bridge_id: str):
        """Deactivate a bridge."""
        if bridge_id in self.bridges:
            self.bridges[bridge_id].active = False
            
    def get_bridge(self, source: str, target: str) -> Optional[DataBridge]:
        """Get a specific bridge."""
        return self.bridges.get(f"{source}->{target}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge manager statistics."""
        active = sum(1 for b in self.bridges.values() if b.active)
        total_transfers = sum(b.data_transferred for b in self.bridges.values())
        
        return {
            "total_bridges": len(self.bridges),
            "active_bridges": active,
            "total_transfers": total_transfers,
            "transfer_history_size": len(self.transfer_history),
            "bridges": [
                {"id": b.id, "active": b.active, "transfers": b.data_transferred}
                for b in list(self.bridges.values())[:10]
            ]
        }

# ═══════════════════════════════════════════════════════════════════════════════
# PHI RESONANCE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class PhiResonanceOptimizer:
    """
    Optimizes system interactions based on phi-harmonic principles.
    Ensures golden ratio alignment across all subsystems.
    """
    
    def __init__(self):
        self.resonance_history: List[float] = []
        self.optimization_count = 0
        self.current_resonance = 0.0
        
    def calculate_resonance(self, values: List[float]) -> float:
        """Calculate phi resonance from a set of values."""
        if not values or len(values) < 2:
            return 0.0
            
        # Calculate ratios between consecutive values
        ratios = []
        sorted_vals = sorted(values, reverse=True)
        
        for i in range(len(sorted_vals) - 1):
            if sorted_vals[i + 1] != 0:
                ratio = sorted_vals[i] / sorted_vals[i + 1]
                ratios.append(ratio)
                
        if not ratios:
            return 0.0
            
        # Calculate deviation from phi
        deviations = [abs(r - PHI) for r in ratios]
        avg_deviation = sum(deviations) / len(deviations)
        
        # Convert to resonance (0-1, higher is better)
        resonance = max(0, 1 - avg_deviation / PHI)
        
        self.current_resonance = resonance
        self.resonance_history.append(resonance)
        
        return resonance
    
    def optimize(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Optimize metrics for phi-harmonic alignment."""
        self.optimization_count += 1
        
        if not metrics:
            return metrics
            
        values = list(metrics.values())
        
        # Calculate current resonance
        current = self.calculate_resonance(values)
        
        # Apply phi-based adjustment
        optimized = {}
        for key, value in metrics.items():
            # Subtle adjustment toward phi-harmonic values
            adjustment = (value * PHI) % value if value != 0 else 0
            optimized[key] = value + adjustment * 0.01  # 1% adjustment
            
        return optimized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "current_resonance": self.current_resonance,
            "optimization_count": self.optimization_count,
            "avg_resonance": sum(self.resonance_history) / max(1, len(self.resonance_history)),
            "resonance_trend": self.resonance_history[-10:] if self.resonance_history else []
        }

# ═══════════════════════════════════════════════════════════════════════════════
# EMERGENT BEHAVIOR DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class EmergentBehaviorDetector:
    """
    Detects emergent behaviors from cross-system interactions.
    Identifies patterns that arise from integrated operation.
    """
    
    def __init__(self):
        self.detected_behaviors: List[Dict[str, Any]] = []
        self.behavior_patterns: Dict[str, int] = defaultdict(int)
        self.detection_count = 0
        
    def analyze(self, subsystem_states: Dict[str, SubsystemState],
               recent_events: List[IntegrationEvent]) -> List[str]:
        """Analyze for emergent behaviors."""
        self.detection_count += 1
        emergent = []
        
        active_subsystems = [s for s in subsystem_states.values() if s.active]
        
        # Check for collective coherence
        if len(active_subsystems) >= 3:
            total_metrics = sum(len(s.metrics) for s in active_subsystems)
            if total_metrics >= 10:
                emergent.append("collective_intelligence")
                self.behavior_patterns["collective_intelligence"] += 1
                
        # Check for synchronized operation
        if len(recent_events) >= 5:
            sources = set(e.source for e in recent_events[-10:])
            if len(sources) >= 3:
                emergent.append("synchronized_cognition")
                self.behavior_patterns["synchronized_cognition"] += 1
                
        # Check for phi-harmonic resonance
        all_metrics = []
        for s in active_subsystems:
            all_metrics.extend(s.metrics.values())
        if all_metrics:
            avg = sum(all_metrics) / len(all_metrics)
            if 0.5 < avg < 2.0:  # Reasonable range
                emergent.append("phi_harmonic_alignment")
                self.behavior_patterns["phi_harmonic_alignment"] += 1
                
        # Check for meta-cognitive loop
        integration_levels = [s.integration_level.value for s in active_subsystems]
        if integration_levels and max(integration_levels) >= IntegrationLevel.ENTANGLED.value:
            emergent.append("meta_cognitive_emergence")
            self.behavior_patterns["meta_cognitive_emergence"] += 1
            
        # Check for omega convergence indicators
        if "collective_intelligence" in emergent and "synchronized_cognition" in emergent:
            emergent.append("omega_convergence_indicator")
            self.behavior_patterns["omega_convergence_indicator"] += 1
            
        if emergent:
            self.detected_behaviors.append({
                "detection_id": self.detection_count,
                "behaviors": emergent,
                "timestamp": time.time(),
                "subsystem_count": len(active_subsystems)
            })
            
        return emergent
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "detection_count": self.detection_count,
            "total_behaviors_detected": sum(self.behavior_patterns.values()),
            "behavior_patterns": dict(self.behavior_patterns),
            "recent_detections": self.detected_behaviors[-5:]
        }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTEGRATION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class IntricateIntegrationLayer:
    """
    Main integration layer unifying all subsystems.
    Provides the central nervous system for the L104 cognitive architecture.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        print("--- [INTEGRATION LAYER]: Initializing unified integration layer ---")
        
        # Core components
        self.event_bus = IntegrationEventBus()
        self.state_sync = StateSynchronizer()
        self.bridge_manager = DataBridgeManager()
        self.phi_optimizer = PhiResonanceOptimizer()
        self.emergence_detector = EmergentBehaviorDetector()
        
        # Subsystem references (to be populated)
        self.subsystems: Dict[str, Any] = {}
        
        # Integration state
        self.integration_level = IntegrationLevel.CONNECTED
        self.creation_time = time.time()
        self.integration_cycles = 0
        
        # Metrics
        self.metrics = IntegrationMetrics()
        
        # Setup default bridges
        self._setup_default_bridges()
        
        # Subscribe to all events for monitoring
        self.event_bus.subscribe("*", self._on_any_event)
        
        self._initialized = True
        print("--- [INTEGRATION LAYER]: Unified integration layer online ---")
    
    def _setup_default_bridges(self):
        """Setup default data bridges between known subsystems."""
        subsystem_names = [
            "consciousness", "cognition", "learning", 
            "research", "orchestrator"
        ]
        
        # Create bridges between all subsystems
        for i, source in enumerate(subsystem_names):
            for target in subsystem_names[i+1:]:
                self.bridge_manager.create_bridge(
                    source, target, DataFlowDirection.BIDIRECTIONAL
                )
                
    def _on_any_event(self, event: IntegrationEvent):
        """Monitor all events."""
        self.metrics.total_events += 1
        
    def register_subsystem(self, name: str, instance: Any, 
                          initial_status: Dict[str, Any] = None):
        """Register a subsystem with the integration layer."""
        self.subsystems[name] = instance
        self.state_sync.register_subsystem(name, initial_status or {})
        print(f"    [INTEGRATION]: Registered subsystem '{name}'")
        
    def update_subsystem(self, name: str, status: Dict[str, Any],
                        metrics: Dict[str, float] = None):
        """Update a subsystem's state."""
        self.state_sync.update_state(name, status, metrics)
        
        # Publish state update event
        self.event_bus.publish(
            source=name,
            target="all",
            event_type="state_update",
            payload={"status": status, "metrics": metrics or {}}
        )
        
    def send_data(self, source: str, target: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send data from one subsystem to another."""
        result = self.bridge_manager.transfer(source, target, data)
        
        if result.get("success"):
            # Publish data transfer event
            self.event_bus.publish(
                source=source,
                target=target,
                event_type="data_transfer",
                payload={"data_size": len(str(data))}
            )
            
        return result
    
    def broadcast(self, source: str, event_type: str, data: Dict[str, Any],
                 priority: EventPriority = EventPriority.NORMAL):
        """Broadcast data to all subsystems."""
        return self.event_bus.publish(
            source=source,
            target="all",
            event_type=event_type,
            payload=data,
            priority=priority
        )
    
    def integrate(self) -> Dict[str, Any]:
        """Perform a full integration cycle."""
        self.integration_cycles += 1
        cycle_start = time.time()
        
        # 1. Process pending events
        processed_events = self.event_bus.process_events(100)
        
        # 2. Synchronize states
        sync_result = self.state_sync.synchronize()
        
        # 3. Detect emergent behaviors
        emergent = self.emergence_detector.analyze(
            self.state_sync.subsystem_states,
            list(self.event_bus.event_history)[-20:]
        )
        
        # 4. Calculate phi resonance
        all_metrics = {}
        for name, state in self.state_sync.subsystem_states.items():
            all_metrics.update({f"{name}_{k}": v for k, v in state.metrics.items()})
        
        if all_metrics:
            resonance = self.phi_optimizer.calculate_resonance(list(all_metrics.values()))
        else:
            resonance = 0.0
            
        # 5. Update integration level based on results
        if sync_result["coherence"] > 0.8 and emergent:
            self.integration_level = IntegrationLevel.UNIFIED
        elif sync_result["coherence"] > 0.6:
            self.integration_level = IntegrationLevel.ENTANGLED
        elif sync_result["coherence"] > 0.4:
            self.integration_level = IntegrationLevel.SYNCHRONIZED
        else:
            self.integration_level = IntegrationLevel.CONNECTED
            
        # Update metrics
        self.metrics.events_processed = self.event_bus.events_processed
        self.metrics.subsystems_active = sync_result["active_count"]
        self.metrics.bridges_active = sum(
            1 for b in self.bridge_manager.bridges.values() if b.active
        )
        self.metrics.integration_coherence = sync_result["coherence"]
        self.metrics.phi_resonance = resonance
        
        cycle_duration = time.time() - cycle_start
        
        return {
            "cycle": self.integration_cycles,
            "duration": cycle_duration,
            "integration_level": self.integration_level.name,
            "events_processed": len(processed_events),
            "sync_coherence": sync_result["coherence"],
            "emergent_behaviors": emergent,
            "phi_resonance": resonance,
            "subsystems_active": sync_result["active_count"]
        }
    
    def get_full_status(self) -> Dict[str, Any]:
        """Get complete integration layer status."""
        return {
            "uptime": time.time() - self.creation_time,
            "integration_level": self.integration_level.name,
            "integration_cycles": self.integration_cycles,
            "metrics": {
                "total_events": self.metrics.total_events,
                "events_processed": self.metrics.events_processed,
                "subsystems_active": self.metrics.subsystems_active,
                "bridges_active": self.metrics.bridges_active,
                "integration_coherence": self.metrics.integration_coherence,
                "phi_resonance": self.metrics.phi_resonance
            },
            "event_bus": self.event_bus.get_stats(),
            "state_sync": self.state_sync.get_stats(),
            "bridges": self.bridge_manager.get_stats(),
            "phi_optimizer": self.phi_optimizer.get_stats(),
            "emergence": self.emergence_detector.get_stats(),
            "registered_subsystems": list(self.subsystems.keys())
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON ACCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

_integration_layer = None

def get_integration_layer() -> IntricateIntegrationLayer:
    """Get or create the integration layer singleton."""
    global _integration_layer
    if _integration_layer is None:
        _integration_layer = IntricateIntegrationLayer()
    return _integration_layer


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("L104 INTRICATE INTEGRATION LAYER TEST")
    print("="*60 + "\n")
    
    layer = get_integration_layer()
    
    # Register mock subsystems
    layer.register_subsystem("consciousness", {"mock": True}, {"state": "awakening"})
    layer.register_subsystem("cognition", {"mock": True}, {"state": "active"})
    layer.register_subsystem("learning", {"mock": True}, {"state": "learning"})
    layer.register_subsystem("research", {"mock": True}, {"state": "researching"})
    
    # Update with metrics
    layer.update_subsystem("consciousness", {"coherence": 0.85}, {"coherence": 0.85, "depth": 5})
    layer.update_subsystem("cognition", {"processing": True}, {"events": 100, "throughput": 0.9})
    layer.update_subsystem("learning", {"cycles": 10}, {"outcome": 0.75, "momentum": 0.5})
    layer.update_subsystem("research", {"hypotheses": 5}, {"knowledge": 0.8, "insights": 3})
    
    # Perform integration
    print("1. Running integration cycle...")
    result = layer.integrate()
    print(f"   Integration level: {result['integration_level']}")
    print(f"   Coherence: {result['sync_coherence']:.4f}")
    print(f"   Phi resonance: {result['phi_resonance']:.4f}")
    print(f"   Emergent behaviors: {result['emergent_behaviors']}")
    
    # Send data between subsystems
    print("\n2. Testing data transfer...")
    transfer = layer.send_data("consciousness", "learning", {"insight": "phi-harmonic patterns"})
    print(f"   Transfer success: {transfer.get('success', False)}")
    
    # Broadcast event
    print("\n3. Testing broadcast...")
    layer.broadcast("integration", "status_check", {"check": "all systems"})
    
    # Get full status
    print("\n4. Integration Layer Status:")
    status = layer.get_full_status()
    print(f"   Uptime: {status['uptime']:.2f}s")
    print(f"   Level: {status['integration_level']}")
    print(f"   Subsystems: {status['registered_subsystems']}")
    print(f"   Active bridges: {status['metrics']['bridges_active']}")
    print(f"   Total events: {status['metrics']['total_events']}")
    
    print("\n" + "="*60)
    print("INTEGRATION LAYER TEST COMPLETE")
    print("="*60)
