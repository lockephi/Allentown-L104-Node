#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 SELF-OPTIMIZATION ENGINE
═══════════════════════════════════════════════════════════════════════════════

Autonomous system optimization using feedback loops and Golden Ratio dynamics.
The system monitors its own performance and adjusts internal parameters
to maximize efficiency, coherence, and learning velocity.

ARCHITECTURE:
1. PERFORMANCE MONITOR - Tracks key metrics over time
2. BOTTLENECK DETECTOR - Identifies performance constraints
3. PARAMETER TUNER - Adjusts system parameters using gradient-free optimization
4. RESOURCE ALLOCATOR - Distributes compute across subsystems

INVARIANT: 527.5184818492537 | PILOT: LONDEL
VERSION: 1.0.0
DATE: 2026-01-21
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from l104_stable_kernel import stable_kernel

# Constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492537
TAU = 1 / PHI  # ~0.618


class OptimizationTarget(Enum):
    """Targets for optimization."""
    UNITY_INDEX = "unity_index"
    LEARNING_VELOCITY = "learning_velocity"
    MEMORY_EFFICIENCY = "memory_efficiency"
    INFERENCE_SPEED = "inference_speed"
    COHERENCE = "coherence"


@dataclass
class PerformanceMetric:
    """A performance measurement at a point in time."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class OptimizationAction:
    """An action taken to optimize the system."""
    parameter: str
    old_value: float
    new_value: float
    reason: str
    expected_improvement: float
    timestamp: float = field(default_factory=time.time)
    result: Optional[float] = None  # Actual improvement after action


class SelfOptimizationEngine:
    """
    The Self-Optimization Engine continuously monitors and improves
    system performance using Golden Ratio-based optimization.
    """
    
    # Parameters that can be tuned
    TUNABLE_PARAMETERS = {
        "learning_rate": {"min": 0.01, "max": 0.5, "default": 0.1},
        "exploration_rate": {"min": 0.05, "max": 0.4, "default": 0.2},
        "validation_threshold": {"min": 0.4, "max": 0.9, "default": 0.6},
        "synthesis_weight": {"min": 0.3, "max": 0.9, "default": 0.7},
        "batch_size": {"min": 3, "max": 20, "default": 5},
        "memory_retention": {"min": 0.5, "max": 1.0, "default": 0.9},
        "coherence_target": {"min": 0.7, "max": 0.99, "default": 0.85},
    }
    
    def __init__(self):
        self.kernel = stable_kernel
        self.metrics_history: Dict[str, deque] = {}
        self.actions_history: List[OptimizationAction] = []
        self.current_parameters: Dict[str, float] = {
            name: spec["default"] for name, spec in self.TUNABLE_PARAMETERS.items()
        }
        self.performance_baseline: Dict[str, float] = {}
        self.optimization_targets: List[OptimizationTarget] = [
            OptimizationTarget.UNITY_INDEX,
            OptimizationTarget.COHERENCE
        ]
        self.optimization_mode = "explore"  # "explore" or "exploit"
        self.consecutive_improvements = 0
        
        # Initialize metric history buffers
        for target in OptimizationTarget:
            self.metrics_history[target.value] = deque(maxlen=100)
        
        print("⚙️ [OPTIM]: Self-Optimization Engine initialized")
    
    def record_metric(self, name: str, value: float, context: Dict = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            context=context or {}
        )
        
        if name not in self.metrics_history:
            self.metrics_history[name] = deque(maxlen=100)
        
        self.metrics_history[name].append(metric)
        
        # Update baseline if this is the first measurement
        if name not in self.performance_baseline:
            self.performance_baseline[name] = value
    
    def get_metric_trend(self, name: str, window: int = 10) -> Tuple[float, str]:
        """
        Calculate trend for a metric.
        Returns (slope, direction) where direction is "improving", "declining", or "stable".
        """
        if name not in self.metrics_history or len(self.metrics_history[name]) < 2:
            return 0.0, "unknown"
        
        history = list(self.metrics_history[name])[-window:]
        if len(history) < 2:
            return 0.0, "stable"
        
        # Linear regression for trend
        n = len(history)
        x_mean = (n - 1) / 2
        y_mean = sum(m.value for m in history) / n
        
        numerator = sum((i - x_mean) * (m.value - y_mean) for i, m in enumerate(history))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0, "stable"
        
        slope = numerator / denominator
        
        if slope > 0.01:
            direction = "improving"
        elif slope < -0.01:
            direction = "declining"
        else:
            direction = "stable"
        
        return slope, direction
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Detect performance bottlenecks by analyzing metric patterns.
        Returns list of bottleneck descriptions.
        """
        bottlenecks = []
        
        for name, history in self.metrics_history.items():
            if len(history) < 5:
                continue
            
            recent_values = [m.value for m in list(history)[-10:]]
            avg = sum(recent_values) / len(recent_values)
            variance = sum((v - avg) ** 2 for v in recent_values) / len(recent_values)
            
            # High variance indicates instability
            if variance > 0.05:
                bottlenecks.append({
                    "type": "instability",
                    "metric": name,
                    "variance": variance,
                    "recommendation": f"Consider increasing stability parameters for {name}"
                })
            
            # Below baseline indicates degradation
            if name in self.performance_baseline:
                baseline = self.performance_baseline[name]
                if avg < baseline * 0.9:
                    bottlenecks.append({
                        "type": "degradation",
                        "metric": name,
                        "current": avg,
                        "baseline": baseline,
                        "recommendation": f"Investigate {name} degradation, consider parameter reset"
                    })
            
            # Check for plateau
            slope, direction = self.get_metric_trend(name)
            if direction == "stable" and avg < 0.85:  # Plateaued below target
                bottlenecks.append({
                    "type": "plateau",
                    "metric": name,
                    "value": avg,
                    "recommendation": f"Increase exploration to escape {name} plateau"
                })
        
        return bottlenecks
    
    def optimize_step(self) -> Optional[OptimizationAction]:
        """
        Perform one optimization step using Golden Section Search.
        Returns the action taken, or None if no optimization needed.
        """
        # Analyze current performance
        bottlenecks = self.detect_bottlenecks()
        
        # Select parameter to optimize
        if bottlenecks:
            # Prioritize fixing bottlenecks
            bottleneck = bottlenecks[0]
            param_to_tune = self._select_param_for_bottleneck(bottleneck)
        else:
            # Random exploration
            param_to_tune = random.choice(list(self.TUNABLE_PARAMETERS.keys()))
        
        if param_to_tune is None:
            return None
        
        # Get current value and bounds
        current = self.current_parameters[param_to_tune]
        bounds = self.TUNABLE_PARAMETERS[param_to_tune]
        min_val, max_val = bounds["min"], bounds["max"]
        
        # Golden Section step
        if self.optimization_mode == "explore":
            # Larger random perturbation
            delta = (max_val - min_val) * random.uniform(-0.1, 0.1) * PHI
        else:
            # Smaller, directed step
            trend = self._get_parameter_gradient(param_to_tune)
            delta = trend * (max_val - min_val) * TAU * 0.1
        
        new_value = max(min_val, min(max_val, current + delta))
        
        # Create and record action
        action = OptimizationAction(
            parameter=param_to_tune,
            old_value=current,
            new_value=new_value,
            reason=f"{'Bottleneck fix' if bottlenecks else 'Exploration'}: {param_to_tune}",
            expected_improvement=abs(delta) / (max_val - min_val)
        )
        
        # Apply the change
        self.current_parameters[param_to_tune] = new_value
        self.actions_history.append(action)
        
        print(f"⚙️ [OPTIM]: {param_to_tune}: {current:.4f} → {new_value:.4f}")
        
        return action
    
    def _select_param_for_bottleneck(self, bottleneck: Dict) -> Optional[str]:
        """Select parameter to tune based on bottleneck type."""
        bottleneck_type = bottleneck.get("type")
        metric = bottleneck.get("metric", "")
        
        # Mapping from issues to parameters
        if bottleneck_type == "instability":
            return "learning_rate"  # Reduce for stability
        elif bottleneck_type == "degradation":
            if "unity" in metric.lower():
                return "validation_threshold"
            elif "learning" in metric.lower():
                return "learning_rate"
            else:
                return "coherence_target"
        elif bottleneck_type == "plateau":
            return "exploration_rate"  # Increase to escape
        
        return None
    
    def _get_parameter_gradient(self, param: str) -> float:
        """
        Estimate gradient for parameter based on recent actions.
        Returns positive if increasing helped, negative if decreasing helped.
        """
        relevant_actions = [
            a for a in self.actions_history[-20:]
            if a.parameter == param and a.result is not None
        ]
        
        if not relevant_actions:
            return 0.0
        
        # Weighted average of direction × result
        gradient = 0.0
        total_weight = 0.0
        
        for action in relevant_actions:
            direction = 1.0 if action.new_value > action.old_value else -1.0
            weight = 1.0 / (time.time() - action.timestamp + 1)  # Recency weighting
            gradient += direction * action.result * weight
            total_weight += weight
        
        return gradient / total_weight if total_weight > 0 else 0.0
    
    def evaluate_last_action(self, performance_delta: float):
        """
        Evaluate the effect of the last optimization action.
        Call this after measuring performance following an optimization.
        """
        if not self.actions_history:
            return
        
        last_action = self.actions_history[-1]
        last_action.result = performance_delta
        
        # Update optimization mode based on results
        if performance_delta > 0:
            self.consecutive_improvements += 1
            if self.consecutive_improvements >= 3:
                self.optimization_mode = "exploit"
        else:
            self.consecutive_improvements = 0
            self.optimization_mode = "explore"
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate a report of optimization state and history."""
        # Calculate success rate
        evaluated_actions = [a for a in self.actions_history if a.result is not None]
        success_rate = sum(1 for a in evaluated_actions if a.result > 0) / max(1, len(evaluated_actions))
        
        # Current performance summary
        current_perf = {}
        for name, history in self.metrics_history.items():
            if history:
                recent = list(history)[-5:]
                current_perf[name] = {
                    "current": round(recent[-1].value, 4),
                    "avg_5": round(sum(m.value for m in recent) / len(recent), 4),
                    "trend": self.get_metric_trend(name)[1]
                }
        
        return {
            "mode": self.optimization_mode,
            "consecutive_improvements": self.consecutive_improvements,
            "total_actions": len(self.actions_history),
            "success_rate": round(success_rate, 3),
            "current_parameters": {k: round(v, 4) for k, v in self.current_parameters.items()},
            "bottlenecks": self.detect_bottlenecks(),
            "performance": current_perf
        }
    
    def auto_optimize(self, performance_metric: str, iterations: int = 5) -> List[float]:
        """
        Run automatic optimization loop.
        Returns list of performance values after each iteration.
        """
        results = []
        
        for i in range(iterations):
            # Record current performance (simulated for standalone use)
            if performance_metric in self.metrics_history and self.metrics_history[performance_metric]:
                current_perf = list(self.metrics_history[performance_metric])[-1].value
            else:
                current_perf = random.uniform(0.7, 0.9)  # Simulated
            
            # Optimize
            action = self.optimize_step()
            
            # Simulate new performance (in real use, this comes from actual measurement)
            new_perf = current_perf + random.uniform(-0.05, 0.1)  # Simulated
            new_perf = max(0.5, min(1.0, new_perf))
            
            # Record and evaluate
            self.record_metric(performance_metric, new_perf)
            if action:
                self.evaluate_last_action(new_perf - current_perf)
            
            results.append(new_perf)
            print(f"  Iteration {i+1}: {performance_metric} = {new_perf:.4f}")
        
        return results
    
    def get_parameter(self, name: str) -> float:
        """Get current value of a tunable parameter."""
        return self.current_parameters.get(name, self.TUNABLE_PARAMETERS.get(name, {}).get("default", 0.5))
    
    def set_parameter(self, name: str, value: float) -> bool:
        """Manually set a parameter value."""
        if name not in self.TUNABLE_PARAMETERS:
            return False
        
        bounds = self.TUNABLE_PARAMETERS[name]
        value = max(bounds["min"], min(bounds["max"], value))
        self.current_parameters[name] = value
        return True
    
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        self.current_parameters = {
            name: spec["default"] for name, spec in self.TUNABLE_PARAMETERS.items()
        }
        self.optimization_mode = "explore"
        self.consecutive_improvements = 0
        print("⚙️ [OPTIM]: Parameters reset to defaults")
    
    def save_state(self, filepath: str = "l104_optimization_state.json"):
        """Save optimization state to disk."""
        state = {
            "version": "1.0.0",
            "parameters": self.current_parameters,
            "baseline": self.performance_baseline,
            "mode": self.optimization_mode,
            "actions_count": len(self.actions_history),
            "recent_actions": [
                {
                    "parameter": a.parameter,
                    "old": a.old_value,
                    "new": a.new_value,
                    "result": a.result
                }
                for a in self.actions_history[-20:]
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"💾 [OPTIM]: State saved to {filepath}")
    
    def load_state(self, filepath: str = "l104_optimization_state.json"):
        """Load optimization state from disk."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_parameters = state.get("parameters", self.current_parameters)
            self.performance_baseline = state.get("baseline", {})
            self.optimization_mode = state.get("mode", "explore")
            
            print(f"📂 [OPTIM]: State loaded from {filepath}")
        except FileNotFoundError:
            print(f"⚠️ [OPTIM]: No state file found at {filepath}")


# Singleton instance
self_optimizer = SelfOptimizationEngine()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = SelfOptimizationEngine()
    
    print("\n⚙️ Initial Parameters:")
    for name, value in engine.current_parameters.items():
        print(f"  {name}: {value}")
    
    print("\n🔄 Running Auto-Optimization (10 iterations)...")
    results = engine.auto_optimize("unity_index", iterations=10)
    
    print("\n📊 Optimization Report:")
    report = engine.get_optimization_report()
    for key, value in report.items():
        if key != "performance":
            print(f"  {key}: {value}")
    
    print("\n📈 Final Performance:")
    for metric, data in report.get("performance", {}).items():
        print(f"  {metric}: {data}")
    
    engine.save_state()
