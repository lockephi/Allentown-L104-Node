"""
L104 AGI Core — Adaptive Pipeline Scheduler v1.0
============================================================================
Intelligent scheduling and learning rate control for the AGI pipeline.

Provides:
  • PHI-decay learning rate scheduler (cosine annealing with golden ratio)
  • Experience replay buffer with importance sampling
  • Predictive pipeline scheduling (pattern-frequency anticipation)
  • Resource budget allocation (subsystem priority × PHI weighting)

All scheduling decisions are modulated by sacred constants.
INVARIANT: 527.5184818492612 | PILOT: LONDEL
============================================================================
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

# Sacred constants
PHI = 1.618033988749895
TAU = 1.0 / PHI  # ≈ 0.618
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # 527.5184818492612
VOID_CONSTANT = 1.0416180339887497
FEIGENBAUM = 4.669201609102990


# ═══════════════════════════════════════════════════════════════════════════════
# PHI-DECAY LEARNING RATE SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class PhiLearningScheduler:
    """
    Learning rate scheduler with golden-ratio cosine annealing.

    The learning rate follows:
        lr(t) = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π × t / T_φ))

    where T_φ = T_base × PHI^epoch — the period grows by PHI each epoch,
    providing increasingly patient exploration as the system matures.

    Includes momentum tracking with TAU-exponential smoothing and
    automatic warmup phase detection.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        lr_max: float = 0.01,
        lr_min: float = 0.0001,
        base_period: int = 104,    # L104 signature — steps per first cycle
        warmup_steps: int = 10,
    ):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.base_period = base_period
        self.warmup_steps = warmup_steps

        self._step: int = 0
        self._epoch: int = 0
        self._current_lr: float = lr_min  # starts at min during warmup
        self._momentum: float = 0.0
        self._lr_history: deque = deque(maxlen=5000)
        self._best_metric: float = float('-inf')
        self._patience_counter: int = 0

    def step(self, metric: Optional[float] = None) -> float:
        """
        Advance one step and return the new learning rate.

        Args:
            metric: Optional performance metric for plateau detection.
        """
        self._step += 1

        # Warmup phase — linear ramp from lr_min to lr_max
        if self._step <= self.warmup_steps:
            frac = self._step / max(self.warmup_steps, 1)
            self._current_lr = self.lr_min + (self.lr_max - self.lr_min) * frac
        else:
            # PHI-expanding cosine annealing
            period = self.base_period * (PHI ** self._epoch)
            t_in_epoch = (self._step - self.warmup_steps) % max(int(period), 1)
            cosine = math.cos(math.pi * t_in_epoch / max(period, 1))
            self._current_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + cosine)

            # Epoch rollover
            if t_in_epoch == 0 and self._step > self.warmup_steps + 1:
                self._epoch += 1

        # Momentum tracking (TAU-exponential smoothing of lr deltas)
        if len(self._lr_history) > 0:
            prev_lr = self._lr_history[-1]["lr"]
            delta = self._current_lr - prev_lr
            self._momentum = TAU * delta + (1 - TAU) * self._momentum

        # Plateau detection
        if metric is not None:
            if metric > self._best_metric:
                self._best_metric = metric
                self._patience_counter = 0
            else:
                self._patience_counter += 1

        self._lr_history.append({
            "step": self._step,
            "lr": round(self._current_lr, 8),
            "momentum": round(self._momentum, 8),
            "epoch": self._epoch,
        })

        return self._current_lr

    @property
    def lr(self) -> float:
        return self._current_lr

    @property
    def is_warming_up(self) -> bool:
        return self._step <= self.warmup_steps

    @property
    def plateau_patience(self) -> int:
        return self._patience_counter

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "step": self._step,
            "epoch": self._epoch,
            "lr": round(self._current_lr, 8),
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
            "momentum": round(self._momentum, 8),
            "warmup_steps": self.warmup_steps,
            "is_warming_up": self.is_warming_up,
            "base_period": self.base_period,
            "current_period": round(self.base_period * (PHI ** self._epoch), 2),
            "best_metric": round(self._best_metric, 6) if self._best_metric != float('-inf') else None,
            "plateau_patience": self._patience_counter,
            "history_size": len(self._lr_history),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIENCE REPLAY WITH IMPORTANCE SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

class ExperienceReplayBuffer:
    """
    Experience replay buffer with importance sampling for pipeline decisions.

    Stores (state, action, reward, next_state) tuples from pipeline solve cycles.
    Supports prioritized sampling where higher-reward experiences are replayed
    more frequently (importance weight ∝ |reward| ^ PHI).

    Used to reinforce successful routing decisions and avoid repeating failures.
    """

    VERSION = "1.0.0"

    def __init__(self, capacity: int = 10000, priority_exponent: float = PHI):
        self._capacity = capacity
        self._priority_exponent = priority_exponent
        self._buffer: deque = deque(maxlen=capacity)
        self._priorities: deque = deque(maxlen=capacity)
        self._total_stored: int = 0

    def store(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store an experience tuple."""
        experience = {
            "time": time.time(),
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "metadata": metadata or {},
        }
        # Priority: absolute reward raised to PHI power
        priority = abs(reward) ** self._priority_exponent + 1e-10
        self._buffer.append(experience)
        self._priorities.append(priority)
        self._total_stored += 1

    def sample(self, batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Sample a batch with importance-weighted probability.
        Higher-reward experiences are sampled more frequently.
        """
        if len(self._buffer) == 0:
            return []

        n = min(batch_size, len(self._buffer))
        priorities = list(self._priorities)
        total_priority = sum(priorities)

        if total_priority <= 0:
            # Uniform sampling fallback
            indices = random.sample(range(len(self._buffer)), n)
        else:
            # Weighted sampling
            probs = [p / total_priority for p in priorities]
            indices = []
            for _ in range(n):
                r = random.random()
                cumulative = 0.0
                for i, p in enumerate(probs):
                    cumulative += p
                    if r <= cumulative:
                        indices.append(i)
                        break
                else:
                    indices.append(len(probs) - 1)

        # Deduplicate indices
        indices = list(set(indices))
        buffer_list = list(self._buffer)
        return [buffer_list[i] for i in indices]

    def top_experiences(self, k: int = 10) -> List[Dict[str, Any]]:
        """Return top-K experiences by reward."""
        sorted_buf = sorted(self._buffer, key=lambda x: x["reward"], reverse=True)
        return sorted_buf[:k]

    def reward_stats(self) -> Dict[str, Any]:
        """Get reward distribution statistics."""
        if not self._buffer:
            return {"count": 0}
        rewards = [e["reward"] for e in self._buffer]
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / max(len(rewards) - 1, 1)
        return {
            "count": len(rewards),
            "total_stored": self._total_stored,
            "mean_reward": round(mean, 6),
            "stddev_reward": round(math.sqrt(max(0, variance)), 6),
            "min_reward": round(min(rewards), 6),
            "max_reward": round(max(rewards), 6),
            "positive_ratio": round(sum(1 for r in rewards if r > 0) / len(rewards), 4),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "capacity": self._capacity,
            "size": len(self._buffer),
            "total_stored": self._total_stored,
            "priority_exponent": round(self._priority_exponent, 4),
            "fill_ratio": round(len(self._buffer) / self._capacity, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTIVE PIPELINE SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class PredictivePipelineScheduler:
    """
    Anticipatory resource pre-allocation based on subsystem call patterns.

    Tracks per-subsystem call frequency over a sliding window and predicts
    the next-likely subsystems to be activated. Uses Bayesian-like frequency
    estimation with PHI-weighted temporal decay.

    Subsystems predicted as "likely next" can be pre-warmed.
    """

    VERSION = "1.0.0"

    def __init__(self, history_window: int = 2000, prediction_threshold: float = 0.05):
        self._history: deque = deque(maxlen=history_window)
        self._prediction_threshold = prediction_threshold
        # Transition matrix: state[last_subsystem] → {next_subsystem → count}
        self._transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._last_subsystem: Optional[str] = None
        self._total_calls: int = 0
        self._predictions_made: int = 0
        self._predictions_correct: int = 0

    def record_call(self, subsystem: str):
        """Record a subsystem call and update transition matrix."""
        self._total_calls += 1
        self._call_counts[subsystem] += 1
        self._history.append({
            "time": time.time(),
            "subsystem": subsystem,
        })

        if self._last_subsystem is not None:
            self._transitions[self._last_subsystem][subsystem] += 1

        self._last_subsystem = subsystem

    def predict_next(self, current: Optional[str] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the next-likely subsystem(s) to be called.

        Uses transition probabilities from the current subsystem,
        blended with global frequency prior (PHI-weighted blend).

        Returns list of (subsystem, probability) tuples.
        """
        self._predictions_made += 1
        src = current or self._last_subsystem
        if src is None or self._total_calls < 10:
            # Not enough data — return global frequency
            return self._global_frequency(top_k)

        # Transition probabilities from current state
        trans = self._transitions.get(src, {})
        total_trans = sum(trans.values())

        if total_trans == 0:
            return self._global_frequency(top_k)

        # Local transition probs
        local_probs = {k: v / total_trans for k, v in trans.items()}

        # Global frequency prior
        global_probs = {k: v / self._total_calls for k, v in self._call_counts.items()}

        # PHI-weighted blend: TAU × local + (1 - TAU) × global
        all_subsystems = set(list(local_probs.keys()) + list(global_probs.keys()))
        blended = {}
        for ss in all_subsystems:
            local = local_probs.get(ss, 0.0)
            global_p = global_probs.get(ss, 0.0)
            blended[ss] = TAU * local + (1 - TAU) * global_p

        # Filter by threshold and sort
        filtered = [(k, v) for k, v in blended.items() if v >= self._prediction_threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_k]

    def _global_frequency(self, top_k: int) -> List[Tuple[str, float]]:
        """Fallback: global frequency prior."""
        if self._total_calls == 0:
            return []
        probs = [(k, v / self._total_calls) for k, v in self._call_counts.items()]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]

    def verify_prediction(self, predicted: str, actual: str):
        """Track prediction accuracy."""
        if predicted == actual:
            self._predictions_correct += 1

    def accuracy(self) -> float:
        """Prediction accuracy."""
        if self._predictions_made == 0:
            return 0.0
        return self._predictions_correct / self._predictions_made

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "total_calls": self._total_calls,
            "unique_subsystems": len(self._call_counts),
            "history_size": len(self._history),
            "transition_states": len(self._transitions),
            "predictions_made": self._predictions_made,
            "predictions_correct": self._predictions_correct,
            "accuracy": round(self.accuracy(), 4),
            "prediction_threshold": self._prediction_threshold,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCE BUDGET ALLOCATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ResourceBudgetAllocator:
    """
    Allocates compute budget across pipeline subsystems.

    Each subsystem has a priority (PHI-weighted) and a soft budget cap.
    The allocator distributes the total budget proportionally to priority,
    with GOD_CODE-aligned fairness correction.

    Budget units are abstract (can map to CPU-ms, memory-MB, or call-count).
    """

    VERSION = "1.0.0"

    def __init__(self, total_budget: float = 1040.0):
        """
        Args:
            total_budget: Total budget units to distribute (default: 1040 = 10 × L104).
        """
        self._total_budget = total_budget
        self._priorities: Dict[str, float] = {}
        self._allocated: Dict[str, float] = {}
        self._consumed: Dict[str, float] = defaultdict(float)
        self._allocation_epoch: int = 0

    def set_priority(self, subsystem: str, priority: float):
        """Set subsystem priority (higher = more budget)."""
        self._priorities[subsystem] = max(0.0, priority)

    def set_priorities_from_pagerank(self, pagerank: Dict[str, float]):
        """Bulk-set priorities from PageRank scores."""
        for ss, score in pagerank.items():
            self._priorities[ss] = score * PHI  # PHI-amplified

    def allocate(self) -> Dict[str, float]:
        """
        Distribute total budget across subsystems proportionally to priority.
        Includes GOD_CODE fairness correction: each subsystem gets at least
        total_budget / (104 × n) as a minimum floor.
        """
        self._allocation_epoch += 1

        if not self._priorities:
            self._allocated = {}
            return {}

        n = len(self._priorities)
        total_priority = sum(self._priorities.values())
        min_floor = self._total_budget / (104 * n) if n > 0 else 0.0

        if total_priority <= 0:
            # Equal distribution
            share = self._total_budget / n
            self._allocated = {ss: round(share, 4) for ss in self._priorities}
            return dict(self._allocated)

        # Proportional allocation with floor
        raw = {ss: (p / total_priority) * self._total_budget for ss, p in self._priorities.items()}
        allocated = {ss: max(budget, min_floor) for ss, budget in raw.items()}

        # Re-normalize to total budget
        alloc_total = sum(allocated.values())
        if alloc_total > 0:
            scale = self._total_budget / alloc_total
            allocated = {ss: round(b * scale, 4) for ss, b in allocated.items()}

        self._allocated = allocated
        self._consumed = defaultdict(float)  # Reset consumption on re-allocation
        return dict(allocated)

    def consume(self, subsystem: str, amount: float) -> bool:
        """
        Consume budget for a subsystem.
        Returns False if budget exhausted (soft limit — does not block).
        """
        self._consumed[subsystem] += amount
        allocated = self._allocated.get(subsystem, 0)
        return self._consumed[subsystem] <= allocated

    def remaining(self, subsystem: str) -> float:
        """Get remaining budget for a subsystem."""
        allocated = self._allocated.get(subsystem, 0)
        return max(0, allocated - self._consumed.get(subsystem, 0))

    def utilization(self) -> Dict[str, float]:
        """Get budget utilization ratio for each subsystem."""
        result = {}
        for ss in self._allocated:
            allocated = self._allocated[ss]
            consumed = self._consumed.get(ss, 0)
            result[ss] = round(consumed / allocated, 4) if allocated > 0 else 0.0
        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "total_budget": self._total_budget,
            "allocation_epoch": self._allocation_epoch,
            "subsystems": len(self._priorities),
            "allocated": dict(self._allocated),
            "consumed": dict(self._consumed),
            "utilization": self.utilization(),
        }
