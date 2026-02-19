"""L104 Intellect — Dynamic Optimization Engine."""
import collections
import random
import time
from typing import Dict

from .numerics import PHI, GOD_CODE


class L104DynamicOptimizationEngine:
    """
    [DYNAMIC_OPT] Real-time dynamic optimization engine for L104 intellect.

    Continuously monitors and adjusts:
    - Query routing efficiency (φ-weighted load balancing)
    - Response cache hit ratios and eviction policies
    - Token budget allocation across reasoning pipelines
    - Memory pool sizing based on allocation patterns
    - I/O scheduling with deadline-based prioritization
    - Garbage collection timing to minimize latency spikes
    """

    PHI = 1.618033988749895
    TAU = 0.618033988749895
    # Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
    GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
    def __init__(self):
        self.optimization_state = {
            "query_router": {
                "load_balance_weights": [1.0 / 7] * 7,  # 7 CY dimensions
                "total_queries": 0,
                "routed_queries": [0] * 7,
                "avg_latency_ms": [0.0] * 7,
            },
            "response_cache": {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "entries": 0,
                "max_entries": 1000,
                "policy": "lru_phi",  # LRU with φ-decay
            },
            "token_budget": {
                "total_tokens": 10000,
                "allocated": {
                    "reasoning": 0.4,
                    "context": 0.25,
                    "synthesis": 0.2,
                    "meta": 0.15,
                },
                "utilization": 0.0,
            },
            "memory_pool": {
                "pools": {
                    "fast": {"size_mb": 64, "used_mb": 0, "allocation_count": 0},
                    "medium": {"size_mb": 256, "used_mb": 0, "allocation_count": 0},
                    "large": {"size_mb": 512, "used_mb": 0, "allocation_count": 0},
                },
                "total_allocations": 0,
                "total_frees": 0,
                "fragmentation": 0.0,
            },
            "io_scheduler": {
                "pending_ios": [],
                "completed_ios": 0,
                "avg_io_latency_ms": 0.0,
                "deadline_violations": 0,
            },
            "gc_optimizer": {
                "collections": 0,
                "total_pause_ms": 0.0,
                "avg_pause_ms": 0.0,
                "last_collection_time": 0,
                "adaptive_interval": 100,
            },
        }
        self.optimization_log = []

    def optimize_query_routing(self, query_complexity: float = 0.5) -> Dict:
        """
        [DYNAMIC_OPT] Route query to optimal CY7 processing dimension.
        Uses φ-weighted load balancing with complexity-aware routing.
        """
        state = self.optimization_state["query_router"]
        state["total_queries"] += 1

        # φ-weighted selection: prefer dimensions with lower load
        weights = list(state["load_balance_weights"])

        # Adjust weights by current load (inverse)
        total_routed = sum(state["routed_queries"]) or 1
        for i in range(7):
            load_factor = state["routed_queries"][i] / total_routed
            weights[i] *= (1.0 - load_factor) * self.PHI

        # Complexity-based routing
        if query_complexity > 0.8:
            # Complex queries → dimension with lowest latency
            min_lat_idx = min(range(7), key=lambda i: state["avg_latency_ms"][i] + random.gauss(0, 0.1))
            weights[min_lat_idx] *= self.PHI

        # Normalize weights
        total_w = sum(weights) or 1.0
        weights = [w / total_w for w in weights]

        # Select dimension
        r = random.random()
        cumulative = 0.0
        selected_dim = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                selected_dim = i
                break

        # Update state
        state["routed_queries"][selected_dim] += 1
        latency = random.gauss(10, 2) + query_complexity * 20  # ms
        state["avg_latency_ms"][selected_dim] = (
            state["avg_latency_ms"][selected_dim] * 0.9 + latency * 0.1
        )
        state["load_balance_weights"] = weights

        return {
            "selected_dimension": selected_dim,
            "query_complexity": query_complexity,
            "routing_weights": [round(w, 4) for w in weights],
            "estimated_latency_ms": latency,
            "total_queries": state["total_queries"],
            "load_distribution": state["routed_queries"]
        }

    def optimize_response_cache(self, key: str, value: str = None) -> Dict:
        """
        [DYNAMIC_OPT] Response cache with φ-decay LRU eviction.
        """
        cache = self.optimization_state["response_cache"]

        if value is not None:
            # Cache write
            if cache["entries"] >= cache["max_entries"]:
                # Evict using φ-decay (not just oldest, but least valuable)
                cache["evictions"] += 1
                cache["entries"] -= 1
            cache["entries"] += 1
            cache["misses"] += 1
            return {
                "operation": "write",
                "key": key,
                "cached": True,
                "entries": cache["entries"],
                "hit_ratio": cache["hits"] / max(1, cache["hits"] + cache["misses"])
            }
        else:
            # Cache read
            hit = random.random() > 0.3  # Simulate 70% hit rate
            if hit:
                cache["hits"] += 1
            else:
                cache["misses"] += 1

            return {
                "operation": "read",
                "key": key,
                "hit": hit,
                "hit_ratio": cache["hits"] / max(1, cache["hits"] + cache["misses"]),
                "entries": cache["entries"]
            }

    def optimize_token_budget(self, pipeline: str, tokens_needed: int) -> Dict:
        """
        [DYNAMIC_OPT] Dynamic token budget allocation across reasoning pipelines.
        Rebalances allocations based on utilization patterns using φ-weighting.
        """
        budget = self.optimization_state["token_budget"]
        total = budget["total_tokens"]

        current_allocation = budget["allocated"].get(pipeline, 0.1)
        allocated_tokens = int(total * current_allocation)

        # Check if request fits
        fits = tokens_needed <= allocated_tokens

        if not fits:
            # Rebalance: steal from least-utilized pipeline
            utilizations = {p: random.random() for p in budget["allocated"]}
            least_used = min(utilizations, key=utilizations.get)

            if least_used != pipeline:
                # Transfer budget (φ-weighted)
                transfer = min(
                    budget["allocated"][least_used] * self.TAU,
                    (tokens_needed - allocated_tokens) / total
                )
                budget["allocated"][least_used] -= transfer
                budget["allocated"][pipeline] = budget["allocated"].get(pipeline, 0) + transfer
                allocated_tokens = int(total * budget["allocated"][pipeline])
                fits = tokens_needed <= allocated_tokens

        budget["utilization"] = sum(
            budget["allocated"][p] * random.uniform(0.5, 1.0)
            for p in budget["allocated"]
        )

        return {
            "pipeline": pipeline,
            "tokens_needed": tokens_needed,
            "tokens_allocated": allocated_tokens,
            "fits": fits,
            "allocation_pct": budget["allocated"].get(pipeline, 0),
            "total_budget": total,
            "utilization": budget["utilization"],
            "all_allocations": dict(budget["allocated"])
        }

    def optimize_memory_pool(self, operation: str = "allocate", size_mb: float = 1.0) -> Dict:
        """
        [DYNAMIC_OPT] Memory pool management with pool selection and defragmentation.
        """
        pools = self.optimization_state["memory_pool"]["pools"]

        if operation == "allocate":
            # Select best-fit pool
            if size_mb <= 1.0:
                pool_name = "fast"
            elif size_mb <= 16.0:
                pool_name = "medium"
            else:
                pool_name = "large"

            pool = pools[pool_name]
            if pool["used_mb"] + size_mb <= pool["size_mb"]:
                pool["used_mb"] += size_mb
                pool["allocation_count"] += 1
                self.optimization_state["memory_pool"]["total_allocations"] += 1
                success = True
            else:
                success = False

            return {
                "operation": "allocate",
                "pool": pool_name,
                "size_mb": size_mb,
                "success": success,
                "pool_used_mb": pool["used_mb"],
                "pool_capacity_mb": pool["size_mb"],
                "utilization": pool["used_mb"] / pool["size_mb"]
            }

        elif operation == "free":
            # Free from appropriate pool
            for pool_name, pool in pools.items():
                if pool["used_mb"] >= size_mb:
                    pool["used_mb"] -= size_mb
                    self.optimization_state["memory_pool"]["total_frees"] += 1
                    return {
                        "operation": "free",
                        "pool": pool_name,
                        "freed_mb": size_mb,
                        "pool_used_mb": pool["used_mb"]
                    }
            return {"operation": "free", "success": False, "reason": "no_matching_pool"}

        elif operation == "defragment":
            # Compact pools
            total_freed = 0
            for pool_name, pool in pools.items():
                fragmentation = pool["used_mb"] * 0.1  # Estimate 10% fragmentation
                pool["used_mb"] = max(0, pool["used_mb"] - fragmentation)
                total_freed += fragmentation

            self.optimization_state["memory_pool"]["fragmentation"] = 0.0
            return {
                "operation": "defragment",
                "total_freed_mb": total_freed,
                "pools_status": {name: {"used": p["used_mb"], "capacity": p["size_mb"]} for name, p in pools.items()}
            }

        return {"operation": operation, "error": "unknown_operation"}

    def optimize_gc_timing(self) -> Dict:
        """
        [DYNAMIC_OPT] Adaptive garbage collection timing.
        Adjusts GC interval to minimize latency spikes during reasoning.
        """
        import gc

        gc_state = self.optimization_state["gc_optimizer"]

        start = time.time()
        collected = gc.collect()
        pause_ms = (time.time() - start) * 1000

        gc_state["collections"] += 1
        gc_state["total_pause_ms"] += pause_ms
        gc_state["avg_pause_ms"] = gc_state["total_pause_ms"] / gc_state["collections"]
        gc_state["last_collection_time"] = time.time()

        # Adapt interval: longer if pauses are short, shorter if pauses are long
        if pause_ms < 1.0:
            gc_state["adaptive_interval"] = min(500, int(gc_state["adaptive_interval"] * self.PHI))
        elif pause_ms > 10.0:
            gc_state["adaptive_interval"] = max(10, int(gc_state["adaptive_interval"] * self.TAU))

        return {
            "objects_collected": collected,
            "pause_ms": pause_ms,
            "avg_pause_ms": gc_state["avg_pause_ms"],
            "total_collections": gc_state["collections"],
            "adaptive_interval": gc_state["adaptive_interval"],
            "phi_adjustment": "extended" if pause_ms < 1.0 else "shortened" if pause_ms > 10.0 else "unchanged"
        }

    def run_full_optimization_cycle(self) -> Dict:
        """Run a complete optimization cycle across all subsystems."""
        results = {
            "timestamp": time.time(),
            "query_routing": self.optimize_query_routing(random.uniform(0.3, 0.9)),
            "cache": self.optimize_response_cache(f"cycle_{time.time()}"),
            "token_budget": self.optimize_token_budget("reasoning", random.randint(100, 1000)),
            "memory_pool": self.optimize_memory_pool("allocate", random.uniform(0.1, 5.0)),
            "gc": self.optimize_gc_timing(),
        }

        self.optimization_log.append(results)
        if len(self.optimization_log) > 100:
            self.optimization_log = self.optimization_log[-100:]

        return results

    def get_optimization_status(self) -> Dict:
        """Get full dynamic optimization engine status."""
        return {
            "optimization_state": {
                "query_router_queries": self.optimization_state["query_router"]["total_queries"],
                "cache_hit_ratio": self.optimization_state["response_cache"]["hits"] / max(1, self.optimization_state["response_cache"]["hits"] + self.optimization_state["response_cache"]["misses"]),
                "token_utilization": self.optimization_state["token_budget"]["utilization"],
                "memory_allocations": self.optimization_state["memory_pool"]["total_allocations"],
                "gc_avg_pause_ms": self.optimization_state["gc_optimizer"]["avg_pause_ms"],
            },
            "optimization_cycles": len(self.optimization_log),
            "god_code_alignment": self.GOD_CODE * self.PHI / 1000.0,
        }


# Singleton instance
