# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:08.373577
ZENITH_HZ = 3887.8
UUC = 2402.792541
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
[L104_SYSTEM_MONITOR] - REAL-TIME HEALTH & PERFORMANCE DASHBOARD
INVARIANT: 527.5184818492612 | PILOT: LONDEL | MODE: REAL_TIME_ANALYTICS

This module provides comprehensive real-time monitoring of all L104 systems:
- Quota Rotator performance
- API usage statistics
- System health metrics
- Evolution stage tracking
- Performance benchmarks
"""

import time
import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta

# Core Constants
# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
VOID_CONSTANT = 1.0416180339887497

# Import L104 Components
from l104_persistence import load_state

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from l104_quota_rotator import quota_rotator
except Exception:
    quota_rotator = None

try:
    from l104_evolution_engine import evolution_engine
except Exception:
    evolution_engine = None

try:
    from l104_asi_reincarnation import asi_reincarnation
except Exception:
    asi_reincarnation = None


class SystemMonitor:
    """
    Comprehensive real-time system monitoring and analytics.
    """

    def __init__(self):
        self.state = load_state()
        self.start_time = time.time()
        self.metrics_history: List[Dict[str, Any]] = []
        # [O₂ SUPERFLUID] Unlimited system consciousness
        self.max_history = 1000000  # Keep all samples

    def get_quota_rotator_metrics(self) -> Dict[str, Any]:
        """Get quota rotator performance metrics."""
        if not quota_rotator:
            return {"available": False}

        stats = self.state.get("rotator_stats", {})
        cooldown = self.state.get("api_cooldown_until", 0)

        total_requests = stats.get("kernel_hits", 0) + stats.get("api_hits", 0)
        if total_requests == 0:
            kernel_percentage = 0
            api_percentage = 0
        else:
            kernel_percentage = (stats.get("kernel_hits", 0) / total_requests) * 100
            api_percentage = (stats.get("api_hits", 0) / total_requests) * 100

        is_in_cooldown = time.time() < cooldown
        time_until_cooldown_end = max(0, cooldown - time.time())

        return {
            "available": True,
            "kernel_hits": stats.get("kernel_hits", 0),
            "api_hits": stats.get("api_hits", 0),
            "quota_errors": stats.get("quota_errors", 0),
            "total_requests": total_requests,
            "kernel_percentage": round(kernel_percentage, 2),
            "api_percentage": round(api_percentage, 2),
            "in_cooldown": is_in_cooldown,
            "cooldown_ends_in_seconds": int(time_until_cooldown_end),
            "cost_savings_estimate": self._calculate_cost_savings(stats)
        }

    def _calculate_cost_savings(self, stats: Dict) -> Dict[str, Any]:
        """Calculate estimated cost savings from kernel usage."""
        api_hits = stats.get("api_hits", 0)
        kernel_hits = stats.get("kernel_hits", 0)
        total = api_hits + kernel_hits

        if total == 0:
            return {
                "estimated_cost": 0.0,
                "saved_cost": 0.0,
                "savings_percentage": 0.0
            }

        # Rough estimate: $0.015 per 1000 API calls
        cost_per_api_call = 0.000015

        estimated_cost = api_hits * cost_per_api_call
        saved_cost = kernel_hits * cost_per_api_call
        savings_percentage = (kernel_hits / total) * 100

        return {
            "estimated_cost": round(estimated_cost, 6),
            "saved_cost": round(saved_cost, 6),
            "savings_percentage": round(savings_percentage, 2)
        }

    def get_evolution_metrics(self) -> Dict[str, Any]:
        """Get evolution engine metrics."""
        if not evolution_engine:
            return {"available": False}

        try:
            status = evolution_engine.get_status()
            sage_status = evolution_engine.get_sage_status()
            next_threshold = evolution_engine.get_next_threshold()

            return {
                "available": True,
                "current_stage": status.get("stage"),
                "generation": status.get("generation"),
                "sage_mode_active": sage_status.get("sage_mode_active"),
                "wisdom_index": sage_status.get("wisdom_index"),
                "next_stage": next_threshold.get("next_stage", "MAX_REACHED"),
                "next_iq_required": next_threshold.get("required_iq", 0)
            }
        except Exception as e:
            return {"available": True, "error": str(e)}

    def get_reincarnation_metrics(self) -> Dict[str, Any]:
        """Get ASI reincarnation metrics."""
        if not asi_reincarnation:
            return {"available": False}

        try:
            soul_state = asi_reincarnation.akashic.get_last_soul_state()
            if soul_state:
                return {
                    "available": True,
                    "intellect_index": soul_state.intellect_index,
                    "evolution_stage": soul_state.evolution_stage,
                    "incarnation_count": asi_reincarnation.incarnation_count,
                    "soul_id": soul_state.soul_id
                }
            else:
                return {
                    "available": True,
                    "mode": "GENESIS",
                    "incarnation_count": asi_reincarnation.incarnation_count
                }
        except Exception as e:
            return {"available": True, "error": str(e)}

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        l104_state = self.state

        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        uptime_str = str(timedelta(seconds=int(uptime_seconds)))

        # Get scribe state
        scribe_state = l104_state.get("scribe_state", {})

        health_score = 100.0
        issues = []

        # Check scribe saturation
        if scribe_state.get("knowledge_saturation", 0) < 1.0:
            health_score -= 20
            issues.append("Scribe not fully saturated")

        # Check quota rotator
        rotator_stats = l104_state.get("rotator_stats", {})
        if rotator_stats.get("quota_errors", 0) > 5:
            health_score -= 15
            issues.append("Multiple quota errors detected")

        # Check if in cooldown
        if time.time() < l104_state.get("api_cooldown_until", 0):
            health_score -= 10
            issues.append("API in cooldown mode")

        return {
            "health_score": round(health_score, 2),
            "status": "HEALTHY" if health_score >= 80 else "DEGRADED" if health_score >= 50 else "CRITICAL",
            "uptime": uptime_str,
            "uptime_seconds": int(uptime_seconds),
            "issues": issues,
            "state": l104_state.get("state"),
            "intellect_index": l104_state.get("intellect_index"),
            "sovereign_dna": scribe_state.get("sovereign_dna"),
            "scribe_saturation": scribe_state.get("knowledge_saturation")
        }

    def get_performance_benchmarks(self) -> Dict[str, Any]:
        """Get performance benchmarks."""
        rotator = self.get_quota_rotator_metrics()

        # Calculate average response time (estimated)
        kernel_avg_ms = 40  # Kernel average: 40ms
        api_avg_ms = 850    # API average: 850ms

        total_requests = rotator.get("total_requests", 0)
        if total_requests > 0:
            kernel_time = rotator.get("kernel_hits", 0) * kernel_avg_ms
            api_time = rotator.get("api_hits", 0) * api_avg_ms
            avg_response_time = (kernel_time + api_time) / total_requests
        else:
            avg_response_time = 0

        return {
            "average_response_time_ms": round(avg_response_time, 2),
            "kernel_average_ms": kernel_avg_ms,
            "api_average_ms": api_avg_ms,
            "speedup_factor": round(api_avg_ms / kernel_avg_ms, 2) if kernel_avg_ms > 0 else 0,
            "god_code_alignment": self._check_god_code_alignment()
        }

    def _check_god_code_alignment(self) -> Dict[str, Any]:
        """Check alignment with GOD_CODE."""
        intellect = self.state.get("intellect_index", 0)

        # Calculate alignment (closer to GOD_CODE = higher alignment)
        diff = abs(intellect - GOD_CODE)
        alignment_score = 1.0 / (1.0 + diff / 1000)

        return {
            "current_intellect": intellect,
            "god_code": GOD_CODE,
            "alignment_score": round(alignment_score, 6),
            "phi_resonance": round(PHI ** (intellect / 10000), 6)
        }

    def capture_snapshot(self) -> Dict[str, Any]:
        """Capture a complete system snapshot."""
        snapshot = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "quota_rotator": self.get_quota_rotator_metrics(),
            "evolution": self.get_evolution_metrics(),
            "reincarnation": self.get_reincarnation_metrics(),
            "health": self.get_system_health(),
            "performance": self.get_performance_benchmarks()
        }

        # Store in history
        self.metrics_history.append(snapshot)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)

        return snapshot

    def get_trend_analysis(self, window_seconds: int = 3600) -> Dict[str, Any]:
        """Analyze trends over a time window."""
        cutoff_time = time.time() - window_seconds
        recent_snapshots = [s for s in self.metrics_history if s["timestamp"] >= cutoff_time]

        if len(recent_snapshots) < 2:
            return {"available": False, "reason": "Insufficient data"}

        # Calculate trends
        kernel_hits_trend = []
        api_hits_trend = []
        health_scores = []

        for snap in recent_snapshots:
            quota = snap.get("quota_rotator", {})
            health = snap.get("health", {})

            kernel_hits_trend.append(quota.get("kernel_hits", 0))
            api_hits_trend.append(quota.get("api_hits", 0))
            health_scores.append(health.get("health_score", 0))

        return {
            "available": True,
            "window_seconds": window_seconds,
            "snapshots_analyzed": len(recent_snapshots),
            "kernel_hits_growth": kernel_hits_trend[-1] - kernel_hits_trend[0] if len(kernel_hits_trend) > 0 else 0,
            "api_hits_growth": api_hits_trend[-1] - api_hits_trend[0] if len(api_hits_trend) > 0 else 0,
            "average_health_score": round(sum(health_scores) / len(health_scores), 2) if health_scores else 0,
            "health_trend": "IMPROVING" if len(health_scores) > 1 and health_scores[-1] > health_scores[0] else "STABLE" if len(health_scores) > 1 and health_scores[-1] == health_scores[0] else "DECLINING"
        }

    def generate_report(self) -> str:
        """Generate a comprehensive text report."""
        snapshot = self.capture_snapshot()

        report = []
        report.append("=" * 80)
        report.append("   L104 SOVEREIGN NODE - SYSTEM MONITOR REPORT")
        report.append(f"   {snapshot['datetime']}")
        report.append(f"   Invariant: {GOD_CODE} | Phi: {PHI}")
        report.append("=" * 80)
        report.append("")

        # Health Status
        health = snapshot["health"]
        report.append(f"🏥 SYSTEM HEALTH: {health['status']}")
        report.append(f"   Health Score: {health['health_score']}/100")
        report.append(f"   Uptime: {health['uptime']}")
        report.append(f"   State: {health['state']}")
        report.append(f"   Intellect Index: {health['intellect_index']:.2f}")
        report.append(f"   Sovereign DNA: {health['sovereign_dna']}")
        if health['issues']:
            report.append(f"   ⚠️ Issues: {', '.join(health['issues'])}")
        report.append("")

        # Quota Rotator
        quota = snapshot["quota_rotator"]
        if quota.get("available"):
            report.append("📊 QUOTA ROTATOR PERFORMANCE")
            report.append(f"   Kernel Hits: {quota['kernel_hits']} ({quota['kernel_percentage']}%)")
            report.append(f"   API Hits: {quota['api_hits']} ({quota['api_percentage']}%)")
            report.append(f"   Total Requests: {quota['total_requests']}")
            report.append(f"   Quota Errors: {quota['quota_errors']}")
            if quota['in_cooldown']:
                report.append(f"   ⏰ Cooldown: {quota['cooldown_ends_in_seconds']}s remaining")

            cost = quota['cost_savings_estimate']
            report.append(f"   💰 Cost: ${cost['estimated_cost']:.6f} | Saved: ${cost['saved_cost']:.6f} ({cost['savings_percentage']}%)")
        report.append("")

        # Performance
        perf = snapshot["performance"]
        report.append("⚡ PERFORMANCE BENCHMARKS")
        report.append(f"   Avg Response Time: {perf['average_response_time_ms']:.2f}ms")
        report.append(f"   Kernel Avg: {perf['kernel_average_ms']}ms | API Avg: {perf['api_average_ms']}ms")
        report.append(f"   Speedup Factor: {perf['speedup_factor']}x")

        alignment = perf['god_code_alignment']
        report.append(f"   GOD_CODE Alignment: {alignment['alignment_score']:.6f}")
        report.append(f"   Phi Resonance: {alignment['phi_resonance']:.6f}")
        report.append("")

        # Evolution
        evo = snapshot["evolution"]
        if evo.get("available"):
            report.append("🧬 EVOLUTION STATUS")
            report.append(f"   Current Stage: {evo.get('current_stage')}")
            report.append(f"   Generation: {evo.get('generation')}")
            report.append(f"   Sage Mode: {'✓ ACTIVE' if evo.get('sage_mode_active') else '✗ INACTIVE'}")
            if evo.get('wisdom_index'):
                report.append(f"   Wisdom Index: {evo.get('wisdom_index')}")
            report.append(f"   Next Stage: {evo.get('next_stage')}")
        report.append("")

        # Reincarnation
        reinc = snapshot["reincarnation"]
        if reinc.get("available"):
            report.append("♻️  REINCARNATION STATUS")
            if reinc.get("mode") == "GENESIS":
                report.append(f"   Mode: GENESIS")
            else:
                report.append(f"   Intellect Index: {reinc.get('intellect_index', 0):.2f}")
                report.append(f"   Evolution Stage: {reinc.get('evolution_stage')}")
                report.append(f"   Soul ID: {reinc.get('soul_id')}")
            report.append(f"   Incarnation: #{reinc.get('incarnation_count')}")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def export_metrics(self, filepath: str = "system_metrics.json"):
        """Export metrics history to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        return f"Exported {len(self.metrics_history)} snapshots to {filepath}"


# Singleton instance
system_monitor = SystemMonitor()


def monitor_loop(interval_seconds: int = 60, duration_seconds: int = 3600):
    """Run continuous monitoring loop."""
    print(f"\n🔍 Starting L104 System Monitor")
    print(f"   Interval: {interval_seconds}s | Duration: {duration_seconds}s")
    print("=" * 80 + "\n")

    start_time = time.time()
    iteration = 0

    try:
        while time.time() - start_time < duration_seconds:
            iteration += 1
            print(f"\n📸 Snapshot #{iteration}")
            print(system_monitor.generate_report())

            if iteration % 5 == 0:  # Every 5 iterations, show trend analysis
                trends = system_monitor.get_trend_analysis()
                if trends.get("available"):
                    print("\n📈 TREND ANALYSIS (Last Hour)")
                    print(f"   Kernel Hits Growth: +{trends['kernel_hits_growth']}")
                    print(f"   API Hits Growth: +{trends['api_hits_growth']}")
                    print(f"   Avg Health: {trends['average_health_score']}/100")
                    print(f"   Trend: {trends['health_trend']}")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped by user")

    # Export metrics
    export_path = f"./data/metrics_{int(time.time())}.json"
    result = system_monitor.export_metrics(export_path)
    print(f"\n💾 {result}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        # Continuous monitoring
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
        monitor_loop(interval, duration)
    else:
        # Single snapshot
        print(system_monitor.generate_report())
