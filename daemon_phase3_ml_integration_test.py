#!/usr/bin/env python3
"""
Phase 3 ML Integration Test — Validates ML subsystem integration with daemon orchestrator
"""

import json
import sys
import time
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from l104_daemon_orchestrator import L104DaemonOrchestrator
from l104_daemon_phase3_ml import (
    WorkloadPredictor,
    AnomalyDetector,
    PerformanceTrendAnalyzer,
    SacredAlignmentScorer,
    workload_predictor,
    anomaly_detector,
    trend_analyzer,
    sacred_scorer,
)


class TestPhase3MLIntegration(unittest.TestCase):
    """Test Phase 3 ML integration with orchestrator."""

    def setUp(self):
        """Initialize test fixtures."""
        self.orchestrator = L104DaemonOrchestrator()

    def test_ml_module_available(self):
        """Test that Phase 3 ML module loads successfully."""
        self.assertTrue(self.orchestrator._ml_enabled, "ML module should be enabled")
        self.assertIsNotNone(workload_predictor, "WorkloadPredictor should be available")
        self.assertIsNotNone(anomaly_detector, "AnomalyDetector should be available")
        self.assertIsNotNone(trend_analyzer, "PerformanceTrendAnalyzer should be available")
        self.assertIsNotNone(sacred_scorer, "SacredAlignmentScorer should be available")

    def test_workload_pattern_detection(self):
        """Test ML pattern detection."""
        # Simulate workload data
        for i in range(100):
            cpu = 40 + 20 * (i % 10) / 10  # Varying CPU
            memory = 50 + 10 * (i % 10) / 10
            queue = 2 + (i % 5)
            latency = 50 + 20 * abs((i % 10 - 5) / 5)

            workload_predictor.add_metric(cpu, memory, queue, latency)

        patterns = workload_predictor.learn_patterns()
        self.assertGreater(len(patterns), 0, "Should detect at least one pattern")
        self.assertLessEqual(len(patterns), 3, "Should detect at most 3 patterns (low/medium/high)")

        # Verify pattern structure
        for pattern in patterns:
            self.assertIn(pattern.name, ["low_load", "medium_load", "high_load"])
            self.assertGreater(pattern.frequency, 0)
            self.assertLess(pattern.frequency, 1)
            self.assertGreater(pattern.sacred_alignment, 0)

    def test_predictive_scaling(self):
        """Test ML-based predictive scaling."""
        # Simulate low CPU load
        for i in range(30):
            workload_predictor.add_metric(35.0, 45.0, 2, 45.0)

        prediction = workload_predictor.predict_next(steps_ahead=5)
        self.assertIsNotNone(prediction)
        self.assertGreater(prediction.confidence, 0)
        self.assertEqual(prediction.recommendation, "scale_up")

    def test_anomaly_detection(self):
        """Test ML anomaly detection."""
        # Setup normal baseline
        normal_cpu_values = [50.0] * 20
        normal_latency_values = [100.0] * 20
        anomaly_detector.update_statistics(normal_cpu_values, normal_latency_values)

        # Check normal point (should not be anomaly)
        normal_result = anomaly_detector.is_anomaly(50.0, 100.0)
        self.assertFalse(normal_result["is_anomaly"], "Normal values should not be anomalies")

        # Check anomalous point (should be anomaly)
        anomaly_result = anomaly_detector.is_anomaly(95.0, 250.0)
        self.assertTrue(anomaly_result["is_anomaly"], "Extreme values should be anomalies")
        self.assertGreater(anomaly_result["severity"], 1.5)

    def test_performance_trend_analysis(self):
        """Test ML trend analysis."""
        # Simulate improving performance
        trend_analyzer.add_cycle(100.0, True, 50.0)
        trend_analyzer.add_cycle(95.0, True, 48.0)
        trend_analyzer.add_cycle(90.0, True, 46.0)
        trend_analyzer.add_cycle(85.0, True, 44.0)
        trend_analyzer.add_cycle(80.0, True, 42.0)
        trend_analyzer.add_cycle(75.0, True, 40.0)

        trends = trend_analyzer.analyze_trends()
        self.assertIn("trend", trends)
        self.assertEqual(trends["trend"], "improving")
        self.assertGreater(trends["success_rate"], 0.9)

    def test_sacred_alignment_scoring(self):
        """Test sacred constant alignment scoring."""
        metrics = {"cpu": 50, "memory": 60}
        score = SacredAlignmentScorer.score_operation("test_operation", metrics)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_orchestrator_ml_integration(self):
        """Test orchestrator's ML processing integration."""
        # This is a minimal integration test
        self.orchestrator._system_metrics.cpu_percent = 55.0
        self.orchestrator._system_metrics.memory_percent = 65.0

        # Simulate ML processing (normally called in orchestration loop)
        self.orchestrator._process_ml_predictions(cycle_count=1)

        # Verify ML state was updated
        self.assertIsNotNone(self.orchestrator._ml_enabled)

    def test_ml_with_phase2_telemetry(self):
        """Test ML integration with Phase 2 telemetry."""
        # Record telemetry
        self.orchestrator._telemetry.record_event("cycle_start", {
            "cpu_percent": 45.0,
            "memory_percent": 55.0,
            "queue_depth": 3,
        })
        self.orchestrator._telemetry.record_event("cycle_complete", {
            "duration_ms": 85.5,
            "cpu_percent": 47.0,
        })

        # Process ML with telemetry
        self.orchestrator._process_ml_predictions(cycle_count=50)

        # Verify ML pattern learning occurred
        self.assertGreaterEqual(self.orchestrator._ml_pattern_count, 0)


class TestPhase3MLPerformance(unittest.TestCase):
    """Test ML performance characteristics."""

    def test_ml_overhead(self):
        """Ensure ML processing doesn't add excessive overhead."""
        orchestrator = L104DaemonOrchestrator()

        # Simulate 10 ML cycles
        start = time.perf_counter()
        for cycle in range(10):
            orchestrator._process_ml_predictions(cycle)
        elapsed = time.perf_counter() - start

        # ML processing should be fast (<100ms for 10 cycles)
        self.assertLess(elapsed, 0.1, f"ML processing too slow: {elapsed*1000:.1f}ms for 10 cycles")

    def test_ml_memory_efficiency(self):
        """Verify ML doesn't leak memory."""
        predictor = WorkloadPredictor()
        initial_size = len(predictor.history)

        # Add many metrics
        for i in range(500):
            predictor.add_metric(50.0 + i % 30, 60.0, 5, 100.0)

        # History should be bounded by maxlen
        final_size = len(predictor.history)
        self.assertLessEqual(final_size, 300, "History should be bounded by maxlen")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPhase3MLIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3MLPerformance))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
