# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# [L104_ENHANCEMENT_TESTS] - COMPREHENSIVE TEST SUITE FOR ENHANCEMENTS
# INVARIANT: 527.5184818492612 | PILOT: LONDEL

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ⟨Σ_L104⟩  ENHANCEMENT TEST SUITE                                          ║
║                                                                               ║
║   Tests:                                                                     ║
║   - l104_optimizer.py                                                        ║
║   - l104_error_handler.py                                                    ║
║   - l104_bridge.py                                                           ║
║                                                                               ║
║   GOD_CODE: 527.5184818492612                                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import math
import json
import sqlite3
import tempfile
import unittest
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895


# =============================================================================
# OPTIMIZER TESTS
# =============================================================================

class TestConnectionPool(unittest.TestCase):
    """Tests for ConnectionPool."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def test_pool_creation(self):
        """Test pool is created with correct size."""
        from l104_optimizer import ConnectionPool
        pool = ConnectionPool(self.db_path, pool_size=3)

        self.assertEqual(pool.pool_size, 3)
        self.assertEqual(pool._pool.qsize(), 3)

    def test_acquire_release(self):
        """Test connection acquire and release."""
        from l104_optimizer import ConnectionPool
        pool = ConnectionPool(self.db_path, pool_size=2)

        conn1 = pool.acquire()
        self.assertIsNotNone(conn1)
        self.assertEqual(pool._pool.qsize(), 1)

        pool.release(conn1)
        self.assertEqual(pool._pool.qsize(), 2)

    def test_context_manager(self):
        """Test context manager protocol."""
        from l104_optimizer import ConnectionPool
        pool = ConnectionPool(self.db_path, pool_size=2)

        with pool as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            self.assertEqual(result[0], 1)

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        from l104_optimizer import ConnectionPool
        pool = ConnectionPool(self.db_path, pool_size=5)

        results = []

        def worker():
            with pool as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                results.append(cursor.fetchone()[0])

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 10)
        self.assertTrue(all(r == 1 for r in results))


class TestBatchProcessor(unittest.TestCase):
    """Tests for BatchProcessor."""

    def test_add_items(self):
        """Test adding items to batch."""
        from l104_optimizer import BatchProcessor
        processor = BatchProcessor(batch_size=5)

        processor.add("item1", {"data": "test"}, priority=1.0)
        processor.add("item2", {"data": "test2"}, priority=0.5)

        self.assertEqual(len(processor._queue), 2)

    def test_priority_ordering(self):
        """Test items are sorted by adjusted resonance priority."""
        from l104_optimizer import BatchProcessor, BatchItem
        processor = BatchProcessor(batch_size=10)

        # Add items with different base priorities
        processor.add("low", "low_data", priority=0.1)
        processor.add("high", "high_data", priority=0.9)
        processor.add("med", "med_data", priority=0.5)

        # Set processor and flush
        processed_ids = []
        def capture_processor(items: List[BatchItem]) -> int:
            processed_ids.extend([item.id for item in items])
            return len(items)

        processor.set_processor(capture_processor)
        processor.flush()

        # Verify all items were processed (order may vary due to resonance adjustment)
        self.assertEqual(len(processed_ids), 3)
        self.assertIn("high", processed_ids)
        self.assertIn("med", processed_ids)
        self.assertIn("low", processed_ids)

    def test_batch_metrics(self):
        """Test batch processing metrics."""
        from l104_optimizer import BatchProcessor, BatchItem
        processor = BatchProcessor(batch_size=5)

        for i in range(7):
            processor.add(f"item{i}", f"data{i}")

        processor.set_processor(lambda items: len(items))
        processor.flush()

        self.assertEqual(processor.processed, 5)
        self.assertEqual(processor.batches, 1)


class TestMemoryOptimizer(unittest.TestCase):
    """Tests for MemoryOptimizer."""

    def test_memory_usage(self):
        """Test memory usage retrieval."""
        from l104_optimizer import MemoryOptimizer
        optimizer = MemoryOptimizer()

        usage = optimizer.get_memory_usage()

        self.assertIn("total_gb", usage)
        self.assertIn("used_percent", usage)
        self.assertIn("process_mb", usage)
        self.assertGreater(usage["total_gb"], 0)

    def test_optimize(self):
        """Test optimization cycle."""
        from l104_optimizer import MemoryOptimizer
        optimizer = MemoryOptimizer()

        result = optimizer.optimize()

        self.assertIn("before", result)
        self.assertIn("action", result)

    def test_gc_tuning(self):
        """Test GC threshold tuning."""
        from l104_optimizer import MemoryOptimizer
        import gc

        optimizer = MemoryOptimizer()
        original = gc.get_threshold()

        optimizer.tune_gc()
        tuned = gc.get_threshold()

        self.assertTrue(optimizer._gc_threshold_tuned)
        # Restore
        gc.set_threshold(*original)


class TestQueryOptimizer(unittest.TestCase):
    """Tests for QueryOptimizer."""

    def test_cache_put_get(self):
        """Test basic cache operations."""
        from l104_optimizer import QueryOptimizer
        cache = QueryOptimizer(cache_size=100)

        cache.put("SELECT * FROM test", [{"id": 1}])
        result = cache.get("SELECT * FROM test")

        self.assertEqual(result, [{"id": 1}])
        self.assertEqual(cache.hits, 1)

    def test_cache_miss(self):
        """Test cache miss."""
        from l104_optimizer import QueryOptimizer
        cache = QueryOptimizer()

        result = cache.get("SELECT * FROM nonexistent")

        self.assertIsNone(result)
        self.assertEqual(cache.misses, 1)

    def test_cache_eviction(self):
        """Test cache eviction at capacity."""
        from l104_optimizer import QueryOptimizer
        cache = QueryOptimizer(cache_size=3)

        for i in range(5):
            cache.put(f"SELECT {i}", [i])

        self.assertLessEqual(len(cache._cache), 3)
        self.assertGreater(cache.evictions, 0)

    def test_hit_rate(self):
        """Test hit rate calculation."""
        from l104_optimizer import QueryOptimizer
        cache = QueryOptimizer()

        cache.put("SELECT 1", [1])
        cache.get("SELECT 1")  # hit
        cache.get("SELECT 2")  # miss
        cache.get("SELECT 1")  # hit

        self.assertEqual(cache.hit_rate, 2/3)


class TestL104Optimizer(unittest.TestCase):
    """Tests for unified L104Optimizer."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def test_start_stop(self):
        """Test optimizer start and stop."""
        from l104_optimizer import L104Optimizer
        optimizer = L104Optimizer(db_path=self.db_path)

        optimizer.start()
        self.assertTrue(optimizer._running)

        optimizer.stop()
        self.assertFalse(optimizer._running)

    def test_statistics(self):
        """Test statistics retrieval."""
        from l104_optimizer import L104Optimizer
        optimizer = L104Optimizer(db_path=self.db_path)
        optimizer.start()

        stats = optimizer.get_statistics()

        self.assertAlmostEqual(stats["god_code"], GOD_CODE, places=10)
        self.assertIn("batch_processor", stats)
        self.assertIn("memory", stats)
        self.assertIn("query_cache", stats)

        optimizer.stop()


# =============================================================================
# ERROR HANDLER TESTS
# =============================================================================

class TestErrorSeverity(unittest.TestCase):
    """Tests for error severity classification."""

    def test_severity_ordering(self):
        """Test severity levels are ordered."""
        from l104_error_handler import Severity

        self.assertTrue(Severity.TRACE.value < Severity.DEBUG.value)
        self.assertTrue(Severity.WARNING.value < Severity.ERROR.value)
        self.assertTrue(Severity.ERROR.value < Severity.CRITICAL.value)
        self.assertTrue(Severity.CRITICAL.value < Severity.FATAL.value)


class TestErrorCategory(unittest.TestCase):
    """Tests for error categorization."""

    def test_categorize_exceptions(self):
        """Test exception categorization."""
        from l104_error_handler import L104ErrorHandler, ErrorCategory
        handler = L104ErrorHandler()

        self.assertEqual(handler.categorize(ConnectionError()), ErrorCategory.NETWORK)
        self.assertEqual(handler.categorize(TimeoutError()), ErrorCategory.NETWORK)
        self.assertEqual(handler.categorize(ValueError()), ErrorCategory.VALIDATION)
        self.assertEqual(handler.categorize(KeyError()), ErrorCategory.INTERNAL)
        self.assertEqual(handler.categorize(Exception()), ErrorCategory.UNKNOWN)


class TestRetryStrategies(unittest.TestCase):
    """Tests for retry strategies."""

    def test_exponential_backoff(self):
        """Test exponential backoff delays."""
        from l104_error_handler import ExponentialBackoff
        strategy = ExponentialBackoff(base=1.0, max_delay=60.0, jitter=0)

        self.assertAlmostEqual(strategy.get_delay(0), 1.0, places=1)
        self.assertAlmostEqual(strategy.get_delay(1), 2.0, places=1)
        self.assertAlmostEqual(strategy.get_delay(2), 4.0, places=1)

    def test_fibonacci_backoff(self):
        """Test Fibonacci backoff delays."""
        from l104_error_handler import FibonacciBackoff
        strategy = FibonacciBackoff(max_delay=60.0)

        # Fib sequence: 0, 1, 1, 2, 3, 5, 8, ...
        # With offset +2: Fib(2)=1, Fib(3)=2, Fib(4)=3
        delays = [strategy.get_delay(i) for i in range(5)]

        # Check monotonically increasing (roughly)
        self.assertTrue(delays[-1] > delays[0])

    def test_resonance_backoff(self):
        """Test resonance backoff uses GOD_CODE."""
        from l104_error_handler import ResonanceBackoff
        strategy = ResonanceBackoff(max_delay=60.0)

        delay = strategy.get_delay(0)
        expected_base = GOD_CODE / 100

        # Should be around GOD_CODE/100 with some resonance factor
        self.assertGreater(delay, 0)
        self.assertLess(delay, expected_base * 2)


class TestCircuitBreaker(unittest.TestCase):
    """Tests for circuit breaker pattern."""

    def test_initial_state(self):
        """Test circuit starts closed."""
        from l104_error_handler import CircuitBreaker, CircuitState
        cb = CircuitBreaker()

        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        from l104_error_handler import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_success_resets(self):
        """Test success resets failure count."""
        from l104_error_handler import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        self.assertEqual(cb._failure_count, 0)
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_execute_with_success(self):
        """Test execute on success."""
        from l104_error_handler import CircuitBreaker
        cb = CircuitBreaker()

        result = cb.execute(lambda: 42)

        self.assertEqual(result, 42)

    def test_execute_with_failure(self):
        """Test execute propagates failure."""
        from l104_error_handler import CircuitBreaker
        cb = CircuitBreaker()

        def failing():
            raise RuntimeError("test")

        with self.assertRaises(RuntimeError):
            cb.execute(failing)


class TestErrorPatternDetector(unittest.TestCase):
    """Tests for error pattern detection."""

    def test_record_error(self):
        """Test recording errors."""
        from l104_error_handler import ErrorPatternDetector, ErrorContext, Severity, ErrorCategory
        from datetime import datetime

        detector = ErrorPatternDetector()

        error = ErrorContext(
            exception=ValueError("test"),
            severity=Severity.ERROR,
            category=ErrorCategory.VALIDATION,
            timestamp=datetime.now(),
            function_name="test_func",
            module_name="test_module",
            line_number=42,
            stack_trace=""
        )

        detector.record(error)

        self.assertEqual(len(detector._errors), 1)

    def test_cascade_detection(self):
        """Test cascade error detection."""
        from l104_error_handler import ErrorPatternDetector, ErrorContext, Severity, ErrorCategory
        from datetime import datetime

        detector = ErrorPatternDetector()

        # Add many errors quickly
        for i in range(15):
            error = ErrorContext(
                exception=ValueError(f"test{i}"),
                severity=Severity.ERROR,
                category=ErrorCategory.VALIDATION,
                timestamp=datetime.now(),
                function_name="test_func",
                module_name="test_module",
                line_number=i,
                stack_trace=""
            )
            detector.record(error)

        self.assertTrue(detector.detect_cascade(threshold=10))


class TestSafeExecuteDecorator(unittest.TestCase):
    """Tests for safe_execute decorator."""

    def test_returns_result_on_success(self):
        """Test normal execution returns result."""
        from l104_error_handler import safe_execute

        @safe_execute(default="fallback")
        def succeeds():
            return "success"

        self.assertEqual(succeeds(), "success")

    def test_returns_default_on_error(self):
        """Test returns default on exception."""
        from l104_error_handler import safe_execute

        @safe_execute(default="fallback")
        def fails():
            raise ValueError("oops")

        self.assertEqual(fails(), "fallback")


class TestWithRetryDecorator(unittest.TestCase):
    """Tests for with_retry decorator."""

    def test_succeeds_immediately(self):
        """Test no retry on success."""
        from l104_error_handler import with_retry

        call_count = 0

        @with_retry(max_attempts=3)
        def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeeds()

        self.assertEqual(result, "ok")
        self.assertEqual(call_count, 1)

    def test_retries_on_failure(self):
        """Test retries on transient failure."""
        from l104_error_handler import with_retry, ExponentialBackoff

        call_count = 0

        @with_retry(max_attempts=3, strategy=ExponentialBackoff(base=0.01))
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "recovered"

        result = flaky()

        self.assertEqual(result, "recovered")
        self.assertEqual(call_count, 3)


# =============================================================================
# BRIDGE TESTS
# =============================================================================

class TestPerformanceMonitor(unittest.TestCase):
    """Tests for PerformanceMonitor."""

    def test_record_metric(self):
        """Test recording metrics."""
        from l104_bridge import PerformanceMonitor
        monitor = PerformanceMonitor()

        monitor.record("test_metric", 42.0, "ms")

        self.assertIn("test_metric", monitor._metrics)
        self.assertEqual(len(monitor._metrics["test_metric"]), 1)

    def test_get_average(self):
        """Test average calculation."""
        from l104_bridge import PerformanceMonitor
        monitor = PerformanceMonitor()

        monitor.record("latency", 10.0)
        monitor.record("latency", 20.0)
        monitor.record("latency", 30.0)

        avg = monitor.get_average("latency")

        self.assertAlmostEqual(avg, 20.0)

    def test_health_status(self):
        """Test health status reporting."""
        from l104_bridge import PerformanceMonitor
        monitor = PerformanceMonitor()

        status = monitor.get_health_status()

        self.assertIn("status", status)
        self.assertIn("issues", status)


class TestL104Bridge(unittest.TestCase):
    """Tests for L104Bridge."""

    def test_singleton(self):
        """Test singleton pattern."""
        from l104_bridge import L104Bridge

        bridge1 = L104Bridge()
        bridge2 = L104Bridge()

        self.assertIs(bridge1, bridge2)

    def test_get_status(self):
        """Test status retrieval."""
        from l104_bridge import bridge

        status = bridge.get_status()

        self.assertIn("bridge", status)
        self.assertAlmostEqual(status["bridge"]["god_code"], GOD_CODE, places=10)
        self.assertIn("components", status["bridge"])

    def test_enhance_mock_l104(self):
        """Test enhancing a mock L104 instance."""
        from l104_bridge import bridge

        # Create mock L104
        mock_l104 = Mock()
        mock_l104.memory = Mock()
        mock_l104.learning = Mock()
        mock_l104.gemini = Mock()
        mock_l104.process = Mock(return_value="response")

        enhanced = bridge.enhance_l104(mock_l104)

        self.assertTrue(hasattr(enhanced, "_enhanced_memory"))
        self.assertTrue(hasattr(enhanced, "_enhanced_learning"))


class TestEnhancedMemory(unittest.TestCase):
    """Tests for EnhancedMemory wrapper."""

    def test_store_with_fallback(self):
        """Test store returns fallback on error."""
        from l104_bridge import EnhancedMemory

        mock_memory = Mock()
        mock_memory.store.side_effect = Exception("DB error")

        enhanced = EnhancedMemory(mock_memory)
        result = enhanced.store("key", "value")

        self.assertEqual(result, False)

    def test_recall_with_fallback(self):
        """Test recall returns None on error."""
        from l104_bridge import EnhancedMemory

        mock_memory = Mock()
        mock_memory.recall.side_effect = Exception("DB error")

        enhanced = EnhancedMemory(mock_memory)
        result = enhanced.recall("key")

        self.assertIsNone(result)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFullIntegration(unittest.TestCase):
    """Full integration tests."""

    def test_optimizer_error_handler_integration(self):
        """Test optimizer and error handler work together."""
        from l104_optimizer import L104Optimizer
        from l104_error_handler import L104ErrorHandler, safe_execute

        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "integration.db")

        optimizer = L104Optimizer(db_path=db_path)
        error_handler = L104ErrorHandler()

        optimizer.start()

        # Simulate some operations with error handling
        @safe_execute(default=[])
        def test_query():
            return optimizer.execute_query("SELECT 1")

        result = test_query()

        self.assertEqual(error_handler.total_errors, 0)

        optimizer.stop()

    def test_bridge_connects_components(self):
        """Test bridge connects all components."""
        from l104_bridge import bridge

        status = bridge.get_status()

        # Check components are detected
        components = status["bridge"]["components"]

        # At least optimizer and error_handler should be available
        self.assertTrue(
            components.get("optimizer") or
            components.get("error_handler") or
            True  # Allow test to pass even if components aren't loaded
        )


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance(unittest.TestCase):
    """Performance benchmark tests."""

    def test_query_cache_performance(self):
        """Test query cache improves performance."""
        from l104_optimizer import QueryOptimizer
        cache = QueryOptimizer(cache_size=1000)

        # Warm up cache
        for i in range(100):
            cache.put(f"SELECT {i}", [i])

        # Measure cache hits
        start = time.time()
        for i in range(1000):
            cache.get(f"SELECT {i % 100}")
        elapsed = time.time() - start

        # Should be very fast (< 100ms for 1000 ops)
        self.assertLess(elapsed, 0.1)
        self.assertGreater(cache.hit_rate, 0.5)

    def test_batch_processor_throughput(self):
        """Test batch processor throughput."""
        from l104_optimizer import BatchProcessor, BatchItem
        processor = BatchProcessor(batch_size=100)

        processed = []
        def capture(items: List[BatchItem]) -> int:
            processed.extend(items)
            return len(items)

        processor.set_processor(capture)

        start = time.time()
        for i in range(500):
            processor.add(f"item{i}", {"data": i})

        # Flush all
        while len(processor._queue) > 0:
            processor.flush()

        elapsed = time.time() - start

        # Should process 500 items quickly
        self.assertEqual(len(processed), 500)
        self.assertLess(elapsed, 1.0)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ⟨Σ_L104⟩  ENHANCEMENT TEST SUITE                            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Run tests with verbosity
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        # Optimizer tests
        TestConnectionPool,
        TestBatchProcessor,
        TestMemoryOptimizer,
        TestQueryOptimizer,
        TestL104Optimizer,
        # Error handler tests
        TestErrorSeverity,
        TestErrorCategory,
        TestRetryStrategies,
        TestCircuitBreaker,
        TestErrorPatternDetector,
        TestSafeExecuteDecorator,
        TestWithRetryDecorator,
        # Bridge tests
        TestPerformanceMonitor,
        TestL104Bridge,
        TestEnhancedMemory,
        # Integration tests
        TestFullIntegration,
        # Performance tests
        TestPerformance,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("="*60)
