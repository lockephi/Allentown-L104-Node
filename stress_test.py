#!/usr/bin/env python3
"""
L104 Sovereign Node - Comprehensive Stress Test Suite
======================================================
Reusable stress test suite for CI/CD integration.

Usage:
    python stress_test.py              # Run all phases
    python stress_test.py --phase 1    # Run specific phase
    python stress_test.py --quick      # Quick validation mode
    python stress_test.py --json       # Output JSON report
"""

import time
import os
import sys
import json
import random
import hashlib
import asyncio
import sqlite3
import tempfile
import threading
import queue
import argparse
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Tuple

# NumPy for numerical operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# HTTP client for API tests
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class StressTestResult:
    """Container for individual test results."""

    def __init__(self, name: str, operations: int, duration: float, passed: bool, error: str = None):
        self.name = name
        self.operations = operations
        self.duration = duration
        self.passed = passed
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "operations": self.operations,
            "duration_seconds": round(self.duration, 4),
            "passed": self.passed,
            "ops_per_second": round(self.operations / self.duration, 2) if self.duration > 0 else 0,
            "error": self.error
        }


class PhaseResult:
    """Container for phase results."""

    def __init__(self, phase_num: int, name: str):
        self.phase_num = phase_num
        self.name = name
        self.tests: List[StressTestResult] = []
        self.start_time = 0
        self.end_time = 0

    @property
    def total_operations(self) -> int:
        return sum(t.operations for t in self.tests)

    @property
    def total_duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def all_passed(self) -> bool:
        return all(t.passed for t in self.tests)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase_num,
            "name": self.name,
            "total_operations": self.total_operations,
            "total_duration_seconds": round(self.total_duration, 4),
            "all_passed": self.all_passed,
            "tests": [t.to_dict() for t in self.tests]
        }


class L104StressTest:
    """L104 Sovereign Node Stress Test Suite."""

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.scale = 0.1 if quick_mode else 1.0
        self.results: List[PhaseResult] = []
        self.api_base = "http://localhost:8081"

    def _scale(self, n: int) -> int:
        """Scale operation count based on mode."""
        return max(1, int(n * self.scale))

    def run_test(self, name: str, func, *args, **kwargs) -> StressTestResult:
        """Execute a single test and capture results."""
        start = time.time()
        try:
            operations = func(*args, **kwargs)
            duration = time.time() - start
            return StressTestResult(name, operations, duration, True)
        except Exception as e:
            duration = time.time() - start
            return StressTestResult(name, 0, duration, False, str(e))

    # ========== PHASE 1: CORE SYSTEMS ==========

    def phase1_core_systems(self) -> PhaseResult:
        """Phase 1: Core system stress tests."""
        phase = PhaseResult(1, "Core Systems")
        phase.start_time = time.time()

        # Test 1: Module imports
        def test_imports():
            modules = [
                'l104_kernel', 'l104_hyper_math', 'l104_security',
                'l104_codec', 'l104_anchor', 'l104_engine',
                'l104_parallel_engine', 'l104_deep_substrate'
            ]
            imported = 0
            for mod in modules:
                try:
                    __import__(mod)
                    imported += 1
                except:
                    pass
            return imported

        phase.tests.append(self.run_test("Module Imports", test_imports))

        # Test 2: HyperMath operations
        def test_hypermath():
            from l104_hyper_math import HyperMath
            hm = HyperMath()
            ops = self._scale(30000)
            for i in range(ops):
                hm.get_lattice_scalar()
            return ops

        phase.tests.append(self.run_test("HyperMath Operations", test_hypermath))

        # Test 3: Cryptographic operations
        def test_crypto():
            from l104_security import SovereignCrypt
            ops = self._scale(3000)
            for i in range(ops):
                sig = SovereignCrypt.hash_with_phi(f"data_{i}")
                SovereignCrypt.validate_signature(f"data_{i}", sig)
            return ops

        phase.tests.append(self.run_test("Cryptographic Operations", test_crypto))

        # Test 4: Codec operations
        def test_codec():
            from l104_codec import SovereignCodec
            ops = self._scale(3000)
            for i in range(ops):
                encoded = SovereignCodec.encode_resonance(float(i))
                SovereignCodec.decode_resonance(encoded)
            return ops

        phase.tests.append(self.run_test("Codec Operations", test_codec))

        # Test 5: Kernel operations
        def test_kernel():
            from l104_kernel import kernel, calculate_resonance
            ops = self._scale(200)
            for i in range(ops):
                calculate_resonance(float(i), 1.381)
            return ops

        phase.tests.append(self.run_test("Kernel Operations", test_kernel))

        # Test 6: Memory allocation
        def test_memory():
            elements = self._scale(1000000)
            data = list(range(elements))
            total = sum(data)
            del data
            return elements

        phase.tests.append(self.run_test("Memory Allocation", test_memory))

        phase.end_time = time.time()
        return phase

    # ========== PHASE 2: DATABASE & ASYNC ==========

    def phase2_database_async(self) -> PhaseResult:
        """Phase 2: Database and async stress tests."""
        phase = PhaseResult(2, "Database & Async")
        phase.start_time = time.time()

        # Test 1: SQLite operations
        def test_sqlite():
            rows = self._scale(5000)
            with tempfile.NamedTemporaryFile(suffix='.db', delete=True) as f:
                conn = sqlite3.connect(f.name)
                c = conn.cursor()
                c.execute('CREATE TABLE test (id INTEGER, data TEXT)')
                for i in range(rows):
                    c.execute('INSERT INTO test VALUES (?, ?)', (i, f'data_{i}'))
                conn.commit()
                c.execute('SELECT COUNT(*) FROM test')
                conn.close()
            return rows

        phase.tests.append(self.run_test("SQLite Operations", test_sqlite))

        # Test 2: JSON serialization
        def test_json():
            ops = self._scale(20000)
            for i in range(ops):
                data = {'id': i, 'nested': {'values': list(range(10))}}
                encoded = json.dumps(data)
                json.loads(encoded)
            return ops

        phase.tests.append(self.run_test("JSON Serialization", test_json))

        # Test 3: Hash generation
        def test_hashes():
            ops = self._scale(100000)
            for i in range(ops):
                hashlib.sha256(f"data_{i}".encode()).hexdigest()
            return ops

        phase.tests.append(self.run_test("Hash Generation", test_hashes))

        # Test 4: Async tasks
        def test_async():
            async def async_work(n):
                await asyncio.sleep(0)
                return n * 2

            async def run_tasks():
                tasks = [async_work(i) for i in range(self._scale(1000))]
                return await asyncio.gather(*tasks)

            results = asyncio.run(run_tasks())
            return len(results)

        phase.tests.append(self.run_test("Async Tasks", test_async))

        # Test 5: String operations
        def test_strings():
            ops = self._scale(400000)
            for i in range(ops):
                s = f"test_string_{i}" * 10
                s.upper().lower().replace('_', '-')
            return ops

        phase.tests.append(self.run_test("String Operations", test_strings))

        phase.end_time = time.time()
        return phase

    # ========== PHASE 3: NEURAL & NUMPY ==========

    def phase3_neural_numpy(self) -> PhaseResult:
        """Phase 3: Neural network and NumPy stress tests."""
        phase = PhaseResult(3, "Neural & NumPy")
        phase.start_time = time.time()

        if not NUMPY_AVAILABLE:
            phase.tests.append(StressTestResult("NumPy", 0, 0, False, "NumPy not available"))
            phase.end_time = time.time()
            return phase

        # Test 1: Matrix multiplication
        def test_matrix():
            ops = self._scale(100)
            for _ in range(ops):
                a = np.random.rand(500, 500)
                b = np.random.rand(500, 500)
                _ = a @ b
            return ops

        phase.tests.append(self.run_test("Matrix Multiply (500x500)", test_matrix))

        # Test 2: FFT transforms
        def test_fft():
            ops = self._scale(2000)
            for _ in range(ops):
                data = np.random.rand(1024)
                np.fft.fft(data)
            return ops

        phase.tests.append(self.run_test("FFT Transforms", test_fft))

        # Test 3: Eigenvalue decomposition
        def test_eigen():
            ops = self._scale(50)
            for _ in range(ops):
                m = np.random.rand(100, 100)
                m = (m + m.T) / 2  # Make symmetric
                np.linalg.eigvalsh(m)
            return ops

        phase.tests.append(self.run_test("Eigenvalue Decomposition", test_eigen))

        # Test 4: Deep substrate (if available)
        def test_deep_substrate():
            try:
                from l104_deep_substrate import deep_substrate
                ops = self._scale(200)
                for i in range(ops):
                    # Deep substrate expects numpy array input
                    pattern = np.random.rand(1, 64)
                    deep_substrate.learn_pattern(pattern)
                return ops
            except Exception as e:
                raise RuntimeError(f"Deep substrate unavailable: {e}")

        phase.tests.append(self.run_test("Deep Substrate Learning", test_deep_substrate))

        # Test 5: Vector operations
        def test_vectors():
            ops = self._scale(4000)
            for _ in range(ops):
                v1 = np.random.rand(1000)
                v2 = np.random.rand(1000)
                np.dot(v1, v2)
                np.cross(v1[:3], v2[:3])
            return ops

        phase.tests.append(self.run_test("Vector Operations", test_vectors))

        phase.end_time = time.time()
        return phase

    # ========== PHASE 4: API & HTTP ==========

    def phase4_api_http(self) -> PhaseResult:
        """Phase 4: API and HTTP stress tests."""
        phase = PhaseResult(4, "API & HTTP")
        phase.start_time = time.time()

        # Test 1: Health endpoint
        def test_health():
            if not HTTPX_AVAILABLE:
                raise RuntimeError("httpx not available")

            async def check_health():
                async with httpx.AsyncClient(timeout=5.0) as client:
                    success = 0
                    for _ in range(self._scale(100)):
                        try:
                            r = await client.get(f"{self.api_base}/health")
                            if r.status_code == 200:
                                success += 1
                        except:
                            pass
                    return success

            return asyncio.run(check_health())

        phase.tests.append(self.run_test("Health Endpoint", test_health))

        # Test 2: RAM Universe
        def test_ram_universe():
            from l104_ram_universe import ram_universe
            ops = self._scale(200)
            for i in range(ops):
                ram_universe.cross_check_hallucination(f"thought_{i}", ["GOD_CODE"])
            return ops

        phase.tests.append(self.run_test("RAM Universe", test_ram_universe))

        # Test 3: Parallel engine
        def test_parallel():
            from l104_parallel_engine import parallel_engine
            ops = self._scale(20)
            for _ in range(ops):
                parallel_engine.run_high_speed_calculation(complexity=100000)
            return ops

        phase.tests.append(self.run_test("Parallel Engine", test_parallel))

        # Test 4: Saturation engine
        def test_saturation():
            from l104_saturation_engine import saturation_engine
            ops = self._scale(50)
            for _ in range(ops):
                saturation_engine.calculate_saturation()
            return ops

        phase.tests.append(self.run_test("Saturation Engine", test_saturation))

        phase.end_time = time.time()
        return phase

    # ========== PHASE 5: FILE I/O & CONCURRENCY ==========

    def phase5_file_concurrency(self) -> PhaseResult:
        """Phase 5: File I/O and concurrency stress tests."""
        phase = PhaseResult(5, "File I/O & Concurrency")
        phase.start_time = time.time()

        # Test 1: File I/O
        def test_file_io():
            ops = self._scale(1000)
            with tempfile.TemporaryDirectory() as tmpdir:
                for i in range(ops):
                    filepath = os.path.join(tmpdir, f"test_{i}.json")
                    data = {'id': i, 'data': 'x' * 100}
                    with open(filepath, 'w') as f:
                        json.dump(data, f)
                for i in range(ops):
                    filepath = os.path.join(tmpdir, f"test_{i}.json")
                    with open(filepath, 'r') as f:
                        json.load(f)
            return ops * 2

        phase.tests.append(self.run_test("File I/O", test_file_io))

        # Test 2: ThreadPool
        def test_threadpool():
            def cpu_task(n):
                return sum(i**2 for i in range(n))

            ops = self._scale(500)
            with concurrent.futures.ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 8) as executor:  # QUANTUM AMPLIFIED
                futures = [executor.submit(cpu_task, 10000) for _ in range(ops)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            return ops

        phase.tests.append(self.run_test("ThreadPool Execution", test_threadpool))

        # Test 3: ProcessPool (using picklable function)
        def test_processpool():
            ops = self._scale(50)
            # Use simple computation that doesn't require pickle
            import math
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(math.factorial, 100) for _ in range(ops)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            return ops

        phase.tests.append(self.run_test("ProcessPool Execution", test_processpool))

        # Test 4: Async gather
        def test_async_gather():
            async def async_work(n):
                await asyncio.sleep(0)
                return n * 2

            async def run_gather():
                tasks = [async_work(i) for i in range(self._scale(5000))]
                return await asyncio.gather(*tasks)

            results = asyncio.run(run_gather())
            return len(results)

        phase.tests.append(self.run_test("Async Gather", test_async_gather))

        # Test 5: Memory allocation
        def test_memory_alloc():
            allocations = []
            count = self._scale(100)
            for _ in range(count):
                allocations.append(bytearray(1024 * 1024))  # 1MB each
            del allocations
            return count

        phase.tests.append(self.run_test("Memory Allocation", test_memory_alloc))

        phase.end_time = time.time()
        return phase

    # ========== PHASE 6: ENDURANCE & CHAOS ==========

    def phase6_endurance_chaos(self) -> PhaseResult:
        """Phase 6: Endurance and chaos stress tests."""
        phase = PhaseResult(6, "Endurance & Chaos")
        phase.start_time = time.time()

        # Test 1: Chaos operations
        def test_chaos():
            ops = self._scale(10000)
            for _ in range(ops):
                op = random.choice(['hash', 'math', 'array', 'string'])
                if op == 'hash':
                    hashlib.sha256(str(random.random()).encode()).hexdigest()
                elif op == 'math' and NUMPY_AVAILABLE:
                    np.sin(np.random.rand(100))
                elif op == 'array' and NUMPY_AVAILABLE:
                    np.random.rand(50, 50) @ np.random.rand(50, 50)
                elif op == 'string':
                    ''.join(random.choices('abc123', k=1000))
            return ops

        phase.tests.append(self.run_test("Chaos Operations", test_chaos))

        # Test 2: Exception recovery
        def test_exceptions():
            ops = self._scale(1000)
            recovered = 0
            for _ in range(ops):
                try:
                    if random.random() < 0.5:
                        raise ValueError("simulated")
                except:
                    recovered += 1
            return ops

        phase.tests.append(self.run_test("Exception Recovery", test_exceptions))

        # Test 3: Queue throughput
        def test_queue():
            q = queue.Queue()
            consumed = [0]
            target = self._scale(5000)

            def producer():
                for i in range(target):
                    q.put(i)

            def consumer():
                while consumed[0] < target:
                    try:
                        q.get(timeout=0.01)
                        consumed[0] += 1
                    except:
                        pass

            threads = [threading.Thread(target=producer) for _ in range(2)]
            threads += [threading.Thread(target=consumer) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            return consumed[0]

        phase.tests.append(self.run_test("Queue Throughput", test_queue))

        # Test 4: Hash uniqueness
        def test_hash_unique():
            ops = self._scale(50000)
            hashes = set()
            for i in range(ops):
                h = hashlib.sha256(f"L104-{i}-{random.random()}".encode()).hexdigest()[:16]
                hashes.add(h)
            collisions = ops - len(hashes)
            if collisions > 0:
                raise RuntimeError(f"{collisions} hash collisions detected")
            return ops

        phase.tests.append(self.run_test("Hash Uniqueness", test_hash_unique))

        # Test 5: Memory pressure
        def test_memory_pressure():
            if not NUMPY_AVAILABLE:
                return 0
            cycles = self._scale(10)
            for _ in range(cycles):
                arrays = [np.random.rand(500, 500) for _ in range(20)]
                _ = sum(arr.sum() for arr in arrays)
                del arrays
            return cycles

        phase.tests.append(self.run_test("Memory Pressure", test_memory_pressure))

        phase.end_time = time.time()
        return phase

    # ========== RUN ALL ==========

    def run_all(self, phases: List[int] = None) -> Dict[str, Any]:
        """Run all or specified phases."""
        all_phases = {
            1: self.phase1_core_systems,
            2: self.phase2_database_async,
            3: self.phase3_neural_numpy,
            4: self.phase4_api_http,
            5: self.phase5_file_concurrency,
            6: self.phase6_endurance_chaos,
        }

        if phases is None:
            phases = list(all_phases.keys())

        start_time = time.time()

        for phase_num in phases:
            if phase_num in all_phases:
                print(f"\n{'=' * 60}")
                print(f"PHASE {phase_num}: {all_phases[phase_num].__doc__.split(':')[1].strip()}")
                print('=' * 60)

                result = all_phases[phase_num]()
                self.results.append(result)

                for test in result.tests:
                    status = "✅" if test.passed else "❌"
                    print(f"  {status} {test.name}: {test.operations} ops ({test.duration:.2f}s)")
                    if test.error:
                        print(f"      Error: {test.error}")

        total_time = time.time() - start_time
        total_ops = sum(r.total_operations for r in self.results)
        all_passed = all(r.all_passed for r in self.results)

        print(f"\n{'=' * 60}")
        print("STRESS TEST COMPLETE")
        print('=' * 60)
        print(f"Total Operations: {total_ops:,}")
        print(f"Total Duration: {total_time:.2f}s")
        print(f"Status: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")

        return {
            "timestamp": datetime.now().isoformat(),
            "quick_mode": self.quick_mode,
            "total_operations": total_ops,
            "total_duration_seconds": round(total_time, 4),
            "all_passed": all_passed,
            "phases": [r.to_dict() for r in self.results]
        }


def main():
    parser = argparse.ArgumentParser(description="L104 Stress Test Suite")
    parser.add_argument("--phase", type=int, nargs="+", help="Run specific phase(s)")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode (10% scale)")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--output", type=str, help="Save report to file")
    args = parser.parse_args()

    print("=" * 60)
    print("L104 SOVEREIGN NODE - STRESS TEST SUITE")
    print("=" * 60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    suite = L104StressTest(quick_mode=args.quick)
    report = suite.run_all(phases=args.phase)

    if args.json:
        print("\n" + json.dumps(report, indent=2))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")

    # Exit code based on pass/fail
    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
