#!/usr/bin/env python3
"""
L104 Benchmark Suite - Comprehensive performance testing and bottleneck detection
"""

import os
import sys
import time
import json
import statistics
import multiprocessing
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float  # operations per second
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'iterations': self.iterations,
            'total_time': self.total_time,
            'avg_time': self.avg_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'std_dev': self.std_dev,
            'throughput': self.throughput,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'timestamp': self.timestamp.isoformat()
        }

class BenchmarkSuite:
    """Comprehensive benchmark suite for L104 system"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.warmup_iterations = 10
        self.measure_iterations = 100
    
    def benchmark_function(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """Benchmark a single function"""
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        # Measure
        times = []
        for _ in range(self.measure_iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = self.measure_iterations / total_time
        
        result = BenchmarkResult(
            name=func.__name__,
            iterations=self.measure_iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            throughput=throughput
        )
        
        self.results.append(result)
        return result
    
    def benchmark_quantum_coherence(self):
        """Benchmark quantum coherence operations"""
        try:
            from l104_quantum_coherence import QuantumCoherenceEngine
            
            engine = QuantumCoherenceEngine()
            
            benchmarks = [
                ('create_superposition', lambda: engine.create_superposition([0, 1])),
                ('create_entanglement', lambda: engine.create_entanglement(0, 1, "phi+")),
                ('calculate_coherence', lambda: engine.register.calculate_coherence()),
                ('apply_god_code_phase', lambda: engine.apply_god_code_phase()),
                ('measure', lambda: engine.measure(qubit=0)),
            ]
            
            results = []
            for name, func in benchmarks:
                result = self.benchmark_function(func)
                result.name = f"quantum_{name}"
                results.append(result)
                print(f"  {name}: {result.avg_time:.6f}s ± {result.std_dev:.6f}s")
            
            return results
            
        except ImportError as e:
            print(f"Quantum coherence benchmark failed: {e}")
            return []
    
    def benchmark_cache_performance(self):
        """Benchmark cache performance"""
        try:
            from l104_optimization_engine import SmartCache
            
            cache = SmartCache(max_size_mb=10)
            
            # Test data
            test_data = {
                'small': 'a' * 100,
                'medium': 'b' * 10000,
                'large': 'c' * 100000,
                'dict': {f'key_{i}': f'value_{i}' for i in range(100)},
                'list': list(range(1000))
            }
            
            def cache_operations():
                # Set operations
                for key, value in test_data.items():
                    cache.set(key, value)
                
                # Get operations
                for key in test_data.keys():
                    cache.get(key)
                
                # Mixed operations
                for i in range(10):
                    cache.set(f'temp_{i}', f'data_{i}')
                    cache.get(f'temp_{i}')
            
            result = self.benchmark_function(cache_operations)
            result.name = "cache_operations"
            
            # Get cache stats
            stats = cache.get_stats()
            result.memory_usage_mb = stats['size_mb']
            
            print(f"  Cache operations: {result.avg_time:.6f}s")
            print(f"  Cache hit rate: {stats['hit_rate']:.1%}")
            
            return [result]
            
        except ImportError as e:
            print(f"Cache benchmark failed: {e}")
            return []
    
    def benchmark_parallel_processing(self):
        """Benchmark parallel processing performance"""
        try:
            from l104_optimization_engine import ParallelExecutor
            
            executor = ParallelExecutor()
            
            # CPU-bound function
            def cpu_intensive(n: int) -> float:
                total = 0.0
                for i in range(n):
                    total += i ** 0.5
                return total
            
            # I/O-bound function (simulated)
            def io_intensive(n: int) -> float:
                time.sleep(0.001)  # Simulate I/O
                return n * 1.5
            
            def benchmark_parallel():
                # Test with CPU-bound tasks
                items = list(range(100))
                results = executor.parallel_map(cpu_intensive, items, use_processes=True)
                return len(results)
            
            def benchmark_sequential():
                # Sequential version for comparison
                items = list(range(100))
                results = [cpu_intensive(item) for item in items]
                return len(results)
            
            # Benchmark parallel
            parallel_result = self.benchmark_function(benchmark_parallel)
            parallel_result.name = "parallel_processing"
            
            # Benchmark sequential
            sequential_result = self.benchmark_function(benchmark_sequential)
            sequential_result.name = "sequential_processing"
            
            # Calculate speedup
            speedup = sequential_result.avg_time / parallel_result.avg_time
            
            print(f"  Parallel: {parallel_result.avg_time:.3f}s")
            print(f"  Sequential: {sequential_result.avg_time:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")
            
            return [parallel_result, sequential_result]
            
        except ImportError as e:
            print(f"Parallel processing benchmark failed: {e}")
            return []
    
    def benchmark_memory_operations(self):
        """Benchmark memory-intensive operations"""
        import gc
        
        def create_large_objects():
            # Create and manipulate large objects
            large_list = [i ** 2 for i in range(10000)]
            large_dict = {str(i): i * 2 for i in range(10000)}
            large_string = 'x' * 100000
            
            # Perform operations
            sorted_list = sorted(large_list)
            filtered_dict = {k: v for k, v in large_dict.items() if v % 2 == 0}
            modified_string = large_string.replace('x', 'y')
            
            return len(sorted_list) + len(filtered_dict) + len(modified_string)
        
        def memory_allocation():
            # Test memory allocation speed
            arrays = []
            for _ in range(100):
                arr = np.random.rand(1000, 1000)
                arrays.append(arr)
            
            # Perform operations
            total = sum(arr.sum() for arr in arrays)
            return total
        
        results = []
        
        # Benchmark Python objects
        py_result = self.benchmark_function(create_large_objects)
        py_result.name = "python_memory_operations"
        results.append(py_result)
        
        # Benchmark NumPy arrays
        try:
            np_result = self.benchmark_function(memory_allocation)
            np_result.name = "numpy_memory_operations"
            results.append(np_result)
            
            print(f"  Python objects: {py_result.avg_time:.3f}s")
            print(f"  NumPy arrays: {np_result.avg_time:.3f}s")
            print(f"  NumPy speedup: {py_result.avg_time / np_result.avg_time:.2f}x")
            
        except ImportError:
            print(f"  Python objects: {py_result.avg_time:.3f}s")
            print("  NumPy not available for comparison")
        
        return results
    
    def benchmark_database_operations(self):
        """Benchmark database operations (if available)"""
        try:
            import sqlite3
            import tempfile
            
            def create_test_db():
                # Create in-memory database
                conn = sqlite3.connect(':memory:')
                cursor = conn.cursor()
                
                # Create table
                cursor.execute('''
                    CREATE TABLE test_data (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        value REAL,
                        timestamp DATETIME
                    )
                ''')
                
                # Insert data
                for i in range(1000):
                    cursor.execute(
                        'INSERT INTO test_data (name, value, timestamp) VALUES (?, ?, ?)',
                        (f'item_{i}', i * 1.5, datetime.now().isoformat())
                    )
                
                # Query data
                cursor.execute('SELECT COUNT(*) FROM test_data')
                count = cursor.fetchone()[0]
                
                # Complex query
                cursor.execute('''
                    SELECT AVG(value), MAX(value), MIN(value)
                    FROM test_data
                    WHERE value > 500
                ''')
                stats = cursor.fetchone()
                
                conn.close()
                return count
            
            result = self.benchmark_function(create_test_db)
            result.name = "database_operations"
            
            print(f"  Database operations: {result.avg_time:.3f}s")
            
            return [result]
            
        except ImportError as e:
            print(f"Database benchmark failed: {e}")
            return []
    
    def benchmark_network_operations(self):
        """Benchmark network operations (simulated)"""
        import socket
        import threading
        
        def simulated_network_request():
            # Simulate network latency
            time.sleep(0.01)  # 10ms latency
            
            # Simulate processing
            data = {'status': 'ok', 'data': list(range(100))}
            processed = json.dumps(data)
            parsed = json.loads(processed)
            
            return len(parsed['data'])
        
        def concurrent_requests():
            # Simulate concurrent requests
            def make_request():
                time.sleep(0.005)
                return 1
            
            # Use threading for concurrency
            threads = []
            results = []
            
            for _ in range(10):
                thread = threading.Thread(target=lambda: results.append(make_request()))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            return sum(results)
        
        results = []
        
        # Single request
        single_result = self.benchmark_function(simulated_network_request)
        single_result.name = "network_single_request"
        results.append(single_result)
        
        # Concurrent requests
        concurrent_result = self.benchmark_function(concurrent_requests)
        concurrent_result.name = "network_concurrent_requests"
        results.append(concurrent_result)
        
        print(f"  Single request: {single_result.avg_time:.3f}s")
        print(f"  Concurrent requests: {concurrent_result.avg_time:.3f}s")
        print(f"  Concurrency efficiency: {single_result.avg_time / concurrent_result.avg_time:.2f}x")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and generate comprehensive report"""
        print("=" * 70)
        print("L104 COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 70)
        
        all_results = []
        
        # 1. Quantum Coherence
        print("\n[1] QUANTUM COHERENCE BENCHMARKS")
        quantum_results = self.benchmark_quantum_coherence()
        all_results.extend(quantum_results)
        
        # 2. Cache Performance
        print("\n[2] CACHE PERFORMANCE BENCHMARKS")
        cache_results = self.benchmark_cache_performance()
        all_results.extend(cache_results)
        
        # 3. Parallel Processing
        print("\n[3] PARALLEL PROCESSING BENCHMARKS")
        parallel_results = self.benchmark_parallel_processing()
        all_results.extend(parallel_results)
        
        # 4. Memory Operations
        print("\n[4] MEMORY OPERATIONS BENCHMARKS")
        memory_results = self.benchmark_memory_operations()
        all_results.extend(memory_results)
        
        # 5. Database Operations
        print("\n[5] DATABASE OPERATIONS BENCHMARKS")
        db_results = self.benchmark_database_operations()
        all_results.extend(db_results)
        
        # 6. Network Operations
        print("\n[6] NETWORK OPERATIONS BENCHMARKS")
        network_results = self.benchmark_network_operations()
        all_results.extend(network_results)
        
        # Generate report
        report = self._generate_report(all_results)
        
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        
        return report
    
    def _generate_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        # Convert results to dict
        results_dict = [r.to_dict() for r in results]
        
        # Find bottlenecks (slowest operations)
        bottlenecks = sorted(results_dict, key=lambda x: x['avg_time'], reverse=True)[:5]
        
        # Calculate overall statistics
        total_time = sum(r['total_time'] for r in results_dict)
        avg_throughput = statistics.mean(r['throughput'] for r in results_dict if r['throughput'] > 0)
        
        # System information
        system_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': multiprocessing.cpu_count(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Recommendations
        recommendations = self._generate_recommendations(bottlenecks)
        
        report = {
            'system': system_info,
            'summary': {
                'total_benchmarks': len(results_dict),
                'total_execution_time': total_time,
                'average_throughput': avg_throughput,
            },
            'results': results_dict,
            'bottlenecks': bottlenecks,
            'recommendations': recommendations
        }
        
        return report
    
    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on bottlenecks"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            name = bottleneck['name']
            avg_time = bottleneck['avg_time']
            
            if avg_time > 1.0:
                priority = 'CRITICAL'
            elif avg_time > 0.1:
                priority = 'HIGH'
            elif avg_time > 0.01:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            
            # Generate specific recommendations based on benchmark type
            if 'quantum_' in name:
                recommendations.append({
                    'priority': priority,
                    'area': 'Quantum Coherence',
                    'issue': f'Slow quantum operation: {name} ({avg_time:.3f}s)',
                    'action': 'Consider optimizing quantum state operations or reducing qubit count'
                })
            
            elif 'cache_' in name:
                recommendations.append({
                    'priority': priority,
                    'area': 'Caching',
                    'issue': f'Cache operations slow: {avg_time:.3f}s',
                    'action': 'Consider increasing cache size or optimizing cache key generation'
                })
            
            elif 'parallel' in name or 'sequential' in name:
                recommendations.append({
                    'priority': priority,
                    'area': 'Parallel Processing',
                    'issue': f'Processing speed: {avg_time:.3f}s',
                    'action': 'Consider better workload balancing or increasing worker count'
                })
            
            elif 'memory' in name:
                recommendations.append({
                    'priority': priority,
                    'area': 'Memory Operations',
                    'issue': f'Memory operations slow: {avg_time:.3f}s',
                    'action': 'Consider using NumPy arrays or object pooling'
                })
            
            elif 'database' in name:
                recommendations.append({
                    'priority': priority,
                    'area': 'Database',
                    'issue': f'Database operations slow: {avg_time:.3f}s',
                    'action': 'Consider adding indexes or optimizing queries'
                })
            
            elif 'network' in name:
                recommendations.append({
                    'priority': priority,
                    'area': 'Network',
                    'issue': f'Network operations slow: {avg_time:.3f}s',
                    'action': 'Consider connection pooling or request batching'
                })
        
        # Add general recommendations
        if not recommendations:
            recommendations.append({
                'priority': 'INFO',
                'area': 'Overall',
                'issue': 'No significant bottlenecks detected',
                'action': 'System performance is good'
            })
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save benchmark report to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'l104_benchmark_report_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Benchmark report saved to: {filename}")
        return filename
    
    def compare_reports(self, report1: Dict[str, Any], report2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two benchmark reports"""
        comparison = {
            'comparison_timestamp': datetime.now().isoformat(),
            'improvements': [],
            'regressions': [],
            'summary': {}
        }
        
        # Create lookup dictionaries
        results1 = {r['name']: r for r in report1.get('results', [])}
        results2 = {r['name']: r for r in report2.get('results', [])}
        
        # Compare common benchmarks
        common_names = set(results1.keys()) & set(results2.keys())
        
        improvements = 0
        regressions = 0
        
        for name in common_names:
            r1 = results1[name]
            r2 = results2[name]
            
            time_diff = r2['avg_time'] - r1['avg_time']
            percent_diff = (time_diff / r1['avg_time']) * 100 if r1['avg_time'] > 0 else 0
            
            if time_diff < 0:
                # Improvement (faster)
                improvements += 1
                comparison['improvements'].append({
                    'name': name,
                    'time_before': r1['avg_time'],
                    'time_after': r2['avg_time'],
                    'improvement': abs(percent_diff),
                    'faster_by': abs(time_diff)
                })
            elif time_diff > 0:
                # Regression (slower)
                regressions += 1
                comparison['regressions'].append({
                    'name': name,
                    'time_before': r1['avg_time'],
                    'time_after': r2['avg_time'],
                    'regression': percent_diff,
                    'slower_by': time_diff
                })
        
        comparison['summary'] = {
            'total_compared': len(common_names),
            'improvements': improvements,
            'regressions': regressions,
            'improvement_ratio': improvements / len(common_names) if common_names else 0
        }
        
        return comparison

def main():
    """Main entry point for benchmark suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='L104 Benchmark Suite')
    parser.add_argument('--run-all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--quantum', action='store_true', help='Run quantum coherence benchmarks')
    parser.add_argument('--cache', action='store_true', help='Run cache benchmarks')
    parser.add_argument('--parallel', action='store_true', help='Run parallel processing benchmarks')
    parser.add_argument('--memory', action='store_true', help='Run memory benchmarks')
    parser.add_argument('--database', action='store_true', help='Run database benchmarks')
    parser.add_argument('--network', action='store_true', help='Run network benchmarks')
    parser.add_argument('--save', type=str, help='Save report to specified file')
    parser.add_argument('--compare', nargs=2, help='Compare two benchmark reports')
    
    args = parser.parse_args()
    
    suite = BenchmarkSuite()
    
    if args.compare:
        # Compare two reports
        file1, file2 = args.compare
        
        try:
            with open(file1, 'r') as f:
                report1 = json.load(f)
            
            with open(file2, 'r') as f:
                report2 = json.load(f)
            
            comparison = suite.compare_reports(report1, report2)
            
            print("\n" + "=" * 70)
            print("BENCHMARK COMPARISON REPORT")
            print("=" * 70)
            
            print(f"\nSummary:")
            print(f"  Compared: {comparison['summary']['total_compared']} benchmarks")
            print(f"  Improvements: {comparison['summary']['improvements']}")
            print(f"  Regressions: {comparison['summary']['regressions']}")
            print(f"  Improvement ratio: {comparison['summary']['improvement_ratio']:.1%}")
            
            if comparison['improvements']:
                print(f"\nTop improvements:")
                for imp in sorted(comparison['improvements'], key=lambda x: x['improvement'], reverse=True)[:3]:
                    print(f"  • {imp['name']}: {imp['improvement']:.1f}% faster")
            
            if comparison['regressions']:
                print(f"\nTop regressions:")
                for reg in sorted(comparison['regressions'], key=lambda x: x['regression'], reverse=True)[:3]:
                    print(f"  • {reg['name']}: {reg['regression']:.1f}% slower")
            
            # Save comparison
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            comp_file = f'l104_benchmark_comparison_{timestamp}.json'
            
            with open(comp_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            print(f"\nComparison saved to: {comp_file}")
            
        except FileNotFoundError as e:
            print(f"Error: Could not find file {e.filename}")
        except Exception as e:
            print(f"Error comparing reports: {e}")
    
    elif args.run_all or not any([args.quantum, args.cache, args.parallel, args.memory, args.database, args.network]):
        # Run comprehensive benchmark
        report = suite.run_comprehensive_benchmark()
        
        # Save if requested
        if args.save:
            suite.save_report(report, args.save)
        else:
            # Ask if user wants to save
            response = input("\nSave benchmark report? (y/n): ")
            if response.lower() == 'y':
                suite.save_report(report)
    
    else:
        # Run specific benchmarks
        results = []
        
        if args.quantum:
            print("\nRunning quantum coherence benchmarks...")
            results.extend(suite.benchmark_quantum_coherence())
        
        if args.cache:
            print("\nRunning cache benchmarks...")
            results.extend(suite.benchmark_cache_performance())
        
        if args.parallel:
            print("\nRunning parallel processing benchmarks...")
            results.extend(suite.benchmark_parallel_processing())
        
        if args.memory:
            print("\nRunning memory benchmarks...")
            results.extend(suite.benchmark_memory_operations())
        
        if args.database:
            print("\nRunning database benchmarks...")
            results.extend(suite.benchmark_database_operations())
        
        if args.network:
            print("\nRunning network benchmarks...")
            results.extend(suite.benchmark_network_operations())
        
        # Generate mini-report
        if results:
            report = suite._generate_report(results)
            
            print("\n" + "=" * 70)
            print("BENCHMARK SUMMARY")
            print("=" * 70)
            
            for result in report['results']:
                print(f"{result['name']:30} {result['avg_time']:8.6f}s ± {result['std_dev']:8.6f}s")
            
            if args.save:
                suite.save_report(report, args.save)

if __name__ == '__main__':
    main()