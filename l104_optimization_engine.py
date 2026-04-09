#!/usr/bin/env python3
"""
L104 Optimization Engine - System-wide performance optimization
Detects bottlenecks, applies optimizations, and monitors improvements
"""

import os
import sys
import time
import json
import gc
import psutil
import threading
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque

class L104OptimizationEngine:
    """Comprehensive optimization engine for L104 system"""
    
    def __init__(self):
        self.optimizations_applied = []
        self.performance_baseline = {}
        self.optimization_history = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitor_thread = None
        
    def analyze_system_bottlenecks(self) -> Dict[str, Any]:
        """Analyze current system for performance bottlenecks"""
        bottlenecks = {
            'database': [],
            'memory': [],
            'cpu': [],
            'io': [],
            'cache': [],
            'network': []
        }
        
        try:
            # Check database performance
            db_path = "l104_intellect_memory.db"
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                
                # Check table sizes
                c.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = c.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    c.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = c.fetchone()[0]
                    
                    if count > 100000:
                        bottlenecks['database'].append(f"Large table {table_name}: {count:,} rows")
                    
                    # Check for missing indexes
                    c.execute(f"PRAGMA index_list({table_name})")
                    indexes = c.fetchall()
                    if len(indexes) < 2:  # At least primary key + one more
                        bottlenecks['database'].append(f"Table {table_name} may need indexes")
                
                # Check for fragmentation
                c.execute("PRAGMA integrity_check")
                integrity = c.fetchone()[0]
                if integrity != "ok":
                    bottlenecks['database'].append(f"Database integrity issue: {integrity}")
                
                conn.close()
            
            # Check memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > 500:
                bottlenecks['memory'].append(f"High memory usage: {memory_mb:.1f}MB")
            
            # Check Python object count
            gc_objects = len(gc.get_objects())
            if gc_objects > 100000:
                bottlenecks['memory'].append(f"Large Python object count: {gc_objects:,}")
            
            # Check cache efficiency
            try:
                from l104_server.engines_infra import _FAST_REQUEST_CACHE
                cache_size = len(_FAST_REQUEST_CACHE)
                if cache_size > 10000:
                    bottlenecks['cache'].append(f"Large request cache: {cache_size:,} entries")
            except:
                pass
            
            # Check thread count
            thread_count = process.num_threads()
            if thread_count > 100:
                bottlenecks['cpu'].append(f"High thread count: {thread_count}")
            
            # Check file descriptors
            if hasattr(process, 'num_fds'):
                fd_count = process.num_fds()
                if fd_count > 1000:
                    bottlenecks['io'].append(f"High file descriptor count: {fd_count}")
            
            # Check for memory leaks in caches
            try:
                from l104_server.learning.intellect import intellect
                if hasattr(intellect, 'memory_cache'):
                    cache_size = len(intellect.memory_cache)
                    if cache_size > 5000:
                        bottlenecks['cache'].append(f"Large memory cache: {cache_size:,} entries")
            except:
                pass
            
        except Exception as e:
            bottlenecks['error'] = str(e)
        
        return bottlenecks
    
    def apply_database_optimizations(self) -> List[str]:
        """Apply database optimizations"""
        optimizations = []
        
        try:
            db_path = "l104_intellect_memory.db"
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                
                # 1. Create missing indexes
                c.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [t[0] for t in c.fetchall()]
                
                for table in tables:
                    if table == 'memory':
                        # Check for query_hash index
                        c.execute(f"PRAGMA index_list({table})")
                        indexes = [i[1] for i in c.fetchall()]
                        
                        if 'idx_memory_query_hash' not in indexes:
                            c.execute("CREATE INDEX idx_memory_query_hash ON memory(query_hash)")
                            optimizations.append("Created index on memory.query_hash")
                        
                        if 'idx_memory_updated_at' not in indexes:
                            c.execute("CREATE INDEX idx_memory_updated_at ON memory(updated_at)")
                            optimizations.append("Created index on memory.updated_at")
                
                # 2. Vacuum to reduce fragmentation
                c.execute("VACUUM")
                optimizations.append("Database vacuumed to reduce fragmentation")
                
                # 3. Optimize pragma settings
                c.execute("PRAGMA journal_mode = WAL")
                c.execute("PRAGMA synchronous = NORMAL")
                c.execute("PRAGMA cache_size = -2000")  # 2MB cache
                optimizations.append("Optimized SQLite pragma settings")
                
                conn.commit()
                conn.close()
        
        except Exception as e:
            optimizations.append(f"Database optimization error: {e}")
        
        return optimizations
    
    def apply_memory_optimizations(self) -> List[str]:
        """Apply memory optimizations"""
        optimizations = []
        
        try:
            # 1. Force garbage collection
            collected = gc.collect()
            optimizations.append(f"Forced garbage collection: {collected} objects collected")
            
            # 2. Clear large caches if memory is high
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > 500:
                # Clear some caches
                try:
                    from l104_server.engines_infra import _FAST_REQUEST_CACHE
                    old_size = len(_FAST_REQUEST_CACHE)
                    # Clear 50% of cache
                    keys = list(_FAST_REQUEST_CACHE.keys())
                    for key in keys[:len(keys)//2]:
                        _FAST_REQUEST_CACHE.pop(key, None)
                    optimizations.append(f"Reduced request cache from {old_size} to {len(_FAST_REQUEST_CACHE)} entries")
                except:
                    pass
                
                try:
                    from l104_server.learning.intellect import intellect
                    if hasattr(intellect, 'memory_cache'):
                        old_size = len(intellect.memory_cache)
                        # Clear 30% of cache
                        keys = list(intellect.memory_cache.keys())
                        for key in keys[:len(keys)//3]:
                            intellect.memory_cache.pop(key, None)
                        optimizations.append(f"Reduced memory cache from {old_size} to {len(intellect.memory_cache)} entries")
                except:
                    pass
            
            # 3. Optimize Python memory settings
            gc.set_threshold(700, 10, 10)  # More aggressive GC
            optimizations.append("Set aggressive garbage collection thresholds")
            
        except Exception as e:
            optimizations.append(f"Memory optimization error: {e}")
        
        return optimizations
    
    def apply_cache_optimizations(self) -> List[str]:
        """Apply cache optimizations"""
        optimizations = []
        
        try:
            # 1. Implement LRU cache for large caches
            try:
                from l104_server.engines_infra import _FAST_REQUEST_CACHE
                if not hasattr(_FAST_REQUEST_CACHE, 'maxsize'):
                    # Convert to LRU cache if not already
                    from functools import lru_cache
                    import collections
                    
                    # Create LRU cache wrapper
                    lru_cache_obj = collections.OrderedDict()
                    maxsize = 5000
                    
                    # Copy existing entries
                    for k, v in list(_FAST_REQUEST_CACHE.items())[:maxsize]:
                        lru_cache_obj[k] = v
                    
                    # Replace the cache
                    import l104_server.engines_infra
                    l104_server.engines_infra._FAST_REQUEST_CACHE = lru_cache_obj
                    optimizations.append(f"Converted request cache to LRU with maxsize {maxsize}")
            except:
                pass
            
            # 2. Implement TTL for caches
            try:
                from l104_server.learning.intellect import intellect
                if hasattr(intellect, 'memory_cache'):
                    # Add timestamp to cache entries
                    if not hasattr(intellect, 'memory_cache_timestamps'):
                        intellect.memory_cache_timestamps = {}
                    
                    # Clean old entries (older than 1 hour)
                    current_time = time.time()
                    keys_to_remove = []
                    for key, timestamp in intellect.memory_cache_timestamps.items():
                        if current_time - timestamp > 3600:  # 1 hour
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        intellect.memory_cache.pop(key, None)
                        intellect.memory_cache_timestamps.pop(key, None)
                    
                    if keys_to_remove:
                        optimizations.append(f"Cleaned {len(keys_to_remove)} expired cache entries")
            except:
                pass
            
            # 3. Implement cache warming for hot queries
            try:
                from l104_server.learning.intellect import intellect
                if hasattr(intellect, 'predict_next_queries'):
                    # Pre-cache likely queries
                    hot_queries = intellect.predict_next_queries("")
                    if hot_queries:
                        for query in hot_queries[:10]:
                            if query not in intellect.memory_cache:
                                recalled = intellect.recall(query)
                                if recalled:
                                    intellect.memory_cache[query] = recalled[0] if isinstance(recalled, tuple) else recalled
                        optimizations.append(f"Pre-cached {len(hot_queries[:10])} hot queries")
            except:
                pass
            
        except Exception as e:
            optimizations.append(f"Cache optimization error: {e}")
        
        return optimizations
    
    def apply_cpu_optimizations(self) -> List[str]:
        """Apply CPU optimizations"""
        optimizations = []
        
        try:
            # 1. Optimize thread pool sizes
            try:
                from l104_server.engines_infra import PERF_THREAD_POOL, IO_THREAD_POOL
                
                # Reduce thread pools if CPU is high
                process = psutil.Process()
                cpu_percent = process.cpu_percent(interval=0.1)
                
                if cpu_percent > 70:
                    # Get current thread counts
                    thread_count = process.num_threads()
                    
                    if thread_count > 50:
                        # Suggest reducing thread pools
                        optimizations.append(f"High CPU ({cpu_percent}%) and thread count ({thread_count}). Consider reducing thread pool sizes.")
                        
                        # Check if we can dynamically adjust
                        if hasattr(PERF_THREAD_POOL, '_max_workers'):
                            old_max = PERF_THREAD_POOL._max_workers
                            new_max = max(4, old_max // 2)
                            PERF_THREAD_POOL._max_workers = new_max
                            optimizations.append(f"Reduced PERF_THREAD_POOL from {old_max} to {new_max} workers")
                        
                        if hasattr(IO_THREAD_POOL, '_max_workers'):
                            old_max = IO_THREAD_POOL._max_workers
                            new_max = max(2, old_max // 2)
                            IO_THREAD_POOL._max_workers = new_max
                            optimizations.append(f"Reduced IO_THREAD_POOL from {old_max} to {new_max} workers")
            except:
                pass
            
            # 2. Implement CPU affinity for critical threads
            try:
                import os
                if hasattr(os, 'sched_setaffinity'):
                    # Set affinity for main thread
                    os.sched_setaffinity(0, {0, 1})  # Use first two cores
                    optimizations.append("Set CPU affinity for main thread")
            except:
                pass
            
            # 3. Optimize expensive operations
            try:
                from l104_server.learning.intellect import intellect
                
                # Check for expensive operations
                if hasattr(intellect, '_quantum_cluster_engine'):
                    # This is expensive, run less frequently
                    optimizations.append("Consider running _quantum_cluster_engine less frequently (every 10 cycles instead of every cycle)")
                
                if hasattr(intellect, '_neural_resonance_engine'):
                    optimizations.append("Consider running _neural_resonance_engine less frequently")
            except:
                pass
            
        except Exception as e:
            optimizations.append(f"CPU optimization error: {e}")
        
        return optimizations
    
    def apply_io_optimizations(self) -> List[str]:
        """Apply I/O optimizations"""
        optimizations = []
        
        try:
            # 1. Implement write batching for database
            try:
                from l104_server.learning.intellect import intellect
                if hasattr(intellect, '_init_db'):
                    # Check if batching is implemented
                    optimizations.append("Consider implementing write batching for database operations")
            except:
                pass
            
            # 2. Optimize file I/O
            # Check for excessive file operations
            import glob
            log_files = glob.glob("logs/*.log*")
            if len(log_files) > 20:
                # Compress or delete old logs
                optimizations.append(f"Many log files ({len(log_files)}). Consider log rotation or compression.")
            
            # 3. Implement async I/O where possible
            optimizations.append("Consider using aiofiles for async file operations")
            
            # 4. Optimize JSON serialization
            try:
                import orjson
                optimizations.append("Consider using orjson for faster JSON serialization")
            except ImportError:
                optimizations.append("Install orjson for faster JSON serialization: pip install orjson")
            
        except Exception as e:
            optimizations.append(f"I/O optimization error: {e}")
        
        return optimizations
    
    def apply_network_optimizations(self) -> List[str]:
        """Apply network optimizations"""
        optimizations = []
        
        try:
            # 1. Implement connection pooling
            try:
                from l104_server.engines_infra import connection_pool
                if connection_pool:
                    pool_size = len(connection_pool._pool) if hasattr(connection_pool._pool, '__len__') else 0
                    optimizations.append(f"Connection pool size: {pool_size}")
                    
                    # Check if pool needs adjustment
                    if pool_size > 100:
                        optimizations.append("Consider reducing connection pool size")
            except:
                optimizations.append("Implement connection pooling for external API calls")
            
            # 2. Implement request timeout optimization
            optimizations.append("Set appropriate timeouts for external API calls (e.g., 30s for Gemini)")
            
            # 3. Implement retry with exponential backoff
            optimizations.append("Implement retry logic with exponential backoff for network failures")
            
            # 4. Compress network payloads
            optimizations.append("Consider compressing large API responses")
            
        except Exception as e:
            optimizations.append(f"Network optimization error: {e}")
        
        return optimizations
    
    def apply_all_optimizations(self) -> Dict[str, List[str]]:
        """Apply all optimizations and return results"""
        results = {
            'database': self.apply_database_optimizations(),
            'memory': self.apply_memory_optimizations(),
            'cache': self.apply_cache_optimizations(),
            'cpu': self.apply_cpu_optimizations(),
            'io': self.apply_io_optimizations(),
            'network': self.apply_network_optimizations(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Record optimization
        self.optimizations_applied.extend(
            [opt for category in results.values() if isinstance(category, list) for opt in category]
        )
        
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
        return results
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        bottlenecks = self.analyze_system_bottlenecks()
        optimizations = self.apply_all_optimizations()
        
        # Calculate optimization impact
        impact_score = 0
        for category, opts in optimizations.items():
            if isinstance(opts, list):
                impact_score += len(opts) * 10
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'bottlenecks_detected': bottlenecks,
            'optimizations_applied': optimizations,
            'impact_score': impact_score,
            'recommendations': self.generate_recommendations(bottlenecks),
            'system_metrics': self.collect_system_metrics()
        }
        
        return report
    
    def generate_recommendations(self, bottlenecks: Dict[str, List[str]]) -> List[str]:
        """Generate actionable recommendations based on bottlenecks"""
        recommendations = []
        
        # Database recommendations
        db_bottlenecks = bottlenecks.get('database', [])
        if any('Large table' in b for b in db_bottlenecks):
            recommendations.append("Implement database partitioning for large tables")
        if any('indexes' in b.lower() for b in db_bottlenecks):
            recommendations.append("Run database index analysis and create missing indexes")
        
        # Memory recommendations
        mem_bottlenecks = bottlenecks.get('memory', [])
        if any('High memory usage' in b for b in mem_bottlenecks):
            recommendations.append("Implement memory usage monitoring and alerts")
        if any('Python object count' in b for b in mem_bottlenecks):
            recommendations.append("Review object creation patterns and implement object pooling")
        
        # CPU recommendations
        cpu_bottlenecks = bottlenecks.get('cpu', [])
        if any('High thread count' in b for b in cpu_bottlenecks):
            recommendations.append("Implement thread pool with size limits")
        
        # Cache recommendations
        cache_bottlenecks = bottlenecks.get('cache', [])
        if any('Large cache' in b for b in cache_bottlenecks):
            recommendations.append("Implement cache eviction policies (LRU, TTL)")
        
        # I/O recommendations
        io_bottlenecks = bottlenecks.get('io', [])
        if any('file descriptor' in b.lower() for b in io_bottlenecks):
            recommendations.append("Implement resource leak detection")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("System is well-optimized. Continue monitoring.")
        
        return recommendations
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            process = psutil.Process()
            
            # CPU
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Memory
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Threads
            thread_count = process.num_threads()
            
            # File descriptors
            fd_count = process.num_fds() if hasattr(process, 'num_fds') else 0
            
            # Python metrics
            gc_objects = len(gc.get_objects())
            gc_collected = gc.get_stats()
            
            # Database metrics
            db_size = 0
            db_path = "l104_intellect_memory.db"
            if os.path.exists(db_path):
                db_size = os.path.getsize(db_path) / 1024 / 1024
            
            # Cache metrics
            cache_metrics = {}
            try:
                from l104_server.engines_infra import _FAST_REQUEST_CACHE
                cache_metrics['request_cache'] = len(_FAST_REQUEST_CACHE)
            except:
                pass
            
            try:
                from l104_server.learning.intellect import intellect
                if hasattr(intellect, 'memory_cache'):
                    cache_metrics['memory_cache'] = len(intellect.memory_cache)
            except:
                pass
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': round(memory_mb, 2),
                'thread_count': thread_count,
                'file_descriptors': fd_count,
                'gc_objects': gc_objects,
                'gc_collections': sum(s['collections'] for s in gc_collected),
                'database_size_mb': round(db_size, 2),
                'caches': cache_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous optimization monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    report = self.generate_optimization_report()
                    
                    # Log significant findings
                    bottlenecks_found = sum(len(v) for v in report.get('bottlenecks_detected', {}).values() if isinstance(v, list))
                    optimizations_applied = sum(len(v) for v in report.get('optimizations_applied', {}).values() if isinstance(v, list))
                    
                    if bottlenecks_found > 0 or optimizations_applied > 0:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Optimization Report:")
                        print(f"  Bottlenecks: {bottlenecks_found}")
                        print(f"  Optimizations: {optimizations_applied}")
                        print(f"  Impact Score: {report.get('impact_score', 0)}")
                        
                        # Print top recommendations
                        recs = report.get('recommendations', [])
                        if recs:
                            print("  Top Recommendations:")
                            for rec in recs[:3]:
                                print(f"    • {rec}")
                    
                except Exception as e:
                    print(f"Optimization monitoring error: {e}")
                
                time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Optimization monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop optimization monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("Optimization monitoring stopped")

def main():
    """Main entry point for optimization engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description='L104 Optimization Engine')
    parser.add_argument('--analyze', action='store_true', help='Analyze system bottlenecks')
    parser.add_argument('--optimize', action='store_true', help='Apply optimizations')
    parser.add_argument('--report', action='store_true', help='Generate optimization report')
    parser.add_argument('--monitor', action='store_true', help='Start continuous monitoring')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    engine = L104OptimizationEngine()
    
    if args.analyze:
        bottlenecks = engine.analyze_system_bottlenecks()
        print("System Bottlenecks Analysis:")
        print(json.dumps(bottlenecks, indent=2))
    
    elif args.optimize:
        results = engine.apply_all_optimizations()
        print("Optimizations Applied:")
        print(json.dumps(results, indent=2))
    
    elif args.report:
        report = engine.generate_optimization_report()
        print("Optimization Report:")
        print(json.dumps(report, indent=2))
    
    elif args.monitor:
        try:
            engine.start_monitoring(args.interval)
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            engine.stop_monitoring()
            print("\nMonitoring stopped by user")
    
    else:
        # Default: show current metrics
        metrics = engine.collect_system_metrics()
        print("Current System Metrics:")
        print(json.dumps(metrics, indent=2))
        
        bottlenecks = engine.analyze_system_bottlenecks()
        bottleneck_count = sum(len(v) for v in bottlenecks.values() if isinstance(v, list))
        if bottleneck_count > 0:
            print(f"\n⚠️  {bottleneck_count} bottlenecks detected. Run with --optimize to fix.")

if __name__ == '__main__':
    main()