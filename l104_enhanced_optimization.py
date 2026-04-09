#!/usr/bin/env python3
"""
L104 Enhanced Optimization System
Comprehensive performance optimization, monitoring, and auto-tuning
"""

import os
import sys
import time
import json
import gc
import psutil
import threading
import sqlite3
import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Optimization priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5

@dataclass
class OptimizationResult:
    """Result of an optimization operation"""
    level: OptimizationLevel
    category: str
    description: str
    impact_score: float  # 0-100
    applied: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    thread_count: int
    file_descriptors: int
    gc_objects: int
    gc_collections: int
    database_size_mb: float
    cache_hit_rate: float
    response_time_ms: float
    request_rate: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': round(self.cpu_percent, 2),
            'memory_mb': round(self.memory_mb, 2),
            'memory_percent': round(self.memory_percent, 2),
            'thread_count': self.thread_count,
            'file_descriptors': self.file_descriptors,
            'gc_objects': self.gc_objects,
            'gc_collections': self.gc_collections,
            'database_size_mb': round(self.database_size_mb, 2),
            'cache_hit_rate': round(self.cache_hit_rate, 4),
            'response_time_ms': round(self.response_time_ms, 2),
            'request_rate': round(self.request_rate, 2),
            'error_rate': round(self.error_rate, 4)
        }

class L104EnhancedOptimizer:
    """Enhanced optimization system for L104"""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=1000)
        self.metrics_history = deque(maxlen=500)
        self.alert_history = deque(maxlen=100)
        self.optimization_rules = self._load_optimization_rules()
        self.monitoring_active = False
        self.monitor_thread = None
        self.alert_callbacks = []
        self.start_time = time.time()
        
        # Performance baselines
        self.baselines = {
            'response_time': 100.0,  # ms
            'memory_usage': 500.0,   # MB
            'cpu_usage': 70.0,       # percent
            'cache_hit_rate': 0.7,   # 70%
            'error_rate': 0.05       # 5%
        }
        
        # Register default alert callbacks
        self.register_alert_callback(self._log_alert)
        self.register_alert_callback(self._console_alert)
    
    def _load_optimization_rules(self) -> List[Dict[str, Any]]:
        """Load optimization rules"""
        return [
            {
                'id': 'memory_high',
                'condition': lambda m: m.memory_percent > 80,
                'action': self._optimize_memory_critical,
                'level': OptimizationLevel.CRITICAL,
                'description': 'Memory usage above 80%'
            },
            {
                'id': 'cpu_high',
                'condition': lambda m: m.cpu_percent > 85,
                'action': self._optimize_cpu_critical,
                'level': OptimizationLevel.CRITICAL,
                'description': 'CPU usage above 85%'
            },
            {
                'id': 'response_slow',
                'condition': lambda m: m.response_time_ms > self.baselines['response_time'] * 2,
                'action': self._optimize_response_time,
                'level': OptimizationLevel.HIGH,
                'description': 'Response time significantly degraded'
            },
            {
                'id': 'cache_poor',
                'condition': lambda m: m.cache_hit_rate < self.baselines['cache_hit_rate'] * 0.5,
                'action': self._optimize_cache,
                'level': OptimizationLevel.HIGH,
                'description': 'Cache hit rate below target'
            },
            {
                'id': 'error_high',
                'condition': lambda m: m.error_rate > self.baselines['error_rate'] * 2,
                'action': self._optimize_error_rate,
                'level': OptimizationLevel.HIGH,
                'description': 'Error rate above threshold'
            },
            {
                'id': 'thread_high',
                'condition': lambda m: m.thread_count > 200,
                'action': self._optimize_threads,
                'level': OptimizationLevel.MEDIUM,
                'description': 'High thread count'
            },
            {
                'id': 'gc_frequent',
                'condition': lambda m: m.gc_collections > 100,
                'action': self._optimize_gc,
                'level': OptimizationLevel.MEDIUM,
                'description': 'Frequent garbage collection'
            },
            {
                'id': 'db_large',
                'condition': lambda m: m.database_size_mb > 1000,
                'action': self._optimize_database,
                'level': OptimizationLevel.MEDIUM,
                'description': 'Database size large'
            }
        ]
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        try:
            process = psutil.Process()
            
            # Basic process metrics
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            thread_count = process.num_threads()
            fd_count = process.num_fds() if hasattr(process, 'num_fds') else 0
            
            # GC metrics
            gc_stats = gc.get_stats()
            gc_collections = sum(stat['collections'] for stat in gc_stats)
            gc_objects = len(gc.get_objects())
            
            # Database metrics
            db_size = 0
            db_path = "l104_intellect_memory.db"
            if os.path.exists(db_path):
                db_size = os.path.getsize(db_path) / 1024 / 1024
            
            # Cache metrics (approximate)
            cache_hit_rate = 0.7  # Default, will be updated from actual cache
            try:
                from l104_server.engines_infra import _FAST_REQUEST_CACHE
                cache_size = len(_FAST_REQUEST_CACHE)
                # Simple cache hit rate estimation
                cache_hit_rate = min(0.95, cache_size / 1000) if cache_size > 0 else 0.7
            except:
                pass
            
            # Response time and request rate (simplified)
            response_time_ms = 50.0  # Default
            request_rate = 1.0  # Default
            error_rate = 0.01  # Default
            
            # Try to get actual metrics from server if available
            try:
                from l104_server.engines_infra import performance_metrics
                perf_report = performance_metrics.get_performance_report()
                recall_stats = perf_report.get('recall_stats', {})
                response_time_ms = recall_stats.get('avg_latency_ms', 50.0)
                cache_eff = perf_report.get('cache_efficiency', {})
                cache_hit_rate = cache_eff.get('total_hit_rate', 0.7)
            except:
                pass
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                thread_count=thread_count,
                file_descriptors=fd_count,
                gc_objects=gc_objects,
                gc_collections=gc_collections,
                database_size_mb=db_size,
                cache_hit_rate=cache_hit_rate,
                response_time_ms=response_time_ms,
                request_rate=request_rate,
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return default metrics on error
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                thread_count=0,
                file_descriptors=0,
                gc_objects=0,
                gc_collections=0,
                database_size_mb=0.0,
                cache_hit_rate=0.7,
                response_time_ms=50.0,
                request_rate=1.0,
                error_rate=0.01
            )
    
    def analyze_metrics(self, metrics: SystemMetrics) -> List[OptimizationResult]:
        """Analyze metrics and generate optimization recommendations"""
        results = []
        
        for rule in self.optimization_rules:
            try:
                if rule['condition'](metrics):
                    result = rule['action'](metrics)
                    if result:
                        results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.get('id', 'unknown')}: {e}")
        
        return results
    
    def _optimize_memory_critical(self, metrics: SystemMetrics) -> Optional[OptimizationResult]:
        """Optimize memory usage (critical)"""
        optimizations = []
        
        try:
            # Force garbage collection
            collected = gc.collect()
            optimizations.append(f"Forced GC collected {collected} objects")
            
            # Clear large caches
            try:
                from l104_server.engines_infra import _FAST_REQUEST_CACHE
                old_size = len(_FAST_REQUEST_CACHE)
                # Clear 70% of cache
                keys = list(_FAST_REQUEST_CACHE.keys())
                for key in keys[:int(len(keys) * 0.7)]:
                    _FAST_REQUEST_CACHE.pop(key, None)
                optimizations.append(f"Reduced request cache from {old_size} to {len(_FAST_REQUEST_CACHE)} entries")
            except:
                pass
            
            # Clear memory cache
            try:
                from l104_server.learning.intellect import intellect
                if hasattr(intellect, 'memory_cache'):
                    old_size = len(intellect.memory_cache)
                    # Clear 50% of cache
                    keys = list(intellect.memory_cache.keys())
                    for key in keys[:int(len(keys) * 0.5)]:
                        intellect.memory_cache.pop(key, None)
                    optimizations.append(f"Reduced memory cache from {old_size} to {len(intellect.memory_cache)} entries")
            except:
                pass
            
            # Adjust GC thresholds
            gc.set_threshold(700, 10, 10)
            optimizations.append("Set aggressive GC thresholds")
            
            if optimizations:
                return OptimizationResult(
                    level=OptimizationLevel.CRITICAL,
                    category='memory',
                    description='Critical memory optimization applied',
                    impact_score=80.0,
                    applied=True,
                    details={'optimizations': optimizations}
                )
                
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")
        
        return None
    
    def _optimize_cpu_critical(self, metrics: SystemMetrics) -> Optional[OptimizationResult]:
        """Optimize CPU usage (critical)"""
        optimizations = []
        
        try:
            # Reduce thread pool sizes
            try:
                from l104_server.engines_infra import PERF_THREAD_POOL, IO_THREAD_POOL
                
                if hasattr(PERF_THREAD_POOL, '_max_workers'):
                    old_max = PERF_THREAD_POOL._max_workers
                    new_max = max(2, old_max // 2)
                    PERF_THREAD_POOL._max_workers = new_max
                    optimizations.append(f"Reduced PERF_THREAD_POOL from {old_max} to {new_max} workers")
                
                if hasattr(IO_THREAD_POOL, '_max_workers'):
                    old_max = IO_THREAD_POOL._max_workers
                    new_max = max(1, old_max // 2)
                    IO_THREAD_POOL._max_workers = new_max
                    optimizations.append(f"Reduced IO_THREAD_POOL from {old_max} to {new_max} workers")
            except:
                pass
            
            # Suggest reducing expensive operations
            try:
                from l104_server.learning.intellect import intellect
                if hasattr(intellect, '_quantum_cluster_engine'):
                    optimizations.append("Consider reducing quantum cluster engine frequency")
                if hasattr(intellect, '_neural_resonance_engine'):
                    optimizations.append("Consider reducing neural resonance engine frequency")
            except:
                pass
            
            if optimizations:
                return OptimizationResult(
                    level=OptimizationLevel.CRITICAL,
                    category='cpu',
                    description='Critical CPU optimization applied',
                    impact_score=70.0,
                    applied=True,
                    details={'optimizations': optimizations}
                )
                
        except Exception as e:
            logger.error(f"CPU optimization error: {e}")
        
        return None
    
    def _optimize_response_time(self, metrics: SystemMetrics) -> Optional[OptimizationResult]:
        """Optimize response time"""
        optimizations = []
        
        try:
            # Implement query caching optimization
            try:
                from l104_server.engines_infra import _FAST_REQUEST_CACHE
                # Increase cache size if it's small
                if len(_FAST_REQUEST_CACHE) < 1000:
                    optimizations.append("Consider increasing cache size for better hit rate")
            except:
                pass
            
            # Database optimization
            db_path = "l104_intellect_memory.db"
            if os.path.exists(db_path):
                optimizations.append("Consider database indexing and query optimization")
            
            # Async operation optimization
            optimizations.append("Review async/await patterns for blocking operations")
            
            if optimizations:
                return OptimizationResult(
                    level=OptimizationLevel.HIGH,
                    category='performance',
                    description='Response time optimization recommendations',
                    impact_score=60.0,
                    applied=False,  # Recommendations only
                    details={'optimizations': optimizations}
                )
                
        except Exception as e:
            logger.error(f"Response time optimization error: {e}")
        
        return None
    
    def _optimize_cache(self, metrics: SystemMetrics) -> Optional[OptimizationResult]:
        """Optimize cache performance"""
        optimizations = []
        
        try:
            # Implement cache warming
            try:
                from l104_server.learning.intellect import intellect
                if hasattr(intellect, 'predict_next_queries'):
                    hot_queries = intellect.predict_next_queries("")
                    if hot_queries:
                        optimizations.append(f"Cache warming available for {len(hot_queries)} predicted queries")
            except:
                pass
            
            # Cache eviction policy
            optimizations.append("Implement LRU/TTL cache eviction policies")
            
            # Cache size optimization
            optimizations.append("Monitor and adjust cache sizes based on hit rate")
            
            if optimizations:
                return OptimizationResult(
                    level=OptimizationLevel.HIGH,
                    category='cache',
                    description='Cache optimization recommendations',
                    impact_score=65.0,
                    applied=False,
                    details={'optimizations': optimizations}
                )
                
        except Exception as e:
            logger.error(f"Cache optimization error: {e}")
        
        return None
    
    def _optimize_error_rate(self, metrics: SystemMetrics) -> Optional[OptimizationResult]:
        """Optimize error rate"""
        optimizations = []
        
        try:
            # Error logging and monitoring
            optimizations.append("Implement comprehensive error logging and alerting")
            
            # Retry logic
            optimizations.append("Add retry logic with exponential backoff for transient failures")
            
            # Circuit breaker pattern
            optimizations.append("Implement circuit breaker for external dependencies")
            
            if optimizations:
                return OptimizationResult(
                    level=OptimizationLevel.HIGH,
                    category='reliability',
                    description='Error rate optimization recommendations',
                    impact_score=75.0,
                    applied=False,
                    details={'optimizations': optimizations}
                )
                
        except Exception as e:
            logger.error(f"Error rate optimization error: {e}")
        
        return None
    
    def _optimize_threads(self, metrics: SystemMetrics) -> Optional[OptimizationResult]:
        """Optimize thread usage"""
        optimizations = []
        
        try:
            # Thread pool management
            optimizations.append("Implement thread pool with size limits and monitoring")
            
            # Async conversion
            optimizations.append("Convert blocking I/O to async/await where possible")
            
            # Resource limits
            optimizations.append("Set resource limits for thread creation")
            
            if optimizations:
                return OptimizationResult(
                    level=OptimizationLevel.MEDIUM,
                    category='concurrency',
                    description='Thread optimization recommendations',
                    impact_score=50.0,
                    applied=False,
                    details={'optimizations': optimizations}
                )
                
        except Exception as e:
            logger.error(f"Thread optimization error: {e}")
        
        return None
    
    def _optimize_gc(self, metrics: SystemMetrics) -> Optional[OptimizationResult]:
        """Optimize garbage collection"""
        optimizations = []
        
        try:
            # Object pooling
            optimizations.append("Consider object pooling for frequently created objects")
            
            # Memory profiling
            optimizations.append("Run memory profiler to identify allocation hotspots")
            
            # GC tuning
            optimizations.append("Tune GC thresholds based on application behavior")
            
            if optimizations:
                return OptimizationResult(
                    level=OptimizationLevel.MEDIUM,
                    category='memory',
                    description='Garbage collection optimization recommendations',
                    impact_score=55.0,
                    applied=False,
                    details={'optimizations': optimizations}
                )
                
        except Exception as e:
            logger.error(f"GC optimization error: {e}")
        
        return None
    
    def _optimize_database(self, metrics: SystemMetrics) -> Optional[OptimizationResult]:
        """Optimize database"""
        optimizations = []
        
        try:
            db_path = "l104_intellect_memory.db"
            if os.path.exists(db_path):
                # Vacuum database
                conn = sqlite3.connect(db_path)
                conn.execute("VACUUM")
                conn.close()
                optimizations.append("Database vacuumed to reduce fragmentation")
                
                # Index optimization
                optimizations.append("Review and optimize database indexes")
                
                # Archiving old data
                optimizations.append("Consider archiving old data to reduce size")
                
                if optimizations:
                    return OptimizationResult(
                        level=OptimizationLevel.MEDIUM,
                        category='database',
                        description='Database optimization applied',
                        impact_score=60.0,
                        applied=True,
                        details={'optimizations': optimizations}
                    )
                    
        except Exception as e:
            logger.error(f"Database optimization error: {e}")
        
        return None
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert to file"""
        try:
            alert_file = "optimization_alerts.jsonl"
            with open(alert_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
        except:
            pass
    
    def _console_alert(self, alert: Dict[str, Any]):
        """Print alert to console"""
        level = alert.get('level', 'INFO')
        message = alert.get('message', '')
        print(f"[{level}] {message}")
    
    def trigger_alert(self, level: str, message: str, details: Dict[str, Any] = None):
        """Trigger an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details or {}
        }
        
        self.alert_history.append(alert)
        
        # Call all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Collect metrics
                    metrics = self.collect_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Analyze and apply optimizations
                    results = self.analyze_metrics(metrics)
                    
                    # Process results
                    for result in results:
                        self.optimization_history.append(result)
                        
                        # Trigger alerts for critical/high level optimizations
                        if result.level in [OptimizationLevel.CRITICAL, OptimizationLevel.HIGH]:
                            self.trigger_alert(
                                level=result.level.name,
                                message=f"{result.category}: {result.description}",
                                details=result.details
                            )
                    
                    # Check for metric anomalies
                    self._check_anomalies(metrics)
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    self.trigger_alert('ERROR', f'Monitoring error: {str(e)}')
                
                time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Enhanced optimization monitoring started (interval: {interval_seconds}s)")
    
    def _check_anomalies(self, metrics: SystemMetrics):
        """Check for metric anomalies"""
        # Check for sudden spikes
        if len(self.metrics_history) > 10:
            recent_metrics = list(self.metrics_history)[-10:]
            
            # Check memory spike
            memory_values = [m.memory_mb for m in recent_metrics]
            if len(memory_values) >= 5:
                recent_avg = statistics.mean(memory_values[-5:])
                previous_avg = statistics.mean(memory_values[-10:-5])
                if recent_avg > previous_avg * 1.5:  # 50% increase
                    self.trigger_alert(
                        'WARNING',
                        f'Memory usage spike detected: {previous_avg:.1f}MB -> {recent_avg:.1f}MB',
                        {'metric': 'memory_mb', 'increase_percent': 50}
                    )
            
            # Check CPU spike
            cpu_values = [m.cpu_percent for m in recent_metrics]
            if len(cpu_values) >= 5:
                recent_avg = statistics.mean(cpu_values[-5:])
                previous_avg = statistics.mean(cpu_values[-10:-5])
                if recent_avg > previous_avg * 2.0:  # 100% increase
                    self.trigger_alert(
                        'WARNING',
                        f'CPU usage spike detected: {previous_avg:.1f}% -> {recent_avg:.1f}%',
                        {'metric': 'cpu_percent', 'increase_percent': 100}
                    )
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Enhanced optimization monitoring stopped")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        latest = self.metrics_history[-1] if self.metrics_history else None
        optimizations = list(self.optimization_history)
        
        # Calculate statistics
        if len(self.metrics_history) > 1:
            cpu_values = [m.cpu_percent for m in self.metrics_history]
            memory_values = [m.memory_mb for m in self.metrics_history]
            response_values = [m.response_time_ms for m in self.metrics_history]
            
            stats = {
                'cpu_avg': statistics.mean(cpu_values),
                'cpu_max': max(cpu_values),
                'memory_avg': statistics.mean(memory_values),
                'memory_max': max(memory_values),
                'response_avg': statistics.mean(response_values),
                'response_max': max(response_values),
                'samples': len(self.metrics_history)
            }
        else:
            stats = {}
        
        # Categorize optimizations
        optimization_summary = defaultdict(int)
        for opt in optimizations:
            optimization_summary[opt.level.name] += 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'current_metrics': latest.to_dict() if latest else {},
            'statistics': stats,
            'optimizations_applied': len([o for o in optimizations if o.applied]),
            'optimizations_recommended': len([o for o in optimizations if not o.applied]),
            'optimization_summary': dict(optimization_summary),
            'recent_optimizations': [
                {
                    'level': opt.level.name,
                    'category': opt.category,
                    'description': opt.description,
                    'applied': opt.applied,
                    'timestamp': opt.timestamp.isoformat()
                }
                for opt in optimizations[-10:]
            ],
            'recent_alerts': list(self.alert_history)[-5:],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        latest = self.metrics_history[-1]
        
        # Memory recommendations
        if latest.memory_percent > 70:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Memory',
                'issue': f'High memory usage ({latest.memory_percent:.1f}%)',
                'action': 'Implement memory monitoring and cleanup routines'
            })
        
        # CPU recommendations
        if latest.cpu_percent > 80:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'CPU',
                'issue': f'High CPU usage ({latest.cpu_percent:.1f}%)',
                'action': 'Optimize CPU-bound operations and reduce thread pool sizes'
            })
        
        # Response time recommendations
        if latest.response_time_ms > self.baselines['response_time']:
            recommendations.append({
                'priority': 'MEDIUM',
                'area': 'Performance',
                'issue': f'High response time ({latest.response_time_ms:.1f}ms)',
                'action': 'Optimize database queries and implement caching'
            })
        
        # Cache recommendations
        if latest.cache_hit_rate < self.baselines['cache_hit_rate']:
            recommendations.append({
                'priority': 'MEDIUM',
                'area': 'Cache',
                'issue': f'Low cache hit rate ({latest.cache_hit_rate:.1%})',
                'action': 'Increase cache size and implement cache warming'
            })
        
        if not recommendations:
            recommendations.append({
                'priority': 'INFO',
                'area': 'Overall',
                'issue': 'System performing optimally',
                'action': 'Continue monitoring'
            })
        
        return recommendations
    
    def optimize_all(self) -> Dict[str, Any]:
        """Run all optimizations"""
        metrics = self.collect_metrics()
        results = self.analyze_metrics(metrics)
        
        applied = [r for r in results if r.applied]
        recommended = [r for r in results if not r.applied]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.to_dict(),
            'optimizations_applied': len(applied),
            'optimizations_recommended': len(recommended),
            'applied_optimizations': [
                {
                    'category': r.category,
                    'description': r.description,
                    'impact_score': r.impact_score
                }
                for r in applied
            ],
            'recommended_optimizations': [
                {
                    'category': r.category,
                    'description': r.description,
                    'impact_score': r.impact_score
                }
                for r in recommended
            ]
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='L104 Enhanced Optimization System')
    parser.add_argument('--monitor', action='store_true', help='Start continuous monitoring')
    parser.add_argument('--report', action='store_true', help='Generate optimization report')
    parser.add_argument('--optimize', action='store_true', help='Run all optimizations')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    optimizer = L104EnhancedOptimizer()
    
    if args.monitor:
        try:
            optimizer.start_monitoring(args.interval)
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            optimizer.stop_monitoring()
            print("\nMonitoring stopped by user")
    
    elif args.report:
        report = optimizer.generate_report()
        print(json.dumps(report, indent=2))
    
    elif args.optimize:
        result = optimizer.optimize_all()
        print(json.dumps(result, indent=2))
    
    else:
        # Default: show current metrics
        metrics = optimizer.collect_metrics()
        print("Current System Metrics:")
        print(json.dumps(metrics.to_dict(), indent=2))
        
        # Show any immediate optimizations needed
        results = optimizer.analyze_metrics(metrics)
        if results:
            print("\nOptimizations Needed:")
            for result in results:
                print(f"  [{result.level.name}] {result.category}: {result.description}")

if __name__ == '__main__':
    main()