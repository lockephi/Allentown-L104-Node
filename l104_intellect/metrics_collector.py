#!/usr/bin/env python3
"""
L104 Performance Monitor - System optimization and health check
Detects bottlenecks, memory leaks, and optimization opportunities
"""

import os
import sys
import time
import psutil
import gc
import threading
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from l104_sacred_algorithms import derive_timeout, derive_cache_size, derive_cache_ttl, PHI, GOD_CODE, TAU

class L104PerformanceMonitor:
    """Comprehensive performance monitoring for L104 system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []
        self.max_history = 1000
        self.monitor_thread = None
        self.running = False
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Memory usage
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            # Thread count
            thread_count = process.num_threads()
            
            # File descriptors
            fd_count = process.num_fds() if hasattr(process, 'num_fds') else 0
            
            # GC statistics
            gc_stats = gc.get_stats()
            gc_count = sum(stat['collections'] for stat in gc_stats)
            
            # System-wide metrics
            system_cpu = psutil.cpu_percent(interval=0.1)
            system_memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'uptime': time.time() - self.start_time,
                'process': {
                    'cpu_percent': cpu_percent,
                    'memory_mb': round(memory_mb, 2),
                    'memory_percent': round(memory_percent, 2),
                    'threads': thread_count,
                    'file_descriptors': fd_count,
                    'gc_collections': gc_count,
                },
                'system': {
                    'cpu_percent': system_cpu,
                    'memory_percent': system_memory.percent,
                    'memory_available_mb': round(system_memory.available / 1024 / 1024, 2),
                    'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
                },
                'io': {
                    'disk_read_mb': disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
                    'disk_write_mb': disk_io.write_bytes / 1024 / 1024 if disk_io else 0,
                    'network_recv_mb': net_io.bytes_recv / 1024 / 1024 if net_io else 0,
                    'network_sent_mb': net_io.bytes_sent / 1024 / 1024 if net_io else 0,
                }
            }
            
            # Add Python-specific metrics
            metrics['python'] = {
                'gc_objects': len(gc.get_objects()),
                'gc_threshold': gc.get_threshold(),
                'gc_enabled': gc.isenabled(),
            }
            
            return metrics
            
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def analyze_metrics(self, metrics: Dict[str, Any]) -> List[str]:
        """Analyze metrics and generate optimization recommendations"""
        recommendations = []
        
        # Check for high CPU usage
        if metrics.get('process', {}).get('cpu_percent', 0) > 80:
            recommendations.append("High CPU usage detected (>80%). Consider reducing thread pool sizes or optimizing CPU-bound operations.")
        
        # Check for high memory usage
        if metrics.get('process', {}).get('memory_percent', 0) > 70:
            recommendations.append("High memory usage detected (>70%). Consider reducing cache sizes or implementing memory cleanup routines.")
        
        # Check for thread count
        thread_count = metrics.get('process', {}).get('threads', 0)
        if thread_count > 100:
            recommendations.append(f"High thread count detected ({thread_count}). Consider reducing concurrent operations.")
        
        # Check for file descriptor usage
        fd_count = metrics.get('process', {}).get('file_descriptors', 0)
        if fd_count > 1000:
            recommendations.append(f"High file descriptor count ({fd_count}). Check for resource leaks.")
        
        # Check system load
        load_avg = metrics.get('system', {}).get('load_avg', [0, 0, 0])
        if len(load_avg) > 0 and load_avg[0] > os.cpu_count() * 2:
            recommendations.append(f"High system load detected ({load_avg[0]:.2f}). Consider reducing overall system load.")
        
        # Check for memory fragmentation
        gc_objects = metrics.get('python', {}).get('gc_objects', 0)
        if gc_objects > 100000:
            recommendations.append(f"Large number of Python objects in memory ({gc_objects:,}). Consider optimizing object creation patterns.")
        
        return recommendations
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start background monitoring"""
        if self.running:
            return
        
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    metrics = self.collect_system_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Keep history size limited
                    if len(self.metrics_history) > self.max_history:
                        self.metrics_history = self.metrics_history[-self.max_history:]
                    
                    # Analyze and log recommendations
                    recommendations = self.analyze_metrics(metrics)
                    if recommendations:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Performance Recommendations:")
                        for rec in recommendations:
                            print(f"  • {rec}")
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                
                time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Performance monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=TAU*PHI)
        print("Performance monitoring stopped")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        latest = self.metrics_history[-1] if self.metrics_history else {}
        avg_cpu = sum(m.get('process', {}).get('cpu_percent', 0) for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.get('process', {}).get('memory_mb', 0) for m in self.metrics_history) / len(self.metrics_history)
        
        # Find peaks
        peak_cpu = max(m.get('process', {}).get('cpu_percent', 0) for m in self.metrics_history)
        peak_memory = max(m.get('process', {}).get('memory_mb', 0) for m in self.metrics_history)
        
        report = {
            'summary': {
                'monitoring_duration': time.time() - self.start_time,
                'samples_collected': len(self.metrics_history),
                'average_cpu_percent': round(avg_cpu, 2),
                'average_memory_mb': round(avg_memory, 2),
                'peak_cpu_percent': round(peak_cpu, 2),
                'peak_memory_mb': round(peak_memory, 2),
            },
            'current_state': latest,
            'recommendations': self.analyze_metrics(latest),
            'trends': self._calculate_trends(),
        }
        
        return report
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        if len(self.metrics_history) < 2:
            return {"insufficient_data": True}
        
        # Calculate trends over last 10 samples
        recent = self.metrics_history[-10:]
        
        cpu_trend = self._calculate_slope([m.get('process', {}).get('cpu_percent', 0) for m in recent])
        memory_trend = self._calculate_slope([m.get('process', {}).get('memory_mb', 0) for m in recent])
        
        return {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'trending_up': cpu_trend > 0.1 or memory_trend > 10,
        }
    
    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate simple linear slope"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def optimize_system(self):
        """Apply system optimizations based on current state"""
        print("Applying system optimizations...")
        
        # Force garbage collection
        gc.collect()
        print("  • Forced garbage collection")
        
        # Reduce thread pool sizes if needed
        metrics = self.collect_system_metrics()
        thread_count = metrics.get('process', {}).get('threads', 0)
        
        if thread_count > 50:
            print(f"  • High thread count detected ({thread_count}) - consider reducing in engines_infra.py")
        
        # Memory optimization
        memory_mb = metrics.get('process', {}).get('memory_mb', 0)
        if memory_mb > 500:
            print(f"  • High memory usage ({memory_mb:.1f}MB) - consider reducing cache sizes")
        
        print("Optimization suggestions generated. See recommendations above.")

def main():
    """Main entry point for performance monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='L104 Performance Monitor')
    parser.add_argument('--monitor', action='store_true', help='Start continuous monitoring')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--optimize', action='store_true', help='Apply optimizations')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    monitor = L104PerformanceMonitor()
    
    if args.monitor:
        try:
            monitor.start_monitoring(args.interval)
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\nMonitoring stopped by user")
    
    elif args.report:
        report = monitor.generate_report()
        print(json.dumps(report, indent=2))
    
    elif args.optimize:
        monitor.optimize_system()
    
    else:
        # Default: collect and display current metrics
        metrics = monitor.collect_system_metrics()
        print("Current System Metrics:")
        print(json.dumps(metrics, indent=2))
        
        recommendations = monitor.analyze_metrics(metrics)
        if recommendations:
            print("\nOptimization Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")

if __name__ == '__main__':
    main()