#!/usr/bin/env python3
"""
Advanced resource monitoring and throttling for L104 daemons.
Provides load-aware backoff, dynamic sleep intervals, and CPU/memory throttling.
"""

import os
import sys
import time
import logging
import psutil
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Current system resource metrics."""
    load_avg_1min: float
    load_avg_5min: float
    load_avg_15min: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    process_cpu_percent: Optional[float] = None
    process_memory_mb: Optional[float] = None

class ResourceMonitor:
    """Monitor system resources and apply throttling policies."""
    
    def __init__(self, 
                 max_cpu_percent: float = 70.0,
                 max_memory_percent: float = 85.0,
                 max_load_1min: float = 4.0,
                 backoff_factor: float = 2.0,
                 max_backoff_seconds: float = 300.0):
        """
        Args:
            max_cpu_percent: CPU usage threshold (percentage) above which throttling kicks in.
            max_memory_percent: Memory usage threshold.
            max_load_1min: 1‑minute load average threshold (per core).
            backoff_factor: Multiplicative factor for exponential backoff.
            max_backoff_seconds: Maximum backoff interval.
        """
        self.max_cpu = max_cpu_percent
        self.max_memory = max_memory_percent
        self.max_load = max_load_1min
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff_seconds
        
        # Get number of CPU cores for load normalization
        self.cores = psutil.cpu_count()
        logger.info(f"ResourceMonitor initialized with {self.cores} cores, thresholds: CPU {self.max_cpu}%, Memory {self.max_memory}%, Load {self.max_load}")
    
    def get_metrics(self, pid: Optional[int] = None) -> SystemMetrics:
        """Collect current system and optionally process‑specific metrics."""
        load = psutil.getloadavg()
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        
        process_cpu = None
        process_mem = None
        if pid:
            try:
                proc = psutil.Process(pid)
                process_cpu = proc.cpu_percent(interval=0.1)
                process_mem = proc.memory_info().rss / 1024 / 1024  # MB
            except psutil.NoSuchProcess:
                pass
        
        return SystemMetrics(
            load_avg_1min=load[0],
            load_avg_5min=load[1],
            load_avg_15min=load[2],
            cpu_percent=cpu,
            memory_percent=mem.percent,
            memory_available_gb=mem.available / 1024**3,
            process_cpu_percent=process_cpu,
            process_memory_mb=process_mem
        )
    
    def should_throttle(self, metrics: SystemMetrics) -> Tuple[bool, str]:
        """
        Determine whether throttling should be applied.
        Returns (should_throttle, reason).
        """
        # Normalize load average by cores
        load_per_core = metrics.load_avg_1min / self.cores if self.cores else metrics.load_avg_1min
        
        reasons = []
        if metrics.cpu_percent > self.max_cpu:
            reasons.append(f"CPU {metrics.cpu_percent:.1f}% > {self.max_cpu}%")
        if metrics.memory_percent > self.max_memory:
            reasons.append(f"Memory {metrics.memory_percent:.1f}% > {self.max_memory}%")
        if load_per_core > self.max_load:
            reasons.append(f"Load {load_per_core:.2f} > {self.max_load}")
        
        if reasons:
            return True, "; ".join(reasons)
        return False, ""
    
    def adaptive_sleep(self, 
                       base_interval: float,
                       pid: Optional[int] = None,
                       backoff_counter: int = 0) -> float:
        """
        Sleep for a dynamic interval based on system load.
        If system is overloaded, increase sleep time using exponential backoff.
        Returns the actual sleep time used.
        """
        metrics = self.get_metrics(pid)
        throttle, reason = self.should_throttle(metrics)
        
        # Calculate sleep duration
        if throttle:
            # Exponential backoff capped at max_backoff
            backoff_seconds = min(
                base_interval * (self.backoff_factor ** backoff_counter),
                self.max_backoff
            )
            sleep_time = backoff_seconds
            logger.warning(f"System overload detected: {reason}. Backoff level {backoff_counter}, sleeping {sleep_time:.1f}s")
        else:
            sleep_time = base_interval
        
        # Sleep in increments to allow early wake‑up if overload subsides
        sleep_step = min(sleep_time, 10.0)  # check every 10 seconds max
        remaining = sleep_time
        while remaining > 0:
            step = min(sleep_step, remaining)
            time.sleep(step)
            remaining -= step
            
            # Re‑evaluate after each step
            if remaining > 0:
                metrics = self.get_metrics(pid)
                throttle, _ = self.should_throttle(metrics)
                if not throttle:
                    logger.info(f"Overload cleared, resuming early (saved {remaining:.1f}s)")
                    break
        
        return sleep_time - remaining  # actual sleep time
    
    def throttled_loop(self,
                       base_interval: float,
                       loop_func: Callable[[], None],
                       pid: Optional[int] = None,
                       loop_name: str = "loop"):
        """
        Run `loop_func` in an infinite loop with adaptive throttling.
        """
        backoff_counter = 0
        while True:
            start_time = time.time()
            
            try:
                loop_func()
            except Exception as e:
                logger.error(f"Error in {loop_name}: {e}", exc_info=True)
            
            # Calculate elapsed and remaining sleep
            elapsed = time.time() - start_time
            sleep_needed = max(0.0, base_interval - elapsed)
            
            # Apply adaptive sleep
            actual_sleep = self.adaptive_sleep(sleep_needed, pid, backoff_counter)
            
            # Adjust backoff counter: increase if throttled, decrease if not
            metrics = self.get_metrics(pid)
            throttle, _ = self.should_throttle(metrics)
            if throttle:
                backoff_counter = min(backoff_counter + 1, 10)  # cap at 10
            elif backoff_counter > 0:
                backoff_counter = max(0, backoff_counter - 1)
            
            logger.debug(f"{loop_name}: loop took {elapsed:.1f}s, slept {actual_sleep:.1f}s, backoff={backoff_counter}")

# Global instance with default thresholds
default_monitor = ResourceMonitor()

if __name__ == "__main__":
    # Quick test
    monitor = ResourceMonitor()
    metrics = monitor.get_metrics()
    print(f"Load: {metrics.load_avg_1min:.2f}, CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%")
    throttle, reason = monitor.should_throttle(metrics)
    print(f"Should throttle: {throttle}, Reason: {reason}")
    print(f"Sleeping with adaptive backoff...")
    monitor.adaptive_sleep(5.0)