#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492612
"""
Memory and Performance Monitor
Provides real-time system resource monitoring and optimization suggestions
"""

import psutil
import os
import sys
from datetime import datetime

def get_size(bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0

def get_memory_info():
    """Get detailed memory information"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print("\n" + "="*60)
    print("MEMORY USAGE")
    print("="*60)
    print(f"Total:     {get_size(mem.total)}")
    print(f"Available: {get_size(mem.available)}")
    print(f"Used:      {get_size(mem.used)} ({mem.percent}%)")
    print(f"Free:      {get_size(mem.free)}")

    if swap.total > 0:
        print(f"\nSwap Total: {get_size(swap.total)}")
        print(f"Swap Used:  {get_size(swap.used)} ({swap.percent}%)")
    else:
        print("\n⚠️  No swap space configured!")

def get_disk_info():
    """Get disk usage information"""
    print("\n" + "="*60)
    print("DISK USAGE")
    print("="*60)

    disk = psutil.disk_usage('/Users')
    print(f"Total:     {get_size(disk.total)}")
    print(f"Used:      {get_size(disk.used)} ({disk.percent}%)")
    print(f"Free:      {get_size(disk.free)}")

    if disk.percent > 95:
        print("\n🚨 CRITICAL: Disk usage above 95%!")
        print("   Consider running: ./optimize_system.sh")

def get_cpu_info():
    """Get CPU information"""
    print("\n" + "="*60)
    print("CPU USAGE")
    print("="*60)
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")

    per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    for i, percent in enumerate(per_cpu):
        print(f"  Core {i}: {percent}%")

def get_top_processes(limit=10):
    """Get top memory consuming processes"""
    print("\n" + "="*60)
    print(f"TOP {limit} MEMORY CONSUMERS")
    print("="*60)

    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    processes.sort(key=lambda x: x['memory_percent'], reverse=True)

    print(f"{'PID':<8} {'Memory%':<10} {'CPU%':<8} {'Name':<40}")
    print("-" * 66)

    for proc in processes[:limit]:
        print(f"{proc['pid']:<8} {proc['memory_percent']:<10.2f} "
              f"{proc['cpu_percent']:<8.1f} {proc['name'][:38]:<40}")

def get_optimization_suggestions():
    """Provide optimization suggestions"""
    print("\n" + "="*60)
    print("OPTIMIZATION SUGGESTIONS")
    print("="*60)

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/Users')

    suggestions = []

    if mem.percent > 80:
        suggestions.append("⚠️  Memory usage > 80%: Consider closing unused applications")

    if disk.percent > 90:
        suggestions.append("🚨 Disk usage > 90%: Run ./optimize_system.sh immediately")
    elif disk.percent > 80:
        suggestions.append("⚠️  Disk usage > 80%: Clean up unnecessary files")

    if psutil.swap_memory().total == 0:
        suggestions.append("ℹ️  No swap configured: System may be unstable under high load")

    # Check for multiple Java processes
    java_procs = [p for p in psutil.process_iter(['name'])
                  if 'java' in p.info['name'].lower()]
    if len(java_procs) > 3:
        suggestions.append(f"⚠️  {len(java_procs)} Java processes running: Consider reducing language servers")

    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    else:
        print("✅ System is running optimally!")

def main():
    """Main monitoring function"""
    print("\n" + "="*60)
    print("SYSTEM RESOURCE MONITOR")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    get_memory_info()
    get_disk_info()
    get_cpu_info()
    get_top_processes()
    get_optimization_suggestions()

    print("\n" + "="*60)
    print("Monitoring complete. Run with --watch for continuous monitoring")
    print("="*60 + "\n")

if __name__ == "__main__":
    if "--watch" in sys.argv:
        import time
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            main()
            print("Refreshing in 0.5 seconds... (Ctrl+C to stop)")
            time.sleep(0.5)  # QUANTUM AMPLIFIED
    else:
        main()
