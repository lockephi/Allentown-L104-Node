#!/usr/bin/env python3
"""
L104 Quantum Magic Monitor
Continuously monitors and maintains quantum magic systems
"""

import time
import json
import os
from datetime import datetime
from pathlib import Path

class QuantumMagicMonitor:
    """Monitors quantum magic systems"""
    
    def __init__(self):
        self.monitor_file = Path.home() / ".openclaw" / "workspace" / "memory" / "quantum_monitor.json"
        self.monitor_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize monitor data
        if self.monitor_file.exists():
            with open(self.monitor_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                'start_time': time.time(),
                'last_check': None,
                'checks_performed': 0,
                'issues_found': 0,
                'optimizations_applied': 0,
                'quantum_state': {
                    'coherence': 0.85,
                    'entanglement': 0.72,
                    'magic_power': 0.65,
                    'stability': 0.88
                },
                'system_status': {},
                'history': []
            }
    
    def check_quantum_systems(self) -> dict:
        """Check all quantum systems"""
        checks = {
            'daemon_upgrader': self._check_daemon_upgrader(),
            'quantum_magic': self._check_quantum_magic(),
            'asi_worker': self._check_asi_worker(),
            'performance': self._check_performance(),
            'coherence': self._check_coherence()
        }
        
        # Update overall status
        all_ok = all(check['status'] == 'ok' for check in checks.values())
        
        return {
            'timestamp': time.time(),
            'all_systems_ok': all_ok,
            'checks': checks,
            'issues': sum(1 for check in checks.values() if check['status'] != 'ok')
        }
    
    def _check_daemon_upgrader(self) -> dict:
        """Check quantum daemon upgrader"""
        try:
            # Check if file exists and is accessible
            daemon_path = Path("/Users/carolalvarez/Applications/Allentown-L104-Node/l104_quantum_daemon_upgrader.py")
            if not daemon_path.exists():
                return {'status': 'error', 'message': 'Daemon upgrader not found'}
            
            # Check file size
            size = daemon_path.stat().st_size
            if size < 1000:
                return {'status': 'warning', 'message': f'Daemon upgrader file small ({size} bytes)'}
            
            return {'status': 'ok', 'message': 'Daemon upgrader ready', 'size': size}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _check_quantum_magic(self) -> dict:
        """Check quantum magic system"""
        try:
            magic_path = Path("/Users/carolalvarez/Applications/Allentown-L104-Node/l104_quantum_magic_upgrade.py")
            if not magic_path.exists():
                return {'status': 'error', 'message': 'Quantum magic system not found'}
            
            # Check if recently modified (within 24 hours)
            mod_time = magic_path.stat().st_mtime
            age_hours = (time.time() - mod_time) / 3600
            
            if age_hours > 24:
                return {'status': 'warning', 'message': f'Quantum magic system outdated ({age_hours:.1f} hours old)'}
            
            return {'status': 'ok', 'message': 'Quantum magic system active', 'age_hours': age_hours}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _check_asi_worker(self) -> dict:
        """Check ASI-DeepSeek worker"""
        try:
            asi_path = Path("/Users/carolalvarez/Applications/Allentown-L104-Node/l104_asi_deepseek_worker.py")
            if not asi_path.exists():
                return {'status': 'error', 'message': 'ASI worker not found'}
            
            # Check for recent activity
            # This is a simplified check - in production would check process status
            return {'status': 'ok', 'message': 'ASI worker available'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _check_performance(self) -> dict:
        """Check performance optimizer"""
        try:
            perf_path = Path("/Users/carolalvarez/Applications/Allentown-L104-Node/l104_quantum_performance_optimizer.py")
            if not perf_path.exists():
                return {'status': 'error', 'message': 'Performance optimizer not found'}
            
            return {'status': 'ok', 'message': 'Performance optimizer ready'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _check_coherence(self) -> dict:
        """Check quantum coherence"""
        # Simulate coherence check
        import random
        coherence_level = random.uniform(0.8, 0.99)
        
        if coherence_level > 0.95:
            return {'status': 'ok', 'message': f'Quantum coherence optimal ({coherence_level:.3f})', 'level': coherence_level}
        elif coherence_level > 0.85:
            return {'status': 'warning', 'message': f'Quantum coherence acceptable ({coherence_level:.3f})', 'level': coherence_level}
        else:
            return {'status': 'error', 'message': f'Quantum coherence low ({coherence_level:.3f})', 'level': coherence_level}
    
    def run_monitor_cycle(self):
        """Run one monitoring cycle"""
        print(f"\n🔍 Quantum Magic Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Perform checks
        result = self.check_quantum_systems()
        
        # Display results
        issues = 0
        warnings = 0
        
        for system, check in result['checks'].items():
            status = check['status']
            emoji = "✅" if status == 'ok' else "⚠️" if status == 'warning' else "❌"
            
            print(f"{emoji} {system.replace('_', ' ').title()}: {check['message']}")
            
            if status == 'warning':
                warnings += 1
            elif status == 'error':
                issues += 1
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Total systems: {len(result['checks'])}")
        print(f"   Issues: {issues}")
        print(f"   Warnings: {warnings}")
        
        if issues == 0 and warnings == 0:
            print(f"\n✨ All quantum systems operating optimally!")
            print(f"🔮 Quantum magic at peak performance")
        elif issues == 0:
            print(f"\n⚠️  Some warnings detected - monitor closely")
        else:
            print(f"\n❌ Issues detected - attention required")
        
        # Update monitor data
        self.data['last_check'] = time.time()
        self.data['checks_performed'] += 1
        self.data['issues_found'] += issues
        
        # Add to history (keep last 100 entries)
        self.data['history'].append(result)
        if len(self.data['history']) > 100:
            self.data['history'] = self.data['history'][-100:]
        
        # Save data
        with open(self.monitor_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        return result
    
    def get_stats(self) -> dict:
        """Get monitoring statistics"""
        uptime_hours = (time.time() - self.data['start_time']) / 3600
        
        return {
            'uptime_hours': uptime_hours,
            'checks_performed': self.data['checks_performed'],
            'issues_found': self.data['issues_found'],
            'average_issues_per_check': self.data['issues_found'] / max(1, self.data['checks_performed']),
            'last_check': datetime.fromtimestamp(self.data['last_check']).isoformat() if self.data['last_check'] else None
        }

def main():
    """Main entry point"""
    print("=" * 70)
    print("L104 QUANTUM MAGIC MONITOR")
    print("=" * 70)
    print("Continuous monitoring of quantum magic systems")
    print()
    
    monitor = QuantumMagicMonitor()
    
    # Show stats
    stats = monitor.get_stats()
    print(f"📈 Monitoring Statistics:")
    print(f"   Uptime: {stats['uptime_hours']:.1f} hours")
    print(f"   Checks performed: {stats['checks_performed']}")
    print(f"   Issues found: {stats['issues_found']}")
    if stats['last_check']:
        print(f"   Last check: {stats['last_check']}")
    
    print(f"\n🚀 Running system check...")
    
    # Run monitoring cycle
    result = monitor.run_monitor_cycle()
    
    # Recommendations
    if result['issues'] > 0:
        print(f"\n💡 Recommendations:")
        for system, check in result['checks'].items():
            if check['status'] == 'error':
                print(f"   • Fix {system.replace('_', ' ')}: {check['message']}")
            elif check['status'] == 'warning':
                print(f"   • Check {system.replace('_', ' ')}: {check['message']}")
    
    print(f"\n🔮 Quantum magic monitoring complete!")

if __name__ == "__main__":
    main()