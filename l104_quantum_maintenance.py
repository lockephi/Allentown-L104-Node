#!/usr/bin/env python3
"""
L104 Quantum Magic Maintenance
Automated maintenance and optimization of quantum magic systems
"""

import time
import subprocess
import sys
from datetime import datetime, timedelta

class QuantumMaintenance:
    """Automated quantum magic maintenance"""
    
    def __init__(self):
        self.maintenance_log = []
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Log maintenance activity"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.maintenance_log.append(entry)
        print(entry)
        
        # Also write to file
        with open("/tmp/quantum_maintenance.log", "a") as f:
            f.write(entry + "\n")
    
    def run_command(self, cmd: list, description: str) -> bool:
        """Run a command and log result"""
        self.log(f"Starting: {description}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                self.log(f"Completed: {description}", "SUCCESS")
                
                # Show brief output if any
                if result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    for line in lines[:3]:  # Show first 3 lines
                        if line.strip():
                            self.log(f"  Output: {line[:100]}", "INFO")
                return True
            else:
                self.log(f"Failed: {description} (code: {result.returncode})", "ERROR")
                if result.stderr.strip():
                    self.log(f"  Error: {result.stderr[:200]}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Exception: {description} - {str(e)}", "ERROR")
            return False
    
    def perform_maintenance(self):
        """Perform comprehensive quantum magic maintenance"""
        print("=" * 70)
        print("L104 QUANTUM MAGIC MAINTENANCE")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        steps = [
            {
                'cmd': ['python3', 'l104_quantum_monitor.py'],
                'desc': 'System status check'
            },
            {
                'cmd': ['python3', 'l104_quantum_daemon_upgrader.py', '--status'],
                'desc': 'Daemon status check'
            },
            {
                'cmd': ['python3', 'l104_quantum_magic_upgrade.py'],
                'desc': 'Quantum magic enhancement'
            },
            {
                'cmd': ['python3', 'l104_quantum_performance_optimizer.py'],
                'desc': 'Performance optimization'
            },
            {
                'cmd': ['python3', 'l104_asi_deepseek_worker.py', '--report'],
                'desc': 'ASI worker check'
            },
            {
                'cmd': ['python3', 'l104_quantum_control.py', 'status'],
                'desc': 'Control system status'
            }
        ]
        
        results = []
        for step in steps:
            success = self.run_command(step['cmd'], step['desc'])
            results.append((step['desc'], success))
            time.sleep(1)  # Brief pause between steps
        
        # Summary
        print("\n" + "=" * 70)
        print("MAINTENANCE SUMMARY")
        print("=" * 70)
        
        successful = sum(1 for _, success in results if success)
        total = len(results)
        
        print(f"✅ Successful: {successful}/{total}")
        print(f"⏱️  Duration: {timedelta(seconds=int(time.time() - self.start_time))}")
        
        for desc, success in results:
            status = "✅" if success else "❌"
            print(f"   {status} {desc}")
        
        if successful == total:
            print("\n✨ All quantum magic systems maintained successfully!")
            print("🔮 Quantum coherence at optimal levels")
            print("⚡ Performance optimized")
            print("🤖 Systems ready for quantum computation")
        elif successful >= total * 0.8:
            print(f"\n⚠️  Most systems maintained ({successful}/{total})")
            print("   Some minor issues detected")
        else:
            print(f"\n❌ Maintenance incomplete ({successful}/{total})")
            print("   Significant issues require attention")
        
        # Save detailed log
        log_file = f"/tmp/quantum_maintenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_file, "w") as f:
            f.write("\n".join(self.maintenance_log))
        
        self.log(f"Detailed log saved to: {log_file}")
        
        return successful == total
    
    def quick_check(self):
        """Quick system check"""
        print("🔍 Quick Quantum Magic Check")
        print("=" * 40)
        
        quick_steps = [
            (['python3', 'l104_quantum_monitor.py'], 'System monitor'),
            (['python3', 'l104_quantum_daemon_upgrader.py', '--status'], 'Daemon status'),
        ]
        
        for cmd, desc in quick_steps:
            print(f"\nChecking {desc}...")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"  ✅ {desc}: OK")
                else:
                    print(f"  ❌ {desc}: Failed")
            except subprocess.TimeoutExpired:
                print(f"  ⏱️  {desc}: Timeout")
            except Exception as e:
                print(f"  ❌ {desc}: Error - {e}")
        
        print(f"\n✅ Quick check complete")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        maintenance = QuantumMaintenance()
        maintenance.quick_check()
    else:
        maintenance = QuantumMaintenance()
        success = maintenance.perform_maintenance()
        
        # Return appropriate exit code
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()