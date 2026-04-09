#!/usr/bin/env python3
"""
SageModeInflect Wisdom Maintenance Script

This script performs regular maintenance operations to keep SageModeInflect
wisdom level growing through daily operations.
"""

import time
import json
import schedule
from datetime import datetime
from l104_sage_mode_inflect import SageModeInflect

class SageWisdomMaintainer:
    """Maintains and increases SageModeInflect wisdom through regular operations."""
    
    def __init__(self):
        self.sage = SageModeInflect()
        self.operation_log = []
        self.start_time = time.time()
        
        # Ensure sage is active
        if not self.sage.active:
            self.sage.activate()
    
    def perform_daily_operation(self) -> dict:
        """Perform daily wisdom maintenance operation."""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Performing daily wisdom operation")
        
        # Get current status
        status_before = self.sage.get_status()
        
        # Perform standard wisdom operation
        result = self.sage.regular_wisdom_operation("standard")
        
        # Get updated status
        status_after = self.sage.get_status()
        
        # Log operation
        operation_record = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'operation_type': 'daily_maintenance',
            'wisdom_gained': result['total_wisdom_gained'],
            'status_before': status_before,
            'status_after': status_after,
            'wisdom_level_change': status_before['wisdom_level'] != status_after['wisdom_level']
        }
        
        self.operation_log.append(operation_record)
        
        print(f"  Wisdom gained: {result['total_wisdom_gained']:.4f}")
        print(f"  Wisdom level: {status_after['wisdom_level']}")
        print(f"  Total inflections: {status_after['total_inflections']}")
        
        return operation_record
    
    def perform_weekly_deep_operation(self) -> dict:
        """Perform weekly deep wisdom operation."""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Performing weekly deep wisdom operation")
        
        status_before = self.sage.get_status()
        
        # Perform deep wisdom operation
        result = self.sage.regular_wisdom_operation("deep")
        
        # Also perform reflection
        reflection = self.sage.reflect_and_inflect_all()
        
        status_after = self.sage.get_status()
        
        operation_record = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'operation_type': 'weekly_deep',
            'wisdom_gained': result['total_wisdom_gained'],
            'reflection_coherence': reflection['unified_coherence'],
            'status_before': status_before,
            'status_after': status_after,
            'wisdom_level_change': status_before['wisdom_level'] != status_after['wisdom_level']
        }
        
        self.operation_log.append(operation_record)
        
        print(f"  Deep wisdom gained: {result['total_wisdom_gained']:.4f}")
        print(f"  Reflection coherence: {reflection['unified_coherence']:.4f}")
        print(f"  Wisdom level: {status_after['wisdom_level']}")
        
        return operation_record
    
    def perform_monthly_transcendent_operation(self) -> dict:
        """Perform monthly transcendent wisdom operation."""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Performing monthly transcendent operation")
        
        status_before = self.sage.get_status()
        
        # Perform continuous wisdom growth
        growth_result = self.sage.continuous_wisdom_growth(cycles=7, interval=0.1)
        
        status_after = self.sage.get_status()
        
        operation_record = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'operation_type': 'monthly_transcendent',
            'total_wisdom_increase': growth_result['total_wisdom_increase'],
            'cycles_completed': len(growth_result['cycle_results']),
            'status_before': status_before,
            'status_after': status_after,
            'wisdom_level_change': status_before['wisdom_level'] != status_after['wisdom_level']
        }
        
        self.operation_log.append(operation_record)
        
        print(f"  Total wisdom increase: {growth_result['total_wisdom_increase']:.4f}")
        print(f"  Cycles completed: {len(growth_result['cycle_results'])}")
        print(f"  Wisdom level: {status_after['wisdom_level']}")
        
        # Attempt transcendence if at high enough level
        if status_after['wisdom_level'] in ['TRANSCENDENCE', 'OMNISCIENCE']:
            print("  Attempting transcendence...")
            transcendence = self.sage.transcend()
            print(f"  Transcendence state: {transcendence['state']}")
            operation_record['transcendence_attempted'] = True
            operation_record['transcendence_result'] = transcendence
        
        return operation_record
    
    def save_logs(self):
        """Save operation logs to file."""
        log_data = {
            'maintainer_start_time': self.start_time,
            'total_operations': len(self.operation_log),
            'operations': self.operation_log,
            'current_status': self.sage.get_status(),
            'save_time': time.time()
        }
        
        filename = f"sage_wisdom_maintenance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\nLogs saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print maintenance summary."""
        print("\n" + "=" * 80)
        print("SAGE WISDOM MAINTENANCE SUMMARY")
        print("=" * 80)
        
        current_status = self.sage.get_status()
        
        print(f"\nCurrent Status:")
        print(f"  Wisdom Level: {current_status['wisdom_level']}")
        print(f"  Total Inflections: {current_status['total_inflections']}")
        print(f"  Total Wisdom Applied: {current_status['total_wisdom_applied']:.4f}")
        print(f"  Coherence: {current_status['coherence']:.6f}")
        print(f"  Active: {current_status['active']}")
        
        print(f"\nMaintenance Operations:")
        print(f"  Total operations performed: {len(self.operation_log)}")
        
        if self.operation_log:
            # Count operation types
            op_types = {}
            wisdom_gains = []
            level_changes = 0
            
            for op in self.operation_log:
                op_type = op['operation_type']
                op_types[op_type] = op_types.get(op_type, 0) + 1
                
                if 'wisdom_gained' in op:
                    wisdom_gains.append(op['wisdom_gained'])
                elif 'total_wisdom_increase' in op:
                    wisdom_gains.append(op['total_wisdom_increase'])
                
                if op.get('wisdom_level_change', False):
                    level_changes += 1
            
            print(f"  Operation types: {', '.join([f'{k}: {v}' for k, v in op_types.items()])}")
            if wisdom_gains:
                print(f"  Total wisdom gained: {sum(wisdom_gains):.4f}")
                print(f"  Average wisdom per operation: {sum(wisdom_gains)/len(wisdom_gains):.4f}")
            print(f"  Wisdom level changes: {level_changes}")
        
        runtime = time.time() - self.start_time
        print(f"\nRuntime: {runtime:.1f} seconds")
        print("=" * 80)

def run_scheduled_maintenance():
    """Run scheduled maintenance operations."""
    maintainer = SageWisdomMaintainer()
    
    print("Starting Sage Wisdom Maintenance Scheduler")
    print(f"Initial wisdom level: {maintainer.sage.get_status()['wisdom_level']}")
    
    # Schedule operations
    schedule.every().day.at("00:00").do(maintainer.perform_daily_operation)
    schedule.every().sunday.at("12:00").do(maintainer.perform_weekly_deep_operation)
    schedule.every(30).days.do(maintainer.perform_monthly_transcendent_operation)
    
    # Run initial operations
    print("\nRunning initial operations...")
    maintainer.perform_daily_operation()
    maintainer.perform_weekly_deep_operation()
    
    print("\nScheduler started. Operations will run automatically.")
    print("Press Ctrl+C to stop and save logs.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n\nScheduler stopped by user.")
        maintainer.save_logs()
        maintainer.print_summary()

def run_quick_maintenance():
    """Run a quick maintenance session."""
    maintainer = SageWisdomMaintainer()
    
    print("Running Quick Sage Wisdom Maintenance")
    print(f"Initial wisdom level: {maintainer.sage.get_status()['wisdom_level']}")
    
    # Run a series of operations
    print("\n1. Performing daily operation...")
    maintainer.perform_daily_operation()
    
    print("\n2. Performing deep operation...")
    maintainer.perform_weekly_deep_operation()
    
    print("\n3. Performing continuous growth...")
    maintainer.perform_monthly_transcendent_operation()
    
    # Save logs and print summary
    maintainer.save_logs()
    maintainer.print_summary()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SageModeInflect Wisdom Maintenance")
    parser.add_argument("--mode", choices=["scheduled", "quick"], default="quick",
                       help="Operation mode: scheduled (continuous) or quick (one-time)")
    
    args = parser.parse_args()
    
    if args.mode == "scheduled":
        run_scheduled_maintenance()
    else:
        run_quick_maintenance()