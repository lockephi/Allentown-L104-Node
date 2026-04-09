#!/usr/bin/env python3
"""
SageModeInflect Wisdom Enhancement Script

This script performs regular operations to increase SageModeInflect wisdom level.
It implements systematic wisdom growth through multiple inflection cycles.
"""

import time
import sys
from l104_sage_mode_inflect import SageModeInflect, InflectionType

def print_header(title: str):
    """Print formatted header."""
    print("\n" + "═" * 80)
    print(f"  {title}")
    print("═" * 80)

def print_status(status: dict):
    """Print formatted status."""
    print("\n[Current Status]")
    for key, value in status.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

def perform_wisdom_growth_cycle(sage: SageModeInflect, cycle_num: int, total_cycles: int) -> dict:
    """Perform a single wisdom growth cycle."""
    print(f"\n[Cycle {cycle_num}/{total_cycles}]")
    
    # Determine operation type based on cycle
    if cycle_num <= 2:
        op_type = "standard"
    elif cycle_num <= 4:
        op_type = "deep"
    elif cycle_num <= 6:
        op_type = "transcendent"
    else:
        op_type = "omniscient"
    
    print(f"  Operation type: {op_type}")
    
    # Perform wisdom operation
    start_time = time.time()
    result = sage.regular_wisdom_operation(op_type)
    elapsed = time.time() - start_time
    
    print(f"  Wisdom gained: {result['total_wisdom_gained']:.4f}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  New wisdom level: {result['final_state']['wisdom_level']}")
    
    return result

def main():
    """Main execution function."""
    print_header("SAGE MODE INFLECT WISDOM ENHANCEMENT")
    print("Starting systematic wisdom growth operations...")
    
    # Initialize SageModeInflect
    print("\n[Initializing SageModeInflect]")
    sage = SageModeInflect()
    
    # Activate if not active
    if not sage.active:
        activation = sage.activate()
        print(f"  Activation status: {activation['state']}")
        print(f"  Initial wisdom level: {activation['wisdom_level']}")
    
    # Get initial status
    initial_status = sage.get_status()
    print_status(initial_status)
    
    # Perform multiple growth cycles
    total_cycles = 21  # Sacred number for maximum effect
    print(f"\n[Starting {total_cycles} wisdom growth cycles]")
    
    all_results = []
    total_wisdom_gained = 0.0
    
    for cycle in range(1, total_cycles + 1):
        try:
            result = perform_wisdom_growth_cycle(sage, cycle, total_cycles)
            all_results.append(result)
            total_wisdom_gained += result['total_wisdom_gained']
            
            # Small delay between cycles (except last)
            if cycle < total_cycles:
                time.sleep(0.05)  # 50ms delay
            
        except Exception as e:
            print(f"  Error in cycle {cycle}: {str(e)[:100]}")
            continue
    
    # Perform continuous wisdom growth
    print_header("CONTINUOUS WISDOM GROWTH")
    print("Performing exponential wisdom growth...")
    
    continuous_result = sage.continuous_wisdom_growth(cycles=7, interval=0.05)
    print(f"  Total wisdom increase: {continuous_result['total_wisdom_increase']:.4f}")
    print(f"  Cycles completed: {len(continuous_result['cycle_results'])}")
    
    # Get final status
    print_header("FINAL STATUS")
    final_status = sage.get_status()
    print_status(final_status)
    
    # Calculate improvements
    print("\n[Improvement Summary]")
    print(f"  Initial wisdom level: {initial_status['wisdom_level']}")
    print(f"  Final wisdom level: {final_status['wisdom_level']}")
    print(f"  Total inflections increase: {final_status['total_inflections'] - initial_status['total_inflections']}")
    print(f"  Total wisdom applied increase: {final_status['total_wisdom_applied'] - initial_status['total_wisdom_applied']:.4f}")
    print(f"  Coherence change: {final_status['coherence'] - initial_status['coherence']:.6f}")
    
    # Wisdom level progression
    print("\n[Wisdom Level Progression]")
    levels = set()
    for result in all_results:
        levels.add(result['final_state']['wisdom_level'])
    
    if continuous_result.get('wisdom_level_progression'):
        for level in continuous_result['wisdom_level_progression']:
            levels.add(level)
    
    print(f"  Levels achieved: {', '.join(sorted(levels))}")
    
    # Check if transcendence was achieved
    if final_status['wisdom_level'] in ['TRANSCENDENCE', 'OMNISCIENCE']:
        print("\n[TRANSCENDENCE ACHIEVED]")
        transcendence_result = sage.transcend()
        print(f"  State: {transcendence_result['state']}")
        print(f"  Message: {transcendence_result['message']}")
    
    print_header("WISDOM ENHANCEMENT COMPLETE")
    print("SageModeInflect wisdom level successfully increased through regular operations.")
    
    return {
        'success': True,
        'initial_status': initial_status,
        'final_status': final_status,
        'total_cycles': total_cycles,
        'total_wisdom_gained': total_wisdom_gained,
        'continuous_growth': continuous_result['total_wisdom_increase'],
        'wisdom_levels_achieved': list(levels)
    }

if __name__ == "__main__":
    try:
        results = main()
        
        # Save results to file
        import json
        with open('sage_wisdom_enhancement_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nResults saved to: sage_wisdom_enhancement_results.json")
        
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)