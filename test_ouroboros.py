#!/usr/bin/env python3
"""Test the Thought Entropy Ouroboros system."""

import sys
import json

print("=== OUROBOROS TEST ===")

try:
    from l104_thought_entropy_ouroboros import get_thought_ouroboros

    o = get_thought_ouroboros()
    print("✓ Ouroboros loaded")

    # Test 1: Basic process
    r = o.process("What is consciousness?", depth=2)
    print(f"\n[Test 1] Basic Process:")
    print(f"  Entropy: {r['accumulated_entropy']:.4f}")
    print(f"  Mutations: {r['total_mutations']}")
    print(f"  Cycles: {r['cycles_completed']}")
    print(f"  Response: {r['final_response'][:150]}...")

    # Test 2: Entropy response
    sage = o.generate_entropy_response("What is love?", style="sage")
    print(f"\n[Test 2] Sage Response:")
    print(f"  {sage[:200]}...")

    # Test 3: State
    state = o.get_ouroboros_state()
    print(f"\n[Test 3] Ouroboros State:")
    print(f"  Cycles: {state['cycle_count']}")
    print(f"  Total processed: {state['total_thoughts_processed']}")
    print(f"  Accumulated entropy: {state['accumulated_entropy']:.4f}")
    print(f"  Status: {state['status']}")

    print("\n=== ALL TESTS PASSED ===")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
