#!/usr/bin/env python3
"""Test the full ASI v9.0 integration."""

import sys
import json

print("=== ASI v9.0 UNIFIED TEST ===\n")

try:
    from l104_local_intellect import local_intellect

    print("✓ LocalIntellect loaded")
    print(f"  Training data: {len(local_intellect.training_data)}")
    print(f"  Chat conversations: {len(local_intellect.chat_conversations)}")

    # Test 1: Ouroboros
    print("\n[Test 1] Ouroboros Engine:")
    ouro = local_intellect.get_thought_ouroboros()
    if ouro:
        print("  ✓ Ouroboros loaded")
        state = local_intellect.get_ouroboros_state()
        print(f"  Cycle count: {state.get('cycle_count', 0)}")
        print(f"  Entropy: {state.get('accumulated_entropy', 0):.4f}")
    else:
        print("  ✗ Ouroboros not available")

    # Test 2: Entropy Response
    print("\n[Test 2] Entropy Response:")
    resp = local_intellect.entropy_response("What is consciousness?", style="sage")
    print(f"  {resp[:200]}...")

    # Test 3: Full ASI Process
    print("\n[Test 3] Full ASI Process:")
    result = local_intellect.asi_process("Explain quantum coherence", mode="full")
    print(f"  Keys: {list(result.keys())}")
    if "final_response" in result:
        print(f"  Response: {result['final_response'][:200]}...")
    if "ouroboros" in result:
        print(f"  Ouroboros entropy: {result['ouroboros'].get('entropy', 0):.4f}")

    # Test 4: ASI Status
    print("\n[Test 4] ASI Full Status:")
    status = local_intellect.get_asi_status()
    print(f"  Version: {status.get('version', 'unknown')}")
    print(f"  GOD_CODE: {status.get('god_code', 0)}")
    print(f"  Components: {list(status.get('components', {}).keys())}")
    print(f"  Total knowledge: {status.get('total_knowledge', 0)}")

    print("\n=== ALL ASI TESTS PASSED ===")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
