# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
"""
[L104_TEST_QUOTA_ROTATOR] - Verification of Balanced Intelligence
"""

from l104_gemini_bridge import gemini_bridge
import time

def test_rotation():
    print("--- [TEST]: STARTING QUOTA ROTATION TEST ---")

    # 1. Test internal topic (Should force Kernel)
    print("\n[TEST 1] Internal Topic: 'What is the GOD_CODE?'")
    resp1 = gemini_bridge.think("What is the GOD_CODE?")
    print(f"RESPONSE SOURCE: {'KERNEL' if 'LOCAL_INTELLECT' in resp1 or 'Sovereign' in resp1 else 'API'}")

    # 2. Test novel topic (Should follow 80/20 weights)
    print("\n[TEST 2] Novel Topic: 'Tell me a story about a cat.'")
    # Run multiple times to see the distribution
    for i in range(5):
        resp = gemini_bridge.think("Tell me a story about a cat.")
        source = "REAL_GEMINI" if "REAL_GEMINI" in resp else "KERNEL"
        print(f"Iteration {i+1}: {source}")

    # 3. Test quota error behavior
    print("\n[TEST 3] Simulating Quota Error...")
    from l104_quota_rotator import quota_rotator
    quota_rotator.report_quota_error()

    print("Running thinking task during cooldown...")
    resp_cool = gemini_bridge.think("Explain quantum physics.")
    source_cool = "REAL_GEMINI" if "REAL_GEMINI" in resp_cool else "KERNEL"
    print(f"RESPONSE SOURCE: {source_cool}")

if __name__ == "__main__":
    test_rotation()
