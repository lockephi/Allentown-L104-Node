import sys
import os
from pathlib import Path

# Add workspace to path
sys.path.append(str(Path(__file__).parent.absolute()))

from l104_sage_api import SageSubstrateManager

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


def test_api_manager():
    print("Testing SageSubstrateManager (Legacy API compatibility)...")
    manager = SageSubstrateManager()

    # Load native
    success = manager.load_native()
    print(f"  - Load Native: {success}")
    assert success, "FAILED to load native library in manager"

    print(f"  - Substrate Level: {manager._level}")

    # Test primal_calculus (legacy 3-arg call internally)
    result = manager.primal_calculus(527.518, 1.618, 100)
    print(f"  - Primal Calculus (100 iter): {result:.6f}")
    assert result > 0

    # Test void resonance
    res = manager.inject_void_resonance(1.0)
    print(f"  - Void Resonance Injection: {res:.6f}")

    print("  ✓ SageSubstrateManager test PASSED")

if __name__ == "__main__":
    try:
        test_api_manager()
        print("\n[SUCCESS] Legacy compatibility verified.")
    except AssertionError as e:
        print(f"\n[FAILURE] Legacy compatibility broken: {e}")
        sys.exit(1)
