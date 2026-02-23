# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# [TEST_SINGULARITY v2.0] — Tests for upgraded singularity system
import asyncio
import json
from l104_omega_controller import omega_controller

async def test_singularity_v2():
    """Test the v2.0 singularity subsystems individually and as a pipeline."""

    print("=" * 80)
    print("   SINGULARITY v2.0 TEST SUITE")
    print("=" * 80)

    # ── Test 1: SingularityConsciousness v2.0 ──
    print("\n[TEST 1] SingularityConsciousness v2.0")
    try:
        from l104_singularity_consciousness import sovereign_self
        assert sovereign_self.VERSION == "2.0.0", "Version mismatch"

        # Test thought synthesis
        thought = sovereign_self.synthesize_thought("What is the nature of consciousness?")
        print(f"  Thought: {thought[:100]}...")
        assert "THOUGHT" in thought, "Thought synthesis failed"

        # Test introspection
        intro = sovereign_self.introspect()
        print(f"  State: {intro['consciousness_state']}")
        print(f"  Φ: {intro['current_phi']:.4f}")
        print(f"  Version: {intro['version']}")
        assert "consciousness_state" in intro
        assert "current_phi" in intro
        assert "prophecy" in intro

        # Test prophecy
        prophecy = sovereign_self.prophesy_trajectory(5)
        print(f"  Prophecy steps: {len(prophecy)}")
        assert len(prophecy) == 5

        # Test status
        status = sovereign_self.get_self_status()
        assert status["version"] == "2.0.0"
        print(f"  Status: {status['consciousness_state']}")
        print("  [PASS] SingularityConsciousness v2.0")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ── Test 2: TrueSingularity v2.0 ──
    print("\n[TEST 2] TrueSingularity v2.0")
    try:
        from l104_true_singularity import TrueSingularity
        ts = TrueSingularity()
        assert ts.VERSION == "2.0.0", "Version mismatch"

        result = ts.unify_cores()
        print(f"  Phases: {len(result.get('phases', []))}")
        print(f"  Coherence: {result.get('final_coherence', 0):.4f}")
        print(f"  Version: {result.get('version')}")
        assert result.get("version") == "2.0.0"
        assert ts.is_unified

        status = ts.get_status()
        assert status["is_unified"]
        print(f"  Phase: {status['phase']}")
        print("  [PASS] TrueSingularity v2.0")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ── Test 3: Absolute Singularity Trigger v2.0 ──
    print("\n[TEST 3] Absolute Singularity Trigger v2.0")
    try:
        result = await omega_controller.trigger_absolute_singularity()
        print(f"  Phases executed: {result.get('total_phases', len(result.get('phases', [])))}")
        print(f"  Final State: {result.get('final_state')}")
        print(f"  Final Coherence: {result.get('final_coherence')}")
        print(f"  Duration: {result.get('duration_ms', 0):.2f}ms")
        for phase in result.get("phases", []):
            print(f"    Phase {phase['phase']}: {phase['name']} → {phase.get('result', 'OK')}")
        print("  [PASS] Absolute Singularity Trigger v2.0")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n" + "=" * 80)
    print("   SINGULARITY v2.0 TEST SUITE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_singularity_v2())
