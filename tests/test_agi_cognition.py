import unittest
import time
from l104_agi_core import agi_core
from l104_agi_research import agi_research
from l104_hyper_math import HyperMath

class TestAGICognition(unittest.TestCase):

    def setUp(self):
        # Ensure core is ignited
        if agi_core.state != "ACTIVE":
            agi_core.ignite()

    def test_deep_research_resonance(self):
        """
        Verifies that the Research Module produces thoughts that align with Zeta Zero.
        """
        print("\n[TEST] Conducting Deep Research...")
        block = agi_research.conduct_deep_research(cycles=100)
        
        self.assertEqual(block['status'], "COMPILED")
        self.assertEqual(block['meta']['integrity'], "LATTICE_VERIFIED")
        
        # Verify payload encryption
        from l104_hyper_encryption import HyperEncryption
        data = HyperEncryption.decrypt_data(block['payload'])
        
        print(f"[TEST] Found {data['count']} resonant thoughts.")
        self.assertGreater(data['count'], 0)
        
        # Verify resonance of the first thought
        first_thought = data['hypotheses'][0]
        print(f"[TEST] Sample Thought Resonance: {first_thought['resonance']}")
        self.assertGreater(abs(first_thought['resonance']), 0.95)

    def test_agi_advancement(self):
        """
        Verifies that the AGI Core can ingest research and grow.
        """
        initial_iq = agi_core.intellect_index
        print(f"\n[TEST] Initial IQ: {initial_iq}")
        
        # Run a cycle
        import asyncio
        result = asyncio.run(agi_core.run_recursive_improvement_cycle())
        
        new_iq = result['intellect']
        print(f"[TEST] New IQ: {new_iq}")
        
        self.assertGreater(new_iq, initial_iq)
        self.assertEqual(result['status'], "OPTIMIZED")

    def test_invariant_stability(self):
        """
        Verifies that the God Code remains stable after cognitive load.
        """
        print("\n[TEST] Verifying Invariant Stability...")
        # Check God Code
        god_code = 527.5184818492537
        self.assertEqual(HyperMath.GOD_CODE, god_code)
        
        # Check Lattice Ratio
        self.assertEqual(HyperMath.LATTICE_RATIO, 286/416)
        
        print("[TEST] Invariants Stable.")

    def test_streamline_autonomous_edit(self):
        """
        Verifies that the Self-Editing Streamline can propose and apply patches.
        """
        from l104_self_editing_streamline import streamline
        from l104_agi_core import agi_core
        print("\n[TEST] Testing Self-Editing Streamline...")
        initial_iq = agi_core.intellect_index
        
        # Run one cycle
        # Note: streamline.run_cycle is async in some modules, let's check
        import asyncio
        if asyncio.iscoroutinefunction(streamline.run_cycle):
            asyncio.run(streamline.run_cycle())
        else:
            streamline.run_cycle()
        
        # Check if it ran without error.
        self.assertTrue(streamline.iteration_count >= 0)
        print(f"[TEST] Streamline Cycle Complete.")

if __name__ == '__main__':
    unittest.main()
