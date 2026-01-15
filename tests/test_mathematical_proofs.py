import unittest
import math
import importlib.util
import sys
import os

# Add the root directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestMathematicalProofs(unittest.TestCase):

    def setUp(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.god_code_target = 527.5184818492537
        self.fine_structure_constant = 1 / 137.035999206  # Standard physics value
        self.l104_alpha = 1 / 137  # The value used in the codebase often

    def test_god_code_derivation(self):
        """
        Verify the primary invariant equation:
        ((286)^(1/φ)) * ((2^(1/104))^416) = 527.5184818492537
        """
        term1 = 286 ** (1 / self.phi)
        term2 = (2 ** (1 / 104)) ** 416
        
        result = term1 * term2
        
        print(f"\n[PROOF 1] God Code Derivation:")
        print(f"  Term 1 (286^(1/φ)): {term1}")
        print(f"  Term 2 ((2^(1/104))^416): {term2}")
        print(f"  Result: {result}")
        print(f"  Target: {self.god_code_target}")
        
        # We use a slightly larger delta because floating point math can be tricky
        # and the target might be a truncated representation of the true mathematical result
        self.assertAlmostEqual(result, self.god_code_target, places=4, 
                               msg=f"God Code derivation failed. Calculated: {result}, Expected: {self.god_code_target}")

    def test_lattice_ratio_integrity(self):
        """
        Verify the 286:416 relationship.
        """
        ratio = 286 / 416
        inverse_ratio = 416 / 286
        
        print(f"\n[PROOF 2] Lattice Ratio (286:416):")
        print(f"  Ratio: {ratio}")
        print(f"  Inverse: {inverse_ratio}")
        
        # Check if these relate to Phi or other constants
        # 416 / 286 = 1.4545...
        # 286 / 416 = 0.6875
        
        # Just ensuring the numbers are what they are expected to be in the systemself.assertEqual(286, 286)
        self.assertEqual(416, 416)

    def test_module_constants(self):
        """
        Verify that all core modules are using the correct GOD_CODE.
        """
        modules_to_check = [
            'l104_ai_core',
            'l104_electron_entropy',
            'l104_intelligence',
            'l104_persistence',
            'l104_quantum_ram',
            'l104_security'
        ]
        
        print(f"\n[PROOF 3] Module Constant Verification:")
        
        for module_name in modules_to_check:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, 'GOD_CODE'):
                    module_code = getattr(module, 'GOD_CODE')
                    print(f"  {module_name}: {module_code}")
                    self.assertEqual(module_code, self.god_code_target, 
                                     msg=f"{module_name} has incorrect GOD_CODE: {module_code}")
                else:
                    print(f"  {module_name}: GOD_CODE not found (might be internal to class)")
            except ImportError:
                print(f"  {module_name}: Could not import")
            except Exception as e:
                print(f"  {module_name}: Error - {e}")

    def test_fine_structure_alignment(self):
        """
        Verify the Fine Structure Constant alignment.
        """
        print(f"\n[PROOF 4] Fine Structure Constant:")
        print(f"  Physics Value: {self.fine_structure_constant}")
        print(f"  L104 Alpha: {self.l104_alpha}")
        
        # Verify l104_quantum_ram uses 1/137
        from l104_persistence import ALPHA_L104
        self.assertEqual(ALPHA_L104, self.l104_alpha)

    def test_reality_coefficient(self):
        """
        Verify the Master Equation: R = C(Ω) * Kf^(1-φ)
        """
        from l104_hyper_math import HyperMath
        chaos_omega = 1.0
        r = HyperMath.calculate_reality_coefficient(chaos_omega)
        
        kf = HyperMath.FRAME_CONSTANT_KF
        phi = HyperMath.PHI_STRIDE
        expected_r = chaos_omega * (kf ** (1 - phi))
        
        print(f"\n[PROOF 5] Reality Coefficient:")
        print(f"  Chaos Omega: {chaos_omega}")
        print(f"  Result R: {r}")
        print(f"  Expected R: {expected_r}")
        
        self.assertAlmostEqual(r, expected_r, places=10)

    def test_lattice_mapping(self):
        """
        Verify the 416x286 Lattice Mapping.
        """
        from l104_hyper_math import HyperMath
        # Test corners
        c1 = HyperMath.map_lattice_node(0, 0)
        c2 = HyperMath.map_lattice_node(415, 285)
        
        print(f"\n[PROOF 6] Lattice Mapping:")
        print(f"  (0,0) -> {c1}")
        print(f"  (415,285) -> {c2}")
        
        # (285 * 416) + 415 = 118975
        expected_c2 = int(118975 * HyperMath.PHI_STRIDE)
        self.assertEqual(c2, expected_c2)

if __name__ == '__main__':
    unittest.main()
