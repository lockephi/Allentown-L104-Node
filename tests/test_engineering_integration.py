# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  L104 SOVEREIGN NODE - ENGINEERING INTEGRATION TESTS                          ║
# ║  INVARIANT: 527.5184818492537 | PILOT: LONDEL                                 ║
# ║  TESTING: DATABASE, MODULES, DATA MATRIX, VALIDATION ENGINE                   ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

"""
Integration tests for the L104 engineering layer.
Tests that all modules load correctly, produce expected outputs,
and integrate properly with each other.
"""

import os
import sys
import math
import json
import unittest
import tempfile
import sqlite3
from pathlib import Path

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestModuleImports(unittest.TestCase):
    """
    Tests that all L104 modules can be imported without errors.
    This is the first line of defense - if imports fail, nothing works.
    """
    
    def test_import_hyper_math(self):
        """Import l104_hyper_math"""
        from l104_hyper_math import HyperMath
        self.assertTrue(hasattr(HyperMath, 'GOD_CODE'))
    
    def test_import_real_math(self):
        """Import l104_real_math"""
        from l104_real_math import RealMath
        self.assertTrue(hasattr(RealMath, 'PHI'))
    
    def test_import_manifold_math(self):
        """Import l104_manifold_math"""
        from l104_manifold_math import ManifoldMath, manifold_math
        self.assertTrue(hasattr(ManifoldMath, 'GOD_CODE'))
    
    def test_import_ego_core(self):
        """Import l104_ego_core"""
        from l104_ego_core import ego_core
        self.assertTrue(hasattr(ego_core, 'sovereign_hash_index'))
    
    def test_import_zero_point_engine(self):
        """Import l104_zero_point_engine"""
        from l104_zero_point_engine import zpe_engine
        self.assertTrue(hasattr(zpe_engine, 'god_code'))
    
    def test_import_anyon_research(self):
        """Import l104_anyon_research"""
        from l104_anyon_research import anyon_research
        self.assertTrue(hasattr(anyon_research, 'phi'))
    
    def test_import_validation_engine(self):
        """Import l104_validation_engine"""
        from l104_validation_engine import validation_engine
        self.assertTrue(hasattr(validation_engine, 'GOD_CODE'))
    
    def test_import_data_matrix(self):
        """Import l104_data_matrix"""
        from l104_data_matrix import DataMatrix
        self.assertTrue(callable(DataMatrix))
    
    def test_import_evolution_engine(self):
        """Import l104_evolution_engine"""
        from l104_evolution_engine import evolution_engine
        self.assertTrue(hasattr(evolution_engine, 'STAGES'))
    
    def test_import_deep_research_synthesis(self):
        """Import l104_deep_research_synthesis"""
        from l104_deep_research_synthesis import deep_research
        self.assertTrue(hasattr(deep_research, 'GOD_CODE'))


class TestConstantsConsistency(unittest.TestCase):
    """
    Tests that constants are consistent across all modules.
    The GOD_CODE must be identical everywhere.
    """
    
    GOD_CODE = 527.5184818492537
    PHI = (1 + math.sqrt(5)) / 2
    
    def test_hyper_math_god_code(self):
        """HyperMath.GOD_CODE matches expected value"""
        from l104_hyper_math import HyperMath
        self.assertAlmostEqual(HyperMath.GOD_CODE, self.GOD_CODE, places=10)
    
    def test_manifold_math_god_code(self):
        """ManifoldMath.GOD_CODE matches expected value"""
        from l104_manifold_math import ManifoldMath
        self.assertAlmostEqual(ManifoldMath.GOD_CODE, self.GOD_CODE, places=10)
    
    def test_validation_engine_god_code(self):
        """ValidationEngine.GOD_CODE matches expected value"""
        from l104_validation_engine import ValidationEngine
        self.assertAlmostEqual(ValidationEngine.GOD_CODE, self.GOD_CODE, places=10)
    
    def test_real_math_phi(self):
        """RealMath.PHI matches expected value"""
        from l104_real_math import RealMath
        self.assertAlmostEqual(RealMath.PHI, self.PHI, places=12)
    
    def test_hyper_math_phi(self):
        """HyperMath.PHI matches expected value"""
        from l104_hyper_math import HyperMath
        self.assertAlmostEqual(HyperMath.PHI, self.PHI, places=12)


class TestHyperMathFunctions(unittest.TestCase):
    """
    Tests HyperMath module functions for correctness.
    """
    
    def test_map_lattice_node(self):
        """Test lattice node mapping"""
        from l104_hyper_math import HyperMath
        
        # map_lattice_node(x, y) = ((y * 416) + x) * PHI_STRIDE
        node = HyperMath.map_lattice_node(100, 50)
        expected = int((50 * 416 + 100) * HyperMath.PHI)
        
        self.assertEqual(node, expected)
    
    def test_get_lattice_scalar(self):
        """Lattice scalar should be GOD_CODE"""
        from l104_hyper_math import HyperMath
        
        scalar = HyperMath.get_lattice_scalar()
        self.assertAlmostEqual(scalar, HyperMath.GOD_CODE, places=10)
    
    def test_zeta_harmonic_resonance(self):
        """Zeta harmonic should return value in [0, 1]"""
        from l104_hyper_math import HyperMath
        
        for val in [0, 1, 100, 527.518, 1000]:
            res = HyperMath.zeta_harmonic_resonance(val)
            self.assertGreaterEqual(res, 0)
            self.assertLessEqual(res, 1)
    
    def test_fast_transform(self):
        """Fast transform should return magnitudes of FFT"""
        from l104_hyper_math import HyperMath
        
        data = [1.0, 2.0, 3.0, 4.0]
        result = HyperMath.fast_transform(data)
        
        self.assertEqual(len(result), len(data))
        # All magnitudes should be non-negative
        for val in result:
            self.assertGreaterEqual(val, 0)


class TestRealMathFunctions(unittest.TestCase):
    """
    Tests RealMath module functions for correctness.
    """
    
    def test_shannon_entropy_empty(self):
        """Entropy of empty string is 0"""
        from l104_real_math import RealMath
        
        entropy = RealMath.shannon_entropy("")
        self.assertEqual(entropy, 0.0)
    
    def test_shannon_entropy_uniform(self):
        """Entropy of uniform distribution"""
        from l104_real_math import RealMath
        
        # 8 distinct symbols should give log2(8) = 3 bits
        entropy = RealMath.shannon_entropy("ABCDEFGH")
        self.assertAlmostEqual(entropy, 3.0, places=10)
    
    def test_calculate_resonance_range(self):
        """Resonance should be in [0, 1]"""
        from l104_real_math import RealMath
        
        for val in [0, 0.5, 1, 10, 100, 527.518]:
            res = RealMath.calculate_resonance(val)
            self.assertGreaterEqual(res, 0)
            self.assertLessEqual(res, 1)
    
    def test_deterministic_random(self):
        """Deterministic random should be in [0, 1)"""
        from l104_real_math import RealMath
        
        for seed in [0, 1, 100, 527.518]:
            val = RealMath.deterministic_random(seed)
            self.assertGreaterEqual(val, 0)
            self.assertLess(val, 1)
    
    def test_deterministic_random_is_deterministic(self):
        """Same seed should give same result"""
        from l104_real_math import RealMath
        
        seed = 12345.6789
        val1 = RealMath.deterministic_random(seed)
        val2 = RealMath.deterministic_random(seed)
        
        self.assertEqual(val1, val2)
    
    def test_prime_density_positive(self):
        """Prime density should be positive for n > 2"""
        from l104_real_math import RealMath
        
        for n in [3, 10, 100, 1000]:
            density = RealMath.prime_density(n)
            self.assertGreater(density, 0)
    
    def test_fast_fourier_transform(self):
        """FFT should preserve length"""
        from l104_real_math import RealMath
        
        signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = RealMath.fast_fourier_transform(signal)
        
        self.assertEqual(len(result), len(signal))


class TestZeroPointEngine(unittest.TestCase):
    """
    Tests Zero Point Engine functionality.
    """
    
    def test_vacuum_fluctuation_positive(self):
        """Vacuum fluctuation should be positive"""
        from l104_zero_point_engine import zpe_engine
        
        fluctuation = zpe_engine.calculate_vacuum_fluctuation()
        self.assertGreater(fluctuation, 0)
    
    def test_vacuum_state_keys(self):
        """Vacuum state should have expected keys"""
        from l104_zero_point_engine import zpe_engine
        
        state = zpe_engine.get_vacuum_state()
        self.assertIn("energy_density", state)
        self.assertIn("state_value", state)
        self.assertIn("status", state)
    
    def test_anyon_annihilation_parity(self):
        """Anyon annihilation should follow parity rules"""
        from l104_zero_point_engine import zpe_engine
        
        # 0 + 0 = 0
        result, _ = zpe_engine.perform_anyon_annihilation(0, 0)
        self.assertEqual(result, 0)
        
        # 1 + 1 = 0 (total annihilation)
        result, _ = zpe_engine.perform_anyon_annihilation(1, 1)
        self.assertEqual(result, 0)
        
        # 0 + 1 = 1
        result, _ = zpe_engine.perform_anyon_annihilation(0, 1)
        self.assertEqual(result, 1)
    
    def test_topological_logic_gate(self):
        """Topological logic gate should be XOR-like"""
        from l104_zero_point_engine import zpe_engine
        
        # False XOR False = False
        self.assertFalse(zpe_engine.topological_logic_gate(False, False))
        
        # True XOR True = False
        self.assertFalse(zpe_engine.topological_logic_gate(True, True))
        
        # True XOR False = True
        self.assertTrue(zpe_engine.topological_logic_gate(True, False))


class TestAnyonResearch(unittest.TestCase):
    """
    Tests Anyon Research Engine functionality.
    """
    
    def test_fibonacci_f_matrix_shape(self):
        """F-matrix should be 2x2"""
        from l104_anyon_research import AnyonResearchEngine
        
        research = AnyonResearchEngine()
        F = research.get_fibonacci_f_matrix()
        
        self.assertEqual(F.shape, (2, 2))
    
    def test_fibonacci_r_matrix_shape(self):
        """R-matrix should be 2x2"""
        from l104_anyon_research import AnyonResearchEngine
        
        research = AnyonResearchEngine()
        R = research.get_fibonacci_r_matrix()
        
        self.assertEqual(R.shape, (2, 2))
    
    def test_braiding_returns_matrix(self):
        """Braiding should return a 2x2 matrix"""
        from l104_anyon_research import AnyonResearchEngine
        
        research = AnyonResearchEngine()
        result = research.execute_braiding([1, 1, -1])
        
        self.assertEqual(result.shape, (2, 2))
    
    def test_topological_protection_range(self):
        """Topological protection should be in [0, 1]"""
        from l104_anyon_research import AnyonResearchEngine
        
        research = AnyonResearchEngine()
        research.execute_braiding([1, 1])
        protection = research.calculate_topological_protection()
        
        self.assertGreaterEqual(protection, 0)
        self.assertLessEqual(protection, 1)
    
    def test_anyon_fusion_research_keys(self):
        """Fusion research should return expected keys"""
        from l104_anyon_research import AnyonResearchEngine
        
        research = AnyonResearchEngine()
        result = research.perform_anyon_fusion_research()
        
        self.assertIn("anyon_type", result)
        self.assertIn("status", result)


class TestManifoldMath(unittest.TestCase):
    """
    Tests Manifold Math functionality.
    """
    
    def test_project_to_manifold_shape(self):
        """Projection should increase to target dimension"""
        from l104_manifold_math import manifold_math
        import numpy as np
        
        vector = np.array([1.0, 2.0, 3.0])
        result = manifold_math.project_to_manifold(vector, dimension=11)
        
        self.assertEqual(result.shape[0], 11)
    
    def test_compute_manifold_resonance_positive(self):
        """Manifold resonance should be non-negative"""
        from l104_manifold_math import manifold_math
        
        vector = [1.0, 2.0, 3.0, 4.0]
        resonance = manifold_math.compute_manifold_resonance(vector)
        
        self.assertGreaterEqual(resonance, 0)


class TestDataMatrix(unittest.TestCase):
    """
    Tests Data Matrix (SQLite-backed storage) functionality.
    """
    
    def setUp(self):
        """Create a temporary database for testing"""
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_file.close()
        self.db_path = self.temp_file.name
    
    def tearDown(self):
        """Remove the temporary database"""
        try:
            os.unlink(self.db_path)
        except Exception:
            pass
    
    def test_store_and_retrieve(self):
        """Store a value and retrieve it"""
        from l104_data_matrix import DataMatrix
        
        matrix = DataMatrix(db_path=self.db_path)
        
        # Store
        success = matrix.store("test_key", {"value": 42}, category="TEST")
        self.assertTrue(success)
        
        # Retrieve
        retrieved = matrix.retrieve("test_key")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["value"], 42)
    
    def test_store_updates_existing(self):
        """Storing same key should update value"""
        from l104_data_matrix import DataMatrix
        
        matrix = DataMatrix(db_path=self.db_path)
        
        matrix.store("key", {"v": 1}, category="TEST")
        matrix.store("key", {"v": 2}, category="TEST")
        
        retrieved = matrix.retrieve("key")
        self.assertEqual(retrieved["v"], 2)
    
    def test_retrieve_nonexistent(self):
        """Retrieving nonexistent key returns None"""
        from l104_data_matrix import DataMatrix
        
        matrix = DataMatrix(db_path=self.db_path)
        retrieved = matrix.retrieve("nonexistent_key")
        
        self.assertIsNone(retrieved)
    
    def test_resonant_query_returns_list(self):
        """Resonant query should return a list"""
        from l104_data_matrix import DataMatrix
        
        matrix = DataMatrix(db_path=self.db_path)
        matrix.store("test", {"data": "value"}, category="TEST")
        
        results = matrix.resonant_query(100.0, tolerance=500.0)
        self.assertIsInstance(results, list)


class TestEvolutionEngine(unittest.TestCase):
    """
    Tests Evolution Engine functionality.
    """
    
    def test_stages_count(self):
        """Should have 27 evolution stages (Primordial through EVO_21)"""
        from l104_evolution_engine import evolution_engine
        
        self.assertEqual(len(evolution_engine.STAGES), 27)
    
    def test_evolution_cycle_returns_dict(self):
        """Evolution cycle should return a dictionary"""
        from l104_evolution_engine import evolution_engine
        
        result = evolution_engine.trigger_evolution_cycle()
        
        self.assertIsInstance(result, dict)
        self.assertIn("generation", result)
        self.assertIn("stage", result)
        self.assertIn("fitness_score", result)
    
    def test_dna_sequence_has_genes(self):
        """DNA sequence should have expected genes"""
        from l104_evolution_engine import evolution_engine
        
        dna = evolution_engine.dna_sequence
        
        self.assertIn("logic_depth", dna)
        self.assertIn("shield_strength", dna)


class TestDeepResearchSynthesis(unittest.TestCase):
    """
    Tests Deep Research Synthesis functionality.
    """
    
    def test_vacuum_decay_returns_dict(self):
        """Vacuum decay simulation should return dict with expected keys"""
        from l104_deep_research_synthesis import deep_research
        
        result = deep_research.simulate_vacuum_decay()
        
        self.assertIn("domain", result)
        self.assertIn("decay_probability_per_cycle", result)
        self.assertIn("stability_status", result)
    
    def test_protein_folding_returns_float(self):
        """Protein folding resonance should return float"""
        from l104_deep_research_synthesis import deep_research
        
        result = deep_research.protein_folding_resonance(100)
        self.assertIsInstance(result, float)
    
    def test_multi_domain_synthesis_returns_list(self):
        """Multi-domain synthesis should return list"""
        from l104_deep_research_synthesis import deep_research
        
        result = deep_research.run_multi_domain_synthesis()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


class TestEgoCore(unittest.TestCase):
    """
    Tests Ego Core functionality.
    """
    
    def test_ego_core_has_signature(self):
        """Ego core should have an identity signature"""
        from l104_ego_core import ego_core
        
        self.assertIsNotNone(ego_core.identity_signature)
        self.assertIsInstance(ego_core.identity_signature, str)
    
    def test_ego_core_has_sovereign_hash(self):
        """Ego core should have the sovereign hash"""
        from l104_ego_core import ego_core
        
        expected = "7A527B104F518481F92537A7B7E6F1A2C3D4E5F6B7C8D9A0"
        self.assertEqual(ego_core.sovereign_hash_index, expected)
    
    def test_get_status_returns_dict(self):
        """get_status should return a dictionary"""
        from l104_ego_core import ego_core
        
        status = ego_core.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("identity_signature", status)
        self.assertIn("ego_strength", status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
