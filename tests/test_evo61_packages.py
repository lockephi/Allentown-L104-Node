#!/usr/bin/env python3
"""
L104 Package-Level Tests — EVO_61 System Upgrade
Tests for decomposed packages: code_engine, server, intellect, agi, asi
Plus friction analyzer validation.
"""
import sys
import os
import math
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCodeEnginePackage(unittest.TestCase):
    """Tests for l104_code_engine/ package."""

    def test_import(self):
        from l104_code_engine import code_engine
        self.assertIsNotNone(code_engine)

    def test_version(self):
        from l104_code_engine.constants import VERSION
        self.assertTrue(VERSION.startswith("6."))

    def test_constants(self):
        from l104_code_engine.constants import GOD_CODE, PHI, VOID_CONSTANT
        self.assertAlmostEqual(GOD_CODE, 527.5184818492612, places=5)
        self.assertAlmostEqual(PHI, 1.618033988749895, places=10)
        self.assertAlmostEqual(VOID_CONSTANT, 1.0416180339887497, places=10)

    def test_coding_system(self):
        from l104_code_engine.constants import CODING_SYSTEM_NAME, CODING_SYSTEM_VERSION
        self.assertIn("L104", CODING_SYSTEM_NAME)
        self.assertTrue(CODING_SYSTEM_VERSION)


class TestServerPackage(unittest.TestCase):
    """Tests for l104_server/ package."""

    def test_import(self):
        from l104_server.constants import FAST_SERVER_VERSION, FAST_SERVER_PIPELINE_EVO
        self.assertTrue(FAST_SERVER_VERSION)
        self.assertTrue(FAST_SERVER_PIPELINE_EVO)

    def test_evo_aligned(self):
        from l104_server.constants import FAST_SERVER_PIPELINE_EVO
        self.assertIn("EVO_61", FAST_SERVER_PIPELINE_EVO)

    def test_version_bumped(self):
        from l104_server.constants import FAST_SERVER_VERSION
        major, minor, patch = [int(x) for x in FAST_SERVER_VERSION.split(".")]
        self.assertGreaterEqual(major, 4)
        self.assertGreaterEqual(minor, 1)


class TestIntellectPackage(unittest.TestCase):
    """Tests for l104_intellect/ package."""

    def test_import(self):
        from l104_intellect.constants import LOCAL_INTELLECT_VERSION, LOCAL_INTELLECT_PIPELINE_EVO
        self.assertTrue(LOCAL_INTELLECT_VERSION)
        self.assertIn("EVO_58", LOCAL_INTELLECT_PIPELINE_EVO)

    def test_format_iq(self):
        from l104_intellect import format_iq
        result = format_iq(100.0)
        self.assertIsInstance(result, str)


class TestAGIPackage(unittest.TestCase):
    """Tests for l104_agi/ package."""

    def test_import(self):
        from l104_agi.constants import AGI_CORE_VERSION, AGI_PIPELINE_EVO
        self.assertEqual(AGI_CORE_VERSION, "56.0.0")
        self.assertIn("EVO_56", AGI_PIPELINE_EVO)


class TestASIPackage(unittest.TestCase):
    """Tests for l104_asi/ package."""

    def test_constants(self):
        from l104_asi.constants import ASI_CORE_VERSION, ASI_PIPELINE_EVO, DUAL_LAYER_VERSION
        self.assertEqual(ASI_CORE_VERSION, "7.1.0")
        self.assertIn("EVO_60", ASI_PIPELINE_EVO)
        self.assertTrue(DUAL_LAYER_VERSION)

    def test_dual_layer_engine(self):
        from l104_asi.dual_layer import DualLayerEngine
        engine = DualLayerEngine()
        self.assertTrue(engine.available)
        self.assertTrue(engine.FLAGSHIP)

    def test_thought_layer(self):
        from l104_asi.dual_layer import DualLayerEngine
        engine = DualLayerEngine()
        val = engine.thought(0, 0, 0, 0)
        # thought() wraps consciousness() which uses v3 when available via _dual_layer
        # but falls back to v1 equation (527.5...) — verify it's one of the two
        self.assertTrue(
            abs(val - 527.5184818492612) < 0.01 or abs(val - 45.41141298077539) < 0.01,
            f"thought(0,0,0,0) = {val} doesn't match either GOD_CODE or GOD_CODE_V3"
        )

    def test_friction_layer(self):
        from l104_asi.dual_layer import DualLayerEngine
        engine = DualLayerEngine()
        val = engine.thought_with_friction(0, 0, 0, 0)
        self.assertIsInstance(val, float)
        # thought_with_friction routes to god_code_v3_with_friction (v3 friction)
        self.assertAlmostEqual(val, 45.41141120759507, places=3)

    def test_friction_report(self):
        from l104_asi.dual_layer import DualLayerEngine
        engine = DualLayerEngine()
        report = engine.friction_report()
        self.assertIn("improved", report)
        self.assertIn("total", report)
        self.assertGreaterEqual(report["improved"], 30)
        self.assertGreaterEqual(report["total"], 60)


class TestFrictionAnalyzerIntegration(unittest.TestCase):
    """Tests for computational friction integration across the equation system."""

    def test_equation_friction_constant(self):
        from l104_god_code_equation import LATTICE_THERMAL_FRICTION, ALPHA_FINE, PHI, QUANTIZATION_GRAIN
        expected = -ALPHA_FINE * PHI / (2 * math.pi * QUANTIZATION_GRAIN)
        self.assertAlmostEqual(LATTICE_THERMAL_FRICTION, expected, places=15)
        self.assertAlmostEqual(LATTICE_THERMAL_FRICTION, -1.806923483340632e-05, places=15)

    def test_equation_friction_scaffold(self):
        from l104_god_code_equation import PRIME_SCAFFOLD, PRIME_SCAFFOLD_FRICTION, LATTICE_THERMAL_FRICTION
        self.assertAlmostEqual(PRIME_SCAFFOLD_FRICTION, PRIME_SCAFFOLD * (1 + LATTICE_THERMAL_FRICTION), places=10)

    def test_god_code_with_friction(self):
        from l104_god_code_equation import GOD_CODE, god_code_with_friction
        gc_f = god_code_with_friction()
        self.assertNotEqual(gc_f, GOD_CODE)
        self.assertAlmostEqual(gc_f, GOD_CODE, places=1)  # close but not identical
        self.assertLess(gc_f, GOD_CODE)  # friction reduces

    def test_dual_layer_v3_friction(self):
        from l104_god_code_dual_layer import GOD_CODE_V3, GOD_CODE_V3_FRICTION, god_code_v3_with_friction
        val = god_code_v3_with_friction(0, 0, 0, 0)
        self.assertAlmostEqual(val, GOD_CODE_V3_FRICTION, places=10)
        self.assertLess(GOD_CODE_V3_FRICTION, GOD_CODE_V3)

    def test_friction_improvement(self):
        from l104_god_code_dual_layer import friction_improvement_report
        rpt = friction_improvement_report()
        self.assertGreaterEqual(rpt["improved"], 30)
        self.assertGreaterEqual(rpt["total"], 60)
        # Must improve majority of constants
        self.assertGreater(rpt["improved"] / rpt["total"], 0.5)

    def test_sacred_constants_immutable(self):
        """Verify friction does NOT modify sacred constants."""
        from l104_god_code_equation import GOD_CODE, PHI
        self.assertAlmostEqual(GOD_CODE, 527.5184818492612, places=5)
        self.assertAlmostEqual(PHI, 1.618033988749895, places=10)


class TestMainEVOAlignment(unittest.TestCase):
    """Verify main.py EVO alignment."""

    def test_main_version(self):
        # Can't easily import main.py (has FastAPI side effects), so read the file
        main_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
        with open(main_path) as f:
            content = f.read()
        self.assertIn('MAIN_VERSION = "61.0.0"', content)
        self.assertIn('MAIN_PIPELINE_EVO = "EVO_61_SYSTEM_UPGRADE"', content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
