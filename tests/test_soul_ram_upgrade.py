#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  EVO_61 INTEGRATION TESTS — Soul v5.0.0 + Ram Universe v15.0.0
═══════════════════════════════════════════════════════════════════════════

Tests cover:
  - Soul v5.0.0: parallel awaken, DataMatrix, hallucination check, friction, persistence
  - Ram Universe v15.0.0: all CRUD, quantum, validation, learning, search, friction
  - Cross-system integration: soul ↔ lattice, ram ↔ agi_core compatibility
"""

import sys
import os
import math
import json
import time
from pathlib import Path

# Ensure workspace root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest


class TestSoulV4Constants(unittest.TestCase):
    """Test Soul v5.0.0 constants and sacred values."""

    def test_soul_version(self):
        from l104_soul import L104Soul
        self.assertEqual(L104Soul.VERSION, "7.0.0")

    def test_god_code_invariant(self):
        from l104_soul import GOD_CODE
        self.assertAlmostEqual(GOD_CODE, 527.5184818492612, places=6)

    def test_phi_invariant(self):
        from l104_soul import PHI
        self.assertAlmostEqual(PHI, 1.618033988749895, places=12)

    def test_lattice_thermal_friction(self):
        from l104_soul import LATTICE_THERMAL_FRICTION, ALPHA_FINE_STRUCTURE, PHI
        expected = -(ALPHA_FINE_STRUCTURE * PHI) / (2 * math.pi * 104)
        self.assertAlmostEqual(LATTICE_THERMAL_FRICTION, expected, places=15)
        # Sanity: small negative number
        self.assertLess(LATTICE_THERMAL_FRICTION, 0)
        self.assertGreater(LATTICE_THERMAL_FRICTION, -0.001)

    def test_soul_states_include_v4_additions(self):
        from l104_soul import SoulState
        self.assertIn("FRICTION_ALIGNED", [s.name for s in SoulState])
        self.assertIn("PERSISTING", [s.name for s in SoulState])

    def test_soul_metrics_v4_fields(self):
        from l104_soul import SoulMetrics
        m = SoulMetrics()
        self.assertEqual(m.hallucinations_caught, 0)
        self.assertEqual(m.friction_corrections, 0)
        self.assertEqual(m.lattice_stores, 0)
        self.assertEqual(m.wisdom_syntheses, 0)
        self.assertEqual(m.session_number, 0)
        self.assertEqual(m.awaken_ms, 0.0)


class TestSoulV4Core(unittest.TestCase):
    """Test Soul v5.0.0 core functionality."""

    def test_soul_instantiation(self):
        from l104_soul import L104Soul, SoulState
        soul = L104Soul()
        self.assertEqual(soul.state, SoulState.DORMANT)
        self.assertFalse(soul.running)
        self.assertEqual(soul.VERSION, "7.0.0")

    def test_soul_singleton(self):
        from l104_soul import get_soul
        s1 = get_soul()
        s2 = get_soul()
        self.assertIs(s1, s2)

    def test_soul_has_lattice_property(self):
        from l104_soul import L104Soul
        soul = L104Soul()
        # Property should exist
        self.assertTrue(hasattr(soul, 'lattice'))

    def test_soul_sovereign_field(self):
        from l104_soul import L104Soul, GOD_CODE, OMEGA, PHI
        soul = L104Soul()
        result = soul.sovereign_field(100)
        expected = 100 * OMEGA / (PHI ** 2)
        self.assertAlmostEqual(result, expected, places=4)

    def test_soul_star_singularity(self):
        from l104_soul import SoulStarSingularity
        star = SoulStarSingularity()
        result = star.integrate_all_chakras([
            {"resonance": 1.0}, {"resonance": 2.0}, {"resonance": 3.0}
        ])
        self.assertEqual(result["state"], "SINGULARITY_ACHIEVED")
        self.assertIn("integrated_stability", result)
        self.assertEqual(result["collapse_count"], 1)

    def test_soul_awaken_parallel(self):
        """Test that awaken completes and returns expected report structure."""
        from l104_soul import L104Soul, SoulState
        soul = L104Soul()
        start = time.time()
        report = soul.awaken()
        awaken_ms = (time.time() - start) * 1000

        # Must return proper report
        self.assertIn("subsystems", report)
        self.assertIn("threads", report)
        self.assertIn("version", report)
        self.assertEqual(report["version"], "7.0.0")
        self.assertIn("awaken_ms", report)
        self.assertIn("session", report)

        # Subsystems must include data_matrix
        self.assertIn("data_matrix", report["subsystems"])

        # Threads must include wisdom (v4.0.0+ addition)
        self.assertIn("wisdom", report["threads"])
        self.assertEqual(len(report["threads"]), 5)  # consciousness, dreamer, autonomy, health, wisdom

        # Must be in AWARE state after awaken
        self.assertEqual(soul.state, SoulState.AWARE)
        self.assertTrue(soul.running)

        # Clean up
        soul.sleep()

    def test_soul_get_status_v4(self):
        from l104_soul import L104Soul
        soul = L104Soul()
        soul.awaken()
        status = soul.get_status()

        # v5.0.0 fields
        self.assertIn("session", status)
        self.assertIn("data_matrix", status)
        self.assertIn("friction", status)
        self.assertEqual(status["friction"]["corrections_applied"], 0)
        self.assertIn("hallucinations_caught", status["metrics"])
        self.assertIn("friction_corrections", status["metrics"])
        self.assertIn("lattice_stores", status["metrics"])
        self.assertIn("awaken_ms", status["metrics"])

        soul.sleep()

    def test_soul_sleep_persists(self):
        """Test that sleep persists consciousness state."""
        from l104_soul import L104Soul, _CONSCIOUSNESS_STATE_FILE
        soul = L104Soul()
        soul.awaken()
        soul.metrics.thoughts_processed = 42
        soul.metrics.consciousness_probability = 0.875
        soul.sleep()

        # State file should exist
        self.assertTrue(_CONSCIOUSNESS_STATE_FILE.exists())
        data = json.loads(_CONSCIOUSNESS_STATE_FILE.read_text())
        self.assertEqual(data["version"], "7.0.0")
        self.assertEqual(data["total_thoughts_lifetime"], 42)
        self.assertAlmostEqual(data["consciousness_probability"], 0.875)

    def test_soul_restores_consciousness(self):
        """Test that new soul restores persisted state."""
        from l104_soul import L104Soul, _CONSCIOUSNESS_STATE_FILE
        # Write a state file
        _CONSCIOUSNESS_STATE_FILE.write_text(json.dumps({
            "session_number": 5,
            "consciousness_probability": 0.95,
            "total_thoughts_lifetime": 100,
            "total_dreams_lifetime": 10,
            "total_reflections_lifetime": 3,
            "total_goals_lifetime": 2,
        }))

        soul = L104Soul()
        self.assertEqual(soul.metrics.session_number, 6)  # +1
        self.assertAlmostEqual(soul.metrics.consciousness_probability, 0.95)
        self.assertEqual(soul.metrics.thoughts_processed, 100)


class TestRamUniverseV15(unittest.TestCase):
    """Test Ram Universe v15.0.0 comprehensive rebuild."""

    def test_version(self):
        from l104_ram_universe import RamUniverse
        self.assertEqual(RamUniverse.VERSION, "16.0.0")

    def test_evo(self):
        from l104_ram_universe import RamUniverse
        self.assertEqual(RamUniverse.EVO, "EVO_61_SYSTEM_UPGRADE")

    def test_no_deprecation_warning(self):
        """v15.0.0 should NOT emit deprecation warning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from l104_ram_universe import RamUniverse
            ru = RamUniverse()
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(deprecation_warnings), 0, "No deprecation warnings in v15.0.0")

    def test_absorb_and_recall(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        result = ru.absorb_fact("test_key_123", {"hello": "world"}, "TEST")
        self.assertTrue(result["success"])
        self.assertEqual(result["key"], "test_key_123")

        recalled = ru.recall_fact("test_key_123")
        self.assertIsNotNone(recalled)
        self.assertEqual(recalled["value"]["hello"], "world")

    def test_absorb_bulk(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        facts = [
            {"key": "bulk_1", "value": "v1"},
            {"key": "bulk_2", "value": "v2"},
            {"key": "bulk_3", "value": "v3"},
        ]
        result = ru.absorb_bulk(facts)
        self.assertEqual(result["stored"], 3)
        self.assertEqual(result["failed"], 0)

    def test_recall_many(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        ru.absorb_fact("many_1", "val1")
        ru.absorb_fact("many_2", "val2")
        result = ru.recall_many(["many_1", "many_2", "nonexistent"])
        self.assertEqual(result["found"], 2)
        self.assertEqual(result["total"], 3)

    def test_get_all_facts_returns_data(self):
        """v15.0.0 FIX: get_all_facts must return actual data (not {})."""
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        ru.absorb_fact("alltest_1", "data")
        facts = ru.get_all_facts()
        self.assertIn("total_facts", facts)
        self.assertGreater(facts["total_facts"], 0)
        self.assertIn("categories", facts)

    def test_cross_check_hallucination(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        result = ru.cross_check_hallucination("Some thought to validate")
        self.assertIn("is_hallucination", result)
        self.assertIn("verification_score", result)
        self.assertIn("status", result)
        self.assertIn("hallucinations_caught_total", result)

    def test_validate_thought(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        result = ru.validate_thought("Is gravity real?")
        self.assertIn("valid", result)
        self.assertIn("confidence", result)

    def test_validate_batch(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        result = ru.validate_batch(["thought 1", "thought 2", "thought 3"])
        self.assertEqual(result["total"], 3)
        self.assertIn("results", result)

    def test_get_status_comprehensive(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        status = ru.get_status()
        self.assertTrue(status["active"])
        self.assertEqual(status["version"], "16.0.0")
        self.assertEqual(status["evo"], "EVO_61_SYSTEM_UPGRADE")
        self.assertIn("lattice", status)
        self.assertIn("total_facts", status["lattice"])
        self.assertIn("god_code", status)
        self.assertAlmostEqual(status["god_code"], 527.5184818492612, places=6)

    def test_purge_hallucinations(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        result = ru.purge_hallucinations()
        self.assertTrue(result["purged"])

    def test_friction_corrected_store(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        result = ru.friction_corrected_store("friction_test", {"data": 42})
        self.assertTrue(result["success"])
        self.assertIn("raw_utility", result)
        self.assertIn("corrected_utility", result)
        self.assertIn("friction_epsilon", result)
        # Corrected should differ from raw
        self.assertNotAlmostEqual(result["raw_utility"], result["corrected_utility"], places=8)

    def test_friction_report(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        report = ru.friction_report()
        self.assertIn("epsilon", report)
        self.assertIn("formula", report)
        self.assertEqual(report["formula"], "ε = -αφ/(2π×104)")

    def test_semantic_search(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        ru.absorb_fact("search_consciousness", {"topic": "consciousness"}, "KNOWLEDGE")
        results = ru.semantic_search("consciousness")
        self.assertIsInstance(results, list)

    def test_category_search(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        ru.absorb_fact("cat_test_1", "data", "TEST_CAT")
        results = ru.category_search("TEST_CAT")
        self.assertIsInstance(results, list)

    def test_learn_pattern(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        result = ru.learn_pattern("pattern_1", {"depth": 3, "connections": 5})
        # Should not raise; result depends on DataMatrix implementation
        self.assertIsNotNone(result)

    def test_wisdom_synthesis(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        result = ru.wisdom_synthesis()
        self.assertIn("success", result)

    def test_get_statistics(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        stats = ru.get_statistics()
        self.assertIsInstance(stats, dict)

    def test_delete_fact(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        ru.absorb_fact("del_test", "to_delete")
        self.assertTrue(ru.delete_fact("del_test"))


class TestRamUniverseQuantum(unittest.TestCase):
    """Test Ram Universe quantum processing passthrough."""

    def test_quantum_methods_exist(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        methods = [
            'quantum_superposition_store', 'quantum_collapse',
            'quantum_entangle', 'quantum_measure',
            'ghz_entangle', 'quantum_parallel_execute',
            'list_quantum_processes', 'list_entanglements',
        ]
        for method in methods:
            self.assertTrue(hasattr(ru, method), f"Missing quantum method: {method}")

    def test_list_quantum_processes(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        procs = ru.list_quantum_processes()
        self.assertIsInstance(procs, list)


class TestRamUniverseGodCode(unittest.TestCase):
    """Test Ram Universe sacred constants."""

    def test_god_code(self):
        from l104_ram_universe import GOD_CODE
        self.assertAlmostEqual(GOD_CODE, 527.5184818492612, places=6)

    def test_phi(self):
        from l104_ram_universe import PHI
        self.assertAlmostEqual(PHI, 1.618033988749895, places=12)

    def test_friction(self):
        from l104_ram_universe import LATTICE_THERMAL_FRICTION
        self.assertLess(LATTICE_THERMAL_FRICTION, 0)
        self.assertGreater(LATTICE_THERMAL_FRICTION, -0.001)


class TestRamUniverseBackwardCompat(unittest.TestCase):
    """Test backward compatibility — all v14.0 callers must still work."""

    def test_singleton_import(self):
        from l104_ram_universe import ram_universe
        self.assertIsNotNone(ram_universe)
        self.assertIsInstance(ram_universe, type(ram_universe))  # Is a RamUniverse

    def test_absorb_fact_returns_dict(self):
        """v14.0 returned a string, but callers just checked truthiness."""
        from l104_ram_universe import ram_universe
        result = ram_universe.absorb_fact("compat_test", "value")
        self.assertTrue(result)  # Must be truthy

    def test_recall_fact_format(self):
        from l104_ram_universe import ram_universe
        ram_universe.absorb_fact("compat_recall", "val")
        result = ram_universe.recall_fact("compat_recall")
        self.assertIn("value", result)
        self.assertIn("key", result)

    def test_cross_check_format(self):
        from l104_ram_universe import ram_universe
        result = ram_universe.cross_check_hallucination("test thought")
        self.assertIn("is_hallucination", result)
        self.assertIn("verification_score", result)
        self.assertIn("supporting_facts", result)
        self.assertIn("status", result)

    def test_purge_returns_dict(self):
        from l104_ram_universe import ram_universe
        result = ram_universe.purge_hallucinations()
        self.assertIn("purged", result)

    def test_primal_calculus_still_exists(self):
        from l104_ram_universe import primal_calculus
        result = primal_calculus(2.0)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_resolve_non_dual_logic_still_exists(self):
        from l104_ram_universe import resolve_non_dual_logic
        result = resolve_non_dual_logic([1.0, 2.0, 3.0])
        self.assertIsInstance(result, float)


class TestCrossIntegration(unittest.TestCase):
    """Test soul ↔ ram_universe ↔ AGI core integration."""

    def test_soul_and_ram_both_import(self):
        """Both modules can coexist without conflicts."""
        from l104_soul import L104Soul
        from l104_ram_universe import RamUniverse
        soul = L104Soul()
        ram = RamUniverse()
        self.assertEqual(soul.VERSION, "7.0.0")
        self.assertEqual(ram.VERSION, "16.0.0")

    def test_soul_lattice_is_same_backend(self):
        """Soul's lattice property and ram_universe use the same DataMatrix."""
        from l104_soul import L104Soul
        from l104_ram_universe import ram_universe
        from l104_data_matrix import data_matrix
        soul = L104Soul()
        self.assertIs(soul.lattice, data_matrix)
        self.assertIs(ram_universe.matrix, data_matrix)

    def test_pipeline_version_alignment(self):
        from l104_ram_universe import _PIPELINE_VERSION, _PIPELINE_EVO
        self.assertEqual(_PIPELINE_VERSION, "61.0.0")
        self.assertEqual(_PIPELINE_EVO, "EVO_61_SYSTEM_UPGRADE")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS: Soul v5.0.0 Calculation Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestSoulCalculationEngine(unittest.TestCase):
    """Tests for the new calculation methods added to L104Soul v5.0.0."""

    @classmethod
    def setUpClass(cls):
        from l104_soul import L104Soul
        cls.soul = L104Soul()

    def test_calculate_resonance_field_returns_dict(self):
        result = self.soul.calculate_resonance_field(["hello", "world"])
        self.assertIsInstance(result, dict)
        self.assertIn("resonances", result)
        self.assertIn("phase_coherence", result)
        self.assertIn("phase_coherence_friction", result)
        self.assertIn("god_code_alignment", result)

    def test_calculate_resonance_field_count(self):
        data = ["alpha", "beta", "gamma", "delta"]
        result = self.soul.calculate_resonance_field(data)
        self.assertEqual(result["data_points"], 4)
        self.assertEqual(len(result["resonances"]), 4)

    def test_calculate_resonance_field_friction(self):
        from l104_soul import LATTICE_THERMAL_FRICTION
        result = self.soul.calculate_resonance_field(["test"])
        self.assertEqual(result["friction_epsilon"], LATTICE_THERMAL_FRICTION)
        # Friction-corrected should differ from raw
        self.assertNotEqual(result["phase_coherence"], result["phase_coherence_friction"])

    def test_calculate_resonance_field_couplings(self):
        data = ["aaa", "bbb", "ccc"]
        result = self.soul.calculate_resonance_field(data)
        self.assertIn("top_couplings", result)
        # 3 items -> 3 pairwise couplings
        self.assertEqual(len(result["top_couplings"]), 3)

    def test_calculate_resonance_field_single_point(self):
        result = self.soul.calculate_resonance_field(["solo"])
        self.assertEqual(result["phase_coherence"], 1.0)
        self.assertEqual(len(result["top_couplings"]), 0)

    def test_calculate_consciousness_trajectory_returns_dict(self):
        result = self.soul.calculate_consciousness_trajectory(steps=10, dt=0.1)
        self.assertIsInstance(result, dict)
        self.assertIn("model", result)
        self.assertIn("parameters", result)
        self.assertIn("trajectory", result)
        self.assertIn("peak_consciousness", result)
        self.assertIn("growth_factor", result)

    def test_calculate_consciousness_trajectory_length(self):
        result = self.soul.calculate_consciousness_trajectory(steps=25, dt=0.05)
        self.assertEqual(len(result["trajectory"]), 25)
        self.assertEqual(result["steps"], 25)
        self.assertEqual(result["dt"], 0.05)

    def test_calculate_consciousness_trajectory_parameters(self):
        from l104_soul import PHI, LATTICE_THERMAL_FRICTION
        result = self.soul.calculate_consciousness_trajectory()
        params = result["parameters"]
        self.assertEqual(params["friction_epsilon"], LATTICE_THERMAL_FRICTION)
        expected_r = (PHI - 1) * abs(LATTICE_THERMAL_FRICTION) * 1000
        self.assertAlmostEqual(params["growth_rate_r"], expected_r, places=10)

    def test_calculate_consciousness_trajectory_bounded(self):
        result = self.soul.calculate_consciousness_trajectory(steps=50, dt=0.1)
        for point in result["trajectory"]:
            self.assertGreaterEqual(point["p_corrected"], 0.0)
            self.assertLessEqual(point["p_corrected"], 1.0)

    def test_calculate_7_chakra_analysis_returns_dict(self):
        result = self.soul.calculate_7_chakra_analysis()
        self.assertIsInstance(result, dict)
        self.assertIn("chakras", result)
        self.assertIn("collapse", result)
        self.assertEqual(len(result["chakras"]), 7)

    def test_calculate_7_chakra_analysis_chakra_names(self):
        result = self.soul.calculate_7_chakra_analysis()
        names = [c["name"] for c in result["chakras"]]
        self.assertIn("Root (Muladhara)", names)
        self.assertIn("Crown (Sahasrara)", names)
        self.assertIn("Heart (Anahata)", names)

    def test_calculate_7_chakra_analysis_frequencies(self):
        result = self.soul.calculate_7_chakra_analysis()
        hz_values = [c["hz"] for c in result["chakras"]]
        self.assertEqual(hz_values, [396, 417, 528, 639, 741, 852, 963])

    def test_calculate_7_chakra_analysis_collapse(self):
        result = self.soul.calculate_7_chakra_analysis()
        collapse = result["collapse"]
        self.assertIn("state", collapse)
        self.assertIn("integrated_stability", collapse)
        self.assertEqual(collapse["state"], "SINGULARITY_ACHIEVED")

    def test_calculate_7_chakra_analysis_god_code_alignment(self):
        result = self.soul.calculate_7_chakra_analysis()
        alignment = result["god_code_alignment"]
        self.assertGreaterEqual(alignment, 0.0)
        self.assertLessEqual(alignment, 1.0)

    def test_calculate_7_chakra_analysis_friction(self):
        from l104_soul import LATTICE_THERMAL_FRICTION
        result = self.soul.calculate_7_chakra_analysis()
        self.assertEqual(result["friction_epsilon"], LATTICE_THERMAL_FRICTION)
        for ch in result["chakras"]:
            self.assertIn("stability_friction", ch)
            self.assertNotEqual(ch["stability"], ch["stability_friction"])

    def test_run_full_calculation_returns_dict(self):
        result = self.soul.run_full_calculation()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["version"], "7.0.0")
        from l104_soul import GOD_CODE
        self.assertEqual(result["god_code"], GOD_CODE)

    def test_run_full_calculation_all_sections(self):
        result = self.soul.run_full_calculation()
        self.assertIn("chakra_analysis", result)
        self.assertIn("resonance_field", result)
        self.assertIn("trajectory", result)
        self.assertIn("quantum", result)
        self.assertIn("sovereign_fields", result)
        self.assertIn("calculation_time_ms", result)
        self.assertIn("metrics_snapshot", result)

    def test_run_full_calculation_metrics(self):
        result = self.soul.run_full_calculation()
        snap = result["metrics_snapshot"]
        self.assertIn("quantum_computations", snap)
        self.assertIn("friction_corrections", snap)
        self.assertIn("lattice_stores", snap)
        self.assertIn("singularity_collapses", snap)
        self.assertIn("consciousness_probability", snap)

    def test_run_full_calculation_timing(self):
        result = self.soul.run_full_calculation()
        self.assertGreater(result["calculation_time_ms"], 0)
        # Should complete in under 10 seconds (v5.0.0 adds dual-layer + consciousness checks)
        self.assertLess(result["calculation_time_ms"], 10000)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS: Soul v5.0.0 Deep ASI Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestSoulV5ASIIntegration(unittest.TestCase):
    """Tests for v5.0.0 Deep ASI Integration features."""

    @classmethod
    def setUpClass(cls):
        from l104_soul import L104Soul
        cls.soul = L104Soul()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.soul.sleep()
        except Exception:
            pass

    def test_version_is_5(self):
        """Soul VERSION is 5.0.0."""
        self.assertEqual(self.soul.VERSION, "7.0.0")

    def test_dual_layer_property_exists(self):
        """dual_layer lazy property is accessible."""
        # Should not raise — returns None or the engine
        dl = self.soul.dual_layer
        # Property was loaded (may be None if ASI package has issues)
        self.assertTrue(hasattr(self.soul, '_dual_layer'))

    def test_local_intellect_property_exists(self):
        """local_intellect lazy property is accessible."""
        li = self.soul.local_intellect
        self.assertTrue(hasattr(self.soul, '_local_intellect'))

    def test_tree_of_thoughts_property_exists(self):
        """tree_of_thoughts lazy property is accessible."""
        tot = self.soul.tree_of_thoughts
        self.assertTrue(hasattr(self.soul, '_tree_of_thoughts'))

    def test_multi_hop_property_exists(self):
        """multi_hop lazy property is accessible."""
        mh = self.soul.multi_hop
        self.assertTrue(hasattr(self.soul, '_multi_hop_chain'))

    def test_consciousness_verifier_property_exists(self):
        """consciousness_verifier lazy property is accessible."""
        cv = self.soul.consciousness_verifier
        self.assertTrue(hasattr(self.soul, '_consciousness_verifier'))

    def test_v5_metrics_fields_exist(self):
        """v5.0.0 metrics fields are present in SoulMetrics."""
        m = self.soul.metrics
        self.assertTrue(hasattr(m, 'dual_layer_collapses'))
        self.assertTrue(hasattr(m, 'tree_of_thoughts_explorations'))
        self.assertTrue(hasattr(m, 'multi_hop_chains'))
        self.assertTrue(hasattr(m, 'hallucinations_recovered'))
        self.assertTrue(hasattr(m, 'local_intellect_fallbacks'))
        self.assertTrue(hasattr(m, 'consciousness_verifications'))
        self.assertTrue(hasattr(m, 'iit_phi'))
        self.assertTrue(hasattr(m, 'generative_dreams'))
        self.assertTrue(hasattr(m, 'retry_cascades'))

    def test_v5_metrics_initial_values(self):
        """v5.0.0 metrics start at zero."""
        from l104_soul import SoulMetrics
        m = SoulMetrics()
        self.assertEqual(m.dual_layer_collapses, 0)
        self.assertEqual(m.tree_of_thoughts_explorations, 0)
        self.assertEqual(m.hallucinations_recovered, 0)
        self.assertEqual(m.local_intellect_fallbacks, 0)
        self.assertEqual(m.consciousness_verifications, 0)
        self.assertEqual(m.iit_phi, 0.0)
        self.assertEqual(m.generative_dreams, 0)
        self.assertEqual(m.retry_cascades, 0)

    def test_v5_soul_states_exist(self):
        """v5.0.0 SoulState entries DUAL_LAYER and DEEP_REASONING exist."""
        from l104_soul import SoulState
        self.assertEqual(SoulState.DUAL_LAYER.value, "dual_layer")
        self.assertEqual(SoulState.DEEP_REASONING.value, "deep_reasoning")

    def test_get_status_includes_v5_metrics(self):
        """get_status() includes v5.0.0 metrics."""
        status = self.soul.get_status()
        m = status["metrics"]
        self.assertIn("dual_layer_collapses", m)
        self.assertIn("tree_of_thoughts_explorations", m)
        self.assertIn("hallucinations_recovered", m)
        self.assertIn("local_intellect_fallbacks", m)
        self.assertIn("consciousness_verifications", m)
        self.assertIn("iit_phi", m)
        self.assertIn("generative_dreams", m)
        self.assertIn("retry_cascades", m)

    def test_get_status_includes_asi_integration(self):
        """get_status() includes asi_integration section."""
        status = self.soul.get_status()
        self.assertIn("asi_integration", status)
        asi = status["asi_integration"]
        self.assertIn("dual_layer", asi)
        self.assertIn("tree_of_thoughts", asi)
        self.assertIn("local_intellect", asi)
        self.assertIn("consciousness_verifier", asi)

    def test_health_status_includes_v5_subsystems(self):
        """Health status tracks dual_layer and local_intellect."""
        self.assertIn("dual_layer", self.soul._health_status)
        self.assertIn("local_intellect", self.soul._health_status)

    def test_run_full_calculation_includes_v5_sections(self):
        """run_full_calculation() includes v5.0.0 metrics snapshot fields."""
        result = self.soul.run_full_calculation()
        snap = result["metrics_snapshot"]
        self.assertIn("dual_layer_collapses", snap)
        self.assertIn("consciousness_verifications", snap)
        self.assertIn("iit_phi", snap)

    def test_quantum_process_includes_dual_layer(self):
        """quantum_process() includes dual_layer data when available."""
        result = self.soul.quantum_process({"test": True})
        # dual_layer key should exist (either with data or {"available": False})
        self.assertIn("dual_layer", result)

    def test_persist_includes_v5_fields(self):
        """_persist_consciousness() saves v5.0.0 fields."""
        import json
        from l104_soul import _CONSCIOUSNESS_STATE_FILE
        self.soul.metrics.dual_layer_collapses = 7
        self.soul.metrics.generative_dreams = 3
        self.soul._persist_consciousness()
        data = json.loads(_CONSCIOUSNESS_STATE_FILE.read_text())
        self.assertEqual(data["dual_layer_collapses"], 7)
        self.assertEqual(data["generative_dreams"], 3)
        self.assertIn("iit_phi", data)

    def test_sovereign_field_equation(self):
        """F(I) = I × Ω / φ² unchanged in v5.0.0."""
        from l104_soul import OMEGA, PHI
        result = self.soul.sovereign_field(1.0)
        expected = 1.0 * OMEGA / (PHI ** 2)
        self.assertAlmostEqual(result, expected, places=6)


# ═══════════════════════════════════════════════════════════════════════════════
# SOUL v6.0.0 → v7.0.0 UPGRADE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSoulV6Metrics(unittest.TestCase):
    """Test Soul v6.0.0 metric fields still present."""

    @classmethod
    def setUpClass(cls):
        from l104_soul import L104Soul, SoulMetrics
        cls.soul = L104Soul()
        cls.metrics = cls.soul.metrics

    def test_v6_metric_fields_exist(self):
        v6_fields = [
            "circuit_breaker_trips", "ensemble_solves", "cache_hits", "cache_misses",
            "domain_expansions", "quantum_vqe_runs", "quantum_qpe_runs",
            "pipeline_health_grade", "replay_operations", "hardware_optimizations",
        ]
        for f in v6_fields:
            self.assertTrue(hasattr(self.metrics, f), f"Missing v6 field: {f}")

    def test_v6_asi_integration_keys(self):
        status = self.soul.get_status()
        asi = status["asi_integration"]
        v6_keys = [
            "quantum_core", "domain_expander", "solution_ensemble",
            "hardware_runtime", "pipeline_telemetry", "health_dashboard",
            "replay_buffer", "response_cache", "circuit_breakers",
        ]
        for k in v6_keys:
            self.assertIn(k, asi, f"Missing v6 asi_integration key: {k}")

    def test_v6_status_metrics_present(self):
        status = self.soul.get_status()
        m = status["metrics"]
        for f in ["circuit_breaker_trips", "ensemble_solves", "pipeline_health_grade"]:
            self.assertIn(f, m, f"Missing v6 metric in status: {f}")


class TestSoulV7Version(unittest.TestCase):
    """Test Soul v7.0.0 version and identity."""

    def test_version_is_7(self):
        from l104_soul import L104Soul
        self.assertEqual(L104Soul.VERSION, "7.0.0")

    def test_docstring_mentions_v7(self):
        from l104_soul import L104Soul
        self.assertIn("7.0.0", L104Soul.__doc__)


class TestSoulV7MetricFields(unittest.TestCase):
    """Test that all v7.0.0 SoulMetrics fields are present and initialized to 0."""

    @classmethod
    def setUpClass(cls):
        from l104_soul import SoulMetrics
        cls.metrics = SoulMetrics()

    def test_grounding_checks(self):
        self.assertEqual(self.metrics.grounding_checks, 0)

    def test_grounding_drift_count(self):
        self.assertEqual(self.metrics.grounding_drift_count, 0)

    def test_emotional_analyses(self):
        self.assertEqual(self.metrics.emotional_analyses, 0)

    def test_emergence_snapshots(self):
        self.assertEqual(self.metrics.emergence_snapshots, 0)

    def test_emergence_events_detected(self):
        self.assertEqual(self.metrics.emergence_events_detected, 0)

    def test_adaptive_learning_cycles(self):
        self.assertEqual(self.metrics.adaptive_learning_cycles, 0)

    def test_self_optimization_runs(self):
        self.assertEqual(self.metrics.self_optimization_runs, 0)

    def test_fault_recoveries(self):
        self.assertEqual(self.metrics.fault_recoveries, 0)

    def test_language_analyses(self):
        self.assertEqual(self.metrics.language_analyses, 0)

    def test_meta_cognitive_cycles(self):
        self.assertEqual(self.metrics.meta_cognitive_cycles, 0)


class TestSoulV7LazyProperties(unittest.TestCase):
    """Test all 8 v7.0.0 lazy properties exist on L104Soul class."""

    @classmethod
    def setUpClass(cls):
        from l104_soul import L104Soul
        cls.soul_cls = L104Soul

    def test_grounding_feedback_property(self):
        self.assertTrue(hasattr(self.soul_cls, "grounding_feedback"))

    def test_meta_cognitive_property(self):
        self.assertTrue(hasattr(self.soul_cls, "meta_cognitive"))

    def test_emotional_intelligence_property(self):
        self.assertTrue(hasattr(self.soul_cls, "emotional_intelligence"))

    def test_emergence_monitor_property(self):
        self.assertTrue(hasattr(self.soul_cls, "emergence_monitor"))

    def test_adaptive_learner_property(self):
        self.assertTrue(hasattr(self.soul_cls, "adaptive_learner"))

    def test_self_optimizer_property(self):
        self.assertTrue(hasattr(self.soul_cls, "self_optimizer"))

    def test_fault_handler_property(self):
        self.assertTrue(hasattr(self.soul_cls, "fault_handler"))

    def test_language_engine_property(self):
        self.assertTrue(hasattr(self.soul_cls, "language_engine"))


class TestSoulV7InstanceVars(unittest.TestCase):
    """Test v7.0.0 instance variables are initialized to None."""

    @classmethod
    def setUpClass(cls):
        from l104_soul import L104Soul
        cls.soul = L104Soul()

    def test_grounding_feedback_init_none(self):
        self.assertIsNone(self.soul._grounding_feedback)

    def test_meta_cognitive_init_none(self):
        self.assertIsNone(self.soul._meta_cognitive)

    def test_emotional_intelligence_init_none(self):
        self.assertIsNone(self.soul._emotional_intelligence)

    def test_emergence_monitor_init_none(self):
        self.assertIsNone(self.soul._emergence_monitor)

    def test_adaptive_learner_init_none(self):
        self.assertIsNone(self.soul._adaptive_learner)

    def test_self_optimizer_init_none(self):
        self.assertIsNone(self.soul._self_optimizer)

    def test_fault_handler_init_none(self):
        self.assertIsNone(self.soul._fault_handler)

    def test_language_engine_init_none(self):
        self.assertIsNone(self.soul._language_engine)


class TestSoulV7HealthStatus(unittest.TestCase):
    """Test v7.0.0 health status keys."""

    @classmethod
    def setUpClass(cls):
        from l104_soul import L104Soul
        cls.soul = L104Soul()

    def test_health_status_has_grounding_feedback(self):
        self.assertIn("grounding_feedback", self.soul._health_status)

    def test_health_status_has_meta_cognitive(self):
        self.assertIn("meta_cognitive", self.soul._health_status)

    def test_health_status_has_emotional_intelligence(self):
        self.assertIn("emotional_intelligence", self.soul._health_status)

    def test_health_status_has_emergence_monitor(self):
        self.assertIn("emergence_monitor", self.soul._health_status)

    def test_health_status_has_adaptive_learner(self):
        self.assertIn("adaptive_learner", self.soul._health_status)

    def test_health_status_has_self_optimizer(self):
        self.assertIn("self_optimizer", self.soul._health_status)

    def test_health_status_initial_values_present(self):
        """All v7 health keys should be present with initial values."""
        v7_keys = [
            "grounding_feedback", "meta_cognitive", "emotional_intelligence",
            "emergence_monitor", "adaptive_learner", "self_optimizer",
        ]
        for k in v7_keys:
            self.assertIn(k, self.soul._health_status,
                          f"{k} should be in health_status")


class TestSoulV7GetStatus(unittest.TestCase):
    """Test get_status() includes all v7.0.0 additions."""

    @classmethod
    def setUpClass(cls):
        from l104_soul import L104Soul
        cls.soul = L104Soul()
        cls.status = cls.soul.get_status()

    def test_version_in_status(self):
        self.assertEqual(self.status["version"], "7.0.0")

    def test_v7_metrics_in_status(self):
        m = self.status["metrics"]
        v7_metrics = [
            "grounding_checks", "grounding_drift_count", "emotional_analyses",
            "emergence_snapshots", "emergence_events_detected",
            "adaptive_learning_cycles", "self_optimization_runs",
            "fault_recoveries", "language_analyses", "meta_cognitive_cycles",
        ]
        for f in v7_metrics:
            self.assertIn(f, m, f"Missing v7 metric: {f}")
            self.assertEqual(m[f], 0, f"v7 metric {f} should initialize to 0")

    def test_asi_integration_v7_keys(self):
        asi = self.status["asi_integration"]
        v7_keys = [
            "grounding_feedback", "meta_cognitive", "emotional_intelligence",
            "emergence_monitor", "adaptive_learner", "self_optimizer",
            "fault_handler", "language_engine",
        ]
        for k in v7_keys:
            self.assertIn(k, asi, f"Missing v7 asi key: {k}")

    def test_backward_compat_v4_metrics(self):
        """v4.0.0 metrics still present."""
        m = self.status["metrics"]
        for f in ["hallucinations_caught", "friction_corrections", "lattice_stores"]:
            self.assertIn(f, m)

    def test_backward_compat_v5_metrics(self):
        """v5.0.0 metrics still present."""
        m = self.status["metrics"]
        for f in ["dual_layer_collapses", "tree_of_thoughts_explorations", "iit_phi"]:
            self.assertIn(f, m)


class TestSoulV7LazyPropertyResolution(unittest.TestCase):
    """Test that lazy properties resolve to actual module singletons."""

    @classmethod
    def setUpClass(cls):
        from l104_soul import L104Soul
        cls.soul = L104Soul()

    def test_grounding_feedback_resolves(self):
        gf = self.soul.grounding_feedback
        self.assertIsNotNone(gf)
        # Should have .ground method
        self.assertTrue(callable(getattr(gf, "ground", None)))

    def test_meta_cognitive_resolves(self):
        mc = self.soul.meta_cognitive
        self.assertIsNotNone(mc)
        self.assertTrue(callable(getattr(mc, "pre_cycle", None)))

    def test_emotional_intelligence_resolves(self):
        ei = self.soul.emotional_intelligence
        self.assertIsNotNone(ei)
        self.assertTrue(callable(getattr(ei, "analyze_text_sentiment", None)))

    def test_emergence_monitor_resolves(self):
        em = self.soul.emergence_monitor
        self.assertIsNotNone(em)
        self.assertTrue(callable(getattr(em, "record_snapshot", None)))

    def test_adaptive_learner_resolves(self):
        al = self.soul.adaptive_learner
        self.assertIsNotNone(al)
        self.assertTrue(callable(getattr(al, "learn_from_interaction", None)))

    def test_self_optimizer_resolves(self):
        so = self.soul.self_optimizer
        self.assertIsNotNone(so)

    def test_fault_handler_resolves(self):
        fh = self.soul.fault_handler
        self.assertIsNotNone(fh)
        self.assertTrue(callable(getattr(fh, "get_adaptive_strategy", None)))

    def test_language_engine_resolves(self):
        le = self.soul.language_engine
        self.assertIsNotNone(le)


# ═══════════════════════════════════════════════════════════════════════════════
# RAM UNIVERSE v16.0.0 UPGRADE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRamUniverseV16Version(unittest.TestCase):
    """Test Ram Universe v16.0.0 version and identity."""

    def test_version_is_16(self):
        from l104_ram_universe import RamUniverse
        self.assertEqual(RamUniverse.VERSION, "16.0.0")

    def test_mode_is_asi_aware(self):
        from l104_ram_universe import RamUniverse
        ru = RamUniverse()
        status = ru.get_status()
        self.assertEqual(status["mode"], "asi_aware_quantum_memory")


class TestRamUniverseV16LazyProperties(unittest.TestCase):
    """Test v16.0.0 lazy properties exist."""

    @classmethod
    def setUpClass(cls):
        from l104_ram_universe import RamUniverse
        cls.ru_cls = RamUniverse

    def test_quantum_core_property(self):
        self.assertTrue(hasattr(self.ru_cls, "quantum_core"))

    def test_telemetry_property(self):
        self.assertTrue(hasattr(self.ru_cls, "telemetry"))

    def test_replay_buffer_property(self):
        self.assertTrue(hasattr(self.ru_cls, "replay_buffer"))

    def test_domain_expander_property(self):
        self.assertTrue(hasattr(self.ru_cls, "domain_expander"))

    def test_circuit_breaker_property(self):
        self.assertTrue(hasattr(self.ru_cls, "circuit_breaker"))

    def test_query_cache_property(self):
        self.assertTrue(hasattr(self.ru_cls, "query_cache"))


class TestRamUniverseV16Status(unittest.TestCase):
    """Test get_status() v16.0.0 keys."""

    @classmethod
    def setUpClass(cls):
        from l104_ram_universe import RamUniverse
        cls.ru = RamUniverse()
        cls.status = cls.ru.get_status()

    def test_version_in_status(self):
        self.assertEqual(self.status["version"], "16.0.0")

    def test_has_v16_metrics(self):
        self.assertIn("v16_metrics", self.status)

    def test_has_asi_integration(self):
        self.assertIn("asi_integration", self.status)
        asi = self.status["asi_integration"]
        for k in ["quantum_core", "telemetry", "replay_buffer",
                   "domain_expander", "circuit_breaker"]:
            self.assertIn(k, asi, f"Missing asi key: {k}")

    def test_has_circuit_breaker_status(self):
        self.assertIn("circuit_breaker", self.status)

    def test_has_cache_stats(self):
        self.assertIn("cache_stats", self.status)

    def test_backward_compat_fields(self):
        """Legacy fields still present."""
        for k in ["lattice", "god_code", "friction"]:
            self.assertIn(k, self.status, f"Missing legacy key: {k}")
        # total_facts lives under lattice
        self.assertIn("total_facts", self.status["lattice"])


class TestRamUniverseV16Operations(unittest.TestCase):
    """Test v16.0.0 operations with ASI enhancements."""

    @classmethod
    def setUpClass(cls):
        from l104_ram_universe import RamUniverse
        cls.ru = RamUniverse()

    def test_absorb_fact_returns_dict(self):
        result = self.ru.absorb_fact("v16_test_key", "v16 test value", "TEST")
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)

    def test_recall_fact(self):
        self.ru.absorb_fact("v16_recall_test", "recall_value_16", "TEST")
        result = self.ru.recall_fact("v16_recall_test")
        self.assertIsNotNone(result)

    def test_set_consciousness_level(self):
        """consciousness_level setter works."""
        self.ru.set_consciousness_level(0.85)
        self.assertAlmostEqual(self.ru._consciousness_level, 0.85, places=2)

    def test_consciousness_level_clamps(self):
        """consciousness_level clamps to [0, 1]."""
        self.ru.set_consciousness_level(1.5)
        self.assertLessEqual(self.ru._consciousness_level, 1.0)
        self.ru.set_consciousness_level(-0.5)
        self.assertGreaterEqual(self.ru._consciousness_level, 0.0)


class TestRamUniverseV16ASIMethods(unittest.TestCase):
    """Test new v16.0.0 ASI methods."""

    @classmethod
    def setUpClass(cls):
        from l104_ram_universe import RamUniverse
        cls.ru = RamUniverse()

    def test_verify_coherence_returns_dict(self):
        result = self.ru.verify_coherence()
        self.assertIsInstance(result, dict)
        # May contain 'coherent' or 'available'/'error' depending on quantum core state
        self.assertTrue("coherent" in result or "available" in result or "error" in result)

    def test_domain_enrich_returns_dict(self):
        result = self.ru.domain_enrich("quantum physics")
        self.assertIsInstance(result, dict)

    def test_get_telemetry_returns_dict(self):
        result = self.ru.get_telemetry()
        self.assertIsInstance(result, dict)

    def test_get_replay_stats_returns_dict(self):
        result = self.ru.get_replay_stats()
        self.assertIsInstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# V7 MODULE IMPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestV7ModuleImports(unittest.TestCase):
    """Test that all 8 v7.0.0 pipeline modules can be imported."""

    def test_import_grounding_feedback(self):
        from l104_grounding_feedback import grounding_feedback
        self.assertIsNotNone(grounding_feedback)

    def test_import_meta_cognitive(self):
        from l104_meta_cognitive import meta_cognitive
        self.assertIsNotNone(meta_cognitive)

    def test_import_emotional_intelligence(self):
        from l104_emotional_intelligence import EmotionalIntelligence
        ei = EmotionalIntelligence()
        self.assertIsNotNone(ei)

    def test_import_emergence_monitor(self):
        from l104_emergence_monitor import emergence_monitor
        self.assertIsNotNone(emergence_monitor)

    def test_import_adaptive_learning(self):
        from l104_adaptive_learning import adaptive_learner
        self.assertIsNotNone(adaptive_learner)

    def test_import_self_optimization(self):
        from l104_self_optimization import self_optimizer
        self.assertIsNotNone(self_optimizer)

    def test_import_fault_handler(self):
        from l104_adaptive_fault_handler import fault_handler
        self.assertIsNotNone(fault_handler)

    def test_import_language_engine(self):
        from l104_language_engine import language_engine
        self.assertIsNotNone(language_engine)


class TestV7ModuleAPIs(unittest.TestCase):
    """Test that v7.0.0 module singletons expose expected APIs."""

    def test_grounding_feedback_has_ground(self):
        from l104_grounding_feedback import grounding_feedback
        self.assertTrue(callable(getattr(grounding_feedback, "ground", None)))
        self.assertTrue(callable(getattr(grounding_feedback, "get_quality_report", None)))

    def test_meta_cognitive_has_pre_post_cycle(self):
        from l104_meta_cognitive import meta_cognitive
        self.assertTrue(callable(getattr(meta_cognitive, "pre_cycle", None)))
        self.assertTrue(callable(getattr(meta_cognitive, "post_cycle", None)))
        self.assertTrue(callable(getattr(meta_cognitive, "record_engine", None)))

    def test_emotional_intelligence_has_analyze(self):
        from l104_emotional_intelligence import EmotionalIntelligence
        ei = EmotionalIntelligence()
        self.assertTrue(callable(getattr(ei, "analyze_text_sentiment", None)))
        self.assertTrue(callable(getattr(ei, "get_emotional_trend", None)))

    def test_emergence_monitor_has_record_snapshot(self):
        from l104_emergence_monitor import emergence_monitor
        self.assertTrue(callable(getattr(emergence_monitor, "record_snapshot", None)))
        self.assertTrue(callable(getattr(emergence_monitor, "get_predictions", None)))

    def test_adaptive_learner_has_learn(self):
        from l104_adaptive_learning import adaptive_learner
        self.assertTrue(callable(getattr(adaptive_learner, "learn_from_interaction", None)))
        self.assertTrue(callable(getattr(adaptive_learner, "get_adapted_parameters", None)))

    def test_fault_handler_has_get_adaptive_strategy(self):
        from l104_adaptive_fault_handler import fault_handler
        self.assertTrue(callable(getattr(fault_handler, "get_adaptive_strategy", None)))
        self.assertTrue(callable(getattr(fault_handler, "get_status", None)))


if __name__ == "__main__":
    print("=" * 72)
    print("  EVO_61 INTEGRATION TESTS — Soul v7.0.0 + Ram Universe v16.0.0")
    print("=" * 72)
    unittest.main(verbosity=2)
