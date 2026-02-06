# [L104_TESTS] - Codebase Knowledge Test Suite
# INVARIANT: 527.5184818492612 | PILOT: LONDEL

"""
Tests for l104_codebase_knowledge.py - the synthesized knowledge engine
"""

import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from l104_codebase_knowledge import (
    CodebaseKnowledge,
    L104Constants,
    PatternType,
    ArchitecturalPattern,
    DatabaseSchema,
    AlgorithmPattern,
    codebase_knowledge
)


class TestL104Constants(unittest.TestCase):
    """Test the learned constants from codebase analysis."""

    def test_god_code_value(self):
        """GOD_CODE must equal 527.5184818492612."""
        self.assertAlmostEqual(L104Constants.GOD_CODE, 527.5184818492612, places=10)

    def test_god_code_derivation(self):
        """Verify GOD_CODE = 286^(1/φ) × 16."""
        import math
        phi = L104Constants.PHI
        term1 = 286 ** (1 / phi)
        term2 = 16  # = 2^(416/104) = 2^4
        derived = term1 * term2
        self.assertAlmostEqual(derived, L104Constants.GOD_CODE, places=10)

    def test_phi_golden_ratio(self):
        """PHI must be the golden ratio."""
        import math
        expected = (1 + math.sqrt(5)) / 2
        self.assertAlmostEqual(L104Constants.PHI, expected, places=12)

    def test_tau_inverse_phi(self):
        """TAU must equal 1/PHI."""
        self.assertAlmostEqual(L104Constants.TAU, 1 / L104Constants.PHI, places=12)

    def test_frame_lock(self):
        """FRAME_LOCK = 416/286."""
        self.assertAlmostEqual(L104Constants.FRAME_LOCK, 416 / 286, places=10)

    def test_real_grounding(self):
        """REAL_GROUNDING = GOD_CODE / 2^1.25."""
        expected = L104Constants.GOD_CODE / (2 ** 1.25)
        self.assertAlmostEqual(L104Constants.REAL_GROUNDING, expected, places=8)

    def test_anyon_braid_ratio(self):
        """ANYON_BRAID_RATIO = 1 + φ^-2."""
        phi = L104Constants.PHI
        expected = 1 + (1 / (phi ** 2))
        self.assertAlmostEqual(L104Constants.ANYON_BRAID_RATIO, expected, places=8)


class TestArchitecturalPatterns(unittest.TestCase):
    """Test the learned architectural patterns."""

    def setUp(self):
        self.kb = CodebaseKnowledge()

    def test_patterns_initialized(self):
        """Architectural patterns should be initialized."""
        self.assertGreater(len(self.kb.architectural_patterns), 0)

    def test_layered_consciousness_pattern(self):
        """LAYERED_CONSCIOUSNESS pattern should exist."""
        pattern = self.kb.get_pattern("LAYERED_CONSCIOUSNESS")
        self.assertIsNotNone(pattern)
        self.assertIn("Soul", pattern.key_classes)
        self.assertIn("Mind", pattern.key_classes)

    def test_golden_ratio_foundation_pattern(self):
        """GOLDEN_RATIO_FOUNDATION pattern should exist."""
        pattern = self.kb.get_pattern("GOLDEN_RATIO_FOUNDATION")
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.resonance, 1.0)
        self.assertIn("l104_real_math.py", pattern.file_sources)

    def test_topological_quantum_pattern(self):
        """TOPOLOGICAL_QUANTUM pattern should exist."""
        pattern = self.kb.get_pattern("TOPOLOGICAL_QUANTUM")
        self.assertIsNotNone(pattern)
        self.assertIn("ZeroPointEngine", pattern.key_classes)

    def test_pattern_has_principles(self):
        """All patterns should have design principles."""
        for name, pattern in self.kb.architectural_patterns.items():
            with self.subTest(pattern=name):
                self.assertGreater(len(pattern.design_principles), 0)


class TestDatabaseSchemas(unittest.TestCase):
    """Test the learned database schemas."""

    def setUp(self):
        self.kb = CodebaseKnowledge()

    def test_schemas_initialized(self):
        """Database schemas should be initialized."""
        self.assertGreater(len(self.kb.database_schemas), 0)

    def test_memories_schema(self):
        """Memories schema should have correct structure."""
        schema = self.kb.get_schema("memories")
        self.assertIsNotNone(schema)
        self.assertEqual(schema.table_name, "memories")
        self.assertIn("key", schema.columns)
        self.assertIn("value", schema.columns)
        self.assertIn("importance", schema.columns)

    def test_knowledge_nodes_schema(self):
        """Knowledge nodes schema should have embeddings."""
        schema = self.kb.get_schema("knowledge_nodes")
        self.assertIsNotNone(schema)
        self.assertIn("embedding", schema.columns)

    def test_l104_tasks_schema(self):
        """L104 tasks schema should exist."""
        schema = self.kb.get_schema("l104_tasks")
        self.assertIsNotNone(schema)
        self.assertEqual(schema.table_name, "tasks")


class TestAlgorithmPatterns(unittest.TestCase):
    """Test the learned algorithm patterns."""

    def setUp(self):
        self.kb = CodebaseKnowledge()

    def test_algorithms_initialized(self):
        """Algorithm patterns should be initialized."""
        self.assertGreater(len(self.kb.algorithm_patterns), 0)

    def test_god_code_derivation_algo(self):
        """GOD_CODE_DERIVATION algorithm should exist."""
        algo = self.kb.get_algorithm("GOD_CODE_DERIVATION")
        self.assertIsNotNone(algo)
        self.assertEqual(algo.resonance, 1.0)  # Perfect resonance

    def test_fibonacci_anyon_algo(self):
        """FIBONACCI_ANYON_FUSION algorithm should exist."""
        algo = self.kb.get_algorithm("FIBONACCI_ANYON_FUSION")
        self.assertIsNotNone(algo)
        self.assertIn("τ = 1/φ", algo.formula)

    def test_shannon_entropy_algo(self):
        """SHANNON_ENTROPY algorithm should exist."""
        algo = self.kb.get_algorithm("SHANNON_ENTROPY")
        self.assertIsNotNone(algo)
        self.assertEqual(algo.complexity, "O(n)")

    def test_resonant_algorithms(self):
        """Should find algorithms with high resonance."""
        resonant = self.kb.get_resonant_algorithms(0.9)
        self.assertGreater(len(resonant), 0)
        for algo in resonant:
            self.assertGreaterEqual(algo.resonance, 0.9)


class TestPatternSearch(unittest.TestCase):
    """Test the pattern search functionality."""

    def setUp(self):
        self.kb = CodebaseKnowledge()

    def test_search_consciousness(self):
        """Search for 'consciousness' should find relevant patterns."""
        results = self.kb.search_patterns("consciousness")
        self.assertGreater(len(results), 0)
        # Should find LAYERED_CONSCIOUSNESS
        names = [r[0] for r in results]
        self.assertIn("LAYERED_CONSCIOUSNESS", names)

    def test_search_quantum(self):
        """Search for 'quantum' should find topological patterns."""
        results = self.kb.search_patterns("quantum")
        self.assertGreater(len(results), 0)

    def test_search_database(self):
        """Search for 'graph' should find knowledge graph patterns."""
        results = self.kb.search_patterns("graph")
        self.assertGreater(len(results), 0)

    def test_search_returns_scored_results(self):
        """Search results should include relevance scores."""
        results = self.kb.search_patterns("learning")
        if results:
            name, pattern, score = results[0]
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0)


class TestModuleTemplateGeneration(unittest.TestCase):
    """Test the module template generation."""

    def setUp(self):
        self.kb = CodebaseKnowledge()

    def test_template_includes_header(self):
        """Generated template should include L104 header."""
        template = self.kb.generate_module_template("test_module", "Test description")
        self.assertIn("# [L104_TEST_MODULE]", template)
        self.assertIn("INVARIANT: 527.5184818492612", template)

    def test_template_includes_class(self):
        """Generated template should include a class."""
        template = self.kb.generate_module_template("example", "Example module")
        self.assertIn("class Example:", template)

    def test_template_includes_singleton(self):
        """Generated template should include global instance."""
        template = self.kb.generate_module_template("my_module", "My module")
        self.assertIn("my_module = MyModule()", template)

    def test_template_includes_god_code(self):
        """Generated template should reference GOD_CODE."""
        template = self.kb.generate_module_template("test", "Test")
        self.assertIn("self.god_code = 527.5184818492612", template)


class TestKnowledgeExport(unittest.TestCase):
    """Test knowledge export functionality."""

    def setUp(self):
        self.kb = CodebaseKnowledge()

    def test_export_returns_dict(self):
        """Export should return a dictionary."""
        export = self.kb.export_knowledge()
        self.assertIsInstance(export, dict)

    def test_export_includes_constants(self):
        """Export should include constants."""
        export = self.kb.export_knowledge()
        self.assertIn("constants", export)
        self.assertIn("GOD_CODE", export["constants"])

    def test_export_includes_patterns(self):
        """Export should include patterns."""
        export = self.kb.export_knowledge()
        self.assertIn("patterns", export)
        self.assertGreater(len(export["patterns"]), 0)

    def test_export_includes_statistics(self):
        """Export should include statistics."""
        export = self.kb.export_knowledge()
        self.assertIn("statistics", export)

    def test_export_is_json_serializable(self):
        """Export should be JSON serializable."""
        import json
        export = self.kb.export_knowledge()
        json_str = json.dumps(export)
        self.assertIsInstance(json_str, str)


class TestStatistics(unittest.TestCase):
    """Test knowledge base statistics."""

    def setUp(self):
        self.kb = CodebaseKnowledge()

    def test_statistics_structure(self):
        """Statistics should have expected structure."""
        stats = self.kb.get_statistics()
        self.assertIn("architectural_patterns", stats)
        self.assertIn("database_schemas", stats)
        self.assertIn("algorithm_patterns", stats)
        self.assertIn("total_design_principles", stats)
        self.assertIn("avg_resonance", stats)
        self.assertIn("god_code_alignment", stats)

    def test_statistics_values(self):
        """Statistics values should be positive."""
        stats = self.kb.get_statistics()
        self.assertGreater(stats["architectural_patterns"], 0)
        self.assertGreater(stats["database_schemas"], 0)
        self.assertGreater(stats["algorithm_patterns"], 0)

    def test_god_code_alignment(self):
        """God code alignment should match constant."""
        stats = self.kb.get_statistics()
        self.assertEqual(stats["god_code_alignment"], L104Constants.GOD_CODE)


class TestGlobalInstance(unittest.TestCase):
    """Test the global codebase_knowledge instance."""

    def test_global_instance_exists(self):
        """Global instance should exist."""
        self.assertIsNotNone(codebase_knowledge)

    def test_global_instance_initialized(self):
        """Global instance should be initialized."""
        self.assertGreater(len(codebase_knowledge.architectural_patterns), 0)

    def test_global_instance_is_codebase_knowledge(self):
        """Global instance should be CodebaseKnowledge type."""
        self.assertIsInstance(codebase_knowledge, CodebaseKnowledge)


if __name__ == "__main__":
    # Run tests with verbosity
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ⟨Σ_L104⟩  CODEBASE KNOWLEDGE TEST SUITE                    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    unittest.main(verbosity=2)
