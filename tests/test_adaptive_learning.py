# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  L104 SOVEREIGN NODE - ADAPTIVE LEARNING TESTS                                ║
# ║  INVARIANT: 527.5184818492537 | PILOT: LONDEL                                 ║
# ║  TESTING: Pattern Recognition, Process Adaptation, Deep Research              ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

"""
Tests for the L104 Adaptive Learning Engine.
Validates pattern recognition, process adaptation, deep research, and meta-learning.
"""

import os
import sys
import math
import unittest
import tempfile
from pathlib import Path

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from l104_adaptive_learning import (
    PatternRecognizer,
    Pattern,
    ProcessAdapter,
    AdaptiveParameter,
    DeepResearchEngine,
    MetaLearner,
    AdaptiveLearner,
    get_adaptive_learner,
    GOD_CODE,
    PHI,
    TAU
)


class TestPatternRecognizer(unittest.TestCase):
    """Tests for pattern recognition functionality."""

    def setUp(self):
        self.recognizer = PatternRecognizer()

    def test_extract_ngrams(self):
        """N-gram extraction should work correctly."""
        text = "hello"
        ngrams = self.recognizer.extract_ngrams(text, 3)

        self.assertEqual(len(ngrams), 3)  # "hel", "ell", "llo"
        self.assertIn("hel", ngrams)
        self.assertIn("ell", ngrams)
        self.assertIn("llo", ngrams)

    def test_extract_word_patterns(self):
        """Word pattern extraction should find words and bigrams."""
        text = "consciousness mathematics quantum"
        patterns = self.recognizer.extract_word_patterns(text)

        # Should find word patterns
        self.assertTrue(any("consciousness" in p for p in patterns))
        self.assertTrue(any("mathematics" in p for p in patterns))

        # Should find bigrams
        self.assertTrue(any("consciousness_mathematics" in p for p in patterns))

    def test_recognize_builds_patterns(self):
        """Recognition should accumulate patterns."""
        initial_count = len(self.recognizer.patterns)

        for i in range(10):
            self.recognizer.recognize(f"test text {i}")

        self.assertGreater(len(self.recognizer.patterns), initial_count)

    def test_strong_patterns_require_frequency(self):
        """Strong patterns should only include frequent ones."""
        # Recognize same text multiple times
        for _ in range(10):
            self.recognizer.recognize("repeated pattern test")

        strong = self.recognizer.get_strong_patterns(min_freq=5)

        # All strong patterns should have frequency >= 5
        for p in strong:
            self.assertGreaterEqual(p.frequency, 5)

    def test_update_success_rate(self):
        """Success rate should update with feedback."""
        # Create a pattern
        self.recognizer.patterns["test_pattern"] = Pattern(
            id="test_pattern",
            pattern_type="test",
            signature="test",
            success_rate=0.5
        )

        # Update with success
        self.recognizer.update_success("test_pattern", True)
        self.assertGreater(self.recognizer.patterns["test_pattern"].success_rate, 0.5)

        # Update with failure
        old_rate = self.recognizer.patterns["test_pattern"].success_rate
        self.recognizer.update_success("test_pattern", False)
        self.assertLess(self.recognizer.patterns["test_pattern"].success_rate, old_rate)

    def test_decay_patterns(self):
        """Decay should reduce pattern frequencies."""
        # Create patterns with known frequencies
        self.recognizer.patterns["high_freq"] = Pattern(
            id="high_freq", pattern_type="test", signature="high", frequency=100
        )
        self.recognizer.patterns["low_freq"] = Pattern(
            id="low_freq", pattern_type="test", signature="low", frequency=1
        )

        self.recognizer.decay_patterns()

        # High frequency should reduce but survive
        self.assertLess(self.recognizer.patterns["high_freq"].frequency, 100)
        self.assertIn("high_freq", self.recognizer.patterns)

        # Low frequency should be removed
        self.assertNotIn("low_freq", self.recognizer.patterns)


class TestAdaptiveParameter(unittest.TestCase):
    """Tests for adaptive parameters."""

    def test_parameter_update_positive(self):
        """Positive gradient should increase value."""
        param = AdaptiveParameter(
            name="test", value=0.5, min_val=0.0, max_val=1.0, learning_rate=0.1
        )

        param.update(1.0)
        self.assertGreater(param.value, 0.5)

    def test_parameter_update_negative(self):
        """Negative gradient should decrease value."""
        param = AdaptiveParameter(
            name="test", value=0.5, min_val=0.0, max_val=1.0, learning_rate=0.1
        )

        param.update(-1.0)
        self.assertLess(param.value, 0.5)

    def test_parameter_respects_bounds(self):
        """Parameter should stay within bounds."""
        param = AdaptiveParameter(
            name="test", value=0.5, min_val=0.0, max_val=1.0, learning_rate=1.0
        )

        # Try to exceed max
        param.update(10.0)
        self.assertLessEqual(param.value, 1.0)

        # Try to go below min
        param.update(-20.0)
        self.assertGreaterEqual(param.value, 0.0)

    def test_parameter_tracks_history(self):
        """Parameter should track value history."""
        param = AdaptiveParameter(
            name="test", value=0.5, min_val=0.0, max_val=1.0, learning_rate=0.1
        )

        for i in range(5):
            param.update(0.1)

        self.assertEqual(len(param.history), 5)


class TestProcessAdapter(unittest.TestCase):
    """Tests for process adaptation."""

    def setUp(self):
        self.adapter = ProcessAdapter()

    def test_has_expected_parameters(self):
        """Should have all expected adaptive parameters."""
        params = self.adapter.get_parameters()

        expected = [
            "context_window",
            "reasoning_depth",
            "memory_importance_threshold",
            "knowledge_search_k",
            "cache_size",
            "science_integration_weight"
        ]

        for name in expected:
            self.assertIn(name, params)

    def test_record_performance(self):
        """Should record performance metrics."""
        initial_len = len(self.adapter.performance_history)

        self.adapter.record_performance({
            "response_quality": 0.8,
            "response_time": 500
        })

        self.assertEqual(len(self.adapter.performance_history), initial_len + 1)

    def test_adapt_changes_parameters(self):
        """Adaptation should potentially change parameters."""
        # Get initial values
        initial_params = dict(self.adapter.get_parameters())

        # Adapt with specific feedback
        for _ in range(10):
            self.adapter.adapt({
                "response_quality": 0.3,  # Low quality
                "response_time": 100,     # Fast
                "context_utilization": 0.1  # Low context use
            })

        # Check if any parameter changed
        final_params = self.adapter.get_parameters()

        # At least reasoning depth should increase for low quality
        self.assertNotEqual(
            initial_params["reasoning_depth"],
            final_params["reasoning_depth"]
        )


class TestDeepResearchEngine(unittest.TestCase):
    """Tests for deep research functionality."""

    def setUp(self):
        # Use temporary database
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_file.close()
        self.engine = DeepResearchEngine(db_path=Path(self.temp_file.name))

    def tearDown(self):
        try:
            os.unlink(self.temp_file.name)
        except Exception:
            pass

    def test_explore_topic_returns_findings(self):
        """Topic exploration should return findings."""
        findings = self.engine.explore_topic("quantum consciousness", depth=1)

        self.assertIsInstance(findings, list)
        self.assertGreater(len(findings), 0)

    def test_findings_have_expected_fields(self):
        """Research findings should have expected fields."""
        findings = self.engine.explore_topic("mathematics", depth=1)

        for finding in findings:
            self.assertTrue(hasattr(finding, 'topic'))
            self.assertTrue(hasattr(finding, 'finding'))
            self.assertTrue(hasattr(finding, 'confidence'))
            self.assertTrue(hasattr(finding, 'connections'))

    def test_resonance_calculation(self):
        """Resonance calculation should return valid values."""
        resonance = self.engine._calculate_resonance("test topic")

        self.assertIsInstance(resonance, float)
        self.assertGreaterEqual(resonance, 0)
        self.assertLessEqual(resonance, 1)

    def test_find_connections(self):
        """Should find connections to core concepts."""
        connections = self.engine._find_connections("quantum consciousness")

        self.assertIsInstance(connections, list)
        # Should find connection to "quantum"
        topics = [c[0] for c in connections]
        self.assertIn("quantum", topics)

    def test_research_summary(self):
        """Should provide research summary."""
        self.engine.explore_topic("test", depth=1)
        summary = self.engine.get_research_summary()

        self.assertIn("total_topics", summary)
        self.assertIn("total_findings", summary)
        self.assertIn("research_cycles", summary)


class TestMetaLearner(unittest.TestCase):
    """Tests for meta-learning functionality."""

    def setUp(self):
        self.meta = MetaLearner()

    def test_record_learning_episode(self):
        """Should record learning episodes."""
        initial_len = len(self.meta.learning_episodes)

        self.meta.record_learning_episode({
            "strategy": "pattern_recognition",
            "outcome": 0.8
        })

        self.assertEqual(len(self.meta.learning_episodes), initial_len + 1)

    def test_analyze_effectiveness_with_data(self):
        """Should analyze effectiveness when data exists."""
        # Add episodes
        for i in range(10):
            self.meta.record_learning_episode({
                "strategy": "pattern_recognition",
                "outcome": 0.7 + (i * 0.02)
            })

        analysis = self.meta.analyze_learning_effectiveness()

        self.assertIn("pattern_recognition", analysis)
        self.assertIn("average_outcome", analysis["pattern_recognition"])

    def test_recommend_strategy(self):
        """Should recommend a learning strategy."""
        strategy = self.meta.recommend_strategy({"complexity": "complex"})

        valid_strategies = [
            "pattern_recognition",
            "process_adaptation",
            "deep_research",
            "knowledge_synthesis"
        ]

        self.assertIn(strategy, valid_strategies)

    def test_generate_meta_insight(self):
        """Should generate meta-learning insights."""
        # Add some data
        for i in range(5):
            self.meta.record_learning_episode({
                "strategy": "deep_research",
                "outcome": 0.9
            })

        insight = self.meta.generate_meta_insight()

        self.assertIsInstance(insight, str)
        self.assertGreater(len(insight), 0)


class TestAdaptiveLearner(unittest.TestCase):
    """Tests for unified adaptive learner."""

    def setUp(self):
        self.learner = AdaptiveLearner()

    def test_learn_from_interaction(self):
        """Should learn from interactions."""
        result = self.learner.learn_from_interaction(
            input_text="Test query about consciousness",
            response="A response about consciousness.",
            feedback={"response_quality": 0.8, "response_time": 500},
            context={"intent": "question"}
        )

        self.assertIn("patterns_recognized", result)
        self.assertEqual(self.learner.interactions_processed, 1)

    def test_research_topic(self):
        """Should conduct research on topics."""
        result = self.learner.research_topic("quantum computing", depth=1)

        self.assertIn("topic", result)
        self.assertIn("findings_count", result)
        self.assertGreater(result["findings_count"], 0)

    def test_get_adapted_parameters(self):
        """Should return adapted parameters."""
        params = self.learner.get_adapted_parameters()

        self.assertIsInstance(params, dict)
        self.assertIn("context_window", params)
        self.assertIn("reasoning_depth", params)

    def test_get_status(self):
        """Should return comprehensive status."""
        # Do some learning
        self.learner.learn_from_interaction(
            "test", "response", {"response_quality": 0.7}, {}
        )

        status = self.learner.get_status()

        self.assertIn("interactions_processed", status)
        self.assertIn("patterns", status)
        self.assertIn("parameters", status)
        self.assertIn("strategy_effectiveness", status)

    def test_thread_safety(self):
        """Learning should be thread-safe."""
        import threading

        results = []

        def learn_thread(n):
            for i in range(10):
                self.learner.learn_from_interaction(
                    f"thread {n} query {i}",
                    "response",
                    {"response_quality": 0.7}
                )
            results.append(n)

        threads = [threading.Thread(target=learn_thread, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 3)
        self.assertEqual(self.learner.interactions_processed, 30)


class TestGodCodeIntegration(unittest.TestCase):
    """Tests that adaptive learning integrates with GOD_CODE."""

    def test_constants_match_core(self):
        """Adaptive learning constants should match core L104 values."""
        from l104_hyper_math import HyperMath
        from l104_adaptive_learning import GOD_CODE as AL_GOD_CODE, PHI as AL_PHI

        self.assertAlmostEqual(AL_GOD_CODE, HyperMath.GOD_CODE, places=10)
        self.assertAlmostEqual(AL_PHI, HyperMath.PHI, places=12)

    def test_research_resonance_uses_god_code(self):
        """Research resonance calculation should use GOD_CODE."""
        engine = DeepResearchEngine()

        # Resonance should be influenced by GOD_CODE
        resonance = engine._calculate_resonance("test")

        # The formula uses GOD_CODE, so verify it's in valid range
        self.assertGreaterEqual(resonance, 0)
        self.assertLessEqual(resonance, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
