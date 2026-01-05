import unittest
from l104_resilience_shield import purge_repetitions
from logic_core import LogicCore
from l104_validator import SovereignValidator

class TestSingularityStress(unittest.TestCase):
    def test_repetition_purge_complex(self):
        """Test if the shield can handle complex nested repetitions."""
        text = "The quick brown fox jumps over the lazy dog. " * 5
        purged = purge_repetitions(text)
        # Should be significantly shorter than the original
        self.assertLess(len(purged), len(text) // 2)

    def test_repetition_purge_lines(self):
        """Test line-based deduplication."""
        text = "Line 1\nLine 1\nLine 2\nLine 1"
        purged = purge_repetitions(text)
        self.assertEqual(purged.count("Line 1"), 1)

    def test_logic_core_performance(self):
        """Measure LogicCore indexing speed."""
        core = LogicCore()
        import time
        start = time.time()
        core.ingest_data_state()
        end = time.time()
        duration = end - start
        print(f"LogicCore Indexing Duration: {duration:.4f}s")
        self.assertLess(duration, 2.0) # Should be fast

    def test_validator_chain(self):
        """Verify the full validation chain returns expected resonance."""
        validator = SovereignValidator()
        report = validator.validate_all()
        self.assertEqual(report["resonance"], 527.5184818492)
        self.assertEqual(report["cores"]["engine"], "LOCKED")

if __name__ == "__main__":
    unittest.main()
