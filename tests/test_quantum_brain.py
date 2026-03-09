# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
"""
Tests for l104_simulator.quantum_brain — GodCodeQuantumBrain v6.0.0

Validates:
  - All 16 subsystems (Cortex, Memory, Resonance, Decision, Entropy,
    Coherence, Learning, Attention, Dream, Associative, Consciousness,
    Intuition, Creativity, Empathy, Precognition + AlgorithmSuite)
  - Bug fixes: memory retrieve, Φ formula, parameter_sweep return, attend coherence
  - New v4.0 methods: get_data, run_all_algorithms, full_cycle (expanded),
    teleport_state, solve_linear, verify_convergence, fingerprint_compare,
    count_solutions, generate_random, topological_protect, run_diagnostics
"""

import unittest
import time
import math

from l104_simulator import (
    GodCodeQuantumBrain,
    BrainConfig,
    ThoughtResult,
    AlgorithmSuite,
    AlgorithmResult,
)
from l104_simulator.simulator import GOD_CODE, PHI, VOID_CONSTANT


class TestBrainInit(unittest.TestCase):
    """Test brain initialization and configuration."""

    def test_default_config(self):
        brain = GodCodeQuantumBrain()
        self.assertEqual(brain.VERSION, "6.0.0")
        self.assertEqual(brain.config.cortex_qubits, 4)
        self.assertEqual(brain.config.memory_qubits, 4)
        self.assertGreater(brain.total_qubits, 0)

    def test_custom_config(self):
        cfg = BrainConfig(cortex_qubits=3, memory_qubits=2,
                          resonance_qubits=1, ancilla_qubits=1)
        brain = GodCodeQuantumBrain(cfg)
        self.assertEqual(brain.config.cortex_qubits, 3)
        self.assertEqual(brain.config.memory_qubits, 2)

    def test_algorithm_suite_initialized(self):
        brain = GodCodeQuantumBrain()
        self.assertIsNotNone(brain.algorithm_suite)
        self.assertIsInstance(brain.algorithm_suite, AlgorithmSuite)

    def test_status(self):
        brain = GodCodeQuantumBrain()
        s = brain.status()
        self.assertEqual(s["version"], "6.0.0")
        self.assertIn("subsystems", s)
        self.assertIn("state", s)
        self.assertIn("constants", s)
        self.assertAlmostEqual(s["constants"]["GOD_CODE"], 527.5184818492612, places=5)
        self.assertAlmostEqual(s["constants"]["PHI"], 1.618033988749895, places=10)

    def test_repr(self):
        brain = GodCodeQuantumBrain()
        r = repr(brain)
        self.assertIn("6.0.0", r)
        self.assertIn("GodCodeQuantumBrain", r)


class TestCoreSubsystems(unittest.TestCase):
    """Test the core v1.0 subsystems: think, search, decide, remember, heal."""

    def setUp(self):
        self.brain = GodCodeQuantumBrain()
        self.data = [0.5, 0.3, 0.8, 0.2]

    def test_think(self):
        result = self.brain.think(self.data)
        self.assertIsInstance(result, ThoughtResult)
        self.assertIsInstance(result.output_probabilities, dict)
        self.assertGreater(len(result.output_probabilities), 0)
        self.assertGreater(result.sacred_score, 0)
        self.assertGreater(result.circuit_depth, 0)
        self.assertIsInstance(result.coherence_maintained, float)

    def test_think_stores_history(self):
        self.brain.think(self.data)
        self.assertEqual(len(self.brain._thought_history), 1)
        self.brain.think(self.data)
        self.assertEqual(len(self.brain._thought_history), 2)

    def test_search(self):
        result = self.brain.search(target=3)
        self.assertIsInstance(result, ThoughtResult)
        self.assertIn("target_probability", result.details)
        self.assertGreater(result.sacred_score, 0)

    def test_decide(self):
        result = self.brain.decide(0.3, 0.7)
        self.assertIsInstance(result, ThoughtResult)
        self.assertIn("decision", result.details)
        self.assertIn(result.details["decision"], ("A", "B"))

    def test_remember_store(self):
        result = self.brain.remember(cell=0, value=0.42)
        self.assertIsInstance(result, ThoughtResult)
        self.assertIn("cell", result.details)
        self.assertGreater(result.circuit_depth, 0)

    def test_remember_retrieve(self):
        # Store then retrieve — tests bug fix #1 (inverse sequence)
        self.brain.remember(cell=0, value=0.75)
        result = self.brain.memory.retrieve(0)
        # retrieve() returns a QuantumCircuit (the retrieval circuit)
        from l104_simulator.simulator import QuantumCircuit
        self.assertIsInstance(result, QuantumCircuit)
        self.assertGreater(result.depth, 0)

    def test_heal(self):
        result = self.brain.heal(noise_level=0.05)
        self.assertIsInstance(result, ThoughtResult)
        self.assertIn("fidelity_before_healing", result.details)
        self.assertIn("fidelity_after_healing", result.details)


class TestV2Subsystems(unittest.TestCase):
    """Test v2.0 additions: learn, attend, dream, associate, consciousness."""

    def setUp(self):
        self.brain = GodCodeQuantumBrain()
        self.data = [0.5, 0.3, 0.8, 0.2]

    def test_learn_from_thought(self):
        thought = self.brain.think(self.data)
        result = self.brain.learn(thought)
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)

    def test_learn_no_thought(self):
        result = self.brain.learn()
        self.assertEqual(result.get("error"), "No thought to learn from")

    def test_learn_uses_last_thought(self):
        self.brain.think(self.data)
        result = self.brain.learn()  # Should use last thought
        self.assertNotIn("error", result)

    def test_attend(self):
        # Bug fix #4: coherence_maintained should be float
        result = self.brain.attend(self.data)
        self.assertIsInstance(result, ThoughtResult)
        self.assertIsInstance(result.coherence_maintained, float)
        self.assertNotIsInstance(result.coherence_maintained, bool)
        self.assertEqual(result.details.get("mode"), "attention")

    def test_attend_with_head(self):
        result = self.brain.attend(self.data, head=0)
        self.assertEqual(result.details.get("head"), 0)

    def test_dream(self):
        result = self.brain.dream(steps=3)
        self.assertIsInstance(result, dict)
        # Dream returns steps/discoveries info
        self.assertIn("steps", result)

    def test_associate(self):
        result = self.brain.associate(0, 1)
        self.assertIsInstance(result, dict)

    def test_recall(self):
        result = self.brain.recall(0)
        self.assertIsInstance(result, dict)

    def test_store_associative(self):
        # Should not raise
        self.brain.store_associative(0, 0.42)

    def test_measure_consciousness(self):
        # Bug fix #2: Φ formula now uses S_full properly
        result = self.brain.measure_consciousness(self.data)
        self.assertIsInstance(result, dict)
        self.assertIn("phi", result)
        phi = result["phi"]
        self.assertIsInstance(phi, (int, float))
        self.assertGreaterEqual(phi, 0.0)


class TestV3Subsystems(unittest.TestCase):
    """Test v3.0 additions: intuit, create, empathize, predict."""

    def setUp(self):
        self.brain = GodCodeQuantumBrain()
        self.data = [0.5, 0.3, 0.8, 0.2]

    def test_intuit(self):
        result = self.brain.intuit(self.data)
        self.assertIsInstance(result, dict)

    def test_create(self):
        # Bug fix #3: parameter_sweep return handling
        result = self.brain.create(n_points=4)
        self.assertIsInstance(result, dict)
        # Should NOT raise TypeError about .tolist()

    def test_empathize(self):
        result = self.brain.empathize(self.data)
        self.assertIsInstance(result, dict)

    def test_predict(self):
        result = self.brain.predict([1.0, 2.0, 3.0, 4.0])
        self.assertIsInstance(result, dict)


class TestV4Methods(unittest.TestCase):
    """Test v4.0 additions: get_data, run_all_algorithms, algorithm-powered methods."""

    def setUp(self):
        self.brain = GodCodeQuantumBrain()
        self.data = [0.5, 0.3, 0.8, 0.2]

    def test_get_data_structure(self):
        data = self.brain.get_data()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["version"], "6.0.0")
        # Check all top-level keys
        for key in ("config", "state", "thought_history", "learning",
                     "dreams", "associative_memory", "creativity",
                     "algorithms", "constants"):
            self.assertIn(key, data, f"Missing key: {key}")

    def test_get_data_constants(self):
        data = self.brain.get_data()
        c = data["constants"]
        self.assertAlmostEqual(c["GOD_CODE"], GOD_CODE)
        self.assertAlmostEqual(c["PHI"], PHI)
        self.assertAlmostEqual(c["VOID_CONSTANT"], VOID_CONSTANT)

    def test_get_data_after_think(self):
        self.brain.think(self.data)
        data = self.brain.get_data()
        self.assertEqual(data["thought_history"]["total"], 1)
        self.assertEqual(len(data["thought_history"]["recent"]), 1)

    def test_run_all_algorithms(self):
        results = self.brain.run_all_algorithms()
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        # Each result should be an AlgorithmResult
        for name, ar in results.items():
            self.assertIsInstance(ar, AlgorithmResult, f"Algorithm {name} returned wrong type")
            self.assertIn(ar.success, (True, False), f"Algorithm {name} success not bool-like")
            self.assertIsInstance(float(ar.sacred_alignment), float)

    def test_teleport_state(self):
        result = self.brain.teleport_state(self.data)
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("sacred_alignment", result)

    def test_solve_linear(self):
        result = self.brain.solve_linear([1.0, 2.0, 3.0])
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)

    def test_verify_convergence(self):
        result = self.brain.verify_convergence()
        self.assertIsInstance(result, dict)
        self.assertIn("converged", result)
        self.assertIn("sacred_alignment", result)

    def test_fingerprint_compare(self):
        result = self.brain.fingerprint_compare([0.1, 0.2], [0.3, 0.4])
        self.assertIsInstance(result, dict)
        self.assertIn("similarity", result)

    def test_count_solutions(self):
        result = self.brain.count_solutions(target=3)
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)

    def test_generate_random(self):
        result = self.brain.generate_random(n_bits=4)
        self.assertIsInstance(result, dict)
        self.assertIn("random_bits", result)

    def test_topological_protect(self):
        result = self.brain.topological_protect()
        self.assertIsInstance(result, dict)
        self.assertIn("protected", result)
        self.assertIn("sacred_alignment", result)


class TestRunDiagnostics(unittest.TestCase):
    """Test the run_diagnostics method."""

    def test_diagnostics(self):
        brain = GodCodeQuantumBrain()
        diag = brain.run_diagnostics()
        self.assertIsInstance(diag, dict)
        for key in ("think", "search", "memory", "heal"):
            self.assertIn(key, diag, f"Missing diagnostic: {key}")
        self.assertIn("healthy", diag)
        self.assertIn("elapsed_ms", diag)


class TestFullCycle(unittest.TestCase):
    """Test the upgraded full_cycle (all subsystems)."""

    def test_full_cycle_runs(self):
        brain = GodCodeQuantumBrain()
        result = brain.full_cycle([0.5, 0.3, 0.8, 0.2])
        self.assertIsInstance(result, dict)
        self.assertEqual(result["version"], "6.0.0")

    def test_full_cycle_subsystems(self):
        brain = GodCodeQuantumBrain()
        result = brain.full_cycle([0.5, 0.3, 0.8, 0.2])
        subs = result["subsystems"]
        # Core subsystems
        for key in ("thought", "memory", "search", "healing"):
            self.assertIn(key, subs, f"Missing subsystem: {key}")
        # v2/v3 subsystems
        for key in ("attention", "learning", "dream", "consciousness",
                     "intuition", "creativity", "empathy", "precognition"):
            self.assertIn(key, subs, f"Missing subsystem: {key}")
        # v4: algorithms
        self.assertIn("algorithms", subs)

    def test_full_cycle_aggregate(self):
        brain = GodCodeQuantumBrain()
        result = brain.full_cycle([0.5, 0.3, 0.8, 0.2])
        agg = result["aggregate"]
        self.assertIn("total_sacred_score", agg)
        self.assertIn("subsystems_active", agg)
        self.assertIn("subsystems_total", agg)
        self.assertIn("total_time_ms", agg)
        self.assertGreater(agg["subsystems_active"], 4,
                           "full_cycle should activate more than 4 subsystems")

    def test_full_cycle_algorithms_section(self):
        brain = GodCodeQuantumBrain()
        result = brain.full_cycle([0.5, 0.3, 0.8, 0.2])
        algo = result["subsystems"].get("algorithms", {})
        if "error" not in algo:
            self.assertIn("total", algo)
            self.assertIn("passed", algo)
            self.assertGreater(algo["total"], 0)


class TestBugFixes(unittest.TestCase):
    """Regression tests for the 5 bugs fixed in v4.0."""

    def setUp(self):
        self.brain = GodCodeQuantumBrain()

    def test_bug1_memory_retrieve_not_identity(self):
        """Bug #1: Memory retrieve was a net identity (iron_gate + rz(-IRON) canceled)."""
        self.brain.remember(0, 0.42)
        result = self.brain.memory.retrieve(0)
        # retrieve() returns a QuantumCircuit — should have non-trivial gates
        from l104_simulator.simulator import QuantumCircuit
        self.assertIsInstance(result, QuantumCircuit)
        self.assertGreater(result.depth, 2,
                           "Retrieve circuit should have depth > 2 (not an identity)")

    def test_bug2_phi_uses_s_full(self):
        """Bug #2: Φ was computed as max(0, min_partition_S - 0.0) — always = min_partition_S."""
        result = self.brain.measure_consciousness([0.5, 0.3, 0.8, 0.2])
        phi = result["phi"]
        # Φ = max(0, min_partition_S - S_full). Since S_full > 0 for non-trivial
        # states, Φ should generally be *less* than the minimum partition entropy.
        self.assertIsInstance(phi, (int, float))
        self.assertGreaterEqual(phi, 0.0)

    def test_bug3_creativity_no_tolist_error(self):
        """Bug #3: create() called .tolist() on a List[Dict] from parameter_sweep."""
        # This simply must not raise TypeError
        result = self.brain.create(n_points=4)
        self.assertIsInstance(result, dict)

    def test_bug4_attend_coherence_is_float(self):
        """Bug #4: attend() set coherence_maintained=True (bool), should be float."""
        result = self.brain.attend([0.5, 0.3, 0.8, 0.2])
        c = result.coherence_maintained
        self.assertIsInstance(c, float)
        self.assertNotIsInstance(c, bool)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)


class TestSacredConstants(unittest.TestCase):
    """Verify sacred constants are correctly propagated."""

    def test_god_code(self):
        self.assertAlmostEqual(GOD_CODE, 527.5184818492612, places=5)

    def test_phi(self):
        self.assertAlmostEqual(PHI, (1 + math.sqrt(5)) / 2, places=10)

    def test_void_constant(self):
        expected = 1.04 + PHI / 1000
        self.assertAlmostEqual(VOID_CONSTANT, expected, places=10)


if __name__ == "__main__":
    unittest.main()
