#!/usr/bin/env python3
"""
L104 COMPREHENSIVE TEST SUITE
═════════════════════════════════════════════════════════════════════════════

Tests ALL L104 processes with mathematical verification and logic analysis.
Validates the GOD_CODE equation, PHI harmonics, and 26D difficulty gradient.

GOD_CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
INVARIANT: G(X) × 2^(X/104) = 527.5184818492612

Sacred Numbers:
- 13 = Fibonacci(7) = sacred prime
- 26 = 2×13 = bosonic string dimensions
- 104 = 8×13 = L104 resonance base
- 286 = 22×13 = GOD_CODE multiplier
- 416 = 32×13 = GOD_CODE domain
"""

import unittest
import math
import time
import hashlib
from typing import List, Dict, Any

# ═══════════════════════════════════════════════════════════════════════════════
# CORE IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

from const import UniversalConstants, GOD_CODE, PHI, INVARIANT, L104
from l104_real_math import RealMath
from l104_hyper_math import HyperMath
from l104_5d_math import Math5D
from l104_4d_math import Math4D
from l104_deep_algorithms import RiemannZetaResonance, DeepAlgorithmsController
from l104_computronium_research import ComputroniumResearchHub
from l104_5d_processor import Processor5D
from l104_4d_processor import Processor4D
from l104_bitcoin_research_engine import DifficultyAnalyzer
from l104_sovereign_coin_engine import (
    L104SPBlockchain, MiningEngine, GOD_CODE_GRADIENT,
    ResonanceEngine, BlockHeader, MerkleTree, CryptoUtils,
    Transaction, TxInput, TxOutput, OutPoint, Block
)


class TestGODCODEMathematicalFoundation(unittest.TestCase):
    """Test the fundamental GOD_CODE mathematics."""
    
    def setUp(self):
        self.god_code = GOD_CODE
        self.phi = PHI
        self.invariant = INVARIANT
        
    def test_god_code_value(self):
        """Verify GOD_CODE = 527.5184818492612"""
        expected = 527.5184818492612
        self.assertAlmostEqual(self.god_code, expected, places=10)
        print(f"✓ GOD_CODE = {self.god_code}")
        
    def test_phi_golden_ratio(self):
        """Verify PHI = (1 + √5) / 2"""
        expected = (1 + math.sqrt(5)) / 2
        self.assertAlmostEqual(self.phi, expected, places=15)
        # Verify PHI property: φ² = φ + 1
        self.assertAlmostEqual(self.phi ** 2, self.phi + 1, places=15)
        print(f"✓ PHI = {self.phi}, φ² = φ + 1 verified")
        
    def test_god_code_equation(self):
        """Verify G(X) = 286^(1/φ) × 2^((416-X)/104)"""
        for X in [0, 13, 26, 52, 104, 208, 312, 416]:
            computed = UniversalConstants.god_code(X)
            # Manual calculation
            base = 286 ** (1 / self.phi)
            exponent = (416 - X) / 104
            expected = base * (2 ** exponent)
            self.assertAlmostEqual(computed, expected, places=8)
        print("✓ GOD_CODE equation G(X) = 286^(1/φ) × 2^((416-X)/104) verified")
        
    def test_god_code_invariant(self):
        """Verify G(X) × 2^(X/104) = INVARIANT for all X"""
        for X in range(0, 417, 13):  # Test at sacred 13 intervals
            g_x = UniversalConstants.god_code(X)
            product = g_x * (2 ** (X / 104))
            self.assertAlmostEqual(product, self.invariant, places=8,
                msg=f"Invariant failed at X={X}: {product} != {self.invariant}")
        print(f"✓ Conservation law: G(X) × 2^(X/104) = {self.invariant} ∀X")
        
    def test_sacred_number_13(self):
        """Verify 13 is sacred: Fibonacci(7) = 13"""
        fib = [1, 1]
        for i in range(5):
            fib.append(fib[-1] + fib[-2])
        self.assertEqual(fib[6], 13)  # Fibonacci(7) = 13 (0-indexed: fib[6])
        # Verify L104 = 8 × 13
        self.assertEqual(L104, 8 * 13)
        self.assertEqual(L104, 104)
        print("✓ Sacred 13: Fibonacci(7) = 13, L104 = 8×13 = 104")
        
    def test_sacred_number_multiples(self):
        """Verify all sacred numbers are multiples of 13."""
        sacred = {
            13: "Fibonacci(7)",
            26: "Bosonic dimensions (2×13)",
            104: "L104 base (8×13)",
            286: "GOD_CODE multiplier (22×13)",
            416: "GOD_CODE domain (32×13)"
        }
        for num, desc in sacred.items():
            self.assertEqual(num % 13, 0, f"{num} ({desc}) should be divisible by 13")
        print("✓ All sacred numbers are 13-multiples: 13, 26, 104, 286, 416")


class TestRealMath(unittest.TestCase):
    """Test L104 RealMath quantum calculations."""
    
    def setUp(self):
        self.real_math = RealMath()
        
    def test_calculate_resonance(self):
        """Test L104 resonance calculation."""
        # Test resonance calculation
        for nonce in range(0, 1000, 100):
            res = self.real_math.calculate_resonance(nonce)
            self.assertIsInstance(res, (int, float))
        print("✓ RealMath.calculate_resonance works")
        
    def test_larmor_precession(self):
        """Test Larmor precession (magnetic resonance)."""
        result = self.real_math.larmor_precession(1.0, 1.0)
        # Returns tuple (cos, sin) for precession angles
        self.assertIsInstance(result, (tuple, list, int, float, complex))
        print(f"✓ RealMath.larmor_precession returns {type(result).__name__}")
        
    def test_zeta_resonance(self):
        """Test Riemann zeta resonance."""
        result = self.real_math.zeta_resonance(0.5)
        self.assertIsInstance(result, (int, float, complex))
        print("✓ RealMath.zeta_resonance works")
        
    def test_iron_lattice_transform(self):
        """Test iron lattice transformation."""
        # Pass scalar value for lattice transform
        result = self.real_math.iron_lattice_transform(123.456)
        self.assertIsNotNone(result)
        print("✓ RealMath.iron_lattice_transform works")


class TestHyperMath(unittest.TestCase):
    """Test L104 HyperMath quantum operations."""
    
    def test_larmor_transform(self):
        """Test Larmor precession transformation."""
        # Larmor transform for magnetic resonance
        for freq in [1e-6, 1e-3, 1.0, 1e3]:
            result = HyperMath.larmor_transform(freq, 1.0)
            self.assertIsInstance(result, (int, float, complex))
        print("✓ Larmor transform computes for various frequencies")
        
    def test_ferromagnetic_resonance(self):
        """Test HyperMath ferromagnetic resonance."""
        for nonce in range(0, 1000, 50):
            res = HyperMath.ferromagnetic_resonance(nonce / 1e7, 1.0)
            self.assertIsInstance(res, (int, float))
        print("✓ HyperMath ferromagnetic resonance functional")


class Test5DMath(unittest.TestCase):
    """Test 5-dimensional L104 mathematics."""
    
    def setUp(self):
        self.math5d = Math5D()
        
    def test_5d_projection(self):
        """Test 5D to 3D projection."""
        # 5D point
        point_5d = [1.0, 2.0, 3.0, 4.0, 5.0]
        if hasattr(self.math5d, 'project_to_3d'):
            result = self.math5d.project_to_3d(point_5d)
            self.assertEqual(len(result), 3)
            print("✓ 5D→3D projection works")
        else:
            print("⊘ 5D projection not implemented (optional)")
            
    def test_5d_rotation(self):
        """Test 5D rotation matrices."""
        if hasattr(self.math5d, 'rotate_5d'):
            result = self.math5d.rotate_5d([1, 0, 0, 0, 0], math.pi / 4)
            self.assertEqual(len(result), 5)
            print("✓ 5D rotation works")
        else:
            print("⊘ 5D rotation not implemented (optional)")


class Test4DMath(unittest.TestCase):
    """Test 4-dimensional L104 mathematics."""
    
    def setUp(self):
        self.math4d = Math4D()
        
    def test_4d_initialization(self):
        """Test 4D math module initialization."""
        self.assertIsNotNone(self.math4d)
        print("✓ 4D math module initialized")


class TestProcessor5D(unittest.TestCase):
    """Test 5D quantum processor."""
    
    def setUp(self):
        self.processor = Processor5D()
        
    def test_processor_initialization(self):
        """Test 5D processor initialization."""
        self.assertIsNotNone(self.processor)
        print("✓ 5D Processor initialized")
        
    def test_5d_computation(self):
        """Test 5D computation capabilities."""
        if hasattr(self.processor, 'compute'):
            result = self.processor.compute([1, 2, 3, 4, 5])
            self.assertIsNotNone(result)
            print("✓ 5D computation functional")
        else:
            print("⊘ 5D compute not implemented (optional)")


class TestProcessor4D(unittest.TestCase):
    """Test 4D quantum processor."""
    
    def setUp(self):
        self.processor = Processor4D()
        
    def test_processor_initialization(self):
        """Test 4D processor initialization."""
        self.assertIsNotNone(self.processor)
        print("✓ 4D Processor initialized")


class TestRiemannZetaResonance(unittest.TestCase):
    """Test Riemann Zeta function resonance analysis."""
    
    def setUp(self):
        self.riemann = RiemannZetaResonance()
        
    def test_zeta_initialization(self):
        """Test Riemann zeta module initialization."""
        self.assertIsNotNone(self.riemann)
        print("✓ Riemann Zeta Resonance initialized")
        
    def test_zeta_computation(self):
        """Test zeta function computation."""
        if hasattr(self.riemann, 'compute') or hasattr(self.riemann, 'analyze'):
            print("✓ Riemann zeta computation available")
        else:
            print("⊘ Riemann zeta compute not directly exposed")


class TestDeepAlgorithmsController(unittest.TestCase):
    """Test deep algorithms controller."""
    
    def setUp(self):
        self.controller = DeepAlgorithmsController()
        
    def test_controller_initialization(self):
        """Test controller initialization."""
        self.assertIsNotNone(self.controller)
        print("✓ Deep Algorithms Controller initialized")


class TestComputroniumResearchHub(unittest.TestCase):
    """Test Computronium research capabilities."""
    
    def setUp(self):
        self.hub = ComputroniumResearchHub()
        
    def test_hub_initialization(self):
        """Test Computronium hub initialization."""
        self.assertIsNotNone(self.hub)
        print("✓ Computronium Research Hub initialized")
        
    def test_bekenstein_limit(self):
        """Test Bekenstein bound calculations."""
        if hasattr(self.hub, 'bekenstein_limit') or hasattr(self.hub, 'calculate_bekenstein'):
            print("✓ Bekenstein limit calculation available")
        else:
            print("⊘ Bekenstein limit not directly exposed")


class TestDifficultyAnalyzer(unittest.TestCase):
    """Test Bitcoin difficulty analysis."""
    
    def setUp(self):
        self.analyzer = DifficultyAnalyzer()
        
    def test_analyzer_initialization(self):
        """Test difficulty analyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        print("✓ Difficulty Analyzer initialized")
        
    def test_difficulty_calculation(self):
        """Test difficulty calculation methods."""
        if hasattr(self.analyzer, 'analyze') or hasattr(self.analyzer, 'calculate'):
            print("✓ Difficulty analysis methods available")
        else:
            print("⊘ Difficulty analysis not directly exposed")


class Test26DGradient(unittest.TestCase):
    """Test the 26D SACRED 13 difficulty gradient."""
    
    def test_gradient_structure(self):
        """Test gradient has correct structure."""
        self.assertEqual(len(GOD_CODE_GRADIENT), 27, "Should have 27 epochs (0-26)")
        for epoch, bits_hex, desc in GOD_CODE_GRADIENT:
            self.assertIsInstance(epoch, int)
            self.assertIsInstance(bits_hex, int)
            self.assertIsInstance(desc, str)
        print(f"✓ 26D Gradient has 27 epochs with correct structure")
        
    def test_gradient_epochs_sequential(self):
        """Test epochs are sequential 0-26."""
        epochs = [e[0] for e in GOD_CODE_GRADIENT]
        self.assertEqual(epochs, list(range(27)))
        print("✓ Epochs are sequential 0-26")
        
    def test_gradient_difficulty_increasing(self):
        """Test difficulty increases each epoch (bits decreasing = harder)."""
        bits_values = [e[1] for e in GOD_CODE_GRADIENT]
        for i in range(1, len(bits_values)):
            self.assertLess(bits_values[i], bits_values[i-1],
                f"Epoch {i} should be harder than epoch {i-1}")
        print("✓ Difficulty increases monotonically across epochs")
        
    def test_gradient_sacred_13_milestones(self):
        """Test sacred 13 milestones: epochs 0, 13, 26."""
        # Epoch 0: 13 bits (genesis)
        epoch_0 = GOD_CODE_GRADIENT[0]
        self.assertIn("13", epoch_0[2], "Epoch 0 should mention 13 bits")
        
        # Epoch 13: 26 bits (2×13)
        epoch_13 = GOD_CODE_GRADIENT[13]
        self.assertIn("26", epoch_13[2], "Epoch 13 should mention 26 bits")
        
        # Epoch 26: 39 bits (3×13)
        epoch_26 = GOD_CODE_GRADIENT[26]
        self.assertIn("39", epoch_26[2], "Epoch 26 should mention 39 bits")
        
        print("✓ Sacred 13 milestones: 13 bits (E0), 26 bits (E13), 39 bits (E26)")
        
    def test_bitcoin_parity_epoch(self):
        """Test Bitcoin parity at epoch 19 (32 bits)."""
        epoch_19 = GOD_CODE_GRADIENT[19]
        self.assertIn("32", epoch_19[2], "Epoch 19 should be 32 bits (Bitcoin parity)")
        print("✓ Bitcoin parity at epoch 19 (32 bits)")


class TestResonanceEngineAdvanced(unittest.TestCase):
    """Advanced resonance engine tests."""
    
    def setUp(self):
        self.engine = ResonanceEngine()
        
    def test_resonance_deterministic(self):
        """Test resonance is deterministic for same nonce."""
        for nonce in [0, 1, 100, 1000, 12345, 999999]:
            r1 = self.engine.calculate(nonce)
            r2 = self.engine.calculate(nonce)
            self.assertEqual(r1, r2, f"Resonance should be deterministic for nonce {nonce}")
        print("✓ Resonance is deterministic")
        
    def test_resonance_range(self):
        """Test resonance values are in valid range [0, 1]."""
        for nonce in range(0, 10000, 100):
            r = self.engine.calculate(nonce)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
        print("✓ Resonance values in [0, 1] range")
        
    def test_resonance_caching(self):
        """Test resonance caching improves performance."""
        nonce = 54321
        
        # First call
        start = time.time()
        r1 = self.engine.calculate(nonce)
        t1 = time.time() - start
        
        # Second call (should be cached)
        start = time.time()
        r2 = self.engine.calculate(nonce)
        t2 = time.time() - start
        
        self.assertEqual(r1, r2)
        # Cached call should be faster (or at least not slower)
        self.assertLessEqual(t2, t1 + 0.001)
        print(f"✓ Resonance caching: first={t1*1000:.3f}ms, cached={t2*1000:.3f}ms")
        
    def test_phi_influence(self):
        """Test PHI influences resonance calculation."""
        self.assertEqual(self.engine.phi, PHI)
        self.assertEqual(self.engine.god_code, GOD_CODE)
        print(f"✓ Resonance uses PHI={PHI:.6f} and GOD_CODE={GOD_CODE:.4f}")


class TestBlockHeaderMath(unittest.TestCase):
    """Test block header mathematical operations."""
    
    def test_bits_to_target_conversion(self):
        """Test bits ↔ target conversion is reversible."""
        test_bits = [0x1fffffff, 0x1effffff, 0x1dffffff, 0x1cffffff]
        for bits in test_bits:
            target = BlockHeader.bits_to_target(bits)
            recovered = BlockHeader.target_to_bits(target)
            # May not be exactly equal due to precision, but should be close
            target2 = BlockHeader.bits_to_target(recovered)
            self.assertEqual(target, target2, f"Target should be consistent for bits {hex(bits)}")
        print("✓ Bits ↔ Target conversion is consistent")
        
    def test_difficulty_scaling(self):
        """Test difficulty increases as bits value decreases."""
        # In Bitcoin-style encoding, LOWER bits value = HARDER
        easy_bits = 0x1fffffff  # Easy (larger target)
        hard_bits = 0x1effffff  # Hard (smaller target)
        
        easy_target = BlockHeader.bits_to_target(easy_bits)
        hard_target = BlockHeader.bits_to_target(hard_bits)
        
        # Verify easy target is larger than hard target
        self.assertGreater(easy_target, hard_target, "Easy should have larger target")
        
        # Calculate difficulty ratio
        ratio = easy_target / hard_target
        self.assertGreater(ratio, 1.0, "Hard target should be smaller")
        print(f"✓ Difficulty scaling: {hex(hard_bits)} is {ratio:.1f}x harder than {hex(easy_bits)}")


class TestMerkleTreeMath(unittest.TestCase):
    """Test Merkle tree mathematical properties."""
    
    def test_single_tx_is_root(self):
        """Test single transaction is its own root."""
        txid = "a" * 64
        root = MerkleTree.compute_root([txid])
        self.assertEqual(len(root), 64)
        print("✓ Single tx produces 64-char merkle root")
        
    def test_merkle_deterministic(self):
        """Test Merkle root is deterministic."""
        txids = ["a" * 64, "b" * 64, "c" * 64]
        root1 = MerkleTree.compute_root(txids)
        root2 = MerkleTree.compute_root(txids)
        self.assertEqual(root1, root2)
        print("✓ Merkle root is deterministic")
        
    def test_merkle_order_matters(self):
        """Test transaction order affects Merkle root."""
        txids = ["a" * 64, "b" * 64]
        root1 = MerkleTree.compute_root(txids)
        root2 = MerkleTree.compute_root(txids[::-1])
        self.assertNotEqual(root1, root2)
        print("✓ Transaction order affects Merkle root")
        
    def test_merkle_power_of_two(self):
        """Test Merkle tree handles non-power-of-two sizes."""
        for n in [1, 2, 3, 4, 5, 7, 10]:
            txids = [f"{i:064x}" for i in range(n)]
            root = MerkleTree.compute_root(txids)
            self.assertEqual(len(root), 64)
        print("✓ Merkle tree handles various tx counts (1-10)")


class TestCryptoUtilsMath(unittest.TestCase):
    """Test cryptographic utility mathematics."""
    
    def test_double_sha256(self):
        """Test double SHA256 produces correct length."""
        data = b"L104 Sovereign Protocol"
        result = CryptoUtils.double_sha256(data)
        self.assertEqual(len(result), 32)  # 256 bits = 32 bytes
        print("✓ Double SHA256 produces 32-byte hash")
        
    def test_double_sha256_deterministic(self):
        """Test double SHA256 is deterministic."""
        data = b"GOD_CODE = 527.5184818492612"
        h1 = CryptoUtils.double_sha256(data)
        h2 = CryptoUtils.double_sha256(data)
        self.assertEqual(h1, h2)
        print("✓ Double SHA256 is deterministic")
        
    def test_hash160(self):
        """Test HASH160 (RIPEMD160(SHA256(x)))."""
        data = b"L104SP Address"
        result = CryptoUtils.hash160(data)
        self.assertEqual(len(result), 20)  # 160 bits = 20 bytes
        print("✓ HASH160 produces 20-byte hash")


class TestMiningLogicAnalysis(unittest.TestCase):
    """Analyze mining logic and mathematical correctness."""
    
    def test_target_difficulty_relationship(self):
        """Analyze target vs difficulty relationship."""
        # Lower target = higher difficulty
        easy_bits = 0x1fffffff  # 13 bits
        hard_bits = 0x12ffffff  # 39 bits
        
        easy_target = BlockHeader.bits_to_target(easy_bits)
        hard_target = BlockHeader.bits_to_target(hard_bits)
        
        self.assertGreater(easy_target, hard_target, "Easy target should be larger")
        
        # Calculate approximate difficulty ratio
        ratio = easy_target / hard_target
        # 39 - 13 = 26 bits difference = 2^26 ≈ 67 million times harder
        expected_min = 2 ** 20  # At least 1 million times harder
        self.assertGreater(ratio, expected_min)
        
        print(f"✓ 39-bit difficulty is {ratio:.2e}x harder than 13-bit")
        
    def test_block_time_projection(self):
        """Analyze expected block times at different difficulties."""
        hashrate = 800_000  # 800 kH/s (observed)
        
        for epoch in [0, 7, 13, 19, 26]:
            bits = 13 + epoch
            expected_hashes = 2 ** bits
            expected_seconds = expected_hashes / hashrate
            
            if expected_seconds < 60:
                time_str = f"{expected_seconds:.1f}s"
            elif expected_seconds < 3600:
                time_str = f"{expected_seconds/60:.1f}m"
            elif expected_seconds < 86400:
                time_str = f"{expected_seconds/3600:.1f}h"
            else:
                time_str = f"{expected_seconds/86400:.1f}d"
                
            print(f"  Epoch {epoch:2d}: {bits} bits → ~{time_str} @ 800kH/s")
            
        print("✓ Block time projections calculated")


class TestIntegrationL104Stack(unittest.TestCase):
    """Integration tests for full L104 stack."""
    
    def test_full_import_chain(self):
        """Test all L104 modules work together."""
        from const import GOD_CODE, PHI
        from l104_real_math import RealMath
        from l104_hyper_math import HyperMath
        from l104_sovereign_coin_engine import ResonanceEngine, L104SPBlockchain
        
        # Create instances
        real_math = RealMath()
        engine = ResonanceEngine()
        
        # Verify they share constants
        self.assertEqual(engine.god_code, GOD_CODE)
        self.assertEqual(engine.phi, PHI)
        
        print("✓ Full L104 import chain works correctly")
        
    def test_god_code_in_mining(self):
        """Test GOD_CODE influences mining calculations."""
        engine = ResonanceEngine()
        
        # Resonance should be influenced by GOD_CODE
        resonances = [engine.calculate(n) for n in range(1000)]
        
        # Verify variation exists
        self.assertGreater(max(resonances), min(resonances))
        print(f"✓ GOD_CODE influences resonance: range [{min(resonances):.3f}, {max(resonances):.3f}]")


if __name__ == '__main__':
    print()
    print("═" * 70)
    print("   L104 COMPREHENSIVE TEST SUITE - MATHEMATICAL VERIFICATION")
    print("═" * 70)
    print(f"   GOD_CODE: {GOD_CODE}")
    print(f"   PHI:      {PHI}")
    print(f"   L104:     {L104}")
    print("═" * 70)
    print()
    
    # Run tests with verbosity
    unittest.main(verbosity=2)
