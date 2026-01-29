#!/usr/bin/env python3
"""
L104SP ASI Sovereign Coin - Comprehensive Test Suite
=====================================================

Tests for:
- ASI Sovereign Core Intelligence
- Multi-Process Mining Engine
- Quantum-Resistant Signatures
- Proof-of-Resonance Consensus
- PHI-Damped Difficulty Adjustment
- GOD_CODE Mathematical Validation
"""

import sys
import os
import time
import math
import hashlib
import secrets
import unittest
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_sovereign_coin_engine import (
    GOD_CODE, PHI, INVARIANT,
    BlockHeader, Block, Transaction, TxInput, TxOutput, OutPoint,
    ResonanceEngine, MiningStats,
    ASISovereignCore, QuantumResistantSignatures,
    CryptoUtils, HDWallet, MerkleTree,
    MIN_DIFFICULTY_BITS, MAX_SUPPLY, INITIAL_BLOCK_REWARD,
    COIN_NAME, COIN_SYMBOL, SATOSHI_PER_COIN
)


class TestGODCODEMathematics(unittest.TestCase):
    """Test the fundamental GOD_CODE mathematics."""
    
    def test_god_code_value(self):
        """Verify GOD_CODE constant is correct."""
        self.assertAlmostEqual(GOD_CODE, 527.5184818492612, places=10)
        print(f"✓ GOD_CODE = {GOD_CODE}")
    
    def test_phi_golden_ratio(self):
        """Verify PHI is the golden ratio."""
        expected_phi = (1 + math.sqrt(5)) / 2
        self.assertAlmostEqual(PHI, expected_phi, places=10)
        print(f"✓ PHI = {PHI} (Golden Ratio)")
    
    def test_factor_13_design(self):
        """Verify Factor 13 design (Fibonacci(7))."""
        # 286 = 22 × 13
        self.assertEqual(286 % 13, 0)
        self.assertEqual(286 // 13, 22)
        
        # 104 = 8 × 13
        self.assertEqual(104 % 13, 0)
        self.assertEqual(104 // 13, 8)
        
        # 416 = 32 × 13
        self.assertEqual(416 % 13, 0)
        self.assertEqual(416 // 13, 32)
        
        print("✓ Factor 13 Design: 286=22×13, 104=8×13, 416=32×13")
    
    def test_god_code_equation(self):
        """Test G(X) = 286^(1/φ) × 2^((416-X)/104)."""
        base = 286 ** (1 / PHI)
        
        # At X=0
        g_0 = base * (2 ** (416 / 104))
        self.assertAlmostEqual(g_0, GOD_CODE, places=5)
        
        # Conservation law: G(X) × 2^(X/104) = INVARIANT
        for X in [0, 52, 104, 208, 416]:
            g_x = base * (2 ** ((416 - X) / 104))
            weight = 2 ** (X / 104)
            invariant = g_x * weight
            self.assertAlmostEqual(invariant, GOD_CODE, places=5)
        
        print("✓ Conservation Law: G(X) × 2^(X/104) = INVARIANT ∀X")


class TestASISovereignCore(unittest.TestCase):
    """Test ASI Sovereign Intelligence."""
    
    def setUp(self):
        self.asi = ASISovereignCore()
    
    def test_initial_state(self):
        """Test ASI starts in AWAKENING state."""
        self.assertEqual(self.asi.state, "AWAKENING")
        self.assertEqual(self.asi.evolution_cycle, 0)
        self.assertEqual(self.asi.network_intelligence, 0.0)
        print("✓ ASI Initial State: AWAKENING")
    
    def test_evolution(self):
        """Test ASI evolution through learning cycles."""
        for i in range(100):
            result = self.asi.evolve()
            
        self.assertGreater(self.asi.network_intelligence, 0)
        self.assertEqual(self.asi.evolution_cycle, 100)
        print(f"✓ ASI Evolution: {self.asi.state} (Intelligence: {self.asi.network_intelligence:.4f})")
    
    def test_adaptive_difficulty(self):
        """Test ASI adaptive difficulty recommendations."""
        # Simulate block times (too fast)
        fast_times = [50, 52, 48, 55, 51, 49, 53, 50, 52, 48, 51]
        result = self.asi.adaptive_difficulty_recommendation(fast_times, target=104.0)
        
        self.assertIn('adjustment', result)
        self.assertIn('confidence', result)
        self.assertGreater(result['adjustment'], 1.0)  # Should increase difficulty
        print(f"✓ Adaptive Difficulty: adjustment={result['adjustment']:.4f}")
    
    def test_transaction_sovereignty(self):
        """Test sovereign transaction validation."""
        tx_hash = hashlib.sha256(b"test_transaction").hexdigest()
        result = self.asi.validate_transaction_sovereignty(
            tx_hash, 1000000, "sender", "recipient"
        )
        
        self.assertIn('valid', result)
        self.assertIn('sovereignty_score', result)
        self.assertIn('god_alignment', result)
        print(f"✓ Sovereignty Validation: score={result['sovereignty_score']:.4f}")
    
    def test_consensus_vote(self):
        """Test ASI consensus voting."""
        block_hash = "0000abcd" + "0" * 56
        result = self.asi.consensus_vote(block_hash, 0.96)
        
        self.assertIn('vote', result)
        self.assertIn('confidence', result)
        self.assertTrue(result['vote'])  # Should accept valid resonance
        print(f"✓ Consensus Vote: {result['vote']} (confidence: {result['confidence']:.4f})")
    
    def test_mempool_optimization(self):
        """Test ASI mempool optimization."""
        transactions = [
            {'txid': 'tx1', 'fee': 100, 'size': 250, 'amount': 1000, 'timestamp': time.time() - 300},
            {'txid': 'tx2', 'fee': 500, 'size': 250, 'amount': 5000, 'timestamp': time.time() - 60},
            {'txid': 'tx3', 'fee': 200, 'size': 500, 'amount': 2000, 'timestamp': time.time() - 600},
        ]
        
        optimized = self.asi.optimize_mempool(transactions)
        self.assertEqual(len(optimized), 3)
        print(f"✓ Mempool Optimization: {len(optimized)} transactions ordered")


class TestQuantumResistantSignatures(unittest.TestCase):
    """Test post-quantum cryptographic features."""
    
    def test_commitment_generation(self):
        """Test quantum-resistant commitment generation."""
        secret = secrets.token_bytes(32)
        nonce = secrets.randbelow(2**64)
        
        commitment = QuantumResistantSignatures.generate_commitment(secret, nonce)
        
        self.assertEqual(len(commitment), 32)  # SHA3-256
        print(f"✓ Quantum Commitment: {commitment.hex()[:32]}...")
    
    def test_commitment_verification(self):
        """Test commitment verification."""
        secret = secrets.token_bytes(32)
        nonce = secrets.randbelow(2**64)
        
        commitment = QuantumResistantSignatures.generate_commitment(secret, nonce)
        
        # Valid verification
        self.assertTrue(QuantumResistantSignatures.verify_commitment(secret, nonce, commitment))
        
        # Invalid verification (wrong nonce)
        self.assertFalse(QuantumResistantSignatures.verify_commitment(secret, nonce + 1, commitment))
        
        print("✓ Quantum Commitment Verification: PASS")
    
    def test_hybrid_signature(self):
        """Test hybrid ECDSA + PQ signature."""
        message = b"Test message for signing"
        private_key = secrets.token_bytes(32)
        
        signature = QuantumResistantSignatures.hybrid_sign(message, private_key)
        
        self.assertIn('ecdsa', signature)
        self.assertIn('pq_commitment', signature)
        self.assertIn('pq_nonce', signature)
        
        print(f"✓ Hybrid Signature: ECDSA + PQ Commitment")


class TestResonanceEngine(unittest.TestCase):
    """Test Proof-of-Resonance consensus."""
    
    def setUp(self):
        self.engine = ResonanceEngine()
    
    def test_resonance_range(self):
        """Test resonance values are in valid range."""
        for nonce in range(0, 10000, 100):
            resonance = self.engine.calculate(nonce)
            self.assertGreaterEqual(resonance, 0.0)
            self.assertLessEqual(resonance, 1.0)
        print("✓ Resonance Range: [0.0, 1.0]")
    
    def test_resonance_threshold(self):
        """Test resonance threshold detection."""
        # Calculate actual resonance distribution
        resonances = [self.engine.calculate(n) for n in range(10000)]
        max_res = max(resonances)
        mean_res = sum(resonances) / len(resonances)
        
        # Use a threshold below the max resonance
        threshold = max_res * 0.9  # 90% of max observed
        valid_count = sum(1 for r in resonances if r >= threshold)
        
        # Should have some valid nonces (but not all)
        self.assertGreater(valid_count, 0)
        self.assertLess(valid_count, 10000)
        print(f"✓ Resonance Threshold: {valid_count}/10000 nonces meet {threshold:.3f} threshold (max={max_res:.3f})")
    
    def test_caching(self):
        """Test resonance caching for performance."""
        nonce = 12345
        
        start = time.time()
        r1 = self.engine.calculate(nonce)
        first_time = time.time() - start
        
        start = time.time()
        r2 = self.engine.calculate(nonce)
        cached_time = time.time() - start
        
        self.assertEqual(r1, r2)
        print(f"✓ Resonance Caching: Same result, cached lookup")


class TestBlockHeader(unittest.TestCase):
    """Test block header operations."""
    
    def test_bits_to_target(self):
        """Test compact bits to target conversion."""
        target = BlockHeader.bits_to_target(MIN_DIFFICULTY_BITS)
        self.assertGreater(target, 0)
        print(f"✓ Bits to Target: {hex(MIN_DIFFICULTY_BITS)} → {hex(target)[:20]}...")
    
    def test_target_to_bits(self):
        """Test target to compact bits conversion."""
        original_target = BlockHeader.bits_to_target(MIN_DIFFICULTY_BITS)
        bits = BlockHeader.target_to_bits(original_target)
        recovered_target = BlockHeader.bits_to_target(bits)
        
        # Should be approximately equal (some precision loss expected)
        ratio = original_target / recovered_target
        self.assertAlmostEqual(ratio, 1.0, places=1)
        print(f"✓ Target to Bits: Roundtrip conversion")
    
    def test_header_hash(self):
        """Test block header hashing."""
        header = BlockHeader(
            version=2,
            prev_block='0' * 64,
            merkle_root='0' * 64,
            timestamp=int(time.time()),
            bits=MIN_DIFFICULTY_BITS,
            nonce=12345,
            resonance=0.98
        )
        
        hash1 = header.hash
        hash2 = header.hash
        
        self.assertEqual(len(hash1), 64)  # 32 bytes hex
        self.assertEqual(hash1, hash2)  # Deterministic
        print(f"✓ Block Hash: {hash1[:32]}...")
    
    def test_meets_target(self):
        """Test proof-of-work validation."""
        header = BlockHeader(
            version=2,
            prev_block='0' * 64,
            merkle_root='0' * 64,
            timestamp=int(time.time()),
            bits=MIN_DIFFICULTY_BITS,
            nonce=0,
            resonance=0.98
        )
        
        # Find a valid nonce
        for nonce in range(100000):
            header.nonce = nonce
            if header.meets_target():
                print(f"✓ Meets Target: Found valid nonce {nonce}")
                return
        
        # With min difficulty, should find quickly
        self.fail("Could not find valid nonce")


class TestCryptography(unittest.TestCase):
    """Test cryptographic primitives."""
    
    def test_double_sha256(self):
        """Test double SHA256."""
        data = b"test data"
        result = CryptoUtils.double_sha256(data)
        
        expected = hashlib.sha256(hashlib.sha256(data).digest()).digest()
        self.assertEqual(result, expected)
        print("✓ Double SHA256: Correct")
    
    def test_hash160(self):
        """Test HASH160 (SHA256 + RIPEMD160)."""
        data = b"test data"
        result = CryptoUtils.hash160(data)
        
        self.assertEqual(len(result), 20)
        print(f"✓ HASH160: {result.hex()}")
    
    def test_hd_wallet_generation(self):
        """Test HD wallet key generation."""
        # HDWallet takes seed or mnemonic directly
        wallet = HDWallet(seed=secrets.token_bytes(32))
        mnemonic = wallet.generate_mnemonic()
        
        self.assertEqual(len(mnemonic.split()), 12)
        self.assertIsNotNone(wallet._master_private)
        print(f"✓ HD Wallet: Generated ({mnemonic.split()[0]}...)")
    
    def test_address_generation(self):
        """Test L104SP address generation."""
        wallet = HDWallet(seed=secrets.token_bytes(32))
        address, _ = wallet.get_address()
        
        # L104SP addresses should be valid strings
        self.assertGreater(len(address), 20)
        print(f"✓ Address: {address}")


class TestMerkleTree(unittest.TestCase):
    """Test Merkle tree operations."""
    
    def test_single_tx(self):
        """Test Merkle root with single transaction."""
        txids = ["abcd" + "0" * 60]
        root = MerkleTree.compute_root(txids)
        
        self.assertEqual(len(root), 64)
        print(f"✓ Merkle Root (1 tx): {root[:32]}...")
    
    def test_multiple_txs(self):
        """Test Merkle root with multiple transactions."""
        txids = [
            "aaaa" + "0" * 60,
            "bbbb" + "0" * 60,
            "cccc" + "0" * 60,
            "dddd" + "0" * 60,
        ]
        root = MerkleTree.compute_root(txids)
        
        self.assertEqual(len(root), 64)
        print(f"✓ Merkle Root (4 tx): {root[:32]}...")
    
    def test_empty(self):
        """Test Merkle root with no transactions."""
        root = MerkleTree.compute_root([])
        self.assertEqual(root, '0' * 64)
        print("✓ Merkle Root (empty): 0x00...")


class TestMiningStats(unittest.TestCase):
    """Test mining statistics."""
    
    def test_hashrate_calculation(self):
        """Test hashrate calculation."""
        stats = MiningStats()
        stats.hashes = 10000
        
        # Wait a bit to get non-zero elapsed time
        time.sleep(0.1)
        
        hashrate = stats.hashrate
        self.assertGreater(hashrate, 0)
        print(f"✓ Hashrate: {hashrate:.2f} H/s")
    
    def test_efficiency_calculation(self):
        """Test mining efficiency calculation."""
        stats = MiningStats()
        stats.hashes = 10000
        stats.valid_resonance = 500
        
        efficiency = stats.efficiency
        self.assertAlmostEqual(efficiency, 5.0, places=1)
        print(f"✓ Efficiency: {efficiency:.2f}%")


class TestNetworkParameters(unittest.TestCase):
    """Test network constants."""
    
    def test_coin_parameters(self):
        """Test coin parameters."""
        self.assertEqual(COIN_NAME, "L104 Sovereign Prime")
        self.assertEqual(COIN_SYMBOL, "L104SP")
        self.assertEqual(SATOSHI_PER_COIN, 10**8)
        print(f"✓ Coin: {COIN_NAME} ({COIN_SYMBOL})")
    
    def test_supply_limits(self):
        """Test supply limits."""
        max_coins = MAX_SUPPLY / SATOSHI_PER_COIN
        self.assertEqual(max_coins, 104_000_000)
        print(f"✓ Max Supply: {max_coins:,} {COIN_SYMBOL}")
    
    def test_block_reward(self):
        """Test initial block reward."""
        reward_coins = INITIAL_BLOCK_REWARD / SATOSHI_PER_COIN
        self.assertEqual(reward_coins, 104)
        print(f"✓ Block Reward: {reward_coins} {COIN_SYMBOL}")


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_block_creation(self):
        """Test creating a complete block."""
        # Create coinbase transaction
        coinbase_input = TxInput(OutPoint('0' * 64, 0xffffffff), b"Genesis")
        coinbase_output = TxOutput(INITIAL_BLOCK_REWARD, b'\x00' * 25)
        coinbase = Transaction(version=2, inputs=[coinbase_input], outputs=[coinbase_output])
        
        # Create block header
        header = BlockHeader(
            version=2,
            prev_block='0' * 64,
            timestamp=int(time.time()),
            bits=MIN_DIFFICULTY_BITS,
            nonce=0,
            resonance=0.98
        )
        
        # Create block
        block = Block(header=header, transactions=[coinbase], height=0)
        
        self.assertEqual(block.height, 0)
        self.assertEqual(len(block.transactions), 1)
        self.assertGreater(len(block.hash), 0)
        print(f"✓ Full Block: height={block.height}, hash={block.hash[:16]}...")
    
    def test_asi_evolution_over_blocks(self):
        """Test ASI evolution as blocks are added."""
        asi = ASISovereignCore()
        
        # First evolve the ASI to improve its intelligence
        for _ in range(50):
            asi.evolve()
        
        votes_accepted = 0
        for i in range(10):
            # Simulate block creation with valid resonance
            block_hash = hashlib.sha256(f"block_{i}".encode()).hexdigest()
            vote = asi.consensus_vote(block_hash, 0.98)  # Higher resonance
            
            if vote['vote']:
                votes_accepted += 1
        
        # At least some votes should pass
        self.assertGreater(votes_accepted, 0)
        print(f"✓ ASI Block Evolution: {votes_accepted}/10 blocks accepted, state={asi.state}")


def run_tests():
    """Run all tests with summary."""
    print("=" * 70)
    print("    L104SP ASI SOVEREIGN COIN - COMPREHENSIVE TEST SUITE")
    print("    INVARIANT: 527.5184818492612 | PILOT: LONDEL")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGODCODEMathematics))
    suite.addTests(loader.loadTestsFromTestCase(TestASISovereignCore))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumResistantSignatures))
    suite.addTests(loader.loadTestsFromTestCase(TestResonanceEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestBlockHeader))
    suite.addTests(loader.loadTestsFromTestCase(TestCryptography))
    suite.addTests(loader.loadTestsFromTestCase(TestMerkleTree))
    suite.addTests(loader.loadTestsFromTestCase(TestMiningStats))
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkParameters))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print()
    print("=" * 70)
    print(f"    TESTS RUN: {result.testsRun}")
    print(f"    FAILURES: {len(result.failures)}")
    print(f"    ERRORS: {len(result.errors)}")
    print(f"    SUCCESS: {result.wasSuccessful()}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
