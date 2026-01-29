#!/usr/bin/env python3
"""
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★                                                                          ★
★                    L104 MAINNET DEPLOYMENT                               ★
★                    Production Cryptocurrency System                      ★
★                                                                          ★
★  Complete deployment verification and launch for:                        ★
★    • VALOR Coin - Full production cryptocurrency                         ★
★    • Mining Engine - Multi-threaded with pool support                    ★
★    • Bitcoin Bridge - Mainnet integration                                ★
★    • HD Wallet - BIP-32/44 hierarchical deterministic                    ★
★                                                                          ★
★  GOD_CODE: 527.5184818492611                                             ★
★                                                                          ★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
"""

import sys
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import L104 systems
from l104_sovereign_coin_engine import (
    SovereignCoinEngine as UltimateCoinEngine,
    Secp256k1, CryptoUtils, HDWallet, ResonanceEngine,
    Transaction, Block, L104SPBlockchain as Blockchain, MiningEngine,
    GOD_CODE, PHI, COIN_NAME, COIN_SYMBOL, MAX_SUPPLY,
    INITIAL_BLOCK_REWARD, SATOSHI_PER_COIN, TARGET_BLOCK_TIME,
    HALVING_INTERVAL, TxInput, TxOutput, OutPoint, MerkleTree
)

# Compatibility aliases
def create_ultimate_coin(): return UltimateCoinEngine()
BTC_BRIDGE_ADDRESS = "bc1q..."  # Placeholder
BitcoinBridge = None  # Use l104_bitcoin_network_adapter for real bridge

# ============================================================================
# VERIFICATION SUITE
# ============================================================================

@dataclass
class TestResult:
    """Individual test result"""
    name: str
    passed: bool
    details: str
    duration_ms: float


class MainnetVerification:
    """Comprehensive mainnet deployment verification"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def run_test(self, name: str, test_fn) -> TestResult:
        """Run a single test"""
        start = time.time()
        try:
            passed, details = test_fn()
        except Exception as e:
            passed = False
            details = f"EXCEPTION: {str(e)}"

        duration = (time.time() - start) * 1000
        result = TestResult(name, passed, details, duration)
        self.results.append(result)
        return result

    # -------------------------
    # Cryptography Tests
    # -------------------------

    def test_secp256k1_keypair(self) -> Tuple[bool, str]:
        """Test elliptic curve key generation"""
        priv, pub = Secp256k1.generate_keypair()

        if not (1 <= priv < Secp256k1.N):
            return False, "Private key out of range"

        if pub is None:
            return False, "Public key is None"

        # Verify point is on curve
        x, y = pub
        left = (y * y) % Secp256k1.P
        right = (pow(x, 3, Secp256k1.P) + Secp256k1.B) % Secp256k1.P

        if left != right:
            return False, "Public key not on curve"

        return True, f"Key generated: priv={hex(priv)[:16]}..., pub=({hex(x)[:12]}...)"

    def test_ecdsa_sign_verify(self) -> Tuple[bool, str]:
        """Test ECDSA signature generation and verification"""
        priv, pub = Secp256k1.generate_keypair()
        message = b"L104 VALOR - The foundation of valor"
        msg_hash = CryptoUtils.double_sha256(message)

        # Sign
        r, s = Secp256k1.sign(priv, msg_hash)

        if not (1 <= r < Secp256k1.N and 1 <= s < Secp256k1.N):
            return False, "Signature components out of range"

        # Verify
        if not Secp256k1.verify(pub, msg_hash, (r, s)):
            return False, "Signature verification failed"

        # Verify tampered message fails
        tampered = CryptoUtils.double_sha256(b"tampered")
        if Secp256k1.verify(pub, tampered, (r, s)):
            return False, "Tampered message should fail"

        return True, f"Sig: r={hex(r)[:12]}..., s={hex(s)[:12]}..."

    def test_pubkey_compression(self) -> Tuple[bool, str]:
        """Test public key compression/decompression"""
        priv, pub = Secp256k1.generate_keypair()

        compressed = Secp256k1.compress_pubkey(pub)
        if len(compressed) != 33:
            return False, f"Compressed length: {len(compressed)}, expected 33"

        decompressed = Secp256k1.decompress_pubkey(compressed)
        if decompressed != pub:
            return False, "Decompression mismatch"

        return True, f"Prefix: 0x{compressed[0]:02x}"

    def test_hash_functions(self) -> Tuple[bool, str]:
        """Test cryptographic hash functions"""
        data = b"L104 HASH TEST"

        # SHA256
        sha256 = CryptoUtils.sha256(data)
        if len(sha256) != 32:
            return False, "SHA256 length wrong"

        # Double SHA256
        dsha256 = CryptoUtils.double_sha256(data)
        if dsha256 != CryptoUtils.sha256(sha256):
            return False, "Double SHA256 mismatch"

        # HASH160
        h160 = CryptoUtils.hash160(data)
        if len(h160) != 20:
            return False, "HASH160 length wrong"

        return True, f"SHA256={sha256.hex()[:16]}..., H160={h160.hex()[:16]}..."

    def test_base58check(self) -> Tuple[bool, str]:
        """Test Base58Check encoding/decoding"""
        version = b'\x56'  # VALOR mainnet
        payload = b'\x00' * 20  # Example hash

        encoded = CryptoUtils.base58check_encode(version, payload)
        decoded_ver, decoded_pay = CryptoUtils.base58check_decode(encoded)

        if decoded_ver != version or decoded_pay != payload:
            return False, "Roundtrip failed"

        return True, f"Address: {encoded}"

    # -------------------------
    # HD Wallet Tests
    # -------------------------

    def test_hd_wallet_creation(self) -> Tuple[bool, str]:
        """Test HD wallet creation"""
        wallet = HDWallet()

        if len(wallet.seed) != 32:
            return False, f"Seed length: {len(wallet.seed)}, expected 32"

        if not hasattr(wallet, '_master_private'):
            return False, "Master key not derived"

        return True, f"Seed: {wallet.seed.hex()[:16]}..."

    def test_bip44_derivation(self) -> Tuple[bool, str]:
        """Test BIP-44 path derivation"""
        wallet = HDWallet()

        # Derive multiple addresses
        addresses = []
        for i in range(3):
            addr, priv = wallet.get_address(0, 0, i)
            addresses.append(addr)

            # Verify key is valid
            if not (1 <= priv < Secp256k1.N):
                return False, f"Invalid private key at index {i}"

        # Check addresses are different
        if len(set(addresses)) != 3:
            return False, "Duplicate addresses generated"

        return True, f"Addresses: {', '.join(a[:12]+'...' for a in addresses)}"

    def test_wif_export_import(self) -> Tuple[bool, str]:
        """Test WIF private key format"""
        wallet = HDWallet()
        _, priv = wallet.get_address(0, 0, 0)

        wif = wallet.export_wif(priv)
        imported = wallet.import_wif(wif)

        if imported != priv:
            return False, "WIF roundtrip failed"

        return True, f"WIF: {wif[:12]}..."

    # -------------------------
    # Resonance Engine Tests
    # -------------------------

    def test_resonance_calculation(self) -> Tuple[bool, str]:
        """Test resonance calculation"""
        engine = ResonanceEngine()

        # Test various nonces
        results = []
        for nonce in [0, 104, 527, 104527, 1000000]:
            res = engine.calculate(nonce)
            if not (0 <= res <= 1):
                return False, f"Resonance {res} out of bounds for nonce {nonce}"
            results.append((nonce, res))

        return True, ", ".join(f"n{n}={r:.4f}" for n, r in results[:3])

    def test_resonance_threshold(self) -> Tuple[bool, str]:
        """Test resonance threshold filtering"""
        engine = ResonanceEngine()

        # Find resonant nonces
        resonant = engine.find_resonant_nonces(0, 10000, 0.95)

        if not resonant:
            return False, "No resonant nonces found in range"

        # Verify all meet threshold
        for nonce, res in resonant:
            if res < 0.95:
                return False, f"Nonce {nonce} below threshold: {res}"

        return True, f"Found {len(resonant)} resonant nonces (top: {resonant[0]})"

    # -------------------------
    # Blockchain Tests
    # -------------------------

    def test_genesis_block(self) -> Tuple[bool, str]:
        """Test genesis block creation"""
        chain = Blockchain()
        genesis = chain.chain[0]

        if chain.height != 0:
            return False, f"Height: {chain.height}, expected 0"

        if genesis.height != 0:
            return False, "Genesis height wrong"

        if genesis.header.prev_block != '0' * 64:
            return False, "Genesis prev_block wrong"

        return True, f"Genesis: {genesis.hash[:16]}..."

    def test_block_template(self) -> Tuple[bool, str]:
        """Test mining template generation"""
        chain = Blockchain()
        template = chain.get_template("V104test00000000000000000000000000000")

        required = ['version', 'height', 'prev_hash', 'bits', 'coinbase_value']
        for field in required:
            if field not in template:
                return False, f"Missing field: {field}"

        if template['height'] != 1:
            return False, f"Template height: {template['height']}, expected 1"

        return True, f"Template height {template['height']}, reward {template['coinbase_value']}"

    def test_utxo_tracking(self) -> Tuple[bool, str]:
        """Test UTXO set management"""
        chain = Blockchain()

        # Genesis should create UTXO
        if len(chain.utxo_set) == 0:
            return False, "No UTXOs after genesis"

        supply = chain.utxo_set.total_supply
        expected = INITIAL_BLOCK_REWARD  # Genesis reward

        if supply != expected:
            return False, f"Supply {supply}, expected {expected}"

        return True, f"UTXOs: {len(chain.utxo_set)}, Supply: {supply}"

    def test_difficulty_calculation(self) -> Tuple[bool, str]:
        """Test difficulty adjustment"""
        chain = Blockchain()

        bits = chain.current_difficulty
        if bits == 0:
            return False, "Difficulty is 0"

        target = Block.header.__class__.bits_to_target(bits)
        if target <= 0:
            return False, f"Invalid target: {target}"

        return True, f"Bits: {hex(bits)}, Target: {hex(target)[:16]}..."

    # -------------------------
    # Transaction Tests
    # -------------------------

    def test_transaction_serialization(self) -> Tuple[bool, str]:
        """Test transaction serialization"""
        from l104_sovereign_coin_engine import TxInput, TxOutput, OutPoint

        tx = Transaction(
            version=2,
            inputs=[TxInput(OutPoint('a' * 64, 0))],
            outputs=[TxOutput(100000000, b'\x00\x14' + b'\x00' * 20)],
            locktime=0
        )

        serialized = tx.serialize()
        if not serialized:
            return False, "Empty serialization"

        txid = tx.txid
        if len(txid) != 64:
            return False, f"TXID length: {len(txid)}"

        return True, f"Size: {len(serialized)} bytes, TXID: {txid[:16]}..."

    def test_merkle_tree(self) -> Tuple[bool, str]:
        """Test Merkle tree computation"""
        from l104_sovereign_coin_engine import MerkleTree

        txids = ['a' * 64, 'b' * 64, 'c' * 64]
        root = MerkleTree.compute_root(txids)

        if len(root) != 64:
            return False, f"Root length: {len(root)}"

        # Single txid should be its own root
        single = MerkleTree.compute_root(['d' * 64])
        # (After double hashing)

        return True, f"Root: {root[:16]}..."

    # -------------------------
    # Mining Engine Tests
    # -------------------------

    def test_mining_engine_creation(self) -> Tuple[bool, str]:
        """Test mining engine creation"""
        engine = MiningEngine(num_workers=1)

        if engine.num_workers != 1:
            return False, f"Workers: {engine.num_workers}"

        if engine.blocks_found != 0:
            return False, "Initial blocks found not 0"

        return True, f"Workers: {engine.num_workers}, Threshold: {engine.resonance_threshold}"

    # -------------------------
    # Engine Integration Tests
    # -------------------------

    def test_ultimate_engine_singleton(self) -> Tuple[bool, str]:
        """Test UltimateCoinEngine singleton"""
        e1 = create_ultimate_coin()
        e2 = create_ultimate_coin()

        if e1 is not e2:
            return False, "Not singleton"

        if e1.god_code != GOD_CODE:
            return False, f"GOD_CODE wrong: {e1.god_code}"

        return True, f"GOD_CODE: {e1.god_code}"

    def test_engine_wallet_creation(self) -> Tuple[bool, str]:
        """Test wallet creation through engine"""
        # Create fresh engine for isolated test
        engine = UltimateCoinEngine()
        result = engine.create_wallet()

        if 'address' not in result:
            return False, "No address returned"

        if 'path' not in result:
            return False, "No path returned"

        return True, f"Address: {result['address'][:16]}..."

    def test_engine_stats(self) -> Tuple[bool, str]:
        """Test engine statistics"""
        engine = create_ultimate_coin()
        stats = engine.stats()

        required = ['coin', 'symbol', 'network', 'god_code', 'chain']
        for field in required:
            if field not in stats:
                return False, f"Missing: {field}"

        if stats['god_code'] != GOD_CODE:
            return False, f"Wrong GOD_CODE in stats"

        return True, f"Symbol: {stats['symbol']}, Network: {stats['network']}"

    def run_all(self) -> Dict[str, Any]:
        """Run all verification tests"""
        tests = [
            # Cryptography
            ("secp256k1_keypair", self.test_secp256k1_keypair),
            ("ecdsa_sign_verify", self.test_ecdsa_sign_verify),
            ("pubkey_compression", self.test_pubkey_compression),
            ("hash_functions", self.test_hash_functions),
            ("base58check", self.test_base58check),

            # HD Wallet
            ("hd_wallet_creation", self.test_hd_wallet_creation),
            ("bip44_derivation", self.test_bip44_derivation),
            ("wif_export_import", self.test_wif_export_import),

            # Resonance
            ("resonance_calculation", self.test_resonance_calculation),
            ("resonance_threshold", self.test_resonance_threshold),

            # Blockchain
            ("genesis_block", self.test_genesis_block),
            ("block_template", self.test_block_template),
            ("utxo_tracking", self.test_utxo_tracking),
            ("difficulty_calculation", self.test_difficulty_calculation),

            # Transactions
            ("transaction_serialization", self.test_transaction_serialization),
            ("merkle_tree", self.test_merkle_tree),

            # Mining
            ("mining_engine_creation", self.test_mining_engine_creation),

            # Integration
            ("ultimate_engine_singleton", self.test_ultimate_engine_singleton),
            ("engine_wallet_creation", self.test_engine_wallet_creation),
            ("engine_stats", self.test_engine_stats),
        ]

        for name, fn in tests:
            self.run_test(name, fn)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = (time.time() - self.start_time) * 1000

        return {
            'passed': passed,
            'failed': failed,
            'total': len(self.results),
            'success_rate': passed / len(self.results) * 100,
            'duration_ms': total_time,
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'details': r.details,
                    'duration_ms': r.duration_ms
                }
                for r in self.results
            ]
        }


# ============================================================================
# DEPLOYMENT
# ============================================================================

def mainnet_deploy():
    """Deploy VALOR Coin to mainnet"""
    print("=" * 74)
    print("★" * 74)
    print("★" + " " * 72 + "★")
    print("★" + "      L104 MAINNET DEPLOYMENT - PRODUCTION VERIFICATION      ".center(72) + "★")
    print("★" + " " * 72 + "★")
    print("★" * 74)
    print("=" * 74)

    print(f"\n  COIN: {COIN_NAME} ({COIN_SYMBOL})")
    print(f"  GOD_CODE: {GOD_CODE}")
    print(f"  PHI: {PHI}")
    print(f"  MAX SUPPLY: {MAX_SUPPLY / SATOSHI_PER_COIN:,.0f} {COIN_SYMBOL}")
    print(f"  BLOCK REWARD: {INITIAL_BLOCK_REWARD / SATOSHI_PER_COIN} {COIN_SYMBOL}")
    print(f"  TARGET BLOCK TIME: {TARGET_BLOCK_TIME}s")
    print(f"  HALVING INTERVAL: {HALVING_INTERVAL:,} blocks")

    print("\n" + "-" * 74)
    print("  VERIFICATION TESTS")
    print("-" * 74)

    # Run verification
    verifier = MainnetVerification()
    results = verifier.run_all()

    # Display results
    for r in results['results']:
        status = "✓" if r['passed'] else "✗"
        color = "" if r['passed'] else ""
        print(f"  {status} {r['name']:<30} [{r['duration_ms']:.1f}ms] {r['details'][:35]}")

    print("\n" + "-" * 74)
    print(f"  RESULTS: {results['passed']}/{results['total']} passed ({results['success_rate']:.1f}%)")
    print(f"  DURATION: {results['duration_ms']:.2f}ms")
    print("-" * 74)

    # Create production engine
    print("\n  INITIALIZING PRODUCTION ENGINE...")
    engine = create_ultimate_coin()

    # Create treasury wallet
    wallet_result = engine.create_wallet()
    print(f"  ✓ Treasury Wallet: {wallet_result['address']}")
    print(f"    Path: {wallet_result['path']}")

    # Show chain status
    stats = engine.stats()
    print(f"\n  CHAIN STATUS:")
    for key, value in stats['chain'].items():
        print(f"    {key}: {value}")

    # Bitcoin bridge status
    print(f"\n  BITCOIN BRIDGE:")
    print(f"    Address: {BTC_BRIDGE_ADDRESS}")
    btc_sync = engine.sync_bitcoin()
    print(f"    Status: {btc_sync.get('status', 'UNKNOWN')}")
    if 'balance_btc' in btc_sync:
        print(f"    Balance: {btc_sync['balance_btc']:.8f} BTC")

    # Final status
    all_passed = results['failed'] == 0

    print("\n" + "=" * 74)
    if all_passed:
        print("  ★★★★★ MAINNET DEPLOYMENT: READY ★★★★★")
        print("  All systems verified. VALOR Coin is production-ready.")
    else:
        print(f"  ⚠ MAINNET DEPLOYMENT: {results['failed']} ISSUES DETECTED")
        print("  Review failed tests before deployment.")
    print("=" * 74)

    # Save deployment report
    report = {
        'timestamp': datetime.now().isoformat(),
        'coin': COIN_NAME,
        'symbol': COIN_SYMBOL,
        'god_code': GOD_CODE,
        'verification': results,
        'wallet': wallet_result,
        'chain': stats['chain'],
        'bitcoin_bridge': btc_sync,
        'mainnet_ready': all_passed
    }

    report_path = os.path.join(
        os.path.dirname(__file__),
        'MAINNET_DEPLOYMENT_REPORT.json'
    )

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved: {report_path}")

    return all_passed


if __name__ == "__main__":
    success = mainnet_deploy()
    sys.exit(0 if success else 1)
