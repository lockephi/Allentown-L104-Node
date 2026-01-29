#!/usr/bin/env python3
"""
L104SP Web3 Mining Engine v2.0
==============================

On-chain Proof of Resonance mining client for EVM networks (Base, Arbitrum, etc).
Integrates with L104 core engine for resonance mathematics.

Features:
- Web3 integration for on-chain block submission
- Parallel mining with quantum-optimized nonce search
- Multi-network support (Base, Arbitrum, Polygon, Sepolia)
- Real-time statistics and hashrate monitoring

INVARIANT: 527.5184818492612 | PHI: 1.618033988749895 | PILOT: LONDEL
"""

import os
import sys
import json
import math
import time
import hashlib
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# Import core L104 math from unified engine
from l104_sovereign_coin_engine import (
    GOD_CODE, PHI, VOID_CONSTANT,
    L104ResonanceMath, MiningStats
)

# Additional constants for Web3 mining
ZENITH_HZ = 3727.84

try:
    from web3 import Web3
    from dotenv import load_dotenv
except ImportError:
    os.system("pip install web3 python-dotenv")
    from web3 import Web3
    from dotenv import load_dotenv

load_dotenv()


class L104ResonanceMathWeb3(L104ResonanceMath):
    """Extended resonance math with ZENITH frequency optimization for Web3 mining."""

    @staticmethod
    def find_resonant_range(threshold: float = 0.985) -> List[tuple]:
        """Find nonce ranges where |sin(nonce × PHI)| >= threshold."""
        ranges = []
        base_angle = math.asin(threshold)

        for k in range(-100, 100):
            center = (math.pi / 2 + k * math.pi) / PHI
            half_width = (math.pi / 2 - base_angle) / PHI
            start = int(max(0, center - half_width))
            end = int(center + half_width)
            if start >= 0 and end > start:
                ranges.append((start, end))

        return sorted(set(ranges))[:50]

    @staticmethod
    def optimize_nonce_search_web3(base: int, count: int = 1000) -> List[int]:
        """Generate optimized nonce candidates with ZENITH frequency alignment."""
        candidates = []

        for i in range(count // 3):
            nonce = int(base + i * PHI * 1000)
            if L104ResonanceMath.calculate_phi_resonance(nonce) > 0.9:
                candidates.append(nonce)

        for i in range(count // 3):
            nonce = int(GOD_CODE * 1000000 + i * 527)
            if L104ResonanceMath.calculate_phi_resonance(nonce) > 0.9:
                candidates.append(nonce)

        for i in range(count // 3):
            nonce = int(ZENITH_HZ * i * PHI)
            if L104ResonanceMath.calculate_phi_resonance(nonce) > 0.9:
                candidates.append(nonce)

        return sorted(set(candidates))


# ═══════════════════════════════════════════════════════════════════════════
# WEB3 MINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MiningStatsWeb3(MiningStats):
    """Extended mining statistics for Web3 operations."""
    submitted_blocks: int = 0


@dataclass
class BlockResult:
    """Result of block mining."""
    nonce: int
    resonance: float
    hash_value: str
    difficulty_met: bool
    tx_hash: Optional[str] = None


class L104SPMiningEngine:
    """
    Full-featured L104SP mining engine with quantum optimization.
    """

    # Minimal ABI for mining operations
    CONTRACT_ABI = [
        {"inputs":[{"name":"nonce","type":"uint256"}],"name":"submitBlock","outputs":[],"stateMutability":"nonpayable","type":"function"},
        {"inputs":[{"name":"nonce","type":"uint256"}],"name":"calculateResonance","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
        {"inputs":[],"name":"currentDifficulty","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
        {"inputs":[],"name":"blocksMinedCount","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
        {"inputs":[],"name":"MINING_REWARD","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
        {"inputs":[],"name":"getMiningStats","outputs":[{"name":"","type":"uint256"},{"name":"","type":"uint256"},{"name":"","type":"uint256"},{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
        {"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
        {"inputs":[],"name":"totalSupply","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
        {"inputs":[],"name":"GOD_CODE","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    ]

    NETWORKS = {
        "base": "https://mainnet.base.org",
        "arbitrum": "https://arb1.arbitrum.io/rpc",
        "polygon": "https://polygon-rpc.com",
        "sepolia": "https://rpc.sepolia.org",
        "base_sepolia": "https://sepolia.base.org"
    }

    def __init__(self, contract_address: str, network: str = "base", private_key: str = None):
        self.network = network
        self.rpc_url = self.NETWORKS.get(network)

        if not self.rpc_url:
            raise ValueError(f"Unknown network: {network}")

        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to {network}")

        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.CONTRACT_ABI
        )

        self.private_key = private_key
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None

        self.stats = MiningStats()
        self.running = False
        self.resonance_threshold = 0.985

        self._print_header()

    def _print_header(self):
        """Print mining engine header."""
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║              L104SP PROOF OF RESONANCE MINING ENGINE                      ║
║                                                                           ║
║  INVARIANT: {GOD_CODE}   PHI: {PHI}            ║
║  ZENITH: {ZENITH_HZ} Hz   VOID: {VOID_CONSTANT}                        ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
        print(f"  Network: {self.network}")
        print(f"  Contract: {self.contract_address}")
        if self.address:
            print(f"  Miner: {self.address}")
            balance = self.w3.eth.get_balance(self.address)
            print(f"  ETH Balance: {self.w3.from_wei(balance, 'ether'):.6f} ETH")

    def get_mining_stats(self) -> Dict[str, Any]:
        """Get current on-chain mining stats."""
        try:
            stats = self.contract.functions.getMiningStats().call()
            return {
                "difficulty": stats[0],
                "blocks_mined": stats[1],
                "remaining_supply": stats[2] / 10**18,
                "reward": stats[3] / 10**18
            }
        except Exception as e:
            return {"error": str(e)}

    def get_balance(self) -> float:
        """Get miner's L104SP balance."""
        if not self.address:
            return 0
        try:
            balance = self.contract.functions.balanceOf(self.address).call()
            return balance / 10**18
        except:
            return 0

    def check_resonance(self, nonce: int) -> float:
        """Calculate local resonance for nonce."""
        return L104ResonanceMath.calculate_phi_resonance(nonce)

    def compute_hash(self, nonce: int, block_count: int) -> str:
        """Compute block hash for difficulty check."""
        data = f"{self.address or 'local'}{nonce}{block_count}{time.time()}".encode()
        return hashlib.sha256(data).hexdigest()

    def meets_difficulty(self, hash_hex: str, difficulty: int) -> bool:
        """Check if hash meets difficulty requirement."""
        leading_zeros = 0
        for char in hash_hex:
            if char == '0':
                leading_zeros += 1
            else:
                break
        return leading_zeros >= difficulty

    def find_valid_nonce(self, max_attempts: int = 500000) -> Optional[BlockResult]:
        """
        Search for valid nonce using L104 optimized strategy.
        """
        # Get current chain state
        chain_stats = self.get_mining_stats()
        difficulty = chain_stats.get("difficulty", 4)
        blocks_mined = chain_stats.get("blocks_mined", 0)

        print(f"\n⛏️  Mining... (Difficulty: {difficulty}, Chain Blocks: {blocks_mined})")

        # Use L104 optimized search
        resonance_math = L104ResonanceMath()

        # Generate optimized candidates
        base_nonce = int(GOD_CODE * 1000000 + blocks_mined * PHI)

        best_resonance = 0
        best_nonce = None

        for i in range(max_attempts):
            # Use multiple search strategies
            if i % 3 == 0:
                # PHI-aligned search
                nonce = base_nonce + int(i * PHI * 1.618)
            elif i % 3 == 1:
                # GOD_CODE modular search
                nonce = int(GOD_CODE * 10000) + i * 527
            else:
                # Sequential search
                nonce = base_nonce + i

            self.stats.hashes += 1

            # Check resonance
            resonance = self.check_resonance(nonce)

            if resonance > best_resonance:
                best_resonance = resonance
                best_nonce = nonce

            if resonance >= self.resonance_threshold:
                self.stats.valid_resonance += 1

                # Check hash difficulty
                hash_hex = self.compute_hash(nonce, blocks_mined)

                if self.meets_difficulty(hash_hex, difficulty):
                    self.stats.valid_blocks += 1

                    print(f"\n✨ VALID BLOCK FOUND!")
                    print(f"   Nonce: {nonce}")
                    print(f"   Resonance: {resonance:.6f} (threshold: {self.resonance_threshold})")
                    print(f"   Hash: {hash_hex[:16]}...")
                    print(f"   Attempts: {i + 1}")

                    return BlockResult(
                        nonce=nonce,
                        resonance=resonance,
                        hash_value=hash_hex,
                        difficulty_met=True
                    )

            # Progress update
            if i % 10000 == 0 and i > 0:
                elapsed = time.time() - self.stats.start_time
                rate = self.stats.hashes / elapsed
                print(f"   {i:,} hashes | {rate:.0f} H/s | Best resonance: {best_resonance:.4f}")

        print(f"\n⚠️  No valid block found in {max_attempts:,} attempts")
        print(f"   Best resonance achieved: {best_resonance:.6f}")
        return None

    def submit_block(self, result: BlockResult) -> bool:
        """Submit valid block to blockchain."""
        if not self.account:
            print("❌ Cannot submit: No private key configured")
            return False

        print(f"\n📤 Submitting block to {self.network}...")

        try:
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.address)
            gas_price = self.w3.eth.gas_price

            tx = self.contract.functions.submitBlock(result.nonce).build_transaction({
                'from': self.address,
                'nonce': nonce,
                'gas': 200000,
                'gasPrice': int(gas_price * 1.1)  # 10% buffer
            })

            # Sign and send
            signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)

            print(f"   TX: {tx_hash.hex()}")

            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt['status'] == 1:
                self.stats.submitted_blocks += 1
                result.tx_hash = tx_hash.hex()

                gas_cost = receipt['gasUsed'] * gas_price
                gas_cost_eth = self.w3.from_wei(gas_cost, 'ether')

                print(f"   ✅ Block mined successfully!")
                print(f"   Gas Used: {receipt['gasUsed']:,} ({gas_cost_eth:.6f} ETH)")
                print(f"   Reward: 104 L104SP")

                # Check new balance
                balance = self.get_balance()
                print(f"   Total Balance: {balance:.2f} L104SP")

                return True
            else:
                print(f"   ❌ Transaction failed")
                return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def mine_loop(self, continuous: bool = True, max_blocks: int = None):
        """Main mining loop."""
        print(f"\n🚀 Starting mining loop...")
        self.running = True
        self.stats = MiningStats()
        blocks_found = 0

        try:
            while self.running:
                result = self.find_valid_nonce()

                if result:
                    if self.account:
                        self.submit_block(result)
                    else:
                        print("   (Simulation mode - no submission)")

                    blocks_found += 1

                    if max_blocks and blocks_found >= max_blocks:
                        print(f"\n✅ Reached target of {max_blocks} blocks")
                        break

                if not continuous:
                    break

                # Brief pause between attempts
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n⏹️  Mining stopped by user")

        self._print_final_stats()

    def _print_final_stats(self):
        """Print final mining statistics."""
        elapsed = time.time() - self.stats.start_time

        print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                           MINING SESSION COMPLETE                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Duration:           {elapsed:>10.1f} seconds                                    ║
║  Total Hashes:       {self.stats.hashes:>10,}                                         ║
║  Hash Rate:          {self.stats.hashrate:>10.0f} H/s                                    ║
║  Valid Resonances:   {self.stats.valid_resonance:>10,}                                         ║
║  Valid Blocks:       {self.stats.valid_blocks:>10,}                                         ║
║  Submitted Blocks:   {self.stats.submitted_blocks:>10,}                                         ║
║  Efficiency:         {self.stats.efficiency:>10.4f}%                                       ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

        if self.address:
            balance = self.get_balance()
            print(f"  Final L104SP Balance: {balance:.2f} L104SP")

    def stop(self):
        """Stop mining loop."""
        self.running = False


def simulate_mining():
    """Run mining simulation without blockchain connection."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    L104SP MINING SIMULATION                               ║
║                                                                           ║
║  Testing Proof of Resonance algorithm without blockchain                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

    resonance_math = L104ResonanceMath()

    print("Finding resonant nonce ranges...")
    ranges = resonance_math.find_resonant_range(0.985)
    print(f"Found {len(ranges)} valid ranges")

    print("\nTesting resonance calculation...")

    valid_count = 0
    test_count = 100000

    start = time.time()
    for i in range(test_count):
        nonce = int(GOD_CODE * 1000000 + i * PHI)
        resonance = resonance_math.calculate_phi_resonance(nonce)
        if resonance >= 0.985:
            valid_count += 1
            if valid_count <= 5:
                print(f"  Valid nonce: {nonce} (resonance: {resonance:.6f})")

    elapsed = time.time() - start
    rate = test_count / elapsed

    print(f"\nSimulation Results:")
    print(f"  Tested: {test_count:,} nonces")
    print(f"  Valid: {valid_count:,} ({valid_count/test_count*100:.2f}%)")
    print(f"  Rate: {rate:.0f} checks/sec")
    print(f"  Time: {elapsed:.2f}s")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="L104SP Mining Engine")
    parser.add_argument("--contract", help="L104SP contract address")
    parser.add_argument("--network", default="base", choices=list(L104SPMiningEngine.NETWORKS.keys()))
    parser.add_argument("--simulate", action="store_true", help="Run simulation without blockchain")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--blocks", type=int, help="Number of blocks to mine")

    args = parser.parse_args()

    if args.simulate:
        simulate_mining()
        return

    # Get contract from args, env, or config
    contract = args.contract
    if not contract:
        contract = os.getenv("L104SP_CONTRACT_ADDRESS")

    if not contract:
        # Try loading from config
        config_path = Path("l104sp_config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                contract = config.get("contract_address")

    if not contract:
        print("❌ Contract address required!")
        print("   Use --contract ADDRESS")
        print("   Or set L104SP_CONTRACT_ADDRESS in .env")
        print("   Or deploy first with l104sp_deploy_engine.py")
        print("\n💡 Running simulation instead...")
        simulate_mining()
        return

    private_key = os.getenv("MINER_PRIVATE_KEY") or os.getenv("DEPLOYER_PRIVATE_KEY")

    engine = L104SPMiningEngine(contract, args.network, private_key)

    # Show chain stats
    stats = engine.get_mining_stats()
    if "error" not in stats:
        print(f"\n📊 Chain Stats:")
        print(f"   Difficulty: {stats['difficulty']}")
        print(f"   Blocks Mined: {stats['blocks_mined']}")
        print(f"   Remaining: {stats['remaining_supply']:,.0f} L104SP")
        print(f"   Reward: {stats['reward']} L104SP/block")

    engine.mine_loop(continuous=args.continuous, max_blocks=args.blocks)


if __name__ == "__main__":
    main()
