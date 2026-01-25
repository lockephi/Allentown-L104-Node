#!/usr/bin/env python3
"""
L104SP Real Mining Client
=========================

Mines L104SP tokens on-chain by finding valid nonces that satisfy:
1. Hash difficulty requirement
2. PHI resonance requirement: |sin(nonce × PHI)| > 0.985

This connects to a deployed L104SP contract and submits valid blocks.

Usage:
    python l104_real_mining.py --contract 0x... --network base
"""

import os
import sys
import time
import math
import hashlib
import argparse
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


try:
    from web3 import Web3
    from dotenv import load_dotenv
except ImportError:
    print("Installing required packages...")
    os.system("pip install web3 python-dotenv")
    from web3 import Web3
    from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
RESONANCE_THRESHOLD = 0.985

NETWORKS = {
    "base": "https://mainnet.base.org",
    "arbitrum": "https://arb1.arbitrum.io/rpc",
    "polygon": "https://polygon-rpc.com",
    "sepolia": "https://rpc.sepolia.org"
}

# Minimal ABI for mining
MINING_ABI = [
    {"inputs":[{"name":"nonce","type":"uint256"}],"name":"submitBlock","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"nonce","type":"uint256"}],"name":"calculateResonance","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"currentDifficulty","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"blocksMinedd","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"MINING_REWARD","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"getMiningStats","outputs":[{"name":"difficulty","type":"uint256"},{"name":"blocksMined","type":"uint256"},{"name":"remainingSupply","type":"uint256"},{"name":"reward","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
]


@dataclass
class MiningResult:
    """Result of mining attempt."""
    nonce: int
    resonance: float
    hash_value: str
    valid: bool
    submitted: bool = False
    tx_hash: Optional[str] = None


class L104SPMiner:
    """
    Real on-chain miner for L104SP tokens.
    Finds nonces that satisfy Proof of Resonance requirements.
    """
    
    def __init__(self, contract_address: str, network: str, private_key: str):
        self.network = network
        self.rpc_url = NETWORKS.get(network)
        if not self.rpc_url:
            raise ValueError(f"Unknown network: {network}")
        
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to {network}")
        
        self.account = self.w3.eth.account.from_key(private_key)
        self.address = self.account.address
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=MINING_ABI
        )
        
        # Mining stats
        self.hashes_computed = 0
        self.blocks_found = 0
        self.start_time = time.time()
        self.running = False
        
        print(f"⛏️  L104SP Miner Initialized")
        print(f"   Network: {network}")
        print(f"   Contract: {contract_address}")
        print(f"   Miner: {self.address}")
    
    def check_resonance(self, nonce: int) -> float:
        """
        Check if nonce satisfies PHI resonance requirement.
        |sin(nonce × PHI)| > 0.985
        """
        resonance = abs(math.sin(nonce * PHI))
        return resonance
    
    def compute_hash(self, nonce: int, block_number: int) -> str:
        """Compute block hash for difficulty check."""
        data = f"{self.address}{nonce}{block_number}{time.time()}".encode()
        return hashlib.sha256(data).hexdigest()
    
    def meets_difficulty(self, hash_hex: str, difficulty: int) -> bool:
        """Check if hash meets difficulty requirement."""
        leading_zeros = len(hash_hex) - len(hash_hex.lstrip('0'))
        return leading_zeros >= difficulty
    
    def find_valid_nonce(self, max_attempts: int = 1_000_000) -> Optional[MiningResult]:
        """
        Search for a nonce that satisfies both requirements.
        """
        try:
            stats = self.contract.functions.getMiningStats().call()
            difficulty = stats[0]
            blocks_mined = stats[1]
        except Exception as e:
            print(f"⚠️  Could not fetch contract stats: {e}")
            difficulty = 4
            blocks_mined = 0
        
        print(f"\n🔍 Mining... (Difficulty: {difficulty}, Blocks: {blocks_mined})")
        
        # Start from GOD_CODE aligned nonce
        base_nonce = int(GOD_CODE * 1000000) % (2**32)
        
        for i in range(max_attempts):
            nonce = base_nonce + i
            self.hashes_computed += 1
            
            # Check resonance first (faster)
            resonance = self.check_resonance(nonce)
            
            if resonance >= RESONANCE_THRESHOLD:
                # Resonance valid, check hash difficulty
                hash_hex = self.compute_hash(nonce, blocks_mined)
                
                if self.meets_difficulty(hash_hex, difficulty):
                    print(f"\n✨ VALID NONCE FOUND!")
                    print(f"   Nonce: {nonce}")
                    print(f"   Resonance: {resonance:.6f}")
                    print(f"   Hash: {hash_hex[:16]}...")
                    
                    return MiningResult(
                        nonce=nonce,
                        resonance=resonance,
                        hash_value=hash_hex,
                        valid=True
                    )
            
            # Progress update every 10000 hashes
            if i % 10000 == 0 and i > 0:
                elapsed = time.time() - self.start_time
                hashrate = self.hashes_computed / elapsed
                print(f"   {i:,} hashes... ({hashrate:.0f} H/s)")
        
        return None
    
    def submit_block(self, result: MiningResult) -> bool:
        """Submit valid block to blockchain."""
        if not result.valid:
            return False
        
        print(f"\n📤 Submitting block to {self.network}...")
        
        try:
            # Build transaction
            tx = self.contract.functions.submitBlock(result.nonce).build_transaction({
                'from': self.address,
                'nonce': self.w3.eth.get_transaction_count(self.address),
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send
            signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
            
            print(f"   TX Hash: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                print(f"✅ Block mined successfully!")
                self.blocks_found += 1
                result.submitted = True
                result.tx_hash = tx_hash.hex()
                return True
            else:
                print(f"❌ Transaction failed!")
                return False
                
        except Exception as e:
            print(f"❌ Submission error: {e}")
            return False
    
    def mine_loop(self, continuous: bool = True):
        """Main mining loop."""
        print(f"\n⛏️  Starting mining loop...")
        self.running = True
        self.start_time = time.time()
        
        while self.running:
            result = self.find_valid_nonce()
            
            if result:
                success = self.submit_block(result)
                if success:
                    # Check balance
                    balance = self.contract.functions.balanceOf(self.address).call()
                    balance_formatted = balance / 10**18
                    print(f"💰 Balance: {balance_formatted:.2f} L104SP")
            
            if not continuous:
                break
            
            # Brief pause between attempts
            time.sleep(1)
        
        # Final stats
        elapsed = time.time() - self.start_time
        print(f"\n📊 Mining Stats:")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Hashes: {self.hashes_computed:,}")
        print(f"   Hashrate: {self.hashes_computed/elapsed:.0f} H/s")
        print(f"   Blocks Found: {self.blocks_found}")
    
    def stop(self):
        """Stop mining."""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="L104SP Real Miner")
    parser.add_argument("--contract", required=True, help="L104SP contract address")
    parser.add_argument("--network", default="base", choices=list(NETWORKS.keys()))
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              L104SP PROOF OF RESONANCE MINER                  ║
║                                                               ║
║  INVARIANT: 527.5184818492537 | PHI: 1.618033988749895        ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    private_key = os.getenv("MINER_PRIVATE_KEY")
    if not private_key:
        print("❌ Set MINER_PRIVATE_KEY in .env file")
        print("   MINER_PRIVATE_KEY=your_private_key_here")
        return
    
    miner = L104SPMiner(args.contract, args.network, private_key)
    
    try:
        miner.mine_loop(continuous=args.continuous)
    except KeyboardInterrupt:
        print("\n⏹️  Mining stopped by user")
        miner.stop()


if __name__ == "__main__":
    main()
