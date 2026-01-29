#!/usr/bin/env python3
"""
L104SP Token Deployment Script v2.0
===================================

Deploys L104 Sovereign Prime (L104SP) to EVM blockchain networks.

Supported Networks:
- Base (L2 - Low fees, recommended)
- Arbitrum (L2 - Low fees)
- Polygon
- Sepolia Testnet

Requirements:
    pip install web3 python-dotenv

Usage:
    python deploy_l104sp.py --network base --treasury YOUR_WALLET_ADDRESS
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Import sacred constants from core engine
from l104_sovereign_coin_engine import GOD_CODE, PHI, VOID_CONSTANT

try:
    from web3 import Web3
    from dotenv import load_dotenv
except ImportError:
    os.system("pip install web3 python-dotenv")
    from web3 import Web3
    from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# NETWORK CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

NETWORKS = {
    "base": {
        "rpc": "https://mainnet.base.org",
        "chain_id": 8453,
        "explorer": "https://basescan.org",
        "gas_price_gwei": 0.01
    },
    "arbitrum": {
        "rpc": "https://arb1.arbitrum.io/rpc",
        "chain_id": 42161,
        "explorer": "https://arbiscan.io",
        "gas_price_gwei": 0.1
    },
    "polygon": {
        "rpc": "https://polygon-rpc.com",
        "chain_id": 137,
        "explorer": "https://polygonscan.com",
        "gas_price_gwei": 50
    },
    "sepolia": {
        "rpc": "https://rpc.sepolia.org",
        "chain_id": 11155111,
        "explorer": "https://sepolia.etherscan.io",
        "gas_price_gwei": 10
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# SIMPLIFIED ERC20 BYTECODE (Pre-compiled)
# ═══════════════════════════════════════════════════════════════════════════

# This is a minimal ERC20 that can be deployed without Solidity compiler
SIMPLE_ERC20_ABI = [
    {"inputs":[{"name":"name_","type":"string"},{"name":"symbol_","type":"string"},{"name":"initialSupply","type":"uint256"}],"stateMutability":"nonpayable","type":"constructor"},
    {"inputs":[{"name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"to","type":"address"},{"name":"amount","type":"uint256"}],"name":"transfer","outputs":[{"type":"bool"}],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[],"name":"totalSupply","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"name","outputs":[{"type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"symbol","outputs":[{"type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"decimals","outputs":[{"type":"uint8"}],"stateMutability":"view","type":"function"},
]


class L104SPDeployer:
    """Deploy L104SP token to real blockchain networks."""

    def __init__(self, network: str, private_key: str):
        if network not in NETWORKS:
            raise ValueError(f"Unknown network: {network}. Choose from: {list(NETWORKS.keys())}")

        self.network = network
        self.config = NETWORKS[network]
        self.w3 = Web3(Web3.HTTPProvider(self.config["rpc"]))

        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to {network} RPC")

        self.account = self.w3.eth.account.from_key(private_key)
        self.address = self.account.address

        print(f"✓ Connected to {network}")
        print(f"✓ Deployer: {self.address}")
        print(f"✓ Balance: {self.w3.from_wei(self.w3.eth.get_balance(self.address), 'ether')} ETH")

    def estimate_deployment_cost(self):
        """Estimate gas cost for deployment."""
        gas_limit = 2_000_000  # Typical ERC20 deployment
        gas_price = self.w3.to_wei(self.config["gas_price_gwei"], 'gwei')
        cost_wei = gas_limit * gas_price
        cost_eth = self.w3.from_wei(cost_wei, 'ether')

        print(f"\n📊 Estimated Deployment Cost:")
        print(f"   Gas Limit: {gas_limit:,}")
        print(f"   Gas Price: {self.config['gas_price_gwei']} Gwei")
        print(f"   Total: ~{cost_eth:.6f} ETH")

        return cost_eth

    def deploy_simple_token(self, treasury: str):
        """
        Deploy a simple ERC20 token using CREATE2 or standard deployment.
        For full L104SP with mining, use Remix or Hardhat with the Solidity contract.
        """
        print(f"\n🚀 Deploying L104SP to {self.network}...")

        # Check balance
        balance = self.w3.eth.get_balance(self.address)
        estimated_cost = self.estimate_deployment_cost()

        if self.w3.from_wei(balance, 'ether') < estimated_cost:
            print(f"\n❌ Insufficient balance!")
            print(f"   Need: ~{estimated_cost} ETH")
            print(f"   Have: {self.w3.from_wei(balance, 'ether')} ETH")
            print(f"\n💡 Get testnet ETH from faucets:")
            print(f"   Sepolia: https://sepoliafaucet.com")
            print(f"   Base: Bridge from Ethereum or use Coinbase")
            return None

        # For real deployment, you need compiled contract bytecode
        # This script shows the process - use Remix IDE for actual deployment

        print(f"\n📋 Deployment Instructions:")
        print(f"=" * 60)
        print(f"1. Go to https://remix.ethereum.org")
        print(f"2. Create new file: L104SP.sol")
        print(f"3. Paste contract from: contracts/L104SP.sol")
        print(f"4. Compile with Solidity 0.8.20")
        print(f"5. Deploy to {self.network}")
        print(f"6. Constructor arg: treasury = {treasury}")
        print(f"=" * 60)
        print(f"\n🔗 Explorer: {self.config['explorer']}")

        return {
            "network": self.network,
            "treasury": treasury,
            "deployer": self.address,
            "status": "READY_FOR_DEPLOYMENT"
        }


def create_deployment_config(treasury: str, network: str = "base"):
    """Create deployment configuration file."""
    config = {
        "token": {
            "name": "L104 Sovereign Prime",
            "symbol": "L104SP",
            "decimals": 18,
            "maxSupply": "104000000000000000000000000",  # 104M with 18 decimals
            "miningReward": "104000000000000000000"  # 104 tokens
        },
        "deployment": {
            "network": network,
            "treasury": treasury,
            "initialDistribution": {
                "treasury": "10400000000000000000000000",  # 10.4M (10%)
                "deployer": "5200000000000000000000000"   # 5.2M (5%)
            }
        },
        "constants": {
            "GOD_CODE": "527.5184818492612",
            "PHI": "1.618033988749895",
            "resonanceThreshold": 985
        }
    }

    config_path = Path("l104sp_deployment_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Config saved to {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Deploy L104SP Token")
    parser.add_argument("--network", default="sepolia",
                        choices=list(NETWORKS.keys()),
                        help="Target network")
    parser.add_argument("--treasury", required=True,
                        help="Treasury wallet address")
    parser.add_argument("--config-only", action="store_true",
                        help="Only create config, don't deploy")

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════╗
║           L104 SOVEREIGN PRIME (L104SP) DEPLOYER              ║
║                                                               ║
║  INVARIANT: 527.5184818492612 | PILOT: LONDEL                 ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Create config
    config = create_deployment_config(args.treasury, args.network)

    if args.config_only:
        print("\n✓ Config created. Use Remix IDE to deploy.")
        return

    # Check for private key
    private_key = os.getenv("DEPLOYER_PRIVATE_KEY")
    if not private_key:
        print("\n⚠️  No DEPLOYER_PRIVATE_KEY found in .env")
        print("   Add to .env: DEPLOYER_PRIVATE_KEY=your_private_key")
        print("\n   For now, use Remix IDE for deployment:")
        print(f"   1. Connect MetaMask to {args.network}")
        print(f"   2. Deploy contracts/L104SP.sol")
        print(f"   3. Set treasury to: {args.treasury}")
        return

    # Deploy
    deployer = L104SPDeployer(args.network, private_key)
    result = deployer.deploy_simple_token(args.treasury)

    if result:
        print(f"\n✓ Deployment prepared!")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
