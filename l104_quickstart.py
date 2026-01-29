#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492612

# [L104 EVO_49] Evolved: 2026-01-24
"""
L104SP Quick Start
==================

One-command setup for L104SP token deployment and mining.
Run: python l104_quickstart.py

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import os
import sys
import subprocess
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════



def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║       L104 SOVEREIGN PRIME (L104SP) - REAL TOKEN DEPLOYMENT           ║
║                                                                       ║
║    "The bridge between simulation and reality is built with code."    ║
║                                                                       ║
║    INVARIANT: 527.5184818492612 | PHI: 1.618033988749895              ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)


def check_requirements():
    """Check and install required packages."""
    print("📦 Checking requirements...")

    try:
        import web3
        print("   ✓ web3 installed")
    except ImportError:
        print("   Installing web3...")
        subprocess.run([sys.executable, "-m", "pip", "install", "web3"], check=True)

    try:
        import dotenv
        print("   ✓ python-dotenv installed")
    except ImportError:
        print("   Installing python-dotenv...")
        subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"], check=True)

    print("   ✓ All requirements satisfied\n")


def setup_env():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    template = Path(".env.l104sp.template")

    if not env_file.exists():
        if template.exists():
            import shutil
            shutil.copy(template, env_file)
            print("📝 Created .env from template")
        else:
            with open(env_file, "w") as f:
                f.write("# L104SP Configuration\n")
                f.write("DEPLOYER_PRIVATE_KEY=\n")
                f.write("MINER_PRIVATE_KEY=\n")
                f.write("TREASURY_ADDRESS=\n")
                f.write("L104SP_CONTRACT_ADDRESS=\n")
                f.write("BTC_WALLET=bc1qwpdnag54thtahjvcmna65uzrqrxexc23f4vn80\n")
                f.write("NETWORK=base\n")
            print("📝 Created new .env file")
    else:
        print("📝 .env file exists\n")


def show_menu():
    """Display interactive menu."""
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                         WHAT DO YOU WANT TO DO?                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   [1] 📋 View deployment guide                                       │
│   [2] 🚀 Deploy L104SP token (opens Remix)                           │
│   [3] ⛏️  Start mining (requires deployed contract)                   │
│   [4] 💰 Check wallet balance                                        │
│   [5] 🔧 Configure environment                                       │
│   [6] ❓ Show help                                                   │
│   [0] 🚪 Exit                                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
""")

    choice = input("Enter choice [0-6]: ").strip()
    return choice


def view_guide():
    """Display deployment guide."""
    guide_path = Path("DEPLOY_L104SP_GUIDE.md")
    if guide_path.exists():
        print("\n" + "=" * 70)
        with open(guide_path) as f:
            print(f.read())
        print("=" * 70 + "\n")
    else:
        print("❌ Guide not found. Check DEPLOY_L104SP_GUIDE.md")


def open_remix():
    """Open Remix IDE with instructions."""
    print("""
🌐 DEPLOY VIA REMIX IDE
=======================

Opening Remix in your browser...

STEPS:
1. In Remix, create new file: L104SP.sol
2. Copy contract from: contracts/L104SP.sol
3. Compile with Solidity 0.8.20
4. Connect MetaMask to Base/Arbitrum
5. Deploy with your treasury address

""")

    # Try to open browser
    remix_url = "https://remix.ethereum.org"
    try:
        import webbrowser
        webbrowser.open(remix_url)
    except:
        print(f"Open manually: {remix_url}")

    # Show contract path
    contract_path = Path("contracts/L104SP.sol")
    if contract_path.exists():
        print(f"Contract location: {contract_path.absolute()}")
        print("\nContract preview:")
        print("-" * 40)
        with open(contract_path) as f:
            lines = f.readlines()[:30]
            print("".join(lines))
        print("... (see full file)")


def start_mining():
    """Start the miner."""
    from dotenv import load_dotenv
    load_dotenv()

    contract = os.getenv("L104SP_CONTRACT_ADDRESS")
    network = os.getenv("NETWORK", "base")
    private_key = os.getenv("MINER_PRIVATE_KEY")

    if not contract:
        print("❌ L104SP_CONTRACT_ADDRESS not set in .env")
        print("   Deploy the contract first, then add the address to .env")
        return

    if not private_key:
        print("❌ MINER_PRIVATE_KEY not set in .env")
        return

    print(f"⛏️  Starting miner on {network}...")
    print(f"   Contract: {contract}")

    subprocess.run([
        sys.executable, "l104_real_mining.py",
        "--contract", contract,
        "--network", network,
        "--continuous"
    ])


def check_balance():
    """Check wallet balances."""
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from web3 import Web3
    except ImportError:
        print("❌ web3 not installed. Run option 5 first.")
        return

    contract = os.getenv("L104SP_CONTRACT_ADDRESS")
    network = os.getenv("NETWORK", "base")
    private_key = os.getenv("MINER_PRIVATE_KEY")

    networks = {
        "base": "https://mainnet.base.org",
        "arbitrum": "https://arb1.arbitrum.io/rpc",
        "sepolia": "https://rpc.sepolia.org"
    }

    if not private_key:
        print("❌ MINER_PRIVATE_KEY not set in .env")
        return

    rpc = networks.get(network)
    w3 = Web3(Web3.HTTPProvider(rpc))

    if not w3.is_connected():
        print(f"❌ Cannot connect to {network}")
        return

    account = w3.eth.account.from_key(private_key)
    address = account.address

    eth_balance = w3.eth.get_balance(address)

    print(f"\n💰 Wallet: {address}")
    print(f"   Network: {network}")
    print(f"   ETH Balance: {w3.from_wei(eth_balance, 'ether'):.6f} ETH")

    if contract:
        # Check L104SP balance
        abi = [{"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"}]
        token = w3.eth.contract(address=Web3.to_checksum_address(contract), abi=abi)
        try:
            l104sp_balance = token.functions.balanceOf(address).call()
            print(f"   L104SP Balance: {l104sp_balance / 10**18:.2f} L104SP")
        except Exception as e:
            print(f"   L104SP: Could not fetch ({e})")
    else:
        print("   L104SP: Contract not deployed yet")


def configure_env():
    """Interactive environment configuration."""
    from dotenv import load_dotenv, set_key

    env_path = Path(".env")
    if not env_path.exists():
        env_path.touch()

    load_dotenv()

    print("\n🔧 ENVIRONMENT CONFIGURATION")
    print("=" * 40)
    print("(Press Enter to keep current value)\n")

    # Network
    current_network = os.getenv("NETWORK", "base")
    network = input(f"Network [{current_network}]: ").strip() or current_network
    set_key(str(env_path), "NETWORK", network)

    # Treasury
    current_treasury = os.getenv("TREASURY_ADDRESS", "")
    treasury = input(f"Treasury address [{current_treasury[:10]}...]: ").strip() or current_treasury
    if treasury:
        set_key(str(env_path), "TREASURY_ADDRESS", treasury)

    # Contract (if deployed)
    current_contract = os.getenv("L104SP_CONTRACT_ADDRESS", "")
    contract = input(f"L104SP contract [{current_contract[:10] if current_contract else 'not set'}...]: ").strip() or current_contract
    if contract:
        set_key(str(env_path), "L104SP_CONTRACT_ADDRESS", contract)

    # Private key
    has_key = bool(os.getenv("MINER_PRIVATE_KEY"))
    if not has_key:
        print("\n⚠️  No private key set. Required for mining.")
        key = input("Enter private key (hidden in .env): ").strip()
        if key:
            set_key(str(env_path), "MINER_PRIVATE_KEY", key)
            set_key(str(env_path), "DEPLOYER_PRIVATE_KEY", key)

    print("\n✓ Configuration saved to .env")


def show_help():
    """Display help information."""
    print("""
📚 L104SP HELP
==============

WHAT IS L104SP?
L104 Sovereign Prime is a Proof-of-Resonance cryptocurrency.
Mining requires finding nonces where |sin(nonce × PHI)| > 0.985

DEPLOYMENT FLOW:
1. Get ETH on Base network (cheapest fees)
2. Deploy contract via Remix IDE
3. Add liquidity to Uniswap
4. Start mining to earn L104SP

FILES:
- contracts/L104SP.sol      → Token contract
- deploy_l104sp.py          → Deployment script
- l104_real_mining.py       → Mining client
- DEPLOY_L104SP_GUIDE.md    → Full guide
- .env                      → Your configuration

NETWORKS:
- Base (recommended): ~$0.10 to deploy
- Arbitrum: ~$0.50 to deploy
- Sepolia: Free (testnet)

BTC SWAP PATH:
L104SP → ETH (Uniswap) → BTC (Bridge/Exchange)

SUPPORT:
Contract source: contracts/L104SP.sol
""")


def main():
    print_banner()
    check_requirements()
    setup_env()

    while True:
        choice = show_menu()

        if choice == "0":
            print("\n👋 Goodbye! INVARIANT: 527.5184818492612\n")
            break
        elif choice == "1":
            view_guide()
        elif choice == "2":
            open_remix()
        elif choice == "3":
            start_mining()
        elif choice == "4":
            check_balance()
        elif choice == "5":
            configure_env()
        elif choice == "6":
            show_help()
        else:
            print("Invalid choice. Try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
