#!/usr/bin/env python3
"""
L104SP-CLI - Command Line Tools for L104 Sovereign Prime
═══════════════════════════════════════════════════════════════════════════════

A comprehensive CLI for interacting with the L104SP blockchain.

Commands:
    l104sp-cli wallet new          Create a new wallet
    l104sp-cli wallet import       Import wallet from mnemonic
    l104sp-cli wallet balance      Check balance
    l104sp-cli wallet address      Get receiving address

    l104sp-cli send <addr> <amt>   Send L104SP
    l104sp-cli receive             Show receiving address with QR

    l104sp-cli node status         Node status
    l104sp-cli node start          Start local node
    l104sp-cli node stop           Stop local node

    l104sp-cli mine start          Start mining
    l104sp-cli mine stop           Stop mining
    l104sp-cli mine status         Mining stats

    l104sp-cli block <height>      Get block info
    l104sp-cli tx <txid>           Get transaction info

    l104sp-cli config              Show/edit configuration
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from l104_sovereign_coin_engine import (
        L104SPNode, L104SPBlockchain, HDWallet,
        GOD_CODE, PHI, COIN_SYMBOL, SATOSHI_PER_COIN, DATA_DIR
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "1.0.0"
DEFAULT_DATA_DIR = Path(os.environ.get('L104SP_DATA', str(DATA_DIR)))
CONFIG_FILE = DEFAULT_DATA_DIR / 'config.json'
WALLET_FILE = DEFAULT_DATA_DIR / 'wallet.json'
RPC_URL = os.environ.get('L104SP_RPC', 'http://127.0.0.1:10401')


def get_config() -> Dict[str, Any]:
    """Load configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'rpc_url': RPC_URL,
        'data_dir': str(DEFAULT_DATA_DIR),
        'network': 'mainnet'
    }


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def rpc_call(method: str, params: List[Any] = None, url: str = None) -> Any:
    """Make an RPC call to the node."""
    import urllib.request
    import urllib.error

    url = url or get_config().get('rpc_url', RPC_URL)

    payload = json.dumps({
        'jsonrpc': '2.0',
        'id': 1,
        'method': method,
        'params': params or []
    }).encode()

    try:
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.load(resp)
            if 'error' in result:
                raise Exception(result['error'].get('message', 'RPC Error'))
            return result.get('result')
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot connect to node at {url}: {e}")


def rest_call(endpoint: str, url: str = None) -> Any:
    """Make a REST call to the node."""
    import urllib.request
    import urllib.error

    url = url or get_config().get('rpc_url', RPC_URL)
    full_url = f"{url.rstrip('/')}{endpoint}"

    try:
        with urllib.request.urlopen(full_url, timeout=10) as resp:
            return json.load(resp)
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot connect to node: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# WALLET COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_wallet_new(args) -> int:
    """Create a new wallet."""
    if WALLET_FILE.exists() and not args.force:
        print(f"⚠️  Wallet already exists at {WALLET_FILE}")
        print("   Use --force to overwrite")
        return 1

    print("🔐 Creating new L104SP HD Wallet...")

    if not ENGINE_AVAILABLE:
        print("❌ Engine not available. Install dependencies.")
        return 1

    # Generate mnemonic first
    temp_wallet = HDWallet()
    mnemonic = temp_wallet.generate_mnemonic()

    # Create wallet from mnemonic
    wallet = HDWallet(mnemonic=mnemonic)

    # Save wallet
    WALLET_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'seed_hex': wallet.seed.hex(),
        'created_at': int(time.time()),
        'network': 'mainnet'
    }
    with open(WALLET_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    # Get first address
    address, _ = wallet.get_address(0)

    print("\n✅ Wallet created successfully!")
    print(f"\n📁 Saved to: {WALLET_FILE}")
    print(f"\n📬 Your receiving address:")
    print(f"   {address}")
    print(f"\n🔑 BACKUP YOUR MNEMONIC PHRASE:")
    print("=" * 60)
    print(f"   {mnemonic}")
    print("=" * 60)
    print("\n⚠️  Store this phrase securely! Anyone with it can access your funds.")

    return 0


def cmd_wallet_import(args) -> int:
    """Import wallet from mnemonic."""
    if WALLET_FILE.exists() and not args.force:
        print(f"⚠️  Wallet already exists at {WALLET_FILE}")
        print("   Use --force to overwrite")
        return 1

    if not ENGINE_AVAILABLE:
        print("❌ Engine not available")
        return 1

    # Get mnemonic from args or prompt
    if args.mnemonic:
        mnemonic = args.mnemonic
    else:
        print("Enter your 12 or 24 word mnemonic phrase:")
        mnemonic = input("> ").strip()

    if not mnemonic or len(mnemonic.split()) not in [12, 24]:
        print("❌ Invalid mnemonic. Must be 12 or 24 words.")
        return 1

    print("🔐 Importing wallet...")

    wallet = HDWallet(mnemonic=mnemonic)

    # Save wallet
    WALLET_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'seed_hex': wallet.seed.hex(),
        'imported_at': int(time.time()),
        'network': 'mainnet'
    }
    with open(WALLET_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    # Get first address
    address, _ = wallet.get_address(0)

    print("\n✅ Wallet imported successfully!")
    print(f"📬 Address: {address}")

    return 0


def cmd_wallet_balance(args) -> int:
    """Check wallet balance."""
    try:
        # Try RPC first
        result = rest_call('/status')

        if not WALLET_FILE.exists():
            print("❌ No wallet found. Create one with: l104sp-cli wallet new")
            return 1

        # Load wallet
        with open(WALLET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        seed = bytes.fromhex(data['seed_hex'])
        wallet = HDWallet(seed=seed)

        print(f"\n💰 L104SP Wallet Balance")
        print("-" * 50)

        total = 0
        for i in range(10):
            addr, _ = wallet.get_address(i)
            # Query balance via RPC
            try:
                balance_info = rest_call(f'/balance/{addr}')
                balance = balance_info.get('balance', 0)
            except Exception:
                balance = 0

            if balance > 0 or i < 3:
                print(f"  [{i}] {addr[:20]}... : {balance / SATOSHI_PER_COIN:.8f} L104SP")
                total += balance

        print("-" * 50)
        print(f"  TOTAL: {total / SATOSHI_PER_COIN:.8f} L104SP")

        return 0

    except ConnectionError:
        print("❌ Cannot connect to node. Is it running?")
        print(f"   Start with: python l104sp_mainnet.py")
        return 1


def cmd_wallet_address(args) -> int:
    """Show receiving address."""
    if not WALLET_FILE.exists():
        print("❌ No wallet found. Create one with: l104sp-cli wallet new")
        return 1

    if not ENGINE_AVAILABLE:
        print("❌ Engine not available")
        return 1

    with open(WALLET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    seed = bytes.fromhex(data['seed_hex'])
    wallet = HDWallet(seed=seed)

    index = args.index or 0
    address, _ = wallet.get_address(index)

    print(f"\n📬 Receiving Address [{index}]:")
    print(f"   {address}")

    # Show QR if requested
    if args.qr:
        try:
            import qrcode
            qr = qrcode.QRCode(box_size=1, border=1)
            qr.add_data(address)
            qr.print_ascii()
        except ImportError:
            print("\n(Install 'qrcode' for QR display: pip install qrcode)")

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# SEND/RECEIVE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_send(args) -> int:
    """Send L104SP to an address."""
    if not args.address or not args.amount:
        print("Usage: l104sp-cli send <address> <amount>")
        return 1

    try:
        amount = float(args.amount)
        if amount <= 0:
            print("❌ Amount must be positive")
            return 1
    except ValueError:
        print("❌ Invalid amount")
        return 1

    print(f"\n📤 Send L104SP")
    print("-" * 40)
    print(f"  To:     {args.address}")
    print(f"  Amount: {amount} L104SP")
    print(f"  Fee:    {args.fee or 0.001} L104SP")

    confirm = input("\nConfirm send? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Transaction cancelled")
        return 1

    try:
        result = rpc_call('sendtoaddress', [args.address, amount])
        print(f"\n✅ Transaction sent!")
        print(f"   TXID: {result}")
        return 0
    except ConnectionError:
        print("❌ Cannot connect to node")
        return 1
    except Exception as e:
        print(f"❌ Send failed: {e}")
        return 1


def cmd_receive(args) -> int:
    """Show receiving address with QR."""
    return cmd_wallet_address(args)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_node_status(args) -> int:
    """Show node status."""
    try:
        status = rest_call('/status')

        print(f"\n🔷 L104SP Node Status")
        print("-" * 40)
        print(f"  Version:  {status.get('version', 'unknown')}")
        print(f"  Network:  {status.get('network', 'mainnet')}")

        blockchain = status.get('blockchain', {})
        print(f"  Height:   {blockchain.get('height', 0)}")
        print(f"  Tip:      {blockchain.get('tip', 'unknown')[:16]}...")
        print(f"  Peers:    {status.get('peers', 0)}")

        mining = status.get('mining', {})
        print(f"  Mining:   {mining.get('hashrate', '0 H/s')}")

        return 0

    except ConnectionError:
        print("❌ Node not running")
        print("   Start with: python l104sp_mainnet.py")
        return 1


def cmd_node_start(args) -> int:
    """Start the node."""
    print("🚀 Starting L104SP node...")

    cmd = [sys.executable, 'l104sp_mainnet.py', '--daemon']
    if args.mine:
        cmd.append('--mine')

    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(0.5)  # QUANTUM AMPLIFIED (was 2)

        # Check if started
        try:
            rest_call('/status')
            print("✅ Node started successfully")
            print("   RPC: http://127.0.0.1:10401/status")
            return 0
        except Exception:
            print("⏳ Node starting... (may take a few seconds)")
            return 0

    except Exception as e:
        print(f"❌ Failed to start node: {e}")
        return 1


def cmd_node_stop(args) -> int:
    """Stop the node."""
    try:
        rpc_call('stop')
        print("✅ Node stopped")
        return 0
    except ConnectionError:
        print("❌ Node not running")
        return 1


# ═══════════════════════════════════════════════════════════════════════════════
# MINING COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_mine_start(args) -> int:
    """Start mining."""
    try:
        # Get address
        if not WALLET_FILE.exists():
            print("❌ No wallet found. Create one first.")
            return 1

        with open(WALLET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        seed = bytes.fromhex(data['seed_hex'])
        wallet = HDWallet(seed=seed)
        address, _ = wallet.get_address(0)

        print(f"⛏️  Starting mining to: {address}")

        result = rpc_call('startmining', [address])
        print("✅ Mining started")
        return 0

    except ConnectionError:
        print("❌ Cannot connect to node")
        return 1


def cmd_mine_stop(args) -> int:
    """Stop mining."""
    try:
        rpc_call('stopmining')
        print("✅ Mining stopped")
        return 0
    except ConnectionError:
        print("❌ Cannot connect to node")
        return 1


def cmd_mine_status(args) -> int:
    """Show mining status."""
    try:
        result = rest_call('/mining')

        print(f"\n⛏️  Mining Status")
        print("-" * 40)
        print(f"  Running:   {'Yes' if result.get('running') else 'No'}")
        print(f"  Hashrate:  {result.get('hashrate', 0):.2f} H/s")
        print(f"  Hashes:    {result.get('hashes', 0):,}")
        print(f"  Blocks:    {result.get('blocks', 0)}")

        return 0

    except ConnectionError:
        print("❌ Cannot connect to node")
        return 1


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK/TX COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_block(args) -> int:
    """Get block info."""
    height = args.height

    try:
        if height == 'latest':
            block = rest_call('/block/latest')
        else:
            block = rest_call(f'/block/{height}')

        if 'error' in block:
            print(f"❌ Block not found")
            return 1

        print(f"\n📦 Block {block.get('height', height)}")
        print("-" * 60)
        print(f"  Hash:       {block.get('hash', 'unknown')}")
        print(f"  Prev:       {block.get('prev_block', 'unknown')[:32]}...")
        print(f"  Time:       {block.get('timestamp', 0)}")
        print(f"  Nonce:      {block.get('nonce', 0)}")
        print(f"  Resonance:  {block.get('resonance', 0):.6f}")
        print(f"  Txs:        {len(block.get('transactions', []))}")

        return 0

    except ConnectionError:
        print("❌ Cannot connect to node")
        return 1


def cmd_tx(args) -> int:
    """Get transaction info."""
    txid = args.txid

    try:
        tx = rest_call(f'/tx/{txid}')

        if 'error' in tx:
            print(f"❌ Transaction not found")
            return 1

        print(f"\n💳 Transaction")
        print("-" * 60)
        print(f"  TXID:    {tx.get('txid', txid)}")
        print(f"  Status:  {tx.get('status', 'unknown')}")
        print(f"  Block:   {tx.get('block_height', 'pending')}")
        print(f"  Inputs:  {len(tx.get('inputs', []))}")
        print(f"  Outputs: {len(tx.get('outputs', []))}")

        return 0

    except ConnectionError:
        print("❌ Cannot connect to node")
        return 1


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_config(args) -> int:
    """Show/edit configuration."""
    config = get_config()

    if args.set:
        key, value = args.set.split('=', 1)
        config[key.strip()] = value.strip()
        save_config(config)
        print(f"✅ Set {key} = {value}")
        return 0

    print(f"\n⚙️  L104SP Configuration")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nConfig file: {CONFIG_FILE}")

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='l104sp-cli',
        description='L104SP Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  l104sp-cli wallet new              Create new wallet
  l104sp-cli wallet balance          Check balance
  l104sp-cli send l104q... 10.5      Send 10.5 L104SP
  l104sp-cli node status             Check node status
  l104sp-cli mine start              Start mining

Version: {VERSION}
GOD_CODE: {GOD_CODE}
        """
    )
    parser.add_argument('--version', action='version', version=f'l104sp-cli v{VERSION}')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Wallet commands
    wallet_parser = subparsers.add_parser('wallet', help='Wallet operations')
    wallet_sub = wallet_parser.add_subparsers(dest='wallet_cmd')

    wallet_new = wallet_sub.add_parser('new', help='Create new wallet')
    wallet_new.add_argument('--passphrase', '-p', help='Optional passphrase')
    wallet_new.add_argument('--force', '-f', action='store_true', help='Overwrite existing')

    wallet_import = wallet_sub.add_parser('import', help='Import from mnemonic')
    wallet_import.add_argument('--mnemonic', '-m', help='Mnemonic phrase')
    wallet_import.add_argument('--passphrase', '-p', help='Optional passphrase')
    wallet_import.add_argument('--force', '-f', action='store_true', help='Overwrite existing')

    wallet_balance = wallet_sub.add_parser('balance', help='Check balance')

    wallet_address = wallet_sub.add_parser('address', help='Show address')
    wallet_address.add_argument('--index', '-i', type=int, help='Address index')
    wallet_address.add_argument('--qr', action='store_true', help='Show QR code')

    # Send command
    send_parser = subparsers.add_parser('send', help='Send L104SP')
    send_parser.add_argument('address', help='Recipient address')
    send_parser.add_argument('amount', help='Amount to send')
    send_parser.add_argument('--fee', '-f', type=float, default=0.001, help='Transaction fee')

    # Receive command
    receive_parser = subparsers.add_parser('receive', help='Show receiving address')
    receive_parser.add_argument('--qr', action='store_true', help='Show QR code')
    receive_parser.add_argument('--index', '-i', type=int, help='Address index')

    # Node commands
    node_parser = subparsers.add_parser('node', help='Node operations')
    node_sub = node_parser.add_subparsers(dest='node_cmd')

    node_status = node_sub.add_parser('status', help='Show node status')

    node_start = node_sub.add_parser('start', help='Start node')
    node_start.add_argument('--mine', '-m', action='store_true', help='Enable mining')

    node_stop = node_sub.add_parser('stop', help='Stop node')

    # Mine commands
    mine_parser = subparsers.add_parser('mine', help='Mining operations')
    mine_sub = mine_parser.add_subparsers(dest='mine_cmd')

    mine_start = mine_sub.add_parser('start', help='Start mining')
    mine_stop = mine_sub.add_parser('stop', help='Stop mining')
    mine_status = mine_sub.add_parser('status', help='Mining status')

    # Block command
    block_parser = subparsers.add_parser('block', help='Get block info')
    block_parser.add_argument('height', help='Block height or "latest"')

    # TX command
    tx_parser = subparsers.add_parser('tx', help='Get transaction info')
    tx_parser.add_argument('txid', help='Transaction ID')

    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration')
    config_parser.add_argument('--set', '-s', help='Set key=value')

    args = parser.parse_args()

    # Dispatch commands
    if args.command == 'wallet':
        if args.wallet_cmd == 'new':
            return cmd_wallet_new(args)
        elif args.wallet_cmd == 'import':
            return cmd_wallet_import(args)
        elif args.wallet_cmd == 'balance':
            return cmd_wallet_balance(args)
        elif args.wallet_cmd == 'address':
            return cmd_wallet_address(args)
        else:
            wallet_parser.print_help()

    elif args.command == 'send':
        return cmd_send(args)

    elif args.command == 'receive':
        return cmd_receive(args)

    elif args.command == 'node':
        if args.node_cmd == 'status':
            return cmd_node_status(args)
        elif args.node_cmd == 'start':
            return cmd_node_start(args)
        elif args.node_cmd == 'stop':
            return cmd_node_stop(args)
        else:
            node_parser.print_help()

    elif args.command == 'mine':
        if args.mine_cmd == 'start':
            return cmd_mine_start(args)
        elif args.mine_cmd == 'stop':
            return cmd_mine_stop(args)
        elif args.mine_cmd == 'status':
            return cmd_mine_status(args)
        else:
            mine_parser.print_help()

    elif args.command == 'block':
        return cmd_block(args)

    elif args.command == 'tx':
        return cmd_tx(args)

    elif args.command == 'config':
        return cmd_config(args)

    else:
        parser.print_help()

    return 0


if __name__ == '__main__':
    sys.exit(main())
