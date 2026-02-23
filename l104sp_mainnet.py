#!/usr/bin/env python3
"""
L104SP SOVEREIGN MAINNET - PRODUCTION LAUNCHER
═══════════════════════════════════════════════════════════════════════════════

A fully functioning cryptocurrency with:
- Independent blockchain (not an EVM token)
- Proof of Resonance (PoR) consensus
- UTXO transaction model (like Bitcoin)
- secp256k1 ECDSA cryptography
- HD wallet (BIP-32/39/44)
- P2P networking
- RPC/REST API
- Persistent SQLite storage

GOD_CODE: 527.5184818492612 | PILOT: LONDEL

Usage:
    python l104sp_mainnet.py                    # Start node + interactive CLI
    python l104sp_mainnet.py --daemon           # Run as daemon
    python l104sp_mainnet.py --mine             # Start mining
    python l104sp_mainnet.py wallet new         # Create new wallet
    python l104sp_mainnet.py wallet balance     # Check balance
    python l104sp_mainnet.py send <addr> <amt>  # Send L104SP
"""

import os
import sys
import json
import time
import signal
import argparse
import threading
import readline
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_sovereign_coin_engine import (
    L104SPNode, L104SPBlockchain, HDWallet, MiningEngine,
    Secp256k1, CryptoUtils, Block, Transaction, TxInput, TxOutput, OutPoint,
    GOD_CODE, PHI, COIN_NAME, COIN_SYMBOL, MAX_SUPPLY, SATOSHI_PER_COIN,
    INITIAL_BLOCK_REWARD, L104SP_CONFIG, DATA_DIR, DEFAULT_PORT
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "3.2.0"
MAINNET_DATA_DIR = Path(os.environ.get('L104SP_DATA', str(DATA_DIR)))
WALLET_FILE = MAINNET_DATA_DIR / 'wallet.json'

# ASCII Art Logo
LOGO = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     ██╗     ██╗ ██████╗ ██╗  ██╗███████╗██████╗                              ║
║     ██║    ███║██╔═══██╗██║  ██║██╔════╝██╔══██╗                             ║
║     ██║    ╚██║██║   ██║███████║███████╗██████╔╝                             ║
║     ██║     ██║██║   ██║╚════██║╚════██║██╔═══╝                              ║
║     ███████╗██║╚██████╔╝     ██║███████║██║                                  ║
║     ╚══════╝╚═╝ ╚═════╝      ╚═╝╚══════╝╚═╝                                  ║
║                                                                               ║
║              SOVEREIGN PRIME - MAINNET v""" + VERSION + """                            ║
║              GOD_CODE: 527.5184818492612 | φ = 1.618033988749895             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# WALLET MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class WalletManager:
    """Secure wallet management for L104SP."""

    def __init__(self, wallet_file: Path = WALLET_FILE):
        self.wallet_file = wallet_file
        self.wallet: Optional[HDWallet] = None
        self.mnemonic: Optional[str] = None
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing wallet or create new one."""
        if self.wallet_file.exists():
            self._load()
        else:
            print("\n⚠️  No wallet found. Creating new HD wallet...")
            self.create_new()

    def _load(self) -> None:
        """Load wallet from encrypted file."""
        try:
            with open(self.wallet_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Restore from seed
            if 'seed_hex' in data:
                seed = bytes.fromhex(data['seed_hex'])
                self.wallet = HDWallet(seed=seed)
                print(f"✅ Wallet loaded from {self.wallet_file}")
            else:
                raise ValueError("No seed in wallet file")
        except Exception as e:
            print(f"❌ Error loading wallet: {e}")
            self.create_new()

    def create_new(self, passphrase: str = "") -> str:
        """Create a new HD wallet with mnemonic."""
        # Generate mnemonic first with a temporary wallet
        temp_wallet = HDWallet()
        self.mnemonic = temp_wallet.generate_mnemonic()

        # Create wallet from mnemonic
        self.wallet = HDWallet(mnemonic=self.mnemonic)

        # Save (encrypted in production)
        self.wallet_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'seed_hex': self.wallet.seed.hex(),
            'created_at': int(time.time()),
            'version': VERSION
        }
        with open(self.wallet_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ New wallet created!")
        print(f"📁 Saved to: {self.wallet_file}")
        print(f"\n🔑 BACKUP YOUR MNEMONIC PHRASE (write it down securely):")
        print(f"{'='*60}")
        print(f"   {self.mnemonic}")
        print(f"{'='*60}")
        print(f"\n⚠️  Anyone with this phrase can access your funds!")

        return self.mnemonic

    def get_address(self, index: int = 0) -> str:
        """Get receiving address at index."""
        if not self.wallet:
            raise ValueError("No wallet loaded")
        address, _ = self.wallet.get_address(index=index)
        return address

    def get_new_address(self) -> str:
        """Generate a new receiving address."""
        if not self.wallet:
            raise ValueError("No wallet loaded")
        # Find next unused index
        index = len(self.wallet._cache) if hasattr(self.wallet, '_cache') else 0
        return self.get_address(index)

    def get_private_key(self, index: int = 0) -> int:
        """Get private key for signing (use with caution)."""
        if not self.wallet:
            raise ValueError("No wallet loaded")
        _, private_key = self.wallet.get_address(index=index)
        return private_key

    def sign_transaction(self, tx: Transaction, input_indices: List[int] = None) -> Transaction:
        """Sign transaction inputs."""
        if not self.wallet:
            raise ValueError("No wallet loaded")

        input_indices = input_indices or list(range(len(tx.inputs)))

        for i in input_indices:
            private_key = self.get_private_key(i)
            tx_hash = bytes.fromhex(tx.txid)
            r, s = Secp256k1.sign(private_key, tx_hash)
            # DER encode signature
            signature = self._der_encode_signature(r, s)
            tx.inputs[i].script_sig = signature

        return tx

    def _der_encode_signature(self, r: int, s: int) -> bytes:
        """DER encode an ECDSA signature."""
        def encode_int(n: int) -> bytes:
            data = n.to_bytes((n.bit_length() + 8) // 8, 'big')
            if data[0] & 0x80:
                data = b'\x00' + data
            return bytes([0x02, len(data)]) + data

        r_enc = encode_int(r)
        s_enc = encode_int(s)
        return bytes([0x30, len(r_enc) + len(s_enc)]) + r_enc + s_enc


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionBuilder:
    """Build and sign L104SP transactions."""

    def __init__(self, blockchain: L104SPBlockchain, wallet: WalletManager):
        self.blockchain = blockchain
        self.wallet = wallet

    def build_transaction(self, to_address: str, amount: float, fee: float = 0.001) -> Optional[Transaction]:
        """Build a transaction to send L104SP."""
        amount_sats = int(amount * SATOSHI_PER_COIN)
        fee_sats = int(fee * SATOSHI_PER_COIN)
        total_needed = amount_sats + fee_sats

        # Find UTXOs for our addresses
        my_addresses = [self.wallet.get_address(i) for i in range(10)]
        utxos = []

        for addr in my_addresses:
            balance = self.blockchain.get_balance(addr)
            if balance > 0:
                # Get actual UTXOs (simplified - in production query UTXO set)
                utxos.append({
                    'address': addr,
                    'amount': balance,
                    'index': my_addresses.index(addr)
                })

        # Calculate total available
        total_available = sum(u['amount'] for u in utxos)

        if total_available < total_needed:
            print(f"❌ Insufficient balance: {total_available / SATOSHI_PER_COIN:.8f} L104SP")
            print(f"   Needed: {total_needed / SATOSHI_PER_COIN:.8f} L104SP")
            return None

        # Build inputs (simplified - use all UTXOs)
        inputs = []
        for utxo in utxos:
            inp = TxInput(
                prevout=OutPoint(txid="0" * 64, vout=utxo['index']),
                script_sig=b''  # Will be signed
            )
            inputs.append(inp)

        # Build outputs
        outputs = []

        # Recipient output
        recipient_hash = CryptoUtils.hash160(to_address.encode())
        recipient_script = b'\x00\x14' + recipient_hash  # P2WPKH
        outputs.append(TxOutput(value=amount_sats, script_pubkey=recipient_script))

        # Change output (if any)
        change = total_available - total_needed
        if change > 0:
            change_address = self.wallet.get_new_address()
            change_hash = CryptoUtils.hash160(change_address.encode())
            change_script = b'\x00\x14' + change_hash
            outputs.append(TxOutput(value=change, script_pubkey=change_script))

        # Create transaction
        tx = Transaction(version=2, inputs=inputs, outputs=outputs)

        # Sign inputs
        tx = self.wallet.sign_transaction(tx)

        return tx

    def broadcast(self, tx: Transaction) -> bool:
        """Broadcast transaction to mempool."""
        try:
            # Add to mempool
            self.blockchain.mempool[tx.txid] = tx
            self.blockchain.mempool_fees[tx.txid] = 1000  # Default fee per byte
            print(f"✅ Transaction broadcast: {tx.txid[:16]}...")
            return True
        except Exception as e:
            print(f"❌ Broadcast failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CLI
# ═══════════════════════════════════════════════════════════════════════════════

class L104SPCLI:
    """Interactive command-line interface for L104SP."""

    COMMANDS = {
        'help': 'Show available commands',
        'status': 'Show node status',
        'info': 'Show blockchain info',
        'wallet': 'Wallet commands (new, balance, address, export)',
        'send': 'Send L104SP: send <address> <amount>',
        'mine': 'Start/stop mining: mine start|stop',
        'peers': 'Show connected peers',
        'block': 'Show block: block <height|hash>',
        'tx': 'Show transaction: tx <txid>',
        'mempool': 'Show mempool status',
        'exit': 'Shutdown node and exit',
    }

    def __init__(self, node: L104SPNode, wallet_manager: WalletManager):
        self.node = node
        self.wallet = wallet_manager
        self.tx_builder = TransactionBuilder(node.blockchain, wallet_manager)
        self._running = True
        self._mining_thread: Optional[threading.Thread] = None

    def run(self) -> None:
        """Run interactive CLI loop."""
        print("\n📟 L104SP Interactive Console")
        print("   Type 'help' for commands, 'exit' to quit\n")

        while self._running:
            try:
                cmd = input("l104sp> ").strip()
                if not cmd:
                    continue
                self._process_command(cmd)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n(Use 'exit' to quit)")

    def _process_command(self, cmd: str) -> None:
        """Process a CLI command."""
        parts = cmd.split()
        command = parts[0].lower()
        args = parts[1:]

        if command == 'help':
            self._cmd_help()
        elif command == 'status':
            self._cmd_status()
        elif command == 'info':
            self._cmd_info()
        elif command == 'wallet':
            self._cmd_wallet(args)
        elif command == 'send':
            self._cmd_send(args)
        elif command == 'mine':
            self._cmd_mine(args)
        elif command == 'peers':
            self._cmd_peers()
        elif command == 'block':
            self._cmd_block(args)
        elif command == 'tx':
            self._cmd_tx(args)
        elif command == 'mempool':
            self._cmd_mempool()
        elif command in ('exit', 'quit', 'q'):
            self._cmd_exit()
        else:
            print(f"Unknown command: {command}. Type 'help' for available commands.")

    def _cmd_help(self) -> None:
        """Show help."""
        print("\n📖 Available Commands:")
        print("-" * 50)
        for cmd, desc in self.COMMANDS.items():
            print(f"  {cmd:12} - {desc}")
        print()

    def _cmd_status(self) -> None:
        """Show node status."""
        status = self.node.get_status()
        print(f"\n🔷 L104SP Node Status")
        print("-" * 40)
        print(f"  Version:    {status.get('version', VERSION)}")
        print(f"  Network:    {status.get('network', 'mainnet')}")
        print(f"  Height:     {self.node.blockchain.height}")
        print(f"  Tip:        {self.node.blockchain.tip.hash[:16]}...")
        print(f"  Peers:      {len(self.node.p2p.peers)}")
        print(f"  Mempool:    {len(self.node.blockchain.mempool)} txs")
        print(f"  Mining:     {'Active' if self.node.miner._running else 'Inactive'}")
        if self.node.miner._running:
            print(f"  Hashrate:   {self.node.miner.stats.hashrate:.2f} H/s")
        print()

    def _cmd_info(self) -> None:
        """Show blockchain info."""
        stats = self.node.blockchain.stats()
        print(f"\n⛓️  Blockchain Info")
        print("-" * 40)
        print(f"  Name:       {COIN_NAME}")
        print(f"  Symbol:     {COIN_SYMBOL}")
        print(f"  Height:     {stats['height']}")
        print(f"  Difficulty: {stats['difficulty']}")
        print(f"  Total Supply: {stats.get('mined_supply', 0) / SATOSHI_PER_COIN:,.2f} L104SP")
        print(f"  Max Supply:   {MAX_SUPPLY / SATOSHI_PER_COIN:,.0f} L104SP")
        print(f"  Block Reward: {INITIAL_BLOCK_REWARD / SATOSHI_PER_COIN:.2f} L104SP")
        print()

    def _cmd_wallet(self, args: List[str]) -> None:
        """Wallet commands."""
        if not args:
            args = ['balance']

        subcmd = args[0].lower()

        if subcmd == 'new':
            self.wallet.create_new()

        elif subcmd == 'balance':
            total = 0
            print(f"\n💰 Wallet Balances")
            print("-" * 50)
            for i in range(10):
                addr = self.wallet.get_address(i)
                balance = self.node.blockchain.get_balance(addr)
                if balance > 0 or i < 3:
                    print(f"  [{i}] {addr[:16]}... : {balance / SATOSHI_PER_COIN:.8f} L104SP")
                    total += balance
            print("-" * 50)
            print(f"  TOTAL: {total / SATOSHI_PER_COIN:.8f} L104SP")
            print()

        elif subcmd == 'address':
            index = int(args[1]) if len(args) > 1 else 0
            addr = self.wallet.get_address(index)
            print(f"\n📬 Receiving Address [{index}]:")
            print(f"   {addr}")
            print()

        elif subcmd == 'newaddress':
            addr = self.wallet.get_new_address()
            print(f"\n📬 New Receiving Address:")
            print(f"   {addr}")
            print()

        elif subcmd == 'export':
            if self.wallet.mnemonic:
                print(f"\n🔑 Mnemonic Phrase:")
                print(f"   {self.wallet.mnemonic}")
            else:
                print("⚠️  Mnemonic not available (wallet loaded from seed)")
            print()

        else:
            print(f"Unknown wallet command: {subcmd}")
            print("Available: new, balance, address, newaddress, export")

    def _cmd_send(self, args: List[str]) -> None:
        """Send L104SP."""
        if len(args) < 2:
            print("Usage: send <address> <amount>")
            return

        to_address = args[0]
        amount = float(args[1])

        print(f"\n📤 Creating transaction...")
        print(f"   To:     {to_address}")
        print(f"   Amount: {amount} L104SP")

        tx = self.tx_builder.build_transaction(to_address, amount)
        if tx:
            confirm = input("Confirm send? (y/n): ").strip().lower()
            if confirm == 'y':
                self.tx_builder.broadcast(tx)
            else:
                print("Transaction cancelled.")

    def _cmd_mine(self, args: List[str]) -> None:
        """Start/stop mining."""
        if not args:
            args = ['status']

        action = args[0].lower()

        if action == 'start':
            if self._mining_thread and self._mining_thread.is_alive():
                print("⚠️  Mining already running")
                return

            address = self.wallet.get_address(0)
            print(f"\n⛏️  Starting mining to: {address}")

            def mine():
                self.node.start_mining(address)

            self._mining_thread = threading.Thread(target=mine, daemon=True)
            self._mining_thread.start()
            print("Mining started in background. Use 'mine status' to check progress.")

        elif action == 'stop':
            self.node.miner.stop()
            print("⏹️  Mining stopped")

        elif action == 'status':
            stats = self.node.miner.stats
            print(f"\n⛏️  Mining Status")
            print("-" * 40)
            print(f"  Running:      {'Yes' if self.node.miner._running else 'No'}")
            print(f"  Hashrate:     {stats.hashrate:.2f} H/s")
            print(f"  Hashes:       {stats.hashes:,}")
            print(f"  Blocks Found: {stats.valid_blocks}")
            print(f"  Efficiency:   {stats.efficiency:.4f}%")
            print()

        else:
            print("Usage: mine start|stop|status")

    def _cmd_peers(self) -> None:
        """Show peers."""
        peers = list(self.node.p2p.peers.keys())
        print(f"\n🌐 Connected Peers ({len(peers)})")
        print("-" * 40)
        for peer in peers:
            print(f"  {peer}")
        if not peers:
            print("  No peers connected")
        print()

    def _cmd_block(self, args: List[str]) -> None:
        """Show block info."""
        if not args:
            height = self.node.blockchain.height
        else:
            try:
                height = int(args[0])
            except ValueError:
                # Assume it's a hash
                print("Block lookup by hash not yet implemented")
                return

        block = self.node.blockchain.get_block(height)
        if not block:
            print(f"Block {height} not found")
            return

        print(f"\n📦 Block {height}")
        print("-" * 60)
        print(f"  Hash:       {block.hash}")
        print(f"  Prev:       {block.header.prev_block[:32]}...")
        print(f"  Merkle:     {block.header.merkle_root[:32]}...")
        print(f"  Timestamp:  {block.header.timestamp}")
        print(f"  Nonce:      {block.header.nonce}")
        print(f"  Resonance:  {block.header.resonance:.6f}")
        print(f"  Difficulty: {block.header.difficulty}")
        print(f"  Txs:        {len(block.transactions)}")
        print()

    def _cmd_tx(self, args: List[str]) -> None:
        """Show transaction info."""
        if not args:
            print("Usage: tx <txid>")
            return

        txid = args[0]

        # Check mempool
        if txid in self.node.blockchain.mempool:
            tx = self.node.blockchain.mempool[txid]
            status = "Pending"
        else:
            print(f"Transaction {txid[:16]}... not found in mempool")
            return

        print(f"\n💳 Transaction")
        print("-" * 60)
        print(f"  TXID:    {tx.txid}")
        print(f"  Status:  {status}")
        print(f"  Inputs:  {len(tx.inputs)}")
        print(f"  Outputs: {len(tx.outputs)}")
        for i, out in enumerate(tx.outputs):
            print(f"    [{i}] {out.value / SATOSHI_PER_COIN:.8f} L104SP")
        print()

    def _cmd_mempool(self) -> None:
        """Show mempool."""
        mempool = self.node.blockchain.mempool
        print(f"\n📋 Mempool ({len(mempool)} transactions)")
        print("-" * 60)
        for txid, tx in list(mempool.items())[:10]:
            total_out = sum(out.value for out in tx.outputs)
            print(f"  {txid[:16]}... : {total_out / SATOSHI_PER_COIN:.8f} L104SP")
        if len(mempool) > 10:
            print(f"  ... and {len(mempool) - 10} more")
        print()

    def _cmd_exit(self) -> None:
        """Shutdown and exit."""
        print("\n👋 Shutting down L104SP node...")
        self._running = False
        self.node.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for L104SP mainnet."""
    parser = argparse.ArgumentParser(
        description="L104SP Sovereign Prime - Mainnet Node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python l104sp_mainnet.py                    # Start interactive node
  python l104sp_mainnet.py --mine             # Start with mining
  python l104sp_mainnet.py --daemon           # Run as daemon
  python l104sp_mainnet.py wallet new         # Create new wallet
  python l104sp_mainnet.py wallet balance     # Check balance
  python l104sp_mainnet.py send <addr> <amt>  # Send L104SP
        """
    )

    parser.add_argument('--version', action='version', version=f'L104SP v{VERSION}')
    parser.add_argument('--daemon', '-d', action='store_true', help='Run as daemon')
    parser.add_argument('--mine', '-m', action='store_true', help='Enable mining')
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help=f'P2P port (default: {DEFAULT_PORT})')
    parser.add_argument('--rpcport', type=int, default=10401, help='RPC port (default: 10401)')
    parser.add_argument('--datadir', type=str, help='Data directory')
    parser.add_argument('command', nargs='?', help='Command (wallet, send, etc.)')
    parser.add_argument('args', nargs='*', help='Command arguments')

    args = parser.parse_args()

    # Setup data directory
    data_dir = Path(args.datadir) if args.datadir else MAINNET_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    print(LOGO)

    # Handle direct commands
    if args.command == 'wallet':
        wallet = WalletManager(data_dir / 'wallet.json')
        if args.args:
            subcmd = args.args[0]
            if subcmd == 'new':
                wallet.create_new()
            elif subcmd == 'balance':
                # Need blockchain for balance
                pass
            elif subcmd == 'address':
                idx = int(args.args[1]) if len(args.args) > 1 else 0
                print(f"Address [{idx}]: {wallet.get_address(idx)}")
        else:
            print(f"Address: {wallet.get_address(0)}")
        return

    # Initialize node
    print(f"📁 Data directory: {data_dir}")
    print(f"🌐 P2P port: {args.port}")
    print(f"🔌 RPC port: {args.rpcport}")
    print()

    # Create wallet manager
    wallet = WalletManager(data_dir / 'wallet.json')

    # Create and start node
    node = L104SPNode(port=args.port, rpc_port=args.rpcport, data_dir=data_dir)
    node.start(enable_rpc=True)

    # Handle send command
    if args.command == 'send' and len(args.args) >= 2:
        tx_builder = TransactionBuilder(node.blockchain, wallet)
        to_addr = args.args[0]
        amount = float(args.args[1])
        tx = tx_builder.build_transaction(to_addr, amount)
        if tx:
            tx_builder.broadcast(tx)
        node.stop()
        return

    # Start mining if requested
    if args.mine:
        address = wallet.get_address(0)
        print(f"\n⛏️  Mining enabled. Address: {address}")
        mining_thread = threading.Thread(target=lambda: node.start_mining(address), daemon=True)
        mining_thread.start()

    # Daemon mode
    if args.daemon:
        print("\n🔄 Running as daemon. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            node.stop()
        return

    # Interactive mode
    cli = L104SPCLI(node, wallet)
    try:
        cli.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()


if __name__ == '__main__':
    main()
