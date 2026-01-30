#!/usr/bin/env python3
"""
L104SP SEED NODE CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════

Seed nodes are the bootstrap infrastructure for the L104SP network.
These nodes are the first connection points for new nodes joining the network.

To run a seed node:
    python l104sp_seed_node.py

To add your node as a seed:
    1. Deploy to a public IP/domain
    2. Open port 10400 (P2P) and 10401 (RPC)
    3. Submit a PR adding your node to SEED_NODES
"""

import os
import sys
import json
import time
import socket
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_sovereign_coin_engine import (
    L104SPNode, L104SPBlockchain, P2PNode, PeerDiscovery,
    DEFAULT_PORT, MAINNET_MAGIC, DATA_DIR
)

# ═══════════════════════════════════════════════════════════════════════════════
# OFFICIAL L104SP SEED NODES
# ═══════════════════════════════════════════════════════════════════════════════

SEED_NODES: List[Tuple[str, int]] = [
    # Primary seed nodes (will be active after initial deployment)
    # Format: (host, port)
    
    # Cloud Run deployed nodes
    ('l104-server-xxxxxxxxxx-uc.a.run.app', 443),  # GCP Cloud Run (HTTPS)
    
    # Fly.io deployed nodes
    ('l104sp-mainnet.fly.dev', 10400),
    
    # Local development
    ('localhost', 10400),
    ('127.0.0.1', 10400),
    
    # Community seed nodes (add yours here!)
    # ('seed1.l104sp.io', 10400),
    # ('seed2.l104sp.io', 10400),
]

# DNS-based seed discovery
DNS_SEEDS: List[str] = [
    'seed.l104sp.io',
    'seeds.l104sp.network',
    'dnsseed.l104.io',
]

# ═══════════════════════════════════════════════════════════════════════════════
# SEED NODE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class SeedNodeManager:
    """Manages seed node discovery and health monitoring."""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.known_peers: Dict[str, Dict[str, Any]] = {}
        self.healthy_seeds: List[Tuple[str, int]] = []
        self._lock = threading.Lock()
        self._load_peers()
    
    def _load_peers(self) -> None:
        """Load known peers from disk."""
        peers_file = self.data_dir / 'peers.json'
        if peers_file.exists():
            try:
                with open(peers_file, 'r') as f:
                    self.known_peers = json.load(f)
                print(f"[SEED] Loaded {len(self.known_peers)} known peers")
            except Exception as e:
                print(f"[SEED] Error loading peers: {e}")
    
    def _save_peers(self) -> None:
        """Save known peers to disk."""
        peers_file = self.data_dir / 'peers.json'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(peers_file, 'w') as f:
            json.dump(self.known_peers, f, indent=2)
    
    def discover_seeds(self) -> List[Tuple[str, int]]:
        """Discover seed nodes via DNS and hardcoded list."""
        discovered = list(SEED_NODES)
        
        # DNS-based discovery
        for dns_seed in DNS_SEEDS:
            try:
                ips = socket.gethostbyname_ex(dns_seed)[2]
                for ip in ips:
                    discovered.append((ip, DEFAULT_PORT))
                    print(f"[SEED] DNS discovery: {dns_seed} -> {ip}")
            except Exception:
                pass
        
        return list(set(discovered))
    
    def check_seed_health(self, host: str, port: int, timeout: float = 5.0) -> bool:
        """Check if a seed node is reachable."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def get_healthy_seeds(self, count: int = 8) -> List[Tuple[str, int]]:
        """Get list of healthy seed nodes."""
        all_seeds = self.discover_seeds()
        healthy = []
        
        print(f"[SEED] Checking {len(all_seeds)} seed nodes...")
        
        for host, port in all_seeds:
            if len(healthy) >= count:
                break
            if self.check_seed_health(host, port):
                healthy.append((host, port))
                print(f"[SEED] ✅ {host}:{port} - ONLINE")
            else:
                print(f"[SEED] ❌ {host}:{port} - OFFLINE")
        
        with self._lock:
            self.healthy_seeds = healthy
        
        return healthy
    
    def add_peer(self, host: str, port: int, metadata: Dict[str, Any] = None) -> None:
        """Add a peer to known peers."""
        peer_id = f"{host}:{port}"
        with self._lock:
            self.known_peers[peer_id] = {
                'host': host,
                'port': port,
                'first_seen': int(time.time()),
                'last_seen': int(time.time()),
                'metadata': metadata or {}
            }
            self._save_peers()
    
    def get_peers(self, count: int = 20) -> List[Tuple[str, int]]:
        """Get known peers, prioritizing recently seen."""
        with self._lock:
            sorted_peers = sorted(
                self.known_peers.values(),
                key=lambda p: p.get('last_seen', 0),
                reverse=True
            )
            return [(p['host'], p['port']) for p in sorted_peers[:count]]


# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP PROCESS
# ═══════════════════════════════════════════════════════════════════════════════

class NetworkBootstrap:
    """Handles initial network bootstrap for new nodes."""
    
    def __init__(self, node: L104SPNode):
        self.node = node
        self.seed_manager = SeedNodeManager(node.data_dir)
        self._syncing = False
    
    def bootstrap(self) -> bool:
        """Bootstrap the node by connecting to seeds and syncing."""
        print("\n" + "=" * 60)
        print("    L104SP NETWORK BOOTSTRAP")
        print("=" * 60 + "\n")
        
        # Step 1: Discover healthy seeds
        print("[BOOTSTRAP] Step 1: Discovering seed nodes...")
        seeds = self.seed_manager.get_healthy_seeds(count=8)
        
        if not seeds:
            print("[BOOTSTRAP] ⚠️  No seed nodes reachable. Running in solo mode.")
            return False
        
        print(f"[BOOTSTRAP] Found {len(seeds)} healthy seed nodes\n")
        
        # Step 2: Connect to seeds
        print("[BOOTSTRAP] Step 2: Connecting to network...")
        connected = 0
        for host, port in seeds:
            try:
                self._connect_to_peer(host, port)
                connected += 1
                print(f"[BOOTSTRAP] ✅ Connected to {host}:{port}")
            except Exception as e:
                print(f"[BOOTSTRAP] ❌ Failed to connect to {host}:{port}: {e}")
        
        if connected == 0:
            print("[BOOTSTRAP] ⚠️  Could not connect to any peers")
            return False
        
        print(f"[BOOTSTRAP] Connected to {connected} peers\n")
        
        # Step 3: Sync blockchain
        print("[BOOTSTRAP] Step 3: Syncing blockchain...")
        self._sync_chain()
        
        # Step 4: Exchange peer lists
        print("\n[BOOTSTRAP] Step 4: Discovering more peers...")
        self._exchange_peer_lists()
        
        print("\n" + "=" * 60)
        print("    BOOTSTRAP COMPLETE!")
        print(f"    Height: {self.node.blockchain.height}")
        print(f"    Peers:  {len(self.node.p2p.peers)}")
        print("=" * 60 + "\n")
        
        return True
    
    def _connect_to_peer(self, host: str, port: int) -> None:
        """Connect to a peer."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect((host, port))
        
        # Send version message
        version_msg = {
            'type': 'version',
            'version': '3.2.0',
            'height': self.node.blockchain.height,
            'timestamp': int(time.time())
        }
        sock.send(MAINNET_MAGIC + json.dumps(version_msg).encode())
        
        # Add to peers
        peer_id = f"{host}:{port}"
        self.node.p2p.peers[peer_id] = (host, port)
        self.seed_manager.add_peer(host, port)
        
        sock.close()
    
    def _sync_chain(self) -> None:
        """Request and sync blockchain from peers."""
        self._syncing = True
        
        for peer_id, addr in list(self.node.p2p.peers.items()):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(30.0)
                sock.connect(addr)
                
                # Request blocks
                getblocks_msg = {
                    'type': 'getblocks',
                    'start_height': self.node.blockchain.height,
                    'limit': 100
                }
                sock.send(MAINNET_MAGIC + json.dumps(getblocks_msg).encode())
                
                # Receive blocks
                data = sock.recv(65536)
                if data and data[:4] == MAINNET_MAGIC:
                    msg = json.loads(data[4:].decode())
                    if msg.get('type') == 'blocks':
                        blocks_received = len(msg.get('blocks', []))
                        print(f"[SYNC] Received {blocks_received} blocks from {peer_id}")
                
                sock.close()
            except Exception as e:
                print(f"[SYNC] Error syncing from {peer_id}: {e}")
        
        self._syncing = False
    
    def _exchange_peer_lists(self) -> None:
        """Exchange peer lists with connected peers."""
        for peer_id, addr in list(self.node.p2p.peers.items()):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect(addr)
                
                # Request peers
                getaddr_msg = {'type': 'getaddr'}
                sock.send(MAINNET_MAGIC + json.dumps(getaddr_msg).encode())
                
                # Receive peer list
                data = sock.recv(4096)
                if data and data[:4] == MAINNET_MAGIC:
                    msg = json.loads(data[4:].decode())
                    if msg.get('type') == 'addr':
                        for peer in msg.get('peers', []):
                            self.seed_manager.add_peer(peer['host'], peer['port'])
                        print(f"[PEER] Received {len(msg.get('peers', []))} peers from {peer_id}")
                
                sock.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# SEED NODE SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class SeedNodeServer:
    """Run as a dedicated seed node for the L104SP network."""
    
    def __init__(self, port: int = DEFAULT_PORT, rpc_port: int = 10401):
        self.port = port
        self.rpc_port = rpc_port
        self.node: Optional[L104SPNode] = None
        self.seed_manager = SeedNodeManager()
        self._running = False
    
    def start(self) -> None:
        """Start the seed node."""
        print("\n" + "=" * 60)
        print("    L104SP SEED NODE")
        print("=" * 60)
        print(f"\n🌱 Starting seed node on port {self.port}...")
        
        # Initialize node
        self.node = L104SPNode(port=self.port, rpc_port=self.rpc_port)
        self.node.start(enable_rpc=True)
        
        # Bootstrap to other seeds first
        bootstrap = NetworkBootstrap(self.node)
        bootstrap.bootstrap()
        
        self._running = True
        
        # Monitor loop
        print("\n🌐 Seed node running. Press Ctrl+C to stop.")
        print(f"   P2P: 0.0.0.0:{self.port}")
        print(f"   RPC: http://127.0.0.1:{self.rpc_port}/status")
        
        try:
            while self._running:
                time.sleep(60)
                self._log_status()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def _log_status(self) -> None:
        """Log seed node status."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        peers = len(self.node.p2p.peers)
        height = self.node.blockchain.height
        print(f"[{now}] Peers: {peers} | Height: {height}")
    
    def stop(self) -> None:
        """Stop the seed node."""
        print("\n🛑 Stopping seed node...")
        self._running = False
        if self.node:
            self.node.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run a L104SP seed node."""
    import argparse
    
    parser = argparse.ArgumentParser(description="L104SP Seed Node")
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help='P2P port')
    parser.add_argument('--rpc', type=int, default=10401, help='RPC port')
    parser.add_argument('--check', action='store_true', help='Only check seed health')
    
    args = parser.parse_args()
    
    if args.check:
        # Just check seed health
        manager = SeedNodeManager()
        healthy = manager.get_healthy_seeds()
        print(f"\n✅ {len(healthy)} healthy seed nodes")
        return
    
    # Run seed node
    server = SeedNodeServer(port=args.port, rpc_port=args.rpc)
    server.start()


if __name__ == '__main__':
    main()
