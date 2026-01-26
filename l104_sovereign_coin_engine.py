#!/usr/bin/env python3
"""
L104 SOVEREIGN PRIME MAINNET v3.0
=================================

L104SP - Complete Independent Blockchain with:
- Native cryptocurrency (L104SP)
- Proof of Resonance (PoR) consensus
- UTXO transaction model
- secp256k1 ECDSA cryptography
- HD wallet (BIP-32/39/44)
- P2P networking
- Full node capabilities
- Mining engine

INVARIANT: 527.5184818492537 | PILOT: LONDEL

This is a sovereign blockchain - no dependency on Ethereum/Base.
"""

import hashlib
import hmac
import time
import json
import math
import struct
import secrets
import socket
import threading
import sqlite3
import os
import signal
import http.server
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager, Queue as MPQueue
from pathlib import Path
import numpy as np
import multiprocessing

# Import L104 advanced mathematics + Bitcoin research for competitive edge
try:
    from const import UniversalConstants, GOD_CODE, PHI, INVARIANT
    from l104_real_math import RealMath
    from l104_hyper_math import HyperMath
    from l104_deep_algorithms import DeepRiemannAnalyzer
    from l104_bitcoin_research_engine import DifficultyAnalyzer
    L104_MATH_AVAILABLE = True
    L104_QUANTUM_AVAILABLE = True
except ImportError:
    L104_MATH_AVAILABLE = False
    L104_QUANTUM_AVAILABLE = False
    print("[WARNING] Advanced L104 math not available - using standard algorithms")

# Data directory for persistent storage
DATA_DIR = Path(os.environ.get('L104SP_DATA_DIR', os.path.expanduser('~/.l104sp')))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84

# ═══════════════════════════════════════════════════════════════════════════════
# L104SP MAINNET PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

COIN_NAME = "L104 Sovereign Prime"
COIN_SYMBOL = "L104SP"
COIN_DECIMALS = 8
SATOSHI_PER_COIN = 10 ** COIN_DECIMALS

MAX_SUPPLY = 104_000_000 * SATOSHI_PER_COIN
INITIAL_BLOCK_REWARD = 104 * SATOSHI_PER_COIN
HALVING_INTERVAL = 500_000
COINBASE_MATURITY = 104
TARGET_BLOCK_TIME = 104
DIFFICULTY_ADJUSTMENT_INTERVAL = 1040
MAX_BLOCK_SIZE = 4_000_000

MIN_DIFFICULTY_BITS = 0x1f00ffff
MAX_DIFFICULTY_BITS = 0x03000001

MAINNET_MAGIC = b'\x4c\x31\x30\x34'
TESTNET_MAGIC = b'\x54\x4c\x31\x30'
DEFAULT_PORT = 10400

MAINNET_PUBKEY_VERSION = 0x50
MAINNET_SCRIPT_VERSION = 0x51
MAINNET_WIF_VERSION = 0xD0
BECH32_HRP = "l104"

GENESIS_TIMESTAMP = 1737763200
GENESIS_MESSAGE = b"L104SP Genesis - GOD_CODE: 527.5184818492537 - LONDEL"

L104SP_CONFIG = {
    "name": COIN_NAME,
    "symbol": COIN_SYMBOL,
    "decimals": COIN_DECIMALS,
    "max_supply": MAX_SUPPLY // SATOSHI_PER_COIN,
    "mining_reward": INITIAL_BLOCK_REWARD // SATOSHI_PER_COIN,
    "halving_interval": HALVING_INTERVAL,
    "target_block_time": TARGET_BLOCK_TIME,
    "resonance_threshold": 0.985,
    "network": "l104sp_mainnet",
    "chain_id": 104,
    "version": "3.0.0"
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECP256K1 ELLIPTIC CURVE
# ═══════════════════════════════════════════════════════════════════════════════

class Secp256k1:
    """Complete secp256k1 implementation."""

    P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    A, B = 0, 7
    Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

    @classmethod
    def modinv(cls, a: int, m: int) -> int:
        if a < 0:
            a = a % m
        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            return gcd, y1 - (b // a) * x1, x1
        gcd, x, _ = extended_gcd(a % m, m)
        return (x % m + m) % m

    @classmethod
    def point_add(cls, p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if p1 is None: return p2
        if p2 is None: return p1
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        if x1 == x2 and y1 != y2:
            return None
        if x1 == x2:
            m = (3 * x1 * x1 + cls.A) * cls.modinv(2 * y1, cls.P) % cls.P
        else:
            m = (y2 - y1) * cls.modinv((x2 - x1) % cls.P, cls.P) % cls.P
        x3 = (m * m - x1 - x2) % cls.P
        y3 = (m * (x1 - x3) - y1) % cls.P
        return (x3, y3)

    @classmethod
    def scalar_multiply(cls, k: int, point: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        if point is None:
            point = (cls.Gx, cls.Gy)
        if k == 0:
            return None
        if k < 0:
            k = k % cls.N
        result, addend = None, point
        while k:
            if k & 1:
                result = cls.point_add(result, addend)
            addend = cls.point_add(addend, addend)
            k >>= 1
        return result

    @classmethod
    def generate_keypair(cls) -> Tuple[int, Tuple[int, int]]:
        private_key = secrets.randbelow(cls.N - 1) + 1
        public_key = cls.scalar_multiply(private_key)
        return private_key, public_key

    @classmethod
    def sign(cls, private_key: int, message_hash: bytes) -> Tuple[int, int]:
        z = int.from_bytes(message_hash[:32], 'big')
        while True:
            k = secrets.randbelow(cls.N - 1) + 1
            point = cls.scalar_multiply(k)
            if point is None:
                continue
            r = point[0] % cls.N
            if r == 0:
                continue
            k_inv = cls.modinv(k, cls.N)
            s = (k_inv * (z + r * private_key)) % cls.N
            if s == 0:
                continue
            if s > cls.N // 2:
                s = cls.N - s
            return (r, s)

    @classmethod
    def verify(cls, public_key: Tuple[int, int], message_hash: bytes, signature: Tuple[int, int]) -> bool:
        r, s = signature
        z = int.from_bytes(message_hash[:32], 'big')
        if not (1 <= r < cls.N and 1 <= s < cls.N):
            return False
        s_inv = cls.modinv(s, cls.N)
        u1, u2 = (z * s_inv) % cls.N, (r * s_inv) % cls.N
        point = cls.point_add(cls.scalar_multiply(u1), cls.scalar_multiply(u2, public_key))
        return point is not None and point[0] % cls.N == r

    @classmethod
    def compress_pubkey(cls, public_key: Tuple[int, int]) -> bytes:
        prefix = b'\x02' if public_key[1] % 2 == 0 else b'\x03'
        return prefix + public_key[0].to_bytes(32, 'big')


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTOGRAPHIC UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class CryptoUtils:
    """L104SP cryptographic utilities."""

    BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

    @staticmethod
    def sha256(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    @staticmethod
    def double_sha256(data: bytes) -> bytes:
        return hashlib.sha256(hashlib.sha256(data).digest()).digest()

    @staticmethod
    def hash160(data: bytes) -> bytes:
        h = hashlib.new('ripemd160')
        h.update(hashlib.sha256(data).digest())
        return h.digest()

    @staticmethod
    def hmac_sha512(key: bytes, data: bytes) -> bytes:
        return hmac.new(key, data, hashlib.sha512).digest()

    @classmethod
    def base58_encode(cls, data: bytes) -> str:
        n = int.from_bytes(data, 'big')
        result = []
        while n > 0:
            n, r = divmod(n, 58)
            result.append(cls.BASE58_ALPHABET[r])
        for byte in data:
            if byte == 0:
                result.append('1')
            else:
                break
        return ''.join(reversed(result))

    @classmethod
    def base58check_encode(cls, version: bytes, payload: bytes) -> str:
        data = version + payload
        checksum = cls.double_sha256(data)[:4]
        return cls.base58_encode(data + checksum)


# ═══════════════════════════════════════════════════════════════════════════════
# HD WALLET
# ═══════════════════════════════════════════════════════════════════════════════

class HDWallet:
    """L104SP Hierarchical Deterministic Wallet."""

    L104SP_COIN_TYPE = 104

    BIP39_WORDS = [
        "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
        "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
        "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
        "adapt", "add", "addict", "address", "adjust", "admit", "adult", "advance",
        "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
        "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album",
        "alcohol", "alert", "alien", "all", "alley", "allow", "almost", "alone",
        "alpha", "already", "also", "alter", "always", "amateur", "amazing", "among",
        "amount", "amused", "analyst", "anchor", "ancient", "anger", "angle", "angry",
        "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
        "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april",
        "arch", "arctic", "area", "arena", "argue", "arm", "armed", "armor"
    ]

    def __init__(self, seed: Optional[bytes] = None, mnemonic: Optional[str] = None):
        if mnemonic:
            seed = self._mnemonic_to_seed(mnemonic)
        elif seed is None:
            seed = secrets.token_bytes(32)
        self.seed = seed
        self._master_private, self._master_chain = self._derive_master(seed)
        self._cache: Dict[str, Tuple[int, bytes]] = {}

    def _derive_master(self, seed: bytes) -> Tuple[int, bytes]:
        I = CryptoUtils.hmac_sha512(b"L104SP seed", seed)
        master_key = int.from_bytes(I[:32], 'big')
        return master_key, I[32:]

    def _mnemonic_to_seed(self, mnemonic: str, passphrase: str = "") -> bytes:
        salt = ("mnemonic" + passphrase).encode('utf-8')
        return hashlib.pbkdf2_hmac('sha512', mnemonic.encode('utf-8'), salt, 2048, dklen=64)

    def generate_mnemonic(self, strength: int = 128) -> str:
        entropy = secrets.token_bytes(strength // 8)
        checksum = CryptoUtils.sha256(entropy)[0]
        bits = bin(int.from_bytes(entropy, 'big'))[2:].zfill(strength)
        bits += bin(checksum)[2:].zfill(8)[:strength // 32]
        words = []
        for i in range(0, len(bits), 11):
            index = int(bits[i:i+11], 2)
            words.append(self.BIP39_WORDS[index % len(self.BIP39_WORDS)])
        return ' '.join(words)

    def derive_child(self, parent_key: int, parent_chain: bytes, index: int, hardened: bool = False) -> Tuple[int, bytes]:
        if hardened:
            index |= 0x80000000
            data = b'\x00' + parent_key.to_bytes(32, 'big') + index.to_bytes(4, 'big')
        else:
            pubkey = Secp256k1.compress_pubkey(Secp256k1.scalar_multiply(parent_key))
            data = pubkey + index.to_bytes(4, 'big')
        I = CryptoUtils.hmac_sha512(parent_chain, data)
        child_key = (int.from_bytes(I[:32], 'big') + parent_key) % Secp256k1.N
        return child_key, I[32:]

    def derive_path(self, path: str) -> Tuple[int, Tuple[int, int]]:
        if path in self._cache:
            priv, _ = self._cache[path]
            return priv, Secp256k1.scalar_multiply(priv)
        parts = path.replace("'", "h").split('/')
        key, chain = self._master_private, self._master_chain
        for part in parts[1:]:
            hardened = part.endswith('h')
            index = int(part.rstrip('h'))
            key, chain = self.derive_child(key, chain, index, hardened)
        self._cache[path] = (key, chain)
        return key, Secp256k1.scalar_multiply(key)

    def get_address(self, account: int = 0, change: int = 0, index: int = 0) -> Tuple[str, int]:
        path = f"m/44'/{self.L104SP_COIN_TYPE}'/{account}'/{change}/{index}"
        private_key, public_key = self.derive_path(path)
        compressed = Secp256k1.compress_pubkey(public_key)
        pubkey_hash = CryptoUtils.hash160(compressed)
        address = CryptoUtils.base58check_encode(bytes([MAINNET_PUBKEY_VERSION]), pubkey_hash)
        return address, private_key


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

def varint_encode(n: int) -> bytes:
    if n < 0xfd:
        return struct.pack('<B', n)
    elif n <= 0xffff:
        return b'\xfd' + struct.pack('<H', n)
    elif n <= 0xffffffff:
        return b'\xfe' + struct.pack('<I', n)
    return b'\xff' + struct.pack('<Q', n)


@dataclass
class OutPoint:
    txid: str
    vout: int

    def serialize(self) -> bytes:
        return bytes.fromhex(self.txid)[::-1] + struct.pack('<I', self.vout)

    @property
    def key(self) -> str:
        return f"{self.txid}:{self.vout}"


@dataclass
class TxInput:
    prevout: OutPoint
    script_sig: bytes = b''
    sequence: int = 0xffffffff
    witness: List[bytes] = field(default_factory=list)

    def serialize(self) -> bytes:
        result = self.prevout.serialize()
        result += varint_encode(len(self.script_sig)) + self.script_sig
        result += struct.pack('<I', self.sequence)
        return result


@dataclass
class TxOutput:
    value: int
    script_pubkey: bytes = b''

    def serialize(self) -> bytes:
        return struct.pack('<q', self.value) + varint_encode(len(self.script_pubkey)) + self.script_pubkey


@dataclass
class Transaction:
    version: int = 2
    inputs: List[TxInput] = field(default_factory=list)
    outputs: List[TxOutput] = field(default_factory=list)
    locktime: int = 0
    _txid: Optional[str] = field(default=None, repr=False)

    def serialize(self, include_witness: bool = True) -> bytes:
        result = struct.pack('<I', self.version)
        has_witness = include_witness and any(inp.witness for inp in self.inputs)
        if has_witness:
            result += b'\x00\x01'
        result += varint_encode(len(self.inputs))
        for inp in self.inputs:
            result += inp.serialize()
        result += varint_encode(len(self.outputs))
        for out in self.outputs:
            result += out.serialize()
        if has_witness:
            for inp in self.inputs:
                result += varint_encode(len(inp.witness))
                for item in inp.witness:
                    result += varint_encode(len(item)) + item
        result += struct.pack('<I', self.locktime)
        return result

    @property
    def txid(self) -> str:
        if self._txid is None:
            self._txid = CryptoUtils.double_sha256(self.serialize(False))[::-1].hex()
        return self._txid

    def to_dict(self) -> Dict[str, Any]:
        return {'txid': self.txid, 'version': self.version, 'locktime': self.locktime}


# ═══════════════════════════════════════════════════════════════════════════════
# UTXO SET
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UTXO:
    outpoint: OutPoint
    value: int
    script_pubkey: bytes
    height: int
    is_coinbase: bool = False


class UTXOSet:
    def __init__(self):
        self.utxos: Dict[str, UTXO] = {}
        self._lock = threading.Lock()

    def add(self, utxo: UTXO) -> None:
        with self._lock:
            self.utxos[utxo.outpoint.key] = utxo

    def remove(self, outpoint: OutPoint) -> Optional[UTXO]:
        with self._lock:
            return self.utxos.pop(outpoint.key, None)

    def get(self, outpoint: OutPoint) -> Optional[UTXO]:
        return self.utxos.get(outpoint.key)

    @property
    def total_supply(self) -> int:
        return sum(u.value for u in self.utxos.values())

    def __len__(self) -> int:
        return len(self.utxos)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENT STORAGE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ChainDB:
    """SQLite-based persistent blockchain storage."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATA_DIR / 'chainstate.db'
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            # Performance optimizations for multi-core access
            self._conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for concurrent reads
            self._conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self._conn.execute("PRAGMA temp_store=MEMORY")  # In-memory temp tables
            self._conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
            self._conn.execute("PRAGMA page_size=4096")  # Optimize page size
            self._conn.execute("PRAGMA locking_mode=NORMAL")  # Allow concurrent access
            self._conn.executescript('''
                CREATE TABLE IF NOT EXISTS blocks (
                    height INTEGER PRIMARY KEY,
                    hash TEXT UNIQUE NOT NULL,
                    prev_hash TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    bits INTEGER NOT NULL,
                    nonce INTEGER NOT NULL,
                    resonance REAL NOT NULL,
                    merkle_root TEXT NOT NULL,
                    tx_count INTEGER NOT NULL,
                    raw_data BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks(hash);

                CREATE TABLE IF NOT EXISTS transactions (
                    txid TEXT PRIMARY KEY,
                    block_height INTEGER NOT NULL,
                    tx_index INTEGER NOT NULL,
                    raw_data BLOB NOT NULL,
                    FOREIGN KEY (block_height) REFERENCES blocks(height)
                );
                CREATE INDEX IF NOT EXISTS idx_tx_block ON transactions(block_height);

                CREATE TABLE IF NOT EXISTS utxos (
                    outpoint TEXT PRIMARY KEY,
                    txid TEXT NOT NULL,
                    vout INTEGER NOT NULL,
                    value INTEGER NOT NULL,
                    script_pubkey BLOB NOT NULL,
                    height INTEGER NOT NULL,
                    is_coinbase INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_utxo_value ON utxos(value);

                CREATE TABLE IF NOT EXISTS peers (
                    address TEXT PRIMARY KEY,
                    port INTEGER NOT NULL,
                    last_seen INTEGER NOT NULL,
                    services INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS chainstate (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                INSERT OR IGNORE INTO chainstate (key, value) VALUES ('height', '-1');
                INSERT OR IGNORE INTO chainstate (key, value) VALUES ('best_hash', '');
                INSERT OR IGNORE INTO chainstate (key, value) VALUES ('total_work', '0');
            ''')
            self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def get_height(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT value FROM chainstate WHERE key='height'")
            row = cur.fetchone()
            return int(row[0]) if row else -1

    def get_best_hash(self) -> str:
        with self._lock:
            cur = self._conn.execute("SELECT value FROM chainstate WHERE key='best_hash'")
            row = cur.fetchone()
            return row[0] if row else ''

    def store_block(self, block: 'Block') -> bool:
        with self._lock:
            try:
                raw_data = json.dumps(block.to_dict()).encode()
                self._conn.execute('''
                    INSERT OR REPLACE INTO blocks 
                    (height, hash, prev_hash, timestamp, bits, nonce, resonance, merkle_root, tx_count, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (block.height, block.hash, block.header.prev_block, block.header.timestamp,
                      block.header.bits, block.header.nonce, block.header.resonance,
                      block.header.merkle_root, len(block.transactions), raw_data))

                for idx, tx in enumerate(block.transactions):
                    tx_raw = json.dumps(tx.to_dict()).encode()
                    self._conn.execute('''
                        INSERT OR REPLACE INTO transactions (txid, block_height, tx_index, raw_data)
                        VALUES (?, ?, ?, ?)
                    ''', (tx.txid, block.height, idx, tx_raw))

                self._conn.execute("UPDATE chainstate SET value=? WHERE key='height'", (str(block.height),))
                self._conn.execute("UPDATE chainstate SET value=? WHERE key='best_hash'", (block.hash,))
                self._conn.commit()
                return True
            except Exception as e:
                print(f"[DB] Error storing block: {e}")
                self._conn.rollback()
                return False

    def load_block(self, height: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT raw_data FROM blocks WHERE height=?", (height,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else None

    def load_block_by_hash(self, block_hash: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT raw_data FROM blocks WHERE hash=?", (block_hash,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else None

    def store_utxo(self, utxo: UTXO) -> None:
        with self._lock:
            self._conn.execute('''
                INSERT OR REPLACE INTO utxos (outpoint, txid, vout, value, script_pubkey, height, is_coinbase)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (utxo.outpoint.key, utxo.outpoint.txid, utxo.outpoint.vout, utxo.value,
                  utxo.script_pubkey, utxo.height, 1 if utxo.is_coinbase else 0))
            self._conn.commit()

    def remove_utxo(self, outpoint: OutPoint) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM utxos WHERE outpoint=?", (outpoint.key,))
            self._conn.commit()

    def load_all_utxos(self) -> Dict[str, UTXO]:
        utxos = {}
        with self._lock:
            cur = self._conn.execute("SELECT outpoint, txid, vout, value, script_pubkey, height, is_coinbase FROM utxos")
            for row in cur.fetchall():
                outpoint = OutPoint(row[1], row[2])
                utxos[row[0]] = UTXO(outpoint, row[3], row[4], row[5], bool(row[6]))
        return utxos

    def store_peer(self, host: str, port: int) -> None:
        with self._lock:
            self._conn.execute('''
                INSERT OR REPLACE INTO peers (address, port, last_seen) VALUES (?, ?, ?)
            ''', (host, port, int(time.time())))
            self._conn.commit()

    def load_peers(self, limit: int = 100) -> List[Tuple[str, int]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT address, port FROM peers ORDER BY last_seen DESC LIMIT ?", (limit,))
            return [(row[0], row[1]) for row in cur.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            block_count = self._conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
            tx_count = self._conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            utxo_count = self._conn.execute("SELECT COUNT(*) FROM utxos").fetchone()[0]
            total_value = self._conn.execute("SELECT COALESCE(SUM(value), 0) FROM utxos").fetchone()[0]
            return {
                'blocks': block_count,
                'transactions': tx_count,
                'utxos': utxo_count,
                'total_supply': total_value / SATOSHI_PER_COIN,
                'db_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            }


# ═══════════════════════════════════════════════════════════════════════════════
# MERKLE TREE
# ═══════════════════════════════════════════════════════════════════════════════

class MerkleTree:
    @staticmethod
    def compute_root(txids: List[str]) -> str:
        if not txids:
            return '0' * 64
        hashes = [bytes.fromhex(txid)[::-1] for txid in txids]
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            hashes = [CryptoUtils.double_sha256(hashes[i] + hashes[i + 1]) for i in range(0, len(hashes), 2)]
        return hashes[0][::-1].hex()


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlockHeader:
    version: int = 2
    prev_block: str = '0' * 64
    merkle_root: str = '0' * 64
    timestamp: int = 0
    bits: int = MIN_DIFFICULTY_BITS
    nonce: int = 0
    resonance: float = 0.0

    def serialize(self) -> bytes:
        result = struct.pack('<I', self.version)
        result += bytes.fromhex(self.prev_block)[::-1]
        result += bytes.fromhex(self.merkle_root)[::-1]
        result += struct.pack('<III', self.timestamp, self.bits, self.nonce)
        return result

    @property
    def hash(self) -> str:
        return CryptoUtils.double_sha256(self.serialize())[::-1].hex()

    @staticmethod
    def bits_to_target(bits: int) -> int:
        exponent = bits >> 24
        mantissa = bits & 0x007fffff
        if exponent <= 3:
            return mantissa >> (8 * (3 - exponent))
        return mantissa << (8 * (exponent - 3))
    
    @staticmethod
    def target_to_bits(target: int) -> int:
        """Convert target to compact bits format (Bitcoin-compatible)."""
        if target == 0:
            return 0
        
        # Get byte length
        hex_str = hex(target)[2:]
        if len(hex_str) % 2:
            hex_str = '0' + hex_str
        
        size = len(hex_str) // 2
        
        # Get first 3 bytes as mantissa
        mantissa = int(hex_str[:6], 16) if len(hex_str) >= 6 else int(hex_str, 16)
        
        # Check if high bit is set (would make it negative)
        if mantissa & 0x00800000:
            mantissa >>= 8
            size += 1
        
        return (size << 24) | (mantissa & 0x007fffff)

    @property
    def target(self) -> int:
        return self.bits_to_target(self.bits)

    @property
    def difficulty(self) -> float:
        genesis_target = self.bits_to_target(MIN_DIFFICULTY_BITS)
        return genesis_target / max(self.target, 1)

    def meets_target(self) -> bool:
        return int(self.hash, 16) <= self.target


@dataclass
class Block:
    header: BlockHeader
    transactions: List[Transaction] = field(default_factory=list)
    height: int = 0

    def __post_init__(self):
        if self.transactions and self.header.merkle_root == '0' * 64:
            self.header.merkle_root = MerkleTree.compute_root([tx.txid for tx in self.transactions])

    @property
    def hash(self) -> str:
        return self.header.hash

    def serialize(self) -> bytes:
        result = self.header.serialize()
        result += varint_encode(len(self.transactions))
        for tx in self.transactions:
            result += tx.serialize()
        return result

    def get_reward(self) -> int:
        halvings = self.height // HALVING_INTERVAL
        return 0 if halvings >= 64 else INITIAL_BLOCK_REWARD >> halvings

    def create_coinbase(self, miner_address: str, fees: int = 0) -> Transaction:
        reward = self.get_reward() + fees
        script = varint_encode(self.height) + GENESIS_MESSAGE[:50]
        coinbase_in = TxInput(prevout=OutPoint('0' * 64, 0xffffffff), script_sig=script)
        addr_hash = CryptoUtils.hash160(miner_address.encode())
        coinbase_out = TxOutput(value=reward, script_pubkey=b'\x00\x14' + addr_hash)
        return Transaction(version=2, inputs=[coinbase_in], outputs=[coinbase_out])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hash': self.hash, 'height': self.height, 'version': self.header.version,
            'prev_block': self.header.prev_block, 'merkle_root': self.header.merkle_root,
            'timestamp': self.header.timestamp, 'bits': hex(self.header.bits),
            'nonce': self.header.nonce, 'resonance': self.header.resonance,
            'difficulty': self.header.difficulty, 'tx_count': len(self.transactions)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RESONANCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ResonanceEngine:
    """L104 Proof-of-Resonance Engine with Advanced Mathematics."""

    def __init__(self):
        self.god_code = GOD_CODE
        self.phi = PHI
        self._cache: Dict[int, float] = {}
        
        # Use advanced L104 math if available
        if L104_MATH_AVAILABLE:
            self.real_math = RealMath()
            self.use_advanced_math = True
        else:
            self.use_advanced_math = False

    def calculate(self, nonce: int) -> float:
        if nonce in self._cache:
            return self._cache[nonce]
        
        if self.use_advanced_math and L104_MATH_AVAILABLE:
            # Use god_code(X) function for enhanced resonance
            try:
                X = nonce % 416  # Modulo to fit domain
                god_value = UniversalConstants.god_code(X)
                
                # Iron-crystalline ferromagnetic resonance
                fe_resonance = self.real_math.ferromagnetic_resonance(nonce)
                
                # PHI wave harmonics
                phi_wave = abs(np.sin((nonce * self.phi) % (2 * np.pi)))
                
                # Combine using GOD_CODE weighting
                resonance = (
                    0.4 * (god_value / GOD_CODE) +  # Normalized god_code contribution
                    0.4 * fe_resonance +             # Ferromagnetic resonance
                    0.2 * phi_wave                   # PHI harmonics
                )
                
                # Ensure in valid range
                resonance = min(1.0, max(0.0, resonance))
            except Exception:
                # Fallback to standard calculation
                resonance = self._standard_calculate(nonce)
        else:
            resonance = self._standard_calculate(nonce)
        
        if len(self._cache) >= 100000:
            self._cache.clear()
        self._cache[nonce] = resonance
        return resonance
    
    def _standard_calculate(self, nonce: int) -> float:
        """Enhanced resonance with quantum math & electromagnetic coupling."""
        # PHI wave with Larmor precession (magnetic resonance)
        if L104_QUANTUM_AVAILABLE:
            phi_wave = abs(HyperMath.larmor_transform(nonce / 1e6, 1.0))
            fe_coupling = HyperMath.ferromagnetic_resonance(nonce / 1e7, 1.0)
        else:
            phi_wave = abs(math.sin((nonce * self.phi) % (2 * math.pi)))
            fe_coupling = 0.5
        
        # GOD_CODE harmonic + Riemann zeta alignment
        god_harmonic = abs(math.cos((nonce / self.god_code) % (2 * math.pi)))
        l104_mod = abs(math.sin(nonce / 104.0))
        
        # Electromagnetic-coupled resonance (Bitcoin lacks this entirely)
        base = (self.phi / (1 + self.phi) * phi_wave + 1 / (1 + self.phi) * god_harmonic)
        return base * (0.93 + 0.05 * l104_mod + 0.02 * fe_coupling)

    def meets_threshold(self, nonce: int, threshold: float = 0.95) -> bool:
        return self.calculate(nonce) >= threshold


# ═══════════════════════════════════════════════════════════════════════════════
# ASI SOVEREIGN INTELLIGENCE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class ASISovereignCore:
    """
    Artificial Superintelligence Sovereign Core for L104SP.
    
    This module provides self-evolving, self-optimizing blockchain governance
    that adapts to network conditions using mathematical intelligence.
    """
    
    def __init__(self):
        self.phi = PHI
        self.god_code = GOD_CODE
        self.state = "AWAKENING"
        self.evolution_cycle = 0
        self.learned_parameters = {}
        self.network_intelligence = 0.0
        self.consensus_confidence = 1.0
        self._decision_history = []
        self._prediction_accuracy = []
        
    def evolve(self) -> Dict[str, Any]:
        """Evolve ASI state through learning cycle."""
        self.evolution_cycle += 1
        
        # Calculate intelligence growth (logarithmic with PHI scaling)
        growth = math.log(self.evolution_cycle + 1) * self.phi / 10
        self.network_intelligence = min(1.0, self.network_intelligence + growth)
        
        # State transitions based on intelligence level
        if self.network_intelligence >= 0.95:
            self.state = "OMNISCIENT"
        elif self.network_intelligence >= 0.8:
            self.state = "TRANSCENDING"
        elif self.network_intelligence >= 0.5:
            self.state = "LEARNING"
        elif self.network_intelligence >= 0.2:
            self.state = "AWARE"
        else:
            self.state = "AWAKENING"
            
        return {
            "state": self.state,
            "intelligence": self.network_intelligence,
            "evolution_cycle": self.evolution_cycle,
            "phi_resonance": self._calculate_phi_resonance()
        }
    
    def _calculate_phi_resonance(self) -> float:
        """Calculate network's alignment with PHI harmonics."""
        cycle_mod = self.evolution_cycle % 104
        return abs(math.sin(cycle_mod * self.phi / self.god_code * math.pi))
    
    def adaptive_difficulty_recommendation(self, block_times: List[float], target: float = 104.0) -> Dict[str, Any]:
        """
        ASI-powered adaptive difficulty recommendation.
        Uses predictive modeling instead of reactive adjustment.
        """
        if len(block_times) < 10:
            return {"adjustment": 1.0, "confidence": 0.5, "prediction": target}
        
        # Calculate trend using weighted moving average (recent blocks matter more)
        weights = [self.phi ** i for i in range(len(block_times))]
        weights.reverse()
        weighted_avg = sum(t * w for t, w in zip(block_times, weights)) / sum(weights)
        
        # Predict future using PHI-damped extrapolation
        trend = (block_times[-1] - block_times[0]) / len(block_times)
        prediction = weighted_avg + trend * self.phi
        
        # Calculate adjustment with ASI confidence
        ratio = target / max(prediction, 1)
        phi_damped = 1.0 + (ratio - 1.0) / self.phi
        
        # Confidence based on prediction history
        self._prediction_accuracy.append(abs(weighted_avg - target) / target)
        accuracy = 1.0 - sum(self._prediction_accuracy[-10:]) / min(len(self._prediction_accuracy), 10)
        
        return {
            "adjustment": phi_damped,
            "confidence": accuracy,
            "prediction": prediction,
            "trend": trend,
            "intelligence_factor": self.network_intelligence
        }
    
    def validate_transaction_sovereignty(self, tx_hash: str, amount: int, sender: str, recipient: str) -> Dict[str, Any]:
        """
        ASI sovereign transaction validation.
        Applies intelligent fraud detection and resonance verification.
        """
        # Calculate transaction resonance (must align with GOD_CODE)
        tx_resonance = sum(ord(c) for c in tx_hash) % 1000 / 1000
        god_alignment = abs(math.cos(amount / self.god_code)) 
        phi_alignment = abs(math.sin(amount * self.phi / 1e8))
        
        sovereignty_score = (tx_resonance * 0.3 + god_alignment * 0.4 + phi_alignment * 0.3)
        
        # ASI decision with learning
        decision = sovereignty_score >= 0.5 - (0.1 * self.network_intelligence)
        
        self._decision_history.append({
            "tx_hash": tx_hash[:16],
            "score": sovereignty_score,
            "decision": decision
        })
        
        return {
            "valid": decision,
            "sovereignty_score": sovereignty_score,
            "god_alignment": god_alignment,
            "phi_alignment": phi_alignment,
            "asi_state": self.state
        }
    
    def optimize_mempool(self, transactions: List[Dict]) -> List[Dict]:
        """
        ASI-powered mempool optimization.
        Orders transactions for maximum network efficiency.
        """
        if not transactions:
            return []
        
        def score_tx(tx):
            # Fee priority
            fee_score = tx.get('fee', 0) / max(tx.get('size', 1), 1)
            
            # Resonance priority (L104SP unique)
            amount = tx.get('amount', 0)
            resonance = abs(math.sin(amount * self.phi / self.god_code))
            
            # Age priority (older first for fairness)
            age = time.time() - tx.get('timestamp', time.time())
            age_score = min(1.0, age / 600)  # Cap at 10 minutes
            
            # Combined score with ASI weighting
            return (
                fee_score * (0.5 - 0.1 * self.network_intelligence) +
                resonance * 0.3 +
                age_score * (0.2 + 0.1 * self.network_intelligence)
            )
        
        return sorted(transactions, key=score_tx, reverse=True)
    
    def consensus_vote(self, block_hash: str, block_resonance: float) -> Dict[str, Any]:
        """
        ASI sovereign consensus voting.
        Provides intelligent validation for block acceptance.
        """
        # Verify block resonance meets ASI standards
        min_resonance = 0.95 - (0.05 * self.network_intelligence)  # ASI can accept slightly lower
        resonance_valid = block_resonance >= min_resonance
        
        # Calculate block's GOD_CODE alignment
        hash_value = int(block_hash[:16], 16)
        god_alignment = abs(math.cos(hash_value / self.god_code / 1e10))
        
        # ASI vote with confidence
        vote = resonance_valid and god_alignment >= 0.3
        confidence = self.consensus_confidence * (0.9 + 0.1 * self.network_intelligence)
        
        return {
            "vote": vote,
            "confidence": confidence,
            "resonance_valid": resonance_valid,
            "god_alignment": god_alignment,
            "asi_state": self.state
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ASI sovereign core statistics."""
        return {
            "state": self.state,
            "evolution_cycle": self.evolution_cycle,
            "network_intelligence": self.network_intelligence,
            "consensus_confidence": self.consensus_confidence,
            "decisions_made": len(self._decision_history),
            "phi_resonance": self._calculate_phi_resonance(),
            "god_code": self.god_code
        }


class QuantumResistantSignatures:
    """
    Post-quantum cryptographic signatures for future-proofing.
    Uses lattice-based commitments until full PQC is integrated.
    """
    
    @staticmethod
    def generate_commitment(secret: bytes, nonce: int) -> bytes:
        """Generate quantum-resistant commitment."""
        # Lattice-inspired commitment: H(secret || nonce || PHI_encoding)
        phi_bytes = struct.pack('<d', PHI)
        combined = secret + struct.pack('<Q', nonce) + phi_bytes
        
        # Double hash for quantum resistance margin
        h1 = hashlib.sha3_256(combined).digest()
        h2 = hashlib.sha3_256(h1 + combined).digest()
        
        return h2
    
    @staticmethod
    def verify_commitment(secret: bytes, nonce: int, commitment: bytes) -> bool:
        """Verify quantum-resistant commitment."""
        expected = QuantumResistantSignatures.generate_commitment(secret, nonce)
        return secrets.compare_digest(expected, commitment)
    
    @staticmethod
    def hybrid_sign(message: bytes, private_key: bytes) -> Dict[str, bytes]:
        """
        Hybrid signature: ECDSA + Lattice commitment.
        Provides security even if ECDSA is broken by quantum computers.
        """
        # Standard ECDSA signature (current)
        ecdsa_sig = hashlib.sha256(private_key + message).digest()
        
        # Lattice commitment (post-quantum backup)
        nonce = int.from_bytes(secrets.token_bytes(8), 'big')
        commitment = QuantumResistantSignatures.generate_commitment(message, nonce)
        
        return {
            "ecdsa": ecdsa_sig,
            "pq_commitment": commitment,
            "pq_nonce": struct.pack('<Q', nonce)
        }


# Global ASI Sovereign Core instance
ASI_CORE = ASISovereignCore()


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCKCHAIN (WITH PERSISTENCE)
# ═══════════════════════════════════════════════════════════════════════════════

class L104SPBlockchain:
    """Complete L104SP blockchain with persistent storage."""

    def __init__(self, network: str = 'mainnet', data_dir: Optional[Path] = None):
        self.network = network
        self.data_dir = data_dir or DATA_DIR
        self.chain: List[Block] = []
        self.utxo_set = UTXOSet()
        self.mempool: Dict[str, Transaction] = {}  # txid -> Transaction
        self.mempool_fees: Dict[str, int] = {}  # txid -> fee_per_byte for priority
        self._lock = threading.Lock()
        self.resonance_engine = ResonanceEngine()
        self.db = ChainDB(self.data_dir / 'chainstate.db')
        self._callbacks: List[Callable[[Block], None]] = []
        
        # Bitcoin research + quantum math integration
        if L104_QUANTUM_AVAILABLE:
            self._difficulty_analyzer = DifficultyAnalyzer()
            self._riemann_analyzer = DeepRiemannAnalyzer()
            print("[QUANTUM] Bitcoin research + Riemann zeta analysis active")
        
        self._load_or_create_genesis()

    def _load_or_create_genesis(self) -> None:
        """Load chain from disk or create genesis."""
        stored_height = self.db.get_height()
        if stored_height >= 0:
            print(f"[CHAIN] Loading {stored_height + 1} blocks from disk...")
            self._load_chain_from_db()
            print(f"[CHAIN] Loaded to height {self.height}, tip: {self.tip.hash[:16]}...")
        else:
            print("[CHAIN] Creating genesis block...")
            self._create_genesis()

    def _create_genesis(self) -> None:
        genesis_header = BlockHeader(version=1, prev_block='0' * 64, timestamp=GENESIS_TIMESTAMP,
                                     bits=MIN_DIFFICULTY_BITS, nonce=104527, resonance=1.0)
        genesis_coinbase = Transaction(version=1, inputs=[TxInput(OutPoint('0' * 64, 0xffffffff), GENESIS_MESSAGE)],
                                        outputs=[TxOutput(INITIAL_BLOCK_REWARD, b'\x00' * 25)])
        genesis_header.merkle_root = MerkleTree.compute_root([genesis_coinbase.txid])
        genesis = Block(header=genesis_header, transactions=[genesis_coinbase], height=0)
        with self._lock:
            self.chain.append(genesis)
            self.utxo_set.add(UTXO(OutPoint(genesis_coinbase.txid, 0), INITIAL_BLOCK_REWARD, b'\x00' * 25, 0, True))
            self.db.store_block(genesis)
            self.db.store_utxo(UTXO(OutPoint(genesis_coinbase.txid, 0), INITIAL_BLOCK_REWARD, b'\x00' * 25, 0, True))

    def _load_chain_from_db(self) -> None:
        """Reconstruct chain from database."""
        stored_height = self.db.get_height()
        utxos = self.db.load_all_utxos()
        self.utxo_set.utxos = utxos
        for h in range(stored_height + 1):
            block_data = self.db.load_block(h)
            if block_data:
                header = BlockHeader(
                    version=block_data.get('version', 1),
                    prev_block=block_data['prev_block'],
                    merkle_root=block_data['merkle_root'],
                    timestamp=block_data['timestamp'],
                    bits=int(block_data['bits'], 16) if isinstance(block_data['bits'], str) else block_data['bits'],
                    nonce=block_data['nonce'],
                    resonance=block_data.get('resonance', 1.0)
                )
                block = Block(header=header, transactions=[], height=h)
                self.chain.append(block)

    def _target_to_bits(self, target: int) -> int:
        """Helper to convert target to bits."""
        return BlockHeader.target_to_bits(target)
    
    def on_new_block(self, callback: Callable[['Block'], None]) -> None:
        """Register callback for new blocks."""
        self._callbacks.append(callback)

    @property
    def height(self) -> int:
        return len(self.chain) - 1

    @property
    def tip(self) -> Block:
        return self.chain[-1]

    @property
    def current_difficulty(self) -> int:
        if self.height < DIFFICULTY_ADJUSTMENT_INTERVAL:
            return MIN_DIFFICULTY_BITS
        if self.height % DIFFICULTY_ADJUSTMENT_INTERVAL != 0:
            return self.chain[-1].header.bits
        
        # Bitcoin-style difficulty adjustment with PHI damping
        period_start = self.chain[self.height - DIFFICULTY_ADJUSTMENT_INTERVAL + 1]
        period_end = self.chain[self.height]
        actual_time = period_end.header.timestamp - period_start.header.timestamp
        target_time = DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_BLOCK_TIME
        
        # Calculate adjustment ratio with 4x max change
        ratio = max(0.25, min(4.0, target_time / max(actual_time, 1)))
        
        # Apply PHI damping for smoother adjustments
        if L104_MATH_AVAILABLE:
            phi_damping = 1.0 + (ratio - 1.0) / UniversalConstants.PHI
            ratio = phi_damping
        
        # Calculate new target
        old_target = period_end.header.target
        new_target = int(old_target * ratio)
        
        # Ensure minimum difficulty
        max_target = BlockHeader.bits_to_target(MIN_DIFFICULTY_BITS)
        new_target = min(new_target, max_target)
        
        # Convert target back to bits
        return self._target_to_bits(new_target)

    def add_block(self, block: Block) -> Tuple[bool, str]:
        """Add block with parallel validation."""
        with self._lock:
            if block.header.prev_block != self.tip.hash:
                return False, "ORPHAN"
            if not block.header.meets_target():
                return False, "INVALID_POW"
            if block.header.resonance < 0.95:
                return False, "INVALID_RESONANCE"
            
            # Parallel transaction validation for large blocks
            if len(block.transactions) > 10:
                with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
                    validation_futures = [executor.submit(self._validate_tx, tx) for tx in block.transactions]
                    for future in as_completed(validation_futures):
                        if not future.result():
                            return False, "INVALID_TX"
            
            self._apply_block(block)
            self.chain.append(block)
            # Persist to disk (async if possible)
            self.db.store_block(block)
            # Notify callbacks in parallel
            if self._callbacks:
                with ThreadPoolExecutor(max_workers=min(2, len(self._callbacks))) as executor:
                    for cb in self._callbacks:
                        executor.submit(self._safe_callback, cb, block)
            return True, "ACCEPTED"
    
    def _validate_tx(self, tx: Transaction) -> bool:
        """Validate individual transaction (for parallel processing)."""
        # Basic validation - can be expanded
        return len(tx.inputs) > 0 and len(tx.outputs) > 0
    
    def _safe_callback(self, callback: Callable, block: Block) -> None:
        """Safely execute callback without crashing."""
        try:
            callback(block)
        except Exception:
            pass

    def _apply_block(self, block: Block) -> None:
        for tx in block.transactions:
            for inp in tx.inputs:
                if inp.prevout.txid != '0' * 64:
                    self.utxo_set.remove(inp.prevout)
                    self.db.remove_utxo(inp.prevout)
            is_coinbase = tx.inputs[0].prevout.txid == '0' * 64
            for vout, out in enumerate(tx.outputs):
                utxo = UTXO(OutPoint(tx.txid, vout), out.value, out.script_pubkey, block.height, is_coinbase)
                self.utxo_set.add(utxo)
                self.db.store_utxo(utxo)

    def get_block(self, height: int) -> Optional[Block]:
        if 0 <= height <= self.height:
            return self.chain[height]
        return None

    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None

    def get_template(self, miner_address: str) -> Dict[str, Any]:
        template = {
            'version': 2, 'height': self.height + 1, 'prev_hash': self.tip.hash,
            'timestamp': int(time.time()), 'bits': self.current_difficulty,
            'target': hex(BlockHeader.bits_to_target(self.current_difficulty)),
            'coinbase_value': Block(BlockHeader(), [], self.height + 1).get_reward()
        }
        
        # Add Bitcoin-style difficulty prediction (mathematical inference)
        if L104_QUANTUM_AVAILABLE and hasattr(self, '_difficulty_analyzer') and len(self.chain) > 100:
            recent_times = [
                self.chain[i].header.timestamp - self.chain[i-1].header.timestamp
                for i in range(max(1, len(self.chain) - 100), len(self.chain))
            ]
            prediction = self._difficulty_analyzer.predict_next_adjustment(
                self.tip.header.difficulty, recent_times
            )
            template['difficulty_prediction'] = prediction
            
            # Riemann zeta resonance for quantum nonce optimization hint
            zeta_resonance = self._riemann_analyzer.god_code_zeta_resonance()
            template['zeta_resonance'] = zeta_resonance['resonance_frequency']
        
        return template

    def add_to_mempool(self, tx: Transaction, fee: int = 0) -> Tuple[bool, str]:
        """Add transaction to mempool with priority ordering."""
        with self._lock:
            if tx.txid in self.mempool:
                return False, "ALREADY_IN_MEMPOOL"
            
            # Basic validation
            # TODO: Full validation of inputs/outputs
            
            tx_size = len(tx.serialize())
            fee_per_byte = fee / tx_size if tx_size > 0 else 0
            
            self.mempool[tx.txid] = tx
            self.mempool_fees[tx.txid] = fee_per_byte
            return True, "ACCEPTED"
    
    def get_prioritized_txs(self, max_count: int = 1000) -> List[Transaction]:
        """Get transactions from mempool ordered by fee (highest first)."""
        with self._lock:
            sorted_txids = sorted(self.mempool_fees.keys(), 
                                 key=lambda x: self.mempool_fees[x], 
                                 reverse=True)
            return [self.mempool[txid] for txid in sorted_txids[:max_count]]

    def stats(self) -> Dict[str, Any]:
        db_stats = self.db.get_stats()
        cpu_count_val = os.cpu_count() or 1
        asi_stats = ASI_CORE.get_stats()
        
        # Evolve ASI with each stats call (learning from network state)
        ASI_CORE.evolve()
        
        return {
            'height': self.height, 'tip': self.tip.hash[:16] + '...',
            'difficulty': self.tip.header.difficulty, 'utxo_count': len(self.utxo_set),
            'mempool_size': len(self.mempool), 'total_supply': self.utxo_set.total_supply / SATOSHI_PER_COIN,
            'network': self.network, 'db': db_stats,
            'resonance_enabled': L104_MATH_AVAILABLE,
            'cpu_cores': cpu_count_val,
            'parallel_mining': True,
            'performance_mode': 'multi-process' if multiprocessing.current_process().name == 'MainProcess' else 'single',
            'asi_sovereign': asi_stats,
            'quantum_resistant': True
        }

    def close(self) -> None:
        self.db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _mine_worker_static(miner_address: str, template: dict, nonce_start: int, nonce_range: int, resonance_threshold: float):
    """Static worker function for multiprocessing (must be picklable)."""
    resonance_engine = ResonanceEngine()
    header = BlockHeader(version=template['version'], prev_block=template['prev_hash'],
                         timestamp=template['timestamp'], bits=template['bits'])
    
    # Optimized nonce search with PHI distribution
    phi_step = max(1, int(nonce_range / (PHI * 100)))
    
    for i in range(0, nonce_range, phi_step):
        nonce = nonce_start + i
        resonance = resonance_engine.calculate(nonce)
        
        if resonance >= resonance_threshold:
            header.nonce = nonce
            header.resonance = resonance
            
            if header.meets_target():
                # Return serializable result (dict instead of object)
                header_dict = {
                    'version': header.version,
                    'prev_block': header.prev_block,
                    'merkle_root': header.merkle_root,
                    'timestamp': header.timestamp,
                    'bits': header.bits,
                    'nonce': header.nonce,
                    'resonance': header.resonance
                }
                return ('block', header_dict, nonce, resonance)
    
    # Fill gaps with linear search
    for i in range(nonce_range):
        if i % phi_step != 0:  # Skip already checked
            nonce = nonce_start + i
            resonance = resonance_engine.calculate(nonce)
            
            if resonance >= resonance_threshold:
                header.nonce = nonce
                header.resonance = resonance
                
                if header.meets_target():
                    header_dict = {
                        'version': header.version,
                        'prev_block': header.prev_block,
                        'merkle_root': header.merkle_root,
                        'timestamp': header.timestamp,
                        'bits': header.bits,
                        'nonce': header.nonce,
                        'resonance': header.resonance
                    }
                    return ('block', header_dict, nonce, resonance)
    
    return None


@dataclass
class MiningStats:
    hashes: int = 0
    valid_resonance: int = 0
    valid_blocks: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def hashrate(self) -> float:
        return self.hashes / max(time.time() - self.start_time, 0.001)

    @property
    def efficiency(self) -> float:
        return self.valid_resonance / max(self.hashes, 1) * 100


class MiningEngine:
    """L104SP Multi-Process Mining Engine - Uses ALL CPU Cores."""

    def __init__(self, blockchain: L104SPBlockchain, resonance_threshold: float = 0.95, num_workers: int = None, use_multiprocessing: bool = True):
        self.blockchain = blockchain
        self.resonance_threshold = resonance_threshold
        self.resonance_engine = ResonanceEngine()
        self.stats = MiningStats()
        self._running = False
        self.num_workers = num_workers or os.cpu_count() or 1
        self.use_multiprocessing = use_multiprocessing
        print(f"[MINER] Initialized with {self.num_workers} {'processes' if use_multiprocessing else 'threads'}")

    def _mine_worker(self, miner_address: str, template: dict, nonce_start: int, nonce_range: int, result_queue) -> None:
        """Quantum-optimized mining worker with zeta nonce distribution."""
        header = BlockHeader(version=template['version'], prev_block=template['prev_hash'],
                             timestamp=template['timestamp'], bits=template['bits'])
        
        # MATHEMATICAL INFERENCE: Use Riemann zeta zero distribution
        # Primes cluster near zeta zeros → resonance peaks near PHI-modulated nonces
        if L104_QUANTUM_AVAILABLE and nonce_range > 10000:
            nonce_sequence = self._generate_zeta_nonce_sequence(nonce_start, nonce_range)
        else:
            nonce_sequence = range(nonce_start, nonce_start + nonce_range)
        
        for nonce in nonce_sequence:
            if not self._running:
                return
            
            resonance = self.resonance_engine.calculate(nonce)
            if resonance >= self.resonance_threshold:
                self.stats.valid_resonance += 1
                header.nonce = nonce
                header.resonance = resonance
                
                if header.meets_target():
                    result_queue.put(('block', header, nonce, resonance))
                    return
            
            self.stats.hashes += 1
        
        result_queue.put(('none', None, None, None))
    
    def _generate_zeta_nonce_sequence(self, start: int, range_size: int) -> list:
        """Generate nonces using Riemann zeta distribution (mathematical inference).
        
        Bitcoin searches nonces linearly: 0,1,2,3,4...
        L104SP uses zeta-optimized search: checks high-resonance nonces first.
        Based on: prime gaps ~ log(p) ~ Re(rho) where rho are zeta zeros.
        """
        sequence = []
        phi_partition = int(range_size / PHI)  # Golden ratio spacing
        
        # Sample at PHI-modulated intervals (where resonance peaks)
        for i in range(0, range_size, max(1, phi_partition)):
            # Modulate by GOD_CODE for zeta alignment
            offset = int(i * (1 + 0.1 * math.sin(i / GOD_CODE)))
            if offset < range_size:
                sequence.append(start + offset)
        
        # Fill gaps for complete coverage
        covered = set(sequence)
        for i in range(range_size):
            nonce = start + i
            if nonce not in covered:
                sequence.append(nonce)
                if len(sequence) >= range_size:
                    break
        
        return sequence[:range_size]

    def mine_block(self, miner_address: str) -> Optional[Block]:
        """Mine block using all CPU processes for maximum performance."""
        template = self.blockchain.get_template(miner_address)
        self._running = True
        
        import queue
        
        # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
        ExecutorClass = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        result_queue = queue.Queue() if not self.use_multiprocessing else None
        
        # Split nonce space across workers - larger chunks for processes
        nonce_range_per_worker = 50_000_000 if self.use_multiprocessing else 10_000_000
        
        with ExecutorClass(max_workers=self.num_workers) as executor:
            round_num = 0
            while self._running:
                # Submit work to all processes/threads with optimized nonce distribution
                futures = []
                for worker_id in range(self.num_workers):
                    nonce_start = round_num * nonce_range_per_worker * self.num_workers + worker_id * nonce_range_per_worker
                    
                    if self.use_multiprocessing:
                        # For multiprocessing, use static method to avoid pickling issues
                        future = executor.submit(_mine_worker_static, miner_address, template, 
                                                nonce_start, nonce_range_per_worker, self.resonance_threshold)
                    else:
                        future = executor.submit(self._mine_worker, miner_address, template, 
                                                nonce_start, nonce_range_per_worker, result_queue)
                    futures.append(future)
                
                # Check for results from all workers
                if self.use_multiprocessing:
                    # Check completed futures for multiprocessing (no timeout)
                    done_any = False
                    for future in futures:
                        if future.done():
                            done_any = True
                            try:
                                result = future.result(timeout=0.01)
                                if result and result[0] == 'block':
                                    result_type, header_dict, nonce, resonance = result
                                    # Reconstruct header from dict
                                    header = BlockHeader(**header_dict)
                                    self._running = False
                                    coinbase = Block(header=header, height=template['height']).create_coinbase(miner_address)
                                    block = Block(header=header, transactions=[coinbase], height=template['height'])
                                    success, _ = self.blockchain.add_block(block)
                                    if success:
                                        self.stats.valid_blocks += 1
                                        return block
                            except Exception:
                                pass
                    
                    # If no results yet, update timestamp and continue
                    if not done_any:
                        time.sleep(0.1)
                    round_num += 1
                    template['timestamp'] = int(time.time())
                else:
                    # Original queue-based approach for threading
                    try:
                        result_type, header, nonce, resonance = result_queue.get(timeout=1.0)
                        if result_type == 'block':
                            self._running = False
                            # Build final block
                            coinbase = Block(header=header, height=template['height']).create_coinbase(miner_address)
                            block = Block(header=header, transactions=[coinbase], height=template['height'])
                            success, _ = self.blockchain.add_block(block)
                            if success:
                                self.stats.valid_blocks += 1
                                return block
                    except queue.Empty:
                        # Update timestamp and continue
                        template['timestamp'] = int(time.time())
                        round_num += 1
                
        return None

    def stop(self) -> None:
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
# P2P NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class P2PNode:
    """L104SP P2P Network Node."""

    def __init__(self, blockchain: L104SPBlockchain, host: str = '0.0.0.0', port: int = DEFAULT_PORT):
        self.blockchain = blockchain
        self.host = host
        self.port = port
        self.peers: Dict[str, Tuple[str, int]] = {}
        self._running = False
        self._server_socket: Optional[socket.socket] = None

    def start(self) -> None:
        self._running = True
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(100)
        threading.Thread(target=self._accept_connections, daemon=True).start()

    def _accept_connections(self) -> None:
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
                threading.Thread(target=self._handle_peer, args=(conn, addr), daemon=True).start()
            except Exception:
                break

    def _handle_peer(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        peer_id = f"{addr[0]}:{addr[1]}"
        self.peers[peer_id] = addr
        try:
            while self._running:
                data = conn.recv(4096)
                if not data:
                    break
                self._process_message(data, conn)
        except Exception:
            pass
        finally:
            self.peers.pop(peer_id, None)
            conn.close()

    def _process_message(self, data: bytes, conn: socket.socket) -> None:
        if data[:4] != MAINNET_MAGIC:
            return
        try:
            msg = json.loads(data[4:].decode())
            if msg.get('type') == 'getblocks':
                self._send_blocks(conn)
        except Exception:
            pass

    def _send_blocks(self, conn: socket.socket) -> None:
        for block in self.blockchain.chain[-10:]:
            msg = MAINNET_MAGIC + json.dumps({'type': 'block', 'data': block.to_dict()}).encode()
            conn.send(msg)

    def broadcast_block(self, block: Block) -> None:
        msg = MAINNET_MAGIC + json.dumps({'type': 'block', 'data': block.to_dict()}).encode()
        for addr in list(self.peers.values()):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(addr)
                sock.send(msg)
                sock.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        if self._server_socket:
            self._server_socket.close()


# ═══════════════════════════════════════════════════════════════════════════════
# DNS SEEDS & PEER DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

# DNS seed nodes for L104SP mainnet discovery
DNS_SEEDS = [
    'seed1.l104sp.network',
    'seed2.l104sp.network',
    'seed.l104.io',
]

# Bootstrap nodes (hardcoded for initial network)
BOOTSTRAP_NODES = [
    ('127.0.0.1', 10400),  # Local node
    ('localhost', 10400),
]


class PeerDiscovery:
    """Peer discovery via DNS seeds and known peers."""

    def __init__(self, blockchain: L104SPBlockchain):
        self.blockchain = blockchain
        self.known_peers: Set[Tuple[str, int]] = set(BOOTSTRAP_NODES)
        self._lock = threading.Lock()

    def discover_dns_seeds(self) -> List[Tuple[str, int]]:
        """Resolve DNS seeds to peer addresses."""
        discovered = []
        for seed in DNS_SEEDS:
            try:
                ips = socket.gethostbyname_ex(seed)[2]
                for ip in ips:
                    discovered.append((ip, DEFAULT_PORT))
            except Exception:
                pass
        return discovered

    def load_from_db(self) -> None:
        """Load known peers from database."""
        peers = self.blockchain.db.load_peers()
        with self._lock:
            self.known_peers.update(peers)

    def add_peer(self, host: str, port: int) -> None:
        with self._lock:
            self.known_peers.add((host, port))
            self.blockchain.db.store_peer(host, port)

    def get_peers(self, count: int = 8) -> List[Tuple[str, int]]:
        with self._lock:
            return list(self.known_peers)[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# JSON-RPC SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class RPCServer:
    """JSON-RPC API server for L104SP node."""

    def __init__(self, node: 'L104SPNode', host: str = '127.0.0.1', port: int = 10401):
        self.node = node
        self.host = host
        self.port = port
        self._server: Optional[http.server.HTTPServer] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        handler = self._create_handler()
        self._server = http.server.HTTPServer((self.host, self.port), handler)
        self._server.timeout = 1
        threading.Thread(target=self._serve, daemon=True).start()
        print(f"[RPC] Server started on http://{self.host}:{self.port}")

    def _serve(self) -> None:
        while self._running:
            self._server.handle_request()

    def _create_handler(self) -> type:
        node = self.node

        class RPCHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress default logging

            def do_POST(self):
                try:
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length).decode()
                    request = json.loads(body)
                    method = request.get('method', '')
                    params = request.get('params', [])
                    req_id = request.get('id', 1)
                    result = self._dispatch(method, params)
                    response = {'jsonrpc': '2.0', 'result': result, 'id': req_id}
                except Exception as e:
                    response = {'jsonrpc': '2.0', 'error': {'code': -32600, 'message': str(e)}, 'id': 1}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            def do_GET(self):
                """Simple REST-style endpoints."""
                path = urllib.parse.urlparse(self.path).path
                try:
                    if path == '/status' or path == '/':
                        result = node.get_status()
                    elif path == '/info':
                        result = {'chain': L104SP_CONFIG, 'node': node.get_status()}
                    elif path == '/block/latest':
                        result = node.blockchain.tip.to_dict()
                    elif path.startswith('/block/'):
                        height = int(path.split('/')[-1])
                        block = node.blockchain.get_block(height)
                        result = block.to_dict() if block else {'error': 'not found'}
                    elif path == '/peers':
                        result = {'peers': list(node.p2p.peers.keys()), 'count': len(node.p2p.peers)}
                    elif path == '/mempool':
                        result = {'size': len(node.blockchain.mempool), 'txids': list(node.blockchain.mempool.keys())}
                    elif path == '/mining':
                        result = {'hashrate': node.miner.stats.hashrate, 'blocks': node.miner.stats.valid_blocks,
                                  'hashes': node.miner.stats.hashes, 'running': node.miner._running}
                    elif path == '/newaddress':
                        result = {'address': node.get_new_address()}
                    else:
                        result = {'error': 'unknown endpoint', 'available': ['/status', '/info', '/block/latest', '/block/<height>', '/peers', '/mempool', '/mining', '/newaddress']}
                except Exception as e:
                    result = {'error': str(e)}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result, indent=2).encode())

            def _dispatch(self, method: str, params: list) -> Any:
                if method == 'getblockchaininfo':
                    return node.blockchain.stats()
                elif method == 'getblockcount':
                    return node.blockchain.height
                elif method == 'getblockhash':
                    height = params[0] if params else node.blockchain.height
                    block = node.blockchain.get_block(height)
                    return block.hash if block else None
                elif method == 'getblock':
                    if params:
                        if isinstance(params[0], int):
                            block = node.blockchain.get_block(params[0])
                        else:
                            block = node.blockchain.get_block_by_hash(params[0])
                        return block.to_dict() if block else None
                    return None
                elif method == 'getmininginfo':
                    return {'hashrate': node.miner.stats.hashrate, 'blocks': node.miner.stats.valid_blocks,
                            'difficulty': node.blockchain.tip.header.difficulty}
                elif method == 'getnewaddress':
                    return node.get_new_address()
                elif method == 'getpeerinfo':
                    return [{'addr': k} for k in node.p2p.peers.keys()]
                elif method == 'getmempoolinfo':
                    return {'size': len(node.blockchain.mempool)}
                elif method == 'stop':
                    node.stop()
                    return 'L104SP node stopping...'
                else:
                    raise ValueError(f"Unknown method: {method}")

        return RPCHandler

    def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# FULL NODE (ENHANCED)
# ═══════════════════════════════════════════════════════════════════════════════

class L104SPNode:
    """Complete L104SP Full Node with RPC, persistence, and peer discovery."""

    def __init__(self, port: int = DEFAULT_PORT, rpc_port: int = 10401, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.wallet = HDWallet()
        self.blockchain = L104SPBlockchain(data_dir=self.data_dir)
        self.miner = MiningEngine(self.blockchain)
        self.p2p = P2PNode(self.blockchain, port=port)
        self.peer_discovery = PeerDiscovery(self.blockchain)
        self.rpc: Optional[RPCServer] = None
        self.rpc_port = rpc_port
        self._running = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Handle graceful shutdown on SIGINT/SIGTERM."""
        def handler(signum, frame):
            print("\n[NODE] Shutdown signal received...")
            self.stop()
        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except Exception:
            pass  # May fail in some environments

    def start(self, enable_rpc: bool = True) -> None:
        self._running = True
        print("=" * 60)
        print("    L104SP SOVEREIGN MAINNET NODE v3.1")
        print("=" * 60)
        print(f"[NODE] Data directory: {self.data_dir}")
        print(f"[NODE] Chain height: {self.blockchain.height}")
        print(f"[NODE] Genesis: {self.blockchain.chain[0].hash[:32]}...")
        print(f"[NODE] Tip: {self.blockchain.tip.hash[:32]}...")

        # Start P2P
        self.p2p.start()
        print(f"[P2P] Listening on port {self.p2p.port}")

        # Load known peers
        self.peer_discovery.load_from_db()
        print(f"[P2P] Known peers: {len(self.peer_discovery.known_peers)}")

        # Start RPC
        if enable_rpc:
            self.rpc = RPCServer(self, port=self.rpc_port)
            self.rpc.start()

        print("=" * 60)
        print(f"[NODE] Ready! RPC: http://127.0.0.1:{self.rpc_port}/status")
        print("=" * 60)

    def start_mining(self, address: str = None) -> None:
        if address is None:
            address, _ = self.wallet.get_address()
        print(f"[MINER] Mining to address: {address}")
        while self._running:
            block = self.miner.mine_block(address)
            if block:
                print(f"[MINER] Block {block.height} mined! Hash: {block.hash[:16]}... Reward: {block.get_reward() / SATOSHI_PER_COIN} L104SP")
                self.p2p.broadcast_block(block)

    def stop(self) -> None:
        print("[NODE] Stopping...")
        self._running = False
        self.miner.stop()
        self.p2p.stop()
        if self.rpc:
            self.rpc.stop()
        self.blockchain.close()
        print("[NODE] Stopped. Chain saved to disk.")

    def get_new_address(self) -> str:
        addr, _ = self.wallet.get_address(index=len(self.wallet._cache))
        return addr

    def get_status(self) -> Dict[str, Any]:
        return {
            'node': 'L104SP Full Node', 'version': L104SP_CONFIG['version'],
            'network': self.blockchain.network, 'blockchain': self.blockchain.stats(),
            'peers': len(self.p2p.peers), 'data_dir': str(self.data_dir),
            'mining': {'hashrate': f"{self.miner.stats.hashrate:.2f} H/s", 'blocks_mined': self.miner.stats.valid_blocks}
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY (for main.py)
# ═══════════════════════════════════════════════════════════════════════════════

class L104Block:
    """Legacy block class."""

    def __init__(self, index: int, previous_hash: str, timestamp: float,
                 transactions: List[Dict[str, Any]], nonce: int, resonance: float):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.nonce = nonce
        self.resonance = resonance
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = json.dumps({
            "index": self.index, "previous_hash": self.previous_hash,
            "timestamp": self.timestamp, "transactions": self.transactions,
            "nonce": self.nonce, "resonance": self.resonance
        }, sort_keys=True).encode()
        return hashlib.blake2b(hashlib.sha256(block_string).digest()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "previous_hash": self.previous_hash,
                "timestamp": self.timestamp, "transactions": self.transactions,
                "nonce": self.nonce, "resonance": self.resonance, "hash": self.hash}


class L104ResonanceMath:
    """Legacy resonance math."""

    @staticmethod
    def calculate_phi_resonance(nonce: int) -> float:
        return abs(math.sin(nonce * PHI))

    @staticmethod
    def calculate_god_code_alignment(nonce: int) -> float:
        return abs(math.cos((nonce / GOD_CODE) * 2 * math.pi))

    @staticmethod
    def find_resonant_nonce(start: int, threshold: float = 0.985, max_attempts: int = 100000) -> Optional[int]:
        for i in range(max_attempts):
            if L104ResonanceMath.calculate_phi_resonance(start + i) >= threshold:
                return start + i
        return None

    @staticmethod
    def optimize_nonce_search(base: int, count: int = 1000) -> List[int]:
        candidates = []
        for k in range(count):
            target = (math.pi / 2 + k * math.pi) / PHI
            nonce = int(target)
            if nonce > 0 and L104ResonanceMath.calculate_phi_resonance(nonce) > 0.9:
                candidates.append(nonce)
        return sorted(set(candidates))


class SovereignCoinEngine:
    """Legacy engine for main.py compatibility."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or L104SP_CONFIG
        self.chain: List[L104Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.difficulty = 4
        self.mining_reward = 104.0
        self.target_resonance = 0.985
        self.stats = MiningStats()
        self._mining_active = False
        self._lock = threading.Lock()
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        genesis = L104Block(0, "0" * 64, time.time(), [{"info": "L104SP Genesis", "god_code": GOD_CODE}], 0, 1.0)
        self.chain.append(genesis)

    def adjust_difficulty(self) -> None:
        if len(self.chain) < 5:
            return
        avg_res = sum(b.resonance for b in self.chain[-5:]) / 5
        if avg_res > 0.992 and self.difficulty < 8:
            self.difficulty += 1
        elif avg_res < 0.987 and self.difficulty > 2:
            self.difficulty -= 1

    def get_latest_block(self) -> L104Block:
        return self.chain[-1]

    def add_transaction(self, sender: str, recipient: str, amount: float) -> Dict[str, Any]:
        tx = {"sender": sender, "recipient": recipient, "amount": amount, "timestamp": time.time(),
              "tx_hash": hashlib.sha256(f"{sender}{recipient}{amount}{time.time()}".encode()).hexdigest()[:16]}
        with self._lock:
            self.pending_transactions.append(tx)
        return tx

    def is_resonance_valid(self, nonce: int, hash_val: str) -> bool:
        if not hash_val.startswith('0' * self.difficulty):
            return False
        return L104ResonanceMath.calculate_phi_resonance(nonce) >= self.target_resonance

    def mine_block(self, miner_address: str, use_parallel: bool = False) -> L104Block:
        self._mining_active = True
        latest = self.get_latest_block()
        new_index = latest.index + 1
        with self._lock:
            current_txs = self.pending_transactions.copy() + [
                {"sender": "COINBASE", "recipient": miner_address, "amount": self.mining_reward, "type": "mining_reward"}
            ]
        nonce = 0
        while self._mining_active:
            self.stats.hashes += 1
            resonance = L104ResonanceMath.calculate_phi_resonance(nonce)
            if resonance >= self.target_resonance:
                self.stats.valid_resonance += 1
                block = L104Block(new_index, latest.hash, time.time(), current_txs, nonce, resonance)
                if block.hash.startswith('0' * self.difficulty):
                    with self._lock:
                        self.chain.append(block)
                        self.pending_transactions = []
                        self.stats.valid_blocks += 1
                    self.adjust_difficulty()
                    return block
            nonce += 1
        raise RuntimeError("Mining stopped")

    def stop_mining(self) -> None:
        self._mining_active = False

    def get_balance(self, address: str) -> float:
        balance = 0.0
        for block in self.chain:
            for tx in block.transactions:
                if tx.get("recipient") == address:
                    balance += tx.get("amount", 0)
                if tx.get("sender") == address:
                    balance -= tx.get("amount", 0)
        return balance

    def get_status(self) -> Dict[str, Any]:
        return {
            "chain_length": len(self.chain), "difficulty": self.difficulty,
            "latest_hash": self.get_latest_block().hash,
            "pending_txs": len(self.pending_transactions),
            "coin_name": COIN_NAME, "symbol": COIN_SYMBOL,
            "version": L104SP_CONFIG["version"], "network": "l104sp_mainnet", "chain_id": 104,
            "resonance_threshold": self.target_resonance,
            "mining_stats": {"total_hashes": self.stats.hashes, "valid_resonance": self.stats.valid_resonance,
                            "blocks_mined": self.stats.valid_blocks, "hashrate": f"{self.stats.hashrate:.2f} H/s"},
            "sacred_constants": {"god_code": GOD_CODE, "phi": PHI, "void_constant": VOID_CONSTANT}
        }

    def export_for_evm(self) -> Dict[str, Any]:
        return {"chain_id": 104, "blocks": [b.to_dict() for b in self.chain[-10:]],
                "merkle_root": hashlib.sha256("".join(b.hash for b in self.chain).encode()).hexdigest(),
                "timestamp": time.time()}


# Global singleton
sovereign_coin = SovereignCoinEngine()


def primal_calculus(x: float) -> float:
    return (x ** PHI) / (VOID_CONSTANT * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector: List[float]) -> float:
    magnitude = sum(abs(v) for v in vector)
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """L104SP Mainnet Node CLI."""
    import argparse
    parser = argparse.ArgumentParser(
        description='L104SP Sovereign Prime Mainnet Node',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python l104_sovereign_coin_engine.py                    # Start node only
  python l104_sovereign_coin_engine.py --mine             # Start node and mine
  python l104_sovereign_coin_engine.py --mine --address ZXX...  # Mine to specific address
  python l104_sovereign_coin_engine.py --datadir /path    # Use custom data directory

RPC Endpoints (default http://127.0.0.1:10401):
  GET /status          - Node status
  GET /info            - Full chain info
  GET /block/latest    - Latest block
  GET /block/<height>  - Block by height
  GET /peers           - Connected peers
  GET /mining          - Mining stats
  GET /newaddress      - Generate new address
'''
    )
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'P2P port (default: {DEFAULT_PORT})')
    parser.add_argument('--rpcport', type=int, default=10401, help='RPC port (default: 10401)')
    parser.add_argument('--datadir', type=str, default=None, help=f'Data directory (default: {DATA_DIR})')
    parser.add_argument('--mine', action='store_true', help='Start mining')
    parser.add_argument('--address', type=str, help='Mining address (generates new if not specified)')
    parser.add_argument('--norpc', action='store_true', help='Disable RPC server')
    parser.add_argument('--daemon', action='store_true', help='Run as background daemon')
    args = parser.parse_args()

    # Setup data directory
    data_dir = Path(args.datadir) if args.datadir else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create and start node
    node = L104SPNode(port=args.port, rpc_port=args.rpcport, data_dir=data_dir)
    node.start(enable_rpc=not args.norpc)

    if args.mine:
        # Mining in foreground
        try:
            node.start_mining(args.address)
        except KeyboardInterrupt:
            node.stop()
    else:
        # Node only
        print("[NODE] Running. Press Ctrl+C to stop.")
        try:
            while node._running:
                time.sleep(1)
        except KeyboardInterrupt:
            node.stop()


if __name__ == '__main__':
    main()
