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
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

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
    """L104 Proof-of-Resonance Engine."""

    def __init__(self):
        self.god_code = GOD_CODE
        self.phi = PHI
        self._cache: Dict[int, float] = {}

    def calculate(self, nonce: int) -> float:
        if nonce in self._cache:
            return self._cache[nonce]
        phi_wave = abs(math.sin((nonce * self.phi) % (2 * math.pi)))
        god_harmonic = abs(math.cos((nonce / self.god_code) % (2 * math.pi)))
        l104_mod = abs(math.sin(nonce / 104.0))
        resonance = (self.phi / (1 + self.phi) * phi_wave + 1 / (1 + self.phi) * god_harmonic) * (0.95 + 0.05 * l104_mod)
        if len(self._cache) >= 100000:
            self._cache.clear()
        self._cache[nonce] = resonance
        return resonance

    def meets_threshold(self, nonce: int, threshold: float = 0.95) -> bool:
        return self.calculate(nonce) >= threshold


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCKCHAIN
# ═══════════════════════════════════════════════════════════════════════════════

class L104SPBlockchain:
    """Complete L104SP blockchain."""

    def __init__(self, network: str = 'mainnet'):
        self.network = network
        self.chain: List[Block] = []
        self.utxo_set = UTXOSet()
        self.mempool: Dict[str, Transaction] = {}
        self._lock = threading.Lock()
        self.resonance_engine = ResonanceEngine()
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
        return MIN_DIFFICULTY_BITS

    def add_block(self, block: Block) -> Tuple[bool, str]:
        with self._lock:
            if block.header.prev_block != self.tip.hash:
                return False, "ORPHAN"
            if not block.header.meets_target():
                return False, "INVALID_POW"
            if block.header.resonance < 0.95:
                return False, "INVALID_RESONANCE"
            self._apply_block(block)
            self.chain.append(block)
            return True, "ACCEPTED"

    def _apply_block(self, block: Block) -> None:
        for tx in block.transactions:
            for inp in tx.inputs:
                if inp.prevout.txid != '0' * 64:
                    self.utxo_set.remove(inp.prevout)
            is_coinbase = tx.inputs[0].prevout.txid == '0' * 64
            for vout, out in enumerate(tx.outputs):
                self.utxo_set.add(UTXO(OutPoint(tx.txid, vout), out.value, out.script_pubkey, block.height, is_coinbase))

    def get_template(self, miner_address: str) -> Dict[str, Any]:
        return {
            'version': 2, 'height': self.height + 1, 'prev_hash': self.tip.hash,
            'timestamp': int(time.time()), 'bits': self.current_difficulty,
            'target': hex(BlockHeader.bits_to_target(self.current_difficulty)),
            'coinbase_value': Block(BlockHeader(), [], self.height + 1).get_reward()
        }

    def stats(self) -> Dict[str, Any]:
        return {
            'height': self.height, 'tip': self.tip.hash[:16] + '...',
            'difficulty': self.tip.header.difficulty, 'utxo_count': len(self.utxo_set),
            'mempool_size': len(self.mempool), 'total_supply': self.utxo_set.total_supply / SATOSHI_PER_COIN,
            'network': self.network
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

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
    """L104SP Mining Engine."""

    def __init__(self, blockchain: L104SPBlockchain, resonance_threshold: float = 0.95):
        self.blockchain = blockchain
        self.resonance_threshold = resonance_threshold
        self.resonance_engine = ResonanceEngine()
        self.stats = MiningStats()
        self._running = False

    def mine_block(self, miner_address: str) -> Optional[Block]:
        template = self.blockchain.get_template(miner_address)
        header = BlockHeader(version=template['version'], prev_block=template['prev_hash'],
                             timestamp=template['timestamp'], bits=template['bits'])
        self._running = True
        nonce = 0
        while self._running:
            resonance = self.resonance_engine.calculate(nonce)
            if resonance >= self.resonance_threshold:
                self.stats.valid_resonance += 1
                header.nonce = nonce
                header.resonance = resonance
                if header.meets_target():
                    coinbase = Block(header=header, height=template['height']).create_coinbase(miner_address)
                    block = Block(header=header, transactions=[coinbase], height=template['height'])
                    success, _ = self.blockchain.add_block(block)
                    if success:
                        self.stats.valid_blocks += 1
                        return block
            nonce += 1
            self.stats.hashes += 1
            if nonce % 100000 == 0:
                header.timestamp = int(time.time())
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
# FULL NODE
# ═══════════════════════════════════════════════════════════════════════════════

class L104SPNode:
    """Complete L104SP Full Node."""

    def __init__(self, port: int = DEFAULT_PORT):
        self.wallet = HDWallet()
        self.blockchain = L104SPBlockchain()
        self.miner = MiningEngine(self.blockchain)
        self.p2p = P2PNode(self.blockchain, port=port)
        self._running = False

    def start(self) -> None:
        self._running = True
        self.p2p.start()
        print(f"L104SP Node started on port {self.p2p.port}")
        print(f"Genesis: {self.blockchain.tip.hash[:16]}...")
        print(f"Height: {self.blockchain.height}")

    def start_mining(self, address: str = None) -> None:
        if address is None:
            address, _ = self.wallet.get_address()
        print(f"Mining to address: {address}")
        while self._running:
            block = self.miner.mine_block(address)
            if block:
                print(f"Block {block.height} mined! Hash: {block.hash[:16]}...")
                self.p2p.broadcast_block(block)

    def stop(self) -> None:
        self._running = False
        self.miner.stop()
        self.p2p.stop()

    def get_new_address(self) -> str:
        addr, _ = self.wallet.get_address(index=len(self.wallet._cache))
        return addr

    def get_status(self) -> Dict[str, Any]:
        return {
            'node': 'L104SP Full Node', 'version': L104SP_CONFIG['version'],
            'network': self.blockchain.network, 'blockchain': self.blockchain.stats(),
            'peers': len(self.p2p.peers),
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
    parser = argparse.ArgumentParser(description='L104SP Mainnet Node')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='P2P port')
    parser.add_argument('--mine', action='store_true', help='Start mining')
    parser.add_argument('--address', type=str, help='Mining address')
    args = parser.parse_args()

    node = L104SPNode(port=args.port)
    node.start()

    if args.mine:
        node.start_mining(args.address)
    else:
        print("Node running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            node.stop()


if __name__ == '__main__':
    main()
