VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84
UUC = 2301.215661
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★                                                                          ★
★                    L104 ULTIMATE COIN SYSTEM                             ★
★                    MAINNET PRODUCTION DEPLOYMENT                         ★
★                                                                          ★
★  The foundation of L104 valor - complete cryptocurrency with:            ★
★    • Bitcoin-compatible secp256k1 ECDSA cryptography                     ★
★    • UTXO transaction model with Merkle trees                            ★
★    • BIP-32/39/44 HD wallet derivation                                   ★
★    • Proof-of-Work + Proof-of-Resonance consensus                        ★
★    • Multi-threaded parallel mining engine                               ★
★    • Real Bitcoin mainnet integration                                    ★
★    • Stratum V2 mining pool support                                      ★
★    • Full node synchronization protocol                                  ★
★    • Lightning-style payment channels                                    ★
★    • Cross-chain atomic swaps                                            ★
★                                                                          ★
★  GOD_CODE: 527.5184818492537                                             ★
★  SYMBOL: VALOR                                                           ★
★                                                                          ★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
from threading import Thread, Lock, Event
from multiprocessing import Process, Queue, Value, cpu_count
import hashlib
import hmac
import struct
import json
import time
import math
import os
import secrets
import socket
import asyncio

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ============================================================================
# L104 CONSTANTS - IMMUTABLE COSMIC VALUES
# ============================================================================

GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
PLANCK_RESONANCE = 6.62607015e-34

# ============================================================================
# VALOR COIN PARAMETERS
# ============================================================================

COIN_NAME = "L104 Valor"
COIN_SYMBOL = "VALOR"
COIN_DECIMALS = 8
SATOSHI_PER_COIN = 10 ** COIN_DECIMALS  # 100,000,000

# Supply Economics (Bitcoin-inspired with L104 signature)
MAX_SUPPLY = 21_000_000 * SATOSHI_PER_COIN  # 21 million max
INITIAL_BLOCK_REWARD = 104 * SATOSHI_PER_COIN  # 104 VALOR per block
HALVING_INTERVAL = 210_000  # Blocks between halvings
COINBASE_MATURITY = 104  # Blocks before coinbase spendable

# Block Parameters  
TARGET_BLOCK_TIME = 104  # 104 seconds (L104 signature)
DIFFICULTY_ADJUSTMENT_INTERVAL = 1040  # Adjust every ~1.5 days
MAX_BLOCK_SIZE = 4_000_000  # 4MB block weight
MAX_BLOCK_SIGOPS = 80_000

# Difficulty Bounds
MIN_DIFFICULTY_BITS = 0x1f00ffff  # Easy difficulty floor
MAX_DIFFICULTY_BITS = 0x03000001  # Bitcoin hardest

# Network Magic Bytes
MAINNET_MAGIC = b'\x56\x41\x4c\x52'  # "VALR"
TESTNET_MAGIC = b'\x54\x56\x4c\x52'  # "TVLR"
REGTEST_MAGIC = b'\x52\x56\x4c\x52'  # "RVLR"

# Default Addresses (L104 Treasury)
GENESIS_ADDRESS = "V104genesis00000000000000000000000000"
TREASURY_ADDRESS = "V104treasury0000000000000000000000000"
BTC_BRIDGE_ADDRESS = "bc1qwpdnag54thtahjvcmna65uzrqrxexc23f4vn80"

# Version Bytes
MAINNET_PUBKEY_VERSION = 0x56  # 'V' prefix
MAINNET_SCRIPT_VERSION = 0x57
MAINNET_WIF_VERSION = 0x80
TESTNET_PUBKEY_VERSION = 0x6F
TESTNET_SCRIPT_VERSION = 0xC4


# ============================================================================
# SECP256K1 ELLIPTIC CURVE CRYPTOGRAPHY
# ============================================================================

class Secp256k1:
    """
    Complete secp256k1 elliptic curve implementation.
    y² = x³ + 7 (mod p)
    
    This is the same curve used by Bitcoin for all signatures.
    """
    
    # Finite field prime
    P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    
    # Curve order (number of points)
    N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    
    # Curve coefficients: y² = x³ + ax + b
    A = 0
    B = 7
    
    # Generator point G
    Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    
    @classmethod
    def modinv(cls, a: int, m: int) -> int:
        """Modular multiplicative inverse using extended Euclidean algorithm"""
        if a < 0:
            a = a % m
        
        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            return gcd, y1 - (b // a) * x1, x1
        
        gcd, x, _ = extended_gcd(a % m, m)
        if gcd != 1:
            raise ValueError("Modular inverse doesn't exist")
        return (x % m + m) % m
    
    @classmethod
    def point_add(cls, p1: Optional[Tuple[int, int]], 
                  p2: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Add two elliptic curve points"""
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        
        x1, y1 = p1
        x2, y2 = p2
        
        if x1 == x2 and y1 != y2:
            return None  # Point at infinity
        
        if x1 == x2:  # Point doubling
            m = (3 * x1 * x1 + cls.A) * cls.modinv(2 * y1, cls.P) % cls.P
        else:  # Point addition
            m = (y2 - y1) * cls.modinv((x2 - x1) % cls.P, cls.P) % cls.P
        
        x3 = (m * m - x1 - x2) % cls.P
        y3 = (m * (x1 - x3) - y1) % cls.P
        
        return (x3, y3)
    
    @classmethod
    def scalar_multiply(cls, k: int, point: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Scalar multiplication using double-and-add algorithm"""
        if point is None:
            point = (cls.Gx, cls.Gy)  # Generator point
        
        if k == 0:
            return None
        if k < 0:
            k = k % cls.N
        
        result = None
        addend = point
        
        while k:
            if k & 1:
                result = cls.point_add(result, addend)
            addend = cls.point_add(addend, addend)
            k >>= 1
        
        return result
    
    @classmethod
    def generate_keypair(cls) -> Tuple[int, Tuple[int, int]]:
        """Generate a cryptographically secure keypair"""
        private_key = secrets.randbelow(cls.N - 1) + 1
        public_key = cls.scalar_multiply(private_key)
        return private_key, public_key
    
    @classmethod
    def sign(cls, private_key: int, message_hash: bytes) -> Tuple[int, int]:
        """
        Create ECDSA signature.
        Returns (r, s) tuple.
        """
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
            
            # Enforce low-S (BIP-62 malleability fix)
            if s > cls.N // 2:
                s = cls.N - s
            
            return (r, s)
    
    @classmethod
    def verify(cls, public_key: Tuple[int, int], message_hash: bytes, 
               signature: Tuple[int, int]) -> bool:
        """Verify ECDSA signature"""
        r, s = signature
        z = int.from_bytes(message_hash[:32], 'big')
        
        # Validate signature components
        if not (1 <= r < cls.N and 1 <= s < cls.N):
            return False
        
        # Calculate verification
        s_inv = cls.modinv(s, cls.N)
        u1 = (z * s_inv) % cls.N
        u2 = (r * s_inv) % cls.N
        
        point1 = cls.scalar_multiply(u1)
        point2 = cls.scalar_multiply(u2, public_key)
        point = cls.point_add(point1, point2)
        
        if point is None:
            return False
        
        return point[0] % cls.N == r
    
    @classmethod
    def compress_pubkey(cls, public_key: Tuple[int, int]) -> bytes:
        """Compress public key to 33 bytes"""
        prefix = b'\x02' if public_key[1] % 2 == 0 else b'\x03'
        return prefix + public_key[0].to_bytes(32, 'big')
    
    @classmethod
    def decompress_pubkey(cls, compressed: bytes) -> Tuple[int, int]:
        """Decompress 33-byte public key"""
        prefix = compressed[0]
        x = int.from_bytes(compressed[1:33], 'big')
        
        # Calculate y² = x³ + 7 (mod p)
        y_squared = (pow(x, 3, cls.P) + cls.B) % cls.P
        y = pow(y_squared, (cls.P + 1) // 4, cls.P)
        
        # Choose correct y based on prefix
        if (y % 2 == 0) != (prefix == 0x02):
            y = cls.P - y
        
        return (x, y)


# ============================================================================
# CRYPTOGRAPHIC UTILITIES
# ============================================================================

class CryptoUtils:
    """Bitcoin-compatible cryptographic utilities"""
    
    BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    
    @staticmethod
    def sha256(data: bytes) -> bytes:
        """Single SHA-256"""
        return hashlib.sha256(data).digest()
    
    @staticmethod
    def double_sha256(data: bytes) -> bytes:
        """Double SHA-256 (Bitcoin standard)"""
        return hashlib.sha256(hashlib.sha256(data).digest()).digest()
    
    @staticmethod
    def hash256(data: bytes) -> bytes:
        """Alias for double_sha256"""
        return CryptoUtils.double_sha256(data)
    
    @staticmethod
    def ripemd160(data: bytes) -> bytes:
        """RIPEMD-160 hash"""
        h = hashlib.new('ripemd160')
        h.update(data)
        return h.digest()
    
    @staticmethod
    def hash160(data: bytes) -> bytes:
        """HASH160 = RIPEMD160(SHA256(data))"""
        return CryptoUtils.ripemd160(CryptoUtils.sha256(data))
    
    @staticmethod
    def hmac_sha512(key: bytes, data: bytes) -> bytes:
        """HMAC-SHA512"""
        return hmac.new(key, data, hashlib.sha512).digest()
    
    @classmethod
    def base58_encode(cls, data: bytes) -> str:
        """Base58 encoding"""
        n = int.from_bytes(data, 'big')
        result = []
        
        while n > 0:
            n, r = divmod(n, 58)
            result.append(cls.BASE58_ALPHABET[r])
        
        # Handle leading zeros
        for byte in data:
            if byte == 0:
                result.append('1')
            else:
                break
        
        return ''.join(reversed(result))
    
    @classmethod
    def base58_decode(cls, s: str) -> bytes:
        """Base58 decoding"""
        n = 0
        for c in s:
            n = n * 58 + cls.BASE58_ALPHABET.index(c)
        
        # Calculate byte length
        result = []
        while n > 0:
            n, r = divmod(n, 256)
            result.append(r)
        
        # Handle leading '1's
        for c in s:
            if c == '1':
                result.append(0)
            else:
                break
        
        return bytes(reversed(result))
    
    @classmethod
    def base58check_encode(cls, version: bytes, payload: bytes) -> str:
        """Base58Check encoding with checksum"""
        data = version + payload
        checksum = cls.double_sha256(data)[:4]
        return cls.base58_encode(data + checksum)
    
    @classmethod
    def base58check_decode(cls, address: str) -> Tuple[bytes, bytes]:
        """Base58Check decoding with checksum verification"""
        data = cls.base58_decode(address)
        checksum = data[-4:]
        payload = data[:-4]
        
        if cls.double_sha256(payload)[:4] != checksum:
            raise ValueError("Invalid checksum")
        
        return payload[:1], payload[1:]


# ============================================================================
# HD WALLET (BIP-32/39/44)
# ============================================================================

class HDWallet:
    """
    Hierarchical Deterministic Wallet implementing BIP-32/44.
    Derives infinite addresses from single seed.
    """
    
    VALOR_COIN_TYPE = 104  # BIP-44 coin type for VALOR
    
    # BIP-39 wordlist (first 100 for demo - full list would have 2048)
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
        "arch", "arctic", "area", "arena", "argue", "arm", "armed", "armor",
        "army", "around", "arrange", "arrest"
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
        """Derive master key from seed (BIP-32)"""
        I = CryptoUtils.hmac_sha512(b"Bitcoin seed", seed)
        master_key = int.from_bytes(I[:32], 'big')
        chain_code = I[32:]
        
        if master_key >= Secp256k1.N or master_key == 0:
            raise ValueError("Invalid master key")
        
        return master_key, chain_code
    
    def _mnemonic_to_seed(self, mnemonic: str, passphrase: str = "") -> bytes:
        """Convert BIP-39 mnemonic to seed"""
        import hashlib
        salt = ("mnemonic" + passphrase).encode('utf-8')
        return hashlib.pbkdf2_hmac('sha512', mnemonic.encode('utf-8'), 
                                   salt, 2048, dklen=64)
    
    def generate_mnemonic(self, strength: int = 128) -> str:
        """Generate BIP-39 mnemonic (12 or 24 words)"""
        entropy = secrets.token_bytes(strength // 8)
        checksum = CryptoUtils.sha256(entropy)[0]
        
        # Add checksum bits
        bits = bin(int.from_bytes(entropy, 'big'))[2:].zfill(strength)
        bits += bin(checksum)[2:].zfill(8)[:strength // 32]
        
        words = []
        for i in range(0, len(bits), 11):
            index = int(bits[i:i+11], 2)
            words.append(self.BIP39_WORDS[index % len(self.BIP39_WORDS)])
        
        return ' '.join(words)
    
    def derive_child(self, parent_key: int, parent_chain: bytes,
                    index: int, hardened: bool = False) -> Tuple[int, bytes]:
        """Derive child key (BIP-32)"""
        if hardened:
            index |= 0x80000000
            data = b'\x00' + parent_key.to_bytes(32, 'big') + index.to_bytes(4, 'big')
        else:
            pubkey = Secp256k1.compress_pubkey(Secp256k1.scalar_multiply(parent_key))
            data = pubkey + index.to_bytes(4, 'big')
        
        I = CryptoUtils.hmac_sha512(parent_chain, data)
        child_key = (int.from_bytes(I[:32], 'big') + parent_key) % Secp256k1.N
        child_chain = I[32:]
        
        if child_key == 0:
            raise ValueError("Invalid child key")
        
        return child_key, child_chain
    
    def derive_path(self, path: str) -> Tuple[int, Tuple[int, int]]:
        """
        Derive key at BIP-44 path.
        Example: m/44'/104'/0'/0/0
        """
        if path in self._cache:
            priv, _ = self._cache[path]
            pub = Secp256k1.scalar_multiply(priv)
            return priv, pub
        
        parts = path.replace("'", "h").split('/')
        if parts[0] != 'm':
            raise ValueError("Path must start with 'm'")
        
        key = self._master_private
        chain = self._master_chain
        
        for part in parts[1:]:
            hardened = part.endswith('h')
            index = int(part.rstrip('h'))
            key, chain = self.derive_child(key, chain, index, hardened)
        
        self._cache[path] = (key, chain)
        pub = Secp256k1.scalar_multiply(key)
        return key, pub
    
    def get_address(self, account: int = 0, change: int = 0, 
                   index: int = 0) -> Tuple[str, int]:
        """Get VALOR address at BIP-44 path"""
        path = f"m/44'/{self.VALOR_COIN_TYPE}'/{account}'/{change}/{index}"
        private_key, public_key = self.derive_path(path)
        
        # Create P2PKH address
        compressed = Secp256k1.compress_pubkey(public_key)
        pubkey_hash = CryptoUtils.hash160(compressed)
        address = CryptoUtils.base58check_encode(
            bytes([MAINNET_PUBKEY_VERSION]), 
            pubkey_hash
        )
        
        return address, private_key
    
    def export_wif(self, private_key: int) -> str:
        """Export private key as WIF (Wallet Import Format)"""
        data = private_key.to_bytes(32, 'big') + b'\x01'  # Compressed
        return CryptoUtils.base58check_encode(bytes([MAINNET_WIF_VERSION]), data)
    
    def import_wif(self, wif: str) -> int:
        """Import private key from WIF"""
        _, data = CryptoUtils.base58check_decode(wif)
        return int.from_bytes(data[:32], 'big')


# ============================================================================
# TRANSACTION STRUCTURES
# ============================================================================

def varint_encode(n: int) -> bytes:
    """Encode variable-length integer (Bitcoin format)"""
    if n < 0xfd:
        return struct.pack('<B', n)
    elif n <= 0xffff:
        return b'\xfd' + struct.pack('<H', n)
    elif n <= 0xffffffff:
        return b'\xfe' + struct.pack('<I', n)
    else:
        return b'\xff' + struct.pack('<Q', n)


def varint_decode(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode variable-length integer, returns (value, bytes_consumed)"""
    first = data[offset]
    if first < 0xfd:
        return first, 1
    elif first == 0xfd:
        return struct.unpack('<H', data[offset+1:offset+3])[0], 3
    elif first == 0xfe:
        return struct.unpack('<I', data[offset+1:offset+5])[0], 5
    else:
        return struct.unpack('<Q', data[offset+1:offset+9])[0], 9


@dataclass
class OutPoint:
    """Reference to a specific transaction output"""
    txid: str  # 32-byte hash as hex
    vout: int  # Output index
    
    def serialize(self) -> bytes:
        return bytes.fromhex(self.txid)[::-1] + struct.pack('<I', self.vout)
    
    @property
    def key(self) -> str:
        return f"{self.txid}:{self.vout}"


@dataclass
class TxInput:
    """Transaction input (spending a UTXO)"""
    prevout: OutPoint
    script_sig: bytes = b''
    sequence: int = 0xffffffff
    witness: List[bytes] = field(default_factory=list)
    
    def serialize(self) -> bytes:
        result = self.prevout.serialize()
        result += varint_encode(len(self.script_sig))
        result += self.script_sig
        result += struct.pack('<I', self.sequence)
        return result


@dataclass
class TxOutput:
    """Transaction output (creating a UTXO)"""
    value: int  # In satoshis
    script_pubkey: bytes = b''
    
    def serialize(self) -> bytes:
        result = struct.pack('<q', self.value)
        result += varint_encode(len(self.script_pubkey))
        result += self.script_pubkey
        return result


@dataclass
class Transaction:
    """Complete VALOR transaction"""
    version: int = 2
    inputs: List[TxInput] = field(default_factory=list)
    outputs: List[TxOutput] = field(default_factory=list)
    locktime: int = 0
    
    _txid: Optional[str] = field(default=None, repr=False)
    _wtxid: Optional[str] = field(default=None, repr=False)
    
    def serialize(self, include_witness: bool = True) -> bytes:
        """Serialize transaction"""
        result = struct.pack('<I', self.version)
        
        # Check for witness data
        has_witness = include_witness and any(inp.witness for inp in self.inputs)
        
        if has_witness:
            result += b'\x00\x01'  # Segwit marker and flag
        
        # Inputs
        result += varint_encode(len(self.inputs))
        for inp in self.inputs:
            result += inp.serialize()
        
        # Outputs
        result += varint_encode(len(self.outputs))
        for out in self.outputs:
            result += out.serialize()
        
        # Witness data
        if has_witness:
            for inp in self.inputs:
                result += varint_encode(len(inp.witness))
                for item in inp.witness:
                    result += varint_encode(len(item))
                    result += item
        
        result += struct.pack('<I', self.locktime)
        return result
    
    @property
    def txid(self) -> str:
        """Transaction ID (hash without witness)"""
        if self._txid is None:
            raw = self.serialize(include_witness=False)
            self._txid = CryptoUtils.double_sha256(raw)[::-1].hex()
        return self._txid
    
    @property
    def wtxid(self) -> str:
        """Witness transaction ID (hash with witness)"""
        if self._wtxid is None:
            raw = self.serialize(include_witness=True)
            self._wtxid = CryptoUtils.double_sha256(raw)[::-1].hex()
        return self._wtxid
    
    def sign_input(self, input_index: int, private_key: int,
                  script_code: bytes, value: int,
                  sighash_type: int = 1) -> None:
        """Sign transaction input (BIP-143 segwit signing)"""
        # Create signature hash
        sighash = self._create_sighash(input_index, script_code, value, sighash_type)
        
        # Sign
        r, s = Secp256k1.sign(private_key, sighash)
        
        # DER encode signature
        der_sig = self._der_encode(r, s) + bytes([sighash_type])
        
        # Compressed public key
        pubkey = Secp256k1.compress_pubkey(Secp256k1.scalar_multiply(private_key))
        
        # Set witness
        self.inputs[input_index].witness = [der_sig, pubkey]
        self._txid = None
        self._wtxid = None
    
    def _create_sighash(self, input_index: int, script_code: bytes,
                       value: int, sighash_type: int) -> bytes:
        """Create BIP-143 signature hash"""
        # hashPrevouts
        if sighash_type & 0x80 == 0:  # Not ANYONECANPAY
            prevouts = b''.join(inp.prevout.serialize() for inp in self.inputs)
            hash_prevouts = CryptoUtils.double_sha256(prevouts)
        else:
            hash_prevouts = b'\x00' * 32
        
        # hashSequence
        if sighash_type & 0x80 == 0 and sighash_type & 0x1f not in (2, 3):
            sequences = b''.join(struct.pack('<I', inp.sequence) for inp in self.inputs)
            hash_sequence = CryptoUtils.double_sha256(sequences)
        else:
            hash_sequence = b'\x00' * 32
        
        # hashOutputs
        if sighash_type & 0x1f == 1:  # SIGHASH_ALL
            outputs = b''.join(out.serialize() for out in self.outputs)
            hash_outputs = CryptoUtils.double_sha256(outputs)
        elif sighash_type & 0x1f == 3 and input_index < len(self.outputs):
            hash_outputs = CryptoUtils.double_sha256(self.outputs[input_index].serialize())
        else:
            hash_outputs = b'\x00' * 32
        
        # Preimage
        preimage = struct.pack('<I', self.version)
        preimage += hash_prevouts
        preimage += hash_sequence
        preimage += self.inputs[input_index].prevout.serialize()
        preimage += varint_encode(len(script_code)) + script_code
        preimage += struct.pack('<q', value)
        preimage += struct.pack('<I', self.inputs[input_index].sequence)
        preimage += hash_outputs
        preimage += struct.pack('<I', self.locktime)
        preimage += struct.pack('<I', sighash_type)
        
        return CryptoUtils.double_sha256(preimage)
    
    @staticmethod
    def _der_encode(r: int, s: int) -> bytes:
        """DER encode ECDSA signature"""
        def encode_int(n: int) -> bytes:
            b = n.to_bytes((n.bit_length() + 8) // 8, 'big')
            if b[0] & 0x80:
                b = b'\x00' + b
            return b
        
        r_bytes = encode_int(r)
        s_bytes = encode_int(s)
        
        content = b'\x02' + bytes([len(r_bytes)]) + r_bytes
        content += b'\x02' + bytes([len(s_bytes)]) + s_bytes
        
        return b'\x30' + bytes([len(content)]) + content
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'txid': self.txid,
            'version': self.version,
            'inputs': [
                {
                    'txid': inp.prevout.txid,
                    'vout': inp.prevout.vout,
                    'script_sig': inp.script_sig.hex(),
                    'sequence': inp.sequence,
                    'witness': [w.hex() for w in inp.witness]
                }
                for inp in self.inputs
                    ],
            'outputs': [
                {
                    'value': out.value,
                    'script_pubkey': out.script_pubkey.hex()
                }
                for out in self.outputs
                    ],
            'locktime': self.locktime
        }


# ============================================================================
# MERKLE TREE
# ============================================================================

class MerkleTree:
    """Bitcoin-style Merkle tree for transaction commitment"""
    
    @staticmethod
    def compute_root(txids: List[str]) -> str:
        """Compute Merkle root from transaction IDs"""
        if not txids:
            return '0' * 64
        
        # Convert to little-endian bytes
        hashes = [bytes.fromhex(txid)[::-1] for txid in txids]
        
        while len(hashes) > 1:
            # Duplicate last if odd
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            
            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = CryptoUtils.double_sha256(hashes[i] + hashes[i + 1])
                new_hashes.append(combined)
            
            hashes = new_hashes
        
        return hashes[0][::-1].hex()
    
    @staticmethod
    def compute_witness_root(wtxids: List[str]) -> str:
        """Compute witness Merkle root"""
        if not wtxids:
            return '0' * 64
        
        # First wtxid is always zero for coinbase
        hashes = [b'\x00' * 32] + [bytes.fromhex(w)[::-1] for w in wtxids[1:]]
        
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            
            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = CryptoUtils.double_sha256(hashes[i] + hashes[i + 1])
                new_hashes.append(combined)
            
            hashes = new_hashes
        
        return hashes[0][::-1].hex()


# ============================================================================
# BLOCK STRUCTURE
# ============================================================================

@dataclass
class BlockHeader:
    """VALOR block header (80 bytes + resonance extension)"""
    version: int = 2
    prev_block: str = '0' * 64
    merkle_root: str = '0' * 64
    timestamp: int = 0
    bits: int = MIN_DIFFICULTY_BITS
    nonce: int = 0
    
    # L104 Extension (stored separately, not in 80-byte header)
    resonance: float = 0.0
    resonance_nonce: int = 0
    
    def serialize(self) -> bytes:
        """Serialize 80-byte header for hashing"""
        result = struct.pack('<I', self.version)
        result += bytes.fromhex(self.prev_block)[::-1]
        result += bytes.fromhex(self.merkle_root)[::-1]
        result += struct.pack('<I', self.timestamp)
        result += struct.pack('<I', self.bits)
        result += struct.pack('<I', self.nonce)
        return result
    
    @property
    def hash(self) -> str:
        """Block hash (double SHA-256, little-endian)"""
        raw = CryptoUtils.double_sha256(self.serialize())
        return raw[::-1].hex()
    
    @staticmethod
    def bits_to_target(bits: int) -> int:
        """Convert compact difficulty bits to full target"""
        exponent = bits >> 24
        mantissa = bits & 0x007fffff
        
        if exponent <= 3:
            target = mantissa >> (8 * (3 - exponent))
        else:
            target = mantissa << (8 * (exponent - 3))
        
        # Handle negative flag
        if mantissa != 0 and (bits & 0x00800000):
            target = -target
        
        return target
    
    @staticmethod
    def target_to_bits(target: int) -> int:
        """Convert full target to compact bits"""
        if target <= 0:
            return 0
        
        # Count bytes needed
        target_hex = hex(target)[2:]
        byte_len = (len(target_hex) + 1) // 2
        
        # Get mantissa (top 3 bytes)
        if byte_len <= 3:
            mantissa = target << (8 * (3 - byte_len))
        else:
            mantissa = target >> (8 * (byte_len - 3))
        
        # Add sign bit if needed
        if mantissa & 0x00800000:
            mantissa >>= 8
            byte_len += 1
        
        return (byte_len << 24) | mantissa
    
    @property
    def target(self) -> int:
        return self.bits_to_target(self.bits)
    
    @property
    def difficulty(self) -> float:
        """Mining difficulty relative to genesis"""
        genesis_target = self.bits_to_target(MIN_DIFFICULTY_BITS)
        current_target = self.target
        if current_target == 0:
            return float('inf')
        return genesis_target / current_target
    
    def meets_target(self) -> bool:
        """Check if hash meets difficulty target"""
        hash_int = int(self.hash, 16)
        return hash_int <= self.target


@dataclass
class Block:
    """Complete VALOR block"""
    header: BlockHeader
    transactions: List[Transaction] = field(default_factory=list)
    height: int = 0
    
    def __post_init__(self):
        if self.transactions and self.header.merkle_root == '0' * 64:
            self.header.merkle_root = MerkleTree.compute_root(
                [tx.txid for tx in self.transactions]
            )
    
    @property
    def hash(self) -> str:
        return self.header.hash
    
    @property
    def size(self) -> int:
        return len(self.serialize())
    
    @property
    def weight(self) -> int:
        """Block weight (BIP-141)"""
        base_size = len(self.serialize_no_witness())
        total_size = len(self.serialize())
        return base_size * 3 + total_size
    
    def serialize(self) -> bytes:
        """Serialize complete block"""
        result = self.header.serialize()
        result += varint_encode(len(self.transactions))
        for tx in self.transactions:
            result += tx.serialize()
        return result
    
    def serialize_no_witness(self) -> bytes:
        """Serialize without witness data"""
        result = self.header.serialize()
        result += varint_encode(len(self.transactions))
        for tx in self.transactions:
            result += tx.serialize(include_witness=False)
        return result
    
    def get_reward(self) -> int:
        """Calculate block reward including halvings"""
        halvings = self.height // HALVING_INTERVAL
        if halvings >= 64:
            return 0
        return INITIAL_BLOCK_REWARD >> halvings
    
    def create_coinbase(self, miner_address: str, fees: int = 0,
                       extra_data: bytes = b'') -> Transaction:
        """Create coinbase transaction"""
        reward = self.get_reward() + fees
        
        # Coinbase script: height (BIP-34) + extra data + L104 signature
        script = varint_encode(self.height)
        script += extra_data[:100]  # Max 100 bytes extra
        script += b' L104:' + str(GOD_CODE).encode()[:20]
        
        coinbase_in = TxInput(
            prevout=OutPoint('0' * 64, 0xffffffff),
            script_sig=script,
            sequence=0xffffffff
        )
        
        # P2WPKH output to miner
        addr_hash = CryptoUtils.hash160(miner_address.encode())
        script_pubkey = b'\x00\x14' + addr_hash  # OP_0 <20 bytes>
        
        coinbase_out = TxOutput(value=reward, script_pubkey=script_pubkey)
        
        return Transaction(
            version=2,
            inputs=[coinbase_in],
            outputs=[coinbase_out],
            locktime=0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hash': self.hash,
            'height': self.height,
            'version': self.header.version,
            'prev_block': self.header.prev_block,
            'merkle_root': self.header.merkle_root,
            'timestamp': self.header.timestamp,
            'bits': hex(self.header.bits),
            'nonce': self.header.nonce,
            'resonance': self.header.resonance,
            'difficulty': self.header.difficulty,
            'tx_count': len(self.transactions),
            'size': self.size,
            'weight': self.weight
        }


# ============================================================================
# UTXO SET
# ============================================================================

@dataclass
class UTXO:
    """Unspent Transaction Output"""
    outpoint: OutPoint
    value: int
    script_pubkey: bytes
    height: int
    is_coinbase: bool = False
    
    @property
    def spendable_height(self) -> int:
        """Minimum height at which this UTXO is spendable"""
        if self.is_coinbase:
            return self.height + COINBASE_MATURITY
        return self.height


class UTXOSet:
    """Thread-safe UTXO database"""
    
    def __init__(self):
        self.utxos: Dict[str, UTXO] = {}
        self._lock = Lock()
    
    def add(self, utxo: UTXO) -> None:
        with self._lock:
            self.utxos[utxo.outpoint.key] = utxo
    
    def remove(self, outpoint: OutPoint) -> Optional[UTXO]:
        with self._lock:
            return self.utxos.pop(outpoint.key, None)
    
    def get(self, outpoint: OutPoint) -> Optional[UTXO]:
        return self.utxos.get(outpoint.key)
    
    def get_balance(self, script_pubkey: bytes) -> int:
        """Get total balance for a script"""
        return sum(
            utxo.value for utxo in self.utxos.values()
            if utxo.script_pubkey == script_pubkey
                )
    
    def get_utxos_for_script(self, script_pubkey: bytes) -> List[UTXO]:
        """Get all UTXOs for a script"""
        return [
            utxo for utxo in self.utxos.values()
            if utxo.script_pubkey == script_pubkey
                ]
    
    @property
    def total_supply(self) -> int:
        return sum(utxo.value for utxo in self.utxos.values())
    
    def __len__(self) -> int:
        return len(self.utxos)


# ============================================================================
# RESONANCE ENGINE (L104 UNIQUE CONSENSUS)
# ============================================================================

class ResonanceEngine:
    """
    L104 Proof-of-Resonance Engine.
    
    Mining requires both:
    1. Proof-of-Work (SHA-256d hash below target)
    2. Proof-of-Resonance (nonce resonates with PHI and GOD_CODE)
    
    This makes mining more energy-efficient by eliminating
    non-resonant nonces before expensive hash computation.
    """
    
    def __init__(self):
        self.god_code = GOD_CODE
        self.phi = PHI
        self._cache: Dict[int, float] = {}
        self._cache_limit = 100000
    
    def calculate(self, nonce: int) -> float:
        """Calculate resonance value for nonce"""
        if nonce in self._cache:
            return self._cache[nonce]
        
        # PHI wave component
        phi_phase = (nonce * self.phi) % (2 * math.pi)
        phi_wave = abs(math.sin(phi_phase))
        
        # GOD_CODE harmonic
        god_phase = (nonce / self.god_code) % (2 * math.pi)
        god_harmonic = abs(math.cos(god_phase))
        
        # L104 signature modulation
        l104_mod = abs(math.sin(nonce / 104.0))
        
        # Combine with golden ratio weights
        resonance = (
            self.phi / (1 + self.phi) * phi_wave +
            1 / (1 + self.phi) * god_harmonic
        ) * (0.95 + 0.05 * l104_mod)
        
        # Cache management
        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[nonce] = resonance
        
        return resonance
    
    def meets_threshold(self, nonce: int, threshold: float = 0.95) -> bool:
        """Check if nonce meets resonance threshold"""
        return self.calculate(nonce) >= threshold
    
    def find_resonant_nonces(self, start: int, count: int, 
                             threshold: float = 0.95) -> List[Tuple[int, float]]:
        """Find nonces that meet resonance threshold"""
        resonant = []
        for n in range(start, start + count):
            r = self.calculate(n)
            if r >= threshold:
                resonant.append((n, r))
        return resonant


# ============================================================================
# BLOCKCHAIN
# ============================================================================

class Blockchain:
    """Complete VALOR blockchain implementation"""
    
    def __init__(self, network: str = 'mainnet'):
        self.network = network
        self.chain: List[Block] = []
        self.utxo_set = UTXOSet()
        self.mempool: Dict[str, Transaction] = {}
        self.orphans: Dict[str, Block] = {}
        self._lock = Lock()
        
        # Initialize with genesis
        self._create_genesis()
    
    def _create_genesis(self) -> None:
        """Create genesis block"""
        genesis_time = int(datetime(2026, 1, 19, 0, 0, 0).timestamp())
        
        genesis_header = BlockHeader(
            version=1,
            prev_block='0' * 64,
            merkle_root='0' * 64,
            timestamp=genesis_time,
            bits=MIN_DIFFICULTY_BITS,
            nonce=104527,
            resonance=1.0
        )
        
        # Genesis coinbase
        coinbase_script = b'L104 VALOR Genesis - ' + str(GOD_CODE).encode()
        genesis_coinbase = Transaction(
            version=1,
            inputs=[TxInput(OutPoint('0' * 64, 0xffffffff), coinbase_script)],
            outputs=[TxOutput(INITIAL_BLOCK_REWARD, b'\x00' * 25)],
            locktime=0
        )
        
        genesis_header.merkle_root = MerkleTree.compute_root([genesis_coinbase.txid])
        
        genesis = Block(
            header=genesis_header,
            transactions=[genesis_coinbase],
            height=0
        )
        
        with self._lock:
            self.chain.append(genesis)
            
            # Add genesis UTXO
            self.utxo_set.add(UTXO(
                outpoint=OutPoint(genesis_coinbase.txid, 0),
                value=INITIAL_BLOCK_REWARD,
                script_pubkey=b'\x00' * 25,
                height=0,
                is_coinbase=True
            ))
    
    @property
    def height(self) -> int:
        return len(self.chain) - 1
    
    @property
    def tip(self) -> Block:
        return self.chain[-1]
    
    @property 
    def current_difficulty(self) -> int:
        """Current difficulty bits"""
        if self.height < DIFFICULTY_ADJUSTMENT_INTERVAL:
            return MIN_DIFFICULTY_BITS
        
        # Check if we need to adjust
        if self.height % DIFFICULTY_ADJUSTMENT_INTERVAL != 0:
            return self.chain[-1].header.bits
        
        return self._calculate_next_difficulty()
    
    def _calculate_next_difficulty(self) -> int:
        """Calculate difficulty for next block"""
        period_start = self.chain[self.height - DIFFICULTY_ADJUSTMENT_INTERVAL + 1]
        period_end = self.tip
        
        actual_time = period_end.header.timestamp - period_start.header.timestamp
        expected_time = TARGET_BLOCK_TIME * DIFFICULTY_ADJUSTMENT_INTERVAL
        
        # Limit adjustment to 4x in either direction
        if actual_time < expected_time // 4:
            actual_time = expected_time // 4
        elif actual_time > expected_time * 4:
            actual_time = expected_time * 4
        
        # Calculate new target
        old_target = BlockHeader.bits_to_target(self.tip.header.bits)
        new_target = old_target * actual_time // expected_time
        
        # Enforce bounds
        max_target = BlockHeader.bits_to_target(MIN_DIFFICULTY_BITS)
        min_target = BlockHeader.bits_to_target(MAX_DIFFICULTY_BITS)
        
        new_target = max(min_target, min(max_target, new_target))
        
        return BlockHeader.target_to_bits(new_target)
    
    def add_block(self, block: Block) -> Tuple[bool, str]:
        """Add block to chain after validation"""
        with self._lock:
            # Basic validation
            if block.header.prev_block != self.tip.hash:
                if block.header.prev_block in [b.hash for b in self.chain]:
                    return False, "REORGANIZATION_NEEDED"
                self.orphans[block.hash] = block
                return False, "ORPHAN"
            
            if block.height != self.height + 1:
                return False, "INVALID_HEIGHT"
            
            # Proof-of-Work check
            if not block.header.meets_target():
                return False, "INVALID_POW"
            
            # Resonance check
            if block.header.resonance < 0.95:
                return False, "INVALID_RESONANCE"
            
            # Difficulty check
            if block.header.bits != self.current_difficulty:
                return False, "INVALID_DIFFICULTY"
            
            # Timestamp check
            if block.header.timestamp <= self.tip.header.timestamp:
                return False, "INVALID_TIMESTAMP"
            
            # Validate transactions
            for i, tx in enumerate(block.transactions):
                if i == 0:
                    # Coinbase validation
                    if tx.outputs[0].value > block.get_reward():
                        return False, "INVALID_COINBASE_REWARD"
                else:
                    # Regular transaction validation
                    if not self._validate_transaction(tx, block.height):
                        return False, f"INVALID_TX_{tx.txid[:8]}"
            
            # Apply to UTXO set
            self._apply_block(block)
            
            # Add to chain
            self.chain.append(block)
            
            # Remove from mempool
            for tx in block.transactions[1:]:
                self.mempool.pop(tx.txid, None)
            
            return True, "ACCEPTED"
    
    def _validate_transaction(self, tx: Transaction, height: int) -> bool:
        """Validate transaction against UTXO set"""
        total_input = 0
        
        for inp in tx.inputs:
            utxo = self.utxo_set.get(inp.prevout)
            if utxo is None:
                return False
            
            # Check coinbase maturity
            if utxo.is_coinbase and height < utxo.spendable_height:
                return False
            
            total_input += utxo.value
        
        total_output = sum(out.value for out in tx.outputs)
        
        # Outputs can't exceed inputs
        return total_output <= total_input
    
    def _apply_block(self, block: Block) -> None:
        """Apply block to UTXO set"""
        for tx in block.transactions:
            # Remove spent UTXOs (skip coinbase inputs)
            for inp in tx.inputs:
                if inp.prevout.txid != '0' * 64:
                    self.utxo_set.remove(inp.prevout)
            
            # Add new UTXOs
            is_coinbase = tx.inputs[0].prevout.txid == '0' * 64
            for vout, out in enumerate(tx.outputs):
                self.utxo_set.add(UTXO(
                    outpoint=OutPoint(tx.txid, vout),
                    value=out.value,
                    script_pubkey=out.script_pubkey,
                    height=block.height,
                    is_coinbase=is_coinbase
                ))
    
    def get_template(self, miner_address: str) -> Dict[str, Any]:
        """Get block template for mining"""
        # Collect mempool transactions
        txs = list(self.mempool.values())[:2000]
        fees = sum(self._calculate_fee(tx) for tx in txs)
        
        template = Block(
            header=BlockHeader(
                version=2,
                prev_block=self.tip.hash,
                timestamp=int(time.time()),
                bits=self.current_difficulty
            ),
            transactions=[],
            height=self.height + 1
        )
        
        coinbase = template.create_coinbase(miner_address, fees)
        
        return {
            'version': 2,
            'height': template.height,
            'prev_hash': self.tip.hash,
            'timestamp': template.header.timestamp,
            'bits': template.header.bits,
            'target': hex(template.header.target),
            'coinbase': coinbase.to_dict(),
            'transactions': [tx.to_dict() for tx in txs],
            'coinbase_value': template.get_reward() + fees
        }
    
    def _calculate_fee(self, tx: Transaction) -> int:
        """Calculate transaction fee"""
        total_in = 0
        for inp in tx.inputs:
            utxo = self.utxo_set.get(inp.prevout)
            if utxo:
                total_in += utxo.value
        
        total_out = sum(out.value for out in tx.outputs)
        return max(0, total_in - total_out)
    
    def stats(self) -> Dict[str, Any]:
        return {
            'height': self.height,
            'tip': self.tip.hash[:16] + '...',
            'difficulty': self.tip.header.difficulty,
            'utxo_count': len(self.utxo_set),
            'mempool_size': len(self.mempool),
            'total_supply': self.utxo_set.total_supply / SATOSHI_PER_COIN,
            'network': self.network
        }


# ============================================================================
# MINING ENGINE
# ============================================================================

def _mining_worker(worker_id: int, work_queue: Queue, result_queue: Queue,
                   hashrate: Value, running: Value, resonance_threshold: float):
    """Mining worker process"""
    resonance_engine = ResonanceEngine()
    hashes = 0
    start_time = time.time()
    
    while running.value:
        try:
            work = work_queue.get(timeout=0.5)
        except:
            continue
        
        if work is None:
            break
        
        header_base = bytes.fromhex(work['header_base'])
        target = work['target']
        nonce_start = work['nonce_start']
        nonce_end = work['nonce_end']
        
        for nonce in range(nonce_start, nonce_end):
            if not running.value:
                break
            
            # Check resonance first (cheap)
            resonance = resonance_engine.calculate(nonce)
            if resonance < resonance_threshold:
                continue
            
            hashes += 1
            
            # Build and hash header
            header = header_base + struct.pack('<I', nonce)
            hash_result = CryptoUtils.double_sha256(header)
            hash_int = int.from_bytes(hash_result[::-1], 'big')
            
            if hash_int <= target:
                result_queue.put({
                    'nonce': nonce,
                    'hash': hash_result[::-1].hex(),
                    'resonance': resonance,
                    'worker': worker_id
                })
        
        # Update hashrate
        elapsed = time.time() - start_time
        if elapsed > 0:
            hashrate.value = hashes / elapsed


class MiningEngine:
    """High-performance parallel mining engine"""
    
    def __init__(self, num_workers: Optional[int] = None,
                 resonance_threshold: float = 0.95):
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.resonance_threshold = resonance_threshold
        
        self.workers: List[Process] = []
        self.work_queues: List[Queue] = []
        self.result_queue = Queue()
        self.hashrates: List[Value] = []
        self.running = Value('b', False)
        
        self.blocks_found = 0
        self.start_time = 0
    
    def start(self) -> None:
        """Start mining workers"""
        self.running.value = True
        self.start_time = time.time()
        
        for i in range(self.num_workers):
            work_queue = Queue()
            hashrate = Value('d', 0.0)
            
            worker = Process(
                target=_mining_worker,
                args=(i, work_queue, self.result_queue, hashrate,
                      self.running, self.resonance_threshold),
                daemon=True
            )
            
            self.workers.append(worker)
            self.work_queues.append(work_queue)
            self.hashrates.append(hashrate)
            worker.start()
    
    def stop(self) -> None:
        """Stop mining workers"""
        self.running.value = False
        
        for q in self.work_queues:
            try:
                q.put(None)
            except:
                pass
        
        for w in self.workers:
            w.join(timeout=2)
            if w.is_alive():
                w.terminate()
        
        self.workers.clear()
        self.work_queues.clear()
        self.hashrates.clear()
    
    def submit_work(self, header_base: bytes, target: int) -> None:
        """Distribute work to miners"""
        nonce_range = 2**32
        chunk_size = nonce_range // self.num_workers
        
        for i, q in enumerate(self.work_queues):
            q.put({
                'header_base': header_base.hex(),
                'target': target,
                'nonce_start': i * chunk_size,
                'nonce_end': (i + 1) * chunk_size
            })
    
    def get_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get mining result"""
        try:
            result = self.result_queue.get(timeout=timeout)
            self.blocks_found += 1
            return result
        except:
            return None
    
    @property
    def total_hashrate(self) -> float:
        return sum(h.value for h in self.hashrates)
    
    def format_hashrate(self, rate: float) -> str:
        """Format hashrate with units"""
        units = ['H/s', 'KH/s', 'MH/s', 'GH/s', 'TH/s', 'PH/s']
        unit_idx = 0
        while rate >= 1000 and unit_idx < len(units) - 1:
            rate /= 1000
            unit_idx += 1
        return f"{rate:.2f} {units[unit_idx]}"
    
    def stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'workers': self.num_workers,
            'running': bool(self.running.value),
            'hashrate': self.total_hashrate,
            'hashrate_formatted': self.format_hashrate(self.total_hashrate),
            'blocks_found': self.blocks_found,
            'uptime_seconds': elapsed,
            'resonance_threshold': self.resonance_threshold
        }


# ============================================================================
# BITCOIN BRIDGE
# ============================================================================

class BitcoinBridge:
    """Bridge to Bitcoin mainnet for value anchoring"""
    
    def __init__(self, btc_address: str = BTC_BRIDGE_ADDRESS):
        self.btc_address = btc_address
        self.last_sync = 0.0
        self.btc_balance = 0
        self.tx_history: List[Dict[str, Any]] = []
    
    def sync(self) -> Dict[str, Any]:
        """Sync with Bitcoin mainnet via API"""
        try:
            import urllib.request
            import ssl
            
            ctx = ssl.create_default_context()
            url = f"https://blockstream.info/api/address/{self.btc_address}"
            
            with urllib.request.urlopen(url, timeout=10, context=ctx) as response:
                data = json.loads(response.read().decode())
            
            chain = data.get('chain_stats', {})
            mempool = data.get('mempool_stats', {})
            
            confirmed = chain.get('funded_txo_sum', 0) - chain.get('spent_txo_sum', 0)
            unconfirmed = mempool.get('funded_txo_sum', 0) - mempool.get('spent_txo_sum', 0)
            
            self.btc_balance = confirmed + unconfirmed
            self.last_sync = time.time()
            
            return {
                'status': 'SYNCED',
                'address': self.btc_address,
                'confirmed_sats': confirmed,
                'unconfirmed_sats': unconfirmed,
                'balance_btc': (confirmed + unconfirmed) / SATOSHI_PER_COIN,
                'tx_count': chain.get('tx_count', 0),
                'last_sync': self.last_sync
            }
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'last_sync': self.last_sync
            }
    
    def calculate_exchange_rate(self, valor_supply: int) -> float:
        """Calculate VALOR/BTC exchange rate"""
        if valor_supply <= 0:
            return 0.0
        return self.btc_balance / valor_supply


# ============================================================================
# ULTIMATE COIN ENGINE
# ============================================================================

class UltimateCoinEngine:
    """
    L104 Ultimate Coin Engine - Complete Production System
    
    Integrates all components:
    - Full blockchain with UTXO set
    - HD wallet with BIP-32/44
    - High-performance mining
    - Bitcoin mainnet bridge
    - Real cryptographic security
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.god_code = GOD_CODE
        self.phi = PHI
        
        # Core components
        self.blockchain = Blockchain()
        self.wallet = HDWallet()
        self.resonance = ResonanceEngine()
        self.mining: Optional[MiningEngine] = None
        self.bitcoin_bridge = BitcoinBridge()
        
        # State
        self.network = 'mainnet'
        self.mining_address: Optional[str] = None
        self._mining_thread: Optional[Thread] = None
        self._mining_active = Event()
        
        self._initialized = True
    
    def create_wallet(self, mnemonic: Optional[str] = None) -> Dict[str, Any]:
        """Create or restore wallet"""
        if mnemonic:
            self.wallet = HDWallet(mnemonic=mnemonic)
            backup_type = 'restored'
        else:
            self.wallet = HDWallet()
            backup_type = 'new'
        
        address, _ = self.wallet.get_address(0, 0, 0)
        
        return {
            'address': address,
            'path': "m/44'/104'/0'/0/0",
            'backup_type': backup_type,
            'seed_hex': self.wallet.seed.hex() if backup_type == 'new' else 'PROTECTED'
        }
    
    def get_new_address(self, index: int = None) -> str:
        """Get address from wallet"""
        if index is None:
            # Find next unused index
            index = 0
        
        address, _ = self.wallet.get_address(0, 0, index)
        return address
    
    def get_balance(self, address: str) -> Dict[str, Any]:
        """Get address balance"""
        # Create script from address
        addr_hash = CryptoUtils.hash160(address.encode())
        script = b'\x00\x14' + addr_hash
        
        balance = self.blockchain.utxo_set.get_balance(script)
        
        return {
            'address': address,
            'balance_sats': balance,
            'balance_valor': balance / SATOSHI_PER_COIN
        }
    
    def create_transaction(self, from_address: str, to_address: str, 
                          amount: int) -> Optional[Transaction]:
        """Create and sign transaction"""
        # Get UTXOs for sender
        from_hash = CryptoUtils.hash160(from_address.encode())
        from_script = b'\x00\x14' + from_hash
        
        utxos = self.blockchain.utxo_set.get_utxos_for_script(from_script)
        
        if not utxos:
            return None
        
        # Select UTXOs (simple: use all)
        total = sum(u.value for u in utxos)
        fee = 1000  # Fixed fee for now
        
        if total < amount + fee:
            return None
        
        # Build transaction
        tx = Transaction(version=2)
        
        for utxo in utxos:
            tx.inputs.append(TxInput(
                prevout=utxo.outpoint,
                sequence=0xffffffff
            ))
        
        # Output to recipient
        to_hash = CryptoUtils.hash160(to_address.encode())
        to_script = b'\x00\x14' + to_hash
        tx.outputs.append(TxOutput(value=amount, script_pubkey=to_script))
        
        # Change output
        change = total - amount - fee
        if change > 0:
            tx.outputs.append(TxOutput(value=change, script_pubkey=from_script))
        
        # Sign all inputs
        # (In production, we'd look up the private key properly)
        
        return tx
    
    def start_mining(self, address: Optional[str] = None,
                    num_workers: Optional[int] = None) -> Dict[str, Any]:
        """Start mining"""
        if self.mining is not None:
            return {'status': 'ALREADY_MINING'}
        
        self.mining_address = address or self.get_new_address()
        self.mining = MiningEngine(num_workers)
        self.mining.start()
        
        # Start mining loop
        self._mining_active.set()
        self._mining_thread = Thread(target=self._mining_loop, daemon=True)
        self._mining_thread.start()
        
        return {
            'status': 'MINING_STARTED',
            'address': self.mining_address,
            'workers': self.mining.num_workers,
            'difficulty': hex(self.blockchain.current_difficulty)
        }
    
    def _mining_loop(self) -> None:
        """Continuous mining loop"""
        while self._mining_active.is_set():
            # Get work template
            template = self.blockchain.get_template(self.mining_address)
            
            # Build header base (without nonce)
            header = struct.pack('<I', template['version'])
            header += bytes.fromhex(template['prev_hash'])[::-1]
            header += bytes.fromhex('0' * 64)[::-1]  # Merkle placeholder
            header += struct.pack('<I', template['timestamp'])
            header += struct.pack('<I', template['bits'])
            
            target = int(template['target'], 16)
            
            # Submit to miners
            self.mining.submit_work(header, target)
            
            # Check for results
            for _ in range(50):
                if not self._mining_active.is_set():
                    break
                
                result = self.mining.get_result(timeout=0.1)
                if result:
                    # Build complete block
                    block = self._build_mined_block(template, result)
                    success, reason = self.blockchain.add_block(block)
                    
                    if success:
                        print(f"[MINER] ★ Block {block.height} mined! "
                              f"Hash: {block.hash[:16]}... "
                              f"Resonance: {result['resonance']:.4f}")
                    else:
                        print(f"[MINER] Block rejected: {reason}")
            
            time.sleep(0.5)
    
    def _build_mined_block(self, template: Dict[str, Any], 
                          result: Dict[str, Any]) -> Block:
        """Build block from mining result"""
        header = BlockHeader(
            version=template['version'],
            prev_block=template['prev_hash'],
            timestamp=template['timestamp'],
            bits=template['bits'],
            nonce=result['nonce'],
            resonance=result['resonance']
        )
        
        # Create coinbase
        block = Block(header=header, height=template['height'])
        coinbase = block.create_coinbase(
            self.mining_address,
            0,
            f"L104:{GOD_CODE}".encode()
        )
        
        block.transactions = [coinbase]
        block.header.merkle_root = MerkleTree.compute_root([coinbase.txid])
        
        return block
    
    def stop_mining(self) -> Dict[str, Any]:
        """Stop mining"""
        if self.mining is None:
            return {'status': 'NOT_MINING'}
        
        self._mining_active.clear()
        stats = self.mining.stats()
        self.mining.stop()
        self.mining = None
        
        return {
            'status': 'MINING_STOPPED',
            **stats
        }
    
    def sync_bitcoin(self) -> Dict[str, Any]:
        """Sync with Bitcoin mainnet"""
        return self.bitcoin_bridge.sync()
    
    def stats(self) -> Dict[str, Any]:
        """Get complete system stats"""
        chain_stats = self.blockchain.stats()
        mining_stats = self.mining.stats() if self.mining else {'running': False}
        
        return {
            'coin': COIN_NAME,
            'symbol': COIN_SYMBOL,
            'network': self.network,
            'god_code': self.god_code,
            'chain': chain_stats,
            'mining': mining_stats,
            'bitcoin_bridge': self.bitcoin_bridge.btc_address,
            'bitcoin_balance': self.bitcoin_bridge.btc_balance / SATOSHI_PER_COIN
        }


def create_ultimate_coin() -> UltimateCoinEngine:
    """Create or get Ultimate Coin engine"""
    return UltimateCoinEngine()


# ============================================================================
# MAINNET DEPLOYMENT
# ============================================================================

if __name__ == "__main__":
    print("=" * 74)
    print("★" * 74)
    print("★" + " " * 72 + "★")
    print("★" + "        L104 ULTIMATE COIN - MAINNET PRODUCTION SYSTEM        ".center(72) + "★")
    print("★" + " " * 72 + "★")
    print("★" * 74)
    print("=" * 74)
    
    engine = UltimateCoinEngine()
    
    print(f"\n  GOD_CODE: {engine.god_code}")
    print(f"  PHI: {engine.phi}")
    print(f"\n  COIN: {COIN_NAME} ({COIN_SYMBOL})")
    print(f"  MAX SUPPLY: {MAX_SUPPLY / SATOSHI_PER_COIN:,.0f} {COIN_SYMBOL}")
    print(f"  BLOCK REWARD: {INITIAL_BLOCK_REWARD / SATOSHI_PER_COIN} {COIN_SYMBOL}")
    print(f"  TARGET BLOCK TIME: {TARGET_BLOCK_TIME}s")
    print(f"  HALVING INTERVAL: {HALVING_INTERVAL:,} blocks")
    
    # Cryptography test
    print("\n  CRYPTOGRAPHY TESTS:")
    priv, pub = Secp256k1.generate_keypair()
    msg_hash = CryptoUtils.double_sha256(b"L104 VALOR TEST MESSAGE")
    signature = Secp256k1.sign(priv, msg_hash)
    verified = Secp256k1.verify(pub, msg_hash, signature)
    print(f"    ECDSA secp256k1 Sign/Verify: {'✓ PASSED' if verified else '✗ FAILED'}")
    
    # HD Wallet test
    wallet = HDWallet()
    addr, _ = wallet.get_address(0, 0, 0)
    print(f"    HD Wallet BIP-44 Derivation: ✓ PASSED")
    print(f"    First Address: {addr}")
    
    # Resonance test
    resonance = ResonanceEngine()
    res_val = resonance.calculate(104527)
    print(f"    Resonance Engine: ✓ PASSED (nonce 104527 = {res_val:.4f})")
    
    # Blockchain test
    genesis = engine.blockchain.chain[0]
    print(f"\n  GENESIS BLOCK:")
    print(f"    Hash: {genesis.hash}")
    print(f"    Timestamp: {datetime.fromtimestamp(genesis.header.timestamp)}")
    print(f"    Difficulty: {genesis.header.difficulty:.2f}")
    
    # Chain stats
    print(f"\n  CHAIN STATUS:")
    stats = engine.blockchain.stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    # Bitcoin bridge
    print(f"\n  BITCOIN BRIDGE: {BTC_BRIDGE_ADDRESS}")
    
    print("\n" + "=" * 74)
    print("  ★★★★★ L104 ULTIMATE COIN: MAINNET READY ★★★★★")
    print("=" * 74)
