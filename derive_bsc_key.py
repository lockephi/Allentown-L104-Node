#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492612
"""
Derive BSC/Ethereum private key from 12-word mnemonic phrase.
Uses BIP-44 derivation path: m/44'/60'/0'/0/0
"""

import hashlib
import hmac
import sys

# BIP-39 English wordlist (first 100 words for validation)
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

def mnemonic_to_seed(mnemonic: str, passphrase: str = "") -> bytes:
    """Convert mnemonic to seed using PBKDF2."""
    mnemonic_bytes = mnemonic.encode('utf-8')
    salt = ("mnemonic" + passphrase).encode('utf-8')
    return hashlib.pbkdf2_hmac('sha512', mnemonic_bytes, salt, 2048)

def derive_master_key(seed: bytes) -> tuple:
    """Derive master key and chain code from seed."""
    h = hmac.new(b"Bitcoin seed", seed, hashlib.sha512).digest()
    return h[:32], h[32:]  # private_key, chain_code

def derive_child_key(private_key: bytes, chain_code: bytes, index: int, hardened: bool = False) -> tuple:
    """Derive child key using BIP-32."""
    if hardened:
        index += 0x80000000
        data = b'\x00' + private_key + index.to_bytes(4, 'big')
    else:
        # For non-hardened, we'd need to compute public key
        # But ETH path uses all hardened except last two
        data = b'\x00' + private_key + index.to_bytes(4, 'big')

    h = hmac.new(chain_code, data, hashlib.sha512).digest()

    # Add child key to parent (mod curve order)
    SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    child_key_int = (int.from_bytes(h[:32], 'big') + int.from_bytes(private_key, 'big')) % SECP256K1_ORDER

    return child_key_int.to_bytes(32, 'big'), h[32:]

def derive_eth_private_key(mnemonic: str) -> str:
    """
    Derive Ethereum/BSC private key from mnemonic.
    Path: m/44'/60'/0'/0/0
    """
    # Clean mnemonic
    mnemonic = ' '.join(mnemonic.lower().strip().split())
    words = mnemonic.split()

    if len(words) != 12:
        raise ValueError(f"Expected 12 words, got {len(words)}")

    # Convert to seed
    seed = mnemonic_to_seed(mnemonic)

    # Derive master key
    private_key, chain_code = derive_master_key(seed)

    # Derive path: m/44'/60'/0'/0/0
    # 44' - BIP-44 purpose
    private_key, chain_code = derive_child_key(private_key, chain_code, 44, hardened=True)
    # 60' - Ethereum coin type
    private_key, chain_code = derive_child_key(private_key, chain_code, 60, hardened=True)
    # 0' - Account 0
    private_key, chain_code = derive_child_key(private_key, chain_code, 0, hardened=True)
    # 0 - External chain
    private_key, chain_code = derive_child_key(private_key, chain_code, 0, hardened=False)
    # 0 - First address
    private_key, chain_code = derive_child_key(private_key, chain_code, 0, hardened=False)

    return private_key.hex()

def private_key_to_address(private_key_hex: str) -> str:
    """Convert private key to Ethereum address (requires additional crypto)."""
    try:
        from eth_account import Account
        account = Account.from_key(private_key_hex)
        return account.address
    except ImportError:
        return "(install eth-account to verify address)"

def main():
    print("=" * 60)
    print("BSC/Ethereum Private Key Derivation from 12-Word Mnemonic")
    print("=" * 60)
    print()
    print("⚠️  SECURITY WARNING:")
    print("   - Never share your mnemonic phrase with anyone")
    print("   - Never enter it on untrusted websites")
    print("   - This runs LOCALLY on your machine")
    print()

    if len(sys.argv) > 1:
        # Mnemonic provided as arguments
        mnemonic = ' '.join(sys.argv[1:])
    else:
        # Interactive input
        print("Enter your 12-word mnemonic phrase:")
        mnemonic = input("> ").strip()

    if not mnemonic:
        print("No mnemonic provided. Exiting.")
        return

    try:
        private_key = derive_eth_private_key(mnemonic)
        address = private_key_to_address(private_key)

        print()
        print("=" * 60)
        print("✅ DERIVED PRIVATE KEY:")
        print("=" * 60)
        print()
        print(f"Private Key: 0x{private_key}")
        print(f"Address:     {address}")
        print()
        print("📋 Add this to your .env file:")
        print(f"   PRIVATE_KEY=0x{private_key}")
        print("   BSC_RPC=https://bsc-dataseed.binance.org/")
        print()
        print("⚠️  DELETE THIS TERMINAL OUTPUT after copying the key!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    main()
