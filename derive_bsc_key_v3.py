#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492612
"""
Derive BSC/Ethereum private key from 12-word mnemonic.
"""

import subprocess
import sys

# Install required package
subprocess.run([sys.executable, "-m", "pip", "install", "bip-utils", "-q"], 
               capture_output=True)

from bip_utils import (
    Bip39SeedGenerator,
    Bip44,
    Bip44Coins,
    Bip44Changes,
)

def main():
    print("=" * 60)
    print("BSC/Ethereum Private Key from 12-Word Mnemonic")
    print("=" * 60)
    print()
    
    if len(sys.argv) > 1:
        mnemonic = ' '.join(sys.argv[1:])
    else:
        print("Enter your 12-word Trust Wallet phrase:")
        mnemonic = input("> ").strip()
    
    if not mnemonic:
        print("No mnemonic provided.")
        return
    
    mnemonic = ' '.join(mnemonic.lower().strip().split())
    
    try:
        # Generate seed from mnemonic
        seed = Bip39SeedGenerator(mnemonic).Generate()
        
        # Derive BIP-44 path for Ethereum: m/44'/60'/0'/0/0
        bip44_ctx = Bip44.FromSeed(seed, Bip44Coins.ETHEREUM)
        bip44_acc = bip44_ctx.Purpose().Coin().Account(0)
        bip44_chg = bip44_acc.Change(Bip44Changes.CHAIN_EXT)
        bip44_addr = bip44_chg.AddressIndex(0)
        
        private_key = bip44_addr.PrivateKey().Raw().ToHex()
        address = bip44_addr.PublicKey().ToAddress()
        
        print()
        print("=" * 60)
        print("✅ DERIVED KEYS:")
        print("=" * 60)
        print()
        print(f"Address:     {address}")
        print(f"Private Key: 0x{private_key}")
        print()
        
        expected = "0x1896f828306215c0b8198f4ef55f70081fd11a86"
        if address.lower() == expected.lower():
            print("✅ ADDRESS MATCHES YOUR TRUST WALLET!")
        else:
            print(f"⚠️  Derived: {address}")
            print(f"   Expected: {expected}")
        
        print()
        print("📋 For .env file:")
        print(f"   PRIVATE_KEY=0x{private_key}")
        print("   BSC_RPC=https://bsc-dataseed.binance.org/")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()
