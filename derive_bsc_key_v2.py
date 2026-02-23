#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492612
"""
Derive BSC/Ethereum private key from 12-word mnemonic using proper libraries.
"""

import subprocess
import sys

# Install required packages
subprocess.run([sys.executable, "-m", "pip", "install", "hdwallet", "-q"],
               capture_output=True)

from hdwallet import HDWallet
from hdwallet.symbols import ETH

def main():
    print("=" * 60)
    print("BSC/Ethereum Private Key from 12-Word Mnemonic")
    print("=" * 60)
    print()
    print("⚠️  SECURITY: This runs LOCALLY. Never share your phrase!")
    print()

    if len(sys.argv) > 1:
        mnemonic = ' '.join(sys.argv[1:])
    else:
        print("Enter your 12-word Trust Wallet phrase:")
        mnemonic = input("> ").strip()

    if not mnemonic:
        print("No mnemonic provided.")
        return

    # Clean it
    mnemonic = ' '.join(mnemonic.lower().strip().split())

    try:
        # Create HD wallet
        hdwallet = HDWallet(symbol=ETH)
        hdwallet.from_mnemonic(mnemonic=mnemonic)

        # Derive using standard ETH/BSC path: m/44'/60'/0'/0/0
        hdwallet.from_path("m/44'/60'/0'/0/0")

        private_key = hdwallet.private_key()
        address = hdwallet.p2pkh_address()

        print()
        print("=" * 60)
        print("✅ DERIVED KEYS:")
        print("=" * 60)
        print()
        print(f"Address:     {address}")
        print(f"Private Key: 0x{private_key}")
        print()

        # Check if it matches expected
        expected = "0x1896f828306215c0b8198f4ef55f70081fd11a86"
        if address.lower() == expected.lower():
            print("✅ ADDRESS MATCHES YOUR TRUST WALLET!")
        else:
            print(f"⚠️  Address doesn't match {expected}")
            print("   This might be a different wallet or passphrase was used.")

        print()
        print("📋 For .env file:")
        print(f"   PRIVATE_KEY=0x{private_key}")
        print("   BSC_RPC=https://bsc-dataseed.binance.org/")
        print()
        print("⚠️  CLEAR TERMINAL after copying!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    main()
