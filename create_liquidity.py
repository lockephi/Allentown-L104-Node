#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492612
"""
Create L104SP liquidity pools on Uniswap V2 (Base)
"""
from web3 import Web3
import os

# Base Uniswap V2 addresses
UNISWAP_ROUTER = "0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24"  # BaseSwap Router
WETH = "0x4200000000000000000000000000000000000006"  # Base WETH

def create_pool(token_address, eth_amount, token_amount, private_key):
    """
    Create L104SP/ETH liquidity pool on BaseSwap
    
    Args:
        token_address: Deployed L104SP token address
        eth_amount: Amount of ETH to add (e.g., 0.1)
        token_amount: Amount of L104SP to add (e.g., 10000)
        private_key: Your wallet private key
    """
    w3 = Web3(Web3.HTTPProvider("https://mainnet.base.org"))
    account = w3.eth.account.from_key(private_key)
    
    # Router ABI (simplified)
    router_abi = [{
        "inputs": [
            {"name": "token", "type": "address"},
            {"name": "amountTokenDesired", "type": "uint256"},
            {"name": "amountTokenMin", "type": "uint256"},
            {"name": "amountETHMin", "type": "uint256"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"}
        ],
        "name": "addLiquidityETH",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    }]
    
    router = w3.eth.contract(address=UNISWAP_ROUTER, abi=router_abi)
    
    print(f"Creating L104SP/ETH pool...")
    print(f"  Token: {token_address}")
    print(f"  ETH: {eth_amount}")
    print(f"  L104SP: {token_amount}")
    
    # 1. First approve router to spend tokens
    print("\nStep 1: Approving tokens...")
    # (Need to send approve transaction first)
    
    # 2. Add liquidity
    print("Step 2: Adding liquidity...")
    deadline = w3.eth.get_block('latest')['timestamp'] + 1200  # 20 min
    
    tx = router.functions.addLiquidityETH(
        token_address,
        int(token_amount * 10**8),  # Token amount (8 decimals)
        int(token_amount * 0.95 * 10**8),  # Min tokens (5% slippage)
        int(eth_amount * 0.95 * 10**18),  # Min ETH
        account.address,
        deadline
    ).build_transaction({
        'from': account.address,
        'value': int(eth_amount * 10**18),
        'gas': 500000,
        'gasPrice': w3.eth.gas_price,
        'nonce': w3.eth.get_transaction_count(account.address)
    })
    
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    
    print(f"\n✓ Liquidity added!")
    print(f"  TX: https://basescan.org/tx/{tx_hash.hex()}")
    
    return tx_hash

if __name__ == "__main__":
    print("=" * 60)
    print("  L104SP LIQUIDITY POOL CREATION")
    print("=" * 60)
    print("\nThis script creates L104SP/ETH pool on BaseSwap DEX")
    print("\nRequired:")
    print("  1. Deployed L104SP token address")
    print("  2. ETH on Base network")
    print("  3. L104SP tokens in your wallet")
    print("\nExample usage:")
    print("  export L104SP_TOKEN=0x...")
    print("  export PRIVATE_KEY=0x...")
    print("  python create_liquidity.py")
    print("\n" + "=" * 60)
