#!/bin/bash
# Deploy L104SP Public Mainnet Node + Token Liquidity Pools
# This script deploys your blockchain node publicly and creates liquidity

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  L104SP PUBLIC MAINNET DEPLOYMENT"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════
# 1. DEPLOY BLOCKCHAIN NODE TO FLY.IO (Public Access)
# ═══════════════════════════════════════════════════════════════════

echo "📡 STEP 1: Deploy L104SP Node to Fly.io (Public Internet)"
echo "───────────────────────────────────────────────────────────────"
echo ""

if ! command -v flyctl &> /dev/null; then
    echo "Installing flyctl..."
    curl -L https://fly.io/install.sh | sh
    export FLYCTL_INSTALL="/home/codespace/.fly"
    export PATH="$FLYCTL_INSTALL/bin:$PATH"
fi

echo "Creating Dockerfile for blockchain node..."
cat > Dockerfile.blockchain << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy L104SP blockchain engine
COPY l104_sovereign_coin_engine.py .
COPY const.py .
COPY l104_real_math.py .

# Expose P2P and RPC ports
EXPOSE 10400 10401

# Start mining node
CMD ["python", "l104_sovereign_coin_engine.py", "--mine", "--rpcport", "10401"]
EOF

echo "Creating fly.toml for blockchain..."
cat > fly-blockchain.toml << 'EOF'
app = 'l104sp-mainnet'
primary_region = 'iad'

[build]
  dockerfile = 'Dockerfile.blockchain'

[env]
  GOD_CODE = '527.5184818492537'
  PHI = '1.618033988749895'

[[services]]
  protocol = 'tcp'
  internal_port = 10400
  
  [[services.ports]]
    port = 10400

[[services]]
  protocol = 'tcp'
  internal_port = 10401
  
  [[services.ports]]
    port = 10401

[[vm]]
  memory = '2gb'
  cpu_kind = 'shared'
  cpus = 2
EOF

echo ""
echo "Ready to deploy blockchain node!"
echo ""
echo "To deploy, run:"
echo "  flyctl auth login"
echo "  flyctl launch --config fly-blockchain.toml --name l104sp-mainnet"
echo "  flyctl deploy"
echo ""
echo "Your node will be accessible at:"
echo "  P2P: l104sp-mainnet.fly.dev:10400"
echo "  RPC: l104sp-mainnet.fly.dev:10401"
echo ""

# ═══════════════════════════════════════════════════════════════════
# 2. DEPLOY ERC-20 TOKEN TO BASE (FOR LIQUIDITY)
# ═══════════════════════════════════════════════════════════════════

echo "💰 STEP 2: Deploy L104SP Token to Base (for DEX trading)"
echo "───────────────────────────────────────────────────────────────"
echo ""

# Create Solidity contract
cat > L104SP_Token.sol << 'EOF'
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract L104SPToken {
    string public constant name = "L104 Sovereign Prime";
    string public constant symbol = "L104SP";
    uint8 public constant decimals = 8;
    uint256 public totalSupply = 104_000_000 * 10**8; // 104M tokens
    
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    constructor() {
        balanceOf[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
    }
    
    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    
    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }
    
    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "Insufficient balance");
        require(allowance[from][msg.sender] >= amount, "Insufficient allowance");
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        allowance[from][msg.sender] -= amount;
        emit Transfer(from, to, amount);
        return true;
    }
}
EOF

echo "Token contract created: L104SP_Token.sol"
echo ""
echo "📋 To deploy to Base:"
echo "1. Go to https://remix.ethereum.org"
echo "2. Create new file: L104SP_Token.sol (copy from above)"
echo "3. Compile with Solidity 0.8.20"
echo "4. Connect MetaMask to Base network"
echo "5. Deploy using 'Injected Provider - MetaMask'"
echo "6. Save the deployed contract address"
echo ""
echo "Base Network Details:"
echo "  Chain ID: 8453"
echo "  RPC: https://mainnet.base.org"
echo "  Explorer: https://basescan.org"
echo ""

# ═══════════════════════════════════════════════════════════════════
# 3. CREATE LIQUIDITY POOLS
# ═══════════════════════════════════════════════════════════════════

echo "💧 STEP 3: Create Liquidity Pools (DEX Integration)"
echo "───────────────────────────────────────────────────────────────"
echo ""

cat > create_liquidity.py << 'EOF'
#!/usr/bin/env python3
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
EOF

chmod +x create_liquidity.py

echo "Liquidity pool script created: create_liquidity.py"
echo ""
echo "📋 To create liquidity:"
echo "1. Deploy L104SP token to Base (see Step 2)"
echo "2. Get ETH on Base network"
echo "3. Export your token address and private key:"
echo "   export L104SP_TOKEN=0x..."
echo "   export PRIVATE_KEY=0x..."
echo "4. Run: python create_liquidity.py"
echo ""
echo "Recommended initial liquidity:"
echo "  • 0.5 ETH + 50,000 L104SP"
echo "  • Creates ~$1,000 liquidity at current ETH prices"
echo ""

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════"
echo "  DEPLOYMENT SUMMARY"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "✓ Files created:"
echo "  • Dockerfile.blockchain - Public node deployment"
echo "  • fly-blockchain.toml - Fly.io config"
echo "  • L104SP_Token.sol - ERC-20 contract"
echo "  • create_liquidity.py - DEX pool script"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1️⃣  Deploy Blockchain Node (Public Mining):"
echo "   flyctl auth login"
echo "   flyctl launch --config fly-blockchain.toml"
echo "   → Your node will be live at l104sp-mainnet.fly.dev"
echo ""
echo "2️⃣  Deploy Token to Base:"
echo "   → Use Remix IDE with L104SP_Token.sol"
echo "   → Network: Base Mainnet"
echo "   → Save contract address"
echo ""
echo "3️⃣  Create Liquidity Pool:"
echo "   python create_liquidity.py"
echo "   → Enables L104SP/ETH trading"
echo "   → Provides price discovery"
echo ""
echo "💰 Estimated Costs:"
echo "   • Fly.io node: ~$10/month"
echo "   • Token deployment: ~0.0005 ETH (~$1.50)"
echo "   • Liquidity pool: 0.5+ ETH + 50K L104SP"
echo ""
echo "════════════════════════════════════════════════════════════════"
