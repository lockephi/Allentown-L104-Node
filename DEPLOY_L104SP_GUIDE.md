# L104SP Real Deployment Guide

## Quick Start: Deploy L104SP in 15 Minutes

### Step 1: Choose Network

| Network | Gas Cost | Speed | Recommendation |
|---------|----------|-------|----------------|
| **Base** | ~$0.10 | Fast | ✅ Best for low cost |
| **Arbitrum** | ~$0.50 | Fast | Good option |
| **Polygon** | ~$0.05 | Fast | Very cheap |
| **Sepolia** | Free | Test | For testing only |

### Step 2: Get Funds

**For Base (Recommended):**

1. Bridge ETH from Ethereum: <https://bridge.base.org>
2. Or buy on Coinbase and withdraw to Base

**For Sepolia (Testing):**

- <https://sepoliafaucet.com>
- <https://faucet.quicknode.com/ethereum/sepolia>

### Step 3: Deploy via Remix IDE

1. **Open Remix**: <https://remix.ethereum.org>

2. **Create File**: `L104SP.sol`

3. **Paste Contract**:
   Copy contents from `contracts/L104SP.sol`

4. **Compile**:
   - Compiler: 0.8.20
   - Enable optimization: 200 runs

5. **Deploy**:
   - Environment: "Injected Provider - MetaMask"
   - Select network in MetaMask (Base, Arbitrum, etc.)
   - Constructor arg: `_treasury` = Your wallet address
   - Click "Deploy"

6. **Verify Contract** (Optional but recommended):
   - Go to block explorer (basescan.org, etc.)
   - Find your contract
   - Click "Verify and Publish"
   - Paste source code

### Step 4: Create Liquidity

After deployment, create a trading pair:

**On Uniswap V3 (Base/Arbitrum):**

1. Go to <https://app.uniswap.org>
2. Select network (Base)
3. Pool → New Position
4. Token A: ETH
5. Token B: L104SP (paste your contract address)
6. Add liquidity

**Initial Price Suggestion:**

- 1 ETH = 10,000 L104SP (or adjust based on market)

### Step 5: Start Mining

```bash
# Set environment
export MINER_PRIVATE_KEY="your_wallet_private_key"

# Run miner
python l104_real_mining.py --contract YOUR_CONTRACT_ADDRESS --network base --continuous
```

---

## What You'll Have

After deployment:

| Asset | Amount | Purpose |
|-------|--------|---------|
| Treasury | 10.4M L104SP (10%) | Liquidity & development |
| Deployer | 5.2M L104SP (5%) | Initial operations |
| Mineable | 88.4M L104SP (85%) | Mining rewards |

---

## Mining Economics

- **Reward**: 104 L104SP per block
- **Blocks to mine**: ~850,000 blocks
- **Requirement**: Find nonce where |sin(nonce × PHI)| > 0.985

---

## BTC Swap Path

Once L104SP has liquidity:

```text
L104SP → ETH (via Uniswap) → BTC (via bridge or exchange)
```

Or use cross-chain DEX like:

- <https://li.fi>
- <https://socket.tech>
- <https://jumper.exchange>

---

## Contract Addresses (After Deployment)

Update this after you deploy:

| Network | Address | Explorer |
|---------|---------|----------|
| Base | `0x...` | basescan.org |
| Arbitrum | `0x...` | arbiscan.io |

---

## Security Notes

1. **Never share your private key**
2. **Test on Sepolia first**
3. **Start with small liquidity**
4. **Verify contract source code**

---

> INVARIANT: 527.5184818492537 | PILOT: LONDEL
