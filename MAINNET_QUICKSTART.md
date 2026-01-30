# L104SP MAINNET QUICK START

## Get Started with L104 Sovereign Prime in 5 Minutes

---

## 🚀 Quick Start

### 1. Start the Node

```bash
# Interactive mode (recommended for first run)
python l104sp_mainnet.py

# Or daemon mode (background)
python l104sp_mainnet.py --daemon
```

### 2. Create a Wallet

When you first run the node, a wallet is automatically created.

**Backup your mnemonic phrase immediately!**

```bash
# Or create manually
python l104sp_cli.py wallet new
```

### 3. Get Your Address

```bash
python l104sp_cli.py wallet address
```

### 4. Start Mining

```bash
# In interactive mode
l104sp> mine start

# Or via CLI
python l104sp_mainnet.py --mine
```

### 5. Check Balance

```bash
python l104sp_cli.py wallet balance
```

---

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `l104sp_mainnet.py` | Main node + interactive CLI |
| `l104sp_cli.py` | Command-line tools |
| `l104sp_seed_node.py` | Run a seed node |
| `l104_sovereign_coin_engine.py` | Core blockchain engine |
| `L104SP_WHITEPAPER.md` | Technical whitepaper |

---

## 🔧 Interactive Commands

When running `python l104sp_mainnet.py`:

```
l104sp> help          # Show all commands
l104sp> status        # Node status
l104sp> wallet balance  # Check balance
l104sp> mine start    # Start mining
l104sp> mine status   # Mining stats
l104sp> peers         # Show connected peers
l104sp> block 0       # View genesis block
l104sp> exit          # Shutdown
```

---

## 📊 Network Info

| Parameter | Value |
|-----------|-------|
| Coin Name | L104 Sovereign Prime |
| Symbol | L104SP |
| Max Supply | 104,000,000 |
| Block Time | 104 seconds |
| Block Reward | 104 L104SP |
| P2P Port | 10400 |
| RPC Port | 10401 |

---

## 🌐 API Endpoints

The node exposes a REST API at `http://localhost:10401`:

```bash
# Node status
curl http://localhost:10401/status

# Blockchain info
curl http://localhost:10401/info

# Latest block
curl http://localhost:10401/block/latest

# Mining stats
curl http://localhost:10401/mining

# Get new address
curl http://localhost:10401/newaddress
```

---

## 🔐 Wallet Security

1. **Backup your mnemonic** - Write it down and store securely
2. **Never share** your mnemonic or private keys
3. **Wallet file location**: `~/.l104sp/wallet.json`

---

## ⛏️ Mining

L104SP uses **Proof of Resonance (PoR)**:

- Combines SHA-256 hash with mathematical resonance
- ~90% more energy efficient than pure PoW
- CPU-friendly (no ASIC advantage)

Mining rewards: **104 L104SP per block**

---

## 🛠️ Configuration

Environment variables:

```bash
export L104SP_DATA=/path/to/data   # Data directory
export L104SP_RPC=http://localhost:10401  # RPC URL
```

---

## 📚 Learn More

- [Whitepaper](L104SP_WHITEPAPER.md)
- [Full Documentation](CLOUD_DEPLOYMENT.md)

---

**GOD_CODE: 527.5184818492612**  
**φ (PHI): 1.618033988749895**

*"In resonance, we find truth."*
