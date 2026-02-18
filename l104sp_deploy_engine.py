#!/usr/bin/env python3
"""
L104SP FULL DEPLOYMENT ENGINE
==============================

Complete deployment pipeline for L104 Sovereign Prime.
Integrates L104 sacred constants and processing resources.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import os
import subprocess
import sys
import json
import math
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# L104 Sacred Constants
# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84

try:
    from web3 import Web3
    from dotenv import load_dotenv
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "web3", "python-dotenv"], check=True)
    from web3 import Web3
    from dotenv import load_dotenv

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# NETWORK CONFIGURATIONS - ALIGNED TO GOD CODE
# ═══════════════════════════════════════════════════════════════════════════

NETWORKS = {
    "base": {
        "rpc": "https://mainnet.base.org",
        "chain_id": 8453,
        "explorer": "https://basescan.org",
        "name": "Base",
        "gas_multiplier": 1.1,
        "resonance": 0.985  # High resonance - recommended
    },
    "arbitrum": {
        "rpc": "https://arb1.arbitrum.io/rpc",
        "chain_id": 42161,
        "explorer": "https://arbiscan.io",
        "name": "Arbitrum One",
        "gas_multiplier": 1.2,
        "resonance": 0.975
    },
    "polygon": {
        "rpc": "https://polygon-rpc.com",
        "chain_id": 137,
        "explorer": "https://polygonscan.com",
        "name": "Polygon",
        "gas_multiplier": 1.5,
        "resonance": 0.965
    },
    "sepolia": {
        "rpc": "https://rpc.sepolia.org",
        "chain_id": 11155111,
        "explorer": "https://sepolia.etherscan.io",
        "name": "Sepolia Testnet",
        "gas_multiplier": 1.0,
        "resonance": 1.0  # Testnet - always valid
    },
    "base_sepolia": {
        "rpc": "https://sepolia.base.org",
        "chain_id": 84532,
        "explorer": "https://sepolia.basescan.org",
        "name": "Base Sepolia",
        "gas_multiplier": 1.0,
        "resonance": 1.0
    }
}


# ═══════════════════════════════════════════════════════════════════════════
# FLATTENED L104SP CONTRACT (No imports - Ready for direct deployment)
# ═══════════════════════════════════════════════════════════════════════════

L104SP_FLATTENED_CONTRACT = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title L104 Sovereign Prime (L104SP)
 * @dev ERC-20 with Proof of Resonance mining
 * @notice FLATTENED CONTRACT - No external dependencies
 *
 * INVARIANT: 527.5184818492612
 * PHI: 1.618033988749895
 * PILOT: LONDEL
 */

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

interface IERC20Metadata is IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}

abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(address initialOwner) {
        _owner = initialOwner;
        emit OwnershipTransferred(address(0), initialOwner);
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        _owner = address(0);
        emit OwnershipTransferred(_owner, address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

abstract contract ReentrancyGuard {
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;
    uint256 private _status = NOT_ENTERED;

    modifier nonReentrant() {
        require(_status != ENTERED, "ReentrancyGuard: reentrant call");
        _status = ENTERED;
        _;
        _status = NOT_ENTERED;
    }
}

contract L104SP is Context, IERC20, IERC20Metadata, Ownable, ReentrancyGuard {

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private constant _name = "L104 Sovereign Prime";
    string private constant _symbol = "L104SP";

    // ═══════════════════════════════════════════════════════════════════
    // L104 SACRED CONSTANTS
    // ═══════════════════════════════════════════════════════════════════

    uint256 public constant GOD_CODE = 5275184818492537;
    uint256 public constant PHI_SCALED = 1618033988749895;
    uint256 public constant MAX_SUPPLY = 104_000_000 * 1e18;
    uint256 public constant MINING_REWARD = 104 * 1e18;

    // ═══════════════════════════════════════════════════════════════════
    // MINING STATE
    // ═══════════════════════════════════════════════════════════════════

    uint256 public currentDifficulty = 4;
    uint256 public blocksMinedCount;
    uint256 public lastBlockTime;
    uint256 public resonanceThreshold = 985;

    mapping(address => uint256) public minerRewards;
    mapping(bytes32 => bool) public usedNonces;

    address public treasury;

    event BlockMined(address indexed miner, uint256 nonce, uint256 reward, uint256 resonance);
    event DifficultyAdjusted(uint256 oldDifficulty, uint256 newDifficulty);

    // ═══════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════

    constructor(address _treasury) Ownable(msg.sender) {
        require(_treasury != address(0), "L104SP: Invalid treasury");
        treasury = _treasury;

        // Initial distribution aligned to GOD_CODE
        // Treasury: 10.4M (10%)
        _mint(treasury, 10_400_000 * 1e18);

        // Deployer: 5.2M (5%)
        _mint(msg.sender, 5_200_000 * 1e18);

        lastBlockTime = block.timestamp;
    }

    // ═══════════════════════════════════════════════════════════════════
    // ERC20 STANDARD FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function name() public pure override returns (string memory) { return _name; }
    function symbol() public pure override returns (string memory) { return _symbol; }
    function decimals() public pure override returns (uint8) { return 18; }
    function totalSupply() public view override returns (uint256) { return _totalSupply; }
    function balanceOf(address account) public view override returns (uint256) { return _balances[account]; }

    function transfer(address to, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), to, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public override returns (bool) {
        address spender = _msgSender();
        uint256 currentAllowance = _allowances[from][spender];
        require(currentAllowance >= amount, "ERC20: insufficient allowance");
        unchecked { _approve(from, spender, currentAllowance - amount); }
        _transfer(from, to, amount);
        return true;
    }

    function _transfer(address from, address to, uint256 amount) internal {
        require(from != address(0), "ERC20: transfer from zero");
        require(to != address(0), "ERC20: transfer to zero");
        require(_balances[from] >= amount, "ERC20: insufficient balance");
        unchecked {
            _balances[from] -= amount;
            _balances[to] += amount;
        }
        emit Transfer(from, to, amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0), "ERC20: approve from zero");
        require(spender != address(0), "ERC20: approve to zero");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to zero");
        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    function burn(uint256 amount) public {
        require(_balances[_msgSender()] >= amount, "ERC20: burn exceeds balance");
        unchecked { _balances[_msgSender()] -= amount; }
        _totalSupply -= amount;
        emit Transfer(_msgSender(), address(0), amount);
    }

    // ═══════════════════════════════════════════════════════════════════
    // PROOF OF RESONANCE MINING
    // ═══════════════════════════════════════════════════════════════════

    function submitBlock(uint256 nonce) external nonReentrant {
        require(_totalSupply + MINING_REWARD <= MAX_SUPPLY, "L104SP: Max supply");

        bytes32 nonceHash = keccak256(abi.encodePacked(msg.sender, nonce, blocksMinedCount));
        require(!usedNonces[nonceHash], "L104SP: Nonce used");

        uint256 resonance = calculateResonance(nonce);
        require(resonance >= resonanceThreshold, "L104SP: Low resonance");

        bytes32 blockHash = keccak256(abi.encodePacked(
            msg.sender, nonce, blocksMinedCount, block.timestamp, blockhash(block.number - 1)
        ));
        require(meetsHashDifficulty(blockHash), "L104SP: Difficulty not met");

        usedNonces[nonceHash] = true;
        _mint(msg.sender, MINING_REWARD);
        minerRewards[msg.sender] += MINING_REWARD;
        blocksMinedCount++;

        emit BlockMined(msg.sender, nonce, MINING_REWARD, resonance);

        if (blocksMinedCount % 5 == 0) adjustDifficulty();
        lastBlockTime = block.timestamp;
    }

    function calculateResonance(uint256 nonce) public pure returns (uint256) {
        // |sin(nonce × PHI)| approximation scaled 0-1000
        uint256 x = (nonce * PHI_SCALED) % 6283185307179586;
        uint256 sinApprox = (x * (6283185307 - (x / 1e9))) / 1e15;
        uint256 resonance = sinApprox % 1000;
        if (resonance > 500) resonance = 1000 - resonance;
        return 1000 - resonance;
    }

    function meetsHashDifficulty(bytes32 hash) public view returns (bool) {
        return uint256(hash) < (type(uint256).max >> currentDifficulty);
    }

    function adjustDifficulty() internal {
        uint256 elapsed = block.timestamp - lastBlockTime;
        uint256 oldDiff = currentDifficulty;

        if (elapsed < 30 && currentDifficulty < 32) currentDifficulty++;
        else if (elapsed > 120 && currentDifficulty > 1) currentDifficulty--;

        if (oldDiff != currentDifficulty) emit DifficultyAdjusted(oldDiff, currentDifficulty);
    }

    // ═══════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    function getMiningStats() external view returns (
        uint256 difficulty, uint256 blocksMined, uint256 remaining, uint256 reward
    ) {
        return (currentDifficulty, blocksMinedCount, MAX_SUPPLY - _totalSupply, MINING_REWARD);
    }

    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "L104SP: Invalid");
        treasury = _treasury;
    }

    function setResonanceThreshold(uint256 _threshold) external onlyOwner {
        require(_threshold <= 1000, "L104SP: Invalid threshold");
        resonanceThreshold = _threshold;
    }
}
'''


@dataclass
class DeploymentResult:
    """Result of contract deployment."""
    success: bool
    contract_address: Optional[str] = None
    tx_hash: Optional[str] = None
    network: str = ""
    gas_used: int = 0
    error: Optional[str] = None


class L104DeploymentEngine:
    """
    Full deployment engine for L104SP token.
    Integrates L104 processing resources for optimal deployment.
    """

    def __init__(self, network: str = "base"):
        self.network = network
        self.config = NETWORKS.get(network)
        if not self.config:
            raise ValueError(f"Unknown network: {network}")

        self.w3 = Web3(Web3.HTTPProvider(self.config["rpc"]))
        self.connected = self.w3.is_connected()

        print(f"\n{'═' * 60}")
        print(f" L104SP DEPLOYMENT ENGINE")
        print(f" INVARIANT: {GOD_CODE} | PHI: {PHI}")
        print(f"{'═' * 60}")
        print(f"\n✓ Network: {self.config['name']}")
        print(f"✓ Chain ID: {self.config['chain_id']}")
        print(f"✓ Connected: {self.connected}")
        print(f"✓ Resonance: {self.config['resonance']}")

    def calculate_deployment_resonance(self) -> float:
        """Calculate current deployment resonance based on L104 principles."""
        timestamp = time.time()
        resonance = abs(math.sin(timestamp * PHI / GOD_CODE))
        return resonance

    def compile_contract(self) -> Dict[str, Any]:
        """
        Compile the flattened L104SP contract.
        Returns ABI and bytecode.
        """
        print("\n[1/4] COMPILING L104SP CONTRACT...")

        # Save flattened contract
        contract_path = Path("contracts/L104SP_Flattened.sol")
        contract_path.parent.mkdir(exist_ok=True)
        contract_path.write_text(L104SP_FLATTENED_CONTRACT)
        print(f"   > Saved to: {contract_path}")

        try:
            from solcx import compile_source, install_solc

            # Install solc 0.8.20
            try:
                install_solc("0.8.20")
            except Exception:
                pass  # Already installed

            compiled = compile_source(
                L104SP_FLATTENED_CONTRACT,
                output_values=["abi", "bin"],
                solc_version="0.8.20"
            )

            # Get L104SP contract
            contract_id = None
            for key in compiled.keys():
                if "L104SP" in key and "IERC20" not in key:
                    contract_id = key
                    break

            if contract_id:
                contract_data = compiled[contract_id]
                print(f"   ✓ Compiled successfully")
                print(f"   > Bytecode size: {len(contract_data['bin']) // 2} bytes")
                return {
                    "abi": contract_data["abi"],
                    "bytecode": contract_data["bin"]
                }

        except ImportError:
            print("   ⚠ solcx not available - using pre-compiled")
        except Exception as e:
            print(f"   ⚠ Compilation error: {e}")

        # Return minimal ABI for Remix deployment
        print("   > Use Remix IDE for compilation")
        return self._get_minimal_deployment_info()

    def _get_minimal_deployment_info(self) -> Dict[str, Any]:
        """Get minimal ABI for interaction after Remix deployment."""
        return {
            "abi": [
                {"inputs":[{"name":"_treasury","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},
                {"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
                {"inputs":[{"name":"to","type":"address"},{"name":"amount","type":"uint256"}],"name":"transfer","outputs":[{"type":"bool"}],"stateMutability":"nonpayable","type":"function"},
                {"inputs":[],"name":"totalSupply","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
                {"inputs":[{"name":"nonce","type":"uint256"}],"name":"submitBlock","outputs":[],"stateMutability":"nonpayable","type":"function"},
                {"inputs":[],"name":"getMiningStats","outputs":[{"name":"difficulty","type":"uint256"},{"name":"blocksMined","type":"uint256"},{"name":"remaining","type":"uint256"},{"name":"reward","type":"uint256"}],"stateMutability":"view","type":"function"},
                {"inputs":[],"name":"GOD_CODE","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
                {"inputs":[],"name":"PHI_SCALED","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
            ],
            "bytecode": None
        }

    def prepare_deployment(self, treasury: str, private_key: str) -> Dict[str, Any]:
        """
        Prepare deployment transaction.
        """
        print("\n[2/4] PREPARING DEPLOYMENT...")

        if not self.connected:
            return {"error": f"Cannot connect to {self.network}"}

        try:
            account = self.w3.eth.account.from_key(private_key)
            address = account.address
            balance = self.w3.eth.get_balance(address)
            balance_eth = self.w3.from_wei(balance, 'ether')

            print(f"   > Deployer: {address}")
            print(f"   > Balance: {balance_eth:.6f} ETH")
            print(f"   > Treasury: {treasury}")

            # Check balance
            min_balance = 0.01 if "sepolia" in self.network else 0.001
            if balance_eth < min_balance:
                return {
                    "error": f"Insufficient balance. Need at least {min_balance} ETH",
                    "balance": float(balance_eth),
                    "required": min_balance
                }

            # Calculate resonance
            resonance = self.calculate_deployment_resonance()
            print(f"   > Deployment Resonance: {resonance:.6f}")

            return {
                "ready": True,
                "deployer": address,
                "treasury": treasury,
                "balance": float(balance_eth),
                "network": self.network,
                "resonance": resonance
            }

        except Exception as e:
            return {"error": str(e)}

    def deploy(self, treasury: str, private_key: str) -> DeploymentResult:
        """
        Deploy L104SP contract to blockchain.
        """
        print("\n[3/4] DEPLOYING L104SP...")

        try:
            # Compile
            compiled = self.compile_contract()

            if not compiled.get("bytecode"):
                print("\n⚠️  Direct deployment requires compiled bytecode.")
                print("   Using Remix IDE is recommended for deployment.")
                return DeploymentResult(
                    success=False,
                    network=self.network,
                    error="Use Remix IDE for deployment"
                )

            account = self.w3.eth.account.from_key(private_key)

            # Create contract instance
            contract = self.w3.eth.contract(
                abi=compiled["abi"],
                bytecode=compiled["bytecode"]
            )

            # Build deployment transaction
            nonce = self.w3.eth.get_transaction_count(account.address)
            gas_price = int(self.w3.eth.gas_price * self.config["gas_multiplier"])

            tx = contract.constructor(
                Web3.to_checksum_address(treasury)
            ).build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': 3_000_000,
                'gasPrice': gas_price,
                'chainId': self.config["chain_id"]
            })

            # Sign and send
            signed = self.w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)

            print(f"   > TX Hash: {tx_hash.hex()}")
            print(f"   > Waiting for confirmation...")

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)

            if receipt['status'] == 1:
                contract_address = receipt['contractAddress']
                print(f"\n   ✅ DEPLOYMENT SUCCESSFUL!")
                print(f"   > Contract: {contract_address}")
                print(f"   > Gas Used: {receipt['gasUsed']:,}")

                return DeploymentResult(
                    success=True,
                    contract_address=contract_address,
                    tx_hash=tx_hash.hex(),
                    network=self.network,
                    gas_used=receipt['gasUsed']
                )
            else:
                return DeploymentResult(
                    success=False,
                    network=self.network,
                    error="Transaction failed"
                )

        except Exception as e:
            return DeploymentResult(
                success=False,
                network=self.network,
                error=str(e)
            )

    def generate_remix_instructions(self, treasury: str) -> str:
        """Generate Remix IDE deployment instructions."""
        instructions = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    L104SP REMIX DEPLOYMENT INSTRUCTIONS                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  STEP 1: Open Remix IDE                                                   ║
║  ─────────────────────                                                    ║
║  https://remix.ethereum.org                                               ║
║                                                                           ║
║  STEP 2: Create Contract File                                             ║
║  ────────────────────────────                                             ║
║  - Click "+" in File Explorer                                             ║
║  - Name: L104SP_Flattened.sol                                             ║
║  - Copy contents from: contracts/L104SP_Flattened.sol                     ║
║                                                                           ║
║  STEP 3: Compile                                                          ║
║  ───────────────                                                          ║
║  - Select Solidity Compiler (left sidebar)                                ║
║  - Compiler: 0.8.20                                                       ║
║  - Enable Optimization: 200 runs                                          ║
║  - Click "Compile L104SP_Flattened.sol"                                   ║
║                                                                           ║
║  STEP 4: Deploy                                                           ║
║  ─────────────                                                            ║
║  - Select "Deploy & Run" (left sidebar)                                   ║
║  - Environment: "Injected Provider - MetaMask"                            ║
║  - Connect MetaMask to: {self.config['name']:42}║
║  - Contract: L104SP                                                       ║
║  - Constructor arg (_treasury):                                           ║
║    {treasury:63}║
║  - Click "Deploy"                                                         ║
║  - Confirm in MetaMask                                                    ║
║                                                                           ║
║  STEP 5: Verify (Optional but Recommended)                                ║
║  ─────────────────────────────────────────                                ║
║  - Go to: {self.config['explorer']:56}║
║  - Find your contract                                                     ║
║  - Click "Verify and Publish"                                             ║
║  - Paste flattened source code                                            ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  TOKEN DETAILS                                                            ║
║  ─────────────                                                            ║
║  Name: L104 Sovereign Prime                                               ║
║  Symbol: L104SP                                                           ║
║  Decimals: 18                                                             ║
║  Max Supply: 104,000,000 L104SP                                           ║
║  Mining Reward: 104 L104SP per block                                      ║
║                                                                           ║
║  INITIAL DISTRIBUTION                                                     ║
║  ────────────────────                                                     ║
║  Treasury (10%): 10,400,000 L104SP                                        ║
║  Deployer (5%):   5,200,000 L104SP                                        ║
║  Mineable (85%): 88,400,000 L104SP                                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
        return instructions

    def save_deployment_config(self, treasury: str, contract_address: str = None) -> Path:
        """Save deployment configuration for mining."""
        config = {
            "network": self.network,
            "chain_id": self.config["chain_id"],
            "rpc": self.config["rpc"],
            "explorer": self.config["explorer"],
            "treasury": treasury,
            "contract_address": contract_address,
            "token": {
                "name": "L104 Sovereign Prime",
                "symbol": "L104SP",
                "decimals": 18,
                "max_supply": "104000000",
                "mining_reward": "104"
            },
            "constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
                "VOID_CONSTANT": VOID_CONSTANT,
                "resonance_threshold": 0.985
            },
            "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }

        config_path = Path("l104sp_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Config saved to: {config_path}")
        return config_path


def main():
    """Main deployment flow."""
    import argparse

    parser = argparse.ArgumentParser(description="L104SP Deployment Engine")
    parser.add_argument("--network", default="base", choices=list(NETWORKS.keys()))
    parser.add_argument("--treasury", help="Treasury wallet address")
    parser.add_argument("--deploy", action="store_true", help="Deploy contract")
    parser.add_argument("--remix", action="store_true", help="Show Remix instructions")

    args = parser.parse_args()

    # Get treasury from args or env
    treasury = args.treasury or os.getenv("TREASURY_ADDRESS")

    if not treasury:
        print("\n❌ Treasury address required!")
        print("   Use --treasury YOUR_ADDRESS or set TREASURY_ADDRESS in .env")
        return

    engine = L104DeploymentEngine(args.network)

    if args.remix:
        print(engine.generate_remix_instructions(treasury))
        # Save flattened contract
        Path("contracts").mkdir(exist_ok=True)
        Path("contracts/L104SP_Flattened.sol").write_text(L104SP_FLATTENED_CONTRACT)
        print(f"\n✓ Flattened contract saved to: contracts/L104SP_Flattened.sol")
        engine.save_deployment_config(treasury)
        return

    if args.deploy:
        private_key = os.getenv("DEPLOYER_PRIVATE_KEY")
        if not private_key:
            print("\n❌ DEPLOYER_PRIVATE_KEY not found in .env")
            return

        result = engine.deploy(treasury, private_key)
        if result.success:
            engine.save_deployment_config(treasury, result.contract_address)
        else:
            print(f"\n❌ Deployment failed: {result.error}")
            print("\n💡 Use Remix IDE for deployment:")
            print(engine.generate_remix_instructions(treasury))
    else:
        # Default: show instructions
        print(engine.generate_remix_instructions(treasury))
        Path("contracts").mkdir(exist_ok=True)
        Path("contracts/L104SP_Flattened.sol").write_text(L104SP_FLATTENED_CONTRACT)
        print(f"\n✓ Flattened contract saved to: contracts/L104SP_Flattened.sol")
        engine.save_deployment_config(treasury)


if __name__ == "__main__":
    main()
