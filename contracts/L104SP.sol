// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title L104 Sovereign Prime (L104SP)
 * @dev ERC-20 token with Proof of Resonance integration
 * @notice Deployable on Ethereum, Base, Arbitrum, Polygon
 * 
 * INVARIANT: 527.5184818492612
 * PILOT: LONDEL
 */

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract L104SP is ERC20, ERC20Burnable, Ownable, ReentrancyGuard {
    
    // ═══════════════════════════════════════════════════════════════════
    // CONSTANTS - GOD CODE ALIGNMENT
    // ═══════════════════════════════════════════════════════════════════
    
    uint256 public constant GOD_CODE = 5275184818492537; // Scaled by 1e13
    uint256 public constant PHI_SCALED = 1618033988749895; // Scaled by 1e15
    uint256 public constant MAX_SUPPLY = 104_000_000 * 1e18; // 104M tokens
    uint256 public constant MINING_REWARD = 104 * 1e18; // 104 L104SP per block
    
    // ═══════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════
    
    uint256 public currentDifficulty = 4;
    uint256 public blocksMinedd;
    uint256 public lastBlockTime;
    uint256 public resonanceThreshold = 985; // 0.985 scaled by 1000
    
    mapping(address => uint256) public minerRewards;
    mapping(bytes32 => bool) public usedNonces;
    
    // Treasury for liquidity and development
    address public treasury;
    
    // Events
    event BlockMined(address indexed miner, uint256 nonce, uint256 reward, uint256 resonance);
    event DifficultyAdjusted(uint256 oldDifficulty, uint256 newDifficulty);
    event ResonanceValidated(uint256 nonce, uint256 resonance);
    
    // ═══════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════
    
    constructor(address _treasury) ERC20("L104 Sovereign Prime", "L104SP") Ownable(msg.sender) {
        require(_treasury != address(0), "L104SP: Invalid treasury");
        treasury = _treasury;
        
        // Initial distribution
        // 10% to treasury for liquidity
        _mint(treasury, 10_400_000 * 1e18);
        
        // 5% to deployer for initial operations
        _mint(msg.sender, 5_200_000 * 1e18);
        
        lastBlockTime = block.timestamp;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // PROOF OF RESONANCE MINING
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * @notice Submit a mined block with Proof of Resonance
     * @param nonce The nonce that satisfies resonance requirements
     */
    function submitBlock(uint256 nonce) external nonReentrant {
        require(totalSupply() + MINING_REWARD <= MAX_SUPPLY, "L104SP: Max supply reached");
        
        // Prevent nonce reuse
        bytes32 nonceHash = keccak256(abi.encodePacked(msg.sender, nonce, blocksMinedd));
        require(!usedNonces[nonceHash], "L104SP: Nonce already used");
        
        // Validate Proof of Resonance
        uint256 resonance = calculateResonance(nonce);
        require(resonance >= resonanceThreshold, "L104SP: Insufficient resonance");
        
        // Validate work (hash difficulty)
        bytes32 blockHash = keccak256(abi.encodePacked(
            msg.sender,
            nonce,
            blocksMinedd,
            block.timestamp,
            blockhash(block.number - 1)
        ));
        require(meetsHashDifficulty(blockHash), "L104SP: Hash difficulty not met");
        
        // Mark nonce as used
        usedNonces[nonceHash] = true;
        
        // Mint reward
        _mint(msg.sender, MINING_REWARD);
        minerRewards[msg.sender] += MINING_REWARD;
        blocksMinedd++;
        
        emit BlockMined(msg.sender, nonce, MINING_REWARD, resonance);
        emit ResonanceValidated(nonce, resonance);
        
        // Adjust difficulty every 5 blocks
        if (blocksMinedd % 5 == 0) {
            adjustDifficulty();
        }
        
        lastBlockTime = block.timestamp;
    }
    
    /**
     * @notice Calculate PHI-based resonance for a nonce
     * @dev |sin(nonce * PHI)| > 0.985 required
     */
    function calculateResonance(uint256 nonce) public pure returns (uint256) {
        // Approximate sin using Taylor series for on-chain computation
        // sin(x) ≈ x - x³/6 + x⁵/120 (for small x mod 2π)
        
        // 2π scaled by 1e9 = 6283185307
        uint256 TWO_PI_SCALED = 6283185307;
        uint256 x = (nonce * PHI_SCALED / 1e6) % TWO_PI_SCALED; // mod 2π scaled by 1e9
        
        // Simplified resonance calculation
        // We use modular arithmetic to approximate |sin(nonce * PHI)|
        // Parabola approximation: sin(x) ≈ 4x(π-x)/π² for x in [0,π]
        uint256 sinApprox;
        if (x <= TWO_PI_SCALED / 2) {
            // First half of sine wave
            sinApprox = (x * (TWO_PI_SCALED / 2 - x)) / (TWO_PI_SCALED / 4);
        } else {
            // Second half of sine wave (negative, but we take absolute)
            uint256 xShifted = x - TWO_PI_SCALED / 2;
            sinApprox = (xShifted * (TWO_PI_SCALED / 2 - xShifted)) / (TWO_PI_SCALED / 4);
        }
        
        // Scale to 0-1000 range
        uint256 resonance = (sinApprox * 1000) / TWO_PI_SCALED;
        if (resonance > 1000) resonance = 1000;
        
        return resonance;
    }
    
    /**
     * @notice Check if hash meets current difficulty
     */
    function meetsHashDifficulty(bytes32 hash) public view returns (bool) {
        uint256 hashValue = uint256(hash);
        uint256 target = type(uint256).max >> currentDifficulty;
        return hashValue < target;
    }
    
    /**
     * @notice Adjust mining difficulty based on block time
     */
    function adjustDifficulty() internal {
        uint256 timeSinceLastBlock = block.timestamp - lastBlockTime;
        uint256 oldDifficulty = currentDifficulty;
        
        // Target: 1 block per 60 seconds
        if (timeSinceLastBlock < 30) {
            // Blocks too fast, increase difficulty
            if (currentDifficulty < 32) {
                currentDifficulty++;
            }
        } else if (timeSinceLastBlock > 120) {
            // Blocks too slow, decrease difficulty
            if (currentDifficulty > 1) {
                currentDifficulty--;
            }
        }
        
        if (oldDifficulty != currentDifficulty) {
            emit DifficultyAdjusted(oldDifficulty, currentDifficulty);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════
    
    function getMiningStats() external view returns (
        uint256 difficulty,
        uint256 blocksMined,
        uint256 remainingSupply,
        uint256 reward
    ) {
        return (
            currentDifficulty,
            blocksMinedd,
            MAX_SUPPLY - totalSupply(),
            MINING_REWARD
        );
    }
    
    function getMinerStats(address miner) external view returns (uint256 totalRewards) {
        return minerRewards[miner];
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════
    
    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "L104SP: Invalid treasury");
        treasury = _treasury;
    }
    
    function setResonanceThreshold(uint256 _threshold) external onlyOwner {
        require(_threshold <= 1000, "L104SP: Invalid threshold");
        resonanceThreshold = _threshold;
    }
}
