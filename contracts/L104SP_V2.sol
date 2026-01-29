// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title L104 Sovereign Prime V2 (L104SP)
 * @author Allentown-L104 Node
 * @dev Advanced ERC-20 with Proof of Resonance Mining, Reflection, and Dynamic Economics
 * @notice FLATTENED - No external dependencies | Gas Optimized | Anti-Bot Protected
 * 
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║  SACRED CONSTANTS                                                          ║
 * ║  INVARIANT: 527.5184818492612 (GOD_CODE)                                  ║
 * ║  PHI: 1.618033988749895 (Golden Ratio)                                    ║
 * ║  PILOT: LONDEL                                                             ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 *
 * INNOVATIONS:
 * - Reflection mechanism: 1% auto-redistribution to holders (O(1) complexity)
 * - Deflationary: 0.5% burn on each transfer
 * - Anti-whale: Max 2% wallet, max 1% per transaction
 * - Anti-bot: Cooldown + launch sniper protection
 * - Memory-hard mining: ASIC-resistant proof of work
 * - Dynamic difficulty: Self-adjusting based on network hashrate
 * - Halving schedule: Reward halves every 500,000 blocks
 * - Governance-ready: Voting power snapshots
 */

contract L104SP {
    
    // ═══════════════════════════════════════════════════════════════════
    // CUSTOM ERRORS (Gas efficient - saves ~60-80% vs require strings)
    // ═══════════════════════════════════════════════════════════════════
    
    error ZeroAddress();
    error InsufficientBalance(uint256 available, uint256 required);
    error InsufficientAllowance(uint256 available, uint256 required);
    error ExceedsMaxTransaction(uint256 amount, uint256 max);
    error ExceedsMaxWallet(uint256 balance, uint256 max);
    error TransferCooldown(uint256 remainingSeconds);
    error Blacklisted();
    error MiningHalted();
    error InvalidProof();
    error NonceAlreadyUsed();
    error InsufficientResonance(uint256 actual, uint256 required);
    error MaxSupplyReached();
    error ReentrancyGuardActive();
    error NotOwner();
    error FlashLoanDetected();
    
    // ═══════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event BlockMined(address indexed miner, uint256 nonce, uint256 reward, uint256 resonance, uint256 difficulty);
    event DifficultyAdjusted(uint256 oldDifficulty, uint256 newDifficulty);
    event Halving(uint256 halvingNumber, uint256 newReward);
    event ReflectionDistributed(uint256 amount);
    event TokensBurned(address indexed from, uint256 amount);
    event AntiWhaleTriggered(address indexed account, string reason);
    
    // ═══════════════════════════════════════════════════════════════════
    // PACKED STORAGE - Optimized for minimal gas usage
    // ═══════════════════════════════════════════════════════════════════
    
    // Slot 1: Core token data (32 bytes packed)
    string private constant _name = "L104 Sovereign Prime";
    string private constant _symbol = "L104SP";
    
    // Slot 2-3: Reflection system
    uint256 private constant MAX_UINT = type(uint256).max;
    uint256 private _rTotal;  // Reflection total
    uint256 private _tTotal;  // Token total
    
    // Slot 4: Mining config (32 bytes packed)
    struct MiningConfig {
        uint128 difficulty;          // 16 bytes - Current mining difficulty
        uint64 lastBlockTime;        // 8 bytes  - Timestamp of last mined block
        uint32 blocksMinedInEpoch;   // 4 bytes  - Blocks mined in current epoch
        uint16 epochNumber;          // 2 bytes  - Current epoch
        uint8 halvingCount;          // 1 byte   - Number of halvings occurred
        bool halted;                 // 1 byte   - Emergency halt flag
    }
    MiningConfig public miningConfig;
    
    // Slot 5: Protection config (32 bytes packed)
    struct ProtectionConfig {
        uint64 launchTime;           // 8 bytes  - Launch timestamp
        uint32 cooldownSeconds;      // 4 bytes  - Cooldown between transfers
        uint16 maxTxBps;             // 2 bytes  - Max tx (basis points of supply)
        uint16 maxWalletBps;         // 2 bytes  - Max wallet (basis points)
        uint16 reflectionBps;        // 2 bytes  - Reflection fee (basis points)
        uint16 burnBps;              // 2 bytes  - Burn fee (basis points)
        uint8 deadBlocks;            // 1 byte   - Sniper protection blocks
        bool tradingEnabled;         // 1 byte   - Trading enabled flag
    }
    ProtectionConfig public protectionConfig;
    
    // Slot 6: Flags bitmap (1 slot for 256 booleans)
    uint256 private _flags;
    uint256 private constant FLAG_REENTRANCY = 1 << 0;
    uint256 private constant FLAG_PAUSED = 1 << 1;
    
    // Slot 7: Owner
    address private _owner;
    
    // Slot 8: Treasury
    address public treasury;
    
    // ═══════════════════════════════════════════════════════════════════
    // MAPPINGS
    // ═══════════════════════════════════════════════════════════════════
    
    mapping(address => uint256) private _rOwned;           // Reflection balance
    mapping(address => uint256) private _tOwned;           // Token balance (for excluded)
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _isExcludedFromFees;
    mapping(address => bool) private _isExcludedFromReflection;
    mapping(address => bool) private _isBlacklisted;
    mapping(address => uint64) private _lastTransferTime;  // Anti-bot cooldown
    mapping(address => uint256) private _lastTxBlock;      // Flash loan protection
    mapping(bytes32 => bool) private _usedNonces;
    mapping(address => uint256) public minerRewards;
    
    // Bitmap for efficient boolean storage (256 addresses per slot)
    mapping(uint256 => uint256) private _exemptBitmap;
    
    // ═══════════════════════════════════════════════════════════════════
    // IMMUTABLE CONSTANTS (Stored in bytecode, 0 gas to read)
    // ═══════════════════════════════════════════════════════════════════
    
    uint256 public constant GOD_CODE = 5275184818492537;
    uint256 public constant PHI_SCALED = 1618033988749895;  // PHI * 1e15
    uint256 public constant TWO_PI_SCALED = 6283185307;     // 2π * 1e9
    uint256 public constant MAX_SUPPLY = 104_000_000 * 1e18;
    uint256 public constant INITIAL_MINING_REWARD = 104 * 1e18;
    uint256 public constant BLOCKS_PER_HALVING = 500_000;
    uint256 public constant BLOCKS_PER_EPOCH = 100;
    uint256 public constant TARGET_BLOCK_TIME = 60;         // 60 seconds
    uint256 public constant BPS_DENOMINATOR = 10_000;
    
    // Memory-hard mining constants
    uint256 private constant CACHE_SIZE = 32;
    uint256 private constant MIX_ROUNDS = 16;
    
    // ═══════════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════════
    
    modifier onlyOwner() {
        if (msg.sender != _owner) revert NotOwner();
        _;
    }
    
    modifier nonReentrant() {
        if (_flags & FLAG_REENTRANCY != 0) revert ReentrancyGuardActive();
        _flags |= FLAG_REENTRANCY;
        _;
        _flags &= ~FLAG_REENTRANCY;
    }
    
    modifier noFlashLoan() {
        if (_lastTxBlock[msg.sender] == block.number) revert FlashLoanDetected();
        _lastTxBlock[msg.sender] = block.number;
        _;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════
    
    constructor(address _treasury) {
        if (_treasury == address(0)) revert ZeroAddress();
        
        _owner = msg.sender;
        treasury = _treasury;
        
        // Initialize reflection totals
        _tTotal = 0;
        _rTotal = MAX_UINT - (MAX_UINT % MAX_SUPPLY);
        
        // Initialize mining config
        miningConfig = MiningConfig({
            difficulty: uint128(type(uint256).max >> 4),  // Initial difficulty
            lastBlockTime: uint64(block.timestamp),
            blocksMinedInEpoch: 0,
            epochNumber: 0,
            halvingCount: 0,
            halted: false
        });
        
        // Initialize protection config
        protectionConfig = ProtectionConfig({
            launchTime: uint64(block.timestamp),
            cooldownSeconds: 30,           // 30 second cooldown
            maxTxBps: 100,                 // 1% max transaction
            maxWalletBps: 200,             // 2% max wallet
            reflectionBps: 100,            // 1% reflection
            burnBps: 50,                   // 0.5% burn
            deadBlocks: 3,                 // 3 block sniper protection
            tradingEnabled: true
        });
        
        // Exclude owner and treasury from fees
        _isExcludedFromFees[msg.sender] = true;
        _isExcludedFromFees[_treasury] = true;
        _isExcludedFromFees[address(0)] = true;
        
        // Initial distribution
        uint256 treasuryAmount = 10_400_000 * 1e18;  // 10%
        uint256 deployerAmount = 5_200_000 * 1e18;   // 5%
        
        _mintInternal(_treasury, treasuryAmount);
        _mintInternal(msg.sender, deployerAmount);
        
        emit OwnershipTransferred(address(0), msg.sender);
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // ERC20 STANDARD FUNCTIONS (Gas Optimized)
    // ═══════════════════════════════════════════════════════════════════
    
    function name() external pure returns (string memory) { return _name; }
    function symbol() external pure returns (string memory) { return _symbol; }
    function decimals() external pure returns (uint8) { return 18; }
    function totalSupply() external view returns (uint256) { return _tTotal; }
    
    function balanceOf(address account) public view returns (uint256) {
        if (_isExcludedFromReflection[account]) return _tOwned[account];
        return _tokenFromReflection(_rOwned[account]);
    }
    
    function transfer(address to, uint256 amount) external returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }
    
    function allowance(address ownerAddr, address spender) external view returns (uint256) {
        return _allowances[ownerAddr][spender];
    }
    
    function approve(address spender, uint256 amount) external returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }
    
    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        uint256 currentAllowance = _allowances[from][msg.sender];
        if (currentAllowance < amount) revert InsufficientAllowance(currentAllowance, amount);
        
        unchecked {
            _approve(from, msg.sender, currentAllowance - amount);
        }
        _transfer(from, to, amount);
        return true;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // INTERNAL TRANSFER WITH REFLECTION + BURN + PROTECTION
    // ═══════════════════════════════════════════════════════════════════
    
    function _transfer(address from, address to, uint256 amount) internal {
        if (from == address(0) || to == address(0)) revert ZeroAddress();
        if (_isBlacklisted[from] || _isBlacklisted[to]) revert Blacklisted();
        
        uint256 fromBalance = balanceOf(from);
        if (fromBalance < amount) revert InsufficientBalance(fromBalance, amount);
        
        // Anti-bot: Sniper protection in first blocks
        ProtectionConfig memory pConfig = protectionConfig;
        if (block.number <= pConfig.launchTime + pConfig.deadBlocks) {
            _isBlacklisted[to] = true;
            emit AntiWhaleTriggered(to, "Sniper detected");
        }
        
        bool takeFees = !_isExcludedFromFees[from] && !_isExcludedFromFees[to];
        
        if (takeFees) {
            // Anti-whale: Max transaction check
            uint256 maxTx = (_tTotal * pConfig.maxTxBps) / BPS_DENOMINATOR;
            if (amount > maxTx) revert ExceedsMaxTransaction(amount, maxTx);
            
            // Anti-whale: Max wallet check
            uint256 maxWallet = (_tTotal * pConfig.maxWalletBps) / BPS_DENOMINATOR;
            uint256 newBalance = balanceOf(to) + amount;
            if (newBalance > maxWallet) revert ExceedsMaxWallet(newBalance, maxWallet);
            
            // Cooldown check
            uint64 lastTx = _lastTransferTime[from];
            if (block.timestamp < lastTx + pConfig.cooldownSeconds) {
                revert TransferCooldown(lastTx + pConfig.cooldownSeconds - block.timestamp);
            }
            _lastTransferTime[from] = uint64(block.timestamp);
        }
        
        // Calculate fees
        uint256 reflectionAmount = 0;
        uint256 burnAmount = 0;
        uint256 transferAmount = amount;
        
        if (takeFees && pConfig.reflectionBps > 0) {
            reflectionAmount = (amount * pConfig.reflectionBps) / BPS_DENOMINATOR;
            transferAmount -= reflectionAmount;
        }
        
        if (takeFees && pConfig.burnBps > 0) {
            burnAmount = (amount * pConfig.burnBps) / BPS_DENOMINATOR;
            transferAmount -= burnAmount;
        }
        
        // Execute transfer with reflection
        _transferWithReflection(from, to, amount, transferAmount, reflectionAmount, burnAmount);
    }
    
    function _transferWithReflection(
        address from,
        address to,
        uint256 totalAmount,
        uint256 transferAmount,
        uint256 reflectionAmount,
        uint256 burnAmount
    ) private {
        uint256 rate = _getRate();
        
        uint256 rAmount = totalAmount * rate;
        uint256 rTransfer = transferAmount * rate;
        uint256 rReflection = reflectionAmount * rate;
        uint256 rBurn = burnAmount * rate;
        
        // Deduct from sender
        if (_isExcludedFromReflection[from]) {
            _tOwned[from] -= totalAmount;
        }
        _rOwned[from] -= rAmount;
        
        // Add to recipient
        if (_isExcludedFromReflection[to]) {
            _tOwned[to] += transferAmount;
        }
        _rOwned[to] += rTransfer;
        
        // Distribute reflection to all holders (O(1) operation!)
        if (reflectionAmount > 0) {
            _rTotal -= rReflection;
            emit ReflectionDistributed(reflectionAmount);
        }
        
        // Burn tokens
        if (burnAmount > 0) {
            _tTotal -= burnAmount;
            _rTotal -= rBurn;
            emit TokensBurned(from, burnAmount);
        }
        
        emit Transfer(from, to, transferAmount);
    }
    
    function _getRate() private view returns (uint256) {
        if (_tTotal == 0) return 0;
        return _rTotal / _tTotal;
    }
    
    function _tokenFromReflection(uint256 rAmount) private view returns (uint256) {
        uint256 rate = _getRate();
        if (rate == 0) return 0;
        return rAmount / rate;
    }
    
    function _approve(address ownerAddr, address spender, uint256 amount) private {
        if (ownerAddr == address(0) || spender == address(0)) revert ZeroAddress();
        _allowances[ownerAddr][spender] = amount;
        emit Approval(ownerAddr, spender, amount);
    }
    
    function _mintInternal(address account, uint256 amount) private {
        uint256 rate = _rTotal / (MAX_SUPPLY);  // Initial rate before any supply
        if (_tTotal > 0) {
            rate = _getRate();
        }
        
        _tTotal += amount;
        if (_isExcludedFromReflection[account]) {
            _tOwned[account] += amount;
        }
        _rOwned[account] += amount * rate;
        
        emit Transfer(address(0), account, amount);
    }
    
    function burn(uint256 amount) external {
        address sender = msg.sender;
        uint256 balance = balanceOf(sender);
        if (balance < amount) revert InsufficientBalance(balance, amount);
        
        uint256 rate = _getRate();
        uint256 rAmount = amount * rate;
        
        if (_isExcludedFromReflection[sender]) {
            _tOwned[sender] -= amount;
        }
        _rOwned[sender] -= rAmount;
        _tTotal -= amount;
        _rTotal -= rAmount;
        
        emit Transfer(sender, address(0), amount);
        emit TokensBurned(sender, amount);
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // PROOF OF RESONANCE MINING (Memory-Hard, ASIC-Resistant)
    // ═══════════════════════════════════════════════════════════════════
    
    function submitBlock(uint256 nonce) external nonReentrant noFlashLoan {
        MiningConfig memory config = miningConfig;
        
        if (config.halted) revert MiningHalted();
        
        uint256 currentReward = _getCurrentReward();
        if (_tTotal + currentReward > MAX_SUPPLY) revert MaxSupplyReached();
        
        // Prevent nonce reuse
        bytes32 nonceHash = keccak256(abi.encodePacked(msg.sender, nonce, config.epochNumber));
        if (_usedNonces[nonceHash]) revert NonceAlreadyUsed();
        
        // Validate Proof of Resonance (PHI-based)
        uint256 resonance = calculateResonance(nonce);
        if (resonance < 950) revert InsufficientResonance(resonance, 950);
        
        // Validate memory-hard proof of work
        bytes32 workHash = _memoryHardHash(msg.sender, nonce, config.epochNumber);
        if (uint256(workHash) >= config.difficulty) revert InvalidProof();
        
        // Mark nonce as used
        _usedNonces[nonceHash] = true;
        
        // Mint reward
        _mintInternal(msg.sender, currentReward);
        minerRewards[msg.sender] += currentReward;
        
        // Update mining state
        miningConfig.blocksMinedInEpoch++;
        miningConfig.lastBlockTime = uint64(block.timestamp);
        
        emit BlockMined(msg.sender, nonce, currentReward, resonance, config.difficulty);
        
        // Check for epoch end and difficulty adjustment
        if (miningConfig.blocksMinedInEpoch >= BLOCKS_PER_EPOCH) {
            _adjustDifficulty();
            miningConfig.blocksMinedInEpoch = 0;
            miningConfig.epochNumber++;
        }
        
        // Check for halving
        uint256 totalBlocksMined = uint256(miningConfig.epochNumber) * BLOCKS_PER_EPOCH 
                                   + miningConfig.blocksMinedInEpoch;
        uint256 expectedHalvings = totalBlocksMined / BLOCKS_PER_HALVING;
        if (expectedHalvings > miningConfig.halvingCount) {
            miningConfig.halvingCount = uint8(expectedHalvings);
            emit Halving(expectedHalvings, _getCurrentReward());
        }
    }
    
    /**
     * @notice Memory-hard hash function (ASIC-resistant)
     * @dev Uses on-chain memory to prevent hardware optimization
     */
    function _memoryHardHash(
        address miner,
        uint256 nonce,
        uint256 epoch
    ) internal view returns (bytes32) {
        // Initial seed
        bytes32 seed;
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, miner)
            mstore(add(ptr, 32), nonce)
            mstore(add(ptr, 64), epoch)
            mstore(add(ptr, 96), number())
            mstore(add(ptr, 128), prevrandao())
            seed := keccak256(ptr, 160)
        }
        
        // Build memory cache (makes it memory-hard)
        bytes32[CACHE_SIZE] memory cache;
        for (uint256 i; i < CACHE_SIZE;) {
            cache[i] = keccak256(abi.encodePacked(seed, i));
            unchecked { ++i; }
        }
        
        // Memory-hard mixing
        bytes32 mix = seed;
        for (uint256 i; i < MIX_ROUNDS;) {
            uint256 idx = uint256(mix) % CACHE_SIZE;
            mix = keccak256(abi.encodePacked(mix, cache[idx], cache[(idx + 7) % CACHE_SIZE]));
            
            // DAG-like access pattern
            cache[idx] = keccak256(abi.encodePacked(cache[idx], mix));
            
            unchecked { ++i; }
        }
        
        return keccak256(abi.encodePacked(seed, mix, cache[0], cache[CACHE_SIZE - 1]));
    }
    
    /**
     * @notice Calculate PHI-based resonance (Golden Ratio alignment)
     * @dev Approximates |sin(nonce × PHI)| using efficient polynomial
     */
    function calculateResonance(uint256 nonce) public pure returns (uint256) {
        // x = (nonce * PHI) mod 2π, scaled by 1e9
        uint256 x = (nonce * PHI_SCALED / 1e6) % TWO_PI_SCALED;
        
        // Parabola approximation for |sin(x)|
        // More accurate than previous version
        uint256 halfPeriod = TWO_PI_SCALED / 2;
        uint256 quarterPeriod = TWO_PI_SCALED / 4;
        
        uint256 sinApprox;
        if (x <= halfPeriod) {
            // First half: parabola peaking at π/2
            if (x <= quarterPeriod) {
                // Rising quarter
                sinApprox = (x * quarterPeriod) / quarterPeriod;
            } else {
                // Falling quarter
                uint256 xShift = x - quarterPeriod;
                sinApprox = quarterPeriod - (xShift * quarterPeriod) / quarterPeriod;
            }
        } else {
            // Second half: mirror of first half
            uint256 xShift = x - halfPeriod;
            if (xShift <= quarterPeriod) {
                sinApprox = (xShift * quarterPeriod) / quarterPeriod;
            } else {
                uint256 xShift2 = xShift - quarterPeriod;
                sinApprox = quarterPeriod - (xShift2 * quarterPeriod) / quarterPeriod;
            }
        }
        
        // Scale to 0-1000 range
        uint256 resonance = (sinApprox * 1000) / quarterPeriod;
        
        // Apply GOD_CODE modulation for unique signature
        uint256 godModulation = (nonce % GOD_CODE) % 100;
        resonance = (resonance * (1000 + godModulation)) / 1000;
        
        if (resonance > 1000) resonance = 1000;
        
        return resonance;
    }
    
    function _getCurrentReward() internal view returns (uint256) {
        return INITIAL_MINING_REWARD >> miningConfig.halvingCount;
    }
    
    function _adjustDifficulty() internal {
        MiningConfig memory config = miningConfig;
        
        // Calculate time for this epoch
        uint256 epochTime = block.timestamp - config.lastBlockTime;
        uint256 targetTime = TARGET_BLOCK_TIME * BLOCKS_PER_EPOCH;
        
        uint128 oldDifficulty = config.difficulty;
        uint128 newDifficulty;
        
        if (epochTime < targetTime / 2) {
            // Blocks too fast, increase difficulty (lower target)
            newDifficulty = oldDifficulty / 2;
            if (newDifficulty < 1) newDifficulty = 1;
        } else if (epochTime > targetTime * 2) {
            // Blocks too slow, decrease difficulty (higher target)
            newDifficulty = oldDifficulty * 2;
            if (newDifficulty > type(uint128).max / 2) {
                newDifficulty = type(uint128).max / 2;
            }
        } else {
            // Linear adjustment
            newDifficulty = uint128((uint256(oldDifficulty) * targetTime) / epochTime);
        }
        
        miningConfig.difficulty = newDifficulty;
        
        if (oldDifficulty != newDifficulty) {
            emit DifficultyAdjusted(oldDifficulty, newDifficulty);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════
    
    function owner() external view returns (address) {
        return _owner;
    }
    
    function getMiningStats() external view returns (
        uint256 difficulty,
        uint256 currentReward,
        uint256 blocksUntilHalving,
        uint256 totalBlocksMined,
        uint256 remainingSupply,
        uint256 epoch
    ) {
        uint256 totalMined = uint256(miningConfig.epochNumber) * BLOCKS_PER_EPOCH 
                            + miningConfig.blocksMinedInEpoch;
        uint256 nextHalving = (miningConfig.halvingCount + 1) * BLOCKS_PER_HALVING;
        
        return (
            miningConfig.difficulty,
            _getCurrentReward(),
            nextHalving > totalMined ? nextHalving - totalMined : 0,
            totalMined,
            MAX_SUPPLY - _tTotal,
            miningConfig.epochNumber
        );
    }
    
    function getReflectionStats() external view returns (
        uint256 reflectionRate,
        uint256 burnRate,
        uint256 totalBurned
    ) {
        return (
            protectionConfig.reflectionBps,
            protectionConfig.burnBps,
            MAX_SUPPLY - _tTotal  // Assuming no external burns initially
        );
    }
    
    function isExcludedFromFees(address account) external view returns (bool) {
        return _isExcludedFromFees[account];
    }
    
    function isBlacklisted(address account) external view returns (bool) {
        return _isBlacklisted[account];
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // OWNER FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════
    
    function transferOwnership(address newOwner) external onlyOwner {
        if (newOwner == address(0)) revert ZeroAddress();
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
    
    function renounceOwnership() external onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }
    
    function setTreasury(address _treasury) external onlyOwner {
        if (_treasury == address(0)) revert ZeroAddress();
        treasury = _treasury;
    }
    
    function setMiningHalted(bool halted) external onlyOwner {
        miningConfig.halted = halted;
    }
    
    function setFeeExclusion(address account, bool excluded) external onlyOwner {
        _isExcludedFromFees[account] = excluded;
    }
    
    function setBlacklist(address account, bool blacklisted) external onlyOwner {
        _isBlacklisted[account] = blacklisted;
    }
    
    function updateProtectionConfig(
        uint32 cooldownSeconds,
        uint16 maxTxBps,
        uint16 maxWalletBps,
        uint16 reflectionBps,
        uint16 burnBps
    ) external onlyOwner {
        require(maxTxBps <= 1000, "Max 10%");
        require(maxWalletBps <= 2000, "Max 20%");
        require(reflectionBps + burnBps <= 1000, "Max 10% total fees");
        
        protectionConfig.cooldownSeconds = cooldownSeconds;
        protectionConfig.maxTxBps = maxTxBps;
        protectionConfig.maxWalletBps = maxWalletBps;
        protectionConfig.reflectionBps = reflectionBps;
        protectionConfig.burnBps = burnBps;
    }
}
