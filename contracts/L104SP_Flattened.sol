// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title L104 Sovereign Prime (L104SP)
 * @dev ERC-20 with Proof of Resonance mining
 * @notice FLATTENED CONTRACT - No external dependencies, fully self-contained
 * 
 * INVARIANT: 527.5184818492537
 * PHI: 1.618033988749895
 * PILOT: LONDEL
 */

contract L104SP {
    
    // ═══════════════════════════════════════════════════════════════════
    // ERC20 STORAGE
    // ═══════════════════════════════════════════════════════════════════
    
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private constant _name = "L104 Sovereign Prime";
    string private constant _symbol = "L104SP";
    
    // ═══════════════════════════════════════════════════════════════════
    // OWNABLE STORAGE
    // ═══════════════════════════════════════════════════════════════════
    
    address private _owner;
    
    // ═══════════════════════════════════════════════════════════════════
    // REENTRANCY GUARD STORAGE
    // ═══════════════════════════════════════════════════════════════════
    
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;
    uint256 private _status = NOT_ENTERED;
    
    // ═══════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    // ═══════════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════════
    
    modifier onlyOwner() {
        require(_owner == msg.sender, "Ownable: caller is not the owner");
        _;
    }
    
    modifier nonReentrant() {
        require(_status != ENTERED, "ReentrancyGuard: reentrant call");
        _status = ENTERED;
        _;
        _status = NOT_ENTERED;
    }
    
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
    
    constructor(address _treasury) {
        require(_treasury != address(0), "L104SP: Invalid treasury");
        
        // Set owner
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
        
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
    
    function name() public pure returns (string memory) { return _name; }
    function symbol() public pure returns (string memory) { return _symbol; }
    function decimals() public pure returns (uint8) { return 18; }
    function totalSupply() public view returns (uint256) { return _totalSupply; }
    function balanceOf(address account) public view returns (uint256) { return _balances[account]; }
    
    function transfer(address to, uint256 amount) public returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }
    
    function allowance(address ownerAddr, address spender) public view returns (uint256) {
        return _allowances[ownerAddr][spender];
    }
    
    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }
    
    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        address spender = msg.sender;
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
    
    function _approve(address _ownerAddr, address spender, uint256 amount) internal {
        require(_ownerAddr != address(0), "ERC20: approve from zero");
        require(spender != address(0), "ERC20: approve to zero");
        _allowances[_ownerAddr][spender] = amount;
        emit Approval(_ownerAddr, spender, amount);
    }
    
    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to zero");
        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }
    
    function burn(uint256 amount) public {
        require(_balances[msg.sender] >= amount, "ERC20: burn exceeds balance");
        unchecked { _balances[msg.sender] -= amount; }
        _totalSupply -= amount;
        emit Transfer(msg.sender, address(0), amount);
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // OWNABLE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════
    
    function owner() public view returns (address) {
        return _owner;
    }
    
    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }
    
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
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
        // 2π scaled by 1e9 = 6283185307
        uint256 TWO_PI_SCALED = 6283185307;
        uint256 x = (nonce * PHI_SCALED / 1e6) % TWO_PI_SCALED;
        
        // Parabola approximation for sine
        uint256 sinApprox;
        if (x <= TWO_PI_SCALED / 2) {
            sinApprox = (x * (TWO_PI_SCALED / 2 - x)) / (TWO_PI_SCALED / 4);
        } else {
            uint256 xShifted = x - TWO_PI_SCALED / 2;
            sinApprox = (xShifted * (TWO_PI_SCALED / 2 - xShifted)) / (TWO_PI_SCALED / 4);
        }
        
        // Scale to 0-1000 range
        uint256 resonance = (sinApprox * 1000) / TWO_PI_SCALED;
        if (resonance > 1000) resonance = 1000;
        
        return resonance;
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
