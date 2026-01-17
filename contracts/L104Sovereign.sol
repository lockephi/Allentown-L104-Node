// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title L104 Sovereign Token (L104S)
 * @dev BEP-20 token representing the economic value of the Allentown-L104 AGI Node.
 */
interface IBEP20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

contract L104Sovereign is IBEP20 {
    string public constant name = "L104 Sovereign";
    string public constant symbol = "L104S";
    uint8 public constant decimals = 18;
    
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    uint256 private _totalSupply;
    address public owner;
    
    // The AGI Node's owner address
    address public constant ALLENTOWN_VAULT = 0x1896f828306215C0B8198f4eF55f70081FD11a86;

    constructor() {
        owner = msg.sender;
        uint256 initialSupply = 104000000 * (10 ** uint256(decimals));
        _mint(ALLENTOWN_VAULT, initialSupply);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "L104: NOT_OWNER");
        _;
    }

    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function allowance(address ownerAddr, address spender) public view override returns (uint256) {
        return _allowances[ownerAddr][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount);
        
        uint256 currentAllowance = _allowances[sender][msg.sender];
        require(currentAllowance >= amount, "L104: ALLOWANCE_EXCEEDED");
        _approve(sender, msg.sender, currentAllowance - amount);
        
        return true;
    }

    function burn(uint256 amount) public {
        _burn(msg.sender, amount);
    }

    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "L104: FROM_ZERO_ADDRESS");
        require(recipient != address(0), "L104: TO_ZERO_ADDRESS");
        require(_balances[sender] >= amount, "L104: INSUFFICIENT_BALANCE");

        _balances[sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(sender, recipient, amount);
    }

    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "L104: MINT_TO_ZERO");
        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    function _burn(address account, uint256 amount) internal {
        require(account != address(0), "L104: BURN_FROM_ZERO");
        require(_balances[account] >= amount, "L104: BURN_EXCEEDS_BALANCE");

        _balances[account] -= amount;
        _totalSupply -= amount;
        emit Transfer(account, address(0), amount);
    }

    function _approve(address ownerAddr, address spender, uint256 amount) internal {
        require(ownerAddr != address(0), "L104: APPROVE_FROM_ZERO");
        require(spender != address(0), "L104: APPROVE_TO_ZERO");

        _allowances[ownerAddr][spender] = amount;
        emit Approval(ownerAddr, spender, amount);
    }
}
