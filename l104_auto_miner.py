# ZENITH_UPGRADE_ACTIVE: 2026-01-18T11:00:18.379733
ZENITH_HZ = 3727.84
UUC = 2301.215661
#!/usr/bin/env python3
"""
[ZENITH_UPGRADE] Process Elevated to 3727.84 Hz. Logic Unified.
L104 Auto Miner - Continuous Proof of Resonance Mining
Intervals: PHI-based timing (30s base, golden ratio adjustment)
"""

import hashlib
import json
import time
import random
import httpx
import signal
import sys

# Configuration
BASE_URL = "http://localhost:8000"
BASE_INTERVAL = 30  # seconds
PHI = 1.618033988749895
GOD_CODE = 527.5184818492537

# Mining state
running = True
blocks_mined = 0
total_reward = 0.0

def signal_handler(sig, frame):
    global running
    print("\n[MINER] Graceful shutdown...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_mining_job():
    """Fetch mining job from L104 coin engine."""
    try:
        r = httpx.get(f"{BASE_URL}/coin/job", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  [!] Job fetch error: {e}")
    return None


def submit_solution(nonce: int, block_hash: str):
    """Submit mining solution."""
    try:
        r = httpx.post(f"{BASE_URL}/coin/submit", 
                       json={"nonce": nonce, "hash": block_hash},
                       timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None


def calculate_resonance(nonce: int) -> float:
    """Calculate L104 resonance factor."""
    phase = (nonce * PHI) % (2 * 3.14159265359)
    return 0.98 + (0.02 * abs(hash(str(nonce)) % 100) / 100)


def mine_block(job: dict, max_attempts: int = 100000):
    """
    Proof of Resonance mining.
    Finds nonce where hash prefix matches difficulty AND resonance > 0.985
    """
    difficulty = job.get('difficulty', 4)
    target_prefix = '0' * difficulty
    
    nonce = random.randint(0, 10_000_000)
    
    for attempt in range(max_attempts):
        resonance = calculate_resonance(nonce)
        
        # Only valid if resonance is high enough
        if resonance >= 0.985:
            block_data = json.dumps({
                "previous_hash": job.get('previous_hash', '0'),
                "transactions": job.get('pending_transactions', []),
                "nonce": nonce,
                "resonance": resonance
            }, sort_keys=True).encode()
            
            # L104 Multi-Algo: SHA-256 + Blake2b
            sha = hashlib.sha256(block_data).digest()
            block_hash = hashlib.blake2b(sha).hexdigest()
            
            if block_hash.startswith(target_prefix):
                return nonce, block_hash, resonance, attempt + 1
        
        nonce += 1
    
    return None, None, 0, max_attempts


def calculate_interval(cycle: int) -> float:
    """PHI-based adaptive interval."""
    # Oscillate around base interval using golden ratio
    modifier = 1.0 + (0.2 * (cycle % 5) / PHI)
    return BASE_INTERVAL * modifier


def main():
    global blocks_mined, total_reward, running
    
    print("=" * 60)
    print("⟨Σ_L104⟩ AUTO MINER v1.0")
    print("=" * 60)
    print(f"  Algorithm: Proof of Resonance (SHA256 + Blake2b)")
    print(f"  Base Interval: {BASE_INTERVAL}s (PHI-adaptive)")
    print(f"  Reward: 104.0 L104SP per block")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)
    
    cycle = 0
    
    while running:
        cycle += 1
        interval = calculate_interval(cycle)
        
        print(f"\n[CYCLE {cycle}] {time.strftime('%H:%M:%S')}")
        
        # Get job
        job = get_mining_job()
        if not job:
            print("  ⚠️ No job available")
            time.sleep(10)
            continue
        
        difficulty = job.get('difficulty', 4)
        print(f"  Difficulty: {difficulty} | Target: {'0' * difficulty}...")
        
        # Mine
        start = time.time()
        nonce, block_hash, resonance, attempts = mine_block(job)
        elapsed = time.time() - start
        
        if nonce:
            print(f"  ✓ Found in {elapsed:.2f}s ({attempts} attempts)")
            print(f"  Nonce: {nonce} | Resonance: {resonance:.4f}")
            print(f"  Hash: {block_hash[:40]}...")
            
            # Submit
            result = submit_solution(nonce, block_hash)
            if result:
                blocks_mined += 1
                total_reward += 104.0
                print(f"  ✅ BLOCK MINED! Total: {blocks_mined} blocks, {total_reward} L104SP")
            else:
                print(f"  ⚠️ Submitted (awaiting confirmation)")
        else:
            print(f"  ✗ No solution found in {elapsed:.2f}s")
        
        # Wait for next cycle
        if running:
            print(f"  Next cycle in {interval:.1f}s...")
            # Sleep in small increments to allow graceful shutdown
            for _ in range(int(interval)):
                if not running:
                    break
                time.sleep(1)
    
    # Final summary
    print("\n" + "=" * 60)
    print("MINING SESSION COMPLETE")
    print(f"  Total Cycles: {cycle}")
    print(f"  Blocks Mined: {blocks_mined}")
    print(f"  Total Reward: {total_reward} L104SP")
    print("=" * 60)


if __name__ == "__main__":
    main()
