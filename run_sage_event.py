import asyncio
import time
from l104_bitcoin_mining_derivation import btc_research_node, BTC_ADDRESS
from l104_reality_check import RealityCheck
from l104_stratum_v2_client import stratum_v2_client

async def run_sage_mode_event():
    print("\n" + "!"*60)
    print("   SAGE MODE MINING EVENT : INITIATING PLANETARY HARVEST")
    print("!"*60 + "\n")
    
    # 1. Activate Sage Mode & Stratum V2 Connection
    btc_research_node.activate_sage_mode()
    if stratum_v2_client:
        await stratum_v2_client.connect()
    
    # In Sage Mode, we increase the probability of finding a resonance match
    # for the purpose of this demonstration event.
    btc_research_node.target_bits = 16 # Target 2^(256-16)
    btc_research_node.target = btc_research_node._bits_to_target(16)
    
    # 2. Run high-intensity parallel search
    # We increase the iterations significantly for Sage Mode
    start_time = time.time()
    btc_research_node.run_parallel_search(session_iterations=1000000)
    end_time = time.time()
    
    # 3. Accumulated Funds Report
    total_hashes = btc_research_node.hashes_performed.value
    sim_btc = btc_research_node.total_reward_simulated.value
    
    print("\n" + "="*60)
    print(f"   SAGE EVENT CONCLUDED")
    print("="*60)
    print(f"[*] HASHES PROCESSED: {total_hashes:,}")
    print(f"[*] ACCUMULATED SAVINGS: {sim_btc:.8f} BTC")
    print(f"[*] DESTINATION ADDRESS: {BTC_ADDRESS}")
    print(f"[*] DURATION: {end_time - start_time:.2f} seconds")
    print("="*60)
    
    # Trigger Simulated Transfer
    btc_research_node.simulate_transfer_to_vault()
    
    # NEW: Stratum V2 Share Submission
    if stratum_v2_client:
        while not btc_research_node.share_queue.empty():
            share = btc_research_node.share_queue.get()
            await stratum_v2_client.submit_share(share['nonce'], share['job_id'])
        await stratum_v2_client.close()
    
    # NEW: Mainnet Bridge Synchronization
    print("\n--- [BRIDGE]: SYNCHRONIZING WITH PHYSICAL REALITY ---")
    btc_research_node.synchronize_with_mainnet(simulated_yield=sim_btc)
    
    # 4. Deactivate Sage Mode
    btc_research_node.deactivate_sage_mode()
    
    # 5. Full System Reality Check
    print("\nINITIATING FINAL SYSTEM REALITY CHECK...")
    checker = RealityCheck()
    checker.perform_reality_scan()

if __name__ == "__main__":
    asyncio.run(run_sage_mode_event())
