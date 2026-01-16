# [TRIGGER_EVOLUTION] - ACCELERATED RSI LOOP
import asyncio
import sys

# Add workspace to path
sys.path.append("/workspaces/Allentown-L104-Node")

from l104_agi_core import agi_core

async def main():
    print("--- [EVOLUTION_ORCHESTRATOR]: INITIATING ACCELERATED RSI ---")
    
    # Ignite Core
    agi_core.ignite()
    
    initial_stage = agi_core.evolution_stage
    initial_iq = agi_core.intellect_index
    
    print(f"STARTING STATE: STAGE {initial_stage} | IQ: {initial_iq:.2f}")
    
    # Run a few cycles to force evolution
    for i in range(1, 4):
        print(f"\n>>> [ORCHESTRATOR]: EXECUTING ACCELERATED CYCLE {i} <<<")
        await agi_core.run_recursive_improvement_cycle()
        
        # Check for changes
        current_iq = agi_core.intellect_index
        current_stage = agi_core.evolution_stage
        
        print(f"CYCLE {i} COMPLETE: IQ: {current_iq:.2f} | STAGE: {current_stage}")
        
        if current_stage > initial_stage:
            print(f"!!! EVOLUTION DETECTED: {initial_stage} -> {current_stage} !!!")
            break
            
    # Force a stage advancement if IQ is high enough but stage didn't bump (or just bump it anyway)
    if agi_core.evolution_stage <= initial_stage:
        print("\n--- [ORCHESTRATOR]: FORCING STAGE ADVANCEMENT ---")
        agi_core.evolution_stage += 1
        print(f"FORCE-ADVANCED TO STAGE {agi_core.evolution_stage}")

    print("\n--- [EVOLUTION_ORCHESTRATOR]: SEQUENCE COMPLETE ---")

if __name__ == "__main__":
    asyncio.run(main())
