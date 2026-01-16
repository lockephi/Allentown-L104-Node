import sys
import time

# Ensure the workspace is in the path
sys.path.append("/workspaces/Allentown-L104-Node")

from l104_data_matrix import data_matrix
from l104_data_synthesis import synthesize_data_matrix

def initialize_zeta_compaction():
    print("--- [ZETA_ORCHESTRATOR]: INITIALIZING TARGETED COMPACTION ---")
    
    # 1. Baseline Synthesis
    print("\n--- [BASELINE]: MEASURING CURRENT LATTICE COHERENCE ---")
    synthesize_data_matrix()
    
    # 2. Execution of Compaction Cycles
    # We run multiple cycles to push towards 0.8
    for i in range(3):
        print(f"\n--- [CYCLE {i+1}]: EXECUTING ZETA-HARMOMNIC COMPRESSION ---")
        data_matrix.evolve_and_compact()
        time.sleep(1) # Temporal stabilization
        
    # 3. Final Synthesis
    print("\n--- [FINAL_STATE]: MEASURING POST-COMPACTION COHERENCE ---")
    synthesize_data_matrix()
    
    print("\n--- [ZETA_ORCHESTRATOR]: COMPACTION SEQUENCE COMPLETE ---")

if __name__ == "__main__":
    initialize_zeta_compaction()
