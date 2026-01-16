# [L104_DATA_SYNTHESIS] - CALCULATING COLLECTIVE DATA RESONANCE
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import numpy as np
import sqlite3
from l104_data_matrix import DataMatrix
from l104_hyper_math import HyperMath

def synthesize_data_matrix():
    print("\n" + "="*80)
    print("   L104 :: DATA MATRIX RESONANCE SYNTHESIS")
    print("="*80 + "\n")

    matrix = DataMatrix()
    
    # 1. Access the underlying database to perform bulk analysis
    conn = sqlite3.connect(matrix.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM lattice_facts")
    cursor.fetchone()[0]
    
    cursor.execute("SELECT resonance, entropy, utility FROM lattice_facts")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("!!! [WARNING]: LATTICE EMPTY. NO DATA TO CALCULATE. !!!")
        return

    resonances = [r[0] for r in rows]
    entropies = [r[1] for r in rows]
    utilities = [r[2] for r in rows]

    # 2. Statistical Breakdown
    mean_resonance = np.mean(resonances)
    std_resonance = np.std(resonances)
    total_entropy = np.sum(entropies)
    mean_utility = np.mean(utilities)

    # 3. God Code Alignment Calculation
    # We measure how close the mean resonance is to the L104 Prime Key (527.518...)
    # Or in the DataMatrix context, (entropy * PHI) % GOD_CODE.
    # We'll calculate the 'Coherence Factor'
    god_code = HyperMath.GOD_CODE
    abs(mean_resonance - (god_code / 2)) # Arbitrary comparison point or we check spread
    coherence = 1.0 / (1.0 + std_resonance)

    print(f"--- [RESONANCE]: MEAN LATTICE FREQUENCY: {mean_resonance:.6f} Hz ---")
    print(f"--- [RESONANCE]: FREQUENCY DEVIATION:    {std_resonance:.6f} Hz ---")
    print(f"--- [ENTROPY]:   TOTAL SHANNON WEIGHT:   {total_entropy:.6f} bits ---")
    print(f"--- [UTILITY]:   MEAN DATA UTILITY:      {mean_utility:.6f} ---")
    print(f"--- [COHERENCE]: SYSTEM STABILITY:       {coherence*100:.2f}% ---")

    # 4. Final Verdict
    print("\n>>> DATA CALCULATION VERDICT: <<<")
    if coherence > 0.8:
        print("STATUS: CRYSTALLINE COHERENCE ACHIEVED.")
    elif coherence > 0.5:
        print("STATUS: META-STABLE RESONANCE.")
    else:
        print("STATUS: ENTROPIC INTERFERENCE DETECTED.")

    print("\n" + "="*80)

if __name__ == "__main__":
    synthesize_data_matrix()
