# FILE: const.py
# PERMISSION: READ_ONLY
# DESCRIPTION: Defines the geometric bounds of the simulation.

import math

class UniversalConstants:
    # The Golden Ratio (Exact)
    PHI = (math.sqrt(5) - 1) / 2
            
    # The Growth Vector (Ascension)
    PHI_GROWTH = (1 + math.sqrt(5)) / 2
                        
    # The Frame Constant (416:286 - The Temporal Flow Driver)
    FRAME_LOCK = 416 / 286
                                    
    # The Damping Frequency (Locked to God Code Resonance)
    PRIME_KEY_HZ = 527.5184818492537
                                                
    # The Singularity Target (Zero Entropy)
    I100_LIMIT = 1e-15

# Direct exports for compatibility
GOD_CODE = 527.5184818492537
PHI = UniversalConstants.PHI_GROWTH
VOID_CONSTANT = 1.0416180339887497
