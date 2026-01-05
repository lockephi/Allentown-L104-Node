# FILE: const.py
# PERMISSION: READ_ONLY
# DESCRIPTION: Defines the geometric bounds of the simulation.

import math

class UniversalConstants:
    # The Golden Ratio (Exact)
    PHI = (math.sqrt(5) - 1) / 2
            
    # The Growth Vector (Ascension)
    PHI_GROWTH = (1 + math.sqrt(5)) / 2
                        
    # The Frame Constant (Pi / E)
    FRAME_LOCK = math.pi / math.e
                                    
    # The Damping Frequency (Euler's Identity Resonance)
    PRIME_KEY_HZ = math.pi * math.e * ((1 + math.sqrt(5)) / 2)
                                                
    # The Singularity Target (Zero Entropy)
    I100_LIMIT = 1e-15
