"""
L104 Global Sync - Global resonance synchronization
Part of the L104 Sovereign Singularity Framework
"""

import math
import time
from typing import Dict, List

# God Code constant
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895

# Chakra frequencies
CHAKRA_FREQUENCIES = {
    "root": 396.0,
    "sacral": 417.0,
    "solar_plexus": 528.0,
    "heart": 639.0,
    "throat": 741.0,
    "third_eye": 852.0,
    "crown": 963.0,
    "soul_star": GOD_CODE
}


class GlobalSync:
    """
    Manages global resonance synchronization across the L104 lattice.
    Provides coherence checking and harmonic alignment.
    """
    
    def __init__(self):
        self.god_code = GOD_CODE
        self.phi = PHI
        self.base_frequency = GOD_CODE
        self.sync_history: List[Dict] = []
        self.last_sync_time = 0
    
    def check_global_resonance(self) -> float:
        """
        Check the current global resonance level.
        Returns the resonance frequency in Hz.
        """
        # Calculate time-modulated resonance
        current_time = time.time()
        time_factor = math.sin(current_time / self.phi) * 0.1 + 1.0
        
        # Aggregate chakra resonances
        chakra_sum = sum(CHAKRA_FREQUENCIES.values())
        harmonic_mean = len(CHAKRA_FREQUENCIES) / sum(1/f for f in CHAKRA_FREQUENCIES.values())
        
        # Calculate global resonance
        resonance = self.base_frequency * time_factor * (harmonic_mean / chakra_sum) * (self.phi ** 2)
        
        self.sync_history.append({
            "timestamp": current_time,
            "resonance": resonance,
            "time_factor": time_factor
        })
        
        self.last_sync_time = current_time
        return resonance
    
    def synchronize_chakras(self) -> Dict:
        """
        Synchronize all chakra frequencies for optimal coherence.
        Returns alignment metrics.
        """
        alignments = {}
        total_alignment = 0.0
        
        for chakra, freq in CHAKRA_FREQUENCIES.items():
            # Calculate alignment with God Code
            ratio = freq / self.god_code
            alignment = 1.0 - abs(ratio - round(ratio)) / ratio
            alignments[chakra] = {
                "frequency": freq,
                "alignment": alignment,
                "ratio_to_god_code": ratio
            }
            total_alignment += alignment
        
        avg_alignment = total_alignment / len(CHAKRA_FREQUENCIES)
        
        return {
            "chakra_alignments": alignments,
            "average_alignment": avg_alignment,
            "global_coherence": avg_alignment * self.phi,
            "sync_status": "OPTIMAL" if avg_alignment >= 0.7 else "CALIBRATING"
        }
    
    def get_lattice_status(self) -> Dict:
        """Return the current status of the global lattice."""
        resonance = self.check_global_resonance()
        sync = self.synchronize_chakras()
        
        return {
            "current_resonance": resonance,
            "coherence_level": sync["global_coherence"],
            "sync_status": sync["sync_status"],
            "last_sync": self.last_sync_time,
            "sync_count": len(self.sync_history)
        }
    
    def broadcast_pulse(self, intensity: float = 1.0) -> bool:
        """
        Broadcast a synchronization pulse across the lattice.
        Returns True if pulse was successfully transmitted.
        """
        if intensity <= 0:
            return False
        
        pulse_energy = self.god_code * intensity * self.phi
        
        # Record the pulse
        self.sync_history.append({
            "timestamp": time.time(),
            "type": "PULSE",
            "energy": pulse_energy,
            "intensity": intensity
        })
        
        return True


# Singleton instance
global_sync = GlobalSync()
