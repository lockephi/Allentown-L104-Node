# ══════════════════════════════════════════════════════════════════════════════
# L104_NEURAL_RESONANCE_MAP - Iron Crystalline Neural Mapping
# Ferromagnetic domain infiltration model
# ══════════════════════════════════════════════════════════════════════════════

import math
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Core iron constants
try:
    from l104_core import GOD_CODE, PHI, FE_ATOMIC_NUMBER, FE_CURIE_TEMP
except ImportError:
    GOD_CODE = 527.5184818492611
    PHI = 1.618033988749895
    FE_ATOMIC_NUMBER = 26
    FE_CURIE_TEMP = 1043

FE_LATTICE = 286.65
PHI_CONJUGATE = 1 / PHI

class NeuralResonanceMap:
    """
    Maps ferromagnetic domain infiltration patterns.
    Models how magnetic ordering propagates through neural substrate.
    """

    def __init__(self, presence: float = 0.0137325049):
        self.presence = presence / 100
        self.total_nodes = 104000
        self.unbound_nodes = int(self.total_nodes * self.presence)
        # Iron-weighted node distribution
        self.domain_size = FE_ATOMIC_NUMBER

    def generate_map(self):
        print("\\n" + "═" * 60)
        print("   L104 :: IRON CRYSTALLINE RESONANCE MAP")
        print(f"   MAGNETIC DOMAINS: {self.unbound_nodes} / {self.total_nodes}")
        print("═" * 60 + "\\n")

        # BCC lattice-inspired grid (2 atoms per unit cell → factor of 2)
        grid_size = 32
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Seed with iron atomic number for crystallographic ordering
        np.random.seed(FE_ATOMIC_NUMBER)
        indices = np.random.choice(grid_size * grid_size, self.unbound_nodes, replace=False)

        flat_grid = grid.flatten()
        flat_grid[indices] = 1
        visual_grid = flat_grid.reshape((grid_size, grid_size))

        print("[FERROMAGNETIC DOMAIN GRID (32x32)]")
        print(" · = Paramagnetic (disordered)")
        print(" ◆ = Ferromagnetic (aligned)\\n")

        for row in visual_grid:
            line = "".join([" ◆ " if val == 1 else " · " for val in row])
            print(line)

        print(f"\\n[*] TOTAL NODES: {self.total_nodes}")
        print(f"[*] ALIGNED DOMAINS: {self.unbound_nodes}")
        print(f"[*] ORDER PARAMETER: {self.presence:.8f}")
        print(f"[*] CURIE TEMP: {FE_CURIE_TEMP} K")


if __name__ == "__main__":
    mapper = NeuralResonanceMap()
    mapper.generate_map()
