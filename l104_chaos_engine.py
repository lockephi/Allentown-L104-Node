# [L104_CHAOS_ENGINE] - Turns Chaos into Power
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

from physics_constants import Laws
def transmute_chaos(chaos_vector: float) -> float:
    """
    RECTIFICATION:
    Takes negative chaos, flips it positive (Abs),
    and refines it through the Steel Frame.
    """
    # 1. RECTIFICATION (Abs)
    raw_power = abs(chaos_vector)
    
    # 2. REFINEMENT (Steel Frame Damping)
    refined_fuel = raw_power * Laws.FRAME_CONSTANT
    
    return refined_fuelif __name__ == "__main__":
    # Test the transmutationtest_chaos = -104.0
    power = transmute_chaos(test_chaos)
    print(f"--- [CHAOS_ENGINE]: TRANSMUTED {test_chaos} -> {power:.4f} POWER ---")
