"""
L104 Science Engine — 26Q Entropy Reversal Integration
═══════════════════════════════════════════════════════════════════════════════
Integrates 26-qubit Fe-mapped entropy reversal units into the Science Engine.

Usage:
    from l104_science_engine import ScienceEngine

    se = ScienceEngine()

    # Create 26Q entropy reversal unit
    eru = se.entropy.create_26q_unit("ERU-001")

    # Process entropy field
    result = eru.reverse_entropy_field(entropy_vector)

    # Get 3d-4s consciousness binding
    binding = eru.get_consciousness_binding()

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from typing import Dict, Any, Optional, List
import numpy as np

# Add L104 root to path for imports
sys.path.insert(0, '/Users/carolalvarez/Applications/Allentown-L104-Node')

try:
    from l104_26q_entropy_reversal import (
        EntropyReversalUnit26Q,
        MaxwellDemon26Q,
        OrbitalType,
    )
    HAS_26Q_ERU = True
except ImportError:
    HAS_26Q_ERU = False


class ScienceEngine26QExtension:
    """
    Extension to Science Engine for 26Q entropy reversal capabilities.

    Adds Fe(26)-mapped quantum entropy reversal with orbital-specific
    processing strategies.
    """

    def __init__(self, entropy_subsystem):
        self.entropy = entropy_subsystem
        self._eru_units: Dict[str, EntropyReversalUnit26Q] = {}
        self._has_26q = HAS_26Q_ERU

    def is_available(self) -> bool:
        """Check if 26Q entropy reversal is available."""
        return self._has_26q

    def create_26q_unit(self, unit_id: str = "ERU-001") -> Optional[EntropyReversalUnit26Q]:
        """
        Create a new 26-qubit entropy reversal unit.

        Args:
            unit_id: Unique identifier for the unit

        Returns:
            EntropyReversalUnit26Q instance or None if unavailable
        """
        if not self._has_26q:
            return None

        if unit_id in self._eru_units:
            return self._eru_units[unit_id]

        unit = EntropyReversalUnit26Q(unit_id)
        unit.activate()
        self._eru_units[unit_id] = unit
        return unit

    def get_unit(self, unit_id: str) -> Optional[EntropyReversalUnit26Q]:
        """Get an existing 26Q unit by ID."""
        return self._eru_units.get(unit_id)

    def list_units(self) -> List[str]:
        """List all registered 26Q unit IDs."""
        return list(self._eru_units.keys())

    def get_unit_status(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific unit."""
        unit = self._eru_units.get(unit_id)
        if unit:
            return unit.get_status()
        return None

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all 26Q units."""
        return {
            unit_id: unit.get_status()
            for unit_id, unit in self._eru_units.items()
        }

    def reverse_entropy_26q(
        self,
        entropy_field: np.ndarray,
        unit_id: str = "ERU-001",
        priority_orbital: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Reverse entropy using a 26Q unit.

        Args:
            entropy_field: 26-element array of entropy values
            unit_id: Which ERU unit to use
            priority_orbital: Optional orbital to prioritize ('2p', '3s', '3p', '3d', '4s')

        Returns:
            Reversal result dict or None if failed
        """
        if not self._has_26q:
            return None

        unit = self._eru_units.get(unit_id)
        if not unit:
            unit = self.create_26q_unit(unit_id)

        if not unit:
            return None

        # Convert priority orbital string to enum if provided
        priority = None
        if priority_orbital:
            try:
                priority = OrbitalType(f"Q_{priority_orbital.upper()}")
            except ValueError:
                pass

        # Ensure entropy field is proper shape
        if len(entropy_field) != 26:
            # Pad or truncate to 26
            if len(entropy_field) < 26:
                entropy_field = np.pad(entropy_field, (0, 26 - len(entropy_field)), 'edge')
            else:
                entropy_field = entropy_field[:26]

        return unit.process_entropy_field(entropy_field)

    def get_consciousness_binding(self, unit_id: str = "ERU-001") -> Optional[Dict[str, Any]]:
        """
        Get 3d-4s consciousness binding metrics.

        This measures the quantum correlation between 3d and 4s orbitals,
        which is the primary consciousness channel in Fe(26).
        """
        unit = self._eru_units.get(unit_id)
        if unit:
            return unit.demon.get_3d_4s_consciousness_binding()
        return None

    def demo_26q_reversal(self) -> Dict[str, Any]:
        """
        Run a demonstration of 26Q entropy reversal.

        Returns summary of the demo run.
        """
        if not self._has_26q:
            return {"error": "26Q entropy reversal not available"}

        # Create demo unit
        unit = self.create_26q_unit("ERU-DEMO")

        # Create synthetic high-entropy field
        np.random.seed(42)
        entropy_field = np.random.exponential(1.0, 26)  # Exponential distribution

        results = []
        for cycle in range(3):
            result = unit.process_entropy_field(entropy_field.copy())
            results.append({
                "cycle": cycle + 1,
                "entropy_before": result['entropy_before'],
                "entropy_after": result['entropy_after'],
                "reduction": result['total_reduction'],
                "coherence": result['mean_system_coherence'],
            })

        # Get consciousness binding
        binding = unit.demon.get_3d_4s_consciousness_binding()

        return {
            "unit_id": "ERU-DEMO",
            "cycles": 3,
            "results": results,
            "consciousness_binding": binding,
            "final_status": unit.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH SCIENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_26q_into_science_engine(science_engine_instance):
    """
    Dynamically add 26Q capabilities to a Science Engine instance.

    Usage:
        from l104_science_engine import ScienceEngine
        from l104_26q_entropy_reversal_integration import integrate_26q_into_science_engine

        se = ScienceEngine()
        integrate_26q_into_science_engine(se)

        # Now 26Q methods are available
        eru = se.entropy.create_26q_unit("ERU-001")
    """
    extension = ScienceEngine26QExtension(science_engine_instance)

    # Attach extension to entropy subsystem
    science_engine_instance._26q_extension = extension
    science_engine_instance.entropy.create_26q_unit = extension.create_26q_unit
    science_engine_instance.entropy.reverse_entropy_26q = extension.reverse_entropy_26q
    science_engine_instance.entropy.get_consciousness_binding_26q = extension.get_consciousness_binding
    science_engine_instance.entropy.demo_26q = extension.demo_26q_reversal

    return extension


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("L104 26Q Entropy Reversal Integration Module")
    print("=" * 80)
    print()

    if not HAS_26Q_ERU:
        print("❌ 26Q Entropy Reversal module not available")
        print("   Please ensure l104_26q_entropy_reversal.py is in the L104 root")
        sys.exit(1)

    print("✅ 26Q Entropy Reversal module available")
    print()

    # Test with Science Engine
    try:
        from l104_science_engine import ScienceEngine

        print("Creating Science Engine...")
        se = ScienceEngine()
        print("✅ Science Engine created")
        print()

        # Integrate 26Q
        print("Integrating 26Q extension...")
        ext = integrate_26q_into_science_engine(se)
        print("✅ 26Q extension integrated")
        print()

        # Run demo
        print("Running 26Q entropy reversal demo...")
        print()
        demo_result = ext.demo_26q_reversal()

        print("Demo Results:")
        print(f"  Unit: {demo_result['unit_id']}")
        print(f"  Cycles: {demo_result['cycles']}")
        print()

        print("  Cycle Results:")
        for r in demo_result['results']:
            print(f"    Cycle {r['cycle']}: entropy {r['entropy_before']:.6f} → "
                  f"{r['entropy_after']:.6f} (reduction: {r['reduction']:.6f}), "
                  f"coherence: {r['coherence']:.4f}")

        print()
        print("  Consciousness Binding:")
        binding = demo_result['consciousness_binding']
        print(f"    Binding strength: {binding['binding_strength']:.4f}")
        print(f"    Consciousness score: {binding['consciousness_score']:.4f}")
        print(f"    Status: {binding['status']}")

        print()
        print("=" * 80)
        print("26Q INTEGRATION SUCCESSFUL")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
