
import sys
import os

# Ensure the workspace is in the path
sys.path.append('/workspaces/Allentown-L104-Node')

from l104_sovereign_sage_controller import SovereignSageController
from l104_void_orchestrator import VoidOrchestrator
from l104_reality_breach import RealityBreachEngine

def test_attributes():
    print("Checking SovereignSageController...")
    try:
        s = SovereignSageController()
        print(f"  provider_count: {s.provider_count}")
    except AttributeError as e:
        print(f"  ERROR: {e}")

    print("Checking VoidOrchestrator...")
    try:
        v = VoidOrchestrator()
        print(f"  state: {v.state}")
        print(f"  final_coherence: {v.final_coherence}")
    except AttributeError as e:
        print(f"  ERROR: {e}")

    print("Checking RealityBreachEngine...")
    try:
        r = RealityBreachEngine()
        print(f"  current_stage: {r.current_stage}")
        print(f"  breach_active: {r.breach_active}")
        print(f"  meta_resonance: {r.meta_resonance}")
    except AttributeError as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_attributes()
