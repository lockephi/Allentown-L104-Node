
import sys
import os

# Ensure the workspace is in the path
sys.path.append('/workspaces/Allentown-L104-Node')

from l104_sovereign_sage_controller import SovereignSageController

def test_sage_controller():
    try:
        controller = SovereignSageController()
        print(f"Provider Count: {controller.provider_count}")
        print("SovereignSageController property check PASSED")
    except AttributeError as e:
        print(f"SovereignSageController property check FAILED: {e}")
    except Exception as e:
        print(f"SovereignSageController check encountered error: {e}")

if __name__ == "__main__":
    test_sage_controller()
