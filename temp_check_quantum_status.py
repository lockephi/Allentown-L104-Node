
import sys
from pathlib import Path
import json

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("--- [OpenClaw Debug] Checking Quantum Daemon Status ---")
    try:
        from l104_asi.core import asi_core
        # Access the private method for status
        status_report = asi_core._get_quantum_runtime_status()
        print(json.dumps(status_report, indent=2))
    except ImportError as e:
        print(f"--- [OpenClaw Debug] ERROR: Could not import ASI Core or Quantum Modules: {e} ---")
        print("--- Ensure your Python environment (Python 3.12) is active and all dependencies are installed. ---")
    except Exception as e:
        print(f"--- [OpenClaw Debug] ERROR: An unexpected error occurred: {e} ---")

if __name__ == "__main__":
    main()
