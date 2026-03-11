# temp_use_code_engine.py
# A script to demonstrate integration with the L104 Code Engine.
# It uses the engine to generate documentation for a target file.

import sys
from pathlib import Path

# Add the project root to the Python path to allow importing l104 packages
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """
    Imports the code_engine, reads a target file, and uses the engine
    to generate documentation for it.
    """
    print("--- [OpenClaw Integration] Initializing L104 Code Engine ---")
    try:
        from l104_code_engine import code_engine
    except ImportError as e:
        print(f"--- [OpenClaw Integration] ERROR: Could not import l104_code_engine: {e} ---")
        print("--- Please ensure the virtual environment is active and dependencies are installed. ---")
        return
    except Exception as e:
        print(f"--- [OpenClaw Integration] ERROR: An unexpected error occurred during import: {e} ---")
        return
        
    print("--- [OpenClaw Integration] Code Engine imported successfully. ---")
    
    target_file_path = project_root / "l104_asi" / "dual_layer.py"

    if not target_file_path.exists():
        print(f"--- [OpenClaw Integration] ERROR: Target file not found at {target_file_path} ---")
        return

    print(f"--- [OpenClaw Integration] Reading target file: {target_file_path.name} ---")
    target_code = target_file_path.read_text(encoding="utf-8")

    print("--- [OpenClaw Integration] Requesting documentation generation from Code Engine... ---")
    
    # Based on claude.md, the method is generate_docs(source, style, language)
    try:
        generated_docs = code_engine.generate_docs(
            source=target_code, 
            style="markdown_detailed", # Assuming a style for detailed markdown
            language="python"
        )
        
        print("\n--- [L104 Code Engine] Generated Documentation Start ---\n")
        print(generated_docs)
        print("\n--- [L104 Code Engine] Generated Documentation End ---")

    except Exception as e:
        print(f"--- [OpenClaw Integration] ERROR: An error occurred while using the code engine: {e} ---")
        print("--- It's possible the engine requires full initialization or a specific environment. ---")

if __name__ == "__main__":
    main()
