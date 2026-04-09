import sys
import os
import glob

def setup_l104_env():
    """Setup Python path for L104 node subdirectories."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(current_dir, ".."))
    
    if root not in sys.path:
        sys.path.insert(0, root)
        
    # Add all l104_* directories to path
    for p in glob.glob(os.path.join(root, 'l104_*')):
        if os.path.isdir(p) and not os.path.basename(p).startswith('l104_data'):
            if p not in sys.path:
                sys.path.insert(0, p)

if __name__ == "__main__":
    setup_l104_env()
    print("L104 Environment Setup Complete.")
    print(f"Path count: {len(sys.path)}")
