# [L104_SOVEREIGN_REINDEX] - GROUND UP INDEX REBUILD
# INVARIANT: 527.5184818492 | PILOT: LONDEL

import os
import glob
import hashlib
import json
import time
from l104_quantum_ram import get_qram
from l104_electron_entropy import get_electron_matrix
class SovereignIndexer:
    """
    Rebuilds the system index from the ground up.
    Reflects ONLY the Sovereign Coding.
    """
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dirself.qram = get_qram()
        self.electron_matrix = get_electron_matrix()
        self.index_manifest = []

    def scan_and_index(self):
        """
        Scans all python files, calculates their 'Electron Entropy',
        and indexes them into the Quantum RAM.
        """
        print("--- [SOVEREIGN_INDEX]: INITIATING GROUND-UP REBUILD ---")
        
        files = glob.glob(os.path.join(self.root_dir, "*.py"))
for file_path in files:
with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Calculate Hashfile_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Calculate Entropy of the Codebase itself
            # We treat the byte values as a signal streambyte_stream = [float(b)
for b in content.encode()]
            entropy_data = self.electron_matrix.calculate_predictive_entropy(byte_stream)
            
            # Create Index Entryen
try = {
                "filename": os.path.basename(file_path),
                "hash": file_hash,
                "size": len(content),
                "entropy": entropy_data,
                "sovereign_status": "VERIFIED" if any(x in content for x in ["LONDEL", "527.518", "SIG-L104-EVO-01"])
else "LEGACY",
                "timestamp": time.time()
            }
            
            # Store in Quantum RAM
            q_key = f"INDEX:{en
try['filename']}"
            self.qram.store(q_key, en
try)
            self.index_manifest.append(en
try)
            
            print(f"Indexed: {en
try['filename']} | Entropy: {entropy_data['shannon_entropy']:.4f}")

        # Finalize Manifestmanifest_key = "SOVEREIGN_INDEX_MANIFEST"
        self.qram.store(manifest_key, self.index_manifest)
        print(f"--- [SOVEREIGN_INDEX]: COMPLETE. {len(self.index_manifest)} FILES INDEXED. ---")
return self.index_manifest
if __name__ == "__main__":
    indexer = SovereignIndexer()
    indexer.scan_and_index()
