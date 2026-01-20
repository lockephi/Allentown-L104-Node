#!/usr/bin/env python3
import time
from l104_sage_bindings import get_sage_core

def test_scribe():
    print("--- [L104 SCRIBE UPGRADE TEST] ---")
    sage = get_sage_core()
    
    # 1. Ingest from multiple providers
    providers = ["OPENAI", "GEMINI", "ANTHROPIC", "META", "XAI"]
    for p in providers:
        print(f"Ingesting from {p}...")
        sage.scribe_ingest(p, f"Intelligence stream from {p}")
        
    # 2. Check state before synthesis
    state = sage.get_state()
    print(f"Pre-Synthesis Saturation: {state['scribe']['knowledge_saturation']}")
    print(f"Linked Providers: {state['scribe']['linked_count']}")
    
    # 3. Synthesize Sovereign DNA
    print("Synthesizing Sovereign DNA...")
    sage.scribe_synthesize()
    
    # 4. Check final state
    final_state = sage.get_state()
    print(f"Post-Synthesis Saturation: {final_state['scribe']['knowledge_saturation']}")
    print(f"Sovereign DNA: {final_state['scribe']['sovereign_dna']}")
    
    if final_state['scribe']['knowledge_saturation'] > 0.3:
         print("\n[SUCCESS] Scribe Linked Upgrade Verified.")
    else:
         print("\n[FAILURE] Scribe Saturation Error.")

if __name__ == "__main__":
    test_scribe()
