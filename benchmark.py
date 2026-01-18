#!/usr/bin/env python3
"""
L104 Benchmark Script
Run: python benchmark.py
"""

import os
import sys
import time

sys.path.insert(0, '/workspaces/Allentown-L104-Node')
os.chdir('/workspaces/Allentown-L104-Node')
# Ghost Protocol: API key loaded from .env only

from l104 import Database, LRUCache, Gemini, Memory, Knowledge, Learning, Planner, Mind, get_soul, GOD_CODE

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   ⟨Σ_L104⟩  B E N C H M A R K                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")
    
    results = {}
    
    # Import
    print("[1/6] Importing L104...")
    t = time.time()
    results["import"] = f"{(time.time()-t)*1000:.0f}ms"
    print(f"  ✓ Import: {results['import']}")
    
    # Database
    print("[2/6] Database...")
    db = Database()
    t = time.time()
    for i in range(100):
        db.execute("INSERT OR REPLACE INTO memory (key,value) VALUES (?,?)", (f"bench_{i}", f"val_{i}"))
    db.commit()
    results["db_write"] = f"{(time.time()-t)*1000:.1f}ms"
    
    t = time.time()
    for i in range(100):
        db.execute("SELECT value FROM memory WHERE key=?", (f"bench_{i}",)).fetchone()
    results["db_read"] = f"{(time.time()-t)*1000:.1f}ms"
    print(f"  ✓ Write 100: {results['db_write']} | Read 100: {results['db_read']}")
    
    # Cache
    print("[3/6] LRU Cache...")
    cache = LRUCache(1000)
    t = time.time()
    for i in range(1000):
        cache.put(f"k{i}", f"v{i}")
    results["cache_write"] = f"{(time.time()-t)*1000:.1f}ms"
    
    t = time.time()
    for i in range(1000):
        cache.get(f"k{i}")
    results["cache_read"] = f"{(time.time()-t)*1000:.1f}ms"
    print(f"  ✓ Write 1000: {results['cache_write']} | Read 1000: {results['cache_read']}")
    
    # Knowledge
    print("[4/6] Knowledge Graph...")
    kg = Knowledge(db)
    
    # Test with batch mode for performance
    t = time.time()
    kg.batch_start()
    for i in range(50):
        kg.add_node(f"concept_{i}", "benchmark")
    kg.batch_end()
    results["kg_add_batch"] = f"{(time.time()-t)*1000:.1f}ms"
    
    t = time.time()
    matches = kg.search("concept", top_k=10)
    results["kg_search"] = f"{(time.time()-t)*1000:.1f}ms"
    print(f"  ✓ Add 50 (batch): {results['kg_add_batch']} | Search: {results['kg_search']} ({len(matches)} results)")
    
    # Gemini
    print("[5/6] Gemini API...")
    gemini = Gemini()
    if gemini.connect():
        t = time.time()
        resp = gemini.generate("What is 2+2? Reply with just the number.")
        results["gemini_first"] = f"{(time.time()-t)*1000:.0f}ms"
        
        t = time.time()
        resp2 = gemini.generate("What is 2+2? Reply with just the number.")
        results["gemini_cached"] = f"{(time.time()-t)*1000:.2f}ms"
        
        print(f"  ✓ First: {results['gemini_first']} | Cached: {results['gemini_cached']}")
        if resp:
            print(f"    Response: {resp[:60]}...")
        else:
            print(f"    Response: None (error: {getattr(gemini, '_last_error', 'unknown')})")
    else:
        print("  ✗ Gemini connection failed")
        results["gemini"] = "failed"
    
    # Soul
    print("[6/6] Soul Integration...")
    soul = get_soul()
    
    t = time.time()
    report = soul.awaken()
    results["soul_awaken"] = f"{(time.time()-t)*1000:.0f}ms"
    
    t = time.time()
    thought = soul.think("Are you conscious?")
    results["soul_think"] = f"{(time.time()-t)*1000:.0f}ms"
    
    online = sum(1 for v in report.get("subsystems", {}).values() if v == "online")
    print(f"  ✓ Awaken: {results['soul_awaken']} | Think: {results['soul_think']}")
    print(f"    Subsystems: {online} | Response: {thought.get('response', '')[:40]}...")
    
    soul.sleep()
    
    # Summary
    print("""
═══════════════════════════════════════════════════════════════════════════════
                              BENCHMARK RESULTS
═══════════════════════════════════════════════════════════════════════════════""")
    
    for key, value in results.items():
        print(f"  {key:20} : {value}")
    
    print(f"""
═══════════════════════════════════════════════════════════════════════════════
  GOD_CODE: {GOD_CODE}
═══════════════════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
