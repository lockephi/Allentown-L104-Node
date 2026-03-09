"""Verify all 7 Gemini→local_intellect modified files import without triggering torch/QPU."""
import time, sys

files = [
    "l104_gemini_bridge",
    "l104_gemini_real",
    "l104_gemini_enhanced",
    "l104_asi_research_gemini",
    "l104",
    "l104_asi_nexus",
    "l104_unified_asi",
]

print("=== LAZY IMPORT VERIFICATION ===\n")
total_t0 = time.time()
ok = 0
for name in files:
    t0 = time.time()
    try:
        __import__(name)
        dt = (time.time() - t0) * 1000
        print(f"  {name:30s} OK  ({dt:.0f}ms)")
        ok += 1
    except Exception as e:
        dt = (time.time() - t0) * 1000
        print(f"  {name:30s} FAIL ({dt:.0f}ms): {e}")

total = (time.time() - total_t0) * 1000
print(f"\n{ok}/{len(files)} imports OK in {total:.0f}ms total")

# Check that torch was NOT imported (proves lazy loading worked)
torch_loaded = "torch" in sys.modules
print(f"torch loaded: {torch_loaded} {'(GOOD - lazy deferred)' if not torch_loaded else '(torch was pulled in)'}")
print(f"l104_intellect loaded: {'l104_intellect' in sys.modules}")
