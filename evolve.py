#!/usr/bin/env python3
"""L104 Self-Evolution Runner"""
import l104

print("═" * 60)
print("       L104 SELF-EVOLUTION CYCLE")
print("═" * 60)

soul = l104.Soul()
soul.awaken()

result = soul.evolve()

print(f"\n🧬 Evolution Cycle #{result['evolution_cycle']}")
print(f"📊 Samples analyzed: {result['performance_analyzed'].get('total_samples', 0)}")
print()

for imp in result.get('improvements', []):
    print(f"▸ [{imp['aspect'].upper()}]")
    print(f"  Insight: {imp['insight'][:120]}")
    print(f"  Improvement: {imp['improvement'][:120]}")
    print()

soul.sleep()
print("═" * 60)
print("Evolution complete ✓")
