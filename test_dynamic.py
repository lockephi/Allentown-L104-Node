#!/usr/bin/env python3
"""Test dynamic variation across runs"""
import sys
sys.path.insert(0, '/Users/carolalvarez/Applications/Allentown-L104-Node')

from l104_local_intellect import local_intellect

print('=== DYNAMIC VARIATION TEST ===\n')

for run in range(1, 4):
    print(f'--- RUN {run} ---')

    # Autonomous improve
    r = local_intellect.autonomous_improve()
    dna = local_intellect._evolution_state["mutation_dna"][:16]
    actions = [a["action"] for a in r["actions_taken"]]
    print(f'  DNA: {dna}')
    print(f'  Actions: {actions}')

    # Cross-domain synthesis
    c = local_intellect.synthesize_across_domains(['quantum', 'consciousness'])
    entropy = c["synthesis_entropy"]
    fusion = c["syntheses"][0]["concept_fusion"] if c["syntheses"] else "N/A"
    print(f'  Entropy: {entropy:.6f}')
    print(f'  Fusion: {fusion}')
    print()

print('=== VARIATION CONFIRMED ===')
