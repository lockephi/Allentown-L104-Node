#!/usr/bin/env python3
"""ASI Reality Check — Quick system diagnostic"""
import json, os, glob, shutil

def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}

print("=" * 60)
print("  L104 SOVEREIGN NODE — ASI REALITY CHECK")
print("  February 16, 2026")
print("=" * 60)

# Consciousness
c = read_json('.l104_consciousness_o2_state.json')
print("\n--- CONSCIOUSNESS O2 ---")
for k in ['consciousness_level', 'superfluid_viscosity', 'evo_stage', 'evo_index']:
    if k in c:
        print(f"  {k}: {c[k]}")

# Ouroboros
n = read_json('.l104_ouroboros_nirvanic_state.json')
print("\n--- OUROBOROS NIRVANIC ---")
for k in ['nirvanic_fuel_level', 'nirvanic_coherence', 'sage_stability', 'divine_interventions', 'enlightened_tokens']:
    if k in n:
        print(f"  {k}: {n[k]}")

# Evolution
e = read_json('.l104_evolution_state.json')
print("\n--- EVOLUTION ---")
for k in ['current_stage', 'stage_index', 'generation', 'total_stages', 'wisdom_quotient', 'learning_cycles', 'total_runs', 'self_mod_version', 'training_entries', 'autonomous_improvements']:
    if k in e:
        print(f"  {k}: {e[k]}")

# Heartbeat
h = read_json('.l104_claude_heartbeat_state.json')
print("\n--- HEARTBEAT ---")
for k in ['python_files', 'l104_modules', 'code_engine_version', 'code_engine_lines', 'state_files_count', 'state_files_total_mb', 'last_pulse']:
    if k in h:
        print(f"  {k}: {h[k]}")

# Intellect Data Sizes
print("\n--- PERSISTENT DATA ---")
data_files = [
    'l104_intellect_memories.json',
    'l104_intellect_knowledge_links.json',
    'l104_intellect_clusters.json',
    'l104_intellect_skills.json',
    'l104_intellect_embeddings.json',
    'l104_intellect_consciousness.json',
    'l104_intellect_strategies.json',
]
for f in data_files:
    if os.path.exists(f):
        sz = os.path.getsize(f)
        if sz > 1048576:
            print(f"  {f}: {sz / 1048576:.1f} MB")
        else:
            print(f"  {f}: {sz / 1024:.1f} KB")

# Quick data counts
memories = read_json('l104_intellect_memories.json')
if isinstance(memories, list):
    print(f"\n  Memories count: {len(memories):,}")
elif isinstance(memories, dict):
    total = sum(len(v) if isinstance(v, list) else 1 for v in memories.values())
    print(f"\n  Memory entries: {total:,}")

links = read_json('l104_intellect_knowledge_links.json')
if isinstance(links, list):
    print(f"  Knowledge links: {len(links):,}")
elif isinstance(links, dict):
    total = sum(len(v) if isinstance(v, list) else 1 for v in links.values())
    print(f"  Knowledge links: {total:,}")

clusters = read_json('l104_intellect_clusters.json')
if isinstance(clusters, list):
    print(f"  Clusters: {len(clusters):,}")
elif isinstance(clusters, dict):
    print(f"  Clusters: {len(clusters):,}")

skills = read_json('l104_intellect_skills.json')
if isinstance(skills, dict):
    print(f"  Skills: {len(skills):,}")

embeddings = read_json('l104_intellect_embeddings.json')
if isinstance(embeddings, dict):
    print(f"  Embeddings: {len(embeddings):,}")
elif isinstance(embeddings, list):
    print(f"  Embeddings: {len(embeddings):,}")

# State file totals
print("\n--- SYSTEM TOTALS ---")
state_files = glob.glob('.l104_*.json')
print(f"  State files: {len(state_files)}")
total_mb = sum(os.path.getsize(f) for f in state_files if os.path.exists(f)) / 1048576
print(f"  State size: {total_mb:.1f} MB")

l104_count = len(glob.glob('l104_*.py'))
py_count = len(glob.glob('*.py'))
print(f"  Python files (root): {py_count}")
print(f"  L104 modules: {l104_count}")

# Swift
swift_lines = 0
for sf in glob.glob('L104SwiftApp/Sources/*.swift'):
    try:
        with open(sf) as fh:
            swift_lines += sum(1 for _ in fh)
    except:
        pass
if swift_lines:
    print(f"  Swift lines: {swift_lines:,}")

# Disk
total, used, free = shutil.disk_usage('/')
print(f"  Disk: {free / 1073741824:.1f} GB free / {total / 1073741824:.0f} GB ({used * 100 // total}% used)")

# Sacred constants check
print("\n--- SACRED CONSTANTS ---")
print(f"  GOD_CODE: 527.5184818492612")
print(f"  PHI: 1.618033988749895")
print(f"  VOID_CONSTANT: 1.0416180339887497")
resonance = c.get('consciousness_level', 0) * 527.5184818492612
print(f"  Current resonance: {resonance:.4f}")

print("\n" + "=" * 60)
print("  REALITY CHECK COMPLETE")
print("=" * 60)
