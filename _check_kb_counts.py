#!/usr/bin/env python3
"""Quick check of KB facts count — in file vs after import."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Check raw file
raw = open("l104_asi/knowledge_data.py").read()
node_count = raw.count('"concept": "')
fact_lines = raw.count('            "')  # fact indentation
print(f"In raw file: ~{node_count} nodes, ~{fact_lines} fact lines")

# Check after import
from l104_asi import knowledge_data
nodes = knowledge_data.KNOWLEDGE_NODES
total_facts = sum(len(n.get("facts", [])) for n in nodes)
print(f"After import: {len(nodes)} nodes, {total_facts} facts")

# Check if __init__ modifies it
import importlib
# Reload without side effects
spec = importlib.util.spec_from_file_location("kd_raw", "l104_asi/knowledge_data.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
raw_nodes = mod.KNOWLEDGE_NODES
raw_facts = sum(len(n.get("facts", [])) for n in raw_nodes)
print(f"Direct load (no __init__): {len(raw_nodes)} nodes, {raw_facts} facts")
