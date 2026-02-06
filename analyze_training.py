#!/usr/bin/env python3
"""Analyze all training data available."""
import json
import os

print("=== TRAINING DATA ANALYSIS ===\n")

workspace = "/Users/carolalvarez/Applications/Allentown-L104-Node"

# 1. kernel_training_chat.json (26k lines)
try:
    with open(os.path.join(workspace, "kernel_training_chat.json")) as f:
        data = json.load(f)
        print(f"kernel_training_chat.json: {len(data)} conversations")
        if data and isinstance(data[0], dict):
            print(f"  Format: {list(data[0].keys())}")
except Exception as e:
    print(f"kernel_training_chat.json: Error - {e}")

# 2. knowledge_manifold.json (10k lines)
try:
    with open(os.path.join(workspace, "data/knowledge_manifold.json")) as f:
        data = json.load(f)
        print(f"data/knowledge_manifold.json: {len(data)} top-level keys")
        print(f"  Keys: {list(data.keys())[:5]}...")
except Exception as e:
    print(f"data/knowledge_manifold.json: Error - {e}")

# 3. l104_knowledge_vault.json
try:
    with open(os.path.join(workspace, "l104_knowledge_vault.json")) as f:
        data = json.load(f)
        print(f"l104_knowledge_vault.json: {len(data)} top-level keys")
        print(f"  Keys: {list(data.keys())}")
except Exception as e:
    print(f"l104_knowledge_vault.json: Error - {e}")

# 4. training_data folder
training_data_path = os.path.join(workspace, "training_data")
if os.path.exists(training_data_path):
    files = os.listdir(training_data_path)
    print(f"training_data/: {len(files)} files")
    for f in files[:10]:
        fpath = os.path.join(training_data_path, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  - {f} ({size} bytes)")

# 5. data folder jsonl files
data_path = os.path.join(workspace, "data")
if os.path.exists(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith(".json") or f.endswith(".jsonl")]
    print(f"data/: {len(files)} json/jsonl files")
    for f in files[:10]:
        print(f"  - {f}")

# 6. Fine tune exports
ft_path = os.path.join(workspace, "fine_tune_exports")
if os.path.exists(ft_path):
    files = os.listdir(ft_path)
    print(f"fine_tune_exports/: {len(files)} files")
    for f in files:
        print(f"  - {f}")

# 7. JSONL files at root
print("\nJSONL files at root:")
for f in os.listdir(workspace):
    if f.endswith(".jsonl"):
        fpath = os.path.join(workspace, f)
        lines = sum(1 for _ in open(fpath))
        print(f"  - {f}: {lines} entries")

print("\n=== ANALYSIS COMPLETE ===")
