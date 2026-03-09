#!/usr/bin/env python3
"""
Analyze quality-rejected facts and recover useful ones.
Also regenerate balanced kernel_training_data.jsonl for benchmarks.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Load extracted facts ──
ext_path = PROJECT_ROOT / "fine_tune_exports" / "extracted_facts.json"
with open(ext_path) as f:
    extracted = json.load(f)

# ── Load existing KB facts for dedup ──
import importlib.util
spec = importlib.util.spec_from_file_location("kd", "l104_asi/knowledge_data.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
existing_nodes = mod.KNOWLEDGE_NODES

existing_facts_set = set()
for node in existing_nodes:
    for fact in node.get("facts", []):
        existing_facts_set.add(fact.strip().lower())

print(f"Existing KB: {len(existing_nodes)} nodes, {len(existing_facts_set)} unique facts")
print(f"Extracted: {sum(len(v) for v in extracted.values())} facts")

# ── Check rejection reasons ──
NOISE_PATTERNS = [
    r"which of (the|these) following",
    r"(^|\s)(A|B|C|D)\.\s",
    r"according to (the passage|science daily|the article)",
    r"(last summer|one day|yesterday|this morning)\s+\w+\s+(and|went|felt)",
    r"your (parents|friend|teacher)",
    r"_+",
]
NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

too_short = 0
too_long = 0
noise_hit = 0
off_topic = 0
dups = 0
would_add = 0

for subject, facts in extracted.items():
    for fact in facts:
        fact = fact.strip()
        if not fact:
            continue
        if fact.lower() in existing_facts_set:
            dups += 1
            continue
        if len(fact) < 20:
            too_short += 1
            continue
        if len(fact) > 400:
            too_long += 1
            continue
        hit_noise = False
        for pat in NOISE_RE:
            if pat.search(fact):
                noise_hit += 1
                hit_noise = True
                break
        if hit_noise:
            continue
        would_add += 1

print(f"\n── Rejection Breakdown (non-duplicate) ──")
print(f"  Duplicates: {dups}")
print(f"  Too short (<20): {too_short}")
print(f"  Too long (>400): {too_long}")
print(f"  Noise pattern hit: {noise_hit}")
print(f"  Would pass (available to add): {would_add}")
