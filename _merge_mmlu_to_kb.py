#!/usr/bin/env python3
"""
Merge MMLU-extracted facts into l104_asi/knowledge_data.py permanently.

Strategy:
1. Load existing KNOWLEDGE_NODES from knowledge_data.py
2. Load extracted_facts.json from fine_tune_exports/
3. Filter facts: remove very short (<20 chars), very long (>400 chars), and noisy text
4. For each subject, merge new unique facts into existing nodes or create new nodes
5. Write the updated knowledge_data.py

Also generates benchmark-aligned kernel training data.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Step 1: Load existing KB ──
from l104_asi import knowledge_data
existing_nodes = knowledge_data.KNOWLEDGE_NODES
existing_relations = knowledge_data.CROSS_SUBJECT_RELATIONS

print("=" * 70)
print("  MMLU KNOWLEDGE MERGE — Permanent Integration")
print("=" * 70)

print(f"\nExisting KB: {len(existing_nodes)} nodes")
existing_facts_total = sum(len(n.get("facts", [])) for n in existing_nodes)
print(f"Existing facts: {existing_facts_total}")

# ── Step 2: Load extracted facts ──
ext_path = PROJECT_ROOT / "fine_tune_exports" / "extracted_facts.json"
with open(ext_path) as f:
    extracted = json.load(f)

total_extracted = sum(len(v) for v in extracted.values())
print(f"\nExtracted facts: {total_extracted} across {len(extracted)} subjects")

# ── Step 3: Quality filter ──
# Noise patterns to reject
NOISE_PATTERNS = [
    r"which of (the|these) following",  # MCQ artifacts
    r"(^|\s)(A|B|C|D)\.\s",  # Choice markers
    r"according to (the passage|science daily|the article)",
    r"(last summer|one day|yesterday|this morning)\s+\w+\s+(and|went|felt)",  # narrative/story
    r"your (parents|friend|teacher)",  # personal narrative
    r"_+",  # fill in blank artifacts
]
NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

# Subject keyword validators — facts must contain at least one keyword to be "on-topic"
SUBJECT_VALIDATORS = {
    "abstract_algebra": ["group", "ring", "field", "homomorphism", "subgroup", "ideal", "abelian",
                          "cyclic", "isomorphism", "kernel", "quotient", "coset", "polynomial",
                          "algebra", "identity element", "inverse", "commutative", "associative",
                          "automorphism", "endomorphism", "module", "vector space", "lattice",
                          "galois", "permutation", "order of"],
    "college_physics": ["force", "energy", "momentum", "wave", "electric", "magnetic", "quantum",
                        "thermal", "newton", "velocity", "acceleration", "circuit", "voltage",
                        "current", "mass", "gravity", "photon", "frequency", "amplitude",
                        "electromagnetic", "capacitor", "resistor", "inductor", "entropy",
                        "temperature", "pressure", "optic", "refraction", "diffraction",
                        "interference", "kinetic", "potential energy", "angular", "torque"],
}


def is_quality_fact(fact: str, subject: str) -> bool:
    """Check if a fact passes quality filters."""
    # Length check
    if len(fact) < 20 or len(fact) > 400:
        return False

    # Noise pattern check
    for pat in NOISE_RE:
        if pat.search(fact):
            return False

    # If subject has a validator and fact has NO relevant keywords, reject
    if subject in SUBJECT_VALIDATORS:
        keywords = SUBJECT_VALIDATORS[subject]
        fl = fact.lower()
        if not any(kw in fl for kw in keywords):
            return False

    return True


# Category mapping (same as in mmlu_finetune_pipeline.py)
CATEGORY_MAP = {}
for cat, subjects in {
    "stem": ["abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
             "college_computer_science", "college_mathematics", "college_physics", "computer_security",
             "conceptual_physics", "electrical_engineering", "elementary_mathematics",
             "high_school_biology", "high_school_chemistry", "high_school_computer_science",
             "high_school_mathematics", "high_school_physics", "high_school_statistics",
             "machine_learning", "medical_genetics"],
    "humanities": ["formal_logic", "high_school_european_history", "high_school_us_history",
                   "high_school_world_history", "international_law", "jurisprudence",
                   "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
                   "prehistory", "professional_law", "world_religions"],
    "social_sciences": ["econometrics", "high_school_geography", "high_school_government_and_politics",
                        "high_school_macroeconomics", "high_school_microeconomics",
                        "high_school_psychology", "human_sexuality", "professional_psychology",
                        "public_relations", "security_studies", "sociology", "us_foreign_policy"],
    "other": ["business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
              "human_aging", "management", "marketing", "miscellaneous", "nutrition",
              "professional_accounting", "professional_medicine", "virology"],
}.items():
    for s in subjects:
        CATEGORY_MAP[s] = cat


# ── Step 4: Build index of existing nodes by subject ──
subject_to_nodes = defaultdict(list)
for i, node in enumerate(existing_nodes):
    subject_to_nodes[node["subject"]].append(i)

# Collect all existing facts as a set for deduplication
existing_facts_set = set()
for node in existing_nodes:
    for fact in node.get("facts", []):
        existing_facts_set.add(fact.strip().lower())

print(f"\nExisting unique facts fingerprints: {len(existing_facts_set)}")

# ── Step 5: Merge ──
merged_count = 0
new_nodes_count = 0
rejected_count = 0
skipped_dup = 0

for subject, facts in sorted(extracted.items()):
    # Filter facts
    clean_facts = []
    for fact in facts:
        fact = fact.strip()
        if not fact:
            continue
        if fact.strip().lower() in existing_facts_set:
            skipped_dup += 1
            continue
        if is_quality_fact(fact, subject):
            clean_facts.append(fact)
            existing_facts_set.add(fact.strip().lower())
        else:
            rejected_count += 1

    if not clean_facts:
        continue

    # Check if subject already has nodes
    if subject in subject_to_nodes:
        # Append to the first existing node for this subject
        idx = subject_to_nodes[subject][0]
        existing_nodes[idx]["facts"].extend(clean_facts)
        merged_count += len(clean_facts)
    else:
        # Create a new node
        category = CATEGORY_MAP.get(subject, "other")
        new_node = {
            "concept": f"mmlu_{subject}",
            "subject": subject,
            "category": category,
            "definition": f"Knowledge from MMLU training data for {subject.replace('_', ' ')}",
            "facts": clean_facts,
        }
        existing_nodes.append(new_node)
        subject_to_nodes[subject].append(len(existing_nodes) - 1)
        new_nodes_count += 1
        merged_count += len(clean_facts)

print(f"\n── Merge Results ──")
print(f"  Facts merged into existing nodes: {merged_count - sum(len(existing_nodes[i]['facts']) for i in range(len(existing_nodes)) if existing_nodes[i].get('concept','').startswith('mmlu_')) if new_nodes_count else merged_count}")
print(f"  New nodes created: {new_nodes_count}")
print(f"  Total facts added: {merged_count}")
print(f"  Duplicates skipped: {skipped_dup}")
print(f"  Quality-rejected: {rejected_count}")

# ── Step 6: Write updated knowledge_data.py ──
output_path = PROJECT_ROOT / "l104_asi" / "knowledge_data.py"
total_nodes = len(existing_nodes)
total_facts = sum(len(n.get("facts", [])) for n in existing_nodes)

lines = []
lines.append('"""')
lines.append('L104 ASI Knowledge Data — Structured knowledge base for language comprehension.')
lines.append('')
lines.append('This module contains the structured knowledge corpus used by the')
lines.append('LanguageComprehensionEngine for MMLU-grade question answering.')
lines.append('Separating data from algorithms enables clean code organization')
lines.append('and makes it easy to expand or update knowledge independently.')
lines.append('')
lines.append(f'Total nodes: {total_nodes}')
lines.append(f'Total facts: {total_facts}')
lines.append(f'Cross-subject relations: {len(existing_relations)}')
lines.append(f'MMLU-extracted facts merged: {merged_count}')
lines.append('"""')
lines.append('')
lines.append('from __future__ import annotations')
lines.append('from typing import Any, Dict, List, Tuple')
lines.append('')
lines.append('')
lines.append('# ═══════════════════════════════════════════════════════════════════════════════')
lines.append('#  KNOWLEDGE NODES — Structured as (concept, subject, category, definition, facts)')
lines.append('# ═══════════════════════════════════════════════════════════════════════════════')
lines.append('')
lines.append('KNOWLEDGE_NODES: List[Dict[str, Any]] = [')

# Group by category for organization
cat_order = ["stem", "humanities", "social_sciences", "other"]
nodes_by_cat = defaultdict(list)
for node in existing_nodes:
    nodes_by_cat[node.get("category", "other")].append(node)

for cat in cat_order:
    cat_nodes = nodes_by_cat.get(cat, [])
    if not cat_nodes:
        continue
    lines.append(f'    # ── {cat.upper()} ({len(cat_nodes)} nodes) ──')
    for node in cat_nodes:
        concept = node["concept"]
        subject = node["subject"]
        category = node["category"]
        definition = node.get("definition", "")
        facts = node.get("facts", [])

        # Escape strings for Python source
        def esc(s):
            return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', '')

        lines.append('    {')
        lines.append(f'        "concept": "{esc(concept)}",')
        lines.append(f'        "subject": "{esc(subject)}",')
        lines.append(f'        "category": "{esc(category)}",')
        lines.append(f'        "definition": "{esc(definition)}",')
        lines.append('        "facts": [')
        for fact in facts:
            lines.append(f'            "{esc(fact)}",')
        lines.append('        ],')
        lines.append('    },')

lines.append(']')
lines.append('')
lines.append('')
lines.append('# ═══════════════════════════════════════════════════════════════════════════════')
lines.append('#  CROSS-SUBJECT RELATIONS — (source_subject/concept, target_subject/concept)')
lines.append('# ═══════════════════════════════════════════════════════════════════════════════')
lines.append('')
lines.append('CROSS_SUBJECT_RELATIONS: List[Tuple[str, str]] = [')
for rel in existing_relations:
    lines.append(f'    ("{rel[0]}", "{rel[1]}"),')
lines.append(']')
lines.append('')

content = '\n'.join(lines)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n── Updated KB Written ──")
print(f"  File: {output_path}")
print(f"  Total nodes: {total_nodes}")
print(f"  Total facts: {total_facts}")
print(f"  File size: {len(content):,} bytes")
print(f"\n✅ MMLU facts permanently merged into knowledge_data.py")
