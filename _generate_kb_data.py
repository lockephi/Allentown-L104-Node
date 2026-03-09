#!/usr/bin/env python3
"""Generate l104_asi/knowledge_data.py from language_comprehension.py _add_node calls.

This script parses the source code to extract all hardcoded knowledge into
a clean data module that can be loaded by the algorithmic pipeline.
"""
import re
import ast
import json
import textwrap

with open('l104_asi/language_comprehension.py', 'r') as f:
    source = f.read()

lines = source.split('\n')

# Step 1: Extract each _add_node call block
call_blocks = []
i = 0
while i < len(lines):
    stripped = lines[i].strip()
    if stripped.startswith('self._add_node('):
        block_lines = [lines[i]]
        depth = lines[i].count('(') - lines[i].count(')')
        j = i + 1
        while depth > 0 and j < len(lines):
            block_lines.append(lines[j])
            depth += lines[j].count('(') - lines[j].count(')')
            j += 1
        call_blocks.append('\n'.join(block_lines))
        i = j
    else:
        i += 1

print(f"Extracted {len(call_blocks)} _add_node call blocks")

# Step 2: Parse each call to extract concept, subject, category, definition, facts
node_entries = []
for block in call_blocks:
    # Normalize: make it a valid expression
    expr = block.strip()
    # Remove self._add_node( prefix
    expr = expr[len('self._add_node('):]
    if expr.endswith(')'):
        expr = expr[:-1]

    # Try to evaluate as a Python expression (tuple of args)
    # We need to handle named args like facts=[...], relations={...}
    # Strategy: wrap in a function call and use ast
    try:
        # Build a fake function call we can parse
        fake_call = f"_f({expr})"
        tree = ast.parse(fake_call, mode='eval')
        call_node = tree.body

        args = call_node.args
        kwargs = {kw.arg: kw.value for kw in call_node.keywords}

        concept = ast.literal_eval(args[0]) if len(args) > 0 else ""
        subject = ast.literal_eval(args[1]) if len(args) > 1 else ""
        category = ast.literal_eval(args[2]) if len(args) > 2 else ""
        definition = ast.literal_eval(args[3]) if len(args) > 3 else ""

        # Facts: either 5th positional arg or keyword
        facts = []
        if len(args) > 4:
            facts = ast.literal_eval(args[4])
        elif 'facts' in kwargs:
            facts = ast.literal_eval(kwargs['facts'])

        # Relations
        relations = {}
        if 'relations' in kwargs:
            try:
                relations = ast.literal_eval(kwargs['relations'])
            except:
                pass

        node_entries.append({
            'concept': concept,
            'subject': subject,
            'category': category,
            'definition': definition,
            'facts': facts,
        })
    except Exception as e:
        print(f"  WARN: Failed to parse block: {str(e)[:80]}")
        print(f"    Block starts: {block[:100]}...")

# Step 3: Extract cross-subject relation pairs
relation_pairs = []
in_relations = False
rel_block = ""
for line in lines:
    stripped = line.strip()
    if 'relation_pairs' in line and '=' in line and '[' in line:
        in_relations = True
        rel_block = line[line.index('['):]
        continue
    if in_relations:
        rel_block += '\n' + line
        if rel_block.count('[') <= rel_block.count(']'):
            # Try to parse
            try:
                relation_pairs = ast.literal_eval(rel_block.strip())
            except:
                print(f"  WARN: Could not parse relation_pairs block")
            in_relations = False

print(f"Extracted {len(node_entries)} nodes, {len(relation_pairs)} relation pairs")
total_facts = sum(len(n['facts']) for n in node_entries)
print(f"Total facts: {total_facts}")

# Step 4: Generate knowledge_data.py
output_lines = []
output_lines.append('"""')
output_lines.append('L104 ASI Knowledge Data — Structured knowledge base for language comprehension.')
output_lines.append('')
output_lines.append('This module contains the structured knowledge corpus used by the')
output_lines.append('LanguageComprehensionEngine for MMLU-grade question answering.')
output_lines.append('Separating data from algorithms enables clean code organization')
output_lines.append('and makes it easy to expand or update knowledge independently.')
output_lines.append('')
output_lines.append(f'Total nodes: {len(node_entries)}')
output_lines.append(f'Total facts: {total_facts}')
output_lines.append(f'Cross-subject relations: {len(relation_pairs)}')
output_lines.append('"""')
output_lines.append('')
output_lines.append('from __future__ import annotations')
output_lines.append('from typing import Any, Dict, List, Tuple')
output_lines.append('')
output_lines.append('')

# Group nodes by category for readability
categories = {}
for entry in node_entries:
    cat = entry['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(entry)

output_lines.append('# ═══════════════════════════════════════════════════════════════════════════════')
output_lines.append('#  KNOWLEDGE NODES — Structured as (concept, subject, category, definition, facts)')
output_lines.append('# ═══════════════════════════════════════════════════════════════════════════════')
output_lines.append('')
output_lines.append('KNOWLEDGE_NODES: List[Dict[str, Any]] = [')

for cat_name in ['stem', 'humanities', 'social_sciences', 'other']:
    entries = categories.get(cat_name, [])
    if not entries:
        continue
    output_lines.append(f'    # ── {cat_name.upper()} ({len(entries)} nodes) ──')
    for entry in entries:
        # Format each node as a dict entry
        facts_str = json.dumps(entry['facts'], ensure_ascii=False)
        output_lines.append('    {')
        output_lines.append(f'        "concept": {json.dumps(entry["concept"])},')
        output_lines.append(f'        "subject": {json.dumps(entry["subject"])},')
        output_lines.append(f'        "category": {json.dumps(entry["category"])},')
        output_lines.append(f'        "definition": {json.dumps(entry["definition"], ensure_ascii=False)},')
        # Format facts nicely — one per line if >2
        if len(entry['facts']) <= 2:
            output_lines.append(f'        "facts": {facts_str},')
        else:
            output_lines.append('        "facts": [')
            for fact in entry['facts']:
                output_lines.append(f'            {json.dumps(fact, ensure_ascii=False)},')
            output_lines.append('        ],')
        output_lines.append('    },')

output_lines.append(']')
output_lines.append('')
output_lines.append('')

# Cross-subject relations
output_lines.append('# ═══════════════════════════════════════════════════════════════════════════════')
output_lines.append('#  CROSS-SUBJECT RELATIONS — Bidirectional edges in the knowledge graph')
output_lines.append('# ═══════════════════════════════════════════════════════════════════════════════')
output_lines.append('')
output_lines.append('CROSS_SUBJECT_RELATIONS: List[Tuple[str, str]] = [')
for pair in relation_pairs:
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        output_lines.append(f'    ({json.dumps(pair[0])}, {json.dumps(pair[1])}),')
output_lines.append(']')
output_lines.append('')

# Write the file
output_content = '\n'.join(output_lines)
with open('l104_asi/knowledge_data.py', 'w') as f:
    f.write(output_content)

print(f"\nGenerated l104_asi/knowledge_data.py ({len(output_content)} bytes)")
print(f"  {len(node_entries)} nodes across {len(categories)} categories")
print(f"  {total_facts} total facts")
print(f"  {len(relation_pairs)} cross-subject relations")
