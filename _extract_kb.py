#!/usr/bin/env python3
"""Extract all _add_node data from language_comprehension.py into knowledge_data.py"""
import re

with open('l104_asi/language_comprehension.py', 'r') as f:
    content = f.read()

lines = content.split('\n')
nodes = []
i = 0
while i < len(lines):
    stripped = lines[i].strip()
    if stripped.startswith('self._add_node('):
        block = lines[i]
        paren_depth = lines[i].count('(') - lines[i].count(')')
        j = i + 1
        while paren_depth > 0 and j < len(lines):
            block += '\n' + lines[j]
            paren_depth += lines[j].count('(') - lines[j].count(')')
            j += 1
        nodes.append((i+1, block))
        i = j
    else:
        i += 1

print(f'Found {len(nodes)} _add_node calls')
print(f'First call at line {nodes[0][0]}')
print(f'Last call at line {nodes[-1][0]}')

# Now extract the cross-subject relations
rel_start = None
rel_end = None
for i, line in enumerate(lines):
    if 'relation_pairs' in line and '=' in line and '[' in line:
        rel_start = i
    if rel_start and line.strip() == ']':
        # Check if this closes relation_pairs
        # Count brackets from rel_start
        block = '\n'.join(lines[rel_start:i+1])
        if block.count('[') == block.count(']'):
            rel_end = i
            break

print(f'Relations block: lines {rel_start+1}-{rel_end+1}' if rel_start else 'Relations not found')

# Parse each node call to extract data
node_data = []
for line_no, block in nodes:
    # Clean up the block
    clean = block.strip()
    # Remove self._add_node( prefix and trailing )
    inner = clean[len('self._add_node('):]
    if inner.endswith(')'):
        inner = inner[:-1]
    node_data.append({'line': line_no, 'raw': block})

print(f'\nTotal nodes to extract: {len(node_data)}')
# Count total facts
total_facts = 0
for _, block in nodes:
    total_facts += block.count('"') // 2  # rough estimate
print(f'Estimated total fact strings: ~{total_facts}')
