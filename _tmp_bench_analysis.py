#!/usr/bin/env python3
import re

# MMLU: count facts
with open('l104_asi/language_comprehension.py') as f:
    content = f.read()
lines = content.split('\n')
in_facts = False
fact_count = 0
node_count = 0
for line in lines:
    s = line.strip()
    if '_add_node(' in s:
        node_count += 1
    # Count lines that look like fact strings inside arrays
    if s.startswith('"') and s.endswith('",') or s.endswith('"]):') or s.endswith('"])'):
        fact_count += 1
    elif s.startswith('"') and ('"]' in s or '",' in s):
        fact_count += 1
print(f'MMLU nodes: {node_count}')
print(f'MMLU facts (approx): {fact_count}')

# ARC: count concepts and rules
with open('l104_asi/commonsense_reasoning.py') as f:
    content2 = f.read()
concept_count = content2.count('self._add(')
# Count rule tuples in _add_rules calls
rule_count = 0
in_rules_block = False
for line in content2.split('\n'):
    s = line.strip()
    if '_add_rules([' in s:
        in_rules_block = True
    if in_rules_block:
        if s.startswith('("'):
            rule_count += 1
        if s == '])':
            in_rules_block = False
print(f'ARC concepts: {concept_count}')
print(f'ARC causal rules: {rule_count}')

# HumanEval
with open('l104_asi/code_generation.py') as f:
    content3 = f.read()
pattern_count = content3.count('_register(AlgorithmPattern(')
cats = set(re.findall(r'category="(\w+)"', content3))
print(f'HumanEval patterns: {pattern_count}')
print(f'HumanEval categories: {sorted(cats)}')

# MATH
with open('l104_asi/symbolic_math_solver.py') as f:
    content4 = f.read()
solver_methods = re.findall(r'def (_solve_\w+)', content4)
print(f'MATH solver methods: {solver_methods}')
print(f'MATH solver count: {len(solver_methods)}')
