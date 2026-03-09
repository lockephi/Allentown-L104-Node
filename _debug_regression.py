#!/usr/bin/env python3
"""Debug: find ALL _SCIENCE_RULES entries that match regression questions."""
import re

filepath = 'l104_asi/commonsense_reasoning.py'
with open(filepath, 'r') as f:
    content = f.read()

q1 = 'a certain type of hybrid car utilizes a braking system in which energy is recovered and stored in batteries during braking this reclaims energy that was previously lost this is an example of which energy conversion'
q2 = 'a student heats the same amount of two different liquids over bunsen burners each liquid is at room temperature and reaches its boiling point liquid a reaches the boiling point first compared with liquid a liquid b will'

# Find the _SCIENCE_RULES block
start_idx = content.find('_SCIENCE_RULES')
if start_idx == -1:
    print("ERROR: Cannot find _SCIENCE_RULES")
    exit(1)

# Extract all raw string patterns from tuples
# Pattern: (r'q_pat', r'correct_pat' or None, r'wrong_pat' or None, boost, penalty)
import ast

# Find each tuple entry by looking for patterns like (r'...',
lines = content[start_idx:].split('\n')
# Collect tuples
tuples_raw = []
buf = ''
depth = 0
started = False
for line in lines:
    for ch in line:
        if ch == '(':
            depth += 1
            started = True
        if started:
            buf += ch
        if ch == ')' and started:
            depth -= 1
            if depth == 0:
                # Check if this looks like a rule tuple (has r' pattern)
                if "r'" in buf or 'r"' in buf:
                    tuples_raw.append(buf)
                buf = ''
                started = False
    if started:
        buf += '\n'
    # Stop after closing bracket of _SCIENCE_RULES
    if ']' in line and not started and len(tuples_raw) > 10:
        stripped = line.strip()
        if stripped == ']' or stripped.startswith(']'):
            break

print(f"Extracted {len(tuples_raw)} rule tuples")

# For each tuple, extract q_pat and test against both questions
q1_matches = []
q2_matches = []

for i, t in enumerate(tuples_raw):
    # Extract the first r'...' string (q_pat)
    m = re.search(r"r'((?:[^'\\]|\\.)*)'", t)
    if not m:
        m = re.search(r'r"((?:[^"\\]|\\.)*)"', t)
    if not m:
        continue
    q_pat = m.group(1)
    try:
        if re.search(q_pat, q1, re.IGNORECASE):
            q1_matches.append((i, q_pat, t))
        if re.search(q_pat, q2, re.IGNORECASE):
            q2_matches.append((i, q_pat, t))
    except re.error:
        pass

print(f"\n=== HYBRID CAR (idx 87) — {len(q1_matches)} matching rules ===")
for idx, qp, full in q1_matches:
    print(f"\nRule #{idx}: q_pat = {qp[:80]}")
    # Extract all r'...' strings
    all_pats = re.findall(r"r'((?:[^'\\]|\\.)*)'", full)
    if len(all_pats) >= 2:
        print(f"  correct_pat = {all_pats[1][:80]}")
    if len(all_pats) >= 3:
        print(f"  wrong_pat   = {all_pats[2][:80]}")
    # Extract boost/penalty
    nums = re.findall(r'(\d+\.?\d*)', full.split(all_pats[-1])[-1] if all_pats else full)
    if nums:
        print(f"  numbers: {nums}")
    # Test choices
    choices = {
        'A': 'kinetic energy being converted to potential energy',
        'B': 'chemical energy being converted to kinetic energy',
        'C': 'electrical energy being converted to light energy',
        'D': 'chemical energy being converted to electrical energy'
    }
    for label, text in choices.items():
        cp_hit = wp_hit = False
        if len(all_pats) >= 2 and all_pats[1] != 'None':
            try:
                cp_hit = bool(re.search(all_pats[1], text, re.IGNORECASE))
            except re.error:
                pass
        if len(all_pats) >= 3 and all_pats[2] != 'None':
            try:
                wp_hit = bool(re.search(all_pats[2], text, re.IGNORECASE))
            except re.error:
                pass
        if cp_hit or wp_hit:
            mark = "CORRECT" if label == 'A' else "WRONG"
            action = "BOOST" if cp_hit else "PENALIZE"
            print(f"  >> {label} ({mark}): {action} — {text[:50]}")

print(f"\n=== HEATING LIQUIDS (idx 94) — {len(q2_matches)} matching rules ===")
for idx, qp, full in q2_matches:
    print(f"\nRule #{idx}: q_pat = {qp[:80]}")
    all_pats = re.findall(r"r'((?:[^'\\]|\\.)*)'", full)
    if len(all_pats) >= 2:
        print(f"  correct_pat = {all_pats[1][:80]}")
    if len(all_pats) >= 3:
        print(f"  wrong_pat   = {all_pats[2][:80]}")
    nums = re.findall(r'(\d+\.?\d*)', full.split(all_pats[-1])[-1] if all_pats else full)
    if nums:
        print(f"  numbers: {nums}")
    choices = {
        'A': 'evaporate sooner',
        'B': 'take longer to increase in temperature',
        'C': 'remain at room temperature for a longer period of time',
        'D': 'need a higher temperature to reach its boiling point'
    }
    for label, text in choices.items():
        cp_hit = wp_hit = False
        if len(all_pats) >= 2 and all_pats[1] != 'None':
            try:
                cp_hit = bool(re.search(all_pats[1], text, re.IGNORECASE))
            except re.error:
                pass
        if len(all_pats) >= 3 and all_pats[2] != 'None':
            try:
                wp_hit = bool(re.search(all_pats[2], text, re.IGNORECASE))
            except re.error:
                pass
        if cp_hit or wp_hit:
            mark = "CORRECT" if label == 'B' else "WRONG"
            action = "BOOST" if cp_hit else "PENALIZE"
            print(f"  >> {label} ({mark}): {action} — {text[:50]}")
