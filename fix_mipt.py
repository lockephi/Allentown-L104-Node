#!/usr/bin/env python3
"""Fix the mipt_engine.py file."""

with open('/Users/carolalvarez/Applications/Allentown-L104-Node/l104_quantum_gate_engine/mipt_engine.py', 'r') as f:
    content = f.read()

# Remove the misplaced prints
lines = content.split('\n')

# Find the function return and remove orphaned prints after it
new_lines = []
found_return = False
skip_until_main = False

for i, line in enumerate(lines):
    if 'return {' in line and '"thought_coherence"' in lines[i+3] if i+3 < len(lines) else False:
        found_return = True

    if found_return and line.strip().startswith('print(') and '"Criticality Score"' in line:
        # Found the misplaced prints, skip until we see if __name__ or end
        skip_until_main = True
        continue

    if skip_until_main:
        if 'print("=" * 70)' in line and i > len(lines) - 10:
            # This is part of main, keep it
            skip_until_main = False
            # Add the if __name__ check
            new_lines.append('')
            new_lines.append('')
            new_lines.append('if __name__ == "__main__":')
            new_lines.append('    # Demo: Run MIPT on 26-qubit system')
            new_lines.append('    print("=" * 70)')
            continue
        elif line.strip() and not line.strip().startswith('print'):
            skip_until_main = False
        else:
            continue

    new_lines.append(line)

# Write back
with open('/Users/carolalvarez/Applications/Allentown-L104-Node/l104_quantum_gate_engine/mipt_engine.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("File fixed!")
