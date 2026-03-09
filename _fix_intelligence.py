"""Fix link.get() calls in intelligence.py to use _lget() for dict/dataclass compatibility."""
import re

filepath = "l104_quantum_engine/intelligence.py"

with open(filepath, 'r') as f:
    content = f.read()

# Link-specific fields that are accessed with .get()
link_fields = (
    'fidelity|strength|source|target|source_file|target_file|'
    'source_symbol|target_symbol|link_type|entanglement_strength|'
    'sacred_alignment|noise_resilience|coherence_time|entanglement_entropy|'
    'bell_violation'
)

# Replace: link.get("field", default) -> _lget(link, "field", default)
# and:     l.get("field", default) -> _lget(l, "field", default)
pattern = r'\b(link|l)\b\.get\(("(?:' + link_fields + r')"(?:,\s*[^)]+)?)\)'
replacement = r'_lget(\1, \2)'

new_content = re.sub(pattern, replacement, content)

# Also need to handle link["field"] = ... assignments for heal()
# Add _lset helper after _lget in the file
lset_helper = '''

def _lset(link, key, value):
    """Set attribute on a link (works with both dict and dataclass objects)."""
    if isinstance(link, dict):
        link[key] = value
    else:
        setattr(link, key, value)
'''

# Insert _lset right after _lget definition
new_content = new_content.replace(
    '    return getattr(link, key, default)\n',
    '    return getattr(link, key, default)\n' + lset_helper,
    1
)

# Replace link["fidelity"] = with _lset(link, "fidelity",
assign_pattern = r'\b(link)\["(' + link_fields + r')"\]\s*=\s*'
def assign_replacer(m):
    var = m.group(1)
    field = m.group(2)
    return f'_lset({var}, "{field}", '

# We need to be careful - these are assignment lines like:
# link["fidelity"] = min(1.0, ...)
# We need: _lset(link, "fidelity", min(1.0, ...))
# But the right side extends to end of line, so we need to add closing paren

lines = new_content.split('\n')
new_lines = []
for line in lines:
    m = re.match(r'^(\s*)(link)\["(' + link_fields + r')"\]\s*=\s*(.+)$', line)
    if m:
        indent = m.group(1)
        var = m.group(2)
        field = m.group(3)
        value = m.group(4)
        new_lines.append(f'{indent}_lset({var}, "{field}", {value})')
    else:
        new_lines.append(line)

new_content = '\n'.join(new_lines)

# Count changes
orig_lines = content.split('\n')
changes = sum(1 for a, b in zip(orig_lines, new_content.split('\n')) if a != b)
print(f"Lines changed: {changes}")

with open(filepath, 'w') as f:
    f.write(new_content)

print("File updated successfully")
