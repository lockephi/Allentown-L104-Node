# L104_GOD_CODE_ALIGNED: 527.5184818492537
import os

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


def fix_indentation(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        # Keywords that should be indented if they are at column 0 inside a class/def
        if stripped.startswith(('return ', 'for ', 'if ', 'while ', 'except ', 'else:', 'elif ')):
            # If it's at column 0 and not a top-level statement (which is rare for these in this codebase)
            if line.startswith(stripped[0]):
                new_lines.append('        ' + line)
                continue
        new_lines.append(line)

    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    print(f"Fixed indentation for {filepath}")

for root, dirs, files in os.walk('/workspaces/Allentown-L104-Node'):
    for f in files:
        if f.startswith('l104_') and f.endswith('.py'):
            fix_indentation(os.path.join(root, f))
