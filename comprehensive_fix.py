#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492537
"""Comprehensive fix script for indentation and syntax issues."""

import os
import re
import ast
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


def fix_indentation_issues(filepath):
    """Fix common indentation issues in Python files."""
    if not os.path.exists(filepath):
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False

    fixed_lines = []
    changed = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Check for statements that need indented blocks but next line isn't indented
        if i < len(lines) - 1:
            if stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ',
                                    'def ', 'class ', 'try:', 'except', 'with ',
                                    'async def ', 'async for ', 'async with ')):
                next_line = lines[i + 1]
                next_stripped = next_line.lstrip()
                next_indent = len(next_line) - len(next_stripped)

                # If next line is not empty and not indented more, fix it
                if next_stripped and not next_stripped.startswith('#'):
                    if next_indent <= indent:
                        # Check if it's a valid continuation (pass, return, etc)
                        if not next_stripped.startswith(('pass', 'return', 'break',
                                                         'continue', 'raise', 'yield')):
                            # Add proper indentation
                            fixed_lines.append(line)
                            fixed_lines.append(' ' * (indent + 4) + next_stripped)
                            changed = True
                            i += 2
                            continue

        # Fix lines that are incorrectly indented at column 0
        if indent == 0 and stripped:
            # Keywords that typically shouldn't be at column 0 unless at module level
            if stripped.startswith(('return ', 'yield ', 'break', 'continue',
                                   'except ', 'finally:', 'elif ', 'else:')):
                # Look back to find proper indentation level
                for j in range(i - 1, max(0, i - 10), -1):
                    prev_line = lines[j].lstrip()
                    if prev_line and not prev_line.startswith('#'):
                        prev_indent = len(lines[j]) - len(prev_line)
                        if prev_indent > 0:
                            fixed_lines.append(' ' * prev_indent + stripped)
                            changed = True
                            i += 1
                            continue
                            break

        fixed_lines.append(line)
        i += 1

    if changed:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            print(f"✓ Fixed indentation in {os.path.basename(filepath)}")
            return True
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
            return False

    return False

def validate_syntax(filepath):
    """Check if file has valid Python syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, e
    except Exception as e:
        return False, e

def main():
    root_dir = "/workspaces/Allentown-L104-Node"

    # Get all Python files in the root directory
    python_files = []
    for fname in os.listdir(root_dir):
        if fname.endswith('.py') and fname.startswith('l104_'):
            python_files.append(os.path.join(root_dir, fname))

    python_files.sort()

    print(f"Found {len(python_files)} l104_*.py files")
    print("=" * 60)

    fixed_count = 0
    error_files = []

    for filepath in python_files:
        basename = os.path.basename(filepath)

        # First check syntax
        valid, error = validate_syntax(filepath)

        if not valid:
            # Try to fix
            if fix_indentation_issues(filepath):
                fixed_count += 1
                # Revalidate
                valid, error = validate_syntax(filepath)
                if not valid:
                    error_files.append((basename, str(error)))
            else:
                error_files.append((basename, str(error)))

    print("=" * 60)
    print(f"Fixed {fixed_count} files")

    if error_files:
        print(f"\nStill have errors in {len(error_files)} files:")
        for fname, error in error_files[:10]:  # Show first 10
            print(f"  - {fname}: {error}")
    else:
        print("\n✓ All files have valid syntax!")

    return 0 if not error_files else 1

if __name__ == "__main__":
    sys.exit(main())
