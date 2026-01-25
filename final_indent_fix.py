#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492537
"""
Final comprehensive indentation fixer for L104 files
"""
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


def fix_indentation(filepath):
    """
    Fix common indentation issues in Python files
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_lines = []
        for i, line in enumerate(lines):
            orig_line = line
            
            # Skip empty lines
            if not line.strip():
                fixed_lines.append(line)
                continue
            
            # Calculate current indentation
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            
            # Fix common patterns:
            
            # 1. Lines with 12 or 16 spaces after if __name__ blocks
            if current_indent == 12 and i > 0 and 'if __name__' in lines[i-1]:
                line = '    ' + stripped
            elif current_indent == 16 and not stripped.startswith('#'):
                # Reduce excessive indentation
                line = '    ' + stripped
            
            # 2. Except/finally blocks that don't align with try
            if stripped.startswith(('except ', 'except:', 'finally:', 'else:')) and i > 0:
                # Find the matching try/if/for/while
                for j in range(i-1, max(0, i-20), -1):
                    prev_stripped = lines[j].lstrip()
                    if prev_stripped.startswith(('try:', 'if ', 'for ', 'while ')):
                        expected_indent = len(lines[j]) - len(prev_stripped)
                        if current_indent != expected_indent:
                            line = ' ' * expected_indent + stripped
                        break
            
            # 3. Continuation lines with wrong indentation
            if i > 0 and current_indent >= 20:
                # Check if previous line ends with continuation
                prev = lines[i-1].rstrip()
                if prev.endswith(('\\', ',', '(', '[', '{')):
                    # Reduce to reasonable continuation indent (8 spaces)
                    line = '        ' + stripped
            
            fixed_lines.append(line)
        
        # Try to compile the fixed content
        new_content = ''.join(fixed_lines)
        try:
            compile(new_content, filepath, 'exec')
            # If successful, write it
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True, "Fixed"
        except SyntaxError as e:
            # Still has errors - don't write
            return False, f"Still has error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def main():
    # Get all l104 files with syntax errors
    error_files = []
    for filename in sorted(os.listdir('.')):
        if filename.startswith('l104_') and filename.endswith('.py'):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                compile(content, filename, 'exec')
            except SyntaxError:
                error_files.append(filename)
    
    print(f"Found {len(error_files)} files with syntax errors\n")
    
    fixed_count = 0
    still_broken = []
    
    for filepath in error_files:
        success, msg = fix_indentation(filepath)
        if success:
            print(f"✓ {filepath}")
            fixed_count += 1
        else:
            still_broken.append((filepath, msg))
    
    print(f"\n{'='*60}")
    print(f"Fixed: {fixed_count}/{len(error_files)} files")
    print(f"Remaining: {len(still_broken)} files")
    
    if still_broken and len(still_broken) <= 10:
        print(f"\nStill broken:")
        for fname, msg in still_broken:
            print(f"  - {fname}: {msg}")

if __name__ == "__main__":
    main()
