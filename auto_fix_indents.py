#!/usr/bin/env python3
"""
Auto-fix common indentation patterns in Python files
"""
import re
from pathlib import Path

def fix_main_block_indentation(content):
    """Fix common pattern: extra indentation before if __name__ == "__main__":"""
    # Pattern: indented "if __name__" at module level
    pattern = r'^(\s+)(if __name__ == ["\']__main__["\']:)'
    
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Check if this is an indented __main__ block at wrong level
        if re.match(r'^\s{4,}if __name__', line):
            # Check if previous line is at module level (not indented or blank)
            if i > 0:
                prev_line = lines[i-1]
                # If previous line is not indented (module level), fix this line
                if not prev_line.strip() or not prev_line[0].isspace():
                    fixed_line = re.sub(r'^\s+', '', line)
                    fixed_lines.append(fixed_line)
                    continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_inconsistent_if_main_blocks(content):
    """Fix if __name__ blocks where content is under-indented"""
    lines = content.split('\n')
    fixed_lines = []
    in_main_block = False
    main_indent = 0
    
    for i, line in enumerate(lines):
        if 'if __name__' in line and line.strip().startswith('if __name__'):
            in_main_block = True
            main_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue
            
        if in_main_block and line.strip() and not line[0].isspace():
            # Line at module level ends the __main__ block
            in_main_block = False
            
        if in_main_block and line.strip():
            current_indent = len(line) - len(line.lstrip())
            expected_indent = main_indent + 4
            
            # If line is under-indented, fix it
            if current_indent < expected_indent and line.strip():
                spaces_needed = expected_indent - current_indent
                fixed_lines.append(' ' * spaces_needed + line)
                continue
                
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_main_block_indentation(content)
        content = fix_inconsistent_if_main_blocks(content)
        
        # Only write if changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Process all Python files"""
    root = Path('/workspaces/Allentown-L104-Node')
    py_files = list(root.glob('*.py'))
    
    fixed_count = 0
    for py_file in py_files:
        if py_file.name == 'auto_fix_indents.py':
            continue
        if process_file(py_file):
            print(f"✓ Fixed {py_file.name}")
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()
