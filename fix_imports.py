#!/usr/bin/env python3
"""
Fix import statements in routers/consciousness.py
"""
import re

def fix_consciousness_imports():
    with open('routers/consciousness.py', 'r') as f:
        content = f.read()
    
    # Fix all imports from l104_consciousness_substrate
    pattern = r'from l104_consciousness_substrate import consciousness_substrate(?:, (.*?))?\n'
    
    def replace_import(match):
        additional_imports = match.group(1)
        if additional_imports:
            return f'from l104_consciousness_substrate import get_consciousness_substrate, {additional_imports}\n'
        else:
            return 'from l104_consciousness_substrate import get_consciousness_substrate\n'
    
    content = re.sub(pattern, replace_import, content)
    
    # Now fix the function calls to use get_consciousness_substrate()
    # Find all functions that use consciousness_substrate directly
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check if line contains consciousness_substrate. but not get_consciousness_substrate
        if 'consciousness_substrate.' in line and 'get_consciousness_substrate' not in line:
            # This is a function call that needs fixing
            # We need to add consciousness_substrate = get_consciousness_substrate() at the beginning of the function
            # For now, just note it
            print(f"Line needs manual fix: {line}")
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Join back
    content = '\n'.join(fixed_lines)
    
    # Write back
    with open('routers/consciousness.py', 'w') as f:
        f.write(content)
    
    print("Fixed consciousness substrate imports")

def fix_intricate_orchestrator_imports():
    with open('routers/consciousness.py', 'r') as f:
        content = f.read()
    
    # Fix imports from l104_intricate_orchestrator
    pattern = r'from l104_intricate_orchestrator import intricate_orchestrator\n'
    replacement = 'from l104_intricate_orchestrator import get_intricate_orchestrator\n'
    
    content = re.sub(pattern, replacement, content)
    
    # Write back
    with open('routers/consciousness.py', 'w') as f:
        f.write(content)
    
    print("Fixed intricate orchestrator imports")

if __name__ == '__main__':
    fix_consciousness_imports()
    fix_intricate_orchestrator_imports()