import re
import os

def fix_indentation(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this line is a top-level def/class that is indented
        if re.match(r'^    (def|async def|class) ', line):
            # This is a block that needs unindenting
            new_lines.append(line[4:])
            i += 1
            # Unindent all subsequent lines that are indented by at least 4 spaces
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip() == "":
                    new_lines.append(next_line)
                    i += 1
                    continue
                if next_line.startswith('    '):
                    new_lines.append(next_line[4:])
                    i += 1
                else:
                    # End of block
                    break
        else:
            new_lines.append(line)
            i += 1
            
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    print(f"Fixed block indentation for {filepath}")

def fix_joined_imports(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix import os
import base64 -> import os\nimport base64
    content = re.sub(r'(import [a-zA-Z0-9_]+)(import [a-zA-Z0-9_]+)', r'\1\n\2', content)
    content = re.sub(r'([a-zA-Z0-9_]+)(from [a-zA-Z0-9_.]+ import)', r'\1\n\2', content)
    content = re.sub(r'(import [a-zA-Z0-9_]+)(def )', r'\1\n\2', content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed joined imports for {filepath}")

if __name__ == "__main__":
    target_files = ['main.py', 'master.py', 'enhance.py', 'enhance.py.fixed']
    for f in target_files:
        path = os.path.join('/workspaces/Allentown-L104-Node', f)
        if os.path.exists(path):
            fix_joined_imports(path)
            fix_indentation(path)
