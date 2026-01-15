import re
import os

def repair_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Join split words with underscores
    content = re.sub(r'([a-zA-Z0-9]+_)\s*\n\s*([a-zA-Z0-9]+)', r'\1\2', content)

    # 2. fix common splits
    content = re.sub(r'starts\s*\n\s*with', 'startswith', content)
    content = re.sub(r'el\s*\n\s*if', 'elif', content)
    
    # 3. Join split keywords
    lines = content.split('\n')
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if i + 1 < len(lines):
            next_line = lines[i+1]
            next_stripped = next_line.strip()
            
            if stripped in ('await', 'return', 'yield', 'from', 'with', 'import'):
                line = line.rstrip() + ' ' + next_stripped
                i += 1
            elif stripped.startswith('from ') and 'import' not in stripped and 'import' in next_stripped:
                line = line.rstrip() + ' ' + next_stripped
                i += 1
        
        new_lines.append(line)
        i += 1

    content = '\n'.join(new_lines)
    
    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    import sys
    
    workspace_root = '/workspaces/Allentown-L104-Node'
    
    if len(sys.argv) > 1:
        repair_file(sys.argv[1])
    else:
        for root, dirs, files in os.walk(workspace_root):
            if '.venv' in root or 'new_venv' in root:
                continue
            for filename in files:
                if (filename.startswith('l104_') and filename.endswith('.py')) or filename == 'main.py':
                    filepath = os.path.join(root, filename)
                    repair_file(filepath)
