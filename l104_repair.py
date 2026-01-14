import re

def repair_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Join split words with underscores (e.g., get_electron_\nmatrix)
    content = re.sub(r'([a-zA-Z0-9]+_)\s*\n\s*([a-zA-Z0-9]+)', r'\1\2', content)

    # 2. Join split keywords that were likely one line
    # Common splits: starts\nwith, el\nif, with\nGoogle, from\nSovereign, if\nauto-approve, etc.
    
    # fix startsWith
    content = re.sub(r'starts\s*\n\s*with', 'startswith', content)
    # fix elif
    content = re.sub(r'el\s*\n\s*if', 'elif', content)
    # fix async def
    # content = re.sub(r'async\s*\n\s*def', 'async def', content)
    
    # 3. Join split comments or keywords on new lines that should be joined
    # This is tricky without breaking everything.
    
    # Try to join lines that start with a keyword that shouldn't be at the start of a line in context
    # or where the previous line ends with a word that expects more.
    
    lines = content.split('\n')
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # If this line ends with a word and the next line starts with a word that completes it
        if i + 1 < len(lines):
            next_line = lines[i+1].strip()
            
            # Common patterns observed in the broken file
            # Pattern: comment \n continued_comment
            if line.strip().startswith('#') and not next_line.startswith('#') and next_line:
                # Check if next_line should have been part of the comment
                # (This is risky, but a lot of comments are broken)
                pass # skip for now
            
            # Pattern: from ... import ... split
            if line.strip().startswith('from ') and 'import' not in line and 'import' in next_line:
                line = line + ' ' + next_line
                i += 1
            
            # Pattern: await + split
            elif line.strip() == 'await' or line.strip() == 'return' or line.strip() == 'yield':
                line = line + ' ' + next_line
                i += 1
                
        new_lines.append(line)
        i += 1

    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        repair_file(sys.argv[1])
    else:
        repair_file('/workspaces/Allentown-L104-Node/main.py')
