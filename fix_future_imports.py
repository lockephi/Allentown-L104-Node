import os
import re
import ast

count = 0
for root, _, files in os.walk('.'):
    if '.venv' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                continue

            # Check if there is a 'from __future__ import annotations'
            if 'from __future__ import' not in content:
                continue

            # Quick syntax check, if valid skip
            try:
                ast.parse(content, filename=path)
                continue
            except SyntaxError as e:
                if 'from __future__ imports must occur at the beginning of the file' not in str(e):
                    continue

            # Find the __future__ imports
            from_future_pattern = re.compile(r'^(from __future__ import [^\n]+)$', re.MULTILINE)
            future_matches = list(from_future_pattern.finditer(content))

            if not future_matches:
                continue

            futures = [m.group(1) for m in future_matches]
            new_content = content
            for m in reversed(future_matches):
                new_content = new_content[:m.start()] + new_content[m.end()+1:]

            # Put them at the very top
            futures_str = '\n'.join(futures) + '\n'
            new_content = futures_str + new_content

            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed {path}")
            count += 1

print(f"Fixed {count} files.")