import ast, os
for root, _, files in os.walk('.'):
    for f in files:
        if f.endswith('.py') and '.venv' not in root:
            path = os.path.join(root, f)
            try:
                ast.parse(open(path, encoding='utf-8').read(), filename=path)
            except SyntaxError as e:
                print(f"Error in {path}: {e}")
