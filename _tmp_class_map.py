#!/usr/bin/env python3
import ast
with open("l104_quantum_magic.py") as f:
    tree = ast.parse(f.read())
classes = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        bases = [getattr(b, "id", getattr(b, "attr", "?")) for b in node.bases]
        classes.append((node.lineno, node.end_lineno or 0, node.name, bases, len(methods)))
classes.sort(key=lambda x: x[0])
print(f"Classes: {len(classes)}")
print(f"{'Line':>6} {'End':>6} {'Methods':>7}  Class")
print("-" * 60)
for ln, end, name, bases, mc in classes:
    b = "(" + ",".join(bases) + ")" if bases else ""
    print(f"{ln:6} {end:6} {mc:7}  {name}{b}")
