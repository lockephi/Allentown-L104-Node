#!/usr/bin/env python3
"""Debug regex patterns for remaining MATH failures."""
import re, math

# Test mod - problem 22
clean = 'what is 17 mod 5?'
m = re.search(r'(\d+)\s+mod\s+(\d+)', clean)
print(f'Mod match: {m.groups() if m else None}')

# Test factorial expression - problem 40
clean = 'what is 10!/(8!*2!)?'
m = re.search(r'(\d+)!\s*/\s*\(?\s*(\d+)!\s*\*?\s*(\d+)!\s*\)?', clean)
print(f'Factorial expr: {m.groups() if m else None}')

# Test choose - problem 43
clean = 'what is 8 choose 2?'
m = re.search(r'(\d+)\s+choose\s+(\d+)', clean)
print(f'Choose: {m.groups() if m else None}')

# Test function eval - problem 18
clean = 'evaluate f(3) if f(x) = x^2 - 2x + 1'
m = re.search(r'(?:f|g)\((\d+)\)\s+(?:if|where|when)\s+(?:f|g)\(x\)\s*=\s*(.+)', clean)
print(f'Func eval: {m.groups() if m else None}')
if m:
    x_val = float(m.group(1))
    expr_str = m.group(2).strip().rstrip('?')
    expr_str = expr_str.replace('^', '**').replace('x', f'({x_val})')
    print(f'  expr: {expr_str}')
    result = eval(expr_str, {'__builtins__': {}}, {})
    print(f'  result: {result}')

# Test composition - problem 48
clean = 'if f(x) = 2x + 1 and g(x) = x^2, what is f(g(3))?'
m = re.search(r'(?:if|where|when)\s+f\(x\)\s*=\s*(.+?)\s+and\s+g\(x\)\s*=\s*(.+?)(?:,|\s+what)', clean)
print(f'Composition: {m.groups() if m else None}')
if m:
    f_expr = m.group(1).strip().replace('^', '**')
    g_expr = m.group(2).strip().replace('^', '**')
    val_m = re.search(r'f\(g\((\d+)\)\)', clean)
    if val_m:
        x_val = float(val_m.group(1))
        g_val = eval(g_expr.replace('x', f'({x_val})'), {'__builtins__': {}}, {})
        print(f'  g_expr={g_expr}, g_val={g_val}')
        result = eval(f_expr.replace('x', f'({g_val})'), {'__builtins__': {}}, {})
        print(f'  f_expr={f_expr}, result={result}')
