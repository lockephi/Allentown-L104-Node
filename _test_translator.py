#!/usr/bin/env python3
"""Test the AST-based CodeTranslator."""
import sys
sys.path.insert(0, '/Users/carolalvarez/Applications/Allentown-L104-Node')

from l104_code_engine import code_engine

test_code = '''
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> str:
    return "Hello, " + name

class Calculator:
    def __init__(self, value: float = 0.0):
        self.value = value

    def add(self, x: float) -> float:
        self.value += x
        return self.value

for i in range(10):
    if i % 2 == 0:
        print(i)
'''

targets = ['rust', 'javascript', 'typescript', 'swift', 'go', 'java']

for lang in targets:
    print(f'\n{"=" * 60}')
    print(f'TARGET: {lang.upper()}')
    print("=" * 60)
    result = code_engine.translate_code(test_code, 'python', lang)
    if 'error' in result:
        print(f"ERROR: {result['error']}")
    else:
        print(result.get('translated', 'NO OUTPUT'))
    if result.get('warnings'):
        print(f"WARNINGS: {result['warnings']}")
