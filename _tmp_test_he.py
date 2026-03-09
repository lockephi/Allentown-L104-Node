#!/usr/bin/env python3
"""Quick test of HumanEval engine init and generation."""
import os, sys, time, traceback
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import logging; logging.disable(logging.WARNING)

t0 = time.time()
from l104_asi.code_generation import CodeGenerationEngine
eng = CodeGenerationEngine()
t1 = time.time()
print(f'Engine init: {t1-t0:.2f}s')

# Test generate_from_docstring
prompt = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n'

try:
    t2 = time.time()
    result = eng.generate_from_docstring(docstring=prompt, func_name='has_close_elements', func_signature='', test_cases=[])
    t3 = time.time()
    print(f'Generation: {t3-t2:.2f}s')
    print(f'Method: {result.get("method", "?")}')
    code = result.get('code', '')
    print(f'Code:\n{code[:300]}')

    # Try to run it
    full = prompt + code
    ns = {}
    exec(full, ns)
    fn = ns.get('has_close_elements')
    if fn:
        print(f'\nTest 1: {fn([1.0, 2.0, 3.0], 0.5)} (expected False)')
        print(f'Test 2: {fn([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)} (expected True)')
except Exception as e:
    traceback.print_exc()

# Quick sanity: count how many offline samples pass
from l104_asi.benchmark_harness import HUMANEVAL_SAMPLES
ok = 0
for s in HUMANEVAL_SAMPLES:
    try:
        gen = eng.generate_from_docstring(docstring=s['docstring'], func_name=s['func_name'], func_signature=s.get('signature',''), test_cases=s.get('tests', []))
        if gen.get('tests_passed', False):
            ok += 1
    except Exception:
        pass
print(f'\nOffline samples: {ok}/{len(HUMANEVAL_SAMPLES)} passed')
print(f'Total time: {time.time()-t0:.2f}s')
