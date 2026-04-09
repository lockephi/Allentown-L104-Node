#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from l104_code_engine import code_engine

with open('l104_code_engine/synthesis.py', 'r') as f:
    source = f.read()

result = code_engine.generate_tests(source, module_name='l104_code_engine.synthesis')
if result['success']:
    test_code = result['test_code']
    with open('l104_code_engine/test_synthesis_new.py', 'w') as out:
        out.write(test_code)
    print(f"Generated {len(test_code)} characters")
    print("First 500 chars:")
    print(test_code[:500])
else:
    print("Failed:", result.get('error'))
    sys.exit(1)