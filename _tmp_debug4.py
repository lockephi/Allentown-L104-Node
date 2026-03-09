#!/usr/bin/env python3
"""Debug the 4 remaining failing HumanEval problems."""
import sys, os, ast, signal
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging; logging.disable(logging.WARNING)
sys.path.insert(0, '.')
import importlib
bh = importlib.import_module('l104_asi.benchmark_harness')

problems = bh._HuggingFaceFetcher.fetch_humaneval()
runner = bh._HumanEvalRunner()
engine = runner._get_engine()

targets = {'largest_prime_factor', 'sort_array', 'sum_squares'}
# Skip find_zero for now — may hang

for prob in problems:
    fn = prob['entry_point']
    if fn not in targets:
        continue
    task = prob['task_id']
    prompt = prob['prompt']
    test_code = prob['test']

    gen = engine.generate_from_docstring(docstring=prompt, func_name=fn, func_signature="", test_cases=[])
    code = gen.get("code", "")
    method = gen.get("method", "?")
    full_code = prompt + code

    print(f"\n{'='*60}")
    print(f"{task} ({fn}) [{method}]")
    print(f"Generated body:\n{code[:300]}")

    try:
        ns = {"__builtins__": __builtins__}
        exec(full_code, ns)
        exec(test_code, ns)
        if "check" in ns and callable(ns["check"]):
            ns["check"](ns.get(fn))
            print("PASSED")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        # For more detail, try some test cases manually
        if fn == 'largest_prime_factor':
            try:
                ns2 = {"__builtins__": __builtins__}
                exec(full_code, ns2)
                f = ns2[fn]
                for x in [15, 13195, 2048, 7]:
                    print(f"  {fn}({x}) = {f(x)}")
            except Exception as e2:
                print(f"  Manual test failed: {e2}")
        elif fn == 'sort_array':
            try:
                ns2 = {"__builtins__": __builtins__}
                exec(full_code, ns2)
                f = ns2[fn]
                print(f"  sort_array([1,5,2,3,4]) = {f([1,5,2,3,4])}")
                print(f"  sort_array([-2,-3,-4,-5,-6]) = {f([-2,-3,-4,-5,-6])}")
                print(f"  sort_array([1,0,2,3,4]) = {f([1,0,2,3,4])}")
            except Exception as e2:
                print(f"  Manual test failed: {e2}")
        elif fn == 'find_zero':
            try:
                ns2 = {"__builtins__": __builtins__}
                exec(full_code, ns2)
                f = ns2[fn]
                r = f([1, 2])
                print(f"  find_zero([1,2]) = {r}")
                r = f([-6, 11, -6, 1])
                print(f"  find_zero([-6,11,-6,1]) = {r}")
            except Exception as e2:
                print(f"  Manual test failed: {e2}")
        elif fn == 'sum_squares':
            try:
                ns2 = {"__builtins__": __builtins__}
                exec(full_code, ns2)
                f = ns2[fn]
                print(f"  sum_squares([1,2,3]) = {f([1,2,3])}")
                print(f"  sum_squares([1,4,9]) = {f([1,4,9])}")
            except Exception as e2:
                print(f"  Manual test failed: {e2}")
