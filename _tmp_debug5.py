#!/usr/bin/env python3
"""Debug the 4 remaining failing HumanEval problems with per-problem timeouts."""
import sys, os, signal, traceback
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging; logging.disable(logging.WARNING)
sys.path.insert(0, '.')

print("Loading modules...", flush=True)
import importlib
bh = importlib.import_module('l104_asi.benchmark_harness')
print("Fetching problems...", flush=True)
problems = bh._HuggingFaceFetcher.fetch_humaneval()
print(f"Loaded {len(problems)} problems", flush=True)

runner = bh._HumanEvalRunner()
engine = runner._get_engine()
print("Engine ready\n", flush=True)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timed out")

targets = {'find_zero', 'largest_prime_factor', 'sort_array', 'sum_squares'}
print(f"Targets: {targets}", flush=True)

for prob in problems:
    fn = prob['entry_point']
    if fn not in targets:
        continue
    task = prob['task_id']
    prompt = prob['prompt']
    test_code = prob['test']

    print(f"{'='*60}", flush=True)
    print(f"Testing {task} ({fn})", flush=True)

    # Generate code
    gen = engine.generate_from_docstring(docstring=prompt, func_name=fn, func_signature="", test_cases=[])
    code = gen.get("code", "")
    method = gen.get("method", "?")
    full_code = prompt + code

    print(f"  Method: {method}", flush=True)
    print(f"  Code preview: {code[:200]}", flush=True)

    # Run with 10s timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    try:
        ns = {"__builtins__": __builtins__}
        exec(full_code, ns)
        exec(test_code, ns)
        if "check" in ns and callable(ns["check"]):
            ns["check"](ns.get(fn))
            print(f"  RESULT: PASSED ✓", flush=True)
    except TimeoutError:
        print(f"  RESULT: TIMEOUT (hung for 10s)", flush=True)
    except AssertionError as e:
        print(f"  RESULT: ASSERTION FAILED: {e}", flush=True)
        # Try manual test cases
        try:
            ns2 = {"__builtins__": __builtins__}
            exec(full_code, ns2)
            f = ns2.get(fn)
            if f:
                if fn == 'largest_prime_factor':
                    for x in [15, 13195, 2048, 7, 13]:
                        signal.alarm(3)
                        try:
                            print(f"    {fn}({x}) = {f(x)}", flush=True)
                        except TimeoutError:
                            print(f"    {fn}({x}) = TIMEOUT", flush=True)
                elif fn == 'sort_array':
                    cases = [[1,5,2,3,4], [-2,-3,-4,-5,-6], [1,0,2,3,4], []]
                    for c in cases:
                        signal.alarm(3)
                        try:
                            print(f"    {fn}({c}) = {f(c)}", flush=True)
                        except TimeoutError:
                            print(f"    {fn}({c}) = TIMEOUT", flush=True)
                elif fn == 'sum_squares':
                    cases = [[1,2,3], [1,4,9], [-1,1,0]]
                    for c in cases:
                        signal.alarm(3)
                        try:
                            print(f"    {fn}({c}) = {f(c)}", flush=True)
                        except TimeoutError:
                            print(f"    {fn}({c}) = TIMEOUT", flush=True)
                elif fn == 'find_zero':
                    cases = [[1, 2], [-6, 11, -6, 1]]
                    for c in cases:
                        signal.alarm(3)
                        try:
                            print(f"    {fn}({c}) = {f(c)}", flush=True)
                        except TimeoutError:
                            print(f"    {fn}({c}) = TIMEOUT", flush=True)
        except Exception as e2:
            print(f"    Manual test error: {e2}", flush=True)
    except Exception as e:
        print(f"  RESULT: ERROR: {type(e).__name__}: {e}", flush=True)
    finally:
        signal.alarm(0)

    print(flush=True)

print("\nDone!", flush=True)
