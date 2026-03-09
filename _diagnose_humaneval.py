#!/usr/bin/env python3
"""Diagnose which HumanEval problems fail using online data (with per-problem timeout)."""
import sys, os, ast, signal
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, '.')
import importlib
bh = importlib.import_module('l104_asi.benchmark_harness')

class TimeoutErr(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutErr("Timeout")

# Fetch from HuggingFace
print("Fetching HumanEval from HuggingFace...", flush=True)
problems = bh._HuggingFaceFetcher.fetch_humaneval()
print(f"Got {len(problems)} problems", flush=True)

runner = bh._HumanEvalRunner()
engine = runner._get_engine()

passed = 0
failed = 0
fail_list = []

for prob in problems:
    task_id = prob.get("task_id", "unknown")
    prompt = prob.get("prompt", "")
    entry_point = prob.get("entry_point", "")
    test_code = prob.get("test", "")

    is_passed = False
    err_detail = ""
    method = "?"

    # Set 5-second timeout per problem
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)

    try:
        if engine is not None:
            gen = engine.generate_from_docstring(
                docstring=prompt,
                func_name=entry_point,
                func_signature="",
                test_cases=[],
            )
            generated_code = gen.get("code", "")
            method = gen.get("method", "?")
            full_code = prompt + generated_code
            try:
                ast.parse(full_code)
                ns = {"__builtins__": __builtins__}
                exec(full_code, ns)
                exec(test_code, ns)
                if "check" in ns and callable(ns["check"]):
                    ns["check"](ns.get(entry_point))
                    is_passed = True
            except AssertionError as e:
                err_detail = f"ASSERT: {str(e)[:80]}"
            except Exception as e:
                err_detail = f"{type(e).__name__}: {str(e)[:80]}"
    except TimeoutErr:
        err_detail = "TIMEOUT (>5s)"
    except Exception as e:
        err_detail = f"GEN: {type(e).__name__}: {str(e)[:80]}"
    finally:
        signal.alarm(0)

    if is_passed:
        passed += 1
    else:
        failed += 1
        fail_list.append((task_id, entry_point, method, err_detail))

    # Print progress every 20 problems
    total = passed + failed
    if total % 20 == 0:
        print(f"  Progress: {total}/{len(problems)} ({passed} passed)", flush=True)

print(f"\n{'='*60}")
print(f"HumanEval: {passed}/{passed+failed} passed ({100*passed/(passed+failed):.1f}%)")
print(f"{'='*60}")
print(f"\nFailed problems ({failed}):")
for task_id, fn, method, err in fail_list:
    print(f"  {task_id} ({fn}) [{method}]: {err}")
