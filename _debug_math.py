#!/usr/bin/env python3
"""Debug MATH benchmark failures."""
import logging, os, sys, importlib
logging.disable(logging.WARNING)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Direct import to avoid l104_asi.__init__ issues
spec = importlib.util.spec_from_file_location(
    "l104_asi.benchmark_harness",
    os.path.join(os.path.dirname(__file__), "l104_asi", "benchmark_harness.py")
)
bh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bh)
MATH_EXPANDED = bh.MATH_EXPANDED
_MATHRunner = bh._MATHRunner

runner = _MATHRunner()
solver = runner._get_solver()

correct = 0
failed = []
for i, sample in enumerate(MATH_EXPANDED):
    problem = sample['problem']
    expected = sample['answer']
    try:
        result = solver.solve(problem)
        answer = str(result.get('final_answer', ''))
        ok = runner._check_math_answer(answer, expected)
    except Exception as e:
        answer = f'ERROR: {e}'
        ok = False
    if ok:
        correct += 1
    else:
        failed.append((i, problem[:60], expected, answer[:40]))

print(f'Score: {correct}/{len(MATH_EXPANDED)} ({correct/len(MATH_EXPANDED)*100:.1f}%)')
print(f'\nFailed problems ({len(failed)}):')
for idx, prob, exp, got in failed:
    print(f'  [{idx}] {prob} => expected={exp}, got={got}')
