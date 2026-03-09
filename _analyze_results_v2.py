#!/usr/bin/env python3
"""Analyze benchmark_full_online_results.json — answer distribution + confusion matrix."""
import json

data = json.load(open('benchmark_full_online_results.json'))
detailed = data.get('detailed_results', {})
benchmarks = data.get('benchmarks', {})

for bench_name in ['ARC', 'MMLU']:
    bench_data = detailed.get(bench_name, {})
    details = bench_data.get('details', [])
    if not details:
        continue
    got_dist = {}
    exp_dist = {}
    for r in details:
        g = r.get('predicted', -1)
        e = r.get('expected', -1)
        if isinstance(g, str): g = ord(g) - ord('A') if len(g)==1 else -1
        if isinstance(e, str): e = ord(e) - ord('A') if len(e)==1 else -1
        got_dist[g] = got_dist.get(g, 0) + 1
        exp_dist[e] = exp_dist.get(e, 0) + 1
    print(f'=== {bench_name} Answer Distribution ({len(details)} questions) ===')
    print(f'Got distribution:  {dict(sorted(got_dist.items()))}')
    print(f'Exp distribution:  {dict(sorted(exp_dist.items()))}')

    # Bias metric
    max_got = max(got_dist.values())
    min_got = min(v for k,v in got_dist.items() if k >= 0)
    print(f'Got max-min spread: {max_got - min_got}  (lower = less biased)')

    # Confusion matrix
    valid_keys = [k for k in set(list(got_dist.keys()) + list(exp_dist.keys())) if k >= 0]
    n = max(valid_keys, default=3) + 1
    matrix = [[0]*n for _ in range(n)]
    correct_per_choice = [0]*n
    total_per_choice = [0]*n
    for r in details:
        e = r.get('expected', -1)
        g = r.get('predicted', -1)
        if isinstance(g, str): g = ord(g) - ord('A') if len(g)==1 else -1
        if isinstance(e, str): e = ord(e) - ord('A') if len(e)==1 else -1
        if 0 <= e < n and 0 <= g < n:
            matrix[e][g] += 1
            total_per_choice[e] += 1
            if e == g:
                correct_per_choice[e] += 1
    print()
    header = '     ' + ''.join(f'Got{i:2d} ' for i in range(n))
    print(f'Confusion Matrix (row=expected, col=got):')
    print(header)
    for i, row in enumerate(matrix):
        cells = ''.join(f'{v:5d} ' for v in row)
        acc = correct_per_choice[i]/total_per_choice[i]*100 if total_per_choice[i] else 0
        print(f'Exp{i}  {cells}  ({acc:.0f}%)')
    print()

# Summary
print('=== SUMMARY ===')
for k in ['MMLU', 'ARC', 'MATH', 'HumanEval']:
    bench = benchmarks.get(k, {})
    score = bench.get('accuracy', bench.get('score', '?'))
    correct = bench.get('correct', '?')
    total = bench.get('total', '?')
    if isinstance(score, float):
        score = f'{score*100:.1f}%'
    print(f'{k:10s}: {correct}/{total} = {score}')
comp = data.get('composite_score', '?')
if isinstance(comp, float):
    comp = f'{comp*100:.1f}%'
print(f'{"COMPOSITE":10s}: {comp}')
