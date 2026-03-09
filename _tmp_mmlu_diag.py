#!/usr/bin/env python3
"""Show full details for all MMLU misses."""
import json

with open('.bench_online_cache.json') as f:
    cache = json.load(f)
with open('_bench_online_100_results.json') as f:
    res = json.load(f)

# MMLU questions start after ARC (index 0-59 are ARC)
mmlu_qs = cache['mmlu']
for r in res['mmlu']['results']:
    if not r['ok']:
        i = r['idx']
        q = mmlu_qs[i]
        got_l = chr(65 + r['got'])
        exp_l = chr(65 + r['expected'])
        subj = r.get('subject', q.get('subject', '?'))
        print(f'=== Q{i:02d} [{subj[:12]}] Got={got_l} Exp={exp_l} ===')
        print(f'  {q["question"][:200]}')
        for j, ch in enumerate(q['choices']):
            if j == r['expected']:
                m = '>>>'
            elif j == r['got']:
                m = 'XXX'
            else:
                m = '   '
            print(f'  {m} {chr(65+j)}: {ch}')
        print()
