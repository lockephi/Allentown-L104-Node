#!/usr/bin/env python3
"""Debug LI KB scoring for specific questions"""
import sys, re
sys.path.insert(0, '.')
from l104_asi.commonsense_reasoning import _get_cached_local_intellect

li = _get_cached_local_intellect()
q = 'All living and nonliving material is composed of'
q_content = re.sub(r'\b(which|what|who|is|are|the|of|following|a|an)\b', '', q.lower()).strip()

for ch in ['cells', 'elements', 'water', 'oxygen']:
    results = li._search_training_data(f'{q_content} {ch}', max_results=5)

    hits = []
    if isinstance(results, list):
        for entry in results:
            if isinstance(entry, dict):
                c = entry.get('completion', '')
                if len(c) > 10:
                    hits.append(c[:100])

    # Simulate LI scoring
    choice_words = {w for w in re.findall(r'\w+', ch.lower()) if len(w) > 2}
    q_content_words = {w for w in re.findall(r'\w+', q.lower()) if len(w) > 3}

    li_total = 0.0
    for fact in hits:
        fl = fact.lower()
        q_in_fact = sum(1 for w in q_content_words if w in fl)
        c_in_fact = sum(1 for w in choice_words if w in fl)
        if q_in_fact >= 1 and c_in_fact >= 1:
            li_total += min(q_in_fact, 3) * min(c_in_fact, 2) * 0.12
        if len(ch) > 4 and ch in fl and q_in_fact >= 1:
            li_total += 0.25

    print(f'{ch:12s} li_total={li_total:.3f}  hits={len(hits)}')
    for h in hits[:2]:
        print(f'  {h[:90]}')
