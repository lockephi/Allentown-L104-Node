"""Check KB coverage for worst-performing MMLU subjects."""
import sys
sys.path.insert(0, '.')
from l104_asi.language_comprehension import MMLUKnowledgeBase
from collections import Counter

kb = MMLUKnowledgeBase()
kb.initialize()

subj_counts = Counter()
for key, node in kb.nodes.items():
    s = node.subject if hasattr(node, 'subject') else ''
    if isinstance(s, list):
        for sub in s:
            subj_counts[sub] += 1
    elif s:
        subj_counts[s] += 1

zero_pct = ['global_facts', 'professional_law', 'public_relations']
low_pct = ['college_chemistry', 'college_physics', 'computer_security',
    'high_school_geography', 'high_school_government_and_politics',
    'high_school_mathematics', 'human_sexuality', 'prehistory', 'world_religions']

print('=== 0% Subjects KB Coverage ===')
for s in zero_pct:
    print(f'  {s}: {subj_counts.get(s, 0)} nodes')

print('\n=== Low% Subjects KB Coverage ===')
for s in low_pct:
    print(f'  {s}: {subj_counts.get(s, 0)} nodes')

print(f'\nTotal nodes: {len(kb.nodes)}')
print(f'Subjects with nodes: {len(subj_counts)}')
print(f'\nTop 10 subjects by coverage:')
for s, c in subj_counts.most_common(10):
    print(f'  {s}: {c}')
print(f'\nBottom 10 subjects by coverage:')
for s, c in subj_counts.most_common()[-10:]:
    print(f'  {s}: {c}')
