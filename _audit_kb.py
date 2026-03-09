"""Audit MMLUKnowledgeBase coverage against official MMLU subjects."""
import sys, os
sys.path.insert(0, os.getcwd())

from l104_asi.language_comprehension import MMLUKnowledgeBase
kb = MMLUKnowledgeBase()
kb.initialize()

MMLU_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
    'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
    'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations',
    'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions',
]

covered = set()
for key, node in kb.nodes.items():
    covered.add(node.subject)

missing = [s for s in MMLU_SUBJECTS if s not in covered]
thin = {}
for s in covered:
    keys = kb.subject_index.get(s, [])
    total_facts = sum(len(kb.nodes[k].facts) for k in keys if k in kb.nodes)
    if total_facts < 15:
        thin[s] = total_facts

print(f"Total nodes: {len(kb.nodes)}")
print(f"Covered subjects: {len(covered)}")
print(f"\nMissing MMLU subjects ({len(missing)}):")
for s in missing:
    print(f"  - {s}")

print(f"\nThin subjects (<15 facts): {len(thin)}")
for s, count in sorted(thin.items(), key=lambda x: x[1]):
    keys = kb.subject_index.get(s, [])
    print(f"  {s}: {count} facts ({len(keys)} nodes)")

if not thin:
    print("  *** ALL subjects have ≥15 facts! ***")

# Show all nodes grouped by subject with fact counts
print(f"\n--- Full subject breakdown ---")
total_facts_all = 0
for s in sorted(kb.subject_index.keys()):
    keys = kb.subject_index[s]
    total = sum(len(kb.nodes[k].facts) for k in keys if k in kb.nodes)
    total_facts_all += total
    node_names = [k.split("/")[1] if "/" in k else k for k in keys]
    print(f"  {s}: {total} facts, {len(keys)} nodes")

print(f"\nGrand total: {len(kb.nodes)} nodes, {total_facts_all} facts, {len(kb.subject_index)} subjects")
print(f"Relation edges: {sum(len(v) for v in kb.relation_graph.values()) // 2}")
