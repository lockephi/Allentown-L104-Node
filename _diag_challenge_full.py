"""Diagnose ARC Challenge failures q0-199 with full detail."""
import requests
import json

url = 'https://datasets-server.huggingface.co/rows'
rows = []
for offset in [0, 100]:
    try:
        r = requests.get(url, params={
            'dataset': 'allenai/ai2_arc', 'config': 'ARC-Challenge',
            'split': 'test', 'offset': offset, 'length': 100
        }, timeout=30)
        rows.extend(r.json().get('rows', []))
    except Exception as e:
        print(f"Fetch error at offset {offset}: {e}")

print(f"Fetched {len(rows)} questions")

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

failures = []
correct = 0
correct_by_choice = [0, 0, 0, 0]
total_by_expected = [0, 0, 0, 0]

for i, item in enumerate(rows):
    row = item.get('row', item)
    q = row.get('question', '')
    choices = row.get('choices', {})
    labels = choices.get('label', [])
    texts = choices.get('text', [])
    answer = row.get('answerKey', '')

    exp_idx = labels.index(answer) if answer in labels else -1
    if exp_idx >= 0 and exp_idx < 4:
        total_by_expected[exp_idx] += 1

    result = engine.answer_mcq(q, texts)
    got = result.get('selected_index', -1)

    if got == exp_idx:
        correct += 1
        if got < 4:
            correct_by_choice[got] += 1
    else:
        failures.append({
            'idx': i,
            'q': q[:100],
            'got': got,
            'exp': exp_idx,
            'got_text': texts[got][:60] if got < len(texts) else '?',
            'exp_text': texts[exp_idx][:60] if exp_idx < len(texts) else '?',
        })

print(f"\nCorrect: {correct}/{len(rows)} = {100*correct/len(rows):.1f}%")
print(f"Correct by expected choice: {correct_by_choice}")
print(f"Total by expected choice:   {total_by_expected}")
print(f"Accuracy by choice: {[f'{100*c/t:.0f}%' if t > 0 else 'n/a' for c, t in zip(correct_by_choice, total_by_expected)]}")
print(f"\nFailures: {len(failures)}")

# Categorize failures by topic
topics = {
    'heat_transfer': [], 'phase_change': [], 'fact_opinion': [],
    'scientific_method': [], 'food_web': [], 'cell_biology': [],
    'solar_system': [], 'states_matter': [], 'buoyancy': [],
    'supply_demand': [], 'frequency': [], 'other': []
}

for f in failures:
    q = f['q'].lower()
    categorized = False
    if any(w in q for w in ['heat', 'temperature', 'warm', 'cool', 'hot', 'cold', 'ice']):
        topics['heat_transfer'].append(f)
        categorized = True
    if any(w in q for w in ['freeze', 'melt', 'boil', 'evaporate', 'condense', 'solid', 'liquid', 'gas']):
        topics['phase_change'].append(f)
        categorized = True
    if any(w in q for w in ['fact', 'opinion', 'true statement']):
        topics['fact_opinion'].append(f)
        categorized = True
    if any(w in q for w in ['investigation', 'experiment', 'scientific', 'research', 'method', 'hypothesis']):
        topics['scientific_method'].append(f)
        categorized = True
    if any(w in q for w in ['food web', 'food chain', 'predator', 'prey', 'population']):
        topics['food_web'].append(f)
        categorized = True
    if any(w in q for w in ['cell', 'nucleus', 'organelle', 'prokaryot', 'eukaryot', 'membrane']):
        topics['cell_biology'].append(f)
        categorized = True
    if any(w in q for w in ['solar system', 'planet', 'sun', 'moon', 'star', 'orbit']):
        topics['solar_system'].append(f)
        categorized = True
    if any(w in q for w in ['float', 'sink', 'buoyan', 'density', 'dense']):
        topics['buoyancy'].append(f)
        categorized = True
    if any(w in q for w in ['price', 'cost', 'supply', 'demand', 'economic']):
        topics['supply_demand'].append(f)
        categorized = True
    if not categorized:
        topics['other'].append(f)

print("\n=== FAILURE CATEGORIES ===")
for topic, items in sorted(topics.items(), key=lambda x: -len(x[1])):
    if items:
        print(f"\n{topic.upper()} ({len(items)} failures):")
        for f in items[:5]:
            print(f"  q{f['idx']}: {f['q'][:80]}")
            print(f"    Got[{f['got']}]: {f['got_text']}")
            print(f"    Exp[{f['exp']}]: {f['exp_text']}")

# Show ALL failures compactly
print("\n\n=== ALL FAILURES (compact) ===")
for i, f in enumerate(failures):
    print(f"  F{i}: q{f['idx']} Got[{f['got']}]:{f['got_text'][:40]} | Exp[{f['exp']}]:{f['exp_text'][:40]}")
