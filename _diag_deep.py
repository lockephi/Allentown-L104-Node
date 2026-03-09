#!/usr/bin/env python3
"""Deep diagnostic: understand WHY close-race failures occur."""
import os, json, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"
import urllib.request

def fetch(config, n=50, offset=0):
    url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config={config}&split=test&offset={offset}&length={n}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "L104/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except:
            import time; time.sleep(2)

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

batch = fetch("ARC-Easy", 50, 0)
rows = batch.get('rows', [])

failures = []
correct_count = 0

for row_data in rows:
    r = row_data.get('row', {})
    q = r.get('question', '')
    choices_data = r.get('choices', {})
    texts = choices_data.get('text', [])
    answer_key = r.get('answerKey', '')
    if not q or not texts:
        continue
    labels = choices_data.get('label', [])
    expected_idx = labels.index(answer_key) if answer_key in labels else -1
    if expected_idx < 0:
        continue
    result = engine.answer_mcq(q, texts)
    pred_idx = result.get('selected_index', 0)
    ok = pred_idx == expected_idx
    if ok:
        correct_count += 1
    else:
        all_scores = result.get('all_scores', {})
        score_vals = sorted(all_scores.values(), reverse=True)
        gap = (score_vals[0] - score_vals[1]) if len(score_vals) > 1 else 0

        # Determine failure type
        ftype = "close_race"
        if max(score_vals) < 0.5:
            ftype = "no_signal"
        elif gap > score_vals[1] * 0.5:
            ftype = "confident_wrong"

        failures.append({
            'q': q,
            'choices': texts,
            'pred_idx': pred_idx,
            'pred': texts[pred_idx] if pred_idx < len(texts) else '?',
            'exp_idx': expected_idx,
            'exp': texts[expected_idx],
            'scores': all_scores,
            'gap': gap,
            'type': ftype,
            'quantum': result.get('quantum', {}),
            'concepts': result.get('concepts_found', []),
        })

print(f"Correct: {correct_count}/50")
print(f"\n=== FAILURE BREAKDOWN ({len(failures)} failures) ===")

# Categorize
types = {}
for f in failures:
    types[f['type']] = types.get(f['type'], 0) + 1
for t, n in sorted(types.items()):
    print(f"  {t}: {n}")

# Detailed analysis of each failure
print(f"\n=== DETAILED FAILURE ANALYSIS ===")
for i, f in enumerate(failures):
    print(f"\n{'='*60}")
    print(f"#{i+1} [{f['type']}] Q: {f['q'][:100]}")
    print(f"  PREDICTED: [{chr(65+f['pred_idx'])}] {f['pred'][:60]}")
    print(f"  EXPECTED:  [{chr(65+f['exp_idx'])}] {f['exp'][:60]}")
    print(f"  Concepts: {f['concepts'][:6]}")
    print(f"  Scores:")
    for label, score in sorted(f['scores'].items(), key=lambda x: -x[1]):
        marker = " <<<WRONG" if label == chr(65+f['pred_idx']) else (" <<<CORRECT" if label == chr(65+f['exp_idx']) else "")
        print(f"    {label}: {score:.4f}{marker}")

    # Show expected vs predicted score gap
    exp_label = chr(65+f['exp_idx'])
    pred_label = chr(65+f['pred_idx'])
    exp_score = f['scores'].get(exp_label, 0)
    pred_score = f['scores'].get(pred_label, 0)
    print(f"  Gap: predicted={pred_score:.4f} expected={exp_score:.4f} diff={pred_score-exp_score:.4f}")

    # Identify what kind of question this is
    q_lower = f['q'].lower()
    if 'what cause' in q_lower or 'what happen' in q_lower:
        print(f"  Type: CAUSE_EFFECT")
    elif 'which' in q_lower:
        print(f"  Type: SELECTION/CLASSIFICATION")
    elif 'what is' in q_lower or 'what are' in q_lower:
        print(f"  Type: DEFINITION")
    elif 'how' in q_lower:
        print(f"  Type: MECHANISM")
    elif 'why' in q_lower:
        print(f"  Type: EXPLANATION")
