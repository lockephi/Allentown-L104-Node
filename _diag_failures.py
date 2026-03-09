#!/usr/bin/env python3
"""Analyze specific ARC failures to find algorithmic improvement targets."""
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

# Analyze failures on Easy q0-49
batch = fetch("ARC-Easy", 50, 0)
rows = batch.get('rows', [])

failures = []
score_gaps = []  # (gap, correct)

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

    all_scores = result.get('all_scores', {})
    score_vals = sorted(all_scores.values(), reverse=True)
    gap = (score_vals[0] - score_vals[1]) if len(score_vals) > 1 else 0
    score_gaps.append((gap, ok))

    if not ok:
        failures.append({
            'q': q[:80],
            'pred': texts[pred_idx] if pred_idx < len(texts) else '?',
            'exp': texts[expected_idx],
            'scores': all_scores,
            'gap': gap,
            'quantum': result.get('quantum', {}),
            'concepts': result.get('concepts_found', []),
        })

print(f"Correct: {50 - len(failures)}/50")
print(f"\n=== FAILURE ANALYSIS ({len(failures)} failures) ===")

# Categorize failures
quantum_dominated = 0
close_race = 0
wrong_concept = 0
no_signal = 0

for f in failures:
    scores = f['scores']
    vals = sorted(scores.values(), reverse=True)

    # Check quantum influence
    qp = f['quantum'].get('quantum_probability', 0)
    if qp >= 0.9:
        quantum_dominated += 1

    # Check if it was a close race
    if len(vals) >= 2 and vals[0] < vals[1] * 1.5:
        close_race += 1

    # Check signal strength
    if max(vals) < 0.5:
        no_signal += 1

print(f"\nFailure categories:")
print(f"  Quantum-dominated (qp>=0.9): {quantum_dominated}/{len(failures)}")
print(f"  Close race (<1.5x gap):      {close_race}/{len(failures)}")
print(f"  No signal (max<0.5):         {no_signal}/{len(failures)}")

# Show top 10 failures by score gap (most confident wrong answers)
failures.sort(key=lambda f: -max(f['scores'].values(), default=0))
print(f"\n=== TOP 10 MOST CONFIDENT WRONG ANSWERS ===")
for f in failures[:10]:
    print(f"\n  Q: {f['q']}")
    print(f"  Got: {f['pred'][:40]}")
    print(f"  Exp: {f['exp'][:40]}")
    print(f"  Concepts: {f['concepts'][:5]}")
    # Show all scores
    for label, score in sorted(f['scores'].items(), key=lambda x: -x[1]):
        print(f"    {label}: {score:.3f}")

# Show distribution of score gaps for correct vs incorrect
correct_gaps = [g for g, ok in score_gaps if ok]
wrong_gaps = [g for g, ok in score_gaps if not ok]
print(f"\n=== SCORE GAP ANALYSIS ===")
print(f"Correct avg gap:  {sum(correct_gaps)/max(len(correct_gaps),1):.3f} (n={len(correct_gaps)})")
print(f"Wrong avg gap:    {sum(wrong_gaps)/max(len(wrong_gaps),1):.3f} (n={len(wrong_gaps)})")
