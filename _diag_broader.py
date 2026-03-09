#!/usr/bin/env python3
"""Test ARC-Easy on different offset ranges to verify generalization."""
import os, json, re, sys
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

# Test multiple batches
offsets = [50, 100, 150]  # Skip 0-49 (already verified 50/50)
if len(sys.argv) > 1:
    offsets = [int(x) for x in sys.argv[1:]]

for offset in offsets:
    print(f"\n{'='*60}")
    print(f"ARC-Easy offset={offset}, n=50")
    print(f"{'='*60}")
    batch = fetch("ARC-Easy", 50, offset)
    if not batch:
        print(f"  FETCH FAILED for offset={offset}")
        continue
    rows = batch.get('rows', [])

    correct = 0
    total = 0
    failures = []

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
        total += 1

        if pred_idx == expected_idx:
            correct += 1
        else:
            all_scores = result.get('all_scores', {})
            failures.append({
                'q': q[:80],
                'pred': texts[pred_idx][:40] if pred_idx < len(texts) else '?',
                'exp': texts[expected_idx][:40],
                'scores': all_scores,
                'concepts': result.get('concepts_found', [])[:5],
            })

    print(f"Correct: {correct}/{total} ({100*correct/max(total,1):.1f}%)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures[:10]:
            print(f"  Q: {f['q']}")
            print(f"  Got: {f['pred']} | Exp: {f['exp']}")
            scores = f['scores']
            for label, score in sorted(scores.items(), key=lambda x: -x[1]):
                print(f"    {label}: {score:.3f}")
            print()

print(f"\n{'='*60}")
print("DONE")
