#!/usr/bin/env python3
"""Analyze ARC Challenge failures for improvement patterns."""
import os, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['L104_QUIET'] = '1'
import logging; logging.disable(logging.WARNING)
import urllib.request

def fetch(config, n, offset):
    url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config={config}&split=test&offset={offset}&length={n}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "L104/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except:
            import time; time.sleep(3)

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

batch1 = fetch("ARC-Challenge", 50, 0)
batch2 = fetch("ARC-Challenge", 50, 50)

failures = []
correct = 0
total = 0
choice_spread = [0, 0, 0, 0, 0]

for batch in [batch1, batch2]:
    if not batch:
        continue
    for row_data in batch.get('rows', []):
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
        pred = result.get('selected_index', -1)
        total += 1
        if pred == expected_idx:
            correct += 1
        else:
            failures.append({
                'q': q[:80], 'exp': expected_idx, 'got': pred,
                'exp_text': texts[expected_idx][:50],
                'got_text': texts[pred][:50] if 0 <= pred < len(texts) else '?'
            })
        if 0 <= pred < len(choice_spread):
            choice_spread[pred] += 1

print(f"Challenge q0-99: {correct}/{total} = {correct/total*100:.1f}%")
print(f"Spread: {choice_spread[:4]}")
print(f"\nFailures: {len(failures)}")
print(f"\nFirst 25 failures:")
for i, f in enumerate(failures[:25]):
    print(f"\n  F{i}: {f['q']}")
    print(f"    Got: [{f['got']}] {f['got_text']}")
    print(f"    Exp: [{f['exp']}] {f['exp_text']}")
