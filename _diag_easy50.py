#!/usr/bin/env python3
"""Analyze failures in Easy q50-99 to find improvement patterns."""
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

batch = fetch("ARC-Easy", 50, 50)
rows = batch.get('rows', [])

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
    pred = result.get('selected_index', -1)
    if pred != expected_idx:
        failures.append({
            'q': q, 'choices': texts,
            'got': pred, 'exp': expected_idx,
            'got_text': texts[pred][:50] if 0 <= pred < len(texts) else '?',
            'exp_text': texts[expected_idx][:50]
        })

print(f"Failures: {len(failures)}/50\n")
# Categorize failures
for i, f in enumerate(failures):
    print(f"F{i}: {f['q'][:80]}")
    print(f"  Got: [{f['got']}] {f['got_text']}")
    print(f"  Exp: [{f['exp']}] {f['exp_text']}")
    print()
