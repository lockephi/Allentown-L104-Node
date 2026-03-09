#!/usr/bin/env python3
"""Run broader ARC diagnostic: Easy q0-99 and Challenge q0-49."""
import os, json, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['L104_QUIET'] = '1'
import logging; logging.disable(logging.WARNING)
import urllib.request

def fetch(config, n, offset):
    url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config={config}&split=test&offset={offset}&length={n}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "L104/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < 4:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"FETCH FAILED: {e}")
                return None

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

# Run on: Easy q0-49, Easy q50-99, Challenge q0-49
configs = [
    ("ARC-Easy", 50, 0, "Easy q0-49"),
    ("ARC-Easy", 50, 50, "Easy q50-99"),
    ("ARC-Challenge", 50, 0, "Challenge q0-49"),
]

totals = {"correct": 0, "total": 0}
results_by_config = {}

for config, n, offset, label in configs:
    batch = fetch(config, n, offset)
    if not batch:
        print(f"Skipping {label}: fetch failed")
        continue
    rows = batch.get('rows', [])
    correct = 0
    total = 0
    spread = [0, 0, 0, 0, 0]  # counts per choice index

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
        total += 1
        if pred == expected_idx:
            correct += 1
        if 0 <= pred < len(spread):
            spread[pred] += 1

    pct = (correct / total * 100) if total > 0 else 0
    print(f"{label}: {correct}/{total} = {pct:.1f}% | Spread: {spread[:len(texts)]}")
    results_by_config[label] = {"correct": correct, "total": total, "pct": pct}
    totals["correct"] += correct
    totals["total"] += total

print(f"\nCOMBINED: {totals['correct']}/{totals['total']} = {totals['correct']/totals['total']*100:.1f}%")
