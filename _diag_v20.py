#!/usr/bin/env python3
"""Diagnostic: analyze ARC failures to find algorithmic improvements."""
import sys, os, json, re
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
        except Exception as e:
            if attempt == 2:
                raise
            import time; time.sleep(2)

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

# Test multiple batches
results = []
for config, label in [("ARC-Easy", "Easy"), ("ARC-Challenge", "Challenge")]:
    for offset in [0, 100, 200]:
        batch = fetch(config, 50, offset)
        rows = batch.get('rows', [])
        correct = 0
        total = 0
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
                correct += 1
            total += 1
        pct = correct / total * 100 if total > 0 else 0
        results.append((label, offset, correct, total, pct))
        print(f"{label} q{offset}-{offset+49}: {correct}/{total} = {pct:.1f}%")

print("\n=== SUMMARY ===")
total_c = sum(r[2] for r in results)
total_t = sum(r[3] for r in results)
print(f"Overall: {total_c}/{total_t} = {total_c/total_t*100:.1f}%")
easy_c = sum(r[2] for r in results if r[0] == "Easy")
easy_t = sum(r[3] for r in results if r[0] == "Easy")
chall_c = sum(r[2] for r in results if r[0] == "Challenge")
chall_t = sum(r[3] for r in results if r[0] == "Challenge")
print(f"Easy: {easy_c}/{easy_t} = {easy_c/easy_t*100:.1f}%")
print(f"Challenge: {chall_c}/{chall_t} = {chall_c/chall_t*100:.1f}%")
