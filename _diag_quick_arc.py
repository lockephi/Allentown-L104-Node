#!/usr/bin/env python3
"""Quick ARC sample test - 50 Easy + 50 Challenge."""
import sys, os, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"
import urllib.request

def fetch(config, n=50, offset=0):
    url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config={config}&split=test&offset={offset}&length={n}"
    req = urllib.request.Request(url, headers={"User-Agent": "L104/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

for label, config, off in [("Easy q0-49", "ARC-Easy", 0), ("Easy q100-149", "ARC-Easy", 100), ("Challenge q0-49", "ARC-Challenge", 0)]:
    batch = fetch(config, 50, off)
    rows = batch.get('rows', [])
    correct = 0
    total = 0
    preds = [0,0,0,0]
    for row_data in rows:
        r = row_data.get('row', {})
        q = r.get('question', '')
        cd = r.get('choices', {})
        labels = cd.get('label', [])
        texts = cd.get('text', [])
        ak = r.get('answerKey', '')
        if not q or not texts: continue
        total += 1
        result = engine.answer_mcq(q, texts)
        pred_idx = result.get('answer_index', 0)
        preds[pred_idx] += 1
        exp_idx = labels.index(ak) if ak in labels else -1
        if pred_idx == exp_idx: correct += 1
    print(f"{label}: {correct}/{total} = {correct/total*100:.1f}%  dist={preds}")
