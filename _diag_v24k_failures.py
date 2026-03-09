#!/usr/bin/env python3
"""Collect ALL failures from ARC-Easy offsets 200-449 for v24k analysis."""
import os, json, sys
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

all_failures = []
for offset in [200, 250, 300, 350, 400]:
    batch = fetch("ARC-Easy", 50, offset)
    if not batch:
        continue
    rows = batch.get("rows", [])
    correct = 0
    total = 0
    for row_data in rows:
        r = row_data.get("row", {})
        q = r.get("question", "")
        choices_data = r.get("choices", {})
        texts = choices_data.get("text", [])
        answer_key = r.get("answerKey", "")
        if not q or not texts:
            continue
        labels = choices_data.get("label", [])
        expected_idx = labels.index(answer_key) if answer_key in labels else -1
        if expected_idx < 0:
            continue
        result = engine.answer_mcq(q, texts)
        pred_idx = result.get("selected_index", 0)
        total += 1
        if pred_idx == expected_idx:
            correct += 1
        else:
            all_failures.append({
                "off": offset,
                "q": q[:150],
                "got": texts[pred_idx][:80] if pred_idx < len(texts) else "?",
                "exp": texts[expected_idx][:80],
            })
    print(f"offset={offset}: {correct}/{total}", flush=True)

print(f"\n=== ALL {len(all_failures)} FAILURES ===")
for i, f in enumerate(all_failures):
    print(f"{i+1}. [OFF={f['off']}] Q: {f['q']}")
    print(f"   GOT: {f['got']}")
    print(f"   EXP: {f['exp']}")
    print()
