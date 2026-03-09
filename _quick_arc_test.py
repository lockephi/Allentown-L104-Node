#!/usr/bin/env python3
"""Quick ARC Easy test — 50 questions offset 100."""
import os, json, logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"
import urllib.request

def fetch(n, offset):
    url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config=ARC-Easy&split=test&offset={offset}&length={n}"
    req = urllib.request.Request(url, headers={"User-Agent": "L104/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

batch = fetch(50, 100)  # Easy offset=100
rows = batch.get('rows', [])
correct = 0
total = 0
fails = []
for rd in rows:
    r = rd.get('row', {})
    q = r.get('question', '')
    cd = r.get('choices', {})
    labels = cd.get('label', [])
    texts = cd.get('text', [])
    ak = r.get('answerKey', '')
    if not q or not texts:
        continue
    total += 1
    exp = labels.index(ak) if ak in labels else -1
    result = engine.answer_mcq(q, texts)
    pred = result.get('selected_index', result.get('answer_index', 0))
    if pred == exp:
        correct += 1
    else:
        fails.append((q[:60], texts[exp][:30], texts[pred][:30], result.get('all_scores', {})))

print(f"\nResult: {correct}/{total} = {100*correct/total:.1f}%")
print(f"Failures: {len(fails)}")
for q, exp, got, scores in fails[:10]:
    print(f"  Q: {q}...")
    print(f"    Exp: {exp} | Got: {got}")
    print(f"    Scores: {scores}")
