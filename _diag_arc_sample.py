#!/usr/bin/env python3
"""Sample random ARC Easy failures and show full details with scores."""
import sys, os, json, random
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"

# Load benchmark results to find failing questions
data = json.load(open('benchmark_full_online_results.json'))
arc = data['detailed_results']['ARC']['details']

# Get the benchmark script to see how it fetches data
import urllib.request

def fetch_arc_easy(n=30, offset=0):
    """Fetch ARC Easy questions from HuggingFace."""
    url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config=ARC-Easy&split=test&offset={offset}&length={n}"
    req = urllib.request.Request(url, headers={"User-Agent": "L104/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def fetch_arc_challenge(n=30, offset=0):
    url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config=ARC-Challenge&split=test&offset={offset}&length={n}"
    req = urllib.request.Request(url, headers={"User-Agent": "L104/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

# Fetch a batch of easy questions and test them
print("Fetching ARC Easy questions...")
batch = fetch_arc_easy(50, offset=100)
rows = batch.get('rows', [])

correct = 0
total = 0
failures = []

for row_data in rows:
    r = row_data.get('row', {})
    q = r.get('question', '')
    choices_data = r.get('choices', {})
    labels = choices_data.get('label', [])
    texts = choices_data.get('text', [])
    answer_key = r.get('answerKey', '')

    if not q or not texts:
        continue

    # Find expected index
    expected = -1
    for i, lbl in enumerate(labels):
        if lbl == answer_key:
            expected = i
            break
    if expected < 0:
        continue

    result = engine.answer_mcq(q, texts)
    predicted = result.get('selected_index', -1)
    total += 1

    if predicted == expected:
        correct += 1
    else:
        failures.append({
            'q': q[:120],
            'choices': texts,
            'expected': expected,
            'predicted': predicted,
            'exp_choice': texts[expected] if expected < len(texts) else '?',
            'got_choice': texts[predicted] if 0 <= predicted < len(texts) else '?',
        })

print(f"\nResults: {correct}/{total} = {100*correct/total:.1f}%")
print(f"\nFailures ({len(failures)}):")
for f in failures[:25]:
    print(f"\n  Q: {f['q']}")
    for i, c in enumerate(f['choices']):
        marker = ' ✓' if i == f['expected'] else (' ✗' if i == f['predicted'] else '  ')
        print(f"    {i}{marker}: {c[:80]}")
