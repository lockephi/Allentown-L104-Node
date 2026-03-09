#!/usr/bin/env python3
"""Diagnostic: Show full scoring breakdown for failing ARC questions."""
import sys, os, json, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"

import urllib.request

def fetch_arc_easy(n=30, offset=0):
    url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config=ARC-Easy&split=test&offset={offset}&length={n}"
    req = urllib.request.Request(url, headers={"User-Agent": "L104/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

# Fetch questions
batch = fetch_arc_easy(50, offset=100)
rows = batch.get('rows', [])

# Test specific failing questions and show all scores
for row_data in rows:
    r = row_data.get('row', {})
    q = r.get('question', '')
    choices_data = r.get('choices', {})
    texts = choices_data.get('text', [])
    answer_key = r.get('answerKey', '')
    labels = choices_data.get('label', [])

    if not q or not texts:
        continue

    expected_idx = labels.index(answer_key) if answer_key in labels else -1
    result = engine.answer_mcq(q, texts)
    predicted_idx = result.get('selected_index', result.get('answer_index', 0))

    if predicted_idx != expected_idx:
        print(f"\n{'='*70}")
        print(f"Q: {q[:80]}")
        print(f"Expected: [{expected_idx}] {texts[expected_idx]}")
        print(f"Got:      [{predicted_idx}] {texts[predicted_idx]}")
        print(f"All scores: {result.get('all_scores', {})}")
        print(f"Concepts: {result.get('concepts_found', [])[:5]}")
        print(f"Causal rules: {result.get('causal_rules_used', 0)}")
        print(f"Quantum: {result.get('quantum', {})}")
