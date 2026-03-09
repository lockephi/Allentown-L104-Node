#!/usr/bin/env python3
"""Test exact offset 50 regression questions from HuggingFace."""
import os, json, sys, urllib.request
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
os.environ['L104_QUIET'] = '1'
sys.path.insert(0, '.')

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
engine = CommonsenseReasoningEngine()

url = 'https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config=ARC-Easy&split=test&offset=50&length=50'
req = urllib.request.Request(url, headers={'User-Agent': 'L104/1.0'})
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read())

rows = data['rows']
for i, row_data in enumerate(rows):
    r = row_data['row']
    q = r.get('question', '')
    if 'hybrid car' in q or 'Bunsen burner' in q:
        choices_data = r.get('choices', {})
        texts = choices_data.get('text', [])
        labels = choices_data.get('label', [])
        answer_key = r.get('answerKey', '')
        expected_idx = labels.index(answer_key) if answer_key in labels else -1

        result = engine.answer_mcq(q, texts)
        pred_idx = result.get('selected_index', 0)
        all_scores = result.get('all_scores', {})

        print(f'\nIdx: {50+i}')
        print(f'Q: {q[:100]}')
        print(f'Choices: {texts}')
        print(f'Expected: {answer_key} = {texts[expected_idx]}')
        print(f'Got: {result.get("answer")} = {texts[pred_idx]}')
        for lbl in sorted(all_scores, key=lambda x: -all_scores[x]):
            print(f'  {lbl}: {all_scores[lbl]:.4f}')
        print(f'PASS: {pred_idx == expected_idx}')
