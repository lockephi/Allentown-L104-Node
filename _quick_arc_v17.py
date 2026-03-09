#!/usr/bin/env python3
"""Quick ARC test for v17 changes."""
import json, urllib.request, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def fetch(split, offset, length):
    url = f'https://datasets-server.huggingface.co/rows?dataset=allenai/ai2_arc&config=ARC-{split}&split=test&offset={offset}&length={length}'
    req = urllib.request.Request(url, headers={'User-Agent': 'L104/1.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())['rows']

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
eng = CommonsenseReasoningEngine()

for split in ['Easy', 'Challenge']:
    for offset in [0, 100]:
        rows = fetch(split, offset, 50)
        correct = 0
        preds = [0,0,0,0]
        for row in rows:
            r = row['row']
            q = r['question']
            labels = r['choices']['label']
            texts = r['choices']['text']
            ans_key = r['answerKey']
            choices = [texts[i] for i in range(len(labels))]
            result = eng.answer_mcq(q, choices)
            pred_idx = result.get('selected_index', result.get('answer_index', 0))
            pred_label = labels[pred_idx] if pred_idx < len(labels) else 'A'
            if pred_label == ans_key:
                correct += 1
            if pred_idx < 4:
                preds[pred_idx] += 1
        print(f'{split} q{offset}-{offset+49}: {correct}/50 = {correct*2}%, dist={preds}')
