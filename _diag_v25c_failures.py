#!/usr/bin/env python3
"""Diagnose v25c failures — get full question text + choices from HuggingFace."""
import json, sys, urllib.request

def fetch(config, n=50, offset=0):
    url = f"https://datasets-server.huggingface.co/rows?dataset=allenai%2Fai2_arc&config={config}&split=test&offset={offset}&length={n}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "L104/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except:
            import time; time.sleep(2)

# Fetch the batches containing failures: offset 100, 350, 400
failure_keywords = [
    (100, "nitrogen fertilizers", "fish populations decrease"),
    (350, "Sodium bicarbonate", "H_{2}O"),
    (400, "rode for 2 hours", "40 km/h"),
    (400, "transfer of energy to survive", "Plants -> Fish -> Birds"),
    (400, "phototropism", "reuse the markers"),
    (400, "Work is a product of force", "riding a bike"),
]

cached_batches = {}
for offset, kw, exp_text in failure_keywords:
    if offset not in cached_batches:
        data = fetch("ARC-Easy", 50, offset)
        cached_batches[offset] = data['rows'] if data else []

    for i, row in enumerate(cached_batches[offset]):
        q = row['row']
        qtext = q.get('question', '')
        if kw in qtext:
            choices = q.get('choices', {})
            labels = choices.get('label', [])
            texts = choices.get('text', [])
            answer_key = q.get('answerKey', '?')
            print(f"\n{'='*80}")
            print(f"OFFSET: {offset}+{i} = {offset+i}  |  ANSWER KEY: {answer_key}")
            print(f"Q: {qtext}")
            for l, t in zip(labels, texts):
                marker = " <<<" if l == answer_key else ""
                print(f"  {l}: {t}{marker}")
            break
