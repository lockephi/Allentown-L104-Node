#!/usr/bin/env python3
"""Diagnose the 7 remaining v25b failures."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset

ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

# Known failure indices (approximate from offsets)
# Offset 300: 2 failures, offset 350: 2 failures, offset 400: 3 failures
# Let me find the exact ones by searching for key phrases

targets = [
    "moth in England",
    "Gulf of Mexico",
    "Sodium bicarbonate",
    "nitrogen-based fertilizers",
    "rode for 2 hours",
    "skunk has been close",
    "word processors on computers",
]

for i in range(500):
    q = ds[i]["question"]
    for t in targets:
        if t.lower() in q.lower():
            labels = ds[i]["choices"]["label"]
            texts = ds[i]["choices"]["text"]
            correct = ds[i]["answerKey"]
            print(f"\n{'='*70}")
            print(f"[{i}] CORRECT={correct}")
            print(f"Q: {q}")
            for l, t2 in zip(labels, texts):
                marker = "<<<" if l == correct else ""
                print(f"  {l}. {t2} {marker}")
            break
