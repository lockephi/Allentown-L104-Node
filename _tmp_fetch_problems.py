#!/usr/bin/env python3
"""Fetch the actual prompts and tests for failing problems."""
import sys, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging; logging.disable(logging.WARNING)
sys.path.insert(0, '.')
import importlib
bh = importlib.import_module('l104_asi.benchmark_harness')
problems = bh._HuggingFaceFetcher.fetch_humaneval()

targets = {32, 59, 88, 142}
for prob in problems:
    tid = prob['task_id']
    num = int(tid.split('/')[-1])
    if num not in targets:
        continue
    print(f"\n{'='*80}")
    print(f"TASK: {tid} | ENTRY: {prob['entry_point']}")
    print(f"{'='*80}")
    print("PROMPT:")
    print(prob['prompt'])
    print("\nTEST:")
    print(prob['test'][:800])
    print()
