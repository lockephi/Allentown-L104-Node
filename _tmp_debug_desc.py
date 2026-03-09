#!/usr/bin/env python3
"""Check what DocstringParser returns for description."""
import sys, os, signal
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging; logging.disable(logging.WARNING)
sys.path.insert(0, '.')
import importlib
bh = importlib.import_module('l104_asi.benchmark_harness')
problems = bh._HuggingFaceFetcher.fetch_humaneval()
runner = bh._HumanEvalRunner()
engine = runner._get_engine()

# Access the synthesizer
synth = engine._synth

targets = {88, 116, 133, 142}
for prob in problems:
    tid = prob['task_id']
    num = int(tid.split('/')[-1])
    if num not in targets:
        continue
    fn = prob['entry_point']
    prompt = prob['prompt']

    # Parse like generate_from_docstring does
    spec = synth.parser.parse(prompt)
    print(f"\n{'='*60}")
    print(f"{tid} ({fn})")
    print(f"spec.name = {spec.name!r}")
    print(f"spec.description[:200] = {spec.description[:200]!r}")
    print(f"'binary' in desc: {'binary' in spec.description.lower()}")
    print(f"'ascending' in desc: {'ascending' in spec.description.lower()}")
    print(f"'ceil' in desc: {'ceil' in spec.description.lower()}")
    print(f"'index' in desc: {'index' in spec.description.lower()}")
    print(f"'entries' in desc: {'entries' in spec.description.lower()}")
    print(f"'multiple' in desc: {'multiple' in spec.description.lower()}")
    print(f"nparams = {len(spec.parameters)}")
    print(f"params = {[p['name'] for p in spec.parameters]}")
