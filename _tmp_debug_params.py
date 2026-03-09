#!/usr/bin/env python3
"""Debug parameter extraction for failing HumanEval problems."""
import sys, os, re
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import logging; logging.disable(logging.WARNING)
sys.path.insert(0, '.')

from l104_asi.code_generation import CodeSynthesizer, DocstringParser
import importlib
bh = importlib.import_module('l104_asi.benchmark_harness')

parser = DocstringParser()
problems = bh._HuggingFaceFetcher.fetch_humaneval()

targets = {'make_palindrome', 'find_zero', 'decode_cyclic', 'largest_prime_factor',
           'sort_array', 'sum_squares', 'specialFilter', 'Strongest_Extension',
           'eat', 'solve', 'minSubArraySum', 'minPath', 'fix_spaces'}

for prob in problems:
    fn = prob['entry_point']
    if fn not in targets:
        continue
    task = prob['task_id']
    prompt = prob['prompt']
    
    doc_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    actual_docstring = doc_match.group(1).strip() if doc_match else prompt
    
    sig_match = re.search(r'(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[\w\[\], .]+)?\s*:)', prompt)
    actual_signature = sig_match.group(1) if sig_match else ''
    
    spec = parser.parse(actual_docstring, fn)
    synth = CodeSynthesizer()
    synth._enrich_spec_from_signature(spec, actual_signature)
    
    nparams = len(spec.parameters)
    pnames = [p["name"] for p in spec.parameters]
    ptypes = [p.get("type", "?") for p in spec.parameters]
    
    # Check if direct lookup would match
    result = synth._direct_solution_lookup(spec)
    hit = "HIT" if result else "MISS"
    
    # Debug: check what fname and key are
    fname_lower = (spec.name or "").lower().strip()
    key = (fname_lower, nparams)
    
    print(f"{task} ({fn}): nparams={nparams}, params={pnames}, types={ptypes}, direct={hit}, key={key}, spec.name={spec.name!r}")
