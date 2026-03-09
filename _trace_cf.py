#!/usr/bin/env python3
"""Trace context_facts retrieval for Q25, Q28, Q29."""
import sys, os, re, logging
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
logging.disable(logging.CRITICAL)
sys.stderr = open(os.devnull, 'w')
sys.path.insert(0, '.')

from l104_asi.language_comprehension.mcq_solver import MCQSolver
from l104_asi.language_comprehension import LanguageComprehensionEngine

_orig_solve = MCQSolver.solve
_captured = {}

def _patched_solve(self, question, choices, subject=None):
    _my_context = []
    _orig_sc = self._score_choice
    def _capture_sc(q, choice, cf, kh, has_context=False):
        if not _my_context:
            _my_context.extend(cf[:30])
        return _orig_sc(q, choice, cf, kh, has_context=has_context)
    self._score_choice = _capture_sc
    result = _orig_solve(self, question, choices, subject=subject)
    _captured['context_facts'] = list(_my_context)
    self._score_choice = _orig_sc
    return result

MCQSolver.solve = _patched_solve
sys.stderr = sys.__stderr__

lce = LanguageComprehensionEngine()

# Q25
q1 = "The term 'cogito ergo sum' is attributed to"
c1 = ["Kant", "Descartes", "Hume", "Locke"]
_captured.clear()
r1 = lce.answer_mcq(q1, c1, subject="philosophy")
print("=== Q25: cogito ergo sum ===")
print(f"Selected: [{r1.get('selected_index')}] {c1[r1.get('selected_index', 0)]}")
cf1 = _captured.get('context_facts', [])
print(f"Context facts count: {len(cf1)}")
for i, f in enumerate(cf1[:8]):
    print(f"  [{i}] {f[:130]}")
print("cogito/descartes matches:")
for i, f in enumerate(cf1):
    fl = f.lower()
    if 'cogito' in fl or 'descartes' in fl:
        print(f"  MATCH [{i}]: {f[:150]}")
if not any('cogito' in f.lower() or 'descartes' in f.lower() for f in cf1):
    print("  NONE FOUND - KB fact not in context_facts!")

# Q28
q2 = "Maslow's hierarchy of needs places which need at the base?"
c2 = ["Safety", "Physiological", "Love", "Esteem"]
_captured.clear()
r2 = lce.answer_mcq(q2, c2, subject="psychology")
print("\n=== Q28: Maslow hierarchy ===")
print(f"Selected: [{r2.get('selected_index')}] {c2[r2.get('selected_index', 0)]}")
cf2 = _captured.get('context_facts', [])
print(f"Context facts count: {len(cf2)}")
for i, f in enumerate(cf2[:8]):
    print(f"  [{i}] {f[:130]}")
print("maslow/physiological/hierarchy/base matches:")
for i, f in enumerate(cf2):
    fl = f.lower()
    if 'maslow' in fl or 'physiological' in fl or 'hierarchy' in fl:
        print(f"  MATCH [{i}]: {f[:150]}")

# Q29
q3 = "The scientific method begins with"
c3 = ["hypothesis", "observation", "experiment", "conclusion"]
_captured.clear()
r3 = lce.answer_mcq(q3, c3, subject="science")
print("\n=== Q29: scientific method ===")
print(f"Selected: [{r3.get('selected_index')}] {c3[r3.get('selected_index', 0)]}")
cf3 = _captured.get('context_facts', [])
print(f"Context facts count: {len(cf3)}")
for i, f in enumerate(cf3[:8]):
    print(f"  [{i}] {f[:130]}")
print("begins/observation/scientific method matches:")
for i, f in enumerate(cf3):
    fl = f.lower()
    if 'begins with' in fl or ('scientific method' in fl and 'observation' in fl):
        print(f"  MATCH [{i}]: {f[:150]}")
if not any('begins with observation' in f.lower() for f in cf3):
    print("  'begins with observation' NOT FOUND in context_facts!")

# Check KB nodes directly
print("\n=== KB node check ===")
mcq = lce._mcq_solver
for key, node in mcq.kb.nodes.items():
    all_text = node.definition + " " + " ".join(node.facts)
    al = all_text.lower()
    if 'cogito' in al:
        print(f"cogito found in node: {key}")
    if 'maslow' in al and 'physiological' in al:
        print(f"maslow+physiological in node: {key}")
    if 'scientific method' in al and 'observation' in al:
        print(f"scientific method+observation in node: {key}")

print("\nDone.")
