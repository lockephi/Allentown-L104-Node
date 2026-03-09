#!/usr/bin/env python3
"""Trace the speed question through the scoring pipeline."""
import re, sys, os, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import logging; logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"

# Monkey-patch to add debug tracing
import l104_asi.commonsense_reasoning as cr

# Store original solve method
_orig_solve = cr.CommonsenseMCQSolver.solve

def _traced_solve(self, question, choices, subject=None):
    q_lower = question.lower()

    # Only trace the speed question
    if 'average speed' not in q_lower:
        return _orig_solve(self, question, choices, subject)

    print("\n[TRACE] Speed question detected! Tracing score evolution...")
    result = _orig_solve(self, question, choices, subject)

    # Check the all_scores
    all_scores = result.get('all_scores', {})
    print(f"[TRACE] Final all_scores: {json.dumps(all_scores, indent=2)}")
    return result

cr.CommonsenseMCQSolver.solve = _traced_solve

# Now also monkey-patch the _SCIENCE_RULES loop to trace
# We need a different approach - add prints around the rule checks
# Let's read the source to find where we can patch

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine

# Actually, let's test the rule matching directly
q = 'in one day, a family in a car rode for 2 hours, stopped for 3 hours, and then rode for another 5 hours. during the day, the family traveled a total distance of 400 kilometers. what was their average speed for the whole trip?'
choices = ['10 km/h', '20 km/h', '40 km/h', '50 km/h']

# Test each rule against this question
speed_pats = [
    (r'(?:average\s+speed.+(?:trip|whole|entire|day)|(?:rode|drove|travel).+stop.+(?:rode|drove|travel).+(?:average|speed))',
     r'\b40\b',
     r'\b10\b|\b20\b|\b50\b|\b80\b|\b100\b', 3.0, 0.1),
]

for q_pat, correct_pat, wrong_pat, boost, penalty in speed_pats:
    m = re.search(q_pat, q, re.IGNORECASE)
    print(f"\nQ_PAT match: {bool(m)}")
    if m:
        for c in choices:
            c_lower = c.lower()
            cm = re.search(correct_pat, c_lower, re.IGNORECASE)
            wm = re.search(wrong_pat, c_lower, re.IGNORECASE) if not cm else None
            if cm:
                print(f"  '{c}' -> CORRECT boost ×{1+boost:.1f}")
            elif wm:
                print(f"  '{c}' -> WRONG penalty ×{penalty}")
            else:
                print(f"  '{c}' -> no match")

# Also check if the \b40\b might match "400" in the question
m400 = re.search(r'\b40\b', q)
print(f"\nDoes \\b40\\b match in question text: {bool(m400)}")
if m400:
    print(f"  Matched at pos {m400.start()}: ...{q[max(0,m400.start()-10):m400.end()+10]}...")

# Also check \b400\b
m400b = re.search(r'\b400\b', q)
print(f"Does \\b400\\b match in question text: {bool(m400b)}")

# Now run the engine
print("\n\n=== Running full engine ===")
engine = CommonsenseReasoningEngine()
result = engine.answer_mcq(q, choices)
all_scores = result.get('all_scores', {})
print(f"Answer: {result.get('answer')}")
print(f"Scores: {json.dumps(all_scores)}")
print(f"Concepts: {result.get('concepts_found', [])[:8]}")
