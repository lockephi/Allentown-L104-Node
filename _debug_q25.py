#!/usr/bin/env python3
"""Debug Q25 (cogito ergo sum) — check _direct_answer_bonus, context_facts, etc."""
import sys, os, re, logging
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
logging.disable(logging.CRITICAL)
sys.stderr = open(os.devnull, 'w')
sys.path.insert(0, '.')

from l104_asi.language_comprehension import LanguageComprehensionEngine
from l104_asi.language_comprehension.mcq_solver import MCQSolver
sys.stderr = sys.__stderr__

# Monkey-patch solve() to capture internals
_orig_solve = MCQSolver.solve

def _patched_solve(self, question, choices, subject=None):
    # Run original but capture key internal state
    # We'll add prints inside the original flow via monkey-patching _score_choice
    return _orig_solve(self, question, choices, subject=subject)

# Instead, let's directly replicate the key parts of solve() to see what happens
lce = LanguageComprehensionEngine()
mcq = lce.mcq_solver

question = "The term 'cogito ergo sum' is attributed to"
choices = ["Kant", "Descartes", "Hume", "Locke"]
subject = "philosophy"

# Step 0: Subject search
subject_lower = subject.lower().replace(" ", "_")
_SUBJECT_ALIASES = {
    "philosophy": ["moral_disputes", "moral_scenarios", "logical_fallacies"],
}
subjects_to_search = [subject_lower]
subjects_to_search.extend(_SUBJECT_ALIASES.get(subject_lower, []))

subject_hits = []
for key, node in mcq.kb.nodes.items():
    node_subj = node.subject.lower().replace(" ", "_")
    for search_subj in subjects_to_search:
        if search_subj in key or search_subj in node_subj:
            rel = 0.5 if search_subj == subject_lower else 0.3
            subject_hits.append((key, node, rel))
            break

print(f"Subject hits for '{subject}': {[k for k,_,_ in subject_hits]}")

# Check if cogito fact is in philosophy_of_mind