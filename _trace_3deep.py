#!/usr/bin/env python3
"""Deep trace of Q25/Q28/Q29 — context_facts + Stage 3b firing."""
import sys, os, re, logging
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
logging.disable(logging.CRITICAL)
sys.stderr = open(os.devnull, 'w')
sys.path.insert(0, '.')

from l104_asi.language_comprehension import LanguageComprehensionEngine
from l104_asi.language_comprehension.mcq_solver import MCQSolver
sys.stderr = sys.__stderr__

# Deep patch _score_choice to log Stage 3b matching
_orig_score = MCQSolver._score_choice
_debug_data = {}

def _deep_trace_score(self, question, choice, context_facts, knowledge_hits, has_context=False):
    # Log context_facts for the first choice only (same for all)
    key = question[:40]
    if key not in _debug_data:
        _debug_data[key] = {
            'context_facts': context_facts[:15] if context_facts else [],
            'knowledge_hits': knowledge_hits,
            'has_context': has_context,
            'choices': {}
        }

    # Reproduce Stage 3b logic to trace it
    choice_lower = choice.lower().strip()
    stage_3b_info = {'fired': False, 'bonus': 0, 'matching_facts': []}
    if context_facts:
        q_words_3b = {w for w in re.findall(r'\w+', question.lower()) if len(w) > 2}
        q_content_3b = q_words_3b - {'what', 'which', 'who', 'whom', 'the', 'is',
            'are', 'was', 'were', 'does', 'did', 'has', 'have', 'had', 'this',
            'that', 'these', 'those', 'and', 'or', 'but', 'for', 'with', 'from',
            'about', 'into', 'how', 'why', 'when', 'where', 'not', 'its'}
        stage_3b_info['q_content'] = q_content_3b
        for fi, fact in enumerate(context_facts[:15]):
            fl = fact.lower()
            fact_words_3b = set(re.findall(r'\w+', fl))
            topic_overlap = len(q_content_3b & fact_words_3b)
            overlapping = q_content_3b & fact_words_3b
            ch_match = re.search(r'\b' + re.escape(choice_lower) + r'\b', fl) if len(choice_lower) >= 2 else None
            if topic_overlap >= 3 and ch_match:
                stage_3b_info['fired'] = True
                stage_3b_info['matching_facts'].append({
                    'fact_idx': fi,
                    'topic_overlap': topic_overlap,
                    'overlapping_words': overlapping,
                    'choice_found': True,
                    'fact_preview': fl[:120]
                })
            elif topic_overlap >= 2:
                stage_3b_info['matching_facts'].append({
                    'fact_idx': fi,
                    'topic_overlap': topic_overlap,
                    'overlapping_words': overlapping,
                    'choice_found': bool(ch_match),
                    'fact_preview': fl[:120]
                })

    _debug_data[key]['choices'][choice] = stage_3b_info

    score = _orig_score(self, question, choice, context_facts, knowledge_hits, has_context=has_context)
    _debug_data[key]['choices'][choice]['final_score'] = score
    return score

MCQSolver._score_choice = _deep_trace_score
lce = LanguageComprehensionEngine()

questions = [
    {"q": "The term 'cogito ergo sum' is attributed to", "c": ["Kant", "Descartes", "Hume", "Locke"], "a": 1, "s": "philosophy"},
    {"q": "Maslow's hierarchy of needs places which need at the base?", "c": ["Safety", "Physiological", "Love", "Esteem"], "a": 1, "s": "psychology"},
    {"q": "The scientific method begins with", "c": ["hypothesis", "observation", "experiment", "conclusion"], "a": 1, "s": "science"},
]

for qi, qdata in enumerate(questions):
    q = qdata["q"]
    choices = qdata["c"]
    correct_idx = qdata["a"]
    _debug_data.clear()
    result = lce.answer_mcq(q, choices, subject=qdata["s"])
    predicted = result.get("selected_index", -1)
    ok = "✓" if predicted == correct_idx else "✗"

    print(f"\n{'='*80}")
    print(f"{ok} Q: {q}")
    print(f"  Expected: [{correct_idx}] {choices[correct_idx]}")
    print(f"  Got:      [{predicted}] {choices[predicted] if 0<=predicted<len(choices) else '?'}")

    key = q[:40]
    data = _debug_data.get(key, {})

    # Show context_facts
    cf = data.get('context_facts', [])
    print(f"\n  context_facts ({len(cf)} facts):")
    for fi, f in enumerate(cf[:10]):
        print(f"    [{fi}] {f[:120]}")

    print(f"  knowledge_hits: {data.get('knowledge_hits', 0)}")
    print(f"  has_context: {data.get('has_context', False)}")

    # Show Stage 3b per choice
    print(f"\n  Stage 3b per choice:")
    for ch_name, ch_data in data.get('choices', {}).items():
        s3b = ch_data
        mark = " ◀ CORRECT" if ch_name == choices[correct_idx] else ""
        print(f"    {ch_name}{mark}: score={s3b.get('final_score',0):.4f}, "
              f"3b_fired={s3b.get('fired', False)}")
        if s3b.get('q_content'):
            print(f"      q_content_3b: {s3b['q_content']}")
        for mf in s3b.get('matching_facts', []):
            print(f"      fact[{mf['fact_idx']}]: overlap={mf['topic_overlap']}, "
                  f"choice_found={mf['choice_found']}, words={mf['overlapping_words']}")
            print(f"        → {mf['fact_preview']}")

print("\nDone.")
