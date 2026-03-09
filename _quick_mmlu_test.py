"""Quick MMLU A/B test: with and without cross-verification + other stages."""
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import logging
logging.disable(logging.WARNING)
sys.path.insert(0, '.')

from l104_asi.language_comprehension import LanguageComprehensionEngine
engine = LanguageComprehensionEngine()
engine.initialize()

from l104_asi.benchmark_harness import _HuggingFaceFetcher
samples = _HuggingFaceFetcher.fetch_mmlu(max_questions=100)

# Run with detailed per-question tracking
correct = 0
total = 0
wrong_questions = []
for s in samples:
    result = engine.answer_mcq(s['question'], s['choices'], subject=s.get('subject'))
    sel = result.get('selected_index', result.get('answer_index', -1))
    is_correct = (sel == s['answer'])
    if is_correct:
        correct += 1
    else:
        wrong_questions.append({
            'q': s['question'][:80],
            'subject': s.get('subject', '?'),
            'expected': s['answer'],
            'got': sel,
            'confidence': result.get('confidence', 0),
        })
    total += 1

print(f'MMLU 100: {correct}/{total} = {100*correct/total:.1f}%')
print(f'\nTop wrong subjects:')
from collections import Counter
subj_counts = Counter(w['subject'] for w in wrong_questions)
for subj, count in subj_counts.most_common(10):
    print(f'  {subj}: {count} wrong')
print(f'\nSample wrong Qs:')
for w in wrong_questions[:5]:
    print(f'  [{w["subject"]}] exp={w["expected"]} got={w["got"]} | {w["q"]}')
