#!/usr/bin/env python3
"""MMLU Diagnostic: trace scoring per-question to find root causes of low accuracy."""
import sys, os, re, json, requests, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress engine init noise
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))

from l104_asi.language_comprehension import LanguageComprehensionEngine

engine = LanguageComprehensionEngine()
engine.initialize()

# Fetch 10 questions from 5 diverse subjects
HF_URL = "https://datasets-server.huggingface.co/rows"
SUBJECTS = ["college_physics", "college_biology", "us_foreign_policy", "elementary_mathematics", "philosophy"]

questions = []
for subj in SUBJECTS:
    url = f"{HF_URL}?dataset=cais/mmlu&config={subj}&split=test&offset=0&length=2"
    try:
        r = requests.get(url, timeout=15)
        rows = r.json().get("rows", [])
        for row_data in rows:
            row = row_data.get("row", {})
            if "question" in row and "choices" in row:
                questions.append({
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],
                    "subject": row.get("subject", subj),
                })
    except Exception as e:
        print(f"FETCH ERROR {subj}: {e}")

print(f"Fetched {len(questions)} questions from {len(SUBJECTS)} subjects\n")

correct = 0
total = 0
categories = {"kb_hit_correct": 0, "kb_hit_wrong": 0, "kb_miss_correct": 0, "kb_miss_wrong": 0}

for qi, q in enumerate(questions):
    print(f"{'='*80}")
    print(f"Q{qi+1} [{q['subject']}]: {q['question'][:120]}")
    for ci, ch in enumerate(q["choices"]):
        marker = " *" if ci == q["answer"] else ""
        print(f"  {chr(65+ci)}: {ch[:80]}{marker}")

    result = engine.answer_mcq(q["question"], q["choices"], subject=q["subject"])
    predicted = result.get("selected_index", result.get("answer_index", -1))
    expected = q["answer"]
    is_correct = predicted == expected

    # Extract per-choice scores
    details = result.get("choice_details", result.get("choices", []))
    scores_str = ""
    if details:
        for d in details:
            idx = d.get("index", d.get("idx", "?"))
            sc = d.get("score", d.get("final_score", 0))
            label = d.get("label", chr(65+idx) if isinstance(idx, int) else "?")
            scores_str += f"  {label}={sc:.3f}"

    # Check KB signal
    kb_signal = result.get("kb_signal", False)
    n_facts = result.get("context_facts_count", result.get("n_facts", "?"))
    confidence = result.get("confidence", 0)

    status = "CORRECT" if is_correct else "WRONG"
    print(f"  -> Predicted: {chr(65+predicted) if predicted >= 0 else '?'} | Expected: {chr(65+expected)} | {status}")
    if scores_str:
        print(f"  -> Scores:{scores_str}")
    print(f"  -> Confidence: {confidence:.3f}, Facts: {n_facts}, KB_signal: {kb_signal}")

    # Reasoning chain
    reasoning = result.get("reasoning", result.get("chain_of_thought", []))
    if reasoning and isinstance(reasoning, list):
        for step in reasoning[:3]:
            print(f"  -> {step[:100]}")

    total += 1
    if is_correct:
        correct += 1

    # Categorize
    has_kb = (isinstance(n_facts, int) and n_facts >= 3) or kb_signal
    if has_kb and is_correct:
        categories["kb_hit_correct"] += 1
    elif has_kb and not is_correct:
        categories["kb_hit_wrong"] += 1
    elif not has_kb and is_correct:
        categories["kb_miss_correct"] += 1
    else:
        categories["kb_miss_wrong"] += 1
    print()

print(f"{'='*80}")
print(f"SUMMARY: {correct}/{total} = {correct/max(total,1)*100:.1f}%")
print(f"Categories: {json.dumps(categories, indent=2)}")
print(f"  KB hit + correct:  {categories['kb_hit_correct']}")
print(f"  KB hit + wrong:    {categories['kb_hit_wrong']}")
print(f"  KB miss + correct: {categories['kb_miss_correct']} (lucky guesses)")
print(f"  KB miss + wrong:   {categories['kb_miss_wrong']} (no coverage)")
