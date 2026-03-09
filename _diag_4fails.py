#!/usr/bin/env python3
"""Quick diagnosis of 4 failing ARC questions."""
import re, sys
sys.stderr = open('/dev/null', 'w')

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
eng = CommonsenseReasoningEngine()
eng.initialize()

failures = [
    ("What happens to water when it freezes?",
     ["It contracts", "It expands", "It evaporates", "It stays the same volume"], 1),
    ("What causes the tides on Earth?",
     ["Wind", "The Moon's gravity", "Earth's rotation", "Volcanic activity"], 1),
    ("What is the function of white blood cells?",
     ["Carry oxygen", "Fight infection", "Clot blood", "Carry nutrients"], 1),
    ("A bat uses sound to navigate in the dark. This is most similar to:",
     ["a dog using smell to track prey", "a submarine using sonar to detect objects",
      "a bird using its eyes to find food", "a plant growing toward light"], 1),
]

for q, choices, expected in failures:
    result = eng.answer_mcq(q, choices)
    predicted = result.get("answer_index", -1)
    status = "PASS" if predicted == expected else "FAIL"
    print(f"\n[{status}] {q}")
    print(f"  Exp: {chr(65+expected)}={choices[expected]} | Got: {chr(65+predicted)}={choices[predicted]}")
    print(f"  Scores: {result['all_scores']}")
    print(f"  Concepts: {result['concepts_found'][:6]}")

    # Causal check
    causal = eng.causal.query(q.lower(), top_k=5)
    for rule, score in causal:
        eff_words = set(re.findall(r'\w+', rule.effect.lower()))
        correct_words = set(re.findall(r'\w+', choices[expected].lower()))
        overlap = eff_words & correct_words
        marker = " <<<CORRECT" if overlap else ""
        print(f"  Causal[{score:.2f}]: '{rule.condition[:50]}' -> '{rule.effect[:50]}'{marker}")

print("\nDone.")
