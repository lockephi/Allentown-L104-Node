#!/usr/bin/env python3
"""Diagnose ARC failure cases to identify algorithmic improvement targets."""

import json, sys, re

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine

eng = CommonsenseReasoningEngine()
eng.initialize()

# All 10 ARC sample questions
questions = [
    ("A metal spoon left in a pot of boiling water becomes hot. This is an example of:",
     ["radiation", "convection", "conduction", "evaporation"], 2),
    ("Which of these objects would be attracted to a magnet?",
     ["a wooden ruler", "a glass marble", "an iron nail", "a rubber band"], 2),
    ("What happens to water when it freezes?",
     ["It contracts", "It expands", "It evaporates", "It stays the same volume"], 1),
    ("What causes the tides on Earth?",
     ["Wind", "The Moon's gravity", "Earth's rotation", "Volcanic activity"], 1),
    ("Which layer of Earth's atmosphere do we live in?",
     ["Stratosphere", "Mesosphere", "Troposphere", "Thermosphere"], 2),
    ("Plants make their own food through a process called:",
     ["respiration", "photosynthesis", "fermentation", "digestion"], 1),
    ("What is the function of white blood cells?",
     ["Carry oxygen", "Fight infection", "Clot blood", "Carry nutrients"], 1),
    ("A simple machine that is a flat surface set at an angle is called a:",
     ["lever", "pulley", "inclined plane", "wheel and axle"], 2),
    ("An ice cube left on a table at room temperature will:",
     ["stay frozen", "melt", "evaporate immediately", "get colder"], 1),
    ("A bat uses sound to navigate in the dark. This is most similar to:",
     ["a dog using smell to track prey",
      "a submarine using sonar to detect objects",
      "a bird using its eyes to find food",
      "a plant growing toward light"], 1),
]

print("=" * 70)
print("ARC FAILURE DIAGNOSIS")
print("=" * 70)

correct_count = 0
for i, (q, choices, expected) in enumerate(questions):
    result = eng.answer_mcq(q, choices)
    predicted = result.get("answer_index", -1)
    is_correct = predicted == expected
    if is_correct:
        correct_count += 1
    status = "PASS" if is_correct else "FAIL"

    print(f"\nQ{i+1} [{status}]: {q}")
    print(f"  Expected: {chr(65+expected)} ({choices[expected]})")
    print(f"  Predicted: {chr(65+predicted)} ({choices[predicted]})")
    print(f"  All scores: {result['all_scores']}")
    print(f"  Concepts: {result['concepts_found'][:8]}")

    if not is_correct:
        print(f"  --- FAILURE ANALYSIS ---")
        # Show causal matches
        causal_matches = eng.causal.query(q.lower(), top_k=5)
        print(f"  Causal rules matched ({len(causal_matches)}):")
        for rule, score in causal_matches:
            print(f"    [{score:.3f}] IF '{rule.condition[:55]}' THEN '{rule.effect[:55]}'")

        # Check what concept properties exist for the correct answer
        correct_choice = choices[expected].lower().replace(' ', '_')
        concept = eng.ontology.concepts.get(correct_choice)
        if concept:
            print(f"  Correct answer has concept: {correct_choice}")
            print(f"    Properties: {dict(list(concept.properties.items())[:5])}")
        else:
            print(f"  Correct answer '{correct_choice}' NOT in ontology")

        # Check what connects question words to correct answer
        q_words = set(re.findall(r'\w+', q.lower()))
        correct_words = set(re.findall(r'\w+', choices[expected].lower()))
        predicted_words = set(re.findall(r'\w+', choices[predicted].lower()))

        # Find causal rules that connect to the correct answer
        correct_causal = []
        for rule, score in eng.causal.query(q.lower(), top_k=20):
            effect_words = set(re.findall(r'\w+', rule.effect.lower()))
            if correct_words & effect_words:
                correct_causal.append((rule, score, correct_words & effect_words))
        print(f"  Causal rules mentioning correct answer:")
        for rule, score, overlap in correct_causal[:3]:
            print(f"    [{score:.3f}] {rule.effect[:60]} (overlap: {overlap})")

        # Check if any causal chain could connect
        print(f"  Multi-step chain check:")
        # Find rules where Q words match condition, then check if effect connects to correct answer
        for rule, score in eng.causal.query(q.lower(), top_k=10):
            eff_words = set(re.findall(r'\w+', rule.effect.lower()))
            # Now search for rules where this effect leads to the correct answer
            for rule2, score2 in eng.causal.rules[:]:
                cond2_words = set(re.findall(r'\w+', rule2.condition.lower()))
                eff2_words = set(re.findall(r'\w+', rule2.effect.lower()))
                if eff_words & cond2_words and eff2_words & correct_words:
                    print(f"    CHAIN: '{rule.condition[:40]}' -> '{rule.effect[:40]}' -> '{rule2.effect[:40]}'")
                    break

print(f"\n{'='*70}")
print(f"TOTAL: {correct_count}/{len(questions)} ({correct_count/len(questions)*100:.0f}%)")
print(f"{'='*70}")

# Now run 20 additional harder multi-step reasoning questions to stress-test
print("\n\n" + "=" * 70)
print("MULTI-STEP REASONING STRESS TEST (20 questions)")
print("=" * 70)

harder_questions = [
    ("Why do we see lightning before we hear thunder?",
     ["Sound is louder than light", "Light travels faster than sound",
      "Lightning is closer than thunder", "Our eyes react before our ears"], 1),
    ("What would happen to a plant kept in complete darkness for several weeks?",
     ["It would grow faster", "It would die because it cannot photosynthesize",
      "It would produce more oxygen", "Nothing would change"], 1),
    ("If you heat a metal bar, what happens to its length?",
     ["It gets shorter", "It stays the same", "It gets longer", "It bends"], 2),
    ("Why does a hot air balloon rise?",
     ["Hot air is less dense than cool air", "Gravity pushes it up",
      "Wind lifts it", "It is filled with helium"], 0),
    ("What happens to sound in a vacuum?",
     ["It gets louder", "It echoes", "It cannot travel", "It travels faster"], 2),
    ("When you rub your hands together quickly, they get warm because of:",
     ["static electricity", "magnetism", "friction", "gravity"], 2),
    ("Which process returns water from the atmosphere to the ground?",
     ["evaporation", "transpiration", "precipitation", "condensation"], 2),
    ("A fox that lives in the Arctic is most likely to have:",
     ["dark fur for warmth", "thin fur for cooling",
      "thick white fur for camouflage and insulation", "no fur at all"], 2),
    ("What type of rock is formed from cooled lava?",
     ["sedimentary", "metamorphic", "igneous", "fossil"], 2),
    ("What gas do plants produce during photosynthesis?",
     ["carbon dioxide", "nitrogen", "oxygen", "hydrogen"], 2),
    ("Where does most evaporation of water take place?",
     ["rivers", "lakes", "oceans", "puddles"], 2),
    ("What causes day and night on Earth?",
     ["Earth orbiting the Sun", "Earth rotating on its axis",
      "The Moon blocking sunlight", "Changes in seasons"], 1),
    ("An object at rest will stay at rest unless acted on by a force. This is:",
     ["Newton's First Law", "Newton's Second Law", "Newton's Third Law", "Law of Gravity"], 0),
    ("What happens when warm air rises in the atmosphere?",
     ["It stays warm", "It cools and may form clouds",
      "It heats up further", "Nothing happens"], 1),
    ("If a food chain loses its producers, what happens?",
     ["Only top predators are affected", "Nothing changes",
      "All consumers are eventually affected", "Only herbivores are affected"], 2),
    ("What energy transformation occurs when a ball rolls down a hill?",
     ["Chemical to thermal", "Kinetic to potential",
      "Potential to kinetic", "Sound to light"], 2),
    ("Which is the best conductor of heat?",
     ["wood", "plastic", "metal", "rubber"], 2),
    ("What process changes liquid water to water vapor?",
     ["condensation", "freezing", "evaporation", "melting"], 2),
    ("Bacteria that break down dead organisms are called:",
     ["producers", "consumers", "decomposers", "predators"], 2),
    ("What would happen if the Earth rotated faster?",
     ["Days would be longer", "Days would be shorter",
      "There would be no seasons", "The Moon would move away"], 1),
]

h_correct = 0
h_fails = []
for i, (q, choices, expected) in enumerate(harder_questions):
    result = eng.answer_mcq(q, choices)
    predicted = result.get("answer_index", -1)
    is_correct = predicted == expected
    if is_correct:
        h_correct += 1
    status = "PASS" if is_correct else "FAIL"
    print(f"Q{i+1} [{status}]: {q[:70]}")
    print(f"  Exp: {chr(65+expected)} ({choices[expected][:40]}) | Got: {chr(65+predicted)} ({choices[predicted][:40]})")
    print(f"  Scores: {result['all_scores']}")
    if not is_correct:
        h_fails.append((q, choices, expected, predicted, result))

print(f"\nHarder set: {h_correct}/{len(harder_questions)} ({h_correct/len(harder_questions)*100:.0f}%)")
print(f"Combined: {correct_count + h_correct}/{len(questions) + len(harder_questions)}")

# Analyze failure patterns
if h_fails:
    print(f"\n--- FAILURE PATTERN ANALYSIS ---")
    for q, choices, expected, predicted, result in h_fails:
        q_words = set(re.findall(r'\w+', q.lower()))
        correct_words = set(re.findall(r'\w+', choices[expected].lower()))
        pred_words = set(re.findall(r'\w+', choices[predicted].lower()))
        concepts = result['concepts_found']

        # Check if correct answer words appear in any concept properties
        has_property_match = False
        for ck in concepts:
            c = eng.ontology.concepts.get(ck)
            if c:
                props_str = str(c.properties).lower()
                for w in correct_words:
                    if len(w) > 3 and w in props_str:
                        has_property_match = True

        # Check causal coverage
        causal = eng.causal.query(q.lower(), top_k=5)
        has_causal = False
        for rule, score in causal:
            if any(w in rule.effect.lower() for w in correct_words if len(w) > 3):
                has_causal = True

        print(f"  Q: {q[:60]}...")
        print(f"    Property match for correct answer: {has_property_match}")
        print(f"    Causal coverage for correct answer: {has_causal}")
        print(f"    Concepts found: {concepts[:5]}")
