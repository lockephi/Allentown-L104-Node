#!/usr/bin/env python3
"""Comprehensive MCQ test battery for quantum probability pipeline.

Tests both CommonsenseMCQSolver (ARC) and MCQSolver (MMLU) across
diverse question types to validate all pipeline upgrades:
- Oracle v6.0/v4.0 (5-tier matching, stemming, trigram fuzzy)
- BM25 stopword filtering
- Causal rule stem matching
- _score_choice tokenization fixes
- Stage 8 causal/composition relation patterns
- Fallback heuristics v4.0
- measurement_collapse sharpening
- CrossVerificationEngine v4.0

Each test specifies the expected correct answer and logs whether
the pipeline gets it right.
"""

import sys
import time

# ═══════════════════════════════════════════════════════════════════════════════
# ARC-style commonsense questions (CommonsenseMCQSolver)
# ═══════════════════════════════════════════════════════════════════════════════

ARC_QUESTIONS = [
    # ── States of Matter ──
    {
        "question": "What happens to water when it is heated to 100 degrees Celsius?",
        "choices": ["It freezes into ice", "It boils and becomes steam",
                    "It stays the same", "It becomes denser"],
        "answer": 1,
        "topic": "states_of_matter",
    },
    {
        "question": "Which process causes water to change from a liquid to a gas?",
        "choices": ["condensation", "evaporation", "precipitation", "freezing"],
        "answer": 1,
        "topic": "states_of_matter",
    },
    {
        "question": "When water vapor in the air cools, what process occurs?",
        "choices": ["evaporation", "melting", "condensation", "sublimation"],
        "answer": 2,
        "topic": "states_of_matter",
    },

    # ── Heat Transfer ──
    {
        "question": "When ice is placed in a glass of warm tea, what happens to the heat?",
        "choices": ["Heat flows from ice to tea", "Heat flows from tea to ice",
                    "No heat transfer occurs", "Heat flows equally both ways"],
        "answer": 1,
        "topic": "heat_transfer",
    },
    {
        "question": "Which type of heat transfer occurs when a metal spoon gets hot in soup?",
        "choices": ["radiation", "convection", "conduction", "evaporation"],
        "answer": 2,
        "topic": "heat_transfer",
    },

    # ── Forces & Motion ──
    {
        "question": "What force causes a ball thrown in the air to come back down?",
        "choices": ["friction", "magnetism", "gravity", "electricity"],
        "answer": 2,
        "topic": "forces",
    },
    {
        "question": "An object at rest will stay at rest unless acted on by what?",
        "choices": ["air resistance", "an unbalanced force", "temperature", "gravity only"],
        "answer": 1,
        "topic": "forces",
    },
    {
        "question": "What happens to the speed of a moving object when friction acts on it?",
        "choices": ["It increases", "It decreases", "It stays the same", "It reverses direction"],
        "answer": 1,
        "topic": "forces",
    },

    # ── Energy ──
    {
        "question": "A ball rolling down a hill converts what type of energy to kinetic energy?",
        "choices": ["thermal energy", "chemical energy", "potential energy", "electrical energy"],
        "answer": 2,
        "topic": "energy",
    },
    {
        "question": "Which is an example of a renewable energy source?",
        "choices": ["coal", "natural gas", "solar power", "petroleum"],
        "answer": 2,
        "topic": "energy",
    },

    # ── Photosynthesis & Biology ──
    {
        "question": "What is the first step in the process of photosynthesis?",
        "choices": ["releasing oxygen", "absorbing carbon dioxide",
                    "chlorophyll captures light energy", "producing glucose"],
        "answer": 2,
        "topic": "biology",
    },
    {
        "question": "Which organelle is known as the powerhouse of the cell?",
        "choices": ["nucleus", "ribosome", "mitochondria", "cell wall"],
        "answer": 2,
        "topic": "biology",
    },
    {
        "question": "What gas do plants produce during photosynthesis?",
        "choices": ["carbon dioxide", "nitrogen", "oxygen", "hydrogen"],
        "answer": 2,
        "topic": "biology",
    },

    # ── Earth Science ──
    {
        "question": "If a tropical plant fossil is found in a cold region, what can we conclude?",
        "choices": ["The fossil was transported from far away",
                    "The area once had a warmer climate",
                    "The plant adapted to cold weather",
                    "The fossil is not real"],
        "answer": 1,
        "topic": "earth_science",
    },
    {
        "question": "What causes the day and night cycle on Earth?",
        "choices": ["Earth's revolution around the Sun",
                    "The Moon's orbit around Earth",
                    "Earth's rotation on its axis",
                    "The Sun moving across the sky"],
        "answer": 2,
        "topic": "earth_science",
    },
    {
        "question": "If Earth rotated faster, what would happen to the length of a day?",
        "choices": ["Days would be longer", "Days would be shorter",
                    "Days would stay the same", "There would be no days"],
        "answer": 1,
        "topic": "earth_science",
    },

    # ── Scientific Method ──
    {
        "question": "What is the first step a scientist should take before conducting an experiment?",
        "choices": ["Record results", "Plan and prepare",
                    "Share findings", "Repeat the experiment"],
        "answer": 1,
        "topic": "scientific_method",
    },

    # ── Weather ──
    {
        "question": "What causes warm air to rise in the atmosphere?",
        "choices": ["gravity", "convection", "condensation", "friction"],
        "answer": 1,
        "topic": "weather",
    },

    # ── Adaptations ──
    {
        "question": "Why do some animals have thick fur in cold environments?",
        "choices": ["to attract mates", "to scare predators",
                    "to keep warm by insulation", "to absorb sunlight"],
        "answer": 2,
        "topic": "biology",
    },

    # ── Simple Machines ──
    {
        "question": "An inclined plane is a flat surface that is:",
        "choices": ["round like a wheel", "at an angle",
                    "very smooth", "made of metal"],
        "answer": 1,
        "topic": "simple_machines",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# MMLU-style knowledge questions (MCQSolver)
# ═══════════════════════════════════════════════════════════════════════════════

MMLU_QUESTIONS = [
    # ── Biology ──
    {
        "question": "What is the primary function of the mitochondria?",
        "choices": ["Protein synthesis", "Energy production through ATP",
                    "DNA replication", "Cell division"],
        "answer": 1,
        "subject": "college_biology",
        "topic": "cell_biology",
    },
    {
        "question": "Which molecule carries genetic information in most organisms?",
        "choices": ["RNA", "DNA", "Protein", "Lipids"],
        "answer": 1,
        "subject": "college_biology",
        "topic": "genetics",
    },

    # ── Chemistry ──
    {
        "question": "What is the chemical symbol for gold?",
        "choices": ["Go", "Gd", "Au", "Ag"],
        "answer": 2,
        "subject": "high_school_chemistry",
        "topic": "elements",
    },
    {
        "question": "What happens during an exothermic reaction?",
        "choices": ["Energy is absorbed from surroundings",
                    "Energy is released to surroundings",
                    "No energy change occurs",
                    "Temperature decreases"],
        "answer": 1,
        "subject": "high_school_chemistry",
        "topic": "reactions",
    },

    # ── Physics ──
    {
        "question": "According to Newton's third law, every action has:",
        "choices": ["no reaction", "a delayed reaction",
                    "an equal and opposite reaction", "a proportional reaction"],
        "answer": 2,
        "subject": "conceptual_physics",
        "topic": "mechanics",
    },
    {
        "question": "What is the SI unit of force?",
        "choices": ["Joule", "Watt", "Newton", "Pascal"],
        "answer": 2,
        "subject": "high_school_physics",
        "topic": "units",
    },

    # ── Geography ──
    {
        "question": "Which planet is closest to the Sun?",
        "choices": ["Venus", "Mercury", "Mars", "Earth"],
        "answer": 1,
        "subject": "astronomy",
        "topic": "solar_system",
    },

    # ── History ──
    {
        "question": "The Renaissance period began in which country?",
        "choices": ["France", "England", "Italy", "Spain"],
        "answer": 2,
        "subject": "high_school_european_history",
        "topic": "renaissance",
    },

    # ── Math ──
    {
        "question": "What is the value of pi rounded to two decimal places?",
        "choices": ["3.12", "3.14", "3.16", "3.18"],
        "answer": 1,
        "subject": "elementary_mathematics",
        "topic": "constants",
    },

    # ── Computer Science ──
    {
        "question": "Which data structure operates on a Last-In-First-Out (LIFO) principle?",
        "choices": ["Queue", "Stack", "Array", "Linked list"],
        "answer": 1,
        "subject": "college_computer_science",
        "topic": "data_structures",
    },
]


def run_arc_battery():
    """Run ARC questions through CommonsenseMCQSolver."""
    print("\n" + "=" * 70)
    print("  ARC BATTERY — CommonsenseMCQSolver")
    print("=" * 70)

    from l104_asi.commonsense_reasoning import (
        CommonsenseMCQSolver, ConceptOntology, CausalReasoningEngine,
        PhysicalIntuition, AnalogicalReasoner, TemporalReasoningEngine,
        CrossVerificationEngine
    )
    ont = ConceptOntology()
    causal = CausalReasoningEngine()
    phys = PhysicalIntuition(ont)
    analog = AnalogicalReasoner(ont)
    temporal = TemporalReasoningEngine()
    verifier = CrossVerificationEngine(ont, causal, temporal)
    solver = CommonsenseMCQSolver(ont, causal, phys, analog, temporal, verifier)

    correct = 0
    total = len(ARC_QUESTIONS)
    results = []

    for i, q in enumerate(ARC_QUESTIONS):
        result = solver.solve(q["question"], q["choices"], subject="science")
        got = result["answer_index"]
        expected = q["answer"]
        ok = got == expected
        if ok:
            correct += 1
        status = "OK" if ok else "MISS"
        label = chr(65 + got)
        exp_label = chr(65 + expected)
        scores = result.get("all_scores", {})
        conf = result.get("confidence", 0)
        results.append({
            "idx": i, "ok": ok, "got": got, "expected": expected,
            "topic": q["topic"], "confidence": conf,
        })
        print(f"  [{status}] Q{i+1:02d} ({q['topic'][:15]:15s}) "
              f"Got={label} Exp={exp_label} Conf={conf:.3f} "
              f"{'| ' + q['question'][:50] if not ok else ''}")

    pct = correct / total * 100
    print(f"\n  ARC RESULT: {correct}/{total} = {pct:.1f}%")
    return correct, total, results


def run_mmlu_battery():
    """Run MMLU questions through MCQSolver."""
    print("\n" + "=" * 70)
    print("  MMLU BATTERY — MCQSolver")
    print("=" * 70)

    from l104_asi.language_comprehension import MCQSolver, MMLUKnowledgeBase
    kb = MMLUKnowledgeBase()
    solver = MCQSolver(knowledge_base=kb)

    correct = 0
    total = len(MMLU_QUESTIONS)
    results = []

    for i, q in enumerate(MMLU_QUESTIONS):
        result = solver.solve(q["question"], q["choices"], subject=q.get("subject"))
        got = result.get("answer_index", result.get("selected_index", -1))
        expected = q["answer"]
        ok = got == expected
        if ok:
            correct += 1
        status = "OK" if ok else "MISS"
        label = chr(65 + got) if got >= 0 else "?"
        exp_label = chr(65 + expected)
        conf = result.get("confidence", 0)
        results.append({
            "idx": i, "ok": ok, "got": got, "expected": expected,
            "topic": q["topic"], "confidence": conf,
        })
        print(f"  [{status}] Q{i+1:02d} ({q['topic'][:15]:15s}) "
              f"Got={label} Exp={exp_label} Conf={conf:.3f} "
              f"{'| ' + q['question'][:50] if not ok else ''}")

    pct = correct / total * 100
    print(f"\n  MMLU RESULT: {correct}/{total} = {pct:.1f}%")
    return correct, total, results


def main():
    t0 = time.time()

    print("=" * 70)
    print("  L104 MCQ TEST BATTERY")
    print(f"  {len(ARC_QUESTIONS)} ARC + {len(MMLU_QUESTIONS)} MMLU = "
          f"{len(ARC_QUESTIONS) + len(MMLU_QUESTIONS)} total questions")
    print("=" * 70)

    arc_correct, arc_total, arc_results = run_arc_battery()
    mmlu_correct, mmlu_total, mmlu_results = run_mmlu_battery()

    elapsed = time.time() - t0
    total_correct = arc_correct + mmlu_correct
    total_questions = arc_total + mmlu_total
    total_pct = total_correct / total_questions * 100

    # Topic breakdown for misses
    arc_misses = [r for r in arc_results if not r["ok"]]
    mmlu_misses = [r for r in mmlu_results if not r["ok"]]

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  ARC:  {arc_correct}/{arc_total} = {arc_correct/arc_total*100:.1f}%")
    print(f"  MMLU: {mmlu_correct}/{mmlu_total} = {mmlu_correct/mmlu_total*100:.1f}%")
    print(f"  TOTAL: {total_correct}/{total_questions} = {total_pct:.1f}%")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total_questions:.2f}s/question)")

    if arc_misses:
        print(f"\n  ARC misses by topic:")
        from collections import Counter
        for topic, count in Counter(r["topic"] for r in arc_misses).most_common():
            print(f"    {topic}: {count} miss(es)")

    if mmlu_misses:
        print(f"\n  MMLU misses by topic:")
        from collections import Counter
        for topic, count in Counter(r["topic"] for r in mmlu_misses).most_common():
            print(f"    {topic}: {count} miss(es)")

    print(f"\n{'=' * 70}")
    return total_correct == total_questions


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
