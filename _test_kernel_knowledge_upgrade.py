#!/usr/bin/env python3
"""Validate v4.0.0 kernel knowledge upgrade."""

from l104_asi.language_comprehension import LanguageComprehensionEngine, MMLUKnowledgeBase

print("=== KERNEL KNOWLEDGE UPGRADE VALIDATION ===\n")

# Test 1: Knowledge base build
print("Phase 1: Building MMLUKnowledgeBase...")
kb = MMLUKnowledgeBase()
kb.initialize()
status = kb.get_status()
print(f"  Total nodes:    {status['total_nodes']}")
print(f"  Total facts:    {status['total_facts']}")
print(f"  Subjects:       {status['subjects_covered']}")
print(f"  Categories:     {status['categories']}")
print(f"  Relation edges: {status['relation_edges']}")
print(f"  N-gram phrases: {status['ngram_phrases_indexed']}")

# Test 2: Engine init
print("\nPhase 2: Initializing LanguageComprehensionEngine...")
engine = LanguageComprehensionEngine()
print(f"  Engine version: {engine.VERSION}")
engine.initialize()
es = engine.get_status()
print(f"  Initialized:    {es['initialized']}")
print(f"  KB nodes:       {es['knowledge_base']['total_nodes']}")
print(f"  KB facts:       {es['knowledge_base']['total_facts']}")

# Test 3: MCQ validation across domains
print("\nPhase 3: MCQ validation across upgraded domains...")
tests = [
    ("What is the powerhouse of the cell?",
     ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"],
     "B", "college_biology"),
    ("What does SN2 stand for in organic chemistry?",
     ["Substitution nucleophilic bimolecular", "Second nucleophilic", "Standard neutralization 2", "None of these"],
     "A", "college_chemistry"),
    ("In Nash equilibrium, what is true?",
     ["All players maximize social welfare", "No player can benefit by unilaterally changing strategy",
      "Players always cooperate", "The game must be zero-sum"],
     "B", "high_school_microeconomics"),
    ("What is the Chinese Room argument about?",
     ["Language translation", "Consciousness and understanding", "Chinese philosophy", "Room design"],
     "B", "philosophy"),
    ("The burden of proof in criminal cases requires?",
     ["Preponderance of evidence", "Clear and convincing evidence",
      "Beyond a reasonable doubt", "Substantial evidence"],
     "C", "professional_law"),
    ("What does GDP stand for?",
     ["Gross Domestic Product", "General Distribution Plan", "Global Development Program", "Government Debt Payment"],
     "A", "high_school_macroeconomics"),
]

correct = 0
for q, choices, expected, subject in tests:
    result = engine.answer_mcq(q, choices, subject)
    answer = result.get("answer", "?")
    match = "✓" if answer == expected else "✗"
    if answer == expected:
        correct += 1
    print(f"  {match} [{subject}] Answer: {answer} (expected {expected})")

print(f"\n  Score: {correct}/{len(tests)}")

# Test 4: New domain verification
print("\nPhase 4: Verifying new knowledge domains exist...")
new_domains = [
    "college_chemistry/organic_chemistry",
    "college_chemistry/equilibrium_kinetics",
    "college_chemistry/electrochemistry",
    "college_physics/statistical_mechanics",
    "college_physics/relativity",
    "college_computer_science/quantum_computing",
    "high_school_computer_science/programming_concepts",
    "college_mathematics/advanced_linear_algebra",
    "college_mathematics/number_theory",
    "high_school_statistics/advanced_statistics",
    "college_biology/molecular_biology",
    "college_biology/metabolism",
    "anatomy/renal_system",
    "anatomy/lymphatic_immune",
    "machine_learning/advanced_ml",
    "philosophy/philosophy_of_mind",
    "philosophy/political_philosophy",
    "philosophy/metaphysics",
    "formal_logic/modal_logic",
    "high_school_world_history/ancient_civilizations",
    "high_school_world_history/modern_history",
    "high_school_microeconomics/behavioral_economics",
    "high_school_microeconomics/game_theory",
    "security_studies/international_relations",
    "professional_psychology/abnormal_psychology",
    "professional_psychology/social_psychology",
    "sociology/social_institutions",
    "high_school_government_and_politics/comparative_politics",
    "professional_law/constitutional_law",
    "professional_law/criminal_law",
    "professional_law/contract_tort_law",
    "clinical_knowledge/laboratory_medicine",
    "clinical_knowledge/infectious_disease",
    "professional_medicine/cardiology",
    "professional_medicine/endocrinology",
    "professional_medicine/neurology",
    "miscellaneous/art_history",
    "miscellaneous/literature",
    "miscellaneous/linguistics",
    "miscellaneous/science_literacy",
    "miscellaneous/technology_literacy",
    "miscellaneous/general_trivia",
    "medical_genetics/molecular_genetics",
    "high_school_macroeconomics/monetary_theory",
    "professional_accounting/advanced_accounting",
    "human_aging/geriatric_medicine",
    "virology/viral_mechanisms",
    "nutrition/clinical_nutrition",
]

found = 0
missing = []
for domain in new_domains:
    if domain in kb.nodes:
        found += 1
    else:
        missing.append(domain)

print(f"  Found: {found}/{len(new_domains)} new knowledge domains")
if missing:
    print(f"  Missing: {missing}")

print(f"\n=== KERNEL KNOWLEDGE v4.0.0 UPGRADE {'VALIDATED' if not missing else 'HAS ISSUES'} ===")
print(f"=== {status['total_nodes']} nodes | {status['total_facts']} facts | {status['relation_edges']} relations ===")
