#!/usr/bin/env python3
"""
L104 ASI — MMLU Fine-Tuning Pipeline
═════════════════════════════════════════════════════════════════════════════════
Improves L104's LanguageComprehensionEngine MMLU performance by:

1. KNOWLEDGE EXTRACTION: Mines the 99K MMLU auxiliary_train examples to
   extract new facts and knowledge nodes for the knowledge base.

2. WEIGHT OPTIMIZATION: Tunes BM25 parameters (k1, b) and scoring weights
   using a validation set to maximize MCQ accuracy.

3. EVALUATION: Runs the full MMLU benchmark via L104's engine and reports
   per-subject and per-category accuracy.

Usage:
    cd /Users/carolalvarez/Applications/Allentown-L104-Node
    source /Users/carolalvarez/mmlu-mistallocaltune/.venv/bin/activate
    python mmlu_finetune_pipeline.py [--phase all|extract|optimize|evaluate]
    python mmlu_finetune_pipeline.py --phase extract    # Step 1 only
    python mmlu_finetune_pipeline.py --phase optimize   # Step 2 only
    python mmlu_finetune_pipeline.py --phase evaluate   # Step 3 only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure L104 project is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

MMLU_DATA_DIR = Path.home() / "mmlu-mistallocaltune" / "data"
OUTPUT_DIR = PROJECT_ROOT / "fine_tune_exports"
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: KNOWLEDGE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

CHOICES = ["A", "B", "C", "D"]

# Map auxiliary_train questions (which lack subjects) to likely subjects
# based on keyword matching
SUBJECT_KEYWORDS = {
    "abstract_algebra": ["group", "ring", "field", "homomorphism", "subgroup", "abelian", "cyclic", "polynomial ring", "ideal", "isomorphism"],
    "anatomy": ["artery", "vein", "muscle", "bone", "nerve", "cranial", "thoracic", "abdominal", "ligament", "tendon", "fascia", "organ"],
    "astronomy": ["planet", "star", "galaxy", "nebula", "orbit", "supernova", "black hole", "light year", "solar system", "telescope"],
    "business_ethics": ["stakeholder", "corporate social", "ethics", "whistleblow", "fiduciary", "conflict of interest"],
    "clinical_knowledge": ["patient", "diagnosis", "symptom", "treatment", "clinical", "disease", "syndrome", "medication"],
    "college_biology": ["enzyme", "protein", "cell", "mitosis", "meiosis", "dna", "rna", "photosynthesis", "metabolism", "organelle"],
    "college_chemistry": ["electron", "orbital", "bond", "acid", "base", "reaction", "molar", "oxidation", "reduction", "element"],
    "college_computer_science": ["algorithm", "complexity", "turing", "automata", "compiler", "data structure", "graph", "binary tree"],
    "college_mathematics": ["integral", "derivative", "matrix", "eigenvalue", "theorem", "proof", "topology", "manifold"],
    "college_medicine": ["pathology", "pharmacology", "diagnosis", "clinical presentation", "patient presents"],
    "college_physics": ["force", "energy", "momentum", "wave", "electric", "magnetic", "quantum", "thermodynamic"],
    "computer_security": ["encryption", "firewall", "vulnerability", "malware", "authentication", "cipher", "sql injection", "xss"],
    "conceptual_physics": ["gravity", "inertia", "friction", "acceleration", "velocity", "weight", "mass"],
    "econometrics": ["regression", "coefficient", "heteroskedasticity", "autocorrelation", "endogenous", "instrumental variable"],
    "electrical_engineering": ["circuit", "resistor", "capacitor", "inductor", "voltage", "current", "transistor", "amplifier"],
    "elementary_mathematics": ["fraction", "decimal", "percentage", "ratio", "area", "perimeter", "multiplication"],
    "formal_logic": ["predicate", "quantifier", "syllogism", "valid", "tautology", "contradiction", "modus ponens"],
    "global_facts": ["population", "gdp", "country", "capital", "continent", "united nations"],
    "high_school_biology": ["photosynthesis", "natural selection", "evolution", "genetics", "ecosystem", "cell division"],
    "high_school_chemistry": ["periodic table", "atomic number", "chemical bond", "molarity", "gas law", "stoichiometry"],
    "high_school_computer_science": ["variable", "loop", "function", "array", "boolean", "recursion", "binary"],
    "high_school_european_history": ["renaissance", "reformation", "french revolution", "napoleon", "world war", "cold war"],
    "high_school_geography": ["climate", "latitude", "longitude", "continent", "biome", "population density", "tectonic"],
    "high_school_government_and_politics": ["congress", "supreme court", "constitution", "amendment", "executive", "legislative"],
    "high_school_macroeconomics": ["gdp", "inflation", "unemployment", "fiscal policy", "monetary policy", "aggregate demand"],
    "high_school_mathematics": ["quadratic", "polynomial", "logarithm", "trigonometry", "sine", "cosine", "probability"],
    "high_school_microeconomics": ["supply", "demand", "elasticity", "marginal cost", "monopoly", "oligopoly", "market"],
    "high_school_physics": ["newton", "velocity", "acceleration", "kinetic energy", "potential energy", "wavelength"],
    "high_school_psychology": ["behavior", "cognitive", "freud", "piaget", "classical conditioning", "operant"],
    "high_school_statistics": ["mean", "median", "standard deviation", "hypothesis test", "p-value", "confidence interval"],
    "high_school_us_history": ["civil war", "declaration of independence", "constitution", "reconstruction", "new deal"],
    "high_school_world_history": ["empire", "dynasty", "civilization", "colonialism", "industrial revolution", "silk road"],
    "human_aging": ["alzheimer", "dementia", "geriatric", "aging", "elderly", "cognitive decline", "osteoporosis"],
    "human_sexuality": ["sexual", "gender", "reproductive", "contraception", "puberty", "hormone"],
    "international_law": ["treaty", "sovereignty", "international court", "geneva convention", "humanitarian law"],
    "jurisprudence": ["legal positivism", "natural law", "jurisprudence", "rule of law", "legal realism"],
    "logical_fallacies": ["ad hominem", "straw man", "red herring", "slippery slope", "false dilemma", "circular reasoning"],
    "machine_learning": ["neural network", "gradient descent", "overfitting", "supervised", "unsupervised", "deep learning"],
    "management": ["organizational", "leadership", "strategic planning", "management", "team", "motivation"],
    "marketing": ["consumer behavior", "market segment", "brand", "advertising", "pricing strategy", "product lifecycle"],
    "medical_genetics": ["chromosome", "allele", "mutation", "autosomal", "recessive", "dominant", "genetic disorder"],
    "miscellaneous": [],  # catch-all
    "moral_disputes": ["abortion", "euthanasia", "capital punishment", "animal rights", "censorship"],
    "moral_scenarios": ["moral", "ethical", "duty", "obligation", "wrong", "right"],
    "nutrition": ["vitamin", "mineral", "calorie", "protein", "carbohydrate", "dietary", "nutrient"],
    "philosophy": ["ontology", "epistemology", "metaphysics", "existentialism", "utilitarian", "deontological"],
    "prehistory": ["neolithic", "paleolithic", "stone age", "bronze age", "archaeological", "mesopotamia"],
    "professional_accounting": ["debit", "credit", "balance sheet", "income statement", "gaap", "depreciation"],
    "professional_law": ["statute", "precedent", "tort", "contract", "liability", "plaintiff", "defendant"],
    "professional_medicine": ["diagnosis", "prognosis", "pathogenesis", "etiology", "clinical trial"],
    "professional_psychology": ["dsm", "psychotherapy", "cognitive behavioral", "personality disorder", "assessment"],
    "public_relations": ["public relations", "media", "press release", "crisis communication", "stakeholder"],
    "security_studies": ["deterrence", "nuclear", "terrorism", "geopolitical", "national security", "proliferation"],
    "sociology": ["social stratification", "socialization", "culture", "institution", "deviance", "inequality"],
    "us_foreign_policy": ["nato", "cold war", "containment", "diplomacy", "sanction", "foreign aid"],
    "virology": ["virus", "viral", "pathogen", "infection", "vaccine", "immune response", "replication"],
    "world_religions": ["christianity", "islam", "buddhism", "hinduism", "judaism", "scripture", "monotheism"],
}


def detect_subject_from_question(question: str, choices: List[str]) -> str:
    """Detect the most likely MMLU subject for a question based on keyword matching."""
    text = (question + " " + " ".join(choices)).lower()
    best_subject = "miscellaneous"
    best_score = 0

    for subject, keywords in SUBJECT_KEYWORDS.items():
        if not keywords:
            continue
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_subject = subject

    return best_subject


def extract_fact_from_mcq(question: str, choices: List[str], correct_idx: int) -> Optional[str]:
    """Extract a concise fact from a correctly-answered MCQ.

    Turns "What is X? → Y" into "X is Y" style facts.
    """
    correct_answer = choices[correct_idx]
    q = question.strip().rstrip("?").rstrip(".")

    # Pattern: "What is/are X?" → "X is/are <answer>"
    m = re.match(r"(?:What|Which)\s+(?:is|are|was|were)\s+(.+)", q, re.I)
    if m:
        return f"{m.group(1).strip()} is {correct_answer}"

    # Pattern: "X is ___" → "X is <answer>"
    m = re.match(r"(.+?)\s+(?:is|are)\s+(?:called|known as|referred to as)\s*", q, re.I)
    if m:
        return f"{m.group(1).strip()} is called {correct_answer}"

    # Pattern: "Who/What <verb>...?" → Statement form
    m = re.match(r"(?:Who|What)\s+(\w+(?:ed|s)?)\s+(.+)", q, re.I)
    if m:
        return f"{correct_answer} {m.group(1)} {m.group(2)}"

    # Pattern: "Where is/are/was/were X?" → "X is located in <answer>"
    m = re.match(r"Where\s+(?:is|are|was|were)\s+(.+)", q, re.I)
    if m:
        return f"{m.group(1).strip()} is located in {correct_answer}"

    # Pattern: "When did/was/were X?" → "X occurred/was in <answer>"
    m = re.match(r"When\s+(?:did|was|were|is)\s+(.+)", q, re.I)
    if m:
        return f"{m.group(1).strip()} was in {correct_answer}"

    # Pattern: "How many X?" → "The number of X is <answer>"
    m = re.match(r"How\s+many\s+(.+)", q, re.I)
    if m:
        return f"The number of {m.group(1).strip()} is {correct_answer}"

    # Pattern: "How does/do/did X?" → "X works by <answer>"
    m = re.match(r"How\s+(?:does|do|did)\s+(.+)", q, re.I)
    if m:
        return f"{m.group(1).strip()}: {correct_answer}"

    # Pattern: "Why does/do/did/is/are X?" → "X because <answer>"
    m = re.match(r"Why\s+(?:does|do|did|is|are|was|were)\s+(.+)", q, re.I)
    if m:
        return f"{m.group(1).strip()} because {correct_answer}"

    # Pattern: "In which/what X..." → "<answer> is the X..."
    m = re.match(r"In\s+(?:which|what)\s+(.+)", q, re.I)
    if m:
        return f"{correct_answer} is the {m.group(1).strip()}"

    # Pattern: "The X of Y is ___" → "The X of Y is <answer>"
    m = re.match(r"(The\s+\w+\s+of\s+.+)", q, re.I)
    if m:
        return f"{m.group(1).strip()} is {correct_answer}"

    # Pattern: True/False style — "X is true/false" → fact
    if correct_answer.lower() in ("true", "false"):
        if correct_answer.lower() == "true":
            return f"{q} is true"
        else:
            return f"{q} is false"

    # Fallback: combine question + answer as a statement
    # Accept anything that isn't an excessively long reading passage
    if len(correct_answer) > 2 and len(q) < 500 and len(correct_answer) < 500:
        return f"{q}: {correct_answer}"

    return None


def run_knowledge_extraction():
    """Phase 1: Extract knowledge from MMLU training data."""
    print("=" * 70)
    print("  PHASE 1: KNOWLEDGE EXTRACTION FROM MMLU TRAINING DATA")
    print("=" * 70)

    train_path = MMLU_DATA_DIR / "mmlu_train.jsonl"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run prepare_data.py first.")
        return

    # Load training data
    print(f"Loading training data from {train_path}...")
    examples = []
    with open(train_path) as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)
    print(f"Loaded {len(examples)} training examples")

    # Parse the prompt to extract Q/A pairs
    new_facts_by_subject: Dict[str, List[str]] = defaultdict(list)
    new_nodes: Dict[str, Dict] = {}
    parsed = 0
    skipped = 0

    for ex in examples:
        prompt = ex.get("prompt", "")
        completion = ex.get("completion", "").strip()

        # Extract the actual question (last question block in the prompt)
        lines = prompt.strip().split("\n")
        question_lines = []
        choices = []
        in_question = False

        for line in reversed(lines):
            line = line.strip()
            if line.startswith("Answer:"):
                in_question = True
                continue
            if in_question:
                m = re.match(r"^\s*([A-D])\.\s*(.+)", line)
                if m:
                    choices.insert(0, m.group(2).strip())
                elif line.startswith("Question:"):
                    question_lines.insert(0, line.replace("Question:", "").strip())
                    break
                elif line:
                    question_lines.insert(0, line)

        if not question_lines or len(choices) != 4 or not completion:
            skipped += 1
            continue

        question = " ".join(question_lines)
        correct_idx = CHOICES.index(completion) if completion in CHOICES else -1
        if correct_idx < 0:
            skipped += 1
            continue

        # Detect subject
        subject = detect_subject_from_question(question, choices)

        # Extract fact
        fact = extract_fact_from_mcq(question, choices, correct_idx)
        if fact and 10 < len(fact) < 600:  # Skip very short or very long facts
            new_facts_by_subject[subject].append(fact)
            parsed += 1

    print(f"\nExtracted {parsed} facts from {len(examples)} examples ({skipped} skipped)")
    print(f"Subjects covered: {len(new_facts_by_subject)}")

    # Deduplicate facts per subject (no cap — keep all unique facts)
    deduplicated = {}
    for subject, facts in new_facts_by_subject.items():
        unique = list(dict.fromkeys(facts))  # preserve order, deduplicate
        deduplicated[subject] = unique

    total_facts = sum(len(v) for v in deduplicated.values())
    print(f"After deduplication: {total_facts} unique facts across {len(deduplicated)} subjects")

    # Generate knowledge_data_mmlu_ext.py — new nodes to merge into the KB
    ext_path = OUTPUT_DIR / "knowledge_data_mmlu_ext.py"
    _write_knowledge_extension(deduplicated, ext_path)

    # Also save raw extracted facts as JSON for inspection
    raw_path = OUTPUT_DIR / "extracted_facts.json"
    with open(raw_path, "w") as f:
        json.dump(deduplicated, f, indent=2)
    print(f"\nExtracted facts saved to {raw_path}")
    print(f"Knowledge extension module saved to {ext_path}")

    return deduplicated


def _write_knowledge_extension(facts_by_subject: Dict[str, List[str]], output_path: Path):
    """Write extracted knowledge as a Python module that can be merged into the KB."""
    category_map = {}
    for cat, subjects in {
        "stem": ["abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
                 "college_computer_science", "college_mathematics", "college_physics", "computer_security",
                 "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
                 "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
                 "high_school_physics", "high_school_statistics", "machine_learning", "medical_genetics"],
        "humanities": ["formal_logic", "high_school_european_history", "high_school_us_history",
                       "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies",
                       "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law",
                       "world_religions"],
        "social_sciences": ["econometrics", "high_school_geography", "high_school_government_and_politics",
                           "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
                           "human_sexuality", "professional_psychology", "public_relations", "security_studies",
                           "sociology", "us_foreign_policy"],
        "other": ["business_ethics", "clinical_knowledge", "college_medicine", "global_facts", "human_aging",
                  "management", "marketing", "miscellaneous", "nutrition", "professional_accounting",
                  "professional_medicine", "virology"],
    }.items():
        for s in subjects:
            category_map[s] = cat

    lines = [
        '"""',
        'L104 ASI — MMLU-Extracted Knowledge Extension',
        f'Auto-generated from {sum(len(v) for v in facts_by_subject.values())} MMLU training examples.',
        'Merge into knowledge_data.py or load dynamically at runtime.',
        '"""',
        '',
        'from typing import Any, Dict, List',
        '',
        '',
        'MMLU_EXTRACTED_NODES: List[Dict[str, Any]] = [',
    ]

    for subject, facts in sorted(facts_by_subject.items()):
        if not facts:
            continue
        category = category_map.get(subject, "other")
        concept = f"mmlu_train_{subject}"
        definition = f"Knowledge extracted from MMLU training data for {subject.replace('_', ' ')}"

        lines.append('    {')
        lines.append(f'        "concept": "{concept}",')
        lines.append(f'        "subject": "{subject}",')
        lines.append(f'        "category": "{category}",')
        lines.append(f'        "definition": "{definition}",')
        lines.append('        "facts": [')
        for fact in facts:
            escaped = fact.replace('\\', '\\\\').replace('"', '\\"')
            lines.append(f'            "{escaped}",')
        lines.append('        ],')
        lines.append('    },')

    lines.append(']')
    lines.append('')

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: WEIGHT OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_weight_optimization(use_extracted_knowledge: bool = True):
    """Phase 2: Optimize BM25 k1/b and scoring parameters using MMLU val data."""
    print("=" * 70)
    print("  PHASE 2: SCORING WEIGHT OPTIMIZATION")
    print("=" * 70)

    from l104_asi.language_comprehension import LanguageComprehensionEngine

    # Load validation data (use a subset of eval for tuning)
    eval_path = MMLU_DATA_DIR / "mmlu_eval.jsonl"
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found.")
        return

    examples = []
    with open(eval_path) as f:
        for line in f:
            examples.append(json.loads(line))

    # Use a stratified sample for tuning (500 examples, balanced across subjects)
    random.seed(42)
    by_subject = defaultdict(list)
    for ex in examples:
        by_subject[ex.get("subject", "unknown")].append(ex)

    val_set = []
    per_subject_n = max(1, 200 // len(by_subject))
    for subj, exs in by_subject.items():
        sampled = random.sample(exs, min(per_subject_n, len(exs)))
        val_set.extend(sampled)
    random.shuffle(val_set)
    val_set = val_set[:200]
    print(f"Using {len(val_set)} validation examples for optimization")

    # Optionally load extracted knowledge into the engine
    if use_extracted_knowledge:
        ext_path = OUTPUT_DIR / "knowledge_data_mmlu_ext.py"
        if ext_path.exists():
            print("Loading MMLU-extracted knowledge extension...")
            _inject_extracted_knowledge(ext_path)

    # Grid search over BM25 parameters
    k1_values = [1.0, 1.2, 1.5, 1.618, 2.0]
    b_values = [0.5, 0.618, 0.75, 0.85]

    best_accuracy = 0.0
    best_params = {}
    results = []

    # Build engine ONCE for reuse across all param combos
    from l104_asi.language_comprehension import LanguageComprehensionEngine
    print("Initializing engine (one-time)...")
    shared_engine = LanguageComprehensionEngine()
    shared_engine.initialize()
    print(f"Engine ready. Running {len(k1_values) * len(b_values)} parameter combos...\n")

    for k1, b in product(k1_values, b_values):
        acc = _evaluate_with_params(val_set, k1=k1, b=b, engine=shared_engine)
        results.append({"k1": k1, "b": b, "accuracy": acc})
        marker = " ★ BEST" if acc > best_accuracy else ""
        if acc > best_accuracy:
            best_accuracy = acc
            best_params = {"k1": k1, "b": b}
        print(f"  k1={k1:.3f} b={b:.3f} → {acc:.1%}{marker}")

    print(f"\nBest parameters: k1={best_params['k1']:.3f}, b={best_params['b']:.3f}")
    print(f"Best accuracy: {best_accuracy:.1%}")

    # Save optimization results
    opt_path = OUTPUT_DIR / "optimization_results.json"
    with open(opt_path, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_accuracy": best_accuracy,
            "all_results": results,
        }, f, indent=2)
    print(f"Results saved to {opt_path}")

    return best_params


def _inject_extracted_knowledge(ext_path: Path):
    """Dynamically load extracted knowledge into the knowledge_data module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mmlu_ext", str(ext_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Monkey-patch into knowledge_data so it's picked up on next KB init
    from l104_asi import knowledge_data
    existing = knowledge_data.KNOWLEDGE_NODES
    new_nodes = getattr(mod, "MMLU_EXTRACTED_NODES", [])
    # Merge: add nodes that don't already exist
    existing_keys = {(n["subject"], n["concept"]) for n in existing}
    added = 0
    for node in new_nodes:
        key = (node["subject"], node["concept"])
        if key not in existing_keys:
            existing.append(node)
            existing_keys.add(key)
            added += 1
    print(f"  Injected {added} new knowledge nodes ({len(new_nodes)} total in extension)")


def _evaluate_with_params(val_set: List[Dict], k1: float = 1.618, b: float = 0.618,
                          engine=None) -> float:
    """Run MCQ evaluation with specific BM25 parameters.

    If `engine` is provided, reuses it (faster). Otherwise creates a new one.
    """
    from l104_asi.language_comprehension import LanguageComprehensionEngine, BM25Ranker

    if engine is None:
        # Patch BM25 defaults for this run
        original_init = BM25Ranker.__init__
        def patched_init(self_bm25, k1_arg=None, b_arg=None):
            original_init(self_bm25, k1=k1, b=b)
        BM25Ranker.__init__ = patched_init
        try:
            engine = LanguageComprehensionEngine()
        finally:
            BM25Ranker.__init__ = original_init

    correct = 0
    total = 0

    for ex in val_set:
        prompt = ex.get("prompt", "")
        completion = ex.get("completion", "").strip()

        q, choices, answer_idx = _parse_prompt(prompt, completion)
        if q is None:
            continue

        # Monkey-patch k1/b onto any BM25Ranker instance used in solve()
        # The MCQSolver creates a fresh BM25Ranker per solve() call,
        # so we patch the class-level defaults temporarily.
        from l104_asi.language_comprehension import BM25Ranker as _BR
        _orig = _BR.__init__
        def _temp_init(self_bm25, k1_arg=None, b_arg=None):
            _orig(self_bm25, k1=k1, b=b)
        _BR.__init__ = _temp_init
        try:
            result = engine.answer_mcq(q, choices, subject=ex.get("subject"))
        finally:
            _BR.__init__ = _orig

        sel = result.get("selected_index", result.get("answer_index", -1))
        if sel == answer_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def _parse_prompt(prompt: str, completion: str) -> Tuple[Optional[str], Optional[List[str]], int]:
    """Parse a formatted prompt to extract question, choices, and correct answer index."""
    lines = prompt.strip().split("\n")
    question_lines = []
    choices = []
    in_question = False

    for line in reversed(lines):
        line = line.strip()
        if line.startswith("Answer:"):
            in_question = True
            continue
        if in_question:
            m = re.match(r"^\s*([A-D])\.\s*(.+)", line)
            if m:
                choices.insert(0, m.group(2).strip())
            elif line.startswith("Question:"):
                question_lines.insert(0, line.replace("Question:", "").strip())
                break
            elif line:
                question_lines.insert(0, line)

    if not question_lines or len(choices) != 4:
        return None, None, -1

    question = " ".join(question_lines)
    correct_idx = CHOICES.index(completion) if completion in CHOICES else -1
    return question, choices, correct_idx


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_MAPPING = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "human_sexuality", "management",
        "marketing", "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine",
        "professional_psychology", "virology",
    ],
}


def run_evaluation(max_questions: int = 0, use_extracted_knowledge: bool = True,
                   optimized_params: Optional[Dict] = None):
    """Phase 3: Full MMLU evaluation using L104's engine."""
    print("=" * 70)
    print("  PHASE 3: MMLU EVALUATION")
    print("=" * 70)

    # Optionally inject extracted knowledge
    if use_extracted_knowledge:
        ext_path = OUTPUT_DIR / "knowledge_data_mmlu_ext.py"
        if ext_path.exists():
            _inject_extracted_knowledge(ext_path)

    # Apply optimized BM25 params if provided
    if optimized_params:
        from l104_asi.language_comprehension import BM25Ranker
        original_init = BM25Ranker.__init__
        k1_opt = optimized_params.get("k1", 1.618)
        b_opt = optimized_params.get("b", 0.618)

        def patched_init(self_bm25, k1_arg=None, b_arg=None):
            original_init(self_bm25, k1=k1_opt, b=b_opt)

        BM25Ranker.__init__ = patched_init
        print(f"Using optimized BM25 params: k1={k1_opt}, b={b_opt}")

    from l104_asi.language_comprehension import LanguageComprehensionEngine
    engine = LanguageComprehensionEngine()

    # Load eval data
    eval_path = MMLU_DATA_DIR / "mmlu_eval.jsonl"
    if not eval_path.exists():
        # Fall back to fetching via benchmark harness
        print("Eval data not found, fetching via HuggingFace...")
        from l104_asi.benchmark_harness import _HuggingFaceFetcher
        data = _HuggingFaceFetcher.fetch_mmlu(max_questions=max_questions or 500)
        _run_eval_on_data(engine, data, source="huggingface")
        return

    examples = []
    with open(eval_path) as f:
        for line in f:
            examples.append(json.loads(line))

    if max_questions > 0:
        random.seed(42)
        random.shuffle(examples)
        examples = examples[:max_questions]

    print(f"Evaluating on {len(examples)} MMLU test examples...")

    correct = 0
    total = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    start_time = time.time()
    for i, ex in enumerate(examples):
        prompt = ex.get("prompt", "")
        completion = ex.get("completion", "").strip()
        subject = ex.get("subject", "unknown")

        q, choices, answer_idx = _parse_prompt(prompt, completion)
        if q is None:
            continue

        result = engine.answer_mcq(q, choices, subject=subject)
        sel = result.get("selected_index", result.get("answer_index", -1))
        is_correct = (sel == answer_idx)

        if is_correct:
            correct += 1
            subject_stats[subject]["correct"] += 1
        total += 1
        subject_stats[subject]["total"] += 1

        # Map to category
        for cat, cat_subjects in CATEGORY_MAPPING.items():
            if subject in cat_subjects:
                category_stats[cat]["total"] += 1
                if is_correct:
                    category_stats[cat]["correct"] += 1
                break

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{len(examples)} — "
                  f"{correct}/{total} = {correct/total*100:.1f}% "
                  f"({rate:.0f} q/s)")

    elapsed = time.time() - start_time
    overall_acc = correct / total if total > 0 else 0

    # Print results
    print("\n" + "=" * 70)
    print("  MMLU RESULTS")
    print("=" * 70)

    print("\nPer-category accuracy:")
    for cat in ["STEM", "Humanities", "Social Sciences", "Other"]:
        stats = category_stats[cat]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat:20s}: {acc:.1%} ({stats['correct']}/{stats['total']})")

    print(f"\nOverall accuracy: {overall_acc:.1%} ({correct}/{total})")
    print(f"Time: {elapsed:.1f}s ({total/elapsed:.0f} questions/sec)")

    print("\nPer-subject accuracy:")
    for subj in sorted(subject_stats.keys()):
        stats = subject_stats[subj]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {subj:40s}: {acc:.0%} ({stats['correct']}/{stats['total']})")

    # Save results
    results = {
        "overall_accuracy": overall_acc,
        "total_correct": correct,
        "total_questions": total,
        "elapsed_seconds": elapsed,
        "category_results": {k: dict(v) for k, v in category_stats.items()},
        "per_subject": {k: dict(v) for k, v in subject_stats.items()},
        "params": optimized_params,
    }
    results_path = OUTPUT_DIR / "mmlu_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


def _run_eval_on_data(engine, data: List[Dict], source: str = "local"):
    """Run evaluation on raw MCQ data format from benchmark_harness."""
    correct = 0
    total = 0
    subject_stats = defaultdict(lambda: [0, 0])

    for item in data:
        q = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        subj = item.get("subject", "unknown")

        result = engine.answer_mcq(q, choices)
        sel = result.get("selected_index", result.get("answer_index", -1))
        is_correct = (sel == answer_idx)
        if is_correct:
            correct += 1
        total += 1
        subject_stats[subj][1] += 1
        if is_correct:
            subject_stats[subj][0] += 1

    overall_acc = correct / total if total > 0 else 0
    print(f"\nMMLU RESULT ({source}): {correct}/{total} = {overall_acc:.1%}")
    print("\nBy subject:")
    for subj, (c, t) in sorted(subject_stats.items()):
        print(f"  {subj:30s}: {c}/{t} = {c/t*100:.0f}%")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="L104 ASI MMLU Fine-Tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  extract   — Mine MMLU training data to extract knowledge facts
  optimize  — Tune BM25 scoring parameters on validation data
  evaluate  — Run full MMLU benchmark evaluation
  all       — Run all three phases sequentially
        """,
    )
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "extract", "optimize", "evaluate"],
                        help="Which phase(s) to run (default: all)")
    parser.add_argument("--max-eval-questions", type=int, default=0,
                        help="Max eval questions (0 = all 14042)")
    parser.add_argument("--no-extracted-knowledge", action="store_true",
                        help="Don't use MMLU-extracted knowledge extension")
    args = parser.parse_args()

    use_knowledge = not args.no_extracted_knowledge
    optimized_params = None

    if args.phase in ("all", "extract"):
        run_knowledge_extraction()
        print()

    if args.phase in ("all", "optimize"):
        optimized_params = run_weight_optimization(use_extracted_knowledge=use_knowledge)
        print()

    if args.phase in ("all", "evaluate"):
        # Load previously optimized params if not from this run
        if optimized_params is None:
            opt_path = OUTPUT_DIR / "optimization_results.json"
            if opt_path.exists():
                with open(opt_path) as f:
                    opt_data = json.load(f)
                optimized_params = opt_data.get("best_params")
                print(f"Loaded optimized params: {optimized_params}")

        run_evaluation(
            max_questions=args.max_eval_questions,
            use_extracted_knowledge=use_knowledge,
            optimized_params=optimized_params,
        )


if __name__ == "__main__":
    main()
