from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

class SubjectDetector:
    """Auto-detect the MMLU subject of a question from its content.

    Uses keyword-to-subject mapping to route questions to the most relevant
    knowledge partition, enabling focused retrieval that dramatically improves
    accuracy over unfocused whole-KB search.

    v3.0: 120+ keyword rules across 30+ MMLU subjects.
    """

    # Keyword → subject mapping (ordered most-specific first)
    KEYWORD_RULES: List[Tuple[List[str], str]] = [
        # Abstract Algebra
        (["group", "subgroup", "abelian", "cyclic group", "isomorphism", "homomorphism",
          "lagrange", "sylow", "quotient group", "normal subgroup", "kernel",
          "ring", "ideal", "field extension", "galois", "automorphism",
          "GF(", "Z_", "S_n", "order of"], "abstract_algebra"),
        # Formal Logic
        (["modus ponens", "modus tollens", "tautology", "contradiction",
          "valid argument", "sound argument", "affirming the consequent",
          "denying the antecedent", "de morgan", "contrapositive", "converse",
          "propositional", "predicate logic", "truth table", "logical connective",
          "universal quantifier", "existential quantifier"], "formal_logic"),
        # Anatomy
        (["femur", "humerus", "tibia", "vertebra", "cranial nerve", "vagus",
          "cerebellum", "medulla oblongata", "thalamus", "hypothalamus",
          "sinoatrial", "pacemaker", "alveoli", "diaphragm", "peristalsis",
          "pituitary", "thyroid", "adrenal", "endocrine", "pancreatic",
          "rotator cuff", "patella", "occipital", "frontal lobe"], "anatomy"),
        # Medical Genetics
        (["autosomal dominant", "autosomal recessive", "x-linked",
          "trisomy", "BRCA", "genetic disorder", "huntington",
          "cystic fibrosis", "sickle cell", "hemophilia", "karyotype"], "medical_genetics"),
        # College Physics
        (["newton's law", "coulomb", "maxwell's equation", "lorentz force",
          "heisenberg", "schrödinger", "carnot", "boltzmann entropy",
          "thermodynamic", "work-energy theorem", "photoelectric"], "college_physics"),
        # College Chemistry
        (["ionic bond", "covalent bond", "VSEPR", "hybridization",
          "hess's law", "gibbs free energy", "enthalpy", "electronegativity",
          "molecular geometry", "sp3", "sp2"], "college_chemistry"),
        # HS Chemistry
        (["periodic table", "noble gas", "halogen", "alkali metal",
          "pH scale", "acid", "base", "oxidation", "reduction",
          "avogadro", "molar", "ideal gas law", "PV = nRT",
          "exothermic", "endothermic", "catalyst"], "high_school_chemistry"),
        # HS Biology
        (["photosynthesis", "mitosis", "meiosis", "DNA", "RNA",
          "natural selection", "evolution", "allele", "genotype", "phenotype",
          "immune", "antibody", "antigen", "vaccine", "trophic",
          "ecosystem", "food chain", "biome", "decomposer",
          "hemoglobin", "insulin", "hormone", "neuron",
          "organelle", "chloroplast"], "high_school_biology"),
        # College Biology
        (["cell biology", "mitochondria", "ATP", "oxidative phosphorylation",
          "hardy-weinberg", "epigenetics", "central dogma",
          "codominance", "genetic drift", "speciation",
          "phylogenetics", "convergent evolution"], "college_biology"),
        # HS Physics
        (["wavelength", "frequency", "hertz", "doppler effect",
          "electromagnetic spectrum", "snell's law", "refraction",
          "reflection", "alpha decay", "beta decay", "gamma decay",
          "half-life", "nuclear fission", "nuclear fusion", "E = mc",
          "kinetic energy", "potential energy", "momentum", "inertia",
          "centripetal", "Bernoulli", "terminal velocity"], "high_school_physics"),
        # Computer Science / ML
        (["algorithm", "big-o", "binary search", "hash table",
          "sorting", "Dijkstra", "NP-complete", "turing machine",
          "BFS", "DFS", "dynamic programming", "data structure"], "college_computer_science"),
        (["neural network", "backpropagation", "transformer",
          "gradient descent", "overfitting", "cross-validation",
          "supervised learning", "unsupervised", "bias-variance",
          "CNN", "LSTM", "dropout"], "machine_learning"),
        # Astronomy
        (["planet", "solar system", "star", "white dwarf", "neutron star",
          "black hole", "red giant", "main sequence", "mercury",
          "jupiter", "saturn", "Neptune", "galaxy", "supernova"], "astronomy"),
        # HS Statistics
        (["bayes", "normal distribution", "standard deviation",
          "central limit theorem", "p-value", "correlation",
          "confidence interval", "sample mean", "hypothesis test"], "high_school_statistics"),
        # Computer Security
        (["RSA", "AES", "SHA-256", "cryptography", "encryption",
          "public key", "zero-knowledge", "firewall", "vulnerability"], "computer_security"),
        # Electrical Engineering
        (["ohm's law", "kirchhoff", "impedance", "capacitance",
          "inductance", "op-amp", "resonant frequency", "RC circuit"], "electrical_engineering"),
        # Philosophy
        (["epistemology", "rationalism", "empiricism", "Descartes", "Hume",
          "Kant", "utilitarianism", "deontological", "virtue ethics",
          "categorical imperative", "veil of ignorance", "Rawls",
          "Plato", "Aristotle", "Socrates", "a priori"], "philosophy"),
        # World Religions
        (["Christianity", "Islam", "Hinduism", "Buddhism", "Judaism",
          "Sikhism", "Confucianism", "five pillars", "Torah",
          "Quran", "dharma", "karma", "eightfold path"], "world_religions"),
        # History
        (["World War I", "World War II", "French Revolution", "Renaissance",
          "Industrial Revolution", "Cold War", "Declaration of Independence",
          "Civil War", "Emancipation", "Civil Rights Act",
          "magna carta", "protestant reformation", "Napoleon"], "high_school_world_history"),
        # Government & Politics
        (["democracy", "federalism", "separation of powers",
          "bill of rights", "electoral college", "filibuster",
          "gerrymandering", "judicial review", "checks and balances",
          "first amendment", "second amendment"], "high_school_government_and_politics"),
        # Psychology
        (["classical conditioning", "operant conditioning", "Pavlov",
          "Skinner", "Piaget", "Erikson", "Maslow", "Freud",
          "cognitive dissonance", "bystander effect",
          "Stanford prison", "Milgram", "confirmation bias"], "high_school_psychology"),
        # Economics
        (["GDP", "inflation", "unemployment", "fiscal policy",
          "monetary policy", "supply and demand", "elasticity",
          "opportunity cost", "marginal", "monopoly", "oligopoly",
          "price ceiling", "price floor", "Federal Reserve"], "high_school_macroeconomics"),
        # Sociology
        (["socialization", "stratification", "deviance", "Durkheim",
          "Marx", "Weber", "functionalism", "conflict theory",
          "symbolic interactionism", "anomie"], "sociology"),
        # International Law
        (["Geneva Convention", "UN Charter", "sovereignty",
          "ICJ", "diplomatic immunity", "jus cogens",
          "customary international law", "Law of the Sea"], "international_law"),
        # Nutrition
        (["macronutrient", "carbohydrate", "protein", "fat", "vitamin",
          "calorie", "BMI", "amino acid", "scurvy", "anemia"], "nutrition"),
        # Clinical Knowledge
        (["vital signs", "blood pressure", "heart rate",
          "diagnosis", "sensitivity", "specificity",
          "CBC", "ECG", "EKG", "differential diagnosis"], "clinical_knowledge"),
        # Professional Medicine
        (["pharmacokinetics", "pharmacodynamics", "agonist", "antagonist",
          "therapeutic index", "bioavailability", "myocardial infarction",
          "stroke", "pneumonia", "sepsis", "anaphylaxis"], "professional_medicine"),
    ]

    def __init__(self):
        self._detections = 0

    def detect(self, question: str, choices: Optional[List[str]] = None) -> Optional[str]:
        """Detect the most likely MMLU subject from question text.

        Returns the subject string or None if no confident match.
        Uses question text as primary signal; choices as secondary (requires
        stronger match to avoid answer-choice bias).
        """
        q_text = question.lower()
        choice_text = " ".join(c.lower() for c in choices) if choices else ""

        best_subject = None
        best_score = 0

        for keywords, subject in self.KEYWORD_RULES:
            q_score = 0
            c_score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in q_text:
                    q_score += len(kw_lower.split())
                elif kw_lower in choice_text:
                    c_score += len(kw_lower.split())
            # Question matches are authoritative; choice-only matches need >= 2
            total = q_score + c_score
            from_question = q_score >= 1
            if total > best_score and (from_question or c_score >= 2):
                best_score = total
                best_subject = subject

        if best_score >= 1:
            self._detections += 1
            return best_subject
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 4c: NUMERICAL REASONER — Extract/compute numerical answers
# ═══════════════════════════════════════════════════════════════════════════════

class NumericalReasoner:
    """Extract numerical values from knowledge facts and compare with answer choices.

    Handles quantitative MMLU questions where the answer is a number, formula,
    or unit-bearing quantity. Parses facts for numerical patterns and matches
    them to multiple-choice options.

    v3.0: supports integers, decimals, scientific notation, common units.
    """

    # Common numerical patterns in facts
    _NUM_PATTERN = re.compile(
        r'(?:approximately|about|roughly|exactly|equals?|is|=)\s*'
        r'([+-]?\d[\d,]*\.?\d*(?:\s*[×x]\s*10\^?\d+)?)\s*'
        r'([a-zA-Z/°²³%]+(?:\s*per\s*[a-z]+)?)?',
        re.IGNORECASE
    )
    _PLAIN_NUM = re.compile(r'(?<!\w)(\d[\d,]*\.?\d*)(?!\w)')

    def __init__(self):
        self._assists = 0

    def extract_numbers(self, text: str) -> List[Tuple[float, Optional[str]]]:
        """Extract (value, unit) pairs from text."""
        results = []
        for m in self._NUM_PATTERN.finditer(text):
            try:
                raw = m.group(1).replace(",", "")
                # Handle scientific notation like "6.022 × 10^23"
                if "×" in raw or "x" in raw:
                    parts = re.split(r'[×x]', raw)
                    if len(parts) == 2 and "10" in parts[1]:
                        exp = re.search(r'10\^?(\d+)', parts[1])
                        if exp:
                            val = float(parts[0].strip()) * (10 ** int(exp.group(1)))
                        else:
                            val = float(parts[0].strip())
                    else:
                        val = float(parts[0].strip())
                else:
                    val = float(raw)
                unit = m.group(2).strip() if m.group(2) else None
                results.append((val, unit))
            except (ValueError, IndexError):
                continue
        return results

    # Scientific notation patterns for choice text (no prefix required)
    _SCI_NOTATION = re.compile(
        r'(\d[\d,]*\.?\d*)\s*[×x]\s*10\s*[\^⁰¹²³⁴⁵⁶⁷⁸⁹]+\s*(\d*)',
        re.IGNORECASE
    )
    _SUPERSCRIPT_MAP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')

    def _parse_choice_numbers(self, text: str) -> List[float]:
        """Parse numbers from choice text, including scientific notation.

        Handles: "3 × 10^8", "3 × 10⁸", "6.022 × 10^23", plain "42"
        """
        results = []
        # 1. Try scientific notation with ^ or Unicode superscripts
        # Normalize Unicode superscripts first
        normalized = text.translate(self._SUPERSCRIPT_MAP)
        for m in re.finditer(
            r'(\d[\d,]*\.?\d*)\s*[×x]\s*10\s*\^?\s*(\d+)',
            normalized, re.IGNORECASE
        ):
            try:
                base = float(m.group(1).replace(",", ""))
                exp = int(m.group(2))
                results.append(base * (10 ** exp))
            except (ValueError, OverflowError):
                continue
        if results:
            return results

        # 2. Fallback to plain numbers
        for m in self._PLAIN_NUM.finditer(text):
            try:
                results.append(float(m.group(1).replace(",", "")))
            except ValueError:
                continue
        return results

    def score_numerical_match(self, choice: str, context_facts: List[str],
                              question: str) -> float:
        """Score how well a choice's numerical content matches context facts.

        Returns a bonus score (0.0-8.0) for numerical agreement.
        v10.0: Parses scientific notation from choices and facts properly.
        """
        # Extract numbers from choice (with scientific notation support)
        choice_nums = self._parse_choice_numbers(choice)

        if not choice_nums:
            return 0.0

        # Extract numbers from context facts (especially those matching question keywords)
        q_words = {w.lower() for w in re.findall(r'\w+', question) if len(w) > 3}
        bonus = 0.0

        for fact in context_facts:
            fact_lower = fact.lower()
            # Only consider facts relevant to the question
            if not any(w in fact_lower for w in q_words):
                continue
            # Also parse fact numbers with scientific notation support
            fact_nums = self._parse_choice_numbers(fact)
            # Plus structured extraction from "is/equals" patterns
            fact_pairs = self.extract_numbers(fact)
            all_fact_vals = [v for v in fact_nums] + [v for v, _ in fact_pairs]
            # Deduplicate
            seen_vals = set()
            unique_fact_vals = []
            for v in all_fact_vals:
                rounded = round(v, 6)
                if rounded not in seen_vals:
                    seen_vals.add(rounded)
                    unique_fact_vals.append(v)

            for fact_val in unique_fact_vals:
                for c_val in choice_nums:
                    # Exact match
                    if abs(fact_val - c_val) < 0.001:
                        bonus += 5.0
                        self._assists += 1
                    # Close match (within 5%)
                    elif fact_val != 0 and abs(fact_val - c_val) / abs(fact_val) < 0.05:
                        bonus += 3.0
                        self._assists += 1
                    # Same order of magnitude
                    elif fact_val > 0 and c_val > 0:
                        ratio = max(fact_val, c_val) / max(min(fact_val, c_val), 1e-30)
                        if ratio < 10:
                            bonus += 0.5

        return min(bonus, 8.0)  # Cap to avoid runaway scoring


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 6: CROSS-VERIFICATION ENGINE — Multi-strategy answer validation
# ═══════════════════════════════════════════════════════════════════════════════
