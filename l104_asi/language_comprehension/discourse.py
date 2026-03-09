from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict

class CoreferenceResolver:
    """Rule-based coreference resolution for pronoun→antecedent linking.

    Resolves third-person pronouns (he/she/it/they/his/her/its/their) to the
    most likely antecedent noun phrase using:
      - Recency heuristic: prefer the most recent compatible NP
      - Gender/number agreement: he→masculine, she→feminine, it→singular neuter
      - Named entity preference: prefer proper nouns over common nouns
      - Syntactic role: subjects are preferred antecedents (centering theory)

    Used in MCQSolver to resolve references in multi-sentence questions so
    scoring stages can match the resolved entity, not just the pronoun.
    """

    _MALE_PRONOUNS = frozenset({"he", "him", "his", "himself"})
    _FEMALE_PRONOUNS = frozenset({"she", "her", "hers", "herself"})
    _NEUTER_PRONOUNS = frozenset({"it", "its", "itself"})
    _PLURAL_PRONOUNS = frozenset({"they", "them", "their", "theirs", "themselves"})
    _ALL_PRONOUNS = _MALE_PRONOUNS | _FEMALE_PRONOUNS | _NEUTER_PRONOUNS | _PLURAL_PRONOUNS

    # Common male/female first names for gender heuristic
    _MALE_NAMES = frozenset({
        "james", "john", "robert", "michael", "william", "david", "richard",
        "joseph", "thomas", "charles", "christopher", "daniel", "matthew",
        "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
        "albert", "isaac", "galileo", "darwin", "newton", "einstein",
        "aristotle", "plato", "socrates", "descartes", "kant", "hegel",
        "marx", "freud", "adam", "alexander", "napoleon", "lincoln",
        "washington", "jefferson", "franklin", "edison", "tesla", "bohr",
        "heisenberg", "maxwell", "faraday", "kepler", "copernicus",
    })
    _FEMALE_NAMES = frozenset({
        "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara",
        "margaret", "susan", "dorothy", "sarah", "jessica", "helen",
        "marie", "ruth", "alice", "anna", "emily", "emma", "grace",
        "jane", "charlotte", "victoria", "catherine", "diana", "cleopatra",
        "rosa", "florence", "amelia", "harriet", "ada", "hypatia",
    })

    # Neuter-leaning common nouns (objects, concepts, processes)
    _NEUTER_NOUNS = frozenset({
        "process", "system", "method", "technique", "approach", "model",
        "theory", "law", "principle", "concept", "mechanism", "structure",
        "element", "compound", "molecule", "cell", "organ", "tissue",
        "experiment", "study", "research", "analysis", "result", "effect",
        "force", "energy", "mass", "velocity", "temperature", "pressure",
        "reaction", "equation", "function", "value", "number", "ratio",
        "algorithm", "program", "device", "machine", "instrument", "tool",
        "country", "city", "river", "mountain", "ocean", "planet", "star",
    })

    def __init__(self):
        self._resolutions = 0

    def _extract_noun_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Extract candidate noun phrases with position and gender info."""
        candidates = []
        # Pattern: optional determiner + optional adjectives + noun(s)
        # Simplified: capture capitalized words (proper nouns) and "the/a/an" + noun phrases
        sentences = re.split(r'[.!?]+', text)
        char_offset = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                char_offset += 1
                continue

            # Proper nouns: capitalized words (not at sentence start) or multi-word names
            words = sent.split()
            for i, word in enumerate(words):
                clean = re.sub(r'[^a-zA-Z]', '', word)
                if not clean:
                    continue

                # Skip pronouns — they are referents, not antecedent candidates
                if clean.lower() in self._ALL_PRONOUNS:
                    continue

                is_proper = (clean[0].isupper() and i > 0) or clean.lower() in self._MALE_NAMES | self._FEMALE_NAMES
                if is_proper or (i == 0 and clean[0].isupper()):
                    # Try to capture multi-word proper nouns
                    name_parts = [clean]
                    j = i + 1
                    while j < len(words):
                        next_clean = re.sub(r'[^a-zA-Z]', '', words[j])
                        if next_clean and next_clean[0].isupper() and next_clean.lower() not in self._ALL_PRONOUNS:
                            name_parts.append(next_clean)
                            j += 1
                        else:
                            break

                    full_name = " ".join(name_parts)
                    first_lower = name_parts[0].lower()

                    # Determine gender
                    gender = "unknown"
                    if first_lower in self._MALE_NAMES:
                        gender = "male"
                    elif first_lower in self._FEMALE_NAMES:
                        gender = "female"

                    # Determine number (simple heuristic)
                    number = "plural" if full_name.lower().endswith("s") and len(full_name) > 3 else "singular"

                    candidates.append({
                        "text": full_name,
                        "position": char_offset + sent.find(word),
                        "gender": gender,
                        "number": number,
                        "is_proper": is_proper or i == 0,
                        "is_subject": i < 3,  # Rough: subjects tend to come early
                    })

            # Common noun phrases: "the/a/an + adjective* + noun"
            for m in re.finditer(r'\b(?:the|a|an)\s+(\w+(?:\s+\w+){0,2})', sent.lower()):
                np_text = m.group(1).strip()
                np_words = np_text.split()
                head_word = np_words[-1] if np_words else np_text

                gender = "neuter" if head_word in self._NEUTER_NOUNS else "unknown"
                number = "plural" if head_word.endswith("s") and not head_word.endswith("ss") else "singular"

                candidates.append({
                    "text": np_text,
                    "position": char_offset + m.start(),
                    "gender": gender,
                    "number": number,
                    "is_proper": False,
                    "is_subject": m.start() < 20,
                })

            char_offset += len(sent) + 1

        return candidates

    def resolve(self, text: str) -> Dict[str, Any]:
        """Resolve pronouns in text to their most likely antecedents.

        Returns:
            Dict with 'resolved_text' (pronouns replaced), 'resolutions' list,
            and 'resolution_count'.
        """
        candidates = self._extract_noun_phrases(text)
        words = text.split()
        resolutions = []

        for i, word in enumerate(words):
            clean = re.sub(r'[^a-zA-Z]', '', word).lower()
            if clean not in self._ALL_PRONOUNS:
                continue

            # Determine pronoun constraints
            if clean in self._MALE_PRONOUNS:
                required_gender = {"male", "unknown"}
                required_number = {"singular"}
            elif clean in self._FEMALE_PRONOUNS:
                required_gender = {"female", "unknown"}
                required_number = {"singular"}
            elif clean in self._NEUTER_PRONOUNS:
                required_gender = {"neuter", "unknown"}
                required_number = {"singular"}
            else:  # plural
                required_gender = {"male", "female", "neuter", "unknown"}
                required_number = {"plural", "singular"}  # "they" can refer to singular too

            # Find best antecedent: most recent compatible candidate before the pronoun
            pronoun_pos = sum(len(words[j]) + 1 for j in range(i))
            best = None
            best_score = -1.0

            for cand in candidates:
                if cand["position"] >= pronoun_pos:
                    continue  # Antecedent must precede the pronoun
                if cand["gender"] not in required_gender:
                    continue
                if clean not in self._PLURAL_PRONOUNS and cand["number"] not in required_number:
                    continue

                # Score: recency + proper noun bonus + subject bonus
                recency = 1.0 / (1.0 + (pronoun_pos - cand["position"]) / 50.0)
                proper_bonus = 0.3 if cand["is_proper"] else 0.0
                subject_bonus = 0.15 if cand["is_subject"] else 0.0
                gender_bonus = 0.2 if cand["gender"] != "unknown" else 0.0
                score = recency + proper_bonus + subject_bonus + gender_bonus

                if score > best_score:
                    best_score = score
                    best = cand

            if best is not None:
                resolutions.append({
                    "pronoun": clean,
                    "antecedent": best["text"],
                    "confidence": round(min(best_score, 1.0), 3),
                    "position": i,
                })
                self._resolutions += 1

        # Build resolved text
        resolved_words = list(words)
        for res in resolutions:
            pos = res["position"]
            # Preserve punctuation around the pronoun
            original = resolved_words[pos]
            prefix = ""
            suffix = ""
            for c in original:
                if c.isalpha():
                    break
                prefix += c
            for c in reversed(original):
                if c.isalpha():
                    break
                suffix = c + suffix
            resolved_words[pos] = prefix + res["antecedent"] + suffix

        return {
            "resolved_text": " ".join(resolved_words),
            "resolutions": resolutions,
            "resolution_count": len(resolutions),
        }

    def resolve_for_scoring(self, question: str) -> str:
        """Return the resolved text suitable for MCQ scoring.

        Replaces pronouns with antecedents so keyword matching can find
        the actual entity being asked about.
        """
        result = self.resolve(question)
        return result["resolved_text"] if result["resolution_count"] > 0 else question


class SentimentAnalyzer:
    """Lexicon-based sentiment analyzer with valence shifters.

    Uses a sentiment lexicon with polarity scores, modified by:
      - Negation: "not good" → flips polarity
      - Intensifiers: "very good" → amplifies polarity
      - Diminishers: "slightly bad" → reduces polarity
      - But-clauses: "good but expensive" → emphasizes post-but sentiment

    Useful for MMLU psychology, ethics, and opinion-based questions where
    sentiment/tone understanding helps distinguish answer choices.
    """

    # Sentiment lexicon: word → polarity score (-1.0 to +1.0)
    _LEXICON = {
        # Strongly positive
        "excellent": 0.9, "outstanding": 0.9, "brilliant": 0.85,
        "wonderful": 0.85, "fantastic": 0.85, "superb": 0.85,
        "exceptional": 0.85, "remarkable": 0.80, "magnificent": 0.80,
        # Moderately positive
        "good": 0.6, "great": 0.7, "nice": 0.5, "best": 0.75,
        "happy": 0.7, "pleased": 0.6, "satisfied": 0.55, "positive": 0.6,
        "beneficial": 0.65, "effective": 0.6, "successful": 0.7,
        "helpful": 0.6, "useful": 0.55, "valuable": 0.65,
        "important": 0.5, "significant": 0.5, "appropriate": 0.45,
        "correct": 0.55, "right": 0.45, "proper": 0.45,
        "healthy": 0.6, "safe": 0.55, "strong": 0.5,
        "improve": 0.55, "increase": 0.35, "enhance": 0.55,
        "support": 0.45, "promote": 0.45, "encourage": 0.5,
        "advantage": 0.6, "benefit": 0.6, "progress": 0.55,
        "agree": 0.4, "accept": 0.4, "approve": 0.5,
        # Mildly positive
        "adequate": 0.25, "sufficient": 0.25, "reasonable": 0.3,
        "fair": 0.3, "moderate": 0.2, "stable": 0.3,
        # Strongly negative
        "terrible": -0.9, "horrible": -0.9, "awful": -0.85,
        "dreadful": -0.85, "atrocious": -0.9, "catastrophic": -0.85,
        "devastating": -0.8, "disastrous": -0.85,
        # Moderately negative
        "bad": -0.6, "poor": -0.55, "wrong": -0.5, "worst": -0.75,
        "sad": -0.6, "unhappy": -0.6, "disappointed": -0.55,
        "harmful": -0.65, "dangerous": -0.6, "toxic": -0.7,
        "negative": -0.5, "ineffective": -0.5, "unsuccessful": -0.6,
        "difficult": -0.35, "problem": -0.4, "issue": -0.25,
        "risk": -0.35, "threat": -0.5, "damage": -0.6,
        "fail": -0.6, "failure": -0.65, "decline": -0.4,
        "decrease": -0.3, "reduce": -0.25, "weaken": -0.45,
        "reject": -0.5, "deny": -0.4, "oppose": -0.4,
        "conflict": -0.45, "crisis": -0.55, "suffer": -0.6,
        "pain": -0.55, "loss": -0.5, "death": -0.6,
        "disease": -0.5, "illness": -0.5, "disorder": -0.4,
        "abuse": -0.75, "violence": -0.7, "crime": -0.6,
        # Mildly negative
        "inadequate": -0.35, "insufficient": -0.35, "limited": -0.2,
        "weak": -0.35, "uncertain": -0.25, "unclear": -0.2,
    }

    _NEGATORS = frozenset({
        "not", "no", "never", "neither", "nor", "nobody", "nothing",
        "nowhere", "hardly", "barely", "scarcely", "rarely", "seldom",
        "without", "lack", "lacking", "absent", "cannot", "can't",
        "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
        "couldn't", "isn't", "aren't", "wasn't", "weren't",
    })

    _INTENSIFIERS = {
        "very": 1.5, "extremely": 1.8, "incredibly": 1.7,
        "remarkably": 1.6, "exceptionally": 1.7, "highly": 1.5,
        "absolutely": 1.6, "completely": 1.5, "totally": 1.5,
        "quite": 1.3, "really": 1.4, "truly": 1.4,
        "particularly": 1.3, "especially": 1.4, "most": 1.4,
    }

    _DIMINISHERS = {
        "slightly": 0.5, "somewhat": 0.6, "rather": 0.7,
        "fairly": 0.7, "a bit": 0.5, "a little": 0.5,
        "mildly": 0.5, "partially": 0.6, "marginally": 0.4,
        "barely": 0.3, "hardly": 0.3,
    }

    def __init__(self):
        self._analyses = 0

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text.

        Returns:
            Dict with 'polarity' (-1 to +1), 'label' (positive/negative/neutral),
            'subjectivity' (0 to 1), and 'word_sentiments' list.
        """
        words = re.findall(r"[a-z']+", text.lower())
        word_sentiments = []
        total_polarity = 0.0
        sentiment_count = 0
        total_words = len(words)

        for i, word in enumerate(words):
            if word not in self._LEXICON:
                continue

            raw_polarity = self._LEXICON[word]
            modifier = 1.0

            # Check for negation in preceding 3 words
            negated = False
            for j in range(max(0, i - 3), i):
                if words[j] in self._NEGATORS:
                    negated = True
                    break

            if negated:
                modifier *= -0.8  # Negate with slight dampening

            # Check for intensifiers in preceding 2 words
            for j in range(max(0, i - 2), i):
                if words[j] in self._INTENSIFIERS:
                    modifier *= self._INTENSIFIERS[words[j]]
                    break

            # Check for diminishers in preceding 2 words
            for j in range(max(0, i - 2), i):
                if words[j] in self._DIMINISHERS:
                    modifier *= self._DIMINISHERS[words[j]]
                    break

            adjusted = raw_polarity * modifier
            word_sentiments.append({
                "word": word,
                "raw_polarity": raw_polarity,
                "adjusted_polarity": round(adjusted, 3),
                "negated": negated,
            })
            total_polarity += adjusted
            sentiment_count += 1

        # But-clause handling: sentiment after "but" is weighted more
        but_idx = None
        for i, word in enumerate(words):
            if word in ("but", "however", "yet", "although", "though", "nevertheless"):
                but_idx = i
                break

        if but_idx is not None and word_sentiments:
            # Reweight: post-but sentiments get 1.5× weight
            for ws in word_sentiments:
                ws_pos = words.index(ws["word"]) if ws["word"] in words else 0
                if ws_pos > but_idx:
                    total_polarity += ws["adjusted_polarity"] * 0.5  # Extra weight

        # Normalize polarity
        if sentiment_count > 0:
            avg_polarity = total_polarity / sentiment_count
            polarity = max(-1.0, min(1.0, avg_polarity))
        else:
            polarity = 0.0

        # Subjectivity: ratio of sentiment-bearing words
        subjectivity = min(1.0, sentiment_count / max(total_words, 1) * 3.0)

        # Label
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"

        self._analyses += 1
        return {
            "polarity": round(polarity, 4),
            "label": label,
            "subjectivity": round(subjectivity, 4),
            "sentiment_words": sentiment_count,
            "word_sentiments": word_sentiments,
        }

    def compare_sentiment(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """Compare sentiment between two texts.

        Returns:
            Dict with polarity difference, agreement flag, and individual results.
        """
        sa = self.analyze(text_a)
        sb = self.analyze(text_b)
        diff = sa["polarity"] - sb["polarity"]
        return {
            "polarity_difference": round(diff, 4),
            "agree": sa["label"] == sb["label"],
            "a": sa,
            "b": sb,
        }


class SemanticFrameAnalyzer:
    """Frame semantics analyzer for question structure understanding.

    Identifies semantic frames evoked by predicates in the question:
      - CAUSE_EFFECT: "X causes Y", "Y results from X"
      - CLASSIFICATION: "X is a type of Y", "X belongs to Y"
      - COMPARISON: "X compared to Y", "X differs from Y"
      - DEFINITION: "X is defined as Y", "X refers to Y"
      - TEMPORAL: "X occurred before/after Y"
      - LOCATION: "X is located in Y", "X occurs at Y"
      - QUANTITY: "how many/much", "the amount of X"
      - FUNCTION: "X is used for Y", "the purpose of X"
      - COMPOSITION: "X consists of Y", "X contains Y"
      - TRANSFORMATION: "X becomes Y", "X converts to Y"

    Helps MCQSolver understand what the question is asking for structurally
    so it can prefer choices that fill the correct frame slot.
    """

    _FRAME_PATTERNS = {
        "CAUSE_EFFECT": [
            (r'what\s+(?:causes?|leads?\s+to|results?\s+in|produces?)\s+([^.?!\n]+)', "effect"),
            (r'what\s+(?:is|are)\s+(?:caused|produced|triggered)\s+by\s+([^.?!\n]+)', "cause"),
            (r'why\s+(?:does|do|did|is|are|was|were)\s+([^.?!\n]+)', "reason"),
            (r'([^.?!\n]+)\s+(?:causes?|leads?\s+to|results?\s+in)\s+([^.?!\n]+)', "cause_effect_pair"),
            (r'the\s+(?:cause|reason|origin)\s+(?:of|for)\s+([^.?!\n]+)', "cause"),
            (r'the\s+(?:effect|result|consequence|outcome)\s+of\s+([^.?!\n]+)', "effect"),
        ],
        "CLASSIFICATION": [
            (r'which\s+(?:of\s+the\s+following\s+)?(?:is|are)\s+(?:a|an)\s+(?:type|kind|form|example)\s+of\s+([^.?!\n]+)', "instance_of"),
            (r'([^.?!\n]+)\s+(?:is|are)\s+(?:a|an)\s+(?:type|kind|form|category)\s+of\s+([^.?!\n]+)', "is_a"),
            (r'([^.?!\n]+)\s+(?:belongs?\s+to|falls?\s+under|is\s+classified\s+as)\s+([^.?!\n]+)', "belongs_to"),
            (r'what\s+(?:type|kind|category|class)\s+(?:of|is)\s+([^.?!\n]+)', "class_query"),
        ],
        "COMPARISON": [
            (r'(?:how|in\s+what\s+way)\s+(?:does|do|is|are)\s+([^.?!\n]+?)\s+(?:differ|compare|contrast)\s+(?:from|with|to)\s+([^.?!\n]+)', "comparison"),
            (r'what\s+is\s+the\s+(?:difference|distinction|similarity)\s+between\s+([^.?!\n]+)\s+and\s+([^.?!\n]+)', "difference"),
            (r'([^.?!\n]+)\s+(?:versus|vs\.?|compared\s+to|as\s+opposed\s+to)\s+([^.?!\n]+)', "versus"),
            (r'which\s+is\s+(?:more|less|better|worse|greater|smaller|larger)\s+([^.?!\n]+)', "superlative"),
        ],
        "DEFINITION": [
            (r'what\s+(?:is|are|does)\s+(?:the\s+)?(?:definition\s+of\s+)?([^.?!\n]+)', "define"),
            (r'(?:define|explain|describe)\s+([^.?!\n]+)', "define"),
            (r'([^.?!\n]+)\s+(?:is\s+defined\s+as|refers?\s+to|means?)\s+([^.?!\n]+)', "definition"),
            (r'what\s+does\s+([^.?!\n]+)\s+(?:mean|stand\s+for|refer\s+to)', "meaning"),
        ],
        "TEMPORAL": [
            (r'when\s+(?:did|does|do|was|were|is|are)\s+([^.?!\n]+)', "time_query"),
            (r'(?:before|after|during|while|since|until)\s+([^.?!\n]+)', "temporal_relation"),
            (r'in\s+what\s+(?:year|century|era|period|decade)\s+([^.?!\n]+)', "time_query"),
            (r'what\s+(?:happened|occurred|took\s+place)\s+(?:in|during|after|before)\s+([^.?!\n]+)', "event_query"),
        ],
        "LOCATION": [
            (r'where\s+(?:is|are|was|were|does|do|did)\s+([^.?!\n]+)', "location_query"),
            (r'in\s+(?:which|what)\s+(?:country|region|city|area|place|location)\s+([^.?!\n]+)', "location_query"),
            (r'([^.?!\n]+)\s+(?:is|are)\s+(?:located|found|situated)\s+(?:in|at|on|near)\s+([^.?!\n]+)', "location"),
        ],
        "QUANTITY": [
            (r'how\s+(?:many|much|often|frequently)\s+([^.?!\n]+)', "quantity_query"),
            (r'what\s+is\s+the\s+(?:number|amount|quantity|percentage|proportion)\s+of\s+([^.?!\n]+)', "quantity_query"),
            (r'(?:approximately|about|roughly)\s+how\s+(?:many|much)\s+([^.?!\n]+)', "approximate_query"),
        ],
        "FUNCTION": [
            (r'what\s+is\s+(?:the\s+)?(?:purpose|function|role|use)\s+of\s+([^.?!\n]+)', "function_query"),
            (r'([^.?!\n]+)\s+(?:is|are)\s+used\s+(?:for|to|in)\s+([^.?!\n]+)', "usage"),
            (r'what\s+(?:does|do)\s+([^.?!\n]+)\s+do', "function_query"),
            (r'the\s+(?:purpose|function|role)\s+of\s+([^.?!\n]+)', "function_query"),
        ],
        "COMPOSITION": [
            (r'what\s+(?:is|are)\s+([^.?!\n]+)\s+(?:made|composed|comprised)\s+of', "component_query"),
            (r'([^.?!\n]+)\s+(?:consists?\s+of|contains?|includes?|is\s+composed\s+of)\s+([^.?!\n]+)', "has_component"),
            (r'what\s+(?:does|do)\s+([^.?!\n]+)\s+(?:contain|include|consist\s+of)', "component_query"),
        ],
        "TRANSFORMATION": [
            (r'([^.?!\n]+)\s+(?:becomes?|turns?\s+into|converts?\s+(?:to|into)|transforms?\s+(?:to|into))\s+([^.?!\n]+)', "becomes"),
            (r'what\s+(?:does|do)\s+([^.?!\n]+)\s+(?:become|turn\s+into|convert\s+to)', "becomes_query"),
            (r'(?:the\s+)?(?:process|conversion)\s+of\s+([^.?!\n]+)\s+(?:to|into)\s+([^.?!\n]+)', "transformation"),
        ],
    }

    def __init__(self):
        self._analyses = 0
        # Pre-compile patterns
        self._compiled_patterns = {}
        for frame, patterns in self._FRAME_PATTERNS.items():
            self._compiled_patterns[frame] = [
                (re.compile(pat, re.IGNORECASE), role) for pat, role in patterns
            ]

    def analyze(self, text: str) -> Dict[str, Any]:
        """Identify semantic frames evoked by the text.

        Returns:
            Dict with 'frames' list (each with frame_type, role, matched_text),
            'primary_frame' (strongest match), and 'frame_count'.
        """
        frames = []
        text_clean = text.strip().rstrip("?.")

        for frame_type, patterns in self._compiled_patterns.items():
            for compiled_pat, role in patterns:
                m = compiled_pat.search(text_clean)
                if m:
                    matched_text = m.group(1).strip() if m.lastindex else ""
                    frames.append({
                        "frame_type": frame_type,
                        "role": role,
                        "matched_text": matched_text,
                        "span": (m.start(), m.end()),
                    })
                    break  # One match per frame type

        # Determine primary frame (first match by pattern specificity)
        primary = frames[0] if frames else None

        self._analyses += 1
        return {
            "frames": frames,
            "primary_frame": primary["frame_type"] if primary else "UNKNOWN",
            "primary_role": primary["role"] if primary else "unknown",
            "frame_count": len(frames),
        }

    def score_choice_frame_fit(self, question: str, choice: str) -> float:
        """Score how well a choice fits the question's semantic frame.

        Returns: 0.0 to 1.0 indicating frame-role compatibility.
        """
        analysis = self.analyze(question)
        if not analysis["frames"]:
            return 0.0

        choice_lower = choice.lower()
        score = 0.0
        primary_frame = analysis["primary_frame"]

        # Frame-specific scoring heuristics
        if primary_frame == "DEFINITION":
            # Definitions tend to be longer, descriptive phrases
            if len(choice.split()) >= 3:
                score += 0.15
            # Check for definitional markers
            if any(w in choice_lower for w in ["process", "method", "state", "condition",
                                                "refers", "means", "type", "form"]):
                score += 0.10

        elif primary_frame == "CAUSE_EFFECT":
            # Causes/effects tend to contain causal language or process nouns
            if any(w in choice_lower for w in ["because", "due", "leads", "causes",
                                                "results", "produces", "increases",
                                                "decreases", "prevents"]):
                score += 0.15
            # Process nouns and scientific terms also indicate causal answers
            if any(w in choice_lower for w in ["oxidation", "reaction", "absorption",
                                                "emission", "diffusion", "erosion",
                                                "evaporation", "condensation",
                                                "presence", "exposure", "interaction"]):
                score += 0.12
            # Longer explanatory answers are more likely causal
            if len(choice.split()) >= 5:
                score += 0.08

        elif primary_frame == "CLASSIFICATION":
            # Classifications match taxonomic language
            if any(w in choice_lower for w in ["type", "kind", "class", "category",
                                                "group", "family", "genus", "species"]):
                score += 0.10

        elif primary_frame == "QUANTITY":
            # Quantity answers tend to contain numbers
            if re.search(r'\d+', choice):
                score += 0.15

        elif primary_frame == "LOCATION":
            # Location answers contain place-like words
            if any(w in choice_lower for w in ["north", "south", "east", "west",
                                                "region", "area", "continent"]):
                score += 0.10

        elif primary_frame == "TEMPORAL":
            # Temporal answers contain time expressions
            if re.search(r'\b\d{3,4}\b', choice):  # Year-like numbers
                score += 0.15
            if any(w in choice_lower for w in ["century", "era", "period", "age",
                                                "year", "decade", "during", "after"]):
                score += 0.10

        elif primary_frame == "FUNCTION":
            # Function answers describe purposes
            if any(w in choice_lower for w in ["to ", "for ", "used", "serves",
                                                "enables", "allows", "helps"]):
                score += 0.10

        elif primary_frame == "COMPOSITION":
            # Composition answers list components
            if any(w in choice_lower for w in ["and", "with", "containing",
                                                "including", "composed"]):
                score += 0.10

        elif primary_frame == "TRANSFORMATION":
            # Transformation answers describe changes
            if any(w in choice_lower for w in ["becomes", "converts", "transforms",
                                                "changes", "turns"]):
                score += 0.10

        return min(score, 0.3)


class TaxonomyClassifier:
    """Hierarchical taxonomy classifier with depth-weighted similarity.

    Maintains a lightweight is-a/part-of taxonomy for common MMLU domains:
    science, medicine, history, law, economics. Supports:
      - Is-a queries: "Is X a type of Y?"
      - Depth-weighted similarity: closer in the hierarchy → higher score
      - Hypernym chain: "cell → eukaryotic cell → animal cell"
      - Part-of chains: "mitochondria → cell → tissue → organ"

    Used to score choices by taxonomic proximity to question concepts.
    """

    # Lightweight taxonomy: child → parent (is-a)
    _IS_A = {
        # Biology taxonomy
        "mitosis": "cell division", "meiosis": "cell division",
        "cell division": "biological process", "photosynthesis": "biological process",
        "respiration": "biological process", "fermentation": "biological process",
        "dna replication": "biological process", "transcription": "biological process",
        "translation": "biological process", "protein synthesis": "biological process",
        "eukaryote": "organism", "prokaryote": "organism",
        "bacteria": "prokaryote", "archaea": "prokaryote",
        "plant": "eukaryote", "animal": "eukaryote", "fungus": "eukaryote",
        "mammal": "animal", "reptile": "animal", "bird": "animal",
        "fish": "animal", "amphibian": "animal", "insect": "animal",
        "vertebrate": "animal", "invertebrate": "animal",
        "nucleus": "organelle", "mitochondria": "organelle",
        "ribosome": "organelle", "chloroplast": "organelle",
        "endoplasmic reticulum": "organelle", "golgi apparatus": "organelle",
        "organelle": "cell component",
        "cell": "biological unit", "tissue": "biological unit",
        "organ": "biological unit", "organ system": "biological unit",
        # Chemistry taxonomy
        "acid": "chemical compound", "base": "chemical compound",
        "salt": "chemical compound", "oxide": "chemical compound",
        "organic compound": "chemical compound", "inorganic compound": "chemical compound",
        "alkane": "hydrocarbon", "alkene": "hydrocarbon", "alkyne": "hydrocarbon",
        "hydrocarbon": "organic compound",
        "amino acid": "organic compound", "carbohydrate": "organic compound",
        "lipid": "organic compound", "protein": "macromolecule",
        "nucleic acid": "macromolecule", "macromolecule": "organic compound",
        "covalent bond": "chemical bond", "ionic bond": "chemical bond",
        "hydrogen bond": "chemical bond", "metallic bond": "chemical bond",
        "chemical bond": "chemical interaction",
        # Physics taxonomy
        "kinetic energy": "energy", "potential energy": "energy",
        "thermal energy": "energy", "electrical energy": "energy",
        "nuclear energy": "energy", "chemical energy": "energy",
        "mechanical energy": "energy", "electromagnetic energy": "energy",
        "gravity": "fundamental force", "electromagnetism": "fundamental force",
        "strong force": "fundamental force", "weak force": "fundamental force",
        "fundamental force": "physical phenomenon",
        "conduction": "heat transfer", "convection": "heat transfer",
        "radiation": "heat transfer", "heat transfer": "physical process",
        # Government taxonomy
        "democracy": "government", "monarchy": "government",
        "oligarchy": "government", "autocracy": "government",
        "republic": "democracy", "theocracy": "government",
        "federalism": "political system", "unitarism": "political system",
        "capitalism": "economic system", "socialism": "economic system",
        "communism": "economic system", "mixed economy": "economic system",
        # Psychology taxonomy
        "classical conditioning": "learning theory",
        "operant conditioning": "learning theory",
        "social learning": "learning theory",
        "cognitive learning": "learning theory",
        "learning theory": "psychological theory",
        "psychoanalysis": "psychological theory",
        "behaviorism": "psychological theory",
        "humanism": "psychological theory",
        "cognitive psychology": "psychological theory",
    }

    # Part-of relationships: part → whole
    _PART_OF = {
        "mitochondria": "cell", "nucleus": "cell", "ribosome": "cell",
        "chloroplast": "plant cell", "cell wall": "plant cell",
        "cell membrane": "cell", "cytoplasm": "cell",
        "cell": "tissue", "tissue": "organ", "organ": "organ system",
        "organ system": "organism",
        "proton": "atom", "neutron": "atom", "electron": "atom",
        "atom": "molecule", "molecule": "compound",
        "chromosome": "nucleus", "gene": "chromosome", "dna": "chromosome",
        "codon": "gene", "nucleotide": "dna",
        "cortex": "brain", "hippocampus": "brain", "cerebellum": "brain",
        "brain": "nervous system", "spinal cord": "nervous system",
        "heart": "circulatory system", "artery": "circulatory system",
        "vein": "circulatory system", "lung": "respiratory system",
        "liver": "digestive system", "kidney": "urinary system",
        "legislature": "government", "judiciary": "government",
        "executive": "government",
    }

    def __init__(self):
        self._lookups = 0

    def _get_ancestors(self, concept: str, relation: dict, max_depth: int = 10) -> List[str]:
        """Get ancestor chain for a concept."""
        ancestors = []
        current = concept.lower()
        visited = set()
        for _ in range(max_depth):
            parent = relation.get(current)
            if parent is None or parent in visited:
                break
            ancestors.append(parent)
            visited.add(parent)
            current = parent
        return ancestors

    def is_a(self, child: str, parent: str) -> bool:
        """Check if child is-a parent (transitive)."""
        self._lookups += 1
        ancestors = self._get_ancestors(child.lower(), self._IS_A)
        return parent.lower() in ancestors

    def part_of(self, part: str, whole: str) -> bool:
        """Check if part is part-of whole (transitive)."""
        self._lookups += 1
        ancestors = self._get_ancestors(part.lower(), self._PART_OF)
        return whole.lower() in ancestors

    def taxonomic_distance(self, concept_a: str, concept_b: str) -> float:
        """Compute taxonomic distance between two concepts.

        Uses the lowest common ancestor (LCA) in the is-a hierarchy.
        Returns: 0.0 (identical) to 1.0 (unrelated). Intermediate values
        indicate taxonomic proximity.
        """
        a_lower = concept_a.lower()
        b_lower = concept_b.lower()

        if a_lower == b_lower:
            return 0.0

        # Check direct is-a
        if self.is_a(a_lower, b_lower):
            return 0.2
        if self.is_a(b_lower, a_lower):
            return 0.2

        # Find LCA
        a_ancestors = [a_lower] + self._get_ancestors(a_lower, self._IS_A)
        b_ancestors = [b_lower] + self._get_ancestors(b_lower, self._IS_A)
        b_set = set(b_ancestors)

        for i, anc in enumerate(a_ancestors):
            if anc in b_set:
                j = b_ancestors.index(anc)
                # Distance based on combined depth
                total_steps = i + j
                return min(1.0, total_steps * 0.15)

        return 1.0  # No common ancestor found

    def taxonomic_similarity(self, concept_a: str, concept_b: str) -> float:
        """Similarity score from taxonomy (1.0 = identical, 0.0 = unrelated)."""
        return 1.0 - self.taxonomic_distance(concept_a, concept_b)

    def score_choice_taxonomy(self, question: str, choice: str) -> float:
        """Score a choice based on taxonomic relevance to the question.

        Extracts concepts from question and choice, computes the best
        taxonomic similarity, and returns a score.
        """
        q_words = set(re.findall(r'\b\w+\b', question.lower()))
        c_words = set(re.findall(r'\b\w+\b', choice.lower()))

        # Find any taxonomy members mentioned
        all_concepts = set(self._IS_A.keys()) | set(self._IS_A.values()) | \
                       set(self._PART_OF.keys()) | set(self._PART_OF.values())

        q_concepts = []
        c_concepts = []
        for concept in all_concepts:
            concept_words = set(concept.split())
            if concept_words <= q_words:
                q_concepts.append(concept)
            if concept_words <= c_words:
                c_concepts.append(concept)

        if not q_concepts or not c_concepts:
            return 0.0

        # Best taxonomic similarity between any question concept and choice concept
        best_sim = 0.0
        for qc in q_concepts:
            for cc in c_concepts:
                sim = self.taxonomic_similarity(qc, cc)
                best_sim = max(best_sim, sim)

        return best_sim * 0.2  # Scale for scoring pipeline


class CausalChainReasoner:
    """Multi-hop causal chain inference engine.

    Maintains a causal knowledge graph and supports:
      - Forward chaining: "X causes Y causes Z" → ask about X given Z
      - Backward chaining: "What causes Z?" → trace back through chain
      - Counterfactual reasoning: "If not X, then not Y?"
      - Causal strength estimation: frequent/well-known causal links score higher

    Extends basic causal detection (DeepNLU) with multi-hop reasoning for
    complex causal questions common in MMLU science/medicine.
    """

    # Causal knowledge: cause → [(effect, strength)]
    _CAUSAL_KB = {
        # Biology causal chains
        "mutation": [("genetic variation", 0.9), ("disease", 0.5), ("evolution", 0.7)],
        "genetic variation": [("natural selection", 0.8), ("adaptation", 0.7)],
        "natural selection": [("evolution", 0.9), ("speciation", 0.7)],
        "deforestation": [("habitat loss", 0.9), ("soil erosion", 0.8), ("climate change", 0.6)],
        "habitat loss": [("species extinction", 0.8), ("biodiversity loss", 0.9)],
        "photosynthesis": [("oxygen production", 0.9), ("glucose production", 0.9)],
        "glucose production": [("cellular respiration", 0.8), ("energy storage", 0.7)],
        "cellular respiration": [("atp production", 0.9), ("carbon dioxide release", 0.8)],
        # Chemistry causal chains
        "temperature increase": [("reaction rate increase", 0.85), ("thermal expansion", 0.8),
                                  ("evaporation", 0.7), ("melting", 0.6)],
        "pressure increase": [("boiling point increase", 0.8), ("volume decrease", 0.85)],
        "catalyst": [("activation energy decrease", 0.9), ("reaction rate increase", 0.85)],
        "oxidation": [("electron loss", 0.95), ("corrosion", 0.7), ("combustion", 0.6)],
        "reduction": [("electron gain", 0.95)],
        # Physics causal chains
        "force": [("acceleration", 0.9), ("deformation", 0.6)],
        "acceleration": [("velocity change", 0.95)],
        "velocity change": [("displacement", 0.9)],
        "heat transfer": [("temperature change", 0.9), ("phase change", 0.7)],
        "electromagnetic radiation": [("photoelectric effect", 0.7), ("heating", 0.6)],
        # Medicine causal chains
        "smoking": [("lung cancer", 0.8), ("cardiovascular disease", 0.7),
                    ("emphysema", 0.75), ("chronic bronchitis", 0.7)],
        "obesity": [("diabetes", 0.7), ("heart disease", 0.65), ("hypertension", 0.7)],
        "hypertension": [("stroke", 0.7), ("heart failure", 0.65), ("kidney disease", 0.6)],
        "infection": [("inflammation", 0.85), ("fever", 0.8), ("immune response", 0.9)],
        "immune response": [("antibody production", 0.85), ("inflammation", 0.7)],
        "vitamin deficiency": [("scurvy", 0.7), ("rickets", 0.7), ("anemia", 0.6)],
        # Economics causal chains
        "interest rate increase": [("borrowing decrease", 0.8), ("inflation decrease", 0.7),
                                    ("investment decrease", 0.65)],
        "money supply increase": [("inflation", 0.8), ("interest rate decrease", 0.7)],
        "inflation": [("purchasing power decrease", 0.9), ("wage pressure", 0.6)],
        "unemployment": [("consumer spending decrease", 0.7), ("poverty increase", 0.6)],
        "tariff": [("import price increase", 0.85), ("trade decrease", 0.7)],
        # Environmental causal chains
        "greenhouse gas emission": [("global warming", 0.85), ("ocean acidification", 0.7)],
        "global warming": [("sea level rise", 0.8), ("glacier melting", 0.85),
                           ("extreme weather", 0.7)],
        "sea level rise": [("coastal flooding", 0.85), ("habitat loss", 0.7)],
    }

    def __init__(self):
        self._inferences = 0
        # Build reverse index for backward chaining
        self._effect_to_causes = {}
        for cause, effects in self._CAUSAL_KB.items():
            for effect, strength in effects:
                if effect not in self._effect_to_causes:
                    self._effect_to_causes[effect] = []
                self._effect_to_causes[effect].append((cause, strength))

    def forward_chain(self, cause: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """Forward chain from cause to all reachable effects.

        Returns list of dicts with 'effect', 'chain' (hop sequence),
        'cumulative_strength' (product of link strengths), and 'hops'.
        """
        results = []
        frontier = [(cause.lower(), [cause.lower()], 1.0)]
        visited = {cause.lower()}

        for _ in range(max_hops):
            new_frontier = []
            for current, chain, strength in frontier:
                effects = self._CAUSAL_KB.get(current, [])
                for effect, link_strength in effects:
                    if effect not in visited:
                        visited.add(effect)
                        new_chain = chain + [effect]
                        cum_strength = strength * link_strength
                        results.append({
                            "effect": effect,
                            "chain": new_chain,
                            "cumulative_strength": round(cum_strength, 4),
                            "hops": len(new_chain) - 1,
                        })
                        new_frontier.append((effect, new_chain, cum_strength))
            frontier = new_frontier
            if not frontier:
                break

        self._inferences += 1
        return sorted(results, key=lambda x: x["cumulative_strength"], reverse=True)

    def backward_chain(self, effect: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """Backward chain from effect to find root causes.

        Returns list of dicts with 'cause', 'chain', 'cumulative_strength', 'hops'.
        """
        results = []
        frontier = [(effect.lower(), [effect.lower()], 1.0)]
        visited = {effect.lower()}

        for _ in range(max_hops):
            new_frontier = []
            for current, chain, strength in frontier:
                causes = self._effect_to_causes.get(current, [])
                for cause, link_strength in causes:
                    if cause not in visited:
                        visited.add(cause)
                        new_chain = [cause] + chain
                        cum_strength = strength * link_strength
                        results.append({
                            "cause": cause,
                            "chain": new_chain,
                            "cumulative_strength": round(cum_strength, 4),
                            "hops": len(new_chain) - 1,
                        })
                        new_frontier.append((cause, new_chain, cum_strength))
            frontier = new_frontier
            if not frontier:
                break

        self._inferences += 1
        return sorted(results, key=lambda x: x["cumulative_strength"], reverse=True)

    def causal_link_strength(self, cause: str, effect: str) -> float:
        """Get the strength of a direct or multi-hop causal link.

        Returns: 0.0 (no link) to 1.0 (strong direct link).
        """
        cause_lower = cause.lower()
        effect_lower = effect.lower()

        # Direct link
        for eff, strength in self._CAUSAL_KB.get(cause_lower, []):
            if eff == effect_lower:
                return strength

        # Multi-hop: find through forward chaining
        chains = self.forward_chain(cause_lower, max_hops=3)
        for chain_entry in chains:
            if chain_entry["effect"] == effect_lower:
                return chain_entry["cumulative_strength"]

        return 0.0

    def score_causal_choice(self, question: str, choice: str) -> float:
        """Score a choice based on causal reasoning relevance.

        Extracts potential cause/effect mentions from question and choice,
        then scores the causal link strength.
        """
        q_lower = question.lower()
        c_lower = choice.lower()

        # Find causal concepts in question and choice
        all_concepts = set(self._CAUSAL_KB.keys())
        for effects in self._CAUSAL_KB.values():
            all_concepts.update(e for e, _ in effects)

        q_concepts = [c for c in all_concepts if c in q_lower]
        c_concepts = [c for c in all_concepts if c in c_lower]

        if not q_concepts or not c_concepts:
            return 0.0

        # Score best causal link
        best_strength = 0.0
        for qc in q_concepts:
            for cc in c_concepts:
                # Forward: question concept causes choice concept
                strength = self.causal_link_strength(qc, cc)
                best_strength = max(best_strength, strength)
                # Backward: choice concept causes question concept
                strength = self.causal_link_strength(cc, qc)
                best_strength = max(best_strength, strength)

        return best_strength * 0.2  # Scale for scoring pipeline


class PragmaticInferenceEngine:
    """Pragmatic inference engine for implicature and presupposition detection.

    Handles Gricean maxims, conversational implicatures, and presuppositions:
      - Scalar implicatures: "some" implies "not all"
      - Presuppositions: "stopped X-ing" presupposes "was X-ing"
      - Speech act classification: question, assertion, directive, commissive
      - Hedge detection: modal verbs and epistemic markers
      - Rhetorical markers: contrast, concession, elaboration

    Helps MCQSolver understand the pragmatic force of question wording,
    which is critical for tricky MMLU questions that rely on implicature.
    """

    # Scalar implicatures: "some" → "not all", "sometimes" → "not always"
    _SCALAR_PAIRS = {
        "some": ("not all", "all"),
        "sometimes": ("not always", "always"),
        "many": ("not most", "most"),
        "most": ("not all", "all"),
        "possible": ("not certain", "certain"),
        "may": ("not must", "must"),
        "might": ("not will", "will"),
        "can": ("not must", "must"),
        "good": ("not excellent", "excellent"),
        "warm": ("not hot", "hot"),
        "like": ("not love", "love"),
        "often": ("not always", "always"),
        "usually": ("not always", "always"),
        "probably": ("not certainly", "certainly"),
    }

    # Presupposition triggers
    _PRESUPPOSITION_TRIGGERS = {
        # Factive verbs: presuppose their complement is true
        "realize": "factual_complement",
        "know": "factual_complement",
        "discover": "factual_complement",
        "regret": "factual_complement",
        "notice": "factual_complement",
        "aware": "factual_complement",
        # Change-of-state: presuppose prior state
        "stop": "prior_activity",
        "stopped": "prior_activity",
        "start": "prior_non_activity",
        "started": "prior_non_activity",
        "begin": "prior_non_activity",
        "began": "prior_non_activity",
        "continue": "ongoing_activity",
        "continued": "ongoing_activity",
        "resume": "prior_activity",
        # Cleft constructions
        "it was": "existence",
        "it is": "existence",
    }

    # Speech act indicators
    _SPEECH_ACT_PATTERNS = [
        (r'^\s*(?:what|which|who|whom|where|when|why|how)\b', "question"),
        (r'\?\s*$', "question"),
        (r'^\s*(?:do|does|did|is|are|was|were|can|could|will|would|should|must|have|has)\b[^?]*\?', "yes_no_question"),
        (r'^\s*(?:please|kindly)\b', "directive"),
        (r'^\s*(?:you\s+should|you\s+must|you\s+need\s+to)', "directive"),
        (r'^\s*(?:I\s+think|I\s+believe|in\s+my\s+opinion)', "assertion_hedged"),
        (r'^\s*(?:it\s+is|there\s+is|there\s+are)\b', "assertion"),
        (r'^\s*(?:I\s+will|I\s+promise|I\s+guarantee)', "commissive"),
        (r'^\s*if\s+[^?!\n]{1,100}?\s+then\b', "conditional"),
    ]

    def __init__(self):
        self._analyses = 0
        self._compiled_speech_acts = [
            (re.compile(pat, re.IGNORECASE), act_type)
            for pat, act_type in self._SPEECH_ACT_PATTERNS
        ]

    def detect_implicatures(self, text: str) -> List[Dict[str, Any]]:
        """Detect scalar implicatures in the text.

        Returns list of dicts with 'trigger', 'implicature', 'negated_stronger'.
        """
        words = text.lower().split()
        implicatures = []

        for word in words:
            if word in self._SCALAR_PAIRS:
                neg_stronger, stronger = self._SCALAR_PAIRS[word]
                # Check if the stronger term is NOT present (confirming implicature)
                if stronger not in words:
                    implicatures.append({
                        "trigger": word,
                        "implicature": neg_stronger,
                        "negated_stronger": stronger,
                        "type": "scalar",
                    })

        return implicatures

    def detect_presuppositions(self, text: str) -> List[Dict[str, Any]]:
        """Detect presuppositions triggered by lexical items.

        Returns list of dicts with 'trigger', 'presupposition_type', 'context'.
        """
        text_lower = text.lower()
        presuppositions = []

        for trigger, ptype in self._PRESUPPOSITION_TRIGGERS.items():
            if trigger in text_lower:
                # Extract context around trigger
                idx = text_lower.index(trigger)
                context = text_lower[max(0, idx - 30):idx + len(trigger) + 30]
                presuppositions.append({
                    "trigger": trigger,
                    "presupposition_type": ptype,
                    "context": context.strip(),
                })

        return presuppositions

    def classify_speech_act(self, text: str) -> Dict[str, Any]:
        """Classify the speech act type of the text.

        Returns dict with 'type', 'confidence', and 'markers'.
        """
        for compiled, act_type in self._compiled_speech_acts:
            if compiled.search(text):
                return {
                    "type": act_type,
                    "confidence": 0.8,
                    "markers": [compiled.pattern],
                }

        return {"type": "assertion", "confidence": 0.5, "markers": []}

    def detect_hedges(self, text: str) -> Dict[str, Any]:
        """Detect hedging and epistemic markers.

        Returns dict with 'hedge_count', 'hedge_words', and 'certainty_level'.
        """
        hedge_words = []
        text_lower = text.lower()

        epistemic_markers = {
            "perhaps": 0.3, "maybe": 0.3, "possibly": 0.3,
            "probably": 0.5, "likely": 0.5, "presumably": 0.4,
            "apparently": 0.4, "seemingly": 0.3, "arguably": 0.4,
            "might": 0.3, "may": 0.4, "could": 0.4,
            "suggest": 0.4, "indicate": 0.5, "appear": 0.4,
            "seem": 0.4, "tend": 0.5, "generally": 0.5,
            "roughly": 0.4, "approximately": 0.5, "about": 0.5,
            "sort of": 0.3, "kind of": 0.3,
        }

        certainty_scores = []
        for marker, certainty in epistemic_markers.items():
            if marker in text_lower:
                hedge_words.append(marker)
                certainty_scores.append(certainty)

        avg_certainty = sum(certainty_scores) / len(certainty_scores) if certainty_scores else 0.8

        return {
            "hedge_count": len(hedge_words),
            "hedge_words": hedge_words,
            "certainty_level": round(avg_certainty, 3),
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Full pragmatic analysis of text.

        Returns combined implicatures, presuppositions, speech act, and hedges.
        """
        self._analyses += 1
        return {
            "implicatures": self.detect_implicatures(text),
            "presuppositions": self.detect_presuppositions(text),
            "speech_act": self.classify_speech_act(text),
            "hedges": self.detect_hedges(text),
        }

    def pragmatic_alignment(self, question: str, choice: str) -> float:
        """Score how well a choice aligns pragmatically with the question.

        Considers:
        - Hedge matching: hedged questions prefer hedged answers
        - Speech act congruence: questions expect assertions
        - Scalar implicature: respecting implied quantity constraints
        """
        q_analysis = self.analyze(question)
        c_analysis = self.analyze(choice)
        score = 0.0

        # Hedge alignment: similar certainty levels match
        q_certainty = q_analysis["hedges"]["certainty_level"]
        c_certainty = c_analysis["hedges"]["certainty_level"]
        certainty_diff = abs(q_certainty - c_certainty)
        if certainty_diff < 0.2:
            score += 0.05

        # Scalar implicature respect: if question implies "some but not all",
        # choices with "all" should be penalized
        for impl in q_analysis["implicatures"]:
            negated = impl["negated_stronger"]
            if negated.lower() in choice.lower():
                score -= 0.1  # Choice contradicts implicature

        return score


class ConceptNetLinker:
    """Commonsense knowledge linker using ConceptNet-style relations.

    Maintains a lightweight commonsense knowledge base with relations:
      - HasA: "bird HasA wings"
      - CapableOf: "bird CapableOf fly"
      - UsedFor: "hammer UsedFor nailing"
      - AtLocation: "fish AtLocation water"
      - HasProperty: "ice HasProperty cold"
      - PartOf: "wheel PartOf car"
      - Causes: "rain Causes wet"
      - DefinedAs: "bachelor DefinedAs unmarried man"
      - IsA: "dog IsA animal"

    Used to bridge knowledge gaps in MMLU questions where domain-specific
    facts are absent but commonsense reasoning can narrow down choices.
    """

    _RELATIONS = {
        "HasA": {
            "bird": ["wings", "feathers", "beak", "talons"],
            "fish": ["fins", "gills", "scales"],
            "tree": ["roots", "branches", "leaves", "bark", "trunk"],
            "human": ["brain", "heart", "lungs", "bones", "muscles"],
            "cell": ["membrane", "nucleus", "cytoplasm", "organelles"],
            "atom": ["protons", "neutrons", "electrons", "nucleus"],
            "car": ["engine", "wheels", "doors", "brakes", "steering"],
            "computer": ["processor", "memory", "keyboard", "screen", "storage"],
            "plant": ["roots", "stem", "leaves", "chloroplasts"],
            "earth": ["atmosphere", "oceans", "continents", "core", "mantle"],
            "molecule": ["atoms", "bonds", "electrons"],
            "government": ["legislature", "executive", "judiciary"],
        },
        "CapableOf": {
            "bird": ["fly", "sing", "build nests", "migrate"],
            "fish": ["swim", "breathe underwater"],
            "plant": ["photosynthesize", "grow", "reproduce"],
            "enzyme": ["catalyze reactions", "lower activation energy"],
            "acid": ["donate protons", "lower pH", "corrode metals"],
            "base": ["accept protons", "raise pH"],
            "conductor": ["conduct electricity", "conduct heat"],
            "insulator": ["resist electric current", "prevent heat transfer"],
            "lens": ["refract light", "focus light", "magnify"],
            "mirror": ["reflect light", "form images"],
        },
        "UsedFor": {
            "microscope": ["viewing small objects", "magnification", "biology"],
            "telescope": ["viewing distant objects", "astronomy"],
            "thermometer": ["measuring temperature"],
            "barometer": ["measuring air pressure"],
            "stethoscope": ["listening to heartbeat"],
            "voltmeter": ["measuring voltage"],
            "ammeter": ["measuring current"],
            "spectrometer": ["analyzing light spectra"],
            "centrifuge": ["separating mixtures by density"],
            "calorimeter": ["measuring heat"],
            "catalyst": ["speeding up reactions"],
            "vaccine": ["preventing disease", "immunity"],
            "antibiotic": ["killing bacteria", "treating infections"],
            "fertilizer": ["promoting plant growth"],
        },
        "AtLocation": {
            "fish": ["water", "ocean", "river", "lake"],
            "mitochondria": ["cell", "cytoplasm"],
            "chloroplast": ["plant cell"],
            "dna": ["nucleus", "chromosome"],
            "ribosome": ["cytoplasm", "endoplasmic reticulum"],
            "magma": ["mantle", "volcano"],
            "ozone": ["stratosphere"],
            "hemoglobin": ["red blood cell"],
        },
        "HasProperty": {
            "ice": ["cold", "solid", "crystalline", "transparent"],
            "water": ["liquid", "transparent", "universal solvent"],
            "steam": ["hot", "gaseous", "invisible"],
            "metal": ["conductive", "malleable", "ductile", "lustrous"],
            "diamond": ["hard", "transparent", "carbon"],
            "rubber": ["elastic", "insulating", "flexible"],
            "glass": ["brittle", "transparent", "amorphous"],
            "acid": ["sour", "corrosive", "low pH"],
            "base": ["bitter", "slippery", "high pH"],
            "noble gas": ["inert", "stable", "colorless"],
        },
        "Causes": {
            "heat": ["expansion", "melting", "evaporation"],
            "cold": ["contraction", "freezing", "condensation"],
            "gravity": ["falling", "weight", "tides", "orbits"],
            "friction": ["heat", "wear", "slowing"],
            "pressure": ["compression", "boiling point change"],
            "radiation": ["mutation", "cancer", "heating"],
            "erosion": ["weathering", "sediment transport"],
            "oxidation": ["rust", "fire", "corrosion"],
            "deforestation": ["habitat loss", "soil erosion"],
            "pollution": ["health problems", "climate change"],
        },
    }

    def __init__(self):
        self._lookups = 0
        # Build reverse index for efficient querying
        self._reverse_index = {}  # (relation, value) → [subjects]
        for relation, subjects in self._RELATIONS.items():
            for subject, values in subjects.items():
                for value in values:
                    key = (relation, value.lower())
                    if key not in self._reverse_index:
                        self._reverse_index[key] = []
                    self._reverse_index[key].append(subject)

    def query(self, subject: str, relation: str = None) -> Dict[str, List[str]]:
        """Query commonsense relations for a subject.

        Args:
            subject: The concept to query about
            relation: Optional specific relation (e.g., "HasA"). If None,
                      returns all relations.

        Returns: Dict mapping relation names to lists of values.
        """
        self._lookups += 1
        subject_lower = subject.lower()
        results = {}

        relations_to_check = {relation: self._RELATIONS[relation]} if relation and relation in self._RELATIONS \
            else self._RELATIONS

        for rel_name, subjects in relations_to_check.items():
            values = subjects.get(subject_lower, [])
            if values:
                results[rel_name] = values

        return results

    def reverse_query(self, value: str, relation: str) -> List[str]:
        """Reverse query: find subjects that have the given value for the given relation.

        E.g., reverse_query("wings", "HasA") → ["bird"]
        """
        self._lookups += 1
        key = (relation, value.lower())
        return self._reverse_index.get(key, [])

    def related(self, concept_a: str, concept_b: str) -> List[Dict[str, Any]]:
        """Find all commonsense relations between two concepts.

        Returns list of dicts with 'relation', 'direction' (a→b or b→a).
        """
        a_lower = concept_a.lower()
        b_lower = concept_b.lower()
        relations = []

        for rel_name, subjects in self._RELATIONS.items():
            # a→b
            if a_lower in subjects and b_lower in subjects[a_lower]:
                relations.append({"relation": rel_name, "direction": f"{concept_a}→{concept_b}"})
            # b→a
            if b_lower in subjects and a_lower in subjects[b_lower]:
                relations.append({"relation": rel_name, "direction": f"{concept_b}→{concept_a}"})

        return relations

    def score_choice_commonsense(self, question: str, choice: str) -> float:
        """Score a choice using commonsense knowledge relevance.

        Extracts concepts from question and choice, finds commonsense
        relations, and returns a relevance score.
        """
        q_lower = question.lower()
        c_lower = choice.lower()

        # Collect all known concepts
        all_subjects = set()
        for subjects in self._RELATIONS.values():
            all_subjects.update(subjects.keys())
        all_values = set()
        for subjects in self._RELATIONS.values():
            for vals in subjects.values():
                all_values.update(v.lower() for v in vals)

        # Find concepts mentioned in question and choice
        q_subjects = [s for s in all_subjects if s in q_lower]
        c_subjects = [s for s in all_subjects if s in c_lower]
        q_values = [v for v in all_values if v in q_lower]
        c_values = [v for v in all_values if v in c_lower]

        score = 0.0

        # Direct relation match: question concept → choice value (or vice versa)
        for qs in q_subjects:
            all_rels = self.query(qs)
            for rel, values in all_rels.items():
                for v in values:
                    if v.lower() in c_lower:
                        score += 0.15

        for cs in c_subjects:
            all_rels = self.query(cs)
            for rel, values in all_rels.items():
                for v in values:
                    if v.lower() in q_lower:
                        score += 0.12

        # Reverse relation: choice value is known for some subject in question
        for cv in c_values:
            for rel in self._RELATIONS:
                subjects = self.reverse_query(cv, rel)
                for s in subjects:
                    if s in q_lower:
                        score += 0.10

        return min(score, 0.4)


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 5: MCQ ANSWER SELECTOR — Confidence-weighted resolution
# ═══════════════════════════════════════════════════════════════════════════════
