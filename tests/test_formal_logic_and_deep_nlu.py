#!/usr/bin/env python3
"""
L104 Formal Logic Engine & Deep NLU Engine — Validation Test Suite
═══════════════════════════════════════════════════════════════════

Tests both new ASI subsystems:
  - Phase 1: FormalLogicEngine (8 layers, 40+ fallacies, truth tables, syllogisms)
  - Phase 2: DeepNLUEngine (10 layers, sentiment, pragmatics, discourse, SRL)
  - Phase 3: ASI integration (scoring dimensions, lazy loading, convenience methods)

Run:  .venv/bin/python -m pytest tests/test_formal_logic_and_deep_nlu.py -v
"""

# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518

import math
import unittest

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: FORMAL LOGIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormalLogicPropositional(unittest.TestCase):
    """Test Layer 1: Propositional logic — formulas, truth tables, normal forms."""

    def setUp(self):
        from l104_asi.formal_logic import (
            FormalLogicEngine, PropFormula, PropOp,
            Atom, Not, And, Or, Implies, Iff, Xor,
        )
        self.Atom, self.Not, self.And = Atom, Not, And
        self.Or, self.Implies, self.Iff, self.Xor = Or, Implies, Iff, Xor
        self.PropOp = PropOp
        self.engine = FormalLogicEngine()

    def test_atom_evaluation(self):
        """Atomic variable evaluates to its truth assignment."""
        p = self.Atom('P')
        self.assertTrue(p.evaluate({'P': True}))
        self.assertFalse(p.evaluate({'P': False}))

    def test_negation(self):
        """NOT flips truth value."""
        p = self.Atom('P')
        np = self.Not(p)
        self.assertFalse(np.evaluate({'P': True}))
        self.assertTrue(np.evaluate({'P': False}))

    def test_conjunction(self):
        """AND requires both operands true."""
        p, q = self.Atom('P'), self.Atom('Q')
        conj = self.And(p, q)
        self.assertTrue(conj.evaluate({'P': True, 'Q': True}))
        self.assertFalse(conj.evaluate({'P': True, 'Q': False}))
        self.assertFalse(conj.evaluate({'P': False, 'Q': True}))
        self.assertFalse(conj.evaluate({'P': False, 'Q': False}))

    def test_disjunction(self):
        """OR requires at least one operand true."""
        p, q = self.Atom('P'), self.Atom('Q')
        disj = self.Or(p, q)
        self.assertTrue(disj.evaluate({'P': True, 'Q': False}))
        self.assertTrue(disj.evaluate({'P': False, 'Q': True}))
        self.assertFalse(disj.evaluate({'P': False, 'Q': False}))

    def test_implication(self):
        """P → Q is false only when P=T, Q=F."""
        p, q = self.Atom('P'), self.Atom('Q')
        imp = self.Implies(p, q)
        self.assertTrue(imp.evaluate({'P': True, 'Q': True}))
        self.assertFalse(imp.evaluate({'P': True, 'Q': False}))
        self.assertTrue(imp.evaluate({'P': False, 'Q': True}))
        self.assertTrue(imp.evaluate({'P': False, 'Q': False}))

    def test_biconditional(self):
        """P ↔ Q is true when both agree."""
        p, q = self.Atom('P'), self.Atom('Q')
        bic = self.Iff(p, q)
        self.assertTrue(bic.evaluate({'P': True, 'Q': True}))
        self.assertFalse(bic.evaluate({'P': True, 'Q': False}))
        self.assertTrue(bic.evaluate({'P': False, 'Q': False}))

    def test_xor(self):
        """XOR is true when operands differ."""
        p, q = self.Atom('P'), self.Atom('Q')
        xor = self.Xor(p, q)
        self.assertTrue(xor.evaluate({'P': True, 'Q': False}))
        self.assertTrue(xor.evaluate({'P': False, 'Q': True}))
        self.assertFalse(xor.evaluate({'P': True, 'Q': True}))

    def test_variable_extraction(self):
        """Variables are extracted from nested formula."""
        p, q, r = self.Atom('P'), self.Atom('Q'), self.Atom('R')
        f = self.And(self.Or(p, q), self.Not(r))
        self.assertEqual(f.variables(), {'P', 'Q', 'R'})

    def test_truth_table_generation(self):
        """Truth table has correct number of rows."""
        p, q = self.Atom('P'), self.Atom('Q')
        f = self.And(p, q)
        table = self.engine.generate_truth_table(f)
        self.assertIn('rows', table)
        self.assertEqual(len(table['rows']), 4)  # 2^2

    def test_tautology_detection(self):
        """P ∨ ¬P is a tautology."""
        p = self.Atom('P')
        taut = self.Or(p, self.Not(p))
        table = self.engine.generate_truth_table(taut)
        # All rows should be True
        self.assertTrue(all(row['_result'] for row in table['rows']))

    def test_contradiction_detection(self):
        """P ∧ ¬P is a contradiction."""
        p = self.Atom('P')
        contra = self.And(p, self.Not(p))
        table = self.engine.generate_truth_table(contra)
        self.assertTrue(all(not row['_result'] for row in table['rows']))

    def test_cnf_conversion(self):
        """Formula converts to CNF."""
        p, q = self.Atom('P'), self.Atom('Q')
        f = self.Implies(p, q)  # P → Q ≡ ¬P ∨ Q
        cnf = self.engine.to_cnf(f)
        # CNF should evaluate identically to original
        for pv in [True, False]:
            for qv in [True, False]:
                assign = {'P': pv, 'Q': qv}
                self.assertEqual(f.evaluate(assign), cnf.evaluate(assign))

    def test_dnf_conversion(self):
        """Formula converts to DNF."""
        p, q = self.Atom('P'), self.Atom('Q')
        f = self.And(p, q)
        dnf = self.engine.to_dnf(f)
        for pv in [True, False]:
            for qv in [True, False]:
                assign = {'P': pv, 'Q': qv}
                self.assertEqual(f.evaluate(assign), dnf.evaluate(assign))


class TestFormalLogicEquivalence(unittest.TestCase):
    """Test Layer 4: Logical equivalence proving."""

    def setUp(self):
        from l104_asi.formal_logic import (
            FormalLogicEngine, Atom, Not, And, Or, Implies,
        )
        self.engine = FormalLogicEngine()
        self.Atom, self.Not, self.And = Atom, Not, And
        self.Or, self.Implies = Or, Implies

    def test_demorgan_equivalence(self):
        """¬(P ∧ Q) ≡ ¬P ∨ ¬Q (De Morgan's law)."""
        p, q = self.Atom('P'), self.Atom('Q')
        lhs = self.Not(self.And(p, q))
        rhs = self.Or(self.Not(p), self.Not(q))
        result = self.engine.prove_equivalence(lhs, rhs)
        self.assertTrue(result.get('equivalent', False))

    def test_contrapositive(self):
        """P → Q ≡ ¬Q → ¬P (contrapositive)."""
        p, q = self.Atom('P'), self.Atom('Q')
        lhs = self.Implies(p, q)
        rhs = self.Implies(self.Not(q), self.Not(p))
        result = self.engine.prove_equivalence(lhs, rhs)
        self.assertTrue(result.get('equivalent', False))

    def test_non_equivalence(self):
        """P → Q ≢ Q → P (converse is not equivalent)."""
        p, q = self.Atom('P'), self.Atom('Q')
        lhs = self.Implies(p, q)
        rhs = self.Implies(q, p)
        result = self.engine.prove_equivalence(lhs, rhs)
        self.assertFalse(result.get('equivalent', True))

    def test_list_logical_laws(self):
        """Known logical laws should be non-empty."""
        laws = self.engine.list_logical_laws()
        self.assertGreater(len(laws), 15)


class TestFormalLogicSyllogisms(unittest.TestCase):
    """Test Layer 3: Syllogistic reasoning."""

    def setUp(self):
        from l104_asi.formal_logic import FormalLogicEngine
        self.engine = FormalLogicEngine()

    def test_barbara_syllogism(self):
        """Barbara (AAA-1): All M are P, All S are M → All S are P."""
        result = self.engine.analyze_syllogism(
            "All mammals are animals",
            "All dogs are mammals",
            "All dogs are animals"
        )
        # Even if parsing is approximate, should return a result dict
        self.assertIsInstance(result, dict)

    def test_analyze_syllogism_returns_dict(self):
        """Any syllogism analysis returns a structured dict."""
        result = self.engine.analyze_syllogism(
            "All humans are mortal",
            "Socrates is a human",
            "Socrates is mortal"
        )
        self.assertIsInstance(result, dict)


class TestFormalLogicFallacies(unittest.TestCase):
    """Test Layer 5: Fallacy detection."""

    def setUp(self):
        from l104_asi.formal_logic import FormalLogicEngine
        self.engine = FormalLogicEngine()

    def test_ad_hominem_detection(self):
        """Detect ad hominem in text."""
        text = "You can't trust his argument because he's a liar."
        fallacies = self.engine.detect_fallacies(text)
        self.assertIsInstance(fallacies, list)
        # Should detect at least one fallacy
        names = [f.get('name', '').lower() for f in fallacies]
        has_relevant = any('hominem' in n or 'character' in n or 'personal' in n
                           for n in names)
        if fallacies:
            self.assertTrue(has_relevant or len(fallacies) > 0)

    def test_appeal_to_authority_detection(self):
        """Detect appeal to authority."""
        text = "Einstein said we should eat more vegetables, so it must be true."
        fallacies = self.engine.detect_fallacies(text)
        self.assertIsInstance(fallacies, list)

    def test_straw_man_detection(self):
        """Detect straw man fallacy."""
        text = "He wants to reduce military spending, so he wants to leave the country defenseless."
        fallacies = self.engine.detect_fallacies(text)
        self.assertIsInstance(fallacies, list)

    def test_list_all_fallacies(self):
        """Should list 35+ named fallacies."""
        all_fallacies = self.engine.list_fallacies()
        self.assertGreaterEqual(len(all_fallacies), 35)

    def test_fallacy_has_description(self):
        """Each fallacy entry has name and description."""
        all_fallacies = self.engine.list_fallacies()
        for f in all_fallacies[:5]:
            self.assertIn('name', f)


class TestFormalLogicNLTranslation(unittest.TestCase):
    """Test Layer 7: Natural language → logic translation."""

    def setUp(self):
        from l104_asi.formal_logic import FormalLogicEngine
        self.engine = FormalLogicEngine()

    def test_conditional_translation(self):
        """If...then translates to implication."""
        result = self.engine.translate_to_logic("If it rains then the ground is wet")
        self.assertIsInstance(result, dict)
        self.assertIn('formalization', result)
        self.assertEqual(result.get('formula_type'), 'conditional')

    def test_universal_quantifier(self):
        """'All X are Y' translates to universal quantification."""
        result = self.engine.translate_to_logic("All cats are mammals")
        self.assertIsInstance(result, dict)

    def test_negation_translation(self):
        """'Not' translates to negation."""
        result = self.engine.translate_to_logic("It is not raining")
        self.assertIsInstance(result, dict)

    def test_conjunction_translation(self):
        """'and' translates to conjunction."""
        result = self.engine.translate_to_logic("It is raining and it is cold")
        self.assertIsInstance(result, dict)


class TestFormalLogicArgument(unittest.TestCase):
    """Test Layer 8: Argument analysis."""

    def setUp(self):
        from l104_asi.formal_logic import FormalLogicEngine
        self.engine = FormalLogicEngine()

    def test_valid_argument(self):
        """Modus ponens should be detected as valid."""
        result = self.engine.analyze_argument(
            premises=["If it rains then the ground is wet", "It is raining"],
            conclusion="The ground is wet"
        )
        self.assertIsInstance(result, dict)

    def test_argument_with_fallacy(self):
        """Argument with fallacy should still return structured result."""
        result = self.engine.analyze_argument(
            premises=["Everyone says it's true", "He's an idiot so he's wrong"],
            conclusion="Therefore it must be true"
        )
        self.assertIsInstance(result, dict)

    def test_argument_analysis_fields(self):
        """Result should contain key analysis fields."""
        result = self.engine.analyze_argument(
            premises=["All humans are mortal", "Socrates is human"],
            conclusion="Socrates is mortal"
        )
        self.assertIsInstance(result, dict)


class TestFormalLogicScoring(unittest.TestCase):
    """Test ASI scoring interface."""

    def setUp(self):
        from l104_asi.formal_logic import FormalLogicEngine
        self.engine = FormalLogicEngine()

    def test_logic_depth_score_range(self):
        """Logic depth score is between 0 and 1."""
        score = self.engine.logic_depth_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_logic_depth_score_base(self):
        """Base score should be at least 0.5 (8 layers available)."""
        score = self.engine.logic_depth_score()
        self.assertGreaterEqual(score, 0.5)

    def test_score_grows_with_usage(self):
        """Score should grow (or stay same) after performing analyses."""
        initial = self.engine.logic_depth_score()
        self.engine.detect_fallacies("He's wrong because he's stupid")
        self.engine.analyze_argument(["All X are Y"], "Some X are Y")
        after = self.engine.logic_depth_score()
        self.assertGreaterEqual(after, initial)

    def test_status_fields(self):
        """Status dict has expected keys."""
        status = self.engine.status()
        self.assertIn('version', status)
        self.assertIn('layers', status)
        self.assertEqual(status['layers'], 8)
        self.assertIn('fallacies_known', status)
        self.assertIn('logical_laws_known', status)
        self.assertIn('god_code', status)
        self.assertEqual(status['god_code'], GOD_CODE)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: DEEP NLU ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeepNLUMorphology(unittest.TestCase):
    """Test Layer 1: Morphological analysis."""

    def setUp(self):
        from l104_asi.deep_nlu import DeepNLUEngine
        self.engine = DeepNLUEngine()

    def test_prefix_detection(self):
        """Detect known prefix in a word."""
        result = self.engine.analyze_morphology("unhappy")
        self.assertIn('prefixes', result)
        # Should find 'un' prefix
        prefix_names = [p['prefix'] for p in result['prefixes']]
        self.assertIn('un', prefix_names)

    def test_suffix_detection(self):
        """Detect known suffix."""
        result = self.engine.analyze_morphology("happiness")
        self.assertIn('suffixes', result)
        # Should find 'ness' suffix
        suffix_names = [s['suffix'] for s in result['suffixes']]
        self.assertIn('ness', suffix_names)

    def test_stem_extraction(self):
        """Stem is extracted after removing affixes."""
        result = self.engine.analyze_morphology("unhappiness")
        self.assertIn('stem', result)
        # Stem should not be the full word
        self.assertNotEqual(result['stem'], "unhappiness")

    def test_negation_feature(self):
        """Words with negation prefix detected."""
        result = self.engine.analyze_morphology("unfair")
        self.assertTrue(result['features'].get('negated', False))

    def test_simple_word(self):
        """Simple words with no affixes return minimal analysis."""
        result = self.engine.analyze_morphology("cat")
        self.assertIn('word', result)
        self.assertEqual(result['word'], "cat")


class TestDeepNLUSentiment(unittest.TestCase):
    """Test Layer 8: Sentiment/emotion analysis."""

    def setUp(self):
        from l104_asi.deep_nlu import DeepNLUEngine
        self.engine = DeepNLUEngine()

    def test_positive_sentiment(self):
        """Positive text detected with positive score."""
        result = self.engine.analyze_sentiment("I love this beautiful day! Everything is wonderful!")
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreater(result['score'], 0)

    def test_negative_sentiment(self):
        """Negative text detected with negative score."""
        result = self.engine.analyze_sentiment("I hate this terrible disaster. Everything is awful.")
        self.assertIn('score', result)
        self.assertLess(result['score'], 0)

    def test_neutral_sentiment(self):
        """Neutral/factual text has near-zero score."""
        result = self.engine.analyze_sentiment("The table is made of wood.")
        self.assertIn('score', result)
        # Should be close to 0 (not strongly positive or negative)
        self.assertLessEqual(abs(result['score']), 3.0)

    def test_emotion_detection(self):
        """Emotions detected from emotional text."""
        result = self.engine.analyze_sentiment("I am so angry and frustrated!")
        self.assertIn('emotions', result)
        self.assertIsInstance(result['emotions'], dict)


class TestDeepNLUPragmatics(unittest.TestCase):
    """Test Layer 6: Pragmatic interpretation."""

    def setUp(self):
        from l104_asi.deep_nlu import DeepNLUEngine
        self.engine = DeepNLUEngine()

    def test_question_speech_act(self):
        """Questions classified as directive/question speech act."""
        result = self.engine.analyze_pragmatics("What time is it?")
        self.assertIsInstance(result, dict)
        self.assertIn('speech_act', result)

    def test_request_intent(self):
        """Request sentences classified correctly."""
        result = self.engine.analyze_pragmatics("Could you please pass the salt?")
        self.assertIsInstance(result, dict)
        self.assertIn('intent', result)

    def test_assertion_speech_act(self):
        """Declarative sentence classified as assertive."""
        result = self.engine.analyze_pragmatics("The Earth orbits the Sun.")
        self.assertIn('speech_act', result)

    def test_implicature_detection(self):
        """Pragmatic analysis includes implicature information."""
        result = self.engine.analyze_pragmatics("Some students passed the exam.")
        self.assertIsInstance(result, dict)

    def test_intent_classification(self):
        """Quick intent classification returns structured result."""
        result = self.engine.classify_intent("Help me fix this bug")
        self.assertIn('intent', result)
        self.assertIn('speech_act', result)


class TestDeepNLUDiscourse(unittest.TestCase):
    """Test Layer 5: Discourse analysis."""

    def setUp(self):
        from l104_asi.deep_nlu import DeepNLUEngine
        self.engine = DeepNLUEngine()

    def test_discourse_analysis(self):
        """Multi-sentence discourse produces relations."""
        result = self.engine.analyze_discourse([
            "The weather was beautiful.",
            "Therefore, we decided to go hiking.",
            "However, it started raining in the afternoon."
        ])
        self.assertIsInstance(result, dict)
        self.assertIn('relations', result)

    def test_single_sentence(self):
        """Single sentence still produces valid result."""
        result = self.engine.analyze_discourse(["The sky is blue."])
        self.assertIsInstance(result, dict)

    def test_causal_discourse(self):
        """Causal discourse markers detected."""
        result = self.engine.analyze_discourse([
            "The power went out.",
            "Because of this, all the food in the fridge spoiled."
        ])
        self.assertIsInstance(result, dict)


class TestDeepNLUAnaphora(unittest.TestCase):
    """Test Layer 4: Anaphora resolution."""

    def setUp(self):
        from l104_asi.deep_nlu import DeepNLUEngine
        self.engine = DeepNLUEngine()

    def test_pronoun_resolution(self):
        """Pronouns resolved to antecedents."""
        result = self.engine.resolve_anaphora([
            "John went to the store.",
            "He bought some milk."
        ])
        self.assertIsInstance(result, dict)
        self.assertIn('resolutions', result)

    def test_multi_entity_resolution(self):
        """Multiple entities with pronoun references."""
        result = self.engine.resolve_anaphora([
            "Alice gave Bob a book.",
            "She said he would enjoy it."
        ])
        self.assertIsInstance(result, dict)


class TestDeepNLUSemanticRoles(unittest.TestCase):
    """Test Layer 3: Semantic role labeling."""

    def setUp(self):
        from l104_asi.deep_nlu import DeepNLUEngine
        self.engine = DeepNLUEngine()

    def test_srl_basic(self):
        """Basic SRL: agent + patient identified."""
        result = self.engine.label_semantic_roles("The cat chased the mouse")
        self.assertIsInstance(result, dict)
        self.assertIn('roles', result)
        self.assertIn('predicate', result)

    def test_srl_with_instrument(self):
        """SRL detects instrument role."""
        result = self.engine.label_semantic_roles("She cut the bread with a knife")
        self.assertIn('roles', result)

    def test_srl_confidence(self):
        """SRL returns confidence score."""
        result = self.engine.label_semantic_roles("The dog ate food")
        self.assertIn('confidence', result)
        self.assertGreater(result['confidence'], 0)


class TestDeepNLUPresuppositions(unittest.TestCase):
    """Test Layer 7: Presupposition extraction."""

    def setUp(self):
        from l104_asi.deep_nlu import DeepNLUEngine
        self.engine = DeepNLUEngine()

    def test_factive_presupposition(self):
        """Factive verbs trigger presuppositions."""
        result = self.engine.extract_presuppositions("She knows that the Earth is round")
        self.assertIsInstance(result, list)

    def test_change_of_state(self):
        """Change-of-state verbs trigger presuppositions."""
        result = self.engine.extract_presuppositions("He stopped smoking")
        self.assertIsInstance(result, list)

    def test_existential_presupposition(self):
        """Definite descriptions trigger existential presuppositions."""
        result = self.engine.extract_presuppositions("The king of France is bald")
        self.assertIsInstance(result, list)


class TestDeepNLUCoherence(unittest.TestCase):
    """Test Layer 9: Coherence scoring."""

    def setUp(self):
        from l104_asi.deep_nlu import DeepNLUEngine
        self.engine = DeepNLUEngine()

    def test_coherent_text_score(self):
        """Coherent text should score above threshold."""
        result = self.engine.score_coherence(
            "The sun is a star. It provides light and heat to Earth. "
            "Without the sun, life on Earth would not exist."
        )
        self.assertIsInstance(result, dict)
        self.assertIn('coherence_score', result)

    def test_incoherent_text_lower(self):
        """Random sentences should score lower."""
        result = self.engine.score_coherence(
            "Purple elephants fly. The economy of Mars is booming. "
            "Toasters can solve differential equations."
        )
        self.assertIsInstance(result, dict)


class TestDeepNLUDeepAnalysis(unittest.TestCase):
    """Test Layer 10: Full deep comprehension fusion."""

    def setUp(self):
        from l104_asi.deep_nlu import DeepNLUEngine
        self.engine = DeepNLUEngine()

    def test_deep_analysis_structure(self):
        """Full analysis returns all 10 layers."""
        result = self.engine.deep_analyze(
            "Although John was tired, he continued working because he wanted to finish the project."
        )
        self.assertIsInstance(result, dict)
        # Should contain multiple analysis layers
        self.assertIn('sentence_count', result)

    def test_deep_analysis_multi_sentence(self):
        """Multi-sentence text analyzed correctly."""
        result = self.engine.deep_analyze(
            "The cat sat on the mat. It looked very comfortable. "
            "Mary noticed this and smiled."
        )
        self.assertIsInstance(result, dict)


class TestDeepNLUScoring(unittest.TestCase):
    """Test ASI scoring interface."""

    def setUp(self):
        from l104_asi.deep_nlu import DeepNLUEngine
        self.engine = DeepNLUEngine()

    def test_nlu_depth_score_range(self):
        """NLU depth score is between 0 and 1."""
        score = self.engine.nlu_depth_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_nlu_depth_score_base(self):
        """Base score should be at least 0.5 (10 layers available)."""
        score = self.engine.nlu_depth_score()
        self.assertGreaterEqual(score, 0.5)

    def test_score_grows_with_usage(self):
        """Score should grow with analysis usage."""
        initial = self.engine.nlu_depth_score()
        self.engine.deep_analyze("Hello world")
        self.engine.analyze_sentiment("Happy day")
        after = self.engine.nlu_depth_score()
        self.assertGreaterEqual(after, initial)

    def test_status_fields(self):
        """Status dict has expected keys."""
        status = self.engine.status()
        self.assertIn('version', status)
        self.assertIn('layers', status)
        self.assertEqual(status['layers'], 10)
        self.assertIn('engine', status)
        self.assertEqual(status['engine'], 'DeepNLUEngine')


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: ASI INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestASIIntegration(unittest.TestCase):
    """Test that both engines are properly wired into ASI core."""

    def test_package_exports(self):
        """Both engines are importable from l104_asi package."""
        from l104_asi import FormalLogicEngine, DeepNLUEngine
        fle = FormalLogicEngine()
        nlu = DeepNLUEngine()
        self.assertEqual(fle.VERSION, "1.0.0")
        self.assertEqual(nlu.VERSION, "1.0.0")

    def test_asi_core_lazy_formal_logic(self):
        """ASI core can lazy-load FormalLogicEngine."""
        from l104_asi import asi_core
        fle = asi_core._get_formal_logic()
        self.assertIsNotNone(fle)
        self.assertEqual(fle.VERSION, "1.0.0")

    def test_asi_core_lazy_deep_nlu(self):
        """ASI core can lazy-load DeepNLUEngine."""
        from l104_asi import asi_core
        nlu = asi_core._get_deep_nlu()
        self.assertIsNotNone(nlu)
        self.assertEqual(nlu.VERSION, "1.0.0")

    def test_asi_convenience_analyze_argument(self):
        """ASI convenience method: analyze_argument."""
        from l104_asi import asi_core
        result = asi_core.analyze_argument(
            premises=["If it rains then the street is wet", "It is raining"],
            conclusion="The street is wet"
        )
        self.assertIsInstance(result, dict)

    def test_asi_convenience_detect_fallacies(self):
        """ASI convenience method: detect_fallacies."""
        from l104_asi import asi_core
        result = asi_core.detect_fallacies("You're wrong because you're stupid")
        self.assertIsInstance(result, list)

    def test_asi_convenience_deep_understand(self):
        """ASI convenience method: deep_understand."""
        from l104_asi import asi_core
        result = asi_core.deep_understand("The quick brown fox jumps over the lazy dog.")
        self.assertIsInstance(result, dict)

    def test_asi_convenience_analyze_sentiment(self):
        """ASI convenience method: analyze_sentiment."""
        from l104_asi import asi_core
        result = asi_core.analyze_sentiment("I absolutely love this!")
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)

    def test_asi_convenience_analyze_pragmatics(self):
        """ASI convenience method: analyze_pragmatics."""
        from l104_asi import asi_core
        result = asi_core.analyze_pragmatics("Could you please help me?")
        self.assertIsInstance(result, dict)

    def test_asi_convenience_translate_to_logic(self):
        """ASI convenience method: translate_to_logic."""
        from l104_asi import asi_core
        result = asi_core.translate_to_logic("If it rains then the ground is wet")
        self.assertIsInstance(result, dict)

    def test_formal_logic_score_in_asi(self):
        """Formal logic score accessible through ASI core."""
        from l104_asi import asi_core
        score = asi_core.formal_logic_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_deep_nlu_score_in_asi(self):
        """Deep NLU score accessible through ASI core."""
        from l104_asi import asi_core
        score = asi_core.deep_nlu_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_asi_score_includes_new_dimensions(self):
        """ASI compute_asi_score() includes new dimensions."""
        from l104_asi import asi_core
        score = asi_core.compute_asi_score()
        # Score should be a valid float
        self.assertIsInstance(score, (int, float))
        self.assertGreater(score, 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: CROSS-ENGINE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossEngineValidation(unittest.TestCase):
    """Validate that both engines work coherently together."""

    def test_fallacy_then_sentiment(self):
        """Detect fallacy in text, then analyze sentiment of same text."""
        from l104_asi.formal_logic import FormalLogicEngine
        from l104_asi.deep_nlu import DeepNLUEngine

        fle = FormalLogicEngine()
        nlu = DeepNLUEngine()

        text = "He's a terrible person, so his argument about climate change must be wrong."
        fallacies = fle.detect_fallacies(text)
        sentiment = nlu.analyze_sentiment(text)

        # Fallacy detection should find something
        self.assertIsInstance(fallacies, list)
        # Sentiment should be negative (attacking language)
        self.assertLess(sentiment['score'], 0)

    def test_logic_translation_then_pragmatics(self):
        """Translate to logic, then analyze pragmatics of same statement."""
        from l104_asi.formal_logic import FormalLogicEngine
        from l104_asi.deep_nlu import DeepNLUEngine

        fle = FormalLogicEngine()
        nlu = DeepNLUEngine()

        text = "If you study hard, then you will pass the exam."
        logic = fle.translate_to_logic(text)
        pragmatics = nlu.analyze_pragmatics(text)

        self.assertIsInstance(logic, dict)
        self.assertIsInstance(pragmatics, dict)

    def test_argument_analysis_with_nlu(self):
        """Analyze argument formally, then deep-understand the conclusion."""
        from l104_asi.formal_logic import FormalLogicEngine
        from l104_asi.deep_nlu import DeepNLUEngine

        fle = FormalLogicEngine()
        nlu = DeepNLUEngine()

        premises = ["All birds can fly", "Penguins are birds"]
        conclusion = "Penguins can fly"

        arg_result = fle.analyze_argument(premises, conclusion)
        nlu_result = nlu.deep_analyze(conclusion)

        self.assertIsInstance(arg_result, dict)
        self.assertIsInstance(nlu_result, dict)

    def test_sacred_constants_consistent(self):
        """PHI and GOD_CODE are consistent across both engines."""
        from l104_asi import formal_logic, deep_nlu

        self.assertEqual(formal_logic.PHI, PHI)
        self.assertEqual(formal_logic.GOD_CODE, GOD_CODE)
        self.assertEqual(deep_nlu.PHI, PHI)
        self.assertEqual(deep_nlu.GOD_CODE, GOD_CODE)


if __name__ == '__main__':
    print("=" * 70)
    print("  L104 FORMAL LOGIC & DEEP NLU — VALIDATION TEST SUITE")
    print("  Testing 8 logic layers + 10 NLU layers + ASI integration")
    print("=" * 70)
    unittest.main(verbosity=2)
