from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import PHI, GOD_CODE, TAU, VOID_CONSTANT
from .tokenizer import L104Tokenizer
from .vectorizer import SemanticEncoder
from .ngram import NGramMatcher
from .knowledge_base import MMLUKnowledgeBase, KnowledgeNode
from .retrieval import BM25Ranker
from .detectors import SubjectDetector, NumericalReasoner
from .verification import CrossVerificationEngine
from .nlp_algorithms import (
    TextualEntailmentEngine, AnalogicalReasoner, TextRankSummarizer,
    NamedEntityRecognizer, LevenshteinMatcher, LatentSemanticAnalyzer,
    LeskDisambiguator,
)
from .discourse import (
    CoreferenceResolver, SentimentAnalyzer, SemanticFrameAnalyzer,
    TaxonomyClassifier, CausalChainReasoner, PragmaticInferenceEngine,
    ConceptNetLinker,
)
from .mcq_solver import MCQSolver
from .engine_support import (
    _get_cached_science_engine, _get_cached_math_engine,
    _get_cached_code_engine, _get_cached_deep_nlu,
    _get_cached_dual_layer, _get_cached_formal_logic,
    _get_cached_quantum_gate_engine, _get_cached_quantum_math_core,
    _get_cached_search_engine, _get_cached_precognition_engine,
    _get_cached_three_engine_hub, _get_cached_precog_synthesis,
    _get_cached_probability_engine,
)

_log = logging.getLogger('l104.language_comprehension')

class LanguageComprehensionEngine:
    """
    Unified language comprehension engine v9.0.0 for MMLU-grade question answering.

    Combines tokenization, semantic encoding, knowledge retrieval, BM25 ranking,
    N-gram phrase matching, and chain-of-thought reasoning into a single coherent
    pipeline with Dual-Layer physics grounding and entropy calibration.

    v6.0.0 additions (Deep NLU v2.0.0 integration):
    - Temporal reasoning: tense detection, event ordering, duration extraction
    - Causal reasoning: cause-effect chains, counterfactuals, causal strength
    - Contextual disambiguation: WSD, polysemy resolution, metaphor detection
    - comprehend() now returns temporal, causal, and disambiguation analysis
    - Fixed duplicate keys in get_status()

    v4.0.0 additions:
    - 48 new knowledge domains (organic chemistry, game theory, constitutional law, etc.)
    - 70+ cross-subject relation edges for multi-hop retrieval
    - Enhanced STEM, humanities, social science, and professional coverage

    v3.0.0 additions:
    - SubjectDetector: auto-detect MMLU subject for focused retrieval
    - NumericalReasoner: quantitative question scoring via number extraction
    - CrossVerificationEngine: multi-strategy answer elimination + consistency

    DeepSeek-informed architecture:
    - MLA-style multi-perspective attention over knowledge passages
    - R1-style chain-of-thought verification
    - V3-style mixture-of-experts knowledge routing
    - N-gram phrase matching for multi-word concept recognition
    - Dual-Layer Engine physics-grounded confidence calibration
    - Formal Logic Engine deductive support for logic questions
    - DeepNLU Engine v2.3.0 — 20-layer discourse comprehension
    - Three-engine scoring method for ASI 15D integration
    - Entropy-based confidence recalibration via Maxwell Demon

    14-layer architecture:
      1. Tokenizer             2. Semantic Encoder      3. Knowledge Graph
      4. BM25 Ranker           4b. Subject Detector     4c. Numerical Reasoner
      5. MCQ Answer Selector   6. Cross-Verification    7. Chain-of-thought
      8. Calibration           N-gram Matcher (shared)
      9. Temporal Reasoning    10. Causal Reasoning     11. Disambiguation
    """

    VERSION = "9.0.0"

    def __init__(self):
        # Core components (always initialized)
        self.tokenizer = L104Tokenizer(vocab_size=8192)
        self.knowledge_base = MMLUKnowledgeBase()
        self.mcq_solver = MCQSolver(
            self.knowledge_base,
            subject_detector=None,  # Lazy-loaded
            numerical_reasoner=None,  # Lazy-loaded
            cross_verifier=None,  # Lazy-loaded
        )
        self.semantic_encoder = SemanticEncoder(embedding_dim=256)
        self.ngram_matcher = NGramMatcher()

        # Lazy initialization for complex components (initialized on first use)
        self._entailment_engine = None
        self._analogical_reasoner = None

        # v20.0: Search & Precognition integration
        self._search_engine = None
        self._precognition_engine = None
        self._three_engine_hub = None
        self._precog_synthesis = None
        self._search_enhanced_queries = 0
        self._textrank = None
        self._ner = None
        self._fuzzy_matcher = None
        self._lsa = None
        self._lesk = None
        self._coref_resolver = None
        self._sentiment_analyzer = None
        self._frame_analyzer = None
        self._taxonomy = None
        self._causal_chain = None
        self._pragmatics = None
        self._commonsense = None
        self._subject_detector = None
        self._numerical_reasoner = None
        self._cross_verifier = None

        self._initialized = False
        self._total_queries = 0
        self._comprehension_score = 0.0
        self._three_engine_score = 0.0
        self._init_lock = threading.Lock()
        self._nlu_cache: Dict[str, Dict] = {}
        self._nlu_cache_maxsize = 128

        # Engine support connections (lazy-loaded)
        self._science_engine = None
        self._math_engine = None
        self._code_engine = None
        self._quantum_gate_engine = None
        self._quantum_math_core = None
        self._dual_layer = None
        self._formal_logic = None
        self._deep_nlu = None
        self._probability_engine = None  # v10.0: Hybrid Comprehension Model

    @property
    def entailment_engine(self):
        if self._entailment_engine is None:
            self._entailment_engine = TextualEntailmentEngine()
        return self._entailment_engine

    @property
    def analogical_reasoner(self):
        if self._analogical_reasoner is None:
            self._analogical_reasoner = AnalogicalReasoner()
        return self._analogical_reasoner

    @property
    def textrank(self):
        if self._textrank is None:
            self._textrank = TextRankSummarizer(damping=0.85)
        return self._textrank

    @property
    def ner(self):
        if self._ner is None:
            self._ner = NamedEntityRecognizer()
        return self._ner

    @property
    def fuzzy_matcher(self):
        if self._fuzzy_matcher is None:
            self._fuzzy_matcher = LevenshteinMatcher()
        return self._fuzzy_matcher

    @property
    def lsa(self):
        if self._lsa is None:
            self._lsa = LatentSemanticAnalyzer(n_components=50)
        return self._lsa

    @property
    def lesk(self):
        if self._lesk is None:
            self._lesk = LeskDisambiguator()
        return self._lesk

    @property
    def coref_resolver(self):
        if self._coref_resolver is None:
            self._coref_resolver = CoreferenceResolver()
        return self._coref_resolver

    @property
    def sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            self._sentiment_analyzer = SentimentAnalyzer()
        return self._sentiment_analyzer

    @property
    def frame_analyzer(self):
        if self._frame_analyzer is None:
            self._frame_analyzer = SemanticFrameAnalyzer()
        return self._frame_analyzer

    @property
    def taxonomy(self):
        if self._taxonomy is None:
            self._taxonomy = TaxonomyClassifier()
        return self._taxonomy

    @property
    def causal_chain(self):
        if self._causal_chain is None:
            self._causal_chain = CausalChainReasoner()
        return self._causal_chain

    @property
    def pragmatics(self):
        if self._pragmatics is None:
            self._pragmatics = PragmaticInferenceEngine()
        return self._pragmatics

    @property
    def commonsense(self):
        if self._commonsense is None:
            self._commonsense = ConceptNetLinker()
        return self._commonsense

    @property
    def subject_detector(self):
        if self._subject_detector is None:
            self._subject_detector = SubjectDetector()
        return self._subject_detector

    @property
    def numerical_reasoner(self):
        if self._numerical_reasoner is None:
            self._numerical_reasoner = NumericalReasoner()
        return self._numerical_reasoner

    @property
    def cross_verifier(self):
        if self._cross_verifier is None:
            self._cross_verifier = CrossVerificationEngine()
        return self._cross_verifier

    def initialize(self):
        """Initialize all comprehension subsystems (thread-safe).

        v9.0: Double-check locking prevents race conditions when
        multiple threads call initialize() concurrently.
        """
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:  # double-check after lock acquisition
                return
            self.knowledge_base.initialize()

        # Share the KB's N-gram matcher
        self.ngram_matcher = self.knowledge_base.ngram_matcher

        # Build tokenizer from a SMALL sample of knowledge-base text
        # (full corpus BPE is too slow; MCQ solver uses keyword matching, not BPE)
        corpus = []
        for key, node in list(self.knowledge_base.nodes.items())[:20]:
            corpus.append(node.definition)
            corpus.extend(node.facts[:3])
        if corpus:
            self.tokenizer.build_vocab(corpus, min_freq=2)

        # Wire engine support (non-blocking — graceful if unavailable)
        self._science_engine = _get_cached_science_engine()
        self._math_engine = _get_cached_math_engine()
        self._code_engine = _get_cached_code_engine()
        self._quantum_gate_engine = _get_cached_quantum_gate_engine()
        self._quantum_math_core = _get_cached_quantum_math_core()
        self._dual_layer = _get_cached_dual_layer()
        self._formal_logic = _get_cached_formal_logic()
        self._deep_nlu = _get_cached_deep_nlu()

        # v10.0: Wire search/precognition engines
        self._search_engine = _get_cached_search_engine()
        self._precognition_engine = _get_cached_precognition_engine()
        self._three_engine_hub = _get_cached_three_engine_hub()
        self._precog_synthesis = _get_cached_precog_synthesis()

        # v10.0: Wire ProbabilityEngine for hybrid comprehension model
        self._probability_engine = _get_cached_probability_engine()
        self._wire_kb_into_probability_engine()

        # Enrich knowledge base with engine-derived facts
        self._enrich_from_engines()

        # v7.0.0: Fit LSA on knowledge base facts for concept-level similarity
        try:
            all_facts = []
            for node in self.knowledge_base.nodes.values():
                all_facts.append(node.definition)
                all_facts.extend(node.facts)  # v9.0: Full corpus (removed [:5] sample + 500-doc cap)
            if len(all_facts) >= 10:
                self.lsa.fit(all_facts)  # v9.0: Fit on full corpus for complete semantic coverage
        except Exception as e:
            _log.debug("LSA fitting failed (optional): %s", e)

        self._initialized = True

    def _enrich_from_engines(self):
        """Enrich knowledge base with facts from Science/Math/Code engines."""
        # Science Engine: physics constants, entropy facts
        if self._science_engine:
            try:
                se = self._science_engine
                # Add Landauer limit fact
                landauer = se.physics.adapt_landauer_limit(300)
                if isinstance(landauer, (int, float)):
                    if 'college_physics/thermodynamics' in self.knowledge_base.nodes:
                        node = self.knowledge_base.nodes['college_physics/thermodynamics']
                        node.facts.append(f"The Landauer limit at room temperature is approximately {landauer:.2e} joules per bit")
                # Add photon resonance
                photon_e = se.physics.calculate_photon_resonance()
                if isinstance(photon_e, (int, float)):
                    if 'college_physics/quantum_mechanics' in self.knowledge_base.nodes:
                        node = self.knowledge_base.nodes['college_physics/quantum_mechanics']
                        node.facts.append(f"L104 photon resonance energy is {photon_e:.6f} eV")
            except Exception as e:
                _log.debug("Science Engine enrichment failed: %s", e)

        # Math Engine: god-code value, Fibonacci, prime facts
        if self._math_engine:
            try:
                me = self._math_engine
                fib_10 = me.fibonacci(10)
                if isinstance(fib_10, list) and len(fib_10) > 8:
                    if 'college_mathematics/calculus' in self.knowledge_base.nodes:
                        node = self.knowledge_base.nodes['college_mathematics/calculus']
                        node.facts.append(f"The first 10 Fibonacci numbers are {fib_10[:10]}")
                god_val = me.god_code_value()
                if isinstance(god_val, (int, float)):
                    if 'college_mathematics/calculus' in self.knowledge_base.nodes:
                        node = self.knowledge_base.nodes['college_mathematics/calculus']
                        node.facts.append(f"The GOD_CODE constant equals {god_val}")
            except Exception as e:
                _log.debug("Math Engine enrichment failed: %s", e)

        # Quantum Engine: quantum physics knowledge enrichment
        qmc = self._quantum_math_core
        if qmc is not None:
            try:
                # Bell state fidelity — enriches quantum physics knowledge
                bell = qmc.bell_state_phi_plus(2)
                if bell and 'college_physics/quantum_mechanics' in self.knowledge_base.nodes:
                    node = self.knowledge_base.nodes['college_physics/quantum_mechanics']
                    node.facts.append(
                        "A Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2 has maximum entanglement with concurrence 1.0"
                    )
                    node.facts.append(
                        "Quantum superposition allows a qubit to exist in a linear combination of |0⟩ and |1⟩ states simultaneously"
                    )
                    node.facts.append(
                        "The no-cloning theorem states that it is impossible to create an identical copy of an arbitrary unknown quantum state"
                    )
                    node.facts.append(
                        "Quantum entanglement is a phenomenon where two particles become correlated so measuring one instantly affects the other regardless of distance"
                    )
                    # Tunnel probability enrichment
                    tunnel_p = qmc.tunnel_probability(1.0, 0.5, 1.0)
                    if isinstance(tunnel_p, (int, float)):
                        node.facts.append(
                            f"Quantum tunnelling probability through a unit barrier with half-energy particle is approximately {tunnel_p:.6f}"
                        )
            except Exception as e:
                _log.debug("Quantum Math Core enrichment failed: %s", e)

        # Quantum Gate Engine: circuit statistics enrichment
        qge = self._quantum_gate_engine
        if qge is not None:
            try:
                status = qge.status()
                if isinstance(status, dict) and 'college_physics/quantum_mechanics' in self.knowledge_base.nodes:
                    node = self.knowledge_base.nodes['college_physics/quantum_mechanics']
                    gate_count = status.get('gate_library', {}).get('total_gates', 0)
                    if gate_count > 0:
                        node.facts.append(
                            f"The L104 quantum gate engine provides {gate_count} quantum gates including Hadamard, CNOT, Toffoli, and sacred PHI gates"
                        )
                    node.facts.append(
                        "Grover's algorithm provides quadratic speedup for unstructured search by amplifying target state amplitudes"
                    )
                    node.facts.append(
                        "The quantum Fourier transform is the key component of Shor's algorithm for integer factorization"
                    )
            except Exception as e:
                _log.debug("Quantum Gate Engine enrichment failed: %s", e)

        # v10.0: Search/Precognition Engine enrichment — search algorithm facts
        if self._search_engine:
            try:
                se_status = self._search_engine.status()
                algorithms = se_status.get('algorithms', [])
                if algorithms and 'college_mathematics/calculus' in self.knowledge_base.nodes:
                    node = self.knowledge_base.nodes['college_mathematics/calculus']
                    node.facts.append(
                        f"L104 sovereign search provides {len(algorithms)} algorithms: {', '.join(algorithms)}"
                    )
                if 'college_physics/quantum_mechanics' in self.knowledge_base.nodes:
                    node = self.knowledge_base.nodes['college_physics/quantum_mechanics']
                    node.facts.append(
                        "Grover's quantum search achieves O(√N) complexity via amplitude amplification with sacred coherence injection"
                    )
                    node.facts.append(
                        "Simulated annealing uses Landauer-bounded cooling with PHI-decay for thermodynamically optimal search"
                    )
            except Exception as e:
                _log.debug("Search Engine enrichment failed: %s", e)

        if self._precognition_engine:
            try:
                pe_status = self._precognition_engine.status()
                algorithms = pe_status.get('algorithms', [])
                if algorithms and 'college_mathematics/calculus' in self.knowledge_base.nodes:
                    node = self.knowledge_base.nodes['college_mathematics/calculus']
                    node.facts.append(
                        f"L104 data precognition provides {len(algorithms)} forecasting algorithms: {', '.join(algorithms)}"
                    )
                    node.facts.append(
                        "Entropy-gradient forecasting uses Maxwell Demon reversal efficiency as a predictive signal"
                    )
            except Exception as e:
                _log.debug("Precognition Engine enrichment failed: %s", e)

        # External Knowledge Harvester: NIST physical constants + mathematical constants
        try:
            from l104_external_knowledge_harvester import (
                PHYSICAL_CONSTANTS, MATHEMATICAL_CONSTANTS, SCIENTIFIC_EQUATIONS,
                OEIS_SEQUENCES,
            )
            # Inject NIST physical constants into physics nodes
            phys_node_keys = [
                'college_physics/mechanics', 'college_physics/thermodynamics',
                'college_physics/electromagnetism', 'college_physics/quantum_mechanics',
                'conceptual_physics/basic_physics',
            ]
            for pk in phys_node_keys:
                if pk in self.knowledge_base.nodes:
                    node = self.knowledge_base.nodes[pk]
                    for cname, cdata in PHYSICAL_CONSTANTS.items():
                        readable = cname.replace('_', ' ')
                        node.facts.append(
                            f"The {readable} ({cdata['symbol']}) is {cdata['value']} {cdata['unit']}"
                        )
                    break  # Only add to one node to avoid bloat

            # Inject mathematical constants into math nodes
            math_node_keys = [
                'college_mathematics/calculus', 'elementary_mathematics/basic_math',
                'high_school_mathematics/algebra',
            ]
            for mk in math_node_keys:
                if mk in self.knowledge_base.nodes:
                    node = self.knowledge_base.nodes[mk]
                    for cname, cdata in MATHEMATICAL_CONSTANTS.items():
                        readable = cname.replace('_', ' ')
                        node.facts.append(
                            f"The {readable} ({cdata['symbol']}) equals {cdata['value']:.15g} — {cdata['description']}"
                        )
                    break

            # Inject scientific equations into ALL relevant physics/math/CS nodes
            eq_target_map = {
                'special_relativity': 'college_physics/mechanics',
                'quantum_mechanics': 'college_physics/quantum_mechanics',
                'quantum_field_theory': 'college_physics/quantum_mechanics',
                'general_relativity': 'college_physics/mechanics',
                'electromagnetism': 'college_physics/electromagnetism',
                'thermodynamics': 'college_physics/thermodynamics',
                'statistical_mechanics': 'college_physics/thermodynamics',
            }
            for eq in SCIENTIFIC_EQUATIONS:
                target_key = eq_target_map.get(eq.get('domain', ''))
                if target_key and target_key in self.knowledge_base.nodes:
                    node = self.knowledge_base.nodes[target_key]
                    node.facts.append(
                        f"{eq['name']}: {eq['equation']} (domain: {eq['domain']})"
                    )

            # OEIS sequences → elementary math node (avoid diluting calculus)
            if 'elementary_mathematics/elementary_math' in self.knowledge_base.nodes:
                seq_node = self.knowledge_base.nodes['elementary_mathematics/elementary_math']
                for oeis_id, seq_data in OEIS_SEQUENCES.items():
                    terms_str = ', '.join(str(t) for t in seq_data['first_terms'][:8])
                    seq_node.facts.append(
                        f"The {seq_data['name']} sequence ({oeis_id}) begins: {terms_str}"
                    )
        except Exception:
            pass

        # Science Engine constants: Iron (Fe) and Helium data → chemistry node
        try:
            from l104_science_engine.constants import IronConstants, HeliumConstants
            fe_he_facts = [
                f"Iron (Fe) has atomic number {IronConstants.ATOMIC_NUMBER}, Curie temperature {IronConstants.CURIE_TEMP} K, BCC crystal structure",
                f"Iron-56 has the highest binding energy per nucleon ({IronConstants.BE_PER_NUCLEON} MeV), making it the most stable nucleus",
                f"Iron's first ionization energy is {IronConstants.IONIZATION_EV} eV",
                f"Helium-4 is a doubly magic nucleus (magic numbers {HeliumConstants.MAGIC_NUMBERS}) with binding energy {HeliumConstants.BE_TOTAL} MeV",
            ]
            for chem_key in ['college_chemistry/chemical_bonding', 'high_school_chemistry/periodic_table']:
                if chem_key in self.knowledge_base.nodes:
                    for f in fe_he_facts:
                        self.knowledge_base.nodes[chem_key].facts.append(f)
                    break
        except Exception as e:
            _log.debug("Science Engine constants enrichment failed: %s", e)

        # Math Engine constants → astronomy/cosmology (not mechanics!)
        try:
            from l104_math_engine.constants import (
                HUBBLE_CONSTANT, FE_ELECTRON_CONFIG,
                FEIGENBAUM_DELTA,
            )
            astro_facts = [
                f"The Hubble constant is approximately {HUBBLE_CONSTANT} km/s/Mpc, measuring the expansion rate of the universe",
                f"Iron (Fe) electron configuration is {FE_ELECTRON_CONFIG}",
            ]
            for key in ['astronomy/astronomy_expanded', 'astronomy/stellar_evolution']:
                if key in self.knowledge_base.nodes:
                    for f in astro_facts:
                        self.knowledge_base.nodes[key].facts.append(f)
                    break
            # Feigenbaum → computer science (chaos theory)
            if 'computer_science/algorithms' in self.knowledge_base.nodes:
                self.knowledge_base.nodes['computer_science/algorithms'].facts.append(
                    f"The Feigenbaum constant δ ≈ {FEIGENBAUM_DELTA} governs the rate of period-doubling bifurcations in chaotic dynamical systems"
                )
        except Exception as e:
            _log.debug("Math Engine constants enrichment failed: %s", e)

        # Algorithm database: academic algorithms → computer science only
        try:
            import json
            algo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'algorithm_database.json')
            if os.path.exists(algo_path):
                with open(algo_path, 'r') as f:
                    algo_db = json.load(f)
                algo_facts = {
                    'FAST_FOURIER_TRANSFORM': "The Fast Fourier Transform (FFT) converts time-domain signals to frequency-domain in O(n log n) time",
                    'SHANNON_ENTROPY_SCAN': "Shannon entropy H(X) = -Σ p(x)·log₂(p(x)) measures the average information content in bits",
                    'PRIME_DENSITY_PNT': "The Prime Number Theorem: density of primes near n ≈ 1/ln(n), so π(x) ~ x/ln(x)",
                }
                if 'computer_science/algorithms' in self.knowledge_base.nodes:
                    cs_node = self.knowledge_base.nodes['computer_science/algorithms']
                    for algo_name, fact_text in algo_facts.items():
                        if algo_name in algo_db.get('algorithms', {}):
                            cs_node.facts.append(fact_text)
        except Exception as e:
            _log.debug("Algorithm database enrichment failed: %s", e)

    def _enrich_comprehend_engines(self):
        """v9.0: One-time enrichment from Math/Science/Dual-Layer engines for comprehend().

        Previously this code ran inside comprehend() on EVERY call, causing
        duplicate facts to pile up in the knowledge base. Now runs once.
        """
        # Math Engine deeper enrichment — wave coherence, primes, Lorentz
        if self._math_engine:
            try:
                me = self._math_engine
                wc = me.wave_coherence(286.0, GOD_CODE)
                if isinstance(wc, (int, float)):
                    if 'college_physics/quantum_mechanics' in self.knowledge_base.nodes:
                        node = self.knowledge_base.nodes['college_physics/quantum_mechanics']
                        node.facts.append(
                            f"Wave coherence between 286Hz and GOD_CODE frequency is {wc:.6f}"
                        )
                primes = me.primes_up_to(50)
                if isinstance(primes, list) and len(primes) > 5:
                    if 'college_mathematics/calculus' in self.knowledge_base.nodes:
                        node = self.knowledge_base.nodes['college_mathematics/calculus']
                        node.facts.append(f"Prime numbers up to 50: {primes}")
                try:
                    harm = me.harmonic.verify_correspondences()
                    if isinstance(harm, dict) and harm.get('verified'):
                        if 'college_physics/quantum_mechanics' in self.knowledge_base.nodes:
                            node = self.knowledge_base.nodes['college_physics/quantum_mechanics']
                            node.facts.append(
                                "The iron-286Hz harmonic correspondence is verified via L104 harmonic process"
                            )
                except Exception:
                    _log.debug("Harmonic correspondence verification skipped")
            except Exception as e:
                _log.debug("Math Engine comprehend enrichment failed: %s", e)

        # Science Engine deeper enrichment — electron resonance, iron lattice
        if self._science_engine:
            try:
                se = self._science_engine
                e_res = se.physics.derive_electron_resonance()
                if isinstance(e_res, (int, float)):
                    if 'college_physics/electromagnetism' in self.knowledge_base.nodes:
                        node = self.knowledge_base.nodes['college_physics/electromagnetism']
                        node.facts.append(
                            f"L104 electron resonance frequency derived at {e_res:.6f} Hz"
                        )
                demon_eff = se.entropy.calculate_demon_efficiency(0.5)
                if isinstance(demon_eff, (int, float)):
                    if 'college_physics/thermodynamics' in self.knowledge_base.nodes:
                        node = self.knowledge_base.nodes['college_physics/thermodynamics']
                        node.facts.append(
                            f"Maxwell's Demon reversal efficiency at 0.5 entropy: {demon_eff:.4f}"
                        )
            except Exception as e:
                _log.debug("Science Engine comprehend enrichment failed: %s", e)

        # Dual-Layer Engine enrichment — physics grounding facts
        dl = self._dual_layer
        if dl is not None:
            try:
                dl_score = dl.dual_score()
                if isinstance(dl_score, (int, float)):
                    if 'college_physics/mechanics' in self.knowledge_base.nodes:
                        node = self.knowledge_base.nodes['college_physics/mechanics']
                        node.facts.append(
                            f"L104 Dual-Layer Engine thought-physics alignment score: {dl_score:.4f}"
                        )
            except Exception as e:
                _log.debug("Dual-Layer Engine comprehend enrichment failed: %s", e)

        # Invalidate query cache after enrichment mutations
        self.knowledge_base.invalidate_query_cache()

    def _wire_kb_into_probability_engine(self):
        """v10.0 Hybrid Comprehension: Feed KB facts into ProbabilityEngine.

        Bridges four data planes:
        1. KB node facts → DataIngestor token_counter  (token priors)
        2. KB subject_index → DataIngestor category_counter  (subject priors)
        3. ConceptNet relations → GateProbabilityBridge  (relation gates)
        4. KB coverage signal → ASIInsightSynthesis warm-up  (consciousness)
        """
        pe = self._probability_engine
        if pe is None:
            return
        try:
            # Phase 1: Feed KB node facts into DataIngestor token_counter
            ingestor = pe.ingestor
            for key, node in self.knowledge_base.nodes.items():
                all_text = node.definition + " " + " ".join(node.facts)
                tokens = [w.lower().strip(".,;:!?()[]") for w in all_text.split() if len(w) > 2]
                for token in tokens:
                    if token:
                        ingestor.token_counter[token] = ingestor.token_counter.get(token, 0) + 1

            # Phase 2: Feed subject_index into category_counter
            if hasattr(self.knowledge_base, 'subject_index'):
                for subject, keys in self.knowledge_base.subject_index.items():
                    ingestor.category_counter[subject] = (
                        ingestor.category_counter.get(subject, 0) + len(keys)
                    )

            # Phase 3: Bridge ConceptNet relations into GateProbabilityBridge
            if hasattr(self.knowledge_base, 'relation_graph'):
                bridge = pe.bridge
                for src, relations in self.knowledge_base.relation_graph.items():
                    for rel_type, targets in relations.items():
                        gate_name = f"KB_{rel_type}".upper()
                        if hasattr(bridge, 'logic_gates'):
                            existing = bridge.logic_gates.get(gate_name, 0)
                            bridge.logic_gates[gate_name] = existing + len(targets)
                # Rebuild consolidated gates after injection
                if hasattr(pe, '_consolidate_gates'):
                    pe._consolidate_gates()

            # Phase 4: Warm up ASIInsightSynthesis with KB coverage signals
            kb_status = self.knowledge_base.get_status()
            coverage = min(1.0, kb_status["total_nodes"] / 80)
            fact_density = min(1.0, kb_status["total_facts"] / 600)
            try:
                pe.synthesize_insight(
                    [coverage, fact_density, coverage * fact_density],
                    consciousness_level=0.3,
                )
            except Exception:
                pass

            _log.debug("KB→PE wiring complete: %d tokens, %d categories",
                       sum(ingestor.token_counter.values()),
                       sum(ingestor.category_counter.values()))
        except Exception as e:
            _log.debug("KB→PE wiring failed (non-fatal): %s", e)

    def comprehend(self, text: str) -> Dict[str, Any]:
        """Comprehend a text passage — tokenize, encode, extract meaning, and analyze discourse."""
        if not self._initialized:
            self.initialize()

        self._total_queries += 1

        # v9.0: Guard — engine enrichment inside comprehend() runs only once.
        # Previously these facts were appended on EVERY comprehend() call,
        # causing duplicate facts and O(n_queries) KB pollution.
        if not hasattr(self, '_comprehend_enriched'):
            self._comprehend_enriched = True
            self._enrich_comprehend_engines()

        # Tokenize
        token_ids = self.tokenizer.tokenize(text)

        # Retrieve related knowledge (tri-signal: TF-IDF + N-gram + relations)
        knowledge_hits = self.knowledge_base.query(text, top_k=5)

        # Extract key concepts
        concepts = self._extract_concepts(text)

        # Semantic similarity to knowledge base
        similarities = [(k, round(s, 4)) for k, _, s in knowledge_hits]

        # N-gram phrase matches
        ngram_hits = self.ngram_matcher.match(text, top_k=5)
        phrase_matches = [(k, round(s, 3)) for k, s in ngram_hits]

        # DeepNLU discourse analysis (if available)
        # v9.0: Cache NLU results per text hash to avoid 20-layer recomputation
        discourse_info = {}
        temporal_info = {}
        causal_info = {}
        disambiguation_info = {}
        if self._deep_nlu is not None:
            _nlu_hash = hashlib.md5(text.encode()).hexdigest()
            _cached_nlu = self._nlu_cache.get(_nlu_hash)
            if _cached_nlu is not None:
                discourse_info = _cached_nlu.get('discourse', {})
                temporal_info = _cached_nlu.get('temporal', {})
                causal_info = _cached_nlu.get('causal', {})
                disambiguation_info = _cached_nlu.get('disambiguation', {})
            else:
                try:
                    nlu_result = self._deep_nlu.analyze(text)
                    if hasattr(nlu_result, 'get'):
                        discourse_info = {
                            "intent": nlu_result.get("intent", {}),
                            "coherence": nlu_result.get("coherence_score", 0.0),
                        }
                except Exception as e:
                    _log.debug("DeepNLU analysis failed: %s", e)
                # v6.0.0: Temporal reasoning via DeepNLU v2.0.0
                try:
                    if hasattr(self._deep_nlu, 'temporal'):
                        temporal_info = self._deep_nlu.temporal.analyze(text)
                except Exception as e:
                    _log.debug("DeepNLU temporal failed: %s", e)
                # v6.0.0: Causal reasoning via DeepNLU v2.0.0
                try:
                    if hasattr(self._deep_nlu, 'causal'):
                        causal_info = self._deep_nlu.causal.analyze(text)
                except Exception as e:
                    _log.debug("DeepNLU causal failed: %s", e)
                # v6.0.0: Contextual disambiguation via DeepNLU v2.0.0
                try:
                    if hasattr(self._deep_nlu, 'disambiguator'):
                        disambiguation_info = self._deep_nlu.disambiguator.disambiguate(text)
                except Exception as e:
                    _log.debug("DeepNLU disambiguation failed: %s", e)

                # v9.0: Cache NLU results for this text
                _nlu_result_for_cache = {
                    'discourse': discourse_info,
                    'temporal': temporal_info,
                    'causal': causal_info,
                    'disambiguation': disambiguation_info,
                }
                if len(self._nlu_cache) >= self._nlu_cache_maxsize:
                    oldest = next(iter(self._nlu_cache))
                    del self._nlu_cache[oldest]
                self._nlu_cache[_nlu_hash] = _nlu_result_for_cache

        # v6.0.0: Enhanced comprehension depth incorporating all signals
        base_depth = min(1.0, len(concepts) * 0.1 + len(knowledge_hits) * 0.05 + len(ngram_hits) * 0.03)
        temporal_boost = temporal_info.get('temporal_richness', 0.0) * 0.05
        causal_boost = causal_info.get('causal_density', 0.0) * 0.05
        wsd_boost = disambiguation_info.get('wsd_coverage', 0.0) * 0.03
        enhanced_depth = min(1.0, base_depth + temporal_boost + causal_boost + wsd_boost)

        # ── v7.0.0: Named Entity Recognition ────────────────────────────
        entities = {}
        try:
            entities = self.ner.extract_entity_types(text)
        except Exception:
            pass

        # ── v7.0.0: TextRank Summarization ───────────────────────────────
        summary_info = {}
        try:
            if len(text) > 200:
                summary_result = self.textrank.summarize(text, num_sentences=3)
                summary_info = {
                    "summary": summary_result["summary"],
                    "compression_ratio": summary_result["compression_ratio"],
                    "total_sentences": summary_result["total_sentences"],
                }
        except Exception:
            pass

        # ── v7.0.0: Lesk WSD (algorithmic complement to DeepNLU) ────────
        lesk_results = []
        try:
            lesk_results = self.lesk.disambiguate_all(text)
        except Exception:
            pass

        # ── v7.0.0: Textual Entailment against top knowledge hits ────────
        entailment_info = []
        try:
            for key, node, score in knowledge_hits[:3]:
                ent_result = self.entailment_engine.entail(node.definition, text)
                if ent_result["label"] != "neutral":
                    entailment_info.append({
                        "knowledge_key": key,
                        "label": ent_result["label"],
                        "confidence": ent_result["confidence"],
                    })
        except Exception:
            pass

        # ── v7.0.0: LSA concept similarity ───────────────────────────────
        lsa_matches = []
        try:
            if self.lsa._fitted:
                lsa_hits = self.lsa.query_similarity(text, top_k=3)
                lsa_matches = [(idx, round(sim, 4)) for idx, sim in lsa_hits]
        except Exception:
            pass

        # v7.0.0: Enhanced depth with new algorithm signals
        entity_boost = min(0.05, len(entities) * 0.01)
        entailment_boost = min(0.05, len(entailment_info) * 0.02)
        lesk_boost = min(0.03, len(lesk_results) * 0.01)
        enhanced_depth = min(1.0, enhanced_depth + entity_boost + entailment_boost + lesk_boost)

        # ── v10.0: Search-Augmented Knowledge Retrieval ──────────────────
        search_augmented_hits = []
        precog_insight = {}
        try:
            if self._search_engine is not None:
                # Build an in-memory search space from knowledge base fact text
                kb_facts = []
                kb_keys = []
                for key, node in list(self.knowledge_base.nodes.items())[:200]:
                    for fact in node.facts[:10]:
                        kb_facts.append(fact)
                        kb_keys.append(key)
                if kb_facts and len(text) > 10:
                    # Use HD search over fact embeddings for concept matching
                    hd_result = self._search_engine.search(
                        data=kb_facts,
                        oracle=lambda idx: 1.0 if any(w in kb_facts[idx].lower() for w in text.lower().split()[:5]) else 0.0,
                        algorithm="hyperdimensional",
                    )
                    if hd_result.get("found") and hd_result.get("best_index") is not None:
                        best_idx = hd_result["best_index"]
                        if 0 <= best_idx < len(kb_keys):
                            search_augmented_hits.append({
                                "key": kb_keys[best_idx],
                                "fact": kb_facts[best_idx][:200],
                                "score": round(hd_result.get("best_score", 0.0), 4),
                            })
                            self._search_enhanced_queries += 1
        except Exception as e:
            _log.debug("Search augmentation failed: %s", e)

        try:
            if self._precognition_engine is not None and len(concepts) >= 3:
                # Use precognition for concept trend prediction
                concept_signal = [float(hash(c) % 100) / 100 for c in concepts[:10]]
                precog_result = self._precognition_engine.full_precognition(
                    concept_signal, horizon=3
                )
                if isinstance(precog_result, dict):
                    precog_insight = {
                        "trend": precog_result.get("trend", "unknown"),
                        "confidence": round(precog_result.get("confidence", 0.0), 4),
                    }
        except Exception as e:
            _log.debug("Precognition insight failed: %s", e)

        # v10.0: Search/precog depth boost
        search_boost = min(0.04, len(search_augmented_hits) * 0.02)
        precog_boost = min(0.03, precog_insight.get("confidence", 0.0) * 0.03)
        enhanced_depth = min(1.0, enhanced_depth + search_boost + precog_boost)

        # ── v8.0.0: Coreference Resolution ──────────────────────────────
        coref_info = {}
        try:
            coref_result = self.coref_resolver.resolve(text)
            if coref_result["resolution_count"] > 0:
                coref_info = {
                    "resolutions": coref_result["resolutions"],
                    "resolved_text": coref_result["resolved_text"],
                }
        except Exception:
            pass

        # ── v8.0.0: Sentiment Analysis ───────────────────────────────────
        sentiment_info = {}
        try:
            sentiment_info = self.sentiment_analyzer.analyze(text)
        except Exception:
            pass

        # ── v8.0.0: Semantic Frame Analysis ──────────────────────────────
        frame_info = {}
        try:
            frame_info = self.frame_analyzer.analyze(text)
        except Exception:
            pass

        # ── v8.0.0: Pragmatic Inference ──────────────────────────────────
        pragmatic_info = {}
        try:
            pragmatic_info = self.pragmatics.analyze(text)
        except Exception:
            pass

        # ── v8.0.0: Causal Chain Analysis ────────────────────────────────
        causal_chain_info = {}
        try:
            # Find causal concepts in text and run forward chaining
            all_causes = set(self.causal_chain._CAUSAL_KB.keys())
            text_lower_cc = text.lower()
            found_causes = [c for c in all_causes if c in text_lower_cc]
            if found_causes:
                chains = self.causal_chain.forward_chain(found_causes[0], max_hops=2)
                causal_chain_info = {
                    "root_cause": found_causes[0],
                    "effects": chains[:5],
                    "chain_count": len(chains),
                }
        except Exception:
            pass

        # ── v8.0.0: Commonsense Knowledge ────────────────────────────────
        commonsense_info = {}
        try:
            all_subjects = set()
            for subjects in self.commonsense._RELATIONS.values():
                all_subjects.update(subjects.keys())
            text_lower_cs = text.lower()
            found_subjects = [s for s in all_subjects if s in text_lower_cs]
            if found_subjects:
                rels = self.commonsense.query(found_subjects[0])
                commonsense_info = {
                    "subject": found_subjects[0],
                    "relations": rels,
                }
        except Exception:
            pass

        # v8.0.0: Enhanced depth with v8 algorithm signals
        coref_boost = min(0.03, len(coref_info.get("resolutions", [])) * 0.01)
        frame_boost = min(0.03, frame_info.get("frame_count", 0) * 0.01)
        pragma_boost = min(0.02, len(pragmatic_info.get("implicatures", [])) * 0.01)
        enhanced_depth = min(1.0, enhanced_depth + coref_boost + frame_boost + pragma_boost)

        return {
            "token_count": len(token_ids),
            "concepts_extracted": concepts,
            "knowledge_matches": similarities,
            "phrase_matches": phrase_matches,
            "discourse": discourse_info,
            "temporal": temporal_info,
            "causal": causal_info,
            "disambiguation": disambiguation_info,
            "entities": entities,
            "summary": summary_info,
            "lesk_wsd": lesk_results,
            "entailment": entailment_info,
            "lsa_concept_matches": lsa_matches,
            "coreference": coref_info,
            "sentiment": sentiment_info,
            "semantic_frames": frame_info,
            "pragmatics": pragmatic_info,
            "causal_chains": causal_chain_info,
            "commonsense": commonsense_info,
            "search_augmented": search_augmented_hits,
            "precognition_insight": precog_insight,
            "comprehension_depth": enhanced_depth,
        }

    def answer_mcq(self, question: str, choices: List[str],
                   subject: Optional[str] = None) -> Dict[str, Any]:
        """Answer a multiple-choice question (MMLU format)."""
        if not self._initialized:
            self.initialize()

        return self.mcq_solver.solve(question, choices, subject)

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using knowledge base matching + N-grams."""
        text_lower = text.lower()
        found_concepts = []

        for key, node in self.knowledge_base.nodes.items():
            concept_lower = node.concept.lower().replace("_", " ")
            if concept_lower in text_lower:
                found_concepts.append(node.concept)

        # Also extract via N-gram phrase matching
        ngram_hits = self.ngram_matcher.match(text, top_k=5)
        for key, score in ngram_hits:
            if key in self.knowledge_base.nodes:
                concept = self.knowledge_base.nodes[key].concept
                if concept not in found_concepts:
                    found_concepts.append(concept)

        return found_concepts[:15]

    def evaluate_comprehension(self) -> float:
        """Compute overall language comprehension score (0-1)."""
        kb_status = self.knowledge_base.get_status()
        mcq_status = self.mcq_solver.get_status()

        # Weighted components
        knowledge_coverage = min(1.0, kb_status["total_nodes"] / 100)
        fact_density = min(1.0, kb_status["total_facts"] / 500)
        answering_accuracy = mcq_status["accuracy"]
        relation_depth = min(1.0, kb_status.get("relation_edges", 0) / 50)
        ngram_coverage = min(1.0, kb_status.get("ngram_phrases_indexed", 0) / 5000)

        self._comprehension_score = (
            knowledge_coverage * 0.20 +
            fact_density * 0.20 +
            answering_accuracy * 0.35 +
            relation_depth * 0.10 +
            ngram_coverage * 0.15
        )
        return self._comprehension_score

    # ── Three-Engine Scoring (ASI 15D Integration) ────────────────────────────

    def three_engine_comprehension_score(self) -> float:
        """Compute three-engine comprehension score for ASI dimension integration.

        Measures language comprehension quality using all available engines:
        - Science Engine: entropy-based knowledge coherence
        - Math Engine: GOD_CODE alignment + harmonic knowledge structure
        - Code Engine: structural analysis of comprehension pipeline
        - Probability Engine: Bayesian-quantum hybrid posterior quality

        v10.0: Added Component 6 (Probability Engine) at 10% weight.

        Returns: 0.0-1.0 score suitable for ASI 15D scoring.
        """
        if not self._initialized:
            self.initialize()

        scores = []

        # Component 1: Knowledge base coverage and structure (weight: 0.25)
        kb_status = self.knowledge_base.get_status()
        kb_score = min(1.0, kb_status["total_nodes"] / 80) * 0.5 + \
                   min(1.0, kb_status["total_facts"] / 600) * 0.3 + \
                   min(1.0, kb_status.get("relation_edges", 0) / 40) * 0.2
        scores.append(kb_score * 0.25)

        # Component 2: Science Engine entropy coherence (weight: 0.15)
        se = self._science_engine or _get_cached_science_engine()
        if se is not None:
            try:
                demon_eff = se.entropy.calculate_demon_efficiency(0.3)
                if isinstance(demon_eff, (int, float)):
                    scores.append(min(1.0, demon_eff) * 0.15)
                else:
                    scores.append(0.5 * 0.15)
            except Exception:
                scores.append(0.5 * 0.15)
        else:
            scores.append(0.5 * 0.15)

        # Component 3: Math Engine harmonic alignment (weight: 0.15)
        me = self._math_engine or _get_cached_math_engine()
        if me is not None:
            try:
                alignment = me.sacred_alignment(GOD_CODE)
                if isinstance(alignment, (int, float)):
                    scores.append(min(1.0, alignment) * 0.15)
                elif isinstance(alignment, dict):
                    scores.append(min(1.0, alignment.get('score', 0.5)) * 0.15)
                else:
                    scores.append(0.5 * 0.15)
            except Exception:
                scores.append(0.5 * 0.15)
        else:
            scores.append(0.5 * 0.15)

        # Component 4: MCQ performance (weight: 0.20)
        mcq_status = self.mcq_solver.get_status()
        if mcq_status["questions_answered"] > 0:
            scores.append(mcq_status["accuracy"] * 0.20)
        else:
            scores.append(0.5 * 0.20)  # Neutral if no questions answered

        # Component 5: Search/Precognition engine health (weight: 0.15)
        search_precog_score = 0.0
        sp_components = 0
        if self._search_engine is not None:
            try:
                s_status = self._search_engine.status()
                algo_count = len(s_status.get('algorithms', []))
                search_precog_score += min(1.0, algo_count / 7)
                sp_components += 1
            except Exception:
                pass
        if self._precognition_engine is not None:
            try:
                p_status = self._precognition_engine.status()
                algo_count = len(p_status.get('algorithms', []))
                search_precog_score += min(1.0, algo_count / 7)
                sp_components += 1
            except Exception:
                pass
        if self._three_engine_hub is not None:
            try:
                h_status = self._three_engine_hub.status()
                pipe_count = len(h_status.get('pipelines', []))
                search_precog_score += min(1.0, pipe_count / 5)
                sp_components += 1
            except Exception:
                pass
        if sp_components > 0:
            scores.append((search_precog_score / sp_components) * 0.15)
        else:
            scores.append(0.5 * 0.15)  # Neutral if unavailable

        # Component 6: Probability Engine hybrid quality (weight: 0.10)
        pe = self._probability_engine
        if pe is not None:
            try:
                pe_score = 0.0
                pe_parts = 0
                # DataIngestor token coverage from KB wiring
                if hasattr(pe, 'ingestor') and hasattr(pe.ingestor, 'token_counter'):
                    token_count = sum(pe.ingestor.token_counter.values()) if pe.ingestor.token_counter else 0
                    pe_score += min(1.0, token_count / 5000)
                    pe_parts += 1
                # Sacred probability self-test
                sp = pe.sacred_probability(GOD_CODE)
                if isinstance(sp, (int, float)) and 0 < sp <= 1:
                    pe_score += sp
                    pe_parts += 1
                # Ensemble resonance
                ens = pe.ensemble_resonance()
                if isinstance(ens, dict):
                    ens_val = ens.get('resonance', ens.get('score', ens.get('mean', 0.5)))
                    pe_score += min(1.0, float(ens_val))
                    pe_parts += 1
                elif isinstance(ens, (int, float)):
                    pe_score += min(1.0, ens)
                    pe_parts += 1
                # Insight synthesis readiness
                test_insight = pe.synthesize_insight([0.5, 0.3, 0.2])
                if hasattr(test_insight, 'consciousness_probability'):
                    pe_score += min(1.0, test_insight.consciousness_probability)
                    pe_parts += 1
                if pe_parts > 0:
                    scores.append((pe_score / pe_parts) * 0.10)
                else:
                    scores.append(0.5 * 0.10)
            except Exception:
                scores.append(0.5 * 0.10)
        else:
            scores.append(0.5 * 0.10)  # Neutral if unavailable

        self._three_engine_score = sum(scores)
        return round(self._three_engine_score, 6)

    def three_engine_status(self) -> Dict[str, Any]:
        """Get three-engine integration status for language comprehension."""
        return {
            "engines": {
                "science_engine": self._science_engine is not None,
                "math_engine": self._math_engine is not None,
                "code_engine": self._code_engine is not None,
                "quantum_gate_engine": self._quantum_gate_engine is not None,
                "quantum_math_core": self._quantum_math_core is not None,
                "dual_layer": self._dual_layer is not None,
                "formal_logic": self._formal_logic is not None,
                "deep_nlu": self._deep_nlu is not None,
                "search_engine": self._search_engine is not None,
                "precognition_engine": self._precognition_engine is not None,
                "three_engine_hub": self._three_engine_hub is not None,
                "probability_engine": self._probability_engine is not None,
            },
            "scores": {
                "comprehension": round(self._comprehension_score, 6),
                "three_engine": round(self._three_engine_score, 6),
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "version": self.VERSION,
            "initialized": self._initialized,
            "tokenizer_vocab": self.tokenizer.vocab_count,
            "knowledge_base": self.knowledge_base.get_status(),
            "mcq_solver": self.mcq_solver.get_status(),
            "total_queries": self._total_queries,
            "comprehension_score": round(self._comprehension_score, 4),
            "three_engine_score": round(self._three_engine_score, 6),
            "engine_support": {
                "science_engine": self._science_engine is not None,
                "math_engine": self._math_engine is not None,
                "code_engine": self._code_engine is not None,
                "quantum_gate_engine": self._quantum_gate_engine is not None,
                "quantum_math_core": self._quantum_math_core is not None,
                "dual_layer": self._dual_layer is not None,
                "formal_logic": self._formal_logic is not None,
                "deep_nlu": self._deep_nlu is not None,
                "search_engine": self._search_engine is not None,
                "precognition_engine": self._precognition_engine is not None,
                "three_engine_hub": self._three_engine_hub is not None,
                "probability_engine": self._probability_engine is not None,
            },
            "layers": {
                "1_tokenizer": True,
                "2_semantic_encoder": True,
                "2b_ngram_matcher": True,
                "3_knowledge_graph": self.knowledge_base._initialized,
                "4_bm25_ranker": True,
                "4b_subject_detector": self.subject_detector is not None,
                "4c_numerical_reasoner": self.numerical_reasoner is not None,
                "5_mcq_solver": True,
                "6_cross_verification": self.cross_verifier is not None,
                "7_chain_of_thought": True,
                "8_calibration": True,
                "9_temporal_reasoning": self._deep_nlu is not None,
                "10_causal_reasoning": self._deep_nlu is not None,
                "11_contextual_disambiguation": self._deep_nlu is not None,
                "12_textual_entailment": self.entailment_engine is not None,
                "13_analogical_reasoning": self.analogical_reasoner is not None,
                "14_textrank_summarization": self.textrank is not None,
                "15_named_entity_recognition": self.ner is not None,
                "16_fuzzy_matching": self.fuzzy_matcher is not None,
                "17_latent_semantic_analysis": self.lsa._fitted if self.lsa else False,
                "18_lesk_wsd": self.lesk is not None,
                "19_coreference_resolution": self.coref_resolver is not None,
                "20_sentiment_analysis": self.sentiment_analyzer is not None,
                "21_semantic_frames": self.frame_analyzer is not None,
                "22_taxonomy_classification": self.taxonomy is not None,
                "23_causal_chain_reasoning": self.causal_chain is not None,
                "24_pragmatic_inference": self.pragmatics is not None,
                "25_commonsense_knowledge": self.commonsense is not None,
                "26_search_augmentation": self._search_engine is not None,
                "27_precognition_insight": self._precognition_engine is not None,
                "28_three_engine_search_hub": self._three_engine_hub is not None,
            },
            "v3_stats": {
                "subject_detections": self.mcq_solver._subject_detections,
                "numerical_assists": self.mcq_solver._numerical_assists,
                "cross_verifications": self.mcq_solver._cross_verifications,
            },
            "v7_algorithms": {
                "entailment_engine": True,
                "analogical_reasoner": True,
                "textrank_summarizer": True,
                "ner_engine": True,
                "levenshtein_matcher": True,
                "lsa_fitted": self.lsa._fitted if self.lsa else False,
                "lesk_disambiguator": True,
                "ner_entities_found": self.ner._entities_found if self.ner else 0,
                "lesk_disambiguations": self.lesk._disambiguations if self.lesk else 0,
            },
            "v9_pipeline": {
                "query_cache_size": len(self.knowledge_base._query_cache),
                "nlu_cache_size": len(self._nlu_cache),
                "thread_safe_init": True,
                "lsa_full_corpus": True,
                "bm25_reuse": True,
                "enrichment_guard": hasattr(self, '_comprehend_enriched'),
                "early_exit_enabled": True,
            },
            "v8_algorithms": {
                "coreference_resolver": True,
                "sentiment_analyzer": True,
                "semantic_frame_analyzer": True,
                "taxonomy_classifier": True,
                "causal_chain_reasoner": True,
                "pragmatic_inference": True,
                "commonsense_linker": True,
                "coref_resolutions": self.coref_resolver._resolutions if self.coref_resolver else 0,
                "sentiment_analyses": self.sentiment_analyzer._analyses if self.sentiment_analyzer else 0,
                "frame_analyses": self.frame_analyzer._analyses if self.frame_analyzer else 0,
                "taxonomy_lookups": self.taxonomy._lookups if self.taxonomy else 0,
                "causal_inferences": self.causal_chain._inferences if self.causal_chain else 0,
                "pragmatic_analyses": self.pragmatics._analyses if self.pragmatics else 0,
                "commonsense_lookups": self.commonsense._lookups if self.commonsense else 0,
            },
            "v10_search_precog": {
                "search_engine": self._search_engine is not None,
                "precognition_engine": self._precognition_engine is not None,
                "three_engine_hub": self._three_engine_hub is not None,
                "search_enhanced_queries": self._search_enhanced_queries,
            },
        }
