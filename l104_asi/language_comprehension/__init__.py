"""
L104 ASI LANGUAGE COMPREHENSION ENGINE v9.0.0
═══════════════════════════════════════════════════════════════════════════════
Decomposed package — refactored from monolith for maintainability.

Submodules:
  constants          Sacred constants (PHI, GOD_CODE, TAU, VOID_CONSTANT)
  engine_support     Lazy-loading engine accessor functions
  tokenizer          L104Tokenizer (BPE-style subword tokenization)
  vectorizer         TFIDFVectorizer, SemanticEncoder
  ngram              NGramMatcher (bigram/trigram phrase matching)
  relation_extractor RelationTripleExtractor
  knowledge_base     KnowledgeNode, MMLUKnowledgeBase (90+ subjects)
  retrieval          BM25Ranker (Okapi BM25 passage retrieval)
  detectors          SubjectDetector, NumericalReasoner
  verification       CrossVerificationEngine
  nlp_algorithms     TextualEntailmentEngine, AnalogicalReasoner,
                     TextRankSummarizer, NamedEntityRecognizer,
                     LevenshteinMatcher, LatentSemanticAnalyzer,
                     LeskDisambiguator
  discourse          CoreferenceResolver, SentimentAnalyzer,
                     SemanticFrameAnalyzer, TaxonomyClassifier,
                     CausalChainReasoner, PragmaticInferenceEngine,
                     ConceptNetLinker
  mcq_solver         MCQSolver
  engine             LanguageComprehensionEngine (main orchestrator)

All public symbols are re-exported here for full backward compatibility.
"""
from __future__ import annotations

# ── Constants ─────────────────────────────────────────────────────────────────
from .constants import PHI, GOD_CODE, TAU, VOID_CONSTANT

# ── Engine Support (lazy loaders) ─────────────────────────────────────────────
from .engine_support import (
    _get_science_engine,
    _get_math_engine,
    _get_code_engine,
    _get_quantum_gate_engine,
    _get_quantum_math_core,
    _get_dual_layer_engine,
    _get_formal_logic_engine,
    _get_deep_nlu_engine,
    _get_cached_local_intellect,
    _get_cached_science_engine,
    _get_cached_math_engine,
    _get_cached_code_engine,
    _get_cached_quantum_gate_engine,
    _get_cached_quantum_math_core,
    _get_cached_dual_layer,
    _get_cached_formal_logic,
    _get_cached_deep_nlu,
    _get_cached_search_engine,
    _get_cached_precognition_engine,
    _get_cached_three_engine_hub,
    _get_cached_precog_synthesis,
    _get_cached_quantum_reasoning,
    _get_cached_quantum_probability,
    _get_cached_probability_engine,
)

# ── Core NLP Components ───────────────────────────────────────────────────────
from .tokenizer import L104Tokenizer
from .vectorizer import TFIDFVectorizer, SemanticEncoder
from .ngram import NGramMatcher
from .relation_extractor import RelationTripleExtractor
from .knowledge_base import KnowledgeNode, MMLUKnowledgeBase
from .retrieval import BM25Ranker
from .detectors import SubjectDetector, NumericalReasoner
from .verification import CrossVerificationEngine

# ── NLP Algorithm Layers (v7.0) ───────────────────────────────────────────────
from .nlp_algorithms import (
    TextualEntailmentEngine,
    AnalogicalReasoner,
    TextRankSummarizer,
    NamedEntityRecognizer,
    LevenshteinMatcher,
    LatentSemanticAnalyzer,
    LeskDisambiguator,
)

# ── Discourse Algorithm Layers (v8.0) ─────────────────────────────────────────
from .discourse import (
    CoreferenceResolver,
    SentimentAnalyzer,
    SemanticFrameAnalyzer,
    TaxonomyClassifier,
    CausalChainReasoner,
    PragmaticInferenceEngine,
    ConceptNetLinker,
)

# ── Solver + Engine ───────────────────────────────────────────────────────────
from .mcq_solver import MCQSolver
from .engine import LanguageComprehensionEngine

__all__ = [
    # Constants
    "PHI", "GOD_CODE", "TAU", "VOID_CONSTANT",
    # Core components
    "L104Tokenizer", "TFIDFVectorizer", "SemanticEncoder",
    "NGramMatcher", "RelationTripleExtractor",
    "KnowledgeNode", "MMLUKnowledgeBase",
    "BM25Ranker", "SubjectDetector", "NumericalReasoner",
    "CrossVerificationEngine",
    # NLP algorithms (v7.0)
    "TextualEntailmentEngine", "AnalogicalReasoner", "TextRankSummarizer",
    "NamedEntityRecognizer", "LevenshteinMatcher", "LatentSemanticAnalyzer",
    "LeskDisambiguator",
    # Discourse algorithms (v8.0)
    "CoreferenceResolver", "SentimentAnalyzer", "SemanticFrameAnalyzer",
    "TaxonomyClassifier", "CausalChainReasoner", "PragmaticInferenceEngine",
    "ConceptNetLinker",
    # Solver + Engine
    "MCQSolver", "LanguageComprehensionEngine",
    # Engine support (private but used externally)
    "_get_cached_deep_nlu", "_get_cached_science_engine",
    "_get_cached_math_engine", "_get_cached_code_engine",
    "_get_cached_dual_layer", "_get_cached_formal_logic",
    "_get_cached_quantum_gate_engine", "_get_cached_quantum_math_core",
    "_get_cached_local_intellect", "_get_cached_search_engine",
    "_get_cached_precognition_engine", "_get_cached_three_engine_hub",
    "_get_cached_precog_synthesis", "_get_cached_quantum_reasoning",
    "_get_cached_quantum_probability",
    "_get_cached_probability_engine",
]
