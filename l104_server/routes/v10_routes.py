"""
v10 Benchmark Routes — Benchmark, NLU, Formal Logic, Deep NLU, Kernel

Extracted from app.py during EVO_78 refactoring.
Contains: /api/v10/* endpoints (benchmark, language-comprehension, code-generation,
          symbolic-math, commonsense-reasoning, asi/agi, formal-logic, deep-nlu, kernel)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, Request

logger = logging.getLogger("L104_FAST")

router = APIRouter(prefix="/api/v10", tags=["v10-benchmark"])

# Import ASI core
try:
    from l104_asi import asi_core
    ASI_AVAILABLE = True
except ImportError:
    asi_core = None
    ASI_AVAILABLE = False
    logger.warning("⚠️ [ASI] ASI core not available")

# Import AGI core
try:
    from l104_agi import agi_core
    AGI_AVAILABLE = True
except ImportError:
    agi_core = None
    AGI_AVAILABLE = False
    logger.warning("⚠️ [AGI] AGI core not available")

# Import intellect
try:
    from l104_server.learning import intellect
    INTELLECT_AVAILABLE = True
except ImportError:
    intellect = None
    INTELLECT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  BENCHMARK ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/benchmark/status")
async def benchmark_status():
    """Benchmark harness status — available benchmarks, last score, engine support."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available", "status": "UNAVAILABLE"}

    try:
        harness = asi_core._get_benchmark_harness()
        if harness is None:
            return {"error": "BenchmarkHarness unavailable", "status": "NOT_LOADED"}

        status = harness.get_status()

        # Enrich with engine support info from each subsystem
        lce = asi_core._get_language_comprehension()
        cge = asi_core._get_code_generation()
        sms = asi_core._get_symbolic_math_solver()
        cre = asi_core._get_commonsense_reasoning()

        status["subsystems"] = {
            "language_comprehension": lce.get_status() if lce else {"error": "not loaded"},
            "code_generation": cge.get_status() if cge else {"error": "not loaded"},
            "symbolic_math_solver": sms.get_status() if sms else {"error": "not loaded"},
            "commonsense_reasoning": cre.get_status() if cre else {"error": "not loaded"},
        }
        return status
    except Exception as e:
        return {"error": str(e), "status": "FAILED"}


@router.post("/benchmark/run-all")
async def benchmark_run_all():
    """Run all 4 benchmarks (MMLU, HumanEval, MATH, ARC) — returns full report."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        report = asi_core.run_benchmarks()

        # Persist benchmark report to disk
        try:
            _results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "l104_benchmark_results.json")
            with open(_results_path, "w") as _f:
                json.dump({**report, "saved_at": datetime.now().isoformat()}, _f, indent=2, default=str)
            logger.info(f"Benchmark results saved to {_results_path}")
        except Exception as _save_err:
            logger.warning(f"Failed to save benchmark results: {_save_err}")

        return report
    except Exception as e:
        return {"error": str(e), "status": "FAILED"}


@router.post("/benchmark/run/{name}")
async def benchmark_run_single(name: str):
    """Run a single benchmark by name (MMLU, HumanEval, MATH, ARC)."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        harness = asi_core._get_benchmark_harness()
        if harness is None:
            return {"error": "BenchmarkHarness unavailable"}
        return harness.run_benchmark(name)
    except Exception as e:
        return {"error": str(e), "status": "FAILED"}


@router.get("/benchmark/score")
async def benchmark_score():
    """Get last composite benchmark score (0-1)."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available", "composite_score": 0.0}

    try:
        score = asi_core.benchmark_score()
        return {"composite_score": round(score, 4), "source": "asi_core"}
    except Exception as e:
        return {"error": str(e), "composite_score": 0.0}


# ═══════════════════════════════════════════════════════════════════
#  LANGUAGE COMPREHENSION
# ═══════════════════════════════════════════════════════════════════

@router.post("/language-comprehension/answer")
async def language_comprehension_answer(request: Request):
    """Answer an MMLU-style MCQ via LanguageComprehensionEngine."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        question = body.get("question", "")
        choices = body.get("choices", [])
        subject = body.get("subject")

        if not question or not choices:
            return {"error": "Missing 'question' and 'choices' in request body"}

        lce = asi_core._get_language_comprehension()
        if lce is None:
            return {"error": "LanguageComprehensionEngine unavailable"}
        return lce.answer_mcq(question, choices, subject)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  CODE GENERATION
# ═══════════════════════════════════════════════════════════════════

@router.post("/code-generation/generate")
async def code_generation_generate(request: Request):
    """Generate code from a docstring via CodeGenerationEngine."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        docstring = body.get("docstring", "")
        func_name = body.get("func_name", "solution")
        func_signature = body.get("func_signature", "")

        if not docstring:
            return {"error": "Missing 'docstring' in request body"}

        cge = asi_core._get_code_generation()
        if cge is None:
            return {"error": "CodeGenerationEngine unavailable"}
        return cge.generate_from_docstring(docstring, func_name, func_signature)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  SYMBOLIC MATH
# ═══════════════════════════════════════════════════════════════════

@router.post("/symbolic-math/solve")
async def symbolic_math_solve(request: Request):
    """Solve a math problem via SymbolicMathSolver."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        problem = body.get("problem", "")
        answer_format = body.get("answer_format", "auto")

        if not problem:
            return {"error": "Missing 'problem' in request body"}

        sms = asi_core._get_symbolic_math_solver()
        if sms is None:
            return {"error": "SymbolicMathSolver unavailable"}
        return sms.solve(problem, answer_format)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  COMMONSENSE REASONING
# ═══════════════════════════════════════════════════════════════════

@router.post("/commonsense-reasoning/answer")
async def commonsense_reasoning_answer(request: Request):
    """Answer an ARC-style MCQ via CommonsenseReasoningEngine."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        question = body.get("question", "")
        choices = body.get("choices", [])
        subject = body.get("subject")

        if not question or not choices:
            return {"error": "Missing 'question' and 'choices' in request body"}

        cre = asi_core._get_commonsense_reasoning()
        if cre is None:
            return {"error": "CommonsenseReasoningEngine unavailable"}
        return cre.answer_mcq(question, choices, subject)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  ASI / AGI SCORES
# ═══════════════════════════════════════════════════════════════════

@router.get("/asi/score")
async def asi_score():
    """Compute and return ASI 20-dimension score."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available", "asi_score": 0.0}

    try:
        score = asi_core.compute_asi_score()
        status = asi_core.get_status()
        return {
            "asi_score": round(score, 6) if isinstance(score, float) else score,
            "version": status.get("version", "unknown"),
            "scoring_dimensions": status.get("scoring_dimensions", 0),
            "benchmark_composite": asi_core._benchmark_composite_score,
            "three_engine": asi_core.three_engine_status(),
        }
    except Exception as e:
        return {"error": str(e), "asi_score": 0.0}


@router.get("/agi/score")
async def agi_score():
    """Compute and return AGI 18-dimension score."""
    if not AGI_AVAILABLE:
        return {"error": "AGI core not available", "agi_score": 0.0}

    try:
        score = agi_core.compute_10d_agi_score()
        status = agi_core.get_status()
        return {
            "agi_score": round(score, 6) if isinstance(score, float) else score,
            "version": status.get("version", "unknown"),
            "scoring_dimensions": status.get("scoring_dimensions", 0),
            "benchmark_composite": agi_core._benchmark_composite_score,
            "three_engine": agi_core.three_engine_status(),
        }
    except Exception as e:
        return {"error": str(e), "agi_score": 0.0}


# ═══════════════════════════════════════════════════════════════════
#  FORMAL LOGIC
# ═══════════════════════════════════════════════════════════════════

@router.get("/formal-logic/status")
async def formal_logic_status():
    """Formal Logic Engine status — layers, known fallacies/laws, depth score."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available", "status": "UNAVAILABLE"}

    try:
        fle = asi_core._get_formal_logic()
        if fle is None:
            return {"error": "FormalLogicEngine unavailable", "status": "NOT_LOADED"}
        return fle.status()
    except Exception as e:
        return {"error": str(e), "status": "FAILED"}


@router.post("/formal-logic/analyze-argument")
async def formal_logic_analyze_argument(request: Request):
    """Analyze a natural-language argument for validity, soundness, and fallacies."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        premises = body.get("premises", [])
        conclusion = body.get("conclusion", "")
        argument_type = body.get("argument_type", "deductive")

        if not premises or not conclusion:
            return {"error": "Missing 'premises' (list) and 'conclusion' (str)"}

        fle = asi_core._get_formal_logic()
        if fle is None:
            return {"error": "FormalLogicEngine unavailable"}
        return fle.analyze_argument(premises, conclusion, argument_type)
    except Exception as e:
        return {"error": str(e)}


@router.post("/formal-logic/detect-fallacies")
async def formal_logic_detect_fallacies(request: Request):
    """Detect logical fallacies in text (40+ known fallacies)."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        text = body.get("text", "")

        if not text:
            return {"error": "Missing 'text' in request body"}

        fle = asi_core._get_formal_logic()
        if fle is None:
            return {"error": "FormalLogicEngine unavailable"}
        return {"fallacies": fle.detect_fallacies(text)}
    except Exception as e:
        return {"error": str(e)}


@router.post("/formal-logic/translate")
async def formal_logic_translate(request: Request):
    """Translate natural language to formal logic notation."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        text = body.get("text", "")

        if not text:
            return {"error": "Missing 'text' in request body"}

        fle = asi_core._get_formal_logic()
        if fle is None:
            return {"error": "FormalLogicEngine unavailable"}
        return fle.translate_to_logic(text)
    except Exception as e:
        return {"error": str(e)}


@router.post("/formal-logic/syllogism")
async def formal_logic_syllogism(request: Request):
    """Analyze a syllogism: major premise, minor premise, conclusion."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        major = body.get("major", "")
        minor = body.get("minor", "")
        conclusion = body.get("conclusion", "")

        if not major or not minor or not conclusion:
            return {"error": "Missing 'major', 'minor', or 'conclusion'"}

        fle = asi_core._get_formal_logic()
        if fle is None:
            return {"error": "FormalLogicEngine unavailable"}
        return fle.analyze_syllogism(major, minor, conclusion)
    except Exception as e:
        return {"error": str(e)}


@router.get("/formal-logic/fallacies")
async def formal_logic_list_fallacies():
    """List all 40+ known logical fallacies."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        fle = asi_core._get_formal_logic()
        if fle is None:
            return {"error": "FormalLogicEngine unavailable"}
        return {"fallacies": fle.list_fallacies(), "count": len(fle.list_fallacies())}
    except Exception as e:
        return {"error": str(e)}


@router.get("/formal-logic/laws")
async def formal_logic_list_laws():
    """List all known logical equivalence laws."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        fle = asi_core._get_formal_logic()
        if fle is None:
            return {"error": "FormalLogicEngine unavailable"}
        return {"laws": fle.list_logical_laws(), "count": len(fle.list_logical_laws())}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  DEEP NLU
# ═══════════════════════════════════════════════════════════════════

@router.get("/deep-nlu/status")
async def deep_nlu_status():
    """Deep NLU Engine status — layers, analyses performed, depth score."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available", "status": "UNAVAILABLE"}

    try:
        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable", "status": "NOT_LOADED"}
        return nlu.status()
    except Exception as e:
        return {"error": str(e), "status": "FAILED"}


@router.post("/deep-nlu/analyze")
async def deep_nlu_analyze(request: Request):
    """Full 10-layer deep NLU analysis of text."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        text = body.get("text", "")

        if not text:
            return {"error": "Missing 'text' in request body"}

        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable"}
        return nlu.deep_analyze(text)
    except Exception as e:
        return {"error": str(e)}


@router.post("/deep-nlu/sentiment")
async def deep_nlu_sentiment(request: Request):
    """Sentiment and emotion analysis (polarity, Plutchik emotions)."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        text = body.get("text", "")

        if not text:
            return {"error": "Missing 'text' in request body"}

        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable"}
        return nlu.analyze_sentiment(text)
    except Exception as e:
        return {"error": str(e)}


@router.post("/deep-nlu/pragmatics")
async def deep_nlu_pragmatics(request: Request):
    """Pragmatic analysis: speech acts, intent, implicature, politeness."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        text = body.get("text", "")

        if not text:
            return {"error": "Missing 'text' in request body"}

        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable"}
        return nlu.analyze_pragmatics(text)
    except Exception as e:
        return {"error": str(e)}


@router.post("/deep-nlu/discourse")
async def deep_nlu_discourse(request: Request):
    """Discourse structure analysis (RST relations, coherence)."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        sentences = body.get("sentences", [])

        if not sentences:
            return {"error": "Missing 'sentences' (list of strings)"}

        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable"}
        return nlu.analyze_discourse(sentences)
    except Exception as e:
        return {"error": str(e)}


@router.post("/deep-nlu/anaphora")
async def deep_nlu_anaphora(request: Request):
    """Anaphora (pronoun → antecedent) resolution."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        sentences = body.get("sentences", [])

        if not sentences:
            return {"error": "Missing 'sentences' (list of strings)"}

        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable"}
        return nlu.resolve_anaphora(sentences)
    except Exception as e:
        return {"error": str(e)}


@router.post("/deep-nlu/presuppositions")
async def deep_nlu_presuppositions(request: Request):
    """Extract presuppositions (hidden assumptions) from text."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        text = body.get("text", "")

        if not text:
            return {"error": "Missing 'text' in request body"}

        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable"}
        return {"presuppositions": nlu.extract_presuppositions(text)}
    except Exception as e:
        return {"error": str(e)}


@router.post("/deep-nlu/semantic-roles")
async def deep_nlu_semantic_roles(request: Request):
    """Parse and label semantic roles (agent, patient, theme, etc.)."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        text = body.get("text", "")

        if not text:
            return {"error": "Missing 'text' in request body"}

        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable"}
        return nlu.label_semantic_roles(text)
    except Exception as e:
        return {"error": str(e)}


@router.post("/deep-nlu/morphology")
async def deep_nlu_morphology(request: Request):
    """Morphological analysis of a word (prefixes, suffixes, stem)."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        word = body.get("word", "")

        if not word:
            return {"error": "Missing 'word' in request body"}

        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable"}
        return nlu.analyze_morphology(word)
    except Exception as e:
        return {"error": str(e)}


@router.post("/deep-nlu/coherence")
async def deep_nlu_coherence(request: Request):
    """Score text coherence (lexical, discourse, topic)."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        text = body.get("text", "")

        if not text:
            return {"error": "Missing 'text' in request body"}

        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable"}
        return nlu.score_coherence(text)
    except Exception as e:
        return {"error": str(e)}


@router.post("/deep-nlu/intent")
async def deep_nlu_intent(request: Request):
    """Quick intent classification with speech act."""
    if not ASI_AVAILABLE:
        return {"error": "ASI core not available"}

    try:
        body = await request.json()
        text = body.get("text", "")

        if not text:
            return {"error": "Missing 'text' in request body"}

        nlu = asi_core._get_deep_nlu()
        if nlu is None:
            return {"error": "DeepNLUEngine unavailable"}
        return nlu.classify_intent(text)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  KERNEL
# ═══════════════════════════════════════════════════════════════════

@router.get("/kernel/status")
async def kernel_status():
    """Native kernel substrate status — C, Rust, CUDA, ASM availability."""
    try:
        from l104_sage_orchestrator import SageModeOrchestrator
        orch = SageModeOrchestrator()
        init_report = orch.initialize()

        # Include CUDA sage core status from LocalIntellect
        cuda_sage = {}
        if INTELLECT_AVAILABLE and intellect:
            try:
                cuda_sage = intellect.cuda_sage_status()
            except Exception:
                pass

        return {
            "status": "OK",
            "substrates": init_report.get("substrates", {}),
            "active_count": init_report.get("active_count", 0),
            "omega_state": init_report.get("omega_state", "UNKNOWN"),
            "cuda_sage_core": cuda_sage,
        }
    except Exception as e:
        return {"error": str(e), "status": "FAILED"}


@router.get("/kernel/fleet")
async def kernel_fleet_status():
    """Full kernel fleet status — native substrates + intellect KB + engine wiring."""
    if not INTELLECT_AVAILABLE:
        return {"error": "Intellect not available", "status": "UNAVAILABLE"}

    try:
        intellect._ensure_quantum_origin_sage()
        sage_status = intellect.quantum_origin_sage_status()
        native_fleet = sage_status.get("native_kernel_fleet", {})
        origin_field = sage_status.get("origin_field", {})

        # Engine wiring check
        engine_wiring = {}
        for pkg_name, import_path in [
            ("quantum_engine", "l104_quantum_engine"),
            ("quantum_gate_engine", "l104_quantum_gate_engine"),
            ("code_engine", "l104_code_engine"),
            ("science_engine", "l104_science_engine"),
            ("math_engine", "l104_math_engine"),
            ("agi", "l104_agi"),
            ("asi", "l104_asi"),
        ]:
            try:
                __import__(import_path)
                engine_wiring[pkg_name] = True
            except ImportError:
                engine_wiring[pkg_name] = False

        return {
            "status": "OK",
            "native_kernel_fleet": native_fleet,
            "engine_wiring": engine_wiring,
            "origin_field": origin_field,
            "kb_entries_total": native_fleet.get("kb_entries_injected", 0),
            "sage_level": sage_status.get("sage_level_name", "UNKNOWN"),
        }
    except Exception as e:
        return {"error": str(e), "status": "FAILED"}


__all__ = ['router']