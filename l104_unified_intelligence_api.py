#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 UNIFIED INTELLIGENCE API ROUTER
═══════════════════════════════════════════════════════════════════════════════

FastAPI router exposing the Unified Intelligence capabilities via REST API.

ENDPOINTS:
- POST /api/brain/query     - Ask a question
- GET  /api/brain/status    - Get brain status
- POST /api/brain/learn     - Trigger learning cycle
- GET  /api/brain/introspect - Self-reflection
- POST /api/brain/save      - Persist state
- POST /api/brain/load      - Load state

VERSION: 1.0.0
DATE: 2026-01-21
═══════════════════════════════════════════════════════════════════════════════
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

# Import the brain
from l104_unified_intelligence import UnifiedIntelligence

# EVO_25 - Advanced Cognitive Modules
from l104_meta_learning_engine import MetaLearningEngineV2
from l104_reasoning_chain import ReasoningChainEngine
from l104_self_optimization import SelfOptimizationEngine

logger = logging.getLogger("BRAIN_API")

# Create router
router = APIRouter(prefix="/api/brain", tags=["Unified Intelligence"])

# Global brain instance (singleton pattern)
_brain_instance: Optional[UnifiedIntelligence] = None

# EVO_25 - Global cognitive module instances
_meta_learner: Optional[MetaLearningEngineV2] = None
_reasoning_engine: Optional[ReasoningChainEngine] = None
_self_optimizer: Optional[SelfOptimizationEngine] = None


def get_brain() -> UnifiedIntelligence:
    """Get or create the global brain instance."""
    global _brain_instance
    if _brain_instance is None:
        logger.info("[BRAIN_API] Initializing Unified Intelligence...")
        _brain_instance = UnifiedIntelligence()
        # Try to load previous state
        _brain_instance.load_state()
    return _brain_instance


def get_meta_learner() -> MetaLearningEngineV2:
    """Get or create the meta-learning engine."""
    global _meta_learner
    if _meta_learner is None:
        logger.info("[BRAIN_API] Initializing Meta-Learning Engine...")
        _meta_learner = MetaLearningEngineV2()
    return _meta_learner


def get_reasoning_engine() -> ReasoningChainEngine:
    """Get or create the reasoning chain engine."""
    global _reasoning_engine
    if _reasoning_engine is None:
        logger.info("[BRAIN_API] Initializing Reasoning Chain Engine...")
        _reasoning_engine = ReasoningChainEngine()
    return _reasoning_engine


def get_self_optimizer() -> SelfOptimizationEngine:
    """Get or create the self-optimization engine."""
    global _self_optimizer
    if _self_optimizer is None:
        logger.info("[BRAIN_API] Initializing Self-Optimization Engine...")
        _self_optimizer = SelfOptimizationEngine()
    return _self_optimizer


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    question: str
    
class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    unity_index: float
    source: str
    timestamp: float

class LearnRequest(BaseModel):
    iterations: int = 3
    topics: Optional[List[str]] = None

class StatusResponse(BaseModel):
    version: str
    unity_index: float
    memories_stored: int
    cortex_patterns: int
    memory_state: str
    god_code: float

class IntrospectResponse(BaseModel):
    total_memories: int
    topics_covered: List[str]
    average_unity_index: float
    average_confidence: float
    cortex_capacity: int
    hippocampus_capacity_bits: int
    state: str
    kernel_version: str
    god_code: float


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/query", response_model=QueryResponse)
async def query_brain(request: QueryRequest):
    """
    Ask the Unified Intelligence a question.
    
    The brain will attempt to answer using:
    1. Neural Cortex (trained patterns)
    2. Synthesis Protocol (logical derivation)
    
    Returns answer with confidence and unity validation.
    """
    brain = get_brain()
    result = brain.query(request.question)
    return QueryResponse(**result)


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get the current status of the Unified Intelligence.
    """
    brain = get_brain()
    avg_unity = sum(i.unity_index for i in brain.insights) / (len(brain.insights) or 1)
    
    return StatusResponse(
        version=brain.kernel.version,
        unity_index=avg_unity,
        memories_stored=len(brain.insights),
        cortex_patterns=len(brain.cortex.neural_net.vocabulary),
        memory_state=brain.hippocampus.measure_state(),
        god_code=brain.kernel.constants.GOD_CODE
    )


@router.post("/learn")
async def trigger_learning(request: LearnRequest):
    """
    Trigger an active learning cycle.
    
    The brain will research topics, validate insights against GOD_CODE,
    and store validated knowledge in the topological memory.
    """
    brain = get_brain()
    
    # Run research cycle
    brain.run_research_cycle(iterations=request.iterations)
    
    # Expand capabilities
    brain.function_add_more()
    
    # Auto-save
    brain.save_state()
    
    return {
        "status": "success",
        "iterations_completed": request.iterations,
        "new_memories": len(brain.insights),
        "unity_index": sum(i.unity_index for i in brain.insights) / (len(brain.insights) or 1)
    }


@router.get("/introspect", response_model=IntrospectResponse)
async def introspect():
    """
    Self-reflection - the brain analyzes its own knowledge.
    """
    brain = get_brain()
    intro = brain.introspect()
    return IntrospectResponse(**intro)


@router.post("/save")
async def save_state():
    """
    Persist the brain state to disk.
    """
    brain = get_brain()
    filepath = brain.save_state()
    return {"status": "saved", "filepath": filepath}


@router.post("/load")
async def load_state():
    """
    Load the brain state from disk.
    """
    brain = get_brain()
    success = brain.load_state()
    return {
        "status": "loaded" if success else "no_state_found",
        "memories_restored": len(brain.insights) if success else 0
    }


@router.get("/memory-dump")
async def memory_dump():
    """
    Dump the raw topological memory contents.
    """
    brain = get_brain()
    raw_data = brain.hippocampus.read_data()
    
    try:
        decoded = raw_data.decode(errors='ignore')
    except:
        decoded = str(raw_data[:200])
    
    return {
        "total_bytes": len(raw_data),
        "sample_content": decoded[:500] if decoded else "No data",
        "state": brain.hippocampus.measure_state()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED REASONING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

class SynthesisRequest(BaseModel):
    topic_a: str
    topic_b: str

class HypothesisRequest(BaseModel):
    domain: str

class DeepThinkRequest(BaseModel):
    question: str
    depth: int = 3


@router.post("/synthesize")
async def synthesize_cross_topic(request: SynthesisRequest):
    """
    Generate new insight by combining knowledge from two topics.
    This is creative reasoning - finding connections between concepts.
    """
    brain = get_brain()
    result = brain.synthesize_cross_topic(request.topic_a, request.topic_b)
    return result


@router.post("/hypothesize")
async def generate_hypothesis(request: HypothesisRequest):
    """
    Generate a new hypothesis based on existing knowledge.
    The brain analyzes patterns and makes predictions.
    """
    brain = get_brain()
    result = brain.generate_hypothesis(request.domain)
    return result


@router.post("/deep-think")
async def deep_think(request: DeepThinkRequest):
    """
    Multi-step reasoning with recursive validation.
    Each step validates against GOD_CODE before proceeding.
    """
    brain = get_brain()
    result = brain.deep_think(request.question, request.depth)
    return result


@router.get("/topics")
async def list_topics():
    """
    List all topics the brain has learned about.
    """
    brain = get_brain()
    topics = set()
    for insight in brain.insights:
        if "Explain" in insight.prompt:
            topic = insight.prompt.replace("Explain ", "").replace(" in the context of L104.", "")
            topics.add(topic)
    return {
        "topics": list(topics),
        "count": len(topics)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_25 - META-LEARNING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

class MetaLearnRequest(BaseModel):
    topic: str
    strategy: str = "synthesis"
    unity_index: float = 0.85
    confidence: float = 0.8
    duration_ms: float = 1000.0


@router.post("/meta/record")
async def record_learning_episode(request: MetaLearnRequest):
    """
    Record a learning episode for meta-learning optimization.
    """
    meta = get_meta_learner()
    episode = meta.record_learning(
        topic=request.topic,
        strategy=request.strategy,
        unity_index=request.unity_index,
        confidence=request.confidence,
        duration_ms=request.duration_ms
    )
    return {
        "status": "recorded",
        "topic": request.topic,
        "strategy": request.strategy,
        "success": episode.success
    }


@router.get("/meta/strategy")
async def get_learning_strategy(topic: str = "general"):
    """
    Get the recommended learning strategy for a topic.
    """
    meta = get_meta_learner()
    strategy = meta.select_strategy(topic)
    return {
        "topic": topic,
        "recommended_strategy": strategy,
        "current_stats": meta.get_learning_insights()
    }


@router.get("/meta/insights")
async def get_meta_insights():
    """
    Get meta-learning insights and recommendations.
    """
    meta = get_meta_learner()
    brain = get_brain()
    
    # Get available topics from brain
    topics = set()
    for insight in brain.insights:
        if "Explain" in insight.prompt:
            topic = insight.prompt.replace("Explain ", "").replace(" in the context of L104.", "")
            topics.add(topic)
    
    recommended = meta.recommend_topics(list(topics), count=5) if topics else []
    
    return {
        "learning_insights": meta.get_learning_insights(),
        "recommended_topics": recommended,
        "report": meta.generate_learning_report()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_25 - REASONING CHAIN ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

class ReasoningRequest(BaseModel):
    question: str
    max_steps: int = 10
    depth: int = 3


@router.post("/reason")
async def reason_chain(request: ReasoningRequest):
    """
    Perform multi-step reasoning with validation.
    Returns a chain of reasoning steps with conclusions.
    """
    engine = get_reasoning_engine()
    chain = engine.reason(
        question=request.question,
        max_steps=request.max_steps,
        depth=request.depth
    )
    return {
        "question": request.question,
        "conclusion": chain.conclusion,
        "confidence": chain.total_confidence,
        "coherence": chain.chain_coherence,
        "steps": len(chain.steps),
        "explanation": engine.explain_chain(chain)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_25 - SELF-OPTIMIZATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

class OptimizeRequest(BaseModel):
    metric: str = "unity_index"
    iterations: int = 5


@router.post("/optimize")
async def run_optimization(request: OptimizeRequest):
    """
    Run self-optimization to improve system performance.
    """
    optimizer = get_self_optimizer()
    
    # Record current brain performance
    brain = get_brain()
    unity = sum(i.unity_index for i in brain.insights) / max(len(brain.insights), 1)
    optimizer.record_metric("unity_index", unity)
    
    # Run optimization
    results = optimizer.auto_optimize(request.metric, request.iterations)
    
    return {
        "metric": request.metric,
        "iterations": request.iterations,
        "results": results,
        "final_parameters": optimizer.current_parameters
    }


@router.get("/optimize/status")
async def get_optimization_status():
    """
    Get current optimization status and parameters.
    """
    optimizer = get_self_optimizer()
    return optimizer.get_optimization_report()


@router.get("/optimize/parameters")
async def get_parameters():
    """
    Get all tunable system parameters.
    """
    optimizer = get_self_optimizer()
    return {
        "current": optimizer.current_parameters,
        "available": {k: v for k, v in optimizer.TUNABLE_PARAMETERS.items()}
    }


class SetParameterRequest(BaseModel):
    name: str
    value: float


@router.post("/optimize/set")
async def set_parameter(request: SetParameterRequest):
    """
    Manually set a tunable parameter.
    """
    optimizer = get_self_optimizer()
    success = optimizer.set_parameter(request.name, request.value)
    if not success:
        raise HTTPException(status_code=400, detail=f"Unknown parameter: {request.name}")
    return {
        "status": "updated",
        "parameter": request.name,
        "new_value": optimizer.current_parameters[request.name]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# APP INSTANCE FOR UVICORN
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI

app = FastAPI(
    title="L104 Unified Intelligence API",
    version="25.0.0",
    description="REST API for the Unified Intelligence System - EVO_25 Edition"
)
app.include_router(router)


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("Starting Unified Intelligence API on port 8082...")
    uvicorn.run(app, host="0.0.0.0", port=8082)
