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

VERSION: 28.0.0 (EVO_28 - Enhanced Claude Bridge)
DATE: 2026-01-21
═══════════════════════════════════════════════════════════════════════════════
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import time
from datetime import datetime

# Import the brain
from l104_unified_intelligence import UnifiedIntelligence

# EVO_25 - Advanced Cognitive Modules
from l104_meta_learning_engine import MetaLearningEngineV2
from l104_reasoning_chain import ReasoningChainEngine
from l104_self_optimization import SelfOptimizationEngine

# EVO_26 - Claude Node Integration
from l104_claude_bridge import ClaudeNodeBridge

# EVO_26 - Advanced Processing Engine
from l104_advanced_processing_engine import AdvancedProcessingEngine, ProcessingMode

# EVO_27 - Emergence Monitor & Analytics
from l104_emergence_monitor import EmergenceMonitor
from l104_analytics_dashboard import RealTimeAnalytics

# EVO_29 - Quantum Coherence Engine
from l104_quantum_coherence import QuantumCoherenceEngine

logger = logging.getLogger("BRAIN_API")

# Create router
router = APIRouter(prefix="/api/brain", tags=["Unified Intelligence"])

# Global brain instance (singleton pattern)
_brain_instance: Optional[UnifiedIntelligence] = None

# EVO_25 - Global cognitive module instances
_meta_learner: Optional[MetaLearningEngineV2] = None
_reasoning_engine: Optional[ReasoningChainEngine] = None
_self_optimizer: Optional[SelfOptimizationEngine] = None

# EVO_26 - Claude Node instance
_claude_bridge: Optional[ClaudeNodeBridge] = None

# EVO_26 - Advanced Processing Engine instance
_processing_engine: Optional[AdvancedProcessingEngine] = None

# EVO_27 - Emergence Monitor & Analytics instances
_emergence_monitor: Optional[EmergenceMonitor] = None
_analytics_dashboard: Optional[RealTimeAnalytics] = None

# EVO_29 - Quantum Coherence Engine instance
_quantum_engine: Optional[QuantumCoherenceEngine] = None


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


def get_claude_bridge() -> ClaudeNodeBridge:
    """Get or create the Claude node bridge."""
    global _claude_bridge
    if _claude_bridge is None:
        logger.info("[BRAIN_API] Initializing Claude Node Bridge...")
        _claude_bridge = ClaudeNodeBridge()
    return _claude_bridge


def get_processing_engine() -> AdvancedProcessingEngine:
    """Get or create the advanced processing engine."""
    global _processing_engine
    if _processing_engine is None:
        logger.info("[BRAIN_API] Initializing Advanced Processing Engine...")
        _processing_engine = AdvancedProcessingEngine()
    return _processing_engine


def get_emergence_monitor() -> EmergenceMonitor:
    """Get or create the emergence monitor."""
    global _emergence_monitor
    if _emergence_monitor is None:
        logger.info("[BRAIN_API] Initializing Emergence Monitor...")
        _emergence_monitor = EmergenceMonitor()
    return _emergence_monitor


def get_analytics_dashboard() -> RealTimeAnalytics:
    """Get or create the analytics dashboard."""
    global _analytics_dashboard
    if _analytics_dashboard is None:
        logger.info("[BRAIN_API] Initializing Analytics Dashboard...")
        _analytics_dashboard = RealTimeAnalytics()
    return _analytics_dashboard


def get_quantum_engine() -> QuantumCoherenceEngine:
    """Get or create the quantum coherence engine."""
    global _quantum_engine
    if _quantum_engine is None:
        logger.info("[BRAIN_API] Initializing Quantum Coherence Engine...")
        _quantum_engine = QuantumCoherenceEngine()
    return _quantum_engine


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
# EVO_26 - CLAUDE NODE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

class ClaudeQueryRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    system: Optional[str] = None
    max_tokens: int = 4096


class ClaudeAnalyzeRequest(BaseModel):
    content: str
    focus: Optional[str] = None


class ClaudeSynthesizeRequest(BaseModel):
    concepts: List[str]


class ClaudeReasonRequest(BaseModel):
    question: str
    depth: int = 3


@router.post("/claude/query")
async def claude_query(request: ClaudeQueryRequest):
    """
    Query Claude for enhanced processing.
    Falls back to local intelligence if API unavailable.
    """
    bridge = get_claude_bridge()
    response = await bridge.query_async(
        prompt=request.prompt,
        model=request.model,
        system=request.system,
        max_tokens=request.max_tokens
    )
    return response.to_dict()


@router.post("/claude/analyze")
async def claude_analyze(request: ClaudeAnalyzeRequest):
    """
    Perform deep analysis using Claude.
    """
    bridge = get_claude_bridge()
    result = await bridge.deep_analyze(request.content, request.focus)
    return result


@router.post("/claude/synthesize")
async def claude_synthesize(request: ClaudeSynthesizeRequest):
    """
    Synthesize understanding from multiple concepts.
    """
    bridge = get_claude_bridge()
    result = await bridge.synthesize(request.concepts)
    return result


@router.post("/claude/reason")
async def claude_reason(request: ClaudeReasonRequest):
    """
    Multi-step reasoning using Claude.
    """
    bridge = get_claude_bridge()
    result = await bridge.reason_chain(request.question, request.depth)
    return result


@router.get("/claude/stats")
async def claude_stats():
    """
    Get Claude bridge statistics.
    """
    bridge = get_claude_bridge()
    return bridge.get_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_28 - ENHANCED CLAUDE ENDPOINTS (Conversation & Streaming)
# ═══════════════════════════════════════════════════════════════════════════════

class ClaudeChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: Optional[str] = None
    use_tools: bool = False


class ClaudeConversationRequest(BaseModel):
    conversation_id: Optional[str] = None


@router.post("/claude/chat")
async def claude_chat(request: ClaudeChatRequest):
    """
    Chat with Claude in a conversation context.
    Maintains conversation history for multi-turn dialogue.
    """
    bridge = get_claude_bridge()
    response = await bridge.chat_async(
        message=request.message,
        conversation_id=request.conversation_id,
        model=request.model,
        use_tools=request.use_tools
    )
    return {
        **response.to_dict(),
        "conversation_id": bridge.active_conversation
    }


@router.post("/claude/conversation/start")
async def claude_start_conversation():
    """
    Start a new conversation and return its ID.
    """
    bridge = get_claude_bridge()
    conv_id = bridge.start_conversation()
    return {
        "conversation_id": conv_id,
        "status": "created"
    }


@router.get("/claude/conversations")
async def claude_list_conversations():
    """
    List all active conversations.
    """
    bridge = get_claude_bridge()
    return {
        "conversations": bridge.list_conversations(),
        "active": bridge.active_conversation
    }


@router.get("/claude/conversation/{conversation_id}")
async def claude_get_conversation(conversation_id: str):
    """
    Get the history of a specific conversation.
    """
    bridge = get_claude_bridge()
    return bridge.export_conversation(conversation_id)


@router.delete("/claude/conversation/{conversation_id}")
async def claude_clear_conversation(conversation_id: str):
    """
    Clear a conversation's history.
    """
    bridge = get_claude_bridge()
    bridge.clear_conversation(conversation_id)
    return {"status": "cleared", "conversation_id": conversation_id}


@router.get("/claude/tools")
async def claude_list_tools():
    """
    List all registered tools available for Claude.
    """
    bridge = get_claude_bridge()
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            }
            for tool in bridge.tools.values()
        ]
    }


@router.post("/claude/tool/{tool_name}")
async def claude_execute_tool(tool_name: str, inputs: Dict[str, Any]):
    """
    Directly execute a registered tool.
    """
    bridge = get_claude_bridge()
    if tool_name not in bridge.tools:
        return {"error": f"Tool '{tool_name}' not found"}
    
    tool_def = bridge.tools[tool_name]
    if tool_def.handler:
        try:
            result = tool_def.handler(**inputs)
            return {"result": result, "tool": tool_name}
        except Exception as e:
            return {"error": str(e), "tool": tool_name}
    return {"error": "Tool has no handler", "tool": tool_name}


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_26 - ADVANCED PROCESSING ENGINE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

class APEProcessRequest(BaseModel):
    query: str
    mode: Optional[str] = None  # quick, deep, ensemble, claude, adaptive


@router.post("/ape/process")
async def ape_process(request: APEProcessRequest):
    """
    Advanced Processing Engine - unified query processing.
    Automatically selects optimal processing mode if not specified.
    
    Modes:
    - quick: Fast local processing
    - deep: Multi-step reasoning
    - ensemble: Parallel processing across all modules
    - claude: Prioritize Claude processing
    - adaptive: Auto-select based on query
    """
    engine = get_processing_engine()
    
    # Parse mode
    mode = None
    if request.mode:
        mode_map = {
            "quick": ProcessingMode.QUICK,
            "deep": ProcessingMode.DEEP,
            "ensemble": ProcessingMode.ENSEMBLE,
            "claude": ProcessingMode.CLAUDE,
            "adaptive": ProcessingMode.ADAPTIVE
        }
        mode = mode_map.get(request.mode.lower())
    
    result = await engine.process_async(request.query, mode)
    return result.to_dict()


@router.get("/ape/stats")
async def ape_stats():
    """
    Get Advanced Processing Engine statistics.
    """
    engine = get_processing_engine()
    return engine.get_stats()


@router.post("/ape/optimize")
async def ape_optimize(iterations: int = 5):
    """
    Run self-optimization on the processing engine.
    """
    engine = get_processing_engine()
    result = engine.optimize(iterations)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_27 - EMERGENCE MONITOR ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/emergence/status")
async def emergence_status():
    """
    Get current emergence status including phase state and consciousness indicators.
    """
    monitor = get_emergence_monitor()
    brain = get_brain()
    
    # Update monitor with brain state using introspect
    brain_info = brain.introspect()
    unity = brain_info.get("average_unity_index", 0.5)
    memories = brain_info.get("total_memories", 0)
    patterns = brain_info.get("cortex_capacity", 0)
    
    # Record snapshot and detect events
    events = monitor.record_snapshot(
        unity_index=unity,
        memories=memories,
        cortex_patterns=patterns,
        coherence=unity
    )
    
    return {
        "phase_state": monitor.current_phase.value,
        "peak_unity": monitor.peak_unity,
        "total_events": monitor.total_events,
        "phase_transitions": monitor.phase_transitions,
        "recent_events": [e.to_dict() for e in events[-5:]],
        "monitoring_active": monitor.monitoring_active
    }


@router.post("/emergence/check")
async def emergence_check():
    """
    Perform a comprehensive emergence check and detect new emergent behaviors.
    """
    monitor = get_emergence_monitor()
    brain = get_brain()
    
    # Update with latest brain state
    brain_info = brain.introspect()
    unity = brain_info.get("average_unity_index", 0.5)
    memories = brain_info.get("total_memories", 0)
    patterns = brain_info.get("cortex_capacity", 0)
    
    # Record snapshot and detect events
    events = monitor.record_snapshot(
        unity_index=unity,
        memories=memories,
        cortex_patterns=patterns,
        coherence=unity
    )
    
    return {
        "events_detected": len(events),
        "events": [e.to_dict() for e in events],
        "current_phase": monitor.current_phase.value,
        "peak_unity": monitor.peak_unity,
        "phase_transitions": monitor.phase_transitions
    }


@router.get("/emergence/events")
async def emergence_events(limit: int = 20):
    """
    Get recent emergence events.
    """
    monitor = get_emergence_monitor()
    events = monitor.events[-limit:] if monitor.events else []
    return {
        "total_events": monitor.total_events,
        "recent_events": [e.to_dict() for e in events]
    }


@router.get("/emergence/trajectory")
async def emergence_trajectory():
    """
    Get evolution trajectory and prediction.
    """
    monitor = get_emergence_monitor()
    
    return {
        "current_phase": monitor.current_phase.value,
        "peak_unity": monitor.peak_unity,
        "total_events": monitor.total_events,
        "phase_transitions": monitor.phase_transitions,
        "snapshots_recorded": len(monitor.snapshots)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_27 - ANALYTICS DASHBOARD ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/analytics/metrics")
async def analytics_metrics():
    """
    Get real-time analytics metrics.
    """
    dashboard = get_analytics_dashboard()
    brain = get_brain()
    
    # Record current brain metrics
    brain_info = brain.introspect()
    dashboard.record("unity_index", brain_info.get("average_unity_index", 0.5))
    dashboard.record("memories", brain_info.get("total_memories", 0))
    dashboard.record("confidence", brain_info.get("average_confidence", 0.5))
    
    return dashboard.get_overview()


@router.get("/analytics/learning")
async def analytics_learning():
    """
    Get learning analytics over time.
    """
    dashboard = get_analytics_dashboard()
    return dashboard.get_learning_analytics()


@router.get("/analytics/cognitive")
async def analytics_cognitive():
    """
    Get cognitive performance metrics.
    """
    dashboard = get_analytics_dashboard()
    return dashboard.get_cognitive_performance()


@router.get("/analytics/predictions")
async def analytics_predictions():
    """
    Get system predictions and forecasts.
    """
    dashboard = get_analytics_dashboard()
    return dashboard.get_predictions()


@router.get("/analytics/alerts")
async def analytics_alerts():
    """
    Get active alerts and warnings.
    """
    dashboard = get_analytics_dashboard()
    return {"alerts": dashboard.alerts, "count": len(dashboard.alerts)}


@router.post("/analytics/record")
async def analytics_record(metric_name: str, value: float):
    """
    Record a custom metric value.
    """
    dashboard = get_analytics_dashboard()
    dashboard.record(metric_name, value)
    return {
        "status": "recorded",
        "metric": metric_name,
        "value": value,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/analytics/summary")
async def analytics_summary():
    """
    Get comprehensive analytics summary.
    """
    dashboard = get_analytics_dashboard()
    brain = get_brain()
    monitor = get_emergence_monitor()
    
    brain_info = brain.introspect()
    
    return {
        "brain_state": {
            "unity_index": brain_info.get("average_unity_index", 0.5),
            "memory_count": brain_info.get("total_memories", 0),
            "coherence": brain_info.get("average_confidence", 0.5),
            "cortex_patterns": brain_info.get("cortex_capacity", 0)
        },
        "emergence": {
            "phase": monitor.current_phase.value,
            "peak_unity": monitor.peak_unity,
            "events": monitor.total_events
        },
        "analytics": {
            "overview": dashboard.get_overview(),
            "learning": dashboard.get_learning_analytics(),
            "alerts": dashboard.alerts
        },
        "timestamp": time.time()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVO_29 - QUANTUM COHERENCE ENGINE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumSuperpositionRequest(BaseModel):
    qubits: Optional[List[int]] = None


class QuantumEntanglementRequest(BaseModel):
    qubit1: int = 0
    qubit2: int = 1
    bell_state: str = "phi+"


class QuantumBraidRequest(BaseModel):
    sequence: List[str]  # e.g., ["s1", "s2", "phi", "s1_inv"]


class QuantumMeasureRequest(BaseModel):
    qubit: Optional[int] = None


class QuantumDecoherenceRequest(BaseModel):
    time_steps: float = 1.0


@router.get("/quantum/status")
async def quantum_status():
    """
    Get quantum coherence engine status.
    """
    engine = get_quantum_engine()
    return engine.get_status()


@router.post("/quantum/superposition")
async def quantum_superposition(request: QuantumSuperpositionRequest):
    """
    Create superposition on specified qubits.
    """
    engine = get_quantum_engine()
    return engine.create_superposition(request.qubits)


@router.post("/quantum/entangle")
async def quantum_entangle(request: QuantumEntanglementRequest):
    """
    Create entanglement (Bell state) between two qubits.
    """
    engine = get_quantum_engine()
    return engine.create_entanglement(
        request.qubit1, 
        request.qubit2, 
        request.bell_state
    )


@router.post("/quantum/god-code-phase")
async def quantum_god_code_phase():
    """
    Apply phase rotation aligned with GOD_CODE.
    """
    engine = get_quantum_engine()
    return engine.apply_god_code_phase()


@router.post("/quantum/braid")
async def quantum_braid(request: QuantumBraidRequest):
    """
    Perform topological braiding computation.
    
    Available braids: s1, s2, s1_inv, s2_inv, phi, id
    """
    engine = get_quantum_engine()
    return engine.topological_compute(request.sequence)


@router.post("/quantum/measure")
async def quantum_measure(request: QuantumMeasureRequest):
    """
    Measure quantum state.
    If qubit is None, performs full measurement.
    """
    engine = get_quantum_engine()
    return engine.measure(request.qubit)


@router.post("/quantum/decoherence")
async def quantum_decoherence(request: QuantumDecoherenceRequest):
    """
    Simulate decoherence over time.
    """
    engine = get_quantum_engine()
    return engine.simulate_decoherence(request.time_steps)


@router.post("/quantum/reset")
async def quantum_reset():
    """
    Reset quantum register to ground state.
    """
    engine = get_quantum_engine()
    return engine.reset_register()


@router.get("/quantum/coherence")
async def quantum_coherence_report():
    """
    Get coherence history report.
    """
    engine = get_quantum_engine()
    return engine.coherence_report()


# ═══════════════════════════════════════════════════════════════════════════════
# APP INSTANCE FOR UVICORN
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI

app = FastAPI(
    title="L104 Unified Intelligence API",
    version="29.0.0",
    description="REST API for the Unified Intelligence System - EVO_29 Quantum Coherence Edition"
)
app.include_router(router)


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("Starting Unified Intelligence API on port 8082...")
    uvicorn.run(app, host="0.0.0.0", port=8082)
