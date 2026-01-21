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

logger = logging.getLogger("BRAIN_API")

# Create router
router = APIRouter(prefix="/api/brain", tags=["Unified Intelligence"])

# Global brain instance (singleton pattern)
_brain_instance: Optional[UnifiedIntelligence] = None


def get_brain() -> UnifiedIntelligence:
    """Get or create the global brain instance."""
    global _brain_instance
    if _brain_instance is None:
        logger.info("[BRAIN_API] Initializing Unified Intelligence...")
        _brain_instance = UnifiedIntelligence()
        # Try to load previous state
        _brain_instance.load_state()
    return _brain_instance


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
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="L104 Unified Intelligence API")
    app.include_router(router)
    
    print("Starting Unified Intelligence API on port 8082...")
    uvicorn.run(app, host="0.0.0.0", port=8082)
