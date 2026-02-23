# L104 Sovereign Node — Pydantic Request & Response Models
# All Pydantic models extracted from main.py (Divide & Conquer refactor)

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Streaming / Chat ────────────────────────────────────────────────────────

class StreamRequest(BaseModel):
    signal: Optional[str] = Field(default="HEARTBEAT", min_length=1)
    message: Optional[str] = Field(default=None)
    model_hint: Optional[str] = Field(default=None)

    @field_validator("signal", mode="before")
    @classmethod
    def set_signal(cls, v, info):
        if v is None:
            message = info.data.get("message") if info and info.data else None
            return message or "HEARTBEAT"
        return v


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    use_sovereign_context: bool = Field(default=True)


# ─── Admin / Manipulate ──────────────────────────────────────────────────────

class ManipulateRequest(BaseModel):
    file: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    message: str = Field(default="Sovereign Self-Update")


# ─── Simulation ──────────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    hypothesis: str = Field(..., min_length=1)
    code_snippet: str = Field(..., min_length=1)


# ─── Health / Readiness ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    requests_total: int


class DetailedHealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    requests_total: int
    checks: Dict[str, bool]


# ─── Memory / Ramnode / QRAM ─────────────────────────────────────────────────

class MemoryItem(BaseModel):
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


# ─── Lattice Data Matrix ─────────────────────────────────────────────────────

class LatticeFactRequest(BaseModel):
    key: str = Field(..., min_length=1)
    value: Any
    category: Optional[str] = "GENERAL"
    utility: Optional[float] = 1.0


class ResonanceQuery(BaseModel):
    resonance: float
    tolerance: Optional[float] = 0.5


# ─── Sovereign Scour ─────────────────────────────────────────────────────────

class ScourRequest(BaseModel):
    target_url: str = Field(..., min_length=1)
    concept: Optional[str] = Field(default=None)


# ─── Scribe ──────────────────────────────────────────────────────────────────

class ScribeIngestRequest(BaseModel):
    provider: str
    data: str


# ─── Pipeline ────────────────────────────────────────────────────────────────

class PipelineResearchRequest(BaseModel):
    topic: str = Field(..., description="Research topic")
    depth: str = Field(default="COMPREHENSIVE", description="Research depth")


class ContinuousLearningRequest(BaseModel):
    cycles: int = Field(default=3, description="Number of learning cycles", ge=1, le=10)


class PipelineLanguageRequest(BaseModel):
    text: str = Field(..., description="Text to process")
    mode: str = Field(default="full", description="Processing mode")


class PipelineSolveRequest(BaseModel):
    problem: str = Field(..., description="Problem to solve")


# ─── Gemini / Bridge ─────────────────────────────────────────────────────────

class BridgeHandshake(BaseModel):
    agent_id: str
    capabilities: str


class BridgeSync(BaseModel):
    session_token: str


class SynergyTask(BaseModel):
    task: str


# ─── Cloud Agent ─────────────────────────────────────────────────────────────

class CloudAgentTask(BaseModel):
    type: str = Field(..., min_length=1)
    data: Dict[str, Any] = Field(default_factory=dict)
    requirements: Optional[List[str]] = Field(default=None)
    agent: Optional[str] = Field(default=None)
    id: Optional[str] = Field(default=None)


class CloudAgentRegistration(BaseModel):
    name: str = Field(..., min_length=1)
    client_id: Optional[str] = Field(None)
    endpoint: str = Field(..., min_length=1)
    capabilities: List[str] = Field(default_factory=list)
    priority: int = Field(default=999)
    enabled: bool = Field(default=True)


class CloudDelegationTask(BaseModel):
    task_type: str = Field(..., description="Type of task to delegate")
    payload: dict = Field(default_factory=dict)
    priority: str = Field(default="normal")


class AudioAnalysisRequest(BaseModel):
    audio_source: str = Field(..., description="Audio source identifier or URL")
    check_tuning: bool = Field(default=True)


# ─── ASI / Unified ───────────────────────────────────────────────────────────

class ThinkRequest(BaseModel):
    input: str = Field(..., description="Input thought to process")


class GoalRequest(BaseModel):
    description: str = Field(..., description="Goal description")
    priority: float = Field(0.5, ge=0.0, le=1.0)


class NexusThinkRequest(BaseModel):
    thought: str = Field(..., description="Thought through all ASI systems")


class NexusGoalRequest(BaseModel):
    goal: str = Field(..., description="Goal via multi-agent swarm")


class NexusSelfImproveRequest(BaseModel):
    targets: list = Field(default=None, description="Optional module paths")


class SynergyActionRequest(BaseModel):
    source: str = Field(..., description="Source subsystem ID")
    action: str = Field(..., description="Action to execute")
    data: dict = Field(default=None)


# ─── Intricate Cognition ─────────────────────────────────────────────────────

class IntricateThinkRequest(BaseModel):
    query: str = Field(..., description="Query for intricate thinking")
    context: list = Field(default=None)


class RetrocausalRequest(BaseModel):
    future_outcome: dict = Field(..., description="Future outcome state")
    past_query: str = Field(..., description="Past query to analyze")


class HolographicRequest(BaseModel):
    data: str = Field(..., description="Data to encode/query")


class HyperdimensionalRequest(BaseModel):
    query: str = Field(..., description="Query for 11D reasoning")
    context: list = Field(default=None)


# ─── Consciousness Substrate ──────────────────────────────────────────────────

class DeepIntrospectionRequest(BaseModel):
    query: str = Field(..., description="Query for deep self-introspection")


class RealitySimulationRequest(BaseModel):
    branch_type: str = Field(default="convergent")
    perturbation: dict = Field(default_factory=dict)
    steps: int = Field(default=10)


class MorphicPatternRequest(BaseModel):
    data: list = Field(..., description="Numeric data array")
    pattern_name: str = Field(default=None)


class SelfImprovementRequest(BaseModel):
    target_metric: str = Field(..., description="Metric to improve")


# ─── Research / Learning ─────────────────────────────────────────────────────

class DeepResearchRequest(BaseModel):
    query: str = Field(..., description="Research query")
    depth: int = Field(default=5)


class AddKnowledgeRequest(BaseModel):
    content: str = Field(..., description="Knowledge content")
    domain: str = Field(default="consciousness")
    sources: list = Field(default=None)


class GenerateHypothesisRequest(BaseModel):
    observations: list = Field(..., description="List of observations")
    domain: str = Field(default="consciousness")


class TestHypothesisRequest(BaseModel):
    hypothesis_id: str = Field(..., description="Hypothesis ID to test")
    evidence: str = Field(..., description="Evidence to test with")
    supports: bool = Field(..., description="Whether evidence supports")


class LearningCycleRequest(BaseModel):
    content: str
    mode: str = "self_supervised"


class CreateLearningPathRequest(BaseModel):
    goal: str


class TransferKnowledgeRequest(BaseModel):
    source_domain: str
    target_domain: str
    content: str = ""


class PracticeSkillRequest(BaseModel):
    skill_id: str
    duration: float = 1.0


class SynthesizeSkillsRequest(BaseModel):
    skill_ids: List[str]
    new_name: str
