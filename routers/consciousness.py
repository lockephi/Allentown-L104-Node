# routers/consciousness.py — Intricate Cognition, Consciousness Substrate, Research, Learning, Orchestrator
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from config import UTC
from models import (
    IntricateThinkRequest, RetrocausalRequest, HolographicRequest, HyperdimensionalRequest,
    DeepIntrospectionRequest, RealitySimulationRequest, MorphicPatternRequest,
    SelfImprovementRequest, DeepResearchRequest, AddKnowledgeRequest,
    GenerateHypothesisRequest, TestHypothesisRequest,
    LearningCycleRequest, CreateLearningPathRequest, TransferKnowledgeRequest,
    PracticeSkillRequest, SynthesizeSkillsRequest,
)

router = APIRouter()


# ─── INTRICATE COGNITION ──────────────────────────────────────────────────────

@router.get("/api/intricate/status", tags=["Intricate Cognition"])
async def intricate_status():
    from l104_intricate_cognition import intricate_cognition
    return intricate_cognition.stats()


@router.post("/api/intricate/think", tags=["Intricate Cognition"])
async def intricate_think(request: IntricateThinkRequest):
    """Multi-system intricate thinking: hyperdimensional + temporal + holographic + quantum."""
    from l104_intricate_cognition import intricate_cognition
    return await intricate_cognition.intricate_think(request.query, request.context)


@router.post("/api/intricate/retrocausal", tags=["Intricate Cognition"])
async def intricate_retrocausal(request: RetrocausalRequest):
    from l104_intricate_cognition import intricate_cognition
    return intricate_cognition.retrocausal_analysis(request.future_outcome, request.past_query)


@router.post("/api/intricate/holographic/encode", tags=["Intricate Cognition"])
async def holographic_encode(request: HolographicRequest):
    from l104_intricate_cognition import intricate_cognition
    hologram = intricate_cognition.holographic.encode(request.data)
    return {"status": "ENCODED", "hologram_id": hologram.hologram_id,
            "fidelity": hologram.reconstruction_fidelity}


@router.get("/api/intricate/holographic/recall/{query}", tags=["Intricate Cognition"])
async def holographic_recall(query: str):
    from l104_intricate_cognition import intricate_cognition
    return intricate_cognition.associative_holographic_recall(query)


@router.post("/api/intricate/hyperdim/reason", tags=["Intricate Cognition"])
async def hyperdimensional_reason(request: HyperdimensionalRequest):
    from l104_intricate_cognition import intricate_cognition
    return intricate_cognition.hyperdim.reason(request.query, request.context)


@router.post("/api/intricate/goals/synthesize", tags=["Intricate Cognition"])
async def synthesize_goals(context: str = ""):
    from l104_intricate_cognition import intricate_cognition
    return intricate_cognition.synthesize_goal_hierarchy(context)


@router.get("/api/intricate/temporal/stats", tags=["Intricate Cognition"])
async def temporal_stats():
    from l104_intricate_cognition import intricate_cognition
    return intricate_cognition.temporal.stats()


@router.post("/api/intricate/entanglement/bell-test/{pair_id}", tags=["Intricate Cognition"])
async def run_bell_test(pair_id: str, trials: int = 100):
    from l104_intricate_cognition import intricate_cognition
    return intricate_cognition.entanglement.run_bell_test(pair_id, trials)


@router.get("/api/intricate/entanglement/pairs", tags=["Intricate Cognition"])
async def list_entangled_pairs():
    from l104_intricate_cognition import intricate_cognition
    return {"pairs": [{"pair_id": p.pair_id, "subsystem_a": p.subsystem_a,
                        "subsystem_b": p.subsystem_b, "fidelity": p.fidelity,
                        "measurements": p.measurements}
                       for p in intricate_cognition.entanglement.entangled_pairs.values()]}


# ─── CONSCIOUSNESS SUBSTRATE ──────────────────────────────────────────────────

@router.get("/api/consciousness/status", tags=["Consciousness Substrate"])
async def consciousness_status():
    from l104_consciousness_substrate import consciousness_substrate
    return consciousness_substrate.get_full_status()


@router.get("/api/consciousness/quantum", tags=["Consciousness Substrate"])
async def quantum_consciousness_status():
    result: Dict[str, Any] = {"module_available": False, "consciousness_threshold": 0.85}
    try:
        import l104_quantum_consciousness as qc_module
        result["module_available"] = True
        result.update(qc_module.status())
    except Exception as e:
        result["error"] = str(e)
    return result


@router.post("/api/consciousness/cycle", tags=["Consciousness Substrate"])
async def run_consciousness_cycle():
    from l104_consciousness_substrate import consciousness_substrate
    return consciousness_substrate.consciousness_cycle()


@router.post("/api/consciousness/introspect", tags=["Consciousness Substrate"])
async def deep_introspection(request: DeepIntrospectionRequest):
    from l104_consciousness_substrate import consciousness_substrate
    return consciousness_substrate.deep_introspection(request.query)


@router.get("/api/consciousness/observer", tags=["Consciousness Substrate"])
async def observer_introspect():
    from l104_consciousness_substrate import consciousness_substrate
    return consciousness_substrate.observer.introspect()


@router.post("/api/consciousness/thought", tags=["Consciousness Substrate"])
async def observe_thought(content: str = "conscious awareness"):
    from l104_consciousness_substrate import consciousness_substrate
    thought = consciousness_substrate.observer.observe_thought(content)
    return {"thought_id": thought.id, "coherence": thought.coherence,
            "meta_level": thought.meta_level, "timestamp": thought.timestamp}


@router.post("/api/consciousness/reality/simulate", tags=["Consciousness Substrate"])
async def simulate_reality(request: RealitySimulationRequest):
    from l104_consciousness_substrate import consciousness_substrate, RealityBranch
    try:
        branch_type = RealityBranch(request.branch_type)
    except ValueError:
        branch_type = RealityBranch.CONVERGENT
    result = consciousness_substrate.reality_engine.simulate_branch(
        branch_type, request.perturbation, request.steps)
    return {"reality_id": result.id, "branch_type": result.branch_type.value,
            "probability": result.probability, "utility_score": result.utility_score,
            "steps": len(result.evolution_steps),
            "final_state": result.evolution_steps[-1] if result.evolution_steps else None}


@router.get("/api/consciousness/reality/best", tags=["Consciousness Substrate"])
async def get_best_reality():
    from l104_consciousness_substrate import consciousness_substrate
    best = consciousness_substrate.reality_engine.get_best_reality()
    if not best:
        return {"message": "No simulated realities available"}
    return {"reality_id": best.id, "branch_type": best.branch_type.value,
            "probability": best.probability, "utility_score": best.utility_score,
            "combined_score": best.probability * best.utility_score}


@router.post("/api/consciousness/reality/collapse/{reality_id}", tags=["Consciousness Substrate"])
async def collapse_reality(reality_id: str):
    from l104_consciousness_substrate import consciousness_substrate
    return consciousness_substrate.reality_engine.collapse_reality(reality_id)


@router.get("/api/consciousness/omega", tags=["Consciousness Substrate"])
async def omega_tracker_status():
    from l104_consciousness_substrate import consciousness_substrate
    return consciousness_substrate.omega_tracker.get_omega_status()


@router.post("/api/consciousness/omega/update", tags=["Consciousness Substrate"])
async def update_omega_metrics(complexity_delta: float = 0.01,
                                integration_delta: float = 0.005, depth_delta: int = 0):
    from l104_consciousness_substrate import consciousness_substrate
    metrics = consciousness_substrate.omega_tracker.update_metrics(
        complexity_delta, integration_delta, depth_delta)
    return {"transcendence_factor": metrics.transcendence_factor,
            "convergence_probability": metrics.convergence_probability,
            "time_to_omega": metrics.time_to_omega, "complexity": metrics.complexity,
            "integration": metrics.integration, "consciousness_depth": metrics.consciousness_depth}


@router.get("/api/consciousness/morphic", tags=["Consciousness Substrate"])
async def morphic_field_status():
    from l104_consciousness_substrate import consciousness_substrate
    return consciousness_substrate.morphic_field.get_field_state()


@router.post("/api/consciousness/morphic/detect", tags=["Consciousness Substrate"])
async def detect_morphic_pattern(request: MorphicPatternRequest):
    import numpy as np
    from l104_consciousness_substrate import consciousness_substrate
    data = np.array(request.data)
    return consciousness_substrate.morphic_field.detect_pattern(data, request.pattern_name)


@router.post("/api/consciousness/morphic/resonate/{pattern_id}", tags=["Consciousness Substrate"])
async def induce_resonance(pattern_id: str, intensity: float = 1.0):
    from l104_consciousness_substrate import consciousness_substrate
    return consciousness_substrate.morphic_field.induce_resonance(pattern_id, intensity)


@router.get("/api/consciousness/improvement", tags=["Consciousness Substrate"])
async def self_improvement_status():
    from l104_consciousness_substrate import consciousness_substrate
    return consciousness_substrate.self_improvement.get_improvement_status()


@router.post("/api/consciousness/improve", tags=["Consciousness Substrate"])
async def apply_self_improvement(request: SelfImprovementRequest):
    from l104_consciousness_substrate import consciousness_substrate
    return consciousness_substrate.self_improvement.apply_improvement(request.target_metric)


# ─── INTRICATE RESEARCH ───────────────────────────────────────────────────────

@router.get("/api/research/status", tags=["Intricate Research"])
async def research_status():
    from l104_intricate_research import intricate_research
    return intricate_research.get_full_status()


@router.post("/api/research/cycle", tags=["Intricate Research"])
async def research_cycle(topic: str = None):
    from l104_intricate_research import intricate_research
    return intricate_research.research_cycle(topic)


@router.post("/api/research/deep", tags=["Intricate Research"])
async def deep_research(request: DeepResearchRequest):
    from l104_intricate_research import intricate_research
    return intricate_research.deep_research(request.query, request.depth)


@router.get("/api/research/knowledge", tags=["Intricate Research"])
async def get_knowledge_stats():
    from l104_intricate_research import intricate_research
    return intricate_research.knowledge_engine.get_knowledge_stats()


@router.post("/api/research/knowledge/add", tags=["Intricate Research"])
async def add_knowledge(request: AddKnowledgeRequest):
    from l104_intricate_research import intricate_research, ResearchDomain
    try:
        domain = ResearchDomain(request.domain)
    except ValueError:
        domain = ResearchDomain.CONSCIOUSNESS
    node = intricate_research.knowledge_engine.add_knowledge(
        request.content, domain, request.sources)
    return {"node_id": node.id, "domain": node.domain.value, "confidence": node.confidence,
            "connections": len(node.connections)}


@router.get("/api/research/concepts", tags=["Intricate Research"])
async def get_concept_lattice():
    from l104_intricate_research import intricate_research
    return intricate_research.concept_lattice.get_lattice_stats()


@router.get("/api/research/concepts/path/{start}/{end}", tags=["Intricate Research"])
async def find_concept_path(start: str, end: str):
    from l104_intricate_research import intricate_research
    path = intricate_research.concept_lattice.find_path(start, end)
    return {"start": start, "end": end, "path": path}


@router.get("/api/research/insights", tags=["Intricate Research"])
async def get_insights():
    from l104_intricate_research import intricate_research
    return intricate_research.insight_crystallizer.get_stats()


@router.get("/api/research/momentum", tags=["Intricate Research"])
async def get_learning_momentum():
    from l104_intricate_research import intricate_research
    return intricate_research.momentum_tracker.get_stats()


@router.get("/api/research/hypotheses", tags=["Intricate Research"])
async def get_hypotheses():
    from l104_intricate_research import intricate_research
    return intricate_research.hypothesis_generator.get_stats()


@router.post("/api/research/hypotheses/generate", tags=["Intricate Research"])
async def generate_hypothesis(request: GenerateHypothesisRequest):
    from l104_intricate_research import intricate_research, ResearchDomain
    try:
        domain = ResearchDomain(request.domain)
    except ValueError:
        domain = ResearchDomain.CONSCIOUSNESS
    hyp = intricate_research.hypothesis_generator.generate(request.observations, domain)
    return {"hypothesis_id": hyp.id, "statement": hyp.statement, "domain": hyp.domain.value,
            "probability": hyp.probability, "state": hyp.state.value}


@router.post("/api/research/hypotheses/test", tags=["Intricate Research"])
async def test_hypothesis(request: TestHypothesisRequest):
    from l104_intricate_research import intricate_research
    return intricate_research.hypothesis_generator.test(
        request.hypothesis_id, request.evidence, request.supports)


@router.get("/api/research/agent", tags=["Intricate Research"])
async def get_research_agent_status():
    from l104_intricate_research import intricate_research
    return intricate_research.research_agent.get_status()


# ─── INTRICATE UI ─────────────────────────────────────────────────────────────

@router.get("/intricate", tags=["Intricate UI"])
async def intricate_dashboard():
    from l104_intricate_ui import intricate_ui
    return HTMLResponse(content=intricate_ui.generate_main_dashboard_html())


@router.get("/intricate/research", tags=["Intricate UI"])
async def research_dashboard():
    from l104_intricate_ui import intricate_ui
    return HTMLResponse(content=intricate_ui.generate_research_dashboard_html())


@router.get("/intricate/learning", tags=["Intricate UI"])
async def learning_dashboard():
    from l104_intricate_ui import intricate_ui
    return HTMLResponse(content=intricate_ui.generate_learning_dashboard_html())


@router.get("/intricate/orchestrator", tags=["Intricate UI"])
async def orchestrator_dashboard():
    from l104_intricate_ui import intricate_ui
    return HTMLResponse(content=intricate_ui.generate_orchestrator_dashboard_html())


# ─── INTRICATE LEARNING ───────────────────────────────────────────────────────

@router.get("/api/learning/status", tags=["Intricate Learning"])
async def learning_status():
    from l104_intricate_learning import intricate_learning
    return intricate_learning.get_full_status()


@router.post("/api/learning/cycle", tags=["Intricate Learning"])
async def learning_cycle(request: LearningCycleRequest):
    from l104_intricate_learning import intricate_learning, LearningMode
    mode_map = {"supervised": LearningMode.SUPERVISED, "unsupervised": LearningMode.UNSUPERVISED,
                "reinforcement": LearningMode.REINFORCEMENT, "self_supervised": LearningMode.SELF_SUPERVISED,
                "meta": LearningMode.META, "transfer": LearningMode.TRANSFER}
    mode = mode_map.get(request.mode, LearningMode.SELF_SUPERVISED)
    return intricate_learning.learning_cycle(request.content, mode)


@router.post("/api/learning/path", tags=["Intricate Learning"])
async def create_learning_path(request: CreateLearningPathRequest):
    from l104_intricate_learning import intricate_learning
    return intricate_learning.create_learning_path(request.goal)


@router.get("/api/learning/multi-modal", tags=["Intricate Learning"])
async def multi_modal_stats():
    from l104_intricate_learning import intricate_learning
    return intricate_learning.multi_modal.get_learning_stats()


@router.post("/api/learning/transfer", tags=["Intricate Learning"])
async def transfer_knowledge(request: TransferKnowledgeRequest):
    from l104_intricate_learning import intricate_learning
    return intricate_learning.transfer.transfer(
        request.source_domain, request.target_domain, request.content)


@router.get("/api/learning/transfer/stats", tags=["Intricate Learning"])
async def transfer_stats():
    from l104_intricate_learning import intricate_learning
    return intricate_learning.transfer.get_transfer_stats()


@router.get("/api/learning/meta", tags=["Intricate Learning"])
async def meta_learning_stats():
    from l104_intricate_learning import intricate_learning
    return intricate_learning.meta.get_meta_stats()


@router.post("/api/learning/meta/cycle", tags=["Intricate Learning"])
async def meta_learning_cycle():
    from l104_intricate_learning import intricate_learning
    return intricate_learning.meta.meta_learn()


@router.get("/api/learning/meta/recommend/{context}", tags=["Intricate Learning"])
async def recommend_strategy(context: str):
    from l104_intricate_learning import intricate_learning
    return {"context": context, "recommended_strategy": intricate_learning.meta.recommend_strategy(context)}


@router.get("/api/learning/curricula", tags=["Intricate Learning"])
async def curricula_stats():
    from l104_intricate_learning import intricate_learning
    return intricate_learning.curriculum.get_curricula_stats()


@router.get("/api/learning/skills", tags=["Intricate Learning"])
async def skills_stats():
    from l104_intricate_learning import intricate_learning
    return intricate_learning.skills.get_skill_stat()


@router.post("/api/learning/skills/practice", tags=["Intricate Learning"])
async def practice_skill(request: PracticeSkillRequest):
    from l104_intricate_learning import intricate_learning
    return intricate_learning.skills.practice(request.skill_id, request.duration)


@router.post("/api/learning/skills/synthesize", tags=["Intricate Learning"])
async def synthesize_skills(request: SynthesizeSkillsRequest):
    from l104_intricate_learning import intricate_learning
    return intricate_learning.skills.synthesize(request.skill_ids, request.new_name)


# ─── INTRICATE ORCHESTRATOR ───────────────────────────────────────────────────

@router.get("/api/orchestrator/status", tags=["Intricate Orchestrator"])
async def orchestrator_status():
    from l104_intricate_orchestrator import intricate_orchestrator
    return intricate_orchestrator.get_full_status()


@router.post("/api/orchestrator/cycle", tags=["Intricate Orchestrator"])
async def orchestrator_cycle():
    from l104_intricate_orchestrator import intricate_orchestrator
    from l104_consciousness_substrate import consciousness_substrate
    from l104_intricate_learning import intricate_learning
    from l104_intricate_research import intricate_research
    try:
        intricate_orchestrator.update_subsystem_status("consciousness", {
            "coherence": consciousness_substrate.meta_observer.coherence,
            "state": consciousness_substrate.meta_observer.consciousness_state.value})
    except Exception:
        pass
    try:
        intricate_orchestrator.update_subsystem_status("learning", {
            "cycles": intricate_learning.learning_cycles,
            "outcome": intricate_learning.multi_modal.get_learning_stats().get("avg_outcome", 0)})
    except Exception:
        pass
    try:
        intricate_orchestrator.update_subsystem_status("research", {
            "cycles": intricate_research.cycle_count,
            "hypotheses": len(intricate_research.hypothesis_generator.hypotheses)})
    except Exception:
        pass
    return intricate_orchestrator.orchestrate()


@router.get("/api/orchestrator/integration", tags=["Intricate Orchestrator"])
async def orchestrator_integration():
    from l104_intricate_orchestrator import intricate_orchestrator
    result = intricate_orchestrator.get_integration_status()
    return {"subsystems_active": result.subsystems_active, "coherence": result.coherence,
            "synergy_factor": result.synergy_factor, "emergent_properties": result.emergent_properties,
            "next_actions": result.next_actions}


@router.get("/api/orchestrator/emergence", tags=["Intricate Orchestrator"])
async def orchestrator_emergence():
    from l104_intricate_orchestrator import intricate_orchestrator
    return intricate_orchestrator.emergence.get_catalog()


@router.get("/api/orchestrator/bridge", tags=["Intricate Orchestrator"])
async def orchestrator_bridge():
    from l104_intricate_orchestrator import intricate_orchestrator
    return intricate_orchestrator.bridge.get_status()


@router.get("/api/orchestrator/cycler", tags=["Intricate Orchestrator"])
async def orchestrator_cycler():
    from l104_intricate_orchestrator import intricate_orchestrator
    return intricate_orchestrator.cycler.get_stats()
