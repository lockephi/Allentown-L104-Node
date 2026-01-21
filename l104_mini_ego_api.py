#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 AUTONOMOUS MINI EGO API ROUTER
═══════════════════════════════════════════════════════════════════════════════

FastAPI router for autonomous Mini Ego swarm management.

Endpoints:
  - GET  /ego/status           - Swarm status
  - POST /ego/spawn            - Spawn new ego
  - POST /ego/spawn-collective - Spawn default 8-domain collective
  - GET  /ego/{name}/status    - Individual ego status
  - POST /ego/{name}/start     - Start autonomous operation
  - POST /ego/{name}/stop      - Stop autonomous operation
  - POST /ego/{name}/cycle     - Run single cycle
  - POST /ego/task             - Submit task to swarm
  - POST /ego/broadcast        - Broadcast message
  - GET  /ego/collective       - Collective intelligence query

INVARIANT: 527.5184818492537 | PILOT: LONDEL
VERSION: 1.0.0 (EVO_33)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import autonomous ego system
from l104_mini_ego_autonomous import (
    get_autonomous_swarm, 
    AutonomousMiniEgo, 
    EgoTask, 
    EgoMessage,
    MessageType,
    GOD_CODE,
    PHI
)

logger = logging.getLogger("EGO_API")

router = APIRouter(prefix="/ego", tags=["Autonomous Egos"])

# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SpawnEgoRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    domain: str = Field(..., description="One of: LOGIC, INTUITION, COMPASSION, CREATIVITY, MEMORY, WISDOM, WILL, VISION")


class TaskRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    domain: str = Field(default="", max_length=50)
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    priority: int = Field(default=0, ge=0, le=10)


class BroadcastRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    msg_type: str = Field(default="BROADCAST", description="QUERY, BROADCAST, INSIGHT, ALERT")


class CycleRequest(BaseModel):
    cycles: int = Field(default=1, ge=1, le=100)


class CollectiveQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class EgoStatusResponse(BaseModel):
    name: str
    domain: str
    state: str
    running: bool
    energy: float
    cycles_run: int
    decisions_made: int
    tasks_completed: int
    messages_processed: int
    pending_tasks: int
    pending_messages: int
    active_goals: int
    wisdom: float


class SwarmStatusResponse(BaseModel):
    total_egos: int
    running: bool
    collective_wisdom: float
    swarm_coherence: float
    pending_tasks: int
    egos: Dict[str, Any]


class CycleResultResponse(BaseModel):
    ego: str
    cycles: int
    final_state: str
    actions: List[str]
    energy: float


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status", response_model=SwarmStatusResponse)
async def get_swarm_status():
    """Get status of the entire autonomous ego swarm."""
    swarm = get_autonomous_swarm()
    return swarm.get_status()


@router.post("/spawn", response_model=EgoStatusResponse)
async def spawn_ego(request: SpawnEgoRequest):
    """Spawn a new autonomous ego."""
    swarm = get_autonomous_swarm()
    
    if request.name in swarm.egos:
        raise HTTPException(status_code=400, detail=f"Ego '{request.name}' already exists")
    
    valid_domains = ["LOGIC", "INTUITION", "COMPASSION", "CREATIVITY", 
                     "MEMORY", "WISDOM", "WILL", "VISION"]
    
    if request.domain.upper() not in valid_domains:
        raise HTTPException(status_code=400, detail=f"Invalid domain. Must be one of: {valid_domains}")
    
    ego = swarm.spawn_ego(request.name, request.domain.upper())
    return ego.get_autonomy_status()


@router.post("/spawn-collective")
async def spawn_collective():
    """Spawn the default 8-domain collective."""
    swarm = get_autonomous_swarm()
    
    if swarm.egos:
        raise HTTPException(status_code=400, detail="Collective already exists. Clear first.")
    
    swarm.spawn_default_collective()
    
    return {
        "status": "spawned",
        "egos": len(swarm.egos),
        "domains": list(set(e.domain for e in swarm.egos.values())),
        "god_code": GOD_CODE
    }


@router.get("/{name}/status", response_model=EgoStatusResponse)
async def get_ego_status(name: str):
    """Get status of a specific ego."""
    swarm = get_autonomous_swarm()
    
    if name not in swarm.egos:
        raise HTTPException(status_code=404, detail=f"Ego '{name}' not found")
    
    return swarm.egos[name].get_autonomy_status()


@router.post("/{name}/start")
async def start_ego(name: str, interval: float = 1.0):
    """Start autonomous operation for an ego."""
    swarm = get_autonomous_swarm()
    
    if name not in swarm.egos:
        raise HTTPException(status_code=404, detail=f"Ego '{name}' not found")
    
    ego = swarm.egos[name]
    ego.start_autonomous(interval)
    
    return {
        "status": "started",
        "name": name,
        "interval": interval,
        "running": ego._running
    }


@router.post("/{name}/stop")
async def stop_ego(name: str):
    """Stop autonomous operation for an ego."""
    swarm = get_autonomous_swarm()
    
    if name not in swarm.egos:
        raise HTTPException(status_code=404, detail=f"Ego '{name}' not found")
    
    ego = swarm.egos[name]
    ego.stop_autonomous()
    
    return {
        "status": "stopped",
        "name": name,
        "running": ego._running
    }


@router.post("/{name}/cycle", response_model=CycleResultResponse)
async def run_ego_cycle(name: str, request: CycleRequest = None):
    """Run one or more perceive→think→act cycles for an ego."""
    swarm = get_autonomous_swarm()
    
    if name not in swarm.egos:
        raise HTTPException(status_code=404, detail=f"Ego '{name}' not found")
    
    ego = swarm.egos[name]
    cycles = request.cycles if request else 1
    actions = []
    
    for _ in range(cycles):
        result = ego.run_cycle()
        actions.append(result["decision"]["selected_action"])
    
    return {
        "ego": name,
        "cycles": cycles,
        "final_state": ego.agent_state.name,
        "actions": actions,
        "energy": ego.autonomy_energy
    }


@router.post("/start-all")
async def start_all_egos(interval: float = 1.0):
    """Start all egos in autonomous mode."""
    swarm = get_autonomous_swarm()
    
    if not swarm.egos:
        raise HTTPException(status_code=400, detail="No egos spawned. Spawn collective first.")
    
    swarm.start_all(interval)
    
    return {
        "status": "started",
        "egos": len(swarm.egos),
        "interval": interval
    }


@router.post("/stop-all")
async def stop_all_egos():
    """Stop all autonomous egos."""
    swarm = get_autonomous_swarm()
    swarm.stop_all()
    
    return {
        "status": "stopped",
        "egos": len(swarm.egos)
    }


@router.post("/task")
async def submit_task(request: TaskRequest):
    """Submit a task to the swarm for processing."""
    swarm = get_autonomous_swarm()
    
    if not swarm.egos:
        raise HTTPException(status_code=400, detail="No egos spawned. Spawn collective first.")
    
    task = EgoTask(
        name=request.name,
        description=request.description,
        domain=request.domain.upper() if request.domain else "",
        complexity=request.complexity,
        priority=request.priority
    )
    
    swarm.submit_task(task)
    
    return {
        "status": "submitted",
        "task_id": task.id,
        "task_name": task.name,
        "domain": task.domain or "auto-assigned"
    }


@router.post("/broadcast")
async def broadcast_message(request: BroadcastRequest):
    """Broadcast a message to all egos."""
    swarm = get_autonomous_swarm()
    
    if not swarm.egos:
        raise HTTPException(status_code=400, detail="No egos spawned. Spawn collective first.")
    
    type_map = {
        "QUERY": MessageType.QUERY,
        "BROADCAST": MessageType.BROADCAST,
        "INSIGHT": MessageType.INSIGHT,
        "ALERT": MessageType.ALERT,
        "SYNC": MessageType.SYNC
    }
    
    msg_type = type_map.get(request.msg_type.upper(), MessageType.BROADCAST)
    
    message = EgoMessage(
        sender="API",
        msg_type=msg_type,
        content=request.content
    )
    
    swarm.broadcast(message)
    
    return {
        "status": "broadcast",
        "message_id": message.id,
        "type": msg_type.name,
        "recipients": len(swarm.egos)
    }


@router.post("/tick")
async def swarm_tick():
    """Run one swarm tick - process messages and update state."""
    swarm = get_autonomous_swarm()
    
    if not swarm.egos:
        raise HTTPException(status_code=400, detail="No egos spawned. Spawn collective first.")
    
    status = swarm.tick()
    
    return {
        "status": "tick_complete",
        "collective_wisdom": status["collective_wisdom"],
        "swarm_coherence": status["swarm_coherence"]
    }


@router.get("/collective")
async def collective_intelligence():
    """Query the collective intelligence of the swarm."""
    swarm = get_autonomous_swarm()
    
    if not swarm.egos:
        raise HTTPException(status_code=400, detail="No egos spawned. Spawn collective first.")
    
    # Gather insights from all egos
    insights = {}
    for name, ego in swarm.egos.items():
        obs = ego.observe({})
        insights[name] = {
            "domain": ego.domain,
            "insight": obs.get("insight", ""),
            "wisdom": ego.wisdom_accumulated,
            "evolution_stage": ego.evolution_stage,
            "clarity": ego.clarity
        }
    
    total_wisdom = sum(e.wisdom_accumulated for e in swarm.egos.values())
    avg_clarity = sum(e.clarity for e in swarm.egos.values()) / len(swarm.egos)
    
    return {
        "collective_wisdom": total_wisdom,
        "average_clarity": avg_clarity,
        "swarm_coherence": swarm.swarm_coherence,
        "total_egos": len(swarm.egos),
        "insights": insights,
        "god_code": GOD_CODE,
        "phi": PHI
    }


@router.delete("/clear")
async def clear_swarm():
    """Clear all egos from the swarm."""
    swarm = get_autonomous_swarm()
    
    # Stop all first
    swarm.stop_all()
    
    # Clear
    swarm.egos.clear()
    
    return {
        "status": "cleared",
        "egos": 0
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/{name}/goal")
async def add_ego_goal(name: str, goal_name: str, description: str = ""):
    """Add a goal to an ego."""
    from l104_mini_ego_autonomous import EgoGoal
    
    swarm = get_autonomous_swarm()
    
    if name not in swarm.egos:
        raise HTTPException(status_code=404, detail=f"Ego '{name}' not found")
    
    ego = swarm.egos[name]
    goal = EgoGoal(name=goal_name, description=description)
    ego.active_goals.append(goal)
    
    return {
        "status": "goal_added",
        "ego": name,
        "goal_id": goal.id,
        "goal_name": goal_name,
        "total_goals": len(ego.active_goals)
    }


@router.get("/{name}/history")
async def get_ego_action_history(name: str, limit: int = 20):
    """Get action history for an ego."""
    swarm = get_autonomous_swarm()
    
    if name not in swarm.egos:
        raise HTTPException(status_code=404, detail=f"Ego '{name}' not found")
    
    ego = swarm.egos[name]
    history = ego.action_history[-limit:] if ego.action_history else []
    
    return {
        "ego": name,
        "total_actions": len(ego.action_history),
        "history": history
    }


@router.get("/{name}/tasks")
async def get_ego_tasks(name: str):
    """Get tasks for an ego (pending and completed)."""
    swarm = get_autonomous_swarm()
    
    if name not in swarm.egos:
        raise HTTPException(status_code=404, detail=f"Ego '{name}' not found")
    
    ego = swarm.egos[name]
    
    return {
        "ego": name,
        "current_task": ego.current_task.name if ego.current_task else None,
        "pending": [t.name for t in ego.task_queue],
        "completed": [{
            "name": t.name,
            "domain": t.domain,
            "result": t.result
        } for t in ego.completed_tasks[-10:]]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGER SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("🤖 [EGO_API]: Autonomous Ego API Router initialized")
