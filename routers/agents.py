"""
L104 Agent System Router — OpenClaw v3.0 DeepSeek agent orchestration.
Provides /api/v14/agents/* endpoints for the running main.py server.
"""
import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("l104.agents")

router = APIRouter(prefix="/api/v14/agents", tags=["agents"])

# Agent system import
try:
    from l104_agent_system import get_orchestrator as _get_agent_orchestrator
    from l104_agent_system import AgentType, AgentPriority, AGENT_DEFAULT_TOOLS
    _agent_system_enabled = True
except ImportError as _err:
    _agent_system_enabled = False
    logger.warning(f"[AGENT_SYSTEM] Import failed: {_err}")

    def _get_agent_orchestrator():
        raise RuntimeError("Agent system not available")


@router.post("/deploy")
async def agent_deploy(request: Request):
    """Deploy an OpenClaw agent with DeepSeek execution."""
    if not _agent_system_enabled:
        return JSONResponse({"error": "Agent system not available"}, status_code=503)
    try:
        body = await request.json()
        orch = _get_agent_orchestrator()
        task = orch.submit(
            prompt=body.get("task", body.get("prompt", "")),
            agent_type=body.get("agent_type", "general"),
            tools=body.get("tools"),
            model=body.get("model", "deepseek-chat"),
            max_rounds=body.get("max_rounds", 25),
            priority=body.get("priority", "normal"),
            timeout=body.get("timeout", 900.0),
            cost_budget=body.get("cost_budget", 0.25),
            depends_on=body.get("depends_on", ""),
            tags=body.get("tags", []),
            source=body.get("source", "api"),
        )
        return {
            "status": "deployed",
            "task_id": task.task_id,
            "agent_type": task.agent_type.value,
            "priority": task.priority.name,
            "tools": task.tools_enabled,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/status")
async def agent_status():
    """Full agent orchestrator status."""
    if not _agent_system_enabled:
        return {"status": "OFFLINE", "error": "Agent system not available"}
    try:
        orch = _get_agent_orchestrator()
        orch.process_mailbox()
        return orch.status()
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


@router.get("/task/{task_id}")
async def agent_task_detail(task_id: str):
    """Get detailed status of a specific agent task."""
    if not _agent_system_enabled:
        return JSONResponse({"error": "Agent system not available"}, status_code=503)
    try:
        orch = _get_agent_orchestrator()
        task = orch.get_task(task_id)
        if not task:
            return JSONResponse({"error": "Task not found"}, status_code=404)
        return task.to_dict()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/chain")
async def agent_chain(request: Request):
    """Deploy a sequential chain of agents with context passing."""
    if not _agent_system_enabled:
        return JSONResponse({"error": "Agent system not available"}, status_code=503)
    try:
        body = await request.json()
        tasks = body.get("tasks", [])
        if not tasks:
            return JSONResponse({"error": "No tasks provided"}, status_code=400)
        orch = _get_agent_orchestrator()
        created = orch.submit_chain(tasks)
        return {
            "status": "chain_deployed",
            "count": len(created),
            "task_ids": [t.task_id for t in created],
            "chain_order": [{"task_id": t.task_id, "agent_type": t.agent_type.value} for t in created],
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/cancel/{task_id}")
async def agent_cancel(task_id: str):
    """Cancel a running agent task."""
    if not _agent_system_enabled:
        return JSONResponse({"error": "Agent system not available"}, status_code=503)
    try:
        orch = _get_agent_orchestrator()
        cancelled = orch.cancel(task_id)
        return {"task_id": task_id, "cancelled": cancelled}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/stats")
async def agent_stats():
    """Aggregate agent statistics."""
    if not _agent_system_enabled:
        return {"error": "Agent system not available"}
    try:
        orch = _get_agent_orchestrator()
        return orch.stats()
    except Exception as e:
        return {"error": str(e)}


@router.get("/history")
async def agent_history(limit: int = 20):
    """Recently completed agent history."""
    if not _agent_system_enabled:
        return {"history": [], "error": "Agent system not available"}
    try:
        orch = _get_agent_orchestrator()
        return {"history": orch.get_history(limit=limit)}
    except Exception as e:
        return {"history": [], "error": str(e)}


@router.get("/types/list")
async def agent_types_list():
    """List all agent types with their default tools."""
    if not _agent_system_enabled:
        return {"types": [], "error": "Agent system not available"}
    try:
        types = []
        for at in AgentType:
            tools = AGENT_DEFAULT_TOOLS.get(at, [])
            types.append({
                "type": at.value,
                "default_tools": tools,
                "tool_count": len(tools),
            })
        return {
            "types": types,
            "priorities": [p.name for p in AgentPriority],
            "total_types": len(types),
        }
    except Exception as e:
        return {"types": [], "error": str(e)}


@router.post("/cleanup")
async def agent_cleanup(max_age_hours: int = 24):
    """Purge old agent state."""
    if not _agent_system_enabled:
        return {"error": "Agent system not available"}
    try:
        orch = _get_agent_orchestrator()
        purged = orch.cleanup(max_age_hours=max_age_hours)
        return {"purged": purged, "max_age_hours": max_age_hours}
    except Exception as e:
        return {"error": str(e)}
