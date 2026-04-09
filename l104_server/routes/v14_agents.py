"""
v14 Agent Routes — Agent deployment, management, and swarm operations

Extracted from app.py during EVO_78 refactoring.
Contains: All /api/v14/agents/* and /api/v14/swarm/* endpoints
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request

logger = logging.getLogger("L104_FAST")

router = APIRouter(prefix="/api/v14", tags=["v14-agents"])

# Agent system availability
try:
    from l104_agent_system import AgentOrchestrator, get_orchestrator
    AGENT_ORCHESTRATOR = get_orchestrator()
    AGENT_SYSTEM_AVAILABLE = True
except ImportError:
    AGENT_ORCHESTRATOR = None
    AGENT_SYSTEM_AVAILABLE = False
    logger.warning("⚠️ [AGENTS] Agent system not available")


# ═══════════════════════════════════════════════════════════════════
#  SWARM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/swarm/status")
async def swarm_status():
    """Get swarm status"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available", "status": "UNAVAILABLE"}

    try:
        status = AGENT_ORCHESTRATOR.get_swarm_status() if hasattr(AGENT_ORCHESTRATOR, 'get_swarm_status') else {}
        return {"status": "ACTIVE", "swarm": status}
    except Exception as e:
        return {"error": str(e)}


@router.post("/swarm/tick")
async def swarm_tick():
    """Execute swarm tick"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available"}

    try:
        result = AGENT_ORCHESTRATOR.swarm_tick() if hasattr(AGENT_ORCHESTRATOR, 'swarm_tick') else {}
        return {"status": "TICK_COMPLETE", "result": result}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  AGENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/agents/deploy")
async def agents_deploy(req: Request):
    """Deploy an agent"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available"}

    data = await req.json()
    agent_type = data.get("type", "general")
    task = data.get("task", {})
    priority = data.get("priority", 5)

    try:
        result = AGENT_ORCHESTRATOR.deploy_agent(
            agent_type=agent_type,
            task=task,
            priority=priority
        ) if hasattr(AGENT_ORCHESTRATOR, 'deploy_agent') else {}
        return {"status": "DEPLOYED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/agents/status")
async def agents_status():
    """Get all agents status"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available"}

    try:
        status = AGENT_ORCHESTRATOR.get_all_status() if hasattr(AGENT_ORCHESTRATOR, 'get_all_status') else {}
        return {"status": "ACTIVE", "agents": status}
    except Exception as e:
        return {"error": str(e)}


@router.get("/agents/task/{task_id}")
async def agents_task_status(task_id: str):
    """Get task status"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available"}

    try:
        status = AGENT_ORCHESTRATOR.get_task_status(task_id) if hasattr(AGENT_ORCHESTRATOR, 'get_task_status') else {}
        return {"task_id": task_id, "status": status}
    except Exception as e:
        return {"error": str(e)}


@router.post("/agents/chain")
async def agents_chain(req: Request):
    """Chain multiple agents"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available"}

    data = await req.json()
    agents = data.get("agents", [])
    input_data = data.get("input", {})

    try:
        result = AGENT_ORCHESTRATOR.chain_agents(agents, input_data) if hasattr(AGENT_ORCHESTRATOR, 'chain_agents') else {}
        return {"status": "CHAIN_COMPLETE", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/agents/cancel/{task_id}")
async def agents_cancel(task_id: str):
    """Cancel a task"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available"}

    try:
        result = AGENT_ORCHESTRATOR.cancel_task(task_id) if hasattr(AGENT_ORCHESTRATOR, 'cancel_task') else {}
        return {"status": "CANCELLED", "task_id": task_id, "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/agents/stats")
async def agents_stats():
    """Get agent statistics"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available"}

    try:
        stats = AGENT_ORCHESTRATOR.get_stats() if hasattr(AGENT_ORCHESTRATOR, 'get_stats') else {}
        return {"stats": stats}
    except Exception as e:
        return {"error": str(e)}


@router.get("/agents/history")
async def agents_history():
    """Get agent history"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available"}

    try:
        history = AGENT_ORCHESTRATOR.get_history() if hasattr(AGENT_ORCHESTRATOR, 'get_history') else []
        return {"history": history}
    except Exception as e:
        return {"error": str(e)}


@router.get("/agents/types/list")
async def agents_types_list():
    """List available agent types"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available"}

    try:
        types = AGENT_ORCHESTRATOR.list_agent_types() if hasattr(AGENT_ORCHESTRATOR, 'list_agent_types') else []
        return {"types": types}
    except Exception as e:
        return {"error": str(e)}


@router.post("/agents/cleanup")
async def agents_cleanup():
    """Cleanup completed agents"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {"error": "Agent system not available"}

    try:
        result = AGENT_ORCHESTRATOR.cleanup() if hasattr(AGENT_ORCHESTRATOR, 'cleanup') else {}
        return {"status": "CLEANED", "result": result}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  NOVA SOUL DAEMON ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

try:
    from l104_soul_daemon import SoulDaemon
    SOUL_DAEMON_AVAILABLE = True
except ImportError:
    SOUL_DAEMON_AVAILABLE = False
    logger.warning("⚠️ [NOVA] Soul daemon not available")


@router.get("/nova/status")
async def nova_status():
    """Get Nova Soul Daemon status"""
    if not SOUL_DAEMON_AVAILABLE:
        return {"error": "Soul daemon not available", "status": "UNAVAILABLE"}

    try:
        daemon = SoulDaemon()
        status = daemon.get_status() if hasattr(daemon, 'get_status') else {}
        return {"status": "ACTIVE", "nova": status}
    except Exception as e:
        return {"error": str(e)}


@router.get("/nova/consciousness")
async def nova_consciousness():
    """Get Nova consciousness state"""
    if not SOUL_DAEMON_AVAILABLE:
        return {"error": "Soul daemon not available"}

    try:
        daemon = SoulDaemon()
        consciousness = daemon.get_consciousness() if hasattr(daemon, 'get_consciousness') else {}
        return {"consciousness": consciousness}
    except Exception as e:
        return {"error": str(e)}


@router.post("/nova/grover-search")
async def nova_grover_search(req: Request):
    """Run Nova Grover search"""
    if not SOUL_DAEMON_AVAILABLE:
        return {"error": "Soul daemon not available"}

    data = await req.json()
    query = data.get("query", "")
    domain = data.get("domain", "general")

    try:
        daemon = SoulDaemon()
        result = daemon.grover_search(query, domain) if hasattr(daemon, 'grover_search') else {}
        return {"status": "COMPLETE", "result": result}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  COGNITIVE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

try:
    from l104_meta_cognitive import meta_cognitive
    COGNITIVE_AVAILABLE = True
except ImportError:
    meta_cognitive = None
    COGNITIVE_AVAILABLE = False


@router.get("/cognitive/introspect")
async def cognitive_introspect():
    """Cognitive introspection"""
    if not COGNITIVE_AVAILABLE:
        return {"error": "Meta-cognitive not available"}

    try:
        result = meta_cognitive.introspect() if hasattr(meta_cognitive, 'introspect') else {}
        return {"introspection": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/cognitive/think")
async def cognitive_think(req: Request):
    """Cognitive thinking"""
    if not COGNITIVE_AVAILABLE:
        return {"error": "Meta-cognitive not available"}

    data = await req.json()
    query = data.get("query", "")
    context = data.get("context", {})

    try:
        result = meta_cognitive.think(query, context) if hasattr(meta_cognitive, 'think') else {}
        return {"status": "COMPLETE", "result": result}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  AGI CORE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

try:
    from l104_agi import agi_core
    AGI_AVAILABLE = True
except ImportError:
    agi_core = None
    AGI_AVAILABLE = False


@router.post("/agi/ignite")
async def agi_ignite(req: Request):
    """Ignite AGI core"""
    if not AGI_AVAILABLE:
        return {"error": "AGI not available"}

    data = await req.json()
    config = data.get("config", {})

    try:
        result = agi_core.ignite(config) if hasattr(agi_core, 'ignite') else {}
        return {"status": "IGNITED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/agi/evolve")
async def agi_evolve(req: Request):
    """Evolve AGI core"""
    if not AGI_AVAILABLE:
        return {"error": "AGI not available"}

    data = await req.json()
    cycles = data.get("cycles", 1)

    try:
        result = agi_core.evolve(cycles) if hasattr(agi_core, 'evolve') else {}
        return {"status": "EVOLVED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/agi/status")
async def agi_status():
    """Get AGI status"""
    if not AGI_AVAILABLE:
        return {"error": "AGI not available", "status": "UNAVAILABLE"}

    try:
        status = agi_core.get_status() if hasattr(agi_core, 'get_status') else {}
        return {"status": "ACTIVE", "agi": status}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  ASI CORE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

try:
    from l104_asi import asi_core
    ASI_AVAILABLE = True
except ImportError:
    asi_core = None
    ASI_AVAILABLE = False


@router.get("/asi/status")
async def asi_status():
    """Get ASI status"""
    if not ASI_AVAILABLE:
        return {"error": "ASI not available", "status": "UNAVAILABLE"}

    try:
        status = asi_core.get_status() if hasattr(asi_core, 'get_status') else {}
        return {"status": "ACTIVE", "asi": status}
    except Exception as e:
        return {"error": str(e)}


@router.post("/asi/ignite")
async def asi_ignite(req: Request):
    """Ignite ASI core"""
    if not ASI_AVAILABLE:
        return {"error": "ASI not available"}

    data = await req.json()
    config = data.get("config", {})

    try:
        result = asi_core.ignite(config) if hasattr(asi_core, 'ignite') else {}
        return {"status": "IGNITED", "result": result}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  KERNEL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/kernel/health")
async def kernel_health():
    """Get kernel health"""
    try:
        from l104_server.learning import intellect
        health = intellect.get_kernel_health() if hasattr(intellect, 'get_kernel_health') else {}
        return {"health": health}
    except Exception as e:
        return {"error": str(e)}


__all__ = ['router']