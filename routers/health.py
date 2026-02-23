# routers/health.py — Health, Readiness & Self-Diagnostic routes
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette.responses import JSONResponse as StarletteJSON

from config import UTC
from db import _log_node
from models import HealthResponse, DetailedHealthResponse
from state import app_metrics

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check() -> HealthResponse:
    """Return basic server health status and uptime."""
    uptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat(),
        uptime_seconds=uptime,
        requests_total=app_metrics["requests_total"],
    )


@router.get("/healthz/detail", tags=["Health"])
async def health_detail() -> DetailedHealthResponse:
    """Detailed health with lattice and invariant checks."""
    from l104_persistence import verify_god_code, verify_lattice, verify_survivor_algorithm, verify_alpha
    uptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    checks = {
        "GOD_CODE_INVARIANT": verify_god_code(),
        "LATTICE_INTEGRITY": verify_lattice(),
        "SURVIVOR_STABILITY": verify_survivor_algorithm(),
        "ALPHA_ALIGNMENT": verify_alpha(),
    }
    return DetailedHealthResponse(
        status="healthy" if all(checks.values()) else "degraded",
        timestamp=datetime.now(UTC).isoformat(),
        uptime_seconds=uptime,
        requests_total=app_metrics["requests_total"],
        checks=checks,
    )


@router.get("/readyz", tags=["Health"])
async def readiness_probe():
    """Kubernetes-style readiness probe. Returns 503 if critical subsystems are down."""
    from l104_agi_core import agi_core
    from l104_asi_core import asi_core
    from l104_evolution_engine import evolution_engine
    from l104_persistence import verify_god_code
    critical = {
        "agi_core": agi_core is not None,
        "asi_core": asi_core is not None,
        "evolution_engine": evolution_engine is not None,
    }
    all_ready = all(critical.values())
    uptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    return StarletteJSON(
        content={
            "ready": all_ready,
            "uptime_seconds": uptime,
            "subsystems": critical,
            "god_code_valid": verify_god_code(),
        },
        status_code=200 if all_ready else 503,
    )


@router.post("/self/rotate", tags=["Diagnostics"])
async def manual_rotate():
    """Manually rotate to the next model in the pool."""
    import state as _state
    _state._current_model_index = (_state._current_model_index + 1) % 8
    _log_node({"tag": "manual_rotate", "new_index": _state._current_model_index})
    return {"status": "OK", "new_index": _state._current_model_index}


@router.post("/self/replay", tags=["Diagnostics"])
async def self_replay(base_url: str = None, dataset: str = None):
    """Trigger a self-replay diagnostic test against the streaming API."""
    from config import SELF_BASE_URL, SELF_DATASET
    from db import _load_jsonl
    from state import get_http_client

    target_base = base_url or SELF_BASE_URL
    target_dataset = dataset or SELF_DATASET
    prompts = _load_jsonl(target_dataset)

    if not prompts:
        return {"status": "NO_DATA", "dataset": target_dataset, "tested": 0}

    client = await get_http_client()
    tested, successes, failures, previews = 0, 0, 0, []
    for row in prompts:
        payload = {"signal": row.get("signal"), "message": row.get("message")}
        try:
            resp = await client.post(f"{target_base.rstrip('/')}/api/v6/stream", json=payload)
            tested += 1
            if resp.status_code == 200:
                successes += 1
                previews.append(resp.text[:200])
            else:
                failures += 1
                previews.append(f"ERR {resp.status_code}: {resp.text[:120]}")
        except Exception as exc:
            failures += 1
            previews.append(f"EXC: {exc}")

    result = {"status": "OK", "dataset": target_dataset, "tested": tested,
              "successes": successes, "failures": failures, "previews": previews[:5]}
    _log_node({"tag": "self_replay", **result})
    return result


@router.post("/self/heal", tags=["Diagnostics"])
async def self_heal(reset_rate_limits: bool = True, reset_http_client: bool = False):
    """Trigger self-healing diagnostics."""
    import state as _state
    actions = []

    if reset_rate_limits:
        _state.rate_limit_store.clear()
        actions.append("rate_limits_cleared")

    # Clear model cooldowns
    _state._model_cooldowns.clear()
    _state._quota_exhausted_until = 0
    _state._consecutive_quota_failures = 0
    actions.append("model_cooldowns_cleared")

    if reset_http_client:
        await _state.close_http_client()
        actions.append("http_client_reset")

    from db import _init_memory_db
    _init_memory_db()
    actions.append("memory_checked")

    result = {"status": "OK", "actions": actions}
    _log_node({"tag": "self_heal", **result})
    return result
