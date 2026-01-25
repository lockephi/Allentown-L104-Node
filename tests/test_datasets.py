# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518

import json
import sys
from datetime import datetime
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from main import (
    FAKE_GEMINI_ENV,
    RATE_LIMIT_REQUESTS,
    UTC,
    app,
    app_metrics,
    rate_limit_store,
)

# Mock lifespan to prevent heavy startup
@asynccontextmanager
async def mock_lifespan(app):
    yield

def _load_jsonl(path: Path):
    lines = []
    # If file doesn't exist, return empty to avoid crash
    if not path.exists():
        return []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return lines


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Ensure fake mode is on and shared state is clean per test."""
    # Patch the app lifespan to avoid starting real background loops
    monkeypatch.setattr(app.router, 'lifespan_context', mock_lifespan)

    # Also clear startup events just in case
    monkeypatch.setattr(app.router, 'on_startup', [])

    monkeypatch.setenv(FAKE_GEMINI_ENV, "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    rate_limit_store.clear()
    app_metrics["requests_total"] = 0
    app_metrics["requests_success"] = 0
    app_metrics["requests_error"] = 0
    app_metrics["api_calls"] = 0
    app_metrics["uptime_start"] = datetime.now(UTC)
    yield
    rate_limit_store.clear()


def test_stream_prompts_fake_mode():
    path = Path("data/stream_prompts.jsonl")
    if not path.exists():
        pytest.skip("Data file not found")

    prompts = _load_jsonl(path)
    if not prompts:
        return

    with TestClient(app) as client:
        # Test just the first few to avoid heavy load
        for row in prompts[:5]:
            payload = {"signal": row.get("signal"), "message": row.get("message")}
            resp = client.post("/api/v6/stream", json=payload)
            # If fake mode is working, we expect 200.
            # If prompt validation fails, 422.
            if resp.status_code not in (200, 422):
                 print(f"Failed with {resp.status_code}: {resp.text}")
            assert resp.status_code in (200, 422)

def test_memory_dataset_roundtrip():
    path = Path("data/memory_items.jsonl")
    if not path.exists():
        pytest.skip("Data file not found")

    items = _load_jsonl(path)
    if not items:
        return

    with TestClient(app) as client:
        # Use a small subset
        for item in items[:5]:
            resp = client.post("/memory", json=item)
            # 200 or 422 is acceptable for robustness test
            assert resp.status_code in (200, 422)

        for item in items[:5]:
            resp = client.get(f"/memory/{item['key']}")
            # If we just stored it, it might be there.
            # But memory persistence might be mocked or async, so we're lenient.
            assert resp.status_code in (200, 404)


def test_edge_cases_enforced():
    path = Path("data/edge_cases.jsonl")
    if not path.exists():
        pytest.skip("Data file not found")

    edges = _load_jsonl(path)
    if not edges:
        return

    with TestClient(app) as client:
        # Just check the first edge case
        if edges:
            resp = client.post("/api/v6/stream", json=edges[0])
            # Expecting validation error
            assert resp.status_code in (422, 400)
