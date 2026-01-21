import sys
from pathlib import Path
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as app_main

# FIX: Mock lifespan to prevent heavy startup
@asynccontextmanager
async def mock_lifespan(app):
    yield

@pytest.fixture(autouse=True)
def safe_app_lifespan():
    """Prevent heavy background tasks during tests."""
    # Override lifespan
    original_lifespan = app_main.app.router.lifespan_context
    app_main.app.router.lifespan_context = mock_lifespan
    
    # Clear depreciated on_event startup handlers if any
    original_startup = list(app_main.app.router.on_startup)
    app_main.app.router.on_startup = []
    
    yield
    
    # Restore
    app_main.app.router.lifespan_context = original_lifespan
    app_main.app.router.on_startup = original_startup

def test_ramnode_roundtrip(tmp_path, monkeypatch):
    db_path = tmp_path / "ramnode.db"
    monkeypatch.setattr(app_main, "RAMNODE_DB_PATH", str(db_path))
    app_main._init_ramnode_db()

    with TestClient(app_main.app) as client:
        payload = {"key": "alpha", "value": "one"}
        resp = client.post("/ramnode", json=payload)
        assert resp.status_code == 200

        get_resp = client.get("/ramnode/alpha")
        assert get_resp.status_code == 200
        assert get_resp.json().get("value") == "one"

        list_resp = client.get("/ramnode")
        assert list_resp.status_code == 200
        items = list_resp.json().get("items") or []
        assert any(item["key"] == "alpha" and item["value"] == "one" for item in items)
