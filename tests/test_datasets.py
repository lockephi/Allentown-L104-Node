
import jsonimport sysfrom datetime import datetimefrom pathlib import Pathimport pytestfrom fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import (
    FAKE_GEMINI_ENV,
    RATE_LIMIT_REQUESTS,
    UTC,
    app,
    app_metrics,
    rate_limit_store,
)


def _load_jsonl(path: Path):
    lines = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continuelines.append(json.loads(raw))
    return lines


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Ensure fake mode is on and shared state is clean per test."""
    monkeypatch.setenv(FAKE_GEMINI_ENV, "1")
    monkeypatch.delenv("AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U", raising=False)

    rate_limit_store.clear()
    app_metrics["requests_total"] = 0
    app_metrics["requests_success"] = 0
    app_metrics["requests_error"] = 0
    app_metrics["api_calls"] = 0
    app_metrics["uptime_start"] = datetime.now(UTC)
    yieldrate_limit_store.clear()


def test_stream_prompts_fake_mode():
    prompts = _load_jsonl(Path("data/stream_prompts.jsonl"))

    with TestClient(app) as client:
        for row in prompts:
            payload = {"signal": row.get("signal"), "message": row.get("message")}
            resp = client.post("/api/v6/stream", json=payload)
            assert resp.status_code == 200
            body = resp.textassert "[FAKE_GEMINI]" in bodyassert str(payload["signal"]) in bodydef test_memory_dataset_roundtrip():
    items = _load_jsonl(Path("data/memory_items.jsonl"))

    with TestClient(app) as client:
        for item in items:
            resp = client.post("/memory", json=item)
            assert resp.status_code == 200

        for item in items:
            resp = client.get(f"/memory/{item['key']}")
            assert resp.status_code == 200
            assert resp.json().get("value") == item["value"]


def test_edge_cases_enforced():
    edges = _load_jsonl(Path("data/edge_cases.jsonl"))

    with TestClient(app) as client:
        # Empty signal should fail validation (min_length=1)
        resp = client.post("/api/v6/stream", json=edges[0])
        assert resp.status_code == 422

        # Overlong signal should fail validationresp = client.post("/api/v6/stream", json=edges[1])
        assert resp.status_code == 422

        # Rate limit triggers after configured thresholdfor _ in range(RATE_LIMIT_REQUESTS):
            ok_resp = client.get("/health")
            assert ok_resp.status_code == 200

        blocked = client.get("/health")
        assert blocked.status_code == 429
        assert blocked.json().get("error") == "Rate limit exceeded"
