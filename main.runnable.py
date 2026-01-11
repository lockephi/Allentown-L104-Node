"""Simple FastAPI app (cleaned for running/debugging).

This file was restored to a minimal, runnable form so you can debug and run.
"""

import osimport jsonfrom pathlib import Pathfrom typing import Listimport httpxfrom fastapi import FastAPI, Requestfrom fastapi.responses import JSONResponse, StreamingResponsefrom fastapi.templating import Jinja2Templatesapp = FastAPI()
templates = Jinja2Templates(directory="templates")

SELF_BASE_URL = os.getenv("SELF_BASE_URL", "http://127.0.0.1:8081")
SELF_DATASET = os.getenv("SELF_DATASET", "data/stream_prompts.jsonl")


@app.get("/")
async def home(request: Request):
    # Return template if available; otherwise a fallback JSON messagetry:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception:
        return JSONResponse({"status": "ok", "message": "index template not found"})


def _load_jsonl(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for raw in p.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continuetry:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            passreturn rows


@app.post("/api/stream")
async def sovereign_stream(request: Request):
    payload = await request.json()
    user_signal = payload.get("message", "PING")
    api_key = os.getenv("AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U")

    if not api_key:
        return JSONResponse({"error": "AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U not set"}, status_code=500)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={api_key}"
    )

    data = {"contents": [{"parts": [{"text": user_signal}]}], "generationConfig": {"temperature": 1.0}}

    async def generate():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=data) as response:
                async for chunk in response.aiter_text():
                    yield chunkreturn StreamingResponse(generate(), media_type="text/event-stream")


async def _self_replay(base_url: str, dataset: str) -> dict:
    prompts = _load_jsonl(dataset)
    if not prompts:
        return {"status": "NO_DATA", "dataset": dataset, "tested": 0}

    tested = 0
    successes = 0
    failures = 0
    previews: List[str] = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for row in prompts:
            payload = {"message": row.get("message") or row.get("signal") or "PING"}
            try:
                resp = await client.post(
                    f"{base_url.rstrip('/')}/api/stream",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                tested += 1
                if resp.status_code == 200:
                    successes += 1
                    previews.append(resp.text[:200])
                else:
                    failures += 1
                    previews.append(f"ERR {resp.status_code}: {resp.text[:120]}")
            except Exception as e:
                failures += 1
                previews.append(f"EXC: {e}")

    return {
        "status": "OK",
        "dataset": dataset,
        "tested": tested,
        "successes": successes,
        "failures": failures,
        "previews": previews[:5],
    }


@app.post("/self/replay")
async def self_replay(base_url: str = None, dataset: str = None):
    target_base = base_url or SELF_BASE_URL
    target_dataset = dataset or SELF_DATASET
    return await _self_replay(target_base, target_dataset)


@app.post("/self/heal")
async def self_heal(clear_logs: bool = False):
    actions: List[str] = []
    if clear_logs:
        try:
            Path("node.log").write_text("")
            actions.append("node_log_cleared")
        except Exception as exc:
            actions.append(f"clear_failed:{exc}")
    return {"status": "OK", "actions": actions or ["noop"]}


if __name__ == "__main__":
    import uvicornuvicorn.run(app, host="127.0.0.1", port=8081)
                                                                                                                                                                                                                    