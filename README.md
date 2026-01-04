# Allentown-L104-Node

FastAPI-based Gemini relay with health/metrics, rate limiting, and optional local fallback.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if present; otherwise pip install fastapi uvicorn httpx pydantic
./scripts/run_services.sh
```

## Environment

```bash
AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U=<your-gemini-key>
GITHUB_TOKEN=<your-github-token>  # optional for /api/v6/manipulate
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_ENDPOINT=:streamGenerateContent
ENABLE_FAKE_GEMINI=1  # optional dev fallback when no Gemini key is set
```

## Endpoints

- GET /health
- GET /metrics
- POST /api/v6/stream
- POST /api/stream
- GET /debug/upstream
- POST /api/v6/manipulate

See /docs for OpenAPI UI when running.