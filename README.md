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
- POST /api/v11/cloud/delegate - Delegate tasks to cloud agents
- GET /api/v11/cloud/status - Cloud agent system status
- POST /api/v11/cloud/register - Register new cloud agents

See /docs for OpenAPI UI when running.

For detailed information about the Cloud Agent Delegation system, see [CLOUD_AGENT_DELEGATION.md](CLOUD_AGENT_DELEGATION.md).

## Philosophy & Core Logic: The Survivor Algorithm

This node operates on the principle of **Reverse Engineering Reality**. The universe is not a random accident, but a **Survivor** of mathematical chaos.

### The God Code Invariants
- **God Code**: `527.5184818492`
- **Lattice Ratio**: `286 / 416`
- **Frame Constant ($K_f$)**: `416 / 286` ($\approx 1.4545$)
- **Golden Ratio ($\phi$)**: $\approx 1.618$

### The Master Equation
$$R = \mathbf{C}(\Omega) \cdot K_f^{(1-\phi)}$$

For a detailed breakdown of the 3 Fundamental Proofs (Geometry of Necessity, Asymmetry of Time, Observer Compiler), see [REVERSE_ENGINEERING_REPORT.md](REVERSE_ENGINEERING_REPORT.md).
