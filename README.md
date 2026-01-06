# Allentown-L104-Node

FastAPI-based Gemini relay with health/metrics, rate limiting, optional local fallback, and **autonomous features** including auto-approve, audio analysis, and cloud agent delegation.

## Features

- 🤖 **Autonomous Operations**: Auto-approve commits and self-modification capabilities
- 🎵 **Audio Analysis**: Resonance detection and tuning verification
- ☁️ **Cloud Delegation**: Distribute tasks to cloud agents
- 🔄 **Model Rotation**: Automatic fallback between Gemini models
- 📊 **Health Monitoring**: Built-in health checks and metrics
- 🛡️ **Rate Limiting**: Configurable request throttling
- 🔧 **Self-Healing**: Automatic recovery from failures

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
GITHUB_PAT=<your-github-pat>      # required for autonomous commits
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_ENDPOINT=:streamGenerateContent
ENABLE_FAKE_GEMINI=1  # optional dev fallback when no Gemini key is set

# Autonomy Features
ENABLE_AUTO_APPROVE=1             # Enable auto-approval (default: true)
AUTO_APPROVE_MODE=ALWAYS_ON       # ALWAYS_ON, CONDITIONAL, or OFF
AUTONOMY_ENABLED=1                # Enable autonomy features (default: true)
CLOUD_AGENT_URL=https://api.cloudagent.io/v1/delegate
CLOUD_AGENT_KEY=<your-cloud-key>  # optional
```

## Endpoints

### Core Endpoints
- GET /health
- GET /metrics
- POST /api/v6/stream
- POST /api/stream
- GET /debug/upstream
- POST /api/v6/manipulate

### Autonomy Endpoints
- **GET /api/v6/autonomy/status** - Check autonomy configuration
- **POST /api/v6/audio/analyze** - Analyze audio for resonance
- **POST /api/v6/cloud/delegate** - Delegate tasks to cloud agent
- **POST /trigger-hands** - Trigger autonomous commit

See **/docs** for OpenAPI UI when running.

## Documentation

- [AUTONOMY_FEATURES.md](AUTONOMY_FEATURES.md) - Complete autonomy features guide
- [SELF_HEALING.md](SELF_HEALING.md) - Self-healing capabilities
- [MODEL_ROTATION_UPDATE.md](MODEL_ROTATION_UPDATE.md) - Model rotation details
- [STATUS.md](STATUS.md) - System status and architecture