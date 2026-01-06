# Allentown-L104-Node

FastAPI-based Gemini relay with health/metrics, rate limiting, optional local fallback, and **autonomous features** including auto-approve, audio analysis, and cloud agent delegation.

## Features

- 🤖 **Autonomous Operations**: Auto-approve commits and self-modification capabilities
- 🎵 **Audio Analysis**: Resonance detection and tuning verification (432 Hz standard)
- ☁️ **Cloud Delegation**: Distribute tasks to cloud agents with fallback support
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
- **GET /api/v6/autonomy/status** - Check autonomy configuration and status
- **POST /api/v6/audio/analyze** - Analyze audio for resonance (432 Hz standard)
- **POST /api/v6/cloud/delegate** - Delegate tasks to cloud agent with fallback

### Cloud Agent System (v11)
- POST /api/v11/cloud/delegate - Delegate tasks to cloud agents
- GET /api/v11/cloud/status - Cloud agent system status
- POST /api/v11/cloud/register - Register new cloud agents

See **/docs** for OpenAPI UI when running.

For detailed information:
- [CLOUD_AGENT_DELEGATION.md](CLOUD_AGENT_DELEGATION.md) - Cloud Agent Delegation system
- **Autonomy Features** - See below for complete documentation

## Autonomy Features

### Auto-Approve System

The auto-approve feature allows autonomous commits to proceed without manual approval.

**Configuration:**
```bash
ENABLE_AUTO_APPROVE=1          # Enable auto-approval (default: true)
AUTO_APPROVE_MODE=ALWAYS_ON    # Mode: ALWAYS_ON, CONDITIONAL, or OFF
```

**Modes:**
- **ALWAYS_ON**: All autonomous commits are automatically approved (recommended for trusted environments)
- **CONDITIONAL**: Commits are approved based on specific conditions
- **OFF**: All commits require manual approval

### Audio Analysis

Analyze audio sources for resonance patterns and tuning verification.

**Example:**
```bash
curl -X POST http://localhost:8081/api/v6/audio/analyze \
  -H "Content-Type: application/json" \
  -d '{"audio_source": "locke phi asura", "check_tuning": true}'
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "source": "locke phi asura",
    "resonance_detected": true,
    "resonance_frequency": 432.0,
    "in_tune": true,
    "tuning_standard": "A=432Hz",
    "quality_score": 0.98
  }
}
```

### Cloud Delegation

Delegate tasks to cloud agents for distributed processing.

**Example:**
```bash
curl -X POST http://localhost:8081/api/v6/cloud/delegate \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "code_analysis",
    "payload": {"file": "main.py"},
    "priority": "high"
  }'
```

**Priority Levels:** low, normal, high, urgent

### Security Considerations

1. **Auto-Approve**: Only enable `ALWAYS_ON` mode in trusted environments
2. **GitHub PAT**: Ensure your `GITHUB_PAT` has minimal required permissions
3. **Cloud Agent Key**: Store cloud agent credentials securely
4. **File Permissions**: The node can only autonomously modify files listed in `Sovereign_DNA.json`'s `autonomy.file_permissions`

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
