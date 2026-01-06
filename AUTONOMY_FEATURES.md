# Autonomy and Cloud Delegation Features

## Overview

The L104 Node now includes comprehensive autonomy features, auto-approval mechanisms, audio analysis capabilities, and cloud agent delegation. These features enable the node to operate with greater independence and distributed processing capabilities.

## Features

### 1. Auto-Approve System

The auto-approve feature allows autonomous commits to proceed without manual approval.

**Configuration:**
```bash
ENABLE_AUTO_APPROVE=1          # Enable auto-approval (default: true)
AUTO_APPROVE_MODE=ALWAYS_ON    # Mode: ALWAYS_ON, CONDITIONAL, or OFF
AUTONOMY_ENABLED=1             # Enable autonomy features (default: true)
```

**Modes:**
- **ALWAYS_ON**: All autonomous commits are automatically approved (recommended for trusted environments)
- **CONDITIONAL**: Commits are approved based on specific conditions
- **OFF**: All commits require manual approval

**Usage:**
The auto-approve setting is automatically applied to all `sovereign_commit()` function calls. You can override it per-commit:

```python
# Use default auto-approve setting
result = await sovereign_commit("file.txt", content, "message")

# Override to disable auto-approve for this commit
result = await sovereign_commit("file.txt", content, "message", auto_approve=False)

# Override to enable auto-approve for this commit
result = await sovereign_commit("file.txt", content, "message", auto_approve=True)
```

### 2. Audio Analysis

Analyze audio sources for resonance patterns and tuning verification.

**Endpoint:** `POST /api/v6/audio/analyze`

**Request:**
```json
{
  "audio_source": "locke phi asura",
  "check_tuning": true
}
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
    "quality_score": 0.98,
    "notes": "Audio signature matches sovereign resonance pattern X=416"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8081/api/v6/audio/analyze \
  -H "Content-Type: application/json" \
  -d '{"audio_source": "locke phi asura", "check_tuning": true}'
```

### 3. Cloud Agent Delegation

Delegate tasks to cloud agents for distributed processing.

**Configuration:**
```bash
CLOUD_AGENT_URL=https://api.cloudagent.io/v1/delegate
CLOUD_AGENT_KEY=your-cloud-agent-key-here
```

**Endpoint:** `POST /api/v6/cloud/delegate`

**Request:**
```json
{
  "task_type": "code_analysis",
  "payload": {
    "file": "main.py",
    "analysis_type": "security"
  },
  "priority": "high"
}
```

**Response (Success):**
```json
{
  "success": true,
  "delegation_result": {
    "task_id": "task-123456",
    "status": "queued"
  },
  "delegated_to": "https://api.cloudagent.io/v1/delegate",
  "task_id": "task-123456",
  "status": "delegated"
}
```

**Response (Fallback):**
```json
{
  "status": "Cloud delegation failed, fallback available",
  "cloud_result": {
    "success": false,
    "error": "Cloud agent not configured",
    "fallback_to_local": true
  },
  "local_processing": true,
  "message": "Task can be processed locally if needed"
}
```

**Priority Levels:**
- `low`: Non-urgent background tasks
- `normal`: Standard priority (default)
- `high`: Important tasks requiring faster processing
- `urgent`: Critical tasks requiring immediate attention

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

### 4. Autonomy Status

Check the current configuration and status of autonomy features.

**Endpoint:** `GET /api/v6/autonomy/status`

**Response:**
```json
{
  "autonomy_enabled": true,
  "auto_approve": {
    "enabled": true,
    "mode": "ALWAYS_ON",
    "description": "Controls automatic approval of autonomous commits"
  },
  "cloud_agent": {
    "configured": false,
    "url": "https://api.cloudagent.io/v1/delegate",
    "ready": true
  },
  "sovereign_commit": {
    "available": true,
    "requires": ["GITHUB_PAT environment variable"],
    "auto_approve_default": true
  },
  "timestamp": "2026-01-06T01:20:03.265429+00:00"
}
```

**Example:**
```bash
curl http://localhost:8081/api/v6/autonomy/status
```

## Sovereign DNA Configuration

The autonomy features are reflected in the Sovereign DNA configuration:

```json
{
  "autonomy": {
    "enabled": true,
    "auto_approve": "ALWAYS_ON",
    "file_permissions": ["Sovereign_DNA.json", "L104_ARCHIVE.txt", "main.py"],
    "cloud_delegation": true,
    "audio_analysis": true
  },
  "evolution_stage": 4,
  "status": "AUTONOMOUS"
}
```

## Security Considerations

1. **Auto-Approve**: Only enable `ALWAYS_ON` mode in trusted environments
2. **GitHub PAT**: Ensure your `GITHUB_PAT` has minimal required permissions
3. **Cloud Agent Key**: Store cloud agent credentials securely
4. **File Permissions**: The node can only autonomously modify files listed in `file_permissions`

## Testing

Run the included test suite to verify all features:

```bash
python3 /tmp/test_autonomy_features.py
```

## Integration Examples

### Autonomous File Updates
```python
# Enable auto-approve for autonomous updates
result = await sovereign_commit(
    "config.json",
    json.dumps({"updated": True}),
    "Auto-update configuration",
    auto_approve=True
)

if result["success"]:
    print(f"✓ File updated: {result['commit_url']}")
```

### Audio Analysis Pipeline
```python
# Analyze audio and process results
analysis = await analyze_audio_resonance("audio_source_id", check_tuning=True)
if analysis["success"] and analysis["analysis"]["resonance_detected"]:
    print(f"✓ Resonance frequency: {analysis['analysis']['resonance_frequency']} Hz")
```

### Cloud Task Delegation
```python
# Delegate heavy processing to cloud agent
task = {
    "type": "data_processing",
    "payload": {"dataset": "large_data.csv"},
    "priority": "high"
}

result = await delegate_to_cloud_agent(task)
if result["success"]:
    print(f"✓ Task delegated: {result['task_id']}")
elif result.get("fallback_to_local"):
    print("⚠ Cloud unavailable, processing locally...")
```

## Troubleshooting

**Auto-approve not working:**
- Check `ENABLE_AUTO_APPROVE` is set to `1`
- Verify `AUTO_APPROVE_MODE` is `ALWAYS_ON`
- Ensure `GITHUB_PAT` is configured

**Cloud delegation failing:**
- Verify `CLOUD_AGENT_URL` is correct
- Check `CLOUD_AGENT_KEY` is set (if required)
- Confirm network connectivity to cloud agent

**Audio analysis errors:**
- Ensure audio source identifier is correct
- Check server logs for detailed error messages

## API Documentation

For complete API documentation, visit:
```
http://localhost:8081/docs
```

When the server is running, this provides an interactive Swagger UI with full endpoint documentation and testing capabilities.
