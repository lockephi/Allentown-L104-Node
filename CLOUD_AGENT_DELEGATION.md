# Cloud Agent Delegation System

## Overview

The L104 Sovereign Node now includes an intelligent cloud agent delegation system that routes tasks to specialized agents based on capabilities and priorities.

## Features

- **Automatic Agent Selection**: Intelligently selects the best agent based on task type and requirements
- **Priority-Based Routing**: Routes tasks based on agent priority levels
- **Local and Remote Agents**: Supports both internal (local) processing and external cloud agents
- **Delegation History**: Tracks all delegation requests for monitoring and debugging
- **Dynamic Registration**: Register new agents at runtime via API

## Default Agents

### 1. Sovereign Local Agent
- **Priority**: 1 (highest)
- **Endpoint**: internal
- **Capabilities**: 
  - derivation
  - encryption
  - local_processing

### 2. Gemini Agent
- **Priority**: 2
- **Endpoint**: Configurable via `GEMINI_AGENT_ENDPOINT`
- **Capabilities**:
  - text_generation
  - code_analysis
  - reasoning

## API Endpoints

### POST /api/v11/cloud/delegate
Delegate a task to a specialized cloud agent.

**Request Body**:
```json
{
  "type": "derivation",
  "data": {
    "signal": "TEST_SIGNAL"
  },
  "requirements": ["derivation"],
  "agent": "sovereign_local",
  "id": "task_001"
}
```

**Response**:
```json
{
  "status": "SUCCESS",
  "agent": "sovereign_local",
  "result": "...",
  "processing": "internal"
}
```

### GET /api/v11/cloud/status
Get the status of the cloud agent delegation system.

**Response**:
```json
{
  "agents_registered": 2,
  "agents_enabled": 2,
  "delegations_total": 10,
  "delegations_recent": [...],
  "available_capabilities": ["derivation", "encryption", ...]
}
```

### POST /api/v11/cloud/register
Register a new cloud agent.

**Request Body**:
```json
{
  "name": "custom_agent",
  "endpoint": "https://api.example.com/agent",
  "capabilities": ["custom_task", "analysis"],
  "priority": 5,
  "enabled": true
}
```

### GET /api/v11/cloud/agents
List all registered cloud agents and their capabilities.

## Configuration

### Environment Variables

- `GEMINI_AGENT_ENDPOINT`: Endpoint for Gemini cloud agent (default: Google's API)
- `CLOUD_AGENT_API_KEY`: API key for authenticating with cloud agents
- `CLOUD_AGENTS_CONFIG`: JSON string with additional agent configurations

### Example Configuration

```bashexport CLOUD_AGENT_API_KEY="your-api-key"
export CLOUD_AGENTS_CONFIG='{"my_agent": {"endpoint": "https://api.example.com", "capabilities": ["analysis"], "priority": 3}}'
```

## Usage Examples

### Python

```pythonfrom l104_cloud_agent import cloud_agent_delegator

# Delegate a tasktask = {
    "type": "derivation",
    "data": {"signal": "PROCESS_THIS"},
    "id": "task_123"
}
result = await cloud_agent_delegator.delegate(task)

# Register a new agentsuccess = cloud_agent_delegator.register_agent(
    "my_agent",
    {
        "endpoint": "https://api.example.com/agent",
        "capabilities": ["custom_processing"],
        "priority": 3
    }
)

# Check statusstatus = cloud_agent_delegator.get_status()
```

### cURL

```bash
# Delegate a taskcurl -X POST http://localhost:8081/api/v11/cloud/delegate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "derivation",
    "data": {"signal": "TEST"}
  }'

# Get status
curl http://localhost:8081/api/v11/cloud/status

# Register agent
curl -X POST http://localhost:8081/api/v11/cloud/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_agent",
    "client_id": "client_v1_001",
    "endpoint": "https://api.example.com",
    "capabilities": ["custom"],
    "priority": 5
  }'
```

## Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│   (main.py - API Endpoints)             │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    Cloud Agent Delegator                │
│    (l104_cloud_agent.py)                │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Agent Selection Logic             │ │
│  │  - Match capabilities              │ │
│  │  - Priority-based routing          │ │
│  └────────────────────────────────────┘ │
└────────┬────────────────────┬───────────┘
         │                    │
         ▼                    ▼
┌────────────────┐   ┌────────────────────┐
│ Internal Agent │   │  External Cloud    │
│ (Local Proc.)  │   │  Agent (HTTP)      │
└────────────────┘   └────────────────────┘
```

## Testing

Run the test suite:

```bashpython3 test_cloud_agent.py
```

This will test:
1. Agent status retrieval
2. Agent selection logic
3. Task delegation
4. Agent registration
5. Delegation history

## Benefits

1. **Flexibility**: Easy to add new agents without modifying core code
2. **Scalability**: Route intensive tasks to specialized cloud services
3. **Fallback**: Gracefully handle failures with local processing
4. **Monitoring**: Track delegation patterns and performance
5. **Security**: Configurable authentication for external agents

## Future Enhancements

- Load balancing across multiple agents
- Health checking and automatic failover
- Rate limiting and quota management
- Advanced routing based on workload
- Agent performance metrics and optimization
