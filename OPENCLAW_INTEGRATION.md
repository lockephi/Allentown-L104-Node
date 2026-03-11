# L104 ↔ OpenClaw.ai Integration Guide

## Overview

This integration provides bidirectional, real-time connectivity between the L104 Sovereign Node and OpenClaw.ai for comprehensive legal AI capabilities:

- ✅ **Document Analysis** — Analyze legal documents with comprehensive, quick, or specialized analysis
- ✅ **Contract Processing** — Extract clauses, assess risk, and process contracts
- ✅ **Legal Research** — Search case law, statutes, regulations, and precedents
- ✅ **Bidirectional Sync** — Keep L104 and OpenClaw data in perfect sync
- ✅ **Real-time Streaming** — WebSocket-based streaming analysis for live updates
- ✅ **API Key Authentication** — Secure, token-based API access

## Installation & Setup

### 1. Set Environment Variables

```bash
# Required
export OPENCLAW_API_KEY="your-openclaw-api-key"

# Optional (defaults provided)
export OPENCLAW_BASE_URL="https://api.openclaw.ai"
export OPENCLAW_WS_URL="wss://ws.openclaw.ai"
```

### 2. Integrate with Existing L104 Server

Add to your FastAPI app initialization in `l104_server/app.py`:

```python
from l104_openclaw_api_routes import setup_openclaw_integration

# In app initialization
app = FastAPI()

# Setup OpenClaw integration (registers routes + startup/shutdown hooks)
setup_openclaw_integration(app)
```

### 3. Verify Integration

```bash
# Health check
curl http://localhost:8081/api/v14/openclaw/health
```

## Authentication

All API requests must be authenticated using a bearer token in the `Authorization` header. Your `OPENCLAW_API_KEY` should be used as the token.

**Example Header:**
```
Authorization: Bearer your-openclaw-api-key
```

Most client libraries and tools like cURL handle this automatically if you provide the key. The provided Python client and cURL examples assume the server-side integration is extracting the key from the environment variables.

## Getting Started: A 5-Minute Tutorial

This tutorial will guide you through making your first API call to analyze a sample legal text.

**Prerequisites:**
- The L104 server is running.
- You have exported your `OPENCLAW_API_KEY`.

**Step 1: Create a sample request file**

Create a file named `sample_request.json` with the following content:

```json
{
  "document_id": "tutorial_doc_001",
  "content": "This Agreement is made and entered into as of the Effective Date by and between the parties. The Disclosing Party agrees not to disclose any Confidential Information to any third party.",
  "analysis_type": "quick",
  "jurisdiction": "US"
}
```

**Step 2: Send the request using cURL**

Open your terminal and run the following command:

```bash
curl -X POST http://localhost:8081/api/v14/openclaw/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENCLAW_API_KEY" \
  -d @sample_request.json
```

**Step 3: Review the response**

You should receive a JSON response similar to this:

```json
{
  "status": "success",
  "analysis_id": "analysis_tutor_123",
  "document_id": "tutorial_doc_001",
  "analysis_type": "quick",
  "results": {
    "key_issues": [
      "Definition of 'Confidential Information' is broad.",
      "No specified term for the agreement."
    ],
    "recommendations": [
      "Define 'Confidential Information' more narrowly.",
      "Add a specific term or duration for the NDA."
    ],
    "risk_level": "low",
    "summary": "A standard non-disclosure clause. Key areas for improvement include defining the scope of confidential information and the agreement's duration."
  },
  "created_at": "..."
}
```

Congratulations! You've successfully analyzed your first legal document using the OpenClaw integration.

## API Endpoints

### Document Analysis

**Endpoint:** `POST /api/v14/openclaw/analyze`

Analyze any legal document with various analysis types.

**Request:**
```json
{
  "document_id": "doc_12345",
  "content": "Full text of the legal document...",
  "analysis_type": "comprehensive",
  "jurisdiction": "US",
  "metadata": {
    "source": "document_management_system",
    "tags": ["confidential", "reviewed"]
  }
}
```

**Analysis Types:**
- `comprehensive` — Full analysis with all details
- `quick` — Fast summary analysis
- `contract` — Contract-specific analysis
- `clause_extraction` — Extract clauses only
- `risk_assessment` — Focus on legal risks
- `summary` — Brief summary

**Response:**
```json
{
  "status": "success",
  "analysis_id": "analysis_abc123",
  "document_id": "doc_12345",
  "analysis_type": "comprehensive",
  "results": {
    "key_issues": [...],
    "recommendations": [...],
    "risk_level": "medium",
    "summary": "..."
  },
  "created_at": "2026-03-10T14:30:00Z"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8081/api/v14/openclaw/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_nda_001",
    "content": "This Non-Disclosure Agreement...",
    "analysis_type": "comprehensive",
    "jurisdiction": "US"
  }'
```

### Contract Processing

**Endpoint:** `POST /api/v14/openclaw/contracts`

Process contracts with clause extraction and risk assessment.

**Request:**
```json
{
  "contract_id": "contract_001",
  "content": "Full contract text...",
  "contract_type": "Service Agreement",
  "extract_clauses": true,
  "risk_level": true,
  "metadata": {}
}
```

**Contract Types:**
- `NDA` — Non-Disclosure Agreement
- `Service Agreement` — Service provision contract
- `License` — License agreement
- `Employment` — Employment contract
- `Partnership` — Partnership agreement
- etc.

**Response:**
```json
{
  "status": "success",
  "processing_id": "proc_xyz789",
  "contract_id": "contract_001",
  "contract_type": "Service Agreement",
  "clauses": [
    {
      "name": "Confidentiality",
      "text": "The parties agree to keep...",
      "risk_level": "low"
    },
    {
      "name": "Termination",
      "text": "This agreement may be terminated...",
      "risk_level": "medium"
    }
  ],
  "risk_assessment": {
    "overall_risk": "medium",
    "critical_issues": ["Vague liability limits"],
    "recommendations": ["Clarify liability caps"]
  },
  "created_at": "2026-03-10T14:30:00Z"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8081/api/v14/openclaw/contracts \
  -H "Content-Type: application/json" \
  -d '{
    "contract_id": "svc_agreement_2026",
    "content": "SERVICE AGREEMENT...",
    "contract_type": "Service Agreement",
    "extract_clauses": true,
    "risk_level": true
  }'
```

### Legal Research

**Endpoint:** `POST /api/v14/openclaw/research`

Search legal databases and retrieve relevant information.

**Request:**
```json
{
  "query": "trademark infringement liability",
  "research_type": "case_law",
  "jurisdiction": "US",
  "limit": 10,
  "metadata": {}
}
```

**Research Types:**
- `case_law` — Court cases and precedents
- `statutes` — Statutory law
- `regulations` — Regulatory law
- `precedent` — Legal precedents
- `jurisdiction` — Jurisdiction-specific law

**Response:**
```json
{
  "status": "success",
  "research_id": "res_abc123",
  "query": "trademark infringement liability",
  "research_type": "case_law",
  "results": [
    {
      "title": "Case Name v. Plaintiff",
      "citation": "123 F.3d 456 (9th Cir. 2020)",
      "summary": "Court held that...",
      "relevance": 0.95
    }
  ],
  "total_results": 42,
  "created_at": "2026-03-10T14:30:00Z"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8081/api/v14/openclaw/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "API terms of service liability",
    "research_type": "case_law",
    "jurisdiction": "US",
    "limit": 10
  }'
```

### Bidirectional Data Sync

**Endpoint:** `POST /api/v14/openclaw/sync`

Synchronize data between L104 and OpenClaw systems.

**Query Parameters:**
- `direction`: `l104_to_openclaw` | `openclaw_to_l104` | `bidirectional` (default)

**Request:**
```json
{
  "document_key": "doc_sync_001",
  "status": "reviewed",
  "annotations": {
    "reviewer": "alice@example.com",
    "notes": "Additional risks noted"
  },
  "timestamp": "2026-03-10T14:30:00Z"
}
```

**Response:**
```json
{
  "status": "success",
  "sync_id": "sync_def456",
  "checksum": "abc123def456...",
  "timestamp": "2026-03-10T14:30:00Z"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8081/api/v14/openclaw/sync?direction=bidirectional" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_123",
    "status": "reviewed",
    "reviewer": "staff@company.com"
  }'
```

### WebSocket Real-Time Streaming

**Endpoint:** `WS /api/v14/openclaw/stream`

Stream analysis results in real-time via WebSocket.

**Query Parameters:**
- `document_id` — *Required* — Document ID to analyze
- `analysis_type` — Type of analysis (default: `comprehensive`)

**JavaScript Example:**
```javascript
const ws = new WebSocket(
  'ws://localhost:8081/api/v14/openclaw/stream' +
  '?document_id=doc_123&analysis_type=comprehensive'
);

ws.onopen = () => {
  console.log('Connected to L104 OpenClaw stream');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Analysis chunk:', {
    status: data.status,
    progress: data.progress,
    results: data.results
  });

  if (data.status === 'completed') {
    console.log('Analysis complete!');
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Stream connection closed');
};
```

**Python Example:**
```python
import asyncio
import websockets
import json

async def stream_analysis():
    uri = "ws://localhost:8081/api/v14/openclaw/stream?document_id=doc_123&analysis_type=comprehensive"

    async with websockets.connect(uri) as websocket:
        print("Connected!")

        async for message in websocket:
            data = json.loads(message)
            print(f"Status: {data['status']}")
            print(f"Progress: {data.get('progress', 'N/A')}")

            if data['status'] == 'completed':
                break

asyncio.run(stream_analysis())
```

### Health Check

**Endpoint:** `GET /api/v14/openclaw/health`

Check OpenClaw integration status.

**Response:**
```json
{
  "status": "healthy",
  "openclaw_api": "healthy",
  "timestamp": "2026-03-10T14:30:00Z"
}
```

**cURL Example:**
```bash
curl http://localhost:8081/api/v14/openclaw/health
```

### Get Task Status

**Endpoint:** `GET /api/v14/openclaw/status/{task_id}`

Get status of async background tasks.

**Response:**
```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "result": {...}
}
```

## Data Models

For clarity, here are the formal data models for the main API request bodies.

### AnalyzeRequest
Used for the `POST /analyze` endpoint.
| Field | Type | Required | Description |
|---|---|---|---|
| `document_id` | string | Yes | A unique identifier for the document. |
| `content` | string | Yes | The full text of the legal document. |
| `analysis_type` | string | Yes | The type of analysis to perform. See `Analysis Types`. |
| `jurisdiction` | string | Yes | The legal jurisdiction (e.g., 'US', 'UK'). |
| `metadata` | object | No | Optional key-value pairs for additional context. |

### ContractRequest
Used for the `POST /contracts` endpoint.
| Field | Type | Required | Description |
|---|---|---|---|
| `contract_id` | string | Yes | A unique identifier for the contract. |
| `content` | string | Yes | The full text of the contract. |
| `contract_type` | string | Yes | The type of contract. See `Contract Types`. |
| `extract_clauses`| boolean| No | If true, clauses will be extracted. Default: `false`. |
| `risk_level` | boolean| No | If true, a risk assessment will be performed. Default: `false`.|
| `metadata` | object | No | Optional key-value pairs for additional context. |

### ResearchRequest
Used for the `POST /research` endpoint.
| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | The legal research query. |
| `research_type` | string | Yes | The type of research to perform. See `Research Types`. |
| `jurisdiction`| string | Yes | The legal jurisdiction. |
| `limit` | integer| No | The maximum number of results to return. Default: `10`. |
| `metadata` | object | No | Optional key-value pairs for additional context. |

## Python Client Usage

### Basic Usage

```python
import asyncio
from l104_openclaw_integration import get_openclaw_client, AnalysisType

async def main():
    client = get_openclaw_client()

    # Analyze document
    result = await client.analyze_document(
        content="Your legal document here...",
        analysis_type=AnalysisType.COMPREHENSIVE,
        jurisdiction="US"
    )

    print(f"Analysis ID: {result.analysis_id}")
    print(f"Results: {result.results}")

asyncio.run(main())
```

### Document Analysis

```python
result = await client.analyze_document(
    content="NDA text...",
    document_id="nda_001",
    analysis_type=AnalysisType.COMPREHENSIVE,
    jurisdiction="US",
    metadata={"source": "email_attachment"}
)
```

### Contract Processing

```python
result = await client.process_contract(
    content="Contract text...",
    contract_type="Service Agreement",
    contract_id="svc_agreement_2026",
    extract_clauses=True,
    risk_level=True
)

# Access results
for clause in result.clauses:
    print(f"Clause: {clause['name']}")
    print(f"Risk: {clause['risk_level']}")

print(f"Overall Risk: {result.risk_assessment['overall_risk']}")
```

### Legal Research

```python
result = await client.legal_research(
    query="trademark liability",
    research_type=ResearchType.CASE_LAW,
    jurisdiction="US",
    limit=10
)

for r in result.results:
    print(f"{r['title']} ({r['citation']})")
    print(f"Relevance: {r['relevance']}")
```

### Real-time Streaming

```python
async for chunk in client.stream_analysis("doc_123"):
    print(f"Status: {chunk['status']}")
    print(f"Progress: {chunk.get('progress')}%")
    print(f"Data: {chunk.get('data')}")
```

### Context Manager

```python
async with get_openclaw_client() as client:
    result = await client.analyze_document(...)
    # Automatically closes connection on exit
```

## Environment Configuration

### Required Variables

```bash
# OpenClaw API Key (get from https://openclaw.ai/settings)
export OPENCLAW_API_KEY="sk-..."
```

### Optional Variables

```bash
# Custom API endpoints (if using self-hosted OpenClaw)
export OPENCLAW_BASE_URL="https://api.openclaw.ai"
export OPENCLAW_WS_URL="wss://ws.openclaw.ai"

# L104 Logging
export LOG_LEVEL="info"
export LOG_FORMAT="json"
```

## Error Handling

### Common HTTP Status Codes

- **200** — Success
- **400** — Bad request (check parameters)
- **401** — Unauthorized (check API key)
- **403** — Forbidden (insufficient permissions)
- **404** — Not found (resource doesn't exist)
- **429** — Rate limited (wait before retrying)
- **500** — Server error (OpenClaw service issue)

### Python Error Handling

```python
from httpx import HTTPError

async def safe_analyze():
    try:
        client = get_openclaw_client()
        result = await client.analyze_document(content="...")
        return result
    except HTTPError as e:
        print(f"API Error: {e.response.status_code}")
        print(f"Details: {e.response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## Logging & Monitoring

The integration automatically logs to `L104_OPENCLAW`:

```python
import logging

# Set log level
logging.getLogger("L104_OPENCLAW").setLevel(logging.DEBUG)
logging.getLogger("L104_OPENCLAW_API").setLevel(logging.INFO)
```

### Log Entries

- `🔗 [OPENCLAW] Client initialized` — Client startup
- `📄 [OPENCLAW] Analyzing document` — Analysis started
- `📋 [OPENCLAW] Processing contract` — Contract processing started
- `🔍 [OPENCLAW] Research` — Research query submitted
- `🔄 [OPENCLAW] Syncing data` — Data sync in progress
- `🔌 [OPENCLAW] Opening WebSocket stream` — WebSocket connected
- `✅ [OPENCLAW] Health check passed` — Service healthy
- `❌ [OPENCLAW] [operation] failed` — Operation error

## Performance Notes

- **Typical analysis time:** 2-10 seconds (varies by document size)
- **WebSocket latency:** <100ms per chunk
- **Sync operations:** <1 second
- **Rate limits:** Check with OpenClaw.ai support

## Troubleshooting

### API Key Not Found

```
ValueError: OPENCLAW_API_KEY environment variable must be set
```

**Solution:** Make sure environment variable is set:
```bash
export OPENCLAW_API_KEY="your-key"
```

### Connection Refused

```
RequestError: Connection failed...
```

**Solution:**
1. Verify OpenClaw service is running
2. Check network connectivity
3. Verify API endpoints in environment variables

### WebSocket Connection Failed

```
WebSocketException: Connection failed
```

**Solution:**
1. Check OPENCLAW_WS_URL is correct
2. Verify document_id is valid
3. Check firewall/proxy settings

## Testing

Run integration tests:

```bash
# Test client directly
python -c "from l104_openclaw_integration import get_openclaw_client; \
import asyncio; \
client = get_openclaw_client(); \
print('Client ready')"

# Test endpoints
python -m pytest tests/test_openclaw_integration.py -v
```

## Support & Documentation

- **OpenClaw.ai Docs:** https://docs.openclaw.ai
- **L104 Integration:** https://github.com/lockephi/Allentown-L104-Node
- **Report Issues:** Create GitHub issue with logs

## Version History

- **v1.0** (2026-03-10) — Initial integration
  - Document analysis
  - Contract processing
  - Legal research
  - Bidirectional sync
  - WebSocket streaming
