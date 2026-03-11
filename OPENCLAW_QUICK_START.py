"""
Quick Integration Guide: Adding OpenClaw.ai to L104 FastAPI App

This file shows the minimal changes needed to integrate OpenClaw with L104.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 1: Add to existing l104_server/app.py (RECOMMENDED)
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Add import at the top of app.py (around line 30-40):

from l104_openclaw_api_routes import setup_openclaw_integration

# 2. After FastAPI app initialization (around line 100-120), add:

# Initialize OpenClaw integration
setup_openclaw_integration(app)

# That's it! The integration will:
# - Register all /api/v14/openclaw/* routes automatically
# - Handle startup/shutdown hooks
# - Log initialization status

# ═══════════════════════════════════════════════════════════════════════════════
# Full Example (minimal app.py changes)
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from l104_openclaw_api_routes import setup_openclaw_integration

# Create app
app = FastAPI(
    title="L104 Sovereign Node",
    version="57.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✨ Initialize OpenClaw integration (ONE LINE!)
setup_openclaw_integration(app)

# ... rest of your routes and configuration ...


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 2: Manual Route Registration (if you prefer more control)
# ═══════════════════════════════════════════════════════════════════════════════

from l104_openclaw_api_routes import register_openclaw_routes
from l104_openclaw_integration import initialize_openclaw, shutdown_openclaw

# Register routes manually
register_openclaw_routes(app)

# Add startup hook
@app.on_event("startup")
async def startup():
    await initialize_openclaw()
    # ... other startup code ...

# Add shutdown hook
@app.on_event("shutdown")
async def shutdown():
    await shutdown_openclaw()
    # ... other shutdown code ...


# ═══════════════════════════════════════════════════════════════════════════════
# TROUBLESHOOTING INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

# If you get ModuleNotFoundError:
# - Make sure l104_openclaw_integration.py is in the workspace root
# - Make sure l104_openclaw_api_routes.py is in the workspace root
# - Verify PYTHONPATH includes the workspace root

# If routes don't appear:
# - Check logs for "[OPENCLAW_API] X routes registered"
# - Verify setup_openclaw_integration() was called
# - Verify no import errors in logs

# If health check fails:
# - Set OPENCLAW_API_KEY environment variable: export OPENCLAW_API_KEY="your-key"
# - Verify network connectivity to api.openclaw.ai
# - Check API key is valid


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

"""
After setup_openclaw_integration(app) is called, the following endpoints are available:

REST Endpoints:
  POST   /api/v14/openclaw/analyze     → Analyze legal documents
  POST   /api/v14/openclaw/contracts   → Process contracts
  POST   /api/v14/openclaw/research    → Legal research
  POST   /api/v14/openclaw/sync        → Bidirectional data sync
  GET    /api/v14/openclaw/health      → Check integration health
  GET    /api/v14/openclaw/status/{id} → Get task status

WebSocket:
  WS     /api/v14/openclaw/stream      → Real-time streaming analysis


Example 1: Analyze a Document
-------------------------------
curl -X POST http://localhost:8081/api/v14/openclaw/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_001",
    "content": "Your legal document text here...",
    "analysis_type": "comprehensive",
    "jurisdiction": "US"
  }'


Example 2: Process a Contract
------------------------------
curl -X POST http://localhost:8081/api/v14/openclaw/contracts \
  -H "Content-Type: application/json" \
  -d '{
    "contract_id": "svc_001",
    "content": "SERVICE AGREEMENT...",
    "contract_type": "Service Agreement"
  }'


Example 3: Legal Research
--------------------------
curl -X POST http://localhost:8081/api/v14/openclaw/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "API liability",
    "research_type": "case_law",
    "jurisdiction": "US",
    "limit": 10
  }'


Example 4: Health Check
-----------------------
curl http://localhost:8081/api/v14/openclaw/health
# Returns: {"status": "healthy", "openclaw_api": "healthy", ...}


Example 5: WebSocket Streaming (JavaScript)
--------------------------------------------
const ws = new WebSocket(
  'ws://localhost:8081/api/v14/openclaw/stream' +
  '?document_id=doc_001&analysis_type=comprehensive'
);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Analysis update:', data);
};


Example 6: Python Client
------------------------
import asyncio
from l104_openclaw_integration import get_openclaw_client, AnalysisType

async def main():
    client = get_openclaw_client()

    result = await client.analyze_document(
        content="Your document...",
        analysis_type=AnalysisType.COMPREHENSIVE
    )

    print(f"Analysis ID: {result.analysis_id}")
    print(f"Results: {result.results}")

asyncio.run(main())
"""

# ═══════════════════════════════════════════════════════════════════════════════
# AVAILABLE ROUTES AFTER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
After calling setup_openclaw_integration(app), you can list all routes:

from fastapi import FastAPI
app = FastAPI()
setup_openclaw_integration(app)

# Print all registered routes
for route in app.routes:
    print(f"{route.methods} {route.path}")

Output will include:
  POST   /api/v14/openclaw/analyze
  POST   /api/v14/openclaw/contracts
  POST   /api/v14/openclaw/research
  POST   /api/v14/openclaw/sync
  WS     /api/v14/openclaw/stream
  GET    /api/v14/openclaw/health
  GET    /api/v14/openclaw/status/{task_id}
"""

if __name__ == "__main__":
    # Minimal working example
    app = FastAPI()
    setup_openclaw_integration(app)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
