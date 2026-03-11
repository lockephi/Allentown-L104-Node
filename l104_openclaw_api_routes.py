"""
L104 ↔ OpenClaw.ai FastAPI Endpoints
REST API routes for legal document analysis, contracts, and research.

Endpoints:
  POST   /api/v14/openclaw/analyze     — Analyze legal document
  POST   /api/v14/openclaw/contracts   — Process contract
  POST   /api/v14/openclaw/research    — Perform legal research
  POST   /api/v14/openclaw/sync        — Bidirectional data sync
  WS     /api/v14/openclaw/stream      — WebSocket streaming analysis
  GET    /api/v14/openclaw/health      — Integration health check
  GET    /api/v14/openclaw/status      — Get async task status
"""

import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import (
    FastAPI, 
    WebSocket, 
    WebSocketDisconnect, 
    HTTPException, 
    BackgroundTasks, 
    Query,
    Depends
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
import json

from l104_openclaw_integration import (
    get_openclaw_client,
    AnalysisType,
    ResearchType,
    SyncDirection,
    DocumentAnalysisRequest,
    ContractProcessingRequest,
    LegalResearchRequest,
    AnalysisResult,
    ContractProcessingResult,
    ResearchResult,
)

logger = logging.getLogger("L104_OPENCLAW_API")

# ═══ Security ═══

api_key_scheme = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(api_key_scheme)):
    """
    Verify the API key from the Authorization header.
    """
    api_key = os.getenv("OPENCLAW_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENCLAW_API_KEY is not set on the server.",
        )
    
    if credentials.scheme != "Bearer" or credentials.credentials != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ═══ Pydantic Request/Response Models for API ═══

class AnalyzeDocumentRequest(DocumentAnalysisRequest):
    """API request for document analysis."""
    pass


class ProcessContractRequest(ContractProcessingRequest):
    """API request for contract processing."""
    pass


class LegalResearchAPIRequest(LegalResearchRequest):
    """API request for legal research."""
    pass


class SyncDataRequest:
    """API request for data synchronization."""
    def __init__(self, data: Dict[str, Any], direction: str = "bidirectional"):
        self.data = data
        self.direction = direction


class TaskStatus:
    """Status of async task."""
    def __init__(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error


# ═══ Background Task Tracking ═══

_active_tasks: Dict[str, TaskStatus] = {}


async def _track_analysis(task_id: str, future):
    """Track analysis task completion."""
    try:
        result = await future
        _active_tasks[task_id] = TaskStatus(task_id, "completed", result=result.model_dump())
    except Exception as e:
        _active_tasks[task_id] = TaskStatus(task_id, "error", error=str(e))


# ═══ FastAPI Endpoints ═══

def register_openclaw_routes(app: FastAPI):
    """
    Register OpenClaw integration routes to FastAPI app.

    Args:
        app: FastAPI application instance
    """

    # ═══ Document Analysis ═══

    @app.post("/api/v14/openclaw/analyze", response_model=Dict[str, Any])
    async def analyze_document(
        request: AnalyzeDocumentRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_api_key),
    ) -> Dict[str, Any]:
        """
        Analyze a legal document.

        Request:
            {
                "document_id": "doc_12345",
                "content": "Document text...",
                "analysis_type": "comprehensive",
                "jurisdiction": "US",
                "metadata": {}
            }

        Response:
            {
                "analysis_id": "analysis_12345",
                "document_id": "doc_12345",
                "status": "completed",
                "results": {...}
            }
        """
        try:
            client = get_openclaw_client()

            logger.info(f"📄 [API] Analyzing document: {request.document_id}")

            result = await client.analyze_document(
                content=request.content,
                document_id=request.document_id,
                analysis_type=AnalysisType(request.analysis_type),
                jurisdiction=request.jurisdiction,
                metadata=request.metadata,
            )

            return {
                "status": "success",
                "analysis_id": result.analysis_id,
                "document_id": result.document_id,
                "analysis_type": result.analysis_type,
                "results": result.results,
                "created_at": result.created_at.isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ [API] Document analysis failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ═══ Contract Processing ═══

    @app.post("/api/v14/openclaw/contracts", response_model=Dict[str, Any])
    async def process_contract(
        request: ProcessContractRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_api_key),
    ) -> Dict[str, Any]:
        """
        Process a legal contract.

        Request:
            {
                "contract_id": "contract_12345",
                "content": "Contract text...",
                "contract_type": "NDA",
                "extract_clauses": true,
                "risk_level": true
            }

        Response:
            {
                "processing_id": "proc_12345",
                "contract_id": "contract_12345",
                "status": "completed",
                "clauses": [...],
                "risk_assessment": {...}
            }
        """
        try:
            client = get_openclaw_client()

            logger.info(f"📋 [API] Processing contract: {request.contract_id} | Type: {request.contract_type}")

            result = await client.process_contract(
                content=request.content,
                contract_type=request.contract_type,
                contract_id=request.contract_id,
                extract_clauses=request.extract_clauses,
                risk_level=request.risk_level,
                metadata=request.metadata,
            )

            return {
                "status": "success",
                "processing_id": result.processing_id,
                "contract_id": result.contract_id,
                "contract_type": result.contract_type,
                "clauses": result.clauses,
                "risk_assessment": result.risk_assessment,
                "created_at": result.created_at.isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ [API] Contract processing failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ═══ Legal Research ═══

    @app.post("/api/v14/openclaw/research", response_model=Dict[str, Any])
    async def legal_research(
        request: LegalResearchAPIRequest,
        api_key: str = Depends(verify_api_key),
    ) -> Dict[str, Any]:
        """
        Perform legal research.

        Request:
            {
                "query": "trademark infringement",
                "research_type": "case_law",
                "jurisdiction": "US",
                "limit": 10
            }

        Response:
            {
                "research_id": "res_12345",
                "query": "trademark infringement",
                "results": [...],
                "total_results": 42
            }
        """
        try:
            client = get_openclaw_client()

            logger.info(f"🔍 [API] Legal research: {request.query} | Type: {request.research_type}")

            result = await client.legal_research(
                query=request.query,
                research_type=ResearchType(request.research_type),
                jurisdiction=request.jurisdiction,
                limit=request.limit,
                metadata=request.metadata,
            )

            return {
                "status": "success",
                "research_id": result.research_id,
                "query": result.query,
                "research_type": result.research_type,
                "results": result.results,
                "total_results": result.total_results,
                "created_at": result.created_at.isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ [API] Legal research failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ═══ Data Synchronization ═══

    @app.post("/api/v14/openclaw/sync", response_model=Dict[str, Any])
    async def sync_data(
        data: Dict[str, Any],
        direction: str = Query("bidirectional", regex="^(l104_to_openclaw|openclaw_to_l104|bidirectional)$"),
        api_key: str = Depends(verify_api_key),
    ) -> Dict[str, Any]:
        """
        Synchronize data between L104 and OpenClaw.

        Query Parameters:
            direction: "l104_to_openclaw" | "openclaw_to_l104" | "bidirectional"

        Request Body:
            {
                "key": "value",
                "nested": {...}
            }

        Response:
            {
                "sync_id": "sync_12345",
                "status": "completed",
                "checksum": "abc123...",
                "timestamp": "2026-03-10T..."
            }
        """
        try:
            client = get_openclaw_client()

            logger.info(f"🔄 [API] Syncing data | Direction: {direction}")

            result = await client.sync_data(
                data=data,
                direction=SyncDirection(direction),
            )

            return {
                "status": "success",
                "sync_id": result.get("sync_id"),
                "checksum": result.get("checksum"),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ [API] Sync failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # ═══ WebSocket Streaming ═══

    @app.websocket("/api/v14/openclaw/stream")
    async def websocket_stream(
        websocket: WebSocket,
        document_id: str = Query(...),
        analysis_type: str = Query("comprehensive", regex="^(comprehensive|quick|contract|clause_extraction|risk_assessment|summary)$"),
        api_key: str = Query(None),
    ):
        """
        WebSocket endpoint for real-time streaming analysis.

        Query Parameters:
            document_id: Document ID to analyze
            analysis_type: Type of analysis
            api_key: Your OPENCLAW_API_KEY for authentication

        Usage:
            const ws = new WebSocket(
                'ws://localhost:8081/api/v14/openclaw/stream?document_id=doc_123&analysis_type=comprehensive&api_key=your-key'
            );

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('Analysis update:', data);
            };
        """
        # Manual API Key verification for WebSocket
        server_api_key = os.getenv("OPENCLAW_API_KEY")
        if not api_key or api_key != server_api_key:
            await websocket.close(code=4001, reason="Invalid or missing API key")
            return

        await websocket.accept()

        try:
            client = get_openclaw_client()

            logger.info(f"🔌 [API] WebSocket stream opened | Document: {document_id}")

            # Stream analysis results
            async for chunk in client.stream_analysis(
                document_id=document_id,
                analysis_type=AnalysisType(analysis_type),
            ):
                await websocket.send_json(chunk)

            logger.info(f"✅ [API] WebSocket stream completed | Document: {document_id}")

        except WebSocketDisconnect:
            logger.info(f"🔌 [API] WebSocket disconnected | Document: {document_id}")
        except Exception as e:
            logger.error(f"❌ [API] WebSocket error | Document: {document_id} | Error: {e}")
            # Ensure the connection is not left open on error
            if websocket.client_state != "disconnected":
                await websocket.send_json({
                    "status": "error",
                    "error": str(e),
                })
                await websocket.close(code=1011)

    # ═══ Health & Status ═══

    @app.get("/api/v14/openclaw/health", response_model=Dict[str, Any])
    async def openclaw_health() -> Dict[str, Any]:
        """
        Check OpenClaw integration health.

        Response:
            {
                "status": "healthy",
                "openclaw_api": "healthy",
                "timestamp": "2026-03-10T..."
            }
        """
        try:
            client = get_openclaw_client()
            health = await client.health_check()

            return {
                "status": "healthy",
                "openclaw_api": health.get("status"),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ [API] Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    @app.get("/api/v14/openclaw/status/{task_id}", response_model=Dict[str, Any])
    async def get_task_status(task_id: str) -> Dict[str, Any]:
        """
        Get status of async task.

        Response:
            {
                "task_id": "task_12345",
                "status": "completed",
                "result": {...}
            }
        """
        if task_id not in _active_tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        task = _active_tasks[task_id]
        return {
            "task_id": task.task_id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
        }

    logger.info("✨ [OPENCLAW_API] 6 routes registered: /api/v14/openclaw/*")


# ═══ Integration Hooks ═══

def setup_openclaw_integration(app: FastAPI):
    """
    Setup OpenClaw integration with startup/shutdown hooks.

    Args:
        app: FastAPI application instance
    """
    from l104_openclaw_integration import initialize_openclaw, shutdown_openclaw

    register_openclaw_routes(app)

    @app.on_event("startup")
    async def startup_openclaw():
        """Initialize OpenClaw on startup."""
        status = await initialize_openclaw()
        logger.info(f"📡 [OPENCLAW_STARTUP] Status: {status}")

    @app.on_event("shutdown")
    async def shutdown_openclaw_on_exit():
        """Shutdown OpenClaw on exit."""
        await shutdown_openclaw()

    logger.info("✅ [OPENCLAW_SETUP] Integration setup complete")


if __name__ == "__main__":
    # Example: Register routes to existing app
    from fastapi import FastAPI

    app = FastAPI(title="L104 ↔ OpenClaw Integration", version="1.0")
    setup_openclaw_integration(app)

    logger.info("Ready to serve L104 ↔ OpenClaw API")
