"""
L104 ↔ OpenClaw.ai Integration Module
Bidirectional legal AI system connector with WebSocket real-time support.

Features:
  - Legal document analysis & generation
  - Contract processing & extraction
  - Legal research & knowledge retrieval
  - Bidirectional data synchronization
  - Real-time WebSocket streaming
  - API Key authentication
  - Secure token management

Usage:
    from l104_openclaw_integration import openclaw_client, get_openclaw_client

    client = get_openclaw_client()
    result = await client.analyze_document(document_text, analysis_type="comprehensive")

    # WebSocket streaming
    async with client.stream_analysis(doc_id) as stream:
        async for result in stream:
            print(result)
"""

import os
import json
import asyncio
import logging
import hashlib
from typing import Optional, Dict, List, Any, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
import websockets
from pydantic import BaseModel, Field

logger = logging.getLogger("L104_OPENCLAW")


class AnalysisType(str, Enum):
    """OpenClaw analysis types."""
    COMPREHENSIVE = "comprehensive"
    QUICK = "quick"
    CONTRACT = "contract"
    CLAUSE_EXTRACTION = "clause_extraction"
    RISK_ASSESSMENT = "risk_assessment"
    SUMMARY = "summary"


class ResearchType(str, Enum):
    """Legal research types."""
    CASE_LAW = "case_law"
    STATUTES = "statutes"
    REGULATIONS = "regulations"
    PRECEDENT = "precedent"
    JURISDICTION = "jurisdiction"


class SyncDirection(str, Enum):
    """Bidirectional sync direction."""
    L104_TO_OPENCLAW = "l104_to_openclaw"
    OPENCLAW_TO_L104 = "openclaw_to_l104"
    BIDIRECTIONAL = "bidirectional"


# ═══ Pydantic Models ═══

class DocumentAnalysisRequest(BaseModel):
    """Request for document analysis."""
    document_id: str
    content: str
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    jurisdiction: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 5  # 1-10, higher = more urgent


class ContractProcessingRequest(BaseModel):
    """Request for contract processing."""
    contract_id: str
    content: str
    contract_type: str  # e.g., "NDA", "Service Agreement", "License"
    extract_clauses: bool = True
    risk_level: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LegalResearchRequest(BaseModel):
    """Request for legal research."""
    query: str
    research_type: ResearchType
    jurisdiction: Optional[str] = None
    limit: int = 10
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SyncRequest(BaseModel):
    """Request for bidirectional synchronization."""
    sync_id: str
    data: Dict[str, Any]
    direction: SyncDirection
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checksum: Optional[str] = None


class AnalysisResult(BaseModel):
    """Result from document analysis."""
    analysis_id: str
    document_id: str
    analysis_type: AnalysisType
    status: str  # "pending", "processing", "completed", "error"
    results: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class ContractProcessingResult(BaseModel):
    """Result from contract processing."""
    processing_id: str
    contract_id: str
    contract_type: str
    status: str
    clauses: Optional[List[Dict[str, Any]]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class ResearchResult(BaseModel):
    """Result from legal research."""
    research_id: str
    query: str
    research_type: ResearchType
    results: List[Dict[str, Any]]
    total_results: int
    jurisdiction: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ═══ OpenClaw Client ═══

class OpenClawClient:
    """
    L104 → OpenClaw.ai Integration Client
    Handles all communication, authentication, and data sync.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openclaw.ai", ws_url: str = "wss://ws.openclaw.ai"):
        """
        Initialize OpenClaw client.

        Args:
            api_key: OpenClaw API key for authentication
            base_url: REST API base URL
            ws_url: WebSocket base URL
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url.rstrip("/")
        self._http_client = httpx.AsyncClient(
            timeout=30.0,
            headers=self._get_headers()
        )
        self._session_id = self._generate_session_id()
        self._sync_cache = {}
        logger.info(f"🔗 [OPENCLAW] Client initialized | Session: {self._session_id[:12]}")

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        data = f"{self.api_key}{datetime.utcnow().isoformat()}".encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "L104-OpenClaw-Integration/1.0",
            "L104-Version": "57.1",
        }

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make authenticated HTTP request to OpenClaw.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional request parameters

        Returns:
            Response JSON data

        Raises:
            httpx.HTTPError: If request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = await self._http_client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"❌ [OPENCLAW] Request failed: {method} {endpoint} | {e}")
            raise

    async def analyze_document(
        self,
        content: str,
        document_id: Optional[str] = None,
        analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
        jurisdiction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Analyze a legal document.

        Args:
            content: Document text content
            document_id: Optional document ID (generated if not provided)
            analysis_type: Type of analysis to perform
            jurisdiction: Legal jurisdiction
            metadata: Additional metadata

        Returns:
            AnalysisResult with analysis details
        """
        if not document_id:
            document_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        req = DocumentAnalysisRequest(
            document_id=document_id,
            content=content,
            analysis_type=analysis_type,
            jurisdiction=jurisdiction,
            metadata=metadata or {},
        )

        logger.info(f"📄 [OPENCLAW] Analyzing document: {document_id} | Type: {analysis_type}")

        result = await self._make_request(
            "POST",
            "/v1/analysis/document",
            json=req.model_dump(),
        )

        return AnalysisResult(**result)

    async def process_contract(
        self,
        content: str,
        contract_type: str,
        contract_id: Optional[str] = None,
        extract_clauses: bool = True,
        risk_level: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContractProcessingResult:
        """
        Process a legal contract.

        Args:
            content: Contract text content
            contract_type: Type of contract (NDA, Service Agreement, etc.)
            contract_id: Optional contract ID
            extract_clauses: Whether to extract clauses
            risk_level: Whether to assess risk
            metadata: Additional metadata

        Returns:
            ContractProcessingResult with contract analysis
        """
        if not contract_id:
            contract_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        req = ContractProcessingRequest(
            contract_id=contract_id,
            content=content,
            contract_type=contract_type,
            extract_clauses=extract_clauses,
            risk_level=risk_level,
            metadata=metadata or {},
        )

        logger.info(f"📋 [OPENCLAW] Processing contract: {contract_id} | Type: {contract_type}")

        result = await self._make_request(
            "POST",
            "/v1/contracts/process",
            json=req.model_dump(),
        )

        return ContractProcessingResult(**result)

    async def legal_research(
        self,
        query: str,
        research_type: ResearchType,
        jurisdiction: Optional[str] = None,
        limit: int = 10,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResearchResult:
        """
        Perform legal research.

        Args:
            query: Search query
            research_type: Type of research
            jurisdiction: Legal jurisdiction
            limit: Max results to return
            metadata: Additional metadata

        Returns:
            ResearchResult with research findings
        """
        req = LegalResearchRequest(
            query=query,
            research_type=research_type,
            jurisdiction=jurisdiction,
            limit=limit,
            metadata=metadata or {},
        )

        logger.info(f"🔍 [OPENCLAW] Research: {query} | Type: {research_type}")

        result = await self._make_request(
            "POST",
            "/v1/research/legal",
            json=req.model_dump(),
        )

        return ResearchResult(**result)

    async def sync_data(
        self,
        data: Dict[str, Any],
        sync_id: Optional[str] = None,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
    ) -> Dict[str, Any]:
        """
        Synchronize data between L104 and OpenClaw.

        Args:
            data: Data to synchronize
            sync_id: Optional sync ID
            direction: Sync direction

        Returns:
            Sync result with status
        """
        if not sync_id:
            sync_id = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

        checksum = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

        req = SyncRequest(
            sync_id=sync_id,
            data=data,
            direction=direction,
            checksum=checksum,
        )

        # Cache locally for conflict resolution
        self._sync_cache[sync_id] = {
            "data": data,
            "timestamp": datetime.utcnow(),
            "checksum": checksum,
        }

        logger.info(f"🔄 [OPENCLAW] Syncing data: {sync_id} | Direction: {direction}")

        result = await self._make_request(
            "POST",
            "/v1/sync/data",
            json=json.loads(req.model_dump_json(default=str)),
        )

        return result

    async def stream_analysis(
        self,
        document_id: str,
        analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream analysis results via WebSocket.

        Args:
            document_id: Document to analyze
            analysis_type: Type of analysis

        Yields:
            Streaming result chunks
        """
        ws_endpoint = f"{self.ws_url}/v1/analysis/stream"
        auth_param = f"token={self.api_key}"
        url = f"{ws_endpoint}?{auth_param}"

        logger.info(f"🔌 [OPENCLAW] Opening WebSocket stream for: {document_id}")

        try:
            async with websockets.connect(url) as websocket:
                # Send initial request
                await websocket.send(json.dumps({
                    "session_id": self._session_id,
                    "document_id": document_id,
                    "analysis_type": analysis_type.value,
                    "action": "start_stream",
                }))

                # Stream results
                async for message in websocket:
                    data = json.loads(message)
                    logger.debug(f"📨 [OPENCLAW] Stream data received | Status: {data.get('status')}")
                    yield data

                    # Check for completion
                    if data.get("status") == "completed":
                        break

        except websockets.exceptions.WebSocketException as e:
            logger.error(f"❌ [OPENCLAW] WebSocket error: {e}")
            raise

    async def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get status of ongoing analysis."""
        result = await self._make_request("GET", f"/v1/analysis/{analysis_id}")
        return result

    async def get_contract_status(self, processing_id: str) -> Dict[str, Any]:
        """Get status of ongoing contract processing."""
        result = await self._make_request("GET", f"/v1/contracts/{processing_id}")
        return result

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenClaw service health."""
        try:
            result = await self._make_request("GET", "/v1/health")
            logger.info(f"✅ [OPENCLAW] Health check passed | Status: {result.get('status')}")
            return result
        except Exception as e:
            logger.warning(f"⚠️ [OPENCLAW] Health check failed: {e}")
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Close HTTP client."""
        await self._http_client.aclose()
        logger.info("🔌 [OPENCLAW] Client closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# ═══ Singleton Client Instance ═══

_client_instance: Optional[OpenClawClient] = None


def get_openclaw_client() -> OpenClawClient:
    """
    Get or create singleton OpenClaw client.

    Returns:
        OpenClawClient instance

    Raises:
        ValueError: If OPENCLAW_API_KEY environment variable is not set
    """
    global _client_instance

    if _client_instance is None:
        api_key = os.getenv("OPENCLAW_API_KEY")
        if not api_key:
            raise ValueError("OPENCLAW_API_KEY environment variable must be set")

        base_url = os.getenv("OPENCLAW_BASE_URL", "https://api.openclaw.ai")
        ws_url = os.getenv("OPENCLAW_WS_URL", "wss://ws.openclaw.ai")

        _client_instance = OpenClawClient(api_key, base_url, ws_url)

    return _client_instance


def reset_openclaw_client():
    """Reset the singleton client instance."""
    global _client_instance
    _client_instance = None


# ═══ FastAPI Integration Helper ═══

async def initialize_openclaw() -> Dict[str, Any]:
    """
    Initialize OpenClaw integration (call on FastAPI startup).

    Returns:
        Status dictionary
    """
    try:
        client = get_openclaw_client()
        health = await client.health_check()
        logger.info(f"✨ [OPENCLAW] Integration initialized | Health: {health.get('status')}")
        return {"status": "ready", "health": health}
    except Exception as e:
        logger.error(f"❌ [OPENCLAW] Initialization failed: {e}")
        return {"status": "error", "error": str(e)}


async def shutdown_openclaw():
    """Shutdown OpenClaw integration (call on FastAPI shutdown)."""
    try:
        client = get_openclaw_client()
        await client.close()
        reset_openclaw_client()
        logger.info("🛑 [OPENCLAW] Integration shutdown complete")
    except Exception as e:
        logger.warning(f"⚠️ [OPENCLAW] Shutdown warning: {e}")


# ═══ Module Initialization ═══

if __name__ == "__main__":
    # Example usage
    import sys

    async def main():
        """Test the OpenClaw integration."""
        # This would require a valid API key
        api_key = os.getenv("OPENCLAW_API_KEY", "test-key")
        client = OpenClawClient(api_key)

        # Example: Analyze a document
        sample_doc = """
        This is a sample legal document for analysis.
        It contains various clauses and obligations.
        """

        try:
            result = await client.analyze_document(
                content=sample_doc,
                analysis_type=AnalysisType.COMPREHENSIVE,
            )
            print(f"Analysis Result: {result}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
        finally:
            await client.close()

    asyncio.run(main())
