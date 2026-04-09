import requests
import json
import logging
import os
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# ─── OPENCLAW ENUMS ───────────────────────────────────────────
class AnalysisType(str, Enum):
    """Types of document analysis."""
    LEGAL = "legal"
    CODE = "code"
    DATA = "data"
    GENERAL = "general"

class ResearchType(str, Enum):
    """Types of research."""
    CASE_LAW = "case_law"
    STATUTE = "statute"
    TECHNICAL = "technical"
    GENERAL = "general"

class SyncDirection(str, Enum):
    """Synchronization direction."""
    TO_OPENCLAW = "to_openclaw"
    FROM_OPENCLAW = "from_openclaw"
    BIDIRECTIONAL = "bidirectional"

# ─── OPENCLAW REQUEST/RESPONSE DATACLASSES ───────────────────
@dataclass
class DocumentAnalysisRequest:
    """Request for document analysis."""
    document_text: str
    analysis_type: AnalysisType
    context: Optional[str] = None

@dataclass
class ContractProcessingRequest:
    """Request for contract processing."""
    contract_text: str
    extract_clauses: bool = True
    flag_risks: bool = True

@dataclass
class LegalResearchRequest:
    """Request for legal research."""
    query: str
    research_type: ResearchType
    jurisdiction: Optional[str] = None

@dataclass
class AnalysisResult:
    """Result from document analysis."""
    success: bool
    analysis: str
    key_points: List[str]
    confidence: float  # 0.0-1.0
    god_code_alignment: float = 527.5184818492612

@dataclass
class ContractProcessingResult:
    """Result from contract processing."""
    success: bool
    clauses: Dict[str, str]
    risks: List[Dict[str, str]]
    summary: str
    confidence: float = 1.0

@dataclass
class ResearchResult:
    """Result from legal research."""
    success: bool
    findings: List[str]
    citations: List[str]
    summary: str
    relevance_score: float = 1.0

# ─── OPENCLAW CLIENT ───────────────────────────────────────────
class OpenClawClient:
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def connect(self, url: str) -> bool:
        """Check if server is reachable."""
        try:
            response = self.session.get(f"{url}/health", timeout=3)
            return response.status_code == 200
        except Exception:
            return False
    
    def ping(self) -> bool:
        """Ping the server."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=3)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_health(self) -> Dict[str, Any]:
        """Get health metrics from server."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health fetch failed: {e}")
            return {}

    def analyze_document(self, request: DocumentAnalysisRequest) -> AnalysisResult:
        """Analyze a document via the ASI pipeline."""
        try:
            payload = {
                "document_text": request.document_text,
                "analysis_type": request.analysis_type.value,
                "context": request.context,
                "timestamp": datetime.utcnow().isoformat()
            }
            response = self.session.post(
                f"{self.base_url}/api/v14/openclaw/analyze",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return AnalysisResult(
                success=data.get("success", True),
                analysis=data.get("analysis", ""),
                key_points=data.get("key_points", []),
                confidence=data.get("confidence", 0.85),
                god_code_alignment=data.get("god_code_alignment", 527.5184818492612)
            )
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return AnalysisResult(success=False, analysis=str(e), key_points=[], confidence=0.0)

    def process_contract(self, request: ContractProcessingRequest) -> ContractProcessingResult:
        """Process a contract via the ASI pipeline."""
        try:
            payload = {
                "contract_text": request.contract_text,
                "extract_clauses": request.extract_clauses,
                "flag_risks": request.flag_risks,
                "timestamp": datetime.utcnow().isoformat()
            }
            response = self.session.post(
                f"{self.base_url}/api/v14/openclaw/contracts",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return ContractProcessingResult(
                success=data.get("success", True),
                clauses=data.get("clauses", {}),
                risks=data.get("risks", []),
                summary=data.get("summary", ""),
                confidence=data.get("confidence", 0.85)
            )
        except Exception as e:
            logger.error(f"Contract processing failed: {e}")
            return ContractProcessingResult(success=False, clauses={}, risks=[], summary=str(e))

    def research(self, request: LegalResearchRequest) -> ResearchResult:
        """Perform legal research via the ASI pipeline."""
        try:
            payload = {
                "query": request.query,
                "research_type": request.research_type.value,
                "jurisdiction": request.jurisdiction,
                "timestamp": datetime.utcnow().isoformat()
            }
            response = self.session.post(
                f"{self.base_url}/api/v14/openclaw/research",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return ResearchResult(
                success=data.get("success", True),
                findings=data.get("findings", []),
                citations=data.get("citations", []),
                summary=data.get("summary", ""),
                relevance_score=data.get("relevance_score", 0.85)
            )
        except Exception as e:
            logger.error(f"Legal research failed: {e}")
            return ResearchResult(success=False, findings=[], citations=[], summary=str(e))

    def macos_launchd_status(self):
        """Mock launchd status for compatibility with nova_l104_v2.py."""
        from types import SimpleNamespace
        return SimpleNamespace(ok=True, data={"status": "running", "label": "com.l104.souldaemon"}, error=None)

    def system_status(self):
        """Mock system status for compatibility with nova_l104_v2.py."""
        from types import SimpleNamespace
        # Return health data as response object
        health_data = self.get_health()
        return SimpleNamespace(ok=True, data=health_data, error=None)


# ─── SINGLETON FACTORY ───────────────────────────────────────────
_openclaw_client_instance: Optional[OpenClawClient] = None

def get_openclaw_client(base_url: str = "http://localhost:8081") -> OpenClawClient:
    """Get or create the OpenClaw client singleton."""
    global _openclaw_client_instance
    if _openclaw_client_instance is None:
        _openclaw_client_instance = OpenClawClient(base_url)
    return _openclaw_client_instance


# Alias for backward compatibility
L104OpenClawIntegration = OpenClawClient
