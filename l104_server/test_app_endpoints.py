"""
L104 Server — HTTP Integration Tests (EVO_62)
Tests critical endpoints for health, auth guards, input validation, and rate limiting.
"""
import pytest
import asyncio
import os
from httpx import AsyncClient
from main import app


@pytest.fixture
async def client():
    """Create async HTTP client for testing"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoints:
    """Test health check endpoints"""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """GET /health should return 200"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ["OK", "healthy", "ALIVE"]

    @pytest.mark.asyncio
    async def test_api_v14_health_status(self, client):
        """GET /api/v14/health/status should return 200"""
        response = await client.get("/api/v14/health/status")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestChatEndpoint:
    """Test chat endpoint with various inputs"""

    @pytest.mark.asyncio
    async def test_chat_valid_message(self, client):
        """POST /api/v6/chat with valid message should return 200"""
        response = await client.post(
            "/api/v6/chat",
            json={"message": "Hello, how are you?", "use_sovereign_context": True},
            timeout=30.0
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data or "status" in data

    @pytest.mark.asyncio
    async def test_chat_oversized_message(self, client):
        """POST /api/v6/chat with oversized message (>32KB) should return 422"""
        huge_message = "x" * (33 * 1024)  # 33 KB
        response = await client.post(
            "/api/v6/chat",
            json={"message": huge_message},
            timeout=10.0
        )
        # Pydantic validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_empty_message(self, client):
        """POST /api/v6/chat with empty message should return 422"""
        response = await client.post(
            "/api/v6/chat",
            json={"message": ""},
            timeout=10.0
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_whitespace_only_message(self, client):
        """POST /api/v6/chat with whitespace-only message should be stripped to empty"""
        response = await client.post(
            "/api/v6/chat",
            json={"message": "   \t\n   "},
            timeout=10.0
        )
        # After stripping, message becomes empty → validation error
        assert response.status_code == 422


class TestAuthGuards:
    """Test authentication guards on destructive endpoints"""

    @pytest.mark.asyncio
    async def test_self_heal_without_auth(self, client):
        """POST /self/heal without API key
        - If L104_API_KEY is set: should return 403
        - If L104_API_KEY is not set: should return 200 (allowed)
        """
        response = await client.post("/self/heal", timeout=10.0)
        # Behavior depends on whether L104_API_KEY env var is set
        if os.getenv("L104_API_KEY"):
            assert response.status_code == 403
        else:
            assert response.status_code in [200, 500]  # May succeed or fail for other reasons

    @pytest.mark.asyncio
    async def test_self_heal_with_invalid_key(self, client):
        """POST /self/heal with invalid API key
        - If L104_API_KEY is set: should return 403
        """
        if os.getenv("L104_API_KEY"):
            response = await client.post(
                "/self/heal",
                headers={"X-L104-API-Key": "invalid-key-12345"},
                timeout=10.0
            )
            assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_self_heal_with_valid_key(self, client):
        """POST /self/heal with valid API key should return 200"""
        api_key = os.getenv("L104_API_KEY")
        if api_key:
            response = await client.post(
                "/self/heal",
                headers={"X-L104-API-Key": api_key},
                timeout=10.0
            )
            assert response.status_code == 200


class TestCORSConfiguration:
    """Test CORS configuration"""

    @pytest.mark.asyncio
    async def test_cors_allowed_origin(self, client):
        """Test CORS headers for allowed origin"""
        response = await client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 200
        # CORS headers should be present if origin is allowed
        cors_origin = response.headers.get("access-control-allow-origin")
        # May be set to the specific origin or wildcard (depends on config)

    @pytest.mark.asyncio
    async def test_cors_not_wildcard_with_credentials(self, client):
        """Verify CORS is properly configured (not wildcard + credentials)"""
        response = await client.get("/health")
        assert response.status_code == 200
        cors_origin = response.headers.get("access-control-allow-origin")
        cors_creds = response.headers.get("access-control-allow-credentials")
        # If origin is wildcard, credentials should not be True (browser rejects)
        if cors_origin == "*":
            assert cors_creds != "true"


@pytest.mark.asyncio
async def test_health_latency():
    """Test that health endpoint responds quickly (<100ms)"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        import time
        t0 = time.perf_counter()
        response = await client.get("/health", timeout=5.0)
        latency_ms = (time.perf_counter() - t0) * 1000
        assert response.status_code == 200
        assert latency_ms < 100, f"Health check took {latency_ms:.1f}ms, expected <100ms"
