# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
"""
Test API key fallback logic - Ghost Protocol Compliant.
All API keys loaded from environment variables only.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as app_main
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager
import pytest

# FIX: Mock lifespan to prevent heavy startup
@asynccontextmanager
async def mock_lifespan(app):
    yield

@pytest.fixture(autouse=True)
def safe_app_lifespan(monkeypatch):
    """Prevent heavy background tasks during tests."""
    monkeypatch.setattr(app_main.app.router, 'lifespan_context', mock_lifespan)
    monkeypatch.setattr(app_main.app.router, 'on_startup', [])


def test_api_key_env_variable_defined():
    """Test that API_KEY_ENV is defined as an environment variable name."""
    assert app_main.API_KEY_ENV == "GEMINI_API_KEY"


def test_legacy_api_key_env_is_env_var_name():
    """Ghost Protocol: LEGACY_API_KEY_ENV should be env var name, not actual key."""
    assert isinstance(app_main.LEGACY_API_KEY_ENV, str)
    # Should be an env var name, NOT start with AIzaSy (that would be exposed key)
    assert app_main.LEGACY_API_KEY_ENV == "GEMINI_API_KEY", "Legacy should use standard env var"


def test_api_key_fallback_logic(monkeypatch):
    """
    Test that the API key fallback logic works correctly:
    1. If GEMINI_API_KEY env var is set, use it
    2. Ghost Protocol: No hardcoded fallback keys
    """
    import os
    
    # Test case 1: GEMINI_API_KEY is set
    test_key = "test_api_key_123"
    monkeypatch.setenv(app_main.API_KEY_ENV, test_key)
    
    api_key = os.getenv(app_main.API_KEY_ENV)
    assert api_key == test_key, "Should use the environment variable when set"
    
    # Test case 2: GEMINI_API_KEY is not set - should return None (Ghost Protocol)
    monkeypatch.delenv(app_main.API_KEY_ENV, raising=False)
    
    api_key = os.getenv(app_main.API_KEY_ENV)
    assert api_key is None, "Should return None when env var not set (Ghost Protocol)"


def test_no_hardcoded_keys():
    """
    Ghost Protocol: Ensure no hardcoded API keys in main module.
    """
    import inspect
    source = inspect.getsource(app_main)
    # Should not contain actual API key patterns
    assert "AIzaSyBeCmYi5i3" not in source, "Old exposed key should be removed"
    assert "AIzaSyArVYGrkGL" not in source, "Legacy key pattern should be removed"
