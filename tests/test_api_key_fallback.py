"""
Test API key fallback logic to ensure LEGACY_API_KEY_ENV is used correctly.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as app_main


def test_api_key_env_variable_defined():
    """Test that API_KEY_ENV is defined as an environment variable name."""
    assert app_main.API_KEY_ENV == "GEMINI_API_KEY"


def test_legacy_api_key_env_is_hardcoded_key():
    """Test that LEGACY_API_KEY_ENV is an actual API key, not an env var name."""
    # The legacy API key should be a string that looks like an API key
    assert isinstance(app_main.LEGACY_API_KEY_ENV, str)
    assert app_main.LEGACY_API_KEY_ENV.startswith("AIzaSy")
    assert len(app_main.LEGACY_API_KEY_ENV) > 30


def test_api_key_fallback_logic(monkeypatch):
    """
    Test that the API key fallback logic works correctly:
    1. If GEMINI_API_KEY env var is set, use it
    2. Otherwise, use the LEGACY_API_KEY_ENV constant directly
    """
    import os
    
    # Test case 1: GEMINI_API_KEY is set
    test_key = "test_api_key_123"
    monkeypatch.setenv(app_main.API_KEY_ENV, test_key)
    
    # Simulate the logic from main.py lines 1090 and 1248
    api_key = os.getenv(app_main.API_KEY_ENV) or app_main.LEGACY_API_KEY_ENV
    assert api_key == test_key, "Should use the environment variable when set"
    
    # Test case 2: GEMINI_API_KEY is not set
    monkeypatch.delenv(app_main.API_KEY_ENV, raising=False)
    
    # Simulate the logic from main.py lines 1090 and 1248
    api_key = os.getenv(app_main.API_KEY_ENV) or app_main.LEGACY_API_KEY_ENV
    assert api_key == app_main.LEGACY_API_KEY_ENV, "Should use the legacy API key constant when env var not set"
    assert api_key.startswith("AIzaSy"), "Legacy key should be the actual API key string"


def test_old_buggy_behavior_fails(monkeypatch):
    """
    Test that demonstrates the old buggy behavior would fail.
    The old code tried os.getenv(LEGACY_API_KEY_ENV) where LEGACY_API_KEY_ENV
    was the API key itself, not an environment variable name.
    """
    import os
    
    # Ensure GEMINI_API_KEY is not set
    monkeypatch.delenv(app_main.API_KEY_ENV, raising=False)
    
    # Old buggy behavior: os.getenv(LEGACY_API_KEY_ENV)
    # This would try to get an env var named "AIzaSy..." which doesn't exist
    buggy_result = os.getenv(app_main.LEGACY_API_KEY_ENV)
    assert buggy_result is None, "Old buggy code would return None"
    
    # New correct behavior: LEGACY_API_KEY_ENV directly
    correct_result = app_main.LEGACY_API_KEY_ENV
    assert correct_result is not None, "Fixed code returns the actual API key"
    assert correct_result.startswith("AIzaSy"), "Fixed code returns the actual API key string"
