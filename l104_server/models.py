"""L104 Server — Pydantic request/response models."""
import os
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

class ChatRequest(BaseModel):
    """Chat request with input validation (EVO_62: added length bounds and auto-strip)"""
    message: str = Field(..., min_length=1, max_length=32_768, description="User message (max 32KB)")
    use_sovereign_context: bool = True
    local_only: bool = False

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, v: str) -> str:
        """Strip leading/trailing whitespace from message"""
        if isinstance(v, str):
            return v.strip()
        return v

class TrainingRequest(BaseModel):
    query: str
    response: str
    quality: float = 1.0

class ProviderStatus(BaseModel):
    gemini: bool = False
    derivation: bool = True
    local: bool = True

# State
provider_status = ProviderStatus()

# Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

