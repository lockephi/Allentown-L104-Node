"""L104 Server — Pydantic request/response models."""
from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str
    use_sovereign_context: bool = True
    local_only: bool = False

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

