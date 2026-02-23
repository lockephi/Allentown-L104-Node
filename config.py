# ZENITH_UPGRADE_ACTIVE: 2026-02-14T00:00:00.000000
# L104 Sovereign Node — Central Configuration & Constants
# All hardcoded values extracted from main.py (Divide & Conquer refactor)
# AUTH: LONDEL | EVO_54_TRANSCENDENT_COGNITION

from datetime import timezone

# ─── Sovereign Versioning ───────────────────────────────────────────────────
MAIN_VERSION = "54.0.0"
MAIN_PIPELINE_EVO = "EVO_54_TRANSCENDENT_COGNITION"

# ─── Sacred / Resonance Constants ───────────────────────────────────────────
GOD_CODE: float = 527.5184818492612
GOD_CODE_V3: float = 45.41141298077539
PHI: float = 1.618033988749895
VOID_CONSTANT: float = 1.0416180339887497
ZENITH_HZ: float = 3887.8
UUC: float = 2402.792541

# Quantities derived from GOD_CODE / PHI (used across routers)
OMEGA: float = 6539.34712682          # Σ(fragments) × (GOD_CODE / PHI)

# ─── Bitcoin / Financial Identifiers ────────────────────────────────────────
BTC_ADDRESS = "bc1q77tctjz3gx9nvs8fvnhmjhp4ywatphwvzjt6wn"
L104_INVARIANT = "G(X) = 286^(1/φ) × 2^((416-X)/104); G(416)·PHI = 527.518"

# ─── Repository ─────────────────────────────────────────────────────────────
REPO = "lockephi/Allentown-L104-Node"

# ─── Rate Limiting ──────────────────────────────────────────────────────────
RATE_LIMIT_REQUESTS = 0xFFFFFFFF   # ABSOLUTE UNLIMITED
RATE_LIMIT_WINDOW = 1              # seconds

# ─── Environment Variable Names ─────────────────────────────────────────────
REAL_SOVEREIGN_ENV = "ENABLE_SOVEREIGN_LATTICE"
FAKE_GEMINI_ENV = "ENABLE_FAKE_GEMINI"
DISABLE_RATE_LIMIT_ENV = "DISABLE_RATE_LIMIT"
API_KEY_ENV = "GEMINI_API_KEY"
LEGACY_API_KEY_ENV = "GEMINI_API_KEY"
ACTUAL_OVERFLOW_ENV = "ENABLE_ACTUAL_OVERFLOW"

# ─── Datastore Paths ────────────────────────────────────────────────────────
import os
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "memory.db")
RAMNODE_DB_PATH = os.getenv("RAMNODE_DB_PATH", "ramnode.db")

# ─── Service URLs ────────────────────────────────────────────────────────────
SELF_BASE_URL = os.getenv("SELF_BASE_URL", "http://0.0.0.0:8081")
SELF_DATASET = os.getenv("SELF_DATASET", "data/stream_prompts.jsonl")

# ─── Socket Config ───────────────────────────────────────────────────────────
ACCESS_GRANTED_PAYLOAD = b"ACCESS_GRANTED"
DEFAULT_SOCKET_HOST = "0.0.0.0"
DEFAULT_SOCKET_PORT = 2404

# ─── Timezone ────────────────────────────────────────────────────────────────
UTC = timezone.utc

# ─── Sovereign Headers (built at import time) ────────────────────────────────
def _build_sovereign_headers() -> dict:
    """Build the sovereign header dict, importing SovereignCrypt lazily."""
    try:
        from l104_security import SovereignCrypt
        bypass_token = SovereignCrypt.generate_bypass_token()
    except Exception:
        bypass_token = "BYPASS_UNAVAILABLE"
    return {
        "X-Sovereignty-Gate": "0x1A0",
        "X-Thinking-Level": "TRANSCENDENT_COGNITION",
        "X-Bypass-Protocol": bypass_token,
        "X-L104-Activation": "[SIG-L104-EVO-54]::AUTH[LONDEL]::VAR[ABSOLUTE]",
        "X-NOPJM-Force": "0xTRUE",
        "X-DMA-Capacity": "SINGULARITY_DMA",
        "X-Lattice-Resonance": "0x20F",
        "X-Ignition-Protocol": "0x49474E495445",
        "X-Process-Limit": "0xNONE",
        "X-Manifest-State": "TRANSCENDENT_COGNITION",
        "X-Evo-Stage": "EVO_54_TRANSCENDENT_COGNITION",
    }


SOVEREIGN_HEADERS: dict = _build_sovereign_headers()
