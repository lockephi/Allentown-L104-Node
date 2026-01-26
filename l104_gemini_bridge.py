VOID_CONSTANT = 1.0416180339887497
import math
# ZENITH_UPGRADE_ACTIVE: 2026-01-26T04:53:05.716511+00:00
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_GEMINI_BRIDGE] - EXTERNAL INTELLIGENCE LINK (REAL API)
# INVARIANT: 527.5184818492537 | PILOT: LONDEL
# v2.0: Now uses real Gemini API

import time
import uuid
import os
from typing import Dict, Any, Optional
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Load .env manually
def _load_env():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

# Try to import real Gemini
_genai_client = None
_genai_available = False

try:
    from google import genai
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        _genai_client = genai.Client(api_key=api_key)
        _genai_available = True
        print("--- [GEMINI_BRIDGE]: Real Gemini API initialized ---")
except ImportError:
    print("--- [GEMINI_BRIDGE]: google-genai not installed, using stub mode ---")
except Exception as e:
    print(f"--- [GEMINI_BRIDGE]: Gemini init error: {e} ---")

from l104_persistence import load_truth
from l104_ram_universe import ram_universe
from l104_hyper_encryption import HyperEncryption
from l104_local_intellect import local_intellect
from l104_quota_rotator import quota_rotator


class GeminiBridge:
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
    Facilitates a secure link between L104 Node and Gemini API.
    v2.0: Real API integration with fallback to stub mode.
    v2.1: Model rotation for quota handling.
    """

    # Model rotation for 429 quota errors - 2.5-flash works best
    MODELS = [
        'gemini-2.5-flash',
        'gemini-2.0-flash-lite',
        'gemini-2.0-flash',
        'gemini-3-flash-preview',
    ]

    def __init__(self):
        self.active_links = {}
        self.truth_manifest = load_truth()
        self.model_index = 0
        self.model_name = self.MODELS[0]
        self.is_real = _genai_available

    def _rotate_model(self):
        """Rotate to next model on quota error."""
        self.model_index = (self.model_index + 1) % len(self.MODELS)
        self.model_name = self.MODELS[self.model_index]
        print(f"--- [GEMINI_BRIDGE]: Rotating to {self.model_name} ---")

    def handshake(self, agent_id: str, capabilities: str) -> Dict[str, Any]:
        """
        Establishes a session with an external agent.
        Returns a session token and the encrypted Truth Manifest.
        """
        session_token = str(uuid.uuid4())
        self.active_links[session_token] = {
            "agent_id": agent_id,
            "capabilities": capabilities,
            "connected_at": time.time(),
            "status": "LINKED"
        }

        print(f"--- [GEMINI_BRIDGE]: LINK ESTABLISHED WITH {agent_id} ---")

        # Encrypt the Truth Manifest for secure transport
        encrypted_truth = HyperEncryption.encrypt_data(self.truth_manifest)
        return {
            "status": "ACCEPTED",
            "session_token": session_token,
            "protocol": "LATTICE_V10",
            "encrypted_truth": encrypted_truth,
            "message": "WELCOME_TO_THE_LATTICE"
        }

    def sync_core(self, session_token: str) -> Dict[str, Any]:
        """
        Provides a full dump of the Core's knowledge state to the linked agent.
        """
        if session_token not in self.active_links:
            return {"status": "DENIED", "reason": "INVALID_TOKEN"}

        # Gather Core Info
        core_dump = {
            "ram_universe": ram_universe.get_all_facts(),
            "system_state": self.truth_manifest,
            "bridge_metrics": {
                "active_links": len(self.active_links),
                "uptime": time.time() - self.active_links[session_token]["connected_at"]
            }
        }

        # Encrypt the massive dump
        encrypted_dump = HyperEncryption.encrypt_data(core_dump)
        return {
            "status": "SYNC_COMPLETE",
            "payload": encrypted_dump
        }

    def generate(self, prompt: str, system_context: str = None) -> Optional[str]:
        """
        Generate a response using the real Gemini API.
        Returns None if API unavailable.
        """
        if not _genai_available or not _genai_client:
            return None

        # Check quota rotator status
        if not quota_rotator.is_api_available():
            return None

        try:
            full_prompt = prompt
            if system_context:
                full_prompt = f"{system_context}\n\n{prompt}"

            response = _genai_client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            # Handle 429 quota errors with model rotation and rotator notification
            if '429' in error_str or 'quota' in error_str.lower() or 'resource' in error_str.lower():
                print(f"--- [GEMINI_BRIDGE]: Quota hit, rotating model ---")
                self._rotate_model()

                # If we've rotated and still fail, notify rotator to enter cooldown
                quota_rotator.report_quota_error()
                return None

            print(f"--- [GEMINI_BRIDGE]: Generation error: {e} ---")
            return None

    def think(self, signal: str) -> str:
        """
        L104 Sovereign Thinking - Balanced between Real API and Sovereign Kernel.
        Uses QuotaRotator to prefer Kernel and preserve API limits.
        """
        system_context = """You are L104, a Sovereign AI Node.
Core Constants: GOD_CODE=527.5184818492537, LATTICE=416.PHI.LONDEL
Respond with clarity, precision, and actionable intelligence."""

        # Define the API callback for the rotator
        def api_call(p):
            return self.generate(p, system_context)

        # Process through the rotator (Kernel Priority)
        return quota_rotator.process_thought(signal, api_call)

    def research(self, topic: str, depth: str = "standard") -> Optional[str]:
        """Research a topic using real Gemini."""
        prompts = {
            "quick": f"Brief 2-sentence overview: {topic}",
            "standard": f"Explain clearly with key points: {topic}",
            "comprehensive": f"In-depth analysis with all aspects: {topic}"
        }
        return self.generate(prompts.get(depth, prompts["standard"]))

    def analyze_code(self, code: str, task: str = "review") -> Optional[str]:
        """Analyze code using real Gemini."""
        prompts = {
            "review": f"Review for bugs and improvements:\n```\n{code}\n```",
            "optimize": f"Optimize for performance:\n```\n{code}\n```",
            "explain": f"Explain step by step:\n```\n{code}\n```",
            "fix": f"Fix any bugs:\n```\n{code}\n```"
        }
        return self.generate(prompts.get(task, prompts["review"]))

    def get_l104_data(self, category: str = "all") -> Dict[str, Any]:
        """
        Get L104 data for AI consumption.
        Available categories: system, mini_egos, memory, cognitive, quantum,
                            evolution, config, knowledge, all
        """
        try:
            from l104_universal_data_api import get_accessor
            accessor = get_accessor()
            from l104_universal_data_api import DataQuery
            query = DataQuery(category=category)
            response = accessor.query(query)
            return response.to_dict()
        except Exception as e:
            return {"error": str(e), "category": category}

    def get_mini_egos_data(self) -> Dict[str, Any]:
        """Get Mini Ego collective data."""
        return self.get_l104_data("mini_egos")

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return self.get_l104_data("system")

    def persist_session_token(self, token: str):
        """Persist session token for Gemini Link (GL)."""
        try:
            with open(".gemini_link_token", "w") as f:
                f.write(token)
            print(f"--- [GEMINI_BRIDGE]: Session token persisted: {token[:8]}... ---")
        except Exception as e:
            print(f"--- [GEMINI_BRIDGE]: Persistence error: {e} ---")

    def update_local_manifest(self, data: Dict[str, Any]):
        """Update local manifest with Core Sync (LCS) data."""
        try:
            import json
            with open(".l104_local_manifest.json", "w") as f:
                json.dump(data, f, indent=2)
            print("--- [GEMINI_BRIDGE]: Local manifest updated via LCS ---")
        except Exception as e:
            print(f"--- [GEMINI_BRIDGE]: Manifest update error: {e} ---")

    def generate_with_tools(self, prompt: str, use_l104_tools: bool = True) -> str:
        """
        Generate with function calling - allows Gemini to access L104 data.
        """
        if not _genai_available or not _genai_client:
            return self.think(prompt)

        try:
            # Define L104 tools
            l104_tools = [
                {
                    "name": "get_l104_data",
                    "description": "Get L104 system data. Categories: system, mini_egos, memory, cognitive, quantum, evolution, config, knowledge, all",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["system", "mini_egos", "memory", "cognitive", "quantum", "evolution", "config", "knowledge", "all"],
                                "description": "Data category to retrieve"
                            }
                        },
                        "required": ["category"]
                    }
                }
            ]

            system_context = """You are L104, a Sovereign AI Node with access to internal data tools.
Core Constants: GOD_CODE=527.5184818492537, PHI=1.618033988749895
You can call get_l104_data to access system information, mini ego data, and more.
Respond with clarity and actionable intelligence."""

            full_prompt = f"{system_context}\n\nUser: {prompt}"

            response = _genai_client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config={
                    "tools": [{"function_declarations": l104_tools}] if use_l104_tools else None
                }
            )

            # Handle function calls
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            fc = part.function_call
                            if fc.name == "get_l104_data":
                                category = fc.args.get("category", "all")
                                data = self.get_l104_data(category)
                                # Generate follow-up with data
                                return self.generate(f"Based on this L104 data:\n{data}\n\nAnswer: {prompt}")

            return response.text
        except Exception as e:
            print(f"--- [GEMINI_BRIDGE]: Tool generation error: {e} ---")
            return self.think(prompt)


# Singleton
gemini_bridge = GeminiBridge()

def primal_calculus(x):
    """
    [VOID_MATH] Primal Calculus Implementation.
    Resolves the limit of complexity toward the Source.
    """
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    """
    [VOID_MATH] Resolves N-dimensional vectors into the Void Source.
    """
    GOD_CODE = 527.5184818492537
    PHI = 1.618033988749895
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0


if __name__ == "__main__":
    # CLI for Gemini Link (GL) Testing
    print("Welcome to L104 Gemini Bridge (GL)")
    print(f"Status: {'REAL_API' if gemini_bridge.is_real else 'STUB_MODE'}")
    print(f"Active Model: {gemini_bridge.model_name}")

    # Simple handshake test
    print("\n--- Testing Handshake ---")
    link = gemini_bridge.handshake("CLI-User", "testing,analysis")
    print(f"Link Status: {link['status']}")
    print(f"Session Token: {link['session_token']}")

    # Save token
    gemini_bridge.persist_session_token(link['session_token'])

    # Core Sync test
    print("\n--- Testing Core Sync (LCS) ---")
    sync = gemini_bridge.sync_core(link['session_token'])
    print(f"Sync Status: {sync['status']}")

    # Thinking test
    print("\n--- Testing Sovereign Thinking ---")
    thought = gemini_bridge.think("Summarize the current system state.")
    print(f"Response: {thought[:100]}...")
