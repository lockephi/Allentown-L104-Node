#!/usr/bin/env python3
"""
L104 UNIFIED PROVIDER ORCHESTRATOR - EVO_48
═══════════════════════════════════════════════════════════════════════════════
Centralized multi-provider management with intelligent routing.
"""

import os
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
import httpx

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


GOD_CODE = 527.5184818492611
PHI = 1.618033988749895

logger = logging.getLogger("L104_PROVIDERS")


class ProviderStatus(Enum):
    AVAILABLE = auto()
    RATE_LIMITED = auto()
    ERROR = auto()
    DISABLED = auto()


@dataclass
class ProviderState:
    name: str
    base_url: str
    api_key: Optional[str]
    models: List[str]
    status: ProviderStatus = ProviderStatus.AVAILABLE
    last_call: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    rate_limit_reset: float = 0.0
    current_model_index: int = 0

    @property
    def is_available(self) -> bool:
        if self.status == ProviderStatus.RATE_LIMITED:
            if time.time() > self.rate_limit_reset:
                self.status = ProviderStatus.AVAILABLE
        return self.status == ProviderStatus.AVAILABLE and self.api_key is not None

    @property
    def current_model(self) -> str:
        return self.models[self.current_model_index % len(self.models)]

    def rotate_model(self):
        self.current_model_index = (self.current_model_index + 1) % len(self.models)

    @property
    def reliability_score(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total


PROVIDER_CONFIGS = {
        "gemini": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "env_key": "GEMINI_API_KEY",
                "models": [
                        "gemini-2.0-flash-exp",
                        "gemini-1.5-flash",
                        "gemini-1.5-pro"
                ]
        },
        "openai": {
                "base_url": "https://api.openai.com/v1",
                "env_key": "OPENAI_API_KEY",
                "models": [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-4-turbo"
                ]
        },
        "anthropic": {
                "base_url": "https://api.anthropic.com/v1",
                "env_key": "ANTHROPIC_API_KEY",
                "models": [
                        "claude-3-5-sonnet-20241022",
                        "claude-3-opus-20240229"
                ]
        },
        "deepseek": {
                "base_url": "https://api.deepseek.com/v1",
                "env_key": "DEEPSEEK_API_KEY",
                "models": [
                        "deepseek-chat",
                        "deepseek-reasoner"
                ]
        },
        "groq": {
                "base_url": "https://api.groq.com/openai/v1",
                "env_key": "GROQ_API_KEY",
                "models": [
                        "llama-3.3-70b-versatile",
                        "mixtral-8x7b-32768"
                ]
        }
}


class UnifiedProviderOrchestrator:
    """
    Unified provider orchestration with:
    - Automatic failover
    - Smart routing based on reliability
    - Consensus synthesis
    - Rate limit coordination
    """

    def __init__(self):
        self.providers: Dict[str, ProviderState] = {}
        self._initialize_providers()
        self._lock = asyncio.Lock()

    def _initialize_providers(self):
        for name, config in PROVIDER_CONFIGS.items():
            api_key = os.getenv(config["env_key"])
            self.providers[name] = ProviderState(
                name=name,
                base_url=config["base_url"],
                api_key=api_key,
                models=config["models"]
            )

    def get_available_providers(self) -> List[ProviderState]:
        """Get list of currently available providers."""
        return [p for p in self.providers.values() if p.is_available]

    def get_best_provider(self) -> Optional[ProviderState]:
        """Get the most reliable available provider."""
        available = self.get_available_providers()
        if not available:
            return None
        return max(available, key=lambda p: p.reliability_score)

    def get_providers_by_priority(self) -> List[ProviderState]:
        """Get providers sorted by reliability."""
        available = self.get_available_providers()
        return sorted(available, key=lambda p: p.reliability_score, reverse=True)

    async def call_provider(
        self,
        provider: ProviderState,
        prompt: str,
        timeout: float = 30.0
    ) -> Optional[str]:
        """Call a specific provider."""
        if not provider.is_available:
            return None

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Provider-specific request formatting
                if provider.name == "gemini":
                    response = await self._call_gemini(client, provider, prompt)
                elif provider.name == "openai":
                    response = await self._call_openai(client, provider, prompt)
                elif provider.name == "anthropic":
                    response = await self._call_anthropic(client, provider, prompt)
                else:
                    response = await self._call_openai_compatible(client, provider, prompt)

                provider.success_count += 1
                provider.last_call = time.time()
                return response

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                provider.status = ProviderStatus.RATE_LIMITED
                provider.rate_limit_reset = time.time() + 60
            provider.failure_count += 1
            return None
        except Exception as e:
            logger.warning(f"Provider {provider.name} error: {e}")
            provider.failure_count += 1
            return None

    async def _call_gemini(self, client, provider, prompt) -> str:
        url = f"{provider.base_url}/models/{provider.current_model}:generateContent"
        response = await client.post(
            url,
            params={"key": provider.api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_openai(self, client, provider, prompt) -> str:
        response = await client.post(
            f"{provider.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {provider.api_key}"},
            json={
                "model": provider.current_model,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def _call_anthropic(self, client, provider, prompt) -> str:
        response = await client.post(
            f"{provider.base_url}/messages",
            headers={
                "x-api-key": provider.api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": provider.current_model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]

    async def _call_openai_compatible(self, client, provider, prompt) -> str:
        response = await client.post(
            f"{provider.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {provider.api_key}"},
            json={
                "model": provider.current_model,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def query_with_failover(
        self,
        prompt: str,
        max_attempts: int = 3
    ) -> Optional[str]:
        """Query with automatic failover to backup providers."""
        providers = self.get_providers_by_priority()

        for provider in providers[:max_attempts]:
            result = await self.call_provider(provider, prompt)
            if result:
                return result
            provider.rotate_model()  # Try different model on failure

        return None

    async def query_consensus(
        self,
        prompt: str,
        min_responses: int = 2
    ) -> Dict[str, Any]:
        """Query multiple providers and synthesize consensus."""
        providers = self.get_available_providers()
        if len(providers) < min_responses:
            single = await self.query_with_failover(prompt)
            return {"consensus": single, "responses": [single] if single else [], "agreement": 1.0}

        # Query in parallel
        tasks = [self.call_provider(p, prompt) for p in providers[:4]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = [r for r in results if isinstance(r, str) and r]

        if not responses:
            return {"consensus": None, "responses": [], "agreement": 0.0}

        # Simple consensus: return longest response (usually most complete)
        consensus = max(responses, key=len)

        return {
            "consensus": consensus,
            "responses": responses,
            "agreement": len(responses) / len(providers),
            "provider_count": len(responses)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "providers": {
                name: {
                    "available": p.is_available,
                    "status": p.status.name,
                    "model": p.current_model,
                    "reliability": p.reliability_score,
                    "success": p.success_count,
                    "failures": p.failure_count
                }
                for name, p in self.providers.items()
            },
            "available_count": len(self.get_available_providers()),
            "god_code": GOD_CODE
        }


# Global instance
_orchestrator: Optional[UnifiedProviderOrchestrator] = None

def get_orchestrator() -> UnifiedProviderOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = UnifiedProviderOrchestrator()
    return _orchestrator
