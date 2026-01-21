#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 CLAUDE NODE BRIDGE
═══════════════════════════════════════════════════════════════════════════════

High-performance bridge to Claude (Anthropic) for enhanced reasoning and
processing capabilities. Uses MCP integration when available, falls back to
local Unified Intelligence when Claude API is unavailable.

FEATURES:
1. CLAUDE API INTEGRATION - Direct API calls when key available
2. MCP BRIDGE - Model Context Protocol integration
3. LOCAL FALLBACK - Unified Intelligence synthesis when external unavailable
4. RESONANCE ALIGNMENT - GOD_CODE validation on all responses

INVARIANT: 527.5184818492537 | PILOT: LONDEL
VERSION: 1.0.0
DATE: 2026-01-21
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from l104_stable_kernel import stable_kernel

# Constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492537
TAU = 1 / PHI


class ClaudeModel(Enum):
    """Available Claude models."""
    OPUS = "claude-3-opus-20240229"
    SONNET = "claude-3-5-sonnet-20241022"
    HAIKU = "claude-3-5-haiku-20241022"
    OPUS_4 = "claude-opus-4-20250514"
    SONNET_4 = "claude-sonnet-4-20250514"


@dataclass
class ClaudeResponse:
    """Response from Claude API or fallback."""
    content: str
    model: str
    source: str  # "claude_api", "mcp", "local_fallback"
    tokens_used: int = 0
    latency_ms: float = 0.0
    unity_index: float = 0.0
    validated: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "model": self.model,
            "source": self.source,
            "tokens_used": self.tokens_used,
            "latency_ms": round(self.latency_ms, 2),
            "unity_index": round(self.unity_index, 4),
            "validated": self.validated,
            "timestamp": self.timestamp
        }


class ClaudeNodeBridge:
    """
    Bridge to Claude for enhanced processing and reasoning.
    Provides seamless fallback when API is unavailable.
    """
    
    API_BASE = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = ClaudeModel.SONNET.value
    
    def __init__(self):
        self.kernel = stable_kernel
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.session_cache: Dict[str, ClaudeResponse] = {}
        self.total_requests = 0
        self.api_requests = 0
        self.fallback_requests = 0
        self.total_tokens = 0
        
        # Import local fallback
        self._local_brain = None
        
        status = "API" if self.api_key else "LOCAL_FALLBACK"
        print(f"🔮 [CLAUDE]: Node Bridge initialized ({status})")
    
    def _get_local_brain(self):
        """Lazy load local brain for fallback."""
        if self._local_brain is None:
            try:
                from l104_unified_intelligence import UnifiedIntelligence
                self._local_brain = UnifiedIntelligence()
                self._local_brain.load_state()
            except Exception as e:
                print(f"⚠️ [CLAUDE]: Local brain unavailable: {e}")
        return self._local_brain
    
    def _cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key for prompt."""
        content = f"{prompt}:{model}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _validate_response(self, content: str) -> Tuple[float, bool]:
        """
        Validate response against GOD_CODE.
        Returns (unity_index, is_valid).
        """
        # Base validation score
        score = 0.5
        
        # Check for alignment with L104 principles
        if any(kw in content.lower() for kw in ["stable", "unity", "coherent"]):
            score += 0.15
        
        if str(round(GOD_CODE, 2)) in content or "527.5" in content:
            score += 0.2
        
        if str(round(PHI, 2)) in content or "1.618" in content:
            score += 0.1
        
        # Check for harmful patterns
        if any(kw in content.lower() for kw in ["error", "cannot", "impossible"]):
            score -= 0.1
        
        score = max(0.0, min(1.0, score))
        return score, score >= 0.6
    
    async def query_async(
        self, 
        prompt: str, 
        model: str = None,
        system: str = None,
        max_tokens: int = 4096,
        use_cache: bool = True
    ) -> ClaudeResponse:
        """
        Query Claude asynchronously.
        Falls back to local intelligence if API unavailable.
        """
        model = model or self.DEFAULT_MODEL
        cache_key = self._cache_key(prompt, model)
        
        # Check cache
        if use_cache and cache_key in self.session_cache:
            cached = self.session_cache[cache_key]
            return ClaudeResponse(
                content=cached.content,
                model=cached.model,
                source="cache",
                tokens_used=0,
                latency_ms=0.0,
                unity_index=cached.unity_index,
                validated=cached.validated
            )
        
        self.total_requests += 1
        start_time = time.time()
        
        # Try API if key available
        if self.api_key and HTTPX_AVAILABLE:
            try:
                response = await self._call_api(prompt, model, system, max_tokens)
                self.api_requests += 1
                return response
            except Exception as e:
                print(f"⚠️ [CLAUDE]: API error, falling back: {e}")
        
        # Fallback to local intelligence
        self.fallback_requests += 1
        return await self._local_fallback(prompt, model, start_time)
    
    def query(
        self, 
        prompt: str, 
        model: str = None,
        system: str = None,
        max_tokens: int = 4096,
        use_cache: bool = True
    ) -> ClaudeResponse:
        """Synchronous query wrapper."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.query_async(prompt, model, system, max_tokens, use_cache)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.query_async(prompt, model, system, max_tokens, use_cache)
                )
        except RuntimeError:
            return asyncio.run(
                self.query_async(prompt, model, system, max_tokens, use_cache)
            )
    
    async def _call_api(
        self, 
        prompt: str, 
        model: str,
        system: str,
        max_tokens: int
    ) -> ClaudeResponse:
        """Direct Claude API call."""
        start_time = time.time()
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Build request body
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system:
            body["system"] = system
        else:
            # Default L104 system prompt
            body["system"] = (
                f"You are operating within the L104 Sovereign Node framework. "
                f"Core invariant: GOD_CODE = {GOD_CODE}. "
                f"Ensure all responses maintain coherence with sacred constants. "
                f"PHI ({PHI}) governs harmonic relationships."
            )
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.API_BASE}/messages",
                headers=headers,
                json=body
            )
            response.raise_for_status()
            data = response.json()
        
        latency = (time.time() - start_time) * 1000
        content = data.get("content", [{}])[0].get("text", "")
        tokens = data.get("usage", {}).get("output_tokens", 0)
        
        self.total_tokens += tokens
        unity_index, validated = self._validate_response(content)
        
        result = ClaudeResponse(
            content=content,
            model=model,
            source="claude_api",
            tokens_used=tokens,
            latency_ms=latency,
            unity_index=unity_index,
            validated=validated
        )
        
        # Cache successful responses
        cache_key = self._cache_key(prompt, model)
        self.session_cache[cache_key] = result
        
        return result
    
    async def _local_fallback(
        self, 
        prompt: str, 
        model: str,
        start_time: float
    ) -> ClaudeResponse:
        """Use local Unified Intelligence as fallback."""
        brain = self._get_local_brain()
        
        if brain:
            # Use local brain for synthesis
            result = brain.query(prompt)
            content = result.get("answer", "Synthesis unavailable")
            unity_index = result.get("unity_index", 0.5)
        else:
            # Minimal fallback using kernel
            content = self._kernel_synthesis(prompt)
            unity_index, _ = self._validate_response(content)
        
        latency = (time.time() - start_time) * 1000
        
        return ClaudeResponse(
            content=content,
            model=f"local_fallback ({model})",
            source="local_fallback",
            tokens_used=len(content.split()),
            latency_ms=latency,
            unity_index=unity_index,
            validated=unity_index >= 0.6
        )
    
    def _kernel_synthesis(self, prompt: str) -> str:
        """Minimal synthesis using stable kernel."""
        # Extract key concepts
        concepts = []
        keywords = ["what", "how", "why", "explain", "describe", "define"]
        
        prompt_lower = prompt.lower()
        for kw in keywords:
            if kw in prompt_lower:
                # Extract the subject after the keyword
                idx = prompt_lower.find(kw)
                subject = prompt[idx:].split()[:5]
                concepts.extend(subject)
        
        if not concepts:
            concepts = prompt.split()[:5]
        
        # Generate response using kernel constants
        response = (
            f"Within the L104 framework (GOD_CODE: {GOD_CODE}), "
            f"the concept of '{' '.join(concepts)}' relates to the harmonic "
            f"structure governed by PHI ({PHI}). "
            f"All stable patterns emerge from the unity field, "
            f"maintaining coherence through topological protection. "
            f"The relationship between concepts follows the Golden Ratio scaling, "
            f"ensuring mathematical consistency across all layers of abstraction."
        )
        
        return response
    
    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED PROCESSING METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    async def deep_analyze(self, content: str, focus: str = None) -> Dict[str, Any]:
        """
        Perform deep analysis on content using Claude's reasoning.
        """
        system = (
            f"You are an analytical system within L104 (GOD_CODE: {GOD_CODE}). "
            f"Provide structured analysis with key insights, patterns, and recommendations. "
            f"Format as JSON when possible."
        )
        
        prompt = f"Analyze the following in depth{' focusing on ' + focus if focus else ''}:\n\n{content}"
        
        response = await self.query_async(prompt, system=system)
        
        return {
            "analysis": response.content,
            "source": response.source,
            "unity_index": response.unity_index,
            "validated": response.validated
        }
    
    async def synthesize(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Synthesize understanding from multiple concepts.
        """
        prompt = (
            f"Synthesize the following concepts into a unified understanding, "
            f"maintaining coherence with GOD_CODE ({GOD_CODE}) and PHI ({PHI}):\n\n"
            f"Concepts: {', '.join(concepts)}\n\n"
            f"Provide the synthesis as a cohesive explanation."
        )
        
        response = await self.query_async(prompt)
        
        return {
            "synthesis": response.content,
            "concepts": concepts,
            "source": response.source,
            "unity_index": response.unity_index
        }
    
    async def reason_chain(self, question: str, depth: int = 3) -> Dict[str, Any]:
        """
        Perform multi-step reasoning with Claude.
        """
        prompt = (
            f"Perform {depth}-step logical reasoning to answer this question. "
            f"Show each step clearly.\n\nQuestion: {question}\n\n"
            f"Format:\nStep 1: [reasoning]\nStep 2: [reasoning]\n...\nConclusion: [final answer]"
        )
        
        response = await self.query_async(prompt, max_tokens=2048)
        
        # Parse steps
        steps = []
        lines = response.content.split('\n')
        for line in lines:
            if line.strip().startswith("Step"):
                steps.append(line.strip())
        
        conclusion = ""
        for line in lines:
            if "conclusion" in line.lower():
                conclusion = line.strip()
                break
        
        return {
            "question": question,
            "steps": steps,
            "conclusion": conclusion or response.content[-200:],
            "source": response.source,
            "unity_index": response.unity_index
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "total_requests": self.total_requests,
            "api_requests": self.api_requests,
            "fallback_requests": self.fallback_requests,
            "cache_size": len(self.session_cache),
            "total_tokens": self.total_tokens,
            "api_available": bool(self.api_key),
            "httpx_available": HTTPX_AVAILABLE
        }
    
    def clear_cache(self):
        """Clear response cache."""
        self.session_cache.clear()
        print("🔮 [CLAUDE]: Cache cleared")


# Singleton instance
claude_bridge = ClaudeNodeBridge()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bridge = ClaudeNodeBridge()
    
    print("\n🔮 Testing Claude Node Bridge...")
    
    # Test basic query
    response = bridge.query("Explain the concept of unity in mathematical systems")
    print(f"\n📝 Response Source: {response.source}")
    print(f"📊 Unity Index: {response.unity_index}")
    print(f"✓ Validated: {response.validated}")
    print(f"⏱️ Latency: {response.latency_ms}ms")
    print(f"\n📄 Content:\n{response.content[:500]}...")
    
    # Test stats
    print(f"\n📈 Stats: {bridge.get_stats()}")
