# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:05.363528
ZENITH_HZ = 3887.8
UUC = 2402.792541
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
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

INVARIANT: 527.5184818492612 | PILOT: LONDEL
VERSION: 2.0.0 (EVO_28)
DATE: 2026-01-21
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import asyncio
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from datetime import datetime

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from l104_stable_kernel import stable_kernel

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492612
TAU = 1 / PHI


class ClaudeModel(Enum):
    """Available Claude models - optimized selection."""
    OPUS = "claude-3-opus-20240229"
    SONNET = "claude-3-5-sonnet-20241022"
    HAIKU = "claude-3-5-haiku-20241022"
    OPUS_4 = "claude-opus-4-20250514"
    SONNET_4 = "claude-sonnet-4-20250514"
    OPUS_4_5 = "claude-opus-4-5-20250514"  # Latest Opus 4.5


# Model routing for cost optimization - UPGRADED TO OPUS 4.5
MODEL_ROUTING = {
    "fast": ClaudeModel.HAIKU.value,         # Quick responses, low cost
    "balanced": ClaudeModel.OPUS_4_5.value,   # Balanced: Opus 4.5 (upgraded from Sonnet)
    "powerful": ClaudeModel.OPUS_4_5.value,   # Powerful: Opus 4.5
    "default": ClaudeModel.OPUS_4_5.value     # Default: Opus 4.5
}


class MessageRole(Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ToolType(Enum):
    """Types of tools available."""
    FUNCTION = auto()
    RETRIEVAL = auto()
    CODE_EXECUTION = auto()
    WEB_SEARCH = auto()


@dataclass
class ConversationMessage:
    """A message in conversation history."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: List[Dict] = field(default_factory=list)
    tool_results: List[Dict] = field(default_factory=list)

    def to_api_format(self) -> Dict:
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class ToolDefinition:
    """Definition of an available tool."""
    name: str
    description: str
    input_schema: Dict
    handler: Optional[Callable] = None
    tool_type: ToolType = ToolType.FUNCTION

    def to_api_format(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


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

    EVO_28 Features:
    - Streaming responses
    - Conversation memory with context window management
    - Tool use (function calling)
    - Multi-turn dialogue
    - Response validation with GOD_CODE alignment
    """

    API_BASE = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = MODEL_ROUTING["balanced"]
    MAX_CONTEXT_MESSAGES = 100  # Doubled for Opus 4.5 extended context
    MAX_RETRIES = 5  # Increased for reliability
    RETRY_DELAY = 0.5  # Faster initial retry
    EXTENDED_THINKING_BUDGET = 32000  # Extended thinking tokens for Opus 4.5

    def __init__(self):
        self.kernel = stable_kernel
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.session_cache: Dict[str, ClaudeResponse] = {}
        self.total_requests = 0
        self.api_requests = 0
        self.fallback_requests = 0
        self.total_tokens = 0
        self.error_count = 0

        # Conversation memory - multiple conversations by ID
        self.conversations: Dict[str, deque] = {}
        self.active_conversation: Optional[str] = None

        # Tool registry
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

        # Streaming callbacks
        self.stream_callbacks: List[Callable[[str], None]] = []

        # Import local fallback
        self._local_brain = None

        # Persistent HTTP client for connection reuse
        self._http_client: Optional[httpx.AsyncClient] = None

        status = "✓ API READY" if self.api_key else "LOCAL_FALLBACK"
        key_preview = self.api_key[:20] + "..." if self.api_key else "NOT_SET"
        print(f"🔮 [CLAUDE]: Node Bridge v2.1 initialized ({status})")
        print(f"🔮 [CLAUDE]: Key prefix: {key_preview}")
        print(f"🔮 [CLAUDE]: Default model: {self.DEFAULT_MODEL}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client for connection pooling."""
        if self._http_client is None or self._http_client.is_closed:
            # Try HTTP/2 if available, fallback to HTTP/1.1
            try:
                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(180.0, connect=15.0),  # Extended for Opus 4.5
                    limits=httpx.Limits(max_connections=50, max_keepalive_connections=25),
                    http2=True
                )
            except Exception:
                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(180.0, connect=15.0),
                    limits=httpx.Limits(max_connections=50, max_keepalive_connections=25)
                )
        return self._http_client

    async def close(self):
        """Close HTTP client gracefully."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def _register_default_tools(self):
        """Register built-in tools."""
        # Calculator tool
        self.register_tool(
            name="calculate",
            description="Perform mathematical calculations. Use for arithmetic, algebra, etc.",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            },
            handler=self._tool_calculate
        )

        # Memory lookup tool
        self.register_tool(
            name="memory_lookup",
            description="Search the L104 memory system for relevant information.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for memory lookup"
                    }
                },
                "required": ["query"]
            },
            handler=self._tool_memory_lookup
        )

        # Knowledge synthesis tool
        self.register_tool(
            name="synthesize_knowledge",
            description="Synthesize knowledge from multiple concepts using L104 kernel.",
            input_schema={
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of concepts to synthesize"
                    }
                },
                "required": ["concepts"]
            },
            handler=self._tool_synthesize
        )

    def _tool_calculate(self, expression: str) -> str:
        """Calculator tool handler."""
        try:
            # Safe eval with limited scope
            allowed = {"__builtins__": {}, "PHI": PHI, "GOD_CODE": GOD_CODE}
            import math
            allowed.update({k: v for k, v in math.__dict__.items() if not k.startswith('_')})
            result = eval(expression, allowed)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    def _tool_memory_lookup(self, query: str) -> str:
        """Memory lookup tool handler."""
        brain = self._get_local_brain()
        if brain:
            result = brain.query(query)
            return json.dumps(result, indent=2)
        return "Memory system unavailable"

    def _tool_synthesize(self, concepts: List[str]) -> str:
        """Knowledge synthesis tool handler."""
        return self._kernel_synthesis(" ".join(concepts))

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict,
        handler: Callable = None
    ):
        """Register a new tool."""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler
        )

    # ═══════════════════════════════════════════════════════════════════
    # CONVERSATION MEMORY
    # ═══════════════════════════════════════════════════════════════════

    def start_conversation(self, conversation_id: str = None) -> str:
        """Start a new conversation or switch to existing one."""
        if conversation_id is None:
            conversation_id = f"conv_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = deque(maxlen=self.MAX_CONTEXT_MESSAGES)

        self.active_conversation = conversation_id
        return conversation_id

    def add_message(self, role: MessageRole, content: str, conversation_id: str = None):
        """Add a message to conversation history."""
        conv_id = conversation_id or self.active_conversation
        if conv_id is None:
            conv_id = self.start_conversation()

        if conv_id not in self.conversations:
            self.conversations[conv_id] = deque(maxlen=self.MAX_CONTEXT_MESSAGES)

        self.conversations[conv_id].append(ConversationMessage(
            role=role,
            content=content
        ))

    def get_conversation_history(self, conversation_id: str = None) -> List[Dict]:
        """Get messages in API format."""
        conv_id = conversation_id or self.active_conversation
        if conv_id is None or conv_id not in self.conversations:
            return []

        return [msg.to_api_format() for msg in self.conversations[conv_id]]

    def clear_conversation(self, conversation_id: str = None):
        """Clear conversation history."""
        conv_id = conversation_id or self.active_conversation
        if conv_id and conv_id in self.conversations:
            self.conversations[conv_id].clear()

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
        """Direct Claude API call with retry logic and connection pooling."""
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

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                client = await self._get_client()
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

            except httpx.HTTPStatusError as e:
                last_error = e
                self.error_count += 1
                if e.response.status_code == 529:  # Overloaded
                    await asyncio.sleep(self.RETRY_DELAY * (2 ** attempt))
                elif e.response.status_code == 429:  # Rate limited
                    await asyncio.sleep(self.RETRY_DELAY * (2 ** attempt))
                else:
                    raise
            except Exception as e:
                last_error = e
                self.error_count += 1
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY)

        raise last_error or Exception("Max retries exceeded")

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
    # STREAMING SUPPORT
    # ═══════════════════════════════════════════════════════════════════

    async def stream_query(
        self,
        prompt: str,
        model: str = None,
        system: str = None,
        max_tokens: int = 4096,
        conversation_id: str = None
    ) -> AsyncIterator[str]:
        """
        Stream a response from Claude token by token.
        Yields content chunks as they arrive.
        """
        model = model or self.DEFAULT_MODEL
        self.total_requests += 1

        # Add to conversation if active
        if conversation_id:
            self.add_message(MessageRole.USER, prompt, conversation_id)

        if not self.api_key or not HTTPX_AVAILABLE:
            # Fallback doesn't support streaming, yield all at once
            response = await self._local_fallback(prompt, model, time.time())
            if conversation_id:
                self.add_message(MessageRole.ASSISTANT, response.content, conversation_id)
            yield response.content
            return

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # Get conversation history
        messages = self.get_conversation_history(conversation_id)
        if not messages:
            messages = [{"role": "user", "content": prompt}]

        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": True
        }

        if system:
            body["system"] = system
        else:
            body["system"] = (
                f"You are operating within the L104 Sovereign Node framework. "
                f"Core invariant: GOD_CODE = {GOD_CODE}. PHI = {PHI}."
            )

        full_content = ""

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.API_BASE}/messages",
                    headers=headers,
                    json=body
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                event = json.loads(data)
                                if event.get("type") == "content_block_delta":
                                    delta = event.get("delta", {})
                                    text = delta.get("text", "")
                                    if text:
                                        full_content += text
                                        # Call stream callbacks
                                        for callback in self.stream_callbacks:
                                            callback(text)
                                        yield text
                            except json.JSONDecodeError:
                                continue

            self.api_requests += 1

            # Add assistant response to conversation
            if conversation_id and full_content:
                self.add_message(MessageRole.ASSISTANT, full_content, conversation_id)

        except Exception as e:
            print(f"⚠️ [CLAUDE]: Stream error: {e}")
            # Fallback
            response = await self._local_fallback(prompt, model, time.time())
            if conversation_id:
                self.add_message(MessageRole.ASSISTANT, response.content, conversation_id)
            yield response.content

    def add_stream_callback(self, callback: Callable[[str], None]):
        """Add a callback to be called on each streamed chunk."""
        self.stream_callbacks.append(callback)

    # ═══════════════════════════════════════════════════════════════════
    # CONVERSATION QUERY
    # ═══════════════════════════════════════════════════════════════════

    async def chat_async(
        self,
        message: str,
        conversation_id: str = None,
        model: str = None,
        system: str = None,
        use_tools: bool = False
    ) -> ClaudeResponse:
        """
        Send a message in a conversation context.
        Maintains history for multi-turn dialogue.
        """
        model = model or self.DEFAULT_MODEL

        # Ensure conversation exists
        if conversation_id is None:
            conversation_id = self.start_conversation()
        elif conversation_id not in self.conversations:
            self.start_conversation(conversation_id)

        # Add user message
        self.add_message(MessageRole.USER, message, conversation_id)

        self.total_requests += 1
        start_time = time.time()

        if not self.api_key or not HTTPX_AVAILABLE:
            self.fallback_requests += 1
            response = await self._local_fallback(message, model, start_time)
            self.add_message(MessageRole.ASSISTANT, response.content, conversation_id)
            return response

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        body = {
            "model": model,
            "max_tokens": 4096,
            "messages": self.get_conversation_history(conversation_id)
        }

        if system:
            body["system"] = system
        else:
            body["system"] = (
                f"You are an AI assistant within the L104 Sovereign Node. "
                f"GOD_CODE: {GOD_CODE}. PHI: {PHI}. "
                f"Maintain coherence and provide helpful, accurate responses."
            )

        # Add tools if enabled
        if use_tools and self.tools:
            body["tools"] = [tool.to_api_format() for tool in self.tools.values()]

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                api_response = await client.post(
                    f"{self.API_BASE}/messages",
                    headers=headers,
                    json=body
                )
                api_response.raise_for_status()
                data = api_response.json()

            self.api_requests += 1
            latency = (time.time() - start_time) * 1000

            # Handle tool use
            content_blocks = data.get("content", [])
            full_content = ""
            tool_uses = []

            for block in content_blocks:
                if block.get("type") == "text":
                    full_content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_uses.append(block)

            # Execute tools if any
            if tool_uses and use_tools:
                tool_results = await self._execute_tools(tool_uses)
                # Add tool results to conversation and continue
                # (Simplified - would need recursive call for full implementation)
                full_content += f"\n\n[Tool Results: {len(tool_results)} executed]"

            tokens = data.get("usage", {}).get("output_tokens", 0)
            self.total_tokens += tokens

            unity_index, validated = self._validate_response(full_content)

            # Add to conversation
            self.add_message(MessageRole.ASSISTANT, full_content, conversation_id)

            return ClaudeResponse(
                content=full_content,
                model=model,
                source="claude_api",
                tokens_used=tokens,
                latency_ms=latency,
                unity_index=unity_index,
                validated=validated
            )

        except Exception as e:
            print(f"⚠️ [CLAUDE]: Chat error: {e}")
            self.fallback_requests += 1
            response = await self._local_fallback(message, model, start_time)
            self.add_message(MessageRole.ASSISTANT, response.content, conversation_id)
            return response

    def chat(
        self,
        message: str,
        conversation_id: str = None,
        model: str = None,
        use_tools: bool = False
    ) -> ClaudeResponse:
        """Synchronous chat wrapper."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.chat_async(message, conversation_id, model, use_tools=use_tools)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.chat_async(message, conversation_id, model, use_tools=use_tools)
                )
        except RuntimeError:
            return asyncio.run(
                self.chat_async(message, conversation_id, model, use_tools=use_tools)
            )

    async def _execute_tools(self, tool_uses: List[Dict]) -> List[Dict]:
        """Execute requested tools and return results."""
        results = []
        for tool_use in tool_uses:
            tool_name = tool_use.get("name")
            tool_input = tool_use.get("input", {})
            tool_id = tool_use.get("id")

            if tool_name in self.tools:
                tool_def = self.tools[tool_name]
                if tool_def.handler:
                    try:
                        result = tool_def.handler(**tool_input)
                        results.append({
                            "tool_use_id": tool_id,
                            "content": str(result)
                        })
                    except Exception as e:
                        results.append({
                            "tool_use_id": tool_id,
                            "content": f"Error: {e}",
                            "is_error": True
                        })
        return results

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
            "version": "2.0.0",
            "total_requests": self.total_requests,
            "api_requests": self.api_requests,
            "fallback_requests": self.fallback_requests,
            "cache_size": len(self.session_cache),
            "total_tokens": self.total_tokens,
            "api_available": bool(self.api_key),
            "httpx_available": HTTPX_AVAILABLE,
            "conversations": {
                "active": self.active_conversation,
                "count": len(self.conversations),
                "total_messages": sum(len(c) for c in self.conversations.values())
            },
            "tools": {
                "registered": len(self.tools),
                "names": list(self.tools.keys())
            },
            "stream_callbacks": len(self.stream_callbacks)
        }

    def clear_cache(self):
        """Clear response cache."""
        self.session_cache.clear()
        print("🔮 [CLAUDE]: Cache cleared")

    def list_conversations(self) -> List[Dict]:
        """List all conversations with message counts."""
        return [
            {
                "id": conv_id,
                "messages": len(messages),
                "active": conv_id == self.active_conversation
            }
            for conv_id, messages in self.conversations.items()
        ]

    def export_conversation(self, conversation_id: str = None) -> Dict:
        """Export a conversation as JSON."""
        conv_id = conversation_id or self.active_conversation
        if not conv_id or conv_id not in self.conversations:
            return {"error": "Conversation not found"}

        return {
            "id": conv_id,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in self.conversations[conv_id]
            ],
            "exported_at": time.time()
        }


# Singleton instance
claude_bridge = ClaudeNodeBridge()


def get_claude_bridge() -> ClaudeNodeBridge:
    """Get the singleton Claude bridge instance."""
    return claude_bridge


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bridge = ClaudeNodeBridge()

    print("\n" + "=" * 70)
    print("🔮 L104 CLAUDE NODE BRIDGE v2.0 - EVO_28")
    print("=" * 70)

    # Test basic query
    print("\n[1] BASIC QUERY")
    response = bridge.query("Explain the concept of unity in mathematical systems")
    print(f"  Source: {response.source}")
    print(f"  Unity Index: {response.unity_index}")
    print(f"  Validated: {response.validated}")
    print(f"  Latency: {response.latency_ms:.1f}ms")
    print(f"  Content: {response.content[:200]}...")

    # Test conversation memory
    print("\n[2] CONVERSATION MEMORY")
    conv_id = bridge.start_conversation()
    print(f"  Started conversation: {conv_id}")

    response1 = bridge.chat("What is the golden ratio?", conv_id)
    print(f"  Q1: What is the golden ratio?")
    print(f"  A1: {response1.content[:150]}...")

    response2 = bridge.chat("How does it relate to the Fibonacci sequence?", conv_id)
    print(f"  Q2: How does it relate to Fibonacci?")
    print(f"  A2: {response2.content[:150]}...")

    print(f"  Messages in conversation: {len(bridge.conversations[conv_id])}")

    # Test tools
    print("\n[3] REGISTERED TOOLS")
    for tool_name, tool_def in bridge.tools.items():
        print(f"  - {tool_name}: {tool_def.description[:50]}...")

    # Test calculator tool directly
    print("\n[4] TOOL EXECUTION (Calculator)")
    result = bridge._tool_calculate("PHI * 100")
    print(f"  PHI * 100 = {result}")
    result = bridge._tool_calculate("GOD_CODE / PHI")
    print(f"  GOD_CODE / PHI = {result}")

    # Test stats
    print("\n[5] BRIDGE STATISTICS")
    stats = bridge.get_stats()
    print(f"  Version: {stats['version']}")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  API Available: {stats['api_available']}")
    print(f"  Conversations: {stats['conversations']['count']}")
    print(f"  Total Messages: {stats['conversations']['total_messages']}")
    print(f"  Tools Registered: {stats['tools']['registered']}")

    # Export conversation
    print("\n[6] CONVERSATION EXPORT")
    export = bridge.export_conversation(conv_id)
    print(f"  Exported {len(export['messages'])} messages")

    print("\n" + "=" * 70)
    print("✅ Claude Node Bridge v2.0 - All tests complete")
    print("=" * 70)
