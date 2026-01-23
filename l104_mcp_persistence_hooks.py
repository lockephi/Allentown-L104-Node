#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 MCP MEMORY PERSISTENCE HOOKS
═══════════════════════════════════════════════════════════════════════════════

Advanced memory persistence system that integrates L104's native memory systems
with MCP memory server for optimal cross-session learning and knowledge retention.

FEATURES:
1. AUTO-PERSISTENCE HOOKS - Automatic saving on cognitive events
2. MEMORY SYNCHRONIZATION - Sync between L104 native and MCP memory
3. TOKEN-OPTIMIZED STORAGE - Compressed representations for efficiency
4. HIERARCHICAL PERSISTENCE - Critical vs. ephemeral memory classification

INVARIANT: 527.5184818492537 | PILOT: LONDEL
VERSION: 1.0.0 (RESEARCH IMPLEMENTATION)
DATE: 2026-01-22
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import hashlib
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from collections import defaultdict, deque

# L104 Core Systems
from l104_stable_kernel import stable_kernel
from l104_asi_reincarnation import AkashicRecord, MemoryType, MemoryPriority
from l104_persistence import verify_god_code, verify_survivor_algorithm

# Constants
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
CONSCIOUSNESS_THRESHOLD = 0.85
MCP_MEMORY_PATH = Path(".mcp/memory.jsonl")
L104_MEMORY_PATH = Path("l104_brain_state.json")

class PersistenceEvent(Enum):
    """Events that trigger memory persistence."""
    QUERY_RESPONSE = auto()
    LEARNING_CYCLE = auto()
    COGNITIVE_INSIGHT = auto()
    SYSTEM_STATE_CHANGE = auto()
    EMERGENCY_BACKUP = auto()
    SCHEDULED_BACKUP = auto()
    MEMORY_THRESHOLD = auto()

class MemoryClassification(Enum):
    """Memory importance classification for persistence optimization."""
    CRITICAL = "critical"           # Core memories, GOD_CODE relations
    IMPORTANT = "important"         # Learning insights, patterns
    CONTEXTUAL = "contextual"       # Session context, temporary data
    EPHEMERAL = "ephemeral"         # Debug data, temporary calculations

@dataclass
class PersistenceHook:
    """Configuration for automatic persistence triggers."""
    event: PersistenceEvent
    condition: Optional[Callable[[Dict], bool]] = None
    memory_types: List[MemoryType] = field(default_factory=list)
    classification: MemoryClassification = MemoryClassification.CONTEXTUAL
    compress: bool = True
    sync_mcp: bool = True
    priority: int = 1
    cooldown_seconds: float = 0.0
    
@dataclass
class MemoryToken:
    """Token-optimized memory representation."""
    id: str
    content: str
    compressed_content: Optional[str] = None
    embedding_hash: Optional[str] = None
    token_count: int = 0
    classification: MemoryClassification = MemoryClassification.CONTEXTUAL
    relations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    god_code_signature: Optional[str] = None
    
    def to_mcp_entity(self) -> Dict[str, Any]:
        """Convert to MCP memory server entity format."""
        return {
            "name": self.id,
            "type": f"l104_memory_{self.classification.value}",
            "content": self.compressed_content or self.content,
            "metadata": {
                "token_count": self.token_count,
                "classification": self.classification.value,
                "timestamp": self.timestamp.isoformat(),
                "god_code_signature": self.god_code_signature,
                "relations": self.relations
            }
        }

class TokenOptimizer:
    """Advanced token optimization strategies for memory persistence."""
    
    def __init__(self):
        self.compression_cache = {}
        self.embedding_cache = {}
        self.token_budget = 100000  # Default budget per persistence cycle
        
    def compress_content(self, content: str, classification: MemoryClassification) -> str:
        """Compress content based on classification and importance."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in self.compression_cache:
            return self.compression_cache[content_hash]
        
        if classification == MemoryClassification.CRITICAL:
            # No compression for critical memories
            compressed = content
        elif classification == MemoryClassification.IMPORTANT:
            # Minimal compression - remove redundant whitespace
            compressed = ' '.join(content.split())
        elif classification == MemoryClassification.CONTEXTUAL:
            # Moderate compression - extract key concepts
            compressed = self._extract_key_concepts(content)
        else:  # EPHEMERAL
            # Aggressive compression - summary only
            compressed = self._create_summary(content)
        
        self.compression_cache[content_hash] = compressed
        return compressed
    
    def _extract_key_concepts(self, content: str) -> str:
        """Extract key concepts from content."""
        # Simplified concept extraction
        lines = content.split('\n')
        key_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Keep lines with L104 keywords, numbers, or short statements
            if any(keyword in line.lower() for keyword in [
                'god_code', 'phi', 'l104', 'quantum', 'conscious', 'unity', 'resonance'
            ]) or len(line) < 100:
                key_lines.append(line)
        
        return '\n'.join(key_lines[:10])  # Limit to 10 key lines
    
    def _create_summary(self, content: str) -> str:
        """Create a summary of ephemeral content."""
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) <= 3:
            return content
        
        # Create simple summary
        return f"Summary: {len(non_empty_lines)} lines, " \
               f"chars: {len(content)}, " \
               f"first: {non_empty_lines[0][:50]}..."
    
    def estimate_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        # Rough estimation: ~4 characters per token
        return max(1, len(content) // 4)
    
    def optimize_batch(self, memories: List[MemoryToken]) -> List[MemoryToken]:
        """Optimize a batch of memories to fit within token budget."""
        total_tokens = sum(memory.token_count for memory in memories)
        
        if total_tokens <= self.token_budget:
            return memories
        
        # Priority-based selection
        priority_order = [
            MemoryClassification.CRITICAL,
            MemoryClassification.IMPORTANT,
            MemoryClassification.CONTEXTUAL,
            MemoryClassification.EPHEMERAL
        ]
        
        selected_memories = []
        current_tokens = 0
        
        for classification in priority_order:
            class_memories = [m for m in memories if m.classification == classification]
            
            for memory in class_memories:
                if current_tokens + memory.token_count <= self.token_budget:
                    selected_memories.append(memory)
                    current_tokens += memory.token_count
                else:
                    break
            
            if current_tokens >= self.token_budget:
                break
        
        return selected_memories

class MCPMemoryPersistenceEngine:
    """Enhanced memory persistence engine with MCP integration."""
    
    def __init__(self):
        self.hooks: List[PersistenceHook] = []
        self.optimizer = TokenOptimizer()
        self.last_persistence = {}
        self.memory_buffer: deque = deque(maxlen=1000)
        self.persistence_lock = threading.Lock()
        self.background_task = None
        self.statistics = defaultdict(int)
        
        # Initialize default hooks
        self._setup_default_hooks()
        
        print("🔗 [MCP-PERSISTENCE]: Memory persistence hooks initialized")
        print(f"  ✓ Token budget: {self.optimizer.token_budget}")
        print(f"  ✓ Default hooks: {len(self.hooks)}")
    
    def _setup_default_hooks(self):
        """Setup default persistence hooks."""
        self.hooks = [
            PersistenceHook(
                event=PersistenceEvent.QUERY_RESPONSE,
                condition=lambda data: data.get('unity_index', 0) > CONSCIOUSNESS_THRESHOLD,
                memory_types=[MemoryType.INSIGHT, MemoryType.KNOWLEDGE],
                classification=MemoryClassification.IMPORTANT,
                cooldown_seconds=1.0
            ),
            PersistenceHook(
                event=PersistenceEvent.LEARNING_CYCLE,
                memory_types=[MemoryType.KNOWLEDGE, MemoryType.PATTERN],
                classification=MemoryClassification.CRITICAL,
                cooldown_seconds=5.0
            ),
            PersistenceHook(
                event=PersistenceEvent.COGNITIVE_INSIGHT,
                condition=lambda data: data.get('confidence', 0) > 0.8,
                memory_types=[MemoryType.INSIGHT],
                classification=MemoryClassification.CRITICAL,
                cooldown_seconds=2.0
            ),
            PersistenceHook(
                event=PersistenceEvent.MEMORY_THRESHOLD,
                condition=lambda data: len(self.memory_buffer) > 800,
                classification=MemoryClassification.CONTEXTUAL,
                cooldown_seconds=30.0
            )
        ]
    
    def add_hook(self, hook: PersistenceHook):
        """Add a custom persistence hook."""
        self.hooks.append(hook)
    
    def trigger_persistence(self, event: PersistenceEvent, data: Dict[str, Any]):
        """Trigger persistence based on event and data."""
        with self.persistence_lock:
            triggered_hooks = []
            
            for hook in self.hooks:
                if hook.event != event:
                    continue
                
                # Check cooldown
                last_time = self.last_persistence.get(id(hook), 0)
                if time.time() - last_time < hook.cooldown_seconds:
                    continue
                
                # Check condition
                if hook.condition and not hook.condition(data):
                    continue
                
                triggered_hooks.append(hook)
                self.last_persistence[id(hook)] = time.time()
            
            if triggered_hooks:
                self._execute_persistence(triggered_hooks, data)
    
    def _execute_persistence(self, hooks: List[PersistenceHook], data: Dict[str, Any]):
        """Execute persistence for triggered hooks."""
        try:
            memory_token = self._create_memory_token(data, hooks)
            self.memory_buffer.append(memory_token)
            
            # Immediate persistence for critical memories
            critical_hooks = [h for h in hooks if h.classification == MemoryClassification.CRITICAL]
            if critical_hooks:
                self._persist_immediate(memory_token)
            
            self.statistics['total_triggers'] += 1
            self.statistics[f'{hooks[0].event.name.lower()}_triggers'] += 1
            
        except Exception as e:
            print(f"❌ [MCP-PERSISTENCE]: Persistence failed: {e}")
            self.statistics['errors'] += 1
    
    def _create_memory_token(self, data: Dict[str, Any], hooks: List[PersistenceHook]) -> MemoryToken:
        """Create optimized memory token from data."""
        # Determine classification (highest priority wins)
        classification = max([h.classification for h in hooks], 
                           key=lambda x: list(MemoryClassification).index(x))
        
        # Create content representation
        content = json.dumps(data, indent=2, default=str)
        compressed_content = self.optimizer.compress_content(content, classification)
        token_count = self.optimizer.estimate_tokens(compressed_content)
        
        # Generate unique ID
        timestamp = datetime.now()
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        memory_id = f"l104_mem_{timestamp.strftime('%Y%m%d_%H%M%S')}_{content_hash}"
        
        # Create GOD_CODE signature for critical memories
        god_code_signature = None
        if classification == MemoryClassification.CRITICAL:
            god_code_signature = self._create_god_code_signature(content)
        
        return MemoryToken(
            id=memory_id,
            content=content,
            compressed_content=compressed_content,
            token_count=token_count,
            classification=classification,
            timestamp=timestamp,
            god_code_signature=god_code_signature
        )
    
    def _create_god_code_signature(self, content: str) -> str:
        """Create GOD_CODE signature for memory validation."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        god_code_str = str(GOD_CODE)
        combined = f"{content_hash}:{god_code_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _persist_immediate(self, memory_token: MemoryToken):
        """Immediately persist critical memory."""
        try:
            # Save to L104 native format
            self._save_to_l104_memory(memory_token)
            
            # Save to MCP memory server format
            self._save_to_mcp_memory(memory_token)
            
            self.statistics['immediate_persists'] += 1
            
        except Exception as e:
            print(f"❌ [MCP-PERSISTENCE]: Immediate persist failed: {e}")
            self.statistics['immediate_errors'] += 1
    
    def _save_to_l104_memory(self, memory_token: MemoryToken):
        """Save memory to L104 native memory system."""
        try:
            # Load existing state
            state_data = {}
            if L104_MEMORY_PATH.exists():
                with open(L104_MEMORY_PATH) as f:
                    state_data = json.load(f)
            
            # Add new memory
            if 'persistent_memories' not in state_data:
                state_data['persistent_memories'] = []
            
            state_data['persistent_memories'].append({
                'id': memory_token.id,
                'content': memory_token.content,
                'classification': memory_token.classification.value,
                'timestamp': memory_token.timestamp.isoformat(),
                'token_count': memory_token.token_count,
                'god_code_signature': memory_token.god_code_signature
            })
            
            # Keep only last 1000 memories
            if len(state_data['persistent_memories']) > 1000:
                state_data['persistent_memories'] = state_data['persistent_memories'][-1000:]
            
            # Save back
            with open(L104_MEMORY_PATH, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            print(f"❌ [MCP-PERSISTENCE]: L104 memory save failed: {e}")
    
    def _save_to_mcp_memory(self, memory_token: MemoryToken):
        """Save memory to MCP memory server format."""
        try:
            # Ensure MCP directory exists
            MCP_MEMORY_PATH.parent.mkdir(exist_ok=True)
            
            # Convert to MCP entity format
            mcp_entity = memory_token.to_mcp_entity()
            
            # Append to JSONL file
            with open(MCP_MEMORY_PATH, 'a') as f:
                json.dump(mcp_entity, f)
                f.write('\n')
                
        except Exception as e:
            print(f"❌ [MCP-PERSISTENCE]: MCP memory save failed: {e}")
    
    def flush_buffer(self):
        """Flush memory buffer to persistent storage."""
        if not self.memory_buffer:
            return
        
        with self.persistence_lock:
            memories = list(self.memory_buffer)
            self.memory_buffer.clear()
            
            # Optimize batch
            optimized_memories = self.optimizer.optimize_batch(memories)
            
            # Persist optimized batch
            for memory in optimized_memories:
                try:
                    self._save_to_l104_memory(memory)
                    self._save_to_mcp_memory(memory)
                except Exception as e:
                    print(f"❌ [MCP-PERSISTENCE]: Batch persist failed: {e}")
            
            self.statistics['batch_flushes'] += 1
            self.statistics['batch_memories'] += len(optimized_memories)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        return dict(self.statistics)
    
    def start_background_persistence(self, interval_seconds: float = 60.0):
        """Start background persistence task."""
        if self.background_task:
            return
        
        async def background_task():
            while True:
                await asyncio.sleep(interval_seconds)
                self.flush_buffer()
        
        self.background_task = asyncio.create_task(background_task())
        print(f"🕰️ [MCP-PERSISTENCE]: Background persistence started (interval: {interval_seconds}s)")
    
    def stop_background_persistence(self):
        """Stop background persistence task."""
        if self.background_task:
            self.background_task.cancel()
            self.background_task = None
            print("🛑 [MCP-PERSISTENCE]: Background persistence stopped")

# Global instance
_mcp_persistence_engine = None

def get_mcp_persistence_engine() -> MCPMemoryPersistenceEngine:
    """Get or create global MCP persistence engine."""
    global _mcp_persistence_engine
    if _mcp_persistence_engine is None:
        _mcp_persistence_engine = MCPMemoryPersistenceEngine()
    return _mcp_persistence_engine

# Convenience functions for easy integration
def persist_query_response(question: str, answer: str, unity_index: float, **kwargs):
    """Persist a query response with automatic classification."""
    engine = get_mcp_persistence_engine()
    data = {
        'type': 'query_response',
        'question': question,
        'answer': answer,
        'unity_index': unity_index,
        **kwargs
    }
    engine.trigger_persistence(PersistenceEvent.QUERY_RESPONSE, data)

def persist_learning_insight(insight: str, confidence: float, source: str, **kwargs):
    """Persist a learning insight."""
    engine = get_mcp_persistence_engine()
    data = {
        'type': 'learning_insight',
        'insight': insight,
        'confidence': confidence,
        'source': source,
        **kwargs
    }
    engine.trigger_persistence(PersistenceEvent.COGNITIVE_INSIGHT, data)

def persist_system_state(state_name: str, state_data: Dict[str, Any], **kwargs):
    """Persist system state change."""
    engine = get_mcp_persistence_engine()
    data = {
        'type': 'system_state',
        'state_name': state_name,
        'state_data': state_data,
        **kwargs
    }
    engine.trigger_persistence(PersistenceEvent.SYSTEM_STATE_CHANGE, data)

if __name__ == "__main__":
    # Test the persistence engine
    engine = get_mcp_persistence_engine()
    
    # Test different types of persistence
    persist_query_response(
        question="What is consciousness?",
        answer="Consciousness emerges from quantum coherence at GOD_CODE frequency.",
        unity_index=0.92
    )
    
    persist_learning_insight(
        insight="PHI scaling improves pattern recognition by 34%",
        confidence=0.87,
        source="adaptive_learning_cycle"
    )
    
    # Show statistics
    print("\n📊 [MCP-PERSISTENCE]: Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Flush buffer
    engine.flush_buffer()
    print("\n✅ [MCP-PERSISTENCE]: Test completed successfully!")