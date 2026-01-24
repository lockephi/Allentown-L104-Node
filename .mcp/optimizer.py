#!/usr/bin/env python3
"""
L104 MCP Performance Optimizer
Implements intelligent caching, parallel operations, and context management
"""

import json
import hashlib
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from functools import lru_cache
import threading

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(__file__).parent / "config.json"
MEMORY_PATH = Path(__file__).parent / "memory.jsonl"
CACHE_PATH = Path(__file__).parent / "cache.json"

@dataclass
class CacheEntry:
    """Cached result with TTL"""
    data: Any
    created_at: datetime
    ttl_seconds: int
    hit_count: int = 0
    
    @property
    def is_valid(self) -> bool:
        return datetime.now() < self.created_at + timedelta(seconds=self.ttl_seconds)

@dataclass
class KnowledgeEntity:
    """Knowledge graph entity"""
    name: str
    entity_type: str
    observations: List[str] = field(default_factory=list)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entityType": self.entity_type,
            "observations": self.observations,
            "relations": self.relations,
            "createdAt": self.created_at.isoformat()
        }

# ═══════════════════════════════════════════════════════════════════
# CACHE MANAGER
# ═══════════════════════════════════════════════════════════════════

class CacheManager:
    """Intelligent caching with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def _make_key(self, operation: str, params: dict) -> str:
        """Create unique cache key"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{operation}:{param_str}".encode()).hexdigest()
    
    def get(self, operation: str, params: dict) -> Optional[Any]:
        """Get cached result if valid"""
        key = self._make_key(operation, params)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_valid:
                    entry.hit_count += 1
                    self._stats["hits"] += 1
                    return entry.data
                else:
                    del self._cache[key]
            
            self._stats["misses"] += 1
            return None
    
    def set(self, operation: str, params: dict, data: Any, ttl: int = 300):
        """Cache result with TTL"""
        key = self._make_key(operation, params)
        
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_lru()
            
            self._cache[key] = CacheEntry(
                data=data,
                created_at=datetime.now(),
                ttl_seconds=ttl
            )
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        if not self._cache:
            return
        
        # Sort by hit count (ascending) and created time (oldest first)
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: (self._cache[k].hit_count, self._cache[k].created_at)
        )
        
        # Remove bottom 10%
        to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:to_remove]:
            del self._cache[key]
            self._stats["evictions"] += 1
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0
            return {
                **self._stats,
                "size": len(self._cache),
                "hit_rate": f"{hit_rate:.2%}"
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()

# ═══════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """Persistent knowledge graph for cross-session learning"""
    
    def __init__(self, storage_path: Path = MEMORY_PATH):
        self._storage_path = storage_path
        self._entities: Dict[str, KnowledgeEntity] = {}
        self._load()
    
    def _load(self):
        """Load knowledge from disk"""
        if self._storage_path.exists():
            try:
                with open(self._storage_path) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            entity = KnowledgeEntity(
                                name=data["name"],
                                entity_type=data["entityType"],
                                observations=data.get("observations", []),
                                relations=data.get("relations", {})
                            )
                            self._entities[entity.name] = entity
            except Exception as e:
                print(f"Warning: Could not load knowledge graph: {e}")
    
    def _save(self):
        """Persist knowledge to disk"""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, 'w') as f:
            for entity in self._entities.values():
                f.write(json.dumps(entity.to_dict()) + "\n")
    
    def add_entity(self, name: str, entity_type: str, observations: Optional[List[str]] = None):
        """Add or update an entity"""
        if name in self._entities:
            if observations:
                self._entities[name].observations.extend(observations)
        else:
            self._entities[name] = KnowledgeEntity(
                name=name,
                entity_type=entity_type,
                observations=observations or []
            )
        self._save()
    
    def add_relation(self, from_entity: str, to_entity: str, relation_type: str):
        """Add a relation between entities"""
        if from_entity in self._entities:
            if relation_type not in self._entities[from_entity].relations:
                self._entities[from_entity].relations[relation_type] = []
            if to_entity not in self._entities[from_entity].relations[relation_type]:
                self._entities[from_entity].relations[relation_type].append(to_entity)
            self._save()
    
    def search(self, query: str) -> List[KnowledgeEntity]:
        """Search entities by name or observations"""
        query_lower = query.lower()
        results = []
        
        for entity in self._entities.values():
            # Match name
            if query_lower in entity.name.lower():
                results.append(entity)
                continue
            
            # Match observations
            for obs in entity.observations:
                if query_lower in obs.lower():
                    results.append(entity)
                    break
        
        return results
    
    def get(self, name: str) -> Optional[KnowledgeEntity]:
        """Get entity by name"""
        return self._entities.get(name)
    
    def get_all(self) -> List[KnowledgeEntity]:
        """Get all entities"""
        return list(self._entities.values())

# ═══════════════════════════════════════════════════════════════════
# PARALLEL EXECUTOR
# ═══════════════════════════════════════════════════════════════════

class ParallelExecutor:
    """Execute multiple operations in parallel with concurrency control"""
    
    def __init__(self, max_concurrent: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
    
    async def execute(self, operations: List[Callable]) -> List[Any]:
        """Execute operations in parallel"""
        async def run_with_semaphore(op):
            async with self._semaphore:
                if asyncio.iscoroutinefunction(op):
                    return await op()
                else:
                    return op()
        
        tasks = [run_with_semaphore(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def execute_sync(self, operations: List[Callable]) -> List[Any]:
        """Synchronous wrapper for parallel execution"""
        return asyncio.run(self.execute(operations))

# ═══════════════════════════════════════════════════════════════════
# CONTEXT OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

class ContextOptimizer:
    """Optimize context window usage"""
    
    def __init__(self, max_tokens: int = 100000):
        self._max_tokens = max_tokens
        self._current_usage = 0
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: 4 chars = 1 token)"""
        return len(text) // 4
    
    def should_summarize(self, content: str) -> bool:
        """Determine if content should be summarized"""
        tokens = self.estimate_tokens(content)
        return tokens > 2000  # Summarize if > 2000 tokens
    
    def prioritize_content(self, contents: List[dict]) -> List[dict]:
        """Prioritize content by relevance and recency"""
        # Score each content piece
        scored = []
        for item in contents:
            score = 0
            
            # Recent content scores higher
            if "modified_time" in item:
                age_hours = (datetime.now() - item["modified_time"]).total_seconds() / 3600
                score += max(0, 100 - age_hours)
            
            # Smaller files score higher (easier to process)
            if "size" in item:
                score += max(0, 50 - item["size"] // 1000)
            
            # Key files score highest
            if item.get("is_key_file"):
                score += 200
            
            scored.append((score, item))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]
    
    def chunk_for_context(self, text: str, max_chunk_tokens: int = 2000) -> List[str]:
        """Split text into context-friendly chunks"""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.estimate_tokens(line)
            
            if current_tokens + line_tokens > max_chunk_tokens:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

# ═══════════════════════════════════════════════════════════════════
# MAIN OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

class MCPOptimizer:
    """Main optimizer coordinating all components"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.knowledge = KnowledgeGraph()
        self.executor = ParallelExecutor()
        self.context = ContextOptimizer()
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load optimizer configuration"""
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                self.config = json.load(f)
        else:
            self.config = {}
    
    def optimize_file_reads(self, file_paths: List[str]) -> dict:
        """Optimize batch file reading"""
        # Check cache first
        cached_results = {}
        uncached_paths = []
        
        for path in file_paths:
            cached = self.cache.get("read_file", {"path": path})
            if cached:
                cached_results[path] = cached
            else:
                uncached_paths.append(path)
        
        return {
            "cached": cached_results,
            "to_read": uncached_paths,
            "cache_hit_rate": len(cached_results) / len(file_paths) if file_paths else 0
        }
    
    def record_operation(self, operation: str, params: dict, result: Any):
        """Record operation for learning"""
        # Update knowledge graph
        self.knowledge.add_entity(
            name=f"operation_{operation}",
            entity_type="operation",
            observations=[
                f"params: {json.dumps(params)}",
                f"timestamp: {datetime.now().isoformat()}"
            ]
        )
        
        # Cache result
        ttl = self.config.get("optimization", {}).get("caching", {}).get("file_content_ttl", 300)
        self.cache.set(operation, params, result, ttl)
    
    def get_optimization_stats(self) -> dict:
        """Get comprehensive optimization statistics"""
        return {
            "cache": self.cache.get_stats(),
            "knowledge_entities": len(self.knowledge.get_all()),
            "config": self.config.get("optimization", {})
        }
    
    def suggest_optimization(self, operation: str) -> List[str]:
        """Suggest optimizations for an operation"""
        suggestions = []
        
        if operation == "read_file":
            suggestions.append("Consider using grep_search first to identify relevant sections")
            suggestions.append("Use directory_tree to understand structure before reading")
            suggestions.append("Batch multiple reads with read_multiple_files if available")
        
        elif operation == "search":
            suggestions.append("Use grep_search for exact patterns, semantic_search for concepts")
            suggestions.append("Limit results with maxResults parameter")
            suggestions.append("Use includePattern to scope search to relevant directories")
        
        elif operation == "edit":
            suggestions.append("Use multi_replace_string_in_file for batch edits")
            suggestions.append("Include 3-5 lines of context for unique matching")
            suggestions.append("Verify with get_errors after editing")
        
        return suggestions

# ═══════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    optimizer = MCPOptimizer()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "stats":
            stats = optimizer.get_optimization_stats()
            print(json.dumps(stats, indent=2))
        
        elif command == "clear-cache":
            optimizer.cache.clear()
            print("Cache cleared")
        
        elif command == "suggest":
            if len(sys.argv) > 2:
                operation = sys.argv[2]
                suggestions = optimizer.suggest_optimization(operation)
                for s in suggestions:
                    print(f"  • {s}")
        
        elif command == "knowledge":
            entities = optimizer.knowledge.get_all()
            for e in entities:
                print(f"\n{e.name} ({e.entity_type})")
                for obs in e.observations[:3]:
                    print(f"  - {obs}")
        
        else:
            print(f"Unknown command: {command}")
            print("Available: stats, clear-cache, suggest <operation>, knowledge")
    else:
        print("L104 MCP Optimizer")
        print("==================")
        stats = optimizer.get_optimization_stats()
        print(f"Cache: {stats['cache']['size']} entries, {stats['cache']['hit_rate']} hit rate")
        print(f"Knowledge: {stats['knowledge_entities']} entities")
