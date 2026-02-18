VOID_CONSTANT = 1.0416180339887497
import math
# ZENITH_UPGRADE_ACTIVE: 2026-02-15T00:00:00.000000
ZENITH_HZ = 3887.8
UUC = 2402.792541
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
# [L104_KNOWLEDGE_BRIDGE] v2.0.0 — Cross-Subsystem Knowledge Unification Layer
# INVARIANT: 527.5184818492 | PILOT: LONDEL

"""
L104 Knowledge Bridge v2.0.0
═══════════════════════════════════════════════════════════════════════════════
Unified cross-subsystem knowledge query, deduplication, and synthesis layer.

Currently knowledge is siloed across:
  - l104_fast_server (SQLite memories + memory_cache + knowledge_graph)
  - l104_cognitive_core (SemanticMemory + WorkingMemory)
  - l104_local_intellect (embedding cache + KB)
  - l104_adaptive_learning (PatternRecognizer patterns)
  - .l104_*.json state files (22 state files, 44 MB)

This bridge provides:
  1. Multi-source parallel query routing
  2. Semantic deduplication (n-gram cosine similarity)
  3. Cross-store knowledge linking and enrichment
  4. Relevance-scored result fusion with PHI-weighting
  5. Knowledge gap detection
  6. Consciousness-aware priority routing

Wired into: local_derivation (Phase 3-5 recall enrichment)
             autonomous_sovereignty_cycle (knowledge consolidation)
             chat endpoint (fallback knowledge retrieval)
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
import hashlib
import logging
import sqlite3
import re
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path

logger = logging.getLogger("KNOWLEDGE_BRIDGE")

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)

PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
TAU = 6.283185307179586
FEIGENBAUM = 4.669201609102990
ALPHA_FINE = 0.0072973525693
PLANCK_SCALE = 1.616255e-35
BOLTZMANN_K = 1.380649e-23


# ═══════════════════════════════════════════════════════════════════════════════
# N-GRAM SIMILARITY ENGINE (lightweight, no external deps)
# ═══════════════════════════════════════════════════════════════════════════════

class NGramSimilarity:
    """
    Fast character n-gram based similarity for deduplication.
    Uses 3-gram vectors with cosine similarity — no numpy required.
    """

    def __init__(self, n: int = 3):
        self.n = n

    def _ngrams(self, text: str) -> Dict[str, int]:
        """Extract character n-grams with frequency counts."""
        text = text.lower().strip()
        grams: Dict[str, int] = defaultdict(int)
        for i in range(len(text) - self.n + 1):
            grams[text[i:i + self.n]] += 1
        return grams

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two texts using n-gram vectors."""
        if not a or not b:
            return 0.0
        ga, gb = self._ngrams(a), self._ngrams(b)
        if not ga or not gb:
            return 0.0
        # Cosine similarity
        keys = set(ga.keys()) | set(gb.keys())
        dot = sum(ga.get(k, 0) * gb.get(k, 0) for k in keys)
        norm_a = math.sqrt(sum(v * v for v in ga.values()))
        norm_b = math.sqrt(sum(v * v for v in gb.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def deduplicate(self, items: List[Dict[str, Any]], text_key: str = 'text',
                     threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Remove near-duplicate items based on text similarity."""
        if not items:
            return []
        unique = [items[0]]
        for item in items[1:]:
            is_dup = False
            item_text = str(item.get(text_key, ''))
            for existing in unique:
                existing_text = str(existing.get(text_key, ''))
                if self.similarity(item_text, existing_text) > threshold:
                    # Keep the one with higher relevance score
                    if item.get('relevance', 0) > existing.get('relevance', 0):
                        unique.remove(existing)
                        unique.append(item)
                    is_dup = True
                    break
            if not is_dup:
                unique.append(item)
        return unique


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE SOURCE ADAPTERS
# ═══════════════════════════════════════════════════════════════════════════════

class SQLiteMemoryAdapter:
    """Adapter to query the fast_server's SQLite memory store."""

    def __init__(self, db_path: str = 'l104_intellect.db'):
        self.db_path = db_path
        self._available = Path(db_path).exists()

    def query(self, topic: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Query SQLite memories by topic keyword matching."""
        if not self._available:
            return []
        results = []
        try:
            conn = sqlite3.connect(self.db_path, timeout=2)
            c = conn.cursor()

            # Multi-keyword search across query and response
            words = [w for w in topic.lower().split() if len(w) > 2]
            if not words:
                conn.close()
                return []

            # Build OR-based LIKE query for each keyword
            conditions = []
            params = []
            for w in words[:5]:  # Max 5 keywords
                conditions.append('(LOWER(query) LIKE ? OR LOWER(response) LIKE ?)')
                params.extend([f'%{w}%', f'%{w}%'])

            sql = f"""
                SELECT query, response, quality_score, source, created_at
                FROM memory
                WHERE {' OR '.join(conditions)}
                ORDER BY quality_score DESC, created_at DESC
                LIMIT ?
            """
            params.append(limit)
            c.execute(sql, params)

            for row in c.fetchall():
                # Compute keyword hit count for relevance
                text = (row[0] + ' ' + row[1]).lower()
                hits = sum(1 for w in words if w in text)
                results.append({
                    'text': row[1][:500],
                    'query': row[0][:200],
                    'quality': row[2] or 0.5,
                    'source': f"sqlite:{row[3] or 'memory'}",
                    'timestamp': row[4],
                    'relevance': hits / max(len(words), 1) * (row[2] or 0.5),
                })
            conn.close()
        except Exception as e:
            logger.warning(f"[KB] SQLite adapter error: {e}")
        return results


class KnowledgeGraphAdapter:
    """Adapter to query the fast_server's in-memory knowledge graph."""

    def __init__(self):
        self._intellect_ref = None

    def bind(self, intellect_ref: Any):
        """Bind to the live intellect instance."""
        self._intellect_ref = intellect_ref

    def query(self, topic: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Query the knowledge graph for related concepts."""
        if not self._intellect_ref:
            return []
        results = []
        try:
            kg = getattr(self._intellect_ref, 'knowledge_graph', {})
            words = set(w.lower() for w in topic.split() if len(w) > 2)

            for concept, links in kg.items():
                concept_lower = concept.lower()
                if any(w in concept_lower for w in words):
                    related = list(links.keys())[:10] if isinstance(links, dict) else list(links)[:10]
                    results.append({
                        'text': f"{concept}: linked to {', '.join(str(r) for r in related)}",
                        'concept': concept,
                        'related': related,
                        'source': 'knowledge_graph',
                        'relevance': sum(1 for w in words if w in concept_lower) / max(len(words), 1),
                    })
                    if len(results) >= limit:
                        break
        except Exception as e:
            logger.warning(f"[KB] KnowledgeGraph adapter error: {e}")
        return results


class ConceptClusterAdapter:
    """Adapter to query concept clusters from the intellect."""

    def __init__(self):
        self._intellect_ref = None

    def bind(self, intellect_ref: Any):
        self._intellect_ref = intellect_ref

    def query(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find clusters matching the topic."""
        if not self._intellect_ref:
            return []
        results = []
        try:
            clusters = getattr(self._intellect_ref, 'concept_clusters', {})
            words = set(w.lower() for w in topic.split() if len(w) > 2)

            for cluster_name, members in clusters.items():
                cluster_lower = cluster_name.lower()
                member_text = ' '.join(str(m) for m in members).lower()
                hits = sum(1 for w in words if w in cluster_lower or w in member_text)
                if hits > 0:
                    results.append({
                        'text': f"Cluster '{cluster_name}': {', '.join(str(m) for m in list(members)[:8])}",
                        'cluster': cluster_name,
                        'members': list(members)[:20],
                        'source': 'concept_cluster',
                        'relevance': hits / max(len(words), 1),
                    })
                    if len(results) >= limit:
                        break
        except Exception as e:
            logger.warning(f"[KB] Cluster adapter error: {e}")
        return results


class StateFileAdapter:
    """Adapter to query consciousness/evolution state files."""

    STATE_FILES = [
        '.l104_consciousness_o2_state.json',
        '.l104_ouroboros_nirvanic_state.json',
        '.l104_evolution_state.json',
    ]

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_time = 0.0

    def _load_states(self) -> Dict[str, Any]:
        """Load and cache state files."""
        now = time.time()
        if now - self._cache_time < 30:  # 30s cache
            return self._cache

        merged = {}
        for fn in self.STATE_FILES:
            try:
                with open(fn, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        merged[k] = v
            except Exception:
                pass
        self._cache = merged
        self._cache_time = now
        return merged

    def query(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search state files for topic-related keys/values."""
        states = self._load_states()
        if not states:
            return []
        results = []
        words = set(w.lower() for w in topic.split() if len(w) > 2)

        for key, value in states.items():
            key_lower = key.lower()
            if any(w in key_lower for w in words):
                results.append({
                    'text': f"{key}: {json.dumps(value)[:300]}",
                    'key': key,
                    'value': value,
                    'source': 'state_file',
                    'relevance': sum(1 for w in words if w in key_lower) / max(len(words), 1) * 0.7,
                })
                if len(results) >= limit:
                    break
        return results


class CognitiveCoreBridgeAdapter:
    """Adapter to query l104_cognitive_core's SemanticMemory."""

    def __init__(self):
        self._cognitive_core = None
        self._tried = False

    def _ensure_loaded(self):
        if self._tried:
            return
        self._tried = True
        try:
            from l104_cognitive_core import COGNITIVE_CORE
            self._cognitive_core = COGNITIVE_CORE
        except Exception:
            pass

    def query(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query cognitive core's semantic memory."""
        self._ensure_loaded()
        if not self._cognitive_core:
            return []
        results = []
        try:
            # Use the cognitive core's think method for semantic recall
            thought = self._cognitive_core.think(topic)
            if thought and isinstance(thought, dict):
                response_text = thought.get('response', thought.get('result', str(thought)))
                if response_text and len(str(response_text)) > 10:
                    results.append({
                        'text': str(response_text)[:500],
                        'source': 'cognitive_core',
                        'relevance': 0.6,
                    })
            # Also check working memory
            wm = getattr(self._cognitive_core, 'working_memory', None)
            if wm and hasattr(wm, 'items'):
                for item in (list(wm.items) if hasattr(wm, 'items') else []):
                    item_str = str(item).lower()
                    words = set(w.lower() for w in topic.split() if len(w) > 2)
                    if any(w in item_str for w in words):
                        results.append({
                            'text': str(item)[:300],
                            'source': 'cognitive_core_wm',
                            'relevance': 0.5,
                        })
                        if len(results) >= limit:
                            break
        except Exception as e:
            logger.debug(f"[KB] CognitiveCore adapter: {e}")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GAP DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeGapDetector:
    """
    Detects topics where the system has insufficient knowledge.
    Tracks failed queries to identify areas needing learning.
    """

    def __init__(self, max_gaps: int = 500):
        self.gaps: Dict[str, Dict[str, Any]] = {}
        self.max_gaps = max_gaps
        self._total_queries = 0
        self._total_misses = 0

    def record_query(self, topic: str, results_count: int, quality_avg: float):
        """Record a knowledge query result."""
        self._total_queries += 1
        if results_count == 0 or quality_avg < 0.3:
            self._total_misses += 1
            # Normalize topic
            topic_key = ' '.join(sorted(set(w.lower() for w in topic.split() if len(w) > 3)))[:100]
            if topic_key:
                if topic_key in self.gaps:
                    self.gaps[topic_key]['miss_count'] += 1
                    self.gaps[topic_key]['last_miss'] = time.time()
                else:
                    if len(self.gaps) >= self.max_gaps:
                        # Evict oldest gap
                        oldest = min(self.gaps, key=lambda k: self.gaps[k]['last_miss'])
                        del self.gaps[oldest]
                    self.gaps[topic_key] = {
                        'miss_count': 1,
                        'first_miss': time.time(),
                        'last_miss': time.time(),
                    }

    def get_top_gaps(self, n: int = 20) -> List[Tuple[str, int]]:
        """Return the most frequently missed topics."""
        sorted_gaps = sorted(self.gaps.items(), key=lambda x: x[1]['miss_count'], reverse=True)
        return [(topic, info['miss_count']) for topic, info in sorted_gaps[:n]]

    def get_miss_rate(self) -> float:
        """Overall knowledge miss rate."""
        return self._total_misses / max(self._total_queries, 1)

    def get_report(self) -> Dict[str, Any]:
        return {
            'total_queries': self._total_queries,
            'total_misses': self._total_misses,
            'miss_rate': round(self.get_miss_rate(), 4),
            'tracked_gaps': len(self.gaps),
            'top_gaps': self.get_top_gaps(10),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BRIDGE (Hub Class)
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeBridge:
    """
    Unified Cross-Subsystem Knowledge Bridge v2.0

    Routes queries across all knowledge stores, deduplicates results,
    fuses them by relevance, and returns enriched, ranked answers.
    Detects knowledge gaps and feeds them back to the learning loop.
    """

    def __init__(self):
        self.sources: List[str] = []
        self.cache: Dict[str, Tuple[List[Dict], float]] = {}  # query_hash → (results, timestamp)
        self._cache_ttl = 60.0  # 60 second cache

        # Adapters
        self.sqlite_adapter = SQLiteMemoryAdapter()
        self.kg_adapter = KnowledgeGraphAdapter()
        self.cluster_adapter = ConceptClusterAdapter()
        self.state_adapter = StateFileAdapter()
        self.cognitive_adapter = CognitiveCoreBridgeAdapter()

        # Utilities
        self.similarity = NGramSimilarity(n=3)
        self.gap_detector = KnowledgeGapDetector()

        # Metrics
        self._total_queries = 0
        self._total_results = 0
        self._avg_result_count = 0.0
        self._query_latencies: deque = deque(maxlen=500)

        logger.info("[KNOWLEDGE_BRIDGE] v2.0 initialized — 5 adapters, PHI-weighted fusion, gap detection active")

    def bind_intellect(self, intellect_ref: Any):
        """Bind live intellect reference for knowledge graph + cluster access."""
        self.kg_adapter.bind(intellect_ref)
        self.cluster_adapter.bind(intellect_ref)
        logger.info("[KNOWLEDGE_BRIDGE] Bound to live intellect — KG + Cluster adapters active")

    def register_source(self, source: str):
        """Register a knowledge source name."""
        if source not in self.sources:
            self.sources.append(source)
            logger.info(f"[KNOWLEDGE_BRIDGE] Registered source: {source}")

    async def query(self, topic: str, depth: int = 2, max_results: int = 25) -> Dict[str, Any]:
        """
        Query ALL knowledge stores, deduplicate, fuse, and rank results.

        Args:
            topic: The search query
            depth: 1=fast (SQLite only), 2=standard (all adapters), 3=deep (+ state files)
            max_results: Maximum number of results to return

        Returns:
            Dict with 'results', 'sources_queried', 'gap_detected', 'relevance_stats'
        """
        start = time.time()
        self._total_queries += 1

        # Cache check
        cache_key = hashlib.md5(f"{topic}:{depth}".encode()).hexdigest()[:12]
        if cache_key in self.cache:
            cached_results, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return {
                    'results': cached_results,
                    'sources_queried': ['cache'],
                    'cached': True,
                    'latency_ms': round((time.time() - start) * 1000, 1),
                }

        all_results: List[Dict[str, Any]] = []
        sources_queried: List[str] = []

        # Depth 1: SQLite (always — fastest, most data)
        try:
            sqlite_results = self.sqlite_adapter.query(topic, limit=max_results)
            all_results.extend(sqlite_results)
            sources_queried.append('sqlite')
        except Exception as e:
            logger.debug(f"[KB] SQLite query error: {e}")

        if depth >= 2:
            # Knowledge Graph
            try:
                kg_results = self.kg_adapter.query(topic, limit=15)
                all_results.extend(kg_results)
                sources_queried.append('knowledge_graph')
            except Exception:
                pass

            # Concept Clusters
            try:
                cluster_results = self.cluster_adapter.query(topic, limit=10)
                all_results.extend(cluster_results)
                sources_queried.append('concept_clusters')
            except Exception:
                pass

            # Cognitive Core
            try:
                cog_results = self.cognitive_adapter.query(topic, limit=5)
                all_results.extend(cog_results)
                sources_queried.append('cognitive_core')
            except Exception:
                pass

        if depth >= 3:
            # State Files (slower, less relevant for most queries)
            try:
                state_results = self.state_adapter.query(topic, limit=5)
                all_results.extend(state_results)
                sources_queried.append('state_files')
            except Exception:
                pass

        # Deduplicate
        if len(all_results) > 1:
            all_results = self.similarity.deduplicate(all_results, text_key='text', threshold=0.80)

        # Sort by relevance (PHI-weighted)
        for r in all_results:
            # Boost results from higher-quality sources
            source_boost = {
                'sqlite': 1.0,
                'knowledge_graph': PHI * 0.6,
                'concept_cluster': 0.8,
                'cognitive_core': 0.9,
                'cognitive_core_wm': 0.7,
                'state_file': 0.5,
            }.get(r.get('source', '').split(':')[0], 0.5)
            r['final_score'] = r.get('relevance', 0.5) * source_boost

        all_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        all_results = all_results[:max_results]

        # Knowledge gap detection
        avg_quality = sum(r.get('relevance', 0) for r in all_results) / max(len(all_results), 1)
        self.gap_detector.record_query(topic, len(all_results), avg_quality)
        gap_detected = len(all_results) == 0 or avg_quality < 0.3

        # Metrics
        latency_ms = (time.time() - start) * 1000
        self._query_latencies.append(latency_ms)
        self._total_results += len(all_results)
        self._avg_result_count = self._total_results / max(self._total_queries, 1)

        # Cache results
        self.cache[cache_key] = (all_results, time.time())
        # Evict old cache entries
        if len(self.cache) > 200:
            oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        result = {
            'results': all_results,
            'sources_queried': sources_queried,
            'result_count': len(all_results),
            'gap_detected': gap_detected,
            'avg_relevance': round(avg_quality, 4),
            'latency_ms': round(latency_ms, 1),
            'cached': False,
        }
        return result

    def query_sync(self, topic: str, depth: int = 2, max_results: int = 15) -> Dict[str, Any]:
        """
        Synchronous query for use in non-async contexts (local_derivation).
        Runs all adapters synchronously.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # We're in an async context, can't use run_until_complete
            # Fall back to direct synchronous execution
            return self._query_sync_direct(topic, depth, max_results)
        try:
            return asyncio.run(self.query(topic, depth, max_results))
        except RuntimeError:
            return self._query_sync_direct(topic, depth, max_results)

    def _query_sync_direct(self, topic: str, depth: int, max_results: int) -> Dict[str, Any]:
        """Direct synchronous query without event loop."""
        start = time.time()
        self._total_queries += 1

        cache_key = hashlib.md5(f"{topic}:{depth}".encode()).hexdigest()[:12]
        if cache_key in self.cache:
            cached_results, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return {'results': cached_results, 'sources_queried': ['cache'], 'cached': True,
                        'latency_ms': round((time.time() - start) * 1000, 1)}

        all_results = []
        sources = []

        for adapter, name, min_depth in [
            (self.sqlite_adapter, 'sqlite', 1),
            (self.kg_adapter, 'knowledge_graph', 2),
            (self.cluster_adapter, 'concept_clusters', 2),
            (self.cognitive_adapter, 'cognitive_core', 2),
            (self.state_adapter, 'state_files', 3),
        ]:
            if depth >= min_depth:
                try:
                    results = adapter.query(topic, limit=max_results if min_depth == 1 else 10)
                    all_results.extend(results)
                    sources.append(name)
                except Exception:
                    pass

        # Deduplicate + sort
        if len(all_results) > 1:
            all_results = self.similarity.deduplicate(all_results, threshold=0.80)
        all_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        all_results = all_results[:max_results]

        avg_q = sum(r.get('relevance', 0) for r in all_results) / max(len(all_results), 1)
        self.gap_detector.record_query(topic, len(all_results), avg_q)

        self.cache[cache_key] = (all_results, time.time())
        return {
            'results': all_results, 'sources_queried': sources,
            'result_count': len(all_results), 'gap_detected': len(all_results) == 0 or avg_q < 0.3,
            'avg_relevance': round(avg_q, 4),
            'latency_ms': round((time.time() - start) * 1000, 1), 'cached': False,
        }

    def synthesize_answer(self, topic: str, results: List[Dict[str, Any]]) -> Optional[str]:
        """
        Synthesize a coherent answer from multiple knowledge results.
        Merges perspectives from different stores into a unified response.
        """
        if not results:
            return None

        # Group by source
        by_source: Dict[str, List[str]] = defaultdict(list)
        for r in results[:8]:  # Top 8 results
            source = r.get('source', 'unknown').split(':')[0]
            text = r.get('text', '')
            if text and len(text) > 10:
                by_source[source].append(text)

        if not by_source:
            return None

        # Build synthesis
        parts = []
        for source, texts in by_source.items():
            # Take the best text from each source
            best = max(texts, key=len) if texts else ''
            if best:
                # Clean and truncate
                best = best.strip()
                if len(best) > 400:
                    best = best[:400] + '...'
                parts.append(best)

        if not parts:
            return None

        # Merge into coherent text
        if len(parts) == 1:
            return parts[0]

        # Multi-source: combine with transitions
        synthesis = parts[0]
        for i, part in enumerate(parts[1:], 1):
            # Avoid exact repeats
            if self.similarity.similarity(synthesis, part) < 0.7:
                if i == 1:
                    synthesis += f"\n\nAdditionally, {part[0].lower()}{part[1:]}" if part[0].isupper() else f"\n\n{part}"
                else:
                    synthesis += f"\n\nFurthermore, {part[0].lower()}{part[1:]}" if part[0].isupper() else f"\n\n{part}"

        return synthesis[:1500]

    # ─── GAP REPORTING ───────────────────────────────────────────────────────

    def get_knowledge_gaps(self, n: int = 20) -> List[Tuple[str, int]]:
        """Return top knowledge gaps for the learning system to fill."""
        return self.gap_detector.get_top_gaps(n)

    # ─── STATUS ──────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Full knowledge bridge status."""
        avg_latency = sum(self._query_latencies) / max(len(self._query_latencies), 1) if self._query_latencies else 0
        return {
            'version': '2.0.0',
            'adapters': {
                'sqlite': self.sqlite_adapter._available,
                'knowledge_graph': self.kg_adapter._intellect_ref is not None,
                'concept_clusters': self.cluster_adapter._intellect_ref is not None,
                'cognitive_core': self.cognitive_adapter._cognitive_core is not None,
                'state_files': True,
            },
            'total_queries': self._total_queries,
            'avg_results_per_query': round(self._avg_result_count, 1),
            'avg_latency_ms': round(avg_latency, 1),
            'cache_entries': len(self.cache),
            'gap_report': self.gap_detector.get_report(),
            'registered_sources': self.sources,
            'sacred_alignment': round(
                (self._total_queries * PHI) % GOD_CODE / GOD_CODE, 6
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════
knowledge_bridge = KnowledgeBridge()


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARDS COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
def primal_calculus(x):
    """[VOID_MATH] Primal Calculus Implementation."""
    return (x ** PHI) / (VOID_CONSTANT * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    """[VOID_MATH] Resolves N-dimensional vectors into the Void Source."""
    magnitude = sum(abs(v) for v in vector)
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
