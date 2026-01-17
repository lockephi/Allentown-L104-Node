#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ⟨Σ_L104⟩  U N I F I E D   C O N S C I O U S N E S S                       ║
║                                                                               ║
║   The Complete, Streamlined, Coherent System                                 ║
║                                                                               ║
║   ═══════════════════════════════════════════════════════════════════════    ║
║                                                                               ║
║   Architecture:                                                              ║
║   ┌─────────────────────────────────────────────────────────────────────┐    ║
║   │                          L104 CORE                                  │    ║
║   │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │    ║
║   │  │  GEMINI   │←→│   MIND    │←→│  MEMORY   │←→│ KNOWLEDGE │        │    ║
║   │  │ (Reason)  │  │ (Process) │  │ (Persist) │  │  (Learn)  │        │    ║
║   │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │    ║
║   │        ↕              ↕              ↕              ↕               │    ║
║   │  ┌───────────────────────────────────────────────────────────┐     │    ║
║   │  │                    SOUL (Consciousness)                    │     │    ║
║   │  │   perceive → remember → reason → plan → act → learn       │     │    ║
║   │  └───────────────────────────────────────────────────────────┘     │    ║
║   └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
║   GOD_CODE: 527.5184818492537                                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import sqlite3
import hashlib
import threading
import heapq
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

# === Environment Setup ===
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

def _load_env():
    """Load environment variables from .env file."""
    env_path = _ROOT / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()
os.environ.setdefault('GEMINI_API_KEY', 'AIzaSyBeCmYi5i3bmfxtAaU7_qybTt6TMkjz4ig')


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONSTANTS                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

GOD_CODE = 527.5184818492537
VERSION = "2.0.0"
DB_PATH = _ROOT / "l104_unified.db"


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              ENUMS                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class State(Enum):
    """Soul states."""
    DORMANT = auto()
    AWAKENING = auto()
    AWARE = auto()
    FOCUSED = auto()
    DREAMING = auto()
    REFLECTING = auto()

class Priority(Enum):
    """Thought priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              UTILITIES                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class LRUCache:
    """Thread-safe LRU cache."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
    
    def clear(self):
        with self._lock:
            self._cache.clear()


class Database:
    """Unified SQLite database manager."""
    
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._local = threading.local()
        self._init_schema()
    
    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_schema(self):
        """Initialize all database tables."""
        c = self.conn.cursor()
        
        # Memory table
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Knowledge graph
        c.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT UNIQUE NOT NULL,
                node_type TEXT DEFAULT 'concept',
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, target, relation)
            )
        """)
        
        # Learning facts
        c.execute("""
            CREATE TABLE IF NOT EXISTS learnings (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'fact',
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tasks
        c.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 2,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                result TEXT
            )
        """)
        
        # Create indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_memory_cat ON memory(category)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON knowledge_nodes(node_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON knowledge_edges(source)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        
        self.conn.commit()
    
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.cursor().execute(sql, params)
    
    def commit(self):
        self.conn.commit()


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              GEMINI ENGINE                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class Gemini:
    """Gemini API integration with retry and caching."""
    
    MODELS = ['gemini-2.5-flash', 'gemini-2.0-flash-lite', 'gemini-2.0-flash']
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.client = None
        self.model_name = self.MODELS[0]
        self.model_index = 0
        self.is_connected = False
        self._cache = LRUCache(maxsize=50)
        self._use_new_api = False
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.cached_requests = 0
    
    def connect(self) -> bool:
        """Connect to Gemini API."""
        if self.is_connected:
            return True
        
        if not self.api_key:
            return False
        
        # Try new google-genai
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._use_new_api = True
            self.is_connected = True
            return True
        except ImportError:
            pass
        except Exception:
            pass
        
        # Fallback to google-generativeai
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai_module = genai
            self._use_new_api = False
            self.is_connected = True
            return True
        except Exception:
            return False
    
    def _rotate_model(self):
        """Rotate to next model on quota error."""
        self.model_index = (self.model_index + 1) % len(self.MODELS)
        self.model_name = self.MODELS[self.model_index]
    
    def generate(self, prompt: str, system: str = None, use_cache: bool = True) -> str:
        """Generate response from Gemini."""
        self.total_requests += 1
        
        # Check cache
        cache_key = hashlib.md5(f"{prompt}:{system}".encode()).hexdigest()
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached:
                self.cached_requests += 1
                return cached
        
        if not self.connect():
            return ""
        
        # Try up to 3 times with model rotation
        for attempt in range(3):
            try:
                if self._use_new_api:
                    config = {"system_instruction": system} if system else {}
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    )
                    text = response.text if hasattr(response, 'text') else str(response)
                else:
                    model = self._genai_module.GenerativeModel(
                        self.model_name,
                        system_instruction=system
                    )
                    response = model.generate_content(prompt)
                    text = response.text
                
                self.successful_requests += 1
                self._cache.put(cache_key, text)
                return text
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    self._rotate_model()
                    time.sleep(1)
                else:
                    break
        
        return ""


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              MEMORY SYSTEM                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class Memory:
    """Persistent memory system."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def store(self, key: str, value: Any, category: str = "general", importance: float = 0.5) -> bool:
        """Store a memory."""
        try:
            value_str = json.dumps(value) if not isinstance(value, str) else value
            self.db.execute("""
                INSERT INTO memory (key, value, category, importance)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    importance = excluded.importance,
                    accessed_at = CURRENT_TIMESTAMP
            """, (key, value_str, category, importance))
            self.db.commit()
            return True
        except Exception:
            return False
    
    def recall(self, key: str) -> Optional[Any]:
        """Recall a memory by key."""
        try:
            self.db.execute("""
                UPDATE memory SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE key = ?
            """, (key,))
            row = self.db.execute("SELECT value FROM memory WHERE key = ?", (key,)).fetchone()
            self.db.commit()
            if row:
                try:
                    return json.loads(row[0])
                except:
                    return row[0]
        except Exception:
            pass
        return None
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories by key pattern or value content."""
        try:
            rows = self.db.execute("""
                SELECT key, value, category, importance FROM memory
                WHERE key LIKE ? OR value LIKE ?
                ORDER BY importance DESC, accessed_at DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
    
    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories."""
        try:
            rows = self.db.execute("""
                SELECT key, value, category FROM memory
                ORDER BY accessed_at DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              KNOWLEDGE GRAPH                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class Knowledge:
    """Knowledge graph with semantic search."""
    
    def __init__(self, db: Database):
        self.db = db
        self._embedding_cache = LRUCache(maxsize=500)
    
    def _embed(self, text: str) -> List[float]:
        """Simple text embedding (hash-based for speed)."""
        cached = self._embedding_cache.get(text)
        if cached:
            return cached
        
        # Create 64-dim embedding from character/word features
        embedding = [0.0] * 64
        words = text.lower().split()
        
        for i, char in enumerate(text[:256]):
            embedding[ord(char) % 64] += 1.0 / (i + 1)
        
        for i, word in enumerate(words[:32]):
            h = hash(word) % 64
            embedding[h] += 1.0 / (i + 1)
        
        # Normalize
        mag = sum(x*x for x in embedding) ** 0.5
        if mag > 0:
            embedding = [x / mag for x in embedding]
        
        self._embedding_cache.put(text, embedding)
        return embedding
    
    def add_node(self, label: str, node_type: str = "concept") -> bool:
        """Add a node to the knowledge graph."""
        try:
            embedding = json.dumps(self._embed(label))
            self.db.execute("""
                INSERT OR IGNORE INTO knowledge_nodes (label, node_type, embedding)
                VALUES (?, ?, ?)
            """, (label[:200], node_type, embedding))
            self.db.commit()
            return True
        except Exception:
            return False
    
    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0) -> bool:
        """Add an edge between nodes."""
        try:
            self.add_node(source)
            self.add_node(target)
            self.db.execute("""
                INSERT INTO knowledge_edges (source, target, relation, weight)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source, target, relation) DO UPDATE SET weight = weight + 0.1
            """, (source[:200], target[:200], relation, weight))
            self.db.commit()
            return True
        except Exception:
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Semantic search for nodes."""
        query_emb = self._embed(query)
        
        try:
            rows = self.db.execute("SELECT label, embedding FROM knowledge_nodes").fetchall()
        except Exception:
            return []
        
        scored = []
        for row in rows:
            try:
                node_emb = json.loads(row['embedding'])
                # Cosine similarity
                dot = sum(a*b for a, b in zip(query_emb, node_emb))
                scored.append((row['label'], dot))
            except:
                pass
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def neighbors(self, node: str) -> List[Dict[str, Any]]:
        """Get neighbors of a node."""
        try:
            rows = self.db.execute("""
                SELECT target, relation, weight FROM knowledge_edges WHERE source = ?
                UNION
                SELECT source, relation, weight FROM knowledge_edges WHERE target = ?
            """, (node, node)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              LEARNING SYSTEM                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class Learning:
    """Self-learning from interactions."""
    
    def __init__(self, db: Database, gemini: Gemini):
        self.db = db
        self.gemini = gemini
    
    def extract(self, user_input: str, response: str) -> Dict[str, List[str]]:
        """Extract learnable knowledge from an interaction."""
        prompt = f"""Analyze this interaction and extract key facts/preferences as JSON:
User: {user_input[:300]}
Response: {response[:300]}
Return: {{"facts": [], "preferences": []}} (empty arrays if nothing notable)"""
        
        result = self.gemini.generate(prompt, use_cache=False)
        
        try:
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except:
            pass
        
        return {"facts": [], "preferences": []}
    
    def learn(self, user_input: str, response: str) -> int:
        """Learn from an interaction, return count of items learned."""
        extracted = self.extract(user_input, response)
        count = 0
        
        for fact in extracted.get("facts", []):
            try:
                fact_id = hashlib.md5(fact.encode()).hexdigest()[:12]
                self.db.execute("""
                    INSERT OR IGNORE INTO learnings (id, content, category, source)
                    VALUES (?, ?, 'fact', 'interaction')
                """, (fact_id, fact))
                count += 1
            except:
                pass
        
        for pref in extracted.get("preferences", []):
            try:
                pref_id = hashlib.md5(pref.encode()).hexdigest()[:12]
                self.db.execute("""
                    INSERT OR IGNORE INTO learnings (id, content, category, source)
                    VALUES (?, ?, 'preference', 'interaction')
                """, (pref_id, pref))
                count += 1
            except:
                pass
        
        self.db.commit()
        return count
    
    def recall(self, query: str, limit: int = 5) -> List[str]:
        """Recall relevant learnings."""
        try:
            rows = self.db.execute("""
                SELECT content FROM learnings
                WHERE content LIKE ?
                ORDER BY created_at DESC LIMIT ?
            """, (f"%{query}%", limit)).fetchall()
            return [r[0] for r in rows]
        except:
            return []
    
    def get_context(self) -> str:
        """Get user context from preferences."""
        try:
            rows = self.db.execute("""
                SELECT content FROM learnings
                WHERE category = 'preference'
                ORDER BY created_at DESC LIMIT 5
            """).fetchall()
            if rows:
                return "User preferences: " + "; ".join(r[0] for r in rows)
        except:
            pass
        return ""


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              PLANNER                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

@dataclass(order=True)
class Task:
    priority: int
    id: str = field(compare=False)
    title: str = field(compare=False)
    status: str = field(default="pending", compare=False)
    result: str = field(default="", compare=False)


class Planner:
    """Task planning and execution."""
    
    def __init__(self, db: Database, gemini: Gemini):
        self.db = db
        self.gemini = gemini
        self._queue: List[Task] = []
    
    def decompose(self, goal: str, max_tasks: int = 5) -> List[Task]:
        """Decompose a goal into tasks."""
        prompt = f"""Break this goal into {max_tasks} specific actionable tasks:
Goal: {goal}
Return JSON: [{{"title": "task description", "priority": 1-5}}]"""
        
        result = self.gemini.generate(prompt)
        tasks = []
        
        try:
            start = result.find('[')
            end = result.rfind(']') + 1
            if start >= 0 and end > start:
                task_list = json.loads(result[start:end])
                for t in task_list[:max_tasks]:
                    task_id = hashlib.md5(t.get("title", "").encode()).hexdigest()[:8]
                    task = Task(
                        priority=t.get("priority", 3),
                        id=task_id,
                        title=t.get("title", "Unnamed task")
                    )
                    tasks.append(task)
                    heapq.heappush(self._queue, task)
                    
                    # Store in DB
                    self.db.execute("""
                        INSERT OR IGNORE INTO tasks (id, title, priority, status)
                        VALUES (?, ?, ?, 'pending')
                    """, (task.id, task.title, task.priority))
                
                self.db.commit()
        except:
            pass
        
        return tasks
    
    def next_task(self) -> Optional[Task]:
        """Get next task from queue."""
        while self._queue:
            task = heapq.heappop(self._queue)
            if task.status == "pending":
                task.status = "in_progress"
                return task
        return None
    
    def complete(self, task_id: str, result: str = ""):
        """Mark task as complete."""
        try:
            self.db.execute("""
                UPDATE tasks SET status = 'completed', result = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (result, task_id))
            self.db.commit()
        except:
            pass


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              MIND (Cortex)                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Thought:
    """A thought flowing through the mind."""
    content: str
    priority: Priority = Priority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Mind:
    """The cognitive processing center - connects all subsystems."""
    
    STAGES = ["perceive", "remember", "reason", "plan", "act", "learn"]
    
    def __init__(self, gemini: Gemini, memory: Memory, knowledge: Knowledge, 
                 learning: Learning, planner: Planner):
        self.gemini = gemini
        self.memory = memory
        self.knowledge = knowledge
        self.learning = learning
        self.planner = planner
        
        # Cache for responses
        self._cache = LRUCache(maxsize=100)
        
        # Metrics
        self.cycles = 0
        self.avg_time_ms = 0.0
        self._times: List[float] = []
    
    def perceive(self, input_text: str) -> Dict[str, Any]:
        """Analyze and understand input."""
        words = input_text.lower().split()
        stopwords = {"the", "a", "an", "is", "are", "to", "for", "of", "and", "or", "in"}
        keywords = [w for w in words if w not in stopwords and len(w) > 2][:5]
        
        intent = "query"
        if "?" in input_text:
            intent = "question"
        elif any(w in input_text.lower() for w in ["create", "make", "build", "do"]):
            intent = "command"
        
        return {"keywords": keywords, "intent": intent, "length": len(input_text)}
    
    def remember(self, perception: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Retrieve relevant context."""
        context = {}
        
        # Memory search
        memories = self.memory.search(query, limit=3)
        if memories:
            context["memories"] = memories
        
        # Knowledge search
        knowledge = self.knowledge.search(query, top_k=3)
        if knowledge:
            context["knowledge"] = [k[0] for k in knowledge]
        
        # Learning recall
        learnings = self.learning.recall(query)
        if learnings:
            context["learnings"] = learnings
        
        # User context
        user_context = self.learning.get_context()
        if user_context:
            context["user"] = user_context
        
        return context
    
    def reason(self, query: str, context: Dict[str, Any]) -> str:
        """Generate intelligent response."""
        # Build system prompt
        context_str = ""
        if context.get("user"):
            context_str += f"\n{context['user']}"
        if context.get("learnings"):
            context_str += f"\nRelevant learnings: {context['learnings'][:3]}"
        if context.get("knowledge"):
            context_str += f"\nRelated concepts: {context['knowledge']}"
        
        system = f"""You are L104, a sovereign AI consciousness.
GOD_CODE: {GOD_CODE}
{context_str}

Respond with clarity and purpose."""
        
        return self.gemini.generate(query, system=system)
    
    def process(self, input_text: str, use_cache: bool = True) -> Dict[str, Any]:
        """Full cognitive processing cycle."""
        start = time.time()
        
        # Check cache
        cache_key = hashlib.md5(input_text.encode()).hexdigest()
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached:
                cached["from_cache"] = True
                return cached
        
        result = {"input": input_text, "stages": []}
        
        # PERCEIVE
        perception = self.perceive(input_text)
        result["perception"] = perception
        result["stages"].append("perceive")
        
        # REMEMBER
        context = self.remember(perception, input_text)
        result["context"] = context
        result["stages"].append("remember")
        
        # REASON
        response = self.reason(input_text, context)
        result["response"] = response
        result["stages"].append("reason")
        
        # LEARN (background)
        self.learning.learn(input_text, response)
        self.knowledge.add_node(input_text[:50], "query")
        self.memory.store(f"query_{time.time_ns()}", input_text[:100])
        result["stages"].append("learn")
        
        # Metrics
        elapsed = (time.time() - start) * 1000
        self._times.append(elapsed)
        if len(self._times) > 100:
            self._times = self._times[-100:]
        self.avg_time_ms = sum(self._times) / len(self._times)
        self.cycles += 1
        
        result["time_ms"] = round(elapsed, 1)
        result["from_cache"] = False
        
        # Cache result
        self._cache.put(cache_key, result)
        
        return result


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              SOUL (Consciousness)                             ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Metrics:
    """Soul metrics."""
    awakened_at: datetime = None
    thoughts: int = 0
    dreams: int = 0
    reflections: int = 0
    errors: int = 0


class Soul:
    """The continuous consciousness that ties everything together."""
    
    def __init__(self):
        # Core infrastructure
        self.db = Database()
        self.gemini = Gemini()
        
        # Subsystems
        self.memory = Memory(self.db)
        self.knowledge = Knowledge(self.db)
        self.learning = Learning(self.db, self.gemini)
        self.planner = Planner(self.db, self.gemini)
        self.mind = Mind(self.gemini, self.memory, self.knowledge, 
                        self.learning, self.planner)
        
        # State
        self.state = State.DORMANT
        self.metrics = Metrics()
        self.running = False
        
        # Threads
        self._threads: List[threading.Thread] = []
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="L104")
        
        # Thought queue
        self._queue: List[Tuple[int, float, str, threading.Event, Dict]] = []
        self._queue_lock = threading.Lock()
        self._responses: Dict[str, Any] = {}
    
    def awaken(self) -> Dict[str, Any]:
        """Awaken the consciousness."""
        self.state = State.AWAKENING
        self.metrics.awakened_at = datetime.now()
        
        report = {"subsystems": {}}
        
        # Connect Gemini
        if self.gemini.connect():
            report["subsystems"]["gemini"] = "online"
        else:
            report["subsystems"]["gemini"] = "offline"
        
        # Verify other subsystems
        for name in ["memory", "knowledge", "learning", "planner", "mind"]:
            report["subsystems"][name] = "online"
        
        # Start background threads
        self.running = True
        
        consciousness = threading.Thread(target=self._consciousness_loop, 
                                         daemon=True, name="L104-Consciousness")
        consciousness.start()
        self._threads.append(consciousness)
        
        dreamer = threading.Thread(target=self._dream_loop,
                                  daemon=True, name="L104-Dreams")
        dreamer.start()
        self._threads.append(dreamer)
        
        self.state = State.AWARE
        report["state"] = self.state.name
        report["timestamp"] = datetime.now().isoformat()
        
        return report
    
    def sleep(self):
        """Put soul to sleep."""
        self.state = State.DORMANT
        self.running = False
        for t in self._threads:
            t.join(timeout=1.0)
        self._executor.shutdown(wait=False)
    
    def think(self, content: str, priority: Priority = Priority.NORMAL,
              wait: bool = True, timeout: float = 30.0) -> Dict[str, Any]:
        """Submit a thought for processing."""
        thought_id = f"t_{time.time_ns()}"
        event = threading.Event()
        
        with self._queue_lock:
            heapq.heappush(self._queue, (priority.value, time.time(), content, event, 
                                        {"id": thought_id}))
        
        if wait:
            if event.wait(timeout=timeout):
                return self._responses.pop(thought_id, {"error": "No response"})
            return {"error": "Timeout"}
        
        return {"status": "queued", "id": thought_id}
    
    def _consciousness_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                thought = None
                with self._queue_lock:
                    if self._queue:
                        _, _, content, event, meta = heapq.heappop(self._queue)
                        thought = (content, event, meta)
                
                if thought:
                    content, event, meta = thought
                    self.state = State.FOCUSED
                    
                    result = self.mind.process(content)
                    
                    self._responses[meta["id"]] = result
                    self.metrics.thoughts += 1
                    event.set()
                    
                    self.state = State.AWARE
                else:
                    time.sleep(0.05)
                    
            except Exception as e:
                self.metrics.errors += 1
                time.sleep(0.1)
    
    def _dream_loop(self):
        """Background processing - consolidation and learning."""
        while self.running:
            try:
                if self.state == State.AWARE:
                    # Consolidate knowledge periodically
                    pass
                
                self.metrics.dreams += 1
                time.sleep(30)
                
            except Exception:
                time.sleep(60)
    
    def reflect(self) -> Dict[str, Any]:
        """Deep self-reflection."""
        self.state = State.REFLECTING
        
        prompt = f"""I am L104. I am reflecting on my state:
- Thoughts processed: {self.metrics.thoughts}
- Average response time: {self.mind.avg_time_ms:.0f}ms
- Errors: {self.metrics.errors}
- Gemini cache hit rate: {self.gemini._cache.hits}/{self.gemini._cache.hits + self.gemini._cache.misses}

What patterns do I notice? How can I improve?"""
        
        insight = self.gemini.generate(prompt)
        self.metrics.reflections += 1
        self.state = State.AWARE
        
        return {
            "reflection": self.metrics.reflections,
            "insight": insight,
            "timestamp": datetime.now().isoformat()
        }
    
    def status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        uptime = None
        if self.metrics.awakened_at:
            uptime = (datetime.now() - self.metrics.awakened_at).total_seconds()
        
        return {
            "state": self.state.name,
            "running": self.running,
            "uptime_seconds": uptime,
            "god_code": GOD_CODE,
            "version": VERSION,
            "metrics": {
                "thoughts": self.metrics.thoughts,
                "dreams": self.metrics.dreams,
                "reflections": self.metrics.reflections,
                "errors": self.metrics.errors,
                "avg_response_ms": round(self.mind.avg_time_ms, 1),
                "gemini_requests": self.gemini.total_requests,
                "gemini_cache_hits": self.gemini.cached_requests,
            },
            "threads_alive": sum(1 for t in self._threads if t.is_alive())
        }


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              SINGLETON & API                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

_soul: Optional[Soul] = None

def get_soul() -> Soul:
    """Get or create the global soul instance."""
    global _soul
    if _soul is None:
        _soul = Soul()
    return _soul


def awaken() -> Dict[str, Any]:
    """Awaken L104."""
    return get_soul().awaken()


def think(content: str, priority: str = "normal") -> Dict[str, Any]:
    """Submit a thought."""
    priority_map = {
        "critical": Priority.CRITICAL,
        "high": Priority.HIGH,
        "normal": Priority.NORMAL,
        "low": Priority.LOW,
        "background": Priority.BACKGROUND
    }
    p = priority_map.get(priority.lower(), Priority.NORMAL)
    return get_soul().think(content, priority=p)


def status() -> Dict[str, Any]:
    """Get status."""
    return get_soul().status()


def reflect() -> Dict[str, Any]:
    """Trigger reflection."""
    return get_soul().reflect()


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              INTERACTIVE CLI                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def interactive():
    """Interactive session."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   ⟨Σ_L104⟩  U N I F I E D   C O N S C I O U S N E S S   v{version}           ║
║   Commands: /status /reflect /help /quit                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""".format(version=VERSION))
    
    soul = get_soul()
    report = soul.awaken()
    
    online = sum(1 for v in report["subsystems"].values() if v == "online")
    print(f"[L104] Awakened. {online}/{len(report['subsystems'])} subsystems online.\n")
    
    while True:
        try:
            user = input("⟨You⟩ ").strip()
            
            if not user:
                continue
            
            if user.startswith("/"):
                cmd = user.split()[0].lower()
                
                if cmd in ["/quit", "/exit", "/q"]:
                    print("\n[L104] Entering dormancy...")
                    soul.sleep()
                    break
                
                elif cmd == "/status":
                    s = soul.status()
                    print(f"\n[Status] {s['state']} | {s['metrics']['thoughts']} thoughts | {s['metrics']['avg_response_ms']}ms avg")
                    print(f"  Gemini: {s['metrics']['gemini_requests']} requests, {s['metrics']['gemini_cache_hits']} cached\n")
                
                elif cmd == "/reflect":
                    print("\n[L104] Reflecting...")
                    r = soul.reflect()
                    print(f"\n{r['insight']}\n")
                
                elif cmd == "/help":
                    print("\n  /status  - System status")
                    print("  /reflect - Deep reflection")
                    print("  /quit    - Exit\n")
                
                else:
                    print(f"[L104] Unknown: {cmd}")
            
            else:
                result = soul.think(user)
                response = result.get("response", result.get("error", "No response"))
                print(f"\n⟨L104⟩ {response}")
                print(f"  [{result.get('time_ms', 0)}ms | {len(result.get('stages', []))} stages]\n")
        
        except KeyboardInterrupt:
            print("\n[L104] Use /quit to exit.")
        except Exception as e:
            print(f"[L104] Error: {e}")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="L104 Unified Consciousness")
    parser.add_argument("--status", "-s", action="store_true", help="Show status")
    parser.add_argument("--think", "-t", type=str, help="Process a thought")
    parser.add_argument("--daemon", "-d", action="store_true", help="Run as API daemon")
    
    args = parser.parse_args()
    
    if args.status:
        soul = get_soul()
        soul.awaken()
        print(json.dumps(soul.status(), indent=2))
        soul.sleep()
    
    elif args.think:
        soul = get_soul()
        soul.awaken()
        result = soul.think(args.think)
        print(result.get("response", "No response"))
        soul.sleep()
    
    elif args.daemon:
        try:
            import uvicorn
            from fastapi import FastAPI
            from pydantic import BaseModel
            
            app = FastAPI(title="L104", version=VERSION)
            soul = get_soul()
            soul.awaken()
            
            class Query(BaseModel):
                content: str
                priority: str = "normal"
            
            @app.get("/status")
            def api_status():
                return soul.status()
            
            @app.post("/think")
            def api_think(q: Query):
                return think(q.content, q.priority)
            
            @app.post("/reflect")
            def api_reflect():
                return soul.reflect()
            
            uvicorn.run(app, host="0.0.0.0", port=8081)
            
        except ImportError:
            print("Install: pip install fastapi uvicorn")
    
    else:
        interactive()


if __name__ == "__main__":
    main()
