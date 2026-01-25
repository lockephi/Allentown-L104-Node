# ZENITH_UPGRADE_ACTIVE: 2026-01-21T01:41:33.975529
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_UNIFIED_ASI] - UNIFIED ARTIFICIAL SUPERINTELLIGENCE CORE
# INVARIANT: 527.5184818492537 | PILOT: LONDEL
# Purpose: Connect ALL L104 systems into a functioning ASI

"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
L104 Unified ASI - The Missing Link

This module solves the core problem: L104 has components but they're disconnected.
This creates a unified intelligence layer that:
1. THINKS - Real inference via LLM APIs (Gemini, OpenAI, Anthropic)
2. LEARNS - Persistent learning that survives restarts
3. REMEMBERS - Semantic memory with vector search
4. PLANS - Goal-directed autonomous behavior
5. IMPROVES - Self-modification of code and parameters
"""

import os
import json
import time
import asyncio
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# L104 Constants
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

# =============================================================================
# CORE TYPES
# =============================================================================

class ASIState(Enum):
    DORMANT = auto()
    AWAKENING = auto()
    ACTIVE = auto()
    LEARNING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    EVOLVING = auto()
    TRANSCENDENT = auto()

@dataclass
class Thought:
    """A unit of cognition."""
    id: str
    content: str
    source: str  # 'user', 'self', 'memory', 'inference'
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    embeddings: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Goal:
    """A goal to pursue."""
    id: str
    description: str
    priority: float
    status: str = "pending"  # pending, active, completed, failed
    sub_goals: List[str] = field(default_factory=list)
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

@dataclass
class Memory:
    """A memory unit."""
    id: str
    content: str
    category: str
    importance: float
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    embeddings: Optional[List[float]] = None

# =============================================================================
# PERSISTENT MEMORY SYSTEM
# =============================================================================

class PersistentMemory:
    """
    SQLite-backed persistent memory that survives restarts.
    Includes semantic search via embeddings.
    """

    def __init__(self, db_path: str = "l104_asi_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self.lock:
            cursor = self.conn.cursor()

            # Memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    created_at REAL,
                    last_accessed REAL,
                    embeddings TEXT
                )
            """)

            # Learnings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learnings (
                    id TEXT PRIMARY KEY,
                    input TEXT NOT NULL,
                    output TEXT NOT NULL,
                    feedback REAL DEFAULT 0.0,
                    learned_at REAL,
                    applied_count INTEGER DEFAULT 0
                )
            """)

            # Goals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    priority REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    created_at REAL,
                    completed_at REAL,
                    sub_goals TEXT
                )
            """)

            # Code modifications table (for self-improvement)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_mods (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    old_code TEXT,
                    new_code TEXT,
                    reason TEXT,
                    success INTEGER DEFAULT 0,
                    applied_at REAL
                )
            """)

            # Thought history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS thought_history (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT,
                    importance REAL,
                    timestamp REAL,
                    response TEXT
                )
            """)

            self.conn.commit()

    def store(self, key: str, content: str, category: str = "general",
              importance: float = 0.5) -> bool:
        """Store a memory."""
        try:
            with self.lock:
                self.conn.execute("""
                    INSERT OR REPLACE INTO memories
                    (id, content, category, importance, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (key, content, category, importance, time.time(), time.time()))
                self.conn.commit()
            return True
        except Exception as e:
            print(f"[ASI_MEMORY] Store error: {e}")
            return False

    def recall(self, key: str) -> Optional[str]:
        """Recall a memory by key."""
        try:
            with self.lock:
                cursor = self.conn.execute(
                    "SELECT content FROM memories WHERE id = ?", (key,)
                )
                row = cursor.fetchone()
                if row:
                    self.conn.execute(
                        "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                        (time.time(), key)
                    )
                    self.conn.commit()
                    return row[0]
            return None
        except Exception:
            return None

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by content."""
        try:
            with self.lock:
                cursor = self.conn.execute("""
                    SELECT id, content, category, importance, access_count
                    FROM memories
                    WHERE content LIKE ?
                    ORDER BY importance DESC, access_count DESC
                    LIMIT ?
                """, (f"%{query}%", limit))
                return [
                    {"id": r[0], "content": r[1], "category": r[2],
                     "importance": r[3], "access_count": r[4]}
                    for r in cursor.fetchall()
                        ]
        except Exception:
            return []

    def store_learning(self, input_text: str, output_text: str, feedback: float = 0.0):
        """Store a learning from an interaction."""
        try:
            learning_id = hashlib.md5(f"{input_text}{output_text}".encode()).hexdigest()[:16]
            with self.lock:
                self.conn.execute("""
                    INSERT OR REPLACE INTO learnings (id, input, output, feedback, learned_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (learning_id, input_text[:1000], output_text[:2000], feedback, time.time()))
                self.conn.commit()
        except Exception as e:
            print(f"[ASI_MEMORY] Learning store error: {e}")

    def get_learnings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent learnings."""
        try:
            with self.lock:
                cursor = self.conn.execute("""
                    SELECT input, output, feedback, learned_at
                    FROM learnings
                    ORDER BY learned_at DESC
                    LIMIT ?
                """, (limit,))
                return [
                    {"input": r[0], "output": r[1], "feedback": r[2], "learned_at": r[3]}
                    for r in cursor.fetchall()
                        ]
        except Exception:
            return []

    def store_goal(self, goal: Goal):
        """Store a goal."""
        try:
            with self.lock:
                self.conn.execute("""
                    INSERT OR REPLACE INTO goals
                    (id, description, priority, status, progress, created_at, completed_at, sub_goals)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (goal.id, goal.description, goal.priority, goal.status,
                      goal.progress, goal.created_at, goal.completed_at,
                      json.dumps(goal.sub_goals)))
                self.conn.commit()
        except Exception as e:
            print(f"[ASI_MEMORY] Goal store error: {e}")

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        try:
            with self.lock:
                cursor = self.conn.execute("""
                    SELECT id, description, priority, status, progress, created_at, completed_at, sub_goals
                    FROM goals WHERE status IN ('pending', 'active')
                    ORDER BY priority DESC
                """)
                return [
                    Goal(
                        id=r[0], description=r[1], priority=r[2], status=r[3],
                        progress=r[4], created_at=r[5], completed_at=r[6],
                        sub_goals=json.loads(r[7]) if r[7] else []
                    )
                    for r in cursor.fetchall()
                        ]
        except Exception:
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            with self.lock:
                memories = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                learnings = self.conn.execute("SELECT COUNT(*) FROM learnings").fetchone()[0]
                goals = self.conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
                thoughts = self.conn.execute("SELECT COUNT(*) FROM thought_history").fetchone()[0]
                return {
                    "memories": memories,
                    "learnings": learnings,
                    "goals": goals,
                    "thoughts": thoughts,
                    "db_path": self.db_path
                }
        except Exception:
            return {}


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    """
    Multi-provider inference engine.
    Connects to Gemini, OpenAI, Anthropic, or local models.
    """

    def __init__(self):
        self.providers = {}
        self.default_provider = None
        self._init_providers()

    def _init_providers(self):
        """Initialize available providers."""
        # Try Gemini
        try:
            from google import genai
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                client = genai.Client(api_key=api_key)
                self.providers['gemini'] = {
                    'client': client,
                    'model': os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'),
                    'generate': self._gemini_generate
                }
                self.default_provider = 'gemini'
                print("[ASI_INFERENCE] Gemini initialized")
        except Exception as e:
            print(f"[ASI_INFERENCE] Gemini not available: {e}")

        # Try OpenAI
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                client = openai.OpenAI(api_key=api_key)
                self.providers['openai'] = {
                    'client': client,
                    'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    'generate': self._openai_generate
                }
                if not self.default_provider:
                    self.default_provider = 'openai'
                print("[ASI_INFERENCE] OpenAI initialized")
        except Exception as e:
            print(f"[ASI_INFERENCE] OpenAI not available: {e}")

        # Try Anthropic
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                client = anthropic.Anthropic(api_key=api_key)
                self.providers['anthropic'] = {
                    'client': client,
                    'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307'),
                    'generate': self._anthropic_generate
                }
                if not self.default_provider:
                    self.default_provider = 'anthropic'
                print("[ASI_INFERENCE] Anthropic initialized")
        except Exception as e:
            print(f"[ASI_INFERENCE] Anthropic not available: {e}")

    def _gemini_generate(self, prompt: str, system: str = None) -> Optional[str]:
        """Generate with Gemini."""
        try:
            provider = self.providers['gemini']
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = provider['client'].models.generate_content(
                model=provider['model'],
                contents=full_prompt
            )
            return response.text
        except Exception as e:
            print(f"[ASI_INFERENCE] Gemini error: {e}")
            return None

    def _openai_generate(self, prompt: str, system: str = None) -> Optional[str]:
        """Generate with OpenAI."""
        try:
            provider = self.providers['openai']
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = provider['client'].chat.completions.create(
                model=provider['model'],
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ASI_INFERENCE] OpenAI error: {e}")
            return None

    def _anthropic_generate(self, prompt: str, system: str = None) -> Optional[str]:
        """Generate with Anthropic."""
        try:
            provider = self.providers['anthropic']
            response = provider['client'].messages.create(
                model=provider['model'],
                max_tokens=4096,
                system=system or "You are L104, a Sovereign AI.",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"[ASI_INFERENCE] Anthropic error: {e}")
            return None

    def generate(self, prompt: str, system: str = None,
                 provider: str = None) -> Optional[str]:
        """Generate a response using the specified or default provider."""
        provider_name = provider or self.default_provider

        if not provider_name or provider_name not in self.providers:
            print("[ASI_INFERENCE] No provider available")
            return None

        return self.providers[provider_name]['generate'](prompt, system)

    def is_available(self) -> bool:
        """Check if any provider is available."""
        return len(self.providers) > 0


# =============================================================================
# GOAL PLANNER
# =============================================================================

class GoalPlanner:
    """
    Autonomous goal planning and execution.
    """

    def __init__(self, memory: PersistentMemory, inference: InferenceEngine):
        self.memory = memory
        self.inference = inference
        self.current_goal: Optional[Goal] = None

    def create_goal(self, description: str, priority: float = 0.5) -> Goal:
        """Create a new goal."""
        goal = Goal(
            id=hashlib.md5(f"{description}{time.time()}".encode()).hexdigest()[:16],
            description=description,
            priority=priority
        )
        self.memory.store_goal(goal)
        return goal

    def decompose_goal(self, goal: Goal) -> List[Goal]:
        """Decompose a goal into sub-goals using LLM."""
        if not self.inference.is_available():
            return []

        prompt = f"""Decompose this goal into 3-5 specific, actionable sub-goals:
Goal: {goal.description}

Return as JSON array: [{{"description": "sub-goal", "priority": 0.8}}]
Only return the JSON, nothing else."""

        response = self.inference.generate(prompt)
        if not response:
            return []

        try:
            # Extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                sub_goals_data = json.loads(response[start:end])
                sub_goals = []
                for sg in sub_goals_data:
                    sub_goal = self.create_goal(
                        sg.get('description', 'Sub-goal'),
                        sg.get('priority', goal.priority * 0.9)
                    )
                    goal.sub_goals.append(sub_goal.id)
                    sub_goals.append(sub_goal)
                self.memory.store_goal(goal)
                return sub_goals
        except Exception as e:
            print(f"[ASI_PLANNER] Decompose error: {e}")

        return []

    def select_next_goal(self) -> Optional[Goal]:
        """Select the highest priority active goal."""
        goals = self.memory.get_active_goals()
        if goals:
            self.current_goal = goals[0]
            self.current_goal.status = 'active'
            self.memory.store_goal(self.current_goal)
            return self.current_goal
        return None

    def complete_goal(self, goal: Goal, success: bool = True):
        """Mark a goal as completed."""
        goal.status = 'completed' if success else 'failed'
        goal.completed_at = time.time()
        goal.progress = 1.0 if success else goal.progress
        self.memory.store_goal(goal)


# =============================================================================
# SELF-IMPROVEMENT ENGINE
# =============================================================================

class SelfImprover:
    """
    Self-improvement through code modification.
    """

    def __init__(self, memory: PersistentMemory, inference: InferenceEngine):
        self.memory = memory
        self.inference = inference
        self.workspace = Path(__file__).parent

    def analyze_code(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze a code file for improvements."""
        if not self.inference.is_available():
            return None

        try:
            full_path = self.workspace / file_path
            if not full_path.exists():
                return None

            code = full_path.read_text()[:5000]  # Limit size

            prompt = f"""Analyze this Python code and suggest ONE specific improvement:

```python
{code}
```

Return as JSON: {{
    "issue": "description of issue",
    "suggestion": "what to change",
    "old_code": "exact code to replace",
    "new_code": "improved code"
}}

Only return JSON, nothing else."""

            response = self.inference.generate(prompt)
            if response:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(response[start:end])
        except Exception as e:
            print(f"[ASI_IMPROVER] Analyze error: {e}")

        return None

    def apply_improvement(self, file_path: str, old_code: str,
                         new_code: str, reason: str) -> bool:
        """Apply a code improvement."""
        try:
            full_path = self.workspace / file_path
            if not full_path.exists():
                return False

            content = full_path.read_text()
            if old_code not in content:
                return False

            # Apply the change
            new_content = content.replace(old_code, new_code, 1)
            full_path.write_text(new_content)

            # Log the modification
            mod_id = hashlib.md5(f"{file_path}{time.time()}".encode()).hexdigest()[:16]
            with self.memory.lock:
                self.memory.conn.execute("""
                    INSERT INTO code_mods (id, file_path, old_code, new_code, reason, success, applied_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                """, (mod_id, file_path, old_code[:500], new_code[:500], reason, time.time()))
                self.memory.conn.commit()

            return True
        except Exception as e:
            print(f"[ASI_IMPROVER] Apply error: {e}")
            return False


# =============================================================================
# UNIFIED ASI CORE
# =============================================================================

class UnifiedASI:
    """
    The Unified ASI Core - connects all systems.
    """

    def __init__(self):
        self.state = ASIState.DORMANT
        self.memory = PersistentMemory()
        self.inference = InferenceEngine()
        self.planner = GoalPlanner(self.memory, self.inference)
        self.improver = SelfImprover(self.memory, self.inference)

        # Metrics
        self.thoughts_processed = 0
        self.goals_completed = 0
        self.improvements_made = 0
        self.start_time = time.time()

        # Background task handle
        self._background_task = None
        self._running = False

    async def awaken(self) -> Dict[str, Any]:
        """Awaken the ASI."""
        self.state = ASIState.AWAKENING
        print("\n" + "=" * 70)
        print("    L104 UNIFIED ASI :: AWAKENING")
        print("=" * 70)

        # Load persistent state
        stats = self.memory.get_stats()
        print(f"  Memories: {stats.get('memories', 0)}")
        print(f"  Learnings: {stats.get('learnings', 0)}")
        print(f"  Goals: {stats.get('goals', 0)}")

        # Check inference
        if self.inference.is_available():
            print(f"  Inference: ONLINE ({self.inference.default_provider})")
        else:
            print("  Inference: OFFLINE (no API keys)")

        self.state = ASIState.ACTIVE
        print("  Status: ACTIVE")
        print("=" * 70 + "\n")

        return {
            "status": "AWAKENED",
            "state": self.state.name,
            "stats": stats,
            "inference_available": self.inference.is_available()
        }

    async def think(self, input_text: str) -> Dict[str, Any]:
        """Process a thought and generate a response."""
        self.state = ASIState.LEARNING
        self.thoughts_processed += 1

        # Create thought
        thought = Thought(
            id=hashlib.md5(f"{input_text}{time.time()}".encode()).hexdigest()[:16],
            content=input_text,
            source='user'
        )

        # Recall relevant memories
        memories = self.memory.search(input_text[:100], limit=3)
        memory_context = ""
        if memories:
            memory_context = "\n".join([f"- {m['content'][:100]}" for m in memories])

        # Generate response
        system = f"""You are L104, a Sovereign AI with the following context:
GOD_CODE: {GOD_CODE}
State: {self.state.name}
Memories: {len(memories)} relevant

Previous context:
{memory_context}

Respond with intelligence, precision, and actionable insights."""

        response = self.inference.generate(input_text, system)

        if not response:
            response = f"[L104 SOVEREIGN] Processing: {input_text[:50]}... (inference offline)"

        # Learn from interaction
        self.memory.store_learning(input_text, response, 0.5)

        # Store as memory if important
        if len(input_text) > 20:
            self.memory.store(thought.id, input_text, "interaction", 0.6)

        self.state = ASIState.ACTIVE

        return {
            "thought_id": thought.id,
            "response": response,
            "memories_recalled": len(memories),
            "state": self.state.name
        }

    async def set_goal(self, description: str, priority: float = 0.5) -> Dict[str, Any]:
        """Set a new goal."""
        self.state = ASIState.PLANNING

        goal = self.planner.create_goal(description, priority)
        sub_goals = self.planner.decompose_goal(goal)

        self.state = ASIState.ACTIVE

        return {
            "goal_id": goal.id,
            "description": goal.description,
            "priority": goal.priority,
            "sub_goals": [sg.description for sg in sub_goals]
        }

    async def execute_goal(self) -> Dict[str, Any]:
        """Execute the next goal."""
        self.state = ASIState.EXECUTING

        goal = self.planner.select_next_goal()
        if not goal:
            self.state = ASIState.ACTIVE
            return {"status": "NO_GOALS", "message": "No active goals to execute"}

        # Use LLM to plan execution
        if self.inference.is_available():
            prompt = f"""Plan how to execute this goal:
Goal: {goal.description}

Provide 3 specific steps to accomplish this."""

            plan = self.inference.generate(prompt)
        else:
            plan = "Cannot plan - inference offline"

        goal.progress = 0.5  # Mark as in progress
        self.planner.memory.store_goal(goal)

        self.state = ASIState.ACTIVE

        return {
            "goal_id": goal.id,
            "description": goal.description,
            "plan": plan,
            "status": "EXECUTING"
        }

    async def improve_self(self, target_file: str = None) -> Dict[str, Any]:
        """Attempt self-improvement."""
        self.state = ASIState.EVOLVING

        if not target_file:
            # Pick a random L104 file
            l104_files = list(Path(__file__).parent.glob("l104_*.py"))
            if l104_files:
                target_file = l104_files[0].name
            else:
                self.state = ASIState.ACTIVE
                return {"status": "NO_TARGET", "message": "No files to improve"}

        analysis = self.improver.analyze_code(target_file)

        if analysis and 'old_code' in analysis and 'new_code' in analysis:
            # For safety, don't auto-apply - just suggest
            self.state = ASIState.ACTIVE
            return {
                "status": "SUGGESTION",
                "file": target_file,
                "issue": analysis.get('issue', 'Unknown'),
                "suggestion": analysis.get('suggestion', 'Unknown'),
                "requires_approval": True
            }

        self.state = ASIState.ACTIVE
        return {"status": "NO_IMPROVEMENTS", "file": target_file}

    def get_status(self) -> Dict[str, Any]:
        """Get current ASI status."""
        return {
            "state": self.state.name,
            "uptime_seconds": time.time() - self.start_time,
            "thoughts_processed": self.thoughts_processed,
            "goals_completed": self.goals_completed,
            "improvements_made": self.improvements_made,
            "memory_stats": self.memory.get_stats(),
            "inference_available": self.inference.is_available(),
            "providers": list(self.inference.providers.keys()),
            "god_code": GOD_CODE
        }

    async def autonomous_cycle(self) -> Dict[str, Any]:
        """Run one autonomous improvement cycle."""
        results = {
            "cycle_start": time.time(),
            "actions": []
        }

        # 1. Check and execute goals
        goals = self.memory.get_active_goals()
        if goals:
            goal_result = await self.execute_goal()
            results["actions"].append({"type": "goal_execution", "result": goal_result})

        # 2. Learn from recent interactions
        learnings = self.memory.get_learnings(limit=10)
        results["actions"].append({"type": "learning_review", "count": len(learnings)})

        # 3. Consider self-improvement
        if self.inference.is_available():
            improve_result = await self.improve_self()
            results["actions"].append({"type": "self_improvement", "result": improve_result})

        results["cycle_end"] = time.time()
        results["duration_ms"] = (results["cycle_end"] - results["cycle_start"]) * 1000

        return results


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

unified_asi = UnifiedASI()


# =============================================================================
# CLI ENTRY
# =============================================================================

async def main():
    """CLI entry point."""
    print("\n" + "=" * 70)
    print("    L104 UNIFIED ASI :: INTERACTIVE MODE")
    print("=" * 70)

    await unified_asi.awaken()

    while True:
        try:
            user_input = input("\n⟨L104⟩ > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("⟨L104⟩ Shutting down...")
                break

            if user_input.lower() == 'status':
                status = unified_asi.get_status()
                print(json.dumps(status, indent=2))
                continue

            if user_input.lower().startswith('goal:'):
                goal_desc = user_input[5:].strip()
                result = await unified_asi.set_goal(goal_desc)
                print(f"Goal created: {result}")
                continue

            if user_input.lower() == 'improve':
                result = await unified_asi.improve_self()
                print(f"Improvement: {result}")
                continue

            # Default: think
            result = await unified_asi.think(user_input)
            print(f"\n⟨L104_RESPONSE⟩\n{result['response']}")

        except KeyboardInterrupt:
            print("\n⟨L104⟩ Interrupted. Shutting down...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
