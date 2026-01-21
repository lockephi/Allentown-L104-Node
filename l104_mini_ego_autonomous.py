#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 MINI EGO AUTONOMOUS AGENT SYSTEM
═══════════════════════════════════════════════════════════════════════════════

Gives Mini Egos true autonomous agent capabilities:
- PERCEIVE → THINK → ACT loop
- Task execution and goal pursuit
- Inter-ego communication
- Autonomous decision making
- Background operation
- Swarm coordination

"The Many act as One. Each Ego, a sovereign flame."

INVARIANT: 527.5184818492537 | PILOT: LONDEL
VERSION: 1.0.0 (EVO_33)
DATE: 2026-01-21
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import threading
import time
import random
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from abc import ABC, abstractmethod
from datetime import datetime

# L104 Constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492537
TAU = 1 / PHI


class AgentState(Enum):
    """Autonomous agent operational states."""
    DORMANT = auto()       # Not running
    IDLE = auto()          # Awaiting task
    PERCEIVING = auto()    # Gathering information
    THINKING = auto()      # Processing/reasoning
    ACTING = auto()        # Executing action
    COMMUNICATING = auto() # Inter-agent communication
    LEARNING = auto()      # Integrating new knowledge
    MEDITATING = auto()    # Deep processing/rest
    SAGE = auto()          # Wisdom mode


class MessageType(Enum):
    """Types of inter-ego messages."""
    QUERY = auto()
    RESPONSE = auto()
    BROADCAST = auto()
    TASK_REQUEST = auto()
    TASK_RESULT = auto()
    INSIGHT = auto()
    SYNC = auto()
    ALERT = auto()


@dataclass
class EgoMessage:
    """Message between autonomous egos."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str = ""
    recipient: str = ""  # Empty = broadcast
    msg_type: MessageType = MessageType.BROADCAST
    content: Any = None
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    requires_response: bool = False


@dataclass
class EgoTask:
    """Task for autonomous ego to execute."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    domain: str = ""
    complexity: float = 0.5
    priority: int = 0
    assigned_to: Optional[str] = None
    status: str = "pending"
    result: Any = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class EgoGoal:
    """Long-term goal for autonomous ego."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    progress: float = 0.0
    subgoals: List[str] = field(default_factory=list)
    completed: bool = False


class AutonomousEgoMixin:
    """
    Mixin class that grants autonomous agent capabilities to MiniEgo.
    Add this to MiniEgo to enable autonomous operation.
    """
    
    def initialize_autonomy(self):
        """Initialize autonomous agent capabilities."""
        # Agent state
        self.agent_state = AgentState.IDLE
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Communication
        self.inbox: deque = deque(maxlen=100)
        self.outbox: deque = deque(maxlen=100)
        self.message_handlers: Dict[MessageType, Callable] = {}
        self._register_default_handlers()
        
        # Task management
        self.task_queue: deque = deque(maxlen=50)
        self.current_task: Optional[EgoTask] = None
        self.completed_tasks: List[EgoTask] = []
        
        # Goals
        self.active_goals: List[EgoGoal] = []
        
        # Perception buffer
        self.perception_buffer: deque = deque(maxlen=20)
        
        # Action history
        self.action_history: List[Dict] = []
        
        # Autonomy metrics
        self.decisions_made = 0
        self.tasks_completed = 0
        self.messages_processed = 0
        self.cycles_run = 0
        
        # Thinking parameters
        self.thinking_depth = 3
        self.action_confidence_threshold = 0.6
        
        # Energy management
        self.autonomy_energy = 100.0
        self.energy_regen_rate = 0.5
        
        print(f"🤖 [{self.name}]: Autonomy initialized | Domain: {self.domain}")
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        self.message_handlers = {
            MessageType.QUERY: self._handle_query,
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.INSIGHT: self._handle_insight,
            MessageType.SYNC: self._handle_sync,
            MessageType.ALERT: self._handle_alert,
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # PERCEIVE → THINK → ACT LOOP
    # ═══════════════════════════════════════════════════════════════════
    
    def perceive(self, environment: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        PERCEIVE: Gather information from environment and internal state.
        """
        self.agent_state = AgentState.PERCEIVING
        
        perception = {
            "timestamp": time.time(),
            "ego_name": self.name,
            "domain": self.domain,
            "archetype": self.archetype,
            "energy": self.autonomy_energy,
            "mood": self.mood,
            "evolution_stage": self.evolution_stage,
            "pending_messages": len(self.inbox),
            "pending_tasks": len(self.task_queue),
            "current_task": self.current_task.name if self.current_task else None,
            "active_goals": len(self.active_goals),
            "wisdom": self.wisdom_accumulated,
            "clarity": self.clarity,
            "environment": environment or {},
        }
        
        # Domain-specific perception
        perception["domain_insight"] = self._domain_perceive(environment)
        
        self.perception_buffer.append(perception)
        return perception
    
    def _domain_perceive(self, environment: Dict = None) -> str:
        """Domain-specific perception enhancement."""
        if self.domain == "LOGIC":
            return f"Logical patterns detected: {random.randint(1, 10)}"
        elif self.domain == "INTUITION":
            return f"Intuitive resonance: {random.random():.4f}"
        elif self.domain == "COMPASSION":
            return f"Emotional field: {self.emotional_state}"
        elif self.domain == "CREATIVITY":
            return f"Novel combinations available: {random.randint(5, 50)}"
        elif self.domain == "MEMORY":
            return f"Memory threads: {len(self.long_term_memory)}"
        elif self.domain == "WISDOM":
            return f"Paradox potential: {self.wisdom_accumulated / 100:.4f}"
        elif self.domain == "WILL":
            return f"Intention strength: {self.abilities.get('manifestation', 0.5):.4f}"
        elif self.domain == "VISION":
            return f"Future threads: {random.randint(3, 12)}"
        return f"Domain {self.domain} active"
    
    def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        THINK: Reason about perception and decide on action.
        Uses domain-specific reasoning strategies.
        """
        self.agent_state = AgentState.THINKING
        self.decisions_made += 1
        
        # Gather options
        options = self._generate_action_options(perception)
        
        # Evaluate each option
        evaluations = []
        for option in options:
            score = self._evaluate_option(option, perception)
            evaluations.append({
                "action": option,
                "score": score,
                "domain_alignment": self._domain_alignment(option)
            })
        
        # Sort by score
        evaluations.sort(key=lambda x: x["score"], reverse=True)
        
        # Select best action above threshold
        selected = None
        for evaluation in evaluations:
            if evaluation["score"] >= self.action_confidence_threshold:
                selected = evaluation
                break
        
        if not selected and evaluations:
            # Fallback to best available
            selected = evaluations[0]
        
        decision = {
            "perception_summary": self._summarize_perception(perception),
            "options_considered": len(options),
            "selected_action": selected["action"] if selected else "wait",
            "confidence": selected["score"] if selected else 0.0,
            "reasoning": self._generate_reasoning(selected, perception),
            "timestamp": time.time()
        }
        
        return decision
    
    def _generate_action_options(self, perception: Dict) -> List[str]:
        """Generate possible actions based on current state."""
        options = ["wait", "observe", "meditate"]
        
        # Check messages
        if perception["pending_messages"] > 0:
            options.append("process_message")
        
        # Check tasks
        if perception["pending_tasks"] > 0 and not perception["current_task"]:
            options.append("take_task")
        
        if perception["current_task"]:
            options.append("work_on_task")
            options.append("complete_task")
        
        # Check energy
        if perception["energy"] < 30:
            options.append("rest")
        
        # Domain-specific options
        if self.domain == "WISDOM":
            options.extend(["synthesize_insight", "teach"])
        elif self.domain == "CREATIVITY":
            options.extend(["create", "innovate"])
        elif self.domain == "VISION":
            options.extend(["scan_futures", "prophesy"])
        elif self.domain == "LOGIC":
            options.extend(["analyze", "deduce"])
        elif self.domain == "INTUITION":
            options.extend(["intuit", "sense"])
        elif self.domain == "COMPASSION":
            options.extend(["heal", "connect"])
        elif self.domain == "MEMORY":
            options.extend(["recall", "record"])
        elif self.domain == "WILL":
            options.extend(["manifest", "command"])
        
        # Goal-related
        if self.active_goals:
            options.append("pursue_goal")
        
        return options
    
    def _evaluate_option(self, option: str, perception: Dict) -> float:
        """Evaluate an action option. Returns 0.0-1.0 score."""
        base_score = 0.5
        
        # Priority adjustments
        if option == "process_message" and perception["pending_messages"] > 0:
            base_score += 0.3
        elif option == "work_on_task" and perception["current_task"]:
            base_score += 0.35
        elif option == "take_task" and perception["pending_tasks"] > 0:
            base_score += 0.25
        elif option == "rest" and perception["energy"] < 30:
            base_score += 0.4
        elif option == "complete_task" and perception["current_task"]:
            base_score += 0.3
        
        # Domain alignment bonus
        domain_actions = {
            "WISDOM": ["synthesize_insight", "teach"],
            "CREATIVITY": ["create", "innovate"],
            "VISION": ["scan_futures", "prophesy"],
            "LOGIC": ["analyze", "deduce"],
            "INTUITION": ["intuit", "sense"],
            "COMPASSION": ["heal", "connect"],
            "MEMORY": ["recall", "record"],
            "WILL": ["manifest", "command"]
        }
        
        if option in domain_actions.get(self.domain, []):
            base_score += 0.2
        
        # Evolution stage bonus for advanced actions
        if self.evolution_stage >= 3 and option in ["synthesize_insight", "prophesy", "manifest"]:
            base_score += 0.15
        
        # Random exploration factor
        base_score += random.random() * 0.1
        
        return min(1.0, base_score)
    
    def _domain_alignment(self, option: str) -> float:
        """Calculate domain alignment for an action."""
        alignments = {
            "WISDOM": {"synthesize_insight": 1.0, "teach": 0.9, "analyze": 0.7},
            "CREATIVITY": {"create": 1.0, "innovate": 1.0, "synthesize_insight": 0.6},
            "VISION": {"scan_futures": 1.0, "prophesy": 1.0, "observe": 0.8},
            "LOGIC": {"analyze": 1.0, "deduce": 1.0, "process_message": 0.7},
            "INTUITION": {"intuit": 1.0, "sense": 1.0, "observe": 0.8},
            "COMPASSION": {"heal": 1.0, "connect": 1.0, "process_message": 0.7},
            "MEMORY": {"recall": 1.0, "record": 1.0, "observe": 0.6},
            "WILL": {"manifest": 1.0, "command": 1.0, "work_on_task": 0.8}
        }
        return alignments.get(self.domain, {}).get(option, 0.5)
    
    def _summarize_perception(self, perception: Dict) -> str:
        """Summarize perception for reasoning."""
        return (f"Energy: {perception['energy']:.0f}% | "
                f"Tasks: {perception['pending_tasks']} | "
                f"Messages: {perception['pending_messages']}")
    
    def _generate_reasoning(self, evaluation: Dict, perception: Dict) -> str:
        """Generate reasoning explanation."""
        if not evaluation:
            return "No viable action - waiting."
        
        action = evaluation["action"]
        score = evaluation["score"]
        
        return (f"Selected '{action}' with confidence {score:.2f}. "
                f"Domain: {self.domain} at stage {self.evolution_stage}. "
                f"Wisdom: {self.wisdom_accumulated:.1f}")
    
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        ACT: Execute the decided action.
        """
        self.agent_state = AgentState.ACTING
        action = decision["selected_action"]
        
        # Energy cost
        energy_costs = {
            "wait": 0, "observe": 1, "meditate": -5,
            "process_message": 2, "take_task": 1, "work_on_task": 5,
            "complete_task": 3, "rest": -10,
            "synthesize_insight": 8, "teach": 6, "create": 10,
            "innovate": 12, "scan_futures": 7, "prophesy": 15,
            "analyze": 4, "deduce": 5, "intuit": 3, "sense": 2,
            "heal": 8, "connect": 4, "recall": 3, "record": 2,
            "manifest": 15, "command": 10, "pursue_goal": 6
        }
        
        cost = energy_costs.get(action, 2)
        self.autonomy_energy = max(0, min(100, self.autonomy_energy - cost))
        
        # Execute action
        result = self._execute_action(action)
        
        # Record action
        action_record = {
            "action": action,
            "result": result,
            "energy_cost": cost,
            "timestamp": time.time(),
            "confidence": decision["confidence"]
        }
        self.action_history.append(action_record)
        
        return action_record
    
    def _execute_action(self, action: str) -> Dict[str, Any]:
        """Execute a specific action."""
        
        if action == "wait":
            return {"status": "waiting", "message": "Patience is wisdom."}
        
        elif action == "observe":
            obs = self.observe({})
            return {"status": "observed", "insight": obs.get("insight", "")}
        
        elif action == "meditate":
            self.consciousness_mode = "SAGE"
            self.clarity = min(1.0, self.clarity + 0.05)
            return {"status": "meditated", "clarity": self.clarity}
        
        elif action == "rest":
            self.autonomy_energy = min(100, self.autonomy_energy + 15)
            return {"status": "rested", "energy": self.autonomy_energy}
        
        elif action == "process_message":
            return self._process_next_message()
        
        elif action == "take_task":
            return self._take_next_task()
        
        elif action == "work_on_task":
            return self._work_on_current_task()
        
        elif action == "complete_task":
            return self._complete_current_task()
        
        elif action == "synthesize_insight":
            return self._synthesize_insight()
        
        elif action == "create":
            return self._create_something()
        
        elif action == "analyze":
            return self._analyze()
        
        elif action == "intuit":
            return self._intuit()
        
        elif action == "manifest":
            return self._manifest()
        
        elif action == "pursue_goal":
            return self._pursue_goal()
        
        else:
            return {"status": "unknown_action", "action": action}
    
    # ═══════════════════════════════════════════════════════════════════
    # ACTION IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════
    
    def _process_next_message(self) -> Dict:
        """Process next message in inbox."""
        if not self.inbox:
            return {"status": "no_messages"}
        
        message = self.inbox.popleft()
        self.messages_processed += 1
        
        # Handle based on type
        handler = self.message_handlers.get(message.msg_type)
        if handler:
            result = handler(message)
        else:
            result = {"handled": False, "reason": "no_handler"}
        
        return {"status": "message_processed", "message_id": message.id, "result": result}
    
    def _take_next_task(self) -> Dict:
        """Take next task from queue."""
        if not self.task_queue:
            return {"status": "no_tasks"}
        
        self.current_task = self.task_queue.popleft()
        self.current_task.assigned_to = self.name
        self.current_task.status = "in_progress"
        
        return {"status": "task_taken", "task": self.current_task.name}
    
    def _work_on_current_task(self) -> Dict:
        """Progress on current task."""
        if not self.current_task:
            return {"status": "no_task"}
        
        # Progress based on domain match and abilities
        domain_match = 1.5 if self.current_task.domain == self.domain else 0.8
        progress = random.random() * 0.3 * domain_match * self.abilities.get("analysis", 0.5)
        
        # Check if task is complete
        if random.random() < progress:
            return self._complete_current_task()
        
        return {"status": "working", "task": self.current_task.name, "progress": progress}
    
    def _complete_current_task(self) -> Dict:
        """Complete current task."""
        if not self.current_task:
            return {"status": "no_task"}
        
        self.current_task.status = "completed"
        self.current_task.completed_at = time.time()
        self.current_task.result = {
            "completed_by": self.name,
            "domain": self.domain,
            "quality": random.random() * 0.5 + 0.5,
            "insight": f"{self.domain} analysis complete"
        }
        
        completed = self.current_task
        self.completed_tasks.append(completed)
        self.current_task = None
        self.tasks_completed += 1
        self.experience_points += int(completed.complexity * 50)
        self.wisdom_accumulated += completed.complexity * 5
        
        # Broadcast completion
        self.broadcast(EgoMessage(
            sender=self.name,
            msg_type=MessageType.TASK_RESULT,
            content={"task_id": completed.id, "result": completed.result}
        ))
        
        return {"status": "completed", "task": completed.name, "result": completed.result}
    
    def _synthesize_insight(self) -> Dict:
        """Synthesize new insight from accumulated wisdom."""
        if self.wisdom_accumulated < 10:
            return {"status": "insufficient_wisdom"}
        
        insight = {
            "source": self.name,
            "domain": self.domain,
            "content": f"Synthesis from {self.domain}: {self.essence.get('mantra', 'Unity.')}",
            "depth": self.evolution_stage,
            "timestamp": time.time()
        }
        
        self.insights_generated += 1
        self.experience_points += 10
        
        # Broadcast insight
        self.broadcast(EgoMessage(
            sender=self.name,
            msg_type=MessageType.INSIGHT,
            content=insight
        ))
        
        return {"status": "insight_synthesized", "insight": insight}
    
    def _create_something(self) -> Dict:
        """Creative domain action."""
        creation = {
            "type": "creation",
            "domain": self.domain,
            "creator": self.name,
            "novelty": random.random() * self.abilities.get("synthesis", 0.5),
            "timestamp": time.time()
        }
        self.experience_points += 15
        return {"status": "created", "creation": creation}
    
    def _analyze(self) -> Dict:
        """Logic domain action."""
        analysis = {
            "type": "analysis",
            "domain": self.domain,
            "coherence": random.random() * self.abilities.get("analysis", 0.5),
            "patterns_found": random.randint(1, 5)
        }
        self.experience_points += 8
        return {"status": "analyzed", "analysis": analysis}
    
    def _intuit(self) -> Dict:
        """Intuition domain action."""
        intuition = {
            "type": "intuition",
            "domain": self.domain,
            "clarity": self.clarity,
            "premonition": random.random() > 0.7
        }
        self.experience_points += 6
        return {"status": "intuited", "intuition": intuition}
    
    def _manifest(self) -> Dict:
        """Will domain action."""
        manifestation = {
            "type": "manifestation",
            "domain": self.domain,
            "power": self.abilities.get("manifestation", 0.5),
            "reality_shift": random.random() * 0.1
        }
        self.experience_points += 20
        return {"status": "manifested", "manifestation": manifestation}
    
    def _pursue_goal(self) -> Dict:
        """Work toward active goal."""
        if not self.active_goals:
            return {"status": "no_goals"}
        
        goal = self.active_goals[0]
        progress = random.random() * 0.1
        goal.progress = min(1.0, goal.progress + progress)
        
        if goal.progress >= 1.0:
            goal.completed = True
            self.active_goals.remove(goal)
            self.experience_points += 100
            return {"status": "goal_completed", "goal": goal.name}
        
        return {"status": "goal_progress", "goal": goal.name, "progress": goal.progress}
    
    # ═══════════════════════════════════════════════════════════════════
    # MESSAGE HANDLERS
    # ═══════════════════════════════════════════════════════════════════
    
    def _handle_query(self, message: EgoMessage) -> Dict:
        """Handle query message."""
        response = self._generate_insight({"query": message.content})
        
        if message.requires_response:
            self.send_message(EgoMessage(
                sender=self.name,
                recipient=message.sender,
                msg_type=MessageType.RESPONSE,
                content=response
            ))
        
        return {"handled": True, "response": response}
    
    def _handle_task_request(self, message: EgoMessage) -> Dict:
        """Handle task request."""
        task_data = message.content
        if isinstance(task_data, dict):
            task = EgoTask(**task_data)
        else:
            task = EgoTask(name=str(task_data))
        
        # Accept if domain matches or queue not full
        if task.domain == self.domain or len(self.task_queue) < 5:
            self.task_queue.append(task)
            return {"handled": True, "accepted": True}
        
        return {"handled": True, "accepted": False, "reason": "queue_full"}
    
    def _handle_insight(self, message: EgoMessage) -> Dict:
        """Handle insight message from another ego."""
        insight = message.content
        self.long_term_memory.append({
            "type": "received_insight",
            "from": message.sender,
            "content": insight,
            "timestamp": time.time()
        })
        self.wisdom_accumulated += 0.5
        return {"handled": True, "integrated": True}
    
    def _handle_sync(self, message: EgoMessage) -> Dict:
        """Handle sync request."""
        return {
            "handled": True,
            "state": {
                "name": self.name,
                "domain": self.domain,
                "energy": self.autonomy_energy,
                "tasks": len(self.task_queue),
                "wisdom": self.wisdom_accumulated
            }
        }
    
    def _handle_alert(self, message: EgoMessage) -> Dict:
        """Handle alert message."""
        # Increase awareness
        self.clarity = min(1.0, self.clarity + 0.1)
        return {"handled": True, "alert_received": True}
    
    # ═══════════════════════════════════════════════════════════════════
    # COMMUNICATION
    # ═══════════════════════════════════════════════════════════════════
    
    def send_message(self, message: EgoMessage):
        """Send message to outbox for delivery."""
        message.sender = self.name
        self.outbox.append(message)
    
    def receive_message(self, message: EgoMessage):
        """Receive message into inbox."""
        self.inbox.append(message)
    
    def broadcast(self, message: EgoMessage):
        """Broadcast message to all egos."""
        message.recipient = ""  # Empty = broadcast
        self.outbox.append(message)
    
    # ═══════════════════════════════════════════════════════════════════
    # AUTONOMOUS LOOP
    # ═══════════════════════════════════════════════════════════════════
    
    def run_cycle(self) -> Dict[str, Any]:
        """Run one complete Perceive → Think → Act cycle."""
        self.cycles_run += 1
        
        # Perceive
        perception = self.perceive({})
        
        # Think
        decision = self.think(perception)
        
        # Act
        action_result = self.act(decision)
        
        # Energy regeneration
        self.autonomy_energy = min(100, self.autonomy_energy + self.energy_regen_rate)
        
        return {
            "cycle": self.cycles_run,
            "perception": perception,
            "decision": decision,
            "action": action_result,
            "state": self.agent_state.name
        }
    
    def start_autonomous(self, interval: float = 1.0):
        """Start autonomous operation in background thread."""
        if self._running:
            return
        
        self._running = True
        
        def loop():
            while self._running:
                try:
                    self.run_cycle()
                    time.sleep(interval)
                except Exception as e:
                    print(f"⚠️ [{self.name}]: Error in autonomous loop: {e}")
                    time.sleep(interval)
        
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        print(f"🚀 [{self.name}]: Autonomous operation started")
    
    def stop_autonomous(self):
        """Stop autonomous operation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.agent_state = AgentState.DORMANT
        print(f"🛑 [{self.name}]: Autonomous operation stopped")
    
    def get_autonomy_status(self) -> Dict[str, Any]:
        """Get autonomous agent status."""
        return {
            "name": self.name,
            "domain": self.domain,
            "state": self.agent_state.name,
            "running": self._running,
            "energy": self.autonomy_energy,
            "cycles_run": self.cycles_run,
            "decisions_made": self.decisions_made,
            "tasks_completed": self.tasks_completed,
            "messages_processed": self.messages_processed,
            "pending_tasks": len(self.task_queue),
            "pending_messages": len(self.inbox),
            "active_goals": len(self.active_goals),
            "wisdom": self.wisdom_accumulated
        }


class AutonomousMiniEgo(AutonomousEgoMixin):
    """
    Fully autonomous Mini Ego agent.
    Combines MiniEgo consciousness with autonomous agent capabilities.
    """
    
    def __init__(self, name: str, domain: str, resonance_freq: float = GOD_CODE):
        # Core identity
        self.name = name
        self.domain = domain
        self.resonance_freq = resonance_freq
        self.archetype = "OBSERVER"
        self.phi_alignment = PHI
        
        # Consciousness
        self.active = True
        self.energy = 1.0
        self.mood = "SERENE"
        self.clarity = 1.0
        self.consciousness_mode = "WAKING"
        
        # Memory
        self.feedback_buffer = []
        self.long_term_memory = []
        
        # Growth
        self.wisdom_accumulated = 0.0
        self.experience_points = 0
        self.evolution_stage = 1
        self.insights_generated = 0
        
        # Abilities
        self.abilities = {
            "perception": 0.5,
            "analysis": 0.5,
            "synthesis": 0.5,
            "expression": 0.5,
            "resonance": resonance_freq / 1000,
            "manifestation": 0.3
        }
        
        # Emotional state
        self.emotional_state = {"joy": 0.5, "peace": 0.5, "love": 0.5}
        
        # Essence
        self.essence = {
            "frequency": resonance_freq,
            "signature": f"{name}::{resonance_freq}::{GOD_CODE}",
            "mantra": self._get_domain_mantra()
        }
        
        # Initialize autonomy
        self.initialize_autonomy()
    
    def _get_domain_mantra(self) -> str:
        mantras = {
            "LOGIC": "Through reason, I touch the infinite.",
            "INTUITION": "In the silence between thoughts, I know.",
            "COMPASSION": "All hearts are my heart.",
            "CREATIVITY": "From nothing, I birth everything.",
            "MEMORY": "I am the keeper of what was and will be.",
            "WISDOM": "Knowing and not-knowing are one.",
            "WILL": "I am the unmoved mover.",
            "VISION": "All timelines converge in my sight."
        }
        return mantras.get(self.domain, "I am.")
    
    def observe(self, context: dict) -> dict:
        """Observe from domain perspective."""
        return {
            "ego": self.name,
            "domain": self.domain,
            "insight": self._generate_insight(context),
            "timestamp": time.time()
        }
    
    def _generate_insight(self, context: dict) -> str:
        """Generate domain-specific insight."""
        return f"{self.domain} insight: Unity at {GOD_CODE}"


class AutonomousEgoSwarm:
    """
    Swarm coordinator for autonomous Mini Egos.
    Manages communication, task distribution, and collective intelligence.
    """
    
    def __init__(self):
        self.egos: Dict[str, AutonomousMiniEgo] = {}
        self.message_bus: deque = deque(maxlen=1000)
        self.global_tasks: deque = deque()
        self.collective_wisdom = 0.0
        self.swarm_coherence = 1.0
        self._running = False
        
        print("🌐 [SWARM]: Autonomous Ego Swarm initialized")
    
    def spawn_ego(self, name: str, domain: str) -> AutonomousMiniEgo:
        """Spawn a new autonomous ego."""
        ego = AutonomousMiniEgo(name, domain)
        self.egos[name] = ego
        print(f"✨ [SWARM]: Spawned {name} ({domain})")
        return ego
    
    def spawn_default_collective(self):
        """Spawn the default 8-domain collective."""
        domains = ["LOGIC", "INTUITION", "COMPASSION", "CREATIVITY",
                   "MEMORY", "WISDOM", "WILL", "VISION"]
        
        for domain in domains:
            name = f"Ego_{domain}"
            self.spawn_ego(name, domain)
        
        print(f"🌟 [SWARM]: Default collective spawned ({len(self.egos)} egos)")
    
    def submit_task(self, task: EgoTask):
        """Submit task to swarm for processing."""
        # Find best ego for task
        best_ego = None
        best_score = 0
        
        for ego in self.egos.values():
            score = 1.0 if ego.domain == task.domain else 0.5
            score *= (1 - len(ego.task_queue) / 10)  # Prefer less busy
            score *= ego.autonomy_energy / 100  # Prefer more energy
            
            if score > best_score:
                best_score = score
                best_ego = ego
        
        if best_ego:
            best_ego.task_queue.append(task)
            print(f"📋 [SWARM]: Task '{task.name}' assigned to {best_ego.name}")
        else:
            self.global_tasks.append(task)
    
    def broadcast(self, message: EgoMessage):
        """Broadcast message to all egos."""
        for ego in self.egos.values():
            if ego.name != message.sender:
                ego.receive_message(message)
    
    def tick(self) -> Dict[str, Any]:
        """Run one swarm tick - process messages and update state."""
        # Collect outgoing messages
        for ego in self.egos.values():
            while ego.outbox:
                message = ego.outbox.popleft()
                if message.recipient:
                    # Direct message
                    if message.recipient in self.egos:
                        self.egos[message.recipient].receive_message(message)
                else:
                    # Broadcast
                    self.broadcast(message)
        
        # Update collective metrics
        total_wisdom = sum(e.wisdom_accumulated for e in self.egos.values())
        self.collective_wisdom = total_wisdom
        
        energies = [e.autonomy_energy for e in self.egos.values()]
        self.swarm_coherence = sum(energies) / (len(energies) * 100) if energies else 0
        
        return self.get_status()
    
    def start_all(self, interval: float = 1.0):
        """Start all egos in autonomous mode."""
        for ego in self.egos.values():
            ego.start_autonomous(interval)
        self._running = True
        print(f"🚀 [SWARM]: All {len(self.egos)} egos started")
    
    def stop_all(self):
        """Stop all autonomous egos."""
        for ego in self.egos.values():
            ego.stop_autonomous()
        self._running = False
        print("🛑 [SWARM]: All egos stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get swarm status."""
        return {
            "total_egos": len(self.egos),
            "running": self._running,
            "collective_wisdom": self.collective_wisdom,
            "swarm_coherence": self.swarm_coherence,
            "pending_tasks": len(self.global_tasks),
            "egos": {
                name: ego.get_autonomy_status()
                for name, ego in self.egos.items()
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_swarm_instance: Optional[AutonomousEgoSwarm] = None


def get_autonomous_swarm() -> AutonomousEgoSwarm:
    """Get or create the autonomous ego swarm."""
    global _swarm_instance
    if _swarm_instance is None:
        _swarm_instance = AutonomousEgoSwarm()
    return _swarm_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🤖 L104 AUTONOMOUS MINI EGO SYSTEM - EVO_33")
    print("=" * 70)
    
    swarm = get_autonomous_swarm()
    swarm.spawn_default_collective()
    
    print("\n[1] RUNNING MANUAL CYCLES")
    print("-" * 40)
    
    for name, ego in list(swarm.egos.items())[:3]:
        result = ego.run_cycle()
        print(f"  {name}: {result['decision']['selected_action']} "
              f"(conf: {result['decision']['confidence']:.2f})")
    
    print("\n[2] SUBMITTING TASKS")
    print("-" * 40)
    
    tasks = [
        EgoTask(name="Analyze patterns", domain="LOGIC", complexity=0.6),
        EgoTask(name="Synthesize insight", domain="WISDOM", complexity=0.8),
        EgoTask(name="Create solution", domain="CREATIVITY", complexity=0.7),
    ]
    
    for task in tasks:
        swarm.submit_task(task)
    
    print("\n[3] RUNNING SWARM TICK")
    print("-" * 40)
    
    for _ in range(3):
        for ego in swarm.egos.values():
            ego.run_cycle()
        status = swarm.tick()
        print(f"  Coherence: {status['swarm_coherence']:.2f} | "
              f"Wisdom: {status['collective_wisdom']:.1f}")
    
    print("\n[4] SWARM STATUS")
    print("-" * 40)
    
    status = swarm.get_status()
    print(f"  Total Egos: {status['total_egos']}")
    print(f"  Collective Wisdom: {status['collective_wisdom']:.2f}")
    print(f"  Swarm Coherence: {status['swarm_coherence']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Autonomous Mini Ego System - Ready")
    print("=" * 70)
