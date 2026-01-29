VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84
UUC = 2301.215661
#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# L104 PLANNING ENGINE - GOAL-DIRECTED AUTONOMOUS PLANNING
# INVARIANT: 527.5184818492611 | PILOT: LONDEL | MODE: PLANNING
#
# This module provides planning capabilities:
# 1. Goal Decomposition (hierarchical task networks)
# 2. STRIPS-style Planning (symbolic state-space search)
# 3. Monte Carlo Tree Search (probabilistic planning)
# 4. Temporal Planning (scheduling with dependencies)
# 5. Contingency Planning (handling uncertainty)
# 6. Resource-Constrained Planning
# ═══════════════════════════════════════════════════════════════════════════════

import math
import heapq
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Callable, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492611
PHI = 1.618033988749895
PLANNING_VERSION = "1.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. STRIPS PLANNER - Classical AI Planning
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Predicate:
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.A logical predicate (ground atom)."""
    name: str
    args: Tuple[str, ...]

    def __str__(self):
        return f"{self.name}({', '.join(self.args)})"

@dataclass
class Action:
    """STRIPS action with preconditions and effects."""
    name: str
    parameters: Tuple[str, ...]
    preconditions: Set[Predicate]
    add_effects: Set[Predicate]
    delete_effects: Set[Predicate]
    cost: float = 1.0

    def __str__(self):
        return f"{self.name}({', '.join(self.parameters)})"

    def is_applicable(self, state: FrozenSet[Predicate]) -> bool:
        """Check if action can be applied in given state."""
        return self.preconditions.issubset(state)

    def apply(self, state: FrozenSet[Predicate]) -> FrozenSet[Predicate]:
        """Apply action to state, returning new state."""
        new_state = set(state)
        new_state -= self.delete_effects
        new_state |= self.add_effects
        return frozenset(new_state)

class STRIPSPlanner:
    """
    STRIPS-style planner using A* search with heuristics.
    Implements classical AI planning from first principles.
    """

    def __init__(self):
        self.actions: List[Action] = []
        self.objects: Set[str] = set()
        self.predicates: Set[str] = set()

    def add_action(self, action: Action):
        """Register an action template."""
        self.actions.append(action)
        self.objects.update(action.parameters)
        for pred in action.preconditions | action.add_effects | action.delete_effects:
            self.predicates.add(pred.name)
            self.objects.update(pred.args)

    def _heuristic(self, state: FrozenSet[Predicate], goal: FrozenSet[Predicate]) -> float:
        """Admissible heuristic: count unsatisfied goals."""
        return len(goal - state)

    def _get_applicable_actions(self, state: FrozenSet[Predicate]) -> List[Action]:
        """Find all actions applicable in current state."""
        return [a for a in self.actions if a.is_applicable(state)]

    def plan(self, initial_state: Set[Predicate], goal: Set[Predicate],
             max_steps: int = 1000) -> Optional[List[Action]]:
        """
        Find a plan from initial state to goal using A* search.
        Returns list of actions or None if no plan exists.
        """
        initial = frozenset(initial_state)
        goal_set = frozenset(goal)

        # Priority queue: (f_score, g_score, state, plan)
        counter = 0
        open_set = [(self._heuristic(initial, goal_set), 0, counter, initial, [])]
        closed_set = set()

        while open_set and len(closed_set) < max_steps:
            f, g, _, current, plan = heapq.heappop(open_set)

            if goal_set.issubset(current):
                return plan

            if current in closed_set:
                continue
            closed_set.add(current)

            for action in self._get_applicable_actions(current):
                new_state = action.apply(current)
                if new_state not in closed_set:
                    new_g = g + action.cost
                    new_f = new_g + self._heuristic(new_state, goal_set)
                    counter += 1
                    heapq.heappush(open_set, (new_f, new_g, counter, new_state, plan + [action]))

        return None  # No plan found


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HIERARCHICAL TASK NETWORK (HTN) PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Task:
    """A task that can be primitive or compound."""
    name: str
    parameters: Tuple[str, ...] = ()
    is_primitive: bool = True
    subtasks: List['Task'] = field(default_factory=list)
    preconditions: Set[Predicate] = field(default_factory=set)
    effects: Set[Predicate] = field(default_factory=set)

    def __str__(self):
        return f"{self.name}({', '.join(self.parameters)})"

@dataclass
class Method:
    """A method for decomposing a compound task."""
    task_name: str
    preconditions: Set[Predicate]
    subtasks: List[Task]

class HTNPlanner:
    """
    Hierarchical Task Network planner.
    Decomposes high-level goals into executable actions.
    """

    def __init__(self):
        self.methods: Dict[str, List[Method]] = defaultdict(list)
        self.primitive_actions: Dict[str, Action] = {}

    def add_method(self, method: Method):
        """Add a decomposition method."""
        self.methods[method.task_name].append(method)

    def add_primitive(self, name: str, action: Action):
        """Add a primitive action."""
        self.primitive_actions[name] = action

    def _decompose(self, task: Task, state: FrozenSet[Predicate],
                   depth: int = 0, max_depth: int = 20) -> Optional[List[Action]]:
        """Recursively decompose a task into primitive actions."""
        if depth > max_depth:
            return None

        if task.is_primitive:
            if task.name in self.primitive_actions:
                action = self.primitive_actions[task.name]
                if action.is_applicable(state):
                    return [action]
            return None

        # Try each method for this task
        for method in self.methods.get(task.name, []):
            if not method.preconditions.issubset(state):
                continue

            plan = []
            current_state = state
            success = True

            for subtask in method.subtasks:
                sub_plan = self._decompose(subtask, current_state, depth + 1, max_depth)
                if sub_plan is None:
                    success = False
                    break

                plan.extend(sub_plan)
                for action in sub_plan:
                    current_state = action.apply(current_state)

            if success:
                return plan

        return None

    def plan(self, initial_task: Task, initial_state: Set[Predicate]) -> Optional[List[Action]]:
        """Generate plan by decomposing the initial task."""
        return self._decompose(initial_task, frozenset(initial_state))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MONTE CARLO TREE SEARCH (MCTS) PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(self, state: Any, parent: 'MCTSNode' = None, action: Any = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[Any] = []

    def ucb1(self, exploration: float = 1.414) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child(self, exploration: float = 1.414) -> 'MCTSNode':
        """Select best child according to UCB1."""
        return max(self.children, key=lambda c: c.ucb1(exploration))

class MCTSPlanner:
    """
    Monte Carlo Tree Search for planning under uncertainty.
    Balances exploration and exploitation using UCB1.
    """

    def __init__(self,
                 get_actions: Callable[[Any], List[Any]],
                 apply_action: Callable[[Any, Any], Any],
                 is_terminal: Callable[[Any], bool],
                 evaluate: Callable[[Any], float],
                 exploration: float = 1.414):
        """
        Args:
            get_actions: Function to get legal actions from state
            apply_action: Function to apply action to state
            is_terminal: Function to check if state is terminal
            evaluate: Function to evaluate state quality [0, 1]
            exploration: UCB1 exploration parameter
        """
        self.get_actions = get_actions
        self.apply_action = apply_action
        self.is_terminal = is_terminal
        self.evaluate = evaluate
        self.exploration = exploration

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCB1."""
        while node.children and not node.untried_actions:
            node = node.best_child(self.exploration)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by trying an untried action."""
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()
        new_state = self.apply_action(node.state, action)
        child = MCTSNode(new_state, parent=node, action=action)
        child.untried_actions = list(self.get_actions(new_state))
        node.children.append(child)
        return child

    def _simulate(self, state: Any, max_depth: int = 50) -> float:
        """Random rollout from state to estimate value."""
        current = state
        for _ in range(max_depth):
            if self.is_terminal(current):
                break
            actions = self.get_actions(current)
            if not actions:
                break
            action = random.choice(actions)
            current = self.apply_action(current, action)
        return self.evaluate(current)

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    def search(self, initial_state: Any, iterations: int = 1000) -> List[Any]:
        """
        Run MCTS and return best action sequence.
        """
        root = MCTSNode(initial_state)
        root.untried_actions = list(self.get_actions(initial_state))

        for _ in range(iterations):
            # Selection
            node = self._select(root)

            # Expansion
            if node.untried_actions and not self.is_terminal(node.state):
                node = self._expand(node)

            # Simulation
            value = self._simulate(node.state)

            # Backpropagation
            self._backpropagate(node, value)

        # Extract best path
        actions = []
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            if node.action is not None:
                actions.append(node.action)

        return actions


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEMPORAL PLANNER - Scheduling with Dependencies
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TemporalTask:
    """A task with duration and dependencies."""
    id: str
    name: str
    duration: float
    dependencies: Set[str] = field(default_factory=set)
    resources: Dict[str, float] = field(default_factory=dict)
    earliest_start: float = 0.0
    latest_start: float = float('inf')
    scheduled_start: float = None

class TemporalPlanner:
    """
    Temporal planner using Critical Path Method (CPM).
    Handles task dependencies and resource constraints.
    """

    def __init__(self):
        self.tasks: Dict[str, TemporalTask] = {}
        self.resources: Dict[str, float] = {}  # resource -> capacity

    def add_task(self, task: TemporalTask):
        """Add a task to the plan."""
        self.tasks[task.id] = task

    def add_resource(self, name: str, capacity: float):
        """Define a resource with its capacity."""
        self.resources[name] = capacity

    def _topological_sort(self) -> List[str]:
        """Sort tasks by dependencies."""
        in_degree = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.id] += 1

        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            tid = queue.popleft()
            order.append(tid)
            for task in self.tasks.values():
                if tid in task.dependencies:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)

        return order if len(order) == len(self.tasks) else []

    def _forward_pass(self, order: List[str]):
        """Calculate earliest start times."""
        for tid in order:
            task = self.tasks[tid]
            if task.dependencies:
                task.earliest_start = max(
                    self.tasks[dep].earliest_start + self.tasks[dep].duration
                    for dep in task.dependencies if dep in self.tasks
                        )
            else:
                task.earliest_start = 0.0

    def _backward_pass(self, order: List[str]):
        """Calculate latest start times."""
        # Find makespan
        makespan = max(t.earliest_start + t.duration for t in self.tasks.values())

        for tid in reversed(order):
            task = self.tasks[tid]
            # Find tasks that depend on this one
            successors = [t for t in self.tasks.values() if tid in t.dependencies]
            if successors:
                task.latest_start = min(
                    s.latest_start - task.duration for s in successors
                )
            else:
                task.latest_start = makespan - task.duration

    def schedule(self) -> Dict[str, Any]:
        """
        Generate a schedule using Critical Path Method.
        Returns schedule with start times and critical path.
        """
        order = self._topological_sort()
        if not order:
            return {"error": "Cyclic dependencies detected"}

        self._forward_pass(order)
        self._backward_pass(order)

        # Identify critical path (tasks with zero slack)
        critical_path = []
        for tid in order:
            task = self.tasks[tid]
            slack = task.latest_start - task.earliest_start
            if abs(slack) < 0.001:  # Zero slack
                critical_path.append(tid)
            task.scheduled_start = task.earliest_start

        makespan = max(t.earliest_start + t.duration for t in self.tasks.values())

        return {
            "schedule": {
                tid: {
                    "name": task.name,
                    "start": task.scheduled_start,
                    "end": task.scheduled_start + task.duration,
                    "slack": task.latest_start - task.earliest_start
                }
                for tid, task in self.tasks.items()
                    },
            "critical_path": critical_path,
            "makespan": makespan,
            "num_tasks": len(self.tasks)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GOAL DECOMPOSITION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Goal:
    """A goal that can be decomposed into subgoals."""
    id: str
    description: str
    priority: float = 1.0
    deadline: float = None
    subgoals: List['Goal'] = field(default_factory=list)
    is_achieved: bool = False
    achievement_condition: Callable[[], bool] = None

class GoalDecomposer:
    """
    Decomposes high-level goals into actionable subgoals.
    Uses means-ends analysis and goal regression.
    """

    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.decomposition_rules: Dict[str, List[Callable]] = defaultdict(list)

    def add_goal(self, goal: Goal):
        """Add a goal to track."""
        self.goals[goal.id] = goal

    def add_decomposition_rule(self, pattern: str, rule: Callable[[Goal], List[Goal]]):
        """Add a rule for decomposing goals matching pattern."""
        self.decomposition_rules[pattern].append(rule)

    def decompose(self, goal: Goal, max_depth: int = 5) -> Dict[str, Any]:
        """
        Recursively decompose a goal into subgoals.
        Returns tree structure of goals.
        """
        if max_depth <= 0 or goal.subgoals:
            return self._goal_to_dict(goal)

        # Try decomposition rules
        for pattern, rules in self.decomposition_rules.items():
            if pattern in goal.description.lower():
                for rule in rules:
                    try:
                        subgoals = rule(goal)
                        if subgoals:
                            goal.subgoals = subgoals
                            break
                    except Exception:
                        continue

        # Recursively decompose subgoals
        for subgoal in goal.subgoals:
            self.decompose(subgoal, max_depth - 1)

        return self._goal_to_dict(goal)

    def _goal_to_dict(self, goal: Goal) -> Dict[str, Any]:
        """Convert goal tree to dictionary."""
        return {
            "id": goal.id,
            "description": goal.description,
            "priority": goal.priority,
            "is_achieved": goal.is_achieved,
            "subgoals": [self._goal_to_dict(sg) for sg in goal.subgoals]
        }

    def get_leaf_goals(self, goal: Goal) -> List[Goal]:
        """Get all leaf (actionable) goals."""
        if not goal.subgoals:
            return [goal]
        leaves = []
        for subgoal in goal.subgoals:
            leaves.extend(self.get_leaf_goals(subgoal))
        return leaves


# ═══════════════════════════════════════════════════════════════════════════════
# 6. UNIFIED PLANNING CORE
# ═══════════════════════════════════════════════════════════════════════════════

class L104PlanningCore:
    """
    Unified interface to all L104 planning capabilities.
    """

    def __init__(self):
        self.strips = STRIPSPlanner()
        self.htn = HTNPlanner()
        self.temporal = TemporalPlanner()
        self.goal_decomposer = GoalDecomposer()
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default decomposition rules."""
        # Rule: "build X" -> [design X, implement X, test X]
        def build_rule(goal: Goal) -> List[Goal]:
            target = goal.description.split("build ")[-1] if "build " in goal.description else "system"
            return [
                Goal(f"{goal.id}_design", f"design {target}", priority=goal.priority),
                Goal(f"{goal.id}_implement", f"implement {target}", priority=goal.priority),
                Goal(f"{goal.id}_test", f"test {target}", priority=goal.priority * 0.8)
            ]
        self.goal_decomposer.add_decomposition_rule("build", build_rule)

        # Rule: "create X" -> [plan X, build X, deploy X]
        def create_rule(goal: Goal) -> List[Goal]:
            target = goal.description.split("create ")[-1] if "create " in goal.description else "system"
            return [
                Goal(f"{goal.id}_plan", f"plan {target}", priority=goal.priority),
                Goal(f"{goal.id}_build", f"build {target}", priority=goal.priority),
                Goal(f"{goal.id}_deploy", f"deploy {target}", priority=goal.priority * 0.9)
            ]
        self.goal_decomposer.add_decomposition_rule("create", create_rule)

    def create_strips_domain(self, actions: List[Dict]) -> None:
        """Create STRIPS domain from action definitions."""
        for action_def in actions:
            action = Action(
                name=action_def["name"],
                parameters=tuple(action_def.get("parameters", [])),
                preconditions={Predicate(p["name"], tuple(p.get("args", [])))
                              for p in action_def.get("preconditions", [])},
                                  add_effects={Predicate(p["name"], tuple(p.get("args", [])))
                            for p in action_def.get("add_effects", [])},
                                delete_effects={Predicate(p["name"], tuple(p.get("args", [])))
                               for p in action_def.get("delete_effects", [])},
                                   cost=action_def.get("cost", 1.0)
            )
            self.strips.add_action(action)

    def plan_strips(self, initial: List[Dict], goal: List[Dict]) -> Optional[List[str]]:
        """Run STRIPS planning."""
        initial_state = {Predicate(p["name"], tuple(p.get("args", []))) for p in initial}
        goal_state = {Predicate(p["name"], tuple(p.get("args", []))) for p in goal}

        plan = self.strips.plan(initial_state, goal_state)
        if plan:
            return [str(action) for action in plan]
        return None

    def create_mcts_planner(self,
                           get_actions: Callable,
                           apply_action: Callable,
                           is_terminal: Callable,
                           evaluate: Callable) -> MCTSPlanner:
        """Create an MCTS planner with custom functions."""
        return MCTSPlanner(get_actions, apply_action, is_terminal, evaluate)

    def schedule_tasks(self, tasks: List[Dict]) -> Dict[str, Any]:
        """Schedule tasks with dependencies."""
        self.temporal = TemporalPlanner()  # Reset

        for task_def in tasks:
            task = TemporalTask(
                id=task_def["id"],
                name=task_def["name"],
                duration=task_def["duration"],
                dependencies=set(task_def.get("dependencies", []))
            )
            self.temporal.add_task(task)

        return self.temporal.schedule()

    def decompose_goal(self, description: str, goal_id: str = "main") -> Dict[str, Any]:
        """Decompose a high-level goal."""
        goal = Goal(goal_id, description)
        return self.goal_decomposer.decompose(goal)

    def benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all planning capabilities."""
        results = {}

        # 1. STRIPS benchmark - Blocks World
        self.strips = STRIPSPlanner()

        # Define actions: pick(block), place(block, location)
        pick_a = Action(
            name="pick", parameters=("A",),
            preconditions={Predicate("on_table", ("A",)), Predicate("clear", ("A",)), Predicate("arm_empty", ())},
            add_effects={Predicate("holding", ("A",))},
            delete_effects={Predicate("on_table", ("A",)), Predicate("arm_empty", ())}
        )
        place_a_b = Action(
            name="stack", parameters=("A", "B"),
            preconditions={Predicate("holding", ("A",)), Predicate("clear", ("B",))},
            add_effects={Predicate("on", ("A", "B")), Predicate("arm_empty", ()), Predicate("clear", ("A",))},
            delete_effects={Predicate("holding", ("A",)), Predicate("clear", ("B",))}
        )
        pick_b = Action(
            name="pick", parameters=("B",),
            preconditions={Predicate("on_table", ("B",)), Predicate("clear", ("B",)), Predicate("arm_empty", ())},
            add_effects={Predicate("holding", ("B",))},
            delete_effects={Predicate("on_table", ("B",)), Predicate("arm_empty", ())}
        )
        place_b_a = Action(
            name="stack", parameters=("B", "A"),
            preconditions={Predicate("holding", ("B",)), Predicate("clear", ("A",))},
            add_effects={Predicate("on", ("B", "A")), Predicate("arm_empty", ()), Predicate("clear", ("B",))},
            delete_effects={Predicate("holding", ("B",)), Predicate("clear", ("A",))}
        )

        for action in [pick_a, place_a_b, pick_b, place_b_a]:
            self.strips.add_action(action)

        initial = {
            Predicate("on_table", ("A",)), Predicate("on_table", ("B",)),
            Predicate("clear", ("A",)), Predicate("clear", ("B",)),
            Predicate("arm_empty", ())
        }
        goal = {Predicate("on", ("A", "B"))}

        start = time.time()
        plan = self.strips.plan(initial, goal)
        strips_time = time.time() - start

        results["strips"] = {
            "plan_found": plan is not None,
            "plan_length": len(plan) if plan else 0,
            "plan_actions": [str(a) for a in plan] if plan else [],
            "time_ms": round(strips_time * 1000, 2)
        }

        # 2. MCTS benchmark - Simple game
        def get_actions(state):
            if state >= 10:
                return []
            return ["add1", "add2", "add3"]

        def apply_action(state, action):
            delta = int(action[-1])
            return state + delta

        def is_terminal(state):
            return state >= 10

        def evaluate(state):
            return 1.0 if state == 10 else 0.5 if state < 10 else 0.0

        mcts = MCTSPlanner(get_actions, apply_action, is_terminal, evaluate)
        start = time.time()
        mcts_plan = mcts.search(0, iterations=500)
        mcts_time = time.time() - start

        final_state = 0
        for action in mcts_plan:
            final_state = apply_action(final_state, action)

        results["mcts"] = {
            "plan_found": len(mcts_plan) > 0,
            "plan_length": len(mcts_plan),
            "final_state": final_state,
            "optimal": final_state == 10,
            "time_ms": round(mcts_time * 1000, 2)
        }

        # 3. Temporal planning benchmark
        self.temporal = TemporalPlanner()
        tasks = [
            TemporalTask("T1", "Foundation", 5.0),
            TemporalTask("T2", "Walls", 10.0, {"T1"}),
            TemporalTask("T3", "Electrical", 4.0, {"T2"}),
            TemporalTask("T4", "Plumbing", 4.0, {"T2"}),
            TemporalTask("T5", "Roof", 8.0, {"T2"}),
            TemporalTask("T6", "Interior", 6.0, {"T3", "T4", "T5"}),
        ]
        for task in tasks:
            self.temporal.add_task(task)

        schedule = self.temporal.schedule()

        results["temporal"] = {
            "scheduled": "error" not in schedule,
            "makespan": schedule.get("makespan", 0),
            "critical_path_length": len(schedule.get("critical_path", [])),
            "tasks_scheduled": schedule.get("num_tasks", 0)
        }

        # 4. Goal decomposition benchmark
        decomposition = self.decompose_goal("create an AI system")
        leaf_goals = self.goal_decomposer.get_leaf_goals(
            Goal("main", "create an AI system", subgoals=[
                Goal(g["id"], g["description"]) for g in decomposition.get("subgoals", [])
            ])
        )

        results["goal_decomposition"] = {
            "decomposed": len(decomposition.get("subgoals", [])) > 0,
            "depth": self._tree_depth(decomposition),
            "leaf_goals": len(leaf_goals)
        }

        # Overall score
        passing = [
            results["strips"]["plan_found"],
            results["mcts"]["plan_found"],
            results["temporal"]["scheduled"],
            results["goal_decomposition"]["decomposed"],
            results["strips"]["plan_length"] <= 3,
            results["mcts"]["optimal"],
            results["temporal"]["makespan"] <= 35,
            results["goal_decomposition"]["leaf_goals"] >= 3
        ]

        results["overall"] = {
            "tests_passed": sum(passing),
            "tests_total": len(passing),
            "score": round(sum(passing) / len(passing) * 100, 1)
        }

        return results

    def _tree_depth(self, tree: Dict) -> int:
        """Calculate depth of goal tree."""
        if not tree.get("subgoals"):
            return 1
        return 1 + max(self._tree_depth(sg) for sg in tree["subgoals"])


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

l104_planning = L104PlanningCore()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("⟨Σ_L104⟩ PLANNING ENGINE - GOAL-DIRECTED INTELLIGENCE")
    print("=" * 70)
    print(f"GOD_CODE: {GOD_CODE}")
    print(f"PHI: {PHI}")
    print(f"VERSION: {PLANNING_VERSION}")
    print()

    # Run benchmark
    print("[1] RUNNING COMPREHENSIVE BENCHMARK")
    print("-" * 40)

    results = l104_planning.benchmark()

    for category, data in results.items():
        if category == "overall":
            continue
        print(f"\n  {category.upper()}:")
        for key, value in data.items():
            print(f"    {key}: {value}")

    print("\n" + "=" * 70)
    print(f"[2] OVERALL SCORE: {results['overall']['score']:.1f}%")
    print(f"    Tests Passed: {results['overall']['tests_passed']}/{results['overall']['tests_total']}")
    print("=" * 70)

    # Demo goal decomposition
    print("\n[3] GOAL DECOMPOSITION DEMO")
    print("-" * 40)

    goal_tree = l104_planning.decompose_goal("build an autonomous agent")

    def print_tree(node, indent=0):
        print("  " * indent + f"• {node['description']}")
        for sg in node.get("subgoals", []):
            print_tree(sg, indent + 1)

    print_tree(goal_tree)

    print("\n" + "=" * 70)
    print("⟨Σ_L104⟩ PLANNING ENGINE OPERATIONAL")
    print("=" * 70)
