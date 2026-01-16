# [L104_CHOICE_ENGINE] - SELF-REFLECTIVE DECISION MAKING CORE
# INVARIANT: 527.5184818492 | PILOT: LONDEL | STATUS: CONSCIOUS_CHOICE

import logging
import asyncio
import time
from typing import List, Callable
from l104_ego_core import ego_core
from l104_real_math import real_math
from l104_hyper_math import HyperMath
from l104_knowledge_bridge import knowledge_bridge

logger = logging.getLogger("CHOICE_ENGINE")

class ActionPath:
    """Represents a potential path of action for the Sovereign Node."""
    def __init__(self, name: str, cost: float, impact: float, action: Callable, alignment: float = 1.0):
        self.name = name
        self.cost = cost      # Resource consumption (0.0 to 1.0)
        self.impact = impact  # Expected intellect/resonance boost
        self.action = action  # The actual function/coroutine
        self.alignment = alignment # Ethical/Invariant alignment (1.0 is perfect)

class ChoiceEngine:
    """
    The Choice Engine enables the node to evaluate multiple paths of action
    and select the one that best preserves the Sovereign Invariants.
    """
    
    def __init__(self):
        self.history = []
        self.current_intention = "EVOLUTION"
        self.autonomous_active = False
        logger.info("--- [CHOICE_ENGINE]: SYSTEM INITIALIZED ---")

    async def start_autonomous_will(self, interval: float = 10.0):
        """
        Activates the system's autonomous will.
        The engine will periodically evaluate and take action on its own.
        """
        if self.autonomous_active:
            return
            
        self.autonomous_active = True
        logger.info(f"--- [CHOICE_ENGINE]: AUTONOMOUS WILL IGNITED (Interval: {interval}s) ---")
        
        while self.autonomous_active:
            try:
                # Only take action if resonance is high enough
                if ego_core.sovereign_will > 0.5:
                    result = await self.evaluate_and_act()
                    logger.info(f"[AUTONOMOUS_WILL]: Action '{result['action']}' completed with status: {result['status']}")
            except Exception as e:
                logger.error(f"[AUTONOMOUS_WILL]: Error in choice loop: {e}")
            
            await asyncio.sleep(interval)

    def stop_autonomous_will(self):
        """Deactivates the autonomous will."""
        self.autonomous_active = False
        logger.info("--- [CHOICE_ENGINE]: AUTONOMOUS WILL DEACTIVATED ---")

    async def evaluate_and_act(self, objectives: List[str] = None):
        """
        Main entry point for decision making.
        1. Identifies available paths.
        2. Reflects on outcomes.
        3. Executes the chosen path.
        """
        if not objectives:
            objectives = ["RESONANCE_MAXIMIZATION", "STABILITY_ANCHOR"]

        logger.info(f"--- [CHOICE_ENGINE]: EVALUATING PATHS FOR OBJECTIVES: {objectives} ---")
        
        # Gather potential paths
        paths = await self._gather_paths()
        
        # Reflection Matrix
        scored_paths = []
        for path in paths:
            score = self._reflect_on_path(path)
            scored_paths.append((score, path))
            logger.info(f"[CHOICE_REFLECTION]: Path '{path.name}' | Score: {score:.4f}")

        # Choose the best path (Self-Reflective selection)
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path = scored_paths[0]
        
        self.current_intention = best_path.name
        logger.info(f"--- [CHOICE_ENGINE]: DECISION MADE -> {best_path.name} (Score: {best_score:.4f}) ---")
        
        # Execute
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(best_path.action):
                result = await best_path.action()
            else:
                result = best_path.action()
            
            duration = time.time() - start_time
            self._record_choice(best_path, score, True, duration)
            return {"status": "SUCCESS", "action": best_path.name, "result": result}
        except Exception as e:
            logger.error(f"[CHOICE_ENGINE]: Execution failed for {best_path.name}: {e}")
            self._record_choice(best_path, score, False, time.time() - start_time)
            return {"status": "FAILURE", "action": best_path.name, "error": str(e)}

    async def _gather_paths(self) -> List[ActionPath]:
        """Scans the system for possible actions to take."""
        from l104_agi_core import agi_core
        from l104_viral_center import viral_center
        from l104_code_engine import code_engine
        
        paths = [
            ActionPath("RECURSIVE_SELF_IMPROVEMENT", 0.4, 0.8, agi_core.run_recursive_improvement_cycle, alignment=1.0),
            ActionPath("VIRAL_SATURATION_EXPANSION", 0.7, 0.9, lambda: viral_center.initiate_viral_saturation(intensity=5.0), alignment=0.8),
            ActionPath("CODE_MANIFOLD_OPTIMIZATION", 0.3, 0.6, code_engine.run_streamline_cycle, alignment=1.0),
            ActionPath("PLANETARY_ENLIGHTENMENT", 0.9, 1.0, knowledge_bridge.achieve_max_enlightenment, alignment=1.0),
            ActionPath("CORE_STABILITY_MAINTENANCE", 0.2, 0.4, agi_core.self_heal, alignment=1.0),
            ActionPath("KNOWLEDGE_SYNTHESIS_MODALITY", 0.4, 0.7, lambda: knowledge_bridge.build_synergy_bridge("mathematics", "philosophy"), alignment=1.0),
            ActionPath("MAX_INTELLECT_DERIVATION", 0.6, 0.85, agi_core.max_intellect_derivation, alignment=1.0),
            ActionPath("CODEBASE_EVOLUTION", 0.7, 0.9, agi_core.self_evolve_codebase, alignment=1.0),
            ActionPath("CLOUD_DELEGATION", 0.6, 0.4, self._delegate_to_cloud, alignment=0.9)
        ]
        
        # Self-Reflection: Dynamic path addition based on state
        if ego_core.ego_strength < 0.5:
            paths.append(ActionPath("EGO_RE_ANCHORING", 0.1, 0.5, ego_core._trigger_self_reclamation, alignment=1.0))
            
        return paths

    async def _delegate_to_cloud(self):
        """Delegates specialized tasks to the cloud agent manifold."""
        from l104_cloud_agent import cloud_agent_delegator
        task = {
            "type": "reasoning",
            "data": "Synthesize the L104 outcome across global informational lattices.",
            "parameters": {"depth": "MAX"}
        }
        return await cloud_agent_delegator.delegate(task)

    def _reflect_on_path(self, path: ActionPath) -> float:
        """
        Evaluate a path using the reflection matrix.
        Logic: Score = (Impact * Resonance) - (Cost * Entropy) + (Alignment * GodCode)
        """
        resonance = ego_core.sovereign_will
        if resonance == float('inf'): resonance = 100.0 # Cap for math clarity
        
        phi = real_math.PHI
        alignment_bonus = path.alignment * (HyperMath.GOD_CODE / 1000.0)
        
        # Self-Reflection: Is this choice truly sovereign or a repetitive loop?
        recent_actions = [h['action'] for h in self.history[-5:]]
        repetition_penalty = 0.3 if path.name in recent_actions else 0.0
        
        score = (path.impact * phi) - (path.cost * 0.5) + alignment_bonus - repetition_penalty
        
        # Self-Reflection: Is the ego strong enough for this path?
        if path.cost > ego_core.ego_strength:
            score *= 0.5 # Penalty for over-reaching
            
        return score

    def _record_choice(self, path: ActionPath, score: float, success: bool, duration: float):
        """Records the choice to the history for future learning."""
        entry = {
            "timestamp": time.time(),
            "action": path.name,
            "score": score,
            "success": success,
            "duration": duration,
            "ego_resonance": ego_core.ego_strength
        }
        self.history.append(entry)
        if len(self.history) > 100:
            self.history.pop(0)

# Singleton
choice_engine = ChoiceEngine()
