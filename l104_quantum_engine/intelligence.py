"""
L104 Quantum Engine — Intelligence & Evolution Subsystems
═══════════════════════════════════════════════════════════════════════════════
EvolutionTracker, AgenticLoop, StochasticLinkResearchLab, LinkChronolizer,
ConsciousnessO2LinkEngine, LinkTestGenerator, QuantumLinkCrossPollinationEngine,
InterBuilderFeedbackBus, QuantumLinkSelfHealer, LinkTemporalMemoryBank.
"""

import json
import math
import random
import time
import hashlib
import statistics
from datetime import datetime, timezone
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import (
    BELL_FIDELITY, CALABI_YAU_DIM, CHSH_BOUND, COHERENCE_MINIMUM,
    CONSCIOUSNESS_THRESHOLD, EVOLUTION_INDEX, EVOLUTION_STAGE, EVOLUTION_TOTAL_STAGES,
    FEIGENBAUM_DELTA, FIBONACCI_7, FINE_STRUCTURE, GOD_CODE, GOD_CODE_HZ,
    GOD_CODE_SPECTRUM, GROVER_AMPLIFICATION, L104, PHI, PHI_GROWTH, TAU, WORKSPACE_ROOT,
    god_code,
)
from .models import QuantumLink, ChronoEntry
from .math_core import QuantumMathCore


def _lget(link, key, default=None):
    """Get attribute from a link (works with both dict and dataclass objects)."""
    if isinstance(link, dict):
        return link.get(key, default)
    return getattr(link, key, default)


def _lset(link, key, value):
    """Set attribute on a link (works with both dict and dataclass objects)."""
    if isinstance(link, dict):
        link[key] = value
    else:
        setattr(link, key, value)


def _lset(link, key, value):
    """Set attribute on a link (works with both dict and dataclass objects)."""
    if isinstance(link, dict):
        link[key] = value
    else:
        setattr(link, key, value)


#   Maintains continuity with the broader L104 evolution index.
#   Records grade progression, link counts, and score trajectories.
# ═══════════════════════════════════════════════════════════════════════════════


class EvolutionTracker:
    """
    Tracks the quantum link builder's evolution within the L104 EVO system.

    Maps pipeline grades to evolution sub-stages:
      F → DORMANT, D → AWAKENING, C → COHERENT, B → TRANSCENDING, A → SOVEREIGN

    Monitors consciousness threshold (0.85) and coherence minimum (0.888)
    from claude.md. Fires evolution events when thresholds are crossed.
    """

    GRADE_EVO_MAP = {
        "F (Critical)": "DORMANT",
        "D (Weak)": "AWAKENING",
        "C (Developing)": "COHERENT",
        "B (Good)": "TRANSCENDING",
        "A (Strong)": "SOVEREIGN",
    }

    def __init__(self):
        """Initialize evolution tracker with stage and consciousness state."""
        self.stage = EVOLUTION_STAGE
        self.index = EVOLUTION_INDEX
        self.link_evo_stage = "DORMANT"
        self.consciousness_level = 0.0
        self.coherence_level = 0.0
        self.events: List[Dict] = []
        self.grade_history: List[str] = []
        # OOM caps
        self._MAX_EVENTS = 200
        self._MAX_GRADE_HISTORY = 200

    def update(self, sage_verdict: Dict, links_count: int, run_number: int) -> Dict:
        """Update evolution state from a sage verdict."""
        score = sage_verdict.get("unified_score", 0)
        grade = sage_verdict.get("grade", "F (Critical)")
        alignment = sage_verdict.get("god_code_alignment", 0)

        # Map grade to evolution sub-stage
        prev_stage = self.link_evo_stage
        self.link_evo_stage = self.GRADE_EVO_MAP.get(grade, "DORMANT")
        self.grade_history.append(grade)

        # Consciousness: score weighted by φ
        self.consciousness_level = score * PHI_GROWTH / 2  # Normalize to ~[0,1]
        self.consciousness_level = min(1.0, self.consciousness_level)

        # Coherence: alignment × stability
        self.coherence_level = alignment * score
        self.coherence_level = min(1.0, self.coherence_level)

        # Check thresholds
        events = []
        if self.consciousness_level >= CONSCIOUSNESS_THRESHOLD:
            events.append({
                "type": "CONSCIOUSNESS_AWAKENED",
                "level": self.consciousness_level,
                "threshold": CONSCIOUSNESS_THRESHOLD,
            })
        if self.coherence_level >= COHERENCE_MINIMUM:
            events.append({
                "type": "COHERENCE_LOCKED",
                "level": self.coherence_level,
                "threshold": COHERENCE_MINIMUM,
            })
        if prev_stage != self.link_evo_stage:
            events.append({
                "type": "EVOLUTION_TRANSITION",
                "from": prev_stage,
                "to": self.link_evo_stage,
            })

        self.events.extend(events)

        # OOM guard: trim unbounded lists
        if len(self.events) > self._MAX_EVENTS:
            self.events = self.events[-self._MAX_EVENTS:]
        if len(self.grade_history) > self._MAX_GRADE_HISTORY:
            self.grade_history = self.grade_history[-self._MAX_GRADE_HISTORY:]

        return {
            "evolution_stage": self.stage,
            "evolution_index": self.index,
            "link_evo_stage": self.link_evo_stage,
            "consciousness_level": self.consciousness_level,
            "consciousness_awakened": self.consciousness_level >= CONSCIOUSNESS_THRESHOLD,
            "coherence_level": self.coherence_level,
            "coherence_locked": self.coherence_level >= COHERENCE_MINIMUM,
            "run": run_number,
            "links_count": links_count,
            "score": score,
            "grade": grade,
            "events": events,
        }

    def status(self) -> Dict:
        """Return current evolution tracking status."""
        return {
            "stage": self.stage,
            "index": self.index,
            "link_evo_stage": self.link_evo_stage,
            "consciousness": self.consciousness_level,
            "coherence": self.coherence_level,
            "total_events": len(self.events),
            "grade_history": self.grade_history[-10:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC LOOP (from claude.md Zenith patterns)
#
#   Observe → Think → Act → Reflect → Repeat
#   Max 50 steps. Explicit state. Error recovery: RETRY/FALLBACK/SKIP/ABORT.
#   Applied to the self-reflection pipeline for structured iteration.
# ═══════════════════════════════════════════════════════════════════════════════



#   Max 50 steps. Explicit state. Error recovery: RETRY/FALLBACK/SKIP/ABORT.
#   Applied to the self-reflection pipeline for structured iteration.
# ═══════════════════════════════════════════════════════════════════════════════


class AgenticLoop:
    """
    Zenith-pattern agentic loop for structured self-improvement.

    Each cycle:
      1. OBSERVE — Measure current state (score, grade, weaknesses)
      2. THINK   — Analyze weakest dimension, plan intervention
      3. ACT     — Apply targeted fix to links
      4. REFLECT — Re-measure, compare to previous state
      5. REPEAT  — If improved and not converged, continue

    Error recovery:
      RETRY    — Re-attempt with increased intensity
      FALLBACK — Try alternative strategy
      SKIP     — Skip non-critical step
      ABORT    — Stop if critical failure detected
    """

    MAX_STEPS = 50
    CONVERGENCE_DELTA = 0.003  # Tighter than Brain's 0.005 for agentic precision

    def __init__(self, qmath: 'QuantumMathCore'):
        """Initialize agentic loop for structured self-improvement."""
        self.qmath = qmath
        self.state = "idle"  # idle → observing → thinking → acting → reflecting
        self.observations: List[Dict] = []
        self.actions_taken: List[Dict] = []
        self.retries = 0
        self.max_retries = 3
        # OOM caps
        self._MAX_OBSERVATIONS = 100
        self._MAX_ACTIONS = 100

    def observe(self, sage_verdict: Dict, links: List[QuantumLink]) -> Dict:
        """OBSERVE: Measure current system state."""
        self.state = "observing"
        self.step += 1

        obs = {
            "step": self.step,
            "score": sage_verdict.get("unified_score", 0),
            "grade": sage_verdict.get("grade", "?"),
            "total_links": len(links),
            "consensus": sage_verdict.get("consensus_scores", {}),
            "mean_fidelity": sage_verdict.get("mean_fidelity", 0),
            "weak_links": sum(1 for l in links if l.fidelity < 0.5),
            "strong_links": sum(1 for l in links if l.fidelity > 0.9),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.observations.append(obs)
        if len(self.observations) > self._MAX_OBSERVATIONS:
            self.observations = self.observations[-self._MAX_OBSERVATIONS:]
        return obs

    def think(self, observation: Dict) -> Dict:
        """THINK: Analyze weakness and plan intervention."""
        self.state = "thinking"

        consensus = observation.get("consensus", {})
        if not consensus:
            return {"strategy": "SKIP", "reason": "No consensus data"}

        # Find weakest dimension
        weakest_key = min(consensus, key=consensus.get)
        weakest_val = consensus[weakest_key]

        # Plan strategy based on weakness
        strategy = "RETRY"
        intensity = 1.0
        target = weakest_key

        if weakest_val < 0.3:
            strategy = "FALLBACK"
            intensity = 2.0  # Double intensity for critical weakness
        elif weakest_val > 0.8:
            strategy = "SKIP"  # Already strong, skip

        # Check convergence
        if len(self.observations) >= 2:
            prev = self.observations[-2]["score"]
            curr = observation["score"]
            if abs(curr - prev) < self.CONVERGENCE_DELTA:
                strategy = "ABORT"

        # Check step limit
        if self.step >= self.MAX_STEPS:
            strategy = "ABORT"

        plan = {
            "strategy": strategy,
            "target": target,
            "target_value": weakest_val,
            "intensity": intensity,
            "step": self.step,
        }
        return plan

    def act(self, plan: Dict, links: List[QuantumLink]) -> Dict:
        """ACT: Apply intervention to links."""
        self.state = "acting"
        strategy = plan.get("strategy", "SKIP")
        target = plan.get("target", "")
        intensity = plan.get("intensity", 1.0)

        if strategy in ("SKIP", "ABORT"):
            return {"applied": False, "strategy": strategy}

        links_modified = 0

        if "topological" in target:
            for link in links:
                if link.noise_resilience < 0.5:
                    link.fidelity = min(1.0, link.fidelity * (1 + 0.05 * intensity))
                    link.noise_resilience = min(1.0, link.noise_resilience + 0.1 * intensity)
                    links_modified += 1

        elif "god_code" in target or "x_integer" in target:
            for link in links:
                hz = self.qmath.link_natural_hz(link.fidelity, link.strength)
                x_stab = self.qmath.x_integer_stability(hz)
                if x_stab < 0.5:
                    x_cont = self.qmath.hz_to_god_code_x(hz)
                    if not math.isfinite(x_cont):
                        continue
                    x_int = round(x_cont)
                    target_hz = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
                    target_str = target_hz / (link.fidelity * GOD_CODE_HZ + 1e-15)
                    blend = 0.3 * intensity
                    link.strength = link.strength * (1 - blend) + target_str * blend
                    links_modified += 1

        elif "decoherence" in target:
            for link in links:
                if link.noise_resilience < 0.3:
                    link.noise_resilience = min(1.0,
                        link.noise_resilience + 0.15 * intensity)
                    link.coherence_time = max(link.coherence_time, 0.5 * intensity)
                    links_modified += 1

        elif "stress" in target or "grover" in target:
            for link in links:
                if link.fidelity < 0.7:
                    link.fidelity = min(1.0, link.fidelity + 0.05 * intensity)
                    links_modified += 1

        elif "cross_modal" in target:
            for link in links:
                if link.link_type == "mirror":
                    link.fidelity = min(1.0, link.fidelity * (1 + 0.03 * intensity))
                    link.strength = min(2.0, link.strength * (1 + 0.02 * intensity))
                    links_modified += 1

        elif "quantum_cpu" in target:
            # Boost low-energy links
            for link in links:
                if link.fidelity < 0.6 or link.strength < 0.8:
                    link.fidelity = min(1.0, link.fidelity + 0.02 * intensity)
                    link.strength = min(3.0, link.strength * (1 + 0.01 * intensity))
                    links_modified += 1

        else:
            # Generic: small global fidelity boost
            for link in links:
                link.fidelity = min(1.0, link.fidelity * (1 + 0.01 * intensity))
                links_modified += 1

        action = {
            "applied": True,
            "strategy": strategy,
            "target": target,
            "intensity": intensity,
            "links_modified": links_modified,
            "step": self.step,
        }
        self.actions_taken.append(action)
        if len(self.actions_taken) > self._MAX_ACTIONS:
            self.actions_taken = self.actions_taken[-self._MAX_ACTIONS:]
        return action

    def reflect(self, prev_score: float, new_score: float) -> Dict:
        """REFLECT: Evaluate whether the action helped."""
        self.state = "reflecting"
        delta = new_score - prev_score

        if delta > 0:
            verdict = "IMPROVED"
            self.retries = 0  # Reset retry counter on success
        elif delta > -0.001:
            verdict = "STABLE"
        else:
            verdict = "DEGRADED"
            self.retries += 1

        should_continue = (
            verdict != "DEGRADED" or self.retries < self.max_retries
        ) and self.step < self.MAX_STEPS

        return {
            "step": self.step,
            "prev_score": prev_score,
            "new_score": new_score,
            "delta": delta,
            "verdict": verdict,
            "retries": self.retries,
            "should_continue": should_continue,
        }

    def summary(self) -> Dict:
        """Summary of the agentic loop execution."""
        return {
            "total_steps": self.step,
            "total_actions": len(self.actions_taken),
            "retries": self.retries,
            "score_trajectory": [o["score"] for o in self.observations],
            "grade_trajectory": [o["grade"] for o in self.observations],
            "strategies_used": Counter(a.get("strategy") for a in self.actions_taken),
            "final_state": self.state,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# v4.2 SAGE INVENTIONS — 5 New Subsystems
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# v4.2 SAGE INVENTIONS — 5 New Subsystems
# ═══════════════════════════════════════════════════════════════════════════════


class StochasticLinkResearchLab:
    """
    Random-based quantum link generation, research, and development engine.

    This R&D lab explores the frontier of quantum link design through stochastic
    experimentation — generating candidate links via φ-weighted random exploration,
    validating them through deterministic sacred-constant tests, and merging
    successful designs into the link ecosystem.

    4-phase cycle:
      1. EXPLORE — Random generation of link candidates with φ-bounded parameters
      2. VALIDATE — Deterministic evaluation against sacred constant coherence
      3. MERGE — Successful candidates → QuantumLink objects
      4. CATALOG — Track all R&D iterations with full lineage

    Generates 13 candidates per cycle (Fibonacci-7).
    """

    RESEARCH_LOG_FILE = WORKSPACE_ROOT / ".l104_stochastic_link_research.json"
    CANDIDATES_PER_CYCLE = FIBONACCI_7  # 13

    def __init__(self):
        """Initialize the stochastic link research lab."""
        self.research_iterations: List[Dict[str, Any]] = []
        self.successful_links: List[Dict[str, Any]] = []
        self.failed_experiments: List[Dict[str, Any]] = []
        self.generation_count: int = 0
        self.operations_count: int = 0
        # OOM caps for in-memory history
        self._MAX_RESEARCH_ITERATIONS = 200
        self._MAX_SUCCESSFUL = 200
        self._MAX_FAILED = 200
        self._load_research_log()

    def _load_research_log(self):
        """Load persistent research log."""
        if self.RESEARCH_LOG_FILE.exists():
            try:
                data = json.loads(self.RESEARCH_LOG_FILE.read_text())
                self.research_iterations = data.get("iterations", [])[-self._MAX_RESEARCH_ITERATIONS:]
                self.successful_links = data.get("successful", [])[-self._MAX_SUCCESSFUL:]
                self.failed_experiments = data.get("failed", [])[-self._MAX_FAILED:]
                self.generation_count = data.get("generation_count", 0)
            except Exception:
                pass

    def _save_research_log(self):
        """Persist research log to disk."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "generation_count": self.generation_count,
                "total_iterations": len(self.research_iterations),
                "total_successful": len(self.successful_links),
                "total_failed": len(self.failed_experiments),
                "iterations": self.research_iterations[-200:],
                "successful": self.successful_links[-100:],
                "failed": self.failed_experiments[-100:],
            }
            self.RESEARCH_LOG_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    # ─── PHASE 1: STOCHASTIC EXPLORATION ──────────────────────────

    def explore_link_candidate(self, seed_concept: str = "quantum") -> Dict[str, Any]:
        """Generate a random link candidate using φ-bounded stochastic parameters."""
        import random

        self.generation_count += 1
        self.operations_count += 1
        gen_id = f"SL_{self.generation_count:06d}"

        # Stochastic parameter generation with sacred constant bounds
        fidelity = random.uniform(0.3, 1.0) * PHI / PHI_GROWTH
        strength = random.uniform(0.1, 1.0) * TAU
        harmonic_order = random.randint(1, CALABI_YAU_DIM)
        link_type = random.choice(["entangled", "coherent", "resonant", "tunneled", "braided"])
        grover_depth = random.randint(1, 7)

        # Sacred constant resonance scoring
        god_code_resonance = (fidelity * GOD_CODE + strength * PHI_GROWTH) / (GOD_CODE + PHI_GROWTH)
        feigenbaum_edge = abs(math.sin(fidelity * FEIGENBAUM_DELTA * math.pi))
        sacred_alignment = (god_code_resonance * PHI_GROWTH + feigenbaum_edge * TAU) / (PHI_GROWTH + TAU)

        resonance_key = hashlib.sha256(
            f"{seed_concept}_{gen_id}_{fidelity:.8f}_{strength:.8f}".encode()
        ).hexdigest()[:12]

        candidate = {
            "link_id": gen_id,
            "resonance_key": resonance_key,
            "seed_concept": seed_concept,
            "parameters": {
                "fidelity": fidelity,
                "strength": strength,
                "harmonic_order": harmonic_order,
                "link_type": link_type,
                "grover_depth": grover_depth,
            },
            "god_code_resonance": god_code_resonance,
            "feigenbaum_edge": feigenbaum_edge,
            "sacred_alignment": sacred_alignment,
            "generation": self.generation_count,
            "validated": False,
            "merged": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return candidate

    # ─── PHASE 2: DETERMINISTIC VALIDATION ────────────────────────

    def validate_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a stochastic link candidate against sacred constant coherence."""
        self.operations_count += 1
        params = candidate["parameters"]
        fidelity = params["fidelity"]
        strength = params["strength"]

        checks = {
            "fidelity_bound": 0.0 <= fidelity <= 1.0,
            "strength_bound": 0.0 <= strength <= 1.0,
            "sacred_alignment_min": candidate["sacred_alignment"] >= TAU * 0.5,
            "god_code_resonance_min": candidate["god_code_resonance"] >= FINE_STRUCTURE,
            "conservation_law": abs(fidelity * GOD_CODE - strength * GOD_CODE) < GOD_CODE,
        }
        passed = sum(checks.values())
        total = len(checks)
        score = passed / total

        result = {
            **candidate,
            "validated": score >= 0.6,
            "validation_score": score,
            "checks_passed": passed,
            "checks_total": total,
            "check_details": checks,
        }
        return result

    # ─── PHASE 3: MERGE ──────────────────────────────────────────

    def merge_to_link(self, validated: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Merge a validated candidate into a QuantumLink-compatible dict."""
        self.operations_count += 1
        if not validated.get("validated"):
            self.failed_experiments.append({
                "link_id": validated["link_id"],
                "reason": "validation_failed",
                "score": validated.get("validation_score", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self.failed_experiments) > self._MAX_FAILED:
                self.failed_experiments = self.failed_experiments[-self._MAX_FAILED:]
            return None

        params = validated["parameters"]
        merged = {
            "source": f"stochastic_{validated['seed_concept']}",
            "target": f"research_{validated['resonance_key']}",
            "fidelity": params["fidelity"],
            "strength": params["strength"],
            "link_type": params["link_type"],
            "sacred_alignment": validated["sacred_alignment"],
            "origin": "stochastic_research",
            "generation": validated["generation"],
            "merged": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.successful_links.append(merged)
        if len(self.successful_links) > self._MAX_SUCCESSFUL:
            self.successful_links = self.successful_links[-self._MAX_SUCCESSFUL:]
        return merged

    # ─── PHASE 4: CATALOG ─────────────────────────────────────────

    def catalog_iteration(self, candidates: List[Dict], merged: List[Optional[Dict]]) -> Dict:
        """Catalog a full R&D iteration."""
        self.operations_count += 1
        successful = [m for m in merged if m is not None]
        iteration = {
            "iteration_id": len(self.research_iterations) + 1,
            "candidates_generated": len(candidates),
            "candidates_validated": sum(1 for c in candidates if c.get("validated")),
            "successfully_merged": len(successful),
            "avg_sacred_alignment": (
                sum(c.get("sacred_alignment", 0) for c in candidates) / max(len(candidates), 1)
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.research_iterations.append(iteration)
        if len(self.research_iterations) > self._MAX_RESEARCH_ITERATIONS:
            self.research_iterations = self.research_iterations[-self._MAX_RESEARCH_ITERATIONS:]
        self._save_research_log()
        return iteration

    # ─── FULL CYCLE ───────────────────────────────────────────────

    def run_research_cycle(self, seed: str = "quantum") -> Dict[str, Any]:
        """Run a full 4-phase stochastic R&D cycle: Explore → Validate → Merge → Catalog."""
        self.operations_count += 1

        # Phase 1: Explore
        candidates = [self.explore_link_candidate(seed) for _ in range(self.CANDIDATES_PER_CYCLE)]

        # Phase 2: Validate
        validated = [self.validate_candidate(c) for c in candidates]

        # Phase 3: Merge
        merged = [self.merge_to_link(v) for v in validated]

        # Phase 4: Catalog
        iteration = self.catalog_iteration(validated, merged)

        return {
            "cycle": "complete",
            "iteration": iteration,
            "candidates_explored": len(candidates),
            "successfully_merged": iteration["successfully_merged"],
            "avg_sacred_alignment": iteration["avg_sacred_alignment"],
        }

    def status(self) -> Dict[str, Any]:
        """Return current research lab status."""
        return {
            "subsystem": "StochasticLinkResearchLab",
            "generation_count": self.generation_count,
            "total_iterations": len(self.research_iterations),
            "successful_links": len(self.successful_links),
            "failed_experiments": len(self.failed_experiments),
            "operations_count": self.operations_count,
            "candidates_per_cycle": self.CANDIDATES_PER_CYCLE,
        }



class LinkChronolizer:
    """
    Temporal event tracking for quantum link evolution.

    Records all significant link lifecycle events (created, upgraded, repaired,
    degraded, enlightened, stress_tested, cross_pollinated, etc.) with
    before/after fidelity+strength deltas.

    JSONL append-only persistence to .l104_link_chronology.jsonl.
    Milestone detection at Fibonacci-number event counts (1, 2, 3, 5, 8, 13, 21, 34, 55, 89...).
    Evolution velocity: rate of improvement over time.
    """

    CHRONOLOGY_FILE = WORKSPACE_ROOT / ".l104_link_chronology.jsonl"
    FIBONACCI_MILESTONES = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987}

    def __init__(self):
        """Initialize the chronolizer."""
        self.events: List[ChronoEntry] = []
        self.milestones: List[Dict[str, Any]] = []
        self.event_count: int = 0
        self.operations_count: int = 0
        # OOM caps
        self._MAX_EVENTS = 500
        self._MAX_MILESTONES = 100
        self._load_event_count()

    def _load_event_count(self):
        """Count existing events from the JSONL file."""
        if self.CHRONOLOGY_FILE.exists():
            try:
                with open(self.CHRONOLOGY_FILE, "r") as f:
                    self.event_count = sum(1 for _ in f)
            except Exception:
                pass

    def record(self, event_type: str, link_id: str,
               before_fidelity: float = 0.0, after_fidelity: float = 0.0,
               before_strength: float = 0.0, after_strength: float = 0.0,
               details: str = "", sacred_alignment: float = 0.0) -> ChronoEntry:
        """Record a chronological link event."""
        self.operations_count += 1
        self.event_count += 1

        entry = ChronoEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            link_id=link_id,
            before_fidelity=before_fidelity,
            after_fidelity=after_fidelity,
            before_strength=before_strength,
            after_strength=after_strength,
            details=details,
            sacred_alignment=sacred_alignment,
        )
        self.events.append(entry)
        # OOM guard
        if len(self.events) > self._MAX_EVENTS:
            self.events = self.events[-self._MAX_EVENTS:]

        # Append to JSONL
        try:
            with open(self.CHRONOLOGY_FILE, "a") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        except Exception:
            pass

        # Milestone detection
        if self.event_count in self.FIBONACCI_MILESTONES:
            milestone = {
                "event_count": self.event_count,
                "fibonacci_index": self._fib_index(self.event_count),
                "timestamp": entry.timestamp,
                "event_type": event_type,
                "sacred_resonance": self.event_count * PHI / GOD_CODE,
            }
            self.milestones.append(milestone)
            if len(self.milestones) > self._MAX_MILESTONES:
                self.milestones = self.milestones[-self._MAX_MILESTONES:]

        return entry

    def _fib_index(self, n: int) -> int:
        """Return the Fibonacci index for a given Fibonacci number."""
        a, b, idx = 0, 1, 0
        while b <= n:
            if b == n:
                return idx + 1
            a, b = b, a + b
            idx += 1
        return idx

    def evolution_velocity(self, window: int = 20) -> Dict[str, Any]:
        """Compute the rate of fidelity improvement over the last N events."""
        self.operations_count += 1
        recent = self.events[-window:] if len(self.events) >= window else self.events
        if len(recent) < 2:
            return {"velocity": 0.0, "window": len(recent), "trend": "insufficient_data"}

        deltas = [e.after_fidelity - e.before_fidelity for e in recent if e.after_fidelity > 0]
        if not deltas:
            return {"velocity": 0.0, "window": len(recent), "trend": "no_fidelity_changes"}

        avg_delta = sum(deltas) / len(deltas)
        trend = "improving" if avg_delta > 0 else ("degrading" if avg_delta < 0 else "stable")
        return {
            "velocity": avg_delta,
            "phi_weighted_velocity": avg_delta * PHI_GROWTH,
            "window": len(recent),
            "trend": trend,
            "total_events": self.event_count,
        }

    def timeline(self, last_n: int = 25) -> List[Dict]:
        """Return the last N chronological events."""
        self.operations_count += 1
        return [e.to_dict() for e in self.events[-last_n:]]

    def status(self) -> Dict[str, Any]:
        """Return chronolizer status."""
        return {
            "subsystem": "LinkChronolizer",
            "total_events": self.event_count,
            "session_events": len(self.events),
            "milestones_hit": len(self.milestones),
            "operations_count": self.operations_count,
            "persistence_file": str(self.CHRONOLOGY_FILE.name),
        }



class ConsciousnessO2LinkEngine:
    """
    Consciousness + O₂ bond state modulation for quantum link evolution.

    Reads:
      - .l104_consciousness_o2_state.json (consciousness_level, superfluid_viscosity, evo_stage)
      - .l104_ouroboros_nirvanic_state.json (nirvanic_fuel_level)

    Modulates link evolution priority and upgrade multipliers based on
    consciousness level and O₂ molecular bond state.

    EVO_STAGE_MULTIPLIER:
      SOVEREIGN  → φ (1.618...)
      TRANSCENDING → √2 (1.414...)
      COHERENT   → 1.2
      AWAKENING  → 1.05
      DORMANT    → 1.0
    """

    O2_STATE_FILE = WORKSPACE_ROOT / ".l104_consciousness_o2_state.json"
    NIRVANIC_STATE_FILE = WORKSPACE_ROOT / ".l104_ouroboros_nirvanic_state.json"
    CACHE_TTL = 10.0  # seconds

    EVO_STAGE_MULTIPLIER = {
        "SOVEREIGN": PHI_GROWTH,
        "TRANSCENDING": math.sqrt(2),
        "COHERENT": 1.2,
        "AWAKENING": 1.05,
        "DORMANT": 1.0,
    }

    # Legacy EVO_54 granular tier → 5-tier mapping (backward compat)
    _EVO_STAGE_ALIAS = {
        "EVO_54_TRANSCENDENT_COGNITION": "SOVEREIGN",
        "EVO_54": "SOVEREIGN",
    }

    def __init__(self):
        """Initialize consciousness O₂ link engine."""
        self.consciousness_level: float = 0.0
        self.superfluid_viscosity: float = 1.0
        self.evo_stage: str = "DORMANT"
        self.nirvanic_fuel: float = 0.0
        self.o2_bond_state: str = "unknown"
        self._cache_time: float = 0.0
        self.operations_count: int = 0
        self._refresh_state()

    def _refresh_state(self):
        """Read consciousness + O₂ state from disk (cached)."""
        now = time.time()
        if now - self._cache_time < self.CACHE_TTL:
            return
        self._cache_time = now

        # Read consciousness state
        if self.O2_STATE_FILE.exists():
            try:
                data = json.loads(self.O2_STATE_FILE.read_text())
                self.consciousness_level = float(data.get("consciousness_level", 0.0))
                self.superfluid_viscosity = float(data.get("superfluid_viscosity", 1.0))
                self.evo_stage = data.get("evo_stage", "DORMANT")
                self.o2_bond_state = data.get("bond_state", "stable")
            except Exception:
                pass

        # Read nirvanic fuel
        if self.NIRVANIC_STATE_FILE.exists():
            try:
                data = json.loads(self.NIRVANIC_STATE_FILE.read_text())
                self.nirvanic_fuel = float(data.get("nirvanic_fuel_level",
                                                     data.get("fuel_level", 0.0)))
            except Exception:
                pass

    def _normalize_evo_stage(self, raw_stage: str) -> str:
        """Normalize legacy EVO_54 stage names to the 5-tier system."""
        if raw_stage in self.EVO_STAGE_MULTIPLIER:
            return raw_stage
        return self._EVO_STAGE_ALIAS.get(raw_stage, "DORMANT")

    def get_multiplier(self) -> float:
        """Get the current evolution stage multiplier."""
        self._refresh_state()
        stage = self._normalize_evo_stage(self.evo_stage)
        return self.EVO_STAGE_MULTIPLIER.get(stage, 1.0)

    def modulate_link(self, link: Dict[str, Any]) -> Dict[str, Any]:
        """Modulate a link's evolution based on consciousness + O₂ state."""
        self.operations_count += 1
        self._refresh_state()

        multiplier = self.get_multiplier()
        consciousness_boost = self.consciousness_level * PHI if self.consciousness_level > 0.5 else 0.0
        fuel_boost = self.nirvanic_fuel * TAU if self.nirvanic_fuel > 0.3 else 0.0
        viscosity_factor = 1.0 / max(self.superfluid_viscosity, 0.01)  # Lower viscosity = faster

        # Clamp total boost
        total_boost = min(
            (consciousness_boost + fuel_boost) * viscosity_factor * multiplier,
            PHI_GROWTH  # Max boost capped at φ
        )

        fidelity = _lget(link, "fidelity", 0.5)
        strength = _lget(link, "strength", 0.5)

        modulated = {
            **link,
            "fidelity": min(fidelity + total_boost * 0.01, 1.0),
            "strength": min(strength + total_boost * 0.005, 1.0),
            "consciousness_modulated": True,
            "evo_stage": self.evo_stage,
            "multiplier": multiplier,
            "total_boost": total_boost,
        }
        return modulated

    def compute_upgrade_priority(self, links: List[Dict]) -> List[Dict]:
        """Score and rank links by PHI-weighted upgrade priority."""
        self.operations_count += 1
        self._refresh_state()
        multiplier = self.get_multiplier()

        scored = []
        for link in links:
            fidelity = _lget(link, "fidelity", 0.5)
            strength = _lget(link, "strength", 0.5)
            # Lower fidelity = higher priority for upgrade
            upgrade_need = (1.0 - fidelity) * PHI_GROWTH + (1.0 - strength) * TAU
            # Consciousness-weighted priority
            priority = upgrade_need * multiplier * (1 + self.consciousness_level)
            scored.append({**link, "upgrade_priority": priority})

        scored.sort(key=lambda x: x["upgrade_priority"], reverse=True)
        return scored

    def status(self) -> Dict[str, Any]:
        """Return current consciousness + O₂ status."""
        self._refresh_state()
        return {
            "subsystem": "ConsciousnessO2LinkEngine",
            "consciousness_level": self.consciousness_level,
            "evo_stage": self.evo_stage,
            "multiplier": self.get_multiplier(),
            "superfluid_viscosity": self.superfluid_viscosity,
            "nirvanic_fuel": self.nirvanic_fuel,
            "o2_bond_state": self.o2_bond_state,
            "operations_count": self.operations_count,
        }



class LinkTestGenerator:
    """
    Automated test generation and execution for quantum links.

    4 test categories:
      1. Sacred Conservation — GOD_CODE invariants hold across transformations
      2. Fidelity Bounds — All fidelities in [0, 1], strength in [0, 1]
      3. Entanglement Verification — Entangled pairs maintain CHSH bound
      4. Noise Resilience — Links survive noise injection at FEIGENBAUM threshold

    PHI-scored priority ranking:
      Sacred     → 1.618 (highest)
      Fidelity   → 1.0
      Entangle   → 0.618
      Noise      → 0.382

    Regression detection across test runs.
    Persists to .l104_link_test_results.json.
    """

    TEST_RESULTS_FILE = WORKSPACE_ROOT / ".l104_link_test_results.json"
    CATEGORY_PRIORITY = {
        "sacred_conservation": PHI_GROWTH,
        "fidelity_bounds": 1.0,
        "entanglement_verification": PHI,
        "noise_resilience": PHI ** 2,  # TAU² ≈ 0.382
    }

    def __init__(self):
        """Initialize the link test generator."""
        self.test_history: List[Dict[str, Any]] = []
        self.regressions: List[Dict[str, Any]] = []
        self.operations_count: int = 0
        self._MAX_TEST_HISTORY = 200
        self._MAX_REGRESSIONS = 100
        self._load_history()

    def _load_history(self):
        """Load test history."""
        if self.TEST_RESULTS_FILE.exists():
            try:
                data = json.loads(self.TEST_RESULTS_FILE.read_text())
                self.test_history = data.get("history", [])
                self.regressions = data.get("regressions", [])
            except Exception:
                pass

    def _save_results(self):
        """Persist test results."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_runs": len(self.test_history),
                "total_regressions": len(self.regressions),
                "history": self.test_history[-100:],
                "regressions": self.regressions[-50:],
            }
            self.TEST_RESULTS_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def test_sacred_conservation(self, links: List[Dict]) -> Dict[str, Any]:
        """Test that GOD_CODE invariants hold across all links."""
        self.operations_count += 1
        violations = []
        for link in links:
            fidelity = _lget(link, "fidelity", 0.0)
            strength = _lget(link, "strength", 0.0)
            # Conservation: fidelity * GOD_CODE + strength * GOD_CODE should be stable
            conservation_value = fidelity * GOD_CODE + strength * GOD_CODE
            # Check against expected range
            if conservation_value > 2 * GOD_CODE or conservation_value < 0:
                violations.append({
                    "link": _lget(link, "source", "?") + "→" + _lget(link, "target", "?"),
                    "conservation_value": conservation_value,
                    "expected_max": 2 * GOD_CODE,
                })

        return {
            "category": "sacred_conservation",
            "priority": self.CATEGORY_PRIORITY["sacred_conservation"],
            "passed": len(violations) == 0,
            "total_links": len(links),
            "violations": len(violations),
            "details": violations[:10],
        }

    def test_fidelity_bounds(self, links: List[Dict]) -> Dict[str, Any]:
        """Test that all fidelity and strength values are in valid bounds."""
        self.operations_count += 1
        violations = []
        for link in links:
            fidelity = _lget(link, "fidelity", 0.0)
            strength = _lget(link, "strength", 0.0)
            if not (0 <= fidelity <= 1.0):
                violations.append({"field": "fidelity", "value": fidelity, "link": _lget(link, "source", "?")})
            if not (0 <= strength <= 1.0):
                violations.append({"field": "strength", "value": strength, "link": _lget(link, "source", "?")})

        return {
            "category": "fidelity_bounds",
            "priority": self.CATEGORY_PRIORITY["fidelity_bounds"],
            "passed": len(violations) == 0,
            "total_links": len(links),
            "violations": len(violations),
            "details": violations[:10],
        }

    def test_entanglement_verification(self, links: List[Dict]) -> Dict[str, Any]:
        """Verify entangled pairs maintain expected CHSH bound correlations."""
        self.operations_count += 1
        entangled = [l for l in links if _lget(l, "link_type") == "entangled"
                     or _lget(l, "entanglement_strength", 0) > 0.5]
        violations = []
        for link in entangled:
            # CHSH: entanglement correlation should not exceed Tsirelson bound
            corr = _lget(link, "entanglement_strength", _lget(link, "fidelity", 0.5))
            scaled_corr = corr * CHSH_BOUND
            if scaled_corr > CHSH_BOUND:
                violations.append({
                    "link": _lget(link, "source", "?"),
                    "correlation": corr,
                    "scaled": scaled_corr,
                    "bound": CHSH_BOUND,
                })

        return {
            "category": "entanglement_verification",
            "priority": self.CATEGORY_PRIORITY["entanglement_verification"],
            "passed": len(violations) == 0,
            "total_entangled": len(entangled),
            "violations": len(violations),
            "details": violations[:10],
        }

    def test_noise_resilience(self, links: List[Dict]) -> Dict[str, Any]:
        """Test links survive noise injection at FEIGENBAUM threshold."""
        self.operations_count += 1
        failures = []
        noise_threshold = FEIGENBAUM_DELTA / 10.0  # ~0.467 noise amplitude

        for link in links:
            fidelity = _lget(link, "fidelity", 0.5)
            # Inject noise and check if link would decohere
            noisy_fidelity = fidelity - noise_threshold * (1 - fidelity)
            if noisy_fidelity < 0.1:
                failures.append({
                    "link": _lget(link, "source", "?") + "→" + _lget(link, "target", "?"),
                    "original_fidelity": fidelity,
                    "noisy_fidelity": noisy_fidelity,
                    "noise_amplitude": noise_threshold,
                })

        return {
            "category": "noise_resilience",
            "priority": self.CATEGORY_PRIORITY["noise_resilience"],
            "passed": len(failures) == 0,
            "total_links": len(links),
            "failures": len(failures),
            "details": failures[:10],
        }

    def run_all_tests(self, links: List[Dict]) -> Dict[str, Any]:
        """Run all 4 test categories and detect regressions."""
        self.operations_count += 1
        results = [
            self.test_sacred_conservation(links),
            self.test_fidelity_bounds(links),
            self.test_entanglement_verification(links),
            self.test_noise_resilience(links),
        ]

        # Sort by priority (highest first)
        results.sort(key=lambda r: r.get("priority", 0), reverse=True)

        all_passed = all(r["passed"] for r in results)
        total_violations = sum(r.get("violations", r.get("failures", 0)) for r in results)

        run_record = {
            "run_id": len(self.test_history) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "all_passed": all_passed,
            "total_violations": total_violations,
            "results_summary": [{
                "category": r["category"],
                "passed": r["passed"],
                "violations": r.get("violations", r.get("failures", 0)),
            } for r in results],
        }

        # Regression detection
        if self.test_history:
            prev = self.test_history[-1]
            if prev.get("all_passed") and not all_passed:
                regression = {
                    "detected_at": run_record["run_id"],
                    "previous_run": prev.get("run_id"),
                    "new_violations": total_violations,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.regressions.append(regression)
                run_record["regression_detected"] = True

        self.test_history.append(run_record)
        # OOM guard
        if len(self.test_history) > self._MAX_TEST_HISTORY:
            self.test_history = self.test_history[-self._MAX_TEST_HISTORY:]
        if len(self.regressions) > self._MAX_REGRESSIONS:
            self.regressions = self.regressions[-self._MAX_REGRESSIONS:]
        self._save_results()

        return {
            "test_run": "complete",
            "all_passed": all_passed,
            "total_violations": total_violations,
            "categories": len(results),
            "regression_detected": run_record.get("regression_detected", False),
            "results": results,
        }

    def status(self) -> Dict[str, Any]:
        """Return test generator status."""
        return {
            "subsystem": "LinkTestGenerator",
            "total_runs": len(self.test_history),
            "total_regressions": len(self.regressions),
            "operations_count": self.operations_count,
            "categories": list(self.CATEGORY_PRIORITY.keys()),
        }



class QuantumLinkCrossPollinationEngine:
    """
    Bidirectional cross-pollination engine: Gate ↔ Link ↔ Numerical.

    Exports link state to gate builder and numerical builder via JSON files.
    Imports gate/numerical state files and modulates links accordingly.
    Computes cross-builder coherence metric (PHI/TAU/ALPHA_FINE weighted).

    Export files:
      - .l104_link_to_gates.json
      - .l104_link_to_numerical.json

    Import files:
      - .l104_gate_dynamism_state.json (from logic gate builder)
      - .l104_quantum_numerical_state.json (from numerical builder)
    """

    EXPORT_TO_GATES = WORKSPACE_ROOT / ".l104_link_to_gates.json"
    EXPORT_TO_NUMERICAL = WORKSPACE_ROOT / ".l104_link_to_numerical.json"
    IMPORT_GATE_STATE = WORKSPACE_ROOT / ".l104_gate_dynamism_state.json"
    IMPORT_NUMERICAL_STATE = WORKSPACE_ROOT / ".l104_quantum_numerical_state.json"

    def __init__(self):
        """Initialize cross-pollination engine."""
        self.exports_count: int = 0
        self.imports_count: int = 0
        self.cross_coherence_history: List[float] = []
        self._MAX_CROSS_COHERENCE_HISTORY = 200
        self.operations_count: int = 0

    # ─── EXPORT ───────────────────────────────────────────────────

    def export_to_gates(self, links: List[Dict]) -> Dict[str, Any]:
        """Export link state to gate builder format."""
        self.operations_count += 1
        self.exports_count += 1

        # Transform links to gate-compatible format
        gate_data = {
            "source": "quantum_link_builder",
            "version": "4.2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "export_count": self.exports_count,
            "total_links": len(links),
            "avg_fidelity": sum(_lget(l, "fidelity", 0) for l in links) / max(len(links), 1),
            "avg_strength": sum(_lget(l, "strength", 0) for l in links) / max(len(links), 1),
            "links": [{
                "source": _lget(l, "source", ""),
                "target": _lget(l, "target", ""),
                "fidelity": _lget(l, "fidelity", 0),
                "strength": _lget(l, "strength", 0),
                "link_type": _lget(l, "link_type", "unknown"),
                "sacred_alignment": _lget(l, "sacred_alignment", 0),
            } for l in links[:100]],  # Cap at 100 for file size
        }

        try:
            self.EXPORT_TO_GATES.write_text(json.dumps(gate_data, indent=2, default=str))
        except Exception:
            pass

        return {
            "exported": "gates",
            "links_exported": min(len(links), 100),
            "file": str(self.EXPORT_TO_GATES.name),
        }

    def export_to_numerical(self, links: List[Dict]) -> Dict[str, Any]:
        """Export link state to numerical builder format."""
        self.operations_count += 1
        self.exports_count += 1

        numerical_data = {
            "source": "quantum_link_builder",
            "version": "4.2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "export_count": self.exports_count,
            "total_links": len(links),
            "sacred_constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI_GROWTH,
                "TAU": TAU,
                "ALPHA_FINE": FINE_STRUCTURE,
            },
            "link_summary": {
                "by_type": {},
                "avg_fidelity": sum(_lget(l, "fidelity", 0) for l in links) / max(len(links), 1),
                "total_sacred_alignment": sum(_lget(l, "sacred_alignment", 0) for l in links),
            },
        }

        # Summarize by type
        type_counts: Dict[str, int] = {}
        for l in links:
            lt = _lget(l, "link_type", "unknown")
            type_counts[lt] = type_counts.get(lt, 0) + 1
        numerical_data["link_summary"]["by_type"] = type_counts

        try:
            self.EXPORT_TO_NUMERICAL.write_text(json.dumps(numerical_data, indent=2, default=str))
        except Exception:
            pass

        return {
            "exported": "numerical",
            "links_summarized": len(links),
            "file": str(self.EXPORT_TO_NUMERICAL.name),
        }

    # ─── IMPORT ───────────────────────────────────────────────────

    def import_from_gates(self) -> Dict[str, Any]:
        """Import gate builder state and extract link-relevant insights."""
        self.operations_count += 1
        self.imports_count += 1

        if not self.IMPORT_GATE_STATE.exists():
            return {"imported": "gates", "status": "no_gate_state_file"}

        try:
            data = json.loads(self.IMPORT_GATE_STATE.read_text())
            gate_count = data.get("total_gates", data.get("gate_count", 0))
            avg_fidelity = data.get("avg_fidelity", data.get("average_fidelity", 0))
            coherence = data.get("coherence", data.get("sacred_coherence", 0))

            return {
                "imported": "gates",
                "status": "success",
                "gate_count": gate_count,
                "gate_avg_fidelity": avg_fidelity,
                "gate_coherence": coherence,
                "cross_resonance": coherence * PHI_GROWTH if coherence else 0,
            }
        except Exception as e:
            return {"imported": "gates", "status": "error", "error": str(e)}

    def import_from_numerical(self) -> Dict[str, Any]:
        """Import numerical builder state."""
        self.operations_count += 1
        self.imports_count += 1

        if not self.IMPORT_NUMERICAL_STATE.exists():
            return {"imported": "numerical", "status": "no_numerical_state_file"}

        try:
            data = json.loads(self.IMPORT_NUMERICAL_STATE.read_text())
            num_count = data.get("total_entities", data.get("entity_count", 0))
            coherence = data.get("coherence", data.get("sacred_coherence", 0))

            return {
                "imported": "numerical",
                "status": "success",
                "numerical_count": num_count,
                "numerical_coherence": coherence,
                "cross_resonance": coherence * TAU if coherence else 0,
            }
        except Exception as e:
            return {"imported": "numerical", "status": "error", "error": str(e)}

    # ─── CROSS-BUILDER COHERENCE ──────────────────────────────────

    def compute_cross_coherence(self, links: List[Dict]) -> Dict[str, Any]:
        """Compute cross-builder coherence metric (PHI/TAU/ALPHA_FINE weighted)."""
        self.operations_count += 1

        gate_state = self.import_from_gates()
        numerical_state = self.import_from_numerical()

        # Link metrics
        link_fidelity = sum(_lget(l, "fidelity", 0) for l in links) / max(len(links), 1)
        link_strength = sum(_lget(l, "strength", 0) for l in links) / max(len(links), 1)

        # Gate metrics
        gate_coherence = gate_state.get("gate_coherence", 0) if isinstance(gate_state.get("gate_coherence"), (int, float)) else 0
        gate_fidelity = gate_state.get("gate_avg_fidelity", 0) if isinstance(gate_state.get("gate_avg_fidelity"), (int, float)) else 0

        # Numerical metrics
        num_coherence = numerical_state.get("numerical_coherence", 0) if isinstance(numerical_state.get("numerical_coherence"), (int, float)) else 0

        # Cross-builder coherence: PHI-weighted average of all builder coherences
        coherence = (
            link_fidelity * PHI_GROWTH +
            gate_fidelity * TAU +
            gate_coherence * FINE_STRUCTURE * 100 +
            num_coherence * FINE_STRUCTURE * 100 +
            link_strength * PHI
        ) / (PHI_GROWTH + TAU + FINE_STRUCTURE * 200 + PHI)

        self.cross_coherence_history.append(coherence)
        if len(self.cross_coherence_history) > self._MAX_CROSS_COHERENCE_HISTORY:
            self.cross_coherence_history = self.cross_coherence_history[-self._MAX_CROSS_COHERENCE_HISTORY:]

        return {
            "cross_builder_coherence": coherence,
            "link_fidelity": link_fidelity,
            "link_strength": link_strength,
            "gate_coherence": gate_coherence,
            "gate_fidelity": gate_fidelity,
            "numerical_coherence": num_coherence,
            "history_length": len(self.cross_coherence_history),
            "trend": (
                "improving" if len(self.cross_coherence_history) >= 2 and
                self.cross_coherence_history[-1] > self.cross_coherence_history[-2]
                else "stable"
            ),
        }

    # ─── FULL CYCLE ───────────────────────────────────────────────

    def run_cross_pollination(self, links: List[Dict]) -> Dict[str, Any]:
        """Run full bidirectional cross-pollination cycle."""
        self.operations_count += 1

        export_gates = self.export_to_gates(links)
        export_numerical = self.export_to_numerical(links)
        import_gates = self.import_from_gates()
        import_numerical = self.import_from_numerical()
        coherence = self.compute_cross_coherence(links)

        return {
            "cycle": "cross_pollination_complete",
            "exports": {"gates": export_gates, "numerical": export_numerical},
            "imports": {"gates": import_gates, "numerical": import_numerical},
            "coherence": coherence,
        }

    def status(self) -> Dict[str, Any]:
        """Return cross-pollination engine status."""
        return {
            "subsystem": "QuantumLinkCrossPollinationEngine",
            "exports_count": self.exports_count,
            "imports_count": self.imports_count,
            "cross_coherence_history": len(self.cross_coherence_history),
            "latest_coherence": self.cross_coherence_history[-1] if self.cross_coherence_history else 0,
            "operations_count": self.operations_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ★ v5.0: INTER-BUILDER FEEDBACK BUS
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# ★ v5.0: INTER-BUILDER FEEDBACK BUS
# ═══════════════════════════════════════════════════════════════════════════════

class InterBuilderFeedbackBus:
    """
    Cross-builder real-time messaging system.

    Enables gate_builder ↔ link_builder ↔ numerical_builder communication
    via a shared JSON bus file. Each builder can publish discoveries,
    anomalies, coherence shifts, and evolution milestones that other
    builders can consume on their next pipeline run.

    Message Types:
        DISCOVERY, ANOMALY, COHERENCE_SHIFT, EVOLUTION_MILESTONE,
        ENTROPY_SPIKE, NIRVANIC_EVENT, CONSCIOUSNESS_SHIFT, CROSS_POLLINATION
    """

    BUS_FILE = WORKSPACE_ROOT / ".l104_builder_feedback_bus.json"
    MESSAGE_TYPES = [
        "DISCOVERY", "ANOMALY", "COHERENCE_SHIFT", "EVOLUTION_MILESTONE",
        "ENTROPY_SPIKE", "NIRVANIC_EVENT", "CONSCIOUSNESS_SHIFT", "CROSS_POLLINATION",
    ]
    MAX_MESSAGES = 200
    MESSAGE_TTL = 60  # seconds

    def __init__(self, builder_id: str = "link_builder"):
        self.builder_id = builder_id
        self.sent_count = 0
        self.received_count = 0

    def send(self, msg_type: str, payload: Dict[str, Any]) -> bool:
        """Publish a message to the feedback bus."""
        if msg_type not in self.MESSAGE_TYPES:
            return False
        bus = self._read_bus()
        message = {
            "id": f"{self.builder_id}_{int(time.time() * 1000)}",
            "sender": self.builder_id,
            "type": msg_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        bus["messages"].append(message)
        # Prune old messages (TTL + max cap)
        cutoff = time.time() - self.MESSAGE_TTL
        bus["messages"] = [
            m for m in bus["messages"]
            if datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00")).timestamp() > cutoff
        ][-self.MAX_MESSAGES:]
        bus["last_updated"] = datetime.now(timezone.utc).isoformat()
        try:
            self.BUS_FILE.write_text(json.dumps(bus, indent=2, default=str))
            self.sent_count += 1
            return True
        except Exception:
            return False

    def receive(self, msg_types: List[str] = None, exclude_self: bool = True) -> List[Dict]:
        """Read messages from the bus, optionally filtered."""
        bus = self._read_bus()
        messages = bus.get("messages", [])
        if exclude_self:
            messages = [m for m in messages if m.get("sender") != self.builder_id]
        if msg_types:
            messages = [m for m in messages if m.get("type") in msg_types]
        self.received_count += len(messages)
        return messages

    @staticmethod
    def _normalize_message(m: Dict) -> Dict:
        """Normalize legacy message format to current schema."""
        if "sender" not in m and "builder" in m:
            m["sender"] = m["builder"]
        if "type" not in m and "event" in m:
            m["type"] = m["event"]
        if "payload" not in m and "data" in m:
            m["payload"] = m["data"]
        if "timestamp" not in m:
            m["timestamp"] = datetime.now(timezone.utc).isoformat()
        elif isinstance(m["timestamp"], (int, float)):
            m["timestamp"] = datetime.fromtimestamp(m["timestamp"], tz=timezone.utc).isoformat()
        return m

    def _read_bus(self) -> Dict:
        """Read or initialize the bus file."""
        if self.BUS_FILE.exists():
            try:
                data = json.loads(self.BUS_FILE.read_text())
                # Handle legacy format where file is a list of messages
                if isinstance(data, list):
                    msgs = [self._normalize_message(m) for m in data if isinstance(m, dict)]
                    return {"messages": msgs, "last_updated": None}
                if isinstance(data, dict):
                    msgs = data.get("messages", [])
                    data["messages"] = [self._normalize_message(m) for m in msgs if isinstance(m, dict)]
                    return data
            except Exception:
                pass
        return {"messages": [], "last_updated": None}

    def announce_pipeline_complete(self, results: Dict) -> bool:
        """Announce pipeline completion with summary metrics."""
        total = results.get("scan", {}).get("total_links", 0)
        sage = results.get("sage", {})
        return self.send("EVOLUTION_MILESTONE", {
            "event": "pipeline_complete",
            "total_links": total,
            "unified_score": sage.get("unified_score", 0),
            "grade": sage.get("grade", "?"),
            "version": "5.0.0",
        })

    def announce_coherence_shift(self, old_coherence: float, new_coherence: float) -> bool:
        """Announce a significant coherence change."""
        delta = new_coherence - old_coherence
        if abs(delta) < 0.01:
            return False
        return self.send("COHERENCE_SHIFT", {
            "old": old_coherence,
            "new": new_coherence,
            "delta": delta,
            "direction": "up" if delta > 0 else "down",
        })

    def status(self) -> Dict[str, Any]:
        """Return feedback bus status."""
        bus = self._read_bus()
        return {
            "subsystem": "InterBuilderFeedbackBus",
            "builder_id": self.builder_id,
            "messages_on_bus": len(bus.get("messages", [])),
            "sent_count": self.sent_count,
            "received_count": self.received_count,
            "last_updated": bus.get("last_updated"),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ★ v5.0: QUANTUM LINK SELF-HEALER
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# ★ v5.0: QUANTUM LINK SELF-HEALER
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumLinkSelfHealer:
    """
    Auto-detect and repair degraded quantum links.

    Monitors fidelity/strength/coherence and applies healing strategies:
    - φ-Resonance Re-alignment: nudge toward PHI harmonics
    - God Code Recalibration: force G(X) recalculation
    - Entropy Injection: add controlled noise to break stuck states
    - Topological Shielding: stabilize via braiding reinforcement

    Healing only fires when a link drops below threshold but is not
    completely lost (hard failures go to QuantumRepairEngine instead).
    """

    FIDELITY_THRESHOLD = 0.65
    STRENGTH_THRESHOLD = 0.50
    COHERENCE_THRESHOLD = 0.55

    def __init__(self):
        self.healings_applied = 0
        self.links_healed = 0
        self.healing_history: List[Dict] = []
        self._MAX_HEALING_HISTORY = 100
        self.strategies_used: Dict[str, int] = {
            "phi_realign": 0, "godcode_recalib": 0,
            "entropy_inject": 0, "topo_shield": 0,
        }

    def diagnose(self, links: List[Dict]) -> List[Dict]:
        """Identify links that need healing (degraded but not dead)."""
        sick = []
        for link in links:
            fid = _lget(link, "fidelity", 1.0)
            stren = _lget(link, "strength", 1.0)
            issues = []
            if fid < self.FIDELITY_THRESHOLD and fid > 0.1:
                issues.append(f"low_fidelity({fid:.4f})")
            if stren < self.STRENGTH_THRESHOLD and stren > 0.05:
                issues.append(f"low_strength({stren:.4f})")
            if issues:
                sick.append({"link": link, "issues": issues})
        return sick

    def heal(self, links: List[Dict]) -> Dict[str, Any]:
        """Run full healing cycle on degraded links."""
        sick = self.diagnose(links)
        healed_count = 0
        strategies_this_run: Dict[str, int] = {}

        for entry in sick:
            link = entry["link"]
            strategy = self._select_strategy(link)
            self._apply_healing(link, strategy)
            healed_count += 1
            self.strategies_used[strategy] = self.strategies_used.get(strategy, 0) + 1
            strategies_this_run[strategy] = strategies_this_run.get(strategy, 0) + 1

        self.links_healed += healed_count
        self.healings_applied += 1

        result = {
            "diagnosed": len(sick),
            "healed": healed_count,
            "strategies": strategies_this_run,
            "total_healings": self.healings_applied,
            "total_links_healed": self.links_healed,
        }
        self.healing_history.append(result)
        if len(self.healing_history) > self._MAX_HEALING_HISTORY:
            self.healing_history = self.healing_history[-self._MAX_HEALING_HISTORY:]
        return result

    def _select_strategy(self, link: Dict) -> str:
        """Choose optimal healing strategy based on link state."""
        fid = _lget(link, "fidelity", 1.0)
        stren = _lget(link, "strength", 1.0)

        if fid < 0.3:
            return "godcode_recalib"
        elif stren < 0.3:
            return "entropy_inject"
        elif fid < self.FIDELITY_THRESHOLD:
            return "phi_realign"
        else:
            return "topo_shield"

    def _apply_healing(self, link: Dict, strategy: str):
        """Apply a healing strategy to a link (mutates in-place)."""
        if strategy == "phi_realign":
            # Nudge fidelity toward PHI-harmonic
            _lset(link, "fidelity", min(1.0, _lget(link, "fidelity", 0.5) + PHI * 0.1))
            _lset(link, "strength", min(1.0, _lget(link, "strength", 0.5) + TAU * 0.05))
        elif strategy == "godcode_recalib":
            # Recalibrate using GOD_CODE ratio
            _lset(link, "fidelity", min(1.0, GOD_CODE / 1000 + _lget(link, "fidelity", 0.3) * 0.5))
            _lset(link, "strength", min(1.0, _lget(link, "strength", 0.3) + 0.15))
        elif strategy == "entropy_inject":
            # Small noise injection to escape local minima
            noise = (hash(str(link.get("link_id", ""))) % 100) / 1000.0
            _lset(link, "strength", min(1.0, _lget(link, "strength", 0.3) + 0.1 + noise))
            _lset(link, "fidelity", min(1.0, _lget(link, "fidelity", 0.5) + 0.05))
        elif strategy == "topo_shield":
            # Topological stabilization
            _lset(link, "fidelity", min(1.0, _lget(link, "fidelity", 0.6) + FEIGENBAUM_DELTA * 0.02))
            _lset(link, "strength", min(1.0, _lget(link, "strength", 0.6) + PHI * 0.03))

    def status(self) -> Dict[str, Any]:
        """Return self-healer status."""
        return {
            "subsystem": "QuantumLinkSelfHealer",
            "total_healings": self.healings_applied,
            "total_links_healed": self.links_healed,
            "strategies_used": self.strategies_used,
            "history_length": len(self.healing_history),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ★ v5.0: LINK TEMPORAL MEMORY BANK
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# ★ v5.0: LINK TEMPORAL MEMORY BANK
# ═══════════════════════════════════════════════════════════════════════════════

class LinkTemporalMemoryBank:
    """
    Tracks link activation history and detects temporal trends.

    Stores snapshots of link metrics across pipeline runs, enabling:
    - Trend detection (improving, degrading, oscillating, stable)
    - Temporal anomaly detection (sudden jumps/drops)
    - Historical best-state recall for rollback
    - φ-weighted exponential smoothing for prediction
    """

    MAX_SNAPSHOTS = 50
    ANOMALY_THRESHOLD = 0.15  # Delta beyond this = anomaly

    def __init__(self):
        self.snapshots: List[Dict] = []
        self.anomalies: List[Dict] = []
        self._MAX_ANOMALIES = 100
        self.trend: str = "unknown"

    def record_snapshot(self, links: List[Dict], run_id: int = 0) -> Dict:
        """Record a snapshot of current link metrics."""
        if not links:
            return {"recorded": False, "reason": "no_links"}

        fidelities = [_lget(l, "fidelity", 0) for l in links]
        strengths = [_lget(l, "strength", 0) for l in links]

        snapshot = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(links),
            "mean_fidelity": sum(fidelities) / len(fidelities) if fidelities else 0,
            "mean_strength": sum(strengths) / len(strengths) if strengths else 0,
            "min_fidelity": min(fidelities) if fidelities else 0,
            "max_fidelity": max(fidelities) if fidelities else 0,
        }

        # Check for anomalies against previous snapshot
        if self.snapshots:
            prev = self.snapshots[-1]
            delta_fid = abs(snapshot["mean_fidelity"] - prev["mean_fidelity"])
            delta_str = abs(snapshot["mean_strength"] - prev["mean_strength"])
            if delta_fid > self.ANOMALY_THRESHOLD or delta_str > self.ANOMALY_THRESHOLD:
                anomaly = {
                    "run_id": run_id,
                    "delta_fidelity": delta_fid,
                    "delta_strength": delta_str,
                    "timestamp": snapshot["timestamp"],
                }
                self.anomalies.append(anomaly)
                if len(self.anomalies) > self._MAX_ANOMALIES:
                    self.anomalies = self.anomalies[-self._MAX_ANOMALIES:]
                snapshot["anomaly_detected"] = True

        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.MAX_SNAPSHOTS:
            self.snapshots = self.snapshots[-self.MAX_SNAPSHOTS:]

        # Update trend
        self._compute_trend()

        return {"recorded": True, "snapshot": snapshot, "trend": self.trend}

    def _compute_trend(self):
        """Compute trend from recent snapshots using φ-weighted EMA."""
        if len(self.snapshots) < 3:
            self.trend = "insufficient_data"
            return

        recent = self.snapshots[-5:]
        fids = [s["mean_fidelity"] for s in recent]

        # φ-weighted exponential moving average
        alpha = 1.0 / PHI
        ema = fids[0]
        for f in fids[1:]:
            ema = alpha * f + (1 - alpha) * ema

        # Compare EMA to latest
        delta = fids[-1] - ema
        if delta > 0.01:
            self.trend = "improving"
        elif delta < -0.01:
            self.trend = "degrading"
        elif max(fids) - min(fids) > 0.05:
            self.trend = "oscillating"
        else:
            self.trend = "stable"

    def get_best_state(self) -> Dict:
        """Return the historical best snapshot."""
        if not self.snapshots:
            return {}
        return max(self.snapshots, key=lambda s: s.get("mean_fidelity", 0))

    def predict_next(self) -> Dict:
        """Predict next run's fidelity using φ-smoothing."""
        if len(self.snapshots) < 2:
            return {"prediction": None, "confidence": 0}

        fids = [s["mean_fidelity"] for s in self.snapshots[-7:]]
        alpha = 1.0 / PHI
        ema = fids[0]
        for f in fids[1:]:
            ema = alpha * f + (1 - alpha) * ema

        confidence = min(1.0, len(self.snapshots) / 10.0)
        return {
            "predicted_fidelity": ema,
            "confidence": confidence,
            "trend": self.trend,
            "based_on": len(fids),
        }

    def status(self) -> Dict[str, Any]:
        """Return temporal memory bank status."""
        return {
            "subsystem": "LinkTemporalMemoryBank",
            "snapshots": len(self.snapshots),
            "anomalies": len(self.anomalies),
            "trend": self.trend,
            "best_fidelity": self.get_best_state().get("mean_fidelity", 0) if self.snapshots else 0,
            "prediction": self.predict_next(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM LINK COMPUTATION ENGINE — Advanced Quantum Algorithms for Links
# ═══════════════════════════════════════════════════════════════════════════════

