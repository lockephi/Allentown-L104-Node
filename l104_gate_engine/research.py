"""L104 Gate Engine — Gate Research Engine + Stochastic Gate Research Lab."""

import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import (
    PHI, TAU, GOD_CODE, CALABI_YAU_DIM, WORKSPACE_ROOT, VERSION,
)
from .models import LogicGate, GateLink
from .gate_functions import sage_logic_gate, quantum_logic_gate


class GateResearchEngine:
    """Advanced research engine for logic gate analysis.

    Ported from QuantumResearchEngine in l104_quantum_link_builder.py:
      Module 1: ANOMALY DETECTION — IQR-based outlier identification on gate metrics
      Module 2: CAUSAL ANALYSIS   — Correlation between complexity, entropy, connectivity
      Module 3: EVOLUTION ANALYSIS — Trend detection from chronolizer history
      Module 4: CROSS-POLLINATION — Insights for quantum link builder integration
      Module 5: KNOWLEDGE SYNTHESIS — Aggregate health score with adaptive learning

    Enables the gate builder to match the quantum link builder's research depth
    and feed learned insights back for cross-system improvement.
    """

    MEMORY_FILE = WORKSPACE_ROOT / ".l104_gate_research_memory.json"

    def __init__(self):
        """Initialize the gate research engine and load persisted memory."""
        self.memory: Dict = {}
        self._load_memory()

    def _load_memory(self):
        """Load persistent research memory."""
        if self.MEMORY_FILE.exists():
            try:
                self.memory = json.loads(self.MEMORY_FILE.read_text())
            except Exception:
                self.memory = {}

    def _save_memory(self):
        """Persist research memory to disk."""
        try:
            self.memory["last_updated"] = datetime.now(timezone.utc).isoformat()
            self.MEMORY_FILE.write_text(json.dumps(self.memory, indent=2, default=str))
        except Exception:
            pass

    def full_research(self, gates: List[LogicGate], links: List[GateLink],
                      chronolizer=None) -> Dict:
        """Run all research modules and produce a unified report."""
        if not gates:
            return {"total_gates": 0, "research_health": 0}

        start = time.time()

        # Extract metrics arrays
        complexities = [g.complexity for g in gates]
        entropies = [g.entropy_score for g in gates]
        link_counts = []
        for g in gates:
            lc = sum(1 for l in links if g.name in (l.source_gate, l.target_gate))
            link_counts.append(lc)

        # Module 1: Anomaly detection
        anomalies = self._detect_anomalies(gates, complexities, entropies, link_counts)

        # Module 2: Causal analysis
        causal = self._causal_analysis(complexities, entropies, link_counts, gates)

        # Module 3: Evolution analysis
        evolution = self._evolution_analysis(chronolizer) if chronolizer else {}

        # Module 4: Cross-pollination data preparation
        cross_poll = self._prepare_cross_pollination(gates, links, anomalies, causal)

        # Module 5: Knowledge synthesis + adaptive learning
        synthesis = self._knowledge_synthesis(
            gates, links, anomalies, causal, evolution, cross_poll)

        elapsed = time.time() - start

        result = {
            "total_gates": len(gates),
            "anomaly_detection": anomalies,
            "causal_analysis": causal,
            "evolution_analysis": evolution,
            "cross_pollination": cross_poll,
            "knowledge_synthesis": synthesis,
            "research_health": synthesis.get("research_health", 0),
            "research_time_ms": elapsed * 1000,
        }

        # Persist snapshot for self-learning
        self._record_snapshot(result)
        self._save_memory()

        return result

    def _detect_anomalies(self, gates: List[LogicGate],
                          complexities: List[int], entropies: List[float],
                          link_counts: List[int]) -> Dict:
        """IQR-based outlier detection on gate metrics."""
        anomalies = []
        for prop_name, values in [("complexity", complexities),
                                   ("entropy", entropies),
                                   ("connectivity", link_counts)]:
            fvals = [float(v) for v in values]
            if len(fvals) < 4:
                continue
            sorted_v = sorted(fvals)
            n = len(sorted_v)
            q1, q3 = sorted_v[n // 4], sorted_v[3 * n // 4]
            iqr = q3 - q1
            lower, upper = q1 - 2.0 * iqr, q3 + 2.0 * iqr

            for i, v in enumerate(fvals):
                if v < lower or v > upper:
                    anomalies.append({
                        "gate": gates[i].name[:60],
                        "property": prop_name,
                        "value": v,
                        "severity": "extreme" if (v < lower - iqr or v > upper + iqr) else "mild",
                    })

        return {
            "total_anomalies": len(anomalies),
            "extreme": sum(1 for a in anomalies if a["severity"] == "extreme"),
            "mild": sum(1 for a in anomalies if a["severity"] == "mild"),
            "anomaly_rate": len(anomalies) / max(1, len(gates) * 3),
            "top_anomalies": anomalies[:10],
        }

    def _causal_analysis(self, complexities: List[int], entropies: List[float],
                         link_counts: List[int], gates: List[LogicGate]) -> Dict:
        """Pearson correlation between gate properties."""
        def _pearson(xs, ys):
            n = len(xs)
            if n < 2:
                return 0.0
            mx, my = sum(xs) / n, sum(ys) / n
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            sxx = sum((x - mx) ** 2 for x in xs)
            syy = sum((y - my) ** 2 for y in ys)
            d = math.sqrt(sxx * syy)
            return sxy / d if d > 1e-15 else 0.0

        props = {
            "complexity": [float(c) for c in complexities],
            "entropy": entropies,
            "connectivity": [float(c) for c in link_counts],
            "has_docstring": [1.0 if g.docstring else 0.0 for g in gates],
            "param_count": [float(len(g.parameters)) for g in gates],
        }

        correlations = {}
        strong = []
        names = list(props.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = _pearson(props[names[i]], props[names[j]])
                key = f"{names[i]}↔{names[j]}"
                correlations[key] = round(r, 4)
                if abs(r) > 0.5:
                    strong.append({"pair": key, "correlation": r,
                                   "strength": "strong" if abs(r) > 0.7 else "moderate"})

        return {
            "correlations": correlations,
            "strong_correlations": strong,
            "total_strong": len(strong),
        }

    def _evolution_analysis(self, chronolizer) -> Dict:
        """Analyze gate evolution trends from chronolizer history."""
        if not chronolizer or not chronolizer.entries:
            return {"has_history": False}

        entries = chronolizer.entries
        summary = chronolizer.summary()

        unique_gates = summary.get("unique_gates_tracked", 1)
        total = summary.get("total_entries", 0)
        velocity = total / max(1, unique_gates)

        recent = entries[-50:]
        recent_events = {}
        for e in recent:
            recent_events[e.event] = recent_events.get(e.event, 0) + 1

        modified = summary.get("events_by_type", {}).get("modified", 0)
        discovered = summary.get("events_by_type", {}).get("discovered", 0)
        churn_rate = modified / max(1, discovered)

        if churn_rate < 0.1:
            stability = "highly_stable"
        elif churn_rate < 0.3:
            stability = "stable"
        elif churn_rate < 0.6:
            stability = "moderate_churn"
        else:
            stability = "high_churn"

        return {
            "has_history": True,
            "total_events": total,
            "unique_gates": unique_gates,
            "evolution_velocity": velocity,
            "churn_rate": churn_rate,
            "stability": stability,
            "recent_activity": recent_events,
            "oldest": summary.get("oldest"),
            "newest": summary.get("newest"),
        }

    def _prepare_cross_pollination(self, gates: List[LogicGate],
                                   links: List[GateLink],
                                   anomalies: Dict, causal: Dict) -> Dict:
        """Prepare data for cross-pollination with quantum link builder."""
        gates_by_file: Dict[str, Dict] = {}
        for g in gates:
            if g.source_file not in gates_by_file:
                gates_by_file[g.source_file] = {
                    "count": 0, "types": [], "complexity_sum": 0,
                    "entropy_sum": 0.0, "tested": 0, "linked": 0}
            gf = gates_by_file[g.source_file]
            gf["count"] += 1
            if g.gate_type not in gf["types"]:
                gf["types"].append(g.gate_type)
            gf["complexity_sum"] += g.complexity
            gf["entropy_sum"] += g.entropy_score
            if g.test_status == "passed":
                gf["tested"] += 1
            if g.quantum_links:
                gf["linked"] += 1

        high_value_files = []
        for fname, data in gates_by_file.items():
            score = (data["count"] * 0.3 + data["tested"] * 0.3
                     + data["linked"] * 0.2 + min(1.0, data["complexity_sum"] / 50) * 0.2)
            if score > 0.5:
                high_value_files.append({"file": fname, "score": score,
                                         "gates": data["count"]})

        insights = []
        if anomalies.get("extreme", 0) > 2:
            insights.append("GATE_ANOMALY: Multiple extreme outliers — "
                           "quantum repair engine should prioritize these files")
        strong_corrs = causal.get("strong_correlations", [])
        for c in strong_corrs:
            if "complexity" in c["pair"] and "entropy" in c["pair"]:
                insights.append(f"GATE_CAUSAL: Complexity↔Entropy correlation "
                               f"(r={c['correlation']:.3f}) — mirrors quantum "
                               f"fidelity↔strength pattern")

        return {
            "gates_by_file": gates_by_file,
            "high_value_files": sorted(high_value_files,
                                       key=lambda x: -x["score"])[:20],
            "cross_system_insights": insights,
            "total_tested": sum(1 for g in gates if g.test_status == "passed"),
            "total_linked": sum(1 for g in gates if g.quantum_links),
            "mean_health": self._compute_mean_health(gates, links),
        }

    def _compute_mean_health(self, gates: List[LogicGate],
                             links: List[GateLink]) -> float:
        """Compute mean gate health score (0-1)."""
        if not gates:
            return 0.0
        scores = []
        for g in gates:
            complexity_score = min(1.0, g.complexity / 20.0)
            has_test = 1.0 if g.test_status == "passed" else 0.3
            has_doc = 1.0 if g.docstring else 0.5
            lc = sum(1 for l in links if g.name in (l.source_gate, l.target_gate))
            connectivity = min(1.0, lc * 0.1)
            h = (complexity_score * 0.2 + has_test * 0.25 + has_doc * 0.15
                 + connectivity * 0.2 + min(1.0, g.entropy_score) * 0.2)
            scores.append(h)
        return sum(scores) / len(scores)

    def _knowledge_synthesis(self, gates: List[LogicGate], links: List[GateLink],
                             anomalies: Dict, causal: Dict,
                             evolution: Dict, cross_poll: Dict) -> Dict:
        """Synthesize all research into a unified health score with adaptive learning."""
        insights = []
        risks = []

        ar = anomalies.get("anomaly_rate", 0)
        if ar > 0.05:
            risks.append(f"HIGH_ANOMALY: {ar:.1%} of gate measurements are outliers")
        elif ar < 0.01:
            insights.append("LOW_ANOMALY: Gate metrics are highly uniform")

        total_linked = cross_poll.get("total_linked", 0)
        total_gates = len(gates)
        link_ratio = total_linked / max(1, total_gates)
        if link_ratio > 0.5:
            insights.append(f"WELL_CONNECTED: {link_ratio:.0%} of gates have cross-file links")
        elif link_ratio < 0.1:
            risks.append(f"ISOLATED: Only {link_ratio:.0%} of gates have cross-file links")

        tested = cross_poll.get("total_tested", 0)
        test_ratio = tested / max(1, total_gates)
        if test_ratio > 0.3:
            insights.append(f"TESTED: {test_ratio:.0%} of gates have passing tests")
        else:
            risks.append(f"UNDERTESTED: Only {test_ratio:.0%} of gates tested")

        if evolution.get("has_history"):
            stability = evolution.get("stability", "unknown")
            if stability in ("highly_stable", "stable"):
                insights.append(f"STABLE_CODEBASE: {stability} gate evolution")
            elif stability == "high_churn":
                risks.append("HIGH_CHURN: Gates changing rapidly — stability concern")

        for c in causal.get("strong_correlations", []):
            insights.append(f"CAUSAL: {c['pair']} (r={c['correlation']:.3f})")

        prev_health = self.memory.get("last_research_health", 0)
        history = self.memory.get("health_history", [])

        mean_health = cross_poll.get("mean_health", 0.3)
        anomaly_factor = max(0, 1.0 - ar * 4)
        link_factor = min(1.0, link_ratio * 2)
        test_factor = min(1.0, test_ratio * 2)
        insight_factor = min(1.0, len(insights) / 5)
        stability_factor = (0.9 if evolution.get("stability") in ("highly_stable", "stable")
                            else 0.6 if evolution.get("stability") == "moderate_churn"
                            else 0.4)

        research_health = (
            mean_health * 0.25
            + anomaly_factor * 0.15
            + link_factor * 0.15
            + test_factor * 0.15
            + insight_factor * 0.10
            + stability_factor * 0.10
            + 0.10
        )
        risk_penalty = min(0.15, len(risks) * 0.03)
        research_health = min(1.0, max(0.05, research_health - risk_penalty))

        learning_trend = "unknown"
        if prev_health > 0:
            if research_health > prev_health + 0.01:
                learning_trend = "improving"
            elif research_health < prev_health - 0.01:
                learning_trend = "degrading"
            else:
                learning_trend = "stable"

        history.append(research_health)
        self.memory["last_research_health"] = research_health
        self.memory["health_history"] = history[-50:]
        self.memory["learning_trend"] = learning_trend

        return {
            "insights": insights,
            "risks": risks,
            "research_health": research_health,
            "learning_trend": learning_trend,
            "prev_health": prev_health,
        }

    def _record_snapshot(self, result: Dict):
        """Record research snapshot for self-learning."""
        snapshots = self.memory.get("snapshots", [])
        snapshots.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": result.get("research_health", 0),
            "total_gates": result.get("total_gates", 0),
            "anomaly_rate": result.get("anomaly_detection", {}).get("anomaly_rate", 0),
        })
        self.memory["snapshots"] = snapshots[-50:]


class StochasticGateResearchLab:
    """
    Random-based logic gate generation, research, and development engine.

    This R&D lab explores the frontier of logic gate design through stochastic
    experimentation — generating candidate gates via random exploration,
    validating them through deterministic tests, and merging successful designs
    into StochasticDeterministicGate hybrid entities.

    The lab implements a full research cycle:
      1. EXPLORE — Random generation of gate candidates with φ-bounded parameters
      2. VALIDATE — Deterministic evaluation against sacred constant coherence
      3. MERGE — Combine stochastic creativity with deterministic reliability
      4. CATALOG — Track all R&D iterations with full lineage

    Registered as invention entity: STOCHASTIC_DETERMINISTIC_GATE
    """

    RESEARCH_LOG_FILE = WORKSPACE_ROOT / ".l104_stochastic_gate_research.json"

    def __init__(self):
        """Initialize the stochastic gate research lab and load persisted log."""
        self.research_iterations: List[Dict[str, Any]] = []
        self.successful_gates: List[Dict[str, Any]] = []
        self.failed_experiments: List[Dict[str, Any]] = []
        self.generation_count: int = 0
        self._load_research_log()

    def _load_research_log(self):
        """Load persistent research log."""
        if self.RESEARCH_LOG_FILE.exists():
            try:
                data = json.loads(self.RESEARCH_LOG_FILE.read_text())
                self.research_iterations = data.get("iterations", [])
                self.successful_gates = data.get("successful", [])
                self.failed_experiments = data.get("failed", [])
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
                "total_successful": len(self.successful_gates),
                "total_failed": len(self.failed_experiments),
                "iterations": self.research_iterations[-200:],
                "successful": self.successful_gates[-100:],
                "failed": self.failed_experiments[-100:],
            }
            self.RESEARCH_LOG_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    # ─── PHASE 1: STOCHASTIC EXPLORATION ──────────────────────────

    def explore_gate_candidate(self, seed_concept: str = "quantum") -> Dict[str, Any]:
        """Generate a random gate candidate using φ-bounded stochastic parameters."""
        self.generation_count += 1
        gen_id = f"SG_{self.generation_count:06d}"

        phase_shift = random.uniform(0, 2 * math.pi) * PHI / (PHI + 1)
        amplitude = random.uniform(0.1, GOD_CODE * 0.01) * TAU
        harmonic_order = random.randint(1, CALABI_YAU_DIM)
        interference_mode = random.choice(["constructive", "destructive", "superposition"])
        grover_depth = random.randint(1, 7)
        entanglement_strength = random.uniform(0, 1.0) * PHI / 2

        exploration_seed = random.random() * GOD_CODE
        resonance_key = hashlib.sha256(
            f"{seed_concept}_{gen_id}_{exploration_seed}".encode()
        ).hexdigest()[:12]

        gate_body = self._synthesize_gate_body(
            phase_shift, amplitude, harmonic_order,
            interference_mode, grover_depth, entanglement_strength
        )

        candidate = {
            "gate_id": gen_id,
            "resonance_key": resonance_key,
            "seed_concept": seed_concept,
            "parameters": {
                "phase_shift": phase_shift,
                "amplitude": amplitude,
                "harmonic_order": harmonic_order,
                "interference_mode": interference_mode,
                "grover_depth": grover_depth,
                "entanglement_strength": entanglement_strength,
            },
            "exploration_seed": exploration_seed,
            "gate_function": gate_body,
            "generation": self.generation_count,
            "validated": False,
            "merged": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return candidate

    def _synthesize_gate_body(self, phase: float, amp: float,
                               order: int, mode: str, depth: int,
                               entangle: float) -> str:
        """Synthesize a gate function body from stochastic parameters."""
        func_name = f"stochastic_gate_{self.generation_count}"

        if mode == "constructive":
            interference = f"abs(math.cos(x * {phase:.6f})) * {amp:.6f}"
        elif mode == "destructive":
            interference = f"math.sin(x * {phase:.6f}) * {amp:.6f} * (1 - abs(math.cos(x * {entangle:.4f})))"
        else:
            interference = f"(math.cos(x * {phase:.6f}) + math.sin(x * {entangle:.4f})) * {amp:.6f} / 2"

        code = (
            f"def {func_name}(x):\n"
            f"    # STOCHASTIC_GATE | Order: {order} | Depth: {depth} | Mode: {mode}\n"
            f"    grover_gain = {PHI} ** {depth}\n"
            f"    harmonic = sum(math.sin(x * math.pi * k / {order}) * ({PHI} ** k / {PHI} ** {order}) for k in range(1, {order + 1}))\n"
            f"    interference = {interference}\n"
            f"    return (x * grover_gain + harmonic + interference) / (1 + abs(harmonic))\n"
        )
        return code

    # ─── PHASE 2: DETERMINISTIC VALIDATION ────────────────────────

    def validate_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a stochastic gate candidate through deterministic tests."""
        test_results = {
            "gate_id": candidate["gate_id"],
            "tests_run": 0,
            "tests_passed": 0,
            "details": [],
        }

        params = candidate["parameters"]
        phase = params["phase_shift"]
        amp = params["amplitude"]
        depth = params["grover_depth"]
        order = params["harmonic_order"]
        mode = params["interference_mode"]
        entangle = params["entanglement_strength"]

        def evaluate(x):
            grover_gain = PHI ** depth
            harmonic = sum(
                math.sin(x * math.pi * k / order) * (PHI ** k / PHI ** order)
                for k in range(1, order + 1)
            )
            if mode == "constructive":
                interference = abs(math.cos(x * phase)) * amp
            elif mode == "destructive":
                interference = math.sin(x * phase) * amp * (1 - abs(math.cos(x * entangle)))
            else:
                interference = (math.cos(x * phase) + math.sin(x * entangle)) * amp / 2
            return (x * grover_gain + harmonic + interference) / (1 + abs(harmonic))

        # Test 1: Numerical stability
        test_results["tests_run"] += 1
        test_inputs = [0.0, 1.0, PHI, -PHI, GOD_CODE * 0.001, math.pi, 100.0, -100.0]
        stable = True
        for inp in test_inputs:
            try:
                result = evaluate(inp)
                if math.isnan(result) or math.isinf(result):
                    stable = False
                    break
            except Exception:
                stable = False
                break
        if stable:
            test_results["tests_passed"] += 1
        test_results["details"].append({"test": "numerical_stability", "passed": stable})

        # Test 2: Sacred constant coherence
        test_results["tests_run"] += 1
        try:
            phi_result = evaluate(PHI)
            coherent = not math.isnan(phi_result) and not math.isinf(phi_result) and abs(phi_result) > 1e-10
        except Exception:
            coherent = False
        if coherent:
            test_results["tests_passed"] += 1
        test_results["details"].append({"test": "sacred_coherence", "passed": coherent})

        # Test 3: Zero-input behavior
        test_results["tests_run"] += 1
        try:
            zero_result = evaluate(0.0)
            zero_bounded = abs(zero_result) < amp * 10 + 1.0
        except Exception:
            zero_bounded = False
        if zero_bounded:
            test_results["tests_passed"] += 1
        test_results["details"].append({"test": "zero_bounded", "passed": zero_bounded})

        # Test 4: φ-alignment
        test_results["tests_run"] += 1
        try:
            r1 = evaluate(PHI)
            r2 = evaluate(PHI * 2)
            if abs(r1) > 1e-10:
                ratio = abs(r2 / r1)
                phi_aligned = 0.1 < ratio < 50.0
            else:
                phi_aligned = abs(r2) < 100.0
        except Exception:
            phi_aligned = False
        if phi_aligned:
            test_results["tests_passed"] += 1
        test_results["details"].append({"test": "phi_alignment", "passed": phi_aligned})

        # Final verdict
        pass_rate = test_results["tests_passed"] / max(test_results["tests_run"], 1)
        test_results["pass_rate"] = pass_rate
        test_results["verdict"] = "VALIDATED" if pass_rate >= 0.75 else "REJECTED"

        candidate["validated"] = pass_rate >= 0.75
        candidate["validation_results"] = test_results

        return test_results

    # ─── PHASE 3: HYBRID MERGE ────────────────────────────────────

    def merge_to_hybrid(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Merge a validated stochastic candidate with deterministic gate logic."""
        if not candidate.get("validated"):
            return None

        params = candidate["parameters"]
        gate_id = candidate["gate_id"]
        resonance_key = candidate["resonance_key"]

        hybrid_function = (
            f"def hybrid_{gate_id}(value):\n"
            f"    # STOCHASTIC_DETERMINISTIC_GATE | Resonance: {resonance_key}\n"
            f"    # Phase 1: Stochastic exploration\n"
            f"    phase = {params['phase_shift']:.6f}\n"
            f"    amp = {params['amplitude']:.6f}\n"
            f"    depth = {params['grover_depth']}\n"
            f"    grover_gain = {PHI} ** depth\n"
            f"    stochastic_output = value * grover_gain + math.sin(value * phase) * amp\n"
            f"    # Phase 2: Deterministic alignment via sage gate\n"
            f"    aligned = sage_logic_gate(stochastic_output, 'align')\n"
            f"    # Phase 3: Quantum verification\n"
            f"    verified = quantum_logic_gate(aligned, depth=min(depth, 5))\n"
            f"    return verified\n"
        )

        hybrid_entity = {
            "entity_type": "STOCHASTIC_DETERMINISTIC_GATE",
            "gate_id": f"HYBRID_{gate_id}",
            "resonance_key": resonance_key,
            "origin_candidate": gate_id,
            "origin_parameters": params,
            "hybrid_function": hybrid_function,
            "complexity_score": (
                params["grover_depth"] * params["harmonic_order"] *
                params["entanglement_strength"] * PHI
            ),
            "creation_method": "stochastic_exploration → deterministic_validation → hybrid_merge",
            "verified": True,
            "tags": ["INVENTION", "NEOTERIC", "STOCHASTIC_DETERMINISTIC", "HYBRID_GATE"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        candidate["merged"] = True
        self.successful_gates.append(hybrid_entity)

        return hybrid_entity

    # ─── PHASE 4: FULL R&D CYCLE ──────────────────────────────────

    def run_rd_cycle(self, seed_concepts: Optional[List[str]] = None,
                     iterations: int = 10) -> Dict[str, Any]:
        """Run a complete R&D cycle: explore → validate → merge → catalog."""
        if seed_concepts is None:
            seed_concepts = [
                "quantum", "consciousness", "entropy", "resonance",
                "harmonic", "emergence", "transcendence"
            ]

        cycle_results = {
            "cycle_id": f"RD_{int(time.time())}",
            "seed_concepts": seed_concepts,
            "iterations_per_concept": iterations,
            "total_candidates": 0,
            "total_validated": 0,
            "total_merged": 0,
            "total_rejected": 0,
            "hybrid_entities": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for concept in seed_concepts:
            for _ in range(iterations):
                candidate = self.explore_gate_candidate(concept)
                cycle_results["total_candidates"] += 1

                validation = self.validate_candidate(candidate)

                iteration_record = {
                    "gate_id": candidate["gate_id"],
                    "concept": concept,
                    "parameters": candidate["parameters"],
                    "validation": validation["verdict"],
                    "pass_rate": validation["pass_rate"],
                }
                self.research_iterations.append(iteration_record)

                if candidate["validated"]:
                    cycle_results["total_validated"] += 1
                    hybrid = self.merge_to_hybrid(candidate)
                    if hybrid:
                        cycle_results["total_merged"] += 1
                        cycle_results["hybrid_entities"].append(hybrid["gate_id"])
                else:
                    cycle_results["total_rejected"] += 1
                    self.failed_experiments.append({
                        "gate_id": candidate["gate_id"],
                        "reason": validation["verdict"],
                        "pass_rate": validation["pass_rate"],
                    })

        total = cycle_results["total_candidates"]
        cycle_results["success_rate"] = cycle_results["total_merged"] / max(total, 1)
        cycle_results["phi_efficiency"] = cycle_results["success_rate"] * PHI

        self._save_research_log()

        return cycle_results

    def get_best_hybrid_gates(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return the highest-scoring hybrid gates from all R&D cycles."""
        sorted_gates = sorted(
            self.successful_gates,
            key=lambda g: g.get("complexity_score", 0),
            reverse=True
        )
        return sorted_gates[:top_n]

    def research_summary(self) -> Dict[str, Any]:
        """Comprehensive R&D summary across all cycles."""
        return {
            "total_generations": self.generation_count,
            "total_iterations": len(self.research_iterations),
            "total_successful": len(self.successful_gates),
            "total_failed": len(self.failed_experiments),
            "overall_success_rate": len(self.successful_gates) / max(self.generation_count, 1),
            "best_gates": [g["gate_id"] for g in self.get_best_hybrid_gates(5)],
            "phi_discovery_index": len(self.successful_gates) * PHI / max(self.generation_count, 1),
        }
