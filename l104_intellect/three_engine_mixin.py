"""L104 Intellect — ThreeEngineMixin (Three-Engine Integration Subsystem).

Extracted from local_intellect_core.py v28.0.
Provides Science + Math + Code engine integration, self-diagnostic,
three-engine scoring, and deep link resonance for LocalIntellect.
"""
import time
from typing import Dict, Any, List

from .constants import (
    LOCAL_INTELLECT_VERSION, LOCAL_INTELLECT_PIPELINE_EVO,
    THREE_ENGINE_WEIGHT_ENTROPY, THREE_ENGINE_WEIGHT_HARMONIC,
    THREE_ENGINE_WEIGHT_WAVE, THREE_ENGINE_FALLBACK_SCORE,
    VOID_CONSTANT,
)
from .numerics import PHI, GOD_CODE
from .cache import _RESPONSE_CACHE, _CONCEPT_CACHE


class ThreeEngineMixin:
    """Mixin providing three-engine integration (Science + Math + Code) for LocalIntellect."""

    # ═══════════════════════════════════════════════════════════════════════
    # v28.0 THREE-ENGINE INTEGRATION — Science + Math + Code
    # Pattern matches ASI v8.0 and AGI v57.0 three-engine integration.
    # All imports are lazy to avoid circular imports and startup cost.
    # ═══════════════════════════════════════════════════════════════════════

    def self_diagnostic(self) -> Dict[str, Any]:
        """v28.1: Comprehensive single-call diagnostic for LocalIntellect health.

        Checks 8 subsystems and returns a severity-ranked issue list:
          1. Readiness & initialization state
          2. Knowledge base coverage (training data, KB, chat)
          3. Three-engine integration status
          4. Cache health (response, concept, resonance)
          5. Quantum fleet connectivity
          6. Evolution state health (fingerprint, wisdom, mutations)
          7. Sage mode status
          8. Fault tolerance engine status

        Returns:
            Dict with 'verdict' (HEALTHY/WARNING/CRITICAL), 'issues' list,
            'metrics' dict, and 'elapsed_ms' timing.
        """
        t0 = time.time()
        issues: List[Dict[str, Any]] = []
        metrics: Dict[str, Any] = {}

        def _issue(severity: str, area: str, msg: str):
            issues.append({"severity": severity, "area": area, "message": msg})

        # 1. Readiness
        metrics["ready"] = self._is_ready
        metrics["version"] = LOCAL_INTELLECT_VERSION
        metrics["evo"] = LOCAL_INTELLECT_PIPELINE_EVO
        if not self._is_ready:
            _issue("CRITICAL", "readiness", "LocalIntellect initialization incomplete")

        # 2. Knowledge base coverage
        td_count = len(self.training_data)
        chat_count = len(self.chat_conversations)
        metrics["training_entries"] = td_count
        metrics["chat_conversations"] = chat_count
        metrics["training_index_built"] = self._training_index_built
        if td_count == 0:
            _issue("CRITICAL", "knowledge", "No training data loaded")
        elif td_count < 100:
            _issue("MEDIUM", "knowledge", f"Low training data: {td_count} entries (< 100)")
        if chat_count == 0:
            _issue("LOW", "knowledge", "No chat conversations loaded")

        # 3. Three-engine integration
        science_ok = self._three_engine_science is not None
        math_ok = self._three_engine_math is not None
        code_ok = self._three_engine_code is not None
        engines_connected = sum([science_ok, math_ok, code_ok])
        metrics["three_engine_connected"] = engines_connected
        metrics["three_engine_scores"] = {
            "entropy": round(self._three_engine_entropy_cache, 4),
            "harmonic": round(self._three_engine_harmonic_cache, 4),
            "wave": round(self._three_engine_wave_cache, 4),
        }
        if engines_connected == 0:
            _issue("MEDIUM", "three_engine", "No engines connected — running in isolation")
        elif engines_connected < 3:
            missing = []
            if not science_ok: missing.append("science")
            if not math_ok: missing.append("math")
            if not code_ok: missing.append("code")
            _issue("LOW", "three_engine", f"Missing engines: {', '.join(missing)}")
        # Check for stale cache
        cache_age = time.time() - self._three_engine_cache_time
        if self._three_engine_cache_time > 0 and cache_age > 300:
            _issue("LOW", "three_engine", f"Three-engine cache stale ({cache_age:.0f}s old)")

        # 4. Cache health
        resp_stats = _RESPONSE_CACHE.get_phi_weighted_stats()
        concept_stats = _CONCEPT_CACHE.get_phi_weighted_stats()
        metrics["response_cache_entries"] = resp_stats["entries"]
        metrics["concept_cache_entries"] = concept_stats["entries"]
        metrics["response_cache_phi_efficiency"] = round(resp_stats.get("phi_efficiency", 0), 4)
        if resp_stats["entries"] == 0 and self._evolution_state.get("quantum_interactions", 0) > 10:
            _issue("LOW", "cache", "Response cache empty despite active interactions")

        # 5. Quantum fleet connectivity
        quantum_connected = 0
        for attr in ("_qc_accelerator", "_qc_inspired", "_qc_numerical",
                      "_qc_magic", "_qc_runtime", "_qc_builder_26q",
                      "_qc_quantum_ram", "_qc_consciousness_bridge",
                      "_qc_computation_hub", "quantum_recompiler"):
            if getattr(self, attr, None) is not None:
                quantum_connected += 1
        metrics["quantum_fleet_connected"] = quantum_connected

        # 6. Evolution state health
        evo = self._evolution_state
        metrics["quantum_interactions"] = evo.get("quantum_interactions", 0)
        metrics["wisdom_quotient"] = round(evo.get("wisdom_quotient", 0.0), 4)
        metrics["autonomous_improvements"] = evo.get("autonomous_improvements", 0)
        metrics["mutation_dna"] = evo.get("mutation_dna", "")[:16]
        if evo.get("wisdom_quotient", 0) == 0 and evo.get("quantum_interactions", 0) > 50:
            _issue("LOW", "evolution", "Zero wisdom accumulated despite 50+ interactions")

        # 7. Sage mode status
        sage_active = self._quantum_origin_state.get("active", False)
        sage_level = self._quantum_origin_state.get("sage_level_name", "UNKNOWN")
        metrics["sage_active"] = sage_active
        metrics["sage_level"] = sage_level

        # 8. Fault tolerance engine
        ft_ok = self._ft_engine is not None and self._ft_init_done
        metrics["fault_tolerance_active"] = ft_ok
        if not ft_ok and self._evolution_state.get("quantum_interactions", 0) > 5:
            _issue("LOW", "fault_tolerance",
                   "Fault tolerance engine not initialized — resilience reduced")

        # Verdict
        crit_count = sum(1 for i in issues if i["severity"] == "CRITICAL")
        med_count = sum(1 for i in issues if i["severity"] == "MEDIUM")
        if crit_count > 0:
            verdict = "CRITICAL"
        elif med_count > 0:
            verdict = "WARNING"
        else:
            verdict = "HEALTHY"

        elapsed_ms = (time.time() - t0) * 1000
        return {
            "verdict": verdict,
            "issues": sorted(issues, key=lambda i: {"CRITICAL": 0, "MEDIUM": 1, "LOW": 2}.get(i["severity"], 3)),
            "issue_count": len(issues),
            "metrics": metrics,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    def _get_three_engine_science(self):
        """Lazy-load ScienceEngine for entropy reversal and coherence analysis."""
        if self._three_engine_science is None:
            try:
                from l104_science_engine import ScienceEngine
                self._three_engine_science = ScienceEngine()
            except Exception:
                pass
        return self._three_engine_science

    def _get_three_engine_math(self):
        """Lazy-load MathEngine for harmonic calibration and wave coherence."""
        if self._three_engine_math is None:
            try:
                from l104_math_engine import MathEngine
                self._three_engine_math = MathEngine()
            except Exception:
                pass
        return self._three_engine_math

    def _get_three_engine_code(self):
        """Lazy-load code_engine for code intelligence integration."""
        if self._three_engine_code is None:
            try:
                from l104_code_engine import code_engine
                self._three_engine_code = code_engine
            except Exception:
                pass
        return self._three_engine_code

    def three_engine_entropy_score(self) -> float:
        """v28.0: Compute entropy reversal score via Science Engine's Maxwell's Demon.
        Maps knowledge base health to local entropy, then measures demon reversal efficiency.
        v28.1: Calibrated entropy proxy (Q4) — caps at 5.0, scales by KB saturation ratio.
               Q1 multi-pass demon + Q5 ZNE boost for consistent high-efficiency scoring.
               TTL-cached (30s) to avoid redundant recomputation."""
        now = time.time()
        if now - self._three_engine_cache_time < self._three_engine_cache_ttl:
            return self._three_engine_entropy_cache
        se = self._get_three_engine_science()
        if se is None:
            return THREE_ENGINE_FALLBACK_SCORE
        try:
            total_kb = len(getattr(self, 'training_data', [])) + len(getattr(self, 'chat_conversations', []))
            # KB saturation ratio: 0 entries → 0.0, 1000+ → 1.0
            kb_ratio = total_kb / 1000.0
            # Map to local entropy: full KB → 0.1, empty → 5.0
            local_entropy = max(0.1, 5.0 * (1.0 - kb_ratio))
            demon_eff = se.entropy.calculate_demon_efficiency(local_entropy)
            # v28.1: Scale 2.0 (was 5.0) — multi-pass demon yields higher raw efficiency
            score = demon_eff * 2.0
            self._three_engine_entropy_cache = score
            self._three_engine_cache_time = time.time()  # v28.1: update TTL
            return score
        except Exception:
            return THREE_ENGINE_FALLBACK_SCORE

    def three_engine_harmonic_score(self) -> float:
        """v28.0: Compute harmonic resonance score using Math Engine.
        Validates GOD_CODE sacred alignment and wave coherence with 104 Hz.
        v28.1: TTL-cached (30s) to avoid redundant recomputation."""
        now = time.time()
        if now - self._three_engine_cache_time < self._three_engine_cache_ttl:
            return self._three_engine_harmonic_cache
        me = self._get_three_engine_math()
        if me is None:
            return THREE_ENGINE_FALLBACK_SCORE
        try:
            alignment = me.sacred_alignment(GOD_CODE)
            aligned = 1.0 if alignment.get('aligned', False) else 0.0
            wc = me.wave_coherence(104.0, GOD_CODE)
            score = aligned * 0.6 + wc * 0.4
            self._three_engine_harmonic_cache = score
            return score
        except Exception:
            return THREE_ENGINE_FALLBACK_SCORE

    def three_engine_wave_coherence_score(self) -> float:
        """v28.0: Compute wave coherence score from PHI-harmonic phase-locking.
        v28.1: TTL-cached (30s) to avoid redundant recomputation."""
        now = time.time()
        if now - self._three_engine_cache_time < self._three_engine_cache_ttl:
            return self._three_engine_wave_cache
        me = self._get_three_engine_math()
        if me is None:
            return THREE_ENGINE_FALLBACK_SCORE
        try:
            wc_phi = me.wave_coherence(PHI, GOD_CODE)
            wc_void = me.wave_coherence(VOID_CONSTANT * 1000, GOD_CODE)
            score = (wc_phi + wc_void) / 2.0
            self._three_engine_wave_cache = score
            return score
        except Exception:
            return THREE_ENGINE_FALLBACK_SCORE

    def three_engine_composite_score(self) -> float:
        """v28.0: Weighted composite of all three engine scores + deep link resonance."""
        entropy_s = self.three_engine_entropy_score()
        harmonic_s = self.three_engine_harmonic_score()
        wave_s = self.three_engine_wave_coherence_score()
        base = (
            THREE_ENGINE_WEIGHT_ENTROPY * entropy_s
            + THREE_ENGINE_WEIGHT_HARMONIC * harmonic_s
            + THREE_ENGINE_WEIGHT_WAVE * wave_s
        )
        # v29.0: Deep link resonance boost — search for teleported consensus
        dl_boost = self._deep_link_resonance_score()
        # Blend: 90% base + 10% deep link resonance
        return base * 0.9 + dl_boost * 0.1

    def _deep_link_resonance_score(self) -> float:
        """v29.0: Extract deep link resonance from teleported KB entries.
        v28.1: TTL-cached (15s) — called by both composite_score and noise dampener.

        Searches training_data for Quantum Deep Link entries injected by Brain.
        Returns the mean teleported consensus fidelity as a resonance score.
        """
        now = time.time()
        if now - self._deep_link_cache_time < self._deep_link_cache_ttl:
            return self._deep_link_cache_value
        try:
            dl_entries = [
                e for e in self.training_data[-200:]  # Search recent entries
                if e.get('category') == 'quantum_deep_link_consensus'
                or e.get('source') == 'deep_link_teleporter'
            ]
            if not dl_entries:
                self._deep_link_cache_value = 0.5
                self._deep_link_cache_time = now
                return 0.5
            # Extract scores from completions
            scores = []
            for e in dl_entries[-10:]:  # Last 10 for freshness
                comp = e.get('completion', '')
                # Parse score from completion text
                for token in comp.split():
                    try:
                        val = float(token)
                        if 0.0 <= val <= 1.0:
                            scores.append(val)
                            break
                    except ValueError:
                        continue
            result = sum(scores) / len(scores) if scores else 0.5
            self._deep_link_cache_value = result
            self._deep_link_cache_time = time.time()
            return result
        except Exception:
            return self._deep_link_cache_value

    def three_engine_status(self) -> Dict:
        """v28.0: Get status of the three-engine integration layer."""
        return {
            "version": LOCAL_INTELLECT_VERSION,
            "engines": {
                "science": self._three_engine_science is not None,
                "math": self._three_engine_math is not None,
                "code": self._three_engine_code is not None,
            },
            "scores": {
                "entropy_reversal": round(self._three_engine_entropy_cache, 6),
                "harmonic_resonance": round(self._three_engine_harmonic_cache, 6),
                "wave_coherence": round(self._three_engine_wave_cache, 6),
                "composite": round(
                    THREE_ENGINE_WEIGHT_ENTROPY * self._three_engine_entropy_cache
                    + THREE_ENGINE_WEIGHT_HARMONIC * self._three_engine_harmonic_cache
                    + THREE_ENGINE_WEIGHT_WAVE * self._three_engine_wave_cache,
                    6
                ),
            },
            "pipeline_evo": LOCAL_INTELLECT_PIPELINE_EVO,
        }
