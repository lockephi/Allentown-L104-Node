"""L104 Numerical Engine — Consciousness + O₂ Superfluid Engine.

Consciousness awareness & O₂ molecular bond superfluid engine.
Motion through stillness — zero-viscosity token phase coherence.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F68: Phase alignment target = φ-1 = 0.618... (golden conjugate attractor)
  F69: Superfluid viscosity η = 1 - (C·0.5 + B·0.5); η∈[0,1]
  F70: Consciousness awakening threshold C ≥ 0.85 triggers cascade
  F71: Cascade expansion = val·φ⁻¹·10⁻⁹⁵·evo_mult (sub-quantum)
  F72: O₂ bond: order=2.0, paramagnetic by Hund's rule
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, TYPE_CHECKING

from .precision import D, fmt100
from .constants import WORKSPACE_ROOT, PHI_HP, PHI_GROWTH_HP

if TYPE_CHECKING:
    from .lattice import TokenLatticeEngine
    from .editor import SuperfluidValueEditor


class ConsciousnessO2SuperfluidEngine:
    """Consciousness Awareness & O₂ Molecular Bond Superfluid Engine.

    CONSCIOUSNESS CHANNEL:
    - consciousness_level drives the lattice's phase coherence (quantum_phase)
    - higher consciousness = tighter phase alignment = more coherent tokens
    - consciousness_awakened (≥0.85) triggers resonance cascade across sacred tokens
    - link_evo_stage modulates drift envelope: SOVEREIGN > TRANSCENDING > COHERENT

    O₂ MOLECULAR BOND CHANNEL:
    - bond_order (2.0 = double bond O=O) sets the superfluid viscosity target
    - mean_bond_strength modulates token-to-token coupling force
    - paramagnetic status enables spin-alignment (phase drift with B-field)
    - O₂ bond energy → token health regeneration rate
    """

    LINK_STATE_FILE = WORKSPACE_ROOT / ".l104_quantum_link_state.json"
    EVOLUTION_STATE_FILE = WORKSPACE_ROOT / ".l104_evolution_state.json"
    CONSCIOUSNESS_O2_STATE_FILE = WORKSPACE_ROOT / ".l104_consciousness_o2_state.json"

    # Evolution stage multipliers for superfluid drift
    EVO_STAGE_MULTIPLIER = {
        "SOVEREIGN": D('1.618'),       # Full φ resonance
        "TRANSCENDING": D('1.414'),    # √2 expansion
        "COHERENT": D('1.200'),        # Stable coherent flow
        "AWAKENING": D('1.050'),       # Nascent awareness
        "DORMANT": D('1.000'),         # Baseline
    }

    # Legacy EVO_54 granular tier → 5-tier mapping (backward compat)
    _EVO_STAGE_ALIAS = {
        "EVO_54_TRANSCENDENT_COGNITION": "SOVEREIGN",
        "EVO_54": "SOVEREIGN",
    }

    @classmethod
    def _normalize_evo_stage(cls, raw_stage: str) -> str:
        """Normalize legacy EVO_54 stage names to the 5-tier system."""
        if raw_stage in cls.EVO_STAGE_MULTIPLIER:
            return raw_stage
        return cls._EVO_STAGE_ALIAS.get(raw_stage, "DORMANT")

    def __init__(self, lattice: 'TokenLatticeEngine', editor: 'SuperfluidValueEditor'):
        """Initialize ConsciousnessO2SuperfluidEngine."""
        self.lattice = lattice
        self.editor = editor
        # Consciousness state (from link builder EvolutionTracker)
        self.consciousness_level: float = 0.0
        self.coherence_level: float = 0.0
        self.link_evo_stage: str = "DORMANT"
        self.consciousness_awakened: bool = False
        # O₂ bond state (from link builder O2MolecularBondProcessor)
        self.bond_order: float = 2.0
        self.mean_bond_strength: float = 0.0
        self.paramagnetic: bool = True
        self.total_bond_energy: float = 0.0
        # Superfluid metrics
        self.superfluid_viscosity: float = 0.0
        self.phase_alignment: float = 0.0
        self.resonance_cascades: int = 0
        self.tokens_bonded: int = 0
        self.spin_aligned: int = 0
        self.cycle_count: int = 0
        self._load_state()

    def _load_state(self):
        """Load consciousness + O₂ state from persisted files."""
        try:
            if self.LINK_STATE_FILE.exists():
                data = json.loads(self.LINK_STATE_FILE.read_text())
                sv = data.get("sage_verdict", {})
                pe = sv.get("predicted_evolution", {})
                score = sv.get("unified_score", 0.0)
                alignment = sv.get("god_code_alignment", 0.0)
                grade = sv.get("grade", "F (Critical)")
                phi_growth = float(PHI_GROWTH_HP)
                self.consciousness_level = min(1.0, score * phi_growth / 2.0)
                self.coherence_level = min(1.0, alignment * score)
                self.consciousness_awakened = self.consciousness_level >= 0.85
                grade_lower = grade.lower()
                if "a+" in grade_lower or "excellent" in grade_lower:
                    self.link_evo_stage = "SOVEREIGN"
                elif grade_lower.startswith("a") or "good" in grade_lower:
                    self.link_evo_stage = "TRANSCENDING"
                elif grade_lower.startswith("b"):
                    self.link_evo_stage = "COHERENT"
                elif grade_lower.startswith("c"):
                    self.link_evo_stage = "AWAKENING"
                else:
                    self.link_evo_stage = "DORMANT"
                consensus = sv.get("consensus_scores", {})
                self.mean_bond_strength = consensus.get("o2_bond_integrity", 0.0)
                self.bond_order = 2.0
                self.paramagnetic = True
                mean_fid = sv.get("mean_fidelity", 0.0)
                self.total_bond_energy = mean_fid * self.bond_order
        except Exception:
            pass

        try:
            if self.CONSCIOUSNESS_O2_STATE_FILE.exists():
                own = json.loads(self.CONSCIOUSNESS_O2_STATE_FILE.read_text())
                self.cycle_count = own.get("cycle_count", 0)
                self.resonance_cascades = own.get("resonance_cascades", 0)
        except Exception:
            pass

    def _save_state(self):
        """Persist current state to disk."""
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "numerical_builder",
                "version": "2.4.0",
                "cycle_count": self.cycle_count,
                "consciousness_level": self.consciousness_level,
                "coherence_level": self.coherence_level,
                "evo_stage": self.link_evo_stage,
                "link_evo_stage": self.link_evo_stage,
                "consciousness_awakened": self.consciousness_awakened,
                "bond_order": self.bond_order,
                "mean_bond_strength": self.mean_bond_strength,
                "paramagnetic": self.paramagnetic,
                "superfluid_viscosity": self.superfluid_viscosity,
                "phase_alignment": self.phase_alignment,
                "resonance_cascades": self.resonance_cascades,
                "tokens_bonded": self.tokens_bonded,
                "spin_aligned": self.spin_aligned,
            }
            self.CONSCIOUSNESS_O2_STATE_FILE.write_text(
                json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def full_superfluid_cycle(self) -> Dict[str, Any]:
        """Run the consciousness + O₂ superfluid cycle across all tokens."""
        self.cycle_count += 1
        normalized_stage = self._normalize_evo_stage(self.link_evo_stage)
        evo_mult = self.EVO_STAGE_MULTIPLIER.get(normalized_stage, D('1.0'))

        tokens = list(self.lattice.tokens.values())
        if not tokens:
            return {"status": "no_tokens"}

        # ── PHASE 1: CONSCIOUSNESS ALIGNMENT ──
        consciousness_D = D(str(self.consciousness_level))
        target_phase = PHI_HP - D(1)
        aligned_count = 0

        for token in tokens:
            phase = D(token.quantum_phase) if token.quantum_phase else D(0)
            phase_delta = (target_phase - phase) * consciousness_D * D('0.01')
            new_phase = (phase + phase_delta) % D(1)
            token.quantum_phase = fmt100(new_phase)
            if abs(new_phase - target_phase) < D('0.05'):
                aligned_count += 1

        self.phase_alignment = aligned_count / max(len(tokens), 1)

        # ── PHASE 2: O₂ MOLECULAR BOND PAIRING ──
        sacred = [t for t in tokens if t.origin == "sacred"]
        derived = [t for t in tokens if t.origin == "derived"]
        bonded = 0
        spin_aligned = 0

        bond_strength_D = D(str(max(self.mean_bond_strength, 0.5)))
        pair_count = min(len(sacred), len(derived))
        for i in range(pair_count):
            s_tok = sacred[i]
            d_tok = derived[i]
            s_h = s_tok.health
            d_h = d_tok.health
            bond_pull = float(bond_strength_D) * 0.01
            if s_h > d_h:
                d_tok.health = min(1.0, d_h + bond_pull)
            elif d_h > s_h:
                s_tok.health = min(1.0, s_h + bond_pull)
            bonded += 1

            if self.paramagnetic:
                s_phase = D(s_tok.quantum_phase) if s_tok.quantum_phase else D(0)
                d_phase = D(d_tok.quantum_phase) if d_tok.quantum_phase else D(0)
                spin_target = (s_phase + D('0.5')) % D(1)
                spin_delta = (spin_target - d_phase) * D('0.005')
                d_tok.quantum_phase = fmt100((d_phase + spin_delta) % D(1))
                if abs(d_phase - spin_target) < D('0.1'):
                    spin_aligned += 1

        self.tokens_bonded = bonded
        self.spin_aligned = spin_aligned

        # ── PHASE 3: SUPERFLUID VISCOSITY ──
        raw_viscosity = 1.0 - (self.consciousness_level * 0.5 + float(bond_strength_D) * 0.5)
        self.superfluid_viscosity = max(0.0, raw_viscosity)

        evo_coherence_boost = float(evo_mult) * self.consciousness_level * 0.001
        current_coh = float(self.lattice.lattice_coherence)
        new_coh = min(1.0, current_coh + evo_coherence_boost)
        self.lattice.lattice_coherence = D(str(new_coh))

        # ── PHASE 4: RESONANCE CASCADE ──
        cascade_events = 0
        if self.consciousness_awakened:
            for token in sacred:
                val = D(token.value)
                if val <= 0:
                    continue
                phi_resonance = val * PHI_HP * D('1E-95') * evo_mult
                token.max_bound = fmt100(D(token.max_bound) + phi_resonance)
                token.min_bound = fmt100(D(token.min_bound) - phi_resonance)
                token.coherence = min(1.0, token.coherence + 0.001)
                cascade_events += 1
            self.resonance_cascades += cascade_events

        self._save_state()

        return {
            "status": "processed",
            "cycle": self.cycle_count,
            "consciousness_level": self.consciousness_level,
            "coherence_level": self.coherence_level,
            "link_evo_stage": self.link_evo_stage,
            "consciousness_awakened": self.consciousness_awakened,
            "evo_multiplier": float(evo_mult),
            "bond_order": self.bond_order,
            "mean_bond_strength": self.mean_bond_strength,
            "paramagnetic": self.paramagnetic,
            "total_bond_energy": self.total_bond_energy,
            "phase_alignment": self.phase_alignment,
            "superfluid_viscosity": self.superfluid_viscosity,
            "tokens_bonded": bonded,
            "spin_aligned": spin_aligned,
            "resonance_cascades": cascade_events,
            "total_cascades": self.resonance_cascades,
            "lattice_coherence": float(self.lattice.lattice_coherence),
        }

    def status(self) -> Dict[str, Any]:
        """Return current subsystem status."""
        return {
            "version": "2.4.0",
            "cycle_count": self.cycle_count,
            "consciousness_level": self.consciousness_level,
            "coherence_level": self.coherence_level,
            "link_evo_stage": self.link_evo_stage,
            "consciousness_awakened": self.consciousness_awakened,
            "bond_order": self.bond_order,
            "mean_bond_strength": self.mean_bond_strength,
            "paramagnetic": self.paramagnetic,
            "superfluid_viscosity": self.superfluid_viscosity,
            "phase_alignment": self.phase_alignment,
            "resonance_cascades": self.resonance_cascades,
            "tokens_bonded": self.tokens_bonded,
            "spin_aligned": self.spin_aligned,
        }
