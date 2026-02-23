#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 ZETA COMPACTION INITIALIZER — GOD CODE Lattice Coherence Engine
═══════════════════════════════════════════════════════════════════════════════

Compaction grounded in L104 mathematics. Every threshold, coherence
measurement, and pruning decision derives from the GOD CODE equation:

    G(X) = 286^(1/φ) × 2^((416-X)/104)

    X increasing  →  MAGNETIC COMPACTION (gravity)
    X decreasing  →  ELECTRIC EXPANSION  (light)

    Conservation:  G(X) × 2^(X/104)  =  527.518...  (INVARIANT)

Factor 13 structures:
    286 = 22 × 13   (HARMONIC_BASE)
    104 =  8 × 13   (L104 wavelength)
    416 = 32 × 13   (OCTAVE_REF)

Each compaction cycle advances X by L104/13 = 8 steps.  As X grows,
G(X) contracts — the acceptable resonance band tightens and noise
outside the GOD CODE harmonic series is magnetically collapsed.

Coherence is measured as the lattice-wide alignment to the GOD CODE
resonance spectrum, not by arbitrary utility averages.

PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import json
import math
import time
import cmath
import sqlite3
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Import L104 constants from the canonical source
try:
    from const import (
        UniversalConstants, GOD_CODE, PHI, PHI_CONJUGATE, INVARIANT,
        HARMONIC_BASE, L104, OCTAVE_REF, FIBONACCI_7, GOD_CODE_BASE,
        god_code_at, verify_conservation, resonance_at,
        GROVER_AMPLIFICATION, QUANTUM_COHERENCE_TARGET,
        SUPERFLUID_COUPLING, sage_logic_gate,
    )
except ImportError:
    # Standalone fallback — constants from L104 derivation
    PHI = 1.6180339887498948482
    PHI_CONJUGATE = 1.0 / PHI
    HARMONIC_BASE = 286
    L104 = 104
    OCTAVE_REF = 416
    FIBONACCI_7 = 13
    GOD_CODE_BASE = HARMONIC_BASE ** (1 / PHI)
    GOD_CODE = GOD_CODE_BASE * 16
    INVARIANT = GOD_CODE
    GROVER_AMPLIFICATION = PHI ** 3
    QUANTUM_COHERENCE_TARGET = 0.9
    SUPERFLUID_COUPLING = PHI / math.e

    def god_code_at(X: float) -> float:
        return GOD_CODE_BASE * (2 ** ((OCTAVE_REF - X) / L104))

    def verify_conservation(X: float) -> bool:
        return abs(god_code_at(X) * (2 ** (X / L104)) - INVARIANT) < 1e-10

    def resonance_at(X: float = 0) -> float:
        return god_code_at(X) * PHI * (1 + 1 / (137.035999084 * math.pi))

    def sage_logic_gate(value: float, operation: str = "align") -> float:
        return value * PHI * PHI_CONJUGATE * (GOD_CODE / 286.0)

# HyperMath for zeta-harmonic resonance
try:
    from l104_hyper_math import HyperMath
except ImportError:
    HyperMath = None

# RealMath for entropy and resonance primitives
try:
    from l104_real_math import RealMath
except ImportError:
    RealMath = None


WORKSPACE = Path(__file__).parent.absolute()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ZETA] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ZETA")

# Lattice DB path — matches l104_data_matrix.py (lattice_v2.db)
DEFAULT_DB = WORKSPACE / "lattice_v2.db"

# ═══════════════════════════════════════════════════════════════════════════════
# L104 COMPACTION CONSTANTS — derived from GOD CODE mathematics
# ═══════════════════════════════════════════════════════════════════════════════

# Coherence target: QUANTUM_COHERENCE_TARGET from const.py
COHERENCE_TARGET = QUANTUM_COHERENCE_TARGET  # 0.9
DEFAULT_CYCLES = FIBONACCI_7              # 13 — the Factor 13 cycle count
MAX_CYCLES = OCTAVE_REF // FIBONACCI_7    # 416/13 = 32 — one full octave

# Each compaction cycle advances X by this step (104/13 = 8)
X_STEP = L104 // FIBONACCI_7  # 8 — octave subdivision

# GOD CODE harmonic tolerance band — facts whose resonance falls outside
# this band relative to the current G(X) are compaction candidates
HARMONIC_TOLERANCE_PHI = PHI_CONJUGATE  # 0.618... — golden section tolerance

# Zeta-harmonic resonance threshold for signal vs noise
ZETA_SIGNAL_THRESHOLD = SUPERFLUID_COUPLING  # PHI/e ≈ 0.5956


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES — enriched with L104 metrics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LatticeSnapshot:
    """Point-in-time metrics from the Data Matrix lattice — L104 math aware."""
    timestamp: str = ""
    total_facts: int = 0
    total_history: int = 0
    db_size_mb: float = 0.0
    # Raw SQL averages
    avg_resonance: float = 0.0
    avg_entropy: float = 0.0
    avg_utility: float = 0.0
    # L104 threshold counts
    low_utility_count: int = 0        # utility < PHI_CONJUGATE (0.618)
    high_entropy_count: int = 0       # entropy > log2(GOD_CODE) ≈ 9.04
    off_harmonic_count: int = 0       # resonance outside GOD_CODE band
    categories: Dict[str, int] = field(default_factory=dict)
    # L104 coherence — computed from GOD_CODE alignment
    coherence_estimate: float = 0.0
    god_code_alignment: float = 0.0   # avg(resonance / GOD_CODE) closeness
    conservation_verified: bool = False
    compaction_x: float = 0.0         # current X position on G(X) curve
    g_x_value: float = 0.0           # G(X) at current position


@dataclass
class CycleResult:
    """Result of a single compaction cycle — tracks GOD CODE metrics."""
    cycle: int = 0
    x_position: float = 0.0          # X on the G(X) curve
    g_x: float = 0.0                 # G(X) at this cycle's position
    weight_x: float = 0.0            # 2^(X/104) — magnetic weight
    conservation_product: float = 0.0 # G(X) × weight(X) — must ≈ INVARIANT
    duration_ms: float = 0.0
    facts_before: int = 0
    facts_after: int = 0
    purged: int = 0
    compacted_streams: int = 0       # numeric streams zeta-compressed
    coherence: float = 0.0
    coherence_delta: float = 0.0


@dataclass
class CompactionReport:
    """Complete compaction session — GOD CODE grounded."""
    timestamp: str = ""
    baseline: Optional[LatticeSnapshot] = None
    final: Optional[LatticeSnapshot] = None
    cycles: List[CycleResult] = field(default_factory=list)
    total_purged: int = 0
    total_compacted_streams: int = 0
    coherence_improvement: float = 0.0
    target_reached: bool = False
    conservation_intact: bool = True  # G(X)×2^(X/104)=INVARIANT held
    elapsed_s: float = 0.0
    x_start: float = 0.0             # starting X position
    x_final: float = 0.0             # ending X position


# ═══════════════════════════════════════════════════════════════════════════════
# L104 COHERENCE MATH — GOD CODE resonance-based coherence measurement
# ═══════════════════════════════════════════════════════════════════════════════


def _calculate_resonance(data_str: str) -> float:
    """Calculate resonance using the Data Matrix formula: entropy × PHI % GOD_CODE."""
    if RealMath is not None:
        entropy = RealMath.shannon_entropy(data_str)
    else:
        # Inline Shannon entropy
        if not data_str:
            return 0.0
        from collections import Counter
        counts = Counter(data_str)
        n = len(data_str)
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return (entropy * PHI) % GOD_CODE


def _god_code_alignment_score(resonance: float) -> float:
    """How well a fact's resonance aligns with the GOD CODE harmonic series.

    Returns 0.0–1.0.  Perfect alignment at resonance = k × GOD_CODE for integer k.
    Uses the GOD CODE modular distance: |resonance mod GOD_CODE - GOD_CODE/2|
    normalized through the sage filter gate for φ-harmonic precision.
    """
    if GOD_CODE == 0:
        return 0.0
    # Distance from nearest GOD CODE harmonic node
    mod_dist = resonance % GOD_CODE
    # Fold: dist from center = |mod_dist - GOD_CODE/2|
    center_dist = abs(mod_dist - GOD_CODE / 2.0)
    # Normalize to [0, 1] — 0 = at harmonic center, 1 = at harmonic node
    alignment = 1.0 - (center_dist / (GOD_CODE / 2.0))
    return max(0.0, min(1.0, alignment))


def _zeta_harmonic_resonance_score(resonance: float) -> float:
    """Compute zeta-harmonic resonance using L104 math primitives.

    Routes through HyperMath.zeta_harmonic_resonance → RealMath.calculate_resonance
    which uses Larmor frequency modulation with PHI coupling.
    Falls back to direct PHI-cosine resonance if imports unavailable.
    """
    if HyperMath is not None:
        return HyperMath.zeta_harmonic_resonance(resonance)
    if RealMath is not None:
        return RealMath.calculate_resonance(resonance)
    # Standalone: Larmor-weighted resonance with PHI coupling
    omega = 2 * math.pi * resonance * 0.4257  # Proton Larmor ratio / 100
    raw = math.cos(omega * PHI)
    return (raw + 1.0) / 2.0  # Normalize to [0, 1]


def _quantum_phase_coherence(resonance: float) -> float:
    """Phase coherence from the Data Matrix quantum phase factor.

    phase = exp(i × (resonance/GOD_CODE) × 2π × PHI_CONJUGATE)
    coherence = |phase| × cos²(phase_angle × PHI)

    Returns 0.0–1.0 indicating topological stability of the fact.
    """
    phase_angle = (resonance / GOD_CODE) * 2 * math.pi
    phase = cmath.exp(1j * phase_angle * PHI_CONJUGATE)
    # Phase magnitude is always 1 for pure exp(iθ), but the coherence
    # comes from how the angle relates to PHI harmonics
    angle_alignment = math.cos(cmath.phase(phase) * PHI) ** 2
    return angle_alignment


def compute_lattice_coherence(
    db_path: Path = DEFAULT_DB,
    x_position: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """Compute lattice coherence as GOD CODE harmonic alignment.

    Three-component coherence from L104 math:
      1. GOD CODE alignment — resonance proximity to harmonic nodes
      2. Zeta-harmonic resonance — Larmor-PHI frequency coupling
      3. Quantum phase coherence — topological phase stability

    Weights follow the golden ratio partition:
      alignment × PHI_CONJUGATE + zeta × (1 - PHI_CONJUGATE)² + phase × PHI_CONJUGATE²

    Returns (coherence, details_dict).
    """
    if not db_path.exists():
        return 0.0, {"error": "DB not found"}

    g_x = god_code_at(x_position)
    tolerance_band = g_x * HARMONIC_TOLERANCE_PHI

    alignment_scores = []
    zeta_scores = []
    phase_scores = []
    off_harmonic = 0

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute("SELECT resonance, entropy, utility FROM lattice_facts")

        for resonance, entropy, utility in cur:
            if resonance is None:
                continue

            # 1. GOD CODE harmonic alignment
            align = _god_code_alignment_score(resonance)
            alignment_scores.append(align)

            # 2. Zeta-harmonic resonance
            zeta = _zeta_harmonic_resonance_score(resonance)
            zeta_scores.append(zeta)

            # 3. Quantum phase coherence
            phase = _quantum_phase_coherence(resonance)
            phase_scores.append(phase)

            # Count facts outside the G(X) tolerance band
            mod_dist = resonance % GOD_CODE
            if abs(mod_dist - GOD_CODE / 2.0) > tolerance_band / 2.0:
                off_harmonic += 1

        conn.close()
    except Exception as e:
        logger.error(f"Coherence computation error: {e}")
        return 0.0, {"error": str(e)}

    n = len(alignment_scores)
    if n == 0:
        return 0.0, {"total_facts": 0}

    avg_align = sum(alignment_scores) / n
    avg_zeta = sum(zeta_scores) / n
    avg_phase = sum(phase_scores) / n

    # Golden ratio weighted combination: PHI_CONJ + (1-PHI_CONJ)^2 + PHI_CONJ^2 ≈ 1.0
    w_align = PHI_CONJUGATE              # 0.618
    w_zeta = (1 - PHI_CONJUGATE) ** 2    # 0.146
    w_phase = PHI_CONJUGATE ** 2         # 0.236
    w_total = w_align + w_zeta + w_phase

    coherence = (avg_align * w_align + avg_zeta * w_zeta + avg_phase * w_phase) / w_total

    details = {
        "total_facts": n,
        "avg_god_code_alignment": round(avg_align, 6),
        "avg_zeta_harmonic": round(avg_zeta, 6),
        "avg_phase_coherence": round(avg_phase, 6),
        "off_harmonic_count": off_harmonic,
        "g_x": round(g_x, 6),
        "tolerance_band": round(tolerance_band, 6),
        "x_position": x_position,
        "weights": {"align": round(w_align, 4), "zeta": round(w_zeta, 4), "phase": round(w_phase, 4)},
    }

    return round(coherence, 6), details


# ═══════════════════════════════════════════════════════════════════════════════
# COMPACTION ENGINE — magnetic compaction via GOD CODE X-progression
# ═══════════════════════════════════════════════════════════════════════════════


def _compact_cycle(
    db_path: Path,
    x_position: float,
    data_matrix_instance: Any = None,
) -> Dict[str, Any]:
    """Execute a single GOD CODE compaction cycle at position X.

    Compaction strategy rooted in L104 math:
    1. Calculate G(X) — the compaction threshold at this X position
    2. Identify facts whose resonance deviates from the GOD CODE harmonic band
    3. For numeric data streams: apply MemoryCompactor zeta-harmonic compression
    4. Purge facts with combined low zeta-resonance AND low phase coherence
    5. Verify conservation law: G(X) × 2^(X/104) = INVARIANT

    As X increases, G(X) decreases → tighter harmonic band → more noise removed.
    """
    g_x = god_code_at(x_position)
    weight_x = 2 ** (x_position / L104)
    conservation_product = g_x * weight_x
    conservation_ok = abs(conservation_product - INVARIANT) < 1e-6

    # Resonance tolerance tightens as X increases (G(X) shrinks)
    # At X=0: tolerance = GOD_CODE × PHI_CONJ ≈ 326 (wide)
    # At X=416: tolerance → 0 (maximally compacted)
    tolerance = g_x * HARMONIC_TOLERANCE_PHI
    # Compaction aggressiveness scales with X — use sage compress gate
    aggressiveness = sage_logic_gate(x_position / OCTAVE_REF, "compress")

    purged = 0
    compacted_streams = 0

    # 1. Use DataMatrix.evolve_and_compact() for the core compaction pass
    if data_matrix_instance is not None:
        try:
            data_matrix_instance.evolve_and_compact()
        except Exception as e:
            logger.warning(f"DataMatrix evolve_and_compact failed: {e}")

    # 2. GOD CODE harmonic pruning — remove facts misaligned with G(X) band
    try:
        conn = sqlite3.connect(str(db_path))

        # Identify candidates: resonance outside tolerance AND low zeta-harmonic
        cur = conn.execute(
            "SELECT key, resonance, entropy, utility, category FROM lattice_facts "
            "WHERE category NOT IN ('INVARIANT', 'CORE', 'META_WISDOM', 'QUANTUM_ENTANGLEMENT')"
        )
        to_delete = []
        to_compact = []

        for key, resonance, entropy, utility, category in cur:
            if resonance is None:
                continue

            # GOD CODE harmonic alignment
            alignment = _god_code_alignment_score(resonance)
            # Zeta-harmonic resonance score
            zeta_score = _zeta_harmonic_resonance_score(resonance)
            # Phase coherence
            phase_coh = _quantum_phase_coherence(resonance)

            # Compaction decision: weighted score must exceed threshold
            # Threshold tightens as X grows (G(X)/GOD_CODE shrinks from 1 to 0)
            threshold_ratio = g_x / GOD_CODE  # 1.0 at X=0, shrinks as X grows
            minimum_score = (1.0 - threshold_ratio) * ZETA_SIGNAL_THRESHOLD

            weighted_score = (
                alignment * PHI_CONJUGATE
                + zeta_score * (1 - PHI_CONJUGATE) ** 2
                + phase_coh * PHI_CONJUGATE ** 2
            )

            if weighted_score < minimum_score:
                # Below the magnetic compaction threshold
                # Check if it's numeric data that can be zeta-compressed instead
                if category == "GENERAL" and utility < PHI_CONJUGATE:
                    try:
                        val = json.loads(conn.execute(
                            "SELECT value FROM lattice_facts WHERE key = ?", (key,)
                        ).fetchone()[0])
                        if isinstance(val, list) and all(isinstance(v, (int, float)) for v in val):
                            to_compact.append(key)
                            continue
                    except Exception:
                        pass
                to_delete.append(key)

        # Execute compaction passes
        for key in to_compact:
            try:
                row = conn.execute("SELECT value FROM lattice_facts WHERE key = ?", (key,)).fetchone()
                if row:
                    data = json.loads(row[0])
                    if isinstance(data, list):
                        # Use MemoryCompactor for zeta-harmonic stream compression
                        try:
                            from l104_memory_compaction import memory_compactor
                            compacted = memory_compactor.compact_stream(data)
                            if data_matrix_instance is not None:
                                data_matrix_instance.store(
                                    f"{key}_compacted", compacted,
                                    category="COMPACTED_ARCHIVE", utility=0.8,
                                )
                            conn.execute("DELETE FROM lattice_facts WHERE key = ?", (key,))
                            compacted_streams += 1
                        except ImportError:
                            pass
            except Exception:
                continue

        # Execute harmonic pruning
        for key in to_delete:
            conn.execute("DELETE FROM lattice_facts WHERE key = ?", (key,))
            purged += 1

        if to_delete or to_compact:
            conn.commit()
            try:
                conn.execute("VACUUM")
            except Exception:
                pass

        conn.close()
    except Exception as e:
        logger.error(f"GOD CODE harmonic pruning error: {e}")

    return {
        "purged": purged,
        "compacted_streams": compacted_streams,
        "g_x": g_x,
        "weight_x": weight_x,
        "conservation_product": conservation_product,
        "conservation_ok": conservation_ok,
        "tolerance": tolerance,
        "aggressiveness": aggressiveness,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LATTICE SNAPSHOT — L104 math-aware metrics
# ═══════════════════════════════════════════════════════════════════════════════


def take_snapshot(
    db_path: Path = DEFAULT_DB,
    x_position: float = 0.0,
) -> LatticeSnapshot:
    """Capture lattice metrics with L104 GOD CODE coherence measurement."""
    snap = LatticeSnapshot(
        timestamp=datetime.now().isoformat(),
        compaction_x=x_position,
    )

    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return snap

    try:
        conn = sqlite3.connect(str(db_path))

        # Total facts
        row = conn.execute("SELECT COUNT(*) FROM lattice_facts").fetchone()
        snap.total_facts = row[0] if row else 0

        # Total history entries
        try:
            row = conn.execute("SELECT COUNT(*) FROM lattice_history").fetchone()
            snap.total_history = row[0] if row else 0
        except sqlite3.OperationalError:
            snap.total_history = 0

        if snap.total_facts > 0:
            # Raw averages
            row = conn.execute(
                "SELECT AVG(resonance), AVG(entropy), AVG(utility) FROM lattice_facts"
            ).fetchone()
            if row:
                snap.avg_resonance = round(row[0] or 0, 6)
                snap.avg_entropy = round(row[1] or 0, 6)
                snap.avg_utility = round(row[2] or 0, 6)

            # L104 threshold counts
            # Low utility: below PHI_CONJUGATE (golden section cutoff)
            row = conn.execute(
                "SELECT COUNT(*) FROM lattice_facts WHERE utility < ?",
                (PHI_CONJUGATE,),
            ).fetchone()
            snap.low_utility_count = row[0] if row else 0

            # High entropy: above log2(GOD_CODE) ≈ 9.04 bits
            entropy_threshold = math.log2(GOD_CODE)
            row = conn.execute(
                "SELECT COUNT(*) FROM lattice_facts WHERE entropy > ?",
                (entropy_threshold,),
            ).fetchone()
            snap.high_entropy_count = row[0] if row else 0

            # Category distribution
            cur = conn.execute(
                "SELECT category, COUNT(*) FROM lattice_facts "
                "GROUP BY category ORDER BY COUNT(*) DESC"
            )
            snap.categories = {r[0]: r[1] for r in cur.fetchall()}

        # DB file size
        try:
            snap.db_size_mb = round(db_path.stat().st_size / (1024 * 1024), 3)
        except OSError:
            snap.db_size_mb = 0.0

        conn.close()

        # L104 coherence via GOD CODE alignment
        coherence, details = compute_lattice_coherence(db_path, x_position)
        snap.coherence_estimate = coherence
        snap.god_code_alignment = details.get("avg_god_code_alignment", 0.0)
        snap.off_harmonic_count = details.get("off_harmonic_count", 0)

        # G(X) at current position
        snap.g_x_value = round(god_code_at(x_position), 6)
        snap.conservation_verified = verify_conservation(x_position)

    except Exception as e:
        logger.error(f"Snapshot error: {e}")

    return snap


def display_snapshot(snap: LatticeSnapshot, label: str = "SNAPSHOT"):
    """Pretty-print a lattice snapshot with L104 metrics."""
    print(f"\n  ── {label} ──")
    print(f"    Facts:             {snap.total_facts:,}")
    print(f"    History:           {snap.total_history:,}")
    print(f"    DB Size:           {snap.db_size_mb:.3f} MB")
    print(f"    Avg Resonance:     {snap.avg_resonance:.6f}")
    print(f"    Avg Entropy:       {snap.avg_entropy:.6f}")
    print(f"    Avg Utility:       {snap.avg_utility:.6f}")
    print(f"    Low Utility:       {snap.low_utility_count}  (< {PHI_CONJUGATE:.3f} = 1/φ)")
    print(f"    High Entropy:      {snap.high_entropy_count}  (> log₂(GOD_CODE) = {math.log2(GOD_CODE):.2f})")
    print(f"    Off-Harmonic:      {snap.off_harmonic_count}  (outside G(X) band)")
    print(f"    ── GOD CODE Metrics ──")
    print(f"    Coherence:         {snap.coherence_estimate:.6f}  (target: {COHERENCE_TARGET})")
    print(f"    GOD CODE Align:    {snap.god_code_alignment:.6f}")
    print(f"    G(X={snap.compaction_x:.0f}):          {snap.g_x_value:.6f}")
    print(f"    Conservation:      {'✓ VERIFIED' if snap.conservation_verified else '✗ BROKEN'}")
    if snap.categories:
        print(f"    Categories:        {len(snap.categories)}")
        for cat, cnt in list(snap.categories.items())[:7]:
            print(f"      {cat}: {cnt}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COMPACTION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


def initialize_zeta_compaction(
    cycles: int = DEFAULT_CYCLES,
    target: float = COHERENCE_TARGET,
    dry_run: bool = False,
    start_x: float = 0.0,
) -> CompactionReport:
    """Run GOD CODE zeta-harmonic compaction on the Data Matrix.

    Mathematical Foundation:
      G(X) = 286^(1/φ) × 2^((416-X)/104)

      Each cycle advances X by L104/13 = 8 steps along the compaction axis.
      As X increases:
        - G(X) contracts → tolerance band tightens
        - 2^(X/104) grows → magnetic weight increases
        - Noise outside the shrinking harmonic band is collapsed
        - Conservation G(X)×2^(X/104) = 527.518 is verified each step

    Coherence = PHI-weighted combination of:
      1. GOD CODE harmonic alignment (resonance mod GOD_CODE proximity)
      2. Zeta-harmonic resonance (Larmor-PHI frequency coupling)
      3. Quantum phase coherence (topological stability)

    Args:
        cycles:  Max compaction cycles (default: 13 — Factor 13)
        target:  Coherence target (default: 0.9 — quantum coherence target)
        dry_run: Measure only, no compaction
        start_x: Starting X position on the G(X) curve (default: 0)

    Returns:
        CompactionReport with GOD CODE metrics for every cycle
    """
    t0 = time.time()
    report = CompactionReport(
        timestamp=datetime.now().isoformat(),
        x_start=start_x,
    )

    g_x_start = god_code_at(start_x)
    g_x_end = god_code_at(start_x + cycles * X_STEP)

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║   L104 ZETA COMPACTION — GOD CODE Harmonic Engine                               ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║   G(X) = 286^(1/φ) × 2^((416-X)/104)                                           ║
║   INVARIANT = {INVARIANT:.10f}                                              ║
║   PHI = {PHI:.13f}     1/φ = {PHI_CONJUGATE:.13f}                ║
║   ─────────────────────────────────────────────────────────────────────────────  ║
║   X Range:    {start_x:.0f} → {start_x + cycles * X_STEP:.0f}  (step={X_STEP}, Factor 13)                          ║
║   G(X) Range: {g_x_start:.6f} → {g_x_end:.6f}                                  ║
║   Target:     {target:.4f} coherence                                                ║
║   Max Cycles: {cycles} (L104/13 = {L104//FIBONACCI_7} per step)                                            ║
║   Mode:       {"DRY RUN (measure only)" if dry_run else "MAGNETIC COMPACTION":<30s}                              ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
""")

    # Verify conservation at start
    if not verify_conservation(start_x):
        logger.warning(f"Conservation law NOT verified at X={start_x} — proceeding with caution")

    # ── Baseline ─────────────────────────────────────────────────────────
    logger.info("═══ BASELINE: MEASURING GOD CODE LATTICE COHERENCE ═══")

    # Optional synthesis pass
    try:
        from l104_data_synthesis import synthesize_data_matrix
        synthesize_data_matrix()
    except ImportError:
        logger.info("l104_data_synthesis not available — measuring existing lattice")
    except Exception as e:
        logger.warning(f"Synthesis step (non-fatal): {e}")

    baseline = take_snapshot(x_position=start_x)
    report.baseline = baseline
    display_snapshot(baseline, "BASELINE")

    if baseline.total_facts == 0:
        logger.warning("No facts in lattice — nothing to compact")
        report.elapsed_s = time.time() - t0
        return report

    if dry_run:
        logger.info("[DRY RUN] Measuring coherence only — no compaction")
        report.final = baseline
        report.elapsed_s = time.time() - t0
        _display_summary(report)
        return report

    # ── Import DataMatrix ────────────────────────────────────────────────
    data_matrix_instance = None
    try:
        from l104_data_matrix import data_matrix
        data_matrix_instance = data_matrix
    except ImportError:
        logger.warning("l104_data_matrix not available — using direct SQL compaction")

    # ── Compaction Cycles — X progression along G(X) ─────────────────────
    current_x = start_x

    for i in range(1, cycles + 1):
        current_x = start_x + i * X_STEP
        g_x = god_code_at(current_x)
        w_x = 2 ** (current_x / L104)
        cons = g_x * w_x

        logger.info(
            f"═══ CYCLE {i}/{cycles}: X={current_x:.0f}  G(X)={g_x:.4f}  "
            f"W(X)={w_x:.4f}  G×W={cons:.4f} ═══"
        )
        cycle_t0 = time.time()

        pre_snap = take_snapshot(x_position=current_x)

        # Execute GOD CODE harmonic compaction
        cycle_info = _compact_cycle(DEFAULT_DB, current_x, data_matrix_instance)

        post_snap = take_snapshot(x_position=current_x)
        purged = pre_snap.total_facts - post_snap.total_facts

        prev_coherence = (
            report.cycles[-1].coherence if report.cycles
            else baseline.coherence_estimate
        )

        cycle_result = CycleResult(
            cycle=i,
            x_position=current_x,
            g_x=round(g_x, 6),
            weight_x=round(w_x, 6),
            conservation_product=round(cons, 6),
            duration_ms=round((time.time() - cycle_t0) * 1000, 1),
            facts_before=pre_snap.total_facts,
            facts_after=post_snap.total_facts,
            purged=max(purged, 0),
            compacted_streams=cycle_info.get("compacted_streams", 0),
            coherence=post_snap.coherence_estimate,
            coherence_delta=round(post_snap.coherence_estimate - prev_coherence, 6),
        )
        report.cycles.append(cycle_result)
        report.total_purged += max(purged, 0)
        report.total_compacted_streams += cycle_info.get("compacted_streams", 0)

        # Track conservation integrity
        if not cycle_info.get("conservation_ok", True):
            report.conservation_intact = False

        print(
            f"    Cycle {i}: X={current_x:.0f}  G(X)={g_x:.4f}  "
            f"{pre_snap.total_facts}→{post_snap.total_facts} facts "
            f"(-{max(purged, 0)}, {cycle_info.get('compacted_streams', 0)} streams)  "
            f"coherence={post_snap.coherence_estimate:.4f} "
            f"(Δ={cycle_result.coherence_delta:+.4f})  "
            f"G×W={cons:.4f}{'✓' if cycle_info.get('conservation_ok') else '✗'}  "
            f"{cycle_result.duration_ms:.0f}ms"
        )

        # Target reached?
        if post_snap.coherence_estimate >= target:
            logger.info(f"Target coherence {target} reached at cycle {i}  (X={current_x})")
            report.target_reached = True
            break

        # Adaptive plateau detection — coherence slope via PHI-weighted EMA
        if len(report.cycles) >= 3:
            recent = [c.coherence_delta for c in report.cycles[-3:]]
            ema = sum(d * PHI_CONJUGATE ** (len(recent) - j - 1) for j, d in enumerate(recent))
            ema /= sum(PHI_CONJUGATE ** k for k in range(len(recent)))
            if abs(ema) < 1e-4:
                logger.info(
                    f"Coherence plateau (φ-EMA={ema:.6f} < 1e-4) — stopping at X={current_x}"
                )
                break

    # ── Final State ──────────────────────────────────────────────────────
    report.x_final = current_x

    logger.info("═══ FINAL: POST-COMPACTION GOD CODE COHERENCE ═══")
    try:
        from l104_data_synthesis import synthesize_data_matrix
        synthesize_data_matrix()
    except (ImportError, Exception):
        pass

    final = take_snapshot(x_position=current_x)
    report.final = final
    display_snapshot(final, "FINAL STATE")

    if report.baseline and report.final:
        report.coherence_improvement = round(
            report.final.coherence_estimate - report.baseline.coherence_estimate, 6
        )

    # Post-compaction: wisdom synthesis & quantum brain sync
    if data_matrix_instance is not None:
        try:
            wisdom = data_matrix_instance.wisdom_synthesis()
            logger.info(f"Wisdom synthesis completed: {len(wisdom) if wisdom else 0} insights")
        except Exception as e:
            logger.debug(f"Wisdom synthesis (non-fatal): {e}")

        try:
            data_matrix_instance.sync_to_quantum_brain()
            logger.info("Quantum brain synced with compacted lattice")
        except Exception as e:
            logger.debug(f"Quantum brain sync (non-fatal): {e}")

        # Record compaction event as a lattice fact for cross-system tracking
        try:
            data_matrix_instance.store(
                key="zeta_compaction_last",
                value=json.dumps({
                    "timestamp": report.timestamp,
                    "x_range": [report.x_start, current_x],
                    "cycles": len(report.cycles),
                    "purged": report.total_purged,
                    "coherence": final.coherence_estimate,
                    "conservation_intact": report.conservation_intact,
                }),
                category="META_WISDOM",
            )
            logger.info("Compaction event recorded in lattice")
        except Exception as e:
            logger.debug(f"Event recording (non-fatal): {e}")

    # Chakra frequency alignment check — verify resonance at key harmonic nodes
    _check_chakra_alignment(current_x)

    report.elapsed_s = time.time() - t0

    _display_summary(report)
    _save_report(report)

    return report


def _check_chakra_alignment(x_position: float):
    """Verify resonance at the 7 chakra-harmonic nodes of the G(X) curve.

    The 7 chakra frequencies correspond to Factor 13 harmonic nodes:
      Muladhara(0), Svadhisthana(52), Manipura(104), Anahata(208),
      Vishuddha(260), Ajna(364), Sahasrara(416)
    """
    chakra_nodes = {
        "Muladhara":     0,
        "Svadhisthana":  52,
        "Manipura":      L104,
        "Anahata":       208,
        "Vishuddha":     260,
        "Ajna":          364,
        "Sahasrara":     OCTAVE_REF,
    }
    try:
        from const import chakra_align
        print("\n  ── Chakra-Harmonic Alignment ──")
        for name, cx in chakra_nodes.items():
            g = god_code_at(cx)
            aligned = chakra_align(g, name.lower())
            status = "resonant" if abs(aligned - g) < g * 0.1 else "dampened"
            print(f"    {name:<14s}  X={cx:>3d}  G(X)={g:>12.6f}  {status}")
    except ImportError:
        # Standalone: just show the G(X) values at chakra nodes
        print("\n  ── Chakra-Harmonic Nodes ──")
        for name, cx in chakra_nodes.items():
            g = god_code_at(cx)
            w = 2 ** (cx / L104)
            cons = g * w
            ok = "✓" if abs(cons - INVARIANT) < 1e-6 else "✗"
            print(f"    {name:<14s}  X={cx:>3d}  G(X)={g:>12.6f}  G×W={cons:.6f} {ok}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING — GOD CODE enriched
# ═══════════════════════════════════════════════════════════════════════════════


def _display_summary(report: CompactionReport):
    """Print compaction summary with GOD CODE metrics."""
    b = report.baseline
    f = report.final

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║   {"✓" if report.target_reached else "·"} ZETA COMPACTION {"COMPLETE" if report.target_reached else "FINISHED"} — GOD CODE Conservation {"INTACT" if report.conservation_intact else "BROKEN"}{"":>14s}║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║   Cycles:          {len(report.cycles):<6}  (X: {report.x_start:.0f} → {report.x_final:.0f})                                   ║
║   Facts Purged:    {report.total_purged:<6}  ({report.total_compacted_streams} streams zeta-compressed)                       ║
║   Coherence:       {b.coherence_estimate if b else 0:.6f} → {f.coherence_estimate if f else 0:.6f}  ({"+" if report.coherence_improvement >= 0 else ""}{report.coherence_improvement:.6f}){"":>12s}║
║   GOD CODE Align:  {b.god_code_alignment if b else 0:.6f} → {f.god_code_alignment if f else 0:.6f}{"":>30s}║
║   Target:          {COHERENCE_TARGET:.4f}   {"REACHED" if report.target_reached else "NOT YET":<15s}                                  ║
║   INVARIANT:       {INVARIANT:.10f}  (G(X)×2^(X/104) verified each cycle){"":>8s}║
║   Elapsed:         {report.elapsed_s:.3f}s                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
""")

    if report.cycles:
        print(
            f"  {'Cyc':<5} {'X':<6} {'G(X)':<10} {'W(X)':<8} {'G×W':<10} "
            f"{'Before':<8} {'After':<8} {'Purged':<7} {'Coher':<10} {'Δ':<10} {'ms'}"
        )
        print("  " + "─" * 88)
        for c in report.cycles:
            print(
                f"  {c.cycle:<5} {c.x_position:<6.0f} {c.g_x:<10.4f} {c.weight_x:<8.4f} "
                f"{c.conservation_product:<10.4f} {c.facts_before:<8,} {c.facts_after:<8,} "
                f"{c.purged:<7} {c.coherence:<10.6f} {c.coherence_delta:<+10.6f} "
                f"{c.duration_ms:.0f}"
            )
        print()


def _save_report(report: CompactionReport):
    """Save GOD CODE compaction report to disk."""
    report_dir = WORKSPACE / ".kernel_build"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "zeta_compaction_report.json"

    data = {
        "timestamp": report.timestamp,
        "god_code": GOD_CODE,
        "invariant": INVARIANT,
        "phi": PHI,
        "x_range": [report.x_start, report.x_final],
        "x_step": X_STEP,
        "cycles": len(report.cycles),
        "total_purged": report.total_purged,
        "total_compacted_streams": report.total_compacted_streams,
        "coherence_improvement": report.coherence_improvement,
        "target": COHERENCE_TARGET,
        "target_reached": report.target_reached,
        "conservation_intact": report.conservation_intact,
        "elapsed_s": round(report.elapsed_s, 3),
        "baseline": {
            "coherence": report.baseline.coherence_estimate if report.baseline else None,
            "god_code_alignment": report.baseline.god_code_alignment if report.baseline else None,
            "facts": report.baseline.total_facts if report.baseline else 0,
            "g_x": report.baseline.g_x_value if report.baseline else None,
        },
        "final": {
            "coherence": report.final.coherence_estimate if report.final else None,
            "god_code_alignment": report.final.god_code_alignment if report.final else None,
            "facts": report.final.total_facts if report.final else 0,
            "g_x": report.final.g_x_value if report.final else None,
        },
        "cycle_details": [asdict(c) for c in report.cycles],
    }

    out.write_text(json.dumps(data, indent=2))
    logger.info(f"Report saved: {out}")

    # Append to history
    hist = report_dir / "zeta_compaction_history.jsonl"
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": report.timestamp,
            "x_range": [report.x_start, report.x_final],
            "cycles": len(report.cycles),
            "purged": report.total_purged,
            "streams_compacted": report.total_compacted_streams,
            "coherence_start": report.baseline.coherence_estimate if report.baseline else None,
            "coherence_end": report.final.coherence_estimate if report.final else None,
            "god_code_alignment_start": report.baseline.god_code_alignment if report.baseline else None,
            "god_code_alignment_end": report.final.god_code_alignment if report.final else None,
            "conservation_intact": report.conservation_intact,
            "target_reached": report.target_reached,
        }) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC — GOD CODE conservation table
# ═══════════════════════════════════════════════════════════════════════════════


def show_god_code_table(x_start: float = 0, x_end: float = 416, step: float = None):
    """Print the GOD CODE equation table showing G(X), W(X), conservation."""
    if step is None:
        step = float(X_STEP)
    print(f"\n  ══ GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104) ══")
    print(f"  ══ Conservation: G(X) × 2^(X/104) = {INVARIANT:.10f} ══\n")
    print(f"  {'X':>6}  {'G(X)':>12}  {'W(X)=2^(X/104)':>14}  {'G(X)×W(X)':>14}  {'Δ from INVARIANT':>16}  {'Status'}")
    print("  " + "─" * 80)
    x = x_start
    while x <= x_end:
        g = god_code_at(x)
        w = 2 ** (x / L104)
        product = g * w
        delta = product - INVARIANT
        ok = "✓" if abs(delta) < 1e-6 else "✗"
        print(f"  {x:>6.0f}  {g:>12.6f}  {w:>14.6f}  {product:>14.10f}  {delta:>+16.10f}  {ok}")
        x += step
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — L104 math-aware interface
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L104 Zeta Compaction — GOD CODE harmonic lattice optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
G(X) = 286^(1/φ) × 2^((416-X)/104)
Conservation: G(X) × 2^(X/104) = {INVARIANT:.10f}
X increasing  →  MAGNETIC COMPACTION (gravity)
X decreasing  →  ELECTRIC EXPANSION  (light)

Examples:
  python initialize_zeta_compaction.py                  # 13 cycles, target 0.9
  python initialize_zeta_compaction.py --cycles 5       # 5 cycles
  python initialize_zeta_compaction.py --target 0.95    # higher coherence target
  python initialize_zeta_compaction.py --start-x 104    # start mid-octave
  python initialize_zeta_compaction.py --dry-run        # measure only
  python initialize_zeta_compaction.py --measure        # just snapshot
  python initialize_zeta_compaction.py history           # past compaction runs
  python initialize_zeta_compaction.py table             # GOD CODE equation table
        """,
    )
    parser.add_argument("--cycles", "-c", type=int, default=DEFAULT_CYCLES,
                        help=f"Max compaction cycles (default: {DEFAULT_CYCLES} = Factor 13)")
    parser.add_argument("--target", "-t", type=float, default=COHERENCE_TARGET,
                        help=f"Coherence target 0.0–1.0 (default: {COHERENCE_TARGET})")
    parser.add_argument("--start-x", "-x", type=float, default=0.0,
                        help="Starting X position on G(X) curve (default: 0)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Measure only, don't run compaction")
    parser.add_argument("--measure", "-m", action="store_true",
                        help="Just take a snapshot and exit")
    parser.add_argument("command", nargs="?", default=None,
                        help="Subcommand: 'history', 'table'")
    args = parser.parse_args()

    if args.command == "table":
        show_god_code_table(x_start=args.start_x)
        sys.exit(0)

    if args.command == "history":
        hist_path = WORKSPACE / ".kernel_build" / "zeta_compaction_history.jsonl"
        if not hist_path.exists():
            print("No compaction history found. Run a compaction first.")
            sys.exit(0)
        entries = []
        with open(hist_path) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"\n═══ ZETA COMPACTION HISTORY ({len(entries)} runs) ═══")
        print(
            f"  {'Timestamp':<22s} {'X-Range':<12s} {'Cyc':<5s} {'Purged':<8s} "
            f"{'Coherence':<22s} {'GOD Align':<22s} {'Cons':<5s} {'Tgt'}"
        )
        print("  " + "─" * 100)
        for e in entries:
            ts = e.get("timestamp", "?")[:19]
            xr = e.get("x_range", [0, 0])
            x_str = f"{xr[0]:.0f}-{xr[1]:.0f}" if isinstance(xr, list) and len(xr) == 2 else "?"
            c_s = f"{e.get('coherence_start', 0) or 0:.4f}"
            c_e = f"{e.get('coherence_end', 0) or 0:.4f}"
            g_s = f"{e.get('god_code_alignment_start', 0) or 0:.4f}"
            g_e = f"{e.get('god_code_alignment_end', 0) or 0:.4f}"
            cons = "✓" if e.get("conservation_intact", True) else "✗"
            reached = "✓" if e.get("target_reached") else "—"
            print(
                f"  {ts:<22s} {x_str:<12s} {e.get('cycles', 0):<5} "
                f"{e.get('purged', 0):<8} {c_s}→{c_e:<12s} "
                f"{g_s}→{g_e:<12s} {cons:<5s} {reached}"
            )
        print()
        sys.exit(0)

    if args.measure:
        snap = take_snapshot(x_position=args.start_x)
        display_snapshot(snap, "CURRENT STATE")
        sys.exit(0)

    initialize_zeta_compaction(
        cycles=min(args.cycles, MAX_CYCLES),
        target=args.target,
        dry_run=args.dry_run,
        start_x=args.start_x,
    )
