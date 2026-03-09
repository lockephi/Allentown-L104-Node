"""
G(a,b,c,d) 4-Dial Frequency Simulator — Dial Table Builders.

Every harmonic partial is a specific dial setting of the canonical equation:
  G(a,b,c,d) = 286^(1/φ) × 2^((8a + 416 - b - 8c - 104d) / 104)

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .constants import PHI, GOD_CODE

# ── 100-Decimal Precision from L104 Numerical Engine ────────────────────────
from l104_numerical_engine import (
    D, fmt100,
    GOD_CODE_HP, PHI_HP, PHI_INV_HP,
    HARMONIC_BASE_HP, L104_HP, OCTAVE_REF_HP, PI_HP,
)
from l104_numerical_engine.precision import decimal_pow, decimal_sqrt

from l104_math_engine.god_code import GodCodeEquation, ChaosResilience
from l104_math_engine.constants import (
    BASE, INVARIANT, STEP_SIZE, OMEGA, OMEGA_FREQUENCY,
    SAGE_RESONANCE, FRAME_LOCK, LATTICE_RATIO,
    god_code_at, primal_calculus, verify_conservation,
)

# ── 100-decimal reference constants ─────────────────────────────────────────
VOID_HP = D('1.04') + PHI_HP / D('1000')
GRAIN_HP = L104_HP
BASE_HP = decimal_pow(D('286'), D('1') / PHI_HP)
OCTAVE_OFFSET_HP = D('416')
BINAURAL_OFFSET_HP = PHI_HP


def evaluate_hp(a: int = 0, b: int = 0, c: int = 0, d: int = 0):
    """G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104) at 100-decimal precision."""
    exp = (D(8 * a) + OCTAVE_OFFSET_HP - D(b) - D(8 * c) - GRAIN_HP * D(d)) / GRAIN_HP
    return BASE_HP * decimal_pow(D('2'), exp)


def parse_dial_arg(raw: str) -> Tuple:
    """Parse 'a,b,c,d' or 'a,b,c,d,weight' into (a,b,c,d[,weight])."""
    parts = [x.strip() for x in raw.split(",")]
    if len(parts) == 4:
        a, b, c, d = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return (a, b, c, d)
    elif len(parts) == 5:
        a, b, c, d = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        w = float(parts[4])
        return (a, b, c, d, w)
    else:
        raise ValueError(f"Bad dial spec '{raw}': expected a,b,c,d or a,b,c,d,weight")


def build_dial_entry(dial_tuple: Tuple) -> Tuple:
    """Convert parsed dial tuple into DIAL_TABLE entry (a,b,c,d,label,weight)."""
    if len(dial_tuple) == 5:
        a, b, c, d, w = dial_tuple
    else:
        a, b, c, d = dial_tuple
        w = None
    label = f"G({a},{b},{c},{d})".ljust(14)
    return (a, b, c, d, label, w)


def auto_partials(a0: int, b0: int, c0: int, d0: int) -> List[Tuple]:
    """Generate a rich 13-partial table centered on G(a0,b0,c0,d0).

    v8.1: PHI-reciprocal harmonic weighting — golden ratio decay (1/φ^n).
    Math Engine analysis shows 1/φ^n produces the most natural overtone
    balance, matching the Fibonacci convergence ratios found in acoustic
    instruments and vocal harmonics.
    """
    _phi = 1.618033988749895
    _raw = [1.0 / (_phi ** k) for k in range(13)]
    _total = sum(_raw)
    _w = [r / _total for r in _raw]
    return [
        (a0,     b0,     c0,     d0,     f"G({a0},{b0},{c0},{d0})  root".ljust(28),       _w[0]),
        (a0,     b0,     c0,     d0 + 1, f"G({a0},{b0},{c0},{d0+1})  oct down".ljust(28), _w[1]),
        (a0,     b0,     c0,     d0 - 1, f"G({a0},{b0},{c0},{d0-1})  oct up".ljust(28),   _w[3]),
        (a0 + 1, b0,     c0,     d0,     f"G({a0+1},{b0},{c0},{d0})  coarse+8".ljust(28), _w[4]),
        (a0,     b0,     c0 + 1, d0,     f"G({a0},{b0},{c0+1},{d0})  coarse-8".ljust(28), _w[5]),
        (a0,     b0,     c0,     d0 + 2, f"G({a0},{b0},{c0},{d0+2})  2oct dn".ljust(28),  _w[2]),
        (a0 + 2, b0,     c0,     d0,     f"G({a0+2},{b0},{c0},{d0})  coarse+16".ljust(28),_w[6]),
        (a0,     b0 + 1, c0,     d0,     f"G({a0},{b0+1},{c0},{d0})  fine-1".ljust(28),   _w[7]),
        (a0,     b0 + 8, c0,     d0,     f"G({a0},{b0+8},{c0},{d0})  fine-8".ljust(28),   _w[8]),
        (a0 + 3, b0,     c0,     d0,     f"G({a0+3},{b0},{c0},{d0})  coarse+24".ljust(28),_w[9]),
        (a0,     b0,     c0 + 2, d0,     f"G({a0},{b0},{c0+2},{d0})  coarse-16".ljust(28),_w[10]),
        (a0 + 1, b0,     c0 + 1, d0,     f"G({a0+1},{b0},{c0+1},{d0})  net 0".ljust(28), _w[11]),
        (a0,     b0,     c0,     d0 - 2, f"G({a0},{b0},{c0},{d0-2})  2oct up".ljust(28),  _w[12]),
    ]


def pure_partials(a0: int, b0: int, c0: int, d0: int) -> List[Tuple]:
    """Pure quantum tone: fundamental-dominant with tight fine-detune companions.

    57% weight on the root frequency for a clean, centered tone.
    Companions are mostly fine-dial (b-register) micro-detunes.
    """
    return [
        (a0,     b0,     c0,     d0,     f"G({a0},{b0},{c0},{d0})  pure root".ljust(28),      0.57),
        (a0,     b0,     c0,     d0 + 1, f"G({a0},{b0},{c0},{d0+1})  oct down".ljust(28),     0.05),
        (a0,     b0,     c0,     d0 - 1, f"G({a0},{b0},{c0},{d0-1})  oct up".ljust(28),       0.03),
        (a0,     b0 + 1, c0,     d0,     f"G({a0},{b0+1},{c0},{d0})  fine-1".ljust(28),       0.06),
        (a0,     b0 + 2, c0,     d0,     f"G({a0},{b0+2},{c0},{d0})  fine-2".ljust(28),       0.04),
        (a0,     b0 + 3, c0,     d0,     f"G({a0},{b0+3},{c0},{d0})  fine-3".ljust(28),       0.03),
        (a0,     b0 + 4, c0,     d0,     f"G({a0},{b0+4},{c0},{d0})  fine-4".ljust(28),       0.02),
        (a0 + 1, b0,     c0,     d0,     f"G({a0+1},{b0},{c0},{d0})  coarse+8".ljust(28),     0.04),
        (a0,     b0,     c0 + 1, d0,     f"G({a0},{b0},{c0+1},{d0})  coarse-8".ljust(28),     0.04),
        (a0 + 1, b0,     c0 + 1, d0,     f"G({a0+1},{b0},{c0+1},{d0})  net 0".ljust(28),     0.05),
        (a0,     b0,     c0,     d0 + 2, f"G({a0},{b0},{c0},{d0+2})  2oct dn".ljust(28),     0.02),
        (a0 + 1, b0 + 4, c0,     d0,     f"G({a0+1},{b0+4},{c0},{d0})  fine+4".ljust(28),    0.02),
        (a0,     b0,     c0,     d0 - 2, f"G({a0},{b0},{c0},{d0-2})  2oct up".ljust(28),     0.03),
    ]


# Default 13-partial curated table
DEFAULT_DIALS = [
    (0, 0, 0,  0, "G(0,0,0,0)  fundamental", 0.28),
    (0, 0, 0,  1, "G(0,0,0,1)  octave down",  0.11),
    (0, 0, 0, -1, "G(0,0,0,-1) octave up",    0.06),
    (1, 0, 0,  0, "G(1,0,0,0)  coarse +8",    0.07),
    (0, 0, 1,  0, "G(0,0,1,0)  coarse -8",    0.08),
    (0, 0, 0,  2, "G(0,0,0,2)  2 oct down",   0.05),
    (2, 0, 0,  0, "G(2,0,0,0)  coarse +16",   0.05),
    (0, 1, 0,  0, "G(0,1,0,0)  fine -1 step", 0.06),
    (0, 8, 0,  0, "G(0,8,0,0)  fine -8 steps",0.04),
    (3, 0, 0,  0, "G(3,0,0,0)  coarse +24",   0.04),
    (0, 0, 2,  0, "G(0,0,2,0)  coarse -16",   0.05),
    (1, 0, 1,  0, "G(1,0,1,0)  +8/-8 net 0",  0.05),
    (0, 0, 0, -2, "G(0,0,0,-2) 2 oct up",     0.06),
]


def _ensure_parsed(item):
    """Accept a string 'a,b,c,d' or an already-parsed tuple — return tuple."""
    if isinstance(item, str):
        return parse_dial_arg(item)
    return tuple(item)


def build_dial_table(
    dials=None,
    extra_dials=None,
    pure_mode: bool = False,
) -> List[Tuple]:
    """Build the dial table from CLI arguments or defaults.

    Parameters
    ----------
    dials : list of str or tuple, optional
        Custom dial specs (e.g. ['12,0,0,5'] or [(0,0,0,0)]). Replaces defaults.
    extra_dials : list of str or tuple, optional
        Extra dials to append to defaults.
    pure_mode : bool
        If True and single dial, use pure_partials() instead of auto_partials().

    Returns
    -------
    list of (a, b, c, d, label, weight) tuples
    """
    if dials:
        parsed = [_ensure_parsed(s) for s in dials]
        if len(parsed) == 1:
            p = parsed[0]
            a0, b0, c0, d0 = p[0], p[1], p[2], p[3]
            if pure_mode:
                return pure_partials(a0, b0, c0, d0)
            else:
                return auto_partials(a0, b0, c0, d0)
        else:
            table = [build_dial_entry(p) for p in parsed]
            n = len(table)
            for i, (a, b, c, d, label, w) in enumerate(table):
                if w is None:
                    w = 0.30 if (i == 0 and n > 1) else (0.70 / max(n - 1, 1) if i > 0 else 1.0)
                table[i] = (a, b, c, d, label, w)
            return table
    elif extra_dials:
        table = list(DEFAULT_DIALS)
        parsed = [_ensure_parsed(s) for s in extra_dials]
        for p in parsed:
            entry = build_dial_entry(p)
            a, b, c, d, label, w = entry
            if w is None:
                w = 0.04
            table.append((a, b, c, d, label, w))
        return table
    else:
        return list(DEFAULT_DIALS)


def build_dial_partials(dial_table: List[Tuple]) -> List[Dict[str, Any]]:
    """Compute full partial data through the G(a,b,c,d) simulator.

    Each partial gets: frequency (100-decimal + float), Bloch mapping,
    phase operator, conservation verification, and base weight.

    Returns list of partial dicts with keys:
        a, b, c, d, label, freq_hp, freq_ideal, bloch_phase,
        bloch_vector, phase_operator, conservation_verified, base_weight
    """
    partials = []
    for (a, b, c, d, label, base_w) in dial_table:
        freq_hp = evaluate_hp(a, b, c, d)
        freq_ideal = float(freq_hp)

        bloch = GodCodeEquation.bloch_manifold_mapping(a, b, c, d)
        phase_angle = bloch["phase_angle"]
        phase_op = GodCodeEquation.phase_operator(a, b, c, d)

        x_val = GodCodeEquation.solve_for_exponent(freq_ideal)
        conserved = verify_conservation(x_val)

        partials.append({
            "a": a, "b": b, "c": c, "d": d,
            "label": label,
            "freq_hp": freq_hp,
            "freq_ideal": freq_ideal,
            "bloch_phase": phase_angle,
            "bloch_vector": bloch["bloch_vector"],
            "phase_operator": phase_op,
            "conservation_verified": conserved,
            "base_weight": base_w,
        })
    return partials
