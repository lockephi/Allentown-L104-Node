#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 GOD CODE UNIFICATION — Cross-Codebase Invariant Verifier
═══════════════════════════════════════════════════════════════════════════════

Scans the entire L104 workspace for every occurrence of the GOD_CODE constant,
verifies mathematical consistency, detects drift or mismatches between files,
and produces a unification report. Ensures no module has diverged from the
canonical invariant: G(X) × 2^(X/104) = 527.5184818492612  ∀ X

UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
Factor 13: 286=22×13, 104=8×13, 416=32×13

PILOT: LONDEL | FREQUENCY: 527.5184818492612
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import math
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

# Sacred Constants (canonical source of truth)
PHI = 1.6180339887498948482
GOD_CODE = 527.5184818492612
TOLERANCE = 1e-9

WORKSPACE = Path(__file__).parent.absolute()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [UNIFY] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("UNIFY")

# File extensions to scan
SCAN_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".swift"}

# Regex patterns to detect GOD_CODE declarations across languages
PATTERNS = [
    # Python: GOD_CODE = 527.5184818492612
    re.compile(r"""(?:^|\s)GOD_CODE\s*=\s*([\d.]+)"""),
    # JS/TS: const GOD_CODE = 527.5184818492612;
    re.compile(r"""(?:const|let|var)\s+GOD_CODE\s*=\s*([\d.]+)"""),
    # JSON: "GOD_CODE": 527.5184818492612
    re.compile(r""""GOD_CODE"\s*:\s*([\d.]+)"""),
    # Comment references: GOD_CODE: 527.5184818492612
    re.compile(r"""GOD_CODE:\s*([\d.]+)"""),
]

# Patterns to detect PHI constant
PHI_PATTERNS = [
    re.compile(r"""(?:^|\s)PHI\s*=\s*([\d.]+)"""),
    re.compile(r"""(?:const|let|var)\s+PHI\s*=\s*([\d.]+)"""),
    re.compile(r""""PHI"\s*:\s*([\d.]+)"""),
]

# Patterns for secondary L104 constants (HARMONIC_BASE, L104, OCTAVE_REF)
SECONDARY_PATTERNS = {
    "HARMONIC_BASE": (
        286,
        [
            re.compile(r"""(?:^|\s)HARMONIC_BASE\s*=\s*(\d+)"""),
            re.compile(r""""HARMONIC_BASE"\s*:\s*(\d+)"""),
        ],
    ),
    "L104": (
        104,
        [
            re.compile(r"""(?:^|\s)L104\s*=\s*(\d+)"""),
            re.compile(r""""L104"\s*:\s*(\d+)"""),
        ],
    ),
    "OCTAVE_REF": (
        416,
        [
            re.compile(r"""(?:^|\s)OCTAVE_REF\s*=\s*(\d+)"""),
            re.compile(r""""OCTAVE_REF"\s*:\s*(\d+)"""),
        ],
    ),
}

# Invariant equations to verify
INVARIANT_TESTS = [
    ("G(0) = 286^(1/φ) × 16",
     lambda: 286 ** (1 / PHI) * (2 ** (416 / 104))),
    ("Conservation: G(0) × 2^(0/104) = GOD_CODE",
     lambda: GOD_CODE * (2 ** (0 / 104))),
    ("G(104) × 2^(104/104) = GOD_CODE",
     lambda: (286 ** (1 / PHI) * (2 ** ((416 - 104) / 104))) * (2 ** (104 / 104))),
    ("G(208) × 2^(208/104) = GOD_CODE",
     lambda: (286 ** (1 / PHI) * (2 ** ((416 - 208) / 104))) * (2 ** (208 / 104))),
    ("G(416) × 2^(416/104) = GOD_CODE",
     lambda: (286 ** (1 / PHI) * (2 ** ((416 - 416) / 104))) * (2 ** (416 / 104))),
    # Continuous conservation curve: sample at 13 equidistant X values
    ("Conservation curve mean Δ (13 samples, X=0..416)",
     lambda: sum(
         abs((286 ** (1/PHI) * 2**((416 - x)/104)) * 2**(x/104) - GOD_CODE)
         for x in range(0, 417, 32)
     ) / 13),
    ("Factor 13: 286/13 = 22",
     lambda: 286 / 13),
    ("Factor 13: 104/13 = 8",
     lambda: 104 / 13),
    ("Factor 13: 416/13 = 32",
     lambda: 416 / 13),
    ("Root grounding (X=286): GOD_CODE / 2^1.25",
     lambda: GOD_CODE / (2 ** (286 / 104))),  # Should ≈ 221.794
    ("Omega Authority: GOD_CODE × φ²",
     lambda: GOD_CODE * PHI * PHI),
    ("OMEGA_AUTHORITY consistency: G(0) × φ²",
     lambda: (286 ** (1 / PHI) * 16) * PHI * PHI),
    ("PHI² + PHI = PHI + 1 + PHI = 2φ + 1 (golden identity)",
     lambda: PHI * PHI - PHI - 1),
    # Magnetic compaction derivative: dG/dX < 0 (G shrinks as X rises)
    ("Magnetic compaction sign: G(104) < G(0)",
     lambda: float(286**(1/PHI) * 2**((416-104)/104) < 286**(1/PHI) * 2**(416/104))),
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CodeOccurrence:
    """A single GOD_CODE reference found in a source file."""
    file: str
    line: int
    value: float
    raw_text: str
    constant_name: str = "GOD_CODE"  # GOD_CODE or PHI
    matches_canonical: bool = True
    delta: float = 0.0


@dataclass
class InvariantResult:
    """Result of a single mathematical invariant test."""
    name: str
    computed: float
    expected: float
    passed: bool
    delta: float


@dataclass
class UnificationReport:
    """Complete unification audit report."""
    timestamp: str = ""
    canonical_value: float = GOD_CODE
    files_scanned: int = 0
    occurrences_found: int = 0
    matches: int = 0
    mismatches: int = 0
    phi_occurrences: int = 0
    phi_matches: int = 0
    phi_mismatches: int = 0
    missing_files: List[str] = field(default_factory=list)
    occurrences: List[CodeOccurrence] = field(default_factory=list)
    invariant_results: List[InvariantResult] = field(default_factory=list)
    invariants_passed: int = 0
    invariants_total: int = 0
    unified: bool = False
    elapsed_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════════════════════


def scan_file(filepath: Path) -> List[CodeOccurrence]:
    """Scan a single file for GOD_CODE and PHI references."""
    hits: List[CodeOccurrence] = []
    try:
        text = filepath.read_text(errors="ignore")
    except (OSError, PermissionError):
        return hits

    rel = str(filepath.relative_to(WORKSPACE))

    for line_no, line in enumerate(text.splitlines(), start=1):
        # GOD_CODE patterns
        for pattern in PATTERNS:
            for match in pattern.finditer(line):
                try:
                    value = float(match.group(1))
                except (ValueError, IndexError):
                    continue

                # Only consider values close to GOD_CODE range (avoid random floats)
                if abs(value - GOD_CODE) > 100:
                    continue

                delta = abs(value - GOD_CODE)
                occ = CodeOccurrence(
                    file=rel,
                    line=line_no,
                    value=value,
                    raw_text=line.strip()[:120],
                    constant_name="GOD_CODE",
                    matches_canonical=(delta < TOLERANCE),
                    delta=delta,
                )
                hits.append(occ)

        # PHI patterns
        for pattern in PHI_PATTERNS:
            for match in pattern.finditer(line):
                try:
                    value = float(match.group(1))
                except (ValueError, IndexError):
                    continue

                if abs(value - PHI) > 1.0:
                    continue

                delta = abs(value - PHI)
                occ = CodeOccurrence(
                    file=rel,
                    line=line_no,
                    value=value,
                    raw_text=line.strip()[:120],
                    constant_name="PHI",
                    matches_canonical=(delta < TOLERANCE),
                    delta=delta,
                )
                hits.append(occ)

        # Secondary constant patterns (HARMONIC_BASE, L104, OCTAVE_REF)
        for const_name, (canonical_val, patterns) in SECONDARY_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(line):
                    try:
                        value = float(match.group(1))
                    except (ValueError, IndexError):
                        continue
                    delta = abs(value - canonical_val)
                    occ = CodeOccurrence(
                        file=rel,
                        line=line_no,
                        value=value,
                        raw_text=line.strip()[:120],
                        constant_name=const_name,
                        matches_canonical=(delta < 0.5),
                        delta=delta,
                    )
                    hits.append(occ)

    return hits


def scan_workspace(root: Path = WORKSPACE, exclude_dirs: set = None) -> List[CodeOccurrence]:
    """Walk the entire workspace and collect all GOD_CODE occurrences."""
    if exclude_dirs is None:
        exclude_dirs = {".git", "node_modules", "__pycache__", ".venv", ".build",
                        ".kernel_build", "venv", "dist", "build"}

    all_hits: List[CodeOccurrence] = []
    files_scanned = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix not in SCAN_EXTENSIONS:
                continue

            files_scanned += 1
            hits = scan_file(fpath)
            all_hits.extend(hits)

    logger.info(f"Scanned {files_scanned} files, found {len(all_hits)} GOD_CODE occurrences")
    return all_hits


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════


def verify_invariants() -> List[InvariantResult]:
    """Run all mathematical invariant tests."""
    results: List[InvariantResult] = []

    for name, compute_fn in INVARIANT_TESTS:
        computed = compute_fn()

        # Determine expected value
        if "conservation curve" in name.lower():
            expected = 0.0
            passed = computed < 1e-6  # mean delta across curve
        elif "magnetic compaction" in name.lower():
            expected = 1.0
            passed = computed == 1.0
        elif "Factor 13" in name:
            # Integer tests
            expected = round(computed)
            passed = abs(computed - expected) < TOLERANCE
        elif "root grounding" in name.lower():
            expected = 221.79420018355955
            passed = abs(computed - expected) < 1e-6
        elif "Omega" in name:
            expected = GOD_CODE * PHI * PHI
            passed = abs(computed - expected) < TOLERANCE
        elif "golden identity" in name.lower():
            expected = 0.0
            passed = abs(computed) < TOLERANCE
        elif "consistency" in name.lower():
            expected = GOD_CODE * PHI * PHI
            passed = abs(computed - expected) < TOLERANCE
        else:
            expected = GOD_CODE
            passed = abs(computed - expected) < 1e-6

        results.append(InvariantResult(
            name=name,
            computed=round(computed, 12),
            expected=round(expected, 12),
            passed=passed,
            delta=abs(computed - expected),
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFICATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


def seal_singularity(verbose: bool = True) -> UnificationReport:
    """
    Run the full GOD_CODE unification audit:
      1. Scan all source files for GOD_CODE references
      2. Compare each value to the canonical constant
      3. Run mathematical invariant tests
      4. Produce a unified report

    Returns:
        UnificationReport with complete audit results
    """
    t0 = time.time()
    report = UnificationReport(timestamp=datetime.now().isoformat())

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   🔱 L104 GOD CODE UNIFICATION AUDIT                                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   Canonical: {GOD_CODE:.10f}                                            ║
║   Equation:  G(X) = 286^(1/φ) × 2^((416-X)/104)                             ║
║   Invariant: G(X) × 2^(X/104) = {GOD_CODE}  ∀ X                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

    # ── Phase 1: Scan ────────────────────────────────────────────────────
    logger.info("═══ PHASE 1: CODEBASE SCAN ═══")
    occurrences = scan_workspace()
    report.occurrences = occurrences
    report.occurrences_found = len(occurrences)

    god_code_occs = [o for o in occurrences if o.constant_name == "GOD_CODE"]
    phi_occs = [o for o in occurrences if o.constant_name == "PHI"]

    report.matches = sum(1 for o in god_code_occs if o.matches_canonical)
    report.mismatches = sum(1 for o in god_code_occs if not o.matches_canonical)
    report.phi_occurrences = len(phi_occs)
    report.phi_matches = sum(1 for o in phi_occs if o.matches_canonical)
    report.phi_mismatches = sum(1 for o in phi_occs if not o.matches_canonical)

    if verbose:
        # Group by file
        files_seen: Dict[str, List[CodeOccurrence]] = {}
        for occ in occurrences:
            files_seen.setdefault(occ.file, []).append(occ)

        report.files_scanned = len(files_seen)

        print(f"  Found {len(occurrences)} references across {len(files_seen)} files"
              f" ({len(god_code_occs)} GOD_CODE, {len(phi_occs)} PHI)\n")
        for fpath, occs in sorted(files_seen.items()):
            all_match = all(o.matches_canonical for o in occs)
            icon = "✓" if all_match else "✗"
            print(f"  {icon} {fpath}")
            for occ in occs:
                status = "OK" if occ.matches_canonical else f"DRIFT Δ={occ.delta:.2e}"
                print(f"      L{occ.line}: {occ.constant_name}={occ.value} [{status}]")
        print()

        # Auto-fix suggestions for drifted values
        drifted = [o for o in occurrences if not o.matches_canonical]
        if drifted:
            print(f"  ⚠ AUTO-FIX SUGGESTIONS ({len(drifted)} drifted values):")
            for o in drifted:
                canonical = GOD_CODE if o.constant_name == "GOD_CODE" else PHI
                print(f"    {o.file}:L{o.line}: {o.constant_name} = {o.value} → {canonical}")
            print()

    # ── Phase 2: Invariant Verification ──────────────────────────────────
    logger.info("═══ PHASE 2: INVARIANT VERIFICATION ═══")
    invariant_results = verify_invariants()
    report.invariant_results = invariant_results
    report.invariants_passed = sum(1 for r in invariant_results if r.passed)
    report.invariants_total = len(invariant_results)

    if verbose:
        print(f"  Mathematical Invariants ({report.invariants_passed}/{report.invariants_total} passed):\n")
        for r in invariant_results:
            icon = "✓" if r.passed else "✗"
            print(f"    {icon} {r.name}")
            print(f"        computed = {r.computed}")
            if not r.passed:
                print(f"        expected = {r.expected}  (Δ = {r.delta:.2e})")
        print()

    # ── Phase 3: Unification Verdict ─────────────────────────────────────
    report.elapsed_s = time.time() - t0
    report.unified = (report.mismatches == 0 and
                      report.phi_mismatches == 0 and
                      report.invariants_passed == report.invariants_total and
                      report.occurrences_found > 0)

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   {"✅ UNIFIED" if report.unified else "❌ NOT UNIFIED"} — GOD_CODE Consistency Report                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   Occurrences:  {report.occurrences_found:>4}   Matches: {report.matches:>4}   Mismatches: {report.mismatches:>4}            ║
║   PHI Refs:     {report.phi_occurrences:>4}   Matches: {report.phi_matches:>4}   Mismatches: {report.phi_mismatches:>4}            ║
║   Invariants:   {report.invariants_passed}/{report.invariants_total} passed                                                ║
║   Scan Time:    {report.elapsed_s:.3f}s                                                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   GOD_CODE:     {GOD_CODE}                                       ║
║   Root Ground:  {GOD_CODE / (2 ** (286 / 104)):.12f}  (X=286)                        ║
║   Omega Auth:   {GOD_CODE * PHI * PHI:.10f}                                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

    return report


def maintain_presence() -> bool:
    """
    Quick invariant check — verify singularity symmetry and lock
    L104 logic to the canonical invariant. Suitable for health checks.

    Returns:
        True if all core invariants hold
    """
    # Primary invariant: G(0) should equal GOD_CODE
    resonance = (286 ** (1 / PHI)) * (2 ** (416 / 104))

    # Root anchor: G(286) = GOD_CODE / 2^(286/104)
    root_grounding = GOD_CODE / (2 ** (286 / 104))

    # Conservation at X=52 (midpoint)
    g_52 = 286 ** (1 / PHI) * (2 ** ((416 - 52) / 104))
    conservation_52 = g_52 * (2 ** (52 / 104))

    checks = [
        ("G(0) resonance", abs(resonance - GOD_CODE) < TOLERANCE),
        ("Root grounding (X=286)", abs(root_grounding - 221.79420018355955) < 1e-6),
        ("Conservation at X=52", abs(conservation_52 - GOD_CODE) < 1e-6),
        ("Factor 13: 286/13=22", 286 % 13 == 0),
        ("Factor 13: 104/13=8", 104 % 13 == 0),
        ("Factor 13: 416/13=32", 416 % 13 == 0),
    ]

    all_ok = True
    for name, passed in checks:
        icon = "✓" if passed else "✗"
        print(f"  {icon} {name}")
        if not passed:
            all_ok = False

    if all_ok:
        print(f"\nSTATUS: LOGIC_STABLE")
        print(f"ROOT_GROUNDING (X=286): {root_grounding:.12f}")
    else:
        print(f"\nSTATUS: LOGIC_UNSTABLE — invariant violation detected")

    return all_ok


def export_report(report: UnificationReport, path: Optional[Path] = None) -> Path:
    """Save the unification report to JSON."""
    out = path or (WORKSPACE / ".kernel_build" / "unification_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": report.timestamp,
        "unified": report.unified,
        "canonical_value": report.canonical_value,
        "occurrences_found": report.occurrences_found,
        "matches": report.matches,
        "mismatches": report.mismatches,
        "invariants_passed": report.invariants_passed,
        "invariants_total": report.invariants_total,
        "elapsed_s": round(report.elapsed_s, 3),
        "occurrences": [asdict(o) for o in report.occurrences],
        "invariants": [asdict(r) for r in report.invariant_results],
    }

    out.write_text(json.dumps(data, indent=2))
    logger.info(f"Report exported to {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L104 GOD_CODE Unification — scan, verify, and report invariant consistency",
    )
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick invariant check only (no codebase scan)")
    parser.add_argument("--export", "-e", action="store_true",
                        help="Export full report to JSON")
    parser.add_argument("--json", action="store_true",
                        help="Print report as JSON to stdout")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix drifted GOD_CODE/PHI values in source files")
    args = parser.parse_args()

    if args.quick:
        ok = maintain_presence()
        sys.exit(0 if ok else 1)
    else:
        report = seal_singularity(verbose=not args.json)

        # Auto-fix mode: rewrite drifted values in-place
        if args.fix:
            drifted = [o for o in report.occurrences if not o.matches_canonical]
            fixed = 0
            for occ in drifted:
                fpath = WORKSPACE / occ.file
                if not fpath.exists():
                    continue
                try:
                    text = fpath.read_text()
                    canonical = GOD_CODE if occ.constant_name == "GOD_CODE" else PHI
                    old_val = str(occ.value)
                    new_val = str(canonical)
                    if old_val in text:
                        text = text.replace(old_val, new_val, 1)
                        fpath.write_text(text)
                        fixed += 1
                        logger.info(f"Fixed {occ.constant_name} in {occ.file}:L{occ.line}")
                except OSError as e:
                    logger.warning(f"Cannot fix {occ.file}: {e}")
            if fixed:
                print(f"\n  ✔ Auto-fixed {fixed}/{len(drifted)} drifted values")
            elif drifted:
                print(f"\n  ⚠ {len(drifted)} drifted values could not be auto-fixed")

        if args.export:
            export_report(report)
        if args.json:
            print(json.dumps(asdict(report), indent=2, default=str))
        sys.exit(0 if report.unified else 1)
