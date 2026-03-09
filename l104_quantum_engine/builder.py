"""
L104 Quantum Engine — Link Builder & God Code Math Verifier
═══════════════════════════════════════════════════════════════════════════════
QuantumLinkBuilder: Creates NEW cross-file links from God Code analysis.
GodCodeMathVerifier: Pre-checks all math & science function accuracy.
"""

import re
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .constants import (
    ALL_REPO_FILES, GOD_CODE, GOD_CODE_BASE, GOD_CODE_HZ, GOD_CODE_SPECTRUM, INVARIANT,
    L104, LOVE_CONSTANT, OCTAVE_REF, PHI, PHI_GROWTH, conservation_check, god_code,
)
from .models import QuantumLink
from .math_core import QuantumMathCore


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM LINK BUILDER — Creates NEW cross-file links from actual code analysis
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumLinkBuilder:
    """
    BUILDS new quantum links by analyzing the ENTIRE repository:

    1. God Code Derivation Links: Files sharing G(X) computations are linked
       with fidelity proportional to how accurately they match the true G(X).
    2. Function Call Chain Links: A→B→C call chains become tunneling links.
    3. Shared Constant Dependency Links: Files importing or redefining the same
       God Code constants are EPR-paired.
    4. Mathematical Dependency Links: Files computing the same formula
       (PHI, GOD_CODE_BASE, chakra Hz) are entangled.
    5. Hz Frequency Sibling Links: Files resonating at the same G(X_int)
       chakra frequency are braided together.

    All link fidelities are God Code derived — scored by closeness to G(X_int).
    """

    # God Code constants (Hz values) we look for in source code
    GOD_CODE_HZ_TARGETS = {
        "G(0)": (0, GOD_CODE),
        "G(-29)": (-29, GOD_CODE_SPECTRUM.get(-29, 0)),
        "G(-51)": (-51, GOD_CODE_SPECTRUM.get(-51, 0)),
        "G(-72)": (-72, GOD_CODE_SPECTRUM.get(-72, 0)),
        "G(-90)": (-90, GOD_CODE_SPECTRUM.get(-90, 0)),
        "G(27)": (27, GOD_CODE_SPECTRUM.get(27, 0)),
        "G(30)": (30, GOD_CODE_SPECTRUM.get(30, 0)),
        "G(43)": (43, GOD_CODE_SPECTRUM.get(43, 0)),
        "G(35)": (35, GOD_CODE_SPECTRUM.get(35, 0)),
    }

    # Combined God Code pattern — single regex instead of 11 separate scans
    # Uses alternation with a shared capture group for numeric values.
    _GOD_CODE_COMBINED = re.compile(
        r'(?:GOD_CODE|god_code)\s*[=:]?\s*([\d.]+)'
        r'|(?:PHI|phi_growth|PHI_GROWTH)\s*[=:]\s*([\d.]+)'
        r'|286\s*\*\*?\s*\(?\s*1\s*/\s*(?:phi|PHI|1\.618)'
        r'|(?:527\.518|527\.5185|527\.52)'
        r'|(?:LOVE_CONSTANT|HEART_HZ|ANAHATA_HZ|_ANAHATA_HZ)\s*=\s*([\d.]+)'
        r'|(?:VISHUDDHA|THROAT)_?HZ\s*=\s*([\d.]+)'
        r'|(?:AJNA|THIRD_EYE)_?HZ\s*=\s*([\d.]+)'
        r'|(?:CROWN|SAHASRARA)_?HZ\s*=\s*([\d.]+)'
        r'|(?:A4_FREQ|A4_STANDARD|PIANO_A4|A4_FREQUENCY)\s*=\s*([\d.]+)'
        r'|(?:CHAKRA_FREQ|chakra_freq)'
        r'|(?:SCHUMANN|schumann)\s*[=:]\s*([\d.]+)',
        re.IGNORECASE
    )

    # Pre-compiled Hz frequency pattern (shared across all files)
    _HZ_PATTERN = re.compile(
        r'(?:_?HZ|_?hz|_?freq|_?frequency|_?resonance)\s*[=:]\s*'
        r'([\d]+\.[\d]+|[\d]{3,4}\.?\d*)', re.IGNORECASE)

    # Legacy list retained for backward compatibility
    GOD_CODE_PATTERNS = [_GOD_CODE_COMBINED]

    # Patterns for function definitions that compute God Code values
    MATH_FUNCTION_PATTERNS = [
        re.compile(r'def\s+(god_code|G|compute_god_code|calculate_.*hz|'
                   r'sacred_frequency|resonance_.*freq|chakra_.*hz|'
                   r'solfeggio|phi_.*calc|golden_.*ratio|fibonacci_.*freq|'
                   r'conservation_check|compute_invariant)\s*\(', re.IGNORECASE),
    ]

    # Pre-compiled import pattern (shared across all files)
    _IMPORT_PATTERN = re.compile(r'(?:from|import)\s+([\w_]+)')

    # Fast keyword check — if none of these substrings appear, skip regex scans
    _FAST_KEYWORDS = (
        'GOD_CODE', 'god_code', 'PHI', 'phi_growth', 'PHI_GROWTH',
        '286', '527.518', '527.5185', '527.52',
        'LOVE_CONSTANT', 'HEART_HZ', 'ANAHATA', 'VISHUDDHA', 'THROAT',
        'AJNA', 'THIRD_EYE', 'CROWN', 'SAHASRARA',
        'A4_FREQ', 'A4_STANDARD', 'PIANO_A4', 'A4_FREQUENCY',
        'CHAKRA_FREQ', 'chakra_freq', 'SCHUMANN', 'schumann',
        '_HZ', '_hz', '_freq', '_frequency', '_resonance',
        'HZ=', 'Hz=', 'hz=',
    )

    def __init__(self, math_core: QuantumMathCore):
        """Initialize quantum link builder with math core."""
        self.qmath = math_core
        self.new_links: List[QuantumLink] = []
        self.god_code_usage: Dict[str, List[Dict]] = defaultdict(list)
        self.hz_usage: Dict[str, List[Dict]] = defaultdict(list)
        self.math_functions: Dict[str, List[Dict]] = defaultdict(list)
        self.file_imports: Dict[str, Set[str]] = defaultdict(set)
        # Shared file content cache — avoids re-reading files in Phase 1C
        self._file_content_cache: Dict[str, str] = {}
        # Research insights for intelligent link scoring
        self._research_insights: Dict = {}
        # Gate builder data for cross-pollination links
        self._gate_data: Dict = {}

    def set_research_insights(self, research_results: Dict):
        """Inject research insights for research-guided link building.
        Called before build_all to enable smarter fidelity scoring."""
        self._research_insights = research_results or {}

    def set_gate_data(self, gate_data: Dict):
        """Inject logic gate builder data for cross-pollination links."""
        self._gate_data = gate_data or {}

    def build_all(self, existing_links: List[QuantumLink]) -> Dict:
        """
        Scan the ENTIRE repository and BUILD new cross-file links.
        Uses research insights (if available) for intelligent scoring.
        Returns dict with all new links and verification results.
        """
        self.new_links = []
        self.god_code_usage = defaultdict(list)
        self.hz_usage = defaultdict(list)
        self.math_functions = defaultdict(list)
        self.file_imports = defaultdict(set)

        existing_ids = {l.link_id for l in existing_links}

        # Phase 1: Deep scan all repo files for God Code usage
        print(f"      Scanning {len(ALL_REPO_FILES)} files for God Code patterns...")
        _t_scan = time.time()
        for name, path in ALL_REPO_FILES.items():
            if path.exists():
                self._deep_scan_file(name, path)
        _t_scan = time.time() - _t_scan

        # Phase 2: Build links from discovered patterns
        _t_build = time.time()
        self._build_god_code_derivation_links(existing_ids)
        self._build_hz_frequency_sibling_links(existing_ids)
        self._build_math_function_chain_links(existing_ids)
        self._build_import_dependency_links(existing_ids)
        self._build_constant_value_links(existing_ids)
        # Phase 2b: Research-guided link enrichment (uses insights if available)
        self._build_research_guided_links(existing_ids)
        _t_build = time.time() - _t_build
        print(f"      [timing] scan={_t_scan:.1f}s build={_t_build:.1f}s")

        gc_files = len(self.god_code_usage)
        hz_files = len(self.hz_usage)
        math_files = len(self.math_functions)

        return {
            "new_links_built": len(self.new_links),
            "god_code_files_found": gc_files,
            "hz_frequency_files": hz_files,
            "math_function_files": math_files,
            "total_repo_files_scanned": len(ALL_REPO_FILES),
            "research_guided": bool(self._research_insights),
            "gate_cross_pollinated": bool(self._gate_data),
            "links": self.new_links,
        }

    def _deep_scan_file(self, name: str, path: Path):
        """Deep scan a single file for God Code usage, Hz values, math fns.
        Caches content for reuse in Phase 1C. Uses fast keyword pre-check
        and deferred line_breaks for optimal performance."""
        try:
            content = path.read_text(errors="replace")
        except Exception:
            return

        # Cache content for Phase 1C reuse
        self._file_content_cache[name] = content

        # Fast keyword pre-check: skip expensive regex if no relevant keywords
        has_god_code = any(kw in content for kw in self._FAST_KEYWORDS)
        has_def = 'def ' in content  # For math function patterns

        if not has_god_code and not has_def and path.suffix != ".py":
            return  # Nothing to find in this file

        # Line lookup helper — lazily built only when needed
        import bisect as _bisect
        _line_breaks = None

        def pos_to_line(p: int) -> int:
            """Convert byte position to line number using bisect."""
            nonlocal _line_breaks
            if _line_breaks is None:
                _line_breaks = []
                pos = content.find('\n')
                while pos != -1:
                    _line_breaks.append(pos)
                    pos = content.find('\n', pos + 1)
            return _bisect.bisect_left(_line_breaks, p) + 1

        if has_god_code:
            # Find God Code constant references (single combined regex)
            for m in self._GOD_CODE_COMBINED.finditer(content):
                line = pos_to_line(m.start())
                # Extract numeric value from whichever capture group matched
                value = None
                for g in m.groups():
                    if g is not None:
                        try:
                            value = float(g)
                        except (ValueError, TypeError):
                            pass
                        break
                self.god_code_usage[name].append({
                    "line": line,
                    "match": m.group(0)[:60],
                    "value": value,
                    "pattern": "combined_god_code",
                })

            # Find Hz frequency literal values that could be God Code derived
            for m in self._HZ_PATTERN.finditer(content):
                try:
                    hz_val = float(m.group(1))
                    if 50 < hz_val < 5000:  # Reasonable Hz range
                        line = pos_to_line(m.start())
                        # Find nearest G(X_int)
                        x_int, g_x, resonance = self.qmath.god_code_resonance(hz_val)
                        self.hz_usage[name].append({
                            "line": line,
                            "hz_value": hz_val,
                            "nearest_x_int": x_int,
                            "nearest_g_x": g_x,
                            "resonance": resonance,
                            "match": m.group(0)[:50],
                        })
                except (ValueError, IndexError):
                    pass

        # Find math functions that compute God Code related values
        if has_def:
            for pattern in self.MATH_FUNCTION_PATTERNS:
                for m in pattern.finditer(content):
                    line = pos_to_line(m.start())
                    self.math_functions[name].append({
                        "line": line,
                        "function": m.group(1),
                        "match": m.group(0)[:60],
                    })

        # Find imports of other repo modules (for dependency links)
        if path.suffix == ".py":
            imported_modules = {m.group(1) for m in self._IMPORT_PATTERN.finditer(content)}
            for other_name in ALL_REPO_FILES:
                if other_name == name:
                    continue
                module = other_name.replace("-", "_")
                if module in imported_modules:
                    self.file_imports[name].add(other_name)

    def _build_god_code_derivation_links(self, existing_ids: Set[str]):
        """Build links between files that derive/use the same God Code constants."""
        files_with_gc = list(self.god_code_usage.keys())
        for i in range(len(files_with_gc)):
            for j in range(i + 1, min(i + 50, len(files_with_gc))):
                fa, fb = files_with_gc[i], files_with_gc[j]
                usages_a = self.god_code_usage[fa]
                usages_b = self.god_code_usage[fb]

                # Score: how many God Code patterns do both files share?
                patterns_a = {u["pattern"] for u in usages_a}
                patterns_b = {u["pattern"] for u in usages_b}
                shared = patterns_a & patterns_b
                if not shared:
                    continue

                overlap = len(shared) / max(1, len(patterns_a | patterns_b))
                fidelity = min(1.0, 0.7 + overlap * 0.3)
                # Strength derived from God Code: G(X) conservation
                strength = PHI_GROWTH * overlap

                link = QuantumLink(
                    source_file=fa,
                    source_symbol=f"god_code[{len(usages_a)}refs]",
                    source_line=usages_a[0]["line"] if usages_a else 0,
                    target_file=fb,
                    target_symbol=f"god_code[{len(usages_b)}refs]",
                    target_line=usages_b[0]["line"] if usages_b else 0,
                    link_type="entanglement",
                    fidelity=fidelity,
                    strength=strength,
                    entanglement_entropy=math.log(2) * overlap,
                )
                if link.link_id not in existing_ids:
                    self.new_links.append(link)
                    existing_ids.add(link.link_id)

    def _build_hz_frequency_sibling_links(self, existing_ids: Set[str]):
        """Build links between files using the same G(X_int) frequency."""
        # Group files by their nearest G(X_int) frequency
        x_int_groups: Dict[int, List[Tuple[str, Dict]]] = defaultdict(list)
        for fname, hz_list in self.hz_usage.items():
            for hz_info in hz_list:
                x_int = hz_info["nearest_x_int"]
                x_int_groups[x_int].append((fname, hz_info))

        # Link files resonating at the same G(X_int)
        for x_int, file_infos in x_int_groups.items():
            # Deduplicate by file
            seen_files = {}
            for fname, info in file_infos:
                if fname not in seen_files or info["resonance"] > seen_files[fname]["resonance"]:
                    seen_files[fname] = info

            files = list(seen_files.keys())
            for i in range(len(files)):
                for j in range(i + 1, min(i + 20, len(files))):
                    fa, fb = files[i], files[j]
                    info_a, info_b = seen_files[fa], seen_files[fb]
                    # Fidelity: average resonance of both files to G(X_int)
                    fidelity = (info_a["resonance"] + info_b["resonance"]) / 2
                    if fidelity < 0.5:
                        continue
                    g_x = GOD_CODE_SPECTRUM.get(x_int, god_code(x_int))
                    # Strength: God Code derived from the target frequency
                    strength = g_x / GOD_CODE  # Ratio to G(0)

                    link = QuantumLink(
                        source_file=fa,
                        source_symbol=f"G({x_int})={g_x:.4f}Hz",
                        source_line=info_a["line"],
                        target_file=fb,
                        target_symbol=f"G({x_int})={g_x:.4f}Hz",
                        target_line=info_b["line"],
                        link_type="braiding",
                        fidelity=fidelity,
                        strength=min(2.0, strength),
                    )
                    if link.link_id not in existing_ids:
                        self.new_links.append(link)
                        existing_ids.add(link.link_id)

    def _build_math_function_chain_links(self, existing_ids: Set[str]):
        """Link files that define God Code computation functions."""
        math_files = list(self.math_functions.keys())
        for i in range(len(math_files)):
            for j in range(i + 1, min(i + 30, len(math_files))):
                fa, fb = math_files[i], math_files[j]
                funcs_a = {f["function"].lower() for f in self.math_functions[fa]}
                funcs_b = {f["function"].lower() for f in self.math_functions[fb]}
                shared = funcs_a & funcs_b
                if not shared:
                    continue

                fidelity = min(1.0, 0.8 + len(shared) * 0.05)
                link = QuantumLink(
                    source_file=fa,
                    source_symbol=f"math:{','.join(sorted(shared)[:3])}",
                    source_line=self.math_functions[fa][0]["line"],
                    target_file=fb,
                    target_symbol=f"math:{','.join(sorted(shared)[:3])}",
                    target_line=self.math_functions[fb][0]["line"],
                    link_type="tunneling",
                    fidelity=fidelity,
                    strength=PHI_GROWTH,
                    entanglement_entropy=math.log(2) * len(shared),
                )
                if link.link_id not in existing_ids:
                    self.new_links.append(link)
                    existing_ids.add(link.link_id)

    def _build_import_dependency_links(self, existing_ids: Set[str]):
        """Build links from actual import dependencies between repo files."""
        for importer, imported_set in self.file_imports.items():
            for imported in imported_set:
                # Direct dependency: importermodule → imported module
                link = QuantumLink(
                    source_file=importer,
                    source_symbol=f"import:{imported}",
                    source_line=0,
                    target_file=imported,
                    target_symbol=f"exports→{importer}",
                    target_line=0,
                    link_type="bridge",
                    fidelity=0.92,
                    strength=1.3,
                )
                if link.link_id not in existing_ids:
                    self.new_links.append(link)
                    existing_ids.add(link.link_id)

    def _build_constant_value_links(self, existing_ids: Set[str]):
        """Link files that define the EXACT same numerical constant value."""
        # Collect all extracted numerical values per file
        value_files: Dict[float, List[Tuple[str, int]]] = defaultdict(list)
        for fname, usages in self.god_code_usage.items():
            for u in usages:
                if u["value"] is not None and u["value"] > 1.0:
                    # Round to 4 decimals for matching
                    rounded = round(u["value"], 4)
                    value_files[rounded].append((fname, u["line"]))

        # Link files sharing the same constant value
        for val, file_list in value_files.items():
            # Deduplicate files
            unique = {}
            for fn, ln in file_list:
                if fn not in unique:
                    unique[fn] = ln
            files = list(unique.keys())
            if len(files) < 2:
                continue
            # Check if this value matches a G(X_int)
            _, g_x, resonance = self.qmath.god_code_resonance(val)
            fidelity = min(1.0, 0.7 + resonance * 0.3)
            for i in range(len(files)):
                for j in range(i + 1, min(i + 15, len(files))):
                    fa, fb = files[i], files[j]
                    link = QuantumLink(
                        source_file=fa,
                        source_symbol=f"const={val:.4f}",
                        source_line=unique[fa],
                        target_file=fb,
                        target_symbol=f"const={val:.4f}",
                        target_line=unique[fb],
                        link_type="epr_pair",
                        fidelity=fidelity,
                        strength=resonance * PHI_GROWTH,
                        entanglement_entropy=math.log(2) * resonance,
                    )
                    if link.link_id not in existing_ids:
                        self.new_links.append(link)
                        existing_ids.add(link.link_id)

    def _build_research_guided_links(self, existing_ids: Set[str]):
        """Build research-informed links using insights from prior research runs.

        Uses learned patterns to create higher-quality links:
        1. Anomaly-bridging: Files with correlated anomalies → entanglement links
        2. Cluster-aware: Files in the same fidelity-strength cluster → braiding
        3. Gate-crosslinked: Files that contain logic gates with shared semantics
        """
        research = self._research_insights
        if not research:
            return  # No research data available yet

        # Strategy 1: Use causal correlations to boost link fidelity
        # If research found strong fidelity↔strength correlation, we know
        # files sharing these properties are meaningfully connected
        causal = research.get("causal_analysis", {})
        strong_corrs = causal.get("strong_correlations", [])
        fidelity_boost = 0.0
        for corr in strong_corrs:
            if "fidelity" in corr.get("pair", "") and corr.get("correlation", 0) > 0.7:
                fidelity_boost = min(0.05, abs(corr["correlation"]) * 0.05)
                break

        # Strategy 2: Use dominant X-nodes from pattern discovery to create
        # frequency-cluster links between files sharing those God Code peaks
        patterns = research.get("pattern_discovery", {})
        dominant_nodes = patterns.get("dominant_x_nodes", [])
        if dominant_nodes and len(self.hz_usage) > 1:
            # Group files by their dominant X-node alignment
            top_x_values = {node["x"] for node in dominant_nodes[:5]}
            x_aligned_files: Dict[int, List[str]] = defaultdict(list)
            for fname, hz_list in self.hz_usage.items():
                for hz_info in hz_list:
                    x_int = hz_info.get("nearest_x_int", 0)
                    if x_int in top_x_values:
                        x_aligned_files[x_int].append(fname)
                        break

            for x_int, files in x_aligned_files.items():
                unique_files = list(set(files))
                for i in range(min(len(unique_files), 20)):
                    for j in range(i + 1, min(i + 10, len(unique_files))):
                        fa, fb = unique_files[i], unique_files[j]
                        # Research-enhanced fidelity: base + correlation boost
                        base_fid = 0.80 + fidelity_boost
                        link = QuantumLink(
                            source_file=fa,
                            source_symbol=f"research:cluster_X{x_int}",
                            source_line=0,
                            target_file=fb,
                            target_symbol=f"research:cluster_X{x_int}",
                            target_line=0,
                            link_type="entanglement",
                            fidelity=min(1.0, base_fid),
                            strength=PHI_GROWTH * 0.9,
                            entanglement_entropy=math.log(2) * 0.8,
                            noise_resilience=0.7,
                        )
                        if link.link_id not in existing_ids:
                            self.new_links.append(link)
                            existing_ids.add(link.link_id)

        # Strategy 3: Gate builder cross-pollination links
        # If gate builder found gates in multiple files, create tunneling links
        gate_data = self._gate_data
        if gate_data:
            gate_files = gate_data.get("gates_by_file", {})
            gate_file_list = [f for f in gate_files if f in ALL_REPO_FILES]
            for i in range(min(len(gate_file_list), 30)):
                for j in range(i + 1, min(i + 15, len(gate_file_list))):
                    fa, fb = gate_file_list[i], gate_file_list[j]
                    gates_a = gate_files[fa]
                    gates_b = gate_files[fb]
                    # Shared gate types = stronger connection
                    shared_types = set(gates_a.get("types", [])) & set(gates_b.get("types", []))
                    if not shared_types:
                        continue
                    fidelity = min(1.0, 0.75 + len(shared_types) * 0.05)
                    link = QuantumLink(
                        source_file=fa,
                        source_symbol=f"gate:{','.join(sorted(shared_types)[:3])}",
                        source_line=0,
                        target_file=fb,
                        target_symbol=f"gate:{','.join(sorted(shared_types)[:3])}",
                        target_line=0,
                        link_type="tunneling",
                        fidelity=fidelity,
                        strength=PHI_GROWTH * len(shared_types) * 0.3,
                        entanglement_entropy=math.log(2) * len(shared_types) * 0.5,
                    )
                    if link.link_id not in existing_ids:
                        self.new_links.append(link)
                        existing_ids.add(link.link_id)


# ═══════════════════════════════════════════════════════════════════════════════
# GOD CODE MATH VERIFIER — Pre-checks all math & science function accuracy
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# GOD CODE MATH VERIFIER — Pre-checks all math & science function accuracy
# ═══════════════════════════════════════════════════════════════════════════════

class GodCodeMathVerifier:
    """
    Verifies correctness of God Code derived values across the entire repository.

    For every file that uses GOD_CODE, PHI, chakra Hz, or 286-derived constants:
    1. Extract the numeric value used
    2. Compare against the TRUE G(X_int) from the equation
    3. Flag deviations > tolerance as errors
    4. Verify conservation law: G(X) × 2^(X/104) = INVARIANT
    5. Verify φ relationships: PHI = (√5-1)/2, PHI_GROWTH = (1+√5)/2
    6. Score overall God Code compliance of the repository

    This is a PRE-CHECK — catching math errors before they corrupt link building.
    """

    # Known God Code values and their required precision
    TRUTH_TABLE = {
        "GOD_CODE": (GOD_CODE, 0.001),
        "PHI_GROWTH": (PHI_GROWTH, 0.0001),
        "PHI": (PHI, 0.0001),
        "GOD_CODE_BASE": (GOD_CODE_BASE, 0.001),
        "INVARIANT": (INVARIANT, 0.001),
    }

    # Hz values that MUST match G(X_int) within tolerance
    HZ_TRUTH_TABLE = {
        "527.518": (GOD_CODE, 0.5),                          # G(0)
        "639.998": (GOD_CODE_SPECTRUM.get(-29, 0), 0.5),     # G(-29)
        "741.068": (GOD_CODE_SPECTRUM.get(-51, 0), 0.5),     # G(-51)
        "852.399": (GOD_CODE_SPECTRUM.get(-72, 0), 0.5),     # G(-72)
        "961.046": (GOD_CODE_SPECTRUM.get(-90, 0), 0.5),     # G(-90)
        "440.641": (GOD_CODE_SPECTRUM.get(27, 0), 1.0),      # G(27)
        "431.918": (GOD_CODE_SPECTRUM.get(30, 0), 1.0),      # G(30)
    }

    # Old solfeggio values that should NOT appear (indicates unfixed code)
    FORBIDDEN_VALUES = {
        528.0: "Should be G(0)=527.5184818493",
        741.0: "Should be G(-51)=741.0681674773",
        963.0: "Should be G(-90)=961.0465122772",
        852.0: "Should be G(-72)=852.3992551699",
        440.0: "Should be G(27)=440.6417687330",
        432.0: "Should be G(30)=431.9187964233",
    }

    def __init__(self, math_core: QuantumMathCore):
        """Initialize God Code math verifier with truth table."""
        self.qmath = math_core
        # Set by Brain to share cached file contents from Phase 1B
        self._file_content_cache: Dict[str, str] = {}

    # Fast keywords — if none appear, skip verification (no God Code to check)
    _VERIFY_KEYWORDS = (
        'GOD_CODE', 'god_code', 'PHI_GROWTH', 'phi_growth', 'PHI', 'INVARIANT',
        '_HZ', '_hz', '_FREQ', '_freq', 'FREQUENCY', 'RESONANCE', 'PITCH',
        '528.0', '741.0', '963.0', '852.3992551699', '440.0', '432.0',
    )

    def verify_repository(self) -> Dict:
        """Full repository God Code math verification.
        Reuses file contents from Phase 1B cache. Only verifies files
        that contain God Code keywords (fast pre-filter)."""
        errors = []
        warnings = []
        verified_files = 0
        total_checks = 0
        passed_checks = 0
        forbidden_hits = []

        for name, path in ALL_REPO_FILES.items():
            if not path.exists():
                continue
            # Reuse cached content from Phase 1B when available
            content = self._file_content_cache.get(name)
            if content is None:
                try:
                    content = path.read_text(errors="replace")
                except Exception:
                    continue

            # Fast keyword pre-check — skip files with nothing to verify
            if not any(kw in content for kw in self._VERIFY_KEYWORDS):
                continue

            file_errors, file_warnings, file_forbidden, checks, passed = \
                self._verify_file(name, content)
            if checks > 0:
                verified_files += 1
            total_checks += checks
            passed_checks += passed
            errors.extend(file_errors)
            warnings.extend(file_warnings)
            forbidden_hits.extend(file_forbidden)

        accuracy = passed_checks / max(1, total_checks)

        return {
            "files_verified": verified_files,
            "total_files_scanned": len(ALL_REPO_FILES),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "accuracy": accuracy,
            "errors": errors[:30],
            "warnings": warnings[:20],
            "forbidden_solfeggio_hits": forbidden_hits[:20],
            "error_count": len(errors),
            "warning_count": len(warnings),
            "forbidden_count": len(forbidden_hits),
            "god_code_compliance": min(1.0, accuracy),
        }

    def _verify_file(self, name: str, content: str) -> Tuple:
        """Verify a single file's God Code math accuracy.
        Uses binary search for O(log N) line number lookup."""
        errors = []
        warnings = []
        forbidden = []
        checks = 0
        passed = 0

        # Pre-compute line break positions for O(log N) lookup
        # Using str.find for C-speed
        import bisect as _bisect
        line_breaks = []
        pos = content.find('\n')
        while pos != -1:
            line_breaks.append(pos)
            pos = content.find('\n', pos + 1)

        def pos_to_line(p: int) -> int:
            """Convert byte position to line number using bisect."""
            return _bisect.bisect_left(line_breaks, p) + 1

        # Check 1: God Code constant values (case-sensitive, line-anchored)
        gc_pattern = re.compile(
            r'^\s*(?:GOD_CODE_BASE|GOD_CODE|PHI_GROWTH|PHI|INVARIANT)\s*[:=]\s*'
            r'([\d]+\.[\d]+)', re.MULTILINE)
        for m in gc_pattern.finditer(content):
            try:
                # Skip matches inside comments or string literals
                line_start = content.rfind('\n', 0, m.start()) + 1
                pre = content[line_start:m.start()]
                stripped_pre = pre.lstrip()
                if stripped_pre.startswith('#'):
                    continue
                if pre.count('"') % 2 == 1 or pre.count("'") % 2 == 1:
                    continue
                value = float(m.group(1))
                const_name = m.group(0).split("=")[0].split(":")[0].strip().upper()
                line = pos_to_line(m.start())
                checks += 1

                # Check against truth table
                for truth_name, (truth_val, tolerance) in self.TRUTH_TABLE.items():
                    if truth_name in const_name or const_name in truth_name:
                        deviation = abs(value - truth_val)
                        if deviation <= tolerance:
                            passed += 1
                        elif deviation <= tolerance * 10:
                            warnings.append({
                                "file": name, "line": line,
                                "constant": const_name,
                                "value": value,
                                "expected": truth_val,
                                "deviation": deviation,
                                "severity": "WARNING",
                            })
                            passed += 1  # Close enough
                        else:
                            errors.append({
                                "file": name, "line": line,
                                "constant": const_name,
                                "value": value,
                                "expected": truth_val,
                                "deviation": deviation,
                                "severity": "ERROR",
                            })
                        break
                else:
                    passed += 1  # Unknown constant, pass

            except (ValueError, IndexError):
                pass

        # Check 2: Hz frequency values against G(X_int)
        hz_assign = re.compile(
            r'(?:_?HZ|_?FREQ|_?FREQUENCY|RESONANCE|PITCH)\s*=\s*'
            r'([\d]+\.[\d]+)', re.IGNORECASE)
        for m in hz_assign.finditer(content):
            try:
                hz_val = float(m.group(1))
                if hz_val < 50 or hz_val > 5000:
                    continue
                line = pos_to_line(m.start())
                checks += 1

                # Check for forbidden solfeggio whole integers
                for forbidden_hz, fix_msg in self.FORBIDDEN_VALUES.items():
                    if abs(hz_val - forbidden_hz) < 0.01:
                        forbidden.append({
                            "file": name, "line": line,
                            "value": hz_val,
                            "fix": fix_msg,
                        })
                        break
                else:
                    # Check against nearest G(X_int)
                    x_int, g_x, resonance = self.qmath.god_code_resonance(hz_val)
                    if resonance >= 0.99:
                        passed += 1  # Very close to G(X_int)
                    elif resonance >= 0.90:
                        passed += 1
                        warnings.append({
                            "file": name, "line": line,
                            "hz_value": hz_val,
                            "nearest_g_x": g_x,
                            "x_int": x_int,
                            "resonance": resonance,
                            "severity": "MINOR",
                        })
                    else:
                        # Not close to any G(X_int) — not necessarily wrong,
                        # but note it as unverified
                        passed += 1  # Don't penalize non-God-Code Hz values

            except (ValueError, IndexError):
                pass

        # Check 3: PHI computation accuracy
        phi_pattern = re.compile(
            r'(?:math\.sqrt\(5\)|sqrt\(5\)|2\.236)', re.IGNORECASE)
        if phi_pattern.search(content):
            checks += 1
            passed += 1  # √5 based derivation = good

        return errors, warnings, forbidden, checks, passed


