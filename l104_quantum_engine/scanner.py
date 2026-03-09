"""
L104 Quantum Engine — Quantum Link Scanner
═══════════════════════════════════════════════════════════════════════════════
Discovers quantum links across the repository via AST analysis, symbol registry,
cross-file call tracking, sacred constants, and quantum keyword co-occurrence.
"""

import ast
import re
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Set

from .constants import (
    ALL_REPO_FILES, BELL_FIDELITY, CALABI_YAU_DIM, CHSH_BOUND, GOD_CODE,
    GROVER_AMPLIFICATION, OMEGA_POINT, PHI, PHI_GROWTH, QUANTUM_LINKED_FILES, TAU,
    god_code,
)
from .models import QuantumLink


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM LINK SCANNER — Discovers all quantum links across the repository
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumLinkScanner:
    """
    Discovers quantum links by scanning all source files for:
    - Shared function/class names (mirrors)
    - Cross-file function calls (entanglement)
    - Shared sacred constants (resonance links)
    - API endpoint pairs (bridge links)
    - Quantum keyword co-occurrence (spooky_action)
    """

    QUANTUM_KEYWORDS = {
        "grover", "bell", "epr", "entangle", "teleport", "decohere",
        "superposition", "qubit", "quantum", "hilbert", "fourier",
        "anyon", "braid", "tunnel", "coherence", "fidelity", "amplitude",
        "resonance", "chakra", "kundalini", "vishuddha", "phi", "god_code",
        "calabi_yau", "planck", "eigenvalue", "hamiltonian", "schrodinger",
        "wave_function", "collapse", "measurement", "density_matrix",
        "bloch_sphere", "pauli", "hadamard", "cnot", "swap",
    }

    SACRED_CONSTANTS = {
        "PHI", "TAU", "GOD_CODE", "OMEGA_POINT", "GROVER_AMPLIFICATION",
        "CALABI_YAU_DIM", "BELL_FIDELITY", "CHAKRA", "KUNDALINI",
        "VISHUDDHA", "EPR_LINK_STRENGTH", "PLANCK",
    }

    def __init__(self):
        """Initialize quantum link scanner with empty registries."""
        self.links: List[QuantumLink] = []
        self.symbol_registry: Dict[str, List[Dict]] = defaultdict(list)
        self.file_symbols: Dict[str, Set[str]] = defaultdict(set)
        self.quantum_density: Dict[str, float] = {}

    def full_scan(self) -> List[QuantumLink]:
        """Scan all quantum-linked files and discover every link."""
        print("\n  ⚛ [QUANTUM LINK SCANNER] Full repository scan...")
        self.links = []
        self.symbol_registry = defaultdict(list)
        self.file_symbols = defaultdict(set)

        # Phase 1: Extract symbols from core files (deep AST analysis)
        for name, path in QUANTUM_LINKED_FILES.items():
            if path.exists():
                self._extract_symbols(name, path)

        # Phase 2: Discover links
        self._discover_mirror_links()        # Same-name symbols across files
        self._discover_call_links()          # Function calls across files
        self._discover_constant_links()      # Shared sacred constants
        self._discover_quantum_keyword_links()  # Quantum keyword co-occurrence
        self._discover_api_bridge_links()    # API endpoint pairs
        self._compute_quantum_density()      # Per-file quantum density

        print(f"    ✓ Discovered {len(self.links)} quantum links across "
              f"{len(QUANTUM_LINKED_FILES)} core files")
        return self.links

    def _extract_symbols(self, name: str, path: Path):
        """Extract function/class/method symbols from a source file."""
        try:
            content = path.read_text(errors="replace")
        except Exception:
            return

        ext = path.suffix
        if ext == ".py":
            self._extract_python_symbols(name, content)
        elif ext == ".swift":
            self._extract_swift_symbols(name, content)
        elif ext == ".js":
            self._extract_js_symbols(name, content)

    def _extract_python_symbols(self, file_name: str, content: str):
        """AST-based Python symbol extraction."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                sym = node.name
                self.symbol_registry[sym].append({
                    "file": file_name, "line": node.lineno,
                    "type": "function", "language": "python"
                })
                self.file_symbols[file_name].add(sym)

            elif isinstance(node, ast.ClassDef):
                sym = node.name
                self.symbol_registry[sym].append({
                    "file": file_name, "line": node.lineno,
                    "type": "class", "language": "python"
                })
                self.file_symbols[file_name].add(sym)

                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_sym = f"{sym}.{item.name}"
                        self.symbol_registry[method_sym].append({
                            "file": file_name, "line": item.lineno,
                            "type": "method", "language": "python"
                        })
                        self.file_symbols[file_name].add(item.name)

    def _extract_swift_symbols(self, file_name: str, content: str):
        """Regex-based Swift symbol extraction with pre-computed line offsets."""
        # Pre-compute line break positions for O(1) line lookup
        line_breaks = [0]
        for i, ch in enumerate(content):
            if ch == '\n':
                line_breaks.append(i + 1)

        def pos_to_line(pos: int) -> int:
            """Convert byte position to line number via binary search."""
            lo, hi = 0, len(line_breaks) - 1
            while lo < hi:
                mid = (lo + hi + 1) >> 1
                if line_breaks[mid] <= pos:
                    lo = mid
                else:
                    hi = mid - 1
            return lo + 1

        patterns = [
            (r'(?:final\s+)?class\s+(\w+)', "class"),
            (r'struct\s+(\w+)', "struct"),
            (r'func\s+(\w+)', "function"),
            (r'enum\s+(\w+)', "enum"),
            (r'protocol\s+(\w+)', "protocol"),
        ]
        for pattern, sym_type in patterns:
            for m in re.finditer(pattern, content):
                sym = m.group(1)
                line = pos_to_line(m.start())
                self.symbol_registry[sym].append({
                    "file": file_name, "line": line,
                    "type": sym_type, "language": "swift"
                })
                self.file_symbols[file_name].add(sym)

    def _extract_js_symbols(self, file_name: str, content: str):
        """Regex-based JavaScript symbol extraction with pre-computed line offsets."""
        line_breaks = [0]
        for i, ch in enumerate(content):
            if ch == '\n':
                line_breaks.append(i + 1)

        def pos_to_line(pos: int) -> int:
            """Convert byte position to line number via binary search."""
            lo, hi = 0, len(line_breaks) - 1
            while lo < hi:
                mid = (lo + hi + 1) >> 1
                if line_breaks[mid] <= pos:
                    lo = mid
                else:
                    hi = mid - 1
            return lo + 1

        patterns = [
            (r'(?:function|const|let|var)\s+(\w+)', "function"),
            (r'class\s+(\w+)', "class"),
        ]
        for pattern, sym_type in patterns:
            for m in re.finditer(pattern, content):
                sym = m.group(1)
                line = pos_to_line(m.start())
                self.symbol_registry[sym].append({
                    "file": file_name, "line": line,
                    "type": sym_type, "language": "javascript"
                })
                self.file_symbols[file_name].add(sym)

    def _discover_mirror_links(self):
        """Find symbols that exist in multiple files (quantum mirrors)."""
        for sym, locations in self.symbol_registry.items():
            if len(locations) < 2:
                continue
            # Create links between all pairs
            for i in range(len(locations)):
                for j in range(i + 1, len(locations)):
                    a, b = locations[i], locations[j]
                    if a["file"] == b["file"]:
                        continue
                    # Cross-language mirrors are higher fidelity
                    cross_lang = a["language"] != b["language"]
                    base_fidelity = 0.95 if cross_lang else 0.85

                    self.links.append(QuantumLink(
                        source_file=a["file"], source_symbol=sym,
                        source_line=a["line"],
                        target_file=b["file"], target_symbol=sym,
                        target_line=b["line"],
                        link_type="mirror" if not cross_lang else "entanglement",
                        fidelity=base_fidelity,
                        strength=PHI_GROWTH if cross_lang else 1.0,
                        entanglement_entropy=math.log(2) if cross_lang else 0.5,
                    ))

    def _discover_call_links(self):
        """Find cross-file function calls through import/reference analysis."""
        for name, path in QUANTUM_LINKED_FILES.items():
            if not path.exists() or path.suffix != ".py":
                continue
            try:
                content = path.read_text(errors="replace")
            except Exception:
                continue

            # Find imports of other quantum-linked modules
            for other_name in QUANTUM_LINKED_FILES:
                if other_name == name:
                    continue
                module = other_name.replace(".py", "").replace(".swift", "")
                # Check for imports
                import_patterns = [
                    rf"from\s+{re.escape(module)}\s+import\s+(\w+)",
                    rf"import\s+{re.escape(module)}",
                ]
                for pat in import_patterns:
                    for m in re.finditer(pat, content):
                        sym = m.group(1) if m.lastindex else module
                        line = content[:m.start()].count('\n') + 1
                        self.links.append(QuantumLink(
                            source_file=name, source_symbol=f"import:{sym}",
                            source_line=line,
                            target_file=other_name, target_symbol=sym,
                            target_line=0,
                            link_type="bridge",
                            fidelity=0.90, strength=1.2,
                        ))

    def _discover_constant_links(self):
        """Find shared sacred constants across files."""
        file_constants: Dict[str, Set[str]] = defaultdict(set)

        for name, path in QUANTUM_LINKED_FILES.items():
            if not path.exists():
                continue
            try:
                content = path.read_text(errors="replace")
            except Exception:
                continue
            for const in self.SACRED_CONSTANTS:
                if const in content:
                    file_constants[name].add(const)

        # Create resonance links for shared constants
        files = list(file_constants.keys())
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                shared = file_constants[files[i]] & file_constants[files[j]]
                if shared:
                    overlap = len(shared) / len(self.SACRED_CONSTANTS)
                    self.links.append(QuantumLink(
                        source_file=files[i],
                        source_symbol=f"constants:{','.join(sorted(shared)[:3])}",
                        source_line=0,
                        target_file=files[j],
                        target_symbol=f"constants:{','.join(sorted(shared)[:3])}",
                        target_line=0,
                        link_type="epr_pair",
                        fidelity=min(1.0, 0.5 + overlap * PHI_GROWTH * 0.3),
                        strength=overlap * PHI_GROWTH,
                        entanglement_entropy=math.log(2) * overlap,
                    ))

    def _discover_quantum_keyword_links(self):
        """Discover spooky-action links through quantum keyword co-occurrence."""
        file_keywords: Dict[str, Set[str]] = defaultdict(set)

        for name, path in QUANTUM_LINKED_FILES.items():
            if not path.exists():
                continue
            try:
                content = path.read_text(errors="replace").lower()
            except Exception:
                continue
            for kw in self.QUANTUM_KEYWORDS:
                if kw in content:
                    file_keywords[name].add(kw)

        # Spooky action: highly correlated keyword patterns
        files = list(file_keywords.keys())
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                shared_kw = file_keywords[files[i]] & file_keywords[files[j]]
                if len(shared_kw) >= 5:
                    correlation = len(shared_kw) / len(self.QUANTUM_KEYWORDS)
                    self.links.append(QuantumLink(
                        source_file=files[i],
                        source_symbol=f"quantum_keywords[{len(shared_kw)}]",
                        source_line=0,
                        target_file=files[j],
                        target_symbol=f"quantum_keywords[{len(shared_kw)}]",
                        target_line=0,
                        link_type="spooky_action",
                        fidelity=min(1.0, correlation * PHI_GROWTH),
                        strength=correlation * GROVER_AMPLIFICATION,
                        bell_violation=CHSH_BOUND * correlation,
                    ))

    def _discover_api_bridge_links(self):
        """Discover API endpoint bridges between server and clients."""
        endpoint_files: Dict[str, List[str]] = defaultdict(list)
        api_pattern = re.compile(r'["\'/]api/v\d+/(\w+(?:/\w+)*)')

        for name, path in QUANTUM_LINKED_FILES.items():
            if not path.exists():
                continue
            try:
                content = path.read_text(errors="replace")
            except Exception:
                continue
            for m in api_pattern.finditer(content):
                endpoint = m.group(1)
                line = content[:m.start()].count('\n') + 1
                endpoint_files[endpoint].append(f"{name}:{line}")

        for endpoint, locations in endpoint_files.items():
            if len(locations) >= 2:
                for i in range(len(locations)):
                    for j in range(i + 1, len(locations)):
                        file_a, line_a = locations[i].rsplit(":", 1)
                        file_b, line_b = locations[j].rsplit(":", 1)
                        if file_a != file_b:
                            self.links.append(QuantumLink(
                                source_file=file_a,
                                source_symbol=f"api:{endpoint}",
                                source_line=int(line_a),
                                target_file=file_b,
                                target_symbol=f"api:{endpoint}",
                                target_line=int(line_b),
                                link_type="bridge",
                                fidelity=0.92,
                                strength=1.5,
                            ))

    def _compute_quantum_density(self):
        """Compute quantum density (quantum symbols / total symbols) per file."""
        for name in QUANTUM_LINKED_FILES:
            syms = self.file_symbols.get(name, set())
            if not syms:
                self.quantum_density[name] = 0.0
                continue
            quantum_count = sum(1 for s in syms
                                if any(kw in s.lower()
                                       for kw in self.QUANTUM_KEYWORDS))
            self.quantum_density[name] = quantum_count / max(1, len(syms))


# ═══════════════════════════════════════════════════════════════════════════════
# GROVER QUANTUM PROCESSOR — Amplified search and verification across links
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM LINK BUILDER — Creates NEW cross-file links from actual code analysis
# ═══════════════════════════════════════════════════════════════════════════════

