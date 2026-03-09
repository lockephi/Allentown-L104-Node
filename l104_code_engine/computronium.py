"""
L104 Code Engine — Computronium & Rayleigh Code Intelligence
═══════════════════════════════════════════════════════════════════════════════
Analyzes code through the lens of fundamental physical computation limits.

COMPUTRONIUM ANALYSIS:
  • Algorithm complexity → Margolus-Levitin gate budget
  • Memory allocation → Bekenstein capacity fraction
  • Data throughput → Bremermann bandwidth utilization
  • Energy efficiency → Landauer erasure overhead
  • Parallel scaling → Amdahl-computronium efficiency

RAYLEIGH-INSPIRED CODE RESOLUTION:
  • Code signal-to-noise (precision of variable naming/typing)
  • Feature resolution (minimum distinguishable abstraction granularity)
  • Information density per line of code
  • Diffraction-limited refactoring (Abbe limit on decomposition)
  • Ultraviolet catastrophe detection (unbounded resource growth)

The key insight: just as Rayleigh and computronium limits constrain
physical systems, analogous limits constrain code systems. This module
quantifies how close code operates to its theoretical optimum.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import ast
import math
import re
import time
from typing import Dict, Any, List, Optional

from .constants import (
    GOD_CODE, PHI, TAU, VOID_CONSTANT, BOLTZMANN_K, PLANCK_SCALE,
    VERSION,
)

# Physical constants for computronium calculations (CODATA 2022)
_H_BAR = 1.054571817e-34
_C = 299792458
_K_B = 1.380649e-23


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPLEXITY → COMPUTRONIUM MAPPER
# ═══════════════════════════════════════════════════════════════════════════════

# Algorithmic complexity classes ranked by ops-growth
COMPLEXITY_CLASSES = {
    "O(1)":          {"rank": 0, "growth": lambda n: 1},
    "O(log n)":      {"rank": 1, "growth": lambda n: math.log2(max(n, 1))},
    "O(√n)":         {"rank": 2, "growth": lambda n: math.sqrt(n)},
    "O(n)":          {"rank": 3, "growth": lambda n: n},
    "O(n log n)":    {"rank": 4, "growth": lambda n: n * math.log2(max(n, 1))},
    "O(n²)":         {"rank": 5, "growth": lambda n: n ** 2},
    "O(n³)":         {"rank": 6, "growth": lambda n: n ** 3},
    "O(2^n)":        {"rank": 7, "growth": lambda n: 2 ** min(n, 60)},
    "O(n!)":         {"rank": 8, "growth": lambda n: math.factorial(min(int(n), 20))},
}


class ComputroniumCodeAnalyzer:
    """
    Analyzes source code against fundamental physical computation limits.

    Maps algorithmic complexity to computronium bounds:
    - How many operations at the Margolus-Levitin limit?
    - What fraction of Bekenstein memory does the algorithm need?
    - What's the Landauer energy cost of its erasures?
    - Does it exhibit ultraviolet catastrophe (unbounded growth)?
    """

    def __init__(self):
        self.analyses = 0

    def analyze_computronium_budget(self, source: str,
                                       input_size: int = 1000000,
                                       temperature_K: float = 293.15,
                                       device_mass_kg: float = 0.5) -> Dict[str, Any]:
        """
        Full computronium budget analysis for source code.

        Estimates how the code's computational demands compare to the
        fundamental physical limits of the hardware it runs on.

        Parameters:
            source: Python source code
            input_size: Expected input size N for complexity scaling
            temperature_K: Operating temperature for Landauer cost
            device_mass_kg: Mass of computing device for Bremermann limit
        """
        self.analyses += 1

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"error": "syntax_error", "message": "Cannot parse source code"}

        # Detect complexity class
        complexity = self._detect_complexity(tree, source)

        # Count operations
        ops_estimate = self._estimate_operations(tree, source, complexity, input_size)

        # Physical limits at device parameters
        device_energy = device_mass_kg * _C ** 2
        bremermann_limit = device_mass_kg * _C ** 2 / (math.pi * _H_BAR)
        ml_limit = 2 * device_energy / (math.pi * _H_BAR)
        landauer_per_bit = _K_B * temperature_K * math.log(2)

        # Time required at Margolus-Levitin limit
        ml_time = ops_estimate["estimated_ops"] / ml_limit if ml_limit > 0 else float('inf')

        # Memory analysis → Bekenstein fraction
        memory = self._estimate_memory_bits(tree, source, input_size)
        bekenstein_radius = 0.05  # ~5cm radius device
        bekenstein_limit = 2 * math.pi * bekenstein_radius * device_energy / (_H_BAR * _C * math.log(2))
        bekenstein_fraction = memory["total_bits"] / bekenstein_limit if bekenstein_limit > 0 else 0

        # Landauer energy cost for all bit erasures
        erasure_bits = self._estimate_erasures(tree, source, input_size)
        landauer_total = erasure_bits * landauer_per_bit

        # Catastrophe detection: does ops grow faster than any physical limit?
        catastrophe = self._detect_catastrophe(complexity, input_size)

        # Efficiency score: how close to theoretical optimum?
        # Lower complexity = higher efficiency (O(n log n) sort is good for sorting)
        efficiency = self._compute_efficiency_score(complexity, ops_estimate, memory)

        return {
            "complexity_class": complexity["class"],
            "complexity_details": complexity,
            "operations": ops_estimate,
            "physical_limits": {
                "device_mass_kg": device_mass_kg,
                "device_energy_J": device_energy,
                "bremermann_limit_ops_per_sec": bremermann_limit,
                "margolus_levitin_limit_ops_per_sec": ml_limit,
                "landauer_per_bit_J": landauer_per_bit,
                "time_at_ml_limit_s": ml_time,
                "bekenstein_limit_bits": bekenstein_limit,
            },
            "memory": memory,
            "bekenstein_fraction": bekenstein_fraction,
            "landauer_erasure_energy_J": landauer_total,
            "erasure_bits": erasure_bits,
            "catastrophe": catastrophe,
            "efficiency_score": efficiency,
        }

    def _detect_complexity(self, tree: ast.AST, source: str) -> Dict[str, Any]:
        """Detect algorithmic complexity class from code structure."""
        max_depth = 0
        loop_count = 0
        recursive = False
        sort_calls = 0
        binary_search = False

        func_names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_names.add(node.name)

        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0
                self.max_depth = 0
                self.loop_count = 0
                self.recursive = False
                self.sort_calls = 0
                self.binary_patterns = 0

            def visit_For(self, node):
                self.depth += 1
                self.loop_count += 1
                self.max_depth = max(self.max_depth, self.depth)
                self.generic_visit(node)
                self.depth -= 1

            def visit_While(self, node):
                self.depth += 1
                self.loop_count += 1
                self.max_depth = max(self.max_depth, self.depth)
                # Check for binary search pattern (n //= 2, n >>= 1, etc.)
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign):
                        if isinstance(child.op, (ast.FloorDiv, ast.RShift)):
                            self.binary_patterns += 1
                self.generic_visit(node)
                self.depth -= 1

            def visit_Call(self, node):
                # Detect recursive calls
                if isinstance(node.func, ast.Name) and node.func.id in func_names:
                    self.recursive = True
                # Detect sort calls
                if isinstance(node.func, ast.Attribute) and node.func.attr in ('sort', 'sorted'):
                    self.sort_calls += 1
                if isinstance(node.func, ast.Name) and node.func.id == 'sorted':
                    self.sort_calls += 1
                self.generic_visit(node)

        visitor = ComplexityVisitor()
        visitor.visit(tree)

        # Determine complexity class from structural analysis
        if visitor.loop_count == 0 and not visitor.recursive:
            cls = "O(1)"
        elif visitor.binary_patterns > 0 and visitor.max_depth == 1:
            cls = "O(log n)"
        elif visitor.max_depth == 1 and not visitor.recursive:
            if visitor.sort_calls > 0:
                cls = "O(n log n)"
            else:
                cls = "O(n)"
        elif visitor.max_depth == 2:
            cls = "O(n²)"
        elif visitor.max_depth == 3:
            cls = "O(n³)"
        elif visitor.max_depth > 3:
            cls = "O(n³)"  # Cap at cubic for nested loops
        elif visitor.recursive:
            # Check for divide-and-conquer pattern
            if visitor.binary_patterns > 0:
                cls = "O(n log n)"
            else:
                cls = "O(2^n)"  # Assume exponential recursion as worst case
        else:
            cls = "O(n)"

        return {
            "class": cls,
            "max_loop_depth": visitor.max_depth,
            "loop_count": visitor.loop_count,
            "is_recursive": visitor.recursive,
            "sort_calls": visitor.sort_calls,
            "binary_patterns": visitor.binary_patterns,
            "rank": COMPLEXITY_CLASSES[cls]["rank"],
        }

    def _estimate_operations(self, tree: ast.AST, source: str,
                               complexity: Dict, input_size: int) -> Dict[str, Any]:
        """Estimate total operations for given input size."""
        cls = complexity["class"]
        growth = COMPLEXITY_CLASSES[cls]["growth"]
        base_ops = growth(input_size)

        # Count AST nodes as proxy for constant factor
        node_count = sum(1 for _ in ast.walk(tree))
        constant_factor = max(1, node_count // 10)

        estimated_ops = base_ops * constant_factor
        log10_ops = math.log10(estimated_ops) if estimated_ops > 0 else 0

        return {
            "complexity_class": cls,
            "input_size": input_size,
            "growth_at_n": base_ops,
            "constant_factor": constant_factor,
            "estimated_ops": estimated_ops,
            "log10_ops": log10_ops,
        }

    def _estimate_memory_bits(self, tree: ast.AST, source: str,
                                input_size: int) -> Dict[str, Any]:
        """Estimate memory usage in bits."""
        # Count variable assignments
        assignments = 0
        container_allocs = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                assignments += 1
            if isinstance(node, (ast.List, ast.Dict, ast.Set)):
                container_allocs += 1

        # Rough memory model: each variable ~64 bits, containers scale with input
        var_bits = assignments * 64
        container_bits = container_allocs * input_size * 64
        total_bits = var_bits + container_bits

        return {
            "variable_count": assignments,
            "container_allocs": container_allocs,
            "var_bits": var_bits,
            "container_bits": container_bits,
            "total_bits": total_bits,
            "total_bytes": total_bits // 8,
            "total_MB": total_bits / (8 * 1024 * 1024),
        }

    def _estimate_erasures(self, tree: ast.AST, source: str,
                             input_size: int) -> int:
        """
        Estimate number of irreversible bit erasures.

        Key insight from Landauer: only irreversible operations cost energy.
        Variable reassignment, overwriting, and garbage collection all erase bits.
        """
        reassignments = 0
        overwrites = 0

        # Track which names are assigned multiple times
        assign_targets = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assign_targets[target.id] = assign_targets.get(target.id, 0) + 1
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    assign_targets[node.target.id] = assign_targets.get(node.target.id, 0) + 1

        # Multiple assignments to same name = erasure of previous value
        for name, count in assign_targets.items():
            if count > 1:
                reassignments += (count - 1) * 64  # Each overwrite erases ~64 bits

        # Container mutations (append, pop, update, etc.)
        mutation_pattern = re.compile(r'\.(append|pop|insert|remove|update|clear|extend)\(')
        mutations = len(mutation_pattern.findall(source))
        overwrites = mutations * 64  # Each mutation erases ~64 bits

        return reassignments + overwrites

    def _detect_catastrophe(self, complexity: Dict,
                              input_size: int) -> Dict[str, Any]:
        """
        Detect ultraviolet catastrophe: does the algorithm's resource
        consumption grow without bound in a way that's physically impossible?

        Analogy to Rayleigh-Jeans: classical physics predicted infinite
        blackbody energy at high frequencies. Similarly, O(2^n) and O(n!)
        algorithms predict resource usage that exceeds the mass-energy of
        the observable universe for moderate n.
        """
        cls = complexity["class"]
        growth = COMPLEXITY_CLASSES[cls]["growth"]
        ops_at_n = growth(input_size)

        # Observable universe limits
        universe_mass_kg = 1.5e53  # ~10^53 kg
        universe_bremermann = universe_mass_kg * _C ** 2 / (math.pi * _H_BAR)
        universe_age_s = 4.35e17  # ~13.8 billion years in seconds
        universe_total_ops = universe_bremermann * universe_age_s

        is_catastrophe = ops_at_n > universe_total_ops
        log_ratio = math.log10(ops_at_n / universe_total_ops) if ops_at_n > 0 and universe_total_ops > 0 else 0

        # Find critical input size where catastrophe begins
        critical_n = None
        if COMPLEXITY_CLASSES[cls]["rank"] >= 7:  # Exponential or worse
            # Solve: growth(n) = universe_total_ops
            # For O(2^n): n = log2(universe_total_ops) ≈ 397
            log_universe = math.log2(universe_total_ops)
            if cls == "O(2^n)":
                critical_n = int(log_universe)
            elif cls == "O(n!)":
                # Stirling: n! ≈ (n/e)^n → n ≈ log(total) / log(n/e)
                critical_n = int(log_universe / math.log2(max(log_universe / math.e, 2)))

        return {
            "has_catastrophe": is_catastrophe,
            "complexity_class": cls,
            "ops_at_input_size": ops_at_n,
            "universe_total_ops": universe_total_ops,
            "log10_ratio": log_ratio,
            "critical_input_size": critical_n,
            "analogy": "Rayleigh-Jeans ultraviolet catastrophe" if is_catastrophe else "within physical bounds",
        }

    def _compute_efficiency_score(self, complexity: Dict,
                                     ops: Dict, memory: Dict) -> Dict[str, float]:
        """
        Compute computronium efficiency score (0-1).

        1.0 = optimal algorithm (O(1) or information-theoretic lower bound)
        0.0 = exceeds physical limits (ultraviolet catastrophe)
        """
        rank = complexity["rank"]
        max_rank = 8  # O(n!)

        # Complexity score: lower rank = higher efficiency
        complexity_score = 1.0 - (rank / max_rank)

        # Memory score: penalize high memory usage (> 1 GB)
        mem_gb = memory["total_bits"] / (8 * 1024 ** 3)
        memory_score = max(0, 1.0 - mem_gb)

        # Combined with PHI weighting
        phi_weight = PHI / (1 + PHI)  # ≈ 0.618
        combined = complexity_score * phi_weight + memory_score * (1 - phi_weight)

        return {
            "complexity_score": round(complexity_score, 6),
            "memory_score": round(memory_score, 6),
            "phi_weight": round(phi_weight, 6),
            "combined_score": round(combined, 6),
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  RAYLEIGH-INSPIRED CODE RESOLUTION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_code_resolution(self, source: str) -> Dict[str, Any]:
        """
        Analyze code through the lens of Rayleigh resolution limits.

        Just as the Rayleigh criterion defines the minimum angular separation
        for distinguishing two point sources, we define analogous limits for
        distinguishing code elements:

        - Variable name resolution: are names distinct enough to avoid confusion?
        - Function granularity: is each function a resolvable "point source"?
        - Abstraction diffraction: does excessive layering blur boundaries?
        - Information density: bits of logic per line of code
        """
        self.analyses += 1

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"error": "syntax_error"}

        # Name resolution analysis (analogous to Rayleigh angular resolution)
        names = self._analyze_name_resolution(tree)

        # Function granularity (analogous to Abbe diffraction limit)
        granularity = self._analyze_function_granularity(tree)

        # Information density (analogous to spectral energy density)
        density = self._analyze_information_density(source, tree)

        # Scattering analysis (analogous to Rayleigh scattering)
        scattering = self._analyze_code_scattering(tree, source)

        # Overall resolution score
        resolution_score = self._compute_resolution_score(names, granularity, density, scattering)

        return {
            "name_resolution": names,
            "function_granularity": granularity,
            "information_density": density,
            "code_scattering": scattering,
            "resolution_score": resolution_score,
        }

    def _analyze_name_resolution(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Name resolution: minimum "angular separation" between variable names.

        Analogous to θ = 1.22 λ/D:
        - Shorter names = shorter "wavelength" = harder to resolve
        - More names in scope = larger "aperture" = better resolution... up to a point

        Levenshtein distance between name pairs determines "angular separation."
        Names too close together are below the diffraction limit.
        """
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
                for arg in node.args.args:
                    names.add(arg.arg)

        name_list = sorted(names)
        min_distance = float('inf')
        min_pair = ("", "")
        confusable_pairs = []

        for i in range(len(name_list)):
            for j in range(i + 1, min(i + 20, len(name_list))):  # Check nearby names
                d = self._levenshtein(name_list[i], name_list[j])
                if d < min_distance:
                    min_distance = d
                    min_pair = (name_list[i], name_list[j])
                if d <= 1 and name_list[i] != name_list[j]:
                    confusable_pairs.append((name_list[i], name_list[j], d))

        # Average name length (analogous to wavelength)
        avg_len = sum(len(n) for n in name_list) / len(name_list) if name_list else 0

        # Resolution criterion: min_distance / avg_len > 1.22 (Rayleigh-inspired)
        rayleigh_ratio = min_distance / avg_len if avg_len > 0 else float('inf')
        is_resolved = rayleigh_ratio > 1.22

        return {
            "total_names": len(name_list),
            "avg_name_length": round(avg_len, 2),
            "min_levenshtein_distance": min_distance if min_distance != float('inf') else 0,
            "closest_pair": min_pair,
            "confusable_pairs": confusable_pairs[:10],
            "rayleigh_ratio": round(rayleigh_ratio, 4),
            "is_name_resolved": is_resolved,
            "analogy": "θ = 1.22 λ/D — names are point sources, Levenshtein = angular separation",
        }

    def _analyze_function_granularity(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Function granularity: analogous to Abbe diffraction limit d = λ/(2NA).

        The minimum useful function size is bounded below:
        - Too small: overhead dominates (function call cost > logic)
        - Too large: can't distinguish responsibilities
        - Optimal: each function is a "resolvable feature"

        Abbe analogy: λ = average function size, NA = coupling (dependencies).
        d_min = avg_size / (2 × coupling) — minimum distinguishable unit.
        """
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_nodes = sum(1 for _ in ast.walk(node))
                n_args = len(node.args.args)
                n_calls = sum(1 for n in ast.walk(node) if isinstance(n, ast.Call))
                functions.append({
                    "name": node.name,
                    "body_nodes": body_nodes,
                    "n_args": n_args,
                    "n_calls": n_calls,
                    "lineno": node.lineno,
                })

        if not functions:
            return {"functions": 0, "analysis": "no functions found"}

        sizes = [f["body_nodes"] for f in functions]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)

        # Coupling: average outgoing calls per function (analogous to NA)
        avg_calls = sum(f["n_calls"] for f in functions) / len(functions)
        coupling = max(avg_calls, 1)

        # Abbe-inspired minimum: d_min = avg_size / (2 × coupling)
        abbe_min = avg_size / (2 * coupling)

        # Functions below Abbe minimum are "sub-diffraction" — too fine-grained
        sub_diffraction = [f for f in functions if f["body_nodes"] < abbe_min]

        # Functions above 10× average are "macro" — not enough resolution
        macro = [f for f in functions if f["body_nodes"] > avg_size * 10]

        return {
            "total_functions": len(functions),
            "avg_body_nodes": round(avg_size, 1),
            "min_body_nodes": min_size,
            "max_body_nodes": max_size,
            "avg_coupling": round(coupling, 2),
            "abbe_minimum_nodes": round(abbe_min, 1),
            "sub_diffraction_functions": [f["name"] for f in sub_diffraction],
            "macro_functions": [f["name"] for f in macro],
            "optimal_range": f"{int(abbe_min)}-{int(avg_size * 3)} nodes",
            "analogy": "d_min = λ/(2NA) — function size vs coupling determines minimum useful granularity",
        }

    def _analyze_information_density(self, source: str,
                                        tree: ast.AST) -> Dict[str, Any]:
        """
        Information density: bits of logic per line of code.

        Analogous to Rayleigh-Jeans spectral density B(ν,T) = 2ν²k_BT/c²:
        - Higher "frequency" (shorter variable names, compact syntax) → more information
        - Higher "temperature" (code complexity) → more energy per mode
        - But: ultraviolet catastrophe = unbounded density signals code smell

        Uses Shannon entropy of the source as a proxy for information content.
        """
        lines = source.strip().split('\n')
        n_lines = len(lines)
        n_chars = len(source)

        # Shannon entropy of character distribution
        char_freq = {}
        for c in source:
            char_freq[c] = char_freq.get(c, 0) + 1
        entropy = 0.0
        for count in char_freq.values():
            p = count / n_chars if n_chars > 0 else 0
            if p > 0:
                entropy -= p * math.log2(p)

        # Bits per line
        total_bits = entropy * n_chars  # Total Shannon information content
        bits_per_line = total_bits / n_lines if n_lines > 0 else 0

        # AST nodes per line (structural density)
        n_nodes = sum(1 for _ in ast.walk(tree))
        nodes_per_line = n_nodes / n_lines if n_lines > 0 else 0

        # Comment ratio (reduces useful information density)
        comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
        blank_lines = sum(1 for l in lines if not l.strip())
        code_lines = n_lines - comment_lines - blank_lines
        code_ratio = code_lines / n_lines if n_lines > 0 else 0

        # Rayleigh-Jeans analogy:
        # "frequency" = nodes_per_line (how tightly packed is the logic?)
        # "temperature" = entropy (how complex is the character distribution?)
        # B_code = 2 × nodes_per_line² × entropy (spectral density analogue)
        rj_density = 2 * nodes_per_line ** 2 * entropy

        return {
            "total_lines": n_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
            "code_ratio": round(code_ratio, 4),
            "shannon_entropy_bits": round(entropy, 4),
            "total_information_bits": round(total_bits, 1),
            "bits_per_line": round(bits_per_line, 2),
            "ast_nodes": n_nodes,
            "nodes_per_line": round(nodes_per_line, 2),
            "rj_spectral_density": round(rj_density, 4),
            "analogy": "B(ν,T) = 2ν²k_BT/c² — information density as spectral radiance",
        }

    def _analyze_code_scattering(self, tree: ast.AST,
                                    source: str) -> Dict[str, Any]:
        """
        Code scattering: analogous to Rayleigh scattering σ ∝ 1/λ⁴.

        Short, abstract elements ("short wavelength") scatter understanding
        more than long, concrete elements. Measures how much cognitive effort
        is "scattered" by abstraction layers.

        - Deep inheritance = high scattering (understanding diffuses)
        - Long import chains = absorption (like atmospheric opacity)
        - Metaclasses/decorators = high-frequency features (strong scattering)
        """
        imports = 0
        decorators = 0
        classes = 0
        metaclasses = 0
        inheritance_depth = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1
            if isinstance(node, ast.ClassDef):
                classes += 1
                bases = len(node.bases)
                if bases > inheritance_depth:
                    inheritance_depth = bases
                for kw in node.keywords:
                    if kw.arg == 'metaclass':
                        metaclasses += 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators += len(node.decorator_list)

        # Scattering cross-section analogy: σ ∝ (abstraction_level)⁴ / (concreteness)⁴
        # High abstraction (metaclasses, deep inheritance) = strong scattering
        abstraction_level = metaclasses * 4 + inheritance_depth * 2 + decorators
        concreteness = max(1, sum(1 for _ in ast.walk(tree)) - abstraction_level * 10)
        scattering_ratio = (abstraction_level ** 4) / (concreteness ** 4) if concreteness > 0 else 0

        # Optical depth: how much understanding is "absorbed" by import chains
        # Analogous to atmospheric optical depth τ = Nσd
        optical_depth = imports * scattering_ratio * 100

        # Transmission: fraction of meaning that reaches the reader
        transmission = math.exp(-optical_depth) if optical_depth < 700 else 0

        return {
            "imports": imports,
            "decorators": decorators,
            "classes": classes,
            "metaclasses": metaclasses,
            "max_inheritance_depth": inheritance_depth,
            "abstraction_level": abstraction_level,
            "scattering_cross_section": round(scattering_ratio, 8),
            "optical_depth": round(optical_depth, 4),
            "transmission": round(transmission, 4),
            "analogy": "σ ∝ 1/λ⁴ — abstract elements scatter comprehension like Rayleigh scattering",
        }

    def _compute_resolution_score(self, names: Dict, granularity: Dict,
                                     density: Dict, scattering: Dict) -> Dict[str, float]:
        """Combined resolution score using PHI-weighted components."""
        scores = {}

        # Name resolution (higher rayleigh_ratio = better)
        rr = names.get("rayleigh_ratio", 0)
        scores["name_resolution"] = min(1.0, rr / 2.44)  # 2× Rayleigh = excellent

        # Granularity (penalize extremes)
        total_fn = granularity.get("total_functions", 1)
        sub_diff = len(granularity.get("sub_diffraction_functions", []))
        macro = len(granularity.get("macro_functions", []))
        scores["granularity"] = max(0, 1.0 - (sub_diff + macro) / max(total_fn, 1))

        # Density (moderate density is best, too high = catastrophe)
        bpl = density.get("bits_per_line", 0)
        scores["density"] = max(0, 1.0 - abs(bpl - 50) / 100)  # Optimal ~50 bits/line

        # Scattering (lower = better, high = opaque code)
        scores["clarity"] = scattering.get("transmission", 0)

        # PHI-weighted combination
        weights = [PHI ** 2, PHI, 1.0, PHI_CONJUGATE := (math.sqrt(5) - 1) / 2]
        total_weight = sum(weights)
        values = [scores["name_resolution"], scores["granularity"],
                  scores["density"], scores["clarity"]]
        combined = sum(w * v for w, v in zip(weights, values)) / total_weight

        scores["combined"] = round(combined, 6)
        return scores

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return ComputroniumCodeAnalyzer._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def get_status(self) -> Dict[str, Any]:
        return {
            "subsystem": "ComputroniumCodeAnalyzer",
            "version": "1.0.0",
            "analyses_performed": self.analyses,
            "capabilities": [
                "computronium_budget",
                "code_resolution",
                "complexity_detection",
                "ultraviolet_catastrophe_detection",
                "bekenstein_memory_analysis",
                "landauer_erasure_cost",
                "rayleigh_name_resolution",
                "abbe_function_granularity",
                "rj_information_density",
                "rayleigh_scattering_clarity",
            ],
        }
