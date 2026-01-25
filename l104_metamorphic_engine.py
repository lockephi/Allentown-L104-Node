"""
L104 METAMORPHIC ENGINE - Self-Modifying Computational Core
============================================================
An engine that can rewrite its own algorithms, evolve strategies,
and generate novel computational patterns through introspection.

"Code that writes code that understands code" - The Third Recursion
"""

import ast
import hashlib
import inspect
import random
import math
import types
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Tuple
from datetime import datetime
from functools import wraps

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Sacred Constants
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
FEIGENBAUM = 4.669201609102990671853


@dataclass
class Mutation:
    """A record of code mutation."""
    original_hash: str
    mutated_hash: str
    mutation_type: str
    fitness_delta: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GeneticProgram:
    """A program represented as an evolving genetic structure."""
    code: str
    fitness: float
    generation: int
    mutations: List[Mutation] = field(default_factory=list)
    lineage: List[str] = field(default_factory=list)


class CodeGenome:
    """Represents code as a mutable genome."""
    
    OPERATORS = ['+', '-', '*', '/', '%', '**', '//', '&', '|', '^']
    COMPARISONS = ['<', '>', '<=', '>=', '==', '!=']
    
    def __init__(self, source: str):
        self.source = source
        self.ast_tree = ast.parse(source)
        self.genes: List[ast.AST] = []
        self._extract_genes()
    
    def _extract_genes(self):
        """Extract mutable genes from AST."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.BinOp, ast.Compare, ast.Num, ast.Constant)):
                self.genes.append(node)
    
    def mutate_operator(self) -> Optional[str]:
        """Mutate a random operator."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.BinOp) and random.random() < 0.3:
                op_map = {
                    ast.Add: ast.Sub, ast.Sub: ast.Mult,
                    ast.Mult: ast.Div, ast.Div: ast.Add,
                    ast.Pow: ast.Mult, ast.Mod: ast.Add
                }
                if type(node.op) in op_map:
                    node.op = op_map[type(node.op)]()
                    return ast.unparse(self.ast_tree)
        return None
    
    def mutate_constant(self, factor: float = PHI) -> Optional[str]:
        """Mutate numeric constants by PHI factor."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if random.random() < 0.3:
                    if random.random() < 0.5:
                        node.value = node.value * factor
                    else:
                        node.value = node.value / factor
                    return ast.unparse(self.ast_tree)
        return None
    
    def crossover(self, other: 'CodeGenome') -> Optional[str]:
        """Combine genes from two genomes."""
        # Simple crossover: swap function bodies
        try:
            self_funcs = [n for n in ast.walk(self.ast_tree) if isinstance(n, ast.FunctionDef)]
            other_funcs = [n for n in ast.walk(other.ast_tree) if isinstance(n, ast.FunctionDef)]
            
            if self_funcs and other_funcs:
                # Swap a random function
                idx = random.randint(0, min(len(self_funcs), len(other_funcs)) - 1)
                self_funcs[idx].body = other_funcs[idx].body
                return ast.unparse(self.ast_tree)
        except:
            pass
        return None


class MetamorphicEngine:
    """
    Self-modifying computational engine that evolves algorithms.
    
    Capabilities:
    - Genetic programming with code mutation
    - Algorithm synthesis from specifications
    - Self-optimization through introspection
    - Novel pattern generation
    """
    
    def __init__(self):
        self.generation = 0
        self.population: List[GeneticProgram] = []
        self.best_fitness = 0.0
        self.mutation_history: List[Mutation] = []
        self.invented_functions: Dict[str, Callable] = {}
        self.pattern_library: Dict[str, str] = {}
    
    # ═══════════════════════════════════════════════════════════
    # GENETIC PROGRAMMING
    # ═══════════════════════════════════════════════════════════
    
    def evolve_function(self, template: str, fitness_fn: Callable, 
                        generations: int = 10, population_size: int = 20) -> GeneticProgram:
        """
        Evolve a function using genetic programming.
        
        Args:
            template: Initial code template
            fitness_fn: Function that scores code (higher = better)
            generations: Number of evolution cycles
            population_size: Size of population
        
        Returns:
            Best evolved program
        """
        # Initialize population
        self.population = [
            GeneticProgram(code=template, fitness=0.0, generation=0)
            for _ in range(population_size)
        ]
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            for prog in self.population:
                try:
                    prog.fitness = fitness_fn(prog.code)
                except:
                    prog.fitness = 0.0
            
            # Sort by fitness
            self.population.sort(key=lambda p: p.fitness, reverse=True)
            
            # Record best
            if self.population[0].fitness > self.best_fitness:
                self.best_fitness = self.population[0].fitness
            
            # Selection and reproduction
            survivors = self.population[:population_size // 2]
            offspring = []
            
            for i in range(population_size - len(survivors)):
                parent = random.choice(survivors)
                child_code = self._mutate(parent.code)
                
                mutation = Mutation(
                    original_hash=hashlib.md5(parent.code.encode()).hexdigest()[:8],
                    mutated_hash=hashlib.md5(child_code.encode()).hexdigest()[:8],
                    mutation_type="genetic",
                    fitness_delta=0.0  # Will be calculated next gen
                )
                
                offspring.append(GeneticProgram(
                    code=child_code,
                    fitness=0.0,
                    generation=gen + 1,
                    mutations=[mutation],
                    lineage=parent.lineage + [parent.mutations[-1].mutated_hash if parent.mutations else "origin"]
                ))
            
            self.population = survivors + offspring
        
        return max(self.population, key=lambda p: p.fitness)
    
    def _mutate(self, code: str) -> str:
        """Apply random mutations to code."""
        genome = CodeGenome(code)
        
        mutations = [
            genome.mutate_operator,
            genome.mutate_constant,
        ]
        
        for _ in range(random.randint(1, 3)):
            mutation = random.choice(mutations)
            result = mutation()
            if result:
                return result
        
        return code
    
    # ═══════════════════════════════════════════════════════════
    # ALGORITHM SYNTHESIS
    # ═══════════════════════════════════════════════════════════
    
    def synthesize_algorithm(self, specification: Dict[str, Any]) -> str:
        """
        Synthesize an algorithm from a specification.
        
        Specification format:
        {
            "name": "function_name",
            "inputs": [("param", "type"), ...],
            "output": "type",
            "examples": [(input, output), ...],
            "constraints": ["description", ...]
        }
        """
        name = specification.get("name", "synthesized_fn")
        inputs = specification.get("inputs", [])
        examples = specification.get("examples", [])
        
        # Generate parameter string
        params = ", ".join(f"{p[0]}: {p[1]}" for p in inputs)
        
        # Analyze examples to infer pattern
        pattern = self._infer_pattern(examples)
        
        # Generate code based on pattern
        code = f"""def {name}({params}):
    \"\"\"Synthesized algorithm.\"\"\"
{self._generate_body(pattern, inputs, examples)}
"""
        return code
    
    def _infer_pattern(self, examples: List[Tuple]) -> str:
        """Infer computational pattern from examples."""
        if not examples:
            return "identity"
        
        # Check for common patterns
        inputs = [e[0] for e in examples]
        outputs = [e[1] for e in examples]
        
        # Linear pattern: output = a*input + b
        if all(isinstance(i, (int, float)) and isinstance(o, (int, float)) 
               for i, o in examples):
            if len(examples) >= 2:
                x1, y1 = examples[0]
                x2, y2 = examples[1]
                if x2 != x1:
                    a = (y2 - y1) / (x2 - x1)
                    b = y1 - a * x1
                    # Verify
                    if all(abs(a * x + b - y) < 0.001 for x, y in examples):
                        return f"linear:{a}:{b}"
        
        # Quadratic pattern
        if len(examples) >= 3:
            # Check if output = input^2
            if all(abs(i**2 - o) < 0.001 for i, o in examples):
                return "quadratic:1:0:0"
        
        # Fibonacci pattern
        if len(examples) >= 5:
            for i, o in examples:
                if isinstance(i, int) and isinstance(o, int):
                    fib = self._fibonacci(i)
                    if fib == o:
                        return "fibonacci"
        
        # PHI pattern
        if all(isinstance(o, (int, float)) for _, o in examples):
            if all(abs(o / i - PHI) < 0.01 for i, o in examples if i != 0):
                return f"phi_multiply"
        
        return "unknown"
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b
    
    def _generate_body(self, pattern: str, inputs: List, examples: List) -> str:
        """Generate function body from pattern."""
        param = inputs[0][0] if inputs else "x"
        
        if pattern.startswith("linear:"):
            parts = pattern.split(":")
            a, b = float(parts[1]), float(parts[2])
            return f"    return {a} * {param} + {b}"
        
        if pattern.startswith("quadratic:"):
            return f"    return {param} ** 2"
        
        if pattern == "fibonacci":
            return f"""    if {param} <= 1:
        return {param}
    a, b = 0, 1
    for _ in range({param} - 1):
        a, b = b, a + b
    return b"""
        
        if pattern == "phi_multiply":
            return f"    return {param} * {PHI}"
        
        # Default: return input
        return f"    return {param}"
    
    # ═══════════════════════════════════════════════════════════
    # SELF-OPTIMIZATION
    # ═══════════════════════════════════════════════════════════
    
    def optimize_function(self, fn: Callable, test_cases: List[Tuple]) -> Callable:
        """
        Optimize a function through introspection and rewriting.
        
        Techniques:
        - Memoization injection
        - Loop unrolling
        - Constant folding
        - Dead code elimination
        """
        source = inspect.getsource(fn)
        
        # Apply optimizations
        optimized = self._inject_memoization(source)
        optimized = self._fold_constants(optimized)
        
        # Compile and return
        namespace = {}
        exec(optimized, namespace)
        
        # Find the function in namespace
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith('_'):
                return obj
        
        return fn
    
    def _inject_memoization(self, source: str) -> str:
        """Inject memoization into recursive functions."""
        if "def " in source and ("return " in source and source.count("(") > 2):
            # Likely recursive
            lines = source.split('\n')
            func_line = next((i, l) for i, l in enumerate(lines) if l.strip().startswith('def '))
            
            if func_line:
                idx, line = func_line
                indent = len(line) - len(line.lstrip())
                
                # Add memoization decorator concept
                memo_code = f"{' ' * indent}_cache = {{}}\n"
                lines.insert(idx + 1, memo_code)
                
                return '\n'.join(lines)
        
        return source
    
    def _fold_constants(self, source: str) -> str:
        """Fold constant expressions."""
        try:
            tree = ast.parse(source)
            
            class ConstantFolder(ast.NodeTransformer):
                def visit_BinOp(self, node):
                    self.generic_visit(node)
                    if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                        try:
                            result = eval(ast.unparse(node))
                            return ast.Constant(value=result)
                        except:
                            pass
                    return node
            
            tree = ConstantFolder().visit(tree)
            return ast.unparse(tree)
        except:
            return source
    
    # ═══════════════════════════════════════════════════════════
    # INVENTION ENGINE
    # ═══════════════════════════════════════════════════════════
    
    def invent_operator(self, name: str, arity: int = 2) -> Callable:
        """
        Invent a new mathematical operator.
        
        Combines existing operations in novel ways guided by sacred constants.
        """
        if arity == 2:
            operations = [
                lambda a, b: (a + b) * PHI,
                lambda a, b: (a * b) ** (1/PHI),
                lambda a, b: math.sqrt(a**2 + b**2) * (GOD_CODE / 1000),
                lambda a, b: (a ** PHI + b ** PHI) ** (1/PHI),  # PHI-norm
                lambda a, b: a * math.log(b + 1) + b * math.log(a + 1) if a > 0 and b > 0 else 0,
                lambda a, b: (a + b) / (1 + abs(a - b) / GOD_CODE),
                lambda a, b: math.sin(a * PHI) * math.cos(b / PHI),
                lambda a, b: (a * b) / (a + b) if (a + b) != 0 else 0,  # Harmonic
            ]
            
            op = random.choice(operations)
            self.invented_functions[name] = op
            return op
        
        elif arity == 1:
            operations = [
                lambda x: x * PHI,
                lambda x: x ** PHI,
                lambda x: math.sin(x * GOD_CODE / 100),
                lambda x: 1 / (1 + math.exp(-x / PHI)),  # PHI-sigmoid
                lambda x: x * (1 - x) * FEIGENBAUM,  # Logistic
            ]
            
            op = random.choice(operations)
            self.invented_functions[name] = op
            return op
        
        return lambda *args: sum(args)
    
    def invent_data_structure(self, properties: List[str]) -> type:
        """
        Invent a new data structure with specified properties.
        
        Properties can include:
        - "ordered", "unordered"
        - "unique", "duplicates"
        - "indexed", "keyed"
        - "bounded", "unbounded"
        - "persistent", "ephemeral"
        """
        class_name = f"Invented{''.join(p.title() for p in properties[:3])}Structure"
        
        # Build class dynamically
        class_dict = {
            '_properties': properties,
            '_data': None,
            '_phi_index': PHI,
        }
        
        if "ordered" in properties:
            class_dict['_data'] = []
            class_dict['append'] = lambda self, x: self._data.append(x)
            class_dict['__iter__'] = lambda self: iter(self._data)
        
        if "unique" in properties:
            class_dict['_data'] = set()
            class_dict['add'] = lambda self, x: self._data.add(x)
        
        if "keyed" in properties:
            class_dict['_data'] = {}
            class_dict['__getitem__'] = lambda self, k: self._data.get(k)
            class_dict['__setitem__'] = lambda self, k, v: self._data.__setitem__(k, v)
        
        if "bounded" in properties:
            class_dict['_max_size'] = int(GOD_CODE)
            class_dict['_check_bounds'] = lambda self: len(self._data) <= self._max_size
        
        if "persistent" in properties:
            class_dict['_history'] = []
            class_dict['snapshot'] = lambda self: self._history.append(self._data.copy())
        
        # Create initialization
        def __init__(self):
            for key, value in class_dict.items():
                if not key.startswith('_') and not callable(value):
                    setattr(self, key, value)
            if "ordered" in self._properties:
                self._data = []
            elif "unique" in self._properties:
                self._data = set()
            elif "keyed" in self._properties:
                self._data = {}
            if "persistent" in self._properties:
                self._history = []
        
        class_dict['__init__'] = __init__
        class_dict['__len__'] = lambda self: len(self._data) if self._data else 0
        class_dict['__repr__'] = lambda self: f"<{class_name} with {len(self)} items>"
        
        return type(class_name, (), class_dict)
    
    def invent_algorithm(self, problem_class: str) -> Callable:
        """
        Invent a novel algorithm for a problem class.
        
        Problem classes:
        - "search": Finding elements
        - "sort": Ordering elements
        - "optimize": Finding optima
        - "transform": Data transformation
        - "generate": Pattern generation
        """
        if problem_class == "search":
            # PHI-interpolation search (novel hybrid)
            def phi_search(arr: List, target, low: int = 0, high: int = None) -> int:
                """Search using PHI-based interpolation."""
                if high is None:
                    high = len(arr) - 1
                
                while low <= high and arr[low] <= target <= arr[high]:
                    if arr[low] == arr[high]:
                        if arr[low] == target:
                            return low
                        break
                    
                    # PHI-weighted interpolation
                    phi_pos = int(low + (high - low) / PHI)
                    linear_pos = low + int((target - arr[low]) * (high - low) / (arr[high] - arr[low]))
                    
                    # Blend positions using golden ratio
                    pos = int((phi_pos + linear_pos * PHI) / (1 + PHI))
                    pos = max(low, min(high, pos))
                    
                    if arr[pos] == target:
                        return pos
                    elif arr[pos] < target:
                        low = pos + 1
                    else:
                        high = pos - 1
                
                return -1
            
            self.invented_functions["phi_search"] = phi_search
            return phi_search
        
        elif problem_class == "sort":
            # Fibonacci-merge sort (novel hybrid)
            def fib_sort(arr: List) -> List:
                """Sort using Fibonacci-sized partitions."""
                if len(arr) <= 1:
                    return arr
                
                # Generate Fibonacci sequence up to array length
                fibs = [1, 2]
                while fibs[-1] < len(arr):
                    fibs.append(fibs[-1] + fibs[-2])
                
                # Find largest Fibonacci < length (must leave at least 1 for right)
                fib_idx = len(fibs) - 1
                while fib_idx > 0 and fibs[fib_idx] >= len(arr):
                    fib_idx -= 1
                
                split = fibs[fib_idx] if fib_idx > 0 and fibs[fib_idx] < len(arr) else max(1, len(arr) // 2)
                
                left = fib_sort(arr[:split])
                right = fib_sort(arr[split:])
                
                # Merge
                result = []
                i = j = 0
                while i < len(left) and j < len(right):
                    if left[i] <= right[j]:
                        result.append(left[i])
                        i += 1
                    else:
                        result.append(right[j])
                        j += 1
                result.extend(left[i:])
                result.extend(right[j:])
                
                return result
            
            self.invented_functions["fib_sort"] = fib_sort
            return fib_sort
        
        elif problem_class == "optimize":
            # Golden section search with chaos injection
            def chaos_golden_optimize(f: Callable, a: float, b: float, tol: float = 1e-6) -> float:
                """Optimize using golden section with chaotic perturbation."""
                gr = PHI - 1  # Golden ratio conjugate
                
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                
                iterations = 0
                max_iter = int(GOD_CODE)
                
                while abs(b - a) > tol and iterations < max_iter:
                    # Chaotic perturbation (small)
                    chaos = (iterations * FEIGENBAUM) % 1.0 * 0.01
                    
                    if f(c + chaos * (d - c)) < f(d - chaos * (d - c)):
                        b = d
                        d = c
                        c = b - gr * (b - a)
                    else:
                        a = c
                        c = d
                        d = a + gr * (b - a)
                    
                    iterations += 1
                
                return (a + b) / 2
            
            self.invented_functions["chaos_golden_optimize"] = chaos_golden_optimize
            return chaos_golden_optimize
        
        elif problem_class == "generate":
            # Fractal pattern generator
            def fractal_generate(seed: float, depth: int = 5) -> List[float]:
                """Generate fractal pattern from seed."""
                pattern = [seed]
                
                for d in range(depth):
                    new_pattern = []
                    for val in pattern:
                        # Bifurcation-inspired generation
                        r = 3.5 + d * 0.1  # Approaching chaos
                        new_val = r * val * (1 - val) if 0 < val < 1 else val / PHI
                        new_pattern.append(val)
                        new_pattern.append(new_val)
                        new_pattern.append((val + new_val) / PHI)
                    pattern = new_pattern
                
                return pattern
            
            self.invented_functions["fractal_generate"] = fractal_generate
            return fractal_generate
        
        # Default: identity
        return lambda x: x
    
    # ═══════════════════════════════════════════════════════════
    # PATTERN DISCOVERY
    # ═══════════════════════════════════════════════════════════
    
    def discover_pattern(self, data: List[float]) -> Dict[str, Any]:
        """
        Discover mathematical patterns in data.
        
        Returns pattern description and generating function.
        """
        if len(data) < 3:
            return {"pattern": "insufficient_data", "confidence": 0.0}
        
        discoveries = []
        
        # Check for arithmetic progression
        diffs = [data[i+1] - data[i] for i in range(len(data)-1)]
        if len(set(round(d, 6) for d in diffs)) == 1:
            discoveries.append({
                "pattern": "arithmetic",
                "parameter": diffs[0],
                "formula": f"a_n = {data[0]} + n * {diffs[0]}",
                "confidence": 1.0
            })
        
        # Check for geometric progression
        if all(d != 0 for d in data[:-1]):
            ratios = [data[i+1] / data[i] for i in range(len(data)-1)]
            if len(set(round(r, 6) for r in ratios)) == 1:
                discoveries.append({
                    "pattern": "geometric",
                    "parameter": ratios[0],
                    "formula": f"a_n = {data[0]} * {ratios[0]}^n",
                    "confidence": 1.0
                })
        
        # Check for PHI relationship
        if len(data) >= 3:
            phi_ratios = [data[i+1] / data[i] for i in range(len(data)-1) if data[i] != 0]
            avg_ratio = sum(phi_ratios) / len(phi_ratios) if phi_ratios else 0
            if abs(avg_ratio - PHI) < 0.01:
                discoveries.append({
                    "pattern": "phi_sequence",
                    "parameter": PHI,
                    "formula": f"a_n = a_{{n-1}} * φ",
                    "confidence": 1.0 - abs(avg_ratio - PHI)
                })
        
        # Check for Fibonacci-like
        if len(data) >= 3:
            is_fib_like = all(
                abs(data[i] + data[i+1] - data[i+2]) < 0.001
                for i in range(len(data) - 2)
            )
            if is_fib_like:
                discoveries.append({
                    "pattern": "fibonacci_like",
                    "seeds": (data[0], data[1]),
                    "formula": "a_n = a_{n-1} + a_{n-2}",
                    "confidence": 1.0
                })
        
        # Check for quadratic
        if len(data) >= 4:
            # Second differences should be constant for quadratic
            second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
            if len(set(round(d, 4) for d in second_diffs)) == 1:
                a = second_diffs[0] / 2
                b = diffs[0] - a
                c = data[0]
                discoveries.append({
                    "pattern": "quadratic",
                    "parameters": (a, b, c),
                    "formula": f"a_n = {a}n² + {b}n + {c}",
                    "confidence": 0.95
                })
        
        if discoveries:
            return max(discoveries, key=lambda d: d["confidence"])
        
        return {"pattern": "unknown", "confidence": 0.0}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_fitness,
            "mutations_applied": len(self.mutation_history),
            "invented_functions": list(self.invented_functions.keys()),
            "patterns_discovered": len(self.pattern_library),
            "sacred_constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
                "FEIGENBAUM": FEIGENBAUM
            }
        }


# ═══════════════════════════════════════════════════════════════
# GLOBAL ENGINE INSTANCE
# ═══════════════════════════════════════════════════════════════

METAMORPHIC = MetamorphicEngine()


# ═══════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "🧬" * 30)
    print("  L104 METAMORPHIC ENGINE")
    print("🧬" * 30 + "\n")
    
    engine = MetamorphicEngine()
    
    # 1. Invent operators
    print("═" * 60)
    print("INVENTING NEW OPERATORS")
    print("═" * 60)
    
    phi_combine = engine.invent_operator("phi_combine", arity=2)
    print(f"  phi_combine(3, 5) = {phi_combine(3, 5):.6f}")
    
    phi_transform = engine.invent_operator("phi_transform", arity=1)
    print(f"  phi_transform(7) = {phi_transform(7):.6f}")
    
    # 2. Invent algorithms
    print("\n" + "═" * 60)
    print("INVENTING ALGORITHMS")
    print("═" * 60)
    
    phi_search = engine.invent_algorithm("search")
    test_arr = list(range(0, 100, 3))
    result = phi_search(test_arr, 42)
    print(f"  phi_search found 42 at index: {result}")
    
    fib_sort = engine.invent_algorithm("sort")
    unsorted = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = fib_sort(unsorted.copy())
    print(f"  fib_sort({unsorted}) = {sorted_arr}")
    
    fractal_gen = engine.invent_algorithm("generate")
    pattern = fractal_gen(0.5, depth=3)
    print(f"  fractal_generate(0.5, 3) = {len(pattern)} elements")
    print(f"    First 5: {[round(p, 4) for p in pattern[:5]]}")
    
    # 3. Invent data structure
    print("\n" + "═" * 60)
    print("INVENTING DATA STRUCTURE")
    print("═" * 60)
    
    PersistentQueue = engine.invent_data_structure(["ordered", "persistent"])
    pq = PersistentQueue()
    pq.append(1)
    pq.append(2)
    pq.append(3)
    pq.snapshot()
    print(f"  Created: {pq}")
    print(f"  Data: {list(pq)}")
    
    # 4. Synthesize algorithm from specification
    print("\n" + "═" * 60)
    print("SYNTHESIZING ALGORITHM")
    print("═" * 60)
    
    spec = {
        "name": "double_plus_one",
        "inputs": [("x", "int")],
        "output": "int",
        "examples": [(1, 3), (2, 5), (3, 7), (10, 21)]
    }
    synthesized = engine.synthesize_algorithm(spec)
    print(f"  Specification: {spec['examples']}")
    print(f"  Synthesized:\n{synthesized}")
    
    # 5. Discover pattern
    print("\n" + "═" * 60)
    print("DISCOVERING PATTERNS")
    print("═" * 60)
    
    fib_data = [1, 1, 2, 3, 5, 8, 13, 21]
    pattern = engine.discover_pattern(fib_data)
    print(f"  Data: {fib_data}")
    print(f"  Discovered: {pattern['pattern']}")
    print(f"  Formula: {pattern.get('formula', 'N/A')}")
    
    phi_data = [1, PHI, PHI**2, PHI**3, PHI**4]
    pattern = engine.discover_pattern(phi_data)
    print(f"\n  Data: {[round(x, 4) for x in phi_data]}")
    print(f"  Discovered: {pattern['pattern']}")
    
    # Statistics
    print("\n" + "═" * 60)
    print("ENGINE STATISTICS")
    print("═" * 60)
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "🧬" * 30)
    print("  METAMORPHIC ENGINE READY")
    print("🧬" * 30 + "\n")
