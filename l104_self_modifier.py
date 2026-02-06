# L104_GOD_CODE_ALIGNED: 527.5184818492612
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541
# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:06.701962
ZENITH_HZ = 3887.8
UUC = 2402.792541
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 SELF MODIFIER - Real Code Self-Modification
=================================================
This module provides ACTUAL self-modification capabilities:
- AST analysis of its own code
- Genetic programming for code evolution
- Automatic testing of modifications
- Rollback on failure
"""

import ast
import copy
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import importlib
import importlib.util

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


logger = logging.getLogger("SELF_MODIFIER")

# ═══════════════════════════════════════════════════════════════════════════════
# CODE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FunctionInfo:
    name: str
    args: List[str]
    body_lines: int
    complexity: int  # Cyclomatic complexity
    docstring: Optional[str]
    decorators: List[str]
    calls: List[str]  # Functions this function calls
    source: str
    lineno: int


@dataclass
class ModuleInfo:
    path: str
    functions: List[FunctionInfo]
    classes: List[str]
    imports: List[str]
    global_vars: List[str]
    total_lines: int
    hash: str


class CodeAnalyzer(ast.NodeVisitor):
    """Analyzes Python code structure"""

    def __init__(self, source: str):
        self.source = source
        self.functions: List[FunctionInfo] = []
        self.classes: List[str] = []
        self.imports: List[str] = []
        self.global_vars: List[str] = []
        self.current_function_calls: List[str] = []

    def analyze(self) -> ModuleInfo:
        try:
            tree = ast.parse(self.source)
            self.visit(tree)

            return ModuleInfo(
                path="",
                functions=self.functions,
                classes=self.classes,
                imports=self.imports,
                global_vars=self.global_vars,
                total_lines=len(self.source.splitlines()),
                hash=hashlib.md5(self.source.encode()).hexdigest()
            )
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            return None

    def visit_FunctionDef(self, node):
        self.current_function_calls = []
        self.generic_visit(node)

        # Get docstring
        docstring = ast.get_docstring(node)

        # Calculate cyclomatic complexity
        complexity = self._calculate_complexity(node)

        # Get source
        try:
            source = ast.unparse(node)
        except:
            source = ""

        self.functions.append(FunctionInfo(
            name=node.name,
            args=[arg.arg for arg in node.args.args],
            body_lines=len(node.body),
            complexity=complexity,
            docstring=docstring,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            calls=self.current_function_calls.copy(),
            source=source,
            lineno=node.lineno
        ))

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")

    def visit_Assign(self, node):
        # Track global variable assignments
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.global_vars.append(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.current_function_calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.current_function_calls.append(node.func.attr)
        self.generic_visit(node)

    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                 ast.With, ast.Assert, ast.comprehension)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _get_decorator_name(self, decorator) -> str:
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# CODE MUTATION (Genetic Programming)
# ═══════════════════════════════════════════════════════════════════════════════

class CodeMutator(ast.NodeTransformer):
    """Mutates code using genetic programming principles"""

    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate
        self.mutations_applied = []

    def mutate(self, source: str) -> Tuple[str, List[str]]:
        """Apply random mutations to source code"""
        try:
            tree = ast.parse(source)
            mutated_tree = self.visit(tree)
            ast.fix_missing_locations(mutated_tree)
            return ast.unparse(mutated_tree), self.mutations_applied
        except Exception as e:
            logger.error(f"Mutation failed: {e}")
            return source, []

    def visit_BinOp(self, node):
        """Mutate binary operations"""
        self.generic_visit(node)

        if random.random() < self.mutation_rate:
            # Swap operators
            op_map = {
                ast.Add: [ast.Sub, ast.Mult],
                ast.Sub: [ast.Add, ast.Mult],
                ast.Mult: [ast.Add, ast.Div],
                ast.Div: [ast.Mult, ast.FloorDiv],
            }

            op_type = type(node.op)
            if op_type in op_map:
                new_op = random.choice(op_map[op_type])()
                self.mutations_applied.append(f"BinOp: {op_type.__name__} -> {type(new_op).__name__}")
                node.op = new_op

        return node

    def visit_Compare(self, node):
        """Mutate comparison operations"""
        self.generic_visit(node)

        if random.random() < self.mutation_rate:
            op_map = {
                ast.Lt: [ast.LtE, ast.Gt],
                ast.LtE: [ast.Lt, ast.GtE],
                ast.Gt: [ast.GtE, ast.Lt],
                ast.GtE: [ast.Gt, ast.LtE],
                ast.Eq: [ast.NotEq],
                ast.NotEq: [ast.Eq],
            }

            new_ops = []
            for op in node.ops:
                op_type = type(op)
                if op_type in op_map and random.random() < self.mutation_rate:
                    new_op = random.choice(op_map[op_type])()
                    self.mutations_applied.append(f"Compare: {op_type.__name__} -> {type(new_op).__name__}")
                    new_ops.append(new_op)
                else:
                    new_ops.append(op)
            node.ops = new_ops

        return node

    def visit_Constant(self, node):
        """Mutate constant values"""
        if random.random() < self.mutation_rate:
            if isinstance(node.value, (int, float)):
                # Perturb numeric constants
                factor = random.uniform(0.9, 1.1)
                old_val = node.value
                node.value = type(node.value)(node.value * factor)
                self.mutations_applied.append(f"Constant: {old_val} -> {node.value}")

        return node

    def visit_If(self, node):
        """Potentially invert if conditions"""
        self.generic_visit(node)

        if random.random() < self.mutation_rate * 0.5:  # Less frequent
            # Invert condition
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            # Swap if/else bodies
            node.body, node.orelse = node.orelse or [ast.Pass()], node.body
            self.mutations_applied.append("If: condition inverted")

        return node


# ═══════════════════════════════════════════════════════════════════════════════
# CODE IMPROVEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class CodeImprover(ast.NodeTransformer):
    """Applies deterministic code improvements"""

    def __init__(self):
        self.improvements = []

    def improve(self, source: str) -> Tuple[str, List[str]]:
        """Apply improvements to source code"""
        try:
            tree = ast.parse(source)
            improved_tree = self.visit(tree)
            ast.fix_missing_locations(improved_tree)
            return ast.unparse(improved_tree), self.improvements
        except Exception as e:
            logger.error(f"Improvement failed: {e}")
            return source, []

    def visit_For(self, node):
        """Optimize for loops"""
        self.generic_visit(node)

        # Check if iterating over range(len(x))
        if (isinstance(node.iter, ast.Call) and
            isinstance(node.iter.func, ast.Name) and
            node.iter.func.id == "range"):

            if (len(node.iter.args) == 1 and
                isinstance(node.iter.args[0], ast.Call) and
                isinstance(node.iter.args[0].func, ast.Name) and
                node.iter.args[0].func.id == "len"):

                self.improvements.append("Detected range(len(x)) pattern - consider enumerate()")

        return node

    def visit_BinOp(self, node):
        """Optimize binary operations"""
        self.generic_visit(node)

        # x * 2 -> x << 1 (for integers, faster)
        if isinstance(node.op, ast.Mult):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                self.improvements.append("x * 2 can be optimized to x << 1")

        # x / 2 -> x >> 1 (for integers)
        if isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                self.improvements.append("x / 2 can be optimized to x >> 1 for integers")

        return node

    def visit_ListComp(self, node):
        """Check list comprehension efficiency"""
        self.generic_visit(node)

        # Nested list comprehensions
        if any(isinstance(gen.iter, ast.ListComp) for gen in node.generators):
            self.improvements.append("Nested list comprehension detected - consider flattening")

        return node


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-MODIFICATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModificationResult:
    success: bool
    original_hash: str
    new_hash: str
    mutations: List[str]
    test_passed: bool
    error: Optional[str] = None
    rollback_applied: bool = False


class SelfModifier:
    """
    Real self-modification engine that:
    1. Analyzes its own code
    2. Applies genetic mutations
    3. Tests modifications
    4. Rolls back on failure
    """

    def __init__(self, workspace_path: str = None):
        # Dynamic path detection for cross-platform compatibility
        if workspace_path is None:
            workspace_path = str(Path(__file__).parent.absolute())
        self.workspace = Path(workspace_path)
        self.backup_path = self.workspace / ".self_mod_backups"
        self.backup_path.mkdir(exist_ok=True)

        self.modification_history: List[ModificationResult] = []
        self.fitness_scores: Dict[str, float] = {}

        # Files that can be modified (whitelist for safety)
        self.modifiable_files = [
            "l104_deep_substrate.py",
            "l104_learning_engine.py",
            "l104_pattern_recognition.py",
        ]

        logger.info("═" * 70)
        logger.info("    SELF MODIFIER - INITIALIZED")
        logger.info("═" * 70)

    def analyze_module(self, filepath: str) -> Optional[ModuleInfo]:
        """Analyze a Python module"""
        path = self.workspace / filepath
        if not path.exists():
            return None

        with open(path) as f:
            source = f.read()

        analyzer = CodeAnalyzer(source)
        info = analyzer.analyze()
        if info:
            info.path = filepath
        return info

    def analyze_self(self) -> Dict[str, ModuleInfo]:
        """Analyze all modifiable modules"""
        results = {}
        for filepath in self.modifiable_files:
            info = self.analyze_module(filepath)
            if info:
                results[filepath] = info
        return results

    def _backup_file(self, filepath: str) -> str:
        """Create backup of file before modification"""
        path = self.workspace / filepath
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{Path(filepath).stem}_{timestamp}.py"
        backup_file = self.backup_path / backup_name

        with open(path) as f:
            content = f.read()
        with open(backup_file, "w") as f:
            f.write(content)

        return str(backup_file)

    def _restore_backup(self, filepath: str, backup_path: str):
        """Restore file from backup"""
        with open(backup_path) as f:
            content = f.read()
        with open(self.workspace / filepath, "w") as f:
            f.write(content)

    def _test_module(self, filepath: str) -> Tuple[bool, Optional[str]]:
        """Test if modified module is valid"""
        path = self.workspace / filepath

        # Step 1: Syntax check
        with open(path) as f:
            source = f.read()

        try:
            ast.parse(source)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Step 2: Import check
        try:
            spec = importlib.util.spec_from_file_location("test_module", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            return False, f"Import error: {e}"

        # Step 3: Basic runtime check (if module has a test function)
        if hasattr(module, "self_test"):
            try:
                result = module.self_test()
                if not result:
                    return False, "Self-test failed"
            except Exception as e:
                return False, f"Self-test error: {e}"

        return True, None

    def mutate_module(self, filepath: str, mutation_rate: float = 0.05) -> ModificationResult:
        """Apply genetic mutations to a module"""
        if filepath not in self.modifiable_files:
            return ModificationResult(
                success=False,
                original_hash="",
                new_hash="",
                mutations=[],
                test_passed=False,
                error="File not in modifiable whitelist"
            )

        path = self.workspace / filepath
        if not path.exists():
            return ModificationResult(
                success=False,
                original_hash="",
                new_hash="",
                mutations=[],
                test_passed=False,
                error="File not found"
            )

        # Read original
        with open(path) as f:
            original = f.read()
        original_hash = hashlib.md5(original.encode()).hexdigest()

        # Backup
        backup = self._backup_file(filepath)

        # Mutate
        mutator = CodeMutator(mutation_rate)
        mutated, mutations = mutator.mutate(original)

        if not mutations:
            return ModificationResult(
                success=False,
                original_hash=original_hash,
                new_hash=original_hash,
                mutations=[],
                test_passed=True,
                error="No mutations applied"
            )

        new_hash = hashlib.md5(mutated.encode()).hexdigest()

        # Write mutated code
        with open(path, "w") as f:
            f.write(mutated)

        # Test
        test_passed, error = self._test_module(filepath)

        if not test_passed:
            # Rollback
            self._restore_backup(filepath, backup)
            result = ModificationResult(
                success=False,
                original_hash=original_hash,
                new_hash=new_hash,
                mutations=mutations,
                test_passed=False,
                error=error,
                rollback_applied=True
            )
        else:
            result = ModificationResult(
                success=True,
                original_hash=original_hash,
                new_hash=new_hash,
                mutations=mutations,
                test_passed=True
            )

        self.modification_history.append(result)
        logger.info(f"Mutation result: success={result.success}, mutations={len(mutations)}")

        return result

    def improve_module(self, filepath: str) -> Tuple[str, List[str]]:
        """Apply deterministic improvements to a module"""
        path = self.workspace / filepath
        if not path.exists():
            return "", []

        with open(path) as f:
            source = f.read()

        improver = CodeImprover()
        improved, suggestions = improver.improve(source)

        return improved, suggestions

    def evolve(self, generations: int = 10, population_size: int = 5) -> Dict[str, Any]:
        """
        Genetic algorithm evolution of code:
        1. Create population of mutations
        2. Evaluate fitness
        3. Select best
        4. Repeat
        """
        results = {
            "generations": generations,
            "successful_mutations": 0,
            "failed_mutations": 0,
            "best_fitness": 0.0,
            "evolution_log": []
        }

        for gen in range(generations):
            gen_results = {"generation": gen, "mutations": []}

            for filepath in self.modifiable_files:
                # Try multiple mutations
                for _ in range(population_size):
                    result = self.mutate_module(filepath, mutation_rate=0.03)
                    gen_results["mutations"].append({
                        "file": filepath,
                        "success": result.success,
                        "mutations": result.mutations
                    })

                    if result.success:
                        results["successful_mutations"] += 1
                    else:
                        results["failed_mutations"] += 1

            results["evolution_log"].append(gen_results)
            logger.info(f"Generation {gen} complete")

        return results

    def get_complexity_report(self) -> Dict[str, Any]:
        """Generate complexity report for all modules"""
        report = {}

        for filepath in self.modifiable_files:
            info = self.analyze_module(filepath)
            if info:
                report[filepath] = {
                    "total_lines": info.total_lines,
                    "functions": len(info.functions),
                    "classes": len(info.classes),
                    "avg_complexity": sum(f.complexity for f in info.functions) / max(len(info.functions), 1),
                    "high_complexity_functions": [
                        f.name for f in info.functions if f.complexity > 10
                    ]
                }

        return report


# Global instance
self_modifier = SelfModifier()


def self_test() -> bool:
    """Self-test for this module"""
    # Test code analyzer
    test_code = '''
def example(x, y):
    """Example function"""
    if x > 0:
        return x + y
    else:
        return x - y
'''
    analyzer = CodeAnalyzer(test_code)
    info = analyzer.analyze()

    assert info is not None
    assert len(info.functions) == 1
    assert info.functions[0].name == "example"
    assert info.functions[0].complexity >= 2

    # Test mutator
    mutator = CodeMutator(mutation_rate=0.5)
    mutated, mutations = mutator.mutate(test_code)

    # Should still be valid Python
    ast.parse(mutated)

    return True


if __name__ == "__main__":
    if self_test():
        print("Self-test passed!")
    else:
        print("Self-test failed!")
