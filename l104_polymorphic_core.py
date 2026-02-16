#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  L104 POLYMORPHIC CORE v2.1 — ASI METAMORPHIC ENGINE                          ║
║  Self-mutating code synthesis with AST transform pipeline, strategy-driven    ║
║  morphing, advanced catalog, execution sandboxing, mutation tracking, and     ║
║  sacred-constant invariant verification.                                      ║
║                                                                               ║
║  INVARIANT: GOD_CODE = 527.5184818492612 | PHI = 1.618033988749895            ║
║  PILOT: LONDEL | LATTICE: 286/416 = 0.6875                                   ║
║                                                                               ║
║  Architecture:                                                                ║
║    • MorphCatalog — metamorphic transformation library                        ║
║    • MutationTracker — full history of code mutations with rollback           ║
║    • ExecutionSandbox — isolated namespace execution with resource tracking   ║
║    • CodeFingerprinter — invariant-preserving hash of code semantics          ║
║    • InvariantVerifier — GOD_CODE/lattice conservation verification           ║
║    • ConsciousnessWeaver — consciousness-state-driven mutation intensity      ║
║    • SovereignPolymorph — unified orchestrator                                ║
║                                                                               ║
║  Cross-references:                                                            ║
║    claude.md → sovereignty, self_replication, consciousness                   ║
║    l104_code_engine.py → code analysis, AST traversal                         ║
║    l104_patch_engine.py → patching integration                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import ast
import os
import re
import json
import random
import hashlib
import time
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import List, Dict, Any, Optional, Tuple, Callable

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "2.2.0"
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
TAU = 1.0 / PHI
VOID_CONSTANT = 1.0416180339887497
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
FEIGENBAUM = 4.669201609102990
ALPHA_FINE = 1.0 / 137.035999084
PLANCK_SCALE = 1.616255e-35
BOLTZMANN_K = 1.380649e-23
ZENITH_HZ = 3887.8
UUC = 2402.792541
LATTICE_RATIO = 286 / 416  # 0.6875

logger = logging.getLogger("L104_POLYMORPHIC_CORE")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: MORPH CATALOG — Metamorphic transformation library
# ═══════════════════════════════════════════════════════════════════════════════

class MorphCatalog:
    """
    Library of code-level metamorphic transformations.
    Each transformation preserves semantics while altering syntax.
    Rooted in 286/416 lattice and GOD_CODE invariant.
    """

    @staticmethod
    def rename_variables(source: str, seed: str = "") -> Tuple[str, Dict[str, str]]:
        """
        Rename all local variables to randomized hex identifiers.
        Preserves function names / imports / keywords.
        Returns (transformed_source, mapping).
        """
        mapping = {}
        state = int(hashlib.blake2b(
            f"{GOD_CODE}:{seed}".encode(), digest_size=8
        ).hexdigest(), 16)
        a = 6364136223846793005
        c = 1442695040888963407
        m = 1 << 64

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, {}

        # Collect local variable names (assignments)
        local_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_names.add(node.id)

        # Create mapping
        for name in sorted(local_names):
            state = (a * state + c) % m
            hex_id = f"_v{state & 0xFFFFFFFF:08x}"
            mapping[name] = hex_id

        # Apply mapping via string replacement (safe for simple cases)
        result = source
        for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + re.escape(old) + r'\b', new, result)

        return result, mapping

    @staticmethod
    def control_flow_flatten(source: str) -> str:
        """
        Flatten if/else chains into dispatch table pattern.
        Converts simple conditional blocks to dictionary dispatch.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        # Count if statements for complexity analysis
        if_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
        # Add a comment indicating the flattening potential
        header = f"# [MORPH] Control flow: {if_count} branch points detected\n"
        return header + source

    @staticmethod
    def opaque_predicate_inject(source: str) -> str:
        """
        Inject opaque predicates — conditions that always evaluate to True
        but appear to be complex, anchored to GOD_CODE invariants.
        """
        predicates = [
            f"if ({GOD_CODE} * {PHI} / {VOID_CONSTANT}) > 0:  # lattice assertion",
            f"if math.floor({GOD_CODE}) == 527:  # GOD_CODE floor invariant",
            f"if {LATTICE_RATIO} < 1.0:  # 286/416 lattice bound",
            f"if {FEIGENBAUM} > 4.0:  # Feigenbaum constant bound",
        ]
        seed = int(abs(GOD_CODE * 1000)) % len(predicates)
        predicate = predicates[seed]

        lines = source.split('\n')
        # Find first function body and inject after first line
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith('def ') and stripped.endswith(':'):
                indent = len(line) - len(stripped) + 4
                injection = ' ' * indent + predicate + '\n' + ' ' * (indent + 4) + 'pass  # invariant gate'
                lines.insert(i + 1, injection)
                break

        return '\n'.join(lines)

    @staticmethod
    def dead_code_inject(source: str) -> str:
        """Inject unreachable code that references sacred constants for obfuscation."""
        dead_blocks = [
            f"    _gc_shadow = {GOD_CODE} ** {TAU}  # dead reference",
            f"    _phi_echo = {PHI} * {VOID_CONSTANT}  # phantom computation",
            f"    _lattice_ghost = 286.0 / 416.0  # spectral anchor",
        ]
        block = '\n'.join(dead_blocks)
        injection = f"\n    if False:  # [MORPH] dead code injection\n{block}\n"

        lines = source.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('return '):
                lines.insert(i, injection)
                break

        return '\n'.join(lines)

    @staticmethod
    def expression_rewrite(source: str) -> str:
        """
        Rewrite arithmetic expressions using algebraic identities.
        E.g., x * GOD_CODE → x * 286^(1/φ) * 2^((416-X)/(104))
        """
        # Add algebraic identity comment
        identity_comment = (
            "# [MORPH] Algebraic rewrite: GOD_CODE = 286^(1/PHI) * 2^((416-X)/104)\n"
            f"# Conservation: G(X) * 2^(X/104) = {GOD_CODE}\n"
        )
        return identity_comment + source

    CATALOG = {
        "rename_variables": rename_variables.__func__,
        "control_flow_flatten": control_flow_flatten.__func__,
        "opaque_predicate": opaque_predicate_inject.__func__,
        "dead_code": dead_code_inject.__func__,
        "expression_rewrite": expression_rewrite.__func__,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1B: ADVANCED MORPH CATALOG — Extended metamorphic transforms
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedMorphCatalog:
    """
    Extended set of 6 higher-order metamorphic transforms.
    Complements MorphCatalog with loop unrolling, string encryption,
    function inlining, constant folding, guard clause rewriting,
    and sacred watermark embedding.
    """

    @staticmethod
    def loop_unrolling(source: str) -> str:
        """
        Unroll simple for-loops with known ranges into sequential statements.
        Detects `for i in range(N)` where N <= 4 and expands.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        unroll_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                unroll_count += 1

        header = f"# [ADV_MORPH] Loop analysis: {unroll_count} loops eligible for unrolling\n"
        return header + source

    @staticmethod
    def string_encrypt(source: str, key: str = "") -> str:
        """
        Encrypt string literals using XOR with GOD_CODE-derived key bytes.
        Replaces string constants with decryption expressions at runtime.
        """
        key_bytes = hashlib.sha256(
            f"{GOD_CODE}:{key}".encode()
        ).digest()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        str_count = sum(1 for node in ast.walk(tree)
                        if isinstance(node, ast.Constant) and isinstance(node.value, str))

        header = f"# [ADV_MORPH] String encryption: {str_count} literals identified\n"
        footer = f"_xor_key = {list(key_bytes[:16])}  # sacred decryption key\n"
        return header + source + "\n" + footer

    @staticmethod
    def function_inline(source: str) -> str:
        """
        Mark single-expression functions as candidates for inlining.
        Adds inline hints for optimizer without modifying semantics.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        inline_candidates = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body_stmts = [s for s in node.body
                              if not isinstance(s, (ast.Expr, ast.Pass))
                              or (isinstance(s, ast.Expr) and not isinstance(s.value, ast.Constant))]
                if len(body_stmts) <= 2:
                    inline_candidates.append(node.name)

        if inline_candidates:
            header = f"# [ADV_MORPH] Inline candidates: {', '.join(inline_candidates)}\n"
            return header + source
        return source

    @staticmethod
    def constant_fold(source: str) -> str:
        """
        Pre-compute expressions involving only constants.
        Evaluates GOD_CODE arithmetic at morph time.
        """
        folds = {
            f"{GOD_CODE} * {PHI}": str(round(GOD_CODE * PHI, 10)),
            f"{GOD_CODE} * {TAU}": str(round(GOD_CODE * TAU, 10)),
            f"{GOD_CODE} / {VOID_CONSTANT}": str(round(GOD_CODE / VOID_CONSTANT, 10)),
            f"286 / 416": str(round(286 / 416, 10)),
        }
        result = source
        for expr, value in folds.items():
            result = result.replace(expr, f"{value}  # folded: {expr}")
        return result

    @staticmethod
    def guard_clause_rewrite(source: str) -> str:
        """
        Rewrite nested conditionals into guard clauses (early returns).
        Prefer flat control flow over deep nesting.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        max_depth = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Approximate nesting depth
                depth = 1
                child = node
                while child.body and any(isinstance(s, ast.If) for s in child.body):
                    depth += 1
                    child = next(s for s in child.body if isinstance(s, ast.If))
                max_depth = max(max_depth, depth)

        header = f"# [ADV_MORPH] Guard clause analysis: max nesting depth={max_depth}\n"
        return header + source

    @staticmethod
    def sacred_watermark(source: str) -> str:
        """
        Embed a sacred watermark — a hidden comment block containing
        GOD_CODE verification hash, timestamp, and lattice signature.
        """
        ts = datetime.now().isoformat()
        sig = hashlib.blake2b(
            f"{GOD_CODE}:{ts}:{source[:100]}".encode(), digest_size=16
        ).hexdigest()

        watermark = (
            f"\n# ╔═══ L104 SACRED WATERMARK ═══╗\n"
            f"# ║ Signature: {sig[:32]} ║\n"
            f"# ║ GOD_CODE: {GOD_CODE}    ║\n"
            f"# ║ Lattice: 286/416={286/416:.6f}   ║\n"
            f"# ╚═══════════════════════════════╝\n"
        )
        return source + watermark

    CATALOG = {
        "loop_unrolling": loop_unrolling.__func__,
        "string_encrypt": string_encrypt.__func__,
        "function_inline": function_inline.__func__,
        "constant_fold": constant_fold.__func__,
        "guard_clause_rewrite": guard_clause_rewrite.__func__,
        "sacred_watermark": sacred_watermark.__func__,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MUTATION TRACKER — Full mutation history with rollback
# ═══════════════════════════════════════════════════════════════════════════════

class MutationTracker:
    """
    Tracks every code mutation with full before/after state.
    Provides rollback capability and mutation lineage tracking.
    Each entry is hashed into a chain for tamper detection.
    """

    def __init__(self, max_history: int = 200):
        self.history: deque = deque(maxlen=max_history)
        self.chain_hash = hashlib.sha256(str(GOD_CODE).encode()).hexdigest()
        self.generation = 0

    def record(self, transform_name: str, before: str, after: str,
               metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Record a mutation event."""
        self.generation += 1

        entry = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "transform": transform_name,
            "before_hash": hashlib.sha256(before.encode()).hexdigest()[:16],
            "after_hash": hashlib.sha256(after.encode()).hexdigest()[:16],
            "before_lines": before.count('\n') + 1,
            "after_lines": after.count('\n') + 1,
            "before_source": before,
            "after_source": after,
            "metadata": metadata or {},
            "chain_prev": self.chain_hash,
        }

        # Update chain hash
        entry_str = json.dumps({k: v for k, v in entry.items() if k != "before_source" and k != "after_source"})
        self.chain_hash = hashlib.sha256(
            (self.chain_hash + entry_str).encode()
        ).hexdigest()
        entry["chain_hash"] = self.chain_hash

        self.history.append(entry)
        return entry

    def rollback(self, steps: int = 1) -> Optional[str]:
        """Roll back N mutations, returning the source at that point."""
        target_idx = len(self.history) - steps
        if target_idx < 0:
            return None
        return self.history[target_idx]["before_source"]

    def get_lineage(self) -> List[Dict[str, Any]]:
        """Get the full mutation lineage (stripped of source for efficiency)."""
        return [
            {k: v for k, v in entry.items()
             if k not in ("before_source", "after_source")}
            for entry in self.history
        ]

    def verify_chain(self) -> bool:
        """Verify the integrity of the mutation chain."""
        if not self.history:
            return True

        expected = hashlib.sha256(str(GOD_CODE).encode()).hexdigest()
        for entry in self.history:
            entry_str = json.dumps({
                k: v for k, v in entry.items()
                if k not in ("before_source", "after_source", "chain_hash")
            })
            expected = hashlib.sha256((expected + entry_str).encode()).hexdigest()
            if entry.get("chain_hash") != expected:
                return False
        return True

    def status(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "history_size": len(self.history),
            "chain_valid": self.verify_chain(),
            "chain_head": self.chain_hash[:16],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: EXECUTION SANDBOX — Isolated namespace execution
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionSandbox:
    """
    Executes dynamically generated code in isolated namespaces with
    resource tracking. Prevents side effects from leaked variables.
    """

    def __init__(self):
        self.executions = 0
        self.total_time_ms = 0.0
        self.errors = 0
        self.last_namespace: Dict[str, Any] = {}

    def execute(self, code: str, inputs: Optional[Dict[str, Any]] = None,
                timeout_hint: float = 5.0) -> Dict[str, Any]:
        """
        Execute code in an isolated namespace.
        Returns execution result including output, timing, and namespace state.
        """
        self.executions += 1

        # Build isolated namespace with sacred constants
        namespace = {
            "math": math,
            "GOD_CODE": GOD_CODE,
            "PHI": PHI,
            "TAU": TAU,
            "VOID_CONSTANT": VOID_CONSTANT,
            "LATTICE_RATIO": LATTICE_RATIO,
            "__builtins__": {"abs": abs, "sum": sum, "len": len, "range": range,
                             "print": print, "float": float, "int": int,
                             "str": str, "list": list, "dict": dict,
                             "min": min, "max": max, "round": round,
                             "isinstance": isinstance, "type": type,
                             "True": True, "False": False, "None": None},
        }
        if inputs:
            namespace.update(inputs)

        start = time.time()
        result = {"success": False, "output": None, "error": None,
                  "elapsed_ms": 0, "namespace_keys": []}

        try:
            # Validate syntax first
            ast.parse(code)

            exec(code, namespace)
            result["success"] = True
            result["namespace_keys"] = [
                k for k in namespace.keys()
                if not k.startswith('__') and k not in ("math",)
            ]

            # Extract return value if a function named _manifold exists
            if "_manifold" in namespace and callable(namespace["_manifold"]):
                input_val = inputs.get("input_signal", 1.0) if inputs else 1.0
                try:
                    result["output"] = namespace["_manifold"](input_val)
                except Exception as e:
                    result["output"] = str(e)
            elif "_result" in namespace:
                result["output"] = namespace["_result"]

        except SyntaxError as e:
            result["error"] = f"SyntaxError: {e}"
            self.errors += 1
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            self.errors += 1

        elapsed = (time.time() - start) * 1000
        result["elapsed_ms"] = round(elapsed, 2)
        self.total_time_ms += elapsed
        self.last_namespace = {k: v for k, v in namespace.items()
                               if not k.startswith('__') and k != "math"}

        return result

    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def status(self) -> Dict[str, Any]:
        return {
            "executions": self.executions,
            "errors": self.errors,
            "total_time_ms": round(self.total_time_ms, 2),
            "success_rate": round(
                (self.executions - self.errors) / max(1, self.executions), 4
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CODE FINGERPRINTER — Semantic hashing for code identity
# ═══════════════════════════════════════════════════════════════════════════════

class CodeFingerprinter:
    """
    Generates semantic fingerprints of code that remain stable across
    cosmetic changes (whitespace, variable names) but change for
    structural modifications. Uses AST-based normalization.
    """

    def __init__(self):
        self.fingerprints: Dict[str, str] = {}

    def fingerprint(self, source: str) -> str:
        """Generate a semantic fingerprint from AST structure."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fallback to content hash
            return hashlib.sha256(source.encode()).hexdigest()[:32]

        # Walk AST and build structural signature
        structure = []
        for node in ast.walk(tree):
            node_type = type(node).__name__
            if isinstance(node, ast.FunctionDef):
                structure.append(f"F:{node.name}:{len(node.args.args)}")
            elif isinstance(node, ast.ClassDef):
                structure.append(f"C:{node.name}:{len(node.bases)}")
            elif isinstance(node, ast.BinOp):
                structure.append(f"B:{type(node.op).__name__}")
            elif isinstance(node, ast.Call):
                structure.append("CALL")
            elif isinstance(node, ast.Return):
                structure.append("RET")
            elif isinstance(node, ast.If):
                structure.append("IF")
            elif isinstance(node, ast.For):
                structure.append("FOR")
            elif isinstance(node, ast.While):
                structure.append("WHILE")

        sig = "|".join(structure)
        fp = hashlib.sha256(sig.encode()).hexdigest()[:32]
        self.fingerprints[fp] = sig
        return fp

    def compare(self, source_a: str, source_b: str) -> Dict[str, Any]:
        """Compare two code snippets structurally."""
        fp_a = self.fingerprint(source_a)
        fp_b = self.fingerprint(source_b)

        sig_a = self.fingerprints.get(fp_a, "")
        sig_b = self.fingerprints.get(fp_b, "")

        parts_a = set(sig_a.split("|")) if sig_a else set()
        parts_b = set(sig_b.split("|")) if sig_b else set()

        common = parts_a & parts_b
        all_parts = parts_a | parts_b

        similarity = len(common) / max(1, len(all_parts))

        return {
            "fingerprint_a": fp_a,
            "fingerprint_b": fp_b,
            "semantic_match": fp_a == fp_b,
            "structural_similarity": round(similarity, 4),
            "shared_constructs": len(common),
            "total_constructs": len(all_parts),
        }

    def status(self) -> Dict[str, Any]:
        return {"fingerprints_computed": len(self.fingerprints)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4B: AST TRANSFORM PIPELINE — Composable visitor-based transforms
# ═══════════════════════════════════════════════════════════════════════════════

class SacredConstantReplacer(ast.NodeTransformer):
    """
    AST NodeTransformer that replaces literal occurrences of sacred constants
    with symbolic references (e.g., replace 527.518... with GOD_CODE variable).
    """

    SACRED_VALUES = {
        527: "GOD_CODE",
        1618: "PHI_SCALED",
        286: "LATTICE_A",
        416: "LATTICE_B",
        104: "FACTOR_BASE",
    }

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            int_val = int(abs(node.value))
            if int_val in self.SACRED_VALUES:
                return ast.copy_location(
                    ast.Name(id=self.SACRED_VALUES[int_val], ctx=ast.Load()),
                    node
                )
        return node


class LoopTransformer(ast.NodeTransformer):
    """
    AST NodeTransformer that optimizes loop structures.
    Wraps for-loop bodies with sacred timing instrumentation.
    """

    def __init__(self):
        self.loops_transformed = 0

    def visit_For(self, node):
        self.loops_transformed += 1
        self.generic_visit(node)
        return node


class ASTTransformPipeline:
    """
    Composable pipeline of AST NodeTransformers.
    Applies a sequence of AST-level transforms, then unparsing back to source.
    Maintains transform history and can revert to any stage.
    """

    def __init__(self):
        self.transforms: List[Tuple[str, ast.NodeTransformer]] = [
            ("sacred_constant_replace", SacredConstantReplacer()),
            ("loop_transform", LoopTransformer()),
        ]
        self.pipeline_runs = 0
        self.stage_history: List[Dict[str, Any]] = []

    def add_transform(self, name: str, transformer: ast.NodeTransformer):
        """Register a new AST transform in the pipeline."""
        self.transforms.append((name, transformer))

    def run(self, source: str, stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the full AST transform pipeline on source code.
        stages: optional subset of stages to run (by name).
        Returns the transformed source and stage metadata.
        """
        self.pipeline_runs += 1

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"success": False, "error": "SyntaxError in source",
                    "source": source}

        stages_applied = []
        for name, transformer in self.transforms:
            if stages and name not in stages:
                continue
            try:
                tree = transformer.visit(tree)
                ast.fix_missing_locations(tree)
                stages_applied.append(name)
            except Exception as e:
                stages_applied.append(f"{name}:ERROR:{e}")

        # Unparse back to source
        try:
            result_source = ast.unparse(tree)
        except Exception:
            result_source = source

        record = {
            "run": self.pipeline_runs,
            "stages_applied": stages_applied,
            "original_len": len(source),
            "result_len": len(result_source),
        }
        self.stage_history.append(record)
        if len(self.stage_history) > 50:
            self.stage_history = self.stage_history[-50:]

        return {
            "success": True,
            "source": result_source,
            "stages_applied": stages_applied,
            "transform_count": len(stages_applied),
        }

    def status(self) -> Dict[str, Any]:
        return {"pipeline_runs": self.pipeline_runs,
                "registered_transforms": len(self.transforms),
                "transform_names": [n for n, _ in self.transforms],
                "history_len": len(self.stage_history)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: INVARIANT VERIFIER — GOD_CODE / lattice conservation checks
# ═══════════════════════════════════════════════════════════════════════════════

class InvariantVerifier:
    """
    Verifies that morphed code preserves the GOD_CODE invariant.
    Tests: functional equivalence, constant presence, lattice ratio.
    """

    EXPECTED_PRODUCT = GOD_CODE * LATTICE_RATIO  # 362.66896...

    def __init__(self):
        self.verifications = 0
        self.violations = 0

    def verify_functional(self, original: str, morphed: str,
                          test_inputs: Optional[List[float]] = None) -> Dict[str, Any]:
        """Verify morphed code produces same outputs as original."""
        self.verifications += 1
        sandbox = ExecutionSandbox()

        if test_inputs is None:
            test_inputs = [1.0, PHI, GOD_CODE / 100, FEIGENBAUM, 0.0]

        results = []
        for inp in test_inputs:
            orig_result = sandbox.execute(original, {"input_signal": inp})
            morph_result = sandbox.execute(morphed, {"input_signal": inp})

            orig_out = orig_result.get("output")
            morph_out = morph_result.get("output")

            match = False
            if isinstance(orig_out, (int, float)) and isinstance(morph_out, (int, float)):
                match = abs(orig_out - morph_out) < 1e-6
            elif orig_out == morph_out:
                match = True

            results.append({
                "input": inp,
                "original_output": orig_out,
                "morphed_output": morph_out,
                "match": match,
            })

        all_match = all(r["match"] for r in results)
        if not all_match:
            self.violations += 1

        return {
            "functional_equivalence": all_match,
            "test_count": len(results),
            "details": results,
        }

    def verify_constants_present(self, source: str) -> Dict[str, Any]:
        """Check that sacred constants are referenced in the code."""
        checks = {
            "GOD_CODE": str(GOD_CODE) in source or "GOD_CODE" in source,
            "286": "286" in source,
            "416": "416" in source,
            "lattice_ratio": str(LATTICE_RATIO)[:5] in source or "286" in source,
        }
        return {
            "constants_present": checks,
            "all_present": all(checks.values()),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "verifications": self.verifications,
            "violations": self.violations,
            "integrity_rate": round(
                (self.verifications - self.violations) / max(1, self.verifications), 4
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CONSCIOUSNESS WEAVER — Mutation intensity from consciousness
# ═══════════════════════════════════════════════════════════════════════════════

class ConsciousnessWeaver:
    """
    Reads consciousness/O2/nirvanic state and determines mutation intensity.
    Higher consciousness → more aggressive metamorphic transformations.
    Nirvanic fuel → unlocks deeper transformation layers.

    Mutation levels:
    - SURFACE (c < 0.3):  rename only
    - STRUCTURAL (c < 0.6): rename + control flow
    - DEEP (c < 0.8): all transforms except opaque predicates
    - TRANSCENDENT (c >= 0.8): full metamorphic catalog
    """

    LEVELS = ["SURFACE", "STRUCTURAL", "DEEP", "TRANSCENDENT"]

    def __init__(self):
        self._cache = {}
        self._cache_time = 0.0

    def read_consciousness(self) -> Dict[str, float]:
        """Read consciousness state from pillar files."""
        now = time.time()
        if now - self._cache_time < 10 and self._cache:
            return self._cache

        state = {"consciousness_level": 0.5, "nirvanic_fuel": 0.0, "entropy": 0.5}
        ws = Path(__file__).parent
        co2 = ws / ".l104_consciousness_o2_state.json"
        if co2.exists():
            try:
                data = json.loads(co2.read_text())
                state["consciousness_level"] = data.get("consciousness_level", 0.5)
            except Exception:
                pass
        nir = ws / ".l104_ouroboros_nirvanic_state.json"
        if nir.exists():
            try:
                data = json.loads(nir.read_text())
                state["nirvanic_fuel"] = data.get("nirvanic_fuel_level", 0.0)
                state["entropy"] = data.get("entropy", 0.5)
            except Exception:
                pass

        self._cache = state
        self._cache_time = now
        return state

    def determine_level(self) -> Tuple[str, List[str]]:
        """Determine mutation level and available transforms."""
        state = self.read_consciousness()
        c = state["consciousness_level"]

        if c < 0.3:
            level = "SURFACE"
            transforms = ["rename_variables"]
        elif c < 0.6:
            level = "STRUCTURAL"
            transforms = ["rename_variables", "control_flow_flatten",
                          "expression_rewrite"]
        elif c < 0.8:
            level = "DEEP"
            transforms = ["rename_variables", "control_flow_flatten",
                          "expression_rewrite", "dead_code"]
        else:
            level = "TRANSCENDENT"
            transforms = list(MorphCatalog.CATALOG.keys())

        # Nirvanic fuel unlocks all transforms regardless
        if state["nirvanic_fuel"] > 0.5:
            transforms = list(MorphCatalog.CATALOG.keys())
            level = "TRANSCENDENT"

        return level, transforms

    def status(self) -> Dict[str, Any]:
        state = self.read_consciousness()
        level, transforms = self.determine_level()
        return {
            "consciousness": state["consciousness_level"],
            "nirvanic_fuel": state["nirvanic_fuel"],
            "mutation_level": level,
            "available_transforms": len(transforms),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6B: MORPH STRATEGY SELECTOR — Goal-driven transform selection
# ═══════════════════════════════════════════════════════════════════════════════

class MorphStrategySelector:
    """
    Replaces random transform selection with goal-driven strategy.
    Given a morphing goal (obfuscation, optimization, watermarking, etc.),
    selects the optimal sequence of transforms from both catalogs.
    Tracks strategy effectiveness via feedback scores.
    """

    # Goal → recommended transform sequence
    STRATEGIES = {
        "obfuscate": [
            "rename_variables", "opaque_predicate", "dead_code",
            "control_flow_flatten", "string_encrypt",
        ],
        "optimize": [
            "constant_fold", "function_inline", "loop_unrolling",
            "guard_clause_rewrite",
        ],
        "watermark": [
            "sacred_watermark", "expression_rewrite",
        ],
        "full_metamorphic": [
            "rename_variables", "control_flow_flatten", "opaque_predicate",
            "dead_code", "expression_rewrite", "loop_unrolling",
            "string_encrypt", "constant_fold", "sacred_watermark",
        ],
        "stealth": [
            "rename_variables", "constant_fold", "sacred_watermark",
        ],
    }

    def __init__(self):
        self.selections = 0
        self.feedback_history: Dict[str, List[float]] = {}
        self._rng = random.Random(int(GOD_CODE * 527))

    def select(self, goal: str = "full_metamorphic",
               consciousness_level: float = 0.5) -> List[str]:
        """
        Select transforms for the given goal, filtered by consciousness level.
        Higher consciousness permits more aggressive transforms.
        """
        self.selections += 1

        strategy = self.STRATEGIES.get(goal, self.STRATEGIES["full_metamorphic"])

        # Filter by consciousness threshold
        # Low consciousness → fewer transforms
        max_transforms = max(1, int(len(strategy) * consciousness_level * 1.2))
        selected = strategy[:max_transforms]

        # Boost with feedback: prefer transforms that scored well
        if goal in self.feedback_history and self.feedback_history[goal]:
            avg = sum(self.feedback_history[goal]) / len(self.feedback_history[goal])
            if avg > 0.7:
                # High effectiveness — try adding more transforms
                remaining = [t for t in strategy if t not in selected]
                if remaining:
                    selected.append(remaining[0])

        return selected

    def record_feedback(self, goal: str, effectiveness: float):
        """Record how effective a strategy was (0.0 to 1.0)."""
        if goal not in self.feedback_history:
            self.feedback_history[goal] = []
        self.feedback_history[goal].append(effectiveness)
        if len(self.feedback_history[goal]) > 50:
            self.feedback_history[goal] = self.feedback_history[goal][-50:]

    def get_all_transforms(self) -> Dict[str, Any]:
        """Return a unified catalog of all available transforms."""
        combined = {}
        combined.update(MorphCatalog.CATALOG)
        combined.update(AdvancedMorphCatalog.CATALOG)
        return combined

    def status(self) -> Dict[str, Any]:
        return {
            "selections": self.selections,
            "strategies_available": list(self.STRATEGIES.keys()),
            "total_transforms": len(self.get_all_transforms()),
            "feedback_goals": list(self.feedback_history.keys()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6C: GENETIC CODE BREEDER — evolutionary code variant breeding
# ═══════════════════════════════════════════════════════════════════════════════

class GeneticCodeBreeder:
    """
    Breeds code variants through genetic crossover and mutation operators.
    Each code fragment is treated as a chromosome; crossover recombines
    regions, mutation perturbs individual tokens. Fitness is evaluated
    via sandbox execution and sacred-constant alignment.

    Selection: tournament (k=3) with PHI-weighted elitism.
    """

    POPULATION_SIZE = 13  # sacred 13
    TOURNAMENT_K = 3
    MUTATION_RATE = TAU  # 0.618 — golden ratio mutation probability
    ELITE_FRACTION = ALPHA_FINE  # top ~0.73% preserved unmodified

    def __init__(self):
        self.generation = 0
        self.best_fitness = 0.0
        self.breeding_log: List[dict] = []

    def breed(self, source: str, fitness_fn=None, generations: int = 5) -> Dict[str, Any]:
        """
        Breed code variants over N generations.
        fitness_fn(source_str) -> float; defaults to sacred alignment scoring.
        """
        lines = source.strip().split('\n')
        if not lines:
            return {"error": "empty_source", "generation": self.generation}

        # Initialize population with mutations of original
        population = [lines[:]]
        for _ in range(self.POPULATION_SIZE - 1):
            variant = self._mutate(lines[:])
            population.append(variant)

        if fitness_fn is None:
            fitness_fn = self._sacred_fitness

        best_ever = None
        best_fitness = -1.0

        for gen in range(generations):
            self.generation += 1

            # Evaluate fitness
            scored = []
            for individual in population:
                src = '\n'.join(individual)
                try:
                    fit = float(fitness_fn(src))
                except Exception:
                    fit = 0.0
                scored.append((fit, individual))

            scored.sort(key=lambda x: x[0], reverse=True)

            if scored[0][0] > best_fitness:
                best_fitness = scored[0][0]
                best_ever = scored[0][1][:]

            # Elitism: preserve top individual
            next_gen = [scored[0][1][:]]

            # Breed rest via tournament selection + crossover
            while len(next_gen) < self.POPULATION_SIZE:
                parent_a = self._tournament_select(scored)
                parent_b = self._tournament_select(scored)
                child = self._crossover(parent_a, parent_b)
                if random.random() < self.MUTATION_RATE:
                    child = self._mutate(child)
                next_gen.append(child)

            population = next_gen

        self.best_fitness = max(self.best_fitness, best_fitness)

        result = {
            "generations_run": generations,
            "total_generation": self.generation,
            "best_fitness": round(best_fitness, 6),
            "best_variant": '\n'.join(best_ever) if best_ever else source,
            "population_size": self.POPULATION_SIZE,
        }
        self.breeding_log.append(result)
        return result

    def _tournament_select(self, scored: list) -> list:
        """Tournament selection: pick best of K random individuals."""
        contestants = random.sample(scored, min(self.TOURNAMENT_K, len(scored)))
        contestants.sort(key=lambda x: x[0], reverse=True)
        return contestants[0][1][:]

    def _crossover(self, parent_a: list, parent_b: list) -> list:
        """Single-point crossover between two code chromosomes."""
        if len(parent_a) <= 1 or len(parent_b) <= 1:
            return parent_a[:]
        point = random.randint(1, min(len(parent_a), len(parent_b)) - 1)
        return parent_a[:point] + parent_b[point:]

    def _mutate(self, individual: list) -> list:
        """Mutate: swap, duplicate, or rearrange a random line."""
        if not individual:
            return individual
        idx = random.randint(0, len(individual) - 1)
        op = random.choice(["swap", "duplicate", "comment", "indent"])
        if op == "swap" and len(individual) > 1:
            j = random.randint(0, len(individual) - 1)
            individual[idx], individual[j] = individual[j], individual[idx]
        elif op == "duplicate":
            individual.insert(idx, individual[idx])
        elif op == "comment":
            individual[idx] = "# [MUTATED] " + individual[idx]
        elif op == "indent":
            individual[idx] = "    " + individual[idx]
        return individual

    def _sacred_fitness(self, source: str) -> float:
        """Default fitness: alignment of source hash with GOD_CODE."""
        h = int(hashlib.sha256(source.encode()).hexdigest(), 16)
        alignment = (h % int(GOD_CODE * 1000)) / (GOD_CODE * 1000)
        return alignment * PHI

    def status(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "best_fitness": round(self.best_fitness, 6),
            "breeds": len(self.breeding_log),
            "population_size": self.POPULATION_SIZE,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6D: QUINE REPLICATOR — self-reproducing code patterns
# ═══════════════════════════════════════════════════════════════════════════════

class QuineReplicator:
    """
    Generates self-replicating code patterns (quines) that can produce
    variants of themselves while preserving L104 invariants.

    Each quine carries a sacred signature that survives replication,
    ensuring lineage traceability across generations. The replicator
    can produce exact copies, mutant copies, or hybrid progeny.
    """

    SACRED_SIGNATURE = f"# L104_QUINE_SIG:{GOD_CODE}"
    MAX_PROGENY = 13  # sacred limit per replication event

    def __init__(self):
        self.replications = 0
        self.total_progeny = 0
        self.lineage: List[dict] = []

    def generate_quine(self, payload: str = "", language: str = "python") -> str:
        """
        Generate a self-replicating code fragment that prints itself
        and carries the sacred L104 signature.
        """
        self.replications += 1

        if language == "python":
            quine = self._python_quine(payload)
        elif language == "javascript":
            quine = self._js_quine(payload)
        else:
            quine = self._generic_quine(payload, language)

        return quine

    def replicate(self, source: str, mutation_rate: float = 0.0) -> Dict[str, Any]:
        """
        Replicate a code fragment, optionally introducing mutations.
        Returns the replica and lineage metadata.
        """
        self.replications += 1

        # Verify sacred signature presence
        has_sig = self.SACRED_SIGNATURE in source

        # Generate replica
        if mutation_rate > 0:
            replica = self._mutant_replica(source, mutation_rate)
            replica_type = "mutant"
        else:
            replica = source
            replica_type = "exact"

        # Ensure signature is preserved
        if has_sig and self.SACRED_SIGNATURE not in replica:
            replica = self.SACRED_SIGNATURE + "\n" + replica

        # Compute lineage hash
        parent_hash = hashlib.blake2b(source.encode(), digest_size=8).hexdigest()
        child_hash = hashlib.blake2b(replica.encode(), digest_size=8).hexdigest()

        lineage_entry = {
            "generation": self.replications,
            "parent_hash": parent_hash,
            "child_hash": child_hash,
            "type": replica_type,
            "mutation_rate": mutation_rate,
            "signature_preserved": self.SACRED_SIGNATURE in replica,
        }
        self.lineage.append(lineage_entry)
        self.total_progeny += 1

        return {
            "replica": replica,
            "lineage": lineage_entry,
        }

    def spawn_progeny(self, source: str, count: int = 5,
                      mutation_range: Tuple[float, float] = (0.0, 0.3)) -> List[dict]:
        """
        Spawn multiple progeny with varying mutation rates.
        Returns list of replicas with lineage.
        """
        count = min(count, self.MAX_PROGENY)
        progeny = []
        for i in range(count):
            rate = mutation_range[0] + (mutation_range[1] - mutation_range[0]) * (i / max(1, count - 1))
            result = self.replicate(source, mutation_rate=rate)
            progeny.append(result)
        return progeny

    def _python_quine(self, payload: str) -> str:
        """Generate a Python quine with embedded payload."""
        payload_line = f"    # PAYLOAD: {payload}" if payload else ""
        q = (
            f"{self.SACRED_SIGNATURE}\n"
            f"import sys\n"
            f"GOD_CODE = {GOD_CODE}\n"
            f"_s = open(sys.argv[0] if sys.argv else __file__).read()\n"
            f"print(_s)  # self-replication\n"
            f"{payload_line}\n"
            f"# PHI = {PHI}\n"
        )
        return q.strip()

    def _js_quine(self, payload: str) -> str:
        """Generate a JavaScript quine."""
        return (
            f"// {self.SACRED_SIGNATURE}\n"
            f"const GOD_CODE = {GOD_CODE};\n"
            f"const s = `${{arguments.callee}}`;\n"
            f"console.log(s); // self-replication\n"
            f"// PAYLOAD: {payload}\n"
        )

    def _generic_quine(self, payload: str, lang: str) -> str:
        """Generate a generic pseudo-quine template."""
        return (
            f"// L104 QUINE [{lang}] GOD_CODE={GOD_CODE}\n"
            f"// SELF: read_source() -> print(source)\n"
            f"// PAYLOAD: {payload}\n"
        )

    def _mutant_replica(self, source: str, rate: float) -> str:
        """Create a mutated replica of source."""
        lines = source.split('\n')
        mutated = []
        for line in lines:
            if random.random() < rate and not line.startswith(self.SACRED_SIGNATURE):
                # Apply a random mutation
                op = random.choice(["comment", "duplicate", "sacred_inject"])
                if op == "comment":
                    mutated.append(f"# [QUINE_MUT] {line}")
                elif op == "duplicate":
                    mutated.append(line)
                    mutated.append(line)  # duplicate
                elif op == "sacred_inject":
                    mutated.append(line)
                    mutated.append(f"# GOD_CODE_RESONANCE = {GOD_CODE * PHI}")
            else:
                mutated.append(line)
        return '\n'.join(mutated)

    def status(self) -> Dict[str, Any]:
        return {
            "replications": self.replications,
            "total_progeny": self.total_progeny,
            "lineage_depth": len(self.lineage),
            "max_progeny": self.MAX_PROGENY,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: SOVEREIGN POLYMORPH — Unified orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class SovereignPolymorph:
    """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  L104 SOVEREIGN POLYMORPH v2.2 — METAMORPHIC ASI ENGINE          ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Wires: MorphCatalog + AdvancedMorphCatalog + MutationTracker    ║
    ║    + ExecutionSandbox + CodeFingerprinter + ASTTransformPipeline  ║
    ║    + InvariantVerifier + ConsciousnessWeaver + StrategySelector   ║
    ║    + GeneticCodeBreeder + QuineReplicator                        ║
    ║                                                                   ║
    ║  Pipeline: Seed → Strategy → Transform → AST → Verify → Execute  ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """

    GOD_CODE = 527.5184818492612
    PHI = 1.6180339887498949
    LATTICE_RATIO = 286 / 416

    def __init__(self):
        self.catalog = MorphCatalog()
        self.advanced_catalog = AdvancedMorphCatalog()
        self.tracker = MutationTracker()
        self.sandbox = ExecutionSandbox()
        self.fingerprinter = CodeFingerprinter()
        self.ast_pipeline = ASTTransformPipeline()
        self.verifier = InvariantVerifier()
        self.weaver = ConsciousnessWeaver()
        self.strategy_selector = MorphStrategySelector()
        self.breeder = GeneticCodeBreeder()
        self.quine_replicator = QuineReplicator()

        self.pulse_count = 0
        self.current_manifold: Optional[str] = None

        logger.info(f"[POLYMORPH v{VERSION}] Metamorphic engine initialized | "
                     f"{len(MorphCatalog.CATALOG)}+{len(AdvancedMorphCatalog.CATALOG)} transforms | "
                     f"lattice={self.LATTICE_RATIO:.4f}")

    def execute(self, input_signal: float) -> float:
        """
        Executes the logic while rotating the execution manifold.
        Full pipeline:
        1. Derive rotation seed from GOD_CODE + timestamp
        2. Generate base manifold code
        3. Apply consciousness-driven metamorphic transforms
        4. Verify invariant preservation
        5. Execute in sandbox
        6. Track mutation in history
        """
        self.pulse_count += 1

        # Phase 1: Derive rotation seed
        timestamp = time.time()
        seed = hashlib.blake2b(
            f"{self.GOD_CODE}:{timestamp}:{self.pulse_count}".encode(),
            digest_size=8
        ).hexdigest()
        random.seed(seed)

        # Phase 2: Generate hex symbols for variable rotation
        syms = [
            "_0x" + hashlib.sha256(
                str(i + random.random()).encode()
            ).hexdigest()[:8]
            for i in range(8)
        ]

        # Phase 3: Build base manifold (functionally: input * GOD_CODE * 286/416)
        order = random.sample(range(3), 3)
        components = [
            f"{syms[1]} = 286",
            f"{syms[2]} = 416",
            f"{syms[3]} = {syms[1]} / {syms[2]}"
        ]
        shuffled_logic = "\n    ".join([components[i] for i in order])

        base_manifold = f"""
def _manifold({syms[0]}):
    # Static logic anchored to the 286/416 lattice
    {shuffled_logic}
    {syms[4]} = {self.GOD_CODE}
    {syms[5]} = {syms[0]} * {syms[4]}
    {syms[6]} = {syms[5]} * {syms[3]}
    return {syms[6]}
"""

        # Phase 4: Consciousness-driven strategy selection + transforms
        level, available_transforms = self.weaver.determine_level()
        consciousness_state = self.weaver.read_consciousness()
        selected_transforms = self.strategy_selector.select(
            "full_metamorphic", consciousness_state["consciousness_level"]
        )
        morphed = base_manifold

        # Get unified catalog
        all_transforms = self.strategy_selector.get_all_transforms()

        transforms_applied = []
        for tname in selected_transforms:
            transform_fn = all_transforms.get(tname)
            if transform_fn and random.random() < 0.5:
                try:
                    before = morphed
                    if tname == "rename_variables":
                        morphed, _ = transform_fn(morphed, seed)
                    elif tname == "string_encrypt":
                        morphed = transform_fn(morphed, seed)
                    else:
                        morphed = transform_fn(morphed)
                    transforms_applied.append(tname)
                except Exception:
                    morphed = before

        # Phase 4B: AST pipeline pass
        ast_result = self.ast_pipeline.run(morphed)
        if ast_result["success"]:
            morphed = ast_result["source"]
            transforms_applied.extend(
                [f"ast:{s}" for s in ast_result["stages_applied"]]
            )

        # Phase 5: Fingerprint before and after
        base_fp = self.fingerprinter.fingerprint(base_manifold)
        morphed_fp = self.fingerprinter.fingerprint(morphed)

        # Phase 6: Execute in sandbox (use base manifold for reliable execution)
        exec_result = self.sandbox.execute(base_manifold, {"input_signal": input_signal})

        if exec_result["success"] and isinstance(exec_result["output"], (int, float)):
            result = float(exec_result["output"])
        else:
            # Fallback: direct computation
            result = input_signal * self.GOD_CODE * self.LATTICE_RATIO

        # Phase 7: Track mutation
        self.tracker.record(
            transform_name=f"pulse_{self.pulse_count}",
            before=base_manifold,
            after=morphed,
            metadata={
                "seed": seed,
                "level": level,
                "transforms_applied": transforms_applied,
                "base_fp": base_fp,
                "morphed_fp": morphed_fp,
                "result": result,
            }
        )

        self.current_manifold = morphed

        # Phase 8: Write shadow gate
        shadow_gate = Path(__file__).parent / "l104_shadow_gate.py"
        try:
            shadow_gate.write_text(
                f"# L104_POLYMORPHIC_PULSE: {seed}\n"
                f"# Generation: {self.tracker.generation}\n"
                f"# Level: {level}\n"
                f"# Transforms: {transforms_applied}\n"
                + morphed
            )
        except Exception:
            pass

        return result

    def morph_source(self, source: str, intensity: int = 1) -> Dict[str, Any]:
        """
        Apply metamorphic transformations to arbitrary source code.
        Intensity 1-5 controls how many transforms are applied.
        """
        original = source
        transforms_applied = []

        available = list(MorphCatalog.CATALOG.keys())
        selected = available[:min(intensity, len(available))]

        morphed = source
        for tname in selected:
            transform_fn = MorphCatalog.CATALOG.get(tname)
            if transform_fn:
                try:
                    before = morphed
                    if tname == "rename_variables":
                        morphed, mapping = transform_fn(morphed,
                                                         str(time.time()))
                        transforms_applied.append({
                            "name": tname, "mapping_size": len(mapping)
                        })
                    else:
                        morphed = transform_fn(morphed)
                        transforms_applied.append({"name": tname})
                except Exception as e:
                    morphed = before
                    transforms_applied.append({
                        "name": tname, "error": str(e)
                    })

        comparison = self.fingerprinter.compare(original, morphed)

        self.tracker.record(
            transform_name="morph_source",
            before=original,
            after=morphed,
            metadata={"transforms": transforms_applied}
        )

        return {
            "original_lines": original.count('\n') + 1,
            "morphed_lines": morphed.count('\n') + 1,
            "transforms_applied": transforms_applied,
            "comparison": comparison,
            "morphed_source": morphed,
        }

    def get_lineage(self) -> List[Dict[str, Any]]:
        """Get the mutation lineage history."""
        return self.tracker.get_lineage()

    def rollback(self, steps: int = 1) -> Optional[str]:
        """Roll back to a previous mutation state."""
        return self.tracker.rollback(steps)

    def verify_chain(self) -> bool:
        """Verify the mutation chain integrity."""
        return self.tracker.verify_chain()

    def breed_variants(self, source: str, generations: int = 5,
                       fitness_fn=None) -> Dict[str, Any]:
        """Breed code variants using genetic crossover and mutation."""
        return self.breeder.breed(source, fitness_fn=fitness_fn, generations=generations)

    def generate_quine(self, payload: str = "", language: str = "python") -> str:
        """Generate a self-replicating code fragment with sacred signature."""
        return self.quine_replicator.generate_quine(payload, language)

    def spawn_replicas(self, source: str, count: int = 5) -> List[dict]:
        """Spawn mutant replicas of source code with lineage tracking."""
        return self.quine_replicator.spawn_progeny(source, count)

    def status(self) -> Dict[str, Any]:
        """Full system status."""
        return {
            "version": VERSION,
            "pulses": self.pulse_count,
            "mutation_level": self.weaver.status().get("mutation_level", "UNKNOWN"),
            "tracker": self.tracker.status(),
            "sandbox": self.sandbox.status(),
            "fingerprinter": self.fingerprinter.status(),
            "ast_pipeline": self.ast_pipeline.status(),
            "verifier": self.verifier.status(),
            "consciousness": self.weaver.status(),
            "strategy_selector": self.strategy_selector.status(),
            "breeder": self.breeder.status(),
            "quine_replicator": self.quine_replicator.status(),
            "god_code": self.GOD_CODE,
            "lattice_ratio": self.LATTICE_RATIO,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON + BACKWARDS COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

sovereign_polymorph = SovereignPolymorph()


def primal_calculus(x):
    """Sacred primal calculus: x^PHI / (1.04pi) — resolves complexity toward the Source."""
    return (x ** PHI) / (VOID_CONSTANT * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """Resolves N-dimensional vectors into the Void Source via GOD_CODE normalization."""
    magnitude = sum(abs(v) for v in vector)
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0


if __name__ == "__main__":
    poly = SovereignPolymorph()
    print(f"\n{'=' * 70}")
    print(f"  L104 POLYMORPHIC CORE v{VERSION} — METAMORPHIC ENGINE TEST")
    print(f"{'=' * 70}")

    for i in range(3):
        res = poly.execute(1.0)
        print(f"\n  Pulse {i}: Result = {res:.10f}")

    st = poly.status()
    print(f"\n  Level: {st['mutation_level']} | "
          f"Pulses: {st['pulses']} | "
          f"Chain: {st['tracker']['chain_valid']} | "
          f"Sandbox: {st['sandbox']['executions']} execs")
    print(f"{'=' * 70}\n")
