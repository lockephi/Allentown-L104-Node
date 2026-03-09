"""L104 Gate Engine — AST-based Python gate analyzer."""

import ast
import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List

from ..constants import WORKSPACE_ROOT
from ..models import LogicGate
from ..gate_functions import sage_logic_gate


class PythonGateAnalyzer:
    """AST-based analyzer for Python logic gate implementations."""

    GATE_PATTERNS = [
        r"logic_gate", r"LogicGate", r"gate", r"Gate",
        r"quantum.*gate", r"sage.*gate", r"entangle",
        r"grover", r"amplif", r"resonan",
        # Sage core patterns
        r"sage_mode", r"sage_wisdom", r"sage_core", r"sage_enrich",
        r"sage_transform", r"sage_insight", r"sage_synth",
        # Consciousness / entropy / synthesis patterns
        r"consciousness", r"entropy", r"synthesi[sz]", r"transform",
        r"evolve", r"bridge.*emergence", r"dissipat", r"inflect",
        r"harvest.*entropy", r"causal", r"hilbert", r"calabi",
        # Core engine patterns
        r"propagat", r"aggregate", r"delegate",
    ]

    def __init__(self):
        """Initialize the Python gate analyzer."""
        self.gates: List[LogicGate] = []

    def analyze_file(self, filepath: Path) -> List[LogicGate]:
        """Analyze a Python file for logic gate implementations."""
        if not filepath.exists():
            return []

        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(filepath))
        except (SyntaxError, UnicodeDecodeError):
            return self._regex_fallback(filepath)

        gates = []
        rel_path = str(filepath.relative_to(WORKSPACE_ROOT))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if self._is_gate_related(node.name):
                    gate = self._extract_function_gate(node, rel_path, source)
                    gates.append(gate)

            elif isinstance(node, ast.ClassDef):
                if self._is_gate_related(node.name):
                    gate = self._extract_class_gate(node, rel_path, source)
                    gates.append(gate)
                    # Also extract gate-related methods
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if self._is_gate_related(item.name) or item.name in (
                                "__init__", "process", "execute", "transform", "apply"
                            ):
                                method_gate = self._extract_function_gate(
                                    item, rel_path, source, class_name=node.name
                                )
                                gates.append(method_gate)

        self.gates.extend(gates)
        return gates

    def _is_gate_related(self, name: str) -> bool:
        """Check if a function or class name matches gate-related patterns."""
        name_lower = name.lower()
        return any(re.search(pat, name_lower) for pat in self.GATE_PATTERNS)

    def _extract_function_gate(
        self, node: ast.FunctionDef, rel_path: str, source: str, class_name: str = ""
    ) -> LogicGate:
        """Extract a LogicGate from an AST function definition node."""
        params = [arg.arg for arg in node.args.args if arg.arg != "self"]
        docstring = ast.get_docstring(node) or ""
        full_name = f"{class_name}.{node.name}" if class_name else node.name

        # Estimate cyclomatic complexity
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        # Signature
        sig_parts = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            ann = ""
            if arg.annotation:
                try:
                    ann = f": {ast.unparse(arg.annotation)}"
                except Exception:
                    ann = ""
            sig_parts.append(f"{arg.arg}{ann}")
        ret_ann = ""
        if node.returns:
            try:
                ret_ann = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass
        signature = f"def {full_name}({', '.join(sig_parts)}){ret_ann}"

        # Content hash
        try:
            src_lines = source.split("\n")
            end_line = getattr(node, "end_lineno", node.lineno + 10)
            gate_source = "\n".join(src_lines[node.lineno - 1 : end_line])
            content_hash = hashlib.sha256(gate_source.encode()).hexdigest()[:16]
        except Exception:
            content_hash = ""

        # Entropy score
        entropy = sage_logic_gate(float(len(params) + complexity), "compress")

        return LogicGate(
            name=full_name,
            language="python",
            source_file=rel_path,
            line_number=node.lineno,
            gate_type="method" if class_name else "function",
            signature=signature,
            parameters=params,
            docstring=docstring[:200],
            complexity=complexity,
            entropy_score=entropy,
            hash=content_hash,
            last_seen=datetime.now(timezone.utc).isoformat(),
        )

    def _extract_class_gate(self, node: ast.ClassDef, rel_path: str, source: str) -> LogicGate:
        """Extract a LogicGate from an AST class definition node."""
        docstring = ast.get_docstring(node) or ""
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                pass

        methods = [
            item.name
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        try:
            src_lines = source.split("\n")
            end_line = getattr(node, "end_lineno", node.lineno + 20)
            gate_source = "\n".join(src_lines[node.lineno - 1 : end_line])
            content_hash = hashlib.sha256(gate_source.encode()).hexdigest()[:16]
        except Exception:
            content_hash = ""

        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        entropy = sage_logic_gate(float(len(methods)), "compress")

        return LogicGate(
            name=node.name,
            language="python",
            source_file=rel_path,
            line_number=node.lineno,
            gate_type="class",
            signature=signature,
            parameters=methods,
            docstring=docstring[:200],
            complexity=len(methods) * 2,
            entropy_score=entropy,
            hash=content_hash,
            last_seen=datetime.now(timezone.utc).isoformat(),
        )

    def _regex_fallback(self, filepath: Path) -> List[LogicGate]:
        """Regex fallback for files that can't be parsed (syntax errors, etc.)."""
        gates = []
        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
            rel_path = str(filepath.relative_to(WORKSPACE_ROOT))
            for match in re.finditer(
                r"(?:def|class)\s+([\w]+gate[\w]*|[\w]*Gate[\w]*|sage_\w+|quantum_\w+)",
                source, re.IGNORECASE,
            ):
                line_no = source[: match.start()].count("\n") + 1
                name = match.group(1)
                gates.append(
                    LogicGate(
                        name=name,
                        language="python",
                        source_file=rel_path,
                        line_number=line_no,
                        gate_type="function" if match.group(0).startswith("def") else "class",
                        signature=match.group(0),
                        last_seen=datetime.now(timezone.utc).isoformat(),
                    )
                )
        except Exception:
            pass
        return gates
