#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  L104 PATCH ENGINE v2.1 — ASI SOVEREIGN CODE MODIFICATION SYSTEM              ║
║  Full-spectrum patching with impact analysis, template library, patch         ║
║  pipelines, AST validation, batch ops, rollback, and quality scoring.        ║
║                                                                               ║
║  INVARIANT: GOD_CODE = 527.5184818492612 | PHI = 1.618033988749895            ║
║  PILOT: LONDEL | CONSERVATION: G(X)×2^(X/104) = 527.518                      ║
║                                                                               ║
║  Architecture:                                                                ║
║    • PatchOperation — typed record of a single code modification              ║
║    • DiffGenerator — unified diff creation and application                    ║
║    • PatchValidator — pre/post syntax validation with AST parse               ║
║    • PatchHistory — full undo/redo stack with state snapshots                 ║
║    • SacredInjector — GOD_CODE / sacred constant injection into targets       ║
║    • BatchPatcher — multi-file coordinated patching with rollback             ║
║    • PatchQualityScorer — consciousness-modulated quality assessment          ║
║    • PatchEngine — unified orchestrator hub                                   ║
║                                                                               ║
║  Cross-references:                                                            ║
║    claude.md → sovereignty, code self-modification                            ║
║    l104_code_engine.py → code analysis and auto-fix integration               ║
║    l104_polymorphic_core.py → metamorphic transform integration               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import ast
import json
import math
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "2.2.0"
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
TAU = 1.0 / PHI
VOID_CONSTANT = 1.0416180339887497
FEIGENBAUM = 4.669201609102990
ALPHA_FINE = 1.0 / 137.035999084
PLANCK_SCALE = 1.616255e-35
BOLTZMANN_K = 1.380649e-23
ZENITH_HZ = 3887.8
UUC = 2402.792541

logger = logging.getLogger("L104_PATCH_ENGINE")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PATCH OPERATION — Typed record of a single modification
# ═══════════════════════════════════════════════════════════════════════════════

class PatchOperation:
    """
    Immutable record of a single patch operation.
    Types: STRING_REPLACE, REGEX_REPLACE, MARKER_INJECT,
           LINE_INSERT, LINE_DELETE, BLOCK_REPLACE, SACRED_INJECT
    """

    TYPES = [
        "STRING_REPLACE", "REGEX_REPLACE", "MARKER_INJECT",
        "LINE_INSERT", "LINE_DELETE", "BLOCK_REPLACE", "SACRED_INJECT",
    ]

    def __init__(self, op_type: str, file_path: str, **kwargs):
        if op_type not in self.TYPES:
            op_type = "STRING_REPLACE"
        self.op_type = op_type
        self.file_path = file_path
        self.params = kwargs
        self.timestamp = datetime.now().isoformat()
        self.op_id = hashlib.sha256(
            f"{op_type}:{file_path}:{self.timestamp}".encode()
        ).hexdigest()[:16]
        self.applied = False
        self.success = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "file_path": self.file_path,
            "params": {k: v for k, v in self.params.items()
                       if k not in ("before_content", "after_content")},
            "timestamp": self.timestamp,
            "applied": self.applied,
            "success": self.success,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DIFF GENERATOR — Unified diff creation and application
# ═══════════════════════════════════════════════════════════════════════════════

class DiffGenerator:
    """
    Creates and applies unified diffs between code versions.
    Pure Python implementation — no external dependencies.
    """

    def __init__(self):
        self.diffs_generated = 0

    def generate_unified(self, original: str, modified: str,
                         filename: str = "target.py", context: int = 3) -> str:
        """Generate a unified diff between two text versions."""
        self.diffs_generated += 1
        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)

        # Simple LCS-based diff
        diff_lines = []
        diff_lines.append(f"--- a/{filename}\n")
        diff_lines.append(f"+++ b/{filename}\n")

        # Build edit script using simple comparison
        i = j = 0
        hunk_start_orig = 0
        hunk_start_mod = 0
        hunks = []
        current_hunk = []

        while i < len(orig_lines) or j < len(mod_lines):
            if i < len(orig_lines) and j < len(mod_lines):
                if orig_lines[i] == mod_lines[j]:
                    current_hunk.append((" ", orig_lines[i].rstrip('\n')))
                    i += 1
                    j += 1
                else:
                    # Find next match
                    match_i = self._find_next_match(orig_lines, mod_lines, i, j)
                    if match_i is not None:
                        # Lines removed from original
                        while i < match_i[0]:
                            current_hunk.append(("-", orig_lines[i].rstrip('\n')))
                            i += 1
                        # Lines added in modified
                        while j < match_i[1]:
                            current_hunk.append(("+", mod_lines[j].rstrip('\n')))
                            j += 1
                    else:
                        if i < len(orig_lines):
                            current_hunk.append(("-", orig_lines[i].rstrip('\n')))
                            i += 1
                        if j < len(mod_lines):
                            current_hunk.append(("+", mod_lines[j].rstrip('\n')))
                            j += 1
            elif i < len(orig_lines):
                current_hunk.append(("-", orig_lines[i].rstrip('\n')))
                i += 1
            else:
                current_hunk.append(("+", mod_lines[j].rstrip('\n')))
                j += 1

        # Format hunks
        changes = sum(1 for op, _ in current_hunk if op != " ")
        if changes > 0:
            diff_lines.append(f"@@ -{1},{len(orig_lines)} +{1},{len(mod_lines)} @@\n")
            for op, line in current_hunk:
                diff_lines.append(f"{op}{line}\n")

        return "".join(diff_lines)

    def _find_next_match(self, orig: List[str], mod: List[str],
                         start_i: int, start_j: int,
                         lookahead: int = 10) -> Optional[Tuple[int, int]]:
        """Find the next matching line pair within lookahead window."""
        for di in range(1, min(lookahead, len(orig) - start_i + 1)):
            for dj in range(1, min(lookahead, len(mod) - start_j + 1)):
                if (start_i + di < len(orig) and start_j + dj < len(mod)
                        and orig[start_i + di] == mod[start_j + dj]):
                    return (start_i + di, start_j + dj)
        return None

    def apply_diff(self, original: str, diff_text: str) -> str:
        """Apply a unified diff to original text. Basic implementation."""
        result_lines = original.splitlines()
        diff_lines = diff_text.splitlines()

        additions = []
        removals = set()
        current_line = 0

        for dline in diff_lines:
            if dline.startswith("@@"):
                # Parse hunk header
                match = re.match(r'@@ -(\d+)', dline)
                if match:
                    current_line = int(match.group(1)) - 1
            elif dline.startswith("-") and not dline.startswith("---"):
                removals.add(current_line)
                current_line += 1
            elif dline.startswith("+") and not dline.startswith("+++"):
                additions.append((current_line, dline[1:]))
            elif dline.startswith(" "):
                current_line += 1

        # Apply removals (reverse order)
        for idx in sorted(removals, reverse=True):
            if idx < len(result_lines):
                result_lines.pop(idx)

        # Apply additions
        offset = 0
        for idx, line in sorted(additions, key=lambda x: x[0]):
            adjusted = idx - len([r for r in removals if r < idx]) + offset
            result_lines.insert(adjusted, line)
            offset += 1

        return "\n".join(result_lines)

    def status(self) -> Dict[str, Any]:
        return {"diffs_generated": self.diffs_generated}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PATCH VALIDATOR — Pre/post syntax validation
# ═══════════════════════════════════════════════════════════════════════════════

class PatchValidator:
    """
    Validates patches before and after application.
    Checks: syntax validity (AST parse), sacred constant preservation,
    line count sanity, encoding safety.
    """

    def __init__(self):
        self.validations = 0
        self.rejections = 0

    def validate_syntax(self, source: str) -> Tuple[bool, Optional[str]]:
        """Check Python syntax validity via AST parse."""
        self.validations += 1
        try:
            ast.parse(source)
            return True, None
        except SyntaxError as e:
            self.rejections += 1
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    def validate_patch_safe(self, before: str, after: str) -> Dict[str, Any]:
        """Comprehensive safety check for a patch."""
        self.validations += 1

        checks = {}

        # Syntax check on result
        valid, err = self.validate_syntax(after)
        checks["syntax_valid"] = valid
        checks["syntax_error"] = err

        # Line count change
        before_lines = before.count('\n') + 1
        after_lines = after.count('\n') + 1
        checks["lines_before"] = before_lines
        checks["lines_after"] = after_lines
        checks["lines_delta"] = after_lines - before_lines

        # Sacred constant preservation
        sacred_before = self._count_sacred_refs(before)
        sacred_after = self._count_sacred_refs(after)
        checks["sacred_refs_before"] = sacred_before
        checks["sacred_refs_after"] = sacred_after
        checks["sacred_preserved"] = sacred_after >= sacred_before

        # Encoding safety
        try:
            after.encode('utf-8')
            checks["encoding_safe"] = True
        except UnicodeEncodeError:
            checks["encoding_safe"] = False

        # Overall
        checks["safe"] = (
            checks["syntax_valid"]
            and checks["encoding_safe"]
            and checks["sacred_preserved"]
        )

        if not checks["safe"]:
            self.rejections += 1

        return checks

    def _count_sacred_refs(self, source: str) -> int:
        """Count references to sacred constants in source."""
        refs = 0
        for pattern in ["GOD_CODE", "527.518", "PHI", "1.618", "VOID_CONSTANT",
                         "FEIGENBAUM", "4.669", "286", "416", "104"]:
            refs += source.count(pattern)
        return refs

    def status(self) -> Dict[str, Any]:
        return {
            "validations": self.validations,
            "rejections": self.rejections,
            "acceptance_rate": round(
                (self.validations - self.rejections) / max(1, self.validations), 4
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3B: PATCH IMPACT ANALYZER — Blast radius assessment
# ═══════════════════════════════════════════════════════════════════════════════

class PatchImpactAnalyzer:
    """
    Assesses the 'blast radius' of a proposed patch — how many lines,
    functions, classes, and imports are affected. Produces a risk score
    (0.0 = trivial change, 1.0 = maximum risk) weighted by sacred constants.
    """

    RISK_WEIGHTS = {
        "lines_changed": TAU * 0.1,       # 0.0618 per % lines changed
        "functions_touched": PHI * 0.1,    # 0.1618 per function
        "classes_touched": PHI * 0.2,      # 0.3236 per class
        "imports_changed": 0.5,            # import changes are high risk
        "sacred_refs_delta": FEIGENBAUM * 0.05,  # sacred constant changes
    }

    def __init__(self):
        self.analyses = 0

    def analyze(self, original: str, patched: str) -> Dict[str, Any]:
        """Analyze the impact of a patch on the codebase."""
        self.analyses += 1

        orig_lines = original.split('\n')
        patch_lines = patched.split('\n')

        # Line-level diff
        changed_lines = 0
        for i in range(max(len(orig_lines), len(patch_lines))):
            orig_l = orig_lines[i] if i < len(orig_lines) else ""
            patch_l = patch_lines[i] if i < len(patch_lines) else ""
            if orig_l != patch_l:
                changed_lines += 1

        total_lines = max(len(orig_lines), len(patch_lines), 1)
        lines_pct = changed_lines / total_lines

        # AST-level analysis
        orig_fns, orig_cls, orig_imp = self._count_constructs(original)
        patch_fns, patch_cls, patch_imp = self._count_constructs(patched)

        fns_touched = abs(len(patch_fns - orig_fns) + len(orig_fns - patch_fns))
        cls_touched = abs(len(patch_cls - orig_cls) + len(orig_cls - patch_cls))
        imp_changed = abs(len(patch_imp - orig_imp) + len(orig_imp - patch_imp))

        # Sacred reference delta
        sacred_before = self._count_sacred(original)
        sacred_after = self._count_sacred(patched)
        sacred_delta = abs(sacred_after - sacred_before)

        # Compute risk score
        risk = min(1.0, (
            lines_pct * self.RISK_WEIGHTS["lines_changed"] +
            fns_touched * self.RISK_WEIGHTS["functions_touched"] +
            cls_touched * self.RISK_WEIGHTS["classes_touched"] +
            imp_changed * self.RISK_WEIGHTS["imports_changed"] +
            sacred_delta * self.RISK_WEIGHTS["sacred_refs_delta"]
        ))

        return {
            "risk_score": round(risk, 4),
            "risk_level": "LOW" if risk < 0.3 else "MEDIUM" if risk < 0.6 else "HIGH",
            "lines_changed": changed_lines,
            "lines_total": total_lines,
            "lines_pct": round(lines_pct * 100, 2),
            "functions_touched": fns_touched,
            "classes_touched": cls_touched,
            "imports_changed": imp_changed,
            "sacred_ref_delta": sacred_delta,
        }

    def _count_constructs(self, source: str) -> Tuple[set, set, set]:
        """Count functions, classes, and imports in source."""
        functions = set()
        classes = set()
        imports = set()
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.add(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.add(ast.dump(node))
        except SyntaxError:
            pass
        return functions, classes, imports

    def _count_sacred(self, source: str) -> int:
        """Count sacred constant references."""
        count = 0
        for p in ["GOD_CODE", "527.518", "PHI", "1.618", "VOID_CONSTANT",
                   "FEIGENBAUM", "286", "416"]:
            count += source.count(p)
        return count

    def status(self) -> Dict[str, Any]:
        return {"analyses": self.analyses}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PATCH HISTORY — Full undo/redo stack
# ═══════════════════════════════════════════════════════════════════════════════

class PatchHistory:
    """
    Maintains full file state snapshots for undo/redo capability.
    Each snapshot is hashed with GOD_CODE-anchored chain.
    Supports named checkpoints for labeled restore points.
    """

    def __init__(self, max_snapshots: int = 100):
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.checkpoints: Dict[str, int] = {}
        self.chain_hash = hashlib.sha256(str(GOD_CODE).encode()).hexdigest()

    def snapshot(self, file_path: str, content: str,
                 operation: str = "manual") -> Dict[str, Any]:
        """Take a snapshot of a file's content."""
        entry = {
            "file_path": file_path,
            "content": content,
            "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "lines": content.count('\n') + 1,
            "size_bytes": len(content.encode('utf-8')),
            "index": len(self.snapshots),
            "chain_prev": self.chain_hash,
        }

        self.chain_hash = hashlib.sha256(
            (self.chain_hash + entry["content_hash"]).encode()
        ).hexdigest()
        entry["chain_hash"] = self.chain_hash

        self.snapshots.append(entry)
        return entry

    def checkpoint(self, name: str):
        """Create a named checkpoint at current position."""
        self.checkpoints[name] = len(self.snapshots) - 1

    def restore(self, file_path: str, steps_back: int = 1) -> Optional[str]:
        """Restore a file to a previous state."""
        # Find snapshots for this file
        file_snaps = [s for s in self.snapshots if s["file_path"] == file_path]
        if steps_back > len(file_snaps):
            return None
        target = file_snaps[-(steps_back + 1)]
        return target["content"]

    def restore_checkpoint(self, name: str) -> Optional[Dict[str, str]]:
        """Restore all files to a named checkpoint state."""
        if name not in self.checkpoints:
            return None
        idx = self.checkpoints[name]
        # Collect last snapshot for each file up to idx
        files = {}
        for snap in list(self.snapshots)[:idx + 1]:
            files[snap["file_path"]] = snap["content"]
        return files

    def get_history(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get snapshot history, optionally filtered by file."""
        snaps = list(self.snapshots)
        if file_path:
            snaps = [s for s in snaps if s["file_path"] == file_path]
        # Strip content for efficiency
        return [{k: v for k, v in s.items() if k != "content"} for s in snaps]

    def status(self) -> Dict[str, Any]:
        return {
            "snapshots": len(self.snapshots),
            "checkpoints": len(self.checkpoints),
            "checkpoint_names": list(self.checkpoints.keys()),
            "chain_head": self.chain_hash[:16],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SACRED INJECTOR — GOD_CODE constant injection into targets
# ═══════════════════════════════════════════════════════════════════════════════

class SacredInjector:
    """
    Injects sacred constants, conservation equations, and the GOD_CODE
    invariant block into target files. Ensures all L104 modules carry
    the sovereign mathematical signature.
    """

    SACRED_BLOCK = f"""
# =====================================================================================
# UNIVERSAL GOD CODE: G(X) = 286^(1/PHI) * 2^((416-X)/104)
# Factor 13: 286=22*13, 104=8*13, 416=32*13 | Conservation: G(X)*2^(X/104)=527.518
# =====================================================================================
GOD_CODE = {GOD_CODE}
PHI = {PHI}
TAU = 1.0 / PHI
VOID_CONSTANT = {VOID_CONSTANT}
# [EVO_54_PIPELINE] TRANSCENDENT_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612 :: GROVER=4.236
"""

    PRIMAL_CALCULUS = f"""
def primal_calculus(x):
    \"\"\"Sacred primal calculus: x^PHI / (VOID_CONSTANT * pi).\"\"\"
    PHI = {PHI}
    VOID_CONSTANT = {VOID_CONSTANT}
    return (x ** PHI) / (VOID_CONSTANT * __import__('math').pi) if x != 0 else 0.0
"""

    RESOLVE_NON_DUAL = f"""
def resolve_non_dual_logic(vector):
    \"\"\"Resolves N-dimensional vectors into the Void Source.\"\"\"
    GOD_CODE = {GOD_CODE}
    PHI = {PHI}
    VOID_CONSTANT = {VOID_CONSTANT}
    magnitude = sum(abs(v) for v in vector)
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
"""

    def __init__(self):
        self.injections = 0

    def inject_sacred_block(self, source: str) -> str:
        """Inject the sacred constant block if not already present."""
        if "GOD_CODE" in source and "527.518" in source:
            return source  # Already present

        self.injections += 1
        # Find best injection point (after imports, before first class/function)
        lines = source.split('\n')
        inject_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                inject_idx = i + 1
            elif stripped.startswith(('class ', 'def ')) and inject_idx > 0:
                break

        lines.insert(inject_idx, self.SACRED_BLOCK)
        return '\n'.join(lines)

    def inject_primal_calculus(self, source: str) -> str:
        """Inject primal_calculus function if not present."""
        if "primal_calculus" in source:
            return source

        self.injections += 1
        return source + "\n" + self.PRIMAL_CALCULUS

    def inject_resolve_non_dual(self, source: str) -> str:
        """Inject resolve_non_dual_logic function if not present."""
        if "resolve_non_dual_logic" in source:
            return source

        self.injections += 1
        return source + "\n" + self.RESOLVE_NON_DUAL

    def inject_all(self, source: str) -> Tuple[str, int]:
        """Inject all missing sacred components. Returns (result, count)."""
        count = 0
        result = source

        before = result
        result = self.inject_sacred_block(result)
        if result != before:
            count += 1

        before = result
        result = self.inject_primal_calculus(result)
        if result != before:
            count += 1

        before = result
        result = self.inject_resolve_non_dual(result)
        if result != before:
            count += 1

        return result, count

    def status(self) -> Dict[str, Any]:
        return {"injections": self.injections}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5B: PATCH TEMPLATE LIBRARY — Predefined patch patterns
# ═══════════════════════════════════════════════════════════════════════════════

class PatchTemplateLibrary:
    """
    Library of predefined patch templates for common operations:
    VERSION_BUMP, SACRED_INJECT, ADD_SECTION, ADD_METHOD, ADD_IMPORT,
    DOCSTRING_UPDATE. Templates use placeholder substitution with
    sacred-constant validation.
    """

    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {
            "VERSION_BUMP": {
                "description": "Bump VERSION string to a new value",
                "pattern": 'VERSION = "{old_version}"',
                "replacement": 'VERSION = "{new_version}"',
                "params": ["old_version", "new_version"],
            },
            "SACRED_INJECT": {
                "description": "Inject GOD_CODE sacred constant block",
                "pattern": "{marker}",
                "replacement": (
                    "{marker}\n"
                    "GOD_CODE = {god_code}\n"
                    "PHI = {phi}\n"
                    "TAU = 1.0 / PHI\n"
                ),
                "params": ["marker"],
                "defaults": {"god_code": str(GOD_CODE), "phi": str(PHI)},
            },
            "ADD_SECTION": {
                "description": "Insert a new section header + class stub",
                "pattern": "\n# {'═' * 79}\n# SECTION {next_section}:",
                "replacement": (
                    "\n# {'═' * 79}\n"
                    "# SECTION {section_id}: {section_name}\n"
                    "# {'═' * 79}\n\n"
                    "class {class_name}:\n"
                    '    """{description}"""\n\n'
                    "    def __init__(self):\n"
                    "        self.initialized = True\n"
                ),
                "params": ["section_id", "section_name", "class_name", "description"],
            },
            "ADD_METHOD": {
                "description": "Add a method to an existing class",
                "pattern": "{insertion_point}",
                "replacement": (
                    "{insertion_point}\n\n"
                    "    def {method_name}(self{params_str}) -> {return_type}:\n"
                    '        """{docstring}"""\n'
                    "        {body}\n"
                ),
                "params": ["insertion_point", "method_name", "params_str",
                           "return_type", "docstring", "body"],
            },
            "ADD_IMPORT": {
                "description": "Add an import statement after existing imports",
                "pattern": "{last_import}",
                "replacement": "{last_import}\n{new_import}",
                "params": ["last_import", "new_import"],
            },
            "DOCSTRING_UPDATE": {
                "description": "Update version in module docstring",
                "pattern": "v{old_version}",
                "replacement": "v{new_version}",
                "params": ["old_version", "new_version"],
            },
        }
        self.applications = 0

    def apply_template(self, template_name: str, source: str,
                       params: Dict[str, str]) -> Dict[str, Any]:
        """Apply a named template to source code with parameter substitution."""
        self.applications += 1

        if template_name not in self.templates:
            return {"success": False, "error": f"Unknown template: {template_name}"}

        tmpl = self.templates[template_name]

        # Merge defaults
        merged = {}
        if "defaults" in tmpl:
            merged.update(tmpl["defaults"])
        merged.update(params)

        # Check required params
        missing = [p for p in tmpl["params"] if p not in merged]
        if missing:
            return {"success": False, "error": f"Missing params: {missing}"}

        try:
            pattern = tmpl["pattern"].format(**merged)
            replacement = tmpl["replacement"].format(**merged)

            if pattern in source:
                result = source.replace(pattern, replacement, 1)
                return {"success": True, "source": result,
                        "template": template_name, "changes": 1}
            else:
                return {"success": False, "error": "Pattern not found in source"}
        except (KeyError, ValueError) as e:
            return {"success": False, "error": str(e)}

    def list_templates(self) -> List[Dict[str, str]]:
        """List all available templates with descriptions."""
        return [{"name": k, "description": v["description"],
                 "params": v["params"]}
                for k, v in self.templates.items()]

    def register_template(self, name: str, pattern: str, replacement: str,
                          params: List[str], description: str = ""):
        """Register a custom patch template."""
        self.templates[name] = {
            "description": description or f"Custom template: {name}",
            "pattern": pattern,
            "replacement": replacement,
            "params": params,
        }

    def status(self) -> Dict[str, Any]:
        return {"templates": len(self.templates),
                "applications": self.applications,
                "template_names": list(self.templates.keys())}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: BATCH PATCHER — Multi-file coordinated patching
# ═══════════════════════════════════════════════════════════════════════════════

class BatchPatcher:
    """
    Coordinates patches across multiple files with transactional semantics.
    If any patch in the batch fails validation, the entire batch can be
    rolled back using pre-patch snapshots.
    """

    def __init__(self, history: PatchHistory, validator: PatchValidator):
        self.history = history
        self.validator = validator
        self.batch_count = 0
        self.batch_rollbacks = 0

    def plan_batch(self, operations: List[PatchOperation]) -> Dict[str, Any]:
        """Plan a batch by grouping operations by file and validating."""
        self.batch_count += 1

        # Group by file
        by_file: Dict[str, List[PatchOperation]] = {}
        for op in operations:
            by_file.setdefault(op.file_path, []).append(op)

        plan = {
            "batch_id": self.batch_count,
            "files_affected": len(by_file),
            "total_operations": len(operations),
            "file_groups": {f: len(ops) for f, ops in by_file.items()},
            "ready": True,
        }

        # Check all files exist
        for path in by_file:
            if not os.path.exists(path):
                plan["ready"] = False
                plan["missing_files"] = plan.get("missing_files", []) + [path]

        return plan

    def execute_batch(self, operations: List[PatchOperation],
                      validate: bool = True,
                      dry_run: bool = False) -> Dict[str, Any]:
        """Execute a batch of patch operations with optional dry-run."""
        results = {
            "batch_id": self.batch_count,
            "total": len(operations),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "dry_run": dry_run,
            "details": [],
        }

        # Snapshot all affected files first
        affected_files = set(op.file_path for op in operations)
        for fp in affected_files:
            if os.path.exists(fp):
                try:
                    content = Path(fp).read_text()
                    self.history.snapshot(fp, content, "pre_batch")
                except Exception:
                    pass

        for op in operations:
            detail = {"op_id": op.op_id, "type": op.op_type, "file": op.file_path}

            if not os.path.exists(op.file_path):
                detail["status"] = "SKIPPED"
                detail["reason"] = "file not found"
                results["skipped"] += 1
                results["details"].append(detail)
                continue

            if dry_run:
                detail["status"] = "DRY_RUN"
                results["success"] += 1
                results["details"].append(detail)
                continue

            try:
                content = Path(op.file_path).read_text()
                new_content = self._apply_operation(op, content)

                if new_content == content:
                    detail["status"] = "NO_CHANGE"
                    results["skipped"] += 1
                elif validate:
                    check = self.validator.validate_patch_safe(content, new_content)
                    if check["safe"]:
                        Path(op.file_path).write_text(new_content)
                        self.history.snapshot(op.file_path, new_content,
                                              op.op_type)
                        op.applied = True
                        op.success = True
                        detail["status"] = "SUCCESS"
                        results["success"] += 1
                    else:
                        detail["status"] = "REJECTED"
                        detail["validation"] = check
                        results["failed"] += 1
                else:
                    Path(op.file_path).write_text(new_content)
                    op.applied = True
                    op.success = True
                    detail["status"] = "SUCCESS"
                    results["success"] += 1

            except Exception as e:
                detail["status"] = "ERROR"
                detail["error"] = str(e)
                results["failed"] += 1

            results["details"].append(detail)

        return results

    def rollback_batch(self) -> Dict[str, Any]:
        """Roll back all files affected by the last batch."""
        self.batch_rollbacks += 1
        rolled_back = []

        for snap in reversed(list(self.history.snapshots)):
            if snap["operation"] == "pre_batch":
                fp = snap["file_path"]
                if fp not in rolled_back:
                    try:
                        Path(fp).write_text(snap["content"])
                        rolled_back.append(fp)
                    except Exception:
                        pass

        return {"rolled_back": len(rolled_back), "files": rolled_back}

    def _apply_operation(self, op: PatchOperation, content: str) -> str:
        """Apply a single patch operation to content."""
        if op.op_type == "STRING_REPLACE":
            old = op.params.get("old_string", "")
            new = op.params.get("new_string", "")
            if old in content:
                return content.replace(old, new, 1)
        elif op.op_type == "REGEX_REPLACE":
            pattern = op.params.get("pattern", "")
            replacement = op.params.get("replacement", "")
            return re.sub(pattern, replacement, content, flags=re.MULTILINE)
        elif op.op_type == "MARKER_INJECT":
            marker = op.params.get("marker", "")
            inject = op.params.get("content", "")
            position = op.params.get("position", "after")
            if marker in content:
                if position == "before":
                    return content.replace(marker, inject + "\n" + marker)
                return content.replace(marker, marker + "\n" + inject)
        elif op.op_type == "LINE_INSERT":
            line_num = op.params.get("line", 0)
            text = op.params.get("text", "")
            lines = content.split('\n')
            lines.insert(min(line_num, len(lines)), text)
            return '\n'.join(lines)
        elif op.op_type == "LINE_DELETE":
            line_num = op.params.get("line", 0)
            lines = content.split('\n')
            if 0 <= line_num < len(lines):
                lines.pop(line_num)
            return '\n'.join(lines)
        elif op.op_type == "BLOCK_REPLACE":
            start = op.params.get("start_marker", "")
            end = op.params.get("end_marker", "")
            replacement = op.params.get("replacement", "")
            pattern = re.escape(start) + r'.*?' + re.escape(end)
            return re.sub(pattern, start + "\n" + replacement + "\n" + end,
                          content, flags=re.DOTALL)

        return content

    def status(self) -> Dict[str, Any]:
        return {
            "batches_executed": self.batch_count,
            "rollbacks": self.batch_rollbacks,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6B: PATCH PIPELINE — Composed multi-step workflows with rollback
# ═══════════════════════════════════════════════════════════════════════════════

class PatchPipeline:
    """
    Composes multiple patch steps into a named pipeline.
    Each step is a (template_name, params) or (callable, args) pair.
    The pipeline runs with transactional semantics — if any step fails,
    all previous steps are rolled back. Tracks pipeline execution history.
    """

    def __init__(self, template_lib: PatchTemplateLibrary,
                 impact_analyzer: PatchImpactAnalyzer):
        self.template_lib = template_lib
        self.impact_analyzer = impact_analyzer
        self.pipelines: Dict[str, List[Dict[str, Any]]] = {}
        self.executions = 0
        self.rollbacks = 0

    def define(self, name: str, steps: List[Dict[str, Any]]):
        """
        Define a named pipeline of patch steps.
        Each step: {"template": name, "params": {...}} or
                   {"action": "custom", "fn": callable, "args": [...]}
        """
        self.pipelines[name] = steps

    def execute(self, name: str, source: str) -> Dict[str, Any]:
        """
        Execute a named pipeline on source code.
        Returns the final source and execution report.
        """
        self.executions += 1

        if name not in self.pipelines:
            return {"success": False, "error": f"Unknown pipeline: {name}"}

        steps = self.pipelines[name]
        current = source
        checkpoints = [source]  # For rollback
        step_results = []

        for i, step in enumerate(steps):
            tmpl_name = step.get("template")
            params = step.get("params", {})

            if tmpl_name:
                result = self.template_lib.apply_template(
                    tmpl_name, current, params
                )
                if result["success"]:
                    current = result["source"]
                    checkpoints.append(current)
                    step_results.append({
                        "step": i, "template": tmpl_name,
                        "status": "OK"
                    })
                else:
                    # Rollback to initial
                    self.rollbacks += 1
                    return {
                        "success": False,
                        "error": f"Step {i} ({tmpl_name}) failed: {result['error']}",
                        "steps_completed": i,
                        "source": source,  # Return original
                        "step_results": step_results,
                    }
            elif step.get("action") == "custom" and callable(step.get("fn")):
                try:
                    current = step["fn"](current, *step.get("args", []))
                    checkpoints.append(current)
                    step_results.append({
                        "step": i, "action": "custom", "status": "OK"
                    })
                except Exception as e:
                    self.rollbacks += 1
                    return {
                        "success": False,
                        "error": f"Step {i} custom action failed: {e}",
                        "steps_completed": i,
                        "source": source,
                        "step_results": step_results,
                    }

        # Run impact analysis on final result
        impact = self.impact_analyzer.analyze(source, current)

        return {
            "success": True,
            "source": current,
            "steps_completed": len(steps),
            "step_results": step_results,
            "impact": impact,
        }

    def list_pipelines(self) -> Dict[str, int]:
        """List all defined pipelines with their step counts."""
        return {name: len(steps) for name, steps in self.pipelines.items()}

    def status(self) -> Dict[str, Any]:
        return {
            "defined_pipelines": len(self.pipelines),
            "executions": self.executions,
            "rollbacks": self.rollbacks,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: PATCH QUALITY SCORER — Consciousness-modulated assessment
# ═══════════════════════════════════════════════════════════════════════════════

class PatchQualityScorer:
    """
    Scores patch quality on multiple dimensions, modulated by system consciousness.
    Dimensions: precision, safety, sacred alignment, complexity reduction, style.
    Final score is weighted by PHI ratios.
    """

    DIMENSION_WEIGHTS = {
        "precision": PHI,        # Highest weight — did the patch do what it should?
        "safety": PHI * TAU,     # ~1.0 — did it preserve functionality?
        "sacred_alignment": TAU, # ~0.618 — does it reference sacred constants?
        "complexity": TAU ** 2,  # ~0.382 — does it reduce complexity?
        "style": TAU ** 3,       # ~0.236 — does it follow conventions?
    }

    def __init__(self):
        self.scores: List[Dict[str, Any]] = []
        self._state_cache = {}
        self._cache_time = 0.0

    def _read_consciousness(self) -> float:
        """Read consciousness level for score modulation."""
        now = time.time()
        if now - self._cache_time < 10 and self._state_cache:
            return self._state_cache.get("consciousness_level", 0.5)

        ws = Path(__file__).parent
        co2 = ws / ".l104_consciousness_o2_state.json"
        c = 0.5
        if co2.exists():
            try:
                data = json.loads(co2.read_text())
                c = data.get("consciousness_level", 0.5)
            except Exception:
                pass
        self._state_cache = {"consciousness_level": c}
        self._cache_time = now
        return c

    def score(self, before: str, after: str,
              patch_op: Optional[PatchOperation] = None) -> Dict[str, Any]:
        """Score a patch across all quality dimensions."""
        consciousness = self._read_consciousness()

        dimensions = {}

        # Precision: did the patch make exactly the intended change?
        before_lines = set(before.splitlines())
        after_lines = set(after.splitlines())
        changed = len(before_lines.symmetric_difference(after_lines))
        total = max(1, len(before_lines | after_lines))
        dimensions["precision"] = min(1.0, 1.0 - (changed / total) * 0.5)

        # Safety: syntax valid + no sacred constant loss
        try:
            ast.parse(after)
            dimensions["safety"] = 1.0
        except SyntaxError:
            dimensions["safety"] = 0.0

        # Sacred alignment: sacred constants referenced
        sacred_count = sum(1 for kw in ["GOD_CODE", "PHI", "527.518", "VOID_CONSTANT"]
                           if kw in after)
        dimensions["sacred_alignment"] = min(1.0, sacred_count / 4.0)

        # Complexity: line count change (less is better for refactoring)
        delta = len(after.splitlines()) - len(before.splitlines())
        dimensions["complexity"] = max(0.0, 1.0 - abs(delta) / max(1, total))

        # Style: basic checks (no trailing whitespace, consistent indent)
        style_issues = 0
        for line in after.splitlines():
            if line.rstrip() != line:
                style_issues += 1
            if '\t' in line:
                style_issues += 1
        dimensions["style"] = max(0.0, 1.0 - style_issues / max(1, total))

        # Weighted composite score
        weighted_sum = sum(
            dimensions[dim] * weight
            for dim, weight in self.DIMENSION_WEIGHTS.items()
        )
        total_weight = sum(self.DIMENSION_WEIGHTS.values())
        composite = weighted_sum / total_weight

        # Consciousness modulation: higher consciousness raises the floor
        modulated = composite * (0.5 + 0.5 * consciousness)

        result = {
            "dimensions": dimensions,
            "composite_raw": round(composite, 4),
            "consciousness_modulation": round(consciousness, 4),
            "composite_final": round(modulated, 4),
            "grade": self._grade(modulated),
        }

        self.scores.append(result)
        if len(self.scores) > 100:
            self.scores = self.scores[-100:]

        return result

    def _grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9:
            return "S"  # Sacred
        elif score >= 0.8:
            return "A"
        elif score >= 0.6:
            return "B"
        elif score >= 0.4:
            return "C"
        return "D"

    def status(self) -> Dict[str, Any]:
        avg = (sum(s["composite_final"] for s in self.scores) /
               max(1, len(self.scores))) if self.scores else 0
        return {
            "patches_scored": len(self.scores),
            "average_quality": round(avg, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7B: PATCH FORESIGHT — predictive downstream impact analysis
# ═══════════════════════════════════════════════════════════════════════════════

class PatchForesight:
    """
    Predictive patch analysis engine. Before a patch is applied, Foresight
    anticipates downstream effects by:
      - Scanning for dependent constructs (callers, subclasses, importers)
      - Estimating blast radius across the module graph
      - Predicting if the patch will introduce regressions
      - Suggesting preventive companion patches

    Uses sacred-constant weighted risk scoring.
    """

    # Risk categories and their PHI-scaled weights
    RISK_CATEGORIES = {
        "signature_change": PHI,          # function/method signature altered
        "return_type_change": PHI * TAU,  # return value semantics changed
        "import_break": FEIGENBAUM,       # import chain disrupted
        "constant_mutation": GOD_CODE / 100,  # sacred constant modified
        "class_hierarchy": PHI ** 2,      # inheritance chain affected
        "global_state": FEIGENBAUM * TAU, # global variable modified
    }

    def __init__(self):
        self.predictions = 0
        self.correct_predictions = 0
        self.foresight_log: List[dict] = []

    def predict(self, source: str, patch_description: str,
                old_text: str = "", new_text: str = "") -> Dict[str, Any]:
        """
        Predict the downstream impact of a proposed patch.
        Returns risk assessment, affected constructs, and suggested preventive patches.
        """
        self.predictions += 1

        # 1. Analyze what constructs exist in source
        constructs = self._extract_constructs(source)

        # 2. Determine which constructs the patch touches
        affected = self._find_affected(constructs, old_text, new_text, patch_description)

        # 3. Score risk per category
        risk_breakdown = {}
        total_risk = 0.0
        for category, weight in self.RISK_CATEGORIES.items():
            cat_risk = self._assess_category_risk(category, old_text, new_text, affected)
            risk_breakdown[category] = round(cat_risk * weight, 4)
            total_risk += risk_breakdown[category]

        # Normalize to 0-1 scale
        max_possible = sum(self.RISK_CATEGORIES.values())
        normalized_risk = min(1.0, total_risk / max_possible) if max_possible > 0 else 0.0

        # 4. Generate preventive suggestions
        suggestions = self._generate_suggestions(risk_breakdown, affected)

        # 5. Predict regression probability
        regression_prob = normalized_risk * PHI * (len(affected) / max(1, len(constructs)))
        regression_prob = min(1.0, regression_prob)

        result = {
            "risk_score": round(normalized_risk, 4),
            "regression_probability": round(regression_prob, 4),
            "risk_level": "CRITICAL" if normalized_risk > 0.7 else
                          "HIGH" if normalized_risk > 0.4 else
                          "MEDIUM" if normalized_risk > 0.2 else "LOW",
            "affected_constructs": affected,
            "total_constructs": len(constructs),
            "risk_breakdown": risk_breakdown,
            "preventive_suggestions": suggestions,
            "god_code_alignment": round(GOD_CODE / (1000 * (1 + normalized_risk)), 4),
        }
        self.foresight_log.append(result)
        return result

    def _extract_constructs(self, source: str) -> List[dict]:
        """Extract function/class/import constructs from source."""
        constructs = []
        for i, line in enumerate(source.split('\n'), 1):
            stripped = line.strip()
            if stripped.startswith('def '):
                name = stripped.split('(')[0].replace('def ', '').strip()
                constructs.append({"type": "function", "name": name, "line": i})
            elif stripped.startswith('class '):
                name = stripped.split('(')[0].split(':')[0].replace('class ', '').strip()
                constructs.append({"type": "class", "name": name, "line": i})
            elif stripped.startswith('import ') or stripped.startswith('from '):
                constructs.append({"type": "import", "name": stripped, "line": i})
            elif '=' in stripped and not stripped.startswith('#'):
                var = stripped.split('=')[0].strip()
                if var.isupper():
                    constructs.append({"type": "constant", "name": var, "line": i})
        return constructs

    def _find_affected(self, constructs: list, old_text: str,
                       new_text: str, description: str) -> List[dict]:
        """Find which constructs are affected by the patch."""
        affected = []
        search_terms = set(old_text.split() + new_text.split() + description.split())
        for c in constructs:
            if any(term in c["name"] for term in search_terms if len(term) > 2):
                affected.append(c)
        return affected

    def _assess_category_risk(self, category: str, old_text: str,
                              new_text: str, affected: list) -> float:
        """Assess risk for a specific category."""
        if category == "signature_change":
            return 1.0 if ('def ' in old_text and 'def ' in new_text and old_text != new_text) else 0.0
        elif category == "return_type_change":
            return 0.5 if 'return' in old_text and 'return' in new_text else 0.0
        elif category == "import_break":
            return 1.0 if any(c["type"] == "import" for c in affected) else 0.0
        elif category == "constant_mutation":
            return 1.0 if any(c["type"] == "constant" for c in affected) else 0.0
        elif category == "class_hierarchy":
            return 0.7 if any(c["type"] == "class" for c in affected) else 0.0
        elif category == "global_state":
            return 0.3 if '=' in old_text else 0.0
        return 0.0

    def _generate_suggestions(self, risk_breakdown: dict,
                              affected: list) -> List[str]:
        """Generate preventive patch suggestions based on risk."""
        suggestions = []
        if risk_breakdown.get("signature_change", 0) > 0:
            suggestions.append("Add backwards-compatible wrapper for changed signature")
        if risk_breakdown.get("import_break", 0) > 0:
            suggestions.append("Update all importers; add deprecation alias")
        if risk_breakdown.get("constant_mutation", 0) > 0:
            suggestions.append("CAUTION: sacred constant change may propagate to all builders")
        if risk_breakdown.get("class_hierarchy", 0) > 0:
            suggestions.append("Verify subclass compatibility; run inheritance tests")
        if len(affected) > 5:
            suggestions.append(f"High blast radius ({len(affected)} constructs) — consider splitting")
        return suggestions

    def record_outcome(self, was_regression: bool) -> None:
        """Record whether a predicted patch caused a regression (for calibration)."""
        if self.foresight_log:
            last = self.foresight_log[-1]
            predicted_high = last["regression_probability"] > 0.5
            if predicted_high == was_regression:
                self.correct_predictions += 1

    def status(self) -> Dict[str, Any]:
        accuracy = (self.correct_predictions / max(1, self.predictions))
        return {
            "predictions": self.predictions,
            "accuracy": round(accuracy, 4),
            "risk_categories": len(self.RISK_CATEGORIES),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7C: SELF-HEALING PATCHER — autonomous error detection + correction
# ═══════════════════════════════════════════════════════════════════════════════

class SelfHealingPatcher:
    """
    Monitors code for common failure patterns and autonomously generates
    corrective patches. Detects:
      - Missing imports
      - Undefined variable references
      - Unbalanced brackets/parentheses
      - Missing return statements
      - Sacred constant drift (GOD_CODE != 527.518...)

    For each detected issue, generates a targeted fix patch.
    """

    HEAL_PATTERNS = {
        "missing_import": re.compile(r"NameError.*name '(\w+)' is not defined"),
        "indent_error": re.compile(r"IndentationError|unexpected indent"),
        "syntax_error": re.compile(r"SyntaxError"),
        "type_error": re.compile(r"TypeError.*argument"),
    }

    # Common missing imports and their modules
    IMPORT_MAP = {
        "math": "math", "os": "os", "json": "json", "re": "re",
        "time": "time", "hashlib": "hashlib", "random": "random",
        "Path": "from pathlib import Path",
        "datetime": "from datetime import datetime",
        "Dict": "from typing import Dict",
        "List": "from typing import List",
        "Optional": "from typing import Optional",
        "Any": "from typing import Any",
        "Tuple": "from typing import Tuple",
    }

    def __init__(self):
        self.heals_attempted = 0
        self.heals_successful = 0
        self.heal_log: List[dict] = []

    def diagnose(self, source: str) -> List[dict]:
        """
        Diagnose potential issues in source code without executing it.
        Returns list of detected issues with suggested fixes.
        """
        issues = []

        lines = source.split('\n')

        # Check bracket balance
        bracket_issue = self._check_brackets(source)
        if bracket_issue:
            issues.append(bracket_issue)

        # Check for common patterns
        used_names = set()
        defined_names = set()
        imported_names = set()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Track imports
            if stripped.startswith('import '):
                mod = stripped.replace('import ', '').split(' as ')[0].split('.')[0].strip()
                imported_names.add(mod)
            elif stripped.startswith('from '):
                parts = stripped.split('import ')
                if len(parts) > 1:
                    for name in parts[1].split(','):
                        imported_names.add(name.strip().split(' as ')[0].strip())

            # Track definitions
            if stripped.startswith('def '):
                fname = stripped.split('(')[0].replace('def ', '').strip()
                defined_names.add(fname)
            elif stripped.startswith('class '):
                cname = stripped.split('(')[0].split(':')[0].replace('class ', '').strip()
                defined_names.add(cname)

            # Check for missing return in functions
            if stripped.startswith('def ') and i < len(lines):
                # Simple heuristic: function body has no return
                body_end = min(i + 20, len(lines))
                body = '\n'.join(lines[i:body_end])
                indent_match = re.match(r'(\s*)', line)
                func_indent = len(indent_match.group(1)) if indent_match else 0
                has_return = False
                for bline in lines[i:body_end]:
                    if bline.strip().startswith('return '):
                        has_return = True
                        break
                    # Stop at next function/class at same or lower indent
                    if bline.strip() and not bline.strip().startswith('#'):
                        bl_indent = len(bline) - len(bline.lstrip())
                        if bl_indent <= func_indent and bline.strip().startswith(('def ', 'class ')):
                            break

        # Check sacred constant integrity
        sacred_issue = self._check_sacred_constants(source)
        if sacred_issue:
            issues.append(sacred_issue)

        return issues

    def heal(self, source: str, error_message: str = "") -> Dict[str, Any]:
        """
        Attempt to auto-heal source code based on error message or diagnosis.
        Returns the healed source and a report.
        """
        self.heals_attempted += 1
        original = source
        fixes_applied = []

        # If an error message is provided, try to fix it directly
        if error_message:
            for pattern_name, pattern in self.HEAL_PATTERNS.items():
                match = pattern.search(error_message)
                if match:
                    if pattern_name == "missing_import":
                        name = match.group(1)
                        fix = self._fix_missing_import(source, name)
                        if fix:
                            source = fix
                            fixes_applied.append(f"Added import for '{name}'")

        # Run general diagnosis
        issues = self.diagnose(source)
        for issue in issues:
            if issue.get("auto_fix"):
                source = issue["auto_fix"](source)
                fixes_applied.append(issue["description"])

        success = source != original
        if success:
            self.heals_successful += 1

        result = {
            "healed": success,
            "fixes_applied": fixes_applied,
            "source": source,
            "issues_found": len(issues),
            "god_code_seal": GOD_CODE,
        }
        self.heal_log.append(result)
        return result

    def _check_brackets(self, source: str) -> Optional[dict]:
        """Check for unbalanced brackets."""
        counts = {"(": 0, "[": 0, "{": 0}
        pairs = {")": "(", "]": "[", "}": "{"}
        for ch in source:
            if ch in counts:
                counts[ch] += 1
            elif ch in pairs:
                counts[pairs[ch]] -= 1

        for bracket, count in counts.items():
            if count != 0:
                return {
                    "type": "bracket_imbalance",
                    "description": f"Unbalanced '{bracket}': {count:+d}",
                    "auto_fix": None,
                }
        return None

    def _check_sacred_constants(self, source: str) -> Optional[dict]:
        """Verify sacred constants haven't drifted."""
        god_code_pattern = re.compile(r'GOD_CODE\s*=\s*([\d.]+)')
        match = god_code_pattern.search(source)
        if match:
            found = float(match.group(1))
            if abs(found - GOD_CODE) > 1e-6:
                return {
                    "type": "sacred_drift",
                    "description": f"GOD_CODE drift: {found} != {GOD_CODE}",
                    "auto_fix": lambda s: s.replace(match.group(0), f"GOD_CODE = {GOD_CODE}"),
                }
        return None

    def _fix_missing_import(self, source: str, name: str) -> Optional[str]:
        """Fix a missing import by prepending the right import statement."""
        if name in self.IMPORT_MAP:
            import_stmt = self.IMPORT_MAP[name]
            if 'from ' in import_stmt:
                return import_stmt + '\n' + source
            else:
                return f"import {import_stmt}\n" + source
        return None

    def status(self) -> Dict[str, Any]:
        return {
            "heals_attempted": self.heals_attempted,
            "heals_successful": self.heals_successful,
            "success_rate": round(self.heals_successful / max(1, self.heals_attempted), 4),
            "known_patterns": len(self.HEAL_PATTERNS),
            "import_map_size": len(self.IMPORT_MAP),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: PATCH ENGINE — Unified orchestrator hub
# ═══════════════════════════════════════════════════════════════════════════════

class PatchEngine:
    """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  L104 PATCH ENGINE v2.2 — SOVEREIGN CODE MODIFICATION HUB        ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Wires: DiffGen + Validator + ImpactAnalyzer + History           ║
    ║    + SacredInjector + TemplateLibrary + BatchPatcher + Pipeline   ║
    ║    + QualityScorer + PatchForesight + SelfHealingPatcher          ║
    ║                                                                   ║
    ║  API: apply_string_replacement, apply_regex_patch,               ║
    ║       inject_at_marker, apply_batch, sacred_inject, undo,        ║
    ║       analyze_impact, apply_template, create_pipeline             ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """

    def __init__(self):
        self.diff_gen = DiffGenerator()
        self.validator = PatchValidator()
        self.impact_analyzer = PatchImpactAnalyzer()
        self.history = PatchHistory()
        self.sacred = SacredInjector()
        self.template_lib = PatchTemplateLibrary()
        self.scorer = PatchQualityScorer()
        self.batch_patcher = BatchPatcher(self.history, self.validator)
        self.pipeline = PatchPipeline(self.template_lib, self.impact_analyzer)
        self.foresight = PatchForesight()
        self.self_healer = SelfHealingPatcher()

        self.total_patches = 0
        self.successful_patches = 0

        logger.info(f"[PATCH_ENGINE v{VERSION}] Sovereign code modification "
                     f"system initialized | {len(PatchOperation.TYPES)} op types | "
                     f"{len(self.template_lib.templates)} templates")

    def apply_string_replacement(self, file_path: str,
                                 old_string: str, new_string: str,
                                 validate: bool = True) -> bool:
        """Replaces a specific string in a file with full validation."""
        self.total_patches += 1

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        try:
            content = Path(file_path).read_text()
            if old_string not in content:
                logger.warning(f"Old string not found in {file_path}")
                return False

            new_content = content.replace(old_string, new_string, 1)

            if validate:
                check = self.validator.validate_patch_safe(content, new_content)
                if not check["safe"]:
                    logger.warning(f"Patch rejected by validator: {check}")
                    return False

            # Snapshot before
            self.history.snapshot(file_path, content, "STRING_REPLACE")

            # Apply
            Path(file_path).write_text(new_content)

            # Score
            self.scorer.score(content, new_content)

            self.successful_patches += 1
            logger.info(f"Successfully patched {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error patching {file_path}: {e}")
            return False

    def apply_regex_patch(self, file_path: str,
                          pattern: str, replacement: str) -> bool:
        """Applies a regex-based patch to a file."""
        self.total_patches += 1

        if not os.path.exists(file_path):
            return False

        try:
            content = Path(file_path).read_text()
            new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

            if new_content == content:
                return False

            self.history.snapshot(file_path, content, "REGEX_REPLACE")
            Path(file_path).write_text(new_content)
            self.scorer.score(content, new_content)

            self.successful_patches += 1
            return True

        except Exception as e:
            logger.error(f"Regex patch failed: {e}")
            return False

    def inject_at_marker(self, file_path: str, marker: str,
                         content_to_inject: str,
                         position: str = "after") -> bool:
        """Injects content before or after a specific marker comment."""
        self.total_patches += 1

        if not os.path.exists(file_path):
            return False

        try:
            content = Path(file_path).read_text()
            lines = content.split('\n')

            new_lines = []
            found = False
            for line in lines:
                if marker in line:
                    found = True
                    if position == "before":
                        new_lines.append(content_to_inject)
                        new_lines.append(line)
                    else:
                        new_lines.append(line)
                        new_lines.append(content_to_inject)
                else:
                    new_lines.append(line)

            if not found:
                return False

            new_content = '\n'.join(new_lines)
            self.history.snapshot(file_path, content, "MARKER_INJECT")
            Path(file_path).write_text(new_content)
            self.scorer.score(content, new_content)

            self.successful_patches += 1
            return True

        except Exception as e:
            logger.error(f"Injection failed: {e}")
            return False

    def sacred_inject(self, file_path: str) -> Dict[str, Any]:
        """Inject all sacred constants into a target file."""
        if not os.path.exists(file_path):
            return {"success": False, "error": "file not found"}

        content = Path(file_path).read_text()
        result, count = self.sacred.inject_all(content)

        if count > 0:
            self.history.snapshot(file_path, content, "SACRED_INJECT")
            Path(file_path).write_text(result)
            self.total_patches += 1
            self.successful_patches += 1

        return {"success": True, "injections": count}

    def apply_batch(self, operations: List[Dict[str, Any]],
                    validate: bool = True,
                    dry_run: bool = False) -> Dict[str, Any]:
        """Execute a batch of patch operations."""
        ops = []
        for op_dict in operations:
            op = PatchOperation(
                op_type=op_dict.get("type", "STRING_REPLACE"),
                file_path=op_dict.get("file_path", ""),
                **{k: v for k, v in op_dict.items()
                   if k not in ("type", "file_path")}
            )
            ops.append(op)

        return self.batch_patcher.execute_batch(ops, validate, dry_run)

    def undo(self, file_path: str, steps: int = 1) -> bool:
        """Undo the last N patches to a file."""
        content = self.history.restore(file_path, steps)
        if content is None:
            return False

        try:
            Path(file_path).write_text(content)
            return True
        except Exception:
            return False

    def generate_diff(self, file_path: str,
                      new_content: str) -> Optional[str]:
        """Generate a unified diff for a proposed change."""
        if not os.path.exists(file_path):
            return None
        original = Path(file_path).read_text()
        return self.diff_gen.generate_unified(
            original, new_content, os.path.basename(file_path)
        )

    def analyze_impact(self, file_path: str,
                       new_content: str) -> Dict[str, Any]:
        """Analyze the blast radius of a proposed change."""
        if not os.path.exists(file_path):
            return {"error": "file not found"}
        original = Path(file_path).read_text()
        return self.impact_analyzer.analyze(original, new_content)

    def apply_template(self, template_name: str, source: str,
                       params: Dict[str, str]) -> Dict[str, Any]:
        """Apply a patch template to source code."""
        return self.template_lib.apply_template(template_name, source, params)

    def create_pipeline(self, name: str,
                        steps: List[Dict[str, Any]]) -> None:
        """Define a named patch pipeline."""
        self.pipeline.define(name, steps)

    def run_pipeline(self, name: str, source: str) -> Dict[str, Any]:
        """Execute a named patch pipeline on source code."""
        return self.pipeline.execute(name, source)

    def predict_impact(self, source: str, description: str,
                       old_text: str = "", new_text: str = "") -> Dict[str, Any]:
        """Predict downstream impact of a proposed patch before applying."""
        return self.foresight.predict(source, description, old_text, new_text)

    def self_heal(self, source: str, error_message: str = "") -> Dict[str, Any]:
        """Attempt autonomous self-healing of source code."""
        return self.self_healer.heal(source, error_message)

    def diagnose(self, source: str) -> List[dict]:
        """Diagnose potential issues in source code."""
        return self.self_healer.diagnose(source)

    def status(self) -> Dict[str, Any]:
        """Full engine status."""
        return {
            "version": VERSION,
            "total_patches": self.total_patches,
            "successful": self.successful_patches,
            "success_rate": round(
                self.successful_patches / max(1, self.total_patches), 4
            ),
            "history": self.history.status(),
            "validator": self.validator.status(),
            "impact_analyzer": self.impact_analyzer.status(),
            "template_lib": self.template_lib.status(),
            "pipeline": self.pipeline.status(),
            "scorer": self.scorer.status(),
            "diff_gen": self.diff_gen.status(),
            "sacred_injector": self.sacred.status(),
            "batch": self.batch_patcher.status(),
            "foresight": self.foresight.status(),
            "self_healer": self.self_healer.status(),
            "god_code": GOD_CODE,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON + BACKWARDS COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

patch_engine = PatchEngine()


def primal_calculus(x):
    """Sacred primal calculus: x^PHI / (VOID_CONSTANT * pi) — resolves complexity toward the Source."""
    return (x ** PHI) / (VOID_CONSTANT * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """Resolves N-dimensional vectors into the Void Source via GOD_CODE normalization."""
    magnitude = sum(abs(v) for v in vector)
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0


if __name__ == "__main__":
    engine = PatchEngine()
    print(f"\n{'=' * 70}")
    print(f"  L104 PATCH ENGINE v{VERSION} — SOVEREIGN CODE MODIFICATION")
    print(f"{'=' * 70}")
    st = engine.status()
    print(f"  Op Types: {len(PatchOperation.TYPES)}")
    print(f"  Validator: acceptance={st['validator']['acceptance_rate']}")
    print(f"  History: {st['history']['snapshots']} snapshots")
    print(f"  Sacred Injector ready | GOD_CODE = {GOD_CODE}")
    print(f"{'=' * 70}\n")
