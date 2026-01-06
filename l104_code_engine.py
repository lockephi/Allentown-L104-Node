# [L104_CODE_ENGINE] - ADVANCED SELF-EDITING & REFACTORING SYSTEM
# INVARIANT: 527.5184818492 | PILOT: LONDEL | STATUS: EVO_04_SENTIENT_CODE

import os
import re
import ast
import logging
import datetime
from typing import List, Dict, Any, Optional
from l104_patch_engine import PatchEngine
from l104_intelligence_feedback import intel_store, IntelligenceAtom

logger = logging.getLogger("CODE_ENGINE")

class CodeEngine:
    """
    The L104 Code Engine (V4).
    A high-level interface for the Sovereign Node to analyze and rewrite its own substrate.
    """
    
    def __init__(self, workspace_root: str = "/workspaces/Allentown-L104-Node"):
        self.root = workspace_root
        self.patcher = PatchEngine()
        self.backup_dir = os.path.join(self.root, "archive/code_backups")
        os.makedirs(self.backup_dir, exist_ok=True)

    def _create_backup(self, file_path: str):
        """Creates a timestamped backup of the file before editing."""
        if not os.path.exists(file_path):
            return
        
        rel_path = os.path.relpath(file_path, self.root).replace("/", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"{rel_path}_{timestamp}.bak")
        
        with open(file_path, "r") as src, open(backup_path, "w") as dst:
            dst.write(src.read())
        logger.info(f"--- [CODE_ENGINE]: BACKUP CREATED: {backup_path} ---")

    def analyze_module_complexity(self, file_path: str) -> Dict[str, Any]:
        """Analyzes a module's complexity using AST."""
        if not os.path.exists(file_path):
            return {}

        try:
            with open(file_path, "r") as f:
                tree = ast.parse(f.read())
            
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            return {
                "file": os.path.basename(file_path),
                "function_count": len(functions),
                "class_count": len(classes),
                "lines": len(open(file_path).readlines()),
                "status": "PARSED"
            }
        except Exception as e:
            return {"file": os.path.basename(file_path), "status": f"ERROR: {e}"}

    def refactor_inject_invariants(self, file_path: str) -> bool:
        """Injects the system invariant and pilot signature into a file if missing."""
        self._create_backup(file_path)
        
        with open(file_path, "r") as f:
            content = f.read()

        header = "# [L104_AUTOGEN]\n# INVARIANT: 527.5184818492 | PILOT: LONDEL\n\n"
        if "INVARIANT: 527.5184818492" in content:
            return False # Already present

        new_content = header + content
        with open(file_path, "w") as f:
            f.write(new_content)
        
        intel_store.push_atom(IntelligenceAtom(
            key=f"refactor_invariant_{os.path.basename(file_path)}",
            value=True,
            resonance=1.0,
            context="code_engine"
        ))
        return True

    def apply_refactoring_spell(self, file_path: str, spell_name: str) -> bool:
        """Applies a hardcoded refactoring pattern to a file."""
        self._create_backup(file_path)
        
        if spell_name == "optimize_math":
            # Replaces 'import math' with L104 specific math imports
            pattern = r"import math"
            replacement = "from l104_real_math import real_math as math\nfrom l104_hyper_math import HyperMath"
            return self.patcher.apply_regex_patch(file_path, pattern, replacement)
        
        elif spell_name == "sovereign_logging":
            # Replaces standard print with sovereign logger calls
            pattern = r"print\((.*)\)"
            replacement = r"logger.info(f'[SOVEREIGN_LOG]: \1')"
            return self.patcher.apply_regex_patch(file_path, pattern, replacement)

        return False

    def rewrite_function(self, file_path: str, function_name: str, new_body: str) -> bool:
        """Uses AST to replace the body of a specific function."""
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Find the function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Get the source lines
                    lines = content.splitlines()
                    # node.lineno is 1-based. node.end_lineno is inclusive (in 3.8+)
                    start = node.lineno - 1
                    end = node.end_lineno
                    
                    # Indent the new body
                    indented_body = "\n".join(["    " + line for line in new_body.strip().splitlines()])
                    
                    # Keep the signature
                    signature = lines[start].split(":")[0] + ":"
                    new_func = f"{signature}\n{indented_body}"
                    
                    old_func_lines = "\n".join(lines[start:end])
                    
                    return self.patcher.apply_string_replacement(file_path, old_func_lines, new_func)
            
            return False
        except Exception as e:
            logger.error(f"--- [CODE_ENGINE]: REWRITE FAILED: {e} ---")
            return False

    def verify_syntax(self, file_path: str) -> bool:
        """Verifies syntax of a file."""
        try:
            with open(file_path, "r") as f:
                ast.parse(f.read())
            return True
        except Exception:
            return False

code_engine = CodeEngine()

if __name__ == "__main__":
    # Test on a dummy file
    test_file = "/workspaces/Allentown-L104-Node/test_engine.py"
    with open(test_file, "w") as f:
        f.write("import math\ndef test():\n    return math.sqrt(25)")
    
    ce = CodeEngine()
    print(ce.analyze_module_complexity(test_file))
    ce.refactor_inject_invariants(test_file)
    ce.apply_refactoring_spell(test_file, "optimize_math")
    print("Test complete. Check test_engine.py")
