"""L104 Gate Engine — Regex-based JavaScript gate analyzer."""

import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List

from ..constants import WORKSPACE_ROOT
from ..models import LogicGate


class JavaScriptGateAnalyzer:
    """Regex-based analyzer for JavaScript logic gate implementations."""

    JS_GATE_PATTERNS = [
        (r"class\s+(\w*[Gg]ate\w*)\s*(?:extends\s+\w+)?\s*\{", "class"),
        (r"(?:export\s+)?(?:async\s+)?function\s+(\w*[Gg]ate\w*)\s*\(", "function"),
        (r"(\w*[Gg]ate\w*)\s*[=:]\s*(?:async\s+)?(?:\([^)]*\))?\s*=>", "function"),
        (r"(\w*[Gg]ate\w*)\s*[=:]\s*function", "function"),
    ]

    def analyze_directory(self, directory: Path) -> List[LogicGate]:
        """Scan a directory for JavaScript logic gate implementations."""
        gates = []
        for js_file in directory.rglob("*.js"):
            try:
                source = js_file.read_text(encoding="utf-8", errors="replace")
                rel_path = str(js_file.relative_to(WORKSPACE_ROOT))
                for pattern, gate_type in self.JS_GATE_PATTERNS:
                    for match in re.finditer(pattern, source, re.MULTILINE):
                        name = match.group(1)
                        line_no = source[: match.start()].count("\n") + 1
                        sig = match.group(0).rstrip(" {")
                        content_hash = hashlib.sha256(sig.encode()).hexdigest()[:16]

                        gates.append(
                            LogicGate(
                                name=name,
                                language="javascript",
                                source_file=rel_path,
                                line_number=line_no,
                                gate_type=gate_type,
                                signature=sig[:200],
                                hash=content_hash,
                                last_seen=datetime.now(timezone.utc).isoformat(),
                            )
                        )
            except Exception:
                continue
        return gates
