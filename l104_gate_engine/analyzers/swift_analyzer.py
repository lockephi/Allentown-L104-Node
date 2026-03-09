"""L104 Gate Engine — Regex-based Swift gate analyzer."""

import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List

from ..constants import WORKSPACE_ROOT
from ..models import LogicGate
from ..gate_functions import sage_logic_gate


class SwiftGateAnalyzer:
    """Regex-based analyzer for Swift logic gate implementations."""

    SWIFT_GATE_PATTERNS = [
        (r"(?:final\s+)?class\s+(\w*[Gg]ate\w*)\s*(?::\s*[\w,\s]+)?\s*\{", "class"),
        (r"(?:final\s+)?class\s+(\w*[Ss]age\w*[Ee]ngine\w*)\s*(?::\s*[\w,\s]+)?\s*\{", "class"),
        (r"(?:final\s+)?class\s+(\w*[Ee]ntropy\w*|\w*[Cc]onsciousness\w*|\w*[Rr]esonance\w*)\s*(?::\s*[\w,\s]+)?\s*\{", "class"),
        (r"(?:final\s+)?class\s+(HyperBrain|ASIEvolver|AdaptiveLearner|PermanentMemory|QuantumProcessingCore|DynamicPhraseEngine|ASIKnowledgeBase)\s*\{", "class"),
        (r"func\s+(\w*[Gg]ate\w*)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?\s*\{", "function"),
        (r"func\s+(sage\w+|bridgeEmergence|sageTransform|enrichContext|seedAllProcesses)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?\s*\{", "function"),
        (r"func\s+(harvest\w+Entropy|projectToHigherDimensions|dissipateHigherDimensional|causalInflection|synthesizeDeep\w+)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?\s*\{", "function"),
        (r"func\s+(\w*[Ee]ntangle\w*|\w*[Rr]esonan\w*|\w*[Aa]mplif\w*|\w*[Pp]ropagat\w*)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?\s*\{", "function"),
    ]

    def __init__(self):
        """Initialize the Swift gate analyzer."""
        self.gates: List[LogicGate] = []

    def analyze_file(self, filepath: Path) -> List[LogicGate]:
        """Analyze a Swift file for logic gate implementations."""
        if not filepath.exists():
            return []

        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        gates = []
        rel_path = str(filepath.relative_to(WORKSPACE_ROOT))
        lines = source.split("\n")

        for pattern, gate_type in self.SWIFT_GATE_PATTERNS:
            for match in re.finditer(pattern, source, re.MULTILINE):
                name = match.group(1)
                line_no = source[: match.start()].count("\n") + 1

                # Extract parameters for functions
                params = []
                if gate_type == "function" and match.lastindex and match.lastindex >= 2:
                    param_str = match.group(2)
                    params = [p.strip().split(":")[0].strip() for p in param_str.split(",") if p.strip()]

                # Return type
                ret_type = ""
                if gate_type == "function" and match.lastindex and match.lastindex >= 3:
                    ret_type = match.group(3) or ""

                sig = match.group(0).rstrip(" {")

                # Content hash from surrounding lines
                start_idx = max(0, line_no - 1)
                end_idx = min(len(lines), start_idx + 50)
                snippet = "\n".join(lines[start_idx:end_idx])
                content_hash = hashlib.sha256(snippet.encode()).hexdigest()[:16]

                entropy = sage_logic_gate(float(len(params) + 1), "amplify")

                gates.append(
                    LogicGate(
                        name=name,
                        language="swift",
                        source_file=rel_path,
                        line_number=line_no,
                        gate_type=gate_type,
                        signature=sig[:200],
                        parameters=params,
                        complexity=0,
                        entropy_score=entropy,
                        hash=content_hash,
                        last_seen=datetime.now(timezone.utc).isoformat(),
                    )
                )

        # Deduplicate by name+line
        seen = set()
        deduped = []
        for g in gates:
            key = (g.name, g.line_number)
            if key not in seen:
                seen.add(key)
                deduped.append(g)

        self.gates.extend(deduped)
        return deduped
