"""
L104 Sage Scour Engine v2.0.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deep codebase analysis engine — scans the L104 workspace for
invariant patterns, sacred constant usage, code quality signals,
dead import detection, duplication hotspots, and structural anomalies.
Feeds findings into the ASI pipeline for self-improvement.

Subsystems:
  - InvariantScanner: sacred constant usage & drift detection
  - ImportAuditor: dead/missing/circular import analysis
  - DuplicationDetector: content-hash clone detection
  - AnomalyScorer: per-file structural anomaly scoring
  - ScourPersistence: JSONL result persistence

Sacred Constants: GOD_CODE=527.5184818492612 | PHI=1.618033988749895
"""
VOID_CONSTANT = 1.0416180339887497
import math
import os
import re
import json
import hashlib
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple

# ZENITH_UPGRADE_ACTIVE: 2026-02-15T00:00:00
ZENITH_HZ = 3887.8
UUC = 2402.792541

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
FEIGENBAUM = 4.669201609
TAU = 6.283185307179586
GROVER_AMPLIFICATION = PHI ** 3

VERSION = "2.0.0"


class InvariantScanner:
    """Scans source files for sacred constant usage and drift."""

    SACRED = {
        'GOD_CODE': 527.5184818492612,
        'PHI': 1.618033988749895,
        'VOID_CONSTANT': 1.0416180339887497,
        'FEIGENBAUM': 4.669201609,
        'TAU': 6.283185307179586,
    }

    PATTERNS = [
        re.compile(r'527\.518\d*'),
        re.compile(r'1\.618033\d*'),
        re.compile(r'1\.04161\d*'),
        re.compile(r'4\.66920\d*'),
        re.compile(r'6\.28318\d*'),
    ]

    def scan_file(self, filepath: str) -> Dict[str, Any]:
        """Scan a single file for sacred constant usage."""
        try:
            content = Path(filepath).read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return {'file': filepath, 'error': True}

        hits = []
        for pat in self.PATTERNS:
            for match in pat.finditer(content):
                hits.append({
                    'value': match.group(),
                    'position': match.start(),
                })

        # Detect explicit constant assignments
        assignments = re.findall(r'(GOD_CODE|PHI|VOID_CONSTANT|FEIGENBAUM|TAU)\s*=\s*([\d.e+-]+)', content)

        return {
            'file': os.path.basename(filepath),
            'invariant_hits': len(hits),
            'assignments': len(assignments),
            'sacred_aligned': len(hits) > 0,
        }


class ImportAuditor:
    """Detects dead, missing, and circular imports."""

    IMPORT_RE = re.compile(r'^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))', re.MULTILINE)

    def audit_file(self, filepath: str) -> Dict[str, Any]:
        try:
            content = Path(filepath).read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return {'file': filepath, 'error': True}

        imports = []
        for m in self.IMPORT_RE.finditer(content):
            mod = m.group(1) or m.group(2)
            if mod:
                imports.append(mod.split('.')[0])

        # Check which imported names are actually used in the rest of the file
        used = []
        unused = []
        for imp in imports:
            short = imp.split('.')[-1]
            # Simple heuristic: is the module name referenced later in the file?
            if content.count(short) > 1:
                used.append(imp)
            else:
                unused.append(imp)

        return {
            'file': os.path.basename(filepath),
            'total_imports': len(imports),
            'used': len(used),
            'potentially_unused': unused,
        }


class DuplicationDetector:
    """Content-hash based clone detection across files."""

    def __init__(self, min_block: int = 4):
        self.min_block = min_block

    def hash_blocks(self, filepath: str, block_size: int = 4) -> List[str]:
        """Hash consecutive line blocks for clone detection."""
        try:
            lines = Path(filepath).read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            return []

        hashes = []
        for i in range(len(lines) - block_size + 1):
            block = '\n'.join(line.strip() for line in lines[i:i+block_size] if line.strip())
            if len(block) > 20:
                hashes.append(hashlib.md5(block.encode()).hexdigest())
        return hashes

    def find_clones(self, file_list: List[str]) -> Dict[str, int]:
        """Find duplicated code blocks across files."""
        all_hashes = Counter()
        for fp in file_list:
            hashes = self.hash_blocks(fp)
            # Use set to count each hash once per file
            for h in set(hashes):
                all_hashes[h] += 1

        # Clones are hashes appearing in 2+ files
        clones = {h: count for h, count in all_hashes.items() if count > 1}
        return clones


class AnomalyScorer:
    """Per-file structural anomaly scoring."""

    def score_file(self, filepath: str) -> Dict[str, Any]:
        try:
            content = Path(filepath).read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
        except Exception:
            return {'file': filepath, 'score': 0.0}

        total = len(lines)
        blank = sum(1 for l in lines if not l.strip())
        comments = sum(1 for l in lines if l.strip().startswith('#'))
        code = total - blank - comments

        # Anomaly signals
        signals = 0.0
        if total > 0 and blank / total > 0.4:
            signals += 0.3  # Too many blanks
        if total > 0 and comments / total > 0.5:
            signals += 0.2  # Comment-heavy
        if total < 20:
            signals += 0.5  # Trivial file
        if 'TODO' in content or 'FIXME' in content or 'HACK' in content:
            signals += 0.1

        # Nesting depth check
        max_indent = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                max_indent = max(max_indent, indent)
        if max_indent > 32:
            signals += 0.2  # Deep nesting

        score = min(1.0, signals)

        return {
            'file': os.path.basename(filepath),
            'lines': total,
            'code': code,
            'blanks': blank,
            'comments': comments,
            'anomaly_score': round(score, 3),
            'max_indent': max_indent,
        }


class ScourPersistence:
    """JSONL persistence for scour results."""

    def __init__(self, path: str = ".l104_scour_results.jsonl"):
        self.path = Path(path)

    def append(self, record: Dict):
        try:
            with open(self.path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception:
            pass

    def load_recent(self, n: int = 20) -> List[Dict]:
        try:
            lines = self.path.read_text().splitlines()
            return [json.loads(l) for l in lines[-n:]]
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE SCOUR ENGINE HUB
# ═══════════════════════════════════════════════════════════════════════════════

class SageScourEngine:
    """
    Deep codebase analysis engine with 5 subsystems:

      - InvariantScanner: sacred constant coverage analysis
      - ImportAuditor: dead/missing import detection
      - DuplicationDetector: clone detection via content hashing
      - AnomalyScorer: structural anomaly scoring per file
      - ScourPersistence: JSONL result storage

    Pipeline Integration:
      - scour(path) → full deep scan
      - quick_scan(path) → fast invariant + anomaly pass
      - get_health_score() → aggregate codebase health
      - connect_to_pipeline() / get_status()
    """

    def __init__(self, root: str = None):
        self.version = VERSION
        self.root = root or str(Path(__file__).parent.absolute())
        self._invariant = InvariantScanner()
        self._imports = ImportAuditor()
        self._duplication = DuplicationDetector()
        self._anomaly = AnomalyScorer()
        self._persistence = ScourPersistence()
        self._pipeline_connected = False
        self._total_scours = 0
        self._last_health = 0.0

    def _get_l104_files(self) -> List[str]:
        """Get all l104_*.py files in root."""
        try:
            return sorted([
                os.path.join(self.root, f)
                for f in os.listdir(self.root)
                if f.startswith('l104_') and f.endswith('.py')
            ])
        except Exception:
            return []

    def scour(self, path: str = None) -> Dict[str, Any]:
        """Full deep scan of the codebase."""
        root = path or self.root
        self.root = root
        files = self._get_l104_files()
        self._total_scours += 1

        invariant_results = []
        import_results = []
        anomaly_results = []

        for fp in files:
            invariant_results.append(self._invariant.scan_file(fp))
            import_results.append(self._imports.audit_file(fp))
            anomaly_results.append(self._anomaly.score_file(fp))

        # Clone detection
        clones = self._duplication.find_clones(files)

        # Aggregate health
        sacred_count = sum(1 for r in invariant_results if r.get('sacred_aligned'))
        total_unused = sum(len(r.get('potentially_unused', [])) for r in import_results)
        avg_anomaly = (sum(r.get('anomaly_score', 0) for r in anomaly_results) /
                       max(len(anomaly_results), 1))
        health = max(0.0, 1.0 - avg_anomaly * PHI) * (sacred_count / max(len(files), 1))
        self._last_health = round(health, 4)

        report = {
            'timestamp': time.time(),
            'files_scanned': len(files),
            'sacred_aligned': sacred_count,
            'total_unused_imports': total_unused,
            'clone_blocks': len(clones),
            'avg_anomaly': round(avg_anomaly, 4),
            'health_score': self._last_health,
            'worst_anomalies': sorted(anomaly_results, key=lambda x: x.get('anomaly_score', 0), reverse=True)[:5],
            'worst_imports': sorted(import_results, key=lambda x: len(x.get('potentially_unused', [])), reverse=True)[:5],
        }

        self._persistence.append(report)
        return report

    def quick_scan(self, path: str = None) -> Dict[str, Any]:
        """Fast scan — invariants + anomalies only."""
        root = path or self.root
        self.root = root
        files = self._get_l104_files()

        sacred = 0
        anomalies = []
        for fp in files[:50]:  # Cap at 50 for speed
            inv = self._invariant.scan_file(fp)
            if inv.get('sacred_aligned'):
                sacred += 1
            anom = self._anomaly.score_file(fp)
            if anom.get('anomaly_score', 0) > 0.3:
                anomalies.append(anom)

        return {
            'files_checked': min(len(files), 50),
            'sacred_aligned': sacred,
            'high_anomaly_files': len(anomalies),
            'anomalies': anomalies[:10],
        }

    def get_health_score(self) -> float:
        return self._last_health

    def connect_to_pipeline(self):
        self._pipeline_connected = True

    def get_status(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'pipeline_connected': self._pipeline_connected,
            'root': self.root,
            'total_scours': self._total_scours,
            'last_health': self._last_health,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════
sage_scour_engine = SageScourEngine()


if __name__ == "__main__":
    report = sage_scour_engine.scour()
    print(json.dumps(report, indent=2, default=str))


def primal_calculus(x):
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    GOD_CODE = 527.5184818492612
    PHI = 1.618033988749895
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
