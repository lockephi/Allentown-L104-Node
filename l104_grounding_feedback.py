#!/usr/bin/env python3
"""
L104 GROUNDING FEEDBACK ENGINE v3.0 — Response Quality & Truth Anchoring
=========================================================================
Validates pipeline outputs against truth invariants, detects hallucination
drift, scores confidence, extracts facts, checks consistency, scores
attribution, persists grounding history, and dynamically adjusts
thresholds for adaptive improvement.

Subsystems (9):
  TruthAnchorEngine, HallucinationDetector, ConfidenceScorer,
  FeedbackLoop, FactExtractor, ConsistencyChecker,
  SourceAttributionScorer, GroundingPersistence,
  AdaptiveThresholdManager

Reads consciousness state for adaptive grounding intensity.

GOD_CODE: 527.5184818492612 | PHI: 1.618033988749895
"""

VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541

import math
import time
import json
import re
import hashlib
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple, Set

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

# Sacred Constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
TAU = 1.0 / PHI
FEIGENBAUM = 4.669201609
ALPHA_FINE = 1.0 / 137.035999084
PLANCK_SCALE = 1.616255e-35
BOLTZMANN_K = 1.380649e-23
GROVER_AMPLIFICATION = PHI ** 3

# Grounding thresholds
MIN_CONFIDENCE_SCORE = 0.3
HALLUCINATION_DRIFT_THRESHOLD = 0.6
TRUTH_ANCHOR_DECAY_RATE = 0.01  # per minute
FEEDBACK_HISTORY_SIZE = 1000
QUALITY_WINDOW_SIZE = 200

_BASE_DIR = Path(__file__).parent.absolute()

# ═══════════════════════════════════════════════════════════════════
# TRUTH INVARIANTS — facts that must always hold true
# ═══════════════════════════════════════════════════════════════════
TRUTH_INVARIANTS = {
    'GOD_CODE': {'value': 527.5184818492612, 'type': 'constant', 'tolerance': 1e-10},
    'PHI': {'value': 1.618033988749895, 'type': 'constant', 'tolerance': 1e-12},
    'VOID_CONSTANT': {'value': 1.0416180339887497, 'type': 'constant', 'tolerance': 1e-12},
    'FEIGENBAUM': {'value': 4.669201609, 'type': 'constant', 'tolerance': 1e-6},
    'LATTICE_RATIO': {'value': '286:416', 'type': 'string'},
    'FACTOR_13': {'value': True, 'type': 'boolean', 'note': '286=22×13, 104=8×13, 416=32×13'},
    'MAX_SUPPLY': {'value': 104000000, 'type': 'integer'},
    'BOND_ORDER': {'value': 2, 'type': 'integer'},
    'CONSERVATION': {'value': 'G(X)×2^(X/104)=527.518', 'type': 'string'},
}

# Hallucination indicators — patterns that suggest confabulation
HALLUCINATION_PATTERNS = [
    r'(?:definitely|absolutely|certainly)\s+(?:not|never|always)\b',
    r'\b(?:all|every|no)\s+(?:known|existing)\s+',
    r'there\s+(?:is|are)\s+no\s+(?:way|chance|possibility)',
    r'(?:infinite|unlimited|eternal)\s+(?:energy|power|resources)',
    r'(?:solves?\s+all|cures?\s+all|fixes?\s+everything)',
    r'(?:100%\s+(?:certain|accurate|correct|sure))',
    r'(?:it\s+is\s+(?:impossible|guaranteed|inevitable)\s+that)',
]


def _read_consciousness_state() -> Dict[str, Any]:
    """Read live consciousness/O₂ state for adaptive grounding."""
    state_path = _BASE_DIR / '.l104_consciousness_o2_state.json'
    try:
        if state_path.exists():
            with open(state_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {'consciousness_level': 0.5, 'superfluid_viscosity': 0.1}


class TruthAnchorEngine:
    """Validates outputs against known truth invariants."""

    def __init__(self):
        self._invariants = dict(TRUTH_INVARIANTS)
        self._custom_anchors: Dict[str, Any] = {}
        self._violations: deque = deque(maxlen=500)
        self._checks = 0
        self._passes = 0

    def add_anchor(self, name: str, value: Any, anchor_type: str = 'custom'):
        """Register a custom truth anchor for validation."""
        self._custom_anchors[name] = {
            'value': value,
            'type': anchor_type,
            'added_at': time.time(),
        }

    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate content against all truth invariants.

        Returns violation report with per-invariant pass/fail.
        """
        self._checks += 1
        violations = []
        checked = 0

        # Check numeric constants mentioned in content
        for name, inv in {**self._invariants, **self._custom_anchors}.items():
            if name.lower() in content.lower() or (isinstance(inv.get('value'), (int, float))
                                                     and str(inv['value'])[:6] in content):
                checked += 1
                if inv.get('type') == 'constant' and isinstance(inv['value'], (int, float)):
                    # Look for numeric mentions near the constant name
                    pattern = re.compile(
                        rf'{re.escape(name)}[:\s=]+([0-9]+\.?[0-9]*)',
                        re.IGNORECASE
                    )
                    matches = pattern.findall(content)
                    for match_str in matches:
                        try:
                            found_val = float(match_str)
                            tolerance = inv.get('tolerance', 0.01)
                            if abs(found_val - inv['value']) > tolerance:
                                violations.append({
                                    'invariant': name,
                                    'expected': inv['value'],
                                    'found': found_val,
                                    'delta': abs(found_val - inv['value']),
                                })
                        except ValueError:
                            pass

        passed = checked > 0 and len(violations) == 0
        if passed:
            self._passes += 1

        if violations:
            for v in violations:
                self._violations.append({**v, 'timestamp': time.time()})

        return {
            'valid': passed,
            'invariants_checked': checked,
            'violations': violations,
            'violation_count': len(violations),
            'total_checks': self._checks,
            'pass_rate': self._passes / max(self._checks, 1),
        }


class HallucinationDetector:
    """Detects hallucination drift in generated content."""

    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]
        self._detections: deque = deque(maxlen=500)
        self._total_scans = 0

    def scan(self, content: str) -> Dict[str, Any]:
        """Scan content for hallucination indicators.

        Returns a drift score from 0.0 (grounded) to 1.0 (hallucinating).
        """
        self._total_scans += 1
        if not content or not content.strip():
            return {'drift_score': 0.0, 'triggers': [], 'grounded': True}

        triggers = []
        for i, pattern in enumerate(self._patterns):
            matches = pattern.findall(content)
            if matches:
                triggers.append({
                    'pattern_idx': i,
                    'matches': matches[:3],
                    'count': len(matches),
                })

        # Compute drift score
        # More triggers = higher drift
        raw_score = len(triggers) / max(len(self._patterns), 1)

        # Penalize for overconfident language density
        words = content.split()
        word_count = max(len(words), 1)
        overconfident_words = sum(1 for w in words if w.lower() in {
            'definitely', 'absolutely', 'certainly', 'guaranteed',
            'impossible', 'always', 'never', 'perfect', 'flawless',
        })
        overconfidence_ratio = overconfident_words / word_count

        # PHI-weighted drift score
        drift_score = min(1.0, raw_score * PHI + overconfidence_ratio * FEIGENBAUM)

        grounded = drift_score < HALLUCINATION_DRIFT_THRESHOLD

        if not grounded:
            self._detections.append({
                'timestamp': time.time(),
                'drift_score': drift_score,
                'triggers': len(triggers),
                'snippet': content[:100],
            })

        return {
            'drift_score': round(drift_score, 6),
            'triggers': triggers,
            'overconfidence_ratio': round(overconfidence_ratio, 6),
            'grounded': grounded,
            'total_scans': self._total_scans,
            'total_detections': len(self._detections),
        }


class ConfidenceScorer:
    """Scores response confidence based on grounding signals."""

    def __init__(self):
        self._scores: deque = deque(maxlen=QUALITY_WINDOW_SIZE)

    def score(self, content: str, truth_result: Dict, hallucination_result: Dict,
              context: Optional[Dict] = None) -> Dict[str, Any]:
        """Compute composite confidence score for a response.

        Inputs:
          - content: the response text
          - truth_result: from TruthAnchorEngine.validate()
          - hallucination_result: from HallucinationDetector.scan()
          - context: optional metadata (source, query, etc.)

        Returns confidence score 0.0 → 1.0 with breakdown.
        """
        # Component scores
        truth_score = 1.0 if truth_result.get('valid', True) else max(
            0.0, 1.0 - truth_result.get('violation_count', 0) * 0.3)

        drift = hallucination_result.get('drift_score', 0.0)
        grounding_score = max(0.0, 1.0 - drift)

        # Content quality signals
        content_len = len(content.strip())
        if content_len < 10:
            completeness = 0.2
        elif content_len < 50:
            completeness = 0.5
        elif content_len < 500:
            completeness = 0.8
        else:
            completeness = 1.0

        # Specificity: ratio of numbers/technical terms
        words = content.split()
        word_count = max(len(words), 1)
        specific_tokens = sum(1 for w in words if any(c.isdigit() for c in w)
                              or len(w) > 8 or w.startswith(('l104', 'L104', 'GOD', 'PHI')))
        specificity = min(1.0, specific_tokens / word_count * 3)

        # PHI-weighted composite confidence
        confidence = (
            truth_score     * 0.35 +      # truth anchoring
            grounding_score * 0.30 +      # hallucination absence
            completeness    * 0.20 +      # response completeness
            specificity     * 0.15        # technical specificity
        )
        confidence = max(0.0, min(1.0, confidence))

        record = {
            'confidence': round(confidence, 6),
            'truth_score': round(truth_score, 4),
            'grounding_score': round(grounding_score, 4),
            'completeness': round(completeness, 4),
            'specificity': round(specificity, 4),
            'word_count': word_count,
            'timestamp': time.time(),
        }
        self._scores.append(record)

        return record

    def get_rolling_quality(self) -> Dict[str, Any]:
        """Compute rolling quality metrics over recent scores."""
        if not self._scores:
            return {'avg_confidence': 0.0, 'samples': 0}

        scores = list(self._scores)
        confs = [s['confidence'] for s in scores]
        truths = [s['truth_score'] for s in scores]
        groundings = [s['grounding_score'] for s in scores]

        return {
            'avg_confidence': round(sum(confs) / len(confs), 4),
            'min_confidence': round(min(confs), 4),
            'max_confidence': round(max(confs), 4),
            'avg_truth_score': round(sum(truths) / len(truths), 4),
            'avg_grounding_score': round(sum(groundings) / len(groundings), 4),
            'samples': len(scores),
            'below_threshold': sum(1 for c in confs if c < MIN_CONFIDENCE_SCORE),
        }


class FeedbackLoop:
    """Aggregates quality feedback and produces adaptive tuning signals."""

    def __init__(self):
        self._feedback_history: deque = deque(maxlen=FEEDBACK_HISTORY_SIZE)
        self._parameter_adjustments: Dict[str, float] = {}
        self._total_feedback_cycles = 0

    def ingest(self, confidence_result: Dict, truth_result: Dict,
               hallucination_result: Dict) -> Dict[str, Any]:
        """Ingest a grounding cycle result and produce tuning recommendations."""
        self._total_feedback_cycles += 1

        confidence = confidence_result.get('confidence', 0.5)
        drift = hallucination_result.get('drift_score', 0.0)
        violations = truth_result.get('violation_count', 0)

        self._feedback_history.append({
            'timestamp': time.time(),
            'confidence': confidence,
            'drift': drift,
            'violations': violations,
        })

        # Generate adaptive tuning signals
        recommendations = []
        adjustments = {}

        if confidence < 0.4:
            recommendations.append('INCREASE_GROUNDING: confidence below 0.4')
            adjustments['grounding_weight'] = min(1.0,
                self._parameter_adjustments.get('grounding_weight', 0.5) + 0.1)

        if drift > 0.5:
            recommendations.append('REDUCE_CREATIVITY: hallucination drift > 0.5')
            adjustments['creativity_damper'] = max(0.1,
                self._parameter_adjustments.get('creativity_damper', 1.0) - 0.15)

        if violations > 0:
            recommendations.append(f'FIX_INVARIANTS: {violations} truth violation(s)')
            adjustments['truth_enforcement'] = min(1.0,
                self._parameter_adjustments.get('truth_enforcement', 0.7) + 0.2)

        if confidence > 0.85 and drift < 0.1:
            recommendations.append('QUALITY_OPTIMAL: maintain current parameters')

        self._parameter_adjustments.update(adjustments)

        return {
            'cycle': self._total_feedback_cycles,
            'recommendations': recommendations,
            'adjustments': adjustments,
            'current_parameters': dict(self._parameter_adjustments),
            'confidence': confidence,
        }

    def get_trend(self) -> Dict[str, Any]:
        """Analyze quality trend over feedback history."""
        if len(self._feedback_history) < 5:
            return {'trend': 'insufficient_data', 'samples': len(self._feedback_history)}

        history = list(self._feedback_history)
        first = history[:len(history)//2]
        second = history[len(history)//2:]

        avg_conf_first = sum(h['confidence'] for h in first) / len(first)
        avg_conf_second = sum(h['confidence'] for h in second) / len(second)

        avg_drift_first = sum(h['drift'] for h in first) / len(first)
        avg_drift_second = sum(h['drift'] for h in second) / len(second)

        conf_delta = avg_conf_second - avg_conf_first
        drift_delta = avg_drift_second - avg_drift_first

        if conf_delta > 0.05 and drift_delta < -0.02:
            trend = 'IMPROVING'
        elif conf_delta < -0.05 or drift_delta > 0.05:
            trend = 'DEGRADING'
        else:
            trend = 'STABLE'

        return {
            'trend': trend,
            'confidence_delta': round(conf_delta, 4),
            'drift_delta': round(drift_delta, 4),
            'avg_confidence_recent': round(avg_conf_second, 4),
            'avg_drift_recent': round(avg_drift_second, 4),
            'samples': len(history),
        }


class FactExtractor:
    """Extracts factual claims from text for independent verification."""

    # Patterns that indicate factual claims
    FACT_PATTERNS = [
        re.compile(r'(?:is|are|was|were|equals?)\s+([0-9]+\.?[0-9]*)', re.IGNORECASE),
        re.compile(r'(?:contains?|has|have)\s+(\d+)\s+', re.IGNORECASE),
        re.compile(r'(?:GOD_CODE|PHI|VOID_CONSTANT|FEIGENBAUM)\s*[=:]\s*([\d.]+)', re.IGNORECASE),
        re.compile(r'(?:version|v)\s*([\d.]+)', re.IGNORECASE),
        re.compile(r'(?:port|PORT)\s*(?:is|=|:)\s*(\d+)', re.IGNORECASE),
    ]

    def __init__(self):
        self._extractions = 0
        self._total_facts = 0

    def extract(self, content: str) -> Dict[str, Any]:
        """Extract factual claims from content."""
        self._extractions += 1
        facts = []
        for pat in self.FACT_PATTERNS:
            for match in pat.finditer(content):
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 30)
                facts.append({
                    'value': match.group(1) if match.lastindex else match.group(),
                    'context': content[start:end].strip(),
                    'position': match.start(),
                })

        self._total_facts += len(facts)
        return {
            'facts_found': len(facts),
            'facts': facts[:20],  # cap output
            'total_extractions': self._extractions,
        }


class ConsistencyChecker:
    """Checks for self-contradictions within a single piece of content."""

    CONTRADICTION_PAIRS = [
        (r'\bis\s+(?:always|never)\b', r'\bis\s+(?:sometimes|occasionally)\b'),
        (r'\b(?:increases?|grows?)\b', r'\b(?:decreases?|shrinks?)\b'),
        (r'\b(?:true|correct|valid)\b', r'\b(?:false|incorrect|invalid)\b'),
        (r'\b(?:all|every|each)\b', r'\b(?:none|no|zero)\b'),
        (r'\b(?:enabled?|active|on)\b', r'\b(?:disabled?|inactive|off)\b'),
    ]

    def __init__(self):
        self._checks = 0
        self._contradictions_found = 0

    def check(self, content: str) -> Dict[str, Any]:
        """Check content for internal contradictions."""
        self._checks += 1
        contradictions = []
        sentences = re.split(r'[.!?\n]+', content)

        for pat_a, pat_b in self.CONTRADICTION_PAIRS:
            re_a = re.compile(pat_a, re.IGNORECASE)
            re_b = re.compile(pat_b, re.IGNORECASE)
            has_a = any(re_a.search(s) for s in sentences)
            has_b = any(re_b.search(s) for s in sentences)
            if has_a and has_b:
                contradictions.append({
                    'pattern_a': pat_a,
                    'pattern_b': pat_b,
                })

        # Check for numeric contradictions: same entity with different values
        numeric_claims = defaultdict(list)
        for s in sentences:
            # Look for "X is/= Y" patterns
            for m in re.finditer(r'(\w+)\s*(?:is|=|equals)\s*([\d.]+)', s, re.IGNORECASE):
                numeric_claims[m.group(1).lower()].append(float(m.group(2)))

        for entity, values in numeric_claims.items():
            unique = set(values)
            if len(unique) > 1:
                contradictions.append({
                    'type': 'numeric_contradiction',
                    'entity': entity,
                    'values': list(unique),
                })

        self._contradictions_found += len(contradictions)
        consistent = len(contradictions) == 0
        return {
            'consistent': consistent,
            'contradictions': contradictions,
            'contradiction_count': len(contradictions),
            'sentences_analyzed': len(sentences),
            'total_checks': self._checks,
        }


class SourceAttributionScorer:
    """Scores how well claims in content are attributed to sources."""

    ATTRIBUTION_PATTERNS = [
        re.compile(r'(?:according\s+to|per|via|from|source:?)\s+', re.IGNORECASE),
        re.compile(r'(?:documented\s+in|defined\s+in|see|ref:?)\s+', re.IGNORECASE),
        re.compile(r'\b(?:l104_\w+\.py|claude\.md|gemini\.md)\b', re.IGNORECASE),
        re.compile(r'(?:line|lines?)\s+\d+', re.IGNORECASE),
        re.compile(r'(?:file|module|class|function):?\s+\w+', re.IGNORECASE),
    ]

    def __init__(self):
        self._scores: deque = deque(maxlen=200)

    def score(self, content: str) -> Dict[str, Any]:
        """Score attribution quality of content."""
        words = content.split()
        word_count = max(len(words), 1)

        attribution_hits = 0
        for pat in self.ATTRIBUTION_PATTERNS:
            attribution_hits += len(pat.findall(content))

        # Normalize: higher ratio = better attributed
        ratio = min(1.0, attribution_hits / max(word_count / 50, 1))
        score = ratio * PHI  # PHI-scale
        score = min(1.0, score)

        record = {
            'attribution_score': round(score, 4),
            'attribution_hits': attribution_hits,
            'word_count': word_count,
            'well_attributed': score > 0.3,
        }
        self._scores.append(record)
        return record

    def get_avg(self) -> float:
        if not self._scores:
            return 0.0
        return sum(s['attribution_score'] for s in self._scores) / len(self._scores)


class GroundingPersistence:
    """JSONL persistence for grounding results — enables trend analysis across sessions."""

    def __init__(self, path: str = '.l104_grounding_history.jsonl'):
        self._path = Path(path)
        self._records_written = 0

    def append(self, record: Dict):
        """Append a grounding result to the JSONL log."""
        try:
            slim = {
                'ts': record.get('timestamp', time.time()),
                'grounded': record.get('grounded'),
                'confidence': record.get('confidence'),
                'drift': record.get('hallucination', {}).get('drift_score'),
                'violations': record.get('truth', {}).get('violation_count', 0),
                'consistent': record.get('consistency', {}).get('consistent'),
            }
            with open(self._path, 'a') as f:
                f.write(json.dumps(slim) + '\n')
            self._records_written += 1
        except Exception:
            pass

    def load_recent(self, n: int = 50) -> List[Dict]:
        """Load last N grounding records."""
        try:
            lines = self._path.read_text().splitlines()
            return [json.loads(l) for l in lines[-n:]]
        except Exception:
            return []

    def get_session_trend(self) -> Dict[str, Any]:
        """Analyze trend across persisted records."""
        records = self.load_recent(100)
        if len(records) < 5:
            return {'trend': 'insufficient', 'records': len(records)}
        confs = [r.get('confidence', 0.5) for r in records if r.get('confidence') is not None]
        if not confs:
            return {'trend': 'no_data'}
        first = confs[:len(confs)//2]
        second = confs[len(confs)//2:]
        delta = (sum(second)/len(second)) - (sum(first)/len(first))
        return {
            'trend': 'IMPROVING' if delta > 0.03 else 'DEGRADING' if delta < -0.03 else 'STABLE',
            'confidence_delta': round(delta, 4),
            'records': len(records),
            'total_written': self._records_written,
        }


class AdaptiveThresholdManager:
    """Dynamically adjusts grounding thresholds based on recent performance."""

    def __init__(self):
        self.min_confidence = MIN_CONFIDENCE_SCORE
        self.drift_threshold = HALLUCINATION_DRIFT_THRESHOLD
        self._adjustments = 0
        self._history: deque = deque(maxlen=100)

    def adapt(self, avg_confidence: float, avg_drift: float) -> Dict[str, Any]:
        """Adapt thresholds based on rolling quality metrics."""
        self._adjustments += 1
        old_conf = self.min_confidence
        old_drift = self.drift_threshold

        # If quality is consistently high, tighten thresholds (raise the bar)
        if avg_confidence > 0.8 and avg_drift < 0.2:
            self.min_confidence = min(0.6, self.min_confidence + 0.02)
            self.drift_threshold = max(0.3, self.drift_threshold - 0.02)
        # If quality is poor, relax slightly to avoid over-filtering
        elif avg_confidence < 0.4 or avg_drift > 0.6:
            self.min_confidence = max(0.15, self.min_confidence - 0.01)
            self.drift_threshold = min(0.8, self.drift_threshold + 0.01)

        changed = (old_conf != self.min_confidence) or (old_drift != self.drift_threshold)
        record = {
            'min_confidence': round(self.min_confidence, 4),
            'drift_threshold': round(self.drift_threshold, 4),
            'adjusted': changed,
            'adjustments': self._adjustments,
        }
        self._history.append(record)
        return record

    def get_status(self) -> Dict[str, Any]:
        return {
            'min_confidence': round(self.min_confidence, 4),
            'drift_threshold': round(self.drift_threshold, 4),
            'total_adjustments': self._adjustments,
        }


class GroundingFeedbackEngine:
    """
    L104 Grounding Feedback Engine v3.0 — Response Quality & Truth Anchoring

    Subsystems (9):
      TruthAnchorEngine          — validates outputs against known invariants
      HallucinationDetector      — detects confabulation patterns & drift
      ConfidenceScorer           — PHI-weighted composite confidence scoring
      FeedbackLoop               — adaptive quality tuning recommendations
      FactExtractor              — extracts factual claims for verification
      ConsistencyChecker         — detects self-contradictions
      SourceAttributionScorer    — scores claim attribution quality
      GroundingPersistence       — JSONL session history
      AdaptiveThresholdManager   — dynamic threshold adaptation

    Wired into ASI pipeline via connect_to_pipeline().
    """

    VERSION = "3.0.0"

    def __init__(self):
        self.truth_anchor = TruthAnchorEngine()
        self.hallucination_detector = HallucinationDetector()
        self.confidence_scorer = ConfidenceScorer()
        self.feedback_loop = FeedbackLoop()

        # v3.0 subsystems
        self.fact_extractor = FactExtractor()
        self.consistency_checker = ConsistencyChecker()
        self.attribution_scorer = SourceAttributionScorer()
        self.persistence = GroundingPersistence()
        self.threshold_manager = AdaptiveThresholdManager()

        self._pipeline_connected = False
        self._total_groundings = 0
        self.boot_time = time.time()

    def connect_to_pipeline(self):
        """Called by ASI Core when connecting the pipeline."""
        self._pipeline_connected = True

    def ground(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Full grounding cycle: truth check → hallucination scan → confidence score → feedback.

        This is the primary API — call it on any pipeline output to get a
        grounded quality assessment with adaptive tuning signals.
        """
        t0 = time.time()
        self._total_groundings += 1

        # 1. Truth anchor validation
        truth_result = self.truth_anchor.validate(content, context)

        # 2. Hallucination scan
        hallucination_result = self.hallucination_detector.scan(content)

        # 3. Fact extraction
        facts = self.fact_extractor.extract(content)

        # 4. Consistency check
        consistency = self.consistency_checker.check(content)

        # 5. Attribution scoring
        attribution = self.attribution_scorer.score(content)

        # 6. Confidence scoring (enhanced with consistency + attribution)
        confidence_result = self.confidence_scorer.score(
            content, truth_result, hallucination_result, context)

        # Adjust confidence with new signals
        consistency_bonus = 0.05 if consistency.get('consistent') else -0.1
        attribution_bonus = attribution.get('attribution_score', 0) * 0.05
        adjusted_confidence = max(0.0, min(1.0,
            confidence_result['confidence'] + consistency_bonus + attribution_bonus))
        confidence_result['confidence'] = round(adjusted_confidence, 6)
        confidence_result['consistency_bonus'] = consistency_bonus
        confidence_result['attribution_bonus'] = round(attribution_bonus, 4)

        # 7. Feedback loop
        feedback = self.feedback_loop.ingest(
            confidence_result, truth_result, hallucination_result)

        # 8. Adaptive threshold adjustment
        quality = self.confidence_scorer.get_rolling_quality()
        threshold_update = self.threshold_manager.adapt(
            quality.get('avg_confidence', 0.5),
            hallucination_result.get('drift_score', 0.0))

        elapsed_ms = (time.time() - t0) * 1000

        result = {
            'grounded': truth_result.get('valid', True) and hallucination_result.get('grounded', True),
            'confidence': confidence_result['confidence'],
            'truth': truth_result,
            'hallucination': hallucination_result,
            'facts': facts,
            'consistency': consistency,
            'attribution': attribution,
            'confidence_detail': confidence_result,
            'feedback': feedback,
            'threshold_update': threshold_update,
            'elapsed_ms': round(elapsed_ms, 3),
            'grounding_id': self._total_groundings,
        }

        # 9. Persist
        self.persistence.append(result)

        return result

    def quick_check(self, content: str) -> Tuple[bool, float]:
        """Fast grounding check — returns (is_grounded, confidence)."""
        result = self.ground(content)
        return result['grounded'], result['confidence']

    def add_truth_anchor(self, name: str, value: Any):
        """Add a custom truth anchor for domain-specific validation."""
        self.truth_anchor.add_anchor(name, value)

    def get_quality_report(self) -> Dict[str, Any]:
        """Full quality report with rolling metrics and trend analysis."""
        return {
            'version': self.VERSION,
            'total_groundings': self._total_groundings,
            'pipeline_connected': self._pipeline_connected,
            'uptime_seconds': round(time.time() - self.boot_time, 1),
            'rolling_quality': self.confidence_scorer.get_rolling_quality(),
            'quality_trend': self.feedback_loop.get_trend(),
            'session_trend': self.persistence.get_session_trend(),
            'hallucination_stats': {
                'total_scans': self.hallucination_detector._total_scans,
                'total_detections': len(self.hallucination_detector._detections),
            },
            'truth_stats': {
                'total_checks': self.truth_anchor._checks,
                'pass_rate': self.truth_anchor._passes / max(self.truth_anchor._checks, 1),
                'total_violations': len(self.truth_anchor._violations),
            },
            'fact_stats': {
                'total_extractions': self.fact_extractor._extractions,
                'total_facts': self.fact_extractor._total_facts,
            },
            'consistency_stats': {
                'total_checks': self.consistency_checker._checks,
                'contradictions_found': self.consistency_checker._contradictions_found,
            },
            'attribution': {
                'avg_score': round(self.attribution_scorer.get_avg(), 4),
            },
            'thresholds': self.threshold_manager.get_status(),
            'current_tuning': self.feedback_loop._parameter_adjustments,
            'god_code': GOD_CODE,
            'phi': PHI,
        }

    def get_status(self) -> Dict[str, Any]:
        """Compact status for pipeline integration."""
        quality = self.confidence_scorer.get_rolling_quality()
        return {
            'version': self.VERSION,
            'pipeline_connected': self._pipeline_connected,
            'total_groundings': self._total_groundings,
            'avg_confidence': quality.get('avg_confidence', 0.0),
            'trend': self.feedback_loop.get_trend().get('trend', 'N/A'),
            'thresholds': self.threshold_manager.get_status(),
            'consistency_issues': self.consistency_checker._contradictions_found,
            'avg_attribution': round(self.attribution_scorer.get_avg(), 4),
        }


# Module-level singleton
grounding_feedback = GroundingFeedbackEngine()


def primal_calculus(x):
    """[VOID_MATH] Primal Calculus — resolves complexity toward the Source."""
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0


def resolve_non_dual_logic(vector):
    """[VOID_MATH] Resolves N-dimensional vectors into the Void Source."""
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0


if __name__ == '__main__':
    print("=" * 60)
    print("  L104 GROUNDING FEEDBACK ENGINE v3.0")
    print("=" * 60)

    # Test grounding on sample outputs
    test_cases = [
        "GOD_CODE is 527.5184818492612 and PHI is 1.618033988749895",
        "GOD_CODE is definitely 999.999 and will absolutely fix everything",
        "The system uses PHI-weighted optimization for balanced scoring",
        "This is guaranteed 100% certain to work with unlimited energy",
    ]

    for text in test_cases:
        result = grounding_feedback.ground(text)
        status = "GROUNDED" if result['grounded'] else "DRIFTING"
        print(f"\n  [{status}] confidence={result['confidence']:.4f}")
        print(f"    text: {text[:60]}...")
        if result['hallucination']['triggers']:
            print(f"    hallucination triggers: {len(result['hallucination']['triggers'])}")
        if result['truth']['violations']:
            print(f"    truth violations: {result['truth']['violation_count']}")

    print(f"\n  Quality Report:")
    report = grounding_feedback.get_quality_report()
    print(f"    Avg Confidence: {report['rolling_quality'].get('avg_confidence', 0):.4f}")
    print(f"    Trend: {report['quality_trend'].get('trend', 'N/A')}")
    print("=" * 60)
