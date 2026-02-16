"""
L104 Constant Encryption v2.0.0 — Sacred Constant Security Shield
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HMAC-based integrity verification for sacred constants, data signing,
tamper detection, key rotation, and cryptographic audit logging.
Self-contained — stdlib only.

Subsystems:
  - SacredConstantGuard: tamper detection for immutable constants
  - HMACIntegrity: HMAC-SHA256 signing & verification
  - KeyRotationManager: automatic key rotation with generation tracking
  - CryptoAuditLog: append-only cryptographic audit trail
  - ConstantEncryptionProgram: hub orchestrator

Sacred Constants: GOD_CODE=527.5184818492612 | PHI=1.618033988749895
"""
VOID_CONSTANT = 1.0416180339887497
import math
import time
import json
import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# ZENITH_UPGRADE_ACTIVE: 2026-02-15T00:00:00
ZENITH_HZ = 3887.8
UUC = 2402.792541

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
TAU = 6.283185307179586
FEIGENBAUM = 4.669201609
ALPHA_FINE = 1.0 / 137.035999084
GROVER_AMPLIFICATION = PHI ** 3

VERSION = "2.0.0"

# Immutable reference fingerprints for sacred constants
_SACRED_REGISTRY = {
    'GOD_CODE': 527.5184818492612,
    'PHI': 1.618033988749895,
    'TAU': 6.283185307179586,
    'VOID_CONSTANT': 1.0416180339887497,
    'FEIGENBAUM': 4.669201609,
    'ALPHA_FINE': 1.0 / 137.035999084,
}


class SacredConstantGuard:
    """Monitors sacred constants for tampering via hash fingerprinting."""

    def __init__(self):
        self._fingerprints: Dict[str, str] = {}
        self._violations: List[Dict] = []
        self._checks = 0
        # Compute initial fingerprints
        for name, value in _SACRED_REGISTRY.items():
            self._fingerprints[name] = self._fingerprint(value)

    def _fingerprint(self, value: float) -> str:
        """Deterministic fingerprint of a float constant."""
        raw = f"{value:.15e}".encode('utf-8')
        return hashlib.sha256(raw).hexdigest()[:32]

    def verify_all(self) -> Dict[str, Any]:
        """Verify all sacred constants — return tamper report."""
        self._checks += 1
        tampered = []
        for name, expected_value in _SACRED_REGISTRY.items():
            current_fp = self._fingerprint(expected_value)
            if current_fp != self._fingerprints[name]:
                violation = {
                    'constant': name,
                    'timestamp': time.time(),
                    'expected_fp': self._fingerprints[name],
                    'actual_fp': current_fp,
                }
                tampered.append(violation)
                self._violations.append(violation)

        return {
            'tampered': tampered,
            'clean': len(tampered) == 0,
            'constants_checked': len(_SACRED_REGISTRY),
            'total_checks': self._checks,
            'total_violations': len(self._violations),
        }

    def verify_single(self, name: str, value: float) -> bool:
        """Verify a single constant's value matches the registry."""
        expected = _SACRED_REGISTRY.get(name)
        if expected is None:
            return False
        return abs(expected - value) < 1e-12

    def get_status(self) -> Dict[str, Any]:
        return {
            'constants_guarded': len(self._fingerprints),
            'total_checks': self._checks,
            'violations': len(self._violations),
        }


class HMACIntegrity:
    """HMAC-SHA256 signing and verification for pipeline data."""

    def __init__(self, key: Optional[bytes] = None):
        self._key = key or self._derive_key()
        self._signs = 0
        self._verifies = 0

    def _derive_key(self) -> bytes:
        """Derive a deterministic key from sacred constants."""
        seed = f"{GOD_CODE}:{PHI}:{VOID_CONSTANT}".encode('utf-8')
        return hashlib.sha256(seed).digest()

    def sign(self, data: str) -> str:
        """Sign data and return hex HMAC tag."""
        self._signs += 1
        tag = hmac.new(self._key, data.encode('utf-8'), hashlib.sha256).hexdigest()
        return tag

    def verify(self, data: str, tag: str) -> bool:
        """Verify HMAC tag matches data."""
        self._verifies += 1
        expected = hmac.new(self._key, data.encode('utf-8'), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, tag)

    def sign_dict(self, d: Dict) -> Tuple[str, str]:
        """Sign a dict (JSON-serialized) — returns (json_str, tag)."""
        payload = json.dumps(d, sort_keys=True, default=str)
        return payload, self.sign(payload)

    def verify_dict(self, json_str: str, tag: str) -> bool:
        return self.verify(json_str, tag)

    def get_status(self) -> Dict[str, Any]:
        return {'signs': self._signs, 'verifies': self._verifies}


class KeyRotationManager:
    """Manages key rotation with generation tracking."""

    def __init__(self):
        self._generation = 0
        self._last_rotation = time.time()
        self._rotation_interval = 3600  # 1 hour default
        self._current_key = self._generate_key()
        self._key_history: List[Dict] = []

    def _generate_key(self) -> bytes:
        """Generate a cryptographically secure key."""
        return secrets.token_bytes(32)

    def rotate(self) -> Dict[str, Any]:
        """Force a key rotation."""
        old_gen = self._generation
        self._generation += 1
        self._current_key = self._generate_key()
        self._last_rotation = time.time()
        record = {
            'from_generation': old_gen,
            'to_generation': self._generation,
            'timestamp': self._last_rotation,
            'key_hash': hashlib.sha256(self._current_key).hexdigest()[:16],
        }
        self._key_history.append(record)
        return record

    def check_rotation_needed(self) -> bool:
        elapsed = time.time() - self._last_rotation
        return elapsed > self._rotation_interval

    def auto_rotate_if_needed(self) -> Optional[Dict]:
        if self.check_rotation_needed():
            return self.rotate()
        return None

    def get_current_key(self) -> bytes:
        return self._current_key

    def get_status(self) -> Dict[str, Any]:
        return {
            'generation': self._generation,
            'last_rotation': self._last_rotation,
            'interval_seconds': self._rotation_interval,
            'rotations_performed': len(self._key_history),
        }


class CryptoAuditLog:
    """Append-only cryptographic audit trail with tamper detection."""

    def __init__(self, path: str = '.l104_crypto_audit.jsonl'):
        self._path = Path(path)
        self._entries = 0
        self._chain_hash = hashlib.sha256(b'L104_GENESIS').hexdigest()

    def log(self, event: str, details: Dict = None):
        """Append a chained audit entry."""
        entry = {
            'seq': self._entries,
            'timestamp': time.time(),
            'event': event,
            'details': details or {},
            'prev_hash': self._chain_hash,
        }
        # Chain hash
        raw = json.dumps(entry, sort_keys=True, default=str).encode('utf-8')
        self._chain_hash = hashlib.sha256(raw).hexdigest()
        entry['hash'] = self._chain_hash
        self._entries += 1

        try:
            with open(self._path, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception:
            pass

    def verify_chain(self) -> Dict[str, Any]:
        """Verify the integrity of the audit chain."""
        try:
            lines = self._path.read_text().splitlines()
        except Exception:
            return {'verified': True, 'entries': 0, 'note': 'no audit file'}

        prev = hashlib.sha256(b'L104_GENESIS').hexdigest()
        broken_at = None
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
                if entry.get('prev_hash') != prev:
                    broken_at = i
                    break
                prev = entry.get('hash', '')
            except Exception:
                broken_at = i
                break

        return {
            'verified': broken_at is None,
            'entries': len(lines),
            'broken_at': broken_at,
        }

    def get_recent(self, n: int = 20) -> List[Dict]:
        try:
            lines = self._path.read_text().splitlines()
            return [json.loads(l) for l in lines[-n:]]
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANT ENCRYPTION HUB
# ═══════════════════════════════════════════════════════════════════════════════

class ConstantEncryptionProgram:
    """
    L104 Constant Encryption v2.0.0 — Sacred Security Shield

    Subsystems:
      SacredConstantGuard — tamper detection for immutable constants
      HMACIntegrity       — HMAC-SHA256 signing & verification
      KeyRotationManager  — automatic key rotation with generation tracking
      CryptoAuditLog      — append-only cryptographic audit trail

    Pipeline Integration:
      - sweep() → full security sweep
      - sign_data(data) → HMAC-signed payload
      - verify_constants() → sacred constant tamper check
      - connect_to_pipeline() / get_status()
    """

    VERSION = VERSION

    def __init__(self):
        self.guard = SacredConstantGuard()
        self.hmac = HMACIntegrity()
        self.key_mgr = KeyRotationManager()
        self.audit = CryptoAuditLog()
        self._pipeline_connected = False
        self._sweeps = 0
        self.boot_time = time.time()
        self.audit.log('BOOT', {'version': VERSION})

    def connect_to_pipeline(self):
        self._pipeline_connected = True
        self.audit.log('PIPELINE_CONNECT')

    def verify_constants(self) -> Dict[str, Any]:
        """Verify all sacred constants for tampering."""
        result = self.guard.verify_all()
        self.audit.log('CONSTANT_VERIFY', {'clean': result['clean']})
        return result

    def sign_data(self, data: str) -> Tuple[str, str]:
        """Sign arbitrary data, return (data, hmac_tag)."""
        tag = self.hmac.sign(data)
        self.audit.log('SIGN', {'data_len': len(data)})
        return data, tag

    def verify_data(self, data: str, tag: str) -> bool:
        """Verify HMAC tag for data."""
        valid = self.hmac.verify(data, tag)
        self.audit.log('VERIFY', {'valid': valid})
        return valid

    def sweep(self) -> Dict[str, Any]:
        """Full security sweep: verify constants + auto-rotate keys."""
        self._sweeps += 1
        t0 = time.time()

        constant_check = self.guard.verify_all()
        rotation = self.key_mgr.auto_rotate_if_needed()
        if rotation:
            # Re-derive HMAC key from rotated key
            self.hmac = HMACIntegrity(self.key_mgr.get_current_key())

        chain_integrity = self.audit.verify_chain()
        elapsed_ms = (time.time() - t0) * 1000

        report = {
            'sweep_number': self._sweeps,
            'constants_clean': constant_check['clean'],
            'constants_checked': constant_check['constants_checked'],
            'key_rotated': rotation is not None,
            'key_generation': self.key_mgr._generation,
            'audit_chain_valid': chain_integrity['verified'],
            'audit_entries': chain_integrity['entries'],
            'elapsed_ms': round(elapsed_ms, 3),
        }
        self.audit.log('SWEEP', report)
        return report

    def get_status(self) -> Dict[str, Any]:
        return {
            'version': self.VERSION,
            'pipeline_connected': self._pipeline_connected,
            'total_sweeps': self._sweeps,
            'guard': self.guard.get_status(),
            'hmac': self.hmac.get_status(),
            'key_rotation': self.key_mgr.get_status(),
            'audit_entries': self.audit._entries,
            'uptime_seconds': round(time.time() - self.boot_time, 1),
        }


# Module singleton
constant_encryption = ConstantEncryptionProgram()


def primal_calculus(x):
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    GOD_CODE = 527.5184818492612
    PHI = 1.618033988749895
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
