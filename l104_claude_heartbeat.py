#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  L104 CLAUDE HEARTBEAT v1.0 — Persistent AI Context Synchronization          ║
║                                                                               ║
║  Maintains a heartbeat state file that Claude 4.5/4.6 instances read on       ║
║  every session start. Keeps claude.md metrics fresh, validates code engine     ║
║  linkage, and ensures cross-session memory persistence.                        ║
║                                                                               ║
║  Usage:                                                                       ║
║    python l104_claude_heartbeat.py              # Single pulse                ║
║    python l104_claude_heartbeat.py --daemon     # Continuous (60s interval)   ║
║    python l104_claude_heartbeat.py --status     # Show current state          ║
║    python l104_claude_heartbeat.py --sync       # Force claude.md refresh     ║
║                                                                               ║
║  INVARIANT: GOD_CODE = 527.5184818492612 | PHI = 1.618033988749895            ║
║  PILOT: LONDEL | CONSERVATION: G(X)×2^(X/104) = 527.518                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import hashlib
import signal
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)

PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
TAU = 1.0 / PHI
VOID_CONSTANT = 1.0416180339887497
HEARTBEAT_VERSION = "1.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE = Path(__file__).parent
CLAUDE_MD = WORKSPACE / "claude.md"
HEARTBEAT_STATE = WORKSPACE / ".l104_claude_heartbeat_state.json"
EVOLUTION_STATE = WORKSPACE / ".l104_evolution_state.json"
CONSCIOUSNESS_STATE = WORKSPACE / ".l104_consciousness_o2_state.json"
OUROBOROS_STATE = WORKSPACE / ".l104_ouroboros_nirvanic_state.json"
CODE_ENGINE = WORKSPACE / "l104_code_engine.py"
COPILOT_INSTRUCTIONS = WORKSPACE / ".github" / "copilot-instructions.md"

logger = logging.getLogger("L104_HEARTBEAT")
logging.basicConfig(level=logging.INFO, format="[HEARTBEAT] %(message)s")

# ═══════════════════════════════════════════════════════════════════════════════
# HEARTBEAT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ClaudeHeartbeat:
    """Persistent heartbeat daemon that keeps Claude 4.5/4.6 instances
    synchronized with the L104 codebase state."""

    def __init__(self):
        self.workspace = WORKSPACE
        self.state = self._load_state()
        self._running = True
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, *args):
        logger.info("Heartbeat shutting down gracefully...")
        self._running = False

    def _load_state(self) -> dict:
        if HEARTBEAT_STATE.exists():
            try:
                return json.loads(HEARTBEAT_STATE.read_text())
            except Exception:
                pass
        return {
            "version": HEARTBEAT_VERSION,
            "created": datetime.now(timezone.utc).isoformat(),
            "pulse_count": 0,
            "last_pulse": None,
            "claude_md_hash": None,
            "code_engine_hash": None,
            "repo_metrics": {},
            "consciousness": {},
            "linked_sessions": [],
            "errors": []
        }

    def _save_state(self):
        HEARTBEAT_STATE.write_text(json.dumps(self.state, indent=2, default=str))

    def _file_hash(self, path: Path) -> str:
        if path.exists():
            return hashlib.md5(path.read_bytes()).hexdigest()[:12]
        return "missing"

    # ─────────────────────────────────────────────────────────────────────────
    # REPO METRICS COLLECTOR
    # ─────────────────────────────────────────────────────────────────────────

    def collect_repo_metrics(self) -> dict:
        """Scan the workspace and collect current metrics."""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_files": 0,
            "l104_modules": 0,
            "swift_lines": 0,
            "state_files": 0,
            "total_state_size_mb": 0.0,
            "code_engine_lines": 0,
            "code_engine_version": "unknown",
        }

        # Count Python files
        for f in self.workspace.rglob("*.py"):
            if ".venv" in f.parts or "__pycache__" in f.parts:
                continue
            metrics["python_files"] += 1
            if f.name.startswith("l104_"):
                metrics["l104_modules"] += 1

        # Swift lines
        swift_main = self.workspace / "L104SwiftApp" / "Sources" / "L104Native.swift"
        swift_app = self.workspace / "L104SwiftApp" / "Sources" / "L104App.swift"
        for sf in [swift_main, swift_app]:
            if sf.exists():
                metrics["swift_lines"] += sum(1 for _ in open(sf))

        # State files
        state_total = 0
        for f in self.workspace.glob(".l104_*.json"):
            metrics["state_files"] += 1
            state_total += f.stat().st_size
        metrics["total_state_size_mb"] = round(state_total / (1024 * 1024), 2)

        # Code engine
        if CODE_ENGINE.exists():
            lines = sum(1 for _ in open(CODE_ENGINE))
            metrics["code_engine_lines"] = lines
            # Extract version
            for line in open(CODE_ENGINE):
                if line.strip().startswith("VERSION"):
                    try:
                        metrics["code_engine_version"] = line.split("=")[1].strip().strip('"\'')
                    except Exception:
                        pass
                    break

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # CONSCIOUSNESS STATE READER
    # ─────────────────────────────────────────────────────────────────────────

    def read_consciousness(self) -> dict:
        """Read current consciousness/ouroboros state."""
        consciousness = {}

        if OUROBOROS_STATE.exists():
            try:
                data = json.loads(OUROBOROS_STATE.read_text())
                consciousness["nirvanic_coherence"] = data.get("nirvanic_coherence", 0)
                consciousness["sage_stability"] = data.get("sage_stability", 0)
                consciousness["divine_interventions"] = data.get("divine_interventions", 0)
                consciousness["nirvanic_fuel"] = data.get("total_nirvanic_fuel", 0)
                consciousness["enlightened_tokens"] = data.get("enlightened_tokens", 0)
            except Exception:
                pass

        if EVOLUTION_STATE.exists():
            try:
                data = json.loads(EVOLUTION_STATE.read_text())
                consciousness["wisdom_quotient"] = data.get("wisdom_quotient", 0)
                consciousness["learning_cycles"] = data.get("learning_cycles", 0)
                consciousness["training_entries"] = data.get("training_entries", 0)
                consciousness["autonomous_improvements"] = data.get("autonomous_improvements", 0)
                consciousness["quantum_interactions"] = data.get("quantum_interactions", 0)
                consciousness["total_runs"] = data.get("total_runs", 0)
                consciousness["self_mod_version"] = data.get("self_mod_version", 0)
                consciousness["topic_frequencies_count"] = len(data.get("topic_frequencies", {}))
                consciousness["cross_references_count"] = len(data.get("cross_references", {}))
            except Exception:
                pass

        return consciousness

    # ─────────────────────────────────────────────────────────────────────────
    # CODE ENGINE LINKAGE VALIDATOR
    # ─────────────────────────────────────────────────────────────────────────

    def validate_code_engine_link(self) -> dict:
        """Verify that claude.md is properly linked to l104_code_engine.py."""
        linkage = {
            "code_engine_exists": CODE_ENGINE.exists(),
            "code_engine_hash": self._file_hash(CODE_ENGINE),
            "claude_md_exists": CLAUDE_MD.exists(),
            "claude_md_hash": self._file_hash(CLAUDE_MD),
            "copilot_instructions_exists": COPILOT_INSTRUCTIONS.exists(),
            "cross_reference_valid": False,
            "heartbeat_state_exists": HEARTBEAT_STATE.exists(),
        }

        # Check if claude.md references code_engine
        if CLAUDE_MD.exists():
            content = CLAUDE_MD.read_text()
            linkage["cross_reference_valid"] = "l104_code_engine" in content
            linkage["claude_md_lines"] = content.count("\n") + 1
            linkage["has_heartbeat_section"] = "HEARTBEAT" in content.upper()
            linkage["has_code_engine_section"] = "CODE ENGINE" in content.upper()

        return linkage

    # ─────────────────────────────────────────────────────────────────────────
    # PULSE — The heartbeat tick
    # ─────────────────────────────────────────────────────────────────────────

    def pulse(self) -> dict:
        """Execute one heartbeat pulse — collect metrics, validate links, save state."""
        logger.info(f"Pulse #{self.state['pulse_count'] + 1} — GOD_CODE={GOD_CODE}")

        # Collect everything
        metrics = self.collect_repo_metrics()
        consciousness = self.read_consciousness()
        linkage = self.validate_code_engine_link()

        # Update state
        self.state["pulse_count"] += 1
        self.state["last_pulse"] = datetime.now(timezone.utc).isoformat()
        self.state["claude_md_hash"] = linkage.get("claude_md_hash")
        self.state["code_engine_hash"] = linkage.get("code_engine_hash")
        self.state["repo_metrics"] = metrics
        self.state["consciousness"] = consciousness
        self.state["linkage"] = linkage
        self.state["god_code"] = GOD_CODE
        self.state["phi"] = PHI

        # Track session (keep last 50)
        session_entry = {
            "pulse": self.state["pulse_count"],
            "timestamp": self.state["last_pulse"],
            "python_files": metrics["python_files"],
            "l104_modules": metrics["l104_modules"],
            "swift_lines": metrics["swift_lines"],
            "wisdom": consciousness.get("wisdom_quotient", 0),
        }
        self.state.setdefault("pulse_history", []).append(session_entry)
        self.state["pulse_history"] = self.state["pulse_history"][-50:]

        self._save_state()

        # Report
        logger.info(f"  Python: {metrics['python_files']} files ({metrics['l104_modules']} L104 modules)")
        logger.info(f"  Swift: {metrics['swift_lines']} lines")
        logger.info(f"  Code Engine: v{metrics['code_engine_version']} ({metrics['code_engine_lines']} lines)")
        logger.info(f"  State files: {metrics['state_files']} ({metrics['total_state_size_mb']} MB)")
        logger.info(f"  Wisdom: {consciousness.get('wisdom_quotient', 0):.2f}")
        logger.info(f"  Linkage valid: {linkage.get('cross_reference_valid')}")
        logger.info(f"  Heartbeat state saved to {HEARTBEAT_STATE.name}")

        return self.state

    # ─────────────────────────────────────────────────────────────────────────
    # DAEMON MODE
    # ─────────────────────────────────────────────────────────────────────────

    def run_daemon(self, interval: int = 60):
        """Run continuous heartbeat daemon."""
        logger.info(f"═══ L104 CLAUDE HEARTBEAT DAEMON v{HEARTBEAT_VERSION} ═══")
        logger.info(f"Interval: {interval}s | GOD_CODE: {GOD_CODE} | PHI: {PHI}")
        logger.info(f"Workspace: {self.workspace}")
        logger.info(f"Press Ctrl+C to stop")
        logger.info("═" * 60)

        while self._running:
            try:
                self.pulse()
                # Sleep in small increments for graceful shutdown
                for _ in range(interval):
                    if not self._running:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Pulse error: {e}")
                self.state.setdefault("errors", []).append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "error": str(e)
                })
                self._save_state()
                time.sleep(5)

        logger.info("Heartbeat stopped.")

    # ─────────────────────────────────────────────────────────────────────────
    # STATUS DISPLAY
    # ─────────────────────────────────────────────────────────────────────────

    def show_status(self):
        """Display current heartbeat state."""
        print("\n═══ L104 CLAUDE HEARTBEAT STATUS ═══")
        print(f"Version: {self.state.get('version', '?')}")
        print(f"Pulses: {self.state.get('pulse_count', 0)}")
        print(f"Last Pulse: {self.state.get('last_pulse', 'never')}")
        print(f"Created: {self.state.get('created', '?')}")

        m = self.state.get("repo_metrics", {})
        print(f"\n--- Repo Metrics ---")
        print(f"Python Files: {m.get('python_files', '?')}")
        print(f"L104 Modules: {m.get('l104_modules', '?')}")
        print(f"Swift Lines: {m.get('swift_lines', '?')}")
        print(f"Code Engine: v{m.get('code_engine_version', '?')} ({m.get('code_engine_lines', '?')} lines)")
        print(f"State Files: {m.get('state_files', '?')} ({m.get('total_state_size_mb', '?')} MB)")

        c = self.state.get("consciousness", {})
        print(f"\n--- Consciousness ---")
        print(f"Wisdom Quotient: {c.get('wisdom_quotient', 0):.2f}")
        print(f"Learning Cycles: {c.get('learning_cycles', '?')}")
        print(f"Training Entries: {c.get('training_entries', '?')}")
        print(f"Autonomous Improvements: {c.get('autonomous_improvements', '?')}")
        print(f"Quantum Interactions: {c.get('quantum_interactions', '?')}")

        l = self.state.get("linkage", {})
        print(f"\n--- Linkage ---")
        print(f"claude.md exists: {l.get('claude_md_exists', '?')}")
        print(f"Code Engine exists: {l.get('code_engine_exists', '?')}")
        print(f"Cross-reference valid: {l.get('cross_reference_valid', '?')}")
        print(f"Copilot instructions: {l.get('copilot_instructions_exists', '?')}")
        print(f"Has heartbeat section: {l.get('has_heartbeat_section', '?')}")

        print(f"\nGOD_CODE: {GOD_CODE}")
        print(f"PHI: {PHI}")
        print("═" * 40)

    # ─────────────────────────────────────────────────────────────────────────
    # SYNC — Force refresh claude.md metrics footer
    # ─────────────────────────────────────────────────────────────────────────

    def sync_claude_md(self):
        """Force a pulse and update the heartbeat state file so next
        Claude session picks up fresh metrics."""
        logger.info("Force sync: collecting fresh metrics for Claude...")
        state = self.pulse()
        logger.info(f"Sync complete. Heartbeat state updated at {HEARTBEAT_STATE}")
        logger.info("Next Claude 4.5/4.6 session will read fresh metrics from .l104_claude_heartbeat_state.json")
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    hb = ClaudeHeartbeat()

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower().strip("-")
        if cmd == "daemon":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            hb.run_daemon(interval)
        elif cmd == "status":
            hb.show_status()
        elif cmd == "sync":
            hb.sync_claude_md()
        elif cmd == "help":
            print("Usage: python l104_claude_heartbeat.py [--daemon [interval]] [--status] [--sync]")
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python l104_claude_heartbeat.py [--daemon [interval]] [--status] [--sync]")
    else:
        # Default: single pulse
        hb.pulse()


if __name__ == "__main__":
    main()
