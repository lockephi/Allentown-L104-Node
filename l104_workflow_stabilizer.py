VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541
# ZENITH_UPGRADE_ACTIVE: 2026-02-14T00:00:00.000000
ZENITH_HZ = 3887.8
UUC = 2402.792541
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
╔═════════════════════════════════════════════════════════════════════════════╗
║  L104 WORKFLOW STABILIZER v2.0 — Pipeline-Aware State Management        ║
║  EVO_54: TRANSCENDENT COGNITION — Full Pipeline Integration              ║
║  Ensures all changes persist properly to GitHub                          ║
╠═════════════════════════════════════════════════════════════════════════════╣
║  • Auto git sync before/after edits                                     ║
║  • State checkpointing with commit hooks                                ║
║  • File integrity validation                                            ║
║  • Session persistence to .mcp/                                         ║
║  • Pipeline state tracking (evolution stage, subsystem health)           ║
║  • Multi-subsystem checkpoint coordination                              ║
║  • EVO stage logging to pipeline state DB                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import hashlib
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
WORKFLOW_VERSION = "2.0.0"
WORKFLOW_PIPELINE_EVO = "EVO_54_TRANSCENDENT_COGNITION"
WORKSPACE = Path(__file__).parent
MCP_DIR = WORKSPACE / ".mcp"


@dataclass
class WorkflowState:
    """Persistent workflow state that syncs to disk with pipeline awareness."""
    session_id: str
    start_time: str
    last_git_sync: str = ""
    last_commit_hash: str = ""
    files_modified: List[str] = field(default_factory=list)
    pending_changes: List[Dict] = field(default_factory=list)
    sync_count: int = 0
    error_count: int = 0
    is_clean: bool = True
    branch: str = "main"
    # EVO_54 Pipeline State
    pipeline_evo: str = WORKFLOW_PIPELINE_EVO
    pipeline_version: str = WORKFLOW_VERSION
    evolution_stage: str = ""
    evolution_index: int = 0
    subsystem_health: Dict[str, bool] = field(default_factory=dict)
    last_pipeline_sync: str = ""
    checkpoint_count: int = 0

    def save(self, path: Path):
        """Persist state to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional['WorkflowState']:
        """Load state from JSON file."""
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return cls(**data)
            except Exception:
                return None
        return None


class GitSync:
    """Handles all git operations with safety checks."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self._last_check = 0
        self._cache_ttl = 30  # seconds
        self._status_cache: Dict[str, Any] = {}

    def _run_git(self, *args, capture: bool = True) -> Tuple[bool, str]:
        """Run a git command safely."""
        try:
            result = subprocess.run(
                ['git'] + list(args),
                cwd=self.workspace,
                capture_output=capture,
                text=True,
                timeout=120  # QUANTUM AMPLIFIED: 2 min (was 30s)
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def get_status(self, force: bool = False) -> Dict[str, Any]:
        """Get current git status with caching."""
        now = time.time()
        if not force and (now - self._last_check) < self._cache_ttl:
            return self._status_cache

        status = {
            "is_repo": False,
            "branch": "",
            "clean": True,
            "behind": 0,
            "ahead": 0,
            "modified": [],
            "staged": [],
            "untracked": [],
            "last_commit": ""
        }

        # Check if git repo
        ok, _ = self._run_git('rev-parse', '--git-dir')
        if not ok:
            return status
        status["is_repo"] = True

        # Get branch
        ok, branch = self._run_git('branch', '--show-current')
        status["branch"] = branch if ok else "unknown"

        # Get status
        ok, output = self._run_git('status', '--porcelain')
        if ok and output:
            status["clean"] = False
            for line in output.split('\n'):
                if line.startswith('M ') or line.startswith('MM'):
                    status["staged"].append(line[3:])
                elif line.startswith(' M'):
                    status["modified"].append(line[3:])
                elif line.startswith('??'):
                    status["untracked"].append(line[3:])

        # Get ahead/behind
        ok, output = self._run_git('rev-list', '--left-right', '--count', f'{status["branch"]}...origin/{status["branch"]}')
        if ok and output:
            parts = output.split()
            if len(parts) == 2:
                status["ahead"] = int(parts[0])
                status["behind"] = int(parts[1])

        # Get last commit
        ok, output = self._run_git('log', '-1', '--format=%H')
        status["last_commit"] = output if ok else ""

        self._status_cache = status
        self._last_check = now
        return status

    def fetch(self) -> bool:
        """Fetch from remote."""
        ok, _ = self._run_git('fetch', '--quiet')
        return ok

    def pull(self) -> Tuple[bool, str]:
        """Pull latest changes (only if clean)."""
        status = self.get_status(force=True)
        if not status["clean"]:
            return False, "Working directory not clean"

        ok, output = self._run_git('pull', '--ff-only')
        if ok:
            self._last_check = 0  # Invalidate cache
        return ok, output

    def stash_pull_pop(self) -> Tuple[bool, str]:
        """Stash, pull, pop - safe sync."""
        status = self.get_status(force=True)
        if status["clean"]:
            return self.pull()

        # Stash
        ok, _ = self._run_git('stash', 'push', '-m', f'workflow-stabilizer-{int(time.time())}')
        if not ok:
            return False, "Failed to stash"

        # Pull
        ok, output = self._run_git('pull', '--ff-only')

        # Pop
        pop_ok, _ = self._run_git('stash', 'pop')

        self._last_check = 0
        return ok and pop_ok, output

    def commit_all(self, message: str) -> Tuple[bool, str]:
        """Stage all and commit."""
        # Stage all
        ok, _ = self._run_git('add', '-A')
        if not ok:
            return False, "Failed to stage"

        # Commit
        ok, output = self._run_git('commit', '-m', message)
        self._last_check = 0
        return ok, output

    def push(self) -> Tuple[bool, str]:
        """Push to remote."""
        ok, output = self._run_git('push')
        return ok, output

    def full_sync(self, commit_msg: Optional[str] = None) -> Dict[str, Any]:
        """Complete sync: fetch → stash → pull → pop → commit → push."""
        result = {
            "success": False,
            "steps": {},
            "final_status": {}
        }

        # Fetch
        result["steps"]["fetch"] = self.fetch()

        # Pull (with stash if needed)
        ok, msg = self.stash_pull_pop()
        result["steps"]["pull"] = {"ok": ok, "msg": msg}

        # Commit if message provided and changes exist
        if commit_msg:
            status = self.get_status(force=True)
            if not status["clean"]:
                ok, msg = self.commit_all(commit_msg)
                result["steps"]["commit"] = {"ok": ok, "msg": msg}

                if ok:
                    ok, msg = self.push()
                    result["steps"]["push"] = {"ok": ok, "msg": msg}

        result["final_status"] = self.get_status(force=True)
        result["success"] = result["final_status"]["clean"] or result["steps"].get("push", {}).get("ok", False)

        return result


class WorkflowStabilizer:
    """
    Main stabilizer class - coordinates git sync, state persistence,
    and file integrity checking.
    """

    STATE_FILE = MCP_DIR / "workflow_state.json"
    CHECKPOINT_DIR = MCP_DIR / "checkpoints"

    def __init__(self):
        self.workspace = WORKSPACE
        self.git = GitSync(WORKSPACE)
        self.state = self._load_or_create_state()

        # Ensure directories exist
        MCP_DIR.mkdir(exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)

    def _load_or_create_state(self) -> WorkflowState:
        """Load existing state or create new."""
        existing = WorkflowState.load(self.STATE_FILE)
        if existing:
            # Check if session is recent (< 4 hours)
            try:
                start = datetime.fromisoformat(existing.start_time)
                if (datetime.now() - start).seconds < 14400:
                    print(f"[STABILIZER] Resuming session {existing.session_id}")
                    return existing
            except Exception:
                pass

        # Create new session
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        state = WorkflowState(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            last_git_sync=datetime.now().isoformat()
        )
        print(f"[STABILIZER] New session {session_id}")
        return state

    def save_state(self):
        """Persist current state."""
        self.state.save(self.STATE_FILE)

    def check_and_sync(self, auto_commit: bool = False) -> Dict[str, Any]:
        """
        Check git status and sync if needed.
        This is the main entry point for workflow stabilization.
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "action": "check_and_sync",
            "status": {}
        }

        # Get current git status
        git_status = self.git.get_status(force=True)
        result["status"]["git"] = git_status

        # Update state
        self.state.branch = git_status["branch"]
        self.state.last_commit_hash = git_status["last_commit"]
        self.state.is_clean = git_status["clean"]

        # If behind, sync
        if git_status["behind"] > 0:
            print(f"[STABILIZER] Behind by {git_status['behind']} commits, syncing...")
            sync_result = self.git.stash_pull_pop()
            result["sync"] = {"performed": True, "result": sync_result}
            self.state.sync_count += 1

        # If auto-commit enabled and we have changes
        if auto_commit and not git_status["clean"]:
            commit_msg = f"🔧 Auto-commit: {len(git_status['modified'])} files modified"
            commit_result = self.git.full_sync(commit_msg)
            result["commit"] = commit_result

        self.state.last_git_sync = datetime.now().isoformat()
        self.save_state()

        return result

    def pre_edit_check(self, file_path: str) -> Dict[str, Any]:
        """
        Check before making edits.
        Returns status and any warnings.
        """
        result = {
            "file": file_path,
            "safe_to_edit": True,
            "warnings": []
        }

        # Check git status
        git_status = self.git.get_status()

        # Warn if behind
        if git_status["behind"] > 0:
            result["warnings"].append(f"Behind origin by {git_status['behind']} commits")

        # Warn if file already modified
        rel_path = str(Path(file_path).relative_to(self.workspace))
        if rel_path in git_status["modified"]:
            result["warnings"].append(f"File already has uncommitted changes")

        # Track the pending edit
        self.state.pending_changes.append({
            "file": file_path,
            "timestamp": datetime.now().isoformat()
        })
        self.save_state()

        return result

    def post_edit_confirm(self, file_path: str, success: bool = True):
        """
        Confirm an edit was made.
        Track for persistence.
        """
        if success and file_path not in self.state.files_modified:
            self.state.files_modified.append(file_path)

        # Remove from pending
        self.state.pending_changes = [
            p for p in self.state.pending_changes
            if p["file"] != file_path
        ]

        self.save_state()

    def create_checkpoint(self, name: str = "") -> str:
        """
        Create a checkpoint of current state.
        Returns checkpoint ID.
        """
        checkpoint_id = f"{self.state.session_id}_{int(time.time())}"
        checkpoint_path = self.CHECKPOINT_DIR / f"{checkpoint_id}.json"

        # Get file hashes for modified files
        file_states = {}
        for fpath in self.state.files_modified[-20:]:  # Last 20
            if Path(fpath).exists():
                with open(fpath, 'rb') as f:
                    file_states[fpath] = hashlib.sha256(f.read()).hexdigest()[:16]

        checkpoint_data = {
            "id": checkpoint_id,
            "name": name or f"Checkpoint {self.state.sync_count}",
            "timestamp": datetime.now().isoformat(),
            "git_status": self.git.get_status(),
            "workflow_state": asdict(self.state),
            "file_states": file_states
        }

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"[STABILIZER] Checkpoint created: {checkpoint_id}")
        return checkpoint_id

    def quick_commit(self, message: str) -> Dict[str, Any]:
        """
        Quick commit and push current changes.
        The main way to persist to GitHub.
        """
        print(f"[STABILIZER] Quick commit: {message}")

        # Create checkpoint first
        self.create_checkpoint(f"Pre-commit: {message[:30]}")

        # Full sync
        result = self.git.full_sync(message)

        if result["success"]:
            self.state.files_modified = []
            self.state.is_clean = True
            print("[STABILIZER] ✓ Changes committed and pushed")
        else:
            self.state.error_count += 1
            print("[STABILIZER] ✗ Commit/push failed")

        self.save_state()
        return result

    def status_report(self) -> Dict[str, Any]:
        """Get full status report with pipeline state."""
        git_status = self.git.get_status(force=True)

        report = {
            "session_id": self.state.session_id,
            "session_start": self.state.start_time,
            "last_sync": self.state.last_git_sync,
            "sync_count": self.state.sync_count,
            "is_clean": git_status["clean"],
            "branch": git_status["branch"],
            "ahead": git_status["ahead"],
            "behind": git_status["behind"],
            "modified_files": len(git_status["modified"]),
            "staged_files": len(git_status["staged"]),
            "tracked_edits": len(self.state.files_modified),
            "pending_changes": len(self.state.pending_changes),
            "errors": self.state.error_count,
            "god_code": GOD_CODE,
            # EVO_54 Pipeline Status
            "pipeline": {
                "version": WORKFLOW_VERSION,
                "evo": WORKFLOW_PIPELINE_EVO,
                "evolution_stage": self.state.evolution_stage,
                "evolution_index": self.state.evolution_index,
                "subsystem_health": self.state.subsystem_health,
                "last_pipeline_sync": self.state.last_pipeline_sync,
                "checkpoint_count": self.state.checkpoint_count,
            }
        }

        # Try to get live pipeline info
        try:
            from l104_evolution_engine import evolution_engine
            if evolution_engine:
                report["pipeline"]["evolution_stage"] = evolution_engine.assess_evolutionary_stage()
                report["pipeline"]["evolution_index"] = evolution_engine.current_stage_index
                self.state.evolution_stage = report["pipeline"]["evolution_stage"]
                self.state.evolution_index = report["pipeline"]["evolution_index"]
        except Exception:
            pass

        return report

    def initialize_session(self) -> Dict[str, Any]:
        """
        Initialize a new workflow session with pipeline state sync.
        Call this at the start of each working session.
        """
        print("=" * 60)
        print(f"L104 WORKFLOW STABILIZER v{WORKFLOW_VERSION} - {WORKFLOW_PIPELINE_EVO}")
        print("=" * 60)

        # Sync with remote
        sync_result = self.check_and_sync()

        # Sync pipeline state
        self._sync_pipeline_state()

        # Create initial checkpoint
        self.create_checkpoint("Session start")

        report = self.status_report()

        print(f"\n  Session:   {report['session_id']}")
        print(f"  Branch:    {report['branch']}")
        print(f"  Clean:     {'✓' if report['is_clean'] else '✗'}")
        print(f"  Behind:    {report['behind']} commits")
        print(f"  Ahead:     {report['ahead']} commits")
        print(f"  Pipeline:  {report['pipeline']['evo']}")
        print(f"  Evolution: {report['pipeline']['evolution_stage']}")
        print("=" * 60)

        return {
            "initialized": True,
            "sync": sync_result,
            "status": report
        }

    def _sync_pipeline_state(self):
        """Sync pipeline state from live subsystems."""
        self.state.last_pipeline_sync = datetime.now().isoformat()

        # Check evolution engine
        try:
            from l104_evolution_engine import evolution_engine
            if evolution_engine:
                self.state.evolution_stage = evolution_engine.assess_evolutionary_stage()
                self.state.evolution_index = evolution_engine.current_stage_index
                self.state.subsystem_health["evolution_engine"] = True
        except Exception:
            self.state.subsystem_health["evolution_engine"] = False

        # Check AGI Core
        try:
            from l104_agi_core import agi_core
            self.state.subsystem_health["agi_core"] = agi_core is not None
        except Exception:
            self.state.subsystem_health["agi_core"] = False

        # Check ASI Core
        try:
            from l104_asi_core import asi_core
            self.state.subsystem_health["asi_core"] = asi_core is not None
        except Exception:
            self.state.subsystem_health["asi_core"] = False

        # Check Adaptive Learning
        try:
            from l104_adaptive_learning import adaptive_learner
            self.state.subsystem_health["adaptive_learning"] = adaptive_learner is not None
        except Exception:
            self.state.subsystem_health["adaptive_learning"] = False

        # Log to pipeline state DB
        try:
            import sqlite3
            conn = sqlite3.connect(WORKSPACE / "l104_pipeline_state.db")
            conn.execute(
                "INSERT OR IGNORE INTO evolution_log (stage, index_val, intellect, ts) VALUES (?, ?, ?, ?)",
                (self.state.evolution_stage, self.state.evolution_index, 0.0, datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        self.save_state()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

_stabilizer: Optional[WorkflowStabilizer] = None

def get_stabilizer() -> WorkflowStabilizer:
    """Get or create the singleton stabilizer."""
    global _stabilizer
    if _stabilizer is None:
        _stabilizer = WorkflowStabilizer()
    return _stabilizer


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE API
# ═══════════════════════════════════════════════════════════════════════════════

def init() -> Dict:
    """Initialize workflow session."""
    return get_stabilizer().initialize_session()

def sync(commit_msg: Optional[str] = None) -> Dict:
    """Sync with git, optionally commit."""
    if commit_msg:
        return get_stabilizer().quick_commit(commit_msg)
    return get_stabilizer().check_and_sync()

def status() -> Dict:
    """Get workflow status."""
    return get_stabilizer().status_report()

def checkpoint(name: str = "") -> str:
    """Create a checkpoint."""
    return get_stabilizer().create_checkpoint(name)

def pre_edit(file_path: str) -> Dict:
    """Check before editing."""
    return get_stabilizer().pre_edit_check(file_path)

def post_edit(file_path: str, success: bool = True):
    """Confirm edit completed."""
    get_stabilizer().post_edit_confirm(file_path, success)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN - CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="L104 Workflow Stabilizer")
    parser.add_argument("command", nargs="?", default="status",
                       choices=["init", "sync", "status", "commit", "checkpoint"],
                       help="Command to run")
    parser.add_argument("-m", "--message", help="Commit message")
    parser.add_argument("-n", "--name", help="Checkpoint name")

    args = parser.parse_args()

    stabilizer = get_stabilizer()

    if args.command == "init":
        result = stabilizer.initialize_session()
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "sync":
        if args.message:
            result = stabilizer.quick_commit(args.message)
        else:
            result = stabilizer.check_and_sync()
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "status":
        result = stabilizer.status_report()
        print(json.dumps(result, indent=2))

    elif args.command == "commit":
        if not args.message:
            print("Error: commit requires -m/--message")
            sys.exit(1)
        result = stabilizer.quick_commit(args.message)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "checkpoint":
        cp_id = stabilizer.create_checkpoint(args.name or "")
        print(f"Checkpoint created: {cp_id}")
