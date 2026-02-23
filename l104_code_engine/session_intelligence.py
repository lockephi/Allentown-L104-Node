"""
L104 Code Engine — Session Intelligence

Tracks coding sessions, learns patterns, and persists state.
Enables cross-session learning so the coding system gets better over time.

Migrated from l104_coding_system.py (lines 1837-2010) during package decomposition.
"""

import hashlib
import time
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


class SessionIntelligence:
    """
    Tracks coding sessions, learns patterns, and persists state.
    Enables cross-session learning so the coding system gets better over time.
    """

    SESSION_FILE = ".l104_coding_session.json"

    def __init__(self):
        self.current_session = None
        self.sessions: List[Dict] = []
        self.patterns_learned: Dict[str, int] = defaultdict(int)
        self._load_history()

    def start_session(self, description: str = "") -> str:
        """Start a new coding session. Returns session_id."""
        session_id = hashlib.sha256(
            f"{time.time()}{description}".encode()
        ).hexdigest()[:16]

        self.current_session = {
            "id": session_id,
            "start_time": datetime.now().isoformat(),
            "description": description,
            "actions": [],
            "files_touched": set(),
            "reviews_performed": 0,
            "suggestions_applied": 0,
            "quality_checks": 0,
        }
        return session_id

    def log_action(self, action_type: str, details: Dict = None) -> None:
        """Log an action in the current session."""
        if not self.current_session:
            self.start_session("auto")

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": action_type,
            "details": details or {},
        }
        self.current_session["actions"].append(entry)

        if "file" in (details or {}):
            self.current_session["files_touched"].add(details["file"])

        # Track patterns
        self.patterns_learned[action_type] += 1

        if action_type == "review":
            self.current_session["reviews_performed"] += 1
        elif action_type == "quality_check":
            self.current_session["quality_checks"] += 1

    def end_session(self) -> Dict[str, Any]:
        """End the current session and persist state."""
        if not self.current_session:
            return {"error": "No active session"}

        session = self.current_session
        session["end_time"] = datetime.now().isoformat()
        session["files_touched"] = list(session["files_touched"])
        session["total_actions"] = len(session["actions"])

        # Calculate session metrics
        start = datetime.fromisoformat(session["start_time"])
        end = datetime.fromisoformat(session["end_time"])
        session["duration_seconds"] = (end - start).total_seconds()

        self.sessions.append(session)
        self.current_session = None
        self._save_history()

        return {
            "session_id": session["id"],
            "duration": session["duration_seconds"],
            "actions": session["total_actions"],
            "files_touched": len(session["files_touched"]),
            "reviews": session["reviews_performed"],
        }

    def get_session_context(self) -> Dict[str, Any]:
        """Get current session context for AI consumption."""
        if not self.current_session:
            return {"active": False}

        return {
            "active": True,
            "session_id": self.current_session["id"],
            "actions_so_far": len(self.current_session["actions"]),
            "files_touched": list(self.current_session["files_touched"]),
            "recent_actions": self.current_session["actions"][-5:],
        }

    def learn_from_history(self) -> Dict[str, Any]:
        """Extract patterns from session history for self-improvement."""
        if not self.sessions:
            return {"patterns": {}, "insights": []}

        # Aggregate stats
        total_sessions = len(self.sessions)
        total_actions = sum(s.get("total_actions", 0) for s in self.sessions)
        most_common_actions = Counter()
        for s in self.sessions:
            for a in s.get("actions", []):
                most_common_actions[a.get("type", "unknown")] += 1

        # Most touched files
        file_freq = Counter()
        for s in self.sessions:
            for f in s.get("files_touched", []):
                file_freq[f] += 1

        # Average session duration
        durations = [s.get("duration_seconds", 0) for s in self.sessions if s.get("duration_seconds")]
        avg_duration = sum(durations) / max(1, len(durations))

        insights = []
        top_actions = most_common_actions.most_common(5)
        if top_actions:
            insights.append(f"Most common action: '{top_actions[0][0]}' ({top_actions[0][1]} times)")
        top_files = file_freq.most_common(3)
        if top_files:
            insights.append(f"Most edited file: '{top_files[0][0]}' ({top_files[0][1]} sessions)")

        return {
            "total_sessions": total_sessions,
            "total_actions": total_actions,
            "avg_session_duration": round(avg_duration, 1),
            "most_common_actions": dict(most_common_actions.most_common(10)),
            "most_touched_files": dict(file_freq.most_common(10)),
            "insights": insights,
            "patterns_learned": dict(self.patterns_learned),
        }

    def _save_history(self):
        """Persist session history to disk."""
        try:
            path = _WORKSPACE_ROOT / self.SESSION_FILE
            data = {
                "sessions": self.sessions[-50:],  # keep last 50 sessions
                "patterns": dict(self.patterns_learned),
                "last_updated": datetime.now().isoformat(),
            }
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def _load_history(self):
        """Load session history from disk."""
        try:
            path = _WORKSPACE_ROOT / self.SESSION_FILE
            if path.exists():
                data = json.loads(path.read_text())
                self.sessions = data.get("sessions", [])
                self.patterns_learned = defaultdict(int, data.get("patterns", {}))
        except Exception:
            pass

    def status(self) -> Dict[str, Any]:
        return {
            "active_session": self.current_session is not None,
            "total_sessions": len(self.sessions),
            "patterns_learned": len(self.patterns_learned),
        }
