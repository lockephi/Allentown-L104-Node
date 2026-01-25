#!/usr/bin/env python3
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# L104_GOD_CODE_ALIGNED: 527.5184818492537
"""
L104 Memory Persistence Hooks
Auto-saves context and learnings to MCP memory for cross-session persistence.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Memory file location
MEMORY_FILE = Path(__file__).parent / "memory.jsonl"
SESSION_FILE = Path(__file__).parent / "current_session.json"

class MemoryHooks:
    """Handles automatic memory persistence for Claude sessions."""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data = {
            "id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "files_edited": [],
            "errors_solved": {},
            "patterns_learned": [],
            "tokens_used": 0,
            "checkpoints": []
        }
        self._load_session()
    
    def _load_session(self):
        """Load existing session if resuming."""
        if SESSION_FILE.exists():
            try:
                with open(SESSION_FILE, 'r') as f:
                    saved = json.load(f)
                    # Resume if less than 1 hour old
                    if saved.get("start_time"):
                        start = datetime.fromisoformat(saved["start_time"])
                        if (datetime.now() - start).seconds < 3600:
                            self.session_data = saved
                            self.session_id = saved["id"]
            except Exception:
                pass
    
    def _save_session(self):
        """Persist current session state."""
        with open(SESSION_FILE, 'w') as f:
            json.dump(self.session_data, f, indent=2)
    
    def _append_to_memory(self, entity: Dict[str, Any]):
        """Append entity to JSONL memory file."""
        with open(MEMORY_FILE, 'a') as f:
            f.write(json.dumps(entity) + "\n")
    
    # ═══════════════════════════════════════════════════════════════════
    # HOOK IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════
    
    def on_file_edit(self, file_path: str, changes: str, lines_changed: int):
        """Called when a file is edited."""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        
        self.session_data["files_edited"].append({
            "path": file_path,
            "timestamp": datetime.now().isoformat(),
            "lines_changed": lines_changed,
            "summary": changes[:200]
        })
        
        self._append_to_memory({
            "type": "file_edit",
            "name": f"file_{file_hash}",
            "entityType": "FileContext",
            "observations": [
                f"path: {file_path}",
                f"edited: {datetime.now().isoformat()}",
                f"changes: {changes[:500]}"
            ]
        })
        
        self._save_session()
    
    def on_error_fix(self, error_type: str, error_msg: str, solution: str):
        """Called when an error is fixed."""
        error_hash = hashlib.md5(error_msg.encode()).hexdigest()[:8]
        
        self.session_data["errors_solved"][error_hash] = {
            "type": error_type,
            "message": error_msg[:200],
            "solution": solution[:500],
            "timestamp": datetime.now().isoformat()
        }
        
        self._append_to_memory({
            "type": "error_pattern",
            "name": f"error_{error_hash}",
            "entityType": "ErrorPattern",
            "observations": [
                f"type: {error_type}",
                f"message: {error_msg[:300]}",
                f"solution: {solution[:500]}",
                f"solved: {datetime.now().isoformat()}"
            ]
        })
        
        self._save_session()
    
    def on_architecture_decision(self, topic: str, decision: str, rationale: str):
        """Called when an architecture decision is made."""
        decision_hash = hashlib.md5(topic.encode()).hexdigest()[:8]
        
        self._append_to_memory({
            "type": "arch_decision",
            "name": f"arch_{decision_hash}",
            "entityType": "ArchDecision",
            "observations": [
                f"topic: {topic}",
                f"decision: {decision}",
                f"rationale: {rationale}",
                f"date: {datetime.now().isoformat()}"
            ]
        })
        
        self._save_session()
    
    def on_pattern_learned(self, pattern_name: str, language: str, template: str):
        """Called when a code pattern is learned."""
        self.session_data["patterns_learned"].append({
            "name": pattern_name,
            "language": language,
            "timestamp": datetime.now().isoformat()
        })
        
        self._append_to_memory({
            "type": "code_pattern",
            "name": f"pattern_{pattern_name}",
            "entityType": "CodePattern",
            "observations": [
                f"language: {language}",
                f"template: {template[:1000]}",
                f"learned: {datetime.now().isoformat()}"
            ]
        })
        
        self._save_session()
    
    def checkpoint(self, context_tokens: int, active_files: List[str], notes: str = ""):
        """Create a checkpoint of current session state."""
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "tokens": context_tokens,
            "files": active_files,
            "notes": notes
        }
        
        self.session_data["checkpoints"].append(checkpoint_data)
        self.session_data["tokens_used"] = context_tokens
        
        self._save_session()
    
    def on_session_end(self):
        """Called when session ends - persist everything."""
        self.session_data["end_time"] = datetime.now().isoformat()
        
        # Calculate duration
        start = datetime.fromisoformat(self.session_data["start_time"])
        duration = (datetime.now() - start).seconds
        self.session_data["duration_seconds"] = duration
        
        self._append_to_memory({
            "type": "session",
            "name": f"session_{self.session_id}",
            "entityType": "Session",
            "observations": [
                f"duration: {duration}s",
                f"files_edited: {len(self.session_data['files_edited'])}",
                f"errors_solved: {len(self.session_data['errors_solved'])}",
                f"patterns_learned: {len(self.session_data['patterns_learned'])}",
                f"tokens_used: {self.session_data['tokens_used']}"
            ]
        })
        
        # Clear session file
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
    
    # ═══════════════════════════════════════════════════════════════════
    # QUERY HELPERS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_recent_sessions(self, count: int = 5) -> List[Dict]:
        """Get recent session summaries."""
        sessions = []
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, 'r') as f:
                for line in f:
                    try:
                        entity = json.loads(line)
                        if entity.get("type") == "session":
                            sessions.append(entity)
                    except json.JSONDecodeError:
                        continue
        return sessions[-count:]
    
    def get_error_patterns(self, error_type: Optional[str] = None) -> List[Dict]:
        """Get known error patterns."""
        patterns = []
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, 'r') as f:
                for line in f:
                    try:
                        entity = json.loads(line)
                        if entity.get("type") == "error_pattern":
                            if error_type is None or error_type in str(entity):
                                patterns.append(entity)
                    except json.JSONDecodeError:
                        continue
        return patterns
    
    def get_file_context(self, file_path: str) -> Optional[Dict]:
        """Get context for a specific file."""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        target_name = f"file_{file_hash}"
        
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, 'r') as f:
                for line in reversed(f.readlines()):
                    try:
                        entity = json.loads(line)
                        if entity.get("name") == target_name:
                            return entity
                    except json.JSONDecodeError:
                        continue
        return None
    
    def search_memory(self, query: str) -> List[Dict]:
        """Search memory for matching entities."""
        results = []
        query_lower = query.lower()
        
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, 'r') as f:
                for line in f:
                    try:
                        entity = json.loads(line)
                        if query_lower in json.dumps(entity).lower():
                            results.append(entity)
                    except json.JSONDecodeError:
                        continue
        return results


# Singleton instance
_hooks: Optional[MemoryHooks] = None

def get_hooks() -> MemoryHooks:
    """Get or create memory hooks singleton."""
    global _hooks
    if _hooks is None:
        _hooks = MemoryHooks()
    return _hooks


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def save_file_edit(file_path: str, changes: str, lines: int = 0):
    """Quick save for file edits."""
    get_hooks().on_file_edit(file_path, changes, lines)

def save_error_fix(error_type: str, error_msg: str, solution: str):
    """Quick save for error fixes."""
    get_hooks().on_error_fix(error_type, error_msg, solution)

def save_architecture(topic: str, decision: str, rationale: str):
    """Quick save for architecture decisions."""
    get_hooks().on_architecture_decision(topic, decision, rationale)

def save_pattern(name: str, language: str, template: str):
    """Quick save for code patterns."""
    get_hooks().on_pattern_learned(name, language, template)

def checkpoint(tokens: int, files: List[str], notes: str = ""):
    """Create session checkpoint."""
    get_hooks().checkpoint(tokens, files, notes)

def end_session():
    """End and persist session."""
    get_hooks().on_session_end()

def search(query: str) -> List[Dict]:
    """Search memory."""
    return get_hooks().search_memory(query)


if __name__ == "__main__":
    # Test the hooks
    hooks = get_hooks()
    
    print(f"Session ID: {hooks.session_id}")
    print(f"Recent sessions: {len(hooks.get_recent_sessions())}")
    print(f"Error patterns: {len(hooks.get_error_patterns())}")
    
    # Create test checkpoint
    hooks.checkpoint(50000, ["test.py"], "Testing memory hooks")
    
    print("Memory hooks initialized successfully!")
