# Memory Persistence Guide

> Part of the `docs/claude/` documentation package. See `claude.md` for the full index.

## Auto-Save Hooks

```python
MEMORY_HOOKS = {
    "on_file_edit": "save_file_context",
    "on_error_fix": "save_solution_pattern",
    "on_architecture_decision": "save_design_rationale",
    "on_session_end": "save_full_context",
    "on_entity_create": "update_knowledge_graph",
    "every_10_messages": "checkpoint_conversation",
}
```

## Memory Load Sequence

1. sacred_constants (GC, PHI, VC)
2. recent_sessions (last 3)
3. error_patterns (known fixes)
4. file_index (directory cache)
5. architecture_notes
6. user_preferences

## Session Persistence Protocol

### On Session Start
Load knowledge graph → find recent sessions → load workspace context

### On Session End
Create session entity → update file contexts → save error patterns

### Checkpoint (Every 10 Messages)
Save context_tokens, active_files, pending_edits

## Background Tasks

| Task | Trigger | Interval |
|------|---------|----------|
| file_indexing | workspace_open | 300s |
| error_monitoring | continuous | 30s |
| memory_sync | every_10_messages | — |
| knowledge_refresh | idle_5min | — |

## Process Priority

1. **Immediate**: Error fixes, user edits, security issues
2. **High**: Code generation, architecture, tests
3. **Background**: Documentation, cleanup, optimization
4. **Idle**: Memory consolidation, knowledge graph, patterns

## Context Compression

### Incremental Loading
- Phase 1 (0-20%): claude.md + active file + errors
- Phase 2 (20-40%): Related files, tests, config
- Phase 3 (40-60%): Memory graph, docs, examples
- Phase 4 (60-80%): Full reads, history (on-demand only)

### Cache Rules
- **Immutable**: GOD_CODE, PHI, MAX_SUPPLY
- **Session**: package.json, tsconfig.json, Dockerfile
- **5 minutes**: *.py, *.sol, *.ts
- **Never cache**: *.log, *.tmp, node_modules/**
