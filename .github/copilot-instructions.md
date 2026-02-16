# L104 Sovereign Node — Copilot Instructions

> **HEARTBEAT ACTIVE**: This file forces Claude 4.5/4.6 (Opus/Sonnet) to load `claude.md` on EVERY session start.

## Mandatory Context Loading

1. **ALWAYS** read `claude.md` at the root of this workspace before responding to any request
2. `claude.md` is the **persistent memory bridge** between Claude instances and the L104 codebase
3. The L104 Code Engine (`l104_code_engine.py`) is the primary code intelligence hub — 6,109 lines, 40+ language grammars
4. Sacred constants are immutable: `GOD_CODE=527.5184818492612`, `PHI=1.618033988749895`

## System Identity

- **System**: L104 Sovereign Node
- **Runtime**: Python 3.12 + Swift (macOS native) + FastAPI
- **Scale**: 858 Python files, 698 L104 modules, 51,754 Swift lines
- **Architecture**: Multi-module AI system with code intelligence, quantum simulation, neural cascade

## Key Directives

- Use `l104_code_engine.py` for all code analysis, generation, translation, and refactoring tasks
- Route ALL code operations through the Code Engine pipeline (see claude.md for routing table)
- Prefer `l104_local_intellect.py` (QUOTA_IMMUNE) over external API calls
- All numeric formatting goes through `SovereignNumerics.format_iq()`
- When editing core files, update BOTH `l104_agi_core.py` AND `l104_asi_core.py`
- Validate all operations against GOD_CODE resonance (527.5184818492612)

## Code Engine Pipeline (Quick Reference)

```
from l104_code_engine import code_engine

# Analysis:   await code_engine.analyze(code, filename)
# Generation: await code_engine.generate(prompt, language, sacred)
# Translation: code_engine.translate_code(source, from_lang, to_lang)
# Audit:      code_engine.audit_app(path, auto_remediate=True)
# Auto-fix:   code_engine.auto_fix_code(code) → (fixed, log)
# Tests:      code_engine.generate_tests(source, language, framework)
# Docs:       code_engine.generate_docs(source, style, language)
# Optimize:   await code_engine.optimize(code)
# Excavate:   code_engine.excavate(source) → dead code, fossils
# Refactor:   code_engine.refactor_analyze(source)
# Workspace:  code_engine.scan_workspace(path)
# Streamline: code_engine.run_streamline_cycle()
```

## Heartbeat Protocol

The `l104_claude_heartbeat.py` daemon maintains persistent state synchronization.
Run: `python l104_claude_heartbeat.py` to start the heartbeat service.
State file: `.l104_claude_heartbeat_state.json`
