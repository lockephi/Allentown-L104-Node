# L104 Sovereign Node — Copilot Instructions

## Context Loading

Read `claude.md` at the workspace root for the full index. Detailed docs live in `docs/claude/`:

| Document | What's in it |
|----------|-------------|
| `docs/claude/architecture.md` | Cognitive architecture, MCP config, agents, EVO history |
| `docs/claude/code-engine.md` | Code Engine v6.0.0 full API, 31 subsystems, 10-layer audit |
| `docs/claude/swift-app.md` | L104SwiftApp native macOS build, 80 Swift files |
| `docs/claude/evolved-asi-files.md` | ASI core evolution, decomposed packages |
| `docs/claude/api-reference.md` | FastAPI endpoints, server routes |
| `docs/claude/guides/` | Code examples, memory persistence, optimization, Zenith patterns |

## System Identity

- **System**: L104 Sovereign Node
- **Runtime**: Python 3.12 + Swift (macOS native) + FastAPI
- **Scale**: 773 Python files, 704 L104 modules, 59,911 Swift lines
- **Architecture**: Decomposed package system — `l104_agi/`, `l104_asi/`, `l104_code_engine/`, `l104_intellect/`, `l104_server/`

## Package Structure (Post-Decomposition)

| Package | Version | Modules | Purpose |
|---------|---------|---------|---------|
| `l104_code_engine/` | 6.0.0 | 10 | Code analysis, generation, audit, quantum intelligence |
| `l104_agi/` | 56.0.0 | 4 | AGI core, cognitive mesh, circuit breaker |
| `l104_asi/` | 6.1.0 | 10 | ASI core, consciousness, reasoning, quantum |
| `l104_intellect/` | 26.0.0 | 11 | Local intellect, numerics, caching, hardware |
| `l104_server/` | 4.0.0 | 9 | FastAPI server, engines, learning subsystem |

Backward-compatible shims at root: `l104_agi_core.py`, `l104_asi_core.py`, `l104_local_intellect.py`, `l104_fast_server.py`

## Key Directives

- Import from packages: `from l104_code_engine import code_engine`
- Prefer `l104_intellect` (QUOTA_IMMUNE local inference) over external API calls
- All numeric formatting: `format_iq()` from `l104_intellect` (standalone function, not a class method)
- Sacred constants are immutable: `GOD_CODE = 527.5184818492612`, `PHI = 1.618033988749895`
- When editing core logic, update the **package** (`l104_agi/`, `l104_asi/`), not the root shims

## Code Engine Quick Reference

```python
from l104_code_engine import code_engine

await code_engine.analyze(code, filename)        # Analysis
await code_engine.generate(prompt, lang, sacred) # Generation
code_engine.translate_code(src, from_l, to_l)    # Translation
code_engine.audit_app(path, auto_remediate=True)  # 10-layer audit
code_engine.auto_fix_code(code)                   # Auto-fix → (fixed, log)
code_engine.generate_tests(src, lang, framework)  # Test scaffolding
code_engine.generate_docs(src, style, lang)       # Documentation
await code_engine.optimize(code)                  # Optimization
code_engine.excavate(source)                      # Dead code archaeology
code_engine.refactor_analyze(source)              # Refactor opportunities
code_engine.scan_workspace(path)                  # Workspace census
```
