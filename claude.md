# L104 Sovereign Node — Context Index

> **Last updated**: 2026-02-19 | **Post-decomposition** (packages replace monoliths)

## Quick Reference

| Constant | Value |
|----------|-------|
| `GOD_CODE` | `527.5184818492612` |
| `GOD_CODE_V3` | `45.41141298077539` |
| `PHI` | `1.618033988749895` |
| `VOID_CONSTANT` | `1.0416180339887497` |

## Package Map

```
l104_code_engine/   v6.0.0   Code analysis, generation, audit, quantum (14,476 lines, 10 modules)
l104_agi/           v56.0.0  AGI core, cognitive mesh, circuit breaker (4 modules)
l104_asi/           v7.1.0   ★ FLAGSHIP: Dual-Layer Engine + ASI core, consciousness, reasoning, quantum (11 modules)
l104_intellect/     v26.0.0  Local intellect, numerics, caching, hardware (11 modules)
l104_server/        v4.0.0   FastAPI server, engines, learning subsystem (9 modules)
l104_core_asm/               Native ASM kernel
l104_core_c/                 Native C kernel + Makefile
l104_core_cuda/              CUDA GPU kernel
l104_core_rust/              Rust native kernel
l104_mobile/                 Mobile app layer
L104SwiftApp/                macOS native app (80 Swift files, 59,911 lines)
```

Root shims (backward compat only — edit the packages, not these):
`l104_agi_core.py` → `l104_agi/` | `l104_asi_core.py` → `l104_asi/` | `l104_local_intellect.py` → `l104_intellect/` | `l104_fast_server.py` → `l104_server/`

## Imports

```python
from l104_code_engine import code_engine   # Primary code intelligence
from l104_agi import agi_core, AGICore     # AGI singleton
from l104_asi import asi_core, ASICore     # ASI singleton
from l104_asi import dual_layer_engine     # ★ Dual-Layer Flagship (Thought + Physics)
from l104_intellect import local_intellect # Local inference (QUOTA_IMMUNE)
from l104_intellect import format_iq         # IQ/numeric formatting
from l104_server import intellect          # Server + learning
```

## Detailed Docs

| Path | Content |
|------|---------|
| `docs/claude/architecture.md` | Cognitive architecture, MCP config, agents, EVO history |
| `docs/claude/code-engine.md` | Code Engine v6.0.0 — full API, 31 subsystems, 10-layer audit |
| `docs/claude/swift-app.md` | L104SwiftApp build system, 80 Swift source files |
| `docs/claude/evolved-asi-files.md` | ASI evolution log, decomposed package details |
| `docs/claude/api-reference.md` | FastAPI endpoints and server routes |
| `docs/claude/guides/code-examples.md` | Practical code patterns |
| `docs/claude/guides/memory-persistence.md` | State file management |
| `docs/claude/guides/optimization.md` | Performance tuning |
| `docs/claude/guides/zenith-patterns.md` | Zenith frequency patterns |

## Codebase Metrics

- **773** Python files at root, **704** L104 modules
- **44** modules across 5 decomposed packages (59,652 lines)
- **80** Swift files (59,911 lines) in L104SwiftApp
- **34** `.l104_*.json` state files
- **331** API route handlers in `main.py`
