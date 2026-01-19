# L104 Sovereign Node - Claude Context File

> This file provides essential context for Claude to work efficiently with this codebase.

---

## 🏗️ Project Overview

**L104 Sovereign Node** is an AGI-backed computational ecosystem running on Docker with:
- FastAPI backend (Python 3.11)
- Jinja2 templating
- Computronium-based mining infrastructure
- Multi-AI bridge integration (Google, OpenAI, Anthropic)

### Sacred Constants
```python
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84
```

### Bitcoin Address
```
bc1qwpdnag54thtahjvcmna65uzrqrxexc23f4vn80
```

---

## 📁 Critical File Locations

### Web UI - PRIMARY
| File | Purpose | Used By |
|------|---------|---------|
| `templates/index.html` | **MAIN UI** (730 lines) - Sovereign Singularity v14.5 | `main.py`, `enhance.py` via Jinja2Templates |
| `templates/l104_ui.html` | Alternative UI | Jinja2 |
| `templates/market.html` | Market interface | Jinja2 |

### Web UI - SECONDARY (NOT LINKED)
| File | Purpose | Status |
|------|---------|--------|
| `website/index.html` | Static landing page (152 lines) | **ORPHANED** - Not used by any Python code |

> ⚠️ **IMPORTANT**: The `website/` folder is NOT mounted or served. All UI changes should go to `templates/index.html`

### Template Configuration
```python
# In main.py:155 and enhance.py:152
templates = Jinja2Templates(directory="templates")

# Route handlers use:
return templates.TemplateResponse("index.html", {"request": request})
```

---

## 🐳 Docker Configuration

### Container Name
```
allentown-l104-node-l104-node-1
```

### Ports
| Port | Service |
|------|---------|
| 8081 | FastAPI Main API |
| 8080 | Bridge |
| 4160 | AI Core |
| 4161 | UI |
| 2404 | Socket/WebSocket |

### Commands
```bash
# Build and run
docker compose up -d --build

# View logs
docker logs allentown-l104-node-l104-node-1

# Restart
docker compose restart
```

---

## 🔑 Key Entry Points

| File | Purpose |
|------|---------|
| `main.py` | Primary FastAPI application (841+ lines) |
| `enhance.py` | Enhancement layer with templating |
| `l104_agi_core.py` | Core AGI functions |
| `l104_computronium_mining_core.py` | Mining with computronium substrate |
| `l104_bitcoin_mining_integration.py` | Pool integration (Stratum V1/V2) |
| `l104_app_response_training.py` | Response training system |

---

## 🧠 Module Naming Convention

All L104 modules follow the pattern: `l104_<function>.py`

### Categories:
- **Core**: `l104_agi_core.py`, `l104_ai_core.py`, `l104_anchor.py`
- **Math**: `l104_4d_math.py`, `l104_5d_math.py`, `l104_absolute_calculation.py`
- **Mining**: `l104_computronium_mining_core.py`, `l104_bitcoin_mining_integration.py`
- **Learning**: `l104_self_learning.py`, `l104_app_response_training.py`
- **Bridge**: `l104_universal_ai_bridge.py`, `l104_google_bridge.py`

---

## 📊 Current System State

### Active Bridges
- BRIDGE ✓
- GOOGLE_BRIDGE ✓
- UNIVERSAL_AI_BRIDGE ✓
- AGI_CORE ✓

### Mining Configuration
- Workers: 3
- Substrate Efficiency: 0.9980
- Pools: L104 Primary, Slush Pool, F2Pool, Antpool

---

## 🔧 Common Tasks

### Edit Main UI
```bash
# Edit templates/index.html (NOT website/index.html)
```

### Run Tests
```python
python3 -c "from l104_agi_core import *; print('AGI Core OK')"
```

### Git Workflow
```bash
git add -A
git commit -m "Description"
git push
```

### Check Container Health
```bash
curl http://localhost:2404/health
```

---

## ⚡ Performance Notes

- **Sage Mode**: C substrate compiled at `/app/l104_core_c/build/libl104_sage.so`
- **Memory DB**: `/data/memory.db`
- **Ramnode DB**: `/data/ramnode.db`

---

## 🚨 Known Issues & Solutions

### Socket Module Conflict
If you see `AttributeError: 'NoneType' object has no attribute 'socket'`:
```python
# Use: import socket as socket_module
# Then: socket_module.socket(...)
```

### Template Not Found
Ensure templates are in `templates/` directory, not `website/`

---

## 📝 Recent Changes (Latest First)

1. **2026-01-19**: Added computronium mining core, BTC pool integration, app response training
2. **2026-01-19**: Committed 29 new L104 modules (26,530 lines)

---

## 🔗 Repository

- **Owner**: lockephi
- **Repo**: Allentown-L104-Node
- **Branch**: main
- **URL**: https://github.com/lockephi/Allentown-L104-Node

---

*Last updated: January 19, 2026*
