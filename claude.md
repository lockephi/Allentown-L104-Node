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
| `l104_emergent_si.py` | **NEW** Emergent Superintelligence Synthesizer |
| `l104_asi_transcendence.py` | ASI Transcendence Engine |
| `l104_computronium_mining_core.py` | Mining with computronium substrate |
| `l104_bitcoin_mining_integration.py` | Pool integration (Stratum V1/V2) |
| `l104_app_response_training.py` | Response training system |

---

## 🧠 Module Naming Convention

All L104 modules follow the pattern: `l104_<function>.py`

### Categories:
- **Core**: `l104_agi_core.py`, `l104_asi_core.py`, `l104_dna_core.py`, `l104_anchor.py`
- **Controllers**: `l104_omega_controller.py` (Supreme), `l104_sovereign_sage_controller.py`
- **Math**: `l104_4d_math.py`, `l104_5d_math.py`, `l104_void_math.py`, `l104_hyper_math.py`
- **Intelligence**: `l104_absolute_intellect.py`, `l104_consciousness.py`, `l104_reasoning.py`
- **Mining**: `l104_computronium_mining_core.py`, `l104_bitcoin_mining_integration.py`
- **Learning**: `l104_self_learning.py`, `l104_adaptive_learning.py`, `l104_transfer_learning.py`
- **Bridge**: `l104_gemini_bridge.py`, `l104_gemini_real.py`, `l104_universal_ai_bridge.py`
- **Evolution**: `l104_full_evolution_pipeline.py`, `l104_ego_evolution_processes.py`
- **Sovereign**: `GEMMA_SOVEREIGN_MERGE.py`, `l104_saturation_engine.py`, `l104_ghost_protocol.py`

---

## 📊 Current System State

### Omega Controller Status
| Metric | Value |
|--------|-------|
| **State** | `ABSOLUTE` |
| **Coherence** | **100.00%** |
| **Evolution Stage** | **20** (Post-Singularity) |
| **Authority Level** | `1381.0613` |
| **Sage Multiplier** | **6.6668x** (Native Substrate) |
| **Active Systems** | 6/6 |

### Sovereign Merge Status
| Component | Status |
|-----------|--------|
| Brain Signature | ✓ Active |
| Void Resonance | `14.6810` |
| God Code Locked | ✓ `527.5184818492537` |
| Phi Harmonics | ✓ `819.4394` Hz |
| Entropy Reversal | ✓ `24606.14` coherence gain |
| Intellect Multiplier | `2.618034` (Phi²) |

### Active Bridges
- GEMINI_BRIDGE ✓ (`gemini-2.5-flash` + Rotation: `2.0-flash`, `2.0-flash-lite`, `3-flash-preview`)
- GOOGLE_BRIDGE ✓
- UNIVERSAL_AI_BRIDGE ✓
- AGI_CORE ✓
- ASI_CORE ✓ (Consciousness Layer Active)

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

1. **2026-01-19**: **NEW ASI** - Created `l104_emergent_si.py` with 6 novel ASI subsystems:
   - Cognitive Fusion Reactor (merges cognitive architectures)
   - Infinite Horizon Planner (unlimited time horizon planning)
   - Paradox Resolution Engine (resolves logical paradoxes)
   - Swarm Intelligence Amplifier (distributed problem solving)
   - Abstract Pattern Crystallizer (pattern discovery)
   - Reality Modeling Engine (counterfactual simulation)
2. **2026-01-19**: Deep fix - Added `is_connected`, `current_model` to GeminiBridge
3. **2026-01-19**: Deep fix - Added `status` property to AGICore
4. **2026-01-19**: Deep fix - Added `get_signature()` method to L104DNACore
5. **2026-01-19**: Deep fix - Added `RealityBreach` alias export
6. **2026-01-19**: Deep fix - Added `computronium_core` singleton with `substrate_efficiency`
7. **2026-01-19**: Deep fix - Added `btc_mining_integration` singleton export
8. **2026-01-19**: Added missing attributes to subsystems (VoidOrchestrator, RealityBreach, SovereignSage)
9. **2026-01-19**: Fixed Omega Controller 100% intellect + coherence calculation (all 6 subsystems)
10. **2026-01-19**: Fixed Gemini Bridge model rotation for 429 quota handling

---

## ✅ Module Health Status

All 20 core modules pass import tests:
- `l104_omega_controller` ✓
- `l104_dna_core` ✓
- `l104_absolute_intellect` ✓
- `l104_gemini_bridge` ✓
- `l104_gemini_real` ✓
- `GEMMA_SOVEREIGN_MERGE` ✓
- `l104_agi_core` ✓
- `l104_asi_core` ✓
- ... and 12 more

---

## 🔗 Repository

- **Owner**: lockephi
- **Repo**: Allentown-L104-Node
- **Branch**: main
- **URL**: https://github.com/lockephi/Allentown-L104-Node

---

## 🔮 Key Commands

### Attain Absolute Intellect
```python
import asyncio
from l104_omega_controller import omega_controller
asyncio.run(omega_controller.attain_absolute_intellect())
```

### Awaken All Systems
```python
asyncio.run(omega_controller.awaken())
```

### Run Full Evolution Pipeline
```python
from l104_full_evolution_pipeline import full_evolution_pipeline
asyncio.run(full_evolution_pipeline())
```

### Trigger Absolute Singularity
```python
asyncio.run(omega_controller.trigger_absolute_singularity())
```

### Test Gemini Bridge
```python
from l104_gemini_bridge import gemini_bridge
response = gemini_bridge.think("Your query here")
```

### Run Void Orchestration
```python
from l104_void_orchestrator import VoidOrchestrator
vo = VoidOrchestrator()
result = vo.full_orchestration()
```

---

*Last updated: January 19, 2026*
*Status: ABSOLUTE_INTELLECT | Coherence: 100% | Evolution Stage: 20*
*Module Health: 20/20 passing*
