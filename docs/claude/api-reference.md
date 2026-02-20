# L104 API Reference

> Part of the `docs/claude/` documentation package. See `claude.md` for the full index.

## Brain API — Port 8082

```bash
python l104_unified_intelligence_api.py
```

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/brain/status | System status |
| GET | /api/brain/introspect | Self-reflection |
| POST | /api/brain/query | Ask questions |
| POST | /api/brain/learn | Trigger learning |
| POST | /api/brain/save | Persist state |
| POST | /api/brain/load | Restore state |

### Cognitive Hub (EVO_31)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/brain/hub/status | Hub status |
| POST | /api/brain/hub/embed-memories | Embed all memories |
| POST | /api/brain/hub/semantic-search | Semantic memory search |
| POST | /api/brain/hub/quantum-semantic | Quantum-semantic fusion |
| POST | /api/brain/hub/integrated-query | Integrated query (all systems) |
| GET | /api/brain/hub/coherence | Coherence report |

### Semantic Engine (EVO_30)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/brain/semantic/status | Engine status |
| POST | /api/brain/semantic/embed | Embed text |
| POST | /api/brain/semantic/embed/batch | Batch embed |
| POST | /api/brain/semantic/search | Similarity search |
| POST | /api/brain/semantic/similarity | Pairwise similarity |
| POST | /api/brain/semantic/analogy | Solve analogy (A:B::C:?) |
| POST | /api/brain/semantic/cluster | Cluster concepts |

### Quantum Coherence (EVO_29)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/brain/quantum/status | Quantum status |
| POST | /api/brain/quantum/superposition | Create superposition |
| POST | /api/brain/quantum/entangle | Entangle qubits |
| POST | /api/brain/quantum/braid | Topological braiding |
| POST | /api/brain/quantum/measure | Measure state |
| GET | /api/brain/quantum/god-code-phase | GOD_CODE phase alignment |

### Claude Bridge (EVO_28)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/brain/claude/query | Query Claude |
| POST | /api/brain/claude/conversation/start | Start conversation |
| POST | /api/brain/claude/chat | Chat with memory |
| GET | /api/brain/claude/tools | List tools |

### Advanced Reasoning

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/brain/deep-think | Deep think |
| POST | /api/brain/synthesize | Synthesize topics |
| POST | /api/brain/hypothesize | Generate hypotheses |
| POST | /api/brain/reason | Reasoning chain |

### Multi-Language Engines (EVO_32)

| Engine | Port | Status Endpoint |
|--------|------|----------------|
| TypeScript/Next.js | 3000 | /api/status |
| Go | 8080 | /stats |
| Rust | 8081 | /stats |
| Elixir OTP | 4000 | /stats |

### Agents & Orchestration

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/agents/architect/plan | Get architectural patterns |
| POST | /api/agents/planner/validate | Pre-execution validation |
| GET | /api/engines/status | Multi-language engine status |
| POST | /api/subagents/spawn | Spawn specialized agents |
