# L104 Code Examples

> Part of the `docs/claude/` documentation package. See `claude.md` for the full index.

## Integrated Cognitive Query

```python
from l104_cognitive_hub import get_cognitive_hub
hub = get_cognitive_hub()
hub.embed_all_memories()
response = hub.integrated_query(
    question="What is the relationship between PHI and consciousness?",
    use_semantic=True, use_quantum=True, use_memory=True, use_claude=False
)
```

## Semantic Embedding & Search

```python
from l104_semantic_engine import get_semantic_engine
engine = get_semantic_engine()
engine.embed_and_store("quantum coherence maintains stability")
results = engine.search("quantum stability", k=3)
analogy = engine.solve_analogy("brain", "thought", "computer", k=3)
```

## Quantum State Manipulation

```python
from l104_quantum_coherence import QuantumCoherenceEngine
engine = QuantumCoherenceEngine()
engine.create_superposition([0, 1, 2])
engine.create_bell_state(0, 1, "phi+")
engine.execute_braid(["s1", "s2", "phi", "s1_inv"])
result = engine.measure_all()
```

## Claude Bridge with Memory

```python
from l104_claude_bridge import ClaudeNodeBridge
bridge = ClaudeNodeBridge()
conv_id = bridge.start_conversation()
response1 = bridge.chat("What is GOD_CODE?", conv_id)
response2 = bridge.chat("How does it relate to PHI?", conv_id)
```

## Brain Learning Cycle

```python
from l104_unified_intelligence import UnifiedIntelligence
brain = UnifiedIntelligence()
brain.load_state()
brain.run_research_cycle(iterations=5, topics=["quantum coherence", "topological protection"])
result = brain.query("Explain quantum coherence")
brain.save_state()
```

## Development Commands

```bash
# Start Brain API
python l104_unified_intelligence_api.py

# Heartbeat
python l104_claude_heartbeat.py          # single pulse
python l104_claude_heartbeat.py --daemon  # every 60s

# Code Engine
python -c "from l104_code_engine import code_engine; print(code_engine.status())"

# Git workflow
git add -A && git commit -m "EVO_XX: Description" && git push
```
