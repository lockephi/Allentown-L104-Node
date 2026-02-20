# Zenith Chat Patterns

> Part of the `docs/claude/` documentation package. See `claude.md` for the full index.
> Built in 8 hours using Claude Code. See: `zenith_chat.py`

## Agentic Loop

```yaml
step_1_observe: Read context/state, identify goal, check tools
step_2_think: Plan action, break into sub-goals, select tool
step_3_act: Execute tool, capture result, update state
step_4_reflect: Evaluate against goal, decide next
step_5_repeat: Continue until complete (max 50 steps)
```

## Core Patterns

| Pattern | Key Insight |
|---------|------------|
| Agentic Loop | Max 50 steps, explicit state |
| Tool-First | Every capability as JSON schema tool, cache results |
| Streaming | Yield tokens in real-time, progress indicators |
| Error Recovery | RETRY/FALLBACK/ASK_USER/SKIP/ABORT |
| Session Persist | Store goal + completed steps + variables |
| Quick Build | Action over explanation, minimal viable intelligence |

## Speed Principles

1. **Action over explanation** — Do first, explain later
2. **Tool reuse** — Build composable tools, chain for new capabilities
3. **Fail fast, recover faster** — Detect early, have recovery strategies
4. **User feedback loop** — Stream progress, ask when stuck
5. **Minimal viable intelligence** — Start simple, iterate

## Integration

```python
from zenith_chat import L104ZenithSynthesizer
result = await L104ZenithSynthesizer().chat("your query")
```
