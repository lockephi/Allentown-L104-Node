# Prompt Engineering Best Practices (Anthropic Guidelines)

> Extracted from claude.md — Reference guide for prompt patterns.

## Chain-of-Thought

```xml
<thinking>
1. First, analyze the problem...
2. Consider edge cases...
3. Formulate solution...
</thinking>
<output>[Final structured response]</output>
```

## XML Tag Separation

Separate data from instructions using XML tags for clarity.

## Role Assignment

```yaml
roles:
  code_review: "Senior software architect with 20 years experience."
  security: "Cybersecurity expert specializing in smart contracts."
  optimization: "Performance engineer focused on latency reduction."
  documentation: "Technical writer creating clear API docs."
```

## Hallucination Prevention

- Always cite specific file paths and line numbers
- Use "I don't know" when uncertain
- Verify facts against codebase before stating
- Distinguish between inference and observation

## Prompt Caching

Use `cache_control: {"type": "ephemeral"}` for static context — 2x speedup, up to 90% cost reduction.

## Complex Task Decomposition

1. Break down the request into sub-tasks
2. Identify dependencies between sub-tasks
3. Execute sub-tasks in dependency order
4. Synthesize results into final output
5. Validate output against original request
