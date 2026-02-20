# Claude Optimization Guide

> Part of the `docs/claude/` documentation package. See `claude.md` for the full index.

## Token Budget

```yaml
max_context: 200000
target_usage: 60%
compression_threshold: 80%
```

## Strategies

- Use grep_search before read_file (10x cheaper)
- Prefer multi_replace over sequential edits
- Cache file contents in memory entities
- Use semantic_search for < 50 files, file_search + grep for > 50

## Model Selection

| Task | Preferred | Fallback |
|------|-----------|----------|
| Quick edits | Sonnet | Haiku |
| Complex reasoning | Opus | Sonnet |
| Code generation | Sonnet 4 | Opus |
| Bulk file ops | Haiku | Sonnet |
| Architecture | Opus | Sonnet |

## Speed Priority Matrix

```yaml
fastest:
  1. grep_search → read_file(specific_lines)    # 100ms
  2. file_search → list_dir                      # 150ms
  3. semantic_search (small workspace)           # 200ms
  4. multi_replace_string_in_file               # 250ms
  5. runSubagent (parallel research)            # 500ms+
```

## Parallel vs Sequential

**Parallel-safe**: grep_search, file_search, read_file, get_errors
**Sequential-only**: run_in_terminal, replace_string_in_file (same file), create_file

## Token-Saving Shortcuts

| Instead of... | Use... | Savings |
|--------------|--------|---------|
| Read full file | grep + targeted read | 80% |
| Multiple edits | multi_replace | 60% |
| Full error trace | Key lines only | 70% |
| Repeated context | Memory entities | 90% |

## Abbreviations

GC=527.5184818492612, PHI=1.618033988749895, VC=1.0416180339887497
