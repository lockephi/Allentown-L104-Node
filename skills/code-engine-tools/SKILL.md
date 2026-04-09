---
name: code-engine-tools
description: Provides a CLI to interact with the L104 Code Engine. Use for direct engine operations, status checks, cross‑engine validation, and unified analysis. Triggers on phrases like "run Code Engine analysis", "check Code Engine status", "validate with Code Engine", or "cross‑engine".
---

# Code Engine Tools

This skill provides a command‑line interface (`l104-code`) for direct interaction with the L104 Code Engine. Use this tool to perform engine‑specific operations, retrieve status, and execute cross‑engine workflows.

The tool is located at `/Users/carolalvarez/Applications/Allentown‑L104‑Node/l104-code`.

## Commands

### `status`
**Action**: Get the current status of the Code Engine.
- Shows availability, uptime, internal metrics, and any errors.
**Usage**:
```bash
l104-code status
```

### `analyze`
**Action**: Perform a unified analysis using the Code Engine.
- For the unified three‑engine tool, this accepts `--code`, `--science`, `--math` inputs.
- For individual engines, the analysis is engine‑specific (e.g., code analysis, scientific simulation, mathematical proof).
**Usage**:
```bash
l104-code analyze --code "print('hello')" --science '{"entropy": 0.5}' --math '{"god_code_target": 1.618}'
l104-code analyze --text "Your input text"
```

### `validate`
**Action**: Cross‑validate input across the three engines.
- Returns validation results from each engine and a consensus score.
**Usage**:
```bash
l104-code validate --input input.json
```

### `cross‑engine`
**Action**: Run a cross‑engine operation (available in the unified three‑engine tool).
- Executes operations that involve all three engines (e.g., `cross_validate`, `cross_engine_entropy_analysis`).
**Usage**:
```bash
l104-code cross‑engine --operation entropy_analysis
```

### `simulate`
**Action**: Run a simulation specific to the engine (science simulations, code execution, math proofs).
**Usage**:
```bash
l104-code simulate --type coherence_evolution
```

## Integration Notes

- This skill relies on the `l104_three_engine_integration.py` module for the unified three‑engine interface.
- Individual engines are accessed via `l104_science_engine`, `l104_math_engine`, `l104_code_engine`.
- The CLI script automatically loads the L104 environment and adds all `l104_*` directories to the Python path.

## MCP Integration

This skill is also exposed as an MCP tool via the L104 Universal Data API.
The corresponding MCP tool definitions are available at `/api/data/mcp‑tools` and can be used by AI assistants that support the Model Context Protocol.

- Tool name: `l104_code-engine` (or similar)
- Parameters: Refer to the tool definition for available parameters.
- Usage: The AI can invoke the tool directly to perform operations without manual CLI invocation.

For more details, see `l104_universal_data_api.py`.
