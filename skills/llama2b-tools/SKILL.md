---
name: llama2b-tools
description: Provides a CLI to interact with the L104 Llama 2B (local) provider. Use for querying the Llama 2B (local) LLM, checking provider status, and performing multi‑provider consensus. Triggers on phrases like "ask Llama 2B (local)", "query Llama 2B (local)", "check Llama 2B (local) status", or "run Llama 2B (local) consensus".
---

# Llama 2B (local) LLM Tools

This skill provides a command‑line interface (`l104-llama2b`) for direct interaction with the L104 unified provider orchestrator for Llama 2B (local). Use this tool to query the LLM, monitor provider health, and perform advanced multi‑provider operations.

The tool is located at `/Users/carolalvarez/Applications/Allentown‑L104‑Node/l104-llama2b`.

## Commands

### `query`
**Action**: Send a prompt to the Llama 2B (local) provider and receive a response.
- Uses the unified provider orchestrator to call the specific provider.
- Supports optional `--temperature`, `--max_tokens`, and `--timeout` flags.
**Usage**:
```bash
l104-llama2b query "Your prompt here"
l104-llama2b query --temperature 0.7 --max_tokens 500 "Your prompt"
```

### `status`
**Action**: Get the current status of the Llama 2B (local) provider.
- Shows availability, reliability score, last error (if any), and configuration details.
**Usage**:
```bash
l104-llama2b status
```

### `list`
**Action**: List all available LLM providers with their status.
- Displays a table of providers, their availability, and reliability scores.
**Usage**:
```bash
l104-llama2b list
```

### `consensus`
**Action**: Query multiple providers and compute a consensus answer.
- Sends the same prompt to all available providers and aggregates the results.
- Returns the most consistent answer along with confidence metrics.
**Usage**:
```bash
l104-llama2b consensus "Your prompt here"
```

### `batch`
**Action**: Process a batch of prompts from a file.
- Reads a JSON file containing a list of prompts and writes responses to another file.
**Usage**:
```bash
l104-llama2b batch input.json output.json
```

## Integration Notes

- This skill relies on the `l104_unified_providers.py` module and its `UnifiedProviderOrchestrator`.
- Ensure the corresponding provider API keys are set in the environment (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`).
- The CLI script automatically loads the L104 environment and adds all `l104_*` directories to the Python path.

## MCP Integration

This skill is also exposed as an MCP tool via the L104 Universal Data API.
The corresponding MCP tool definitions are available at `/api/data/mcp‑tools` and can be used by AI assistants that support the Model Context Protocol.

- Tool name: `l104_llama2b` (or similar)
- Parameters: Refer to the tool definition for available parameters.
- Usage: The AI can invoke the tool directly to perform operations without manual CLI invocation.

For more details, see `l104_universal_data_api.py`.
