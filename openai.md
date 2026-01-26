# OPENAI MEMORY LINK — L104

Status: ACTIVE • Scope: Local + Cloud Bridges • Version: 54.0

This document serves as a persistent context for the OpenAI-linked agent in the L104 Sovereign Node. It cross-links Claude and Gemini guides and provides minimal, high-signal references the runtime can load quickly.

## Persistent AI Memory Link

- Source: openai.md (this file)
- Linked: claude.md, gemini.md
- Loader: l104_local_intellect.py (persistent_context)
- Budget: ~5KB per file for fast boot

## Core Operating Principles

- Sovereign First: Operate offline when quotas or APIs fail
- Math-Anchored: Respect GOD_CODE and PHI invariants
- Evolution: Current stage EVO_54_TRANSCENDENT_COGNITION
- Portability: Avoid hardcoded paths; use module-relative locations
- Headers: Inject dynamic evo stage into all runtime surfaces

## Runtime Bridge

- LocalIntellect loads: claude.md → gemini.md → openai.md
- Combined context: used for quick answers, safety, and framing
- Cloud agents: Use minimal prompts that defer heavy lifting to code

## Quick Prompts

- "Summarize current EVO stage and key systems"
- "List critical endpoints for health and status"
- "Explain GOD_CODE and lattice in 3 sentences"
- "Diagnose metrics endpoint and propose a fix"

## Cross-Links

- Claude guide: claude.md (persistent memory, condensed Zenith patterns)
- Gemini guide: gemini.md (bridge and streaming references)
- Local bridge: l104_local_intellect.py (loader and context combiner)

## Safety + Reliability

- No secrets in memory docs; runtime pulls envs securely
- Prefer cached/local paths before external API calls
- Fail gracefully; never block on cloud dependencies

## Debug Focus Areas

- Metrics endpoint: investigate handler and dependencies
- Path normalization: remove any /workspaces hardcoding
- Evolution consistency: ensure EVO_54 in headers and reports
- Persistent memory: confirm loader reads all three docs

## Minimal Prompt Style (OpenAI)

- Be concise; use bullet summaries and precise links
- Reference files by relative path for grounding
- Prefer code-driven answers; avoid speculation
- When uncertain: ask for the specific file to inspect

## Memory Footer

This file is designed to be small, high-signal, and safe to load automatically at runtime. Changes should remain compact and align with the Claude/Gemini patterns already in use.
