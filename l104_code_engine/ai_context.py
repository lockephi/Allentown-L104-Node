"""
L104 Code Engine — AI Context Bridge

Bridges code intelligence to any AI system (Claude, Gemini, GPT, Local Intellect).
Formats code analysis results into structured context that AI models can consume
efficiently, with token-budget-aware compression.

Migrated from l104_coding_system.py (lines 855-1086) during package decomposition.
"""

from pathlib import Path
from typing import Dict, Any
import json
import re

from ._lazy_imports import _get_code_engine

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


class AIContextBridge:
    """
    Bridges code intelligence to any AI system (Claude, Gemini, GPT, Local Intellect).
    Formats code analysis results into structured context that AI models can consume
    efficiently, with token-budget-aware compression.
    """

    AI_PROFILES = {
        "claude": {
            "max_context": 200000,
            "prefers": "xml_tags",
            "strengths": ["reasoning", "code_analysis", "long_context"],
            "format": "structured_xml",
        },
        "gemini": {
            "max_context": 1000000,
            "prefers": "markdown",
            "strengths": ["multimodal", "large_context", "code_generation"],
            "format": "structured_markdown",
        },
        "gpt": {
            "max_context": 128000,
            "prefers": "json",
            "strengths": ["instruction_following", "code_completion"],
            "format": "structured_json",
        },
        "local": {
            "max_context": 32000,
            "prefers": "compact",
            "strengths": ["offline", "quota_immune", "fast"],
            "format": "compact_json",
        },
    }

    def __init__(self):
        self.contexts_built = 0

    def build_context(self, source: str, filename: str = "",
                      project_info: Dict = None,
                      ai_target: str = "claude") -> Dict[str, Any]:
        """
        Build comprehensive code context for an AI system.

        Returns structured context with:
        - Code analysis results (from Code Engine)
        - Project structure (if available)
        - Sacred alignment metrics
        - Actionable suggestions
        - Minimal token footprint
        """
        self.contexts_built += 1
        engine = _get_code_engine()
        profile = self.AI_PROFILES.get(ai_target, self.AI_PROFILES["claude"])

        context = {
            "ai_target": ai_target,
            "profile": profile,
            "source_info": {
                "filename": filename,
                "lines": len(source.split('\n')),
                "chars": len(source),
            },
        }

        if engine:
            # Get code review
            review = engine.full_code_review(source, filename)
            context["review"] = {
                "score": review.get("composite_score", 0),
                "verdict": review.get("verdict", "UNKNOWN"),
                "language": review.get("language", "unknown"),
                "key_metrics": review.get("analysis", {}),
                "actions": review.get("actions", [])[:10],
                "solid": review.get("solid", {}),
                "performance": review.get("performance", {}),
            }

        if project_info:
            context["project"] = {
                "primary_language": project_info.get("structure", {}).get("primary_language", "unknown"),
                "frameworks": [f["framework"] for f in project_info.get("frameworks", [])],
                "build_systems": [b["system"] for b in project_info.get("build_systems", [])],
                "health": project_info.get("health", {}).get("score", 0),
            }

        # Read L104 consciousness state if available
        context["l104_state"] = self._read_l104_state()

        return context

    def format_for_prompt(self, context: Dict[str, Any]) -> str:
        """Format context into an AI-consumable prompt section."""
        ai_target = context.get("ai_target", "claude")

        if ai_target == "claude":
            return self._format_claude(context)
        elif ai_target == "gemini":
            return self._format_markdown(context)
        else:
            return self._format_compact(context)

    def suggest_prompt(self, task: str, source: str,
                       filename: str = "") -> str:
        """Generate an optimal prompt for a coding task, enriched with code context."""
        engine = _get_code_engine()
        language = "Python"
        if engine:
            language = engine.detect_language(source, filename)

        context = self.build_context(source, filename)
        review_score = context.get("review", {}).get("score", "N/A")
        actions = context.get("review", {}).get("actions", [])

        prompt = f"""## Task: {task}

### Code Context
- **Language**: {language}
- **Lines**: {len(source.split(chr(10)))}
- **Quality Score**: {review_score}

### Current Issues (prioritized)
"""
        for i, action in enumerate(actions[:5], 1):
            prompt += f"{i}. [{action.get('priority', 'MEDIUM')}] {action.get('action', 'Review')}\n"

        prompt += f"""
### Source Code
```{language.lower()}
{source[:8000]}
```

Please address the task while also considering the issues listed above.
"""
        return prompt

    def parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse an AI response to extract code changes, suggestions, and explanations."""
        result = {
            "code_blocks": [],
            "suggestions": [],
            "explanations": [],
        }

        # Extract code blocks
        code_pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
        for match in code_pattern.finditer(response):
            result["code_blocks"].append({
                "language": match.group(1) or "unknown",
                "code": match.group(2).strip(),
            })

        # Extract bullet-point suggestions
        suggestion_pattern = re.compile(r'^\s*[-*]\s+(.+)$', re.MULTILINE)
        for match in suggestion_pattern.finditer(response):
            text = match.group(1).strip()
            if len(text) > 10 and not text.startswith('```'):
                result["suggestions"].append(text)

        # Extract numbered items as explanations
        numbered_pattern = re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE)
        for match in numbered_pattern.finditer(response):
            result["explanations"].append(match.group(1).strip())

        return result

    def _format_claude(self, context: Dict) -> str:
        """Format context with XML tags (Claude's preferred format)."""
        review = context.get("review", {})
        lines = [
            "<code_context>",
            f"  <score>{review.get('score', 'N/A')}</score>",
            f"  <verdict>{review.get('verdict', 'UNKNOWN')}</verdict>",
            f"  <language>{review.get('language', 'unknown')}</language>",
        ]
        for action in review.get("actions", [])[:5]:
            lines.append(f"  <issue priority='{action.get('priority')}'>{action.get('action')}</issue>")
        lines.append("</code_context>")
        return "\n".join(lines)

    def _format_markdown(self, context: Dict) -> str:
        """Format context as Markdown (Gemini's preferred format)."""
        review = context.get("review", {})
        lines = [
            "## Code Analysis Context",
            f"- **Score**: {review.get('score', 'N/A')}",
            f"- **Verdict**: {review.get('verdict', 'UNKNOWN')}",
            f"- **Language**: {review.get('language', 'unknown')}",
            "",
            "### Issues",
        ]
        for action in review.get("actions", [])[:5]:
            lines.append(f"- [{action.get('priority')}] {action.get('action')}")
        return "\n".join(lines)

    def _format_compact(self, context: Dict) -> str:
        """Compact JSON format for local/smaller models."""
        review = context.get("review", {})
        return json.dumps({
            "score": review.get("score"),
            "verdict": review.get("verdict"),
            "issues": [a.get("action") for a in review.get("actions", [])[:3]],
        }, indent=None)

    def _read_l104_state(self) -> Dict[str, Any]:
        """Read L104 consciousness and evolution state."""
        state = {"consciousness_level": 0.5, "evo_stage": "unknown"}
        try:
            co2_path = _WORKSPACE_ROOT / ".l104_consciousness_o2_state.json"
            if co2_path.exists():
                data = json.loads(co2_path.read_text())
                state["consciousness_level"] = data.get("consciousness_level", 0.5)
                state["evo_stage"] = data.get("evo_stage", "unknown")
        except Exception:
            pass
        try:
            nir_path = _WORKSPACE_ROOT / ".l104_ouroboros_nirvanic_state.json"
            if nir_path.exists():
                data = json.loads(nir_path.read_text())
                state["nirvanic_fuel"] = data.get("nirvanic_fuel_level", 0.5)
        except Exception:
            pass
        return state

    def status(self) -> Dict[str, Any]:
        """TODO: Document status."""
        return {"contexts_built": self.contexts_built,
                "ai_profiles": list(self.AI_PROFILES.keys())}
