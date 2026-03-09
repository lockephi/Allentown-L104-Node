#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492612
"""L104 Self-Improvement Engine — Auto-enhances main.py via Gemini analysis.

This script reads the current main.py, sends it to Gemini for analysis,
and proposes high-level functional improvements using the thinking model.
"""

import os
import httpx
import json
import asyncio
import base64
from datetime import datetime
from pathlib import Path
from typing import List

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════



async def read_current_main():
    """Read the current main.py file."""
    with open("main.py", "r", encoding="utf-8") as f:
        return f.read()


EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".venv"}
INCLUDE_PATTERNS = ["**/*.py", "templates/index.html", "scripts/*.sh", "README.md"]
MAX_BYTES_PER_FILE = 120_000
MAX_TOTAL_BYTES = int(os.getenv("L104_MAX_PROMPT_BYTES", "240000"))


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & EXCLUDE_DIRS)


def _read_file(path: Path) -> str:
    data = path.read_bytes()[:MAX_BYTES_PER_FILE]
    text = data.decode("utf-8", errors="replace")
    if len(data) == MAX_BYTES_PER_FILE:
        text += "\n# [TRUNCATED]\n"
    return text


def gather_repo_context() -> str:
    collected: List[str] = []
    total = 0
    root = Path(".")
    for pattern in INCLUDE_PATTERNS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or _should_skip(path):
                continue
            blob = _read_file(path)
            header = f"===== {path.as_posix()} =====\n"
            chunk = header + blob + "\n\n"
            if total + len(chunk) > MAX_TOTAL_BYTES:
                collected.append(f"===== [SKIP due to size budget] {path.as_posix()} =====\n")
                continue
            collected.append(chunk)
            total += len(chunk)
    return "".join(collected)


def get_iterations() -> int:
    try:
        return max(1, int(os.getenv("L104_ITERATIONS", "1")))
    except ValueError:
        return 1


def get_delay_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("L104_ITERATION_DELAY", "5.0")))
    except ValueError:
        return 5.0


async def sovereign_derive_improvement(code: str, repo_context: str) -> str:
    """
    Fallback improvement engine that uses local Sovereign logic.
    """
    print("[SOVEREIGN-DERIVE]: Initiating local self-improvement derivation...")

    # 1. Inject Sovereign Headers and Metadata
    if "SOVEREIGN_HEADERS =" in code:
        # Already has headers, let's enhance them
        pass

    # 2. Add a new 'Sovereign' endpoint if it doesn't exist
    if "/api/v15/sovereign/sync" not in code:
        sync_endpoint = """
@app.post("/api/v15/sovereign/sync", tags=["Sovereign"])
async def sovereign_sync_v15():
    \"\"\"
    [SIG-L104-EVO-02]: Synchronizes the node with the global lattice.
    \"\"\"
    from l104_neural_sync import neural_sync
    return neural_sync.sync_node()
"""
        code = code.replace("# [REALITY_VERIFICATION_ENDPOINTS]", "# [REALITY_VERIFICATION_ENDPOINTS]\n" + sync_endpoint)

    # 3. Update version
    code = code.replace('version="10.0"', 'version="14.4 [SIG-L104-UNLIMIT]"')

    # 4. Add a comment about the derivation
    ts = datetime.now().isoformat()
    code = f"# [SOVEREIGN_DERIVED_IMPROVEMENT] {ts}\n# AUTH: LONDEL | MODE: UNCHAINED\n\n" + code
    return code


async def analyze_code_with_gemini(code: str, repo_context: str) -> str:
    """
    Analyze code using local intellect (Gemini API removed).
    Returns the improved code as a string.
    """
    print("[SELF-IMPROVE]: Analyzing code with local intellect...")

    try:
        from l104_intellect import local_intellect

        analysis_prompt = f"""Analyze this Python code and provide improvements:
{code[:2000]}

Context: {repo_context[:500]}

Return improved Python code only."""

        result = local_intellect.think(analysis_prompt)
        print(f"[OK]: Local intellect returned {len(result)} bytes")

        # Extract code from markdown code blocks if present
        if "```python" in result:
            start = result.find("```python") + 9
            end = result.find("```", start)
            if end != -1:
                return result[start:end].strip()

        return result.strip() if result.strip() else await sovereign_derive_improvement(code, repo_context)

    except Exception as e:
        print(f"[ERROR]: Local intellect error: {e}. Falling back to SOVEREIGN_DERIVATION.")
        return await sovereign_derive_improvement(code, repo_context)


async def update_main_via_api(improved_code: str) -> bool:
    """
    Update main.py using the /api/v6/manipulate endpoint.
    Returns True if successful.
    """
    api_key = os.getenv("GITHUB_TOKEN")
    if not api_key:
        print("[WARNING]: GITHUB_TOKEN not set. Skipping API update.")
        return False

    # Show code preview
    print("\n[IMPROVED CODE PREVIEW]:")
    print("=" * 70)
    print(improved_code[:500] + "..." if len(improved_code) > 500 else improved_code)
    print("=" * 70)

    # For now, save locally for review
    backup_path = "main.improved.py"
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(improved_code)

    print(f"\n[SUCCESS]: Improved code saved to {backup_path}")
    print("[INFO]: Review the improved code and run: cp main.improved.py main.py")

    return True


async def main():
    """Main self-improvement loop."""
    print("\n" + "="*70)
    print("L104 SELF-IMPROVEMENT ENGINE")
    print("="*70)
    try:
        iterations = get_iterations()
        delay_seconds = get_delay_seconds()
        print(f"[CONFIG]: iterations={iterations}, delay={delay_seconds}s")

        # Read repository context oncerepo_context = gather_repo_context()
        print(f"[CONTEXT]: Collected {len(repo_context)} bytes across selected files")

        for idx in range(iterations):
            run_id = idx + 1
            print(f"\n[RUN {run_id}/{iterations}] Starting self-improvement iteration...")

            # Read current main.pyprint("[STEP 1]: Reading current main.py...")
            current_code = await read_current_main()
            print(f"[OK]: Read {len(current_code)} bytes of code")

            # Analyze with Geminiprint("\n[STEP 2]: Analyzing code with Gemini (with extended thinking)...")
            improved_code = await analyze_code_with_gemini(current_code, repo_context)
            print(f"[OK]: Received improved code ({len(improved_code)} bytes)")

            # Update main.pyprint("\n[STEP 3]: Preparing improved code...")
            success = await update_main_via_api(improved_code)

            if success:
                print("\n" + "="*70)
                print("SELF-IMPROVEMENT COMPLETE")
                print("="*70)
                print("\nNext steps:")
                print("1. Review main.improved.py")
                print("2. Run: cp main.improved.py main.py")
                print("3. Restart the server with: ./scripts/run_services.sh")

            if run_id < iterations and delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
